"""End-to-end monitoring pipeline with threaded capture / inference / display.

Flow per processed frame::

    RTSP/Webcam frame
      -> YOLOv8n person detection + ByteTrack tracking
      -> (per new track) face detect -> InsightFace embed -> FAISS search -> name
      -> MediaPipe pose -> activity label
      -> annotated frame + live stats

Recognition is *cached per track id* and only re-run when a track is lost and a
new id appears, satisfying the "recognise once per person" requirement.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Settings
from core.camera import CameraManager
from core.detection import Detection, PersonDetector
from core.pose import PoseEstimator
from core.recognition import FaceRecognizer
from core.tracking import PersonTracker
from core.utils import clamp_box, get_logger

logger = get_logger(__name__)


@dataclass
class TrackState:
    """Cached recognition + activity state for one tracked person."""

    track_id: int
    name: str = "Guest"
    department: str = ""
    color: str = ""                   # hex department colour (or guest colour)
    is_known: bool = False
    score: float = 0.0
    recognized: bool = False          # name locked (known or exhausted attempts)
    recognized_at: float = 0.0        # timestamp of recognition (smart cache)
    attempts: int = 0
    activity: str = "Standing"
    box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    centers: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=5))
    activity_history: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    last_seen: float = 0.0
    last_seen_frame: int = 0          # global frame index of last detection


class MonitoringPipeline:
    """Owns the camera + models and runs the inference loop on its own thread."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cam_manager = CameraManager(settings)

        # Models are loaded once and reused across restarts.
        logger.info("Initialising pipeline models…")
        self.detector = PersonDetector(settings)
        self.tracker = PersonTracker(settings, self.detector)
        self.recognizer = FaceRecognizer(settings)
        self.pose = PoseEstimator(settings)

        # Shared output state.
        self._lock = threading.Lock()
        self._output_jpeg: Optional[bytes] = None
        self._tracks: Dict[int, TrackState] = {}
        self._stats: Dict[str, object] = self._empty_stats()

        # Threading / lifecycle. Capture is owned by CameraManager — the pipeline
        # never touches cv2.VideoCapture directly.
        self._infer_thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_idx = 0
        self._fps = 0.0
        self._fps_ema = 0.0

        # Session-based camera selection. No camera is opened until the user picks
        # one from the Live Monitoring dialog. The choice is in-memory only.
        self._has_source = False
        self._active_cfg: Optional[Dict[str, object]] = None
        self._source_lost = False

        pcfg = settings.section("pipeline")
        self.stream_resolution = self._parse_res(pcfg.get("stream_resolution", "1280x720"))
        self.detect_every_n = max(1, int(pcfg.get("detect_every_n_frames", 2)))
        self.pose_every_n = max(1, int(pcfg.get("pose_every_n_frames", 10)))
        self.fps_limit = max(1, int(pcfg.get("fps_limit", 15)))
        self.max_missed_frames = int(pcfg.get("max_missed_frames", 10))
        self.walk_thresh = float(pcfg.get("walk_motion_threshold", 0.018))
        self.idle_thresh = float(pcfg.get("idle_motion_threshold", 0.005))
        self.activity_window = max(1, int(pcfg.get("activity_smooth_window", 5)))
        self.jpeg_quality = int(pcfg.get("jpeg_quality", 70))
        self.draw_fps = bool(pcfg.get("draw_fps", True))
        self.recog_max_attempts = int(
            settings.get("recognition", "recognize_max_attempts", 3)
        )
        self.recognition_interval = max(1, int(
            settings.get("recognition", "recognition_interval", 20)
        ))
        self.recognition_cache_seconds = float(
            settings.get("recognition", "cache_seconds", 60)
        )

        # Adaptive low-latency controller (degrades work, never the stream).
        self.adaptive_enabled = bool(pcfg.get("adaptive_enabled", True))
        self.cpu_high = float(pcfg.get("cpu_high_threshold", 80))
        self.cpu_low = float(pcfg.get("cpu_low_threshold", 60))
        self._adapt_level = 0           # 0 = full quality, up to 2 = max throttle
        self._last_cpu_check = 0.0
        self._cpu_pct = 0.0

        # How long to retain a track's cached identity after it disappears —
        # matched to the ByteTrack buffer so brief occlusions don't drop the name.
        track_buffer = int(settings.get("tracking", "track_buffer", 60))
        self._track_ttl = max(2.0, track_buffer / float(self.fps_limit))

    @staticmethod
    def _parse_res(value: str) -> Tuple[int, int]:
        try:
            w, h = str(value).lower().split("x")
            return int(w), int(h)
        except (ValueError, AttributeError):
            return 1280, 720

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Start the inference loop in *idle* mode — no camera is opened yet.

        Detection begins only once :meth:`select_source` is called from the
        camera-selection dialog.
        """
        if self._running:
            return
        self._running = True
        self.tracker.reset()
        self._infer_thread = threading.Thread(
            target=self._inference_loop, name="inference", daemon=True
        )
        self._infer_thread.start()
        logger.info("Pipeline started (idle — detection OFF until user presses Start)")

    # --------------------------------------------------- detection start/stop
    def start_detection(self) -> Dict[str, object]:
        """Start camera + detection/recognition/tracking using the active camera.

        Triggered by the dashboard "Start Detection" button — never automatic.
        """
        cfg = self.cam_manager.get_active_camera()
        if not cfg:
            return {"success": False,
                    "message": "No active camera. Select one in Camera Settings first."}
        if self._has_source:
            return {"success": True, "message": "Detection already running",
                    **self.cam_manager.describe()}
        logger.info("Start Detection requested")
        return self.select_source(self._with_stream_res(cfg), persist=False)

    def _with_stream_res(self, cfg: Dict[str, object]) -> Dict[str, object]:
        """Process every source at the configured stream resolution.

        For USB this requests the resolution from the device; for RTSP/IP the
        decoded frame is resized to it. A higher value (e.g. 1920x1080) keeps far
        faces larger and more recognisable, at some extra CPU cost.
        """
        cfg = dict(cfg)
        cfg["width"], cfg["height"] = self.stream_resolution
        cfg.setdefault("fps", 30)
        return cfg

    def stop_detection(self) -> Dict[str, object]:
        """Stop all processing and release the camera (CPU drops to near zero)."""
        logger.info("Stop Detection requested")
        self.stop_source()
        return {"success": True, "message": "Detection stopped"}

    @property
    def detection_running(self) -> bool:
        return self._has_source

    def select_source(self, cfg: Dict[str, object], persist: bool = True) -> Dict[str, object]:
        """Make ``cfg`` the active camera and begin detection.

        When ``persist`` is True the choice is written to ``camera_config.json``
        so the Live Monitoring page (and next boot) use it automatically.
        """
        logger.info("Selecting camera source: %s", cfg)
        with self._lock:
            self._tracks.clear()

        if persist:
            result = self.cam_manager.set_active_camera(cfg)   # opens + persists
        else:
            # Boot/auto path: open without re-persisting identical config.
            opened = self._open_only(cfg)
            result = ({"success": True, "message": "Camera Connected",
                       **self.cam_manager.describe()} if opened
                      else {"success": False, "message": "Camera Connection Failed"})

        if not result.get("success"):
            self._has_source = False
            return result

        self.tracker.reset()
        self._active_cfg = dict(cfg)
        self._has_source = True
        self._source_lost = False
        logger.info("Camera source active: %s", result.get("label"))
        return result

    def _open_only(self, cfg: Dict[str, object]) -> bool:
        try:
            self.cam_manager.open(cfg)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed opening camera: %s", exc)
            self.cam_manager.stop()
            return False
        if not self.cam_manager.wait_until_connected(timeout=5.0):
            self.cam_manager.stop()
            return False
        return True

    def stop_source(self) -> None:
        """Release the active camera and return the pipeline to idle."""
        self.cam_manager.stop()
        self._has_source = False
        self._source_lost = False
        self._active_cfg = None
        with self._lock:
            self._tracks.clear()
        logger.info("Camera source released — pipeline idle")

    def stop(self) -> None:
        self._running = False
        if self._infer_thread:
            self._infer_thread.join(timeout=3.0)
        self.cam_manager.stop()
        self._has_source = False
        with self._lock:
            self._tracks.clear()
            self._stats = self._empty_stats()
        logger.info("Pipeline stopped")

    def restart(self) -> None:
        """Reload settings and restart the loop, resuming detection if it was on.

        Used when System Settings change so new performance values take effect
        without the user having to press Start again.
        """
        logger.info("Restarting pipeline…")
        was_running = self._has_source
        self.stop()
        self.settings.reload()
        self._reload_runtime_params()
        time.sleep(0.3)
        self.start()
        if was_running:
            self.start_detection()

    def _reload_runtime_params(self) -> None:
        """Re-read tunable performance parameters after a settings change."""
        pcfg = self.settings.section("pipeline")
        dcfg = self.settings.section("detection")
        rcfg = self.settings.section("recognition")
        tcfg = self.settings.section("tracking")
        self.stream_resolution = self._parse_res(pcfg.get("stream_resolution", "1280x720"))
        self.detect_every_n = max(1, int(pcfg.get("detect_every_n_frames", 2)))
        self.pose_every_n = max(1, int(pcfg.get("pose_every_n_frames", 10)))
        self.fps_limit = max(1, int(pcfg.get("fps_limit", 15)))
        self.max_missed_frames = int(pcfg.get("max_missed_frames", 10))
        self.walk_thresh = float(pcfg.get("walk_motion_threshold", 0.018))
        self.idle_thresh = float(pcfg.get("idle_motion_threshold", 0.005))
        self.activity_window = max(1, int(pcfg.get("activity_smooth_window", 5)))
        self.jpeg_quality = int(pcfg.get("jpeg_quality", 70))
        self.adaptive_enabled = bool(pcfg.get("adaptive_enabled", True))
        self.cpu_high = float(pcfg.get("cpu_high_threshold", 80))
        self.cpu_low = float(pcfg.get("cpu_low_threshold", 60))
        self.recognition_interval = max(1, int(rcfg.get("recognition_interval", 20)))
        self.recog_max_attempts = int(rcfg.get("recognize_max_attempts", 3))
        self.recognition_cache_seconds = float(rcfg.get("cache_seconds", 60))
        self.recognizer.min_face_size = int(rcfg.get("min_face_size", 28))
        self._track_ttl = max(2.0, int(tcfg.get("track_buffer", 60)) / float(self.fps_limit))

        # Push live tunables onto the already-loaded models (no reload needed).
        imgsz = int(dcfg.get("imgsz", 640))
        conf = float(dcfg.get("confidence", 0.40))
        self.detector.imgsz = imgsz
        self.detector.confidence = conf
        self.tracker.imgsz = imgsz
        self.tracker.confidence = conf
        self.recognizer.threshold = float(rcfg.get("threshold", 0.45))

    @property
    def running(self) -> bool:
        return self._running

    @property
    def has_source(self) -> bool:
        return self._has_source

    def camera_state(self) -> str:
        """One of: ``no_source`` | ``connected`` | ``lost``."""
        if not self._has_source:
            return "no_source"
        return "connected" if self.cam_manager.connected else "lost"

    # ------------------------------------------------------------- outputs
    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._output_jpeg

    def get_stats(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._stats)

    def _empty_stats(self) -> Dict[str, object]:
        return {
            "people_count": 0,
            "recognized_count": 0,
            "guest_count": 0,
            "fps": 0.0,
            "camera_status": "no_source",
        }

    # ------------------------------------------------------- inference loop
    def _inference_loop(self) -> None:
        last_t = time.time()
        idle_published = False
        while self._running:
            # Detection OFF -> near-zero CPU: publish the "stopped" frame once,
            # then sleep long. No camera is open, no inference runs.
            if not self._has_source:
                if not idle_published:
                    self._publish_placeholder()
                    self._mark_status("no_source")
                    idle_published = True
                self._fps_ema = 0.0
                time.sleep(0.3)
                continue
            idle_published = False

            frame = self.cam_manager.read()
            if frame is None:
                state = self.camera_state()
                self._source_lost = (state == "lost")
                self._publish_placeholder()
                self._mark_status(state)
                time.sleep(0.05)
                continue

            self._source_lost = False
            self._frame_idx += 1
            self._update_adaptive()
            try:
                self._process_frame(frame)
            except Exception as exc:  # noqa: BLE001 - loop must survive
                logger.exception("Inference error: %s", exc)
                time.sleep(0.02)
                continue

            # FPS limiter — cap processing rate (major CPU saving on laptops and
            # avoids reprocessing identical frames faster than the camera delivers).
            now = time.time()
            dt = now - last_t
            min_period = 1.0 / float(self.fps_limit)
            if dt < min_period:
                time.sleep(min_period - dt)
                now = time.time()
                dt = now - last_t
            last_t = now
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema == 0 else 0.9 * self._fps_ema + 0.1 * inst

    # ----------------------------------------------- adaptive low-latency mode
    def _update_adaptive(self) -> None:
        """Raise/lower the throttle level based on CPU load (checked ~1.5s)."""
        if not self.adaptive_enabled:
            return
        now = time.time()
        if now - self._last_cpu_check < 1.5:
            return
        self._last_cpu_check = now
        try:
            import psutil
            self._cpu_pct = psutil.cpu_percent(interval=None)
        except Exception:  # noqa: BLE001
            return
        if self._cpu_pct >= self.cpu_high and self._adapt_level < 2:
            self._adapt_level += 1
            logger.info("CPU %.0f%% high -> low-latency level %d", self._cpu_pct, self._adapt_level)
        elif self._cpu_pct <= self.cpu_low and self._adapt_level > 0:
            self._adapt_level -= 1
            logger.info("CPU %.0f%% normal -> low-latency level %d", self._cpu_pct, self._adapt_level)

    def _eff_detect_every(self) -> int:
        return self.detect_every_n + self._adapt_level

    def _eff_pose_every(self) -> int:
        # Reduce posture updates before touching stream/tracking (per spec).
        return self.pose_every_n + self._adapt_level * 4

    def _eff_jpeg(self) -> int:
        return max(50, self.jpeg_quality - self._adapt_level * 10)

    def _process_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        run_detect = (self._frame_idx % self._eff_detect_every() == 0)

        if run_detect:
            detections = self.tracker.update(frame)
            self._sync_tracks(detections, frame, width, height)

        # Remove stale tracks every frame so boxes vanish ~1s after a person leaves.
        self._cleanup_tracks()

        annotated = self._draw(frame)
        self._publish(annotated)
        self._update_stats()

    def _cleanup_tracks(self) -> None:
        """Drop any track not seen for > max_missed_frames frames (no ghost boxes)."""
        stale = [tid for tid, st in self._tracks.items()
                 if self._frame_idx - st.last_seen_frame > self.max_missed_frames]
        for tid in stale:
            del self._tracks[tid]

    # -------------------------------------------------------- track syncing
    def _sync_tracks(
        self, detections: List[Detection], frame: np.ndarray, width: int, height: int
    ) -> None:
        now = time.time()
        seen_ids = set()

        for det in detections:
            tid = det.track_id
            if tid is None:
                continue
            seen_ids.add(tid)
            box = clamp_box(det.box, width, height)
            state = self._tracks.get(tid)
            if state is None:
                state = TrackState(track_id=tid)
                state.activity_history = deque(maxlen=self.activity_window)
                self._tracks[tid] = state
            state.box = box
            state.last_seen = now
            state.last_seen_frame = self._frame_idx

            # Motion = normalised centroid displacement across recent frames.
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            box_h = max(1.0, box[3] - box[1])
            state.centers.append((cx, cy))
            motion = self._motion_speed(state.centers, box_h)

            # Recognise once per track; retry every recognition_interval frames
            # until a confident known match (never every frame, never locks Guest).
            if not state.recognized and (
                state.attempts == 0 or self._frame_idx % self.recognition_interval == 0
            ):
                self._try_recognize(state, frame, box)

            # Activity from MOTION (not pose) -> Standing / Walking / Idle, smoothed
            # over a window so the displayed label stays stable (no flicker).
            self._update_activity(state, motion)

        # Tracks not seen this update accumulate "missed" frames; _cleanup_tracks()
        # removes them once they exceed max_missed_frames (handled every frame).

    def _try_recognize(self, state: TrackState, frame: np.ndarray, box) -> None:
        """Detect a face inside the person box and associate the recognised name."""
        # Search the upper ~55% of the body box where the face is expected.
        x1, y1, x2, y2 = box
        face_region_y2 = y1 + int((y2 - y1) * 0.55)
        crop = frame[y1:face_region_y2, x1:x2]
        if crop.size == 0:
            return

        matches = self.recognizer.recognize_faces(crop)
        state.attempts += 1
        if matches:
            best = max(matches, key=lambda m: m.score)
            if best.is_known:
                state.name = best.name
                state.department = best.department
                state.color = best.color
                state.is_known = True
                state.score = best.score
                state.recognized = True       # lock — known person (cached on track)
                state.recognized_at = time.time()
                logger.info("Track %d recognised as %s [%s] (%.2f) — cached for %.0fs",
                            state.track_id, best.name, best.department, best.score,
                            self.recognition_cache_seconds)
                return
        # Not recognised yet: show a stable "Guest" label but KEEP retrying at the
        # recognition interval — never lock as Guest. This is essential for far
        # PTZ/IP cameras: the name appears as soon as a clear face is seen when the
        # person approaches. Only a confident known match locks the track.
        state.name = self.recognizer.guest_label
        state.is_known = False
        state.department = ""
        state.color = ""

    def _update_activity(self, state: TrackState, motion: float) -> None:
        """Motion-driven activity (Standing / Walking / Idle) with majority smoothing.

        Walking is decided from sustained centroid movement — never from a single
        frame or pose landmarks — so sitting/partial-body people are not mislabelled.
        """
        if motion >= self.walk_thresh:
            raw = "Walking"
        elif motion <= self.idle_thresh:
            raw = "Idle"
        else:
            raw = "Standing"
        state.activity_history.append(raw)
        # Display the most frequent label over the recent window (stable, no flicker).
        state.activity = Counter(state.activity_history).most_common(1)[0][0]

    @staticmethod
    def _motion_speed(centers: Deque[Tuple[float, float]], box_h: float) -> float:
        if len(centers) < 2:
            return 0.0
        diffs = []
        pts = list(centers)
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            diffs.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
        return (sum(diffs) / len(diffs)) / box_h

    # --------------------------------------------------------------- drawing
    def _draw(self, frame: np.ndarray) -> np.ndarray:
        from core.utils import departments as dept

        out = frame.copy()
        # Scale all annotations to the frame size for a clean large-display look.
        scale = max(0.7, frame.shape[1] / 1280.0)
        box_th = max(2, int(round(3 * scale)))
        for state in self._tracks.values():
            x1, y1, x2, y2 = state.box
            if x2 <= x1 or y2 <= y1:
                continue
            # Department colour for known people; neutral colour for guests.
            hex_color = state.color or (dept.GUEST_COLOR if not state.is_known else "#22C55E")
            color = dept.hex_to_bgr(hex_color)

            # Full-body bounding box only — never a separate face box.
            cv2.rectangle(out, (x1, y1), (x2, y2), color, box_th)

            # Label lines: Name, then Department (known only), then Activity.
            name = state.name or self.recognizer.guest_label
            lines = [(name, 0.85, (255, 255, 255))]
            if state.is_known and state.department:
                lines.append((state.department, 0.6, color))
            if state.activity:
                lines.append((state.activity, 0.55, (185, 195, 210)))
            self._draw_labels(out, x1, y1, lines, color, scale)
        if self.draw_fps:
            self._draw_hud(out)
        return out

    def _draw_labels(self, img, x1, y1, lines, color, scale=1.0) -> None:
        """Render a stacked label box. ``lines`` = list of (text, base_scale, bgr)."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        pad = int(14 * scale)
        gap = int(8 * scale)
        sizes = []
        for text, bscale, _ in lines:
            th = max(1, int(round((2 if bscale >= 0.8 else 1.4) * scale)))
            (w, h), _ = cv2.getTextSize(text, font, bscale * scale, th)
            sizes.append((w, h, th))
        box_w = max(w for w, _, _ in sizes) + pad * 2
        box_h = sum(h for _, h, _ in sizes) + gap * (len(lines) - 1) + pad * 2
        top = max(0, y1 - box_h)
        accent = max(4, int(6 * scale))

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, top), (x1 + box_w, y1), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.62, img, 0.38, 0, img)
        cv2.rectangle(img, (x1, top), (x1 + accent, y1), color, -1)  # dept accent bar

        cy = top + pad
        for (text, bscale, tcolor), (w, h, th) in zip(lines, sizes):
            cy += h
            cv2.putText(img, text, (x1 + pad, cy), font, bscale * scale, tcolor, th, cv2.LINE_AA)
            cy += gap

    def _draw_hud(self, img) -> None:
        text = f"FPS: {self._fps_ema:0.1f}"
        cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (76, 217, 100), 1, cv2.LINE_AA)

    # ----------------------------------------------------------- publishing
    def _publish(self, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self._eff_jpeg()])
        if ok:
            with self._lock:
                self._output_jpeg = buf.tobytes()

    def _publish_placeholder(self) -> None:
        """A clean, dark branded frame. The dashboard draws its own HTML overlay
        on top, so this stays minimal (and ASCII-only to avoid glyph issues)."""
        w, h = self.cam_manager.resolution()
        # Subtle vertical gradient background.
        top = np.linspace(16, 26, h, dtype=np.uint8)
        img = np.repeat(top[:, None], w, axis=1)
        img = np.stack([img, img, np.clip(img + 6, 0, 255)], axis=2)
        text = "AI RECEPTION MONITORING"
        scale = max(0.6, w / 1400.0)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (70, 78, 92), 2, cv2.LINE_AA)
        self._publish(img)

    def _mark_status(self, status: str) -> None:
        with self._lock:
            self._stats = {
                "people_count": 0,
                "recognized_count": 0,
                "guest_count": 0,
                "fps": 0.0,
                "camera_status": status,
            }

    def _update_stats(self) -> None:
        status = self.camera_state()
        with self._lock:
            people = len(self._tracks)
            recognized = sum(1 for s in self._tracks.values() if s.is_known)
            guests = people - recognized
            self._stats = {
                "people_count": people,
                "recognized_count": recognized,
                "guest_count": guests,
                "fps": round(self._fps_ema, 1),
                "camera_status": status,
            }

    # ----------------------------------------------- recognition hot-reload
    def reload_recognition(self) -> None:
        """Refresh the FAISS index after person enrollment changes."""
        self.recognizer.refresh_index()
        # Force re-recognition of currently tracked people.
        with self._lock:
            for st in self._tracks.values():
                st.recognized = False
                st.attempts = 0
        logger.info("Recognition index reloaded; tracks reset for re-recognition")
