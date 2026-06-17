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

import os
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
from core.utils import clamp_box, get_logger, iou

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
    recognized: bool = False          # a name is currently bound to this track
    recognized_at: float = 0.0        # timestamp of recognition (smart cache)
    identity_emb: Optional[np.ndarray] = None   # embedding that matched (for swap checks)
    verify_fail: int = 0              # consecutive failed re-validations
    last_verified_frame: int = 0      # frame index of last successful verification
    force_reverify: bool = False      # set on abrupt box change (possible track swap)
    appearance: Optional[np.ndarray] = None   # body colour histogram (swap detection)
    appearance_mismatch: int = 0      # consecutive frames the body looks different
    attempts: int = 0
    pending_match: Optional[dict] = None         # candidate identity not yet confirmed
    pending_emb: Optional[np.ndarray] = None     # embedding for pending candidate
    pending_confirm: int = 0                     # consecutive rounds agreeing on pending
    activity: str = "Standing"
    box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    draw_box: Optional[Tuple[float, float, float, float]] = None  # smoothed (EMA) box
    hits: int = 0                     # consecutive detections (for confirmation)
    confirmed: bool = False           # shown only after enough confirmations
    centers: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=3))
    activity_history: Deque[str] = field(default_factory=lambda: deque(maxlen=3))
    last_seen: float = 0.0
    last_seen_frame: int = 0          # global frame index of last detection
    gen: int = 0                      # identity-epoch token (bumped on reuse/invalidate)


class MonitoringPipeline:
    """Owns the camera + models and runs the inference loop on its own thread."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cam_manager = CameraManager(settings)

        # Cap OpenCV's internal thread pool so its many small ops (resize, colour
        # convert, histograms) don't spawn threads that contend with YOLO and the
        # recognition worker — this removes the CPU-contention stutter that shows up
        # right when a person is detected and recognition kicks in.
        try:
            cv2.setNumThreads(4)
        except Exception:  # noqa: BLE001
            pass

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

        # Async recognition worker — face detection/embedding runs on its OWN
        # thread so it never blocks/stutters the live stream.
        self._recog_thread: Optional[threading.Thread] = None
        self._recog_lock = threading.Lock()
        self._recog_event = threading.Event()
        self._recog_req = None            # (frame, boxes, gens)  latest only
        self._recog_out = None            # {tid: (match, emb, gen)}  latest results
        self._gen_counter = 0             # monotonic identity-epoch counter
        self._new_track_confirmed = False # signals immediate recognition for new tracks

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
        self.box_smooth_alpha = float(pcfg.get("box_smooth_alpha", 0.5))
        self.async_recognition = bool(pcfg.get("async_recognition", True))
        self.min_confirm_frames = max(1, int(self.settings.get("detection", "min_confirm_frames", 3)))
        self.min_det_score = float(self.settings.get("recognition", "min_det_score", 0.50))
        self.min_sharpness = float(self.settings.get("recognition", "min_sharpness", 18.0))
        self.jpeg_quality = int(pcfg.get("jpeg_quality", 70))
        self.draw_fps = bool(pcfg.get("draw_fps", True))
        self.recog_max_attempts = int(
            settings.get("recognition", "recognize_max_attempts", 3)
        )
        self.recognition_interval = max(1, int(
            settings.get("recognition", "recognition_interval", 12)
        ))
        self.recognition_cache_seconds = float(
            settings.get("recognition", "cache_seconds", 60)
        )
        self.revalidation_interval = max(1, int(
            settings.get("recognition", "revalidation_interval", 45)))
        self.max_validation_fails = max(1, int(
            settings.get("recognition", "max_validation_fails", 3)))
        self.reverify_gap_frames = max(2, int(
            settings.get("recognition", "reverify_gap_frames", 5)))
        self.appearance_corr_threshold = float(
            settings.get("recognition", "appearance_corr_threshold", 0.40))
        self.appearance_mismatch_frames = max(1, int(
            settings.get("recognition", "appearance_mismatch_frames", 4)))
        self.debug_recognition = bool(settings.get("recognition", "debug", False))

        # Adaptive low-latency controller (degrades work, never the stream).
        self.adaptive_enabled = bool(pcfg.get("adaptive_enabled", True))
        self.cpu_high = float(pcfg.get("cpu_high_threshold", 80))
        self.cpu_low = float(pcfg.get("cpu_low_threshold", 60))
        self._adapt_level = 0           # 0 = full quality, up to 2 = max throttle
        self._last_cpu_check = 0.0
        self._cpu_pct = 0.0

        # Runtime config hot-reload: pick up runtime.json edits without restart.
        self._last_config_check = 0.0
        try:
            self._runtime_mtime = os.path.getmtime(
                os.path.join(settings.base_dir, "data", "configs", "runtime.json"))
        except OSError:
            self._runtime_mtime = 0.0

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
        if self.async_recognition:
            self._recog_thread = threading.Thread(
                target=self._recognition_loop, name="recognition", daemon=True
            )
            self._recog_thread.start()
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
        self._recog_event.set()           # wake the recognition worker to exit
        if self._infer_thread:
            self._infer_thread.join(timeout=3.0)
        if self._recog_thread:
            self._recog_thread.join(timeout=3.0)
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
        self.box_smooth_alpha = float(pcfg.get("box_smooth_alpha", 0.5))
        self.async_recognition = bool(pcfg.get("async_recognition", True))
        self.min_confirm_frames = max(1, int(self.settings.get("detection", "min_confirm_frames", 3)))
        self.min_det_score = float(self.settings.get("recognition", "min_det_score", 0.50))
        self.min_sharpness = float(self.settings.get("recognition", "min_sharpness", 18.0))
        self.jpeg_quality = int(pcfg.get("jpeg_quality", 70))
        self.adaptive_enabled = bool(pcfg.get("adaptive_enabled", True))
        self.cpu_high = float(pcfg.get("cpu_high_threshold", 80))
        self.cpu_low = float(pcfg.get("cpu_low_threshold", 60))
        self.recognition_interval = max(1, int(rcfg.get("recognition_interval", 12)))
        self.recog_max_attempts = int(rcfg.get("recognize_max_attempts", 3))
        self.recognition_cache_seconds = float(rcfg.get("cache_seconds", 60))
        self.recognizer.min_face_size = int(rcfg.get("min_face_size", 28))
        self.revalidation_interval = max(1, int(rcfg.get("revalidation_interval", 45)))
        self.max_validation_fails = max(1, int(rcfg.get("max_validation_fails", 3)))
        self.reverify_gap_frames = max(2, int(rcfg.get("reverify_gap_frames", 5)))
        self.appearance_corr_threshold = float(rcfg.get("appearance_corr_threshold", 0.40))
        self.appearance_mismatch_frames = max(1, int(rcfg.get("appearance_mismatch_frames", 4)))
        self.debug_recognition = bool(rcfg.get("debug", False))
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
                time.sleep(0.01)   # short retry — frame is usually available next tick
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
        """Raise/lower the throttle level based on CPU load (checked ~1.5s).
        Also hot-reloads runtime.json every 30 s so config edits apply live.
        """
        now = time.time()

        # Hot-reload runtime.json when the file changes (no restart needed).
        if now - self._last_config_check >= 30.0:
            self._last_config_check = now
            try:
                mtime = os.path.getmtime(
                    os.path.join(self.settings.base_dir, "data", "configs", "runtime.json"))
                if mtime != self._runtime_mtime:
                    self._runtime_mtime = mtime
                    self.settings.reload()
                    self._reload_runtime_params()
                    logger.info("runtime.json changed — params hot-reloaded")
            except OSError:
                pass

        if not self.adaptive_enabled:
            return
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

    def _eff_recognition_interval(self) -> int:
        """Person-count-aware: recognise less often in crowds (lean on tracking).

        Identity stays stable via track persistence, so spacing out the scans with
        many people keeps the recognition worker (and CPU) comfortable.
        """
        n = sum(1 for s in self._tracks.values() if s.confirmed)
        if n >= 7:
            return self.recognition_interval * 2
        if n >= 4:
            return int(self.recognition_interval * 1.5)
        return self.recognition_interval

    def _process_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        run_detect = (self._frame_idx % self._eff_detect_every() == 0)

        if run_detect:
            detections = self.tracker.update(frame)
            self._sync_tracks(detections, frame, width, height)
            # Submit recognition on the regular interval OR immediately when a
            # new track just confirmed — skips the full interval wait so a new
            # person's face is checked within one detection cycle.
            force_recog, self._new_track_confirmed = self._new_track_confirmed, False
            if self._frame_idx % self._eff_recognition_interval() == 0 or force_recog:
                self._submit_recognition(frame)

        # Apply any recognition results produced by the worker thread.
        self._apply_recognition_results()

        # Enforce ONE active track per identity (no two people with the same name).
        self._resolve_duplicate_identities()

        # Remove stale tracks every frame so boxes vanish ~1s after a person leaves.
        self._cleanup_tracks()

        annotated = self._draw(frame)
        self._publish(annotated)
        self._update_stats()

    def _resolve_duplicate_identities(self) -> None:
        """Guarantee a name is on at most ONE active track.

        If two tracks end up with the same identity (e.g. a brief mis-association
        during an overlap), keep the ESTABLISHED holder (recognised earliest) and
        send the other back to Guest so it must re-verify — so two people can never
        show the same name at once.
        """
        by_name: Dict[str, list] = {}
        for st in self._tracks.values():
            if st.is_known and st.recognized and st.name:
                by_name.setdefault(st.name, []).append(st)
        for name, group in by_name.items():
            if len(group) < 2:
                continue
            group.sort(key=lambda s: s.recognized_at)   # earliest-recognised first
            for st in group[1:]:
                if self.debug_recognition:
                    logger.info("[ID] duplicate name '%s' on track %d -> Guest "
                                "(kept track %d)", name, st.track_id, group[0].track_id)
                self._invalidate_identity(st, "duplicate-name")

    # ---------------------------------------------------- recognition plumbing
    def _confirmed_boxes(self) -> Dict[int, Tuple[int, int, int, int]]:
        return {tid: st.box for tid, st in self._tracks.items() if st.confirmed}

    def _submit_recognition(self, frame: np.ndarray) -> None:
        boxes, gens = {}, {}
        for tid, st in self._tracks.items():
            if st.confirmed:
                boxes[tid] = st.box
                gens[tid] = st.gen
        if not boxes:
            return
        if self.async_recognition:
            with self._recog_lock:
                self._recog_req = (frame, boxes, gens)   # latest only — drop older
            self._recog_event.set()
        else:
            # Synchronous fallback (no worker thread).
            self._apply_outcomes(self._recognize_faces(frame, boxes, gens))

    def _apply_recognition_results(self) -> None:
        with self._recog_lock:
            out = self._recog_out
            self._recog_out = None
        if out:
            self._apply_outcomes(out)

    def _apply_outcomes(self, out: dict) -> None:
        for tid, (match, emb, gen) in out.items():
            st = self._tracks.get(tid)
            # Drop stale results: if the track's identity epoch changed since the
            # worker started (id reuse / invalidation / recreate), the result was
            # computed for a DIFFERENT person -> never apply it.
            if st is None or st.gen != gen:
                continue
            self._verify_track(st, match, emb)

    def _recognition_loop(self) -> None:
        """Dedicated thread: detect faces + embed + match, never blocks the stream."""
        while self._running:
            if not self._recog_event.wait(timeout=0.5):
                continue
            self._recog_event.clear()
            with self._recog_lock:
                req = self._recog_req
                self._recog_req = None
            if req is None:
                continue
            frame, boxes, gens = req
            try:
                out = self._recognize_faces(frame, boxes, gens)
            except Exception as exc:  # noqa: BLE001 - worker must survive
                logger.exception("Recognition worker error: %s", exc)
                continue
            with self._recog_lock:
                self._recog_out = out

    def _cleanup_tracks(self) -> None:
        """Drop any track not seen for > max_missed_frames frames (no ghost boxes).

        Removing the TrackState clears its name, recognition state and cached
        embedding — a leaving person can never transfer their identity to a newcomer.
        """
        stale = [tid for tid, st in self._tracks.items()
                 if self._frame_idx - st.last_seen_frame > self.max_missed_frames]
        for tid in stale:
            if self.debug_recognition:
                st = self._tracks[tid]
                logger.info("[ID] track LOST id=%d (was %s) -> identity cleared",
                            tid, st.name)
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
                state.gen = self._next_gen()
                self._tracks[tid] = state
                if self.debug_recognition:
                    logger.info("[ID] track CREATED id=%d -> Guest (awaiting verification)", tid)
            else:
                # Identity is NOT invalidated on gap/box-jump for the same person —
                # a brief occlusion keeps the identity through tracking persistence.
                # HOWEVER: ByteTrack can re-associate a DIFFERENT person to this
                # track_id while the original occupant is temporarily lost. We catch
                # that by running a single-frame body-colour check on re-entry: if
                # the clothing histogram differs from the stored baseline the identity
                # is cleared immediately (before the frame is drawn) rather than
                # waiting for 3 sustained mismatches with the wrong name visible.
                gap = self._frame_idx - state.last_seen_frame
                if state.recognized and gap > self.reverify_gap_frames:
                    if state.appearance is not None:
                        reentry_hist = self._body_hist(frame, box)
                        if reentry_hist is not None:
                            corr = cv2.compareHist(
                                state.appearance, reentry_hist,
                                cv2.HISTCMP_CORREL)
                            if corr < self.appearance_corr_threshold:
                                # Body looks different from the baseline —
                                # this is likely a different person reusing
                                # the track id. Clear identity now.
                                self._invalidate_identity(
                                    state, "re-entry-appearance-mismatch")
                                if self.debug_recognition:
                                    logger.info(
                                        "[ID] track id=%d re-entry corr=%.2f "
                                        "< %.2f -> identity cleared immediately",
                                        state.track_id, corr,
                                        self.appearance_corr_threshold)
                    if state.recognized:
                        # Same appearance (or no baseline yet): identity
                        # persists but a face re-check is scheduled.
                        state.force_reverify = True
            state.box = box
            state.last_seen = now
            state.last_seen_frame = self._frame_idx

            # Confirmation: only show a track after it has been seen a few times
            # (filters out random/transient false detections).
            state.hits += 1
            if not state.confirmed and state.hits >= self.min_confirm_frames:
                state.confirmed = True
                self._new_track_confirmed = True  # skip interval — recognise now

            # Smooth the drawn box (EMA) so it moves fluidly without jitter.
            a = self.box_smooth_alpha
            if state.draw_box is None:
                state.draw_box = tuple(float(v) for v in box)
            else:
                state.draw_box = tuple(a * n + (1 - a) * o
                                       for n, o in zip(box, state.draw_box))

            # Motion = normalised centroid displacement across recent frames.
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            box_h = max(1.0, box[3] - box[1])
            state.centers.append((cx, cy))
            motion = self._motion_speed(state.centers, box_h)

            # Activity: pose every pose_every_n frames for Sitting/Bending;
            # motion-only on all other frames for instant Walking/Idle.
            self._update_activity_smart(state, frame, box, motion)

            # Appearance guard: a recognised track whose body colours suddenly and
            # persistently change is a DIFFERENT person now — this catches a smooth
            # ByteTrack swap when two people cross/pass (no gap, no box jump, and
            # back-facing so the face can't help). -> invalidate -> re-verify.
            if state.recognized:
                self._check_appearance(state, frame, box)

        # Tracks not seen this update accumulate "missed" frames; _cleanup_tracks()
        # removes them once they exceed max_missed_frames (handled every frame).

    @staticmethod
    def _is_abrupt_change(old_box, new_box) -> bool:
        """Detect a sudden box jump/resize that suggests the track swapped person."""
        if old_box == (0, 0, 0, 0):
            return False
        if iou(old_box, new_box) < 0.2:
            return True
        ow = max(1, old_box[2] - old_box[0]); nw = max(1, new_box[2] - new_box[0])
        ratio = ow / nw if ow > nw else nw / ow
        return ratio > 1.8

    # ------------------------------------------------------ appearance guard
    @staticmethod
    def _body_hist(frame: np.ndarray, box) -> Optional[np.ndarray]:
        """Hue-Saturation histogram of the torso (clothing colours), lighting-robust."""
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw < 12 or bh < 24:
            return None
        # Central torso band — captures clothing, avoids background / legs.
        cx1, cx2 = x1 + int(0.20 * bw), x2 - int(0.20 * bw)
        cy1, cy2 = y1 + int(0.15 * bh), y1 + int(0.55 * bh)
        crop = frame[max(0, cy1):cy2, max(0, cx1):cx2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _check_appearance(self, state: TrackState, frame: np.ndarray, box) -> None:
        """Persistence-safe anti-transfer signal.

        Compares the track's current body colours to the baseline captured at
        recognition. A SUSTAINED drastic change (different clothing) means a
        different person reused this id -> invalidate. A matching body keeps the
        identity even with the face hidden (back/side/occluded). The baseline
        slowly adapts to the SAME person (turning, lighting) so it never
        false-triggers, but a sudden swap (low correlation) does not update it.
        """
        # When another confirmed person's box significantly overlaps this one their
        # clothing contaminates the torso histogram and would cause a false identity
        # reset.  Skip the check entirely while the overlap lasts; the mismatch
        # counter is NOT incremented so returning to normal appearance is seamless.
        for other in self._tracks.values():
            if other.track_id != state.track_id and other.confirmed \
                    and iou(box, other.box) > 0.15:
                return
        cur = self._body_hist(frame, box)
        if cur is None:
            return
        if state.appearance is None:
            state.appearance = cur                 # capture baseline for this identity
            return
        corr = cv2.compareHist(state.appearance, cur, cv2.HISTCMP_CORREL)
        if corr < self.appearance_corr_threshold:
            state.appearance_mismatch += 1
            if state.appearance_mismatch >= self.appearance_mismatch_frames:
                self._invalidate_identity(state, "appearance-change")
        else:
            state.appearance_mismatch = 0
            # Slowly adapt the baseline to the same person (handles turning/lighting).
            cv2.addWeighted(state.appearance, 0.92, cur, 0.08, 0, state.appearance)

    # --------------------------------------------- frame-level face recognition
    def _recognize_faces(self, frame: np.ndarray,
                         boxes: Dict[int, Tuple[int, int, int, int]],
                         gens: Dict[int, int]) -> dict:
        """Detect faces on the WHOLE frame, quality-gate them, assign each to ONE
        track, match embeddings. Returns ``{track_id: (match, emb, gen)}``.

        Runs on the recognition worker thread. Detecting on the full frame plus a
        unique one-face-per-track assignment prevents a neighbour's face leaking
        into another person's box (the multi-person identity-leakage bug).
        """
        if not boxes:
            return {}
        faces = []
        for f in self.recognizer.detect_faces(frame):
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            if not self._face_quality_ok(frame, f, (x1, y1, x2, y2)):
                continue
            faces.append({
                "box": (x1, y1, x2, y2),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "area": (x2 - x1) * (y2 - y1),
                "emb": np.asarray(f.normed_embedding, dtype=np.float32),
            })

        assignments = self._assign_faces_to_tracks(faces, boxes)
        out = {}
        for tid, face in assignments.items():
            # Carry the identity-epoch token so a delayed result is dropped if the
            # track has since changed person (id reuse / invalidation).
            out[tid] = (self.recognizer.match_embedding(face["emb"]), face["emb"], gens[tid])
        return out

    def _face_quality_ok(self, frame: np.ndarray, face, box) -> bool:
        """Quality gate — reject small / low-confidence / blurry faces.

        We would rather show *Guest* than risk a wrong identity from a poor face.
        """
        x1, y1, x2, y2 = box
        if min(x2 - x1, y2 - y1) < self.recognizer.min_face_size:
            return False
        if float(getattr(face, "det_score", 1.0)) < self.min_det_score:
            return False
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        return sharpness >= self.min_sharpness

    def _assign_faces_to_tracks(self, faces: list, boxes: dict) -> dict:
        """Greedily assign each detected face to at most ONE track (and vice versa).

        A face belongs to the track whose body box contains the face in its upper
        region; ties go to the box whose top is closest to the face (its head).
        """
        assignments: dict = {}
        claimed = set()
        for face in sorted(faces, key=lambda f: f["area"], reverse=True):
            cx, cy = face["center"]
            fw = face["box"][2] - face["box"][0]
            cands = []
            for tid, bbox in boxes.items():
                if tid in claimed:
                    continue
                bx1, by1, bx2, by2 = bbox
                bw, bh = bx2 - bx1, by2 - by1
                if bw <= 0 or bh <= 0:
                    continue
                bcx = (bx1 + bx2) / 2.0
                # Face must be inside the body box, in its upper ~60% (head region).
                # NOTE: no hard horizontal filter — a leaning / looking-down person's
                # face can be off-centre; alignment is only used to RANK candidates
                # and to reject genuinely ambiguous overlaps below.
                if not (bx1 <= cx <= bx2 and by1 <= cy <= by1 + 0.60 * bh):
                    continue
                if fw > bw * 1.2:                  # face wider than body -> implausible
                    continue
                hscore = abs(cx - bcx) / bw        # horizontal alignment (0 = centered)
                vscore = (cy - by1) / bh           # vertical (0 = at the top)
                cands.append((hscore + 0.5 * vscore, tid))
            if not cands:
                continue
            cands.sort()
            # Overlap safety: only when TWO bodies are almost equally plausible for
            # this face do we refuse to guess (could give one person the other's
            # name). A single clear body always gets the face.
            if len(cands) >= 2 and (cands[1][0] - cands[0][0]) < 0.10:
                continue
            best_tid = cands[0][1]
            claimed.add(best_tid)
            assignments[best_tid] = face
        return assignments

    def _verify_track(self, state: TrackState, match: dict, emb: np.ndarray) -> None:
        """Verify/assign identity for a track from a precomputed FAISS match."""
        known = match["is_known"]

        if not state.recognized:
            # Guest track: require the same person to match across N consecutive
            # recognition rounds before binding the identity.  One lucky frame
            # (face briefly inside the wrong box during an overlap or occlusion)
            # can NEVER transfer a name — the face must be persistently visible in
            # this track's box, not just transiently.
            if known:
                if (state.pending_match is not None
                        and match["name"] == state.pending_match["name"]):
                    state.pending_confirm += 1
                    state.pending_match = match
                    state.pending_emb = emb
                    if state.pending_confirm >= self.recog_max_attempts:
                        self._bind_identity(state, state.pending_match, state.pending_emb)
                        # _bind_identity clears pending state
                    elif self.debug_recognition:
                        logger.info("[ID] track id=%d pending %s %d/%d",
                                    state.track_id, match["name"],
                                    state.pending_confirm, self.recog_max_attempts)
                else:
                    # New name (or first match): start/restart the confirmation window.
                    state.pending_match = match
                    state.pending_emb = emb
                    state.pending_confirm = 1
                    if self.debug_recognition:
                        logger.info("[ID] track id=%d pending reset -> %s (1/%d)",
                                    state.track_id, match["name"], self.recog_max_attempts)
            else:
                # No face match this round: clear pending — the face is not
                # consistently visible in this box, so it was a transient overlap.
                if state.pending_match is not None:
                    state.pending_match = None
                    state.pending_emb = None
                    state.pending_confirm = 0
            return

        # Already named -> identity PERSISTS with the track. Re-validate only when
        # due (interval) or forced (abrupt box change). Crucially, a weak/hidden/
        # angled face never downgrades to Guest — tracking alone maintains identity.
        due = (self._frame_idx - state.last_verified_frame >= self.revalidation_interval) \
            or state.force_reverify
        if not due:
            return
        state.force_reverify = False

        if known and match["name"] == state.name:
            # Same person confirmed by a clear face -> refresh the binding.
            state.verify_fail = 0
            state.last_verified_frame = self._frame_idx
            state.identity_emb = emb
            return

        if not known:
            # Face visible but too weak/angled to match (e.g. side/partial view of
            # the SAME person). Keep the identity — do NOT count as a failure and
            # never drop to Guest. The track holds the name until it is lost.
            if self.debug_recognition:
                logger.info("[ID] track id=%d weak re-check -> keep %s",
                            state.track_id, state.name)
            return

        # A DIFFERENT enrolled person's face is confidently in this box -> likely a
        # ByteTrack swap. Switch identity only after repeated agreement (verified),
        # never instantly, so a single noisy frame can't move an identity.
        state.verify_fail += 1
        if self.debug_recognition:
            logger.info("[ID] track id=%d sees DIFFERENT %s (was %s) %d/%d",
                        state.track_id, match["name"], state.name,
                        state.verify_fail, self.max_validation_fails)
        if state.verify_fail >= self.max_validation_fails:
            self._bind_identity(state, match, emb)   # verified switch to the new person

    def _bind_identity(self, state: TrackState, match: dict, emb: np.ndarray) -> None:
        state.name = match["name"]
        state.department = match["department"]
        state.color = match["color"]
        state.is_known = True
        state.score = match["score"]
        state.recognized = True
        state.recognized_at = time.time()
        state.identity_emb = emb
        state.verify_fail = 0
        state.last_verified_frame = self._frame_idx
        # Clear any pending confirmation state.
        state.pending_match = None
        state.pending_emb = None
        state.pending_confirm = 0
        # Re-capture a fresh body-colour baseline for the newly bound person.
        state.appearance = None
        state.appearance_mismatch = 0
        if self.debug_recognition:
            logger.info("[ID] track id=%d -> %s [%s] conf=%.2f @frame %d",
                        state.track_id, match["name"], match["department"],
                        match["score"], self._frame_idx)

    def _invalidate_identity(self, state: TrackState, reason: str) -> None:
        """Clear a track's bound identity (track-id reuse / swap / re-entry).

        The newcomer must earn a name through fresh face verification — the old
        identity can never carry over to a different person.
        """
        if state.recognized and self.debug_recognition:
            logger.info("[ID] track id=%d identity INVALIDATED (%s): %s -> Guest, re-verify",
                        state.track_id, reason, state.name)
        state.name = self.recognizer.guest_label
        state.department = ""
        state.color = ""
        state.is_known = False
        state.recognized = False
        state.identity_emb = None
        state.verify_fail = 0
        state.force_reverify = True
        state.appearance = None
        state.appearance_mismatch = 0
        state.pending_match = None
        state.pending_emb = None
        state.pending_confirm = 0
        state.gen = self._next_gen()       # new identity epoch -> stale results dropped

    def _next_gen(self) -> int:
        self._gen_counter += 1
        return self._gen_counter

    def _downgrade_to_guest(self, state: TrackState) -> None:
        if self.debug_recognition:
            logger.info("[ID] track id=%d DOWNGRADED %s -> Guest (failed re-validation)",
                        state.track_id, state.name)
        state.name = self.recognizer.guest_label
        state.department = ""
        state.color = ""
        state.is_known = False
        state.recognized = False
        state.identity_emb = None
        state.verify_fail = 0
        state.pending_match = None
        state.pending_emb = None
        state.pending_confirm = 0

    def _motion_to_activity(self, motion: float) -> str:
        """Map normalised centroid speed to Walking / Standing / Idle."""
        if motion >= self.walk_thresh:
            return "Walking"
        if motion <= self.idle_thresh:
            return "Idle"
        return "Standing"

    def _update_activity(self, state: TrackState, motion: float) -> None:
        """Motion-only activity update (fast fallback)."""
        label = self._motion_to_activity(motion)
        state.activity_history.append(label)
        state.activity = Counter(state.activity_history).most_common(1)[0][0]

    def _update_activity_smart(self, state: TrackState, frame: np.ndarray,
                                box: tuple, motion: float) -> None:
        """Pose-aware activity — MediaPipe every pose_every_n frames for Sitting/
        Bending; motion-only on all other frames for instant Walking response.
        Smoothing window is 3 so labels change within 2 frames of an actual
        posture transition while still suppressing single-frame flicker.
        """
        run_pose = (self._frame_idx % self._eff_pose_every() == 0
                    and self.pose.enabled)
        if run_pose:
            x1, y1, x2, y2 = box
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            label = (self.pose.estimate(crop, motion)
                     if crop.size > 0
                     else self._motion_to_activity(motion))
        else:
            label = self._motion_to_activity(motion)
        state.activity_history.append(label)
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
            # Only draw confirmed tracks (no random/transient false boxes).
            if not state.confirmed:
                continue
            draw = state.draw_box or state.box     # smoothed coords for fluid motion
            x1, y1, x2, y2 = (int(round(v)) for v in draw)
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

        # Translucent label background — blend ONLY the small label rectangle, not
        # the whole frame. This keeps per-person cost tiny so many people stay smooth.
        ih, iw = img.shape[:2]
        rx1, ry1 = max(0, x1), max(0, top)
        rx2, ry2 = min(iw, x1 + box_w), min(ih, y1)
        if rx2 > rx1 and ry2 > ry1:
            roi = img[ry1:ry2, rx1:rx2]
            dark = np.empty_like(roi)
            dark[:] = (18, 18, 22)
            cv2.addWeighted(dark, 0.62, roi, 0.38, 0, roi)
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
            confirmed = [s for s in self._tracks.values() if s.confirmed]
            people = len(confirmed)
            recognized = sum(1 for s in confirmed if s.is_known)
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
        # Force re-recognition of currently tracked people (clear identity bindings).
        with self._lock:
            for st in self._tracks.values():
                st.recognized = False
                st.is_known = False
                st.name = self.recognizer.guest_label
                st.department = ""
                st.color = ""
                st.identity_emb = None
                st.verify_fail = 0
                st.attempts = 0
        logger.info("Recognition index reloaded; tracks reset for re-recognition")
