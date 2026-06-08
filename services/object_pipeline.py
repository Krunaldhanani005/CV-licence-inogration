"""Object Detection pipeline — YOLO + ByteTrack, no face recognition.

Mirrors the MonitoringPipeline public interface so the same API layer drives
both modes without modification.  Detects multiple COCO classes simultaneously
and returns per-class object counts on the dashboard.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import Settings
from core.camera import CameraManager
from core.utils import get_logger

logger = get_logger(__name__)

# --------------------------------------------------- COCO class catalogue
COCO_CLASSES: Dict[int, str] = {
    0: "Person",    1: "Bicycle",   2: "Car",       3: "Motorcycle",
    5: "Bus",       7: "Truck",
    14: "Bird",     15: "Cat",      16: "Dog",
    24: "Backpack", 25: "Umbrella", 26: "Handbag",  28: "Suitcase",
    39: "Bottle",   41: "Cup",
    56: "Chair",    57: "Couch",    60: "Table",
    62: "TV",       63: "Laptop",   64: "Mouse",
    66: "Keyboard", 67: "Cell Phone",
    73: "Book",     74: "Clock",
}

DEFAULT_ENABLED: List[int] = [0, 67, 63, 56, 39, 24, 26, 28, 2, 3, 5, 7, 1, 16, 15]

# Per-class confidence thresholds.
# Classes with many false positives (e.g. Chair) get a higher threshold;
# small hard-to-detect objects (Phone, Laptop) get a lower one.
DEFAULT_CLASS_THRESHOLDS: Dict[int, float] = {
    0:  0.40,   # Person
    1:  0.45,   # Bicycle
    2:  0.45,   # Car
    3:  0.45,   # Motorcycle
    5:  0.45,   # Bus
    7:  0.45,   # Truck
    14: 0.45,   # Bird
    15: 0.40,   # Cat
    16: 0.40,   # Dog
    24: 0.40,   # Backpack
    25: 0.45,   # Umbrella
    26: 0.40,   # Handbag
    28: 0.45,   # Suitcase
    39: 0.45,   # Bottle
    41: 0.45,   # Cup
    56: 0.60,   # Chair — prone to false positives; keep strict
    57: 0.55,   # Couch
    60: 0.50,   # Table
    62: 0.45,   # TV
    63: 0.35,   # Laptop — small on tables; needs lower threshold
    64: 0.40,   # Mouse
    66: 0.40,   # Keyboard
    67: 0.30,   # Cell Phone — very small; needs the lowest threshold
    73: 0.45,   # Book
    74: 0.45,   # Clock
}

_PALETTE = [
    (59, 130, 246), (34, 197, 94),  (245, 158, 11), (239, 68, 68),
    (139, 92, 246), (6,  182, 212), (236, 72,  153), (132, 204, 22),
    (249, 115,  22), (99, 102, 241), (20, 184, 166),  (251, 191, 36),
]

_OD_SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "configs", "object_settings.json",
)


def _bgr(cls_id: int) -> Tuple[int, int, int]:
    r, g, b = _PALETTE[cls_id % len(_PALETTE)]
    return b, g, r   # OpenCV BGR


def load_od_settings() -> dict:
    defaults: dict = {
        "detection": {
            "confidence": 0.45, "imgsz": 640, "fps_limit": 15,
            "track_buffer": 10, "min_object_size": 20, "jpeg_quality": 60,
        },
        "enabled_classes": list(DEFAULT_ENABLED),
        "class_thresholds": dict(DEFAULT_CLASS_THRESHOLDS),
    }
    if not os.path.exists(_OD_SETTINGS_PATH):
        return defaults
    try:
        with open(_OD_SETTINGS_PATH, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        result = dict(defaults)
        if isinstance(saved.get("detection"), dict):
            result["detection"] = {**defaults["detection"], **saved["detection"]}
        if isinstance(saved.get("enabled_classes"), list):
            result["enabled_classes"] = [int(c) for c in saved["enabled_classes"]]
        if isinstance(saved.get("class_thresholds"), dict):
            merged = dict(DEFAULT_CLASS_THRESHOLDS)
            merged.update({int(k): float(v) for k, v in saved["class_thresholds"].items()})
            result["class_thresholds"] = merged
        return result
    except Exception as exc:
        logger.warning("Could not read OD settings: %s", exc)
        return defaults


def save_od_settings(data: dict) -> None:
    os.makedirs(os.path.dirname(_OD_SETTINGS_PATH), exist_ok=True)
    with open(_OD_SETTINGS_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# ---------------------------------------------------------- track state
@dataclass
class ObjTrack:
    track_id: int
    class_id: int
    class_name: str
    box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    draw_box: Optional[Tuple[float, float, float, float]] = None
    hits: int = 0
    confirmed: bool = False
    last_seen_frame: int = 0


# ---------------------------------------------------------- pipeline
class ObjectDetectionPipeline:
    """Object Detection pipeline — shares public interface with MonitoringPipeline."""

    _MIN_CONFIRM = 2

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cam_manager = CameraManager(settings)
        try:
            cv2.setNumThreads(4)
        except Exception:
            pass

        logger.info("Loading OD pipeline YOLO model…")
        model_path = os.path.join(
            settings.base_dir,
            settings.get("detection", "model_path", "yolov8n.pt"))
        self._model = YOLO(model_path)
        logger.info("OD pipeline YOLO model loaded")

        trk = settings.get("tracking", "tracker", "bytetrack.yaml")
        local = os.path.join(settings.base_dir, "config", trk)
        self._tracker_cfg = local if os.path.exists(local) else trk

        self._lock = threading.Lock()
        self._output_jpeg: Optional[bytes] = None
        self._tracks: Dict[int, ObjTrack] = {}
        self._stats: dict = self._empty_stats()

        self._infer_thread: Optional[threading.Thread] = None
        self._running = False
        self._has_source = False
        self._source_lost = False
        self._active_cfg: Optional[dict] = None
        self._frame_idx = 0
        self._fps_ema = 0.0

        self._reload_params()

    def _reload_params(self) -> None:
        cfg = load_od_settings()
        det = cfg["detection"]
        self._confidence  = float(det.get("confidence", 0.45))
        self._imgsz       = int(det.get("imgsz", 640))
        self._fps_limit   = max(1, int(det.get("fps_limit", 15)))
        self._max_missed  = max(5, int(det.get("track_buffer", 10)))
        self._min_size    = int(det.get("min_object_size", 20))
        self._jpeg_q      = int(det.get("jpeg_quality", 60))
        raw = cfg.get("enabled_classes", DEFAULT_ENABLED)
        self._enabled: Optional[List[int]] = ([int(c) for c in raw] if raw else None)

        # Per-class confidence thresholds.
        self._class_thresholds: Dict[int, float] = dict(
            cfg.get("class_thresholds", DEFAULT_CLASS_THRESHOLDS))

        # YOLO runs at the minimum threshold across all enabled classes so it
        # doesn't silently drop low-threshold detections (e.g. Cell Phone at
        # 0.30).  We post-filter to each class's actual threshold below.
        enabled_set = set(self._enabled) if self._enabled else set(self._class_thresholds)
        relevant = [v for k, v in self._class_thresholds.items() if k in enabled_set]
        self._yolo_conf = min(relevant) if relevant else self._confidence

    @staticmethod
    def _parse_res(value: str) -> Tuple[int, int]:
        try:
            w, h = str(value).lower().split("x")
            return int(w), int(h)
        except (ValueError, AttributeError):
            return 1280, 720

    # --------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._infer_thread = threading.Thread(
            target=self._inference_loop, name="od-inference", daemon=True)
        self._infer_thread.start()
        logger.info("OD pipeline started (idle)")

    def stop(self) -> None:
        self._running = False
        if self._infer_thread:
            self._infer_thread.join(timeout=3.0)
        self.cam_manager.stop()
        self._has_source = False
        with self._lock:
            self._tracks.clear()
            self._stats = self._empty_stats()
        logger.info("OD pipeline stopped")

    def start_detection(self) -> dict:
        cfg = self.cam_manager.get_active_camera()
        if not cfg:
            return {"success": False,
                    "message": "No active camera. Select one in Camera Settings first."}
        if self._has_source:
            return {"success": True, "message": "Object Detection already running",
                    **self.cam_manager.describe()}
        res_str = self.settings.get("pipeline", "stream_resolution", "1280x720")
        w, h = self._parse_res(res_str)
        cfg = dict(cfg); cfg["width"] = w; cfg["height"] = h; cfg.setdefault("fps", 30)
        try:
            self.cam_manager.open(cfg)
        except Exception as exc:
            logger.error("OD pipeline: camera open failed: %s", exc)
            return {"success": False, "message": "Camera Connection Failed"}
        if not self.cam_manager.wait_until_connected(timeout=5.0):
            self.cam_manager.stop()
            return {"success": False, "message": "Camera Connection Failed"}
        self._reset_tracker()
        with self._lock:
            self._tracks.clear()
        self._active_cfg = dict(cfg)
        self._has_source = True
        self._source_lost = False
        logger.info("OD pipeline: detection started")
        return {"success": True, "message": "Object Detection started",
                **self.cam_manager.describe()}

    def stop_detection(self) -> dict:
        self.cam_manager.stop()
        self._has_source = False
        self._source_lost = False
        self._active_cfg = None
        with self._lock:
            self._tracks.clear()
        logger.info("OD pipeline: detection stopped")
        return {"success": True, "message": "Object Detection stopped"}

    def select_source(self, cfg: dict, persist: bool = True) -> dict:
        """Camera switch while detection is running (mirrors FR interface)."""
        if persist:
            result = self.cam_manager.set_active_camera(cfg)
        else:
            opened = self._open_only(cfg)
            result = ({"success": True, "message": "Camera Connected",
                       **self.cam_manager.describe()} if opened
                      else {"success": False, "message": "Camera Connection Failed"})
        if result.get("success"):
            self._reset_tracker()
            with self._lock:
                self._tracks.clear()
            self._active_cfg = dict(cfg)
            self._has_source = True
            self._source_lost = False
        return result

    def _open_only(self, cfg: dict) -> bool:
        try:
            self.cam_manager.open(cfg)
        except Exception as exc:
            logger.error("OD open_only failed: %s", exc)
            self.cam_manager.stop()
            return False
        if not self.cam_manager.wait_until_connected(timeout=5.0):
            self.cam_manager.stop()
            return False
        return True

    def _reset_tracker(self) -> None:
        try:
            if hasattr(self._model, "predictor") and self._model.predictor:
                trackers = getattr(self._model.predictor, "trackers", None)
                if trackers:
                    for t in trackers:
                        if hasattr(t, "reset"):
                            t.reset()
        except Exception:
            pass

    def reload_settings(self) -> None:
        """Hot-reload OD settings without restart."""
        self._reload_params()
        logger.info("OD pipeline: settings reloaded")

    def reload_recognition(self) -> None:
        """No-op — keeps API compatibility with FR pipeline."""

    def restart(self) -> None:
        was_running = self._has_source
        self.stop_detection()
        self.stop()
        self._reload_params()
        time.sleep(0.2)
        self.start()
        if was_running:
            self.start_detection()

    # ------------------------------------------------------------ outputs
    @property
    def running(self) -> bool:
        return self._running

    @property
    def detection_running(self) -> bool:
        return self._has_source

    @property
    def has_source(self) -> bool:
        return self._has_source

    def camera_state(self) -> str:
        if not self._has_source:
            return "no_source"
        return "connected" if self.cam_manager.connected else "lost"

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._output_jpeg

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    # ------------------------------------------------------- inference loop
    def _inference_loop(self) -> None:
        last_t = time.time()
        idle_published = False
        while self._running:
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
            try:
                self._process_frame(frame)
            except Exception as exc:
                logger.exception("OD inference error: %s", exc)
                time.sleep(0.02)
                continue

            now = time.time()
            dt = now - last_t
            min_period = 1.0 / float(self._fps_limit)
            if dt < min_period:
                time.sleep(min_period - dt)
                now = time.time()
                dt = now - last_t
            last_t = now
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = (inst if self._fps_ema == 0
                                 else 0.9 * self._fps_ema + 0.1 * inst)

    def _process_frame(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]

        results = self._model.track(
            frame,
            imgsz=self._imgsz,
            conf=self._yolo_conf,   # minimum across per-class thresholds
            iou=0.5,
            classes=self._enabled if self._enabled else None,
            device="cpu",
            tracker=self._tracker_cfg,
            persist=True,
            verbose=False,
        )

        seen_ids: set = set()
        if results and results[0].boxes is not None:
            boxes_data = results[0].boxes
            names = results[0].names or {}
            for i, box_xyxy in enumerate(boxes_data.xyxy):
                if boxes_data.id is None:
                    continue
                track_id = int(boxes_data.id[i].item())
                cls_id   = int(boxes_data.cls[i].item())

                # Per-class confidence gate — apply the class-specific threshold
                # AFTER YOLO so each class gets its own strictness level.
                score = (float(boxes_data.conf[i].item())
                         if boxes_data.conf is not None else 1.0)
                per_thresh = self._class_thresholds.get(cls_id, self._confidence)
                if score < per_thresh:
                    continue

                cls_name = COCO_CLASSES.get(cls_id, names.get(cls_id, f"Object{cls_id}"))
                x1, y1, x2, y2 = (int(v) for v in box_xyxy.tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if min(x2 - x1, y2 - y1) < self._min_size:
                    continue
                seen_ids.add(track_id)
                st = self._tracks.get(track_id)
                if st is None:
                    st = ObjTrack(track_id=track_id, class_id=cls_id,
                                  class_name=cls_name)
                    self._tracks[track_id] = st
                st.box = (x1, y1, x2, y2)
                st.last_seen_frame = self._frame_idx
                st.hits += 1
                if not st.confirmed and st.hits >= self._MIN_CONFIRM:
                    st.confirmed = True
                a = 0.85
                if st.draw_box is None:
                    st.draw_box = (float(x1), float(y1), float(x2), float(y2))
                else:
                    st.draw_box = tuple(a * n + (1 - a) * o
                                        for n, o in zip((x1, y1, x2, y2), st.draw_box))

        stale = [tid for tid, st in self._tracks.items()
                 if self._frame_idx - st.last_seen_frame > self._max_missed]
        for tid in stale:
            del self._tracks[tid]

        annotated = self._draw(frame)
        self._publish(annotated)
        self._update_stats()

    def _draw(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        scale   = max(0.7, frame.shape[1] / 1280.0)
        box_th  = max(2, int(round(3 * scale)))
        font    = cv2.FONT_HERSHEY_SIMPLEX
        ts      = 0.72 * scale
        th      = max(1, int(round(1.4 * scale)))
        pad     = int(6 * scale)

        for st in self._tracks.values():
            if not st.confirmed:
                continue
            draw = st.draw_box or st.box
            x1, y1, x2, y2 = (int(round(v)) for v in draw)
            if x2 <= x1 or y2 <= y1:
                continue
            color = _bgr(st.class_id)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, box_th)

            label = st.class_name
            (tw, lh), _ = cv2.getTextSize(label, font, ts, th)
            bg_y1 = max(0, y1 - lh - pad * 2)
            # label background
            cv2.rectangle(out, (x1, bg_y1), (x1 + tw + pad * 2, y1), color, -1)
            cv2.putText(out, label, (x1 + pad, y1 - pad // 2),
                        font, ts, (255, 255, 255), th, cv2.LINE_AA)
        return out

    def _publish(self, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_q])
        if ok:
            with self._lock:
                self._output_jpeg = buf.tobytes()

    def _publish_placeholder(self) -> None:
        w, h = self.cam_manager.resolution()
        top = np.linspace(16, 26, h, dtype=np.uint8)
        img = np.repeat(top[:, None], w, axis=1)
        img = np.stack([img, img, np.clip(img + 6, 0, 255)], axis=2)
        text = "OBJECT DETECTION"
        sc = max(0.6, w / 1400.0)
        (tw, tlh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
        cv2.putText(img, text, ((w - tw) // 2, (h + tlh) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, (70, 78, 92), 2, cv2.LINE_AA)
        self._publish(img)

    def _mark_status(self, status: str) -> None:
        with self._lock:
            self._stats = {**self._empty_stats(), "camera_status": status}

    @staticmethod
    def _empty_stats() -> dict:
        return {
            "total_objects": 0, "unique_classes": 0, "fps": 0.0,
            "camera_status": "no_source", "class_counts": {}, "mode": "od",
            # Aliases so shared dashboard fields work without JS branching
            "people_count": 0, "recognized_count": 0, "guest_count": 0,
        }

    def _update_stats(self) -> None:
        status = self.camera_state()
        with self._lock:
            confirmed = [st for st in self._tracks.values() if st.confirmed]
            counts: Dict[str, int] = {}
            for st in confirmed:
                counts[st.class_name] = counts.get(st.class_name, 0) + 1
            self._stats = {
                "total_objects":   len(confirmed),
                "unique_classes":  len(counts),
                "fps":             round(self._fps_ema, 1),
                "camera_status":   status,
                "class_counts":    counts,
                "mode":            "od",
                # Aliases
                "people_count":    len(confirmed),
                "recognized_count": len(counts),
                "guest_count":     0,
            }
