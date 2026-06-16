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

# --------------------------------------------------- COCO class catalogue (all 80)
COCO_CLASSES: Dict[int, str] = {
    0:  "Person",        1:  "Bicycle",       2:  "Car",           3:  "Motorcycle",
    4:  "Airplane",      5:  "Bus",           6:  "Train",         7:  "Truck",
    8:  "Boat",          9:  "Traffic Light", 10: "Fire Hydrant",  11: "Stop Sign",
    12: "Parking Meter", 13: "Bench",         14: "Bird",          15: "Cat",
    16: "Dog",           17: "Horse",         18: "Sheep",         19: "Cow",
    20: "Elephant",      21: "Bear",          22: "Zebra",         23: "Giraffe",
    24: "Backpack",      25: "Umbrella",      26: "Handbag",       27: "Tie",
    28: "Suitcase",      29: "Frisbee",       30: "Skis",          31: "Snowboard",
    32: "Sports Ball",   33: "Kite",          34: "Baseball Bat",  35: "Baseball Glove",
    36: "Skateboard",    37: "Surfboard",     38: "Tennis Racket", 39: "Bottle",
    40: "Wine Glass",    41: "Cup",           42: "Fork",          43: "Knife",
    44: "Spoon",         45: "Bowl",          46: "Banana",        47: "Apple",
    48: "Sandwich",      49: "Orange",        50: "Broccoli",      51: "Carrot",
    52: "Hot Dog",       53: "Pizza",         54: "Donut",         55: "Cake",
    56: "Chair",         57: "Couch",         58: "Potted Plant",  59: "Bed",
    60: "Dining Table",  61: "Toilet",        62: "TV",            63: "Laptop",
    64: "Mouse",         65: "Remote",        66: "Keyboard",      67: "Cell Phone",
    68: "Microwave",     69: "Oven",          70: "Toaster",       71: "Sink",
    72: "Refrigerator",  73: "Book",          74: "Clock",         75: "Vase",
    76: "Scissors",      77: "Teddy Bear",    78: "Hair Drier",    79: "Toothbrush",
}

# Empty list = all classes enabled (YOLO passes classes=None which detects everything)
DEFAULT_ENABLED: List[int] = []

# Per-class confidence thresholds for all 80 COCO classes.
# Small/far objects and animals get lower thresholds for better recall;
# large static objects prone to false-positives stay higher.
DEFAULT_CLASS_THRESHOLDS: Dict[int, float] = {
    0:  0.40,   # Person
    1:  0.40,   # Bicycle
    2:  0.40,   # Car
    3:  0.40,   # Motorcycle
    4:  0.40,   # Airplane
    5:  0.40,   # Bus
    6:  0.40,   # Train
    7:  0.40,   # Truck
    8:  0.40,   # Boat
    9:  0.35,   # Traffic Light — small, high priority
    10: 0.40,   # Fire Hydrant
    11: 0.40,   # Stop Sign
    12: 0.40,   # Parking Meter
    13: 0.40,   # Bench
    14: 0.40,   # Bird
    15: 0.35,   # Cat
    16: 0.35,   # Dog
    17: 0.40,   # Horse
    18: 0.40,   # Sheep
    19: 0.40,   # Cow
    20: 0.40,   # Elephant
    21: 0.40,   # Bear
    22: 0.40,   # Zebra
    23: 0.40,   # Giraffe
    24: 0.35,   # Backpack — small, commonly carried
    25: 0.40,   # Umbrella
    26: 0.35,   # Handbag — small, commonly carried
    27: 0.35,   # Tie — very small
    28: 0.40,   # Suitcase
    29: 0.40,   # Frisbee
    30: 0.40,   # Skis
    31: 0.40,   # Snowboard
    32: 0.35,   # Sports Ball — small, round, hard to detect
    33: 0.35,   # Kite
    34: 0.40,   # Baseball Bat
    35: 0.40,   # Baseball Glove
    36: 0.40,   # Skateboard
    37: 0.40,   # Surfboard
    38: 0.40,   # Tennis Racket
    39: 0.35,   # Bottle — small, important to detect
    40: 0.35,   # Wine Glass — small
    41: 0.35,   # Cup — small
    42: 0.35,   # Fork — very small
    43: 0.35,   # Knife — very small
    44: 0.35,   # Spoon — very small
    45: 0.40,   # Bowl
    46: 0.35,   # Banana
    47: 0.35,   # Apple
    48: 0.40,   # Sandwich
    49: 0.35,   # Orange
    50: 0.40,   # Broccoli
    51: 0.40,   # Carrot
    52: 0.40,   # Hot Dog
    53: 0.40,   # Pizza
    54: 0.40,   # Donut
    55: 0.40,   # Cake
    56: 0.50,   # Chair — prone to false positives; keep higher
    57: 0.50,   # Couch — large static object, keep higher
    58: 0.35,   # Potted Plant — often missed at distance
    59: 0.45,   # Bed
    60: 0.45,   # Dining Table — large, can overlap other objects
    61: 0.45,   # Toilet
    62: 0.40,   # TV
    63: 0.30,   # Laptop — small on tables; lowest threshold
    64: 0.30,   # Mouse — very small
    65: 0.30,   # Remote — very small
    66: 0.35,   # Keyboard
    67: 0.25,   # Cell Phone — smallest common object; needs lowest threshold
    68: 0.40,   # Microwave
    69: 0.40,   # Oven
    70: 0.40,   # Toaster
    71: 0.40,   # Sink
    72: 0.40,   # Refrigerator
    73: 0.35,   # Book
    74: 0.35,   # Clock
    75: 0.40,   # Vase
    76: 0.35,   # Scissors
    77: 0.35,   # Teddy Bear
    78: 0.35,   # Hair Drier
    79: 0.35,   # Toothbrush — very small
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
            "confidence": 0.25, "imgsz": 960, "fps_limit": 15,
            "track_buffer": 10, "min_object_size": 15, "jpeg_quality": 60,
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

        # Use dedicated OD model (od_model_path) when configured; fall back to
        # the shared model_path so existing deployments keep working unchanged.
        od_model = settings.get("detection", "od_model_path",
                                settings.get("detection", "model_path", "yolov8m.pt"))
        model_path = os.path.join(settings.base_dir, od_model)
        # If the file doesn't exist at the absolute path let Ultralytics
        # auto-download it (e.g. "yolov8m.pt" fetches from the hub on first run).
        if not os.path.isfile(model_path):
            model_path = od_model
        logger.info("Loading OD pipeline YOLO model: %s", model_path)
        self._model = YOLO(model_path)
        logger.info("OD pipeline YOLO model loaded: %s", getattr(self._model, "ckpt_path", model_path))

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
        self._confidence  = float(det.get("confidence", 0.35))
        self._imgsz       = int(det.get("imgsz", 1280))
        self._fps_limit   = max(1, int(det.get("fps_limit", 15)))
        self._max_missed  = max(5, int(det.get("track_buffer", 10)))
        self._min_size    = int(det.get("min_object_size", 15))
        self._jpeg_q      = int(det.get("jpeg_quality", 60))
        raw = cfg.get("enabled_classes", DEFAULT_ENABLED)
        self._enabled: Optional[List[int]] = ([int(c) for c in raw] if raw else None)

        # Per-class confidence thresholds.
        self._class_thresholds: Dict[int, float] = dict(
            cfg.get("class_thresholds", DEFAULT_CLASS_THRESHOLDS))

        # YOLO runs at the minimum threshold across all enabled classes so it
        # doesn't silently drop low-threshold detections (e.g. Cell Phone at
        # 0.25).  We post-filter to each class's actual threshold below.
        enabled_set = set(self._enabled) if self._enabled else set(self._class_thresholds)
        relevant = [v for k, v in self._class_thresholds.items() if k in enabled_set]
        self._yolo_conf = min(relevant) if relevant else self._confidence
        enabled_count = len(self._enabled) if self._enabled else len(COCO_CLASSES)
        logger.info(
            "OD params loaded — model_imgsz=%d  yolo_conf=%.2f  "
            "enabled_classes=%d  min_size=%dpx  fps_limit=%d",
            self._imgsz, self._yolo_conf, enabled_count,
            self._min_size, self._fps_limit,
        )

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
        cam_desc = self.cam_manager.describe()
        logger.info(
            "OD pipeline: detection started — camera=%s  imgsz=%d  yolo_conf=%.2f  classes=%s",
            cam_desc.get("label", "?"), self._imgsz, self._yolo_conf,
            f"all {len(COCO_CLASSES)}" if not self._enabled else len(self._enabled),
        )
        return {"success": True, "message": "Object Detection started", **cam_desc}

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
