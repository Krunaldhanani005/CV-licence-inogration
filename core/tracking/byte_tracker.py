"""ByteTrack-based person tracking.

Ultralytics ships ByteTrack natively. We drive it through ``model.track`` so that
detection + association happen in a single forward pass — important for CPU
throughput. The tracker emits :class:`Detection` objects carrying stable
``track_id`` values plus bookkeeping of which tracks are currently active so the
pipeline knows when a track is lost (and a re-recognition is required).
"""
from __future__ import annotations

import os
from typing import List, Set

import cv2
import numpy as np

from config import Settings
from core.detection import Detection, PersonDetector
from core.utils import get_logger

logger = get_logger(__name__)


class PersonTracker:
    """Stateful ByteTrack wrapper over a shared YOLO model."""

    def __init__(self, settings: Settings, detector: PersonDetector) -> None:
        self.settings = settings
        self.detector = detector
        det_cfg = settings.section("detection")
        trk_cfg = settings.section("tracking")
        self.person_class_id = int(det_cfg.get("person_class_id", 0))
        self.confidence = float(det_cfg.get("confidence", 0.40))
        self.iou = float(det_cfg.get("iou", 0.50))
        self.imgsz = int(det_cfg.get("imgsz", 640))
        self.device = det_cfg.get("device", "cpu")
        self.tracker_cfg = self._resolve_tracker_cfg(trk_cfg.get("tracker", "bytetrack.yaml"))
        self._active_ids: Set[int] = set()

    def _resolve_tracker_cfg(self, name: str) -> str:
        """Use a project-local tracker yaml if present, else the bundled one."""
        local = os.path.join(self.settings.base_dir, "config", name)
        return local if os.path.exists(local) else name

    # ------------------------------------------------------------------ track
    def update(self, frame: np.ndarray) -> List[Detection]:
        """Run tracking on a frame and return tracked person detections."""
        h, w = frame.shape[:2]
        # Pre-resize to imgsz so YOLO doesn't letterbox a full 1920×1080 frame
        # internally — saves ~10-20ms per detection on CPU.  Boxes are scaled
        # back to the original frame's coordinate space afterwards.
        if w > self.imgsz or h > self.imgsz:
            scale = self.imgsz / max(w, h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            sx, sy = w / nw, h / nh
        else:
            small = frame
            sx = sy = 1.0

        results = self.detector.model.track(
            small,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.person_class_id],
            device=self.device,
            tracker=self.tracker_cfg,
            persist=True,
            verbose=False,
        )
        detections = PersonDetector._parse(results)
        # Keep only detections that ByteTrack actually assigned an id to.
        detections = [d for d in detections if d.track_id is not None]
        # Scale boxes from the downsampled frame back to original coordinates.
        if sx != 1.0 or sy != 1.0:
            for d in detections:
                x1, y1, x2, y2 = d.box
                d.box = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
        self._active_ids = {d.track_id for d in detections}
        return detections

    # --------------------------------------------------------------- bookkeep
    @property
    def active_ids(self) -> Set[int]:
        return set(self._active_ids)

    def reset(self) -> None:
        """Reset tracker state (e.g. after a camera reconnect)."""
        self._active_ids.clear()
        try:
            # Clears Ultralytics' internal per-stream tracker state.
            if hasattr(self.detector.model, "predictor") and self.detector.model.predictor:
                trackers = getattr(self.detector.model.predictor, "trackers", None)
                if trackers:
                    for t in trackers:
                        if hasattr(t, "reset"):
                            t.reset()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker reset skipped: %s", exc)
