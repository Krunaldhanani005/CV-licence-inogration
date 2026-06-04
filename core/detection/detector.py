"""YOLOv8n person detection optimised for CPU.

The detector owns the underlying Ultralytics model instance. The tracking module
reuses this same model so that ByteTrack runs in a single forward pass.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import Settings
from core.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """A single person detection."""

    box: Tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float
    track_id: Optional[int] = None


class PersonDetector:
    """Wraps a YOLOv8n model and filters to the ``person`` class."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        cfg = settings.section("detection")
        self.person_class_id = int(cfg.get("person_class_id", 0))
        self.confidence = float(cfg.get("confidence", 0.40))
        self.iou = float(cfg.get("iou", 0.50))
        self.imgsz = int(cfg.get("imgsz", 640))
        self.device = cfg.get("device", "cpu")
        self._model = None
        self._load(cfg.get("model_path", "yolov8n.pt"))

    # --------------------------------------------------------------- model io
    def _load(self, model_path: str) -> None:
        from ultralytics import YOLO  # local import keeps startup fast

        # Prefer a copy in models/ if present, else let Ultralytics fetch it.
        models_dir = self.settings.path("models_dir")
        local = os.path.join(models_dir, os.path.basename(model_path))
        path = local if os.path.exists(local) else model_path
        logger.info("Loading YOLOv8n model: %s (device=%s)", path, self.device)
        self._model = YOLO(path)
        self._model.to(self.device)
        logger.info("YOLOv8n model ready")

    @property
    def model(self):  # exposed for the tracker
        return self._model

    # ---------------------------------------------------------------- detect
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run plain detection (no tracking) and return person detections."""
        results = self._model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.person_class_id],
            device=self.device,
            verbose=False,
        )
        return self._parse(results)

    @staticmethod
    def _parse(results) -> List[Detection]:
        detections: List[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(xyxy)
        for (x1, y1, x2, y2), conf, tid in zip(xyxy, confs, ids):
            detections.append(
                Detection(
                    box=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(conf),
                    track_id=int(tid) if tid is not None else None,
                )
            )
        return detections
