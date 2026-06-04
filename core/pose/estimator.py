"""Posture / activity classification using MediaPipe Pose.

Static posture (Standing / Sitting / Bending) is inferred from body landmarks of
a cropped person ROI. Dynamic activity (Walking / Running / Idle) is decided from
a motion-speed hint supplied by the tracker (normalised displacement per frame),
because a single frame cannot distinguish standing-still from walking.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from config import Settings
from core.utils import get_logger

logger = get_logger(__name__)

# Activity labels (exact strings shown on screen).
STANDING = "Standing"
SITTING = "Sitting"
WALKING = "Walking"
RUNNING = "Running"
BENDING = "Bending"
IDLE = "Idle"


class PoseEstimator:
    """Wraps MediaPipe Pose and maps landmarks + motion to an activity label."""

    def __init__(self, settings: Settings) -> None:
        cfg = settings.section("pose")
        self.enabled = bool(cfg.get("enabled", True))
        self._pose = None
        if not self.enabled:
            return
        try:
            import mediapipe as mp

            self._mp = mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=int(cfg.get("model_complexity", 0)),
                enable_segmentation=False,
                min_detection_confidence=float(cfg.get("min_detection_confidence", 0.5)),
                min_tracking_confidence=float(cfg.get("min_tracking_confidence", 0.5)),
            )
            self._lm = mp.solutions.pose.PoseLandmark
            logger.info("MediaPipe Pose ready (complexity=%s)", cfg.get("model_complexity", 0))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to init MediaPipe Pose: %s", exc)
            self.enabled = False

    # ----------------------------------------------------------------- public
    def estimate(self, crop: np.ndarray, motion_speed: float = 0.0) -> str:
        """Return an activity label for a person crop.

        ``motion_speed`` is the centroid displacement between recent frames,
        normalised by the person-box height (so it is scale invariant).
        """
        if not self.enabled or self._pose is None or crop is None or crop.size == 0:
            return self._activity_from_motion(motion_speed, STANDING)

        try:
            import cv2

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = self._pose.process(rgb)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pose process error: %s", exc)
            return self._activity_from_motion(motion_speed, STANDING)

        if not result.pose_landmarks:
            return self._activity_from_motion(motion_speed, STANDING)

        posture = self._classify_posture(result.pose_landmarks.landmark)
        return self._combine(posture, motion_speed)

    # -------------------------------------------------------------- internals
    def _classify_posture(self, lm) -> str:
        """Heuristic static posture from normalised landmarks (0..1)."""
        L = self._lm
        try:
            sh = self._mid(lm, L.LEFT_SHOULDER, L.RIGHT_SHOULDER)
            hip = self._mid(lm, L.LEFT_HIP, L.RIGHT_HIP)
            knee = self._mid(lm, L.LEFT_KNEE, L.RIGHT_KNEE)
            ankle = self._mid(lm, L.LEFT_ANKLE, L.RIGHT_ANKLE)
        except Exception:  # noqa: BLE001
            return STANDING

        # Torso tilt: horizontal offset vs vertical extent (shoulder->hip).
        torso_dx = abs(sh[0] - hip[0])
        torso_dy = abs(hip[1] - sh[1]) or 1e-3
        torso_tilt = torso_dx / torso_dy

        # Vertical gaps.
        hip_knee = abs(knee[1] - hip[1])
        knee_ankle = abs(ankle[1] - knee[1])

        # Strongly tilted torso -> bending over.
        if torso_tilt > 0.8:
            return BENDING

        # Legs folded: hip close to knee level => sitting.
        if hip_knee < 0.12 and knee[1] < ankle[1]:
            return SITTING

        # Compressed lower body (thigh ~ shin collapsed) also reads as sitting.
        if hip_knee < 0.10 or (knee_ankle < 0.08 and hip_knee < 0.18):
            return SITTING

        return STANDING

    def _combine(self, posture: str, motion: float) -> str:
        """Blend static posture with motion to pick the final activity."""
        # While clearly standing/upright, movement implies walking/running.
        if posture == STANDING:
            return self._activity_from_motion(motion, STANDING)
        # Sitting/Bending are reported as-is (motion is unreliable there).
        return posture

    @staticmethod
    def _activity_from_motion(motion: float, upright: str) -> str:
        if motion >= 0.060:
            return RUNNING
        if motion >= 0.018:
            return WALKING
        if motion < 0.004:
            return IDLE if upright == STANDING else upright
        return upright

    @staticmethod
    def _mid(lm, a, b):
        pa, pb = lm[a.value], lm[b.value]
        return ((pa.x + pb.x) / 2.0, (pa.y + pb.y) / 2.0)

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
