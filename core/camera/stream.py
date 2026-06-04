"""Threaded RTSP / webcam capture.

A dedicated capture thread keeps only the *latest* frame so that the inference
pipeline never blocks on slow network reads. Automatic reconnection is handled
transparently.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from core.utils import get_logger

logger = get_logger(__name__)


class CameraStream:
    """Background-threaded video capture with auto-reconnect."""

    def __init__(
        self,
        source: str | int,
        resolution: Tuple[int, int] = (1280, 720),
        fps: int = 30,
        reconnect_delay: float = 3.0,
        name: str = "camera",
    ) -> None:
        self.source = source
        self.width, self.height = resolution
        self.fps = fps
        self.reconnect_delay = reconnect_delay
        self.name = name

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._last_frame_ts = 0.0

    # --------------------------------------------------------------- lifecycle
    def start(self) -> "CameraStream":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"capture-{self.name}", daemon=True)
        self._thread.start()
        logger.info("Capture thread started for source=%s", self.source)
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._release()
        logger.info("Capture thread stopped for source=%s", self.source)

    # ------------------------------------------------------------------ access
    def read(self) -> Optional[np.ndarray]:
        """Return a copy of the most recent frame (or ``None``)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    @property
    def connected(self) -> bool:
        # Consider the stream stale if no frame arrived for 5s.
        return self._connected and (time.time() - self._last_frame_ts) < 5.0

    # -------------------------------------------------------------- internals
    def _open(self) -> bool:
        try:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG) \
                if isinstance(self.source, str) else cv2.VideoCapture(self.source)
            # Keep buffer tiny to favour latest frame over latency build-up.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if isinstance(self.source, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                if self.fps:
                    cap.set(cv2.CAP_PROP_FPS, self.fps)
            if not cap.isOpened():
                cap.release()
                return False
            self._cap = cap
            self._connected = True
            logger.info("Connected to camera source=%s", self.source)
            return True
        except Exception as exc:  # noqa: BLE001 - capture must never crash
            logger.error("Error opening camera %s: %s", self.source, exc)
            return False

    def _release(self) -> None:
        self._connected = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _loop(self) -> None:
        while self._running:
            if self._cap is None or not self._connected:
                if not self._open():
                    logger.warning("Camera connect failed, retrying in %.1fs", self.reconnect_delay)
                    time.sleep(self.reconnect_delay)
                    continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                logger.warning("Frame read failed; reconnecting…")
                self._release()
                time.sleep(self.reconnect_delay)
                continue

            # Resize to target resolution for predictable downstream cost.
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            with self._lock:
                self._frame = frame
                self._last_frame_ts = time.time()

        self._release()
