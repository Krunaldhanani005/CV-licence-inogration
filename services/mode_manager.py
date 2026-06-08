"""Manages the active detection mode: Face Recognition (fr) or Object Detection (od).

Only one pipeline runs at a time. Switching stops the current pipeline completely
before starting the new one — no shared state, no cross-mode CPU contention.
"""
from __future__ import annotations

import threading

from core.utils import get_logger

logger = get_logger(__name__)

FR = "fr"
OD = "od"


class ModeManager:
    """Holds both pipelines; exactly one is active at a time."""

    def __init__(self, fr_pipeline, od_pipeline) -> None:
        self._fr = fr_pipeline
        self._od = od_pipeline
        self._mode = FR
        self._lock = threading.Lock()

    # -------------------------------------------------------------- access
    @property
    def mode(self) -> str:
        return self._mode

    def pipeline(self):
        """Return the currently active pipeline."""
        return self._fr if self._mode == FR else self._od

    def fr_pipeline(self):
        return self._fr

    def od_pipeline(self):
        return self._od

    # ------------------------------------------------------------ switching
    def switch(self, mode: str) -> dict:
        """Stop the active pipeline and start the requested one.

        If detection was running it is automatically restarted on the new
        pipeline so the user does not need to press Start Detection again.
        """
        with self._lock:
            if mode not in (FR, OD):
                return {"success": False, "message": f"Unknown mode: {mode!r}"}
            if mode == self._mode:
                return {"success": True, "message": f"Already in {mode} mode",
                        "mode": mode}

            current = self.pipeline()
            was_running = current.detection_running

            logger.info("ModeManager: switching %s → %s (was_running=%s)",
                        self._mode, mode, was_running)

            current.stop_detection()
            current.stop()

            self._mode = mode
            new_pipe = self.pipeline()
            new_pipe.start()

            if was_running:
                result = new_pipe.start_detection()
                if not result.get("success"):
                    logger.warning("Mode switch: new pipeline start_detection failed: %s",
                                   result.get("message"))

        logger.info("ModeManager: now in %s mode", mode)
        return {"success": True, "message": f"Switched to {mode} mode",
                "mode": mode, "was_running": was_running}
