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

        Detection is always automatically started on the new pipeline when a
        camera is configured — the user never needs to press Start again after
        switching modes.  If no camera is configured yet, the new pipeline
        starts idle (same as before) and a warning is logged.
        """
        with self._lock:
            if mode not in (FR, OD):
                return {"success": False, "message": f"Unknown mode: {mode!r}"}
            if mode == self._mode:
                return {"success": True, "message": f"Already in {mode} mode",
                        "mode": mode}

            current = self.pipeline()
            was_running = current.detection_running

            logger.info(
                "ModeManager: switching %s → %s  (detection_was_running=%s)",
                self._mode, mode, was_running,
            )

            current.stop_detection()
            current.stop()
            logger.info("ModeManager: %s pipeline stopped", self._mode)

            self._mode = mode
            new_pipe = self.pipeline()
            new_pipe.start()
            logger.info("ModeManager: %s pipeline started", mode)

            # Always attempt to auto-start detection on the new pipeline.
            # This removes the need for a manual Start button press after
            # every mode switch.  If no camera is configured the call
            # returns success=False with a clear message — log it and
            # continue; the UI overlay will prompt the user to configure one.
            auto_result = new_pipe.start_detection()
            if auto_result.get("success"):
                logger.info(
                    "ModeManager: detection auto-started on %s pipeline", mode,
                )
            else:
                logger.info(
                    "ModeManager: detection auto-start skipped on %s — %s",
                    mode, auto_result.get("message", "no camera"),
                )

        logger.info("ModeManager: now in %s mode", mode)
        return {
            "success": True,
            "message": f"Switched to {mode} mode",
            "mode": mode,
            "was_running": was_running,
            "detection_started": auto_result.get("success", False),
        }
