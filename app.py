"""AI Reception Monitoring System — Flask application entrypoint.

Run::

    python app.py

Then open http://localhost:5000 on the reception monitor.
"""
from __future__ import annotations

import logging
import os
import signal
import sys

# Limit ORT (InsightFace + YOLO) and OpenMP threads BEFORE any model is loaded.
# Default is all cores; with 16 cores both YOLO and InsightFace grab 16 threads
# each → they contend and BOTH slow down when recognition fires.
# 6 threads each leaves enough headroom so YOLO keeps its pace while InsightFace
# runs concurrently in the recognition worker thread.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "ORT_NUM_THREADS"):
    os.environ.setdefault(_var, "6")

from flask import Flask

from config import get_settings
from core.utils import setup_logging, get_logger
from routes import api_bp, pages_bp
from routes.object_api import object_bp
from services import MonitoringPipeline, PersonService, ObjectDetectionPipeline, ModeManager


def create_app() -> Flask:
    settings = get_settings()
    setup_logging(settings.path("logs_dir"),
                  level=logging.DEBUG if settings.get("app", "debug") else logging.INFO)
    logger = get_logger("app")
    logger.info("=== AI Reception Monitoring System starting ===")

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["SECRET_KEY"] = settings.get("app", "secret_key", "change-me")
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
    app.config["SETTINGS"] = settings

    # Build both pipelines — only one runs at a time (ModeManager controls this).
    pipeline    = MonitoringPipeline(settings)
    od_pipeline = ObjectDetectionPipeline(settings)
    person_service = PersonService(pipeline.recognizer)
    mode_manager   = ModeManager(pipeline, od_pipeline)

    app.config["PIPELINE"]     = pipeline        # kept for PersonService wiring
    app.config["PERSON_SERVICE"] = person_service
    app.config["MODE_MANAGER"] = mode_manager

    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(object_bp)

    # Start the FR pipeline by default (idle — camera not opened yet).
    mode_manager.fr_pipeline().start()

    # Graceful shutdown — stop whichever pipeline is currently active.
    def _shutdown(signum, _frame):
        logger.info("Signal %s received — shutting down", signum)
        mode_manager.pipeline().stop()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _shutdown)
        except ValueError:
            pass

    logger.info("=== Initialisation complete ===")
    return app


def main() -> None:
    settings = get_settings()
    app = create_app()
    host = settings.get("app", "host", "0.0.0.0")
    port = int(settings.get("app", "port", 5000))
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )

if __name__ == "__main__":
    main()
