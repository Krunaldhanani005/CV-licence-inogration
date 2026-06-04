"""AI Reception Monitoring System — Flask application entrypoint.

Run::

    python app.py

Then open http://localhost:5000 on the reception monitor.
"""
from __future__ import annotations

import logging
import signal
import sys

from flask import Flask

from config import get_settings
from core.utils import setup_logging, get_logger
from routes import api_bp, pages_bp
from services import MonitoringPipeline, PersonService


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
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64MB photo uploads
    app.config["SETTINGS"] = settings

    # Build the long-lived pipeline + person service (models load here).
    pipeline = MonitoringPipeline(settings)
    person_service = PersonService(pipeline.recognizer)
    app.config["PIPELINE"] = pipeline
    app.config["PERSON_SERVICE"] = person_service

    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp)

    # Start the capture/inference threads immediately.
    pipeline.start()

    # Graceful shutdown.
    def _shutdown(signum, _frame):
        logger.info("Signal %s received — shutting down", signum)
        pipeline.stop()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _shutdown)
        except ValueError:
            pass  # not in main thread (e.g. under some WSGI servers)

    logger.info("=== Initialisation complete ===")
    return app


def main() -> None:
    settings = get_settings()
    app = create_app()
    host = settings.get("app", "host", "0.0.0.0")
    port = int(settings.get("app", "port", 5000))
    # use_reloader=False: avoid loading heavy models twice and killing threads.
    app.run(
    host="0.0.0.0",
    port=5000,
    debug=False
)

if __name__ == "__main__":
    main()
