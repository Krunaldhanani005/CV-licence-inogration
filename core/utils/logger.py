"""Centralised logging configuration."""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_CONFIGURED = False

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(logs_dir: str, level: int = logging.INFO) -> None:
    """Configure root logging with console + rotating file handlers (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    os.makedirs(logs_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, "system.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Quiet down noisy third-party loggers.
    for noisy in ("ultralytics", "werkzeug", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    root.info("Logging initialised -> %s", logs_dir)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)
