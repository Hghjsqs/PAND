"""Application logging setup."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging() -> logging.Logger:
    """Configure console and file logging once, then return the app logger."""
    logger = logging.getLogger("kbqa")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    project_root = Path(__file__).resolve().parents[1]
    log_path = project_root / "logs" / "app.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
