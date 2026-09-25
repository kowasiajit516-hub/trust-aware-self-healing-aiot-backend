"""
utils/logger.py
----------------
Centralized logging configuration for the backend.

Call setup_logging() once, on app startup (in main.py), before any
other module logs anything. Every other file just does:

    import logging
    logger = logging.getLogger(__name__)

and gets consistent formatting/level automatically.
"""

import logging
import sys

from config import settings


def setup_logging() -> None:
    """Configure root logging for the whole application."""

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers if setup_logging() is called more than once
    # (e.g. during tests with app reloads).
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    # Quiet down noisy third-party loggers unless we're debugging.
    if level > logging.DEBUG:
        logging.getLogger("pymongo").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured at level '%s'", settings.log_level.upper()
    )
