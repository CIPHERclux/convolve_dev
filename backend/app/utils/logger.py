"""
Structured Logging — replaces 200+ print() statements.
"""

import logging
import sys


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Get a configured logger for a module."""
    logger = logging.getLogger(f"convolve.{name}")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s │ %(levelname)-7s │ %(name)-28s │ %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if level is not None:
        logger.setLevel(level)
    elif not logger.level:
        logger.setLevel(logging.DEBUG)

    return logger
