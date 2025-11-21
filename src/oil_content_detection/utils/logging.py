"""Logging configuration utilities."""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "oil_content_detection",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure and return a logger for the application.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional path to write logs to file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Format: [2024-09-30 10:15:30] INFO: message
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "oil_content_detection") -> logging.Logger:
    """Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


__all__ = ["setup_logger", "get_logger"]