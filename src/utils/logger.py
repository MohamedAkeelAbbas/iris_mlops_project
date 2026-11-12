"""
Logging Configuration Module
============================
Centralized logging setup using loguru.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_file: str = "logs/pipeline.log", level: str = "INFO"):
    """
    Configure the logger for the entire application.

    Args:
        log_file: Path to the log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Remove default handler
    logger.remove()

    # Create log directory if not exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Add console handler with color
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Add file handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    logger.info(f"Logger initialized. Log file: {log_file}")
    return logger


# Create a default logger instance
def get_logger():
    """Get the configured logger instance."""
    return logger
