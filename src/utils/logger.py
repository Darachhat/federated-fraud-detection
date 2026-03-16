"""
Structured logging utility using loguru.
Provides per-component loggers with file and console output.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    component_name: str,
    log_dir: str = "logs/",
    level: str = "INFO"
) -> logger:
    """
    Configure and return a named logger for a specific component.

    Args:
        component_name: Identifier for the logging component
                        (e.g., 'server', 'bank1', 'bank2').
        log_dir:        Directory where log files are written.
        level:          Logging level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured loguru logger instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{component_name}.log"

    # Remove default handler
    logger.remove()

    # Console handler — colored output
    logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            f"<cyan>{component_name}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True
    )

    # File handler — plain text
    logger.add(
        str(log_file),
        level=level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            f"{component_name} | "
            "{message}"
        ),
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )

    return logger