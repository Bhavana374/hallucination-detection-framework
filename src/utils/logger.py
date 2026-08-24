"""Logging utilities providing structured logging with console and file handlers."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "hallucination_detection",
    log_level: str = "INFO",
    log_file: Optional[str] = "results/raw/framework.log",
    console_output: bool = True,
) -> logging.Logger:

    """Configure and return a structured logger.

    Args:
        name: Name of the logger instance.
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        log_file: Path to the log file. If provided, parent directories will be created.
        console_output: Whether to attach a stdout StreamHandler.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logger is called repeatedly
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


get_logger = setup_logger
