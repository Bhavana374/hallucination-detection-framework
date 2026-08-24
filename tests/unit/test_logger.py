"""Unit tests for structured logging utility."""

import logging
from pathlib import Path
from src.utils.logger import setup_logger


def test_setup_logger_console_only():
    """Test logger setup without file output."""
    logger = setup_logger("test_console_logger", log_level="DEBUG", log_file=None, console_output=True)
    assert logger.name == "test_console_logger"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_setup_logger_with_file(tmp_path: Path):
    """Test logger setup writing to a temporary file."""
    log_file = tmp_path / "test.log"
    logger = setup_logger("test_file_logger", log_level="INFO", log_file=str(log_file), console_output=False)
    logger.info("Test log message for verification.")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test log message for verification." in content
