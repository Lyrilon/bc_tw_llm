"""Dual-output logging with memory monitoring."""

from __future__ import annotations

import logging
import os
from datetime import datetime

import psutil


class MemoryFilter(logging.Filter):
    """Inject current RSS (MB) into every log record."""

    def __init__(self) -> None:
        super().__init__()
        self._process = psutil.Process(os.getpid())

    def filter(self, record: logging.LogRecord) -> bool:
        record.rss_mb = self._process.memory_info().rss / (1024 * 1024)  # type: ignore[attr-defined]
        return True


def setup_logging(log_dir: str) -> logging.Logger:
    """Configure and return the project logger.

    - Console handler: INFO level, concise format.
    - File handler: DEBUG level, verbose format with timestamp and RSS.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"dataset_generation_{timestamp}.log")

    logger = logging.getLogger("threat_dataset")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    mem_filter = MemoryFilter()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    ch.addFilter(mem_filter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] [RSS=%(rss_mb).0fMB] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    fh.addFilter(mem_filter)
    logger.addHandler(fh)

    logger.info("Logging initialised — file: %s", log_file)
    return logger
