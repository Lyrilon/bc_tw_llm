"""Dual-output logging for the classify module, tqdm-compatible."""

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


class TqdmLoggingHandler(logging.StreamHandler):
    """Console handler that writes through tqdm so progress bars stay clean."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
        except ImportError:
            super().emit(record)


def setup_logging(log_dir: str) -> logging.Logger:
    """Configure and return the ``classify`` logger.

    - Console: INFO, concise, routed through tqdm.write().
    - File: DEBUG, verbose with timestamp + RSS.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"classify_{timestamp}.log")

    logger = logging.getLogger("classify")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    mem_filter = MemoryFilter()

    # Console handler (tqdm-safe)
    ch = TqdmLoggingHandler()
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
