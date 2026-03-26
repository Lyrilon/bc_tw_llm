"""Dual-output logging for the classify module, tqdm-compatible.

Provides:
  - Standard Python logging (console INFO via tqdm + file DEBUG with RSS)
  - LogContext: a tree-structured logger for hierarchical pipeline output
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
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
    ch.setFormatter(logging.Formatter("%(message)s"))
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


# ---------------------------------------------------------------------------
# LogContext — tree-structured hierarchical logging
# ---------------------------------------------------------------------------

# Box-drawing characters
_BRANCH = "├── "
_LAST   = "└── "
_PIPE   = "│   "
_SPACE  = "    "


class LogContext:
    """Manages indentation depth and tree-drawing for hierarchical log output.

    Usage::

        ctx = LogContext(logger)
        ctx.section("Pipeline", "Task: cross-layer | PCA: 128")
        ctx.phase(1, 2, "Track 1: Agile Baseline")
        with ctx.group("Loading data"):
            ctx.step("Loaded 1,600,000 records")
            ctx.step("Sampled 100,000 records", last=True)
        with ctx.group("Classical ML (3 models)", last=True):
            with ctx.group("[1/3] LogisticRegression"):
                ctx.step("Training...done (3.3s)")
                ctx.step("Val:  acc=0.4622", last=True)
    """

    def __init__(self, logger: logging.Logger | None = None):
        self._log = logger or logging.getLogger("classify")
        # Stack of booleans: for each depth level, is the current group
        # the last sibling at that level?
        self._last_stack: list[bool] = []

    # -- Helpers ------------------------------------------------------------

    def _prefix(self, is_last: bool = False) -> str:
        """Build the tree prefix string for the current depth."""
        parts: list[str] = []
        for parent_is_last in self._last_stack:
            parts.append(_SPACE if parent_is_last else _PIPE)
        parts.append(_LAST if is_last else _BRANCH)
        return "".join(parts)

    def _continuation_prefix(self) -> str:
        """Prefix for continuation lines (no branch char, just pipes)."""
        parts: list[str] = []
        for parent_is_last in self._last_stack:
            parts.append(_SPACE if parent_is_last else _PIPE)
        return "".join(parts)

    def _emit(self, msg: str):
        """Send a message through the logger."""
        self._log.info(msg)

    # -- Public API ---------------------------------------------------------

    def section(self, title: str, subtitle: str | None = None):
        """Top-level section header with ══ borders."""
        width = 70
        self._emit("")
        self._emit("═" * width)
        self._emit(f"  {title}")
        if subtitle:
            self._emit(f"  {subtitle}")
        self._emit("═" * width)
        self._emit("")

    def phase(self, index: int, total: int, title: str):
        """Numbered phase header: [1/2] Title."""
        self._emit(f"[{index}/{total}] {title}")

    @contextmanager
    def group(self, title: str, last: bool = False):
        """Context manager that prints a titled group and indents children.

        Args:
            title: Group header text
            last: True if this is the last sibling at the current depth
        """
        prefix = self._prefix(is_last=last)
        self._emit(f"{prefix}{title}")
        self._last_stack.append(last)
        try:
            yield
        finally:
            self._last_stack.pop()

    def step(self, msg: str, last: bool = False):
        """A single step line at the current depth."""
        prefix = self._prefix(is_last=last)
        self._emit(f"{prefix}{msg}")

    def detail(self, msg: str):
        """A continuation/detail line (no branch character)."""
        prefix = self._continuation_prefix()
        self._emit(f"{prefix}{msg}")

    def blank(self):
        """Emit a blank line."""
        self._emit("")

    def text(self, msg: str):
        """Emit raw text with no prefix (for reports, tables)."""
        self._emit(msg)
