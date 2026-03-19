"""Centralized logging configuration for the Credit Risk Modeling Platform."""

import logging
import logging.handlers
import sys
from pathlib import Path

_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s -- %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Noisy third-party loggers to suppress
_QUIET_LOGGERS = ("httpx", "anthropic", "strands", "urllib3", "httpcore")


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logger with console and rotating file handlers.

    Safe to call multiple times — skips if handlers are already attached.
    """
    root = logging.getLogger()

    # Guard against duplicate attachment
    if root.handlers:
        return

    level = logging.getLevelName(log_level.upper())
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler — respects configured level
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Rotating file handler — captures everything
    from backend.config import PROJECT_ROOT

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10_485_760,  # 10 MB
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def add_pipeline_log_handler(run_dir: Path) -> logging.FileHandler:
    """Attach a per-pipeline-run file handler for regulatory audit trail.

    Returns the handler so the caller can remove it after the run completes.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run_dir / "pipeline.log")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logging.getLogger().addHandler(handler)
    return handler
