"""
Structured logging configuration for the Focused Research Agent.

Fixes a real bug in the original setup: the root logger was hardcoded to
ERROR, which silently dropped every INFO/DEBUG/WARNING call throughout the
nodes and services despite the codebase being full of them. This version:

- Reads the level from LOG_LEVEL (default INFO) instead of hardcoding ERROR.
- Emits structured JSON in production (LOG_FORMAT=json) or readable text in
  development (LOG_FORMAT=console), controlled by env var.
- Automatically attaches the current run_id to every log record via a
  contextvar + logging.Filter, so run correlation works without having to
  thread run_id through every single logger.info() call by hand.
- Still writes to a rotating file (kept from the original), and additionally
  writes to stdout so container log collectors (Docker, CloudWatch, etc.)
  pick it up without extra configuration.
"""

import json
import logging
import os
import sys
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

_current_run_id: ContextVar[str] = ContextVar("current_run_id", default="-")


def bind_run_id(run_id: str) -> None:
    """Bind a run_id to the current async task/thread's logging context."""
    _current_run_id.set(run_id or "-")


def get_run_id() -> str:
    """Return the run_id bound to the current context, or '-' if none."""
    return _current_run_id.get()


class RunContextFilter(logging.Filter):
    """Attaches the current run_id to every log record automatically."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = get_run_id()
        return True


class JsonFormatter(logging.Formatter):
    """Renders log records as single-line JSON for log aggregation systems."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _resolve_level(raw: str | None) -> int:
    """Resolve a LOG_LEVEL env value to a logging level constant, defaulting to INFO."""
    if not raw:
        return logging.INFO
    return getattr(logging, raw.strip().upper(), logging.INFO)


def setup_logging() -> logging.Logger:
    """Configure and return the root logger for the application.

    Controlled by two env vars:
    - LOG_LEVEL: DEBUG | INFO | WARNING | ERROR (default INFO)
    - LOG_FORMAT: json | console (default console)

    Safe to call multiple times — returns the existing configured logger
    if handlers are already attached.

    Returns:
        logging.Logger: The configured root logger.
    """
    py_logger = logging.getLogger()

    if py_logger.handlers:
        return py_logger

    level = _resolve_level(os.getenv("LOG_LEVEL"))
    log_format = os.getenv("LOG_FORMAT", "console").strip().lower()
    py_logger.setLevel(level)

    run_filter = RunContextFilter()

    if log_format == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s [run_id=%(run_id)s] %(name)s: %(message)s"
        )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(run_filter)
    py_logger.addHandler(stream_handler)

    log_path = (
        Path(__file__).parent.parent.parent.parent
        / "logs"
        / "focused_research_agent.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        filename=log_path, mode="a", maxBytes=1_048_576, backupCount=10
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(run_filter)
    py_logger.addHandler(file_handler)

    return py_logger
