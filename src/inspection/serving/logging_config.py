"""Structured (JSON) logging for the API.

Plain-text logs are fine for a script you run and read yourself, which is all
this project needed through Milestone 5. A deployed service is different:
whatever platform runs it (Cloud Run, in this project's case) collects stdout
as log lines with no idea they're related, so each line needs to carry its own
context (request id, latency, decision) as structured fields rather than
relying on a human reading the surrounding lines.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Anything passed via logger.info(..., extra={...}) rides along here.
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_RECORD_KEYS:
                continue
            if key.startswith("_"):
                continue
            payload.setdefault(key, value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


_RESERVED_LOG_RECORD_KEYS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "taskName",
}


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)

    # Replace any pre-existing handlers (e.g. uvicorn's default) so every log
    # line - ours and the framework's - comes out as one consistent JSON shape.
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
