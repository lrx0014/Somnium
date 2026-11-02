"""Structured logging with request/job correlation IDs and metrics interface (M0.8).

Correlation IDs (`request_id`, `job_id`, `tenant_id`) are propagated via
`contextvars` so every log line emitted during an API request or Dream cycle
can be traced deterministically.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

request_id_var: ContextVar[str | None] = ContextVar("somnium_request_id", default=None)
job_id_var: ContextVar[str | None] = ContextVar("somnium_job_id", default=None)
tenant_id_var: ContextVar[str | None] = ContextVar("somnium_tenant_id", default=None)


@contextmanager
def correlation_context(
    *,
    request_id: str | None = None,
    job_id: str | None = None,
    tenant_id: str | None = None,
) -> Iterator[None]:
    """Bind request, job, and tenant correlation IDs for the duration of a block."""
    tokens: list[tuple[ContextVar[str | None], Any]] = []
    if request_id is not None:
        tokens.append((request_id_var, request_id_var.set(request_id)))
    if job_id is not None:
        tokens.append((job_id_var, job_id_var.set(job_id)))
    if tenant_id is not None:
        tokens.append((tenant_id_var, tenant_id_var.set(tenant_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


class StructuredLogFormatter(logging.Formatter):
    """Formatter emitting either structured JSON lines or key-value text lines."""

    def __init__(self, *, json_output: bool = True) -> None:
        super().__init__()
        self._json_output = json_output

    def format(self, record: logging.LogRecord) -> str:
        """Format a LogRecord with correlation IDs and optional structured fields."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
            "job_id": getattr(record, "job_id", None) or job_id_var.get(),
            "tenant_id": getattr(record, "tenant_id", None) or tenant_id_var.get(),
        }
        extra_fields = getattr(record, "structured_fields", None)
        if isinstance(extra_fields, Mapping):
            payload.update(extra_fields)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        if self._json_output:
            return json.dumps(payload, default=str, sort_keys=False)

        parts = [
            f"{payload['timestamp']} [{payload['level']}] {payload['logger']}: {payload['message']}"
        ]
        for key in ("request_id", "job_id", "tenant_id"):
            if payload.get(key) is not None:
                parts.append(f"{key}={payload[key]}")
        if isinstance(extra_fields, Mapping):
            for k, v in extra_fields.items():
                parts.append(f"{k}={v}")
        return " ".join(parts)


class StructuredLogger:
    """Thin wrapper over `logging.Logger` supporting keyword structured fields."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @property
    def name(self) -> str:
        return self._logger.name

    def log(self, level: int, msg: str, **fields: Any) -> None:
        """Emit a log message at `level` with optional structured key-value fields."""
        self._logger.log(level, msg, extra={"structured_fields": fields})

    def debug(self, msg: str, **fields: Any) -> None:
        self.log(logging.DEBUG, msg, **fields)

    def info(self, msg: str, **fields: Any) -> None:
        self.log(logging.INFO, msg, **fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self.log(logging.WARNING, msg, **fields)

    def error(self, msg: str, **fields: Any) -> None:
        self.log(logging.ERROR, msg, **fields)


def get_logger(name: str = "somnium") -> StructuredLogger:
    """Return a `StructuredLogger` instance for `name`."""
    return StructuredLogger(logging.getLogger(name))


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Configure the root `somnium` logger with `StructuredLogFormatter`."""
    root = logging.getLogger("somnium")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredLogFormatter(json_output=json_output))
    root.handlers = [handler]
    root.propagate = False


@runtime_checkable
class MetricsSink(Protocol):
    """Observability interface for recording counters, histograms, and gauges."""

    def increment(self, name: str, value: int = 1, tags: Mapping[str, str] | None = None) -> None:
        """Increment a counter metric by `value`."""
        ...

    def observe(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        """Record a value in a histogram/distribution metric (e.g., latency)."""
        ...

    def gauge(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        """Set the current value of a gauge metric."""
        ...


class NoOpMetricsSink:
    """Default no-op metrics implementation used in v0.1."""

    def increment(self, name: str, value: int = 1, tags: Mapping[str, str] | None = None) -> None:
        """Intentionally do nothing."""
        return None

    def observe(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        """Intentionally do nothing."""
        return None

    def gauge(self, name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
        """Intentionally do nothing."""
        return None
