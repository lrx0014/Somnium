"""Unit tests for structured logging with correlation IDs and metrics interface (M0.8)."""

from __future__ import annotations

import io
import json
import logging

from somnium.observability import (
    MetricsSink,
    NoOpMetricsSink,
    StructuredLogFormatter,
    correlation_context,
    get_logger,
)


def test_structured_logger_injects_correlation_ids() -> None:
    """Verify request_id, job_id, and tenant_id are emitted in JSON log records."""
    stream = io.StringIO()
    raw_logger = logging.getLogger("somnium.test_corr")
    raw_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(StructuredLogFormatter(json_output=True))
    raw_logger.handlers = [handler]
    raw_logger.propagate = False

    logger = get_logger("somnium.test_corr")
    with correlation_context(
        request_id="req-123",
        job_id="dream-456",
        tenant_id="local-user",
    ):
        logger.info("Executing recall query", recall_mode="evidence", latency_ms=4.2)

    output = stream.getvalue().strip()
    parsed = json.loads(output)
    assert parsed["message"] == "Executing recall query"
    assert parsed["request_id"] == "req-123"
    assert parsed["job_id"] == "dream-456"
    assert parsed["tenant_id"] == "local-user"
    assert parsed["recall_mode"] == "evidence"
    assert parsed["latency_ms"] == 4.2


def test_noop_metrics_sink_conforms_to_protocol() -> None:
    """Verify NoOpMetricsSink satisfies MetricsSink and executes safely."""
    sink = NoOpMetricsSink()
    assert isinstance(sink, MetricsSink)
    sink.increment("somnium.requests.total", tags={"mode": "evidence"})
    sink.observe("somnium.recall.latency_ms", 12.5, tags={"mode": "evidence"})
    sink.gauge("somnium.dream.pending_examples", 42.0)
