"""Idle detection and agent traffic tracking for Dream Engine scheduling (M6.1).

Per `design.md` §7:
- Dream cycles run per tenant when the tenant is idle from external agent traffic.
- Idle refers to the absence of external agent requests (`store`, `ask`, `correct`, `forget`),
  not database availability.
- Backed by an injectable `Clock` for deterministic testing.
"""

from __future__ import annotations

import threading
from datetime import datetime

from somnium.backends.fake import SystemClock
from somnium.domain import TenantId
from somnium.interfaces import Clock


class TrafficTracker:
    """Thread-safe tracker for external agent request activity per tenant (M6.1)."""

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock: Clock = clock or SystemClock()
        self._lock = threading.Lock()
        self._last_traffic: dict[TenantId, datetime] = {}

    @property
    def clock(self) -> Clock:
        """Underlying clock instance."""
        return self._clock

    def record_traffic(
        self,
        tenant_id: TenantId,
        timestamp: datetime | None = None,
    ) -> None:
        """Record an external agent request for `tenant_id` at `timestamp` (or now)."""
        now = timestamp if timestamp is not None else self._clock.now()
        with self._lock:
            self._last_traffic[tenant_id] = now

    def get_last_traffic(self, tenant_id: TenantId) -> datetime | None:
        """Return the timestamp of the last agent request for `tenant_id`."""
        with self._lock:
            return self._last_traffic.get(tenant_id)

    def idle_seconds(self, tenant_id: TenantId) -> float:
        """Return elapsed seconds since the last agent request for `tenant_id`."""
        now = self._clock.now()
        with self._lock:
            last = self._last_traffic.get(tenant_id)
            if last is None:
                return float("inf")
            delta = (now - last).total_seconds()
            return max(0.0, delta)

    def is_idle(self, tenant_id: TenantId, threshold_seconds: int | float) -> bool:
        """Return True if `tenant_id` has been inactive for at least `threshold_seconds`."""
        return self.idle_seconds(tenant_id) >= threshold_seconds


__all__ = ["TrafficTracker"]
