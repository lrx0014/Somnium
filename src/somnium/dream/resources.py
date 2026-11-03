"""System and environment resource policy for Dream Engine scheduling (M6.4).

Per `design.md` §7 and `Milestones.md` M6.4:
- Development profile on Apple Silicon: train only when on AC power and under
  acceptable memory pressure / thermal state.
- Inspects macOS power management (`pmset`) and Mach virtual memory (`vm_stat`).
- Gracefully degrades to permissive defaults when running on non-Darwin platforms
  or in sandboxed environments.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from typing import Protocol, runtime_checkable

_BATT_AC_RE = re.compile(r"Now drawing from '(AC Power|Battery Power)'", re.IGNORECASE)
_VM_PAGE_SIZE_RE = re.compile(r"page size of (\d+) bytes", re.IGNORECASE)
_VM_PAGE_RE = re.compile(r"^Pages\s+([^:]+):\s+(\d+)\.", re.MULTILINE)


@runtime_checkable
class ResourcePolicy(Protocol):
    """Protocol for checking system resource availability before launching training."""

    def check_resources(
        self,
        *,
        require_ac_power: bool = True,
        max_memory_pressure_pct: int = 85,
    ) -> tuple[bool, str | None]:
        """Return (True, None) if resources permit training, or (False, reason) if deferred."""
        ...


class SystemResourcePolicy:
    """Live operating system resource checker for macOS and Unix development machines (M6.4)."""

    def __init__(self, *, enable_subprocess: bool = True) -> None:
        self._enable_subprocess = enable_subprocess
        self._is_darwin = platform.system() == "Darwin"

    def is_ac_power(self) -> tuple[bool, str]:
        """Check if the system is currently connected to AC power."""
        if not self._is_darwin or not self._enable_subprocess or not shutil.which("pmset"):
            return True, "AC Power (assumed desktop or non-Darwin platform)"

        try:
            proc = subprocess.run(
                ["pmset", "-g", "batt"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            match = _BATT_AC_RE.search(proc.stdout)
            if match:
                state = match.group(1).lower()
                if "battery" in state:
                    return False, "Battery Power"
                return True, "AC Power"
            if "InternalBattery" not in proc.stdout:
                return True, "AC Power (desktop/no battery)"
        except Exception as exc:
            return True, f"AC Power (pmset error: {exc})"

        return True, "AC Power"

    def memory_pressure_pct(self) -> float:
        """Estimate current memory usage percentage using vm_stat on macOS."""
        if not self._is_darwin or not self._enable_subprocess or not shutil.which("vm_stat"):
            return 50.0

        try:
            proc = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            counts: dict[str, int] = {}
            for m in _VM_PAGE_RE.finditer(proc.stdout):
                counts[m.group(1).strip().lower()] = int(m.group(2))

            free = counts.get("free", 0) + counts.get("speculative", 0)
            active = counts.get("active", 0)
            inactive = counts.get("inactive", 0)
            wired = counts.get("wired down", 0)
            occupied_compressor = counts.get("occupied by compressor", 0)

            used = active + wired + occupied_compressor
            total = free + inactive + used
            if total > 0:
                return round((used / total) * 100.0, 1)
        except Exception:
            return 50.0

        return 50.0

    def check_resources(
        self,
        *,
        require_ac_power: bool = True,
        max_memory_pressure_pct: int = 85,
    ) -> tuple[bool, str | None]:
        """Verify AC power and memory pressure against configured thresholds."""
        if require_ac_power:
            on_ac, source = self.is_ac_power()
            if not on_ac:
                return False, f"Device is running on battery ({source}); AC power required."

        pressure = self.memory_pressure_pct()
        if pressure > max_memory_pressure_pct:
            return (
                False,
                f"Memory pressure {pressure:.1f}% exceeds threshold {max_memory_pressure_pct}%.",
            )

        return True, None


class StaticResourcePolicy:
    """Configurable resource checker for deterministic tests and synthetic scenarios."""

    def __init__(
        self,
        *,
        on_ac_power: bool = True,
        memory_pressure: float = 40.0,
    ) -> None:
        self.on_ac_power = on_ac_power
        self.memory_pressure = memory_pressure

    def check_resources(
        self,
        *,
        require_ac_power: bool = True,
        max_memory_pressure_pct: int = 85,
    ) -> tuple[bool, str | None]:
        """Check against injected static values."""
        if require_ac_power and not self.on_ac_power:
            return False, "Device is running on battery; AC power required."
        if self.memory_pressure > max_memory_pressure_pct:
            return (
                False,
                f"Memory pressure {self.memory_pressure:.1f}% exceeds threshold "
                f"{max_memory_pressure_pct}%.",
            )
        return True, None


__all__ = ["ResourcePolicy", "StaticResourcePolicy", "SystemResourcePolicy"]
