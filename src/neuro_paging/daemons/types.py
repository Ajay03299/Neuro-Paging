"""Shared types for daemons.

PowerStateProvider — the contract Christine implements to surface
device telemetry (battery percent, charging state, idle state).
Daemons consult this before doing work so we don't drain the
user's battery in the foreground.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class PowerSnapshot:
    """Single observation of the device's power + idle state.

    All fields are best-effort. If a platform can't provide one,
    the provider returns a safe default (e.g. is_idle=False, which
    keeps daemons conservative).
    """

    battery_pct: int  # 0-100
    is_charging: bool
    is_idle: bool  # screen off, no recent input
    is_foreground_app_active: bool  # the host app is on-screen


class PowerStateProvider(Protocol):
    """The contract Christine's sensor layer implements.

    Daemons call .snapshot() before each tick of work. If the snapshot
    says 'not OK to run', the daemon skips this tick.
    """

    def snapshot(self) -> PowerSnapshot:
        """Return the current power + idle observation."""
        ...


class _DefaultPowerState:
    """Always-on stub for dev + tests.

    Says: 100% battery, charging, idle, app backgrounded. So every
    daemon tick will pass the power-gate check. Replaced by Christine's
    real sensor wiring at production wire-up time.
    """

    def snapshot(self) -> PowerSnapshot:
        return PowerSnapshot(
            battery_pct=100,
            is_charging=True,
            is_idle=True,
            is_foreground_app_active=False,
        )
