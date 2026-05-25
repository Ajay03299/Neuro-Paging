"""Pruner daemon — demote stale L2 memories to L3.

The deck's L2 spec: 'cold for 14 days → demote to L3'. This file
makes that real. Runs every PRUNE_INTERVAL_SECONDS (default 1 hour),
finds L2 memories not touched since `now - cold_threshold`, calls
manager.demote() on each.

Power gating: respects PowerStateProvider. Skip the tick if:
  - battery_pct < min_battery_pct AND not charging
  - foreground app is active (don't compete for cycles)
  - device is not idle (user is interacting)

Bounded work per tick: max_demotions_per_tick (default 32). Avoids
holding locks too long; the next tick picks up where we left off.

Lifecycle:
  pruner = Pruner(manager, power=power_provider)
  pruner.start()    # launches BackgroundScheduler
  ...
  pruner.stop()     # graceful shutdown, waits for any in-flight tick

For tests / on-demand runs, tick_once() does one synchronous pass
without the scheduler. This is also how the dashboard's 'Run pruner
now' button works.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from neuro_paging.daemons.types import PowerStateProvider, _DefaultPowerState
from neuro_paging.memory.manager import MemoryManager
from neuro_paging.memory.types import Tier

# ── Configuration ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PrunerConfig:
    """Pruner tuning. All values overridable at construction time."""

    # How often to wake up and check (seconds)
    interval_seconds: int = 3600  # 1 hour

    # A memory is "cold" if not touched in this many seconds
    cold_threshold_seconds: int = 14 * 24 * 3600  # 14 days

    # Cap per tick so we don't hold locks too long
    max_demotions_per_tick: int = 32

    # Skip the tick if battery is below this AND not charging
    min_battery_pct: int = 50

    # Skip the tick if foreground app is active
    skip_when_foreground: bool = True

    # Skip the tick if device is not idle
    require_idle: bool = True


# ── Stats ────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PrunerStats:
    """Rolling stats for observability. Powers the dashboard tile."""

    ticks_total: int = 0
    ticks_skipped_power: int = 0
    demotions_total: int = 0
    last_tick_at: datetime | None = None
    last_demoted_count: int = 0
    last_skip_reason: str | None = None
    errors_total: int = 0
    recent_demotions: list[str] = field(default_factory=list)  # last N memory_ids

    def snapshot(self) -> dict:
        return {
            "ticks_total": self.ticks_total,
            "ticks_skipped_power": self.ticks_skipped_power,
            "demotions_total": self.demotions_total,
            "last_tick_at": self.last_tick_at.isoformat() if self.last_tick_at else None,
            "last_demoted_count": self.last_demoted_count,
            "last_skip_reason": self.last_skip_reason,
            "errors_total": self.errors_total,
            "recent_demotions": list(self.recent_demotions),
        }


# ── The pruner ───────────────────────────────────────────────────────────────


class Pruner:
    """Background daemon that demotes cold L2 memories to L3.

    Thread-safe. Start once at app boot; call stop() at shutdown.
    """

    def __init__(
        self,
        manager: MemoryManager,
        config: PrunerConfig | None = None,
        power: PowerStateProvider | None = None,
    ) -> None:
        self._manager = manager
        self._config = config or PrunerConfig()
        self._power = power or _DefaultPowerState()
        self._stats = PrunerStats()
        self._scheduler: BackgroundScheduler | None = None
        self._lock = threading.Lock()
        self._running = False

        logger.debug(
            f"Pruner initialised — interval={self._config.interval_seconds}s "
            f"cold_threshold={self._config.cold_threshold_seconds}s "
            f"max_per_tick={self._config.max_demotions_per_tick}"
        )

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background scheduler. Idempotent."""
        with self._lock:
            if self._running:
                logger.warning("Pruner.start() called but already running")
                return

            self._scheduler = BackgroundScheduler(daemon=True)
            self._scheduler.add_job(
                self._safe_tick,
                trigger="interval",
                seconds=self._config.interval_seconds,
                next_run_time=datetime.now(UTC) + timedelta(seconds=self._config.interval_seconds),
                id="pruner_tick",
                max_instances=1,  # never overlap ticks
                coalesce=True,  # if we fall behind, only run once
            )
            self._scheduler.start()
            self._running = True
            logger.info(f"Pruner started — first tick in {self._config.interval_seconds}s")

    def stop(self) -> None:
        """Stop the scheduler. Waits for any in-flight tick. Idempotent."""
        with self._lock:
            if not self._running or self._scheduler is None:
                return
            self._scheduler.shutdown(wait=True)
            self._scheduler = None
            self._running = False
            logger.info("Pruner stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Tick implementations ─────────────────────────────────────────────────

    def _safe_tick(self) -> None:
        """Wrap tick_once() so a thrown exception doesn't kill the scheduler."""
        try:
            self.tick_once()
        except Exception as e:  # noqa: BLE001
            self._stats.errors_total += 1
            logger.exception(f"Pruner tick failed: {e}")

    def tick_once(self) -> int:
        """Run a single prune pass synchronously. Returns # demotions done.

        Public so tests + the dashboard can trigger ticks manually.
        """
        self._stats.ticks_total += 1
        self._stats.last_tick_at = datetime.now(UTC)
        self._stats.last_demoted_count = 0
        self._stats.last_skip_reason = None

        # ── Power gate ──
        skip_reason = self._should_skip()
        if skip_reason is not None:
            self._stats.ticks_skipped_power += 1
            self._stats.last_skip_reason = skip_reason
            logger.debug(f"Pruner skipping tick: {skip_reason}")
            return 0

        # ── Find cold candidates ──
        cutoff = datetime.now(UTC) - timedelta(seconds=self._config.cold_threshold_seconds)
        candidates = self._manager._l2._metadata.find_cold(Tier.L2, older_than=cutoff)
        if not candidates:
            logger.trace("Pruner found no cold candidates")
            return 0

        # ── Demote up to max_per_tick ──
        batch = candidates[: self._config.max_demotions_per_tick]
        demoted = 0
        for mem_id in batch:
            try:
                self._manager.demote(mem_id)
                demoted += 1
                # Track last few for dashboard tile (cap to 20)
                self._stats.recent_demotions.append(str(mem_id))
                if len(self._stats.recent_demotions) > 20:
                    self._stats.recent_demotions = self._stats.recent_demotions[-20:]
            except Exception as e:  # noqa: BLE001
                self._stats.errors_total += 1
                logger.warning(f"Pruner demote failed for {mem_id}: {e}")

        self._stats.demotions_total += demoted
        self._stats.last_demoted_count = demoted
        logger.info(
            f"Pruner tick complete: demoted={demoted} remaining_cold={len(candidates) - demoted}"
        )
        return demoted

    # ── Power gate ───────────────────────────────────────────────────────────

    def _should_skip(self) -> str | None:
        """Return a reason string if the tick should be skipped, else None."""
        snap = self._power.snapshot()

        if snap.battery_pct < self._config.min_battery_pct and not snap.is_charging:
            return f"battery_low ({snap.battery_pct}%, not charging)"

        if self._config.skip_when_foreground and snap.is_foreground_app_active:
            return "foreground_app_active"

        if self._config.require_idle and not snap.is_idle:
            return "device_not_idle"

        return None

    # ── Observability ────────────────────────────────────────────────────────

    def stats(self) -> PrunerStats:
        return self._stats

    @property
    def config(self) -> PrunerConfig:
        return self._config
