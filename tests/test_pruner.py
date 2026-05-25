"""Tests for the pruner daemon.

Strategy: never rely on the scheduler firing on its own (would make
tests flaky). Instead, construct the pruner, drive ticks via
tick_once(), and verify state transitions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from neuro_paging import ContextTags, TimeBucket
from neuro_paging.daemons.pruner import Pruner, PrunerConfig
from neuro_paging.daemons.types import PowerSnapshot
from neuro_paging.memory.manager import MemoryManager
from neuro_paging.memory.types import Tier

# ── Helpers ──────────────────────────────────────────────────────────────────


class _ScriptedPower:
    """Test double: returns whatever snapshot you give it."""

    def __init__(self, snap: PowerSnapshot) -> None:
        self.snap = snap

    def snapshot(self) -> PowerSnapshot:
        return self.snap


def _ok_snap() -> PowerSnapshot:
    return PowerSnapshot(
        battery_pct=100, is_charging=True, is_idle=True, is_foreground_app_active=False
    )


def _make_manager(tmp_path):
    return MemoryManager(data_dir=tmp_path / "np")


def _insert_cold_l2_memory(mgr: MemoryManager, text: str, days_old: int) -> str:
    """Helper: insert directly into L2, then backdate last_touch so it looks cold."""
    ctx = ContextTags.now(time_bucket=TimeBucket.EVENING)
    mid = mgr.insert(text, ctx, tier=Tier.L2)
    # Reach behind the API for the test setup — we want this memory to appear
    # to have been untouched for `days_old`.
    old_ts = (datetime.now(UTC) - timedelta(days=days_old)).isoformat()
    with mgr._l2._metadata._lock:
        mgr._l2._metadata._conn.execute(
            "UPDATE memories SET last_touch = ? WHERE memory_id = ?",
            (old_ts, mid),
        )
    return mid


# ── Construction ─────────────────────────────────────────────────────────────


class TestConstruction:
    def test_starts_not_running(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pruner = Pruner(mgr)
        assert pruner.is_running is False
        mgr.close()

    def test_config_defaults(self, tmp_path):
        pruner = Pruner(_make_manager(tmp_path))
        cfg = pruner.config
        assert cfg.interval_seconds == 3600
        assert cfg.cold_threshold_seconds == 14 * 24 * 3600
        assert cfg.max_demotions_per_tick == 32

    def test_custom_config_respected(self, tmp_path):
        custom = PrunerConfig(
            interval_seconds=60,
            cold_threshold_seconds=3600,
            max_demotions_per_tick=4,
        )
        pruner = Pruner(_make_manager(tmp_path), config=custom)
        assert pruner.config.max_demotions_per_tick == 4


# ── Tick: empty + happy path ─────────────────────────────────────────────────


class TestTickEmpty:
    def test_tick_on_empty_manager_returns_zero(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pruner = Pruner(mgr, power=_ScriptedPower(_ok_snap()))
        assert pruner.tick_once() == 0
        assert pruner.stats().demotions_total == 0
        mgr.close()

    def test_tick_advances_counters(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pruner = Pruner(mgr, power=_ScriptedPower(_ok_snap()))
        pruner.tick_once()
        assert pruner.stats().ticks_total == 1
        assert pruner.stats().last_tick_at is not None
        mgr.close()


class TestTickDemotes:
    def test_demotes_cold_memories(self, tmp_path):
        mgr = _make_manager(tmp_path)
        # 3 cold memories (15 days untouched) + 2 fresh ones
        cold_ids = [_insert_cold_l2_memory(mgr, f"cold {i}", days_old=15) for i in range(3)]
        for i in range(2):
            mgr.insert(f"fresh {i}", ContextTags.now(), tier=Tier.L2)

        pruner = Pruner(mgr, power=_ScriptedPower(_ok_snap()))
        n = pruner.tick_once()

        assert n == 3
        stats = mgr.get_stats()
        # Cold ones moved to L3; fresh stayed in L2
        assert stats.l3_count == 3
        assert stats.l2_count == 2
        # All cold ids are now in L3 (or at least not in L2)
        for mid in cold_ids:
            assert not mgr._l2.contains(mid)
        mgr.close()

    def test_respects_max_per_tick(self, tmp_path):
        mgr = _make_manager(tmp_path)
        for i in range(10):
            _insert_cold_l2_memory(mgr, f"cold {i}", days_old=20)

        cfg = PrunerConfig(max_demotions_per_tick=4)
        pruner = Pruner(mgr, config=cfg, power=_ScriptedPower(_ok_snap()))

        n = pruner.tick_once()
        assert n == 4
        # Another tick should pick up the next 4
        n2 = pruner.tick_once()
        assert n2 == 4
        # And a third the remaining 2
        n3 = pruner.tick_once()
        assert n3 == 2
        mgr.close()

    def test_fresh_memories_not_demoted(self, tmp_path):
        mgr = _make_manager(tmp_path)
        for i in range(5):
            mgr.insert(f"fresh {i}", ContextTags.now(), tier=Tier.L2)

        pruner = Pruner(mgr, power=_ScriptedPower(_ok_snap()))
        n = pruner.tick_once()
        assert n == 0
        assert mgr.get_stats().l2_count == 5
        mgr.close()


# ── Power gating ─────────────────────────────────────────────────────────────


class TestPowerGate:
    def test_skips_on_low_battery(self, tmp_path):
        mgr = _make_manager(tmp_path)
        _insert_cold_l2_memory(mgr, "cold", days_old=20)

        low_battery = PowerSnapshot(
            battery_pct=30, is_charging=False, is_idle=True, is_foreground_app_active=False
        )
        pruner = Pruner(mgr, power=_ScriptedPower(low_battery))

        n = pruner.tick_once()
        assert n == 0
        assert pruner.stats().ticks_skipped_power == 1
        assert pruner.stats().last_skip_reason is not None
        assert "battery" in pruner.stats().last_skip_reason
        mgr.close()

    def test_low_battery_ok_if_charging(self, tmp_path):
        mgr = _make_manager(tmp_path)
        _insert_cold_l2_memory(mgr, "cold", days_old=20)

        low_but_charging = PowerSnapshot(
            battery_pct=20, is_charging=True, is_idle=True, is_foreground_app_active=False
        )
        pruner = Pruner(mgr, power=_ScriptedPower(low_but_charging))
        assert pruner.tick_once() == 1
        mgr.close()

    def test_skips_when_foreground_active(self, tmp_path):
        mgr = _make_manager(tmp_path)
        _insert_cold_l2_memory(mgr, "cold", days_old=20)

        fg = PowerSnapshot(
            battery_pct=100, is_charging=True, is_idle=True, is_foreground_app_active=True
        )
        pruner = Pruner(mgr, power=_ScriptedPower(fg))
        assert pruner.tick_once() == 0
        assert pruner.stats().last_skip_reason == "foreground_app_active"
        mgr.close()

    def test_skips_when_not_idle(self, tmp_path):
        mgr = _make_manager(tmp_path)
        _insert_cold_l2_memory(mgr, "cold", days_old=20)

        active = PowerSnapshot(
            battery_pct=100, is_charging=True, is_idle=False, is_foreground_app_active=False
        )
        pruner = Pruner(mgr, power=_ScriptedPower(active))
        assert pruner.tick_once() == 0
        assert pruner.stats().last_skip_reason == "device_not_idle"
        mgr.close()


# ── Lifecycle ────────────────────────────────────────────────────────────────


class TestLifecycle:
    def test_start_stop_idempotent(self, tmp_path):
        mgr = _make_manager(tmp_path)
        cfg = PrunerConfig(interval_seconds=3600)  # long interval — won't fire in test
        pruner = Pruner(mgr, config=cfg)

        pruner.start()
        assert pruner.is_running
        pruner.start()  # idempotent
        assert pruner.is_running

        pruner.stop()
        assert not pruner.is_running
        pruner.stop()  # idempotent

        mgr.close()


# ── Stats ────────────────────────────────────────────────────────────────────


class TestStats:
    def test_recent_demotions_capped_at_20(self, tmp_path):
        mgr = _make_manager(tmp_path)
        for i in range(30):
            _insert_cold_l2_memory(mgr, f"cold {i}", days_old=20)

        cfg = PrunerConfig(max_demotions_per_tick=100)
        pruner = Pruner(mgr, config=cfg, power=_ScriptedPower(_ok_snap()))
        pruner.tick_once()

        assert len(pruner.stats().recent_demotions) <= 20
        mgr.close()

    def test_stats_snapshot_serialisable(self, tmp_path):
        import json

        mgr = _make_manager(tmp_path)
        pruner = Pruner(mgr, power=_ScriptedPower(_ok_snap()))
        pruner.tick_once()

        snap = pruner.stats().snapshot()
        # Must be JSON-serialisable (dashboard streams it)
        json.dumps(snap)
        mgr.close()
