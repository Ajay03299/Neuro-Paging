"""Context types — what 'context' means in Neuro-Paging.

Christine's sensors produce these. The router consumes them. The manager
attaches them to every stored memory.

These types are the lingua franca between layers. Don't change them
without bumping the API contract version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class TimeBucket(StrEnum):
    """Coarse-grained time-of-day buckets — what the router conditions on.

    We don't use raw timestamps for ctxSim because 'memories from 6:55 PM'
    should match a 7:05 PM query. Bucketing handles this cleanly.
    """

    EARLY_MORNING = "early_morning"  # 04:00–07:59
    MORNING = "morning"  # 08:00–11:59
    AFTERNOON = "afternoon"  # 12:00–16:59
    EVENING = "evening"  # 17:00–20:59
    NIGHT = "night"  # 21:00–03:59

    @classmethod
    def from_datetime(cls, dt: datetime) -> TimeBucket:
        h = dt.hour
        if 4 <= h < 8:
            return cls.EARLY_MORNING
        if 8 <= h < 12:
            return cls.MORNING
        if 12 <= h < 17:
            return cls.AFTERNOON
        if 17 <= h < 21:
            return cls.EVENING
        return cls.NIGHT


class BatteryState(StrEnum):
    """Coarse battery state — drives prefetcher gating."""

    CRITICAL = "critical"  # <15% and discharging
    LOW = "low"  # 15–40% and discharging
    NORMAL = "normal"  # 40–80% or charging
    HIGH = "high"  # >80%
    CHARGING = "charging"  # plugged in


@dataclass(frozen=True, slots=True)
class ContextTags:
    """The 'where/when/what' of a memory or a query.

    Frozen + slots → fast hashing, low memory overhead, safe to use as
    dict keys for the per-context access counters.

    Christine's sensors populate this. The router scores against it.
    Stored alongside every memory in L2/L3 metadata.
    """

    # Temporal
    timestamp: datetime
    time_bucket: TimeBucket

    # Spatial — coarse-grained for privacy (e.g. "home", "office", "transit")
    # Never raw lat/lng in memory unless user explicitly opted in.
    location: str | None = None

    # Device — what app the user is in right now
    foreground_app: str | None = None

    # Calendar — current event title if any, else None
    calendar_event: str | None = None

    # Battery — gates daemons and prefetcher
    battery_state: BatteryState = BatteryState.NORMAL
    battery_pct: int = 100  # 0-100; informational

    # Free-form semantic tags — e.g. ["food", "italian", "weeknight"]
    # Used both by the router's ctxSim and by the consolidator's cluster grouping.
    semantic_tags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def now(cls, **overrides) -> ContextTags:
        """Build a ContextTags for the current moment.

        Convenience for tests and quick demos. Real use goes through
        Christine's sensor providers.
        """
        ts = datetime.now(UTC)
        defaults: dict = {
            "timestamp": ts,
            "time_bucket": TimeBucket.from_datetime(ts),
        }
        defaults.update(overrides)
        return cls(**defaults)

    def with_tags(self, *tags: str) -> ContextTags:
        """Return a new ContextTags with extra semantic tags appended.

        Frozen dataclasses don't allow mutation, so we return a copy.
        """
        return ContextTags(
            timestamp=self.timestamp,
            time_bucket=self.time_bucket,
            location=self.location,
            foreground_app=self.foreground_app,
            calendar_event=self.calendar_event,
            battery_state=self.battery_state,
            battery_pct=self.battery_pct,
            semantic_tags=(*self.semantic_tags, *tags),
        )
