"""Neuro-Paging — context-aware adaptive memory for mobile agentic systems.

An operating system for AI memory. Three tiers, context-aware routing,
predictive prefetch. 100% on-device. Apache-2.0.

Built for Samsung ennovateX AX Hackathon 2026 by team ByteMe.
"""

from neuro_paging.context import BatteryState, ContextTags, TimeBucket
from neuro_paging.memory import (
    Hit,
    Memory,
    MemoryId,
    MemoryManager,
    Provenance,
    Scorer,
    Tier,
    TierStats,
)

__version__ = "0.1.0"
__author__ = "Ajay Javali, Christine R"
__license__ = "Apache-2.0"

__all__ = [
    "BatteryState",
    "ContextTags",
    "Hit",
    "Memory",
    "MemoryId",
    "MemoryManager",
    "Provenance",
    "Scorer",
    "Tier",
    "TierStats",
    "TimeBucket",
]
