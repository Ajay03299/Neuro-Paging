"""Neuro-Paging — context-aware adaptive memory for mobile agentic systems.

An operating system for AI memory. Three tiers, context-aware routing,
predictive prefetch. 100% on-device. Apache-2.0.

Built for Samsung ennovateX AX Hackathon 2026 by team ByteMe.
"""

from neuro_paging.context import BatteryState, ContextTags, TimeBucket
from neuro_paging.daemons import PowerStateProvider, Pruner, PrunerConfig
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
from neuro_paging.pipeline import AssembledContext, MemoryAgent

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
    "PowerStateProvider",
    "Provenance",
    "Pruner",
    "PrunerConfig",
    "Scorer",
    "Tier",
    "TierStats",
    "TimeBucket",
    "AssembledContext",
    "MemoryAgent",
]
