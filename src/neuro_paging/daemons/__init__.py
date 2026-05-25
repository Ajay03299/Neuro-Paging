"""Background daemons: pruner, consolidator runner.

Daemons run in the background under APScheduler. They mutate the
MemoryManager but never expose new public APIs to Christine's layer
beyond start()/stop() — the manager is still the only contract.

Power gating: every daemon checks a PowerStateProvider before doing
work. The default stub always says 'OK to run'. Christine plugs in
real iOS/Android sensors via the same protocol.
"""

from neuro_paging.daemons.pruner import Pruner, PrunerConfig, PrunerStats
from neuro_paging.daemons.types import PowerStateProvider, _DefaultPowerState

__all__ = [
    "PowerStateProvider",
    "Pruner",
    "PrunerConfig",
    "PrunerStats",
    "_DefaultPowerState",
]
