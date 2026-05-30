"""Routing — the context-aware scorer that re-ranks retrieval candidates.

The substrate defines a Scorer protocol (see memory.manager.Scorer).
This package provides the real implementation.

- ContextAwareScorer: the deck's
      Score = α·cos(e_m, e_q) + β·ctxSim(tags_m, c) + γ·log(1+freq)·decay(Δt)
  with fixed, hand-tuned weights. Online α/β/γ learning is future work.

The default stub scorer (word-overlap) lives in memory.manager and is
used for tests + dev so the core never depends on the routing layer.
"""

from neuro_paging.routing.scorer import ContextAwareScorer, ScorerWeights

__all__ = ["ContextAwareScorer", "ScorerWeights"]
