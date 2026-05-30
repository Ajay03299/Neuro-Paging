"""Tests for MemoryAgent — the orchestration layer.

Network-free: uses the manager's default stub embedder/scorer. We test the
orchestration logic (remember/recall/respond, context assembly), not the
embedder quality (that's tested in test_bge_runtime + test_scorer).
"""

from __future__ import annotations

from neuro_paging import ContextTags
from neuro_paging.memory.types import Tier
from neuro_paging.pipeline import AssembledContext, MemoryAgent
from neuro_paging.pipeline.agent import _EchoGenerator


def _agent(tmp_path, **kw) -> MemoryAgent:
    return MemoryAgent(data_dir=tmp_path / "agent", **kw)


# ── Remember (write path) ────────────────────────────────────────────────────


class TestRemember:
    def test_remember_returns_id(self, tmp_path):
        agent = _agent(tmp_path)
        mid = agent.remember("user likes window seats", ContextTags.now())
        assert mid is not None
        agent.close()

    def test_remembered_memory_is_findable(self, tmp_path):
        agent = _agent(tmp_path)
        mid = agent.remember("unique-needle-phrase here", ContextTags.now())
        result = agent.recall("needle", ContextTags.now(), k=10)
        found = {h.memory_id for h in result.hits}
        assert mid in found
        agent.close()

    def test_remember_into_specific_tier(self, tmp_path):
        agent = _agent(tmp_path)
        mid = agent.remember("archived fact", ContextTags.now(), tier=Tier.L3)
        mem = agent.manager.get(mid)
        assert mem is not None and mem.tier == Tier.L3
        agent.close()


# ── Recall (read path) ───────────────────────────────────────────────────────


class TestRecall:
    def test_recall_returns_assembled_context(self, tmp_path):
        agent = _agent(tmp_path)
        agent.remember("the cat sat on the mat", ContextTags.now())
        result = agent.recall("cat", ContextTags.now(), k=5)
        assert isinstance(result, AssembledContext)
        assert result.query == "cat"
        agent.close()

    def test_recall_empty_when_nothing_stored(self, tmp_path):
        agent = _agent(tmp_path)
        result = agent.recall("anything", ContextTags.now(), k=5)
        assert result.is_empty
        assert result.context_block == ""
        agent.close()

    def test_context_block_lists_hits_with_tier_and_score(self, tmp_path):
        agent = _agent(tmp_path)
        agent.remember("alpha memory content", ContextTags.now())
        agent.remember("beta memory content", ContextTags.now())
        result = agent.recall("memory", ContextTags.now(), k=5)
        block = result.context_block
        # Block should mention tiers and be non-empty
        assert "Relevant memories" in block
        assert any(t in block for t in ("L1", "L2", "L3"))
        # One numbered line per hit
        assert block.count("\n") >= len(result.hits)
        agent.close()

    def test_recall_respects_k(self, tmp_path):
        agent = _agent(tmp_path)
        for i in range(10):
            agent.remember(f"memory number {i} content", ContextTags.now())
        result = agent.recall("memory", ContextTags.now(), k=3)
        assert len(result.hits) <= 3
        agent.close()


# ── Respond (recall + generate) ──────────────────────────────────────────────


class TestRespond:
    def test_respond_with_echo_generator_returns_context(self, tmp_path):
        agent = _agent(tmp_path)
        agent.remember("the user is vegetarian", ContextTags.now())
        answer = agent.respond("dietary preference", ContextTags.now())
        # Echo generator surfaces the assembled context
        assert "vegetarian" in answer or "Generator stub" in answer
        agent.close()

    def test_respond_no_memories_is_graceful(self, tmp_path):
        agent = _agent(tmp_path)
        answer = agent.respond("anything at all", ContextTags.now())
        assert "no relevant memories" in answer.lower()
        agent.close()

    def test_custom_generator_is_used(self, tmp_path):
        class _UpperGen:
            def generate(self, query: str, context_block: str) -> str:
                return f"RESPONSE FOR {query.upper()}"

        agent = _agent(tmp_path, generator=_UpperGen())
        agent.remember("something", ContextTags.now())
        answer = agent.respond("test query", ContextTags.now())
        assert answer == "RESPONSE FOR TEST QUERY"
        agent.close()


# ── Echo generator unit ──────────────────────────────────────────────────────


class TestEchoGenerator:
    def test_empty_context_block(self):
        gen = _EchoGenerator()
        out = gen.generate("my query", "")
        assert "no relevant memories" in out.lower()

    def test_nonempty_context_block_passthrough(self):
        gen = _EchoGenerator()
        out = gen.generate("q", "Relevant memories:\n1. [L1] hi")
        assert "Relevant memories" in out


# ── Stats passthrough ────────────────────────────────────────────────────────


class TestStats:
    def test_get_stats_reflects_remembered(self, tmp_path):
        agent = _agent(tmp_path)
        for i in range(3):
            agent.remember(f"mem {i}", ContextTags.now(), tier=Tier.L1)
        stats = agent.get_stats()
        assert stats.total_count == 3
        agent.close()
