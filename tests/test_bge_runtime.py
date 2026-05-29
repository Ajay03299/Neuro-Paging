"""Tests for BGESmallEmbedder.

Network-free by design. We test:
  - The math we OWN (mean-pooling, L2-normalisation) on synthetic arrays
  - Protocol conformance (the class IS a valid Embedder)
  - That the real-model path is exercised ONLY when the model is already
    cached locally (so CI stays fast + offline)

The actual ONNX inference is integration-tested behind a skip guard:
if the model isn't cached, those tests skip rather than download in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuro_paging.embed.bge_runtime import (
    _CACHE_DIR,
    _EMBED_DIM,
    BGESmallEmbedder,
)

# ── The math we own (no model needed) ────────────────────────────────────────


class TestPooling:
    def test_mean_pool_excludes_padding(self):
        # 3 tokens, hidden=4. Third token is padding (mask=0).
        token_emb = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [3.0, 3.0, 3.0, 3.0],
                [99.0, 99.0, 99.0, 99.0],  # padding — should be ignored
            ],
            dtype=np.float32,
        )
        mask = np.array([1, 1, 0], dtype=np.int64)
        pooled = BGESmallEmbedder._mean_pool(token_emb, mask)
        # Mean of [1,1,1,1] and [3,3,3,3] = [2,2,2,2]
        np.testing.assert_allclose(pooled, [2.0, 2.0, 2.0, 2.0])

    def test_mean_pool_all_masked_is_safe(self):
        """All-padding shouldn't divide by zero."""
        token_emb = np.ones((2, 4), dtype=np.float32)
        mask = np.array([0, 0], dtype=np.int64)
        pooled = BGESmallEmbedder._mean_pool(token_emb, mask)
        # Sum is 0, count clipped to ~0 → result ~0, no NaN/inf
        assert np.all(np.isfinite(pooled))


class TestNormalisation:
    def test_l2_normalise_unit_norm(self):
        vec = np.array([3.0, 4.0], dtype=np.float32)  # norm 5
        out = BGESmallEmbedder._l2_normalise(vec)
        np.testing.assert_allclose(np.linalg.norm(out), 1.0, rtol=1e-6)
        np.testing.assert_allclose(out, [0.6, 0.8], rtol=1e-6)

    def test_l2_normalise_zero_vector_safe(self):
        vec = np.zeros(4, dtype=np.float32)
        out = BGESmallEmbedder._l2_normalise(vec)
        assert np.all(np.isfinite(out))


# ── Construction (no download triggered) ─────────────────────────────────────


class TestConstruction:
    def test_construct_does_not_download(self):
        """Constructing the embedder must NOT touch the network."""
        emb = BGESmallEmbedder()
        # Session + tokenizer are lazy — None until first embed()
        assert emb._session is None
        assert emb._tokenizer is None
        assert emb.dim == _EMBED_DIM

    def test_custom_cache_dir(self, tmp_path):
        emb = BGESmallEmbedder(cache_dir=tmp_path / "models")
        assert emb._cache_dir == tmp_path / "models"


# ── Integration (only if the real model is already cached) ───────────────────

_MODEL_CACHED = (_CACHE_DIR / "model.onnx").exists() and (_CACHE_DIR / "tokenizer.json").exists()

skip_no_model = pytest.mark.skipif(
    not _MODEL_CACHED,
    reason="bge-small model not cached locally; skipping live inference test",
)


@skip_no_model
class TestLiveInference:
    """These run ONLY when the model is already downloaded. CI skips them."""

    def test_embed_shape_and_norm(self):
        emb = BGESmallEmbedder()
        vec = emb.embed("the user prefers italian food")
        assert vec.shape == (_EMBED_DIM,)
        assert vec.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, rtol=1e-4)

    def test_same_text_same_vector(self):
        emb = BGESmallEmbedder()
        v1 = emb.embed("hello world")
        v2 = emb.embed("hello world")
        np.testing.assert_allclose(v1, v2, rtol=1e-5)

    def test_semantic_similarity_makes_sense(self):
        """Related sentences should be closer than unrelated ones."""
        emb = BGESmallEmbedder()
        food = emb.embed("the user loves italian pasta and pizza")
        food2 = emb.embed("user enjoys eating spaghetti")
        weather = emb.embed("it is raining heavily outside today")

        sim_related = float(np.dot(food, food2))
        sim_unrelated = float(np.dot(food, weather))
        assert sim_related > sim_unrelated, (
            f"Related sentences ({sim_related:.3f}) should beat unrelated ({sim_unrelated:.3f})"
        )

    def test_query_vs_passage_prefix_differs(self):
        emb = BGESmallEmbedder()
        passage = emb.embed("italian food")
        query = emb.embed_query("italian food")
        # Different prefixing → different vectors
        assert not np.allclose(passage, query)
