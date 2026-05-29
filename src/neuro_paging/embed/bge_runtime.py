"""bge-small-en-v1.5 embeddings via ONNX Runtime.

Why ONNX Runtime (and not sentence-transformers)?
--------------------------------------------------
- No torch dependency. ONNX Runtime is ~90 MB vs torch's ~2 GB. That
  matters for the on-device / mobile target this whole project is about.
- Explicit pipeline: we tokenize, run the transformer forward pass, then
  do mean-pooling + L2-normalisation OURSELVES. The embedding step is
  transparent, not a black box.
- Runs anywhere ONNX runs (CPU, CoreML, mobile NNAPI), which aligns with
  the '100% on-device' thesis.

The model + tokenizer are fetched from HuggingFace on first use and cached
under ~/.cache/neuro-paging/. If the machine is offline and the model
isn't cached, we raise a clear, actionable error rather than silently
returning garbage.

This class implements the Embedder protocol (see memory.manager.Embedder):
    embed(text: str) -> np.ndarray  # shape (384,), float32, unit-norm

bge-small-en-v1.5 specifics:
- Output dim: 384
- Recommended: prepend a query instruction for retrieval queries. bge
  models were trained with "Represent this sentence for searching
  relevant passages:" prefixing for the QUERY side only. We expose this
  via embed_query() vs embed() (passages/documents use no prefix).
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from loguru import logger

# Model identity
_HF_REPO = "BAAI/bge-small-en-v1.5"
_EMBED_DIM = 384
_MAX_TOKENS = 512  # bge-small context window

# The query-side instruction bge recommends for retrieval (query side only)
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Where we cache the downloaded ONNX model + tokenizer
_CACHE_DIR = Path.home() / ".cache" / "neuro-paging" / "bge-small-en-v1.5"


class BGESmallEmbedder:
    """bge-small-en-v1.5 embedder backed by ONNX Runtime.

    Thread-safe: the ONNX session is created once (lazily) and guarded by
    a lock for the first-use initialisation. Inference itself is
    thread-safe in ONNX Runtime.

    Usage:
        embedder = BGESmallEmbedder()           # nothing downloaded yet
        vec = embedder.embed("user likes pizza")  # downloads on first call
        qv  = embedder.embed_query("what food?")   # query-instruction prefixed

        mgr = MemoryManager(embedder=embedder)   # plug into the substrate
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        device: str = "cpu",
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self._device = device
        self._session = None  # ONNX InferenceSession, lazily created
        self._tokenizer = None  # tokenizers.Tokenizer, lazily created
        self._init_lock = threading.Lock()
        self._dim = _EMBED_DIM

    # ── Lazy initialisation ──────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Download (if needed) + load the model and tokenizer. Idempotent."""
        if self._session is not None and self._tokenizer is not None:
            return

        with self._init_lock:
            # Double-check inside the lock (another thread may have loaded it)
            if self._session is not None and self._tokenizer is not None:
                return

            model_path, tokenizer_path = self._fetch_model_files()

            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as e:
                raise RuntimeError(
                    "BGESmallEmbedder needs the 'ml' extra. Install with:\n"
                    "    uv pip install -e '.[ml]'\n"
                    f"(import failed: {e})"
                ) from e

            providers = ["CPUExecutionProvider"]
            if self._device == "coreml":
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

            self._session = ort.InferenceSession(str(model_path), providers=providers)
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self._tokenizer.enable_truncation(max_length=_MAX_TOKENS)
            self._tokenizer.enable_padding()

            logger.info(
                f"BGESmallEmbedder loaded — model={model_path.name} "
                f"providers={providers} dim={self._dim}"
            )

    def _fetch_model_files(self) -> tuple[Path, Path]:
        """Return (onnx_model_path, tokenizer_path), downloading if needed.

        Raises a clear error if files are missing and can't be fetched.
        """
        model_path = self._cache_dir / "model.onnx"
        tokenizer_path = self._cache_dir / "tokenizer.json"

        if model_path.exists() and tokenizer_path.exists():
            return model_path, tokenizer_path

        # Need to download. Use huggingface_hub if available.
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                "bge-small model not cached and huggingface_hub is not "
                "installed to fetch it. Either:\n"
                "  1. Install the ml extra: uv pip install -e '.[ml]'\n"
                "  2. Manually place model.onnx + tokenizer.json in\n"
                f"     {self._cache_dir}\n"
                f"(import failed: {e})"
            ) from e

        try:
            logger.info(f"Downloading {_HF_REPO} ONNX model (first use)…")
            # bge-small-en-v1.5 ships an ONNX export under onnx/
            dl_model = hf_hub_download(
                repo_id=_HF_REPO,
                filename="onnx/model.onnx",
            )
            dl_tokenizer = hf_hub_download(
                repo_id=_HF_REPO,
                filename="tokenizer.json",
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to download {_HF_REPO} from HuggingFace. If you're "
                f"offline, pre-download model.onnx + tokenizer.json into\n"
                f"  {self._cache_dir}\n"
                f"or use the default stub embedder for offline/dev work.\n"
                f"(error: {e})"
            ) from e

        # Copy into our cache dir for a stable path
        import shutil

        shutil.copy(dl_model, model_path)
        shutil.copy(dl_tokenizer, tokenizer_path)
        return model_path, tokenizer_path

    # ── The math we own: pooling + normalisation ─────────────────────────────

    @staticmethod
    def _mean_pool(
        token_embeddings: np.ndarray,  # (seq_len, hidden)
        attention_mask: np.ndarray,  # (seq_len,)
    ) -> np.ndarray:
        """Mean-pool token embeddings into one sentence vector.

        Masked (padding) tokens are excluded from the average. This is the
        standard pooling for bge / sentence-transformers models.
        """
        mask = attention_mask.astype(np.float32)[:, None]  # (seq_len, 1)
        summed = (token_embeddings * mask).sum(axis=0)  # (hidden,)
        counts = np.clip(mask.sum(axis=0), a_min=1e-9, a_max=None)
        return summed / counts

    @staticmethod
    def _l2_normalise(vec: np.ndarray) -> np.ndarray:
        """Scale to unit L2 norm so cosine similarity == dot product."""
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return vec
        return vec / norm

    # ── Public API (the Embedder protocol) ───────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Embed a passage/document. Returns (384,) float32 unit-norm vector.

        Use this for stored memories (the 'passage' side of retrieval).
        """
        return self._embed_one(text)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query, with bge's recommended query instruction prefix.

        Use this for the user's search query (the 'query' side). bge-small
        was trained asymmetrically: queries get the instruction prefix,
        passages don't. Using the right side improves retrieval quality.
        """
        return self._embed_one(_QUERY_INSTRUCTION + text)

    def _embed_one(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        assert self._session is not None and self._tokenizer is not None

        enc = self._tokenizer.encode(text)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        # Build the ONNX input dict — only feed inputs the model expects
        model_inputs = {i.name for i in self._session.get_inputs()}
        feed: dict[str, np.ndarray] = {}
        if "input_ids" in model_inputs:
            feed["input_ids"] = input_ids
        if "attention_mask" in model_inputs:
            feed["attention_mask"] = attention_mask
        if "token_type_ids" in model_inputs:
            feed["token_type_ids"] = token_type_ids

        outputs = self._session.run(None, feed)
        # First output is last_hidden_state: (batch, seq_len, hidden)
        last_hidden = outputs[0][0]  # (seq_len, hidden)

        pooled = self._mean_pool(last_hidden, attention_mask[0])
        return self._l2_normalise(pooled).astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim
