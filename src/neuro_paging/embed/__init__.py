"""Embedding runtimes — text → vector.

The substrate defines an Embedder protocol (see memory.manager.Embedder).
This package provides real implementations of it.

- BGESmallEmbedder: bge-small-en-v1.5 via ONNX Runtime. No torch.
  ~90 MB, runs on-device. The production embedder.

The default stub embedder (deterministic random vectors) lives in
memory.manager and is used for tests + dev so the core never depends
on a model download.
"""

from neuro_paging.embed.bge_runtime import BGESmallEmbedder

__all__ = ["BGESmallEmbedder"]
