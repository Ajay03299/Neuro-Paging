"""Pipeline — the orchestration layer that ties the substrate together.

The substrate (memory.MemoryManager), the embedder (embed.BGESmallEmbedder),
and the scorer (routing.ContextAwareScorer) are independent components. This
package wires them into one thing you can actually use:

    agent = MemoryAgent(...)
    agent.remember("user prefers window seats", ctx)   # write path
    block = agent.recall("seating preference?", ctx)    # read path → context block
    answer = agent.respond("seating preference?", ctx)  # recall + generate

The Generator is a plugin (protocol). The default echoes the assembled
context block — because a memory system's job is to retrieve and assemble
the RIGHT context. Real token generation (llama.cpp / Qwen) slots in via
the same protocol without touching the pipeline.
"""

from neuro_paging.pipeline.agent import (
    AssembledContext,
    Generator,
    MemoryAgent,
)

__all__ = ["AssembledContext", "Generator", "MemoryAgent"]
