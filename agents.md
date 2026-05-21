# agents.md

> Agent contracts, tool schemas, and refusal rules for **Neuro-Paging**.
> Required by Samsung ennovateX AX Hackathon 2026 rules.

This document defines how AI-driven components inside Neuro-Paging interact with one another, what each is authorised to do, and what they must refuse. It applies to two layers:

1. **Runtime agents** — components that run *inside* Neuro-Paging on the user's device.
2. **Development-time agents** — coding assistants used by the team during the build.

---

## 1. Runtime agents

### 1.1 Retrieval Agent (Context-Aware Router)

| Property | Value |
| :--- | :--- |
| Module | `src/neuro_paging/routing/` |
| Inputs | user query (str), context tags (dict), top-k (int) |
| Outputs | ranked list of `Hit` objects with provenance |
| Side effects | updates `last_touch_ts` + `access_count` on returned memories |
| May call | `MemoryManager.query()`, embedder, scorer |
| Must NOT call | LLM, network, filesystem outside its sandbox |

**Refusal rules**

- Never returns memories tagged `private:strict` regardless of score.
- If `battery_pct < 15` and `is_charging == False`, falls back to L1-only retrieval (no L2/L3 disk reads).
- If the scoring function returns NaN or Inf for any hit, that hit is discarded with a logged warning.

### 1.2 Consolidator Agent (Nightly Summariser)

| Property | Value |
| :--- | :--- |
| Module | `src/neuro_paging/consolidator/` |
| Inputs | a cluster of 30–80 L2 vectors + their source texts |
| Outputs | one summary string + one averaged embedding (the new L3 concept) |
| Triggered by | `daemons/consolidator_runner.py` on idle CPU + charging |
| LLM used | Qwen2.5-1.5B-Instruct (later: Qwen2.5-1.5B-Consolidator LoRA) |

**Refusal rules**

- Will not consolidate clusters containing any memory tagged `private:strict`. Those are passed through to L3 verbatim.
- If summary length > 2× average input length, retry with a tighter prompt. If still long after 2 retries, skip and log.
- All summaries must include a provenance trail listing source memory IDs.

### 1.3 Predictive Prefetcher

| Property | Value |
| :--- | :--- |
| Module | `src/neuro_paging/prefetch/` |
| Inputs | recent (time, app, topic) tuples |
| Outputs | speculative L3→L1 promotion (write-only, never blocks read path) |
| Triggered by | pattern match with `confidence > 0.7` |

**Refusal rules**

- Will not fire when battery is discharging and `< 20%`.
- Will not fire more than once per 5-minute window per detected pattern (rate limit).
- Failed prefetches are silent — never surface a UI error for a speculative load.

---

## 2. Tool schemas (MCP-style)

Internal modules expose tools through a uniform schema, modelled on MCP. Used during dev for tool-calling tests against Christine's pipeline.

### `memory.insert`

```json
{
  "name": "memory.insert",
  "description": "Insert a new memory into the tiered store. Routes to L1 by default.",
  "parameters": {
    "text": { "type": "string", "required": true },
    "context": { "type": "object", "required": true,
                 "properties": { "time": "iso8601", "app": "string", "location": "string" } },
    "tags": { "type": "array", "items": { "type": "string" }, "default": [] }
  },
  "returns": { "memory_id": "string", "tier": "L1|L2|L3" }
}
```

### `memory.query`

```json
{
  "name": "memory.query",
  "description": "Retrieve top-k memories ranked by context-aware score.",
  "parameters": {
    "text": { "type": "string", "required": true },
    "context": { "type": "object", "required": true },
    "k": { "type": "integer", "default": 5, "max": 50 }
  },
  "returns": { "hits": [{ "text": "string", "score": "float", "tier": "string", "provenance": "object" }] }
}
```

### `memory.stats`

```json
{
  "name": "memory.stats",
  "description": "Return current tier occupancy, hit-rate, and storage usage.",
  "parameters": {},
  "returns": { "l1_count": "int", "l2_count": "int", "l3_count": "int",
               "l2_bytes": "int", "l3_bytes": "int",
               "hit_rate_24h": "float" }
}
```

Full schema definitions land in `src/neuro_paging/pipeline/tool_schemas.py`.

---

## 3. Development-time agents

We use AI coding assistants during development. Per Samsung ennovateX disclosure rules:

| Tool | Where used | Disclosed in |
| :--- | :--- | :--- |
| **Cursor** | All Python and C++ editing | This file, README |
| **Claude Code** | Architectural decisions, refactors, code review | This file, README |
| **aider** | Repo-aware multi-file refactors | This file, README |
| **Continue.dev** | Offline IDE assistant when on flaky networks | This file, README |

**Contracts we hold these agents to**

- No commits without human review. Every PR is reviewed by the team member who does **not** own that file.
- No closed APIs introduced as runtime dependencies. AI agents may suggest them; we reject.
- No license-incompatible code suggestions accepted. All deps must be OSI-permissive (MIT, Apache-2.0, BSD-3).
- Every AI-assisted commit message stays factual — no fabricated benchmarks, no aspirational claims.

---

## 4. Refusal rules (global)

Across both runtime and dev-time, Neuro-Paging agents and tools will refuse:

1. **Memories or operations involving PII** that the user hasn't explicitly opted to retain. (Hash + tag scheme detailed in `docs/privacy.md` once written.)
2. **Cross-device sync.** All memory stays on the device. There is no cloud round-trip, ever.
3. **Operations that would exceed declared tier budgets** without explicit pruner consent. L2 cannot exceed its 8 MB cap by inserting; it must demote first.
4. **Speculative writes during query** — the read path never mutates store contents beyond updating access counters.

---

## 5. Eval-as-CI commitments

Per the deck and hackathon rules:

- Every PR runs the LongMemEval and MobileMem-Bench eval subsets.
- A regression budget is enforced — PRs that drop recall@5 by >2% relative are blocked.
- Provenance is checked: every retrieved memory must carry tier + score breakdown.
- AI-assisted contributions are disclosed in the README's "AI-assisted development" section.

---

*This document is living. As the system evolves, contracts here are the source of truth.*
*Last updated alongside repo scaffold. — Team ByteMe.*