"""L2 metadata sidecar — SQLite-backed per-memory metadata.

The HNSW index only stores vectors keyed by integer labels (0, 1, 2…).
Everything else — text, MemoryId UUID, context, access counters,
last-touch timestamps, tier — lives in SQLite alongside the index.

Design choices
--------------
- One table: `memories`. Keep it simple.
- Bidirectional mapping: MemoryId (UUID string) <-> hnsw_label (int64).
  HNSW labels are assigned monotonically by L2HotVectorCache.
- WAL mode for crash resilience and concurrent reader/writer. The
  pruner daemon (Sprint 2) will hold a reader connection; the
  main pipeline writes.
- mmap-mapped reads for hot lookups (PRAGMA mmap_size).
- Schema version table — future migrations land here, not in
  ad-hoc ALTERs.
- All writes are transactional. No half-updated metadata.

This module owns ONLY metadata. The vector data lives in the HNSW
index file. The two together form the L2 tier.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from neuro_paging.context.types import BatteryState, ContextTags, TimeBucket
from neuro_paging.memory.types import Memory, MemoryId, Tier

# Schema version. Bump when changing tables; add a migration in _migrate().
SCHEMA_VERSION = 1

# SQLite pragmas applied per connection. WAL mode = concurrent reads.
_PRAGMAS = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",  # WAL + NORMAL is durable enough
    "PRAGMA cache_size = -8000",  # 8 MB page cache
    "PRAGMA temp_store = MEMORY",
    "PRAGMA mmap_size = 67108864",  # 64 MB mmap window
    "PRAGMA foreign_keys = ON",
)


@dataclass(frozen=True, slots=True)
class L2MetadataRow:
    """One row from the memories table. Hydrated into Memory by the cache."""

    memory_id: MemoryId
    hnsw_label: int
    text: str
    embedding_ref: str
    tier: Tier
    created_at: datetime
    last_touch: datetime
    access_count: int
    is_consolidated: bool
    # Flattened context — see _row_to_memory / _memory_to_row helpers
    ctx_timestamp: datetime
    ctx_time_bucket: TimeBucket
    ctx_location: str | None
    ctx_foreground_app: str | None
    ctx_calendar_event: str | None
    ctx_battery_state: BatteryState
    ctx_battery_pct: int
    ctx_semantic_tags_json: str  # JSON-serialised tuple


class L2Metadata:
    """SQLite-backed metadata store for L2 (and L3 later).

    Thread-safety: SQLite connections are NOT thread-safe by default.
    We hold one connection per L2Metadata instance and serialise all
    access via a threading.Lock. WAL mode means SQLite can still
    handle concurrent connections from other processes (the pruner
    daemon could open its own).
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # check_same_thread=False — we serialise via _lock ourselves so
        # SQLite's threadcheck is redundant and just gets in our way.
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; we use BEGIN/COMMIT manually
        )
        self._conn.row_factory = sqlite3.Row

        for p in _PRAGMAS:
            self._conn.execute(p)

        self._migrate()
        logger.debug(f"L2Metadata opened db={self._db_path} schema_v={SCHEMA_VERSION}")

    # ── Schema management ────────────────────────────────────────────────────

    def _migrate(self) -> None:
        """Create tables if not present; run migrations if schema is older."""
        with self._lock:
            cur = self._conn.cursor()

            # Bootstrap the schema_version table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL,
                    applied_at TEXT NOT NULL
                )
            """)
            cur.execute("SELECT MAX(version) FROM schema_version")
            current = cur.fetchone()[0] or 0

            if current >= SCHEMA_VERSION:
                return  # nothing to do

            # ── v1: initial schema ──
            if current < 1:
                cur.executescript("""
                    CREATE TABLE memories (
                        memory_id           TEXT PRIMARY KEY,
                        hnsw_label          INTEGER NOT NULL UNIQUE,
                        text                TEXT NOT NULL,
                        embedding_ref       TEXT NOT NULL,
                        tier                TEXT NOT NULL,

                        created_at          TEXT NOT NULL,
                        last_touch          TEXT NOT NULL,
                        access_count        INTEGER NOT NULL DEFAULT 0,
                        is_consolidated     INTEGER NOT NULL DEFAULT 0,

                        ctx_timestamp       TEXT NOT NULL,
                        ctx_time_bucket     TEXT NOT NULL,
                        ctx_location        TEXT,
                        ctx_foreground_app  TEXT,
                        ctx_calendar_event  TEXT,
                        ctx_battery_state   TEXT NOT NULL,
                        ctx_battery_pct     INTEGER NOT NULL,
                        ctx_semantic_tags   TEXT NOT NULL DEFAULT '[]'
                    );

                    CREATE INDEX idx_memories_tier ON memories(tier);
                    CREATE INDEX idx_memories_last_touch ON memories(last_touch);
                    CREATE INDEX idx_memories_label ON memories(hnsw_label);
                """)
                cur.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (1, datetime.now(UTC).isoformat()),
                )

            self._conn.commit()
            logger.info(f"L2Metadata migrated to schema v{SCHEMA_VERSION}")

    # ── Transaction helper ───────────────────────────────────────────────────

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Cursor]:
        """Yield a cursor inside an atomic transaction."""
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn.cursor()
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # ── Memory <-> row conversion ────────────────────────────────────────────

    @staticmethod
    def _memory_to_row(memory: Memory, hnsw_label: int) -> tuple:
        """Flatten a Memory into a tuple ready for parameterised INSERT."""
        ctx = memory.context
        return (
            str(memory.id),
            hnsw_label,
            memory.text,
            memory.embedding_ref,
            memory.tier.value,
            memory.created_at.isoformat(),
            memory.last_touch.isoformat(),
            memory.access_count,
            1 if memory.is_consolidated else 0,
            ctx.timestamp.isoformat(),
            ctx.time_bucket.value,
            ctx.location,
            ctx.foreground_app,
            ctx.calendar_event,
            ctx.battery_state.value,
            ctx.battery_pct,
            json.dumps(list(ctx.semantic_tags)),
        )

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        """Hydrate a row back into a Memory."""
        ctx = ContextTags(
            timestamp=datetime.fromisoformat(row["ctx_timestamp"]),
            time_bucket=TimeBucket(row["ctx_time_bucket"]),
            location=row["ctx_location"],
            foreground_app=row["ctx_foreground_app"],
            calendar_event=row["ctx_calendar_event"],
            battery_state=BatteryState(row["ctx_battery_state"]),
            battery_pct=row["ctx_battery_pct"],
            semantic_tags=tuple(json.loads(row["ctx_semantic_tags"])),
        )
        return Memory(
            id=MemoryId(row["memory_id"]),
            text=row["text"],
            embedding_ref=row["embedding_ref"],
            context=ctx,
            tier=Tier(row["tier"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_touch=datetime.fromisoformat(row["last_touch"]),
            access_count=row["access_count"],
            is_consolidated=bool(row["is_consolidated"]),
        )

    # ── Writes ───────────────────────────────────────────────────────────────

    def insert(self, memory: Memory, hnsw_label: int) -> None:
        """Insert metadata for a new memory. Raises if memory_id exists."""
        row = self._memory_to_row(memory, hnsw_label)
        with self._txn() as cur:
            cur.execute(
                """
                INSERT INTO memories (
                    memory_id, hnsw_label, text, embedding_ref, tier,
                    created_at, last_touch, access_count, is_consolidated,
                    ctx_timestamp, ctx_time_bucket, ctx_location,
                    ctx_foreground_app, ctx_calendar_event,
                    ctx_battery_state, ctx_battery_pct, ctx_semantic_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                row,
            )

    def upsert(self, memory: Memory, hnsw_label: int) -> None:
        """Insert OR replace (use during tier moves where the row may exist)."""
        row = self._memory_to_row(memory, hnsw_label)
        with self._txn() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO memories (
                    memory_id, hnsw_label, text, embedding_ref, tier,
                    created_at, last_touch, access_count, is_consolidated,
                    ctx_timestamp, ctx_time_bucket, ctx_location,
                    ctx_foreground_app, ctx_calendar_event,
                    ctx_battery_state, ctx_battery_pct, ctx_semantic_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                row,
            )

    def touch(self, memory_id: MemoryId) -> bool:
        """Bump access_count + refresh last_touch. Returns True if row existed."""
        now = datetime.now(UTC).isoformat()
        with self._txn() as cur:
            cur.execute(
                "UPDATE memories SET access_count = access_count + 1, last_touch = ? "
                "WHERE memory_id = ?",
                (now, str(memory_id)),
            )
            return cur.rowcount > 0

    def delete(self, memory_id: MemoryId) -> bool:
        """Remove a memory's metadata row. Returns True if it existed."""
        with self._txn() as cur:
            cur.execute("DELETE FROM memories WHERE memory_id = ?", (str(memory_id),))
            return cur.rowcount > 0

    def update_tier(self, memory_id: MemoryId, new_tier: Tier) -> bool:
        """Move a memory's tier label without touching anything else."""
        now = datetime.now(UTC).isoformat()
        with self._txn() as cur:
            cur.execute(
                "UPDATE memories SET tier = ?, last_touch = ? WHERE memory_id = ?",
                (new_tier.value, now, str(memory_id)),
            )
            return cur.rowcount > 0

    # ── Reads ────────────────────────────────────────────────────────────────

    def get(self, memory_id: MemoryId) -> Memory | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (str(memory_id),)
            )
            row = cur.fetchone()
        return self._row_to_memory(row) if row else None

    def get_by_label(self, hnsw_label: int) -> Memory | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM memories WHERE hnsw_label = ?", (hnsw_label,))
            row = cur.fetchone()
        return self._row_to_memory(row) if row else None

    def label_for(self, memory_id: MemoryId) -> int | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT hnsw_label FROM memories WHERE memory_id = ?",
                (str(memory_id),),
            )
            row = cur.fetchone()
        return row["hnsw_label"] if row else None

    def contains(self, memory_id: MemoryId) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "SELECT 1 FROM memories WHERE memory_id = ? LIMIT 1",
                (str(memory_id),),
            )
            return cur.fetchone() is not None

    def count(self, tier: Tier | None = None) -> int:
        with self._lock:
            if tier is None:
                cur = self._conn.execute("SELECT COUNT(*) FROM memories")
            else:
                cur = self._conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE tier = ?", (tier.value,)
                )
            return cur.fetchone()[0]

    def total_text_bytes(self, tier: Tier | None = None) -> int:
        """Sum of utf-8 text bytes — coarse 'how big is this tier' signal."""
        with self._lock:
            if tier is None:
                cur = self._conn.execute("SELECT COALESCE(SUM(LENGTH(text)), 0) FROM memories")
            else:
                cur = self._conn.execute(
                    "SELECT COALESCE(SUM(LENGTH(text)), 0) FROM memories WHERE tier = ?",
                    (tier.value,),
                )
            return cur.fetchone()[0]

    def iter_memories(self, tier: Tier | None = None) -> Iterator[Memory]:
        """Stream all memories (optionally filtered by tier). Order: undefined."""
        with self._lock:
            if tier is None:
                cur = self._conn.execute("SELECT * FROM memories")
            else:
                cur = self._conn.execute("SELECT * FROM memories WHERE tier = ?", (tier.value,))
            rows = cur.fetchall()
        for row in rows:
            yield self._row_to_memory(row)

    def find_cold(self, tier: Tier, older_than: datetime) -> list[MemoryId]:
        """Return memory_ids in `tier` not touched since `older_than`.

        Used by the pruner daemon (Sprint 2) to find demote candidates.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT memory_id FROM memories "
                "WHERE tier = ? AND last_touch < ? "
                "ORDER BY last_touch ASC",
                (tier.value, older_than.isoformat()),
            )
            return [MemoryId(r["memory_id"]) for r in cur.fetchall()]

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @property
    def db_path(self) -> Path:
        return self._db_path
