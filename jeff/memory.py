"""Postgres + pgvector semantic memory.

Schema is created idempotently on first connect, so the bot can boot against
an empty database without a separate migration step. One async connection
pool per process; embeddings come from Ollama via the provided client.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np
from pgvector.psycopg import register_vector_async
from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from .screen import strip_chat_template_tokens


class _Embedder(Protocol):
    async def embed(self, text: str, *, model: str) -> list[float]: ...


# Hard ceiling on stored content bytes. The intended input cap is much lower
# (8 KiB via JEFF_MAX_MESSAGE_BYTES), enforced in main._drain_events; this is
# defense-in-depth so a future code path that bypasses screen_text cannot
# silently insert a 1 GB row. Sized at 8x the default input cap so legitimate
# operator bumps of JEFF_MAX_MESSAGE_BYTES don't immediately hit this.
MAX_CONTENT_BYTES = 65536

# Default cosine-distance ceiling for recall(). A row further than this from the
# query isn't relevant enough to feed the model (see recall() for the rationale).
# Tuned for bge-m3 (the default embed model), which separates relevant (~0.4-0.52)
# from unrelated (~0.7) cleanly. The runtime value is config-driven
# (MEMORY_RECALL_DISTANCE); this constant is the shared default for direct callers
# and tests. Keep in sync with config._parse_recall_distance's default.
DEFAULT_RECALL_DISTANCE_MAX = 0.55


@dataclass(frozen=True)
class Message:
    id: int
    peer: str
    role: str
    content: str
    ts: datetime


# W3 #20123205: the only env-derived value in DDL is the embedding
# dimension. config.Config bounds it to [1, 8192] at load time, but the
# defense-in-depth shape here is what stops a future change to that bound
# (or a direct caller that bypasses Config) from injecting through .format.
# Each statement is composed with psycopg.sql primitives + executed
# separately rather than a single multi-statement .format string.
_DDL_EXTENSION = sql.SQL("CREATE EXTENSION IF NOT EXISTS vector")

_DDL_TABLE = sql.SQL(
    "CREATE TABLE IF NOT EXISTS messages ("
    "id BIGSERIAL PRIMARY KEY, "
    "peer TEXT NOT NULL, "
    "role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')), "
    "content TEXT NOT NULL, "
    "embedding vector({dim}) NOT NULL, "
    "ts TIMESTAMPTZ NOT NULL DEFAULT now())"
)

_DDL_IDX_PEER_TS = sql.SQL(
    "CREATE INDEX IF NOT EXISTS idx_messages_peer_ts ON messages(peer, ts DESC)"
)

_DDL_IDX_EMBEDDING = sql.SQL(
    "CREATE INDEX IF NOT EXISTS idx_messages_embedding "
    "ON messages USING hnsw (embedding vector_cosine_ops)"
)

# Per-peer control state. `history_cutoff` is the session watermark for the
# `/clear` chat command (jeff ticket dc1791d5): BOTH `recent()` and `recall()`
# only return rows newer than it, so `/clear` starts a genuinely fresh session
# — the active window empties and semantic recall is scoped to the new session.
# Pre-cutoff rows still exist (only `/forget` deletes); cross-session recall is
# a deliberate future step. No embedding here, so it's a plain CREATE TABLE (no
# env-derived dimension), but kept in the same idempotent psycopg.sql shape as
# the rest.
_DDL_PEER_STATE = sql.SQL(
    "CREATE TABLE IF NOT EXISTS peer_state ("
    "peer TEXT PRIMARY KEY, "
    "history_cutoff TIMESTAMPTZ)"
)


class Memory:
    """Async memory store backed by Postgres + pgvector."""

    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedder: _Embedder,
        *,
        embed_model: str,
        embed_dim: int,
    ):
        self._pool = pool
        self._embedder = embedder
        self._embed_model = embed_model
        self._embed_dim = embed_dim

    @classmethod
    async def create(
        cls,
        pool: AsyncConnectionPool,
        embedder: _Embedder,
        *,
        embed_model: str,
        embed_dim: int,
    ) -> "Memory":
        m = cls(pool, embedder, embed_model=embed_model, embed_dim=embed_dim)
        await m._init_schema()
        return m

    async def _init_schema(self) -> None:
        # Composes the per-statement DDL pieces with psycopg.sql so the
        # only env-derived value (the embedding dimension) lands inside a
        # Literal placeholder rather than a Python f-string. The dim is
        # already bounded at config load (W3 #20123205) so this is
        # defense in depth.
        if not (1 <= self._embed_dim <= 8192):
            raise ValueError(
                f"embed_dim out of range: {self._embed_dim} (allowed 1..8192)"
            )
        table_ddl = _DDL_TABLE.format(dim=sql.Literal(self._embed_dim))
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_DDL_EXTENSION)
                await self._guard_embed_dim(cur)
                await cur.execute(table_ddl)
                await cur.execute(_DDL_IDX_PEER_TS)
                await cur.execute(_DDL_IDX_EMBEDDING)
                await cur.execute(_DDL_PEER_STATE)
            await conn.commit()
            await register_vector_async(conn)

    async def _guard_embed_dim(self, cur) -> None:
        # If `messages` already exists at a different embedding dimension (e.g.
        # the operator switched embed models), CREATE TABLE IF NOT EXISTS is a
        # no-op and every later insert would fail on a dim mismatch — silently,
        # inside the per-turn except. Detect it here and fail loudly with the fix
        # rather than letting memory quietly break.
        await cur.execute("SELECT to_regclass('messages')")
        row = await cur.fetchone()
        if row is None or row[0] is None:
            return  # fresh database — nothing to compare against
        await cur.execute(
            "SELECT format_type(a.atttypid, a.atttypmod) "
            "FROM pg_attribute a "
            "WHERE a.attrelid = 'messages'::regclass AND a.attname = 'embedding'"
        )
        ftrow = await cur.fetchone()
        if not ftrow or not ftrow[0]:
            return
        m = re.search(r"\((\d+)\)", ftrow[0])  # e.g. 'vector(768)' -> 768
        existing_dim = int(m.group(1)) if m else None
        if existing_dim is not None and existing_dim != self._embed_dim:
            raise ValueError(
                f"stored embedding dimension {existing_dim} != configured "
                f"{self._embed_dim} (OLLAMA_EMBED_DIM). The embedding model "
                "changed; run `python -m jeff reset-memory --yes` to start fresh."
            )

    async def _connection(self):
        # psycopg_pool reuses connections, so register the vector adapter on
        # each checkout. Cheap, idempotent.
        return self._pool.connection()

    async def remember(self, peer: str, role: str, content: str) -> int:
        # W3 #dc9acd3c: an allowlisted peer can otherwise plant a row
        # containing gemma chat-template tokens (<start_of_turn>model...)
        # that would be rendered as a forged assistant turn the next time
        # the row is recalled and re-fed to the LLM. Strip on insert so
        # memory never carries them; recall does the same strip as belt
        # and braces for rows that predate this guard.
        content = strip_chat_template_tokens(content)

        # Belt-and-braces: main._drain_events is supposed to have already
        # rejected oversize input via screen_text. If something slips past
        # (test seam, future caller), refuse here rather than embedding +
        # inserting a multi-MB row.
        size = len(content.encode("utf-8"))
        if size > MAX_CONTENT_BYTES:
            raise ValueError(
                f"content too large: {size} bytes (limit {MAX_CONTENT_BYTES})"
            )
        emb = await self._embedder.embed(content, model=self._embed_model)
        if len(emb) != self._embed_dim:
            raise ValueError(
                f"embedding dim mismatch: got {len(emb)}, configured {self._embed_dim} "
                f"(check OLLAMA_EMBED_DIM matches the {self._embed_model} model)"
            )
        vec = np.asarray(emb, dtype=np.float32)
        async with self._pool.connection() as conn:
            await register_vector_async(conn)
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO messages (peer, role, content, embedding) "
                    "VALUES (%s, %s, %s, %s) RETURNING id",
                    (peer, role, content, vec),
                )
                row = await cur.fetchone()
            await conn.commit()
            return int(row[0])

    async def recall(
        self,
        peer: str,
        query: str,
        k: int = 5,
        *,
        distance_max: float = DEFAULT_RECALL_DISTANCE_MAX,
    ) -> list[Message]:
        # Session scope: like recent(), recall now honours the per-peer
        # history_cutoff, so `/clear` gives a genuinely fresh session —
        # semantic recall no longer drags pre-clear lines back into context
        # the moment a new message is topically similar. Rows older than the
        # cutoff still exist (they're not deleted — that's /forget's job);
        # they're simply out of reach until cross-session recall is added.
        # COALESCE to -infinity so the no-cutoff case is unchanged.
        #
        # W3 #dc9acd3c: pgvector "<=>" returns cosine *distance* (0 =
        # identical, 1 = orthogonal). Two reasons to floor it:
        #   1. Near-zero distance means a stored row that looks just like
        #      the query — quite possibly the peer's own crafted text
        #      (echo attack).
        #   2. Far distances (>0.4) mean the row isn't actually relevant;
        #      feeding it to the LLM under "you have access to memory of
        #      prior conversations" steers replies toward irrelevant
        #      context. 0.4 is empirically tight for nomic-embed-text;
        #      callers can pass a wider distance_max for noisier embeds.
        emb = await self._embedder.embed(query, model=self._embed_model)
        vec = np.asarray(emb, dtype=np.float32)
        async with self._pool.connection() as conn:
            await register_vector_async(conn)
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, peer, role, content, ts FROM messages "
                    "WHERE peer = %s "
                    "  AND ts > COALESCE("
                    "    (SELECT history_cutoff FROM peer_state WHERE peer = %s), "
                    "    '-infinity'::timestamptz) "
                    "  AND (embedding <=> %s) <= %s "
                    "ORDER BY embedding <=> %s "
                    "LIMIT %s",
                    (peer, peer, vec, distance_max, vec, k),
                )
                rows = await cur.fetchall()
        return [
            Message(
                id=r[0],
                peer=r[1],
                role=r[2],
                content=strip_chat_template_tokens(r[3]),
                ts=r[4],
            )
            for r in rows
        ]

    async def recall_scored(
        self,
        peer: str,
        query: str,
        *,
        limit: int = 8,
    ) -> list[tuple[Message, float]]:
        """Introspection view for `/debug recall`: the closest rows to `query`
        with their cosine distances, honouring the session cutoff.

        Unlike recall(), this deliberately does NOT filter by distance_max — it
        returns the top `limit` candidates ordered by distance so the operator
        can see the near-misses just over the threshold (useful for tuning
        DEFAULT_RECALL_DISTANCE_MAX). It still respects the cutoff, so it mirrors
        the session scope a real turn would see. The caller decides which rows
        the live recall() would actually have kept.
        """
        emb = await self._embedder.embed(query, model=self._embed_model)
        vec = np.asarray(emb, dtype=np.float32)
        async with self._pool.connection() as conn:
            await register_vector_async(conn)
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, peer, role, content, ts, (embedding <=> %s) AS dist "
                    "FROM messages "
                    "WHERE peer = %s "
                    "  AND ts > COALESCE("
                    "    (SELECT history_cutoff FROM peer_state WHERE peer = %s), "
                    "    '-infinity'::timestamptz) "
                    "ORDER BY embedding <=> %s "
                    "LIMIT %s",
                    (vec, peer, peer, vec, limit),
                )
                rows = await cur.fetchall()
        return [
            (
                Message(
                    id=r[0],
                    peer=r[1],
                    role=r[2],
                    content=strip_chat_template_tokens(r[3]),
                    ts=r[4],
                ),
                float(r[5]),
            )
            for r in rows
        ]

    async def recent(self, peer: str, n: int = 10) -> list[Message]:
        # Respect the session watermark: after `/clear` (set_history_cutoff),
        # the conversational window only includes rows newer than the cutoff.
        # COALESCE to -infinity when no cutoff is set so the unfiltered case is
        # identical to before. recall() now applies the same filter, so a
        # cleared session is fresh in both windows.
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, peer, role, content, ts FROM messages "
                    "WHERE peer = %s "
                    "  AND ts > COALESCE("
                    "    (SELECT history_cutoff FROM peer_state WHERE peer = %s), "
                    "    '-infinity'::timestamptz) "
                    "ORDER BY ts DESC "
                    "LIMIT %s",
                    (peer, peer, n),
                )
                rows = await cur.fetchall()
        # Return chronological (oldest first) so callers can append to chat history directly.
        rows = list(reversed(rows))
        return [
            Message(
                id=r[0],
                peer=r[1],
                role=r[2],
                content=strip_chat_template_tokens(r[3]),
                ts=r[4],
            )
            for r in rows
        ]

    async def forget(self, peer: str) -> int:
        """Flush all memory for a single peer. Returns the row count deleted.

        Admin-only path for W3 #dc9acd3c: if an allowlisted peer has been
        observed poisoning memory, the operator can clear that peer's slice
        without nuking everyone else's history. Cross-peer leak is not a
        concern today (recall is peer-keyed), but this gives the operator
        an explicit recovery tool.
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM messages WHERE peer = %s",
                    (peer,),
                )
                deleted = cur.rowcount
                # Clear any soft-reset watermark too — a stale cutoff over an
                # empty history is harmless but tidy, and keeps peer_state from
                # leaking rows for peers whose messages are all gone.
                await cur.execute(
                    "DELETE FROM peer_state WHERE peer = %s",
                    (peer,),
                )
            await conn.commit()
        return int(deleted)

    async def reset(self) -> None:
        """Destructive fresh start: drop ALL messages and per-peer cutoffs, then
        recreate the schema at the configured embedding dimension.

        This is the operator's path when the embedding model/dimension changes
        and re-embedding old rows isn't wanted (`python -m jeff reset-memory`).
        Wipes every peer, unlike forget() which is scoped to one.
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP TABLE IF EXISTS messages")
                await cur.execute("DROP TABLE IF EXISTS peer_state")
            await conn.commit()
        await self._init_schema()

    async def set_history_cutoff(self, peer: str) -> None:
        """Soft reset for `/new`: mark now() as this peer's history watermark.

        Upsert so repeated `/new`s just advance the cutoff. `recent()` honours
        it (fresh conversational window); `recall()` ignores it (long-term
        semantic memory is unaffected).
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO peer_state (peer, history_cutoff) "
                    "VALUES (%s, now()) "
                    "ON CONFLICT (peer) DO UPDATE SET history_cutoff = now()",
                    (peer,),
                )
            await conn.commit()

    async def get_history_cutoff(self, peer: str) -> datetime | None:
        """The peer's current session watermark, or None if never `/clear`-ed.

        Read-only companion to set_history_cutoff, used by `/debug` to show where
        the active session starts. None means the whole history is in scope.
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT history_cutoff FROM peer_state WHERE peer = %s",
                    (peer,),
                )
                row = await cur.fetchone()
        return row[0] if row else None

    async def count(self, peer: str) -> int:
        """How many stored messages this peer has (for `/stats`)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT count(*) FROM messages WHERE peer = %s",
                    (peer,),
                )
                row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def total(self) -> int:
        """Total stored messages across all peers (for `/stats`)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT count(*) FROM messages")
                row = await cur.fetchone()
        return int(row[0]) if row else 0
