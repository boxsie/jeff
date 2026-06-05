"""Pinned memory — things Jeff (or the operator) deliberately chose to keep.

Explicit, intentional memory: distinct from the automatic episodic store (every
turn is saved regardless) and from the fire-and-forget reflection drive (facts +
opinions auto-distilled and salience-decayed). A pin is something marked "keep
this" on purpose, via one of two surfaces that share this one store:

- the ``remember`` TOOL — Jeff calls it mid-turn when it decides something is
  worth keeping (agency over its own memory);
- the ``/remember`` operator COMMAND — the operator pins something directly.

Like moods, this is deliberately NOT a pgvector store: pins are **always
injected** into the prompt as a small labelled block (not semantically
recalled), so they need no embedding — exact-normalised dedup is enough, and it
keeps the ``/remember`` command pure-DB (no Ollama embed call inside a command
handler). And unlike reflection, pins do NOT decay or prune: deliberate means
permanent until explicitly forgotten — the whole point of pinning is "always
remember this."

One plain table::

    pinned_memory(id, peer, text, norm, source, created_at)  UNIQUE(peer, norm)

``norm`` is the dedup key (strip-tokens + collapse-ws + lowercase); inserting a
near-identical pin is a no-op via ``ON CONFLICT (peer, norm) DO NOTHING``.
``source`` records who pinned it (``jeff`` via the tool, ``operator`` via the
command) so ``/mind`` can show provenance.

Everything here is gated by ``JEFF_REMEMBER_ENABLED`` (default off): when off the
store/tool/command are never constructed, so behaviour is byte-identical to today.

Public-repo hygiene: the mechanism is neutral; pinned content lives only in the DB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from .screen import strip_chat_template_tokens


log = logging.getLogger("jeff.pinned")


# Hard ceiling on a stored pin's bytes — defence-in-depth, far above any real
# one-line note. The tool/command also enforce the (operator-tunable, tighter)
# JEFF_REMEMBER_MAX_CHARS before calling in.
MAX_ITEM_BYTES = 4096

# Sources, recorded so /mind can show who pinned a thing.
SOURCE_JEFF = "jeff"
SOURCE_OPERATOR = "operator"


@dataclass(frozen=True)
class Pinned:
    """One pinned memory."""

    id: int
    text: str
    source: str
    created_at: datetime


def _normalize(text: str) -> str:
    """The dedup key: strip template tokens, collapse whitespace, lowercase.

    So "I'm vegetarian" and "  i'm   vegetarian " collapse to one pin, while
    genuinely different notes each get their own row. The original-case text is
    what's stored and shown; ``norm`` only drives the UNIQUE constraint.
    """
    text = strip_chat_template_tokens(text)
    return " ".join(text.split()).lower()


_DDL_TABLE = sql.SQL(
    "CREATE TABLE IF NOT EXISTS pinned_memory ("
    "id BIGSERIAL PRIMARY KEY, "
    "peer TEXT NOT NULL, "
    "text TEXT NOT NULL, "
    "norm TEXT NOT NULL, "
    "source TEXT NOT NULL DEFAULT 'jeff', "
    "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
    "UNIQUE (peer, norm))"
)

_DDL_IDX = sql.SQL(
    "CREATE INDEX IF NOT EXISTS idx_pinned_peer_created "
    "ON pinned_memory(peer, created_at DESC)"
)


class PinnedMemoryStore:
    """Async store of a peer's deliberately-pinned memories (plain Postgres).

    No embedder, no vector column — pins are always injected, never semantically
    recalled (see module docstring), and never decay.
    """

    def __init__(self, pool: AsyncConnectionPool):
        self._pool = pool

    @classmethod
    async def create(cls, pool: AsyncConnectionPool) -> "PinnedMemoryStore":
        s = cls(pool)
        await s._init_schema()
        return s

    async def _init_schema(self) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_DDL_TABLE)
                await cur.execute(_DDL_IDX)
            await conn.commit()

    async def add(
        self, peer: str, text: str, *, source: str = SOURCE_JEFF
    ) -> int | None:
        """Pin a note, skipping exact-normalised duplicates of existing pins.

        Returns the new row id, ``None`` when it was a duplicate (an equivalent
        pin already exists), or ``None`` when the text was blank. Dedup is atomic
        via ``ON CONFLICT (peer, norm) DO NOTHING``.
        """
        text = strip_chat_template_tokens(text).strip()
        if not text:
            return None
        size = len(text.encode("utf-8"))
        if size > MAX_ITEM_BYTES:
            raise ValueError(
                f"pinned memory too large: {size} bytes (limit {MAX_ITEM_BYTES})"
            )
        norm = _normalize(text)
        if not norm:
            return None
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO pinned_memory (peer, text, norm, source) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (peer, norm) DO NOTHING RETURNING id",
                    (peer, text, norm, source),
                )
                row = await cur.fetchone()
            await conn.commit()
        return int(row[0]) if row else None

    async def list(self, peer: str, *, limit: int = 50) -> list[Pinned]:
        """A peer's pins, most recent first (for injection + /mind)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, text, source, created_at FROM pinned_memory "
                    "WHERE peer = %s ORDER BY created_at DESC, id DESC LIMIT %s",
                    (peer, limit),
                )
                rows = await cur.fetchall()
        return [
            Pinned(
                id=int(r[0]),
                text=strip_chat_template_tokens(r[1]),
                source=r[2],
                created_at=r[3],
            )
            for r in rows
        ]

    async def forget(self, peer: str) -> int:
        """Delete every pin for one peer (so /forget wipes these too)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM pinned_memory WHERE peer = %s", (peer,)
                )
                deleted = cur.rowcount
            await conn.commit()
        return int(deleted)

    async def count(self, peer: str) -> int:
        """How many pins this peer has (for /stats-style summaries)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT count(*) FROM pinned_memory WHERE peer = %s", (peer,)
                )
                row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def reset(self) -> None:
        """Destructive fresh start: drop the table and recreate it (called by
        ``reset-memory``; mirrors the other stores' reset)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP TABLE IF EXISTS pinned_memory")
            await conn.commit()
        await self._init_schema()
