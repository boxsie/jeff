"""Musings — Jeff's idle train of thought, carried between conversations.

The idle self-turn (``selfturn.py``) hands Jeff a tool-enabled "quiet moment,
what do you want to do?" turn even when the operator is offline. Jeff reasons,
maybe shifts a mood or pins something — and then the *narration* of that turn was
historically logged and thrown away. So Jeff would think between conversations
and instantly forget she'd done so; the next real turn started cold, with no
sense of what she'd been mulling in the gap.

This store closes that loop. It keeps the **latest** idle musing per peer (one
row, upserted), so the next reactive turn can surface a short "what you've been
mulling" block. It's deliberately last-write-wins, not a log: a self-turn is a
fleeting thought, and only the freshest one is worth carrying forward — older
musings are superseded, not accumulated.

Consume-by-recency (no explicit clear): the reader only surfaces the musing when
it's **newer than the last stored conversation turn** — i.e. it happened during
the idle gap. Once the operator replies and Jeff answers (a new message lands),
the musing is older than the latest turn and naturally stops surfacing. So a
single idle thought colours the *next* exchange, then fades, with no consume
column or clear step.

One plain table (no embedding — it's always-or-never injected, never recalled)::

    musings(peer PRIMARY KEY, text, created_at)

Gated by ``JEFF_MUSINGS_ENABLED`` (default off): when off the store is never
constructed and nothing records or surfaces, so behaviour is byte-identical.

Public-repo hygiene: the mechanism is neutral; musing content lives only in the DB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from ._schema import assert_columns
from .screen import strip_chat_template_tokens


log = logging.getLogger("jeff.musings")


# Hard ceiling on a stored musing's bytes — defence-in-depth. The recorder also
# truncates to the (operator-tunable) JEFF_MUSINGS_MAX_CHARS before calling in;
# this is the backstop so a runaway narration can't bloat the row / prompt block.
MAX_ITEM_BYTES = 8192


@dataclass(frozen=True)
class Musing:
    """Jeff's latest idle musing for a peer."""

    text: str
    created_at: datetime


_DDL_TABLE = sql.SQL(
    "CREATE TABLE IF NOT EXISTS musings ("
    "peer TEXT PRIMARY KEY, "
    "text TEXT NOT NULL, "
    "created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
)

# Columns the queries below rely on — checked at startup by the shared drift
# guard so an older musings table missing one fails loudly at deploy, not mid-turn.
_EXPECTED_COLUMNS = frozenset({"peer", "text", "created_at"})


class MusingStore:
    """Async store of a peer's latest idle musing (plain Postgres, one row/peer).

    No embedder, no vector column — a musing is injected by recency, never
    semantically recalled, so it needs no embedding.
    """

    def __init__(self, pool: AsyncConnectionPool):
        self._pool = pool

    @classmethod
    async def create(cls, pool: AsyncConnectionPool) -> "MusingStore":
        s = cls(pool)
        await s._init_schema()
        return s

    async def _init_schema(self) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_DDL_TABLE)
                await assert_columns(cur, "musings", _EXPECTED_COLUMNS)
            await conn.commit()

    async def record(self, peer: str, text: str) -> None:
        """Store this peer's latest idle musing, replacing any previous one.

        ``created_at`` is bumped to ``now()`` on every write (it's the recency
        key the reader compares against the last turn). Blank text is a no-op so
        a self-turn where Jeff did/said nothing leaves no stale musing.
        """
        text = strip_chat_template_tokens(text).strip()
        if not text:
            return
        encoded = text.encode("utf-8")
        if len(encoded) > MAX_ITEM_BYTES:
            text = encoded[:MAX_ITEM_BYTES].decode("utf-8", "ignore").rstrip()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO musings (peer, text, created_at) "
                    "VALUES (%s, %s, now()) "
                    "ON CONFLICT (peer) DO UPDATE "
                    "SET text = EXCLUDED.text, created_at = now()",
                    (peer, text),
                )
            await conn.commit()

    async def latest(self, peer: str) -> Musing | None:
        """This peer's most recent idle musing, or None if there is none."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT text, created_at FROM musings WHERE peer = %s",
                    (peer,),
                )
                row = await cur.fetchone()
        if not row:
            return None
        return Musing(text=strip_chat_template_tokens(row[0]), created_at=row[1])

    async def forget(self, peer: str) -> int:
        """Delete this peer's musing (so /forget wipes it too)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM musings WHERE peer = %s", (peer,))
                deleted = cur.rowcount
            await conn.commit()
        return int(deleted)

    async def reset(self) -> None:
        """Destructive fresh start: drop the table and recreate it (called by
        ``reset-memory``; mirrors the other stores' reset)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP TABLE IF EXISTS musings")
            await conn.commit()
        await self._init_schema()
