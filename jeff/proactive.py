"""Proactive autonomy loop — Jeff reaches out unprompted, naturally.

The motivation arc's plank 4. Until now Jeff is purely *reactive*: every store
(curiosity, reflection, mood, drives) is read only inside a turn the operator
starts, so the whole inner life is write-and-inject — state accumulates but never
*initiates*. This module is the missing consumer: a heartbeat that, between
operator turns, decides whether Jeff has something worth saying and reaches out.

The design separates three things a naive "ask the LLM if it wants to talk" loop
mashes together (which makes the model feel compelled to speak every tick):

1. **The timer paces *checking*, not speaking.** A short interval just wakes the
   loop to look. Frequent checking ≠ frequent messages — the next two layers gate.
2. **The trigger is real accrued state, computed with NO LLM.** Two cheap signals:
   *drive pressure* (the `connection` drive decayed below a threshold — the honest
   "I miss the conversation" deficit the decay tune produces) and *candidates*
   (concrete things worth raising, primarily Jeff's open curiosities). Zero
   candidates ⇒ the loop returns without ever calling the model ⇒ silent for free.
   Most ticks are silent and cost nothing.
3. **The LLM is a gatekeeper, not an author-on-command.** Only when pressure AND a
   candidate both hold is Jeff asked to decide — framed so silence is the easy
   default ("does this genuinely add to their day right now, or do you wait?").

Guardrails are deliberately minimal (this is a single-operator sandbox, not a
product): presence (don't shout into the void), a manual `/mute`, an
anti-machine-gun min-gap fuse, and a nudge-key dedup so the same thing isn't
raised twice running. No quiet hours, no contactable-hours, no cooldown ladder.

Everything is gated by ``JEFF_PROACTIVE_ENABLED`` (default off): when off the
store/loop are never constructed, so behaviour is byte-identical to today.

`ProactiveStore` lives here (the `/mute` command and the loop share one source of
truth); the `ProactiveLoop` heartbeat is appended below it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from psycopg import sql
from psycopg_pool import AsyncConnectionPool


log = logging.getLogger("jeff.proactive")


@dataclass(frozen=True)
class ProactiveState:
    """Per-peer proactive bookkeeping — just enough for the minimal gates."""

    peer: str
    last_send_at: datetime | None
    last_nudge_key: str | None
    muted_until: datetime | None


_DDL = sql.SQL(
    "CREATE TABLE IF NOT EXISTS proactive_state ("
    "peer TEXT PRIMARY KEY, "
    "last_send_at TIMESTAMPTZ, "
    "last_nudge_key TEXT, "
    "muted_until TIMESTAMPTZ)"
)


class ProactiveStore:
    """Per-peer proactive state (plain Postgres; mirrors MoodStore/DriveState).

    Stripped to the minimum the gates need: when Jeff last reached out
    (`last_send_at`, for the anti-machine-gun floor), what it last reached out
    about (`last_nudge_key`, for dedup), and a manual mute window (`muted_until`).
    No daily counters, no backoff ladder — those were product guardrails the
    operator cut.
    """

    def __init__(self, pool: AsyncConnectionPool):
        self._pool = pool

    @classmethod
    async def create(cls, pool: AsyncConnectionPool) -> "ProactiveStore":
        s = cls(pool)
        await s._init_schema()
        return s

    async def _init_schema(self) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_DDL)
            await conn.commit()

    async def get_state(self, peer: str) -> ProactiveState:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT last_send_at, last_nudge_key, muted_until "
                    "FROM proactive_state WHERE peer = %s",
                    (peer,),
                )
                row = await cur.fetchone()
        if row is None:
            return ProactiveState(peer, None, None, None)
        return ProactiveState(peer, row[0], row[1], row[2])

    async def record_send(self, peer: str, nudge_key: str, now: datetime) -> None:
        """Stamp a reach-out: update the floor timestamp + dedup key."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO proactive_state (peer, last_send_at, last_nudge_key) "
                    "VALUES (%s, %s, %s) "
                    "ON CONFLICT (peer) DO UPDATE SET "
                    "  last_send_at = EXCLUDED.last_send_at, "
                    "  last_nudge_key = EXCLUDED.last_nudge_key",
                    (peer, now, nudge_key),
                )
            await conn.commit()

    async def set_mute(self, peer: str, until: datetime | None) -> None:
        """Set (or clear, with ``until=None``) the operator's mute window."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO proactive_state (peer, muted_until) VALUES (%s, %s) "
                    "ON CONFLICT (peer) DO UPDATE SET muted_until = EXCLUDED.muted_until",
                    (peer, until),
                )
            await conn.commit()

    async def forget(self, peer: str) -> int:
        """Delete one peer's proactive state (so /forget wipes it too)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM proactive_state WHERE peer = %s", (peer,)
                )
                deleted = cur.rowcount
            await conn.commit()
        return int(deleted)

    async def reset(self) -> None:
        """Destructive fresh start: drop + recreate (called by ``reset-memory``)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP TABLE IF EXISTS proactive_state")
            await conn.commit()
        await self._init_schema()
