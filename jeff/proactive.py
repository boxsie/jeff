"""Proactive bookkeeping store — the floor/mute/dedup state behind reaching out.

The motivation arc's plank 4 began as a standalone heartbeat (`ProactiveLoop`)
that decided, with a `{"send": …}` gatekeeper, whether to message the operator.
Slice 2 of the idle self-turn (ticket 51e22abf) retired that gatekeeper: reaching
out is now a *verb* (`tools.reach.ReachOutTool`) the model invokes from inside the
self-turn (`selfturn.py`), and the door-to-the-operator gate (presence + min-gap
+ mute) lives in that tool. One loop, the model decides by calling the verb.

What survives here is the **state** both the verb and the `/mute` command share:

- ``last_send_at`` — the anti-machine-gun floor the reach_out tool checks.
- ``muted_until`` — the operator's manual mute window (`/mute`).
- ``last_nudge_key`` — a dedup marker stamped on each reach-out.
- ``last_asked_curiosity_ids`` — the open-curiosity ids a reach-out was sitting on,
  consumed once by the curiosity resolver so a reply can close the loop (342c7071).

Guardrails stay deliberately minimal (single-operator sandbox): presence, a manual
`/mute`, and the min-gap fuse — no quiet hours, no cooldown ladder. Gated by
``JEFF_PROACTIVE_ENABLED`` (default off): when off the store and the reach_out
verb are never constructed, so the self-turn is inward-only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from ._schema import assert_columns


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
    "muted_until TIMESTAMPTZ, "
    "last_asked_curiosity_ids BIGINT[])"
)

# Idempotent column add for tables created before last_asked_curiosity_ids
# existed: CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so a fresh
# column would never appear on an already-deployed proactive_state (the exact
# schema-drift landmine filed as dbafb5ea). ADD COLUMN IF NOT EXISTS is a safe,
# self-applying migration — no reset-memory needed to pick up this column.
_DDL_MIGRATE_ASKED = sql.SQL(
    "ALTER TABLE proactive_state "
    "ADD COLUMN IF NOT EXISTS last_asked_curiosity_ids BIGINT[]"
)

# Columns the queries below rely on — checked at startup by the shared drift
# guard. The original dbafb5ea failure was a proactive_state from the reverted
# inner-life branch missing last_send_at/last_nudge_key/muted_until: those are a
# shape change the additive ALTER above can't express, so the guard names them at
# deploy (→ reset-memory) instead of letting get_state throw UndefinedColumn mid-turn.
_EXPECTED_COLUMNS = frozenset(
    {
        "peer",
        "last_send_at",
        "last_nudge_key",
        "muted_until",
        "last_asked_curiosity_ids",
    }
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
                await cur.execute(_DDL_MIGRATE_ASKED)
                await assert_columns(cur, "proactive_state", _EXPECTED_COLUMNS)
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

    async def record_send(
        self,
        peer: str,
        nudge_key: str,
        now: datetime,
        asked_curiosity_ids: Sequence[int] | None = None,
    ) -> None:
        """Stamp a reach-out: update the floor timestamp + dedup key, and record
        which open curiosity id(s) fuelled this unprompted message so the next
        inbound turn can close the loop on them. An empty/absent list stores NULL.
        """
        asked = [int(i) for i in asked_curiosity_ids] if asked_curiosity_ids else None
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO proactive_state "
                    "  (peer, last_send_at, last_nudge_key, last_asked_curiosity_ids) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (peer) DO UPDATE SET "
                    "  last_send_at = EXCLUDED.last_send_at, "
                    "  last_nudge_key = EXCLUDED.last_nudge_key, "
                    "  last_asked_curiosity_ids = EXCLUDED.last_asked_curiosity_ids",
                    (peer, now, nudge_key, asked),
                )
            await conn.commit()

    async def take_asked_curiosity_ids(self, peer: str) -> list[int]:
        """Read AND clear the curiosity id(s) a proactive reach-out asked about.

        Consume-once: the asked ids are a hint that *this peer's next inbound
        message is likely the reply*, so they're cleared as they're handed out.
        Atomic via a writable CTE — the `cur` SELECT sees the pre-UPDATE snapshot,
        so the old value is returned even as the same statement nulls it. Returns
        [] when there's nothing pending (the common case, every ordinary turn).
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "WITH cur AS ("
                    "  SELECT last_asked_curiosity_ids AS ids "
                    "  FROM proactive_state WHERE peer = %s"
                    "), upd AS ("
                    "  UPDATE proactive_state SET last_asked_curiosity_ids = NULL "
                    "  WHERE peer = %s AND last_asked_curiosity_ids IS NOT NULL"
                    ") "
                    "SELECT ids FROM cur",
                    (peer, peer),
                )
                row = await cur.fetchone()
            await conn.commit()
        ids = row[0] if row and row[0] else []
        return [int(i) for i in ids]

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
