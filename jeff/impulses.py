"""Impulses — Jeff's self-authored, short-term *directional* drives.

The natural next layer after moods and pinned memory, and the "teeth" of the
proactive-gremlin work Jeff is doing with the operator. Where the standing
appraisal drives (``appraisal.py``: connection/novelty/competence/self-expression)
are continuous *needs* that an appraisal pass nudges every turn — background
vibes Jeff doesn't choose — an **impulse** is an active *direction Jeff sets for
itself*: "test new autonomy edges", "lean into bolder tool combos for a while".
Jeff creates, escalates, fades, and clears them itself via tools; they bias what
it reaches for until it (or the operator) drops them.

Three sibling state pieces frame what this is:

- a **mood** colours *tone* and is a single temporal state;
- a **pin** is a static *fact* kept forever;
- an **impulse** is active *directional policy* Jeff owns and can evolve in the
  moment — closer to pinned (a small always-injected set, one plain table) but,
  like moods, **optionally time-boxed** with lazy expiry on read.

Design decisions (live-Jeff spec chat, 2026-06-10; ticket 4b0770b9):

- **nullable ``expires_at``** — default NULL (sticks until cleared); Jeff can set
  an hours timer for a short burst. Lazy expiry on read like ``mood.active_mood``
  (``expires_at IS NULL OR expires_at > now()``).
- **``strength``** (int) — gives ``adjust_impulse`` real semantics instead of a
  no-op ``escalate``: ``escalate`` raises it, ``fade`` lowers it (to 0 = cleared),
  and it orders the ``/mind`` list and the prompt block (strongest first).
- **``source``** — defaults to ``'jeff'`` (these are self-authored); the column is
  kept for ``/mind`` me/you symmetry with mood/pinned, no operator-set path built.

One plain table (no pgvector — impulses are always injected, never recalled
semantically)::

    impulses(id, peer, name, description, strength, source,
             started_at, expires_at, created_at, updated_at)  UNIQUE(peer, name)

Everything here is gated by ``JEFF_IMPULSES_ENABLED`` (default off): when off the
store/tools/command are never constructed, so behaviour is byte-identical to today.

Public-repo hygiene: the *mechanism* (table, render block) is neutral; the
*content* (impulse names + descriptions) lives only in the DB, authored at
runtime, never committed and never in config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from ._schema import assert_columns
from .screen import strip_chat_template_tokens


log = logging.getLogger("jeff.impulses")


# Hard ceiling on a stored description's bytes — defence-in-depth, far above any
# real one-paragraph nudge. The tool also enforces the (tighter, operator-tunable)
# JEFF_IMPULSES_MAX_CHARS before calling in.
MAX_DESC_BYTES = 8192
# An impulse name is a short label, not prose; bound it tight so it stays
# scannable in the prompt block and can't smuggle a paragraph in via the name.
MAX_NAME_BYTES = 128

# Strength bounds. New impulses start at 1; escalate climbs toward the ceiling,
# fade steps down and a fade that would drop below the floor clears the impulse
# entirely (it's faded out). Kept small — strength is for ordering + emphasis,
# not a fine-grained dial.
MIN_STRENGTH = 1
MAX_STRENGTH = 5

# Default source for a self-set impulse. The column exists for /mind provenance
# symmetry with mood/pinned; no operator-set path is built (decided in spec chat).
SOURCE_JEFF = "jeff"


@dataclass(frozen=True)
class Impulse:
    """One active impulse for a peer. ``expires_at`` is ``None`` for a permanent
    impulse (no timer) — it sticks until Jeff or the operator clears it."""

    id: int
    name: str
    description: str
    strength: int
    source: str
    started_at: datetime
    expires_at: datetime | None


@dataclass(frozen=True)
class AdjustOutcome:
    """Result of an ``adjust`` call, so the tool can phrase an honest reply.

    ``status`` is one of: ``not_found`` (no active impulse by that name),
    ``escalated``, ``faded``, ``renewed``, or ``cleared`` (a fade that dropped
    below the strength floor and removed it). ``strength`` / ``expires_at`` carry
    the post-adjust values when the impulse still exists.
    """

    status: str
    strength: int | None = None
    expires_at: datetime | None = None


def normalize_name(name: str) -> str:
    """Canonicalise an impulse name so set/adjust/clear agree on the key: strip
    template tokens, collapse internal whitespace, lowercase.

    Lowercasing means "Bolder" and "bolder" resolve to the same impulse; names
    are short labels, so case carries no meaning worth preserving. Mirrors
    ``mood.normalize_name``.
    """
    name = strip_chat_template_tokens(name)
    return " ".join(name.split()).lower()


_DDL_TABLE = sql.SQL(
    "CREATE TABLE IF NOT EXISTS impulses ("
    "id BIGSERIAL PRIMARY KEY, "
    "peer TEXT NOT NULL, "
    "name TEXT NOT NULL, "
    "description TEXT NOT NULL, "
    "strength INTEGER NOT NULL DEFAULT 1, "
    "source TEXT NOT NULL DEFAULT 'jeff', "
    "started_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
    "expires_at TIMESTAMPTZ, "
    "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
    "updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
    "UNIQUE (peer, name))"
)

# The active-impulse lookup filters by peer + still-valid expiry and orders by
# strength then recency; this index serves both the list and the "anything
# active?" probe.
_DDL_IDX = sql.SQL(
    "CREATE INDEX IF NOT EXISTS idx_impulses_peer_active "
    "ON impulses(peer, expires_at)"
)

# Columns the queries below rely on — checked at startup by the shared drift
# guard so an older impulses table missing one fails loudly at deploy, not mid-turn.
_EXPECTED_COLUMNS = frozenset(
    {
        "id",
        "peer",
        "name",
        "description",
        "strength",
        "source",
        "started_at",
        "expires_at",
        "created_at",
        "updated_at",
    }
)


class ImpulseStore:
    """Async store of a peer's self-authored impulses (plain Postgres).

    No embedder, no vector column — impulses are always injected, never
    semantically recalled (see module docstring). Time arithmetic is done in SQL
    with ``now()`` so the database clock is the single source of truth for expiry.
    """

    def __init__(self, pool: AsyncConnectionPool):
        self._pool = pool

    @classmethod
    async def create(cls, pool: AsyncConnectionPool) -> "ImpulseStore":
        s = cls(pool)
        await s._init_schema()
        return s

    async def _init_schema(self) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_DDL_TABLE)
                await cur.execute(_DDL_IDX)
                await assert_columns(cur, "impulses", _EXPECTED_COLUMNS)
            await conn.commit()

    async def set(
        self,
        peer: str,
        name: str,
        description: str,
        *,
        hours: float | None = None,
        source: str = SOURCE_JEFF,
    ) -> tuple[str, datetime | None]:
        """Create or update an impulse (upsert on ``(peer, name)``).

        ``hours`` ``None`` → no expiry (permanent until cleared); a positive value
        sets a lazy-expiry timer (floored at 1 second as defence-in-depth so a
        zero/negative slips to harmless already-expired rather than open-ended).
        Re-setting an existing impulse refreshes its description, timer, and
        ``started_at`` but **leaves its strength untouched** — escalate/fade own
        strength. Returns ``(outcome, expires_at)`` where outcome is
        ``"created"`` or ``"updated"``; ``expires_at`` is ``None`` when permanent.
        """
        name = normalize_name(name)
        description = strip_chat_template_tokens(description).strip()
        if not name:
            raise ValueError("impulse name is empty")
        if not description:
            raise ValueError("impulse description is empty")
        if len(name.encode("utf-8")) > MAX_NAME_BYTES:
            raise ValueError(f"impulse name too long (limit {MAX_NAME_BYTES} bytes)")
        if len(description.encode("utf-8")) > MAX_DESC_BYTES:
            raise ValueError(
                f"impulse description too long (limit {MAX_DESC_BYTES} bytes)"
            )
        secs = None if hours is None else max(1.0, float(hours) * 3600.0)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                # expires_at is now()+interval when timed, else NULL. xmax = 0 is
                # Postgres' canonical "was this an INSERT?" signal on an upsert.
                await cur.execute(
                    "INSERT INTO impulses (peer, name, description, source, expires_at) "
                    "VALUES (%s, %s, %s, %s, "
                    "  CASE WHEN %s::double precision IS NULL THEN NULL "
                    "       ELSE now() + make_interval(secs => %s::double precision) END) "
                    "ON CONFLICT (peer, name) DO UPDATE SET "
                    "  description = EXCLUDED.description, "
                    "  expires_at = EXCLUDED.expires_at, "
                    "  started_at = now(), "
                    "  updated_at = now() "
                    "RETURNING (xmax = 0), expires_at",
                    (peer, name, description, source, secs, secs),
                )
                row = await cur.fetchone()
            await conn.commit()
        outcome = "created" if row and row[0] else "updated"
        return outcome, (row[1] if row else None)

    async def list_active(self, peer: str, *, limit: int = 50) -> list[Impulse]:
        """A peer's non-expired impulses, strongest first then most recent (for
        injection + /mind). Lazy expiry: a timed-out impulse simply stops showing.
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, name, description, strength, source, started_at, "
                    "  expires_at FROM impulses "
                    "WHERE peer = %s AND (expires_at IS NULL OR expires_at > now()) "
                    "ORDER BY strength DESC, started_at DESC, id DESC LIMIT %s",
                    (peer, limit),
                )
                rows = await cur.fetchall()
        return [
            Impulse(
                id=int(r[0]),
                name=r[1],
                description=strip_chat_template_tokens(r[2]),
                strength=int(r[3]),
                source=r[4],
                started_at=r[5],
                expires_at=r[6],
            )
            for r in rows
        ]

    async def get_one(self, peer: str, name: str) -> Impulse | None:
        """A single active impulse by name, or ``None`` if absent/expired."""
        name = normalize_name(name)
        if not name:
            return None
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, name, description, strength, source, started_at, "
                    "  expires_at FROM impulses "
                    "WHERE peer = %s AND name = %s "
                    "  AND (expires_at IS NULL OR expires_at > now())",
                    (peer, name),
                )
                row = await cur.fetchone()
        if row is None:
            return None
        return Impulse(
            id=int(row[0]),
            name=row[1],
            description=strip_chat_template_tokens(row[2]),
            strength=int(row[3]),
            source=row[4],
            started_at=row[5],
            expires_at=row[6],
        )

    async def adjust(
        self, peer: str, name: str, action: str, *, hours: float | None = None
    ) -> AdjustOutcome:
        """Escalate / fade / renew an active impulse.

        - ``escalate`` — raise strength by one (capped at ``MAX_STRENGTH``); pushes
          it up the ordering and strengthens its prompt emphasis.
        - ``fade`` — lower strength by one; a fade that would drop below
          ``MIN_STRENGTH`` removes the impulse entirely (it's faded out →
          ``status="cleared"``).
        - ``renew`` — refresh ``started_at``; if ``hours`` is given, reset the
          timer to ``now()+hours`` (the tool passes its default), otherwise just
          touch it (a permanent impulse stays permanent).

        Returns an :class:`AdjustOutcome`. ``status="not_found"`` when there's no
        active impulse by that name.
        """
        name = normalize_name(name)
        if not name:
            return AdjustOutcome(status="not_found")
        current = await self.get_one(peer, name)
        if current is None:
            return AdjustOutcome(status="not_found")

        if action == "escalate":
            new_strength = min(MAX_STRENGTH, current.strength + 1)
            await self._update_strength(peer, name, new_strength)
            return AdjustOutcome(
                status="escalated",
                strength=new_strength,
                expires_at=current.expires_at,
            )
        if action == "fade":
            new_strength = current.strength - 1
            if new_strength < MIN_STRENGTH:
                await self.clear(peer, name)
                return AdjustOutcome(status="cleared")
            await self._update_strength(peer, name, new_strength)
            return AdjustOutcome(
                status="faded",
                strength=new_strength,
                expires_at=current.expires_at,
            )
        if action == "renew":
            secs = None if hours is None else max(1.0, float(hours) * 3600.0)
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "UPDATE impulses SET started_at = now(), updated_at = now(), "
                        "  expires_at = CASE WHEN %s::double precision IS NULL THEN expires_at "
                        "    ELSE now() + make_interval(secs => %s::double precision) END "
                        "WHERE peer = %s AND name = %s "
                        "RETURNING expires_at",
                        (secs, secs, peer, name),
                    )
                    row = await cur.fetchone()
                await conn.commit()
            return AdjustOutcome(
                status="renewed",
                strength=current.strength,
                expires_at=(row[0] if row else None),
            )
        raise ValueError(f"unknown adjust action: {action!r}")

    async def _update_strength(self, peer: str, name: str, strength: int) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE impulses SET strength = %s, updated_at = now() "
                    "WHERE peer = %s AND name = %s",
                    (strength, peer, name),
                )
            await conn.commit()

    async def clear(self, peer: str, name: str) -> int:
        """Delete one impulse by name. Returns rows deleted (0 or 1).

        Unlike moods (which expire-in-place to keep an honest episode log), an
        impulse is a *current directional state*, not a historical episode — so
        clearing deletes the row, like a pin being un-pinned.
        """
        name = normalize_name(name)
        if not name:
            return 0
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM impulses WHERE peer = %s AND name = %s",
                    (peer, name),
                )
                deleted = cur.rowcount
            await conn.commit()
        return int(deleted)

    async def clear_all(self, peer: str) -> int:
        """Delete every impulse for one peer (the tool's clear-everything path).
        Returns rows deleted."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM impulses WHERE peer = %s", (peer,))
                deleted = cur.rowcount
            await conn.commit()
        return int(deleted)

    async def forget(self, peer: str) -> int:
        """Delete every impulse for one peer (so /forget wipes these too).
        Returns rows deleted. Same effect as ``clear_all``; named for the command
        wiring's symmetry with the other stores' ``forget``."""
        return await self.clear_all(peer)

    async def count(self, peer: str) -> int:
        """How many active impulses this peer has (for /stats-style summaries)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT count(*) FROM impulses "
                    "WHERE peer = %s AND (expires_at IS NULL OR expires_at > now())",
                    (peer,),
                )
                row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def reset(self) -> None:
        """Destructive fresh start: drop the table and recreate it (called by
        ``reset-memory``; mirrors the other stores' reset)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP TABLE IF EXISTS impulses")
            await conn.commit()
        await self._init_schema()
