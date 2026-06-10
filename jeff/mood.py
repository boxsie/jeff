"""Mood drive — Jeff carries a short-lived, self-chosen affective state.

Third buildable slice of the inner-life work and the first feature **Jeff itself
asked for**: a mood ("today I woke up playful / soft / restless") that colours
how Jeff talks for a few hours to a day, then fades. It's the *affective* leg
alongside the two cognitive drives already shipped — curiosity (open *questions*)
and reflection (settled *opinions*). Where those accumulate a semantic set, a
mood is a **single temporal state**, so this is deliberately NOT a pgvector store:

- one current state, not a growing, ranked set;
- **temporal** — expiry is the whole mechanic (neither other drive has it);
- never recalled semantically — you only ever ask "what's active right now?".

So there's no embedding column and no ``_guard_embed_dim`` ceremony — just two
plain tables and SQL ``now()`` arithmetic:

- ``mood_definitions(peer, name, description, …)`` — Jeff's **own**, editable
  vocabulary: what each mood *means to me*, authored at runtime via the
  ``define_mood`` tool. The affective twin of reflection's self-authored opinions.
- ``mood_state(peer, name, started_at, expires_at, source, note)`` — an append
  log of set moods; the one non-expired row is the active mood. ``source``
  distinguishes ``jeff`` (this slice — set deliberately, Jeff's "I'd rather
  decide") from a future ``spontaneous`` drift source, so the follow-up needs no
  schema change.

The active mood is surfaced back into context by ``prompt.build_history``'s
"## How you're feeling right now" block — **additive**: it tints tone but never
overrides the operator-owned base prompt's character or boundaries.

Everything here is gated by ``JEFF_MOOD_ENABLED`` (default off): when off the
store/tools are never constructed, so behaviour is byte-identical to today.

Public-repo hygiene: the *mechanism* (tables, render block) is neutral; the
*content* (mood names + definitions) lives only in the DB, authored at runtime,
never committed and never in config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from ._schema import assert_columns
from .screen import strip_chat_template_tokens


log = logging.getLogger("jeff.mood")


# Hard ceiling on a stored definition's bytes — defence-in-depth, far above any
# real one-paragraph definition. The tool also enforces the (tighter, operator-
# tunable) JEFF_MOOD_MAX_CHARS before calling in.
MAX_DESC_BYTES = 8192
# A mood name is a short label, not prose; bound it tight so it stays scannable
# in the prompt block and can't smuggle a paragraph in through the name field.
MAX_NAME_BYTES = 128

# Default source for a deliberately-set mood (this slice). The follow-up
# spontaneous-drift slice writes 'spontaneous' into the same column.
SOURCE_JEFF = "jeff"


@dataclass(frozen=True)
class MoodDefinition:
    """Jeff's own definition of one mood (what it means to me)."""

    name: str
    description: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ActiveMood:
    """The currently-active mood for a peer. ``description`` is the joined
    definition text, or ``None`` when the mood was set before being defined."""

    name: str
    description: str | None
    started_at: datetime
    expires_at: datetime
    source: str


def normalize_name(name: str) -> str:
    """Canonicalise a mood name so ``set_mood`` and ``define_mood`` agree on the
    join key: strip template tokens, collapse internal whitespace, lowercase.

    Lowercasing means "Playful" and "playful" resolve to the same definition; mood
    names are short labels, so case carries no meaning worth preserving.
    """
    name = strip_chat_template_tokens(name)
    return " ".join(name.split()).lower()


_DDL_DEFINITIONS = sql.SQL(
    "CREATE TABLE IF NOT EXISTS mood_definitions ("
    "id BIGSERIAL PRIMARY KEY, "
    "peer TEXT NOT NULL, "
    "name TEXT NOT NULL, "
    "description TEXT NOT NULL, "
    "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
    "updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
    "UNIQUE (peer, name))"
)

_DDL_STATE = sql.SQL(
    "CREATE TABLE IF NOT EXISTS mood_state ("
    "id BIGSERIAL PRIMARY KEY, "
    "peer TEXT NOT NULL, "
    "name TEXT NOT NULL, "
    "started_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
    "expires_at TIMESTAMPTZ NOT NULL, "
    "source TEXT NOT NULL DEFAULT 'jeff', "
    "note TEXT)"
)

# The active-mood lookup orders by expires_at within the still-valid rows, so the
# index that serves it also serves the "is anything active?" probe.
_DDL_IDX_STATE = sql.SQL(
    "CREATE INDEX IF NOT EXISTS idx_mood_state_peer_active "
    "ON mood_state(peer, expires_at DESC)"
)

# Columns the queries below rely on — checked at startup by the shared drift
# guard so an older mood table missing one fails loudly at deploy, not mid-turn.
_DEFINITIONS_COLUMNS = frozenset(
    {"id", "peer", "name", "description", "created_at", "updated_at"}
)
_STATE_COLUMNS = frozenset(
    {"id", "peer", "name", "started_at", "expires_at", "source", "note"}
)


class MoodStore:
    """Async store of a peer's mood definitions + state (plain Postgres).

    No embedder, no vector column — a mood is temporal single-state, not a
    semantic set (see module docstring). All time arithmetic is done in SQL with
    ``now()`` so the database clock is the single source of truth for expiry.
    """

    def __init__(self, pool: AsyncConnectionPool):
        self._pool = pool

    @classmethod
    async def create(cls, pool: AsyncConnectionPool) -> "MoodStore":
        s = cls(pool)
        await s._init_schema()
        return s

    async def _init_schema(self) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_DDL_DEFINITIONS)
                await cur.execute(_DDL_STATE)
                await cur.execute(_DDL_IDX_STATE)
                await assert_columns(cur, "mood_definitions", _DEFINITIONS_COLUMNS)
                await assert_columns(cur, "mood_state", _STATE_COLUMNS)
            await conn.commit()

    # --- definitions (Jeff's own vocabulary) -------------------------------

    async def define(self, peer: str, name: str, description: str) -> str:
        """Create or edit Jeff's definition of a mood (upsert on ``(peer, name)``).

        Returns ``"created"`` for a brand-new mood, ``"updated"`` when an existing
        definition's text was replaced, or ``""`` when the name/description was
        blank. The name is normalised so it matches what ``set_mood`` later looks
        up.
        """
        name = normalize_name(name)
        description = strip_chat_template_tokens(description).strip()
        if not name or not description:
            return ""
        if len(name.encode("utf-8")) > MAX_NAME_BYTES:
            raise ValueError(f"mood name too long (limit {MAX_NAME_BYTES} bytes)")
        if len(description.encode("utf-8")) > MAX_DESC_BYTES:
            raise ValueError(
                f"mood definition too long (limit {MAX_DESC_BYTES} bytes)"
            )
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                # xmax = 0 is Postgres' canonical "was this an INSERT?" signal on
                # an upsert: a fresh insert has no delete-marker, an update does.
                await cur.execute(
                    "INSERT INTO mood_definitions (peer, name, description) "
                    "VALUES (%s, %s, %s) "
                    "ON CONFLICT (peer, name) DO UPDATE "
                    "SET description = EXCLUDED.description, updated_at = now() "
                    "RETURNING (xmax = 0)",
                    (peer, name, description),
                )
                row = await cur.fetchone()
            await conn.commit()
        return "created" if row and row[0] else "updated"

    async def get_definition(self, peer: str, name: str) -> str | None:
        """The definition text for one mood name, or ``None`` if undefined."""
        name = normalize_name(name)
        if not name:
            return None
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT description FROM mood_definitions "
                    "WHERE peer = %s AND name = %s",
                    (peer, name),
                )
                row = await cur.fetchone()
        return strip_chat_template_tokens(row[0]) if row else None

    async def list_definitions(self, peer: str) -> list[MoodDefinition]:
        """A peer's defined moods, most-recently-updated first (for /mind)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT name, description, created_at, updated_at "
                    "FROM mood_definitions WHERE peer = %s "
                    "ORDER BY updated_at DESC, id DESC",
                    (peer,),
                )
                rows = await cur.fetchall()
        return [
            MoodDefinition(
                name=r[0],
                description=strip_chat_template_tokens(r[1]),
                created_at=r[2],
                updated_at=r[3],
            )
            for r in rows
        ]

    # --- state (the active mood) -------------------------------------------

    async def set_mood(
        self,
        peer: str,
        name: str,
        *,
        hours: float,
        source: str = SOURCE_JEFF,
        note: str | None = None,
    ) -> datetime:
        """Set the active mood, superseding any current one (append a state row).

        ``hours`` is bounded by the caller (the tool); a floor of 1 second is
        enforced here as defence-in-depth so a zero/negative slips to a harmless
        already-expired-immediately rather than an open-ended mood. Returns the
        new ``expires_at`` (computed by the DB clock).
        """
        name = normalize_name(name)
        if not name:
            raise ValueError("mood name is empty")
        if len(name.encode("utf-8")) > MAX_NAME_BYTES:
            raise ValueError(f"mood name too long (limit {MAX_NAME_BYTES} bytes)")
        secs = max(1.0, float(hours) * 3600.0)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO mood_state (peer, name, expires_at, source, note) "
                    "VALUES (%s, %s, now() + make_interval(secs => %s), %s, %s) "
                    "RETURNING expires_at",
                    (peer, name, secs, source, note),
                )
                row = await cur.fetchone()
            await conn.commit()
        return row[0]

    async def active_mood(self, peer: str) -> ActiveMood | None:
        """The one non-expired mood for this peer (latest wins), or ``None``.

        Left-joins the definition so a single round-trip yields both the name and
        (if Jeff has defined it) the text that colours the prompt. A mood set but
        not yet defined still returns, with ``description=None``.
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT s.name, d.description, s.started_at, s.expires_at, s.source "
                    "FROM mood_state s "
                    "LEFT JOIN mood_definitions d "
                    "  ON d.peer = s.peer AND d.name = s.name "
                    "WHERE s.peer = %s AND s.expires_at > now() "
                    "ORDER BY s.started_at DESC, s.id DESC LIMIT 1",
                    (peer,),
                )
                row = await cur.fetchone()
        if row is None:
            return None
        return ActiveMood(
            name=row[0],
            description=strip_chat_template_tokens(row[1]) if row[1] else None,
            started_at=row[2],
            expires_at=row[3],
            source=row[4],
        )

    async def clear_mood(self, peer: str) -> int:
        """End any active mood now (expire it). Returns the number of rows ended.

        Expiring in place (rather than deleting) keeps ``mood_state`` an honest
        append log — the mood happened, it just ended early.
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE mood_state SET expires_at = now() "
                    "WHERE peer = %s AND expires_at > now()",
                    (peer,),
                )
                ended = cur.rowcount
            await conn.commit()
        return int(ended)

    async def forget(self, peer: str) -> int:
        """Delete every mood + definition for one peer (so /forget wipes these too).
        Returns the total rows deleted across both tables."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM mood_state WHERE peer = %s", (peer,))
                deleted = cur.rowcount
                await cur.execute(
                    "DELETE FROM mood_definitions WHERE peer = %s", (peer,)
                )
                deleted += cur.rowcount
            await conn.commit()
        return int(deleted)

    async def count_definitions(self, peer: str) -> int:
        """How many moods this peer has defined (for /stats-style summaries)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT count(*) FROM mood_definitions WHERE peer = %s", (peer,)
                )
                row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def reset(self) -> None:
        """Destructive fresh start: drop both tables and recreate them (called by
        ``reset-memory``; mirrors CuriosityStore.reset)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP TABLE IF EXISTS mood_state")
                await cur.execute("DROP TABLE IF EXISTS mood_definitions")
            await conn.commit()
        await self._init_schema()
