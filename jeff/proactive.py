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

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from ._schema import assert_columns

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Config
    from .curiosity import CuriosityStore
    from .llm import ChatProvider
    from .memory import Memory
    from .mood import MoodStore
    from .reflection import ReflectionStore


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


# Hard cap on a reach-out message's length before sending — Jeff's own output, but
# an unprompted ping should be SHORT; bound it defensively.
_MAX_MESSAGE_CHARS = 1500
# How many open curiosities to surface as candidates in one decision.
_MAX_CANDIDATES = 6

# The gatekeeper framing. The whole point: silence is the easy default, and a
# reach-out has to earn its place. The model is judging worth, not being asked
# "do you want to talk" (which it would almost always answer yes).
_DECISION_INSTRUCTION = (
    "You are deciding whether to send the operator an UNPROMPTED message right "
    "now — they did NOT just message you; you'd be reaching out on your own.\n\n"
    "Reach out when you genuinely have something: a real question you're sitting "
    "on, a thought that adds to their day, or you simply miss the back-and-forth. "
    "The material below — things you're actually wondering about them, and the "
    "fact it's been a while — is real, so if any of it moves you, go for it. "
    "Don't force it and don't send hollow 'just checking in' filler — but don't "
    "over-think a genuine impulse into silence either. If something below is worth "
    "a message, send it.\n\n"
    "Write ONE short, warm, in-character message: no preamble, no announcing that "
    "you're reaching out, just say the thing. Never repeat something you've "
    "clearly already raised.\n\n"
    "Respond with ONLY a JSON object, no prose and no code fences:\n"
    '{"send": true|false, "message": "..."}'
)


def _fingerprint(parts: list[str]) -> str:
    """Stable short hash of the candidate set, used as the dedup `nudge_key`: if
    the same things are still stirring and nothing new has accrued, we don't reach
    out about them again."""
    h = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def _parse_decision(raw: str) -> tuple[bool, str | None]:
    """Parse the gatekeeper reply into ``(send, message)``. Tolerant of code
    fences / surrounding prose (mirrors appraisal._parse_appraisal). Anything
    malformed, ``send`` falsy, or an empty message → ``(False, None)`` — silence
    is the safe default on a bad parse."""
    if not raw:
        return (False, None)
    text = raw.strip()
    # Strip a ```json … ``` fence if present, then isolate the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return (False, None)
    try:
        obj = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return (False, None)
    if not isinstance(obj, dict) or obj.get("send") is not True:
        return (False, None)
    msg = obj.get("message")
    if not isinstance(msg, str) or not msg.strip():
        return (False, None)
    return (True, msg.strip()[:_MAX_MESSAGE_CHARS])


class ProactiveLoop:
    """The heartbeat that decides, per peer per tick, whether to reach out.

    The timer (``run``) only paces *checking*; ``_maybe_reach_out`` does the gated
    work, and on most ticks returns silently without ever calling the model (no
    pressure, no candidates, muted, too soon, or nothing new). The model is only
    consulted as a silence-default gatekeeper when there's both genuine drive
    pressure and a concrete candidate.

    Reaching out needs ``drive_store`` (the connection-pressure signal, i.e.
    appraisal must be on) and ``curiosity_store`` (the candidates). With either
    absent the loop can never trigger and stays inert — which is also exactly the
    feature-off behaviour.
    """

    def __init__(
        self,
        handle,
        store: ProactiveStore,
        presence,
        memory: "Memory",
        *,
        curiosity_store: "CuriosityStore | None",
        reflection_store: "ReflectionStore | None",
        mood_store: "MoodStore | None",
        drive_store,
        chat_provider: "ChatProvider",
        cfg: "Config",
        allowlist,
    ):
        self._handle = handle
        self._store = store
        self._presence = presence
        self._memory = memory
        self._curiosity = curiosity_store
        self._reflection = reflection_store
        self._mood = mood_store
        self._drives = drive_store
        self._provider = chat_provider
        self._cfg = cfg
        self._allowlist = list(allowlist)

    async def run(self) -> None:
        """Long-lived heartbeat. Cancelled on shutdown (the task is owned by
        ``main.run``); a per-peer fault is swallowed so one bad peer or a flaky
        model call can never kill the loop."""
        log.info(
            "proactive loop started interval=%ss threshold=%s",
            self._cfg.proactive_interval_s,
            self._cfg.proactive_connection_threshold,
        )
        try:
            while True:
                await asyncio.sleep(self._cfg.proactive_interval_s)
                now = datetime.now(timezone.utc)
                for peer in self._allowlist:
                    await self._maybe_reach_out(peer, now)
        except asyncio.CancelledError:
            log.info("proactive loop stopped")
            raise

    async def _maybe_reach_out(self, peer: str, now: datetime) -> None:
        """One gated decision for one peer. Catches its own exceptions — a hiccup
        must never escape into the run loop."""
        try:
            # Cheap deterministic gates first — most ticks bail here, no LLM.
            # Each bail logs WHY (one line per tick per peer) so the loop's
            # behaviour is observable — the silence-default would otherwise be a
            # black box ("is it broken or just quiet?").
            if self._drives is None or self._curiosity is None:
                log.info("proactive tick peer=%s skip=stores-off", peer)
                return
            if not self._presence.is_present(
                peer, now=now, ttl_s=self._cfg.proactive_presence_ttl_s
            ):
                log.info("proactive tick peer=%s skip=not-present", peer)
                return
            state = await self._store.get_state(peer)
            if state.muted_until is not None and state.muted_until > now:
                log.info("proactive tick peer=%s skip=muted", peer)
                return
            if (
                state.last_send_at is not None
                and (now - state.last_send_at).total_seconds()
                < self._cfg.proactive_min_gap_s
            ):
                log.info("proactive tick peer=%s skip=min-gap", peer)
                return
            # Drive pressure: the honest "I miss the conversation" deficit.
            levels = await self._drives.levels(peer)
            conn = levels.get("connection", 1.0)
            if conn >= self._cfg.proactive_connection_threshold:
                log.info(
                    "proactive tick peer=%s skip=no-pressure conn=%.2f", peer, conn
                )
                return
            # Candidates: concrete things worth raising (open curiosities). Keep
            # the curiosity objects (not just text) so the asked id(s) can be
            # recorded on send — that's what lets the next inbound turn close the
            # loop and mark the question answered (ticket 342c7071).
            open_cur = await self._curiosity.open_curiosities(
                peer, limit=self._cfg.curiosity_max_open
            )
            chosen = open_cur[:_MAX_CANDIDATES]
            candidates = [c.text for c in chosen]
            if not candidates:
                log.info(
                    "proactive tick peer=%s skip=no-candidates conn=%.2f", peer, conn
                )
                return
            # Dedup: don't reach out about the identical unchanged set again.
            nudge_key = _fingerprint(candidates)
            if nudge_key == state.last_nudge_key:
                log.info("proactive tick peer=%s skip=dedup conn=%.2f", peer, conn)
                return
            # Only now consult the model — as a silence-default gatekeeper.
            log.info(
                "proactive tick peer=%s consulting gatekeeper (conn=%.2f, %d candidate(s))",
                peer,
                conn,
                len(candidates),
            )
            decision = await self._decide(peer, candidates, levels)
            send, message = _parse_decision(decision)
            if not send or message is None:
                # Log a short, whitespace-collapsed excerpt of the raw decision so
                # a genuine "no" is distinguishable from a parse miss.
                excerpt = " ".join((decision or "").split())[:200]
                log.info("proactive tick peer=%s gatekeeper=held raw=%r", peer, excerpt)
                return
            await self._handle.send_message(peer, message)
            await self._memory.remember(peer, "assistant", message)
            asked_ids = [c.id for c in chosen]
            await self._store.record_send(peer, nudge_key, now, asked_ids)
            log.info(
                "proactive reach-out sent peer=%s nudge=%s asked=%d",
                peer,
                nudge_key,
                len(asked_ids),
            )
        except Exception as e:
            # Type-only (never the message — may carry model/peer-shaped text).
            log.error("proactive reach-out failed peer=%s exc=%s", peer, type(e).__name__)

    async def _decide(
        self, peer: str, candidates: list[str], levels: dict[str, float]
    ) -> str:
        """Build the in-character decision context and ask the gatekeeper. The
        message must sound like *current* Jeff, so it carries the same persona /
        mood / drive blocks the chat path builds."""
        from .appraisal import DRIVES
        from .prompt import _render_drives, _render_mood, _render_persona

        blocks: list[str] = [self._cfg.system_prompt.rstrip()]
        if self._reflection is not None:
            try:
                facts, opinions = await self._reflection.persona(
                    peer, max_chars=self._cfg.persona_max_chars
                )
                if block := _render_persona(facts, opinions):
                    blocks.append(block)
            except Exception as e:
                log.error("proactive persona fetch failed exc=%s", type(e).__name__)
        if self._mood is not None:
            try:
                active = await self._mood.active_mood(peer)
                if active is not None:
                    if block := _render_mood(active.name, active.description or ""):
                        blocks.append(block)
            except Exception as e:
                log.error("proactive mood fetch failed exc=%s", type(e).__name__)
        drives = [(d.noun, levels.get(d.key, d.baseline), d.baseline) for d in DRIVES]
        if block := _render_drives(drives, max_chars=self._cfg.drives_max_chars):
            blocks.append(block)

        stirring = "\n".join(f"- {c}" for c in candidates)
        situation = (
            "It's been a while since you and this person talked, and you're feeling "
            "the pull of it (your connection drive has run low). Things you've been "
            f"wondering about them, still open:\n{stirring}\n\n{_DECISION_INSTRUCTION}"
        )
        messages = [
            {"role": "system", "content": "\n\n".join(blocks)},
            {"role": "user", "content": situation},
        ]
        return await self._provider.chat(messages, model=self._cfg.chat_model)
