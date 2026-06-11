"""The idle self-turn — Jeff's first taste of agency over its own inner life.

Where the proactive loop (`proactive.py`) asks a yes/no "should I message the
operator?", this asks "here's your state — what, if anything, do you want to
do?" and hands Jeff a real tool-enabled turn to answer it. The verbs are
INWARD-ONLY in this slice: shift a mood, form/adjust an impulse, pin a thought,
scan its own memory. Nothing here reaches the operator, so it fires **even when
they're offline** — which is the whole point: Jeff drifts between sessions
instead of sitting frozen until the next message. (The outward `reach_out` edge,
and folding the proactive gatekeeper in, is a later slice.)

Cost discipline mirrors the proactive loop: the timer only paces *checking*;
`_maybe_self_turn` bails cheaply (logging why) on most ticks — within the
min-gap, or nothing worth chewing on — and only spends an LLM call when there's
genuinely something to act on. The min-gap is tracked in memory for this slice
(resets on restart; durable bookkeeping is a follow-up), which is fine: the cost
of one extra self-turn after a redeploy is a single bounded turn.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .appraisal import DRIVES, DriveReading, classify_pressure
from .prompt import DRIVE_BAND_MARGIN, build_self_turn_messages
from .turnloop import run_tool_loop

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Config
    from .curiosity import CuriosityStore
    from .impulses import ImpulseStore
    from .llm import ChatProvider
    from .memory import Memory
    from .mood import MoodStore
    from .reflection import ReflectionStore
    from .appraisal import DriveState
    from .tools.base import ToolRegistry


log = logging.getLogger("jeff.selfturn")


# A banked drive not spent within this window counts as "idle" (hoarded) for the
# spend-pressure guard (slice b3) — long enough that an actively-spent drive
# doesn't trip it, short enough that a genuinely hoarded balance gets a nudge
# before the leak quietly bleeds it away. 6 hours (matches connection's leak).
_SPEND_PRESSURE_WINDOW_S = 6 * 3600.0


def _pressure_addendum(has_banked: bool, has_depleted: bool) -> str:
    """A standing self-turn note encoding the two affordability guards (slice b3),
    appended to the instruction only when there's pressure to act on. Drive-name-
    free (the drives block already names them) — this is the *principle*: hoarding
    does nothing, and a depleted Jeff should make a cheap recovering move rather
    than withdraw. Empty string when neither pressure applies (no nudge)."""
    if has_banked and has_depleted:
        return (
            "Where you're at: you're sitting on banked drives with nothing in motion "
            "— a good moment to actually use them — while running low on others, "
            "where one small, cheap move to feed them beats withdrawing. Hoarding "
            "leaks away and waiting it out fixes nothing; a little, done well, is "
            "worth more than either."
        )
    if has_banked:
        return (
            "Where you're at: you're sitting on banked drives with nothing in motion. "
            "Hoarding them does nothing — they just bleed away. If something "
            "genuinely calls, this is a good moment to put them to use."
        )
    if has_depleted:
        return (
            "Where you're at: you're running low on a drive or two. One small, cheap "
            "move to feed one beats withdrawing and waiting it out — you don't need a "
            "full tank to make a single good move."
        )
    return ""


# Jeff's own synthetic nudge — the "user" message of a self-turn. It frames the
# moment as inward and unforced: doing nothing is a valid outcome, and Jeff must
# not manufacture activity just because it was prompted (the spitball's
# drift/thrash watch-out). Deliberately NOT wrapped in <peer_message> by
# build_self_turn_messages — it's Jeff's own, the thing we want it to act on.
_SELF_TURN_INSTRUCTION = (
    "Quiet moment — no one's messaging you right now, this is your own time. "
    "Above is where things stand: how you're feeling, what you're driving toward, "
    "what you've been curious about, and your recent conversation.\n\n"
    "Take a beat and decide what, if anything, you want to do with it. You might "
    "shift your mood, form or adjust an impulse, pin a thought worth keeping, or "
    "look back through your memory — or simply sit with it and do nothing. Don't "
    "manufacture activity: only act if something genuinely moves you. Nothing you "
    "do here is sent to anyone — it's just you, with yourself."
)

# Used when the outward `reach_out` verb is available (proactive enabled): the
# inward framing still leads — most moments are inward — but reaching out is on
# the table when something genuinely warrants it AND the door is open. The tool
# itself gates presence/min-gap/mute, so the model can try and be told "not now".
_SELF_TURN_INSTRUCTION_OUTWARD = (
    "Quiet moment — no one's messaging you right now, this is your own time. "
    "Above is where things stand: how you're feeling, what you're driving toward, "
    "what you've been curious about, and your recent conversation.\n\n"
    "Take a beat and decide what, if anything, you want to do with it. Mostly this "
    "is inward: shift your mood, form or adjust an impulse, pin a thought worth "
    "keeping, look back through your memory — or simply sit with it and do nothing. "
    "And if something genuinely worth their attention has come up, you can reach "
    "out and message them — but keep that rare; most moments the inward stuff is "
    "enough, and they may not be around. Don't manufacture activity: only act if "
    "something really moves you."
)


class SelfTurnLoop:
    """Heartbeat that periodically hands Jeff an inward, tool-enabled self-turn.

    Inert (and cheap) unless there's something to act on. Needs the inward
    registry (the verbs) plus the state stores it reads for the gate and the
    prompt blocks; with all of them absent it simply never finds anything to
    chew on and stays silent — which is also the feature-off behaviour.
    """

    def __init__(
        self,
        memory: "Memory",
        registry: "ToolRegistry",
        chat_provider: "ChatProvider",
        *,
        curiosity_store: "CuriosityStore | None",
        reflection_store: "ReflectionStore | None",
        mood_store: "MoodStore | None",
        drive_store: "DriveState | None",
        impulse_store: "ImpulseStore | None",
        cfg: "Config",
        allowlist,
    ):
        self._memory = memory
        self._registry = registry
        self._provider = chat_provider
        self._curiosity = curiosity_store
        self._reflection = reflection_store
        self._mood = mood_store
        self._drives = drive_store
        self._impulses = impulse_store
        self._cfg = cfg
        self._allowlist = list(allowlist)
        # Pick the framing once: if the outward `reach_out` verb is in the
        # registry (proactive enabled), offer it; otherwise inward-only.
        self._instruction = (
            _SELF_TURN_INSTRUCTION_OUTWARD
            if "reach_out" in registry.names()
            else _SELF_TURN_INSTRUCTION
        )
        # Per-peer last-self-turn time, in memory (resets on restart — acceptable
        # for this slice; durable bookkeeping is a follow-up).
        self._last: dict[str, datetime] = {}

    async def run(self) -> None:
        """Long-lived heartbeat. Cancelled on shutdown (the task is owned by
        ``main.run``); a per-peer fault is swallowed so one bad peer or a flaky
        model call can never kill the loop."""
        log.info(
            "self-turn loop started interval=%ss min_gap=%ss tools=%d",
            self._cfg.self_turn_interval_s,
            self._cfg.self_turn_min_gap_s,
            len(self._registry),
        )
        try:
            while True:
                await asyncio.sleep(self._cfg.self_turn_interval_s)
                now = datetime.now(timezone.utc)
                for peer in self._allowlist:
                    await self._maybe_self_turn(peer, now)
        except asyncio.CancelledError:
            log.info("self-turn loop stopped")
            raise

    async def _maybe_self_turn(self, peer: str, now: datetime) -> None:
        """One gated inward turn for one peer. Catches its own exceptions — a
        hiccup must never escape into the run loop."""
        try:
            # Cheap deterministic gates first — most ticks bail here, no LLM.
            if len(self._registry) == 0:
                log.info("self-turn tick peer=%s skip=no-verbs", peer)
                return
            last = self._last.get(peer)
            if (
                last is not None
                and (now - last).total_seconds() < self._cfg.self_turn_min_gap_s
            ):
                log.info("self-turn tick peer=%s skip=min-gap", peer)
                return

            # Gather the inner state — used BOTH for the "anything to chew on?"
            # gate and (reused) for the prompt blocks, so one read serves both.
            impulses = (
                await self._impulses.list_active(peer) if self._impulses else []
            )
            open_cur = (
                await self._curiosity.open_curiosities(
                    peer, limit=self._cfg.curiosity_max_open
                )
                if self._curiosity
                else []
            )
            reading = await self._drives.state(peer) if self._drives else {}
            # Which drives were spent recently — a banked drive being actively
            # spent isn't "hoarded", so it shouldn't trip the spend-pressure nudge.
            recently_spent = await self._recently_spent(peer, now)
            # A drive is worth chewing on when it sits off its *personal* reference
            # (the rolling EMA), not off some absolute mark — "low/high for me
            # lately". A drive banked-and-idle (rich-inert) or depleted (bankrupt-
            # inert) relative to your norm is the signal; a banked drive you're
            # already spending is being handled, so it no longer counts on its own.
            pressure = classify_pressure(
                reading, recently_spent, margin=DRIVE_BAND_MARGIN
            )
            off_reference = bool(pressure.banked_idle or pressure.depleted)
            if not (impulses or open_cur or off_reference):
                log.info("self-turn tick peer=%s skip=nothing-to-chew", peer)
                return

            # The two affordability guards (slice b3) ride the framing: a standing
            # note appended to the instruction (the directive) and the named banked
            # drives in the drives block (the colour). Neither is a hard gate.
            noun_by_key = {d.key: d.noun for d in DRIVES}
            banked_nouns = [
                noun_by_key[k] for k in pressure.banked_idle if k in noun_by_key
            ]
            instruction = self._instruction
            addendum = _pressure_addendum(
                bool(pressure.banked_idle), bool(pressure.depleted)
            )
            if addendum:
                instruction = instruction + "\n\n" + addendum

            messages = await self._build_messages(
                peer, impulses, open_cur, reading, instruction, banked_nouns
            )
            # The execute-and-loop runs Jeff's chosen inward verbs (or none). The
            # result is Jeff's own narration — not sent anywhere, just logged.
            result = await run_tool_loop(
                self._provider,
                self._registry,
                messages,
                self._cfg,
                peer,
                drive_store=self._drives,
            )
            # Advance the min-gap whether or not Jeff acted: the turn was spent.
            self._last[peer] = now
            excerpt = " ".join((result or "").split())[:200]
            log.info(
                "self-turn done peer=%s impulses=%d curiosities=%d narration=%r",
                peer,
                len(impulses),
                len(open_cur),
                excerpt,
            )
        except Exception as e:
            # Type-only (never the message — may carry model/peer-shaped text).
            log.error("self-turn failed peer=%s exc=%s", peer, type(e).__name__)

    async def _recently_spent(self, peer: str, now: datetime) -> set[str]:
        """Drive keys spent within the spend-pressure window. Best-effort — a read
        fault yields an empty set (a banked drive then simply reads as idle, which
        is the safe default: it gets the nudge rather than being silently skipped)."""
        if self._drives is None:
            return set()
        try:
            spends = await self._drives.recent_spends(peer, limit=20)
        except Exception as e:
            log.error("self-turn spend read failed peer=%s exc=%s", peer, type(e).__name__)
            return set()
        return {
            s.drive
            for s in spends
            if (now - s.spent_at).total_seconds() <= _SPEND_PRESSURE_WINDOW_S
        }

    async def _build_messages(
        self, peer, impulses, open_cur, reading, instruction, spend_pressure
    ) -> list[dict]:
        """Assemble the self-turn context: base persona + the state blocks (mood,
        drives, impulses, curiosities) + the recent thread + the synthetic nudge.
        ``instruction`` is the (possibly pressure-augmented) self-turn directive
        and ``spend_pressure`` the banked-idle drive nouns to nudge in the drives
        block. Each block fetch is best-effort — a read fault drops that block,
        never the turn (mirrors handle_turn's additive-block discipline)."""
        facts: list[str] = []
        opinions: list[str] = []
        if self._reflection is not None:
            try:
                facts, opinions = await self._reflection.persona(
                    peer, max_chars=self._cfg.persona_max_chars
                )
            except Exception as e:
                log.error("self-turn persona fetch failed exc=%s", type(e).__name__)
        mood_name = ""
        mood_description = ""
        if self._mood is not None:
            try:
                active = await self._mood.active_mood(peer)
                if active is not None:
                    mood_name = active.name
                    mood_description = active.description or ""
            except Exception as e:
                log.error("self-turn mood fetch failed exc=%s", type(e).__name__)
        drives = [
            (d.noun, *(reading.get(d.key) or DriveReading(d.baseline, d.baseline)))
            for d in DRIVES
        ]
        return await build_self_turn_messages(
            self._memory,
            peer,
            instruction,
            recent_turns=self._cfg.recent_turns,
            system_prompt=self._cfg.system_prompt,
            curiosities=[c.text for c in open_cur],
            facts=facts,
            opinions=opinions,
            mood_name=mood_name,
            mood_description=mood_description,
            drives=drives,
            drives_max_chars=self._cfg.drives_max_chars,
            drives_spend_pressure=spend_pressure,
            impulses=[(i.name, i.description) for i in impulses],
            impulses_max_chars=self._cfg.impulses_max_chars,
        )
