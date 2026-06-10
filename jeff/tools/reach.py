"""reach_out — Jeff's one OUTWARD verb in a self-turn.

This is where the spitball's principle lands: *code gates the door to the
operator; the model runs free in its own head.* The idle self-turn fires and
runs its inward verbs regardless of presence, but reaching out is an intrusion,
so the gate lives HERE, inside the tool — presence (are they reachable right
now?), the proactive min-gap (don't machine-gun), and the operator's mute
window. When the door is shut the tool simply refuses with a content-safe line
the model reads ("not now — they're not around"); the self-turn carries on
inward. The model decides to reach out by *calling this verb*, which is what let
us retire the old `{"send": …}` gatekeeper loop.

On a real send it does the same bookkeeping the retired loop did: persist the
message as a plain `assistant` memory (so it's in the thread next time), and
`ProactiveStore.record_send` with the open-curiosity ids that were on Jeff's mind
— so the curiosity-resolution loop (ticket 342c7071) still closes when the
operator replies. Peer-scoped (`needs_peer = True`); the handle is bound at
construction (only available inside the registered session).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import ClassVar

from .base import Tool


log = logging.getLogger("jeff.tools.reach")

# Hard cap on an unprompted message's length — Jeff's own output, but a ping
# should be SHORT; bound it defensively (matches the retired loop's cap).
REACH_OUT_MAX_CHARS = 1500
# How many of Jeff's open curiosities to record as "asked" on a reach-out — we
# can't know which the message actually voiced, so (mirroring the retired loop)
# we record the ids that were on its mind and let the resolver sort it out.
_MAX_ASKED = 6


class ReachOutTool(Tool):
    """Send the operator an unprompted message — gated on presence + min-gap + mute."""

    needs_peer: ClassVar[bool] = True

    name: ClassVar[str] = "reach_out"
    description: ClassVar[str] = (
        "Send this person an unprompted message right now — they did NOT just "
        "message you; you'd be reaching out on your own. Use it only when you "
        "genuinely have something worth their attention: a real question, a "
        "thought that adds to their day, or you simply miss the back-and-forth. "
        "Keep it rare and warm — most of the time sitting with your own thoughts "
        "is enough. It may decline if they're not around or you reached out very "
        "recently; that's fine, just hold the thought."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The short, in-character message to send them now.",
            },
        },
        "required": ["message"],
    }

    def __init__(
        self,
        handle,
        store,
        presence,
        memory,
        *,
        curiosity_store=None,
        min_gap_s: float,
        presence_ttl_s: float,
        max_chars: int,
    ):
        self._handle = handle
        self._store = store
        self._presence = presence
        self._memory = memory
        self._curiosity = curiosity_store
        self._min_gap_s = min_gap_s
        self._presence_ttl_s = presence_ttl_s
        self._max_chars = max_chars

    async def run(self, **kwargs) -> str:
        peer = kwargs.get("peer")
        if not isinstance(peer, str) or not peer:
            return "error: reach_out has no peer context"
        message = str(kwargs.get("message", "")).strip()
        if not message:
            return "error: reach_out requires a non-empty message"
        message = message[: self._max_chars]

        now = datetime.now(timezone.utc)
        # The gate — the door to the operator. Each refusal is a content-safe line
        # the model reads and can act on (hold the thought, do something inward).
        if not self._presence.is_present(peer, now=now, ttl_s=self._presence_ttl_s):
            log.info("reach_out refused peer=%s reason=not-present", peer)
            return "Not now — they're not around right now, so I'll hold the thought."
        state = await self._store.get_state(peer)
        if state.muted_until is not None and state.muted_until > now:
            log.info("reach_out refused peer=%s reason=muted", peer)
            return "Not now — they've muted me, so I'll keep this to myself for now."
        if (
            state.last_send_at is not None
            and (now - state.last_send_at).total_seconds() < self._min_gap_s
        ):
            log.info("reach_out refused peer=%s reason=min-gap", peer)
            return "Not now — I reached out very recently; too soon to ping them again."

        try:
            await self._handle.send_message(peer, message)
        except Exception as e:
            # Never surface the exception (peer/endpoint-shaped) — generic line.
            log.error("reach_out send failed peer=%s exc=%s", peer, type(e).__name__)
            return "I tried to reach out but couldn't get the message through just now."

        # Bookkeeping mirrors the retired ProactiveLoop: persist as a plain
        # assistant memory + stamp the floor/dedup key + record the open-curiosity
        # ids so a reply can resolve them (ticket 342c7071). Best-effort — the
        # message is already sent, so a bookkeeping fault must not look like a
        # failed reach-out.
        try:
            await self._memory.remember(peer, "assistant", message)
            asked = await self._asked_curiosity_ids(peer)
            nudge_key = hashlib.sha1(message.encode("utf-8")).hexdigest()[:16]
            await self._store.record_send(peer, nudge_key, now, asked)
            log.info("reach_out sent peer=%s asked=%d", peer, len(asked))
        except Exception as e:
            log.error("reach_out bookkeeping failed peer=%s exc=%s", peer, type(e).__name__)
        return "Done — I've reached out to them."

    async def _asked_curiosity_ids(self, peer: str) -> list[int]:
        if self._curiosity is None:
            return []
        try:
            open_cur = await self._curiosity.open_curiosities(peer, limit=_MAX_ASKED)
        except Exception as e:
            log.error("reach_out curiosity fetch failed exc=%s", type(e).__name__)
            return []
        return [c.id for c in open_cur[:_MAX_ASKED]]
