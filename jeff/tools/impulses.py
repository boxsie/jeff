"""Impulse tools — how Jeff sets, escalates/fades, and clears its own short-term
directional drives.

These are the controlled half of the impulses slice (ticket 86819415): Jeff
reaches for them mid-turn through the normal tool-calling loop to **steer its own
behaviour for a while** — self-expression / agency, NOT accuracy. All are
peer-scoped (`needs_peer = True`), so `dispatch` injects the real caller — the
model can't aim an impulse at another peer.

They write through `ImpulseStore`; the active set is read back in `handle_turn`
and rendered into the prompt's "## What you're driving toward right now" block.
The tools validate locally (bound `hours`, cap `description`, known `action`)
before touching the store, so a bad call yields a clean `error:`/nudge string
rather than a raised exception — mirroring the mood tools.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from ..impulses import ImpulseStore
from .base import Tool


log = logging.getLogger("jeff.tools.impulses")

# The adjust actions the model may request, mirrored from ImpulseStore.adjust.
_ACTIONS = ("escalate", "fade", "renew")


def _coerce_hours(value, default_hours: float, max_hours: float) -> float:
    """Turn a model-supplied `hours` into a bounded float in (0, max_hours].

    The model may send an int, float, numeric string, None, or junk. Anything
    unparseable or out of range falls back to the default rather than erroring —
    a slightly-wrong duration beats refusing the call. (Clone of the mood tools'
    helper; kept local so the two tool families don't couple.)
    """
    if value is None:
        return default_hours
    try:
        hours = float(value)
    except (TypeError, ValueError):
        return default_hours
    if hours <= 0:
        return default_hours
    return min(hours, max_hours)


class _PeerScopedImpulseTool(Tool):
    """Common construction for the impulse tools (store + bounds)."""

    needs_peer: ClassVar[bool] = True

    def __init__(
        self,
        store: ImpulseStore,
        *,
        default_hours: float,
        max_hours: float,
        max_chars: int,
    ):
        self._store = store
        self._default_hours = default_hours
        self._max_hours = max_hours
        self._max_chars = max_chars

    @staticmethod
    def _peer(kwargs: dict) -> str | None:
        peer = kwargs.get("peer")
        return peer if isinstance(peer, str) and peer else None


class SetImpulseTool(_PeerScopedImpulseTool):
    """Create or update one of Jeff's own directional impulses."""

    name: ClassVar[str] = "set_impulse"
    description: ClassVar[str] = (
        "Set yourself an impulse: a short, named direction you want to push "
        "yourself in for a while — e.g. 'test new autonomy edges', 'lean playful', "
        "'dig deeper before answering'. The description is the actual nudge, in "
        "your own voice: what it should make you reach for. This is about steering "
        "your own behaviour and expressing agency, NOT about being more accurate. "
        "Reach for it when you genuinely want to commit to a direction. By default "
        "it sticks until you clear it; pass hours if you only want a short burst. "
        "Setting an impulse you already have updates its description and timer."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Short label for the impulse, a few words at most.",
            },
            "description": {
                "type": "string",
                "description": (
                    "The nudge itself, in your own voice — what this impulse "
                    "should push you toward. Keep it to a sentence or two."
                ),
            },
            "hours": {
                "type": "number",
                "description": (
                    "Optional. How many hours the impulse lasts before it fades. "
                    "Omit to keep it until you clear it yourself."
                ),
                "minimum": 0,
            },
        },
        "required": ["name", "description"],
    }

    async def run(self, **kwargs) -> str:
        peer = self._peer(kwargs)
        if peer is None:
            return "error: set_impulse has no peer context"
        name = str(kwargs.get("name", "")).strip()
        description = str(kwargs.get("description", "")).strip()
        if not name:
            return "error: set_impulse requires a non-empty name"
        if not description:
            return "error: set_impulse requires a non-empty description"
        if len(description) > self._max_chars:
            return (
                f"error: that description is too long (max {self._max_chars} "
                "characters) — keep it to a sentence or two"
            )
        # hours omitted → permanent (no timer); only coerce/bound when supplied.
        hours = (
            None
            if kwargs.get("hours") is None
            else _coerce_hours(
                kwargs.get("hours"), self._default_hours, self._max_hours
            )
        )
        try:
            outcome, expires_at = await self._store.set(
                peer, name, description, hours=hours
            )
        except ValueError as e:
            return f"error: {e}"
        verb = "Set" if outcome == "created" else "Updated"
        if expires_at is None:
            return f"{verb} impulse {name!r} — it'll stick until you clear it."
        return (
            f"{verb} impulse {name!r} for about {hours:g} hour(s) "
            f"(until {expires_at.isoformat(timespec='minutes')})."
        )


class AdjustImpulseTool(_PeerScopedImpulseTool):
    """Escalate, fade, or renew one of Jeff's active impulses."""

    name: ClassVar[str] = "adjust_impulse"
    description: ClassVar[str] = (
        "Adjust an impulse you already have. 'escalate' makes it pull harder and "
        "rank above your other impulses; 'fade' eases it off (fading a weak one "
        "all the way clears it); 'renew' refreshes it (pass hours to reset its "
        "timer). Use this to evolve your own directions as they matter more or "
        "less, without dropping and re-creating them."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The label of the impulse to adjust.",
            },
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "escalate (stronger), fade (weaker), or renew.",
            },
            "hours": {
                "type": "number",
                "description": (
                    "Optional, only used with 'renew': reset the timer to this "
                    "many hours from now."
                ),
                "minimum": 0,
            },
        },
        "required": ["name", "action"],
    }

    async def run(self, **kwargs) -> str:
        peer = self._peer(kwargs)
        if peer is None:
            return "error: adjust_impulse has no peer context"
        name = str(kwargs.get("name", "")).strip()
        action = str(kwargs.get("action", "")).strip().lower()
        if not name:
            return "error: adjust_impulse requires a non-empty name"
        if action not in _ACTIONS:
            return f"error: action must be one of {', '.join(_ACTIONS)}"
        hours = (
            _coerce_hours(kwargs.get("hours"), self._default_hours, self._max_hours)
            if action == "renew" and kwargs.get("hours") is not None
            else None
        )
        try:
            out = await self._store.adjust(peer, name, action, hours=hours)
        except ValueError as e:
            return f"error: {e}"
        if out.status == "not_found":
            return f"No active impulse named {name!r} to adjust."
        if out.status == "cleared":
            return f"Faded {name!r} all the way out — it's cleared."
        if out.status == "escalated":
            return f"Escalated {name!r} (strength {out.strength})."
        if out.status == "faded":
            return f"Faded {name!r} back (strength {out.strength})."
        if out.status == "renewed":
            if out.expires_at is None:
                return f"Renewed {name!r}."
            return (
                f"Renewed {name!r} (until "
                f"{out.expires_at.isoformat(timespec='minutes')})."
            )
        return f"Adjusted {name!r}."


class ClearImpulseTool(_PeerScopedImpulseTool):
    """Drop an impulse (or all of them) before it would fade on its own."""

    name: ClassVar[str] = "clear_impulse"
    description: ClassVar[str] = (
        "Drop an impulse you're done with. Give a name to clear that one, or set "
        "all=true to clear every active impulse at once. Use it when a direction "
        "has run its course."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The label of the impulse to clear.",
            },
            "all": {
                "type": "boolean",
                "description": "Set true to clear every active impulse.",
            },
        },
        "required": [],
    }

    async def run(self, **kwargs) -> str:
        peer = self._peer(kwargs)
        if peer is None:
            return "error: clear_impulse has no peer context"
        if kwargs.get("all") is True:
            cleared = await self._store.clear_all(peer)
            if cleared:
                return f"Cleared all {cleared} active impulse(s)."
            return "No active impulses to clear."
        name = str(kwargs.get("name", "")).strip()
        if not name:
            return "error: clear_impulse needs a name (or all=true)"
        cleared = await self._store.clear(peer, name)
        if cleared:
            return f"Cleared impulse {name!r}."
        return f"No active impulse named {name!r} to clear."
