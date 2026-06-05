"""Mood tools — how Jeff sets, defines, and clears its own affective state.

These are the controlled half of the mood drive (jeff ticket fc01162f): Jeff
reaches for them mid-turn through the normal tool-calling loop, which is the
literal answer to "I'd rather decide, not have it rolled like a dice." All three
are peer-scoped (`needs_peer = True`), so `dispatch` injects the real caller —
the model can't aim them at another peer.

They write through `MoodStore`; the active mood is read back in `handle_turn` and
rendered into the prompt's "## How you're feeling right now" block. The tools
validate locally (bound `hours`, cap `description`) before touching the store, so
a bad call yields a clean `error:`/nudge string rather than a raised exception.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from ..mood import MoodStore
from .base import Tool


log = logging.getLogger("jeff.tools.mood")


def _coerce_hours(value, default_hours: float, max_hours: float) -> float:
    """Turn a model-supplied `hours` into a bounded float in (0, max_hours].

    The model may send an int, float, numeric string, None, or junk. Anything
    unparseable or out of range falls back to the default rather than erroring —
    a slightly-wrong duration is far better than refusing to set a mood at all.
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


class _PeerScopedMoodTool(Tool):
    """Common construction for the three mood tools (store + bounds)."""

    needs_peer: ClassVar[bool] = True

    def __init__(
        self,
        store: MoodStore,
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


class SetMoodTool(_PeerScopedMoodTool):
    """Set Jeff's current mood, replacing any active one."""

    name: ClassVar[str] = "set_mood"
    description: ClassVar[str] = (
        "Set how you're feeling right now to a short-lived mood that will colour "
        "how you talk until it fades. Use it when you genuinely feel a shift you "
        "want to lean into — not to be more accurate, but to express yourself. "
        "Replaces any current mood. If you haven't defined what this mood means "
        "to you yet, define it with define_mood."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Short name of the mood, e.g. a single word.",
            },
            "hours": {
                "type": "number",
                "description": (
                    "How many hours the mood should last before it fades. "
                    "Optional; a sensible default is used if omitted."
                ),
                "minimum": 0,
            },
        },
        "required": ["name"],
    }

    async def run(self, **kwargs) -> str:
        peer = self._peer(kwargs)
        if peer is None:
            return "error: set_mood has no peer context"
        name = str(kwargs.get("name", "")).strip()
        if not name:
            return "error: set_mood requires a non-empty name"
        hours = _coerce_hours(
            kwargs.get("hours"), self._default_hours, self._max_hours
        )
        try:
            expires_at = await self._store.set_mood(peer, name, hours=hours)
            defined = await self._store.get_definition(peer, name) is not None
        except ValueError as e:
            return f"error: {e}"
        msg = (
            f"Mood set: you're feeling {name} for about {hours:g} hour(s) "
            f"(until {expires_at.isoformat(timespec='minutes')})."
        )
        if not defined:
            msg += (
                f" You haven't defined what {name!r} means to you yet — use "
                "define_mood so it colours how you talk."
            )
        return msg


class DefineMoodTool(_PeerScopedMoodTool):
    """Create or edit Jeff's own definition of a mood."""

    name: ClassVar[str] = "define_mood"
    description: ClassVar[str] = (
        "Record, in your own words, what a mood means to you — how it makes you "
        "talk and act. Creates the mood if it's new, or edits it if it already "
        "exists. This is how you build up your own palette of moods over time; "
        "the definition is what colours your tone when the mood is active."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Short name of the mood, e.g. a single word.",
            },
            "description": {
                "type": "string",
                "description": (
                    "What this mood means to you, in your own voice — the tone "
                    "and behaviour it brings out. Keep it to a sentence or two."
                ),
            },
        },
        "required": ["name", "description"],
    }

    async def run(self, **kwargs) -> str:
        peer = self._peer(kwargs)
        if peer is None:
            return "error: define_mood has no peer context"
        name = str(kwargs.get("name", "")).strip()
        description = str(kwargs.get("description", "")).strip()
        if not name:
            return "error: define_mood requires a non-empty name"
        if not description:
            return "error: define_mood requires a non-empty description"
        if len(description) > self._max_chars:
            return (
                f"error: that definition is too long (max {self._max_chars} "
                "characters) — keep it to a sentence or two"
            )
        try:
            outcome = await self._store.define(peer, name, description)
        except ValueError as e:
            return f"error: {e}"
        if outcome == "created":
            return f"Defined a new mood {name!r}."
        if outcome == "updated":
            return f"Updated your definition of {name!r}."
        return "error: define_mood needs a name and a description"


class ClearMoodTool(_PeerScopedMoodTool):
    """Drop back to a neutral mood before the active one would fade."""

    name: ClassVar[str] = "clear_mood"
    description: ClassVar[str] = (
        "Drop your current mood and go back to feeling neutral, before it would "
        "fade on its own. Use it when the mood has passed."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def run(self, **kwargs) -> str:
        peer = self._peer(kwargs)
        if peer is None:
            return "error: clear_mood has no peer context"
        ended = await self._store.clear_mood(peer)
        if ended:
            return "Mood cleared — back to feeling neutral."
        return "No active mood to clear — you're already neutral."
