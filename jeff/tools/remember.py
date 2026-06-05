"""Remember tool — Jeff deliberately pins something to long-term memory.

The write half of explicit memory (jeff ticket 9a24de21): Jeff reaches for this
mid-turn through the normal tool-calling loop when it decides something is worth
keeping, rather than relying on the automatic episodic store or the fire-and-
forget reflection pass. It's peer-scoped (`needs_peer = True`), so `dispatch`
injects the real caller. The operator's `/remember` command writes to the same
`PinnedMemoryStore`, tagged with a different source.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from ..pinned import PinnedMemoryStore
from .base import Tool


log = logging.getLogger("jeff.tools.remember")


class RememberTool(Tool):
    """Pin a single note to Jeff's long-term memory."""

    needs_peer: ClassVar[bool] = True

    name: ClassVar[str] = "remember"
    description: ClassVar[str] = (
        "Deliberately keep a single fact or note in your long-term memory so you "
        "always have it later — something you decide is worth holding onto about "
        "them, your relationship, or how they like things. Use it when something "
        "feels important to remember, not for ordinary chit-chat (the conversation "
        "is already saved automatically). Pin one concise thing per call."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The single fact or note to remember, kept concise.",
            },
        },
        "required": ["text"],
    }

    def __init__(self, store: PinnedMemoryStore, *, max_chars: int):
        self._store = store
        self._max_chars = max_chars

    async def run(self, **kwargs) -> str:
        peer = kwargs.get("peer")
        if not isinstance(peer, str) or not peer:
            return "error: remember has no peer context"
        text = str(kwargs.get("text", "")).strip()
        if not text:
            return "error: remember requires a non-empty note"
        if len(text) > self._max_chars:
            return (
                f"error: that note is too long (max {self._max_chars} characters) "
                "— keep each pinned note concise"
            )
        try:
            pid = await self._store.add(peer, text, source="jeff")
        except ValueError as e:
            return f"error: {e}"
        if pid is None:
            return "I already had that pinned — nothing to add."
        return "Pinned that to memory — I'll keep it in mind."
