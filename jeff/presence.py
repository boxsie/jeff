"""Lightweight peer-presence tracking from the event stream.

The Ensemble service SDK exposes no presence query and no disconnect event — a
connection is a session, and the daemon delivers a peer's events (a connection
dial-in, a chat message, a command) only while there's a live session. So the
only honest signal a service has that a peer is *reachable right now* is recent
inbound activity. `main`'s event drain marks every inbound event here, and the
proactive loop treats a peer as present if they were seen within a TTL.

This is intentionally an in-memory, best-effort proxy (it resets on restart and
can't observe a silent disconnect) — but it's exactly enough for the rule
"don't reach out into the void". It deliberately does NOT fight the proactive
loop's silence-pressure trigger: the natural reach-out moment is when the
operator *reconnects after a long absence* — their dial-in marks them present
again, and by then the connection drive has decayed low, so a "thinking about
you" nudge can fire the moment they're reachable, not while they're offline.
"""

from __future__ import annotations

from datetime import datetime, timezone


class Presence:
    """In-memory last-seen tracker, keyed by peer address."""

    def __init__(self) -> None:
        self._last_seen: dict[str, datetime] = {}

    def mark(self, peer: str, *, now: datetime | None = None) -> None:
        """Record that `peer` is active as of `now` (defaults to wall-clock UTC)."""
        self._last_seen[peer] = now or datetime.now(timezone.utc)

    def last_seen(self, peer: str) -> datetime | None:
        return self._last_seen.get(peer)

    def is_present(self, peer: str, *, now: datetime, ttl_s: float) -> bool:
        """True if `peer` was seen within `ttl_s` seconds of `now`. A peer never
        seen (or seen longer ago than the TTL) is treated as not reachable."""
        seen = self._last_seen.get(peer)
        if seen is None:
            return False
        return (now - seen).total_seconds() <= ttl_s
