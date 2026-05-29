"""Per-peer concurrency + global-fan-out bounds for Jeff's turn handler.

Background
----------
The Ensemble daemon hands Jeff a 256-deep backpressure queue. Before this
module, every inbound `ChatMessage` was dispatched with a bare
`asyncio.create_task(handle_turn(...))`. A single abusive (or compromised)
allowlisted peer could fan out enough concurrent Ollama / Postgres calls to
exhaust the connection pool (max_size=4) and the Ollama 120s timeout — tasks
pile up, memory grows, daemon eventually OOM-kills.

W2 #02736b52 closes that by inserting a `TurnDispatcher` between the events
iterator and the work coroutine. Three independent caps:

1. **Per-peer serialisation (`asyncio.Semaphore(per_peer_concurrency)`)** —
   a single peer cannot have more than `per_peer_concurrency` turns in
   flight at once. Defaults to 1 (chat-style strict serialisation).

2. **Per-peer token bucket** — limits the *rate* of accepted turns from one
   peer regardless of in-flight count. A peer that hammers the daemon while
   one turn waits on Ollama still gets back-pressured at the dispatcher.

3. **Global in-flight cap (`MAX_INFLIGHT`)** — total live tasks across all
   peers. Excess events are *dropped with a warning log* rather than
   queued, so backpressure is visible instead of latent.

Design notes
------------
- The dispatcher is intentionally peer-keyed by the raw `from_addr` string.
  The allowlist check happens *before* `submit()` — see `main._drain_events`.
- `peer_sems` and `peer_buckets` are bounded by lazy pruning inside
  `submit()`: any peer idle longer than `peer_idle_timeout_s` is evicted.
  This is O(N) over the dict on each submit but N is bounded by the
  allowlist size in practice.
- We deliberately avoid a separate sweeper task — one less moving part to
  cancel during shutdown, and the prune cost is negligible at the rates we
  expect.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable


log = logging.getLogger("jeff.dispatch")


@dataclass
class _PeerState:
    """Tracks one peer's semaphore + token bucket + last-seen time."""

    sem: asyncio.Semaphore
    tokens: float
    last_refill: float
    last_seen: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class DispatchPolicy:
    """Knobs that bound per-peer and global concurrency.

    All defaults are conservative — operators can loosen via env vars (see
    Config) without re-deploying code.
    """

    max_inflight: int = 32
    per_peer_concurrency: int = 1
    peer_rate_per_minute: float = 6.0
    peer_rate_burst: int = 3
    peer_idle_timeout_s: float = 600.0


class TurnDispatcher:
    """Owns the in-flight task set and per-peer rate/concurrency state.

    Construct once; call `submit(peer, ...)` to fire-and-forget a turn.
    Returns True if accepted, False if dropped (over-rate or over-cap).
    """

    def __init__(
        self,
        handler: Callable[..., Awaitable[None]],
        policy: DispatchPolicy | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._handler = handler
        self._policy = policy or DispatchPolicy()
        self._clock = clock
        self._peers: dict[str, _PeerState] = {}
        self._inflight: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    @property
    def inflight(self) -> int:
        """How many turns are currently executing. Read for tests/metrics."""
        return len(self._inflight)

    @property
    def peers_tracked(self) -> int:
        return len(self._peers)

    async def submit(self, peer: str, *args, **kwargs) -> bool:
        """Try to enqueue a turn. Returns True if accepted, False if dropped.

        Drops are logged at WARNING level with the reason — that's how the
        operator sees a flood without scraping a metrics endpoint.
        """
        now = self._clock()
        p = self._policy

        async with self._lock:
            self._prune_idle_locked(now)

            if len(self._inflight) >= p.max_inflight:
                log.warning(
                    "dropping turn from peer=%s — global inflight cap reached (%d)",
                    peer,
                    p.max_inflight,
                )
                return False

            state = self._peers.get(peer)
            if state is None:
                state = _PeerState(
                    sem=asyncio.Semaphore(p.per_peer_concurrency),
                    tokens=float(p.peer_rate_burst),
                    last_refill=now,
                )
                self._peers[peer] = state

            self._refill_locked(state, now)
            if state.tokens < 1.0:
                log.warning(
                    "dropping turn from peer=%s — rate cap (%.1f/min, burst %d)",
                    peer,
                    p.peer_rate_per_minute,
                    p.peer_rate_burst,
                )
                return False
            state.tokens -= 1.0
            state.last_seen = now

            task = asyncio.create_task(self._run(peer, *args, **kwargs))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)
        return True

    async def _run(self, peer: str, *args, **kwargs) -> None:
        """Acquire the peer semaphore, then execute the handler.

        The semaphore is acquired *outside* the dispatcher lock — otherwise a
        long-running turn would block every other peer's submit(). Holding the
        peer sem here means a flood from one peer queues *inside* the task
        rather than racing through to Ollama.
        """
        state = self._peers.get(peer)
        if state is None:
            return
        async with state.sem:
            try:
                await self._handler(peer, *args, **kwargs)
            except Exception:
                log.exception("turn failed for peer=%s", peer)

    def _refill_locked(self, state: _PeerState, now: float) -> None:
        elapsed = max(0.0, now - state.last_refill)
        added = (self._policy.peer_rate_per_minute / 60.0) * elapsed
        state.tokens = min(float(self._policy.peer_rate_burst), state.tokens + added)
        state.last_refill = now

    def _prune_idle_locked(self, now: float) -> None:
        cutoff = now - self._policy.peer_idle_timeout_s
        stale = [
            peer
            for peer, state in self._peers.items()
            if state.last_seen < cutoff and not state.sem.locked()
        ]
        for peer in stale:
            del self._peers[peer]

    async def drain(self) -> None:
        """Wait for all in-flight turns to finish.

        Called from `run()`'s shutdown path so we don't lose replies that are
        mid-flight when SIGTERM arrives.
        """
        if not self._inflight:
            return
        log.info("waiting for %d in-flight turns", len(self._inflight))
        await asyncio.gather(*self._inflight, return_exceptions=True)
