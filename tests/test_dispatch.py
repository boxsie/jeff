"""Tests for jeff.dispatch.TurnDispatcher.

Covers the three independent caps:
  1. per-peer concurrency (sem=N → never more than N in flight for one peer)
  2. global MAX_INFLIGHT (drop excess, do not queue)
  3. per-peer token-bucket rate limit (drop over-rate, refill over time)

Also covers idle-peer pruning so the dict can't grow unboundedly under a
hostile flood of distinct peer addresses.
"""

from __future__ import annotations

import asyncio

import pytest

from jeff.dispatch import DispatchPolicy, TurnDispatcher


@pytest.mark.asyncio
async def test_per_peer_serialises_when_concurrency_is_one():
    in_flight = 0
    peak = 0
    enter = asyncio.Event()
    release = asyncio.Event()

    async def handler(peer: str, text: str) -> None:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        enter.set()
        await release.wait()
        in_flight -= 1

    d = TurnDispatcher(
        handler,
        DispatchPolicy(
            max_inflight=100,
            per_peer_concurrency=1,
            peer_rate_per_minute=100000,
            peer_rate_burst=100,
        ),
    )

    accepted = 0
    for _ in range(50):
        if await d.submit("Epeer", "hi"):
            accepted += 1
    assert accepted == 50, "all submissions should be accepted under the rate cap"

    # Let the first handler enter, then release the gate so they drain serially.
    await enter.wait()
    assert peak == 1, "per_peer_concurrency=1 must serialise turns"
    release.set()
    await d.drain()
    assert peak == 1


@pytest.mark.asyncio
async def test_global_max_inflight_drops_excess():
    """If a flood comes in across many peers, the dispatcher drops over the cap."""

    release = asyncio.Event()
    started = asyncio.Event()
    started_count = 0

    async def handler(peer: str, text: str) -> None:
        nonlocal started_count
        started_count += 1
        if started_count == 1:
            started.set()
        await release.wait()

    d = TurnDispatcher(
        handler,
        DispatchPolicy(
            max_inflight=5,
            per_peer_concurrency=1,
            peer_rate_per_minute=100000,
            peer_rate_burst=100,
        ),
    )

    accepted = 0
    rejected = 0
    # 100 distinct peers — each only sends one event, so per-peer caps are
    # irrelevant. The dispatcher's only barrier is max_inflight.
    for i in range(100):
        ok = await d.submit(f"Epeer{i}", "hi")
        if ok:
            accepted += 1
        else:
            rejected += 1

    assert accepted == 5
    assert rejected == 95
    assert d.inflight == 5

    release.set()
    await d.drain()


@pytest.mark.asyncio
async def test_per_peer_rate_limit_drops_over_burst():
    """A single peer cannot fan out more than burst before any refill."""

    async def handler(peer: str, text: str) -> None:
        # Instant return — we just want to count submission accept/reject.
        return None

    fake_now = [0.0]

    def clock() -> float:
        return fake_now[0]

    d = TurnDispatcher(
        handler,
        DispatchPolicy(
            max_inflight=1000,
            per_peer_concurrency=10,
            peer_rate_per_minute=6,
            peer_rate_burst=3,
        ),
        clock=clock,
    )

    # Burst of 3 should pass, 4th onward dropped until a token refills.
    results = [await d.submit("Epeer", "hi") for _ in range(10)]
    assert results[:3] == [True, True, True]
    assert all(r is False for r in results[3:])

    # Advance the clock by 10 s — at 6/min that's exactly 1 token refilled.
    await d.drain()
    fake_now[0] += 10.0
    assert await d.submit("Epeer", "hi") is True
    assert await d.submit("Epeer", "hi") is False


@pytest.mark.asyncio
async def test_idle_peer_state_is_pruned():
    """Old peer entries are dropped so the dict can't grow unboundedly."""

    async def handler(peer: str, text: str) -> None:
        return None

    fake_now = [0.0]

    def clock() -> float:
        return fake_now[0]

    d = TurnDispatcher(
        handler,
        DispatchPolicy(
            max_inflight=1000,
            per_peer_concurrency=1,
            peer_rate_per_minute=100000,
            peer_rate_burst=10,
            peer_idle_timeout_s=60.0,
        ),
        clock=clock,
    )

    for i in range(20):
        assert await d.submit(f"Epeer{i}", "hi") is True
    await d.drain()
    assert d.peers_tracked == 20

    # Jump well past the idle timeout, then submit from a fresh peer — that
    # triggers the lazy prune.
    fake_now[0] += 120.0
    assert await d.submit("EpeerFresh", "hi") is True
    await d.drain()
    # The 20 stale peers should have been evicted; only the fresh one remains.
    assert d.peers_tracked == 1


@pytest.mark.asyncio
async def test_handler_exception_does_not_kill_dispatcher():
    """A blown handler must not poison the in-flight set or stop future submits."""

    calls = 0

    async def handler(peer: str, text: str) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    d = TurnDispatcher(
        handler,
        DispatchPolicy(max_inflight=10, per_peer_concurrency=1, peer_rate_burst=10),
    )

    for _ in range(3):
        assert await d.submit("Epeer", "hi") is True
    await d.drain()
    assert calls == 3
    assert d.inflight == 0
