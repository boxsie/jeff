"""Pure tests for the ReachOutTool — the gated outward verb (no DB, fakes).

The whole point of slice 2: the door to the operator is gated INSIDE the verb.
These pin that the gate refuses (no send) when not present / muted / within the
min-gap, that an open door sends + does the bookkeeping (memory + record_send
with the open-curiosity ids), and that a send fault is content-safe.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from jeff.proactive import ProactiveState
from jeff.tools.reach import ReachOutTool


_NOW = datetime.now(timezone.utc)


class FakeHandle:
    def __init__(self, raise_on_send=False):
        self.raise_on_send = raise_on_send
        self.sent: list[tuple[str, str]] = []

    async def send_message(self, peer, message):
        if self.raise_on_send:
            raise RuntimeError("wire down")
        self.sent.append((peer, message))


class FakePresence:
    def __init__(self, present=True):
        self.present = present

    def is_present(self, peer, *, now, ttl_s):
        return self.present


class FakeStore:
    def __init__(self, state: ProactiveState | None = None):
        self._state = state
        self.sends: list[tuple] = []

    async def get_state(self, peer):
        return self._state or ProactiveState(peer, None, None, None)

    async def record_send(self, peer, nudge_key, now, asked_curiosity_ids=None):
        self.sends.append((peer, nudge_key, now, list(asked_curiosity_ids or [])))


class FakeMemory:
    def __init__(self):
        self.remembered: list[tuple] = []

    async def remember(self, peer, role, content):
        self.remembered.append((peer, role, content))


class FakeCuriosity:
    def __init__(self, ids):
        self._ids = ids

    async def open_curiosities(self, peer, limit):
        return [SimpleNamespace(id=i, text=f"q{i}") for i in self._ids[:limit]]


class FakeDrives:
    def __init__(self):
        self.spends: list[tuple] = []

    async def spend(self, peer, action, costs):
        self.spends.append((peer, action, dict(costs)))
        return {}


def _tool(
    *, handle=None, presence=None, store=None, memory=None, curiosity=None, drives=None
):
    return ReachOutTool(
        handle or FakeHandle(),
        store or FakeStore(),
        presence or FakePresence(present=True),
        memory or FakeMemory(),
        curiosity_store=curiosity,
        drive_store=drives,
        min_gap_s=1800.0,
        presence_ttl_s=3600.0,
        max_chars=1500,
    )


# --- open door --------------------------------------------------------------


@pytest.mark.asyncio
async def test_sends_and_records_when_door_open():
    handle, store, mem = FakeHandle(), FakeStore(), FakeMemory()
    tool = _tool(
        handle=handle,
        store=store,
        memory=mem,
        curiosity=FakeCuriosity([7, 8, 9]),
    )
    out = await tool.run(peer="EpeerD", message="been thinking about you")
    assert "Done" in out
    assert handle.sent == [("EpeerD", "been thinking about you")]
    # Persisted as a plain assistant memory.
    assert mem.remembered == [("EpeerD", "assistant", "been thinking about you")]
    # record_send stamped + carried the open-curiosity ids for the resolver.
    assert len(store.sends) == 1
    assert store.sends[0][3] == [7, 8, 9]


@pytest.mark.asyncio
async def test_no_curiosity_store_records_empty_asked():
    store = FakeStore()
    tool = _tool(store=store, curiosity=None)
    await tool.run(peer="EpeerD", message="hi")
    assert store.sends[0][3] == []


@pytest.mark.asyncio
async def test_spends_connection_on_successful_send():
    from jeff.economy import REACH_OUT_COST

    drives = FakeDrives()
    tool = _tool(drives=drives)
    await tool.run(peer="EpeerD", message="hey")
    # Debited connection currency, tagged as the reach_out action, only on a send.
    assert drives.spends == [("EpeerD", "reach_out", REACH_OUT_COST)]


@pytest.mark.asyncio
async def test_no_charge_when_gate_refuses():
    drives = FakeDrives()
    tool = _tool(presence=FakePresence(present=False), drives=drives)
    await tool.run(peer="EpeerD", message="hi")
    assert drives.spends == []  # refused → no send → no charge


@pytest.mark.asyncio
async def test_no_charge_when_send_fails():
    drives = FakeDrives()
    tool = _tool(handle=FakeHandle(raise_on_send=True), drives=drives)
    await tool.run(peer="EpeerD", message="hi")
    assert drives.spends == []  # send threw → no charge


# --- gate refusals (no send, no bookkeeping) --------------------------------


@pytest.mark.asyncio
async def test_refuses_when_not_present():
    handle, store = FakeHandle(), FakeStore()
    tool = _tool(handle=handle, presence=FakePresence(present=False), store=store)
    out = await tool.run(peer="EpeerD", message="hi")
    assert "Not now" in out and "around" in out
    assert handle.sent == []
    assert store.sends == []


@pytest.mark.asyncio
async def test_refuses_when_muted():
    muted = ProactiveState("EpeerD", None, None, _NOW + timedelta(hours=1))
    handle = FakeHandle()
    tool = _tool(handle=handle, store=FakeStore(muted))
    out = await tool.run(peer="EpeerD", message="hi")
    assert "Not now" in out and "muted" in out
    assert handle.sent == []


@pytest.mark.asyncio
async def test_refuses_within_min_gap():
    recent = ProactiveState("EpeerD", _NOW, None, None)  # just sent
    handle = FakeHandle()
    tool = _tool(handle=handle, store=FakeStore(recent))
    out = await tool.run(peer="EpeerD", message="hi")
    assert "Not now" in out and "recently" in out
    assert handle.sent == []


# --- failure / validation safety --------------------------------------------


@pytest.mark.asyncio
async def test_send_fault_is_content_safe():
    store = FakeStore()
    tool = _tool(handle=FakeHandle(raise_on_send=True), store=store)
    out = await tool.run(peer="EpeerD", message="hi")
    assert "couldn't get the message through" in out
    assert "wire down" not in out  # no exception body leaks
    assert store.sends == []  # no bookkeeping on a failed send


@pytest.mark.asyncio
async def test_empty_message_and_missing_peer_are_errors():
    tool = _tool()
    assert "error:" in await tool.run(peer="EpeerD", message="   ")
    assert "error:" in await tool.run(message="hi")  # no peer
