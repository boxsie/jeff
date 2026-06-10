"""Pure tests for the ProactiveLoop gating + decision parsing (no DB, fakes).

The whole design rests on silence being cheap and default: most ticks must bail
on a deterministic gate WITHOUT ever calling the model. These tests pin that.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from jeff.proactive import (
    ProactiveLoop,
    ProactiveState,
    _fingerprint,
    _parse_decision,
)


_NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)


# --- fakes ------------------------------------------------------------------


class FakeProvider:
    def __init__(self, reply: str = '{"send": false}'):
        self.reply = reply
        self.calls: list[list[dict]] = []

    async def chat(self, messages, *, model):
        self.calls.append(messages)
        return self.reply


class FakeDrives:
    def __init__(self, connection: float):
        self._c = connection

    async def levels(self, peer):
        return {"connection": self._c}


class FakeCuriosity:
    def __init__(self, texts: list[str]):
        # ids are 1-based so the reach-out has a concrete id to record per text.
        self._texts = texts

    async def open_curiosities(self, peer, limit):
        return [
            SimpleNamespace(id=i + 1, text=t)
            for i, t in enumerate(self._texts[:limit])
        ]


class FakeStore:
    def __init__(self, state: ProactiveState | None = None):
        self._state = state
        self.sends: list[tuple[str, str, datetime, list[int]]] = []

    async def get_state(self, peer):
        return self._state or ProactiveState(peer, None, None, None)

    async def record_send(self, peer, nudge_key, now, asked_curiosity_ids=None):
        self.sends.append((peer, nudge_key, now, list(asked_curiosity_ids or [])))


class FakePresence:
    def __init__(self, present: bool = True):
        self._present = present

    def is_present(self, peer, *, now, ttl_s):
        return self._present


class FakeHandle:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    async def send_message(self, peer, text):
        self.sent.append((peer, text))


class FakeMemory:
    def __init__(self):
        self.remembered: list[tuple[str, str, str]] = []

    async def remember(self, peer, role, content):
        self.remembered.append((peer, role, content))


def _cfg(**over):
    base = dict(
        proactive_interval_s=300,
        proactive_connection_threshold=0.35,
        proactive_presence_ttl_s=3600,
        proactive_min_gap_s=1800,
        curiosity_max_open=10,
        persona_max_chars=2000,
        drives_max_chars=2000,
        chat_model="test-model",
        system_prompt="You are Jeff.",
    )
    base.update(over)
    return SimpleNamespace(**base)


_UNSET = object()  # so an explicit None (store absent) is distinguishable


def _loop(*, provider=None, drives=_UNSET, curiosity=_UNSET, store=None, presence=None,
          handle=None, memory=None, mood=None, reflection=None, cfg=None):
    return ProactiveLoop(
        handle or FakeHandle(),
        store or FakeStore(),
        presence or FakePresence(present=True),
        memory or FakeMemory(),
        curiosity_store=FakeCuriosity(["q1"]) if curiosity is _UNSET else curiosity,
        reflection_store=reflection,
        mood_store=mood,
        drive_store=FakeDrives(0.2) if drives is _UNSET else drives,
        chat_provider=provider or FakeProvider(),
        cfg=cfg or _cfg(),
        allowlist=["EpeerD"],
    )


# --- parse + fingerprint (pure) ---------------------------------------------


def test_parse_decision_send_true():
    assert _parse_decision('{"send": true, "message": "hi"}') == (True, "hi")


def test_parse_decision_with_fence_and_prose():
    raw = 'Sure:\n```json\n{"send": true, "message": "hey you"}\n```'
    assert _parse_decision(raw) == (True, "hey you")


def test_parse_decision_send_false():
    assert _parse_decision('{"send": false}') == (False, None)


def test_parse_decision_empty_message_is_no_send():
    assert _parse_decision('{"send": true, "message": "   "}') == (False, None)
    assert _parse_decision('{"send": true}') == (False, None)


def test_parse_decision_malformed_defaults_to_silence():
    for raw in ("", "not json", "{nope", "[1,2]", '{"send": "yes"}'):
        assert _parse_decision(raw) == (False, None)


def test_parse_decision_caps_message_length():
    long = "x" * 5000
    send, msg = _parse_decision('{"send": true, "message": "%s"}' % long)
    assert send is True
    assert len(msg) == 1500


def test_fingerprint_stable_and_content_sensitive():
    assert _fingerprint(["a", "b"]) == _fingerprint(["a", "b"])
    assert _fingerprint(["a", "b"]) != _fingerprint(["a", "c"])


# --- gating: most ticks must NOT call the model -----------------------------


@pytest.mark.asyncio
async def test_no_candidates_means_no_model_call():
    prov = FakeProvider()
    handle = FakeHandle()
    loop = _loop(provider=prov, curiosity=FakeCuriosity([]), handle=handle)
    await loop._maybe_reach_out("EpeerD", _NOW)
    assert prov.calls == []
    assert handle.sent == []


@pytest.mark.asyncio
async def test_connection_above_threshold_means_no_model_call():
    prov = FakeProvider()
    loop = _loop(provider=prov, drives=FakeDrives(0.9))  # well-connected → no pull
    await loop._maybe_reach_out("EpeerD", _NOW)
    assert prov.calls == []


@pytest.mark.asyncio
async def test_not_present_skips():
    prov = FakeProvider()
    loop = _loop(provider=prov, presence=FakePresence(present=False))
    await loop._maybe_reach_out("EpeerD", _NOW)
    assert prov.calls == []


@pytest.mark.asyncio
async def test_muted_skips():
    prov = FakeProvider()
    store = FakeStore(ProactiveState("EpeerD", None, None, _NOW + timedelta(hours=1)))
    loop = _loop(provider=prov, store=store)
    await loop._maybe_reach_out("EpeerD", _NOW)
    assert prov.calls == []


@pytest.mark.asyncio
async def test_within_min_gap_skips():
    prov = FakeProvider()
    store = FakeStore(
        ProactiveState("EpeerD", _NOW - timedelta(minutes=5), None, None)
    )  # last send 5 min ago, floor is 30 min
    loop = _loop(provider=prov, store=store)
    await loop._maybe_reach_out("EpeerD", _NOW)
    assert prov.calls == []


@pytest.mark.asyncio
async def test_unchanged_candidate_set_is_deduped():
    prov = FakeProvider()
    key = _fingerprint(["q1"])
    store = FakeStore(ProactiveState("EpeerD", None, key, None))
    loop = _loop(provider=prov, store=store, curiosity=FakeCuriosity(["q1"]))
    await loop._maybe_reach_out("EpeerD", _NOW)
    assert prov.calls == []  # same things stirring, nothing new → no reach-out


@pytest.mark.asyncio
async def test_missing_stores_make_loop_inert():
    prov = FakeProvider()
    loop = _loop(provider=prov, drives=None)
    await loop._maybe_reach_out("EpeerD", _NOW)
    assert prov.calls == []


# --- the gatekeeper path ----------------------------------------------------


@pytest.mark.asyncio
async def test_pressure_and_candidate_consults_gatekeeper_and_sends():
    prov = FakeProvider('{"send": true, "message": "been wondering how X went"}')
    handle = FakeHandle()
    mem = FakeMemory()
    store = FakeStore()
    loop = _loop(provider=prov, handle=handle, memory=mem, store=store)
    await loop._maybe_reach_out("EpeerD", _NOW)
    assert len(prov.calls) == 1  # model consulted only when pressure + candidate
    assert handle.sent == [("EpeerD", "been wondering how X went")]
    assert mem.remembered == [("EpeerD", "assistant", "been wondering how X went")]
    assert store.sends and store.sends[0][0] == "EpeerD"


@pytest.mark.asyncio
async def test_reach_out_records_asked_curiosity_ids():
    # The whole point of ticket 342c7071: a reach-out must persist WHICH open
    # curiosities fuelled it, so the next inbound turn can mark them answered.
    prov = FakeProvider('{"send": true, "message": "how did the bike thing go?"}')
    store = FakeStore()
    loop = _loop(
        provider=prov, store=store, curiosity=FakeCuriosity(["q1", "q2", "q3"])
    )
    await loop._maybe_reach_out("EpeerD", _NOW)
    assert len(store.sends) == 1
    _, _, _, asked = store.sends[0]
    assert asked == [1, 2, 3]  # the FakeCuriosity ids for the surfaced candidates


@pytest.mark.asyncio
async def test_gatekeeper_declines_sends_nothing():
    prov = FakeProvider('{"send": false}')
    handle = FakeHandle()
    loop = _loop(provider=prov, handle=handle)
    await loop._maybe_reach_out("EpeerD", _NOW)
    assert len(prov.calls) == 1
    assert handle.sent == []


@pytest.mark.asyncio
async def test_gatekeeper_junk_stays_silent():
    prov = FakeProvider("I think maybe? sure why not")
    handle = FakeHandle()
    loop = _loop(provider=prov, handle=handle)
    await loop._maybe_reach_out("EpeerD", _NOW)
    assert handle.sent == []  # unparseable → silence default


@pytest.mark.asyncio
async def test_maybe_reach_out_swallows_store_fault():
    class Boom(FakeDrives):
        async def levels(self, peer):
            raise RuntimeError("db down")

    handle = FakeHandle()
    loop = _loop(drives=Boom(0.2), handle=handle)
    # Must not raise — a hiccup can't escape into the run loop.
    await loop._maybe_reach_out("EpeerD", _NOW)
    assert handle.sent == []
