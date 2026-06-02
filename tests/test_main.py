"""Stub event-loop test: handle_turn drives memory + ollama + handle correctly."""

from __future__ import annotations

import ensemble
import pytest

from jeff.config import Config
from jeff.dispatch import DispatchPolicy, TurnDispatcher
from jeff.main import _drain_events, handle_turn
from jeff.memory import MAX_CONTENT_BYTES, Memory


class FakeHandle:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []
        self.address = "EnodeD"
        self.onion = "fake.onion"

    async def send_message(self, to_addr: str, text: str) -> None:
        self.sent.append((to_addr, text))


class FakeOllama:
    def __init__(self, reply: str = "ack"):
        self.reply = reply
        self.chat_calls: list[list[dict]] = []
        self.embed_calls: list[str] = []

    async def chat(self, messages, *, model: str) -> str:
        self.chat_calls.append(list(messages))
        return self.reply

    async def embed(self, text: str, *, model: str) -> list[float]:
        self.embed_calls.append(text)
        return [0.0, 0.0, 0.0, 0.0]


class FakeMemory:
    def __init__(self):
        self.remembered: list[tuple[str, str, str]] = []

    async def remember(self, peer, role, content):
        self.remembered.append((peer, role, content))
        return len(self.remembered)

    async def recall(self, peer, query, k=5):
        return []

    async def recent(self, peer, n=10):
        return []


def _cfg() -> Config:
    return Config.from_env(
        {
            "JEFF_DB_URL": "postgresql://unused",
            "JEFF_ALLOWLIST": "EpeerD",
        }
    )


@pytest.mark.asyncio
async def test_handle_turn_writes_user_calls_chat_sends_writes_assistant():
    handle = FakeHandle()
    ollama = FakeOllama(reply="hello peer")
    memory = FakeMemory()
    cfg = _cfg()

    await handle_turn(handle, memory, ollama, cfg, "EpeerD", "ping")

    assert memory.remembered == [
        ("EpeerD", "user", "ping"),
        ("EpeerD", "assistant", "hello peer"),
    ]
    assert handle.sent == [("EpeerD", "hello peer")]
    assert len(ollama.chat_calls) == 1
    history = ollama.chat_calls[0]
    assert history[0]["role"] == "system"
    # W3 #dc9acd3c: user turn is wrapped in <peer_message> delimiters so
    # the system prompt has a syntactic target for its untrusted-data rule.
    assert history[-1] == {
        "role": "user",
        "content": "<peer_message>ping</peer_message>",
    }


class _FakeEventsHandle:
    """Minimal ServiceHandle-like object for driving `_drain_events` in tests."""

    def __init__(self, events: list):
        self._events = events
        self.sent: list[tuple[str, str]] = []

    async def events(self):
        for ev in self._events:
            yield ev

    async def send_message(self, to_addr: str, text: str) -> None:
        self.sent.append((to_addr, text))


@pytest.mark.asyncio
async def test_drain_events_drops_oversize_and_sends_polite_reply():
    """Oversize messages get a rejection reply and never hit the dispatcher."""
    cfg = Config.from_env(
        {
            "JEFF_DB_URL": "postgresql://unused",
            "JEFF_ALLOWLIST": "EpeerD",
            "JEFF_MAX_MESSAGE_BYTES": "100",
        }
    )

    dispatched: list[str] = []

    async def handler(peer: str, text: str) -> None:
        dispatched.append(text)

    dispatcher = TurnDispatcher(handler, DispatchPolicy(max_inflight=10, peer_rate_burst=10))

    handle = _FakeEventsHandle(
        [
            ensemble.ChatMessage(type="chat", from_addr="EpeerD", text="hi", ts=0),
            ensemble.ChatMessage(type="chat", from_addr="EpeerD", text="a" * 200, ts=0),
            ensemble.ChatMessage(type="chat", from_addr="EpeerD", text="bye", ts=0),
        ]
    )

    await _drain_events(handle, dispatcher, cfg)
    await dispatcher.drain()

    # The oversize message never reached the dispatcher.
    assert dispatched == ["hi", "bye"]
    # A polite rejection was sent for the dropped message.
    assert len(handle.sent) == 1
    to_addr, reply = handle.sent[0]
    assert to_addr == "EpeerD"
    assert "200" in reply and "100" in reply


@pytest.mark.asyncio
async def test_memory_remember_rejects_oversize_content_without_db():
    """Belt-and-braces guard fires before embed/DB so it's testable without Docker."""

    class NeverCalledEmbedder:
        async def embed(self, text, *, model):
            raise AssertionError("embed should not be called for oversize content")

    mem = Memory(
        pool=None,  # guard fires before pool is touched
        embedder=NeverCalledEmbedder(),
        embed_model="fake",
        embed_dim=4,
    )
    with pytest.raises(ValueError, match="content too large"):
        await mem.remember("Epeer", "user", "a" * (MAX_CONTENT_BYTES + 1))


@pytest.mark.asyncio
async def test_handle_turn_swallows_exceptions():
    handle = FakeHandle()

    class BoomOllama:
        async def chat(self, messages, *, model):
            raise RuntimeError("ollama is down")

        async def embed(self, text, *, model):
            return [0.0] * 4

    memory = FakeMemory()
    cfg = _cfg()
    # Must not raise — the loop relies on per-turn errors being contained.
    await handle_turn(handle, memory, BoomOllama(), cfg, "EpeerD", "ping")
    # User turn still got remembered before the failure.
    assert ("EpeerD", "user", "ping") in memory.remembered
    # No assistant message stored, no send.
    assert handle.sent == []


@pytest.mark.asyncio
async def test_handle_turn_does_not_log_peer_or_exception_message(caplog):
    """Peer-controlled text and exception strings must not reach the log."""
    import logging

    sentinel_msg = "PEER_CONTROLLED_PAYLOAD_DO_NOT_LEAK"
    peer_text = "ping"

    class BoomOllama:
        async def chat(self, messages, *, model):
            raise RuntimeError(sentinel_msg)

        async def embed(self, text, *, model):
            return [0.0] * 4

    handle = FakeHandle()
    memory = FakeMemory()
    cfg = _cfg()

    with caplog.at_level(logging.DEBUG, logger="jeff"):
        await handle_turn(handle, memory, BoomOllama(), cfg, "EpeerD", peer_text)

    records = caplog.records
    assert records, "expected at least one log record"
    for r in records:
        rendered = r.getMessage()
        # Exception message (Ollama-shaped string) must not be in the log.
        assert sentinel_msg not in rendered, f"exception text leaked: {rendered!r}"
        # No exc_info / traceback attached either.
        assert r.exc_info is None, "log.exception was used — should be log.error without traceback"
    # The structured fields we DO want should be present.
    joined = " ".join(r.getMessage() for r in records)
    assert "EpeerD" in joined
    assert "RuntimeError" in joined
