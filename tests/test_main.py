"""Stub event-loop test: handle_turn drives memory + ollama + handle correctly."""

from __future__ import annotations

import ensemble
import pytest

from jeff.chat_types import ChatResult, ToolCall
from jeff.config import Config
from jeff.dispatch import DispatchPolicy, TurnDispatcher
from jeff.main import _drain_events, handle_turn
from jeff.memory import MAX_CONTENT_BYTES, Memory
from jeff.tools import ToolRegistry
from jeff.tools.builtins import GetTimeTool


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


class FakeToolProvider:
    """A ChatProvider whose `complete` returns a scripted sequence of results.

    Each call pops the next `ChatResult`; if the script is exhausted it returns
    the last one forever (useful for the cap test). Records every `messages`
    list it was handed so tests can assert the tool round-trip.
    """

    def __init__(self, script: list[ChatResult]):
        self._script = script
        self.calls: list[list[dict]] = []

    async def complete(self, messages, *, model, tools=None):
        self.calls.append(list(messages))
        idx = min(len(self.calls) - 1, len(self._script) - 1)
        return self._script[idx]

    async def chat(self, messages, *, model):  # pragma: no cover - loop uses complete
        res = await self.complete(messages, model=model)
        return res.content or ""


def _tools_cfg(**extra) -> Config:
    env = {"JEFF_DB_URL": "postgresql://unused", "JEFF_ALLOWLIST": "EpeerD"}
    env.update(extra)
    return Config.from_env(env)


@pytest.mark.asyncio
async def test_tool_loop_executes_tool_then_sends_final_answer_once():
    handle = FakeHandle()
    memory = FakeMemory()
    registry = ToolRegistry([GetTimeTool()])
    provider = FakeToolProvider(
        [
            ChatResult(
                content=None,
                tool_calls=(ToolCall(id="c1", name="get_time", arguments="{}"),),
            ),
            ChatResult(content="It is now-ish.", tool_calls=()),
        ]
    )
    cfg = _tools_cfg()

    await handle_turn(handle, memory, provider, cfg, "EpeerD", "what time is it", registry)

    # Exactly one reply — the final answer, not the intermediate tool chatter.
    assert handle.sent == [("EpeerD", "It is now-ish.")]
    # Only the user turn and the final assistant turn are persisted.
    assert memory.remembered == [
        ("EpeerD", "user", "what time is it"),
        ("EpeerD", "assistant", "It is now-ish."),
    ]
    # The second provider call carried the assistant tool-call turn + a tool
    # result message correlated by id.
    second = provider.calls[1]
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in second)
    tool_msgs = [m for m in second if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "c1"
    assert "Current UTC time" in tool_msgs[0]["content"]


@pytest.mark.asyncio
async def test_tool_loop_unknown_tool_feeds_safe_error_and_recovers():
    handle = FakeHandle()
    memory = FakeMemory()
    registry = ToolRegistry([GetTimeTool()])
    provider = FakeToolProvider(
        [
            ChatResult(
                content=None,
                tool_calls=(ToolCall(id="c1", name="no_such_tool", arguments="{}"),),
            ),
            ChatResult(content="Sorry, I couldn't do that.", tool_calls=()),
        ]
    )
    cfg = _tools_cfg()

    await handle_turn(handle, memory, provider, cfg, "EpeerD", "do a thing", registry)

    assert handle.sent == [("EpeerD", "Sorry, I couldn't do that.")]
    tool_msgs = [m for m in provider.calls[1] if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith("error:")
    assert "unknown tool" in tool_msgs[0]["content"]


@pytest.mark.asyncio
async def test_tool_loop_cap_sends_graceful_message_exactly_once():
    handle = FakeHandle()
    memory = FakeMemory()
    registry = ToolRegistry([GetTimeTool()])
    # Always asks for a tool — never converges.
    provider = FakeToolProvider(
        [
            ChatResult(
                content=None,
                tool_calls=(ToolCall(id="c1", name="get_time", arguments="{}"),),
            )
        ]
    )
    cfg = _tools_cfg(JEFF_MAX_TOOL_ITERS="2")

    await handle_turn(handle, memory, provider, cfg, "EpeerD", "loop forever", registry)

    # Provider called exactly the cap number of times, one graceful reply sent.
    assert len(provider.calls) == 2
    assert len(handle.sent) == 1
    to_addr, reply = handle.sent[0]
    assert to_addr == "EpeerD"
    assert "tool-use limit" in reply
    # The graceful message is what gets persisted as the assistant turn.
    assert memory.remembered[-1] == ("EpeerD", "assistant", reply)


@pytest.mark.asyncio
async def test_tools_disabled_uses_single_shot_chat_path():
    """tools_enabled=false → registry path is skipped, byte-identical to before."""
    handle = FakeHandle()
    memory = FakeMemory()
    registry = ToolRegistry([GetTimeTool()])
    ollama = FakeOllama(reply="plain reply")
    cfg = _tools_cfg(JEFF_TOOLS_ENABLED="false")

    await handle_turn(handle, memory, ollama, cfg, "EpeerD", "hi", registry)

    assert handle.sent == [("EpeerD", "plain reply")]
    # The single-shot chat() path ran, not complete().
    assert len(ollama.chat_calls) == 1


@pytest.mark.asyncio
async def test_handle_turn_uses_passed_system_prompt():
    handle = FakeHandle()
    ollama = FakeOllama(reply="ok")
    memory = FakeMemory()
    cfg = _cfg()

    await handle_turn(
        handle, memory, ollama, cfg, "EpeerD", "ping", None, "SYSTEM-PROMPT-OVERRIDE"
    )

    history = ollama.chat_calls[0]
    assert history[0] == {"role": "system", "content": "SYSTEM-PROMPT-OVERRIDE"}


class FakeCommandMemory(FakeMemory):
    """FakeMemory plus the command-path methods, recording cutoff calls."""

    def __init__(self):
        super().__init__()
        self.cutoffs: list[str] = []

    async def set_history_cutoff(self, peer):
        self.cutoffs.append(peer)

    async def count(self, peer):
        return len(self.remembered)

    async def total(self):
        return len(self.remembered)


@pytest.mark.asyncio
async def test_handle_turn_intercepts_command_no_memory_no_llm():
    """A /command is handled in-band: reply sent once, nothing stored, model
    never called."""
    from jeff.commands import build_command_registry

    handle = FakeHandle()
    memory = FakeCommandMemory()
    ollama = FakeOllama(reply="should not be used")
    cfg = _cfg()

    await handle_turn(
        handle, memory, ollama, cfg, "EpeerD", "/new", None, None, build_command_registry()
    )

    # The soft-reset ran, exactly one reply went out, and the model was never
    # consulted — nor was the command stored as a conversational turn.
    assert memory.cutoffs == ["EpeerD"]
    assert len(handle.sent) == 1 and handle.sent[0][0] == "EpeerD"
    assert ollama.chat_calls == []
    assert memory.remembered == []


@pytest.mark.asyncio
async def test_handle_turn_unknown_command_does_not_reach_llm():
    from jeff.commands import build_command_registry

    handle = FakeHandle()
    memory = FakeCommandMemory()
    ollama = FakeOllama(reply="should not be used")
    cfg = _cfg()

    await handle_turn(
        handle, memory, ollama, cfg, "EpeerD", "/bogus", None, None, build_command_registry()
    )

    assert ollama.chat_calls == []
    assert memory.remembered == []
    assert len(handle.sent) == 1
    assert "Unknown command" in handle.sent[0][1]


@pytest.mark.asyncio
async def test_handle_turn_normal_text_with_commands_still_runs_turn():
    """Non-/ text falls through to the normal turn even when commands are wired."""
    from jeff.commands import build_command_registry

    handle = FakeHandle()
    memory = FakeCommandMemory()
    ollama = FakeOllama(reply="hi there")
    cfg = _cfg()

    await handle_turn(
        handle, memory, ollama, cfg, "EpeerD", "hello", None, None, build_command_registry()
    )

    assert handle.sent == [("EpeerD", "hi there")]
    assert len(ollama.chat_calls) == 1
    assert ("EpeerD", "user", "hello") in memory.remembered


@pytest.mark.asyncio
async def test_handle_turn_commands_disabled_treats_slash_as_normal_text():
    from jeff.commands import build_command_registry

    handle = FakeHandle()
    memory = FakeCommandMemory()
    ollama = FakeOllama(reply="llm reply")
    cfg = _tools_cfg(JEFF_COMMANDS_ENABLED="false")

    await handle_turn(
        handle, memory, ollama, cfg, "EpeerD", "/new", None, None, build_command_registry()
    )

    # Commands off → "/new" is just text: it hits the LLM and is remembered.
    assert memory.cutoffs == []
    assert len(ollama.chat_calls) == 1
    assert ("EpeerD", "user", "/new") in memory.remembered


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
