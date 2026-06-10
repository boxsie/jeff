"""Stub event-loop test: handle_turn drives memory + ollama + handle correctly."""

from __future__ import annotations

import ensemble
import pytest

from jeff.chat_types import ChatResult, ToolCall
from jeff.config import Config
from jeff.dispatch import DispatchPolicy, TurnDispatcher
from jeff.main import _TURN_FAILED_MESSAGE, _drain_events, handle_turn
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

    async def recall(self, peer, query, k=5, *, distance_max=0.55):
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


class _FakeCuriosity:
    """A stored open question with just the field handle_turn reads."""

    def __init__(self, text: str):
        self.text = text


class FakeCuriosityStore:
    def __init__(self, open_texts: list[str] | None = None, *, boom: bool = False):
        self._open = [_FakeCuriosity(t) for t in (open_texts or [])]
        self._boom = boom
        self.open_calls: list[tuple[str, int]] = []

    async def open_curiosities(self, peer, *, limit=10):
        self.open_calls.append((peer, limit))
        if self._boom:
            raise RuntimeError("DSN=postgresql://secret store down")
        return self._open[:limit]


class FakeCuriosityDriver:
    def __init__(self):
        self.detect_calls: list[tuple[str, str, str]] = []

    async def maybe_detect(self, peer, user_text, assistant_text):
        self.detect_calls.append((peer, user_text, assistant_text))


@pytest.mark.asyncio
async def test_handle_turn_injects_open_curiosities_and_fires_detection():
    handle = FakeHandle()
    ollama = FakeOllama(reply="sure")
    memory = FakeMemory()
    store = FakeCuriosityStore(["What's your homelab called?"])
    driver = FakeCuriosityDriver()
    cfg = _cfg()

    await handle_turn(
        handle, memory, ollama, cfg, "EpeerD", "hey",
        None, None, store, driver,
    )

    # The open question rode into the system message as a curiosity block.
    system = ollama.chat_calls[0][0]
    assert system["role"] == "system"
    assert "## You're curious about" in system["content"]
    assert "What's your homelab called?" in system["content"]
    # Detection fired with the actual exchange (user text + the reply sent).
    assert driver.detect_calls == [("EpeerD", "hey", "sure")]
    assert store.open_calls == [("EpeerD", cfg.curiosity_max_open)]


@pytest.mark.asyncio
async def test_handle_turn_survives_curiosity_store_read_fault():
    """Curiosity is additive: a store read fault must not break the reply or
    leak DSN/exception text — the turn proceeds with no curiosity block."""
    handle = FakeHandle()
    ollama = FakeOllama(reply="still here")
    memory = FakeMemory()
    store = FakeCuriosityStore(boom=True)
    driver = FakeCuriosityDriver()
    cfg = _cfg()

    await handle_turn(
        handle, memory, ollama, cfg, "EpeerD", "hey",
        None, None, store, driver,
    )

    # Reply still sent + stored despite the store blowing up.
    assert handle.sent == [("EpeerD", "still here")]
    system = ollama.chat_calls[0][0]
    assert "## You're curious about" not in system["content"]
    # Detection still fires for the completed exchange.
    assert driver.detect_calls == [("EpeerD", "hey", "still here")]


@pytest.mark.asyncio
async def test_handle_turn_without_curiosity_makes_no_curiosity_calls():
    """Flag-off parity: no store/driver → no curiosity block, byte-identical."""
    handle = FakeHandle()
    ollama = FakeOllama(reply="ok")
    memory = FakeMemory()
    cfg = _cfg()

    await handle_turn(handle, memory, ollama, cfg, "EpeerD", "hey")

    system = ollama.chat_calls[0][0]
    assert "## You're curious about" not in system["content"]


class FakeReflectionStore:
    def __init__(self, facts=None, opinions=None, *, boom: bool = False):
        self._facts = facts or []
        self._opinions = opinions or []
        self._boom = boom
        self.persona_calls: list[tuple[str, int]] = []

    async def persona(self, peer, *, max_chars):
        self.persona_calls.append((peer, max_chars))
        if self._boom:
            raise RuntimeError("DSN=postgresql://secret store down")
        return list(self._facts), list(self._opinions)


class FakeReflector:
    def __init__(self):
        self.reflect_calls: list[str] = []

    async def maybe_reflect(self, peer):
        self.reflect_calls.append(peer)


@pytest.mark.asyncio
async def test_handle_turn_injects_persona_and_fires_reflection():
    handle = FakeHandle()
    ollama = FakeOllama(reply="sure")
    memory = FakeMemory()
    store = FakeReflectionStore(
        facts=["Works as a backend developer"], opinions=["I like their bluntness"]
    )
    reflector = FakeReflector()
    cfg = _cfg()

    await handle_turn(
        handle, memory, ollama, cfg, "EpeerD", "hey",
        None, None, None, None, store, reflector,
    )

    system = ollama.chat_calls[0][0]
    assert system["role"] == "system"
    assert "## What you've come to know" in system["content"]
    assert "Works as a backend developer" in system["content"]
    assert "I like their bluntness" in system["content"]
    # Persona was fetched with the configured char cap; reflection fired post-reply.
    assert store.persona_calls == [("EpeerD", cfg.persona_max_chars)]
    assert reflector.reflect_calls == ["EpeerD"]


@pytest.mark.asyncio
async def test_handle_turn_survives_persona_read_fault():
    """Persona is additive: a store read fault must not break the reply or leak
    DSN/exception text — the turn proceeds with no persona block."""
    handle = FakeHandle()
    ollama = FakeOllama(reply="still here")
    memory = FakeMemory()
    store = FakeReflectionStore(boom=True)
    reflector = FakeReflector()
    cfg = _cfg()

    await handle_turn(
        handle, memory, ollama, cfg, "EpeerD", "hey",
        None, None, None, None, store, reflector,
    )

    assert handle.sent == [("EpeerD", "still here")]
    system = ollama.chat_calls[0][0]
    assert "## What you've come to know" not in system["content"]
    # Reflection still fires for the completed exchange.
    assert reflector.reflect_calls == ["EpeerD"]


@pytest.mark.asyncio
async def test_handle_turn_without_reflection_makes_no_persona_calls():
    """Flag-off parity: no store/reflector → no persona block, byte-identical."""
    handle = FakeHandle()
    ollama = FakeOllama(reply="ok")
    memory = FakeMemory()
    cfg = _cfg()

    await handle_turn(handle, memory, ollama, cfg, "EpeerD", "hey")

    system = ollama.chat_calls[0][0]
    assert "## What you've come to know" not in system["content"]


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
    """FakeMemory plus the command-path methods, recording cutoff/forget calls."""

    def __init__(self):
        super().__init__()
        self.cutoffs: list[str] = []
        self.forgotten: list[str] = []

    async def set_history_cutoff(self, peer):
        self.cutoffs.append(peer)

    async def forget(self, peer):
        self.forgotten.append(peer)
        return 4

    async def count(self, peer):
        return len(self.remembered)

    async def total(self):
        return len(self.remembered)


class _FakeEventsHandle:
    """Minimal ServiceHandle-like object for driving `_drain_events` in tests."""

    def __init__(self, events: list):
        self._events = events
        self.sent: list[tuple[str, str]] = []
        self.results: list[tuple[str, str]] = []

    async def events(self):
        for ev in self._events:
            yield ev

    async def send_message(self, to_addr: str, text: str) -> None:
        self.sent.append((to_addr, text))

    async def send_command_result(self, command_id: str, text: str, ok: bool = True) -> None:
        self.results.append((command_id, text))


def _commands_cfg(**extra) -> Config:
    env = {"JEFF_DB_URL": "postgresql://unused", "JEFF_ALLOWLIST": "EpeerD"}
    env.update(extra)
    return Config.from_env(env)


def _invocation(name: str, args: str = "", *, peer: str = "EpeerD", cid: str = "c1"):
    return ensemble.CommandInvocation(
        type="command", command_id=cid, from_addr=peer, name=name, args=args
    )


async def _drain_commands(events: list, memory) -> _FakeEventsHandle:
    """Drive `_drain_events` over a fixed event list with commands wired."""
    from jeff.commands import build_command_registry

    handle = _FakeEventsHandle(events)
    dispatcher = TurnDispatcher(
        lambda peer, text: _noop(), DispatchPolicy(max_inflight=10, peer_rate_burst=10)
    )
    await _drain_events(handle, dispatcher, _commands_cfg(), memory, build_command_registry())
    await dispatcher.drain()
    return handle


async def _noop() -> None:
    return None


@pytest.mark.asyncio
async def test_command_invocation_clear_resets_and_replies_on_command_channel():
    """A /clear invocation runs the session reset and replies via the command
    channel (a CommandResult), NOT as a chat message — and stores nothing."""
    memory = FakeCommandMemory()
    handle = await _drain_commands([_invocation("clear")], memory)

    assert memory.cutoffs == ["EpeerD"]
    assert len(handle.results) == 1
    cid, text = handle.results[0]
    assert cid == "c1" and "fresh conversation" in text.lower()
    # Reply went on the command channel, not the chat channel; nothing stored.
    assert handle.sent == []
    assert memory.remembered == []


@pytest.mark.asyncio
async def test_command_invocation_forget_yes_wipes_only_jeff_memory():
    """/forget yes wipes Jeff's memory and makes no attempt at a daemon transcript."""
    memory = FakeCommandMemory()
    handle = await _drain_commands([_invocation("forget", "yes")], memory)

    assert memory.forgotten == ["EpeerD"]
    assert len(handle.results) == 1 and "clean slate" in handle.results[0][1].lower()
    # No chat-channel traffic at all (no cross-channel reach).
    assert handle.sent == []


@pytest.mark.asyncio
async def test_command_invocation_from_non_allowlisted_peer_is_ignored():
    memory = FakeCommandMemory()
    handle = await _drain_commands([_invocation("clear", peer="Estranger")], memory)

    assert memory.cutoffs == []
    assert handle.results == []


@pytest.mark.asyncio
async def test_command_invocation_handler_raise_replies_safely():
    """A handler that raises still replies (content-safe), never escapes the loop."""

    class BoomMemory(FakeCommandMemory):
        async def set_history_cutoff(self, peer):
            raise RuntimeError("DSN=postgresql://secret leaked")

    memory = BoomMemory()
    handle = await _drain_commands([_invocation("clear")], memory)

    assert len(handle.results) == 1
    _, text = handle.results[0]
    assert "snag" in text.lower()
    assert "secret" not in text and "DSN" not in text


@pytest.mark.asyncio
async def test_drain_events_marks_presence_on_inbound_events():
    """Every inbound event marks the peer present, so the proactive loop's
    'don't shout into the void' gate sees chats AND commands as reachability."""
    from datetime import datetime, timezone

    from jeff.commands import build_command_registry
    from jeff.presence import Presence

    presence = Presence()
    handle = _FakeEventsHandle(
        [
            _invocation("clear"),  # a command counts as reachability too
            ensemble.ChatMessage(type="chat", from_addr="EpeerD", text="hi", ts=0),
        ]
    )
    dispatcher = TurnDispatcher(
        lambda peer, text: _noop(), DispatchPolicy(max_inflight=10, peer_rate_burst=10)
    )
    await _drain_events(
        handle,
        dispatcher,
        _commands_cfg(),
        FakeCommandMemory(),
        build_command_registry(),
        presence=presence,
    )
    await dispatcher.drain()
    assert presence.is_present(
        "EpeerD", now=datetime.now(timezone.utc), ttl_s=60
    )


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
    # No assistant message stored (the failed reply isn't a real turn), but the
    # peer DOES get a graceful, content-safe apology instead of silence.
    assert ("EpeerD", "assistant", _TURN_FAILED_MESSAGE) not in memory.remembered
    assert handle.sent == [("EpeerD", _TURN_FAILED_MESSAGE)]


@pytest.mark.asyncio
async def test_handle_turn_failure_reply_never_leaks_exception_text():
    """The graceful apology must carry nothing from the exception — the exc
    string can embed peer-shaped Ollama response bodies (ollama._safe_excerpt)."""
    sentinel = "PEER_CONTROLLED_PAYLOAD_DO_NOT_LEAK"

    class BoomOllama:
        async def chat(self, messages, *, model):
            raise RuntimeError(sentinel)

        async def embed(self, text, *, model):
            return [0.0] * 4

    handle = FakeHandle()
    memory = FakeMemory()
    cfg = _cfg()

    await handle_turn(handle, memory, BoomOllama(), cfg, "EpeerD", "ping")

    assert len(handle.sent) == 1
    _, reply = handle.sent[0]
    assert reply == _TURN_FAILED_MESSAGE
    assert sentinel not in reply
    assert "RuntimeError" not in reply


@pytest.mark.asyncio
async def test_handle_turn_swallows_send_failure_during_error_path(caplog):
    """If even the failure-reply send raises, it's logged and never escapes the
    handler — failure-handling can't itself crash the event loop."""
    import logging

    class BoomOllama:
        async def chat(self, messages, *, model):
            raise RuntimeError("primary failure")

        async def embed(self, text, *, model):
            return [0.0] * 4

    class BoomSendHandle(FakeHandle):
        async def send_message(self, to_addr, text):
            raise RuntimeError("transport down too")

    handle = BoomSendHandle()
    memory = FakeMemory()
    cfg = _cfg()

    with caplog.at_level(logging.DEBUG, logger="jeff"):
        # Must not raise even though both the turn AND the apology send fail.
        await handle_turn(handle, memory, BoomOllama(), cfg, "EpeerD", "ping")

    # Nothing escaped; the send failure was reported on the operator log.
    assert any(
        "failed to send turn-failure reply" in r.getMessage() for r in caplog.records
    )


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
