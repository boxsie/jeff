"""Tests for the tool registry, dispatch safety, and built-in tools."""

from __future__ import annotations

import asyncio
import json

import pytest

from jeff.config import Config
from jeff.tools import ToolRegistry, build_registry
from jeff.tools.base import Tool
from jeff.tools.builtins import GetTimeTool


class EchoTool(Tool):
    name = "echo"
    description = "Echo the message back."
    parameters = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    async def run(self, **kwargs) -> str:
        return f"echo: {kwargs['message']}"


class BoomTool(Tool):
    name = "boom"
    description = "Always raises."
    parameters = {"type": "object", "properties": {}, "required": []}

    async def run(self, **kwargs) -> str:
        raise RuntimeError("SECRET_INTERNAL_DETAIL_DO_NOT_LEAK")


class SlowTool(Tool):
    name = "slow"
    description = "Sleeps forever."
    parameters = {"type": "object", "properties": {}, "required": []}

    async def run(self, **kwargs) -> str:
        await asyncio.sleep(60)
        return "done"


class NonStringTool(Tool):
    name = "nonstring"
    description = "Returns a non-string."
    parameters = {"type": "object", "properties": {}, "required": []}

    async def run(self, **kwargs):
        return {"not": "a string"}


# --- registry / specs ------------------------------------------------------


def test_specs_are_valid_openai_function_shape():
    reg = ToolRegistry([EchoTool()])
    specs = reg.specs()
    assert len(specs) == 1
    spec = specs[0]
    assert spec["type"] == "function"
    fn = spec["function"]
    assert fn["name"] == "echo"
    assert fn["description"]
    assert fn["parameters"]["type"] == "object"
    # The whole spec must be JSON-serialisable (it's sent on the wire).
    json.dumps(spec)


def test_names_sorted_and_lookup():
    reg = ToolRegistry([EchoTool(), GetTimeTool()])
    assert reg.names() == ["echo", "get_time"]
    assert isinstance(reg.get("echo"), EchoTool)
    assert reg.get("missing") is None
    assert len(reg) == 2


def test_duplicate_registration_rejected():
    reg = ToolRegistry([EchoTool()])
    with pytest.raises(ValueError, match="duplicate"):
        reg.register(EchoTool())


class PeerEchoTool(Tool):
    """Declares needs_peer; echoes the injected peer + a model-supplied arg."""

    name = "peer_echo"
    description = "Echo the peer."
    parameters = {
        "type": "object",
        "properties": {"note": {"type": "string"}},
        "required": [],
    }
    needs_peer = True

    async def run(self, **kwargs) -> str:
        return f"peer={kwargs.get('peer')} note={kwargs.get('note')}"


# --- dispatch happy + failure paths ----------------------------------------


@pytest.mark.asyncio
async def test_dispatch_runs_tool_with_parsed_args():
    reg = ToolRegistry([EchoTool()])
    out = await reg.dispatch("echo", json.dumps({"message": "hi"}))
    assert out == "echo: hi"


@pytest.mark.asyncio
async def test_dispatch_injects_peer_for_needs_peer_tool():
    reg = ToolRegistry([PeerEchoTool()])
    out = await reg.dispatch("peer_echo", json.dumps({"note": "x"}), peer="EpeerA")
    assert "peer=EpeerA" in out
    assert "note=x" in out


@pytest.mark.asyncio
async def test_dispatch_peer_cannot_be_spoofed_by_model_args():
    # A model-supplied `peer` is dropped by validation (not a declared property),
    # then the real caller is injected — so it can't be aimed at another peer.
    reg = ToolRegistry([PeerEchoTool()])
    out = await reg.dispatch(
        "peer_echo", json.dumps({"peer": "EVIL", "note": "x"}), peer="EpeerA"
    )
    assert "peer=EpeerA" in out
    assert "EVIL" not in out


@pytest.mark.asyncio
async def test_dispatch_does_not_inject_peer_for_normal_tool():
    # Default needs_peer=False → no peer kwarg leaks into a tool that doesn't want
    # it (would otherwise trip run(**kwargs) for tools with strict signatures).
    reg = ToolRegistry([EchoTool()])
    out = await reg.dispatch("echo", json.dumps({"message": "hi"}), peer="EpeerA")
    assert out == "echo: hi"


@pytest.mark.asyncio
async def test_dispatch_empty_args_string_is_empty_object():
    reg = ToolRegistry([GetTimeTool()])
    out = await reg.dispatch("get_time", "")
    assert out.startswith("Current UTC time:")


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_returns_safe_error():
    reg = ToolRegistry([EchoTool()])
    out = await reg.dispatch("nope", "{}")
    assert out.startswith("error:")
    assert "unknown tool" in out


@pytest.mark.asyncio
async def test_dispatch_bad_json_returns_safe_error():
    reg = ToolRegistry([EchoTool()])
    out = await reg.dispatch("echo", "{not json")
    assert out.startswith("error:")
    assert "malformed JSON" in out


@pytest.mark.asyncio
async def test_dispatch_missing_required_returns_safe_error():
    reg = ToolRegistry([EchoTool()])
    out = await reg.dispatch("echo", json.dumps({"wrong": "key"}))
    assert out.startswith("error:")
    assert "missing required" in out


@pytest.mark.asyncio
async def test_dispatch_drops_unknown_kwargs():
    # An extra key the schema doesn't declare must not trip run(**kwargs).
    reg = ToolRegistry([EchoTool()])
    out = await reg.dispatch("echo", json.dumps({"message": "hi", "extra": 1}))
    assert out == "echo: hi"


@pytest.mark.asyncio
async def test_dispatch_raising_tool_does_not_leak_exception_text():
    reg = ToolRegistry([BoomTool()])
    out = await reg.dispatch("boom", "{}")
    assert out.startswith("error:")
    assert "SECRET_INTERNAL_DETAIL_DO_NOT_LEAK" not in out


@pytest.mark.asyncio
async def test_dispatch_timeout_returns_safe_error():
    reg = ToolRegistry([SlowTool()])
    out = await reg.dispatch("slow", "{}", timeout=0.01)
    assert out.startswith("error:")
    assert "timed out" in out


@pytest.mark.asyncio
async def test_dispatch_non_string_result_is_rejected():
    reg = ToolRegistry([NonStringTool()])
    out = await reg.dispatch("nonstring", "{}")
    assert out.startswith("error:")
    assert "non-string" in out


# --- builtin + builder -----------------------------------------------------


@pytest.mark.asyncio
async def test_get_time_returns_iso_utc():
    out = await GetTimeTool().run()
    assert "Current UTC time:" in out
    assert "+00:00" in out or "T" in out


def _cfg(**extra) -> Config:
    env = {"JEFF_DB_URL": "postgresql://x"}
    env.update(extra)
    return Config.from_env(env)


def test_build_registry_enabled_has_get_time():
    reg = build_registry(_cfg(JEFF_TOOLS_ENABLED="true"))
    assert "get_time" in reg.names()


def test_build_registry_disabled_is_empty():
    reg = build_registry(_cfg(JEFF_TOOLS_ENABLED="false"))
    assert len(reg) == 0


def test_build_registry_adds_search_tools_when_enabled():
    # build_registry only stores the client reference — a sentinel is enough.
    sentinel = object()
    reg = build_registry(
        _cfg(JEFF_SEARCH_ENABLED="true", JEFF_SEARXNG_URL="http://searx"),
        searxng=sentinel,
    )
    assert "web_search" in reg.names()
    assert "image_search" in reg.names()


def test_build_registry_omits_search_when_disabled():
    reg = build_registry(_cfg(), searxng=object())
    assert "web_search" not in reg.names()


def test_build_registry_adds_memory_tools_when_memory_supplied():
    # Sentinels are enough — build only stores the references.
    reg = build_registry(
        _cfg(JEFF_TOOLS_ENABLED="true"),
        memory=object(),
        chat_provider=object(),
    )
    assert "recall_memory" in reg.names()
    assert "summarize_recent" in reg.names()


def test_build_registry_omits_summarize_without_provider():
    reg = build_registry(_cfg(JEFF_TOOLS_ENABLED="true"), memory=object())
    assert "recall_memory" in reg.names()  # recall needs only memory
    assert "summarize_recent" not in reg.names()  # summarize needs the provider


def test_build_self_turn_registry_has_inward_and_memory_tools():
    from jeff.tools import build_self_turn_registry

    reg = build_self_turn_registry(
        _cfg(
            JEFF_SELF_TURN_ENABLED="true",
            JEFF_MOOD_ENABLED="true",
            JEFF_REMEMBER_ENABLED="true",
            JEFF_IMPULSES_ENABLED="true",
        ),
        memory=object(),
        chat_provider=object(),
        mood_store=object(),
        pinned_store=object(),
        impulse_store=object(),
    )
    names = reg.names()
    assert "set_mood" in names
    assert "set_impulse" in names
    assert "remember" in names
    assert "recall_memory" in names
    assert "summarize_recent" in names
    # No search, no get_time. No reach_out either — proactive wasn't enabled and
    # no wire collaborators were supplied.
    assert "web_search" not in names
    assert "get_time" not in names
    assert "reach_out" not in names


def test_build_self_turn_registry_adds_reach_out_when_proactive_on():
    from jeff.tools import build_self_turn_registry

    reg = build_self_turn_registry(
        _cfg(JEFF_SELF_TURN_ENABLED="true", JEFF_PROACTIVE_ENABLED="true"),
        memory=object(),
        handle=object(),
        presence=object(),
        proactive_store=object(),
        curiosity_store=object(),
    )
    assert "reach_out" in reg.names()


def test_build_self_turn_registry_omits_reach_out_without_collaborators():
    from jeff.tools import build_self_turn_registry

    # Proactive on but no handle/presence/store supplied → no outward verb.
    reg = build_self_turn_registry(
        _cfg(JEFF_SELF_TURN_ENABLED="true", JEFF_PROACTIVE_ENABLED="true"),
        memory=object(),
    )
    assert "reach_out" not in reg.names()


def test_build_self_turn_registry_empty_when_disabled():
    from jeff.tools import build_self_turn_registry

    reg = build_self_turn_registry(
        _cfg(JEFF_SELF_TURN_ENABLED="false", JEFF_MOOD_ENABLED="true"),
        memory=object(),
        chat_provider=object(),
        mood_store=object(),
    )
    assert len(reg) == 0
