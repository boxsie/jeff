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


# --- dispatch happy + failure paths ----------------------------------------


@pytest.mark.asyncio
async def test_dispatch_runs_tool_with_parsed_args():
    reg = ToolRegistry([EchoTool()])
    out = await reg.dispatch("echo", json.dumps({"message": "hi"}))
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
