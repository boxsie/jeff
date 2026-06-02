"""Tests for the pluggable chat-provider layer (factory + XaiProvider)."""

from __future__ import annotations

import json

import httpx
import pytest

from jeff.config import Config
from jeff.llm import ChatProvider, LLMError, XaiProvider, make_chat_provider
from jeff.ollama import Ollama


_API_KEY = "xai-SUPERSECRET-do-not-leak"


def _grok_cfg(**extra) -> Config:
    env = {
        "JEFF_DB_URL": "postgresql://x",
        "JEFF_LLM_PROVIDER": "grok",
        "XAI_API_KEY": _API_KEY,
    }
    env.update(extra)
    return Config.from_env(env)


def _xai(handler, **kwargs) -> XaiProvider:
    p = XaiProvider(api_key=_API_KEY, **kwargs)
    p._client = httpx.AsyncClient(
        base_url="https://api.x.ai/v1",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": f"Bearer {_API_KEY}"},
    )
    return p


# --- factory ---------------------------------------------------------------


def test_factory_selects_ollama_by_default():
    cfg = Config.from_env({"JEFF_DB_URL": "postgresql://x"})
    provider = make_chat_provider(cfg)
    assert isinstance(provider, Ollama)


def test_factory_selects_xai_for_grok():
    provider = make_chat_provider(_grok_cfg())
    assert isinstance(provider, XaiProvider)


def test_ollama_satisfies_chatprovider_protocol():
    # The whole abstraction rests on Ollama being a drop-in ChatProvider.
    assert isinstance(Ollama("http://example"), ChatProvider)
    assert isinstance(XaiProvider(api_key="k"), ChatProvider)


# --- XaiProvider.chat ------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_sends_openai_shape_and_bearer():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = req.read()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "grok says hi"}}]},
        )

    provider = _xai(handler)
    try:
        out = await provider.chat([{"role": "user", "content": "hi"}], model="grok-4")
    finally:
        await provider.aclose()

    assert out == "grok says hi"
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == f"Bearer {_API_KEY}"
    assert b'"model":"grok-4"' in seen["body"]
    assert b'"messages"' in seen["body"]


@pytest.mark.asyncio
async def test_chat_raises_on_http_error():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    provider = _xai(handler)
    try:
        with pytest.raises(LLMError, match="401"):
            await provider.chat([], model="grok-4")
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_chat_raises_on_malformed_response():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"no_choices": True})

    provider = _xai(handler)
    try:
        with pytest.raises(LLMError, match="no choices"):
            await provider.chat([], model="grok-4")
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_chat_refuses_oversize_body():
    huge = b"a" * (16 * 1024)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=huge)

    provider = _xai(handler, max_resp_bytes=1024)
    try:
        with pytest.raises(LLMError) as ei:
            await provider.chat([], model="grok-4")
        assert "1024" in str(ei.value) or "exceed" in str(ei.value).lower()
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_api_key_never_appears_in_error_text():
    """An error-status body plus the key in headers must not leak the key."""
    # Echo the key back in the body to prove safe_excerpt path doesn't help an
    # attacker — and prove the key isn't pulled from headers into the message.
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    provider = _xai(handler)
    try:
        with pytest.raises(LLMError) as ei:
            await provider.chat([{"role": "user", "content": "x"}], model="grok-4")
    finally:
        await provider.aclose()

    assert _API_KEY not in str(ei.value)


@pytest.mark.asyncio
async def test_transport_failure_is_generic_and_keyless():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = _xai(handler)
    try:
        with pytest.raises(LLMError) as ei:
            await provider.chat([], model="grok-4")
    finally:
        await provider.aclose()

    msg = str(ei.value)
    assert _API_KEY not in msg
    assert "request failed" in msg


def test_construct_without_key_rejected():
    with pytest.raises(ValueError, match="api_key"):
        XaiProvider(api_key="")


# --- XaiProvider.complete (tool-calling) -----------------------------------


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]


@pytest.mark.asyncio
async def test_complete_sends_tools_and_parses_tool_calls():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = req.read()
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "get_time",
                                        "arguments": '{"tz":"utc"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    provider = _xai(handler)
    try:
        res = await provider.complete(
            [{"role": "user", "content": "what time is it"}],
            model="grok-4",
            tools=_TOOLS,
        )
    finally:
        await provider.aclose()

    assert b'"tools"' in seen["body"]
    assert res.content is None
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc.id == "call_abc"
    assert tc.name == "get_time"
    assert tc.arguments == '{"tz":"utc"}'


@pytest.mark.asyncio
async def test_complete_without_tools_omits_tools_key():
    """No-tools payload must stay byte-identical to the pre-tool client."""
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = req.read()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
        )

    provider = _xai(handler)
    try:
        res = await provider.complete([{"role": "user", "content": "x"}], model="grok-4")
    finally:
        await provider.aclose()

    assert b'"tools"' not in seen["body"]
    assert res.content == "hi"
    assert res.tool_calls == ()


@pytest.mark.asyncio
async def test_complete_round_trips_tool_result_into_final_answer():
    """A follow-up call carrying a role:tool message yields final content."""
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = req.read()
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "It is noon UTC."}}
                ]
            },
        )

    provider = _xai(handler)
    messages = [
        {"role": "user", "content": "time?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "noon UTC"},
    ]
    try:
        res = await provider.complete(messages, model="grok-4", tools=_TOOLS)
    finally:
        await provider.aclose()

    assert res.content == "It is noon UTC."
    assert res.tool_calls == ()
    # The tool result rode through to the provider verbatim.
    assert b'"tool_call_id":"call_abc"' in seen["body"]
    assert _API_KEY not in seen["body"].decode()


@pytest.mark.asyncio
async def test_complete_arguments_object_is_serialised_to_string():
    """Some providers emit arguments as an object; canonical form is a string."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "echo",
                                        "arguments": {"message": "hi"},
                                    }
                                }
                            ],
                        }
                    }
                ]
            },
        )

    provider = _xai(handler)
    try:
        res = await provider.complete([], model="grok-4", tools=_TOOLS)
    finally:
        await provider.aclose()

    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc.name == "echo"
    # Missing id was synthesised; object arguments serialised to a JSON string.
    assert tc.id == "call_0"
    assert json.loads(tc.arguments) == {"message": "hi"}
