"""Pluggable chat-LLM providers.

Jeff's chat turn talks to whatever provider `cfg.llm_provider` selects, behind
the minimal `ChatProvider` interface below. The local `Ollama` client already
satisfies it (its `chat` signature is the contract), so `ollama` stays the
back-compat default; `XaiProvider` adds cloud Grok.

Embeddings are deliberately NOT part of this abstraction — they stay on the
local Ollama (`nomic-embed-text` is tiny and xAI has no embeddings API). See
`jeff/main.py`, which keeps a separate Ollama client for `Memory` regardless of
the chat provider.

Adding a provider = one new class implementing `ChatProvider` + a branch in
`make_chat_provider` + its config keys in `jeff/config.py`. Nothing else moves.
"""

from __future__ import annotations

import json
from typing import Protocol, Sequence, runtime_checkable

import httpx

from ._http import ResponseTooLargeError, read_bounded, safe_excerpt
from .chat_types import ChatResult, ToolCall, parse_openai_tool_calls
from .config import Config, ConfigError
from .ollama import DEFAULT_MAX_RESP_BYTES, Ollama

# Re-exported so callers can keep importing the canonical shapes from `jeff.llm`
# (their natural home is the provider layer); the definitions live in
# `chat_types` only to break the llm<->ollama import cycle.
__all__ = [
    "ChatProvider",
    "ChatResult",
    "LLMError",
    "ToolCall",
    "XaiProvider",
    "make_chat_provider",
]


class LLMError(Exception):
    """Provider-agnostic chat failure (bad status, malformed body, oversize)."""


@runtime_checkable
class ChatProvider(Protocol):
    """The contract Jeff's turn loop depends on.

    A provider is an async context manager (so the loop can `async with` it for
    connection-pool lifecycle). `chat` is the simple single-shot path (returns
    the assistant string); `complete` is the tool-aware path (returns a
    `ChatResult` and accepts optional `tools` JSON-schema specs). The signatures
    match `Ollama` on purpose — `Ollama` satisfies this unchanged.
    """

    async def chat(self, messages: Sequence[dict], *, model: str) -> str: ...

    async def complete(
        self,
        messages: Sequence[dict],
        *,
        model: str,
        tools: Sequence[dict] | None = None,
    ) -> ChatResult: ...

    async def __aenter__(self) -> "ChatProvider": ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...


class XaiProvider:
    """xAI Grok chat provider.

    xAI exposes an OpenAI-compatible surface: `POST {base_url}/chat/completions`
    with a `Bearer` token and `{model, messages}`, returning
    `choices[0].message.content`.

    Security: the API key lives only in the `Authorization` request header — it
    is never placed in a URL, a payload, a log line, or an exception. Response
    bodies (attacker-influenceable via the peer's prompt) are size-bounded and
    run through `safe_excerpt` before they can reach an error string, mirroring
    the Ollama client's hardening.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        timeout: float = 120.0,
        max_resp_bytes: int = DEFAULT_MAX_RESP_BYTES,
    ):
        if not api_key:
            # Defence in depth — config fails fast first, but never construct a
            # provider that would send an empty Bearer token to the wire.
            raise ValueError("XaiProvider requires a non-empty api_key")
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            limits=limits,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._max_resp_bytes = max_resp_bytes

    async def chat(self, messages: Sequence[dict], *, model: str) -> str:
        """POST /chat/completions. Returns the assistant message content.

        The single-shot path — no tools. Thin wrapper over `complete`; raises
        if the model didn't return a content string (e.g. it tried to call a
        tool when none were offered, which shouldn't happen here).
        """
        res = await self.complete(messages, model=model)
        if res.content is None:
            raise LLMError("chat: malformed response (no choices[0].message.content)")
        return res.content

    async def complete(
        self,
        messages: Sequence[dict],
        *,
        model: str,
        tools: Sequence[dict] | None = None,
    ) -> ChatResult:
        """POST /chat/completions, returning content and/or tool-call requests.

        Messages are already in the OpenAI canonical shape Grok speaks (the
        turn loop builds them that way), so they pass straight through —
        including assistant `tool_calls` and `role:"tool"` follow-ups. `tools`
        is added to the payload only when supplied, so the no-tools payload is
        byte-identical to the pre-tool client.
        """
        payload: dict = {"model": model, "messages": list(messages), "stream": False}
        if tools:
            payload["tools"] = list(tools)
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                try:
                    body = await read_bounded(resp, self._max_resp_bytes)
                except ResponseTooLargeError as e:
                    raise LLMError(f"chat: {e}") from None
                status = resp.status_code
        except httpx.HTTPError as e:
            # Transport-level failure (timeout, connect, read). httpx exception
            # text never carries our auth header, but stay generic regardless.
            raise LLMError(f"chat: request failed ({type(e).__name__})") from None

        if status >= 400:
            text = body.decode("utf-8", errors="replace")
            raise LLMError(f"chat: {status} {safe_excerpt(text)}")
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise LLMError(f"chat: invalid JSON ({e.msg})") from None

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            keys = sorted(data.keys()) if isinstance(data, dict) else type(data).__name__
            raise LLMError(f"chat: malformed response (no choices); keys={keys}")
        first = choices[0]
        msg = first.get("message") if isinstance(first, dict) else None
        if not isinstance(msg, dict):
            raise LLMError("chat: malformed response (no choices[0].message)")
        content = msg.get("content")
        content = content if isinstance(content, str) else None
        tool_calls = parse_openai_tool_calls(msg.get("tool_calls"))
        if content is None and not tool_calls:
            raise LLMError("chat: malformed response (no content or tool_calls)")
        return ChatResult(content=content, tool_calls=tool_calls)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "XaiProvider":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


def make_chat_provider(cfg: Config) -> ChatProvider:
    """Build the chat provider selected by `cfg.llm_provider`.

    `cfg` has already validated the provider name and (for grok) the presence
    of the API key at load time, so this is a straight dispatch with an
    exhaustive guard.
    """
    if cfg.llm_provider == "ollama":
        return Ollama(
            cfg.ollama_url,
            max_resp_bytes=cfg.ollama_max_resp_bytes,
            max_embed_dim=cfg.ollama_max_embed_dim,
        )
    if cfg.llm_provider == "grok":
        # Validated non-None in Config.from_env when provider == grok.
        assert cfg.xai_api_key is not None
        return XaiProvider(
            api_key=cfg.xai_api_key,
            base_url=cfg.xai_base_url,
            max_resp_bytes=cfg.ollama_max_resp_bytes,
        )
    raise ConfigError(f"unknown llm provider: {cfg.llm_provider!r}")
