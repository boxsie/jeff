"""Async Ollama HTTP client.

Wraps just the two endpoints the bot needs: /api/chat for conversation and
/api/embeddings for memory vectors. Streaming chat is intentionally deferred.

Hardening (W2 #beec3eed + #072b9e83)
-------------------------------------
Ollama response bodies are shaped by the peer's prompt (chat) or by the
attacker's text (embed). `_safe_excerpt` truncates + strips control chars
before they reach OllamaError / container logs (log-injection defence).

Response sizes are also bounded: a hostile or replaced local Ollama could
ship a 10 GiB body and OOM-kill us. `read_bounded` aborts streaming once
the byte budget is hit, and a Content-Length pre-check shortcuts the obvious
case. Embedding length is capped separately before `float()` materialisation
so a `{"embedding": [0]*10_000_000}` reply can't drive the daemon OOM.

The log-injection + response-size helpers are shared with the other LLM
providers — see `jeff/_http.py`.
"""

from __future__ import annotations

import json
from typing import Sequence

import httpx

from ._http import ResponseTooLargeError, read_bounded
from ._http import safe_excerpt as _safe_excerpt

# Defaults — overridable via Ollama() ctor kwargs / env-driven Config.
DEFAULT_MAX_RESP_BYTES = 8 * 1024 * 1024  # 8 MiB
DEFAULT_MAX_EMBED_DIM = 8192


class OllamaError(Exception):
    pass


class Ollama:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 120.0,
        max_resp_bytes: int = DEFAULT_MAX_RESP_BYTES,
        max_embed_dim: int = DEFAULT_MAX_EMBED_DIM,
    ):
        # Connection pool caps prevent slow-Ollama scenarios from leaking
        # sockets faster than they can drain. Numbers are small because the
        # daemon is single-tenant (one Jeff process → one Ollama).
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            limits=limits,
        )
        self._max_resp_bytes = max_resp_bytes
        self._max_embed_dim = max_embed_dim

    async def _post_bounded(self, path: str, payload: dict, *, kind: str) -> tuple[int, bytes]:
        """POST and read the body under the byte cap.

        Returns (status_code, body_bytes). `kind` is "chat" or "embed" so
        OllamaError messages carry the offending endpoint.
        """
        async with self._client.stream("POST", path, json=payload) as resp:
            try:
                body = await read_bounded(resp, self._max_resp_bytes)
            except ResponseTooLargeError as e:
                raise OllamaError(f"{kind}: {e}") from None
            return resp.status_code, body

    async def chat(self, messages: Sequence[dict], *, model: str) -> str:
        """POST /api/chat. Returns the assistant message content."""
        status, body = await self._post_bounded(
            "/api/chat",
            {"model": model, "messages": list(messages), "stream": False},
            kind="chat",
        )
        if status >= 400:
            text = body.decode("utf-8", errors="replace")
            raise OllamaError(f"chat: {status} {_safe_excerpt(text)}")
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise OllamaError(f"chat: invalid JSON ({e.msg})") from None
        msg = data.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            keys = sorted(data.keys()) if isinstance(data, dict) else type(data).__name__
            raise OllamaError(f"chat: malformed response (no message.content); keys={keys}")
        return content

    async def embed(self, text: str, *, model: str) -> list[float]:
        """POST /api/embeddings. Returns the embedding vector."""
        status, body = await self._post_bounded(
            "/api/embeddings",
            {"model": model, "prompt": text},
            kind="embed",
        )
        if status >= 400:
            err_text = body.decode("utf-8", errors="replace")
            raise OllamaError(f"embed: {status} {_safe_excerpt(err_text)}")
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise OllamaError(f"embed: invalid JSON ({e.msg})") from None
        emb = data.get("embedding")
        if not isinstance(emb, list) or not emb:
            keys = sorted(data.keys()) if isinstance(data, dict) else type(data).__name__
            raise OllamaError(f"embed: malformed response (no embedding); keys={keys}")
        # Cheap length check before the (potentially expensive) float() pass.
        # A hostile Ollama could ship {"embedding": [0]*10_000_000} — the
        # response-size cap caught most of it, but the *parsed* list could
        # still be huge if it's mostly small ints. Cap here too.
        if len(emb) > self._max_embed_dim:
            raise OllamaError(
                f"embed: embedding length {len(emb)} exceeds max {self._max_embed_dim}"
            )
        return [float(x) for x in emb]

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "Ollama":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
