"""Async client for a self-hosted SearXNG metasearch instance.

SearXNG is a privacy-preserving metasearch proxy: it queries upstream engines
(Google, Bing, …) on Jeff's behalf, so Jeff only ever talks to SearXNG. That's
the whole point — see the phase's outbound-privacy rule. This client speaks the
JSON API (`GET /search?format=json`) and returns the parsed `results[]` for the
web-search / image-search tools to format.

Hardening mirrors the LLM clients (`jeff/_http.py`):
- `read_bounded` caps the response body (a hostile/replaced upstream can't OOM
  us with a giant body),
- `safe_excerpt` strips control chars + truncates before any body text reaches
  an exception/log (log-injection defence),
- an optional auth header value lives only in the request header — never in a
  URL, payload, log line, or exception (same discipline as the xAI key).

This client does NOT fetch result/image URLs — only the SearXNG endpoint. The
tools surface links; the operator clicks. Auto-fetching would re-leak interest
to third parties and is an SSRF vector.
"""

from __future__ import annotations

import json
import logging
from typing import Sequence

import httpx

from ._http import ResponseTooLargeError, read_bounded, safe_excerpt
from .ollama import DEFAULT_MAX_RESP_BYTES


log = logging.getLogger("jeff.searxng")

# How many results to keep at the client boundary regardless of what the model
# asks for — keeps the tool result small enough to stay cheap in context.
MAX_RESULTS_CAP = 8
DEFAULT_RESULTS = 5


class SearxngError(Exception):
    """A SearXNG request failed (bad status, malformed body, oversize, transport)."""


class SearxngClient:
    """Async SearXNG JSON-API client. One per process; an async ctx manager."""

    def __init__(
        self,
        base_url: str,
        *,
        auth: str | None = None,
        timeout: float = 30.0,
        max_resp_bytes: int = DEFAULT_MAX_RESP_BYTES,
        safesearch: int = 0,
    ):
        headers: dict[str, str] = {}
        if auth:
            # The operator supplies the full header value (e.g. "Basic <b64>" or
            # "Bearer <token>"). It rides only in the header — never logged.
            headers["Authorization"] = auth
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            limits=limits,
            headers=headers,
        )
        self._max_resp_bytes = max_resp_bytes
        # Default safesearch level applied to every query unless a call overrides
        # it. 0 = off (the operator wants unfiltered search), 1 = moderate,
        # 2 = strict. Wired from JEFF_SEARCH_SAFESEARCH via config.
        self._safesearch = safesearch

    async def search(
        self,
        query: str,
        *,
        categories: str = "general",
        max_results: int = DEFAULT_RESULTS,
        safesearch: int | None = None,
    ) -> list[dict]:
        """GET /search?format=json. Returns the parsed `results[]`, bounded.

        `categories` is "general" for web search, "images" for image search.
        `safesearch` defaults to the client-wide level (`JEFF_SEARCH_SAFESEARCH`,
        off by default) when not given. Raises `SearxngError` on any failure; the
        calling tool turns that into a safe string for the model.
        """
        n = max(1, min(int(max_results), MAX_RESULTS_CAP))
        level = self._safesearch if safesearch is None else safesearch
        params = {
            "q": query,
            "format": "json",
            "categories": categories,
            "safesearch": level,
        }
        try:
            async with self._client.stream("GET", "/search", params=params) as resp:
                try:
                    body = await read_bounded(resp, self._max_resp_bytes)
                except ResponseTooLargeError as e:
                    raise SearxngError(f"search: {e}") from None
                status = resp.status_code
        except httpx.HTTPError as e:
            # Transport-level failure. Stay generic — the exception text could
            # carry the endpoint, and we never want the auth header near a log.
            raise SearxngError(f"search: request failed ({type(e).__name__})") from None

        if status >= 400:
            text = body.decode("utf-8", errors="replace")
            raise SearxngError(f"search: {status} {safe_excerpt(text)}")
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise SearxngError(f"search: invalid JSON ({e.msg})") from None

        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return []
        return [r for r in results[:n] if isinstance(r, dict)]

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "SearxngClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


def _clip(value, limit: int = 300) -> str:
    """Coerce an upstream field to a short single-line string.

    Result fields are upstream-controlled (they come from third-party engines
    via SearXNG), so flatten newlines and bound length before they reach the
    model — keeps each result line compact and avoids a hostile snippet padding
    the context.
    """
    s = value if isinstance(value, str) else ("" if value is None else str(value))
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit] + "…"


def format_web_results(results: Sequence[dict]) -> str:
    """Numbered `title — url` + snippet lines for the model."""
    if not results:
        return "No results found."
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = _clip(r.get("title"), 160) or "(untitled)"
        url = _clip(r.get("url"), 400)
        snippet = _clip(r.get("content"), 300)
        line = f"{i}. {title} — {url}"
        if snippet:
            line += f"\n   {snippet}"
        lines.append(line)
    return "\n".join(lines)


def format_image_results(results: Sequence[dict]) -> str:
    """Numbered `title — image_url (source: page_url)` lines for the model."""
    if not results:
        return "No image results found."
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = _clip(r.get("title"), 160) or "(untitled)"
        img = _clip(r.get("img_src"), 400)
        source = _clip(r.get("url"), 400)
        resolution = _clip(r.get("resolution"), 40)
        line = f"{i}. {title} — {img}"
        if source:
            line += f" (source: {source})"
        if resolution:
            line += f" [{resolution}]"
        lines.append(line)
    return "\n".join(lines)
