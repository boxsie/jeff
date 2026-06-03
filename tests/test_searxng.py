"""Tests for the SearXNG client, result formatting, and the search tools."""

from __future__ import annotations

import httpx
import pytest

from jeff.searxng import (
    MAX_RESULTS_CAP,
    SearxngClient,
    SearxngError,
    format_image_results,
    format_web_results,
)
from jeff.tools.search import ImageSearchTool, WebSearchTool


_AUTH = "Bearer SEARX-SUPERSECRET-do-not-leak"


def _client(handler, **kwargs) -> SearxngClient:
    c = SearxngClient("http://searx.local", **kwargs)
    headers = {"Authorization": _AUTH} if kwargs.get("auth") else {}
    c._client = httpx.AsyncClient(
        base_url="http://searx.local",
        transport=httpx.MockTransport(handler),
        headers=headers,
    )
    return c


_WEB_BODY = {
    "results": [
        {"title": "Python", "url": "https://python.org", "content": "The language."},
        {"title": "Docs", "url": "https://docs.python.org", "content": "Reference."},
    ]
}
_IMAGE_BODY = {
    "results": [
        {
            "title": "A cat",
            "img_src": "https://img.example/cat.jpg",
            "url": "https://example.com/cat-page",
            "resolution": "1920x1080",
        }
    ]
}


# --- client request shape + parsing ----------------------------------------


@pytest.mark.asyncio
async def test_search_builds_json_general_query_and_parses_web():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json=_WEB_BODY)

    client = _client(handler)
    try:
        results = await client.search("python", categories="general")
    finally:
        await client.aclose()

    assert seen["path"] == "/search"
    assert seen["params"]["format"] == "json"
    assert seen["params"]["categories"] == "general"
    assert seen["params"]["q"] == "python"
    assert [r["title"] for r in results] == ["Python", "Docs"]


@pytest.mark.asyncio
async def test_search_defaults_safesearch_off():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json=_WEB_BODY)

    # No safesearch arg → client default (off) rides on the request.
    client = _client(handler)
    try:
        await client.search("python")
    finally:
        await client.aclose()
    assert seen["params"]["safesearch"] == "0"


@pytest.mark.asyncio
async def test_search_safesearch_client_default_and_call_override():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json=_WEB_BODY)

    # Client constructed with strict default…
    client = _client(handler, safesearch=2)
    try:
        await client.search("python")
        assert seen["params"]["safesearch"] == "2"
        # …but an explicit per-call level wins.
        await client.search("python", safesearch=0)
        assert seen["params"]["safesearch"] == "0"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_search_images_category():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json=_IMAGE_BODY)

    client = _client(handler)
    try:
        results = await client.search("cat", categories="images")
    finally:
        await client.aclose()

    assert seen["params"]["categories"] == "images"
    assert results[0]["img_src"] == "https://img.example/cat.jpg"


@pytest.mark.asyncio
async def test_search_bounds_result_count():
    many = {"results": [{"title": f"r{i}", "url": f"u{i}"} for i in range(50)]}

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=many)

    client = _client(handler)
    try:
        # Asked for more than the hard cap → clamped to the cap.
        results = await client.search("x", max_results=100)
        assert len(results) == MAX_RESULTS_CAP
        # A small explicit count is honoured.
        results2 = await client.search("x", max_results=3)
        assert len(results2) == 3
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_search_missing_results_key_returns_empty():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"query": "x"})

    client = _client(handler)
    try:
        assert await client.search("x") == []
    finally:
        await client.aclose()


# --- hardening -------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_refuses_oversize_body():
    huge = b"a" * (16 * 1024)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=huge)

    client = _client(handler, max_resp_bytes=1024)
    try:
        with pytest.raises(SearxngError) as ei:
            await client.search("x")
        assert "1024" in str(ei.value) or "exceed" in str(ei.value).lower()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_search_error_status_is_sanitised():
    hostile = "x" * 500 + "\n\r\x1b[31m<INJECT>\x1b[0m"

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=hostile)

    client = _client(handler)
    try:
        with pytest.raises(SearxngError) as ei:
            await client.search("x")
    finally:
        await client.aclose()

    msg = str(ei.value)
    assert len(msg) < 400
    for bad in ("\n", "\r", "\x1b"):
        assert bad not in msg
    assert "500" in msg


@pytest.mark.asyncio
async def test_search_transport_failure_is_generic():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _client(handler)
    try:
        with pytest.raises(SearxngError) as ei:
            await client.search("x")
    finally:
        await client.aclose()
    assert "request failed" in str(ei.value)


@pytest.mark.asyncio
async def test_auth_header_sent_and_never_in_error_text():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers.get("authorization") == _AUTH
        return httpx.Response(500, text="server error")

    client = _client(handler, auth=_AUTH)
    try:
        with pytest.raises(SearxngError) as ei:
            await client.search("x")
    finally:
        await client.aclose()
    assert _AUTH not in str(ei.value)


# --- formatting ------------------------------------------------------------


def test_format_web_results_is_compact_and_cited():
    out = format_web_results(_WEB_BODY["results"])
    assert "1. Python — https://python.org" in out
    assert "The language." in out
    assert "2. Docs — https://docs.python.org" in out


def test_format_web_results_empty():
    assert format_web_results([]) == "No results found."


def test_format_image_results_has_image_and_source_links():
    out = format_image_results(_IMAGE_BODY["results"])
    assert "https://img.example/cat.jpg" in out
    assert "source: https://example.com/cat-page" in out
    assert "1920x1080" in out


def test_format_clips_newlines_in_snippet():
    noisy = [{"title": "T", "url": "u", "content": "line1\nline2\n\nline3"}]
    out = format_web_results(noisy)
    assert "\nline2" not in out.split("\n", 1)[1].lstrip()  # snippet flattened
    assert "line1 line2 line3" in out


# --- tools -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_search_tool_returns_cited_string_and_hits_only_searxng():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        return httpx.Response(200, json=_WEB_BODY)

    client = _client(handler)
    try:
        out = await WebSearchTool(client).run(query="python")
    finally:
        await client.aclose()

    assert "python.org" in out
    # Exactly one request, to the SearXNG /search endpoint — no URL was fetched.
    assert calls == ["/search"]


@pytest.mark.asyncio
async def test_image_search_tool_returns_cited_string():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_IMAGE_BODY)

    client = _client(handler)
    try:
        out = await ImageSearchTool(client).run(query="cat")
    finally:
        await client.aclose()
    assert "img.example/cat.jpg" in out
    assert "source:" in out


@pytest.mark.asyncio
async def test_search_tool_survives_searxng_outage_with_safe_string():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = _client(handler)
    try:
        out = await WebSearchTool(client).run(query="python")
    finally:
        await client.aclose()
    # No raise — the model gets a safe, content-only string.
    assert out.startswith("error:")
    assert "unavailable" in out


@pytest.mark.asyncio
async def test_search_tool_empty_query_is_rejected_without_request():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:  # pragma: no cover
        calls.append(req.url.path)
        return httpx.Response(200, json=_WEB_BODY)

    client = _client(handler)
    try:
        out = await WebSearchTool(client).run(query="   ")
    finally:
        await client.aclose()
    assert out.startswith("error:")
    assert calls == []
