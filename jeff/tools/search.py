"""Web + image search tools backed by SearXNG.

Both tools talk only to the SearXNG instance (the privacy win) and return
links + text only — they never fetch the result/image URLs. The descriptions
say so explicitly, so the model surfaces citations the operator can click
rather than claiming it "read" or "viewed" anything.

A SearXNG failure is caught and turned into a short safe string — the turn
survives a search outage (the registry would also catch a raise, but handling
it here lets us give the model a cleaner, search-specific message).
"""

from __future__ import annotations

import logging
from typing import ClassVar

from ..searxng import (
    DEFAULT_RESULTS,
    MAX_RESULTS_CAP,
    SearxngClient,
    SearxngError,
    format_image_results,
    format_web_results,
)
from .base import Tool


log = logging.getLogger("jeff.tools.search")

_MAX_RESULTS_SCHEMA = {
    "type": "integer",
    "description": f"How many results to return (1–{MAX_RESULTS_CAP}, default {DEFAULT_RESULTS}).",
    "minimum": 1,
    "maximum": MAX_RESULTS_CAP,
}


class WebSearchTool(Tool):
    """Search the web via SearXNG. Returns titles, URLs, and snippets."""

    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = (
        "Search the web for current information and return a list of results, "
        "each with a title, URL, and a short snippet. Use this for anything you "
        "don't reliably know or that may have changed (news, facts, docs). "
        "Results are links only — you do NOT browse or read the pages, so cite "
        "the URLs and let the user open them; never claim you viewed a page."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "max_results": _MAX_RESULTS_SCHEMA,
        },
        "required": ["query"],
    }

    def __init__(self, client: SearxngClient):
        self._client = client

    async def run(self, **kwargs) -> str:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return "error: web_search requires a non-empty query"
        max_results = kwargs.get("max_results", DEFAULT_RESULTS)
        try:
            results = await self._client.search(
                query, categories="general", max_results=max_results
            )
        except SearxngError:
            log.warning("web_search failed")
            return "error: web search is unavailable right now"
        return format_web_results(results)


class ImageSearchTool(Tool):
    """Search for images via SearXNG. Returns image URLs + their source pages."""

    name: ClassVar[str] = "image_search"
    description: ClassVar[str] = (
        "Search for images and return a list of results, each with a title, the "
        "direct image URL, and the source page URL. Results are links only — you "
        "do NOT download or view the images; present the links so the user can "
        "open them, and never describe an image as if you had seen it."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The image search query."},
            "max_results": _MAX_RESULTS_SCHEMA,
        },
        "required": ["query"],
    }

    def __init__(self, client: SearxngClient):
        self._client = client

    async def run(self, **kwargs) -> str:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return "error: image_search requires a non-empty query"
        max_results = kwargs.get("max_results", DEFAULT_RESULTS)
        try:
            results = await self._client.search(
                query, categories="images", max_results=max_results
            )
        except SearxngError:
            log.warning("image_search failed")
            return "error: image search is unavailable right now"
        return format_image_results(results)
