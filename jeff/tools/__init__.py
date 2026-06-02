"""Jeff's tool-use subsystem: the registry, the base `Tool`, and a builder.

`build_registry(cfg)` is the single place that decides which tools Jeff exposes
for a given config. The turn loop (`jeff.main`) calls it once at startup and
hands the resulting `ToolRegistry` to every turn. Feature tickets (web search,
file transfer, image gen) add their tool here behind whatever config gate they
own — keeping the wiring in one auditable spot.
"""

from __future__ import annotations

from ..config import Config
from ..searxng import SearxngClient
from .base import DEFAULT_TOOL_TIMEOUT_S, Tool, ToolRegistry
from .builtins import GetTimeTool
from .search import ImageSearchTool, WebSearchTool


__all__ = [
    "DEFAULT_TOOL_TIMEOUT_S",
    "Tool",
    "ToolRegistry",
    "build_registry",
]


def build_registry(cfg: Config, *, searxng: SearxngClient | None = None) -> ToolRegistry:
    """Construct the active tool registry for this configuration.

    Returns an empty registry when tools are disabled, so the turn loop's
    "registry is empty → behave exactly like the no-tools path" branch fires
    and production is byte-identical to today until a tool is actually enabled.

    The SearXNG-backed search tools are added only when `cfg.search_enabled`
    AND a `searxng` client is supplied — search rides on the tool-use
    foundation, so it requires tools to be on as well.
    """
    if not cfg.tools_enabled:
        return ToolRegistry()

    tools: list[Tool] = [GetTimeTool()]
    if cfg.search_enabled and searxng is not None:
        tools.append(WebSearchTool(searxng))
        tools.append(ImageSearchTool(searxng))
    return ToolRegistry(tools)
