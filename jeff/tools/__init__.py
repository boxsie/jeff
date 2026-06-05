"""Jeff's tool-use subsystem: the registry, the base `Tool`, and a builder.

`build_registry(cfg)` is the single place that decides which tools Jeff exposes
for a given config. The turn loop (`jeff.main`) calls it once at startup and
hands the resulting `ToolRegistry` to every turn. Feature tickets (web search,
file transfer, image gen) add their tool here behind whatever config gate they
own — keeping the wiring in one auditable spot.
"""

from __future__ import annotations

from ..config import Config
from ..mood import MoodStore
from ..pinned import PinnedMemoryStore
from ..searxng import SearxngClient
from .base import DEFAULT_TOOL_TIMEOUT_S, Tool, ToolRegistry
from .builtins import GetTimeTool
from .mood import ClearMoodTool, DefineMoodTool, SetMoodTool
from .remember import RememberTool
from .search import ImageSearchTool, WebSearchTool


__all__ = [
    "DEFAULT_TOOL_TIMEOUT_S",
    "Tool",
    "ToolRegistry",
    "build_registry",
]


def build_registry(
    cfg: Config,
    *,
    searxng: SearxngClient | None = None,
    mood_store: MoodStore | None = None,
    pinned_store: PinnedMemoryStore | None = None,
) -> ToolRegistry:
    """Construct the active tool registry for this configuration.

    Returns an empty registry when tools are disabled, so the turn loop's
    "registry is empty → behave exactly like the no-tools path" branch fires
    and production is byte-identical to today until a tool is actually enabled.

    The SearXNG-backed search tools are added only when `cfg.search_enabled`
    AND a `searxng` client is supplied — search rides on the tool-use
    foundation, so it requires tools to be on as well. The mood tools (when
    `cfg.mood_enabled` + a `mood_store`) and the remember tool (when
    `cfg.remember_enabled` + a `pinned_store`) follow the same principle.
    """
    if not cfg.tools_enabled:
        return ToolRegistry()

    tools: list[Tool] = [GetTimeTool()]
    if cfg.search_enabled and searxng is not None:
        tools.append(WebSearchTool(searxng))
        tools.append(ImageSearchTool(searxng))
    if cfg.mood_enabled and mood_store is not None:
        kw = {
            "default_hours": cfg.mood_default_hours,
            "max_hours": cfg.mood_max_hours,
            "max_chars": cfg.mood_max_chars,
        }
        tools.append(SetMoodTool(mood_store, **kw))
        tools.append(DefineMoodTool(mood_store, **kw))
        tools.append(ClearMoodTool(mood_store, **kw))
    if cfg.remember_enabled and pinned_store is not None:
        tools.append(RememberTool(pinned_store, max_chars=cfg.remember_max_chars))
    return ToolRegistry(tools)
