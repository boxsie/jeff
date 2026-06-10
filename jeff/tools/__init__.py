"""Jeff's tool-use subsystem: the registry, the base `Tool`, and a builder.

`build_registry(cfg)` is the single place that decides which tools Jeff exposes
for a given config. The turn loop (`jeff.main`) calls it once at startup and
hands the resulting `ToolRegistry` to every turn. Feature tickets (web search,
file transfer, image gen) add their tool here behind whatever config gate they
own — keeping the wiring in one auditable spot.
"""

from __future__ import annotations

from ..config import Config
from ..impulses import ImpulseStore
from ..llm import ChatProvider
from ..memory import Memory
from ..mood import MoodStore
from ..pinned import PinnedMemoryStore
from ..searxng import SearxngClient
from .base import DEFAULT_TOOL_TIMEOUT_S, Tool, ToolRegistry
from .builtins import GetTimeTool
from .impulses import AdjustImpulseTool, ClearImpulseTool, SetImpulseTool
from .mood import ClearMoodTool, DefineMoodTool, SetMoodTool
from .reach import REACH_OUT_MAX_CHARS, ReachOutTool
from .recall import (
    RECALL_MAX_CHARS,
    SUMMARIZE_SPAN,
    SUMMARY_MAX_CHARS,
    RecallMemoryTool,
    SummarizeRecentTool,
)
from .remember import RememberTool
from .search import ImageSearchTool, WebSearchTool


__all__ = [
    "DEFAULT_TOOL_TIMEOUT_S",
    "Tool",
    "ToolRegistry",
    "build_registry",
    "build_self_turn_registry",
]


def _memory_tools(
    cfg: Config,
    memory: Memory | None,
    chat_provider: ChatProvider | None,
) -> list[Tool]:
    """The read-only memory-scan tools (recall + summarize), shared by the chat
    and self-turn registries. Both reuse the existing message embeddings; they're
    added only when a `Memory` (and, for the summariser, a provider) is supplied.
    """
    tools: list[Tool] = []
    if memory is not None:
        tools.append(
            RecallMemoryTool(memory, k=cfg.recall_k, max_chars=RECALL_MAX_CHARS)
        )
        if chat_provider is not None:
            tools.append(
                SummarizeRecentTool(
                    memory,
                    chat_provider,
                    model=cfg.chat_model,
                    span=SUMMARIZE_SPAN,
                    max_chars=SUMMARY_MAX_CHARS,
                )
            )
    return tools


def _inward_tools(
    cfg: Config,
    *,
    mood_store: MoodStore | None,
    pinned_store: PinnedMemoryStore | None,
    impulse_store: ImpulseStore | None,
) -> list[Tool]:
    """Jeff's self-directed inward verbs (mood / impulse / remember), each added
    only when its feature is enabled and its store is supplied. Shared by the
    chat registry and the self-turn registry — the self-turn is exactly "Jeff
    acting on its own state through these verbs"."""
    tools: list[Tool] = []
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
    if cfg.impulses_enabled and impulse_store is not None:
        kw = {
            "default_hours": cfg.impulses_default_hours,
            "max_hours": cfg.impulses_max_hours,
            "max_chars": cfg.impulses_max_chars,
        }
        tools.append(SetImpulseTool(impulse_store, **kw))
        tools.append(AdjustImpulseTool(impulse_store, **kw))
        tools.append(ClearImpulseTool(impulse_store, **kw))
    return tools


def build_registry(
    cfg: Config,
    *,
    searxng: SearxngClient | None = None,
    memory: Memory | None = None,
    chat_provider: ChatProvider | None = None,
    mood_store: MoodStore | None = None,
    pinned_store: PinnedMemoryStore | None = None,
    impulse_store: ImpulseStore | None = None,
) -> ToolRegistry:
    """Construct the active tool registry for this configuration.

    Returns an empty registry when tools are disabled, so the turn loop's
    "registry is empty → behave exactly like the no-tools path" branch fires
    and production is byte-identical to today until a tool is actually enabled.

    The SearXNG-backed search tools are added only when `cfg.search_enabled`
    AND a `searxng` client is supplied — search rides on the tool-use
    foundation, so it requires tools to be on as well. The mood tools (when
    `cfg.mood_enabled` + a `mood_store`) and the remember tool (when
    `cfg.remember_enabled` + a `pinned_store`) follow the same principle. The
    memory-scan tools (recall / summarize) are added whenever a `memory` (and,
    for the summariser, a `chat_provider`) is supplied.
    """
    if not cfg.tools_enabled:
        return ToolRegistry()

    tools: list[Tool] = [GetTimeTool()]
    if cfg.search_enabled and searxng is not None:
        tools.append(WebSearchTool(searxng))
        tools.append(ImageSearchTool(searxng))
    tools.extend(_memory_tools(cfg, memory, chat_provider))
    tools.extend(
        _inward_tools(
            cfg,
            mood_store=mood_store,
            pinned_store=pinned_store,
            impulse_store=impulse_store,
        )
    )
    return ToolRegistry(tools)


def build_self_turn_registry(
    cfg: Config,
    *,
    memory: Memory | None = None,
    chat_provider: ChatProvider | None = None,
    mood_store: MoodStore | None = None,
    pinned_store: PinnedMemoryStore | None = None,
    impulse_store: ImpulseStore | None = None,
    handle=None,
    presence=None,
    proactive_store=None,
    curiosity_store=None,
) -> ToolRegistry:
    """The registry the idle self-turn wields: Jeff's INWARD verbs (mood /
    impulse / remember) + the read-only memory-scan tools (recall / summarize),
    plus the ONE outward verb `reach_out` when proactive messaging is enabled.

    `reach_out` is added only when `cfg.proactive_enabled` AND the wire-bound
    collaborators are supplied (`handle`, `presence`, `proactive_store`, plus a
    `memory` for send bookkeeping). Its presence/min-gap/mute gate lives inside
    the tool, so the loop still fires inward-only when the operator's offline —
    the door to them is gated on the verb, not the loop. Search is never added
    (network/outward, not inner life).

    Gated by `cfg.self_turn_enabled` (returns empty when off, so the loop is
    never built). Independent of `cfg.tools_enabled`: the self-turn is its own
    feature and can run even if the inbound chat path isn't tool-enabled — though
    in practice both are on together.
    """
    if not cfg.self_turn_enabled:
        return ToolRegistry()
    tools = _memory_tools(cfg, memory, chat_provider)
    tools.extend(
        _inward_tools(
            cfg,
            mood_store=mood_store,
            pinned_store=pinned_store,
            impulse_store=impulse_store,
        )
    )
    if (
        cfg.proactive_enabled
        and handle is not None
        and presence is not None
        and proactive_store is not None
        and memory is not None
    ):
        tools.append(
            ReachOutTool(
                handle,
                proactive_store,
                presence,
                memory,
                curiosity_store=curiosity_store,
                min_gap_s=cfg.proactive_min_gap_s,
                presence_ttl_s=cfg.proactive_presence_ttl_s,
                max_chars=REACH_OUT_MAX_CHARS,
            )
        )
    return ToolRegistry(tools)
