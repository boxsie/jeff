"""The execute-and-loop that turns a message list + a tool registry into a final
answer, shared by the inbound chat path (`main.handle_turn`) and the idle
self-turn (`selfturn.SelfTurnLoop`).

Extracted from `main` so the self-turn can reuse it WITHOUT importing `main`
(which would be a cycle — `main` imports the self-turn loop). It depends only on
the provider contract (`llm.ChatProvider`/`ChatResult`) and the registry
(`tools.base.ToolRegistry`), never on `main`.

Safety carries over from the registry: `dispatch` returns a content-safe string
for every tool failure mode (unknown tool / bad args / raise / timeout), so the
loop never crashes on a tool fault. If the model never stops calling tools, the
loop returns a graceful cap message rather than spinning forever.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .config import Config
from .economy import cost_for
from .llm import ChatProvider, ChatResult
from .tools.base import ToolRegistry

if TYPE_CHECKING:
    from .appraisal import DriveState


log = logging.getLogger("jeff.turnloop")


# Returned when the model keeps calling tools past `cfg.max_tool_iters` without
# settling on a final answer — a graceful, content-safe stop.
TOOL_CAP_MESSAGE = (
    "Sorry — I couldn't finish working through that within my tool-use limit. "
    "Could you try rephrasing or narrowing the request?"
)


def _assistant_tool_message(result: ChatResult) -> dict:
    """Render a tool-calling assistant turn back into OpenAI-canonical form.

    This is the message the model must see echoed before its tool results, so
    it can correlate each `tool_call_id`. Content is preserved if the model
    narrated alongside the calls.
    """
    return {
        "role": "assistant",
        "content": result.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in result.tool_calls
        ],
    }


async def run_tool_loop(
    chat_provider: ChatProvider,
    registry: ToolRegistry,
    history: list[dict],
    cfg: Config,
    peer: str,
    *,
    drive_store: "DriveState | None" = None,
) -> str:
    """Call the provider, execute any tool calls, repeat until a final answer.

    Bounded by `cfg.max_tool_iters`. Each tool result is fed back as a
    `role:"tool"` message; the registry guarantees a safe string even for
    unknown tools / bad args / raises / timeouts, so the loop never crashes on
    a tool fault. Returns the model's final content, or a graceful cap message
    if it never stops calling tools.

    `peer` is the current turn's address; it's passed to `dispatch` so
    peer-scoped tools (e.g. the mood tools) write to the right peer's state.

    When `drive_store` is supplied (the drive economy is on), invoking a costed
    verb **debits** its drive currency (`economy.COSTS`) — this is the single seam
    where self-turn verbs and impulse/mood acts spend, in both the chat and
    self-turn paths. `reach_out` is deliberately NOT in COSTS: it self-charges
    inside the tool, conditional on the send succeeding. The debit is best-effort
    and post-dispatch — a spend fault never breaks the turn (mirrors appraisal).
    """
    specs = registry.specs()
    messages = list(history)
    for _ in range(cfg.max_tool_iters):
        result = await chat_provider.complete(messages, model=cfg.chat_model, tools=specs)
        if not result.tool_calls:
            return result.content or ""
        log.info("tool turn: calls=%s", ",".join(tc.name for tc in result.tool_calls))
        messages.append(_assistant_tool_message(result))
        for tc in result.tool_calls:
            out = await registry.dispatch(
                tc.name, tc.arguments, peer=peer, timeout=cfg.tool_timeout_s
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            await _charge(drive_store, peer, tc.name)
    log.warning("tool loop hit iteration cap (%d)", cfg.max_tool_iters)
    return TOOL_CAP_MESSAGE


async def _charge(drive_store: "DriveState | None", peer: str, action: str) -> None:
    """Best-effort debit of a costed verb's drive currency. Never raises — a spend
    fault must not look like a failed turn (a tool that already ran shouldn't be
    undone by a bookkeeping hiccup). Charges on dispatch-attempt; a tool that
    failed still debits (rare, and the earn-back pass reconciles by outcome)."""
    if drive_store is None:
        return
    costs = cost_for(action)
    if not costs:
        return
    try:
        await drive_store.spend(peer, action, costs)
    except Exception as e:  # type-only — the provider/tool surface may carry text
        log.error("spend failed action=%s exc=%s", action, type(e).__name__)
