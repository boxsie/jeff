"""Canonical chat/tool-call shapes shared by every ChatProvider.

These live in their own module (rather than `llm.py`) so both `llm.XaiProvider`
and `ollama.Ollama` can import them without a circular import — `llm` imports
`Ollama`, so `ollama` can't import back from `llm`.

The canonical form is OpenAI/xAI-shaped, because that's what Grok (the prod
provider) speaks natively: the turn loop builds assistant `tool_calls` /
`role:"tool"` messages in this shape and hands them straight to Grok. Providers
that speak a different dialect (Ollama) translate at their own boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolCall:
    """A model's request to invoke one tool (OpenAI canonical shape).

    `arguments` is the raw JSON string the model emitted — kept as a string
    rather than parsed because (a) that's exactly what xAI/OpenAI return and
    expect echoed back in the assistant turn, and (b) parsing/validation is the
    registry's job at dispatch time. `id` correlates the call with its result
    in the follow-up `role:"tool"` message; providers that don't emit one (e.g.
    Ollama) get a synthesised `call_<n>` id so the loop is uniform.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatResult:
    """One provider response: either final content, or a batch of tool calls.

    A model turn is one or the other in practice — when `tool_calls` is
    non-empty the loop executes them and calls again; otherwise `content` is the
    final answer. `content` may be present alongside tool calls (some models
    narrate), but the loop treats tool calls as authoritative when present.
    """

    content: str | None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)


def parse_openai_tool_calls(raw) -> tuple[ToolCall, ...]:
    """Parse an OpenAI/xAI `message.tool_calls` array into canonical ToolCalls.

    Defensive against shape drift (the body is upstream-controlled): entries
    missing a function name are skipped; a missing `id` is synthesised; a
    non-string `arguments` (some providers emit an object) is re-serialised to a
    JSON string so the canonical form is always a string.
    """
    if not isinstance(raw, list):
        return ()
    out: list[ToolCall] = []
    for i, call in enumerate(raw):
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args) if args is not None else "{}"
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{i}"
        out.append(ToolCall(id=call_id, name=name, arguments=args))
    return tuple(out)
