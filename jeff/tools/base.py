"""Tool abstraction + registry/dispatch for Jeff's execute-and-loop turn.

A `Tool` is the unit the LLM can invoke: a name, a description, a JSON-schema
for its parameters, and an `async run(**kwargs) -> str`. The `ToolRegistry`
turns a set of tools into (a) the provider-facing JSON specs and (b) a
name→executor dispatch that is **safe by construction**:

- args that don't parse, an unknown tool, missing required params, a tool that
  raises, or a tool that overruns its timeout all resolve to a short, content-
  safe string fed back to the model — never an exception that crashes the turn,
  and never raw internal/exception text (mirrors `_http.safe_excerpt`
  discipline). The model sees "error: …" and can recover or apologise.

Single-user threat model (ACL = operator only) means we don't sandbox tools
against the *caller*; the discipline here is about not leaking internals and not
letting a tool fault take down the turn loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import ClassVar


log = logging.getLogger("jeff.tools")

# Default ceiling on a single tool's run time. The turn is already bounded by
# the dispatcher; this stops one wedged tool (e.g. a hung HTTP call) from eating
# the whole turn budget. Overridable per-dispatch.
DEFAULT_TOOL_TIMEOUT_S = 30.0


class Tool(ABC):
    """Base class for an LLM-invocable tool.

    Subclasses set the three class attributes and implement `run`. `parameters`
    is a JSON-schema object (`{"type": "object", "properties": {...},
    "required": [...]}`) describing the kwargs `run` accepts.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict]

    @abstractmethod
    async def run(self, **kwargs) -> str:
        """Execute the tool and return a content-safe string for the model."""
        raise NotImplementedError

    def spec(self) -> dict:
        """The OpenAI/xAI function-tool spec for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Holds the active tools and dispatches calls to them safely."""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if not getattr(tool, "name", None):
            raise ValueError("tool must have a non-empty name")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict]:
        """JSON-schema function specs to hand the provider (stable order)."""
        return [self._tools[n].spec() for n in self.names()]

    async def dispatch(
        self,
        name: str,
        arguments: str,
        *,
        timeout: float = DEFAULT_TOOL_TIMEOUT_S,
    ) -> str:
        """Run tool `name` with JSON-string `arguments`; return a safe string.

        Every failure mode (unknown tool, bad JSON, bad args, raise, timeout)
        becomes a short `error: …` string the model can read. Nothing from an
        exception body reaches the model — only the tool name and a category.
        """
        tool = self._tools.get(name)
        if tool is None:
            log.info("tool dispatch: unknown tool=%s", name)
            return f"error: unknown tool {name!r}"

        try:
            parsed = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            log.info("tool dispatch: bad-args tool=%s", name)
            return f"error: tool {name!r} received malformed JSON arguments"
        if not isinstance(parsed, dict):
            return f"error: tool {name!r} arguments must be a JSON object"

        valid, kwargs_or_msg = _validate_args(tool.parameters, parsed)
        if not valid:
            log.info("tool dispatch: invalid-args tool=%s", name)
            return f"error: tool {name!r} {kwargs_or_msg}"

        try:
            result = await asyncio.wait_for(tool.run(**kwargs_or_msg), timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("tool %s timed out after %.0fs", name, timeout)
            return f"error: tool {name!r} timed out"
        except Exception:
            # Never surface the exception text (it may carry endpoint detail or
            # peer-shaped content). Log the type only, mirror the turn-loop
            # discipline of one operator-readable line with no PII.
            log.exception("tool %s raised", name)
            return f"error: tool {name!r} failed"

        if not isinstance(result, str):
            return f"error: tool {name!r} returned a non-string result"
        return result


def _validate_args(schema: dict, args: dict) -> tuple[bool, object]:
    """Lightweight JSON-schema check: required keys present, drop unknown keys.

    Returns `(True, kwargs)` with kwargs filtered to declared properties, or
    `(False, message)`. This is deliberately not a full validator — it guards
    the two failure modes that actually break `run(**kwargs)`: a missing
    required parameter (TypeError) and an unexpected keyword (TypeError). Type
    coercion is left to the tool, which knows its own contract.
    """
    props = schema.get("properties") if isinstance(schema, dict) else None
    props = props if isinstance(props, dict) else {}
    required = schema.get("required") if isinstance(schema, dict) else None
    required = required if isinstance(required, list) else []

    missing = [r for r in required if r not in args]
    if missing:
        return False, f"missing required argument(s): {', '.join(missing)}"

    # Drop keys the tool doesn't declare so an over-eager model can't trip
    # run(**kwargs) with an unexpected keyword.
    kwargs = {k: v for k, v in args.items() if k in props}
    return True, kwargs
