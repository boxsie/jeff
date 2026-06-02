"""Built-in tools that need no external service.

`get_time` exists mainly to exercise the execute-and-loop end-to-end with a
zero-dependency tool (the foundation ticket's "trivial built-in"), but it's
genuinely useful — the model has no clock otherwise, and an assistant that
confidently states the wrong date is a real failure mode.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar

from .base import Tool


class GetTimeTool(Tool):
    """Return the current UTC date and time."""

    name: ClassVar[str] = "get_time"
    description: ClassVar[str] = (
        "Get the current date and time in UTC (ISO-8601). Use this whenever the "
        "user asks what the time or date is, or when you need to reason about "
        "'now' — do not guess."
    )
    parameters: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}

    async def run(self, **kwargs) -> str:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        return f"Current UTC time: {now.isoformat()}"
