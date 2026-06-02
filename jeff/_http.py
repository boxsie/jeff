"""Shared HTTP hardening for LLM-provider HTTP clients.

Both the local Ollama client and the cloud (xAI/Grok) provider read response
bodies that are shaped by the peer's prompt — so the same two defences apply
to both:

- `safe_excerpt` truncates + strips control chars before an attacker-shaped
  body reaches an exception message / container log (log-injection defence).
- `read_bounded` refuses a response larger than a byte budget, so a hostile or
  replaced upstream can't ship a multi-GiB body and OOM-kill us.

These started life in `ollama.py`; they're factored here so every provider
shares one hardened implementation rather than copy-pasting it.
"""

from __future__ import annotations

import httpx


# Maximum chars of a response body to surface in an error. The body is
# attacker-influenced; never include the full thing in an exception message.
EXCERPT_LIMIT = 256


class ResponseTooLargeError(Exception):
    """Raised by `read_bounded` when a response body exceeds its byte budget."""


def safe_excerpt(s: str, limit: int = EXCERPT_LIMIT) -> str:
    """Sanitise an attacker-influenced string for inclusion in a log/exception.

    Drops control chars (0x00–0x1F and 0x7F) except plain space, replaces
    them with `?`, and truncates to `limit` chars with a `…(N more)` tail
    so the size info survives but the bytes don't.
    """
    cleaned = "".join(
        ch if (ch == " " or (ord(ch) >= 0x20 and ord(ch) != 0x7F)) else "?" for ch in s
    )
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}…({len(cleaned) - limit} more)"


async def read_bounded(resp: httpx.Response, max_bytes: int) -> bytes:
    """Read a streaming response, refusing more than `max_bytes` of body.

    Two layers: (1) if Content-Length is declared and exceeds the cap, fail
    before reading a single byte. (2) Otherwise stream chunks and abort the
    moment the running total crosses the cap. Both raise ResponseTooLargeError —
    callers don't need to special-case the source of the overflow.
    """
    cl = resp.headers.get("content-length")
    if cl is not None:
        try:
            declared = int(cl)
        except ValueError:
            declared = None
        if declared is not None and declared > max_bytes:
            raise ResponseTooLargeError(
                f"response Content-Length {declared} exceeds max {max_bytes}"
            )

    total = 0
    chunks: list[bytes] = []
    async for chunk in resp.aiter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise ResponseTooLargeError(f"response exceeded {max_bytes} bytes during stream")
        chunks.append(chunk)
    return b"".join(chunks)
