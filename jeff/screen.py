"""Inbound-message screening for Jeff.

A separate module so the size-cap decision can be unit-tested without
spinning up the daemon-event loop. Right now it only checks length; future
content filters (e.g. drop empty / whitespace-only / control-char messages)
can land here without touching main.py.

The byte budget is measured against the *UTF-8 encoding* of the text, not
character count. A 1000-character message of CJK glyphs is ~3 KB; capping
on chars would let it sneak past the embedding/storage cap. UTF-8 is what
hits Ollama and Postgres, so UTF-8 is what we measure.
"""

from __future__ import annotations

import re


# Gemma3 (and most chat-template LLMs) wrap turns with control tokens.
# An allowlisted peer who sends raw "<start_of_turn>model\n..." can otherwise
# embed a forged assistant turn that the chat template renders as a real one.
# We strip these tokens both on insert (so memory never holds them) and on
# recall (defense in depth in case a row predates this fix). W3 #dc9acd3c.
#
# Pattern is intentionally case-insensitive and matches both "<bos>"-style
# and the Gemma "<start_of_turn>"/"<end_of_turn>" forms. Adding a new
# template family is a one-line addition.
_CHAT_TEMPLATE_TOKEN_RE = re.compile(
    r"<(?:bos|eos|start_of_turn|end_of_turn|s|/s|im_start|im_end|"
    r"\|im_start\||\|im_end\||\|system\||\|user\||\|assistant\|)>",
    flags=re.IGNORECASE,
)


def strip_chat_template_tokens(text: str) -> str:
    """Remove chat-template control tokens from text.

    Returns the input unchanged if no tokens match. Multi-occurrence safe.
    """
    return _CHAT_TEMPLATE_TOKEN_RE.sub("", text)


def screen_text(text: str, max_bytes: int) -> str | None:
    """Decide whether to accept an inbound message.

    Returns `None` if the message is acceptable, or a user-facing rejection
    string if it should be refused. The rejection string is suitable to
    send back as a reply — the caller is responsible for the actual send.

    `max_bytes` is the UTF-8 byte budget. Set <= 0 to disable the cap
    (returns `None` for any input).
    """
    if max_bytes <= 0:
        return None
    size = len(text.encode("utf-8"))
    if size <= max_bytes:
        return None
    return (
        f"Sorry — your message was {size} bytes; the cap is {max_bytes}. "
        "Please shorten and try again."
    )
