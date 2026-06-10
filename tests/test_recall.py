"""Pure tests for the memory-scan tools (recall_memory / summarize_recent).

No DB: a fake Memory returns canned Message rows and a fake provider returns a
canned summary, so these pin the tool contracts (formatting, empty cases, error
safety) without exercising pgvector — the recall/recent SQL is covered by the
Memory store's own DB-gated tests.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jeff.llm import LLMError
from jeff.memory import Message
from jeff.tools.recall import RecallMemoryTool, SummarizeRecentTool


_TS = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)


def _msg(role: str, content: str, mid: int = 1) -> Message:
    return Message(id=mid, peer="EpeerD", role=role, content=content, ts=_TS)


class FakeMemory:
    def __init__(self, *, recall_rows=None, recent_rows=None):
        self._recall = recall_rows or []
        self._recent = recent_rows or []
        self.recall_calls: list[tuple] = []
        self.recent_calls: list[tuple] = []

    async def recall(self, peer, query, k, **kw):
        self.recall_calls.append((peer, query, k))
        return self._recall

    async def recent(self, peer, n=10):
        self.recent_calls.append((peer, n))
        return self._recent


class FakeChatProvider:
    def __init__(self, reply="the gist of it", raise_llm=False):
        self.reply = reply
        self.raise_llm = raise_llm
        self.calls: list[list[dict]] = []

    async def chat(self, messages, *, model):
        self.calls.append(list(messages))
        if self.raise_llm:
            raise LLMError("upstream body that must not leak")
        return self.reply


# --- recall_memory ----------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_formats_snippets_and_labels_roles():
    mem = FakeMemory(
        recall_rows=[_msg("user", "I love sci-fi", 1), _msg("assistant", "noted", 2)]
    )
    tool = RecallMemoryTool(mem, k=5, max_chars=1000)
    out = await tool.run(peer="EpeerD", query="hobbies")
    assert mem.recall_calls == [("EpeerD", "hobbies", 5)]
    assert "From your memory:" in out
    assert "- them: I love sci-fi" in out
    assert "- you: noted" in out


@pytest.mark.asyncio
async def test_recall_empty_is_graceful():
    tool = RecallMemoryTool(FakeMemory(recall_rows=[]), k=5, max_chars=1000)
    out = await tool.run(peer="EpeerD", query="hobbies")
    assert "Nothing relevant" in out


@pytest.mark.asyncio
async def test_recall_requires_peer_and_query():
    tool = RecallMemoryTool(FakeMemory(), k=5, max_chars=1000)
    assert "error:" in await tool.run(query="x")  # no peer
    assert "error:" in await tool.run(peer="EpeerD", query="   ")  # blank query


@pytest.mark.asyncio
async def test_recall_respects_char_cap():
    rows = [_msg("user", "x" * 500, i) for i in range(5)]
    tool = RecallMemoryTool(FakeMemory(recall_rows=rows), k=5, max_chars=600)
    out = await tool.run(peer="EpeerD", query="q")
    # Header + at most one 500-char snippet fits under the cap; not all five.
    assert out.count("- them:") <= 2


# --- summarize_recent -------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_returns_provider_summary():
    mem = FakeMemory(recent_rows=[_msg("user", "hi", 1), _msg("assistant", "hey", 2)])
    prov = FakeChatProvider(reply="You greeted each other.")
    tool = SummarizeRecentTool(mem, prov, model="m", span=30, max_chars=800)
    out = await tool.run(peer="EpeerD")
    assert out == "You greeted each other."
    assert mem.recent_calls == [("EpeerD", 30)]
    # The transcript was handed to the provider wrapped as untrusted content.
    assert "<recent_exchange>" in prov.calls[0][1]["content"]


@pytest.mark.asyncio
async def test_summarize_no_recent_is_graceful():
    tool = SummarizeRecentTool(
        FakeMemory(recent_rows=[]), FakeChatProvider(), model="m", span=30, max_chars=800
    )
    assert "No recent conversation" in await tool.run(peer="EpeerD")


@pytest.mark.asyncio
async def test_summarize_provider_error_is_safe():
    mem = FakeMemory(recent_rows=[_msg("user", "hi", 1)])
    prov = FakeChatProvider(raise_llm=True)
    tool = SummarizeRecentTool(mem, prov, model="m", span=30, max_chars=800)
    out = await tool.run(peer="EpeerD")
    # Generic line, nothing from the exception body.
    assert "Couldn't pull a summary" in out
    assert "upstream body" not in out


@pytest.mark.asyncio
async def test_summarize_requires_peer():
    tool = SummarizeRecentTool(
        FakeMemory(), FakeChatProvider(), model="m", span=30, max_chars=800
    )
    assert "error:" in await tool.run()
