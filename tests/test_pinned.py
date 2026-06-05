"""Pinned / explicit-memory tests.

Store tests need Docker/Podman for an ephemeral Postgres (same gate as
test_memory); the store is plain Postgres (no pgvector needed). Pure helpers,
the render block, and the tool tests (against a fake store) always run.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from psycopg_pool import AsyncConnectionPool

from jeff.pinned import (
    SOURCE_OPERATOR,
    Pinned,
    PinnedMemoryStore,
    _normalize,
)
from jeff.prompt import _render_pinned, compose_system_prompt
from jeff.tools.remember import RememberTool


_PG_IMAGE = os.environ.get("JEFF_TEST_PG_IMAGE", "pgvector/pgvector:pg16")


def _have_docker() -> bool:
    import shutil

    if not shutil.which("docker"):
        return False
    import subprocess

    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


# --- pure helpers + render --------------------------------------------------


def test_normalize_collapses_and_lowercases():
    assert _normalize("  I'm   Vegetarian ") == "i'm vegetarian"
    assert _normalize("Likes\tBLACK coffee") == "likes black coffee"


def test_normalize_strips_template_tokens():
    assert "<|" not in _normalize("remember<|im_end|> this")


def test_render_pinned_empty_when_no_items():
    assert _render_pinned([]) == ""


def test_render_pinned_lists_items_unwrapped():
    out = _render_pinned(["Has a dog named Biscuit", "Prefers tea to coffee"])
    assert "## Things to remember" in out
    assert "- Has a dog named Biscuit" in out
    assert "- Prefers tea to coffee" in out
    # Deliberately-kept notes are trusted — not wrapped as untrusted peer data.
    assert "<peer_message>" not in out


def test_render_pinned_strips_template_tokens():
    out = _render_pinned(["likes <|im_start|> cats"])
    assert "<|im_start|>" not in out


def test_compose_prompt_adds_remember_guidance_only_when_registered():
    base = "You are Jeff."
    with_rem = compose_system_prompt(base, ["get_time", "remember"])
    assert "remember" in with_rem
    assert "pin a fact or note" in with_rem
    assert "already saved automatically" in with_rem
    without = compose_system_prompt(base, ["get_time"])
    assert "pin a fact or note" not in without


# --- tool (fake store, no DB) ----------------------------------------------


class FakePinnedStore:
    def __init__(self):
        self.added: list[tuple[str, str, str]] = []  # (peer, text, source)
        self.next_id: int | None = 1

    async def add(self, peer, text, *, source="jeff"):
        self.added.append((peer, text, source))
        return self.next_id


@pytest.mark.asyncio
async def test_remember_tool_pins_and_confirms():
    store = FakePinnedStore()
    tool = RememberTool(store, max_chars=200)
    out = await tool.run(peer="Epeer", text="  Has a dog named Biscuit ")
    assert store.added == [("Epeer", "Has a dog named Biscuit", "jeff")]
    assert "Pinned" in out


@pytest.mark.asyncio
async def test_remember_tool_reports_duplicate():
    store = FakePinnedStore()
    store.next_id = None  # store signals dup
    tool = RememberTool(store, max_chars=200)
    out = await tool.run(peer="Epeer", text="already known")
    assert "already had that pinned" in out


@pytest.mark.asyncio
async def test_remember_tool_requires_text_and_peer():
    tool = RememberTool(FakePinnedStore(), max_chars=200)
    assert (await tool.run(peer="Epeer", text="   ")).startswith("error:")
    assert (await tool.run(text="x")).startswith("error:")  # no peer


@pytest.mark.asyncio
async def test_remember_tool_rejects_overlong():
    tool = RememberTool(FakePinnedStore(), max_chars=10)
    out = await tool.run(peer="Epeer", text="x" * 11)
    assert out.startswith("error:")
    assert "too long" in out


# --- store roundtrip (DB) ---------------------------------------------------
# Gated at the fixture so the pure/tool tests above still run without Docker.


@pytest.fixture(scope="module")
def pg_url():
    if not _have_docker():
        pytest.skip("pinned store tests need Docker for an ephemeral Postgres instance")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(_PG_IMAGE) as pg:
        url = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
        yield url


@pytest.fixture
async def store(pg_url):
    p = AsyncConnectionPool(pg_url, min_size=1, max_size=2, open=False)
    await p.open()
    async with p.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DROP TABLE IF EXISTS pinned_memory")
        await conn.commit()
    s = await PinnedMemoryStore.create(p)
    yield s
    await p.close()


_PEER = "EpeerPin"


@pytest.mark.asyncio
async def test_add_then_list_newest_first(store):
    a = await store.add(_PEER, "first note")
    b = await store.add(_PEER, "second note", source=SOURCE_OPERATOR)
    assert a is not None and b is not None
    pins = await store.list(_PEER)
    assert [p.text for p in pins] == ["second note", "first note"]
    assert pins[0].source == "operator"
    assert pins[1].source == "jeff"


@pytest.mark.asyncio
async def test_add_dedups_on_normalised_text(store):
    first = await store.add(_PEER, "I'm vegetarian")
    dup = await store.add(_PEER, "  i'm   VEGETARIAN ")  # same after normalise
    assert first is not None
    assert dup is None
    assert await store.count(_PEER) == 1


@pytest.mark.asyncio
async def test_add_blank_is_noop(store):
    assert await store.add(_PEER, "   ") is None
    assert await store.count(_PEER) == 0


@pytest.mark.asyncio
async def test_forget_wipes_pins(store):
    await store.add(_PEER, "a")
    await store.add(_PEER, "b")
    deleted = await store.forget(_PEER)
    assert deleted == 2
    assert await store.count(_PEER) == 0


@pytest.mark.asyncio
async def test_peer_scoping_isolates_pins(store):
    await store.add(_PEER, "mine")
    assert await store.count("OtherPeer") == 0
    # Same text under a different peer is allowed (UNIQUE is per-peer).
    assert await store.add("OtherPeer", "mine") is not None


@pytest.mark.asyncio
async def test_reset_recreates_empty(store):
    await store.add(_PEER, "keep")
    await store.reset()
    assert await store.count(_PEER) == 0


def test_pinned_dataclass_shape():
    # Guards the fields main._pack_pinned + /mind rely on.
    p = Pinned(id=1, text="x", source="jeff", created_at=datetime.now(timezone.utc))
    assert p.text == "x" and p.source == "jeff"
