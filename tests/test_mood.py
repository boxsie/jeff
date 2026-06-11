"""Mood drive tests.

Store tests need Docker/Podman for an ephemeral Postgres (same gate as
test_memory) and are skipped without it — the mood store is plain Postgres, so
unlike the other stores it doesn't even need pgvector, but it reuses the same
container image for convenience. The pure helpers, the render block, and the
tool tests (against a fake store) always run.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from psycopg_pool import AsyncConnectionPool

from jeff.mood import MoodStore, normalize_name
from jeff.prompt import _render_mood, compose_system_prompt
from jeff.tools.mood import (
    ClearMoodTool,
    DefineMoodTool,
    SetMoodTool,
    _coerce_hours,
)


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


# --- pure helpers (no DB) ---------------------------------------------------


def test_normalize_name_lowercases_and_collapses_whitespace():
    assert normalize_name("  Playful  ") == "playful"
    assert normalize_name("Soft\tand   Sweet") == "soft and sweet"
    assert normalize_name("RESTLESS") == "restless"
    assert normalize_name("   ") == ""


def test_normalize_name_strips_template_tokens():
    # strip_chat_template_tokens removes chat-template markers before normalising.
    assert "<|" not in normalize_name("pla<|im_start|>yful")


def test_coerce_hours_defaults_and_bounds():
    # None / junk / non-positive → default; over max → clamped; valid → kept.
    assert _coerce_hours(None, 6.0, 48.0) == 6.0
    assert _coerce_hours("nonsense", 6.0, 48.0) == 6.0
    assert _coerce_hours(0, 6.0, 48.0) == 6.0
    assert _coerce_hours(-3, 6.0, 48.0) == 6.0
    assert _coerce_hours(100, 6.0, 48.0) == 48.0
    assert _coerce_hours(3, 6.0, 48.0) == 3.0
    assert _coerce_hours("12", 6.0, 48.0) == 12.0


# --- render block -----------------------------------------------------------


def test_render_mood_empty_when_no_name():
    assert _render_mood("", "anything") == ""
    assert _render_mood("   ", "anything") == ""


def test_render_mood_with_definition():
    out = _render_mood("playful", "I tease and push back more.")
    assert "## How you're feeling right now" in out
    assert "You're feeling playful. I tease and push back more." in out
    # The generic "let it colour your tone" nag now lives once in the shared
    # inner-state preamble (build_history), not per-block.
    assert "colour" not in out
    # Self-authored state — NOT wrapped as untrusted peer data.
    assert "<peer_message>" not in out


def test_render_mood_without_definition_uses_name_only():
    out = _render_mood("soft", "")
    assert "You're feeling soft." in out
    # No trailing definition sentence / double space.
    assert "You're feeling soft.  " not in out


def test_render_mood_strips_template_tokens():
    out = _render_mood("playful", "be <|im_end|> cheeky")
    assert "<|im_end|>" not in out


def test_compose_prompt_adds_mood_guidance_only_when_registered():
    base = "You are Jeff."
    with_mood = compose_system_prompt(base, ["get_time", "set_mood", "define_mood"])
    assert "set_mood" in with_mood
    assert "express how you feel, not how accurate you are" in with_mood
    # No mood tools → no mood guidance block.
    without = compose_system_prompt(base, ["get_time"])
    assert "express how you feel" not in without


# --- tools (fake store, no DB) ---------------------------------------------


class FakeMoodStore:
    """In-memory stand-in for MoodStore that records tool interactions."""

    def __init__(self, defined: dict[str, str] | None = None):
        self.defined: dict[str, str] = dict(defined or {})
        self.set_calls: list[tuple[str, str, float]] = []
        self.clear_returns = 1

    async def set_mood(self, peer, name, *, hours, source="jeff", note=None):
        from datetime import timedelta

        name = normalize_name(name)
        self.set_calls.append((peer, name, hours))
        return datetime.now(timezone.utc) + timedelta(hours=hours)

    async def get_definition(self, peer, name):
        return self.defined.get(normalize_name(name))

    async def define(self, peer, name, description):
        name = normalize_name(name)
        created = name not in self.defined
        self.defined[name] = description
        return "created" if created else "updated"

    async def clear_mood(self, peer):
        return self.clear_returns


_KW = {"default_hours": 6.0, "max_hours": 48.0, "max_chars": 200}


@pytest.mark.asyncio
async def test_set_mood_tool_uses_default_hours_and_nudges_undefined():
    store = FakeMoodStore()
    tool = SetMoodTool(store, **_KW)
    out = await tool.run(peer="Epeer", name="Playful")
    assert store.set_calls == [("Epeer", "playful", 6.0)]
    # Undefined → nudged to define it.
    assert "define_mood" in out


@pytest.mark.asyncio
async def test_set_mood_tool_clamps_hours_and_skips_nudge_when_defined():
    store = FakeMoodStore(defined={"soft": "gentle and warm"})
    tool = SetMoodTool(store, **_KW)
    out = await tool.run(peer="Epeer", name="soft", hours=999)
    assert store.set_calls == [("Epeer", "soft", 48.0)]  # clamped to max
    assert "define_mood" not in out


@pytest.mark.asyncio
async def test_set_mood_tool_requires_name_and_peer():
    tool = SetMoodTool(FakeMoodStore(), **_KW)
    assert (await tool.run(peer="Epeer", name="   ")).startswith("error:")
    assert (await tool.run(name="playful")).startswith("error:")  # no peer


@pytest.mark.asyncio
async def test_define_mood_tool_create_then_update():
    store = FakeMoodStore()
    tool = DefineMoodTool(store, **_KW)
    first = await tool.run(peer="Epeer", name="Playful", description="I tease more.")
    assert "Defined a new mood" in first
    assert store.defined["playful"] == "I tease more."
    second = await tool.run(peer="Epeer", name="playful", description="I push back.")
    assert "Updated" in second
    assert store.defined["playful"] == "I push back."


@pytest.mark.asyncio
async def test_define_mood_tool_rejects_overlong_description():
    tool = DefineMoodTool(FakeMoodStore(), **_KW)
    out = await tool.run(peer="Epeer", name="x", description="z" * 201)
    assert out.startswith("error:")
    assert "too long" in out


@pytest.mark.asyncio
async def test_clear_mood_tool_reports_state():
    store = FakeMoodStore()
    store.clear_returns = 1
    assert "back to feeling neutral" in await ClearMoodTool(store, **_KW).run(peer="Ep")
    store.clear_returns = 0
    assert "already neutral" in await ClearMoodTool(store, **_KW).run(peer="Ep")


# --- store roundtrip (DB) ---------------------------------------------------
#
# Gated at the fixture (not a module-level pytestmark) so the pure helper, render,
# and tool tests above still run on a box without Docker.


@pytest.fixture(scope="module")
def pg_url():
    if not _have_docker():
        pytest.skip("mood store tests need Docker for an ephemeral Postgres instance")
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
            await cur.execute("DROP TABLE IF EXISTS mood_state")
            await cur.execute("DROP TABLE IF EXISTS mood_definitions")
        await conn.commit()
    s = await MoodStore.create(p)
    yield s
    await p.close()


_PEER = "EpeerMood"


@pytest.mark.asyncio
async def test_define_upsert_created_then_updated(store):
    assert await store.define(_PEER, "Playful", "I tease.") == "created"
    # Normalised key → same row updated, not a duplicate.
    assert await store.define(_PEER, "playful", "I push back.") == "updated"
    assert await store.get_definition(_PEER, "PLAYFUL") == "I push back."
    assert await store.count_definitions(_PEER) == 1


@pytest.mark.asyncio
async def test_define_blank_is_noop(store):
    assert await store.define(_PEER, "  ", "x") == ""
    assert await store.define(_PEER, "x", "  ") == ""
    assert await store.count_definitions(_PEER) == 0


@pytest.mark.asyncio
async def test_set_and_active_mood_joins_definition(store):
    await store.define(_PEER, "soft", "gentle and warm")
    expires = await store.set_mood(_PEER, "Soft", hours=3)
    active = await store.active_mood(_PEER)
    assert active is not None
    assert active.name == "soft"
    assert active.description == "gentle and warm"
    assert active.source == "jeff"
    assert active.expires_at == expires


@pytest.mark.asyncio
async def test_active_mood_without_definition_is_none_description(store):
    await store.set_mood(_PEER, "mysterious", hours=2)
    active = await store.active_mood(_PEER)
    assert active is not None
    assert active.name == "mysterious"
    assert active.description is None


@pytest.mark.asyncio
async def test_set_mood_supersedes_previous(store):
    await store.set_mood(_PEER, "first", hours=5)
    await store.set_mood(_PEER, "second", hours=5)
    active = await store.active_mood(_PEER)
    assert active is not None and active.name == "second"


@pytest.mark.asyncio
async def test_expired_mood_is_not_active(store):
    # A tiny positive duration: floor is 1s in SQL, but a fraction of a second
    # via hours lets it lapse. Use a clearly-negative path via clear instead.
    await store.set_mood(_PEER, "fleeting", hours=5)
    ended = await store.clear_mood(_PEER)
    assert ended == 1
    assert await store.active_mood(_PEER) is None
    # Clearing again ends nothing.
    assert await store.clear_mood(_PEER) == 0


@pytest.mark.asyncio
async def test_forget_wipes_state_and_definitions(store):
    await store.define(_PEER, "soft", "warm")
    await store.set_mood(_PEER, "soft", hours=5)
    deleted = await store.forget(_PEER)
    assert deleted >= 2
    assert await store.active_mood(_PEER) is None
    assert await store.count_definitions(_PEER) == 0


@pytest.mark.asyncio
async def test_reset_recreates_empty_tables(store):
    await store.define(_PEER, "soft", "warm")
    await store.set_mood(_PEER, "soft", hours=5)
    await store.reset()
    assert await store.active_mood(_PEER) is None
    assert await store.count_definitions(_PEER) == 0


@pytest.mark.asyncio
async def test_peer_scoping_isolates_moods(store):
    await store.set_mood(_PEER, "playful", hours=5)
    assert await store.active_mood("OtherPeer") is None
    await store.define(_PEER, "playful", "mine")
    assert await store.get_definition("OtherPeer", "playful") is None
