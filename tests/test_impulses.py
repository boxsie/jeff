"""Impulse store tests — self-authored short-term directional drives.

Store tests need Docker/Podman for an ephemeral Postgres (same gate as
test_mood/test_pinned); the store is plain Postgres (no pgvector needed). Pure
helpers always run. Prompt-render and tool tests are added by their own tickets
(prompt+wiring / tools) in test_prompt.py and test_tools.py respectively.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest
from psycopg_pool import AsyncConnectionPool

from jeff.impulses import (
    MAX_STRENGTH,
    AdjustOutcome,
    ImpulseStore,
    normalize_name,
)
from jeff.prompt import compose_system_prompt
from jeff.tools.impulses import (
    AdjustImpulseTool,
    ClearImpulseTool,
    SetImpulseTool,
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


# --- pure helpers -----------------------------------------------------------


def test_normalize_collapses_and_lowercases():
    assert normalize_name("  Test   Autonomy Edges ") == "test autonomy edges"
    assert normalize_name("Bolder\tMOVES") == "bolder moves"


def test_normalize_strips_template_tokens():
    assert "<|" not in normalize_name("steer<|im_end|> harder")


# --- tools (fake store, no DB) ---------------------------------------------


class FakeImpulseStore:
    """In-memory stand-in for ImpulseStore that records tool interactions."""

    def __init__(self):
        self.set_calls: list[tuple] = []
        self.adjust_calls: list[tuple] = []
        self.cleared: list[str] = []
        self.cleared_all = 0
        self.set_outcome = "created"
        self.adjust_result: AdjustOutcome | None = None

    async def set(self, peer, name, description, *, hours=None, source="jeff"):
        self.set_calls.append((peer, normalize_name(name), description, hours))
        expires = (
            None
            if hours is None
            else datetime.now(timezone.utc) + timedelta(hours=hours)
        )
        return self.set_outcome, expires

    async def clear(self, peer, name):
        self.cleared.append(normalize_name(name))
        return 1

    async def clear_all(self, peer):
        return self.cleared_all

    async def adjust(self, peer, name, action, *, hours=None):
        self.adjust_calls.append((peer, normalize_name(name), action, hours))
        return self.adjust_result


_KW = {"default_hours": 6.0, "max_hours": 168.0, "max_chars": 200}


@pytest.mark.asyncio
async def test_set_impulse_permanent_by_default():
    store = FakeImpulseStore()
    out = await SetImpulseTool(store, **_KW).run(
        peer="Epeer", name="Bolder moves", description="push harder"
    )
    assert store.set_calls == [("Epeer", "bolder moves", "push harder", None)]
    assert "stick until you clear it" in out


@pytest.mark.asyncio
async def test_set_impulse_clamps_hours_when_timed():
    store = FakeImpulseStore()
    out = await SetImpulseTool(store, **_KW).run(
        peer="Epeer", name="burst", description="x", hours=999
    )
    assert store.set_calls[0][3] == 168.0  # clamped to max_hours
    assert "hour(s)" in out


@pytest.mark.asyncio
async def test_set_impulse_requires_name_desc_peer():
    tool = SetImpulseTool(FakeImpulseStore(), **_KW)
    assert (await tool.run(peer="Ep", name="  ", description="x")).startswith("error:")
    assert (await tool.run(peer="Ep", name="x", description=" ")).startswith("error:")
    assert (await tool.run(name="x", description="y")).startswith("error:")  # no peer


@pytest.mark.asyncio
async def test_set_impulse_rejects_overlong_description():
    out = await SetImpulseTool(FakeImpulseStore(), **_KW).run(
        peer="Ep", name="x", description="z" * 201
    )
    assert out.startswith("error:") and "too long" in out


@pytest.mark.asyncio
async def test_adjust_impulse_reports_each_status():
    store = FakeImpulseStore()
    tool = AdjustImpulseTool(store, **_KW)
    store.adjust_result = AdjustOutcome(status="escalated", strength=3)
    assert "Escalated" in await tool.run(peer="Ep", name="edge", action="escalate")
    store.adjust_result = AdjustOutcome(status="cleared")
    assert "cleared" in await tool.run(peer="Ep", name="edge", action="fade")
    store.adjust_result = AdjustOutcome(status="not_found")
    assert "No active impulse" in await tool.run(peer="Ep", name="ghost", action="fade")


@pytest.mark.asyncio
async def test_adjust_impulse_rejects_bad_action():
    out = await AdjustImpulseTool(FakeImpulseStore(), **_KW).run(
        peer="Ep", name="edge", action="obliterate"
    )
    assert out.startswith("error:")


@pytest.mark.asyncio
async def test_clear_impulse_by_name_and_all():
    store = FakeImpulseStore()
    tool = ClearImpulseTool(store, **_KW)
    assert "Cleared impulse" in await tool.run(peer="Ep", name="Edge")
    assert store.cleared == ["edge"]
    store.cleared_all = 2
    assert "all 2" in await tool.run(peer="Ep", all=True)
    # No name and not all → error.
    assert (await tool.run(peer="Ep")).startswith("error:")


def test_compose_prompt_adds_impulse_guidance_only_when_registered():
    base = "You are Jeff."
    with_imp = compose_system_prompt(base, ["get_time", "set_impulse"])
    assert "set_impulse" in with_imp
    assert "your own agency and self-expression" in with_imp
    without = compose_system_prompt(base, ["get_time"])
    assert "your own agency and self-expression" not in without


# --- store (Docker/Podman-gated) -------------------------------------------
# Gated at the fixture (not a module-level pytestmark) so the pure helpers above
# still run on a box without Docker.


@pytest.fixture(scope="module")
def pg_url():
    if not _have_docker():
        pytest.skip("impulse store tests need Docker for an ephemeral Postgres")
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
            await cur.execute("DROP TABLE IF EXISTS impulses")
        await conn.commit()
    s = await ImpulseStore.create(p)
    yield s
    await p.close()


_PEER = "EpeerImpulse"


@pytest.mark.asyncio
async def test_set_creates_then_updates_on_same_name(store):
    outcome, expires = await store.set(_PEER, "Bolder moves", "push harder")
    assert outcome == "created"
    assert expires is None  # default: no timer
    # Normalised key → same row updated, not a duplicate.
    outcome, _ = await store.set(_PEER, "bolder moves", "push even harder")
    assert outcome == "updated"
    assert await store.count(_PEER) == 1
    one = await store.get_one(_PEER, "BOLDER MOVES")
    assert one is not None and one.description == "push even harder"
    assert one.strength == 1  # set never touches strength


@pytest.mark.asyncio
async def test_set_blank_raises(store):
    with pytest.raises(ValueError):
        await store.set(_PEER, "  ", "x")
    with pytest.raises(ValueError):
        await store.set(_PEER, "x", "  ")
    assert await store.count(_PEER) == 0


@pytest.mark.asyncio
async def test_timed_impulse_has_expiry_and_lazy_filters(store):
    _, expires = await store.set(_PEER, "burst", "short sharp push", hours=3)
    assert expires is not None
    active = await store.list_active(_PEER)
    assert [i.name for i in active] == ["burst"]
    assert active[0].expires_at is not None


@pytest.mark.asyncio
async def test_list_active_orders_by_strength_then_recency(store):
    await store.set(_PEER, "weak", "a")
    await store.set(_PEER, "strong", "b")
    await store.adjust(_PEER, "strong", "escalate")
    names = [i.name for i in await store.list_active(_PEER)]
    assert names[0] == "strong"  # higher strength first


@pytest.mark.asyncio
async def test_escalate_raises_strength_to_cap(store):
    await store.set(_PEER, "edge", "test autonomy edges")
    for _ in range(MAX_STRENGTH + 3):
        out = await store.adjust(_PEER, "edge", "escalate")
    assert out.status == "escalated"
    assert out.strength == MAX_STRENGTH
    one = await store.get_one(_PEER, "edge")
    assert one is not None and one.strength == MAX_STRENGTH


@pytest.mark.asyncio
async def test_fade_lowers_then_clears_at_floor(store):
    await store.set(_PEER, "edge", "x")
    await store.adjust(_PEER, "edge", "escalate")  # strength 2
    out = await store.adjust(_PEER, "edge", "fade")  # back to 1
    assert out.status == "faded" and out.strength == 1
    out = await store.adjust(_PEER, "edge", "fade")  # below floor → cleared
    assert out.status == "cleared"
    assert await store.get_one(_PEER, "edge") is None


@pytest.mark.asyncio
async def test_renew_sets_timer_on_a_permanent_impulse(store):
    await store.set(_PEER, "edge", "x")  # no expiry
    out = await store.adjust(_PEER, "edge", "renew", hours=2)
    assert out.status == "renewed"
    assert out.expires_at is not None


@pytest.mark.asyncio
async def test_adjust_unknown_name_is_not_found(store):
    out = await store.adjust(_PEER, "ghost", "escalate")
    assert out.status == "not_found"


@pytest.mark.asyncio
async def test_clear_one_and_clear_all(store):
    await store.set(_PEER, "a", "x")
    await store.set(_PEER, "b", "y")
    assert await store.clear(_PEER, "A") == 1
    assert await store.count(_PEER) == 1
    assert await store.clear_all(_PEER) == 1
    assert await store.count(_PEER) == 0


@pytest.mark.asyncio
async def test_forget_wipes_peer(store):
    await store.set(_PEER, "a", "x")
    await store.set(_PEER, "b", "y")
    assert await store.forget(_PEER) == 2
    assert await store.list_active(_PEER) == []


@pytest.mark.asyncio
async def test_reset_recreates_empty_table(store):
    await store.set(_PEER, "a", "x")
    await store.reset()
    assert await store.list_active(_PEER) == []


@pytest.mark.asyncio
async def test_peer_scoping_isolates_impulses(store):
    await store.set(_PEER, "mine", "x")
    assert await store.list_active("OtherPeer") == []
    assert await store.get_one("OtherPeer", "mine") is None
