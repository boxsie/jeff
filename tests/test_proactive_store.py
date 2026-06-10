"""Docker-gated tests for ProactiveStore against a real Postgres (podman OK).

Mirrors the appraisal store test harness: an ephemeral pgvector container, a
fresh `proactive_state` table per test. Skipped without Docker/podman.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest
from psycopg_pool import AsyncConnectionPool

from jeff.proactive import ProactiveStore


_PG_IMAGE = os.environ.get("JEFF_TEST_PG_IMAGE", "pgvector/pgvector:pg16")
_NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)


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


pytestmark_db = pytest.mark.skipif(
    not _have_docker(),
    reason="proactive store tests need Docker for an ephemeral Postgres instance",
)


@pytest.fixture(scope="module")
def pg_url():
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(_PG_IMAGE) as pg:
        url = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
        yield url


@pytest.fixture
async def pool(pg_url):
    p = AsyncConnectionPool(pg_url, min_size=1, max_size=2, open=False)
    await p.open()
    async with p.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DROP TABLE IF EXISTS proactive_state")
        await conn.commit()
    yield p
    await p.close()


@pytestmark_db
@pytest.mark.asyncio
async def test_get_state_defaults_when_absent(pool):
    store = await ProactiveStore.create(pool)
    st = await store.get_state("EpeerD")
    assert st.peer == "EpeerD"
    assert st.last_send_at is None
    assert st.last_nudge_key is None
    assert st.muted_until is None


@pytestmark_db
@pytest.mark.asyncio
async def test_record_send_stamps_time_and_key(pool):
    store = await ProactiveStore.create(pool)
    await store.record_send("EpeerD", "curiosity:abc", _NOW)
    st = await store.get_state("EpeerD")
    assert st.last_send_at == _NOW
    assert st.last_nudge_key == "curiosity:abc"
    # A second send updates both in place (upsert, not a new row).
    later = _NOW + timedelta(hours=2)
    await store.record_send("EpeerD", "curiosity:xyz", later)
    st = await store.get_state("EpeerD")
    assert st.last_send_at == later
    assert st.last_nudge_key == "curiosity:xyz"


@pytestmark_db
@pytest.mark.asyncio
async def test_set_and_clear_mute(pool):
    store = await ProactiveStore.create(pool)
    until = _NOW + timedelta(hours=8)
    await store.set_mute("EpeerD", until)
    assert (await store.get_state("EpeerD")).muted_until == until
    # Clearing the mute leaves the row but nulls the window.
    await store.set_mute("EpeerD", None)
    assert (await store.get_state("EpeerD")).muted_until is None


@pytestmark_db
@pytest.mark.asyncio
async def test_mute_and_send_coexist_on_one_row(pool):
    store = await ProactiveStore.create(pool)
    await store.record_send("EpeerD", "k", _NOW)
    await store.set_mute("EpeerD", _NOW + timedelta(hours=1))
    st = await store.get_state("EpeerD")
    # Both upserts target the same peer PK without clobbering each other.
    assert st.last_nudge_key == "k"
    assert st.muted_until == _NOW + timedelta(hours=1)


@pytestmark_db
@pytest.mark.asyncio
async def test_forget_wipes_one_peer(pool):
    store = await ProactiveStore.create(pool)
    await store.record_send("EpeerD", "k", _NOW)
    await store.record_send("EotherD", "k", _NOW)
    deleted = await store.forget("EpeerD")
    assert deleted == 1
    assert (await store.get_state("EpeerD")).last_send_at is None
    assert (await store.get_state("EotherD")).last_send_at == _NOW


@pytestmark_db
@pytest.mark.asyncio
async def test_reset_wipes_and_recreates(pool):
    store = await ProactiveStore.create(pool)
    await store.record_send("EpeerD", "k", _NOW)
    await store.reset()
    assert (await store.get_state("EpeerD")).last_send_at is None
    # Usable after reset.
    await store.record_send("EpeerD", "k2", _NOW)
    assert (await store.get_state("EpeerD")).last_nudge_key == "k2"


@pytestmark_db
@pytest.mark.asyncio
async def test_record_send_persists_asked_curiosity_ids(pool):
    store = await ProactiveStore.create(pool)
    await store.record_send("EpeerD", "k", _NOW, [10, 11, 12])
    # take returns them, then clears (consume-once).
    assert await store.take_asked_curiosity_ids("EpeerD") == [10, 11, 12]
    assert await store.take_asked_curiosity_ids("EpeerD") == []


@pytestmark_db
@pytest.mark.asyncio
async def test_take_asked_curiosity_ids_empty_and_none(pool):
    store = await ProactiveStore.create(pool)
    # Never sent → no row → empty.
    assert await store.take_asked_curiosity_ids("EpeerD") == []
    # A send with no asked ids stores NULL, not an empty array we'd re-hand-out.
    await store.record_send("EpeerD", "k", _NOW)
    assert await store.take_asked_curiosity_ids("EpeerD") == []


@pytestmark_db
@pytest.mark.asyncio
async def test_record_send_overwrites_prior_asked_ids(pool):
    store = await ProactiveStore.create(pool)
    await store.record_send("EpeerD", "k1", _NOW, [1, 2])
    # A fresh reach-out replaces the pending set (not appends).
    await store.record_send("EpeerD", "k2", _NOW + timedelta(hours=2), [9])
    assert await store.take_asked_curiosity_ids("EpeerD") == [9]


@pytestmark_db
@pytest.mark.asyncio
async def test_forget_clears_asked_ids(pool):
    store = await ProactiveStore.create(pool)
    await store.record_send("EpeerD", "k", _NOW, [5])
    await store.forget("EpeerD")
    assert await store.take_asked_curiosity_ids("EpeerD") == []


@pytestmark_db
@pytest.mark.asyncio
async def test_create_migrates_preexisting_table_without_asked_column(pool):
    """The deployed proactive_state predates last_asked_curiosity_ids; create()
    must ADD COLUMN IF NOT EXISTS so the column appears without a reset-memory."""
    # Hand-build the OLD table shape (no last_asked_curiosity_ids column).
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DROP TABLE IF EXISTS proactive_state")
            await cur.execute(
                "CREATE TABLE proactive_state ("
                "peer TEXT PRIMARY KEY, last_send_at TIMESTAMPTZ, "
                "last_nudge_key TEXT, muted_until TIMESTAMPTZ)"
            )
        await conn.commit()
    # create() runs the migration; the column now exists and round-trips.
    store = await ProactiveStore.create(pool)
    await store.record_send("EpeerD", "k", _NOW, [7, 8])
    assert await store.take_asked_curiosity_ids("EpeerD") == [7, 8]
