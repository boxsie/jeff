"""Musings store tests — Jeff's carried idle thought.

Store tests need Docker/Podman for an ephemeral Postgres (same gate as
test_pinned); the store is plain Postgres (no pgvector). The render block + the
recency gate live in test_prompt.py and always run.
"""

from __future__ import annotations

import os

import pytest
from psycopg_pool import AsyncConnectionPool

from jeff.musings import MAX_ITEM_BYTES, MusingStore


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


@pytest.fixture(scope="module")
def pg_url():
    if not _have_docker():
        pytest.skip("musings store tests need Docker for an ephemeral Postgres")
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
            await cur.execute("DROP TABLE IF EXISTS musings")
        await conn.commit()
    s = await MusingStore.create(p)
    yield s
    await p.close()


_PEER = "EpeerMuse"


@pytest.mark.asyncio
async def test_record_then_latest(store):
    await store.record(_PEER, "wondering about the bike")
    m = await store.latest(_PEER)
    assert m is not None
    assert m.text == "wondering about the bike"
    assert m.created_at is not None


@pytest.mark.asyncio
async def test_record_is_last_write_wins(store):
    await store.record(_PEER, "first thought")
    await store.record(_PEER, "newer thought")
    m = await store.latest(_PEER)
    assert m.text == "newer thought"  # one row per peer, upserted


@pytest.mark.asyncio
async def test_record_blank_is_noop(store):
    await store.record(_PEER, "   ")
    assert await store.latest(_PEER) is None


@pytest.mark.asyncio
async def test_record_truncates_oversized(store):
    await store.record(_PEER, "x" * (MAX_ITEM_BYTES + 500))
    m = await store.latest(_PEER)
    assert len(m.text.encode("utf-8")) <= MAX_ITEM_BYTES


@pytest.mark.asyncio
async def test_latest_none_for_unknown_peer(store):
    assert await store.latest("ENobody") is None


@pytest.mark.asyncio
async def test_forget_wipes_musing(store):
    await store.record(_PEER, "a thought")
    deleted = await store.forget(_PEER)
    assert deleted == 1
    assert await store.latest(_PEER) is None
