"""Memory roundtrip against an ephemeral pgvector Postgres.

Requires Docker + the `testcontainers` extra (`pip install -e .[dev]`). Tests
are skipped if Docker is unavailable so unit-test runs on dev machines without
docker still work.
"""

from __future__ import annotations

import os

import pytest
from psycopg_pool import AsyncConnectionPool

from jeff.memory import Memory


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


pytestmark = pytest.mark.skipif(
    not _have_docker(),
    reason="memory tests need Docker for an ephemeral pgvector instance",
)


class FakeEmbedder:
    """Deterministic 4-d embedder so we can assert on ordering.

    Maps a small set of phrases to fixed vectors; everything else gets a
    zero vector with a tiny offset so cosine similarity is well-defined.
    """

    DIM = 4

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def embed(self, text: str, *, model: str) -> list[float]:
        self.calls.append((text, model))
        t = text.lower()
        if "cycling" in t or "bike" in t:
            return [1.0, 0.0, 0.0, 0.0]
        if "coffee" in t or "espresso" in t:
            return [0.0, 1.0, 0.0, 0.0]
        if "weather" in t or "rain" in t:
            return [0.0, 0.0, 1.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0]


@pytest.fixture(scope="module")
def pg_url():
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(_PG_IMAGE) as pg:
        # testcontainers builds a SQLAlchemy URL; psycopg uses libpq form.
        url = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
        yield url


@pytest.fixture
async def pool(pg_url):
    p = AsyncConnectionPool(pg_url, min_size=1, max_size=2, open=False)
    await p.open()
    # Reset the table between tests so ordering assertions are stable.
    async with p.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DROP TABLE IF EXISTS messages")
        await conn.commit()
    yield p
    await p.close()


@pytest.mark.asyncio
async def test_remember_then_recent(pool):
    emb = FakeEmbedder()
    mem = await Memory.create(pool, emb, embed_model="fake", embed_dim=FakeEmbedder.DIM)

    await mem.remember("EabcD", "user", "I love cycling")
    await mem.remember("EabcD", "assistant", "Cycling is great")
    await mem.remember("EotherD", "user", "weather today?")

    recent = await mem.recent("EabcD", n=10)
    assert [m.content for m in recent] == ["I love cycling", "Cycling is great"]
    assert [m.role for m in recent] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_recall_orders_by_similarity(pool):
    emb = FakeEmbedder()
    mem = await Memory.create(pool, emb, embed_model="fake", embed_dim=FakeEmbedder.DIM)

    await mem.remember("EabcD", "user", "I love cycling")
    await mem.remember("EabcD", "user", "espresso every morning")
    await mem.remember("EabcD", "user", "rain forecast tomorrow")

    hits = await mem.recall("EabcD", "tell me about my bike", k=2)
    assert hits, "expected at least one hit"
    assert "cycling" in hits[0].content.lower()


@pytest.mark.asyncio
async def test_recall_scopes_by_peer(pool):
    emb = FakeEmbedder()
    mem = await Memory.create(pool, emb, embed_model="fake", embed_dim=FakeEmbedder.DIM)

    await mem.remember("EaliceD", "user", "I love cycling")
    await mem.remember("EbobD", "user", "I love cycling too")

    hits = await mem.recall("EaliceD", "cycling", k=5)
    assert all(m.peer == "EaliceD" for m in hits)
    assert len(hits) == 1


@pytest.mark.asyncio
async def test_remember_rejects_dim_mismatch(pool):
    class BadEmbedder:
        async def embed(self, text, *, model):
            return [0.0] * 7  # wrong dim

    mem = await Memory.create(pool, FakeEmbedder(), embed_model="fake", embed_dim=4)
    mem._embedder = BadEmbedder()  # swap after schema creation
    with pytest.raises(ValueError, match="dim mismatch"):
        await mem.remember("Epeer", "user", "hello")
