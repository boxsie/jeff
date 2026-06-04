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
async def test_clear_cutoff_scopes_both_recent_and_recall(pool):
    """`/clear` semantics: set_history_cutoff starts a fresh session — it drops
    older rows from BOTH the recent window and semantic recall, so pre-clear
    lines can't bleed back in via topical similarity."""
    emb = FakeEmbedder()
    mem = await Memory.create(pool, emb, embed_model="fake", embed_dim=FakeEmbedder.DIM)

    # Pre-cutoff content the operator wants out of the active session.
    await mem.remember("EabcD", "user", "I love cycling")
    await mem.set_history_cutoff("EabcD")
    # Post-cutoff content (the new conversation).
    await mem.remember("EabcD", "user", "espresso every morning")

    recent = await mem.recent("EabcD", n=10)
    assert [m.content for m in recent] == ["espresso every morning"]

    # recall() now honours the cutoff: a query about the pre-clear topic finds
    # nothing from before the cutoff (the row still exists, it's just out of
    # this session's reach).
    hits = await mem.recall("EabcD", "tell me about my bike", k=5)
    assert all("cycling" not in m.content.lower() for m in hits)


@pytest.mark.asyncio
async def test_recall_finds_post_cutoff_rows(pool):
    """The cutoff scopes recall to the current session, but still finds rows
    created after the cutoff — a fresh session has working semantic memory."""
    emb = FakeEmbedder()
    mem = await Memory.create(pool, emb, embed_model="fake", embed_dim=FakeEmbedder.DIM)

    await mem.set_history_cutoff("EabcD")
    await mem.remember("EabcD", "user", "I love cycling")

    hits = await mem.recall("EabcD", "tell me about my bike", k=5)
    assert any("cycling" in m.content.lower() for m in hits)


@pytest.mark.asyncio
async def test_forget_hard_deletes_and_clears_watermark(pool):
    emb = FakeEmbedder()
    mem = await Memory.create(pool, emb, embed_model="fake", embed_dim=FakeEmbedder.DIM)

    await mem.remember("EabcD", "user", "I love cycling")
    await mem.set_history_cutoff("EabcD")
    await mem.remember("EabcD", "user", "espresso every morning")
    await mem.remember("EotherD", "user", "weather today?")

    deleted = await mem.forget("EabcD")
    assert deleted == 2
    assert await mem.count("EabcD") == 0
    # Other peers are untouched.
    assert await mem.count("EotherD") == 1
    assert await mem.total() == 1

    # Watermark was cleared: a fresh row for EabcD is immediately visible in
    # recent() (the stale cutoff didn't survive the wipe).
    await mem.remember("EabcD", "user", "back again")
    recent = await mem.recent("EabcD", n=10)
    assert [m.content for m in recent] == ["back again"]


@pytest.mark.asyncio
async def test_count_and_total(pool):
    emb = FakeEmbedder()
    mem = await Memory.create(pool, emb, embed_model="fake", embed_dim=FakeEmbedder.DIM)

    await mem.remember("EabcD", "user", "one")
    await mem.remember("EabcD", "assistant", "two")
    await mem.remember("EotherD", "user", "three")

    assert await mem.count("EabcD") == 2
    assert await mem.count("EotherD") == 1
    assert await mem.count("Enobody") == 0
    assert await mem.total() == 3


@pytest.mark.asyncio
async def test_remember_rejects_dim_mismatch(pool):
    class BadEmbedder:
        async def embed(self, text, *, model):
            return [0.0] * 7  # wrong dim

    mem = await Memory.create(pool, FakeEmbedder(), embed_model="fake", embed_dim=4)
    mem._embedder = BadEmbedder()  # swap after schema creation
    with pytest.raises(ValueError, match="dim mismatch"):
        await mem.remember("Epeer", "user", "hello")
