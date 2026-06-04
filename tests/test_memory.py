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
async def test_clear_resets_recent_thread_but_recall_spans_all_sessions(pool):
    """`/clear` semantics: set_history_cutoff resets the recent THREAD only.
    recall() still spans every session, so a query about a pre-clear topic can
    still surface it (long-term memory persists across /clear)."""
    emb = FakeEmbedder()
    mem = await Memory.create(pool, emb, embed_model="fake", embed_dim=FakeEmbedder.DIM)

    peer = "EthreadD"  # unique peer: the module pool is shared across tests
    await mem.remember(peer, "user", "I love cycling")  # pre-clear
    await mem.set_history_cutoff(peer)
    await mem.remember(peer, "user", "espresso every morning")  # post-clear

    # recent() is scoped to the fresh thread.
    recent = await mem.recent(peer, n=10)
    assert [m.content for m in recent] == ["espresso every morning"]

    # recall() spans all sessions — the pre-clear topic is still reachable.
    hits = await mem.recall(peer, "tell me about my bike", k=5)
    assert any("cycling" in m.content.lower() for m in hits)


@pytest.mark.asyncio
async def test_recall_scored_returns_distances_spanning_sessions(pool):
    """`/debug recall` view: mirrors recall() — spans all sessions (ignores the
    cutoff), ascends by distance, returns floats."""
    emb = FakeEmbedder()
    mem = await Memory.create(pool, emb, embed_model="fake", embed_dim=FakeEmbedder.DIM)

    peer = "EscoredD"
    await mem.remember(peer, "user", "I love cycling")  # pre-cutoff
    await mem.set_history_cutoff(peer)
    await mem.remember(peer, "user", "espresso every morning")  # post-cutoff

    scored = await mem.recall_scored(peer, "tell me about my bike", limit=8)

    contents = [m.content.lower() for m, _ in scored]
    # Spans the cutoff: the pre-cutoff "cycling" is still a candidate.
    assert any("cycling" in c for c in contents)
    dists = [d for _, d in scored]
    assert dists == sorted(dists)
    assert all(isinstance(d, float) for d in dists)


@pytest.mark.asyncio
async def test_get_history_cutoff_roundtrip(pool):
    emb = FakeEmbedder()
    mem = await Memory.create(pool, emb, embed_model="fake", embed_dim=FakeEmbedder.DIM)

    # Fresh peer — the module-scoped pool is shared, so reuse of "EabcD" would
    # already carry a cutoff from earlier tests.
    peer = "EnevercutD"
    assert await mem.get_history_cutoff(peer) is None
    await mem.set_history_cutoff(peer)
    assert await mem.get_history_cutoff(peer) is not None


@pytest.mark.asyncio
async def test_reset_wipes_all_peers_and_recreates(pool):
    emb = FakeEmbedder()
    mem = await Memory.create(pool, emb, embed_model="fake", embed_dim=FakeEmbedder.DIM)

    await mem.remember("EabcD", "user", "hello")
    await mem.remember("EotherD", "user", "hi")
    await mem.set_history_cutoff("EabcD")
    assert await mem.total() >= 2

    await mem.reset()

    # Everything gone, schema intact and usable.
    assert await mem.total() == 0
    assert await mem.get_history_cutoff("EabcD") is None
    await mem.remember("EabcD", "user", "starting over")
    assert await mem.count("EabcD") == 1


@pytest.mark.asyncio
async def test_init_schema_rejects_embed_dim_mismatch(pool):
    """Switching embed models (dim change) without a reset must fail loudly at
    startup, not silently break every insert later."""
    emb = FakeEmbedder()
    # Establish the table at the fake embedder's dim (idempotent if it exists).
    await Memory.create(pool, emb, embed_model="fake", embed_dim=FakeEmbedder.DIM)

    with pytest.raises(ValueError, match="reset-memory"):
        await Memory.create(
            pool, emb, embed_model="fake", embed_dim=FakeEmbedder.DIM + 4
        )


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
