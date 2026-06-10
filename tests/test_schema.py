"""Tests for the shared schema-drift guard (jeff/_schema.py).

Two layers, mirroring the store tests:
- Pure tests against a fake cursor (always run) — the raise/pass/no-op logic and
  the SchemaDriftError message, no DB needed.
- Docker-gated tests against a real Postgres (podman OK) — an OLD-shape table
  missing a column makes a store's create() fail loudly at startup, which is the
  whole point (catch drift at deploy, not mid-turn). Skipped without Docker.
"""

from __future__ import annotations

import os

import pytest
from psycopg_pool import AsyncConnectionPool

from jeff._schema import SchemaDriftError, assert_columns


# --------------------------------------------------------------------------- #
# Pure tests — fake cursor, no DB
# --------------------------------------------------------------------------- #


class _FakeCursor:
    """Minimal async cursor that answers the two queries the guard issues:
    `SELECT to_regclass(%s)` (→ fetchone) and the pg_attribute column list
    (→ fetchall). Routes by substring of the executed SQL."""

    def __init__(self, *, exists: bool, columns):
        self._exists = exists
        self._columns = list(columns)
        self._mode = None

    async def execute(self, query, params=None):
        q = str(query)
        # Order matters: the column-list query *also* contains to_regclass() in
        # its WHERE clause, so match the more specific pg_attribute marker first.
        if "pg_attribute" in q:
            self._mode = "attrs"
        elif "to_regclass" in q:
            self._mode = "regclass"
        else:
            self._mode = None

    async def fetchone(self):
        if self._mode == "regclass":
            return ["public.t" if self._exists else None]
        return None

    async def fetchall(self):
        if self._mode == "attrs":
            return [(c,) for c in self._columns]
        return []


@pytest.mark.asyncio
async def test_assert_columns_passes_when_all_present():
    cur = _FakeCursor(exists=True, columns={"id", "peer", "level"})
    # Superset on disk is fine; we only require the expected subset.
    await assert_columns(cur, "t", {"id", "peer"})


@pytest.mark.asyncio
async def test_assert_columns_raises_naming_missing():
    cur = _FakeCursor(exists=True, columns={"id", "peer"})
    with pytest.raises(SchemaDriftError) as ei:
        await assert_columns(cur, "drive_state", {"id", "peer", "level", "updated_at"})
    err = ei.value
    assert err.table == "drive_state"
    # Sorted, and only the genuinely-absent ones.
    assert err.missing == ["level", "updated_at"]
    msg = str(err)
    assert "drive_state" in msg
    assert "level" in msg and "updated_at" in msg
    # Points the reader at both fix paths.
    assert "ADD COLUMN IF NOT EXISTS" in msg
    assert "reset-memory" in msg


@pytest.mark.asyncio
async def test_assert_columns_noop_when_table_absent():
    # to_regclass NULL → the CREATE that runs before us owns that case; the guard
    # must not raise just because the table isn't there yet.
    cur = _FakeCursor(exists=False, columns=set())
    await assert_columns(cur, "t", {"id", "peer", "level"})


def test_schema_drift_error_dedups_and_sorts_missing():
    err = SchemaDriftError("t", ["b", "a", "b"])
    assert err.missing == ["a", "b"]


# --------------------------------------------------------------------------- #
# Docker-gated tests — real Postgres, real store
# --------------------------------------------------------------------------- #

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


pytestmark_db = pytest.mark.skipif(
    not _have_docker(),
    reason="schema-guard DB tests need Docker for an ephemeral Postgres instance",
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
    yield p
    await p.close()


@pytestmark_db
@pytest.mark.asyncio
async def test_guard_passes_against_live_table(pool):
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DROP TABLE IF EXISTS guard_demo")
            await cur.execute(
                "CREATE TABLE guard_demo (id BIGINT, peer TEXT, level DOUBLE PRECISION)"
            )
            await assert_columns(cur, "guard_demo", {"id", "peer", "level"})
        await conn.commit()


@pytestmark_db
@pytest.mark.asyncio
async def test_guard_raises_against_drifted_table(pool):
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DROP TABLE IF EXISTS guard_demo")
            await cur.execute("CREATE TABLE guard_demo (id BIGINT, peer TEXT)")
            with pytest.raises(SchemaDriftError) as ei:
                await assert_columns(cur, "guard_demo", {"id", "peer", "level"})
            assert ei.value.missing == ["level"]
        await conn.commit()


@pytestmark_db
@pytest.mark.asyncio
async def test_store_create_fails_loudly_on_drifted_table(pool):
    """The end-to-end point: an OLD drive_state missing `level` (a shape change
    the additive ALTER path can't express) makes DriveState.create() raise at
    startup, naming the column — instead of throwing UndefinedColumn mid-turn."""
    from jeff.appraisal import DriveState

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DROP TABLE IF EXISTS drive_state")
            # Pre-existing table from a hypothetical older build, minus `level`.
            await cur.execute(
                "CREATE TABLE drive_state ("
                "id BIGSERIAL PRIMARY KEY, peer TEXT NOT NULL, drive TEXT NOT NULL, "
                "updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), UNIQUE (peer, drive))"
            )
        await conn.commit()
    with pytest.raises(SchemaDriftError) as ei:
        await DriveState.create(pool, half_life_hours=12.0)
    assert ei.value.table == "drive_state"
    assert "level" in ei.value.missing
    # Clean up so a later module-scoped test sees a fresh DB.
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DROP TABLE IF EXISTS drive_state")
        await conn.commit()
