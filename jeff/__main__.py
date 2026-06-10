"""Entry point: `python -m jeff` and the `jeff` console script.

Subcommands (W3 #dc9acd3c):
    forget <peer-address>   delete every memory row for <peer-address>
    reset-memory --yes      drop ALL memory and recreate the schema (used after
                            an embedding-model/dimension change)
"""

from __future__ import annotations

import asyncio
import sys

from psycopg_pool import AsyncConnectionPool

from .appraisal import DriveState
from .config import Config, ConfigError
from .curiosity import CuriosityStore
from .main import run as _run
from .memory import Memory
from .mood import MoodStore
from .ollama import Ollama
from .pinned import PinnedMemoryStore
from .proactive import ProactiveStore
from .reflection import ReflectionStore


def run() -> None:
    try:
        cfg = Config.from_env()
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        sys.exit(2)

    if len(sys.argv) >= 2 and sys.argv[1] == "forget":
        if len(sys.argv) != 3 or not sys.argv[2]:
            print(
                "usage: python -m jeff forget <peer-address>",
                file=sys.stderr,
            )
            sys.exit(2)
        asyncio.run(_forget(cfg, sys.argv[2]))
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "reset-memory":
        if "--yes" not in sys.argv[2:]:
            print(
                "reset-memory DROPS all stored messages and cutoffs for every "
                "peer (no undo).\nRe-run with --yes to confirm:\n"
                "  python -m jeff reset-memory --yes",
                file=sys.stderr,
            )
            sys.exit(2)
        asyncio.run(_reset_memory(cfg))
        return

    asyncio.run(_run(cfg))


async def _forget(cfg: Config, peer: str) -> None:
    """Admin path: delete every memory row for the given peer.

    Opens a pool, instantiates Memory without running _init_schema (the
    table must already exist from a prior production run), calls forget,
    prints the row count.
    """
    pool = AsyncConnectionPool(cfg.db_url, min_size=1, max_size=2, open=False)
    await pool.open()
    try:
        async with Ollama(cfg.ollama_url) as ollama:
            memory = Memory(
                pool,
                ollama,
                embed_model=cfg.embed_model,
                embed_dim=cfg.embed_dim,
            )
            deleted = await memory.forget(peer)
            print(f"deleted {deleted} row(s) for peer={peer}")
    finally:
        await pool.close()


async def _reset_memory(cfg: Config) -> None:
    """Admin path: drop ALL memory and recreate the schema at the configured
    embedding dimension. Used after switching embed models (e.g. nomic-768 ->
    bge-m3-1024), where re-embedding old rows isn't wanted.
    """
    pool = AsyncConnectionPool(cfg.db_url, min_size=1, max_size=2, open=False)
    await pool.open()
    try:
        async with Ollama(cfg.ollama_url) as ollama:
            memory = Memory(
                pool,
                ollama,
                embed_model=cfg.embed_model,
                embed_dim=cfg.embed_dim,
            )
            await memory.reset()
            # Same embed dim drives the curiosity store, so a dimension change
            # invalidates it too — recreate it alongside messages.
            curiosity = CuriosityStore(
                pool,
                ollama,
                embed_model=cfg.embed_model,
                embed_dim=cfg.embed_dim,
            )
            await curiosity.reset()
            # Derived memory (facts + opinions) is embedded at the same dim, so a
            # dimension change invalidates it too — recreate it alongside.
            reflection = ReflectionStore(
                pool,
                ollama,
                embed_model=cfg.embed_model,
                embed_dim=cfg.embed_dim,
            )
            await reflection.reset()
            # Moods + pinned memory are plain tables (no embedding), so they're
            # unaffected by a dimension change — but reset-memory means "clean
            # slate", so drop and recreate them too for consistency.
            mood = MoodStore(pool)
            await mood.reset()
            pinned = PinnedMemoryStore(pool)
            await pinned.reset()
            # Drive state is a plain table (no embedding) too — unaffected by a
            # dimension change, but "clean slate" means drop and recreate it.
            drives = DriveState(pool, half_life_hours=cfg.drive_decay_half_life_hours)
            await drives.reset()
            # Proactive bookkeeping is a plain table too — recreate on clean slate.
            proactive = ProactiveStore(pool)
            await proactive.reset()
            print(
                f"memory reset — messages + curiosities + derived_memory + mood + "
                f"pinned + drive_state + proactive_state schema recreated at "
                f"dim={cfg.embed_dim} (model={cfg.embed_model})"
            )
    finally:
        await pool.close()


if __name__ == "__main__":
    run()
