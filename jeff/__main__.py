"""Entry point: `python -m jeff` and the `jeff` console script.

Subcommands (W3 #dc9acd3c):
    forget <peer-address>   delete every memory row for <peer-address>
"""

from __future__ import annotations

import asyncio
import sys

from psycopg_pool import AsyncConnectionPool

from .config import Config, ConfigError
from .main import run as _run
from .memory import Memory
from .ollama import Ollama


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


if __name__ == "__main__":
    run()
