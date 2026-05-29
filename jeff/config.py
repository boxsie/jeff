"""Env-driven configuration for the Jeff bot."""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(Exception):
    """Raised when required environment configuration is missing or malformed."""


# W3 #20123205: bound the embedding-dimension envelope so an env-supplied
# value can't slip through into the schema CREATE statement. Lower bound
# rules out the obvious typo / zero, upper bound is the same cap the wire
# reader uses (OLLAMA_MAX_EMBED_DIM). Operator-controlled value, so we fail
# fast at config load — production callers don't get a chance to set up a
# pool against a malformed schema.
_EMBED_DIM_MIN = 1
_EMBED_DIM_MAX = 8192


def _parse_embed_dim(raw: str) -> int:
    try:
        n = int(raw)
    except ValueError as e:
        raise ConfigError(
            f"OLLAMA_EMBED_DIM must be an integer; got {raw!r}"
        ) from e
    if n < _EMBED_DIM_MIN or n > _EMBED_DIM_MAX:
        raise ConfigError(
            f"OLLAMA_EMBED_DIM out of range: {n} (allowed {_EMBED_DIM_MIN}..{_EMBED_DIM_MAX})"
        )
    return n


def _csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


@dataclass(frozen=True)
class Config:
    name: str
    description: str
    allowlist: list[str]

    socket: str
    auth_seed_path: str | None

    ollama_url: str
    chat_model: str
    embed_model: str
    embed_dim: int

    db_url: str

    recall_k: int
    recent_turns: int

    max_inflight: int
    per_peer_concurrency: int
    peer_rate_per_minute: float
    peer_rate_burst: int
    peer_idle_timeout_s: float

    max_message_bytes: int
    ollama_max_resp_bytes: int
    ollama_max_embed_dim: int

    @staticmethod
    def from_env(env: dict[str, str] | None = None) -> "Config":
        e = env if env is not None else os.environ

        db_url = e.get("JEFF_DB_URL")
        if not db_url:
            raise ConfigError(
                "JEFF_DB_URL is required (e.g. postgresql://jeff:pwd@host:5432/jeff)"
            )

        return Config(
            name=e.get("JEFF_NAME", "jeff"),
            description=e.get("JEFF_DESCRIPTION", "Personal AI assistant"),
            allowlist=_csv(e.get("JEFF_ALLOWLIST")),
            socket=e.get("ENSEMBLE_SOCKET", "/run/ensemble/sock"),
            auth_seed_path=e.get("ENSEMBLE_AUTH_SEED") or None,
            ollama_url=e.get("OLLAMA_URL", "http://localhost:11434"),
            chat_model=e.get("OLLAMA_CHAT_MODEL", "gemma3:12b-it-qat"),
            embed_model=e.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            embed_dim=_parse_embed_dim(e.get("OLLAMA_EMBED_DIM", "768")),
            db_url=db_url,
            recall_k=int(e.get("MEMORY_RECALL_K", "5")),
            recent_turns=int(e.get("MEMORY_RECENT_TURNS", "10")),
            max_inflight=int(e.get("JEFF_MAX_INFLIGHT", "32")),
            per_peer_concurrency=int(e.get("JEFF_PER_PEER_CONCURRENCY", "1")),
            peer_rate_per_minute=float(e.get("JEFF_PEER_RATE_PER_MINUTE", "6")),
            peer_rate_burst=int(e.get("JEFF_PEER_RATE_BURST", "3")),
            peer_idle_timeout_s=float(e.get("JEFF_PEER_IDLE_TIMEOUT_S", "600")),
            max_message_bytes=int(e.get("JEFF_MAX_MESSAGE_BYTES", "8192")),
            ollama_max_resp_bytes=int(
                e.get("OLLAMA_MAX_RESP_BYTES", str(8 * 1024 * 1024))
            ),
            ollama_max_embed_dim=int(e.get("OLLAMA_MAX_EMBED_DIM", "8192")),
        )
