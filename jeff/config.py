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

# Chat providers Jeff knows how to build (see jeff/llm.py:make_chat_provider).
# `ollama` is the back-compat default (local); `grok` is xAI's cloud API.
_KNOWN_PROVIDERS = frozenset({"ollama", "grok"})

# Provider-specific default chat models. Both are overridable: the
# provider-agnostic JEFF_CHAT_MODEL wins everywhere; under ollama the legacy
# OLLAMA_CHAT_MODEL is still honoured. The Grok default tracks an available
# xAI model id (flagship is grok-4.3 as of 2026-06; grok-4 is the stable
# baseline) — the serves deploy wires the actual id via env.
_DEFAULT_OLLAMA_CHAT_MODEL = "gemma3:12b-it-qat"
_DEFAULT_GROK_MODEL = "grok-4"


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

    llm_provider: str
    ollama_url: str
    chat_model: str
    embed_model: str
    embed_dim: int

    xai_api_key: str | None
    xai_base_url: str

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

        provider = e.get("JEFF_LLM_PROVIDER", "ollama").strip().lower()
        if provider not in _KNOWN_PROVIDERS:
            raise ConfigError(
                f"JEFF_LLM_PROVIDER must be one of {sorted(_KNOWN_PROVIDERS)}; got {provider!r}"
            )

        # Chat model: provider-agnostic JEFF_CHAT_MODEL overrides everywhere;
        # otherwise fall back to a provider-specific default (and the legacy
        # OLLAMA_CHAT_MODEL under the ollama provider).
        generic_model = e.get("JEFF_CHAT_MODEL")
        if provider == "ollama":
            chat_model = generic_model or e.get("OLLAMA_CHAT_MODEL", _DEFAULT_OLLAMA_CHAT_MODEL)
        else:  # grok
            chat_model = generic_model or _DEFAULT_GROK_MODEL

        # Fail fast: don't construct a Grok provider with no key and discover it
        # via a wire 401 mid-turn.
        xai_api_key = e.get("XAI_API_KEY") or None
        if provider == "grok" and not xai_api_key:
            raise ConfigError("JEFF_LLM_PROVIDER=grok requires XAI_API_KEY to be set")

        return Config(
            name=e.get("JEFF_NAME", "jeff"),
            description=e.get("JEFF_DESCRIPTION", "Personal AI assistant"),
            allowlist=_csv(e.get("JEFF_ALLOWLIST")),
            socket=e.get("ENSEMBLE_SOCKET", "/run/ensemble/sock"),
            auth_seed_path=e.get("ENSEMBLE_AUTH_SEED") or None,
            llm_provider=provider,
            ollama_url=e.get("OLLAMA_URL", "http://localhost:11434"),
            chat_model=chat_model,
            embed_model=e.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            embed_dim=_parse_embed_dim(e.get("OLLAMA_EMBED_DIM", "768")),
            xai_api_key=xai_api_key,
            xai_base_url=e.get("XAI_BASE_URL", "https://api.x.ai/v1"),
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
