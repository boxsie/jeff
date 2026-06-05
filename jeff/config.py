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

# Upper bound on a custom system prompt (characters). Generous — several pages
# of persona text — but bounded so a runaway file/env can't blow the context
# window or per-turn token cost. Operator-controlled, so fail fast at load.
_SYSTEM_PROMPT_MAX_CHARS = 32_768


def _resolve_system_prompt(e: dict[str, str]) -> tuple[str, str]:
    """Resolve the active system prompt and a source label for logging.

    Precedence: ``JEFF_SYSTEM_PROMPT_FILE`` > ``JEFF_SYSTEM_PROMPT`` > the
    built-in ``prompt.SYSTEM_PROMPT`` default. The override replaces the
    entire prompt verbatim (after a surrounding-whitespace strip) — Jeff is
    single-user (ACL = allowlist of just the operator), so the operator owns
    the whole prompt, guardrail and all (jeff ticket 5d94d5b1). Returns
    ``(prompt_text, source)`` with source one of ``"file" | "env" | "default"``.
    """
    path = e.get("JEFF_SYSTEM_PROMPT_FILE")
    if path:
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError as ex:
            # Fail fast — don't silently fall back to the default when the
            # operator explicitly pointed at a file that can't be read.
            raise ConfigError(
                f"JEFF_SYSTEM_PROMPT_FILE could not be read ({path!r}): "
                f"{ex.strerror or ex}"
            ) from ex
        text = text.strip()
        if not text:
            raise ConfigError(f"JEFF_SYSTEM_PROMPT_FILE is empty: {path!r}")
        _check_prompt_length(text, "JEFF_SYSTEM_PROMPT_FILE")
        return text, "file"

    inline = e.get("JEFF_SYSTEM_PROMPT")
    if inline and inline.strip():
        text = inline.strip()
        _check_prompt_length(text, "JEFF_SYSTEM_PROMPT")
        return text, "env"

    # Lazy import: prompt.py pulls in memory (numpy/pgvector/psycopg), and
    # config is imported early + in lightweight unit tests. Only pay that
    # cost when no override is supplied.
    from .prompt import SYSTEM_PROMPT

    return SYSTEM_PROMPT, "default"


def _check_prompt_length(text: str, var: str) -> None:
    if len(text) > _SYSTEM_PROMPT_MAX_CHARS:
        raise ConfigError(
            f"{var} too long: {len(text)} chars (max {_SYSTEM_PROMPT_MAX_CHARS})"
        )


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


# SearXNG safesearch levels: 0 = off, 1 = moderate, 2 = strict. Jeff defaults
# to off (the operator wants to search for anything); operator-controlled, so a
# bad value fails fast at load rather than being silently coerced.
_SAFESEARCH_LEVELS = frozenset({0, 1, 2})


def _parse_safesearch(raw: str) -> int:
    try:
        n = int(raw)
    except ValueError as e:
        raise ConfigError(
            f"JEFF_SEARCH_SAFESEARCH must be 0, 1, or 2; got {raw!r}"
        ) from e
    if n not in _SAFESEARCH_LEVELS:
        raise ConfigError(
            f"JEFF_SEARCH_SAFESEARCH must be 0 (off), 1 (moderate), or 2 (strict); got {n}"
        )
    return n


# Cosine *distance* ceiling for recall (pgvector "<=>" ranges 0..2; 0 = identical,
# 1 = orthogonal). Default 0.55 is tuned for bge-m3, which cleanly separates
# relevant (~0.4-0.52) from unrelated (~0.7); the old 0.4 with nomic-embed-text
# dropped almost everything. Operator-tunable per embed model, so a bad value
# fails fast at load.
def _parse_recall_distance(raw: str) -> float:
    try:
        n = float(raw)
    except ValueError as e:
        raise ConfigError(
            f"MEMORY_RECALL_DISTANCE must be a number; got {raw!r}"
        ) from e
    if not (0.0 < n <= 2.0):
        raise ConfigError(
            f"MEMORY_RECALL_DISTANCE out of range: {n} (allowed 0 < x <= 2)"
        )
    return n


def _parse_bool(raw: str | None) -> bool:
    """Parse a permissive boolean env value. Anything truthy-looking is True."""
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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

    system_prompt: str
    system_prompt_source: str

    db_url: str

    recall_k: int
    recent_turns: int
    recall_distance_max: float

    tools_enabled: bool
    max_tool_iters: int
    tool_timeout_s: float

    search_enabled: bool
    searxng_url: str
    searxng_auth: str | None
    search_safesearch: int

    commands_enabled: bool

    curiosity_enabled: bool
    curiosity_every_turns: int
    curiosity_max_open: int
    curiosity_max_new_per_pass: int

    reflection_enabled: bool
    reflection_every_turns: int
    reflection_max_chars: int
    reflection_max_items: int
    reflection_use_persona: bool
    persona_max_chars: int

    mood_enabled: bool
    mood_default_hours: float
    mood_max_hours: float
    mood_max_chars: int

    remember_enabled: bool
    remember_max_items: int
    remember_max_chars: int

    appraisal_enabled: bool
    appraisal_every_turns: int
    drive_decay_half_life_hours: float
    drives_max_chars: int

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

        system_prompt, system_prompt_source = _resolve_system_prompt(e)

        # Search: the in-cluster SearXNG URL is NOT committed (leaky-info rule)
        # — it arrives via the serves ConfigMap; the code default is localhost.
        # Fail fast if search is turned on but the URL was explicitly cleared,
        # mirroring the provider=grok + missing-key precedent: an operator who
        # enables search and forgets the endpoint should hear about it at load,
        # not as a runtime connection error mid-turn.
        search_enabled = _parse_bool(e.get("JEFF_SEARCH_ENABLED", "false"))
        searxng_url = e.get("JEFF_SEARXNG_URL", "http://localhost:8888").strip()
        if search_enabled and not searxng_url:
            raise ConfigError("JEFF_SEARCH_ENABLED is on but JEFF_SEARXNG_URL is empty")

        return Config(
            name=e.get("JEFF_NAME", "jeff"),
            description=e.get("JEFF_DESCRIPTION", "Personal AI assistant"),
            allowlist=_csv(e.get("JEFF_ALLOWLIST")),
            socket=e.get("ENSEMBLE_SOCKET", "/run/ensemble/sock"),
            auth_seed_path=e.get("ENSEMBLE_AUTH_SEED") or None,
            llm_provider=provider,
            ollama_url=e.get("OLLAMA_URL", "http://localhost:11434"),
            chat_model=chat_model,
            embed_model=e.get("OLLAMA_EMBED_MODEL", "bge-m3"),
            embed_dim=_parse_embed_dim(e.get("OLLAMA_EMBED_DIM", "1024")),
            xai_api_key=xai_api_key,
            xai_base_url=e.get("XAI_BASE_URL", "https://api.x.ai/v1"),
            system_prompt=system_prompt,
            system_prompt_source=system_prompt_source,
            db_url=db_url,
            recall_k=int(e.get("MEMORY_RECALL_K", "5")),
            recent_turns=int(e.get("MEMORY_RECENT_TURNS", "10")),
            recall_distance_max=_parse_recall_distance(
                e.get("MEMORY_RECALL_DISTANCE", "0.55")
            ),
            tools_enabled=_parse_bool(e.get("JEFF_TOOLS_ENABLED", "true")),
            max_tool_iters=int(e.get("JEFF_MAX_TOOL_ITERS", "5")),
            tool_timeout_s=float(e.get("JEFF_TOOL_TIMEOUT_S", "30")),
            search_enabled=search_enabled,
            searxng_url=searxng_url,
            searxng_auth=e.get("JEFF_SEARXNG_AUTH") or None,
            search_safesearch=_parse_safesearch(e.get("JEFF_SEARCH_SAFESEARCH", "0")),
            commands_enabled=_parse_bool(e.get("JEFF_COMMANDS_ENABLED", "true")),
            # Curiosity drive (motivation-system slice 1) — default OFF: when off
            # the store/driver are never built, so behaviour is byte-identical.
            curiosity_enabled=_parse_bool(e.get("JEFF_CURIOSITY_ENABLED", "false")),
            curiosity_every_turns=int(e.get("JEFF_CURIOSITY_EVERY_TURNS", "1")),
            curiosity_max_open=int(e.get("JEFF_CURIOSITY_MAX_OPEN", "6")),
            curiosity_max_new_per_pass=int(e.get("JEFF_CURIOSITY_MAX_NEW", "3")),
            # Reflection / emergent personality (motivation-system slice 2) —
            # default OFF: when off the store/reflector are never built, so
            # behaviour is byte-identical. Defaults mirror the deployed configmap.
            reflection_enabled=_parse_bool(e.get("JEFF_REFLECTION_ENABLED", "false")),
            reflection_every_turns=int(e.get("JEFF_REFLECTION_EVERY_TURNS", "8")),
            reflection_max_chars=int(e.get("JEFF_REFLECTION_MAX_CHARS", "6000")),
            reflection_max_items=int(e.get("JEFF_REFLECTION_MAX_ITEMS", "6")),
            # Feed the base persona into the distillation pass (default on) so
            # opinions inherit Jeff's voice. Off → the neutral extractor only.
            reflection_use_persona=_parse_bool(
                e.get("JEFF_REFLECTION_USE_PERSONA", "true")
            ),
            persona_max_chars=int(e.get("JEFF_PERSONA_MAX_CHARS", "2000")),
            # Mood drive (affective state, inner-life slice 3) — default OFF:
            # when off the store/tools are never built, so behaviour is
            # byte-identical. Jeff sets/defines moods at runtime via tools; there
            # is deliberately no seed palette (co-authored with the operator).
            # default_hours is how long a set mood lasts when Jeff doesn't say;
            # max_hours is the hard ceiling on any one mood; max_chars caps a
            # stored definition (tool-enforced, with a higher hard byte cap in
            # the store as defence-in-depth).
            mood_enabled=_parse_bool(e.get("JEFF_MOOD_ENABLED", "false")),
            mood_default_hours=float(e.get("JEFF_MOOD_DEFAULT_HOURS", "6")),
            mood_max_hours=float(e.get("JEFF_MOOD_MAX_HOURS", "48")),
            mood_max_chars=int(e.get("JEFF_MOOD_MAX_CHARS", "2000")),
            # Explicit/pinned memory (inner-life slice 4) — default OFF: when off
            # the store/tool/command are never built, so behaviour is
            # byte-identical. Jeff pins via the `remember` tool, the operator via
            # `/remember`; both share one plain table (no embedding). max_items
            # caps how many pins are injected + listed; max_chars caps one pin
            # (tool/command-enforced, with a higher hard byte cap in the store).
            remember_enabled=_parse_bool(e.get("JEFF_REMEMBER_ENABLED", "false")),
            remember_max_items=int(e.get("JEFF_REMEMBER_MAX_ITEMS", "20")),
            remember_max_chars=int(e.get("JEFF_REMEMBER_MAX_CHARS", "2000")),
            # Appraisal / reward (motivation-system slice 3) — default OFF: when
            # off the store/driver are never built, so behaviour is byte-identical.
            # This is the feedback edge that closes the motivation loop — a
            # post-turn pass rates each exchange and nudges the standing drives,
            # which decay toward their baseline with the configured half-life. The
            # drive set + per-drive baselines are a constant registry in
            # appraisal.py (adding a drive is one line, not an env knob).
            appraisal_enabled=_parse_bool(e.get("JEFF_APPRAISAL_ENABLED", "false")),
            appraisal_every_turns=int(e.get("JEFF_APPRAISAL_EVERY_TURNS", "1")),
            drive_decay_half_life_hours=float(
                e.get("JEFF_DRIVE_DECAY_HALF_LIFE_HOURS", "24")
            ),
            drives_max_chars=int(e.get("JEFF_DRIVES_MAX_CHARS", "2000")),
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
