"""Declared commands: safe dispatch, the handler set, and the daemon specs.

No DB here — `Memory` is faked so these run without Docker. The soft/hard
memory semantics (the `peer_state` watermark) are covered against real Postgres
in tests/test_memory.py. Parsing/routing is the daemon's job now (ensemble), so
there's nothing to test on that front here.
"""

from __future__ import annotations

import pytest

from jeff.commands import (
    Command,
    CommandContext,
    CommandRegistry,
    build_command_registry,
)
from jeff.config import Config


class FakeHandle:
    def __init__(self):
        self.results: list[tuple[str, str]] = []

    async def send_command_result(self, command_id: str, text: str, ok: bool = True) -> None:
        self.results.append((command_id, text))


class FakeMemory:
    """Records command-driven mutations; raises if a turn-path method is hit."""

    def __init__(self, *, count: int = 3, total: int = 7):
        self.cutoffs: list[str] = []
        self.forgotten: list[str] = []
        self._count = count
        self._total = total

    async def set_history_cutoff(self, peer: str) -> None:
        self.cutoffs.append(peer)

    async def forget(self, peer: str) -> int:
        self.forgotten.append(peer)
        return 5

    async def count(self, peer: str) -> int:
        return self._count

    async def total(self) -> int:
        return self._total

    async def remember(self, *a, **k):  # pragma: no cover - must never run
        raise AssertionError("commands must not write to memory")


def _cfg(**extra) -> Config:
    env = {
        "JEFF_DB_URL": "postgresql://jeff:pwd@db.internal:5432/jeff",
        "JEFF_LLM_PROVIDER": "grok",
        "XAI_API_KEY": "xai-SUPERSECRET-do-not-leak",
    }
    env.update(extra)
    return Config.from_env(env)


def _ctx(memory=None, args="", cfg=None) -> CommandContext:
    return CommandContext(
        handle=FakeHandle(),
        memory=memory or FakeMemory(),
        cfg=cfg or _cfg(),
        peer="EpeerD",
        args=args,
    )


# --- the declared set ------------------------------------------------------


def test_registry_declares_exactly_clear_forget_stats():
    reg = build_command_registry()
    assert reg.names() == ["clear", "forget", "stats"]
    # The retired commands are gone — the daemon owns /help and /whoami, and the
    # old soft /new is subsumed by the augmented /clear.
    for gone in ("new", "help", "whoami"):
        assert reg.get(gone) is None


def test_to_ensemble_commands_specs_are_single_sourced():
    import ensemble

    specs = build_command_registry().to_ensemble_commands()
    assert [s.name for s in specs] == ["clear", "forget", "stats"]
    assert all(isinstance(s, ensemble.Command) for s in specs)
    # Description + usage flow straight from the registry (can't drift).
    forget = next(s for s in specs if s.name == "forget")
    assert forget.usage == "yes"
    assert "wipe" in forget.description.lower()


# --- dispatch safety -------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_command_returns_hint_not_exception():
    reply = await build_command_registry().dispatch("nope", _ctx())
    assert reply.startswith("Unknown command /nope")


@pytest.mark.asyncio
async def test_handler_that_raises_yields_safe_apology():
    async def boom(ctx):
        raise RuntimeError("DSN=postgresql://secret leaked")

    reg = CommandRegistry([Command("boom", "explodes", boom)])
    reply = await reg.dispatch("boom", _ctx())
    assert "snag" in reply.lower()
    assert "DSN" not in reply and "secret" not in reply


# --- the handlers ----------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_sets_cutoff_and_keeps_long_term_memory():
    mem = FakeMemory()
    reply = await build_command_registry().dispatch("clear", _ctx(memory=mem))
    assert mem.cutoffs == ["EpeerD"]
    assert "fresh conversation" in reply.lower()


@pytest.mark.asyncio
async def test_forget_requires_confirmation():
    mem = FakeMemory()
    # Bare /forget must NOT delete — it asks for confirmation.
    reply = await build_command_registry().dispatch("forget", _ctx(memory=mem, args=""))
    assert mem.forgotten == []
    assert "/forget yes" in reply


@pytest.mark.asyncio
async def test_forget_yes_wipes_and_reports_count():
    mem = FakeMemory()
    reply = await build_command_registry().dispatch("forget", _ctx(memory=mem, args="yes"))
    assert mem.forgotten == ["EpeerD"]
    assert "5" in reply and "clean slate" in reply.lower()


@pytest.mark.asyncio
async def test_stats_has_counts_model_prompt_source_and_no_secrets():
    cfg = _cfg()
    reply = await build_command_registry().dispatch("stats", _ctx(cfg=cfg))
    assert "3" in reply  # this peer's count
    assert "7" in reply  # total
    assert "uptime" in reply.lower()
    assert cfg.llm_provider in reply
    assert cfg.chat_model in reply
    # /stats absorbed what /whoami used to report.
    assert cfg.system_prompt_source in reply
    _assert_no_secrets(reply, cfg)


def _assert_no_secrets(reply: str, cfg: Config) -> None:
    """No DSN / API key / socket / seed path may appear in a command reply."""
    for secret in (cfg.db_url, cfg.xai_api_key, cfg.socket):
        if secret:
            assert secret not in reply, f"secret leaked into reply: {secret!r}"
    for token in ("postgresql://", "XAI_API_KEY", "Bearer", "api.x.ai"):
        assert token not in reply
