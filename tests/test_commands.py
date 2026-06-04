"""Declared commands: safe dispatch, the handler set, and the daemon specs.

No DB here — `Memory` is faked so these run without Docker. The soft/hard
memory semantics (the `peer_state` watermark) are covered against real Postgres
in tests/test_memory.py. Parsing/routing is the daemon's job now (ensemble), so
there's nothing to test on that front here.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jeff.commands import (
    Command,
    CommandContext,
    CommandRegistry,
    build_command_registry,
)
from jeff.config import Config
from jeff.memory import Message


class FakeHandle:
    def __init__(self):
        self.results: list[tuple[str, str]] = []

    async def send_command_result(self, command_id: str, text: str, ok: bool = True) -> None:
        self.results.append((command_id, text))


class FakeMemory:
    """Records command-driven mutations; raises if a turn-path method is hit."""

    def __init__(
        self,
        *,
        count: int = 3,
        total: int = 7,
        cutoff: datetime | None = None,
        recent: list[Message] | None = None,
        scored: list[tuple[Message, float]] | None = None,
    ):
        self.cutoffs: list[str] = []
        self.forgotten: list[str] = []
        self._count = count
        self._total = total
        self._cutoff = cutoff
        self._recent = recent or []
        self._scored = scored or []

    async def set_history_cutoff(self, peer: str) -> None:
        self.cutoffs.append(peer)

    async def forget(self, peer: str) -> int:
        self.forgotten.append(peer)
        return 5

    async def count(self, peer: str) -> int:
        return self._count

    async def total(self) -> int:
        return self._total

    async def get_history_cutoff(self, peer: str) -> datetime | None:
        return self._cutoff

    async def recent(self, peer: str, n: int = 10) -> list[Message]:
        return self._recent[:n]

    async def recall_scored(self, peer, query, *, limit: int = 8):
        return self._scored[:limit]

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


def _ctx(memory=None, args="", cfg=None, system_prompt="", tool_names=()) -> CommandContext:
    return CommandContext(
        handle=FakeHandle(),
        memory=memory or FakeMemory(),
        cfg=cfg or _cfg(),
        peer="EpeerD",
        args=args,
        system_prompt=system_prompt,
        tool_names=tool_names,
    )


def _msg(id: int, role: str, content: str, ts: datetime | None = None) -> Message:
    return Message(
        id=id,
        peer="EpeerD",
        role=role,
        content=content,
        ts=ts or datetime(2026, 6, 4, 11, 2, 14, tzinfo=timezone.utc),
    )


# --- the declared set ------------------------------------------------------


def test_registry_declares_clear_debug_forget_stats():
    reg = build_command_registry()
    assert reg.names() == ["clear", "debug", "forget", "stats"]
    # The retired commands are gone — the daemon owns /help and /whoami, and the
    # old soft /new is subsumed by the augmented /clear.
    for gone in ("new", "help", "whoami"):
        assert reg.get(gone) is None


def test_to_ensemble_commands_specs_are_single_sourced():
    import ensemble

    specs = build_command_registry().to_ensemble_commands()
    assert [s.name for s in specs] == ["clear", "debug", "forget", "stats"]
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


# --- /debug ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_debug_overview_shows_cutoff_recent_counts_and_tools():
    cfg = _cfg()
    mem = FakeMemory(
        count=2,
        total=9,
        cutoff=datetime(2026, 6, 4, 11, 0, 0, tzinfo=timezone.utc),
        recent=[
            _msg(84, "user", "espresso every morning"),
            _msg(85, "assistant", "Noted — a morning espresso."),
        ],
    )
    reply = await build_command_registry().dispatch(
        "debug",
        _ctx(memory=mem, cfg=cfg, system_prompt="SYSTEM-PROMPT-BODY",
             tool_names=("web_search", "get_time")),
    )
    assert "debug — context" in reply
    assert "2026-06-04 11:00:00" in reply           # session cutoff shown
    assert "you=2" in reply and "all peers=9" in reply
    assert "#84" in reply and "espresso every morning" in reply
    assert "#85" in reply
    assert "web_search, get_time" in reply          # tools listed
    # Overview reports the prompt's length, not its body.
    assert f"{len('SYSTEM-PROMPT-BODY')} chars" in reply
    assert "SYSTEM-PROMPT-BODY" not in reply
    _assert_no_secrets(reply, cfg)


@pytest.mark.asyncio
async def test_debug_overview_empty_session():
    mem = FakeMemory(count=0, total=0, cutoff=None, recent=[])
    reply = await build_command_registry().dispatch("debug", _ctx(memory=mem))
    assert "full history in window" in reply        # no cutoff set
    assert "empty — fresh session" in reply


@pytest.mark.asyncio
async def test_debug_prompt_shows_full_effective_prompt():
    reply = await build_command_registry().dispatch(
        "debug", _ctx(args="prompt", system_prompt="FULL-EFFECTIVE-PROMPT-HERE")
    )
    assert "effective system prompt" in reply.lower()
    assert "FULL-EFFECTIVE-PROMPT-HERE" in reply


@pytest.mark.asyncio
async def test_debug_recall_marks_kept_rows_with_distances():
    # 0.18 and 0.37 are within the 0.4 threshold (kept ✓); 0.52 is over it.
    scored = [
        (_msg(84, "user", "espresso"), 0.18),
        (_msg(85, "assistant", "noted"), 0.37),
        (_msg(50, "user", "cycling"), 0.52),
    ]
    mem = FakeMemory(scored=scored)
    reply = await build_command_registry().dispatch(
        "debug", _ctx(memory=mem, args="recall espresso morning")
    )
    assert "debug — recall" in reply
    assert "0.180" in reply and "0.370" in reply and "0.520" in reply
    assert "✓ 0.180" in reply          # within threshold → kept
    assert "✓ 0.520" not in reply      # over threshold → not kept
    assert "#84" in reply and "#50" in reply


@pytest.mark.asyncio
async def test_debug_recall_without_query_shows_usage():
    reply = await build_command_registry().dispatch("debug", _ctx(args="recall"))
    assert "usage" in reply.lower() and "/debug recall" in reply


@pytest.mark.asyncio
async def test_debug_recall_no_candidates():
    mem = FakeMemory(scored=[])
    reply = await build_command_registry().dispatch(
        "debug", _ctx(memory=mem, args="recall anything")
    )
    assert "no candidates" in reply.lower()


@pytest.mark.asyncio
async def test_debug_unknown_subcommand_hints():
    reply = await build_command_registry().dispatch("debug", _ctx(args="wibble"))
    assert "unknown debug view" in reply.lower()
