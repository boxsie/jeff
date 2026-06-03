"""Chat command framework: parsing, safe dispatch, and the starter commands.

No DB here — `Memory` is faked so these run without Docker. The soft/hard
memory semantics (the `peer_state` watermark) are covered against real Postgres
in tests/test_memory.py.
"""

from __future__ import annotations

import pytest

from jeff.commands import (
    Command,
    CommandContext,
    CommandRegistry,
    build_command_registry,
    parse_command,
)
from jeff.config import Config


class FakeHandle:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    async def send_message(self, to_addr: str, text: str) -> None:
        self.sent.append((to_addr, text))


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


class ExplodingProvider:
    """Any provider call from the command path is a bug."""

    async def chat(self, *a, **k):  # pragma: no cover
        raise AssertionError("commands must not call the chat provider")

    async def complete(self, *a, **k):  # pragma: no cover
        raise AssertionError("commands must not call the chat provider")


def _cfg(**extra) -> Config:
    env = {
        "JEFF_DB_URL": "postgresql://jeff:pwd@db.internal:5432/jeff",
        "JEFF_LLM_PROVIDER": "grok",
        "XAI_API_KEY": "xai-SUPERSECRET-do-not-leak",
    }
    env.update(extra)
    return Config.from_env(env)


def _ctx(memory=None, args="", registry=None, cfg=None) -> CommandContext:
    registry = registry if registry is not None else build_command_registry()
    return CommandContext(
        handle=FakeHandle(),
        memory=memory or FakeMemory(),
        cfg=cfg or _cfg(),
        chat_provider=ExplodingProvider(),
        peer="EpeerD",
        args=args,
        registry=registry,
    )


# --- parsing ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("/help", ("help", "")),
        ("/new", ("new", "")),
        ("/forget", ("forget", "")),
        ("/forget yes", ("forget", "yes")),
        ("/stats", ("stats", "")),
        ("/whoami", ("whoami", "")),
        ("/HELP", ("help", "")),  # name is case-folded
        ("  /help  ", ("help", "")),  # leading whitespace tolerated
        ("/echo a b c", ("echo", "a b c")),  # args are the stripped remainder
    ],
)
def test_parse_recognises_commands(text, expected):
    assert parse_command(text, "/") == expected


@pytest.mark.parametrize(
    "text",
    [
        "hello there",  # plain chat
        "what is /etc for?",  # slash not at the start
        "/",  # bare prefix, no name
        "/   ",  # prefix + whitespace only
        "",
    ],
)
def test_parse_passes_normal_text_through(text):
    assert parse_command(text, "/") is None


def test_parse_honours_custom_prefix():
    assert parse_command("!ping now", "!") == ("ping", "now")
    assert parse_command("/ping", "!") is None


# --- dispatch safety -------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_command_returns_hint_not_exception():
    reply = await build_command_registry().dispatch("nope", _ctx())
    assert reply.startswith("Unknown command /nope")
    assert "/help" in reply


@pytest.mark.asyncio
async def test_handler_that_raises_yields_safe_apology():
    async def boom(ctx):
        raise RuntimeError("DSN=postgresql://secret leaked")

    reg = CommandRegistry([Command("boom", "explodes", boom)])
    reply = await reg.dispatch("boom", _ctx(registry=reg))
    assert "snag" in reply.lower()
    assert "DSN" not in reply and "secret" not in reply


# --- starter commands ------------------------------------------------------


@pytest.mark.asyncio
async def test_new_sets_cutoff_and_keeps_long_term_memory_in_copy():
    mem = FakeMemory()
    reply = await build_command_registry().dispatch("new", _ctx(memory=mem))
    assert mem.cutoffs == ["EpeerD"]
    assert "fresh conversation" in reply.lower()


@pytest.mark.asyncio
async def test_clear_is_an_alias_for_new():
    mem = FakeMemory()
    await build_command_registry().dispatch("clear", _ctx(memory=mem))
    assert mem.cutoffs == ["EpeerD"]


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
async def test_help_lists_every_registered_command():
    reg = build_command_registry()
    reply = await reg.dispatch("help", _ctx(registry=reg))
    for name in reg.names():
        assert f"/{name}" in reply
    # Descriptions come straight from the registry (can't drift).
    assert reg.get("stats").description in reply


@pytest.mark.asyncio
async def test_stats_has_counts_and_model_and_no_secrets():
    cfg = _cfg()
    reply = await build_command_registry().dispatch("stats", _ctx(cfg=cfg))
    assert "3" in reply  # this peer's count
    assert "7" in reply  # total
    assert "uptime" in reply.lower()
    assert cfg.chat_model in reply
    _assert_no_secrets(reply, cfg)


@pytest.mark.asyncio
async def test_whoami_reports_provider_model_prompt_source_no_secrets():
    cfg = _cfg()
    reply = await build_command_registry().dispatch("whoami", _ctx(cfg=cfg))
    assert cfg.llm_provider in reply
    assert cfg.chat_model in reply
    assert cfg.system_prompt_source in reply
    _assert_no_secrets(reply, cfg)


def _assert_no_secrets(reply: str, cfg: Config) -> None:
    """No DSN / API key / socket / seed path may appear in a command reply."""
    for secret in (cfg.db_url, cfg.xai_api_key, cfg.socket):
        if secret:
            assert secret not in reply, f"secret leaked into reply: {secret!r}"
    for token in ("postgresql://", "XAI_API_KEY", "Bearer", "api.x.ai"):
        assert token not in reply
