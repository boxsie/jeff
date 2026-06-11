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


class FakeCuriosityStore:
    """Records curiosity mutations for the /forget + /mind command paths."""

    def __init__(self, open_q=None, satisfied=None):
        self._open = open_q or []
        self._satisfied = satisfied or []
        self.forgotten: list[str] = []

    async def open_curiosities(self, peer, *, limit=10):
        return self._open[:limit]

    async def recently_satisfied(self, peer, *, limit=10):
        return self._satisfied[:limit]

    async def forget(self, peer):
        self.forgotten.append(peer)
        return len(self._open) + len(self._satisfied)


class _Cur:
    def __init__(self, text: str):
        self.text = text


class FakeReflectionStore:
    """Records derived-memory reads/wipes for the /forget + /mind command paths."""

    def __init__(self, facts=None, opinions=None):
        self._facts = facts or []
        self._opinions = opinions or []
        self.forgotten: list[str] = []

    async def fetch(self, peer, *, kind=None, limit=30):
        from jeff.reflection import FACT

        if kind == FACT:
            return self._facts[:limit]
        return self._opinions[:limit]

    async def forget(self, peer):
        self.forgotten.append(peer)
        return len(self._facts) + len(self._opinions)


class _Derived:
    def __init__(self, text: str):
        self.text = text


class FakePinnedStore:
    """Records pinned-memory reads/writes/wipes for the command paths."""

    def __init__(self, pins=None):
        self._pins = list(pins or [])
        self.added: list[tuple[str, str]] = []  # (text, source)
        self.forgotten: list[str] = []
        self.next_id: int | None = 1

    async def add(self, peer, text, *, source="jeff"):
        self.added.append((text, source))
        return self.next_id

    async def list(self, peer, *, limit=50):
        return self._pins[:limit]

    async def forget(self, peer):
        self.forgotten.append(peer)
        return len(self._pins)


class _Pin:
    def __init__(self, text: str, source: str = "jeff"):
        self.text = text
        self.source = source


class FakeImpulseStore:
    """Records impulse reads/wipes for the command paths."""

    def __init__(self, active=None):
        self._active = list(active or [])
        self.forgotten: list[str] = []

    async def list_active(self, peer, *, limit=50):
        return self._active[:limit]

    async def forget(self, peer):
        self.forgotten.append(peer)
        return len(self._active)


class _Imp:
    def __init__(self, name, description, *, strength=1, source="jeff", expires_at=None):
        self.name = name
        self.description = description
        self.strength = strength
        self.source = source
        self.expires_at = expires_at


def _ctx(
    memory=None,
    args="",
    cfg=None,
    system_prompt="",
    tool_names=(),
    curiosity=None,
    reflection=None,
    pinned=None,
    drives=None,
    proactive=None,
    impulses=None,
    musings=None,
) -> CommandContext:
    return CommandContext(
        handle=FakeHandle(),
        memory=memory or FakeMemory(),
        cfg=cfg or _cfg(),
        peer="EpeerD",
        args=args,
        system_prompt=system_prompt,
        tool_names=tool_names,
        curiosity=curiosity,
        reflection=reflection,
        pinned=pinned,
        drives=drives,
        proactive=proactive,
        impulses=impulses,
        musings=musings,
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
    # Default threshold is 0.55: 0.18 and 0.37 are kept (✓); 0.62 is over it.
    scored = [
        (_msg(84, "user", "espresso"), 0.18),
        (_msg(85, "assistant", "noted"), 0.37),
        (_msg(50, "user", "cycling"), 0.62),
    ]
    mem = FakeMemory(scored=scored)
    reply = await build_command_registry().dispatch(
        "debug", _ctx(memory=mem, args="recall espresso morning")
    )
    assert "debug — recall" in reply
    assert "dist <= 0.55" in reply       # configured threshold shown
    assert "0.180" in reply and "0.370" in reply and "0.620" in reply
    assert "✓ 0.180" in reply            # within threshold → kept
    assert "✓ 0.620" not in reply        # over threshold → not kept
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
    assert "no stored messages" in reply.lower()


@pytest.mark.asyncio
async def test_debug_unknown_subcommand_hints():
    reply = await build_command_registry().dispatch("debug", _ctx(args="wibble"))
    assert "unknown debug view" in reply.lower()


# --- curiosity drive: /mind + /forget wiring -------------------------------


def test_mind_declared_only_when_curiosity_enabled():
    # Default (off) → no /mind, so the declared set is unchanged.
    assert build_command_registry().get("mind") is None
    assert "mind" not in build_command_registry().names()
    # Enabled → /mind appears.
    reg = build_command_registry(curiosity_enabled=True)
    assert reg.get("mind") is not None
    assert "mind" in reg.names()


@pytest.mark.asyncio
async def test_mind_lists_open_and_satisfied_curiosities():
    store = FakeCuriosityStore(
        open_q=[_Cur("What's your homelab called?"), _Cur("Road or MTB?")],
        satisfied=[_Cur("Where do you live?")],
    )
    reg = build_command_registry(curiosity_enabled=True)
    reply = await reg.dispatch("mind", _ctx(curiosity=store))
    assert "on my mind" in reply.lower()
    assert "What's your homelab called?" in reply
    assert "Road or MTB?" in reply
    assert "recently answered" in reply.lower()
    assert "Where do you live?" in reply


@pytest.mark.asyncio
async def test_mind_without_any_store_says_off():
    # Both drives off → /mind explains there's nothing to show.
    reg = build_command_registry(curiosity_enabled=True)
    reply = await reg.dispatch("mind", _ctx(curiosity=None, reflection=None))
    assert "switched on" in reply.lower()
    assert "nothing on my mind" in reply.lower()


@pytest.mark.asyncio
async def test_forget_yes_also_wipes_curiosity_store():
    mem = FakeMemory()
    store = FakeCuriosityStore(open_q=[_Cur("anything?")])
    reply = await build_command_registry().dispatch(
        "forget", _ctx(memory=mem, args="yes", curiosity=store)
    )
    assert mem.forgotten == ["EpeerD"]
    assert store.forgotten == ["EpeerD"]
    assert "clean slate" in reply.lower()


@pytest.mark.asyncio
async def test_forget_yes_without_curiosity_store_is_fine():
    mem = FakeMemory()
    reply = await build_command_registry().dispatch(
        "forget", _ctx(memory=mem, args="yes", curiosity=None)
    )
    assert mem.forgotten == ["EpeerD"]
    assert "clean slate" in reply.lower()


# --- reflection drive: /mind persona + /forget wiring ----------------------


def test_mind_declared_when_only_reflection_enabled():
    # Reflection alone is enough to surface /mind (curiosity off).
    reg = build_command_registry(reflection_enabled=True)
    assert reg.get("mind") is not None
    assert "mind" in reg.names()


@pytest.mark.asyncio
async def test_mind_lists_facts_and_opinions():
    store = FakeReflectionStore(
        facts=[_Derived("Works as a backend developer")],
        opinions=[_Derived("I like their blunt questions")],
    )
    reg = build_command_registry(reflection_enabled=True)
    reply = await reg.dispatch("mind", _ctx(reflection=store))
    assert "on my mind" in reply.lower()
    assert "what i know about you" in reply.lower()
    assert "Works as a backend developer" in reply
    assert "my own take" in reply.lower()
    assert "I like their blunt questions" in reply


@pytest.mark.asyncio
async def test_mind_shows_both_curiosity_and_reflection_when_present():
    cur = FakeCuriosityStore(open_q=[_Cur("Road or MTB?")])
    refl = FakeReflectionStore(facts=[_Derived("Rides a steel road bike")])
    reg = build_command_registry(curiosity_enabled=True, reflection_enabled=True)
    reply = await reg.dispatch("mind", _ctx(curiosity=cur, reflection=refl))
    assert "Road or MTB?" in reply
    assert "Rides a steel road bike" in reply


@pytest.mark.asyncio
async def test_forget_yes_also_wipes_reflection_store():
    mem = FakeMemory()
    refl = FakeReflectionStore(facts=[_Derived("a fact")])
    reply = await build_command_registry().dispatch(
        "forget", _ctx(memory=mem, args="yes", reflection=refl)
    )
    assert mem.forgotten == ["EpeerD"]
    assert refl.forgotten == ["EpeerD"]
    assert "clean slate" in reply.lower()


# --- /remember + pinned memory --------------------------------------------


def test_remember_declared_only_when_enabled():
    assert build_command_registry().get("remember") is None
    assert build_command_registry(remember_enabled=True).get("remember") is not None
    # /mind is also declared once remember is on (even with the others off).
    assert build_command_registry(remember_enabled=True).get("mind") is not None


@pytest.mark.asyncio
async def test_remember_pins_text():
    store = FakePinnedStore()
    reg = build_command_registry(remember_enabled=True)
    reply = await reg.dispatch("remember", _ctx(args="Has a dog named Biscuit", pinned=store))
    assert store.added == [("Has a dog named Biscuit", "operator")]
    assert "pinned" in reply.lower()


@pytest.mark.asyncio
async def test_remember_empty_shows_usage():
    store = FakePinnedStore()
    reg = build_command_registry(remember_enabled=True)
    reply = await reg.dispatch("remember", _ctx(args="   ", pinned=store))
    assert store.added == []
    assert "usage" in reply.lower()


@pytest.mark.asyncio
async def test_remember_reports_duplicate():
    store = FakePinnedStore()
    store.next_id = None
    reg = build_command_registry(remember_enabled=True)
    reply = await reg.dispatch("remember", _ctx(args="dup", pinned=store))
    assert "already had that pinned" in reply


@pytest.mark.asyncio
async def test_mind_shows_pinned_with_provenance():
    store = FakePinnedStore(pins=[_Pin("Likes tea", source="operator"), _Pin("Has a dog", source="jeff")])
    reg = build_command_registry(remember_enabled=True)
    reply = await reg.dispatch("mind", _ctx(pinned=store))
    assert "pinned memory" in reply.lower()
    assert "[you] Likes tea" in reply
    assert "[me] Has a dog" in reply


@pytest.mark.asyncio
async def test_forget_yes_also_wipes_pinned_store():
    mem = FakeMemory()
    store = FakePinnedStore(pins=[_Pin("x")])
    reply = await build_command_registry().dispatch(
        "forget", _ctx(memory=mem, args="yes", pinned=store)
    )
    assert mem.forgotten == ["EpeerD"]
    assert store.forgotten == ["EpeerD"]
    assert "clean slate" in reply.lower()


# --- appraisal drive: /mind drives + /forget wiring ------------------------


class FakeDriveStore:
    """In-memory stand-in for DriveState that records /forget + serves /mind."""

    def __init__(self, levels=None, references=None, income=None, spends=None):
        self._levels = levels or {"connection": 0.8, "novelty": 0.2,
                                  "competence": 0.5, "autonomy": 0.5}
        self._refs = references or {}
        self._income = income or []  # list[IncomeRecord]
        self._spends = spends or []  # list[SpendRecord]
        self.forgotten: list[str] = []

    async def state(self, peer):
        from jeff.appraisal import DriveReading

        return {
            k: DriveReading(v, self._refs.get(k, 0.0))
            for k, v in self._levels.items()
        }

    async def recent_income(self, peer, limit=20):
        return list(self._income[:limit])

    async def recent_spends(self, peer, limit=20):
        return list(self._spends[:limit])

    async def forget(self, peer):
        self.forgotten.append(peer)
        return len(self._levels)


def test_mind_declared_when_only_appraisal_enabled():
    # Appraisal alone is enough to surface /mind (the others off).
    reg = build_command_registry(appraisal_enabled=True)
    assert reg.get("mind") is not None
    assert "mind" in reg.names()


@pytest.mark.asyncio
async def test_mind_shows_drive_balance_with_bands():
    # Bands are judged against each drive's rolling reference (avg), not an
    # absolute mark: 0.80 well above its 0.40 norm → well-met; 0.20 well below its
    # 0.50 norm → running low; 0.50 at its 0.50 norm → steady.
    store = FakeDriveStore(
        levels={"connection": 0.8, "novelty": 0.2, "competence": 0.5, "autonomy": 0.5},
        references={"connection": 0.4, "novelty": 0.5, "competence": 0.5, "autonomy": 0.5},
    )
    reg = build_command_registry(appraisal_enabled=True)
    reply = await reg.dispatch("mind", _ctx(drives=store))
    assert "drives:" in reply.lower()
    # Each drive shows its level + a one-word band + its reference (avg).
    assert "connection: 0.80 (well-met, avg 0.40)" in reply
    assert "novelty: 0.20 (running low, avg 0.50)" in reply
    assert "competence: 0.50 (steady, avg 0.50)" in reply
    # No economy activity yet → the actions line reads empty, no flow suffixes.
    assert "recent actions: (nothing spent yet)" in reply
    assert "fed" not in reply


@pytest.mark.asyncio
async def test_mind_shows_economy_flow_and_pnl():
    # Income feeds + a settled spend (net +0.05), a pending spend, and a flop
    # (credit 0 → net −cost). The view shows per-drive fed/spent and a P&L list.
    from jeff.appraisal import IncomeRecord, SpendRecord

    store = FakeDriveStore(
        income=[IncomeRecord("connection", 0.2, None),
                IncomeRecord("novelty", 0.1, None)],
        spends=[
            SpendRecord("reach_out", "connection", 0.15, None, 0.2, None),  # net +0.05
            SpendRecord("set_mood", "autonomy", 0.05, None, None, None),    # pending
            SpendRecord("recall_memory", "novelty", 0.03, None, 0.0, None),  # flop
        ],
    )
    reg = build_command_registry(appraisal_enabled=True)
    reply = await reg.dispatch("mind", _ctx(drives=store))
    # Per-drive flow suffix (gross fed signed, total spent).
    assert "fed +0.20, spent 0.15" in reply  # connection
    # Per-action P&L lines: settled / pending / flop.
    assert "reach_out: −0.15 connection → +0.20 back, net +0.05" in reply
    assert "set_mood: −0.05 self-expression → pending" in reply
    assert "recall_memory: −0.03 novelty → +0.00 back, net -0.03" in reply


@pytest.mark.asyncio
async def test_forget_yes_also_wipes_drive_store():
    mem = FakeMemory()
    store = FakeDriveStore()
    reply = await build_command_registry().dispatch(
        "forget", _ctx(memory=mem, args="yes", drives=store)
    )
    assert mem.forgotten == ["EpeerD"]
    assert store.forgotten == ["EpeerD"]
    assert "clean slate" in reply.lower()


# --- proactive: /mute, /unmute, /mind section, /forget wipe ----------------


class FakeProactiveStore:
    """Stand-in for ProactiveStore: records mutes/forgets, returns a state."""

    def __init__(self, *, muted_until=None, last_send_at=None):
        from jeff.proactive import ProactiveState

        self._state = ProactiveState("EpeerD", last_send_at, None, muted_until)
        self.muted: list = []
        self.forgotten: list[str] = []

    async def get_state(self, peer):
        return self._state

    async def set_mute(self, peer, until):
        self.muted.append((peer, until))

    async def forget(self, peer):
        self.forgotten.append(peer)
        return 1


def test_mute_unmute_declared_only_when_proactive_enabled():
    assert "mute" not in build_command_registry().names()
    reg = build_command_registry(proactive_enabled=True)
    assert "mute" in reg.names()
    assert "unmute" in reg.names()
    assert "mind" in reg.names()  # /mind lights up on proactive alone too


@pytest.mark.asyncio
async def test_mute_with_duration_sets_future_window():
    from datetime import datetime, timezone

    store = FakeProactiveStore()
    reg = build_command_registry(proactive_enabled=True)
    reply = await reg.dispatch("mute", _ctx(args="2h", proactive=store))
    assert len(store.muted) == 1
    peer, until = store.muted[0]
    assert peer == "EpeerD"
    delta = (until - datetime.now(timezone.utc)).total_seconds()
    assert 7000 < delta < 7300  # ~2 hours out
    assert "mut" in reply.lower()


@pytest.mark.asyncio
async def test_mute_bare_uses_default_window():
    store = FakeProactiveStore()
    reg = build_command_registry(proactive_enabled=True)
    await reg.dispatch("mute", _ctx(args="", proactive=store))
    assert len(store.muted) == 1


@pytest.mark.asyncio
async def test_mute_bad_duration_does_not_mute():
    store = FakeProactiveStore()
    reg = build_command_registry(proactive_enabled=True)
    reply = await reg.dispatch("mute", _ctx(args="wibble", proactive=store))
    assert store.muted == []  # nothing set on a bad duration
    assert "didn't catch" in reply.lower() or "try" in reply.lower()


@pytest.mark.asyncio
async def test_unmute_clears_window():
    store = FakeProactiveStore()
    reg = build_command_registry(proactive_enabled=True)
    await reg.dispatch("unmute", _ctx(proactive=store))
    assert store.muted == [("EpeerD", None)]


@pytest.mark.asyncio
async def test_forget_yes_also_wipes_proactive_store():
    mem = FakeMemory()
    store = FakeProactiveStore()
    await build_command_registry().dispatch(
        "forget", _ctx(memory=mem, args="yes", proactive=store)
    )
    assert store.forgotten == ["EpeerD"]


@pytest.mark.asyncio
async def test_mind_shows_proactive_section():
    from datetime import datetime, timedelta, timezone

    store = FakeProactiveStore(
        muted_until=datetime.now(timezone.utc) + timedelta(hours=1)
    )
    reg = build_command_registry(proactive_enabled=True)
    reply = await reg.dispatch("mind", _ctx(proactive=store))
    assert "proactive:" in reply.lower()
    assert "muted until" in reply.lower()


# --- impulses: /impulses + /mind section + /forget wiring ------------------


def test_impulses_command_declared_only_when_enabled():
    assert build_command_registry().get("impulses") is None
    reg = build_command_registry(impulses_enabled=True)
    assert reg.get("impulses") is not None
    # Impulses alone also surfaces /mind.
    assert reg.get("mind") is not None


@pytest.mark.asyncio
async def test_impulses_command_lists_active_strongest_first():
    store = FakeImpulseStore(
        active=[
            _Imp("test autonomy edges", "try bolder combos", strength=3),
            _Imp("lean playful", "be cheeky", strength=1),
        ]
    )
    reg = build_command_registry(impulses_enabled=True)
    reply = await reg.dispatch("impulses", _ctx(impulses=store))
    assert "test autonomy edges" in reply
    assert "×3" in reply
    assert "[me]" in reply


@pytest.mark.asyncio
async def test_impulses_command_empty_says_none():
    reg = build_command_registry(impulses_enabled=True)
    reply = await reg.dispatch("impulses", _ctx(impulses=FakeImpulseStore()))
    assert "none right now" in reply.lower()


@pytest.mark.asyncio
async def test_mind_shows_impulses_section():
    store = FakeImpulseStore(active=[_Imp("dig deeper", "ask why first", strength=2)])
    reg = build_command_registry(impulses_enabled=True)
    reply = await reg.dispatch("mind", _ctx(impulses=store))
    assert "impulses (1):" in reply
    assert "dig deeper" in reply


@pytest.mark.asyncio
async def test_forget_yes_also_wipes_impulses():
    mem = FakeMemory()
    store = FakeImpulseStore(active=[_Imp("x", "y")])
    await build_command_registry().dispatch(
        "forget", _ctx(memory=mem, args="yes", impulses=store)
    )
    assert store.forgotten == ["EpeerD"]


class _Muse:
    def __init__(self, text, created_at):
        self.text = text
        self.created_at = created_at


class FakeMusingStore:
    def __init__(self, latest=None):
        self._latest = latest
        self.forgotten: list[str] = []

    async def latest(self, peer):
        return self._latest

    async def forget(self, peer):
        self.forgotten.append(peer)
        return 1


def test_mind_declared_when_only_musings_enabled():
    reg = build_command_registry(musings_enabled=True)
    assert "mind" in reg.names()


@pytest.mark.asyncio
async def test_mind_shows_musing_section():
    store = FakeMusingStore(
        _Muse("wondering if they ever fixed that bike", datetime(2026, 6, 9, 14, 0, tzinfo=timezone.utc))
    )
    reg = build_command_registry(musings_enabled=True)
    reply = await reg.dispatch("mind", _ctx(musings=store))
    assert "musing:" in reply
    assert "fixed that bike" in reply


@pytest.mark.asyncio
async def test_mind_musing_section_empty_state():
    reg = build_command_registry(musings_enabled=True)
    reply = await reg.dispatch("mind", _ctx(musings=FakeMusingStore(None)))
    assert "musing:" in reply
    assert "mull things over" in reply


@pytest.mark.asyncio
async def test_forget_yes_also_wipes_musing():
    mem = FakeMemory()
    store = FakeMusingStore(_Muse("a thought", datetime(2026, 6, 9, tzinfo=timezone.utc)))
    await build_command_registry().dispatch(
        "forget", _ctx(memory=mem, args="yes", musings=store)
    )
    assert store.forgotten == ["EpeerD"]
