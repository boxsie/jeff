"""Pure tests for the SelfTurnLoop gating + inward dispatch (no DB, fakes).

Like the proactive loop, the design rests on the loop staying cheap and silent
unless there's something to act on: most ticks must bail on a deterministic gate
WITHOUT calling the model. These pin that, plus the happy path (an inward tool
actually gets dispatched) and fault isolation (a provider blow-up never escapes
the per-peer turn).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import ClassVar

import pytest

from jeff.appraisal import DriveReading
from jeff.chat_types import ChatResult, ToolCall
from jeff.selfturn import (
    _SELF_TURN_INSTRUCTION,
    _SELF_TURN_INSTRUCTION_OUTWARD,
    SelfTurnLoop,
)
from jeff.tools.base import Tool, ToolRegistry


_NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)


# --- fakes ------------------------------------------------------------------


class _ReachStub(Tool):
    """Stand-in named like the outward verb, to flip the instruction selection."""

    name: ClassVar[str] = "reach_out"
    description: ClassVar[str] = "reach out"
    parameters: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}

    async def run(self, **kwargs) -> str:
        return "ok"


class RecordMoodTool(Tool):
    """Stand-in inward verb that records its dispatched kwargs."""

    needs_peer: ClassVar[bool] = True
    name: ClassVar[str] = "set_mood"
    description: ClassVar[str] = "set a mood"
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    def __init__(self):
        self.calls: list[dict] = []

    async def run(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return "mood set"


class FakeProvider:
    """Scripted `complete`: returns each queued ChatResult, then a final answer."""

    def __init__(self, scripted=None, raise_on_call=False):
        self._scripted = list(scripted or [])
        self.raise_on_call = raise_on_call
        self.completes = 0

    async def complete(self, messages, *, model, tools=None):
        self.completes += 1
        if self.raise_on_call:
            raise RuntimeError("provider boom")
        if self._scripted:
            return self._scripted.pop(0)
        return ChatResult(content="done, nothing to do")


class FakeMemory:
    async def recent(self, peer, n=10):
        return []


class FakeCuriosity:
    def __init__(self, texts):
        self._texts = texts

    async def open_curiosities(self, peer, limit):
        return [SimpleNamespace(id=i + 1, text=t) for i, t in enumerate(self._texts)]


class FakeImpulses:
    def __init__(self, items):
        self._items = items

    async def list_active(self, peer):
        return [SimpleNamespace(name=n, description=d) for n, d in self._items]


class FakeDrives:
    def __init__(self, levels, references=None):
        self._levels = levels
        self._refs = references or {}
        self.spends: list[tuple] = []

    async def state(self, peer):
        # Reference defaults to 0.0 → a non-zero level reads as off its norm,
        # which is what the off-reference gate keys on. Pass `references` to put a
        # drive AT its norm (level == reference → unremarkable).
        return {
            k: DriveReading(v, self._refs.get(k, 0.0))
            for k, v in self._levels.items()
        }

    async def spend(self, peer, action, costs):
        self.spends.append((peer, action, dict(costs)))
        return {}


class FakeReflection:
    async def persona(self, peer, max_chars):
        return ([], [])


class FakeMood:
    async def active_mood(self, peer):
        return None


def _cfg(**extra):
    base = dict(
        self_turn_interval_s=900.0,
        self_turn_min_gap_s=3600.0,
        curiosity_max_open=8,
        recent_turns=10,
        persona_max_chars=2000,
        system_prompt="You are Jeff.",
        drives_max_chars=2000,
        impulses_max_chars=2000,
        chat_model="grok",
        max_tool_iters=4,
        tool_timeout_s=30.0,
    )
    base.update(extra)
    return SimpleNamespace(**base)


def _loop(
    *,
    registry,
    provider,
    curiosity=None,
    impulses=None,
    drives=None,
    cfg=None,
):
    return SelfTurnLoop(
        FakeMemory(),
        registry,
        provider,
        curiosity_store=curiosity,
        reflection_store=FakeReflection(),
        mood_store=FakeMood(),
        drive_store=drives,
        impulse_store=impulses,
        cfg=cfg or _cfg(),
        allowlist=["EpeerD"],
    )


# --- gate tests -------------------------------------------------------------


@pytest.mark.asyncio
async def test_skip_when_nothing_to_chew_on():
    prov = FakeProvider()
    loop = _loop(
        registry=ToolRegistry([RecordMoodTool()]),
        provider=prov,
        curiosity=FakeCuriosity([]),
        impulses=FakeImpulses([]),
        drives=FakeDrives({}),  # no drive rows → nothing off its norm
    )
    await loop._maybe_self_turn("EpeerD", _NOW)
    assert prov.completes == 0
    assert "EpeerD" not in loop._last


@pytest.mark.asyncio
async def test_skip_within_min_gap():
    prov = FakeProvider()
    loop = _loop(
        registry=ToolRegistry([RecordMoodTool()]),
        provider=prov,
        impulses=FakeImpulses([("poke", "look into X")]),  # something to chew on
    )
    loop._last["EpeerD"] = _NOW - timedelta(seconds=60)  # well within 3600 gap
    await loop._maybe_self_turn("EpeerD", _NOW)
    assert prov.completes == 0


@pytest.mark.asyncio
async def test_skip_when_no_verbs():
    prov = FakeProvider()
    loop = _loop(
        registry=ToolRegistry([]),  # empty inward registry
        provider=prov,
        impulses=FakeImpulses([("poke", "look into X")]),
    )
    await loop._maybe_self_turn("EpeerD", _NOW)
    assert prov.completes == 0


# --- happy path -------------------------------------------------------------


@pytest.mark.asyncio
async def test_runs_and_dispatches_inward_tool():
    tool = RecordMoodTool()
    prov = FakeProvider(
        scripted=[
            ChatResult(
                content="",
                tool_calls=(ToolCall(id="c1", name="set_mood", arguments='{"name": "restless"}'),),
            ),
            ChatResult(content="settled in"),
        ]
    )
    drives = FakeDrives({"connection": 0.2})
    loop = _loop(
        registry=ToolRegistry([tool]),
        provider=prov,
        impulses=FakeImpulses([("poke", "look into X")]),
        drives=drives,
    )
    await loop._maybe_self_turn("EpeerD", _NOW)
    assert prov.completes == 2
    # The inward tool ran, with the real peer injected (needs_peer).
    assert len(tool.calls) == 1
    assert tool.calls[0]["peer"] == "EpeerD"
    assert tool.calls[0]["name"] == "restless"
    # Running the set_mood verb spent its drive currency (economy.COSTS) — the
    # self-turn-verb-spends edge, charged at the run_tool_loop dispatch seam.
    assert drives.spends == [("EpeerD", "set_mood", {"autonomy": 0.05})]
    # Min-gap advanced after the turn was spent.
    assert loop._last["EpeerD"] == _NOW


@pytest.mark.asyncio
async def test_runs_even_when_drive_off_reference_only():
    # No impulses, no curiosities — but a drive sits well off its norm → chew.
    prov = FakeProvider()  # model decides to do nothing (no tool calls)
    loop = _loop(
        registry=ToolRegistry([RecordMoodTool()]),
        provider=prov,
        curiosity=FakeCuriosity([]),
        impulses=FakeImpulses([]),
        drives=FakeDrives({"connection": 0.95}),  # level 0.95 vs ref 0.0 → way off
    )
    await loop._maybe_self_turn("EpeerD", _NOW)
    assert prov.completes == 1
    assert loop._last["EpeerD"] == _NOW


# --- fault isolation --------------------------------------------------------


def test_instruction_switches_on_reach_out_presence():
    inward = _loop(registry=ToolRegistry([RecordMoodTool()]), provider=FakeProvider())
    assert inward._instruction is _SELF_TURN_INSTRUCTION
    outward = _loop(
        registry=ToolRegistry([RecordMoodTool(), _ReachStub()]),
        provider=FakeProvider(),
    )
    assert outward._instruction is _SELF_TURN_INSTRUCTION_OUTWARD


@pytest.mark.asyncio
async def test_provider_fault_is_swallowed():
    prov = FakeProvider(raise_on_call=True)
    loop = _loop(
        registry=ToolRegistry([RecordMoodTool()]),
        provider=prov,
        impulses=FakeImpulses([("poke", "X")]),
    )
    # Must not raise out of the per-peer turn.
    await loop._maybe_self_turn("EpeerD", _NOW)
    assert "EpeerD" not in loop._last  # not advanced on failure
