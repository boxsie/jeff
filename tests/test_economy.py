"""Pure tests for the drive-economy cost table (jeff.economy).

The spend-side store mechanics (DriveState.spend / recent_spends) are exercised
against real Postgres in test_appraisal.py; these cover the constant registry.
"""

from __future__ import annotations

from jeff.appraisal import DRIVES
from jeff.economy import COSTS, REACH_OUT_COST, cost_for


def test_reach_out_not_in_loop_costs():
    # reach_out self-charges inside the tool (conditional on the send). It must
    # NOT also sit in COSTS, or run_tool_loop would double-debit it.
    assert "reach_out" not in COSTS
    assert REACH_OUT_COST == {"connection": 0.15}


def test_cost_for_known_and_unknown():
    assert cost_for("set_impulse") == {"autonomy": 0.08}
    assert cost_for("get_time") == {}  # a free verb
    assert cost_for("not_a_tool") == {}


def test_cost_for_returns_a_fresh_copy():
    c = cost_for("set_mood")
    c["autonomy"] = 999.0
    assert COSTS["set_mood"]["autonomy"] == 0.05  # registry not mutated


def test_all_costs_target_known_drives_and_are_positive():
    keys = {d.key for d in DRIVES}
    for action, costs in {**COSTS, "reach_out": REACH_OUT_COST}.items():
        assert costs, f"{action} has an empty cost"
        for drive, amount in costs.items():
            assert drive in keys, f"{action} debits unknown drive {drive}"
            assert amount > 0.0, f"{action} debits non-positive {drive}"
