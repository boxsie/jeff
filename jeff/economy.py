"""The spend side of the drive economy — the cost table that turns actions into
drive-currency expenditure.

Slice (b1) of the drives-as-currency arc (spitball ``2342241d``, phase
``c6076ef0``). Slice (a) made drive levels an uncapped, leaking **balance** with a
rolling EMA reference (``appraisal.DriveState``). This module adds the **sink**:
a constant registry mapping each spendable action to the drive(s) it costs, so
acting actually *spends* currency instead of leaving the balance to be hoarded.
That's the structural kill for drivemaxxing — the value isn't the balance, it's
spending it well — and it wires drives into the OUTPUT side of the loop (reach
outs, self-turn verbs, impulses) rather than being passive mood-colour.

Two collaborating constants, both keyed by the **tool name** Jeff invokes:

- ``COSTS`` — charged generically at the ``turnloop.run_tool_loop`` dispatch seam:
  whenever Jeff invokes one of these verbs (in a chat turn OR an idle self-turn),
  the loop debits the listed drive(s). This is where "a self-turn verb spends"
  and "acting on an impulse spends" both land, uniformly, with no per-call site.
- ``REACH_OUT_COST`` — the one verb NOT charged at the loop seam, because its
  real-world effect is conditional: ``reach_out`` only spends **connection** when
  the message actually sends. It self-charges inside ``ReachOutTool.run`` (in the
  same best-effort block as the rest of its bookkeeping), so it must be kept OUT
  of ``COSTS`` to avoid a double-debit.

Amounts are small and in the same units as the (now uncapped) levels — tiny for
inward verbs, a meaningful chunk for a reach-out (it satisfies the pent-up
connection want). They're deliberately a flat per-action cost; the **earn-back**
half (slice b2) credits the outcome so a good action nets richer than it cost.
Honest framing carries over: this is bookkeeping over stored scalars, not RL.
"""

from __future__ import annotations


# What a reach-out costs — spent only when the send succeeds, charged inside
# ReachOutTool (NOT at the loop seam), so it is intentionally absent from COSTS.
# Connection is the want a reach-out uses: reaching out spends the pent-up
# "I miss the back-and-forth" deficit.
REACH_OUT_COST: dict[str, float] = {"connection": 0.15}


# Tool name → the drive(s) it debits, charged generically at the run_tool_loop
# dispatch seam (every invocation, chat path or self-turn). Rationale per group:
#
# - Self-authoring verbs (mood + impulses) spend **autonomy** (self-expression):
#   Jeff is steering her own affective state / standing intention, an act of
#   initiative rather than dutiful service. Setting a fresh impulse costs a touch
#   more than nudging an existing one; *clearing* a mood/impulse is letting go,
#   not an act, so it's free.
# - Memory-scan verbs spend **novelty**: going looking through history is inward
#   exploration — Jeff seeking something she doesn't have to hand.
#
# Costs are tiny relative to a turn's appraisal income (±0.3 clamped deltas) so
# Jeff can act several times before a drive runs dry. reach_out is the deliberate
# exception above. Adding/removing a costed verb is one line here, like DRIVES.
COSTS: dict[str, dict[str, float]] = {
    "set_mood": {"autonomy": 0.05},
    "set_impulse": {"autonomy": 0.08},
    "adjust_impulse": {"autonomy": 0.04},
    "recall_memory": {"novelty": 0.03},
    "summarize_recent": {"novelty": 0.04},
}


def cost_for(action: str) -> dict[str, float]:
    """The drive cost of a loop-charged ``action`` (tool name), or ``{}`` if the
    action is free / self-charged. A fresh dict each call so callers can't mutate
    the registry."""
    return dict(COSTS.get(action, {}))
