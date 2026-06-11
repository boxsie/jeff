"""Chat commands — deterministic control actions declared to the Ensemble daemon.

Jeff **declares** its commands at registration (`ServiceHandle` via
`register(commands=[...])`) and **receives** invocations as the daemon's
`CommandInvocation` events — the daemon owns parsing and routing now, so there's
no text-sniffing here. A command and its reply are control traffic: they never
touch memory's recall window and never reach the model.

The daemon supports **augment dispatch**: a command both it and Jeff handle runs
*both* legs (the built-in always runs and can't be suppressed; Jeff only adds
behaviour). That's why `/clear` works as one keystroke — the daemon's built-in
clears the operator's local transcript while Jeff's `/clear` resets the working
window of its memory. Jeff therefore declares only what it uniquely owns:

  - `/clear`  — start a fresh conversational *thread*: resets the recent window
                only. Long-term semantic recall still spans every session, so
                older things can resurface naturally (Jeff remembers the whole
                relationship). Augments the daemon transcript-clear; the hard
                wipe is `/forget`.
  - `/forget` — hard wipe of everything Jeff remembers about this peer (confirm-gated).
  - `/stats`  — memory counts, uptime, active provider/model, prompt source.
  - `/debug`  — deterministic introspection: dump the real working context
                (effective system prompt, session cutoff, recent window, and
                exactly what recall would surface with cosine distances).

`/help`/`/whoami` are ceded to the daemon's built-ins; the old soft `/new` is
subsumed by the augmented `/clear`.

`dispatch` is safe-by-construction, mirroring `tools.base.ToolRegistry`: a
handler that raises becomes a short, content-safe apology rather than an
exception that escapes the event loop. Single-user threat model (ACL = operator
only) means there's no per-command permission split.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:  # avoid import cost / cycles at runtime; these are type-only
    import ensemble

    from .appraisal import DriveState
    from .config import Config
    from .curiosity import CuriosityStore
    from .impulses import ImpulseStore
    from .memory import Memory, Message
    from .mood import MoodStore
    from .musings import MusingStore
    from .pinned import PinnedMemoryStore
    from .proactive import ProactiveStore
    from .reflection import ReflectionStore


log = logging.getLogger("jeff.commands")

# Captured at import (≈ process start) so /stats can report uptime without
# threading a start time through every turn. Monotonic so it's immune to wall
# clock changes.
_PROCESS_START = time.monotonic()


@dataclass
class CommandContext:
    """Everything a command handler might need to act and to render a reply.

    `args` is the text after the command name (the daemon's `CommandInvocation`
    already stripped the name + prefix). `peer` is the invoking address. `handle`
    isn't used by the current commands but is here for future commands that send
    their own traffic (e.g. a `/backup` that offers a file over Ensemble).

    `system_prompt` is the *effective* prompt actually sent on a turn (operator
    base + capabilities addendum), composed once at startup; `tool_names` is the
    registered tool set. Both are here so `/debug` can show the real working
    context rather than reconstructing it. `curiosity` is the open-questions store
    (None when the curiosity drive is off), `reflection` is the derived persona
    store (None when reflection is off), `mood` is the affective-state store
    (None when the mood drive is off), and `pinned` is the explicit/pinned-memory
    store (None when the remember drive is off) so `/mind`/`/mood`/`/remember` can
    show/write them and `/forget` can wipe them. They default to empty/None so
    callers that don't care (and older tests) need not supply them.
    """

    handle: "ensemble.ServiceHandle"
    memory: "Memory"
    cfg: "Config"
    peer: str
    args: str
    system_prompt: str = ""
    tool_names: tuple[str, ...] = ()
    curiosity: "CuriosityStore | None" = None
    reflection: "ReflectionStore | None" = None
    mood: "MoodStore | None" = None
    pinned: "PinnedMemoryStore | None" = None
    drives: "DriveState | None" = None
    proactive: "ProactiveStore | None" = None
    impulses: "ImpulseStore | None" = None
    musings: "MusingStore | None" = None


CommandHandler = Callable[[CommandContext], Awaitable[str]]


@dataclass(frozen=True)
class Command:
    """One declared command: bare name (no prefix), help text, arg hint, handler.

    `name` must be slug-safe lowercase (the daemon matches case-insensitively and
    rejects non-slug names). `description` is the one-line `/help` text; `usage`
    is an optional arg hint (e.g. `"yes"`). `name`/`description`/`usage` are the
    single source for the specs declared to the daemon (`to_ensemble_commands`).
    """

    name: str
    description: str
    handler: CommandHandler
    usage: str = ""


class CommandRegistry:
    """Holds the active commands, declares them, and dispatches safely."""

    def __init__(self, commands: list[Command] | None = None):
        self._commands: dict[str, Command] = {}
        for c in commands or []:
            self.register(c)

    def register(self, cmd: Command) -> None:
        if not cmd.name:
            raise ValueError("command must have a non-empty name")
        if cmd.name in self._commands:
            raise ValueError(f"duplicate command name: {cmd.name!r}")
        self._commands[cmd.name] = cmd

    def __len__(self) -> int:
        return len(self._commands)

    def names(self) -> list[str]:
        return sorted(self._commands)

    def get(self, name: str) -> Command | None:
        return self._commands.get(name)

    def to_ensemble_commands(self) -> list["ensemble.Command"]:
        """The specs Jeff declares at registration, single-sourced from the
        registry so help text can't drift from behaviour."""
        import ensemble  # hard dep of jeff; lazy to keep this module import-light

        return [
            ensemble.Command(name=c.name, description=c.description, usage=c.usage)
            for c in (self._commands[n] for n in self.names())
        ]

    async def dispatch(self, name: str, ctx: CommandContext) -> str:
        """Run command `name`; always return a content-safe reply string.

        The daemon only routes commands Jeff declared, so an unknown name here is
        effectively unreachable — but it still yields a friendly hint rather than
        an error. A handler that raises is logged (type only) and yields a generic
        apology, never an exception string (which could carry endpoint/peer
        detail — same discipline as the turn loop + tool dispatch).
        """
        cmd = self._commands.get(name)
        if cmd is None:
            log.info("unknown command=%s", name)
            return f"Unknown command /{name}."
        try:
            return await cmd.handler(ctx)
        except Exception:
            log.exception("command %s raised", name)
            return "Sorry — that command hit a snag on my end. Try again in a moment."


# --- command handlers ------------------------------------------------------


async def _cmd_clear(ctx: CommandContext) -> str:
    """Fresh conversational thread: advance the history cutoff, which resets the
    `recent` window only. Long-term semantic recall deliberately keeps spanning
    every session, so older things can still surface naturally — `/clear` resets
    what we're actively talking about, not the whole relationship. `/forget` is
    the hard wipe.

    This is the service leg of the daemon's `/clear` augment — the daemon clears
    the operator's local transcript; this resets Jeff's active thread."""
    await ctx.memory.set_history_cutoff(ctx.peer)
    return (
        "Fresh conversation — I've reset what we're actively on, but I still "
        "remember our history, so older stuff can come back up naturally. Use "
        "`/forget` if you really want a clean slate."
    )


async def _cmd_forget(ctx: CommandContext) -> str:
    """Hard wipe: irreversible DELETE of every stored message for this peer.

    Confirm-gated (`/forget yes`) on purpose — it's irreversible and a stray
    `/forget` is easy to fat-finger. The CLI `forget` subcommand stays
    no-confirm (it's an explicit admin action); chat gets the guardrail. Wipes
    only Jeff's own memory — no attempt to reach any daemon transcript.
    """
    if ctx.args.strip().lower() != "yes":
        return (
            "⚠️ This permanently deletes everything I remember about our "
            "conversations — there's no undo. If you're sure, send `/forget yes`."
        )
    deleted = await ctx.memory.forget(ctx.peer)
    # Wipe the derived stores too (when those drives are on) so a clean slate
    # really is clean — open questions and the persona are both distilled from
    # the same conversations.
    if ctx.curiosity is not None:
        await ctx.curiosity.forget(ctx.peer)
    if ctx.reflection is not None:
        await ctx.reflection.forget(ctx.peer)
    # Moods + their definitions are part of the same relationship; wipe them too.
    if ctx.mood is not None:
        await ctx.mood.forget(ctx.peer)
    # Deliberately-pinned memories are wiped too — a clean slate is clean.
    if ctx.pinned is not None:
        await ctx.pinned.forget(ctx.peer)
    # Drive levels are part of the relationship too (they're nudged by these very
    # conversations); wipe them so they reset to baseline on a clean slate.
    if ctx.drives is not None:
        await ctx.drives.forget(ctx.peer)
    # Proactive bookkeeping (last reach-out, mute, dedup key) is per-relationship
    # too — wipe it so a clean slate starts the reach-out cadence fresh.
    if ctx.proactive is not None:
        await ctx.proactive.forget(ctx.peer)
    # Self-set impulses are directions Jeff chose within this relationship — wipe
    # them too so a clean slate carries no leftover steering.
    if ctx.impulses is not None:
        await ctx.impulses.forget(ctx.peer)
    # The carried-over idle musing is a thought from these conversations — wipe it
    # so a clean slate doesn't surface a stale "what you've been mulling".
    if ctx.musings is not None:
        await ctx.musings.forget(ctx.peer)
    return f"Wiped {deleted} stored message(s) — clean slate."


async def _cmd_stats(ctx: CommandContext) -> str:
    """Counts + uptime + active provider/model + prompt source. No DSN / key /
    seed / endpoint. Absorbs what the old `/whoami` reported."""
    mine = await ctx.memory.count(ctx.peer)
    total = await ctx.memory.total()
    uptime = _format_uptime(time.monotonic() - _PROCESS_START)
    return (
        "**Stats**\n"
        f"- stored messages (you): {mine}\n"
        f"- stored messages (all peers): {total}\n"
        f"- uptime: {uptime}\n"
        f"- provider: {ctx.cfg.llm_provider}\n"
        f"- model: {ctx.cfg.chat_model}\n"
        f"- system prompt source: {ctx.cfg.system_prompt_source}"
    )


def _format_uptime(seconds: float) -> str:
    """Render an elapsed-seconds duration as a compact `1d 2h 3m 4s` string."""
    s = int(max(0, seconds))
    days, s = divmod(s, 86400)
    hours, s = divmod(s, 3600)
    minutes, s = divmod(s, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{s}s")
    return " ".join(parts)


# --- /debug: deterministic introspection ----------------------------------
#
# /debug prints Jeff's *real* working context — the system prompt, the session
# cutoff, the recent window, and exactly what recall would surface (with cosine
# distances). It's a deterministic dump, never a model call, so it can't
# confabulate: it shows the data structures themselves. This is the truthful
# answer to "what are you working with?" that the model can't reliably give
# about its own context. Single-user threat model (peer == operator) means
# exposing the prompt/messages here is fine — the operator owns both.

_DEBUG_CONTENT_WIDTH = 100  # per-row content truncation in the dumps


def _truncate(text: str, width: int = _DEBUG_CONTENT_WIDTH) -> str:
    """One-line, length-bounded rendering of stored content for the dumps."""
    flat = " ".join(text.split())  # collapse newlines/runs so rows stay one line
    if len(flat) <= width:
        return flat
    return flat[: width - 1] + "…"


def _fmt_ts(ts) -> str:
    """Compact, second-resolution timestamp for the dumps."""
    return ts.isoformat(sep=" ", timespec="seconds")


def _fmt_msg_row(prefix: str, m: "Message") -> str:
    # role padded to the widest value ("assistant" = 9) so columns line up.
    return f"{prefix}#{m.id} {m.role:<9} {_fmt_ts(m.ts)}  {_truncate(m.content)}"


async def _debug_overview(ctx: CommandContext) -> str:
    cutoff = await ctx.memory.get_history_cutoff(ctx.peer)
    mine = await ctx.memory.count(ctx.peer)
    total = await ctx.memory.total()
    recent = await ctx.memory.recent(ctx.peer, n=ctx.cfg.recent_turns)
    prompt = ctx.system_prompt or ctx.cfg.system_prompt

    lines = [
        "session cutoff: "
        + (_fmt_ts(cutoff) if cutoff else "(none — full history in window)"),
        f"stored msgs:    you={mine}  all peers={total}",
        f"knobs:          recent_turns={ctx.cfg.recent_turns}  recall_k={ctx.cfg.recall_k}"
        f"  recall_dist<={ctx.cfg.recall_distance_max}",
        f"system prompt:  {len(prompt)} chars  (source={ctx.cfg.system_prompt_source})",
        "tools:          " + (", ".join(ctx.tool_names) if ctx.tool_names else "(none)"),
        f"recent window ({len(recent)} / max {ctx.cfg.recent_turns}):",
    ]
    if recent:
        lines.extend(_fmt_msg_row("  ", m) for m in recent)
    else:
        lines.append("  (empty — fresh session)")

    body = "\n".join(lines)
    return (
        "**debug — context**\n```\n" + body + "\n```\n"
        "`/debug prompt` for the full system prompt · "
        "`/debug recall <query>` to test what I'd recall."
    )


def _debug_prompt(ctx: CommandContext) -> str:
    prompt = ctx.system_prompt or ctx.cfg.system_prompt
    return (
        f"**debug — effective system prompt** "
        f"(source={ctx.cfg.system_prompt_source}, {len(prompt)} chars)\n"
        "```\n" + prompt + "\n```"
    )


async def _debug_recall(ctx: CommandContext, query: str) -> str:
    # Pull a few extra beyond recall_k so near-misses just over the threshold are
    # visible (the whole point of a tuning view). The live recall() keeps the
    # first recall_k rows with dist <= DEFAULT_RECALL_DISTANCE_MAX; because rows
    # are distance-ordered, those are exactly the leading ✓ rows below.
    threshold = ctx.cfg.recall_distance_max
    scored = await ctx.memory.recall_scored(
        ctx.peer, query, limit=ctx.cfg.recall_k + 3
    )
    lines = [
        f'query: "{_truncate(query, 80)}"',
        f"kept by a real turn: ✓ = dist <= {threshold} and within recall_k={ctx.cfg.recall_k}",
    ]
    if not scored:
        lines.append("(no stored messages for this peer yet)")
    for i, (m, dist) in enumerate(scored):
        kept = dist <= threshold and i < ctx.cfg.recall_k
        mark = "✓" if kept else " "
        lines.append(f"{mark} {dist:.3f}  " + _fmt_msg_row("", m))
    body = "\n".join(lines)
    return "**debug — recall**\n```\n" + body + "\n```"


async def _cmd_debug(ctx: CommandContext) -> str:
    """Deterministic introspection: show the real context Jeff is working with.

    Subcommands: bare `/debug` (overview), `/debug prompt` (full system prompt),
    `/debug recall <query>` (what recall would surface, with distances).
    """
    sub = ctx.args.strip()
    low = sub.lower()
    if not sub:
        return await _debug_overview(ctx)
    if low == "prompt":
        return _debug_prompt(ctx)
    if low == "recall" or low.startswith("recall "):
        query = sub[len("recall"):].strip()
        if not query:
            return (
                "Usage: `/debug recall <query>` — shows what I'd recall for that "
                "text, with cosine distances."
            )
        return await _debug_recall(ctx, query)
    return (
        "Unknown debug view. Try `/debug` (overview), `/debug prompt`, or "
        "`/debug recall <query>`."
    )


# --- /mind: the curiosity drive's introspection view ----------------------


def _format_remaining(seconds: float) -> str:
    """A coarse '~3h 20m left' rendering for a mood's remaining lifetime."""
    s = int(max(0, seconds))
    hours, s = divmod(s, 3600)
    minutes, _ = divmod(s, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts) + " left"


def _drive_band(level: float, reference: float) -> str:
    """One-word band for a drive level in the `/mind` dump, judged relative to the
    drive's rolling EMA reference — its personal baseline (matches the prompt
    block's ``DRIVE_BAND_MARGIN`` so the operator view and what Jeff sees stay
    aligned)."""
    from .prompt import DRIVE_BAND_MARGIN

    if level >= reference + DRIVE_BAND_MARGIN:
        return "well-met"
    if level <= reference - DRIVE_BAND_MARGIN:
        return "running low"
    return "steady"


# How many recent income/spend events the `/mind` economy view summarises — a
# legible window, not the whole append-only ledger (which grows per turn).
_MIND_ECONOMY_LIMIT = 8


def _spend_outcome(rec) -> str:
    """Render a spend row's earn-back settlement for the `/mind` P&L line: still
    pending (no payback yet), or settled with the credit earned and the net
    (credit − cost). A flop reads as '+0.00 back, net −cost' — honest."""
    if rec.credit is None:
        return "pending"
    return f"{rec.credit:+.2f} back, net {rec.net:+.2f}"


async def _drives_section(ctx: CommandContext) -> str:
    """The drives + economy block of `/mind`: per-drive balance/band/avg, the
    recent feed and spend per drive, and a per-action P&L list. Read-only and
    content-safe — only numbers, action names, and drive nouns (no peer text)."""
    from .appraisal import DRIVES

    reading = await ctx.drives.state(ctx.peer)
    income = await ctx.drives.recent_income(ctx.peer, limit=_MIND_ECONOMY_LIMIT)
    spends = await ctx.drives.recent_spends(ctx.peer, limit=_MIND_ECONOMY_LIMIT)
    nouns = {d.key: d.noun for d in DRIVES}

    # Aggregate the windowed flow per drive: gross fed (signed) and total spent.
    fed: dict[str, float] = {}
    for r in income:
        fed[r.drive] = fed.get(r.drive, 0.0) + r.amount
    spent: dict[str, float] = {}
    for r in spends:
        spent[r.drive] = spent.get(r.drive, 0.0) + r.amount

    lines = ["drives:"]
    # Each drive's leaked-to-now balance, its rolling reference (the personal
    # baseline it's judged against), and a one-word band — plus the recent flow
    # when there's been any, so the operator sees both the balance the appraisal
    # is steering and what's been moving it. Uncapped, so a number can exceed 1.
    for d in DRIVES:
        r = reading[d.key]
        line = (
            f"  • {d.noun}: {r.level:.2f} "
            f"({_drive_band(r.level, r.reference)}, avg {r.reference:.2f})"
        )
        if d.key in fed or d.key in spent:
            line += f" — fed {fed.get(d.key, 0.0):+.2f}, spent {spent.get(d.key, 0.0):.2f}"
        lines.append(line)

    # Per-action P&L: what each recent spend cost and whether it earned back.
    if spends:
        lines.append(f"recent actions ({len(spends)}):")
        lines.extend(
            f"  • {r.action}: −{r.amount:.2f} {nouns.get(r.drive, r.drive)} "
            f"→ {_spend_outcome(r)}"
            for r in spends
        )
    else:
        lines.append("recent actions: (nothing spent yet)")

    return "\n".join(lines)


async def _mood_line(ctx: CommandContext) -> str:
    """One-line description of the active mood (or neutral), shared by `/mood`
    and the `/mind` mood section. Computes 'time left' from the DB expiry against
    wall-clock now() — coarse minutes are plenty for a status line."""
    from datetime import datetime, timezone

    active = await ctx.mood.active_mood(ctx.peer)
    if active is None:
        return "neutral (no active mood)"
    remaining = (active.expires_at - datetime.now(timezone.utc)).total_seconds()
    return f"{active.name} — {_format_remaining(remaining)}"


async def _cmd_mood(ctx: CommandContext) -> str:
    """Show Jeff's current mood and how long it has left (or 'neutral'). Declared
    only when the mood drive is on. Deterministic dump, no model call."""
    if ctx.mood is None:
        return "Moods aren't switched on right now."
    line = await _mood_line(ctx)
    defined = await ctx.mood.count_definitions(ctx.peer)
    return (
        f"**mood**\n```\n{line}\n"
        f"defined moods: {defined}\n```"
    )


def _impulse_ttl(impulse) -> str:
    """A ' · ~2h 10m left' suffix for a time-boxed impulse, '' when permanent.

    Permanent impulses (no expiry) show nothing — the absence of a timer is the
    signal. Shared by `/mind` and `/impulses`."""
    if impulse.expires_at is None:
        return ""
    from datetime import datetime, timezone

    remaining = (impulse.expires_at - datetime.now(timezone.utc)).total_seconds()
    return " · " + _format_remaining(remaining)


async def _cmd_impulses(ctx: CommandContext) -> str:
    """List Jeff's active self-set impulses (strongest first). Declared only when
    impulses are on. Deterministic dump, no model call."""
    if ctx.impulses is None:
        return "Impulses aren't switched on right now."
    active = await ctx.impulses.list_active(ctx.peer)
    if not active:
        return (
            "**impulses**\n```\nnone right now — I set these to steer myself.\n```"
        )
    lines = [
        f"• [{'you' if i.source == 'operator' else 'me'}] "
        f"{i.name} (×{i.strength}{_impulse_ttl(i)}): {_truncate(i.description)}"
        for i in active
    ]
    return "**impulses**\n```\n" + "\n".join(lines) + "\n```"


async def _cmd_remember(ctx: CommandContext) -> str:
    """Operator-facing pin: `/remember <text>` writes a note to the shared pinned
    store (same store the `remember` tool writes to), tagged source=operator.
    Declared only when the remember drive is on. Pure DB — no model call."""
    if ctx.pinned is None:
        return "Explicit memory isn't switched on right now."
    text = ctx.args.strip()
    if not text:
        return (
            "Usage: `/remember <something>` — I'll pin it to my long-term memory "
            "so it stays in mind. Use `/mind` to see what's pinned."
        )
    if len(text) > ctx.cfg.remember_max_chars:
        return (
            f"That's a bit long to pin (max {ctx.cfg.remember_max_chars} "
            "characters) — try a more concise note."
        )
    from .pinned import SOURCE_OPERATOR

    pid = await ctx.pinned.add(ctx.peer, text, source=SOURCE_OPERATOR)
    if pid is None:
        return "I already had that pinned — nothing to add."
    return "Pinned — I'll keep that in mind."


_MUTE_DEFAULT_HOURS = 8


def _parse_duration(raw: str):
    """Parse a compact `<n><unit>` duration (s/m/h/d) into a timedelta, or None
    if it doesn't parse. Used by `/mute`."""
    import re
    from datetime import timedelta

    m = re.fullmatch(r"\s*(\d+)\s*([smhd])\s*", raw.lower())
    if not m:
        return None
    n = int(m.group(1))
    if n <= 0:
        return None
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    return timedelta(seconds=n * mult)


async def _cmd_mute(ctx: CommandContext) -> str:
    """Silence proactive (unprompted) reach-outs for a while. Bare `/mute` uses a
    default window; `/mute 2h` / `/mute 30m` / `/mute 1d` set an explicit one.
    Reactive replies are unaffected — Jeff still answers when you message."""
    if ctx.proactive is None:
        return "Proactive messaging isn't switched on, so there's nothing to mute."
    from datetime import datetime, timedelta, timezone

    arg = ctx.args.strip()
    dur = _parse_duration(arg) if arg else timedelta(hours=_MUTE_DEFAULT_HOURS)
    if dur is None:
        return (
            "I didn't catch that duration — try `/mute 2h`, `/mute 30m`, `/mute 1d`, "
            "or just `/mute` for a few hours."
        )
    until = datetime.now(timezone.utc) + dur
    await ctx.proactive.set_mute(ctx.peer, until)
    return (
        f"Muted — I won't message you on my own until about "
        f"{until.isoformat(timespec='minutes')}. `/unmute` to lift it early."
    )


async def _cmd_unmute(ctx: CommandContext) -> str:
    """Lift a `/mute` so Jeff can reach out unprompted again."""
    if ctx.proactive is None:
        return "Proactive messaging isn't switched on."
    await ctx.proactive.set_mute(ctx.peer, None)
    return "Unmuted — I might reach out when something's genuinely on my mind."


async def _cmd_mind(ctx: CommandContext) -> str:
    """Show what's on Jeff's mind: open questions it wants to ask (curiosity), the
    facts + opinions it has formed (reflection), its current mood, and what it has
    deliberately pinned (remember). A deterministic dump (no model call), sibling
    to `/debug`. Declared whenever any of those drives is enabled; each section
    appears only when its store is present.
    """
    if (
        ctx.curiosity is None
        and ctx.reflection is None
        and ctx.mood is None
        and ctx.pinned is None
        and ctx.drives is None
        and ctx.proactive is None
        and ctx.impulses is None
        and ctx.musings is None
    ):
        return (
            "None of curiosity, reflection, moods, pinned memory, drives, "
            "impulses, musings, or proactive messaging is switched on right now, "
            "so there's nothing on my mind to show."
        )

    sections: list[str] = []

    if ctx.pinned is not None:
        pins = await ctx.pinned.list(ctx.peer, limit=ctx.cfg.remember_max_items)
        lines = [f"pinned memory ({len(pins)}):"]
        if pins:
            # Show provenance: who chose to keep it (you vs me).
            lines.extend(
                f"  • [{'you' if p.source == 'operator' else 'me'}] {_truncate(p.text)}"
                for p in pins
            )
        else:
            lines.append("  (nothing pinned yet — ask me to remember something)")
        sections.append("\n".join(lines))

    if ctx.mood is not None:
        line = await _mood_line(ctx)
        defs = await ctx.mood.list_definitions(ctx.peer)
        lines = [f"mood: {line}"]
        if defs:
            lines.append(f"moods you've defined ({len(defs)}):")
            lines.extend(f"  • {d.name}: {_truncate(d.description)}" for d in defs)
        else:
            lines.append("  (no moods defined yet — we author them together)")
        sections.append("\n".join(lines))

    if ctx.drives is not None:
        sections.append(await _drives_section(ctx))

    if ctx.impulses is not None:
        active = await ctx.impulses.list_active(ctx.peer)
        lines = [f"impulses ({len(active)}):"]
        if active:
            # Strongest first (list_active orders them). Show strength, source
            # (me/you), and remaining time when time-boxed — these are Jeff's own
            # directions, distinct from the standing drives above.
            lines.extend(
                f"  • [{'you' if i.source == 'operator' else 'me'}] "
                f"{i.name} (×{i.strength}{_impulse_ttl(i)}): {_truncate(i.description)}"
                for i in active
            )
        else:
            lines.append("  (none right now — I set these to steer myself)")
        sections.append("\n".join(lines))

    if ctx.proactive is not None:
        from datetime import datetime, timezone

        st = await ctx.proactive.get_state(ctx.peer)
        now = datetime.now(timezone.utc)
        lines = ["proactive:"]
        if st.muted_until is not None and st.muted_until > now:
            lines.append(
                f"  • muted until {st.muted_until.isoformat(timespec='minutes')}"
            )
        else:
            lines.append("  • not muted")
        if st.last_send_at is not None:
            lines.append(
                f"  • last reached out: "
                f"{st.last_send_at.isoformat(timespec='minutes')}"
            )
        else:
            lines.append("  • haven't reached out on my own yet")
        sections.append("\n".join(lines))

    if ctx.musings is not None:
        # The idle thought Jeff carried out of her last quiet moment — the thing
        # that surfaces on the next reactive turn (while it's fresher than the
        # last thing said). Inherently produced while you're away; this is how
        # you peek at it on demand once you're back.
        m = await ctx.musings.latest(ctx.peer)
        lines = ["musing:"]
        if m is not None:
            when = m.created_at.isoformat(timespec="minutes")
            lines.append(f"  • (from {when}) {_truncate(m.text)}")
        else:
            lines.append("  (nothing yet — I mull things over in my idle moments)")
        sections.append("\n".join(lines))

    if ctx.curiosity is not None:
        open_cur = await ctx.curiosity.open_curiosities(ctx.peer, limit=20)
        satisfied = await ctx.curiosity.recently_satisfied(ctx.peer, limit=5)
        lines = [f"open questions ({len(open_cur)}):"]
        if open_cur:
            lines.extend(f"  • {_truncate(c.text)}" for c in open_cur)
        else:
            lines.append("  (nothing yet — I get curious as we talk)")
        if satisfied:
            lines.append(f"recently answered ({len(satisfied)}):")
            lines.extend(f"  ✓ {_truncate(c.text)}" for c in satisfied)
        sections.append("\n".join(lines))

    if ctx.reflection is not None:
        from .reflection import FACT, REFLECTION

        facts = await ctx.reflection.fetch(ctx.peer, kind=FACT, limit=20)
        opinions = await ctx.reflection.fetch(ctx.peer, kind=REFLECTION, limit=20)
        lines = [f"what I know about you ({len(facts)}):"]
        if facts:
            lines.extend(f"  • {_truncate(d.text)}" for d in facts)
        else:
            lines.append("  (nothing yet — I learn as we talk)")
        lines.append(f"my own take ({len(opinions)}):")
        if opinions:
            lines.extend(f"  • {_truncate(d.text)}" for d in opinions)
        else:
            lines.append("  (no opinions formed yet)")
        sections.append("\n".join(lines))

    body = "\n\n".join(sections)
    return "**on my mind**\n```\n" + body + "\n```"


def build_command_registry(
    *,
    curiosity_enabled: bool = False,
    reflection_enabled: bool = False,
    mood_enabled: bool = False,
    remember_enabled: bool = False,
    appraisal_enabled: bool = False,
    proactive_enabled: bool = False,
    impulses_enabled: bool = False,
    musings_enabled: bool = False,
) -> CommandRegistry:
    """Jeff's declared command set (see main.run). `/help`/`/whoami` are the
    daemon's built-ins; the old `/new` is subsumed by the augmented `/clear`.

    `/mind` is declared when any of curiosity / reflection / mood / remember /
    appraisal / proactive / impulses / musings is on; `/mood` only when the mood drive is
    on; `/impulses` only when impulses are on; `/remember` only when the remember
    drive is on; `/mute`+`/unmute` only when proactive messaging is on — keeping
    the feature-off path's declared command set unchanged."""
    cmds = [
        Command(
            "clear",
            "start a fresh conversation (keeps long-term memory)",
            _cmd_clear,
        ),
        Command(
            "forget",
            "permanently wipe everything I remember",
            _cmd_forget,
            usage="yes",
        ),
        Command("stats", "memory counts, uptime, and active model", _cmd_stats),
        Command(
            "debug",
            "inspect my working context (prompt, recent window, recall)",
            _cmd_debug,
            usage="[prompt|recall <query>]",
        ),
    ]
    if (
        curiosity_enabled
        or reflection_enabled
        or mood_enabled
        or remember_enabled
        or appraisal_enabled
        or proactive_enabled
        or impulses_enabled
        or musings_enabled
    ):
        cmds.append(
            Command("mind", "show what's on my mind right now", _cmd_mind)
        )
    if mood_enabled:
        cmds.append(
            Command("mood", "show my current mood and how long it has left", _cmd_mood)
        )
    if impulses_enabled:
        cmds.append(
            Command("impulses", "show the directions I'm steering myself in", _cmd_impulses)
        )
    if remember_enabled:
        cmds.append(
            Command(
                "remember",
                "pin something to my long-term memory",
                _cmd_remember,
                usage="<something to remember>",
            )
        )
    if proactive_enabled:
        cmds.append(
            Command(
                "mute",
                "stop me reaching out on my own for a while",
                _cmd_mute,
                usage="[2h|30m|1d]",
            )
        )
        cmds.append(
            Command("unmute", "let me reach out unprompted again", _cmd_unmute)
        )
    return CommandRegistry(cmds)
