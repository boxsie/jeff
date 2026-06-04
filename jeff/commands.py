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

    from .config import Config
    from .curiosity import CuriosityStore
    from .memory import Memory, Message


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
    (None when the curiosity drive is off) so `/mind` can show it and `/forget`
    can wipe it. They default to empty/None so callers that don't care (and older
    tests) need not supply them.
    """

    handle: "ensemble.ServiceHandle"
    memory: "Memory"
    cfg: "Config"
    peer: str
    args: str
    system_prompt: str = ""
    tool_names: tuple[str, ...] = ()
    curiosity: "CuriosityStore | None" = None


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
    # Wipe the curiosity store too (when the drive is on) so a clean slate really
    # is clean — open questions are derived from the same conversations.
    if ctx.curiosity is not None:
        await ctx.curiosity.forget(ctx.peer)
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


async def _cmd_mind(ctx: CommandContext) -> str:
    """Show what's on Jeff's mind: open questions it wants to ask, and the most
    recently answered ones. Deterministic dump (no model call), sibling to
    `/debug`. Only declared when the curiosity drive is enabled."""
    if ctx.curiosity is None:
        return "Curiosity isn't switched on right now, so there's nothing on my mind to show."
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

    body = "\n".join(lines)
    return "**on my mind**\n```\n" + body + "\n```"


def build_command_registry(*, curiosity_enabled: bool = False) -> CommandRegistry:
    """Jeff's declared command set (see main.run). `/help`/`/whoami` are the
    daemon's built-ins; the old `/new` is subsumed by the augmented `/clear`.

    `/mind` is declared only when the curiosity drive is on — keeping the
    feature-off path's declared command set unchanged."""
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
    if curiosity_enabled:
        cmds.append(
            Command("mind", "show what I'm curious about right now", _cmd_mind)
        )
    return CommandRegistry(cmds)
