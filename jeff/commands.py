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

  - `/clear`  — session reset: drop the active thread from the recent window,
                keep long-term semantic memory (augments the daemon transcript-clear).
  - `/forget` — hard wipe of everything Jeff remembers about this peer (confirm-gated).
  - `/stats`  — memory counts, uptime, active provider/model, prompt source.

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
    from .memory import Memory


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
    """

    handle: "ensemble.ServiceHandle"
    memory: "Memory"
    cfg: "Config"
    peer: str
    args: str


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
    """Session reset: drop the active thread from the conversational window but
    keep long-term semantic memory (older lines can still resurface via recall).

    This is the service leg of the daemon's `/clear` augment — the daemon clears
    the operator's local transcript; this clears Jeff's working window."""
    await ctx.memory.set_history_cutoff(ctx.peer)
    return (
        "Started a fresh conversation — I've cleared this thread from my active "
        "memory. I'll still remember older things if they come up naturally."
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


def build_command_registry() -> CommandRegistry:
    """Jeff's declared command set (see main.run). `/help`/`/whoami` are the
    daemon's built-ins; the old `/new` is subsumed by the augmented `/clear`."""
    return CommandRegistry(
        [
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
        ]
    )
