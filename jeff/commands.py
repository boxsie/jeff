"""In-band chat commands — deterministic control messages, not LLM turns.

The operator can type `/help`, `/new`, `/forget`, `/stats`, `/whoami` in chat.
These are intercepted at the top of `main.handle_turn` *before* anything reaches
memory or the model: a command and its reply are **control traffic** and are
never stored (they'd pollute recall + the recent window) and never sent to the
LLM. This is a different mechanism from the tool-use foundation — there the
model decides to call a tool mid-turn; here the operator drives Jeff directly.

`dispatch` is safe-by-construction, mirroring `tools.base.ToolRegistry`: a
handler that raises becomes a short, content-safe apology rather than an
exception that escapes into the turn loop (which would log + reply nothing).
Single-user threat model (ACL = operator only) means there's no per-command
permission split — only the operator ever reaches this code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:  # avoid import cost / cycles at runtime; these are type-only
    import ensemble

    from .config import Config
    from .llm import ChatProvider
    from .memory import Memory


log = logging.getLogger("jeff.commands")

# Captured at import (≈ process start) so /stats can report uptime without
# threading a start time through every turn. Monotonic so it's immune to wall
# clock changes.
_PROCESS_START = time.monotonic()


@dataclass
class CommandContext:
    """Everything a command handler might need to act and to render a reply.

    `args` is the text after the command name (stripped). `registry` is the
    live registry so `/help` enumerates the real command set instead of a
    hand-maintained list that could drift. `handle` isn't used by the starter
    commands but is here for future commands that send their own traffic (e.g.
    a `/backup` that offers a file over Ensemble).
    """

    handle: "ensemble.ServiceHandle"
    memory: "Memory"
    cfg: "Config"
    chat_provider: "ChatProvider"
    peer: str
    args: str
    registry: "CommandRegistry"


CommandHandler = Callable[[CommandContext], Awaitable[str]]


@dataclass(frozen=True)
class Command:
    """One registered command: its bare name (no prefix), help text, handler."""

    name: str
    description: str
    handler: CommandHandler


def parse_command(text: str, prefix: str) -> tuple[str, str] | None:
    """Split a message into `(name, args)` iff it is a command.

    A message is a command exactly when (after leading whitespace) it starts
    with `prefix`. The first whitespace-delimited token after the prefix is the
    (case-folded) command name; the remainder, stripped, is the args. Returns
    `None` for normal chat (no prefix) and for a bare prefix with no name — both
    flow on to the normal turn untouched.
    """
    if not prefix:
        return None
    s = text.lstrip()
    if not s.startswith(prefix):
        return None
    body = s[len(prefix):]
    parts = body.split(None, 1)
    if not parts or not parts[0]:
        return None
    name = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""
    return name, args


class CommandRegistry:
    """Holds the active commands and dispatches to them safely."""

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

    async def dispatch(self, name: str, ctx: CommandContext) -> str:
        """Run command `name`; always return a content-safe reply string.

        An unknown command yields a friendly hint (never passed to the model);
        a handler that raises is logged (type only) and yields a generic
        apology — never an exception string, which could carry endpoint/peer
        detail (same discipline as the turn loop + tool dispatch).
        """
        prefix = ctx.cfg.command_prefix
        cmd = self._commands.get(name)
        if cmd is None:
            log.info("unknown command=%s", name)
            return f"Unknown command {prefix}{name} — try {prefix}help for the list."
        try:
            return await cmd.handler(ctx)
        except Exception:
            log.exception("command %s raised", name)
            return "Sorry — that command hit a snag on my end. Try again in a moment."


# --- starter command handlers ----------------------------------------------


async def _cmd_new(ctx: CommandContext) -> str:
    """Soft reset: drop the active thread from the conversational window but
    keep long-term semantic memory (older lines can still resurface via recall
    if relevant to a future question)."""
    await ctx.memory.set_history_cutoff(ctx.peer)
    return (
        "Started a fresh conversation — I've cleared this thread from my active "
        "memory. I'll still remember older things if they come up naturally."
    )


async def _cmd_forget(ctx: CommandContext) -> str:
    """Hard wipe: irreversible DELETE of every stored message for this peer.

    Confirm-gated (`/forget yes`) on purpose — it's irreversible and a stray
    `/forget` in chat is easy to fat-finger. The CLI `forget` subcommand stays
    no-confirm (it's an explicit admin action); chat gets the guardrail.
    """
    if ctx.args.strip().lower() != "yes":
        return (
            "⚠️ This permanently deletes everything I remember about our "
            "conversations — there's no undo. If you're sure, send "
            f"`{ctx.cfg.command_prefix}forget yes`."
        )
    deleted = await ctx.memory.forget(ctx.peer)
    return f"Wiped {deleted} stored message(s) — clean slate."


async def _cmd_help(ctx: CommandContext) -> str:
    prefix = ctx.cfg.command_prefix
    lines = ["Here's what I can do on command:"]
    for name in ctx.registry.names():
        cmd = ctx.registry.get(name)
        if cmd is not None:
            lines.append(f"- `{prefix}{name}` — {cmd.description}")
    return "\n".join(lines)


async def _cmd_stats(ctx: CommandContext) -> str:
    """Counts + uptime + active model. No DSN / key / seed / endpoint."""
    mine = await ctx.memory.count(ctx.peer)
    total = await ctx.memory.total()
    uptime = _format_uptime(time.monotonic() - _PROCESS_START)
    return (
        "**Stats**\n"
        f"- stored messages (you): {mine}\n"
        f"- stored messages (all peers): {total}\n"
        f"- uptime: {uptime}\n"
        f"- provider: {ctx.cfg.llm_provider}\n"
        f"- model: {ctx.cfg.chat_model}"
    )


async def _cmd_whoami(ctx: CommandContext) -> str:
    """Mirrors the startup log lines so the operator can confirm the live
    config/build from chat. Provider/model/prompt-source only — no secrets."""
    return (
        "I'm Jeff.\n"
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
    """The default command set wired into production (see main.run)."""
    return CommandRegistry(
        [
            Command(
                "new",
                "start a fresh conversation (keeps long-term memory)",
                _cmd_new,
            ),
            Command("clear", "alias for /new", _cmd_new),
            Command(
                "forget",
                "permanently wipe everything I remember (needs /forget yes)",
                _cmd_forget,
            ),
            Command("help", "list these commands", _cmd_help),
            Command("stats", "memory counts, uptime, and active model", _cmd_stats),
            Command(
                "whoami",
                "show my active provider, model, and prompt source",
                _cmd_whoami,
            ),
        ]
    )
