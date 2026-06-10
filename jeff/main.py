"""Jeff event loop: register service, drain events, reply via Ollama."""

from __future__ import annotations

import asyncio
import logging
import signal

import ensemble
from psycopg_pool import AsyncConnectionPool

from .appraisal import DRIVES, AppraisalDriver, DriveState
from .commands import CommandContext, CommandRegistry, build_command_registry
from .config import Config
from .curiosity import CuriosityDriver, CuriosityStore
from .dispatch import DispatchPolicy, TurnDispatcher
from .impulses import ImpulseStore
from .llm import ChatProvider, ChatResult
from .llm import make_chat_provider
from .memory import Memory
from .mood import MoodStore
from .ollama import Ollama
from .pinned import Pinned, PinnedMemoryStore
from .presence import Presence
from .proactive import ProactiveLoop, ProactiveStore
from .prompt import build_history, compose_system_prompt
from .reflection import Reflector, ReflectionStore
from .screen import screen_text
from .searxng import SearxngClient
from .signal_cli import SignalCliClient
from .signal_front import SignalHandle, run_signal_front
from .tools import ToolRegistry, build_registry


log = logging.getLogger("jeff")

# Sent to the peer when the tool loop hits its iteration cap without the model
# producing a final answer. No exception/internal text — same discipline as the
# silent-failure path (pairs with ticket 2b5e93f8).
_TOOL_CAP_MESSAGE = (
    "Sorry — I couldn't finish working through that within my tool-use limit. "
    "Could you try rephrasing or narrowing the request?"
)

# Sent to the peer when a turn raises (provider timeout, DB fault, …) so Jeff
# reports the glitch in-character instead of going silent. Deliberately generic:
# the exception string may embed an Ollama response body shaped by peer prompts
# (see ollama._safe_excerpt), so NOTHING from the exception goes on the wire —
# only this canned, content-safe line. The operator log line keeps the type name.
_TURN_FAILED_MESSAGE = "Sorry — I glitched out on that one. Give me another go in a bit?"


def _pack_pinned(rows: list[Pinned], max_chars: int) -> list[str]:
    """Pack pinned-memory texts into the prompt block within a char budget.

    Rows arrive most-recent-first; we keep taking until the budget is spent (but
    always keep at least one, so a single over-budget pin still shows). Mirrors
    ReflectionStore.persona's packing discipline.
    """
    out: list[str] = []
    used = 0
    for r in rows:
        cost = len(r.text)
        if used + cost > max_chars and out:
            break
        used += cost
        out.append(r.text)
    return out


async def handle_turn(
    handle: ensemble.ServiceHandle,
    memory: Memory,
    chat_provider: ChatProvider,
    cfg: Config,
    peer: str,
    text: str,
    registry: ToolRegistry | None = None,
    system_prompt: str | None = None,
    curiosity_store: "CuriosityStore | None" = None,
    curiosity_driver: "CuriosityDriver | None" = None,
    reflection_store: "ReflectionStore | None" = None,
    reflector: "Reflector | None" = None,
    mood_store: "MoodStore | None" = None,
    pinned_store: "PinnedMemoryStore | None" = None,
    drive_store: "DriveState | None" = None,
    appraisal_driver: "AppraisalDriver | None" = None,
    impulse_store: "ImpulseStore | None" = None,
) -> None:
    """Process a single inbound chat turn.

    The dispatcher (see jeff.dispatch) wraps each call in a per-peer
    semaphore + global in-flight cap. Exceptions are still caught here
    because the dispatcher's wrapper logs them generically; we want a
    handler-specific log line for debuggability.

    `system_prompt` is the effective prompt (base + capabilities addendum)
    composed once at startup; when omitted it falls back to `cfg.system_prompt`
    so existing callers/tests are unaffected.

    Commands no longer flow through here — the daemon parses them and delivers
    `CommandInvocation` events handled by `_handle_command`, so this is a pure
    chat turn.

    When `registry` has tools and tools are enabled, the reply is produced by
    the execute-and-loop (`_run_tool_loop`); otherwise the single-shot
    `chat()` path runs, byte-identical to the pre-tool behaviour. Either way
    only the *final* assistant text is stored in memory and sent — intermediate
    tool calls/results are working state, not conversational turns, so they
    don't pollute recall.
    """
    try:
        # Jeff's open questions for this peer, surfaced into the prompt's
        # "## You're curious about" block. Best-effort: curiosity is additive, so
        # a store read fault must never break the reply — fall back to no block.
        curiosities: list[str] = []
        if curiosity_store is not None:
            try:
                open_cur = await curiosity_store.open_curiosities(
                    peer, limit=cfg.curiosity_max_open
                )
                curiosities = [c.text for c in open_cur]
            except Exception as e:
                log.error("curiosity fetch failed peer=%s exc=%s", peer, type(e).__name__)

        # Jeff's distilled persona for this peer — durable facts + first-person
        # opinions — surfaced into the prompt's "## What you've come to know"
        # block. Best-effort + additive: a store read fault must never break the
        # reply, so fall back to no block.
        facts: list[str] = []
        opinions: list[str] = []
        if reflection_store is not None:
            try:
                facts, opinions = await reflection_store.persona(
                    peer, max_chars=cfg.persona_max_chars
                )
            except Exception as e:
                log.error("persona fetch failed peer=%s exc=%s", peer, type(e).__name__)

        # Jeff's current mood for this peer — surfaced into the prompt's "## How
        # you're feeling right now" block. Best-effort + additive: a store read
        # fault (or no active mood) must never break the reply — fall back to no
        # block. The DB clock decides expiry, so an expired mood reads as None.
        mood_name = ""
        mood_description = ""
        if mood_store is not None:
            try:
                active = await mood_store.active_mood(peer)
                if active is not None:
                    mood_name = active.name
                    mood_description = active.description or ""
            except Exception as e:
                log.error("mood fetch failed peer=%s exc=%s", peer, type(e).__name__)

        # Jeff's deliberately-pinned memories — surfaced into the prompt's "##
        # Things to remember" block. Best-effort + additive: a store read fault
        # must never break the reply, so fall back to no block.
        pinned: list[str] = []
        if pinned_store is not None:
            try:
                rows = await pinned_store.list(peer, limit=cfg.remember_max_items)
                pinned = _pack_pinned(rows, cfg.remember_max_chars)
            except Exception as e:
                log.error("pinned fetch failed peer=%s exc=%s", peer, type(e).__name__)

        # Jeff's current drive balance for this peer — surfaced into the prompt's
        # "## Your drives right now" block. Best-effort + additive: a store read
        # fault must never break the reply, so fall back to no block. Levels are
        # decayed-to-now in the store; here we just pair each drive's prose noun
        # with its current level for the renderer.
        drives: list[tuple[str, float, float]] = []
        if drive_store is not None:
            try:
                levels = await drive_store.levels(peer)
                drives = [(d.noun, levels[d.key], d.baseline) for d in DRIVES]
            except Exception as e:
                log.error("drives fetch failed peer=%s exc=%s", peer, type(e).__name__)

        # Jeff's active self-set impulses for this peer — surfaced into the
        # prompt's "## What you're driving toward right now" block as ordered
        # (name, description) pairs (strongest first; list_active does the
        # ordering and lazy-expiry filtering). Best-effort + additive: a store
        # read fault must never break the reply, so fall back to no block.
        impulses: list[tuple[str, str]] = []
        if impulse_store is not None:
            try:
                active_impulses = await impulse_store.list_active(peer)
                impulses = [(i.name, i.description) for i in active_impulses]
            except Exception as e:
                log.error("impulses fetch failed peer=%s exc=%s", peer, type(e).__name__)

        # Build the prompt BEFORE persisting the current user message.
        # build_history appends the current turn explicitly; if we stored it
        # first, recent()/recall() would also return it and the message would
        # appear twice in the prompt. Per-peer turns are serialised by the
        # dispatcher, so there's no interleaving risk in reordering this.
        history = await build_history(
            memory,
            peer,
            text,
            recent_turns=cfg.recent_turns,
            recall_k=cfg.recall_k,
            recall_distance_max=cfg.recall_distance_max,
            system_prompt=system_prompt or cfg.system_prompt,
            curiosities=curiosities,
            facts=facts,
            opinions=opinions,
            mood_name=mood_name,
            mood_description=mood_description,
            pinned=pinned,
            drives=drives,
            drives_max_chars=cfg.drives_max_chars,
            impulses=impulses,
            impulses_max_chars=cfg.impulses_max_chars,
        )
        # Persist after building but before the model call, so the user turn is
        # still recorded even if the chat/tool call fails (matches prior behaviour).
        await memory.remember(peer, "user", text)
        if registry is not None and cfg.tools_enabled and len(registry):
            reply = await _run_tool_loop(chat_provider, registry, history, cfg, peer)
        else:
            reply = await chat_provider.chat(history, model=cfg.chat_model)
        await handle.send_message(peer, reply)
        await memory.remember(peer, "assistant", reply)
        # Fire-and-forget: distil what Jeff became curious about / what got
        # answered. maybe_detect never blocks (spawns) and never raises, so this
        # can't delay or break the turn that already sent its reply.
        if curiosity_driver is not None:
            await curiosity_driver.maybe_detect(peer, text, reply)
        # Fire-and-forget: every N turns, consolidate the recent window into
        # durable facts + opinions. maybe_reflect never blocks (spawns) and never
        # raises, so it can't delay or break the turn that already sent its reply.
        if reflector is not None:
            await reflector.maybe_reflect(peer)
        # Fire-and-forget: appraise this exchange against Jeff's drives and nudge
        # their levels — the feedback edge that closes the motivation loop.
        # maybe_appraise never blocks (spawns) and never raises, so it can't delay
        # or break the turn that already sent its reply.
        if appraisal_driver is not None:
            await appraisal_driver.maybe_appraise(peer, text, reply)
    except Exception as e:
        # Deliberately structured: don't log the exception message (it may
        # contain Ollama response body shaped by peer prompts — see
        # ollama._safe_excerpt) and don't log the traceback (it embeds the
        # same string via __cause__). One operator-readable line, no PII.
        log.error("turn failed peer=%s exc=%s", peer, type(e).__name__)
        # Don't leave the peer in silence — send a generic, content-safe apology
        # so a failed turn looks like a glitch, not a dead Jeff. Mirrors the
        # oversize-reject path: the reply carries NOTHING from the exception, and
        # the send is wrapped on its own so a failure here can't escape the
        # handler (the dispatcher would otherwise log it a second time).
        try:
            await handle.send_message(peer, _TURN_FAILED_MESSAGE)
        except Exception:
            log.exception("failed to send turn-failure reply to peer=%s", peer)


def _assistant_tool_message(result: ChatResult) -> dict:
    """Render a tool-calling assistant turn back into OpenAI-canonical form.

    This is the message the model must see echoed before its tool results, so
    it can correlate each `tool_call_id`. Content is preserved if the model
    narrated alongside the calls.
    """
    return {
        "role": "assistant",
        "content": result.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in result.tool_calls
        ],
    }


async def _run_tool_loop(
    chat_provider: ChatProvider,
    registry: ToolRegistry,
    history: list[dict],
    cfg: Config,
    peer: str,
) -> str:
    """Call the provider, execute any tool calls, repeat until a final answer.

    Bounded by `cfg.max_tool_iters`. Each tool result is fed back as a
    `role:"tool"` message; the registry guarantees a safe string even for
    unknown tools / bad args / raises / timeouts, so the loop never crashes on
    a tool fault. Returns the model's final content, or a graceful cap message
    if it never stops calling tools.

    `peer` is the current turn's address; it's passed to `dispatch` so
    peer-scoped tools (e.g. the mood tools) write to the right peer's state.
    """
    specs = registry.specs()
    messages = list(history)
    for _ in range(cfg.max_tool_iters):
        result = await chat_provider.complete(messages, model=cfg.chat_model, tools=specs)
        if not result.tool_calls:
            return result.content or ""
        log.info("tool turn: calls=%s", ",".join(tc.name for tc in result.tool_calls))
        messages.append(_assistant_tool_message(result))
        for tc in result.tool_calls:
            out = await registry.dispatch(
                tc.name, tc.arguments, peer=peer, timeout=cfg.tool_timeout_s
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    log.warning("tool loop hit iteration cap (%d)", cfg.max_tool_iters)
    return _TOOL_CAP_MESSAGE


def _policy_from_config(cfg: Config) -> DispatchPolicy:
    return DispatchPolicy(
        max_inflight=cfg.max_inflight,
        per_peer_concurrency=cfg.per_peer_concurrency,
        peer_rate_per_minute=cfg.peer_rate_per_minute,
        peer_rate_burst=cfg.peer_rate_burst,
        peer_idle_timeout_s=cfg.peer_idle_timeout_s,
    )


async def run(cfg: Config) -> None:
    """Main run loop. Returns when shutdown is requested."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not cfg.allowlist:
        log.warning("JEFF_ALLOWLIST is empty — every chat will be ignored")

    pool = AsyncConnectionPool(
        cfg.db_url,
        min_size=1,
        max_size=4,
        open=False,
    )
    await pool.open()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            # add_signal_handler is unavailable on Windows / inside threads.
            pass

    try:
        # Embeddings always run on the local Ollama (nomic-embed-text is tiny
        # and fits the GPU); chat goes to whatever provider cfg selects. When
        # the chat provider is also ollama these are two clients to the same
        # URL — cheap, and it keeps the embed path independent of chat config.
        async with (
            Ollama(
                cfg.ollama_url,
                max_resp_bytes=cfg.ollama_max_resp_bytes,
                max_embed_dim=cfg.ollama_max_embed_dim,
            ) as embed_client,
            make_chat_provider(cfg) as chat_provider,
            SearxngClient(
                cfg.searxng_url,
                auth=cfg.searxng_auth,
                max_resp_bytes=cfg.ollama_max_resp_bytes,
                safesearch=cfg.search_safesearch,
            ) as searxng,
        ):
            log.info("chat provider=%s model=%s", cfg.llm_provider, cfg.chat_model)
            log.info("system prompt source=%s", cfg.system_prompt_source)

            memory = await Memory.create(
                pool,
                embed_client,
                embed_model=cfg.embed_model,
                embed_dim=cfg.embed_dim,
            )

            # Mood drive (inner-life slice 3) — default OFF. Built only when
            # enabled so the disabled path makes no extra DB calls and stays
            # byte-identical. Plain Postgres (no embedder) — a mood is temporal
            # single-state, not a semantic set. The store backs the mood tools
            # (added to the registry below) and the per-turn active-mood fetch.
            mood_store: MoodStore | None = None
            if cfg.mood_enabled:
                mood_store = await MoodStore.create(pool)
                log.info(
                    "mood enabled (default %g h, max %g h)",
                    cfg.mood_default_hours,
                    cfg.mood_max_hours,
                )
            else:
                log.info("mood disabled")

            # Pinned / explicit memory (inner-life slice 4) — default OFF. Built
            # only when enabled so the disabled path makes no extra DB calls and
            # stays byte-identical. Plain Postgres (no embedder) — pins are always
            # injected, never semantically recalled. Backs the `remember` tool
            # (added to the registry below) + the `/remember` command + the
            # per-turn pinned-memory fetch.
            pinned_store: PinnedMemoryStore | None = None
            if cfg.remember_enabled:
                pinned_store = await PinnedMemoryStore.create(pool)
                log.info(
                    "remember enabled (inject up to %d pins)",
                    cfg.remember_max_items,
                )
            else:
                log.info("remember disabled")

            # Curiosity drive (motivation slice 1) — default OFF. Built only when
            # enabled so the disabled path makes no extra DB/LLM calls and stays
            # byte-identical to today. The store shares the embed client/model/dim
            # with memory; the driver runs the fire-and-forget detection pass.
            curiosity_store: CuriosityStore | None = None
            curiosity_driver: CuriosityDriver | None = None
            if cfg.curiosity_enabled:
                curiosity_store = await CuriosityStore.create(
                    pool,
                    embed_client,
                    embed_model=cfg.embed_model,
                    embed_dim=cfg.embed_dim,
                )
                curiosity_driver = CuriosityDriver(curiosity_store, chat_provider, cfg)
                log.info(
                    "curiosity enabled (detect every %d turn(s), inject up to %d)",
                    cfg.curiosity_every_turns,
                    cfg.curiosity_max_open,
                )
            else:
                log.info("curiosity disabled")

            # Reflection / emergent personality (motivation slice 2) — default
            # OFF. Built only when enabled so the disabled path makes no extra
            # DB/LLM calls and stays byte-identical. The store shares the embed
            # client/model/dim with memory; the reflector reads the episodic
            # window from memory and runs the fire-and-forget consolidation pass.
            reflection_store: ReflectionStore | None = None
            reflector: Reflector | None = None
            if cfg.reflection_enabled:
                reflection_store = await ReflectionStore.create(
                    pool,
                    embed_client,
                    embed_model=cfg.embed_model,
                    embed_dim=cfg.embed_dim,
                )
                reflector = Reflector(reflection_store, memory, chat_provider, cfg)
                log.info(
                    "reflection enabled (consolidate every %d turn(s), persona cap %d chars)",
                    cfg.reflection_every_turns,
                    cfg.persona_max_chars,
                )
            else:
                log.info("reflection disabled")

            # Appraisal / reward drive (motivation slice 3) — default OFF. Built
            # only when enabled so the disabled path makes no extra DB/LLM calls
            # and stays byte-identical. Plain Postgres (no embedder) — a drive
            # level is a scalar, not a semantic set. The store backs the per-turn
            # drive-balance fetch + /mind; the driver runs the fire-and-forget
            # post-turn appraisal pass that nudges the levels.
            drive_store: DriveState | None = None
            appraisal_driver: AppraisalDriver | None = None
            if cfg.appraisal_enabled:
                drive_store = await DriveState.create(
                    pool, half_life_hours=cfg.drive_decay_half_life_hours
                )
                appraisal_driver = AppraisalDriver(drive_store, chat_provider, cfg)
                log.info(
                    "appraisal enabled (appraise every %d turn(s), decay half-life %g h)",
                    cfg.appraisal_every_turns,
                    cfg.drive_decay_half_life_hours,
                )
            else:
                log.info("appraisal disabled")

            # Proactive autonomy loop (motivation slice 4) — default OFF. Built
            # only when enabled so the disabled path makes no extra DB calls and
            # stays byte-identical. Needs appraisal (the connection-pressure
            # signal) AND curiosity (the candidates) on to ever reach out; with
            # either off the loop is inert. Presence is shared with the event
            # drain, which marks every inbound event so a reconnect after silence
            # counts as "reachable now". The loop task is started after
            # registration (it needs the handle); see below.
            proactive_store: ProactiveStore | None = None
            presence: Presence | None = None
            if cfg.proactive_enabled:
                proactive_store = await ProactiveStore.create(pool)
                presence = Presence()
                if not (cfg.appraisal_enabled and cfg.curiosity_enabled):
                    log.warning(
                        "proactive enabled but appraisal=%s curiosity=%s — the loop "
                        "stays inert until both are on",
                        cfg.appraisal_enabled,
                        cfg.curiosity_enabled,
                    )
                log.info(
                    "proactive enabled (check every %gs, connection<%.2f, min-gap %gs, "
                    "presence ttl %gs)",
                    cfg.proactive_interval_s,
                    cfg.proactive_connection_threshold,
                    cfg.proactive_min_gap_s,
                    cfg.proactive_presence_ttl_s,
                )
            else:
                log.info("proactive disabled")

            # Close the proactive-ask → resolution loop: when both the proactive
            # store and the curiosity driver exist, let detection passes consume
            # the curiosity id(s) a reach-out asked about so an answered question
            # gets marked satisfied instead of lingering in /mind (ticket
            # 342c7071). No-op when either is off.
            if curiosity_driver is not None and proactive_store is not None:
                curiosity_driver.attach_proactive_store(proactive_store)

            # Impulses (motivation slice) — self-authored short-term directional
            # drives. Default OFF: built only when enabled so the disabled path
            # makes no extra DB calls and stays byte-identical. Plain Postgres (no
            # embedder). The store backs the per-turn impulse-block fetch, the
            # set/adjust/clear tools, and /mind.
            impulse_store: ImpulseStore | None = None
            if cfg.impulses_enabled:
                impulse_store = await ImpulseStore.create(pool)
                log.info(
                    "impulses enabled (default %gh, max %gh, max %d chars)",
                    cfg.impulses_default_hours,
                    cfg.impulses_max_hours,
                    cfg.impulses_max_chars,
                )
            else:
                log.info("impulses disabled")

            # Registry built AFTER the stores so the mood/impulse tools can be
            # wired to their stores. Empty when tools are off → the no-tools path.
            registry = build_registry(
                cfg,
                searxng=searxng,
                mood_store=mood_store,
                pinned_store=pinned_store,
                impulse_store=impulse_store,
            )
            if cfg.tools_enabled and len(registry):
                # Names only — never tool args (the leaky-info discipline).
                log.info("tools enabled: %s", ", ".join(registry.names()))
            else:
                log.info("tools disabled")
            # Compose the effective prompt once: operator's base (file/env/
            # default) + a capabilities addendum describing the registered tools
            # and Markdown rendering. Tools off → addendum is just the formatting
            # note. Built once, reused for every turn.
            system_prompt = compose_system_prompt(cfg.system_prompt, registry.names())

            # Chat commands are declared to the daemon at registration and
            # received as CommandInvocation events (the daemon owns parsing).
            # Built once; log enabled names (mirror the tools/prompt-source
            # startup lines). Disabled → declare nothing, receive nothing.
            commands = (
                build_command_registry(
                    curiosity_enabled=cfg.curiosity_enabled,
                    reflection_enabled=cfg.reflection_enabled,
                    mood_enabled=cfg.mood_enabled,
                    remember_enabled=cfg.remember_enabled,
                    appraisal_enabled=cfg.appraisal_enabled,
                    proactive_enabled=cfg.proactive_enabled,
                    impulses_enabled=cfg.impulses_enabled,
                )
                if cfg.commands_enabled
                else None
            )
            if commands is not None and len(commands):
                log.info("commands enabled: %s", ", ".join(commands.names()))
            else:
                log.info("commands disabled")

            client_kwargs: dict = {"socket_path": cfg.socket}
            if cfg.auth_seed_path:
                client_kwargs["auth_seed"] = cfg.auth_seed_path

            # Specs declared at registration so the daemon can route/aggregate
            # them (and surface them in its unified /help). None → not declared.
            command_specs = commands.to_ensemble_commands() if commands is not None else None

            async with ensemble.Client(**client_kwargs) as client:
                handle = await client.register(
                    name=cfg.name,
                    acl=ensemble.ACL.ALLOWLIST,
                    allowlist=cfg.allowlist,
                    description=cfg.description,
                    commands=command_specs,
                )
                async with handle:
                    log.info(
                        "registered service=%s address=%s onion=%s",
                        cfg.name,
                        handle.address,
                        handle.onion,
                    )

                    async def _on_turn(peer: str, text: str) -> None:
                        await handle_turn(
                            handle,
                            memory,
                            chat_provider,
                            cfg,
                            peer,
                            text,
                            registry,
                            system_prompt,
                            curiosity_store,
                            curiosity_driver,
                            reflection_store,
                            reflector,
                            mood_store,
                            pinned_store,
                            drive_store,
                            appraisal_driver,
                            impulse_store,
                        )

                    dispatcher = TurnDispatcher(_on_turn, _policy_from_config(cfg))

                    # Signal front door (default OFF) — Jeff's second inbound
                    # channel. Built only when enabled so the disabled path makes
                    # no extra connections and stays byte-identical. It reuses the
                    # WHOLE turn pipeline: a SignalHandle adapter stands in for the
                    # Ensemble handle (handle_turn only calls send_message), and a
                    # dedicated dispatcher gives Signal its own per-peer
                    # serialisation / rate-limit / drain. Replies route back over
                    # Signal because the handler binds the SignalHandle.
                    signal_client: SignalCliClient | None = None
                    signal_dispatcher: TurnDispatcher | None = None
                    signal_task: asyncio.Task | None = None
                    if cfg.signal_enabled:
                        if not cfg.signal_allowlist:
                            log.warning(
                                "JEFF_SIGNAL_ALLOWLIST is empty — Signal answers nobody"
                            )
                        signal_client = SignalCliClient(
                            number=cfg.signal_number,
                            allowlist=set(cfg.signal_allowlist),
                            base_url=cfg.signal_api_url,
                        )
                        signal_handle = SignalHandle(signal_client)

                        async def _on_signal_turn(peer: str, text: str) -> None:
                            await handle_turn(
                                signal_handle,
                                memory,
                                chat_provider,
                                cfg,
                                peer,
                                text,
                                registry,
                                system_prompt,
                                curiosity_store,
                                curiosity_driver,
                                reflection_store,
                                reflector,
                                mood_store,
                                pinned_store,
                                drive_store,
                                appraisal_driver,
                                impulse_store,
                            )

                        signal_dispatcher = TurnDispatcher(
                            _on_signal_turn, _policy_from_config(cfg)
                        )
                        signal_task = asyncio.create_task(
                            run_signal_front(
                                signal_client,
                                signal_dispatcher,
                                poll_interval=cfg.signal_poll_interval,
                            ),
                            name="jeff-signal",
                        )
                        log.info(
                            "signal front door enabled (api=%s, allowlist=%d)",
                            cfg.signal_api_url,
                            len(cfg.signal_allowlist),
                        )
                    else:
                        log.info("signal front door disabled")

                    # Proactive heartbeat (default off) — the consumer that turns
                    # accrued state (curiosities, drive pressure) into unprompted
                    # contact. Reaches out over the Ensemble handle to allowlisted
                    # peers; presence is fed by the event drain below. (Signal-side
                    # proactivity would need its own loop on the SignalHandle — a
                    # future add; v1 reaches out over Ensemble only.)
                    proactive_task: asyncio.Task | None = None
                    if proactive_store is not None and presence is not None:
                        proactive_loop = ProactiveLoop(
                            handle,
                            proactive_store,
                            presence,
                            memory,
                            curiosity_store=curiosity_store,
                            reflection_store=reflection_store,
                            mood_store=mood_store,
                            drive_store=drive_store,
                            chat_provider=chat_provider,
                            cfg=cfg,
                            allowlist=cfg.allowlist,
                        )
                        proactive_task = asyncio.create_task(
                            proactive_loop.run(), name="jeff-proactive"
                        )

                    events_task = asyncio.create_task(
                        _drain_events(
                            handle,
                            dispatcher,
                            cfg,
                            memory,
                            commands,
                            system_prompt,
                            tuple(registry.names()),
                            curiosity_store,
                            reflection_store,
                            mood_store,
                            pinned_store,
                            drive_store,
                            proactive_store,
                            presence,
                            impulse_store,
                        ),
                        name="jeff-events",
                    )
                    stop_task = asyncio.create_task(stop.wait(), name="jeff-stop")
                    wait_set = {events_task, stop_task}
                    if signal_task is not None:
                        wait_set.add(signal_task)
                    if proactive_task is not None:
                        wait_set.add(proactive_task)
                    done, pending = await asyncio.wait(
                        wait_set,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                    for t in done:
                        exc = t.exception()
                        if exc is not None:
                            log.error("task %s failed: %s", t.get_name(), exc)
                    await dispatcher.drain()
                    if signal_dispatcher is not None:
                        await signal_dispatcher.drain()
                    if signal_client is not None:
                        await signal_client.aclose()
                    if curiosity_driver is not None:
                        await curiosity_driver.aclose()
                    if reflector is not None:
                        await reflector.aclose()
                    if appraisal_driver is not None:
                        await appraisal_driver.aclose()
    finally:
        await pool.close()


async def _handle_command(
    handle: ensemble.ServiceHandle,
    memory: Memory,
    cfg: Config,
    commands: CommandRegistry,
    inv: ensemble.CommandInvocation,
    system_prompt: str = "",
    tool_names: tuple[str, ...] = (),
    curiosity_store: "CuriosityStore | None" = None,
    reflection_store: "ReflectionStore | None" = None,
    mood_store: "MoodStore | None" = None,
    pinned_store: "PinnedMemoryStore | None" = None,
    drive_store: "DriveState | None" = None,
    proactive_store: "ProactiveStore | None" = None,
    impulse_store: "ImpulseStore | None" = None,
) -> None:
    """Run a daemon-routed command invocation and reply via the command channel.

    `dispatch()` is safe-by-construction (unknown command / handler raise both
    become content-safe strings), so the only thing that can fail here is the
    reply send — wrapped on its own so a send fault can't escape the event loop.
    The reply goes back as a CommandResult (not a chat message), so the daemon
    can merge it with its own built-in leg under the augment model.

    `system_prompt`/`tool_names` are the effective turn context, passed through to
    CommandContext so `/debug` can show what Jeff actually works with.
    """
    ctx = CommandContext(
        handle=handle,
        memory=memory,
        cfg=cfg,
        peer=inv.from_addr,
        args=inv.args,
        system_prompt=system_prompt,
        tool_names=tool_names,
        curiosity=curiosity_store,
        reflection=reflection_store,
        mood=mood_store,
        pinned=pinned_store,
        drives=drive_store,
        proactive=proactive_store,
        impulses=impulse_store,
    )
    reply = await commands.dispatch(inv.name, ctx)
    try:
        await handle.send_command_result(inv.command_id, reply)
    except Exception:
        log.exception("failed to send command result to peer=%s", inv.from_addr)


async def _drain_events(
    handle: ensemble.ServiceHandle,
    dispatcher: TurnDispatcher,
    cfg: Config,
    memory: Memory | None = None,
    commands: CommandRegistry | None = None,
    system_prompt: str = "",
    tool_names: tuple[str, ...] = (),
    curiosity_store: "CuriosityStore | None" = None,
    reflection_store: "ReflectionStore | None" = None,
    mood_store: "MoodStore | None" = None,
    pinned_store: "PinnedMemoryStore | None" = None,
    drive_store: "DriveState | None" = None,
    proactive_store: "ProactiveStore | None" = None,
    presence: "Presence | None" = None,
    impulse_store: "ImpulseStore | None" = None,
) -> None:
    allow = set(cfg.allowlist)
    async for event in handle.events():
        # Any inbound event means the peer is reachable right now — mark presence
        # before the type dispatch so the proactive loop's "don't shout into the
        # void" gate sees connects/commands/chats alike (a reconnect after silence
        # is exactly when a reach-out should be allowed to fire).
        src = getattr(event, "from_addr", None)
        if presence is not None and isinstance(src, str) and src:
            presence.mark(src)
        if isinstance(event, ensemble.CommandInvocation):
            # The daemon already gates invocations by the service ACL; re-check
            # the allowlist as defence-in-depth, mirroring the chat path.
            if commands is None or memory is None:
                continue
            if event.from_addr not in allow:
                log.info("ignoring command from non-allowlisted peer=%s", event.from_addr)
                continue
            # Commands are fast, deterministic, and never call the model, so they
            # run inline rather than through the turn dispatcher (which rate-limits
            # and serialises LLM turns).
            await _handle_command(
                handle,
                memory,
                cfg,
                commands,
                event,
                system_prompt,
                tool_names,
                curiosity_store,
                reflection_store,
                mood_store,
                pinned_store,
                drive_store,
                proactive_store,
                impulse_store,
            )
            continue
        if not isinstance(event, ensemble.ChatMessage):
            continue
        peer = event.from_addr
        if peer not in allow:
            log.info("ignoring chat from non-allowlisted peer=%s", peer)
            continue
        rejection = screen_text(event.text, cfg.max_message_bytes)
        if rejection is not None:
            # Oversize: send a polite refusal directly and drop the turn before
            # it touches Ollama embed / Postgres insert. Don't run through the
            # dispatcher — this is a cheap synchronous reply.
            log.warning(
                "dropping oversize message from peer=%s (%d bytes, cap=%d)",
                peer,
                len(event.text.encode("utf-8")),
                cfg.max_message_bytes,
            )
            try:
                await handle.send_message(peer, rejection)
            except Exception:
                log.exception("failed to send oversize-rejection reply to peer=%s", peer)
            continue
        await dispatcher.submit(peer, event.text)
