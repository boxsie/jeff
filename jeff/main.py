"""Jeff event loop: register service, drain events, reply via Ollama."""

from __future__ import annotations

import asyncio
import logging
import signal

import ensemble
from psycopg_pool import AsyncConnectionPool

from .config import Config
from .dispatch import DispatchPolicy, TurnDispatcher
from .llm import ChatProvider, ChatResult
from .llm import make_chat_provider
from .memory import Memory
from .ollama import Ollama
from .prompt import build_history, compose_system_prompt
from .screen import screen_text
from .searxng import SearxngClient
from .tools import ToolRegistry, build_registry


log = logging.getLogger("jeff")

# Sent to the peer when the tool loop hits its iteration cap without the model
# producing a final answer. No exception/internal text — same discipline as the
# silent-failure path (pairs with ticket 2b5e93f8).
_TOOL_CAP_MESSAGE = (
    "Sorry — I couldn't finish working through that within my tool-use limit. "
    "Could you try rephrasing or narrowing the request?"
)


async def handle_turn(
    handle: ensemble.ServiceHandle,
    memory: Memory,
    chat_provider: ChatProvider,
    cfg: Config,
    peer: str,
    text: str,
    registry: ToolRegistry | None = None,
    system_prompt: str | None = None,
) -> None:
    """Process a single inbound chat turn.

    The dispatcher (see jeff.dispatch) wraps each call in a per-peer
    semaphore + global in-flight cap. Exceptions are still caught here
    because the dispatcher's wrapper logs them generically; we want a
    handler-specific log line for debuggability.

    `system_prompt` is the effective prompt (base + capabilities addendum)
    composed once at startup; when omitted it falls back to `cfg.system_prompt`
    so existing callers/tests are unaffected.

    When `registry` has tools and tools are enabled, the reply is produced by
    the execute-and-loop (`_run_tool_loop`); otherwise the single-shot
    `chat()` path runs, byte-identical to the pre-tool behaviour. Either way
    only the *final* assistant text is stored in memory and sent — intermediate
    tool calls/results are working state, not conversational turns, so they
    don't pollute recall.
    """
    try:
        await memory.remember(peer, "user", text)
        history = await build_history(
            memory,
            peer,
            text,
            recent_turns=cfg.recent_turns,
            recall_k=cfg.recall_k,
            system_prompt=system_prompt or cfg.system_prompt,
        )
        if registry is not None and cfg.tools_enabled and len(registry):
            reply = await _run_tool_loop(chat_provider, registry, history, cfg)
        else:
            reply = await chat_provider.chat(history, model=cfg.chat_model)
        await handle.send_message(peer, reply)
        await memory.remember(peer, "assistant", reply)
    except Exception as e:
        # Deliberately structured: don't log the exception message (it may
        # contain Ollama response body shaped by peer prompts — see
        # ollama._safe_excerpt) and don't log the traceback (it embeds the
        # same string via __cause__). One operator-readable line, no PII.
        log.error("turn failed peer=%s exc=%s", peer, type(e).__name__)


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
) -> str:
    """Call the provider, execute any tool calls, repeat until a final answer.

    Bounded by `cfg.max_tool_iters`. Each tool result is fed back as a
    `role:"tool"` message; the registry guarantees a safe string even for
    unknown tools / bad args / raises / timeouts, so the loop never crashes on
    a tool fault. Returns the model's final content, or a graceful cap message
    if it never stops calling tools.
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
            out = await registry.dispatch(tc.name, tc.arguments, timeout=cfg.tool_timeout_s)
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
            registry = build_registry(cfg, searxng=searxng)
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
            memory = await Memory.create(
                pool,
                embed_client,
                embed_model=cfg.embed_model,
                embed_dim=cfg.embed_dim,
            )

            client_kwargs: dict = {"socket_path": cfg.socket}
            if cfg.auth_seed_path:
                client_kwargs["auth_seed"] = cfg.auth_seed_path

            async with ensemble.Client(**client_kwargs) as client:
                handle = await client.register(
                    name=cfg.name,
                    acl=ensemble.ACL.ALLOWLIST,
                    allowlist=cfg.allowlist,
                    description=cfg.description,
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
                        )

                    dispatcher = TurnDispatcher(_on_turn, _policy_from_config(cfg))

                    events_task = asyncio.create_task(
                        _drain_events(handle, dispatcher, cfg),
                        name="jeff-events",
                    )
                    stop_task = asyncio.create_task(stop.wait(), name="jeff-stop")
                    done, pending = await asyncio.wait(
                        {events_task, stop_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                    for t in done:
                        exc = t.exception()
                        if exc is not None:
                            log.error("task %s failed: %s", t.get_name(), exc)
                    await dispatcher.drain()
    finally:
        await pool.close()


async def _drain_events(
    handle: ensemble.ServiceHandle,
    dispatcher: TurnDispatcher,
    cfg: Config,
) -> None:
    allow = set(cfg.allowlist)
    async for event in handle.events():
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
