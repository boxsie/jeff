"""The Signal front door — Jeff's second inbound channel, alongside Ensemble.

You text Jeff's dedicated Signal number from your phone like texting a person;
Jeff replies on the same thread. The you↔Jeff path is **pure Signal** — Ensemble
is not the transport for it. This module is the in-process replacement for the
retired ``wingit`` relay's Ensemble far-leg: instead of dialling Jeff over the
network, an inbound Signal message is fed straight into Jeff's existing turn
pipeline (``handle_turn`` → dispatch → LLM → pgvector memory) and the reply goes
back out over Signal.

Two small pieces live here:

  - **SignalHandle** — an adapter exposing the one method ``handle_turn`` uses on
    an ``ensemble.ServiceHandle`` (``send_message``), backed by a
    ``SignalCliClient``. Because ``handle_turn`` only ever calls
    ``handle.send_message(peer, reply)``, Signal is just *another channel to
    reply on*: the entire turn pipeline (memory, recall, tools, curiosity, mood,
    drives, the graceful-failure reply) is reused verbatim, no fork.

  - **run_signal_front** — the receive loop: pull allowlisted messages off the
    ``SignalCliClient``, dedup, and hand each to a ``TurnDispatcher`` (its handler
    bound to a ``SignalHandle``, so replies route back over Signal). The
    dispatcher gives the same per-peer serialisation / rate-limit / graceful
    drain the Ensemble channel gets.

Everything here is gated by ``JEFF_SIGNAL_ENABLED`` (default off): when off the
client/loop are never constructed, so behaviour is byte-identical to today.

Public-repo hygiene: neutral wording, no numbers/infra in code or logs.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Protocol

from .signal_cli import SignalCliClient

log = logging.getLogger("jeff.signal")


# Bound on the in-memory dedup set. signal-cli's /v1/receive consumes its queue
# per poll, so cross-poll redelivery only happens on a crash mid-processing; this
# guards mainly against a double-delivered envelope within a poll window. Kept
# small — it only needs to cover a burst, not history.
_DEDUP_MAX = 512


class _Dispatcher(Protocol):
    """The slice of TurnDispatcher the front door needs."""

    async def submit(self, peer: str, *args: object, **kwargs: object) -> bool: ...


class SignalHandle:
    """Adapter exposing the ``send_message`` surface ``handle_turn`` expects,
    backed by a ``SignalCliClient``.

    Lets the Signal front door reuse ``handle_turn`` unchanged — to that code a
    ``handle`` is anything with ``async send_message(to_addr, text)``, and Signal
    is just a different wire to reply on. The reply always goes back to the
    address of the turn being answered (``to_addr`` == the inbound sender).
    """

    def __init__(self, client: SignalCliClient):
        self._client = client

    async def send_message(self, to_addr: str, text: str) -> None:
        await self._client.send(text, recipient=to_addr)


async def run_signal_front(
    client: SignalCliClient,
    dispatcher: _Dispatcher,
    *,
    poll_interval: float = 1.0,
) -> None:
    """Receive allowlisted Signal messages forever and dispatch each as a turn.

    Dedups on ``(from_number, msg_id)`` so a redelivered envelope isn't answered
    twice. Unlike wingit's relay (which recorded the seen-key only after a
    successful relay so a failed delivery could be retried), the key is recorded
    *before* dispatch here: ``handle_turn`` is fire-and-forget and self-recovers
    (it sends its own graceful-failure reply on error), so there is no "failed
    delivery to retry" — answering once is the correct contract.

    The loop ends only when the underlying ``client.receive`` async-generator is
    closed (cancellation on shutdown); transient signal-cli faults are absorbed
    by ``receive``'s internal backoff, so a flaky leg never ends the front door.
    """
    seen: OrderedDict[tuple[str, str], None] = OrderedDict()
    async for msg in client.receive(poll_interval=poll_interval):
        key = (msg.from_number, msg.msg_id)
        if key in seen:
            log.debug("dropping duplicate signal message")
            continue
        seen[key] = None
        while len(seen) > _DEDUP_MAX:
            seen.popitem(last=False)
        # Fire-and-forget through the dispatcher; it logs its own drops
        # (rate/inflight caps) and runs handle_turn under the per-peer semaphore.
        await dispatcher.submit(msg.from_number, msg.text)
