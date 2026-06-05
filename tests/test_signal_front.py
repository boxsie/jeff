"""Tests for the Signal front door (SignalHandle adapter + receive loop).

Pure, no network: a fake SignalCliClient yields a scripted message list and a
fake dispatcher records what gets submitted. Covers the send_message adapter,
dedup on (from_number, msg_id), and that each unique message is dispatched with
the right peer + text.
"""

from __future__ import annotations

import pytest

from jeff.signal_cli import SignalMessage
from jeff.signal_front import SignalHandle, run_signal_front

OPERATOR = "+14441111111"


class FakeClient:
    """Stand-in SignalCliClient: receive() yields a fixed list then stops; send()
    records calls so the SignalHandle adapter can be asserted."""

    def __init__(self, messages: list[SignalMessage] | None = None):
        self._messages = messages or []
        self.sent: list[tuple[str, str]] = []

    async def receive(self, *, poll_interval: float = 1.0):
        for m in self._messages:
            yield m

    async def send(self, text: str, *, recipient: str) -> None:
        self.sent.append((recipient, text))


class FakeDispatcher:
    def __init__(self):
        self.submitted: list[tuple[str, str]] = []

    async def submit(self, peer: str, *args) -> bool:
        self.submitted.append((peer, args[0]))
        return True


def _msg(text: str, ts: int, frm: str = OPERATOR) -> SignalMessage:
    return SignalMessage(from_number=frm, text=text, timestamp=ts, msg_id=str(ts))


# --- SignalHandle adapter ---------------------------------------------------


@pytest.mark.asyncio
async def test_signal_handle_send_message_routes_to_recipient():
    client = FakeClient()
    handle = SignalHandle(client)
    await handle.send_message(OPERATOR, "hello from jeff")
    # handle_turn calls handle.send_message(peer, reply); the adapter turns that
    # into client.send(reply, recipient=peer) so the reply hits the sender's thread.
    assert client.sent == [(OPERATOR, "hello from jeff")]


# --- run_signal_front -------------------------------------------------------


@pytest.mark.asyncio
async def test_run_signal_front_dispatches_each_message():
    client = FakeClient([_msg("hi", 1), _msg("what's up", 2)])
    dispatcher = FakeDispatcher()
    await run_signal_front(client, dispatcher)
    assert dispatcher.submitted == [(OPERATOR, "hi"), (OPERATOR, "what's up")]


@pytest.mark.asyncio
async def test_run_signal_front_dedups_on_from_number_and_msg_id():
    # Same (from_number, msg_id) delivered twice → dispatched once; a distinct
    # msg_id from the same sender still dispatches.
    client = FakeClient([_msg("once", 5), _msg("once", 5), _msg("again", 6)])
    dispatcher = FakeDispatcher()
    await run_signal_front(client, dispatcher)
    assert dispatcher.submitted == [(OPERATOR, "once"), (OPERATOR, "again")]


@pytest.mark.asyncio
async def test_run_signal_front_same_msg_id_different_sender_not_deduped():
    # Dedup keys on the PAIR, so the same timestamp from two senders is distinct.
    other = "+13332222222"
    client = FakeClient([_msg("a", 9, frm=OPERATOR), _msg("b", 9, frm=other)])
    dispatcher = FakeDispatcher()
    await run_signal_front(client, dispatcher)
    assert dispatcher.submitted == [(OPERATOR, "a"), (other, "b")]
