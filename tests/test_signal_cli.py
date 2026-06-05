"""Tests for the Signal leg (signal-cli-rest-api client).

These exercise the client against a mocked signal-cli HTTP surface via
``httpx.MockTransport`` (the house pattern, see test_searxng) — no real
signal-cli, no network. They cover normalisation, the allowlist drop, the send
call shape, send-failure surfacing, and the receive backoff loop.
"""

from __future__ import annotations

import json

import httpx
import pytest

from jeff.signal_cli import (
    SignalCliClient,
    SignalError,
    SignalMessage,
    SignalSendError,
    _parse_envelope,
)

BASE_URL = "http://signal-cli:8080"
BOT_NUMBER = "+15550000000"
OPERATOR = "+14441111111"
OPERATOR2 = "+13332222222"
STRANGER = "+19998887777"
ALLOW = frozenset({OPERATOR, OPERATOR2})


def _client(handler=None, *, allowlist=ALLOW) -> SignalCliClient:
    http = None
    if handler is not None:
        http = httpx.AsyncClient(
            base_url=BASE_URL, transport=httpx.MockTransport(handler)
        )
    return SignalCliClient(
        number=BOT_NUMBER,
        allowlist=allowlist,
        base_url=BASE_URL if http is None else None,
        http=http,
        backoff_initial=0.01,
        backoff_max=0.04,
    )


def _envelope(source: str, message: str | None, timestamp: int = 1717000000000) -> dict:
    data_message = None if message is None else {"message": message, "timestamp": timestamp}
    return {
        "envelope": {
            "source": source,
            "sourceNumber": source,
            "timestamp": timestamp,
            "dataMessage": data_message,
        }
    }


# --- normalisation -----------------------------------------------------------


def test_parse_envelope_normalises_to_signal_message():
    item = _envelope(OPERATOR, "hello jeff", timestamp=1717000000123)
    msg = _parse_envelope(item, ALLOW)
    assert msg == SignalMessage(
        from_number=OPERATOR,
        text="hello jeff",
        timestamp=1717000000123,
        msg_id="1717000000123",
    )


def test_parse_envelope_accepts_source_only_field():
    item = {"envelope": {"source": OPERATOR, "timestamp": 42, "dataMessage": {"message": "hi"}}}
    msg = _parse_envelope(item, ALLOW)
    assert msg is not None
    assert msg.from_number == OPERATOR and msg.text == "hi"


def test_parse_envelope_accepts_any_allowlisted_number():
    # The allowlist is a set, not a single operator — a second number passes too.
    assert _parse_envelope(_envelope(OPERATOR2, "yo"), ALLOW) is not None


def test_parse_envelope_drops_non_allowlisted():
    assert _parse_envelope(_envelope(STRANGER, "intrude"), ALLOW) is None


def test_parse_envelope_empty_allowlist_denies_all():
    assert _parse_envelope(_envelope(OPERATOR, "hi"), frozenset()) is None


def test_parse_envelope_drops_bodiless():
    assert _parse_envelope(_envelope(OPERATOR, None), ALLOW) is None  # receipt
    assert _parse_envelope(_envelope(OPERATOR, ""), ALLOW) is None  # empty body


def test_parse_envelope_drops_garbage():
    assert _parse_envelope({"not": "an envelope"}, ALLOW) is None


# --- receive_once ------------------------------------------------------------


@pytest.mark.asyncio
async def test_receive_once_yields_only_allowlisted_text():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                _envelope(OPERATOR, "first", timestamp=1),
                _envelope(STRANGER, "ignored", timestamp=2),
                _envelope(OPERATOR, None, timestamp=3),  # receipt
                _envelope(OPERATOR2, "second", timestamp=4),
            ],
        )

    async with _client(handler) as client:
        msgs = await client.receive_once()

    assert [m.text for m in msgs] == ["first", "second"]
    assert [m.msg_id for m in msgs] == ["1", "4"]


@pytest.mark.asyncio
async def test_receive_once_raises_on_non_2xx():
    async with _client(lambda req: httpx.Response(500)) as client:
        with pytest.raises(SignalError):
            await client.receive_once()


# --- send --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_posts_correct_v2_send_call():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return httpx.Response(201, json={"timestamp": 1717000000999})

    async with _client(handler) as client:
        await client.send("reply from jeff", recipient=OPERATOR)

    assert seen["path"] == "/v2/send"
    assert seen["body"] == {
        "number": BOT_NUMBER,
        "recipients": [OPERATOR],
        "message": "reply from jeff",
    }


@pytest.mark.asyncio
async def test_send_routes_to_the_given_recipient():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(201)

    async with _client(handler) as client:
        await client.send("hi", recipient=OPERATOR2)
    assert seen["body"]["recipients"] == [OPERATOR2]


@pytest.mark.asyncio
async def test_send_failure_surfaced_not_swallowed():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request: unregistered")

    async with _client(handler) as client:
        with pytest.raises(SignalSendError) as exc_info:
            await client.send("nope", recipient=OPERATOR)

    err = exc_info.value
    assert err.status_code == 400
    assert "unregistered" in (err.body or "")


@pytest.mark.asyncio
async def test_send_transport_error_surfaced():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with _client(handler) as client:
        with pytest.raises(SignalSendError) as exc_info:
            await client.send("nope", recipient=OPERATOR)
    assert exc_info.value.status_code is None


# --- receive loop backoff ----------------------------------------------------


@pytest.mark.asyncio
async def test_receive_loop_backs_off_then_recovers():
    """A failing poll triggers backoff; a later success yields the message."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=[_envelope(OPERATOR, "after recovery", timestamp=7)])

    waits: list[float] = []

    async def fake_clock(delay: float) -> None:
        waits.append(delay)

    async with _client(handler) as client:
        agen = client.receive(poll_interval=5.0, _clock=fake_clock)
        first = await agen.__anext__()
        await agen.aclose()

    assert first.text == "after recovery"
    assert waits[0] == pytest.approx(0.01)  # backoff_initial after the 503
    assert calls["n"] == 2


class _StopLoop(Exception):
    """Sentinel to break out of the otherwise-infinite receive loop in tests."""


@pytest.mark.asyncio
async def test_receive_loop_backoff_caps():
    """Repeated failures grow the backoff but cap at backoff_max."""
    waits: list[float] = []

    async def fake_clock(delay: float) -> None:
        waits.append(delay)
        if len(waits) >= 5:
            raise _StopLoop  # break out of the infinite loop

    async with _client(lambda req: httpx.Response(500)) as client:
        agen = client.receive(_clock=fake_clock)
        with pytest.raises(_StopLoop):
            await agen.__anext__()

    # 0.01, 0.02, 0.04, then capped at backoff_max=0.04.
    assert waits[:3] == [pytest.approx(0.01), pytest.approx(0.02), pytest.approx(0.04)]
    assert max(waits) == pytest.approx(0.04)


# --- construction ------------------------------------------------------------


def test_requires_base_url_or_http():
    with pytest.raises(ValueError):
        SignalCliClient(number=BOT_NUMBER, allowlist=ALLOW)
