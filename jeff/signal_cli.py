"""The Signal leg: an async client for signal-cli-rest-api.

Jeff owns a dedicated Signal number through an off-the-shelf
[signal-cli-rest-api](https://github.com/bbernhard/signal-cli-rest-api)
container running in ``json-rpc``/``native`` mode (the ``normal`` mode does not
behave for receive). This module drives that container's HTTP/REST surface: it
**receives** inbound Signal messages and **sends** replies, normalising inbound
traffic to a small internal ``SignalMessage`` shape the front-door loop consumes.

Lifted from the retired ``wingit`` bridge's Signal leg (its Ensemble far-leg is
dead now that Signal lands directly inside Jeff). Generalised from wingit's
single ``operator_number`` to an **allowlist set**, mirroring Jeff's Ensemble
allowlist: default-deny, and any number in the set is treated as an authorised
operator. The Signal protocol authenticates the *sender*; the allowlist
authorises it.

Scope for v1:

  - **Text only.** Attachments are out of scope; inbound messages with no text
    body are skipped, outbound is plain text.
  - **Allowlisted senders only.** Inbound is restricted to the configured
    operator allowlist; anything else is dropped silently (logged at debug).
    An empty allowlist answers nobody (default-deny).
  - **Failures surface.** A non-2xx from ``/v2/send`` raises ``SignalSendError``
    rather than being swallowed — the caller decides how to react. The receive
    loop reconnects with capped exponential backoff so a flaky signal-cli
    (restart, JVM hiccup) doesn't end the front door.

## Provisioning (one-time, operator step — NOT scripted here)

Standing up the front door requires registering the dedicated number with
signal-cli so Jeff OWNS it as *primary* (preferred over linking to a personal
phone — a linked device on a primary number can read all of that account's
chats). This is interactive and handles a secret (the resulting Signal
session/identity keys), so it is an operator runbook step (see the provisioning
runbook), never automated in code:

    # 1. Request a registration code for the DEDICATED number. Signal requires a
    #    CAPTCHA token — solve it at
    #    https://signalcaptchas.org/registration/generate.html and pass it:
    curl -X POST 'http://signal-cli:8080/v1/register/+1XXXXXXXXXX' \
         -H 'Content-Type: application/json' \
         -d '{"captcha": "<token>", "use_voice": false}'

    # 2. Confirm with the SMS/voice code that arrives at that number:
    curl -X POST 'http://signal-cli:8080/v1/register/+1XXXXXXXXXX/verify/123-456'

After verification the container's data dir holds the Signal session/identity
keys — a top-tier secret, persisted to an encrypted volume, never logged. From
then on the number is registered as primary and this client can send/receive.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import httpx

log = logging.getLogger("jeff.signal")


@dataclass(frozen=True, slots=True)
class SignalMessage:
    """A normalised inbound Signal message — the front-door loop's input shape.

    Fields:
      - ``from_number``: E.164 sender number (the ``envelope.source`` /
        ``sourceNumber`` from signal-cli).
      - ``text``: the message body (``dataMessage.message``). Always non-empty
        for messages this client yields — bodiless envelopes (typing
        indicators, receipts, attachment-only) are skipped upstream.
      - ``timestamp``: the Signal envelope timestamp, integer ms since epoch.
      - ``msg_id``: a stable per-message id for dedup. signal-cli has no
        dedicated id field, so we use the envelope timestamp rendered as a
        string (unique per sent message from a given source). The front door
        dedups on ``(from_number, msg_id)``.
    """

    from_number: str
    text: str
    timestamp: int
    msg_id: str


class SignalError(Exception):
    """Base class for Signal-leg failures."""


class SignalSendError(SignalError):
    """A ``/v2/send`` call failed (non-2xx, or transport error).

    Carries the HTTP ``status_code`` (``None`` for a transport-level failure)
    and the response ``body`` if any, so a caller can react without silently
    dropping the reply.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _parse_envelope(item: dict, allowlist: frozenset[str]) -> SignalMessage | None:
    """Normalise one signal-cli receive item to a ``SignalMessage``.

    Returns ``None`` (and logs at debug) when the item is not an allowlisted text
    message we should handle: wrong sender, no data message, or empty body.

    signal-cli wraps the payload under an ``envelope`` key for both the REST
    ``/v1/receive`` endpoint and the receive websocket. Older/newer builds vary
    on ``source`` vs ``sourceNumber``; we accept either.
    """
    envelope = item.get("envelope")
    if not isinstance(envelope, dict):
        log.debug("dropping non-envelope receive item")
        return None

    source = envelope.get("sourceNumber") or envelope.get("source")
    if source not in allowlist:
        # Default-deny: drop anything not from an allowlisted operator, quietly.
        log.debug("dropping message from non-allowlisted sender")
        return None

    data_message = envelope.get("dataMessage")
    if not isinstance(data_message, dict):
        # Receipts, typing indicators, sync messages, etc. — not handleable.
        log.debug("dropping envelope with no dataMessage")
        return None

    text = data_message.get("message")
    if not text:
        # Attachment-only or empty body — out of scope for text-only v1.
        log.debug("dropping bodiless dataMessage")
        return None

    timestamp = int(envelope.get("timestamp") or data_message.get("timestamp") or 0)
    return SignalMessage(
        from_number=source,
        text=text,
        timestamp=timestamp,
        msg_id=str(timestamp),
    )


class SignalCliClient:
    """Async client for one signal-cli-rest-api instance, scoped to one number.

    Bound to a single registered ``number`` (Jeff's dedicated Signal number) and
    an ``allowlist`` of authorised operator numbers. Construct with an explicit
    ``httpx.AsyncClient`` (so tests inject a mock transport), or let it build one
    from ``base_url``.

    Use as an async context manager so the owned HTTP client is closed::

        async with SignalCliClient(
            base_url="http://signal-cli:8080",
            number="+1555...",
            allowlist={"+1444..."},
        ) as signal:
            await signal.send("hi", recipient="+1444...")
            async for msg in signal.receive():
                ...
    """

    def __init__(
        self,
        *,
        number: str,
        allowlist: set[str] | frozenset[str],
        base_url: str | None = None,
        http: httpx.AsyncClient | None = None,
        receive_timeout: float = 30.0,
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
    ) -> None:
        if http is None and base_url is None:
            raise ValueError("provide either base_url or an httpx.AsyncClient")
        self.number = number
        self.allowlist = frozenset(allowlist)
        self._owns_http = http is None
        self._http = http if http is not None else httpx.AsyncClient(base_url=base_url)
        self._receive_timeout = receive_timeout
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max

    async def __aenter__(self) -> SignalCliClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def send(self, text: str, *, recipient: str) -> None:
        """Send a text message via ``POST /v2/send``.

        Sends from Jeff's ``number`` to ``recipient`` (the address of the turn
        being answered). Raises ``SignalSendError`` on any non-2xx response or
        transport error — the failure is never swallowed.
        """
        payload = {
            "number": self.number,
            "recipients": [recipient],
            "message": text,
        }
        try:
            resp = await self._http.post("/v2/send", json=payload)
        except httpx.HTTPError as exc:
            raise SignalSendError(
                f"signal-cli send transport error: {type(exc).__name__}",
            ) from exc

        if resp.status_code // 100 != 2:
            body: str | None
            try:
                body = resp.text
            except Exception:  # pragma: no cover - defensive
                body = None
            raise SignalSendError(
                f"signal-cli send failed: HTTP {resp.status_code}",
                status_code=resp.status_code,
                body=body,
            )

    async def receive_once(self) -> list[SignalMessage]:
        """Poll ``GET /v1/receive/{number}`` once, returning allowlisted messages.

        Drains whatever signal-cli has queued, normalising and allowlist-filtering
        each item. Non-allowlisted / non-text items are dropped silently. Raises
        ``SignalError`` on a non-2xx or transport error so the caller's backoff
        loop can react.
        """
        try:
            resp = await self._http.get(
                f"/v1/receive/{self.number}",
                timeout=self._receive_timeout,
            )
        except httpx.HTTPError as exc:
            raise SignalError(
                f"signal-cli receive transport error: {type(exc).__name__}"
            ) from exc

        if resp.status_code // 100 != 2:
            raise SignalError(f"signal-cli receive failed: HTTP {resp.status_code}")

        items = resp.json()
        if not isinstance(items, list):
            log.debug("unexpected receive payload (not a list)")
            return []

        out: list[SignalMessage] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            msg = _parse_envelope(item, self.allowlist)
            if msg is not None:
                out.append(msg)
        return out

    async def receive(
        self,
        *,
        poll_interval: float = 1.0,
        _clock: Callable[[float], object] = asyncio.sleep,
    ) -> AsyncIterator[SignalMessage]:
        """Yield allowlisted ``SignalMessage``s forever (polling loop).

        This is the resilient receive loop: it polls ``receive_once`` and yields
        each allowlisted message, reconnecting with capped exponential backoff on
        any ``SignalError`` (signal-cli restart, transport blip) so a flaky leg
        never ends the front door. Between successful polls it waits
        ``poll_interval``.

        A websocket-preferred variant (the ``/v1/receive/{number}`` ws upgrade)
        is the intended primary transport; this polling loop is the documented
        fallback and the shape the front door iterates either way. ``_clock`` is
        injectable for tests.
        """
        backoff = self._backoff_initial
        while True:
            try:
                messages = await self.receive_once()
            except SignalError as exc:
                log.warning(
                    "signal-cli receive error, backing off %.1fs: %s",
                    backoff,
                    type(exc).__name__,
                )
                await _clock(backoff)
                backoff = min(backoff * 2, self._backoff_max)
                continue

            backoff = self._backoff_initial
            for msg in messages:
                yield msg
            await _clock(poll_interval)
