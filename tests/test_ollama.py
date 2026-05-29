import httpx
import pytest

from jeff.ollama import Ollama, OllamaError, _safe_excerpt


@pytest.mark.asyncio
async def test_chat_extracts_message_content():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/chat"
        body = req.read()
        assert b'"model":"m"' in body
        assert b'"stream":false' in body
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "hello back"}}
        )

    transport = httpx.MockTransport(handler)
    client = Ollama("http://example")
    client._client = httpx.AsyncClient(base_url="http://example", transport=transport)
    try:
        out = await client.chat([{"role": "user", "content": "hi"}], model="m")
        assert out == "hello back"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_raises_on_http_error():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    client = Ollama("http://example")
    client._client = httpx.AsyncClient(base_url="http://example", transport=transport)
    try:
        with pytest.raises(OllamaError):
            await client.chat([], model="m")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_embed_returns_floats():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/embeddings"
        return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})

    transport = httpx.MockTransport(handler)
    client = Ollama("http://example")
    client._client = httpx.AsyncClient(base_url="http://example", transport=transport)
    try:
        out = await client.embed("hi", model="m")
        assert out == [0.1, 0.2, 0.3]
    finally:
        await client.aclose()


def test_safe_excerpt_strips_control_chars_and_truncates():
    raw = "ok\n\r\x1b[31mred\x1b[0m\x00 next"
    cleaned = _safe_excerpt(raw, limit=64)
    # Newlines / CRs / ANSI escapes / NULs all replaced with `?` — no original
    # control bytes remain in the result.
    for bad in ("\n", "\r", "\x1b", "\x00"):
        assert bad not in cleaned
    assert "red" in cleaned  # printable content survives


def test_safe_excerpt_truncates_long_string():
    raw = "a" * 1000
    out = _safe_excerpt(raw, limit=64)
    assert len(out) <= 64 + len("…(936 more)") + 1
    assert "936 more" in out
    # The first 64 chars should be the body, not the suffix.
    assert out.startswith("a" * 64)


def test_safe_excerpt_short_string_unchanged():
    assert _safe_excerpt("hello world", limit=64) == "hello world"


@pytest.mark.asyncio
async def test_chat_error_sanitises_response_body():
    """Peer-shaped Ollama response body must be truncated + control-stripped."""
    hostile = "x" * 500 + "\n\r\x1b[31m<INJECT>\x1b[0m"

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=hostile)

    transport = httpx.MockTransport(handler)
    client = Ollama("http://example")
    client._client = httpx.AsyncClient(base_url="http://example", transport=transport)
    try:
        with pytest.raises(OllamaError) as ei:
            await client.chat([], model="m")
    finally:
        await client.aclose()

    msg = str(ei.value)
    # The full 500-char body must not be in the exception.
    assert len(msg) < 400
    # Control chars (newline, CR, ANSI ESC) must not survive into the message.
    for bad in ("\n", "\r", "\x1b"):
        assert bad not in msg
    # Status code must be there for operator debuggability.
    assert "500" in msg


@pytest.mark.asyncio
async def test_chat_malformed_does_not_echo_body():
    """A 200 with the wrong JSON shape must not include the raw response in the error."""
    secret_in_body = "USER_PROMPT_ECHOED_BACK_VERBATIM"

    def handler(req: httpx.Request) -> httpx.Response:
        # No "message" key — triggers the malformed branch.
        return httpx.Response(200, json={"echo": secret_in_body})

    transport = httpx.MockTransport(handler)
    client = Ollama("http://example")
    client._client = httpx.AsyncClient(base_url="http://example", transport=transport)
    try:
        with pytest.raises(OllamaError) as ei:
            await client.chat([], model="m")
    finally:
        await client.aclose()

    assert secret_in_body not in str(ei.value)
    # Only the top-level key list should leak — that's a shape diagnostic, not data.
    assert "echo" in str(ei.value)


@pytest.mark.asyncio
async def test_chat_refuses_oversize_content_length():
    """Pre-stream Content-Length pre-check rejects without reading the body."""

    big = 10 * 1024 * 1024  # 10 MiB declared

    def handler(req: httpx.Request) -> httpx.Response:
        # Body is small but Content-Length lies — our pre-check should fire
        # on the declared size before any bytes are read.
        return httpx.Response(
            200,
            content=b"{}",
            headers={"content-length": str(big)},
        )

    transport = httpx.MockTransport(handler)
    client = Ollama("http://example", max_resp_bytes=1024)
    client._client = httpx.AsyncClient(base_url="http://example", transport=transport)
    try:
        with pytest.raises(OllamaError, match="Content-Length"):
            await client.chat([], model="m")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_refuses_oversize_streamed_body():
    """When Content-Length is absent/wrong, the streaming guard fires mid-body."""

    huge = b"a" * (16 * 1024)  # 16 KB body, cap is 1 KB

    def handler(req: httpx.Request) -> httpx.Response:
        # No content-length header: httpx will set Transfer-Encoding: chunked
        # for in-memory bytes, or the test path will just stream the content.
        return httpx.Response(200, content=huge)

    transport = httpx.MockTransport(handler)
    client = Ollama("http://example", max_resp_bytes=1024)
    client._client = httpx.AsyncClient(base_url="http://example", transport=transport)
    try:
        with pytest.raises(OllamaError) as ei:
            await client.chat([], model="m")
        # Either guard is acceptable — the test verifies that *some* size cap
        # fires before the 16 KB body is fully read.
        assert "1024" in str(ei.value) or "exceed" in str(ei.value).lower()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_embed_refuses_oversize_dim_before_float_conversion():
    """Even a small JSON body can pack a giant array — cap the list length."""

    # Build an embedding of 16k integers — fits well under the response-size
    # cap as JSON text but blows past the embed-dim cap.
    bloated = list(range(16384))

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": bloated})

    transport = httpx.MockTransport(handler)
    client = Ollama("http://example", max_resp_bytes=10 * 1024 * 1024, max_embed_dim=8192)
    client._client = httpx.AsyncClient(base_url="http://example", transport=transport)
    try:
        with pytest.raises(OllamaError, match="exceeds max 8192"):
            await client.embed("hi", model="m")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_embed_raises_on_malformed():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    transport = httpx.MockTransport(handler)
    client = Ollama("http://example")
    client._client = httpx.AsyncClient(base_url="http://example", transport=transport)
    try:
        with pytest.raises(OllamaError):
            await client.embed("hi", model="m")
    finally:
        await client.aclose()
