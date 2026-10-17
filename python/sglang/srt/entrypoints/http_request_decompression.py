"""Pure-ASGI middleware that decompresses compressed request bodies.

Gated on `SGLANG_ENABLE_REQUEST_DECOMPRESSION`. A body is decompressed when it
carries either the standard `Content-Encoding` header (`gzip`, `x-gzip`,
`deflate`, `zstd`; what OpenAI-compatible gateways send) or the legacy
`x-body-compressed` header naming the method (e.g. `x-body-compressed: zstd`).
"""

import asyncio
import io
import logging
import zlib

import zstandard
from fastapi.responses import Response
from starlette.datastructures import Headers

logger = logging.getLogger(__name__)

# Decompressed-size ceiling: a small compressed body must not be able to
# expand into unbounded memory (decompression bomb).
MAX_DECOMPRESSED_BYTES = 128 * 1024 * 1024


class _TooLarge(ValueError):
    pass


def _zstd_decompress(raw: bytes) -> bytes:
    body = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read(
        MAX_DECOMPRESSED_BYTES + 1
    )
    if len(body) > MAX_DECOMPRESSED_BYTES:
        raise _TooLarge()
    return body


def _zlib_decompress(raw: bytes, wbits: int) -> bytes:
    decompressor = zlib.decompressobj(wbits)
    body = decompressor.decompress(raw, MAX_DECOMPRESSED_BYTES + 1)
    if len(body) > MAX_DECOMPRESSED_BYTES or decompressor.unconsumed_tail:
        raise _TooLarge()
    return body + decompressor.flush()


def _gzip_decompress(raw: bytes) -> bytes:
    return _zlib_decompress(raw, 16 + zlib.MAX_WBITS)


def _deflate_decompress(raw: bytes) -> bytes:
    # RFC 9110 "deflate" is zlib-wrapped; some clients send raw deflate.
    try:
        return _zlib_decompress(raw, zlib.MAX_WBITS)
    except zlib.error:
        return _zlib_decompress(raw, -zlib.MAX_WBITS)


_DECOMPRESSORS = {
    "zstd": _zstd_decompress,
    "gzip": _gzip_decompress,
    "x-gzip": _gzip_decompress,
    "deflate": _deflate_decompress,
}


def _rewrite_headers(headers, new_len):
    """Update headers to reflect body status after decompression."""
    out = [
        (k, v)
        for (k, v) in headers
        if k not in (b"content-length", b"x-body-compressed", b"content-encoding")
    ]
    out.append((b"content-length", str(new_len).encode()))
    return out


def _requested_method(headers: Headers) -> str | None:
    legacy = headers.get("x-body-compressed")
    if legacy is not None:
        return legacy
    encoding = (headers.get("content-encoding") or "").strip().lower()
    if not encoding or encoding == "identity":
        return None
    return encoding


class RequestDecompressionMiddleware:
    """Decompress request bodies per `Content-Encoding` / `x-body-compressed`."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # No-op passthrough for any request without a compression header.
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        method = _requested_method(Headers(scope=scope))
        if method is None:
            return await self.app(scope, receive, send)

        # Fail loud on an unsupported (or stacked, e.g. "gzip, br") method.
        decompress = _DECOMPRESSORS.get(method)
        if decompress is None:
            return await Response(
                f"unsupported request body encoding {method!r}; "
                f"supported: {sorted(_DECOMPRESSORS)}",
                status_code=415,
            )(scope, receive, send)

        # Collect request body.
        body = b""
        more_body = True
        while more_body:
            message = await receive()
            # Incomplete body (e.g. client disconnect); hand off to later stages.
            if message["type"] != "http.request":
                return await self.app(scope, receive, send)
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        # Decompress off the event loop by releasing the GIL around the C decompress.
        try:
            loop = asyncio.get_running_loop()
            body = await loop.run_in_executor(None, decompress, body)
        except _TooLarge:
            return await Response(
                f"decompressed body exceeds {MAX_DECOMPRESSED_BYTES} bytes",
                status_code=413,
            )(scope, receive, send)
        except Exception as e:
            logger.warning("request body decompress failed: %s", e)
            return await Response("decompress failed", status_code=400)(
                scope, receive, send
            )

        # Update the headers after decompression
        scope = dict(scope)
        scope["headers"] = _rewrite_headers(scope["headers"], len(body))

        # Fake receiver to let later stages see the decompressed body.
        body_sent = False

        async def wrapped_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, wrapped_receive, send)
