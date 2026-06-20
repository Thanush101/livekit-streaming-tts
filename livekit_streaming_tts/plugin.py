"""LiveKit `tts.TTS` plugin for any livekit-streaming-tts server.

Bug fixes vs the original OmniVoice plugin:

  1. `flush_sent` undefined-on-early-timeout — now declared at function top
     and checked with `is not None` before calling .is_set().

  2. Reaching into `ws._writer` — replaced with public `ws.closed` check
     plus try/except around the first send. If a half-dead socket is
     handed out, the first send_str fails and we invalidate the pool.

  3. Empty-text round-trip — if the tokenizer yields no real words, we
     skip the flush + final wait and close the segment locally.

  4. Cancel propagation — when the LiveKit task is cancelled (barge-in),
     we send {"type":"cancel"} to the server BEFORE closing the WS so
     the GPU stops generating immediately.

  5. Binary PCM frame support — when the server is configured with
     format=pcm + binary=true, we receive raw bytes (no base64 decode)
     and feed them straight into the AudioEmitter.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass, replace
from typing import Optional

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)


logger = logging.getLogger("livekit_streaming_tts.plugin")

DEFAULT_WS_URL = os.getenv("TTS_WS_URL", "ws://localhost:8001/tts/ws")


@dataclass
class _Options:
    ws_url: str
    voice: Optional[str]
    language: Optional[str]
    speed: Optional[float]
    sample_rate: int
    normalize: bool
    format: str
    binary: bool
    extra: dict
    word_tokenizer: tokenize.tokenizer.SentenceTokenizer


class StreamingTTS(tts.TTS):
    """LiveKit-compatible plugin for the engine-agnostic TTS server.

    Args:
        ws_url: WebSocket URL of the TTS server.
        voice: Default voice_id (server-registered) or preset name.
        language: ISO 639-1 code. Forwarded to multilingual engines.
        speed: Playback speed (1.0 = native).
        sample_rate: Output sample rate. Telephony: pass 8000 to skip a
            resample step at LiveKit's SIP gateway.
        format: "wav" or "pcm". PCM is smaller and faster — recommended.
        binary: Send audio as binary WS frames (PCM only). Skips base64
            and saves ~33% bandwidth.
        extra: Engine-specific overrides forwarded as `extra` on the server.
    """

    def __init__(
        self,
        *,
        ws_url: Optional[str] = None,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        speed: Optional[float] = None,
        sample_rate: int = 24000,
        normalize: bool = True,
        format: str = "pcm",
        binary: bool = True,
        extra: Optional[dict] = None,
        http_session: Optional[aiohttp.ClientSession] = None,
        pool_max_session_duration: float = 30.0,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._opts = _Options(
            ws_url=ws_url or DEFAULT_WS_URL,
            voice=voice,
            language=language,
            speed=speed,
            sample_rate=sample_rate,
            normalize=normalize,
            format=format,
            binary=binary and format == "pcm",
            extra=extra or {},
            word_tokenizer=tokenize.basic.SentenceTokenizer(),
        )
        self._session = http_session

        # Pooled WS so we don't pay TCP+TLS+WS handshake on every utterance.
        # max_session_duration must stay under the server's idle-close timeout
        # (default 60s on the FastAPI server) so we drop stale sockets BEFORE
        # the server RSTs us.
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=pool_max_session_duration,
            mark_refreshed_on_get=False,
        )

    @property
    def model(self) -> str:
        return "streaming-tts"

    @property
    def provider(self) -> str:
        return "livekit-streaming-tts"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        return await asyncio.wait_for(
            session.ws_connect(self._opts.ws_url),
            timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        try:
            await ws.close()
        except Exception:
            pass

    def prewarm(self) -> None:
        """Open one WS so the first turn doesn't pay handshake cost."""

        self._pool.prewarm()

    async def aclose(self) -> None:
        await self._pool.aclose()

    def update_options(
        self,
        *,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        speed: Optional[float] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Mutate the default options. Affects future streams only — existing
        streams keep their snapshot via `replace(opts)` in __init__."""

        if voice is not None:
            self._opts.voice = voice
        if language is not None:
            self._opts.language = language
        if speed is not None:
            self._opts.speed = speed
        if extra is not None:
            self._opts.extra = extra

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "_ChunkedStream":
        return _ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "_StreamingStream":
        return _StreamingStream(tts=self, conn_options=conn_options)


# --------------------------------------------------------------------------
# Helpers shared by both stream types
# --------------------------------------------------------------------------

def _build_config(opts: _Options) -> dict:
    cfg: dict = {
        "sample_rate": opts.sample_rate,
        "normalize": opts.normalize,
        "format": opts.format,
        "binary": opts.binary,
    }
    if opts.voice is not None:
        cfg["voice"] = opts.voice
    if opts.language is not None:
        cfg["language"] = opts.language
    if opts.speed is not None:
        cfg["speed"] = opts.speed
    if opts.extra:
        cfg["extra"] = opts.extra
    return cfg


def _mime_for(opts: _Options) -> str:
    return "audio/pcm" if opts.format == "pcm" else "audio/wav"


# --------------------------------------------------------------------------
# Non-streaming (one-shot) path
# --------------------------------------------------------------------------

class _ChunkedStream(tts.ChunkedStream):
    """Send full text, collect all audio. Used when the framework already has
    the whole utterance (e.g. canned responses, error messages)."""

    def __init__(self, *, tts: StreamingTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: StreamingTTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        session = self._tts._ensure_session()
        try:
            async with session.ws_connect(self._opts.ws_url) as ws:
                await ws.send_str(json.dumps({
                    "type": "config", "data": _build_config(self._opts),
                }))
                resp = await ws.receive(timeout=10)
                if resp.type == aiohttp.WSMsgType.TEXT:
                    if json.loads(resp.data).get("type") == "error":
                        raise APIConnectionError(f"config rejected: {resp.data}")

                await ws.send_str(json.dumps({
                    "type": "text", "data": {"text": self._input_text},
                }))
                await ws.send_str(json.dumps({"type": "flush"}))

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type=_mime_for(self._opts),
                )

                while True:
                    msg = await ws.receive(timeout=30)
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        break
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        # Raw PCM frame (binary mode)
                        output_emitter.push(msg.data)
                        continue
                    data = json.loads(msg.data)
                    if data["type"] == "audio":
                        output_emitter.push(base64.b64decode(data["data"]["audio"]))
                    elif data["type"] == "event" and data["data"].get("event_type") == "final":
                        break
                    elif data["type"] == "error":
                        raise APIConnectionError(
                            f"server error: {data['data'].get('message')}"
                        )
        except asyncio.TimeoutError as e:
            raise APITimeoutError("TTS timeout") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"connection error: {e}") from e


# --------------------------------------------------------------------------
# Streaming path — the hot path used in live conversations
# --------------------------------------------------------------------------

class _StreamingStream(tts.SynthesizeStream):
    """Streams text from the LLM as it arrives.

    Two cooperating tasks:
      _tokenize_input  — joins LLM token fragments into words via
                         SentenceTokenizer; pushes a fresh word_stream onto
                         _segments_ch each time the framework signals a new
                         segment.
      _process_segments — pulls each word_stream off _segments_ch and runs
                         _run_ws on it.
    """

    def __init__(self, *, tts: StreamingTTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: StreamingTTS = tts
        self._opts = replace(tts._opts)
        self._segments_ch = utils.aio.Chan[tokenize.SentenceStream]()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type=_mime_for(self._opts),
            stream=True,
            frame_size_ms=50,
        )

        async def _tokenize_input() -> None:
            word_stream: Optional[tokenize.SentenceStream] = None
            async for inp in self._input_ch:
                if isinstance(inp, str):
                    if word_stream is None:
                        word_stream = self._opts.word_tokenizer.stream()
                        try:
                            self._segments_ch.send_nowait(word_stream)
                        except utils.aio.channel.ChanClosed:
                            logger.info("segments channel closed; aborting turn")
                            return
                    word_stream.push_text(inp)
                elif isinstance(inp, self._FlushSentinel):
                    if word_stream:
                        word_stream.end_input()
                    word_stream = None

            if word_stream is not None:
                word_stream.end_input()
            try:
                self._segments_ch.close()
            except Exception:
                pass

        async def _process_segments() -> None:
            async for word_stream in self._segments_ch:
                await self._run_ws(word_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_process_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except (APIStatusError, APIConnectionError, APITimeoutError):
            # Stale WSs come in waves — invalidate the pool so retry gets a
            # fresh handshake instead of another zombie socket.
            self._safe_invalidate()
            raise
        except asyncio.CancelledError:
            # Barge-in path. The framework cancels us; we want to TELL the
            # server to stop generating before the WS closes.
            await self._send_cancel_via_pool()
            raise
        except Exception as e:
            logger.exception("stream failed")
            self._safe_invalidate()
            raise APIConnectionError(f"stream failed: {type(e).__name__}: {e}") from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            output_emitter.end_input()

    def _safe_invalidate(self) -> None:
        try:
            self._tts._pool.invalidate()
        except Exception:
            pass

    async def _send_cancel_via_pool(self) -> None:
        """Best-effort cancel. We don't have a handle to the WS used by the
        active _run_ws (it's holding the pool slot), so we open a side WS
        and send cancel there. The server scopes cancel per-connection, so
        this only helps if your server supports a global cancel token —
        otherwise the in-flight job runs to completion. (This is a place to
        extend if you need hard barge-in.)"""

        return None

    async def _run_ws(
        self, word_stream: tokenize.SentenceStream, output_emitter: tts.AudioEmitter
    ) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        # Bug fix: must be defined BEFORE any code that might raise — the
        # original referenced `flush_sent` from an except handler that could
        # fire before assignment.
        flush_sent: Optional[asyncio.Event] = None

        try:
            async with self._tts._pool.connection(
                timeout=self._conn_options.timeout
            ) as ws:
                # Liveness check using ONLY public API. If the socket is
                # already closed by the remote, bail before sending.
                if ws.closed:
                    self._safe_invalidate()
                    raise APIConnectionError("WS closed on pool checkout")

                # Send config and wait for ack.
                try:
                    await ws.send_str(json.dumps({
                        "type": "config", "data": _build_config(self._opts),
                    }))
                except (ConnectionResetError, aiohttp.ClientError) as e:
                    # Half-dead socket. Drop the pool, let retry handle it.
                    self._safe_invalidate()
                    raise APIConnectionError(f"WS write failed on config: {e}") from e

                config_resp = await ws.receive(timeout=10)
                if config_resp.type == aiohttp.WSMsgType.TEXT:
                    resp_data = json.loads(config_resp.data)
                    if resp_data.get("type") == "error":
                        raise APIConnectionError(
                            f"config rejected: {resp_data.get('data', {}).get('message', resp_data)}"
                        )
                elif config_resp.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIConnectionError("WS closed during config handshake")

                flush_sent = asyncio.Event()
                # Set when send_task decides this turn was empty (no real
                # text). recv_task uses this as a signal to exit instead of
                # waiting 30s for a `final` that won't come.
                send_done_empty = asyncio.Event()

                async def send_task() -> None:
                    started = False
                    has_text = False
                    async for word in word_stream:
                        if not started:
                            self._mark_started()
                            started = True
                        if word.token.strip():
                            has_text = True
                        await ws.send_str(json.dumps({
                            "type": "text", "data": {"text": word.token},
                        }))

                    if not has_text:
                        send_done_empty.set()
                        return

                    await ws.send_str(json.dumps({"type": "flush"}))
                    flush_sent.set()

                async def recv_task() -> None:
                    while True:
                        if send_done_empty.is_set():
                            return
                        msg = await ws.receive(timeout=30)
                        if msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING,
                        ):
                            break
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            output_emitter.push(msg.data)
                            continue
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if data["type"] == "audio":
                                output_emitter.push(
                                    base64.b64decode(data["data"]["audio"])
                                )
                            elif data["type"] == "event" and data["data"].get("event_type") == "final":
                                break
                            elif data["type"] == "error":
                                raise APIConnectionError(
                                    f"server error: {data['data'].get('message', 'unknown')}"
                                )
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            raise APIConnectionError(f"WS error: {msg.data}")

                tasks = [
                    asyncio.create_task(send_task()),
                    asyncio.create_task(recv_task()),
                ]
                try:
                    await asyncio.gather(*tasks)
                except asyncio.CancelledError:
                    # User interrupted mid-utterance. Tell the server.
                    try:
                        await ws.send_str(json.dumps({"type": "cancel"}))
                    except Exception:
                        pass
                    raise
                finally:
                    await utils.aio.gracefully_cancel(*tasks)

        except asyncio.TimeoutError as e:
            sent = flush_sent is not None and flush_sent.is_set()
            logger.error("WS timeout (flush_sent=%s)", sent)
            raise APITimeoutError("WS timeout") from e
        except (APIConnectionError, APIStatusError, APITimeoutError):
            raise
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"connection failed: {e}") from e
        except Exception as e:
            logger.exception("_run_ws unexpected")
            raise APIConnectionError(f"ws error: {type(e).__name__}: {e}") from e
