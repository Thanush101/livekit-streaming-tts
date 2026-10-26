"""Engine-agnostic FastAPI server.

What this fixes vs the original OmniVoice server:

  1. Normalization moved OFF the GPU worker thread (was holding the queue
     slot for 5–20ms; now runs in the WS handler thread before enqueue).

  2. PCM-over-binary-WS support — reduces wire size by ~33% and skips
     base64+JSON parse on the client. Telephony pipelines that need 8kHz
     PCM benefit the most (smaller frames, faster decode).

  3. LRU voice cache — was unbounded, now configurable cap. Prevents OOM
     when users register lots of voices.

  4. register_voice gracefully refuses on engines without cloning instead
     of silently no-op'ing.

  5. Cancel signal — client can send {"type": "cancel"} to stop in-flight
     generation when the user interrupts (barge-in). Without this, the GPU
     keeps producing audio for an utterance the user already cut off.

  6. Unicode-aware sentence chunking (Hindi danda, CJK 。, Arabic ؟, ...).

  7. Restart-storm protection — server returns 503 until the engine is
     loaded, so 50 reconnecting clients don't pile up while torch warms.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import io
import json
import logging
import os
import time
import uuid

import numpy as np
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ._paths import data_dir
from .audio import encode, resample
from .base import AudioChunk, GenerationParams, TTSEngine
from .normalize import is_speakable, normalize_text, split_sentences
from .pauses import silence_samples, speech_plan
from .registry import load_engine

logger = logging.getLogger("livekit_streaming_tts.server")


# --------------------------------------------------------------------------
# GPU worker — serializes engine access, keeps the event loop responsive.
# --------------------------------------------------------------------------

@dataclass
class _Job:
    params: GenerationParams
    cancel_event: asyncio.Event
    result_future: asyncio.Future = field(
        default_factory=lambda: asyncio.get_event_loop().create_future()
    )


class GPUWorker:
    """Single-consumer queue feeding one engine instance.

    Why one engine for the whole process:
        TTS models are big (multi-GB weights). One copy serves N concurrent
        clients. CUDA also doesn't truly parallelize across Python threads —
        serializing through a queue is faster than thrashing.

    Why a Future per job:
        FastAPI WS handlers are async; the engine is sync (runs on a thread).
        The Future bridges them — handler awaits, worker resolves.
    """

    def __init__(self, engine: TTSEngine, *, max_queue_size: int = 50, voice_cache_max: int = 64):
        self._engine = engine
        self._queue: asyncio.Queue[_Job] = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: Optional[asyncio.Task] = None
        self._active_connections = 0
        self._total_generated = 0

        # LRU voice cache (engine-side). Bounded to prevent OOM if users
        # upload lots of voices. Most-recently-used stays warm.
        self._voice_lru: OrderedDict[str, None] = OrderedDict()
        self._voice_cache_max = voice_cache_max

    @property
    def engine(self) -> TTSEngine:
        return self._engine

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def active_connections(self) -> int:
        return self._active_connections

    def list_voices(self) -> list[str]:
        return self._engine.list_voices()

    def register_voice(
        self, voice_id: str, audio_path: str, ref_text: Optional[str] = None
    ) -> None:
        if not self._engine.capabilities.voice_cloning:
            raise HTTPException(
                status_code=400,
                detail=f"engine {self._engine.name!r} does not support voice cloning",
            )
        # voice_id is normalized to lowercase to avoid path-collision bugs
        # where Voice.mp3 and voice.mp3 land at the same path.
        voice_id = voice_id.lower().strip()
        self._engine.register_voice(voice_id, audio_path, ref_text)

        # LRU bookkeeping — evict the coldest voice if over cap.
        self._voice_lru.pop(voice_id, None)
        self._voice_lru[voice_id] = None
        while len(self._voice_lru) > self._voice_cache_max:
            evicted, _ = self._voice_lru.popitem(last=False)
            self._engine.delete_voice(evicted)
            logger.info("evicted cold voice from LRU: %s", evicted)

    def delete_voice(self, voice_id: str) -> None:
        voice_id = voice_id.lower().strip()
        self._voice_lru.pop(voice_id, None)
        self._engine.delete_voice(voice_id)

    async def start(self) -> None:
        self._worker_task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._engine.close()

    async def _loop(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            try:
                job = await self._queue.get()
                if job.cancel_event.is_set():
                    # Client interrupted before we got to it.
                    job.result_future.set_result(None)
                    self._queue.task_done()
                    continue

                start = time.perf_counter()
                try:
                    audio = await loop.run_in_executor(
                        None, self._generate_sync, job
                    )
                    elapsed = time.perf_counter() - start
                    self._total_generated += 1
                    logger.debug(
                        "generated %.3fs of audio in %.3fs",
                        len(audio.samples) / audio.sample_rate if audio else 0,
                        elapsed,
                    )
                    job.result_future.set_result(audio)
                except Exception as e:
                    logger.exception("generation failed")
                    job.result_future.set_exception(e)
                finally:
                    self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("worker loop error")

    def _generate_sync(self, job: _Job):
        """Runs in the executor thread. Final cancel check before GPU work."""

        if job.cancel_event.is_set():
            return None
        # Touch LRU on use — keeps active voices hot.
        if job.params.voice and job.params.voice in self._voice_lru:
            self._voice_lru.move_to_end(job.params.voice)
        return self._engine.generate(job.params)

    async def submit(self, job: _Job):
        await self._queue.put(job)
        return await job.result_future


# --------------------------------------------------------------------------
# FastAPI surface
# --------------------------------------------------------------------------

VOICES_DIR = data_dir("voices", env_var="TTS_VOICES_DIR")


# Module-level so FastAPI's OpenAPI schema generator can resolve forward refs
# when /docs hits /openapi.json. Defining it inside create_app() works at
# runtime but breaks Pydantic v2's TypeAdapter resolution.
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    language: Optional[str] = None
    speed: Optional[float] = None
    seed: Optional[int] = None
    sample_rate: Optional[int] = None
    normalize: bool = True
    pauses: bool = False
    format: str = "wav"  # "wav" or "pcm"
    extra: dict = {}


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return default


def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Token from ``Authorization: Bearer <key>``. Scheme compare is case-insensitive."""

    if not isinstance(authorization, str) or authorization == "":
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    if token == "":
        return None
    return token


def _presented_keys(authorization: Optional[str], x_api_key: Optional[str]) -> list[str]:
    found: list[str] = []
    if isinstance(x_api_key, str):
        header_key = x_api_key.strip()
        if header_key:
            found.append(header_key)
    bearer = _bearer_token(authorization)
    if bearer is not None:
        found.append(bearer)
    return found


def credentials_accepted(
    expected: Optional[str],
    authorization: Optional[str] = None,
    x_api_key: Optional[str] = None,
) -> bool:
    """Whether this request may synthesize.

    ``expected`` is the ``TTS_API_KEY`` value. ``None`` or ``""`` leaves the
    server open, which is the local-dev behavior. Otherwise the client must
    send ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``. The compare
    is :func:`hmac.compare_digest`. This does not read a key from a JSON body.
    """

    if expected is None or expected == "":
        return True
    presented = _presented_keys(authorization, x_api_key)
    if not presented:
        return False
    # Always run compare_digest. A boolean ``or`` after it must not skip the call.
    matched = False
    for candidate in presented:
        matched = hmac.compare_digest(candidate, expected) or matched
    return matched


def _require_api_key(headers) -> None:
    """401 JSON when a key is configured and the headers do not carry it.

    The header values are not logged.
    """

    if credentials_accepted(
        os.environ.get("TTS_API_KEY"),
        authorization=headers.get("authorization"),
        x_api_key=headers.get("x-api-key"),
    ):
        return
    raise HTTPException(status_code=401, detail="Unauthorized")


async def _reject_websocket(ws: WebSocket) -> None:
    """Refuse the upgrade with 401. The socket is never accepted."""

    await ws.send_denial_response(
        JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    )


def _plan_text(text: str, *, pauses: bool, normalize: bool, language: Optional[str]) -> list:
    """Spoken spans and silence seconds. Markers are split off before normalize."""

    return speech_plan(
        text,
        pauses=pauses,
        normalize=normalize,
        language=language,
        normalize_fn=normalize_text,
        speakable=is_speakable,
    )


def create_app(
    *,
    engine: Optional[TTSEngine] = None,
    engine_name: Optional[str] = None,
    engine_kwargs: Optional[dict] = None,
    max_queue_size: int = 50,
    voice_cache_max: int = 64,
) -> FastAPI:
    """Build a FastAPI app with one engine instance.

    Pass `engine` to inject a pre-built engine (for tests), or `engine_name`
    to load via the registry. Without either, falls back to env var
    TTS_ENGINE.
    """

    state = {"worker": None, "ready": False}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        eng = engine
        if eng is None:
            eng = load_engine(engine_name, **(engine_kwargs or {}))

        # Pre-load any voices persisted in VOICES_DIR.
        if eng.capabilities.voice_cloning and os.path.isdir(VOICES_DIR):
            for fname in os.listdir(VOICES_DIR):
                if not fname.endswith(".json"):
                    continue
                voice_id = fname[:-5]
                meta_path = os.path.join(VOICES_DIR, fname)
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    audio_path = os.path.join(VOICES_DIR, meta["audio_file"])
                    if os.path.exists(audio_path):
                        eng.register_voice(voice_id, audio_path, meta.get("ref_text") or None)
                        logger.info("preloaded voice %s", voice_id)
                except Exception:
                    logger.exception("failed preload of %s", voice_id)

        worker = GPUWorker(
            eng,
            max_queue_size=max_queue_size,
            voice_cache_max=voice_cache_max,
        )
        await worker.start()
        state["worker"] = worker
        state["ready"] = True
        logger.info("engine %s ready (sr=%d)", eng.name, eng.native_sample_rate)
        yield
        state["ready"] = False
        await worker.stop()

    app = FastAPI(title="livekit-streaming-tts", lifespan=lifespan)

    def _worker() -> GPUWorker:
        if not state["ready"]:
            # Reject loudly during startup. Clients that respect 503 will back
            # off instead of slamming the half-loaded server.
            raise HTTPException(status_code=503, detail="engine warming up")
        return state["worker"]  # type: ignore[return-value]

    # ---- Health & metadata --------------------------------------------

    @app.get("/health")
    async def health():
        if not state["ready"]:
            raise HTTPException(status_code=503, detail="warming up")
        w = state["worker"]
        return {
            "status": "ok",
            "engine": w.engine.name,
            "native_sample_rate": w.engine.native_sample_rate,
            "capabilities": w.engine.capabilities.__dict__,
            "queue_size": w.queue_size,
            "active_connections": w.active_connections,
            "voices": w.list_voices(),
        }

    @app.get("/v1/voices")
    async def list_voices_ep(request: Request):
        _require_api_key(request.headers)
        return {"voices": _worker().list_voices()}

    # ---- Voice management ---------------------------------------------

    @app.post("/v1/voices/upload")
    async def upload_voice(
        request: Request,
        file: UploadFile = File(...),
        voice_id: str = Form("default"),
        ref_text: str = Form(""),
    ):
        _require_api_key(request.headers)
        worker = _worker()
        if not worker.engine.capabilities.voice_cloning:
            raise HTTPException(
                status_code=400,
                detail=f"engine {worker.engine.name!r} does not support voice cloning",
            )

        os.makedirs(VOICES_DIR, exist_ok=True)
        voice_id = voice_id.lower().strip()
        ext = os.path.splitext(file.filename or "")[1] or ".wav"
        dest = os.path.join(VOICES_DIR, f"{voice_id}{ext}")

        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)

        meta = {"audio_file": f"{voice_id}{ext}", "ref_text": ref_text or ""}
        with open(os.path.join(VOICES_DIR, f"{voice_id}.json"), "w") as f:
            json.dump(meta, f)

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, worker.register_voice, voice_id, dest, ref_text or None
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"voice_id": voice_id}

    @app.delete("/v1/voices/{voice_id}")
    async def delete_voice(voice_id: str, request: Request):
        _require_api_key(request.headers)
        worker = _worker()
        worker.delete_voice(voice_id)
        meta = os.path.join(VOICES_DIR, f"{voice_id}.json")
        if os.path.exists(meta):
            os.remove(meta)
        return {"deleted": voice_id}

    # ---- One-shot HTTP TTS --------------------------------------------

    @app.post("/v1/tts")
    async def http_tts(req: TTSRequest, request: Request):
        _require_api_key(request.headers)
        worker = _worker()
        try:
            plan = _plan_text(
                req.text,
                pauses=req.pauses,
                normalize=req.normalize,
                language=req.language,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not plan:
            return {"audio": "", "sample_rate": req.sample_rate or 0, "format": req.format}

        native = worker.engine.native_sample_rate
        pieces: list[np.ndarray] = []
        for part in plan:
            if isinstance(part, float):
                samples = silence_samples(part, native)
                if samples.size:
                    pieces.append(samples)
                continue
            params = GenerationParams(
                text=part,
                voice=req.voice,
                ref_audio=req.ref_audio,
                ref_text=req.ref_text,
                language=req.language,
                speed=req.speed,
                seed=req.seed,
                sample_rate=req.sample_rate,
                extra=req.extra,
            )
            job = _Job(params=params, cancel_event=asyncio.Event())
            try:
                audio = await worker.submit(job)
            except asyncio.QueueFull:
                raise HTTPException(status_code=503, detail="server overloaded")
            if audio is None or getattr(audio.samples, "size", 0) == 0:
                continue
            chunk = np.asarray(audio.samples, dtype=np.float32)
            if audio.sample_rate != native:
                chunk = resample(chunk, audio.sample_rate, native)
            pieces.append(chunk)

        if not pieces:
            return {"audio": "", "sample_rate": req.sample_rate or 0, "format": req.format}

        mixed = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        encoded, sr = encode(
            mixed,
            src_sample_rate=native,
            target_sample_rate=req.sample_rate,
            format=req.format,
        )
        return {
            "audio": base64.b64encode(encoded).decode(),
            "sample_rate": sr,
            "format": req.format,
        }

    # ---- WebSocket streaming TTS --------------------------------------
    #
    # Protocol (JSON text frames C↔S, optional binary frames S→C for PCM):
    #
    #   C→S {"type":"config","data":{...}}        ← session config
    #   C→S {"type":"text","data":{"text":"..."}} ← text fragment
    #   C→S {"type":"flush"}                      ← end of utterance
    #   C→S {"type":"cancel"}                     ← barge-in / interrupt
    #   C→S {"type":"ping"}
    #
    #   S→C {"type":"event","data":{"event_type":"config_accepted"}}
    #   S→C {"type":"audio","data":{"audio":"<b64>","sample_rate":24000}}
    #         OR a binary WS frame containing raw PCM (when format=pcm and
    #         binary=true in config — saves base64 overhead).
    #   S→C {"type":"event","data":{"event_type":"final"}}
    #   S→C {"type":"error","data":{"message":"...","code":"..."}}
    #   S→C {"type":"pong"}

    @app.websocket("/tts/ws")
    async def tts_ws(ws: WebSocket):
        # Reject before accept. A key inside a later JSON frame is not consulted,
        # so a client cannot open the socket and then stall.
        if not credentials_accepted(
            os.environ.get("TTS_API_KEY"),
            authorization=ws.headers.get("authorization"),
            x_api_key=ws.headers.get("x-api-key"),
        ):
            await _reject_websocket(ws)
            return

        if not state["ready"]:
            await ws.close(code=1013, reason="warming up")
            return

        await ws.accept()
        worker = state["worker"]
        worker._active_connections += 1
        request_id = str(uuid.uuid4())[:8]
        logger.info("[%s] connected (active=%d)", request_id, worker.active_connections)

        # Per-session state
        config = {
            "voice": None,
            "language": None,
            "speed": None,
            "seed": None,
            "sample_rate": worker.engine.native_sample_rate,
            "normalize": True,
            "pauses": False,      # splice [pause] / <break> as exact zeros
            "format": "wav",      # "wav" or "pcm"
            "binary": False,      # if True + format=pcm, send PCM as binary frames
            "extra": {},
        }
        text_buffer = ""

        # Cancel signaling: each in-flight job gets the same cancel_event.
        # When the client sends {"type":"cancel"}, we set it; the worker
        # checks it before starting GPU work and any pending future returns
        # None instead of audio.
        cancel_event = asyncio.Event()

        async def _send_audio(audio, sr_target: int):
            """Encode and send one audio chunk."""

            if audio is None:
                return
            encoded, sr = encode(
                audio.samples,
                src_sample_rate=audio.sample_rate,
                target_sample_rate=sr_target,
                format=config["format"],
            )
            if config["binary"] and config["format"] == "pcm":
                # Binary frame — no JSON, no base64. Fastest path.
                await ws.send_bytes(encoded)
            else:
                await ws.send_text(json.dumps({
                    "type": "audio",
                    "data": {
                        "audio": base64.b64encode(encoded).decode(),
                        "sample_rate": sr,
                        "format": config["format"],
                    },
                }))

        async def _generate_and_send(text: str):
            nonlocal cancel_event
            # Pause markers are split off before normalize. With pauses off,
            # the text is left as-is (still normalized when that flag is on)
            # and no silence is inserted.
            try:
                plan = _plan_text(
                    text,
                    pauses=bool(config["pauses"]),
                    normalize=bool(config["normalize"]),
                    language=config["language"],
                )
            except ValueError as exc:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "data": {"message": str(exc), "code": "invalid_text"},
                }))
                return

            native = worker.engine.native_sample_rate
            for part in plan:
                if cancel_event.is_set():
                    return
                if isinstance(part, float):
                    samples = silence_samples(part, native)
                    if samples.size == 0:
                        continue
                    await _send_audio(
                        AudioChunk(samples=samples, sample_rate=native, is_final=True),
                        config["sample_rate"],
                    )
                    continue

                params = GenerationParams(
                    text=part,
                    voice=config["voice"],
                    language=config["language"],
                    speed=config["speed"],
                    seed=config["seed"],
                    sample_rate=config["sample_rate"],
                    extra=config["extra"],
                )
                job = _Job(params=params, cancel_event=cancel_event)
                try:
                    audio = await worker.submit(job)
                except asyncio.QueueFull:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "data": {"message": "queue full", "code": "queue_full"},
                    }))
                    return
                await _send_audio(audio, config["sample_rate"])

        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "config":
                    data = msg.get("data", {})
                    if not isinstance(data, dict):
                        data = {}
                    for key in config:
                        if key == "pauses":
                            continue
                        if key in data:
                            config[key] = data[key]
                    if "pauses" in data:
                        config["pauses"] = _as_bool(data["pauses"])
                    # Validate sample_rate range — common telephony rates
                    # are 8000/16000; max sensible is the engine's native.
                    sr = config["sample_rate"]
                    if not isinstance(sr, int) or sr < 8000 or sr > 48000:
                        config["sample_rate"] = worker.engine.native_sample_rate
                    await ws.send_text(json.dumps({
                        "type": "event",
                        "data": {"event_type": "config_accepted", "request_id": request_id},
                    }))

                elif msg_type == "text":
                    text_buffer += msg.get("data", {}).get("text", "")
                    sentences = split_sentences(text_buffer)
                    if len(sentences) > 1:
                        # Dispatch all complete sentences immediately; keep
                        # the (possibly incomplete) tail.
                        for s in sentences[:-1]:
                            await _generate_and_send(s)
                            if cancel_event.is_set():
                                break
                        text_buffer = sentences[-1]

                elif msg_type == "flush":
                    if is_speakable(text_buffer):
                        await _generate_and_send(text_buffer)
                    text_buffer = ""
                    await ws.send_text(json.dumps({
                        "type": "event",
                        "data": {"event_type": "final", "request_id": request_id},
                    }))
                    # Reset cancel_event for the next utterance — a single WS
                    # session can carry many turns.
                    cancel_event = asyncio.Event()

                elif msg_type == "cancel":
                    # Barge-in. Stop dispatching new text, drop the buffer,
                    # let any in-flight job return None.
                    cancel_event.set()
                    text_buffer = ""

                elif msg_type == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))

                else:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "data": {"message": f"unknown type: {msg_type}", "code": "invalid_type"},
                    }))

        except WebSocketDisconnect:
            logger.info("[%s] disconnected", request_id)
        except json.JSONDecodeError as e:
            try:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "data": {"message": f"invalid JSON: {e}", "code": "parse_error"},
                }))
            except Exception:
                pass
        except Exception as e:
            logger.exception("[%s] handler error", request_id)
            try:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "data": {"message": str(e), "code": "internal_error"},
                }))
            except Exception:
                pass
        finally:
            worker._active_connections -= 1

    return app
