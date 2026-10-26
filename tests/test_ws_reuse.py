"""Several utterances on one websocket. The engine is a fake. No weights and no port 8001."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import unittest

import numpy as np

from livekit_streaming_tts.base import AudioChunk, EngineCapabilities, GenerationParams, TTSEngine
from livekit_streaming_tts.server import create_app

_KEY = "tts-key-9f3a"


class _FastEngine(TTSEngine):
    name = "fake"
    native_sample_rate = 24000
    capabilities = EngineCapabilities()

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, params: GenerationParams) -> AudioChunk:
        self.calls += 1
        return AudioChunk(
            samples=np.full(8, 0.2, dtype=np.float32),
            sample_rate=self.native_sample_rate,
            is_final=True,
        )

    def generate_stream(self, params: GenerationParams, stop=None):
        self.calls += 1
        yield AudioChunk(
            samples=np.full(8, 0.2, dtype=np.float32),
            sample_rate=self.native_sample_rate,
            is_final=False,
        )
        if stop is None or not stop.is_set():
            yield AudioChunk(
                samples=np.full(8, 0.4, dtype=np.float32),
                sample_rate=self.native_sample_rate,
                is_final=True,
            )

    def list_voices(self) -> list[str]:
        return ["alba"]


class _GateEngine(TTSEngine):
    """Blocks inside generate_stream until cancel sets ``stop``. The second call speaks."""

    name = "fake-gate"
    native_sample_rate = 24000
    capabilities = EngineCapabilities()

    def __init__(self) -> None:
        self.calls = 0
        self.started = threading.Event()
        self.saw_stop = threading.Event()

    def generate(self, params: GenerationParams) -> AudioChunk:
        self.calls += 1
        return AudioChunk(
            samples=np.full(8, 0.2, dtype=np.float32),
            sample_rate=self.native_sample_rate,
            is_final=True,
        )

    def generate_stream(self, params: GenerationParams, stop=None):
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            yield AudioChunk(
                samples=np.full(4, 0.1, dtype=np.float32),
                sample_rate=self.native_sample_rate,
                is_final=False,
            )
            deadline = time.time() + 3
            while not (stop is not None and stop.is_set()):
                if time.time() > deadline:
                    return
                time.sleep(0.01)
            self.saw_stop.set()
            return
            yield
        yield AudioChunk(
            samples=np.full(8, 0.25, dtype=np.float32),
            sample_rate=self.native_sample_rate,
            is_final=False,
        )
        yield AudioChunk(
            samples=np.full(8, 0.5, dtype=np.float32),
            sample_rate=self.native_sample_rate,
            is_final=True,
        )

    def list_voices(self) -> list[str]:
        return ["alba"]


class _Running:
    def __init__(self, app) -> None:
        self.app = app
        self._in: asyncio.Queue = asyncio.Queue()
        self._out: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> "_Running":
        scope = {
            "type": "lifespan",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "state": {},
        }

        async def receive():
            return await self._in.get()

        async def send(message):
            await self._out.put(message)

        self._task = asyncio.create_task(self.app(scope, receive, send))
        await self._in.put({"type": "lifespan.startup"})
        message = await self._out.get()
        if message["type"] != "lifespan.startup.complete":
            await self._task
            raise RuntimeError(message.get("message") or message["type"])
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._in.put({"type": "lifespan.shutdown"})
        message = await self._out.get()
        if self._task is not None:
            await self._task
        if message["type"] != "lifespan.shutdown.complete":
            raise RuntimeError(message.get("message") or message["type"])


class _Socket:
    def __init__(self, app, headers: dict[str, str] | None = None) -> None:
        self.app = app
        self.headers = headers or {}
        self._in: asyncio.Queue = asyncio.Queue()
        self._out: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self.first: dict | None = None

    async def __aenter__(self) -> "_Socket":
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "scheme": "ws",
            "path": "/tts/ws",
            "raw_path": b"/tts/ws",
            "root_path": "",
            "query_string": b"",
            "headers": [
                (name.lower().encode("latin-1"), value.encode("latin-1"))
                for name, value in self.headers.items()
            ],
            "client": ("127.0.0.1", 1234),
            "server": ("test", 80),
            "subprotocols": [],
            "state": {},
            "extensions": {"websocket.http.response": {}},
            "app": self.app,
        }

        async def receive():
            return await self._in.get()

        async def send(message):
            await self._out.put(message)

        self._task = asyncio.create_task(self.app(scope, receive, send))
        await self._in.put({"type": "websocket.connect"})
        self.first = await asyncio.wait_for(self._out.get(), 5)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._task is None:
            return
        if not self._task.done():
            await self._in.put({"type": "websocket.disconnect", "code": 1000})
            await asyncio.wait_for(self._task, 5)
        else:
            await self._task

    async def send(self, payload: dict) -> None:
        await self._in.put({
            "type": "websocket.receive",
            "text": json.dumps(payload),
        })

    async def next_message(self, timeout: float = 5) -> dict:
        return await asyncio.wait_for(self._out.get(), timeout)


def _pcm_config(*, pauses: bool = False) -> dict:
    return {
        "type": "config",
        "data": {
            "format": "pcm",
            "binary": True,
            "sample_rate": 24000,
            "normalize": True,
            "pauses": pauses,
            "voice": "alba",
        },
    }


async def _utterance(sock: _Socket, text: str, *, pauses: bool = False) -> list[bytes]:
    await sock.send(_pcm_config(pauses=pauses))
    await sock.send({"type": "text", "data": {"text": text}})
    await sock.send({"type": "flush"})
    audio: list[bytes] = []
    while True:
        message = await sock.next_message()
        if message.get("type") != "websocket.send":
            raise AssertionError(message["type"])
        if "bytes" in message and message["bytes"]:
            audio.append(message["bytes"])
            continue
        body = json.loads(message.get("text") or "{}")
        event = (body.get("data") or {}).get("event_type")
        if event == "final":
            return audio
        if body.get("type") == "error":
            raise AssertionError(body)


class ReuseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._saved = os.environ.get("TTS_API_KEY")
        os.environ.pop("TTS_API_KEY", None)

    async def asyncTearDown(self) -> None:
        if self._saved is None:
            os.environ.pop("TTS_API_KEY", None)
        else:
            os.environ["TTS_API_KEY"] = self._saved

    async def test_two_turns_and_a_pause_on_one_socket(self) -> None:
        engine = _FastEngine()
        app = create_app(engine=engine)
        async with _Running(app):
            async with _Socket(app) as sock:
                self.assertEqual(sock.first["type"], "websocket.accept")
                first = await _utterance(sock, "Hello.")
                second = await _utterance(sock, "Again.")
                paused = await _utterance(
                    sock, "Hello. [pause:0.5] World.", pauses=True
                )
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertTrue(all(frame and any(frame) for frame in first + second))
        self.assertGreaterEqual(engine.calls, 2)
        self.assertTrue(any(frame and not any(frame) for frame in paused))
        self.assertTrue(any(frame and any(frame) for frame in paused))

    async def test_cancel_during_audio_then_the_next_turn_speaks(self) -> None:
        engine = _GateEngine()
        app = create_app(engine=engine)
        async with _Running(app):
            async with _Socket(app) as sock:
                self.assertEqual(sock.first["type"], "websocket.accept")
                await sock.send(_pcm_config())
                await sock.send({"type": "text", "data": {"text": "Hello."}})
                await sock.send({"type": "flush"})
                await asyncio.wait_for(asyncio.to_thread(engine.started.wait, 3), 4)
                await sock.send({"type": "cancel"})
                await asyncio.wait_for(asyncio.to_thread(engine.saw_stop.wait, 3), 4)
                nxt = await _utterance(sock, "Next sentence.")
        self.assertTrue(engine.saw_stop.is_set())
        self.assertGreaterEqual(engine.calls, 2)
        self.assertGreaterEqual(len(nxt), 2)
        self.assertTrue(all(any(frame) for frame in nxt))

    async def test_idle_cancel_does_not_silence_the_next_turn(self) -> None:
        engine = _FastEngine()
        app = create_app(engine=engine)
        async with _Running(app):
            async with _Socket(app) as sock:
                first = await _utterance(sock, "Hello.")
                await sock.send({"type": "cancel"})
                await asyncio.sleep(0.05)
                second = await _utterance(sock, "Again.")
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertEqual(engine.calls, 2)

    async def test_api_key_on_the_reused_socket_and_on_reconnect(self) -> None:
        os.environ["TTS_API_KEY"] = _KEY
        engine = _FastEngine()
        app = create_app(engine=engine)
        async with _Running(app):
            async with _Socket(app) as missing:
                self.assertEqual(missing.first["type"], "websocket.http.response.start")
                self.assertEqual(missing.first["status"], 401)
            async with _Socket(app, {"authorization": f"Bearer {_KEY}"}) as sock:
                self.assertEqual(sock.first["type"], "websocket.accept")
                first = await _utterance(sock, "Hello.")
                second = await _utterance(sock, "Again.")
                self.assertTrue(first and second)
            before = engine.calls
            async with _Socket(app, {"x-api-key": "wrong-key-9f3a"}) as denied:
                self.assertEqual(denied.first["status"], 401)
            self.assertEqual(engine.calls, before)
            async with _Socket(app, {"authorization": f"Bearer {_KEY}"}) as again:
                self.assertEqual(again.first["type"], "websocket.accept")
                third = await _utterance(again, "After reconnect.")
            self.assertTrue(third)
            self.assertGreater(engine.calls, before)


if __name__ == "__main__":
    unittest.main()
