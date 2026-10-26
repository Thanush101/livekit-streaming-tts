"""API key checks. No model download and no listening socket.

The route tests speak ASGI to the FastAPI app. Starlette's TestClient needs
httpx, which this environment does not have.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import unittest
from unittest.mock import patch

import numpy as np

from livekit_streaming_tts.base import AudioChunk, EngineCapabilities, GenerationParams, TTSEngine
from livekit_streaming_tts.server import create_app, credentials_accepted

_KEY = "tts-key-9f3a"
_WRONG = "wrong-key-9f3a"


class _FakeEngine(TTSEngine):
    name = "fake"
    native_sample_rate = 24000
    capabilities = EngineCapabilities()

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, params: GenerationParams) -> AudioChunk:
        self.calls += 1
        return AudioChunk(
            samples=np.full(16, 0.1, dtype=np.float32),
            sample_rate=self.native_sample_rate,
            is_final=True,
        )

    def list_voices(self) -> list[str]:
        return ["alba"]


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())
        if record.args:
            self.lines.append(repr(record.args))


def _scope_headers(headers: dict[str, str]) -> list[tuple[bytes, bytes]]:
    return [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in headers.items()
    ]


class CredentialTests(unittest.TestCase):
    def test_empty_config_does_not_require_a_key(self) -> None:
        for expected in (None, ""):
            self.assertTrue(credentials_accepted(expected))
            self.assertTrue(credentials_accepted(expected, authorization="Bearer nope"))
            with patch("livekit_streaming_tts.server.hmac.compare_digest") as compare:
                self.assertTrue(credentials_accepted(expected, x_api_key=_WRONG))
                compare.assert_not_called()

    def test_correct_key_passes_and_wrong_key_fails(self) -> None:
        self.assertTrue(
            credentials_accepted(_KEY, authorization=f"Bearer {_KEY}")
        )
        self.assertTrue(credentials_accepted(_KEY, authorization=f"bearer {_KEY}"))
        self.assertTrue(credentials_accepted(_KEY, x_api_key=_KEY))
        self.assertFalse(credentials_accepted(_KEY))
        self.assertFalse(credentials_accepted(_KEY, authorization=f"Bearer {_WRONG}"))
        self.assertFalse(credentials_accepted(_KEY, x_api_key=_WRONG))
        self.assertFalse(credentials_accepted(_KEY, authorization=f"Basic {_KEY}"))
        self.assertFalse(credentials_accepted(_KEY, authorization="Bearer"))
        self.assertTrue(
            credentials_accepted(
                _KEY,
                authorization=f"Bearer {_WRONG}",
                x_api_key=_KEY,
            )
        )

    def test_match_comes_from_compare_digest(self) -> None:
        with patch(
            "livekit_streaming_tts.server.hmac.compare_digest", return_value=False
        ) as compare:
            self.assertFalse(
                credentials_accepted(_KEY, authorization=f"Bearer {_KEY}")
            )
            compare.assert_called_with(_KEY, _KEY)

        calls: list[tuple[str, str]] = []

        def _record(left: str, right: str) -> bool:
            calls.append((left, right))
            return False

        with patch(
            "livekit_streaming_tts.server.hmac.compare_digest", side_effect=_record
        ):
            self.assertFalse(
                credentials_accepted(
                    _KEY,
                    authorization=f"Bearer {_WRONG}",
                    x_api_key="other-header",
                )
            )
        self.assertEqual(calls, [("other-header", _KEY), (_WRONG, _KEY)])


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


class RouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = _FakeEngine()
        self.app = create_app(engine=self.engine)
        self._running = _Running(self.app)
        await self._running.__aenter__()
        self._saved = os.environ.get("TTS_API_KEY")
        os.environ.pop("TTS_API_KEY", None)
        self._logs = _Capture()
        self._logger = logging.getLogger("livekit_streaming_tts.server")
        self._logger.addHandler(self._logs)

    async def asyncTearDown(self) -> None:
        self._logger.removeHandler(self._logs)
        if self._saved is None:
            os.environ.pop("TTS_API_KEY", None)
        else:
            os.environ["TTS_API_KEY"] = self._saved
        await self._running.__aexit__(None, None, None)

    def _logged(self) -> str:
        return "\n".join(self._logs.lines)

    async def _http(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ) -> tuple[int, dict[str, str], bytes]:
        sent: list[dict] = []
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "root_path": "",
            "query_string": b"",
            "headers": _scope_headers(headers or {}),
            "client": ("127.0.0.1", 1234),
            "server": ("test", 80),
            "state": {},
            "app": self.app,
        }
        await asyncio.wait_for(self.app(scope, receive, send), timeout=5)
        status = next(msg["status"] for msg in sent if msg["type"] == "http.response.start")
        raw_headers = next(
            msg["headers"] for msg in sent if msg["type"] == "http.response.start"
        )
        payload = b"".join(
            msg.get("body", b"") for msg in sent if msg["type"] == "http.response.body"
        )
        decoded = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in raw_headers
        }
        return status, decoded, payload

    async def _ws(self, headers: dict[str, str] | None = None) -> list[dict]:
        sent: list[dict] = []
        incoming = [
            {"type": "websocket.connect"},
            {
                "type": "websocket.receive",
                "text": json.dumps({"type": "config", "data": {"api_key": _KEY}}),
            },
            {"type": "websocket.disconnect", "code": 1000},
        ]

        async def receive():
            if incoming:
                return incoming.pop(0)
            return {"type": "websocket.disconnect", "code": 1000}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "scheme": "ws",
            "path": "/tts/ws",
            "raw_path": b"/tts/ws",
            "root_path": "",
            "query_string": b"",
            "headers": _scope_headers(headers or {}),
            "client": ("127.0.0.1", 1234),
            "server": ("test", 80),
            "subprotocols": [],
            "state": {},
            "extensions": {"websocket.http.response": {}},
            "app": self.app,
        }
        await asyncio.wait_for(self.app(scope, receive, send), timeout=5)
        return sent

    async def test_open_server_synthesizes_without_a_header(self) -> None:
        for label, setup in (
            ("unset", lambda: os.environ.pop("TTS_API_KEY", None)),
            ("empty", lambda: os.environ.__setitem__("TTS_API_KEY", "")),
        ):
            with self.subTest(env=label):
                setup()
                before = self.engine.calls
                health, _, _ = await self._http("GET", "/health")
                self.assertEqual(health, 200)
                voices, _, voice_body = await self._http("GET", "/v1/voices")
                self.assertEqual(voices, 200)
                self.assertEqual(json.loads(voice_body)["voices"], ["alba"])
                tts, _, tts_body = await self._http(
                    "POST",
                    "/v1/tts",
                    headers={"content-type": "application/json"},
                    body=json.dumps({"text": "hello", "format": "wav"}).encode(),
                )
                self.assertEqual(tts, 200)
                self.assertTrue(json.loads(tts_body)["audio"])
                self.assertEqual(self.engine.calls, before + 1)
                frames = await self._ws()
                self.assertEqual(frames[0]["type"], "websocket.accept")

    async def test_configured_key_rejects_missing_and_wrong_keys(self) -> None:
        os.environ["TTS_API_KEY"] = _KEY
        before = self.engine.calls

        health, _, _ = await self._http("GET", "/health")
        self.assertEqual(health, 200)

        for headers in (
            {},
            {"authorization": f"Bearer {_WRONG}"},
            {"x-api-key": _WRONG},
            {"authorization": f"Basic {_KEY}"},
        ):
            status, response_headers, body = await self._http("GET", "/v1/voices", headers=headers)
            self.assertEqual(status, 401)
            self.assertIn("application/json", response_headers.get("content-type", ""))
            self.assertEqual(json.loads(body), {"detail": "Unauthorized"})
            self.assertNotIn(_KEY.encode(), body)
            self.assertNotIn(_WRONG.encode(), body)

        tts_body = json.dumps(
            {"text": "hello", "format": "wav", "extra": {"api_key": _KEY}}
        ).encode()
        status, _, body = await self._http(
            "POST",
            "/v1/tts",
            headers={"content-type": "application/json"},
            body=tts_body,
        )
        self.assertEqual(status, 401)
        self.assertNotIn(b"audio", body)
        self.assertEqual(json.loads(body)["detail"], "Unauthorized")
        self.assertEqual(self.engine.calls, before)

        deleted, _, _ = await self._http("DELETE", "/v1/voices/zz-auth-missing")
        self.assertEqual(deleted, 401)

        frames = await self._ws()
        self.assertEqual(frames[0]["type"], "websocket.http.response.start")
        self.assertEqual(frames[0]["status"], 401)
        self.assertNotIn("websocket.accept", [msg["type"] for msg in frames])
        raw = b"".join(
            msg.get("body", b"")
            for msg in frames
            if msg["type"] == "websocket.http.response.body"
        )
        self.assertEqual(json.loads(raw), {"detail": "Unauthorized"})
        self.assertNotIn(_KEY.encode(), raw)
        self.assertNotIn(_KEY, self._logged())
        self.assertNotIn(_WRONG, self._logged())
        self.assertNotIn(f"Bearer {_KEY}", self._logged())

    async def test_configured_key_allows_bearer_and_x_api_key(self) -> None:
        os.environ["TTS_API_KEY"] = _KEY
        for headers in (
            {"authorization": f"Bearer {_KEY}"},
            {"x-api-key": _KEY},
        ):
            status, _, body = await self._http("GET", "/v1/voices", headers=headers)
            self.assertEqual(status, 200, body)
            self.assertEqual(json.loads(body)["voices"], ["alba"])

            before = self.engine.calls
            tts_headers = {"content-type": "application/json", **headers}
            status, _, body = await self._http(
                "POST",
                "/v1/tts",
                headers=tts_headers,
                body=json.dumps({"text": "hello", "format": "wav"}).encode(),
            )
            self.assertEqual(status, 200, body)
            self.assertTrue(json.loads(body)["audio"])
            self.assertEqual(self.engine.calls, before + 1)

            frames = await self._ws(headers)
            self.assertEqual(frames[0]["type"], "websocket.accept")


if __name__ == "__main__":
    unittest.main()
