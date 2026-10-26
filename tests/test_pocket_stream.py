"""Device parsing and websocket PCM streaming. No weights and no GPU."""

from __future__ import annotations

import asyncio
import os
import threading
import unittest

import numpy as np

from livekit_streaming_tts.audio import to_int16_pcm
from livekit_streaming_tts.base import AudioChunk, EngineCapabilities, GenerationParams, TTSEngine
from livekit_streaming_tts.engines.pocket import (
    PocketEngine,
    _to_float32,
    iter_pocket_pcm,
    resolve_pocket_device,
)
from livekit_streaming_tts.engines.pocket_voice import VoiceStateCache
from livekit_streaming_tts.server import GPUWorker, _Job, websocket_pcm_messages


class DeviceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = os.environ.get("POCKET_TTS_DEVICE")
        os.environ.pop("POCKET_TTS_DEVICE", None)

    def tearDown(self) -> None:
        if self._saved is None:
            os.environ.pop("POCKET_TTS_DEVICE", None)
        else:
            os.environ["POCKET_TTS_DEVICE"] = self._saved

    def test_cpu_is_the_default(self) -> None:
        self.assertEqual(resolve_pocket_device(None), "cpu")
        self.assertEqual(resolve_pocket_device(""), "cpu")
        self.assertEqual(resolve_pocket_device("cpu"), "cpu")
        self.assertEqual(resolve_pocket_device(" CPU "), "cpu")
        os.environ["POCKET_TTS_DEVICE"] = ""
        self.assertEqual(resolve_pocket_device(None), "cpu")

    def test_cuda_and_gpu_alias(self) -> None:
        self.assertEqual(resolve_pocket_device("cuda"), "cuda")
        self.assertEqual(resolve_pocket_device("CUDA"), "cuda")
        self.assertEqual(resolve_pocket_device("gpu"), "cuda")
        self.assertEqual(resolve_pocket_device(" GPU "), "cuda")
        os.environ["POCKET_TTS_DEVICE"] = "gpu"
        self.assertEqual(resolve_pocket_device(None), "cuda")
        # An explicit cpu wins over a cuda env value.
        self.assertEqual(resolve_pocket_device("cpu"), "cpu")

    def test_nonsense_is_rejected(self) -> None:
        for value in ("mps", "auto", "cuda:0", "rocm", "yes"):
            with self.assertRaises(ValueError):
                resolve_pocket_device(value)
        os.environ["POCKET_TTS_DEVICE"] = "auto"
        with self.assertRaises(ValueError):
            resolve_pocket_device(None)


class _Scripted:
    """Stand-in for TTSModel. generate_audio is the concatenation of the stream."""

    def __init__(self) -> None:
        self.chunks = [
            np.array([0.25, -0.5], dtype=np.float32),
            np.array([0.125, 1.0], dtype=np.float32),
        ]
        self.audio_copy: list[bool] = []
        self.stream_copy: list[bool] = []

    def get_state_for_audio_prompt(self, prompt, truncate: bool = False):
        return {"prompt": str(prompt)}

    def generate_audio(self, state, text, copy_state=None):
        self.audio_copy.append(copy_state)
        return np.concatenate(list(self.generate_audio_stream(state, text, copy_state=copy_state)))

    def generate_audio_stream(self, state, text, copy_state=True, stop=None):
        self.stream_copy.append(copy_state)
        for chunk in self.chunks:
            if stop is not None and stop.is_set():
                return
            yield chunk


class StreamSampleTests(unittest.TestCase):
    def test_streamed_chunks_match_generate_audio_and_keep_the_start(self) -> None:
        model = _Scripted()
        streamed = list(iter_pocket_pcm(model, {"s": 1}, ["Hello."]))
        self.assertEqual(len(streamed), 2)
        self.assertEqual(float(streamed[0][0]), 0.25)
        joined = np.concatenate(streamed)
        # A second pass. generate_audio itself walks the stream and concatenates.
        full = _to_float32(model.generate_audio({"s": 1}, "Hello.", copy_state=True))
        self.assertTrue(np.array_equal(joined, full))
        self.assertEqual(model.stream_copy, [True, True])
        self.assertEqual(model.audio_copy, [True])

    def test_engine_stream_matches_one_shot_generate(self) -> None:
        model = _Scripted()
        engine = PocketEngine.__new__(PocketEngine)
        engine.native_sample_rate = 24000
        engine._model = model
        engine._voices = VoiceStateCache(model, "", custom_weights=False)

        params = GenerationParams(text="Hello.", voice="alba")
        one_shot = engine.generate(params)
        frames = list(engine.generate_stream(params))
        self.assertGreaterEqual(len(frames), 2)
        self.assertEqual(float(frames[0].samples[0]), 0.25)
        self.assertTrue(np.array_equal(np.concatenate([f.samples for f in frames]), one_shot.samples))
        self.assertTrue(all(copy is True for copy in model.audio_copy))
        self.assertTrue(all(copy is True for copy in model.stream_copy))

    def test_websocket_helper_emits_both_frames_before_final(self) -> None:
        first = np.array([0.25, -0.5], dtype=np.float32)
        second = np.array([0.125, 1.0], dtype=np.float32)
        messages = websocket_pcm_messages([first, second], sample_rate=24000, request_id="req")
        self.assertEqual(len(messages), 3)
        self.assertIsInstance(messages[0], bytes)
        self.assertIsInstance(messages[1], bytes)
        self.assertTrue(messages[0])
        self.assertTrue(messages[1])
        self.assertEqual(messages[2]["type"], "event")
        self.assertEqual(messages[2]["data"]["event_type"], "final")
        self.assertEqual(messages[2]["data"]["request_id"], "req")
        self.assertEqual(b"".join(messages[:2]), to_int16_pcm(np.concatenate([first, second])))
        # The first sample is the start of the first frame, not a later one.
        self.assertEqual(messages[0][:2], to_int16_pcm(first[:1]))


class _Blocked(TTSEngine):
    name = "blocked"
    native_sample_rate = 24000
    capabilities = EngineCapabilities(streaming_generation=True)

    def __init__(self) -> None:
        self.release = threading.Event()
        self.chunks = [
            np.array([0.25, -0.5], dtype=np.float32),
            np.array([0.125, 1.0], dtype=np.float32),
        ]
        self.yielded = 0

    def generate(self, params: GenerationParams) -> AudioChunk:
        return AudioChunk(
            samples=np.concatenate(self.chunks),
            sample_rate=self.native_sample_rate,
            is_final=True,
        )

    def generate_stream(self, params: GenerationParams, stop: threading.Event | None = None):
        self.yielded += 1
        yield AudioChunk(samples=self.chunks[0], sample_rate=24000, is_final=False)
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release the second chunk")
        if stop is not None and stop.is_set():
            return
        self.yielded += 1
        yield AudioChunk(samples=self.chunks[1], sample_rate=24000, is_final=False)


class WorkerStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_chunk_is_available_before_the_utterance_finishes(self) -> None:
        engine = _Blocked()
        worker = GPUWorker(engine)
        await worker.start()
        try:
            job = _Job(
                params=GenerationParams(text="Hello."),
                cancel_event=asyncio.Event(),
                stop=threading.Event(),
            )
            stream = worker.iter_stream(job)
            first = await asyncio.wait_for(stream.__anext__(), timeout=2)
            self.assertEqual(engine.yielded, 1)
            self.assertFalse(engine.release.is_set())
            self.assertEqual(first.samples.tolist(), [0.25, -0.5])

            engine.release.set()
            second = await asyncio.wait_for(stream.__anext__(), timeout=2)
            self.assertEqual(second.samples.tolist(), [0.125, 1.0])
            with self.assertRaises(StopAsyncIteration):
                await asyncio.wait_for(stream.__anext__(), timeout=2)

            joined = np.concatenate([first.samples, second.samples])
            self.assertTrue(np.array_equal(joined, np.concatenate(engine.chunks)))
            messages = websocket_pcm_messages(
                [first.samples, second.samples],
                sample_rate=24000,
                request_id="turn",
            )
            self.assertEqual(messages[2]["data"]["event_type"], "final")
            self.assertLess(
                messages.index(messages[0]),
                messages.index(messages[2]),
            )
        finally:
            engine.release.set()
            await worker.stop()


if __name__ == "__main__":
    unittest.main()
