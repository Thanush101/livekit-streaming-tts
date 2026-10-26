"""Voice-state cache. The model is a fake, so this does not load weights."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from livekit_streaming_tts.base import GenerationParams
from livekit_streaming_tts.engines.pocket import PocketEngine
from livekit_streaming_tts.engines.pocket_voice import (
    VoiceStateCache,
    is_voice_tensor,
    voice_cache_filename,
)


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.generated: list[tuple] = []

    def get_state_for_audio_prompt(self, prompt, truncate: bool = False):
        self.calls.append(str(prompt))
        return {"prompt": str(prompt), "n": len(self.calls)}

    def generate_audio(self, state, text, copy_state=None):
        self.generated.append((state, text, copy_state))
        return np.zeros(4, dtype=np.float32)


class VoiceCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = self._tmp.name
        self.exports: list[str] = []

        def fake_export(state, dest: str) -> None:
            self.exports.append(dest)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            Path(dest).write_bytes(b"tensor")

        self._patch = patch(
            "livekit_streaming_tts.engines.pocket_voice._export",
            fake_export,
        )
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self._tmp.cleanup()

    def test_filename_is_stable_and_has_no_slash(self) -> None:
        voice = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
        name = voice_cache_filename(voice)
        self.assertEqual(name, voice_cache_filename(voice))
        self.assertTrue(name.endswith(".safetensors"))
        self.assertNotIn("/", name)
        self.assertTrue(is_voice_tensor("voice.safetensors"))
        self.assertFalse(is_voice_tensor(voice))

    def test_preset_is_encoded_once_and_reused(self) -> None:
        model = FakeModel()
        cache = VoiceStateCache(model, self.cache_dir, custom_weights=False)
        first = cache.get("alba")
        second = cache.get("alba")
        self.assertIs(first, second)
        self.assertEqual(model.calls, ["alba"])
        self.assertEqual(len(self.exports), 1)

    def test_restart_loads_the_exported_tensor(self) -> None:
        model = FakeModel()
        cache = VoiceStateCache(model, self.cache_dir, custom_weights=False)
        cache.get("alba")
        self.assertEqual(model.calls, ["alba"])

        restarted = FakeModel()
        again = VoiceStateCache(restarted, self.cache_dir, custom_weights=False)
        state = again.get("alba")
        again.get("alba")
        self.assertEqual(len(restarted.calls), 1)
        self.assertTrue(restarted.calls[0].endswith(".safetensors"))
        self.assertNotEqual(restarted.calls[0], "alba")
        self.assertIs(again.get("alba"), state)
        self.assertEqual(len(self.exports), 1)

    def test_legacy_safetensors_name_is_loaded(self) -> None:
        legacy = Path(self.cache_dir) / "alba.safetensors"
        legacy.write_bytes(b"tensor")
        model = FakeModel()
        cache = VoiceStateCache(model, self.cache_dir, custom_weights=False)
        cache.get("alba")
        cache.get("alba")
        self.assertEqual(model.calls, [str(legacy)])
        self.assertEqual(self.exports, [])

    def test_safetensors_path_is_loaded_not_encoded(self) -> None:
        path = Path(self.cache_dir) / "speaker.safetensors"
        path.write_bytes(b"tensor")
        model = FakeModel()
        cache = VoiceStateCache(model, self.cache_dir, custom_weights=True)
        first = cache.get(str(path))
        second = cache.get(str(path))
        self.assertIs(first, second)
        self.assertEqual(model.calls, [str(path)])
        self.assertEqual(self.exports, [])

    def test_pt_voice_is_rejected(self) -> None:
        model = FakeModel()
        cache = VoiceStateCache(model, self.cache_dir, custom_weights=False)
        with self.assertRaises(ValueError):
            cache.get("/voices/speaker.pt")
        self.assertEqual(model.calls, [])

    def test_wav_and_hf_url_are_encoded_once(self) -> None:
        model = FakeModel()
        cache = VoiceStateCache(model, self.cache_dir, custom_weights=True)
        wav = "/voices/speaker.wav"
        url = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
        wav_state = cache.get(wav)
        url_state = cache.get(url)
        self.assertIs(cache.get(wav), wav_state)
        self.assertIs(cache.get(url), url_state)
        self.assertEqual(model.calls, [wav, url])
        self.assertEqual(len(self.exports), 2)

    def test_replaced_wav_is_encoded_again(self) -> None:
        wav = Path(self.cache_dir) / "speaker.wav"
        wav.write_bytes(b"one")
        model = FakeModel()
        cache = VoiceStateCache(model, self.cache_dir, custom_weights=False)
        cache.get(str(wav))
        later = wav.stat().st_mtime + 10
        os.utime(wav, (later, later))
        cache.get(str(wav))
        self.assertEqual(model.calls, [str(wav), str(wav)])

    def test_community_preset_uses_the_wav_url(self) -> None:
        try:
            from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES
        except Exception:
            self.skipTest("pocket-tts is not installed")
        model = FakeModel()
        cache = VoiceStateCache(model, self.cache_dir, custom_weights=True)
        cache.get("alba")
        cache.get("alba")
        self.assertEqual(model.calls, [_ORIGINS_OF_PREDEFINED_VOICES["alba"]])
        self.assertEqual(len(self.exports), 1)

    def test_community_model_does_not_reuse_a_kyutai_preset_export(self) -> None:
        try:
            from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES
        except Exception:
            self.skipTest("pocket-tts is not installed")
        kyutai = FakeModel()
        VoiceStateCache(kyutai, self.cache_dir, custom_weights=False).get("alba")
        self.assertEqual(kyutai.calls, ["alba"])

        community = FakeModel()
        cache = VoiceStateCache(community, self.cache_dir, custom_weights=True)
        url = _ORIGINS_OF_PREDEFINED_VOICES["alba"]
        cache.get("alba")
        cache.get("alba")
        self.assertEqual(community.calls, [url])

        restarted = FakeModel()
        again = VoiceStateCache(restarted, self.cache_dir, custom_weights=True)
        again.get("alba")
        self.assertEqual(len(restarted.calls), 1)
        self.assertTrue(restarted.calls[0].endswith(".safetensors"))
        self.assertNotEqual(restarted.calls[0], "alba")
        self.assertNotEqual(restarted.calls[0], url)

    def test_voice_id_survives_restart_and_a_new_source_reencodes(self) -> None:
        model = FakeModel()
        cache = VoiceStateCache(model, self.cache_dir, custom_weights=False)
        cache.get("alice", source="/voices/a.wav")
        cache.get("alice", source="/voices/b.wav")
        self.assertEqual(model.calls, ["/voices/a.wav", "/voices/b.wav"])

        restarted = FakeModel()
        again = VoiceStateCache(restarted, self.cache_dir, custom_weights=False)
        again.get("alice")
        again.get("alice")
        self.assertEqual(len(restarted.calls), 1)
        self.assertTrue(restarted.calls[0].endswith(".safetensors"))

    def test_generate_reuses_the_state_and_asks_for_a_copy(self) -> None:
        model = FakeModel()
        engine = PocketEngine.__new__(PocketEngine)
        engine.native_sample_rate = 24000
        engine._model = model
        engine._voices = VoiceStateCache(model, self.cache_dir, custom_weights=False)

        engine.register_voice("alice", "/voices/alice.wav")
        first = engine.generate(GenerationParams(text="Hello.", voice="alice"))
        second = engine.generate(GenerationParams(text="World.", voice="alice"))
        self.assertEqual(model.calls, ["/voices/alice.wav"])
        self.assertIs(model.generated[0][0], model.generated[1][0])
        self.assertEqual([item[1] for item in model.generated], ["Hello.", "World."])
        self.assertTrue(all(item[2] is True for item in model.generated))
        self.assertEqual(first.samples.dtype, np.float32)
        self.assertEqual(second.sample_rate, 24000)

        engine.generate(GenerationParams(text="A", ref_audio="/voices/other.wav"))
        engine.generate(GenerationParams(text="B", ref_audio="/voices/other.wav"))
        self.assertEqual(
            model.calls,
            ["/voices/alice.wav", "/voices/other.wav"],
        )


if __name__ == "__main__":
    unittest.main()
