"""Pocket TTS — Kyutai Labs small, fast TTS with voice cloning via kvcache.

Install: pip install livekit-streaming-tts[pocket]
Repo:    https://github.com/kyutai-labs/pocket-tts
Voices:  https://huggingface.co/kyutai/pocket-tts (preset names like "alba")
         and any local .wav, hf:// URL, or exported .safetensors kvcache.

Why register_voice is fast here:
    Pocket TTS lets you pre-compute a "model state" (kvcache) for a voice
    and reload it from a .safetensors file. We export on first use and
    re-load on subsequent registrations — much cheaper than recomputing
    the prompt from raw audio every time.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ..base import AudioChunk, EngineCapabilities, GenerationParams, TTSEngine


class PocketEngine(TTSEngine):
    name = "pocket"
    native_sample_rate = 24000  # overridden in __init__ from the model
    capabilities = EngineCapabilities(
        voice_cloning=True,
        streaming_generation=False,
        multilingual=False,
        speed_control=False,
    )

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        from pocket_tts import TTSModel

        self._model = TTSModel.load_model()
        # The model exposes its native rate after load.
        self.native_sample_rate = int(self._model.sample_rate)
        # voice_id -> model_state (in-memory). We also persist exported
        # kvcaches to `cache_dir` so a server restart re-loads instantly.
        self._voice_cache: dict[str, object] = {}
        self._cache_dir = cache_dir or os.getenv(
            "POCKET_TTS_CACHE_DIR", "/var/lib/livekit-streaming-tts/pocket-cache"
        )
        os.makedirs(self._cache_dir, exist_ok=True)

    def _cache_path(self, voice_id: str) -> str:
        return os.path.join(self._cache_dir, f"{voice_id}.safetensors")

    def register_voice(
        self, voice_id: str, audio_path: str, ref_text: Optional[str] = None
    ) -> None:
        from pocket_tts import export_model_state

        # If we already exported this voice, reload from disk (fast path).
        cache_path = self._cache_path(voice_id)
        if os.path.exists(cache_path):
            self._voice_cache[voice_id] = self._model.get_state_for_audio_prompt(cache_path)
            return

        # First-time: compute from raw audio (or accept built-in preset name
        # like "alba", or hf://... URL — get_state_for_audio_prompt handles
        # all three).
        state = self._model.get_state_for_audio_prompt(audio_path)
        self._voice_cache[voice_id] = state

        # Persist for fast reload after restart.
        try:
            export_model_state(state, cache_path)
        except Exception:
            # Persistence is best-effort; in-memory state still works.
            pass

    def delete_voice(self, voice_id: str) -> None:
        self._voice_cache.pop(voice_id, None)
        cache_path = self._cache_path(voice_id)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except OSError:
                pass

    def list_voices(self) -> list[str]:
        return list(self._voice_cache.keys())

    def generate(self, params: GenerationParams) -> AudioChunk:
        # Pocket TTS REQUIRES a voice state. If no voice is cached, fall
        # through to a built-in preset (e.g. "alba") so the engine still
        # works without explicit registration.
        if params.voice and params.voice in self._voice_cache:
            state = self._voice_cache[params.voice]
        elif params.ref_audio:
            # Ad-hoc cloning from a path. Slower than cached voice.
            state = self._model.get_state_for_audio_prompt(params.ref_audio)
        else:
            # Fall back to a default preset. "alba" is shipped with the model.
            preset = params.voice or "alba"
            state = self._voice_cache.setdefault(
                preset, self._model.get_state_for_audio_prompt(preset)
            )

        audio = self._model.generate_audio(state, params.text)
        # Pocket returns a 1-D torch tensor of PCM samples.
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        else:
            audio = np.asarray(audio)
        return AudioChunk(
            samples=audio.astype(np.float32),
            sample_rate=self.native_sample_rate,
            is_final=True,
        )
