"""Suno Bark — expressive, slow, voice-prompt-based.

Install: pip install livekit-streaming-tts[bark]

`params.voice` is a Bark voice prompt name (e.g. "v2/en_speaker_6").
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import AudioChunk, EngineCapabilities, GenerationParams, TTSEngine


class BarkEngine(TTSEngine):
    name = "bark"
    native_sample_rate = 24000
    capabilities = EngineCapabilities(
        voice_cloning=False,
        streaming_generation=False,
        multilingual=True,
        speed_control=False,
    )

    def __init__(self, default_voice: str = "v2/en_speaker_6") -> None:
        from bark import SAMPLE_RATE, generate_audio, preload_models

        preload_models()
        self._generate_audio = generate_audio
        self.native_sample_rate = SAMPLE_RATE
        self._default_voice = default_voice

    def list_voices(self) -> list[str]:
        # Subset of Bark's preset library. See suno-ai/bark for the full list.
        return [f"v2/en_speaker_{i}" for i in range(10)]

    def generate(self, params: GenerationParams) -> AudioChunk:
        voice = params.voice or self._default_voice
        out = self._generate_audio(params.text, history_prompt=voice)
        return AudioChunk(
            samples=np.asarray(out, dtype=np.float32),
            sample_rate=self.native_sample_rate,
            is_final=True,
        )
