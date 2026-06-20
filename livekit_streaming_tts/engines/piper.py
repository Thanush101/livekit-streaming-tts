"""Piper — fastest CPU TTS in this lineup.

Install: pip install livekit-streaming-tts[piper]

One model file per voice — instantiate one PiperEngine per voice. Or use a
PiperPool wrapper (not provided here) if you need many.
"""

from __future__ import annotations

import io
import wave
from typing import Optional

import numpy as np

from ..base import AudioChunk, EngineCapabilities, GenerationParams, TTSEngine


class PiperEngine(TTSEngine):
    name = "piper"
    native_sample_rate = 22050  # overridden in __init__ from the model
    capabilities = EngineCapabilities(
        voice_cloning=False,
        streaming_generation=False,
        multilingual=False,
        speed_control=True,
    )

    def __init__(self, model_path: str, config_path: Optional[str] = None) -> None:
        from piper import PiperVoice

        self._voice = PiperVoice.load(model_path, config_path=config_path)
        self.native_sample_rate = self._voice.config.sample_rate

    def list_voices(self) -> list[str]:
        return ["default"]

    def generate(self, params: GenerationParams) -> AudioChunk:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            self._voice.synthesize(
                params.text,
                wav,
                length_scale=(1.0 / params.speed) if params.speed else 1.0,
            )
        buf.seek(0)
        with wave.open(buf, "rb") as wav:
            frames = wav.readframes(wav.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        return AudioChunk(
            samples=samples, sample_rate=self.native_sample_rate, is_final=True
        )
