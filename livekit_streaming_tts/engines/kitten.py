"""KittenTTS — tiny CPU TTS with a fixed voice list.

Install: pip install livekit-streaming-tts[kitten]

No voice cloning; `params.voice` selects from the model's preset list.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import AudioChunk, EngineCapabilities, GenerationParams, TTSEngine


class KittenEngine(TTSEngine):
    name = "kitten"
    native_sample_rate = 24000
    capabilities = EngineCapabilities(
        voice_cloning=False,
        streaming_generation=False,
        multilingual=False,
        speed_control=False,
    )

    def __init__(
        self,
        model_name: str = "KittenML/kitten-tts-nano-0.1",
        default_voice: str = "expr-voice-2-f",
    ) -> None:
        from kittentts import KittenTTS

        self._model = KittenTTS(model_name)
        if default_voice not in self._model.available_voices:
            raise ValueError(
                f"voice {default_voice!r} not in {self._model.available_voices}"
            )
        self._default_voice = default_voice

    def list_voices(self) -> list[str]:
        return list(self._model.available_voices)

    def generate(self, params: GenerationParams) -> AudioChunk:
        voice = params.voice or self._default_voice
        if voice not in self._model.available_voices:
            voice = self._default_voice

        # Tail padding prevents word cut-offs on very short inputs.
        padded = params.text.strip() + " ... "
        audio = self._model.generate(padded, voice=voice)
        return AudioChunk(
            samples=np.asarray(audio, dtype=np.float32),
            sample_rate=self.native_sample_rate,
            is_final=True,
        )
