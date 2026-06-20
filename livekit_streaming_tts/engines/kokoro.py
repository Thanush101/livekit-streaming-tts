"""Kokoro — small, fast, high-quality English TTS (82M params).

Install: pip install livekit-streaming-tts[kokoro]

No voice cloning; ships with a fixed list of voices (af_*, am_*, bf_*, ...).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import AudioChunk, EngineCapabilities, GenerationParams, TTSEngine


class KokoroEngine(TTSEngine):
    name = "kokoro"
    native_sample_rate = 24000
    capabilities = EngineCapabilities(
        voice_cloning=False,
        streaming_generation=False,
        multilingual=True,  # supports en/ja/zh via different model variants
        speed_control=True,
    )

    def __init__(
        self,
        model_name: str = "hexgrad/Kokoro-82M",
        default_voice: str = "af_heart",
        device: Optional[str] = None,
        lang_code: str = "a",  # 'a' = American English, 'b' = British, 'j' = Japanese...
        prewarm_voices: Optional[list[str]] = None,
    ) -> None:
        from kokoro import KPipeline

        self._pipeline = KPipeline(
            lang_code=lang_code,
            repo_id=model_name,
            device=device,
        )
        self._default_voice = default_voice
        self._lang_code = lang_code

        # Pre-warm voice tensors at startup so the first user turn doesn't pay
        # the HuggingFace download (~3s for af_heart.pt). Override via env:
        # KOKORO_PREWARM=af_heart,am_michael tts-server ...
        import os as _os
        env_warm = _os.getenv("KOKORO_PREWARM", "")
        warm_list = (
            prewarm_voices
            if prewarm_voices is not None
            else (
                [v.strip() for v in env_warm.split(",") if v.strip()]
                or [default_voice]
            )
        )
        for v in warm_list:
            try:
                # KPipeline.load_voice fetches the .pt file from HF and caches
                # it in memory. After this the first generate() is hot.
                self._pipeline.load_voice(v)
            except Exception:
                # Pre-warm is best-effort. If a name isn't in this language's
                # catalog, the user can still pass it at generate time.
                pass

    def list_voices(self) -> list[str]:
        # Kokoro has a fixed catalog; the most common are exposed below.
        # Users can pass any HuggingFace voice id via params.voice.
        return [
            "af_heart", "af_bella", "af_nicole", "af_sky",
            "am_adam", "am_michael",
            "bf_emma", "bf_isabella",
            "bm_george", "bm_lewis",
        ]

    def generate(self, params: GenerationParams) -> AudioChunk:
        voice = params.voice or self._default_voice
        speed = params.speed or 1.0
        # Kokoro yields chunks per sentence; we collect to a single AudioChunk
        # since the server's chunking already split sentences upstream.
        audio_pieces: list[np.ndarray] = []
        for _, _, audio in self._pipeline(params.text, voice=voice, speed=speed):
            arr = audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio)
            audio_pieces.append(arr.astype(np.float32))
        if not audio_pieces:
            return AudioChunk(
                samples=np.zeros(0, dtype=np.float32),
                sample_rate=self.native_sample_rate,
                is_final=True,
            )
        merged = np.concatenate(audio_pieces)
        return AudioChunk(
            samples=merged, sample_rate=self.native_sample_rate, is_final=True
        )
