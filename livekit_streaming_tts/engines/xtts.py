"""Coqui XTTS v2 — multilingual zero-shot voice cloning.

Install: pip install livekit-streaming-tts[xtts]

Supports 16 languages and zero-shot cloning from a 6-second reference clip.
Streaming generation IS supported by the underlying model — we expose it
via generate_stream for lower TTFB.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import torch

from ..base import AudioChunk, EngineCapabilities, GenerationParams, TTSEngine


class XTTSEngine(TTSEngine):
    name = "xtts"
    native_sample_rate = 24000
    capabilities = EngineCapabilities(
        voice_cloning=True,
        streaming_generation=True,
        multilingual=True,
        speed_control=True,
    )

    def __init__(self, device: str = "cuda:0") -> None:
        from TTS.api import TTS as CoquiTTS

        self._tts = CoquiTTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
        ).to(device)
        # voice_id -> path to ref audio (XTTS recomputes embedding per call,
        # so caching paths is the right unit of indirection).
        self._voice_paths: dict[str, str] = {}

    def register_voice(
        self, voice_id: str, audio_path: str, ref_text: Optional[str] = None
    ) -> None:
        self._voice_paths[voice_id] = audio_path

    def delete_voice(self, voice_id: str) -> None:
        self._voice_paths.pop(voice_id, None)

    def list_voices(self) -> list[str]:
        return list(self._voice_paths.keys())

    def _resolve_speaker(self, params: GenerationParams) -> str:
        if params.voice and params.voice in self._voice_paths:
            return self._voice_paths[params.voice]
        if params.ref_audio:
            return params.ref_audio
        raise ValueError(
            "XTTS needs a registered `voice` or `ref_audio` — no default voice."
        )

    def generate(self, params: GenerationParams) -> AudioChunk:
        if params.seed is not None:
            torch.manual_seed(params.seed)

        speaker_wav = self._resolve_speaker(params)
        out = self._tts.tts(
            text=params.text,
            speaker_wav=speaker_wav,
            language=params.language or "en",
            speed=params.speed or 1.0,
        )
        return AudioChunk(
            samples=np.asarray(out, dtype=np.float32),
            sample_rate=self.native_sample_rate,
            is_final=True,
        )

    def generate_stream(self, params: GenerationParams) -> Iterator[AudioChunk]:
        # Coqui's streaming API on the underlying tts.synthesizer model.
        # Falls back to one-shot if streaming isn't available in this
        # version of TTS.
        if params.seed is not None:
            torch.manual_seed(params.seed)
        speaker_wav = self._resolve_speaker(params)

        synth = self._tts.synthesizer
        if not hasattr(synth, "tts") or not hasattr(synth, "tts_model"):
            yield from super().generate_stream(params)
            return

        try:
            stream = synth.tts_model.inference_stream(
                params.text,
                params.language or "en",
                gpt_cond_latent=None,
                speaker_embedding=None,
                speaker_wav=speaker_wav,
                speed=params.speed or 1.0,
            )
        except Exception:
            yield from super().generate_stream(params)
            return

        last: Optional[AudioChunk] = None
        for piece in stream:
            arr = piece.cpu().numpy() if hasattr(piece, "cpu") else np.asarray(piece)
            chunk = AudioChunk(
                samples=arr.astype(np.float32),
                sample_rate=self.native_sample_rate,
                is_final=False,
            )
            if last is not None:
                yield last
            last = chunk
        if last is not None:
            last.is_final = True
            yield last
