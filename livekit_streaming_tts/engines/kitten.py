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
        model_path: Optional[str] = None,
        voices_path: Optional[str] = None,
        default_voice: str = "expr-voice-2-f",
    ) -> None:
        from kittentts import KittenTTS

        # KittenTTS auto-downloads the ONNX model and voices.npz from
        # HuggingFace (KittenML/kitten-tts-nano-0.1) when no paths are
        # supplied. Passing a HF repo string here is wrong; the upstream
        # library tries to load it as a local file and fails with
        # NoSuchFile. Leave model_path/voices_path as None for the default,
        # or pass real local paths if you've predownloaded them.
        self._model = KittenTTS(model_path=model_path, voices_path=voices_path)
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

        text = params.text.strip()
        # Ensure terminal punctuation so the model gets a prosody anchor.
        # WITHOUT this it sometimes truncates the last word; the upstream
        # blog suggested `" ... "` as padding, but those literal dots get
        # phonemized as a mumble/breath sound at the end of the audio.
        # A single period gives the same anchor without the artifact.
        if text and text[-1] not in ".!?,;:":
            text = text + "."

        audio = self._model.generate(text, voice=voice)
        samples = np.asarray(audio, dtype=np.float32)

        # Trim silence/noise at start and end. KittenTTS emits a short
        # transient at chunk boundaries (clicks / breath sounds) which
        # gets amplified when many short sentences play back-to-back.
        samples = _trim_silence(samples, self.native_sample_rate)
        # Apply a 5ms fade in/out so adjacent chunks splice cleanly.
        samples = _apply_fade(samples, self.native_sample_rate, fade_ms=5)

        return AudioChunk(
            samples=samples,
            sample_rate=self.native_sample_rate,
            is_final=True,
        )


def _trim_silence(
    samples: np.ndarray, sample_rate: int, threshold_db: float = -45.0
) -> np.ndarray:
    """Trim leading and trailing samples below `threshold_db` of full scale.

    Conservative threshold (-45 dB) so we don't eat real speech onsets.
    """

    if samples.size == 0:
        return samples
    threshold = 10 ** (threshold_db / 20.0)
    abs_samples = np.abs(samples)
    above = np.where(abs_samples > threshold)[0]
    if above.size == 0:
        return samples
    # Pad a tiny bit on each side so we don't clip the very first phoneme.
    pad = int(0.005 * sample_rate)  # 5 ms
    start = max(0, above[0] - pad)
    end = min(samples.size, above[-1] + pad)
    return samples[start:end]


def _apply_fade(
    samples: np.ndarray, sample_rate: int, fade_ms: int = 5
) -> np.ndarray:
    """Linear fade in/out at chunk edges. Eliminates click/pop on splice."""

    if samples.size == 0:
        return samples
    n = int(sample_rate * fade_ms / 1000)
    n = min(n, samples.size // 2)
    if n <= 0:
        return samples
    ramp_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    ramp_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    samples = samples.copy()
    samples[:n] *= ramp_in
    samples[-n:] *= ramp_out
    return samples
