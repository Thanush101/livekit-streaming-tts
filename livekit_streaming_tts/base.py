"""Abstract base for every TTS backend.

Why this exists:
    Every TTS model takes text in and returns audio out. The *common* knobs are
    text, voice, speed, language, seed, sample rate. Everything else (steps,
    guidance scale, temperature, cfg, etc.) is engine-specific and goes through
    a single `extra` dict so adapters can grab what they need without polluting
    the shared interface.

What an engine MUST implement:
    - capabilities (class-level): what features the engine supports.
    - generate(params)            : one-shot synthesis returning a single AudioChunk.

What an engine MAY implement:
    - generate_stream(params)     : yield AudioChunks as they're ready (lower TTFB).
    - register_voice(voice_id, audio_path, ref_text=None) : voice cloning.
    - delete_voice(voice_id)      : drop a registered voice.
    - list_voices()               : enumerate available voices.
    - close()                     : free resources (GPU memory, model handles).

Audio contract:
    AudioChunk.samples is mono float32 in [-1.0, 1.0].
    AudioChunk.sample_rate is the engine's native sample rate.
    Conversion to int16 PCM / WAV is handled centrally in audio.py — engines
    must NOT clip, scale, or encode their own output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class EngineCapabilities:
    """What an engine can do. Used by the server to validate requests early.

    Each flag describes a *guarantee* — `True` means the engine implements it,
    `False` means callers must not rely on it (the server will return a clear
    error instead of silently dropping the param).
    """

    voice_cloning: bool = False
    """True if `register_voice` produces usable speaker conditioning.

    KittenTTS / Piper: False (fixed voice list).
    XTTS v2 / OmniVoice / Bark (with prompts): True.
    """

    streaming_generation: bool = False
    """True if `generate_stream` yields chunks *during* synthesis (lower TTFB).

    Most engines complete the whole utterance before returning anything; for
    those, the default `generate_stream` falls back to a single-chunk wrapper
    around `generate`. XTTS v2's streaming mode and Piper's incremental
    output set this to True.
    """

    multilingual: bool = False
    """True if the engine accepts a `language` param meaningfully."""

    speed_control: bool = False
    """True if `params.speed` actually changes playback speed."""


@dataclass
class GenerationParams:
    """Common params every engine accepts.

    Engine-specific knobs go in `extra` — adapters pop what they recognize and
    ignore the rest. This keeps the public API tiny while letting power users
    pass through provider-specific tuning.
    """

    text: str
    """The text to synthesize. Already normalized by the time it reaches the
    engine (server runs `normalize_text` first)."""

    voice: Optional[str] = None
    """A registered voice_id (for engines that support cloning) or a preset
    name (for engines like KittenTTS that ship a fixed list). None = default
    voice."""

    ref_audio: Optional[str] = None
    """Filesystem path to a reference clip for ad-hoc (uncached) voice
    cloning. Engines without cloning ignore this."""

    ref_text: Optional[str] = None
    """Transcript of `ref_audio`. Some engines (the ones that need it for
    speaker embedding) require this; others ignore it."""

    language: Optional[str] = None
    """ISO 639-1 code (`"en"`, `"hi"`, `"ja"`, ...). Engines that don't
    support multilingual ignore this."""

    speed: Optional[float] = None
    """Playback speed. 1.0 = native, >1.0 = faster, <1.0 = slower. None =
    engine default. Ignored by engines without speed control."""

    seed: Optional[int] = None
    """Random seed for reproducibility. None = non-deterministic."""

    sample_rate: Optional[int] = None
    """Requested output sample rate. If None or != engine.native_sample_rate,
    audio.py resamples on the way out. Telephony usually wants 8000."""

    extra: dict = field(default_factory=dict)
    """Engine-specific overrides. Keys are matched case-sensitively. Examples:
        steps=8, guidance_scale=2.0       (diffusion engines)
        temperature=0.7, top_k=50         (autoregressive)
        length_scale=1.0                  (Piper)
    """


@dataclass
class AudioChunk:
    """A chunk of synthesized audio.

    Always mono float32 in [-1.0, 1.0]. The server clips, scales to int16,
    and encodes to PCM/WAV — engines do not.
    """

    samples: np.ndarray
    """1-D float32 numpy array."""

    sample_rate: int
    """Native sample rate of the engine."""

    is_final: bool = False
    """True on the last chunk of a single `generate_stream` call. Lets the
    server send a `final` event without inspecting the iterator."""


class TTSEngine(ABC):
    """Subclass this to add a new backend.

    Lifecycle:
        engine = MyEngine(...)        # __init__ loads weights, allocates GPU.
        await engine.generate(params) # called many times.
        engine.close()                # release resources on shutdown.

    Threading:
        The server calls `generate` and `generate_stream` from a single worker
        thread (via `loop.run_in_executor`). Engines do NOT need to be
        thread-safe across multiple concurrent calls — but they MUST release
        the GIL on heavy work so the FastAPI event loop can keep serving WS
        traffic. Standard PyTorch/numpy ops do this; pure-Python loops over
        long arrays do not.
    """

    capabilities: EngineCapabilities = EngineCapabilities()
    """Override at class level. The server reads this to reject impossible
    requests early (e.g. cloning request to a non-cloning engine)."""

    name: str = "base"
    """Display name. Used in /v1/models and logs."""

    native_sample_rate: int = 24000
    """The engine's natural output rate. The server resamples to whatever the
    client asks for."""

    @abstractmethod
    def generate(self, params: GenerationParams) -> AudioChunk:
        """One-shot synthesis. Blocking — runs on the GPU worker thread.

        MUST return audio at `self.native_sample_rate` as float32 in [-1, 1].
        DO NOT clip, scale to int16, or write WAV — `audio.py` handles that.
        """

    def generate_stream(self, params: GenerationParams) -> Iterator[AudioChunk]:
        """Stream chunks as they're produced. Default: one chunk = full audio.

        Override if the engine supports incremental decode (XTTS v2 streaming,
        Piper). Lower TTFB at the cost of more WS frames.

        The last yielded chunk MUST have is_final=True.
        """

        chunk = self.generate(params)
        yield AudioChunk(
            samples=chunk.samples, sample_rate=chunk.sample_rate, is_final=True
        )

    def register_voice(
        self,
        voice_id: str,
        audio_path: str,
        ref_text: Optional[str] = None,
    ) -> None:
        """Cache a voice prompt. No-op for engines without cloning.

        The server pre-checks `capabilities.voice_cloning` before calling this,
        so engines that DO support cloning can assume they're being called
        legitimately and just raise on bad input.
        """

        raise NotImplementedError(
            f"{self.name} does not support voice cloning"
        )

    def delete_voice(self, voice_id: str) -> None:
        """Drop a registered voice. No-op if not registered (idempotent)."""

        return None

    def list_voices(self) -> list[str]:
        """Return all voices the engine can use right now (built-in + cached
        clones). Empty list = engine has no concept of named voices."""

        return []

    def close(self) -> None:
        """Release GPU memory, close model handles. Called on server shutdown.

        Default is a no-op since most engines clean up via Python GC.
        """

        return None
