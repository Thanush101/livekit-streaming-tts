"""livekit-streaming-tts — engine-agnostic, low-latency TTS for LiveKit agents.

Public API:
    TTSEngine        — abstract base every backend implements
    GenerationParams — common knobs (text, voice, speed, language, seed, ...)
    AudioChunk       — what `generate_stream` yields (float32 mono + sample_rate)
    StreamingTTS     — the LiveKit `tts.TTS` plugin (client side)
    create_app       — engine-agnostic FastAPI server (server side)
"""

from .base import TTSEngine, GenerationParams, AudioChunk, EngineCapabilities
from .registry import load_engine, register_engine, available_engines

# The LiveKit client and the FastAPI app pull optional deps (livekit-agents,
# fastapi). Load them on first use so importing an engine does not require either.


def __getattr__(name: str):
    if name == "StreamingTTS":
        from .plugin import StreamingTTS

        return StreamingTTS
    if name == "create_app":
        from .server import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TTSEngine",
    "GenerationParams",
    "AudioChunk",
    "EngineCapabilities",
    "StreamingTTS",
    "create_app",
    "load_engine",
    "register_engine",
    "available_engines",
]

__version__ = "0.1.2"
