"""livekit-streaming-tts — engine-agnostic, low-latency TTS for LiveKit agents.

Public API:
    TTSEngine        — abstract base every backend implements
    GenerationParams — common knobs (text, voice, speed, language, seed, ...)
    AudioChunk       — what `generate_stream` yields (float32 mono + sample_rate)
    StreamingTTS     — the LiveKit `tts.TTS` plugin (client side)
    create_app       — engine-agnostic FastAPI server (server side)
"""

from .base import TTSEngine, GenerationParams, AudioChunk, EngineCapabilities
from .plugin import StreamingTTS
from .server import create_app
from .registry import load_engine, register_engine, available_engines

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

__version__ = "0.1.0"
