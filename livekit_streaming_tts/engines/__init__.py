"""Engine adapters. Each file is independent and only imports its own
optional dependency at module load time.

Add a new engine in three steps:
    1. Create engines/<name>.py exposing class <Name>Engine(TTSEngine).
    2. Register it in `livekit_streaming_tts/registry.py::_BUILTIN`.
    3. Add a pyproject extra `[<name>] = ["library==..."]`.
"""
