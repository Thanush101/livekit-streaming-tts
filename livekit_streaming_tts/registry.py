"""Engine discovery and instantiation.

How users pick an engine:
    1. Set TTS_ENGINE=kitten (or omnivoice / xtts / piper / bark / ...).
    2. Call `load_engine()` which lazy-imports the adapter so installing
       `pip install livekit-streaming-tts[kitten]` doesn't drag torch into a
       Piper-only deployment.

How contributors add an engine:
    1. Drop a file in `engines/<name>.py` exposing `class <Name>Engine(TTSEngine)`.
    2. Add an entry to `_BUILTIN` below.
    3. Add a pyproject extra `[<name>] = ["library==1.2"]`.
"""

from __future__ import annotations

import importlib
import os
from typing import Callable, Optional

from .base import TTSEngine


# name → (module path, class name). Lazy imports so missing optional deps
# don't break the package.
_BUILTIN: dict[str, tuple[str, str]] = {
    "omnivoice": ("livekit_streaming_tts.engines.omnivoice", "OmniVoiceEngine"),
    "kitten":    ("livekit_streaming_tts.engines.kitten",    "KittenEngine"),
    "kokoro":    ("livekit_streaming_tts.engines.kokoro",    "KokoroEngine"),
    "pocket":    ("livekit_streaming_tts.engines.pocket",    "PocketEngine"),
    "xtts":      ("livekit_streaming_tts.engines.xtts",      "XTTSEngine"),
    "piper":     ("livekit_streaming_tts.engines.piper",     "PiperEngine"),
    "bark":      ("livekit_streaming_tts.engines.bark",      "BarkEngine"),
}

# Custom engines registered at runtime by user code.
_CUSTOM: dict[str, Callable[..., TTSEngine]] = {}


def register_engine(name: str, factory: Callable[..., TTSEngine]) -> None:
    """Register a user-supplied engine factory under `name`.

    Useful for closed-source or internal engines that don't ship with the
    package. After registration, `TTS_ENGINE=name load_engine()` works.
    """

    _CUSTOM[name.lower()] = factory


def available_engines() -> list[str]:
    """Names that `load_engine` can resolve right now."""

    return sorted(set(_BUILTIN) | set(_CUSTOM))


def load_engine(name: Optional[str] = None, **kwargs) -> TTSEngine:
    """Instantiate an engine by name.

    Args:
        name: Engine name. Defaults to env var TTS_ENGINE, then "kitten"
            (smallest dependency footprint).
        **kwargs: Forwarded to the engine constructor.
    """

    chosen = (name or os.getenv("TTS_ENGINE") or "kitten").lower()

    if chosen in _CUSTOM:
        return _CUSTOM[chosen](**kwargs)

    if chosen not in _BUILTIN:
        raise ValueError(
            f"Unknown engine {chosen!r}. Available: {available_engines()}"
        )

    module_path, class_name = _BUILTIN[chosen]
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Engine {chosen!r} requires extra dependencies. Install with: "
            f"pip install livekit-streaming-tts[{chosen}]\n"
            f"Underlying error: {e}"
        ) from e

    engine_cls = getattr(module, class_name)
    return engine_cls(**kwargs)
