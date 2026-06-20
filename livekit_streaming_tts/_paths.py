"""Portable, write-able default paths for caches and persisted voices.

Why this exists:
    Hardcoding `/var/lib/...` works on Linux as root, but fails on macOS,
    in containers run as non-root, on shared dev boxes, etc. We pick a
    user-writable location by default and let it be overridden via env.

Resolution order:
    1. Explicit env var (e.g. TTS_VOICES_DIR or POCKET_TTS_CACHE_DIR).
    2. XDG_DATA_HOME / livekit-streaming-tts / <subdir>      (Linux convention)
    3. ~/.local/share/livekit-streaming-tts / <subdir>        (fallback)

These are all user-writable without sudo, so a fresh `pip install` plus
`tts-server --engine pocket` just works.
"""

from __future__ import annotations

import os
from pathlib import Path


_APP_NAME = "livekit-streaming-tts"


def _default_data_root() -> Path:
    xdg = os.getenv("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / _APP_NAME
    return Path.home() / ".local" / "share" / _APP_NAME


def data_dir(subdir: str, *, env_var: str | None = None) -> str:
    """Resolve a writable data subdirectory.

    Args:
        subdir: relative subpath under the app's data root (e.g. "voices").
        env_var: optional env var name that overrides the default if set.

    Returns:
        Absolute path. Caller is responsible for `os.makedirs(...,
        exist_ok=True)` when it actually intends to write.
    """

    if env_var:
        override = os.getenv(env_var)
        if override:
            return os.path.abspath(os.path.expanduser(override))
    return str(_default_data_root() / subdir)
