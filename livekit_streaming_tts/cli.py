"""Console entry point — `tts-server` after pip install.

Usage:
    tts-server                          # uses TTS_ENGINE env var (default: kitten)
    tts-server --engine pocket
    tts-server --engine omnivoice --port 8001 --host 0.0.0.0

Why this exists:
    Without it, every user has to write the uvicorn boilerplate themselves.
    `tts-server` is one command, no Python file to author.
"""

from __future__ import annotations

import argparse
import os
import sys

from .registry import available_engines


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tts-server",
        description="livekit-streaming-tts — engine-agnostic TTS server.",
    )
    parser.add_argument(
        "--engine",
        default=os.getenv("TTS_ENGINE"),
        help=(
            "TTS engine name. Falls back to env TTS_ENGINE, then 'kitten'. "
            f"Available: {', '.join(available_engines())}."
        ),
    )
    parser.add_argument(
        "--host", default=os.getenv("TTS_HOST", "0.0.0.0"),
        help="Bind host (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("TTS_PORT", "8001")),
        help="Bind port (default: 8001).",
    )
    parser.add_argument(
        "--max-queue-size", type=int, default=50,
        help="Max GPU queue size before back-pressure (default: 50).",
    )
    parser.add_argument(
        "--voice-cache-max", type=int, default=64,
        help="Max cloned voices kept hot in LRU (default: 64).",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is required. Install with: pip install uvicorn[standard]",
            file=sys.stderr,
        )
        sys.exit(1)

    # Late import — keeps `tts-server --help` snappy even when the engine
    # has heavy deps.
    from .server import create_app

    app = create_app(
        engine_name=args.engine,
        max_queue_size=args.max_queue_size,
        voice_cache_max=args.voice_cache_max,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ws_max_size=16 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
