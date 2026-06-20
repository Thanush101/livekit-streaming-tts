"""Start the TTS server.

First-time setup (from the repo root):

    pip install -e .[pocket]          # or [omnivoice], [kitten], [kokoro], ...

Then:

    TTS_ENGINE=kitten    python examples/run_server.py
    TTS_ENGINE=pocket    python examples/run_server.py
    TTS_ENGINE=omnivoice python examples/run_server.py

If you see `ModuleNotFoundError: No module named 'livekit_streaming_tts'`,
you skipped the `pip install -e .` step.
"""

import os
import uvicorn

from livekit_streaming_tts import create_app


app = create_app(engine_name=os.getenv("TTS_ENGINE"))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("TTS_PORT", "8001")),
        ws_max_size=16 * 1024 * 1024,
    )
