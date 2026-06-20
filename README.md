# livekit-streaming-tts

Engine-agnostic, low-latency, self-hosted TTS for LiveKit voice agents. One WebSocket protocol, one client plugin, swappable backends — write your agent once, switch from OmniVoice to Pocket TTS to Kokoro by changing one env var.

## Why this exists

Most LiveKit TTS plugins are tightly coupled to one provider — change the model and you rewrite the plugin, redo latency tuning, and rediscover the same edge cases (idle-closed sockets, half-words from the LLM, no barge-in support, etc.). This package factors out the parts that are the same for every TTS:

- A LiveKit `tts.TTS` plugin with pooled WebSockets, sentence-aware streaming, and barge-in.
- A FastAPI server that does text normalization, sentence chunking, voice caching, and audio encoding (PCM/WAV, any sample rate).
- A small `TTSEngine` ABC — the only thing each backend implements.

## Supported engines

| Engine     | Voice cloning | Streaming gen | Multilingual | Speed | Footprint    |
|------------|---------------|---------------|--------------|-------|--------------|
| OmniVoice  | yes           | no            | yes          | yes   | GPU, ~3GB    |
| Pocket TTS | yes (kvcache) | no            | en           | no    | GPU, ~1GB    |
| Kokoro     | no (presets)  | no            | en/ja/zh     | yes   | GPU, ~330MB  |
| KittenTTS  | no (presets)  | no            | en           | no    | CPU, ~50MB   |
| XTTS v2    | yes (zero-shot)| yes          | 16 langs     | yes   | GPU, ~2GB    |
| Piper      | no (per-file) | no            | many         | yes   | CPU, ~30MB   |
| Bark       | no (presets)  | no            | yes          | no    | GPU, ~5GB    |

## Topology

```
┌────────────────────────────┐                ┌────────────────────────────┐
│  CLIENT host (CPU)         │     WS         │  SERVER host (GPU)         │
│  - LiveKit agent           │ ◄─────────────►│  - FastAPI + uvicorn       │
│  - StreamingTTS plugin     │                │  - TTSEngine (pocket /     │
│  - microphone, speaker     │                │    omnivoice / kokoro / …) │
└────────────────────────────┘                └────────────────────────────┘
```

The **same `pip install`** ships both halves:
- the **server** entrypoint (`create_app`, `examples/run_server.py`)
- the **client** plugin (`StreamingTTS` — drop into a LiveKit agent)

You install the package on the server with the engine extra you want, and on
the client with no extra (it doesn't need torch / engine deps — only
`aiohttp` + `livekit-agents`).

## Install

```bash
git clone https://github.com/<you>/livekit-streaming-tts
cd livekit-streaming-tts

# --- server host (GPU box, where the model runs) ---
pip install -e .[pocket]              # or [omnivoice], [kokoro], [xtts] ...
pip install -e .[kitten]              # CPU-only, no GPU needed

# --- client host (where your LiveKit agent runs) ---
pip install -e .                      # no extras — just the plugin & deps
```

Once published to PyPI, those become `pip install livekit-streaming-tts[pocket]`
on the server and `pip install livekit-streaming-tts` on the client.

## Run the server

```bash
TTS_ENGINE=pocket    TTS_PORT=8001 python examples/run_server.py
TTS_ENGINE=omnivoice TTS_PORT=8001 python examples/run_server.py
TTS_ENGINE=kitten    TTS_PORT=8001 python examples/run_server.py
```

The engine is loaded lazily — only the deps you `pip install`'d are imported.

Verify it's up:

```bash
curl http://localhost:8001/health
# {"status":"ok","engine":"pocket","capabilities":{...},"voices":[...]}
```

## Connect from the client

Two supported clients:

### 1) LiveKit agent (the main use case)

```python
from livekit.agents import Agent, AgentSession, WorkerOptions, cli
from livekit.plugins import openai, silero
from livekit_streaming_tts import StreamingTTS

tts = StreamingTTS(
    ws_url="ws://your-server:8001/tts/ws",
    voice="alba",
    sample_rate=24000,   # 8000 for telephony
    format="pcm",
    binary=True,
)
tts.prewarm()

session = AgentSession(
    vad=silero.VAD.load(),
    stt=openai.STT(),
    llm=openai.LLM(model="gpt-4o-mini"),
    tts=tts,                          # ← that's the entire integration
)
```

That's the whole client side. The plugin handles WS pooling, sentence
tokenization, retries, and barge-in.

See `examples/livekit_agent.py` for a runnable version.

### 2) Raw WebSocket (non-LiveKit integrations)

If you're building a telephony bridge, a CLI, or any non-LiveKit pipeline,
you can speak the WS protocol directly. See `examples/raw_ws_client.py`:

```bash
python examples/raw_ws_client.py "Hello from the streaming TTS server."
# wrote 96000 bytes of PCM → output.wav
```

The protocol is documented inline in that file and in `server.py`.

## Use in a LiveKit agent

```python
from livekit_streaming_tts import StreamingTTS

tts = StreamingTTS(
    ws_url="ws://your-tts-host:8001/tts/ws",
    voice="alba",            # any registered voice / preset / engine default
    sample_rate=24000,       # 8000 for telephony — see below
    format="pcm",            # "pcm" recommended; "wav" if you need a header
    binary=True,             # binary WS frames for PCM (~33% bandwidth saved)
)
tts.prewarm()                # opens one WS so the first turn is fast

session = AgentSession(stt=..., llm=..., tts=tts, vad=...)
```

## PCM vs WAV — which should you use?

**Use PCM with binary frames.** Three reasons:

1. **Smaller wire size.** WAV adds a 44-byte RIFF header per chunk (small) plus base64 inflates payload by 33% (significant on long replies). PCM-binary skips both.
2. **No decode step on the client.** The PCM bytes go straight into LiveKit's `AudioEmitter`. WAV-base64 requires `base64.b64decode` + WAV parse on every chunk.
3. **Quality is identical.** PCM and WAV are the same int16 samples — WAV just wraps them in a header. There is no quality difference.

When to use WAV: only if you're consuming the bytes outside a streaming pipeline (saving to disk, posting to an HTTP endpoint that wants a file format).

## Telephony

PSTN/SIP is 8kHz μ-law or A-law. LiveKit's SIP integration accepts 8kHz/16kHz PCM and handles G.711 conversion. For lowest latency on phone calls:

```python
StreamingTTS(sample_rate=8000, format="pcm", binary=True)
```

This makes the server resample once (engine native → 8kHz) and ship 1/3 the bytes vs 24kHz. Install `soxr` for higher-quality resampling:

```bash
pip install livekit-streaming-tts[hq-resample]
```

## Voice cloning

Engines that support it: **OmniVoice**, **Pocket TTS**, **XTTS v2**.

```bash
# Upload a reference clip and register under a voice_id
curl -F "file=@my-voice.wav" \
     -F "voice_id=alice" \
     -F "ref_text=transcript of the clip" \
     http://your-tts-host:8001/v1/voices/upload
```

Then pass `voice="alice"` to `StreamingTTS`. The server caches the prompt — subsequent generations are fast.

Pocket TTS bonus: voice prompts are exported to safetensors (kvcache). On restart the server reloads voices from disk in milliseconds instead of recomputing.

## Adding a new engine

```python
# my_engine.py
from livekit_streaming_tts.base import TTSEngine, GenerationParams, AudioChunk, EngineCapabilities
import numpy as np

class MyEngine(TTSEngine):
    name = "myengine"
    native_sample_rate = 22050
    capabilities = EngineCapabilities(voice_cloning=False, multilingual=False)

    def __init__(self):
        # Load your model here.
        ...

    def generate(self, params: GenerationParams) -> AudioChunk:
        # Your synthesis logic. Return float32 in [-1, 1].
        samples = self._model.synth(params.text)
        return AudioChunk(samples=samples.astype(np.float32),
                          sample_rate=self.native_sample_rate, is_final=True)

# Register at runtime:
from livekit_streaming_tts import register_engine
register_engine("myengine", MyEngine)
```

That's it. Sentence chunking, normalization, voice caching, WS protocol, audio encoding, the LiveKit plugin — all reused.

## Architecture

```
LLM tokens (sub-word frags)
    │
    │ via livekit input_ch
    ▼
SentenceTokenizer (joins frags → words)        ← client side
    │
    ▼
Pooled WebSocket (max 30s session)             ← client side
    │
    │ {"type":"text","data":{"text":"word "}}
    ▼
text_buffer + split_sentences (unicode)        ← server side
    │
    ▼
normalize_text (numbers, money, units, MD)     ← server side, off the GPU thread
    │
    ▼
GPUWorker queue (serialized engine access)     ← server side
    │
    ▼
TTSEngine.generate(params)                     ← swappable backend
    │
    ▼
np.clip + int16 + (optional) resample          ← server side
    │
    │ binary PCM frame (or base64 WAV)
    ▼
LiveKit AudioEmitter → WebRTC → user
```

## License

Apache 2.0.
