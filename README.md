# livekit-streaming-tts

Engine agnostic, low latency, self hosted TTS for LiveKit voice agents. One WebSocket protocol, one client plugin, swappable backends. Write your agent once, switch from OmniVoice to Pocket TTS to Kokoro by changing one env var.

## Why this exists

Most LiveKit TTS plugins are tightly coupled to one provider. Change the model and you rewrite the plugin, redo latency tuning, and rediscover the same edge cases (idle-closed sockets, half-words from the LLM, no barge-in support, etc.). This package factors out the parts that are the same for every TTS:

- A LiveKit `tts.TTS` plugin with pooled WebSockets, sentence-aware streaming, and barge-in.
- A FastAPI server that does text normalization, sentence chunking, voice caching, and audio encoding (PCM/WAV, any sample rate).
- A small `TTSEngine` ABC. The only thing each backend implements.

## Supported engines

| Engine     | Voice cloning  | Streaming gen | Multilingual | Speed | Footprint    | PyPI dep                |
|------------|----------------|---------------|--------------|-------|--------------|-------------------------|
| OmniVoice  | yes            | no            | yes          | yes   | GPU, ~3GB    | `omnivoice`             |
| Pocket TTS | yes (kvcache)  | no            | en           | no    | GPU, ~1GB    | `pocket-tts`            |
| Kokoro     | no (presets)   | no            | en/ja/zh     | yes   | GPU, ~330MB  | `kokoro`                |
| KittenTTS  | no (presets)   | no            | en           | no    | CPU, ~50MB   | `kittentts` + `espeak`  |
| XTTS v2    | yes (zero-shot)| yes           | 16 langs     | yes   | GPU, ~2GB    | `TTS`                   |
| Piper      | no (per-file)  | no            | many         | yes   | CPU, ~30MB   | `piper-tts`             |
| Bark       | no (presets)   | no            | yes          | no    | GPU, ~5GB    | `bark`                  |

## Topology

```
┌────────────────────────────┐                ┌────────────────────────────┐
│  CLIENT host (CPU)         │     WS         │  SERVER host (GPU)         │
│  - LiveKit agent           │ ◄─────────────►│  - FastAPI + uvicorn       │
│  - StreamingTTS plugin     │                │  - TTSEngine (pocket /     │
│  - microphone, speaker     │                │    omnivoice / kokoro)     │
└────────────────────────────┘                └────────────────────────────┘
```

The same `pip install` ships both halves:

- the **server** entrypoint (`tts-server` console command, `create_app` factory)
- the **client** plugin (`StreamingTTS`, drop into a LiveKit agent)

You install the package on the server with the engine extra you want, and on the client with no extra (it does not need torch or engine deps, only `aiohttp` and `livekit-agents`).

## Install

### From PyPI (recommended)

```bash
# CLIENT host (where your LiveKit agent runs)
pip install livekit-streaming-tts

# SERVER host (where the model runs)
pip install 'livekit-streaming-tts[pocket]'      # Pocket TTS
pip install 'livekit-streaming-tts[omnivoice]'   # OmniVoice
pip install 'livekit-streaming-tts[kokoro]'      # Kokoro
pip install 'livekit-streaming-tts[kitten]'      # KittenTTS (CPU only)
pip install 'livekit-streaming-tts[xtts]'        # Coqui XTTS v2
pip install 'livekit-streaming-tts[piper]'       # Piper (CPU only)
pip install 'livekit-streaming-tts[bark]'        # Suno Bark
```

The single quotes around `'livekit-streaming-tts[pocket]'` are required on **zsh** (macOS default), which otherwise treats `[pocket]` as a glob pattern. On bash they are optional.

### If an extra fails to install

Some engine packages require extra system tooling that pip cannot install for you. Install the engine package directly and the matching system deps:

```bash
# KittenTTS needs espeak on the system
sudo apt install -y espeak       # Linux
brew install espeak              # macOS
pip install kittentts livekit-streaming-tts

# Pocket TTS
pip install pocket-tts livekit-streaming-tts

# OmniVoice (k2-fsa)
pip install omnivoice livekit-streaming-tts

# Bark
pip install bark livekit-streaming-tts

# XTTS (Coqui)
pip install TTS livekit-streaming-tts
```

This is the fallback path. Use it if `pip install 'livekit-streaming-tts[engine]'` errors out (typically because of platform-specific wheels, espeak, or torch index URLs).

### From source (for development / contributing)

```bash
git clone https://github.com/Thanush101/livekit-streaming-tts
cd livekit-streaming-tts

pip install -e '.[pocket]'      # editable install + Pocket TTS
pip install -e '.'              # editable install, no engine
```

## Run the server

After `pip install`, use the `tts-server` console command:

```bash
tts-server --engine pocket --port 8001
tts-server --engine omnivoice --port 8001
tts-server --engine kitten --port 8001
```

Or via env var:

```bash
TTS_ENGINE=pocket TTS_PORT=8001 tts-server
```

If the engine library is missing you will see `ModuleNotFoundError: No module named 'pocket_tts'` (or similar). Install the engine package as shown in the *If an extra fails to install* section.

If you cloned the repo for development instead of pip-installing, the equivalent is `python examples/run_server.py`.

The engine is loaded lazily, so only the deps you `pip install`'d are imported.

Verify it is up:

```bash
curl http://localhost:8001/health
# {"status":"ok","engine":"pocket","capabilities":{...},"voices":[...]}
```

## Connect from the client

Two supported clients.

### 1. LiveKit agent (the main use case)

```python
from livekit.agents import Agent, AgentSession, WorkerOptions, cli
from livekit.plugins import openai, silero
from livekit_streaming_tts import StreamingTTS

tts = StreamingTTS(
    ws_url="ws://your-server:8001/tts/ws",
    voice="alba",
    sample_rate=24000,    # 8000 for telephony
    format="pcm",
    binary=True,
)
tts.prewarm()             # opens one WS so the first turn is fast

session = AgentSession(
    vad=silero.VAD.load(),
    stt=openai.STT(),
    llm=openai.LLM(model="gpt-4o-mini"),
    tts=tts,              # the entire integration
)
```

That is the whole client side. The plugin handles WS pooling, sentence tokenization, retries, and barge-in.

See `examples/livekit_agent.py` for a runnable version.

#### Drop-in replacement for an existing TTS plugin

If you are already using `OmniVoiceTTS`, `elevenlabs.TTS`, etc. inside a TTS-provider switch:

```python
elif tts_provider == "omnivoice":
    tts_engine = StreamingTTS(
        ws_url=cfg.get("tts_ws_url", os.getenv("OMNIVOICE_WS_URL", "ws://localhost:8001/tts/ws")),
        voice="alba",
        sample_rate=24000,
        format="pcm",
        binary=True,
    )
```

This single block works for **every** engine the server supports. Switching from Pocket TTS to OmniVoice to KittenTTS is a server-side env var change (`TTS_ENGINE=...`); the agent code does not change. The only thing you might want to update is `voice=` to a name the new engine recognises (see the voice mapping below).

### Voice names by engine

`voice=` is forwarded to the engine. What names are valid depends on which engine the server is running:

| Engine     | Built-in voice names                              | Custom voices?                                  |
|------------|---------------------------------------------------|-------------------------------------------------|
| Pocket TTS | `alba` (default), plus any HuggingFace voice URL  | yes, via `/v1/voices/upload` or `hf://...` URL  |
| OmniVoice  | none built-in                                     | yes, via `/v1/voices/upload` (always required)  |
| Kokoro     | `af_heart`, `af_bella`, `am_adam`, `bf_emma`, ... | no (preset list only)                           |
| KittenTTS  | `expr-voice-2-f`, `expr-voice-2-m`, ...           | no (preset list only)                           |
| XTTS v2    | none built-in                                     | yes, via `/v1/voices/upload` (always required)  |
| Piper      | `default` only                                    | no (one model file = one voice)                 |
| Bark       | `v2/en_speaker_0` ... `v2/en_speaker_9`           | no (preset list only)                           |

If you pass a name the engine does not recognise, it falls back to its default. To list what your running server actually accepts:

```bash
curl http://localhost:8001/v1/voices
```

### 2. Raw WebSocket (non-LiveKit integrations)

If you are building a telephony bridge, a CLI, or any non-LiveKit pipeline, you can speak the WS protocol directly. See `examples/raw_ws_client.py`:

```bash
python examples/raw_ws_client.py "Hello from the streaming TTS server."
# wrote 96000 bytes of PCM → output.wav
```

The protocol is documented inline in that file and in `server.py`.

## PCM vs WAV: which should you use?

**Use PCM with binary frames.** Three reasons:

1. **Smaller wire size.** WAV adds a 44-byte RIFF header per chunk (small), plus base64 inflates payload by 33% (significant on long replies). PCM-binary skips both.
2. **No decode step on the client.** The PCM bytes go straight into LiveKit's `AudioEmitter`. WAV-base64 requires `base64.b64decode` plus a WAV parse on every chunk.
3. **Quality is identical.** PCM and WAV are the same int16 samples. WAV just wraps them in a header. There is no quality difference.

When to use WAV: only if you are consuming the bytes outside a streaming pipeline (saving to disk, posting to an HTTP endpoint that wants a file format).

## Telephony

PSTN/SIP is 8kHz μ-law or A-law. LiveKit's SIP integration accepts 8kHz/16kHz PCM and handles G.711 conversion. For lowest latency on phone calls:

```python
StreamingTTS(sample_rate=8000, format="pcm", binary=True)
```

This makes the server resample once (engine native to 8kHz) and ship 1/3 the bytes vs 24kHz. Install `soxr` for higher-quality resampling:

```bash
pip install 'livekit-streaming-tts[hq-resample]'
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

Then pass `voice="alice"` to `StreamingTTS`. The server caches the prompt and subsequent generations are fast.

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

That is it. Sentence chunking, normalization, voice caching, WS protocol, audio encoding, the LiveKit plugin: all reused.

## Architecture

```
LLM tokens (sub-word frags)
    │
    │ via livekit input_ch
    ▼
SentenceTokenizer (joins frags into words)     ← client side
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
