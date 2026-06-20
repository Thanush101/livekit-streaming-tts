"""Raw WebSocket client — for non-LiveKit integrations.

Use this if you want to consume the TTS server WITHOUT LiveKit (e.g. a
custom telephony bot, a CLI tool, an Asterisk/FreeSWITCH bridge, or any
non-Python client).

The protocol is identical to what the LiveKit `StreamingTTS` plugin uses,
just spoken directly over `aiohttp` here so you can see every wire frame.

For LiveKit users: don't use this. Use `examples/livekit_agent.py` — the
plugin handles tokenization, pooling, retries, and barge-in for you.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import wave

import aiohttp


WS_URL = "ws://localhost:8001/tts/ws"


async def synthesize(text: str, *, voice: str | None = None) -> bytes:
    """Send `text` and return raw int16 PCM bytes (24kHz mono)."""

    pcm = bytearray()

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(WS_URL) as ws:
            # 1) Configure the session.
            config = {
                "sample_rate": 24000,
                "format": "pcm",
                "binary": True,           # binary frames = no base64 overhead
                "normalize": True,
            }
            if voice:
                config["voice"] = voice
            await ws.send_str(json.dumps({"type": "config", "data": config}))

            # 2) Wait for config_accepted.
            ack = await ws.receive(timeout=10)
            if ack.type == aiohttp.WSMsgType.TEXT:
                if json.loads(ack.data).get("type") == "error":
                    raise RuntimeError(f"server rejected config: {ack.data}")

            # 3) Stream text. You can call send_str many times to simulate
            #    LLM streaming — each chunk gets sentence-split server-side.
            await ws.send_str(json.dumps({
                "type": "text",
                "data": {"text": text},
            }))

            # 4) Tell the server you're done sending text.
            await ws.send_str(json.dumps({"type": "flush"}))

            # 5) Drain audio + final event.
            while True:
                msg = await ws.receive(timeout=30)
                if msg.type == aiohttp.WSMsgType.BINARY:
                    pcm.extend(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data["type"] == "audio":
                        # Fallback path if the server isn't using binary frames.
                        pcm.extend(base64.b64decode(data["data"]["audio"]))
                    elif data["type"] == "event" and data["data"].get("event_type") == "final":
                        break
                    elif data["type"] == "error":
                        raise RuntimeError(f"server error: {data['data']}")
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break

    return bytes(pcm)


def write_wav(path: str, pcm: bytes, sample_rate: int = 24000) -> None:
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


async def main() -> None:
    text = " ".join(sys.argv[1:]) or "Hello from the streaming TTS server."
    pcm = await synthesize(text, voice=None)
    write_wav("output.wav", pcm)
    print(f"wrote {len(pcm)} bytes of PCM → output.wav")


if __name__ == "__main__":
    asyncio.run(main())
