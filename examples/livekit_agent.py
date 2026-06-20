"""LiveKit voice agent using livekit-streaming-tts as the TTS plugin.

Replaces any `tts=openai.TTS()` with our engine-agnostic streaming plugin.
The agent doesn't care which engine the server is running — pocket, kitten,
omnivoice, xtts — same WS protocol, same client.
"""

import os

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai, silero  # for STT/LLM/VAD

from livekit_streaming_tts import StreamingTTS


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # PCM + binary frames is the recommended low-latency configuration.
    # For telephony (8kHz pipeline), set sample_rate=8000 — the server
    # resamples once and skips a step at LiveKit's SIP gateway.
    tts = StreamingTTS(
        ws_url=os.getenv("TTS_WS_URL", "ws://localhost:8001/tts/ws"),
        voice=os.getenv("TTS_VOICE", "alba"),  # works for pocket; harmless for others
        sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "24000")),
        format="pcm",
        binary=True,
    )
    tts.prewarm()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=tts,
    )
    await session.start(agent=Agent(instructions="You are a helpful voice assistant."), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
