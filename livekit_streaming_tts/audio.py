"""Audio post-processing — clipping, scaling, encoding, resampling.

Centralizing this means engines just emit float32 in [-1, 1] and the server
handles everything else uniformly. Two formats supported:

    PCM (audio/pcm)  — raw int16 little-endian samples. No header.
                       Smaller, no decode step, telephony-friendly.
    WAV (audio/wav)  — int16 PCM with a 44-byte RIFF header per chunk.

Sample rate handling for telephony:
    Most engines run at 24kHz or 22.05kHz. Telephony (PSTN, SIP) is 8kHz μ-law
    or A-law (G.711). LiveKit's SIP integration accepts 8kHz/16kHz PCM and
    handles the codec conversion. So if the user is on a phone, set
    `sample_rate=8000` in the request and we resample once on the server,
    saving bandwidth and avoiding double-resampling at LiveKit.

Why np.clip is mandatory:
    Some engines emit samples slightly over [-1, 1] (numerical drift in the
    decoder). Without clip, `(audio * 32767).astype(int16)` overflows and
    wraps to -32768 — an audible click on every overflow.
"""

from __future__ import annotations

import io
import wave
from typing import Optional

import numpy as np


def to_int16_pcm(samples: np.ndarray) -> bytes:
    """Float32 [-1, 1] → int16 LE PCM bytes. Clips out-of-range values."""

    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    # CRITICAL: clip before scale. Without this, 1.0001 → -32768 (click).
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Float32 → 16-bit PCM WAV bytes (RIFF header + samples)."""

    pcm = to_int16_pcm(samples)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # int16 = 2 bytes
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def resample(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear resample. Good enough for speech; if you need higher quality,
    install `soxr` and we'll use that instead.

    Telephony note: 24000 → 8000 via this function works fine for voice;
    LiveKit's SIP path applies an anti-aliasing filter on top. If you hear
    aliasing artifacts on phone calls, install `soxr`.
    """

    if src_sr == dst_sr:
        return samples

    try:
        import soxr  # type: ignore

        return soxr.resample(samples, src_sr, dst_sr).astype(np.float32)
    except ImportError:
        # Linear fallback. Acceptable for most speech use cases.
        ratio = dst_sr / src_sr
        new_len = int(round(len(samples) * ratio))
        x_old = np.linspace(0, 1, num=len(samples), endpoint=False)
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        return np.interp(x_new, x_old, samples).astype(np.float32)


def encode(
    samples: np.ndarray,
    *,
    src_sample_rate: int,
    target_sample_rate: Optional[int] = None,
    format: str = "pcm",
) -> tuple[bytes, int]:
    """One-call resample + encode. Returns (bytes, final_sample_rate).

    Used by the server's WS handler — just pass it the raw engine output and
    the requested format/rate, get back wire-ready bytes.
    """

    sr = target_sample_rate or src_sample_rate
    if sr != src_sample_rate:
        samples = resample(samples, src_sample_rate, sr)

    if format == "pcm":
        return to_int16_pcm(samples), sr
    if format == "wav":
        return to_wav(samples, sr), sr
    raise ValueError(f"unsupported format: {format!r} (expected pcm or wav)")
