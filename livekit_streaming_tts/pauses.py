"""Pause markers for Pocket TTS.

pocket-tts 2.1.0 has no pause API (no ``split_on_pauses``). The marker
grammar here follows the unmerged kyutai-labs/pocket-tts#242: shorthand
``[pause]`` / ``[pause:N]`` and SSML ``<break>``. Silence is not generated
by the model. Callers turn durations into exact zeros and run text
normalization only on the spoken pieces, so a marker never reaches
``prepare_text_prompt``.
"""

from __future__ import annotations

import re
from typing import Callable, Optional, Union

import numpy as np

DEFAULT_PAUSE_SECONDS = 0.5
MAX_PAUSE_SECONDS = 10.0

# SSML names the levels and does not define seconds. These are the
# conventional durations used by pocket-tts#242.
SSML_BREAK_STRENGTHS = {
    "none": 0.0,
    "x-weak": 0.1,
    "weak": 0.25,
    "medium": 0.5,
    "strong": 0.75,
    "x-strong": 1.0,
}

PAUSE_PATTERN = re.compile(
    r"""
      \[\s*pause\s*(?::\s*(?P<shorthand>[0-9]*\.?[0-9]+)\s*)?\]
    | <\s*break\b(?P<attrs>[^>]*?)/?\s*>
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SPEAK_TAG = re.compile(r"</?\s*speak\b[^>]*>", re.IGNORECASE)
_CLOSING_BREAK = re.compile(r"</\s*break\s*>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<\s*/?\s*([a-zA-Z][\w-]*)\b[^>]*>")
_BREAK_TIME = re.compile(
    r"""time\s*=\s*["']?\s*([0-9]*\.?[0-9]+)\s*(ms|s)\b""",
    re.IGNORECASE,
)
_BREAK_STRENGTH = re.compile(
    r"""strength\s*=\s*["']?\s*([a-zA-Z-]+)""",
    re.IGNORECASE,
)

SpeechPart = Union[str, float]
NormalizeFn = Callable[..., str]
SpeakableFn = Callable[[Optional[str]], bool]


def _break_seconds(attrs: str, default_seconds: float) -> float:
    """Duration of one ``<break>`` tag. ``time`` wins over ``strength``."""

    time_match = _BREAK_TIME.search(attrs)
    if time_match is not None:
        value = float(time_match.group(1))
        return value / 1000.0 if time_match.group(2).lower() == "ms" else value

    strength_match = _BREAK_STRENGTH.search(attrs)
    if strength_match is not None:
        name = strength_match.group(1).lower()
        if name not in SSML_BREAK_STRENGTHS:
            raise ValueError(
                f"Unknown <break strength={name!r}>. "
                f"Valid values: {', '.join(SSML_BREAK_STRENGTHS)}."
            )
        return SSML_BREAK_STRENGTHS[name]
    return default_seconds


def split_on_pauses(
    text: str,
    default_seconds: float = DEFAULT_PAUSE_SECONDS,
    max_seconds: float = MAX_PAUSE_SECONDS,
) -> list[SpeechPart]:
    """Split text into spoken strings and silence durations in seconds.

    ``[pause]`` and ``<break/>`` last ``default_seconds`` (0.5). ``[pause:1.5]``,
    ``<break time="1.5s"/>``, and ``<break time="500ms"/>`` use that duration.
    ``<break strength="weak"/>`` uses :data:`SSML_BREAK_STRENGTHS`. When a tag
    has both ``time`` and ``strength``, ``time`` wins. Durations are clamped
    to ``[0, max_seconds]`` (10s).

    A surrounding ``<speak>`` wrapper is removed. Any other SSML tag raises
    ``ValueError``. Malformed shorthand such as ``[pause:]`` stays in the
    spoken text. Text with no markers is returned as one string.
    """

    text = _SPEAK_TAG.sub("", text)
    text = _CLOSING_BREAK.sub("", text)

    unsupported = {
        match.group(1).lower()
        for match in _ANY_TAG.finditer(text)
        if match.group(1).lower() != "break"
    }
    if unsupported:
        listed = ", ".join(f"<{name}>" for name in sorted(unsupported))
        raise ValueError(
            f"Unsupported SSML tag(s): {listed}. Pocket TTS supports <break> only "
            "(and an optional <speak> wrapper); remove the others."
        )

    parts: list[SpeechPart] = []
    cursor = 0
    for match in PAUSE_PATTERN.finditer(text):
        before = text[cursor : match.start()]
        if before.strip():
            parts.append(before)
        if match.group("shorthand") is not None:
            seconds = float(match.group("shorthand"))
        elif match.group("attrs") is not None:
            seconds = _break_seconds(match.group("attrs"), default_seconds)
        else:
            seconds = default_seconds
        parts.append(min(max(seconds, 0.0), max_seconds))
        cursor = match.end()

    tail = text[cursor:]
    if tail.strip():
        parts.append(tail)
    return parts


def silence_samples(seconds: float, sample_rate: int) -> np.ndarray:
    """Exact zeros at ``sample_rate``. A non-positive duration is empty."""

    if seconds <= 0 or sample_rate <= 0:
        return np.zeros(0, dtype=np.float32)
    count = int(round(float(seconds) * int(sample_rate)))
    return np.zeros(count, dtype=np.float32)


def speech_plan(
    text: str,
    *,
    pauses: bool,
    normalize: bool,
    language: Optional[str] = None,
    normalize_fn: Optional[NormalizeFn] = None,
    speakable: Optional[SpeakableFn] = None,
) -> list[SpeechPart]:
    """Spoken strings and silence durations for one chunk.

    With ``pauses`` false, the text is returned unchanged except for the
    optional normalizer. No silence is invented, and pause markers are not
    parsed.

    With ``pauses`` true, markers are split off before normalization. The
    normalizer sees only spoken spans, never ``[pause:1.5]`` or ``<break>``.
    """

    if normalize_fn is None or speakable is None:
        from .normalize import is_speakable, normalize_text

        if normalize_fn is None:
            normalize_fn = normalize_text
        if speakable is None:
            speakable = is_speakable

    if not pauses:
        spoken = normalize_fn(text, language=language) if normalize else text
        return [spoken] if speakable(spoken) else []

    planned: list[SpeechPart] = []
    for part in split_on_pauses(text):
        if isinstance(part, float):
            planned.append(part)
            continue
        spoken = normalize_fn(part, language=language) if normalize else part
        if speakable(spoken):
            planned.append(spoken)
    return planned
