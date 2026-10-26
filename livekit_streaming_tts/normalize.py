"""Text normalization that runs BEFORE the engine sees the input.

Why server-side, not engine-side:
    Every TTS model mispronounces "$5.50" or "3:30pm" the same way. Doing the
    rewrite once here means swapping engines doesn't change pronunciation.

Why on the WS handler thread, not the GPU worker:
    `normalize_text` takes 5–20ms on long input. If it ran inside the GPU
    worker (as the original OmniVoice server did), the worker queue slot would
    be held during normalization — blocking the next request from starting on
    the GPU. Doing it here frees the GPU.

Languages handled:
    All Latin-script languages (number/symbol normalization via inflect).
    Sentence boundaries: . ! ?  plus  । (Devanagari)  ။ (Burmese)  。 (CJK)
    ？ (CJK)  ! (CJK)  ؟ (Arabic)  ։ (Armenian)  ። (Ge'ez)  ။ (Burmese).
    Number-to-words is English-only via `inflect`; for other languages we
    leave digits alone (most modern multilingual TTS handle them natively).
"""

from __future__ import annotations

import re
from typing import Callable, Optional

import inflect


_INFLECT = inflect.engine()


# ---- Sentence boundary characters --------------------------------------
#
# These are the characters that mark "this sentence is done — flush to TTS".
# Adding a character here means the server will dispatch on it; languages we
# don't list will only flush at end-of-stream (still works, just higher
# latency for users speaking that language).
SENTENCE_TERMINATORS = (
    ".!?"             # Latin
    "।॥"              # Devanagari (Hindi, Marathi, Sanskrit, ...)
    "။"               # Burmese
    "。？！"           # CJK fullwidth
    "؟"               # Arabic
    "።"               # Ge'ez (Amharic, Tigrinya)
    "។"               # Khmer
    "᠃᠉"              # Mongolian
    "៕"               # Khmer end-of-section
    "‼⁇⁈⁉"           # Unicode emphatic
)

# Compile once — sentence split looks for a terminator followed by whitespace
# OR end-of-string. The `(?<=...)` lookbehind keeps the terminator with the
# previous sentence (important for prosody).
_SENTENCE_SPLIT_RE = re.compile(
    rf"(?<=[{re.escape(SENTENCE_TERMINATORS)}])\s+"
)

# A "speakable" chunk has at least one word character in any script. This
# filters out fragments like ".", "—", or "" that the TTS would otherwise
# vocalize as an artifact glitch.
_HAS_WORD_RE = re.compile(r"\w", re.UNICODE)


def is_speakable(text: Optional[str]) -> bool:
    return bool(text) and _HAS_WORD_RE.search(text) is not None


# ---- English-specific normalizers (ASCII inputs) ------------------------

_NUMBER_RE = re.compile(r"(-?)(\d+(?:\.\d+)?)")
_MONEY_RE = re.compile(r"([$£€¥₹])(\d+(?:\.\d+)?)")
_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2})(?::(\d{2}))?(\s*(?:am|pm))?", re.IGNORECASE
)
_UNIT_MAP = {
    "km": "kilometers", "m": "meters", "cm": "centimeters", "mm": "millimeters",
    "kg": "kilograms", "g": "grams", "mg": "milligrams",
    "lb": "pounds", "oz": "ounces",
    "mph": "miles per hour", "kph": "kilometers per hour",
    "ms": "milliseconds", "s": "seconds", "min": "minutes", "h": "hours",
    "hz": "hertz", "khz": "kilohertz", "mhz": "megahertz", "ghz": "gigahertz",
    "gb": "gigabytes", "mb": "megabytes", "kb": "kilobytes", "tb": "terabytes",
    "%": "percent",
}
_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(" + "|".join(sorted(_UNIT_MAP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE = re.compile(r"\*(.+?)\*")
_MD_CODE_RE = re.compile(r"`[^`]+`")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # [text](url) → text
_SYMBOL_MAP = {"&": " and ", "%": " percent ", "+": " plus ", "=": " equals ", "@": " at "}


def _money_repl(m: "re.Match[str]") -> str:
    symbol, amount = m.group(1), m.group(2)
    currency = {
        "$": "dollar", "£": "pound", "€": "euro",
        "¥": "yen", "₹": "rupee",
    }.get(symbol, "dollar")
    parts = amount.split(".")
    whole = int(parts[0])
    result = (
        _INFLECT.number_to_words(whole)
        + " "
        + (currency + "s" if whole != 1 else currency)
    )
    if len(parts) == 2 and int(parts[1]) > 0:
        cents = int(parts[1])
        result += " and " + _INFLECT.number_to_words(cents) + " cents"
    return result


def _time_repl(m: "re.Match[str]") -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    ampm = m.group(4) or ""
    result = _INFLECT.number_to_words(hour)
    if minute > 0:
        if minute < 10:
            result += " oh " + _INFLECT.number_to_words(minute)
        else:
            result += " " + _INFLECT.number_to_words(minute)
    return result + " " + ampm.strip()


def _unit_repl(m: "re.Match[str]") -> str:
    num, unit = m.group(1), m.group(2).lower()
    word = _UNIT_MAP.get(unit, unit)
    return (
        _INFLECT.number_to_words(float(num) if "." in num else int(num))
        + " "
        + word
    )


def _num_repl(m: "re.Match[str]") -> str:
    sign, num = m.group(1), m.group(2)
    prefix = "negative " if sign == "-" else ""
    if "." in num:
        return prefix + _INFLECT.number_to_words(float(num))
    return prefix + _INFLECT.number_to_words(int(num))


def normalize_text(text: str, *, language: Optional[str] = None) -> str:
    """Strip markdown, expand numbers/money/units, collapse whitespace.

    Args:
        text: Raw text from LLM.
        language: ISO code. Currently English-only number expansion; for any
            other language we still strip markdown and URLs but leave digits.
    """

    text = _THINK_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_CODE_RE.sub("", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    text = _MD_HEADING_RE.sub("", text)
    text = _URL_RE.sub("", text)

    # English number/money/unit expansion. Other languages: leave digits —
    # most modern multilingual TTS pronounce them in the target language.
    if language is None or language.startswith("en"):
        text = _MONEY_RE.sub(_money_repl, text)
        text = _TIME_RE.sub(_time_repl, text)
        text = _UNIT_RE.sub(_unit_repl, text)
        text = _NUMBER_RE.sub(_num_repl, text)
        for sym, word in _SYMBOL_MAP.items():
            text = text.replace(sym, word)

    # Collapse whitespace before punctuation (URL/markdown stripping can leave
    # " . " behind). Use a unicode-aware punctuation class.
    text = re.sub(
        rf"\s+([{re.escape(SENTENCE_TERMINATORS)},;:])", r"\1", text
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str, *, max_words: int = 25) -> list[str]:
    """Split into sentences on strong terminators across many scripts.

    Long sentences (>max_words) are further split on weak boundaries (`,;:`)
    so the TTS doesn't get a 60-word run-on, which causes attention drift and
    audible mispronunciation in the tail.
    """

    parts = _SENTENCE_SPLIT_RE.split(text)
    out: list[str] = []
    for p in parts:
        if not is_speakable(p):
            continue
        if len(p.split()) > max_words:
            # Fall back to weak boundaries inside long runs.
            sub = re.split(r"(?<=[,;:])\s+", p)
            out.extend(s for s in sub if is_speakable(s))
        else:
            out.append(p)
    return out


# pocket-tts warns "Chunk has N tokens (max 50)" and can skip words past that.
# Its splitter does not treat । as a sentence end, and the comma in "word, word"
# is a different sentencepiece id from the comma it searches for. A 25-word
# window of this Hindi is already over 50 tokens, so chunks stay shorter.
POCKET_MAX_TOKENS = 50
POCKET_SAFE_MAX_WORDS = 20

# U+0964 DEVANAGARI DANDA, U+0965 DEVANAGARI DOUBLE DANDA.
# Both are already in SENTENCE_TERMINATORS; this is the short-text trigger.
_INDIC_FULL_STOPS = "\u0964\u0965"
_INDIC_SPLIT_RE = re.compile(rf"(?<=[{_INDIC_FULL_STOPS}])\s+")


def _group_words(text: str, max_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text] if is_speakable(text) else []
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
    ]


def split_for_generation(
    text: str, *, max_words: int = POCKET_SAFE_MAX_WORDS
) -> list[str]:
    """Split one spoken span so Pocket is not handed a 50-token overrun.

    Short spans are returned unchanged, spaces included, so a pause split
    that left ``"Hello. "`` stays ``"Hello. "``. A danda with more text
    after it is a boundary even when the line is short. Anything still
    over ``max_words`` is split on ``, ; :`` and then on spaces.
    """

    if not is_speakable(text):
        return []
    long = len(text.split()) > max_words
    if not long and _INDIC_SPLIT_RE.search(text) is None:
        return [text]

    out: list[str] = []
    for piece in split_sentences(text, max_words=max_words):
        if len(piece.split()) > max_words:
            out.extend(_group_words(piece, max_words))
        elif is_speakable(piece):
            out.append(piece)
    return out or [text]


def _bisect_to_tokens(
    text: str,
    token_len: Callable[[str], int],
    max_tokens: int,
) -> list[str]:
    """Halve on spaces, then on characters, until each piece fits."""

    if token_len(text) <= max_tokens:
        return [text]
    words = text.split()
    if len(words) >= 2:
        mid = len(words) // 2
        parts = [" ".join(words[:mid]), " ".join(words[mid:])]
    elif len(text) >= 2:
        mid = len(text) // 2
        parts = [text[:mid], text[mid:]]
    else:
        return [text]

    out: list[str] = []
    for part in parts:
        if not part.strip():
            continue
        if token_len(part) <= max_tokens:
            out.append(part)
        else:
            out.extend(_bisect_to_tokens(part, token_len, max_tokens))
    return out or [text]


def fit_pocket_text(
    text: str,
    token_len: Optional[Callable[[str], int]] = None,
    *,
    max_tokens: int = POCKET_MAX_TOKENS,
    max_words: int = POCKET_SAFE_MAX_WORDS,
) -> list[str]:
    """Punctuation split, then the loaded Pocket tokenizer if a piece is still over.

    ``token_len`` is optional so tests can run without model weights. When
    the engine passes the tokenizer it already loaded, a piece the word cap
    left too long is cut until it is at or under ``max_tokens``.
    """

    pieces = split_for_generation(text, max_words=max_words)
    if token_len is None:
        return pieces
    fitted: list[str] = []
    for piece in pieces:
        if token_len(piece) <= max_tokens:
            fitted.append(piece)
        else:
            fitted.extend(_bisect_to_tokens(piece, token_len, max_tokens))
    return fitted
