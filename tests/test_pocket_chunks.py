"""Chunking before Pocket generate. No model weights."""

from __future__ import annotations

import unittest
from pathlib import Path

from livekit_streaming_tts.normalize import (
    POCKET_MAX_TOKENS,
    POCKET_SAFE_MAX_WORDS,
    SENTENCE_TERMINATORS,
    fit_pocket_text,
    split_for_generation,
)
from livekit_streaming_tts.pauses import speech_plan

# The reported line. It has four ASCII commas and a trailing danda.
HINDI = (
    "ज़िंदगी में हर मोड़ पर हमें कुछ नया सीखने को मिलता है, "
    "कभी खुशियाँ मिलती हैं, तो कभी मुश्किलें आती हैं, "
    "लेकिन जो इंसान हर परिस्थिति में मुस्कुराना सीख लेता है, "
    "वही सच में ज़िंदगी जीना जानता है।"
)


def _multilingual_token_len():
    """SentencePiece counter for the multilingual Pocket model, if it is cached.

    Loading ``tokenizer.model`` does not load ``model.safetensors``. The
    English ``tokenizer.json`` counts this line as far more than the 83
    tokens in the warning, so it is not the budget that fired.
    """

    try:
        import sentencepiece
    except ImportError:
        return None
    root = Path.home() / ".cache" / "huggingface" / "hub"
    paths = sorted(
        root.glob(
            "models--Thanushttk--pocket-tts-multilingual/snapshots/*/tokenizer.model"
        )
    )
    if not paths:
        return None
    processor = sentencepiece.SentencePieceProcessor(model_file=str(paths[-1]))

    def token_len(text: str) -> int:
        return len(processor.encode(text, out_type=int))

    return token_len


class HindiChunkTests(unittest.TestCase):
    def test_sentence_terminators_include_devanagari_stops(self) -> None:
        self.assertIn("\u0964", SENTENCE_TERMINATORS)  # ।
        self.assertIn("\u0965", SENTENCE_TERMINATORS)  # ॥

    def test_hindi_sentence_becomes_several_pieces(self) -> None:
        plan = speech_plan(HINDI, pauses=False, normalize=True, language="hi")
        spoken = [part for part in plan if isinstance(part, str)]

        self.assertGreater(len(spoken), 1)
        self.assertTrue(all(part != HINDI and part.strip() != HINDI for part in spoken))
        self.assertNotIn(HINDI, spoken)
        # The trailing danda stays on a piece; it is not dropped.
        self.assertTrue(any(part.rstrip().endswith("\u0964") for part in spoken))

        # । between two short clauses is a boundary on its own.
        danda = speech_plan(
            "पहला वाक्य। दूसरा वाक्य।",
            pauses=False,
            normalize=False,
        )
        self.assertEqual(danda, ["पहला वाक्य।", "दूसरा वाक्य।"])
        double = split_for_generation("एक॥ दो॥")
        self.assertEqual(double, ["एक॥", "दो॥"])

        token_len = _multilingual_token_len()
        if token_len is None:
            return
        self.assertGreater(token_len(HINDI), POCKET_MAX_TOKENS)
        for part in spoken:
            self.assertLessEqual(token_len(part), POCKET_MAX_TOKENS, part)
        # Pocket appends '.' because । is not one of its terminal marks.
        self.assertEqual(token_len(HINDI + "."), token_len(HINDI) + 1)
        self.assertGreater(token_len(HINDI + "."), POCKET_MAX_TOKENS)

    def test_pause_marker_is_silence_and_not_spoken(self) -> None:
        text = HINDI.replace(", ", " [pause:1.5] ", 1)
        plan = speech_plan(text, pauses=True, normalize=True, language="hi")

        self.assertIn(1.5, plan)
        spoken = [part for part in plan if isinstance(part, str)]
        self.assertGreater(len(spoken), 1)
        self.assertTrue(all("[pause" not in part for part in spoken))
        self.assertTrue(all("1.5" not in part for part in spoken))

        # A long span that still contains the marker (pauses off, or a
        # splitter that sees the raw string) must not cut inside it.
        raw = ("शब्द " * 20) + "[pause:1.5] " + ("शब्द " * 20)
        pieces = split_for_generation(raw.strip())
        self.assertTrue(any("[pause:1.5]" in part for part in pieces))
        self.assertTrue(all(len(part.split()) <= POCKET_SAFE_MAX_WORDS for part in pieces))

    def test_token_budget_bisects_when_the_word_cap_is_not_enough(self) -> None:
        text = "one two three four"

        def token_len(piece: str) -> int:
            return len(piece.split()) * 10

        pieces = fit_pocket_text(text, token_len, max_tokens=25, max_words=20)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(token_len(part) <= 25 for part in pieces))
        self.assertEqual(" ".join(pieces), text)


if __name__ == "__main__":
    unittest.main()
