"""Pause parsing. No model and no GPU."""

from __future__ import annotations

import unittest

import numpy as np

from livekit_streaming_tts.audio import to_int16_pcm
from livekit_streaming_tts.pauses import (
    DEFAULT_PAUSE_SECONDS,
    MAX_PAUSE_SECONDS,
    silence_samples,
    speech_plan,
    split_on_pauses,
)


def _speakable(text) -> bool:
    return bool(text and str(text).strip())


class SplitTests(unittest.TestCase):
    def test_bare_pause_defaults_to_half_a_second(self) -> None:
        self.assertEqual(
            split_on_pauses("Ready. [pause] Go."),
            ["Ready. ", DEFAULT_PAUSE_SECONDS, " Go."],
        )
        self.assertEqual(DEFAULT_PAUSE_SECONDS, 0.5)

    def test_explicit_duration(self) -> None:
        self.assertEqual(
            split_on_pauses("Ready. [pause:1.5] Go."),
            ["Ready. ", 1.5, " Go."],
        )

    def test_shorthand_ignores_case_and_spaces(self) -> None:
        self.assertEqual(split_on_pauses("A [PAUSE: 2] B"), ["A ", 2.0, " B"])
        self.assertEqual(split_on_pauses("A [ pause ] B"), ["A ", 0.5, " B"])
        self.assertEqual(split_on_pauses("A [pause:.25] B"), ["A ", 0.25, " B"])

    def test_malformed_shorthand_stays_in_the_text(self) -> None:
        self.assertEqual(split_on_pauses("A [pause:] B"), ["A [pause:] B"])
        self.assertEqual(split_on_pauses("A [paws:1] B"), ["A [paws:1] B"])

    def test_break_time_units(self) -> None:
        self.assertEqual(
            split_on_pauses('Ready. <break time="1.5s"/> Go.'),
            ["Ready. ", 1.5, " Go."],
        )
        self.assertEqual(
            split_on_pauses('Ready. <break time="500ms"/> Go.'),
            ["Ready. ", 0.5, " Go."],
        )
        self.assertEqual(
            split_on_pauses('A <break time="1500ms"/> B'),
            ["A ", 1.5, " B"],
        )

    def test_bare_break_and_strength(self) -> None:
        self.assertEqual(split_on_pauses("A <break/> B"), ["A ", 0.5, " B"])
        self.assertEqual(
            split_on_pauses('A <break strength="weak"/> B'),
            ["A ", 0.25, " B"],
        )

    def test_time_wins_over_strength(self) -> None:
        self.assertEqual(
            split_on_pauses('A <break time="2s" strength="weak"/> B'),
            ["A ", 2.0, " B"],
        )

    def test_duration_clamps_at_ten_seconds(self) -> None:
        self.assertEqual(split_on_pauses("A [pause:100] B"), ["A ", MAX_PAUSE_SECONDS, " B"])
        self.assertEqual(MAX_PAUSE_SECONDS, 10.0)
        self.assertEqual(split_on_pauses("A [pause:0] B"), ["A ", 0.0, " B"])

    def test_unknown_strength_and_other_tags_raise(self) -> None:
        with self.assertRaises(ValueError):
            split_on_pauses('A <break strength="enormous"/> B')
        with self.assertRaises(ValueError) as caught:
            split_on_pauses("<emphasis>a</emphasis> <prosody>b</prosody>")
        message = str(caught.exception)
        self.assertIn("<emphasis>", message)
        self.assertIn("<prosody>", message)

    def test_speak_wrapper_is_removed(self) -> None:
        text = '<speak>Ready. <break time="1s"/> Go.</speak>'
        self.assertEqual(split_on_pauses(text), ["Ready. ", 1.0, " Go."])

    def test_comparison_signs_are_not_tags(self) -> None:
        self.assertEqual(split_on_pauses("5 < 6 and 7 > 3."), ["5 < 6 and 7 > 3."])

    def test_markers_at_the_edges_and_adjacent(self) -> None:
        self.assertEqual(split_on_pauses("[pause:1] hi [pause:2]"), [1.0, " hi ", 2.0])
        self.assertEqual(split_on_pauses("[pause:1]"), [1.0])
        self.assertEqual(
            split_on_pauses('Hi [pause:1]<break time="2s"/> there'),
            ["Hi ", 1.0, 2.0, " there"],
        )


class PlanTests(unittest.TestCase):
    def test_normalize_does_not_see_pause_marker(self) -> None:
        seen: list[str] = []

        def normalize(text: str, language: str | None = None) -> str:
            seen.append(text)
            if "[pause" in text or "<break" in text or "1.5" in text:
                raise AssertionError(f"normalizer saw a marker: {text!r}")
            return text

        plan = speech_plan(
            "Hello. [pause:1.5] World.",
            pauses=True,
            normalize=True,
            normalize_fn=normalize,
            speakable=_speakable,
        )
        self.assertEqual(plan, ["Hello. ", 1.5, " World."])
        self.assertEqual(seen, ["Hello. ", " World."])
        spoken = [part for part in plan if isinstance(part, str)]
        self.assertTrue(all("pause" not in part.lower() for part in spoken))

    def test_break_is_split_before_normalize(self) -> None:
        seen: list[str] = []

        def normalize(text: str, language: str | None = None) -> str:
            seen.append(text)
            return text

        plan = speech_plan(
            'Hello. <break time="1.5s"/> World.',
            pauses=True,
            normalize=True,
            normalize_fn=normalize,
            speakable=_speakable,
        )
        self.assertEqual(plan, ["Hello. ", 1.5, " World."])
        self.assertTrue(all("break" not in part.lower() for part in seen))

    def test_tail_after_a_sentence_split_still_pauses(self) -> None:
        """The server sentence splitter keeps a marker on the following chunk."""

        plan = speech_plan(
            "[pause:0.5] World.",
            pauses=True,
            normalize=False,
            normalize_fn=lambda text, language=None: text,
            speakable=_speakable,
        )
        self.assertEqual(plan, [0.5, " World."])

    def test_pauses_off_does_not_invent_silence(self) -> None:
        def normalize(text: str, language: str | None = None) -> str:
            raise AssertionError("normalize should not run")

        plan = speech_plan(
            "Hello. [pause:1.5] World.",
            pauses=False,
            normalize=False,
            normalize_fn=normalize,
            speakable=_speakable,
        )
        self.assertEqual(plan, ["Hello. [pause:1.5] World."])
        self.assertTrue(all(isinstance(part, str) for part in plan))

    def test_pauses_off_still_hands_the_marker_to_normalize(self) -> None:
        seen: list[str] = []

        def normalize(text: str, language: str | None = None) -> str:
            seen.append(text)
            return text

        plan = speech_plan(
            "Hello. [pause:1.5] World.",
            pauses=False,
            normalize=True,
            normalize_fn=normalize,
            speakable=_speakable,
        )
        self.assertEqual(seen, ["Hello. [pause:1.5] World."])
        self.assertEqual(plan, seen)
        self.assertNotIsInstance(plan[0], float)

    def test_unsupported_tag_raises_before_normalize(self) -> None:
        def normalize(text: str, language: str | None = None) -> str:
            raise AssertionError("normalize should not see an unsupported tag")

        with self.assertRaises(ValueError):
            speech_plan(
                '<prosody rate="slow">Slow</prosody>',
                pauses=True,
                normalize=True,
                normalize_fn=normalize,
                speakable=_speakable,
            )

    def test_half_second_is_exact_zeros_at_the_model_rate(self) -> None:
        samples = silence_samples(0.5, 24000)
        self.assertEqual(samples.shape, (12000,))
        self.assertEqual(samples.dtype, np.float32)
        self.assertTrue(np.all(samples == 0))
        pcm = to_int16_pcm(samples)
        self.assertEqual(pcm, b"\x00" * (12000 * 2))
        self.assertEqual(silence_samples(0, 24000).size, 0)


if __name__ == "__main__":
    unittest.main()
