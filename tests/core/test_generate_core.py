# SPDX-FileCopyrightText: 2016-2026 PyThaiNLP Project
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: Apache-2.0
"""
Characterization tests for pythainlp.generate.core gen_sentence methods.

Golden cases were recorded from the code before the complexity refactor.
The model is a small in-memory fake, so no corpus is loaded or downloaded.
"""

from __future__ import annotations

import random
import unittest
from typing import Any

from pythainlp.generate.core import Bigram, Trigram

BIGRAMS: dict[tuple[str, str], int] = {
    ("a", "b"): 4,
    ("b", "c"): 3,
    ("b", "d"): 1,
    ("c", "a"): 2,
    ("a", "c"): 2,
    ("d", "a"): 1,
    ("z", "y"): 0,
    ("c", "c"): 1,
}
TRIGRAMS: dict[tuple[str, str, str], int] = {
    ("a", "b", "c"): 2,
    ("a", "b", "d"): 1,
    ("b", "c", "a"): 2,
    ("c", "a", "b"): 1,
    ("b", "d", "a"): 1,
    ("a", "c", "a"): 1,
    ("d", "a", "b"): 1,
    ("c", "a", "c"): 1,
    ("z", "y", "a"): 1,
    ("q", "r", "s"): 1,
    ("c", "c", "c"): 1,
    ("a", "c", "c"): 1,
    ("c", "c", "a"): 1,
}


def make_model() -> Trigram:
    """Build a Trigram without loading corpora."""
    model = Trigram.__new__(Trigram)
    model.uni = {"a": 5, "b": 4, "c": 3, "d": 1}
    model.bi = dict(BIGRAMS)
    model.ti = dict(TRIGRAMS)
    model.uni_keys = list(model.uni)
    model.bi_keys = list(model.bi)
    model.ti_keys = list(model.ti)
    model.words = [i[-1] for i in model.bi_keys]
    return model


# (start_seq, N, prob, output_str, duplicate, random seed, outcome)
# The outcome is ["ok", result] or ["exc", exception name, message].
GOLDEN: list[list[Any]] = [
    ["", 0, 0.001, True, False, 30, ["ok", "ac"]],
    ["", 0, 0.001, False, False, 75, ["ok", ["c"]]],
    ["", 1, 0.001, True, False, 60, ["ok", "ac"]],
    ["", 1, 2.0, True, False, 77, ["ok", "ac"]],
    ["", 3, 2.0, True, False, 24, ["ok", "zy"]],
    ["", 1, 0.001, True, True, 94, ["ok", "bda"]],
    [("a", "b"), 0, 0.001, True, False, 54, ["ok", "ab"]],
    [("q", "r"), 1, 0.001, True, False, 43, ["exc", "KeyError", "('q', 'r')"]],
    ["a", 0, 0.001, True, False, 69, ["ok", "a"]],
    [("c", "a"), 3, 0.001, True, True, 17, ["ok", "cacca"]],
    [None, 6, 0.5, False, False, 48, ["ok", ["d", "a", "b", "c"]]],
    ["", 3, 0.001, False, False, 33, ["ok", ["b", "d", "a", "c"]]],
    [(), 6, 0.5, False, False, 7, ["ok", ["d", "a", "b", "c"]]],
    [None, 6, 0.001, True, True, 84, ["ok", "acacacac"]],
    [("a", "b"), 6, 0.001, True, True, 91, ["ok", "abcabdab"]],
    [("c", "c"), 6, 0.001, False, False, 48, ["ok", ["c", "a", "b", "d"]]],
    [(), 6, 0.001, True, True, 53, ["ok", "caccacab"]],
    [
        "",
        6,
        0.001,
        False,
        True,
        60,
        ["ok", ["a", "c", "c", "c", "a", "b", "d", "a"]],
    ],
    ["", 3, 0.001, True, True, 97, ["ok", "caccc"]],
    [("b", "c"), 6, 0.001, True, False, 77, ["ok", "bca"]],
    [("c", "a"), 3, 0.001, True, False, 79, ["ok", "cabd"]],
    [(), 3, 0.5, True, True, 76, ["ok", "dabca"]],
    [("b", "c"), 6, 0.001, False, False, 78, ["ok", ["b", "c", "a", "d"]]],
    [("c", "a"), 6, 0.001, True, True, 45, ["ok", "caccacac"]],
    [
        None,
        6,
        0.001,
        False,
        True,
        48,
        ["ok", ["d", "a", "b", "d", "a", "b", "c", "a"]],
    ],
    [
        ("c", "a"),
        6,
        0.001,
        False,
        True,
        86,
        ["ok", ["c", "a", "b", "d", "a", "b", "d", "a"]],
    ],
    [
        (),
        6,
        0.001,
        False,
        True,
        10,
        ["ok", ["a", "b", "d", "a", "b", "c", "a", "c"]],
    ],
    ["", 3, 0.001, True, False, 60, ["ok", "acb"]],
    ["", 6, 0.5, True, True, 76, ["ok", "dabcacac"]],
    [None, 3, 0.001, False, True, 5, ["ok", ["a", "c", "c", "c", "a"]]],
    ["", 6, 0.001, True, True, 34, ["ok", "dabcaccc"]],
    ["", 6, 0.001, True, False, 60, ["ok", "acb"]],
    [
        ("b", "c"),
        6,
        0.001,
        False,
        True,
        19,
        ["ok", ["b", "c", "a", "b", "c", "a", "c", "c"]],
    ],
    [None, 3, 0.001, False, False, 33, ["ok", ["b", "d", "a", "c"]]],
    [(), 3, 0.001, False, False, 67, ["ok", ["b", "c", "a"]]],
    [("b", "c"), 3, 0.001, False, False, 54, ["ok", ["b", "c", "a"]]],
    [(), 6, 0.001, True, False, 95, ["ok", "bdac"]],
    [
        ("q", "r"),
        1,
        0.001,
        False,
        False,
        94,
        ["exc", "KeyError", "('q', 'r')"],
    ],
    [("q", "r"), 1, 0.5, True, False, 83, ["exc", "KeyError", "('q', 'r')"]],
    [("q", "r"), 1, 0.5, False, False, 27, ["exc", "KeyError", "('q', 'r')"]],
]


def make_bigram_model() -> Bigram:
    """Build a Bigram with two equally likely successors of "x"."""
    model = Bigram.__new__(Bigram)
    model.uni = {"x": 4, "p": 2, "q": 2}
    model.bi = {("x", "p"): 2, ("x", "q"): 2}
    model.uni_keys = list(model.uni)
    model.bi_keys = list(model.bi)
    model.words = [i[-1] for i in model.bi_keys]
    return model


class TrigramGenSentenceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # The tests seed the global random generator; restore its state.
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        self.model = make_model()

    def test_golden_cases(self) -> None:
        for start, n, prob, output_str, duplicate, seed, expected in GOLDEN:
            with self.subTest(
                start=start, n=n, prob=prob, str=output_str, dup=duplicate
            ):
                random.seed(seed)
                if expected[0] == "exc":
                    with self.assertRaises(KeyError) as ctx:
                        self.model.gen_sentence(
                            start, n, prob, output_str, duplicate
                        )
                    self.assertEqual(str(ctx.exception), expected[2])
                    continue
                got = self.model.gen_sentence(
                    start, n, prob, output_str, duplicate
                )
                self.assertEqual(got, expected[1])

    def test_prob_handles_zero_denominator(self) -> None:
        self.assertEqual(self.model.prob("z", "y", "a"), 0.0)

    def test_stops_when_no_candidate_passes_threshold(self) -> None:
        random.seed(0)
        self.assertEqual(
            self.model.gen_sentence(("a", "b"), 5, 2.0, False), ["a", "b"]
        )

    def test_output_is_a_string_by_default(self) -> None:
        random.seed(0)
        result = self.model.gen_sentence(("a", "b"), 0)
        self.assertEqual(result, "ab")

    def test_equal_probabilities_pick_any_candidate(self) -> None:
        outputs = set()
        for seed in range(20):
            random.seed(seed)
            outputs.add(
                tuple(self.model.gen_sentence(("a", "c"), 2, 0.0001, False))
            )
        self.assertGreater(len(outputs), 1)

    def test_duplicate_keeps_repeated_words(self) -> None:
        # Each step adds one word; with duplicate=True repeats stay.
        for seed in range(5):
            with self.subTest(seed=seed):
                random.seed(seed)
                words = self.model.gen_sentence(
                    ("c", "c"), 4, 0.0001, False, True
                )
                self.assertEqual(len(words), 6)
                self.assertEqual(words[:2], ["c", "c"])

    def test_start_item_of_other_type_is_skipped(self) -> None:
        self.assertEqual(
            self.model.gen_sentence(["a", "b"], 1, 0.0001, False),  # type: ignore[arg-type]
            [],
        )
        self.assertEqual(
            self.model.gen_sentence((1, 2), 0, 0.0001, False),  # type: ignore[arg-type]
            [1, 2],
        )

    def test_unknown_bigram_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            self.model.gen_sentence(("q", "r"), 1)

    def test_string_start_sequence_matches_no_trigram(self) -> None:
        self.assertEqual(self.model.gen_sentence("a", 3, 0.0001, False), ["a"])


class BigramGenSentenceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        self.model = make_bigram_model()

    def test_equal_probabilities_pick_any_candidate(self) -> None:
        outputs = set()
        for seed in range(20):
            random.seed(seed)
            outputs.add(tuple(self.model.gen_sentence("x", 1, 0.0001, False)))
        self.assertEqual(outputs, {("x", "p"), ("x", "q")})


if __name__ == "__main__":
    unittest.main()
