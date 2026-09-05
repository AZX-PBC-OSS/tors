"""Correctness gate for ``tors.normalize``, ported from cennan's
``tests/unit/test_normalize.py`` behavioral cases plus an independent, self-contained
differential proof against a pure-Python reimplementation of the same pipeline.

cennan's ``normalize_text`` chunks its two ``re.sub`` passes into ~1MiB seam-safe pieces
purely to bound GIL-hold time; ``tors.normalize`` does the same transform as a single
native Rust pass under ``py.detach`` instead, so cennan's seam/chunking-specific tests
(``TestSeamSafety``, ``TestWhitespaceRunLongerThanTheWindow``, ``TestSeamRuleInvariant``,
``TestSubChunkContract``, ...) do not apply here and are intentionally not ported.
"""

from __future__ import annotations

import re
import string
import unicodedata

import pytest
from hypothesis import given, settings, strategies as st

from tors import normalize

_BLANK_RUN = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")


def reference_normalize(text: str) -> str:
    """Pure-Python reimplementation of cennan's ``normalize_text``, unchunked. The
    self-contained correctness oracle: equivalence with this holds independent of cennan
    ever being checked out."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _TRAILING_WS.sub("\n", normalized)
    normalized = _BLANK_RUN.sub("\n\n", normalized)
    return normalized.strip()


class TestBehavioralCases:
    """Ported from cennan's TestNormalizeTextEndToEnd and general normalize_text usage."""

    def test_empty_string(self) -> None:
        assert normalize("") == ""

    def test_all_whitespace_collapses_to_empty(self) -> None:
        assert normalize("   \t\n\n\n   ") == ""

    def test_pins_the_pattern_semantics_on_the_boundary_cases(self) -> None:
        # A tab before a newline is trimmed; a THREE-newline run collapses to two.
        assert normalize("a\t\nb\n\n\nc") == "a\nb\n\nc"

    def test_composes_decomposed_sequences_before_collapsing(self) -> None:
        # "cafe" + combining acute accent (U+0301), decomposed, must compose to
        # "café" before the trailing-space trim and blank-run collapse run.
        assert normalize("cafe\u0301 \n\n\n\nwater") == "caf\u00e9\n\nwater"

    def test_crlf_and_lone_cr_fold_to_lf(self) -> None:
        assert normalize("a\r\nb\rc\nd") == "a\nb\nc\nd"

    def test_mixed_crlf_cr_lf_with_blank_runs(self) -> None:
        text = "line1\r\n\r\n\r\nline2\rline3\n\n\nline4"
        assert normalize(text) == reference_normalize(text)

    def test_trailing_whitespace_at_end_with_no_final_newline(self) -> None:
        assert normalize("hello   ") == "hello"
        assert normalize("hello\t\t") == "hello"

    def test_leading_whitespace_is_stripped(self) -> None:
        assert normalize("   hello") == "hello"

    def test_multi_run_blank_collapses(self) -> None:
        text = "a\n\n\n\n\n\nb\n\n\nc\n\nd"
        assert normalize(text) == reference_normalize(text)

    def test_unicode_nfc_affecting_characters(self) -> None:
        # "e" + combining acute vs. precomposed "é" must normalize identically.
        decomposed = "cafe\u0301"
        precomposed = "caf\u00e9"
        assert normalize(decomposed) == normalize(precomposed) == "caf\u00e9"

    def test_mid_line_whitespace_is_preserved(self) -> None:
        assert normalize("a  b\tc   d") == "a  b\tc   d"

    def test_blank_run_of_exactly_two_is_untouched(self) -> None:
        assert normalize("a\n\nb") == "a\n\nb"

    def test_single_newline_is_untouched(self) -> None:
        assert normalize("a\nb") == "a\nb"


class TestExhaustiveSmallAlphabet:
    """Exhaustive coverage of every short string over the alphabet whose characters can
    trigger each stage of the pipeline: space, tab, \\n, \\r, an ordinary letter, and a
    combining-mark pair (to exercise NFC)."""

    _ALPHABET = [" ", "\t", "\n", "\r", "a", "é"]

    @pytest.mark.parametrize("length", [0, 1, 2, 3])
    def test_all_strings_up_to_length(self, length: int) -> None:
        from itertools import product

        for parts in product(self._ALPHABET, repeat=length):
            text = "".join(parts)
            assert normalize(text) == reference_normalize(text), repr(text)


class TestRandomizedProperty:
    @given(
        st.text(
            alphabet=st.sampled_from(" \t\n\r\u00a0\u00e9a\u00e9.\u3000"),
            max_size=200,
        )
    )
    @settings(max_examples=300)
    def test_matches_reference_pipeline(self, text: str) -> None:
        assert normalize(text) == reference_normalize(text)

    @given(st.text(max_size=200))
    @settings(max_examples=300)
    def test_matches_reference_pipeline_over_arbitrary_unicode(self, text: str) -> None:
        assert normalize(text) == reference_normalize(text)


class TestSeededRandomGenerator:
    """A seeded, dependency-free random generator as a second randomized methodology,
    independent of hypothesis being installed/available."""

    def test_matches_reference_over_seeded_random_corpora(self) -> None:
        import random

        alphabet = list(" \t\n\r" + string.ascii_letters + "\u00e9  \u3000")
        rng = random.Random(1234567)
        for _ in range(500):
            length = rng.randint(0, 120)
            text = "".join(rng.choice(alphabet) for _ in range(length))
            assert normalize(text) == reference_normalize(text), repr(text)
