"""Correctness gate for ``tors.normalize``: behavioral cases from the original
specification plus an independent, self-contained differential proof against a
pure-Python reimplementation of the same pipeline.

The pipeline's original pure-Python spelling chunks its two ``re.sub`` passes into
~1MiB seam-safe pieces purely to bound GIL-hold time; ``tors.normalize`` does the same
transform as a single native Rust pass under ``py.detach`` instead, so that spelling's
seam/chunking-specific tests do not apply here and are intentionally not ported.
"""

from __future__ import annotations

import itertools
import random
import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reference import (  # noqa: I001 -- the shared oracle module (tests/reference.py)
    _COMBINING_ACUTE,
    _E_ACUTE_PRECOMPOSED,
    _IDEOGRAPHIC_SPACE,
    _NBSP,
    pathological_text,
    reference_normalize,
)
from tors import normalize


class TestBehavioralCases:
    """Behavioral cases from the original specification's end-to-end normalize usage."""

    def test_empty_string(self) -> None:
        assert normalize("") == ""

    def test_all_whitespace_collapses_to_empty(self) -> None:
        assert normalize("   \t\n\n\n   ") == ""

    def test_all_whitespace_variants_collapse_to_empty(self) -> None:
        for text in ["\n", "\n\n", "\n\n\n", " ", "\t", "\r", "\r\n", " \t\r\n" * 10]:
            assert normalize(text) == "", repr(text)

    def test_pins_the_pattern_semantics_on_the_boundary_cases(self) -> None:
        # A tab before a newline is trimmed; a three-newline run collapses to two.
        assert normalize("a\t\nb\n\n\nc") == "a\nb\n\nc"

    def test_composes_decomposed_sequences_before_collapsing(self) -> None:
        # "cafe" + combining acute accent (U+0301), decomposed, must compose to
        # "café" before the trailing-space trim and blank-run collapse run.
        text = "cafe" + _COMBINING_ACUTE + " \n\n\n\nwater"
        assert normalize(text) == "caf" + _E_ACUTE_PRECOMPOSED + "\n\nwater"

    def test_entirely_decomposed_string(self) -> None:
        # A string built only from decomposed base+combining-mark pairs, with no ordinary
        # ASCII interspersed, must still fully compose under NFC.
        decomposed = ("e" + _COMBINING_ACUTE) * 20
        composed = _E_ACUTE_PRECOMPOSED * 20
        assert normalize(decomposed) == composed
        assert normalize(decomposed) == reference_normalize(decomposed)

    def test_crlf_and_lone_cr_fold_to_lf(self) -> None:
        assert normalize("a\r\nb\rc\nd") == "a\nb\nc\nd"

    def test_mixed_crlf_cr_lf_with_blank_runs(self) -> None:
        text = "line1\r\n\r\n\r\nline2\rline3\n\n\nline4"
        assert normalize(text) == reference_normalize(text)

    def test_mixed_crlf_cr_lf_every_line_ending_kind_in_one_string(self) -> None:
        # \r\n, bare \r, and bare \n all appear, several times each, interleaved with
        # blank runs that only become visible once every ending is folded to \n.
        text = "a\r\nb\rc\nd\r\ne\r\rf\n\ng\r\n\r\n\r\nh"
        assert normalize(text) == reference_normalize(text)

    def test_trailing_whitespace_at_end_with_no_final_newline(self) -> None:
        assert normalize("hello   ") == "hello"
        assert normalize("hello\t\t") == "hello"

    def test_leading_whitespace_is_stripped(self) -> None:
        assert normalize("   hello") == "hello"

    def test_multi_run_blank_collapses(self) -> None:
        text = "a\n\n\n\n\n\nb\n\n\nc\n\nd"
        assert normalize(text) == reference_normalize(text)

    def test_multiple_consecutive_blank_run_collapses_stay_independent(self) -> None:
        # Five separate 3+-newline runs at different points in the same string: each must
        # collapse to exactly two newlines on its own, not merge with its neighbors.
        text = "\n\n\n".join(f"para{i}" for i in range(6))
        expected = "\n\n".join(f"para{i}" for i in range(6))
        assert normalize(text) == expected == reference_normalize(text)

    def test_unicode_nfc_affecting_characters(self) -> None:
        # "e" + combining acute vs. precomposed "é" must normalize identically.
        decomposed = "cafe" + _COMBINING_ACUTE
        precomposed = "caf" + _E_ACUTE_PRECOMPOSED
        expected = "caf" + _E_ACUTE_PRECOMPOSED
        assert normalize(decomposed) == normalize(precomposed) == expected

    def test_mid_line_whitespace_is_preserved(self) -> None:
        assert normalize("a  b\tc   d") == "a  b\tc   d"

    def test_blank_run_of_exactly_two_is_untouched(self) -> None:
        assert normalize("a\n\nb") == "a\n\nb"

    def test_single_newline_is_untouched(self) -> None:
        assert normalize("a\nb") == "a\nb"

    def test_very_long_single_whitespace_run_before_newline(self) -> None:
        text = "a" + (" \t" * 50_000) + "\nb"
        assert normalize(text) == "a\nb" == reference_normalize(text)

    def test_very_long_single_newline_run(self) -> None:
        text = "a" + ("\n" * 100_000) + "b"
        assert normalize(text) == "a\n\nb" == reference_normalize(text)

    def test_very_long_whitespace_run_at_end_of_string(self) -> None:
        # No following non-whitespace character at all: the whole run is trailing
        # whitespace, dropped entirely by the final strip.
        text = "leading text" + (" \t\n" * 40_000)
        assert normalize(text) == "leading text" == reference_normalize(text)

    def test_very_long_whitespace_run_from_start_of_string(self) -> None:
        text = (" \t\n" * 40_000) + "trailing text"
        assert normalize(text) == "trailing text" == reference_normalize(text)

    def test_long_mixed_corpus_matches_reference(self) -> None:
        # A large (~500K char) corpus mixing prose, CRLF/CR/LF, trailing whitespace, and
        # blank runs; no chunking exists in tors, but this guards against any
        # length-dependent bug (e.g. an off-by-one that only a long buffer exposes).
        rng = random.Random(42)
        units = []
        for i in range(20_000):
            m = i % 7
            if m == 0:
                units.append(f"line {i} with trailing spaces   \n")
            elif m == 1:
                units.append("\n" * rng.randint(3, 6))
            elif m == 2:
                units.append(f"prose {i}\r\n")
            elif m == 3:
                units.append(f"lone cr {i}\r")
            elif m == 4:
                units.append(" \t" * rng.randint(1, 5) + "\n")
            elif m == 5:
                units.append(f"caf{_COMBINING_ACUTE}e naive prose {i}\n")
            else:
                units.append(f"plain line {i}\n")
        text = "".join(units)
        assert len(text) > 300_000
        assert normalize(text) == reference_normalize(text)


class TestExhaustiveSmallAlphabet:
    """Exhaustive coverage of every short string over the alphabet whose characters can
    trigger each stage of the pipeline: space, tab, \\n, \\r, an ordinary letter, an NFC
    base character, and its combining mark (so composition is exercised not just as a
    single pre-picked character but built up from combinatorial placement)."""

    _ALPHABET = [" ", "\t", "\n", "\r", "a", "e", _COMBINING_ACUTE]

    @pytest.mark.parametrize("length", [0, 1, 2, 3, 4])
    def test_all_strings_up_to_length(self, length: int) -> None:
        for parts in itertools.product(self._ALPHABET, repeat=length):
            text = "".join(parts)
            assert normalize(text) == reference_normalize(text), repr(text)

    def test_all_strings_up_to_length_including_precomposed_and_exotic_spaces(self) -> None:
        # A second, smaller sweep (length <= 3) over an alphabet that also includes the
        # precomposed accented letter directly, NBSP, and the ideographic space, distinct
        # Unicode whitespace/NFC classes the first sweep's alphabet doesn't cover.
        alphabet = [" ", "\n", "a", _E_ACUTE_PRECOMPOSED, _NBSP, _IDEOGRAPHIC_SPACE]
        for length in range(4):
            for parts in itertools.product(alphabet, repeat=length):
                text = "".join(parts)
                assert normalize(text) == reference_normalize(text), repr(text)


class TestIdentityReturnContract:
    """The identity-return contract for ``tors.normalize``: when the
    complete pipeline is a no-op: NFC quick-check Yes and no CR and no
    ``[ \\t]`` run before a newline and no 3+ newline run and no strip delta,
    the original object comes back (``normalize(s) is s``): the identity probe
    is a handful of SIMD sentinel scans, no allocation at all. Inputs the
    probe cannot prove clean still run the scan, and an output==input
    comparison after it extends the same guarantee to them, so the complete
    property is: ``normalize(s) is s`` whenever ``normalize(s) == s``."""

    def test_already_clean_inputs_return_the_same_object(self) -> None:
        clean = "plain text\n\nwith paragraphs\n\ncaf\u00e9 na\u00efve"
        assert normalize(clean) == clean  # the precondition, checked
        assert normalize(clean) is clean

    @pytest.mark.parametrize(
        "dirty",
        [
            "a\r\nb",  # CR folds
            "a\rb",  # lone CR folds
            "a \nb",  # [ \t] before a newline is dropped
            "a\t\nb",
            "a\n\n\nb",  # blank run collapses
            "  leading",  # strip delta
            "trailing  ",
            "trailing\n",
            "\u00a0x",  # NBSP is Python whitespace: leading strip fires
            "x\u00a0",
            "cafe\u0301",  # QC-Maybe: NFC composes, value changes
        ],
        ids=[
            "crlf",
            "lone-cr",
            "space-before-nl",
            "tab-before-nl",
            "blank-run",
            "leading-ws",
            "trailing-ws",
            "trailing-nl",
            "nbsp-lead",
            "nbsp-trail",
            "decomposed",
        ],
    )
    def test_dirty_inputs_return_a_new_object(self, dirty: str) -> None:
        result = normalize(dirty)
        assert result == reference_normalize(dirty)  # the value stays pinned
        assert result is not dirty

    @given(pathological_text())
    @settings(max_examples=500)
    def test_value_identity_implies_object_identity(self, text: str) -> None:
        result = normalize(text)
        if result == text:
            assert result is text


class TestRandomizedProperty:
    @given(
        st.text(
            alphabet=st.sampled_from(f" \t\n\r{_NBSP}{_E_ACUTE_PRECOMPOSED}a{_IDEOGRAPHIC_SPACE}"),
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

    @given(pathological_text())
    @settings(max_examples=500)
    def test_matches_reference_over_pathological_whitespace_and_line_endings(
        self, text: str
    ) -> None:
        assert normalize(text) == reference_normalize(text)


class TestSeededRandomGenerator:
    """A seeded, dependency-free random generator as a second randomized methodology,
    independent of hypothesis being installed/available."""

    def test_matches_reference_over_seeded_random_corpora(self) -> None:
        alphabet = list(
            " \t\n\r" + string.ascii_letters + _E_ACUTE_PRECOMPOSED + "  " + _IDEOGRAPHIC_SPACE
        )
        rng = random.Random(1234567)
        for _ in range(500):
            length = rng.randint(0, 120)
            text = "".join(rng.choice(alphabet) for _ in range(length))
            assert normalize(text) == reference_normalize(text), repr(text)

    def test_matches_reference_over_seeded_pathological_whitespace_runs(self) -> None:
        # Long, biased-toward-whitespace random corpora: runs of 1-200 identical
        # whitespace/newline characters chained together, occasionally interrupted by a
        # word or a decomposed accent pair.
        rng = random.Random(987654321)
        chunks = ["a", "b", "e" + _COMBINING_ACUTE, "word"]
        ws_chars = [" ", "\t", "\n", "\r"]
        for _ in range(200):
            parts = []
            for _ in range(rng.randint(0, 15)):
                if rng.random() < 0.6:
                    parts.append(rng.choice(ws_chars) * rng.randint(1, 200))
                else:
                    parts.append(rng.choice(chunks))
            text = "".join(parts)
            assert normalize(text) == reference_normalize(text), repr(text)
