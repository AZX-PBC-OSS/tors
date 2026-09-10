"""Contract gate for ``tors.truncate_to_bounds``: cut at the last word/sentence
boundary at or before ``max_chars`` rather than mid-word/mid-sentence: a
composition of the crate's own ``word_bounds``/``sentence_bounds``
segmentation, no new algorithm and no stdlib equivalent to parity-pin
against. The one invariant every property below exists to pin: the result
never exceeds ``max_chars`` codepoints, whichever boundary mode or fallback
path produced it.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import truncate_to_bounds

_TEXT = st.text(
    alphabet=st.characters(
        # L/N/Zs/P for the bulk (plain words/numbers/spaces/punctuation),
        # plus Mn/Mc (combining marks, the class that grapheme-safety
        # depends on) and Cf (format chars, e.g. ZWJ) so property
        # testing can actually reach the class of input where a word/
        # sentence boundary can land mid-grapheme-cluster.
        whitelist_categories=("L", "N", "Zs", "P", "Mn", "Mc", "Cf"),
        max_codepoint=0x2FFFF,
    ),
    max_size=300,
)


class TestNeverExceedsBudget:
    @given(text=_TEXT, max_chars=st.integers(min_value=0, max_value=320))
    @settings(max_examples=300)
    def test_word_boundary_never_exceeds_max_chars(self, text: str, max_chars: int) -> None:
        assert len(truncate_to_bounds(text, max_chars, "word")) <= max_chars

    @given(text=_TEXT, max_chars=st.integers(min_value=0, max_value=320))
    @settings(max_examples=300)
    def test_sentence_boundary_never_exceeds_max_chars(self, text: str, max_chars: int) -> None:
        assert len(truncate_to_bounds(text, max_chars, "sentence")) <= max_chars

    @given(text=_TEXT)
    @settings(max_examples=200)
    def test_identity_return_when_no_truncation_happens(self, text: str) -> None:
        # is-identity, not just equality: the zero-cost Cow::Borrowed path.
        assert truncate_to_bounds(text, len(text)) is text
        assert truncate_to_bounds(text, len(text) + 50) is text


class TestWordBoundary:
    def test_cuts_after_the_last_full_word_and_trims_the_dangling_space(self) -> None:
        assert truncate_to_bounds("cats are cute", 9) == "cats are"
        assert truncate_to_bounds("cats are cute", 13) == "cats are cute"

    def test_no_op_when_text_already_fits(self) -> None:
        assert truncate_to_bounds("short", 100) == "short"

    def test_falls_back_to_a_hard_cut_when_no_word_boundary_fits(self) -> None:
        text = "Supercalifragilisticexpialidocious"
        got = truncate_to_bounds(text, 10)
        assert got == text[:10]
        assert len(got) == 10

    def test_max_chars_zero_is_empty_string(self) -> None:
        assert truncate_to_bounds("hello", 0) == ""

    def test_empty_text_is_identity(self) -> None:
        text = ""
        assert truncate_to_bounds(text, 5) is text

    def test_default_boundary_is_word(self) -> None:
        assert truncate_to_bounds("cats are cute", 9) == truncate_to_bounds(
            "cats are cute", 9, "word"
        )


class TestSentenceBoundary:
    def test_cuts_at_the_last_full_sentence(self) -> None:
        text = "One. Two. Three."
        assert truncate_to_bounds(text, 10, "sentence") == "One. Two."

    def test_falls_back_to_a_hard_cut_when_no_sentence_boundary_fits(self) -> None:
        got = truncate_to_bounds("One. Two. Three.", 8, "sentence")
        assert len(got) <= 8


class TestGraphemeSafety:
    """A word/sentence boundary is not automatically a grapheme-cluster
    boundary: ``word_bounds`` can score a combining mark as its own
    word-segment even though it renders as one unit with the preceding
    base character. Without cluster-safety, ``truncate_to_bounds("0" +
    chr(0x0E33), 1, "word")`` would silently drop the combining mark and
    return ``"0"``; these pin that case and the sibling cluster classes
    that could also be split."""

    def test_thai_sara_am_is_never_split_from_its_base(self) -> None:
        # U+0E33 combines with the preceding base into one grapheme
        # cluster; budget 1 can't fit the 2-codepoint cluster at all, so
        # the cluster-safe answer is empty, not a mangled lone "0".
        text = "0" + chr(0x0E33)
        assert truncate_to_bounds(text, 1, "word") == ""
        assert truncate_to_bounds(text, 2, "word") == text

    def test_combining_accent_hard_cut_backs_off_rather_than_splitting(self) -> None:
        # "ab" + combining acute accent + "cd": "b" + the accent is one
        # cluster spanning codepoints 1-3, so a budget of 2 must back off
        # to "a" rather than emitting "b" without its accent.
        text = "ab" + chr(0x0301) + "cd"
        assert truncate_to_bounds(text, 2, "word") == "a"
        assert truncate_to_bounds(text, 3, "word") == "ab" + chr(0x0301)

    def test_zwj_emoji_sequence_is_never_partially_included(self) -> None:
        # woman + ZWJ + microscope is one grapheme cluster; it must appear
        # whole or not at all in the result, at every budget.
        woman, zwj, microscope = chr(0x1F469), chr(0x200D), chr(0x1F52C)
        text = f"hi {woman}{zwj}{microscope} there"
        for max_chars in range(len(text) + 1):
            got = truncate_to_bounds(text, max_chars, "word")
            has_zwj = zwj in got
            has_woman = woman in got
            has_scope = microscope in got
            assert has_zwj == (has_woman and has_scope), (max_chars, got)

    def test_regional_indicator_flag_pair_is_never_split(self) -> None:
        text = "a\U0001f1fa\U0001f1f8b"
        for max_chars in range(len(text) + 1):
            got = truncate_to_bounds(text, max_chars, "word")
            assert ("\U0001f1fa" in got) == ("\U0001f1f8" in got), (max_chars, got)


class TestArgumentContract:
    def test_negative_max_chars_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="max_chars"):
            truncate_to_bounds("x", -1)

    def test_unrecognized_boundary_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="boundary"):
            truncate_to_bounds("x", 1, "paragraph")  # type: ignore[arg-type]
