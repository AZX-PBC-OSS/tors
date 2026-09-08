"""Contract gate for ``tors.truncate_ellipsis``: hard cut to at most
``max_chars`` codepoints plus a U+2026 ``…`` marker, never
mid-grapheme-cluster — the DB-column truncation shape (a storage bound is
positional, not semantic, so unlike ``truncate_to_bounds`` there is no
word/sentence awareness).

What this gate pins:

- IDENTITY: ``<= max_chars`` codepoints comes back as the original object
  (``is``, not just ``==``).
- EXACT BOUND: on plain text the stored length is exactly ``max_chars``
  (``max_chars - 1`` kept + the marker), and it NEVER exceeds ``max_chars``
  on any input (hypothesis, over text rich in combining marks and format
  chars).
- ASCII PARITY: on ASCII input graphemes are codepoints, so the result
  equals the naive ``value[:max_chars - 1] + "…"`` spelling exactly — the
  ta_sync column-bound ``truncate`` this replaces, pinned byte-identical
  where the naive cut is already safe.
- CLUSTER SAFETY: literal pins for combining accents, ZWJ sequences, and
  regional-indicator flags (the classes where the naive cut corrupts).
- EDGES: ``max_chars == 0`` yields ``""`` (no room for even the marker —
  the naive spelling answers ``"…"`` here, exceeding a zero bound);
  negative raises ``ValueError``; no trailing-whitespace trim.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import truncate_ellipsis

_ELLIPSIS = "\u2026"

_TEXT = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Zs", "P", "Mn", "Mc", "Cf"),
        max_codepoint=0x2FFFF,
    ),
    max_size=300,
)

_ASCII = st.text(alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E), max_size=300)


def _naive(value: str, max_chars: int) -> str:
    """The ta_sync column-bound spelling this replaces."""
    return value[: max(max_chars - 1, 0)] + _ELLIPSIS


class TestIdentity:
    @given(text=_TEXT)
    @settings(max_examples=200)
    def test_no_truncation_returns_the_original_object(self, text: str) -> None:
        assert truncate_ellipsis(text, len(text)) is text
        assert truncate_ellipsis(text, len(text) + 50) is text

    def test_empty_text_at_zero_bound_is_identity(self) -> None:
        assert truncate_ellipsis("", 0) == ""


class TestExactBound:
    def test_plain_text_lands_exactly_on_the_bound(self) -> None:
        assert truncate_ellipsis("hello world", 6) == "hello" + _ELLIPSIS
        got = truncate_ellipsis("hello world", 6)
        assert len(got) == 6

    def test_single_char_bound_is_just_the_marker(self) -> None:
        assert truncate_ellipsis("abcdef", 1) == _ELLIPSIS

    def test_zero_bound_is_empty_not_a_marker(self) -> None:
        assert truncate_ellipsis("hello", 0) == ""

    def test_no_trailing_whitespace_trim(self) -> None:
        assert truncate_ellipsis("abc   def", 5) == "abc " + _ELLIPSIS

    def test_negative_bound_raises(self) -> None:
        with pytest.raises(ValueError, match="max_chars"):
            truncate_ellipsis("hello", -1)


class TestAsciiParity:
    @given(text=_ASCII, max_chars=st.integers(min_value=0, max_value=320))
    @settings(max_examples=300)
    def test_ascii_matches_the_naive_spelling_exactly(self, text: str, max_chars: int) -> None:
        # On ASCII every cut is cluster-safe, so the two spellings agree
        # wherever truncation happens — except max_chars == 0, where the
        # naive spelling wrongly emits the marker past a zero bound, and
        # inputs that already fit, which come back untouched.
        if max_chars == 0:
            assert truncate_ellipsis(text, 0) == ""
        elif len(text) <= max_chars:
            assert truncate_ellipsis(text, max_chars) == text
        else:
            assert truncate_ellipsis(text, max_chars) == _naive(text, max_chars)

    @given(text=_TEXT, max_chars=st.integers(min_value=1, max_value=320))
    @settings(max_examples=300)
    def test_never_exceeds_max_chars(self, text: str, max_chars: int) -> None:
        assert len(truncate_ellipsis(text, max_chars)) <= max_chars

    @given(text=_TEXT, max_chars=st.integers(min_value=1, max_value=320))
    @settings(max_examples=300)
    def test_truncated_result_keeps_a_text_prefix_plus_marker(
        self, text: str, max_chars: int
    ) -> None:
        got = truncate_ellipsis(text, max_chars)
        if got != text:
            assert got.endswith(_ELLIPSIS)
            assert text.startswith(got[: -len(_ELLIPSIS)])


class TestClusterSafety:
    def test_combining_accent_snaps_back(self) -> None:
        text = "ab\u0301cd"
        assert truncate_ellipsis(text, 3) == "a" + _ELLIPSIS
        assert truncate_ellipsis(text, 4) == "ab\u0301" + _ELLIPSIS

    def test_zwj_sequence_is_never_split(self) -> None:
        text = "hi \U0001F469\u200d\U0001F52C there"
        for max_chars in range(len(text) + 3):
            got = truncate_ellipsis(text, max_chars)
            assert len(got) <= max_chars
            kept = got[:-1] if got != text and got.endswith(_ELLIPSIS) else got
            assert ("\u200d" in kept) == ("\U0001F469" in kept and "\U0001F52C" in kept), (
                f"split ZWJ at {max_chars}: {got!r}"
            )

    def test_flag_pair_is_never_split(self) -> None:
        text = "a\U0001F1FA\U0001F1F8b"
        for max_chars in range(len(text) + 1):
            got = truncate_ellipsis(text, max_chars)
            assert len(got) <= max_chars
            kept = got[:-1] if got != text and got.endswith(_ELLIPSIS) else got
            assert ("\U0001F1FA" in kept) == ("\U0001F1F8" in kept), (
                f"split flag at {max_chars}: {got!r}"
            )

    def test_thai_sara_am_backs_off(self) -> None:
        assert truncate_ellipsis("0\u0e33", 2) == "0\u0e33"
        # Budget 2 keeps one codepoint plus the marker, but the first
        # cluster is TWO codepoints wide: keeping "0" alone would split
        # it, so the cut snaps back past the whole cluster ("0" without
        # its SARA AM renders differently) and only the marker remains.
        assert truncate_ellipsis("0\u0e33x", 2) == _ELLIPSIS


class TestArgumentBoundary:
    def test_lone_surrogate_is_refused(self) -> None:
        # The str-in family boundary: a str CPython can hold but UTF-8
        # cannot encode never reaches the Rust core.
        with pytest.raises(UnicodeEncodeError):
            truncate_ellipsis("\ud800abc", 2)
