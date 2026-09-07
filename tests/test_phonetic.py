"""Contract gate for ``tors.soundex`` and ``tors.metaphone``, classic
ENGLISH/Latin-script-oriented phonetic-code algorithms (``rphonetic``, an
Apache Commons Codec port), at native speed with the GIL released.

Both pre-filter their input to ASCII letters before encoding, because
``rphonetic`` 4.0.0's ``Soundex::encode`` and ``DoubleMetaphone::encode``
both panic on ordinary accented input. ``Soundex``'s own "clean" step
filters by the full-Unicode ``char::is_alphabetic`` (too broad: Cyrillic,
CJK, Greek, and accented Latin like ``'é'`` all pass it) and then
unconditionally indexes a 26-element ASCII mapping table, and
``DoubleMetaphone`` separately byte-slices assuming one byte per
character. tors never lets a Rust panic reach Python, so both functions
strip non-ASCII-letter characters first. The regression class pinned by
``test_accented_names_do_not_crash_upstream_bug_regression`` below covers
this.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import metaphone, soundex


class TestSoundex:
    def test_classic_homophone_pair(self) -> None:
        assert soundex("Robert") == soundex("Rupert") == "R163"

    def test_textbook_vector(self) -> None:
        assert soundex("jumped") == "J513"

    def test_empty_input_is_empty_output(self) -> None:
        assert soundex("") == ""

    def test_deterministic(self) -> None:
        assert soundex("Ashcraft") == soundex("Ashcraft")


class TestMetaphone:
    def test_textbook_vector(self) -> None:
        assert metaphone("jumped") == "JMPT"

    def test_groups_smith_and_smyth(self) -> None:
        assert metaphone("Smith") == metaphone("Smyth")

    def test_empty_input_is_empty_output(self) -> None:
        assert metaphone("") == ""

    def test_deterministic(self) -> None:
        assert metaphone("Ashcraft") == metaphone("Ashcraft")


class TestAsciiOnlyScopeAndPanicAvoidance:
    def test_non_alphabetic_input_has_no_letters_to_encode(self) -> None:
        assert soundex("12345") == ""
        assert metaphone("12345") == ""
        assert soundex("!!!") == ""
        assert metaphone("!!!") == ""

    def test_non_ascii_scripts_have_no_ascii_letters_to_encode(self) -> None:
        assert soundex("日本語") == ""
        assert metaphone("日本語") == ""

    def test_accented_names_do_not_crash_upstream_bug_regression(self) -> None:
        """The exact inputs that panic the raw ``rphonetic`` 4.0.0 crate
        when called directly: ordinary accented names, not adversarial
        input. tors's ASCII-letter pre-filter drops the accents (an
        documented degradation) rather than crashing."""
        assert soundex("José") == "J200"
        assert soundex("café") == "C100"
        assert soundex("Björk") == "B262"
        assert metaphone("José") == "JS"
        assert metaphone("café") == "KF"
        assert metaphone("Björk") == "PJRK"

    def test_mixed_ascii_and_non_ascii_keeps_only_the_ascii_letters(self) -> None:
        # "Müller" -> ASCII letters "Mller" (the ü is dropped, not the
        # word): a partial-degradation case distinct from the
        # all-non-ASCII "" cases above.
        assert soundex("Müller") == soundex("Mller")
        assert metaphone("Müller") == metaphone("Mller")


class TestNeverPanicsOverArbitraryInput:
    """`rphonetic` 4.0.0's `Soundex::encode`/`DoubleMetaphone::encode` are
    documented (see ``src/phonetic_impl.rs``) to panic on realistic
    accented input when called directly, the exact bug class the
    ``ascii_alphabetic`` pre-filter exists to close, currently pinned only
    by three hand-picked names (Jose/cafe/Bjork). Every other
    panic-avoidance module in this crate (``strip_code_fences``,
    ``utf8_is_valid``, ``truncate_to_bounds``) backs its hand-picked
    regression cases with a wide-alphabet property battery so a future
    ``rphonetic`` upgrade or filter refactor reintroducing a panic on some
    OTHER Unicode category doesn't go undetected between the three fixed
    examples. This is that same battery for soundex/metaphone."""

    _WIDE_ALPHABET = st.one_of(
        st.sampled_from("Aa1 \n\t!?'-"),
        st.characters(min_codepoint=0x300, max_codepoint=0x36F),  # combining marks
        st.characters(min_codepoint=0x1F300, max_codepoint=0x1FAFF),  # astral emoji
        st.characters(min_codepoint=0x0600, max_codepoint=0x06FF),  # Arabic (RTL)
        st.characters(min_codepoint=0x4E00, max_codepoint=0x9FFF),  # CJK
        st.characters(min_codepoint=0x00C0, max_codepoint=0x024F),  # accented Latin
        st.sampled_from(["​", "‌", "‍", "﻿"]),  # zero-width + BOM
    )

    @given(st.text(alphabet=_WIDE_ALPHABET, max_size=60))
    @settings(max_examples=500)
    def test_soundex_never_panics(self, text: str) -> None:
        result = soundex(text)
        assert isinstance(result, str)

    @given(st.text(alphabet=_WIDE_ALPHABET, max_size=60))
    @settings(max_examples=500)
    def test_metaphone_never_panics(self, text: str) -> None:
        result = metaphone(text)
        assert isinstance(result, str)


class TestArgumentContract:
    def test_soundex_rejects_non_str(self) -> None:
        with pytest.raises(TypeError):
            soundex(123)  # type: ignore[arg-type]

    def test_metaphone_rejects_non_str(self) -> None:
        with pytest.raises(TypeError):
            metaphone(123)  # type: ignore[arg-type]
