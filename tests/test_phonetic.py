"""Contract gate for the ``tors`` phonetic-code algorithms (classic
ENGLISH/Latin-script-oriented heuristics via ``rphonetic``, an Apache
Commons Codec port), at native speed with the GIL released: ``soundex``,
``metaphone``, ``double_metaphone``, ``nysiis``, and ``daitch_mokotoff``.

All pre-filter their input to ASCII letters before encoding, because
``rphonetic`` 4.0.0's ``Soundex::encode`` and ``DoubleMetaphone::encode``
both panic on ordinary accented input. ``Soundex``'s own "clean" step
filters by the full-Unicode ``char::is_alphabetic`` (too broad: Cyrillic,
CJK, Greek, and accented Latin like ``'é'`` all pass it) and then
unconditionally indexes a 26-element ASCII mapping table, and
``DoubleMetaphone`` separately byte-slices assuming one byte per
character. tors never lets a Rust panic reach Python, so all five
functions strip non-ASCII-letter characters first. The regression class
pinned by ``test_accented_names_do_not_crash_upstream_bug_regression``
below covers this; for the newer three the filter additionally keeps
NYSIIS keys pure ASCII (the crate's own clean step would let accented
letters through into the code) and gives Daitch-Mokotoff one uniform
degradation rule instead of per-character folding.

Every pinned vector below was derived by probing the built extension
(and cross-checked against the vector tables in rphonetic's own test
suite, which ports Apache Commons Codec's test data verbatim); none is
pinned from memory.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import daitch_mokotoff, double_metaphone, metaphone, nysiis, refined_soundex, soundex


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

    @given(st.text(alphabet=_WIDE_ALPHABET, max_size=60))
    @settings(max_examples=500)
    def test_refined_soundex_never_panics(self, text: str) -> None:
        # refined_soundex shares soundex's exact upstream bug (verified
        # directly against the raw crate: RefinedSoundex::default().encode
        # panics with an out-of-bounds table index on "José", the identical
        # unguarded `ch as usize - 65`), so it gets the same battery.
        result = refined_soundex(text)
        assert isinstance(result, str)


class TestRefinedSoundex:
    def test_classic_homophone_pair(self) -> None:
        assert refined_soundex("Robert") == refined_soundex("Rupert") == "R901096"

    def test_textbook_vector(self) -> None:
        assert refined_soundex("jumped") == "J408106"

    def test_distinct_from_classic_soundex(self) -> None:
        # Not a formatting variant: a different mapping table, verified to
        # actually produce a different code for the same input.
        assert refined_soundex("Robert") != soundex("Robert")

    def test_empty_input_is_empty_output(self) -> None:
        assert refined_soundex("") == ""

    def test_accented_name_does_not_panic_upstream_bug_regression(self) -> None:
        # THE regression this needs: RefinedSoundex::default().encode
        # panics on "José" when called directly against the raw crate
        # (verified, not assumed) — an ordinary accented name, not
        # adversarial input.
        assert refined_soundex("José") == "J403"

    def test_rejects_non_str(self) -> None:
        with pytest.raises(TypeError):
            refined_soundex(123)  # type: ignore[arg-type]


class TestArgumentContract:
    def test_soundex_rejects_non_str(self) -> None:
        with pytest.raises(TypeError):
            soundex(123)  # type: ignore[arg-type]

    def test_metaphone_rejects_non_str(self) -> None:
        with pytest.raises(TypeError):
            metaphone(123)  # type: ignore[arg-type]


class TestDoubleMetaphone:
    def test_textbook_vector(self) -> None:
        # commons-codec's own test-table vector (rphonetic ports it
        # verbatim): the pair is (primary, alternate).
        assert double_metaphone("jumped") == ("JMPT", "AMPT")

    def test_alternate_key_matches_schmidt_primary(self) -> None:
        # The algorithm's headline demonstration, only expressible in
        # the dual-key form: Smith and Schmidt match CROSS-KEY (Smith's
        # alternate equals Schmidt's primary "XMT"), not
        # primary-to-primary ("SM0" vs "XMT").
        assert double_metaphone("Smith") == ("SM0", "XMT")
        assert double_metaphone("Schmidt") == ("XMT", "SMT")

    def test_groups_smith_and_smythe_on_both_keys(self) -> None:
        assert double_metaphone("Smith") == double_metaphone("Smythe")

    def test_single_pronunciation_words_have_equal_keys(self) -> None:
        # The other half of the pair contract: no second pronunciation
        # means the alternate degenerates to the primary.
        assert double_metaphone("Alexander") == ("ALKS", "ALKS")

    def test_returns_a_2_tuple_of_str(self) -> None:
        result = double_metaphone("Ashcraft")
        assert type(result) is tuple
        assert len(result) == 2
        assert all(type(code) is str for code in result)

    def test_empty_input_is_empty_pair(self) -> None:
        assert double_metaphone("") == ("", "")

    def test_deterministic(self) -> None:
        assert double_metaphone("Ashcraft") == double_metaphone("Ashcraft")


class TestNysiis:
    def test_literature_grouping_rows(self) -> None:
        # The classic commons-codec test-table grouping rows: several
        # spellings, one NYSIIS code.
        for word in ("Brian", "Brown", "Brun"):
            assert nysiis(word) == "BRAN", word
        for word in ("Capp", "Cope", "Kipp"):
            assert nysiis(word) == "CAP", word
        for word in ("Dane", "Dean", "Dionne"):
            assert nysiis(word) == "DAN", word
        assert nysiis("Dent") == "DAD"
        assert nysiis("Phil") == "FAL"

    def test_strict_westerlund_and_washington(self) -> None:
        # "Westerlund" -> "WASTAR" is the crate's own strict-mode doctest
        # vector. "Washington" -> "WASANG" is hand-traced from the
        # published procedure (the H-after-S step duplicates the S, and
        # the strict 6-character cap trims "WASANGT") and probe-confirmed;
        # the oft-quoted "WASAN" does not come out of the procedure as
        # reproduced by commons-codec/rphonetic.
        assert nysiis("Westerlund") == "WASTAR"
        assert nysiis("Washington") == "WASANG"

    def test_schmidt_and_smith_differ(self) -> None:
        assert nysiis("Schmidt") == "SNAD"
        assert nysiis("Smith") == "SNAT"

    def test_empty_input_is_empty_output(self) -> None:
        assert nysiis("") == ""

    def test_deterministic(self) -> None:
        assert nysiis("Ashcraft") == nysiis("Ashcraft")


class TestDaitchMokotoff:
    def test_documented_examples(self) -> None:
        # The commons-codec DaitchMokotoffSoundex test table (ported
        # verbatim by rphonetic, probe-confirmed): the famous
        # same-surname pairs collapse to equal or intersecting code
        # lists.
        assert daitch_mokotoff("AUERBACH") == ["097400", "097500"]
        assert daitch_mokotoff("OHRBACH") == ["097400", "097500"]
        assert daitch_mokotoff("LIPSHITZ") == ["874400"]
        assert daitch_mokotoff("LIPPSZYC") == ["874400", "874500"]
        assert daitch_mokotoff("Moskowitz") == ["645740"]
        assert daitch_mokotoff("Moskovitz") == ["645740"]
        assert daitch_mokotoff("Jackson") == [
            "154600",
            "145460",
            "454600",
            "445460",
        ]

    def test_heavy_branching_example(self) -> None:
        # The crate's own showcase for rule-table branching: one Polish
        # spelling, eight candidate codes.
        assert daitch_mokotoff("Rosochowaciec") == [
            "944744",
            "944745",
            "944754",
            "944755",
            "945744",
            "945745",
            "945754",
            "945755",
        ]

    def test_branch_lists_intersect_across_transliterations(self) -> None:
        # The matching rule the list shape exists for: the Anglicized
        # spelling's single code is one of the original spelling's
        # candidates.
        original = daitch_mokotoff("Rosochowaciec")
        anglicized = daitch_mokotoff("Rosokhovatsets")
        assert anglicized == ["945744"]
        assert any(code in anglicized for code in original)

    def test_returns_a_list_of_str(self) -> None:
        result = daitch_mokotoff("Ashcraft")
        assert type(result) is list
        assert all(type(code) is str for code in result)

    def test_no_encodable_letters_is_all_padding(self) -> None:
        # Unlike the string-returning algorithms, DM pads every code to
        # 6 digits, so letterless input is the all-padding code, not
        # "". Pinned crate behavior.
        assert daitch_mokotoff("") == ["000000"]
        assert daitch_mokotoff("12345") == ["000000"]
        assert daitch_mokotoff("!!!") == ["000000"]
        assert daitch_mokotoff("日本語") == ["000000"]

    def test_deterministic(self) -> None:
        assert daitch_mokotoff("Ashcraft") == daitch_mokotoff("Ashcraft")


class TestNewAlgorithmsAsciiScope:
    """The same ASCII-letters-only scope pins as the existing pair's
    ``TestAsciiOnlyScopeAndPanicAvoidance`` (not re-pinned there: those
    tests belong to soundex/metaphone). For DM this deliberately
    bypasses the crate's own per-character ASCII folding (raw
    ``DaitchMokotoffSoundex`` maps ``"ţamas"`` to ``"364000|464000"``;
    tors drops the ``ţ`` first, so it encodes like ``"amas"``): the
    uniform documented scope wins over per-algorithm Unicode behavior.
    """

    def test_non_alphabetic_input_has_no_letters_to_encode(self) -> None:
        assert double_metaphone("12345") == ("", "")
        assert nysiis("12345") == ""
        assert daitch_mokotoff("12345") == ["000000"]

    def test_accented_names_degrade_not_crash(self) -> None:
        # "José" -> ASCII letters "Jos" etc: the accents are dropped,
        # the remaining letters keep their code, same degradation class
        # as the existing pair's pinned "José"/"café"/"Björk" rows.
        assert double_metaphone("José") == double_metaphone("Jos") == ("JS", "AS")
        assert double_metaphone("café") == double_metaphone("cafe") == ("KF", "KF")
        assert nysiis("José") == nysiis("Jos")
        assert nysiis("Müller") == nysiis("Mller") == "MLAR"
        assert daitch_mokotoff("Müller") == daitch_mokotoff("Mller") == ["689000"]
        assert daitch_mokotoff("ţamas") == daitch_mokotoff("amas")

    def test_mixed_ascii_and_non_ascii_keeps_only_the_ascii_letters(self) -> None:
        assert double_metaphone("Müller") == double_metaphone("Mller")
        assert nysiis("Müller") == nysiis("Mller")
        assert daitch_mokotoff("Müller") == daitch_mokotoff("Mller")


class TestNewAlgorithmsNeverPanicOverArbitraryInput:
    """The same wide-alphabet property battery as the existing pair's
    ``TestNeverPanicsOverArbitraryInput`` (see that class's docstring
    for why a hand-picked regression row is not enough), run for the
    three new functions.
    """

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
    def test_double_metaphone_never_panics(self, text: str) -> None:
        result = double_metaphone(text)
        assert isinstance(result, tuple)

    @given(st.text(alphabet=_WIDE_ALPHABET, max_size=60))
    @settings(max_examples=500)
    def test_nysiis_never_panics(self, text: str) -> None:
        result = nysiis(text)
        assert isinstance(result, str)

    @given(st.text(alphabet=_WIDE_ALPHABET, max_size=60))
    @settings(max_examples=500)
    def test_daitch_mokotoff_never_panics(self, text: str) -> None:
        result = daitch_mokotoff(text)
        assert isinstance(result, list)


class TestNewAlgorithmsArgumentContract:
    def test_rejects_non_str(self) -> None:
        for fn in (double_metaphone, nysiis, daitch_mokotoff):
            with pytest.raises(TypeError):
                fn(123)  # type: ignore[arg-type]

    def test_rejects_lone_surrogates_at_the_boundary(self) -> None:
        # A str holding lone surrogates (possible in CPython, impossible
        # in UTF-8) cannot cross the &str boundary: the same
        # UnicodeEncodeError-before-any-Rust-code-runs class as the rest
        # of tors's str-in functions.
        for fn in (double_metaphone, nysiis, daitch_mokotoff):
            with pytest.raises(UnicodeEncodeError):
                fn("\ud800")
