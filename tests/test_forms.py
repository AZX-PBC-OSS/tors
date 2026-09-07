"""Contract gate for the standalone normalization forms:
``tors.nfc / nfd / nfkc / nfkd``: ``unicodedata.normalize(form, text)`` in one
GIL-released pass. ``unicodedata.normalize`` is the original GIL-held whole-text
C call this class exists for (a 12 MiB decomposed corpus holds the GIL for the
whole transform); the standalone forms expose each form without normalize's
folding/collapsing/strip stages.

Parity is pinned three ways, complementary by construction: the exhaustive
decomposable-codepoint and Hangul sweeps in tests/test_parity.py (every
decomposable codepoint under all four forms, both raw and NFD-rendered inputs,
per CI matrix leg), the hypothesis differentials below (arbitrary and
pathological text, including strings whose NFD is 3x their composed size), and
the behavioral pins that distinguish the FORMS from each other and from the
pipeline (compat decompositions fire under the K-forms and only under them).

The exhaustive-sweep confound note, extended per form: the NFC sweep
carves out {U+2000, U+2001} because ``tors.normalize`` strips whitespace-only
results, the strip stage confounds raw NFC parity there. The standalone forms
have NO strip stage, so the confound does not exist for them; that is pinned
explicitly in tests/test_parity.py (raw equality, whitespace preserved) rather
than silently assumed.
"""

from __future__ import annotations

import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors
from reference import pathological_text

# Built from codepoints (pure-ASCII source, per the suite's convention):
_E_ACUTE_PRECOMPOSED = chr(0x00E9)
_COMBINING_ACUTE = chr(0x0301)
_LIGATURE_FI = chr(0xFB01)
_HW_BEGIN = chr(0xFF61)  # halfwidth ideographic full stop, a compat-only mapping
_FORMS = ("NFC", "NFD", "NFKC", "NFKD")
_TORS_FORMS = {"NFC": tors.nfc, "NFD": tors.nfd, "NFKC": tors.nfkc, "NFKD": tors.nfkd}


class TestHypothesisParity:
    @given(pathological_text())
    @settings(max_examples=500)
    def test_nfc_nfd_agree_with_the_interpreter_on_pathological_text(self, text: str) -> None:
        for form in ("NFC", "NFD"):
            assert _TORS_FORMS[form](text) == unicodedata.normalize(form, text), form

    @given(pathological_text())
    @settings(max_examples=500)
    def test_nfkc_nfkd_agree_with_the_interpreter_on_pathological_text(self, text: str) -> None:
        for form in ("NFKC", "NFKD"):
            assert _TORS_FORMS[form](text) == unicodedata.normalize(form, text), form

    @given(st.text(alphabet=st.characters(exclude_categories=("Cn", "Cs")), max_size=64))
    @settings(max_examples=500)
    def test_all_four_forms_agree_with_the_interpreter_on_arbitrary_assigned_text(
        self, text: str
    ) -> None:
        """Uniform random unicode over the codepoints the RUNNING interpreter
        assigns (``Cn`` unassigned and ``Cs`` surrogates excluded; the
        latter because a lone surrogate is refused at the ``&str`` argument
        boundary by contract; hypothesis's charmap is built from this
        interpreter's own ``unicodedata``, keyed on ``unidata_version``, so the
        exclusion is per-leg correct by construction), the complement of
        ``pathological_text``'s bias. Any assigned codepoint can appear, so
        this guards table entries the pathological strategy never draws.

        The unassigned codepoints are intentionally OUT of this property's
        domain, and that is a measured finding, not a dodge: tors's tables
        (the ``unicode-normalization`` crate, 0.1.25) are Unicode 16.0.0,
        the same UCD CPython 3.14 ships, while this suite also runs on legs
        whose interpreters are behind (3.12 = UCD 15.0). Measured on the 3.12
        leg, the ONLY divergences between tors and the interpreter, over every
        non-surrogate codepoint, are on codepoints that leg leaves UNASSIGNED
        (i.e. tors normalizes characters the interpreter does not know exist
        yet): 20 new canonical decompositions (NFD: Todhri U+105C9/U+105E4,
        whose compositions pair a new base with the ancient U+0307, plus
        Tulu-Tigalari, Gurung Khema, and Kirat Rai precomposed marks), and 37
        new compatibility mappings (NFKC/NFKD: the Outlined Latin capital
        letters and digits, U+1CCD6-U+1CCF9, ``<font>``-decomposing to ASCII,
        plus U+A7F1), 57 total under NFKD, 0 under NFC for single characters.
        No codepoint the interpreter ASSIGNS diverges in any form (pinned
        exhaustively by the confinement sweep in tests/test_parity.py), which
        is the load-bearing parity guarantee; a future crate or interpreter
        change that breaks it fails these tests loudly. On UCD-16.0 legs
        (CPython 3.14) the residual is empty and this property's domain is the
        full range."""
        for form in _FORMS:
            assert _TORS_FORMS[form](text) == unicodedata.normalize(form, text), form


class TestFormBehavior:
    def test_nfc_composes_a_decomposed_sequence(self) -> None:
        assert tors.nfc("e" + _COMBINING_ACUTE) == _E_ACUTE_PRECOMPOSED

    def test_nfd_decomposes_a_precomposed_sequence(self) -> None:
        assert tors.nfd(_E_ACUTE_PRECOMPOSED) == "e" + _COMBINING_ACUTE

    def test_compat_decompositions_fire_only_under_the_k_forms(self) -> None:
        """The form-level distinction the pipeline pinned negatively
        (tests/test_parity.py's not-NFKC pin): the C-forms leave compatibility
        composites alone; the K-forms map them: U+FB01 "ﬁ" -> "f" + "i" under
        NFKD, "fi" under NFKC."""
        for form in ("NFC", "NFD"):
            assert _TORS_FORMS[form](_LIGATURE_FI) == _LIGATURE_FI, form
        assert tors.nfkc(_LIGATURE_FI) == "fi"
        assert tors.nfkd(_LIGATURE_FI) == "f" + "i"

    def test_fullwidth_to_halfwidth_is_a_compat_mapping(self) -> None:
        # U+FF01 fullwidth exclamation -> "!", the classic NFKC visual-folding
        # example; NFC must leave it fullwidth.
        assert tors.nfkc(chr(0xFF01)) == "!"
        assert tors.nfc(chr(0xFF01)) == chr(0xFF01)

    def test_the_empty_string_is_the_identity_in_every_form(self) -> None:
        for form in _FORMS:
            assert _TORS_FORMS[form]("") == "", form

    def test_forms_of_already_normalized_text_are_no_ops(self) -> None:
        text = "plain composed text"
        for form in _FORMS:
            assert _TORS_FORMS[form](text) == text, form

    def test_k_forms_decompose_the_decomposed_accent_further_to_nothing_new(
        self,
    ) -> None:
        """An acute accent has no compatibility mapping, so NFKC/NFKD of the
        decomposed pair equals NFC/NFD of it; the K-forms are supersets that
        add nothing here (guards against a K-table accidentally firing a
        canonical-only entry as compat)."""
        decomposed = "e" + _COMBINING_ACUTE
        assert tors.nfkc(decomposed) == _E_ACUTE_PRECOMPOSED
        assert tors.nfkd(decomposed) == decomposed

    def test_out_of_order_combining_marks_are_canonically_reordered(self) -> None:
        """Every combining-mark case elsewhere in this suite (and in
        ``reference.py``'s ``pathological_text`` strategy) either uses a
        single mark or marks already in canonical-combining-class order --
        the reordering ALGORITHM itself (UAX #15's stable sort by ccc) was
        never exercised on hand-built, out-of-order input.
        ``a`` + acute (U+0301, ccc 230) + dot-below (U+0323, ccc 220), in
        that (wrong) order: NFD/NFC must reorder to dot-below-then-acute,
        matching the running interpreter exactly. Built via ``chr()`` rather
        than literal combining-mark characters in source, to sidestep any
        editor/encoding normalization of the source file itself."""
        acute = chr(0x0301)
        dot_below = chr(0x0323)
        text = "a" + acute + dot_below
        expected_nfd = unicodedata.normalize("NFD", text)
        # Sanity: the interpreter really does reorder here (dot-below's
        # lower ccc must sort first) -- otherwise this test would pass
        # vacuously even with a broken reordering implementation.
        assert expected_nfd == "a" + dot_below + acute
        assert tors.nfd(text) == expected_nfd
        assert tors.nfc(text) == unicodedata.normalize("NFC", text)
        assert tors.nfkd(text) == unicodedata.normalize("NFKD", text)
        assert tors.nfkc(text) == unicodedata.normalize("NFKC", text)


class TestArgumentContract:
    @pytest.mark.parametrize("form_name", _FORMS)
    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), 123, None],
        ids=["bytes", "bytearray", "int", "none"],
    )
    def test_non_str_arguments_raise_type_error(
        self, form_name: str, not_str: object
    ) -> None:
        with pytest.raises(TypeError):
            _TORS_FORMS[form_name](not_str)  # type: ignore[arg-type]

    @pytest.mark.parametrize("form_name", _FORMS)
    def test_lone_surrogates_are_refused_at_the_argument_boundary(
        self, form_name: str
    ) -> None:
        """Same pyo3 ``&str`` boundary as ``tors.normalize`` /
        ``tors.finalize`` (pinned in tests/test_finalize.py's surrogate class):
        a str holding a lone surrogate cannot be UTF-8-borrowed, so the call
        raises ``UnicodeEncodeError`` ("surrogates not allowed") before any
        Rust code runs, the identical stdlib behavior happens at
        ``text.encode("utf-8")`` in a pure-Python pipeline."""
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            _TORS_FORMS[form_name]("\ud800")  # type: ignore[arg-type]


class TestIdentityReturnContract:
    """The identity-return contract: when a form changes nothing, the
    caller gets the ORIGINAL object back (``f(s) is s``), CPython's own
    ``unicodedata.normalize`` fast-path idiom (``is_normalized`` quick check ->
    return input). Zero allocation, zero copy, zero marshalling: the GIL model
    in the crate docs records this as the zero-cost path. Two lanes make the
    contract complete: the quick check proves it up front for QC-Yes inputs,
    and an output==input comparison after the full pass catches the
    QC-Maybe-but-already-normalized class (canonically-ordered combining marks
    with no composable pair)."""

    # Quick-check-Yes spellings per form: composed text for the C-forms,
    # decomposed text for the D-forms (each is that form's fixed point; the
    # K-forms' fixed points are decomposed AND compat-free).
    _CLEAN: dict[str, str] = {
        "NFC": "caf\u00e9 na\u00efve composed text",
        "NFD": "cafe\u0301 na\u0301ve decomposed text",
        "NFKC": "caf\u00e9 plain composed text",
        "NFKD": "plain text cafe\u0301",
    }

    @pytest.mark.parametrize("form_name", _FORMS)
    def test_quick_check_yes_inputs_return_the_same_object(self, form_name: str) -> None:
        clean = self._CLEAN[form_name]
        assert _TORS_FORMS[form_name](clean) is clean, form_name

    @pytest.mark.parametrize("form_name", _FORMS)
    def test_transform_bearing_inputs_return_a_new_object(self, form_name: str) -> None:
        dirty = {
            "NFC": "cafe\u0301",  # composes
            "NFD": "caf\u00e9",  # decomposes
            "NFKC": "\ufb01",  # compat ligature fires
            "NFKD": "\ufb01",
        }[form_name]
        result = _TORS_FORMS[form_name](dirty)
        assert result == unicodedata.normalize(form_name, dirty), form_name
        assert result is not dirty, form_name

    def test_qc_maybe_value_identity_input_returns_the_same_object(self) -> None:
        # "q" + ogonek + acute: canonically ordered, no composable pair; NFC
        # is the identity but the quick check is Maybe, so only the
        # post-pass output==input comparison can see it.
        text = "q\u0328\u0301"
        assert unicodedata.normalize("NFC", text) == text
        assert tors.nfc(text) is text

    @given(pathological_text())
    @settings(max_examples=500)
    def test_value_identity_implies_object_identity(self, text: str) -> None:
        # The complete contract as one property: whenever the form's output
        # equals the input, the returned object IS the input; the two lanes
        # above are the mechanism, this is the promise.
        for form in _FORMS:
            result = _TORS_FORMS[form](text)
            if result == text:
                assert result is text, form
