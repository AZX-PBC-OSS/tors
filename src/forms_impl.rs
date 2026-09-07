//! The standalone Unicode normalization forms: the pure-Rust core of
//! `tors.nfc` / `tors.nfd` / `tors.nfkc` / `tors.nfkd`, exposed without the
//! v0.1 pipeline's folding/collapsing/strip stages. `unicodedata.normalize`
//! is the original GIL-held whole-text C call this class of work exists for;
//! these are that call's four spellings as one detached pass each.
//!
//! Same tables as the pipeline (the `unicode-normalization` crate), so the
//! cross-interpreter determinism guarantee carries over unchanged: same input
//! -> same output on every supported Python. Parity with each interpreter's
//! own `unicodedata` is pinned per CI matrix leg by the exhaustive sweeps in
//! tests/test_parity.py (every decomposable codepoint, both canonical AND
//! compatibility, under all four forms, on both the raw character and its
//! NFD rendering, plus every Hangul syllable) and by the hypothesis
//! differentials in tests/test_forms.py.
//!
//! v0.4 identity-return contract: each form consults the crate's quick check
//! first (`is_nfc_quick` etc., the same property data CPython's
//! `unicodedata.normalize` fast path uses); a Yes proves the form is the
//! identity and the input is returned BORROWED: the pyo3 wrapper then hands
//! back the ORIGINAL `PyString` object, zero allocation and zero marshalling
//! (CPython's own `if is_normalized(...): return input` idiom). A No/Maybe
//! runs the full pass, and a final output==input comparison catches the
//! Maybe-but-already-normalized class (e.g. canonically-ordered combining
//! marks with no composable pair), so the contract is complete: the caller
//! gets the same object back whenever the form changes nothing. The quick
//! check's cost is the whole fast path: 2.4ms on 12 MiB ASCII prose,
//! measured, against the 94ms NFC collect it skips (README performance
//! section: the end-to-end before/after numbers).
//!
//! Pure Rust, no pyo3 types: the criterion bench (benches/normalize.rs,
//! benches/text.rs) drives these paths directly; the pyo3 wrappers in
//! `lib.rs` add only the argument borrow and return marshalling (see the
//! crate GIL model there, including the one-time O(input) UTF-8
//! materialization on the first non-ASCII call, the same class
//! `tors.normalize` pays, pinned by the decomposed-corpus cells of
//! tests/test_gil_release.py).

use std::borrow::Cow;

use unicode_normalization::{
    IsNormalized, UnicodeNormalization, is_nfc_quick, is_nfd_quick, is_nfkc_quick, is_nfkd_quick,
};

/// NFC: canonical composition, exactly `unicodedata.normalize("NFC", text)`.
/// Quick-check Yes (or an output==input collect) returns the input borrowed.
pub fn nfc(text: &str) -> Cow<'_, str> {
    if is_nfc_quick(text.chars()) == IsNormalized::Yes {
        return Cow::Borrowed(text);
    }
    let composed: String = text.nfc().collect();
    if composed == text {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(composed)
    }
}

/// NFD: canonical decomposition, exactly `unicodedata.normalize("NFD", text)`.
pub fn nfd(text: &str) -> Cow<'_, str> {
    if is_nfd_quick(text.chars()) == IsNormalized::Yes {
        return Cow::Borrowed(text);
    }
    let decomposed: String = text.nfd().collect();
    if decomposed == text {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(decomposed)
    }
}

/// NFKC: compatibility composition, canonical decomposition with the
/// compatibility (`<...>`) mappings applied, then composed. Exactly
/// `unicodedata.normalize("NFKC", text)`.
pub fn nfkc(text: &str) -> Cow<'_, str> {
    if is_nfkc_quick(text.chars()) == IsNormalized::Yes {
        return Cow::Borrowed(text);
    }
    let composed: String = text.nfkc().collect();
    if composed == text {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(composed)
    }
}

/// NFKD: compatibility decomposition, exactly
/// `unicodedata.normalize("NFKD", text)`.
pub fn nfkd(text: &str) -> Cow<'_, str> {
    if is_nfkd_quick(text.chars()) == IsNormalized::Yes {
        return Cow::Borrowed(text);
    }
    let decomposed: String = text.nfkd().collect();
    if decomposed == text {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(decomposed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;
    use unicode_normalization::{IsNormalized, is_nfd_quick, is_nfkc_quick, is_nfkd_quick};

    const E_ACUTE_PRECOMPOSED: &str = "\u{e9}";
    const E_ACUTE_DECOMPOSED: &str = "e\u{0301}";
    const LIGATURE_FI: &str = "\u{fb01}";

    #[test]
    fn nfc_composes_and_nfd_decomposes_the_acute_pair() {
        assert_eq!(nfc(E_ACUTE_DECOMPOSED), E_ACUTE_PRECOMPOSED);
        assert_eq!(nfd(E_ACUTE_PRECOMPOSED), E_ACUTE_DECOMPOSED);
        // Both are idempotent on their own output.
        assert_eq!(nfc(E_ACUTE_PRECOMPOSED), E_ACUTE_PRECOMPOSED);
        assert_eq!(nfd(E_ACUTE_DECOMPOSED), E_ACUTE_DECOMPOSED);
    }

    #[test]
    fn compat_decompositions_fire_only_under_the_k_forms() {
        // The form-level distinction: U+FB01 is untouched by NFC/NFD, mapped to
        // "fi" (composed, two chars) by NFKC and its decomposed spelling by NFKD.
        assert_eq!(nfc(LIGATURE_FI), LIGATURE_FI);
        assert_eq!(nfd(LIGATURE_FI), LIGATURE_FI);
        assert_eq!(nfkc(LIGATURE_FI), "fi");
        assert_eq!(nfkd(LIGATURE_FI), "fi");
    }

    #[test]
    fn fullwidth_maps_to_halfwidth_only_under_the_k_forms() {
        assert_eq!(nfkc("\u{ff01}"), "!");
        assert_eq!(nfkd("\u{ff01}"), "!");
        assert_eq!(nfc("\u{ff01}"), "\u{ff01}");
        assert_eq!(nfd("\u{ff01}"), "\u{ff01}");
    }

    #[test]
    fn hangul_syllables_compose_decompose_and_survive_the_k_forms() {
        // U+AC00 "가": NFC of its jamo NFD is the syllable; NFKC leaves the
        // composed syllable alone (Hangul has no compatibility mapping).
        let syllable = "\u{ac00}";
        let jamo = nfd(syllable);
        assert_eq!(jamo, "\u{1100}\u{1161}");
        assert_eq!(nfc(&jamo), syllable);
        assert_eq!(nfkc(syllable), syllable);
        assert_eq!(nfkd(syllable), jamo);
    }

    #[test]
    fn every_string_round_trips_through_nfd_then_nfc() {
        // The canonical-forms invariant: NFC(NFD(x)) == NFC(x) for any x.
        let repeated_decomposed = E_ACUTE_DECOMPOSED.repeat(10);
        let cases = [
            "",
            "plain text",
            "caf\u{e9} na\u{ef}ve",
            repeated_decomposed.as_str(),
            LIGATURE_FI,
            "\u{ac00}\u{1100}\u{1161}\u{11a8}",
        ];
        for case in cases {
            assert_eq!(nfc(&nfd(case)), nfc(case), "mismatch for {case:?}");
        }
    }

    #[test]
    fn empty_and_ascii_text_are_untouched_by_every_form() {
        for case in ["", "plain composed text"] {
            assert_eq!(nfc(case), case);
            assert_eq!(nfd(case), case);
            assert_eq!(nfkc(case), case);
            assert_eq!(nfkd(case), case);
        }
    }

    // --- v0.4: quick-check fast paths + the identity-return contract ---------

    fn qc_all_forms(text: &str) -> [IsNormalized; 4] {
        [
            unicode_normalization::is_nfc_quick(text.chars()),
            is_nfd_quick(text.chars()),
            is_nfkc_quick(text.chars()),
            is_nfkd_quick(text.chars()),
        ]
    }

    #[test]
    fn ascii_quick_checks_yes_under_every_form() {
        // The v0.4 subsumption claim, pinned: ASCII carries no canonical or
        // compatibility mappings and no combining marks, so every ASCII string
        // quick-checks Yes under all four forms and the whole normalization
        // pass is skippable, which is why tors needs no separate ASCII
        // special case. (The crate's quick-check loop has an explicit ASCII
        // lane, `if ch <= '\x7f' { continue }`, measured 2.4ms against the
        // 94ms NFC collect it can skip on 12 MiB of ASCII prose.)
        for text in [
            "",
            "plain text",
            "The quarterly oil sample. ".repeat(500).as_str(),
        ] {
            for (i, result) in qc_all_forms(text).into_iter().enumerate() {
                assert_eq!(
                    result,
                    IsNormalized::Yes,
                    "form {i}, text len {}",
                    text.len()
                );
            }
        }
        for byte in 0u32..0x80 {
            let ch = char::from_u32(byte).expect("ASCII byte is a char");
            for (i, result) in qc_all_forms(&ch.to_string()).into_iter().enumerate() {
                assert_eq!(result, IsNormalized::Yes, "form {i}, U+{byte:04X}");
            }
        }
    }

    #[test]
    fn quick_check_yes_inputs_borrow_the_input_instead_of_allocating() {
        // The identity-return contract's core lane: the crate's quick check
        // proves the form is the identity, so the pass is skipped entirely and
        // the input is returned borrowed (the pyo3 layer hands back the
        // ORIGINAL PyObject, as CPython's own `unicodedata.normalize` fast path does).
        for text in ["", "plain text", "caf\u{e9} na\u{ef}ve", "\u{ac00}"] {
            assert!(
                matches!(nfc(text), Cow::Borrowed(s) if s == text),
                "nfc {text:?}"
            );
            assert!(
                matches!(nfkc(text), Cow::Borrowed(s) if s == text),
                "nfkc {text:?}"
            );
        }
        for text in ["", "plain text", "cafe\u{0301}"] {
            assert!(
                matches!(nfd(text), Cow::Borrowed(s) if s == text),
                "nfd {text:?}"
            );
            assert!(
                matches!(nfkd(text), Cow::Borrowed(s) if s == text),
                "nfkd {text:?}"
            );
        }
    }

    #[test]
    fn transform_bearing_inputs_return_owned_output() {
        for (text, form) in [
            ("cafe\u{0301}", nfc as fn(&str) -> Cow<'_, str>),
            ("caf\u{e9}", nfd),
            ("\u{fb01}", nfkc),
            ("\u{fb01}", nfkd),
            ("\u{ac00}", nfd),
            ("\u{ac00}", nfkd),
        ] {
            assert!(matches!(form(text), Cow::Owned(_)), "form on {text:?}");
        }
    }

    #[test]
    fn qc_maybe_value_identity_inputs_borrow_via_the_output_comparison() {
        // "q" + ogonek + acute: canonically ordered (ccc 202 < 230), no
        // composable pair (no precomposed q-ogonek exists); NFC leaves it
        // byte-identical, but its combining marks make the quick check Maybe.
        // The collect runs, then the output==input comparison returns the
        // input instead of marshalling a fresh copy: the contract's second
        // lane, so `f(s) is s` holds whenever `f(s) == s`, not only when the
        // quick check could prove it up front.
        let text = "q\u{0328}\u{0301}";
        assert_eq!(
            unicode_normalization::is_nfc_quick(text.chars()),
            IsNormalized::Maybe
        );
        assert!(matches!(nfc(text), Cow::Borrowed(s) if s == text));
    }
}
