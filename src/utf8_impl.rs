//! SIMD UTF-8 validity checking, the pure-Rust core of `tors.utf8_is_valid`.
//!
//! `simdutf8::basic::from_utf8` answers "are these bytes well-formed UTF-8?"
//! with the same semantics as `core::str::from_utf8` (it is validated against
//! the std implementation in its own test suite; the crate-side parity test
//! below re-pins it over every ill-formed class this repo's decoder contract
//! names). It uses a SIMD scan instead of the std decoder's byte-at-a-time
//! loop, and returns a `bool`: no error classification, no exception
//! construction. The caller asked a yes/no question and pays for exactly that.
//!
//! Pure Rust, no pyo3 types: the criterion bench (benches/utf8.rs) drives this
//! path directly; the pyo3 wrapper in `lib.rs` adds only the zero-copy
//! `PyBytes` argument borrow (see the crate GIL model there).

/// Is `raw` well-formed UTF-8: the boolean the stdlib has no primitive for
/// (its only spelling is decode-and-catch, which materializes the decoded
/// `str` on the yes path and pays exception flow on the no path).
pub fn is_valid(raw: &[u8]) -> bool {
    simdutf8::basic::from_utf8(raw).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stdlib's own notion of well-formedness: `simdutf8::basic` must
    /// answer identically, which is the crate's headline guarantee and the
    /// property the Python-side hypothesis pin re-runs against the RUNNING
    /// interpreter's decoder.
    fn std_valid(raw: &[u8]) -> bool {
        core::str::from_utf8(raw).is_ok()
    }

    #[test]
    fn valid_inputs_including_every_range_boundary_are_accepted() {
        assert!(is_valid(b""));
        assert!(is_valid(b"hello world, plain ASCII text."));
        assert!(is_valid("caf\u{e9} na\u{ef}ve \u{1f600} text".as_bytes()));
        // One anchor on each side of every exclusion zone: the 1/2/3/4-byte
        // range boundaries, the surrogate block's edges, and the U+10FFFF
        // ceiling: the cases a hand-rolled validator most easily rejects
        // wrongly. U+FFFF is a noncharacter but a VALID encoding.
        assert!(is_valid(b"\x7f")); // U+007F, last 1-byte codepoint
        assert!(is_valid(b"\xc2\x80")); // U+0080, first 2-byte
        assert!(is_valid(b"\xdf\xbf")); // U+07FF, last 2-byte
        assert!(is_valid(b"\xe0\xa0\x80")); // U+0800, first 3-byte
        assert!(is_valid(b"\xed\x9f\xbf")); // U+D7FF, before surrogates
        assert!(is_valid(b"\xee\x80\x80")); // U+E000, after surrogates
        assert!(is_valid(b"\xef\xbf\xbf")); // U+FFFF, noncharacter, valid
        assert!(is_valid(b"\xf0\x90\x80\x80")); // U+10000, first 4-byte
        assert!(is_valid(b"\xf4\x8f\xbf\xbf")); // U+10FFFF, maximum
    }

    #[test]
    fn every_ill_formed_class_is_rejected() {
        // Truncated multi-byte tails.
        assert!(!is_valid(b"a\xc3"));
        assert!(!is_valid(b"\xf0\x9f"));
        assert!(!is_valid(b"\xf0\x9f\x98\x80\xf0\x9f"));
        assert!(!is_valid(b"\xe0\xa0"));
        // Overlong encodings (2/3/4-byte spellings of representable codepoints).
        assert!(!is_valid(b"\xc0\x80"));
        assert!(!is_valid(b"\xe0\x80\x80"));
        assert!(!is_valid(b"\xf0\x80\x80\x80"));
        // Surrogate encodings: CESU-8 high/low halves and a full pair.
        assert!(!is_valid(b"\xed\xa0\x80"));
        assert!(!is_valid(b"\xed\xb0\x80"));
        assert!(!is_valid(b"\xed\xa0\x80\xed\xb0\x80"));
        assert!(!is_valid(b"\xed\xbf\xbf"));
        // Lone continuation bytes.
        assert!(!is_valid(b"\x80"));
        assert!(!is_valid(b"a\x80\x80b"));
        // Invalid lead bytes: the legacy 5/6-byte leads FC/FD and 0xFF.
        assert!(!is_valid(b"\xfc\x84\x80\x80\x80\x80"));
        assert!(!is_valid(b"\xfd\x84\x80\x80\x80\x80"));
        assert!(!is_valid(b"\xff"));
        assert!(!is_valid(b"ok\xffok"));
        // Out-of-range (F4 continuation above U+10FFFF) and maximal subparts.
        assert!(!is_valid(b"\xf4\x90"));
        assert!(!is_valid(b"\xf0\x9f\x41"));
        assert!(!is_valid(b"fo\xd8o"));
    }

    #[test]
    fn answers_match_core_str_from_utf8_on_the_full_battery() {
        // The parity spine: every case above (both classes), plus the boundary
        // anchors, cross-checked against the std validity oracle one by one.
        // simdutf8 must never disagree with `core::str::from_utf8`.
        let cases: Vec<&[u8]> = vec![
            b"",
            b"hello world",
            "caf\u{e9} na\u{ef}ve \u{1f600} text".as_bytes(),
            b"\x7f",
            b"\xc2\x80",
            b"\xdf\xbf",
            b"\xe0\xa0\x80",
            b"\xed\x9f\xbf",
            b"\xee\x80\x80",
            b"\xef\xbf\xbf",
            b"\xf0\x90\x80\x80",
            b"\xf4\x8f\xbf\xbf",
            b"a\xc3",
            b"\xf0\x9f",
            b"\xf0\x9f\x98\x80\xf0\x9f",
            b"\xe0\xa0",
            b"\xc0\x80",
            b"\xe0\x80\x80",
            b"\xf0\x80\x80\x80",
            b"\xed\xa0\x80",
            b"\xed\xb0\x80",
            b"\xed\xa0\x80\xed\xb0\x80",
            b"\xed\xbf\xbf",
            b"\x80",
            b"a\x80\x80b",
            b"\xfc\x84\x80\x80\x80\x80",
            b"\xfd\x84\x80\x80\x80\x80",
            b"\xff",
            b"ok\xffok",
            b"\xf4\x90",
            b"\xf0\x9f\x41",
            b"fo\xd8o",
        ];
        for raw in cases {
            assert_eq!(is_valid(raw), std_valid(raw), "diverged on {raw:?}");
        }
    }
}
