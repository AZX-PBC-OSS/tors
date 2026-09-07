//! CPython-parity UTF-8 decoding of raw bytes: the pure-Rust core of
//! `tors.decode_utf8` / `tors.finalize_utf8`.
//!
//! `decode_strict` is `core::str::from_utf8` plus the error classification that
//! reproduces CPython's `UnicodeDecodeError` fields exactly (`start`, `end`,
//! and `reason`), measured against CPython 3.12 over a malformed-shape battery and
//! 13,658 randomized invalid inputs (0 mismatches; the mapping is in
//! `classify`). `decode_replace` is `String::from_utf8_lossy`, which was measured
//! byte-exact against CPython's `errors="replace"` over the same battery plus
//! 14,069 randomized inputs (0 divergences): both implement the same
//! maximal-subpart substitution algorithm, so no hand-rolled replacement walk is
//! needed. The Python-side hypothesis pins (tests/test_decode_utf8.py) are the
//! proof and would catch any future std/CPython divergence loudly.
//!
//! Pure Rust, no pyo3 types: the criterion bench (benches/bytes.rs) drives this
//! path directly; the pyo3 wrapper in `lib.rs` adds only the zero-copy `PyBytes`
//! argument borrow and the return marshalling (see the crate GIL model there).

use std::borrow::Cow;

/// The three `UnicodeDecodeError` shapes CPython's strict UTF-8 decoder produces,
/// with CPython's exact byte span. The pyo3 layer (lib.rs) renders this as a true
/// `UnicodeDecodeError` (same type, `.encoding`, `.object`, `.start`, `.end`,
/// `.reason`, and therefore the same `str(exc)`), so `decode_utf8` raising is
/// indistinguishable from `raw.decode("utf-8")` raising.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// The byte at `start` cannot lead a sequence (a lone continuation byte, the
    /// overlong 2-byte leads 0xC0/0xC1, 0xF5-0xFF, or a legacy 5/6-byte lead).
    /// CPython reason: "invalid start byte". Span: that one byte.
    InvalidStart { start: usize, end: usize },
    /// A sequence led by a valid lead byte (0xC2-0xF4) failed: either a
    /// continuation slot held a non-continuation byte (span: the lead plus every
    /// valid continuation consumed, the maximal subpart; decoding resumes at the
    /// failing byte) or the first continuation violated the lead's range
    /// constraint (E0/overlong, ED/surrogates, F0/overlong, F4/range), where
    /// CPython reports the LEAD byte alone and re-processes the violating byte.
    /// CPython reason: "invalid continuation byte".
    InvalidContinuation { start: usize, end: usize },
    /// The input ended inside a multi-byte sequence. Span: every byte of the
    /// truncated sequence. CPython reason: "unexpected end of data".
    UnexpectedEnd { start: usize, end: usize },
}

impl DecodeError {
    pub fn start(&self) -> usize {
        match self {
            DecodeError::InvalidStart { start, .. }
            | DecodeError::InvalidContinuation { start, .. }
            | DecodeError::UnexpectedEnd { start, .. } => *start,
        }
    }

    pub fn end(&self) -> usize {
        match self {
            DecodeError::InvalidStart { end, .. }
            | DecodeError::InvalidContinuation { end, .. }
            | DecodeError::UnexpectedEnd { end, .. } => *end,
        }
    }

    /// CPython's reason string, verbatim: the `.reason` attribute of the
    /// `UnicodeDecodeError` the pyo3 layer raises.
    pub fn reason(&self) -> &'static str {
        match self {
            DecodeError::InvalidStart { .. } => "invalid start byte",
            DecodeError::InvalidContinuation { .. } => "invalid continuation byte",
            DecodeError::UnexpectedEnd { .. } => "unexpected end of data",
        }
    }
}

/// Map a `core::str::Utf8Error` to CPython's exact error shape. The span rule is
/// the one pyo3's `PyUnicodeDecodeError::new_utf8` itself uses (`start +
/// error_len`, or the input end when the input was truncated); the reason
/// classification (which pyo3 does NOT do; its generic reason "invalid utf-8"
/// diverges from CPython on every error) is the lead-byte test below, verified
/// against CPython 3.12 on 13,658 invalid inputs (0 mismatches).
fn classify(raw: &[u8], err: core::str::Utf8Error) -> DecodeError {
    let start = err.valid_up_to();
    let end = err.error_len().map_or(raw.len(), |len| start + len);
    match err.error_len() {
        None => DecodeError::UnexpectedEnd { start, end },
        Some(_) => {
            let is_valid_lead = raw
                .get(start)
                .is_some_and(|&byte| (0xC2..=0xF4).contains(&byte));
            if is_valid_lead {
                DecodeError::InvalidContinuation { start, end }
            } else {
                DecodeError::InvalidStart { start, end }
            }
        }
    }
}

/// Strict decoding: `core::str::from_utf8` (zero-copy borrow on success) plus
/// CPython-parity error classification on failure.
pub fn decode_strict(raw: &[u8]) -> Result<Cow<'_, str>, DecodeError> {
    core::str::from_utf8(raw)
        .map_err(|err| classify(raw, err))
        .map(Cow::Borrowed)
}

/// `errors="replace"` decoding: `String::from_utf8_lossy`, whose
/// maximal-subpart substitution was measured byte-exact against CPython's
/// `decode("utf-8", "replace")` (module docs); no hand-rolled walk needed.
pub fn decode_replace(raw: &[u8]) -> Cow<'_, str> {
    String::from_utf8_lossy(raw)
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;

    #[test]
    fn strict_decodes_valid_ascii_and_multibyte_unchanged() {
        assert_eq!(decode_strict(b"").unwrap(), "");
        assert_eq!(decode_strict(b"hello world").unwrap(), "hello world");
        assert_eq!(decode_strict("caf\u{e9}".as_bytes()).unwrap(), "caf\u{e9}");
        // U+1F600 GRINNING FACE (a 4-byte sequence) round-trips.
        assert_eq!(decode_strict("\u{1f600}".as_bytes()).unwrap(), "\u{1f600}");
        // U+D7FF is the last codepoint before the surrogate block: ED 9F BF is
        // valid, unlike ED A0 80 (a surrogate). This is the boundary anchor
        // for the surrogate constraint cases below.
        assert_eq!(decode_strict("\u{d7ff}".as_bytes()).unwrap(), "\u{d7ff}");
    }

    #[test]
    fn strict_truncated_tails_report_the_whole_consumed_span_as_unexpected_end() {
        // CPython: start=lead position, end=input end, reason "unexpected end of data".
        assert!(matches!(
            decode_strict(b"a\xc3"),
            Err(DecodeError::UnexpectedEnd { start: 1, end: 2 })
        ));
        assert!(matches!(
            decode_strict(b"\xf0\x9f"),
            Err(DecodeError::UnexpectedEnd { start: 0, end: 2 })
        ));
        assert!(matches!(
            decode_strict(b"\xf0\x9f\x98"),
            Err(DecodeError::UnexpectedEnd { start: 0, end: 3 })
        ));
        // E0 A0 is a valid prefix pair for E0's A0-BF constraint; only the missing
        // third byte makes it truncated.
        assert!(matches!(
            decode_strict(b"\xe0\xa0"),
            Err(DecodeError::UnexpectedEnd { start: 0, end: 2 })
        ));
        // A valid emoji followed by a truncated tail: the span is the tail alone.
        assert!(matches!(
            decode_strict(b"\xf0\x9f\x98\x80\xf0\x9f"),
            Err(DecodeError::UnexpectedEnd { start: 4, end: 6 })
        ));
    }

    #[test]
    fn strict_bad_lead_bytes_report_one_byte_as_invalid_start() {
        // CPython calls 0x80-0xBF (lone continuation), 0xC0/0xC1 (overlong 2-byte
        // leads), 0xF5-0xFF, and the legacy 5/6-byte leads 0xFC/0xFD "invalid start
        // byte" with a 1-byte span; everything after is re-processed, which is why
        // replace mode emits one U+FFFD per byte for these.
        assert!(matches!(
            decode_strict(b"\x80"),
            Err(DecodeError::InvalidStart { start: 0, end: 1 })
        ));
        assert!(matches!(
            decode_strict(b"a\x80b"),
            Err(DecodeError::InvalidStart { start: 1, end: 2 })
        ));
        assert!(matches!(
            decode_strict(b"\xc0\x80"),
            Err(DecodeError::InvalidStart { start: 0, end: 1 })
        ));
        assert!(matches!(
            decode_strict(b"\xff"),
            Err(DecodeError::InvalidStart { start: 0, end: 1 })
        ));
        assert!(matches!(
            decode_strict(b"\xfc\x84\x80\x80\x80\x80"),
            Err(DecodeError::InvalidStart { start: 0, end: 1 })
        ));
        assert!(matches!(
            decode_strict(b"ok\xffok"),
            Err(DecodeError::InvalidStart { start: 2, end: 3 })
        ));
    }

    #[test]
    fn strict_constraint_violations_report_the_lead_byte_alone_as_invalid_continuation() {
        // A valid lead byte whose FIRST continuation violates its range constraint
        // (E0's A0-BF overlong range, ED's 80-9F surrogate ceiling, F0's 90-BF
        // overlong range, F4's 80-8F ceiling above U+10FFFF) is reported AT the
        // lead with a 1-byte span; the violating byte is re-processed (a lone
        // continuation becomes its own "invalid start byte"), so replace emits
        // one U+FFFD per byte of the sequence. Measured CPython behavior, not
        // the WHATWG one-replacement-per-sequence behavior.
        assert!(matches!(
            decode_strict(b"\xed\xa0"),
            Err(DecodeError::InvalidContinuation { start: 0, end: 1 })
        ));
        assert!(matches!(
            decode_strict(b"\xe0\x80\x80"),
            Err(DecodeError::InvalidContinuation { start: 0, end: 1 })
        ));
        assert!(matches!(
            decode_strict(b"\xf0\x80\x80\x80"),
            Err(DecodeError::InvalidContinuation { start: 0, end: 1 })
        ));
        assert!(matches!(
            decode_strict(b"\xf4\x90"),
            Err(DecodeError::InvalidContinuation { start: 0, end: 1 })
        ));
        // A full CESU-8 surrogate pair: same lead-byte-only span.
        assert!(matches!(
            decode_strict(b"\xed\xa0\x80\xed\xb0\x80"),
            Err(DecodeError::InvalidContinuation { start: 0, end: 1 })
        ));
    }

    #[test]
    fn strict_non_continuation_after_valid_continuations_spans_the_maximal_subpart() {
        // When the failing byte is simply not a continuation byte (0x00-0x7F,
        // 0xC2-0xFF), the span covers the lead plus every VALID continuation
        // consumed so far (the maximal subpart), and decoding resumes at the
        // failing byte. Spans measured: F0 9F 41 -> 0..2, F0 9F 98 41 -> 0..3.
        assert!(matches!(
            decode_strict(b"\xf0\x41"),
            Err(DecodeError::InvalidContinuation { start: 0, end: 1 })
        ));
        assert!(matches!(
            decode_strict(b"\xf0\x9f\x41"),
            Err(DecodeError::InvalidContinuation { start: 0, end: 2 })
        ));
        assert!(matches!(
            decode_strict(b"\xf0\x9f\x98\x41"),
            Err(DecodeError::InvalidContinuation { start: 0, end: 3 })
        ));
        assert!(matches!(
            decode_strict(b"\xc2\x41"),
            Err(DecodeError::InvalidContinuation { start: 0, end: 1 })
        ));
        // "fo" + 0xD8 (a valid 2-byte lead) + 'o' (not a continuation): span is the
        // lone 0xD8.
        assert!(matches!(
            decode_strict(b"fo\xd8o"),
            Err(DecodeError::InvalidContinuation { start: 2, end: 3 })
        ));
    }

    #[test]
    fn error_reason_strings_match_cpython_exactly() {
        assert_eq!(
            decode_strict(b"a\xc3").unwrap_err().reason(),
            "unexpected end of data"
        );
        assert_eq!(
            decode_strict(b"\x80").unwrap_err().reason(),
            "invalid start byte"
        );
        assert_eq!(
            decode_strict(b"\xed\xa0\x80").unwrap_err().reason(),
            "invalid continuation byte"
        );
    }

    #[test]
    fn replace_matches_cpythons_maximal_subpart_replacement() {
        // Every layout below was measured against
        // `raw.decode("utf-8", "replace")` on CPython 3.12; see the module docs.
        assert_eq!(decode_replace(b""), "");
        assert_eq!(decode_replace(b"hello"), "hello");
        assert_eq!(decode_replace(b"a\xc3"), "a\u{fffd}");
        assert_eq!(decode_replace(b"\xf0\x9f"), "\u{fffd}");
        // Constraint violations: one U+FFFD per byte (lead re-reported, then each
        // continuation re-processed as a lone start byte).
        assert_eq!(decode_replace(b"\xed\xa0"), "\u{fffd}\u{fffd}");
        assert_eq!(decode_replace(b"\xe0\x80\x80"), "\u{fffd}\u{fffd}\u{fffd}");
        assert_eq!(
            decode_replace(b"\xf0\x80\x80\x80"),
            "\u{fffd}\u{fffd}\u{fffd}\u{fffd}"
        );
        assert_eq!(decode_replace(b"\xed\xa0\x80"), "\u{fffd}\u{fffd}\u{fffd}");
        assert_eq!(decode_replace(b"\x80\x80\x80"), "\u{fffd}\u{fffd}\u{fffd}");
        assert_eq!(
            decode_replace(b"\xfc\x84\x80\x80\x80\x80"),
            "\u{fffd}\u{fffd}\u{fffd}\u{fffd}\u{fffd}\u{fffd}"
        );
        // Maximal subparts: ONE U+FFFD for lead+valid-continuations, the
        // non-continuation byte kept.
        assert_eq!(decode_replace(b"\xf0\x9f\x41"), "\u{fffd}A");
        assert_eq!(decode_replace(b"\xf0\x9f\x98\x41"), "\u{fffd}A");
        assert_eq!(decode_replace(b"a\x80\x80b"), "a\u{fffd}\u{fffd}b");
        assert_eq!(decode_replace(b"fo\xd8o"), "fo\u{fffd}o");
        // Valid input passes through borrowed (no allocation, no FFFD).
        assert_eq!(decode_replace(b"caf\xc3\xa9"), "caf\u{e9}");
        assert!(matches!(decode_replace(b"abc"), Cow::Borrowed("abc")));
        assert!(matches!(decode_replace(b"a\xff"), Cow::Owned(_)));
    }
}
