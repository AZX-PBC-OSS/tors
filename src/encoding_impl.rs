//! Legacy/unlabeled byte-content encoding detection: the pure-Rust core of
//! `tors.detect_encoding`.
//!
//! `chardetng` (Mozilla's detector, the one Firefox ships) over the whole
//! input in one `feed(..., last = true)` call, then `guess`. This is a
//! heuristic, not a validator: unlike `utf8_is_valid`/`decode_utf8`, there is
//! no ground truth to be byte-exact against, and the detector always returns
//! some encoding: confidence is not exposed by the crate beyond the single
//! best guess, so neither is it here. The intended pipeline shape is
//! `utf8_is_valid` first; only reach for `detect_encoding` on the bytes that
//! already failed that check, then decode with the returned codec name.
//!
//! Pure Rust, no pyo3 types: the pyo3 wrapper in `lib.rs` adds only the
//! zero-copy `PyBytes` argument borrow (see the crate GIL model there).

use chardetng::{EncodingDetector, Iso2022JpDetection, Utf8Detection};
use encoding_rs::Encoding;

/// `chardetng::EncodingDetector::guess`'s `tld` argument panics (an
/// `assert!`, not a `Result`) if it contains an uppercase letter, a period,
/// or any non-ASCII byte; so a caller passing the natural spellings
/// `".jp"` or `"JP"` (rather than the crate's exact expected `"jp"`) would
/// crash the whole call. Normalize defensively instead: lowercase, strip
/// one leading `.` (the shape everyone actually has on hand: `some.tld`
/// off a parsed URL, or a leading-dot literal), and if what's left still
/// trips the crate's own precondition (an embedded period from a
/// caller passing a full domain, non-ASCII, or empty), treat it as no hint
/// at all rather than propagate to the panicking assert: a bad hint
/// degrading to "no hint" is the correct heuristic-detector behavior; a
/// panic on ordinary caller input is not.
fn normalize_tld(tld: &str) -> Option<String> {
    let lower = tld.strip_prefix('.').unwrap_or(tld).to_ascii_lowercase();
    if lower.is_empty() || !lower.is_ascii() || lower.contains('.') {
        return None;
    }
    Some(lower)
}

/// `encoding_rs::Encoding::name()` returns WHATWG Encoding Standard labels,
/// which mostly double as valid Python codec names (`codecs.lookup` is
/// case/punctuation-insensitive) but not always: `"windows-874"` is the
/// one label chardetng's candidate set can actually produce that Python's
/// codec registry does not recognize under that spelling (it wants
/// `"cp874"`); this table exists for exactly that gap, not as a general
/// WHATWG→Python translator; and it's a Rust `&'static str` on both sides,
/// so an unmatched name still returns the original label unchanged.
fn python_codec_name(whatwg_name: &'static str) -> &'static str {
    match whatwg_name {
        "windows-874" => "cp874",
        // The logical-ordered Hebrew label; CPython's registry only knows
        // iso-8859-8 (the byte mapping is identical: the -I suffix only
        // signals logical bidi ordering to downstream renderers, which a
        // decode does not consult).
        "ISO-8859-8-I" => "ISO-8859-8",
        other => other,
    }
}

/// Best-guess encoding name for `raw`, as a Python `bytes.decode`-usable
/// codec name (see [`python_codec_name`] for the two labels this crate's
/// candidate set can produce that need translating). `tld` is an optional
/// top-level-domain hint that disambiguates language-family-ambiguous
/// input: accepted in whatever natural spelling a caller has (leading dot
/// or not, any case; see [`normalize_tld`]), degrading to "no hint" rather
/// than panicking on anything chardetng's own precondition would reject.
///
/// UTF-8 and ISO-2022-JP are both allowed guess results: this is a
/// general-purpose detector over arbitrary bytes, not a Web browser (whose
/// security posture is why the crate makes both of those opt-in rather than
/// default). Empty input guesses UTF-8: chardetng's own answer for a
/// zero-byte feed, and the least surprising one (empty bytes are trivially
/// valid UTF-8 already).
pub fn detect(raw: &[u8], tld: Option<&str>) -> &'static str {
    let mut detector = EncodingDetector::new(Iso2022JpDetection::Allow);
    detector.feed(raw, true);
    let normalized_tld = tld.and_then(normalize_tld);
    let tld_bytes = normalized_tld.as_deref().map(str::as_bytes);
    let encoding: &'static Encoding = detector.guess(tld_bytes, Utf8Detection::Allow);
    python_codec_name(encoding.name())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_guesses_utf8() {
        assert_eq!(detect(b"", None), "UTF-8");
    }

    #[test]
    fn plain_ascii_guesses_utf8() {
        assert_eq!(detect(b"hello, plain ASCII text.", None), "UTF-8");
    }

    #[test]
    fn well_formed_utf8_with_multibyte_content_guesses_utf8() {
        assert_eq!(detect("café société 日本語".as_bytes(), None), "UTF-8");
    }

    #[test]
    fn never_panics_on_arbitrary_bytes() {
        // A heuristic detector always answers
        // something for any byte string, valid UTF-8 or not.
        for raw in [
            &b"\xff\xfe\x00\x01"[..],
            &b"\x00\x00\x00\x00"[..],
            &(0u8..=255).collect::<Vec<u8>>()[..],
        ] {
            let _ = detect(raw, None);
        }
    }

    #[test]
    fn tld_hint_is_threaded_through_without_panicking() {
        let raw = b"\x82\xa0\x82\xa2\x82\xa4\x82\xa6\x82\xa8"; // Shift_JIS hiragana
        let without_hint = detect(raw, None);
        let with_hint = detect(raw, Some("jp"));
        // Both must be valid, non-empty guesses; the hint changing the
        // outcome on this specific ambiguous input is a bonus, not a
        // contract this test enforces (chardetng's disambiguation logic is
        // its own to evolve); the contract pinned here is "doesn't panic,
        // always returns a name".
        assert!(!without_hint.is_empty());
        assert!(!with_hint.is_empty());
    }

    #[test]
    fn tld_hint_never_panics_on_the_spellings_that_would_trip_the_crates_own_assert() {
        let raw = b"plain ascii text";
        for tld in [".jp", "JP", "Jp", "example.co.jp", "日本", ""] {
            let _ = detect(raw, Some(tld));
        }
    }

    #[test]
    fn windows_874_is_translated_to_the_python_codec_name() {
        // b'A\xa7\xa8', the exact hypothesis-shrunk input that surfaced this:
        // chardetng's WHATWG label "windows-874" is not a name Python's
        // codecs registry recognizes (it wants "cp874").
        assert_eq!(detect(b"A\xa7\xa8", None), "cp874");
    }
}
