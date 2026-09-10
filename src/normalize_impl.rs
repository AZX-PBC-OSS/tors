use std::borrow::Cow;

use memchr::{memchr, memmem};
use sha2::{Digest, Sha256};
use unicode_normalization::{IsNormalized, UnicodeNormalization, is_nfc_quick};

/// Matches CPython's `str.isspace()` character set (Unicode `White_Space` plus the four
/// ASCII separator controls 0x1c-0x1f that Python treats as space but Unicode's formal
/// `White_Space` property excludes), so `.strip()` here drops exactly what Python's would.
pub(crate) fn is_py_whitespace(c: char) -> bool {
    matches!(
        c,
        '\u{09}'..='\u{0d}'
            | '\u{1c}'..='\u{1f}'
            | ' '
            | '\u{85}'
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
    )
}

/// Byte bounds of `s` with Python-whitespace stripped from both ends:
/// `(start, end)`, or `None` when every char is whitespace. Test-only since
/// v0.4 (production strips inside the scan): the reference-oracle test's
/// pure-Python-style pipeline spells its strip through it.
#[cfg(test)]
fn strip_bounds(s: &str) -> Option<(usize, usize)> {
    let start = s
        .char_indices()
        .find(|&(_, c)| !is_py_whitespace(c))
        .map(|(i, _)| i)?;
    let end = s
        .char_indices()
        .rev()
        .find(|&(_, c)| !is_py_whitespace(c))
        .map(|(i, c)| i + c.len_utf8())?;
    Some((start, end))
}

/// The strip as a borrowed slice: the shape this module's unit tests use in their
/// pure-Python-style reference pipeline. Production strips inside the scan (the
/// leading run is dropped at the first non-whitespace char, the trailing run by the
/// end-of-input truncate to `confirmed`), so this stays test-only.
#[cfg(test)]
fn py_strip(s: &str) -> &str {
    match strip_bounds(s) {
        Some((start, end)) => &s[start..end],
        None => "",
    }
}

/// Would the fold/drop/collapse/strip stages leave `text` byte-for-byte verbatim?
/// Each stage's fingerprint as a byte-level sentinel scan, all memchr/memmem
/// (SIMD): any CR (folding), a `[ \t]` directly before a newline (the drop
/// rule: the run's last member is the byte adjacent to the newline, so the
/// two 2-byte needles cover every run shape), a `\n\n\n` run (the collapse),
/// and a Python-whitespace char at either end (the strip). The NFC stage's
/// fingerprint is the quick check, consulted by the caller ([`is_identity`]).
fn scan_is_verbatim(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return true; // normalize("") == ""
    }
    if memchr(b'\r', bytes).is_some() {
        return false;
    }
    if memmem::find(bytes, b" \n").is_some() || memmem::find(bytes, b"\t\n").is_some() {
        return false;
    }
    if memmem::find(bytes, b"\n\n\n").is_some() {
        return false;
    }
    let first = text.chars().next().expect("checked non-empty");
    if is_py_whitespace(first) {
        return false;
    }
    let last = text.chars().next_back().expect("checked non-empty");
    !is_py_whitespace(last)
}

/// The identity probe: `(qc_yes, identity)`. `identity` says the complete
/// pipeline is a no-op on `text`: NFC quick-check Yes (the crate's
/// `is_nfc_quick`, the same property data CPython's
/// `unicodedata.normalize` fast path consults; 2.4ms on 12 MiB ASCII
/// prose, measured, against the 94ms collect it can skip) and every scan
/// stage's fingerprint absent ([`scan_is_verbatim`]). `qc_yes` is returned
/// alongside so the caller can reuse it for the scan-input-direct decision
/// instead of consulting the quick check twice. The pyo3 layer returns the
/// original input object on the identity path: zero allocation, zero
/// marshalling.
fn probe(text: &str) -> (bool, bool) {
    let qc_yes = is_nfc_quick(text.chars()) == IsNormalized::Yes;
    (qc_yes, qc_yes && scan_is_verbatim(text))
}

/// Hasher feed granularity: the integrated SHA-256 consumes the confirmed
/// output bytes in spans of at least this size (one `update` per 64 KiB at
/// 12 MiB, ~192 calls), so the per-char cost of finalize's hash is a
/// predictable branch while the bytes are still cache-hot from the scan that
/// wrote them: one pass, not a scan pass plus a digest pass.
const HASH_CHUNK: usize = 64 * 1024;

fn flush(out: &mut String, nl_run: &mut usize, pending_ws: &mut String) {
    if *nl_run > 0 {
        out.push_str(if *nl_run >= 3 { "\n\n" } else { "\n" });
        if *nl_run == 2 {
            out.push('\n');
        }
        *nl_run = 0;
    }
    if !pending_ws.is_empty() {
        out.push_str(pending_ws);
        pending_ws.clear();
    }
}

/// Single native-Rust pass over the NFC-normalized text doing what the pipeline's
/// original pure-Python spelling does with two whole-string-scale `re.sub` calls
/// (chunked in GIL-sensitive callers only to bound GIL-hold time under `re`, which never
/// releases the GIL regardless of input size):
/// fold CRLF/CR to LF, drop trailing `[ \t]+` runs immediately before a newline, and
/// collapse runs of 3+ newlines to exactly two. Folding, dropping, and collapsing are all
/// interleaved in one left-to-right scan because each only needs to know, at any position,
/// the run of pending newlines and the run of pending space/tab since the last newline.
///
/// v0.4, two structural changes with the same output bytes (the module's
/// reference-oracle battery and the Python-side exhaustive/hypothesis suites
/// pin the equivalence):
///
/// - The strip is fused into the scan instead of a post-pass: whitespace
///   emitted before the first non-whitespace char is dropped at that char
///   (the leading strip: `out.clear()`, free), and everything after the
///   last non-whitespace char is dropped by the end-of-input truncate to
///   `confirmed` (the trailing strip). `confirmed` is the byte length of the
///   output that can never be stripped: everything up to and including the
///   most recent non-whitespace char.
/// - An optional SHA-256 is fed the confirmed bytes as they are written, in
///   `HASH_CHUNK`-sized spans: finalize's digest is computed in the same pass
///   that builds the buffer, not a second walk of the finished output.
fn scan(text: &str, mut hasher: Option<Sha256>) -> (String, Option<[u8; 32]>) {
    let mut out = String::with_capacity(text.len());
    let mut pending_ws = String::new();
    let mut nl_run: usize = 0;
    let mut hashed: usize = 0;
    let mut confirmed: usize = 0;
    let mut saw_nonws = false;

    let mut chars = text.chars().peekable();
    while let Some(mut c) = chars.next() {
        if c == '\r' {
            if let Some('\n') = chars.peek() {
                chars.next();
            }
            c = '\n';
        }
        match c {
            ' ' | '\t' => pending_ws.push(c),
            '\n' => {
                pending_ws.clear();
                nl_run += 1;
            }
            _ => {
                // Exotic whitespace (everything is_py_whitespace covers beyond
                // ' ', '\t', '\n': the 0x1c-0x1f controls, NBSP, U+2028,
                // U+3000, ...) lives entirely outside 0x21..=0x7e, so the
                // common mid-line char is classified by one range comparison
                // and the full property match runs only for candidates.
                let is_ws = !('\u{21}'..='\u{7e}').contains(&c) && is_py_whitespace(c);
                if is_ws {
                    // Emitted like any char, but unconfirmed: it may yet prove
                    // to be part of the trailing strip.
                    flush(&mut out, &mut nl_run, &mut pending_ws);
                    out.push(c);
                } else {
                    if !saw_nonws {
                        // The first non-whitespace char: everything emitted or
                        // pending so far is leading whitespace (the leading
                        // strip, applied here and never hashed).
                        out.clear();
                        pending_ws.clear();
                        nl_run = 0;
                        hashed = 0;
                        saw_nonws = true;
                    } else {
                        flush(&mut out, &mut nl_run, &mut pending_ws);
                    }
                    out.push(c);
                    confirmed = out.len();
                    if let Some(h) = hasher.as_mut()
                        && confirmed - hashed >= HASH_CHUNK
                    {
                        h.update(&out.as_bytes()[hashed..confirmed]);
                        hashed = confirmed;
                    }
                }
            }
        }
    }
    // End of input: everything after the last non-whitespace char is the
    // trailing strip, dropped and never hashed. All-whitespace input (no
    // non-whitespace char ever seen) empties the output the same way.
    out.truncate(confirmed);
    let digest = hasher.map(|mut h| {
        if confirmed > hashed {
            h.update(&out.as_bytes()[hashed..confirmed]);
        }
        h.finalize().into()
    });
    (out, digest)
}

/// The shared pipeline driver behind [`normalize`] / [`normalize_cow`] (no
/// hasher) and `finalize_impl::finalize_checked` (hasher supplied): consult
/// the quick check once; a Yes-with-clean-scan input returns the input
/// borrowed with the digest (when hashing) computed straight from the input
/// bytes: no output allocation at all; a Yes-with-dirty-scan input skips the
/// NFC materialization and scans the input directly; anything else runs the
/// NFC pass first. A final output==input comparison extends the identity
/// return to quick-check-Maybe inputs the pass leaves unchanged, so the
/// caller-visible contract is complete: the same object comes back whenever
/// the pipeline changes nothing.
pub(crate) fn pipeline(text: &str, hasher: Option<Sha256>) -> (Cow<'_, str>, Option<[u8; 32]>) {
    let (qc_yes, identity) = probe(text);
    if identity {
        let digest = hasher.map(|_| Sha256::digest(text.as_bytes()).into());
        return (Cow::Borrowed(text), digest);
    }
    let (out, digest) = if qc_yes {
        scan(text, hasher)
    } else {
        scan(&text.nfc().collect::<String>(), hasher)
    };
    if out == text {
        (Cow::Borrowed(text), digest)
    } else {
        (Cow::Owned(out), digest)
    }
}

/// The v0.1 pipeline, identity-aware: `Cow::Borrowed(text)` exactly when the
/// transform changes nothing (the identity probe's fast lane, or the
/// output==input comparison after a full pass), `Cow::Owned` otherwise.
pub fn normalize_cow(text: &str) -> Cow<'_, str> {
    pipeline(text, None).0
}

/// The allocating spelling (the criterion benches drive this): always a
/// fresh `String`, so bench numbers measure the transform, not the identity
/// shortcut; the identity path's own numbers are the bench's
/// already-normalized cells over `normalize_cow`.
pub fn normalize(text: &str) -> String {
    normalize_cow(text).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string() {
        assert_eq!(normalize(""), "");
    }

    #[test]
    fn all_whitespace_collapses_to_empty() {
        assert_eq!(normalize("   \t\n\n\n   "), "");
    }

    #[test]
    fn crlf_and_lone_cr_fold_to_lf() {
        assert_eq!(normalize("a\r\nb\rc\nd"), "a\nb\nc\nd");
    }

    #[test]
    fn trailing_tab_before_newline_is_trimmed() {
        assert_eq!(normalize("a\t\nb\n\n\nc"), "a\nb\n\nc");
    }

    #[test]
    fn blank_run_of_three_collapses_to_two() {
        assert_eq!(normalize("a\n\n\nb"), "a\n\nb");
    }

    #[test]
    fn blank_run_of_two_is_untouched() {
        assert_eq!(normalize("a\n\nb"), "a\n\nb");
    }

    #[test]
    fn nfc_composes_decomposed_sequences() {
        assert_eq!(normalize("cafe\u{0301} \n\n\n\nwater"), "café\n\nwater");
    }

    #[test]
    fn entirely_decomposed_string_fully_composes() {
        let decomposed: String = "e\u{0301}".repeat(20);
        let composed: String = "\u{e9}".repeat(20);
        assert_eq!(normalize(&decomposed), composed);
    }

    #[test]
    fn all_whitespace_variants_collapse_to_empty() {
        for text in [
            "\n",
            "\n\n",
            "\n\n\n",
            " ",
            "\t",
            "\r",
            "\r\n",
            &" \t\r\n".repeat(10),
        ] {
            assert_eq!(normalize(text), "", "mismatch for {text:?}");
        }
    }

    #[test]
    fn mixed_crlf_cr_lf_every_line_ending_kind_in_one_string() {
        // \r\n, bare \r, and bare \n all appear, several times each, interleaved with
        // blank runs only visible once every ending is folded to \n.
        assert_eq!(
            normalize("a\r\nb\rc\nd\r\ne\r\rf\n\ng\r\n\r\n\r\nh"),
            "a\nb\nc\nd\ne\n\nf\n\ng\n\nh"
        );
    }

    #[test]
    fn multiple_consecutive_blank_run_collapses_stay_independent() {
        let text = (0..6)
            .map(|i| format!("para{i}"))
            .collect::<Vec<_>>()
            .join("\n\n\n");
        let expected = (0..6)
            .map(|i| format!("para{i}"))
            .collect::<Vec<_>>()
            .join("\n\n");
        assert_eq!(normalize(&text), expected);
    }

    #[test]
    fn very_long_single_whitespace_run_before_newline() {
        let text = format!("a{}\nb", " \t".repeat(50_000));
        assert_eq!(normalize(&text), "a\nb");
    }

    #[test]
    fn very_long_single_newline_run() {
        let text = format!("a{}b", "\n".repeat(100_000));
        assert_eq!(normalize(&text), "a\n\nb");
    }

    #[test]
    fn very_long_whitespace_run_at_end_of_string() {
        let text = format!("leading text{}", " \t\n".repeat(40_000));
        assert_eq!(normalize(&text), "leading text");
    }

    #[test]
    fn very_long_whitespace_run_from_start_of_string() {
        let text = format!("{}trailing text", " \t\n".repeat(40_000));
        assert_eq!(normalize(&text), "trailing text");
    }

    #[test]
    fn trailing_whitespace_with_no_final_newline_is_stripped() {
        assert_eq!(normalize("hello   "), "hello");
    }

    #[test]
    fn leading_whitespace_is_stripped() {
        assert_eq!(normalize("   hello"), "hello");
    }

    #[test]
    fn interleaved_whitespace_and_newlines_collapse_correctly() {
        // " \t" before the first "\n", then a bare "\n", then " " before the last "\n":
        // every space/tab run immediately preceding a newline is dropped, and the three
        // resulting newlines collapse to two.
        assert_eq!(normalize("a \t\n\n \nb"), "a\n\nb");
    }

    // --- v0.4: the identity probe and the Cow lanes ------------------------------

    /// The probe's `identity` verdict, spelled for readability: the same
    /// `(qc_yes, identity)` pair `pipeline` consults.
    fn is_identity(text: &str) -> bool {
        probe(text).1
    }

    #[test]
    fn identity_probe_matches_each_pipeline_stage_fingerprint() {
        for clean in [
            "",
            "plain text",
            "a\n\nb",               // a 2-newline run is untouched
            "caf\u{e9} na\u{ef}ve", // composed accents: quick-check Yes
            "mid  line   spaces",   // mid-line [ \t] runs are never dropped
            "x\u{a0}y",             // interior exotic whitespace is not stripped
        ] {
            assert!(is_identity(clean), "expected identity: {clean:?}");
        }
        for dirty in [
            "a\rb",   // lone CR folds
            "a\r\nb", // CRLF folds
            "a \nb",  // [ \t] before a newline drops
            "a\t\nb",
            "a \t\nb",    // the run's last member is the needle hit
            "a\n \nb",    // space before the second newline
            "a\n\n\nb",   // blank run collapses
            "  leading",  // strip delta, leading
            "trailing  ", // strip delta, trailing
            "trailing\n",
            "\u{a0}x", // exotic whitespace at the ends strips
            "x\u{a0}",
            "x\u{2028}",
            "cafe\u{0301}", // quick-check Maybe: NFC composes
        ] {
            assert!(!is_identity(dirty), "expected NOT identity: {dirty:?}");
        }
    }

    fn all_strings(alphabet: &[&str], max_len: usize) -> Vec<String> {
        let mut out = vec![String::new()];
        for _ in 0..max_len {
            let mut longer = Vec::with_capacity(out.len() * alphabet.len());
            for prefix in &out {
                for piece in alphabet {
                    longer.push(format!("{prefix}{piece}"));
                }
            }
            out.append(&mut longer);
        }
        out
    }

    #[test]
    fn identity_probe_implies_the_pipeline_is_the_identity() {
        // One-directional tautology pin over the trigger alphabet (the same
        // space the exhaustive oracle battery uses, plus the exotic spaces):
        // every probe-true input normalizes to itself. The converse is
        // not asserted: quick-check-Maybe inputs can also
        // normalize to themselves, which is the post-compare lane's job.
        let alphabet = [
            " ", "\t", "\n", "\r", "a", "e", "\u{0301}", "\u{e9}", "\u{a0}",
        ];
        for text in all_strings(&alphabet, 4) {
            if is_identity(&text) {
                assert_eq!(normalize(&text), text, "probe-true but dirty: {text:?}");
            }
        }
    }

    #[test]
    fn normalize_cow_borrows_identity_and_value_identity_inputs() {
        // The probe's fast lane: quick-check Yes + clean scan, borrowed with
        // no allocation at all.
        let clean = "plain text\n\nwith paragraphs";
        assert!(matches!(normalize_cow(clean), Cow::Borrowed(s) if s == clean));
        // The post-compare lane: quick-check Maybe (combining marks) but the
        // NFC pass leaves the text unchanged and the scan is clean, so the
        // input still comes back borrowed.
        let maybe_clean = "q\u{0328}\u{0301} text";
        assert!(!is_identity(maybe_clean));
        assert!(matches!(normalize_cow(maybe_clean), Cow::Borrowed(s) if s == maybe_clean));
        // Quick-check Yes but scan-dirty: an owned, different output.
        assert!(matches!(normalize_cow("a \nb"), Cow::Owned(s) if s == "a\nb"));
        assert!(matches!(normalize_cow("cafe\u{0301}"), Cow::Owned(s) if s == "caf\u{e9}"));
    }

    #[test]
    fn whitespace_not_before_a_newline_is_preserved() {
        assert_eq!(normalize("a\n\n\n  b"), "a\n\n  b");
    }

    #[test]
    fn mid_line_spaces_are_never_touched() {
        assert_eq!(normalize("a  b   c"), "a  b   c");
    }

    #[test]
    fn matches_a_pure_python_reference_pipeline() {
        // Mirrors the pipeline's original pure-Python spelling, unchunked, as an
        // independent oracle.
        fn reference(text: &str) -> String {
            let nfc: String = text.nfc().collect();
            let folded = nfc.replace("\r\n", "\n").replace('\r', "\n");
            let mut trimmed = String::new();
            let mut ws = String::new();
            for c in folded.chars() {
                match c {
                    ' ' | '\t' => ws.push(c),
                    '\n' => {
                        ws.clear();
                        trimmed.push('\n');
                    }
                    _ => {
                        trimmed.push_str(&ws);
                        ws.clear();
                        trimmed.push(c);
                    }
                }
            }
            trimmed.push_str(&ws);
            let mut collapsed = String::new();
            let mut run = 0usize;
            for c in trimmed.chars() {
                if c == '\n' {
                    run += 1;
                } else {
                    if run > 0 {
                        let piece = if run >= 3 {
                            "\n\n".to_string()
                        } else {
                            "\n".repeat(run)
                        };
                        collapsed.push_str(&piece);
                        run = 0;
                    }
                    collapsed.push(c);
                }
            }
            if run > 0 {
                let piece = if run >= 3 {
                    "\n\n".to_string()
                } else {
                    "\n".repeat(run)
                };
                collapsed.push_str(&piece);
            }
            py_strip(&collapsed).to_string()
        }

        let cases = [
            "",
            "plain text",
            "a\r\nb\rc\n\n\n\nd   \n",
            "  \t\n\n\n leading and trailing \t\n\n\n\n  ",
            "line1\nline2\n\n\n\n\nline3\t \t\n",
            "café\u{0301}\n\n\nnaïve",
        ];
        for case in cases {
            assert_eq!(normalize(case), reference(case), "mismatch for {case:?}");
        }
    }
}
