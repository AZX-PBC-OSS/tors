//! CPython-parity percent encoding: the pure-Rust core of `tors.quote`,
//! `tors.quote_plus`, `tors.unquote`, and `tors.unquote_plus`, the
//! `urllib.parse` pair CPython implements in PURE PYTHON (Lib/urllib/parse.py,
//! `quote`/`quote_from_bytes`/`quote_plus`/`unquote`/`unquote_plus`). The
//! whole-text pass holds the GIL from end to end, which makes it the
//! single most-used encoding operation in web/ingestion pipelines and a
//! worst offender for the crate's GIL model. The semantics below are
//! derived from that source, including its quirks, and pinned against
//! the RUNNING interpreter per CI leg; the crate-side battery mirrors the
//! literals the Python gate differentials over.
//!
//! # Parity contract
//!
//! `quote(text, safe)` equals `urllib.parse.quote(text, safe)` and
//! `quote_plus(text, safe)` equals `urllib.parse.quote_plus(text, safe)`
//! for `str` inputs, utf-8 only: the never-quoted set is the RFC 3986
//! unreserved characters (`A-Z a-z 0-9 _ . - ~`, the stdlib's
//! `_ALWAYS_SAFE`) plus the ASCII members of `safe`; every other byte of
//! the input's utf-8 encoding becomes `%XX` with UPPERCASE hex.
//! `unquote(text)` equals `urllib.parse.unquote(text)` with the default
//! `encoding='utf-8', errors='replace'`, and `unquote_plus(text)` equals
//! `unquote_plus(text)` with the same defaults. The stdlib's `encoding`/
//! `errors` parameters are OUT OF SCOPE and documented as such: the encode
//! side is always utf-8-strict (an in-memory `&str` cannot fail to encode),
//! the decode side always utf-8-replace.
//!
//! # Defaults are the pyo3 layer's
//!
//! This core takes `safe` explicitly. The stdlib defaults (`safe='/'` for
//! `quote`, `safe=''` for `quote_plus`) are pinned by the pyo3 wrappers,
//! which own every argument default; the core itself is default-free.
//!
//! # Identity lanes (`Cow`)
//!
//! All four functions return `Cow<'_, str>`: the pyo3 layer's identity
//! contract (`f(s) is s` iff `f(s) == s`) wants the `Borrowed` lane, and
//! each lane is the cheapest true "nothing changes" test available.
//! `quote`/`quote_plus` borrow when no byte needs encoding (output is then
//! byte-equal to the input, so the iff holds both ways). `unquote` borrows
//! when the input contains no `%`: exactly CPython's `'%' not in string`
//! early return, which is where it returns the ORIGINAL object; note the
//! asymmetric residue: an input whose every `%` is invalid hex (e.g.
//! `"%zz"`) decodes to an EQUAL but NEW string in CPython, so the core
//! mirrors that with `Owned` rather than widening the borrow lane past
//! parity. `unquote_plus` borrows when the input contains neither `+` nor
//! `%`.
//!
//! # Stdlib quirks pinned here (a naive RFC 3986 implementation gets
//! these wrong)
//!
//! * `safe` is BYTE-level and ASCII-only: the stdlib normalizes str `safe`
//!   with `safe.encode('ascii', 'ignore')`, silently DROPPING non-ASCII
//!   members: `quote("é", "é")` is `"%C3%A9"`, not `"é"`. (A byte-table
//!   built from `safe.as_bytes()` without the `< 0x80` filter would let
//!   the utf-8 bytes of a non-ASCII `safe` char leak in and keep it
//!   unquoted: the exact wrong answer.)
//! * `%` in `safe` is honored like any other byte: it stays literal.
//! * `quote_plus` is NOT "quote then swap `%20` for `+`": the stdlib
//!   quotes with `' '` APPENDED to `safe` (so spaces never encode at all)
//!   and then replaces every `' '` with `'+'`: same output for valid
//!   input, but it also means a literal `'+'` in the text is escaped to
//!   `%2B` unless the CALLER put `+` in `safe`.
//! * `unquote` accepts LOWERCASE hex (`%c3%a9` → `é`), not just the
//!   uppercase a quoter emits.
//! * A `%` not followed by two hex digits: `%zz`, a trailing `%`, `%e`
//!   at end of input, `%%41`'s first pair: stays VERBATIM; `%%41` is
//!   `"%A"`, not `"%A"`-by-way-of-`%25`-then-reparse or an error.
//! * The stdlib unquotes and utf-8-decodes each MAXIMAL ASCII RUN of the
//!   input INDEPENDENTLY (its `_asciire` fragmentation), passing non-ASCII
//!   segments through verbatim. Consequences a whole-string decoder gets
//!   wrong: a multi-byte escape INTERRUPTED by a non-ASCII character is
//!   not an escape at all (`unquote("%Cé3")` → `"%Cé3"`), and an escape
//!   split across an ASCII/non-ASCII boundary decodes as TWO fragments
//!   each with their own `errors='replace'` verdict
//!   (`unquote("%C3é%A9")` → `"\u{FFFD}é\u{FFFD}"`, NOT `"éé"`).
//! * Invalid utf-8 from escapes decodes with the replace handler:
//!   `%e2%28%a1` → `"\u{FFFD}(\u{FFFD}"`, via
//!   [`crate::decode_impl::decode_replace`], whose maximal-subpart
//!   substitution was already measured byte-exact against CPython's
//!   `decode("utf-8", "replace")` for this crate's `decode_utf8` family.
//! * `unquote_plus` replaces `'+'` with `' '` BEFORE unquoting, so an
//!   escaped `%2B` survives as a literal `'+'` (`unquote_plus("%2B")` →
//!   `"+"`) while a raw `'+'` becomes a space: the order is the
//!   semantics, not an implementation detail.
//! * Empty inputs are empty outputs on every path, with no special-casing
//!   beyond the empty fast lanes.
//!
//! Pure Rust, no pyo3 types: the pyo3 wrappers in `lib.rs` add only the
//! argument borrow, the default `safe` values, and the return marshalling
//! (a `Cow::Borrowed` returns the original `PyObject` itself, which is the
//! identity contract above).

use std::borrow::Cow;

use memchr::memchr;

use crate::decode_impl::decode_replace;

/// The never-quoted byte set, the stdlib's `_ALWAYS_SAFE`: RFC 3986
/// unreserved (ASCII alphanumerics plus `_ . - ~`). Bytes >= 0x80 are
/// never members: the table below is only filled for the ASCII range,
/// which is also what keeps `byte as char` below exact.
const ALWAYS_SAFE: [bool; 256] = {
    let mut table = [false; 256];
    let mut byte = 0usize;
    while byte < 128 {
        let b = byte as u8;
        table[byte] = b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'~');
        byte += 1;
    }
    table
};

const HEX_UPPER: [u8; 16] = *b"0123456789ABCDEF";

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// The two bytes after the `%` at `at`, as one byte: `None` when either
/// is missing or not a hex digit (case-insensitive, the stdlib's
/// `_hexdig` accepts both cases).
fn hex_pair(run: &[u8], at: usize) -> Option<u8> {
    let hi = hex_val(*run.get(at + 1)?);
    let lo = hex_val(*run.get(at + 2)?);
    Some(hi? << 4 | lo?)
}

/// `urllib.parse.quote(text, safe)` for `str` input, utf-8 only.
///
/// ASCII letters/digits/`_.-~` plus `safe`'s ASCII members pass through;
/// every other utf-8 byte of `text` becomes `%XX` uppercase hex.
/// Non-ASCII members of `safe` are ignored (the stdlib's
/// `encode('ascii', 'ignore')` normalization). Returns
/// [`Cow::Borrowed`] when no byte needs encoding.
pub fn quote<'a>(text: &'a str, safe: &str) -> Cow<'a, str> {
    let mut table = ALWAYS_SAFE;
    for &byte in safe.as_bytes() {
        if byte.is_ascii() {
            table[byte as usize] = true;
        }
    }
    let bytes = text.as_bytes();
    // Everything before the first encoding byte is ASCII (table-true
    // implies < 0x80), so this is a char boundary; and the no-hit case
    // is the identity lane.
    let Some(first) = bytes.iter().position(|&byte| !table[byte as usize]) else {
        return Cow::Borrowed(text);
    };
    let mut out = String::with_capacity(bytes.len() + bytes.len() / 3);
    out.push_str(&text[..first]);
    for &byte in &bytes[first..] {
        if table[byte as usize] {
            out.push(byte as char);
        } else {
            out.push('%');
            out.push(HEX_UPPER[(byte >> 4) as usize] as char);
            out.push(HEX_UPPER[(byte & 0x0f) as usize] as char);
        }
    }
    Cow::Owned(out)
}

/// `urllib.parse.quote_plus(text, safe)` for `str` input, utf-8 only.
///
/// The stdlib's exact spelling: when the text contains no space this is
/// plain [`quote`]; otherwise it quotes with `' '` appended to `safe`
/// (spaces never encode) and then replaces every `' '` with `'+'`. A
/// literal `'+'` in the text is escaped to `%2B` unless the caller put
/// `'+'` in `safe`. Returns [`Cow::Borrowed`] exactly when [`quote`]
/// would (the space branch always rewrites at least one character).
///
/// The space branch costs two full passes and two allocations: the
/// `quote` call, then a second scan for the `' '` -> `'+'` replace:
/// instead of folding the swap into the encode loop. Left as is: the
/// realistic input for `quote_plus` is a URL component or a form field,
/// bytes to low kilobytes, where a second linear pass is noise next to
/// the `py.detach` and marshalling cost around it. A one-pass version
/// would need its own byte-table variant (push `'+'` in place of the
/// space-safe branch) purely for this rarer path; not worth the
/// duplication unless a profiled workload proves the two-pass cost
/// matters at the sizes this function actually sees.
pub fn quote_plus<'a>(text: &'a str, safe: &str) -> Cow<'a, str> {
    if !text.as_bytes().contains(&b' ') {
        return quote(text, safe);
    }
    let mut safe_with_space = String::with_capacity(safe.len() + 1);
    safe_with_space.push_str(safe);
    safe_with_space.push(' ');
    let quoted = quote(text, &safe_with_space).into_owned();
    Cow::Owned(quoted.replace(' ', "+"))
}

/// The stdlib's `_unquote_impl` over one maximal ASCII run: a `%` before
/// two hex digits contributes that byte, anything else stays verbatim.
/// A no-`%` run is the run itself (the `Cow` keeps that lane allocation
/// free: the bytes go straight through the decode as a borrow).
fn unquote_run(run: &[u8]) -> Cow<'_, [u8]> {
    let Some(first) = memchr(b'%', run) else {
        return Cow::Borrowed(run);
    };
    let mut out = Vec::with_capacity(run.len());
    out.extend_from_slice(&run[..first]);
    let mut pos = first;
    while let Some(hit) = memchr(b'%', &run[pos..]) {
        let pct = pos + hit;
        out.extend_from_slice(&run[pos..pct]);
        match hex_pair(run, pct) {
            Some(byte) => {
                out.push(byte);
                pos = pct + 3;
            }
            None => {
                out.push(b'%');
                pos = pct + 1;
            }
        }
    }
    out.extend_from_slice(&run[pos..]);
    Cow::Owned(out)
}

/// `urllib.parse.unquote(text)`: utf-8, `errors='replace'`.
///
/// Walks the input the way the stdlib's `_asciire` fragmentation does:
/// each maximal ASCII run is unquoted and then utf-8-decoded with the
/// replace handler INDEPENDENTLY, non-ASCII segments pass through
/// verbatim. A `%` without two hex digits stays verbatim. Returns
/// [`Cow::Borrowed`] when the input contains no `%`: CPython's own
/// early-return lane, where it returns the original object.
pub fn unquote(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();
    if memchr(b'%', bytes).is_none() {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(bytes.len());
    let mut cursor = 0;
    while cursor < bytes.len() {
        let ascii_run = bytes[cursor] < 0x80;
        let start = cursor;
        while cursor < bytes.len() && (bytes[cursor] < 0x80) == ascii_run {
            cursor += 1;
        }
        if ascii_run {
            out.push_str(&decode_replace(&unquote_run(&bytes[start..cursor])));
        } else {
            // Both ends of a non-ASCII run are char boundaries (an ASCII
            // byte always is; the first >= 0x80 after one is a lead byte),
            // so the slice is whole utf-8 chars passed through verbatim.
            out.push_str(&text[start..cursor]);
        }
    }
    Cow::Owned(out)
}

/// `urllib.parse.unquote_plus(text)`: utf-8, `errors='replace'`.
///
/// Every `'+'` becomes a space FIRST (the stdlib's ordering), so an
/// escaped `%2B` decodes to a literal `'+'` while a raw `'+'` becomes a
/// space. Returns [`Cow::Borrowed`] when the input contains neither `'+'`
/// nor `'%'`.
pub fn unquote_plus(text: &str) -> Cow<'_, str> {
    if memchr(b'+', text.as_bytes()).is_none() {
        return unquote(text);
    }
    let spaced = text.replace('+', " ");
    Cow::Owned(unquote(&spaced).into_owned())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;

    #[test]
    fn quote_leaves_unreserved_untouched_and_borrows() {
        let s = "AZaz09_.-~";
        let quoted = quote(s, "");
        assert_eq!(quoted, s);
        assert!(matches!(quoted, Cow::Borrowed(_)), "identity lane");
    }

    #[test]
    fn quote_encodes_space_and_reserved() {
        assert_eq!(quote("a b", ""), "a%20b");
        assert_eq!(quote("a/b", ""), "a%2Fb");
        assert_eq!(quote("50%", ""), "50%25");
        assert_eq!(quote("a?b#c", ""), "a%3Fb%23c");
        assert_eq!(quote("a+b", ""), "a%2Bb");
    }

    #[test]
    fn quote_encodes_non_ascii_as_utf8_bytes_uppercase() {
        assert_eq!(quote("é", ""), "%C3%A9");
        assert_eq!(quote("\u{1f600}", ""), "%F0%9F%98%80");
        assert_eq!(quote("café ☕", ""), "caf%C3%A9%20%E2%98%95");
    }

    #[test]
    fn quote_honors_safe_ascii_members_including_percent() {
        assert_eq!(quote("a/b", "/"), "a/b");
        assert_eq!(quote("50%", "%"), "50%");
        assert_eq!(quote("a b", " "), "a b");
        assert_eq!(quote("a/b?c", "/?"), "a/b?c");
        let borrowed = quote("a/b", "/");
        assert!(matches!(borrowed, Cow::Borrowed(_)), "all-safe borrows");
    }

    #[test]
    fn quote_ignores_non_ascii_safe_members() {
        // The stdlib normalizes safe with encode('ascii', 'ignore'): the
        // non-ASCII member is dropped, and its utf-8 bytes must not leak
        // into the byte table (the naive bug: é stays unquoted).
        assert_eq!(quote("é", "é"), "%C3%A9");
        assert_eq!(quote("aéb", "é/"), "a%C3%A9b");
    }

    #[test]
    fn quote_plus_is_space_safe_then_plus_swap() {
        assert_eq!(quote_plus("a b", ""), "a+b");
        assert_eq!(quote_plus("a b c", ""), "a+b+c");
        // A literal '+' encodes unless the caller safes it.
        assert_eq!(quote_plus("a+b", ""), "a%2Bb");
        assert_eq!(quote_plus("a+b", "+"), "a+b");
        // No space: identical to quote.
        assert_eq!(quote_plus("a/b", "/"), "a/b");
        assert_eq!(quote_plus("a/b", ""), "a%2Fb");
        // Non-ASCII still encodes; a non-ASCII safe member is still ignored.
        assert_eq!(quote_plus("é x", ""), "%C3%A9+x");
        assert_eq!(quote_plus("é", "é"), "%C3%A9");
    }

    #[test]
    fn empty_inputs_are_empty_outputs() {
        assert_eq!(quote("", ""), "");
        assert_eq!(quote_plus("", ""), "");
        assert_eq!(unquote(""), "");
        assert_eq!(unquote_plus(""), "");
        assert!(matches!(quote("", "/"), Cow::Borrowed(_)));
        assert!(matches!(unquote(""), Cow::Borrowed(_)));
        assert!(matches!(unquote_plus(""), Cow::Borrowed(_)));
    }

    #[test]
    fn unquote_decodes_valid_escapes_both_hex_cases() {
        assert_eq!(unquote("abc%20def"), "abc def");
        assert_eq!(unquote("%C3%A9"), "é");
        assert_eq!(unquote("%c3%a9"), "é");
        assert_eq!(unquote("%F0%9F%98%80"), "\u{1f600}");
        assert_eq!(unquote("%2F"), "/");
    }

    #[test]
    fn unquote_keeps_invalid_escapes_verbatim() {
        assert_eq!(unquote("%zz"), "%zz");
        assert_eq!(unquote("abc%"), "abc%");
        assert_eq!(unquote("%e"), "%e");
        assert_eq!(unquote("100%"), "100%");
        // %g is not hex: verbatim, and the 41 after it still decodes.
        assert_eq!(unquote("%g%41"), "%gA");
        // %%41: the first pair is '%' + '%' (not hex), verbatim; the
        // second % pairs with 41.
        assert_eq!(unquote("%%41"), "%A");
        // Verbatim-but-equal still allocates: CPython returns a NEW
        // string here (only the no-'%' lane returns the original object).
        assert!(matches!(unquote("%zz"), Cow::Owned(_)));
    }

    #[test]
    fn unquote_replaces_invalid_utf8_from_escapes() {
        assert_eq!(unquote("%ff"), "\u{fffd}");
        // E2 28 A1: maximal-subpart replacement, one U+FFFD per invalid
        // subpart with the valid '(' between: the replace handler's
        // exact output (lowercase hex accepted).
        assert_eq!(unquote("%e2%28%a1"), "\u{fffd}(\u{fffd}");
        // A truncated two-byte lead at end of input.
        assert_eq!(unquote("x%c3"), "x\u{fffd}");
    }

    #[test]
    fn unquote_fragments_at_non_ascii_and_decodes_runs_independently() {
        // A non-ASCII char interrupts an escape: it is not an escape.
        assert_eq!(unquote("%Cé3"), "%Cé3");
        // An escape split across an ASCII/non-ASCII boundary decodes as
        // two fragments, each with its own replace verdict: NOT éé.
        assert_eq!(unquote("%C3é%A9"), "\u{fffd}é\u{fffd}");
        // Non-ASCII text passes verbatim; escapes around it still decode.
        assert_eq!(unquote("é%41"), "éA");
        assert_eq!(unquote("%41é%42"), "AéB");
    }

    #[test]
    fn unquote_borrows_when_no_percent() {
        assert_eq!(unquote("a+b c/d"), "a+b c/d");
        assert!(matches!(unquote("a+b c/d"), Cow::Borrowed(_)));
    }

    #[test]
    fn unquote_plus_swaps_plus_before_unquoting() {
        assert_eq!(unquote_plus("a+b"), "a b");
        assert_eq!(unquote_plus("a+b%41"), "a bA");
        // The ordering: '+' becomes a space BEFORE unquote, so an escaped
        // %2B survives as a literal '+'.
        assert_eq!(unquote_plus("%2B"), "+");
        assert_eq!(unquote_plus("+%2B"), " +");
        assert_eq!(unquote_plus("%c3%a9+ok"), "é ok");
        // No '+' and no '%': borrowed identity.
        assert!(matches!(unquote_plus("abc"), Cow::Borrowed(_)));
    }

    #[test]
    fn round_trips_through_both_pairs() {
        let samples = [
            "",
            "hello world",
            "caf\u{e9} na\u{ef}ve",
            "\u{1f600} emoji \u{2603} snowman",
            "path/to/file?query=1&other=2",
            "50% plus + signs",
            "tilde~under_score.dot-dash",
            "latin \u{e0}\u{e9}\u{ee}\u{f2}\u{fb} extremes \u{ff}",
            "cjk \u{4e16}\u{754c} katakana \u{30ab}",
        ];
        for &s in &samples {
            assert_eq!(unquote(&quote(s, "")), s, "quote/unquote: {s:?}");
            assert_eq!(
                unquote(&quote(s, "/?")),
                s,
                "safe members round trip: {s:?}"
            );
            assert_eq!(
                unquote_plus(&quote_plus(s, "")),
                s,
                "quote_plus/unquote_plus: {s:?}"
            );
        }
    }

    #[test]
    fn sweep_agrees_with_independent_reference_encoder() {
        // Reference derived independently from the stdlib docstring, not
        // the implementation: unreserved chars pass, everything else is
        // the uppercase %XX of its utf-8 bytes.
        fn reference(c: char) -> String {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-' | '~') {
                c.to_string()
            } else {
                let mut out = String::new();
                for byte in c.to_string().as_bytes() {
                    out.push('%');
                    out.push(HEX_UPPER[(byte >> 4) as usize] as char);
                    out.push(HEX_UPPER[(byte & 0x0f) as usize] as char);
                }
                out
            }
        }
        for cp in 0..0x300u32 {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            let s = c.to_string();
            let encoded = quote(&s, "");
            assert_eq!(encoded, reference(c), "U+{cp:04X}");
            // And the decode direction closes the loop per character.
            assert_eq!(unquote(&encoded), s, "U+{cp:04X}");
        }
        // Spot rows beyond the sweep: BMP symbol, astral plane, the max.
        for c in ['\u{2028}', '\u{1f600}', '\u{10ffff}'] {
            let s = c.to_string();
            let encoded = quote(&s, "");
            assert_eq!(encoded, reference(c), "U+{:04X}", c as u32);
            assert_eq!(unquote(&encoded), s);
        }
    }

    #[test]
    fn long_mixed_input_round_trips_and_pins_structure() {
        // Hand-pinned mixed row: unreserved pass through, reserved and
        // non-ASCII encode, the safe members survive.
        let input = "user name+50%/café?page=1&x=~y#frag\u{1f600}";
        assert_eq!(
            quote(input, "/?&=#"),
            "user%20name%2B50%25/caf%C3%A9?page=1&x=~y#frag%F0%9F%98%80"
        );
        assert_eq!(
            quote_plus(input, ""),
            "user+name%2B50%25%2Fcaf%C3%A9%3Fpage%3D1%26x%3D~y%23frag%F0%9F%98%80"
        );
        // Quoted-printable-style bulk: a long mixed payload closes both
        // loops and never crashes the fragment walk.
        let mut bulk = String::new();
        for i in 0..2000 {
            bulk.push_str("field");
            bulk.push_str(&(i % 97).to_string());
            bulk.push_str(" value with spaces & symbols %\u{e9}\u{4e16}\u{1f600}+\r\n");
        }
        assert_eq!(unquote(&quote(&bulk, "")), bulk);
        assert_eq!(unquote_plus(&quote_plus(&bulk, "")), bulk);
        // Mixed valid/invalid escapes at bulk scale.
        let mixed = "%zz%C3%A9%%41%e%20%ffx".repeat(500);
        let decoded = unquote(&mixed);
        assert_eq!(decoded, "%zzé%A%e \u{fffd}x".repeat(500));
    }
}
