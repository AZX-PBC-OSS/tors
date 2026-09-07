//! CPython-parity HTML entity unescaping: the pure-Rust core of
//! `tors.html_unescape`.
//!
//! The algorithm is a port of CPython's `html.unescape` (Lib/html/__init__.py):
//! the `_charref` regex `&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)`
//! as a single left-to-right scan, plus `_replace_charref`'s classification:
//! exact-table lookup for named refs with the longest-matching-prefix fallback
//! (prefix lengths 2..len-1, remainder verbatim), and for numeric refs the
//! invalid-charref remap FIRST, then the surrogate/range guard to U+FFFD, then
//! the invalid-codepoint set to the empty string, else the codepoint. Both the
//! named table and the two numeric sets are the generated data in
//! `html_table.rs`, taken from the same interpreter the Python-side contract
//! gate (tests/test_html_unescape.py) verifies against per CI leg.
//!
//! The integer string conversion limit (backported to 3.10.7+ as part of the
//! CVE-2020-10735 fix, and in 3.11+ as `sys.get_int_max_str_digits()`,
//! default 4300): `_replace_charref`'s `int()` raises a `ValueError` for a
//! DECIMAL ref whose digit run exceeds the limit: BEFORE any classification.
//! `unescape_checked` replicates that: the limit is a per-call parameter the
//! pyo3 layer reads from the running interpreter under the GIL (a Python call
//! per REF would be a real cost on entity-dense text; per call it is
//! µs-scale), over-long decimal runs raise [`IntMaxStrDigits`] (counting the
//! run's full length, leading zeros included, exactly CPython's count), and
//! HEX refs are exempt (base 16 is a power of two; the limit applies only to
//! non-power-of-two bases; measured on 3.12.7 and 3.13.14,
//! arbitrarily long hex refs still classify). `None` is the no-limit spelling: 3.10.0–3.10.6
//! (the last legs without `sys.get_int_max_str_digits`; 3.10.7+ DO have the
//! limit and the attribute, so tors enforces it there too) and a
//! `sys.set_int_max_str_digits(0)`-disabled limit both map to it, matching
//! the running stdlib exactly. The message's WORDING is version-dependent:
//! 3.12+ and late 3.11.x say "Exceeds the limit (4300 digits) for integer
//! string conversion: value has 4301 digits; ..." while 3.10.7–3.11.x say
//! "Exceeds the limit (4300) for integer string conversion: ..." (no
//! "digits" after the limit), which is why the pyo3 layer does NOT format
//! the message itself: it replays the interpreter's own `int()` over the
//! error's [`IntMaxStrDigits::digit_run`] and raises the very `ValueError`
//! that returns, exact on every interpreter; `IntMaxStrDigits::message()` is
//! only the fallback for when the replay cannot run.
//!
//! Crate decision, measured (the same gate, pre-implementation, over 4505
//! oracle cases): `htmlescape` 0.3.6 fails 3609 (incomplete table, hard
//! errors on legacy without-semicolon refs); `html_escape` 0.2 fails 1495
//! (without-semicolon refs left verbatim, multi-char values truncated:
//! `&acE;` loses its combining U+0333, and WHATWG numeric semantics where
//! CPython's differ). Neither achieves parity, hence this port.
//!
//! Pure Rust, no pyo3 types: the pyo3 wrapper in `lib.rs` adds only the
//! argument borrow, the per-call limit read, and return marshalling (see the
//! crate GIL model there); the criterion bench drives this path directly.

use std::borrow::Cow;

use memchr::memchr;

use crate::html_table::{HTML5_ENTITIES, INVALID_CHARREFS, INVALID_CODEPOINTS};

/// The one error `unescape_checked` can raise: a decimal numeric reference
/// whose digit run exceeds the integer string conversion limit in force for
/// this call. Carries the limit plus the offending digit run itself.
///
/// The run can be arbitrarily long (attacker input): carrying it is one
/// O(run) copy on the ERROR path only, linear, the same order as the input
/// the caller already supplied, and CPython's own `html.unescape`
/// materializes the same run inside `int()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntMaxStrDigits {
    /// The limit in force (read from the running interpreter by the pyo3
    /// layer, once per call).
    pub limit: usize,
    /// The offending digit run, verbatim (ASCII digits): the payload the
    /// pyo3 layer replays through the running interpreter's `int()` to
    /// raise THAT interpreter's exact ValueError.
    pub digit_run: String,
}

impl IntMaxStrDigits {
    /// The run's FULL length: leading zeros included, exactly CPython's
    /// count. The run is pure ASCII digits, so its byte length IS its char
    /// length.
    pub fn digits(&self) -> usize {
        self.digit_run.len()
    }

    /// The FALLBACK message: the 3.12+/late-3.11 wording of the `ValueError`
    /// CPython's `int()` raises (the shape measured on 3.12.7 and 3.13.14,
    /// identical). Used only when the pyo3 layer's `int()` replay of
    /// [`Self::digit_run`] cannot run: e.g. the limit was disabled
    /// concurrently mid-call and `int()` unexpectedly succeeds; the primary
    /// path raises the interpreter's own exact `ValueError` verbatim.
    pub fn message(&self) -> String {
        format!(
            "Exceeds the limit ({} digits) for integer string conversion: \
             value has {} digits; use sys.set_int_max_str_digits() to increase the limit",
            self.limit,
            self.digits()
        )
    }
}

/// The name-character exclusion set of the `_charref` regex's third
/// alternative: a name is up to 32 CHARS of anything EXCEPT these (note \r is
/// NOT excluded: it is a name char, matching the regex).
fn is_name_char(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\u{c}' | ' ' | '<' | '&' | '#' | ';')
}

/// Exact key lookup in the sorted table.
fn lookup_entity(key: &str) -> Option<&'static str> {
    HTML5_ENTITIES
        .binary_search_by(|probe| probe.0.cmp(key))
        .ok()
        .map(|index| HTML5_ENTITIES[index].1)
}

/// `_replace_charref`'s numeric classification: invalid-charref remap first,
/// then surrogate/out-of-range to U+FFFD, then invalid-codepoint to "", else
/// the codepoint itself. `num` may exceed U+10FFFF (the parse saturates
/// there); such values always classify to U+FFFD.
fn classify_numeric(num: u64) -> String {
    if let Ok(cp) = u32::try_from(num)
        && let Ok(index) = INVALID_CHARREFS.binary_search_by_key(&cp, |&(k, _)| k)
    {
        return INVALID_CHARREFS[index].1.to_string();
    }
    if (0xD800..=0xDFFF).contains(&num) || num > 0x10FFFF {
        return "\u{fffd}".to_string();
    }
    let cp = num as u32;
    if INVALID_CODEPOINTS.binary_search(&cp).is_ok() {
        return String::new();
    }
    // cp is a valid scalar value here: not a surrogate, not above U+10FFFF.
    char::from_u32(cp)
        .expect("classified codepoint is a valid scalar")
        .to_string()
}

/// Parse a digit run in `base`, stopping early once the value provably
/// classifies as out-of-range (it only grows, and every invalid-charref key
/// is far below U+10FFFF); exact for classification, immune to overflow on
/// arbitrarily long digit runs.
fn parse_saturating(digits: &str, base: u64) -> u64 {
    let mut value: u64 = 0;
    for digit in digits.chars() {
        let d = u64::from(
            digit
                .to_digit(base as u32)
                .expect("scan already validated the digit"),
        );
        value = value * base + d;
        if value > 0x10FFFF {
            return value;
        }
    }
    value
}

/// One matched reference after a `&`: how many BYTES of `rest` the group
/// consumed, and what it replaces to. `None` = no match at this position
/// (the `&` is emitted verbatim and the scan resumes one char later, exactly
/// as `re.sub` advances past a failed match attempt). `Err` = a decimal
/// numeric ref whose digit run exceeds the per-call limit (the integer
/// string conversion limit, 3.10.7+: raised before any classification; see
/// the module docs).
///
/// This is the `_charref` regex's alternation in scan form: the numeric
/// alternatives first (`#[0-9]+;?` and `#[xX][0-9a-fA-F]+;?`: the decimal
/// class gives up at the first non-digit, which is the "&#10FFFF;" quirk),
/// then the name class (`[^\t\n\f <&#;]{1,32};?`, up to 32 CHARS, `\r`
/// included as a name char, non-ASCII included).
fn try_match_ref(
    rest: &str,
    max_decimal_digits: Option<usize>,
) -> Result<Option<(usize, String)>, IntMaxStrDigits> {
    let bytes = rest.as_bytes();
    let Some(&first) = bytes.first() else {
        return Ok(None); // a '&' at end of input: nothing to match
    };
    if first == b'#' {
        let Some(&second) = bytes.get(1) else {
            return Ok(None); // a bare "&#": neither numeric alternative matches
        };
        // Hex needs at least one hex digit after the x/X marker; decimal
        // starts right after '#'. No digits → the numeric alternatives fail
        // and '#' cannot start a name: no match at all ("&#;", "&#x;").
        let (base, digits_at): (u64, usize) = if second == b'x' || second == b'X' {
            (16, 2)
        } else {
            (10, 1)
        };
        let digits_len = rest[digits_at..]
            .chars()
            .take_while(|&c| c.is_digit(base as u32))
            .map(char::len_utf8)
            .sum::<usize>();
        if digits_len == 0 {
            return Ok(None);
        }
        let digits = &rest[digits_at..digits_at + digits_len];
        // The integer string conversion limit (3.10.7+/3.11+), raised BEFORE
        // any classification (`int()` runs first in `_replace_charref`, so
        // neither the remap nor the range guard can rescue the value).
        // DECIMAL only: base 16 is a power of two and the limit applies to
        // non-power-of-two bases alone (measured; see the module docs). The
        // count is the run's full length: leading zeros included, exactly
        // CPython's count (measured: a run of 4800 zeros reports 4800). The
        // error carries the run verbatim so the pyo3 layer can replay the
        // running interpreter's own `int()` over it (see the struct docs).
        if base == 10
            && let Some(limit) = max_decimal_digits
            && digits_len > limit
        {
            return Err(IntMaxStrDigits {
                limit,
                digit_run: digits.to_string(),
            });
        }
        let mut consumed = digits_at + digits_len;
        if bytes.get(consumed) == Some(&b';') {
            consumed += 1;
        }
        let num = parse_saturating(digits, base);
        return Ok(Some((consumed, classify_numeric(num))));
    }

    // Named reference: up to 32 name chars, then an optional ';'.
    let mut name_end = 0;
    let mut name_chars = 0;
    for (index, c) in rest.char_indices() {
        if name_chars == 32 || !is_name_char(c) {
            break;
        }
        name_chars += 1;
        name_end = index + c.len_utf8();
    }
    if name_chars == 0 {
        return Ok(None);
    }
    let mut group_end = name_end;
    if bytes.get(name_end) == Some(&b';') {
        group_end += 1;
    }
    let group = &rest[..group_end];
    if let Some(value) = lookup_entity(group) {
        return Ok(Some((group_end, value.to_string())));
    }

    // Longest matching prefix, exactly Python's `for x in range(len(s)-1, 1,
    // -1)`: prefix CHAR lengths from char_len-1 down to 2 (the full group
    // was just tried; length-1 prefixes are never tried: no 1-char keys
    // exist), remainder verbatim. No hit at all → '&' + group verbatim.
    // Enumerated as one reverse pass over the char boundaries, with no Vec
    // collected per failed exact lookup: walking the boundaries backwards,
    // item i's byte index is the end of the prefix of char length
    // char_len-1-i, so the candidates appear longest-first with no
    // intermediate collection.
    let char_len = group.chars().count();
    if char_len >= 2 {
        for (i, (end, _)) in group.char_indices().rev().enumerate() {
            let prefix_len = char_len - 1 - i;
            if prefix_len < 2 {
                break;
            }
            if let Some(value) = lookup_entity(&group[..end]) {
                return Ok(Some((group_end, format!("{value}{}", &group[end..]))));
            }
        }
    }
    Ok(Some((group_end, format!("&{group}"))))
}

/// `html.unescape(text)`: see the module docs. Borrowed when the unescape
/// changes nothing: no `&` at all (CPython's own early bail), or ampersands
/// that all fail to decode (failed refs, bare `&`, degenerate shapes: the
/// rebuilt output equals the input byte-for-byte, so the input comes back
/// instead of a marshalled copy). A scanned rebuild otherwise.
///
/// The no-`&` bail is memchr (SIMD: measured 0.107ms vs 0.404ms for the
/// slice `contains` it replaced, and vs 2.394ms for a manual byte loop, on
/// 12 MiB prose; the no-amp bench cell measured 481µs -> 99µs end-to-end).
/// The BETWEEN-entity hop STAYS a plain byte loop: on the
/// dense entities corpus the gaps average ~14 bytes, and a measured A/B
/// (criterion, same box) showed memchr's per-call setup on slices that
/// short COSTS ~5% of the whole unescape: the conversion does not move
/// the needle where the needle is dense, only where it is absent or rare.
///
/// The NO-LIMIT spelling (the 3.10.0–3.10.6 / disabled-limit semantics) delegates to
/// [`unescape_checked`] with `None`, where the only error class is
/// unreachable: benches and crate tests drive this path; the pyo3 layer
/// calls the checked spelling with the running interpreter's limit.
pub fn unescape(text: &str) -> Cow<'_, str> {
    unescape_checked(text, None).expect("no limit set: the only error class is unreachable")
}

/// The limit-aware spelling the pyo3 layer calls: `max_decimal_digits` is the
/// running interpreter's `sys.get_int_max_str_digits()` (read once per call
/// under the GIL by the wrapper), where `None` means no limit: 3.10.0–3.10.6
/// (the legs without the attribute), or a
/// `sys.set_int_max_str_digits(0)`-disabled limit. Every other behavior
/// is [`unescape`]'s.
pub fn unescape_checked(
    text: &str,
    max_decimal_digits: Option<usize>,
) -> Result<Cow<'_, str>, IntMaxStrDigits> {
    let bytes = text.as_bytes();
    if memchr(b'&', bytes).is_none() {
        return Ok(Cow::Borrowed(text));
    }

    let mut out = String::with_capacity(text.len());
    let mut last_emit = 0;
    let mut search_from = 0;
    while let Some(offset) = bytes[search_from..].iter().position(|&b| b == b'&') {
        let amp = search_from + offset;
        // amp+1 is a char boundary (just after an ASCII '&').
        match try_match_ref(&text[amp + 1..], max_decimal_digits)? {
            Some((consumed, replacement)) => {
                out.push_str(&text[last_emit..amp]);
                out.push_str(&replacement);
                search_from = amp + 1 + consumed;
                last_emit = search_from;
            }
            None => {
                search_from = amp + 1;
            }
        }
    }
    out.push_str(&text[last_emit..]);
    if out == text {
        Ok(Cow::Borrowed(text))
    } else {
        Ok(Cow::Owned(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_generated_table_lengths_match_the_header_counts() {
        // The crate-side half of the generated-table pin. The Python gate
        // verifies the forward direction only: every entry of the running
        // interpreter's tables decodes right, so a regenerated
        // src/html_table.rs carrying stale keys the interpreter no longer
        // has still passes it (the reverse-containment gap). These count
        // pins tie the file to the numbers in its own header, so a
        // regeneration against a changed interpreter table is visible even
        // without running Python. When a future Python changes the tables,
        // update these numbers together with
        // `make gen-html-table`, as one change.
        assert_eq!(HTML5_ENTITIES.len(), 2231);
        let with_semi = HTML5_ENTITIES
            .iter()
            .filter(|(key, _)| key.ends_with(';'))
            .count();
        assert_eq!(
            (with_semi, HTML5_ENTITIES.len() - with_semi),
            (2125, 106),
            "the with/without-semicolon split drifted from the header's numbers"
        );
        assert_eq!(INVALID_CHARREFS.len(), 34);
        assert_eq!(INVALID_CODEPOINTS.len(), 126);
    }

    #[test]
    fn the_full_table_round_trips_every_entity() {
        for (key, value) in HTML5_ENTITIES {
            let input = format!("&{key}");
            assert_eq!(unescape(&input).as_ref(), *value, "ref {input:?}");
        }
    }

    #[test]
    fn no_ampersand_is_a_zero_copy_borrow() {
        assert!(matches!(
            unescape("plain text"),
            Cow::Borrowed("plain text")
        ));
        assert!(matches!(unescape(""), Cow::Borrowed("")));
        assert!(matches!(unescape("\u{e9}\u{1f600}"), Cow::Borrowed(_)));
    }

    #[test]
    fn named_refs_with_and_without_semicolon_both_decode() {
        assert_eq!(unescape("&amp;").as_ref(), "&");
        assert_eq!(unescape("&amp").as_ref(), "&");
        assert_eq!(unescape("&AMP;").as_ref(), "&");
        assert_eq!(unescape("&AMP").as_ref(), "&");
        // Case-sensitive: no such key, no prefix hit.
        assert_eq!(unescape("&Amp;").as_ref(), "&Amp;");
    }

    #[test]
    fn longest_prefix_fallback_appends_the_remainder_verbatim() {
        assert_eq!(unescape("&notit;").as_ref(), "\u{ac}it;");
        assert_eq!(unescape("&notit").as_ref(), "\u{ac}it");
        assert_eq!(unescape("&ampx;").as_ref(), "&x;");
        assert_eq!(unescape("&amper;&amper;").as_ref(), "&er;&er;");
        // '&' terminates the name chars: only "am" is the group.
        assert_eq!(unescape("&am&amp").as_ref(), "&am&");
        // 'there4;' exists as a key; 'there4' does not and no prefix hits.
        assert_eq!(unescape("&there4;").as_ref(), "\u{2234}");
        assert_eq!(unescape("&there4").as_ref(), "&there4");
    }

    #[test]
    fn the_name_cap_is_32_chars() {
        let long = format!("&{}", "a".repeat(40));
        assert_eq!(unescape(&long).as_ref(), long.as_str());
        let capped = format!("&{};", "a".repeat(32));
        assert_eq!(unescape(&capped).as_ref(), capped.as_str());
    }

    #[test]
    fn numeric_refs_decode_in_every_spelling() {
        assert_eq!(unescape("&#65;").as_ref(), "A");
        assert_eq!(unescape("&#x41;").as_ref(), "A");
        assert_eq!(unescape("&#X41;").as_ref(), "A");
        assert_eq!(unescape("&#41").as_ref(), ")");
        assert_eq!(unescape("&#00000065;").as_ref(), "A");
        assert_eq!(unescape("&#x00000041;").as_ref(), "A");
        // U+10FFFF is a NONCHARACTER (last two of the plane): Python maps it
        // to the empty string, not to the max scalar.
        assert_eq!(unescape("&#x10FFFF;").as_ref(), "");
    }

    #[test]
    fn numeric_classification_matches_cpython() {
        assert_eq!(unescape("&#128;").as_ref(), "\u{20ac}"); // Windows-1252 remap
        assert_eq!(unescape("&#0;").as_ref(), "\u{fffd}");
        assert_eq!(unescape("&#13;").as_ref(), "\r");
        assert_eq!(unescape("&#1;").as_ref(), ""); // invalid codepoint -> empty
        assert_eq!(unescape("&#xFFFE;").as_ref(), "");
        assert_eq!(unescape("&#xD800;").as_ref(), "\u{fffd}");
        assert_eq!(unescape("&#x110000;").as_ref(), "\u{fffd}");
        // Saturation: arbitrarily long digit runs classify out-of-range.
        let huge = format!("&#{};", "9".repeat(60));
        assert_eq!(unescape(&huge).as_ref(), "\u{fffd}");
        // The greedy-decimal quirk: the digit class stops at 'F'.
        assert_eq!(unescape("&#10FFFF;").as_ref(), "\nFFFF;");
        assert_eq!(unescape("&#0x41;").as_ref(), "\u{fffd}x41;");
    }

    #[test]
    fn degenerate_refs_stay_verbatim() {
        for text in ["&#;", "&#x;", "&;", "&", "&# 65;", "& #65;", "&x;", "&#g;"] {
            assert_eq!(unescape(text).as_ref(), text, "{text:?}");
        }
    }

    #[test]
    fn adjacent_entities_and_the_single_pass_guarantee() {
        assert_eq!(unescape("&&amp;;").as_ref(), "&&;");
        assert_eq!(unescape("&amp&amp").as_ref(), "&&");
        assert_eq!(unescape("&#65&#66;").as_ref(), "AB");
        assert_eq!(unescape("a&amp;b").as_ref(), "a&b");
        // Replacements are never re-scanned.
        assert_eq!(unescape("&#38;amp;Dangerous").as_ref(), "&amp;Dangerous");
    }

    #[test]
    fn multibyte_text_around_entities_survives_the_scan() {
        assert_eq!(unescape(" &amp; ").as_ref(), " & ");
        assert_eq!(
            unescape("\u{e9}&amp;\u{1f600}").as_ref(),
            "\u{e9}&\u{1f600}"
        );
        // A non-ASCII char IS a name char per the regex's class; no key can
        // match it, so the group comes back verbatim.
        assert_eq!(unescape("&\u{e9};").as_ref(), "&\u{e9};");
    }

    #[test]
    fn every_numeric_sweep_case_from_the_oracle() {
        // The classification boundaries the Python-side gate sweeps; pinned
        // crate-side for the no-interpreter path (cargo test).
        for (input, expected) in [
            ("&#xD7FF;", "\u{d7ff}"),
            ("&#xE000;", "\u{e000}"),
            ("&#xFFFD;", "\u{fffd}"),
            ("&#127;", ""),
            ("&#0x41;", "\u{fffd}x41;"),
            ("&#x10fffe;", ""),
            ("&#xfffd;", "\u{fffd}"),
        ] {
            assert_eq!(unescape(input).as_ref(), expected, "{input:?}");
        }
        // Every member of both numeric sets, in both spellings: respecting
        // the classification ORDER: an invalid_charrefs remap (the C1 range
        // 0x80-0x9F is in BOTH sets) wins over the invalid-codepoint-to-empty
        // mapping, exactly as _replace_charref checks them.
        for &(cp, value) in INVALID_CHARREFS {
            for input in [format!("&#{cp};"), format!("&#x{cp:x};")] {
                assert_eq!(unescape(&input).as_ref(), value, "{input:?}");
            }
        }
        for &cp in INVALID_CODEPOINTS {
            let expected = match INVALID_CHARREFS.binary_search_by_key(&cp, |&(k, _)| k) {
                Ok(index) => INVALID_CHARREFS[index].1,
                Err(_) => "",
            };
            let input = format!("&#{cp};");
            assert_eq!(unescape(&input).as_ref(), expected, "{input:?}");
        }
    }

    // --- the integer string conversion limit (unescape_checked) ----------
    //
    // The Python-side gate (tests/test_html_unescape.py::
    // TestIntegerParseLimit) pins the boundary against the RUNNING
    // interpreter at the default 4300 and at a lowered limit; these crate
    // rows pin the core mechanics without an interpreter, using small limits
    // and the default-shape (fallback) message rendering.

    #[test]
    fn over_limit_decimal_refs_raise_with_cpythons_message_shape() {
        let limit = 100;
        let at_limit = format!("&#{};", "9".repeat(limit));
        let over = format!("&#{};", "9".repeat(limit + 1));
        // At-limit classifies normally (out of range → U+FFFD); over-limit
        // raises carrying the limit and the run itself.
        assert_eq!(
            unescape_checked(&at_limit, Some(limit)).unwrap().as_ref(),
            "\u{fffd}"
        );
        assert_eq!(
            unescape_checked(&over, Some(limit)).unwrap_err(),
            IntMaxStrDigits {
                limit,
                digit_run: "9".repeat(limit + 1),
            }
        );
        // The fallback message shape (the 3.12+/late-3.11 wording, measured
        // on 3.12.7 and 3.13.14); the primary path is the pyo3 layer's
        // `int()` replay.
        assert_eq!(
            unescape_checked(&over, Some(limit)).unwrap_err().message(),
            "Exceeds the limit (100 digits) for integer string conversion: \
             value has 101 digits; use sys.set_int_max_str_digits() to \
             increase the limit"
        );
    }

    #[test]
    fn leading_zeros_count_and_the_raise_precedes_classification() {
        let limit = 100;
        // The count is the run's full length, zeros included: carried by the
        // run itself, so the zeros are in it verbatim.
        let zeros = format!("&#{};", "0".repeat(limit + 1));
        let err = unescape_checked(&zeros, Some(limit)).unwrap_err();
        assert_eq!(err.digits(), limit + 1);
        assert_eq!(err.digit_run, "0".repeat(limit + 1));
        // A value that would Windows-1252-remap (128, zero-padded past the
        // limit) raises too: int() runs before any classification.
        let would_remap = format!("&#{}128;", "0".repeat(limit));
        assert!(unescape_checked(&would_remap, Some(limit)).is_err());
    }

    #[test]
    fn hex_refs_are_exempt_from_the_limit() {
        // Base 16 is a power of two: no limit (measured on 3.12.7/3.13.14),
        // arbitrarily long hex refs still classify, out-of-range → U+FFFD.
        let limit = 100;
        let long_hex = format!("&#x{};", "f".repeat(limit + 500));
        assert_eq!(
            unescape_checked(&long_hex, Some(limit)).unwrap().as_ref(),
            "\u{fffd}"
        );
    }

    #[test]
    fn none_is_the_no_limit_spelling() {
        // Arbitrary-length decimal runs classify under None: the
        // 3.10.0–3.10.6 semantics, and the disabled-limit spelling on
        // 3.10.7+.
        let huge = format!("&#{};", "9".repeat(5000));
        assert_eq!(unescape_checked(&huge, None).unwrap().as_ref(), "\u{fffd}");
        assert_eq!(unescape(&huge).as_ref(), "\u{fffd}");
    }
}
