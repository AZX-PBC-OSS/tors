//! The RFC 8259 validity scanner, the pure-Rust core of
//! `tors.json_is_valid` (#61): one iterative pass over the raw bytes that
//! answers "would a JSON parser accept this?" without building any object
//! tree — the validate-and-discard shape's primitive (the consumer's gate
//! sits in front of bytes that are parsed once and thrown away, where
//! ~95% of a full parse is constructing objects nobody reads).
//!
//! # The acceptance set is orjson's, deliberately
//!
//! The gate stands in front of a consumer that would call `orjson.loads`
//! when the bytes need to be USED; the validator's contract is to answer
//! `False` exactly when that call would raise, so the acceptance set is
//! matched to orjson 3.x (differentially probed, and pinned as tests in
//! tests/test_json_is_valid.py). Where RFC 8259 is silent or permissive,
//! orjson's reading wins, and every seam is documented in docs/api.md:
//!
//! * **Float-overflow literals reject** (`1e400`, `-1e400`, `1e309`,
//!   `2e308`, and integer literals past ~512 digits, which orjson parses
//!   as doubles): orjson raises `JSONDecodeError` ("number is infinity
//!   when parsed as double") where stdlib `json.loads` returns
//!   `inf`/`-inf`. The scanner computes the literal's f64 value with
//!   Rust's correctly-rounded parser and rejects a non-finite result —
//!   underflow (`1e-400` → `0.0`) is finite, so it accepts, matching
//!   orjson.
//! * **Integers**: ≤ 19 digits always accept (orjson builds a Python
//!   int); 20+ digits accept exactly when the literal's f64 value is
//!   finite (orjson falls back to its double path for long integers and
//!   raises on infinity — 309 `9`s = 9.99e308 reject, 308 accept).
//! * **Depth cap 1024** (orjson's): the 1025th open bracket rejects.
//!   orjson counts ALL containers against one cap, objects and arrays
//!   alike (`[`×1024 accepts, `[`×1025 and `{"a":`×1025 reject).
//! * **Lone surrogate escapes reject** (`"\ud800"` alone: orjson raises;
//!   stdlib `json.loads` would build the lone surrogate). A high
//!   surrogate escape must be immediately followed by `\u` + a low
//!   surrogate escape; a low surrogate escape never pairs retroactively.
//! * **Raw control chars reject, `\u0000` accepts**; NaN/Infinity/-Infinity
//!   literals reject (RFC 8259 has no such grammar and orjson raises); a
//!   UTF-8 BOM rejects; invalid UTF-8 anywhere rejects (the structural
//!   bytes are ASCII by grammar, so validating UTF-8 inside strings is
//!   equivalent to validating the whole input); duplicate keys accept
//!   (orjson last-wins); trailing garbage, trailing commas, leading
//!   zeros, and unterminated strings reject.
//!
//! The one seam where the match is not proven byte-for-byte: the
//! overflow decision trusts Rust's correctly-rounded `f64` parser, so a
//! knife-edge literal sitting within rounding of ±1.8e308 could in
//! principle disagree with orjson's own float parser by one rounding
//! step. The differential corpus (2,000+ inputs including the
//! `1.7976931348623157e308` / `1.7976931348623159e308` boundary) found
//! zero such disagreements; the caveat stands because orjson's parser is
//! not a document of ours to verify. See docs/api.md.
//!
//! # The engine
//!
//! An iterative state machine, no recursion and no heap: container
//! context lives in a fixed 128-byte bitset on the scanner's stack (one
//! bit per open container — object or array — at most 1024, the depth
//! cap), so `{"a":[1]}`'s closes never need a dynamic stack. Whitespace
//! between tokens is a tight byte walk (runs are short — tokens are
//! dense); the string scanner is the hot path and skips plain runs
//! (bytes `0x20..0x7F` minus backslash) 8 bytes at a time with a SWAR
//! check, falling to per-byte handling only at escapes, controls, and
//! multibyte UTF-8. Numbers are grammar-validated in place, with the
//! f64 overflow gate above the only value-level computation anywhere in
//! the pass. Total: one linear scan, O(1) space, no allocation.

/// orjson's depth cap: the number of open containers the parser tolerates.
pub const MAX_DEPTH: usize = 1024;

/// One bit per open container (1 = object, 0 = array), indexed by depth.
/// At the 1024 cap this is 16 `u64`s = 128 bytes of stack, which is why
/// the scanner needs no heap at all.
const STACK_WORDS: usize = MAX_DEPTH / 64;

#[inline]
fn has_zero_byte(w: u64) -> bool {
    w.wrapping_sub(0x0101_0101_0101_0101) & !w & 0x8080_8080_8080_8080 != 0
}

/// Is every byte of the 8-byte window plain string content — ASCII
/// `0x20..=0x7F` minus the two specials (quote, backslash), which the
/// per-byte loop below must handle? Controls and high bytes break the
/// run too. Three per-byte predicates, none foolable: the `0x80` bit
/// test is carry-free per lane, the `& 0xE0` zero-byte test detects
/// exactly the bytes `< 0x20`, and each special has its own XOR
/// zero-byte detector.
#[inline]
fn plain8(w: u64) -> bool {
    (w & 0x8080_8080_8080_8080) == 0
        && !has_zero_byte(w & 0xE0E0_E0E0_E0E0_E0E0)
        && !has_zero_byte(w ^ 0x2222_2222_2222_2222)
        && !has_zero_byte(w ^ 0x5C5C_5C5C_5C5C_5C5C)
}

/// JSON whitespace: exactly space, tab, newline, carriage return (RFC
/// 8259's `WS`); orjson matches, and every other byte (a BOM included)
/// is a syntax error here.
#[inline]
fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

#[inline]
fn skip_ws(data: &[u8], mut i: usize) -> usize {
    while i < data.len() && is_ws(data[i]) {
        i += 1;
    }
    i
}

/// Validate the UTF-8 multibyte sequence starting at `i` (its lead byte
/// is `>= 0x80`); on success return the index just past it. The ranges
/// are std's own (`str::from_utf8` accepts exactly these): no overlong
/// encodings, no surrogate codepoints, nothing past U+10FFFF. A
/// continuation byte is always `0x80..=0xBF`, so a valid sequence can
/// never cross the closing quote (`0x22`) or a backslash (`0x5C`) — an
/// invalid sequence that contains one is rejected at its bad byte, not
/// skipped past.
fn utf8_step(data: &[u8], i: usize) -> Option<usize> {
    let (len, lo, hi) = match data[i] {
        0xC2..=0xDF => (1usize, 0x80u8, 0xBFu8),
        0xE0 => (2, 0xA0, 0xBF),
        0xE1..=0xEC | 0xEE..=0xEF => (2, 0x80, 0xBF),
        0xED => (2, 0x80, 0x9F), // ED: surrogates U+D800..U+DFFF excluded
        0xF0 => (3, 0x90, 0xBF),
        0xF1..=0xF3 => (3, 0x80, 0xBF),
        0xF4 => (3, 0x80, 0x8F), // F4: nothing past U+10FFFF
        _ => return None,        // 0x80..=0xC1 (continuation/overlong), 0xF5..=0xFF
    };
    let mut j = i + 1;
    for k in 0..len {
        let c = *data.get(j)?;
        let (lo_k, hi_k) = if k == 0 { (lo, hi) } else { (0x80, 0xBF) };
        if c < lo_k || c > hi_k {
            return None;
        }
        j += 1;
    }
    Some(j)
}

/// One ASCII hex digit quartet (`\u`'s payload), case-insensitive.
fn hex4(data: &[u8], i: usize) -> Option<u16> {
    let mut v: u16 = 0;
    for k in 0..4 {
        let c = *data.get(i + k)?;
        let d = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => return None,
        };
        v = (v << 4) | u16::from(d);
    }
    Some(v)
}

/// Scan a string whose opening quote sits just before `*pos`; on success
/// `*pos` is left just past the closing quote. The hot path is the SWAR
/// plain-run skip; escapes, controls, and multibyte UTF-8 are handled
/// per byte.
fn scan_string(data: &[u8], pos: &mut usize) -> bool {
    let n = data.len();
    let mut i = *pos;
    loop {
        // SWAR skip over plain runs: the common case for real content.
        while i + 8 <= n && plain8(u64::from_le_bytes(data[i..i + 8].try_into().unwrap())) {
            i += 8;
        }
        let b = match data.get(i) {
            Some(&b) => b,
            None => return false, // EOF before the closing quote
        };
        if b == b'"' {
            *pos = i + 1;
            return true;
        }
        if b == b'\\' {
            let e = match data.get(i + 1) {
                Some(&e) => e,
                None => return false,
            };
            match e {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => i += 2,
                b'u' => {
                    let Some(cp) = hex4(data, i + 2) else {
                        return false;
                    };
                    i += 6;
                    if (0xD800..0xDC00).contains(&cp) {
                        // High surrogate: orjson requires the immediate
                        // `\u` + low-surrogate pair, no gaps, no reuse.
                        if data.get(i) != Some(&b'\\') || data.get(i + 1) != Some(&b'u') {
                            return false;
                        }
                        let Some(lo) = hex4(data, i + 2) else {
                            return false;
                        };
                        if !(0xDC00..0xE000).contains(&lo) {
                            return false;
                        }
                        i += 6;
                    } else if (0xDC00..0xE000).contains(&cp) {
                        return false; // lone low surrogate
                    }
                }
                _ => return false, // `\x`, `\'`, `\-` ...: not JSON escapes
            }
            continue;
        }
        if b < 0x20 {
            return false; // raw control char (NUL included)
        }
        if b < 0x80 {
            i += 1;
            continue;
        }
        match utf8_step(data, i) {
            Some(j) => i = j,
            None => return false, // invalid UTF-8 inside the string
        }
    }
}

/// Does the number literal spanning `data[start..end]` round to a finite
/// f64? The span is ASCII by grammar, so the UTF-8 conversion cannot
/// fail; if it somehow did, the answer is `false` — the fail-safe
/// direction for a validity gate.
fn finite_f64(data: &[u8], start: usize, end: usize) -> bool {
    match std::str::from_utf8(&data[start..end]) {
        Ok(text) => text.parse::<f64>().is_ok_and(|v| v.is_finite()),
        Err(_) => false,
    }
}

/// Scan a number literal starting at `*pos` (its first byte is `-` or a
/// digit); on success `*pos` is left just past it. Grammar:
/// `-? (0 | [1-9][0-9]*) (\.[0-9]+)? ([eE][+-]?[0-9]+)?` — leading zeros
/// reject, `1.` / `.5` / `1e` reject. Value gates (orjson's reading,
/// both via the f64 fallback): a float literal or a 20+-digit integer
/// rejects exactly when its correctly-rounded f64 value is not finite.
fn scan_number(data: &[u8], pos: &mut usize) -> bool {
    let start = *pos;
    let mut i = *pos;
    if data[i] == b'-' {
        i += 1;
    }
    match data.get(i) {
        Some(b'0') => {
            i += 1;
            // `0` is a whole integer part by itself: `01`, `0123` reject.
            if matches!(data.get(i), Some(b'0'..=b'9')) {
                return false;
            }
        }
        Some(b'1'..=b'9') => {
            while matches!(data.get(i), Some(b'0'..=b'9')) {
                i += 1;
            }
        }
        _ => return false, // `-`, `-.`, a letter: no digits at all
    }
    let mut is_float = false;
    if data.get(i) == Some(&b'.') {
        is_float = true;
        i += 1;
        if !matches!(data.get(i), Some(b'0'..=b'9')) {
            return false;
        }
        while matches!(data.get(i), Some(b'0'..=b'9')) {
            i += 1;
        }
    }
    if matches!(data.get(i), Some(b'e' | b'E')) {
        is_float = true;
        i += 1;
        if matches!(data.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        if !matches!(data.get(i), Some(b'0'..=b'9')) {
            return false;
        }
        while matches!(data.get(i), Some(b'0'..=b'9')) {
            i += 1;
        }
    }
    *pos = i;
    if is_float {
        // orjson: "number is infinity when parsed as double" — 1e400
        // and friends reject, underflow (1e-400 -> 0.0) accepts.
        finite_f64(data, start, i)
    } else if i - start > 19 + usize::from(data[start] == b'-') {
        // Long integers: orjson's fallback parses them as doubles and
        // rejects infinity — by value, not digit count (309 `9`s =
        // 9.99e308 reject; 300 accept). 19 or fewer digits (plus sign)
        // never reach here: orjson builds a Python int for those,
        // unconditionally valid.
        finite_f64(data, start, i)
    } else {
        true
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum St {
    /// A value is required here: top level, after `[`, after `,` in an
    /// array, or after `:` in an object. A `]` never closes from here
    /// (that is `ArrayOrEnd`'s job), which is what makes `[1,]` and
    /// `{"a":]` reject.
    Value,
    /// Just after `[`: a value, or `]` closing the empty array.
    ArrayOrEnd,
    /// Just after `{`: a key, or `}` closing the empty object.
    ObjKeyOrEnd,
    /// After `,` in an object: a key, no `}` (trailing commas reject).
    ObjKey,
    /// After a complete value: `,` dispatching on the container kind, a
    /// kind-matched close, or EOF closing the top level.
    AfterValue,
}

/// The validity question itself: would a JSON parser with orjson's
/// acceptance set (see the module docs) accept `data` as one complete
/// document? One linear pass, no recursion, no heap, no exceptions.
pub fn is_valid(data: &[u8]) -> bool {
    let mut st = St::Value;
    let mut pos = 0usize;
    let mut stack = [0u64; STACK_WORDS];
    let mut depth = 0usize;

    // The container stack: bit `depth` set = object. Clear-on-push keeps
    // stale bits from an outer scope (closed earlier at the same depth)
    // from masquerading as the new container's kind.
    #[inline]
    fn push(stack: &mut [u64; STACK_WORDS], depth: &mut usize, is_object: bool) -> bool {
        if *depth >= MAX_DEPTH {
            return false;
        }
        let bit = 1u64 << (*depth % 64);
        if is_object {
            stack[*depth / 64] |= bit;
        } else {
            stack[*depth / 64] &= !bit;
        }
        *depth += 1;
        true
    }
    #[inline]
    fn top_is_object(stack: &[u64; STACK_WORDS], depth: usize) -> bool {
        debug_assert!(depth >= 1);
        stack[(depth - 1) / 64] >> ((depth - 1) % 64) & 1 == 1
    }

    loop {
        match st {
            St::AfterValue => {
                pos = skip_ws(data, pos);
                match data.get(pos) {
                    // The one EOF that is a success: the top-level value
                    // is complete and nothing trails it.
                    None => return depth == 0,
                    Some(b',') => {
                        pos += 1;
                        if depth == 0 {
                            return false; // `,` after a complete top-level value
                        }
                        st = if top_is_object(&stack, depth) {
                            St::ObjKey
                        } else {
                            St::Value
                        };
                    }
                    Some(b']') => {
                        if depth == 0 || top_is_object(&stack, depth) {
                            return false;
                        }
                        depth -= 1;
                        pos += 1;
                    }
                    Some(b'}') => {
                        if depth == 0 || !top_is_object(&stack, depth) {
                            return false; // `]` closing an object and vice versa
                        }
                        depth -= 1;
                        pos += 1;
                    }
                    Some(_) => return false, // trailing garbage
                }
            }
            St::Value | St::ArrayOrEnd => {
                pos = skip_ws(data, pos);
                match data.get(pos) {
                    None => return false, // EOF where a value is required
                    Some(b'[') => {
                        pos += 1;
                        if !push(&mut stack, &mut depth, false) {
                            return false;
                        }
                        st = St::ArrayOrEnd;
                    }
                    Some(b'{') => {
                        pos += 1;
                        if !push(&mut stack, &mut depth, true) {
                            return false;
                        }
                        st = St::ObjKeyOrEnd;
                    }
                    Some(b'"') => {
                        pos += 1;
                        if !scan_string(data, &mut pos) {
                            return false;
                        }
                        st = St::AfterValue;
                    }
                    Some(b't') if data[pos..].starts_with(b"true") => {
                        pos += 4;
                        st = St::AfterValue;
                    }
                    Some(b'f') if data[pos..].starts_with(b"false") => {
                        pos += 5;
                        st = St::AfterValue;
                    }
                    Some(b'n') if data[pos..].starts_with(b"null") => {
                        pos += 4;
                        st = St::AfterValue;
                    }
                    // `truex` and friends fall through to the digit arm
                    // and reject (a literal prefix is not the literal).
                    Some(b'-') | Some(b'0'..=b'9') => {
                        if !scan_number(data, &mut pos) {
                            return false;
                        }
                        st = St::AfterValue;
                    }
                    Some(b']') if st == St::ArrayOrEnd => {
                        depth -= 1;
                        pos += 1;
                        st = St::AfterValue;
                    }
                    Some(_) => return false,
                }
            }
            St::ObjKeyOrEnd | St::ObjKey => {
                pos = skip_ws(data, pos);
                match data.get(pos) {
                    Some(b'}') if st == St::ObjKeyOrEnd => {
                        depth -= 1;
                        pos += 1;
                        st = St::AfterValue;
                    }
                    Some(b'"') => {
                        pos += 1;
                        if !scan_string(data, &mut pos) {
                            return false;
                        }
                        pos = skip_ws(data, pos);
                        match data.get(pos) {
                            Some(b':') => {
                                pos += 1;
                                st = St::Value;
                            }
                            _ => return false, // `{"a" 1}`, `{"a":`
                        }
                    }
                    _ => return false,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::is_valid;

    #[test]
    fn scalar_and_container_grammar() {
        assert!(is_valid(b"{}"));
        assert!(is_valid(b"[]"));
        assert!(is_valid(b"[1, 2, 3]"));
        assert!(is_valid(br#"{"a": 1, "b": [true, false, null]}"#));
        assert!(is_valid(b"  \t\r\n [1] \t \r\n "));
        assert!(is_valid(b"42"));
        assert!(is_valid(b"-0"));
        assert!(is_valid(b"1e5"));
        assert!(is_valid(br#""top-level string""#));
        assert!(!is_valid(b""));
        assert!(!is_valid(b"   "));
        assert!(!is_valid(b"[1 2]"));
        assert!(!is_valid(b"[1,]"));
        assert!(!is_valid(br#"{"a":1,}"#));
        assert!(!is_valid(b"[1] x"));
        assert!(!is_valid(br#"{"a":1}{"b":2}"#));
        assert!(!is_valid(b"01"));
        assert!(!is_valid(b"truex"));
        assert!(!is_valid(b"NaN"));
        assert!(!is_valid(b"Infinity"));
        assert!(!is_valid(b"\xef\xbb\xbf{}")); // BOM
        assert!(!is_valid(b"[1]\x00"));
    }

    #[test]
    fn depth_cap_is_orjsons_1024() {
        let ok_arr = ("[".repeat(1024) + &"]".repeat(1024)).into_bytes();
        assert!(is_valid(&ok_arr));
        let over_arr = ("[".repeat(1025) + &"]".repeat(1025)).into_bytes();
        assert!(!is_valid(&over_arr));
        let ok_obj = ((r#"{"a":"#).repeat(1024) + "0" + &"}".repeat(1024)).into_bytes();
        assert!(is_valid(&ok_obj));
        let over_obj = ((r#"{"a":"#).repeat(1025) + "0" + &"}".repeat(1025)).into_bytes();
        assert!(!is_valid(&over_obj));
        // Mixed containers share one cap: nested objects each holding
        // one array nest (`{"a":{"a":...[0...]}}`), 512+512 = 1024 open
        // containers exactly at the cap; 600+600 = 1200 rejects.
        let mixed_ok = ((r#"{"a":"#).repeat(512)
            + &"[".repeat(512)
            + "0"
            + &"]".repeat(512)
            + &"}".repeat(512))
            .into_bytes();
        assert!(is_valid(&mixed_ok));
        let mixed = ((r#"{"a":"#).repeat(600)
            + &"[".repeat(600)
            + "0"
            + &"]".repeat(600)
            + &"}".repeat(600))
            .into_bytes();
        assert!(!is_valid(&mixed));
    }

    #[test]
    fn string_rules() {
        assert!(is_valid(br#""\u0000""#));
        assert!(is_valid(br#""\ud800\udc00""#));
        assert!(is_valid(br#""\uD83D\uDE00""#));
        assert!(is_valid(br#""""#));
        assert!(is_valid("\"a\u{1511}\"".as_bytes())); // valid multibyte in a raw string
        assert!(!is_valid(br#""\ud800""#)); // lone high surrogate
        assert!(!is_valid(br#""\udc00""#)); // lone low surrogate
        assert!(!is_valid(br#""\ud800x""#)); // high surrogate, no pair
        assert!(!is_valid(br#""\ud800\n""#)); // escape between the pair
        assert!(!is_valid(br#""\udc00\ud800""#)); // low first, never pairs
        assert!(!is_valid(br#""\q""#));
        assert!(!is_valid(br#""\u{41}""#));
        assert!(!is_valid(b"\"a\x01b\"")); // raw control
        assert!(!is_valid(b"\"a\x00b\"")); // raw NUL
        assert!(!is_valid(b"\"ab")); // unterminated at EOF
        assert!(!is_valid(br#""a\""#)); // trailing lone backslash
        assert!(!is_valid(b"\"\xc3\"")); // truncated multibyte
        assert!(!is_valid(b"\"\xc0\x80\"")); // overlong NUL
        assert!(!is_valid(b"\"\xed\xa0\x80\"")); // encoded surrogate
        assert!(!is_valid(b"\"\xff\"")); // bare invalid byte
    }

    #[test]
    fn number_value_gates() {
        // Float overflow rejects; underflow accepts.
        for lit in [
            "1e400",
            "-1e400",
            "1e309",
            "2e308",
            "1.7976931348623159e308",
        ] {
            assert!(!is_valid(lit.as_bytes()), "{lit} must reject");
        }
        for lit in [
            "1e-400", "-1e-400", "1e-323", "0.0", "-0.0", "1e0", "0e0", "1.5e2",
        ] {
            assert!(is_valid(lit.as_bytes()), "{lit} must accept");
        }
        // Integer digit gates: <= 19 digits always; past that, the f64
        // fallback's infinity is the reject — by VALUE, not digit count
        // (309 `9`s = 9.99e308 overflows; 308 is finite).
        assert!(is_valid(b"9223372036854775807")); // i64::MAX, 19 digits
        assert!(is_valid(b"99999999999999999999")); // 20 digits -> 1e19
        assert!(is_valid(&[b'9'; 300][..])); // 9.99e299 finite
        assert!(is_valid(&[b'9'; 308][..])); // 9.99e307 finite
        assert!(!is_valid(&[b'9'; 309][..])); // 9.99e308 -> inf
        assert!(!is_valid(&[b'9'; 512][..]));
        let neg_big = format!("-{}", "9".repeat(512)).into_bytes();
        assert!(!is_valid(&neg_big));
        assert!(!is_valid(b"1."));
        assert!(!is_valid(b".5"));
        assert!(!is_valid(b"1e"));
        assert!(!is_valid(b"-"));
        assert!(!is_valid(b"+1"));
    }

    #[test]
    fn utf8_outside_strings_is_grammar_rejected() {
        // Structural positions are ASCII by grammar, so a high byte
        // there is a syntax error before UTF-8 even matters; the
        // string-internal UTF-8 checks make the whole input validated.
        assert!(!is_valid(b"[1, \xff]"));
        assert!(!is_valid(b"{\xff: 1}"));
        assert!(is_valid("[\"\u{e9}\"]".as_bytes()));
    }
}
