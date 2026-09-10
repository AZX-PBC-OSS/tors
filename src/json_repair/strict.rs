//! The strict fast path: `json.loads` byte-parity over whole inputs, plus
//! the `raw_decode` suffix probe the repair parser drives. Not a
//! json_repair file: a CPython-stdlib parity layer the port needs
//! (json_repair.py's fast path calls `json.loads(json_str)`; json_parser's
//! `_try_parse_valid_json_value` calls
//! `json.JSONDecoder().raw_decode(json_str[index:])`), pinned to the
//! scanner's documented behavior:
//!
//! - Whitespace is exactly CPython's JSON set (space, `\t`, `\n`, `\r`),
//!   allowed around the whole document and between tokens, nothing else.
//! - The constants `true`/`false`/`null` and `NaN`/`Infinity`/`-Infinity`
//!   are accepted anywhere a value is expected (CPython's scanner allows
//!   them by default, not just at the top level).
//! - Numbers follow the scanner regex `-?(0|[1-9][0-9]*)(\.[0-9]+)?
//!   ([eE][-+]?[0-9]+)?`: the fraction requires at least one digit, so
//!   `"1."`/`"1.e5"` are invalid ("Extra data"), as are `"01"`, `"1e"`,
//!   `"-"`, `".5"`, `"+1"`. (The port design doc pins the fraction as
//!   optional with `"1."` valid: that is a mis-reading of the stdlib:
//!   every CPython 3.10-3.14, pure and C scanner alike, requires a digit
//!   after the point, verified on 3.10/3.11/3.12/3.13/3.14. Following the
//!   doc would also break differential parity: upstream json_repair
//!   0.63.4 returns `""` for `"1."`-shaped inputs precisely because its
//!   own json.loads fast path fails identically. The stdlib behavior
//!   wins; the conflict is flagged to the lead.) Integers beyond i64
//!   range become [`Value::BigInt`] (Python ints are unbounded); anything
//!   with a fraction or exponent is an f64 (CPython's parse_float runs on
//!   exactly these forms: Rust's f64 parser accepts every one of them).
//! - Strings take the full escape set including `\uXXXX` and surrogate
//!   pairs combining to the astral char; a raw control char < 0x20
//!   inside a string is rejected (the strict scanner's rule). A lone
//!   `\ud800`-`\udfff` escape maps to U+FFFD: a Rust `String` cannot
//!   hold a lone surrogate where CPython's can; this is the port's
//!   documented divergence (design doc §9.2), and inputs containing lone
//!   surrogates cannot reach tors through `&str` extraction anyway.
//! - Duplicate object keys update the value in place at the
//!   first-occurrence position (CPython dict semantics).
//! - Container nesting is capped at 200 (`MAX_NESTING`, see mod.rs's
//!   docs): over-cap inputs fail here as `Err(())`, so `repair` falls
//!   through to the repair parser, whose own cap surfaces upstream's
//!   normalized ValueError where CPython would raise an uncaught
//!   RecursionError near its own limit (divergence §9.6). Depth is
//!   decremented on the success exits only; every `Err` path abandons
//!   the Scanner for good (no caller reuses one after an error), so the
//!   asymmetry is safe: pin it if a reuse ever appears.

use super::{MAX_NESTING, ObjectBuilder, Value};

/// `json.loads(s)` parity over the whole input: leading/trailing
/// whitespace (space/`\t`/`\n`/`\r` only) is fine, any trailing garbage
/// fails, top-level scalars/strings are allowed, and every other rule is
/// the scanner grammar in the module docs. `Err(())` on any failure:
/// callers distinguish nothing, exactly like upstream catching
/// `json.JSONDecodeError` and falling through to the repair parser.
// The unit error type is the pinned cross-module contract (design doc
// §3): there is no failure detail to carry, so () it stays.
#[allow(clippy::result_unit_err)]
pub fn loads_strict(s: &str) -> Result<Value, ()> {
    let mut sc = Scanner::new(s);
    sc.skip_ws();
    let v = sc.parse_value()?;
    sc.skip_ws();
    if sc.i != sc.s.len() {
        // CPython's "Extra data": anything but whitespace after the value.
        return Err(());
    }
    Ok(v)
}

/// `json.JSONDecoder().raw_decode(&s[char_start..])` parity: skip leading
/// whitespace, parse one value, return it plus the end index in codepoints
/// relative to `char_start` (upstream advances its codepoint cursor by
/// exactly that amount). Content after the value is fine: that is the
/// point: this is the "everything from here on is already valid JSON"
/// probe. Depth-capped like [`loads_strict`].
///
/// Divergence note: CPython's own `raw_decode` does not pre-skip
/// whitespace (`JSONDecoder.decode` does that before calling it), but
/// upstream only ever invokes it with the cursor already sitting on a
/// `{`/`[`, so the skip is unobservable through the repair flow and makes
/// the probe self-contained for other call sites; the design doc pins the
/// skipping form.
// Unit error type per the pinned contract, like loads_strict above.
#[allow(clippy::result_unit_err)]
pub fn raw_decode(s: &str, char_start: usize) -> Result<(Value, usize), ()> {
    // `s[char_start..]` is a codepoint slice in Python; find the byte
    // offset of that codepoint (past the end behaves like Python's
    // clamping slice: an empty suffix).
    let byte_start = s.char_indices().nth(char_start).map_or(s.len(), |(b, _)| b);
    let t = &s[byte_start..];
    let mut sc = Scanner::new(t);
    sc.skip_ws();
    let v = sc.parse_value()?;
    // The end index is in codepoints relative to the slice start (the
    // unit of upstream's parser.index), whitespace included.
    Ok((v, t[..sc.i].chars().count()))
}

/// A byte-cursor JSON scanner over a `&str` slice. All structural bytes
/// are ASCII; multibyte UTF-8 is payload that never matches a delimiter,
/// so byte-level scanning is exact (and fast) over `&str` input.
struct Scanner<'a> {
    s: &'a [u8],
    i: usize,
    depth: usize,
}

impl<'a> Scanner<'a> {
    fn new(s: &'a str) -> Scanner<'a> {
        Scanner {
            s: s.as_bytes(),
            i: 0,
            depth: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    /// CPython's whitespace set, and only it.
    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn enter_depth(&mut self) -> Result<(), ()> {
        self.depth += 1;
        if self.depth > MAX_NESTING {
            // See the module docs: a plain failure here so the repair
            // parser's own cap raises the normalized ValueError.
            Err(())
        } else {
            Ok(())
        }
    }

    fn parse_value(&mut self) -> Result<Value, ()> {
        match self.peek() {
            None => Err(()),
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => {
                self.i += 1;
                Ok(Value::Str(self.parse_string_body()?))
            }
            Some(b't') => self.expect_literal("true", Value::Bool(true)),
            Some(b'f') => self.expect_literal("false", Value::Bool(false)),
            Some(b'n') => self.expect_literal("null", Value::Null),
            // The scanner accepts the non-finite constants everywhere a
            // value can appear.
            Some(b'N') => self.expect_literal("NaN", Value::Float(f64::NAN)),
            Some(b'I') => self.expect_literal("Infinity", Value::Float(f64::INFINITY)),
            Some(b'-') if self.s[self.i..].starts_with(b"-Infinity") => {
                self.i += "-Infinity".len();
                Ok(Value::Float(f64::NEG_INFINITY))
            }
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            _ => Err(()),
        }
    }

    fn expect_literal(&mut self, word: &str, value: Value) -> Result<Value, ()> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(value)
        } else {
            // "tru"/"True"/"nan"/...: no candidate matches, and the
            // number regex cannot either.
            Err(())
        }
    }

    /// The scanner's number regex, `-?(0|[1-9][0-9]*)(\.[0-9]+)?
    /// ([eE][-+]?[0-9]+)?`, consumed exactly as far as it matches (an
    /// optional group that lacks its required digits is not consumed at
    /// all: that is what turns "01" into "0" + garbage and "1e" into "1"
    /// + garbage, CPython's "Extra data").
    fn parse_number(&mut self) -> Result<Value, ()> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        match self.peek() {
            // "0" matches only itself; a following digit is left as
            // trailing garbage ("01" fails downstream, like CPython).
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.i += 1;
                }
            }
            _ => return Err(()),
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') && matches!(self.s.get(self.i + 1), Some(b'0'..=b'9')) {
            is_float = true;
            self.i += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            let mut j = self.i + 1;
            if matches!(self.s.get(j), Some(b'+') | Some(b'-')) {
                j += 1;
            }
            if matches!(self.s.get(j), Some(b'0'..=b'9')) {
                is_float = true;
                self.i = j;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.i += 1;
                }
            }
        }
        // The consumed bytes are grammar-ASCII by construction; the
        // from_utf8/map_err pair just keeps the scanner panic-free.
        let text = std::str::from_utf8(&self.s[start..self.i]).map_err(|_| ())?;
        if is_float {
            // Only well-formed `-?d+(.d+)?(e[+-]?d+)?` shapes reach the
            // f64 parser, every one of them accepted by Rust (verified:
            // Rust, like CPython's float(), is the more lenient of the
            // two grammars here).
            Ok(Value::Float(text.parse::<f64>().map_err(|_| ())?))
        } else {
            match text.parse::<i64>() {
                Ok(n) => Ok(Value::Int(n)),
                // Python ints are unbounded; the grammar already forbids
                // leading zeros, but the BigInt text is normalized anyway
                // per the Value contract.
                Err(_) => Ok(Value::BigInt(normalize_big_int(text))),
            }
        }
    }

    /// The string body after the opening `"`. Raw bytes < 0x20 are
    /// rejected (the strict scanner's control-character rule); UTF-8
    /// payload bytes (all >= 0x80) are copied verbatim in runs.
    fn parse_string_body(&mut self) -> Result<String, ()> {
        let mut out = String::new();
        loop {
            let Some(&b) = self.s.get(self.i) else {
                return Err(()); // unterminated
            };
            match b {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.i += 1;
                    self.parse_escape(&mut out)?;
                }
                0x00..=0x1f => return Err(()),
                _ => {
                    // Copy the run of ordinary bytes up to the next
                    // quote/backslash/control; multibyte sequences never
                    // contain those (their bytes are all >= 0x80), so the
                    // run ends on a char boundary and is valid UTF-8 by
                    // construction (the slice came from a &str).
                    let seg_start = self.i;
                    self.i += 1;
                    while let Some(&b) = self.s.get(self.i) {
                        if b == b'"' || b == b'\\' || b < 0x20 {
                            break;
                        }
                        self.i += 1;
                    }
                    out.push_str(std::str::from_utf8(&self.s[seg_start..self.i]).map_err(|_| ())?);
                }
            }
        }
    }

    /// One escape after the backslash (cursor on the escape character).
    fn parse_escape(&mut self, out: &mut String) -> Result<(), ()> {
        let Some(&b) = self.s.get(self.i) else {
            return Err(()); // trailing backslash
        };
        self.i += 1;
        match b {
            b'"' => out.push('"'),
            b'\\' => out.push('\\'),
            b'/' => out.push('/'),
            b'b' => out.push('\u{08}'),
            b'f' => out.push('\u{0c}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => {
                let cp = u32::from(self.parse_hex4()?);
                if (0xd800..=0xdbff).contains(&cp) {
                    // High surrogate: combine with a directly following
                    // \uDC00-\uDFFF escape into the astral char. Anything
                    // else leaves it lone, and a lone surrogate maps to
                    // U+FFFD (a Rust String cannot hold it; documented
                    // divergence, see the module docs), with the lookahead
                    // unconsumed so the following escape is still
                    // processed normally.
                    if self.s.get(self.i) == Some(&b'\\') && self.s.get(self.i + 1) == Some(&b'u') {
                        let save = self.i;
                        self.i += 2;
                        let lo = u32::from(self.parse_hex4()?);
                        if (0xdc00..=0xdfff).contains(&lo) {
                            let c = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                            out.push(char::from_u32(c).ok_or(())?);
                        } else {
                            self.i = save;
                            out.push('\u{fffd}');
                        }
                    } else {
                        out.push('\u{fffd}');
                    }
                } else if (0xdc00..=0xdfff).contains(&cp) {
                    // A lone low surrogate.
                    out.push('\u{fffd}');
                } else {
                    out.push(char::from_u32(cp).ok_or(())?);
                }
            }
            _ => return Err(()), // CPython's "Invalid \escape"
        }
        Ok(())
    }

    /// Exactly four hex digits (both cases, like CPython's scanner).
    fn parse_hex4(&mut self) -> Result<u16, ()> {
        let mut v: u16 = 0;
        for _ in 0..4 {
            let Some(&b) = self.s.get(self.i) else {
                return Err(());
            };
            let d = match b {
                b'0'..=b'9' => b - b'0',
                b'a'..=b'f' => b - b'a' + 10,
                b'A'..=b'F' => b - b'A' + 10,
                _ => return Err(()),
            };
            self.i += 1;
            v = v * 16 + u16::from(d);
        }
        Ok(v)
    }

    fn parse_object(&mut self) -> Result<Value, ()> {
        self.enter_depth()?;
        self.i += 1; // '{'
        // The builder's side index keeps duplicate-key updates O(1):
        // the linear object_insert scan is O(n²) on large objects.
        let mut obj = ObjectBuilder::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            self.depth -= 1;
            return Ok(obj.finish());
        }
        loop {
            self.skip_ws();
            // Keys must be double-quoted strings ("Expecting property
            // name enclosed in double quotes").
            if self.peek() != Some(b'"') {
                return Err(());
            }
            self.i += 1;
            let key = self.parse_string_body()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(());
            }
            self.i += 1;
            self.skip_ws();
            let value = self.parse_value()?;
            // CPython dict semantics: a duplicate key updates the value
            // in place at its first-occurrence position.
            obj.insert(key, value);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    self.depth -= 1;
                    return Ok(obj.finish());
                }
                _ => return Err(()),
            }
        }
    }

    fn parse_array(&mut self) -> Result<Value, ()> {
        self.enter_depth()?;
        self.i += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            self.depth -= 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                // A trailing comma leaves ']' where a value is expected:
                // rejected, exactly like CPython.
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    self.depth -= 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(()),
            }
        }
    }
}

/// Normalized decimal text for a beyond-i64 integer: sign kept, leading
/// zeros stripped, an all-zero body collapses to "0" (so "-0" normalizes
/// to "0"). The strict grammar can never produce text that needs this,
/// but the `Value::BigInt` contract normalizes regardless.
fn normalize_big_int(text: &str) -> String {
    let (sign, digits) = match text.strip_prefix('-') {
        Some(d) => ("-", d),
        None => ("", text),
    };
    let stripped = digits.trim_start_matches('0');
    if stripped.is_empty() {
        "0".to_string()
    } else {
        format!("{sign}{stripped}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_scalars_and_constants() {
        assert_eq!(loads_strict("0").unwrap(), Value::Int(0));
        assert_eq!(loads_strict("-0").unwrap(), Value::Int(0));
        assert_eq!(loads_strict("42").unwrap(), Value::Int(42));
        assert_eq!(loads_strict("-42").unwrap(), Value::Int(-42));
        assert_eq!(loads_strict("1.5").unwrap(), Value::Float(1.5));
        assert_eq!(loads_strict("1e5").unwrap(), Value::Float(100000.0));
        assert_eq!(loads_strict("1E5").unwrap(), Value::Float(100000.0));
        assert_eq!(loads_strict("1.5e-3").unwrap(), Value::Float(0.0015));
        // CPython keeps the sign on a negative zero float.
        let neg_zero = loads_strict("-0.0").unwrap();
        assert!(matches!(neg_zero, Value::Float(f) if f.to_bits() == (-0.0f64).to_bits()));
        // The non-finite constants, at the top level and nested.
        assert!(matches!(loads_strict("NaN"), Ok(Value::Float(x)) if x.is_nan()));
        assert_eq!(
            loads_strict("Infinity").unwrap(),
            Value::Float(f64::INFINITY)
        );
        assert_eq!(
            loads_strict("-Infinity").unwrap(),
            Value::Float(f64::NEG_INFINITY)
        );
        assert!(matches!(
            loads_strict("[NaN, Infinity]"),
            Ok(Value::Array(items)) if matches!(&items[..], [Value::Float(a), Value::Float(b)] if a.is_nan() && *b == f64::INFINITY)
        ));
        assert_eq!(loads_strict("true").unwrap(), Value::Bool(true));
        assert_eq!(loads_strict("false").unwrap(), Value::Bool(false));
        assert_eq!(loads_strict("null").unwrap(), Value::Null);
        assert_eq!(loads_strict("\"hi\"").unwrap(), Value::Str("hi".into()));
        // Beyond i64: Python's unbounded int.
        assert_eq!(
            loads_strict("9223372036854775808").unwrap(),
            Value::BigInt("9223372036854775808".into())
        );
        assert_eq!(
            loads_strict("-9223372036854775809").unwrap(),
            Value::BigInt("-9223372036854775809".into())
        );
    }

    #[test]
    fn loads_number_grammar_edges() {
        // The fraction needs a digit: every CPython 3.10-3.14 rejects
        // these (see the module docs for the design-doc conflict).
        assert_eq!(loads_strict("1."), Err(()));
        assert_eq!(loads_strict("1.e5"), Err(()));
        assert_eq!(loads_strict("-0."), Err(()));
        assert_eq!(loads_strict("1.5e"), Err(()));
        // Leading zeros / bare signs / dangling exponents / stray pluses.
        assert_eq!(loads_strict("01"), Err(()));
        assert_eq!(loads_strict("00"), Err(()));
        assert_eq!(loads_strict("1e"), Err(()));
        assert_eq!(loads_strict("1e+"), Err(()));
        assert_eq!(loads_strict("-"), Err(()));
        assert_eq!(loads_strict(".5"), Err(()));
        assert_eq!(loads_strict("+1"), Err(()));
        // Case-sensitive spellings.
        assert_eq!(loads_strict("True"), Err(()));
        assert_eq!(loads_strict("nan"), Err(()));
        assert_eq!(loads_strict("-NaN"), Err(()));
        assert_eq!(loads_strict("Infinityx"), Err(()));
        assert_eq!(loads_strict("tru"), Err(()));
    }

    #[test]
    fn loads_whitespace_handling() {
        // CPython's JSON whitespace set only: space, \t, \n, \r.
        assert_eq!(loads_strict(" \t\n\r 7 \n").unwrap(), Value::Int(7));
        assert_eq!(loads_strict("\u{b}7"), Err(()));
        assert_eq!(loads_strict("\u{a0}7"), Err(()));
        // Trailing garbage fails; trailing whitespace does not.
        assert_eq!(loads_strict("7 x"), Err(()));
        assert_eq!(loads_strict("[]extra"), Err(()));
        assert_eq!(loads_strict("").unwrap_err(), ());
        assert_eq!(loads_strict("   "), Err(()));
    }

    #[test]
    fn loads_string_escapes() {
        let v = loads_strict(r#""\" \\ \/ \b \f \n \r \t \u0041""#).unwrap();
        assert_eq!(v, Value::Str("\" \\ / \u{08} \u{0c} \n \r \t A".into()));
        // Uppercase hex digits are accepted.
        assert_eq!(
            loads_strict(r#""\uABCD""#).unwrap(),
            Value::Str("\u{abcd}".into())
        );
        assert_eq!(
            loads_strict(r#""\u0000""#).unwrap(),
            Value::Str("\u{0}".into())
        );
        // Surrogate pairs combine to the astral char.
        assert_eq!(
            loads_strict(r#""\ud83d\ude00""#).unwrap(),
            Value::Str("\u{1F600}".into())
        );
        // A raw DEL inside a string is fine (only < 0x20 is rejected).
        assert_eq!(
            loads_strict("\"a\u{7f}b\"").unwrap(),
            Value::Str("a\u{7f}b".into())
        );
        // Raw control chars, invalid escapes, unterminated strings.
        assert_eq!(loads_strict("\"a\u{1}b\""), Err(()));
        assert_eq!(loads_strict(r#""\q""#), Err(()));
        assert_eq!(loads_strict(r#""\u12""#), Err(()));
        assert_eq!(loads_strict(r#""\u12g4""#), Err(()));
        assert_eq!(loads_strict("\"abc"), Err(()));
        assert_eq!(loads_strict("\"ab\\"), Err(()));
    }

    #[test]
    fn loads_lone_surrogates_become_fffd() {
        // Documented divergence: CPython returns a str holding the lone
        // surrogate; a Rust String cannot, so it maps to U+FFFD.
        assert_eq!(
            loads_strict(r#""\ud800""#).unwrap(),
            Value::Str("\u{FFFD}".into())
        );
        assert_eq!(
            loads_strict(r#""\udc00""#).unwrap(),
            Value::Str("\u{FFFD}".into())
        );
        assert_eq!(
            loads_strict(r#""\ud800\ud800""#).unwrap(),
            Value::Str("\u{FFFD}\u{FFFD}".into())
        );
        // High surrogate followed by a non-low escape: FFFD plus the
        // escape processed normally (CPython: lone surrogate + 'A').
        assert_eq!(
            loads_strict(r#""\ud83d\u0041""#).unwrap(),
            Value::Str("\u{FFFD}A".into())
        );
        assert_eq!(
            loads_strict(r#""\ud83dx""#).unwrap(),
            Value::Str("\u{FFFD}x".into())
        );
    }

    #[test]
    fn loads_containers() {
        let v = loads_strict("{ \"key\": [1, 2.5, true, null] }").unwrap();
        assert_eq!(
            v,
            Value::Object(vec![(
                "key".into(),
                Value::Array(vec![
                    Value::Int(1),
                    Value::Float(2.5),
                    Value::Bool(true),
                    Value::Null,
                ])
            )])
        );
        assert_eq!(loads_strict("{}").unwrap(), Value::Object(vec![]));
        assert_eq!(loads_strict("[]").unwrap(), Value::Array(vec![]));
        assert_eq!(
            loads_strict("[[[]]]").unwrap(),
            Value::Array(vec![Value::Array(vec![Value::Array(vec![])])])
        );
        // An empty key is a legal string.
        assert_eq!(
            loads_strict(r#"{"" : 1}"#).unwrap(),
            Value::Object(vec![("".into(), Value::Int(1))])
        );
    }

    #[test]
    fn loads_duplicate_keys_update_in_place() {
        let v = loads_strict("{\"a\": 1, \"b\": 2, \"a\": 3}").unwrap();
        assert_eq!(
            v,
            Value::Object(vec![
                ("a".into(), Value::Int(3)),
                ("b".into(), Value::Int(2)),
            ])
        );
    }

    #[test]
    fn loads_container_errors() {
        assert_eq!(loads_strict("[1,2,]"), Err(()));
        assert_eq!(loads_strict("[,1]"), Err(()));
        assert_eq!(loads_strict("[1 2]"), Err(()));
        assert_eq!(loads_strict("{\"a\" 1}"), Err(()));
        assert_eq!(loads_strict("{a: 1}"), Err(()));
        assert_eq!(loads_strict("{\"a\":}"), Err(()));
        assert_eq!(loads_strict("{\"a\": 1,"), Err(()));
        assert_eq!(loads_strict("["), Err(()));
        assert_eq!(loads_strict("{"), Err(()));
        assert_eq!(loads_strict("[1"), Err(()));
    }

    #[test]
    fn loads_depth_cap() {
        // 200 nested containers fit; the 201st fails (Python would raise
        // an uncaught RecursionError near its own limit: see mod.rs's
        // MAX_NESTING docs).
        let ok = format!("{}0{}", "[".repeat(200), "]".repeat(200));
        assert!(loads_strict(&ok).is_ok());
        let deep = format!("{}0{}", "[".repeat(201), "]".repeat(201));
        assert_eq!(loads_strict(&deep), Err(()));
        // An unclosed run hits the cap before it runs out of input.
        assert_eq!(loads_strict(&"[".repeat(250)), Err(()));
    }

    #[test]
    fn raw_decode_offsets() {
        // The end index is in codepoints relative to char_start and
        // includes the leading whitespace the skip consumed.
        assert_eq!(raw_decode("  42 rest", 0).unwrap(), (Value::Int(42), 4));
        assert_eq!(
            raw_decode("\t\n [1] x", 0).unwrap(),
            (Value::Array(vec![Value::Int(1)]), 6)
        );
        // Mid-string starts: char_start is a codepoint offset into s.
        assert_eq!(
            raw_decode("junk{\"a\":1}", 4).unwrap(),
            (Value::Object(vec![("a".into(), Value::Int(1))]), 7)
        );
        // Astral chars in the skipped prefix keep offsets codepoint-true
        // (U+1D400 is two UTF-16 units, one codepoint, four bytes).
        assert_eq!(
            raw_decode("\u{1D400}[2]", 1).unwrap(),
            (Value::Array(vec![Value::Int(2)]), 3)
        );
        // Content after the value is fine (the point of the probe).
        assert_eq!(
            raw_decode("[1]]]]", 0).unwrap(),
            (Value::Array(vec![Value::Int(1)]), 3)
        );
        // The number match stops exactly where the scanner regex would:
        // "0" claims one char and leaves the "[1]".
        assert_eq!(raw_decode("0[1]", 0).unwrap(), (Value::Int(0), 1));
        // Failures: no value at the cursor, a whitespace-only suffix, a
        // start past the end (Python's clamping slice -> empty suffix).
        assert_eq!(raw_decode("zz", 0), Err(()));
        assert_eq!(raw_decode("[1]   ", 3), Err(()));
        assert_eq!(raw_decode("[1]", 99), Err(()));
    }

    #[test]
    fn big_int_text_is_normalized() {
        // Unreachable through the grammar (no leading zeros possible),
        // but the Value::BigInt contract normalizes anyway.
        assert_eq!(normalize_big_int("123"), "123");
        assert_eq!(normalize_big_int("-123"), "-123");
        assert_eq!(normalize_big_int("000123"), "123");
        assert_eq!(normalize_big_int("-0"), "0");
        assert_eq!(normalize_big_int("0000"), "0");
    }
}
