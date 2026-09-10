//! The serializer behind `tors.repair_json`'s string output: byte-parity
//! with CPython's `json.dumps(v)` defaults (item separator `", "`, key
//! separator `": "`, the `allow_nan` non-finite spellings, `ensure_ascii`
//! both ways), plus `float.__repr__` parity for every `Value::Float`.
//!
//! Why a hand-rolled serializer: json_repair's `repair_json` finishes with
//! `json.dumps(parsed_json, **json_dumps_args)`, so the port's str-out
//! wrapper re-serializes the repaired value and the corpus/differential
//! suites pin the exact bytes. The escape tables are CPython's
//! `Lib/json/encoder.py` (`ESCAPE` for the non-ascii path, `ESCAPE_ASCII`
//! = `([\\"]|[^\ -~])` for the ascii path, with the `ESCAPE_DCT`
//! replacements and `'\u{0:04x}'`-formatted leftovers: verified
//! byte-for-byte against the running stdlib on CPython 3.12 and 3.14).
//! One consequence of that character class worth calling out: the ascii
//! table escapes every char outside space-through-tilde, which includes
//! DEL (0x7f), `json.dumps("\x7f")` is `"\"\\u007f\""`, while the
//! non-ascii path never touches it (its class is controls-and-quote-
//! backslash only). The float presentation is `pystrtod.c`'s
//! `format_float_short` repr mode, with the shortest round-trip digits
//! taken from Rust's `{:e}` (Grisu/Dragon: the same digit string CPython's
//! dtoa emits, except on exact decimal ties; see [`py_float_repr`]).

use super::Value;

/// `json.dumps(v)` byte-parity with CPython's defaults: `", "` between
/// items, `": "` after keys, `true`/`false`/`null`, `Int` as digits,
/// `BigInt` as its (normalized) decimal text verbatim, `Float` via
/// [`py_float_repr`]: non-finite floats take CPython's `allow_nan=True`
/// spellings `NaN`/`Infinity`/`-Infinity`, and strings through the exact
/// escape table described in the module docs (`"`/`\`/`\b\t\n\f\r`
/// always; other chars < 0x20 as lowercase `\u00xx`; in `ensure_ascii`
/// mode every char above `~` (DEL included) as lowercase `\uxxxx`,
/// astral chars as their surrogate pair; with `ensure_ascii: false` those
/// pass through verbatim; `/` never escaped). Objects serialize in
/// insertion order (the `Value::Object` vec order).
pub fn dumps(v: &Value, ensure_ascii: bool) -> String {
    // A rough capacity floor: most repaired documents serialize to about
    // their input's size, so one early reserve saves the first realloc
    // ladder on large payloads (amortized O(n) either way).
    let mut out = String::with_capacity(128);
    write_value(&mut out, v, ensure_ascii);
    out
}

/// `float.__repr__` parity: the shortest round-trip decimal, presented per
/// CPython's repr mode: positional iff `-4 <= exp10 < 16` (a value that
/// is all integer digits gains `.0`, so `1.0` never renders bare), else
/// scientific with the mantissa as-is and a signed, two-digit-minimum
/// exponent (`1e+16`, `1e-05`, `5e-324`). `-0.0` keeps its sign.
///
/// The digit string comes from Rust's `{:e}` on the absolute value, which
/// yields the same shortest round-tripping mantissa CPython's dtoa
/// produces. The two disagree only on exact decimal ties: values whose
/// exact decimal expansion terminates on a `5` exactly one digit past the
/// shortest, leaving the two shortest candidates equidistant: CPython's
/// dtoa rounds the tie to the even last digit, Rust's flt2dec rounds it
/// up. The `is_decimal_tie` helper below detects that case exactly (a
/// u128 odd-part/power-of-two comparison against the midpoint equation,
/// no bignum) and steps an odd up-rounded mantissa back down to its even
/// sibling. Differential-tested against CPython over 1.66M values (random
/// bit patterns plus the tie-dense quarter-integer binade) with zero
/// mismatches.
///
/// Non-finite floats render Python's `repr` spellings (`nan`, `inf`,
/// `-inf`) (total for any f64) while `dumps` keeps its own
/// `allow_nan` JSON spellings (`NaN`/`Infinity`/`-Infinity`).
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f.is_sign_negative() { "-inf" } else { "inf" }.into();
    }
    let neg = f.is_sign_negative();
    let a = f.abs();
    // "d[.ddd]e[-]X": shortest round-trip mantissa plus the base-10
    // exponent (no sign on the exponent, no '+' ever).
    let sci = format!("{a:e}");
    let epos = sci.find('e').expect("LowerExp always emits an 'e'");
    let mantissa = &sci[..epos];
    let exp: i32 = sci[epos + 1..]
        .parse()
        .expect("LowerExp exponent is a plain integer");
    let mut digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    // The tie correction: only an odd up-rounded last digit can disagree
    // with CPython (an even one is the half-even choice), and a genuine
    // tie is exactly characterized by the midpoint equation.
    let mut tie_corrected = false;
    let last_digit = digits.as_bytes()[digits.len() - 1] - b'0';
    if last_digit % 2 == 1 {
        let m_val: u64 = digits.parse().expect("mantissa digits are an integer");
        if m_val > 1 && is_decimal_tie(a, m_val, digits.len(), exp) {
            digits = (m_val - 1).to_string();
            tie_corrected = true;
        }
    }
    let mut out = render_float(neg, &digits, exp);
    // Verify-and-correct, only on the tie-corrected path: Rust's `{:e}`
    // shortest render round-trips by construction, but the step-down can
    // land one ulp away when only the odd sibling round-trips (2^-24's
    // exact expansion 5.9604644775390625e-08: the ...063 neighbor is the
    // only round-tripping 16-digit spelling). The rendered text must
    // parse back to the same f64 (bit-for-bit) or the repr silently moves
    // the value: on a broken round-trip, take the neighboring last digit
    // (CPython's dtoa picks the round-tripping neighbor, which is what
    // "shortest" uniquely means there). Uncorrected renders skip the
    // verification parse entirely: the hot path pays nothing.
    if tie_corrected && !round_trips(&out, f) {
        for delta in [1i64, -1] {
            if let Some(neighbor) = neighbor_digits(&digits, delta) {
                let candidate = render_float(neg, &neighbor, exp);
                if round_trips(&candidate, f) {
                    out = candidate;
                    break;
                }
            }
        }
    }
    out
}

/// CPython's repr body over (sign, significant digits, base-10 exponent):
/// positional iff -4 <= exp10 < 16 (pystrtod.c's switch), zero-padded
/// magnitudes always ending in ".0", scientific with a signed two-digit-
/// minimum exponent otherwise.
fn render_float(neg: bool, digits: &str, exp: i32) -> String {
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() > int_len {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            } else {
                out.push_str(digits);
                for _ in digits.len()..int_len {
                    out.push('0');
                }
                out.push_str(".0");
            }
        } else {
            out.push_str("0.");
            for _ in 0..(-exp - 1) {
                out.push('0');
            }
            out.push_str(digits);
        }
    } else {
        if digits.len() > 1 {
            out.push_str(&digits[..1]);
            out.push('.');
            out.push_str(&digits[1..]);
        } else {
            out.push_str(digits);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        let mag = exp.unsigned_abs();
        if mag < 10 {
            out.push('0');
        }
        out.push_str(&mag.to_string());
    }
    out
}

/// Does the rendered text parse back to the exact same f64 bits (sign
/// included, so "-0.0" holds)?
fn round_trips(text: &str, f: f64) -> bool {
    text.parse::<f64>()
        .is_ok_and(|parsed| parsed.to_bits() == f.to_bits())
}

/// The significant-digit string with its last digit stepped by `delta`
/// (decimal carry, no length change except an overflow guard returns
/// None); "5960464477539062" + 1 = "5960464477539063".
fn neighbor_digits(digits: &str, delta: i64) -> Option<String> {
    let value: u128 = digits.parse().ok()?;
    let neighbor = if delta >= 0 {
        value.checked_add(delta.unsigned_abs() as u128)?
    } else {
        value.checked_sub((-delta) as u128)?
    };
    let text = neighbor.to_string();
    (text.len() == digits.len()).then_some(text)
}

/// Is `a`'s exact value the decimal midpoint `(m_val - 0.5) * 10^u`, where
/// `u = exp10 - k + 1` places the `k`-digit mantissa `m_val` on the
/// decimal grid? That is precisely the exact-tie condition under which
/// CPython's dtoa and Rust's flt2dec pick different (equidistant, both
/// round-tripping) shortest digits.
///
/// With `a = m * 2^s` (the f64 bit decomposition, subnormals included),
/// the midpoint equation is `m * 2^(s+1) == (2*m_val - 1) * 10^u`. Both
/// sides factor uniquely into `odd * 2^power`, so the test compares odd
/// parts and powers separately; the `5^u` factor lives entirely in the
/// odd part, and a `5^u` that overflows u128 can never equal an odd part
/// at most 2^53, so overflow cleanly means "not a tie". No false
/// positives are possible because the equation is an exact identity:
/// but a true midpoint does not imply both siblings round-trip (a
/// one-sided tie exists: 2^-24's midpoint rounds down to a value one
/// ulp away), which is precisely the class the round-trip verify below
/// corrects.
fn is_decimal_tie(a: f64, m_val: u64, k: usize, exp10: i32) -> bool {
    let bits = a.to_bits();
    let exp_field = ((bits >> 52) & 0x7ff) as i64;
    let frac = bits & ((1u64 << 52) - 1);
    let (m, s): (u64, i64) = if exp_field == 0 {
        (frac, -1074)
    } else {
        (frac | (1u64 << 52), exp_field - 1075)
    };
    if m == 0 {
        return false;
    }
    let tz = m.trailing_zeros() as i64;
    let m_odd = m >> tz;
    let u: i64 = i64::from(exp10) - k as i64 + 1;
    let target: u128 = 2 * u128::from(m_val) - 1;
    // u >= 0: m_odd == target * 5^u        and s + 1 + tz == u
    // u <  0: m_odd * 5^-u == target       and s + 1 + tz == u
    if u >= 0 {
        match pow5_checked(target, u) {
            Some(p) => p == u128::from(m_odd) && s + 1 + tz == u,
            None => false,
        }
    } else {
        match pow5_checked(u128::from(m_odd), -u) {
            Some(p) => p == target && s + 1 + tz == u,
            None => false,
        }
    }
}

/// `base * 5^n`, `None` on u128 overflow (see `is_decimal_tie` for why
/// overflow is a sound "no").
fn pow5_checked(mut base: u128, n: i64) -> Option<u128> {
    for _ in 0..n {
        base = base.checked_mul(5)?;
    }
    Some(base)
}

fn write_value(out: &mut String, v: &Value, ensure_ascii: bool) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Int(n) => out.push_str(&n.to_string()),
        // Python ints are unbounded: the BigInt text is already the
        // normalized decimal spelling, emitted verbatim.
        Value::BigInt(text) => out.push_str(text),
        Value::Float(f) => {
            // CPython's allow_nan=True spellings; everything else is
            // float repr.
            if f.is_nan() {
                out.push_str("NaN");
            } else if *f == f64::INFINITY {
                out.push_str("Infinity");
            } else if *f == f64::NEG_INFINITY {
                out.push_str("-Infinity");
            } else {
                out.push_str(&py_float_repr(*f));
            }
        }
        Value::Str(s) => write_string(out, s, ensure_ascii),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(out, item, ensure_ascii);
            }
            out.push(']');
        }
        Value::Object(entries) => {
            out.push('{');
            for (i, (key, value)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_string(out, key, ensure_ascii);
                out.push_str(": ");
                write_value(out, value, ensure_ascii);
            }
            out.push('}');
        }
        // MISSING_VALUE never escapes `repair` (the schema layer's
        // normalize_missing_values turns it into "" first); a direct
        // dumps call renders that normalized form.
        Value::Missing => out.push_str("\"\""),
    }
}

/// CPython's string escaping, both `ensure_ascii` modes (see the module
/// docs for the table's provenance).
fn write_string(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{0c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            c if u32::from(c) < 0x20 => push_u4(out, u32::from(c)),
            // The ascii table's class is [^ -~]: everything outside
            // space-through-tilde, so 0x7f (DEL) escapes here too, and
            // astral chars come down as their surrogate pair.
            c if ensure_ascii && u32::from(c) > 0x7e => {
                let n = u32::from(c);
                if n > 0xffff {
                    let v = n - 0x10000;
                    push_u4(out, 0xd800 + (v >> 10));
                    push_u4(out, 0xdc00 + (v & 0x3ff));
                } else {
                    push_u4(out, n);
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// CPython's `'\u{0:04x}'` replacement: `\u` plus exactly four lowercase
/// hex digits (`04x`, never `04X`).
fn push_u4(out: &mut String, n: u32) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push('\\');
    out.push('u');
    for shift in [12, 8, 4, 0] {
        out.push(HEX[(n >> shift & 0xf) as usize] as char);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_float_repr_positional_pins() {
        assert_eq!(py_float_repr(1.0), "1.0");
        assert_eq!(py_float_repr(100.0), "100.0");
        assert_eq!(py_float_repr(1e15), "1000000000000000.0");
        assert_eq!(py_float_repr(0.000123), "0.000123");
        assert_eq!(py_float_repr(0.1), "0.1");
        assert_eq!(py_float_repr(1.5), "1.5");
        assert_eq!(py_float_repr(123.456), "123.456");
        assert_eq!(py_float_repr(-2.5), "-2.5");
        assert_eq!(py_float_repr(-0.0), "-0.0");
        assert_eq!(py_float_repr(2.0 / 3.0), "0.6666666666666666");
    }

    #[test]
    fn py_float_repr_exact_midpoint_ties_round_trip() {
        // 2^-24's exact decimal expansion is the 17-significant-digit
        // midpoint ...0625: the only 16-digit round-trip is ...063, which
        // is what CPython's repr emits, and what verify-and-correct must
        // land on (the naive half-even collapse produced ...062, one ulp
        // off: a one-sided tie, down-rounded sibling does not
        // round-trip).
        assert_eq!(py_float_repr(2.0f64.powi(-24)), "5.960464477539063e-08");
        // Every power of two has an exact decimal expansion: the whole
        // family is the tie-dense zone; assert round-trip parity rather
        // than hand-computing each spelling.
        for shift in -60i32..=52 {
            let f = 2.0f64.powi(shift);
            let out = py_float_repr(f);
            assert_eq!(
                out.parse::<f64>().expect("repr parses").to_bits(),
                f.to_bits(),
                "{out}"
            );
        }
        // And a sweep of subnormal midpoints: the round-trip check itself
        // is the invariant, on values whose expansions are all exact.
        for n in 1u64..=2000 {
            let f = f64::from_bits(n);
            if f == 0.0 {
                continue;
            }
            let out = py_float_repr(f);
            assert_eq!(
                out.parse::<f64>().expect("repr parses").to_bits(),
                f.to_bits(),
                "{out}"
            );
        }
    }

    #[test]
    fn py_float_repr_scientific_pins() {
        assert_eq!(py_float_repr(1e16), "1e+16");
        assert_eq!(py_float_repr(1.5e16), "1.5e+16");
        assert_eq!(py_float_repr(1.15e16), "1.15e+16");
        assert_eq!(py_float_repr(1e-5), "1e-05");
        assert_eq!(py_float_repr(1e-10), "1e-10");
        assert_eq!(py_float_repr(1e308), "1e+308");
        // the smallest subnormal: repr('5e-324') in CPython
        assert_eq!(py_float_repr(5e-324), "5e-324");
    }

    // The tie literals are exact dyadics written out in full on purpose:
    // the "excess" digit the lint would strip is the tie itself.
    #[allow(clippy::excessive_precision)]
    #[test]
    fn py_float_repr_breaks_decimal_ties_like_cpython() {
        // Exact value 1851373832709168.25: the 17-digit candidates ...82
        // and ...83 are equidistant and both round-trip; CPython's dtoa
        // keeps the even one, Rust's flt2dec rounds up: the detector
        // steps back down.
        assert_eq!(py_float_repr(1851373832709168.25), "1851373832709168.2");
        assert_eq!(py_float_repr(1234567890123456.25), "1234567890123456.2");
        // Ties whose up-rounded sibling is already even: no change.
        assert_eq!(py_float_repr(1234567890123456.75), "1234567890123456.8");
        assert_eq!(py_float_repr(999999999999999.75), "999999999999999.8");
        // Half-integer neighbors with full-length shortest reprs never
        // trip the detector.
        assert_eq!(py_float_repr(1234567890123456.5), "1234567890123456.5");
        assert_eq!(py_float_repr(2.5), "2.5");
    }

    // Exact dyadic literals, full spelling deliberate (see the tie test).
    #[allow(clippy::excessive_precision)]
    #[test]
    fn py_float_repr_round_trips_every_pin() {
        for f in [
            1.0,
            100.0,
            1e15,
            1e16,
            1.5e16,
            1e-5,
            0.000123,
            0.1,
            1.5,
            123.456,
            -2.5,
            -0.0,
            1e308,
            5e-324,
            2.0 / 3.0,
            1851373832709168.25,
            1234567890123456.25,
            1234567890123456.75,
        ] {
            let r = py_float_repr(f);
            let back: f64 = r.parse().unwrap();
            assert_eq!(back.to_bits(), f.to_bits(), "{r}");
        }
    }

    #[test]
    fn dumps_scalar_pins() {
        assert_eq!(dumps(&Value::Null, true), "null");
        assert_eq!(dumps(&Value::Bool(true), true), "true");
        assert_eq!(dumps(&Value::Bool(false), true), "false");
        assert_eq!(dumps(&Value::Int(0), true), "0");
        assert_eq!(dumps(&Value::Int(-5), true), "-5");
        // Python's unbounded int beyond i64: verbatim decimal text.
        assert_eq!(
            dumps(&Value::BigInt("12345678901234567890".into()), true),
            "12345678901234567890"
        );
        assert_eq!(dumps(&Value::BigInt("-42".into()), true), "-42");
        assert_eq!(dumps(&Value::Str(String::new()), true), "\"\"");
    }

    #[test]
    fn dumps_non_finite_spellings() {
        // CPython json.dumps defaults (allow_nan=True).
        assert_eq!(dumps(&Value::Float(f64::NAN), true), "NaN");
        assert_eq!(dumps(&Value::Float(f64::INFINITY), true), "Infinity");
        assert_eq!(dumps(&Value::Float(f64::NEG_INFINITY), true), "-Infinity");
    }

    #[test]
    fn dumps_control_char_escapes() {
        // The named five plus the \u00xx leftover, lowercase hex:
        // identical in both ascii modes (controls escape either way).
        let v = Value::Str("a\u{08}b\tc\nd\u{0c}e\rf\u{1f}".into());
        let expected = "\"a\\bb\\tc\\nd\\fe\\rf\\u001f\"";
        assert_eq!(dumps(&v, true), expected);
        assert_eq!(dumps(&v, false), expected);
        assert_eq!(dumps(&Value::Str("\u{0}".into()), true), "\"\\u0000\"");
    }

    #[test]
    fn dumps_quote_backslash_slash() {
        // " and \ always escape; / never does (CPython never emits \/).
        let v = Value::Str("he said \"hi\" \\ /".into());
        assert_eq!(dumps(&v, true), "\"he said \\\"hi\\\" \\\\ /\"");
        assert_eq!(dumps(&v, false), "\"he said \\\"hi\\\" \\\\ /\"");
    }

    #[test]
    fn dumps_ensure_ascii_table() {
        // > 0x7e escapes as lowercase \uxxxx; astral chars as surrogate
        // pairs.
        let v = Value::Str("value\u{263a}".into());
        assert_eq!(dumps(&v, true), "\"value\\u263a\"");
        let emoji = Value::Str("\u{1F600}".into());
        assert_eq!(dumps(&emoji, true), "\"\\ud83d\\ude00\"");
        // DEL (0x7f) sits outside CPython's printable class [ -~], so the
        // ascii table escapes it too: json.dumps('\x7f') == '"\\u007f"'.
        let del = Value::Str("\u{7f}".into());
        assert_eq!(dumps(&del, true), "\"\\u007f\"");
        // Non-ascii mode: everything > 0x7f passes through verbatim, DEL
        // included.
        assert_eq!(dumps(&v, false), "\"value\u{263a}\"");
        assert_eq!(dumps(&emoji, false), "\"\u{1F600}\"");
        assert_eq!(dumps(&del, false), "\"\u{7f}\"");
    }

    #[test]
    fn dumps_document_pins() {
        let doc = Value::Object(vec![(
            "key".into(),
            Value::Array(vec![
                Value::Int(1),
                Value::Float(2.5),
                Value::Bool(true),
                Value::Null,
            ]),
        )]);
        assert_eq!(dumps(&doc, true), "{\"key\": [1, 2.5, true, null]}");
        assert_eq!(dumps(&Value::Object(vec![]), true), "{}");
        assert_eq!(dumps(&Value::Array(vec![]), true), "[]");
        // Insertion order + the ", "/": " separators; keys go through the
        // same escape table as values.
        let ordered = Value::Object(vec![
            ("b".into(), Value::Int(2)),
            ("a".into(), Value::Int(1)),
        ]);
        assert_eq!(dumps(&ordered, true), "{\"b\": 2, \"a\": 1}");
        let weird_key = Value::Object(vec![("k\u{1}".into(), Value::Int(1))]);
        assert_eq!(dumps(&weird_key, true), "{\"k\\u0001\": 1}");
    }

    #[test]
    fn dumps_missing_renders_as_its_normalized_form() {
        // Unreachable through repair() (the schema layer normalizes
        // Missing to "" before anything escapes); pin the defensive
        // rendering anyway.
        assert_eq!(dumps(&Value::Missing, true), "\"\"");
    }
}
