//! The canonical-form emitter behind `tors.content_hash`: a hand-written
//! byte emitter over an owned value tree ([`Canon`]), producing EXACTLY
//! `json.dumps(obj, sort_keys=True, separators=(",", ":"))` with the
//! default `ensure_ascii=True` and the default `allow_nan`, then
//! SHA-256 over the emitted bytes (lowercase hex, `const_hex`), the same
//! digest spelling `finalize` uses.
//!
//! # Why the emitter is hand-written (the dependency-policy answer, in the
//! record)
//!
//! No maintained crate emits this byte format. `serde_json` (already in the
//! tree as jsonschema's engine) outputs raw UTF-8 for non-ASCII strings,
//! escapes only what JSON requires (`"` and `\` and the two-char controls),
//! and has no `ensure_ascii` mode at all; `orjson` likewise emits raw UTF-8
//! with its own escape choices and sorts keys by its own comparison. The
//! contract here is the stdlib's exact wire format -- every non-ASCII
//! codepoint as a lowercase `\uXXXX`, astral codepoints as surrogate-pair
//! escapes, DEL as `\u007f` -- so the emission is this module's own ~150
//! lines rather than a dependency that approximates it. Nothing else in the
//! pipeline is hand-rolled where a crate exists: the hash is `sha2`, the
//! hex is `const_hex`, and the float/int spellings are Python's own (the
//! pyo3 walk materializes them; see `src/py/canon.rs`).
//!
//! # The escape table (the byte contract, pinned crate-side and
//! differentially from Python)
//!
//! Inside a string, the RAW bytes are exactly the printable ASCII range
//! U+0020-U+007E minus `"` (0x22) and `\` (0x5C) plus the forward slash
//! (never escaped). Everything else is escaped:
//!
//! - `"` -> `\"`, `\` -> `\\`
//! - the five short escapes: U+0008 `\b`, U+0009 `\t`, U+000A `\n`,
//!   U+000C `\f`, U+000D `\r`
//! - every other codepoint below U+0020, plus DEL (U+007F, which is outside
//!   printable ASCII), as `\u00XX` with LOWERCASE hex
//! - every codepoint >= U+080's own scalar value above ASCII as `\uXXXX`
//!   lowercase; astral codepoints (>= U+10000) as the UTF-16 surrogate-pair
//!   escapes `\uHHHH\uLLLL` (lowercase), e.g. U+1F600 -> `\ud83d\ude00`
//!
//! Non-string spellings: `null` / `true` / `false`; ints as plain decimal
//! digits (the walk pre-materializes arbitrary-precision ints and floats
//! as their Python spellings, so the emitter never formats a float or a
//! big int); containers as `{"key":value,...}` with keys already coerced
//! and sorted by the walk (sorted keys are part of the tree, not the
//! emitter's job); lists and tuples both as `[a,b,c]` (tuple == list is a
//! walk-level equivalence pinned from Python).
//!
//! The emitted bytes are pure ASCII by construction (`ensure_ascii`), a
//! structural invariant the fuzz target asserts over adversarial strings.
//!
//! # Emission strategy
//!
//! The emitter is ITERATIVE (an explicit work stack), not recursive: the
//! tree's depth is bounded only by the caller's memory, so a recursive
//! emitter would trade a heap-bound walk for a stack-overflow crash on
//! deep input. The pyo3 walk is iterative for the same reason
//! (`src/py/canon.rs`), and the two agree structurally: the walk can build
//! a tree of any depth the interpreter can hold, and the emitter can hash
//! it back out at the same depth.
//!
//! Digits are written into a fixed stack buffer (no per-int `format!`
//! allocation: the i64 fast path's whole point, measured against the
//! Python-repr fallback in `src/py/canon.rs`'s docs).

use sha2::{Digest, Sha256};

/// The owned value tree the pyo3 walk produces under the GIL and this
/// emitter consumes under `py.detach`. Every leaf is pre-materialized:
/// strings as their Rust `String` content (the walk's `to_str` borrow,
/// one copy per string -- the O(tree) GIL-held residue the crate GIL model
/// documents), arbitrary-precision ints and floats as their exact PYTHON
/// spellings (decimal digits / `repr` / the `NaN`/`Infinity` literals),
/// so the emitter does no formatting except i64 decimal digits.
///
/// The bench (`benches/canon.rs`) and the fuzz target
/// (`fuzz/fuzz_targets/canon.rs`) build `Canon` trees directly, driving
/// this emitter without an interpreter.
#[derive(Debug, Clone, PartialEq)]
pub enum Canon {
    /// Python `None` -> `null`.
    Null,
    /// `True`/`False` -> `true`/`false`.
    Bool(bool),
    /// An int inside the i64 fast path; plain decimal digits at emission.
    Int(i64),
    /// An int beyond i64: its Python `str()` spelling, materialized by the
    /// walk (arbitrary precision, and the interpreter's
    /// `sys.set_int_max_str_digits` limit already applied at both ends).
    BigInt(String),
    /// A float's exact spelling: `repr` for finite values (Python's own,
    /// never reimplemented here), `NaN`/`Infinity`/`-Infinity` for the
    /// non-finite ones (json's `allow_nan` literals).
    Float(String),
    /// A string's content; the emitter applies the escape table.
    Str(String),
    /// A list or a tuple (they serialize identically; the walk normalizes
    /// both into `Seq`, the equivalence pinned from Python).
    Seq(Vec<Canon>),
    /// A dict: pairs of (coerced key string, value), ALREADY sorted by the
    /// walk (sort-before-stringify: the walk sorts the original key
    /// objects, then coerces; see `src/py/canon.rs`).
    Map(Vec<(String, Canon)>),
}

/// The byte sink the single emitter feeds: the digest path updates
/// SHA-256 incrementally (no intermediate canonical-form buffer, the whole
/// point of the one-detach shape), and `canonical_bytes` collects into a
/// `Vec` for the crate-side pins and the fuzz target's structural
/// assertions.
trait Sink {
    fn emit(&mut self, bytes: &[u8]);
}

impl Sink for Sha256 {
    fn emit(&mut self, bytes: &[u8]) {
        self.update(bytes);
    }
}

impl Sink for Vec<u8> {
    fn emit(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

/// The lowercase hex digits (`\u00XX` spellings are lowercase in
/// `json.dumps`, verified against the running interpreter, not assumed).
const HEX: &[u8; 16] = b"0123456789abcdef";

/// Writes `v` as exactly 4 lowercase hex digits (the `\uXXXX` payload).
fn hex4(dst: &mut [u8], v: u32) {
    dst[0] = HEX[((v >> 12) & 0xF) as usize];
    dst[1] = HEX[((v >> 8) & 0xF) as usize];
    dst[2] = HEX[((v >> 4) & 0xF) as usize];
    dst[3] = HEX[(v & 0xF) as usize];
}

/// Is `ch` a raw (unescaped) character: printable ASCII minus the two
/// JSON-forced escapes? The forward slash sits inside this range and is
/// never escaped (verified: `json.dumps` leaves `/` raw).
#[inline]
fn is_raw(ch: char) -> bool {
    let v = ch as u32;
    (0x20..=0x7E).contains(&v) && v != 0x22 && v != 0x5C
}

/// Emits one escaped character (never a raw one) into `sink`, straight
/// into a fixed stack buffer: 2 bytes for the short escapes, 6 for
/// `\uXXXX`, 12 for an astral surrogate pair.
fn write_escape<S: Sink>(ch: char, sink: &mut S) {
    let mut buf = [0u8; 12];
    let len = match ch {
        '"' => {
            buf[..2].copy_from_slice(b"\\\"");
            2
        }
        '\\' => {
            buf[..2].copy_from_slice(b"\\\\");
            2
        }
        '\u{08}' => {
            buf[..2].copy_from_slice(b"\\b");
            2
        }
        '\t' => {
            buf[..2].copy_from_slice(b"\\t");
            2
        }
        '\n' => {
            buf[..2].copy_from_slice(b"\\n");
            2
        }
        '\u{0C}' => {
            buf[..2].copy_from_slice(b"\\f");
            2
        }
        '\r' => {
            buf[..2].copy_from_slice(b"\\r");
            2
        }
        // The remaining controls (U+0000-U+001F minus the five above) and
        // DEL: \u00 plus the value's two low hex digits (the value is
        // always < 0x100 here).
        _ if (ch as u32) < 0x20 || ch as u32 == 0x7F => {
            let v = ch as u32;
            buf[..4].copy_from_slice(b"\\u00");
            buf[4] = HEX[((v >> 4) & 0xF) as usize];
            buf[5] = HEX[(v & 0xF) as usize];
            6
        }
        // BMP non-ASCII: \uXXXX.
        _ if (ch as u32) < 0x10000 => {
            buf[..2].copy_from_slice(b"\\u");
            hex4(&mut buf[2..6], ch as u32);
            6
        }
        // Astral: the UTF-16 surrogate pair, both halves lowercase.
        _ => {
            let cp = (ch as u32) - 0x10000;
            buf[..2].copy_from_slice(b"\\u");
            hex4(&mut buf[2..6], 0xD800 + (cp >> 10));
            buf[6..8].copy_from_slice(b"\\u");
            hex4(&mut buf[8..12], 0xDC00 + (cp & 0x3FF));
            12
        }
    };
    sink.emit(&buf[..len]);
}

/// Emits a string: quote, then raw runs copied in place and escaped
/// characters via [`write_escape`], then quote. Prose is dominated by raw
/// runs, so the common path is one slice emit per run and the escape
/// machinery only wakes on the rare character.
fn emit_string<S: Sink>(s: &str, sink: &mut S) {
    sink.emit(b"\"");
    let bytes = s.as_bytes();
    let mut run_start = 0usize;
    for (idx, ch) in s.char_indices() {
        if is_raw(ch) {
            continue;
        }
        if run_start < idx {
            sink.emit(&bytes[run_start..idx]);
        }
        write_escape(ch, sink);
        run_start = idx + ch.len_utf8();
    }
    if run_start < bytes.len() {
        sink.emit(&bytes[run_start..]);
    }
    sink.emit(b"\"");
}

/// Emits `i` as plain decimal digits into a fixed stack buffer (20 bytes
/// covers every i64 including `i64::MIN`; no `format!` allocation -- this
/// is the emission half of the i64 fast path).
fn emit_int<S: Sink>(i: i64, sink: &mut S) {
    let mut buf = [0u8; 20];
    let negative = i < 0;
    let mut val = i.unsigned_abs();
    let mut pos = buf.len();
    loop {
        pos -= 1;
        buf[pos] = b'0' + (val % 10) as u8;
        val /= 10;
        if val == 0 {
            break;
        }
    }
    if negative {
        pos -= 1;
        buf[pos] = b'-';
    }
    sink.emit(&buf[pos..]);
}

/// The pending work of the iterative emitter: a subtree to emit, a map key
/// to emit as an escaped string, or a structural token (`[` `]` `,` `{`
/// `}` `:`) to copy.
enum Work<'a> {
    Node(&'a Canon),
    Key(&'a str),
    Token(&'static [u8]),
}

/// Tears a tree down ITERATIVELY: the default recursive `Drop` a deeply
/// nested `Vec` chain gets would overflow the call stack at exactly the
/// depths the iterative walk and emitter exist to support (measured: a
/// 100k-deep tree's default teardown overflows an 8 MiB stack), so the
/// teardown drains children into a work stack the same way the emitter
/// does. Each `Drop` call handles one node's own fields (the enum shell,
/// its `String`s); nesting never recurses.
impl Drop for Canon {
    fn drop(&mut self) {
        let mut stack: Vec<Canon> = Vec::new();
        match self {
            Canon::Seq(items) => stack.append(items),
            Canon::Map(pairs) => stack.extend(pairs.drain(..).map(|(_, value)| value)),
            _ => {}
        }
        while let Some(mut node) = stack.pop() {
            match &mut node {
                Canon::Seq(items) => stack.append(items),
                Canon::Map(pairs) => stack.extend(pairs.drain(..).map(|(_, value)| value)),
                _ => {}
            }
        }
    }
}

/// The one emitter every consumer drives (the digest path and the
/// byte-collecting path), iterative over an explicit work stack: no
/// recursion, so tree depth costs heap (the stack of `Work` frames), never
/// the call stack.
fn emit_into<S: Sink>(tree: &Canon, sink: &mut S) {
    let mut stack: Vec<Work<'_>> = Vec::with_capacity(16);
    stack.push(Work::Node(tree));
    while let Some(work) = stack.pop() {
        match work {
            Work::Token(bytes) => sink.emit(bytes),
            Work::Key(key) => emit_string(key, sink),
            Work::Node(node) => match node {
                Canon::Null => sink.emit(b"null"),
                Canon::Bool(true) => sink.emit(b"true"),
                Canon::Bool(false) => sink.emit(b"false"),
                Canon::Int(i) => emit_int(*i, sink),
                Canon::BigInt(spelling) | Canon::Float(spelling) => sink.emit(spelling.as_bytes()),
                Canon::Str(s) => emit_string(s, sink),
                Canon::Seq(items) => {
                    sink.emit(b"[");
                    stack.push(Work::Token(b"]"));
                    for (i, child) in items.iter().enumerate().rev() {
                        if i + 1 < items.len() {
                            stack.push(Work::Token(b","));
                        }
                        stack.push(Work::Node(child));
                    }
                }
                Canon::Map(pairs) => {
                    sink.emit(b"{");
                    stack.push(Work::Token(b"}"));
                    for (i, (key, value)) in pairs.iter().enumerate().rev() {
                        if i + 1 < pairs.len() {
                            stack.push(Work::Token(b","));
                        }
                        stack.push(Work::Node(value));
                        stack.push(Work::Token(b":"));
                        stack.push(Work::Key(key));
                    }
                }
            },
        }
    }
}

/// The canonical form's bytes (the `json.dumps(..., sort_keys=True,
/// separators=(",", ":"))` wire format): for the crate-side literal pins
/// and the fuzz target's structural assertions (all-ASCII, determinism).
pub fn canonical_bytes(tree: &Canon) -> Vec<u8> {
    let mut out = Vec::new();
    emit_into(tree, &mut out);
    out
}

/// SHA-256 over the canonical form, lowercase hex: byte-identical with
/// `hashlib.sha256(json.dumps(obj, sort_keys=True,
/// separators=(",", ":")).encode("utf-8")).hexdigest()`, the expression
/// `tors.content_hash` replaces. Streams into the hasher (no intermediate
/// canonical-form buffer), the pass that runs under `py.detach`.
pub fn digest_hex(tree: &Canon) -> String {
    let mut hasher = Sha256::new();
    emit_into(tree, &mut hasher);
    const_hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(text: &str) -> Canon {
        Canon::Str(text.to_string())
    }

    fn seq(items: Vec<Canon>) -> Canon {
        Canon::Seq(items)
    }

    fn map(pairs: Vec<(&str, Canon)>) -> Canon {
        Canon::Map(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    fn str_bytes(text: &str) -> Vec<u8> {
        canonical_bytes(&s(text))
    }

    #[test]
    fn ascii_escape_battery_every_codepoint_below_0x80() {
        // The expected spelling, re-derived independently of the emitter's
        // table: the five short escapes, \u00xx lowercase for the other
        // controls and DEL, the two forced escapes, raw for the rest.
        for cp in 0u32..0x80 {
            let ch = char::from_u32(cp).unwrap();
            let expected: String = match cp {
                0x08 => "\\b".into(),
                0x09 => "\\t".into(),
                0x0A => "\\n".into(),
                0x0C => "\\f".into(),
                0x0D => "\\r".into(),
                0x22 => "\\\"".into(),
                0x5C => "\\\\".into(),
                c if c < 0x20 || c == 0x7F => format!("\\u{c:04x}"),
                _ => ch.to_string(),
            };
            assert_eq!(
                str_bytes(&ch.to_string()),
                format!("\"{expected}\"").into_bytes(),
                "U+{cp:04X}"
            );
        }
    }

    #[test]
    fn bmp_and_astral_boundary_escape_spellings() {
        for (cp, expected) in [
            (0x80, "\\u0080"),
            (0xE9, "\\u00e9"),
            (0x7FF, "\\u07ff"),
            (0x800, "\\u0800"),
            (0xD7FF, "\\ud7ff"),
            (0xE000, "\\ue000"),
            (0xFFFD, "\\ufffd"),
            (0xFFFF, "\\uffff"),
            (0x10000, "\\ud800\\udc00"),
            (0x1F600, "\\ud83d\\ude00"),
            (0x10FFFF, "\\udbff\\udfff"),
        ] {
            let ch = char::from_u32(cp).unwrap();
            assert_eq!(
                str_bytes(&ch.to_string()),
                format!("\"{expected}\"").into_bytes(),
                "U+{cp:05X}"
            );
        }
    }

    #[test]
    fn the_last_astral_codepoint_and_the_first() {
        // The extreme astral values: the surrogate-pair arithmetic at both
        // ends of the 0x10000..=0x10FFFF range.
        assert_eq!(str_bytes("\u{10FFFF}"), b"\"\\udbff\\udfff\"".to_vec());
        assert_eq!(str_bytes("\u{10000}"), b"\"\\ud800\\udc00\"".to_vec());
    }

    #[test]
    fn raw_runs_copy_in_place_and_escapes_intersperse() {
        assert_eq!(str_bytes("plain"), b"\"plain\"".to_vec());
        assert_eq!(str_bytes("a\tb"), b"\"a\\tb\"".to_vec());
        assert_eq!(str_bytes("\t mid \n"), b"\"\\t mid \\n\"".to_vec());
        assert_eq!(str_bytes("a/b"), b"\"a/b\"".to_vec()); // never escaped
        assert_eq!(
            str_bytes("say \"hi\" \\ ok"),
            b"\"say \\\"hi\\\" \\\\ ok\"".to_vec()
        );
        assert_eq!(
            str_bytes("a\u{0}\u{8}\u{1F}\u{7F}\u{E9}\u{1F600}\t"),
            b"\"a\\u0000\\b\\u001f\\u007f\\u00e9\\ud83d\\ude00\\t\"".to_vec()
        );
    }

    #[test]
    fn empty_and_nested_containers() {
        assert_eq!(canonical_bytes(&seq(vec![])), b"[]".to_vec());
        assert_eq!(canonical_bytes(&map(vec![])), b"{}".to_vec());
        assert_eq!(
            canonical_bytes(&seq(vec![seq(vec![]), map(vec![("k", seq(vec![]))])])),
            b"[[],{\"k\":[]}]".to_vec()
        );
    }

    #[test]
    fn int_digits_cover_the_i64_boundaries() {
        assert_eq!(canonical_bytes(&Canon::Int(0)), b"0".to_vec());
        assert_eq!(canonical_bytes(&Canon::Int(-1)), b"-1".to_vec());
        assert_eq!(
            canonical_bytes(&Canon::Int(i64::MAX)),
            b"9223372036854775807".to_vec()
        );
        assert_eq!(
            canonical_bytes(&Canon::Int(i64::MIN)),
            b"-9223372036854775808".to_vec()
        );
    }

    #[test]
    fn pre_materialized_spellings_emit_verbatim() {
        // 2**100's actual decimal spelling (the walk materializes it via
        // Python's own int->str; the emitter copies it verbatim).
        assert_eq!(
            canonical_bytes(&Canon::BigInt("1267650600228229401496703205376".into())),
            b"1267650600228229401496703205376".to_vec()
        );
        assert_eq!(
            canonical_bytes(&Canon::Float("1e+16".into())),
            b"1e+16".to_vec()
        );
        assert_eq!(
            canonical_bytes(&Canon::Float("NaN".into())),
            b"NaN".to_vec()
        );
        assert_eq!(
            canonical_bytes(&Canon::Float("-Infinity".into())),
            b"-Infinity".to_vec()
        );
    }

    #[test]
    fn tree_literals_the_full_structural_shape() {
        let tree = map(vec![
            ("a", seq(vec![Canon::Int(1), s("x"), Canon::Null])),
            ("b", map(vec![("k", Canon::Bool(true))])),
        ]);
        assert_eq!(
            canonical_bytes(&tree),
            b"{\"a\":[1,\"x\",null],\"b\":{\"k\":true}}".to_vec()
        );
    }

    #[test]
    fn emitted_bytes_are_always_ascii() {
        let tree = map(vec![
            ("cl\u{E9}", seq(vec![s("h\u{E9}llo"), s("\u{10FFFF}")])),
            ("\u{1F600}", s("\u{E01F0}")),
        ]);
        let bytes = canonical_bytes(&tree);
        assert!(bytes.is_ascii());
        assert!(bytes.contains(&b'\\')); // and it escaped to get there
    }

    #[test]
    fn digest_is_stable_64_lowercase_hex() {
        let tree = map(vec![("a", Canon::Int(1))]);
        let d = digest_hex(&tree);
        assert_eq!(d.len(), 64);
        assert_eq!(d, d.to_lowercase());
        assert!(
            d.bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        );
        assert_eq!(d, digest_hex(&tree)); // deterministic
    }

    #[test]
    fn digest_matches_sha256_over_the_canonical_bytes() {
        // The digest path (streaming) and the byte-collecting path must
        // agree: same emitter, two sinks.
        let tree = seq(vec![
            Canon::Int(-42),
            s("esc\tape\u{1F600}"),
            map(vec![("k", Canon::Float("0.1".into())), ("j", Canon::Null)]),
        ]);
        let bytes = canonical_bytes(&tree);
        assert_eq!(digest_hex(&tree), const_hex::encode(Sha256::digest(&bytes)));
    }

    #[test]
    fn deep_trees_emit_without_recursion() {
        // The work stack, not the call stack: a tree far deeper than any
        // plausible call-stack budget (the Python-side differential pins
        // the same depth through the real walk).
        let mut tree = Canon::Null;
        for _ in 0..100_000 {
            tree = Canon::Seq(vec![tree]);
        }
        let bytes = canonical_bytes(&tree);
        assert_eq!(bytes.len(), 100_000 * 2 + 4); // "[" * depth + "null" + "]" * depth
        assert_eq!(bytes.first(), Some(&b'['));
        assert_eq!(bytes.last(), Some(&b']'));
        // And the map flavor: each level emits {"k":[ ... ]} (8 bytes).
        let mut tree = Canon::Null;
        for _ in 0..50_000 {
            tree = Canon::Map(vec![("k".to_string(), Canon::Seq(vec![tree]))]);
        }
        let bytes = canonical_bytes(&tree);
        assert_eq!(bytes.len(), 50_000 * 8 + 4);
    }
}
