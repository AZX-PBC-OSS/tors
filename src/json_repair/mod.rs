//! LLM-JSON repair: a Rust port of json_repair (upstream:
//! https://github.com/mangiucugna/json_repair by Stefano Baccianella, MIT,
//! pinned at commit 251d141786d0f6ff561f6ec04d90188a338e2470, version
//! 0.63.4) — the repair parser for malformed JSON from LLMs, its
//! schema-guided alignment/coercion, plus bounded tors-native extensions.
//! Provenance note: every behavior here is a faithful port of the upstream
//! sources EXCEPT the divergences documented in this module tree (see each
//! file's header); the ported pytest corpus in tests/test_json_repair_*.py
//! (core/schema/native/e2e/parity) and the differential suite against the
//! pinned `json-repair` package keep the parity honest.
//!
//! # Layout
//!
//! A directory module (the one deviation from this crate's flat
//! `*_impl.rs`-per-feature convention, deliberately mirroring upstream's own
//! file split — the port is ~3000 lines and one flat file would be
//! unmaintainable; the convention was sized for ~500-line features):
//!
//! - `mod.rs` (this file): the [`Value`] tree with PYTHON value semantics
//!   (`==`, truthiness, structural `is_same_object`), [`RepairConfig`]/
//!   [`Diagnostic`], and the [`repair`] orchestration (json_repair.py's
//!   `repair_json` flow: fence pre-pass, strict fast path, schema fast path,
//!   repair parser, empty-string sentinel).
//! - `dumps.rs`: `json.dumps` byte-parity serializer + `float.__repr__`
//!   parity.
//! - `strict.rs`: `json.loads` byte-parity parser (the fast path and the
//!   `raw_decode` suffix probe).
//! - `parser.rs`: the repair parser's `Parser` core (json_parser.py):
//!   shared utilities + `parse_json` orchestration + top-level loop +
//!   salvage fragment loop + number/comment parsing + the depth guard.
//! - `string.rs`: parse_string.py + parse_string_helpers — the string repair
//!   heuristics (the heart of the port).
//! - `object.rs`/`array.rs`/`parenthesized.rs`: parse_object.py,
//!   parse_array.py, parser_parenthesized.py.
//! - `../json_schema_impl.rs`: schema_repair.py + parser_schema.py — the
//!   schema-guided alignment layer (coercion, fills, unions, `$ref`,
//!   validation via the `jsonschema` crate, and the tors-native typo-remap/
//!   enum-suggestion/date-normalization extensions).
//!
//! # Scope (upstream features NOT ported — documented divergences)
//!
//! `stream_stable`, the `logging=True` repair log (superseded by
//! [`repair_json_diagnostics`], which records schema-layer actions and
//! tors-native suggestions; parser-level narration is a follow-up), the
//! file-descriptor flavors (`load`/`from_file`/`json_fd`), the CLI,
//! `json.dumps`
//! pass-through kwargs beyond `ensure_ascii`, and the `pattern` keyword is
//! enforced (upstream delegates to the Python `jsonschema` package; tors
//! uses the Rust one — see `json_schema_impl`'s docs for the small set of
//! validation-boundary divergences).

use crate::fence_impl;

mod array;
mod dumps;
mod object;
mod parenthesized;
mod parser;
mod strict;
mod string;

pub use dumps::{dumps, py_float_repr};
pub use strict::{loads_strict, raw_decode};

/// The EXACT decimal expansion of an integral f64 (Python's unbounded
/// `int(float)` semantics — `int(1e30)` is 1000000000000000019884624838656):
/// fixed-point `{:.*}` formatting renders the true expansion of the binary
/// value. `-0` normalizes to `0` (Python's `int(-0.0)`). Non-integral or
/// non-finite floats give None (no exact integer exists).
pub(crate) fn exact_decimal_of_float(f: f64) -> Option<String> {
    if !f.is_finite() || f.fract() != 0.0 {
        return None;
    }
    let text = format!("{f:.0}");
    Some(if text == "-0" { "0".to_string() } else { text })
}
pub(crate) use parser::DEADLINE_TAG;
pub(crate) use parser::normalize_big_int_text;

use crate::json_schema_impl::SchemaRepairer;

/// json_repair's `STRING_DELIMITERS`: the quote characters the repair parser
/// recognizes as string delimiters (ASCII double/single plus the curly pair).
pub(crate) const STRING_DELIMITERS: [char; 4] = ['"', '\'', '\u{201C}', '\u{201D}'];

/// parse_number.py's `NUMBER_CHARS`: everything a "number" run may contain
/// before the fallback heuristics (currency commas, fraction slashes,
/// exponents, Python digit-group underscores) sort it out.
pub(crate) const NUMBER_CHARS: [char; 17] = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.', 'e', 'E', '/', ',', '_',
];

/// The container-nesting cap for both parsers (strict and repair): beyond
/// it the parse fails with json_repair's normalized message instead of a
/// stack overflow. Python raises an uncaught `RecursionError` near its own
/// recursion limit at roughly this depth for the repair parser; tors
/// normalizes to `ValueError` at a fixed, lower, documented threshold
/// (see `repair`'s docs).
pub(crate) const MAX_NESTING: usize = 200;

/// Object sizes above which py_eq/is_same_object hash one side instead of
/// scanning it per key (CPython dict cost; small objects skip the map
/// build).
pub(crate) const OBJECT_COMPARE_HASH_MIN: usize = 16;

/// json_repair's `JSONReturnType` (`dict | list | str | float | int | bool |
/// None`) plus its `MISSING_VALUE` sentinel, with PYTHON value semantics:
///
/// - [`Value::Object`] preserves insertion order; a duplicate key updates
///   the value IN PLACE at its first-occurrence position (CPython `dict`
///   semantics — the position of `d[k] = v` on an existing key).
/// - [`Value::BigInt`] is Python's unbounded `int` beyond i64 range, held as
///   normalized decimal text (an optional `-`, then digits with no leading
///   zeros; never just `-`), serialized verbatim.
/// - [`Value::Missing`] is `MISSING_VALUE`: it exists only mid-repair (the
///   schema layer turns it into fills/`""`) and never escapes [`repair`].
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    BigInt(String),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
    Missing,
}

/// An insertion-ordered object accumulator: `Value::Object`'s pinned Vec
/// shape with CPython dict cost — a side index maps each key to its entry
/// slot, so the duplicate-key update-in-place at first-occurrence-position
/// semantics stay exact while member insertion stays O(1) on large
/// objects. The index is LAZY: small objects (the common LLM shape —
/// a handful of keys) keep the cheaper linear scan and never allocate a
/// map; it materializes only past [`LINEAR_OBJECT_MAX`] members, where the
/// O(n²) scan would bite (CPython dicts hash; the Vec shape cannot).
pub(crate) struct ObjectBuilder {
    entries: Vec<(String, Value)>,
    index: Option<std::collections::HashMap<String, usize>>,
}

/// Below this member count the linear duplicate scan beats a HashMap's
/// constants; above it the map wins (measured crossover, small-object
/// LLM documents).
const LINEAR_OBJECT_MAX: usize = 32;

impl ObjectBuilder {
    pub(crate) fn new() -> Self {
        ObjectBuilder {
            entries: Vec::new(),
            index: None,
        }
    }

    /// CPython `dict[k] = v`: insert at the end when new, update IN PLACE
    /// (keeping the first-occurrence position) when the key exists.
    pub(crate) fn insert(&mut self, key: String, value: Value) {
        if let Some(index) = &mut self.index {
            match index.get(&key) {
                Some(&slot) => self.entries[slot].1 = value,
                None => {
                    index.insert(key.clone(), self.entries.len());
                    self.entries.push((key, value));
                }
            }
            return;
        }
        if let Some(slot) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
            return;
        }
        self.entries.push((key, value));
        if self.entries.len() > LINEAR_OBJECT_MAX {
            self.index = Some(
                self.entries
                    .iter()
                    .enumerate()
                    .map(|(slot, (k, _))| (k.clone(), slot))
                    .collect(),
            );
        }
    }

    /// The most recently inserted entry (the array-continuation merge
    /// reads and rewrites it in place).
    pub(crate) fn last_mut(&mut self) -> Option<&mut (String, Value)> {
        self.entries.last_mut()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn finish(self) -> Value {
        Value::Object(self.entries)
    }
}

impl Default for ObjectBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Value {
    /// Is this key present? CPython `dict.__contains__` — first occurrence
    /// position, `None` if absent.
    pub(crate) fn object_get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// CPython `dict[k] = v`: insert at the end when new, update IN PLACE
    /// (keeping the first-occurrence position) when the key exists.
    pub(crate) fn object_insert(&mut self, key: String, value: Value) {
        if let Value::Object(entries) = self
            && let Some(slot) = entries.iter_mut().find(|(k, _)| *k == key)
        {
            slot.1 = value;
        } else if let Value::Object(entries) = self {
            entries.push((key, value));
        }
    }

    /// Python truthiness: `None`/`False`/`0`/`0.0`/`-0.0`/`""`/`[]`/`{}`
    /// are falsy; `float("nan")` is TRUTHY (CPython semantics, which the
    /// top-level multi-value logic depends on).
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null | Value::Bool(false) => false,
            Value::Bool(true) => true,
            Value::Int(n) => *n != 0,
            Value::BigInt(digits) => digits.as_str() != "0" && digits.as_str() != "-0",
            Value::Float(f) => {
                // NaN compares unequal to 0.0 and is truthy in CPython.
                *f != 0.0
            }
            Value::Str(s) => !s.is_empty(),
            Value::Array(items) => !items.is_empty(),
            Value::Object(entries) => !entries.is_empty(),
            // The sentinel is consumed by the schema layer before any
            // truthiness site can see it; defaulting to falsy matches ""
            // (its normalized form).
            Value::Missing => false,
        }
    }

    /// Python `==` over the JSON value domain:
    ///
    /// - cross-numeric: `Int`/`BigInt` compare exactly (BigInt vs Int via
    ///   decimal-text comparison — no precision loss); `1 == 1.0` holds;
    ///   `True == 1` holds (Python `bool` is an `int`); `NaN != NaN` holds
    ///   (IEEE); Int-vs-Float compares the float's EXACT decimal expansion
    ///   against the int's digits — Python's arbitrary-precision semantics,
    ///   no f64-coercion edge at any magnitude.
    /// - containers: dicts compare order-INSENSITIVELY on keys, recursively
    ///   on values; lists elementwise.
    pub fn py_eq(&self, other: &Value) -> bool {
        /// Exact numeric identity between the two int-carrying shapes
        /// (`Int`, `BigInt`, and `Bool` as 0/1), via decimal text so no
        /// precision is lost on huge ints.
        fn int_text(v: &Value) -> Option<String> {
            match v {
                Value::Bool(b) => Some(if *b { "1".into() } else { "0".into() }),
                Value::Int(n) => Some(n.to_string()),
                Value::BigInt(s) => Some(s.clone()),
                _ => None,
            }
        }
        fn int_kind(v: &Value) -> bool {
            matches!(v, Value::Bool(_) | Value::Int(_) | Value::BigInt(_))
        }
        /// Python's int/float comparison is EXACT: equal iff the float is
        /// integral and its exact decimal expansion equals the int's
        /// digits (`9223372036854775807 == 9223372036854775808.0` is
        /// False; a naive f64 coercion of the int says true).
        fn exact_int_float_eq(int: &Value, float: &Value) -> bool {
            match float {
                Value::Float(f) => match exact_decimal_of_float(*f) {
                    Some(expansion) => int_text(int).is_some_and(|t| t == expansion),
                    None => false,
                },
                _ => false,
            }
        }
        match (self, other) {
            (Value::Null, Value::Null) => true,
            // bool-to-bool first so Bool never falls into the numeric lanes.
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (a, b) if int_kind(a) && int_kind(b) => int_text(a) == int_text(b),
            (a, b) if int_kind(a) && matches!(b, Value::Float(_)) => exact_int_float_eq(a, b),
            (a, b) if matches!(a, Value::Float(_)) && int_kind(b) => exact_int_float_eq(b, a),
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::Array(a), Value::Array(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.py_eq(y))
            }
            (Value::Object(a), Value::Object(b)) => {
                // Python dict == is key-set symmetric with recursive value
                // compare; hashing one side restores CPython's O(1) per-key
                // cost on large objects (the linear any() is O(n²)).
                if a.len() != b.len() {
                    return false;
                }
                if b.len() > OBJECT_COMPARE_HASH_MIN {
                    let slots: std::collections::HashMap<&str, &Value> =
                        b.iter().map(|(k, v)| (k.as_str(), v)).collect();
                    return a
                        .iter()
                        .all(|(k, v)| slots.get(k.as_str()).is_some_and(|w| v.py_eq(w)));
                }
                a.iter()
                    .all(|(k, v)| b.iter().any(|(k2, v2)| k == k2 && v.py_eq(v2)))
            }
            _ => false,
        }
    }

    /// json_repair's `ObjectComparer.is_same_object`: EXACT Python type
    /// identity (`type(a) is type(b)` — so `True` and `1` are NOT the same
    /// object even though equal; `Int` and `BigInt` both ARE `int`), the
    /// same key SET for dicts (any order), and a recursive structural match
    /// where scalar VALUES are ignored (the same shape, not the same data).
    /// This is the "repeated object is an update, keep the newest" test the
    /// top-level multi-value loop runs.
    pub fn is_same_object(&self, other: &Value) -> bool {
        fn int_kind(v: &Value) -> bool {
            // Python: type(x) is int — true for Int and BigInt, NOT for bool.
            matches!(v, Value::Int(_) | Value::BigInt(_))
        }
        match (self, other) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(_), Value::Bool(_)) => true,
            (a, b) if int_kind(a) && int_kind(b) => true,
            (Value::Float(_), Value::Float(_)) => true,
            (Value::Str(_), Value::Str(_)) => true,
            (Value::Array(a), Value::Array(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.is_same_object(y))
            }
            (Value::Object(a), Value::Object(b)) => {
                // Same shape compare, values' types still matched — same
                // hashing threshold as py_eq (this runs per element pair
                // in the top-level update check).
                if a.len() != b.len() {
                    return false;
                }
                if b.len() > OBJECT_COMPARE_HASH_MIN {
                    let slots: std::collections::HashMap<&str, &Value> =
                        b.iter().map(|(k, v)| (k.as_str(), v)).collect();
                    return a
                        .iter()
                        .all(|(k, v)| slots.get(k.as_str()).is_some_and(|w| v.is_same_object(w)));
                }
                a.iter()
                    .all(|(k, v)| b.iter().any(|(k2, v2)| k == k2 && v.is_same_object(v2)))
            }
            _ => false,
        }
    }

    /// The Python type name for error messages (json_repair formats
    /// `type(value).__name__`): the CPython spellings, not Rust's.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Object(_) => "dict",
            Value::Array(_) => "list",
            Value::Str(_) => "str",
            Value::Int(_) | Value::BigInt(_) => "int",
            Value::Float(_) => "float",
            Value::Bool(_) => "bool",
            Value::Null => "NoneType",
            // Never user-visible: the schema layer normalizes Missing to ""
            // before any message that would name it.
            Value::Missing => "str",
        }
    }
}

/// The number-separator convention a caller KNOWS about their model's
/// output. `Auto` (the default) assumes en-US for the separator-ambiguous
/// shapes — but only schema-checked and disclosed: both readings are
/// extracted and filtered by the declared type and the schema, a single
/// survivor is the deterministic answer, and when both survive the en-US
/// reading wins WITH a suggestion diagnostic naming the discarded
/// reading's `locale=` override (never a silent 1000x guess; an unfixed
/// value round-trips while a mis-fixed one silently corrupts). Shapes no
/// locale could resolve refuse with the disambiguation hint, safe to feed
/// back to the model for a retry. A BCP 47 tag (or a custom separators
/// dict, resolved in the py layer) becomes a
/// [`LocaleSpec`](crate::json_schema_impl::LocaleSpec) — the CLDR decimal
/// and grouping separators for that locale — which makes every form
/// deterministic: the locale's own separators are normalized onto `.`/`,`
/// before parsing, so German reads `"1,234"` as 1.234 and English as
/// 1234, by data rather than by guess.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NumericLocale {
    #[default]
    Auto,
    Known(crate::json_schema_impl::LocaleSpec),
}

/// The knobs of one [`repair`] call. Mirrors json_repair's `repair_json`
/// arguments (minus the unported ones, see the module docs): `skip_json_loads`
/// skips the whole-input strict-parse fast path (NOT the parser-internal
/// suffix probe — upstream keeps that one on for string inputs, and so does
/// tors); `strict` turns repair heuristics into `ValueError`s; `salvage`
/// enables schema-guided salvage mode (requires `schema`); `schema` is a
/// JSON Schema as a [`Value::Object`] or boolean schema; `diagnostics`
/// collects the action log; `locale` resolves the locale-ambiguous
/// numeric formats (Auto: en-US assumed, schema-checked, disclosed).
#[derive(Clone, Debug, Default)]
pub struct RepairConfig {
    pub skip_json_loads: bool,
    pub strict: bool,
    pub salvage: bool,
    pub schema: Option<Value>,
    pub diagnostics: bool,
    pub locale: NumericLocale,
    pub deadline_ms: Option<f64>,
}

/// One recorded action from [`repair`] — the structured successor of
/// json_repair's `logging=True` text log. `action` is a closed vocabulary
/// (see the `repair_json_diagnostics` docs for the table); `path` is
/// json_repair's `$`-rooted instance path; `from`/`to` carry the value
/// before/after where a single value moved; `suggestion` is the
/// "did you mean" payload for report-only hints.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub action: String,
    pub path: String,
    pub detail: String,
    pub from: Option<Value>,
    pub to: Option<Value>,
    pub suggestion: Option<String>,
}

/// The core of `tors.repair_json` / `tors.repair_json_loads` /
/// `tors.repair_json_diagnostics`: json_repair.py's `repair_json` flow.
///
/// 1. Fence pre-pass (tors-native): if the whole (trimmed) input is one
///    fenced code block, unwrap to its RAW inner text first
///    (see `fence_impl::unwrap_code_fence`) — so the strict fast path runs
///    on the payload and `~~~`/longer/indented fences are handled by the
///    CommonMark grammar instead of by the repair parser's garbage-skip
///    reaching the same place by accident.
/// 2. Unless `skip_json_loads`: strict-parse the text (json.loads parity).
///    Ok: that value IS the answer (no schema) or runs the schema fast path
///    (validate, then repair-in-place, then fall through) exactly per
///    upstream's fast-path block.
/// 3. Repair parser over the text (strict-mode violations and the depth cap
///    raise `Err` with upstream's message catalog); schema-guided parsing
///    when `schema` is set (standard or salvage), then the final validation.
/// 4. The empty-string sentinel: a repaired value of `Value::Str("")` means
///    "nothing recoverable" and is returned AS the empty string (the str-out
///    wrapper renders it bare, matching upstream).
pub fn repair(s: &str, cfg: &RepairConfig) -> Result<(Value, Vec<Diagnostic>), String> {
    if cfg.schema.is_some() && cfg.strict {
        return Err("schema and strict cannot be used together.".into());
    }
    if cfg.salvage && cfg.schema.is_none() {
        return Err("salvage=True requires schema.".into());
    }
    // The text every downstream stage sees: the fence-unwrapped payload when
    // the whole input is one fenced block, the input itself otherwise. A
    // same-line opening fence ("```[1,2]") makes the payload CommonMark's
    // INFO STRING, so the unwrap yields empty content — falling back to
    // the original text lets the repair parser's garbage-skip find the
    // payload exactly like upstream, instead of reporting nothing
    // recoverable.
    let text: &str = match fence_impl::unwrap_code_fence(s) {
        Some(inner) if !inner.trim().is_empty() => inner,
        _ => s,
    };
    // upstream's schema_from_input has already run at the Python boundary
    // (the py layer admits dict/bool schemas only), so the schema Value
    // passes straight through. The repairer owns its own copy (schemas are
    // small — the same clone-per-call tradeoff parser_schema makes);
    // repair() keeps `schema_value` as the "root" reference for
    // is_valid/repair_value/validate, the role upstream's shared schema_obj
    // dict plays.
    let schema_value = cfg.schema.clone();
    let mut repairer = schema_value
        .clone()
        .map(|root| SchemaRepairer::new(root, cfg.salvage, cfg.diagnostics, cfg.locale));

    // json_repair.py's fast-path block (lines 163-215): unless
    // skip_json_loads, run the strict parser over the whole text first. No
    // schema: valid JSON is the answer as-is. With a schema, upstream
    // deliberately refuses to shortcut valid-but-schema-noncompliant JSON:
    // validate, then try an in-place repair_value, and only fall through
    // to the repair parser (which replaces the value entirely) when
    // neither validates.
    if !cfg.skip_json_loads
        && let Ok(loaded) = loads_strict(text)
    {
        match &mut repairer {
            // No schema: nothing to validate, nothing to record (v1's
            // diagnostics are schema-layer actions only).
            None => return Ok((loaded, Vec::new())),
            Some(rep) => {
                let root = schema_value.as_ref().expect("a repairer implies a schema");
                // The date pre-pass runs before the validity shortcut: the
                // format keyword is never asserted, so date strings would
                // otherwise shortcut past normalization as "already valid".
                // It is a no-op for schemas without date formats (one cheap
                // flag check) and for values with nothing to normalize.
                let mut loaded = loaded;
                if rep.has_formats() {
                    rep.prenormalize_dates(&mut loaded, root, "$", 0);
                }
                // The mechanical key pass: fold-matching slop keys rename
                // onto their properties even when the input is otherwise
                // valid and the schema is permissive (the un-renamed shape
                // strands real data on a dead key). Deterministic tier
                // only; validity cannot drop, so no re-validation.
                rep.normalize_keys(&mut loaded, root, "$", 0);
                if rep.is_valid(&loaded, root) {
                    // Valid input reports only hints (never rewrites):
                    // the suggest scan covers exactly this shape, and only
                    // runs when recording (zero cost otherwise).
                    if cfg.diagnostics {
                        rep.suggest_scan(&loaded, root, "$", 0);
                    }
                    return Ok((loaded, rep.take_diagnostics()));
                }
                // repair_value may rework containers in place; a ValueError
                // out of it — including the schema-depth one, which
                // upstream's fast path likewise swallows — or a
                // still-invalid result falls through to the repair parser.
                if let Ok(repaired) = rep.repair_value(loaded, root, "$")
                    && rep.is_valid(&repaired, root)
                {
                    return Ok((repaired, rep.take_diagnostics()));
                }
            }
        }
    }

    // The repair parser (json_parser.py's parse()/parse_with_schema()).
    // With a schema the repairer MOVES into the parser — diagnostics
    // accumulated during the failed fast path survive, exactly like
    // upstream passing the same SchemaRepairer object along. Strict-mode
    // ValueErrors and both depth caps propagate as Err(msg) unchanged
    // (upstream translates its RecursionErrors here; tors' parser and
    // repairer raise the normalized messages themselves).
    let mut parser = parser::Parser::new(text, cfg.strict, repairer);
    if let Some(ms) = cfg.deadline_ms {
        parser.set_deadline(std::time::Instant::now(), ms);
    }
    let (value, diagnostics) = match schema_value.as_ref() {
        Some(_) => {
            let value = parser.parse_with_schema()?;
            // Upstream validates the parsed value against the root schema
            // AFTER parse_with_schema returns, with the ValueError
            // propagating (the salvage loop's per-fragment validate
            // already ran inside). validate() resolves first and rides the
            // precompiled root validator via pointer identity.
            let rep = parser
                .take_repairer()
                .expect("the schema path always carries the repairer");
            rep.validate(
                &value,
                schema_value.as_ref().expect("schema path implies a schema"),
            )?;
            (value, rep.take_diagnostics())
        }
        // No schema: no diagnostics in v1 (schema-layer actions only).
        None => (parser.parse()?, Vec::new()),
    };
    // A Value::Str("") outcome is the "nothing recoverable" sentinel,
    // returned as-is — rendering it bare is the str-out wrapper's job.
    Ok((value, diagnostics))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truthiness_is_pythons() {
        assert!(!Value::Null.is_truthy());
        assert!(!Value::Bool(false).is_truthy());
        assert!(Value::Bool(true).is_truthy());
        assert!(!Value::Int(0).is_truthy());
        assert!(Value::Int(-1).is_truthy());
        assert!(!Value::BigInt("0".into()).is_truthy());
        assert!(Value::BigInt("-7".into()).is_truthy());
        assert!(!Value::Float(0.0).is_truthy());
        assert!(!Value::Float(-0.0).is_truthy());
        // CPython: bool(float("nan")) is True.
        assert!(Value::Float(f64::NAN).is_truthy());
        assert!(!Value::Str(String::new()).is_truthy());
        assert!(!Value::Array(vec![]).is_truthy());
        assert!(!Value::Object(vec![]).is_truthy());
    }

    #[test]
    fn py_eq_is_cross_numeric_and_order_insensitive() {
        assert!(Value::Int(1).py_eq(&Value::Float(1.0)));
        assert!(Value::Bool(true).py_eq(&Value::Int(1)));
        assert!(Value::Bool(false).py_eq(&Value::Int(0)));
        assert!(Value::Bool(true).py_eq(&Value::Float(1.0)));
        assert!(!Value::Bool(true).py_eq(&Value::Float(1.5)));
        // Huge-int exactness: BigInt vs Int goes through decimal text, so
        // 2**63 (one past i64's range) still compares exactly.
        let big = Value::BigInt("9223372036854775808".into());
        assert!(big.py_eq(&Value::BigInt("9223372036854775808".into())));
        assert!(!big.py_eq(&Value::BigInt("9223372036854775809".into())));
        assert!(!big.py_eq(&Value::Int(9))); // "9223372036854775808" != 9
        assert!(Value::BigInt("42".into()).py_eq(&Value::Int(42)));
        // NaN is never equal, including to itself.
        assert!(!Value::Float(f64::NAN).py_eq(&Value::Float(f64::NAN)));
        // Dict order insensitivity; list order sensitivity.
        let a = Value::Object(vec![
            ("a".into(), Value::Int(1)),
            ("b".into(), Value::Int(2)),
        ]);
        let b = Value::Object(vec![
            ("b".into(), Value::Int(2)),
            ("a".into(), Value::Int(1)),
        ]);
        assert!(a.py_eq(&b));
        assert!(
            !Value::Array(vec![Value::Int(1), Value::Int(2)])
                .py_eq(&Value::Array(vec![Value::Int(2), Value::Int(1)]))
        );
        // Type mismatches.
        assert!(!Value::Str("1".into()).py_eq(&Value::Int(1)));
        assert!(!Value::Null.py_eq(&Value::Bool(false)));
    }

    #[test]
    fn is_same_object_is_structural_with_exact_types() {
        // Bool is never int here: Python's `type(True) is type(1)` is False.
        assert!(!Value::Bool(true).is_same_object(&Value::Int(1)));
        assert!(Value::Int(1).is_same_object(&Value::Int(2)));
        assert!(Value::Int(1).is_same_object(&Value::BigInt("9".into())));
        assert!(!Value::Int(1).is_same_object(&Value::Float(1.0)));
        // Same keys, scalar values of the SAME type (structural match ignores
        // values; a type mismatch — Int vs Str — is NOT the same object,
        // upstream's type check fails first).
        let a = Value::Object(vec![("k".into(), Value::Int(1))]);
        let b = Value::Object(vec![("k".into(), Value::Int(2))]);
        assert!(a.is_same_object(&b));
        let c = Value::Object(vec![("k".into(), Value::Str("v".into()))]);
        assert!(!a.is_same_object(&c));
        // Different key sets: not the same object.
        let c = Value::Object(vec![("k2".into(), Value::Int(1))]);
        assert!(!a.is_same_object(&c));
        assert!(
            !Value::Array(vec![Value::Int(1)])
                .is_same_object(&Value::Array(vec![Value::Int(1), Value::Int(2)]))
        );
    }

    #[test]
    fn object_insert_updates_in_place() {
        let mut v = Value::Object(vec![
            ("a".into(), Value::Int(1)),
            ("b".into(), Value::Int(2)),
        ]);
        v.object_insert("a".into(), Value::Int(9));
        assert_eq!(
            v,
            Value::Object(vec![
                ("a".into(), Value::Int(9)),
                ("b".into(), Value::Int(2)),
            ])
        );
        v.object_insert("c".into(), Value::Int(3));
        assert_eq!(
            v,
            Value::Object(vec![
                ("a".into(), Value::Int(9)),
                ("b".into(), Value::Int(2)),
                ("c".into(), Value::Int(3)),
            ])
        );
        assert_eq!(v.object_get("b"), Some(&Value::Int(2)));
        assert_eq!(v.object_get("zz"), None);
    }

    #[test]
    fn type_names_are_cpythons() {
        assert_eq!(Value::Object(vec![]).type_name(), "dict");
        assert_eq!(Value::Array(vec![]).type_name(), "list");
        assert_eq!(Value::Str(String::new()).type_name(), "str");
        assert_eq!(Value::Int(0).type_name(), "int");
        assert_eq!(Value::BigInt("1".into()).type_name(), "int");
        assert_eq!(Value::Float(0.0).type_name(), "float");
        assert_eq!(Value::Bool(false).type_name(), "bool");
        assert_eq!(Value::Null.type_name(), "NoneType");
    }

    // repair() smoke tests. The guard cases and the strict fast path are
    // self-contained; the parser-dependent cases (garbage fallback,
    // skip_json_loads) pin the same shapes the FULL corpus asserts in
    // tests/test_json_repair.py (the ported upstream suite plus the
    // differential oracle), so they stay minimal here.

    #[test]
    fn repair_valid_json_fast_path() {
        let (v, diagnostics) = repair("{\"key\": \"value\"}", &RepairConfig::default()).unwrap();
        assert_eq!(
            v,
            Value::Object(vec![("key".into(), Value::Str("value".into()))])
        );
        // v1: no schema, no diagnostics.
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn repair_garbage_yields_the_empty_string_sentinel() {
        // Nothing JSON-shaped anywhere: the repair parser's "" outcome is
        // the sentinel repair() returns as-is (the str-out wrapper renders
        // it bare, matching upstream's repair_json).
        let (v, _) = repair(
            "just some prose, nothing JSON here",
            &RepairConfig::default(),
        )
        .unwrap();
        assert_eq!(v, Value::Str(String::new()));
    }

    #[test]
    fn repair_skip_json_loads_still_parses_valid_json() {
        // skip_json_loads skips only the whole-input fast path — the
        // repair parser still parses the document (and tolerates the
        // surrounding whitespace upstream's parser skips).
        let cfg = RepairConfig {
            skip_json_loads: true,
            ..RepairConfig::default()
        };
        let (v, _) = repair("  {\"key\": \"value\"}  ", &cfg).unwrap();
        assert_eq!(
            v,
            Value::Object(vec![("key".into(), Value::Str("value".into()))])
        );
    }

    #[test]
    fn repair_rejects_schema_with_strict() {
        let cfg = RepairConfig {
            schema: Some(Value::Object(vec![(
                "type".into(),
                Value::Str("object".into()),
            )])),
            strict: true,
            ..RepairConfig::default()
        };
        assert_eq!(
            repair("{}", &cfg).unwrap_err(),
            "schema and strict cannot be used together."
        );
    }

    #[test]
    fn repair_rejects_salvage_without_schema() {
        let cfg = RepairConfig {
            salvage: true,
            ..RepairConfig::default()
        };
        assert_eq!(
            repair("{}", &cfg).unwrap_err(),
            "salvage=True requires schema."
        );
    }
}
