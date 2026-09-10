//! Schema-guided repair and alignment: a port of json_repair's
//! `schema_repair.py` + `parser_schema.py` + `utils/pattern_properties.py`
//! with `json_repair`'s standard and salvage modes (upstream:
//! https://github.com/mangiucugna/json_repair by Stefano Baccianella, MIT,
//! commit 251d141786d0f6ff561f6ec04d90188a338e2470), plus tors-native
//! extensions and a different choice of validation engine.
//!
//! # What is ported, what is native
//!
//! The repair mechanics are upstream's: missing-value fills (`const`
//! \> `enum[0]` > `default` > type-derived), scalar coercion
//! (`"1"` -> `1`, `"yes"` -> `true`, `1` -> `"1"`), object/array alignment
//! (`required`, `properties`, `patternProperties`, `additionalProperties`,
//! `items` (single and tuple form), `additionalItems`, `minItems`,
//! `minProperties`), union semantics (`allOf` sequential, `oneOf`/`anyOf`
//! first-branch-that-validates, `type` lists), double-serialized-string
//! unwrapping, `$ref` resolution (`#/...` pointers only, `~0`/`~1`,
//! cycle detection), and every salvage behavior (fragment selection,
//! invalid-item dropping, list-to-object mapping, set-like objects, the
//! root single-item unwrap).
//!
//! What diverges, deliberately:
//!
//! - **Validation engine**: upstream delegates to the Python `jsonschema`
//!   package; tors uses the Rust `jsonschema` crate (drafts 4-2020-12,
//!   auto-detected like upstream's `validator_for`). Each call compiles one
//!   root validator plus fresh allOf-wrapped branch validators (union
//!   branches keep `#/...` ref scope: the Python side's `evolve`
//!   equivalent); the prepared-for-validation schema normalizes draft-07
//!   `items`-lists to `prefixItems` exactly like upstream's normalization.
//!   Wording of validation failures follows the Rust crate (not the Python
//!   package's); instances containing non-finite numbers are rejected under
//!   schema mode (the Python side tolerates `NaN`, but JSON and serde_json
//!   cannot represent it); `BigInt` beyond `u64` validates through `f64`
//!   (lossy at the boundary only); `format` is never asserted (upstream
//!   passes no `format_checker` either: parity).
//! - **tors-native repairs** (all recorded as diagnostics, all opt-out by
//!   simply not using schema mode): the key-normalization ladder
//!   (case/`snake`/`kebab`/`spaces` folding to an exact property) ahead of
//!   the fuzzy remap; the fuzzy remap itself; comma-separated-string to
//!   array recovery that wins only when it validates and upstream's wrap
//!   does not; digit-group-separator tolerance in numeric coercion; enum
//!   "did you mean" suggestions; RFC 3339 date normalization behind
//!   `format: date`/`date-time`; enriched validation errors (message plus
//!   instance path).
//! - **Pydantic** models are accepted by the py layer (it calls
//!   `model_json_schema()`), not by this module: by the time a schema
//!   arrives here it is a plain JSON `Value`.
//!
//! # Ownership shape (a deliberate idiom note)
//!
//! The repairer holds the root schema, the salvage flag, a
//! `RefCell`-guarded diagnostics recorder, and the compiled root validator.
//! All repair methods take `&self`: the `&mut`-through-`RefCell` recorder
//! keeps the borrow checker satisfied without cloning subschemas out of the
//! tree: a line-by-line port would clone the schema at every `repair_value`
//! call (`&mut` cannot coexist with the `&self.root` borrows), which would
//! cost O(schema x nodes) memory on every repair. The recorder is a
//! `RefCell` (not thread-shared: each repair constructs its own repairer
//! inside one call), and no record borrow is ever held across a repair
//! call, so it cannot re-enter.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use regex::Regex;
use serde_json::Map;

use crate::fuzzy_impl::jaro_winkler;
use crate::json_repair::{
    Diagnostic, NumericLocale, RepairConfig, Value, exact_decimal_of_float, loads_strict,
    normalize_big_int_text, py_float_repr, repair,
};

/// The schema-nesting cap, matching the parser side's `MAX_NESTING` shape:
/// a straight line from `repair_value` through `$ref` chains, `allOf`
/// folds and the validation-preparation walks. The 550-deep `allOf` and
/// 550-deep `properties` corpus payloads raise exactly this message.
const MAX_SCHEMA_DEPTH: usize = 200;

/// jaro-winkler score for the fuzzy key ladder (unique-best, target absent,
/// loss-or-failure gating: see the repair_object docs), and the lower
/// report-only bar for both keys and enum members.
const TYPO_REMAP_MIN: f64 = 0.75;
const TYPO_REMAP_MARGIN: f64 = 0.05;
const SUGGEST_MIN: f64 = 0.60;

// ============================================================
// Locale-aware numeric coercion: CLDR separator data, two compiled
// grammars, and the single-number extraction tier. Design invariants:
// the value's separators resolve by which grammar arm matched (never
// post-hoc text guessing); a known locale's own separators are
// normalized onto '.'/',' before parsing (data, not guessing); the
// Auto default assumes en-US for the separator-ambiguous shapes, but
// only schema-checked and disclosed (a reading the type or schema
// rejects is not a candidate, and `locale=` always overrides), while
// shapes no locale could resolve refuse with an actionable error: a
// mis-fixed value silently corrupts by 1000x, so nothing is guessed
// silently.
// ============================================================

/// A locale's grouping separator: a literal char, or the whitespace
/// class CLDR uses for NBSP/narrow-NBSP/space groupings (fr, ru, sv...).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GroupSeparator {
    Char(char),
    Space,
}

/// One resolved locale's number-separator convention (CLDR's decimal
/// mark and grouping separator for the tag, or a caller's explicit dict),
/// plus the locale's month-name table for date normalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocaleSpec {
    pub(crate) decimal: char,
    pub(crate) grouping: GroupSeparator,
}

impl LocaleSpec {
    /// The default convention (dot decimal, comma groups, English month
    /// names): the JSON spelling every locale shares.
    const DOT_COMMA: LocaleSpec = LocaleSpec {
        decimal: '.',
        grouping: GroupSeparator::Char(','),
    };

    /// A caller-supplied convention: one decimal char and one grouping
    /// char (a whitespace grouping char selects the whitespace class).
    pub fn custom(decimal: char, grouping: char) -> LocaleSpec {
        let grouping = if grouping.is_whitespace() {
            GroupSeparator::Space
        } else {
            GroupSeparator::Char(grouping)
        };
        LocaleSpec { decimal, grouping }
    }
}

/// BCP 47 tag -> CLDR separators: the language-default table with region
/// overrides, case-insensitive, `-` or `_` separated, language-only tags
/// taking the default region. Well-formed-but-unlisted tags resolve to
/// the dot/comma convention; Lakh-style grouping (en-IN, en-PK) is
/// unsupported and refused loudly rather than mis-read. The table is the
/// CLDR number-symbols data for the locales LLMs actually emit numbers
/// in: standard data, not a two-way invention.
pub fn locale_spec_from_tag(tag: &str) -> Option<LocaleSpec> {
    let tag = tag.trim().to_lowercase().replace('_', "-");
    let mut parts = tag.split('-');
    let language = parts.next()?;
    if language.is_empty() || !language.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let region = parts.next().unwrap_or_default();
    if !region.is_empty() && !region.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    match (language, region) {
        ("de", "ch") => Some(LocaleSpec {
            decimal: '.',
            grouping: GroupSeparator::Char('\u{2019}'),
        }),
        ("es", "419") | ("es", "mx") | ("es", "us") => Some(LocaleSpec::DOT_COMMA),
        // Lakh grouping is a different grouping shape, not a different
        // separator: unsupported, and the error says so.
        ("en", "in") | ("en", "pk") => None,
        _ => match language {
            "de" | "es" | "it" | "pt" | "nl" | "ca" | "el" | "sl" | "hr" | "sr" => {
                Some(LocaleSpec {
                    decimal: ',',
                    grouping: GroupSeparator::Char('.'),
                })
            }
            "fr" | "ru" | "uk" | "pl" | "cs" | "sk" | "da" | "fi" | "sv" | "ro" | "hu" | "vi"
            | "lt" | "lv" | "et" | "bg" | "no" | "nb" | "nn" => Some(LocaleSpec {
                decimal: ',',
                grouping: GroupSeparator::Space,
            }),
            // Arabic-script digits and separators, handled by the shared
            // normalization below.
            "ar" | "fa" | "ur" => Some(LocaleSpec {
                decimal: '\u{066B}',
                grouping: GroupSeparator::Char('\u{066C}'),
            }),
            _ => Some(LocaleSpec::DOT_COMMA),
        },
    }
}

/// The shared script normalization: CJK fullwidth forms and Arabic-Indic
/// digits onto ASCII, digit-group underscores dropped. An explicit table
/// of the forms that are unambiguous digits and punctuation: not an
/// NFKC pass, which would also fold superscripts ("5²" -> "52") and
/// corrupt values.
fn normalize_script(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '０'..='９' => char::from(b'0' + (c as u32 - '０' as u32) as u8),
            '٠'..='٩' => char::from(b'0' + (c as u32 - '٠' as u32) as u8),
            '۰'..='۹' => char::from(b'0' + (c as u32 - '۰' as u32) as u8),
            '＋' => '+',
            '－' => '-',
            '．' => '.',
            '，' => ',',
            '％' => '%',
            '_' => '\0',
            other => other,
        })
        .filter(|c| *c != '\0')
        .collect()
}

/// Locale-aware normalization: the script pass, then the locale's own
/// decimal mark onto '.' and its grouping separator onto ','. After this,
/// every locale's numbers are in the one canonical shape the grammars
/// below parse: German "1,234" becomes "1.234" (the comma was the
/// decimal), French "1 234,56" becomes "1,234.56".
fn normalize_locale(text: &str, spec: &LocaleSpec) -> String {
    normalize_script(text)
        .chars()
        .map(|c| {
            if c == spec.decimal {
                '.'
            } else if c == ','
                || match spec.grouping {
                    GroupSeparator::Char(group) => c == group,
                    GroupSeparator::Space => c.is_whitespace(),
                }
            {
                ','
            } else {
                c
            }
        })
        .collect()
}

/// The number-token grammars, compiled once into linear-time automata
/// (the `regex` crate: no backtracking class exists for these). Structural
/// arms run in every mode because structure beats locale: repeated comma
/// groups with an optional dot fraction (`1,234,567.89`: repetition
/// proves grouping, a decimal mark appears exactly once), one comma group
/// plus a dot fraction (`1,234.56`: the dot proves the commas group),
/// and the European mirror (`1.234,56`: dot groups, comma decimal).
/// The known grammar adds the single comma group: with a locale in hand
/// the ambiguity is already resolved (its decimal mark was normalized),
/// so `1,234` is one token. The plain arm runs last in both so ambiguous
/// shapes fall through it and split into two tokens: the Auto refusal
/// that keeps a wrong 1000x reading impossible.
static NUMBER_TOKEN_AUTO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"[-+]?(?P<us_multi>\d{1,3}(?:,\d{3}){2,}(?:\.\d+)?)",
        r"|(?P<us_group_frac>\d{1,3},\d{3}\.\d+)",
        r"|(?P<eu_struct>\d{1,3}(?:\.\d{3})+,\d+)",
        r"|(?P<plain>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    ))
    .expect("the number-token grammar is a compile-time constant")
});
static NUMBER_TOKEN_KNOWN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"[-+]?(?P<us_multi>\d{1,3}(?:,\d{3}){2,}(?:\.\d+)?)",
        r"|(?P<us_group_frac>\d{1,3},\d{3}\.\d+)",
        r"|(?P<eu_struct>\d{1,3}(?:\.\d{3})+,\d+)",
        r"|(?P<us_single>\d{1,3},\d{3})",
        r"|(?P<plain>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    ))
    .expect("the number-token grammar is a compile-time constant")
});

/// One extracted number: the token with its separators resolved by the
/// arm that matched (the European arm drops dot groups and turns the
/// comma into the decimal point; every other arm drops comma groups),
/// and whether a `%` qualifies it: percent values coerce as fractions
/// ("50%" is 0.5, "200%" is 2).
struct ExtractedNumber {
    token: String,
    percent: bool,
}

/// The single-number extraction tier: when a number-typed field holds a
/// string that is not a number even modulo formatting, look for exactly
/// one number token in it: the model's answer with prose or symbols
/// around it ("$1,234.56", "USD 50", "about 50 dollars"). Zero tokens or
/// more than one is ambiguous data ("between 10 and 20") and
/// returns None: the value stays a string for validation to judge,
/// never a guess. A lone `-` between prose and the token (after currency
/// symbols and whitespace) negates it ("-$50.0" -> -50), the one sign
/// position the grammar itself cannot see.
fn extract_single_number(text: &str, locale: NumericLocale) -> Option<ExtractedNumber> {
    let (normalized, grammar) = match locale {
        NumericLocale::Known(spec) => (normalize_locale(text, &spec), &*NUMBER_TOKEN_KNOWN),
        NumericLocale::Auto => (normalize_script(text), &*NUMBER_TOKEN_AUTO),
    };
    let mut captures = grammar.captures_iter(&normalized);
    let first = captures.next()?;
    if captures.next().is_some() {
        return None;
    }
    let whole = first.get(0).expect("group 0 is the whole match");
    // A separator immediately before the match means the grammar dropped
    // a decimal marker (".5" -> "5"): the token is not the number the
    // text carried (".5" must never coerce to 5 on an integer field)
    // so no single number exists. (Grouping arms match their separators
    // inside the token, so this only fires on the dropped-marker shape.)
    if normalized[..whole.start()].ends_with(['.', ',']) {
        return None;
    }
    let percent = normalized[whole.end()..].trim_start().starts_with('%');
    let mut token: String = if first.name("eu_struct").is_some() {
        whole
            .as_str()
            .chars()
            .filter(|c| *c != '.')
            .map(|c| if c == ',' { '.' } else { c })
            .collect()
    } else {
        whole.as_str().chars().filter(|c| *c != ',').collect()
    };
    let prefix: String = normalized[..whole.start()]
        .chars()
        .filter(|c| *c == '-' || c.is_whitespace())
        .collect();
    if prefix.trim() == "-" {
        token.insert(0, '-');
    }
    Some(ExtractedNumber { token, percent })
}

/// The European mirror convention (comma decimal, dot groups): the
/// second candidate in Auto mode's schema-checked ambiguity resolution.
const EU_MIRROR: LocaleSpec = LocaleSpec {
    decimal: ',',
    grouping: GroupSeparator::Char('.'),
};

/// The regex metacharacters upstream's patternProperties subset does not
/// support: any pattern whose literal remainder uses one is skipped and
/// reported through the unsupported list.
const UNSUPPORTED_REGEX_TOKENS: [char; 14] = [
    '.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '|', '(', ')', '\\',
];

/// Resolved schema, the Rust shape of upstream's `dict | bool` after `$ref`
/// resolution. There is no `False` variant on purpose: a `false` schema
/// raises `"Schema does not allow any values."` at every call site (the
/// parser's `_resolve_schema_for_parse` behavior), produced inside
/// [`SchemaRepairer::resolve_schema`].
pub(crate) enum ResolvedSchema<'a> {
    True,
    Schema(&'a Value),
}

/// The declared `type` keyword in its two spellings: one name, or a list
/// of names (the union branch repair synthesizes one single-type schema
/// per member).
enum ExpectedType {
    One(String),
    Many(Vec<String>),
}

/// The fate of one array item under schema repair: kept (the repaired
/// value), dropped (salvage only, recorded), or failed (standard mode,
/// the error propagates: upstream raises here).
enum ItemFate {
    Keep(Value),
    Drop,
    Fail(String),
}

/// The tors-native key ladder's verdict for one unknown key: remap onto a
/// schema property (both tiers), a report-only suggestion, or nothing.
enum Ladder {
    Remap(String),
    Suggest(String),
    Nothing,
}

/// parser_schema.py's object config: the per-parse guidance for one
/// object, with property schemas owned (one compact clone per container
/// parse keeps the repairer borrows out of the parser's `&mut`).
#[derive(Clone, Debug, Default)]
pub(crate) struct ObjectSchemaConfig {
    pub properties: Vec<(String, Value)>,
    pub pattern_properties: Option<Value>,
    pub additional_properties: Option<Value>,
    pub required: Vec<String>,
}

impl ObjectSchemaConfig {
    fn property(&self, key: &str) -> Option<&Value> {
        self.properties
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }
}

/// parser_schema.py's array config: `items_list` is the draft-07 tuple
/// form, `items_schema` the single-schema form, `additional_items` the
/// tuple tail rule.
#[derive(Clone, Debug, Default)]
pub(crate) struct ArraySchemaConfig {
    pub items_schema: Option<Value>,
    pub items_list: Option<Vec<Value>>,
    pub additional_items: Option<Value>,
}

/// The internal `$ref`-chain walker step: the public
/// [`SchemaRepairer::resolve_schema`] maps `False` to the catalog error
/// and shapes the other two, while the internal object/array-shape probes
/// treat `False` (and `True`) as "not that shape" without raising.
enum Chain<'a> {
    True,
    False,
    Schema(&'a Value),
}

/// The schema layer: upstream's `SchemaRepairer`, owned by the repair call
/// (constructed by `repair()`, moved into the parser for schema-guided
/// parsing, taken back for the final validation).
pub(crate) struct SchemaRepairer {
    root: Value,
    salvage: bool,
    /// The caller's numeric-locale knowledge (never guessed): resolves the
    /// locale-ambiguous number formats during string coercion.
    locale: NumericLocale,
    recorded: RefCell<Option<Vec<Diagnostic>>>,
    root_validator: Result<jsonschema::Validator, String>,
    /// Compiled subschema validators, keyed by the resolved schema's
    /// address (upstream caches by `id(schema)` with an identity recheck;
    /// pointer identity is the Rust spelling, stable because every
    /// subschema reference points into this call's own schema tree).
    /// Union repair (per branch, per value) and the tier-4 ambiguity
    /// probe otherwise recompile the same wrapper per call.
    sub_validators: RefCell<HashMap<usize, Arc<jsonschema::Validator>>>,
    /// Whether the schema tree declares any normalizable `format`
    /// (date/date-time/time/uuid, computed once): only then does the fast
    /// path pay for the format pre-pass (formats are never asserted by
    /// validation, so these strings would otherwise shortcut past the
    /// normalization hooks as "already valid").
    has_formats: bool,
}

impl SchemaRepairer {
    /// Construct the repairer: the root schema is owned (the caller keeps
    /// its own handle for the fast-path calls), diagnostics are recorded
    /// only when asked, and the draft-normalized root validator compiles
    /// once up front (compile failures surface as validation errors with
    /// the compile message: the upstream "evolve lazily, surface late"
    /// shape, without the lazy machinery).
    pub(crate) fn new(
        root: Value,
        salvage: bool,
        diagnostics: bool,
        locale: NumericLocale,
    ) -> Self {
        // Upstream's deep-schema failures surface from validation recursion
        // (a 550-deep schema raises RecursionError inside is_valid/validate,
        // never inside repair): gate the schema's own nesting here so the
        // validator carries the normalized error from the start.
        let root_validator = if schema_nesting(&root, 0) > MAX_SCHEMA_DEPTH {
            Err("Input schema nesting exceeds the supported schema recursion depth.".into())
        } else {
            match prepare_for_validation(&root, 0) {
                Ok(schema) => jsonschema::validator_for(&schema).map_err(|err| format!("{err}")),
                Err(message) => Err(message),
            }
        };
        let has_formats = schema_has_formats(&root, 0);
        SchemaRepairer {
            root,
            salvage,
            locale,
            recorded: RefCell::new(diagnostics.then(Vec::new)),
            root_validator,
            sub_validators: RefCell::new(HashMap::new()),
            has_formats,
        }
    }

    pub(crate) fn root(&self) -> &Value {
        &self.root
    }

    pub(crate) fn is_salvage(&self) -> bool {
        self.salvage
    }

    pub(crate) fn has_formats(&self) -> bool {
        self.has_formats
    }

    /// Whether this repair records diagnostics (the diagnostics spelling).
    /// The suggest scan consults it to stay zero-cost otherwise.
    fn recording(&self) -> bool {
        self.recorded.borrow().is_some()
    }

    /// Date pre-pass for the strict fast path: walk the parsed value
    /// alongside the schema and normalize date strings in place. The
    /// format keyword is never asserted by validation, so without this
    /// pass every date string would shortcut past the normalization hook
    /// as "already valid". Only string+format nodes move; everything else
    /// passes through untouched (and unrecorded when recording is off).
    pub(crate) fn prenormalize_dates(
        &self,
        value: &mut Value,
        schema: &Value,
        path: &str,
        depth: usize,
    ) {
        if depth > MAX_SCHEMA_DEPTH {
            return;
        }
        let resolved = match self.resolve_chain(schema) {
            Ok(Chain::Schema(resolved)) => resolved,
            _ => return,
        };
        // See suggest_scan: allOf is conjunctive guidance, reached like
        // the parent's own; oneOf/anyOf stay fast-path-unreached.
        for member in all_of_members(resolved) {
            self.prenormalize_dates(value, member, path, depth + 1);
        }
        if let Value::Str(text) = value {
            if let Some(format) = schema_format(resolved) {
                if format == "uuid" {
                    if let Some(lower) = normalize_uuid(text) {
                        self.record(
                            "coerce",
                            path,
                            "Normalized uuid to lowercase",
                            Some(Value::Str(text.clone())),
                            Some(Value::Str(lower.clone())),
                            None,
                        );
                        *value = Value::Str(lower);
                    }
                    return;
                }
                if let Some(date_format) = date_format_of(format) {
                    match normalize_date(text, date_format) {
                        DateOutcome::Normalized(normalized) if normalized != *text => {
                            self.record(
                                "format_date",
                                path,
                                format!("Normalized {format} to canonical form"),
                                Some(Value::Str(text.clone())),
                                Some(Value::Str(normalized.clone())),
                                None,
                            );
                            *value = Value::Str(normalized);
                        }
                        DateOutcome::Ambiguous => {
                            self.record(
                                "suggest",
                                path,
                                "Ambiguous date: month/day order unclear, left unchanged",
                                None,
                                None,
                                Some("use an unambiguous form like 2024-04-13".into()),
                            );
                        }
                        DateOutcome::Normalized(_) | DateOutcome::Invalid => {}
                    }
                }
            }
            return;
        }
        match value {
            Value::Object(entries) => {
                let config = object_schema_config(resolved);
                // Properties first (present keys only: absent keys have no
                // value to normalize).
                for (key, prop) in &config.properties {
                    if let Some(slot) = entries.iter_mut().find(|(k, _)| k == key) {
                        let key_path = format!("{path}.{key}");
                        self.prenormalize_dates(&mut slot.1, prop, &key_path, depth + 1);
                    }
                }
                // Extras through their pattern/additional schemas.
                for slot in entries.iter_mut() {
                    if config.property(&slot.0).is_some() {
                        continue;
                    }
                    let key_path = format!("{path}.{}", slot.0);
                    let (matched, _) = match config.pattern_properties.as_ref() {
                        Some(patterns) => match_pattern_properties(patterns, &slot.0),
                        None => (Vec::new(), Vec::new()),
                    };
                    if let Some(first) = matched.first() {
                        self.prenormalize_dates(&mut slot.1, first, &key_path, depth + 1);
                    } else if let Some(Value::Object(_)) = config.additional_properties.as_ref() {
                        let extra = config.additional_properties.as_ref().unwrap();
                        self.prenormalize_dates(&mut slot.1, extra, &key_path, depth + 1);
                    }
                }
            }
            Value::Array(items) => {
                let config = array_schema_config(resolved);
                match (config.items_list, config.items_schema) {
                    (Some(tuple), _) => {
                        for (idx, item_schema) in tuple.iter().enumerate() {
                            if let Some(item) = items.get_mut(idx) {
                                self.prenormalize_dates(
                                    item,
                                    item_schema,
                                    &format!("{path}[{idx}]"),
                                    depth + 1,
                                );
                            }
                        }
                    }
                    (None, Some(item_schema)) => {
                        for (idx, item) in items.iter_mut().enumerate() {
                            self.prenormalize_dates(
                                item,
                                &item_schema,
                                &format!("{path}[{idx}]"),
                                depth + 1,
                            );
                        }
                    }
                    (None, None) => {}
                }
            }
            _ => {}
        }
    }

    /// Suggest-only scan over a value that already validates: unknown keys
    /// with a confident ladder match are reported as hints, never remapped
    /// (the value is valid as-is, additionalProperties allows them, so
    /// silence would hide probable typos, but rewriting valid input would
    /// break the is_valid contract). The repair path records its own
    /// suggests inline, so this covers exactly the fast-path-valid shape;
    /// it runs only when recording (zero cost otherwise).
    pub(crate) fn suggest_scan(&self, value: &Value, schema: &Value, path: &str, depth: usize) {
        if !self.recording() || depth > MAX_SCHEMA_DEPTH {
            return;
        }
        let resolved = match self.resolve_chain(schema) {
            Ok(Chain::Schema(resolved)) => resolved,
            _ => return,
        };
        // allOf members carry conjunctive guidance that applies exactly
        // like the parent's own (pydantic v2 emits this shape for model
        // inheritance). oneOf/anyOf are deliberately not reached here:
        // which branch applies is value-dependent, and the fast path does
        // not guess (upstream's fast path has the same reach: design §9).
        for member in all_of_members(resolved) {
            self.suggest_scan(value, member, path, depth + 1);
        }
        match value {
            Value::Object(entries) => {
                let config = object_schema_config(resolved);
                for (key, raw) in entries {
                    let key_path = format!("{path}.{key}");
                    if let Some(prop) = config.property(key) {
                        self.suggest_scan(raw, prop, &key_path, depth + 1);
                        continue;
                    }
                    let (matched, _) = match config.pattern_properties.as_ref() {
                        Some(patterns) => match_pattern_properties(patterns, key),
                        None => (Vec::new(), Vec::new()),
                    };
                    if let Some(first) = matched.first() {
                        self.suggest_scan(raw, first, &key_path, depth + 1);
                        continue;
                    }
                    match self.key_ladder(key, raw, value, &config, path, depth) {
                        Ladder::Remap(target) | Ladder::Suggest(target) => {
                            self.record(
                                "suggest",
                                &key_path,
                                format!("Unknown key '{key}'"),
                                None,
                                None,
                                Some(format!("did you mean '{target}'?")),
                            );
                        }
                        Ladder::Nothing => {}
                    }
                }
            }
            Value::Array(items) => {
                let config = array_schema_config(resolved);
                match (config.items_list, config.items_schema) {
                    (Some(tuple), _) => {
                        for (idx, item_schema) in tuple.iter().enumerate() {
                            if let Some(item) = items.get(idx) {
                                self.suggest_scan(
                                    item,
                                    item_schema,
                                    &format!("{path}[{idx}]"),
                                    depth + 1,
                                );
                            }
                        }
                    }
                    (None, Some(item_schema)) => {
                        for (idx, item) in items.iter().enumerate() {
                            self.suggest_scan(
                                item,
                                &item_schema,
                                &format!("{path}[{idx}]"),
                                depth + 1,
                            );
                        }
                    }
                    (None, None) => {}
                }
            }
            _ => {}
        }
    }

    /// Record one action into the diagnostics log (a no-op when the repair
    /// was constructed with recording off).
    pub(crate) fn record(
        &self,
        action: &str,
        path: &str,
        detail: impl Into<String>,
        from: Option<Value>,
        to: Option<Value>,
        suggestion: Option<String>,
    ) {
        if let Some(log) = self.recorded.borrow_mut().as_mut() {
            log.push(Diagnostic {
                action: action.into(),
                path: path.into(),
                detail: detail.into(),
                from,
                to,
                suggestion,
            });
        }
    }

    /// Drain the recorded diagnostics for `repair()`'s return (empty when
    /// recording was off). Takes `&self` like every other method: the
    /// recorder is interior-mutable by design.
    pub(crate) fn take_diagnostics(&self) -> Vec<Diagnostic> {
        self.recorded.borrow_mut().take().unwrap_or_default()
    }

    /// Run `f` with the diagnostics log rolled back afterwards:
    /// speculative repairs (fold-tier compatibility probes, union branches
    /// that may fail validation) must not leave records of actions whose
    /// result was discarded.
    fn probe<T>(&self, f: impl FnOnce(&Self) -> T) -> T {
        let mark = self.recorded.borrow().as_ref().map_or(0, Vec::len);
        let out = f(self);
        if let Some(log) = self.recorded.borrow_mut().as_mut() {
            log.truncate(mark);
        }
        out
    }

    /// The `$ref` chain walker: `"#/..."` JSON pointers only (`~0`/`~1`
    /// unescaping in upstream's order), ref-string cycle detection, boolean
    /// targets terminating the chain. Non-dict non-bool schemas, non-string
    /// `$ref`s, foreign ref spellings, and unresolvable pointers all raise
    /// upstream's catalog messages.
    fn resolve_chain<'a>(&'a self, schema: &'a Value) -> Result<Chain<'a>, String> {
        if let Value::Bool(true) = schema {
            return Ok(Chain::True);
        }
        if let Value::Bool(false) = schema {
            return Ok(Chain::False);
        }
        let Value::Object(_) = schema else {
            return Err("Schema must be an object.".into());
        };
        let mut current = schema;
        let mut seen: Vec<&str> = Vec::new();
        loop {
            let Value::Object(entries) = current else {
                return Err("Schema must be an object.".into());
            };
            let Some(reference) = entries.iter().find(|(k, _)| k == "$ref").map(|(_, v)| v) else {
                return Ok(Chain::Schema(current));
            };
            if seen.len() > MAX_SCHEMA_DEPTH {
                return Err(
                    "Input schema nesting exceeds the supported schema recursion depth.".into(),
                );
            }
            let Value::Str(reference) = reference else {
                return Err("$ref must be a string.".into());
            };
            if seen.contains(&reference.as_str()) {
                return Err(format!("Circular $ref detected: {reference}"));
            }
            seen.push(reference);
            let Some(pointer) = reference.strip_prefix("#/") else {
                return Err(format!("Unsupported $ref: {reference}"));
            };
            let mut node = &self.root;
            for raw in pointer.split('/') {
                // Upstream unescapes ~1 before ~0 (the order matters for a
                // literal "~1" key: it must survive as "~1", not become "/").
                let part = raw.replace("~1", "/").replace("~0", "~");
                let Value::Object(members) = node else {
                    return Err(format!("Unresolvable $ref: {reference}"));
                };
                node = match members.iter().find(|(k, _)| *k == part) {
                    Some((_, value)) => value,
                    None => return Err(format!("Unresolvable $ref: {reference}")),
                };
            }
            match node {
                Value::Bool(true) => return Ok(Chain::True),
                Value::Bool(false) => return Ok(Chain::False),
                Value::Object(_) => current = node,
                _ => return Err(format!("Unresolvable $ref: {reference}")),
            }
        }
    }

    /// The parser-facing resolver: `false` raises, `true` passes unguided,
    /// dicts return borrowed (borrowing `schema` when it has no `$ref`,
    /// borrowing `self.root` through the chain otherwise: either way the
    /// lifetime is the caller's).
    pub(crate) fn resolve_schema<'a>(
        &'a self,
        schema: &'a Value,
    ) -> Result<ResolvedSchema<'a>, String> {
        match self.resolve_chain(schema)? {
            Chain::True => Ok(ResolvedSchema::True),
            Chain::False => Err("Schema does not allow any values.".into()),
            Chain::Schema(resolved) => Ok(ResolvedSchema::Schema(resolved)),
        }
    }

    /// Upstream's `is_object_schema`: the `type` keyword (string or list
    /// form) or the object-shape keywords. `true`/`false`/scalars are not
    /// object schemas (never raises: the chain's `False` is a shape miss).
    pub(crate) fn is_object_schema(&self, schema: &Value) -> bool {
        let resolved = match self.resolve_chain(schema) {
            Ok(Chain::Schema(resolved)) => resolved,
            _ => return false,
        };
        let Value::Object(entries) = resolved else {
            return false;
        };
        if schema_type_allows(entries, "object") {
            return true;
        }
        entries.iter().any(|(k, _)| {
            matches!(
                k.as_str(),
                "properties" | "patternProperties" | "additionalProperties" | "required"
            )
        })
    }

    /// Upstream's `is_array_schema`: the `type` keyword or an `items` key.
    pub(crate) fn is_array_schema(&self, schema: &Value) -> bool {
        let resolved = match self.resolve_chain(schema) {
            Ok(Chain::Schema(resolved)) => resolved,
            _ => return false,
        };
        let Value::Object(entries) = resolved else {
            return false;
        };
        if schema_type_allows(entries, "array") {
            return true;
        }
        entries.iter().any(|(k, _)| k == "items")
    }

    /// Upstream's `repair_value`: apply schema rules to a parsed value:
    /// unions, coercions, containers, fills: then the enum/const tail.
    pub(crate) fn repair_value(
        &self,
        value: Value,
        schema: &Value,
        path: &str,
    ) -> Result<Value, String> {
        self.repair_value_d(value, schema, path, 0)
    }

    fn repair_value_d(
        &self,
        value: Value,
        schema: &Value,
        path: &str,
        depth: usize,
    ) -> Result<Value, String> {
        if depth > MAX_SCHEMA_DEPTH {
            return Err(
                "Input schema nesting exceeds the supported schema recursion depth.".into(),
            );
        }
        let resolved = match self.resolve_chain(schema)? {
            Chain::True => return normalize_missing_values(value),
            Chain::False => return Err("Schema does not allow any values.".into()),
            Chain::Schema(resolved) => resolved,
        };
        let Value::Object(entries) = resolved else {
            return normalize_missing_values(value);
        };
        if entries.is_empty() {
            return normalize_missing_values(value);
        }
        if let Value::Missing = value {
            return self.fill_missing(resolved, path, depth);
        }
        if let Some(Value::Array(subs)) = entries.iter().find(|(k, _)| k == "allOf").map(|(_, v)| v)
        {
            if subs.is_empty() {
                return normalize_missing_values(value);
            }
            // Sequential fold over the allOf members (upstream passes the
            // repaired value along, no per-branch copies).
            let mut acc = self.repair_value_d(value, &subs[0], path, depth + 1)?;
            for sub in &subs[1..] {
                acc = self.repair_value_d(acc, sub, path, depth + 1)?;
            }
            return Ok(acc);
        }
        if let Some(subs) = get_array(entries, "oneOf") {
            return self.repair_union(value, subs, path, depth);
        }
        if let Some(subs) = get_array(entries, "anyOf") {
            return self.repair_union(value, subs, path, depth);
        }
        // The expected type: declared, or inferred from the schema's shape
        // when `type` is omitted.
        let declared = entries.iter().find(|(k, _)| k == "type").map(|(_, v)| v);
        let expected: Option<ExpectedType> = match declared {
            Some(Value::Str(name)) => Some(ExpectedType::One(name.clone())),
            Some(Value::Array(names)) => {
                let mut kinds = Vec::with_capacity(names.len());
                for name in names {
                    if let Value::Str(name) = name {
                        kinds.push(name.clone());
                    }
                }
                Some(ExpectedType::Many(kinds))
            }
            _ if self.is_object_schema(resolved) => Some(ExpectedType::One("object".into())),
            _ if self.is_array_schema(resolved) => Some(ExpectedType::One("array".into())),
            _ => None,
        };
        let repaired = match expected {
            Some(ExpectedType::One(name)) if name == "object" => {
                self.repair_object(value, resolved, path, depth)?
            }
            Some(ExpectedType::One(name)) if name == "array" => {
                self.repair_array(value, resolved, path, depth)?
            }
            Some(ExpectedType::One(name)) => self.coerce_scalar(value, &name, resolved, path)?,
            Some(ExpectedType::Many(kinds)) => {
                self.repair_type_union(value, &kinds, resolved, path, depth)?
            }
            None => normalize_missing_values(value)?,
        };
        self.apply_enum_const(repaired, resolved, path)
    }

    /// Upstream's `_repair_union` (oneOf/anyOf): the first branch whose
    /// repaired candidate also validates wins; each branch works on its own
    /// clone (upstream's per-branch deepcopy).
    fn repair_union(
        &self,
        value: Value,
        subs: &[Value],
        path: &str,
        depth: usize,
    ) -> Result<Value, String> {
        let mut last_error: Option<String> = None;
        for sub in subs {
            // Each branch's diagnostics live only if the branch wins: a
            // failed branch's records describe actions whose result was
            // discarded (upstream's append-only log has the same noise,
            // but the diagnostics API reports the repairs that produced
            // the returned value).
            let attempt = self.probe(|s| {
                s.repair_value_d(value.clone(), sub, path, depth + 1)
                    .and_then(|candidate| s.validate(&candidate, sub).map(|()| candidate))
            });
            match attempt {
                Ok(candidate) => return Ok(candidate),
                Err(message) => last_error = Some(message),
            }
        }
        Err(last_error.unwrap_or_else(|| "No schema matched the value.".into()))
    }

    /// Upstream's `_repair_type_union`: same shape, over synthesized
    /// per-type branches (`{**schema, "type": t}` minus the list).
    fn repair_type_union(
        &self,
        value: Value,
        kinds: &[String],
        schema: &Value,
        path: &str,
        depth: usize,
    ) -> Result<Value, String> {
        let mut last_error: Option<String> = None;
        for kind in kinds {
            let branch = synthesize_branch(schema, kind);
            // See repair_union: a losing branch leaves no diagnostics.
            let attempt = self.probe(|s| {
                s.repair_by_type(value.clone(), kind, schema, path, depth)
                    .and_then(|candidate| s.apply_enum_const(candidate, &branch, path))
                    .and_then(|candidate| s.validate(&candidate, &branch).map(|()| candidate))
            });
            match attempt {
                Ok(candidate) => return Ok(candidate),
                Err(message) => last_error = Some(message),
            }
        }
        Err(last_error.unwrap_or_else(|| "No schema type matched the value.".into()))
    }

    fn repair_by_type(
        &self,
        value: Value,
        kind: &str,
        schema: &Value,
        path: &str,
        depth: usize,
    ) -> Result<Value, String> {
        match kind {
            "array" => self.repair_array(value, schema, path, depth),
            "object" => self.repair_object(value, schema, path, depth),
            _ => self.coerce_scalar(value, kind, schema, path),
        }
    }

    /// The repair side of double-serialized containers: a `Str` that parses
    /// (strictly) to the expected container type is unwrapped in both modes;
    /// salvage additionally re-repairs a malformed string through the plain
    /// repair flow and takes the result when it lands the right shape.
    fn load_json_string_container(
        &self,
        value: Value,
        expected: char,
        path: &str,
        unwrap_detail: &str,
        salvage_detail: &str,
    ) -> Value {
        let Value::Str(text) = &value else {
            return value;
        };
        if let Ok(parsed) = loads_strict(text)
            && matches!(
                (expected, &parsed),
                ('[', Value::Array(_)) | ('{', Value::Object(_))
            )
        {
            self.record(
                "unwrap_string",
                path,
                unwrap_detail,
                Some(value),
                Some(parsed.clone()),
                None,
            );
            return parsed;
        }
        if self.salvage
            && let Ok((repaired, _)) = repair(
                text,
                &RepairConfig {
                    skip_json_loads: true,
                    ..RepairConfig::default()
                },
            )
            && matches!(
                (expected, &repaired),
                ('[', Value::Array(_)) | ('{', Value::Object(_))
            )
        {
            self.record(
                "unwrap_string",
                path,
                salvage_detail,
                Some(value),
                Some(repaired.clone()),
                None,
            );
            return repaired;
        }
        value
    }

    /// Upstream's `_repair_array`: string-container unwrapping, the
    /// non-list wrap, the single/tuple items pipeline, salvage item
    /// dropping, the `minItems` gate: plus the tors-native comma-split
    /// recovery, which wins only when upstream's wrap-singleton does not
    /// validate and the split does (tie: parity's wrap).
    fn repair_array(
        &self,
        value: Value,
        schema: &Value,
        path: &str,
        depth: usize,
    ) -> Result<Value, String> {
        let value = self.load_json_string_container(
            value,
            '[',
            path,
            "Unwrapped JSON string to array to match schema",
            "Repaired malformed JSON string to array to match schema",
        );
        // The items pipeline for one candidate item list, so the split and
        // wrap candidates share the exact same code.
        let pipeline = |items: Vec<Value>| self.repair_items(items, schema, path, depth);
        match value {
            Value::Array(items) => pipeline(items).map(Value::Array),
            other => {
                let wrapped = pipeline(vec![other.clone()]);
                // Tors-native 1b: the value stayed a string (it did not
                // unwrap as a JSON container) and carries commas. If the
                // schema actually guides items and the split validates where
                // upstream's wrap fails, the model meant a list: take the
                // split. When both validate, parity's wrap wins; when
                // neither does, the wrap's error propagates, also parity.
                if let Value::Str(text) = &other
                    && text.contains(',')
                    && array_is_guided(schema)
                    && let items = comma_parts(text)
                        .into_iter()
                        .map(Value::Str)
                        .collect::<Vec<_>>()
                    && !items.is_empty()
                    && let Ok(split) = pipeline(items)
                    && wrapped.is_err()
                {
                    self.record(
                        "coerce",
                        path,
                        "split comma-separated string to array",
                        Some(other),
                        Some(Value::Array(split.clone())),
                        None,
                    );
                    return Ok(Value::Array(split));
                }
                // Upstream logs the wrap; §6.4's diagnostic vocabulary maps
                // it to "wrap_array".
                let to = wrapped
                    .as_ref()
                    .ok()
                    .map(|items| Value::Array(items.clone()));
                self.record(
                    "wrap_array",
                    path,
                    "Wrapped value in array to match schema",
                    Some(other),
                    to,
                    None,
                );
                wrapped.map(Value::Array)
            }
        }
    }

    /// The shared items pipeline: single-schema and tuple-form items,
    /// `additionalItems`, salvage dropping, the `minItems` gate.
    fn repair_items(
        &self,
        items: Vec<Value>,
        schema: &Value,
        path: &str,
        depth: usize,
    ) -> Result<Vec<Value>, String> {
        let config = array_schema_config(schema);
        let mut out = Vec::with_capacity(items.len());
        match (config.items_list, config.items_schema) {
            (Some(tuple), _) => {
                for (idx, item_schema) in tuple.iter().enumerate() {
                    if idx >= items.len() {
                        break;
                    }
                    let item_path = format!("{path}[{idx}]");
                    match self.repair_or_drop(items[idx].clone(), item_schema, &item_path, depth) {
                        ItemFate::Keep(repaired) => out.push(repaired),
                        ItemFate::Drop => {}
                        ItemFate::Fail(message) => return Err(message),
                    }
                }
                let tail = &items[tuple.len().min(items.len())..];
                match config.additional_items.as_ref() {
                    Some(Value::Object(_)) => {
                        let extra = config.additional_items.as_ref().unwrap();
                        for (offset, item) in tail.iter().enumerate() {
                            let item_path = format!("{path}[{}]", tuple.len() + offset);
                            match self.repair_or_drop(item.clone(), extra, &item_path, depth) {
                                ItemFate::Keep(repaired) => out.push(repaired),
                                ItemFate::Drop => {}
                                ItemFate::Fail(message) => return Err(message),
                            }
                        }
                    }
                    Some(Value::Bool(true)) | None => {
                        // Upstream extends with the normalized tail when
                        // additionalItems is true or absent (only `false`
                        // drops tuple overflow).
                        out.extend(
                            tail.iter()
                                .cloned()
                                .map(normalize_missing_values)
                                .collect::<Result<Vec<_>, _>>()?,
                        );
                    }
                    _ => {
                        for (offset, _) in tail.iter().enumerate() {
                            self.record(
                                "drop_item",
                                &format!("{path}[{}]", tuple.len() + offset),
                                "Dropped extra array item not covered by schema",
                                None,
                                None,
                                None,
                            );
                        }
                    }
                }
            }
            (None, Some(item_schema)) => {
                for (idx, item) in items.into_iter().enumerate() {
                    let item_path = format!("{path}[{idx}]");
                    match self.repair_or_drop(item, &item_schema, &item_path, depth) {
                        ItemFate::Keep(repaired) => out.push(repaired),
                        ItemFate::Drop => {}
                        ItemFate::Fail(message) => return Err(message),
                    }
                }
            }
            (None, None) => {
                out = items
                    .into_iter()
                    .map(normalize_missing_values)
                    .collect::<Result<Vec<_>, _>>()?;
            }
        }
        if let Some(min) = min_items(schema)
            && out.len() < min
        {
            return Err(format!("Array at {path} does not meet minItems."));
        }
        Ok(out)
    }

    /// The repair-or-drop primitive for array items: salvage drops failures
    /// (recorded), every other mode propagates them: upstream's exact
    /// rule, with schema-definition errors always propagating.
    fn repair_or_drop(
        &self,
        item: Value,
        item_schema: &Value,
        item_path: &str,
        depth: usize,
    ) -> ItemFate {
        match self.repair_value_d(item, item_schema, item_path, depth + 1) {
            Ok(repaired) => ItemFate::Keep(repaired),
            Err(message) if self.salvage && !is_schema_definition_error(&message) => {
                self.record(
                    "drop_item",
                    item_path,
                    "Dropped invalid array item while salvaging",
                    None,
                    None,
                    None,
                );
                ItemFate::Drop
            }
            Err(message) => ItemFate::Fail(message),
        }
    }

    /// Upstream's `_repair_object`: salvage list-to-object salvage + the
    /// root single-item unwrap, string-container unwrapping, the salvage
    /// required-fills, the required gate, then the object assembly:
    /// restructured in one deliberate way: extra-key resolution (pattern
    /// matches, the normalization ladder, the fuzzy ladder, and the
    /// additionalProperties keeps/drops) runs before the properties pass,
    /// so a remapped typo lands on its property instead of being dropped
    /// while the (absent) property takes a default. For every non-remap
    /// input the observable behavior is upstream's exactly.
    fn repair_object(
        &self,
        value: Value,
        schema: &Value,
        path: &str,
        depth: usize,
    ) -> Result<Value, String> {
        let mut value = value;
        // Salvage list-to-object: upstream tries the positional map first
        // and falls back to the root single-item unwrap when the map
        // refuses (same count required, per-key repair must succeed):
        // the `elif` belongs to the map attempt, not to the shape gate.
        if self.salvage
            && let Value::Array(items) = &value
            && self.can_map_list_to_object(schema)
        {
            if let Some(mapped) = self.map_list_to_object(items, schema, path, depth)? {
                self.record(
                    "map_array_to_object",
                    path,
                    "Mapped array to object by schema property order",
                    Some(value.clone()),
                    Some(Value::Object(mapped.clone())),
                    None,
                );
                value = Value::Object(mapped);
            } else if path == "$" && items.len() == 1 && matches!(items[0], Value::Object(_)) {
                let unwrapped = items[0].clone();
                self.record(
                    "unwrap_root_array",
                    path,
                    "Unwrapped single-item root array to object while salvaging",
                    Some(value),
                    Some(unwrapped.clone()),
                    None,
                );
                value = unwrapped;
            }
        }
        let mut value = self.load_json_string_container(
            value,
            '{',
            path,
            "Unwrapped JSON string to object to match schema",
            "Repaired malformed JSON string to object to match schema",
        );
        let Value::Object(_) = value else {
            return Err(format!(
                "Expected object at {path}, got {}.",
                value.type_name()
            ));
        };
        let config = object_schema_config(schema);
        // Salvage required-fills for safe sources only (upstream's
        // _fill_missing_required_for_salvage).
        if self.salvage {
            for key in &config.required {
                if value.object_get(key).is_some() {
                    continue;
                }
                let Some(prop) = config.property(key) else {
                    continue;
                };
                let key_path = format!("{path}.{key}");
                if let Some(filled) = self.salvage_fill(prop, &key_path) {
                    self.record(
                        "fill_required",
                        &key_path,
                        "Filled missing required property while salvaging",
                        None,
                        Some(filled.clone()),
                        None,
                    );
                    value.object_insert(key.clone(), filled);
                }
            }
        }
        // Extra-key resolution first (patterns, normalization ladder,
        // fuzzy ladder, additionalProperties keeps/drops): see the method
        // docs for why this precedes the properties pass. The plan keeps
        // input order so the emission pass below reproduces upstream's
        // value.items() ordering exactly (pattern folds inline with kept
        // extras, at their input positions).
        let mut extras: Vec<(String, Value, ExtraPlan)> = Vec::new();
        for (key, raw) in entries_of(&value) {
            if config.property(&key).is_some() {
                continue;
            }
            let key_path = format!("{path}.{key}");
            let (matched, unsupported) = config
                .pattern_properties
                .as_ref()
                .map_or((Vec::new(), Vec::new()), |pp| {
                    match_pattern_properties(pp, &key)
                });
            for pattern in &unsupported {
                self.record(
                    "suggest",
                    &key_path,
                    format!("Skipped unsupported patternProperties regex '{pattern}'"),
                    None,
                    None,
                    None,
                );
            }
            if !matched.is_empty() {
                extras.push((key, raw, ExtraPlan::Pattern(matched)));
                continue;
            }
            match self.key_ladder(&key, &raw, &value, &config, path, depth) {
                Ladder::Remap(target) => {
                    // The remap renames the entry onto the property; the
                    // properties pass below repairs it through the
                    // property's own schema.
                    self.record(
                        "remap_key",
                        &key_path,
                        format!("Remapped unknown key '{key}' to schema property '{target}'"),
                        Some(Value::Str(key.clone())),
                        Some(Value::Str(target.clone())),
                        None,
                    );
                    rename_entry(&mut value, &key, &target);
                }
                Ladder::Suggest(candidate) => {
                    self.record(
                        "suggest",
                        &key_path,
                        format!("Unknown key '{key}'"),
                        None,
                        None,
                        Some(format!("did you mean '{candidate}'?")),
                    );
                    extras.push((key, raw, extra_plan(&config)));
                }
                Ladder::Nothing => extras.push((key, raw, extra_plan(&config))),
            }
        }

        // The required gate, on the renamed set.
        let missing: Vec<&String> = config
            .required
            .iter()
            .filter(|key| value.object_get(key).is_none())
            .collect();
        if !missing.is_empty() {
            let names: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
            return Err(format!(
                "Missing required properties at {path}: {}",
                names.join(", ")
            ));
        }
        // The properties pass + pattern-folding + the kept extras.
        let mut repaired: Vec<(String, Value)> = Vec::new();
        for (key, prop) in &config.properties {
            let key_path = format!("{path}.{key}");
            if let Some(raw) = value.object_get(key) {
                repaired.push((
                    key.clone(),
                    self.repair_value_d(raw.clone(), prop, &key_path, depth + 1)?,
                ));
            } else if let Some(default) = prop_default(prop)
                && !config.required.iter().any(|r| r == key)
            {
                let filled = copy_json_value(default, &key_path, "default")?;
                self.record(
                    "insert_default",
                    &key_path,
                    "Inserted default value for missing property",
                    None,
                    Some(filled.clone()),
                    None,
                );
                repaired.push((key.clone(), filled));
            }
        }
        // The extras, in input order (upstream's value.items() pass):
        // pattern folds, additionalProperties repairs, keeps, drops.
        for (key, raw, plan) in extras {
            let key_path = format!("{path}.{key}");
            match plan {
                ExtraPlan::Pattern(matched) => {
                    let mut current =
                        self.repair_value_d(raw, &matched[0], &key_path, depth + 1)?;
                    for extra in &matched[1..] {
                        current = self.repair_value_d(current, extra, &key_path, depth + 1)?;
                    }
                    repaired.push((key, current));
                }
                ExtraPlan::RepairAdditional => {
                    let Some(extra @ Value::Object(_)) = config.additional_properties.as_ref()
                    else {
                        unreachable!("the plan is only built for dict additionalProperties");
                    };
                    repaired.push((key, self.repair_value_d(raw, extra, &key_path, depth + 1)?));
                }
                ExtraPlan::KeepNormalized => {
                    repaired.push((key, normalize_missing_values(raw)?));
                }
                ExtraPlan::Drop => {
                    self.record(
                        "drop_property",
                        &key_path,
                        "Dropped extra property not covered by schema",
                        Some(raw),
                        None,
                        None,
                    );
                }
            }
        }
        if let Some(min) = min_properties(schema)
            && repaired.len() < min
        {
            return Err(format!("Object at {path} does not meet minProperties."));
        }
        Ok(Value::Object(repaired))
    }

    /// The salvage-only safe fill for missing required properties: default
    /// \> const > enum[0] > empty container by shape (with the minItems/
    /// minProperties guards). Upstream returns it as a maybe-tuple; the
    /// Option shape is the same contract (Some only when a safe value
    /// exists).
    fn salvage_fill(&self, prop: &Value, path: &str) -> Option<Value> {
        let resolved = match self.resolve_chain(prop) {
            Ok(Chain::Schema(resolved)) => resolved,
            _ => return None,
        };
        let Value::Object(entries) = resolved else {
            return None;
        };
        if let Some(default) = get(entries, "default") {
            return copy_json_value(default, path, "default").ok();
        }
        if let Some(value) = get(entries, "const") {
            return copy_json_value(value, path, "const").ok();
        }
        if let Some(Value::Array(items)) = get(entries, "enum")
            && !items.is_empty()
        {
            return copy_json_value(&items[0], path, "enum").ok();
        }
        // The fill gate: a `type` that is not a plain string, or no type
        // and no object/array shape to infer from, leaves nothing safe to
        // fill (upstream refuses scalars in salvage too).
        if !matches!(get(entries, "type"), Some(Value::Str(_)))
            && !self.is_object_schema(resolved)
            && !self.is_array_schema(resolved)
        {
            return None;
        }
        if self.is_array_schema(resolved) && get(entries, "minItems").is_none() {
            return Some(Value::Array(vec![]));
        }
        if self.is_object_schema(resolved) && get(entries, "minProperties").is_none() {
            return Some(Value::Object(vec![]));
        }
        None
    }

    /// Whether a list may salvage-map to an object: the schema allows
    /// object, does not allow array. (Upstream's `_can_salvage_list_as_object`.)
    fn can_map_list_to_object(&self, schema: &Value) -> bool {
        self.is_object_schema(schema) && !self.is_array_schema(schema)
    }

    /// Upstream's `_map_list_to_object`: same count as the property list,
    /// each item repaired through its positional property; any failure (or
    /// a count mismatch) means no mapping.
    fn map_list_to_object(
        &self,
        items: &[Value],
        schema: &Value,
        path: &str,
        depth: usize,
    ) -> Result<Option<Vec<(String, Value)>>, String> {
        let Value::Object(entries) = schema else {
            return Ok(None);
        };
        let Some(Value::Object(props)) = entries
            .iter()
            .find(|(k, _)| k == "properties")
            .map(|(_, v)| v)
        else {
            return Ok(None);
        };
        if props.is_empty() || items.len() != props.len() {
            return Ok(None);
        }
        let mut mapped = Vec::with_capacity(props.len());
        for ((key, prop), item) in props.iter().zip(items.iter()) {
            let key_path = format!("{path}.{key}");
            match self.repair_value_d(item.clone(), prop, &key_path, depth + 1) {
                Ok(repaired) => mapped.push((key.clone(), repaired)),
                // Upstream's map loop re-raises SchemaDefinitionError and
                // only a plain ValueError counts as "this map will not
                // work"; salvage must not swallow a broken schema either.
                Err(message) if is_schema_definition_error(&message) => return Err(message),
                Err(_) => return Ok(None),
            }
        }
        Ok(Some(mapped))
    }

    /// The two-tier key ladder (tors-native). Tier one is mechanical: the
    /// normalization fold (case + separator style) matching exactly one
    /// property is deterministic confidence, so it remaps whenever the
    /// target is absent: permissive schemas included, because the
    /// un-remapped outcome strands real data on a dead key while the
    /// property takes its default. The rename is kept only when the value
    /// can live under the property (a speculative repair through the
    /// property's schema succeeds): an incompatible value stays on its
    /// original key (upstream's permissive-schema outcome) instead of
    /// turning valid input into a coercion failure. Tier two is a guess
    /// (jaro-winkler), so it remaps only when the alternative is loss or
    /// failure (`additionalProperties: false` would drop the key, or the
    /// property is required and the repair would fail); otherwise it
    /// suggests.
    fn key_ladder(
        &self,
        key: &str,
        raw: &Value,
        value: &Value,
        config: &ObjectSchemaConfig,
        path: &str,
        depth: usize,
    ) -> Ladder {
        let folded = fold_key(key);
        let mut fold_hits: Vec<&String> = Vec::new();
        for (prop, _) in &config.properties {
            if fold_key(prop) == folded {
                fold_hits.push(prop);
            }
        }
        if fold_hits.len() == 1 && value.object_get(fold_hits[0]).is_none() {
            let target = fold_hits[0];
            let compatible = config.property(target).is_some_and(|prop| {
                self.probe(|s| {
                    s.repair_value_d(raw.clone(), prop, &format!("{path}.{target}"), depth + 1)
                })
                .is_ok()
            });
            if compatible {
                return Ladder::Remap(target.clone());
            }
            // Incompatible value: fall through to the fuzzy tier, which
            // only suggests on permissive schemas.
        }
        let mut scored: Vec<(&String, f64)> = config
            .properties
            .iter()
            .map(|(prop, _)| (prop, jaro_winkler(key, prop, None).unwrap_or(0.0)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let winner = match scored.as_slice() {
            [(best, top), (_, second), ..]
                if *top >= TYPO_REMAP_MIN && top - second > TYPO_REMAP_MARGIN =>
            {
                Some((*best, *top))
            }
            [(only, top)] if *top >= TYPO_REMAP_MIN => Some((*only, *top)),
            _ => None,
        };
        if let Some((target, _)) = winner {
            if value.object_get(target).is_none()
                && self.remap_allowed(target, &config.required, config)
            {
                return Ladder::Remap(target.clone());
            }
            return Ladder::Suggest(target.clone());
        }
        // Below the remap bar: still offer the best candidate as a hint.
        if let Some((best, score)) = scored.first()
            && *score >= SUGGEST_MIN
        {
            return Ladder::Suggest((*best).clone());
        }
        Ladder::Nothing
    }

    /// The fuzzy tier's remap gate: upstream would drop the key
    /// (`additionalProperties: false`) or fail the repair outright (the
    /// property is required): only then does a guess beat the default.
    /// (The mechanical fold tier is ungated by design; see `key_ladder`.)
    fn remap_allowed(
        &self,
        target: &str,
        required: &[String],
        config: &ObjectSchemaConfig,
    ) -> bool {
        matches!(config.additional_properties, Some(Value::Bool(false)))
            || required.iter().any(|name| name == target)
    }

    /// The mechanical key pass for the strict fast path: walk a valid value
    /// alongside the schema and rename fold-matching keys onto their
    /// properties (the deterministic tier only: fuzzy guessing never
    /// rewrites input that already validates). Without this pass, a
    /// permissive schema would shortcut `{"First Name": ...}` as "already
    /// valid" and strand the data on a dead key; with it, the model's
    /// separator/case slop is repaired on every path. A rename is kept
    /// only when the value is valid under the target property, so the
    /// pass can never reduce validity: an incompatible value keeps its
    /// original (valid) key.
    pub(crate) fn normalize_keys(
        &self,
        value: &mut Value,
        schema: &Value,
        path: &str,
        depth: usize,
    ) {
        if depth > MAX_SCHEMA_DEPTH {
            return;
        }
        let resolved = match self.resolve_chain(schema) {
            Ok(Chain::Schema(resolved)) => resolved,
            _ => return,
        };
        // See suggest_scan: allOf is conjunctive guidance, reached like
        // the parent's own; oneOf/anyOf stay fast-path-unreached.
        for member in all_of_members(resolved) {
            self.normalize_keys(value, member, path, depth + 1);
        }
        match value {
            Value::Object(entries) => {
                let config = object_schema_config(resolved);
                // Collect renames before mutating: the target must be
                // absent both in the original keys and among earlier
                // renames (two slop spellings of one property must not
                // create a duplicate key: first spelling wins, in
                // document order). The value must also be valid under the
                // target property: this pass runs on already-valid input,
                // so a rename that would break validity (an incompatible
                // type) is skipped and the key stays a valid extra,
                // exactly upstream's permissive-schema outcome.
                let mut renames: Vec<(String, String)> = Vec::new();
                let slots: Vec<(String, Value)> = entries
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                for (key, raw) in &slots {
                    if config.property(key).is_some() {
                        continue;
                    }
                    let (matched, _) = match config.pattern_properties.as_ref() {
                        Some(patterns) => match_pattern_properties(patterns, key),
                        None => (Vec::new(), Vec::new()),
                    };
                    if !matched.is_empty() {
                        continue;
                    }
                    let folded = fold_key(key);
                    let hits: Vec<&String> = config
                        .properties
                        .iter()
                        .map(|(prop, _)| prop)
                        .filter(|prop| fold_key(prop) == folded)
                        .collect();
                    if hits.len() != 1 {
                        continue;
                    }
                    let target = hits[0];
                    let taken = entries.iter().any(|(k, _)| *k == *target)
                        || renames.iter().any(|(_, to)| to == target);
                    let compatible = config
                        .property(target)
                        .is_some_and(|prop| self.is_valid(raw, prop));
                    if !taken && compatible {
                        renames.push((key.clone(), target.clone()));
                    }
                }
                for (from, to) in renames {
                    let key_path = format!("{path}.{from}");
                    self.record(
                        "remap_key",
                        &key_path,
                        format!("Remapped unknown key '{from}' to schema property '{to}'"),
                        Some(Value::Str(from.clone())),
                        Some(Value::Str(to.clone())),
                        None,
                    );
                    if let Some(slot) = entries.iter_mut().find(|(k, _)| *k == from) {
                        slot.0 = to;
                    }
                }
                // Recurse through every guided member (properties first,
                // then pattern/additional for the extras that remain).
                for (key, prop) in &config.properties {
                    if let Some(slot) = entries.iter_mut().find(|(k, _)| k == key) {
                        let key_path = format!("{path}.{key}");
                        self.normalize_keys(&mut slot.1, prop, &key_path, depth + 1);
                    }
                }
                for slot in entries.iter_mut() {
                    if config.property(&slot.0).is_some() {
                        continue;
                    }
                    let key_path = format!("{path}.{}", slot.0);
                    let (matched, _) = match config.pattern_properties.as_ref() {
                        Some(patterns) => match_pattern_properties(patterns, &slot.0),
                        None => (Vec::new(), Vec::new()),
                    };
                    if let Some(first) = matched.first() {
                        self.normalize_keys(&mut slot.1, first, &key_path, depth + 1);
                    } else if let Some(Value::Object(_)) = config.additional_properties.as_ref() {
                        let extra = config.additional_properties.as_ref().unwrap();
                        self.normalize_keys(&mut slot.1, extra, &key_path, depth + 1);
                    }
                }
            }
            Value::Array(items) => {
                let config = array_schema_config(resolved);
                match (config.items_list, config.items_schema) {
                    (Some(tuple), _) => {
                        for (idx, item_schema) in tuple.iter().enumerate() {
                            if let Some(item) = items.get_mut(idx) {
                                self.normalize_keys(
                                    item,
                                    item_schema,
                                    &format!("{path}[{idx}]"),
                                    depth + 1,
                                );
                            }
                        }
                    }
                    (None, Some(item_schema)) => {
                        for (idx, item) in items.iter_mut().enumerate() {
                            self.normalize_keys(
                                item,
                                &item_schema,
                                &format!("{path}[{idx}]"),
                                depth + 1,
                            );
                        }
                    }
                    (None, None) => {}
                }
            }
            _ => {}
        }
    }

    /// Upstream's `_fill_missing`: const > enum[0] > default > type-derived
    /// ("" / 0 / false / [] with the minItems guard / {} with the
    /// minProperties guard / None), type lists first-branch-wins, shape
    /// inference when `type` is omitted.
    fn fill_missing(&self, schema: &Value, path: &str, depth: usize) -> Result<Value, String> {
        if depth > MAX_SCHEMA_DEPTH {
            return Err(
                "Input schema nesting exceeds the supported schema recursion depth.".into(),
            );
        }
        let Value::Object(entries) = schema else {
            return Err(format!("Cannot infer missing value at {path}."));
        };
        if let Some(value) = get(entries, "const") {
            self.record(
                "fill",
                path,
                "Filled missing value with const",
                None,
                Some(value.clone()),
                None,
            );
            return copy_json_value(value, path, "const");
        }
        if let Some(Value::Array(items)) = get(entries, "enum") {
            if items.is_empty() {
                return Err(format!("Enum at {path} has no values."));
            }
            self.record(
                "fill",
                path,
                "Filled missing value with first enum value",
                None,
                Some(items[0].clone()),
                None,
            );
            return copy_json_value(&items[0], path, "enum");
        }
        if let Some(default) = get(entries, "default") {
            self.record(
                "fill",
                path,
                "Filled missing value with default",
                None,
                Some(default.clone()),
                None,
            );
            return copy_json_value(default, path, "default");
        }
        if let Some(Value::Array(kinds)) = get(entries, "type") {
            for kind in kinds {
                let Value::Str(kind) = kind else { continue };
                let branch = synthesize_branch(schema, kind);
                if let Ok(filled) = self.fill_missing(&branch, path, depth + 1) {
                    return Ok(filled);
                }
            }
            return Err(format!("Cannot infer missing value at {path}."));
        }
        let kind: Option<&str> = match get(entries, "type") {
            Some(Value::Str(name)) => Some(name),
            None if self.is_object_schema(schema) => Some("object"),
            None if self.is_array_schema(schema) => Some("array"),
            _ => None,
        };
        let filled = match kind {
            Some("string") => {
                self.record(
                    "fill",
                    path,
                    "Filled missing value with empty string",
                    None,
                    Some(Value::Str(String::new())),
                    None,
                );
                Value::Str(String::new())
            }
            Some("integer") | Some("number") => {
                self.record(
                    "fill",
                    path,
                    "Filled missing value with 0",
                    None,
                    Some(Value::Int(0)),
                    None,
                );
                Value::Int(0)
            }
            Some("boolean") => {
                self.record(
                    "fill",
                    path,
                    "Filled missing value with false",
                    None,
                    Some(Value::Bool(false)),
                    None,
                );
                Value::Bool(false)
            }
            Some("array") => {
                if let Some(Value::Int(min)) = get(entries, "minItems")
                    && *min > 0
                {
                    return Err(format!("Array at {path} requires at least {min} items."));
                }
                self.record(
                    "fill",
                    path,
                    "Filled missing value with empty array",
                    None,
                    Some(Value::Array(vec![])),
                    None,
                );
                Value::Array(vec![])
            }
            Some("object") => {
                if let Some(Value::Int(min)) = get(entries, "minProperties")
                    && *min > 0
                {
                    return Err(format!(
                        "Object at {path} requires at least {min} properties."
                    ));
                }
                self.record(
                    "fill",
                    path,
                    "Filled missing value with empty object",
                    None,
                    Some(Value::Object(vec![])),
                    None,
                );
                Value::Object(vec![])
            }
            Some("null") => {
                self.record(
                    "fill",
                    path,
                    "Filled missing value with null",
                    None,
                    Some(Value::Null),
                    None,
                );
                Value::Null
            }
            _ => return Err(format!("Cannot infer missing value at {path}.")),
        };
        Ok(filled)
    }

    /// Upstream's `_coerce_scalar` with two tors-native tolerances: digit-
    /// group separators in numeric strings, and RFC 3339 date normalization
    /// behind `format: date`/`date-time`.
    fn coerce_scalar(
        &self,
        value: Value,
        kind: &str,
        schema: &Value,
        path: &str,
    ) -> Result<Value, String> {
        match kind {
            "string" => {
                if let Value::Str(text) = &value {
                    // Tors-native format normalizers, behind an explicit
                    // format declaration: date/date-time/time via jiff,
                    // uuid via the shape-gated lowercase pass.
                    if let Some(format) = schema_format(schema) {
                        match format {
                            "uuid" => {
                                if let Some(lower) = normalize_uuid(text) {
                                    self.record(
                                        "coerce",
                                        path,
                                        "Normalized uuid to lowercase",
                                        Some(value.clone()),
                                        Some(Value::Str(lower.clone())),
                                        None,
                                    );
                                    return Ok(Value::Str(lower));
                                }
                            }
                            _ => {
                                if let Some(date_format) = date_format_of(format) {
                                    match normalize_date(text, date_format) {
                                        DateOutcome::Normalized(normalized)
                                            if normalized != *text =>
                                        {
                                            self.record(
                                                "format_date",
                                                path,
                                                format!("Normalized {format} to canonical form"),
                                                Some(value.clone()),
                                                Some(Value::Str(normalized.clone())),
                                                None,
                                            );
                                            return Ok(Value::Str(normalized));
                                        }
                                        DateOutcome::Ambiguous => {
                                            self.record(
                                                "suggest",
                                                path,
                                                "Ambiguous date: month/day order unclear, \
                                                 left unchanged",
                                                None,
                                                None,
                                                Some(
                                                    "use an unambiguous form like 2024-04-13"
                                                        .into(),
                                                ),
                                            );
                                        }
                                        DateOutcome::Normalized(_) | DateOutcome::Invalid => {}
                                    }
                                }
                            }
                        }
                    }
                    return Ok(value);
                }
                match value {
                    Value::Int(n) => {
                        self.record(
                            "coerce",
                            path,
                            "Coerced number to string",
                            Some(Value::Int(n)),
                            Some(Value::Str(n.to_string())),
                            None,
                        );
                        Ok(Value::Str(n.to_string()))
                    }
                    Value::Float(f) => {
                        let text = py_float_repr(f);
                        self.record(
                            "coerce",
                            path,
                            "Coerced number to string",
                            Some(Value::Float(f)),
                            Some(Value::Str(text.clone())),
                            None,
                        );
                        Ok(Value::Str(text))
                    }
                    Value::BigInt(text) => {
                        self.record(
                            "coerce",
                            path,
                            "Coerced number to string",
                            Some(Value::BigInt(text.clone())),
                            Some(Value::Str(text.clone())),
                            None,
                        );
                        Ok(Value::Str(text))
                    }
                    Value::Str(text) => Ok(Value::Str(text)),
                    _ => Err(format!("Expected string at {path}.")),
                }
            }
            "integer" => match value {
                Value::Bool(_) => Err(format!("Expected integer at {path}.")),
                Value::Int(_) | Value::BigInt(_) => Ok(value),
                Value::Float(f) => {
                    if let Some(exact) = integral_value(f) {
                        self.record(
                            "coerce",
                            path,
                            "Coerced number to integer",
                            Some(Value::Float(f)),
                            Some(exact.clone()),
                            None,
                        );
                        return Ok(exact);
                    }
                    Err(format!("Expected integer at {path}."))
                }
                Value::Str(text) => self.coerce_numeric_string(&text, true, path, schema),
                _ => Err(format!("Expected integer at {path}.")),
            },
            "number" => match value {
                Value::Bool(_) => Err(format!("Expected number at {path}.")),
                Value::Int(_) | Value::BigInt(_) | Value::Float(_) => Ok(value),
                Value::Str(text) => self.coerce_numeric_string(&text, false, path, schema),
                _ => Err(format!("Expected number at {path}.")),
            },
            "boolean" => {
                if let Value::Bool(_) = value {
                    return Ok(value);
                }
                if let Value::Str(text) = &value {
                    let lowered = text.to_lowercase();
                    if matches!(lowered.as_str(), "true" | "yes" | "y" | "on" | "1") {
                        self.record(
                            "coerce",
                            path,
                            "Coerced string to boolean",
                            Some(value),
                            Some(Value::Bool(true)),
                            None,
                        );
                        return Ok(Value::Bool(true));
                    }
                    if matches!(lowered.as_str(), "false" | "no" | "n" | "off" | "0") {
                        self.record(
                            "coerce",
                            path,
                            "Coerced string to boolean",
                            Some(value),
                            Some(Value::Bool(false)),
                            None,
                        );
                        return Ok(Value::Bool(false));
                    }
                    return Err(format!("Expected boolean at {path}."));
                }
                // Int/Float in (0, 1): upstream compares Python-equality to
                // 0 and 1 (so 1.0 counts, 1.5 does not).
                if let Some(number) = as_number(&value) {
                    if number == 0.0 {
                        self.record(
                            "coerce",
                            path,
                            "Coerced number to boolean",
                            Some(value),
                            Some(Value::Bool(false)),
                            None,
                        );
                        return Ok(Value::Bool(false));
                    }
                    if number == 1.0 {
                        self.record(
                            "coerce",
                            path,
                            "Coerced number to boolean",
                            Some(value),
                            Some(Value::Bool(true)),
                            None,
                        );
                        return Ok(Value::Bool(true));
                    }
                    return Err(format!("Expected boolean at {path}."));
                }
                Err(format!("Expected boolean at {path}."))
            }
            "null" => {
                if let Value::Null = value {
                    Ok(value)
                } else {
                    Err(format!("Expected null at {path}."))
                }
            }
            // Upstream's SchemaDefinitionError subclasses ValueError, so it
            // surfaces through the same channel.
            _ => Err(format!("Unsupported schema type {kind} at {path}.")),
        }
    }

    /// The shared string-to-numeric coercion ladder for the integer and
    /// number arms: one typed helper instead of two duplicated tails.
    /// Tier 1: the whole string is the number. Tier 2: the string minus
    /// unambiguous noise (underscores, script forms, and, with a known
    /// locale, the locale's own separators normalized onto `.`/`,`).
    /// Tier 3: exactly one number token in the prose, percent suffixes
    /// coercing as fractions. Every tier records its action; the refusal
    /// that reaches the caller carries the retry-able reason (the error
    /// is designed to be fed back to the model: what failed, where, and
    /// the disambiguation knob that would fix it).
    fn coerce_numeric_string(
        &self,
        text: &str,
        integral: bool,
        path: &str,
        schema: &Value,
    ) -> Result<Value, String> {
        let expected = if integral { "integer" } else { "number" };
        let reject = || format!("Expected {expected} at {path}.");
        // Tier 1: the whole string (Python's int()/float() accept
        // surrounding whitespace, so parse the trimmed text).
        let trimmed = text.trim();
        if integral {
            if let Some(v) = parse_exact_integer(trimmed) {
                self.record(
                    "coerce",
                    path,
                    "Coerced string to integer",
                    Some(Value::Str(text.into())),
                    Some(v.clone()),
                    None,
                );
                return Ok(v);
            }
        } else if let Ok(number) = trimmed.parse::<f64>() {
            self.record(
                "coerce",
                path,
                "Coerced string to number",
                Some(Value::Str(text.into())),
                Some(Value::Float(number)),
                None,
            );
            return Ok(Value::Float(number));
        }
        if integral
            && let Ok(number) = trimmed.parse::<f64>()
            && let Some(exact) = integral_value(number)
        {
            self.record(
                "coerce",
                path,
                "Coerced number to integer",
                Some(Value::Str(text.into())),
                Some(exact.clone()),
                None,
            );
            return Ok(exact);
        }
        // Tier 2: unambiguous noise stripped (a known locale's separators
        // normalize here, making "1,234" German 1.234 / English 1234 by
        // data).
        let noiseless = match self.locale {
            NumericLocale::Known(spec) => normalize_locale(text, &spec),
            NumericLocale::Auto => normalize_script(text),
        };
        if noiseless != text {
            if integral && let Some(v) = parse_exact_integer(&noiseless) {
                self.record(
                    "coerce",
                    path,
                    "Coerced string to integer (normalized separators and digit groups)",
                    Some(Value::Str(text.into())),
                    Some(v.clone()),
                    None,
                );
                return Ok(v);
            }
            if let Ok(number) = noiseless.parse::<f64>() {
                if !integral {
                    self.record(
                        "coerce",
                        path,
                        "Coerced string to number (normalized separators and digit groups)",
                        Some(Value::Str(text.into())),
                        Some(Value::Float(number)),
                        None,
                    );
                    return Ok(Value::Float(number));
                }
                match integral_value(number) {
                    Some(exact) => {
                        self.record(
                            "coerce",
                            path,
                            "Coerced number to integer",
                            Some(Value::Str(text.into())),
                            Some(exact.clone()),
                            None,
                        );
                        return Ok(exact);
                    }
                    None => return Err(reject()),
                }
            }
        }
        // Tier 3: exactly one number token; a percent suffix is read by the
        // declared type: number fields hold the fraction ("50%" -> 0.5),
        // integer fields hold the percent count ("50%" -> 50, the
        // progress-integer convention), and plain tokens keep the declared
        // type's spelling (an integer field gets the integer, a number
        // field the float: Python's int()/float() typing).
        if let Some(found) = extract_single_number(text, self.locale) {
            if !found.percent
                && integral
                && let Some(v) = parse_exact_integer(&found.token)
            {
                self.record(
                    "coerce",
                    path,
                    "Extracted the single number from a string value",
                    Some(Value::Str(text.into())),
                    Some(v.clone()),
                    None,
                );
                return Ok(v);
            }
            if let Ok(number) = found.token.parse::<f64>() {
                let (value, detail) = if found.percent {
                    if integral {
                        (
                            number,
                            "Extracted the single number from a string value (percent as integer)",
                        )
                    } else {
                        (
                            number / 100.0,
                            "Extracted the single number from a string value (percent as fraction)",
                        )
                    }
                } else {
                    (number, "Extracted the single number from a string value")
                };
                if !integral {
                    self.record(
                        "coerce",
                        path,
                        detail,
                        Some(Value::Str(text.into())),
                        Some(Value::Float(value)),
                        None,
                    );
                    return Ok(Value::Float(value));
                }
                if let Some(exact) = integral_value(value) {
                    self.record(
                        "coerce",
                        path,
                        detail,
                        Some(Value::Str(text.into())),
                        Some(exact.clone()),
                        None,
                    );
                    return Ok(exact);
                }
                return Err(reject());
            }
        }
        // Tier 4 (Auto only): the ambiguous shapes the strict grammar
        // refuses. Assume en-US: the convention LLM output overwhelmingly
        // follows, but let the schema check the assumption first: both
        // separator readings are extracted, parsed under the declared
        // type, and validated against the property schema; a reading the
        // type or schema rejects is not a candidate. One survivor is the
        // deterministic answer; two survivors take the US reading with a
        // disclosure diagnostic naming the discarded reading's locale (the
        // assumption is never silent, and `locale=` always overrides it).
        if matches!(self.locale, NumericLocale::Auto) && text.chars().any(|c| c.is_ascii_digit()) {
            let mut candidates: Vec<(Value, &'static str)> = Vec::new();
            // Whether any grammar saw exactly one separated number token
            // whose readings the declared type rejected: the only inputs
            // where the locale knob could change the outcome (and so the
            // only ones the refusal suffix names it for).
            let mut separated_token_seen = false;
            for (spec, label) in [(LocaleSpec::DOT_COMMA, "en-US"), (EU_MIRROR, "de-DE")] {
                if let Some(found) = extract_single_number(text, NumericLocale::Known(spec)) {
                    match parse_extracted(&found, integral) {
                        Some(value) => candidates.push((value, label)),
                        None => separated_token_seen = true,
                    }
                }
            }
            candidates.dedup_by(|a, b| a.0.py_eq(&b.0));
            match candidates.len() {
                1 => {
                    let (value, _) = candidates.pop().expect("len checked");
                    self.record(
                        "coerce",
                        path,
                        "Disambiguated an ambiguous numeric format by the declared type",
                        Some(Value::Str(text.into())),
                        Some(value.clone()),
                        None,
                    );
                    return Ok(value);
                }
                2 => {
                    // The schema gets the deciding vote when it has one.
                    let valid: Vec<bool> = candidates
                        .iter()
                        .map(|(value, _)| self.is_valid(value, schema))
                        .collect();
                    let winner = match valid[..] {
                        [true, false] => Some(0),
                        [false, true] => Some(1),
                        _ => None,
                    };
                    let (value, discarded) = match winner {
                        Some(idx) => {
                            let winner = candidates.swap_remove(idx);
                            let discarded = candidates.pop().expect("two candidates");
                            (winner, discarded)
                        }
                        // Both (or neither) validate: the en-US assumption
                        // wins, disclosed with the override that would
                        // produce the other reading.
                        None => {
                            let winner = candidates.remove(0);
                            let discarded = candidates.pop().expect("two candidates");
                            (winner, discarded)
                        }
                    };
                    let detail = if winner.is_some() {
                        "Disambiguated an ambiguous numeric format against the schema"
                    } else {
                        "Assumed en-US separators for an ambiguous numeric format"
                    };
                    self.record(
                        "coerce",
                        path,
                        detail,
                        Some(Value::Str(text.into())),
                        Some(value.0.clone()),
                        None,
                    );
                    if winner.is_none() {
                        self.record(
                            "suggest",
                            path,
                            "Ambiguous numeric format assumed en-US; pass locale= to override",
                            None,
                            None,
                            Some(format!("locale='{}'", discarded.1)),
                        );
                    }
                    return Ok(value.0);
                }
                _ => {
                    if separated_token_seen {
                        // The retry-able refusal: a single separated number
                        // exists but its readings all failed the declared
                        // type, so the message names the knob.
                        return Err(format!(
                            "Expected {expected} at {path}. (string value could not be coerced: \
                             ambiguous numeric format — pass locale='en-US' or 'de-DE' (or a \
                             {{'decimal': ..., 'grouping': ...}} dict) if you know the separator \
                             convention)"
                        ));
                    }
                }
            }
        }
        Err(reject())
    }

    /// Upstream's `_apply_enum_const`: const is strict equality; enum is
    /// membership (Python `==` via `py_eq`, so `1` satisfies `enum:
    /// [1.0]`, upstream's real semantics). The tors-native "did you mean"
    /// suffix fires on enum misses against string members.
    fn apply_enum_const(&self, value: Value, schema: &Value, path: &str) -> Result<Value, String> {
        let Value::Object(entries) = schema else {
            return Ok(value);
        };
        if let Some(expected) = get(entries, "const")
            && !value.py_eq(expected)
        {
            return Err(format!("Value at {path} does not match const."));
        }
        if let Some(Value::Array(members)) = get(entries, "enum") {
            if members.iter().any(|member| value.py_eq(member)) {
                return Ok(value);
            }
            if let Some(hint) = closest_enum_member(&value, members) {
                self.record(
                    "suggest",
                    path,
                    format!("Value at {path} does not match enum"),
                    None,
                    None,
                    Some(format!("did you mean '{hint}'?")),
                );
                return Err(format!(
                    "Value at {path} does not match enum. Did you mean '{hint}'?"
                ));
            }
            return Err(format!("Value at {path} does not match enum."));
        }
        Ok(value)
    }
}

// ============================================================
// Validation glue: the Value <-> serde_json boundary and the
// jsonschema-crate validators (upstream's Python-jsonschema delegation).
// ============================================================

impl SchemaRepairer {
    /// Validate one instance against one (sub)schema. The resolved schema
    /// that is the root rides the precompiled root validator; every other
    /// subschema compiles fresh inside the root's `#/...` scope wrapper.
    /// Non-finite numbers reject (the boundary cannot represent them);
    /// compile failures surface their message. The final error carries the
    /// instance path: the "why and where" reasoning over upstream's bare
    /// message.
    pub(crate) fn validate(&self, value: &Value, schema: &Value) -> Result<(), String> {
        match self.resolve_chain(schema) {
            Ok(Chain::True) => Ok(()),
            Ok(Chain::False) => Err("Schema does not allow any values.".into()),
            Ok(Chain::Schema(resolved)) => {
                let instance = to_serde(value, 0)?;
                let owned;
                let validator = if std::ptr::eq(resolved, &self.root) {
                    match &self.root_validator {
                        Ok(compiled) => compiled,
                        Err(message) => return Err(message.clone()),
                    }
                } else {
                    let key = std::ptr::from_ref(resolved) as usize;
                    let cached = self.sub_validators.borrow().get(&key).cloned();
                    owned = match cached {
                        Some(cached) => cached,
                        None => {
                            let compiled = Arc::new(self.compile_sub(resolved)?);
                            self.sub_validators
                                .borrow_mut()
                                .insert(key, Arc::clone(&compiled));
                            compiled
                        }
                    };
                    &owned
                };
                validator
                    .validate(&instance)
                    .map_err(|err| validation_message(&err))
            }
            Err(message) => Err(message),
        }
    }

    /// The `is_valid` twin: compile problems and non-finite instances read
    /// as invalid, never as errors (upstream's boolean gate).
    pub(crate) fn is_valid(&self, value: &Value, schema: &Value) -> bool {
        self.validate(value, schema).is_ok()
    }

    /// Upstream's `root_validator.evolve(schema=...)` equivalent: compile a
    /// subschema standalone while inheriting the root's `$defs` (and
    /// draft-07 `definitions`) so `#/...` refs resolve in root scope.
    fn compile_sub(&self, branch: &Value) -> Result<jsonschema::Validator, String> {
        let mut map = Map::new();
        if let Value::Object(root_entries) = &self.root {
            if let Some(defs) = get(root_entries, "$defs") {
                map.insert("$defs".into(), prepare_for_validation(defs, 0)?);
            }
            if let Some(defs) = get(root_entries, "definitions") {
                map.insert("definitions".into(), prepare_for_validation(defs, 0)?);
            }
        }
        map.insert(
            "allOf".into(),
            serde_json::Value::Array(vec![prepare_for_validation(branch, 0)?]),
        );
        jsonschema::validator_for(&serde_json::Value::Object(map)).map_err(|err| format!("{err}"))
    }
}

/// The enriched failure message: the crate's message plus the instance
/// path (empty for root-level failures), so callers see what failed where.
fn validation_message(err: &jsonschema::ValidationError<'_>) -> String {
    let message = format!("{err}");
    let location = err.instance_path().to_string();
    if location.is_empty() {
        message
    } else {
        format!("{message} (at {location})")
    }
}

/// Closest string enum member by jaro-winkler at the report-only bar, or
/// nothing (never auto-remap: an enum miss is the caller's data to fix,
/// the suggestion is the help).
fn closest_enum_member(value: &Value, members: &[Value]) -> Option<String> {
    let Value::Str(text) = value else { return None };
    let mut best: Option<(&str, f64)> = None;
    for member in members {
        if let Value::Str(candidate) = member {
            let score = jaro_winkler(text, candidate, None).unwrap_or(0.0);
            if best.is_none_or(|(_, top)| score > top) {
                best = Some((candidate, score));
            }
        }
    }
    match best {
        Some((candidate, score)) if score >= SUGGEST_MIN => Some(candidate.to_string()),
        _ => None,
    }
}

/// Upstream's `_copy_json_value` for defaults/consts/enum members: deep
/// copy of real JSON; anything else (only the `Missing` sentinel can reach
/// here in this typed tree) is a schema-authoring error.
fn copy_json_value(value: &Value, path: &str, label: &str) -> Result<Value, String> {
    let label = label[..1].to_uppercase() + &label[1..];
    match value {
        Value::Null
        | Value::Bool(_)
        | Value::Int(_)
        | Value::BigInt(_)
        | Value::Float(_)
        | Value::Str(_) => Ok(value.clone()),
        Value::Array(items) => items
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                copy_json_value(item, &format!("{path}[{idx}]"), &label.to_lowercase())
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        Value::Object(entries) => entries
            .iter()
            .map(|(key, item)| {
                copy_json_value(item, &format!("{path}.{key}"), &label.to_lowercase())
                    .map(|copied| (key.clone(), copied))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Object),
        Value::Missing => Err(format!("{label} value at {path} is not JSON compatible.")),
    }
}

/// parser_schema.py's object config builder: owned clones keep the repairer
/// borrows out of the parser's `&mut` (schemas are small; the cost is one
/// compact clone per container parse: noted, accepted).
pub(crate) fn object_schema_config(schema: &Value) -> ObjectSchemaConfig {
    let Value::Object(entries) = schema else {
        return ObjectSchemaConfig::default();
    };
    let properties = match get(entries, "properties") {
        Some(Value::Object(props)) => props.clone(),
        _ => Vec::new(),
    };
    let pattern_properties = match get(entries, "patternProperties") {
        Some(Value::Object(_)) => get(entries, "patternProperties").cloned(),
        _ => None,
    };
    let additional_properties = get(entries, "additionalProperties").cloned();
    // Upstream stores required as a Python set: duplicates collapse.
    // The Vec keeps first-occurrence order (stable error messages) with
    // the same collapse.
    let required = match get(entries, "required") {
        Some(Value::Array(names)) => {
            let mut seen: Vec<String> = Vec::new();
            for name in names {
                if let Value::Str(name) = name
                    && !seen.iter().any(|s| s == name)
                {
                    seen.push(name.clone());
                }
            }
            seen
        }
        _ => Vec::new(),
    };
    ObjectSchemaConfig {
        properties,
        pattern_properties,
        additional_properties,
        required,
    }
}

/// parser_schema.py's array config builder: the list-form `items` (draft-07
/// tuple validation) lands in `items_list`; the single-schema form in
/// `items_schema`; neither is the "no guidance" case the parser treats as
/// plain repair.
pub(crate) fn array_schema_config(schema: &Value) -> ArraySchemaConfig {
    let Value::Object(entries) = schema else {
        return ArraySchemaConfig::default();
    };
    match get(entries, "items") {
        Some(Value::Array(tuple)) => ArraySchemaConfig {
            items_schema: None,
            items_list: Some(tuple.clone()),
            additional_items: get(entries, "additionalItems").cloned(),
        },
        Some(single) => ArraySchemaConfig {
            items_schema: Some(single.clone()),
            items_list: None,
            additional_items: get(entries, "additionalItems").cloned(),
        },
        None => ArraySchemaConfig {
            items_schema: None,
            items_list: None,
            additional_items: get(entries, "additionalItems").cloned(),
        },
    }
}

type ResolvedObjectSchema<'r> = (
    Option<&'r SchemaRepairer>,
    Option<Value>,
    Option<ObjectSchemaConfig>,
);

pub(crate) fn resolve_parser_object_schema<'r>(
    repairer: Option<&'r SchemaRepairer>,
    schema: Option<&Value>,
) -> Result<ResolvedObjectSchema<'r>, String> {
    let (Some(repairer), Some(schema)) = (repairer, schema) else {
        return Ok((None, schema.cloned(), None));
    };
    if let Value::Bool(true) = schema {
        return Ok((None, Some(schema.clone()), None));
    }
    let resolved = match repairer.resolve_schema(schema)? {
        ResolvedSchema::True => return Ok((None, Some(Value::Bool(true)), None)),
        ResolvedSchema::Schema(resolved) => resolved.clone(),
    };
    if !repairer.is_object_schema(&resolved) {
        return Ok((None, Some(resolved), None));
    }
    Ok((
        Some(repairer),
        Some(resolved.clone()),
        Some(object_schema_config(&resolved)),
    ))
}

type ResolvedArraySchema<'r> = (
    Option<&'r SchemaRepairer>,
    Option<Value>,
    Option<ArraySchemaConfig>,
);

pub(crate) fn resolve_parser_array_schema<'r>(
    repairer: Option<&'r SchemaRepairer>,
    schema: Option<&Value>,
) -> Result<ResolvedArraySchema<'r>, String> {
    let (Some(repairer), Some(schema)) = (repairer, schema) else {
        return Ok((None, schema.cloned(), None));
    };
    if let Value::Bool(true) = schema {
        return Ok((None, Some(schema.clone()), None));
    }
    let resolved = match repairer.resolve_schema(schema)? {
        ResolvedSchema::True => return Ok((None, Some(Value::Bool(true)), None)),
        ResolvedSchema::Schema(resolved) => resolved.clone(),
    };
    if !repairer.is_array_schema(&resolved) {
        return Ok((None, Some(resolved), None));
    }
    Ok((
        Some(repairer),
        Some(resolved.clone()),
        Some(array_schema_config(&resolved)),
    ))
}

/// pattern_properties.py's port: the safe literal-plus-anchor subset.
/// Anchored `^token` matches starts, `token$` ends, `^token$` equality, a
/// bare token matches containment: anything whose literal remainder uses
/// regex metacharacters is skipped and reported through the unsupported
/// list for the caller to note.
pub(crate) fn match_pattern_properties(
    pattern_properties: &Value,
    key: &str,
) -> (Vec<Value>, Vec<String>) {
    let Value::Object(patterns) = pattern_properties else {
        return (Vec::new(), Vec::new());
    };
    let mut matched = Vec::new();
    let mut unsupported = Vec::new();
    for (pattern, sub) in patterns {
        let anchored_start = pattern.starts_with('^');
        let anchored_end = pattern.ends_with('$');
        let from = usize::from(anchored_start);
        // The `-1 if anchored_end` slice: byte-safe because both anchors
        // are ASCII (one byte), so the cuts always land on char boundaries.
        let literal = &pattern[from..pattern.len() - usize::from(anchored_end)];
        if literal
            .chars()
            .any(|c| UNSUPPORTED_REGEX_TOKENS.contains(&c))
        {
            unsupported.push(pattern.clone());
            continue;
        }
        let hit = match (anchored_start, anchored_end) {
            (true, true) => key == literal,
            (true, false) => key.starts_with(literal),
            (false, true) => key.ends_with(literal),
            (false, false) => key.contains(literal),
        };
        if hit {
            matched.push(sub.clone());
        }
    }
    (matched, unsupported)
}

/// Upstream's `normalize_missing_values`: the repair tree's empty corners
/// fold to `""`, recursively; the typed tree cannot hold non-JSON values
/// (the Python side's errors are structurally unreachable), so this is a
/// total walk over a `Result` skin the callers share.
pub(crate) fn normalize_missing_values(value: Value) -> Result<Value, String> {
    match value {
        Value::Missing => Ok(Value::Str(String::new())),
        Value::Array(items) => items
            .into_iter()
            .map(normalize_missing_values)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        Value::Object(entries) => entries
            .into_iter()
            .map(|(key, item)| normalize_missing_values(item).map(|value| (key, value)))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Object),
        scalar => Ok(scalar),
    }
}

// ============================================================
// Small typed helpers: keyword accessors, branch synthesis, key
// folding, entry surgery, the Value <-> serde_json boundary, dates.
// ============================================================

/// Read one keyword off an object-schema's entries (the typed `dict.get`).
fn get<'a>(entries: &'a [(String, Value)], key: &str) -> Option<&'a Value> {
    entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn get_array<'a>(entries: &'a [(String, Value)], key: &str) -> Option<&'a [Value]> {
    match get(entries, key) {
        Some(Value::Array(items)) => Some(items),
        _ => None,
    }
}

/// The `"default"` member of one property schema, when it is there.
fn prop_default(prop: &Value) -> Option<&Value> {
    if let Value::Object(entries) = prop {
        get(entries, "default")
    } else {
        None
    }
}

/// Upstream's `{**schema, "type": t}` synthesized branch for type-union
/// member `t`: the same keyword set, narrowed to one type.
fn synthesize_branch(schema: &Value, kind: &str) -> Value {
    let Value::Object(entries) = schema else {
        return schema.clone();
    };
    let mut branch: Vec<(String, Value)> = entries
        .iter()
        .filter(|(k, _)| k != "type")
        .cloned()
        .collect();
    branch.push(("type".into(), Value::Str(kind.into())));
    Value::Object(branch)
}

/// The schema's own nesting depth (dict/array levels): upstream's
/// deep-schema failures come from validation-side recursion, so the
/// constructor gates the tree itself: a schema deeper than the cap can
/// never validate, and carries the normalized error from the start.
fn schema_nesting(value: &Value, depth: usize) -> usize {
    match value {
        Value::Object(entries) => entries
            .iter()
            .map(|(_, member)| schema_nesting(member, depth + 1))
            .max()
            .unwrap_or(depth),
        Value::Array(items) => items
            .iter()
            .map(|item| schema_nesting(item, depth + 1))
            .max()
            .unwrap_or(depth),
        _ => depth,
    }
}

/// Whether the schema tree declares any `format: date`/`date-time`
/// (depth-capped): the fast path only pays for the date pre-pass when one
/// of these exists.
fn schema_has_formats(schema: &Value, depth: usize) -> bool {
    if depth > MAX_SCHEMA_DEPTH {
        return false;
    }
    match schema {
        Value::Object(entries) => {
            if let Some(Value::Str(format)) = get(entries, "format")
                && (date_format_of(format).is_some() || format == "uuid")
            {
                return true;
            }
            entries
                .iter()
                .any(|(_, member)| schema_has_formats(member, depth + 1))
        }
        Value::Array(items) => items.iter().any(|item| schema_has_formats(item, depth + 1)),
        _ => false,
    }
}

/// The `type` keyword test behind the shape probes: a string spelling or a
/// member of the list spelling.
fn schema_type_allows(entries: &[(String, Value)], kind: &str) -> bool {
    match get(entries, "type") {
        Some(Value::Str(name)) => name == kind,
        // A non-string, non-list `type` (e.g. a number) is a schema-author
        // error the validator reports; the shape probes stay quiet.
        Some(Value::Array(names)) => names
            .iter()
            .any(|name| matches!(name, Value::Str(name) if name == kind)),
        _ => false,
    }
}

/// The normalization fold behind the deterministic key ladder: lowercase,
/// then drop every separator and whitespace char, so `first_name`,
/// `first-name`, `First Name` and `FIRSTNAME` all fold to `firstname`.
/// Only the exact one-property fold match remaps: everything else falls
/// through to the fuzzy tier.
fn fold_key(key: &str) -> String {
    key.to_lowercase()
        .chars()
        .filter(|c| *c != '_' && *c != '-' && !c.is_whitespace())
        .collect()
}

/// The tors-native comma-split candidate parts: split on top-level commas,
/// trimmed; empty parts stay (they fail validation naturally unless the
/// items schema allows empty strings).
fn comma_parts(text: &str) -> Vec<String> {
    text.split(',')
        .map(|part| part.trim().to_string())
        .collect()
}

/// Whether the array schema actually guides items (the comma-split only
/// runs when there is an items schema to repair through).
fn array_is_guided(schema: &Value) -> bool {
    if let Value::Object(entries) = schema {
        matches!(
            get(entries, "items"),
            Some(Value::Array(_)) | Some(Value::Object(_))
        )
    } else {
        false
    }
}

/// Rename one entry in place (the remap's key surgery).
fn rename_entry(value: &mut Value, from: &str, to: &str) {
    if let Value::Object(entries) = value
        && let Some(slot) = entries.iter_mut().find(|(k, _)| k == from)
    {
        slot.0 = to.to_string();
    }
}

/// The emission plan for one non-property extra key, resolved during
/// the resolution pre-pass (so key-ladder renames land before the
/// properties pass) and consumed by the emission pass in input order:
/// upstream's `value.items()` ordering, pattern folds inline.
enum ExtraPlan {
    /// patternProperties matched: fold through the first schema, then the
    /// rest in declaration order.
    Pattern(Vec<Value>),
    /// `additionalProperties` is a dict: repair the value through it.
    RepairAdditional,
    /// `additionalProperties` true or absent: keep the normalized value.
    KeepNormalized,
    /// `additionalProperties` false: drop (recorded).
    Drop,
}

/// Classify one ladder-unresolved extra by the schema's
/// `additionalProperties` semantics.
fn extra_plan(config: &ObjectSchemaConfig) -> ExtraPlan {
    match config.additional_properties.as_ref() {
        Some(Value::Object(_)) => ExtraPlan::RepairAdditional,
        Some(Value::Bool(false)) => ExtraPlan::Drop,
        _ => ExtraPlan::KeepNormalized,
    }
}

/// Upstream re-raises `SchemaDefinitionError` (a ValueError subclass)
/// out of its salvage-drop sites: a broken schema is the caller's bug,
/// not salvageable data. The Rust port carries errors as strings, so the
/// §8 catalog's schema-definition messages are matched by prefix; the
/// catalog is closed and pinned by tests.
fn is_schema_definition_error(message: &str) -> bool {
    const SCHEMA_DEFINITION_PREFIXES: [&str; 7] = [
        "Unsupported schema type ",
        "Unresolvable $ref",
        "Unsupported $ref",
        "Circular $ref detected",
        "Schema must be an object",
        "Schema keys must be strings",
        "$ref must be a string",
    ];
    SCHEMA_DEFINITION_PREFIXES
        .iter()
        .any(|prefix| message.starts_with(prefix))
}

/// The effective-schema fan-out for the fast-path pre-passes: a resolved
/// schema's `allOf` members: conjunctive guidance that applies exactly
/// like the parent's own (pydantic v2 emits this shape for model
/// inheritance). Empty for schemas without `allOf`.
fn all_of_members(schema: &Value) -> Vec<&Value> {
    let Value::Object(entries) = schema else {
        return Vec::new();
    };
    match entries.iter().find(|(k, _)| k == "allOf") {
        Some((_, Value::Array(members))) => members.iter().collect(),
        _ => Vec::new(),
    }
}

/// A snapshot clone of an object's entries for iteration-while-mutating
/// (the extra-key resolution renames and drops in place).
fn entries_of(value: &Value) -> Vec<(String, Value)> {
    if let Value::Object(entries) = value {
        entries.clone()
    } else {
        Vec::new()
    }
}

fn min_items(schema: &Value) -> Option<usize> {
    if let Value::Object(entries) = schema
        && let Some(Value::Int(min)) = get(entries, "minItems")
    {
        return Some((*min).max(0) as usize);
    }
    None
}

fn min_properties(schema: &Value) -> Option<usize> {
    if let Value::Object(entries) = schema
        && let Some(Value::Int(min)) = get(entries, "minProperties")
    {
        return Some((*min).max(0) as usize);
    }
    None
}

/// The `format` declaration of a string subschema, when present.
fn schema_format(schema: &Value) -> Option<&str> {
    if let Value::Object(entries) = schema
        && let Some(Value::Str(format)) = get(entries, "format")
    {
        return Some(format);
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DateFormat {
    Date,
    DateTime,
    Time,
}

fn date_format_of(format: &str) -> Option<DateFormat> {
    match format {
        "date" => Some(DateFormat::Date),
        "date-time" => Some(DateFormat::DateTime),
        "time" => Some(DateFormat::Time),
        _ => None,
    }
}

/// The date parser's verdict: a successful parse (compare against the input
/// to find the no-op case: the caller owns that comparison), an
/// irresolvable month/day ambiguity, or a non-date.
enum DateOutcome {
    Normalized(String),
    Ambiguous,
    Invalid,
}

/// The date/time normalization engine: jiff does all parsing and every
/// calendar/clock validation (`strtime::parse` per shape, the civil
/// `FromStr` parsers for ISO, `BrokenDownTime::to_date`/`to_time` as the
/// validators, the civil types' ISO `Display` as the canonical output).
/// tors contributes only the accept-list (a declarative table of shapes)
/// and the one policy jiff cannot have: US vs day-first numeric dates,
/// where both shapes parsing is the ambiguity (jiff's own `%m` validation
/// already rejects month 13, so "13/04/2024" only parses day-first: the
/// \>12 disambiguation falls out of the library for free). Offset-bearing
/// date-times normalize to UTC (the same instant, `Z` rendering);
/// no-offset inputs keep their no-offset civil rendering; nothing is
/// ever invented.
static DATE_SHAPES: &[&str] = &[
    "%B %d, %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %B, %Y",
    "%d %b %Y",
    "%d %b, %Y",
    "%Y/%m/%d",
];
static US_SHAPE: &str = "%m/%d/%Y";
static DAY_FIRST_SHAPE: &str = "%d/%m/%Y";
// Space-separated date + clock, in either date spelling (dash or slash).
static DATETIME_SHAPES: &[&str] = &[
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
];
static TIME_SHAPES: &[&str] = &["%H:%M", "%H:%M:%S"];

fn parse_shape(format: &str, text: &str) -> Option<jiff::fmt::strtime::BrokenDownTime> {
    jiff::fmt::strtime::parse(format, text).ok()
}

fn normalize_date(text: &str, format: DateFormat) -> DateOutcome {
    let text = text.trim();
    match format {
        DateFormat::Date => {
            // ISO first (jiff's own FromStr: strict two-digit padding,
            // so "2024-3-5" stays untouched, while the strtime shapes
            // below are padding-tolerant, so "2024/3/5" normalizes).
            if let Ok(date) = text.parse::<jiff::civil::Date>() {
                return DateOutcome::Normalized(date.to_string());
            }
            for shape in DATE_SHAPES {
                if let Some(broken) = parse_shape(shape, text)
                    && broken.to_date().is_ok()
                {
                    return DateOutcome::Normalized(broken.to_date().expect("checked").to_string());
                }
            }
            // Both numeric shapes parsing is the ambiguity ("03/04/2024");
            // jiff's %m validation already resolved every >12 case.
            let us = parse_shape(US_SHAPE, text).filter(|b| b.to_date().is_ok());
            let day_first = parse_shape(DAY_FIRST_SHAPE, text).filter(|b| b.to_date().is_ok());
            match (us.is_some(), day_first.is_some()) {
                (true, false) => DateOutcome::Normalized(
                    us.expect("checked").to_date().expect("checked").to_string(),
                ),
                (false, true) => DateOutcome::Normalized(
                    day_first
                        .expect("checked")
                        .to_date()
                        .expect("checked")
                        .to_string(),
                ),
                (true, true) => DateOutcome::Ambiguous,
                (false, false) => DateOutcome::Invalid,
            }
        }
        DateFormat::DateTime => {
            // Offset-bearing (Z or ±HH:MM) input: the same instant,
            // normalized to UTC by jiff's Timestamp parser.
            if let Ok(stamp) = text.parse::<jiff::Timestamp>() {
                return DateOutcome::Normalized(stamp.to_string());
            }
            // Civil ISO (T-separated, optional subsecond, no offset).
            if let Ok(datetime) = text.parse::<jiff::civil::DateTime>() {
                return DateOutcome::Normalized(datetime.to_string());
            }
            // Space-separated ISO date + clock.
            for shape in DATETIME_SHAPES {
                if let Some(broken) = parse_shape(shape, text)
                    && broken.to_datetime().is_ok()
                {
                    return DateOutcome::Normalized(
                        broken.to_datetime().expect("checked").to_string(),
                    );
                }
            }
            DateOutcome::Invalid
        }
        DateFormat::Time => {
            for shape in TIME_SHAPES {
                if let Some(broken) = parse_shape(shape, text)
                    && broken.to_time().is_ok()
                {
                    return DateOutcome::Normalized(broken.to_time().expect("checked").to_string());
                }
            }
            DateOutcome::Invalid
        }
    }
}

/// `format: "uuid"` normalization: canonical lowercase, gated on the
/// actual UUID shape (a case-insensitive full match) so non-UUID strings
/// pass through untouched for validation to judge.
static UUID_SHAPE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
        .expect("the uuid shape is a compile-time constant")
});

fn normalize_uuid(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if UUID_SHAPE.is_match(trimmed) && trimmed != trimmed.to_lowercase() {
        Some(trimmed.to_lowercase())
    } else {
        None
    }
}

/// An integral f64 as its exact integer Value: within i64 range the cast
/// is exact; beyond it, the exact decimal expansion of the binary value
/// (Python's unbounded `int(float)`: `int(1e30)` is
/// 1000000000000000019884624838656, never a saturating cast).
/// Non-integral or non-finite input is None.
fn integral_value(f: f64) -> Option<Value> {
    exact_decimal_of_float(f).map(|digits| {
        if f.abs() < 2f64.powi(63) {
            Value::Int(f as i64)
        } else {
            Value::BigInt(digits)
        }
    })
}

/// A pure integer text as its exact Value: i64 first, then the BigInt
/// fallback for out-of-range magnitudes (Python's unbounded `int(str)`).
fn parse_exact_integer(text: &str) -> Option<Value> {
    if let Ok(n) = text.parse::<i64>() {
        return Some(Value::Int(n));
    }
    normalize_big_int_text(text).map(Value::BigInt)
}

/// Parse one extracted token under the declared numeric type, percent
/// included (integer fields keep the count, number fields take the
/// fraction). The shared tier-3/tier-4 candidate parser.
fn parse_extracted(found: &ExtractedNumber, integral: bool) -> Option<Value> {
    if !found.percent
        && integral
        && let Some(v) = parse_exact_integer(&found.token)
    {
        return Some(v);
    }
    let number = found.token.parse::<f64>().ok()?;
    let value = if found.percent && !integral {
        number / 100.0
    } else {
        number
    };
    if integral {
        integral_value(value)
    } else {
        Some(Value::Float(value))
    }
}

/// Numeric value of an int/float-carrying `Value` for the boolean-coercion
/// membership test (`1.0` counts as `1`; `True` reaches here only via the
/// int lane: upstream excludes `bool` before this).
fn as_number(value: &Value) -> Option<f64> {
    match value {
        Value::Int(n) => Some(*n as f64),
        Value::Float(f) => Some(*f),
        Value::BigInt(text) => text.parse::<f64>().ok(),
        _ => None,
    }
}

/// The `Value` -> `serde_json::Value` boundary for the validator. Notes on
/// the lossy corners (module docs): objects cross through the sorted
/// `serde_json::Map` (keyword lookup is name-based, so validation is
/// unaffected); `BigInt` beyond `u64` falls back to `f64`; non-finite
/// floats reject the whole conversion (schema mode cannot represent them,
/// same as JSON itself); `Missing` cannot arrive (normalized upstream),
/// and renders as `""` defensively.
fn to_serde(value: &Value, depth: usize) -> Result<serde_json::Value, String> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err("Input schema nesting exceeds the supported schema recursion depth.".into());
    }
    match value {
        Value::Null => Ok(serde_json::Value::Null),
        Value::Bool(flag) => Ok(serde_json::Value::Bool(*flag)),
        Value::Int(n) => Ok(serde_json::Value::Number((*n).into())),
        Value::BigInt(text) => {
            if let Ok(unsigned) = text.parse::<u64>() {
                Ok(serde_json::Value::Number(unsigned.into()))
            } else if let Ok(signed) = text.parse::<i64>() {
                Ok(serde_json::Value::Number(signed.into()))
            } else if let Ok(float) = text.parse::<f64>() {
                serde_json::Number::from_f64(float)
                    .map(serde_json::Value::Number)
                    .ok_or_else(|| {
                        "Value contains non-finite numbers, which JSON Schema validation cannot represent.".to_string()
                    })
            } else {
                Err("Value contains non-finite numbers, which JSON Schema validation cannot represent.".into())
            }
        }
        Value::Float(number) => serde_json::Number::from_f64(*number)
            .map(serde_json::Value::Number)
            .ok_or_else(|| {
                "Value contains non-finite numbers, which JSON Schema validation cannot represent."
                    .to_string()
            }),
        Value::Str(text) => Ok(serde_json::Value::String(text.clone())),
        Value::Array(items) => items
            .iter()
            .map(|item| to_serde(item, depth + 1))
            .collect::<Result<Vec<_>, _>>()
            .map(serde_json::Value::Array),
        Value::Object(entries) => entries
            .iter()
            .map(|(key, item)| to_serde(item, depth + 1).map(|converted| (key.clone(), converted)))
            .collect::<Result<serde_json::Map<_, _>, _>>()
            .map(serde_json::Value::Object),
        Value::Missing => Ok(serde_json::Value::String(String::new())),
    }
}

/// Upstream's `_prepare_schema_for_validation_node`, run over the schema
/// `Value` into the validator's `serde_json` form: draft-07 `items`-lists
/// become `prefixItems` (with `additionalItems` hoisted into `items`),
/// everything else converts verbatim. This copy exists only for
/// validation: the repairer keeps the draft-07 spelling throughout.
fn prepare_for_validation(schema: &Value, depth: usize) -> Result<serde_json::Value, String> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err("Input schema nesting exceeds the supported schema recursion depth.".into());
    }
    if let Value::Object(entries) = schema {
        let mut map = Map::new();
        if let Some(Value::Array(tuple)) = get(entries, "items") {
            map.insert(
                "prefixItems".into(),
                serde_json::Value::Array(
                    tuple
                        .iter()
                        .map(|item| prepare_for_validation(item, depth + 1))
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            );
            match get(entries, "additionalItems") {
                Some(Value::Bool(false)) => {
                    map.insert("items".into(), serde_json::Value::Bool(false));
                }
                Some(extra @ Value::Object(_)) => {
                    map.insert("items".into(), prepare_for_validation(extra, depth + 1)?);
                }
                _ => {}
            }
        }
        for (key, member) in entries {
            if key == "additionalItems" {
                continue;
            }
            if key == "items" && matches!(get(entries, "items"), Some(Value::Array(_))) {
                continue;
            }
            map.insert(key.clone(), prepare_for_validation(member, depth + 1)?);
        }
        return Ok(serde_json::Value::Object(map));
    }
    if let Value::Array(items) = schema {
        return items
            .iter()
            .map(|item| prepare_for_validation(item, depth + 1))
            .collect::<Result<Vec<_>, _>>()
            .map(serde_json::Value::Array);
    }
    to_serde(schema, depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_log() -> SchemaRepairer {
        SchemaRepairer::new(Value::Bool(true), false, false, NumericLocale::Auto)
    }

    fn repairer() -> SchemaRepairer {
        SchemaRepairer::new(Value::Bool(true), false, true, NumericLocale::Auto)
    }

    fn obj(entries: Vec<(&str, Value)>) -> Value {
        Value::Object(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    #[test]
    fn coerce_string_passthrough_and_number_tostring() {
        let schema = obj(vec![("type", Value::Str("string".into()))]);
        let r = repairer();
        assert_eq!(
            r.repair_value(Value::Str("a".into()), &schema, "$")
                .unwrap(),
            Value::Str("a".into())
        );
        assert_eq!(
            r.repair_value(Value::Int(1), &schema, "$").unwrap(),
            Value::Str("1".into())
        );
        assert_eq!(
            r.repair_value(Value::Float(1.5), &schema, "$").unwrap(),
            Value::Str("1.5".into())
        );
        assert!(
            r.repair_value(Value::Bool(true), &schema, "$")
                .is_err_and(|e| e == "Expected string at $.")
        );
    }

    #[test]
    fn coerce_integer_string_and_separators() {
        let schema = obj(vec![("type", Value::Str("integer".into()))]);
        let r = repairer();
        // Tier 1: the whole string.
        assert_eq!(
            r.repair_value(Value::Str("1".into()), &schema, "$")
                .unwrap(),
            Value::Int(1)
        );
        assert_eq!(
            r.repair_value(Value::Str("1.0".into()), &schema, "$")
                .unwrap(),
            Value::Int(1)
        );
        // Tier 2: unambiguous noise (underscores only: a comma that is not
        // a proper group is the extraction tier's ambiguity to refuse).
        assert_eq!(
            r.repair_value(Value::Str("82_461_110".into()), &schema, "$")
                .unwrap(),
            Value::Int(82461110)
        );
        // Structural forms (Auto): both-separator and repeated groups.
        assert_eq!(
            r.repair_value(Value::Str("1,234,567".into()), &schema, "$")
                .unwrap(),
            Value::Int(1234567)
        );
        assert_eq!(
            r.repair_value(Value::Str("1,234.0".into()), &schema, "$")
                .unwrap(),
            Value::Int(1234)
        );
        // Extraction (Auto): single token in prose, sign before currency.
        assert_eq!(
            r.repair_value(Value::Str("USD 50".into()), &schema, "$")
                .unwrap(),
            Value::Int(50)
        );
        assert_eq!(
            r.repair_value(Value::Str("50x".into()), &schema, "$")
                .unwrap(),
            Value::Int(50)
        );
        assert_eq!(
            r.repair_value(Value::Str("-$50".into()), &schema, "$")
                .unwrap(),
            Value::Int(-50)
        );
        assert_eq!(
            r.repair_value(Value::Str("about 50 dollars".into()), &schema, "$")
                .unwrap(),
            Value::Int(50)
        );
        // Percent by declared type: integer keeps the count.
        assert_eq!(
            r.repair_value(Value::Str("50%".into()), &schema, "$")
                .unwrap(),
            Value::Int(50)
        );
        assert_eq!(
            r.repair_value(Value::Str("200%".into()), &schema, "$")
                .unwrap(),
            Value::Int(200)
        );
        // Tier 4 (Auto): the type itself disambiguates: "1,234" has only
        // one integral reading (1234; 1.234 is not an integer).
        assert_eq!(
            r.repair_value(Value::Str("1,234".into()), &schema, "$")
                .unwrap(),
            Value::Int(1234)
        );
        assert_eq!(
            r.repair_value(Value::Str("1.234.567".into()), &schema, "$")
                .unwrap(),
            Value::Int(1234567)
        );
        // A single separated number whose readings all fail the type
        // keeps the retry-able suffix naming the locale knob.
        let err = r
            .repair_value(Value::Str("1,23".into()), &schema, "$")
            .unwrap_err();
        assert!(err.starts_with("Expected integer at $."));
        assert!(err.contains("ambiguous numeric format"));
        // Prose with several numbers (or none) gets upstream's plain
        // rejection: the locale knob cannot change the outcome, so it
        // is not named.
        for plain in ["between 10 and 20", "0x10"] {
            let err = r
                .repair_value(Value::Str(plain.into()), &schema, "$")
                .unwrap_err();
            assert_eq!(err, "Expected integer at $.", "{plain}: {err}");
        }
        // Exactness: Python's int() is unbounded: big strings and big
        // floats become exact BigInts, never saturating casts.
        let big_str_schema = obj(vec![("type", Value::Str("integer".into()))]);
        assert_eq!(
            r.repair_value(
                Value::Str("12345678901234567890123".into()),
                &big_str_schema,
                "$"
            )
            .unwrap(),
            Value::BigInt("12345678901234567890123".into())
        );
        assert_eq!(
            r.repair_value(Value::Float(1e30), &big_str_schema, "$")
                .unwrap(),
            Value::BigInt("1000000000000000019884624838656".into())
        );
        // Type rules unchanged.
        assert_eq!(
            r.repair_value(Value::Float(2.0), &schema, "$").unwrap(),
            Value::Int(2)
        );
        assert!(r.repair_value(Value::Float(2.5), &schema, "$").is_err());
        assert!(r.repair_value(Value::Bool(true), &schema, "$").is_err());
        assert!(
            r.repair_value(Value::Str("abc".into()), &schema, "$")
                .is_err()
        );
    }

    #[test]
    fn coerce_number_extraction_and_percent() {
        let schema = obj(vec![("type", Value::Str("number".into()))]);
        let r = repairer();
        assert_eq!(
            r.repair_value(Value::Str("$1,234.56".into()), &schema, "$")
                .unwrap(),
            Value::Float(1234.56)
        );
        assert_eq!(
            r.repair_value(Value::Str("1.234,56".into()), &schema, "$")
                .unwrap(),
            Value::Float(1234.56)
        );
        // Percent as a fraction for number fields.
        assert_eq!(
            r.repair_value(Value::Str("50%".into()), &schema, "$")
                .unwrap(),
            Value::Float(0.5)
        );
        assert_eq!(
            r.repair_value(Value::Str("0.5%".into()), &schema, "$")
                .unwrap(),
            Value::Float(0.005)
        );
        // Tier 4 (Auto): comma-decimal shapes resolve to their only
        // coherent reading under a number type...
        assert_eq!(
            r.repair_value(Value::Str("0,5".into()), &schema, "$")
                .unwrap(),
            Value::Float(0.5)
        );
        assert_eq!(
            r.repair_value(Value::Str("1,23".into()), &schema, "$")
                .unwrap(),
            Value::Float(1.23)
        );
        assert_eq!(
            r.repair_value(Value::Str("1_000,5".into()), &schema, "$")
                .unwrap(),
            Value::Float(1000.5)
        );
        // ...and "1,234" (both readings coherent) takes the disclosed
        // en-US assumption, with the suggestion naming the discarded
        // reading's locale (suggesting the winner's own is a no-op).
        let r_fresh = SchemaRepairer::new(Value::Bool(true), false, true, NumericLocale::Auto);
        let value = r_fresh
            .repair_value(Value::Str("1,234".into()), &schema, "$")
            .unwrap();
        assert_eq!(value, Value::Float(1234.0));
        let diags = r_fresh.take_diagnostics();
        let suggests: Vec<&Diagnostic> = diags.iter().filter(|d| d.action == "suggest").collect();
        assert_eq!(suggests.len(), 1, "{diags:?}");
        assert_eq!(suggests[0].suggestion.as_deref(), Some("locale='de-DE'"));
    }

    #[test]
    fn numeric_locale_resolves_ambiguity() {
        let int = obj(vec![("type", Value::Str("integer".into()))]);
        let num = obj(vec![("type", Value::Str("number".into()))]);
        let en = LocaleSpec::DOT_COMMA;
        let de = locale_spec_from_tag("de-DE").expect("listed");
        let fr = locale_spec_from_tag("fr-FR").expect("listed");
        // Auto refuses "1,234"; a locale reads it by its own convention.
        let r_en = SchemaRepairer::new(Value::Bool(true), false, true, NumericLocale::Known(en));
        assert_eq!(
            r_en.repair_value(Value::Str("1,234".into()), &int, "$")
                .unwrap(),
            Value::Int(1234)
        );
        let r_de = SchemaRepairer::new(Value::Bool(true), false, true, NumericLocale::Known(de));
        assert_eq!(
            r_de.repair_value(Value::Str("1,234".into()), &num, "$")
                .unwrap(),
            Value::Float(1.234)
        );
        assert_eq!(
            r_de.repair_value(Value::Str("1.234".into()), &int, "$")
                .unwrap(),
            Value::Int(1234)
        );
        assert_eq!(
            r_de.repair_value(Value::Str("0,5".into()), &num, "$")
                .unwrap(),
            Value::Float(0.5)
        );
        // French: space grouping, comma decimal.
        let r_fr = SchemaRepairer::new(Value::Bool(true), false, true, NumericLocale::Known(fr));
        assert_eq!(
            r_fr.repair_value(Value::Str("1 234,56".into()), &num, "$")
                .unwrap(),
            Value::Float(1234.56)
        );
        // A custom dict carries separators the table does not.
        let custom = LocaleSpec::custom(',', '.');
        let r_custom =
            SchemaRepairer::new(Value::Bool(true), false, true, NumericLocale::Known(custom));
        assert_eq!(
            r_custom
                .repair_value(Value::Str("1,234".into()), &num, "$")
                .unwrap(),
            Value::Float(1.234)
        );
        // Tag resolution: region overrides, Lakh refusal, malformed.
        assert_eq!(
            locale_spec_from_tag("de-CH").expect("listed").grouping,
            GroupSeparator::Char('\u{2019}')
        );
        assert_eq!(
            locale_spec_from_tag("es-MX").expect("listed"),
            LocaleSpec::DOT_COMMA
        );
        assert!(locale_spec_from_tag("en-IN").is_none());
        assert!(locale_spec_from_tag("banana!").is_none());
        assert!(locale_spec_from_tag("en_US").is_some());
    }

    #[test]
    fn coerce_boolean_words_and_numbers() {
        let schema = obj(vec![("type", Value::Str("boolean".into()))]);
        let r = repairer();
        for yes in ["yes", "y", "on", "1", "YES", "True", "true"] {
            assert_eq!(
                r.repair_value(Value::Str(yes.into()), &schema, "$")
                    .unwrap(),
                Value::Bool(true),
                "{yes}"
            );
        }
        for no in ["no", "n", "off", "0", "False", "false"] {
            assert_eq!(
                r.repair_value(Value::Str(no.into()), &schema, "$").unwrap(),
                Value::Bool(false),
                "{no}"
            );
        }
        assert_eq!(
            r.repair_value(Value::Int(1), &schema, "$").unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            r.repair_value(Value::Int(0), &schema, "$").unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            r.repair_value(Value::Float(1.0), &schema, "$").unwrap(),
            Value::Bool(true)
        );
        assert!(r.repair_value(Value::Float(1.5), &schema, "$").is_err());
        assert!(
            r.repair_value(Value::Str("maybe".into()), &schema, "$")
                .is_err()
        );
    }

    #[test]
    fn coerce_number_and_null() {
        let schema = obj(vec![("type", Value::Str("number".into()))]);
        let r = repairer();
        assert_eq!(
            r.repair_value(Value::Str("1.5".into()), &schema, "$")
                .unwrap(),
            Value::Float(1.5)
        );
        assert_eq!(
            r.repair_value(Value::Str("1,234.5".into()), &schema, "$")
                .unwrap(),
            Value::Float(1234.5)
        );
        let null_schema = obj(vec![("type", Value::Str("null".into()))]);
        assert_eq!(
            r.repair_value(Value::Null, &null_schema, "$").unwrap(),
            Value::Null
        );
        assert!(r.repair_value(Value::Int(0), &null_schema, "$").is_err());
        let bogus = obj(vec![("type", Value::Str("bogus".into()))]);
        assert!(
            r.repair_value(Value::Int(1), &bogus, "$")
                .is_err_and(|e| e == "Unsupported schema type bogus at $.")
        );
    }

    #[test]
    fn missing_value_fills_by_type() {
        let r = repairer();
        let fills = [
            ("string", Value::Str(String::new())),
            ("integer", Value::Int(0)),
            ("number", Value::Int(0)),
            ("boolean", Value::Bool(false)),
            ("array", Value::Array(vec![])),
            ("object", Value::Object(vec![])),
            ("null", Value::Null),
        ];
        for (kind, expected) in fills {
            let schema = obj(vec![("type", Value::Str(kind.into()))]);
            assert_eq!(
                r.repair_value(Value::Missing, &schema, "$").unwrap(),
                expected,
                "{kind}"
            );
        }
        // const > enum[0] > default priority.
        let const_schema = obj(vec![("const", Value::Int(7))]);
        assert_eq!(
            r.repair_value(Value::Missing, &const_schema, "$").unwrap(),
            Value::Int(7)
        );
        let enum_schema = obj(vec![(
            "enum",
            Value::Array(vec![
                Value::Str("first".into()),
                Value::Str("second".into()),
            ]),
        )]);
        assert_eq!(
            r.repair_value(Value::Missing, &enum_schema, "$").unwrap(),
            Value::Str("first".into())
        );
        let default_schema = obj(vec![("default", Value::Str("x".into()))]);
        assert_eq!(
            r.repair_value(Value::Missing, &default_schema, "$")
                .unwrap(),
            Value::Str("x".into())
        );
        let empty_enum = obj(vec![("enum", Value::Array(vec![]))]);
        assert!(r.repair_value(Value::Missing, &empty_enum, "$").is_err());
    }

    #[test]
    fn missing_required_raises_and_optional_default_inserts() {
        let schema = obj(vec![
            ("type", Value::Str("object".into())),
            (
                "properties",
                obj(vec![
                    (
                        "required_value",
                        obj(vec![
                            ("type", Value::Str("integer".into())),
                            ("default", Value::Int(1)),
                        ]),
                    ),
                    (
                        "note",
                        obj(vec![
                            ("type", Value::Str("string".into())),
                            ("default", Value::Str("n/a".into())),
                        ]),
                    ),
                ]),
            ),
            (
                "required",
                Value::Array(vec![Value::Str("required_value".into())]),
            ),
        ]);
        let r = repairer();
        assert!(
            r.repair_value(Value::Object(vec![]), &schema, "$")
                .is_err_and(|e| e == "Missing required properties at $: required_value")
        );
        assert_eq!(
            r.repair_value(obj(vec![("required_value", Value::Int(1))]), &schema, "$")
                .unwrap(),
            obj(vec![
                ("required_value", Value::Int(1)),
                ("note", Value::Str("n/a".into()))
            ])
        );
    }

    #[test]
    fn double_serialized_unwrap_standard_and_salvage() {
        let schema = obj(vec![
            ("type", Value::Str("object".into())),
            (
                "properties",
                obj(vec![(
                    "summary",
                    obj(vec![
                        ("type", Value::Str("object".into())),
                        (
                            "properties",
                            obj(vec![
                                ("verdict", obj(vec![("type", Value::Str("string".into()))])),
                                (
                                    "confidence",
                                    obj(vec![("type", Value::Str("string".into()))]),
                                ),
                            ]),
                        ),
                        (
                            "required",
                            Value::Array(vec![
                                Value::Str("verdict".into()),
                                Value::Str("confidence".into()),
                            ]),
                        ),
                    ]),
                )]),
            ),
            ("required", Value::Array(vec![Value::Str("summary".into())])),
        ]);
        let raw = Value::Str("{\"verdict\": \"malicious\", \"confidence\": \"high\"}".into());
        let expected = obj(vec![
            ("verdict", Value::Str("malicious".into())),
            ("confidence", Value::Str("high".into())),
        ]);
        let std = SchemaRepairer::new(schema.clone(), false, false, NumericLocale::Auto);
        assert_eq!(
            std.repair_value(
                Value::Object(vec![("summary".into(), raw.clone())]),
                &schema,
                "$"
            )
            .unwrap(),
            Value::Object(vec![("summary".into(), expected.clone())])
        );
        // Malformed double-serialized strings repair only under salvage.
        let malformed = Value::Str("{verdict: malicious, confidence: high}".into());
        assert!(
            std.repair_value(
                Value::Object(vec![("summary".into(), malformed.clone())]),
                &schema,
                "$"
            )
            .is_err()
        );
        let salv = SchemaRepairer::new(schema.clone(), true, false, NumericLocale::Auto);
        assert_eq!(
            salv.repair_value(
                Value::Object(vec![("summary".into(), malformed)]),
                &schema,
                "$"
            )
            .unwrap(),
            Value::Object(vec![("summary".into(), expected)])
        );
    }

    #[test]
    fn union_anyof_with_defs_ref() {
        let schema = Value::Object(vec![
            (
                "$defs".into(),
                obj(vec![(
                    "Item",
                    obj(vec![
                        ("type", Value::Str("object".into())),
                        (
                            "properties",
                            obj(vec![(
                                "name",
                                obj(vec![
                                    ("type", Value::Str("string".into())),
                                    ("pattern", Value::Str("^example$".into())),
                                ]),
                            )]),
                        ),
                        ("required", Value::Array(vec![Value::Str("name".into())])),
                    ]),
                )]),
            ),
            ("type".into(), Value::Str("object".into())),
            (
                "properties".into(),
                obj(vec![(
                    "value",
                    obj(vec![(
                        "anyOf",
                        Value::Array(vec![
                            obj(vec![
                                ("type", Value::Str("array".into())),
                                (
                                    "items",
                                    obj(vec![("$ref", Value::Str("#/$defs/Item".into()))]),
                                ),
                            ]),
                            obj(vec![("type", Value::Str("null".into()))]),
                        ]),
                    )]),
                )]),
            ),
            (
                "required".into(),
                Value::Array(vec![Value::Str("value".into())]),
            ),
        ]);
        let r = SchemaRepairer::new(schema.clone(), false, false, NumericLocale::Auto);
        let good = obj(vec![(
            "value",
            Value::Array(vec![obj(vec![("name", Value::Str("example".into()))])]),
        )]);
        assert_eq!(
            r.repair_value(good, &schema, "$").unwrap(),
            obj(vec![(
                "value",
                Value::Array(vec![obj(vec![("name", Value::Str("example".into()))])])
            )])
        );
        let bad = obj(vec![(
            "value",
            Value::Array(vec![obj(vec![("name", Value::Int(1))])]),
        )]);
        assert!(r.repair_value(bad, &schema, "$").is_err());
    }

    #[test]
    fn ref_cycles_and_shapes() {
        let circular = obj(vec![
            ("$ref", Value::Str("#/definitions/a".into())),
            (
                "definitions",
                obj(vec![(
                    "a",
                    obj(vec![("$ref", Value::Str("#/definitions/a".into()))]),
                )]),
            ),
        ]);
        let r = SchemaRepairer::new(circular.clone(), false, false, NumericLocale::Auto);
        assert!(
            r.repair_value(Value::Object(vec![]), &circular, "$")
                .is_err_and(|e| e.contains("Circular $ref"))
        );
        let non_string = obj(vec![("$ref", Value::Int(123))]);
        let r2 = SchemaRepairer::new(non_string.clone(), false, false, NumericLocale::Auto);
        assert!(
            r2.repair_value(Value::Object(vec![]), &non_string, "$")
                .is_err_and(|e| e == "$ref must be a string.")
        );
        let foreign = obj(vec![("$ref", Value::Str("http://x/y.json".into()))]);
        let r3 = SchemaRepairer::new(foreign.clone(), false, false, NumericLocale::Auto);
        assert!(
            r3.repair_value(Value::Object(vec![]), &foreign, "$")
                .is_err_and(|e| e.contains("Unsupported $ref"))
        );
        let missing = obj(vec![("$ref", Value::Str("#/definitions/nope".into()))]);
        let r4 = SchemaRepairer::new(missing.clone(), false, false, NumericLocale::Auto);
        assert!(
            r4.repair_value(Value::Object(vec![]), &missing, "$")
                .is_err_and(|e| e.contains("Unresolvable $ref"))
        );
        assert!(
            no_log().is_object_schema(&obj(vec![("properties", obj(vec![]))]))
                && !no_log().is_object_schema(&Value::Bool(true))
        );
        assert!(
            no_log().is_array_schema(&obj(vec![("items", obj(vec![]))]))
                && !no_log().is_array_schema(&Value::Bool(true))
        );
    }

    #[test]
    fn deep_schemas_raise_instead_of_overflowing() {
        let mut schema = obj(vec![("type", Value::Str("string".into()))]);
        for _ in 0..550 {
            schema = obj(vec![("allOf", Value::Array(vec![schema]))]);
        }
        let r = SchemaRepairer::new(schema.clone(), false, false, NumericLocale::Auto);
        assert!(
            r.repair_value(Value::Str("ok".into()), &schema, "$")
                .is_err_and(|e| e.contains("schema recursion depth"))
        );
        let mut nested = obj(vec![("type", Value::Str("string".into()))]);
        for depth in 0..550 {
            nested = obj(vec![
                ("type", Value::Str("object".into())),
                (
                    "properties",
                    obj(vec![(format!("level_{depth}").as_str(), nested)]),
                ),
            ]);
        }
        let r2 = SchemaRepairer::new(nested.clone(), false, false, NumericLocale::Auto);
        // The properties-depth raise surfaces through the full repair flow
        // (validation-side, like upstream): the repair alone returns the
        // empty object, the final validation raises.
        assert_eq!(
            r2.repair_value(Value::Object(vec![]), &nested, "$")
                .unwrap(),
            Value::Object(vec![])
        );
        assert!(
            crate::json_repair::repair(
                "{}",
                &RepairConfig {
                    schema: Some(nested),
                    ..RepairConfig::default()
                }
            )
            .is_err_and(|e| e.contains("schema recursion depth"))
        );
    }

    #[test]
    fn validation_gate_accepts_and_rejects() {
        let schema = obj(vec![
            ("type", Value::Str("object".into())),
            (
                "properties",
                obj(vec![(
                    "value",
                    obj(vec![("type", Value::Str("integer".into()))]),
                )]),
            ),
            ("required", Value::Array(vec![Value::Str("value".into())])),
        ]);
        let r = SchemaRepairer::new(schema.clone(), false, false, NumericLocale::Auto);
        assert!(r.is_valid(&obj(vec![("value", Value::Int(1))]), &schema));
        assert!(!r.is_valid(&obj(vec![("value", Value::Str("x".into()))]), &schema));
        assert!(
            r.validate(&obj(vec![("value", Value::Str("x".into()))]), &schema)
                .is_err()
        );
        let nan = obj(vec![("value", Value::Float(f64::NAN))]);
        assert!(
            r.validate(&nan, &schema)
                .is_err_and(|e| e.contains("non-finite"))
        );
    }

    #[test]
    fn tuple_items_and_min_items() {
        let schema = obj(vec![
            ("type", Value::Str("array".into())),
            (
                "items",
                Value::Array(vec![
                    obj(vec![("type", Value::Str("integer".into()))]),
                    obj(vec![("type", Value::Str("string".into()))]),
                ]),
            ),
            ("minItems", Value::Int(2)),
        ]);
        let r = SchemaRepairer::new(schema.clone(), false, false, NumericLocale::Auto);
        // Positional tuple validation: "1" coerces to 1, 7 coerces to "7".
        assert_eq!(
            r.repair_value(
                Value::Array(vec![Value::Str("1".into()), Value::Int(7)]),
                &schema,
                "$"
            )
            .unwrap(),
            Value::Array(vec![Value::Int(1), Value::Str("7".into())])
        );
        assert!(
            r.repair_value(Value::Array(vec![Value::Int(1)]), &schema, "$")
                .is_err_and(|e| e == "Array at $ does not meet minItems.")
        );
    }

    #[test]
    fn salvage_drops_items_and_maps_lists() {
        let items_schema = obj(vec![
            ("type", Value::Str("object".into())),
            (
                "properties",
                obj(vec![
                    ("id", obj(vec![("type", Value::Str("integer".into()))])),
                    ("score", obj(vec![("type", Value::Str("number".into()))])),
                ]),
            ),
            (
                "required",
                Value::Array(vec![Value::Str("id".into()), Value::Str("score".into())]),
            ),
        ]);
        let schema = obj(vec![
            ("type", Value::Str("object".into())),
            (
                "properties",
                obj(vec![(
                    "items",
                    obj(vec![
                        ("type", Value::Str("array".into())),
                        ("items", items_schema),
                    ]),
                )]),
            ),
            ("required", Value::Array(vec![Value::Str("items".into())])),
        ]);
        let raw = obj(vec![(
            "items",
            Value::Array(vec![
                obj(vec![("id", Value::Int(1)), ("score", Value::Float(85.6))]),
                obj(vec![
                    ("id", Value::Int(2)),
                    ("score", Value::Str("N/A".into())),
                ]),
            ]),
        )]);
        let std = SchemaRepairer::new(schema.clone(), false, false, NumericLocale::Auto);
        assert!(std.repair_value(raw.clone(), &schema, "$").is_err());
        let salv = SchemaRepairer::new(schema.clone(), true, false, NumericLocale::Auto);
        assert_eq!(
            salv.repair_value(raw, &schema, "$").unwrap(),
            obj(vec![(
                "items",
                Value::Array(vec![obj(vec![
                    ("id", Value::Int(1)),
                    ("score", Value::Float(85.6))
                ])])
            )])
        );
        let map_schema = obj(vec![
            ("type", Value::Str("object".into())),
            (
                "properties",
                obj(vec![
                    ("name", obj(vec![("type", Value::Str("string".into()))])),
                    (
                        "tags",
                        obj(vec![
                            ("type", Value::Str("array".into())),
                            ("items", obj(vec![("type", Value::Str("string".into()))])),
                        ]),
                    ),
                ]),
            ),
            (
                "required",
                Value::Array(vec![Value::Str("name".into()), Value::Str("tags".into())]),
            ),
        ]);
        let salv_map = SchemaRepairer::new(map_schema.clone(), true, false, NumericLocale::Auto);
        let list = Value::Array(vec![
            Value::Str("hello".into()),
            Value::Array(vec![Value::Str("a".into()), Value::Str("b".into())]),
        ]);
        assert_eq!(
            salv_map.repair_value(list, &map_schema, "$").unwrap(),
            obj(vec![
                ("name", Value::Str("hello".into())),
                (
                    "tags",
                    Value::Array(vec![Value::Str("a".into()), Value::Str("b".into())])
                )
            ])
        );
    }

    #[test]
    fn key_ladder_normalization_before_fuzzy() {
        let schema = obj(vec![
            ("type", Value::Str("object".into())),
            (
                "properties",
                obj(vec![
                    (
                        "first_name",
                        obj(vec![("type", Value::Str("string".into()))]),
                    ),
                    ("age", obj(vec![("type", Value::Str("integer".into()))])),
                ]),
            ),
            (
                "required",
                Value::Array(vec![
                    Value::Str("first_name".into()),
                    Value::Str("age".into()),
                ]),
            ),
            ("additionalProperties", Value::Bool(false)),
        ]);
        let salvage = SchemaRepairer::new(schema.clone(), false, true, NumericLocale::Auto);
        // Kebab, case, and space variants fold to the same property.
        for variant in ["First Name", "first-name", "FIRST_NAME", "firstname"] {
            let input = obj(vec![
                (variant, Value::Str("Ada".into())),
                ("age", Value::Int(30)),
            ]);
            let out = salvage.repair_value(input, &schema, "$").unwrap();
            assert_eq!(
                out,
                obj(vec![
                    ("first_name", Value::Str("Ada".into())),
                    ("age", Value::Int(30))
                ]),
                "{variant}"
            );
        }
        let diags = salvage.take_diagnostics();
        assert!(diags.iter().filter(|d| d.action == "remap_key").count() >= 4);
        // An unknown key with no confident match drops cleanly.
        let dropped = obj(vec![
            ("zzqx", Value::Str("x".into())),
            ("first_name", Value::Str("Ada".into())),
            ("age", Value::Int(30)),
        ]);
        assert_eq!(
            salvage.repair_value(dropped, &schema, "$").unwrap(),
            obj(vec![
                ("first_name", Value::Str("Ada".into())),
                ("age", Value::Int(30))
            ])
        );
        // No remap when the target property is already present (the typo
        // stays a drop, never an overwrite of real data).
        let clash = obj(vec![
            ("first_name", Value::Str("Ada".into())),
            ("first-name", Value::Str("Eve".into())),
            ("age", Value::Int(30)),
        ]);
        assert_eq!(
            salvage.repair_value(clash, &schema, "$").unwrap(),
            obj(vec![
                ("first_name", Value::Str("Ada".into())),
                ("age", Value::Int(30))
            ])
        );
    }

    #[test]
    fn enum_suggestion_suffix() {
        let schema = obj(vec![
            ("type", Value::Str("string".into())),
            (
                "enum",
                Value::Array(vec![Value::Str("blue".into()), Value::Str("green".into())]),
            ),
        ]);
        let r = repairer();
        assert_eq!(
            r.repair_value(Value::Str("blue".into()), &schema, "$")
                .unwrap(),
            Value::Str("blue".into())
        );
        assert!(
            r.repair_value(Value::Str("blu".into()), &schema, "$")
                .is_err_and(|e| e == "Value at $ does not match enum. Did you mean 'blue'?")
        );
        let diags = r.take_diagnostics();
        assert!(
            diags.iter().any(|d| d.action == "suggest"
                && d.suggestion.as_deref() == Some("did you mean 'blue'?"))
        );
    }

    #[test]
    fn date_normalization_accept_list() {
        let date = obj(vec![
            ("type", Value::Str("string".into())),
            ("format", Value::Str("date".into())),
        ]);
        let r = repairer();
        assert_eq!(
            r.repair_value(Value::Str("2024/03/15".into()), &date, "$")
                .unwrap(),
            Value::Str("2024-03-15".into())
        );
        assert_eq!(
            r.repair_value(Value::Str("March 15, 2024".into()), &date, "$")
                .unwrap(),
            Value::Str("2024-03-15".into())
        );
        assert_eq!(
            r.repair_value(Value::Str("15 Mar 2024".into()), &date, "$")
                .unwrap(),
            Value::Str("2024-03-15".into())
        );
        assert_eq!(
            r.repair_value(Value::Str("2024-03-15".into()), &date, "$")
                .unwrap(),
            Value::Str("2024-03-15".into())
        );
        // 13 can only be a day: disambiguated.
        assert_eq!(
            r.repair_value(Value::Str("13/04/2024".into()), &date, "$")
                .unwrap(),
            Value::Str("2024-04-13".into())
        );
        // Both fit: ambiguous, unchanged, with a suggestion recorded.
        assert_eq!(
            r.repair_value(Value::Str("03/04/2024".into()), &date, "$")
                .unwrap(),
            Value::Str("03/04/2024".into())
        );
        assert!(r.take_diagnostics().iter().any(|d| d.action == "suggest"));
        // Impossible: unchanged.
        assert_eq!(
            r.repair_value(Value::Str("2024-02-30".into()), &date, "$")
                .unwrap(),
            Value::Str("2024-02-30".into())
        );
        assert_eq!(
            r.repair_value(Value::Str("2023-02-29".into()), &date, "$")
                .unwrap(),
            Value::Str("2023-02-29".into())
        );
        let datetime = obj(vec![
            ("type", Value::Str("string".into())),
            ("format", Value::Str("date-time".into())),
        ]);
        assert_eq!(
            r.repair_value(Value::Str("2024-03-15 14:30".into()), &datetime, "$")
                .unwrap(),
            Value::Str("2024-03-15T14:30:00".into())
        );
        assert_eq!(
            r.repair_value(
                Value::Str("2024-03-15T14:30:00.500+0530".into()),
                &datetime,
                "$"
            )
            .unwrap(),
            // Offset-bearing input normalizes to its UTC instant: the
            // same moment, jiff's rendering (minimal subsecond, Z).
            Value::Str("2024-03-15T09:00:00.5Z".into())
        );
        assert_eq!(
            r.repair_value(Value::Str("2024-03-15T14:30:00Z".into()), &datetime, "$")
                .unwrap(),
            Value::Str("2024-03-15T14:30:00Z".into())
        );
        // The time format: seconds always present, no offset invented.
        let time_only = obj(vec![
            ("type", Value::Str("string".into())),
            ("format", Value::Str("time".into())),
        ]);
        assert_eq!(
            r.repair_value(Value::Str("14:30".into()), &time_only, "$")
                .unwrap(),
            Value::Str("14:30:00".into())
        );
        assert_eq!(
            r.repair_value(Value::Str("14:30:00".into()), &time_only, "$")
                .unwrap(),
            Value::Str("14:30:00".into())
        );
        // The uuid format: shape-gated lowercase.
        let uuid = obj(vec![
            ("type", Value::Str("string".into())),
            ("format", Value::Str("uuid".into())),
        ]);
        assert_eq!(
            r.repair_value(
                Value::Str("A1B2C3D4-0000-1111-2222-333344445555".into()),
                &uuid,
                "$"
            )
            .unwrap(),
            Value::Str("a1b2c3d4-0000-1111-2222-333344445555".into())
        );
        // Non-uuid strings pass through untouched.
        assert_eq!(
            r.repair_value(Value::Str("not a uuid".into()), &uuid, "$")
                .unwrap(),
            Value::Str("not a uuid".into())
        );
    }

    #[test]
    fn comma_split_and_pattern_subset() {
        let schema = obj(vec![
            ("type", Value::Str("object".into())),
            (
                "properties",
                obj(vec![(
                    "items",
                    obj(vec![
                        ("type", Value::Str("array".into())),
                        ("items", obj(vec![("type", Value::Str("integer".into()))])),
                    ]),
                )]),
            ),
            ("required", Value::Array(vec![Value::Str("items".into())])),
        ]);
        let std = SchemaRepairer::new(schema.clone(), false, false, NumericLocale::Auto);
        // Upstream's wrap fails the items schema; the split wins.
        assert_eq!(
            std.repair_value(
                obj(vec![("items", Value::Str("1, 2, 3".into()))]),
                &schema,
                "$"
            )
            .unwrap(),
            obj(vec![(
                "items",
                Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
            )])
        );
        // Without commas upstream's wrap stands (parity): a single string
        // that cannot coerce to the items schema raises, exactly like
        // upstream.
        assert_eq!(
            std.repair_value(
                obj(vec![("items", Value::Str("not json".into()))]),
                &schema,
                "$"
            )
            .unwrap_err(),
            "Expected integer at $.items[0]."
        );
        let (matched, unsupported) = match_pattern_properties(
            &obj(vec![("^pre", obj(vec![])), ("a.c", obj(vec![]))]),
            "prefix",
        );
        assert_eq!(matched.len(), 1);
        assert_eq!(unsupported, vec!["a.c".to_string()]);
        assert_eq!(
            normalize_missing_values(Value::Missing).unwrap(),
            Value::Str(String::new())
        );
    }
}
