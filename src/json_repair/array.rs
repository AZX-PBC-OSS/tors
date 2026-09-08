//! The array-repair parser: a port of json_repair's `parse_array.py`
//! (upstream: https://github.com/mangiucugna/json_repair by Stefano
//! Baccianella, MIT, pinned at commit
//! 251d141786d0f6ff561f6ec04d90188a338e2470, version 0.63.4): the array
//! item loop (closing-delimiter aware, so the same loop parses `(...)`
//! groups), the string-then-`:` object-in-array recovery, the schema-layer
//! item gating (`parser_schema.py`'s array config, drop/additionalItems),
//! and `utils/object_comparer.py`'s `is_strictly_empty` skip rule. Every
//! branch is ported 1:1 in upstream's order of checks.
//!
//! # Porting notes (the decisions this file's dynamics forced)
//!
//! - `self.log(...)` sites upstream become rationale COMMENTS here (log
//!   texts are not ported; the heuristic each explained gets the comment),
//!   per the port contract in `parser.rs`'s docs. The parser-level
//!   `repairer._log("Dropped extra array item ...")` site is wired to
//!   §6.4's diagnostic vocabulary ("drop_item") through the take/put-back
//!   repairer helper.
//! - The ARRAY context brackets the WHOLE item loop: the body lives in
//!   [`Parser::parse_array_items`] (the with-region) and the pop in
//!   [`Parser::parse_array`] runs on every exit path, Err included — the
//!   `with self.context.enter(ARRAY)` translation `parser.rs`'s docs pin.
//! - The repairer never stays borrowed across parser mutation: the resolver
//!   runs once per `parse_array` call to an ACTIVITY bit plus an owned
//!   config copy, and every later schema-layer access re-acquires the
//!   repairer through the take/put-back helper ([`with_repairer`], the
//!   same shape `parser.rs` pins for its own repairer calls).
//! - Python passes `None` schema slots into `repair_value`; the pinned Rust
//!   signature takes `&Value`, and upstream's `resolve_schema(None) is True`
//!   (no constraints) makes `Value::Bool(true)` the None spelling at that
//!   boundary. JSON `null` in a schema slot IS Python's `None` and maps to
//!   `Option::None` where slots flow onward.
//! - `_resolve_array_item_schema`: Python dispatches on
//!   `isinstance(items_schema, list)`; the pinned Rust config carries the
//!   tuple form as `items_list` (and tolerates a raw `Value::Array` left in
//!   `items_schema`), the single-schema form as an object `items_schema`,
//!   and every other spelling (absent included) is upstream's `true`.
//! - `ObjectComparer.is_strictly_empty` is [`is_strictly_empty`] here: an
//!   EMPTY CONTAINER only (str/list/dict — Python also set/tuple, outside
//!   the Value domain); it is NOT `Value::is_truthy` (None/0/False are
//!   NOT strictly empty and must survive the skip rule).
//!
//! # Test provenance
//!
//! The `#[cfg(test)]` batteries port the VALUE-level assertions of
//! upstream's `tests/test_parse_array.py` (every case except the
//! parenthesized-literal ones, which live in `parenthesized.rs`'s tests as
//! the parser_parenthesized.py battery), driven through `Parser::parse`
//! (what `repair_json(..., skip_json_loads=True, return_objects=True)`
//! runs), plus the array-context strict case of
//! `tests/test_strict_mode.py`-adjacent behavior (the contextual close in
//! strict mode, which must NOT raise). The assertions that intentionally
//! live ONLY in the pytest corpus (`tests/test_json_repair.py`, agent C's):
//! the serialized-STRING forms of these inputs, the `logging=True`
//! log-text assertions (e.g. the missed-closing-bracket log), and every
//! case whose behavior belongs to `string.rs`/`parser.rs`'s own batteries.

use super::parser::{Ctx, Parser};
use super::{STRING_DELIMITERS, Value};
use crate::json_schema_impl::{ArraySchemaConfig, SchemaRepairer, resolve_parser_array_schema};
use crate::normalize_impl::is_py_whitespace;

/// `parser.rs`'s `with_repairer` take/put-back pattern, replicated for this
/// file (the landed helper is private to `parser`'s module): every
/// schema-layer call needs `&mut SchemaRepairer` while the parser state
/// around it needs `&mut self`, and the repairer must be back in place on
/// every path — `repair()` still needs it for the final validation.
fn with_repairer<T>(parser: &mut Parser, f: impl FnOnce(Option<&mut SchemaRepairer>) -> T) -> T {
    let mut taken = std::mem::take(&mut parser.schema_repairer);
    let out = f(taken.as_mut());
    parser.schema_repairer = taken;
    out
}

/// `utils/object_comparer.py`'s `ObjectComparer.is_strictly_empty`: True
/// only for an EMPTY CONTAINER (str/list/dict here); False for None, 0,
/// False — the array skip rule must not eat falsy scalars.
fn is_strictly_empty(value: &Value) -> bool {
    match value {
        Value::Str(text) => text.is_empty(),
        Value::Array(items) => items.is_empty(),
        Value::Object(entries) => entries.is_empty(),
        _ => false,
    }
}

/// Python's `raw_schema is not None and not isinstance(raw_schema, (dict,
/// bool))` raise shape, over the Value domain: an items-list slot must be
/// null (Python `None`), bool, or dict.
fn as_schema_slot(value: &Value) -> Result<Option<Value>, String> {
    match value {
        // JSON null is Python's None spelling for "no schema here".
        Value::Null => Ok(None),
        Value::Bool(_) | Value::Object(_) => Ok(Some(value.clone())),
        _ => Err("Schema must be an object.".into()),
    }
}

/// parse_array.py's `_resolve_array_item_schema`: the schema guiding item
/// `idx` — the tuple form's per-index slot (then additionalItems, with the
/// drop flag when it is `false`), the single-schema form for every index,
/// or `true` when nothing narrows the items. Returns
/// `(item_schema, drop_item)`.
fn resolve_array_item_schema(
    schema_config: Option<&ArraySchemaConfig>,
    idx: usize,
) -> Result<(Option<Value>, bool), String> {
    let Some(schema_config) = schema_config else {
        return Ok((None, false));
    };

    let item_schema: Option<Value>;
    let mut drop_item = false;
    // The tuple form: `items` was a list (array_schema_config carries it
    // as items_list; items_schema is only ever the single-schema form).
    let tuple_form: Option<&Vec<Value>> = schema_config.items_list.as_ref();
    if let Some(items_schema) = tuple_form {
        if idx < items_schema.len() {
            item_schema = as_schema_slot(&items_schema[idx])?;
        } else if matches!(schema_config.additional_items, Some(Value::Bool(false))) {
            // additionalItems: false — extra tuple items are dropped.
            drop_item = true;
            item_schema = None;
        } else if let Some(dict @ Value::Object(_)) = &schema_config.additional_items {
            item_schema = Some(dict.clone());
        } else {
            item_schema = Some(Value::Bool(true));
        }
    } else if let Some(dict @ Value::Object(_)) = &schema_config.items_schema {
        // The single-schema form: every item follows it.
        item_schema = Some(dict.clone());
    } else {
        // No usable items schema (absent, bool, or any other spelling):
        // upstream's `True`.
        item_schema = Some(Value::Bool(true));
    }

    Ok((item_schema, drop_item))
}

impl Parser {
    /// parse_array.py's `parse_array`: the array main loop.
    /// `<array> ::= '[' [ <json> *(', ' <json>) ] ']'` — a sequence of JSON
    /// values separated by commas. `closing_delimiter` generalizes the
    /// closer (`)` for the parenthesized form).
    pub(crate) fn parse_array(
        &mut self,
        schema: Option<&Value>,
        path: &str,
        closing_delimiter: char,
    ) -> Result<Value, String> {
        let (repairer, _schema, schema_config) =
            resolve_parser_array_schema(self.schema_repairer.as_ref(), schema)?;
        // The resolver's repairer reference borrows self only long enough to
        // answer "is schema-guided array parsing active"; every later
        // schema-layer access goes back through with_repairer.
        let repairer_active = repairer.is_some();
        let salvage_mode = repairer_active
            && self
                .schema_repairer
                .as_ref()
                .is_some_and(|r| r.is_salvage());
        let array_is_object_value = self.ctx_current() == Some(Ctx::ObjectValue);
        // `with self.context.enter(ARRAY)`: parse_array_items is the whole
        // with-region; the pop below runs on EVERY exit path (the `?`
        // included), exactly like the context manager's __exit__.
        self.ctx_push(Ctx::Array);
        let arr = self.parse_array_items(
            repairer_active,
            salvage_mode,
            array_is_object_value,
            schema_config.as_ref(),
            path,
            closing_delimiter,
        );
        self.ctx_pop();
        Ok(Value::Array(arr?))
    }

    /// The with-region of parse_array.py's `parse_array`: the item loop
    /// from the leading `skip_whitespaces` through the closing-delimiter
    /// consumption.
    fn parse_array_items(
        &mut self,
        repairer_active: bool,
        salvage_mode: bool,
        array_is_object_value: bool,
        schema_config: Option<&ArraySchemaConfig>,
        path: &str,
        closing_delimiter: char,
    ) -> Result<Vec<Value>, String> {
        let mut arr: Vec<Value> = Vec::new();
        let mut closed_before_parent_member = false;

        self.skip_whitespaces();
        let mut ch = self.cur();
        let mut idx = 0usize;
        while ch.is_some() && ch != Some(closing_delimiter) && ch != Some('}') {
            let (item_schema, drop_item) = resolve_array_item_schema(schema_config, idx)?;
            let item_path = format!("{path}[{idx}]");
            // The active gate: the schema layer engages only when the
            // resolver engaged it AND this item is not being dropped AND
            // the repairer is not salvaging.
            let active = repairer_active && !drop_item && !salvage_mode;
            // Python's None schema slot resolves to `true` at the
            // repair_value boundary (resolve_schema(None) is True): one
            // shared no-constraints spelling.
            let no_constraints = Value::Bool(true);
            let schema_arg = item_schema.as_ref().unwrap_or(&no_constraints);

            let value = if let Some(delimiter) = ch
                && STRING_DELIMITERS.contains(&delimiter)
            {
                // A string followed by ':' is often a missing object start;
                // treat it as an object.
                let i = self.skip_to_character(&[delimiter], 1);
                let i = self.scroll_whitespaces(i + 1);
                if self.get(i as isize) == Some(':') {
                    if array_is_object_value
                        && !arr.is_empty()
                        && arr
                            .iter()
                            .all(|item| !matches!(item, Value::Object(_) | Value::Array(_)))
                    {
                        // The parent object member resumes at this string:
                        // close the array before it (after scalar items).
                        closed_before_parent_member = true;
                        break;
                    }
                    if active {
                        // Schema-guided object parsing, then enforce the
                        // schema on the parsed object.
                        let parsed = self.parse_object(item_schema.as_ref(), &item_path)?;
                        with_repairer(self, |repairer| match repairer {
                            Some(repairer) => repairer.repair_value(parsed, schema_arg, &item_path),
                            // Unreachable from the active gate (the resolver
                            // pairs an active repairer with
                            // self.schema_repairer present, and with_repairer
                            // always puts it back): the value passes through
                            // unchanged, exactly the branch the gate would
                            // have taken.
                            None => Ok(parsed),
                        })?
                    } else {
                        // No schema (or dropping): still parse to keep the
                        // cursor in sync.
                        self.parse_object(None, "$")?
                    }
                } else {
                    let parsed = self.parse_string()?;
                    if active {
                        // Apply schema constraints/coercions to scalar
                        // values when configured.
                        with_repairer(self, |repairer| match repairer {
                            Some(repairer) => repairer.repair_value(parsed, schema_arg, &item_path),
                            // Unreachable from the active gate; passes
                            // through unchanged.
                            None => Ok(parsed),
                        })?
                    } else {
                        parsed
                    }
                }
            } else {
                // Use schema-aware parsing to guide nested repairs when
                // configured.
                if active {
                    self.parse_json(item_schema.as_ref(), &item_path, true, false)?
                } else {
                    self.parse_json(None, "$", true, false)?
                }
            };

            let cur = self.cur();
            if is_strictly_empty(&value)
                && !matches!(cur, Some(c) if c == closing_delimiter || c == ',')
            {
                // An empty container item away from any separator: skip one
                // char so the scan cannot stall on it.
                self.index += 1;
            } else if matches!(&value, Value::Str(text) if text == "...")
                && self.get(-1) == Some('.')
            {
                // Upstream logs the stray '...' and ignores it.
            } else if !drop_item {
                arr.push(value);
            } else if repairer_active {
                // Upstream logs the dropped extra item; the §6.4 diagnostic
                // vocabulary maps it to "drop_item".
                with_repairer(self, |repairer| {
                    if let Some(repairer) = repairer {
                        repairer.record(
                            "drop_item",
                            &item_path,
                            "Dropped extra array item not allowed by the schema",
                            Some(value),
                            None,
                            None,
                        );
                    }
                });
            }

            idx += 1;
            ch = self.cur();
            while let Some(c) = ch
                && c != closing_delimiter
                && (is_py_whitespace(c) || c == ',')
            {
                self.index += 1;
                ch = self.cur();
            }
        }

        if ch != Some(closing_delimiter) {
            // Upstream logs the missed closing delimiter and ignores it.
        }
        if !closed_before_parent_member {
            // Consume the closer (or step past the end when it never came).
            self.index += 1;
        }

        Ok(arr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(raw: &str) -> Value {
        Parser::new(raw, false, None)
            .parse()
            .expect("plain parse should not fail")
    }

    fn parse_strict(raw: &str) -> Result<Value, String> {
        Parser::new(raw, true, None).parse()
    }

    fn s(text: &str) -> Value {
        Value::Str(text.to_string())
    }

    fn obj(entries: &[(&str, Value)]) -> Value {
        Value::Object(
            entries
                .iter()
                .map(|(key, value)| (key.to_string(), value.clone()))
                .collect(),
        )
    }

    fn a(items: Vec<Value>) -> Value {
        Value::Array(items)
    }

    fn i(n: i64) -> Value {
        Value::Int(n)
    }

    #[test]
    fn parse_array() {
        // test_parse_array.py::test_parse_array
        assert_eq!(parse_ok("[]"), a(vec![]));
        assert_eq!(parse_ok("[1, 2, 3, 4]"), a(vec![i(1), i(2), i(3), i(4)]));
        assert_eq!(parse_ok("["), a(vec![]));
        assert_eq!(parse_ok("[[1\n\n]"), a(vec![a(vec![i(1)])]));
    }

    #[test]
    fn parse_array_edge_cases() {
        // test_parse_array.py::test_parse_array_edge_cases (every
        // assertion)
        assert_eq!(parse_ok("[{]"), a(vec![]));
        assert_eq!(parse_ok("["), a(vec![]));
        assert_eq!(parse_ok("[\""), a(vec![]));
        assert_eq!(parse_ok("]"), s(""));
        assert_eq!(parse_ok("[1, 2, 3,"), a(vec![i(1), i(2), i(3)]));
        assert_eq!(parse_ok("[1, 2, 3, ...]"), a(vec![i(1), i(2), i(3)]));
        assert_eq!(parse_ok("[1, 2, ... , 3]"), a(vec![i(1), i(2), i(3)]));
        assert_eq!(
            parse_ok("[1, 2, '...', 3]"),
            a(vec![i(1), i(2), s("..."), i(3)])
        );
        assert_eq!(
            parse_ok("[true, false, null, ...]"),
            a(vec![Value::Bool(true), Value::Bool(false), Value::Null])
        );
        assert_eq!(
            parse_ok(r#"["a" "b" "c" 1"#),
            a(vec![s("a"), s("b"), s("c"), i(1)])
        );
        assert_eq!(
            parse_ok(r#"{"employees":["John", "Anna","#),
            obj(&[("employees", a(vec![s("John"), s("Anna")]))])
        );
        assert_eq!(
            parse_ok(r#"{"employees":["John", "Anna", "Peter"#),
            obj(&[("employees", a(vec![s("John"), s("Anna"), s("Peter")]))])
        );
        assert_eq!(
            parse_ok(r#"{"key1": {"key2": [1, 2, 3"#),
            obj(&[("key1", obj(&[("key2", a(vec![i(1), i(2), i(3)]))]),)])
        );
        assert_eq!(
            parse_ok(r#"{"key": ["value]}"#),
            obj(&[("key", a(vec![s("value")]))])
        );
        assert_eq!(
            parse_ok(r#"["lorem "ipsum" sic"]"#),
            a(vec![s("lorem \"ipsum\" sic")])
        );
        assert_eq!(
            parse_ok(r#"{"key1": ["value1", "value2"}, "key2": ["value3", "value4"]}"#),
            obj(&[
                ("key1", a(vec![s("value1"), s("value2")])),
                ("key2", a(vec![s("value3"), s("value4")])),
            ])
        );
        // the headers/rows regrouping: trailing loose row values fold into
        // rows of the shared width
        assert_eq!(
            parse_ok(
                r#"{"headers": ["A", "B", "C"], "rows": [["r1a", "r1b", "r1c"], ["r2a", "r2b", "r2c"], "r3a", "r3b", "r3c"], ["r4a", "r4b", "r4c"], ["r5a", "r5b", "r5c"]]}"#
            ),
            obj(&[
                ("headers", a(vec![s("A"), s("B"), s("C")])),
                (
                    "rows",
                    a(vec![
                        a(vec![s("r1a"), s("r1b"), s("r1c")]),
                        a(vec![s("r2a"), s("r2b"), s("r2c")]),
                        a(vec![s("r3a"), s("r3b"), s("r3c")]),
                        a(vec![s("r4a"), s("r4b"), s("r4c")]),
                        a(vec![s("r5a"), s("r5b"), s("r5c")]),
                    ])
                ),
            ])
        );
        assert_eq!(
            parse_ok(r#"{"key": ["value" "value1" "value2"]}"#),
            obj(&[("key", a(vec![s("value"), s("value1"), s("value2")]))])
        );
        assert_eq!(
            parse_ok(
                r#"{"key": ["lorem "ipsum" dolor "sit" amet, "consectetur" ", "lorem "ipsum" dolor", "lorem"]}"#
            ),
            obj(&[(
                "key",
                a(vec![
                    s("lorem \"ipsum\" dolor \"sit\" amet, \"consectetur\" "),
                    s("lorem \"ipsum\" dolor"),
                    s("lorem"),
                ])
            )])
        );
        assert_eq!(
            parse_ok(r#"{"k"e"y": "value"}"#),
            obj(&[("k\"e\"y", s("value"))])
        );
        assert_eq!(
            parse_ok(r#"["key":"value"}]"#),
            a(vec![obj(&[("key", s("value"))])])
        );
        assert_eq!(
            parse_ok(r#"["key":"value"]"#),
            a(vec![obj(&[("key", s("value"))])])
        );
        assert_eq!(
            parse_ok(r#"[ "key":"value"]"#),
            a(vec![obj(&[("key", s("value"))])])
        );
        // the object-in-array duplicate split leaves the tail for the next
        // item
        assert_eq!(
            parse_ok(r#"[{"key": "value", "key"#),
            a(vec![obj(&[("key", s("value"))]), a(vec![s("key")])])
        );
        assert_eq!(parse_ok("{'key1', 'key2'}"), a(vec![s("key1"), s("key2")]));
    }

    #[test]
    fn parse_array_closes_before_object_member_after_scalar_items() {
        // test_parse_array.py::test_parse_array_closes_before_object_member_after_scalar_items
        assert_eq!(
            parse_ok(r#"{"outer": ["a", "b", "next": "value"}"#),
            obj(&[("outer", a(vec![s("a"), s("b")])), ("next", s("value")),])
        );
    }

    #[test]
    fn parse_array_contextually_closes_in_strict_mode() {
        // test_parse_array.py::test_parse_array_contextually_closes_in_strict_mode:
        // the contextual close must NOT raise in strict mode
        assert_eq!(
            parse_strict(r#"{"outer": ["a", "b", "next": "value"}"#),
            Ok(obj(&[
                ("outer", a(vec![s("a"), s("b")])),
                ("next", s("value")),
            ]))
        );
    }

    #[test]
    fn parse_array_mismatched_parenthesis_still_missing_bracket() {
        // test_parse_array.py::test_parse_array_mismatched_parenthesis_still_logs_missing_bracket
        // (log-text assertion skipped: logs are not ported)
        assert_eq!(parse_ok("[1, 2)"), a(vec![i(1), i(2)]));
    }

    #[test]
    fn parse_array_missing_quotes() {
        // test_parse_array.py::test_parse_array_missing_quotes
        assert_eq!(
            parse_ok(r#"["value1" value2", "value3"]"#),
            a(vec![s("value1"), s("value2"), s("value3")])
        );
        assert_eq!(
            parse_ok(
                r#"{"bad_one":["Lorem Ipsum", "consectetur" comment" ], "good_one":[ "elit", "sed", "tempor"]}"#
            ),
            obj(&[
                (
                    "bad_one",
                    a(vec![s("Lorem Ipsum"), s("consectetur"), s("comment")])
                ),
                ("good_one", a(vec![s("elit"), s("sed"), s("tempor")])),
            ])
        );
        assert_eq!(
            parse_ok(
                r#"{"bad_one": ["Lorem Ipsum","consectetur" comment],"good_one": ["elit","sed","tempor"]}"#
            ),
            obj(&[
                (
                    "bad_one",
                    a(vec![s("Lorem Ipsum"), s("consectetur"), s("comment")])
                ),
                ("good_one", a(vec![s("elit"), s("sed"), s("tempor")])),
            ])
        );
    }
}
