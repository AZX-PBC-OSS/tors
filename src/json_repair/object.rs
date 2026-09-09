//! The object-repair parser: a port of json_repair's `parse_object.py`
//! (upstream: https://github.com/mangiucugna/json_repair by Stefano
//! Baccianella, MIT, pinned at commit
//! 251d141786d0f6ff561f6ec04d90188a338e2470, version 0.63.4): the object
//! main loop, the key/value sub-parsers, the duplicate-key split, the
//! empty-object classifier (object/array/salvage-set repairs), and the
//! schema-layer hooks (`parser_schema.py`'s object config,
//! `pattern_properties.py` matching, the required/default finalization).
//! Every branch is ported 1:1 in upstream's order of checks.
//!
//! # Porting notes (the decisions this file's dynamics forced)
//!
//! - `self.log(...)` sites upstream become rationale COMMENTS here (log
//!   texts are not ported; the heuristic each explained gets the comment),
//!   per the port contract in `parser.rs`'s docs. The parser-level
//!   `repairer._log(...)` sites (inserted default / dropped extra property)
//!   are wired to §6.4's diagnostic vocabulary ("insert_default" /
//!   "drop_property") through the take/put-back repairer helper.
//! - The repairer never stays borrowed across parser mutation: the resolver
//!   runs once per `parse_object` call to an ACTIVITY bit plus owned
//!   config/schema copies, and every later schema-layer access re-acquires
//!   the repairer through the take/put-back helper ([`with_repairer`], the
//!   same shape `parser.rs` pins for its own repairer calls).
//! - Python passes `None` schema slots into `repair_value`; the pinned Rust
//!   signature takes `&Value`, and upstream's `resolve_schema(None) is True`
//!   (no constraints) makes `Value::Bool(true)` the None spelling at that
//!   boundary. JSON `null` in a schema slot IS Python's `None` and maps to
//!   `Option::None` everywhere a `dict|bool|None` slot flows onward.
//! - `_copy_json_value` (deep copy + non-JSON raise) is `Value::clone()`
//!   here: the schema tree is already a `Value` — JSON-domain by
//!   construction — so the deep copy is trivial and the raise arms are
//!   structurally unreachable.
//! - `_finalize_object`'s missing-required message lists the keys in the
//!   schema's `required` order; upstream iterates a Python SET there (its
//!   order is hash-arbitrary, so there is no order to preserve).
//! - Upstream's `_parse_object_key` ASSERTS the key parse returned a str;
//!   in OBJECT_KEY context the only non-str direct results come from
//!   pathological comment/LLM-block re-entries, where upstream CRASHES with
//!   AssertionError. tors must stay total (the fuzz/hypothesis gates), so a
//!   non-str result maps to the empty key and the scan continues — the one
//!   deliberate behavior divergence in this file, unreachable from any
//!   corpus input.
//! - Python `try/finally` context regions port to explicit
//!   `ctx_push`/`ctx_pop` pairs with the pop on every exit path; upstream's
//!   `with self.context.enter(...)` regions likewise bracket the recursive
//!   reparse calls below.
//! - Splice sites (upstream slice-assignment on `json_str`): the
//!   duplicate-key split INSERTS `{` at `index + 1`; the escaped-object
//!   repair REPLACES `[start_index - 1, index + 1)` (both ends clamped the
//!   way Python slicing clamps — the cursor can sit past the end of the
//!   input when these fire, and `Vec::splice` would panic on an unclamped
//!   end).
//!
//! # Test provenance
//!
//! The `#[cfg(test)]` batteries port the VALUE-level assertions of
//! upstream's `tests/test_parse_object.py` (every case, driven through
//! `Parser::parse`, which is what `repair_json(..., skip_json_loads=True,
//! return_objects=True)` runs) and the object/array strict raises of
//! `tests/test_strict_mode.py` with the exact catalog strings. The
//! assertions that intentionally live ONLY in the pytest corpus
//! (`tests/test_json_repair.py`, agent C's): the serialized-STRING forms of
//! these same inputs (`repair_json(...)` without `return_objects`, the
//! dumps parity `dumps.rs` owns), the `logging=True` log-text assertions,
//! and every case whose behavior belongs to `string.rs`/`parser.rs`'s own
//! batteries.

use super::parser::{Ctx, Parser};
use super::{ObjectBuilder, STRING_DELIMITERS, Value};
use crate::json_schema_impl::{
    ObjectSchemaConfig, SchemaRepairer, match_pattern_properties, resolve_parser_object_schema,
};
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

/// parse_object.py's `_classify_empty_object_repair` return kinds.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EmptyObjectRepair {
    Keep,
    Object,
    SchemaSetObject,
    Array,
}

/// parse_object.py's `_finalize_object`: after a schema-guided object parse
/// closed, enforce `required` (standard mode raises; salvage defers) and
/// insert declared `default` values for absent optional properties.
fn finalize_object(
    mut obj: Value,
    repairer: Option<&SchemaRepairer>,
    schema_config: Option<&ObjectSchemaConfig>,
    path: &str,
) -> Result<Value, String> {
    let (repairer, schema_config) = match (repairer, schema_config) {
        (Some(repairer), Some(schema_config)) => (repairer, schema_config),
        _ => return Ok(obj),
    };

    let missing_required: Vec<&str> = schema_config
        .required
        .iter()
        .filter(|key| obj.object_get(key).is_none())
        .map(|key| key.as_str())
        .collect();
    if !missing_required.is_empty() && !repairer.is_salvage() {
        let joined = missing_required.join(", ");
        return Err(format!("Missing required properties at {path}: {joined}"));
    }

    for (key, prop_schema) in &schema_config.properties {
        if obj.object_get(key).is_some() || schema_config.required.iter().any(|r| r == key) {
            continue;
        }
        if let Value::Object(entries) = prop_schema
            && let Some((_, default)) = entries.iter().find(|(k, _)| k == "default")
        {
            // Upstream deep-copies the default via repairer._copy_json_value
            // (a Value tree is already JSON-domain, so clone is that copy
            // and the non-JSON raise arms cannot fire).
            obj.object_insert(key.to_string(), default.clone());
            repairer.record(
                "insert_default",
                &format!("{path}.{key}"),
                "Inserted default value for missing property",
                None,
                Some(default.clone()),
                None,
            );
        }
    }
    Ok(obj)
}

/// parse_object.py's `_strip_comments_for_empty_object_classification`: a
/// pure scan over the object BODY that drops `#`/`//`/`/*...*/` comments
/// while preserving quoted spans and backslash runs, for the empty-object
/// classifier's "is anything left?" probe.
fn strip_comments_for_empty_object_classification(body: &str) -> String {
    let body: Vec<char> = body.chars().collect();
    let mut stripped = String::new();
    let mut in_quote: Option<char> = None;
    let mut backslashes = 0usize;
    let mut index = 0usize;
    while index < body.len() {
        let ch = body[index];
        let next_char = body.get(index + 1).copied();

        if ch == '\\' {
            backslashes += 1;
            stripped.push(ch);
            index += 1;
            continue;
        }
        if let Some(quote) = in_quote {
            stripped.push(ch);
            if ch == quote && backslashes.is_multiple_of(2) {
                in_quote = None;
            }
            backslashes = 0;
            index += 1;
            continue;
        }
        if STRING_DELIMITERS.contains(&ch) && backslashes.is_multiple_of(2) {
            in_quote = Some(ch);
            stripped.push(ch);
            backslashes = 0;
            index += 1;
            continue;
        }
        backslashes = 0;

        if ch == '#' || (ch == '/' && next_char == Some('/')) {
            index += if ch == '/' { 2 } else { 1 };
            while index < body.len() && !matches!(body[index], '\n' | '\r') {
                index += 1;
            }
            continue;
        }
        if ch == '/' && next_char == Some('*') {
            index += 2;
            while index + 1 < body.len() && !(body[index] == '*' && body[index + 1] == '/') {
                index += 1;
            }
            index = (index + 2).min(body.len());
            continue;
        }

        stripped.push(ch);
        index += 1;
    }
    stripped
}

/// Python's `schema_value is not None and not isinstance(schema_value,
/// (dict, bool))` raise shape, over the Value domain: a per-key schema slot
/// must be null (Python `None`), bool, or dict.
fn as_schema_slot(value: &Value) -> Result<Option<Value>, String> {
    match value {
        // JSON null is Python's None spelling for "no schema here".
        Value::Null => Ok(None),
        Value::Bool(_) | Value::Object(_) => Ok(Some(value.clone())),
        _ => Err("Schema must be an object.".into()),
    }
}

/// parse_object.py's `_resolve_object_property_schema` return: the guiding
/// schema (None = Python's null slot), the extra patternProperties schemas,
/// and the additionalProperties:false drop flag.
type PropertySchemaResolution = (Option<Value>, Vec<Option<Value>>, bool);

/// The row-width summary of an array's inner arrays: the incremental
/// spelling of parse_object.py's whole-array `list_lengths` rescan.
/// `Uniform(w)` answers upstream's "all inner arrays share one width"
/// directly; `None` is the no-inner-arrays case (an empty `list_lengths`);
/// `Mixed` is differing widths. Folding only what a merge appends is
/// equivalent to rescanning because within one key-scan loop the merge is
/// the previous array's only mutator — see
/// [`Parser::merge_object_array_continuation`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MergeWidths {
    /// No inner arrays seen.
    None,
    /// Every inner array so far has this width (zero included — Python's
    /// falsy zero width reaches the no-width branch at USE, not here).
    Uniform(usize),
    /// Inner arrays of differing widths seen.
    Mixed,
}

impl MergeWidths {
    /// The full scan, exactly upstream's `list_lengths` collection: fold
    /// every inner array's width, in order.
    fn scan<'a>(items: impl IntoIterator<Item = &'a Value>) -> MergeWidths {
        let mut summary = MergeWidths::None;
        summary.fold_over(items);
        summary
    }

    /// Fold every inner-array width of `items` into this summary, in order.
    fn fold_over<'a>(&mut self, items: impl IntoIterator<Item = &'a Value>) {
        for item in items {
            if let Value::Array(inner) = item {
                self.fold(inner.len());
            }
        }
    }

    /// Fold one inner-array width into the summary.
    fn fold(&mut self, width: usize) {
        *self = match (*self, width) {
            (MergeWidths::None, _) => MergeWidths::Uniform(width),
            (MergeWidths::Uniform(existing), _) if existing == width => *self,
            (MergeWidths::Uniform(_), _) | (MergeWidths::Mixed, _) => MergeWidths::Mixed,
        };
    }
}

impl Parser {
    /// parse_object.py's `parse_object`: the object main loop.
    /// `<object> ::= '{' [ <member> *(', ' <member>) ] '}'` — a sequence of
    /// members, repaired member by member.
    pub(crate) fn parse_object(
        &mut self,
        schema: Option<&Value>,
        path: &str,
    ) -> Result<Value, String> {
        // The builder's side index keeps member insertion O(1) (a linear
        // object_insert scan is O(n²) on large objects), and the seen set
        // carries the duplicate-key gate the same way.
        let mut obj = ObjectBuilder::new();
        let mut seen_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
        let start_index = self.index;
        let parsing_object_value = self.ctx_current() == Some(Ctx::ObjectValue);
        let (repairer, schema, schema_config) =
            resolve_parser_object_schema(self.schema_repairer.as_ref(), schema)?;
        // The resolver's repairer reference borrows self only long enough to
        // answer "is schema-guided object parsing active"; every later
        // schema-layer access goes back through with_repairer.
        let repairer_active = repairer.is_some();

        while self.cur().unwrap_or('}') != '}' {
            self.skip_whitespaces();

            if self.cur() == Some(':') {
                // Upstream logs a ':' before any key and ignores it.
                self.index += 1;
            }

            let (key, rollback_index) = self.parse_object_key(&mut obj)?;
            if self.ctx_has(Ctx::Array) && seen_keys.contains(&key) {
                if self.strict {
                    // Upstream logs the duplicate key found in strict mode
                    // before raising.
                    return Err("Duplicate key found in strict mode while parsing object.".into());
                }
                if !parsing_object_value && self.should_split_duplicate_object(rollback_index) {
                    // A duplicate key that does not look like a plain
                    // comma-separated repeat: close the object here and
                    // roll the cursor back (the split below).
                    self.split_object_on_duplicate_key(rollback_index);
                    break;
                }
                // Not splitting (an object-value context, or a duplicate
                // key with a normal comma separator): upstream logs keeping
                // the duplicate-key overwrite behavior.
            }

            self.skip_whitespaces();
            if self.cur().unwrap_or('}') == '}' {
                continue;
            }

            self.skip_whitespaces();
            if self.cur() != Some(':') && self.strict {
                // Upstream logs the missing ':' in strict mode before
                // raising.
                return Err("Missing ':' after key in strict mode while parsing object.".into());
            }
            // A missing ':' outside strict mode: upstream logs the missed
            // ':' after a key and carries on.

            self.index += 1;
            let (prop_schema, extra_schemas, drop_property) =
                self.resolve_object_property_schema(repairer_active, schema_config.as_ref(), &key)?;
            let key_path = format!("{path}.{key}");
            let mut value =
                self.parse_object_value(repairer_active, prop_schema.as_ref(), &key_path)?;

            if repairer_active {
                // Python passes each extra schema (possibly None) into
                // repair_value; the pinned &Value signature takes the
                // no-constraints resolution Bool(true) as None's spelling.
                let no_constraints = Value::Bool(true);
                for extra_schema in &extra_schemas {
                    let schema_arg = extra_schema.as_ref().unwrap_or(&no_constraints);
                    value = with_repairer(self, |repairer| match repairer {
                        Some(repairer) => repairer.repair_value(value, schema_arg, &key_path),
                        // Unreachable from the active gate (the resolver
                        // pairs an active repairer with self.schema_repairer
                        // present, and with_repairer always puts it back):
                        // the value passes through unchanged, exactly the
                        // branch the gate would have taken.
                        None => Ok(value),
                    })?;
                }
            }

            if !repairer_active
                && matches!(&value, Value::Str(text) if text.is_empty())
                && self.strict
                && !self.get(-1).is_some_and(|c| STRING_DELIMITERS.contains(&c))
            {
                // Upstream logs the empty parsed value in strict mode before
                // raising.
                return Err("Parsed value is empty in strict mode while parsing object.".into());
            }

            if !repairer_active || !drop_property {
                seen_keys.insert(key.clone());
                obj.insert(key, value);
            } else {
                // Upstream logs the dropped extra property; the §6.4
                // diagnostic vocabulary maps it to "drop_property".
                with_repairer(self, |repairer| {
                    if let Some(repairer) = repairer {
                        repairer.record(
                            "drop_property",
                            &key_path,
                            "Dropped extra property not allowed by the schema",
                            Some(value),
                            None,
                            None,
                        );
                    }
                });
            }

            if matches!(self.cur(), Some(',') | Some('\'') | Some('"')) {
                self.index += 1;
            }
            if self.cur() == Some(']') && self.ctx_has(Ctx::Array) {
                // A closing array bracket while an array encloses this
                // object: close the object here and leave the ']' for the
                // array (the -1/+1 pair below nets to "not consumed").
                self.index -= 1;
                break;
            }
            self.skip_whitespaces();
        }

        self.index += 1;

        let (repaired_empty_object, repaired_value) = self.repair_empty_object_result(
            &obj,
            start_index,
            schema.as_ref(),
            path,
            repairer_active,
        )?;
        if repaired_empty_object {
            // Every repaired branch carries a value; the keep branches
            // never set the flag (the fallback is inert).
            return Ok(repaired_value.unwrap_or(Value::Str(String::new())));
        }

        self.complete_object_parse(
            obj,
            schema.as_ref(),
            path,
            repairer_active,
            schema_config.as_ref(),
        )
    }

    /// parse_object.py's `_parse_object_key`: scan one object key (with the
    /// array-continuation merge hook), returning `(key, rollback_index)`.
    /// The OBJECT_KEY context brackets the whole scan, popping on every
    /// exit path (Python's try/finally).
    fn parse_object_key(&mut self, obj: &mut ObjectBuilder) -> Result<(String, usize), String> {
        let mut key = String::new();
        let mut rollback_index = self.index;
        self.ctx_push(Ctx::ObjectKey);
        // Python's try/finally: the pop below runs on every exit path, so
        // the loop communicates through key/rollback_index and the Result
        // carried out of `loop`.
        // The merge-continuation run's row-width summary: initialized by
        // the first merge's full scan, folded forward by every later merge
        // in THIS loop (the loop breaks as soon as a key parses, so a
        // later run rescans — never a stale summary).
        let mut merge_widths: Option<MergeWidths> = None;
        let outcome: Result<(), String> = loop {
            if self.cur().is_none() {
                break Ok(());
            }
            rollback_index = self.index;
            if self.cur() == Some('[') && key.is_empty() {
                match self.merge_object_array_continuation(obj, &mut merge_widths) {
                    Ok(true) => continue,
                    Ok(false) => {}
                    Err(err) => break Err(err),
                }
            }

            let raw_key = match self.parse_string() {
                Ok(value) => value,
                Err(err) => break Err(err),
            };
            // Upstream asserts the key parse returned a str; in OBJECT_KEY
            // context the only non-str direct results come from
            // pathological comment/LLM-block re-entries, where upstream
            // crashes on the assert. tors stays total: the empty key stands
            // in and the scan continues (the module docs' one deliberate
            // divergence).
            key = match raw_key {
                Value::Str(text) => text,
                _ => String::new(),
            };
            if key.is_empty() {
                self.skip_whitespaces();
            }
            if !key.is_empty() || matches!(self.cur(), Some(':') | Some('}')) {
                if key.is_empty() && self.strict {
                    // Upstream logs the empty key found in strict mode
                    // before raising.
                    break Err("Empty key found in strict mode while parsing object.".into());
                }
                break Ok(());
            }
        };
        self.ctx_pop();
        outcome?;
        Ok((key, rollback_index))
    }

    /// parse_object.py's `_merge_object_array_continuation`: a '[' at the
    /// key position continues the PREVIOUS member's array value (rows
    /// regrouped when the existing rows share one width); returns whether
    /// the continuation was taken.
    ///
    /// `widths` is the row-width summary of the previous member's array —
    /// the incremental spelling of upstream's whole-array `list_lengths`
    /// rescan. The summary lives in parse_object_key's key-scan loop for
    /// exactly one sequential-merge run: the first merge folds the whole
    /// previous array (upstream's scan, once), every later merge folds
    /// only what THIS merge appends, because within the key-scan loop the
    /// merge is the previous array's only mutator (a parsed key breaks the
    /// loop and the next run starts a fresh summary). Rescanning per merge
    /// made same-level continuation runs — `{"a":[1], [2], [3], ...` —
    /// O(M²) in the merge count (~5s at 200k merges); the fold keeps the
    /// same outputs at O(total input).
    fn merge_object_array_continuation(
        &mut self,
        obj: &mut ObjectBuilder,
        widths: &mut Option<MergeWidths>,
    ) -> Result<bool, String> {
        let (prev_key, prev_is_list) = match obj.last_mut() {
            Some((key, value)) => (key.clone(), matches!(value, Value::Array(_))),
            None => return Ok(false),
        };
        // Python: `if not prev_key or not isinstance(obj[prev_key], list) or
        // self.strict` — an empty-string key is falsy there.
        if prev_key.is_empty() || !prev_is_list || self.strict {
            return Ok(false);
        }

        self.index += 1;
        // Depth-guard the recursion. This continuation calls parse_array,
        // whose first item is often a string followed by ':' — a missing
        // object start parsed by parse_object directly — and that object's
        // key scan can take another '[' continuation, so a nested chain
        // (`{"a":[0],` followed by `["b":[0],` repeated) grows the native
        // stack one frame pair per fragment with no cap anywhere on the
        // cycle — an uncatchable SIGSEGV, not the documented catchable
        // ValueError. enter_depth caps it at MAX_NESTING like every other
        // deep-recursion path (the same enter/parse/leave/`?` shape
        // parse_json's `{`/`[`/`(` branches and complete_object_parse's
        // comma-merge use); balanced on every non-abort path, so same-level
        // sequential merges (`{"a":[1], [2], [3]}`) never accrue depth.
        self.enter_depth()?;
        let new_array = self.parse_array(None, "$", ']');
        self.leave_depth();
        let new_array = new_array?;
        if let Value::Array(new_items) = new_array
            && let Some((_, prev_value)) = obj.last_mut()
            && let Value::Array(prev_items) = prev_value
        {
            // The first merge of a run folds the whole previous array
            // (upstream's list_lengths scan, verbatim); later merges find
            // the summary already current.
            let summary = widths.get_or_insert_with(|| MergeWidths::scan(prev_items.iter()));
            // Upstream's expected_len: Some(width) iff the array HAS inner
            // arrays and they all share one width; Python's truthiness
            // then drops a zero shared width to the no-width branch.
            let expected_len = match *summary {
                MergeWidths::Uniform(width) if width != 0 => Some(width),
                MergeWidths::Uniform(_) | MergeWidths::Mixed | MergeWidths::None => None,
            };
            if let Some(expected_len) = expected_len {
                let mut tail: Vec<Value> = Vec::new();
                while !matches!(prev_items.last(), Some(Value::Array(_))) {
                    match prev_items.pop() {
                        Some(item) => tail.push(item),
                        None => break,
                    }
                }
                if !tail.is_empty() {
                    tail.reverse();
                    if tail.len().is_multiple_of(expected_len) {
                        // Row values found without an inner array:
                        // group them into rows of the shared width.
                        for chunk in tail.chunks(expected_len) {
                            prev_items.push(Value::Array(chunk.to_vec()));
                        }
                        // Regrouped rows carry the summary's own width.
                        summary.fold(expected_len);
                    } else {
                        prev_items.extend(tail);
                        // The popped tail was non-arrays by construction:
                        // no widths to fold.
                    }
                }
                if !new_items.is_empty() {
                    if new_items.iter().all(|item| matches!(item, Value::Array(_))) {
                        // Additional rows: append them without
                        // flattening.
                        summary.fold_over(new_items.iter());
                        prev_items.extend(new_items);
                    } else {
                        summary.fold(new_items.len());
                        prev_items.push(Value::Array(new_items));
                    }
                }
            } else {
                // No shared row width to regroup around: a lone list
                // item flattens into the previous value; anything else
                // extends as-is. Either way the appended items keep
                // their own shapes, so their array widths fold in.
                match new_items.as_slice() {
                    [Value::Array(inner)] => {
                        summary.fold_over(inner.iter());
                        prev_items.extend(inner.iter().cloned());
                    }
                    _ => {
                        summary.fold_over(new_items.iter());
                        prev_items.extend(new_items);
                    }
                }
            }
        }

        self.skip_whitespaces();
        if self.cur() == Some(',') {
            self.index += 1;
        }
        self.skip_whitespaces();
        Ok(true)
    }

    /// parse_object.py's `_should_split_duplicate_object`: a duplicate key
    /// splits the object UNLESS it looks like a plain comma-separated
    /// repeat (quoted key, comma before it, colon after it).
    fn should_split_duplicate_object(&self, rollback_index: usize) -> bool {
        let mut lookback_idx: isize = rollback_index as isize - self.index as isize - 1;
        let mut prev_non_whitespace = self.get(lookback_idx);
        while prev_non_whitespace.is_some_and(is_py_whitespace) {
            lookback_idx -= 1;
            prev_non_whitespace = self.get(lookback_idx);
        }
        let key_start_char = self.get(rollback_index as isize - self.index as isize);
        let next_non_whitespace = self.get(self.scroll_whitespaces(0) as isize);
        !(key_start_char.is_some_and(|c| STRING_DELIMITERS.contains(&c))
            && prev_non_whitespace == Some(',')
            && next_non_whitespace == Some(':'))
    }

    /// parse_object.py's `_split_object_on_duplicate_key` — THE SPLICE:
    /// rewind onto the key's opening and insert a `{` there, so the parent
    /// container re-parses the tail as a fresh object. The insert shifts
    /// every absolute position at/after it, so the parser-level lookahead
    /// memo (pure buffer facts keyed by absolute positions) is cleared
    /// here — the only buffer-mutating site.
    fn split_object_on_duplicate_key(&mut self, rollback_index: usize) {
        self.index = rollback_index - 1;
        // Python's json_str[:index+1] + "{" + json_str[index+1:] — an insert
        // at index + 1.
        self.s.insert(self.index + 1, '{');
        // An O(n) buffer splice: the next deadline check must read the
        // clock, keeping the splice-rescan bound tight. It also shifts
        // every absolute position at/after it, so the parser-level
        // lookahead memo (pure buffer facts keyed by absolute positions)
        // is cleared here — the only buffer-mutating site.
        self.force_deadline_check();
        self.lookahead_cache.clear();
    }

    /// parse_object.py's `_resolve_object_property_schema`: pick the schema
    /// guiding one member's value — the declared property, the first
    /// patternProperties match (extras carried alongside), the
    /// additionalProperties dict, or `true`; plus the drop flag for
    /// `additionalProperties: false`.
    fn resolve_object_property_schema(
        &self,
        repairer_active: bool,
        schema_config: Option<&ObjectSchemaConfig>,
        key: &str,
    ) -> Result<PropertySchemaResolution, String> {
        let Some(schema_config) = schema_config else {
            return Ok((None, Vec::new(), false));
        };
        if !repairer_active {
            return Ok((None, Vec::new(), false));
        }

        if let Some((_, schema_value)) = schema_config.properties.iter().find(|(k, _)| k == key) {
            let prop_schema = as_schema_slot(schema_value)?;
            return Ok((prop_schema, Vec::new(), false));
        }

        let mut matched: Vec<Value> = Vec::new();
        // The unsupported-regex patterns the matcher reports are logged one
        // by one upstream and skipped; tors keeps the skip (the log site is
        // a comment per the port contract).
        let mut _unsupported_patterns: Vec<String> = Vec::new();
        if let Some(pattern_properties) = &schema_config.pattern_properties
            && pattern_properties.is_truthy()
        {
            (matched, _unsupported_patterns) = match_pattern_properties(pattern_properties, key);
        }
        if !matched.is_empty() {
            let prop_schema = as_schema_slot(&matched[0])?;
            let mut extra_schemas: Vec<Option<Value>> = Vec::new();
            for extra_schema in &matched[1..] {
                extra_schemas.push(as_schema_slot(extra_schema)?);
            }
            return Ok((prop_schema, extra_schemas, false));
        }

        match &schema_config.additional_properties {
            // additionalProperties: false — this property is not allowed.
            Some(Value::Bool(false)) => Ok((None, Vec::new(), true)),
            Some(dict @ Value::Object(_)) => Ok((Some(dict.clone()), Vec::new(), false)),
            // Absent (or any non-dict, non-false spelling): anything goes.
            _ => Ok((Some(Value::Bool(true)), Vec::new(), false)),
        }
    }

    /// parse_object.py's `_parse_object_value`: parse one member's value in
    /// the OBJECT_VALUE context (bracketed with the pop on every exit path,
    /// Python's try/finally), routing through the schema layer when active.
    fn parse_object_value(
        &mut self,
        repairer_active: bool,
        prop_schema: Option<&Value>,
        key_path: &str,
    ) -> Result<Value, String> {
        self.ctx_push(Ctx::ObjectValue);
        self.skip_whitespaces();
        let ch = self.cur();
        let result: Result<Value, String> = if matches!(ch, Some(',') | Some('}')) {
            // Upstream logs the stray separator standing in for a value.
            if repairer_active {
                // Python passes prop_schema (possibly None) into
                // repair_value; the pinned &Value signature takes the
                // no-constraints resolution Bool(true) as None's spelling.
                let no_constraints = Value::Bool(true);
                let schema_arg = prop_schema.unwrap_or(&no_constraints);
                with_repairer(self, |repairer| match repairer {
                    Some(repairer) => repairer.repair_value(Value::Missing, schema_arg, key_path),
                    // Unreachable from the active gate (see the loop's
                    // with_repairer note): the no-repairer branch's "".
                    None => Ok(Value::Str(String::new())),
                })
            } else {
                Ok(Value::Str(String::new()))
            }
        } else if repairer_active {
            self.parse_json(prop_schema, key_path, true, false)
        } else {
            self.parse_json(None, "$", true, false)
        };
        self.ctx_pop();
        result
    }

    /// parse_object.py's `_repair_empty_object_result`: an object that
    /// parsed empty over a non-trivial span gets a second chance — the
    /// escaped-key normalization reparse, the salvage set-as-object
    /// reparse, or the array fallback.
    fn repair_empty_object_result(
        &mut self,
        obj: &ObjectBuilder,
        start_index: usize,
        schema: Option<&Value>,
        path: &str,
        repairer_active: bool,
    ) -> Result<(bool, Option<Value>), String> {
        // isize arithmetic: the trailing skips can leave the cursor past the
        // end, and the span check must not underflow below start_index.
        // An object is truthy iff non-empty (CPython).
        if !obj.is_empty() || (self.index as isize - start_index as isize) <= 2 {
            return Ok((false, None));
        }

        if self.strict {
            // Upstream logs the empty object with extra characters in
            // strict mode before raising.
            return Err(
                "Parsed object is empty but contains extra characters in strict mode.".into(),
            );
        }

        let (empty_object_repair, normalized_object) =
            self.classify_empty_object_repair(start_index, schema, repairer_active);
        if empty_object_repair == EmptyObjectRepair::Object
            && let Some(normalized_object) = normalized_object
        {
            // THE SPLICE: replace the attempted span (the '{' at
            // start_index - 1 through the cursor) with the normalized
            // object text; Python's slice assignment clamps its end, and so
            // does the min() here (the cursor may sit past the end).
            let end_index = (self.index + 1).min(self.s.len());
            self.s
                .splice(start_index - 1..end_index, normalized_object.chars());
            // An O(n) buffer splice: the next deadline check must read the
            // clock, keeping the reparse bound tight.
            self.force_deadline_check();
            self.index = start_index;
            self.ctx_push(Ctx::ObjectKey);
            let repaired = self.parse_object(schema, path);
            self.ctx_pop();
            let repaired_value = repaired?;
            self.deferred_contexts.push(Ctx::ObjectKey);
            return Ok((true, Some(repaired_value)));
        }
        if empty_object_repair == EmptyObjectRepair::SchemaSetObject {
            // Salvage schema expects an object here: re-parse the set-like
            // members as object keys (null-valued when they are all
            // non-empty strings).
            self.index = start_index;
            self.ctx_push(Ctx::ObjectKey);
            let set_items = self.parse_array(None, "$", ']');
            self.ctx_pop();
            let set_items = set_items?;
            self.deferred_contexts.push(Ctx::ObjectKey);
            if let Value::Array(items) = &set_items {
                let key_candidates: Vec<&str> = items
                    .iter()
                    .filter_map(|item| match item {
                        Value::Str(text) if !text.is_empty() => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                if key_candidates.len() == items.len() {
                    // dict.fromkeys: the candidate keys in order, every
                    // value null, DUPLICATES COLLAPSED to the first
                    // occurrence (build through object_insert, which
                    // updates in place at first position).
                    let mut set_object = Value::Object(Vec::new());
                    for key in key_candidates {
                        set_object.object_insert(key.to_string(), Value::Null);
                    }
                    return Ok((true, Some(set_object)));
                }
            }
            return Ok((true, Some(set_items)));
        }
        if empty_object_repair == EmptyObjectRepair::Array {
            // Upstream logs the empty object and retries it as an array.
            self.index = start_index;
            self.ctx_push(Ctx::ObjectKey);
            let repaired_array = self.parse_array(None, "$", ']');
            self.ctx_pop();
            let repaired_array = repaired_array?;
            self.deferred_contexts.push(Ctx::ObjectKey);
            return Ok((true, Some(repaired_array)));
        }
        Ok((false, None))
    }

    /// parse_object.py's `_classify_empty_object_repair`: decide what the
    /// empty object's span actually contains — keep (object-shaped
    /// leftovers), normalize-and-reparse (escaped object keys), the salvage
    /// set-as-object, or the array fallback.
    fn classify_empty_object_repair(
        &self,
        start_index: usize,
        schema: Option<&Value>,
        repairer_active: bool,
    ) -> (EmptyObjectRepair, Option<String>) {
        // attempted_object spans the '{' at start_index - 1 through the
        // cursor; Python slicing clamps both ends (the cursor can sit past
        // the end), which the min() and get() reproduce.
        let end = (self.index + 1).min(self.s.len());
        // A span this wide is O(remaining) work per empty-object exit (the
        // empty-object quadratic's other half): force the next check.
        self.note_scan_distance(end - (start_index - 1));
        let attempted_object: String = self
            .s
            .get(start_index - 1..end)
            .map_or_else(String::new, |chars| chars.iter().collect());
        let mut body: String = attempted_object.chars().skip(1).collect();
        if body.ends_with('}') {
            // body.removesuffix("}")
            body.pop();
        }
        let body = body.trim_start_matches(is_py_whitespace);
        if body.is_empty() {
            return (EmptyObjectRepair::Keep, None);
        }
        if (body.starts_with("\\\"") && body.contains("\\\":"))
            || (body.starts_with("\\'") && body.contains("\\':"))
        {
            // Upstream logs the escaped-object-key normalization before
            // reparsing it as an object.
            let normalized_object = attempted_object.replace("\\\"", "\"").replace("\\'", "'");
            return (EmptyObjectRepair::Object, Some(normalized_object));
        }
        let stripped = strip_comments_for_empty_object_classification(body);
        let body = stripped.trim_start_matches(is_py_whitespace);
        if body.is_empty() {
            return (EmptyObjectRepair::Keep, None);
        }

        let mut in_quote: Option<char> = None;
        let mut backslashes = 0usize;
        for ch in body.chars() {
            if ch == '\\' {
                backslashes += 1;
                continue;
            }
            if let Some(quote) = in_quote {
                if ch == quote && backslashes.is_multiple_of(2) {
                    in_quote = None;
                }
            } else if STRING_DELIMITERS.contains(&ch) && backslashes.is_multiple_of(2) {
                in_quote = Some(ch);
            } else if ch == ':' && backslashes.is_multiple_of(2) {
                // Upstream logs the object-style separator still present:
                // keep the object repair.
                return (EmptyObjectRepair::Keep, None);
            }
            backslashes = 0;
        }

        if repairer_active
            && let Some(repairer) = self.schema_repairer.as_ref()
            && repairer.is_salvage()
            && schema.is_some_and(|schema_value| {
                matches!(schema_value, Value::Object(_))
                    && repairer.is_object_schema(schema_value)
                    && !repairer.is_array_schema(schema_value)
            })
        {
            return (EmptyObjectRepair::SchemaSetObject, None);
        }
        (EmptyObjectRepair::Array, None)
    }

    /// parse_object.py's `_complete_object_parse`: the object's exit —
    /// skip one extra closing brace when nested, merge a comma-then-
    /// delimiter continuation into this object, and finalize against the
    /// schema config.
    fn complete_object_parse(
        &mut self,
        mut obj: ObjectBuilder,
        schema: Option<&Value>,
        path: &str,
        repairer_active: bool,
        schema_config: Option<&ObjectSchemaConfig>,
    ) -> Result<Value, String> {
        if !self.ctx_empty() {
            if self.cur() == Some('}')
                && !matches!(
                    self.ctx_current(),
                    Some(Ctx::ObjectKey) | Some(Ctx::ObjectValue)
                )
            {
                // Upstream logs the extra closing brace and skips it.
                self.index += 1;
            }
            return Ok(obj.finish());
        }

        self.skip_whitespaces();
        if self.cur() == Some(',') {
            self.index += 1;
            self.skip_whitespaces();
            if self.cur().is_some_and(|c| STRING_DELIMITERS.contains(&c)) && !self.strict {
                // Upstream logs the comma + string delimiter after the
                // closing brace and checks for additional key-value pairs.
                //
                // Depth-guard the recursion. This continuation calls
                // parse_object, which can reach another comma-merge and
                // recurse again, so an unbounded chain (`{"a":1}` followed by
                // `, "k":1}` repeated) grows the native stack one frame per
                // fragment and overflows it — an uncatchable SIGSEGV, not the
                // documented catchable ValueError. enter_depth caps it at
                // MAX_NESTING and raises "Input nesting exceeds ...", the same
                // enter/parse/leave/`?` shape parse_json's `{`/`[`/`(` branches
                // use (parser.rs); balanced on every non-abort path.
                self.enter_depth()?;
                let additional_obj = self.parse_object(schema, path);
                self.leave_depth();
                if let Value::Object(additional) = additional_obj? {
                    // dict.update: overwrite in place, append the new.
                    for (key, value) in additional {
                        obj.insert(key, value);
                    }
                }
            }
        }

        // The resolver's local repairer: Some only while schema-guided.
        let repairer = if repairer_active {
            self.schema_repairer.as_ref()
        } else {
            None
        };
        finalize_object(obj.finish(), repairer, schema_config, path)
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

    #[test]
    fn parse_object() {
        // test_parse_object.py::test_parse_object
        assert_eq!(parse_ok("{}"), obj(&[]));
        assert_eq!(
            parse_ok(r#"{ "key": "value", "key2": 1, "key3": True }"#),
            obj(&[
                ("key", s("value")),
                ("key2", Value::Int(1)),
                ("key3", Value::Bool(true)),
            ])
        );
        assert_eq!(parse_ok("{"), obj(&[]));
        assert_eq!(
            parse_ok(r#"{ "key": value, "key2": 1 "key3": null }"#),
            obj(&[
                ("key", s("value")),
                ("key2", Value::Int(1)),
                ("key3", Value::Null),
            ])
        );
        // upstream asserts the serialized "{}" for these; the Value form is
        // the parser-level pin (dumps parity is dumps.rs's battery)
        assert_eq!(parse_ok("   {  }   "), obj(&[]));
        assert_eq!(parse_ok("{"), obj(&[]));
        assert_eq!(parse_ok("}"), s(""));
        assert_eq!(parse_ok("{\""), obj(&[]));
    }

    #[test]
    fn parse_object_edge_cases() {
        // test_parse_object.py::test_parse_object_edge_cases (every
        // assertion)
        assert_eq!(parse_ok("{foo: [}"), obj(&[("foo", a(vec![]))]));
        assert_eq!(parse_ok(r#"{"": "value""#), obj(&[("", s("value"))]));
        assert_eq!(
            parse_ok(r#"{"key": "v"alue"}"#),
            obj(&[("key", s("v\"alue\""))])
        );
        assert_eq!(
            parse_ok(r#"{"value_1": true, COMMENT "value_2": "data"}"#),
            obj(&[("value_1", Value::Bool(true)), ("value_2", s("data"))])
        );
        assert_eq!(
            parse_ok(r#"{"value_1": true, SHOULD_NOT_EXIST "value_2": "data" AAAA }"#),
            obj(&[("value_1", Value::Bool(true)), ("value_2", s("data"))])
        );
        assert_eq!(
            parse_ok(r#"{"" : true, "key2": "value2"}"#),
            obj(&[("", Value::Bool(true)), ("key2", s("value2"))])
        );
        assert_eq!(
            parse_ok(r#"{""answer"":[{""traits"":''Female aged 60+'',""answer1"":""5""}]}"#),
            obj(&[(
                "answer",
                a(vec![obj(&[
                    ("traits", s("Female aged 60+")),
                    ("answer1", s("5")),
                ])])
            )])
        );
        assert_eq!(
            parse_ok(r#"{ "words": abcdef", "numbers": 12345", "words2": ghijkl" }"#),
            obj(&[
                ("words", s("abcdef")),
                ("numbers", Value::Int(12345)),
                ("words2", s("ghijkl")),
            ])
        );
        assert_eq!(
            parse_ok(r#"{"number": 1,"reason": "According...""ans": "YES"}"#),
            obj(&[
                ("number", Value::Int(1)),
                ("reason", s("According...")),
                ("ans", s("YES")),
            ])
        );
        assert_eq!(
            parse_ok(r#"{ "a" : "{ b": {} }" }"#),
            obj(&[("a", s("{ b"))])
        );
        assert_eq!(
            parse_ok(r#"{"b": "xxxxx" true}"#),
            obj(&[("b", s("xxxxx"))])
        );
        assert_eq!(
            parse_ok(r#"{"key": "Lorem "ipsum" s,"}"#),
            obj(&[("key", s("Lorem \"ipsum\" s,"))])
        );
        assert_eq!(
            parse_ok(r#"{"lorem": ipsum, sic, datum.",}"#),
            obj(&[("lorem", s("ipsum, sic, datum."))])
        );
        assert_eq!(
            parse_ok(r#"{"lorem": sic tamet. "ipsum": sic tamet, quick brown fox. "sic": ipsum}"#),
            obj(&[
                ("lorem", s("sic tamet.")),
                ("ipsum", s("sic tamet")),
                ("sic", s("ipsum")),
            ])
        );
        assert_eq!(
            parse_ok(r#"{"lorem_ipsum": "sic tamet, quick brown fox. }"#),
            obj(&[("lorem_ipsum", s("sic tamet, quick brown fox."))])
        );
        assert_eq!(
            parse_ok(r#"{"key":value, " key2":"value2" }"#),
            obj(&[("key", s("value")), (" key2", s("value2"))])
        );
        assert_eq!(
            parse_ok(r#"{"key":value "key2":"value2" }"#),
            obj(&[("key", s("value")), ("key2", s("value2"))])
        );
        assert_eq!(
            parse_ok(r#"{'text': 'words{words in brackets}more words'}"#),
            obj(&[("text", s("words{words in brackets}more words"))])
        );
        assert_eq!(
            parse_ok("{text:words{words in brackets}}"),
            obj(&[("text", s("words{words in brackets}"))])
        );
        assert_eq!(
            parse_ok("{text:words{words in brackets}m}"),
            obj(&[("text", s("words{words in brackets}m"))])
        );
        assert_eq!(
            parse_ok(r#"{"key": "value, value2"```"#),
            obj(&[("key", s("value, value2"))])
        );
        assert_eq!(
            parse_ok(r#"{"key": "value}```"#),
            obj(&[("key", s("value"))])
        );
        assert_eq!(
            parse_ok("{key:value,key2:value2}"),
            obj(&[("key", s("value")), ("key2", s("value2"))])
        );
        assert_eq!(parse_ok(r#"{"key:"value"}"#), obj(&[("key", s("value"))]));
        assert_eq!(parse_ok("{key:value}"), obj(&[("key", s("value"))]));
        // the duplicate-key split at the """" (a second member follows the
        // doubled quotes)
        let lorem = || obj(&[("lorem", obj(&[("ipsum", s("sic"))]))]);
        assert_eq!(
            parse_ok(r#"[{"lorem": {"ipsum": "sic"}, """" "lorem": {"ipsum": "sic"}]"#),
            a(vec![lorem(), lorem()])
        );
        // array-continuation merges at the key position
        assert_eq!(
            parse_ok(
                r#"{ "key": ["arrayvalue"], ["arrayvalue1"], ["arrayvalue2"], "key3": "value3" }"#
            ),
            obj(&[
                (
                    "key",
                    a(vec![s("arrayvalue"), s("arrayvalue1"), s("arrayvalue2")])
                ),
                ("key3", s("value3")),
            ])
        );
        assert_eq!(
            parse_ok(r#"{ "key": [[1, 2, 3], "a", "b"], [[4, 5, 6], [7, 8, 9]] }"#),
            obj(&[(
                "key",
                a(vec![
                    a(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
                    s("a"),
                    s("b"),
                    a(vec![Value::Int(4), Value::Int(5), Value::Int(6)]),
                    a(vec![Value::Int(7), Value::Int(8), Value::Int(9)]),
                ])
            )])
        );
        assert_eq!(
            parse_ok(r#"{ "key": ["arrayvalue"], "key3": "value3", ["arrayvalue1"] }"#),
            obj(&[
                ("key", a(vec![s("arrayvalue")])),
                ("key3", s("value3")),
                ("arrayvalue1", s("")),
            ])
        );
        // the double-escaped inner object stays a string value
        assert_eq!(
            parse_ok(r#"{"key": "{\\"key\\":[\"value\"],\\"key2\":"value2"}"}"#),
            obj(&[("key", s(r#"{"key":["value"],"key2":"value2"}"#))])
        );
        assert_eq!(
            parse_ok(r#"{"key": , "key2": "value2"}"#),
            obj(&[("key", s("")), ("key2", s("value2"))])
        );
        // the ']' rollback leaves the bracket for the enclosing array
        assert_eq!(
            parse_ok(r#"{"array":[{"key": "value"], "key2": "value2"}"#),
            obj(&[
                ("array", a(vec![obj(&[("key", s("value"))])])),
                ("key2", s("value2")),
            ])
        );
        // the duplicate-key split splice
        assert_eq!(
            parse_ok(r#"[{"key":"value"}},{"key":"value"}]"#),
            a(vec![
                obj(&[("key", s("value"))]),
                obj(&[("key", s("value"))])
            ])
        );
        assert_eq!(
            parse_ok(
                r#"{'key': ['a':{'duplicated_key': 'duplicated_value', 'duplicated_key': 'duplicated_value'}]}"#
            ),
            obj(&[(
                "key",
                a(vec![obj(&[(
                    "a",
                    obj(&[("duplicated_key", s("duplicated_value"))]),
                )])])
            )])
        );
        // skip_json_loads upstream: the parser path, duplicate key kept via
        // the in-place overwrite
        assert_eq!(
            parse_ok(r#"[{"b":"v2","b":"v2"}]"#),
            a(vec![obj(&[("b", s("v2"))])])
        );
        // the set-like object falls back to an array of its items
        assert_eq!(
            parse_ok("{'item1', 'item2', 'item3'}"),
            a(vec![s("item1"), s("item2"), s("item3")])
        );
    }

    #[test]
    fn parse_object_preserves_backslash_escaped_keys() {
        // test_parse_object.py::test_parse_object_preserves_backslash_escaped_keys
        // (log-text assertions skipped: logs are not ported)
        let raw = r#"{\"key\": \"value\"}"#;
        assert_eq!(parse_ok(raw), obj(&[("key", s("value"))]));
    }

    #[test]
    fn parse_object_empty_object_classifier_keeps_objectish_inputs() {
        // test_parse_object.py::test_parse_object_empty_object_classifier_keeps_objectish_inputs
        // (log-text assertions skipped: logs are not ported)
        assert_eq!(parse_ok("{:}"), obj(&[]));
        assert_eq!(parse_ok("{   }"), obj(&[]));
    }

    #[test]
    fn parse_object_empty_object_classifier_keeps_array_fallback_for_backslash_noise() {
        // test_parse_object.py::test_parse_object_empty_object_classifier_keeps_array_fallback_for_backslash_noise
        // (log-text assertions skipped: logs are not ported): the escaped
        // backslash normalizes to a backspace inside the fallback array's
        // single string item
        assert_eq!(parse_ok(r#"{foo\bar}"#), a(vec![s("foo\u{8}ar}")]));
    }

    #[test]
    fn parse_object_empty_object_array_fallback_preserves_legacy_key_context() {
        // test_parse_object.py::test_parse_object_empty_object_array_fallback_preserves_legacy_key_context
        assert_eq!(parse_ok("[{5}s "), a(vec![a(vec![Value::Int(5)])]));
    }

    #[test]
    fn comma_merged_object_fragments_hit_the_depth_cap_not_the_stack() {
        // `{"a":1}` + `, "k":1}` * N recurses through complete_object_parse's
        // comma-merge continuation; without the depth guard it overflows the
        // native stack (uncatchable). Guarded, it raises the same capped error
        // as every other deep-recursion path — never a crash.
        let payload = format!("{}{}", r#"{"a":1}"#, r#", "k":1}"#.repeat(2_000));
        let err = Parser::new(&payload, false, None)
            .parse()
            .expect_err("a runaway comma-merge chain must raise, not recurse unbounded");
        assert!(err.contains("Input nesting exceeds"));
    }

    #[test]
    fn comma_merged_fragments_below_the_cap_still_merge() {
        // The guard fires only past MAX_NESTING; an ordinary comma-merge
        // chain must still parse and merge every fragment (so an off-by-one
        // that trips the guard early would fail here).
        let mut payload = String::from(r#"{"a":1}"#);
        for i in 0..150 {
            payload.push_str(&format!(r#", "k{i}":1}}"#));
        }
        match parse_ok(&payload) {
            Value::Object(entries) => assert_eq!(entries.len(), 151),
            other => panic!("expected a merged object, got {other:?}"),
        }
    }

    #[test]
    fn merged_array_continuation_chains_hit_the_depth_cap_not_the_stack() {
        // `{"a":[0],` + `["b":[0],` * N nests through the array-continuation
        // merge: a '[' at the key position merges into the previous
        // array-valued member (merge_object_array_continuation →
        // parse_array), whose first item — a string followed by ':' — is a
        // missing object start parsed by parse_object directly, and that
        // object's key scan sees another '[' and merges again. Without a
        // depth guard on that continuation the cycle grew the native stack
        // with no cap anywhere on it — an uncatchable SIGSEGV around 8k
        // fragments (main thread; ~4k fewer on worker-sized stacks), not
        // the documented catchable ValueError.
        let payload = format!("{}{}1]", r#"{"a":[0],"#, r#"["b":[0],"#.repeat(2_000));
        let err = Parser::new(&payload, false, None)
            .parse()
            .expect_err("a runaway array-merge chain must raise, not recurse unbounded");
        assert!(err.contains("Input nesting exceeds"));
    }

    #[test]
    fn merged_array_continuations_below_the_cap_still_merge() {
        // The guard fires only past MAX_NESTING. An ordinary nested merge
        // chain still parses and merges every fragment (an off-by-one that
        // trips the guard early would fail the 150-level descent below),
        // and same-level sequential merges never accrue depth at all —
        // enter/leave is balanced per continuation, so merged items land
        // flat in the previous array.
        // N=2 pins the exact merged shape:
        assert_eq!(
            parse_ok(r#"{"a":[0],["b":[0],["b":[0],1]"#),
            obj(&[(
                "a",
                a(vec![
                    Value::Int(0),
                    obj(&[(
                        "b",
                        a(vec![
                            Value::Int(0),
                            obj(&[("b", a(vec![Value::Int(0)])), ("1", s(""))]),
                        ]),
                    )]),
                ])
            )])
        );
        // A 150-fragment chain descends 150 objects deep to the innermost
        // fragment's exact shape.
        let payload = format!("{}{}1]", r#"{"a":[0],"#, r#"["b":[0],"#.repeat(150));
        let value = parse_ok(&payload);
        fn member<'a>(value: &'a Value, key: &str) -> &'a Value {
            match value {
                Value::Object(entries) => entries
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| v)
                    .unwrap_or_else(|| panic!("member {key} missing")),
                other => panic!("expected an object, got {other:?}"),
            }
        }
        fn item(value: &Value, idx: usize) -> &Value {
            match value {
                Value::Array(items) => &items[idx],
                other => panic!("expected an array, got {other:?}"),
            }
        }
        let mut node = item(member(&value, "a"), 1);
        for _ in 1..150 {
            node = item(member(node, "b"), 1);
        }
        assert_eq!(node, &obj(&[("b", a(vec![Value::Int(0)])), ("1", s(""))]));
        // Same-level sequential merges stay flat:
        assert_eq!(
            parse_ok(r#"{"a":[1], [2], [3]}"#),
            obj(&[("a", a(vec![Value::Int(1), Value::Int(2), Value::Int(3)]))])
        );
    }

    #[test]
    fn continuation_chains_cap_at_max_nesting_exactly() {
        // Both continuation recursions (comma-merge and array-merge) share
        // the MAX_NESTING budget with structural nesting. The comma chain
        // spends 1 (the initial `{`) + 1 per fragment (scalar values add
        // nothing): 199 fragments parse (depth 200), the 200th raises.
        // The array-merge chain spends the same 1 + 1 per fragment PLUS 1
        // for the innermost fragment's `[0]` value (a container nested
        // inside every merge): 198 fragments parse (depth 200), the 199th
        // raises. Pinning the exact edges catches future accounting drift
        // in either direction — over-counting an edge rejects inputs the
        // cap admits, missing one reopens the crash.
        let comma_ok = format!("{}{}", r#"{"a":1}"#, r#", "k":1}"#.repeat(199));
        match parse_ok(&comma_ok) {
            // "a" plus the repeated "k" (dict.update collapses the rest)
            Value::Object(entries) => assert_eq!(entries.len(), 2),
            other => panic!("expected a merged object, got {other:?}"),
        }
        let comma_cap = format!("{}{}", r#"{"a":1}"#, r#", "k":1}"#.repeat(200));
        assert!(
            Parser::new(&comma_cap, false, None)
                .parse()
                .unwrap_err()
                .contains("Input nesting exceeds")
        );

        let merge_ok = format!("{}{}1]", r#"{"a":[0],"#, r#"["b":[0],"#.repeat(198));
        assert!(matches!(parse_ok(&merge_ok), Value::Object(_)));
        let merge_cap = format!("{}{}1]", r#"{"a":[0],"#, r#"["b":[0],"#.repeat(199));
        assert!(
            Parser::new(&merge_cap, false, None)
                .parse()
                .unwrap_err()
                .contains("Input nesting exceeds")
        );
    }

    #[test]
    fn parse_object_merge_continuation_row_widths() {
        // The array-merge continuation's row-width summary (the incremental
        // spelling of upstream's list_lengths rescan) must produce the
        // identical outputs across every summary transition: uniform kept,
        // uniform broken to mixed, mixed from the start, zero shared width
        // (Python truthiness: falsy -> the no-width branch), divisible and
        // non-divisible trailing tails, the lone-list flatten, and leading
        // scalars. Each expected value is upstream-differential-verified
        // (json_repair 0.63.4, skip_json_loads).
        let cases: &[(&str, Value)] = &[
            // uniform rows + a same-width row: uniformity kept.
            (
                r#"{"a": [[1,2]], [3,4]}"#,
                obj(&[(
                    "a",
                    a(vec![
                        a(vec![Value::Int(1), Value::Int(2)]),
                        a(vec![Value::Int(3), Value::Int(4)]),
                    ]),
                )]),
            ),
            // a width-breaking row: uniform -> mixed.
            (
                r#"{"a": [[1,2]], [3,4], [5]}"#,
                obj(&[(
                    "a",
                    a(vec![
                        a(vec![Value::Int(1), Value::Int(2)]),
                        a(vec![Value::Int(3), Value::Int(4)]),
                        a(vec![Value::Int(5)]),
                    ]),
                )]),
            ),
            // past the break the no-width branch extends as-is.
            (
                r#"{"a": [[1,2]], [3,4], [5], [6]}"#,
                obj(&[(
                    "a",
                    a(vec![
                        a(vec![Value::Int(1), Value::Int(2)]),
                        a(vec![Value::Int(3), Value::Int(4)]),
                        a(vec![Value::Int(5)]),
                        Value::Int(6),
                    ]),
                )]),
            ),
            // a divisible trailing tail regroups into rows of the shared
            // width.
            (
                r#"{"a": [[1,2], 3, 4], [5, 6]}"#,
                obj(&[(
                    "a",
                    a(vec![
                        a(vec![Value::Int(1), Value::Int(2)]),
                        a(vec![Value::Int(3), Value::Int(4)]),
                        a(vec![Value::Int(5), Value::Int(6)]),
                    ]),
                )]),
            ),
            // a non-divisible tail stays flat.
            (
                r#"{"a": [[1,2], 3], [4]}"#,
                obj(&[(
                    "a",
                    a(vec![
                        a(vec![Value::Int(1), Value::Int(2)]),
                        Value::Int(3),
                        a(vec![Value::Int(4)]),
                    ]),
                )]),
            ),
            // a zero shared width is falsy: the no-width branch.
            (
                r#"{"a": [[]], [1]}"#,
                obj(&[("a", a(vec![a(vec![]), Value::Int(1)]))]),
            ),
            // zero width, then more scalars.
            (
                r#"{"a": [[]], [1], [2]}"#,
                obj(&[("a", a(vec![a(vec![]), Value::Int(1), Value::Int(2)]))]),
            ),
            // mixed widths from the start: no regroup, extend as-is.
            (
                r#"{"a": [[1], [2,3]], [4]}"#,
                obj(&[(
                    "a",
                    a(vec![
                        a(vec![Value::Int(1)]),
                        a(vec![Value::Int(2), Value::Int(3)]),
                        Value::Int(4),
                    ]),
                )]),
            ),
            // the lone list item flattens into the previous value.
            (
                r#"{"a": [1], [[2]]"#,
                obj(&[("a", a(vec![Value::Int(1), Value::Int(2)]))]),
            ),
            // two rows extend as-is (not the lone-list shape).
            (
                r#"{"a": [1], [[2], [3]]"#,
                obj(&[(
                    "a",
                    a(vec![
                        Value::Int(1),
                        a(vec![Value::Int(2)]),
                        a(vec![Value::Int(3)]),
                    ]),
                )]),
            ),
            // a leading scalar before the rows: widths unaffected.
            (
                r#"{"a": [1, [2]], [3]"#,
                obj(&[(
                    "a",
                    a(vec![
                        Value::Int(1),
                        a(vec![Value::Int(2)]),
                        a(vec![Value::Int(3)]),
                    ]),
                )]),
            ),
            // no comma between the member and its continuation.
            (
                r#"{"a": [[1,2]] [3,4]"#,
                obj(&[(
                    "a",
                    a(vec![
                        a(vec![Value::Int(1), Value::Int(2)]),
                        a(vec![Value::Int(3), Value::Int(4)]),
                    ]),
                )]),
            ),
        ];
        for (raw, expected) in cases {
            assert_eq!(parse_ok(raw), *expected, "raw: {raw}");
        }
        // A same-width run keeps uniformity across many folds: every
        // continuation lands as one more row.
        let mut payload = String::from(r#"{"a": [[0,1]]"#);
        for i in 0..8 {
            payload.push_str(&format!(", [{},{}]", 2 * i + 2, 2 * i + 3));
        }
        let rows: Vec<Value> = (0..9)
            .map(|i| a(vec![Value::Int(2 * i), Value::Int(2 * i + 1)]))
            .collect();
        assert_eq!(parse_ok(&payload), obj(&[("a", a(rows))]));
    }

    #[test]
    fn parse_object_merge_at_the_end() {
        // test_parse_object.py::test_parse_object_merge_at_the_end
        assert_eq!(
            parse_ok(r#"{"key": "value"}, "key2": "value2"}"#),
            obj(&[("key", s("value")), ("key2", s("value2"))])
        );
        assert_eq!(
            parse_ok(r#"{"key": "value"}, "key2": }"#),
            obj(&[("key", s("value")), ("key2", s(""))])
        );
        assert_eq!(
            parse_ok(r#"{"key": "value"}, []"#),
            obj(&[("key", s("value"))])
        );
        assert_eq!(
            parse_ok(r#"{"key": "value"}, ["abc"]"#),
            a(vec![obj(&[("key", s("value"))]), a(vec![s("abc")])])
        );
        assert_eq!(
            parse_ok(r#"{"key": "value"}, {}"#),
            obj(&[("key", s("value"))])
        );
        assert_eq!(
            parse_ok(r#"{"key": "value"}, "" : "value2"}"#),
            obj(&[("key", s("value")), ("", s("value2"))])
        );
        assert_eq!(
            parse_ok(r#"{"key": "value"}, "key2" "value2"}"#),
            obj(&[("key", s("value")), ("key2", s("value2"))])
        );
        assert_eq!(
            parse_ok(r#"{"key1": "value1"}, "key2": "value2", "key3": "value3"}"#),
            obj(&[
                ("key1", s("value1")),
                ("key2", s("value2")),
                ("key3", s("value3")),
            ])
        );
    }

    #[test]
    fn strict_mode_object_raises() {
        // test_strict_mode.py::test_strict_duplicate_keys_inside_array
        assert_eq!(
            parse_strict(r#"[{"key": "first", "key": "second"}]"#),
            Err("Duplicate key found in strict mode while parsing object.".to_string())
        );
        // test_strict_mode.py::test_strict_rejects_empty_keys
        assert_eq!(
            parse_strict(r#"{"" : "value"}"#),
            Err("Empty key found in strict mode while parsing object.".to_string())
        );
        // test_strict_mode.py::test_strict_requires_colon_between_key_and_value
        assert_eq!(
            parse_strict(r#"{"missing" "colon"}"#),
            Err("Missing ':' after key in strict mode while parsing object.".to_string())
        );
        // test_strict_mode.py::test_strict_rejects_empty_values
        assert_eq!(
            parse_strict(r#"{"key": , "key2": "value2"}"#),
            Err("Parsed value is empty in strict mode while parsing object.".to_string())
        );
        // test_strict_mode.py::test_strict_rejects_empty_object_with_extra_characters
        assert_eq!(
            parse_strict(r#"{"dangling"}"#),
            Err("Parsed object is empty but contains extra characters in strict mode.".to_string())
        );
        // test_strict_mode.py::test_strict_rejects_empty_escaped_object_with_extra_characters
        assert_eq!(
            parse_strict(r#"{\"key\": \"value\"}"#),
            Err("Parsed object is empty but contains extra characters in strict mode.".to_string())
        );
        // test_strict_mode.py::test_strict_detects_immediate_doubled_quotes
        assert_eq!(
            parse_strict(r#"{"key": """"}"#),
            Err("Found doubled quotes followed by another quote.".to_string())
        );
        // test_strict_mode.py::test_strict_detects_doubled_quotes_followed_by_string
        assert_eq!(
            parse_strict(r#"{"key": "" "value"}"#),
            Err(
                "Found doubled quotes followed by another quote while parsing a string."
                    .to_string()
            )
        );
    }
}
