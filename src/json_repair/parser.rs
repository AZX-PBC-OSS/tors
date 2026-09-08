//! The repair parser core: a port of json_repair's `json_parser.py`
//! (upstream: https://github.com/mangiucugna/json_repair, MIT, commit
//! 251d141786d0f6ff561f6ec04d90188a338e2470). THIS file holds the shared
//! `Parser` state and the utility methods every sibling file drives
//! (the `get`/`skip`/`scroll`/`skip_to_character` trio, the context stack,
//! the whitespace scans); the parse methods themselves are split across the
//! sibling files exactly like upstream split its modules:
//!
//! - HERE (agent A2's section, below the utilities): `parse_json`
//!   (the dispatch loop + schema fast-path suffix probe), `parse`
//!   /`parse_with_schema` (the top-level orchestration incl. the salvage
//!   fragment loop), `parse_number` (parse_number.py), `parse_comment`
//!   (parse_comment.py).
//! - `string.rs` (agent A3): `parse_string` + every parse_string.py helper.
//! - `object.rs` (agent A4): `parse_object` (parse_object.py).
//! - `array.rs` (agent A4): `parse_array` (parse_array.py).
//! - `parenthesized.rs` (agent A4): the tuple/grouping classifiers and
//!   `parse_parenthesized` (parser_parenthesized.py).
//!
//! # The pinned cross-file surface
//!
//! `impl Parser` blocks are split across this crate's files (legal in one
//! crate); these are the signatures every cross-file call uses — port
//! against them EXACTLY:
//!
//! ```rust,ignore
//! // parser.rs (A2):
//! impl Parser {
//!     fn parse_json(&mut self, schema: Option<&Value>, path: &str,
//!                   finalize_schema: bool, record_top_level_value: bool)
//!         -> Result<Value, String>;          // json_parser.py parse_json
//!     fn parse(&mut self) -> Result<Value, String>;             // .parse()
//!     fn parse_with_schema(&mut self) -> Result<Value, String>; // .parse_with_schema()
//!     fn parse_number(&mut self) -> Result<Value, String>;      // parse_number.py
//!     fn parse_comment(&mut self, record_top_level_value: bool) -> Result<Value, String>;
//!     fn take_repairer(&mut self) -> Option<SchemaRepairer>;    // back to repair() for the final validate
//! }
//! // string.rs (A3):
//! impl Parser {
//!     fn parse_string(&mut self) -> Result<Value, String>;
//!     fn parse_json_llm_block(&mut self) -> Result<Option<Value>, String>; // Python's `False` -> Ok(None)
//! }
//! // object.rs (A4):
//! impl Parser { fn parse_object(&mut self, schema: Option<&Value>, path: &str) -> Result<Value, String>; }
//! // array.rs (A4):
//! impl Parser { fn parse_array(&mut self, schema: Option<&Value>, path: &str,
//!                              closing_delimiter: char) -> Result<Value, String>; }
//! // parenthesized.rs (A4):
//! impl Parser {
//!     fn parenthesized_is_explicit_tuple(&self) -> bool;
//!     fn top_level_parenthesized_can_start_value(&self) -> bool;
//!     fn parse_parenthesized(&mut self, schema: Option<&Value>, path: &str) -> Result<Value, String>;
//! }
//! ```
//!
//! Python `with self.context.enter(X):` blocks port to explicit
//! `self.ctx_push(X)` / `self.ctx_pop()` pairs bracketing the same region
//! (including on the early-return paths — the Python context manager pops
//! on every exit, so every `?`/early return inside such a region needs its
//! pop; the simplest faithful shape is a closure or a guard run to the
//! region's end, A3/A4's call).
//!
//! `self.log(...)` sites in the Python sources become rationale COMMENTS
//! here (upstream's log texts are not ported; the diagnostics surface is
//! schema-layer actions in v1 — see mod.rs's scope note).

use super::{MAX_NESTING, NUMBER_CHARS, STRING_DELIMITERS, Value};
use crate::json_schema_impl::{ResolvedSchema, SchemaRepairer};
use crate::normalize_impl::is_py_whitespace;

/// json_context.py's `ContextValues`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ctx {
    ObjectKey,
    ObjectValue,
    Array,
}

/// json_parser.py's `JSONParser` — the mutable repair machine. The input is
/// a `Vec<char>` (Python str indexing is codepoint indexing; exact parity,
/// astral chars included) that STAYS MUTABLE: the duplicate-key split and
/// the escaped-object reparse splice new text into it and rewind `index`
/// (port those sites faithfully — they are rare, so the O(n) splice is
/// fine).
pub(crate) struct Parser {
    /// The input as codepoints; spliced by the object-repair heuristics.
    pub(crate) s: Vec<char>,
    /// The cursor, in codepoint units (Python's `self.index`).
    pub(crate) index: usize,
    /// json_context.py's context stack: `context.current` is the LAST entry
    /// (`ctx_current`), `ContextValues.X in self.context.context` is
    /// "anywhere in the stack" (`ctx_has`).
    pub(crate) context: Vec<Ctx>,
    /// json_parser.py's `deferred_contexts`: contexts re-entered on the next
    /// `parse_json` call after an object/array reparse rewound the cursor.
    pub(crate) deferred_contexts: Vec<Ctx>,
    /// Strict mode: repair heuristics raise instead (the §8 catalog).
    pub(crate) strict: bool,
    /// json_fd is None for every tors input, so upstream's
    /// `try_valid_json_suffix` is ALWAYS true here: after the parser has
    /// skipped garbage to a `{`/`[`, one `raw_decode` probe may claim the
    /// rest as valid JSON (once per Parser).
    pub(crate) try_valid_json_suffix: bool,
    pub(crate) has_tried_valid_json_suffix: bool,
    /// The salvage loop's "did this parse_json call actually find a value"
    /// flag (record_top_level_value machinery).
    pub(crate) last_parse_found_value: bool,
    /// The schema layer, attached for schema-guided parsing (None for plain
    /// repairs). Returned to `repair()` afterwards via `take_repairer`.
    pub(crate) schema_repairer: Option<SchemaRepairer>,
    /// Container-nesting depth for the MAX_NESTING guard: parse_json's `{`
    /// and `[` branches increment on entry and decrement on their way out;
    /// exceeding the cap raises the recursion-depth ValueError (upstream
    /// hits Python's RecursionError at a comparable depth; tors normalizes
    /// it — see mod.rs's docs).
    pub(crate) depth: usize,
    /// parse_comment's parse_json re-entry depth. Garbage-separated
    /// comment runs (`'/x' * n`) chain parse_json → parse_comment →
    /// parse_json without unwinding — 2 stack frames per 2 input chars in
    /// tors (upstream's RecursionError fires at ~500 pairs and is
    /// normalized to the same ValueError); the cap keeps the chain from
    /// overflowing the native stack. Consecutive comment runs are absorbed
    /// by parse_comment's own loop and cost one re-entry, so legitimate
    /// inputs never approach the cap.
    comment_depth: usize,
}

impl Parser {
    pub(crate) fn new(s: &str, strict: bool, schema_repairer: Option<SchemaRepairer>) -> Parser {
        Parser {
            s: s.chars().collect(),
            index: 0,
            context: Vec::new(),
            deferred_contexts: Vec::new(),
            strict,
            try_valid_json_suffix: true,
            has_tried_valid_json_suffix: false,
            last_parse_found_value: false,
            schema_repairer,
            depth: 0,
            comment_depth: 0,
        }
    }

    /// json_parser.py's `get_char_at` — with PYTHON indexing semantics: the
    /// position is `index + count` in codepoints; a negative position wraps
    /// from the end (`json_str[-1]` is the last character); out of range
    /// (either side) is `None` (upstream's IndexError -> None).
    pub(crate) fn get(&self, count: isize) -> Option<char> {
        let pos = self.index as isize + count;
        if pos >= 0 {
            self.s.get(pos as usize).copied()
        } else {
            let len = self.s.len() as isize;
            if pos >= -len {
                self.s.get((len + pos) as usize).copied()
            } else {
                None
            }
        }
    }

    /// The character at the cursor (the overwhelmingly common `get(0)`).
    pub(crate) fn cur(&self) -> Option<char> {
        self.get(0)
    }

    /// json_parser.py's `skip_whitespaces`: advance `index` past whitespace.
    pub(crate) fn skip_whitespaces(&mut self) {
        while let Some(c) = self.s.get(self.index) {
            if crate::normalize_impl::is_py_whitespace(*c) {
                self.index += 1;
            } else {
                break;
            }
        }
    }

    /// json_parser.py's `scroll_whitespaces`: the non-advancing twin — how
    /// far (in codepoints, from `index`) the whitespace run at `index + idx`
    /// extends. Returns the offset RELATIVE to `index` (idx plus the run).
    pub(crate) fn scroll_whitespaces(&self, idx: usize) -> usize {
        let mut idx = idx;
        while let Some(c) = self.s.get(self.index + idx) {
            if crate::normalize_impl::is_py_whitespace(*c) {
                idx += 1;
            } else {
                break;
            }
        }
        idx
    }

    /// json_parser.py's `skip_to_character`: advance a virtual cursor from
    /// `index + idx` until an UNESCAPED target character (a target preceded
    /// by an EVEN run of backslashes); returns the offset from `index` to
    /// that position, or the distance to the end when not found.
    pub(crate) fn skip_to_character(&self, targets: &[char], idx: usize) -> usize {
        let mut i = self.index + idx;
        let n = self.s.len();
        let mut backslashes = 0usize;
        while i < n {
            let ch = self.s[i];
            if ch == '\\' {
                backslashes += 1;
                i += 1;
                continue;
            }
            if targets.contains(&ch) && backslashes.is_multiple_of(2) {
                return i - self.index;
            }
            backslashes = 0;
            i += 1;
        }
        n - self.index
    }

    /// json_context.py's `context.current`: the innermost context.
    pub(crate) fn ctx_current(&self) -> Option<Ctx> {
        self.context.last().copied()
    }

    /// json_context.py's `context.empty`.
    pub(crate) fn ctx_empty(&self) -> bool {
        self.context.is_empty()
    }

    /// `ContextValues.X in self.context.context`: anywhere in the stack.
    pub(crate) fn ctx_has(&self, ctx: Ctx) -> bool {
        self.context.contains(&ctx)
    }

    /// `with self.context.enter(X)`'s `__enter__`.
    pub(crate) fn ctx_push(&mut self, ctx: Ctx) {
        self.context.push(ctx);
    }

    /// `with self.context.enter(X)`'s `__exit__` (and `context.reset()`):
    /// pop the innermost context.
    pub(crate) fn ctx_pop(&mut self) {
        self.context.pop();
    }

    /// json_context.py's `context.clear()`.
    pub(crate) fn ctx_clear(&mut self) {
        self.context.clear();
    }

    /// The depth guard shared by parse_json's container branches: raises
    /// upstream's normalized message at the cap (see the struct docs).
    pub(crate) fn enter_depth(&mut self) -> Result<(), String> {
        self.depth += 1;
        if self.depth > MAX_NESTING {
            return Err("Input nesting exceeds the supported parser recursion depth.".into());
        }
        Ok(())
    }

    pub(crate) fn leave_depth(&mut self) {
        self.depth -= 1;
    }

    // =====================================================================
    // json_parser.py below the utility methods (plus parse_number.py and
    // parse_comment.py): the parse orchestration — parse/parse_with_schema,
    // the top-level multi-value loop, the salvage fragment loop, the
    // strict-suffix fast path, and the number/comment fallbacks. The
    // string/object/array/parenthesized parse methods live in the sibling
    // files, per this file's module docs.
    // =====================================================================

    /// json_parser.py's `parse`: the plain top-level parse — no schema, the
    /// default finalize/record flags (upstream passes the bound method with
    /// all-default arguments).
    pub(crate) fn parse(&mut self) -> Result<Value, String> {
        self.parse_top_level(|p| p.parse_json(None, "$", true, false))
    }

    /// json_parser.py's `parse_with_schema`: schema-guided parsing. The
    /// repairer is attached at construction (upstream passes it in here;
    /// tors attaches it in `Parser::new` so the nested `parse_object`/
    /// `parse_array` calls can reach it through `self`). Standard mode runs
    /// the ordinary top-level loop with the root schema in hand; salvage
    /// mode runs the fragment loop.
    pub(crate) fn parse_with_schema(&mut self) -> Result<Value, String> {
        let Some(repairer) = self.schema_repairer.as_ref() else {
            // Unreachable from repair(): parse_with_schema is only called
            // with a schema attached.
            return Err("parse_with_schema requires a schema".into());
        };
        let (is_salvage, root) = (repairer.is_salvage(), repairer.root().clone());
        if is_salvage {
            return self.parse_top_level_salvage_with_schema();
        }
        // The root schema has to stay referenced across the whole loop while
        // the parser is &mut, so clone it once into this frame (schemas are
        // small; the per-parse clone is the pattern the design doc pins for
        // exactly this borrow tangle).
        self.parse_top_level(|p| p.parse_json(Some(&root), "$", true, false))
    }

    /// json_parser.py's `_parse_top_level`: parse one element, and when the
    /// parser stopped early, keep collecting the sequential top-level values
    /// that follow — a comma-separated sequence stays a list, a repeated
    /// (non-comma-separated) object is an UPDATE that replaces the previous
    /// value, falsy in-betweens are dropped — unwrapping back to the single
    /// element when nothing more was found.
    fn parse_top_level(
        &mut self,
        mut parse_element: impl FnMut(&mut Self) -> Result<Value, String>,
    ) -> Result<Value, String> {
        let mut json = parse_element(&mut *self)?;
        if self.has_more_input() {
            // The parser returned early: there may be more JSON elements.
            let mut items = vec![json];
            while self.has_more_input() {
                self.ctx_clear();
                self.deferred_contexts.clear();
                let is_comma_separated = self.next_top_level_value_is_comma_separated();
                let element_start_index = self.index;
                let j = parse_element(&mut *self)?;
                if self.strict && self.index > element_start_index {
                    return Err("Multiple top-level JSON elements found in strict mode.".into());
                }
                if j.is_truthy() {
                    let is_update = !is_comma_separated
                        && items.last().is_some_and(|last| last.is_same_object(&j));
                    if is_update {
                        // A repeated object is an update: keep the newest.
                        items.pop();
                    } else if items.last().is_some_and(|last| !last.is_truthy()) {
                        // The previous slot holds nothing worth keeping.
                        items.pop();
                    }
                    items.push(j);
                } else {
                    // No value found here: nudge the cursor so an element
                    // that parsed to "" without consuming input still
                    // advances one character per iteration.
                    self.index += 1;
                }
            }
            json = if items.len() == 1 {
                // No more elements: return the element without the array.
                items.remove(0)
            } else {
                Value::Array(items)
            };
        }
        Ok(json)
    }

    /// json_parser.py's `_parse_top_level_salvage_with_schema`: return the
    /// first top-level fragment that survives schema repair AND validation,
    /// skipping (and remembering the error of) every fragment that does
    /// not. `parse_json` is what advances the cursor past each fragment,
    /// so the loop always makes progress.
    fn parse_top_level_salvage_with_schema(&mut self) -> Result<Value, String> {
        let root = match self.schema_repairer.as_ref() {
            Some(repairer) => repairer.root().clone(),
            None => return Err("parse_with_schema requires a schema".into()),
        };
        let mut last_error: Option<String> = None;
        while self.has_more_input() {
            self.ctx_clear();
            self.deferred_contexts.clear();
            let value = self.parse_json(Some(&root), "$", false, true)?;
            if !self.last_parse_found_value {
                break;
            }
            let attempt: Result<Value, String> = self.with_repairer(|repairer| {
                let Some(repairer) = repairer else {
                    return Err("parse_with_schema requires a schema".into());
                };
                let repaired = repairer.repair_value(value, &root, "$")?;
                repairer.validate(&repaired, &root)?;
                Ok(repaired)
            });
            match attempt {
                Ok(repaired) => return Ok(repaired),
                Err(exc) => {
                    last_error = Some(exc);
                    if !self.has_more_input() {
                        break;
                    }
                    // Fragment skipped: it did not match the schema while
                    // salvaging; try the next one. Upstream logs the skip;
                    // §6.4's diagnostic vocabulary maps it to
                    // "skip_fragment".
                    self.with_repairer(|repairer| {
                        if let Some(repairer) = repairer {
                            repairer.record(
                                "skip_fragment",
                                "$",
                                "Skipped fragment: it did not match the schema while salvaging",
                                None,
                                None,
                                None,
                            );
                        }
                    });
                    continue;
                }
            }
        }
        match last_error {
            Some(err) => Err(err),
            None => Ok(Value::Str(String::new())),
        }
    }

    /// json_parser.py's `_next_top_level_value_is_comma_separated`: is the
    /// next top-level value separated from the previous one by a comma —
    /// either one right after the cursor (across whitespace) or one right
    /// before it (a separator the previous value left dangling).
    fn next_top_level_value_is_comma_separated(&self) -> bool {
        let idx = self.scroll_whitespaces(0);
        if self.get(idx as isize) == Some(',') {
            return true;
        }
        let mut idx = self.index as isize - 1;
        while idx >= 0
            && self
                .s
                .get(idx as usize)
                .is_some_and(|&c| is_py_whitespace(c))
        {
            idx -= 1;
        }
        idx >= 0 && self.s.get(idx as usize) == Some(&',')
    }

    /// json_parser.py's `_initial_container_has_non_comma_trailing_content`:
    /// the gate for the strict-suffix fast path. It returns `true` unless
    /// the cursor is at the very start of the input AND the first container
    /// closes balanced with only a comma (or nothing but whitespace)
    /// behind it — exactly the shapes where one `raw_decode` of the rest
    /// would NOT capture the same value the multi-value top-level loop
    /// should build. Mismatched or premature closers return `false` (the
    /// input is not "one valid value plus trailing junk"; let the repair
    /// parser handle it). Upstream also short-circuits to `true` for the
    /// file-wrapper flavor of the input; json_fd is None for every tors
    /// input, so that arm is always true and folded into the index check.
    fn initial_container_has_non_comma_trailing_content(&self) -> bool {
        if self.index != 0 {
            // Not at the start of the input: there is no "initial
            // container" to inspect, so the gate is open.
            return true;
        }
        let mut containers: Vec<char> = Vec::new();
        let mut in_string = false;
        let mut escaped = false;
        for (pos, &ch) in self.s.iter().enumerate() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    in_string = false;
                }
                continue;
            }
            if ch == '"' {
                in_string = true;
            } else if ch == '{' || ch == '[' {
                containers.push(ch);
            } else if ch == '}' || ch == ']' {
                let Some(open) = containers.pop() else {
                    // A closer with nothing open: mismatched delimiters.
                    return false;
                };
                let expected = if open == '{' { '}' } else { ']' };
                if ch != expected {
                    return false;
                }
                if containers.is_empty() {
                    // The initial container closed: there is trailing
                    // content iff something other than a comma separator
                    // follows it.
                    let mut trailing = pos + 1;
                    while self.s.get(trailing).is_some_and(|&c| is_py_whitespace(c)) {
                        trailing += 1;
                    }
                    return self.s.get(trailing).is_some_and(|&c| c != ',');
                }
            }
        }
        // The first container never closed: not the fast-path shape either.
        false
    }

    /// json_parser.py's `_try_parse_valid_json_value`: after the repair
    /// parser has skipped garbage onto a `{` or `[`, one `raw_decode` probe
    /// may claim the rest of the input as a single valid JSON value (at
    /// most once per Parser, only outside any container). `None` means the
    /// probe did not fire or failed, and the caller falls through to the
    /// ordinary repair branches.
    fn try_parse_valid_json_value(&mut self) -> Option<Value> {
        if !self.try_valid_json_suffix || self.has_tried_valid_json_suffix || !self.ctx_empty() {
            return None;
        }
        self.has_tried_valid_json_suffix = true;
        // strict.rs parses &str, so materialize the tail as a String once.
        // The probe fires at most once per Parser, so this single
        // allocation is the entire cost of the fast path.
        let tail: String = self
            .s
            .get(self.index..)
            .map_or_else(String::new, |chars| chars.iter().copied().collect());
        let (value, end) = super::raw_decode(&tail, 0).ok()?;
        self.index += end;
        Some(value)
    }

    /// json_parser.py's `parse_json`: parse the next JSON value and, when
    /// configured, enforce schema constraints over it. `finalize_schema`
    /// runs the schema layer over the parsed value (upstream's default; the
    /// salvage loop defers it to its own repair+validate pass);
    /// `record_top_level_value` drives the salvage loop's "was a value
    /// actually found here" flag.
    pub(crate) fn parse_json(
        &mut self,
        schema: Option<&Value>,
        path: &str,
        finalize_schema: bool,
        record_top_level_value: bool,
    ) -> Result<Value, String> {
        if record_top_level_value {
            self.last_parse_found_value = false;
        }
        if !self.deferred_contexts.is_empty() {
            // An object/array repair rewound the cursor mid-container: re-
            // enter with the deferred contexts on the context stack. Like
            // upstream's ExitStack, they come back off when the recursive
            // call returns (error paths included), and the deferred list
            // itself stays taken.
            let deferred = std::mem::take(&mut self.deferred_contexts);
            for ctx in &deferred {
                self.ctx_push(*ctx);
            }
            let result = self.parse_json(schema, path, finalize_schema, record_top_level_value);
            for _ in 0..deferred.len() {
                self.ctx_pop();
            }
            return result;
        }

        let (repairer_active, resolved_schema) = self.resolve_schema_for_parse(schema)?;

        loop {
            // None means that we are at the end of the string provided.
            let Some(ch) = self.cur() else {
                return Ok(Value::Str(String::new()));
            };
            // One-shot strict fast path: the rest of the input may just be
            // one valid JSON value with junk behind it.
            if self.try_valid_json_suffix
                && (ch == '{' || ch == '[')
                && self.initial_container_has_non_comma_trailing_content()
                && let Some(value) = self.try_parse_valid_json_value()
            {
                return self.finalize_parsed_value(
                    value,
                    repairer_active,
                    resolved_schema.as_ref(),
                    path,
                    finalize_schema,
                    record_top_level_value,
                );
            }
            // <object> starts with '{'
            if ch == '{' {
                self.index += 1;
                self.enter_depth()?;
                let parsed = if repairer_active {
                    self.parse_object(resolved_schema.as_ref(), path)
                } else {
                    // Upstream's no-arg call also resets the path to "$".
                    self.parse_object(None, "$")
                };
                self.leave_depth();
                return self.finalize_parsed_value(
                    parsed?,
                    repairer_active,
                    resolved_schema.as_ref(),
                    path,
                    finalize_schema,
                    record_top_level_value,
                );
            }
            // <array> starts with '['
            if ch == '[' {
                self.index += 1;
                self.enter_depth()?;
                let parsed = if repairer_active {
                    self.parse_array(resolved_schema.as_ref(), path, ']')
                } else {
                    self.parse_array(None, "$", ']')
                };
                self.leave_depth();
                return self.finalize_parsed_value(
                    parsed?,
                    repairer_active,
                    resolved_schema.as_ref(),
                    path,
                    finalize_schema,
                    record_top_level_value,
                );
            }
            // Python tuple literals and grouped values start with '('
            if ch == '(' {
                // Keep top-level tuple detection conservative so inline
                // prose like "note (clarification):" does not hijack later
                // JSON blocks.
                if !self.ctx_empty() || self.top_level_parenthesized_can_start_value() {
                    self.enter_depth()?;
                    let parsed = if repairer_active {
                        self.parse_parenthesized(resolved_schema.as_ref(), path)
                    } else {
                        self.parse_parenthesized(None, "$")
                    };
                    self.leave_depth();
                    return self.finalize_parsed_value(
                        parsed?,
                        repairer_active,
                        resolved_schema.as_ref(),
                        path,
                        finalize_schema,
                        record_top_level_value,
                    );
                }
                self.index += 1;
                continue;
            }
            // <string> starts with a quote (or a bare word) — only inside a
            // container; at the top level those are prose to skip past.
            if !self.ctx_empty() && (STRING_DELIMITERS.contains(&ch) || ch.is_alphabetic()) {
                let parsed = self.parse_string();
                return self.finalize_parsed_value(
                    parsed?,
                    repairer_active,
                    resolved_schema.as_ref(),
                    path,
                    finalize_schema,
                    record_top_level_value,
                );
            }
            // <number> starts with [0-9] or minus or '.' — likewise only
            // inside a container. isdigit is ASCII-only here (documented
            // divergence: non-ASCII digits never enter the number path).
            if !self.ctx_empty() && (ch.is_ascii_digit() || ch == '-' || ch == '.') {
                let parsed = self.parse_number();
                return self.finalize_parsed_value(
                    parsed?,
                    repairer_active,
                    resolved_schema.as_ref(),
                    path,
                    finalize_schema,
                    record_top_level_value,
                );
            }
            if matches!(ch, '#' | '/') {
                let parsed = self.parse_comment(record_top_level_value);
                // record_top_level_value is deliberately NOT forwarded: the
                // parse_json re-entry inside parse_comment already recorded
                // whether a value was found behind the comments (and an
                // empty-context comment-only result must not record one).
                return self.finalize_parsed_value(
                    parsed?,
                    repairer_active,
                    resolved_schema.as_ref(),
                    path,
                    finalize_schema,
                    false,
                );
            }
            // If everything else fails, we just ignore and move on.
            self.index += 1;
        }
    }

    /// json_parser.py's `_resolve_schema_for_parse`: decide whether this
    /// `parse_json` call is schema-guided, and with which schema. The
    /// repairer engages only when real guidance was passed (None and a
    /// `true` schema mean "no constraints"); `$ref` chains resolve here so
    /// the container parsers see the target schema. Returns
    /// (repairer-active, schema-to-use). The resolved schema is CLONED out
    /// of the repairer because a `$ref` target borrows the repairer's root
    /// while every consumer below needs `&mut self`.
    fn resolve_schema_for_parse(
        &self,
        schema: Option<&Value>,
    ) -> Result<(bool, Option<Value>), String> {
        let Some(repairer) = self.schema_repairer.as_ref() else {
            return Ok((false, schema.cloned()));
        };
        let Some(schema_value) = schema else {
            return Ok((false, schema.cloned()));
        };
        if matches!(schema_value, Value::Bool(true)) {
            return Ok((false, schema.cloned()));
        }
        match repairer.resolve_schema(schema_value)? {
            // Resolved to `true`: no guidance. Upstream keeps the resolved
            // schema and drops the repairer; the schema value is dead
            // weight from here on either way. A `false` resolution has no
            // ResolvedSchema variant — the repairer maps it to the
            // "Schema does not allow any values." Err carried by `?` above,
            // which is exactly upstream's raise for it.
            ResolvedSchema::True => Ok((false, Some(Value::Bool(true)))),
            ResolvedSchema::Schema(resolved) => Ok((true, Some(resolved.clone()))),
        }
    }

    /// json_parser.py's `_finalize_parsed_value`: the common tail of every
    /// parse_json branch — record that a value was found (the salvage
    /// loop's flag), and when schema-guided and finalizing, run the schema
    /// layer's repair over the value.
    fn finalize_parsed_value(
        &mut self,
        value: Value,
        repairer_active: bool,
        schema: Option<&Value>,
        path: &str,
        finalize_schema: bool,
        record_top_level_value: bool,
    ) -> Result<Value, String> {
        if record_top_level_value {
            self.last_parse_found_value = true;
        }
        if !repairer_active || !finalize_schema {
            return Ok(value);
        }
        self.with_repairer(|repairer| match (repairer, schema) {
            (Some(repairer), Some(schema)) => repairer.repair_value(value, schema, path),
            // Unreachable: the resolver only reports an active repairer
            // with a resolved schema in hand. Kept panic-free rather than
            // unreachable!() — this parser is a fuzz target.
            _ => Ok(value),
        })
    }

    /// Take the schema repairer out of `self`, run `f` on it, put it back.
    /// Every schema-layer call needs `&mut SchemaRepairer` while the parser
    /// state around it needs `&mut self` — take/put-back is the only shape
    /// that satisfies both without cloning the repairer. Restoring it on
    /// every path (including `f`'s Err) matters: `repair()` still needs the
    /// repairer afterwards for its final validation.
    fn with_repairer<T>(&mut self, f: impl FnOnce(Option<&mut SchemaRepairer>) -> T) -> T {
        let mut taken = std::mem::take(&mut self.schema_repairer);
        let out = f(taken.as_mut());
        self.schema_repairer = taken;
        out
    }

    /// parse_number.py's `parse_number`: a "number" is whatever run of
    /// NUMBER_CHARS sits at the cursor — then the fallbacks sort out the
    /// runs that only look like numbers (currency "1,000", fractions
    /// "1/2", ranges "10-20", trailing separators), which come back as
    /// strings.
    fn parse_number(&mut self) -> Result<Value, String> {
        let mut number_str = String::new();
        let is_array = self.ctx_current() == Some(Ctx::Array);
        while let Some(ch) = self.cur() {
            if !NUMBER_CHARS.contains(&ch) || (is_array && ch == ',') {
                break;
            }
            // Python digit-group underscores are consumed but not kept.
            if ch != '_' {
                number_str.push(ch);
            }
            self.index += 1;
        }
        if self.cur().is_some_and(|ch| ch.is_alphabetic()) {
            // This was a string instead, sorry. Rewind over the accumulated
            // characters — by len(number_str), NOT by the consumed-run
            // length: the cursor lands len(underscores) PAST the run
            // start, so skipped underscores end up inside the reparsed
            // string ('{"k": 12_abc}' -> "2_abc") — upstream's exact
            // arithmetic ('{"k": 12_abc}' keeps the underscore), kept.
            self.index -= number_str.chars().count();
            return self.parse_string();
        }
        if matches!(
            number_str.chars().last(),
            Some('-') | Some('e') | Some('E') | Some('/') | Some(',')
        ) {
            // The number ends with a character that cannot end a number or
            // currency: rolling back one.
            number_str.pop();
            self.index -= 1;
        }
        if number_str.contains(',') {
            return Ok(Value::Str(number_str));
        }
        if number_str.contains('.') || number_str.contains('e') || number_str.contains('E') {
            return match number_str.parse::<f64>() {
                Ok(f) => Ok(Value::Float(f)),
                Err(_) => Ok(Value::Str(number_str)),
            };
        }
        match number_str.parse::<i64>() {
            Ok(n) => Ok(Value::Int(n)),
            // Python's int() is unbounded: an out-of-i64-range run becomes
            // BigInt on normalized decimal text; empty/garbage runs fall
            // back to the string itself.
            Err(_) => match normalize_big_int_text(&number_str) {
                Some(text) => Ok(Value::BigInt(text)),
                None => Ok(Value::Str(number_str)),
            },
        }
    }

    /// parse_comment.py's `parse_comment`: skip code-like comments —
    /// `# ...`, `// ...`, `/* ... */` — and return an empty string so they
    /// do not interfere with the actual JSON elements. At the top level
    /// (empty context) all consecutive comments are consumed here and
    /// `parse_json` is re-entered ONCE afterwards; inside a container the
    /// comment is skipped and the caller carries on from the terminator.
    pub(crate) fn parse_comment(&mut self, record_top_level_value: bool) -> Result<Value, String> {
        self.comment_depth += 1;
        let result = if self.comment_depth > MAX_NESTING {
            Err("Input nesting exceeds the supported parser recursion depth.".into())
        } else {
            self.parse_comment_inner(record_top_level_value)
        };
        self.comment_depth -= 1;
        result
    }

    fn parse_comment_inner(&mut self, record_top_level_value: bool) -> Result<Value, String> {
        loop {
            let ch = self.cur();
            let mut termination_characters = vec!['\n', '\r'];
            if self.ctx_has(Ctx::Array) {
                termination_characters.push(']');
            }
            if self.ctx_has(Ctx::ObjectValue) {
                termination_characters.push('}');
            }
            if self.ctx_has(Ctx::ObjectKey) {
                termination_characters.push(':');
            }
            match ch {
                // Line comment starting with '#': runs to the newline or a
                // container terminator (the terminator itself is left in
                // place for the caller).
                Some('#') => {
                    while self
                        .cur()
                        .is_some_and(|c| !termination_characters.contains(&c))
                    {
                        self.index += 1;
                    }
                }
                Some('/') => match self.get(1) {
                    // Line comment starting with '//': these terminate on
                    // newlines only, ignoring the container terminators.
                    Some('/') => {
                        self.index += 2;
                        while self.cur().is_some_and(|c| c != '\n' && c != '\r') {
                            self.index += 1;
                        }
                    }
                    // Block comment '/* ... */' — or to the end of the
                    // string when the closer never comes.
                    Some('*') => {
                        self.index += 2;
                        // Upstream checks the accumulated comment text for
                        // endswith("*/"); tracking the previous character
                        // (seeded with the opener's '*') is that check
                        // exactly — including the quirk that "/*/" closes
                        // on its third character.
                        let mut prev = '*';
                        // Unclosed block comment at end-of-string.
                        while let Some(c) = self.cur() {
                            self.index += 1;
                            if prev == '*' && c == '/' {
                                break;
                            }
                            prev = c;
                        }
                    }
                    // A standalone '/' that is not part of a comment: skip
                    // it so the scan cannot loop on the same character.
                    _ => {
                        self.index += 1;
                    }
                },
                _ => {}
            }
            if self.ctx_empty() {
                // Avoid a parse_json -> parse_comment -> parse_json chain
                // for long runs of top-level comments; re-enter only once
                // after consuming them all.
                self.skip_whitespaces();
                if matches!(self.cur(), Some('#') | Some('/')) {
                    continue;
                }
                return self.parse_json(None, "$", true, record_top_level_value);
            }
            break;
        }
        Ok(Value::Str(String::new()))
    }

    /// Hand the schema repairer back to `repair()` for the final validation
    /// (and diagnostics collection) after parsing.
    pub(crate) fn take_repairer(&mut self) -> Option<SchemaRepairer> {
        std::mem::take(&mut self.schema_repairer)
    }

    /// `self.index < len(self.json_str)` — is there input left to parse?
    fn has_more_input(&self) -> bool {
        self.index < self.s.len()
    }
}

/// The BigInt fallback for `parse_number`'s int path: Python's `int()` is
/// unbounded, so tors holds out-of-i64-range integers as normalized decimal
/// text (see `Value::BigInt`). A pure sign+digits run normalizes (optional
/// '-', digits, no leading zeros, `-0`/all-zeros -> "0"); anything else
/// (empty, a lone '-', interleaved garbage like "1-2") returns None so the
/// caller falls back to the raw string — Python's ValueError branch.
/// Shared with the schema layer's exact string→integer coercion.
pub(crate) fn normalize_big_int_text(number_str: &str) -> Option<String> {
    let (negative, digits) = match number_str.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, number_str),
    };
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let magnitude = digits.trim_start_matches('0');
    if magnitude.is_empty() {
        return Some("0".to_string());
    }
    Some(if negative {
        format!("-{magnitude}")
    } else {
        magnitude.to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(raw: &str) -> Value {
        Parser::new(raw, false, None)
            .parse()
            .expect("plain parse should not fail")
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

    #[test]
    fn parse_number_basics() {
        // test_parse_number.py::test_parse_number
        // "1" and "1.2" are valid JSON, so upstream's answers come from the
        // json.loads fast path; the same number runs are exercised here by
        // calling parse_number in the object-value context parse_json
        // would have used for them.
        let mut p = Parser::new("1", false, None);
        p.ctx_push(Ctx::ObjectValue);
        assert_eq!(p.parse_number(), Ok(Value::Int(1)));
        let mut p = Parser::new("1.2", false, None);
        p.ctx_push(Ctx::ObjectValue);
        assert_eq!(p.parse_number(), Ok(Value::Float(1.2)));
        assert_eq!(
            parse_ok(r#"{"value": 82_461_110}"#),
            obj(&[("value", Value::Int(82_461_110))])
        );
        assert_eq!(
            parse_ok(r#"{"value": 1_234.5_6}"#),
            obj(&[("value", Value::Float(1234.56))])
        );
    }

    #[test]
    fn parse_number_int_overflow_becomes_normalized_bigint() {
        // tors-native pin: Python's unbounded int() past i64's range
        // (upstream has no such case; i64::MAX + 1 and i64::MIN - 1)
        let mut p = Parser::new("9223372036854775808", false, None);
        p.ctx_push(Ctx::ObjectValue);
        assert_eq!(
            p.parse_number(),
            Ok(Value::BigInt("9223372036854775808".into()))
        );
        let mut p = Parser::new("-9223372036854775809", false, None);
        p.ctx_push(Ctx::ObjectValue);
        assert_eq!(
            p.parse_number(),
            Ok(Value::BigInt("-9223372036854775809".into()))
        );
    }

    #[test]
    fn parse_number_edge_cases() {
        // test_parse_number.py::test_parse_number_edge_cases
        assert_eq!(
            parse_ok(r#" - { "test_key": ["test_value", "test_value2"] }"#),
            obj(&[(
                "test_key",
                Value::Array(vec![s("test_value"), s("test_value2")])
            )])
        );
        assert_eq!(parse_ok(r#"{"key": 1/3}"#), obj(&[("key", s("1/3"))]));
        assert_eq!(
            parse_ok(r#"{"key": .25}"#),
            obj(&[("key", Value::Float(0.25))])
        );
        assert_eq!(
            parse_ok(r#"{"here": "now", "key": 1/3, "foo": "bar"}"#),
            obj(&[("here", s("now")), ("key", s("1/3")), ("foo", s("bar"))])
        );
        assert_eq!(
            parse_ok(r#"{"key": 12345/67890}"#),
            obj(&[("key", s("12345/67890"))])
        );
        assert_eq!(
            parse_ok("[105,12"),
            Value::Array(vec![Value::Int(105), Value::Int(12)])
        );
        assert_eq!(parse_ok(r#"{"key", 105,12,"#), obj(&[("key", s("105,12"))]));
        assert_eq!(
            parse_ok(r#"{"key": 1/3, "foo": "bar"}"#),
            obj(&[("key", s("1/3")), ("foo", s("bar"))])
        );
        assert_eq!(parse_ok(r#"{"key": 10-20}"#), obj(&[("key", s("10-20"))]));
        assert_eq!(parse_ok(r#"{"key": 1.1.1}"#), obj(&[("key", s("1.1.1"))]));
        assert_eq!(parse_ok("[- "), Value::Array(vec![]));
        assert_eq!(
            parse_ok(r#"{"key": 1. }"#),
            obj(&[("key", Value::Float(1.0))])
        );
        assert_eq!(
            parse_ok(r#"{"key": 1e10 }"#),
            obj(&[("key", Value::Float(1e10))])
        );
        assert_eq!(parse_ok(r#"{"key": 1e }"#), obj(&[("key", Value::Int(1))]));
        assert_eq!(
            parse_ok(r#"{"key": 1notanumber }"#),
            obj(&[("key", s("1notanumber"))])
        );
        assert_eq!(
            parse_ok(r#"{"rowId": 57eeeeb1-450b-482c-81b9-4be77e95dee2}"#),
            obj(&[("rowId", s("57eeeeb1-450b-482c-81b9-4be77e95dee2"))])
        );
        assert_eq!(
            parse_ok(r#"[1, 2notanumber]"#),
            Value::Array(vec![Value::Int(1), s("2notanumber")])
        );
    }

    #[test]
    fn parse_comment_cases() {
        // test_parse_comment.py::test_parse_comment
        assert_eq!(parse_ok("/"), s(""));
        assert_eq!(
            parse_ok(r#"/* comment */ {"key": "value"}"#),
            obj(&[("key", s("value"))])
        );
        // A // line comment with no newline swallows the rest of the line,
        // closing braces included.
        assert_eq!(
            parse_ok(r#"{ "key": { "key2": "value2" // comment }, "key3": "value3" }"#),
            obj(&[("key", obj(&[("key2", s("value2"))]))])
        );
        assert_eq!(
            parse_ok("{ \"key\": { \"key2\": \"value2\" // comment\n}, \"key3\": \"value3\" }"),
            obj(&[
                ("key", obj(&[("key2", s("value2"))])),
                ("key3", s("value3")),
            ])
        );
        assert_eq!(
            parse_ok(r#"{ "key": { "key2": "value2" # comment }, "key3": "value3" }"#),
            obj(&[
                ("key", obj(&[("key2", s("value2"))])),
                ("key3", s("value3")),
            ])
        );
        assert_eq!(
            parse_ok(r#"{ "key": { "key2": "value2" /* comment */ }, "key3": "value3" }"#),
            obj(&[
                ("key", obj(&[("key2", s("value2"))])),
                ("key3", s("value3")),
            ])
        );
        assert_eq!(
            parse_ok(r#"[ "value", /* comment */ "value2" ]"#),
            Value::Array(vec![s("value"), s("value2")])
        );
        // An unclosed block comment at end-of-input is consumed silently.
        assert_eq!(
            parse_ok(r#"{ "key": "value" /* comment"#),
            obj(&[("key", s("value"))])
        );
    }

    #[test]
    fn line_comment_brackets_do_not_trigger_empty_object_array_fallback() {
        // test_parse_comment.py::test_line_comment_brackets_do_not_trigger_empty_object_array_fallback
        // (log-text assertions skipped: logs are not ported)
        assert_eq!(parse_ok("{\n// comment ]\n}"), obj(&[]));
    }

    #[test]
    fn block_comment_brackets_do_not_trigger_empty_object_array_fallback() {
        // test_parse_comment.py::test_block_comment_brackets_do_not_trigger_empty_object_array_fallback
        // (log-text assertions skipped: logs are not ported)
        assert_eq!(parse_ok("{/* comment ] */}"), obj(&[]));
    }

    #[test]
    fn line_comment_brackets_do_not_close_array_items() {
        // test_parse_comment.py::test_line_comment_brackets_do_not_close_array_items
        let raw = r#"
    {
        "Changes": [
            //object a
            {
                "Action": "1"
            },
            //object b ]
            {
                "Action": "2"
            },
            //object c ]
            {
                "Action": "3"
            }
        ]
    }
    "#;
        assert_eq!(
            parse_ok(raw),
            obj(&[(
                "Changes",
                Value::Array(vec![
                    obj(&[("Action", s("1"))]),
                    obj(&[("Action", s("2"))]),
                    obj(&[("Action", s("3"))]),
                ])
            )])
        );
    }

    #[test]
    fn many_top_level_comments_parse_linearly() {
        // test_parse_comment.py::test_parse_many_top_level_comments_without_recursion_error
        // (log-text assertions skipped: logs are not ported): the comment
        // loop is iterative and re-enters parse_json once, so 600 leading
        // comments cost 600 loop iterations, not 600 stack frames.
        let raw = format!("{}{}", "# comment\n".repeat(600), r#"{"key": "value"}"#);
        assert_eq!(parse_ok(&raw), obj(&[("key", s("value"))]));
    }

    #[test]
    fn multiple_jsons() {
        // test_json_repair.py::test_multiple_jsons
        assert_eq!(parse_ok("[]{}"), Value::Array(vec![]));
        assert_eq!(
            parse_ok(r#"[]{"key":"value"}"#),
            obj(&[("key", s("value"))])
        );
        assert_eq!(
            parse_ok(r#"{"key":"value"}[1,2,3,True]"#),
            Value::Array(vec![
                obj(&[("key", s("value"))]),
                Value::Array(vec![
                    Value::Int(1),
                    Value::Int(2),
                    Value::Int(3),
                    Value::Bool(true),
                ]),
            ])
        );
        assert_eq!(
            parse_ok(r#"{"key":"value"}, {"key":"value_after"}"#),
            Value::Array(vec![
                obj(&[("key", s("value"))]),
                obj(&[("key", s("value_after"))]),
            ])
        );
        assert_eq!(
            parse_ok(r#"lorem ```json {"key":"value"} ``` ipsum ```json [1,2,3,True] ``` 42"#),
            Value::Array(vec![
                obj(&[("key", s("value"))]),
                Value::Array(vec![
                    Value::Int(1),
                    Value::Int(2),
                    Value::Int(3),
                    Value::Bool(true),
                ]),
            ])
        );
        // The second array is the same object shape as the first: an
        // update, so the newest replaces it and the list unwraps.
        assert_eq!(
            parse_ok(r#"[{"key":"value"}][{"key":"value_after"}]"#),
            Value::Array(vec![obj(&[("key", s("value_after"))])])
        );
    }

    #[test]
    fn top_level_separator_detects_pending_comma() {
        // test_json_repair.py::test_top_level_separator_detects_pending_comma
        let p = Parser::new(r#" , {"key": "value"}"#, false, None);
        assert!(p.next_top_level_value_is_comma_separated());
    }

    #[test]
    fn initial_container_trailing_content_rejects_mismatched_delimiters() {
        // test_json_repair.py::test_initial_container_trailing_content_rejects_mismatched_delimiters
        let p = Parser::new("{]", false, None);
        assert!(!p.initial_container_has_non_comma_trailing_content());
    }

    #[test]
    fn deep_nesting_raises_the_normalized_recursion_error() {
        // test_json_repair.py::test_repair_json_normalizes_real_parser_recursion_error,
        // pinned at fixed depth 500 (upstream probes for its runtime's
        // RecursionError threshold; tors raises at MAX_NESTING)
        let payload = format!("{}1{}", "{a: [".repeat(500), "]}".repeat(500));
        assert_eq!(
            Parser::new(&payload, false, None).parse(),
            Err("Input nesting exceeds the supported parser recursion depth.".to_string())
        );
    }

    #[test]
    fn garbage_separated_comment_runs_raise_instead_of_overflowing_the_stack() {
        // '/x' chains parse_json ↔ parse_comment without unwinding — 2
        // frames per 2 chars, unbounded in a naive port. Upstream's
        // RecursionError (→ ValueError) fires near 500 pairs; tors
        // normalizes at MAX_NESTING. Sizes chosen well past the crash
        // threshold a build without the guard would hit (~11k pairs).
        // ('/ ' does NOT belong here: whitespace-separated runs are
        // absorbed by parse_comment's own loop and never recurse.)
        for pattern in ["/x", "/*", "a/"] {
            let payload = pattern.repeat(12_000);
            assert_eq!(
                Parser::new(&payload, false, None).parse(),
                Err("Input nesting exceeds the supported parser recursion depth.".to_string())
            );
        }
    }

    #[test]
    fn strict_mode_rejects_multiple_top_level_elements() {
        // tors-native pin for json_parser.py's strict raise inside
        // _parse_top_level (upstream exercises it via test_strict_mode.py)
        assert_eq!(
            Parser::new(r#"{"a": 1}{"b": 2}"#, true, None).parse(),
            Err("Multiple top-level JSON elements found in strict mode.".to_string())
        );
    }
}
