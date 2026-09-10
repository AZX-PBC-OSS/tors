//! The string-repair heuristics: a port of json_repair's `parse_string.py`
//! plus its `parse_string_helpers/` (object_value_context.py,
//! parse_boolean_or_null.py, parse_json_llm_block.py): upstream:
//! https://github.com/mangiucugna/json_repair by Stefano Baccianella, MIT,
//! pinned at commit 251d141786d0f6ff561f6ec04d90188a338e2470 (version
//! 0.63.4). This is the heart of the repair parser: the 900-odd lines of
//! heuristics that decide, for every quote/comma/brace the cursor meets,
//! whether it closes the current string, belongs to it, or starts the next
//! object member. Every branch is ported 1:1 in upstream's order of checks.
//!
//! # Porting notes (the decisions this file's dynamics forced)
//!
//! - `self.log(...)` sites upstream become rationale comments here (the log
//!   text is not ported; the heuristic it explained gets the comment), per
//!   the port contract in `parser.rs`'s docs.
//! - Python's `NO_DIRECT_RESULT` sentinel object becomes the [`StringEntry`]
//!   enum: `_prepare_string_entry` returns either a direct `Value` (any of
//!   upstream's non-sentinel results: the comment parse's value, the empty
//!   string, the fast-path value, the boolean/null literal, the LLM block's
//!   value) or the state to keep scanning with.
//! - `stream_stable` is not ported (design doc §9.1) and upstream defaults
//!   it to `False`, so every `not self.stream_stable` guard here is
//!   constantly true and the one `if self.stream_stable and ...` tail-trim
//!   is constantly false: both kept in code as [`STREAM_STABLE`] reads so
//!   the branch structure stays upstream's.
//! - Python truthiness on `get_char_at()` results: `None` is falsy and any
//!   one-character string is truthy: including `"\0"` (a non-empty str is
//!   truthy in CPython). Every `while char and ...` / `if char:` site is
//!   therefore ported as `Option::is_some`/`is_some_and` with no '\0'
//!   exclusion; a NUL in the input is an ordinary character exactly as
//!   upstream treats it.
//! - `str.isdigit()` sites use `char::is_ascii_digit` (documented divergence
//!   §9.3: Python's isdigit accepts non-ASCII Nd digits); `str.isalnum` uses
//!   `char::is_alphanumeric`, `str.isspace` uses
//!   `crate::normalize_impl::is_py_whitespace`.
//! - `\uXXXX` escapes decoding to lone surrogates become U+FFFD (§9.2: a
//!   Rust `String` cannot hold a lone surrogate); every other 4-hex-digit
//!   value is a valid codepoint.
//! - parse_string.py contains no `with self.context.enter(...)` regions:
//!   all context usage here is read-only (`ctx_current`/`ctx_has`); the
//!   pushes live in the object/array callers.
//! - Python's `char` local is `ch` here (naming only).

use super::Value;
use super::parser::{Ctx, LookaheadEntry, LookaheadKey, Parser};
use crate::normalize_impl::is_py_whitespace;

/// parse_string.py's `stream_stable`: not ported (§9.1) and always `False`
/// upstream, so the guards that read it are constant. Kept as a named const
/// so each guard site stays visibly upstream's branch.
const STREAM_STABLE: bool = false;

/// parse_string.py's `INLINE_CONTAINER_OPENERS` = `tuple(["[", "{", "("])`
/// (the keys of INLINE_CONTAINER_CLOSING_DELIMITERS, in dict order).
const INLINE_CONTAINER_OPENERS: [char; 3] = ['[', '{', '('];

/// parse_string.py's `LOW_SMART_QUOTE_SENTINEL`: the rstring-delimiter stack
/// entry marking an open `„...”` span. '\0' is safe as the sentinel because
/// no scan target is ever '\0' (and because a NUL in the input is just an
/// ordinary character: see the truthiness note in the module docs).
const LOW_SMART_QUOTE_SENTINEL: char = '\0';

/// parse_json_llm_block.py's literal opener match: `json_str[index..index+7]
/// == "```json"`, case-sensitive, no space. Deliberately not
/// CommonMark-ified (documented non-divergence: upstream's own rule).
const JSON_LLM_BLOCK_OPENER: [char; 7] = ['`', '`', '`', 'j', 's', 'o', 'n'];

/// parse_string.py's `STRING_DELIMITERS` membership test.
fn is_string_delimiter(c: char) -> bool {
    super::STRING_DELIMITERS.contains(&c)
}

/// parse_string.py's `INLINE_CONTAINER_CLOSING_DELIMITERS` (`{"[": "]", "{":
/// "}", "(": ")"}`): opener -> closer, `None` when `c` is not an opener.
fn inline_container_closer(c: char) -> Option<char> {
    match c {
        '[' => Some(']'),
        '{' => Some('}'),
        '(' => Some(')'),
        _ => None,
    }
}

/// parse_string.py's `_matching_string_delimiter`: the left curly quote's
/// match is the right curly quote; every other delimiter matches itself.
fn matching_string_delimiter(delimiter: char) -> char {
    if delimiter == '\u{201C}' {
        '\u{201D}'
    } else {
        delimiter
    }
}

/// Python's `.lower()` on a single character for literal comparisons
/// (`char.lower() in ["t", "f", "n"]`): the first codepoint of the full
/// lowercase mapping.
fn lower_char(c: char) -> char {
    c.to_lowercase().next().unwrap_or(c)
}

/// Python's `str.rstrip()` (no args): drop trailing `str.isspace()` chars.
fn rstrip_py(s: &mut String) {
    while s.chars().next_back().is_some_and(is_py_whitespace) {
        s.pop();
    }
}

/// parse_string.py's `StringParseState` dataclass. Field-for-field, with the
/// two representation decisions documented on the fields that carry them.
#[derive(Clone, Debug)]
struct StringParseState {
    missing_quotes: bool,
    doubled_quotes: bool,
    lstring_delimiter: char,
    /// Upstream models this as a `str` that grows/shrinks: element 0 is the
    /// string's outer right delimiter; `_push_low_smart_quote_span` appends
    /// `'\0'` sentinel entries while inside `„...”` spans and the active
    /// delimiter is always the last element. A `Vec<char>` with the same
    /// push/pop semantics is the Rust shape.
    rstring_delimiter: Vec<char>,
    string_acc: String,
    unmatched_delimiter: bool,
    pending_inline_container: bool,
    inline_container_stack: Vec<char>,
    object_value_has_no_future_delimiter: bool,
    object_value_unmatched_opening_braces: usize,
    /// Lazy whitespace probe for the open regex character class: the
    /// accumulator is scanned for py-whitespace only from
    /// `class_ws_probed` (the byte length already probed) forward, at
    /// query time in quote_belongs_to_regex_character_class, OR-ing into
    /// `class_ws_found`. The class region is exactly
    /// `[class_start, string_acc.len())`: it only ever grows by
    /// appends, so probing the new tail per query is amortized O(1) and
    /// the append path stays byte-for-byte upstream's (upstream rescans
    /// the whole tail per closing-quote candidate, O(n^2) on
    /// `'{"a": "[' + 'x"'*n + '"}'`; an eagerly maintained flag was
    /// tried and measurably taxed every appended char). Reset when a
    /// class opens; the escape-tail rewrites clamp `class_ws_probed`
    /// back to the popped length.
    class_ws_found: bool,
    class_ws_probed: usize,
    /// Offset just past the last `[` in `string_acc`. Upstream tracks
    /// codepoint indices; we track byte offsets into the `String`: a
    /// representation-only choice: every producer and consumer of this
    /// index (append, rebuild, the whitespace probe in
    /// quote_belongs_to_regex_character_class) uses the same unit, so the
    /// observable behavior is identical.
    regex_character_class_start: Option<usize>,
    /// Walk-outcome memo for handle_right_delimiter_candidate's Array
    /// delimiter-pairing: candidate quote position (absolute, the parser's
    /// index at the candidate) -> the `even_delimiters` verdict a fresh
    /// pairing walk from there would return. Populated by every walk for
    /// each quote it passes, so an internal-quote run costs one walk
    /// instead of one per quote (`'["' + 'a"'*n + '"]'` was O(n^2)).
    /// Validity: the verdict is a pure function of (buffer, position,
    /// outer, the frozen context stack): all fixed for one string parse
    /// (the context stack never changes inside scan_string_body, and the
    /// buffer is only spliced by parse_object after the state is dropped),
    /// so an entry is exact whenever its position becomes a candidate.
    /// Not ported from upstream (upstream re-walks per quote; see the
    /// array_pairing_walk docs for why the memo is exact).
    ///
    /// Shape: a position-sorted Vec, binary-searched: candidates are
    /// consulted on every array-context delimiter candidate (HashMap
    /// hashing measurably taxed the fenced-repair benchmark), and the
    /// entries of one walk append in ascending order, so the common
    /// insert is a plain push and the empty case (most strings never
    /// pair) is one length check.
    pairing_outcomes: Vec<(usize, bool)>,
}

impl Default for StringParseState {
    fn default() -> Self {
        Self::new()
    }
}

impl StringParseState {
    /// The dataclass defaults.
    fn new() -> Self {
        StringParseState {
            missing_quotes: false,
            doubled_quotes: false,
            lstring_delimiter: '"',
            rstring_delimiter: vec!['"'],
            string_acc: String::new(),
            unmatched_delimiter: false,
            pending_inline_container: false,
            inline_container_stack: Vec::new(),
            object_value_has_no_future_delimiter: false,
            object_value_unmatched_opening_braces: 0,
            class_ws_found: false,
            class_ws_probed: 0,
            regex_character_class_start: None,
            pairing_outcomes: Vec::new(),
        }
    }

    /// parse_string.py's `_outer_rstring_delimiter`: `rstring_delimiter[0]`.
    fn outer_rstring_delimiter(&self) -> char {
        self.rstring_delimiter[0]
    }

    /// parse_string.py's `_active_rstring_delimiter`: `rstring_delimiter[-1]`.
    fn active_rstring_delimiter(&self) -> char {
        *self.rstring_delimiter.last().unwrap_or(&'"')
    }

    /// parse_string.py's `_in_low_smart_quote_span`.
    fn in_low_smart_quote_span(&self) -> bool {
        self.active_rstring_delimiter() == LOW_SMART_QUOTE_SENTINEL
    }

    /// parse_string.py's `_push_low_smart_quote_span`.
    fn push_low_smart_quote_span(&mut self) {
        self.rstring_delimiter.push(LOW_SMART_QUOTE_SENTINEL);
    }

    /// parse_string.py's `_pop_low_smart_quote_span` (`rstring_delimiter =
    /// rstring_delimiter[:-1]`); only ever called while a sentinel entry is
    /// on top, so the pop drops exactly that.
    fn pop_low_smart_quote_span(&mut self) {
        self.rstring_delimiter.pop();
    }

    /// Binary-search the pairing-outcome memo (see the field docs: entries
    /// are position-sorted; the empty case, most strings, is one check).
    fn pairing_outcome(&self, pos: usize) -> Option<bool> {
        let idx = self.pairing_outcomes.partition_point(|(p, _)| *p < pos);
        self.pairing_outcomes
            .get(idx)
            .filter(|(p, _)| *p == pos)
            .map(|&(_, v)| v)
    }
}

/// The pairing-outcome memo's write side (field docs on
/// `StringParseState::pairing_outcomes`): a walk's records arrive in
/// ascending position order, so the common case extends the vec; a
/// defensively unsorted position (no reachable case is known: within one
/// string parse, walk-start candidates sit past every recorded position)
/// falls back to a sorted insert rather than breaking the search order.
fn record_pairing_outcome(entries: &mut Vec<(usize, bool)>, pos: usize, verdict: bool) {
    if let Some(&(last_pos, _)) = entries.last()
        && pos > last_pos
    {
        entries.push((pos, verdict));
        return;
    }
    let idx = entries.partition_point(|(p, _)| *p < pos);
    if entries.get(idx).is_some_and(|&(p, _)| p == pos) {
        entries[idx].1 = verdict;
    } else {
        entries.insert(idx, (pos, verdict));
    }
}

/// Build the fixed-size lookahead-cache key from a target slice: the targets
/// in call order, '\0'-padded (upstream's dict keys are the target tuples;
/// no target is ever '\0').
fn lookahead_key(targets: &[char]) -> LookaheadKey {
    debug_assert!(targets.len() <= 6);
    let mut key = [LOW_SMART_QUOTE_SENTINEL; 6];
    for (slot, target) in key.iter_mut().zip(targets.iter()) {
        *slot = *target;
    }
    key
}

/// parse_string.py's `NO_DIRECT_RESULT` flow: `_prepare_string_entry`
/// returns either a finished value (upstream's non-sentinel results: see
/// the module docs) or the state that `parse_string` keeps scanning with.
enum StringEntry {
    Direct(Value),
    Scan(StringParseState),
}

/// parse_string.py's `_append_string_content`: append content to the
/// accumulator while maintaining the unmatched-brace count and the regex
/// character-class start (see the field docs for the byte-unit choice).
/// The class's whitespace probe is deliberately not maintained here:
/// see `class_ws_found`'s docs for the lazy incremental design.
fn append_string_content(state: &mut StringParseState, content: &[char]) {
    let start_byte = state.string_acc.len();
    let mut byte_off = 0usize;
    for &ch in content {
        state.string_acc.push(ch);
        if ch == '{' {
            state.object_value_unmatched_opening_braces += 1;
        } else if ch == '}' && state.object_value_unmatched_opening_braces > 0 {
            state.object_value_unmatched_opening_braces -= 1;
        } else if ch == '[' {
            // just past the '[' ('[' is one byte, so +1 is the byte unit's
            // equivalent of upstream's codepoint `+ 1`); the class region
            // starts empty here, so the lazy probe starts from scratch.
            state.regex_character_class_start = Some(start_byte + byte_off + 1);
            state.class_ws_found = false;
            state.class_ws_probed = start_byte + byte_off + 1;
        } else if ch == ']' {
            state.regex_character_class_start = None;
            state.class_ws_found = false;
            state.class_ws_probed = 0;
        }
        byte_off += ch.len_utf8();
    }
}

/// parse_string.py's `_rebuild_unmatched_opening_braces`: recompute the
/// brace count and class start from the whole accumulator. (After the
/// escape normalizer switched to incremental tail rewrites, every live
/// caller is gone: this remains for the const-false STREAM_STABLE tail
/// trim, keeping the ported shape.) The probe resets to "nothing probed
/// within the recomputed class", so a next query would rescan the whole
/// class region: correct if ever reached.
fn rebuild_unmatched_opening_braces(state: &mut StringParseState) {
    state.object_value_unmatched_opening_braces = 0;
    state.regex_character_class_start = None;
    state.class_ws_found = false;
    state.class_ws_probed = 0;
    for (index, ch) in state.string_acc.char_indices() {
        if ch == '{' {
            state.object_value_unmatched_opening_braces += 1;
        } else if ch == '}' && state.object_value_unmatched_opening_braces > 0 {
            state.object_value_unmatched_opening_braces -= 1;
        } else if ch == '[' {
            state.regex_character_class_start = Some(index + 1);
            state.class_ws_found = false;
            state.class_ws_probed = index + 1;
        } else if ch == ']' {
            state.regex_character_class_start = None;
            state.class_ws_found = false;
            state.class_ws_probed = 0;
        }
    }
}

/// The escape normalizer's tail rewrite: pop the just-appended backslash
/// (counter-neutral: a backslash affects no brace/class/whitespace
/// bookkeeping) and push the replacement through append_string_content,
/// which maintains every accumulator-derived counter incrementally in
/// O(push). Upstream (and the old port) rebuilt the whole accumulator per
/// rewrite (`_rebuild_unmatched_opening_braces`), O(n) per escape:
/// O(n^2) on escape-dense input like `'["' + (']' + '\\\\')*k + '" x'`,
/// where the interleaved backslash runs normalize once per pair.
fn rewrite_escape_tail(state: &mut StringParseState, replacement: &[char]) {
    state.string_acc.pop();
    // The popped char is a backslash (never probed-found whitespace), so
    // only the probe's watermark needs clamping back to the popped
    // length; the pushed replacement is unprobed new tail.
    state.class_ws_probed = state.class_ws_probed.min(state.string_acc.len());
    append_string_content(state, replacement);
}

/// object_value_context.py threads a `skip_to_character` callable through
/// `classify_object_value_comma`/`_bare_member_has_recoverable_value`:
/// upstream's callable is the lookahead-cached flavor from
/// `_scan_string_body` when the string state is at hand, else the parser's
/// plain method. tors's lookahead memo lives on the parser (see
/// cached_skip_to_character's docs), so the two flavors collapse into one
/// always-memoized callable and the state-vs-default distinction
/// disappears; the struct remains the ported call shape.
struct CommaSkip<'a> {
    parser: &'a mut Parser,
}

impl CommaSkip<'_> {
    fn skip(&mut self, targets: &[char], idx: usize) -> usize {
        self.parser.cached_skip_to_character(targets, idx)
    }
}

/// `[*STRING_DELIMITERS, "}"]`: the member-recovery lookahead target set
/// (upstream builds this list inline; mirrored here in the same order).
const DELIMITERS_PLUS_CLOSE_BRACE: [char; 5] = ['"', '\'', '\u{201C}', '\u{201D}', '}'];

/// `[*STRING_DELIMITERS, "{", "["]`: the next-special lookahead target set.
const DELIMITERS_PLUS_OPENERS: [char; 6] = ['"', '\'', '\u{201C}', '\u{201D}', '{', '['];

/// object_value_context.py's `_bare_member_has_recoverable_value`: can the
/// text after a bare `key:` start a value we could recover as the next
/// member? (If not, the `key:`-looking prose more likely belongs to the
/// current string.)
fn bare_member_has_recoverable_value(parser: &mut Parser, value_idx: usize) -> bool {
    let value_start_idx = parser.scroll_whitespaces(value_idx);
    let value_start = parser.get(value_start_idx as isize);
    if matches!(
        value_start,
        Some(c) if is_string_delimiter(c) || c == '{' || c == '[' || c == '-'
    ) {
        return true;
    }
    // Python `value_start.isdigit()`: Unicode Nd digits upstream, ASCII
    // here (documented divergence §9.3).
    if value_start.is_some_and(|c| c.is_ascii_digit()) {
        return true;
    }

    for literal in ["true", "false", "null"] {
        let mut all_match = true;
        for (offset, expected) in literal.char_indices() {
            if parser.get((value_start_idx + offset) as isize) != Some(expected) {
                all_match = false;
                break;
            }
        }
        if all_match {
            let value_end = parser.get((value_start_idx + literal.chars().count()) as isize);
            if value_end.is_none()
                || value_end.is_some_and(is_py_whitespace)
                || matches!(value_end, Some(',') | Some('}') | Some(']'))
            {
                return true;
            }
        }
    }

    // An unquoted value is only a safe member boundary when its object closes
    // before the current string can close. Otherwise prose such as
    // `, floof: explanation` is more likely to belong to the string.
    // (upstream's own comment on this heuristic)
    let mut skip = CommaSkip {
        parser: &mut *parser,
    };
    let value_end_idx = skip.skip(&DELIMITERS_PLUS_CLOSE_BRACE, value_start_idx);
    parser.get(value_end_idx as isize) == Some('}')
}

/// object_value_context.py's `classify_object_value_comma`, returning one of
/// upstream's literal classification strings. (Upstream threads the string
/// state through for the lookahead cache; tors's parser-level memo makes
/// that threading unnecessary: see CommaSkip's docs.)
fn classify_object_value_comma(parser: &mut Parser) -> &'static str {
    let next_idx = parser.scroll_whitespaces(1);
    let next_c = parser.get(next_idx as isize);
    if matches!(next_c, Some('}') | None) {
        return "member";
    }

    if let Some(c) = next_c
        && is_string_delimiter(c)
    {
        // Upstream uses the parser's plain skip_to_character here (not the
        // threaded callable): kept distinct below, where the callable runs.
        let key_end_idx = parser.skip_to_character(&[c], next_idx + 1);
        if parser.get(key_end_idx as isize).is_none() {
            return "string";
        }
        let key_end_idx = parser.scroll_whitespaces(key_end_idx + 1);
        return if parser.get(key_end_idx as isize) == Some(':') {
            "member"
        } else {
            "string"
        };
    }

    if next_c == Some('`') {
        let mut bare_key_idx = next_idx + 1;
        loop {
            let key_char = parser.get(bare_key_idx as isize);
            if !key_char.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '-') {
                break;
            }
            bare_key_idx += 1;
        }
        let bare_key_idx = parser.scroll_whitespaces(bare_key_idx);
        return if parser.get(bare_key_idx as isize) == Some(':') {
            "member"
        } else {
            "string"
        };
    }

    if next_c.is_some_and(|c| c.is_alphanumeric() || c == '_') {
        let mut bare_key_idx = next_idx;
        loop {
            let key_char = parser.get(bare_key_idx as isize);
            if !key_char.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '-') {
                break;
            }
            bare_key_idx += 1;
        }
        let bare_key_idx = parser.scroll_whitespaces(bare_key_idx);
        if parser.get(bare_key_idx as isize) == Some(':')
            && bare_member_has_recoverable_value(parser, bare_key_idx + 1)
        {
            return "member";
        }
    }

    if matches!(next_c, Some('{') | Some('[')) {
        return "container";
    }

    let mut skip = CommaSkip {
        parser: &mut *parser,
    };
    let next_special_idx = skip.skip(&DELIMITERS_PLUS_OPENERS, next_idx);
    let next_special = skip.parser.get(next_special_idx as isize);
    let Some(next_special) = next_special else {
        return "string_no_future_delimiter";
    };
    if next_special == '{' || next_special == '[' {
        return "string";
    }

    let key_end_idx = skip.skip(&[next_special], next_special_idx + 1);
    if skip.parser.get(key_end_idx as isize).is_none() {
        return "string";
    }
    let key_end_idx = skip.parser.scroll_whitespaces(key_end_idx + 1);
    if skip.parser.get(key_end_idx as isize) == Some(':') {
        "member"
    } else {
        "string"
    }
}

/// object_value_context.py's `update_inline_container_stack`: track nested
/// `{`/`[` opened inside a string value; returns
/// `(pending_inline_container, keep_inline_container_char)`.
fn update_inline_container_stack(
    ch: char,
    pending_inline_container: bool,
    inline_container_stack: &mut Vec<char>,
) -> (bool, bool) {
    if ch == '{' || ch == '[' {
        if pending_inline_container {
            inline_container_stack.push(ch);
            return (false, false);
        }
        if !inline_container_stack.is_empty() {
            inline_container_stack.push(ch);
        }
    }

    if let Some(&top) = inline_container_stack.last()
        && ((ch == '}' && top == '{') || (ch == ']' && top == '['))
    {
        inline_container_stack.pop();
        return (pending_inline_container, true);
    }

    (pending_inline_container, false)
}

impl Parser {
    /// parse_string.py's `parse_string`: the public entry. Always returns a
    /// `Value::Str` from the scan path (upstream's cast); the direct-result
    /// paths can return any JSON value (the LLM-block parse, the boolean /
    /// null literals, the comment delegate).
    pub(crate) fn parse_string(&mut self) -> Result<Value, String> {
        let mut state = match self.prepare_string_entry()? {
            StringEntry::Direct(value) => return Ok(value),
            StringEntry::Scan(state) => state,
        };
        let ch = self.scan_string_body(&mut state);
        // Before finalize: a deadline abort broke the scan mid-string, and
        // the sticky error (not the partial state) is the answer.
        if let Some(err) = self.take_deadline_error() {
            return Err(err);
        }
        Ok(Value::Str(self.finalize_string_result(&mut state, ch)))
    }

    /// parse_string_helpers/parse_json_llm_block.py's
    /// `parse_json_llm_block`: extract and normalize JSON enclosed in
    /// ```json ... ``` blocks. Python returns `False` when no block was
    /// found: `Ok(None)` here; a block that parses returns its value.
    pub(crate) fn parse_json_llm_block(&mut self) -> Result<Option<Value>, String> {
        // The opener is the literal match rule (see JSON_LLM_BLOCK_OPENER);
        // a short tail of input simply fails the comparison, as upstream's
        // slice compare does.
        if self.s.get(self.index..self.index + 7) == Some(&JSON_LLM_BLOCK_OPENER[..]) {
            let i = self.skip_to_character(&['`'], 7);
            let closer = self.index + i;
            // The closer must be an unescaped ``` run (skip_to_character
            // stops only at unescaped backticks; the two chars after it are
            // plain equality, exactly upstream's slice check).
            if self.s.get(closer..closer + 3) == Some(&['`', '`', '`'][..]) {
                self.index += 7; // Move past ```json
                return self.parse_json(None, "$", true, false).map(Some);
            }
        }
        Ok(None)
    }

    /// parse_string.py's `_prepare_string_entry`.
    fn prepare_string_entry(&mut self) -> Result<StringEntry, String> {
        let mut ch = self.cur();
        if matches!(ch, Some('#') | Some('/')) {
            return Ok(StringEntry::Direct(self.parse_comment(false)?));
        }

        // Skip garbage before the string's first meaningful character: any
        // char that is neither a delimiter nor alphanumeric. (Truthiness:
        // any char, '\0' included, keeps the loop going upstream.)
        let garbage_start = self.index;
        while ch.is_some_and(|c| !is_string_delimiter(c) && !c.is_alphanumeric()) {
            self.index += 1;
            ch = self.cur();
        }
        // A quote-free tail makes this skip O(remaining) per string parse
        // (the empty-object quadratic's driver): force the next check.
        self.note_scan_distance(self.index - garbage_start);

        let Some(ch) = ch else {
            return Ok(StringEntry::Direct(Value::Str(String::new())));
        };

        let fast_path_value = self.try_parse_simple_quoted_string();
        if let Some(value) = fast_path_value {
            return Ok(StringEntry::Direct(Value::Str(value)));
        }

        let mut state = StringParseState::new();

        if ch == '\'' {
            state.lstring_delimiter = '\'';
            state.rstring_delimiter = vec!['\''];
        } else if ch == '\u{201C}' {
            state.lstring_delimiter = '\u{201C}';
            state.rstring_delimiter = vec!['\u{201D}'];
        } else if ch.is_alphanumeric() {
            if matches!(lower_char(ch), 't' | 'f' | 'n')
                && self.ctx_current() != Some(Ctx::ObjectKey)
            {
                let (found_literal, value) = self.parse_boolean_or_null();
                if found_literal {
                    return Ok(StringEntry::Direct(value));
                }
            }
            // Upstream logs "found a literal instead of a quote" here; the
            // heuristic: an unquoted literal starts a missing-quotes string.
            state.missing_quotes = true;
        }

        if !state.missing_quotes {
            self.index += 1;
        }
        if self.cur() == Some('`') {
            match self.parse_json_llm_block()? {
                Some(value) if !matches!(value, Value::Bool(false)) => {
                    return Ok(StringEntry::Direct(value));
                }
                // Python's `ret_val is not False` is an identity check:
                // a fenced block that parses to the value False is
                // indistinguishable from "no block found" (upstream
                // returns the same False singleton for both), so the
                // string parse continues past it.
                Some(_) => {}
                None => {
                    // Upstream logs "code fences but they did not enclose
                    // valid JSON, continuing parsing the string".
                }
            }
        }

        if self.cur() == Some(state.lstring_delimiter) {
            let cur1 = self.get(1);
            if (self.ctx_current() == Some(Ctx::ObjectKey) && cur1 == Some(':'))
                || (self.ctx_current() == Some(Ctx::ObjectValue)
                    && matches!(cur1, Some(',') | Some('}')))
                || (self.ctx_current() == Some(Ctx::Array) && matches!(cur1, Some(',') | Some(']')))
            {
                self.index += 1;
                return Ok(StringEntry::Direct(Value::Str(String::new())));
            }
            if cur1 == Some(state.lstring_delimiter) {
                // Upstream logs "found a doubled quote and then a quote
                // again, ignoring it".
                if self.strict {
                    return Err("Found doubled quotes followed by another quote.".into());
                }
                return Ok(StringEntry::Direct(Value::Str(String::new())));
            }
            let outer = state.outer_rstring_delimiter();
            let i = self.skip_to_character(&[outer], 1);
            if self.get(i as isize + 1) == Some(outer) {
                // Upstream logs "found a valid starting doubled quote".
                state.doubled_quotes = true;
                self.index += 1;
            } else {
                if self.get(i as isize) == Some(outer) {
                    let outer_quote_idx = self.skip_to_character(&[outer], i + 1);
                    let after_outer_quote_idx = self.scroll_whitespaces(outer_quote_idx + 1);
                    let after_outer_quote = self.get(after_outer_quote_idx as isize);
                    if self.ctx_current() == Some(Ctx::ObjectValue)
                        && !self.only_whitespace_until(i)
                        && self.get(outer_quote_idx as isize) == Some(outer)
                        && matches!(after_outer_quote, Some(',') | Some('}') | None)
                    {
                        // Upstream logs "found a leading quote that starts
                        // a quoted span, keeping it".
                        let lstring = state.lstring_delimiter;
                        append_string_content(&mut state, &[lstring]);
                        state.unmatched_delimiter = true;
                        self.index += 1;
                        return Ok(StringEntry::Scan(state));
                    }
                }
                let i = self.scroll_whitespaces(1);
                let next_c = self.get(i as isize);
                if next_c.is_some_and(|c| is_string_delimiter(c) || c == '{' || c == '[') {
                    // Upstream logs "found a doubled quote but also another
                    // quote afterwards, ignoring it".
                    if self.strict {
                        return Err(
                            "Found doubled quotes followed by another quote while parsing a string."
                                .into(),
                        );
                    }
                    self.index += 1;
                    return Ok(StringEntry::Direct(Value::Str(String::new())));
                }
                if !matches!(next_c, Some(',') | Some(']') | Some('}')) {
                    // Upstream logs "found a doubled quote but it was a
                    // mistake, removing one quote".
                    self.index += 1;
                }
            }
        }

        Ok(StringEntry::Scan(state))
    }

    /// parse_string.py's `_try_parse_simple_quoted_string`: the fast path
    /// for a plain `"..."` value with no escapes and a clean boundary after
    /// it. Upstream has an isinstance(str) branch and a StringFileWrapper
    /// branch; json_fd is not ported (§9.1) so every input takes the str
    /// branch, which is what is ported here.
    fn try_parse_simple_quoted_string(&mut self) -> Option<String> {
        if self.cur() != Some('"') {
            return None;
        }

        let start = self.index + 1;
        let end = self.s[start..].iter().position(|&c| c == '"')? + start;
        let value: String = self.s[start..end].iter().collect();
        if value.contains('\\') || value.contains('\n') || value.contains('\r') {
            return None;
        }

        let mut next_index = end + 1;
        let limit = self.s.len();
        while next_index < limit && is_py_whitespace(self.s[next_index]) {
            next_index += 1;
        }
        let next_char = if next_index < limit {
            Some(self.s[next_index])
        } else {
            None
        };

        match self.ctx_current() {
            Some(Ctx::ObjectKey) => {
                if next_char != Some(':') {
                    return None;
                }
            }
            Some(Ctx::ObjectValue) => {
                if !matches!(next_char, Some(',') | Some('}') | None) {
                    return None;
                }
            }
            Some(Ctx::Array) => {
                if !matches!(next_char, Some(',') | Some(']') | None) {
                    return None;
                }
            }
            None => {
                if next_char.is_some() {
                    return None;
                }
            }
        }

        self.index = end + 1;
        Some(value)
    }

    /// parse_string.py's `_append_literal_char`: append one literal
    /// character (through the content bookkeeping) and advance.
    fn append_literal_char(&mut self, state: &mut StringParseState, ch: char) -> Option<char> {
        append_string_content(state, &[ch]);
        self.index += 1;
        self.cur()
    }

    /// parse_string.py's `_quote_belongs_to_regex_character_class`: is the
    /// current quote inside a compact `[...]` character class (no
    /// whitespace since the last unmatched `[`, and an unescaped `]`
    /// ahead)? The whitespace half runs as the state's lazy incremental
    /// probe (see the class_ws_found field docs), and the `]` lookahead
    /// runs through the parser-level lookahead memo: together they turn
    /// the per-candidate cost from O(accumulator + to-end-of-input)
    /// upstream into amortized O(new tail).
    fn quote_belongs_to_regex_character_class(&mut self, state: &mut StringParseState) -> bool {
        let Some(_class_start) = state.regex_character_class_start else {
            return false;
        };
        // The lazy whitespace probe: scan only the unprobed accumulator
        // tail (the class region only ever grows by appends, so this is
        // amortized O(1) per query: upstream rescans the whole
        // string_acc[start:] per closing-quote candidate, O(n^2) on
        // `'{"a": "[' + 'x"'*n + '"}'`).
        if state.class_ws_probed < state.string_acc.len() {
            let probed = state.class_ws_probed;
            state.class_ws_found |= state.string_acc[probed..].chars().any(is_py_whitespace);
            state.class_ws_probed = state.string_acc.len();
        }
        if state.class_ws_found {
            return false;
        }
        let closing_bracket_idx = self.cached_skip_to_character(&[']'], 1);
        self.get(closing_bracket_idx as isize) == Some(']')
    }

    /// parse_string.py's `_cached_skip_to_character`: skip_to_character with
    /// a memo keyed by the target set, holding the O(1)-amortized promise
    /// the far-quote corpus depends on:
    ///
    /// - a cached `(start, None)` entry means "no unescaped target anywhere
    ///   at or after `start`": any later scan starting at or past `start`
    ///   returns the to-end distance without rescanning;
    /// - a cached `(start, Some(match))` entry covers every start in
    ///   `[start, match]`: the first unescaped match from any of those
    ///   positions is still `match`;
    /// - every match is cached, backslash-adjacent ones included. Upstream
    ///   guards writes with `json_str[match_index - 1] != "\\"`, which
    ///   suppresses the memo exactly on the inputs where it is needed most
    ///   (`'["' + ']'*n + '\\\\' + '" x'` stays O(n^2) upstream); tors drops
    ///   the guard. Safety: a target's escape parity only depends on the
    ///   scan start when the start sits inside the backslash run immediately
    ///   preceding the target, and every caller of this function is
    ///   anchored: it starts on a non-backslash char, or on the first
    ///   backslash of a run whose preceding char is a known non-backslash
    ///   (`}`, `]`, `:`, `,`, a delimiter, whitespace, or the scan's own
    ///   previous match). No anchored start can land mid-run, so parity is
    ///   start-independent for all readers of a key and the memo is
    ///   uncached-exact for them. Do not add a cached call at a
    ///   non-anchored site.
    ///
    /// The memo lives on the parser (upstream scopes it to one string's
    /// parse state): entries are pure buffer facts, so sharing them across
    /// the many short string parses of one repair is exact, and both
    /// buffer-mutating sites (split_object_on_duplicate_key's insert and
    /// repair_empty_object_result's normalization splice) clear it.
    fn cached_skip_to_character(&mut self, targets: &[char], idx: usize) -> usize {
        let key = lookahead_key(targets);
        let start_index = self.index + idx;
        // Any covering entry answers exactly (entry claim ranges are
        // non-conflicting: two entries covering one start would both claim
        // its first unescaped match, forcing the matches to be equal).
        let cached = self
            .lookahead_cache
            .iter()
            .find(|(k, (cached_start, cached_match))| {
                *k == key
                    && match cached_match {
                        None => start_index >= *cached_start,
                        Some(m) => *cached_start <= start_index && start_index <= *m,
                    }
            })
            .map(|&(_, v)| v);
        if let Some((_, cached_match)) = cached {
            return match cached_match {
                None => self.s.len() - self.index,
                Some(m) => m - self.index,
            };
        }

        let match_offset = self.skip_to_character(targets, idx);
        if self.get(match_offset as isize).is_none() {
            self.cache_put(key, (start_index, None));
            return match_offset;
        }

        let match_index = self.index + match_offset;
        self.cache_put(key, (start_index, Some(match_index)));
        match_offset
    }

    /// Upstream's `lookahead_cache[targets] = entry` is a dict
    /// insert-or-replace. tors keeps a bounded per-key list instead (see
    /// the field docs): every write appends (a covering entry would have
    /// produced a hit, so no write is redundant), a not-found write
    /// retires the older not-found entries it dominates (it starts before
    /// them, it missed, so it covers everything they did), and the list
    /// is capped per key: on overflow the entry with the smallest match
    /// (the least forward coverage for the scan cursor's mostly-forward
    /// motion) is dropped, bounding both the read scan and memory. A drop
    /// costs one rescan, never an asymptotic regression.
    fn cache_put(&mut self, key: LookaheadKey, entry: LookaheadEntry) {
        const MAX_ENTRIES_PER_KEY: usize = 8;
        if entry.1.is_none() {
            self.lookahead_cache
                .retain(|(k, e)| *k != key || e.1.is_some());
        }
        let same_key = self
            .lookahead_cache
            .iter()
            .filter(|(k, _)| *k == key)
            .count();
        if same_key >= MAX_ENTRIES_PER_KEY {
            let mut victim = None;
            let mut smallest = usize::MAX;
            for (idx, (k, e)) in self.lookahead_cache.iter().enumerate() {
                if *k == key
                    && let Some(m) = e.1
                    && m < smallest
                {
                    smallest = m;
                    victim = Some(idx);
                }
            }
            if let Some(victim) = victim {
                self.lookahead_cache.swap_remove(victim);
            }
        }
        self.lookahead_cache.push((key, entry));
    }

    /// parse_string.py's `_normalize_escape_sequence`, called with the char
    /// following a just-appended backslash. Returns
    /// `(handled, next_char)`.
    fn normalize_escape_sequence(
        &mut self,
        state: &mut StringParseState,
        ch: char,
    ) -> (bool, Option<char>) {
        // Upstream logs "Found a stray escape sequence, normalizing it" on
        // entry.
        let active = state.active_rstring_delimiter();
        if state.in_low_smart_quote_span() && ch == '"' {
            // A bare ASCII quote inside a „...” span: replace the backslash
            // with the quote and close the span.
            rewrite_escape_tail(state, &[ch]);
            state.pop_low_smart_quote_span();
            self.index += 1;
            return (true, self.cur());
        }
        if ch == '\\' {
            // self.index >= 1 whenever this runs (the backslash before the
            // cursor was just appended to the accumulator).
            let run_start = self.index - 1;
            let mut run_end = self.index + 1;
            while run_end < self.s.len() && self.s[run_end] == '\\' {
                run_end += 1;
            }
            let run_length = run_end - run_start;
            let next_char = self.get((run_end - self.index) as isize);
            if run_length.is_multiple_of(2) && next_char != Some(active) {
                // Halve an even backslash run (it escapes itself).
                rewrite_escape_tail(state, &vec!['\\'; run_length / 2]);
                self.index = run_end;
                return (true, self.cur());
            }
        }
        if ch == active || matches!(ch, 't' | 'n' | 'r' | 'b' | '\\') {
            let escape_seqs = match ch {
                't' => '\t',
                'n' => '\n',
                'r' => '\r',
                'b' => '\u{8}',
                _ => ch,
            };
            rewrite_escape_tail(state, &[escape_seqs]);
            self.index += 1;
            let mut next_char = self.cur();
            // Fold any immediately-following `\"`/`\\`-style pairs into the
            // same normalization.
            while let Some(nc) = next_char
                && !state.string_acc.is_empty()
                && state.string_acc.ends_with('\\')
                && (nc == active || nc == '\\')
            {
                rewrite_escape_tail(state, &[nc]);
                self.index += 1;
                next_char = self.cur();
            }
            return (true, next_char);
        }
        if ch == 'u' || ch == 'x' {
            let num_chars = if ch == 'u' { 4 } else { 2 };
            let lo = self.index + 1;
            let hi = lo + num_chars;
            if hi <= self.s.len() && self.s[lo..hi].iter().all(|c| c.is_ascii_hexdigit()) {
                // Upstream logs "Found a unicode escape sequence,
                // normalizing it". Lone surrogates become U+FFFD (§9.2: a
                // Rust String cannot hold them); a well-formed pair
                // combines to its astral scalar (Python strings hold both
                // halves and json.dumps re-emits the pair verbatim, so
                // combining is the byte-identical port).
                let code = self.s[lo..hi]
                    .iter()
                    .fold(0u32, |acc, &c| acc * 16 + c.to_digit(16).unwrap_or(0));
                let decoded = match char::from_u32(code) {
                    Some(c) => c,
                    None => match self.decode_surrogate_pair(code, hi) {
                        Some((astral, consumed)) => {
                            rewrite_escape_tail(state, &[astral]);
                            self.index += 1 + num_chars + consumed;
                            return (true, self.cur());
                        }
                        None => '\u{FFFD}',
                    },
                };
                rewrite_escape_tail(state, &[decoded]);
                self.index += 1 + num_chars;
                return (true, self.cur());
            }
        } else if ch == '\u{201E}' || (is_string_delimiter(ch) && ch != active) {
            // Upstream logs "Found a delimiter that was escaped but
            // shouldn't be escaped, removing the escape" (Python's
            // precedence: `char == "„" or (char in STRING_DELIMITERS and
            // char != active)`).
            rewrite_escape_tail(state, &[ch]);
            self.index += 1;
            return (true, self.cur());
        }
        (false, Some(ch))
    }

    /// The well-formed half of a `\udXXX` surrogate escape: when `code` is
    /// a high surrogate immediately followed by a `\udCXX` low-surrogate
    /// escape, combine the pair into its astral scalar and report how many
    /// input codepoints the low half consumed (`\`, `u`, 4 hex: `hi` is
    /// already past the high half). Anything else is a lone surrogate
    /// (§9.2 → U+FFFD): None.
    fn decode_surrogate_pair(&self, code: u32, hi: usize) -> Option<(char, usize)> {
        let is_high = (0xD800..=0xDBFF).contains(&code);
        if !is_high || hi + 6 > self.s.len() {
            return None;
        }
        if self.s[hi] != '\\' || self.s[hi + 1] != 'u' {
            return None;
        }
        let lo = hi + 2;
        if !self.s[lo..lo + 4].iter().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        let low = self.s[lo..lo + 4]
            .iter()
            .fold(0u32, |acc, &c| acc * 16 + c.to_digit(16).unwrap_or(0));
        if !(0xDC00..=0xDFFF).contains(&low) {
            return None;
        }
        let scalar = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
        Some((char::from_u32(scalar)?, 6))
    }

    /// parse_string.py's `_brace_before_code_fence_belongs_to_string`:
    /// distinguish a trailing wrapper fence from a literal fenced snippet
    /// that belongs inside the current string. (Upstream's own docstring
    /// comment on the distinction.)
    fn brace_before_code_fence_belongs_to_string(
        &self,
        state: &mut StringParseState,
        fence_idx: usize,
    ) -> bool {
        let mut quote_search_idx = fence_idx + 3;
        let next_content_idx = self.scroll_comment_prefixed_member_start(quote_search_idx);
        let mut keep_post_fence_container = false;
        if self
            .get(next_content_idx as isize)
            .is_some_and(|c| INLINE_CONTAINER_OPENERS.contains(&c))
            && let Some(container_end_idx) = self.skip_inline_container(next_content_idx)
        {
            if self.post_fence_container_starts_next_member(container_end_idx) {
                return false;
            }
            keep_post_fence_container = true;
            quote_search_idx = container_end_idx;
        }

        let outer = state.outer_rstring_delimiter();
        let mut quote_idx = self.skip_to_character(&[outer], quote_search_idx);
        while self.get(quote_idx as isize) == Some(outer) {
            let after_quote_idx = self.scroll_whitespaces(quote_idx + 1);
            let after_quote = self.get(after_quote_idx as isize);
            if matches!(after_quote, Some(',') | Some('}') | Some(']') | None) {
                if keep_post_fence_container {
                    state.pending_inline_container = true;
                }
                return true;
            }
            quote_idx = self.skip_to_character(&[outer], quote_idx + 1);
        }
        false
    }

    /// parse_string.py's `_bare_key_is_followed_by_colon`. Note the entry
    /// check allows only alnum/`_`, while the walk also allows `-`:
    /// upstream's own asymmetry, kept.
    fn bare_key_is_followed_by_colon(&self, key_idx: usize) -> bool {
        let key_char = self.get(key_idx as isize);
        if !key_char.is_some_and(|c| c.is_alphanumeric() || c == '_') {
            return false;
        }

        let mut key_idx = key_idx;
        loop {
            let key_char = self.get(key_idx as isize);
            if !key_char.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '-') {
                break;
            }
            key_idx += 1;
        }

        let key_idx = self.scroll_whitespaces(key_idx);
        self.get(key_idx as isize) == Some(':')
    }

    /// parse_string.py's `_post_fence_container_starts_next_member`.
    fn post_fence_container_starts_next_member(&self, container_end_idx: usize) -> bool {
        let after_container_idx = self.scroll_whitespaces(container_end_idx);
        let after_container = self.get(after_container_idx as isize);
        if matches!(after_container, Some('}') | None) {
            return true;
        }
        if after_container != Some(',') {
            return false;
        }

        let next_member_idx = self.scroll_comment_prefixed_member_start(after_container_idx + 1);
        matches!(self.get(next_member_idx as isize), Some('}') | None)
            || self.object_member_starts_at(next_member_idx)
    }

    /// parse_string.py's `_starts_nested_inline_container`: does the opener
    /// at `idx` start a container nested in the current one (vs. prose that
    /// merely looks like one)? `idx` and the backward scan are relative
    /// offsets from `self.index` (upstream's `get_char_at(prev_idx)`), and
    /// the scan never goes before the cursor.
    fn starts_nested_inline_container(&self, idx: usize) -> bool {
        let opening_delimiter = self.get(idx as isize);
        let mut prev_idx: isize = idx as isize - 1;
        while prev_idx >= 0 {
            let prev_char = self.get(prev_idx);
            let Some(prev_char) = prev_char else {
                return true;
            };
            if !is_py_whitespace(prev_char) {
                if INLINE_CONTAINER_OPENERS.contains(&prev_char) {
                    return true;
                }
                if prev_char != ',' && prev_char != ':' {
                    return false;
                }

                let next_idx = self.scroll_whitespaces(idx + 1);
                let next_char = self.get(next_idx as isize);
                if matches!(opening_delimiter, Some('[') | Some('(')) {
                    // Python's membership list: ["]", ")", *STRING_DELIMITERS,
                    // "-", *INLINE_CONTAINER_OPENERS, "t", "f", "n"], then
                    // `next_char.isdigit()`: ASCII digits here (§9.3).
                    return match next_char {
                        Some(c) => {
                            c == ']'
                                || c == ')'
                                || is_string_delimiter(c)
                                || c == '-'
                                || INLINE_CONTAINER_OPENERS.contains(&c)
                                || matches!(c, 't' | 'f' | 'n')
                                || c.is_ascii_digit()
                        }
                        None => false,
                    };
                }
                if opening_delimiter != Some('{') {
                    return false;
                }
                if matches!(next_char, Some(c) if c == '}' || is_string_delimiter(c)) {
                    return true;
                }
                return prev_char == ':' && self.bare_key_is_followed_by_colon(next_idx);
            }
            prev_idx -= 1;
        }
        true
    }

    /// parse_string.py's `_skip_inline_container`: skip a balanced inline
    /// container starting at `idx` (a relative offset), returning the
    /// relative offset just past its closer, or `None` when it never
    /// balances. Non-openers return `idx` itself.
    fn skip_inline_container(&self, idx: usize) -> Option<usize> {
        let opening_delimiter = self.get(idx as isize)?;
        let Some(closer) = inline_container_closer(opening_delimiter) else {
            return Some(idx);
        };

        let mut stack = vec![closer];
        let mut i = idx + 1;
        while !stack.is_empty() {
            let ch = match self.get(i as isize) {
                Some(ch) => ch,
                None => {
                    self.note_scan_distance(i - idx);
                    return None;
                }
            };
            if is_string_delimiter(ch) {
                let end_delimiter = matching_string_delimiter(ch);
                i = self.skip_to_character(&[end_delimiter], i + 1);
                if self.get(i as isize) != Some(end_delimiter) {
                    self.note_scan_distance(i - idx);
                    return None;
                }
            } else if let Some(nested_closer) = inline_container_closer(ch)
                && self.starts_nested_inline_container(i)
            {
                stack.push(nested_closer);
            } else if stack.last() == Some(&ch) {
                stack.pop();
                if stack.is_empty() {
                    self.note_scan_distance(i + 1 - idx);
                    return Some(i + 1);
                }
            }
            i += 1;
        }

        None // upstream: # pragma: no cover; the loop only exits by return
    }

    /// parse_string.py's `_scroll_comment_prefixed_member_start`: advance
    /// past whitespace and any `#`/`//`/`/*...*/` comments to where the
    /// next object member would start.
    fn scroll_comment_prefixed_member_start(&self, idx: usize) -> usize {
        let start = idx;
        let mut idx = self.scroll_whitespaces(idx);
        loop {
            let ch = self.get(idx as isize);
            if ch == Some('#') {
                let mut ch = ch;
                while ch.is_some() && !matches!(ch, Some('\n') | Some('\r')) {
                    idx += 1;
                    ch = self.get(idx as isize);
                }
                idx = self.scroll_whitespaces(idx);
                continue;
            }
            if ch == Some('/') {
                let next_char = self.get(idx as isize + 1);
                if next_char == Some('/') {
                    idx += 2;
                    let mut ch = self.get(idx as isize);
                    while ch.is_some() && !matches!(ch, Some('\n') | Some('\r')) {
                        idx += 1;
                        ch = self.get(idx as isize);
                    }
                    idx = self.scroll_whitespaces(idx);
                    continue;
                }
                if next_char == Some('*') {
                    idx += 2;
                    loop {
                        let ch = self.get(idx as isize);
                        let Some(ch) = ch else {
                            self.note_scan_distance(idx - start);
                            return idx;
                        };
                        if ch == '*' && self.get(idx as isize + 1) == Some('/') {
                            idx += 2;
                            break;
                        }
                        idx += 1;
                    }
                    idx = self.scroll_whitespaces(idx);
                    continue;
                }
            }
            self.note_scan_distance(idx - start);
            return idx;
        }
    }

    /// parse_string.py's `_quoted_object_member_follows`: after the quote at
    /// `quote_idx` (a relative offset), does a `, "key":` member start?
    fn quoted_object_member_follows(&self, quote_idx: usize) -> bool {
        let comma_idx = self.scroll_whitespaces(quote_idx + 1);
        if self.get(comma_idx as isize) != Some(',') {
            return false;
        }

        let next_member_idx = self.scroll_comment_prefixed_member_start(comma_idx + 1);
        self.object_member_starts_at(next_member_idx)
    }

    /// parse_string.py's `_object_member_starts_at`.
    fn object_member_starts_at(&self, next_member_idx: usize) -> bool {
        if matches!(self.get(next_member_idx as isize), Some('}') | None) {
            return false;
        }

        let next_member = self.get(next_member_idx as isize);
        if let Some(next_member) = next_member
            && is_string_delimiter(next_member)
        {
            let key_end_delimiter = matching_string_delimiter(next_member);
            let key_end_idx = self.skip_to_character(&[key_end_delimiter], next_member_idx + 1);
            if self.get(key_end_idx as isize) != Some(key_end_delimiter) {
                return false;
            }
            let after_key_idx = self.scroll_whitespaces(key_end_idx + 1);
            return self.get(after_key_idx as isize) == Some(':');
        }

        if next_member.is_some_and(|c| c.is_alphanumeric() || c == '_') {
            return self.bare_key_is_followed_by_colon(next_member_idx);
        }

        false
    }

    /// The Array-context delimiter-pairing walk of
    /// handle_right_delimiter_candidate. `stop_idx` is the pre-walk's stop
    /// offset (relative to the cursor): an unescaped `outer` (the 1271-gate
    /// guarantees it on entry). The walk pairs up the following unescaped
    /// `[outer, ']']` targets: the first non-`outer` one at an odd chain
    /// index (1-based) makes the verdict false; at an even index (or the
    /// end of input, which acts as an even-index terminator) it stays true.
    ///
    /// While walking, the exact same verdict is derived for every `outer`
    /// quote the walk passes (each is a future delimiter candidate) and
    /// recorded in `state.pairing_outcomes`, so a quote run costs one walk
    /// total instead of one walk per quote. The derivation for the quote
    /// at chain position `t_j` (j = 0 is the pre-walk stop itself):
    ///
    /// - that candidate's own pre-walk stops at the first stop-worthy char
    ///   after `t_j`: any-parity `outer`/`lstring`, plus `}`/`:` per the
    ///   frozen context stack, which is either an in-interval stop `e_j`
    ///   or the next chain target `t_{j+1}` (both `outer` and `]` are
    ///   stop-worthy);
    /// - if that stop char is not an unescaped `outer`, the candidate never
    ///   reaches the pairing branch (the 1271 gate fails) and nothing is
    ///   recorded: its natural path is already O(1);
    /// - otherwise its walk pairs the same remaining chain, shifted by one
    ///   element when the stop was an in-interval `e_j` (the chain from
    ///   `e_j` starts at `t_{j+1}`, from `t_{j+1}` at `t_{j+2}`), so the
    ///   first non-`outer` target sits at chain index `m - j - 1 + [e_j]`
    ///   and the verdict is that index's parity.
    fn array_pairing_walk(
        &mut self,
        state: &mut StringParseState,
        stop_idx: usize,
        outer: char,
    ) -> bool {
        let lstring = state.lstring_delimiter;
        let has_key = self.ctx_has(Ctx::ObjectKey);
        let has_val = self.ctx_has(Ctx::ObjectValue);
        // ctx_has(Ctx::Array) is implied (ctx_current() == Array here), so
        // `]` and `,` are always stop-worthy in this walk.
        let is_stop = |c: char| {
            c == outer
                || c == lstring
                || c == ']'
                || c == ','
                || (has_key && (c == ':' || c == '}'))
                || (has_val && c == '}')
        };

        // One pass over one interval (prev, target): the next unescaped
        // [outer, ']'] target, and the first stop-worthy char (any escape
        // parity: the pre-walk does not skip escapes) strictly before it.
        let scan_interval = |from: usize| -> (Option<usize>, Option<usize>) {
            let mut backslashes = 0usize;
            let mut stop = None;
            let mut i = from;
            while i < self.s.len() {
                let c = self.s[i];
                if c == '\\' {
                    backslashes += 1;
                    i += 1;
                    continue;
                }
                if (c == outer || c == ']') && backslashes.is_multiple_of(2) {
                    return (Some(i), stop);
                }
                if stop.is_none() && is_stop(c) {
                    stop = Some(i);
                }
                backslashes = 0;
                i += 1;
            }
            (None, stop)
        };

        // Scratch owned by the parser, taken for this walk and returned
        // with capacity kept (see the field docs).
        let mut outers = std::mem::take(&mut self.pairing_scratch_outers);
        let mut interval_stops = std::mem::take(&mut self.pairing_scratch_stops);
        outers.clear();
        interval_stops.clear();

        // t_0 = the pre-walk stop; t_1.. the outer chain; t_m the first
        // non-outer target or the end of input.
        let mut prev = self.index + stop_idx; // t_0
        let m = loop {
            let (target, stop) = scan_interval(prev + 1);
            interval_stops.push(stop);
            match target {
                None => break interval_stops.len(),
                Some(t) if self.s[t] != outer => break interval_stops.len(),
                Some(t) => {
                    outers.push(t);
                    prev = t;
                }
            }
        };

        // Record the verdict for every quote the walk passed (t_0..t_{m-1}
        // are all `outer` by construction; the terminal t_m is not).
        let t0 = self.index + stop_idx;
        for j in 0..m {
            let pos = if j == 0 { t0 } else { outers[j - 1] };
            let first_non_outer_idx = match interval_stops[j] {
                // An in-interval stop char is that candidate's own pre-walk
                // stop; record only if it passes the 1271 gate (an unescaped
                // `outer`). Its chain starts at t_{j+1}, so t_m sits at
                // index m - j.
                Some(e) => {
                    if self.s[e] == outer && self.s[e - 1] != '\\' {
                        m - j
                    } else {
                        continue;
                    }
                }
                // No in-interval stop: the next chain target is the stop:
                // an unescaped `outer` whenever it exists (j + 1 < m), and
                // t_m (non-outer, or the end of input) otherwise, which
                // fails the gate. Its chain starts at t_{j+2}: index m-j-1.
                None if j + 1 < m => m - j - 1,
                None => continue,
            };
            record_pairing_outcome(
                &mut state.pairing_outcomes,
                pos,
                first_non_outer_idx.is_multiple_of(2),
            );
        }

        outers.clear();
        interval_stops.clear();
        self.pairing_scratch_outers = outers;
        self.pairing_scratch_stops = interval_stops;
        m.is_multiple_of(2)
    }

    /// parse_string.py's `_handle_right_delimiter_candidate`, called with the
    /// right-delimiter candidate `ch` at the cursor. Returns
    /// `(handled_delimiter, char, should_break)`.
    fn handle_right_delimiter_candidate(
        &mut self,
        state: &mut StringParseState,
        ch: char,
    ) -> (bool, Option<char>, bool) {
        let outer = state.outer_rstring_delimiter();

        if state.doubled_quotes && self.get(1) == Some(outer) {
            // Upstream logs "found a doubled quote, ignoring it".
            self.index += 1;
            return (true, Some(ch), false);
        }

        if state.missing_quotes && self.ctx_current() == Some(Ctx::ObjectValue) {
            // A delimiter in an unquoted object value might actually be the
            // opening quote of the next key.
            let mut i: usize = 1;
            let mut next_c = self.get(1);
            while next_c.is_some()
                && next_c != Some(outer)
                && next_c != Some(state.lstring_delimiter)
            {
                i += 1;
                next_c = self.get(i as isize);
            }
            if next_c.is_some() {
                let i = i + 1;
                let i = self.scroll_whitespaces(i);
                if self.get(i as isize) == Some(':') {
                    // self.index >= 1 here (the accumulator is non-empty
                    // whenever this helper runs): step back onto the
                    // delimiter so the finalizer consumes it as the closer.
                    self.index -= 1;
                    let next_char = self.cur();
                    // Upstream logs "it turns out it was the beginning of
                    // the next key. Stopping here."
                    return (false, next_char, true);
                }
            }
            return (false, Some(ch), false);
        }

        if state.unmatched_delimiter {
            state.unmatched_delimiter = false;
            let next_char = self.append_literal_char(state, ch);
            return (true, next_char, false);
        }

        let mut i: usize = 1;
        let mut next_c = self.get(1);
        let mut check_comma_in_object_value = true;
        while next_c.is_some() && next_c != Some(outer) && next_c != Some(state.lstring_delimiter) {
            if let Some(c) = next_c
                && check_comma_in_object_value
                && c.is_alphabetic()
            {
                check_comma_in_object_value = false;
            }
            // The walk stops at the first char that would end the enclosing
            // member/element from any context on the stack.
            let stop = match next_c {
                Some(c) => {
                    (self.ctx_has(Ctx::ObjectKey) && (c == ':' || c == '}'))
                        || (self.ctx_has(Ctx::ObjectValue) && c == '}')
                        || (self.ctx_has(Ctx::Array) && (c == ']' || c == ','))
                        || (check_comma_in_object_value
                            && self.ctx_current() == Some(Ctx::ObjectValue)
                            && c == ',')
                }
                None => false,
            };
            if stop {
                break;
            }
            i += 1;
            next_c = self.get(i as isize);
        }
        if next_c == Some(',') && self.ctx_current() == Some(Ctx::ObjectValue) {
            let i = i + 1;
            let i = self.skip_to_character(&[outer], i);
            // Upstream assigns next_c here and overwrites it two statements
            // later; the dead binding is not repeated.
            let i = i + 1;
            let i = self.scroll_whitespaces(i);
            let next_c = self.get(i as isize);
            if matches!(next_c, Some('}') | Some(',')) {
                // Upstream logs "a misplaced quote that would have closed
                // the string but has a different meaning here, ignoring it".
                let next_char = self.append_literal_char(state, ch);
                return (true, next_char, false);
            }
        } else if next_c == Some(outer) && self.get(i as isize - 1) != Some('\\') {
            if self.only_whitespace_until(i)
                && !(self.ctx_current() == Some(Ctx::ObjectValue)
                    && self.quoted_object_member_follows(i))
            {
                return (false, Some(ch), true);
            }
            if self.ctx_current() == Some(Ctx::ObjectValue) {
                if self.quoted_object_member_follows(i) {
                    // Upstream logs "a misplaced quote ... different meaning
                    // here, ignoring it".
                    let next_char = self.append_literal_char(state, ch);
                    return (true, next_char, false);
                }
                let i = self.skip_to_character(&[outer], i + 1);
                let mut i = i + 1;
                let mut next_c = self.get(i as isize);
                while next_c.is_some() && next_c != Some(':') {
                    let stop = match next_c {
                        Some(c) => {
                            c == ','
                                || c == ']'
                                || c == '}'
                                || (c == outer && self.get(i as isize - 1) != Some('\\'))
                        }
                        None => false,
                    };
                    if stop {
                        break;
                    }
                    i += 1;
                    next_c = self.get(i as isize);
                }
                if next_c != Some(':') {
                    // Upstream logs "a misplaced quote ... different meaning
                    // here, ignoring it"; flipping the unmatched flag keeps
                    // the next delimiter candidate in the string.
                    state.unmatched_delimiter = !state.unmatched_delimiter;
                    let next_char = self.append_literal_char(state, ch);
                    return (true, next_char, false);
                }
            } else if self.ctx_current() == Some(Ctx::Array) {
                // Pair up delimiters: an even count means this quote is
                // prose inside the string, an odd count means it closes.
                // The verdict is memoized per candidate position: without
                // the memo, an internal-quote run (`'["' + 'a"'*n + '"]'`)
                // re-walks the remaining quotes once per candidate (O(n^2);
                // upstream's own behavior; see array_pairing_walk).
                let even_delimiters = match state.pairing_outcome(self.index) {
                    Some(cached) => cached,
                    None => self.array_pairing_walk(state, i, outer),
                };
                if even_delimiters {
                    // Upstream logs "a quoted section that would have closed
                    // the string but has a different meaning here".
                    state.unmatched_delimiter = !state.unmatched_delimiter;
                    let next_char = self.append_literal_char(state, ch);
                    return (true, next_char, false);
                }
                return (false, Some(ch), true);
            } else if self.ctx_current() == Some(Ctx::ObjectKey) {
                // Upstream logs "a quoted section ... in Object Key context".
                let next_char = self.append_literal_char(state, ch);
                return (true, next_char, false);
            }
        }

        (false, Some(ch), false)
    }

    /// parse_string.py's `_scan_string_body`: the main scan loop. Returns
    /// the char at the stop position (the right delimiter when the string
    /// closed, whatever ended the scan otherwise, `None` at end of input).
    ///
    /// Split on a `const` deadline flag so the unbounded path (every
    /// existing caller) compiles to the pre-deadline loop with zero
    /// per-char cost: the armed check is dead code when `DL` is false.
    fn scan_string_body(&mut self, state: &mut StringParseState) -> Option<char> {
        if self.deadline_armed() {
            self.scan_string_body_impl::<true>(state)
        } else {
            self.scan_string_body_impl::<false>(state)
        }
    }

    fn scan_string_body_impl<const DL: bool>(
        &mut self,
        state: &mut StringParseState,
    ) -> Option<char> {
        let outer = state.outer_rstring_delimiter();

        let mut ch = self.cur();
        while let Some(c) = ch
            && (c != outer || state.in_low_smart_quote_span())
        {
            if DL && self.deadline_expired() {
                // True only with the sticky error set; parse_string polls
                // it below once the scan unwinds.
                break;
            }
            if state.missing_quotes {
                if self.ctx_current() == Some(Ctx::ObjectKey) && (c == ':' || is_py_whitespace(c)) {
                    // Upstream logs "missing the left delimiter in object
                    // key context, we found a :, stopping here".
                    break;
                }
                if self.ctx_current() == Some(Ctx::Array) && (c == ']' || c == ',') {
                    // Upstream logs "missing the left delimiter in array
                    // context, we found a ] or ,, stopping here".
                    break;
                }
            }
            if c == '\u{201E}' && (state.string_acc.is_empty() || !state.string_acc.ends_with('\\'))
            {
                // An unescaped low smart quote opens a span where even the
                // outer delimiter is ordinary content until the span closes.
                state.push_low_smart_quote_span();
                ch = self.append_literal_char(state, c);
                continue;
            }
            if state.in_low_smart_quote_span() && c == '\u{201D}' {
                state.pop_low_smart_quote_span();
                ch = self.append_literal_char(state, c);
                continue;
            }
            if (state.pending_inline_container
                || (self.ctx_current() == Some(Ctx::ObjectValue)
                    && c == '{'
                    && self.get(-1) != Some('\\')
                    && self.bare_key_is_followed_by_colon(self.scroll_whitespaces(1))))
                && INLINE_CONTAINER_OPENERS.contains(&c)
                && (state.string_acc.is_empty() || !state.string_acc.ends_with('\\'))
                && let Some(container_end_idx) = self.skip_inline_container(0)
            {
                // Upstream logs "a balanced inline container that
                // belongs to the string, keeping it".
                state.pending_inline_container = false;
                state.inline_container_stack.clear();
                let content = &self.s[self.index..self.index + container_end_idx];
                append_string_content(state, content);
                self.index += container_end_idx;
                ch = self.cur();
                continue;
            }
            if !STREAM_STABLE
                && self.ctx_current() == Some(Ctx::ObjectValue)
                && c == ','
                && !state.pending_inline_container
                && state.inline_container_stack.is_empty()
            {
                let comma_classification = if state.object_value_has_no_future_delimiter {
                    "string"
                } else {
                    classify_object_value_comma(self)
                };
                if comma_classification == "member" {
                    // Upstream logs "a comma that starts the next object
                    // member. Stopping here".
                    break;
                }
                if comma_classification == "string_no_future_delimiter" {
                    state.object_value_has_no_future_delimiter = true;
                }
                state.pending_inline_container = comma_classification == "container";
                // Upstream logs "a comma that belongs to the string, keeping
                // it".
                ch = self.append_literal_char(state, c);
                continue;
            }
            let (pending, keep_inline_container_char) = update_inline_container_stack(
                c,
                state.pending_inline_container,
                &mut state.inline_container_stack,
            );
            state.pending_inline_container = pending;
            if keep_inline_container_char {
                ch = self.append_literal_char(state, c);
                continue;
            }
            if !STREAM_STABLE
                && self.ctx_current() == Some(Ctx::ObjectValue)
                && c == '}'
                && (state.string_acc.is_empty() || !state.string_acc.ends_with(outer))
            {
                if state.object_value_unmatched_opening_braces > 0 {
                    // The string opened more braces than it closed: this '}'
                    // is content, not the object's closer.
                    ch = self.append_literal_char(state, c);
                    continue;
                }
                // Is a right delimiter present later, or does this '}' close
                // the object? (upstream's rstring_delimiter_missing probe)
                let mut rstring_delimiter_missing = true;
                self.skip_whitespaces();
                if self.get(1) == Some('\\') {
                    rstring_delimiter_missing = false;
                }
                let i = self.cached_skip_to_character(&[outer], 1);
                let next_c = self.get(i as isize);
                if next_c.is_some() {
                    let i = i + 1;
                    let i = self.scroll_whitespaces(i);
                    let next_c = self.get(i as isize);
                    if next_c.is_none() || matches!(next_c, Some(',') | Some('}')) {
                        rstring_delimiter_missing = false;
                    } else {
                        // Memoized: with the `"`-lookahead memo-hit above,
                        // every `}` of a close run starts this scan at the
                        // same anchored position (one past the matched
                        // delimiter), so an uncached scan here is O(n^2) on
                        // `'{"a": "' + '}'*n + '"' + 'y'*n + '"z'`.
                        let i = self.cached_skip_to_character(&[state.lstring_delimiter], i);
                        let next_c = self.get(i as isize);
                        if next_c.is_none() {
                            rstring_delimiter_missing = false;
                        } else {
                            let i = self.scroll_whitespaces(i + 1);
                            let next_c = self.get(i as isize);
                            if next_c.is_some() && next_c != Some(':') {
                                rstring_delimiter_missing = false;
                            }
                        }
                    }
                } else {
                    let i = self.skip_to_character(&[':'], 1);
                    let next_c = self.get(i as isize);
                    if next_c.is_some() {
                        break;
                    }
                    let i = self.scroll_whitespaces(1);
                    let j = self.skip_to_character(&['}'], i);
                    if j - i > 1 {
                        rstring_delimiter_missing = false;
                    }
                }
                if rstring_delimiter_missing {
                    // Upstream logs "we couldn't determine that a right
                    // delimiter was present. Stopping here".
                    break;
                }
            }
            if !STREAM_STABLE
                && c == ']'
                && self.ctx_has(Ctx::Array)
                && (state.string_acc.is_empty() || !state.string_acc.ends_with(outer))
            {
                // Memoized, like the sibling `}` lookahead earlier in this
                // scan. This branch fires once per `]` while scanning a string
                // body in array context, so an uncached forward scan makes
                // `'["' + ']'*n + '" x'` O(n^2).
                //
                // Parity-safety of sharing the `[outer]` memo key: every
                // reader/writer of that key starts its scan one past a
                // non-backslash char: this `]` site (idx 0, after `]`), the
                // sibling `}` probe above (idx 1, after `}`), and
                // classify_object_value_comma's CommaSkip `[next_special]`
                // scan (after a quote). None starts inside a backslash run, so
                // escape parity is fixed and the memo is uncached-exact for all
                // of them (verified: zero divergences vs json-repair 0.63.4 over
                // an exhaustive `]`/`}`/`\`/`"` sweep). This is not
                // "cached == uncached" in general: a scan starting inside a
                // backslash run can flip escape parity, so do not add a cached
                // [outer] call at a non-anchored site.
                //
                // The write guard upstream keeps here (matches preceded by
                // a backslash are found but not memoized) is lifted in
                // tors's cached_skip_to_character: every caller of a key
                // is anchored, so backslash-adjacent matches memoize
                // exactly and `'["' + ']'*n + '\\\\' + '" x'` is linear
                // here too (upstream stays O(n^2) on it; see #13).
                let i = self.cached_skip_to_character(&[outer], 0);
                if self.get(i as isize).is_none() {
                    break;
                }
            }
            if self.ctx_current() == Some(Ctx::ObjectValue) && c == '}' {
                let i = self.scroll_whitespaces(1);
                let next_c = self.get(i as isize);
                if next_c == Some('`')
                    && self.get(i as isize + 1) == Some('`')
                    && self.get(i as isize + 2) == Some('`')
                {
                    if self.brace_before_code_fence_belongs_to_string(state, i) {
                        // Upstream logs "a literal fenced snippet after },
                        // keeping it in the string".
                        ch = self.append_literal_char(state, c);
                        continue;
                    }
                    // Upstream logs "a } that closes the object before code
                    // fences, stopping here".
                    break;
                }
                if next_c.is_none() {
                    // Upstream logs "a } that closes the object, stopping
                    // here".
                    break;
                }
            }
            append_string_content(state, &[c]);
            self.index += 1;
            ch = self.cur();
            let Some(c) = ch else {
                if STREAM_STABLE && !state.string_acc.is_empty() && state.string_acc.ends_with('\\')
                {
                    state.string_acc.pop();
                    rebuild_unmatched_opening_braces(state);
                }
                break;
            };
            if !state.string_acc.is_empty() && state.string_acc.ends_with('\\') {
                let (handled_escape, nc) = self.normalize_escape_sequence(state, c);
                ch = nc;
                if handled_escape {
                    continue;
                }
            }
            if c == ':' && !state.missing_quotes && self.ctx_current() == Some(Ctx::ObjectKey) {
                // Both lookaheads memoized: unquoted-key input like
                // `'{' + 'a:b,'*n + '}'` fires this branch once per `:` and
                // each scan covers the whole remaining member run (~25s at
                // n=100k uncached). The starts are anchored (one past the
                // `:`, then one past the matched left delimiter).
                let i = self.cached_skip_to_character(&[state.lstring_delimiter], 1);
                let next_c = self.get(i as isize);
                if next_c.is_some() {
                    let i = i + 1;
                    let i = self.cached_skip_to_character(&[outer], i);
                    let next_c = self.get(i as isize);
                    if next_c.is_some() {
                        let i = i + 1;
                        let i = self.scroll_whitespaces(i);
                        let ch_after = self.get(i as isize);
                        if matches!(ch_after, Some(',') | Some('}')) {
                            // Upstream logs "missing the right delimiter in
                            // object key context, we found a {ch} stopping
                            // here".
                            break;
                        }
                    }
                } else {
                    // Upstream logs "missing the right delimiter in object
                    // key context, we found a :, stopping here".
                    break;
                }
            }
            if state.in_low_smart_quote_span() && c == '"' {
                state.pop_low_smart_quote_span();
                ch = self.append_literal_char(state, c);
                continue;
            }
            if c == outer
                && self.ctx_current() == Some(Ctx::ObjectValue)
                && self.quote_belongs_to_regex_character_class(state)
            {
                // Upstream logs "a bare quote inside a regex character
                // class, keeping it".
                ch = self.append_literal_char(state, c);
                continue;
            }
            if c == outer && !state.string_acc.is_empty() && !state.string_acc.ends_with('\\') {
                let (handled_delimiter, nc, should_break) =
                    self.handle_right_delimiter_candidate(state, c);
                ch = nc;
                if should_break {
                    break;
                }
                if handled_delimiter {
                    continue;
                }
            }
        }
        ch
    }

    /// parse_string.py's `_finalize_string_result`.
    fn finalize_string_result(&mut self, state: &mut StringParseState, ch: Option<char>) -> String {
        let outer = state.outer_rstring_delimiter();
        if let Some(c) = ch
            && state.missing_quotes
            && self.ctx_current() == Some(Ctx::ObjectKey)
            && is_py_whitespace(c)
        {
            // Upstream logs the "extreme corner case in which the LLM added
            // a comment instead of valid string, invalidate the string and
            // return an empty value" heuristic.
            self.skip_whitespaces();
            if !matches!(self.cur(), Some(':') | Some(',')) {
                return String::new();
            }
        }

        if ch != Some(outer) {
            if !STREAM_STABLE {
                // Upstream logs "we missed the closing quote, ignoring".
                rstrip_py(&mut state.string_acc);
            }
        } else {
            self.index += 1;
        }

        if !STREAM_STABLE
            && (state.missing_quotes
                || (!state.string_acc.is_empty() && state.string_acc.ends_with('\n')))
        {
            rstrip_py(&mut state.string_acc);
        }

        std::mem::take(&mut state.string_acc)
    }

    /// parse_string.py's `_only_whitespace_until`.
    fn only_whitespace_until(&self, end: usize) -> bool {
        for j in 1..end {
            if let Some(c) = self.get(j as isize)
                && !is_py_whitespace(c)
            {
                return false;
            }
        }
        true
    }

    /// parse_string_helpers/parse_boolean_or_null.py's
    /// `parse_boolean_or_null`. LITERAL_VALUES is an insertion-ordered dict
    /// upstream; the order matters for 'n' ("null" is tried before "none")
    /// and is kept. Returns `(found, value)`; when not found the value is
    /// upstream's `None` placeholder (ignored by the caller).
    fn parse_boolean_or_null(&mut self) -> (bool, Value) {
        const LITERAL_VALUES: [(&str, Value); 4] = [
            ("true", Value::Bool(true)),
            ("false", Value::Bool(false)),
            ("null", Value::Null),
            ("none", Value::Null),
        ];
        let ch = self.cur().map(lower_char);
        let starting_index = self.index;
        for (literal, value) in LITERAL_VALUES {
            if ch != Some(literal.chars().next().unwrap_or('\0')) {
                continue;
            }
            self.index = starting_index;
            let mut matched = true;
            for expected_char in literal.chars() {
                if self.cur().map(lower_char) != Some(expected_char) {
                    matched = false;
                    break;
                }
                self.index += 1;
            }
            if matched {
                let next_char = self.cur();
                if next_char.is_none()
                    || next_char.is_some_and(is_py_whitespace)
                    || matches!(next_char, Some(',') | Some(')') | Some(']') | Some('}'))
                {
                    // When literal == "none", upstream logs "Converted
                    // unquoted Python None literal to JSON null" here.
                    return (true, value);
                }
            }
        }
        self.index = starting_index;
        (false, Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The naive whole-accumulator scan: upstream's
    /// `_rebuild_unmatched_opening_braces`, kept verbatim as the oracle
    /// for the incremental twin's differential test below.
    fn oracle_scan(acc: &str) -> (usize, Option<usize>) {
        let mut braces = 0usize;
        let mut class = None;
        for (index, ch) in acc.char_indices() {
            if ch == '{' {
                braces += 1;
            } else if ch == '}' && braces > 0 {
                braces -= 1;
            } else if ch == '[' {
                class = Some(index + 1);
            } else if ch == ']' {
                class = None;
            }
        }
        (braces, class)
    }

    /// Differential proof of the incremental counters: replay seeded
    /// random push / pop-then-push / slice-append sequences (the exact
    /// mutation shapes the escape normalizer performs) and assert the
    /// maintained (brace balance, class start) equals the oracle's full
    /// rescan after every step, on accumulators built from the characters
    /// that actually drive the counters (`{`, `}`, `[`, `]`, letters,
    /// backslashes, multibyte).
    #[test]
    fn incremental_brace_counters_match_the_full_rescan_oracle() {
        let mut rng: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        const ALPHABET: [char; 8] = ['{', '}', '[', ']', 'a', '\\', '\u{201E}', '\u{1F600}'];
        for _case in 0..128 {
            let mut state = StringParseState::new();
            for _step in 0..512 {
                match next() % 10 {
                    0..=4 => {
                        let ch = ALPHABET[(next() % ALPHABET.len() as u64) as usize];
                        append_string_content(&mut state, &[ch]);
                    }
                    5..=7 => {
                        // The escape-repair shape: rewrite_escape_tail's
                        // contract is popping a just-appended backslash
                        // (counter-neutral: no real call site ever pops
                        // anything else), so the grammar mirrors that
                        // precondition exactly: rewrite only when the
                        // accumulator ends in a backslash.
                        if state.string_acc.ends_with('\\') {
                            let ch = ALPHABET[(next() % ALPHABET.len() as u64) as usize];
                            rewrite_escape_tail(&mut state, &[ch]);
                        } else {
                            let ch = ALPHABET[(next() % ALPHABET.len() as u64) as usize];
                            append_string_content(&mut state, &[ch]);
                        }
                    }
                    _ => {
                        // slice append (the inline-container shape)
                        let len = (next() % 4) as usize;
                        let content: Vec<char> = (0..len)
                            .map(|_| ALPHABET[(next() % ALPHABET.len() as u64) as usize])
                            .collect();
                        append_string_content(&mut state, &content);
                    }
                }
                let got = (
                    state.object_value_unmatched_opening_braces,
                    state.regex_character_class_start,
                );
                let want = oracle_scan(&state.string_acc);
                assert_eq!(
                    got, want,
                    "divergent counters after step (acc {:?})",
                    state.string_acc
                );
                // The lazy whitespace probe's watermark can never pass the
                // accumulator's length, and a closed class carries no
                // probe state.
                assert!(state.class_ws_probed <= state.string_acc.len());
                if state.regex_character_class_start.is_none() {
                    assert!(!state.class_ws_found && state.class_ws_probed == 0);
                }
            }
        }
    }

    fn parse_string_in(raw: &str, ctx: Ctx) -> Result<Value, String> {
        let mut parser = Parser::new(raw, false, None);
        parser.ctx_push(ctx);
        parser.parse_string()
    }

    fn parse_string_strict_in(raw: &str, ctx: Ctx) -> Result<Value, String> {
        let mut parser = Parser::new(raw, true, None);
        parser.ctx_push(ctx);
        parser.parse_string()
    }

    #[test]
    fn well_formed_surrogate_pairs_combine_to_the_astral_scalar() {
        // Python strings hold both halves of a legal pair and json.dumps
        // re-emits them verbatim; the port combines them into the astral
        // scalar (byte-identical after dumps' ensure_ascii re-encoding).
        // Lone halves keep §9.2's U+FFFD, high or low.
        let parsed = Parser::new(r#"["\ud83d\ude00"]"#, false, None)
            .parse()
            .unwrap();
        assert_eq!(parsed, Value::Array(vec![Value::Str("\u{1F600}".into())]));
        let lone_high = Parser::new(r#"["\ud83d"]"#, false, None).parse().unwrap();
        assert_eq!(lone_high, Value::Array(vec![Value::Str("\u{FFFD}".into())]));
        let lone_low = Parser::new(r#"["\ude00"]"#, false, None).parse().unwrap();
        assert_eq!(lone_low, Value::Array(vec![Value::Str("\u{FFFD}".into())]));
        // A high half not followed by a low-half escape is lone: FFFD and
        // the following content still parses normally.
        let stray = Parser::new(r#"["\ud83d x"]"#, false, None).parse().unwrap();
        assert_eq!(stray, Value::Array(vec![Value::Str("\u{FFFD} x".into())]));
    }

    fn parser_in(raw: &str) -> Parser {
        Parser::new(raw, false, None)
    }

    fn str_value(s: &str) -> Value {
        Value::Str(s.to_string())
    }

    // ---- _try_parse_simple_quoted_string (direct) ----

    #[test]
    fn try_parse_simple_quoted_string_rejects_ambiguous_trailing_text() {
        // test_parse_string_fast_path_rejects_ambiguous_top_level_trailing_text
        let mut parser = parser_in(r#""value" trailing"#);
        assert_eq!(parser.try_parse_simple_quoted_string(), None);
    }

    #[test]
    fn try_parse_simple_quoted_string_rejects_escapes_and_unterminated() {
        // test_parse_string_fast_path_string_wrapper_fallbacks (the str-input
        // equivalents of upstream's two StringFileWrapper cases; the json_fd
        // wrapper itself is not ported, §9.1)
        let mut escaped = parser_in(r#""va\lue""#);
        assert_eq!(escaped.try_parse_simple_quoted_string(), None);
        let mut unterminated = parser_in(r#""value"#);
        assert_eq!(unterminated.try_parse_simple_quoted_string(), None);
    }

    // ---- _brace_before_code_fence_belongs_to_string (direct) ----

    #[test]
    fn brace_before_code_fence_rejects_non_delimiter_after_quote() {
        // test_brace_before_code_fence_helper_rejects_non_delimiter_after_quote
        let parser = parser_in(r#"}```"oops"#);
        let mut state = StringParseState::new();
        assert!(!parser.brace_before_code_fence_belongs_to_string(&mut state, 1));
    }

    #[test]
    fn brace_before_code_fence_rejects_unterminated_container() {
        // test_brace_before_code_fence_helper_rejects_unterminated_container_after_fence
        let parser = parser_in("}``` [1,2");
        let mut state = StringParseState::new();
        assert!(!parser.brace_before_code_fence_belongs_to_string(&mut state, 1));
    }

    #[test]
    fn brace_before_code_fence_accepts_unbalanced_container_like_prose() {
        // test_brace_before_code_fence_helper_accepts_unbalanced_container_like_prose_after_fence
        let parser = parser_in("}``` [{\n\", \"b\": 1");
        let mut state = StringParseState::new();
        assert!(parser.brace_before_code_fence_belongs_to_string(&mut state, 1));
    }

    #[test]
    fn brace_before_code_fence_accepts_later_closing_quote_after_prose() {
        // test_brace_before_code_fence_helper_accepts_later_closing_quote_after_quoted_prose
        let parser = parser_in(r#"}```Implementation: "xxx", xxx", "b": 1"#);
        let mut state = StringParseState::new();
        assert!(parser.brace_before_code_fence_belongs_to_string(&mut state, 1));
    }

    #[test]
    fn brace_before_code_fence_rejects_container_started_after_fence() {
        // test_brace_before_code_fence_helper_rejects_container_started_after_fence
        let parser = parser_in(r#"}``` [1,"z"], "b": 1"#);
        let mut state = StringParseState::new();
        assert!(!parser.brace_before_code_fence_belongs_to_string(&mut state, 1));
    }

    #[test]
    fn brace_before_code_fence_rejects_container_closing_object() {
        // test_brace_before_code_fence_helper_rejects_container_closing_object_after_fence
        let parser = parser_in("}``` [1,2]}");
        let mut state = StringParseState::new();
        assert!(!parser.brace_before_code_fence_belongs_to_string(&mut state, 1));
    }

    #[test]
    fn brace_before_code_fence_accepts_literal_container_after_fence() {
        // test_brace_before_code_fence_helper_accepts_literal_container_after_fence
        let parser = parser_in("}``` [1,2]\n\", \"b\": 1");
        let mut state = StringParseState::new();
        assert!(parser.brace_before_code_fence_belongs_to_string(&mut state, 1));
    }

    #[test]
    fn brace_before_code_fence_rejects_comment_prefixed_container() {
        // test_brace_before_code_fence_helper_rejects_comment_prefixed_container_after_fence
        let parser = parser_in("}``` // c\n [1,\"z\"], \"b\": 1");
        let mut state = StringParseState::new();
        assert!(!parser.brace_before_code_fence_belongs_to_string(&mut state, 1));
    }

    #[test]
    fn brace_before_code_fence_accepts_comment_prefixed_literal_container() {
        // test_brace_before_code_fence_helper_accepts_comment_prefixed_literal_container_after_fence
        let parser = parser_in("}``` // c\n [1,2]\n\", \"b\": 1");
        let mut state = StringParseState::new();
        assert!(parser.brace_before_code_fence_belongs_to_string(&mut state, 1));
    }

    #[test]
    fn brace_before_code_fence_accepts_literal_container_with_trailing_comma() {
        // test_brace_before_code_fence_helper_accepts_literal_container_after_fence_with_trailing_comma
        let parser = parser_in("}``` [1,2],\n\", \"b\": 1");
        let mut state = StringParseState::new();
        assert!(parser.brace_before_code_fence_belongs_to_string(&mut state, 1));
    }

    #[test]
    fn brace_before_code_fence_accepts_comment_prefixed_literal_with_trailing_comma() {
        // test_brace_before_code_fence_helper_accepts_comment_prefixed_literal_container_after_fence_with_trailing_comma
        let parser = parser_in("}``` // c\n [1,2],\n\", \"b\": 1");
        let mut state = StringParseState::new();
        assert!(parser.brace_before_code_fence_belongs_to_string(&mut state, 1));
    }

    // ---- _skip_inline_container / _starts_nested_inline_container (direct) ----

    #[test]
    fn skip_inline_container_returns_same_index_for_non_container() {
        // test_skip_inline_container_returns_same_index_for_non_container
        let parser = parser_in("text");
        assert_eq!(parser.skip_inline_container(0), Some(0));
    }

    #[test]
    fn starts_nested_inline_container_accepts_container_at_start() {
        // test_starts_nested_inline_container_accepts_container_at_start
        let parser = parser_in("[1, 2]");
        assert!(parser.starts_nested_inline_container(0));
    }

    #[test]
    fn starts_nested_inline_container_accepts_out_of_range_prefix() {
        // test_starts_nested_inline_container_accepts_out_of_range_prefix_conservatively
        let parser = parser_in("[1, 2]");
        assert!(parser.starts_nested_inline_container(10));
    }

    #[test]
    fn starts_nested_inline_container_rejects_unmatched_inner_array() {
        // test_starts_nested_inline_container_rejects_unmatched_inner_array_after_comma
        let parser = parser_in("[foo, [bar]");
        assert!(!parser.starts_nested_inline_container(6));
    }

    #[test]
    fn starts_nested_inline_container_accepts_object_with_quoted_key() {
        // test_starts_nested_inline_container_accepts_object_with_quoted_key_after_comma
        let parser = parser_in(r#"[foo, {"k": 1}]"#);
        assert!(parser.starts_nested_inline_container(6));
    }

    #[test]
    fn starts_nested_inline_container_accepts_numeric_array() {
        // test_starts_nested_inline_container_accepts_numeric_array_after_comma
        let parser = parser_in("[foo, [2]]");
        assert!(parser.starts_nested_inline_container(6));
    }

    #[test]
    fn starts_nested_inline_container_accepts_numeric_parenthesized() {
        // test_starts_nested_inline_container_accepts_numeric_parenthesized_value_after_comma
        let parser = parser_in("(foo, (2))");
        assert!(parser.starts_nested_inline_container(6));
    }

    #[test]
    fn starts_nested_inline_container_accepts_object_with_bare_key_after_colon() {
        // test_starts_nested_inline_container_accepts_object_with_bare_key_after_colon
        let parser = parser_in("{foo: {bar: 1}}");
        assert!(parser.starts_nested_inline_container(6));
    }

    #[test]
    fn starts_nested_inline_container_rejects_object_with_bare_key_after_comma() {
        // test_starts_nested_inline_container_rejects_object_with_bare_key_after_comma
        let parser = parser_in("[foo, {bar}]");
        assert!(!parser.starts_nested_inline_container(6));
    }

    #[test]
    fn starts_nested_inline_container_rejects_non_container_after_separator() {
        // test_starts_nested_inline_container_rejects_non_container_after_separator
        let parser = parser_in("[foo, xbar]");
        assert!(!parser.starts_nested_inline_container(6));
    }

    #[test]
    fn starts_nested_inline_container_rejects_non_key_start_after_colon() {
        // test_starts_nested_inline_container_rejects_object_with_non_key_start_after_colon
        let parser = parser_in("{foo: {-bar}}");
        assert!(!parser.starts_nested_inline_container(6));
    }

    #[test]
    fn skip_inline_container_skips_nested_inline_container() {
        // test_skip_inline_container_skips_nested_inline_container
        let parser = parser_in("[{items: [1, 2]}] tail");
        assert_eq!(parser.skip_inline_container(0), Some(17));
    }

    #[test]
    fn skip_inline_container_keeps_hash_like_literal_content() {
        // test_skip_inline_container_keeps_hash_like_literal_content
        let parser = parser_in("[# literal] tail");
        assert_eq!(parser.skip_inline_container(0), Some(11));
    }

    #[test]
    fn skip_inline_container_keeps_line_comment_like_literal_content() {
        // test_skip_inline_container_keeps_line_comment_like_literal_content
        let parser = parser_in("[http://x] tail");
        assert_eq!(parser.skip_inline_container(0), Some(10));
    }

    #[test]
    fn skip_inline_container_keeps_block_comment_like_literal_content() {
        // test_skip_inline_container_keeps_block_comment_like_literal_content
        let parser = parser_in("[a/*b*/c] tail");
        assert_eq!(parser.skip_inline_container(0), Some(9));
    }

    #[test]
    fn skip_inline_container_keeps_regex_like_literal_content() {
        // test_skip_inline_container_keeps_regex_like_literal_content
        let parser = parser_in("[/a//b/] tail");
        assert_eq!(parser.skip_inline_container(0), Some(8));
    }

    #[test]
    fn skip_inline_container_keeps_unmatched_inner_delimiter() {
        // test_skip_inline_container_keeps_unmatched_inner_delimiter_as_literal_content
        let parser = parser_in("[foo[bar] tail");
        assert_eq!(parser.skip_inline_container(0), Some(9));
    }

    #[test]
    fn skip_inline_container_rejects_unterminated_container() {
        // test_skip_inline_container_rejects_unterminated_container
        let parser = parser_in("[1, 2");
        assert_eq!(parser.skip_inline_container(0), None);
    }

    #[test]
    fn skip_inline_container_rejects_unterminated_string_inside() {
        // test_skip_inline_container_rejects_unterminated_string_inside_container
        let parser = parser_in(r#"["unterminated"#);
        assert_eq!(parser.skip_inline_container(0), None);
    }

    #[test]
    fn skip_inline_container_rejects_unterminated_block_comment() {
        // test_skip_inline_container_rejects_unterminated_block_comment
        let parser = parser_in("[/* c");
        assert_eq!(parser.skip_inline_container(0), None);
    }

    // ---- update_inline_container_stack (direct) ----

    #[test]
    fn update_inline_container_stack_starts_tracking_pending_container() {
        // test_update_inline_container_stack_starts_tracking_pending_container
        let mut inline_container_stack: Vec<char> = Vec::new();
        let (pending, keep) = update_inline_container_stack('[', true, &mut inline_container_stack);
        assert!(!pending);
        assert!(!keep);
        assert_eq!(inline_container_stack, vec!['[']);
    }

    #[test]
    fn update_inline_container_stack_tracks_nested_container() {
        // test_update_inline_container_stack_tracks_nested_container
        let mut inline_container_stack = vec!['['];
        let (pending, keep) =
            update_inline_container_stack('{', false, &mut inline_container_stack);
        assert!(!pending);
        assert!(!keep);
        assert_eq!(inline_container_stack, vec!['[', '{']);
    }

    #[test]
    fn update_inline_container_stack_keeps_closing_container_character() {
        // test_update_inline_container_stack_keeps_closing_container_character
        let mut inline_container_stack = vec!['['];
        let (pending, keep) =
            update_inline_container_stack(']', false, &mut inline_container_stack);
        assert!(!pending);
        assert!(keep);
        assert_eq!(inline_container_stack, Vec::<char>::new());
    }

    // ---- _scan_string_body (direct, constructed states) ----

    #[test]
    fn scan_string_body_keeps_closing_inline_container_character() {
        // test_scan_string_body_keeps_closing_inline_container_character
        let mut parser = parser_in(r#"]""#);
        parser.ctx_push(Ctx::ObjectValue);
        let mut state = StringParseState {
            string_acc: "x".to_string(),
            inline_container_stack: vec!['['],
            ..StringParseState::new()
        };
        let ch = parser.scan_string_body(&mut state);
        assert_eq!(ch, Some('"'));
        assert_eq!(state.string_acc, "x]");
        assert_eq!(state.inline_container_stack, Vec::<char>::new());
    }

    #[test]
    fn scan_string_body_closes_low_smart_quote_span_with_unicode_quote() {
        // test_scan_string_body_closes_low_smart_quote_span_with_unicode_quote
        let mut parser = parser_in("\u{201D}\"");
        parser.ctx_push(Ctx::ObjectValue);
        let mut state = StringParseState {
            rstring_delimiter: vec!['"', LOW_SMART_QUOTE_SENTINEL],
            string_acc: "prefix".to_string(),
            ..StringParseState::new()
        };
        let ch = parser.scan_string_body(&mut state);
        assert_eq!(ch, Some('"'));
        assert_eq!(state.string_acc, "prefix\u{201D}");
        assert_eq!(state.rstring_delimiter, vec!['"']);
    }

    // ---- _quoted_object_member_follows (direct) ----

    #[test]
    fn quoted_object_member_follows_rejects_unquoted_next_key() {
        // test_quoted_object_member_follows_rejects_unquoted_next_key
        let parser = parser_in("\"\n\", bare");
        assert!(!parser.quoted_object_member_follows(2));
    }

    #[test]
    fn quoted_object_member_follows_rejects_unterminated_next_key() {
        // test_quoted_object_member_follows_rejects_unterminated_next_key
        let parser = parser_in("\"\n\", \"unterminated");
        assert!(!parser.quoted_object_member_follows(2));
    }

    #[test]
    fn quoted_object_member_follows_accepts_single_quoted_next_key() {
        // test_quoted_object_member_follows_accepts_single_quoted_next_key
        let parser = parser_in("\"\n\", 'b': 1");
        assert!(parser.quoted_object_member_follows(2));
    }

    #[test]
    fn quoted_object_member_follows_accepts_curly_quoted_next_key() {
        // test_quoted_object_member_follows_accepts_curly_quoted_next_key
        let parser = parser_in("\"\n\", \u{201C}b\u{201D}: 1");
        assert!(parser.quoted_object_member_follows(2));
    }

    #[test]
    fn quoted_object_member_follows_accepts_comment_before_next_key() {
        // test_quoted_object_member_follows_accepts_comment_before_next_key
        let parser = parser_in("\"\n\", // c\n\"b\": 1");
        assert!(parser.quoted_object_member_follows(2));
    }

    #[test]
    fn quoted_object_member_follows_accepts_hash_comment_before_next_key() {
        // test_quoted_object_member_follows_accepts_hash_comment_before_next_key
        let parser = parser_in("\"\n\", # c\n\"b\": 1");
        assert!(parser.quoted_object_member_follows(2));
    }

    #[test]
    fn quoted_object_member_follows_accepts_block_comment_before_next_key() {
        // test_quoted_object_member_follows_accepts_block_comment_before_next_key
        let parser = parser_in("\"\n\", /* c */ \"b\": 1");
        assert!(parser.quoted_object_member_follows(2));
    }

    #[test]
    fn quoted_object_member_follows_accepts_comment_before_bare_next_key() {
        // test_quoted_object_member_follows_accepts_comment_before_bare_next_key
        let parser = parser_in("\"\n\", // c\n b: 1");
        assert!(parser.quoted_object_member_follows(2));
    }

    #[test]
    fn quoted_object_member_follows_accepts_bare_next_key() {
        // test_quoted_object_member_follows_accepts_bare_next_key
        let parser = parser_in("\"\n\",\n b: 1");
        assert!(parser.quoted_object_member_follows(2));
    }

    #[test]
    fn quoted_object_member_follows_rejects_trailing_comma_endings() {
        // test_quoted_object_member_follows_rejects_trailing_comma_endings
        assert!(!parser_in("\"\n\",}").quoted_object_member_follows(2));
        assert!(!parser_in("\"\n\",").quoted_object_member_follows(2));
    }

    #[test]
    fn quoted_object_member_follows_rejects_unclosed_block_comment() {
        // test_quoted_object_member_follows_rejects_unclosed_block_comment_before_next_key
        let parser = parser_in("\"\n\", /* c");
        assert!(!parser.quoted_object_member_follows(2));
    }

    #[test]
    fn quoted_object_member_follows_rejects_array_after_comment() {
        // test_quoted_object_member_follows_rejects_array_after_comment
        let parser = parser_in("\"\n\", /* c */ [1, 2]");
        assert!(!parser.quoted_object_member_follows(2));
    }

    // ---- parse_string battery (context pushed, values pinned against the
    // pinned upstream parser) ----

    #[test]
    fn parse_string_fast_path_and_boundaries() {
        // test_parse_string / test_parse_string_fast_path_keeps_clean_values_log_free
        assert_eq!(
            parse_string_in(r#""value""#, Ctx::ObjectValue),
            Ok(str_value("value"))
        );
        // _try_parse_simple_quoted_string context boundary checks
        assert_eq!(
            parse_string_in(r#""value","#, Ctx::ObjectValue),
            Ok(str_value("value"))
        );
        assert_eq!(
            parse_string_in(r#""value"}"#, Ctx::ObjectValue),
            Ok(str_value("value"))
        );
        assert_eq!(
            parse_string_in(r#""value"]"#, Ctx::Array),
            Ok(str_value("value"))
        );
        assert_eq!(
            parse_string_in(r#""value":"#, Ctx::ObjectKey),
            Ok(str_value("value"))
        );
        // trailing text after the closer falls back to the slow path
        assert_eq!(
            parse_string_in(r#""value" trailing"#, Ctx::ObjectValue),
            Ok(str_value("value"))
        );
        // empty value then next member
        assert_eq!(
            parse_string_in(r#""", "b": 1}"#, Ctx::ObjectValue),
            Ok(str_value(""))
        );
        // test_missing_and_mixed_quotes: single-quoted delimiters
        assert_eq!(
            parse_string_in(r#"'single'"#, Ctx::ObjectValue),
            Ok(str_value("single"))
        );
        assert_eq!(
            parse_string_in(r#"'"'"#, Ctx::ObjectValue),
            Ok(str_value("\""))
        );
    }

    #[test]
    fn parse_string_missing_quotes() {
        // test_missing_and_mixed_quotes (unquoted value / key fragments)
        assert_eq!(
            parse_string_in("value", Ctx::ObjectValue),
            Ok(str_value("value"))
        );
        assert_eq!(
            parse_string_in("key: 1", Ctx::ObjectKey),
            Ok(str_value("key"))
        );
        // missing right quote: rstrip trailing whitespace (the "New York" case)
        assert_eq!(
            parse_string_in("\"value  ", Ctx::ObjectValue),
            Ok(str_value("value"))
        );
        assert_eq!(
            parse_string_in("value\n", Ctx::ObjectValue),
            Ok(str_value("value"))
        );
        // test_escaping: a key fragment ending at ':'
        assert_eq!(
            parse_string_in(r#""key\t_""#, Ctx::ObjectKey),
            Ok(str_value("key\t_"))
        );
        assert_eq!(parse_string_in(r#"":"#, Ctx::ObjectKey), Ok(str_value(":")));
    }

    #[test]
    fn parse_string_escape_normalization() {
        // test_escaping
        assert_eq!(
            parse_string_in(r#""a\tb""#, Ctx::ObjectValue),
            Ok(str_value("a\tb"))
        );
        assert_eq!(
            parse_string_in(r#""\u0076alue""#, Ctx::ObjectValue),
            Ok(str_value("value"))
        );
        // escaped wrong delimiter: the escape is removed
        assert_eq!(
            parse_string_in(r#""valu\'e""#, Ctx::ObjectValue),
            Ok(str_value("valu'e"))
        );
        // backslash runs halve when even and not escaping the delimiter
        assert_eq!(
            parse_string_in(r#""a\\b""#, Ctx::ObjectValue),
            Ok(str_value(r"a\b"))
        );
        // test_parse_string_preserves_repeated_escaped_backslashes_during_repair
        assert_eq!(
            parse_string_in(r#""a\\b\\c""#, Ctx::ObjectValue),
            Ok(str_value(r"a\b\c"))
        );
        assert_eq!(
            parse_string_in(r#""a\\\\b\\\\c""#, Ctx::ObjectValue),
            Ok(str_value(r"a\\b\\c"))
        );
        // single-quoted string with escapes and a bare double quote inside
        assert_eq!(
            parse_string_in("'string\"\n\t\\le'", Ctx::ObjectValue),
            Ok(str_value("string\"\n\t\\le"))
        );
    }

    #[test]
    fn parse_string_low_smart_quote_spans() {
        // test_parse_string_keeps_low_smart_quote_span_closed_by_ascii_quote
        assert_eq!(
            parse_string_in(
                r#""despre „autocritică" și autocompasiune""#,
                Ctx::ObjectValue
            ),
            Ok(str_value("despre „autocritică\" și autocompasiune"))
        );
        // test_parse_string_keeps_low_smart_quote_span_closed_by_unicode_quote
        assert_eq!(
            parse_string_in(
                r#""despre „autocritică” și autocompasiune""#,
                Ctx::ObjectValue
            ),
            Ok(str_value("despre „autocritică” și autocompasiune"))
        );
        // test_parse_string_keeps_low_smart_quote_span_closed_by_escaped_ascii_quote
        assert_eq!(
            parse_string_in(r#""aplicație „sham\"), a făcut""#, Ctx::ObjectValue),
            Ok(str_value("aplicație „sham\"), a făcut"))
        );
        // test_parse_string_escaped_low_smart_quote_does_not_open_inner_span
        assert_eq!(
            parse_string_in(r#""a \„ b", "y": 1"#, Ctx::ObjectValue),
            Ok(str_value("a „ b"))
        );
        // test_parse_string_keeps_multiline_curly_quoted_prose_after_comma
        // (a left curly quote inside the value is ordinary content)
        assert_eq!(
            parse_string_in(
                "\"a,\n \u{201C}term\u{201D}: explanation\", \"y\": 2",
                Ctx::ObjectValue
            ),
            Ok(str_value("a,\n \u{201C}term\u{201D}: explanation"))
        );
    }

    #[test]
    fn parse_string_regex_character_classes() {
        // test_parse_string_keeps_bare_quotes_inside_regex_character_classes
        assert_eq!(
            parse_string_in(r#""^\s*path\(\s*['"]([^'"]+)['"]\s*,""#, Ctx::ObjectValue),
            Ok(str_value(r#"^\s*path\(\s*['"]([^'"]+)['"]\s*,"#))
        );
        assert_eq!(
            parse_string_in(r#""^\s*re_path\(\s*[^'"]+['"]\s*,""#, Ctx::ObjectValue),
            Ok(str_value(r#"^\s*re_path\(\s*[^'"]+['"]\s*,"#))
        );
    }

    #[test]
    fn parse_string_inline_container_after_comma() {
        // test_parse_string_keeps_inline_object_literal_after_comma
        assert_eq!(
            parse_string_in(r#""a, {"k": 1}, "y": 2}"#, Ctx::ObjectValue),
            Ok(str_value(r#"a, {"k": 1}"#))
        );
        // test_parse_string_keeps_bare_member_recovery... (member boundaries)
        assert_eq!(
            parse_string_in(r#""first, b: "second"}"#, Ctx::ObjectValue),
            Ok(str_value("first"))
        );
        assert_eq!(
            parse_string_in(r#""first, b: 1}"#, Ctx::ObjectValue),
            Ok(str_value("first"))
        );
        assert_eq!(
            parse_string_in(r#""first, b: true}"#, Ctx::ObjectValue),
            Ok(str_value("first"))
        );
        assert_eq!(
            parse_string_in(r#""first, b: [1]}"#, Ctx::ObjectValue),
            Ok(str_value("first"))
        );
        assert_eq!(
            parse_string_in(r#""first, b: prose}"#, Ctx::ObjectValue),
            Ok(str_value("first"))
        );
        // test_parse_string_preserves_escaped_braces_after_comma_group
        assert_eq!(
            parse_string_in(r#""\{1,2\} \{3\}""#, Ctx::ObjectValue),
            Ok(str_value(r"\{1,2\} \{3\}"))
        );
    }

    #[test]
    fn parse_string_literal_fenced_snippets() {
        // test_parse_string_keeps_literal_fenced_snippet_cases:
        // "inline-array-literal"
        assert_eq!(
            parse_string_in("\"x}``` [1,2]\\n\",\"b\":\"y\"}", Ctx::ObjectValue),
            Ok(str_value("x}``` [1,2]"))
        );
        // "inline-array-with-trailing-comma"
        assert_eq!(
            parse_string_in("\"x}``` [1,2],\\n\",\"b\":\"y\"}", Ctx::ObjectValue),
            Ok(str_value("x}``` [1,2],"))
        );
        // "comment-prefixed-inline-array"
        assert_eq!(
            parse_string_in("\"x}``` // c\\n [1,2]\\n\",\"b\":\"y\"}", Ctx::ObjectValue),
            Ok(str_value("x}``` // c\n [1,2]"))
        );
        // "multiline-object-value"
        assert_eq!(
            parse_string_in("\"\n```{}```\n\",\n\"b\": \"x\",\n}", Ctx::ObjectValue),
            Ok(str_value("\n```{}```"))
        );
        // "fenced-code-block-before-inline-quoted-prose"
        assert_eq!(
            parse_string_in(
                "\"\n```c\nint main() {\n}\n```\nImplementation: \"xxx\", xxx\n\",\n\"b\": \"x\",\n}",
                Ctx::ObjectValue
            ),
            Ok(str_value(
                "\n```c\nint main() {\n}\n```\nImplementation: \"xxx\", xxx"
            ))
        );
        // test_parse_string_stray_quote_line_before_trailing_comma_drops_stray_quote
        assert_eq!(
            parse_string_in("\"hello\n\"\n\",}", Ctx::ObjectValue),
            Ok(str_value("hello"))
        );
    }

    #[test]
    fn parse_string_object_value_brace_heuristics() {
        // test_parse_string_object_value_brace_heuristics
        assert_eq!(
            parse_string_in(r#""value} "tail" more}"#, Ctx::ObjectValue),
            Ok(str_value("value} \"tail\" more"))
        );
        assert_eq!(
            parse_string_in(r#""value}\"more"}"#, Ctx::ObjectValue),
            Ok(str_value("value}\"more"))
        );
    }

    #[test]
    fn parse_string_misplaced_quotes() {
        // test_missing_and_mixed_quotes: the '"key": "v"alue"} key:' fragment
        assert_eq!(
            parse_string_in(r#""v"alue""#, Ctx::ObjectValue),
            Ok(str_value("v\"alue\""))
        );
        assert_eq!(
            parse_string_in(r#""v"alue"} key:"#, Ctx::ObjectValue),
            Ok(str_value("v\"alue\"} key:"))
        );
        // '"key": "lorem ipsum ... "sic " tamet. ...}'
        assert_eq!(
            parse_string_in(r#""lorem ipsum ... "sic " tamet. ...}"#, Ctx::ObjectValue),
            Ok(str_value("lorem ipsum ... \"sic \" tamet. ..."))
        );
        // '"comment": "lorem, "ipsum" sic "tamet". To improve"'
        assert_eq!(
            parse_string_in(
                r#""lorem, "ipsum" sic "tamet". To improve"}"#,
                Ctx::ObjectValue
            ),
            Ok(str_value("lorem, \"ipsum\" sic \"tamet\". To improve"))
        );
        // '{"key": "{"key": 1, "key2": 1}"}' (leading inline object)
        assert_eq!(
            parse_string_in(r#""{"key": 1, "key2": 1}""#, Ctx::ObjectValue),
            Ok(str_value("{\"key\": 1"))
        );
    }

    #[test]
    fn parse_string_doubled_quotes() {
        // test_parse_string_removes_redundant_leading_quote
        assert_eq!(
            parse_string_in(r#"""value"}"#, Ctx::ObjectValue),
            Ok(str_value("value"))
        );
        // test_parse_string_preserves_leading_quoted_phrase (two leading
        // quotes: the second opens a quoted span the parser keeps)
        assert_eq!(
            parse_string_in(r#"""hello" world""#, Ctx::ObjectValue),
            Ok(str_value("\"hello\" world"))
        );
        // non-strict doubled-quote recoveries return the empty string
        assert_eq!(
            parse_string_in(r#""""value""#, Ctx::ObjectValue),
            Ok(str_value(""))
        );
        assert_eq!(
            parse_string_in(r#""" "value""#, Ctx::ObjectValue),
            Ok(str_value(""))
        );
        // strict mode raises the §8 catalog messages
        assert_eq!(
            parse_string_strict_in(r#""""value""#, Ctx::ObjectValue),
            Err("Found doubled quotes followed by another quote.".to_string())
        );
        assert_eq!(
            parse_string_strict_in(r#""" "value""#, Ctx::ObjectValue),
            Err(
                "Found doubled quotes followed by another quote while parsing a string."
                    .to_string()
            )
        );
    }

    #[test]
    fn parse_string_boolean_and_null_literals() {
        // test_parse_boolean_or_null
        assert_eq!(parse_string_in("None ", Ctx::ObjectValue), Ok(Value::Null));
        assert_eq!(parse_string_in("null", Ctx::ObjectValue), Ok(Value::Null));
        assert_eq!(
            parse_string_in("TRUE", Ctx::ObjectValue),
            Ok(Value::Bool(true))
        );
        // boundary: the char after the literal must end it
        assert_eq!(parse_string_in("true]", Ctx::Array), Ok(Value::Bool(true)));
        assert_eq!(
            parse_string_in("false)", Ctx::Array),
            Ok(Value::Bool(false))
        );
        // test_parse_literals_require_boundaries_and_support_python_none
        assert_eq!(
            parse_string_in("trueblue", Ctx::ObjectValue),
            Ok(str_value("trueblue"))
        );
        assert_eq!(
            parse_string_in("falsehood", Ctx::ObjectValue),
            Ok(str_value("falsehood"))
        );
        assert_eq!(
            parse_string_in("nullify", Ctx::ObjectValue),
            Ok(str_value("nullify"))
        );
        assert_eq!(
            parse_string_in("NoneType", Ctx::ObjectValue),
            Ok(str_value("NoneType"))
        );
    }

    #[test]
    fn parse_string_json_llm_blocks() {
        // test_string_json_llm_block
        assert_eq!(
            parse_string_in(
                r#""```json {"key": [{"key1": 1},{"key2": 2}]}```""#,
                Ctx::ObjectValue
            ),
            Ok(Value::Object(vec![(
                "key".to_string(),
                Value::Array(vec![
                    Value::Object(vec![("key1".to_string(), Value::Int(1))]),
                    Value::Object(vec![("key2".to_string(), Value::Int(2))]),
                ])
            )]))
        );
        // test_parse_string_logs_invalid_code_fences (value assertion)
        assert_eq!(
            parse_string_in(r#""```json nope\n""#, Ctx::ObjectValue),
            Ok(str_value("```json nope"))
        );
        // '{"response": "```json{}"' and '{"key": "```json"'
        assert_eq!(
            parse_string_in(r#""```json{}""#, Ctx::ObjectValue),
            Ok(str_value("```json{}"))
        );
        assert_eq!(
            parse_string_in(r#""```json""#, Ctx::ObjectValue),
            Ok(str_value("```json"))
        );
        assert_eq!(
            parse_string_in(r#""``""#, Ctx::ObjectValue),
            Ok(str_value("``"))
        );
    }

    #[test]
    fn parse_string_prose_with_colons_and_containers() {
        // test_parse_string_keeps_colon_prose_inside_wrapped_valid_json
        assert_eq!(
            parse_string_in(
                r#""x\nrollback: (registry: Registry, snapshot: EntitySnapshot) => void;""#,
                Ctx::ObjectValue
            ),
            Ok(str_value(
                "x\nrollback: (registry: Registry, snapshot: EntitySnapshot) => void;"
            ))
        );
        // test_parse_string_keeps_pseudo_object_code_inside_valid_json
        assert_eq!(
            parse_string_in(
                r#""x\nrollback: {registry: Registry, snapshot: EntitySnapshot}""#,
                Ctx::ObjectValue
            ),
            Ok(str_value(
                "x\nrollback: {registry: Registry, snapshot: EntitySnapshot}"
            ))
        );
        // the wrapped-valid-json "text" value
        assert_eq!(
            parse_string_in(
                r#""a\n b c, floof: a\n ... a b (c), floof: \n a", "id": 8"#,
                Ctx::ObjectValue
            ),
            Ok(str_value("a\n b c, floof: a\n ... a b (c), floof: \n a"))
        );
    }

    #[test]
    fn parse_string_object_value_comma_without_future_delimiter() {
        // test_object_value_comma_without_future_delimiter_scans_once:
        // upstream asserts exactly one skip_to_character call: the
        // lookahead cache answers every later comma probe in O(1); the call
        // count itself is not observable from the Rust port.
        let mut parser = Parser::new("\"value,fragment,fragment,fragment", false, None);
        parser.ctx_push(Ctx::ObjectValue);
        assert_eq!(
            parser.parse_string(),
            Ok(str_value("value,fragment,fragment,fragment"))
        );
    }

    #[test]
    fn parse_string_object_value_bare_key_prose_scans_once() {
        // test_object_value_bare_key_prose_scans_once: upstream asserts 3
        // skip_to_character calls total (the [delimiters+openers] scan, the
        // single-quote not-found scan, and the [delimiters+}] recovery
        // scan); the cache serves every later iteration without rescanning.
        let raw = format!("\"value\\n{}\"", ", floof: prose".repeat(100));
        let mut parser = Parser::new(&raw, false, None);
        parser.ctx_push(Ctx::ObjectValue);
        // Upstream normalizes the `\n` escape to a real newline (the
        // expected Python literal `"value\n..."` holds a newline, not a
        // backslash): the value keeps every `, floof: prose` repetition.
        let expected = Value::Str(format!("value\n{}", ", floof: prose".repeat(100)));
        assert_eq!(parser.parse_string(), Ok(expected));
    }

    #[test]
    fn parse_string_far_quote_comma_payload() {
        // test_parse_string_far_quote_comma_payload_keeps_existing_repair_shape
        // (smaller repeat than upstream's 10_000; the cache amortization is
        // what the upstream CountingParser variant pinned)
        let raw = format!("\"{}\" tail", "x,".repeat(1_000));
        let mut parser = Parser::new(&raw, false, None);
        parser.ctx_push(Ctx::ObjectValue);
        assert_eq!(parser.parse_string(), Ok(Value::Str("x,".repeat(1_000))));
    }
}
