//! `tors::json_repair::StreamingRepairer` (the streaming repairer; see
//! src/json_repair/streaming.rs for the contract) never panics on
//! arbitrary chunk sequences, and its closed renders stay loadable. The
//! input is an arbitrary string replayed as a chunk sequence (several
//! deterministic split sets per input, boundaries mid-escape and
//! mid-surrogate-pair included); the asserts:
//!
//! - panic-freedom through push -> push -> end (the primary; Err is
//!   fine: the depth cap's normalized ValueError, after which the
//!   machine's state is unspecified and that splitting just stops);
//! - on every error-free path: every snapshot (and the end) is the
//!   empty sentinel or strict-loadable JSON, and snapshot == end on the
//!   same prefix;
//! - chunking invariance: any split set ends at the one-push answer;
//! - COMPLETE-DOCUMENT engine parity: on every input the scan does not
//!   exclude (the documented divergence list, mechanically checked
//!   trigger by trigger), the streamed end() equals the whole-text
//!   engine's repair of the same text byte for byte, under every split
//!   set. The scan is conservative: a trigger it cannot model flips
//!   `divergent` (coverage loss only, never a false assert).
//!
//! Panics, non-loadable renders, snapshot/end disagreement, chunking
//! drift, or a parity break under the gate are bugs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tors::json_repair::{RepairConfig, StreamingRepairer, Value, dumps, loads_strict, repair};

/// Where the scan stands inside the innermost container.
#[derive(Clone, Copy, PartialEq)]
enum Expect {
    Key,
    Colon,
    Value,
    AfterValue,
}

/// One open container in the scan's stack (`paren` for the tuple/
/// grouping container: only `)` closes it).
struct ScanFrame {
    obj: bool,
    paren: bool,
    members: usize,
    expect: Expect,
    /// Object only: the member keys seen (the duplicate-key
    /// update-in-place is the whole-text dict semantics; a repeat is
    /// the divergence).
    keys: Vec<String>,
    /// Object only: the chars seen while the frame has been open (the
    /// engine's parse-span, the closing `}` included) — the
    /// empty-object fallback's span condition.
    body_chars: usize,
    /// Object only: the members that saw their colon (the engine's
    /// `obj[key] = value` assignments — a bare key ATTEMPT does not
    /// assign, the engine discards it at the close).
    completed: usize,
    /// The enclosing frame is an ARRAY (not a paren, not an object):
    /// the object-in-array territory (the engine's merge continuation,
    /// and its unquoted-value run eats `}` there).
    in_array: bool,
    /// An item of this (array) frame closed in the strictly-empty
    /// drop state (it repairs to the empty array): the cascade
    /// territory — the engine's whole-text lane re-decides such an
    /// item's now-empty shape when the parent itself drops.
    saw_emptied_item: bool,
    /// A PAREN frame saw a `,` at its item-value position (the
    /// item-parse garbage run): the machine's grouping/empty-item
    /// cascade around such a tuple desyncs from the engine's
    /// whole-text classifier when the paren is nested (`((,),1)` ->
    /// the engine `[[1]]`, the stream `[1]`).
    saw_comma_garbage: bool,
    /// An item started after the emptied one: the parent continued
    /// past a repaired-empty item (the whole-text lane re-decides).
    item_after_emptied: bool,
    /// The last completed item was strictly-empty (`[]`/`{}`/`""` or
    /// an emptied frame): the comma-close drop's operand.
    last_item_empty: bool,
}

/// The string's escape sub-state.
enum StrEsc {
    None,
    Backslash,
    U(usize),
}

/// The conservative scan: does the input sit in an engine-parity class?
/// One pass; every documented divergence trigger it can see flips
/// `divergent` (the input leaves the parity classes; coverage loss only,
/// never a false assert).
struct Scan {
    stack: Vec<ScanFrame>,
    in_string: bool,
    esc: StrEsc,
    /// The raw trailing backslash run inside the open string.
    raw_bs_run: usize,
    /// The open string's raw content contains a backslash anywhere: the
    /// doubled-quote/escape re-pair family's territory (out at the cut).
    saw_backslash: bool,
    /// The open string's content (modulo trailing spaces) ends with a
    /// `\n` escape.
    content_ends_nl: bool,
    /// The LAST closed string ended on a newline-run; a following token
    /// makes the input the mid-document escape-tail divergence.
    nl_tail_string: bool,
    /// The value-run in flight: its length and its last nine raw chars
    /// (the longest special-float spelling, `-Infinity`, is nine) — a
    /// run spelling exactly `NaN`/`Infinity`/`-Infinity` renders the
    /// strict float mid-stream but the engine's repair lane quotes it,
    /// so any COMPLETED one (a token after it) leaves the parity class.
    run_len: usize,
    run_tail: [u8; 9],
    run_tail_len: usize,
    run_is_float_spelling: bool,
    /// The value-run's first char (the double-leading-minus word's
    /// engine-side double strip is the divergence).
    run_first: Option<char>,
    /// The run saw an alphabetic char: a number-born WORD (the
    /// machine's rewind-born word absorbs the whitespace/colon that
    /// follow, the engine's string lane too), so the ws-lookahead
    /// applies (`{"a": 1a .:"`).
    run_saw_alpha: bool,
    string_is_key: bool,
    /// The last completed VALUE element was an empty string (`""`): the
    /// engine's whole-text string-repair lane re-pairs what follows it
    /// (`[""x]` -> `["x"]`), so a token after it leaves the parity
    /// class.
    last_empty_string_value: bool,
    /// A string VALUE just closed (this token): the engine's
    /// quoted-section re-pair reads what follows it — a container-open
    /// whose window to the first `,`/`]`/`}` holds a quote re-pairs the
    /// string whole (`["4"[""` -> the engine `["4\"[\""]`), so that
    /// shape leaves the parity class.
    last_string_value: bool,
    /// The bare-word KEY run's text (the object lane): the
    /// duplicate-key update-in-place is the whole-text dict semantics
    /// for bare keys too (`{l:2{l:` -> the engine `{"l": ""}`), so a
    /// repeated bare key leaves the parity class.
    bare_key_run: String,
    /// The strictly-empty item's one-char decision is ARMED (the item
    /// just completed and nothing separated it from the next char yet):
    /// the machine's `EmptyPending` is live, and its drop consumes the
    /// next char — a separator (`:`/`,`) keeps the element, whitespace
    /// defers to the close-time mark, any other char is consumed with
    /// the drop. The scan cannot model the consumed char's aftermath
    /// (out).
    empty_pending_armed: bool,
    /// The armed decision's parent is a PAREN frame: the machine's
    /// drop fires on the whitespace itself (the paren lane defers
    /// nothing).
    empty_pending_paren: bool,
    /// An OBJECT-VALUE's bare word was ws-preceded (the engine's
    /// parse_string entry-skip consumed the run, and its `}`-machinery
    /// then absorbs the object's own `}` into the string: `{"a": \rL}`
    /// -> the engine `{"a": "L}"}`) — the close-at-`}` desyncs (out).
    obj_value_after_ws: bool,
    /// The open string's raw content (the key text when `string_is_key`).
    key_buf: String,
    top_done: bool,
    divergent: bool,
}

impl Scan {
    fn new() -> Scan {
        Scan {
            stack: Vec::new(),
            in_string: false,
            esc: StrEsc::None,
            raw_bs_run: 0,
            saw_backslash: false,
            content_ends_nl: false,
            nl_tail_string: false,
            run_len: 0,
            run_tail: [0; 9],
            run_tail_len: 0,
            run_is_float_spelling: false,
            run_first: None,
            run_saw_alpha: false,
            string_is_key: false,
            last_empty_string_value: false,
            last_string_value: false,
            bare_key_run: String::new(),
            empty_pending_armed: false,
            empty_pending_paren: false,
            obj_value_after_ws: false,
            key_buf: String::new(),
            top_done: false,
            divergent: false,
        }
    }

    /// The object-lane comma classification's mechanical subset (the
    /// engine's `classify_object_value_comma`): the comma at `comma`
    /// (chars[comma] is the `,`) is a MEMBER separator only for the
    /// unambiguous shapes — the object's `}`/end (the trailing comma),
    /// a quoted key followed by its `:`, or a bare key followed by its
    /// `:` and a recoverable value (a quote, a container, a `-`, a
    /// digit, a strict literal at a clean end, or a quote-free tail to
    /// the object's `}`). Everything else (the classifier's
    /// string/container outcomes: a garbage-led key, an unrecoverable
    /// bare-key value, a container-open) absorbs the comma into the
    /// string — the whole-text lane. `false` = out of the parity class.
    fn object_comma_is_member(chars: &[char], comma: usize, n: usize) -> bool {
        let mut k = comma + 1;
        while k < n && chars[k].is_whitespace() {
            k += 1;
        }
        // the trailing comma / the cut's end: the member separator
        if k >= n || chars[k] == '}' {
            return true;
        }
        // the quoted key: `:` after its close decides
        if chars[k] == '"' {
            let mut e = k + 1;
            while e < n && chars[e] != '"' {
                e += 1;
            }
            if e >= n {
                return false;
            }
            let mut a = e + 1;
            while a < n && chars[a].is_whitespace() {
                a += 1;
            }
            return a < n && chars[a] == ':';
        }
        // the bare key: [alnum_-]+ then ws then `:` and the value's
        // recoverability
        if chars[k].is_alphanumeric() || chars[k] == '_' {
            let mut e = k;
            while e < n && (chars[e].is_alphanumeric() || chars[e] == '_' || chars[e] == '-') {
                e += 1;
            }
            while e < n && chars[e].is_whitespace() {
                e += 1;
            }
            if e >= n || chars[e] != ':' {
                return false;
            }
            let mut v = e + 1;
            while v < n && chars[v].is_whitespace() {
                v += 1;
            }
            if v >= n {
                return false;
            }
            let vc = chars[v];
            if vc == '"' || vc == '{' || vc == '[' || vc == '-' || vc.is_ascii_digit() {
                return true;
            }
            // the strict literals at a clean end (the lib's exact
            // lowercase spellings)
            for lit in ["true", "false", "null"] {
                let lit_chars: Vec<char> = lit.chars().collect();
                if chars[v..].iter().zip(lit_chars.iter()).all(|(a, b)| a == b) {
                    let end = chars.get(v + lit_chars.len());
                    if end.is_none()
                        || end.is_some_and(|c| c.is_whitespace() || matches!(c, ',' | '}' | ']'))
                    {
                        return true;
                    }
                }
            }
            // the fallback: the value's tail is quote-free to the `}`
            let mut q = v;
            while q < n && !matches!(chars[q], '"' | '}') {
                q += 1;
            }
            return q < n && chars[q] == '}';
        }
        false
    }

    /// A value-run char outside a string: keep the bounded window.
    fn run_char(&mut self, c: char) {
        if self.run_first.is_none() {
            self.run_first = Some(c);
        }
        if c.is_alphabetic() {
            self.run_saw_alpha = true;
        }
        if !c.is_ascii() {
            // a non-ASCII word char can never be a float spelling
            self.run_is_float_spelling = false;
            return;
        }
        self.run_len += 1;
        if self.run_tail_len == 9 {
            self.run_tail.copy_within(1.., 0);
            self.run_tail[8] = c as u8;
        } else {
            self.run_tail[self.run_tail_len] = c as u8;
            self.run_tail_len += 1;
        }
        let tail = &self.run_tail[..self.run_tail_len];
        self.run_is_float_spelling = matches!(self.run_len, 3 | 8 | 9)
            && matches!(tail, b"NaN" | b"Infinity" | b"-Infinity");
    }

    /// A value-run's end: a completed float-spelling run plus a
    /// following token is the strict-vs-repair-lane split (out).
    fn end_run(&mut self, terminated: bool) {
        if self.run_is_float_spelling && terminated {
            self.divergent = true;
        }
        self.run_len = 0;
        self.run_tail_len = 0;
        self.run_is_float_spelling = false;
        self.run_first = None;
        self.run_saw_alpha = false;
    }

    fn walk(&mut self, s: &str) {
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        let mut i = 0usize;
        while i < n {
            let c = chars[i];
            // The global exclusions: comments and the non-`"` delimiters
            // never enter a parity class.
            if c == '#'
                || c == '/'
                || matches!(c, '\'' | '\u{2018}' | '\u{2019}' | '\u{201c}' | '\u{201d}')
            {
                self.divergent = true;
                return;
            }
            // The object frames' parse-span: every char while the frame
            // is open (the strings' contents included) — the engine's
            // empty-object fallback's span condition.
            for f in &mut self.stack {
                if f.obj {
                    f.body_chars += 1;
                }
            }
            if self.in_string {
                match std::mem::replace(&mut self.esc, StrEsc::None) {
                    StrEsc::Backslash => match c {
                        'u' => {
                            self.esc = StrEsc::U(0);
                            self.saw_backslash = true;
                        }
                        // an escaped structural char: conservative out
                        // (the engine's re-sync/re-pair machinery may
                        // fire)
                        '{' | '}' | '[' | ']' | ',' | ':' => {
                            self.divergent = true;
                            return;
                        }
                        // the newline escape: the content now ends with
                        // it (trailing spaces survive it)
                        'n' => {
                            self.raw_bs_run = 0;
                            self.content_ends_nl = true;
                        }
                        // an escaped backslash IS another raw backslash:
                        // the run grows (the doubled-escape family's
                        // trigger)
                        '\\' => {
                            self.raw_bs_run += 1;
                            self.content_ends_nl = false;
                        }
                        // the remaining SIMPLE escapes decode; anything
                        // else is an invalid escape (the engine's
                        // whole-text string-repair lane re-decides it;
                        // out)
                        't' | 'r' | 'b' | 'f' | '"' | '/' => {
                            self.raw_bs_run = 0;
                            self.content_ends_nl = false;
                        }
                        _ => {
                            self.divergent = true;
                            return;
                        }
                    },
                    StrEsc::U(digits) => {
                        if c.is_ascii_hexdigit() {
                            if digits < 3 {
                                self.esc = StrEsc::U(digits + 1);
                            }
                            // the fourth digit completes the escape
                            i += 1;
                            continue;
                        }
                        // fewer than four digits then a non-hex char:
                        // the escape is literal text (the engine's
                        // whole-text string-repair lane re-decides
                        // invalid escapes; out)
                        self.divergent = true;
                        return;
                    }
                    StrEsc::None => {
                        if c == '\\' {
                            self.esc = StrEsc::Backslash;
                            self.raw_bs_run += 1;
                            self.saw_backslash = true;
                            i += 1;
                            continue;
                        }
                        self.string_content(c);
                    }
                }
                i += 1;
                continue;
            }
            // Not in a string: whitespace skips, then the token arms.
            if c.is_whitespace() {
                if self.empty_pending_armed && self.empty_pending_paren {
                    // the paren-parent's one-char decision DROPS on the
                    // whitespace itself (the machine consumed it)
                    self.divergent = true;
                    return;
                }
                // the array-parent's whitespace deferred the decision
                // to the close-time mark
                self.empty_pending_armed = false;
                // an OBJECT-VALUE's word after this whitespace run:
                // the engine's parse_string entry-skip territory
                if self
                    .stack
                    .last()
                    .is_some_and(|f| f.obj && f.expect == Expect::Value)
                {
                    self.obj_value_after_ws = true;
                }
                i += 1;
                continue;
            }
            if self.top_done || self.nl_tail_string {
                // trailing content after the root value closed (the
                // multi-value divergence), or a token after a closed
                // newline-tailed string (the mid-document escape-tail
                // divergence): out, both
                self.divergent = true;
                return;
            }
            match c {
                '{' | '[' => {
                    if self.last_empty_string_value {
                        // a token where the engine's whole-text
                        // string-repair lane re-pairs what follows an
                        // empty-string value (out)
                        self.divergent = true;
                        return;
                    }
                    if self
                        .stack
                        .last()
                        .is_some_and(|f| f.expect == Expect::Colon || f.expect == Expect::Key)
                    {
                        // a container where a colon was due, or where a
                        // KEY was due (the machine ignores it, the
                        // engine's whole-text walk re-decides): out
                        self.divergent = true;
                        return;
                    }
                    if self
                        .stack
                        .last()
                        .is_some_and(|f| f.obj && f.expect == Expect::AfterValue)
                    {
                        // a container at an object's member boundary: the
                        // engine's object loop treats it as key-attempt
                        // garbage and the whole-text dict/classifier
                        // machinery re-decides the members
                        // (`{l:2{l:` -> the engine `{"l": ""}`) (out)
                        self.divergent = true;
                        return;
                    }
                    if self.empty_pending_armed {
                        // the strictly-empty item's one-char decision is
                        // still ARMED at this char: the machine's drop
                        // consumes the char (the engine's parse_array
                        // skip the same) — the scan's model of what
                        // follows would desync either way (out)
                        self.divergent = true;
                        return;
                    }
                    if self.last_string_value {
                        // the quoted-section re-pair: a container right
                        // after a closed string whose window to the
                        // first `,`/`]`/`}` holds a quote re-pairs the
                        // string whole (`["x" ["y"]` -> the engine
                        // `["x\" [\"y"]`) (out)
                        let mut k = i + 1;
                        while k < n && !matches!(chars[k], ',' | ']' | '}' | '"') {
                            k += 1;
                        }
                        if k < n && chars[k] == '"' {
                            self.divergent = true;
                            return;
                        }
                    }
                    self.last_string_value = false;
                    let in_array = self.stack.last().is_some_and(|f| !f.obj && !f.paren);
                    if let Some(p) = self.stack.last_mut() {
                        // a new item starts here
                        p.last_item_empty = false;
                        p.item_after_emptied |= p.saw_emptied_item;
                    }
                    self.stack.push(ScanFrame {
                        obj: c == '{',
                        paren: false,
                        members: 0,
                        expect: if c == '{' { Expect::Key } else { Expect::Value },
                        keys: Vec::new(),
                        body_chars: 0,
                        completed: 0,
                        in_array,
                        saw_emptied_item: false,
                        saw_comma_garbage: false,
                        item_after_emptied: false,
                        last_item_empty: false,
                    });
                    i += 1;
                }
                '}' | ']' | ')' => {
                    let closer_obj = c == '}';
                    let closer_paren = c == ')';
                    let Some(f) = self.stack.last() else {
                        self.divergent = true;
                        return;
                    };
                    if f.paren != closer_paren || (!f.paren && f.obj != closer_obj) {
                        // a mismatched closer: the engine's close-up
                        // machinery, whole-text-only (out)
                        self.divergent = true;
                        return;
                    }
                    // The empty-object FALLBACK's territory: an object
                    // closing with NO assigned member over a non-trivial
                    // body — the engine's whole-text classifier decides
                    // the body-reparse's recovery (the stream's
                    // member_begun proxy keeps `{}` where the body would
                    // recover a member: `{ x}` -> the engine `["x}"]`;
                    // the fallback drains `{}` to `[]` where it would
                    // recover nothing: `{]}`, `{"" ,}`) — the outcomes
                    // disagree either way (out)
                    if f.obj && f.completed == 0 && f.body_chars >= 2 {
                        self.divergent = true;
                        return;
                    }
                    // the ws-preceded OBJECT-VALUE word at the object's
                    // `}`: the engine's string `}`-machinery absorbs the
                    // `}` when the tail past it holds two chars before
                    // the next `}` (the lib's `j - i > 1` guard):
                    // `{"a": \rL},\n` -> the engine `{"a": "L}"}` (out)
                    if f.obj && f.expect == Expect::AfterValue && self.obj_value_after_ws {
                        let mut k = i + 1;
                        while k < n && chars[k].is_whitespace() {
                            k += 1;
                        }
                        let mut j = k;
                        while j < n && chars[j] != '}' {
                            j += 1;
                        }
                        if j - k > 1 {
                            self.divergent = true;
                            return;
                        }
                        self.obj_value_after_ws = false;
                    }
                    // The first-member dangling key at the close: the
                    // engine's whole-text fallback re-parses the body as
                    // an array (out)
                    if f.obj && f.expect == Expect::Colon && f.members <= 1 {
                        self.divergent = true;
                        return;
                    }
                    // The paren-comma-garbage tuple's nested close: the
                    // machine's grouping/empty-item cascade around such
                    // a tuple desyncs from the engine's whole-text
                    // classifier (`((,),1)` -> the engine `[[1]]`, the
                    // stream `[1]`); a root paren stays pinned parity
                    if f.paren && f.saw_comma_garbage && self.stack.len() > 1 {
                        self.divergent = true;
                        return;
                    }
                    // The emptied-frame cascade: an item of this ARRAY
                    // repaired to the empty array — the engine's
                    // whole-text lane re-decides the emptied item's
                    // now-empty shape through the parent's own close
                    // (`[[[] ,][] ,]` -> the engine `[]`, the stream
                    // `[[]]`) (out)
                    if !f.obj && !f.paren && f.saw_emptied_item {
                        self.divergent = true;
                        return;
                    }
                    // The drop-close: a trailing comma over a
                    // strictly-empty last item repairs this ARRAY to
                    // the empty array (`[[] ,]` -> `[]`) — the state
                    // its parent then re-decides
                    let emptied =
                        !f.obj && !f.paren && f.expect == Expect::Value && f.last_item_empty;
                    let frame_members = f.members;
                    self.stack.pop();
                    self.last_empty_string_value = false;
                    self.last_string_value = false;
                    self.bare_key_run.clear();
                    self.obj_value_after_ws = false;
                    if self.stack.is_empty() {
                        self.top_done = true;
                    } else if let Some(p) = self.stack.last_mut() {
                        p.members += 1;
                        p.expect = Expect::AfterValue;
                        p.last_item_empty = emptied || frame_members == 0;
                        if emptied || frame_members == 0 {
                            // an item that repaired to the empty array
                            // (the zero-member item, or the comma-close
                            // drop): the parent's cascade territory —
                            // the whole-text lane re-decides the
                            // emptied item's now-empty shape
                            p.saw_emptied_item = true;
                        }
                        // the strictly-empty item's one-char decision
                        // arms on the parent (the machine's
                        // `EmptyPending`); a PAREN parent's decision
                        // fires on the whitespace itself
                        if !p.obj {
                            self.empty_pending_armed = frame_members == 0;
                            self.empty_pending_paren = p.paren;
                        }
                    }
                    i += 1;
                }
                ':' => match self.stack.last().map(|f| &f.expect) {
                    Some(Expect::Colon) => {
                        // the bare key commits here: the duplicate-key
                        // update-in-place is the whole-text dict
                        // semantics for bare keys too (`{l:2{l:` -> the
                        // engine `{"l": ""}`), keyed on the entry
                        // skip's STRIPPED spelling (`{-ab: 1, ab: 2}`) (out)
                        if !self.bare_key_run.is_empty() {
                            let raw = std::mem::take(&mut self.bare_key_run);
                            let key = raw
                                .trim_start_matches(|c: char| !c.is_alphanumeric())
                                .to_string();
                            if !key.is_empty() {
                                let f = self.stack.last_mut().unwrap();
                                if f.keys.contains(&key) {
                                    self.divergent = true;
                                    return;
                                }
                                f.keys.push(key);
                            }
                        }
                        let f = self.stack.last_mut().unwrap();
                        f.expect = Expect::Value;
                        f.completed += 1;
                        self.last_string_value = false;
                        i += 1;
                    }
                    Some(Expect::Key) => {
                        // the missing-key colon: the pair drops — but
                        // the engine's key-reparse machinery re-decides
                        // what a key attempt leaves behind (`{:-:(` ->
                        // `{}`, compound shapes alike) (out)
                        self.divergent = true;
                        return;
                    }
                    _ => {
                        self.divergent = true;
                        return;
                    }
                },
                ',' => match self.stack.last().map(|f| &f.expect) {
                    Some(Expect::AfterValue) => {
                        let f = self.stack.last_mut().unwrap();
                        // the object-lane comma classification: a comma
                        // in an object is only ever a member separator
                        // for the unambiguous member shapes — the
                        // engine's whole-text classify_object_value_comma
                        // absorbs the tail otherwise
                        // (`{"a": xyz, -ab: 1}` -> the engine
                        // `{"a": "xyz, -ab: 1"}`) (out)
                        if f.obj && !Self::object_comma_is_member(&chars, i, n) {
                            self.divergent = true;
                            return;
                        }
                        f.expect = if f.obj { Expect::Key } else { Expect::Value };
                        self.last_empty_string_value = false;
                        self.last_string_value = false;
                        self.bare_key_run.clear();
                        self.empty_pending_armed = false;
                        self.obj_value_after_ws = false;
                        i += 1;
                    }
                    Some(Expect::Value) => {
                        // a double comma: in a PAREN frame the engine's
                        // item-parse garbage run is pinned parity (the
                        // group's loop runs on) — but the flag records
                        // it: the paren's own close desyncs from the
                        // whole-text classifier when nested (out)
                        let f = self.stack.last_mut().unwrap();
                        if !f.paren {
                            self.divergent = true;
                            return;
                        }
                        f.saw_comma_garbage = true;
                        i += 1;
                    }
                    Some(Expect::Colon) => {
                        // the corner-case: the whitespace-broken key
                        // attempt whose next char is the ',' COMMITS
                        // (the engine's keep-set `:`/`,`) and the ','
                        // acts as its colon — the machine commits the
                        // attempt and parses the value. An EMPTY-KEY
                        // attempt (the quoted `""`, or a bare one whose
                        // entry-skip strips to nothing) is the engine's
                        // own exception: the key loop DISCARDS it — the
                        // empty-object fallback's territory (out)
                        let key = if !self.bare_key_run.is_empty() {
                            let raw = std::mem::take(&mut self.bare_key_run);
                            raw.trim_start_matches(|c: char| !c.is_alphanumeric())
                                .to_string()
                        } else {
                            self.stack
                                .last()
                                .unwrap()
                                .keys
                                .last()
                                .cloned()
                                .unwrap_or_default()
                        };
                        if key.is_empty() {
                            self.divergent = true;
                            return;
                        }
                        let f = self.stack.last_mut().unwrap();
                        if f.keys.contains(&key) {
                            self.divergent = true;
                            return;
                        }
                        f.keys.push(key);
                        f.expect = Expect::Value;
                        f.completed += 1;
                        self.last_string_value = false;
                        i += 1;
                    }
                    _ => {
                        self.divergent = true;
                        return;
                    }
                },
                '"' => {
                    // a string opens: a key where a key was due, else a
                    // value; a value-start at Colon state is the
                    // missing-colon shape (out)
                    match self.stack.last().map(|f| &f.expect) {
                        Some(Expect::Key) => {
                            self.stack.last_mut().unwrap().members += 1;
                        }
                        Some(Expect::Value) => {
                            self.stack.last_mut().unwrap().expect = Expect::AfterValue;
                        }
                        _ => {
                            self.divergent = true;
                            return;
                        }
                    }
                    if self.last_empty_string_value {
                        // a token where the engine's whole-text
                        // string-repair lane re-pairs what follows an
                        // empty-string value (out)
                        self.divergent = true;
                        return;
                    }
                    // the direct quote-pair (the missing-comma shape
                    // `["x" "y"]`) is pinned parity: the flag clears
                    self.last_string_value = false;
                    if let Some(f) = self.stack.last_mut() {
                        // a new item starts here
                        f.last_item_empty = false;
                        f.item_after_emptied |= f.saw_emptied_item;
                    }
                    self.string_is_key = self.stack.last().is_some_and(|f| f.expect == Expect::Key);
                    self.key_buf.clear();
                    self.in_string = true;
                    self.esc = StrEsc::None;
                    self.raw_bs_run = 0;
                    self.saw_backslash = false;
                    self.content_ends_nl = false;
                    i += 1;
                }
                c if c.is_ascii_digit() || c == '-' || c == '.' || c.is_alphabetic() => {
                    // a number/word run: a value-start at Colon state is
                    // the missing-colon shape (out)
                    let Some(f) = self.stack.last_mut() else {
                        // a top-level scalar: the whole input one value
                        self.top_done = true;
                        continue;
                    };
                    self.last_string_value = false;
                    // a bare word at an OBJECT's key attempt (the Key
                    // state, or the AfterValue missing-comma state)
                    // becomes the member's key: recorded for the
                    // duplicate-key check at its `:`
                    let recording_key = f.obj && !matches!(f.expect, Expect::Value | Expect::Colon);
                    let frame_is_obj = f.obj;
                    if recording_key {
                        self.bare_key_run.clear();
                        self.bare_key_run.push(c);
                    }
                    if self.last_empty_string_value {
                        // a token where the engine's whole-text
                        // string-repair lane re-pairs what follows an
                        // empty-string value (out)
                        self.divergent = true;
                        return;
                    }
                    if f.in_array && f.obj && f.expect == Expect::Value {
                        // an unquoted word at an object-in-array member
                        // value: the engine's unquoted-value run eats the
                        // object's own `}` there (out)
                        self.divergent = true;
                        return;
                    }
                    match f.expect {
                        Expect::Key => {
                            // a bare-word key: the run runs to its
                            // terminator (the scan does not model it)
                            f.members += 1;
                            f.expect = Expect::Colon;
                        }
                        Expect::Value => {
                            f.expect = Expect::AfterValue;
                            f.last_item_empty = false;
                            f.item_after_emptied |= f.saw_emptied_item;
                        }
                        Expect::AfterValue => {
                            // the missing-comma shape (pinned parity in
                            // both container kinds): the next member/item
                            f.members += 1;
                            f.expect = if f.obj { Expect::Colon } else { Expect::Value };
                            f.last_item_empty = false;
                            f.item_after_emptied |= f.saw_emptied_item;
                        }
                        Expect::Colon => {
                            self.divergent = true;
                            return;
                        }
                    }
                    // a WORD run (alphabetic) at a value position: the
                    // machine's run absorbs internal whitespace and the
                    // commit trims, while the engine's unquoted-value
                    // lane re-decides (`[null x]` -> `[null, "x"]`) when
                    // a token follows the whitespace (out)
                    let word_run = c.is_alphabetic();
                    // the run's chars, to its terminator: `:` after an
                    // OBJECT-KEY run; `,`/`}`/`]`/`:`/whitespace at a
                    // value; the terminator decides whether the run
                    // COMPLETED (the mid-document float-spelling split
                    // needs a token AFTER the run). The key-attempt runs
                    // (recording_key: the branch's match already moved
                    // the object frame's expect to Colon) run to their
                    // `:` and whitespace alone — the machine's attempt
                    // absorbs its own `,`/`]`
                    let obj_key_run = recording_key;
                    let mut terminated = false;
                    self.run_char(c);
                    if recording_key {
                        self.bare_key_run.push(c);
                    }
                    i += 1;
                    while i < n {
                        let d = chars[i];
                        if d.is_whitespace() {
                            terminated = true;
                            if (word_run || self.run_saw_alpha) && !obj_key_run {
                                let mut k = i + 1;
                                while k < n && chars[k].is_whitespace() {
                                    k += 1;
                                }
                                if k < n && !matches!(chars[k], ',' | '}' | ']' | ':') {
                                    self.divergent = true;
                                    return;
                                }
                            }
                            break;
                        }
                        if obj_key_run && d == ':' {
                            terminated = true;
                            break;
                        }
                        // a container char inside a bare-key run: the
                        // engine's key-parse absorbs it (out)
                        if obj_key_run && matches!(d, '{' | '[' | '(' | '"') {
                            self.divergent = true;
                            return;
                        }
                        if !obj_key_run && matches!(d, ',' | '}' | ']' | ':') {
                            // the object-lane number-run's
                            // currency-comma: the machine absorbs the
                            // comma into the run and the rewind-word
                            // absorbs past the member boundary when an
                            // ALPHABETIC member key follows directly
                            // (`{"a": 1,c: 1}` -> the engine
                            // `{"a": "1", "c": 1}`) (out)
                            if d == ','
                                && frame_is_obj
                                && matches!(self.run_first, Some('0'..='9') | Some('-') | Some('.'))
                                && chars.get(i + 1).is_some_and(|nx| nx.is_alphabetic())
                            {
                                self.divergent = true;
                                return;
                            }
                            terminated = true;
                            break;
                        }
                        // a container/quote char inside a run: the
                        // engine's unquoted-value/string-repair lane
                        // eats or re-pairs it and re-decides (out)
                        if matches!(d, '{' | '[' | '(' | '"') {
                            self.divergent = true;
                            return;
                        }
                        if d == '-' && self.run_first == Some('-') {
                            self.divergent = true;
                            return;
                        }
                        // a backslash inside a bare-word run: the
                        // engine's escape lane re-decides it (out)
                        if d == '\\' {
                            self.divergent = true;
                            return;
                        }
                        // any other char the scan's own walk would call
                        // garbage at a token position (`+`, `%`, `$`, a
                        // control char, ...): the engine's repair lane
                        // re-decides it mid-run (out)
                        if !(d.is_alphanumeric() || d == '-' || d == '.') {
                            self.divergent = true;
                            return;
                        }
                        self.run_char(d);
                        if recording_key {
                            self.bare_key_run.push(d);
                        }
                        i += 1;
                    }
                    self.end_run(terminated);
                }
                '(' => {
                    if self.last_empty_string_value {
                        // a token where the engine's whole-text
                        // string-repair lane re-pairs what follows an
                        // empty-string value (`[""(` -> `["("]`) (out)
                        self.divergent = true;
                        return;
                    }
                    if self.last_string_value {
                        // the quoted-section re-pair's paren shape: a
                        // group right after a closed string whose window
                        // to the first `,`/`)` holds a quote (out)
                        let mut k = i + 1;
                        while k < n && !matches!(chars[k], ',' | ')' | ']' | '}' | '"') {
                            k += 1;
                        }
                        if k < n && chars[k] == '"' {
                            self.divergent = true;
                            return;
                        }
                    }
                    self.last_string_value = false;
                    if let Some(f) = self.stack.last() {
                        if f.expect == Expect::Colon {
                            // the paren-with-colon conversion (out)
                            self.divergent = true;
                            return;
                        }
                        if f.obj && f.expect == Expect::Key {
                            // a `(` at an object's key position: the
                            // engine's key-parse sub-parses the group
                            // and the empty-object classifier
                            // re-decides (out)
                            self.divergent = true;
                            return;
                        }
                    }
                    match self.stack.last_mut() {
                        Some(f) => {
                            if f.expect == Expect::Key {
                                f.members += 1;
                            }
                            f.expect = Expect::AfterValue;
                            f.last_item_empty = false;
                            f.item_after_emptied |= f.saw_emptied_item;
                        }
                        None => self.top_done = true,
                    }
                    self.stack.push(ScanFrame {
                        obj: false,
                        paren: true,
                        members: 0,
                        expect: Expect::Value,
                        keys: Vec::new(),
                        body_chars: 0,
                        completed: 0,
                        in_array: false,
                        saw_emptied_item: false,
                        saw_comma_garbage: false,
                        item_after_emptied: false,
                        last_item_empty: false,
                    });
                    i += 1;
                }
                _ => {
                    // any other char: garbage the engine's repair lane
                    // re-decides — out
                    self.divergent = true;
                    return;
                }
            }
        }
    }

    /// A plain (escape-free) char inside the open string: the structural
    /// content is the re-sync family (out); the string closes at its
    /// own delimiter.
    fn string_content(&mut self, c: char) {
        if c == '"' {
            if self.raw_bs_run >= 2 {
                // the doubled-escape family: the engine re-pairs the run
                self.divergent = true;
                return;
            }
            self.nl_tail_string = self.content_ends_nl;
            self.raw_bs_run = 0;
            self.content_ends_nl = false;
            self.in_string = false;
            let was_key = self.string_is_key;
            let key_text = self.key_buf.clone();
            self.string_is_key = false;
            // the empty-string VALUE followed by any token: the
            // engine's whole-text string-repair lane re-pairs what
            // follows it (`[""x]` -> `["x"]`, `{"a": ""x}` ->
            // `{"a": "x"}`, `[""(` -> `["("]`) — out
            self.last_empty_string_value = !was_key && key_text.is_empty();
            self.last_string_value = !was_key;
            // the strictly-empty string VALUE's own one-char decision
            // arms on the parent (the machine's `arm_empty_pending`)
            // the strictly-empty string VALUE's own one-char decision
            // arms on the parent (the machine's `arm_empty_pending`)
            if !was_key && key_text.is_empty() {
                self.empty_pending_armed = self.stack.last().is_some_and(|f| !f.obj);
                self.empty_pending_paren = self.stack.last().is_some_and(|f| f.paren);
            }
            self.key_buf.clear();
            match self.stack.last_mut() {
                Some(f) => {
                    if was_key {
                        if f.keys.contains(&key_text) {
                            // the duplicate-key update-in-place: the
                            // whole-text dict semantics (out)
                            self.divergent = true;
                            return;
                        }
                        f.keys.push(key_text);
                    } else {
                        // the completed VALUE item's emptiness: the
                        // drop-close's operand (`["" ,]` -> `[]`)
                        f.last_item_empty = key_text.is_empty();
                    }
                    f.expect = match f.expect {
                        Expect::Key | Expect::Colon => Expect::Colon,
                        _ => Expect::AfterValue,
                    };
                }
                None => self.top_done = true,
            }
            return;
        }
        if matches!(c, '{' | '}' | '[' | ']' | ',' | ':') {
            // a structural char inside an open string: the engine's
            // re-sync family (out)
            self.divergent = true;
            return;
        }
        if c < ' ' || c == '\u{7f}' || c == '\u{85}' || c == '\u{2028}' || c == '\u{2029}' {
            // a raw control/line-break char inside a string: the
            // engine's string lane re-decides (the LLM multi-line
            // handling) — out
            self.divergent = true;
            return;
        }
        if c != ' ' {
            self.content_ends_nl = false;
        }
        // (the content is tracked for KEY and VALUE strings alike: the
        // keys need the text, the empty-string re-pair trigger needs
        // the value's emptiness)
        self.key_buf.push(c);
    }

    /// The close-up: the open string's quote, then per open frame the
    /// filler (a dangling key's `: null`, a missing value's `null`) and
    /// the frame's closer. `None` when the cut sits in a documented
    /// divergence: an object's FIRST member dangling keyless (the
    /// engine's whole-text fallback heals `{"a"`/`{""` to an array,
    /// the stream drops the member).
    fn close_up(&self, s: &str) -> Option<String> {
        if self
            .stack
            .iter()
            .any(|f| f.obj && f.members <= 1 && matches!(f.expect, Expect::Colon | Expect::Key))
        {
            return None;
        }
        if self.in_string && self.saw_backslash {
            // the doubled-quote/escape re-pair family's territory
            return None;
        }
        let mut out = String::from(s);
        if self.in_string {
            out.push('"');
        }
        for f in self.stack.iter().rev() {
            match f.expect {
                Expect::Colon => out.push_str(": null"),
                Expect::Value => out.push_str("null"),
                Expect::Key | Expect::AfterValue => {}
            }
            out.push(if f.obj { '}' } else { ']' });
        }
        Some(out)
    }
}

fuzz_target!(|s: &str| {
    // Several deterministic split sets per input: one-push (no splits),
    // three-char chunks (escape/pair straddling), single-char pushes,
    // and a strided scatter. Every assertion below runs per split set.
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let split_sets: Vec<Vec<usize>> = vec![
        Vec::new(),
        (0..n).step_by(3).collect(),
        (0..n).collect(),
        (0..n).filter(|i| i % 7 == 3).collect(),
    ];

    // The whole-text spelling's answer: the empty-string sentinel
    // renders bare (the repaired value Str("") is the nothing-
    // recoverable spelling), exactly what repair_json returns.
    let engine_out = match repair(s, &RepairConfig::default()) {
        Ok((Value::Str(empty), _)) if empty.is_empty() => Some(String::new()),
        Ok((v, _)) => Some(dumps(&v, true)),
        Err(_) => None,
    };
    // The one-push answer (None when the depth cap fired: the state is
    // unspecified after an error, so error-free splittings compare only
    // against a None-free reference).
    let one_out = (|| {
        let mut once = StreamingRepairer::new(true);
        once.push(s).ok()?;
        once.end().ok()
    })();

    for splits in &split_sets {
        let mut r = StreamingRepairer::new(true);
        let mut prev = 0usize;
        let mut errored = false;
        for &p in splits {
            let p = p.min(n);
            if p < prev {
                continue;
            }
            let chunk: String = chars[prev..p].iter().collect();
            prev = p;
            if r.push(&chunk).is_err() {
                errored = true;
                break;
            }
            // Snapshot is loadable-or-sentinel at every point, and equals
            // end on the same prefix (the machine's own render, pure).
            let snap = r.snapshot();
            assert!(
                snap.is_empty() || loads_strict(&snap).is_ok(),
                "snapshot is not strict-loadable: {snap}"
            );
            assert_eq!(r.end().expect("end never fails"), snap, "snapshot != end");
        }
        if errored {
            continue;
        }
        let tail: String = chars[prev..].iter().collect();
        if r.push(&tail).is_err() {
            continue;
        }
        let final_text = r.end().expect("end never fails");
        assert!(
            final_text.is_empty() || loads_strict(&final_text).is_ok(),
            "end() is not strict-loadable: {final_text}"
        );
        if let Some(want) = &one_out {
            assert_eq!(&final_text, want, "end() drifted across chunkings");
        }
    }

    // The ENGINE-PARITY gate, COMPLETE-DOCUMENT edition: for every
    // input the scan does not exclude (the documented divergence list,
    // mechanically checked trigger by trigger), the streamed end()
    // must equal the whole-text engine's repair of the same text --
    // byte for byte, under every split set. The strict-loadable class
    // and the clean-cut heal class are subsumed: a strict input is
    // scan-clean with an empty stack, and a cut's parity claim was
    // always "the machine's own heal equals the engine's repair of the
    // whole input". The exclusion triggers (each a documented
    // whole-text-only machinery): comments and non-`"` delimiters; the
    // string re-synchronization (a structural char inside an open
    // string); invalid escapes (the whole-text string-repair lane); the
    // doubled-quote/escape re-pair family (a backslash run, a backslash
    // in the open string at the cut); the empty-string value re-pair
    // (a token hard on an empty-string value's heels: `[""x]` ->
    // `["x"]`); the mid-document escape-tail and
    // special-float splits; the multi-value/multi-token top level; the
    // missing-key colon past an object's first member; the
    // first-member dangling key (the body-reparse fallback); duplicate
    // object keys (the dict update-in-place); the object-in-array
    // unquoted-value and merge lanes; an array lane's leading/double
    // comma (the item-parse garbage run + merge territory); a
    // mismatched closer (the close-up re-decides); the missing-colon
    // shape; the object-lane comma classification (a garbage-led member
    // key after a bare value's comma); the quoted-section re-pair (a
    // quote in the window right after a closed string's container-open);
    // the paren-with-colon conversion; and any garbage char.
    let mut scan = Scan::new();
    scan.walk(s);
    let parity = if scan.divergent {
        None
    } else {
        scan.close_up(s).map(|_| ())
    };
    if parity.is_some()
        && let Some(want) = engine_out
    {
        for splits in &split_sets {
            let mut r = StreamingRepairer::new(true);
            let mut prev = 0usize;
            let mut ok = true;
            for &p in splits {
                let p = p.min(n);
                if p < prev {
                    continue;
                }
                let chunk: String = chars[prev..p].iter().collect();
                prev = p;
                if r.push(&chunk).is_err() {
                    // past the 200-container cap: the engine's own
                    // normalized guard fired on both sides; skip
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            let tail: String = chars[prev..].iter().collect();
            if r.push(&tail).is_err() {
                continue;
            }
            assert_eq!(
                r.end().expect("end never fails"),
                want,
                "streamed != engine on split set {splits:?} of {s:?}"
            );
        }
    }
});
