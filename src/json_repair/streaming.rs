//! The stateful streaming repairer: an incremental front end for the whole
//! text engine's truncation-heal semantics, built for LLM token streams
//! (`tors.JsonRepairer`).
//!
//! # WHY
//!
//! `repair` is a whole-text engine: it owns the entire input, looks back
//! and forward across it (the string-repair pairing walks, the splice
//! repairs, the strict-suffix probe), and re-serializes the parsed value.
//! A token stream cannot pay that shape per chunk: re-running any
//! whole-text pass per push is quadratic in the total bytes (the
//! suture/repair-json-stream publish measures the naive re-parse shape at
//! ~15x the wall when the chunk count quadruples; a linear stream must
//! hold ~4x). This module is the streaming answer: ONE left-to-right
//! state machine over the chars of the stream, holding only structural
//! state (the container stack, the string/escape state, one pending
//! token), emitting canonical JSON as it goes. Every chunk is processed
//! exactly once; nothing is ever re-parsed; the total work is linear in
//! the total bytes at any chunk size.
//!
//! # WHAT (the contract)
//!
//! - [`push`] feeds one chunk and returns the text THIS call newly
//!   emitted (the repaired delta): canonical JSON text, with the stream's
//!   open state (an unterminated string, unclosed containers) still open.
//!   Concatenating the deltas reconstructs the emitted stream, with one
//!   documented exception: a late repair that retracts text (the
//!   dangling-member drop, the grouping-paren unwrap, the empty-element
//!   retract) truncates `out`
//!   behind the delta cursor, so the final authority is `snapshot`/`end`,
//!   never the concatenation.
//! - [`snapshot`] renders the current state closed: the pending token
//!   resolved, the open string quoted shut, the open containers closed.
//!   The result is valid JSON (loadable) at every point of the stream,
//!   and it is exactly what `end` would return if the stream stopped
//!   here. Non-destructive.
//! - [`end`] finalizes: resolves the pending token, closes the open
//!   string and every open container (the truncated-output heal), and
//!   marks the repairer finished. A second `end` returns the same text;
//!   `push` after `end` is an error; [`reset`] restores a fresh state.
//!
//! # The repair semantics (what the machine repairs, per character)
//!
//! Everything here mirrors the whole-text engine's observable behavior on
//! the class of inputs a stream can decide incrementally; the decisions
//! were pinned against the engine (and through it the oracle,
//! json-repair 0.63.5) case by case:
//!
//! - **Canonical emission**: the output is the `json.dumps`-parity
//!   serialization (the same `", "`/`": "` separators, the same escape
//!   table, `ensure_ascii` on by default), so a completed stream's
//!   `end()` output is byte-identical to `repair_json` on the same text
//!   whenever the machine's parse agrees with the engine's. Strings are
//!   decoded (`\n`, `\uXXXX`, surrogate pairs) and re-encoded through the
//!   serializer's own per-char table, so incremental emission cannot
//!   drift from `dumps`.
//! - **Delimiter normalization**: single and curly quotes open and close
//!   strings exactly as the engine's repair lane does; the emitted
//!   delimiters are always `"`.
//! - **Missing separators**: a member or item that starts where `,` was
//!   due gets the comma (deferred: the separator is emitted when the next
//!   member starts, which makes both missing commas and trailing commas
//!   correct with no rewriting: `{"a": 1 "b": 2}` and `[1 2]` heal
//!   mid-stream, `{"a": 1,}` and `[1, 2,]` never emit the stray comma).
//! - **Bare words**: an unquoted run (keys and values) is held until its
//!   terminator, then emitted as a string, or as the literal when it is
//!   the CASE-INSENSITIVE `true`/`false`/`null`/`none` family (the
//!   repair lane's keyword family: `TRUE`/`tRuE`/`NONE` repair to
//!   `true`/`true`/`null`) or the strict grammar's special floats at
//!   their exact spellings `NaN`/`Infinity`/`-Infinity` (loads-valid, so
//!   the whole-text observable keeps the floats; `nan`/`INFINITY` are
//!   strings at their own casing). A word born from a number run drops
//!   its stray leading sign (`{"a": -NaN}` -> `{"a": "NaN"}`,
//!   `{"a": -abc}` -> `{"a": "abc"}`). Other words repair to strings
//!   (`undefined`, `NaN2`).
//! - **Numbers**: the engine's `parse_number` rules, mirrored: the
//!   `NUMBER_CHARS` run (`,` joins the run inside objects, breaks it
//!   inside arrays; `_` is consumed and dropped), the single trailing
//!   `- e E / , +` rollback, then the same value lanes (int, unbounded
//!   int, float, currency-comma string).
//! - **Comments**: `# ...`, `// ...`, `/* ... */` are skipped with the
//!   engine's terminator rules (a `#` comment also stops at `]`/`}`/`:`
//!   per the context it fired in; the terminator stays in place).
//! - **Parenthesized tuples**: `(` at a value position opens an array
//!   that closes as a tuple at `)` when it held a comma or more than one
//!   element; a single-element comma-less `(...)` is a grouping and
//!   unwraps to its element (`(1)` repairs to `1`, `(1,)` to `[1]`, the
//!   engine's own split). At a clean top level the `(` is gated on the
//!   engine's own conservative value-detection (`(,` was inline prose:
//!   the first inner char and the post-group tail decide — `(1)` is a
//!   value, `(,)`/`(x)`/`(1) x` are not; `(true)`'s literal prefix
//!   passes). The group's loop runs under `)` as ITS closing
//!   delimiter (the engine's parse_array lane): a `]` inside an open
//!   group is a garbage item the strictly-empty skip consumes — nothing
//!   closes, the group keeps parsing (`[(], 1, 2]` -> `[[1, 2]]`, the
//!   `]` swallowed) — and a `}` terminates the group and is consumed by
//!   it (`[(}]` -> `[[]]`). An empty group closed under its own `)`
//!   defers the item skip to the NEXT char (`,` or the parent's closer
//!   keeps the element, anything else drops it and is eaten with it:
//!   `[()]` -> `[[]]` but `[() ]` -> `[]`); a group the `]` left open
//!   retracts at close time like any empty container (`[(]` -> `[]`).
//! - **Mismatched closers**: a closer matching a frame below the
//!   innermost one closes the levels between (the engine's
//!   `{"a": [1, 2}` -> `{"a": [1, 2]}` shape); a closer matching nothing
//!   is skipped as garbage.
//! - **The truncated-output heal** (`end`/`snapshot` on a cut-off
//!   stream): an open string closes with `"` (its content rstripped —
//!   Python's whitespace set, the engine's own escape-tail heal:
//!   `{"k": "a\n ` -> `{"k": "a"}`), a CLOSED string that ends the
//!   stream on a newline-run loses the run (`{"k": "a\n"` ->
//!   `{"k": "a"}`), open containers close with their own brackets — a
//!   still-open EMPTY container drops whole at an item position (`[[`
//!   -> `[]`, `[1, [` -> `[1]`) and closes at a member-value position
//!   or the root (`{"a": [` -> `{"a": []}`), an array's trailing
//!   strictly-empty member drops with it (`[[], []` -> `[[]]`,
//!   `[1, []` -> `[1]`), and the engine's two array-lane item drops
//!   reproduce: the stray `...` (an item whose parse is the exact
//!   string `"..."` with the parse ending on a `.` — the cut's open
//!   string and a bare number-run alike: `["...` and `[1, ...` to
//!   `[]`/`[1]`, while the CLOSED `["..."]` element's parse ends on
//!   its quote and stays) and the strictly-empty item whose next char
//!   is not a separator (`[[] ,` -> `[]`, `[[] , 1` -> `[1]`; the
//!   decision defers to close time, and its gate is the ENGINE'S OWN
//!   DECISION PROCEDURE: the drops happen exactly when the completed
//!   text is NOT strict-valid JSON -- `json.loads` failed so the
//!   engine's repair lane decided and its strictly-empty skips fire;
//!   when the machine's own state says the parse was strict-clean
//!   end-to-end AND the root closed cleanly, the whole-input fast path
//!   kept every element and the marks drain nowhere -- the cut-only
//!   justification "whenever the text ends up valid, the drops are
//!   cut-only" was WRONG: the repair lane also drops in
//!   cleanly-closed-yet-INVALID roots (`[[] ,]` -> `[]`,
//!   `[1, (,)]` -> `[1]`) -- and the same gate re-decides the
//!   garbage-item run (a `,` at an item position consumes the
//!   non-value-start chars that follow, the frame's own closer
//!   included: `[(,)]` -> `[]`, `[(,), 1]` -> `[[1]]`) and the
//!   empty-object array fallback (an object that closed with no
//!   members over a non-trivial colon-free body closes as `[]`:
//!   `{"a": {(]` -> `{"a": []}`, `{]}` -> `[]`)) — the pending
//!   number/word resolves by the same
//!   rules that terminate it mid-stream (an element that renders empty
//!   retracts whole: `[1, -` -> `[1]`), through the literal table (the
//!   special floats' spellings heal to STRINGS: `[NaN` -> `["NaN"]`,
//!   the strict class keeps the float in the complete document), and
//!   the rewind-born word takes the engine's string-lane ENTRY SKIP:
//!   the run's leading non-alphanumeric garbage (a dot run, sign and
//!   dot, an underscore) drops before the literal check (`[.t]` ->
//!   `["t"]`, `[.true]` -> `[true]`, `[.5x]` -> `["5x"]`,
//!   `[-.t]` -> `["t"]`; the strict `-Infinity` spelling keeps its
//!   float — the fast path may still win the complete document), a
//!   missing value after `:`
//!   heals to `""`, a missing KEY drops the whole pair (`{: 1}` ->
//!   `{}`), and a trailing separator disappears. `{"a": "hel` heals to
//!   `{"a": "hel"}`, `{"a": [1, 2` to `{"a": [1, 2]}`, `{"a":` to
//!   `{"a": ""}`.
//! - **The object-key lane's string semantics**: an unquoted key
//!   attempt is the engine's STRING lane, not a number lane: the
//!   entry skip's leading non-alphanumeric garbage drops at the
//!   commit (`{-ab: 1}` -> `{"ab": 1}`, `{.5: 1}` -> `{"5": 1}`,
//!   `{-eOe: 6}` -> `{"eOe": 6}`), and a whitespace holds the attempt
//!   for the next char -- the attempt commits at its `:`, else it
//!   invalidates and the object's key loop retries from the next word
//!   (`{a b: 1}` -> `{"b": 1}`, `{-ab x: 1}` -> `{"x": 1}`); an
//!   attempt that strips to NOTHING drops the pair whole
//!   (`{-: 1` -> `{}`).
//! - **Partial literals** (the decision the surface is asked about
//!   explicitly): there is NO `tru` -> `true` healing. The engine's own
//!   partial semantics decide, and they are unambiguous:
//!   `repair_json("tru")` is `""` (a top-level word is prose unless the
//!   whole input is one strict JSON value, which `tru` is not) and
//!   `repair_json('{"a": tru')` is `{"a": "tru"}` (a bare word in a
//!   container is a string). The streaming machine reproduces exactly
//!   that: a top-level word/number is held as a provisional scalar and
//!   committed at `end` only when the text held so far is one strict JSON
//!   value (`loads`-parity; a top-level string commits only from the
//!   strict `"` delimiter — `'hi'` was prose to the engine, `''`), else
//!   discarded as prose; an in-container word heals to its string.
//!   Deciding otherwise would break the end()-equals-engine differential
//!   this surface is tested with.
//! - **Top-level prose**: input before the first container is skipped
//!   (`Answer: {"a": 1}` -> `{"a": 1}`), as are trailing top-level tokens
//!   after the root value closed (`{"a": 1} junk` -> `{"a": 1}`). A
//!   leading `(` opens the tuple container only from a clean top level
//!   (the engine's own conservative tuple detection: inline prose like
//!   `note (clarification):` must not hijack).
//!
//! # Scope (documented divergences from the whole-text engine)
//!
//! The engine's deep repair machinery re-decides earlier text with
//! lookahead and splices; a linear stream cannot replay it. Where the two
//! differ, the streaming output stays valid JSON (or the nothing-
//! recoverable empty string), and the differential suite pins the classes
//! where equality holds:
//!
//! - **String re-synchronization**: the engine terminates a damaged
//!   string at a structural closer (`{"a": "hello}` -> `{"a": "hello"}`);
//!   the stream closes a string only at its own delimiter or at `end`
//!   (`{"a": "hello}` streams to `{"a": "hello}"}`). Doubled-quote
//!   repair (`""hi""` -> `"hi"`) is likewise whole-text-only: the stream
//!   keeps the parts it saw, validly.
//! - **In-container comments**: the engine's comment arm consumes the
//!   value that follows it inside a container (`{"a": /*x*/ 1}` ->
//!   `{"a": ""}`); the stream skips the comment and parses the value
//!   (`{"a": 1}`). Top-level comments match the engine exactly.
//! - **Dangling object keys**: a key with no value heals to the pair's
//!   drop (the engine's continuation behavior: `{"a": "x", "b"` ->
//!   `{"a": "x"}`); the engine's first-member shape instead falls back to
//!   an array (`{"a"` -> `["a"]`), which the stream does not reproduce
//!   (`{"a"` -> `{}`).
//! - **Value-then-value in an object**: the engine drops the stray value
//!   (`{"a": 1 2}` -> `{"a": 1}`); the stream skips it too when it cannot
//!   be a key (digits), and keeps string/word continuations as keys (the
//!   engine's `{"a": 1 "b": 2}` -> `{"a": 1, "b": 2}` shape, which the
//!   stream matches).
//! - **The word-swallow**: a bare word born at an array position runs
//!   past a mismatched closer in the engine's reparse (`[1e}` -> the
//!   engine `[1, "e}"]`, the stream `[1, "e"]`; `[1, tru}` ->
//!   `[1, "tru}"]` vs `[1, "tru"]`); the compound missing-key shape's
//!   key no longer diverges (the key lane's string semantics commit
//!   `{: 1, : 2}` -> `{"1,": 2}` the engine's own way); the outputs
//!   stay valid JSON on both sides.
//! - **The unquoted-value run's container/whitespace absorption**: the
//!   engine's unquoted-value lane eats container chars and internal
//!   whitespace into the string and re-decides literal prefixes
//!   (`[{"b": undefined},]` -> the engine `[{"b": "undefined},"}]`,
//!   the stream `[{"b": "undefined"}]`; `[null x]` -> the engine
//!   `[null, "x"]`, the stream `"null x"`); the stream's word stops at
//!   the structural chars, validly.
//! - **The empty-object fallback's body reparse**: an object that
//!   closed empty over a non-trivial colon-free body re-parses the
//!   body as an array — the stream reproduces the fallback only when
//!   the body recovers nothing (`{,}`-shaped bodies: `{]}` -> `[]`,
//!   `{"a": {(]` -> `{"a": []}`) and keeps `{}` where the body would
//!   re-parse to members (`{ x}` -> the engine `["x}"]`, the stream
//!   `{}`); duplicate object keys (the dict update-in-place,
//!   `{"a": 1, "a": 2}` -> the engine `{"a": 2}`) are likewise
//!   whole-text-only: the stream keeps both members.
//! - **The strict-vs-repair split on trailing tails**: the engine's
//!   REPAIR lane decides by the WHOLE input — the complete document
//!   `{"a": "x\n", "b": 1}` keeps the string's newline and `[NaN, 1]`
//!   keeps the float, but the cut `{"a": "x\n", "b": 1` strips the
//!   newline and `[NaN, 1` quotes the float. The stream keeps the
//!   strict spelling mid-document and matches only the tail-of-stream
//!   cases (the string last before the cut; the word still in flight,
//!   healed through the literal table).
//! - **The paren-with-colon conversion**: the engine turns a colon
//!   inside the parenthesized container into an object (`("a": ` ->
//!   `{"a": ""}`); the stream keeps the tuple machinery (`"a"`), valid
//!   JSON on both sides.
//! - **The mismatched-closer garbage**: the engine's whole-text
//!   close-up re-decides earlier structure on garbage
//!   (`{{\r]0\x01...('` -> `[]`, the stream `{"0": []}`), both valid.
//! - **Multiple top-level values**: the engine's whole-text loop may
//!   merge or array-wrap them (`{"a":1}{"b":2}` -> `[{"a": 1}, {"b": 2}]`);
//!   the stream finalizes the first and drops the rest. Compose with
//!   `repair_json` when the multi-value semantics matter.
//! - **The fence pre-pass**: not streamed (the fence decision needs the
//!   whole input); feed unwrapped text, or compose with
//!   `tors.extract_code_blocks`.
//! - **Lone `\uD800`-range escapes**: decode to U+FFFD per the engine's
//!   documented divergence from the oracle (a valid surrogate PAIR still
//!   decodes exactly, including when the two escapes straddle a chunk
//!   boundary; the held high surrogate is part of the machine's state).//!
//! # Cost
//!
//! `push` is O(chunk) plus O(delta) for its return: one pass, no
//! allocation proportional to the stream so far (the pending token and
//! the escape state are bounded; `out` grows once per emitted byte). The
//! linearity pin (`tests/test_json_repair_streaming.py`) holds the chunk
//! SIZE fixed and quadruples the total, where the per-chunk re-parse
//! anti-pattern measures ~16x and this machine ~4x.

use super::dumps::push_escaped_char;
use super::strict::loads_strict;
use super::{MAX_NESTING, STRING_DELIMITERS, Value, dumps as serializer, normalize_big_int_text};
use crate::normalize_impl::is_py_whitespace;

/// The close-time cascade's LOCAL copy of one frame's state (the
/// cascade retracts renderings and walks the stack without touching
/// the machine: `render_closed` is pure, `snapshot` purity).
struct FrameInfo {
    kind: FrameKind,
    brace_pos: usize,
    body_chars: usize,
    body_nonspace: usize,
    member_begun: bool,
    body_colon: bool,
    member_count: usize,
    member_start: Option<usize>,
    paren: Option<ParenFrame>,
}

/// The streaming repairer: see the module docs for the contract, the
/// semantics, and the divergences.
pub struct StreamingRepairer {
    /// The emitted canonical text so far, in its open form: an open
    /// string's quote is out with its content streamed, open containers
    /// are unclosed, the pending token is not yet in here.
    out: String,
    /// The container stack (the engine's context stack, structural only).
    frames: Vec<Frame>,
    /// Where the machine stands inside the innermost frame.
    pos: Pos,
    /// The open string, when one is (its delimiter, escape state, and the
    /// held high surrogate of a split `\uD800`-`\uDC00` pair).
    string: Option<StrState>,
    /// The one held-back token (a number run, a bare word, a provisional
    /// top-level scalar): bounded by the token's own length, never by the
    /// stream's.
    pending: Option<Pending>,
    /// The comment skipper, when inside one.
    comment: Option<CommentState>,
    /// Nothing has been discarded at the top level yet: the precondition
    /// for a provisional top-level scalar (the whole input could still be
    /// one strict JSON value).
    top_clean: bool,
    /// The root value closed: everything after it is trailing junk.
    top_done: bool,
    /// The `json.dumps` escape-table mode (the `repair_json` knob).
    ensure_ascii: bool,
    /// `end` ran: the repairer is final (push refuses; reset revives).
    finished: bool,
    /// The `out` length returned by the last `push`: the delta cursor.
    emitted: usize,
    /// The `out` position of the most recent string rendering's opening
    /// quote (set at `open_string`; a closed string's value stays): the
    /// escape-tail strip's lower bound at close time.
    last_string_start: Option<usize>,
    /// The last RAW char consumed while a string was open (reset at
    /// `open_string`): the cut-time stand-in for the engine's `get(-1)`
    /// cursor look at the same position (the engine reads the raw input,
    /// not the healed accumulator).
    string_raw_tail: Option<char>,
    /// The immediately-preceding char's `str.isspace()` (the machine's
    /// one-char look-back): the engine's strictly-empty item skip reads
    /// the char at the same spot.
    prev_char_ws: bool,
    /// The deferred strictly-empty element drops (the engine's array
    /// skip): a `(rendering start, rendering end, the element was the
    /// frame's first, the frame's index)` span per strictly-empty
    /// element whose next char was whitespace. The engine decides at
    /// the element's own return (the next char, whitespace included,
    /// drops it); the machine defers to close time because the
    /// WHOLE-INPUT `json.loads` fast path keeps the element whenever
    /// the text ends up one strict JSON value. Drained by
    /// `render_closed` (pure, local) exactly when the engine's own
    /// decision procedure ran the repair lane: the completed text is
    /// NOT strict-valid JSON (the root did not close cleanly, or the
    /// machine's `strict_clean` flag fell -- any repair/garbage/
    /// malformed marker), so the fast path never applied.
    empty_marks: Vec<EmptyMark>,
    /// The strictly-empty element's deferred one-char decision; see
    /// `EmptyPending`. Armed when a strictly-empty element (an empty
    /// container's pop, a closed `""` string) completes inside an
    /// array/tuple frame, resolved by the next char, never carried
    /// across a push boundary in any observable way (the state is part
    /// of the machine, so chunk splits do not change the outcome).
    empty_pending: Option<EmptyPending>,
    /// The machine's own strict-clean flag: has the input so far been
    /// one strict JSON value in progress -- no repair lane fired, no
    /// garbage consumed, no malformed structure seen? Every departure
    /// from the strict grammar flips it false (the engine's
    /// `json.loads` fast path would fail on the whole input, so its
    /// repair lane decides: the strictly-empty drops and the other
    /// repair-lane observables apply). The engine's own decision
    /// procedure, tracked as the machine sees it; see `render_closed`'s
    /// drain gate.
    strict_clean: bool,
    /// The last closed KEY string's content was empty (the engine's
    /// key loop discards an empty-key attempt at anything but its own
    /// `:` — the `{"" ,}`-corner; `{"": 1}`'s `:` keeps it).
    last_key_empty: bool,
    /// The engine's garbage-item run (parse_array_items' item parse
    /// started at a `,`): parse_json's ignore-and-move-on loop consumed
    /// the `,` and every following non-value-start char -- the frame's
    /// own closer included -- until a value starts, a `}` terminates
    /// the loop (consumed by it), or the input ends. While set, every
    /// char is consumed silently on those terms.
    garbage_run: bool,
    /// The top-level `(`'s conservative tuple gate (the engine's
    /// `top_level_parenthesized_can_start_value`): the `(` was seen at
    /// a clean top level but the machine has not committed to the
    /// tuple container yet -- the first inner char decides, and after
    /// the group closes only whitespace may precede the next line end.
    paren_gate: Option<ParenGate>,
}

/// One deferred strictly-empty element drop; see `empty_marks`.
struct EmptyMark {
    start: usize,
    end: usize,
    was_first: bool,
    frame: usize,
}

/// The strictly-empty element's one-char decision: the element's
/// rendering (from the parent frame's `member_start` to the element's
/// end) is exactly `[]`/`{}`/`""`, its parent is an array-lane frame
/// holding it as an in-flight member, and the engine's item skip now
/// reads the NEXT char: a `,` or the parent's closer keeps the
/// element, whitespace converts the decision to a deferred
/// [`EmptyMark`] (the fast path may still keep the element -- only the
/// close-time strict-clean gate decides), any other char drops it and
/// is consumed with it (the engine's one-char stall guard: the skip's
/// `index += 1` eats exactly one char, whatever it is).
/// `parent_paren` distinguishes the paren lane's `)` closer.
struct EmptyPending {
    start: usize,
    end: usize,
    was_first: bool,
    frame: usize,
    parent_paren: bool,
}

/// The top-level `(`'s conservative tuple gate, the engine's
/// `top_level_parenthesized_can_start_value` split into the two
/// decisions a stream can make incrementally:
///
/// - `FirstChar`: the `(` was seen at a clean top level; the machine
///   holds the container open (nothing emitted). The next non-ws char
///   decides: a value-start-ish char (`)`, `{`, `[`, `(`, a quote, a
///   digit, `-`, `.`) or a case-insensitive `true`/`null`/`none`/
///   `false` literal prefix opens the tuple container; anything else
///   was inline prose -- the `(` is dropped and the chars replay at
///   the top level.
/// - `TailCheck { }: the group closed at the top level; the engine's
///   gate also requires only whitespace before the next line end (or
///   the end of input) after the group's `)`. A token arrives:
///   the group's rendering retracts whole and the char replays as
///   prose (the engine never took the `(` as a value).
enum ParenGate {
    FirstChar { buf: String },
    TailCheck,
}

/// One open container: the engine's context entry, structural part.
struct Frame {
    kind: FrameKind,
    /// The open bracket's `out` position (the `{`/`[`/the paren's `[`):
    /// the empty-object fallback's drain coordinate.
    brace_pos: usize,
    /// Object only: the total chars seen while the frame has been open
    /// (whitespace included, the closing char included -- the engine's
    /// parse-span length), and the same count over non-whitespace
    /// chars only (the classifier's trimmed body). The engine's
    /// empty-object fallback fires when the span is non-trivial
    /// (`index - start_index > 2`) AND the trimmed body is non-empty.
    body_chars: usize,
    body_nonspace: usize,
    /// A member ever began in this frame (a `begin_common` ran): the
    /// engine's empty-object fallback is about objects that never got
    /// one.
    member_begun: bool,
    /// The body saw a `:` where a key was due (the missing-key drop's
    /// shape): the engine's classifier keeps the object (`{: 1}` ->
    /// `{}`) -- the colon is object-style evidence.
    body_colon: bool,
    /// Completed members/items so far: the next one emits `", "` first.
    member_count: usize,
    /// Object only: where the in-flight member's rendering began, BEFORE
    /// its deferred separator (the dangling-member drop truncates here).
    /// `None` when no member is in flight.
    member_start: Option<usize>,
    /// Paren only: the grouping-vs-tuple bookkeeping.
    paren: Option<ParenFrame>,
}

#[derive(Clone, Copy, PartialEq)]
enum FrameKind {
    Obj,
    Arr,
    Paren,
}

impl FrameKind {
    fn matches(self, closer: char) -> bool {
        matches!(
            (self, closer),
            (FrameKind::Obj, '}') | (FrameKind::Arr, ']') | (FrameKind::Paren, ')')
        )
    }
}

/// The `(`-container's tuple decision state: `elems` counts values that
/// started inside, `comma_seen` records a separator; at `)` (or at end),
/// exactly one element and no comma means a grouping (unwrap), anything
/// else means a tuple (close as an array) — the engine's own split,
/// `(1)` -> `1` vs `(1,)`/`(1, 2)`/`()` -> `[..]`.
#[derive(Clone, Copy)]
struct ParenFrame {
    brace_pos: usize,
    elems: usize,
    comma_seen: bool,
    /// The group's single element was itself a grouping's render (a
    /// nested paren's own `[]`, or a scalar the nested group
    /// unwrapped): the engine's re-dispatch of the inner text
    /// recovers nothing once a token follows (`(())x`/`((1))x` -> ``).
    elem_groupish: bool,
}

/// Where the machine stands inside the innermost frame.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pos {
    /// Expecting an object member's key (after `{` or after `,`).
    Key,
    /// A key completed, expecting `:`.
    Colon,
    /// Expecting a value that starts a member/item (after `[`, after
    /// `,` in an array or tuple, after `(`): the deferred separator
    /// emits when it starts.
    Value,
    /// Expecting the value of an already-counted member (after `:`):
    /// its separator went out with the key.
    MemberValue,
    /// A value completed, expecting `,` or the frame's closer.
    AfterValue,
}

/// The open string's scan state. `feed` decodes escapes and emits the
/// canonical form of each content char through the serializer's table.
struct StrState {
    delim: char,
    /// Key vs value string: only the post-close transition differs.
    is_key: bool,
    esc: Esc,
    /// A decoded high surrogate awaiting its low half (the pair may
    /// straddle chunks; the hold is part of the machine's state).
    high: Option<u16>,
}

/// The escape sub-state of an open string.
enum Esc {
    None,
    /// A backslash was consumed; the next char decides.
    Backslash,
    /// A `\u` was consumed; 0..=4 hex chars collected so far. The escape
    /// is literal text unless exactly four hex digits arrive.
    U(String),
}

impl StrState {
    /// Feed one content char; append its canonical form to `sink`.
    /// Returns true when the string closed (the delimiter, un-emitted:
    /// the caller emits the canonical closing quote). Any repair-lane
    /// content (an invalid escape, a literal-text `\u` run, a lone
    /// surrogate, a raw control char) flips `strict` false: the whole
    /// input was not one strict JSON value.
    fn feed(&mut self, c: char, sink: &mut String, ensure_ascii: bool, strict: &mut bool) -> bool {
        match std::mem::replace(&mut self.esc, Esc::None) {
            Esc::Backslash => {
                match c {
                    'u' => {
                        // The escape continues; a held high surrogate
                        // stays held (this `\u` may be its low half).
                        self.esc = Esc::U(String::new());
                        return false;
                    }
                    // Valid simple escapes decode; the canonical emission
                    // re-encodes through the same table (`"` -> `\"`, a
                    // raw `\/` -> `/`, ...).
                    '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => {
                        self.flush_high(sink, ensure_ascii, strict);
                        let decoded = match c {
                            'b' => '\u{8}',
                            'f' => '\u{c}',
                            'n' => '\n',
                            'r' => '\r',
                            't' => '\t',
                            other => other,
                        };
                        push_escaped_char(sink, decoded, ensure_ascii);
                        return false;
                    }
                    // An invalid escape keeps the backslash literally (the
                    // engine's `"x\qy"` -> `"x\\qy"` shape: the decoded
                    // text is backslash + q, re-encoded backslash-backslash).
                    _ => {
                        *strict = false;
                        self.flush_high(sink, ensure_ascii, strict);
                        push_escaped_char(sink, '\\', ensure_ascii);
                        push_escaped_char(sink, c, ensure_ascii);
                        return false;
                    }
                }
            }
            Esc::U(mut hex) => {
                if c.is_ascii_hexdigit() && hex.len() < 4 {
                    hex.push(c);
                    if hex.len() == 4 {
                        // Four collected hex digits: cannot fail to parse.
                        let code = u16::from_str_radix(&hex, 16).unwrap_or(0xFFFD);
                        self.emit_code(code, sink, ensure_ascii, strict);
                    } else {
                        self.esc = Esc::U(hex);
                    }
                    return false;
                }
                // Fewer than four hex digits: the escape is literal text
                // (the engine's `"\u41"` -> `"\\u41"` shape), and c is
                // ordinary content: fall through to the plain path.
                *strict = false;
                self.flush_high(sink, ensure_ascii, strict);
                push_escaped_char(sink, '\\', ensure_ascii);
                push_escaped_char(sink, 'u', ensure_ascii);
                for h in hex.chars() {
                    push_escaped_char(sink, h, ensure_ascii);
                }
            }
            Esc::None => {}
        }
        // Plain content. A backslash defers the flush: the next escape
        // may be the held surrogate's low half.
        if c == '\\' {
            self.esc = Esc::Backslash;
            return false;
        }
        self.flush_high(sink, ensure_ascii, strict);
        if c == self.delim {
            return true;
        }
        // A raw control char inside a string is the strict scanner's
        // rejection (json.loads fails on the whole input).
        if (c as u32) < 0x20 {
            *strict = false;
        }
        push_escaped_char(sink, c, ensure_ascii);
        false
    }

    /// A held lone high surrogate stranded by content that is not its
    /// low half flushes as U+FFFD (the engine's decode order: the
    /// surrogate's position first, then the content).
    fn flush_high(&mut self, sink: &mut String, ensure_ascii: bool, strict: &mut bool) {
        if self.high.take().is_some() {
            *strict = false;
            push_escaped_char(sink, '\u{fffd}', ensure_ascii);
        }
    }

    /// One complete `\uXXXX` code unit: a surrogate pair combines; a lone
    /// high surrogate is HELD (its low half may arrive in a later chunk);
    /// everything else (and a lone low surrogate) is U+FFFD, the engine's
    /// documented lone-surrogate divergence. The held surrogate at
    /// string close renders U+FFFD too (`{"a": "\ud83d` ->
    /// `{"a": "\ufffd"}`, the engine's own heal).
    fn emit_code(&mut self, code: u16, sink: &mut String, ensure_ascii: bool, strict: &mut bool) {
        if let Some(high) = self.high.take() {
            if (0xDC00..=0xDFFF).contains(&code) {
                let combined =
                    0x10000 + ((u32::from(high) - 0xD800) << 10) + (u32::from(code) - 0xDC00);
                if let Some(ch) = char::from_u32(combined) {
                    push_escaped_char(sink, ch, ensure_ascii);
                    return;
                }
            }
            // Not the low half: flush the held surrogate as U+FFFD, then
            // place the fresh unit on its own merits (it may itself be a
            // high surrogate to hold).
            *strict = false;
            push_escaped_char(sink, '\u{fffd}', ensure_ascii);
        }
        match char::from_u32(u32::from(code)) {
            Some(ch) => push_escaped_char(sink, ch, ensure_ascii),
            None if (0xD800..=0xDBFF).contains(&code) => self.high = Some(code),
            None => {
                *strict = false;
                push_escaped_char(sink, '\u{fffd}', ensure_ascii)
            }
        }
    }

    /// The held state at a close-time render: an incomplete `\u` escape
    /// or a dangling backslash is literal text; a held high surrogate is
    /// U+FFFD (the engine's `{"a": "\ud83d` heal).
    fn render_tail(&self, sink: &mut String, ensure_ascii: bool) {
        if self.high.is_some() {
            push_escaped_char(sink, '\u{fffd}', ensure_ascii);
            return;
        }
        match &self.esc {
            Esc::None => {}
            Esc::Backslash => push_escaped_char(sink, '\\', ensure_ascii),
            Esc::U(hex) => {
                push_escaped_char(sink, '\\', ensure_ascii);
                push_escaped_char(sink, 'u', ensure_ascii);
                for h in hex.chars() {
                    push_escaped_char(sink, h, ensure_ascii);
                }
            }
        }
    }
}

/// The held-back token. In-container kinds (`Num`/`Word`) render at their
/// terminator into `out`; top-level kinds (`TopNum`/`TopWord`/`TopStr`)
/// are provisional scalars: the whole input could still be one strict
/// JSON value, so they commit at `end` only when the held text loads,
/// else they were prose and drop.
enum Pending {
    Num {
        text: String,
        key: bool,
    },
    Word {
        text: String,
        key: bool,
        /// The word was born from a number run (the engine's
        /// rewind-and-reparse): a leading `-` was a stray sign, dropped
        /// at the render (`{"a": -NaN}` -> `{"a": "NaN"}`) unless the
        /// full word is the strict `-Infinity` spelling (the number
        /// lane keeps it, a special float).
        from_number_run: bool,
    },
    TopNum {
        text: String,
        done: bool,
    },
    TopWord {
        text: String,
        done: bool,
    },
    TopStr {
        buf: String,
        st: StrState,
        done: bool,
    },
}

/// The comment skipper. `Hash` carries the engine's context-dependent
/// terminator set, computed where the comment started (the terminator is
/// left in place for the structural machine); `Line` (`//`) stops at
/// newlines only; `Block` carries the previous char for the `*/` scan
/// (seeded `*`, the engine's `/*/` quirk included); `Slash` is the
/// one-char buffer that decides what a bare `/` started.
enum CommentState {
    Hash {
        arr: bool,
        obj_value: bool,
        obj_key: bool,
    },
    Line,
    Block {
        prev: char,
    },
    Slash,
}

impl StreamingRepairer {
    /// A fresh repairer. `ensure_ascii` is `repair_json`'s serialization
    /// knob (default true: every char above `~` escapes as `\uXXXX`,
    /// astral chars as surrogate pairs).
    pub fn new(ensure_ascii: bool) -> StreamingRepairer {
        StreamingRepairer {
            out: String::new(),
            frames: Vec::new(),
            pos: Pos::Value,
            string: None,
            pending: None,
            comment: None,
            top_clean: true,
            top_done: false,
            ensure_ascii,
            finished: false,
            emitted: 0,
            last_string_start: None,
            string_raw_tail: None,
            prev_char_ws: false,
            empty_marks: Vec::new(),
            empty_pending: None,
            strict_clean: true,
            last_key_empty: false,
            garbage_run: false,
            paren_gate: None,
        }
    }

    /// Feed one chunk; returns the text this call newly emitted (the
    /// repaired delta; see the module docs for the retraction exception).
    /// The only error is the container-nesting cap, the engine's own
    /// normalized message; after an error the machine's state is
    /// unspecified: `reset` before reuse.
    pub fn push(&mut self, chunk: &str) -> Result<String, String> {
        if self.finished {
            return Err("the repairer is finalized: end() already ran; reset() to reuse".into());
        }
        for c in chunk.chars() {
            self.process(c)?;
        }
        let start = self.emitted.min(self.out.len());
        self.emitted = self.out.len();
        Ok(self.out[start..].to_string())
    }

    /// Finalize: resolve the pending token, close the open string and
    /// every open container (the truncated-output heal), return the final
    /// text. Idempotent (a second `end` returns the same text).
    pub fn end(&mut self) -> Result<String, String> {
        if !self.finished {
            let final_text = self.render_closed();
            self.out = final_text;
            self.emitted = self.out.len();
            self.frames.clear();
            self.string = None;
            self.pending = None;
            self.comment = None;
            self.empty_pending = None;
            self.paren_gate = None;
            self.garbage_run = false;
            self.finished = true;
        }
        Ok(self.out.clone())
    }

    /// The current state rendered closed (valid JSON now), without
    /// consuming anything: `snapshot() == end()` on the same prefix,
    /// always, and pushing continues exactly as if it had not run.
    pub fn snapshot(&self) -> String {
        if self.finished {
            return self.out.clone();
        }
        self.render_closed()
    }

    /// Drop everything: a fresh repairer for the next document.
    pub fn reset(&mut self) {
        let ensure_ascii = self.ensure_ascii;
        *self = StreamingRepairer::new(ensure_ascii);
    }

    // -- the machine ----------------------------------------------------

    /// The per-char dispatch: comments first (they consume anything),
    /// then the open string, then the pending token, then structure.
    /// The one-char look-back (`prev_char_ws`) settles AFTER the
    /// dispatch: the readers (the empty-element skip's mark) want the
    /// char BEFORE this one.
    fn process(&mut self, c: char) -> Result<(), String> {
        // The strictly-empty element's decision (see `EmptyPending`): the
        // char after the element is the engine's skip cursor. A `,` or
        // the parent frame's closer keeps the element; whitespace
        // defers to close time (the whole-input `json.loads` fast path
        // may still keep it: the mark arms, and only the strict-clean
        // gate drains); everything else drops it and is CONSUMED with
        // it (parse_array's one-char stall guard: the skip's
        // `index += 1` eats exactly one char, whatever it is).
        if let Some(pending) = self.empty_pending.take() {
            let parent_closer = if pending.parent_paren { ')' } else { ']' };
            if c == ',' || c == parent_closer {
                // The repair lane's skip does not fire on a separator:
                // the element keeps in both lanes.
            } else if c.is_whitespace() && !pending.parent_paren {
                // The repair lane drops here (its cursor reads the
                // whitespace char); the fast path may still keep the
                // whole document. Deferred: the mark drains iff the
                // engine's own decision procedure ran the repair lane
                // (the strict-clean gate at close time). A PAREN lane's
                // element never defers: the tuple grammar is never
                // strict-valid, so the repair lane always decides --
                // the drop fires now, on fresh coordinates.
                self.empty_marks.push(EmptyMark {
                    start: pending.start,
                    end: pending.end,
                    was_first: pending.was_first,
                    frame: pending.frame,
                });
            } else {
                // The engine's repair-lane skip: drop the element and
                // consume the char with it. The parent frame may
                // already be gone (the element's own parse consumed a
                // mismatched closer that closed the document): the
                // span's truncation stands on its own then, and the
                // root's closer (consumed with the mismatched close)
                // re-emits.
                if self.frames.last().is_some() {
                    self.drop_marked_member(pending.start);
                } else if pending.start <= self.out.len() {
                    self.out.truncate(pending.start);
                    self.emitted = self.emitted.min(pending.start);
                    self.out.push(']');
                }
                self.prev_char_ws = is_py_whitespace(c);
                return Ok(());
            }
        }
        // The top-level `(`'s conservative gate (see `ParenGate`): the
        // engine decides the tuple container from the first inner char
        // and from what follows the group's `)`; both decisions replay
        // here before anything structural runs.
        match &mut self.paren_gate {
            Some(ParenGate::FirstChar { .. }) => {
                if c.is_whitespace() {
                    return Ok(());
                }
                let gate = self.paren_gate.take();
                if let Some(ParenGate::FirstChar { buf }) = gate {
                    self.paren_gate_first_char(c, buf)?;
                }
                return Ok(());
            }
            Some(ParenGate::TailCheck) => {
                if c == '\n' || c == '\r' {
                    // The line end settles the gate: only whitespace
                    // preceded it, the engine's own true-exit.
                    self.paren_gate = None;
                    return Ok(());
                }
                if c.is_whitespace() {
                    return Ok(());
                }
                self.paren_gate = None;
                // A token where the engine's gate demanded whitespace:
                // the `(` was never a value. The group's rendering
                // retracts whole (it began at the clean top level's
                // empty buffer) and the char replays as prose.
                self.out.clear();
                self.emitted = 0;
                self.top_clean = false;
                self.strict_clean = false;
                return self.process(c);
            }
            None => {}
        }
        let out = self.process_inner(c);
        self.prev_char_ws = is_py_whitespace(c);
        out
    }

    /// The gate's first-inner-char decision (see `ParenGate::FirstChar`).
    /// `buf` holds the alphabetic chars buffered so far (at most five:
    /// the longest literal spelling, `false`). A value-start-ish char
    /// or a literal prefix opens the tuple container; anything else was
    /// inline prose -- the `(` drops and the chars replay at the top
    /// level (the engine's parse_json skipped it and moved on).
    fn paren_gate_first_char(&mut self, c: char, buf: String) -> Result<(), String> {
        let mut buf = buf;
        if c.is_alphabetic() && buf.len() < 5 {
            buf.push(c);
            if buf.len() < 4 {
                self.paren_gate = Some(ParenGate::FirstChar { buf });
                return Ok(());
            }
            // Four (or five) chars: the literal prefix decides. The
            // check reads the lowercased buffer, the engine's inner-text
            // prefix.
            let lower = buf.to_lowercase();
            let is_literal = lower.starts_with("true")
                || lower.starts_with("null")
                || lower.starts_with("none")
                || (buf.len() >= 5 && lower.starts_with("false"));
            if buf.len() == 4 && !is_literal {
                // A fifth char could still complete `false`.
                self.paren_gate = Some(ParenGate::FirstChar { buf });
                return Ok(());
            }
            if !is_literal {
                // Not a literal prefix (and the buffer is full): the
                // `(` was prose.
                self.top_clean = false;
                self.strict_clean = false;
                for b in buf.chars() {
                    self.process(b)?;
                }
                return Ok(());
            }
        } else if !buf.is_empty() {
            // A terminator arrived on a short non-literal buffer (the
            // engine's inner-text prefix `buf + c` is no literal): the
            // `(` was prose; the buffered chars replay at the top level.
            self.top_clean = false;
            self.strict_clean = false;
            for b in buf.chars() {
                self.process(b)?;
            }
            return self.process(c);
        } else {
            match c {
                // The empty group `()`: the engine's own value-start.
                ')' | '{' | '[' | '(' => {
                    self.open_container(FrameKind::Paren)?;
                    return self.process(c);
                }
                c if STRING_DELIMITERS.contains(&c) => {
                    self.open_container(FrameKind::Paren)?;
                    return self.process(c);
                }
                c if c.is_ascii_digit() || c == '-' || c == '.' => {
                    self.open_container(FrameKind::Paren)?;
                    return self.process(c);
                }
                // Inline prose: the engine's parse_json skipped the `(`
                // and re-ran the top-level dispatch on the char.
                _ => {
                    self.top_clean = false;
                    self.strict_clean = false;
                    return self.process(c);
                }
            }
        }
        // Literal prefix confirmed: open the tuple container and replay
        // the buffered chars into it (they spell the word).
        self.open_container(FrameKind::Paren)?;
        for b in buf.chars() {
            self.process(b)?;
        }
        Ok(())
    }

    fn process_inner(&mut self, c: char) -> Result<(), String> {
        // The engine's garbage-item run (see `garbage_run`): the item
        // parse started at a `,`, so parse_json's ignore-and-move-on
        // loop consumes every non-value-start char -- the frame's own
        // closer included -- until a value starts (the item's parse
        // resumes on it), a `}` terminates the loop (consumed by it),
        // or the input ends.
        if self.garbage_run {
            if c == '}' {
                self.garbage_run = false;
                self.pop_frame();
                return Ok(());
            }
            if Self::is_value_start(c) {
                self.garbage_run = false;
                // fall through: the value starts the item's parse
            } else {
                return Ok(());
            }
        }
        for f in &mut self.frames {
            if f.kind == FrameKind::Obj {
                f.body_chars += 1;
                if !is_py_whitespace(c) {
                    f.body_nonspace += 1;
                }
            }
        }
        if self.comment.is_some() {
            self.feed_comment(c);
            return Ok(());
        }
        if self.string.is_some() {
            // Take and put back (state moves, nothing reallocates): the
            // close transition needs `&mut self` while the feed needs the
            // string state out of it.
            self.string_raw_tail = Some(c);
            let mut st = self.string.take().expect("checked above");
            let closed = st.feed(c, &mut self.out, self.ensure_ascii, &mut self.strict_clean);
            if closed {
                self.out.push('"');
                self.pos = if st.is_key {
                    // the key's own emptiness: the engine's key loop
                    // DISCARDS an empty-key attempt at anything but its
                    // `:` (the `{"" ,}`-corner)
                    self.last_key_empty = self.last_string_start == Some(self.out.len() - 1);
                    Pos::Colon
                } else {
                    Pos::AfterValue
                };
                // A strictly-empty string element (`""`) completing
                // inside an array/tuple frame arms the same one-char
                // skip decision the container pops arm (the engine's
                // item skip reads the next char).
                self.arm_empty_pending();
            } else {
                self.string = Some(st);
            }
            return Ok(());
        }
        if self.pending.is_some() {
            return self.feed_pending(c);
        }
        if self.frames.is_empty() {
            return self.feed_top(c);
        }
        self.feed_structure(c)
    }

    /// The top level (no frames open): whitespace and comments pass,
    /// containers start, and a clean top level may hold a provisional
    /// scalar; everything else is prose. After the root value closed
    /// (`top_done`), only whitespace and comments are tolerated.
    fn feed_top(&mut self, c: char) -> Result<(), String> {
        if self.top_done {
            match c {
                c if c.is_whitespace() => {
                    // The strict scanner's whitespace set only: any
                    // other Unicode whitespace failed `json.loads` (the
                    // repair lane decides).
                    if !matches!(c, ' ' | '\t' | '\n' | '\r') {
                        self.strict_clean = false;
                    }
                }
                '#' | '/' => {
                    // A comment (or any trailing token) after the root:
                    // the whole input was not one strict JSON value.
                    self.strict_clean = false;
                    match c {
                        '#' => self.comment = Some(self.hash_comment()),
                        _ => self.comment = Some(CommentState::Slash),
                    }
                }
                _ => self.strict_clean = false,
            }
            return Ok(());
        }
        let clean = self.top_clean;
        match c {
            c if c.is_whitespace() => {
                if !matches!(c, ' ' | '\t' | '\n' | '\r') {
                    self.strict_clean = false;
                }
            }
            '{' => self.open_container(FrameKind::Obj)?,
            '[' => self.open_container(FrameKind::Arr)?,
            // A leading `(` opens the tuple container only from a clean
            // top level (the engine's conservative tuple detection),
            // and even then only after the first-inner-char and the
            // post-group tail checks (see `ParenGate`).
            '(' if clean => self.paren_gate = Some(ParenGate::FirstChar { buf: String::new() }),
            c if STRING_DELIMITERS.contains(&c) && clean => {
                let mut buf = String::new();
                buf.push('"');
                self.pending = Some(Pending::TopStr {
                    buf,
                    st: StrState {
                        delim: c,
                        is_key: false,
                        esc: Esc::None,
                        high: None,
                    },
                    done: false,
                });
            }
            c if (c.is_ascii_digit() || c == '-' || c == '.') && clean => {
                self.pending = Some(Pending::TopNum {
                    text: c.to_string(),
                    done: false,
                });
            }
            c if c.is_alphabetic() && clean => {
                self.pending = Some(Pending::TopWord {
                    text: c.to_string(),
                    done: false,
                });
            }
            // Prose (or a second scalar after a provisional one): dropped.
            _ => {
                self.top_clean = false;
                self.strict_clean = false;
            }
        }
        Ok(())
    }

    /// The structural positions inside the innermost frame. Whitespace
    /// and comments are handled before the position dispatch (they are
    /// legal between every pair of tokens).
    fn feed_structure(&mut self, c: char) -> Result<(), String> {
        match c {
            c if c.is_whitespace() => {
                // The strict scanner's whitespace set only.
                if !matches!(c, ' ' | '\t' | '\n' | '\r') {
                    self.strict_clean = false;
                }
            }
            '#' | '/' => {
                // Comments are a repair lane (json.loads fails on them).
                self.strict_clean = false;
                match c {
                    '#' => self.comment = Some(self.hash_comment()),
                    _ => self.comment = Some(CommentState::Slash),
                }
            }
            _ => match self.pos {
                Pos::Key => match c {
                    '}' => self.close_up(c),
                    // A `]`/`)` where a key was due is garbage to the
                    // engine's key parse (the object continues -- the
                    // object closes only at its own `}`, or at `]` after
                    // a member); the machine must not close levels with
                    // it: `{]'k': 1}` keeps its member.
                    ']' | ')' => {
                        self.strict_clean = false;
                    }
                    ',' => {
                        // A separator where a key was due: `{,}`-shapes,
                        // the engine's empty-object repair territory.
                        self.strict_clean = false;
                    }
                    ':' => {
                        // A missing key: the whole pair drops (the
                        // engine's repair lane: `{: 1}` -> `{}`). The
                        // `:` is skipped and the would-be value becomes
                        // a dangling key that drops the same way. Valid
                        // input never lands here: a closed key string
                        // leaves Pos::Colon, so `{"": 1}` passes through
                        // untouched.
                        self.strict_clean = false;
                        self.mark_body_colon();
                    }
                    c if STRING_DELIMITERS.contains(&c) => {
                        self.begin_common();
                        self.open_string(c, true);
                    }
                    c if c.is_alphanumeric() => {
                        self.begin_common();
                        self.pending = Some(Pending::Word {
                            text: c.to_string(),
                            key: true,
                            from_number_run: false,
                        });
                    }
                    c if c.is_ascii_digit() || c == '-' || c == '.' => {
                        self.begin_common();
                        self.pending = Some(Pending::Num {
                            text: c.to_string(),
                            key: true,
                        });
                    }
                    _ => {
                        self.strict_clean = false;
                    }
                },
                Pos::Colon => match c {
                    ':' => {
                        self.strict_clean = false;
                        self.out.push_str(": ");
                        self.pos = Pos::MemberValue;
                    }
                    // The member-boundary heals: the engine's object
                    // loop consumes a ',' or a mismatched closer where
                    // the colon was due and parses the VALUE — the
                    // char acts AS the colon (`{"a", 1}` ->
                    // `{"a": 1}`, `{"a"]` -> `{"a": ""}` at the
                    // close-time heal) — an EMPTY-KEY attempt is the
                    // engine's own exception: the key loop DISCARDS it
                    // at anything but its `:` (the `{"" ,}`-corner),
                    // the pair drops.
                    ',' | ']' | ')' => {
                        self.strict_clean = false;
                        if self.last_key_empty {
                            self.drop_dangling_member();
                            return self.process(c);
                        }
                        self.out.push_str(": ");
                        self.pos = Pos::MemberValue;
                    }
                    '}' => {
                        self.drop_dangling_member();
                        self.process(c)?;
                    }
                    // A missing colon: insert one and parse the value (a
                    // documented divergence: the engine's repair lane
                    // heals `{"a" 1}` to `{"a": ""}`; the stream prefers
                    // keeping the value it saw).
                    c if STRING_DELIMITERS.contains(&c)
                        || c.is_alphabetic()
                        || c.is_ascii_digit()
                        || c == '-'
                        || c == '.'
                        || c == '{'
                        || c == '['
                        || c == '(' =>
                    {
                        self.strict_clean = false;
                        self.out.push_str(": ");
                        self.pos = Pos::MemberValue;
                        self.process(c)?;
                    }
                    _ => {
                        self.strict_clean = false;
                    }
                },
                Pos::Value | Pos::MemberValue => {
                    // A structural char arriving with the member's value
                    // still missing heals the value first (`{"a": }` ->
                    // `{"a": ""}`, the engine's own heal) and completes
                    // the member (the heal fires once); arrays and
                    // tuples heal nothing (`[1, }` -> `[1]`).
                    if self.pos == Pos::MemberValue && matches!(c, ']' | '}' | ')' | ',') {
                        self.strict_clean = false;
                        self.out.push_str("\"\"");
                        self.pos = Pos::AfterValue;
                    }
                    match c {
                        ']' | '}' | ')' => {
                            // A closer where a value was due is a trailing
                            // comma's shape when members exist (never
                            // strict); an empty container's own closer is
                            // the strict `[]`/`{}`. (A MemberValue closer
                            // healed `""` above and already flipped.)
                            if self.pos == Pos::Value
                                && self.frames.last().is_some_and(|f| f.member_count > 0)
                            {
                                self.strict_clean = false;
                            }
                            self.close_up(c)
                        }
                        // The engine's garbage-item run (parse_array
                        // items: the item parse started at a `,`):
                        // parse_json's ignore loop consumes it and every
                        // following non-value-start char, the frame's
                        // own closer included. The item slot is not
                        // spent: the next value becomes the item.
                        ',' if self.pos == Pos::Value => {
                            self.strict_clean = false;
                            // The comma is the tuple classifier's
                            // separator evidence (the explicit-tuple
                            // scan sees it) even though the item parse
                            // treats it as garbage.
                            self.note_comma();
                            self.garbage_run = true;
                        }
                        ',' => self.note_comma(),
                        // The member's separator already went out with
                        // its key: no begin_common here.
                        _ if self.pos == Pos::MemberValue && Self::is_value_start(c) => {
                            self.feed_structure_value(c)?
                        }
                        _ if self.pos == Pos::Value && Self::is_value_start(c) => {
                            self.begin_common();
                            self.feed_structure_value(c)?;
                        }
                        _ => self.strict_clean = false,
                    }
                }
                Pos::AfterValue => {
                    let obj = matches!(self.frames.last().map(|f| &f.kind), Some(FrameKind::Obj));
                    match c {
                        ',' => {
                            self.note_comma();
                            self.pos = if obj { Pos::Key } else { Pos::Value };
                        }
                        '}' | ']' | ')' => self.close_up(c),
                        // In an object a string or bare word continues as
                        // the next member's key (the missing-comma shape
                        // `{"a": 1 "b": 2}`); a stray number starts no
                        // key (the engine drops it: `{"a": 1 2}` ->
                        // `{"a": 1}`).
                        _ if obj => {
                            self.strict_clean = false;
                            if STRING_DELIMITERS.contains(&c) {
                                self.begin_common();
                                self.open_string(c, true);
                            } else if c.is_alphabetic() {
                                self.begin_common();
                                self.pending = Some(Pending::Word {
                                    text: c.to_string(),
                                    key: true,
                                    from_number_run: false,
                                });
                            } else if c.is_ascii_digit() || c == '-' || c == '.' {
                                // A stray number became the next
                                // member's KEY ATTEMPT (the engine's key
                                // lane): the attempt discards at its
                                // own '}' or the input end (`{"a": 1 2}`
                                // -> `{"a": 1}`) and commits at its `:`
                                // (`{"a": 1 2]: 3}` -> `{"a": 1, "2]": 3}`).
                                self.begin_common();
                                self.pending = Some(Pending::Num {
                                    text: c.to_string(),
                                    key: true,
                                });
                            }
                        }
                        // Arrays and tuples take the missing comma and
                        // parse the next value — real value-starts only
                        // (a garbage char must not emit the deferred
                        // separator into an array it cannot close).
                        _ => {
                            if Self::is_value_start(c) {
                                self.strict_clean = false;
                                self.begin_common();
                                self.feed_structure_value(c)?;
                            } else {
                                self.strict_clean = false;
                            }
                        }
                    }
                }
            },
        }
        Ok(())
    }

    /// Does this char start a value (a container, a string, a bare word,
    /// a number)? The guard that keeps garbage from consuming the
    /// deferred separator or a member slot.
    fn is_value_start(c: char) -> bool {
        c == '{'
            || c == '['
            || c == '('
            || STRING_DELIMITERS.contains(&c)
            || c.is_alphabetic()
            || c.is_ascii_digit()
            || c == '-'
            || c == '.'
    }

    /// A value-start char dispatched from `Pos::Value`'s arms, shared
    /// with `AfterValue`'s missing-comma path (which already ran the
    /// deferred separator).
    fn feed_structure_value(&mut self, c: char) -> Result<(), String> {
        match c {
            '{' => self.open_container(FrameKind::Obj)?,
            '[' => self.open_container(FrameKind::Arr)?,
            '(' => self.open_container(FrameKind::Paren)?,
            c if STRING_DELIMITERS.contains(&c) => self.open_string(c, false),
            c if c.is_alphabetic() => {
                self.pending = Some(Pending::Word {
                    text: c.to_string(),
                    key: false,
                    from_number_run: false,
                });
            }
            c if c.is_ascii_digit() || c == '-' || c == '.' => {
                self.pending = Some(Pending::Num {
                    text: c.to_string(),
                    key: false,
                });
            }
            _ => {}
        }
        Ok(())
    }

    /// The pending token's per-char rules.
    fn feed_pending(&mut self, c: char) -> Result<(), String> {
        match self.pending.take() {
            Some(Pending::Num { mut text, key }) => {
                let in_object = matches!(self.frames.last().map(|f| &f.kind), Some(FrameKind::Obj));
                // Python digit-group underscores ride along in the run:
                // the value/key renders filter them (`render_number_value`,
                // the key's filtered spelling) and the rewind-to-word
                // keeps them (the engine's whole-run word, underscores
                // included: `[5_!5_K` -> `[5, "5_K"]`).
                if c == '_' {
                    text.push(c);
                    self.pending = Some(Pending::Num { text, key });
                    return Ok(());
                }
                // The engine's object-KEY attempt is the STRING lane:
                // whitespace holds the attempt and the NEXT char decides
                // (the kept attempt commits at its `:`; an
                // attempt-invalidating char drops it -- the Word{key}
                // arm's ws-touched drop below). The deferral rides the
                // word state (the entry skip's strip applies at the
                // commit: `{-ab : 1}` -> `{"ab": 1}`).
                if key && c.is_whitespace() {
                    self.pending = Some(Pending::Word {
                        text,
                        key: true,
                        from_number_run: true,
                    });
                    return Ok(());
                }
                // The key-attempt string absorbs the array bracket too
                // (its break set is the attempt's `:` and whitespace
                // only): `{... 2]: 3}` -> `{"2]": 3}`; at the input end
                // the attempt discards.
                if key && c == ']' {
                    text.push(c);
                    self.pending = Some(Pending::Word {
                        text,
                        key: true,
                        from_number_run: true,
                    });
                    return Ok(());
                }
                // The KEY-attempt IS the engine's STRING lane: every
                // char that is not the object's own `}` and not a raw
                // control char (the engine's string scan ends there:
                // `{{\r]0\x01...` keeps the key `0`) is the attempt's
                // content (`{3½:` -> `{"3½": ""}`); the attempt's own
                // `:` commits it (the finish_number render below).
                if key && c != '}' && c != ':' && c >= ' ' {
                    text.push(c);
                    self.pending = Some(Pending::Word {
                        text,
                        key: true,
                        from_number_run: true,
                    });
                    return Ok(());
                }
                // A '+'/'-' continues the run only as an exponent sign.
                let sign_after_e = matches!(c, '+' | '-')
                    && text
                        .chars()
                        .rev()
                        .find(|&x| x != '_')
                        .is_some_and(|p| p == 'e' || p == 'E');
                // The `,` joins the run in the object lane only (the
                // engine's currency-comma shape); in arrays and tuple
                // groups it is the item separator.
                let is_number_char =
                    (c == ',' && in_object) || matches!(c, '0'..='9' | '-' | '.' | 'e' | 'E' | '/');
                if is_number_char || sign_after_e {
                    text.push(c);
                    self.pending = Some(Pending::Num { text, key });
                    return Ok(());
                }
                // Terminator: the engine's single trailing-char rollback,
                // then the same value lanes. The popped char (and the
                // terminator) are reprocessed structurally. The KEY
                // attempt is the engine's STRING lane: no rollback — the
                // run's trailing `-`/`,` are the attempt's own content
                // (`{8-:(` -> `{"8-": []}`).
                if c.is_alphabetic() {
                    // The run was a string after all (the engine's
                    // rewind-and-reparse: the whole run becomes a bare
                    // word, digits and underscores included:
                    // `{"a": 12abc}` -> `{"a": "12abc"}`). A leading
                    // `-` followed by a letter was a stray sign, dropped
                    // at the render (when the word is complete).
                    text.push(c);
                    self.pending = Some(Pending::Word {
                        text,
                        key,
                        from_number_run: true,
                    });
                    return Ok(());
                }
                let popped = if key {
                    None
                } else {
                    pop_trailing_number_char(&mut text)
                };
                // An empty run at an item position retracts the element
                // whole (the engine's falsy-nudge progress rule: the
                // rolled-back sign emits nothing in an array, `[1,-]` ->
                // `[1]`, `[1e-}` -> `["1e"]`); in an object the member
                // value heals to `""` (`{"a": -}`), the engine's own
                // split.
                if text.is_empty()
                    && !key
                    && self
                        .frames
                        .last()
                        .is_some_and(|f| !matches!(f.kind, FrameKind::Obj))
                {
                    self.retract_in_flight_member();
                    self.pos = Pos::AfterValue;
                    return self.process(c);
                }
                self.finish_number(&text, key, popped.or_else(|| text.chars().next_back()));
                // A pop that EMPTIES the run drops the popped char: the
                // reprocess would re-create the same one-char pending at
                // an after-value position (a `-` or `.` in an array),
                // whose termination would pop it again, forever (the
                // engine escapes the same cycle through its falsy-nudge
                // progress rule; this is the machine's: the reprocessed
                // sign's one-char element retracts, the heal terminates).
                if let Some(p) = popped
                    && !text.is_empty()
                {
                    self.process(p)?;
                }
                self.process(c)
            }
            Some(Pending::Word {
                mut text,
                key,
                from_number_run,
            }) => {
                if key && c == ':' {
                    // The colon proves it was a key: commit it. A bare
                    // word key is the repair lane (strict keys are
                    // quoted), and the key attempt is the engine's
                    // STRING lane: the entry skip's leading
                    // non-alphanumeric garbage drops with it (`{-ab: 1}`
                    // -> `{"ab": 1}`, `{.5: 1}` -> `{"5": 1}`,
                    // `{-eOe: 6}` -> `{"eOe": 6}`). An attempt that
                    // strips to NOTHING drops the pair whole (the
                    // engine's empty missing-quote key attempt:
                    // `{-\x0b\r:-` -> `{}`).
                    self.strict_clean = false;
                    self.last_key_empty = false;
                    let stripped = text
                        .trim()
                        .trim_start_matches(|ch: char| !ch.is_alphanumeric());
                    if stripped.is_empty() {
                        self.drop_dangling_member();
                        return self.process(c);
                    }
                    let rendered =
                        serializer::dumps(&Value::Str(stripped.to_string()), self.ensure_ascii);
                    self.out.push_str(&rendered);
                    self.out.push_str(": ");
                    // The colon was this key's own commit char: the value
                    // that follows is the member's value, already counted.
                    self.pos = Pos::MemberValue;
                    return Ok(());
                }
                if c == ',' || c == '}' || c == ']' {
                    if key {
                        // The object-KEY attempt is the engine's
                        // missing-quote STRING lane: its break set is
                        // the attempt's own `:` and whitespace ONLY —
                        // the ',' and ']' are the attempt's OWN content
                        // (`{a,b: 1}` -> `{"a,b": 1}`, `{a]: 1}` ->
                        // `{"a]": 1}`); the '}' is the object's closer
                        // and the attempt dies with it (the engine's
                        // `{a: 1, b}` -> `{"a": 1}` shape).
                        if c == '}' {
                            self.drop_dangling_member();
                            return self.process(c);
                        }
                        if c == ',' && self.prev_char_ws {
                            // The corner-case: the whitespace-broken
                            // attempt whose next non-ws char is the ','
                            // COMMITS (the engine's keep-set is
                            // `:`/`,`), and the ',' acts as the
                            // attempt's colon (the engine's object loop
                            // consumes it and parses the value).
                            let stripped = text
                                .trim()
                                .trim_start_matches(|ch: char| !ch.is_alphanumeric());
                            if !stripped.is_empty() {
                                self.strict_clean = false;
                                let rendered = serializer::dumps(
                                    &Value::Str(stripped.to_string()),
                                    self.ensure_ascii,
                                );
                                self.out.push_str(&rendered);
                                self.out.push_str(": ");
                                self.pos = Pos::MemberValue;
                                return Ok(());
                            }
                        }
                        if c == ',' || c == ']' {
                            text.push(c);
                            self.pending = Some(Pending::Word {
                                text,
                                key,
                                from_number_run,
                            });
                            return Ok(());
                        }
                        self.drop_dangling_member();
                        return self.process(c);
                    }
                    let value = render_word(text.trim(), from_number_run);
                    // The strict grammar's own spellings only: the
                    // case-insensitive literal family and every other
                    // word is the repair lane (a rewind-born word's
                    // stray-sign drop included).
                    if from_number_run
                        || !matches!(
                            text.trim(),
                            "true" | "false" | "null" | "NaN" | "Infinity" | "-Infinity"
                        )
                    {
                        self.strict_clean = false;
                    }
                    self.out
                        .push_str(&serializer::dumps(&value, self.ensure_ascii));
                    self.pos = Pos::AfterValue;
                    return self.process(c);
                }
                // The engine's unquoted-literal lane: the run's text is
                // the CASE-INSENSITIVE literal spelling and the next
                // char is the paren group's `)` -- the literal commits
                // and the `)` stays for the group's loop (`(true)` ->
                // `true`, while a non-literal run swallows the `)`: the
                // engine's `[(x)]` -> `["x)"]`).
                if c == ')'
                    && !key
                    && !from_number_run
                    && matches!(
                        text.trim().to_lowercase().as_str(),
                        "true" | "false" | "null" | "none"
                    )
                {
                    let value = keyword_or_str(text.trim());
                    self.out
                        .push_str(&serializer::dumps(&value, self.ensure_ascii));
                    self.pos = Pos::AfterValue;
                    return self.process(c);
                }
                // The engine's object-key attempt INVALIDATES at a
                // whitespace when the next char is not its `:` (the
                // missing-quote key returns "" and the object's key
                // loop retries from the next word): `{a b: 1}` ->
                // `{"b": 1}`, `{-ab x: 1}` -> `{"x": 1}`. Whitespace
                // itself never invalidates (the attempt holds across a
                // whitespace run: `{.x1  : 1}` -> `{"x1": 1}`); the
                // separator/closer arms above already drop the attempt
                // the same way.
                if key && self.prev_char_ws && !c.is_whitespace() && c != ':' {
                    self.drop_dangling_member();
                    return self.process(c);
                }
                text.push(c);
                self.pending = Some(Pending::Word {
                    text,
                    key,
                    from_number_run,
                });
                Ok(())
            }
            Some(Pending::TopNum { mut text, done }) => {
                if c.is_whitespace() {
                    // Whitespace after a (terminated) provisional scalar
                    // keeps it provisionally complete: the commit
                    // decision waits for end or the next token.
                    self.pending = Some(Pending::TopNum { text, done: true });
                    return Ok(());
                }
                let sign_after_e = matches!(c, '+' | '-')
                    && text
                        .chars()
                        .rev()
                        .find(|&x| x != '_')
                        .is_some_and(|p| p == 'e' || p == 'E');
                let is_number_char = matches!(c, '0'..='9' | '-' | '.' | 'e' | 'E' | '/' | ',');
                // An alphabetic char directly after a bare minus plus
                // letters only keeps the provisional scalar alive: the
                // strict special-float spellings (`-Infinity`) load at
                // the commit; after a number it is prose (`12abc`).
                let rest = text.strip_prefix('-').unwrap_or(text.as_str());
                let special_float_run =
                    c.is_alphabetic() && rest.chars().all(|x| x.is_alphabetic() || x == '_');
                if is_number_char || sign_after_e || special_float_run {
                    if !done {
                        text.push(c);
                        self.pending = Some(Pending::TopNum { text, done: false });
                    } else {
                        // A second run after a terminated scalar: the
                        // whole input is not one strict value: prose.
                        self.top_clean = false;
                        self.strict_clean = false;
                    }
                    return Ok(());
                }
                self.top_clean = false;
                self.strict_clean = false;
                if c == '{' || c == '[' {
                    // Prose, then a container: the container wins.
                    return self.process(c);
                }
                Ok(())
            }
            Some(Pending::TopWord { mut text, done }) => {
                if c.is_whitespace() {
                    // The TopNum arm's whitespace rule.
                    self.pending = Some(Pending::TopWord { text, done: true });
                    return Ok(());
                }
                if c.is_alphabetic() {
                    if !done {
                        text.push(c);
                        self.pending = Some(Pending::TopWord { text, done: false });
                    } else {
                        self.top_clean = false;
                        self.strict_clean = false;
                    }
                    return Ok(());
                }
                self.top_clean = false;
                self.strict_clean = false;
                if c == '{' || c == '[' {
                    return self.process(c);
                }
                Ok(())
            }
            Some(Pending::TopStr {
                mut buf,
                mut st,
                done,
            }) => {
                if done {
                    if c.is_whitespace() {
                        self.pending = Some(Pending::TopStr {
                            buf,
                            st,
                            done: true,
                        });
                        return Ok(());
                    }
                    self.top_clean = false;
                    self.strict_clean = false;
                    if c == '{' || c == '[' {
                        return self.process(c);
                    }
                    return Ok(());
                }
                let closed = st.feed(c, &mut buf, self.ensure_ascii, &mut self.strict_clean);
                if closed {
                    buf.push('"');
                    self.pending = Some(Pending::TopStr {
                        buf,
                        st,
                        done: true,
                    });
                } else {
                    self.pending = Some(Pending::TopStr {
                        buf,
                        st,
                        done: false,
                    });
                }
                Ok(())
            }
            None => Ok(()),
        }
    }

    /// Comments consume every char until their terminator (which stays in
    /// place for the structural machine).
    fn feed_comment(&mut self, c: char) {
        match self.comment.take() {
            Some(CommentState::Hash {
                arr,
                obj_value,
                obj_key,
            }) => {
                if c == '\n'
                    || c == '\r'
                    || (arr && c == ']')
                    || (obj_value && c == '}')
                    || (obj_key && c == ':')
                {
                    // The terminator is left in place: reprocess it. No
                    // arm reachable from a comment terminator can fail
                    // (none opens a container), so the Result is dropped.
                    let _ = self.process(c);
                } else {
                    self.comment = Some(CommentState::Hash {
                        arr,
                        obj_value,
                        obj_key,
                    });
                }
            }
            Some(CommentState::Line) => {
                if c != '\n' && c != '\r' {
                    self.comment = Some(CommentState::Line);
                }
            }
            Some(CommentState::Block { prev }) => {
                if prev == '*' && c == '/' {
                    // Closed (the `/*/` quirk included: prev is seeded
                    // with the opener's `*`).
                } else {
                    self.comment = Some(CommentState::Block { prev: c });
                }
            }
            Some(CommentState::Slash) => match c {
                '/' => self.comment = Some(CommentState::Line),
                '*' => self.comment = Some(CommentState::Block { prev: '*' }),
                // A standalone `/` is one skipped char (the engine's
                // own no-loop rule).
                _ => {}
            },
            None => {}
        }
    }

    /// The `#`-comment's context-dependent terminator set, computed where
    /// the comment starts (the engine's `ctx_has` rules).
    fn hash_comment(&self) -> CommentState {
        let arr = self.frames.iter().any(|f| matches!(f.kind, FrameKind::Arr));
        let obj_key = self.pos == Pos::Key || self.pos == Pos::Colon;
        let obj_value = self.pos == Pos::Value
            && matches!(self.frames.last().map(|f| &f.kind), Some(FrameKind::Obj));
        CommentState::Hash {
            arr,
            obj_value,
            obj_key,
        }
    }

    /// The member/item bookkeeping, at a KEY's or ITEM's start:
    /// `member_start` lands BEFORE the deferred separator (so the
    /// dangling-member drop and the heal's empty-element retract both
    /// take the separator with it), for every frame kind — object keys,
    /// array items and tuple items alike.
    fn begin_common(&mut self) {
        if let Some(f) = self.frames.last_mut() {
            f.member_begun = true;
            f.member_start = Some(self.out.len());
            if f.member_count > 0 {
                self.out.push_str(", ");
            }
            f.member_count += 1;
            if let Some(p) = f.paren.as_mut() {
                p.elems += 1;
            }
        }
    }

    /// Retract the in-flight member/item whole: the rendering since
    /// `member_start` (the deferred separator included) truncates back,
    /// the member count and the tuple's element count restore. The
    /// mid-stream form of the heal's empty-element retract.
    fn retract_in_flight_member(&mut self) {
        if let Some(f) = self.frames.last_mut()
            && let Some(start) = f.member_start.take()
        {
            if start <= self.out.len() {
                self.out.truncate(start);
                self.emitted = self.emitted.min(start);
            }
            f.member_count = f.member_count.saturating_sub(1);
            if let Some(p) = f.paren.as_mut() {
                p.elems = p.elems.saturating_sub(1);
            }
        }
    }

    /// The innermost frame's body saw a `:` where a key was due: the
    /// engine's classifier keeps such an object (see `Frame::body_colon`).
    fn mark_body_colon(&mut self) {
        if let Some(f) = self.frames.last_mut() {
            f.body_colon = true;
        }
    }

    /// A `,` seen at a separator position: inside a paren frame it is
    /// tuple evidence (the grouping-vs-tuple decision at `)`).
    fn note_comma(&mut self) {
        if let Some(f) = self.frames.last_mut()
            && let Some(p) = f.paren.as_mut()
        {
            p.comma_seen = true;
        }
    }

    fn open_string(&mut self, delim: char, is_key: bool) {
        // A non-strict delimiter (single or curly quote) is a repair
        // lane: json.loads fails on the whole input.
        if delim != '"' {
            self.strict_clean = false;
        }
        self.last_string_start = Some(self.out.len());
        self.string_raw_tail = None;
        self.out.push('"');
        if is_key {
            // A key string is in key position whatever the position it
            // started from (the AfterValue missing-comma path opens keys
            // too); the dangling-member check tests this state.
            self.pos = Pos::Key;
        }
        self.string = Some(StrState {
            delim,
            is_key,
            esc: Esc::None,
            high: None,
        });
    }

    fn open_container(&mut self, kind: FrameKind) -> Result<(), String> {
        if self.frames.len() >= MAX_NESTING {
            return Err("Input nesting exceeds the supported parser recursion depth.".into());
        }
        // The parenthesized lane is the repair lane (json.loads has no
        // tuple grammar).
        if kind == FrameKind::Paren {
            self.strict_clean = false;
        }
        let brace_pos = self.out.len();
        let paren = match kind {
            FrameKind::Obj => {
                self.out.push('{');
                self.pos = Pos::Key;
                None
            }
            FrameKind::Arr => {
                self.out.push('[');
                self.pos = Pos::Value;
                None
            }
            FrameKind::Paren => {
                self.out.push('[');
                self.pos = Pos::Value;
                Some(ParenFrame {
                    brace_pos,
                    elems: 0,
                    comma_seen: false,
                    elem_groupish: false,
                })
            }
        };
        self.frames.push(Frame {
            kind,
            brace_pos,
            body_chars: 0,
            body_nonspace: 0,
            member_begun: false,
            body_colon: false,
            member_count: 0,
            member_start: None,
            paren,
        });
        Ok(())
    }

    /// The dangling-member drop: truncate the in-flight member's rendering
    /// (the deferred separator included) and restore the pre-member state.
    fn drop_dangling_member(&mut self) {
        if let Some(f) = self.frames.last_mut()
            && let Some(start) = f.member_start.take()
        {
            if start <= self.out.len() {
                self.out.truncate(start);
                self.emitted = self.emitted.min(start);
            }
            f.member_count = f.member_count.saturating_sub(1);
            self.pos = Pos::Key;
        }
        // The engine's continuation repair lane ran (json.loads never
        // leaves a dangling key).
        self.strict_clean = false;
    }

    /// Close the deepest frame this closer matches, closing the levels
    /// above it first (the engine's `{"a": [1, 2}` shape). Two closer
    /// shapes belong to the parenthesized lane instead (parse_array runs
    /// the paren's group with `)` as its closing delimiter):
    ///
    /// - a `]` while a paren is the innermost frame is a garbage ITEM
    ///   inside the group's array loop (the strictly-empty skip consumes
    ///   it): nothing closes -- the group stays open and keeps parsing;
    /// - a `}` while a paren is the innermost frame TERMINATES the
    ///   group's loop and is consumed by it: the paren closes as if its
    ///   own `)` had arrived (and the strictly-empty decision defers to
    ///   the next char, the group's own item skip reading the char after
    ///   the consumed `}`).
    fn close_up(&mut self, closer: char) {
        if matches!(self.frames.last().map(|f| f.kind), Some(FrameKind::Paren)) {
            if closer == ']' {
                // A garbage item the strictly-empty skip consumes: the
                // group stays open and keeps parsing (a repair lane).
                self.strict_clean = false;
                return;
            }
            if closer == '}' {
                self.strict_clean = false;
                self.pop_frame();
                return;
            }
        }
        let Some(idx) = self.frames.iter().rposition(|f| f.kind.matches(closer)) else {
            // A closer matching nothing is skipped as garbage.
            self.strict_clean = false;
            return;
        };
        if self.frames.len() > idx + 1 {
            // The mismatched-closer close-up: levels between close
            // (the engine's whole-text re-decision shape).
            self.strict_clean = false;
        }
        while self.frames.len() > idx + 1 {
            self.pop_frame();
        }
        self.pop_frame();
    }

    /// Pop the innermost frame, emitting its proper closer (a tuple-
    /// vs-grouping paren included), and land in the parent's
    /// after-value position (or finish the document). A strictly-empty
    /// element's rendering (`[]`/`{}`/`""`) completing inside an
    /// array/tuple frame arms the one-char skip decision
    /// (`EmptyPending`) -- the engine's item skip reads the NEXT char.
    fn pop_frame(&mut self) {
        let Some(mut frame) = self.frames.pop() else {
            return;
        };
        // The frame's own closing char is not part of its body (the
        // engine's span ends before the suffix-strip): the counters
        // gave it back here.
        frame.body_chars = frame.body_chars.saturating_sub(1);
        frame.body_nonspace = frame.body_nonspace.saturating_sub(1);
        let mut paren_groupish = false;
        match frame.kind {
            FrameKind::Obj => {
                // The engine's empty-object array fallback (an object
                // that parsed empty over a non-trivial body): a
                // garbage-bearing colon-free body re-parses as an array
                // -- its observable for a body that recovers nothing is
                // `[]`. A colon-bearing body is object-style evidence
                // and keeps `{}` (`{: 1}`).
                if !frame.body_colon
                    && !frame.member_begun
                    && frame.body_chars >= 1
                    && frame.body_nonspace >= 1
                {
                    self.strict_clean = false;
                    let pos = frame.brace_pos;
                    if pos < self.out.len() && self.out.as_bytes()[pos] == b'{' {
                        self.out.drain(pos..pos + 1);
                        self.emitted = self.emitted.min(pos);
                        for m in &mut self.empty_marks {
                            if m.start > pos {
                                m.start -= 1;
                                m.end -= 1;
                            }
                        }
                        if let Some(p) = &mut self.empty_pending
                            && p.start > pos
                        {
                            p.start -= 1;
                            p.end -= 1;
                        }
                    }
                    self.out.push_str("[]");
                } else {
                    self.out.push('}');
                }
            }
            FrameKind::Arr => self.out.push(']'),
            FrameKind::Paren => {
                // The grouping decision reads the engine's unwrap guard
                // (the kept-items length): a strictly-empty single
                // element still under the item-skip's one-char decision
                // (`empty_pending` armed with this paren as its parent)
                // cannot count as the grouping's one value -- the
                // engine's skip may drop it and the group closes empty
                // (`([...)` -> `[]`).
                let grouped = frame.paren.as_ref().is_some_and(|p| {
                    p.comma_seen
                        || p.elems != 1
                        || self
                            .empty_pending
                            .as_ref()
                            .is_some_and(|pend| pend.parent_paren)
                });
                let mut groupish = false;
                if grouped {
                    self.out.push(']');
                    // The EMPTY tuple's render: the engine's re-dispatch
                    // of the group's inner text recovers nothing once a
                    // token follows (`()x` -> ``).
                    groupish = self.out.get(frame.brace_pos..) == Some("[]");
                } else {
                    // A grouping paren unwraps: drop the `[` we emitted.
                    if let Some(p) = frame.paren {
                        let pos = p.brace_pos;
                        if pos < self.out.len() && self.out.as_bytes()[pos] == b'[' {
                            self.out.drain(pos..pos + 1);
                            self.emitted = self.emitted.min(pos);
                            // The deferred empty-element marks live in
                            // the same buffer: every span at/after the
                            // removed bracket shifts left with it (a
                            // span the bracket sat inside cannot exist —
                            // a marked element's rendering is exactly
                            // `[]`/`{}`/`""`, bracket-free).
                            for m in &mut self.empty_marks {
                                if m.start > pos {
                                    m.start -= 1;
                                    m.end -= 1;
                                }
                            }
                            if let Some(p) = &mut self.empty_pending
                                && p.start > pos
                            {
                                p.start -= 1;
                                p.end -= 1;
                            }
                        }
                        // The unwrapped element's own shape decides the
                        // groupish-ness: a scalar/string render (`(1)`,
                        // `("hi")`), or an element that was itself a
                        // grouping (`(())`). A real container's render
                        // stands (`([1])`, `([])`).
                        let scalar_led = self
                            .out
                            .as_bytes()
                            .get(pos)
                            .is_none_or(|b| !matches!(b, b'[' | b'{'));
                        groupish = scalar_led || p.elem_groupish;
                    }
                }
                if groupish && let Some(pp) = self.frames.last_mut().and_then(|f| f.paren.as_mut())
                {
                    pp.elem_groupish = true;
                }
                paren_groupish = groupish;
            }
        }
        self.pos = Pos::AfterValue;
        self.top_done = self.frames.is_empty();
        // The pending's parent frame is gone: its one-char skip
        // decision's coordinates died with the pop (the engine's
        // close-up re-decides the emptied item's shape through its own
        // cascade; the machine's deferred marks do that work) -- a
        // stale decision must not fire into the finished rendering
        // (`[([)$` shaped: the `)` cascade pops the emptied inner
        // array and then its own paren parent).
        if self
            .empty_pending
            .as_ref()
            .is_some_and(|p| p.frame >= self.frames.len())
        {
            self.empty_pending = None;
        }
        if self.top_done {
            // The stack drained: any strictly-empty skip decision armed
            // by an intermediate pop of the cascade points into the
            // finished render with its parent frame gone -- the
            // document's own gates govern now (the trailing-token drop,
            // the grouping's tail check). Letting it fire would truncate
            // the closed render at stale coordinates (a broken
            // `end()`).
            self.empty_pending = None;
            if paren_groupish {
                // A root GROUPING-ish group closed: the engine's gate
                // also reads what follows (only whitespace before the
                // next line end is a standalone value) -- see
                // `ParenGate::TailCheck`. A GROUPED root tuple with real
                // content arms nothing: the engine keeps its render and
                // drops what follows (`([1])x` -> `[1]`,
                // `([1],[2])x` -> `[[1], [2]]`).
                self.paren_gate = Some(ParenGate::TailCheck);
            }
        }
        self.arm_empty_pending();
    }

    /// Arm the strictly-empty element's one-char skip decision when the
    /// just-completed element's rendering (the parent frame's
    /// `member_start` to here) is exactly `[]`/`{}`/`""` and the parent
    /// is an array-lane frame holding it as an in-flight member -- the
    /// engine's item skip reads the NEXT char. An object-lane parent
    /// keeps its member value (the skip is parse_array's rule alone); a
    /// root frame has no parent to decide about.
    fn arm_empty_pending(&mut self) {
        if let Some(parent) = self.frames.last()
            && !matches!(parent.kind, FrameKind::Obj)
            && let Some(start) = parent.member_start
            && start <= self.out.len()
        {
            let raw = &self.out[start..];
            let content = raw.strip_prefix(", ").unwrap_or(raw);
            if matches!(content, "[]" | "{}" | "\"\"") {
                self.empty_pending = Some(EmptyPending {
                    start,
                    end: self.out.len(),
                    was_first: parent.member_count == 1,
                    frame: self.frames.len() - 1,
                    parent_paren: parent.kind == FrameKind::Paren,
                });
            }
        }
    }

    /// The pending decision's drop: the member rendering at `start` (the
    /// deferred separator included) retracts whole and the parent
    /// frame's bookkeeping restores (the completed-member form of
    /// `retract_in_flight_member`).
    fn drop_marked_member(&mut self, start: usize) {
        if let Some(f) = self.frames.last_mut() {
            if start <= self.out.len() {
                self.out.truncate(start);
                self.emitted = self.emitted.min(start);
            }
            f.member_start = None;
            f.member_count = f.member_count.saturating_sub(1);
            if let Some(p) = &mut f.paren {
                p.elems = p.elems.saturating_sub(1);
            }
        }
    }

    /// The number run's terminator: the engine's value lanes over the
    /// (underscore-filtered) run text; a key-position number renders as
    /// the quoted run text (`{12: 1}` -> `{"12": 1}`). `raw_last` is the
    /// run's own last raw char (the popped rollback char when one
    /// popped, else the run's tail): the engine's `get(-1)` look at the
    /// same cursor.
    fn finish_number(&mut self, text: &str, key: bool, raw_last: Option<char>) {
        // A number KEY renders quoted (never strict); any run that is
        // not one strict number token (`1_0`, `1,000`, `01`, `1.`, the
        // empty rollback run) is the repair lane too.
        if key || loads_strict(text).is_err() {
            self.strict_clean = false;
        }
        if key {
            // The string-lane entry skip rides the key render too: the
            // run's leading non-alphanumeric garbage drops (`{-12: 1}`
            // -> `{"12": 1}`, `{.5: 1}` -> `{"5": 1}`). An attempt that
            // strips to NOTHING is the engine's empty missing-quote key
            // attempt: the pair drops whole (`{-: 1` -> `{}`).
            let filtered: String = text
                .chars()
                .filter(|&c| c != '_')
                .collect::<String>()
                .trim_start_matches(|c: char| !c.is_alphanumeric())
                .to_string();
            if filtered.is_empty() {
                self.drop_dangling_member();
                return;
            }
            self.last_key_empty = false;
            let rendered = serializer::dumps(&Value::Str(filtered), self.ensure_ascii);
            self.out.push_str(&rendered);
            self.pos = Pos::Colon;
            return;
        }
        let value = render_number_value(text);
        // The stray-`...` drop: the run's value lanes made the exact
        // string `...` and the parse ended on the final dot — an array
        // item position ignores it whole (the engine's array lane; the
        // element's slot and separator retract with it).
        if is_stray_ellipsis(&value, raw_last, self.frames.last().map(|f| f.kind)) {
            self.retract_in_flight_member();
            self.pos = Pos::AfterValue;
            return;
        }
        let rendered = serializer::dumps(&value, self.ensure_ascii);
        self.out.push_str(&rendered);
        self.pos = Pos::AfterValue;
    }

    /// The close-time render: the pending token resolved, the open string
    /// quoted shut, the open containers closed. Pure (`snapshot`).
    fn render_closed(&self) -> String {
        let mut s = self.out.clone();
        // The top-level `(`'s gate still undecided at the close (the
        // input ended inside the first inner token): the engine's
        // can_start_value read the lowercased inner-text prefix -- a
        // literal prefix commits the tuple's single word (the grouping
        // unwrap's observable); anything else was prose to it.
        if let Some(ParenGate::FirstChar { buf }) = &self.paren_gate {
            let lower = buf.to_lowercase();
            if lower.starts_with("true")
                || lower.starts_with("null")
                || lower.starts_with("none")
                || lower.starts_with("false")
            {
                let value = keyword_or_str(buf);
                s.push_str(&serializer::dumps(&value, self.ensure_ascii));
            }
        }
        // A dangling object member (a key pending, a key emitted and
        // awaiting its colon, or a key string still open) drops whole
        // (the engine's continuation shape: `{"a": "x", "b"` ->
        // `{"a": "x"}`); when the member's key string is still open its
        // close-time render is suppressed with it.
        let mut suppress_open_string = false;
        // The pending number's rollback-reprocess appended a word
        // element the machine's frames never counted (render_closed is
        // pure): the cascade's bookkeeping learns about it here.
        let mut appended_word_frame: Option<usize> = None;
        let dangling = match &self.pending {
            Some(Pending::Word { key: true, .. }) | Some(Pending::Num { key: true, .. }) => {
                self.frames.last().and_then(|f| f.member_start)
            }
            None if self.pos == Pos::Colon => self.frames.last().and_then(|f| f.member_start),
            None if self.pos == Pos::Key => {
                let key_open = self.string.as_ref().is_some_and(|st| st.is_key);
                if key_open {
                    suppress_open_string = true;
                    self.frames.last().and_then(|f| f.member_start)
                } else {
                    None
                }
            }
            _ => None,
        };
        let mut dangling_dropped = false;
        if let Some(start) = dangling
            && start <= s.len()
        {
            s.truncate(start);
            dangling_dropped = true;
        }
        match &self.pending {
            Some(Pending::Num {
                text, key: false, ..
            }) => {
                let mut run = text.clone();
                let popped = pop_trailing_number_char(&mut run);
                // An empty run at an item position renders nothing: the
                // element retracts in the cascade below (the engine's
                // `[1, -` -> `[1]` heal); an object member value still
                // heals to `""` (`{"a": -` -> `{"a": ""}`). The same
                // lane's stray-`...` drop: a run whose value is the
                // exact string `...` and whose last raw char is the
                // final dot retracts at an item position too (the
                // engine's array lane; the cascade takes the separator).
                let array_like = self
                    .frames
                    .last()
                    .is_some_and(|f| !matches!(f.kind, FrameKind::Obj));
                let value = render_number_value(&run);
                let stray = array_like
                    && is_stray_ellipsis(
                        &value,
                        popped.or_else(|| run.chars().next_back()),
                        self.frames.last().map(|f| f.kind),
                    );
                if !(run.is_empty() && array_like) && !stray {
                    s.push_str(&serializer::dumps(&value, self.ensure_ascii));
                    // The rollback's reprocess (the mid-stream path's
                    // own rule): a popped ALPHABETIC char at an item
                    // position was a word's first char — the engine's
                    // falsy-nudge reparse makes it the next element
                    // (`[1E` -> `[1, "E"]`); in an object the popped
                    // char's key-attempt drops (`{"a": 1E` ->
                    // `{"a": 1}`), and a non-alphabetic pop renders
                    // nothing (`[1-` -> `[1]`).
                    if array_like
                        && let Some(p) = popped
                        && !run.is_empty()
                        && p.is_alphabetic()
                    {
                        s.push_str(", ");
                        let value = literal_or_str(&p.to_string());
                        s.push_str(&serializer::dumps(&value, self.ensure_ascii));
                        appended_word_frame = Some(self.frames.len() - 1);
                    }
                }
            }
            Some(Pending::Word {
                text,
                key: false,
                from_number_run,
            }) => {
                // The engine's repair-lane observable for the in-flight
                // word at the cut: the literal table only — the special
                // floats' spellings heal to STRINGS (the strict fast
                // path keeps them mid-stream: `[NaN]` streams to
                // `[NaN]`, the cut `[NaN` heals to `["NaN"]`), and a
                // word born from a number run drops its stray leading
                // sign (`{"a": -Infinity` -> `{"a": "Infinity"}`).
                let word = text.trim();
                let value = match from_number_run {
                    // The rewind lane's string-entry skip: the run's
                    // leading non-alphanumeric chars drop before the
                    // literal table (`[-.t` -> `["t"]`), the close-time
                    // spelling of `render_word`'s mid-stream rule. The
                    // literal table stays float-free (the cut's own
                    // observable: `[NaN` heals to `["NaN"]`).
                    true => literal_or_str(word.trim_start_matches(|c: char| !c.is_alphanumeric())),
                    false => literal_or_str(word),
                };
                s.push_str(&serializer::dumps(&value, self.ensure_ascii));
            }
            // Provisional top-level scalars: committed only when the held
            // text is one strict JSON value (the engine's own partial
            // semantics: `tru` was prose, `12` a number); an open
            // top-level string was never recoverable (`"hi` -> `""`).
            Some(Pending::TopNum { text, .. }) | Some(Pending::TopWord { text, .. }) => {
                if let Ok(v) = loads_strict(text) {
                    s.push_str(&serializer::dumps(&v, self.ensure_ascii));
                }
            }
            // A completed top-level string commits only from the strict
            // `"` delimiter (the engine's observable: `"hi"` -> `"hi"`,
            // but `'hi'` — repaired mid-stream, quoted and valid in the
            // deltas and the snapshot — was prose to the whole-text
            // engine, `''`); the top-level empty string IS the
            // nothing-recoverable sentinel's spelling: the engine
            // renders it bare.
            Some(Pending::TopStr {
                buf,
                st,
                done: true,
            }) if buf != "\"\"" && st.delim == '"' => {
                s.push_str(buf);
            }
            _ => {}
        }
        // A member whose value never arrived heals the value first (the
        // same heal the structural closer path applies: `{"a":` ->
        // `{"a": ""}`), unless a value is already in flight.
        let value_in_flight = self.pending.is_some() || self.string.is_some();
        if self.pos == Pos::MemberValue && !value_in_flight && !self.frames.is_empty() {
            s.push_str("\"\"");
        }
        if let Some(st) = &self.string
            && !suppress_open_string
        {
            st.render_tail(&mut s, self.ensure_ascii);
            s.push('"');
        }
        // The escape-tail heal: a string rendering that ends the stream
        // with a trailing newline-run loses it (the engine's own heal:
        // `{"k": "a\n` -> `{"k": "a"}`, `{"k": "\n"` -> `{"k": ""}`),
        // where a tab or a mid-string newline passes. The strip runs on
        // the closed rendering (the string quoted shut above), so a
        // consumed close quote goes straight back; the cascade may still
        // retract the member whole.
        if !self.frames.is_empty()
            && let Some(qs) = self.last_string_start
            && strip_trailing_newline_tail(&mut s, qs + 1, self.string.is_some())
        {
            s.push('"');
        }
        // The engine's stray-`...` drop (array.rs's parse_array_items:
        // "the stray '...' is ignored"): an ARRAY item whose parse is the
        // exact string `...` with the parse ending on a `.` is skipped
        // whole. At a cut the open string IS that parse: the rendered
        // element spells `"..."` and the last raw char is the final dot
        // (the engine's own two conditions -- the parsed value and the
        // `get(-1)` cursor look). The parent's deferred-separator cascade
        // below then retracts the element; a CLOSED `"..."` element does
        // not qualify (its parse ended on the quote, not a dot).
        let stray_ellipsis = self.string.is_some()
            && self
                .frames
                .last()
                .is_some_and(|f| f.member_start.is_some())
            && self
                .last_string_start
                .is_some_and(|qs| s.get(qs..).is_some_and(|rendered| rendered == "\"...\""))
            // The rendered element IS the engine's parsed value here: the
            // same classification (the raw `get(-1)` look, the array
            // lane's frame) decides.
            && is_stray_ellipsis(
                &Value::Str("...".into()),
                self.string_raw_tail,
                self.frames.last().map(|f| f.kind),
            );
        // The empty-container cascade (the engine's truncated-output
        // heal): a still-open container with no members drops whole when
        // it sits at an item position (`[[` -> `[]`, `[1, [` -> `[1]`)
        // and closes as `[]`/`{}` at a member-value position or the root
        // (`{"a": [` -> `{"a": []}`); an array's trailing strictly-empty
        // member (closed or just rendered: `[]`, `{}`, `""`) drops too
        // (`[[], []` -> `[[]]`, `[1, []` -> `[1]`), as does the trailing
        // stray-`...` string element the engine's array lane ignores.
        // The walk is local: snapshot purity (the machine's own state
        // never moves).
        let mut info: Vec<FrameInfo> = self
            .frames
            .iter()
            .map(|f| FrameInfo {
                kind: f.kind,
                brace_pos: f.brace_pos,
                body_chars: f.body_chars,
                body_nonspace: f.body_nonspace,
                member_begun: f.member_begun,
                body_colon: f.body_colon,
                member_count: f.member_count,
                member_start: f.member_start,
                paren: f.paren,
            })
            .collect();
        if let Some(idx) = appended_word_frame
            && let Some(f) = info.get_mut(idx)
        {
            f.member_count += 1;
            if let Some(p) = &mut f.paren {
                p.elems += 1;
            }
        }
        let mut i = info.len();
        if dangling_dropped && i > 0 {
            // the dropped member's slot goes with it: the cascade may
            // now walk on through the emptied frame
            info[i - 1].member_count = info[i - 1].member_count.saturating_sub(1);
            info[i - 1].member_start = None;
        }
        // The deferred strictly-empty element drops (the engine's array
        // skip; see `empty_marks`): the marked renderings drain here,
        // newest first (later spans sit at higher offsets). The gate is
        // the ENGINE'S OWN DECISION PROCEDURE: the drops happen exactly
        // when the completed text is NOT one strict JSON value --
        // `json.loads` failed, so the repair lane decided, and its
        // strictly-empty skips fire. A root that closed cleanly AND
        // strict-clean (`strict_clean`: no repair lane fired, no
        // garbage consumed, no malformed structure seen) took the
        // whole-input fast path, which keeps every element -- the drops
        // are cut-only there. The one-char decision still armed
        // (`empty_pending`: the element was the input's last token)
        // joins the marks as the newest span. The demotion: a dropped
        // FIRST element promotes the next one to the frame's first
        // slot, and its deferred separator goes with the drop.
        if !self.top_done || !self.strict_clean {
            let pending_mark = self.empty_pending.as_ref().map(|p| EmptyMark {
                start: p.start,
                end: p.end,
                was_first: p.was_first,
                frame: p.frame,
            });
            for mark in pending_mark.iter().chain(self.empty_marks.iter().rev()) {
                let (start, end) = (mark.start, mark.end);
                if end > s.len() || start >= end {
                    continue;
                }
                // The marked frame's own bookkeeping first, on the
                // ORIGINAL coordinates: the marked element was the
                // frame's last member exactly when the member_start
                // still points into the span.
                if let Some(f) = info.get_mut(mark.frame) {
                    f.member_count = f.member_count.saturating_sub(1);
                    if f.member_start.is_some_and(|ms| ms >= start && ms < end) {
                        f.member_start = None;
                    }
                    if let Some(p) = &mut f.paren {
                        p.elems = p.elems.saturating_sub(1);
                    }
                }
                // The span's drain: every position at/after its end
                // shifts left by the span's length.
                s.drain(start..end);
                let span = end - start;
                for f in info.iter_mut() {
                    if f.member_start.is_some_and(|ms| ms >= end) {
                        f.member_start = f.member_start.map(|ms| ms - span);
                    }
                    if f.brace_pos >= end {
                        f.brace_pos -= span;
                    }
                    if let Some(p) = &mut f.paren
                        && p.brace_pos >= end
                    {
                        p.brace_pos -= span;
                    }
                }
                // The demotion: a dropped FIRST element promotes the
                // next one to the frame's first slot, and its deferred
                // separator (the two chars now at the slot) goes too.
                if mark.was_first && s[start..].starts_with(", ") {
                    s.drain(start..start + 2);
                    for f in info.iter_mut() {
                        if f.member_start.is_some_and(|ms| ms >= start + 2) {
                            f.member_start = f.member_start.map(|ms| ms - 2);
                        }
                        if f.brace_pos >= start + 2 {
                            f.brace_pos -= 2;
                        }
                        if let Some(p) = &mut f.paren
                            && p.brace_pos >= start + 2
                        {
                            p.brace_pos -= 2;
                        }
                    }
                }
            }
        }
        // The retraction floor: the lowest truncate-point of the walk.
        // A frame's brace_pos at/above it pointed into rendering that
        // no longer exists -- its drain guards must not fire.
        let mut floor = usize::MAX;
        while i > 0 {
            let (kind, mc, ms) = {
                let f = &info[i - 1];
                (f.kind, f.member_count, f.member_start)
            };
            if mc == 0 {
                if i == 1 || info[i - 2].kind == FrameKind::Obj {
                    // The root, or a member value: keep, close it below.
                    break;
                }
                // An item-position container: retract the rendering
                // (the parent's deferred separator included) and keep
                // walking: the parent may itself have become empty.
                if let Some(start) = info[i - 2].member_start
                    && start <= s.len()
                {
                    s.truncate(start);
                    floor = floor.min(start);
                }
                info[i - 2].member_count = info[i - 2].member_count.saturating_sub(1);
                info[i - 2].member_start = None;
                if let Some(p) = &mut info[i - 2].paren {
                    p.elems = p.elems.saturating_sub(1);
                }
                i -= 1;
                continue;
            }
            if kind == FrameKind::Obj || ms.is_none() {
                break;
            }
            let start = ms.expect("checked above");
            if start > s.len() {
                break;
            }
            let raw = &s[start..];
            let content = raw.strip_prefix(", ").unwrap_or(raw);
            let separator_pending = self.pos == Pos::Value && !value_in_flight;
            if !separator_pending
                && (content.is_empty()
                    || content == "\"\""
                    || content == "[]"
                    || content == "{}"
                    || (stray_ellipsis && content == "\"...\""))
            {
                s.truncate(start);
                floor = floor.min(start);
                info[i - 1].member_count = info[i - 1].member_count.saturating_sub(1);
                info[i - 1].member_start = None;
                // the tuple's element count goes with the member (an
                // emptied paren must close as an array, never take the
                // grouping unwrap)
                if let Some(p) = &mut info[i - 1].paren {
                    p.elems = p.elems.saturating_sub(1);
                }
                continue;
            }
            break;
        }
        // Only the frames the cascade kept still close (the retracted
        // containers' renderings are gone with their brackets), and the
        // paren decision reads the cascade-adjusted element counts.
        for (idx, frame) in self.frames.iter().enumerate().take(i).rev() {
            match frame.kind {
                FrameKind::Obj => {
                    // The pop's body rule, at the cascade's close (see
                    // pop_frame): the engine's empty-object fallback —
                    // the CUT's span ends at the input (no closer was
                    // consumed), one shorter than the complete
                    // object's, hence the 2 here against the pop's 1.
                    if !info[idx].body_colon
                        && !info[idx].member_begun
                        && info[idx].body_chars >= 2
                        && info[idx].body_nonspace >= 1
                    {
                        let pos = info[idx].brace_pos;
                        if pos < floor && pos < s.len() && s.as_bytes()[pos] == b'{' {
                            s.drain(pos..pos + 1);
                        }
                        s.push_str("[]");
                    } else {
                        s.push('}');
                    }
                }
                FrameKind::Arr => s.push(']'),
                FrameKind::Paren => match info[idx].paren {
                    Some(p) if p.comma_seen || p.elems != 1 => s.push(']'),
                    Some(p) => {
                        // An unclosed single-element grouping unwraps.
                        if p.brace_pos < s.len() && s.as_bytes()[p.brace_pos] == b'[' {
                            s.drain(p.brace_pos..p.brace_pos + 1);
                        } else {
                            s.push(']');
                        }
                    }
                    None => s.push(']'),
                },
            }
        }
        s
    }
}

/// parse_number's trailing-rollback: one trailing char that cannot end a
/// number (`-`, `e`, `E`, `/`, `,`, `+`) pops back off; the caller
/// reprocesses it structurally.
fn pop_trailing_number_char(run: &mut String) -> Option<char> {
    match run.chars().last() {
        Some('-') | Some('e') | Some('E') | Some('/') | Some(',') | Some('+') => run.pop(),
        _ => None,
    }
}

/// The escape-tail heal's scan: does the text end (past an optional close
/// quote) with the tail the engine's heal strips, and if so truncate it
/// and return true (the caller re-appends the consumed close quote).
/// `lower` bounds the scan (the string rendering's content start: the
/// opening quote and everything before it stay). The two cases the
/// engine's own heal splits:
///
/// - an UNTERMINATED string (`open_ended`, the string still open at the
///   cut): the decoded content RSTRIPS — Python's whitespace set in its
///   rendered forms (a raw space, the `\t`/`\n`/`\r`/`\f` escapes, the
///   6-byte `\u000b`), backspace included in nothing (`{"k": "a b ` ->
///   `{"k": "a b"}`, `{"k": "  ` -> `{"k": ""}`),
/// - a CLOSED string: only a trailing NEWLINE-run (a `\n` required, a
///   `\r` only directly before a `\n`) plus the plain spaces that
///   preceded it, the run adjacent to the close (`{"k": "a\n"` ->
///   `{"k": "a"}`, `{"k": "a \n"` -> `{"k": "a"}`, while `{"k": "a\n "`
///   and `{"k": "a\r"` and `{"k": "a\n\t"` keep their content).
///
/// An escaped backslash before a 2-byte escape blocks it (a literal
/// backslash-n content, `{"k": "a\\n"`, is not a newline at all).
fn strip_trailing_newline_tail(s: &mut String, lower: usize, open_ended: bool) -> bool {
    let b = s.as_bytes();
    let mut i = s.len();
    let mut quote = false;
    if i > lower && b[i - 1] == b'"' {
        i -= 1;
        quote = true;
    }
    let mut any = false;
    if open_ended {
        loop {
            if i > lower && b[i - 1] == b' ' {
                i -= 1;
                any = true;
            } else if i >= lower + 2
                && b[i - 2] == b'\\'
                && matches!(b[i - 1], b't' | b'n' | b'r' | b'f')
                && (i < 3 || b[i - 3] != b'\\')
            {
                i -= 2;
                any = true;
            } else if i >= lower + 6
                && b[i - 6] == b'\\'
                && b[i - 5] == b'u'
                && (i < 7 || b[i - 7] != b'\\')
            {
                // the 6-byte escape's decoded char is Python
                // whitespace: `\u000b`, and any other `\uXXXX` whose
                // code point isspace()s (the raw `\u2000` en quad the
                // content carried) — the engine's rstrip is Python's
                // own whitespace set
                let hex = std::str::from_utf8(&b[i - 4..i]).unwrap_or("");
                match u16::from_str_radix(hex, 16)
                    .ok()
                    .and_then(|cp| char::from_u32(u32::from(cp)))
                {
                    Some(ch) if is_py_whitespace(ch) => {
                        i -= 6;
                        any = true;
                    }
                    _ => break,
                }
            } else {
                break;
            }
        }
    } else {
        while i >= lower + 2
            && b[i - 2] == b'\\'
            && b[i - 1] == b'n'
            && (i < 3 || b[i - 3] != b'\\')
        {
            i -= 2;
            any = true;
            // a \r directly before the consumed \n joins the run
            if i >= lower + 2
                && b[i - 2] == b'\\'
                && b[i - 1] == b'r'
                && (i < 3 || b[i - 3] != b'\\')
            {
                i -= 2;
            }
            while i > lower && b[i - 1] == b' ' {
                i -= 1;
            }
        }
    }
    if any {
        s.truncate(i);
    }
    any && quote
}

/// The engine's stray-`...` classification (array.rs's parse_array_items,
/// upstream's "the stray '...' is ignored" branch): an ARRAY item whose
/// parse came back the exact string `...` and whose parse's last consumed
/// char was a `.` is skipped whole. The raw look (`self.get(-1)`) is what
/// separates a cut's unterminated string and a bare number-run from a
/// CLOSED `"..."` element (its parse ends on the quote) — the same value,
/// kept there. Object member values keep their `...`: the rule lives in
/// the array lane alone (parenthesized containers parse through it too).
fn is_stray_ellipsis(value: &Value, raw_last: Option<char>, frame: Option<FrameKind>) -> bool {
    raw_last == Some('.')
        && matches!(value, Value::Str(text) if text == "...")
        && frame.is_some_and(|k| !matches!(k, FrameKind::Obj))
}

/// parse_number's value lanes over the run text (underscores were
/// consumed but not kept: dropped here). An empty run (a lone `-` that
/// rolled back) is the empty-string value.
fn render_number_value(text: &str) -> Value {
    let filtered: String = text.chars().filter(|&c| c != '_').collect();
    if filtered.is_empty() {
        return Value::Str(String::new());
    }
    if filtered.contains(',') {
        return Value::Str(filtered);
    }
    if filtered.contains('.') || filtered.contains('e') || filtered.contains('E') {
        return match filtered.parse::<f64>() {
            Ok(f) => Value::Float(f),
            Err(_) => Value::Str(filtered),
        };
    }
    match filtered.parse::<i64>() {
        Ok(n) => Value::Int(n),
        Err(_) => match normalize_big_int_text(&filtered) {
            Some(t) => Value::BigInt(t),
            None => Value::Str(filtered),
        },
    }
}

/// A bare word's render: the keyword table, and — for a word born from
/// a number run — the engine's stray-sign drop (`{"a": -NaN}` ->
/// `{"a": "NaN"}` as a string, `{"a": -abc}` -> `{"a": "abc"}`), the
/// strict `-Infinity` spelling excepted (the number lane keeps it, a
/// special float). The word is complete here: the decision sees the
/// whole run.
fn render_word(word: &str, from_number_run: bool) -> Value {
    if from_number_run {
        // The one strict spelling the rewind can be riding keeps the
        // special-float table: the fast path may still win the whole
        // document (`{"a": -Infinity}` -> the float, Python's own
        // json.loads). Everything else takes the engine's rewind
        // re-entry: the STRING lane, whose entry skip consumes every
        // leading non-alphanumeric char the number run had gathered (a
        // stray sign, a dot run, an underscore) before its literal
        // check (`[.5x]` -> `["5x"]`, `[-.t]` -> `["t"]`, `[.true]` ->
        // `[true]`, the cut `{"a": -Infinity` -> `{"a": "Infinity"}` --
        // the rewind lane's literal table has no float branch).
        if word == "-Infinity" {
            return keyword_or_str(word);
        }
        let stripped = word.trim_start_matches(|c: char| !c.is_alphanumeric());
        return literal_or_str(stripped);
    }
    keyword_or_str(word)
}

/// The keyword set for bare words. The Python-literal family follows the
/// repair lane and matches CASE-INSENSITIVELY (the whole-text engine's
/// observable: `TRUE`/`tRuE`/`NONE` repair to `true`/`true`/`null`); the
/// strict grammar's special floats follow the fast path (it loads them,
/// so the whole-text observable keeps the floats) but only at their exact
/// spellings (`nan`/`INFINITY` stay strings at their own casing —
/// `{"a": NAN}` -> `{"a": "NAN"}` — the engine's own split).
fn keyword_or_str(word: &str) -> Value {
    match word.to_lowercase().as_str() {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        "null" | "none" => Value::Null,
        _ => match word {
            "NaN" => Value::Float(f64::NAN),
            "Infinity" => Value::Float(f64::INFINITY),
            "-Infinity" => Value::Float(f64::NEG_INFINITY),
            // the string spelling keeps the word's own casing
            _ => Value::Str(word.to_string()),
        },
    }
}

/// The literal-only variant for a word whose leading `-` was a stripped
/// stray sign: the case-insensitive literal table without the special
/// floats (the engine's word lane after the rewind: `{"a": -NaN}` ->
/// `{"a": "NaN"}` as a STRING, `{"a": -true}` -> `{"a": true}`).
fn literal_or_str(word: &str) -> Value {
    match word.to_lowercase().as_str() {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        "null" | "none" => Value::Null,
        _ => Value::Str(word.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn streamed(text: &str, chunk: usize) -> String {
        let mut r = StreamingRepairer::new(true);
        let chars: Vec<char> = text.chars().collect();
        for group in chars.chunks(chunk.max(1)) {
            let s: String = group.iter().collect();
            r.push(&s).unwrap();
        }
        r.end().unwrap()
    }

    fn streamed_by_splits(text: &str, splits: &[usize]) -> String {
        let mut r = StreamingRepairer::new(true);
        let mut prev = 0;
        for &p in splits {
            r.push(&text[prev..p]).unwrap();
            prev = p;
        }
        r.push(&text[prev..]).unwrap();
        r.end().unwrap()
    }

    #[test]
    fn heal_truncated_string_and_containers() {
        assert_eq!(streamed(r#"{"a": "hel"#, 3), r#"{"a": "hel"}"#);
        assert_eq!(streamed(r#"{"a": [1, 2"#, 3), r#"{"a": [1, 2]}"#);
        assert_eq!(streamed(r#"{"a":"#, 1), r#"{"a": ""}"#);
        assert_eq!(streamed("{", 1), "{}");
        assert_eq!(streamed("[", 1), "[]");
        assert_eq!(streamed(r#"{"a": [1,"#, 2), r#"{"a": [1]}"#);
    }

    #[test]
    fn the_strictly_empty_drops_gate_on_the_engines_own_decision() {
        // The drain's gate is the ENGINE'S OWN DECISION PROCEDURE: the
        // drops happen exactly when the completed text is NOT
        // strict-valid JSON (the repair lane ran); a strict-clean
        // parse that closed cleanly took the whole-input fast path and
        // keeps every element. Byte-exact against the engine for every
        // family.
        for text in [
            // the six families (the re-verdict round 3's list)
            "[(,)]",
            "[( , )]",
            "[1, (,)]",
            "[(,), 1]",
            "[(, ), 1]",
            "[[] ,]",
            "[{} ,]",
            "[\"\" ,]",
            "(,)",
            "( , )",
            "[{]",
            "[{]}",
            "{\"a\": {(]",
            "{\"a\": {]}}",
            "{]}",
            "{,}",
            // the strict-valid neighbors the gate must keep
            "[[], 1]",
            "[[] , 1]",
            "[true, [] , 1]",
            "[{}]",
            "[( )]",
            "[()]",
            "()",
            "[]",
            "{}",
            // the adjacent lanes the same rule re-decides
            "[(, 1)]",
            "[1, (,), 2]",
            "[(,), 1, 2]",
            "[(, ), (,)]",
            "{\"a\": [1, (,)]}",
            "[[,]]",
            "[[,] ,]",
            "[1,,2]",
            "[1,,]",
            "[,1]",
            "[,]",
            "(1,2)",
            "(1,000)",
            "(1) x",
            "(x)",
            "(tru)",
            "(t)",
            "(fals)",
            "(1,)",
            "(1, 2)",
            "{]",
            "{ ]}",
            "[() ]",
            "[() ,]",
            "[()]x",
            "[1, () ]",
        ] {
            let (value, _) = super::super::repair(text, &super::super::RepairConfig::default())
                .expect("engine repair succeeds");
            // The nothing-recoverable sentinel renders bare through the
            // repair_json surface (the machine's own spelling).
            let engine = match &value {
                Value::Str(empty) if empty.is_empty() => String::new(),
                _ => serializer::dumps(&value, true),
            };
            for chunk in [1, 2, 3, usize::MAX] {
                assert_eq!(
                    streamed(text, chunk),
                    engine,
                    "streamed({chunk}) != engine for {text:?}"
                );
            }
        }
    }

    #[test]
    fn the_whole_differential_corpus_agrees_with_the_engine() {
        for text in [
            r#"{"a": 1}"#,
            r#"{ "a" : 1 }"#,
            r#"{"a": "hel"#,
            r#"{"a": [1, 2"#,
            r#"{"a": tru"#,
            r#"{"a": [1, 2}"#,
            r#"{"a": 1 "b": 2}"#,
            "[1 2]",
            "{'a': 1}",
            "{a: 1}",
            r#"{"a": None}"#,
            r#"[True, False, None]"#,
            r#"{"a": 1,}"#,
            "[1, 2,]",
            r#"{"a": [1, }"#,
            r#"{"a": 1]"#,
            r#"[{"a": 1]"#,
            r#"{"a": "x", "b""#,
            r#"{"a": 1, b"#,
            "{a: 1, b}",
            "(1)",
            "(1,)",
            "(1, 2)",
            r#"{"a": (1, 2)}"#,
            r#"{"a": 1e}"#,
            r#"{"a": 1e+}"#,
            r#"{"a": -}"#,
            r#"{"a": 1,000}"#,
            r#"{"a": 12abc}"#,
            r#"{"a": "x\qy"}"#,
            r#"{"a": "\u41"}"#,
            r#"{"a": "x\u41"#,
            r#"{"a": "\ud83d\ude00x"}"#,
            r#"{"a": "\ud83dx"}"#,
            r#"{"a": "\ud83d"#,
            r#"{"a": "x\ny"}"#,
            "// lead
{\"a\": 1}",
            r#"/*x*/{"a": 1}"#,
            r#"Answer: {"a": 1}"#,
            r#"{"a": 1} junk"#,
            r#"{"a": 12}"#,
            r#"{"a": 1.5e3}"#,
            r#"{"a": 123456789012345678901234567890}"#,
            // the heal class: empty containers, the falsy-nudge
            // retractions, the escape tails, the missing-key colon
            "[[",
            "[1, [",
            r#"{"a": {"b": ["#,
            r#"{"a": ["#,
            "[[], [",
            "[[], []",
            "[[[]",
            r#"{"a": [["#,
            "[{}, [",
            r#"{"a": [{"b": {"#,
            r#"{"a": 1, "b": ["#,
            "(1, [",
            "([",
            r#"{"a": ("#,
            r#"{"a": [("#,
            "[1, \"",
            "[1, -",
            "[1,-]",
            "[1e-}",
            "[1e+-}",
            "[1e- 2]",
            "(\u{b}-\0",
            r#"{"k": "a\n"#,
            r#"{"k": "a\n""#,
            r#"{"k": "a\n\t"#,
            r#"{"k": "a\n ""#,
            r#"{"k": "a \n""#,
            r#"{"k": "\n""#,
            r#"{"k": "a\t""#,
            r#"{"k": "a\r""#,
            r#"{"k": "a ""#,
            r#"{"k": "a b "#,
            r#"{"k": "  "#,
            "[\"a\\n",
            r#"{"a": "x\n", "b": 1}"#,
            r#"{"a": "x\n""#,
            "{: 1}",
            "{ : 1}",
            "{:}",
            r#"{"": 1}"#,
            r#"{: 1, "b": 2}"#,
            // the keyword family: case-insensitive literals, the exact
            // special-float spellings, the stray-sign drop
            r#"{"a": TRUE}"#,
            r#"{"a": tRuE}"#,
            r#"{"a": FALSE}"#,
            r#"{"a": NULL}"#,
            r#"{"a": NONE}"#,
            "[TRUE, FALSE, NULL, NONE, tRuE]",
            r#"{"a": nan}"#,
            r#"{"a": NAN}"#,
            r#"{"a": NaN}"#,
            r#"{"a": Infinity}"#,
            r#"{"a": INFINITY}"#,
            r#"{"a": -Infinity}"#,
            r#"{"a": -infinity}"#,
            r#"{"a": -NaN}"#,
            r#"{"a": -TRUE}"#,
            r#"{"a": -tru}"#,
            r#"{"a": -abc}"#,
            r#"{"a": -12abc}"#,
            "[-NaN]",
            r#"{"TRUE": 1}"#,
            // the rewind lane's string-entry skip: the number run
            // terminated by an alphabetic char re-enters the string
            // lane, whose leading skip consumes the run's
            // non-alphanumeric garbage (a dot run, sign and dot, an
            // underscore) before the literal table
            "[.t]",
            "[..t]",
            "[...t]",
            "[....t]",
            "[...t x]",
            "[.x.y]",
            "[.5x]",
            "[.5t]",
            "[.5x, 1]",
            "[-.t]",
            "[-.5x]",
            "[-.Infinity]",
            "[._t]",
            "[.true]",
            "[.true, 1]",
            "[.false]",
            "[.none]",
            "[.null]",
            "[.NaN]",
            "[--t]",
            "[1, .t]",
            r#"{"a": .t}"#,
            r#"{"a": .false}"#,
            "[...]",
            "[....]",
            "[.e]",
            "[..e]",
            "[.1e]",
            "[-e]",
            // the object-key lane's string-lane semantics: the entry
            // skip's strip and the whitespace-touched invalidation
            // (the key attempt dies at whitespace unless the next char
            // is its colon)
            "{-ab: 1}",
            "{.ab: 1}",
            "{-.ab: 1}",
            "{-12: 1}",
            "{.5: 1}",
            "{-true: 1}",
            "{--ab: 1}",
            "{-eOe: 6}",
            "{12ab: 1}",
            "{a b: 1}",
            "{a  b: 1}",
            "{a b: 1, c: 2}",
            "{a b c: 1}",
            "{-ab x: 1}",
            "{-ab : 1}",
            "{: 1, : 2}",
            // the attempt that strips to nothing drops the pair
            "{-: 1",
            "{-.: 1}",
            "{-:}",
            "{-: , \"a\": 1}",
            // the member-boundary heals: the ','/closer where the
            // colon was due acts AS the colon; the key attempt's
            // string absorbs its own ','/']'
            "{a,b: 1}",
            "{a]: 1}",
            "{a ,: 1}",
            "{a b,c: 1}",
            "{-one,f: false}",
            "{\"a\" , 1}",
            // the open string's tail rstrip: the escaped
            // Python-whitespace code points join the escape-tail heal
            // (the raw `\u2000` en quad the content carried)
            "[\"\\u2000",
            "[\"\\u2000\\u2000",
            "{\"k\": \"\\u2000",
        ] {
            let (value, _) = super::super::repair(text, &super::super::RepairConfig::default())
                .expect("engine repair succeeds");
            let engine = serializer::dumps(&value, true);
            for chunk in [1, 2, 3, 7, usize::MAX] {
                assert_eq!(
                    streamed(text, chunk),
                    engine,
                    "streamed({chunk}) != engine for {text:?}"
                );
            }
        }
    }

    #[test]
    fn split_invariance_on_every_boundary() {
        let text = r#"{"k": [1, 2.5, "x", None], "done": tru}"#;
        let mut splits = Vec::new();
        for (i, _) in text.char_indices() {
            splits.push(i);
        }
        assert_eq!(
            streamed_by_splits(text, &splits),
            streamed(text, usize::MAX)
        );
    }

    #[test]
    fn snapshot_equals_end_and_is_pure() {
        let text = r#"{"a": [1, "hel"#;
        let mut r = StreamingRepairer::new(true);
        r.push(&text[..5]).unwrap();
        let s1 = r.snapshot();
        let s2 = r.snapshot();
        assert_eq!(s1, s2);
        r.push(&text[5..]).unwrap();
        assert_eq!(r.snapshot(), r.end().unwrap());
        // A second end is idempotent, and push after end refuses.
        assert_eq!(r.end().unwrap(), r.snapshot());
        assert!(r.push("x").is_err());
    }

    #[test]
    fn reset_clears_everything() {
        let mut r = StreamingRepairer::new(true);
        r.push(r#"{"a": [1, 2"#).unwrap();
        r.reset();
        assert_eq!(r.snapshot(), "");
        r.push("12").unwrap();
        assert_eq!(r.end().unwrap(), "12");
    }

    #[test]
    fn provisional_top_level_scalars_match_the_engine() {
        // The partial-literal decision: 'tru' is the engine's prose
        // sentinel at top level, and a complete strict scalar commits.
        // A completed top-level STRING commits only from the strict `"`
        // delimiter: `"hi"` is `"hi"`, `'hi'` was prose to the whole-
        // text engine (`''`) — end() equals the engine on both.
        assert_eq!(streamed("tru", 1), "");
        assert_eq!(streamed("true", 1), "true");
        assert_eq!(streamed("12", 1), "12");
        assert_eq!(streamed(r#""hi""#, 1), r#""hi""#);
        assert_eq!(streamed("'hi'", 1), "");
        assert_eq!(streamed(r#""hi"#, 1), "");
        assert_eq!(streamed("None", 1), "");
        assert_eq!(streamed("NaN", 1), "NaN");
        assert_eq!(streamed("12 13", 1), "");
    }

    #[test]
    fn the_in_flight_word_heals_through_the_literal_table() {
        // The engine's repair-lane observable for a word still in
        // flight at the cut: the special floats' spellings heal to
        // STRINGS (the strict fast path keeps them mid-stream:
        // `[NaN]` streams to `[NaN]`, the cut `[NaN` heals to
        // `["NaN"]`), and a word born from a number run drops its
        // stray leading sign.
        for (text, want) in [
            ("[NaN", r#"["NaN"]"#),
            ("[Infinity", r#"["Infinity"]"#),
            (r#"{"a": NaN"#, r#"{"a": "NaN"}"#),
            (r#"{"a": Infinity"#, r#"{"a": "Infinity"}"#),
            (r#"{"a": -Infinity"#, r#"{"a": "Infinity"}"#),
            ("[1, -NaN", r#"[1, "NaN"]"#),
            // the complete documents keep the floats (the strict class)
            ("[NaN]", "[NaN]"),
            ("[Infinity]", "[Infinity]"),
            (r#"{"a": NaN}"#, r#"{"a": NaN}"#),
            (r#"{"a": -Infinity}"#, r#"{"a": -Infinity}"#),
        ] {
            assert_eq!(streamed(text, 3), want, "{text:?}");
            let (value, _) = super::super::repair(text, &super::super::RepairConfig::default())
                .expect("engine repair succeeds");
            assert_eq!(streamed(text, 3), serializer::dumps(&value, true));
        }
    }

    #[test]
    fn the_empty_container_heal_matches_the_engine() {
        // The engine's truncated-output heal: a still-open empty
        // container drops at an item position, closes at a member-value
        // position or the root; an array's trailing strictly-empty
        // member drops; the cascade walks up.
        for (text, want) in [
            ("[[", "[]"),
            ("[[[", "[]"),
            ("[[{", "[]"),
            ("[1, [", "[1]"),
            ("[1, [{", "[1]"),
            ("[{}, [", "[{}]"),
            ("[[], [", "[[]]"),
            ("[[1], [", "[[1]]"),
            (r#"["x", ["#, r#"["x"]"#),
            (r#"{"a": [1, {"#, r#"{"a": [1]}"#),
            (r#"{"a": [1, ["#, r#"{"a": [1]}"#),
            (r#"{"a": {"b": ["#, r#"{"a": {"b": []}}"#),
            (r#"{"a": ["#, r#"{"a": []}"#),
            (r#"{"a": {"#, r#"{"a": {}}"#),
            (r#"{"a": [{"b": {"#, r#"{"a": [{"b": {}}]}"#),
            ("[{", "[]"),
            ("[1, {", "[1]"),
            (r#"{"a": 1, "b": ["#, r#"{"a": 1, "b": []}"#),
            ("[1, []", "[1]"),
            (r#"{"a": []"#, r#"{"a": []}"#),
            (r#"{"a": [["#, r#"{"a": []}"#),
            ("[1, [2, [", "[1, [2]]"),
            ("[[], []", "[[]]"),
            ("[[[]", "[]"),
            ("[1, [[[]", "[1]"),
            ("([", "[]"),
            ("(1, [", "[1]"),
            (r#"{"a": ("#, r#"{"a": []}"#),
            (r#"{"a": [("#, r#"{"a": []}"#),
            ("[[], [{\"a\": [", "[[], [{\"a\": []}]]"),
        ] {
            assert_eq!(streamed(text, 3), want, "{text:?}");
            let (value, _) = super::super::repair(text, &super::super::RepairConfig::default())
                .expect("engine repair succeeds");
            assert_eq!(streamed(text, 3), serializer::dumps(&value, true));
        }
    }

    #[test]
    fn the_empty_element_retract_matches_the_engine() {
        // The falsy-nudge class: an element that renders empty retracts
        // whole (the deferred separator included); the popped rollback
        // sign's one-char element retracts the same way, no cycle.
        for (text, want) in [
            (r#"[1, ""#, "[1]"),
            (r#"[1, """#, "[1]"),
            ("[1, -", "[1]"),
            ("[1,-]", "[1]"),
            ("[1-]", "[1]"),
            ("[1e-}", r#"["1e"]"#),
            ("[1e+-}", r#"["1e+"]"#),
            ("[1, 2e-}", r#"[1, "2e"]"#),
            ("[1e-]", r#"["1e"]"#),
            ("[1e- 2]", r#"["1e", 2]"#),
            ("(\u{b}-\0", "[]"),
            ("(1, -)", "[1]"),
            (r#"{"a": [1, ""#, r#"{"a": [1]}"#),
            // in an object the member value heals to "" (the engine's
            // own split)
            (r#"{"a": -}"#, r#"{"a": ""}"#),
            (r#"{"a": -"#, r#"{"a": ""}"#),
        ] {
            assert_eq!(streamed(text, 3), want, "{text:?}");
            let (value, _) = super::super::repair(text, &super::super::RepairConfig::default())
                .expect("engine repair succeeds");
            assert_eq!(streamed(text, 3), serializer::dumps(&value, true));
        }
    }

    #[test]
    fn the_stray_ellipsis_drop_matches_the_engine() {
        // The engine's array-lane stray-'...' drop (parse_array_items):
        // an item whose parse is the exact string "..." with the parse
        // ending on a '.' is ignored whole. At a cut that is the open
        // string's content (the fuzz-found divergence: " [\r\r\"..."
        // streamed ["..."] where the engine retracts to []), and a bare
        // number-run's value lanes make the same string (the closed
        // ["..."] element's parse ends on its QUOTE and stays).
        for (text, want) in [
            (" [\r\r\"...", "[]"),
            ("[\"...", "[]"),
            ("[[\"...", "[]"),
            ("[1, \"...", "[1]"),
            ("[1, 2, \"...", "[1, 2]"),
            ("{\"a\": [\"...", "{\"a\": []}"),
            ("[1, [\"...", "[1]"),
            ("(\"...", "[]"),
            ("[[...", "[]"),
            ("[1, ...", "[1]"),
            ("[..., 1]", "[1]"),
            ("[1, [] , 2 ", "[1, 2]"),
            // not the drop: the element's parse ends on something else,
            // or the content is not exactly three dots
            ("[\"....", "[\"....\"]"),
            ("[\"..", "[\"..\"]"),
            ("[\".", "[\".\"]"),
            ("[\" ...", "[\" ...\"]"),
            ("[\"  ...  ", "[\"  ...\"]"),
            ("[\"a...b", "[\"a...b\"]"),
            ("[\"...\", \"x\"", "[\"...\", \"x\"]"),
            ("[\"...\"]", "[\"...\"]"),
            ("[\"...\n", "[\"...\"]"),
            ("[\"...\\", "[\"...\\\\\"]"),
            ("{\"k\": \"...", "{\"k\": \"...\"}"),
        ] {
            assert_eq!(streamed(text, 3), want, "{text:?}");
            let (value, _) = super::super::repair(text, &super::super::RepairConfig::default())
                .expect("engine repair succeeds");
            assert_eq!(streamed(text, 3), serializer::dumps(&value, true));
        }
    }

    #[test]
    fn the_strictly_empty_item_skip_matches_the_engine() {
        // The engine's array-lane skip (parse_array_items): a
        // strictly-empty container/"" item is appended only when `,` or
        // the closer follows it IMMEDIATELY; whitespace between drops
        // the item at its own return. The machine defers the decision
        // to close time (the engine's whole-input loads fast path keeps
        // every element whenever the text ends up valid), so every pin
        // here is a cut.
        for (text, want) in [
            ("[[]\r, ", "[]"),
            ("[[] ,", "[]"),
            ("[[]  ,", "[]"),
            ("[[] , 1", "[1]"),
            ("[[] , 2", "[2]"),
            ("[[] ,[]", "[]"),
            ("[[] , []", "[]"),
            ("[[] , [] , [] ", "[]"),
            ("[1, []\r, ", "[1]"),
            ("[1, []\r, 2", "[1, 2]"),
            ("[1, [] , 2 ", "[1, 2]"),
            ("[1, [] , 2", "[1, 2]"),
            ("[1, [2, [] , 3", "[1, [2, 3]]"),
            ("[[]\r, []", "[]"),
            ("[[]\r, [] , 1", "[1]"),
            ("[[] , [] , 1", "[1]"),
            ("[[], [] ,", "[[]]"),
            ("[[], [] , 1", "[[], 1]"),
            ("[[] , [] , 1", "[1]"),
            ("[[] , 1 , 2 ", "[1, 2]"),
            ("[[] , \"k", "[\"k\"]"),
            ("[[] , 1", "[1]"),
            ("[{}\r, ", "[]"),
            ("[\"\"\r, ", "[]"),
            ("[\"\" , 1", "[1]"),
            ("[1, [[] , 2], 3", "[1, [2], 3]"),
            ("[[], [[] , 2], 3", "[[], [2], 3]"),
            ("{\"a\": [[] , ", "{\"a\": []}"),
            ("[[[] , 1", "[[1]]"),
            ("[[] , 1 , 2 ", "[1, 2]"),
            ("[[], [] , [] ,", "[[]]"),
            // the comma DIRECTLY after the element keeps it (the
            // engine's own split), and the complete documents keep
            // everything (the loads class)
            ("[[], ", "[[]]"),
            ("[[], 1", "[[], 1]"),
            ("[[], 1]", "[[], 1]"),
            ("[[] , 1]", "[[], 1]"),
            ("[\"\" , 1]", "[\"\", 1]"),
            ("[[], [] , 1]", "[[], [], 1]"),
            ("[1, [] , 2]", "[1, [], 2]"),
            ("[[] , 1 , 2]", "[[], 1, 2]"),
            ("[[], [[] , [] , 1]]", "[[], [[], [], 1]]"),
            // in an object the member value keeps its empty container
            // (the rule is the array lane's alone)
            ("{\"a\": [] , 1", "{\"a\": []}"),
            ("{\"a\": []\r, ", "{\"a\": []}"),
            ("{\"a\": {} , ", "{\"a\": {}}"),
        ] {
            assert_eq!(streamed(text, 3), want, "{text:?}");
            let (value, _) = super::super::repair(text, &super::super::RepairConfig::default())
                .expect("engine repair succeeds");
            assert_eq!(streamed(text, 3), serializer::dumps(&value, true));
        }
    }

    #[test]
    fn the_mismatched_closer_paren_family_matches_the_engine() {
        // The paren lane's closer rules (the group parses through
        // parse_array with `)` as ITS closing delimiter): a `]` inside an
        // open group is a garbage item the strictly-empty skip consumes —
        // the group stays open and keeps parsing (`[(], 1, 2]` re-lands
        // the `1, 2` inside the group, the engine's own swallow); a `}`
        // TERMINATES the group's loop and is consumed by it (`[(}]` —
        // the group closes at the `}` and the `]` keeps the element). An
        // EMPTY group closed under its own `)` defers the item-skip
        // decision to the NEXT char (`,` or the parent's closer keeps;
        // anything else — whitespace included — drops the element and is
        // eaten with it, the skip's one-char stall guard), and a group
        // the `]` left open retracts at close time like any empty
        // container. The last pair is the libFuzzer-minimized underflow
        // repro (the close-time walk's `member_count` decrement is
        // saturating now).
        for (text, want) in [
            ("[(]", "[]"),
            ("[(],", "[]"),
            ("[(] ]", "[]"),
            ("[(])", "[]"),
            ("[(,]", "[]"),
            ("[( ]", "[]"),
            ("[( ],", "[]"),
            ("((]", "[]"),
            ("[[(]", "[]"),
            ("([( ]", "[]"),
            ("[1,(]", "[1]"),
            ("[(),(]", "[[]]"),
            ("[(]x", "[\"x\"]"),
            ("[(], 1, 2]", "[[1, 2]]"),
            ("[( ], 1", "[[1]]"),
            ("[(}]", "[[]]"),
            ("[() ]", "[]"),
            ("[() ,]", "[]"),
            ("[()), ]", "[]"),
            ("[()x]", "[]"),
            ("[()x", "[]"),
            ("[()]x", "[[]]"),
            ("[()]", "[[]]"),
            ("[()],", "[[]]"),
            ("[(), ", "[[]]"),
            ("[1, ()]", "[1, []]"),
            ("[1, () ", "[1]"),
            ("{\"a\": (]", "{\"a\": []}"),
            ("{\"a\": [()]", "{\"a\": [[]]}"),
            ("\u{16}[(()\t,)((\u{1}", "[]"),
        ] {
            assert_eq!(streamed(text, 1), want, "{text:?} (one-char pushes)");
            assert_eq!(streamed(text, 3), want, "{text:?}");
            let (value, _) = super::super::repair(text, &super::super::RepairConfig::default())
                .expect("engine repair succeeds");
            assert_eq!(streamed(text, 3), serializer::dumps(&value, true));
        }
    }

    #[test]
    fn the_escape_tail_heal_matches_the_engine() {
        // The engine's escape-tail heal: an unterminated string's
        // content rstrips (Python's whitespace set: the raw space, the
        // \t/\n/\r/\f escapes, the 6-byte \u000b; a backspace passes);
        // a CLOSED string loses only a trailing newline-run adjacent to
        // its close; an escaped backslash-n is not a newline at all.
        for (text, want) in [
            (r#"{"k": "a\n"#, r#"{"k": "a"}"#),
            (r#"{"k": "a\n\t"#, r#"{"k": "a"}"#),
            (r#"{"k": "a\n "#, r#"{"k": "a"}"#),
            (r#"{"k": "a \n""#, r#"{"k": "a"}"#),
            (r#"{"k": "a\n""#, r#"{"k": "a"}"#),
            (r#"{"k": "\n""#, r#"{"k": ""}"#),
            (r#"{"k": "\n"#, r#"{"k": ""}"#),
            (r#"{"k": "a\n\n"#, r#"{"k": "a"}"#),
            (r#"{"k": "a\r\n""#, r#"{"k": "a"}"#),
            (r#"{"k": "a\r""#, r#"{"k": "a\r"}"#),
            (r#"{"k": "a\t""#, r#"{"k": "a\t"}"#),
            (r#"{"k": "a\n\t""#, r#"{"k": "a\n\t"}"#),
            (r#"{"k": "a\n ""#, r#"{"k": "a\n "}"#),
            (r#"{"k": "a\\n""#, r#"{"k": "a\\n"}"#),
            (r#"{"k": "a ""#, r#"{"k": "a "}"#),
            (r#"{"k": "a b ""#, r#"{"k": "a b "}"#),
            (r#"{"k": "a "#, r#"{"k": "a"}"#),
            (r#"{"k": "a b "#, r#"{"k": "a b"}"#),
            (r#"{"k": "  "#, r#"{"k": ""}"#),
            ("{\"k\": \"a\x08", "{\"k\": \"a\\b\"}"),
            (r#"["a\n"#, r#"["a"]"#),
            (r#"{"a": "x\n", "b": 1}"#, r#"{"a": "x\n", "b": 1}"#),
        ] {
            assert_eq!(streamed(text, 3), want, "{text:?}");
            let (value, _) = super::super::repair(text, &super::super::RepairConfig::default())
                .expect("engine repair succeeds");
            assert_eq!(streamed(text, 3), serializer::dumps(&value, true));
        }
    }

    #[test]
    fn the_missing_key_colon_drops_the_pair() {
        // `{: 1}` -> `{}` (the engine's repair lane: the whole pair
        // drops); the valid `{"": 1}` passes through untouched.
        assert_eq!(streamed("{: 1}", 2), "{}");
        assert_eq!(streamed("{ : 1}", 2), "{}");
        assert_eq!(streamed("{:}", 1), "{}");
        assert_eq!(streamed(r#"{: 1, "b": 2}"#, 2), r#"{"b": 2}"#);
        assert_eq!(streamed(r#"{"": 1}"#, 2), r#"{"": 1}"#);
    }

    #[test]
    fn depth_cap_errors_like_the_engine() {
        let mut r = StreamingRepairer::new(true);
        // Nested containers in member-value position (the shape that
        // actually recurses): 200 nest fine, 201 errors.
        let deep = r#"{"a": "#.repeat(MAX_NESTING);
        assert!(r.push(&deep).is_ok());
        assert!(r.push("{").is_err());
        assert!(r.push("{}").is_err());
        r.reset();
        assert!(r.push(&deep[..deep.len() - 1]).is_ok());
    }
}

#[test]
fn open_key_string_heals_to_a_dropped_member() {
    // The prefix-sweep shape: every prefix of a document stays
    // valid-or-sentinel, including the ones cut mid-key.
    for text in [r#"{"a"#, r#"{"a": 1, "b"#, r#"{"ab"#, r#"{'a"#] {
        let mut r = StreamingRepairer::new(true);
        r.push(text).unwrap();
        let closed = r.end().unwrap();
        assert!(
            closed.is_empty() || loads_strict(&closed).is_ok(),
            "{text:?} closed to invalid {closed:?}"
        );
    }
}

#[test]
fn popped_rollback_chars_cannot_cycle() {
    // The fuzz-found cycle: a popped rollback char re-creates the same
    // one-char number pending at an after-value position in an array,
    // whose termination pops it again. The machine escapes by dropping a
    // from-pop run's pop; the heal stays valid and the call terminates.
    for text in ["(\u{b}-\0", "[1e-}", "[1,-]", "[1e+-}"] {
        let mut r = StreamingRepairer::new(true);
        r.push(text).unwrap();
        let closed = r.end().unwrap();
        assert!(
            closed.is_empty() || loads_strict(&closed).is_ok(),
            "{text:?} closed to invalid {closed:?}"
        );
    }
}
