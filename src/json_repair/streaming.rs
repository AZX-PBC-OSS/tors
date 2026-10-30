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
//! - **The deferred-escape strings** (the weekly-fuzz crash class): the
//!   whole-text engine's two lanes decode `\f` and `\/` differently, and
//!   the lane that decides is the one a stream cannot know at the
//!   string's own close. The strict lanes (the whole-input `json.loads`
//!   fast path, the repair parser's `raw_decode` suffix probe) decode
//!   both escapes through the canonical table (`\f` to the form feed,
//!   `\/` to `/`); the REPAIR lane's `_normalize_escape_sequence`
//!   accepts neither -- the backslash stays a literal character in the
//!   value and the re-dump re-escapes it (`\f` to `\\f`, `\/` to
//!   `\\/`) -- and the repair lane decides whenever the document is CUT
//!   (no balanced-prefix strict load), a paren opened anywhere (the
//!   tuple grammar is never strict-valid), or the parse was otherwise
//!   repaired mid-document (the fast paths died on the INPUT: a bare
//!   word, a non-`"` delimiter, an in-container comment, a missing-
//!   comma/colon/key/value heal, a non-strict number run, a trailing
//!   separator, a mismatched closer). The machine's streaming decode
//!   emits the canonical form at the string's close, so committing
//!   those bytes finally would break the cut class (`["\f%"` streams
//!   `["\f%"]` where the engine heals to `["\\f%"]`). The machine
//!   therefore holds every string whose raw source contains a
//!   repair-lane-divergent escape -- exactly: a `\f` or `\/` escape (an
//!   odd backslash run followed by `f` or `/`), an odd backslash run of
//!   three or more (the lib's run scan collapses it to one literal
//!   backslash with the char after the run ordinary, where the
//!   streaming decode pairs the run), or a backslash before a
//!   delimiter that is not the string's own (the lib strips that
//!   escape) -- as a deferred-escape mark (the rendering's span in
//!   `out` plus the string's raw source, the `EmptyMark` pattern), and
//!   resolves every mark at `end`/`snapshot` close time: the render
//!   with the verbatim strings is one strict JSON value and neither a
//!   paren nor a mid-document repair fired (the fast paths' own
//!   verdict -- a complete document, trailing garbage after a valid
//!   root and comments behind a root included: `["\f"]`, `["\f%"]`,
//!   `{"\f\/%": 1}`, `["\f"]x`, `["\f"] // x`) emits the held span
//!   verbatim (the keep emits the canonical decode's OWN bytes: the
//!   `{"\f\/%": 1}` example's emitted text is `{"\f/%": 1}` -- the
//!   canonical table melts the escaped solidus and does not re-escape
//!   it, the engine's own bytes), anything else (the cut and drain classes: `["\f%"` to
//!   `["\\f%"]`, `["\f", 1` to `["\\f", 1]`, `{"\f\/%": 1` to
//!   `{"\\f\\/%": 1}`, `[[ "\f"]` to `[["\\f"]]`; the
//!   repair-lane-decided complete documents: `("\f")` to `"\\f"`,
//!   `[tru, "\f"]` to `["tru", "\\f"]`, `[1, "\f",]` to `[1, "\\f"]`)
//!   re-renders the held raw source through the lib's per-escape
//!   normalization (see `render_repair_lane_string`) and re-dumps it
//!   through the canonical table. An escape the two lanes decode
//!   IDENTICALLY never defers: the streaming decode's own table
//!   carries the repair lane's `x`-plus-two-hex-digits acceptance
//!   (`\x41` to `A`, `\x%` literal) beside the strict set. The mark
//!   flushes ONLY at close time: every separator keeps the verbatim
//!   rendering mid-stream, so the distinction survives the separator
//!   (`["\f", 1` escapes while `["\f", 1]` keeps), and the mark
//!   composes with the strictly-empty drops (the spans shift with
//!   their drains: `[[] , "\f", 1` to `["\\f", 1]`), the paren taint,
//!   the grouping unwrap and the empty-object fallback (both shift the
//!   marks with their one-char drains), and multi-string documents
//!   (each string holds its own mark). A string still OPEN at the cut
//!   is the same class the mark resolves: the cut's repair lane always
//!   decides (a truncated document never strict-loads), so the
//!   in-flight content's raw source re-renders through the SAME escape
//!   walk (`render_open_string_repair_lane`, the closed mark's
//!   `render_repair_lane_string` twin), then the engine's
//!   unterminated-string finalize rstrips the DECODED value (Python's
//!   whitespace set, the escape-tail heal's own operand: `["\f` ->
//!   `["\\f"]`, `["\/` -> `["\\/"]`, `{"k": "\f\n` -> `{"k": "\\f"}`,
//!   `{"k": "x\n\f` -> `{"k": "x\n\\f"}` -- the rstrip is the tail's,
//!   the mid-string newline-escape stands); a clean tail renders
//!   through the escape-tail path as before (the streaming decode's
//!   own bytes, the mark arms at the close only). The walk mirrors the
//!   lib's low smart-quote span machinery: an ordinary `„` in the
//!   value (never hard on a backslash-ending value) pushes the span
//!   and the odd-run collapse then KEEPS the collapsed backslash with
//!   the run's own delimiter joining it as content (`["e„\\\"` ->
//!   `["e\u201e\\\""]`, where the plain content drops it:
//!   `["e\\\"` -> `["e\""]`) -- `„` is the only pusher (no other
//!   `STRING_DELIMITERS` member re-decides; `”`, the `\`+`"` escape
//!   and the check-point `"` pop the span, spans nest; the check-point
//!   `"` is the lib's second pop arm -- a plain `"` whose predecessor
//!   took the default-append path, `_scan_string_body`'s low-span `"`
//!   arm at the loop tail -- and the streaming decode's own escape
//!   table carries the lib's active-delimiter arm beside it: an
//!   escaped OWN delimiter decodes to the delimiter alone, the
//!   backslash dropped, `['a\'` -> `["a'"]`); and the drain's trigger
//!   fires on a held partial escape too (`["\ud83d\u` -> `["\ufffd\\u"]`,
//!   the oracle keeping the `\ud83d` escape -- the documented
//!   lone-surrogate divergence). The `'`-delimited strings' span
//!   machinery lives in the streaming decode whole (they always take
//!   the lib's slow-path scan): the push gates, the pops, and the
//!   span-active close absorption -- the own-delimiter quote is
//!   content while a span is active (`['e„b'` -> `["e\u201eb'"]`); the
//!   span-active close with a follow-up quote/terminator ahead is the
//!   engine's handler lookahead's to decide (the documented
//!   divergence, DIVERGENCES-pinned).
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
//!   `{"k": "a\n ` -> `{"k": "a"}`), where a raw source holding a
//!   repair-lane-divergent escape resolves it FIRST through the
//!   repair lane's own normalization (the cut class always decides:
//!   `["\f` -> `["\\f"]`, `["\/` -> `["\\/"]` -- see the
//!   deferred-escape bullet), a CLOSED string that ends the
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
//!   its quote and stays; the rollback's popped ALPHABETIC char still
//!   re-parses as the next element — the skip ate exactly one char —
//!   `[...e` to `["e"]`, `[1, ...E` to `[1, "E"]`) and the strictly-empty item whose next char
//!   is not a separator (`[[] ,` -> `[]`, `[[] , 1` -> `[1]`).
//! - **The strictly-empty element's drops are LOCAL to their own
//!   array's closing shape** (the round-5 contract; the two earlier
//!   whole-document gates — "cut-only", then "the drain's gate is the
//!   engine's own loads decision" — were both WRONG abstractions and
//!   are gone): the element's one-char skip decision (the engine's
//!   `parse_array_items` skip reads the char right after the element)
//!   resolves per char —
//!   a `,` or the array's closer DIRECTLY keeps the element finally
//!   (the skip never fires on a separator: `[[], 1]`), whitespace
//!   HOLDS the decision, and any other char drops it — the engine's
//!   one-char stall guard eating exactly one char: the WHITESPACE when
//!   one preceded (the char re-parses as the next member/item:
//!   `[[] x, 1]` -> `["x", 1]`), else the char itself (`[[]x]` ->
//!   `[]`). A held decision (whitespace seen) converts to a deferred
//!   MARK at the next `,`/closer, and the mark's fate is the ARRAY'S
//!   OWN closing shape, decided at that array's pop: a TRAILING
//!   separator where a value was required drops it (`[[] ,]` -> `[]`,
//!   `[1, [] ,]` -> `[1]`, `{"a": [[] ,]}` -> `{"a": []}` — a trailing
//!   comma is never strict-valid, so neither fast path can cover the
//!   array and the engine's repair lane decided it); a PROPER close
//!   keeps it (`[[] , 1]` -> `[[], 1]`, `[true, [] , 1]`,
//!   `[[] ]` -> `[[]]`, `{"a": ["" , 1]}` -> `{"a": ["", 1]}` — the
//!   whole-input `json.loads` fast path and the strict-suffix probe
//!   keep every element of an array whose own body is strict-valid,
//!   trailing garbage after the valid root included: `[[] , 1] x` ->
//!   `[[], 1]`); and a CUT root (the root never closed) drops every
//!   mark still alive, properly-closed arrays' marks included — the
//!   repair lane decided the whole parse there (`[[], [[] , 2], 3` ->
//!   `[[], [2], 3]`). The PAREN TAINT never defers: a paren-derived
//!   empty group, or any element inside a paren's parse, drops on the
//!   first char whatever it is (the tuple grammar is never
//!   strict-valid, so the repair lane always decides: `[() ]` -> `[]`,
//!   `[([] , 1)]` -> `[[1]]`), keeping only for a `,`/closer that
//!   arrives directly (`[()]` -> `[[]]`). The same shape class
//!   re-decides the garbage-item run (a `,` at an item position
//!   consumes the non-value-start chars that follow, the frame's own
//!   closer included: `[(,)]` -> `[]`, `[(,), 1]` -> `[[1]]`) and the
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
//!   keeps the parts it saw, validly. The smart-quote span machinery is
//!   the same class: a slow-path string (one holding a backslash, where
//!   the strict fast paths died) whose value holds an ordinary `„`
//!   pushes the engine's low smart-quote span, the span-ACTIVE close
//!   quote is CONTENT (the string runs on, `["e„\\\"""` -> the value
//!   `e„\"""`), and the post-close tail re-synchronizes with it
//!   (`["e„"x"]` -> the engine `["e„\"x\"]"]`); the stream closes at the
//!   quote (`["e„\\\"""` -> `["e„\\\""]`, the `„`-free analog
//!   `["e"x"]` diverging the same way through the quoted-section
//!   re-pair).
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
//! - **The object-lane comma-glue's split residual**: the engine's
//!   classify_object_value_comma re-decides a comma glued into an
//!   object-lane number run from the WHOLE tail; the stream re-decides
//!   it at the `:` that proves the post-comma text a key (the run
//!   splits there: the value renders up to the comma, the post-comma
//!   text re-enters the key lane — `{"a": 1,7Z: 8}` -> `{"a": "1",
//!   "7Z": 8}`, `{Z7:8,7Z:8` -> `{"Z7": "8", "7Z": 8}`, the engine's
//!   own shapes). The half of the engine's classification the linear
//!   machine cannot see at that `:` — the recoverable-value check
//!   needs the post-colon value's OWN tail to the object's `}` — is
//!   decided optimistically: a value led by a bare word whose tail
//!   holds a quote before the close, or a cut before the close, flips
//!   the engine to keep-the-comma where the split already fired
//!   (`{"a": 1,b: x"y}` -> the engine `{"a": "1,b: x"}`, the stream
//!   `{"a": "1", "b": "x\"y"}`; the cut `{"a": 1,b: x` -> the engine
//!   `{"a": "1,b: x"}`, the stream `{"a": "1", "b": "x"}`). Both
//!   outputs stay valid JSON; the parity harness's comma gate keeps
//!   the class out of the differential.
//! - **The unquoted-value run's container/whitespace absorption**: the
//!   engine's unquoted-value lane eats container chars and internal
//!   whitespace into the string and re-decides literal prefixes
//!   (`[{"b": undefined},]` -> the engine `[{"b": "undefined},"}]`,
//!   the stream `[{"b": "undefined"}]`; `[null x]` -> the engine
//!   `[null, "x"]`, the stream `["null x"]`); the stream's word stops at
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
//! - **The missing-colon shape with a CONTAINER where the colon was
//!   due**: the engine's object loop consumes the container-open as the
//!   colon-substitute and its repair lane empties the array
//!   (`{"a" [[] , 1]}` -> `{"a": []}`); the stream inserts the colon
//!   and keeps the value it saw (`{"a": [[], 1]}`).
//! - **The completed-paren taint on a later deferred element**: a paren
//!   anywhere in the input kills both fast paths (the whole-input loads
//!   and the suffix probe both fail on a tuple), so the engine's repair
//!   lane decided the array and its skip fired (`[(2), [] , 1]` ->
//!   the engine `[2, 1]`); the stream's local rule (the array's own
//!   closing shape) kept it (`[2, [], 1]`).
//! - **The COMMA-first tail after a valid root**: the engine's suffix
//!   probe is gated on non-comma trailing content, so the repair lane
//!   decided and the element dropped (`[[] , 1] ,` -> the engine `[1]`);
//!   the stream's local rule kept it (`[[], 1]`). Any other trailing
//!   garbage keeps on both sides.
//! - **The unquoted-value `}`-absorption without the ws**: a non-strict
//!   run VALUE at an object's close absorbs the object's own `}` when
//!   the tail past it holds two chars before the next `}`
//!   (`{"a": b}66` -> the engine `{"a": "b}66"}`); the stream keeps the
//!   run's own termination (`{"a": "b"}`).
//! - **The mismatched-closer garbage**: the engine's whole-text
//!   close-up re-decides earlier structure on garbage
//!   (`{{\r]0\x01...('` -> `[]`, the stream `{}`), both valid.
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
    /// A char was consumed at the top level BEFORE the root container's
    /// own char (leading whitespace, prose, a comment, a skipped `(` --
    /// the engine's `scroll_whitespaces` + ignore-and-move-on loop both
    /// advance `self.index`). The suffix-probe gate's first arm: with
    /// `index != 0` the probe fires WHATEVER the tail holds (raw_decode
    /// ignores the trailing text behind the first value), so the tail
    /// never re-decides the held strings -- the taint is the root
    /// rendering's own strictness alone (see `render_closed`).
    top_led: bool,
    /// A non-whitespace token was consumed after the root closed (a
    /// prose char, a comment opener): the suffix-probe gate's trailing
    /// scan reads the FIRST char after the root's closer (whitespace
    /// skipped) and a non-comma token there fires the probe. Once seen,
    /// later whitespace is behind the probe, never the input's last
    /// word.
    top_tail_token: bool,
    /// A NON-STRICT whitespace char was consumed after the root closed
    /// with no token behind it yet (Python `isspace` set: `\x0b`, `\x0c`
    /// ... -- the strict scanner's four never arm this): the deferred
    /// half of the suffix-probe gate's trailing scan. The input may END
    /// here -- `["\f"]\x0b` -- where the probe gate reads no trailing
    /// content (False), the repair lane decides, and the held strings
    /// re-render; or a further char may arrive, and ITS class decides
    /// (`,` still taints -- the comma-first tail -- anything else
    /// disarms, the probe fired). Settled at the close: the pending arm
    /// taints (`render_closed`), a disarmed one is gone.
    top_tail_ws_pending: bool,
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
    /// frame's first, the frame's index, the array's own closing shape
    /// said DROP)` span per strictly-empty element whose next char was
    /// whitespace. The engine decides at the element's own return (the
    /// next char, whitespace included, drops it); the machine defers
    /// because the WHOLE-INPUT `json.loads` fast path and the strict-
    /// suffix probe keep the element whenever the text (or the initial
    /// container plus junk) ends up one strict JSON value. The marks'
    /// fate is the LOCAL rule of `empty_pending`'s doc: each mark's own
    /// array's closing shape decides at that array's pop, and a cut
    /// (the root never closed) drops every mark still alive. Drained
    /// by `render_closed` (pure, local).
    empty_marks: Vec<EmptyMark>,
    /// The strictly-empty element's deferred one-char decision; see
    /// `EmptyPending`. Armed when a strictly-empty element (an empty
    /// container's pop, a closed `""` string) completes inside an
    /// array/tuple frame, resolved by the next char, never carried
    /// across a push boundary in any observable way (the state is part
    /// of the machine, so chunk splits do not change the outcome).
    empty_pending: Option<EmptyPending>,
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
    /// The deferred-escape strings (see `EscapeMark`): closed strings
    /// whose raw source holds an escape the repair lane's normalization
    /// decides differently from the streaming decode. Armed at the
    /// string's close, resolved at every close render, never flushed at
    /// a separator.
    escape_marks: Vec<EscapeMark>,
    /// A `(`-container opened anywhere (the paren taint): the tuple
    /// grammar is never strict-valid, so both fast paths died and the
    /// repair lane decided the whole parse -- the deferred-escape
    /// strings re-render at close time even when the root closed.
    saw_paren: bool,
    /// The repair lane decided the parse MID-DOCUMENT (a repair the
    /// strict grammar has no reading for fired inside the stream: a bare
    /// word or a non-strict number run committed in a container, a
    /// non-`"` delimiter, an in-container comment, a missing-comma/
    /// colon/key/value heal, a dangling-member or empty-element drop, a
    /// mismatched-closer close-up, the empty-object fallback, a
    /// trailing-separator close, the garbage-item run): the whole-input
    /// `json.loads` fast path and the repair parser's suffix probe both
    /// died on the INPUT, so the engine's repair lane re-normalized
    /// every string -- the deferred-escape strings re-render at close
    /// time even when the machine's render is itself strict-loadable
    /// (`[tru, "\f"]` to the engine `["tru", "\\f"]`). Top-level prose
    /// and trailing junk do NOT taint (the suffix probe keeps those:
    /// `["\f"] x` stays verbatim).
    repair_decided: bool,
}

/// One deferred strictly-empty element drop; see `empty_marks`. `dropped`
/// is the mark's OWN array's closing-shape verdict, recorded ONCE when
/// that array pops (the `None` guard: a popped frame's index is reused
/// by the next frame at the same depth -- only the mark's own frame's
/// pop may verdict): a trailing separator where a value was required
/// drops the element (the engine's repair lane decided the array -- a
/// trailing comma is never strict-valid, so no fast path can cover it);
/// a proper close keeps it (the fast paths do). A mark whose array never
/// popped (the cut) stays `None`, and a cut root drains every mark still
/// alive: the engine's repair lane decided the whole parse.
struct EmptyMark {
    start: usize,
    end: usize,
    was_first: bool,
    frame: usize,
    dropped: Option<bool>,
}

/// One deferred-escape string: a CLOSED string whose raw source held a
/// repair-lane-divergent escape (`\f`, `\/`, an odd backslash run of
/// three or more, a misescaped non-active delimiter -- see
/// `raw_source_needs_escape_repair`). The machine's streaming decode
/// committed the canonical rendering at the string's own close, but the
/// engine's lane that decides is not known until close time: the strict
/// lanes (the whole-input `json.loads`, the repair parser's suffix
/// probe) keep the canonical rendering verbatim, the repair lane
/// re-normalizes the raw source's escapes and re-dumps (`\f` to `\\f`).
/// The mark holds the rendering's span in `out` (the quoted string
/// whole), the raw source between the delimiters, the opening delimiter
/// and the rendering itself (the drain guard's identity): resolved at
/// every `end`/`snapshot` close render, verbatim or re-rendered (see
/// `render_closed`), never mid-stream (a separator must not flush it:
/// `["\f", 1` escapes while `["\f", 1]` keeps).
struct EscapeMark {
    /// The rendering's span in `out`: the opening quote through the
    /// closing quote.
    start: usize,
    end: usize,
    /// The string's raw source between the delimiters (the re-render's
    /// operand).
    raw: String,
    /// The string's opening delimiter (the escape normalization's
    /// rstring-delimiter rule).
    delim: char,
    /// The exact rendering at the span (the close-time drain guard: a
    /// span the mid-stream surgery retracted or moved no longer
    /// resolves).
    rendered: String,
}

/// The strictly-empty element's one-char decision: the element's
/// rendering (from the parent frame's `member_start` to the element's
/// end) is exactly `[]`/`{}`/`""`, its parent is an array-lane frame
/// holding it as an in-flight member, and the engine's item skip now
/// reads the NEXT char: a `,` or the parent's closer keeps the
/// element (finally so when no whitespace preceded -- the engine's
/// skip never fires on a separator; through a deferred mark when it
/// did -- the fast paths may still keep it, the array's own closing
/// shape decides), whitespace holds the decision (`ws_seen`: the skip
/// fired on the whitespace char, but what it implies waits on the next
/// char), any other char drops it -- and the one-char stall guard (the
/// skip's `index += 1` eats exactly one char) eats the WHITESPACE when
/// one preceded (this char re-parses fresh), else THIS char (consumed
/// with the drop). `parent_paren` distinguishes the paren lane's `)`
/// closer. `never_defers` is the PAREN TAINT: the element is itself a
/// paren-derived empty group, or some enclosing frame is a paren -- the
/// tuple grammar is never strict-valid, so no fast path can cover the
/// element and the repair lane always decides: the drop fires on the
/// first char whatever it is (round 3's immediate semantics), keeping
/// only for a `,`/closer that arrives DIRECTLY.
struct EmptyPending {
    start: usize,
    end: usize,
    was_first: bool,
    frame: usize,
    parent_paren: bool,
    never_defers: bool,
    ws_seen: bool,
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
/// `raw` accumulates the string's raw source between the delimiters
/// (the deferred-escape check's operand at the close; freed there when
/// no mark arms).
struct StrState {
    delim: char,
    /// Key vs value string: only the post-close transition differs.
    is_key: bool,
    esc: Esc,
    /// A decoded high surrogate awaiting its low half (the pair may
    /// straddle chunks; the hold is part of the machine's state).
    high: Option<u16>,
    /// The raw source seen so far, delimiters excluded (the closing
    /// delimiter's own char pops off at the close).
    raw: String,
    /// The LOW SMART-QUOTE SPAN depth (the lib's `rstring_delimiter`
    /// stack gaining the `"\0` sentinels): an ordinary `„` -- the char
    /// the scan meets at its loop top, never an escape's operand, and
    /// not hard on a backslash-ending value -- pushes one; an ordinary
    /// `”` pops one; a `\` + `"` escape pops one; a check-point `"` (a
    /// plain `"` whose predecessor took the default-append path) pops
    /// one. While a span is ACTIVE the string cannot close: the scan's
    /// `while char != outer or in_span` gate keeps the own-delimiter
    /// quote in the body as content (the close absorption -- the
    /// engine's own observable, `['e„b\\'c'` -> the value `e„b'c'`).
    spans: usize,
    /// Whether the DECODED value ends on a literal backslash (the
    /// `„`-push gate's `string_acc[-1] != "\\"`): only a decoded `\\`
    /// leaves one.
    decoded_ends_backslash: bool,
    /// Whether the char just consumed took the lib's DEFAULT-APPEND
    /// path (appended at the loop body's tail, the post-checks fallen
    /// through): the check-point `"` pop's gate -- a `"` whose
    /// predecessor was default-appended pops the span (the scan's
    /// low-span `"` arm examines the char AFTER the appended one); a
    /// `"` whose predecessor took any other arm (a `„` push, a `”` pop,
    /// an escape decode, a popped-`"`'s own literal append) is the
    /// loop-top CURRENT char there and appends itself without popping.
    last_default_append: bool,
}

/// The escape sub-state of an open string.
enum Esc {
    None,
    /// A backslash was consumed; the next char decides.
    Backslash,
    /// A `\u` was consumed; 0..=4 hex chars collected so far. The escape
    /// is literal text unless exactly four hex digits arrive.
    U(String),
    /// A `\x` was consumed; 0..=2 hex chars collected so far. The
    /// repair lane's own acceptance (`_normalize_escape_sequence`):
    /// exactly two hex digits decode the code point, fewer leave the
    /// escape literal text.
    X(String),
}

impl StrState {
    /// Feed one content char; append its canonical form to `sink`.
    /// Returns true when the string closed (the delimiter, un-emitted:
    /// the caller emits the canonical closing quote).
    fn feed(&mut self, c: char, sink: &mut String, ensure_ascii: bool) -> bool {
        match std::mem::replace(&mut self.esc, Esc::None) {
            Esc::Backslash => {
                match c {
                    'u' => {
                        // The escape continues; a held high surrogate
                        // stays held (this `\u` may be its low half).
                        self.esc = Esc::U(String::new());
                        return false;
                    }
                    'x' => {
                        // The repair lane's own x-acceptance: exactly
                        // two hex digits decode the code point.
                        self.esc = Esc::X(String::new());
                        return false;
                    }
                    // The escaped OWN delimiter decodes to the delimiter
                    // alone (the lib's `_normalize_escape_sequence`
                    // active-delimiter arm: `['a\\'` -> the value
                    // `a'`, the backslash dropped) -- the static list
                    // below only spells the `"`-delimited form, and an
                    // escaped own `'` fell through to the
                    // invalid-escape arm (the backslash standing, the
                    // stream `["a\\'"]` where the engine `["a'"]`).
                    // Either spelling arm the lib takes (the
                    // active-delimiter arm without a span, the
                    // smart-`„` arm under one), the escape decode
                    // consumed the char: no span pop, and the char
                    // after re-enters at the loop top.
                    _ if c == self.delim => {
                        self.flush_high(sink, ensure_ascii);
                        self.last_default_append = false;
                        self.decoded_ends_backslash = false;
                        push_escaped_char(sink, c, ensure_ascii);
                        return false;
                    }
                    // Valid simple escapes decode; the canonical emission
                    // re-encodes through the same table (`"` -> `\"`, a
                    // raw `\/` -> `/`, ...).
                    '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => {
                        self.flush_high(sink, ensure_ascii);
                        let decoded = match c {
                            'b' => '\u{8}',
                            'f' => '\u{c}',
                            'n' => '\n',
                            'r' => '\r',
                            't' => '\t',
                            other => other,
                        };
                        // The low-span arm of `_normalize_escape_sequence`
                        // (`parse_string.py` ~298): a `\`+`"` escape
                        // inside the span pops it -- the arm tests the
                        // LITERAL `"` whatever the string's own delimiter
                        // is (the span's machinery is quote-shaped, not
                        // delimiter-shaped).
                        if c == '"' && self.spans > 0 {
                            self.spans -= 1;
                        }
                        self.last_default_append = false;
                        self.decoded_ends_backslash = c == '\\';
                        push_escaped_char(sink, decoded, ensure_ascii);
                        return false;
                    }
                    // An invalid escape keeps the backslash literally (the
                    // engine's `"x\qy"` -> `"x\\qy"` shape: the decoded
                    // text is backslash + q, re-encoded backslash-backslash).
                    // The lib's own escape char re-parses at the loop top
                    // (the default append), so the flag OPENS.
                    _ => {
                        self.flush_high(sink, ensure_ascii);
                        self.last_default_append = true;
                        self.decoded_ends_backslash = false;
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
                        self.emit_code(code, sink, ensure_ascii);
                        self.last_default_append = false;
                        // A decoded U+005C IS a backslash: the value ends
                        // on one (the `„`-push gate's own
                        // `string_acc[-1] != "\\"` case, the DRAIN pins'
                        // `\u005c` no-push control).
                        self.decoded_ends_backslash = code == 0x5C;
                    } else {
                        self.esc = Esc::U(hex);
                    }
                    return false;
                }
                // Fewer than four hex digits: the escape is literal text
                // (the engine's `"\u41"` -> `"\\u41"` shape), and c is
                // ordinary content: fall through to the plain path (the
                // lib's own escape char re-parsed at the loop top; its
                // default-append sets the flag there).
                self.flush_high(sink, ensure_ascii);
                push_escaped_char(sink, '\\', ensure_ascii);
                push_escaped_char(sink, 'u', ensure_ascii);
                for h in hex.chars() {
                    push_escaped_char(sink, h, ensure_ascii);
                }
            }
            Esc::X(mut hex) => {
                if c.is_ascii_hexdigit() && hex.len() < 2 {
                    hex.push(c);
                    if hex.len() == 2 {
                        // Two collected hex digits: cannot fail to parse
                        // (the code point is at most 0xFF, always a char).
                        let code = u16::from_str_radix(&hex, 16).unwrap_or(0xFFFD);
                        self.flush_high(sink, ensure_ascii);
                        let decoded = char::from_u32(u32::from(code)).unwrap_or('\u{fffd}');
                        push_escaped_char(sink, decoded, ensure_ascii);
                        self.last_default_append = false;
                        self.decoded_ends_backslash = code == 0x5C;
                    } else {
                        self.esc = Esc::X(hex);
                    }
                    return false;
                }
                // Fewer than two hex digits: the escape is literal text
                // (the lib's not-accepted path, the engine's
                // `"\x%"` -> `"\\x%"` shape), and c is ordinary content:
                // fall through to the plain path.
                self.flush_high(sink, ensure_ascii);
                push_escaped_char(sink, '\\', ensure_ascii);
                push_escaped_char(sink, 'x', ensure_ascii);
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
        self.flush_high(sink, ensure_ascii);
        // The span-active close absorption: while a low smart-quote span
        // is ACTIVE the scan's `while char != outer or in_span` gate
        // keeps the own-delimiter quote in the body as content -- BOTH
        // arms the quote arrives by: the loop-top current char (the
        // predecessor took a continue arm) is absorbed by the while
        // gate itself, and the POST-CHECK char (the predecessor
        // default-appended) falls through `_handle_right_delimiter_
        // candidate` when nothing lies ahead (the input ends on the
        // quote: `['e„b'` -> the value `e„b'`) and re-enters at the
        // loop top, absorbed the same. The handler's CLOSE arms (a
        // follow-up quote/terminator ahead: `['e„b'c']` closes at the
        // first quote) are the whole-text lookahead's to decide -- the
        // linear machine's documented divergence there.
        if c == self.delim && self.spans > 0 {
            self.decoded_ends_backslash = false;
            push_escaped_char(sink, c, ensure_ascii);
            return false;
        }
        if c == self.delim {
            return true;
        }
        // The span's own loop-top arms (the push/pop/checkpoint trio),
        // the `„`-push gate reading the backslash state BEFORE this char
        // clears it. The machinery is the `'`-delimited strings' own:
        // the lib's fast path (`_try_parse_simple_quoted_string`) is
        // `"`-delimited ONLY -- a `"`-delimited body with no backslash
        // and a clean follower closes through it without EVER running
        // the slow-path span scan, so a `"`-delimited `„` pushes nothing
        // here (the machine's span state stays 0, the close is the
        // plain one, and the slow-path span shapes keep their
        // documented divergences); a `'`-delimited string ALWAYS takes
        // the slow path.
        if c == '\u{201e}' && !self.decoded_ends_backslash {
            if self.delim == '\'' {
                self.spans += 1;
                self.last_default_append = false;
            } else {
                self.last_default_append = true;
            }
        } else if c == '\u{201e}' {
            // Hard on a backslash-ending value: the entry check fails,
            // the `„` falls through to the default append.
            self.last_default_append = true;
        } else if c == '\u{201d}' && self.spans > 0 {
            self.spans -= 1;
            self.last_default_append = false;
        } else if c == '"' && self.spans > 0 && self.last_default_append {
            // THE CHECK-POINT POP (the scan's low-span `"` arm at the
            // loop tail): the span pops, the `"` appends literally --
            // and the pop arm's own literal append is NOT a
            // default-append (the doubled-quote pop: a `"` hard on this
            // one's heels does not pop again).
            self.spans -= 1;
            self.last_default_append = false;
        } else {
            // An ordinary char (a span-less `”` included): the default
            // append.
            self.last_default_append = true;
        }
        self.decoded_ends_backslash = false;
        // A raw control char inside a string is the strict scanner's
        // rejection (json.loads fails on the whole input).
        push_escaped_char(sink, c, ensure_ascii);
        false
    }

    /// A held lone high surrogate stranded by content that is not its
    /// low half flushes as U+FFFD (the engine's decode order: the
    /// surrogate's position first, then the content).
    fn flush_high(&mut self, sink: &mut String, ensure_ascii: bool) {
        if self.high.take().is_some() {
            push_escaped_char(sink, '\u{fffd}', ensure_ascii);
        }
    }

    /// `flush_high`'s raw-char twin for the repair-lane re-render's
    /// value building: the same decision, the decoded char left raw
    /// (the re-dump escapes it).
    fn flush_high_raw(&mut self, value: &mut String) {
        if self.high.take().is_some() {
            value.push('\u{fffd}');
        }
    }

    /// One complete `\uXXXX` code unit: a surrogate pair combines; a lone
    /// high surrogate is HELD (its low half may arrive in a later chunk);
    /// everything else (and a lone low surrogate) is U+FFFD, the engine's
    /// documented lone-surrogate divergence. The held surrogate at
    /// string close renders U+FFFD too (`{"a": "\ud83d` ->
    /// `{"a": "\ufffd"}`, the engine's own heal).
    fn emit_code(&mut self, code: u16, sink: &mut String, ensure_ascii: bool) {
        let mut decoded = String::new();
        self.emit_code_raw(code, &mut decoded);
        for ch in decoded.chars() {
            push_escaped_char(sink, ch, ensure_ascii);
        }
    }

    /// `emit_code`'s raw-char twin: the same hold-and-combine decision
    /// tree, the decoded chars pushed unescaped into the value (the
    /// re-dump escapes them).
    fn emit_code_raw(&mut self, code: u16, value: &mut String) {
        if let Some(high) = self.high.take() {
            if (0xDC00..=0xDFFF).contains(&code) {
                let combined =
                    0x10000 + ((u32::from(high) - 0xD800) << 10) + (u32::from(code) - 0xDC00);
                if let Some(ch) = char::from_u32(combined) {
                    value.push(ch);
                    return;
                }
            }
            // Not the low half: flush the held surrogate as U+FFFD, then
            // place the fresh unit on its own merits (it may itself be a
            // high surrogate to hold).
            value.push('\u{fffd}');
        }
        match char::from_u32(u32::from(code)) {
            Some(ch) => value.push(ch),
            None if (0xD800..=0xDBFF).contains(&code) => self.high = Some(code),
            None => value.push('\u{fffd}'),
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
            Esc::X(hex) => {
                push_escaped_char(sink, '\\', ensure_ascii);
                push_escaped_char(sink, 'x', ensure_ascii);
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
            top_led: false,
            top_tail_token: false,
            top_tail_ws_pending: false,
            ensure_ascii,
            finished: false,
            emitted: 0,
            last_string_start: None,
            string_raw_tail: None,
            prev_char_ws: false,
            empty_marks: Vec::new(),
            empty_pending: None,
            last_key_empty: false,
            garbage_run: false,
            paren_gate: None,
            escape_marks: Vec::new(),
            saw_paren: false,
            repair_decided: false,
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
            self.escape_marks.clear();
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
        // the parent frame's closer keeps the element -- FINALLY so when
        // it arrives directly (the repair lane's skip never fires on a
        // separator), through a deferred mark when whitespace preceded
        // (the mark's own array's closing shape then decides: the
        // trailing-separator class drops, the proper close keeps).
        // Whitespace holds the decision (`ws_seen`). Everything else
        // drops it, and parse_array's one-char stall guard (the skip's
        // `index += 1` eats exactly one char) eats the WHITESPACE when
        // one preceded -- this char re-parses fresh -- else THIS char,
        // consumed with the drop.
        if let Some(mut pending) = self.empty_pending.take() {
            let parent_closer = if pending.parent_paren { ')' } else { ']' };
            if c == ',' || c == parent_closer {
                if pending.never_defers {
                    // The repair lane's skip does not fire on a
                    // separator: the element keeps, finally (the
                    // paren-tainted element's own lane decided it).
                } else if pending.ws_seen {
                    // The repair lane's skip fired on the whitespace
                    // char; whether the element survives is the ARRAY'S
                    // OWN closing shape's call (the mark, decided at
                    // that array's pop): a trailing separator where a
                    // value was required drops (`[[] ,]` -> `[]`), a
                    // proper close keeps (`[[] , 1]` -> `[[], 1]`,
                    // `[[] ]` -> `[[]]`) -- the fast paths (the
                    // whole-input `json.loads`, the strict-suffix
                    // probe) keep every element of an array whose own
                    // body is strict-valid.
                    self.empty_marks.push(EmptyMark {
                        start: pending.start,
                        end: pending.end,
                        was_first: pending.was_first,
                        frame: pending.frame,
                        dropped: None,
                    });
                }
            } else if pending.never_defers {
                // The PAREN TAINT: a paren-derived or paren-enclosed
                // element never defers -- the tuple grammar is never
                // strict-valid, so the repair lane always decides --
                // the drop fires now, on the first char whatever it is
                // (round 3's semantics, which matched the engine).
                if self.frames.last().is_some() {
                    self.drop_marked_member(pending.start);
                } else if pending.start <= self.out.len() {
                    self.repair_decided = true;
                    self.out.truncate(pending.start);
                    self.emitted = self.emitted.min(pending.start);
                    self.out.push(']');
                }
                self.prev_char_ws = is_py_whitespace(c);
                return Ok(());
            } else if c.is_whitespace() {
                // The decision holds: the next char (or the cut) still
                // decides. `ws_seen` records that the skip's one-char
                // stall guard has a whitespace char to eat.
                pending.ws_seen = true;
                self.empty_pending = Some(pending);
            } else if pending.ws_seen {
                // The skip fired on the whitespace and ate exactly one
                // char -- the whitespace. This char re-parses fresh:
                // drop the element and REPLAY the char (it becomes the
                // next member/item: `[[] x, 1]` -> `["x", 1]`). The
                // element's fate is certain here: a body with `[] x`
                // in it is never strict-valid, so no fast path can
                // ever keep it. The parent frame may already be gone
                // (the element's own parse consumed a mismatched
                // closer that closed the document): the span's
                // truncation stands on its own then.
                if self.frames.last().is_some() {
                    self.drop_marked_member(pending.start);
                } else if pending.start <= self.out.len() {
                    self.repair_decided = true;
                    self.out.truncate(pending.start);
                    self.emitted = self.emitted.min(pending.start);
                    self.out.push(']');
                }
                return self.process(c);
            } else {
                // The engine's repair-lane skip, no whitespace
                // involved: drop the element and consume the char with
                // it (`[[]x]` -> `[]` -- the guard eats exactly one
                // char, this one). The parent frame may already be
                // gone (see above): the root's closer (consumed with
                // the mismatched close) re-emits.
                if self.frames.last().is_some() {
                    self.drop_marked_member(pending.start);
                } else if pending.start <= self.out.len() {
                    self.repair_decided = true;
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
                // The `(` and its group were consumed at the top level:
                // a later root container's char sits at `index != 0`.
                self.top_led = true;
                // The group's render is gone: any deferred-escape marks
                // inside it died with the span.
                self.escape_marks.clear();
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
                self.top_led = true;
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
            self.top_led = true;
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
            st.raw.push(c);
            let closed = st.feed(c, &mut self.out, self.ensure_ascii);
            if closed {
                self.out.push('"');
                // The deferred-escape check: a string whose raw source
                // holds a repair-lane-divergent escape (`\f`, `\/`, an
                // odd backslash run of three or more, a misescaped
                // non-active delimiter) does not commit its
                // rendering finally at its own close -- the rendering
                // stays in `out` (the separators and containers after it
                // build on it) but the mark holds the span and the raw
                // source, and the close render resolves it (verbatim or
                // re-rendered, see `render_closed`). The closing
                // delimiter's own char pops off the raw source first
                // (the plain path consumed it as the delimiter).
                st.raw.pop();
                let needs_mark = raw_source_needs_escape_repair(&st.raw, st.delim);
                let strict_decodable = raw_source_is_strict_decodable(&st.raw);
                if needs_mark {
                    let start = self.last_string_start.expect("a closed string had a start");
                    self.escape_marks.push(EscapeMark {
                        start,
                        end: self.out.len(),
                        raw: std::mem::take(&mut st.raw),
                        delim: st.delim,
                        rendered: self.out[start..].to_string(),
                    });
                }
                // A string whose raw source is not one strict-JSON-
                // decodable body (an escape outside the strict set, a
                // `\u` without four hex digits, a raw control char)
                // failed the whole-input `json.loads` on the INPUT: the
                // repair lane decided the document and re-normalized
                // every string -- even where the machine's own decode
                // repaired the escape into strict validity (`\x41`).
                if !strict_decodable {
                    self.repair_decided = true;
                }
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
                self.arm_empty_pending(false);
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
                    // The suffix-probe gate's trailing scan (the lib's
                    // `_initial_container_has_non_comma_trailing_content`,
                    // read from the root's closer): a NON-STRICT
                    // whitespace char is a repair-lane input only when
                    // the input can still END here -- the root's own char
                    // opened the input (`index == 0`: no scroll ran, the
                    // gate scans the tail) and no token sits behind the
                    // root yet (the scan reads the FIRST char past the
                    // whitespace). Otherwise the scan's decision char is
                    // not this whitespace: a token behind the root fired
                    // the probe already (`["\f"]\x0b x`), or the root's
                    // char sat behind leading content (`index != 0`, the
                    // probe fires whatever the tail holds -- the
                    // leading-consumption arm). The arm DEFERS: the tail
                    // (or the cut) settles it (see the arms below and
                    // `render_closed`).
                    if !matches!(c, ' ' | '\t' | '\n' | '\r')
                        && !self.top_led
                        && !self.top_tail_token
                    {
                        self.top_tail_ws_pending = true;
                    }
                }
                ',' => {
                    // The COMMA-FIRST tail: the gate's trailing scan reads
                    // a `,` as the first char past the root's closer (the
                    // whitespace skipped) and returns False -- no probe,
                    // the whole-input loads died on the tail, the repair
                    // lane decided (`["\f"] ,` -> `["\\f"]`). Behind
                    // leading consumption (`x ["\f"] ,`) or an earlier
                    // token (`["\f"] x ,`) the probe fired already: keep.
                    if !self.top_led && !self.top_tail_token {
                        self.repair_decided = true;
                    }
                    self.top_tail_token = true;
                    self.top_tail_ws_pending = false;
                }
                '#' | '/' => {
                    // A comment (or any trailing token) after the root:
                    // the whole input was not one strict JSON value.
                    // The gate's trailing scan reads the comment OPENER
                    // as the decision char -- non-comma, the probe
                    // fires (`["\f"]\x0b // c` keeps) -- and the
                    // deferred whitespace arm above settles too.
                    self.top_tail_token = true;
                    self.top_tail_ws_pending = false;
                    match c {
                        '#' => self.comment = Some(self.hash_comment()),
                        _ => self.comment = Some(CommentState::Slash),
                    }
                }
                _ => {
                    // Any other trailing token: the gate's decision char
                    // (non-comma) fired the probe -- no taint here, the
                    // root rendering's own strictness decides at the
                    // close. The deferred whitespace arm settles: this
                    // char, not the input's end, answered the scan.
                    self.top_tail_token = true;
                    self.top_tail_ws_pending = false;
                }
            }
            return Ok(());
        }
        let clean = self.top_clean;
        match c {
            c if c.is_whitespace() => {
                // LEADING whitespace (strict or not): the engine's
                // `scroll_whitespaces` consumed it before the first
                // `parse_element`, so the root container's char sits at
                // `index != 0` -- the suffix-probe gate's first arm fires
                // WHATEVER the tail holds and the probe's own success is
                // the root rendering's strictness (`render_closed`'s
                // third arm). No taint here: `\x0b["\f"]` keeps.
                self.top_led = true;
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
                        raw: String::new(),
                        spans: 0,
                        decoded_ends_backslash: false,
                        last_default_append: false,
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
                // Consumed at the top level: the root container's char,
                // when one follows, sits at `index != 0` (the probe
                // gate's first arm).
                self.top_led = true;
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
                // The strict scanner's whitespace set only: any other
                // Unicode whitespace failed `json.loads` (the repair
                // lane decides).
                if !matches!(c, ' ' | '\t' | '\n' | '\r') {
                    self.repair_decided = true;
                }
            }
            '#' | '/' => {
                // Comments are a repair lane (json.loads fails on them).
                // IN-CONTAINER only: a top-level comment sits behind the
                // suffix probe, which keeps the root's own rendering
                // verbatim.
                self.repair_decided = true;
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
                    ']' | ')' => {}
                    ',' => {
                        // A separator where a key was due: `{,}`-shapes,
                        // the engine's empty-object repair territory.
                    }
                    ':' => {
                        // A missing key: the whole pair drops (the
                        // engine's repair lane: `{: 1}` -> `{}`). The
                        // `:` is skipped and the would-be value becomes
                        // a dangling key that drops the same way. Valid
                        // input never lands here: a closed key string
                        // leaves Pos::Colon, so `{"": 1}` passes through
                        // untouched.
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
                    _ => {}
                },
                Pos::Colon => match c {
                    ':' => {
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
                        if self.last_key_empty {
                            self.drop_dangling_member();
                            return self.process(c);
                        }
                        self.repair_decided = true;
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
                        self.repair_decided = true;
                        self.out.push_str(": ");
                        self.pos = Pos::MemberValue;
                        self.process(c)?;
                    }
                    _ => {}
                },
                Pos::Value | Pos::MemberValue => {
                    // A structural char arriving with the member's value
                    // still missing heals the value first (`{"a": }` ->
                    // `{"a": ""}`, the engine's own heal) and completes
                    // the member (the heal fires once); arrays and
                    // tuples heal nothing (`[1, }` -> `[1]`).
                    if self.pos == Pos::MemberValue && matches!(c, ']' | '}' | ')' | ',') {
                        self.repair_decided = true;
                        self.out.push_str("\"\"");
                        self.pos = Pos::AfterValue;
                    }
                    match c {
                        ']' | '}' | ')' => {
                            // A closer where a value was due is a trailing
                            // comma's shape when members exist (the local
                            // rule reads `pos` at the frame's pop); an
                            // empty container's own closer is the strict
                            // `[]`/`{}`.
                            self.close_up(c)
                        }
                        // The engine's garbage-item run (parse_array
                        // items: the item parse started at a `,`):
                        // parse_json's ignore loop consumes it and every
                        // following non-value-start char, the frame's
                        // own closer included. The item slot is not
                        // spent: the next value becomes the item.
                        ',' if self.pos == Pos::Value => {
                            // The comma is the tuple classifier's
                            // separator evidence (the explicit-tuple
                            // scan sees it) even though the item parse
                            // treats it as garbage.
                            self.note_comma();
                            self.garbage_run = true;
                            self.repair_decided = true;
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
                        _ => {}
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
                            // The missing comma: the strict grammar has
                            // no member continuation -- the repair lane
                            // decided the object.
                            self.repair_decided = true;
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
                        _ if Self::is_value_start(c) => {
                            // The missing comma (the repair lane: the
                            // strict grammar has no item continuation).
                            self.repair_decided = true;
                            self.begin_common();
                            self.feed_structure_value(c)?;
                        }
                        _ => {}
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
                self.finish_number(&text, key, text.chars().next_back());
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
                    self.repair_decided = true;
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
                // The object-lane comma-glue SPLIT (the rewind's
                // member-boundary re-entry): a rewind-born word holding
                // the run's ONE absorbed comma, meeting the `:` that
                // proves the post-comma text a KEY. The engine's
                // classify_object_value_comma re-decided that comma a
                // MEMBER separator at the value parse (the bare-key +
                // recoverable-value shape: `{"a": 1,7Z: 8}` -> the
                // engine `{"a": "1", "7Z": 8}`, `{Z7:8,7Z:8` -> the
                // engine `{"Z7": "8", "7Z": 8}`); without the split the
                // rewound word rides the `:` as literal text and the
                // member shape disappears inside one string
                // (`{"a": "1,7Z:8"}`). The value renders up to the
                // comma -- the engine's rewind re-entry, the STRING
                // lane's entry-skip spelling (`render_word`'s: the
                // stray-sign/underscore/dot leads drop, `-8` -> `"8"`,
                // `1e2` stays the string `"1e2"`) -- and the post-comma
                // text re-enters the KEY lane, whose own commit the
                // reprocessed `:` runs. The gate is the engine's own
                // bare-key branch, all of it in hand at the `:` (no
                // lookahead): exactly one absorbed comma (a second
                // comma is the engine's "string" classification -- the
                // bare-key loop stops at it: `{"a": 1,7,9Z: 8}` keeps
                // the run whole, both sides), the key part bare-key
                // spelled (alnum/`_`/`-` runs with trailing whitespace
                // only; a `-`/`.`/container char leads or body stops
                // the engine's loop short of the `:`: `{"a": 1,-b: 2}`
                // keeps the comma), and the object lane (the comma
                // only ever glued there). The RESIDUAL (the half the
                // linear machine cannot see at the `:`): the engine's
                // recoverable-value check needs the post-colon value's
                // own tail to the object's `}` -- a value led by a bare
                // word whose tail holds a quote, or a CUT before the
                // object's close, flips the engine to keep-the-comma
                // where the split already fired (`{"a": 1,b: x"y}`,
                // `{"a": 1,b: x`); those stay the documented divergence
                // below, the fuzz Scan's comma gate keeps them out of
                // the parity assert.
                if !key
                    && from_number_run
                    && c == ':'
                    && matches!(self.frames.last().map(|f| &f.kind), Some(FrameKind::Obj))
                    && text.matches(',').count() == 1
                {
                    let comma = text.find(',').expect("the count proved one comma");
                    let (value_part, key_part) = (text[..comma].trim(), text[comma + 1..].trim());
                    // The engine's bare-key branch: alnum-or-`_` LED, and
                    // the run to the `:` is bare-key chars ONLY (internal
                    // whitespace stops the engine's loop short of the `:`
                    // -- `{"a": 1,7Z x: 8}` keeps the comma -- so the
                    // trimmed key part admits no ws inside either).
                    let key_led = key_part
                        .starts_with(|ch: char| ch.is_alphanumeric() || ch == '_')
                        && key_part
                            .chars()
                            .all(|ch: char| ch.is_alphanumeric() || ch == '_' || ch == '-');
                    if key_led {
                        self.repair_decided = true;
                        let value = render_word(value_part, true);
                        self.out
                            .push_str(&serializer::dumps(&value, self.ensure_ascii));
                        // The second member's bookkeeping: the same
                        // begin_common the structural comma's key start
                        // runs (the member_start lands BEFORE the
                        // separator, so a later dangling-member drop
                        // takes the separator with it).
                        self.begin_common();
                        self.pending = Some(Pending::Word {
                            text: key_part.to_string(),
                            key: true,
                            from_number_run: false,
                        });
                        return self.process(c);
                    }
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
                                self.repair_decided = true;
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
                    // stray-sign drop included). The taint: a run the
                    // strict parser cannot read as one token (`true`/
                    // `false`/`null`/`NaN`/`Infinity`/`-Infinity` load;
                    // `tru`, `None`, `-NaN`, a rewind-born run do not --
                    // the whole-input loads died on the INPUT text).
                    if loads_strict(text.trim()).is_err() {
                        self.repair_decided = true;
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
                        self.top_led = true;
                    }
                    return Ok(());
                }
                self.top_clean = false;
                self.top_led = true;
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
                        self.top_led = true;
                    }
                    return Ok(());
                }
                self.top_clean = false;
                self.top_led = true;
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
                    self.top_led = true;
                    if c == '{' || c == '[' {
                        return self.process(c);
                    }
                    return Ok(());
                }
                let closed = st.feed(c, &mut buf, self.ensure_ascii);
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
            self.repair_decided = true;
            if start <= self.out.len() {
                self.out.truncate(start);
                self.emitted = self.emitted.min(start);
            }
            f.member_count = f.member_count.saturating_sub(1);
            if let Some(p) = f.paren.as_mut() {
                p.elems = p.elems.saturating_sub(1);
            }
            // The rendering from `start` on is gone: deferred-escape
            // marks inside it died with the span.
            self.escape_marks.retain(|m| m.start < start);
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

    /// The one-char drain sites (the empty-object fallback's `{`
    /// removal, the grouping unwrap's `[` removal): every deferred span
    /// at/after the removed char shifts left with it -- the strictly-
    /// empty marks (`EmptyMark`), the armed decision (`EmptyPending`)
    /// and the deferred-escape strings (`EscapeMark`) live in the same
    /// buffer.
    fn shift_deferred_spans(&mut self, pos: usize, delta: isize) {
        for m in &mut self.empty_marks {
            if m.start > pos {
                m.start = (m.start as isize + delta) as usize;
                m.end = (m.end as isize + delta) as usize;
            }
        }
        if let Some(p) = &mut self.empty_pending
            && p.start > pos
        {
            p.start = (p.start as isize + delta) as usize;
            p.end = (p.end as isize + delta) as usize;
        }
        for m in &mut self.escape_marks {
            if m.start > pos {
                m.start = (m.start as isize + delta) as usize;
                m.end = (m.end as isize + delta) as usize;
            }
        }
    }

    fn open_string(&mut self, delim: char, is_key: bool) {
        // A non-`"` delimiter normalizes (the repair lane: the strict
        // grammar knows one delimiter).
        if delim != '"' {
            self.repair_decided = true;
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
            raw: String::new(),
            spans: 0,
            decoded_ends_backslash: false,
            last_default_append: false,
        });
    }

    fn open_container(&mut self, kind: FrameKind) -> Result<(), String> {
        if self.frames.len() >= MAX_NESTING {
            return Err("Input nesting exceeds the supported parser recursion depth.".into());
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
                // The paren taint (sticky): the tuple grammar is never
                // strict-valid, so both fast paths died -- the repair
                // lane decided the whole parse and the deferred-escape
                // strings re-render at close time.
                self.saw_paren = true;
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
            self.repair_decided = true;
            if start <= self.out.len() {
                self.out.truncate(start);
                self.emitted = self.emitted.min(start);
            }
            f.member_count = f.member_count.saturating_sub(1);
            self.pos = Pos::Key;
            // The rendering from `start` on is gone: deferred-escape
            // marks inside it died with the span.
            self.escape_marks.retain(|m| m.start < start);
        }
        // The engine's continuation repair lane ran (json.loads never
        // leaves a dangling key).
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
                self.repair_decided = true;
                return;
            }
            if closer == '}' {
                self.repair_decided = true;
                self.pop_frame();
                return;
            }
        }
        let Some(idx) = self.frames.iter().rposition(|f| f.kind.matches(closer)) else {
            // A closer matching nothing is skipped as garbage.
            self.repair_decided = true;
            return;
        };
        if self.frames.len() > idx + 1 {
            // The mismatched-closer close-up: levels between close
            // (the engine's whole-text re-decision shape).
            self.repair_decided = true;
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
        // THIS array's closing shape, read before the pop lands the
        // after-value position: a closer arriving at a separator-pending
        // position with members on the table is a TRAILING separator
        // where a value was required (`[[] ,]`, `[(,)]`-class) -- the
        // engine's repair lane decided this array (a trailing comma is
        // never strict-valid, so neither fast path can cover it) and
        // its strictly-empty skips fired: the frame's deferred marks
        // drop. Any other closing shape (a value completed at
        // AfterValue, the strict empty `[]`/`{}` closer at member_count
        // 0) keeps them: the array's own body is strict-valid-shaped
        // and the fast paths decide in the element's favor.
        let closed_trailing = self.pos == Pos::Value && frame.member_count > 0;
        // The pop's own repair verdict: a frame closing while a member
        // was still due (a trailing separator: `[1, "\f",]`,
        // `{"a": 1,}`) is never strict-valid -- the repair lane decided
        // the array/object and its string re-normalization ran.
        if frame.member_count > 0 && self.pos != Pos::AfterValue {
            self.repair_decided = true;
        }
        let popped = self.frames.len();
        for m in &mut self.empty_marks {
            if m.frame == popped && m.dropped.is_none() {
                m.dropped = Some(closed_trailing);
            }
        }
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
                    // The empty-object fallback: the whole-text
                    // classifier re-decided the body (the repair lane).
                    self.repair_decided = true;
                    let pos = frame.brace_pos;
                    if pos < self.out.len() && self.out.as_bytes()[pos] == b'{' {
                        self.out.drain(pos..pos + 1);
                        self.emitted = self.emitted.min(pos);
                        self.shift_deferred_spans(pos, -1);
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
                            // The deferred spans live in the same
                            // buffer: every span at/after the removed
                            // bracket shifts left with it (a span the
                            // bracket sat inside cannot exist -- a
                            // marked element's rendering is exactly
                            // `[]`/`{}`/`""`, bracket-free, and a
                            // deferred-escape string's rendering sits
                            // wholly after the group's bracket).
                            self.shift_deferred_spans(pos, -1);
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
            // The root closed: every mark's own array has popped and
            // recorded its closing-shape verdict. The PROPER-close
            // marks resolve KEEP (the fast paths kept their elements:
            // the whole-input `json.loads` -- the closed document is
            // strict-valid up to any junk behind the root -- or the
            // strict-suffix probe) and leave the machine for good; the
            // TRAILING-separator marks stay (their arrays were decided
            // by the repair lane) and drain at every render from now
            // on (`[[] ,]` -> `[]`, also with trailing junk after the
            // root: `[[] ,] x` -> `[]`).
            self.empty_marks.retain(|m| m.dropped != Some(false));
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
        self.arm_empty_pending(matches!(frame.kind, FrameKind::Paren));
    }

    /// Arm the strictly-empty element's one-char skip decision when the
    /// just-completed element's rendering (the parent frame's
    /// `member_start` to here) is exactly `[]`/`{}`/`""` and the parent
    /// is an array-lane frame holding it as an in-flight member -- the
    /// engine's item skip reads the NEXT char. An object-lane parent
    /// keeps its member value (the skip is parse_array's rule alone); a
    /// root frame has no parent to decide about. `elem_paren`: the
    /// element's own rendering came from a popped PAREN frame (the
    /// empty group) -- with any paren ancestor (the element inside a
    /// tuple's parse) it makes the decision NEVER defer (the paren
    /// taint: no fast path can cover a tuple grammar, the repair lane
    /// always decides).
    fn arm_empty_pending(&mut self, elem_paren: bool) {
        if let Some(parent) = self.frames.last()
            && !matches!(parent.kind, FrameKind::Obj)
            && let Some(start) = parent.member_start
            && start <= self.out.len()
        {
            let raw = &self.out[start..];
            let content = raw.strip_prefix(", ").unwrap_or(raw);
            if matches!(content, "[]" | "{}" | "\"\"") {
                let parent_paren = parent.kind == FrameKind::Paren;
                let never_defers = elem_paren
                    || parent_paren
                    || self.frames.iter().any(|f| f.kind == FrameKind::Paren);
                self.empty_pending = Some(EmptyPending {
                    start,
                    end: self.out.len(),
                    was_first: parent.member_count == 1,
                    frame: self.frames.len() - 1,
                    parent_paren,
                    never_defers,
                    ws_seen: false,
                });
            }
        }
    }

    /// The pending decision's drop: the member rendering at `start` (the
    /// deferred separator included) retracts whole and the parent frame's
    /// bookkeeping restores (the completed-member form of
    /// `retract_in_flight_member`).
    fn drop_marked_member(&mut self, start: usize) {
        if let Some(f) = self.frames.last_mut() {
            self.repair_decided = true;
            if start <= self.out.len() {
                self.out.truncate(start);
                self.emitted = self.emitted.min(start);
            }
            f.member_start = None;
            f.member_count = f.member_count.saturating_sub(1);
            if let Some(p) = &mut f.paren {
                p.elems = p.elems.saturating_sub(1);
            }
            // The rendering from `start` on is gone: deferred-escape
            // marks inside it died with the span.
            self.escape_marks.retain(|m| m.start < start);
        }
    }

    /// The number run's terminator: the engine's value lanes over the
    /// (underscore-filtered) run text; a key-position number renders as
    /// the quoted run text (`{12: 1}` -> `{"12": 1}`). `raw_last` is the
    /// run's own last char AFTER the rollback (the engine's parse_number
    /// pops its own trailing `- e E / , +` before the caller looks, so
    /// its `get(-1)` cursor sits on the run's tail): the stray-`...`
    /// rule's exact operand.
    fn finish_number(&mut self, text: &str, key: bool, raw_last: Option<char>) {
        // A number KEY renders quoted (never strict); any run that is
        // not one strict number token (`1_0`, `1,000`, `01`, `1.`, the
        // empty rollback run) is the repair lane too.
        if key {
            // The string-lane entry skip rides the key render too: the
            // run's leading non-alphanumeric garbage drops (`{-12: 1}`
            // -> `{"12": 1}`, `{.5: 1}` -> `{"5": 1}`). An attempt that
            // strips to NOTHING is the engine's empty missing-quote key
            // attempt: the pair drops whole (`{-: 1` -> `{}`).
            self.repair_decided = true;
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
        // The value lane decided: a run that is not one strict number
        // token (`1,000`, `01`, a rollback run the strict grammar cannot
        // end) committed through the repair lane.
        if loads_strict(text).is_err() {
            self.repair_decided = true;
        }
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
        // The deferred-escape resolution (see `EscapeMark`): every held
        // string's rendering is provisional until the lane that decides
        // is known. The verdict is the fast paths' own: the render with
        // the verbatim strings is one strict JSON value (the
        // whole-input `json.loads` fast path and the repair parser's
        // suffix probe keep it -- a complete document, trailing garbage
        // after a valid root and comments behind a root included) and
        // no paren opened anywhere (the tuple grammar is never
        // strict-valid) -- the held spans stand verbatim. Anything else
        // (the cut and drain classes, the paren taint, a repair-lane-
        // decided complete document like `["\x41"]`, the deferred
        // trailing-whitespace arm of the suffix-probe gate) re-renders
        // each held raw source through the lib's escape normalization
        // and re-dumps it canonically.
        //
        // The rebuild is ONE ASCENDING PASS over the marks: the spans
        // are disjoint and armed in ascending order, so the final text
        // is the verbatim segments BETWEEN the spans with each span's
        // resolved rendering spliced in place -- built once, never
        // re-written (a per-span in-place `replace_range` over a large
        // `out` is quadratic in the mark count: every replacement
        // memmoves the whole tail). The re-renders move every `out`
        // coordinate at/after a replaced span's end; the prefix-summed
        // deltas below re-base every coordinate this function reads
        // (the cascade, the strictly-empty drains, the string
        // bookkeeping) in O(log n) per read.
        struct EscapeShifts {
            /// The replaced spans' ends, ascending (the marks' own
            /// order).
            ends: Vec<usize>,
            /// The cumulative length delta at each end (`delta[i]` sums
            /// every replacement up to and including `ends[i]`).
            cum: Vec<isize>,
        }
        impl EscapeShifts {
            /// An `out` coordinate's position in the rebuilt text: every
            /// replaced span at/after it moved the text behind it.
            fn shift(&self, p: usize) -> usize {
                let n = self.ends.partition_point(|&e| e <= p);
                let total = if n == 0 { 0 } else { self.cum[n - 1] };
                (p as isize + total).max(0) as usize
            }
        }
        let mut shifts = EscapeShifts {
            ends: Vec::new(),
            cum: Vec::new(),
        };
        if !self.escape_marks.is_empty()
            && (self.saw_paren
                || self.repair_decided
                || self.top_tail_ws_pending
                || loads_strict(&s).is_err())
        {
            let mut rebuilt = String::with_capacity(s.len() + self.escape_marks.len());
            let mut cursor = 0usize;
            let mut acc: isize = 0;
            for mark in &self.escape_marks {
                if mark.end > s.len() || mark.start >= mark.end {
                    continue;
                }
                // The drain guard: the span must still hold the
                // rendering it armed with (a mid-stream retract took the
                // string whole with its member).
                if self.out.get(mark.start..mark.end) != Some(mark.rendered.as_str()) {
                    continue;
                }
                let replacement =
                    render_repair_lane_string(&mark.raw, mark.delim, self.ensure_ascii);
                if replacement == mark.rendered {
                    continue;
                }
                // Splice: the verbatim segment up to the span, then the
                // span's resolved rendering. The segments between marks
                // are copied exactly once.
                let from = cursor.min(mark.start);
                rebuilt.push_str(&self.out[from..mark.start]);
                rebuilt.push_str(&replacement);
                acc += replacement.len() as isize - (mark.end - mark.start) as isize;
                shifts.ends.push(mark.end);
                shifts.cum.push(acc);
                cursor = mark.end;
            }
            rebuilt.push_str(&self.out[cursor.min(self.out.len())..]);
            s = rebuilt;
        }
        // Every coordinate below is an `out` coordinate; the re-renders
        // moved the text at/after each resolved span's end.
        let shift = |p: usize| shifts.shift(p);
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
        if let Some(start) = dangling.map(shift)
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
                        run.chars().next_back(),
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
                } else if stray
                    && let Some(p) = popped
                    && !run.is_empty()
                    && p.is_alphabetic()
                {
                    // The stray element's own rollback-reprocess: the
                    // engine's skip is a ONE-char decision (the raw
                    // `get(-1)` cursor look drops the `...` and nothing
                    // else) -- the popped char is still the cursor's
                    // char, the item loop re-parses it fresh, and its
                    // falsy-nudge makes it the next element (`[...e` ->
                    // `["e"]`, `[1, ...e` -> `[1, "e"]`, `[...E` ->
                    // `["E"]`). No separator prefix here: the stray
                    // element's own begin put it in place (or the slot
                    // was the array's first), so the word element takes
                    // the stray element's slot as it stands.
                    let value = literal_or_str(&p.to_string());
                    s.push_str(&serializer::dumps(&value, self.ensure_ascii));
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
            // The open string's in-flight content at the cut: when its
            // raw source holds a repair-lane-divergent escape the cut's
            // repair lane re-normalizes it exactly like a closed
            // string's mark (the cut class always decides: a truncated
            // document never strict-loads), then the unterminated-
            // string finalize rstrips the DECODED value (`["\f` ->
            // `["\\f"]`, `["\/` -> `["\\/"]`, `{"k": "\f\n` ->
            // `{"k": "\\f"}`). The streaming decode's own bytes (the
            // canonical form already emitted behind the opening quote)
            // are provisional exactly as a mark's rendering is: the
            // span is replaced by the re-render. A clean tail renders
            // through the escape-tail path as before (the streaming
            // decode's own bytes, the rstrip in
            // `strip_trailing_newline_tail`).
            if raw_source_needs_escape_repair(&st.raw, st.delim) {
                let qs = self.last_string_start.map(shift);
                if let Some(qs) = qs
                    && qs <= s.len()
                {
                    s.truncate(qs);
                }
                s.push_str(&render_open_string_repair_lane(
                    &st.raw,
                    st.delim,
                    self.ensure_ascii,
                ));
            } else {
                st.render_tail(&mut s, self.ensure_ascii);
                s.push('"');
            }
        }
        // The escape-tail heal: a string rendering that ends the stream
        // with a trailing newline-run loses it (the engine's own heal:
        // `{"k": "a\n` -> `{"k": "a"}`, `{"k": "\n"` -> `{"k": ""}`),
        // where a tab or a mid-string newline passes. The strip runs on
        // the closed rendering (the string quoted shut above), so a
        // consumed close quote goes straight back; the cascade may still
        // retract the member whole.
        if !self.frames.is_empty()
            && let Some(qs) = self.last_string_start.map(shift)
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
                .map(shift)
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
                brace_pos: shift(f.brace_pos),
                body_chars: f.body_chars,
                body_nonspace: f.body_nonspace,
                member_begun: f.member_begun,
                body_colon: f.body_colon,
                member_count: f.member_count,
                member_start: f.member_start.map(shift),
                paren: f.paren.map(|p| ParenFrame {
                    brace_pos: shift(p.brace_pos),
                    elems: p.elems,
                    comma_seen: p.comma_seen,
                    elem_groupish: p.elem_groupish,
                }),
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
        // newest first (later spans sit at higher offsets). Every mark
        // still alive at close time drops, and that is the LOCAL rule
        // the engine's own decision procedure reduces to: a mark whose
        // array closed with a TRAILING separator where a value was
        // required stays alive (`dropped` recorded at that array's pop
        // -- a trailing comma is never strict-valid, so no fast path
        // can cover the array and the repair lane's skip fired), and a
        // CUT root (the root never closed) drains every mark still
        // alive, the properly-closed arrays' marks included -- the
        // engine's repair lane decided the whole parse there (the
        // whole-input `json.loads` fast path and the strict-suffix
        // probe both need a closed root or a decodable container). A
        // mark whose array closed PROPERLY was resolved-keep at the
        // root's own close (`pop_frame`'s retain) and never gets here.
        // The one-char decision still armed (`empty_pending`: the
        // element was the input's last token) joins the marks as the
        // newest span: the element was pending at the cut, the cut
        // class. The demotion: a dropped FIRST element promotes the
        // next one to the frame's first slot, and its deferred
        // separator goes with the drop.
        {
            let pending_mark = self.empty_pending.as_ref().map(|p| EmptyMark {
                start: shift(p.start),
                end: shift(p.end),
                was_first: p.was_first,
                frame: p.frame,
                dropped: Some(true), // the element was pending at the cut
            });
            let empty_marks: Vec<EmptyMark> = self
                .empty_marks
                .iter()
                .map(|m| EmptyMark {
                    start: shift(m.start),
                    end: shift(m.end),
                    was_first: m.was_first,
                    frame: m.frame,
                    dropped: m.dropped,
                })
                .collect();
            for mark in pending_mark.iter().chain(empty_marks.iter().rev()) {
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

/// Does this closed string's raw source hold an escape the repair lane's
/// normalization (`parse_string.py`'s `_normalize_escape_sequence`)
/// decides differently from the machine's streaming decode? Exactly
/// three classes:
///
/// - a `\f` or `\/` escape (an ODD backslash run followed by `f` or
///   `/`): the two escapes the repair lane does not accept -- the
///   backslash stays a literal character in the value and the re-dump
///   re-escapes it -- where the streaming decode emitted the canonical
///   form;
/// - an odd backslash run of three or more: the lib's run scan
///   collapses it to ONE literal backslash with the char after the run
///   ordinary, where the streaming decode pairs the run's backslashes
///   and decodes (or rejects) the escape the tail forms;
/// - a backslash before a delimiter that is not the string's own (or
///   the smart-`„`): the lib strips that escape, the streaming decode
///   keeps both chars literally.
///
/// An even backslash run halves identically on both sides and never
/// triggers (`\\f` is parity in both lanes); every accepted escape
/// (`t`, `n`, `r`, `b`, the string's own delimiter, `u` with four hex
/// digits, `x` with two -- the streaming decode's own table) decodes
/// identically and never triggers. The walk is run-aware, not a
/// substring scan: `\\f`'s `f` is ordinary content to the lib, not an
/// escape.
fn raw_source_needs_escape_repair(raw: &str, delim: char) -> bool {
    if !raw.contains('\\') {
        return false;
    }
    let chars: Vec<char> = raw.chars().collect();
    let n = chars.len();
    let mut i = 0usize;
    while i < n {
        if chars[i] != '\\' {
            i += 1;
            continue;
        }
        let mut run = 0usize;
        while i + run < n && chars[i + run] == '\\' {
            run += 1;
        }
        // A closed string's raw source cannot end on an unescaped
        // backslash (the escape state would have consumed the closing
        // delimiter as content); a dangling run is treated as the
        // collapse class, defensively.
        let Some(&e) = chars.get(i + run) else {
            return true;
        };
        if run % 2 == 1
            && (e == 'f'
                || e == '/'
                || e == '\u{201e}'
                || (STRING_DELIMITERS.contains(&e) && e != delim)
                || run >= 3)
        {
            return true;
        }
        // The held partial escape (Class D): a `u`/`x` escape without
        // its full hex tail -- the shape `["\ud83d\u` holds at the cut.
        // The streaming decode's tail DROPS the held partial (the
        // stream rendered `["\ufffd"]` where the engine keeps the
        // literal `\u` after the flushed U+FFFD); the repair lane's
        // walk keeps the backslash literal with the escape char
        // ordinary -- the engine's own observable.
        if run == 1
            && (e == 'u' || e == 'x')
            && !chars
                .get(i + 2..i + 2 + if e == 'u' { 4 } else { 2 })
                .is_some_and(|hex| {
                    hex.len() == if e == 'u' { 4 } else { 2 }
                        && hex.iter().all(|h| h.is_ascii_hexdigit())
                })
        {
            return true;
        }
        i += run + 1;
    }
    false
}

/// Is this closed string's raw source one strict-JSON-decodable body
/// (the `json.loads` string grammar)? Every escape in the strict set
/// (`"`, `\`, `/`, `b`, `f`, `n`, `r`, `t`, `u` with exactly four hex
/// digits) and no raw control char. Anything else failed the whole-
/// input `json.loads` fast path ON THE INPUT TEXT -- the repair lane
/// decided the document and re-normalized every string -- even where
/// the machine's own decode repaired the escape into strict validity
/// (`\x41` decodes to `A`, a raw form feed is whitespace to the
/// machine's skipper): the machine's rendered text loading tells
/// nothing about the input's lane, so the verdict rides the raw
/// source. Used for the `repair_decided` taint, not for the mark
/// trigger (a strict-decodable `\f` escape can still need the mark).
fn raw_source_is_strict_decodable(raw: &str) -> bool {
    let chars: Vec<char> = raw.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        // A raw control char: CPython's "Invalid control character".
        if c < ' ' {
            return false;
        }
        if c != '\\' {
            i += 1;
            continue;
        }
        let Some(&e) = chars.get(i + 1) else {
            return false; // a trailing backslash: never decodable
        };
        match e {
            '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => i += 2,
            'u' => {
                let hex: Vec<char> = chars.get(i + 2..i + 6).unwrap_or(&[]).to_vec();
                if hex.len() == 4 && hex.iter().all(|h| h.is_ascii_hexdigit()) {
                    i += 6;
                } else {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

/// The repair lane's escape normalization over one held string's raw
/// source (`parse_string.py`'s `_normalize_escape_sequence`, mirrored
/// exactly), as the machine's canonical QUOTED rendering -- the CLOSED
/// string's rendering (the deferred-escape mark's resolution): the
/// accepted set is the string's own delimiter, `t`/`n`/
/// `r`/`b`, the backslash run's own collapse, and `u`/`x` followed by
/// the right number of hex digits; `f` and `/` are NOT accepted (the
/// backslash stays a literal character in the value and the re-dump
/// re-escapes it); `u`/`x` without the right hex tail leave the
/// backslash literal with the escape char re-parsed as ordinary
/// content; a delimiter that should not be escaped loses its backslash.
/// The run rules: an even run halves (`run_length % 2`), an odd run of
/// three or more collapses to ONE literal backslash with the char after
/// the run ordinary -- the char the run's own delimiter takes the
/// collapsed backslash INTO the value (the lib's trailing-run while
/// loop, the escaped-delimiter decode), consumed with the run -- a
/// single backslash forms the escape with the char after it.
/// The output is the machine's canonical re-dump of the decoded value:
/// the same `push_escaped_char` table `dumps` uses, the surrogate
/// hold-and-combine semantics included (a valid pair decodes exactly, a
/// stranded high surrogate flushes U+FFFD, the machine's documented
/// lone-surrogate divergence).
fn render_repair_lane_string(raw: &str, delim: char, ensure_ascii: bool) -> String {
    let mut value = repair_lane_value(raw, delim);
    // The lib's own finalize (`_finalize_string_result`): a CLOSED
    // string whose decoded value ends on a newline loses ALL its
    // trailing whitespace.
    if value.ends_with('\n') {
        let keep = value.trim_end_matches(is_py_whitespace).len();
        value.truncate(keep);
    }
    serializer::dumps(&Value::Str(value), ensure_ascii)
}

/// The repair-lane rendering of an OPEN string's in-flight content at
/// the cut: the SAME escape walk as the closed string's
/// (`repair_lane_value`), finalized by the engine's UNTERMINATED-string
/// heal -- the decoded value rstrips UNCONDITIONALLY (Python's
/// whitespace set), the closed string's newline-gated finalize does
/// not. The cut's repair lane always decides (a truncated document
/// never strict-loads), so the open string's divergent escapes resolve
/// here exactly like the closed string's do at the mark (`["\f` ->
/// `["\\f"]`, `["\/` -> `["\\/"]`, `{"k": "x\n\f` -> `{"k": "x\n\\f"}`:
/// the rstrip is the TAIL's, the mid-string newline-escape stands).
fn render_open_string_repair_lane(raw: &str, delim: char, ensure_ascii: bool) -> String {
    let mut value = repair_lane_value(raw, delim);
    let keep = value.trim_end_matches(is_py_whitespace).len();
    value.truncate(keep);
    serializer::dumps(&Value::Str(value), ensure_ascii)
}

/// The shared escape walk (`_normalize_escape_sequence` mirrored): the
/// DECODED VALUE of one string's raw source, the operand both renders
/// finalize. A dangling backslash run at the end of the raw source (the
/// OPEN string's cut shape: `["\`, `["\\\`) is a literal character --
/// run==1 one literal backslash (`["\` -> `["\\"]`, the engine's own
/// observable), an odd run of three or more the same collapse as
/// anywhere (one literal backslash, `["\\\` -> `["\\"]`).
///
/// The walk also tracks the lib's LOW SMART-QUOTE SPAN
/// (`parse_string.py`'s `_push_low_smart_quote_span` /
/// `_pop_low_smart_quote_span`, the `rstring_delimiter` stack gaining
/// the `"\0` sentinel): an ORDINARY `„` -- one the scan meets at its
/// loop top, so never the escape char of a pending backslash, and not
/// hard on a backslash-ending value (the halved even run's own `\`: the
/// entry check `string_acc[-1] != "\\"`) -- pushes one; an ordinary `”`
/// pops one (the scan's low-span `”` arm); a `\` + `"` escape pops one
/// (`_normalize_escape_sequence`'s low-span arm, which tests the
/// LITERAL `"` whatever the string's own delimiter is); and a CHECK-
/// POINT `"` pops one -- the scan's low-span `"` arm at the loop TAIL
/// (`parse_string.py` ~873), which examines the char AFTER a
/// default-appended one: a `"` whose predecessor took the DEFAULT-
/// APPEND path (ordinary content, or a span-less `”`, or a `„` hard on
/// a backslash-ending value) pops the span and appends literally,
/// while a `"` whose predecessor took any other arm (the `„` push, the
/// `”` pop, an escape decode, the run-scan branch, a popped-`"`'s own
/// literal append) is the loop-top CURRENT char and appends itself
/// without popping. `„` is the only
/// pusher -- no other STRING_DELIMITERS member pushes (`'`/`“`/`”` as
/// content are ordinary chars, `”` only pops) -- so the re-decision the
/// span buys is `„`-only. And the re-decision is exactly one: with the
/// sentinel active, the odd-run collapse's trailing-run while loop can
/// no longer take the string's own delimiter INTO the value (the
/// sentinel matches nothing), so the collapsed backslash STANDS and the
/// delimiter joins it as content (the scan's low-span `"` arm appends
/// it literally) -- `["e„\\\"` -> the value `e„\"`, where the plain
/// content drops it (`["e\\\"` -> the value `e"`). (The lib's odd-run
/// arm was dead code for closed strings -- a closed string's raw never
/// ends on the run's own delimiter without more content after -- so
/// this re-decision is observable only through the open-string drain.)
fn repair_lane_value(raw: &str, delim: char) -> String {
    // The decoded value, through the machine's own surrogate carrier:
    // the u/x decode holds a high surrogate for its low half, combines
    // a valid pair, flushes a stranded one U+FFFD (`emit_code`'s tree
    // with the decoded chars left raw; the re-dump below escapes them).
    let mut value = String::new();
    let mut st = StrState {
        delim,
        is_key: false,
        esc: Esc::None,
        high: None,
        raw: String::new(),
        spans: 0,
        decoded_ends_backslash: false,
        last_default_append: false,
    };
    let mut low_smart_quote_spans: usize = 0;
    // Whether the char just consumed took the lib's DEFAULT-APPEND path
    // (`_scan_string_body`'s tail: the char appended at the loop body's
    // end, the post-checks fallen through). The check-point `"`'s
    // low-span pop reads it: a `"` whose PREDECESSOR was default-
    // appended pops the span (parse_string.py's second pop arm, the
    // scan's low-span `"` check running on the char AFTER the appended
    // one); a `"` whose predecessor took any other arm does not -- it
    // is the loop-top CURRENT char there (a `„` push, a `”` pop, an
    // escape decode, the run-scan branch, a popped-`"`'s own literal
    // append) and appends itself without popping.
    let mut last_default_append = false;
    let chars: Vec<char> = raw.chars().collect();
    let n = chars.len();
    let mut i = 0usize;
    while i < n {
        let c = chars[i];
        if c != '\\' {
            st.flush_high_raw(&mut value);
            if c == '\u{201e}' && !value.ends_with('\\') {
                // The push arm (`_append_literal_char` + continue):
                // never the default-append.
                low_smart_quote_spans += 1;
                last_default_append = false;
            } else if c == '\u{201e}' {
                // Hard on a backslash-ending value: the entry check
                // (`string_acc[-1] != "\\"`) fails, the `„` falls
                // through to the default append.
                last_default_append = true;
            } else if c == '\u{201d}' && low_smart_quote_spans > 0 {
                // The pop arm: literal append + continue.
                low_smart_quote_spans -= 1;
                last_default_append = false;
            } else if c == '"' && low_smart_quote_spans > 0 && last_default_append {
                // THE CHECK-POINT POP (the scan's low-span `"` arm at
                // the loop tail): the span pops and the `"` appends
                // literally -- the arm's own `_append_literal_char` +
                // continue is NOT a default-append, so a `"` hard on
                // this one's heels is the loop-top current char and
                // does not pop again (the doubled-quote pop).
                low_smart_quote_spans -= 1;
                last_default_append = false;
            } else {
                // An ordinary char (a span-less `”` included): the
                // default append.
                last_default_append = true;
            }
            value.push(c);
            i += 1;
            continue;
        }
        let mut run = 0usize;
        while i + run < n && chars[i + run] == '\\' {
            run += 1;
        }
        let next = chars.get(i + run).copied();
        if run.is_multiple_of(2) && next != Some(delim) {
            // The even run halves; the char after it is ordinary content.
            st.flush_high_raw(&mut value);
            for _ in 0..run / 2 {
                value.push('\\');
            }
            // The run-scan branch consumed the run (the halving
            // arm's `continue`): the char after it re-enters at the
            // loop top, never through the default-append's post-checks.
            last_default_append = false;
            i += run;
            continue;
        }
        if run >= 3 {
            // The odd-run collapse: one literal backslash, the char
            // after the run ordinary -- EXCEPT a delimiter right after
            // the run: the lib's trailing-run while loop takes the
            // delimiter INTO the value (the collapsed backslash becomes
            // the escaped-delimiter decode, `\\\"` -> `"`), consumed
            // with the run (an even run followed by the string's own
            // delimiter cannot occur in a closed string's raw source:
            // the streaming decode would have closed the string there;
            // an open string's raw source has no trailing delimiter).
            // INSIDE a low smart-quote span the sentinel is the active
            // delimiter, so the while loop cannot take the delimiter in:
            // the collapsed backslash STANDS and the delimiter joins it
            // as content (the scan's low-span `"` arm appends it
            // literally, the span staying active) -- the engine's
            // smart-`„` re-decision (`["e„\\\"` -> the value `e„\"`,
            // `["e\\\"` -> the value `e"`).
            st.flush_high_raw(&mut value);
            if next == Some(delim) {
                if low_smart_quote_spans > 0 {
                    // The span-active join: the delimiter is the
                    // loop-top current char (the sentinel refused the
                    // decode's while loop) and default-appends as
                    // content.
                    value.push('\\');
                    last_default_append = true;
                } else {
                    // The decode took the delimiter in (the
                    // escaped-delimiter decode consumed it with the
                    // run).
                    last_default_append = false;
                }
                value.push(delim);
                i += run + 1;
            } else {
                value.push('\\');
                // The collapse leaves the char after the run to re-enter
                // at the loop top (the run-scan branch's returned
                // next_char).
                last_default_append = false;
                i += run;
            }
            continue;
        }
        // run == 1: the escape char decides.
        let Some(e) = next else {
            // The cut's dangling backslash: a literal character.
            st.flush_high_raw(&mut value);
            value.push('\\');
            break;
        };
        if e == '"' && low_smart_quote_spans > 0 {
            // The low-span arm of `_normalize_escape_sequence`: a `\`+`"`
            // escape inside the span pops it (the value keeps just the
            // `"`, the arms below already spell it that way whatever the
            // string's own delimiter is; the arm tests the LITERAL `"`
            // because the span's machinery is quote-shaped, not
            // delimiter-shaped).
            low_smart_quote_spans -= 1;
        }
        match e {
            _ if e == delim || e == 't' || e == 'n' || e == 'r' || e == 'b' => {
                st.flush_high_raw(&mut value);
                let decoded = match e {
                    't' => '\t',
                    'n' => '\n',
                    'r' => '\r',
                    'b' => '\u{8}',
                    other => other, // the delimiter escape
                };
                value.push(decoded);
                // The escape decode consumed the escape char (`continue`
                // at the normalize call): the char after it re-enters at
                // the loop top.
                last_default_append = false;
                i += 2;
            }
            'u' | 'x' => {
                let num = if e == 'u' { 4 } else { 2 };
                let hex: String = chars
                    .get(i + 2..i + 2 + num)
                    .unwrap_or(&[])
                    .iter()
                    .collect();
                if hex.chars().count() == num && hex.chars().all(|h| h.is_ascii_hexdigit()) {
                    let code = u16::from_str_radix(&hex, 16).unwrap_or(0xFFFD);
                    st.emit_code_raw(code, &mut value);
                    last_default_append = false;
                    i += 2 + num;
                } else {
                    // Not accepted: the backslash stays literal and the
                    // escape char re-parses as ordinary content -- the
                    // lib's post-checks ran on it and fell through, so
                    // its own re-parse at the loop top IS the default
                    // append (the plain arm below sets the flag).
                    st.flush_high_raw(&mut value);
                    value.push('\\');
                    i += 1;
                }
            }
            _ if e == '\u{201e}' || STRING_DELIMITERS.contains(&e) => {
                // A delimiter escape that should not be escaped loses
                // the backslash (the lib's smart-`„` arm; the string's
                // own delimiter was the accepted arm above).
                st.flush_high_raw(&mut value);
                value.push(e);
                last_default_append = false;
                i += 2;
            }
            _ => {
                // Everything else: the backslash is literal text, the
                // escape char ordinary content (its own re-parse sets
                // the flag, as above).
                st.flush_high_raw(&mut value);
                value.push('\\');
                i += 1;
            }
        }
    }
    // A held high surrogate stranded by the end of the raw source
    // flushes U+FFFD (the machine's own heal; the escape state is never
    // set here -- the walker consumes whole escapes in one step).
    st.flush_high_raw(&mut value);
    value
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
/// - a CLOSED string: the lib's own finalize
///   (`_finalize_string_result`): the decoded value ending on a
///   NEWLINE rstrips ALL its trailing whitespace, the same rendered
///   forms (`{"k": "a\n"` -> `{"k": "a"}`, `{"k": "a\t\n"` ->
///   `{"k": "a"}`, `{"k": "a\r\n"` -> `{"k": "a"}`); any other tail
///   keeps (`{"k": "a\n\t"` keeps the tab, `{"k": "a\n "` the space,
///   `{"k": "a\r"` the carriage return).
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
    // Does the rendered tail end on the `\n` escape (an escaped
    // backslash before it is literal content, not a newline)?
    let ends_nl =
        i >= lower + 2 && b[i - 2] == b'\\' && b[i - 1] == b'n' && (i < 3 || b[i - 3] != b'\\');
    let mut any = false;
    if open_ended || ends_nl {
        // The rendered-form rstrip: Python's whitespace set, one tail
        // piece at a time (a raw space, the \t/\n/\r/\f escapes, the
        // 6-byte \uXXXX escapes whose code point isspace()s).
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
    fn the_strictly_empty_drops_are_local_to_their_arrays_closing_shape() {
        // THE LOCAL RULE: a deferred strictly-empty element drops iff
        // its OWN array closed on a trailing separator (a `,` where a
        // value was required -- the repair lane decided that array, no
        // fast path can cover it), or the stream cut with it pending,
        // or the paren taint holds (a paren-derived element, or a
        // paren anywhere enclosing it: the tuple grammar is never
        // strict-valid). A proper array close keeps it (the fast
        // paths do), trailing garbage after a valid root included.
        // Byte-exact against the engine for every family.
        for text in [
            // the trailing-separator drops (the array's own closer)
            "[[] ,]",
            "[{} ,]",
            "[\"\" ,]",
            "[1, [] ,]",
            "[[], [] ,]",
            "[[] , [] ,]",
            "{\"a\": [[] ,]}",
            "{\"a\": [1, [] ,]}",
            "[[] ,] x",
            // the paren-taint drops (round 3's immediate semantics)
            "[(,)]",
            "[( , )]",
            "[1, (,)]",
            "[(,), 1]",
            "[(, ), 1]",
            "(,)",
            "( , )",
            "[{]",
            "[{]}",
            "{\"a\": {(]",
            "{\"a\": {]}}",
            "{]}",
            "{,}",
            "[() ]",
            "[() ,]",
            "[()]x",
            "[1, () ]",
            "[([] , 1)]",
            // the cut drops (the stream ended while pending, or the
            // root never closed: the repair lane decided the parse)
            "[[",
            "[1, [",
            "[[] ,",
            "[[]  ,",
            "[[] , 1",
            "[[] ",
            "[[], [] ,",
            "[[], [] , 1",
            "[[] ,[]",
            "[[] , []",
            "[1, []\r, ",
            "[1, [] , 2 ",
            "[[], [[] , 2], 3",
            "[[] , \"k",
            "[{}\r, ",
            "[\"\"\r, ",
            "[\"\" , 1",
            "[[] , 1 , 2 ",
            // the valid-keeps neighbors (the fast paths kept them)
            "[[], 1]",
            "[[] , 1]",
            "[true, [] , 1]",
            "[{}]",
            "[[] ]",
            "[\"\"]",
            "[( )]",
            "[()]",
            "()",
            "[]",
            "{}",
            "{\"a\": [\"\" , 1]}",
            "{\"a\": [[] , 1]}",
            // the trailing-garbage keeps (the strict-suffix probe
            // re-decided the valid root; the element's own array
            // closed properly)
            "[[] , 1] x",
            "{\"a\": [\"\" , 1]} x",
            "[true, [] , 1] x",
            // the adjacent lanes the same shape class re-decides
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
            // the ws-then-value-start drop: the stall guard ate the
            // whitespace, the char re-parses as the next item
            "[[] x]",
            "[[] x, 1]",
            "[1, [] x, 1]",
            "[[]x]",
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
    fn the_deferred_escape_strings_resolve_at_end() {
        // The weekly-fuzz crash class: a closed string whose raw source
        // holds `\f` or `\/` (the two escapes the repair lane does not
        // accept), an odd backslash run of three or more, or a `\x`,
        // resolves at close time -- the strict lanes keep the verbatim
        // rendering (a complete document, trailing garbage after a valid
        // root included), the repair lane (a cut root, a paren anywhere,
        // a repair-lane-decided complete document) re-renders the raw
        // source through the lib's escape normalization. The distinction
        // survives every separator: `["\f", 1` escapes while
        // `["\f", 1]` keeps.
        for text in [
            // the cut family (the crash artifact's class): the repair
            // lane re-escapes
            "[\"\\f%\"",
            "[\"\\f\", 1",
            "[\"\\f\"",
            "{\"\\f\\/%\": 1",
            "[[ \"\\f\"]",
            "[\"\\/%\"",
            "[\"\\f\\/%\", 1",
            "{\"k\": \"a\\/b\"",
            "[{\"k\": \"\\f\"}, 1",
            "{\"\\f\": 1",
            "[\"a\\fb\"",
            "[\"\\t\\f\"",
            "[\"\\f\\t\"",
            "[\"\\/\\f\"",
            "[\"a\\/b\"",
            // the odd-run collapse (the lib's run scan): the same class
            "[\"\\\\\\f\"",
            "[\"\\\\\\f\", 1",
            "[\"\\\\\\t\"",
            "[\"\\\\\\t\", 1",
            "[\"\\\\\\u0046\"",
            // the x-escape class: the streaming decode's own table
            // carries the repair lane's x-acceptance (`\x41` decodes,
            // `\x%` stays literal) -- parity without a mark
            "[\"\\x%\"",
            "[\"\\x41\"",
            "[\"\\x41\", 1",
            // the strictly-empty-dropping array composition: the marks
            // shift with the empty-element drain
            "[[] , \"\\f\", 1",
            "[1, [] , \"\\f\"",
            // the paren taint: the repair lane decided the complete
            // document too
            "(\"\\f\")",
            "(\"\\f\", 1)",
            "[(\"\\f\", 1)]",
            // the repair-lane-decided complete document (the whole-input
            // loads fails on the bare word)
            "[tru, \"\\f\"]",
            "[tru, \"\\f\"] x",
            // the valid keeps: the strict lanes decided, verbatim
            "[\"\\f\"]",
            "[\"\\f%\"]",
            "{\"\\f\\/%\": 1}",
            "[\"\\f\"]x",
            "[\"\\f\"] // x",
            "[\"\\f\", \"\\/\"]",
            "[1, \"\\f\"]",
            "{\"a\": [\"\\f\"], \"b\": 1}",
            "[[\"\\f\"], [\"\\/\"]]",
            "[\"\\\\f\"]",
            "[\"\\\\\\t\"]",
            "[\"\\\\f\"",
            "{\"k\": \"a\\f\\n\"",
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
            // the rollback's popped ALPHABETIC char re-parses as the
            // next element (the skip ate exactly one char): the
            // e/E-initial word run keeps — the engine's own cursor sits
            // on the char after the final dot, the stray check drops
            // the `...` and the loop parses `e` fresh (the falsy-nudge
            // makes it the word)
            ("[...e", "[\"e\"]"),
            ("[...E", "[\"E\"]"),
            ("[1, ...e", "[1, \"e\"]"),
            ("[[...e", "[[\"e\"]]"),
            ("[[[...e", "[[[\"e\"]]]"),
            ("[1, [...e", "[1, [\"e\"]]"),
            ("[...e]", "[\"e\"]"),
            ("[1, ...e]", "[1, \"e\"]"),
            ("[[...e]]", "[[\"e\"]]"),
            ("[...E]", "[\"E\"]"),
            ("[...e,", "[\"e\"]"),
            ("[ ...e", "[\"e\"]"),
            ("[... e", "[\"e\"]"),
            // not the keep: a second exponent char made the run's own
            // value "…e" (no rollback to replay — `eE` ends the run)
            ("[...eE", "[\"...e\", \"E\"]"),
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
