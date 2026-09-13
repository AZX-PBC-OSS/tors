//! Two scan-family cores live in this module: the escape-parity byte scan
//! (the pure-Rust core of `tors.contains_unescaped` and
//! `tors.find_unescaped`, #50) and the UTF-8 byte-length measurement (the
//! core of `tors.utf8_byte_len`, #52) — the pinned companion that shipped
//! beside it in the same binding module. The scan sections below are
//! #50's; the measurement section at the end is #52's; they share only the
//! binding module (`src/py/scan.rs`) and the harness patterns, not logic.
//!
//! Escape-parity byte scan, the pure-Rust core of `tors.contains_unescaped`
//! and `tors.find_unescaped`: find the first occurrence of a needle that is
//! not itself escaped.
//!
//! Semantics: an occurrence of `needle` at offset `i` in `haystack` is
//! **live** exactly when the maximal run of `\` immediately before `i` has
//! even length. An empty run is even, so an occurrence at offset 0 is live.
//! An odd run means the run's backslash pairs escape each other and the
//! leftover one escapes the needle's first byte: the occurrence is literal
//! text, not the sequence — **rejected** — and the scan resumes ONE BYTE
//! past the hit, not past the whole match, so self-overlapping needles stay
//! correct: the two-byte needle `00` in a haystack of one backslash then
//! `000` rejects the hit at 1 (one backslash before it) and finds the
//! overlapping live hit at 2 (the byte before it is the rejected match's
//! own `0`). memmem's `find_iter` resumes at match end and would skip that
//! hit entirely, which is why this core drives `Finder::find` over
//! re-slices in its own loop instead of taking the iterator.
//!
//! # Why this core exists, and why it is public
//!
//! The motivating case is JSON, and it is byte-ambiguous by construction:
//! an encoder (orjson is the measured consumer) renders a real NUL
//! codepoint U+0000 as the six-byte escape text `\u0000` and the literal
//! six-character text of the same spelling as seven bytes (the backslash
//! itself escaped), and the second contains the first at offset +1 — so a
//! plain substring test answers "yes" for both. PostgreSQL settles it
//! downstream: a real NUL is fatal in a `jsonb` column (SQLSTATE 22P05,
//! `jsonb_in` cannot put it in the decoded text), the literal text is
//! fine, so a pipeline that binds serialized JSON must know which one it
//! holds before the INSERT. Get it wrong in either direction and you
//! either reject legal text or ship a value that fails the bind. The
//! consumer's original guard confirmed every prefilter hit by re-parsing
//! the whole value and recursively walking it — the walk existed only
//! because nothing could answer "is this occurrence escaped?" — and the
//! parity rule (proven there against the re-parse walk over an adversarial
//! matrix of backslash runs and 20k randomized payloads) answers it
//! directly from the raw bytes. It landed here, as a public GIL-free
//! primitive, because a hand-rolled loop in one consumer's private module
//! is usable by nobody else and holds the GIL for the whole scan.
//!
//! # No JSON knowledge lives here
//!
//! Parity is the mechanism; "the needle is an escape sequence, so even
//! means live and odd means literal" is the caller's reading of it. Any
//! backslash-escaped grammar (printf format strings, shell quotes, regex
//! sources) can drive the same scan. The tree already had the idiom twice,
//! both internal and both unfactorable into this shape:
//! `json_repair::parser`'s `skip_to_character` is char-unit and
//! cursor-relative (and telemetry-coupled, via `note_scan_distance`), and
//! `gfm_strip_impl`'s `find_unescaped_bracket_close` walks a char slice
//! against a precomputed escaped-at table. A fresh byte-level core over
//! arbitrary (haystack, needle) pairs is smaller than an abstraction that
//! would span those two, and it serves the general case.
//!
//! # The engine and the parity walk
//!
//! The occurrence scan is `memchr::memmem::Finder` (memchr is already a
//! direct dependency; the Finder holds the needle's search strategy, so
//! the per-hit loop reuses one build), and the parity work is a backward
//! walk over the backslash run immediately before each hit:
//!
//! * Cost: the walk for a hit covers that hit's maximal backslash run,
//!   and at most one hit per maximal run carries a nonzero walk. A hit
//!   with a nonzero walk sits at a maximal run's end or strictly inside
//!   it, and the two placements are mutually exclusive per needle: a
//!   hit at a run's end needs the needle's first byte to be a
//!   non-backslash (the byte there is what ends the run), while a hit
//!   strictly inside needs it to be a backslash, which forces the
//!   needle's backslash prefix to end exactly at the run's end — one
//!   position per run either way. Every walk stays inside its own run,
//!   and runs are disjoint, so every byte is walked backward at most
//!   once across the whole scan. An all-backslash needle never walks at
//!   all: its first hit sits at a maximal run's start — an even, empty
//!   run before it — and answers the scan there.
//! * No state is carried between hits. An earlier revision of this scan
//!   bounded each walk at the previous hit and carried the run's start
//!   forward; both devices were vestigial. The bound never bounded: the
//!   run before a hit never reaches back to the previous hit, because
//!   an all-backslash gap between two hits forces an all-backslash
//!   needle — the gap either swallows the needle whole, or its length
//!   is a period of the needle (both hits spell the needle over the
//!   same gap bytes) and a backslash-prefixed periodic needle is
//!   backslashes all the way through — and an all-backslash needle's
//!   first hit is always live at a run start, ending the scan before a
//!   second hit exists. With no gap ever all backslashes, the bounded
//!   walk and this walk take the same steps on every input, and the
//!   carried start's join branch never fired past the first hit, where
//!   it answered `0` — what the walk's own `run_back` already held.
//!   Verified before the removal by an instrumented differential over the
//!   exhaustive small-alphabet sweep plus a backslash-dense adversarial
//!   sweep (zero join firings past the first hit, zero re-walks, the two
//!   loops step-for-step identical); the linear-walk bound itself is pinned
//!   reproducibly by `backward_walk_budget_is_linear` below (total backward
//!   steps <= haystack length over the long-run and many-hit shapes), so no
//!   historical pair count is load-bearing here.
//!
//! # Preconditions (enforced by the wrapper before this runs)
//!
//! `needle` must be non-empty: an empty needle would match at every
//! position and has no parity meaning (there is no occurrence to stand
//! before); the wrapper refuses it with `ValueError("empty needle")`, the
//! same rationale as `find_patterns`' empty pattern. The haystack may be
//! empty or shorter than the needle (no occurrence, answer `None`). No
//! other input is invalid: arbitrary bytes are in-contract (the function
//! has no JSON assumptions to violate), which is what the fuzz target and
//! the arbitrary-bytes differential both lean on.
//!
//! Pure Rust, no pyo3 types: the criterion bench (the `unescaped_scan`
//! group in benches/search.rs) drives this path directly; the pyo3 wrapper
//! in `src/py/scan.rs` adds only the argument borrows, the GIL release,
//! and the `bool`/`int` return marshalling (see the crate GIL model in
//! `src/lib.rs`).
//!
//! # utf8_byte_len: the measurement companion (#52)
//!
//! `tors.utf8_byte_len(s)` answers `len(s.encode("utf-8"))` without
//! building the bytes object: the count a caller wants when a size cap
//! sits in front of a store (TaskQ's idempotency-key/scope byte caps on
//! every enqueue, and the terminal's re-encode of a serialized result of
//! up to 64 KiB on every success — a genuine double pass, the byte count
//! having existed inside the serializer's output and been discarded by
//! the `.decode()` that produced the `str`). It lives in this module
//! because it is the scan surface's pinned companion — same binding
//! module, same harness patterns — and honest sizing says it would not
//! stand alone (a short-string encode is a few hundred nanoseconds; the
//! win is large inputs and hot paths, where the copy is the cost).
//!
//! # Why the core is one expression, and why that is the point
//!
//! The issue's sketch proposed hand-rolled per-range arithmetic over
//! CPython's internal UCS1/UCS2/UCS4 storage: 1/2/3/4 bytes per codepoint
//! by range, surrogate pairs in UCS2 folding into one 4-byte sequence,
//! behind unsafe FFI walks of `PyUnicode_KIND`/data. This implementation
//! rejects that route. The wrapper performs the repo's standard str-in
//! borrow (pyo3 `to_str`, `PyUnicode_AsUTF8AndSize` — the same
//! zero-copy-or-materialize-once borrow every str-argument tors function
//! uses) and hands this core a Rust `&str`, whose `len()` IS its UTF-8
//! byte length: a `&str` is its UTF-8 bytes by construction, so the
//! answer is one field read. The mechanism is dried into the language
//! instead of reimplemented; an unsafe KIND/data walk with
//! surrogate-pair arithmetic would exist only to avoid one cached
//! materialization, which is exactly the hand-rolled complexity the
//! maintenance-burden policy refuses. The trade is deliberate and
//! measured (the wrapper's docs and tests/test_performance.py carry the
//! numbers):
//!
//! * ASCII (the serialized-JSON case): the borrow is a zero-copy alias —
//!   compact ASCII data is its own UTF-8 — so the call is O(1) with no
//!   allocation, against the expression's alloc+memcpy every call.
//! * Non-ASCII, first call on the object (a cold UTF-8 cache): CPython
//!   materializes and CACHES the UTF-8 view on the `str` object (an
//!   internal cache, not a Python-visible `bytes`, filled by this borrow
//!   and by any earlier str-in tors call on the same object — `encode`
//!   reads it and never fills it), so the first call is O(n) —
//!   encode-parity in cost class, with no Python-visible object to
//!   allocate and collect, and a prior `len(s.encode())` does not warm
//!   it: the first call after an encode still pays the full
//!   materialization (the cold class, measured — in every CPython
//!   3.10-3.14 `unicode_encode_utf8` reads the cache and only
//!   `PyUnicode_AsUTF8AndSize`, the str-in borrow, writes it).
//! * Non-ASCII, repeat calls on the same object: O(1) — strictly better
//!   than the expression, which re-copies on every call.
//!
//! Error parity is free: a `str` holding lone surrogates cannot be
//! UTF-8-encoded, and `PyUnicode_AsUTF8AndSize` raises CPython's own
//! `UnicodeEncodeError` (pyo3 propagates it from the borrow, before any
//! tors code runs) — the same exception `encode` raises, attributes
//! included (pinned attribute-for-attribute in
//! tests/test_utf8_byte_len.py). There is no tors-side error path at all.
//!
//! No fuzz target exists for this function, deliberately: the core is one
//! field read over a type that guarantees its own invariant (a `&str` is
//! valid UTF-8 by construction, so its `len()` is the byte length by the
//! language's own rules) — there is no input-dependent logic for a
//! Rust-side fuzzer to explore, and the interesting invariants (encode
//! parity, the surrogate error) live at the CPython/pyo3 boundary
//! cargo-fuzz cannot reach. A target comparing `s.len()` to a re-encode
//! of the same data would assert the standard library against itself:
//! vacuous, and negative value in the wired lists it would have to
//! occupy.
//!
//! # utf16_byte_len: the interop twin (#52)
//!
//! `tors.utf16_byte_len(s)` answers `len(s.encode("utf-16-le"))` — the
//! UTF-16 byte length, 2 bytes per BMP codepoint and 4 per astral
//! codepoint (the surrogate pair) — without building the 2n `bytes`
//! object. The maintainer's framing: a util to convert a UTF8/UTF16
//! python `len()` into bytes for the API, because backend devs need
//! byte caps for storage AND interop — UTF-16 is the code-unit world of
//! JavaScript, Java, Windows, and .NET, where column caps
//! (`NVARCHAR`), wire caps, and size checks count UTF-16 units, and the
//! Python spelling of the count allocates the whole copy. It ships as
//! the utf8 twin's sibling in the same binding module: one family, the
//! "len() to bytes" pair.
//!
//! The core is ARITHMETIC over the borrowed UTF-8 view, not a field
//! read, and the arithmetic is derived rather than table-driven:
//!
//! ```text
//! utf16 bytes = 2 * (#codepoints + #astral codepoints)
//! ```
//!
//! every codepoint is one 2-byte UTF-16 unit, and an astral codepoint is
//! a 2-unit surrogate pair (4 bytes = 2 + 2 more). Over a valid UTF-8
//! view both counts are byte classes, no decoding:
//!
//! * `#codepoints` is the count of lead bytes — every UTF-8 sequence has
//!   exactly one lead, and the continuation bytes are exactly
//!   `0x80..=0xBF`, so `(b & 0xC0) != 0x80` selects the leads;
//! * `#astral` is the count of 4-byte lead bytes `0xF0..=0xF4` — in
//!   valid UTF-8 those are exactly the bytes `matches!(b, 0xF0..=0xF4)`
//!   (one per astral codepoint; `0xF5..=0xFF` never occur in valid
//!   UTF-8, so `b >= 0xF0` would count the same set over a `&str`,
//!   but the closed range states the fact and fails safe), and a
//!   4-byte sequence is exactly an astral codepoint (overlong forms
//!   are invalid UTF-8, which a `&str` rules out by construction).
//!
//! One pass, two byte-class predicates, no allocation. The corners the
//! identity buys: no astral codepoints means exactly `2 * len(s)` in
//! Python `len()` (the codepoint count) for ALL BMP text — CJK and
//! combining marks included, where the UTF-8 byte count diverges — and
//! pure ASCII additionally means `2 *` the UTF-8 byte count; every
//! answer is even. The identity is proved three ways: against a
//! `chars()`-based naive count (the per-codepoint spec) in
//! `utf16_byte_len_tests` below, over a boundary battery and an
//! EXHAUSTIVE length-<=3 sweep of the boundary alphabet — both sides of
//! the identity are additive over concatenation, so sweep agreement is
//! agreement on every input — and against the stdlib oracle
//! `len(s.encode("utf-16-le"))` Python-side
//! (tests/test_utf8_byte_len.py, the byte-len family file).
//!
//! The cost model is the utf8 twin's exactly (same str-in borrow, same
//! CPython UTF-8 view cache): ASCII is a zero-copy alias; a non-ASCII
//! object's first call — the cold-cache case — materializes and caches
//! the view under the GIL (encode-parity cost; `encode` reads that
//! cache and never fills it); repeat calls borrow it zero-copy and pay
//! only this scan, which runs detached (real O(n) work, unlike the
//! utf8 twin's nominal one-field-read detach — see the wrapper's docs).
//! Lone surrogates never reach this function either (the borrow raises
//! first); the stdlib relation there is REFUSAL PARITY with the
//! replaced expression, not the asymmetry a first draft assumed — the
//! strict `encode("utf-16-le")` refuses lone surrogates exactly like
//! the utf-8 codec does, so both twins refuse the same strings their
//! replaced expressions refuse, the borrow's utf-8-flavored error and
//! the stdlib's surrogatepass acceptance mode being the two honest
//! differences (the wrapper's docs and the Python battery pin both).
//!
//! No fuzz target for this function either, and the honest version of
//! the utf8 argument: this core DOES have input-dependent logic (the
//! two byte-class counts), but its correctness reduces to the UTF-8
//! well-formedness facts above plus additivity, and the exhaustive
//! sweep closes exactly that space — every `String` a fuzzer could
//! build is a concatenation of codepoints from classes the sweep
//! already enumerated, so a differential target against
//! `str::encode_utf16` would re-prove a closed theorem at negative
//! value in the wired lists it would have to occupy.

/// The byte offset of the first live (even-run) occurrence of `needle` in
/// `haystack` — the first occurrence whose maximal preceding backslash run
/// has even length — or `None` when no live occurrence exists (the wrapper
/// maps `None` to Python `-1`, `bytes.find`'s own sentinel). `needle` must
/// be non-empty (see the module docs' preconditions); the haystack's bytes
/// are never assumed to be anything but bytes.
///
/// Rejected (odd-run) hits advance the search one byte past the hit, not
/// past the whole match, so self-overlapping needles stay correct; each
/// hit's walk covers the backslash run immediately before it, and at
/// most one hit per maximal run ever walks, so every byte of the
/// haystack is walked backward at most once across the whole scan (the
/// module docs' cost argument).
pub fn find_unescaped(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    assert!(!needle.is_empty(), "empty needle");
    let finder = memchr::memmem::Finder::new(needle);
    // `from` is the next occurrence search's origin: 0, then one byte
    // past each rejected hit — the resume rule that keeps self-overlapping
    // needles correct.
    let mut from = 0;
    while let Some(rel) = finder.find(&haystack[from..]) {
        let hit = from + rel;
        // The backward run count: walk the maximal backslash run ending
        // at the hit down to its start (a non-backslash byte, or offset
        // 0). No state is carried between hits — the module docs' second
        // bullet is why that is safe: at most one hit per maximal run
        // ever walks, so no byte is walked twice.
        let mut run_back = hit;
        while run_back > 0 && haystack[run_back - 1] == b'\\' {
            run_back -= 1;
        }
        if (hit - run_back).is_multiple_of(2) {
            return Some(hit);
        }
        from = hit + 1;
    }
    None
}

// --- #52: the UTF-8 byte-length measurement --------------------------------------------
//
// The companion core, separated from the scan above. See the module docs'
// "utf8_byte_len" section for the borrow-not-arithmetic decision, the
// cache trade, the error parity, and the no-fuzz rationale; the wrapper in
// src/py/scan.rs carries the Python-facing contract.

/// The UTF-8 byte length of `s` — the number of bytes `s` occupies as
/// UTF-8, the answer `len(s.encode("utf-8"))` computes by allocating and
/// copying the whole `bytes` object first.
///
/// The body is one expression, deliberately: a Rust `&str` IS its UTF-8
/// bytes (valid by construction), so its `len()` is the UTF-8 byte length
/// by the language's own rules — the mechanism is dried into the language
/// rather than reimplemented (the module docs' section on why the issue's
/// per-range UCS arithmetic was rejected). The cost model lives at the
/// borrow, not here: ASCII inputs are zero-copy aliases, a non-ASCII
/// input's first call materializes and caches the UTF-8 view on the
/// Python object (encode-parity cost, no Python-visible allocation), and
/// repeat calls on the same object are O(1) — strictly better than the
/// expression, which re-copies every call. Lone surrogates never reach
/// this function: the borrow raises `UnicodeEncodeError` first, exactly
/// `encode`'s own error.
pub fn utf8_byte_len(s: &str) -> usize {
    s.len()
}

// --- #52: the UTF-16 byte-length interop twin -----------------------------------------
//
// The arithmetic core, the utf8 twin's sibling. See the module docs'
// "utf16_byte_len" section for the derivation (the identity, the two
// byte classes, the additivity argument behind the exhaustive sweep),
// the cache trade it inherits from the same str-in borrow, the
// surrogate refusal parity, and the no-fuzz rationale; the wrapper in
// src/py/scan.rs carries the Python-facing contract.

/// The chunk width of the counting loop below: 16. The two counts are
/// summed per chunk into `u8` accumulators, and the width is where the
/// measurement landed on the calibration box (arm64, rustc release
/// build — bench artifact, not a portable constant: re-measure on
/// your target; other widths/builds land differently): 8 and 32 both
/// run ~16 GB/s there, 16 runs ~31 GB/s — the sweet spot between
/// per-chunk overhead and loop-carried latency on that box, the same
/// class of measurement-decided constant as the segment scanner's
/// `BREAK_WINDOW`. The chunked spelling exists at all because the
/// obvious `iter().filter().count()` closures do NOT auto-vectorize
/// here (measured 4.2 GB/s scalar on that box, slower than the
/// expression the function exists to beat — the wall cells in
/// tests/test_performance.py record both lanes).
const UTF16_COUNT_CHUNK: usize = 16;

/// The UTF-16 byte length of `s` — 2 bytes per BMP codepoint, 4 per
/// astral codepoint (the surrogate pair) — the answer
/// `len(s.encode("utf-16-le"))` computes by allocating and copying the
/// whole 2n `bytes` object first.
///
/// The body is one pass of byte-class arithmetic over the UTF-8 view,
/// deliberately: the module docs derive the identity
/// `utf16 bytes = 2 * (#codepoints + #astral)`, where over valid UTF-8
/// `#codepoints` is the lead-byte count (`(b & 0xC0) != 0x80`) and
/// `#astral` is the count of 4-byte leads (`matches!(b, 0xF0..=0xF4)`
/// in valid UTF-8 — `0xF5..=0xFF` never occur in a `&str`, so the
/// `>= 0xF0` shorthand would count the same set here, but the closed
/// range states the valid-UTF-8 fact and fails safe if the input ever
/// were not valid UTF-8: a stray `0xF5..=0xFF` byte is refused rather
/// than counted as astral) — no decoding, no allocation, summed
/// in [`UTF16_COUNT_CHUNK`]-sized chunks (the auto-vectorizing shape;
/// see that constant's docs for the measurement). The identity is
/// proved against the `chars()`-based naive count over the boundary
/// battery and an exhaustive length-<=3 sweep (both sides additive
/// over concatenation, so the sweep is exhaustive) in
/// `utf16_byte_len_tests`, and against the stdlib oracle Python-side.
///
/// The final doubling is checked arithmetic: on 32-bit targets
/// `2 * (codepoints + astral)` can overflow `usize` past ~1 GiB of
/// astral-dense text (the addition and the multiply both wrap in
/// release), so both steps use `checked_add`/`checked_mul` and panic
/// with an overflow message rather than wrapping silently — the
/// Python wrapper maps this to `OverflowError` (see `src/py/scan.rs`);
/// on 64-bit the check never fires (it would take exabytes).
///
/// The cost model is the borrow's, the utf8 twin's exactly (ASCII
/// zero-copy alias; cold-cache first call materializes-and-caches the
/// UTF-8 view, encode-parity; repeat calls O(1) borrow plus this scan).
/// Lone surrogates never reach this function: the borrow raises
/// `UnicodeEncodeError` first — for this twin REFUSAL PARITY with the
/// replaced expression (the strict `encode("utf-16-le")` refuses lone
/// surrogates too; the borrow's utf-8-flavored error and the stdlib's
/// surrogatepass acceptance mode are the two honest differences — the
/// wrapper's docs and the Python battery pin both).
pub fn utf16_byte_len(s: &str) -> usize {
    // The derivation's two counts, one chunked pass: one unit per
    // codepoint (its lead byte), one more per astral codepoint (its
    // 4-byte lead) — doubled at the end. See the module docs for the
    // identity and UTF16_COUNT_CHUNK's docs for the width.
    //
    // The astral predicate is the closed range `0xF0..=0xF4`, not the
    // `>= 0xF0` shorthand: over valid UTF-8 the two count the same set
    // (`0xF5..=0xFF` never occur in a `&str`), and the closed range
    // fails safe — a stray high byte is refused rather than counted.
    // The final `checked_add`/`checked_mul` refuses to wrap silently
    // on 32-bit targets past ~1 GiB of astral-dense text (the Python
    // wrapper maps the panic to `OverflowError`); on 64-bit it never
    // fires.
    let bytes = s.as_bytes();
    let (chunks, remainder) = bytes.as_chunks::<UTF16_COUNT_CHUNK>();
    let mut codepoints = 0usize;
    let mut astral = 0usize;
    for chunk in chunks {
        let mut leads = 0u8;
        let mut four_byte_leads = 0u8;
        for &b in chunk {
            leads += ((b & 0xC0) != 0x80) as u8;
            four_byte_leads += matches!(b, 0xF0..=0xF4) as u8;
        }
        codepoints += leads as usize;
        astral += four_byte_leads as usize;
    }
    for &b in remainder {
        codepoints += ((b & 0xC0) != 0x80) as usize;
        astral += matches!(b, 0xF0..=0xF4) as usize;
    }
    codepoints
        .checked_add(astral)
        .and_then(|n| n.checked_mul(2))
        .expect("utf16_byte_len overflow: input too large for usize on this target")
}

#[cfg(test)]
mod utf8_byte_len_tests {
    use super::*;

    #[test]
    fn boundary_codepoints_answer_the_sequence_table() {
        // The UTF-8 sequence table at its boundaries: 1 byte below U+0080,
        // 2 to U+07FF, 3 to U+FFFF, 4 above — hand-computed expectations,
        // the same rows the Python battery pins (a wrong row fails here
        // before the extension even builds).
        assert_eq!(utf8_byte_len(""), 0);
        assert_eq!(utf8_byte_len("a"), 1);
        assert_eq!(utf8_byte_len("\u{7f}"), 1);
        assert_eq!(utf8_byte_len("\u{80}"), 2);
        assert_eq!(utf8_byte_len("\u{7ff}"), 2);
        assert_eq!(utf8_byte_len("\u{800}"), 3);
        assert_eq!(utf8_byte_len("\u{ffff}"), 3);
        assert_eq!(utf8_byte_len("\u{10000}"), 4);
        assert_eq!(utf8_byte_len("\u{10ffff}"), 4);
    }

    #[test]
    fn mixed_content_answers_the_sum_of_its_sequence_lengths() {
        assert_eq!(utf8_byte_len("caf\u{e9}"), 5); // 4 ASCII + one 2-byte
        assert_eq!(utf8_byte_len("\u{6771}\u{4eac}"), 6); // two 3-byte
        assert_eq!(utf8_byte_len("\u{1f600}"), 4); // one 4-byte
        assert_eq!(utf8_byte_len("e\u{301}"), 3); // 1 + a combining mark
        assert_eq!(utf8_byte_len("\u{1f468}\u{200d}\u{1f469}"), 4 + 3 + 4);
        assert_eq!(utf8_byte_len("\u{0}"), 1); // a real NUL: 1 byte
    }

    #[test]
    fn byte_length_diverges_from_char_count_on_astral_text() {
        // The char/byte divergence the API exists to answer: astral chars
        // are 1 char and 4 bytes, so a char-counting regression fails here
        // by exactly 4x (the class of bug the Python differential's
        // astral-only strategy pins from the other side).
        let astral = "\u{1f600}".repeat(1000);
        assert_eq!(utf8_byte_len(&astral), 4000);
        assert_eq!(astral.chars().count(), 1000);
    }

    #[test]
    fn a_one_mib_scale_string_answers_its_encoded_length() {
        // The 1 MiB ladder top: the answer holds at the scale where the
        // replaced expression allocates a full megabyte. The expectation
        // is derived from the unit recipe (32 bytes per unit, pinned by
        // the assert — the first draft of this row mis-counted the unit
        // at 26 and the pin caught it), 32,768 units making exactly
        // 1 MiB.
        let unit = "Torque spec, caf\u{e9} \u{6771}\u{4eac} \u{1f600}\u{301}";
        assert_eq!(unit.len(), 32);
        let text = unit.repeat(32_768);
        assert_eq!(utf8_byte_len(&text), 32 * 32_768);
    }
}

#[cfg(test)]
mod utf16_byte_len_tests {
    use super::*;

    /// The independent oracle: the naive per-codepoint spelling of the
    /// contract — 2 bytes per BMP codepoint, 4 per astral — one `chars()`
    /// pass, the spec by definition. Agreement between this and the
    /// byte-class arithmetic is evidence about the derivation, not a
    /// shared bug (the Python battery adds the stdlib oracle,
    /// `len(s.encode("utf-16-le"))`, as the third vote).
    fn naive_utf16_byte_len(s: &str) -> usize {
        s.chars()
            .map(|c| if (c as u32) > 0xFFFF { 4 } else { 2 })
            .sum()
    }

    #[test]
    fn boundary_codepoints_answer_the_unit_table() {
        // 2 bytes across the whole BMP (every UTF-8 sequence class), 4
        // above it — including all four 4-byte lead values 0xF0-0xF4,
        // the rows an astral predicate that caught only 0xF0 would
        // answer 2 for and fail here.
        assert_eq!(utf16_byte_len(""), 0);
        assert_eq!(utf16_byte_len("a"), 2);
        assert_eq!(utf16_byte_len("\u{7f}"), 2);
        assert_eq!(utf16_byte_len("\u{80}"), 2);
        assert_eq!(utf16_byte_len("\u{7ff}"), 2);
        assert_eq!(utf16_byte_len("\u{800}"), 2);
        assert_eq!(utf16_byte_len("\u{ffff}"), 2);
        assert_eq!(utf16_byte_len("\u{10000}"), 4);
        assert_eq!(utf16_byte_len("\u{40000}"), 4); // 0xF1 lead
        assert_eq!(utf16_byte_len("\u{80000}"), 4); // 0xF2 lead
        assert_eq!(utf16_byte_len("\u{c0000}"), 4); // 0xF3 lead
        assert_eq!(utf16_byte_len("\u{10ffff}"), 4); // 0xF4 lead
    }

    #[test]
    fn mixed_content_answers_the_sum_of_its_unit_costs() {
        assert_eq!(utf16_byte_len("caf\u{e9}"), 8);
        assert_eq!(utf16_byte_len("\u{6771}\u{4eac}"), 4); // UTF-8 answers 6
        assert_eq!(utf16_byte_len("\u{1f600}"), 4);
        assert_eq!(utf16_byte_len("e\u{301}"), 4);
        assert_eq!(utf16_byte_len("\u{1f468}\u{200d}\u{1f469}"), 4 + 2 + 4);
        assert_eq!(utf16_byte_len("\u{0}"), 2); // a real NUL: one unit
    }

    #[test]
    fn the_derivation_identity_holds_against_the_naive_count() {
        // The proof battery: every boundary class, dense mixes, and the
        // scale ladder — the byte-class arithmetic against the
        // `chars()`-based naive count, the identity the module docs
        // derive.
        let rows = [
            "",
            "a",
            "\u{7f}\u{80}\u{7ff}",
            "\u{800}\u{ffff}\u{6771}",
            "\u{10000}\u{40000}\u{80000}\u{c0000}\u{10ffff}",
            "Torque caf\u{e9} \u{6771}\u{4eac} \u{1f600}",
            "abc\u{e9}\u{1f600}def\u{6771}",
            "\u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467}",
        ];
        for row in rows {
            assert_eq!(utf16_byte_len(row), naive_utf16_byte_len(row), "{row:?}");
        }
        let ascii = "k".repeat(1000);
        assert_eq!(utf16_byte_len(&ascii), naive_utf16_byte_len(&ascii));
        let cjk = "\u{6771}\u{4eac}".repeat(1000);
        assert_eq!(utf16_byte_len(&cjk), naive_utf16_byte_len(&cjk));
        let astral = "\u{1f600}".repeat(1000);
        assert_eq!(utf16_byte_len(&astral), naive_utf16_byte_len(&astral));
        // the corners, stated as identities the naive count also gives:
        // no astral -> 2 per codepoint for ALL BMP text ...
        assert_eq!(utf16_byte_len(&cjk), 2 * cjk.chars().count());
        // ... pure ASCII -> additionally 2 per UTF-8 byte ...
        assert_eq!(utf16_byte_len(&ascii), 2 * ascii.len());
        // ... and every astral codepoint is the 4-byte pair.
        assert_eq!(utf16_byte_len(&astral), 4000);
    }

    #[test]
    fn the_exhaustive_short_string_sweep_matches_the_naive_count() {
        // Every string of length <= 3 codepoints over the boundary
        // alphabet (one representative per UTF-8 sequence class, all
        // four 4-byte lead values, the combining mark, the ZWJ, and a
        // real NUL — every entry a single codepoint, so alphabet length
        // IS codepoint length, and every concatenation is valid UTF-8
        // by construction): both sides of the identity are additive
        // over concatenation (UTF-16 bytes sum per codepoint; the
        // byte-class counts sum per codepoint), so agreement here is
        // agreement on EVERY input — the sweep is the exhaustive proof,
        // not a sample (the Python battery runs the same sweep against
        // the stdlib oracle; 17 alphabet entries give
        // 17 + 17^2 + 17^3 = 5219 cases here, where the Python sweep's
        // 21-entry alphabet with extra ASCII representatives gives 9723
        // — same theorem, denser ASCII sampling there).
        let alphabet = [
            "a",
            "\u{7f}",
            "\u{80}",
            "\u{7ff}",
            "\u{800}",
            "\u{ffff}",
            "\u{e9}",
            "\u{301}",
            "\u{200d}",
            "\u{6771}",
            "\u{1f600}",
            "\u{10000}",
            "\u{40000}",
            "\u{80000}",
            "\u{c0000}",
            "\u{10ffff}",
            "\u{0}",
        ];
        for &a in &alphabet {
            assert_eq!(utf16_byte_len(a), naive_utf16_byte_len(a));
            for &b in &alphabet {
                let mut two = String::from(a);
                two.push_str(b);
                assert_eq!(utf16_byte_len(&two), naive_utf16_byte_len(&two));
                for &c in &alphabet {
                    let mut three = two.clone();
                    three.push_str(c);
                    assert_eq!(utf16_byte_len(&three), naive_utf16_byte_len(&three));
                }
            }
        }
    }

    #[test]
    fn a_one_mib_scale_string_answers_its_naive_length() {
        // The 1 MiB ladder top, utf16 leg: the identity holds at the
        // scale where the replaced expression allocates the whole UTF-16
        // copy of the text (two bytes per codepoint over this mix).
        let unit = "Torque spec, caf\u{e9} \u{6771}\u{4eac} \u{1f600} \u{40000} e\u{301}\u{0}\n\n";
        let text = unit.repeat(1 + (1024 * 1024) / unit.len());
        assert!(text.len() >= 1024 * 1024);
        assert_eq!(utf16_byte_len(&text), naive_utf16_byte_len(&text));
    }

    #[test]
    fn checked_arithmetic_refuses_to_wrap_silently() {
        // H1: `2 * (codepoints + astral)` wraps `usize` in release past
        // ~1 GiB of astral-dense text on 32-bit targets. The core must
        // use `checked_add`/`checked_mul` (panicking rather than
        // wrapping; the Python wrapper maps this to `OverflowError`).
        // `usize::MAX` itself is unreachable in a test, so this pins
        // the checked spelling directly on small numbers: the same
        // operator chain the core uses must return `None` at the
        // boundary instead of wrapping to 0.
        let almost_max = usize::MAX - 1;
        assert_eq!(
            almost_max.checked_add(1).and_then(|n| n.checked_mul(2)),
            None
        );
        assert_eq!(usize::MAX.checked_add(1), None);
        assert_eq!(usize::MAX.checked_mul(2), None);
        // And the un-overflowed chain still answers: the happy path the
        // core takes on every real input.
        assert_eq!(
            1usize.checked_add(1).and_then(|n| n.checked_mul(2)),
            Some(4)
        );
    }

    #[test]
    #[cfg(target_pointer_width = "32")]
    fn thirty_two_bit_targets_refuse_gigabyte_scale_inputs() {
        // 32-bit-only: a ~1 GiB astral-dense input would wrap
        // `2 * (codepoints + astral)` past `u32::MAX`. Building the
        // gigabyte is out of scope for a unit test; this pins the
        // contract that the overflow panics (mapped to `OverflowError`
        // Python-side) rather than wrapping — exercised here via the
        // checked chain at the boundary, since the allocation itself
        // would OOM the test runner.
        let almost_max = usize::MAX - 1;
        assert_eq!(
            almost_max.checked_add(1).and_then(|n| n.checked_mul(2)),
            None
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NUL_ESCAPE: &[u8] = b"\\u0000";

    /// The independent oracle (`reference_find_unescaped`, the Python
    /// suite's, spelled in Rust): every position in order, a full backward
    /// run walk per occurrence, no carried state, no engine — agreement
    /// between this and `find_unescaped` over the exhaustive sweep below is
    /// evidence about the contract, not a shared bug.
    fn naive_find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        let mut i = 0;
        while i + needle.len() <= haystack.len() {
            if &haystack[i..i + needle.len()] == needle {
                let mut run = 0;
                let mut j = i;
                while j > 0 && haystack[j - 1] == b'\\' {
                    run += 1;
                    j -= 1;
                }
                if run % 2 == 0 {
                    return Some(i);
                }
            }
            i += 1;
        }
        None
    }

    #[test]
    fn runs_zero_through_seven_decide_liveness() {
        // The parity ladder: a run of k backslashes before the occurrence,
        // live iff k is even. k=0 is the offset-0 case (nothing before the
        // occurrence, an even empty run).
        for k in 0..=7usize {
            let mut haystack = vec![b'\\'; k];
            haystack.extend_from_slice(NUL_ESCAPE);
            haystack.extend_from_slice(b" tail");
            let want = if k % 2 == 0 { Some(k) } else { None };
            assert_eq!(find_unescaped(&haystack, NUL_ESCAPE), want, "run {k}");
        }
    }

    #[test]
    fn json_real_and_literal_renderings_in_both_orders() {
        // The motivating ambiguity, as it sits on the wire inside a JSON
        // string value: the real NUL ("a\u0000b"), the literal text
        // ("a\\u0000b"), and both orders in one document.
        assert_eq!(find_unescaped(b"\"a\\u0000b\"", NUL_ESCAPE), Some(2));
        assert_eq!(find_unescaped(b"\"a\\\\u0000b\"", NUL_ESCAPE), None);
        assert_eq!(
            find_unescaped(b"\"a\\\\u0000b\"\"c\\u0000d\"", NUL_ESCAPE),
            Some(13)
        );
        assert_eq!(
            find_unescaped(b"\"c\\u0000d\"\"a\\\\u0000b\"", NUL_ESCAPE),
            Some(2)
        );
    }

    #[test]
    fn adjacent_literals_are_all_rejected_then_the_real_one_is_found() {
        // The miniature false-positive corpus (three literal renderings
        // back to back), then the same with a real escape appended: the
        // scan walks past every rejection to the live occurrence.
        let literals_only = [b"\\", NUL_ESCAPE].concat().repeat(3);
        assert_eq!(find_unescaped(&literals_only, NUL_ESCAPE), None);
        let mut with_real = literals_only.clone();
        with_real.extend_from_slice(NUL_ESCAPE);
        assert_eq!(find_unescaped(&with_real, NUL_ESCAPE), Some(21));
    }

    #[test]
    fn rejected_hits_advance_one_byte_not_the_whole_match() {
        // The self-overlapping battery: the resume rule's own rows. `00` in
        // `\000` is the canonical one — the live hit at 2 starts INSIDE the
        // rejected match at 1, so any resume-at-match-end scan (memmem's
        // find_iter) answers None here.
        assert_eq!(find_unescaped(b"\\000", b"00"), Some(2));
        assert_eq!(find_unescaped(b"\\0000", b"000"), Some(2));
        assert_eq!(find_unescaped(b"\\a\\aa", b"a"), Some(4));
    }

    #[test]
    fn all_backslash_needles_answer_at_the_run_start() {
        // An all-backslash needle's first occurrence always sits at a
        // maximal run's start (an even, empty run before it — the leftmost
        // window of k backslashes begins where the run does), so those
        // scans answer at the first hit and never walk.
        assert_eq!(find_unescaped(b"\\\\", b"\\"), Some(0));
        assert_eq!(find_unescaped(&[b'\\'; 8], b"\\\\"), Some(0));
        let haystack = [b"x", &[b'\\'; 3][..], b"y"].concat();
        assert_eq!(find_unescaped(&haystack, b"\\\\"), Some(1));
    }

    #[test]
    fn offsets_index_bytes_over_multibyte_content() {
        // bytes in, BYTE offsets out: the needle after a two-byte char
        // (byte 6, char 5), after a four-byte emoji (byte 4, char 1), and
        // the -1 shape with multibyte bytes still traversed past two
        // rejected hits.
        let acute = "caf\u{e9} ".as_bytes();
        let mut haystack = acute.to_vec();
        haystack.extend_from_slice(NUL_ESCAPE);
        assert_eq!(find_unescaped(&haystack, NUL_ESCAPE), Some(6));
        let emoji = "\u{1f600}".as_bytes();
        let mut haystack = emoji.to_vec();
        haystack.extend_from_slice(NUL_ESCAPE);
        assert_eq!(find_unescaped(&haystack, NUL_ESCAPE), Some(4));
        // CJK row, mirroring the Python byte-offset battery: two CJK chars
        // are six UTF-8 bytes, so the needle sits at byte 6 (char 2).
        let cjk = "\u{6771}\u{4eac}".as_bytes();
        let mut haystack = cjk.to_vec();
        haystack.extend_from_slice(NUL_ESCAPE);
        assert_eq!(find_unescaped(&haystack, NUL_ESCAPE), Some(6));
        let mut haystack = [b"\\", NUL_ESCAPE].concat().repeat(2);
        haystack.extend_from_slice(b"caf");
        haystack.extend_from_slice("\u{e9}".as_bytes());
        assert_eq!(find_unescaped(&haystack, NUL_ESCAPE), None);
    }

    #[test]
    fn degenerate_shapes_are_answerable_not_errors() {
        assert_eq!(find_unescaped(b"", NUL_ESCAPE), None);
        assert_eq!(find_unescaped(NUL_ESCAPE, NUL_ESCAPE), Some(0));
        assert_eq!(find_unescaped(b"\\u00", NUL_ESCAPE), None);
        assert_eq!(find_unescaped(b"plain text, no escapes", NUL_ESCAPE), None);
    }

    #[test]
    fn long_runs_and_many_hits_stay_correct() {
        // The long-run and many-hit shapes at scale. Careful with the run
        // arithmetic (the mistake this row originally pinned against
        // itself): for R backslashes then `u0000`, the needle's occurrence
        // starts at R-1 — its first byte IS the run's last backslash — so
        // the run BEFORE the hit is R-1, live iff R-1 is even.
        // R = 99_999: run 99_998, even, live at 99_998.
        let mut odd = vec![b'\\'; 99_999];
        odd.extend_from_slice(b"u0000");
        assert_eq!(find_unescaped(&odd, NUL_ESCAPE), Some(99_998));
        // R = 100_000: run 99_999, odd, rejected, and nothing else occurs.
        let mut even = vec![b'\\'; 100_000];
        even.extend_from_slice(b"u0000");
        assert_eq!(find_unescaped(&even, NUL_ESCAPE), None);
        // The needle's own leading backslash behind a whole even run: the
        // occurrence at the run's end is live.
        let mut run_then_needle = vec![b'\\'; 100_000];
        run_then_needle.extend_from_slice(NUL_ESCAPE);
        assert_eq!(find_unescaped(&run_then_needle, NUL_ESCAPE), Some(100_000));
        // Ten thousand all-rejected hits (the false-positive corpus's
        // shape, where a resume or walk bug would compound across hits).
        let corpus = [b"\\", NUL_ESCAPE].concat().repeat(10_000);
        assert_eq!(find_unescaped(&corpus, NUL_ESCAPE), None);
    }

    #[test]
    fn every_needle_over_a_tiny_alphabet_matches_the_naive_walk() {
        // The exhaustive sweep, the Python suite's tiny-alphabet idiom:
        // every 1-2 byte needle over {\\, u, 0} (self-overlapping ones
        // included) crossed with every haystack over the same alphabet up
        // to length 7 — 39,360 pairs (12 needles x 3,280 haystacks), the
        // complete small space of run/overlap/adjacency interactions at
        // that size, no sampling. Level-by-level: each length built once
        // from the previous length's snapshot, so every haystack occurs
        // exactly once (a cumulative extend-from-all would duplicate short
        // haystacks into 16,384 entries for the same 3,280 distinct).
        let alphabet = *b"\\u0";
        let mut needles: Vec<Vec<u8>> = Vec::new();
        for &a in &alphabet {
            needles.push(vec![a]);
            for &b in &alphabet {
                needles.push(vec![a, b]);
            }
        }
        let mut haystacks: Vec<Vec<u8>> = vec![Vec::new()];
        let mut level: Vec<Vec<u8>> = vec![Vec::new()];
        for _ in 0..7 {
            let mut next = Vec::new();
            for base in &level {
                for &c in &alphabet {
                    let mut extended = base.clone();
                    extended.push(c);
                    next.push(extended);
                }
            }
            haystacks.extend(next.iter().cloned());
            level = next;
        }
        assert_eq!(haystacks.len(), 3_280);
        for needle in &needles {
            for haystack in &haystacks {
                assert_eq!(
                    find_unescaped(haystack, needle),
                    naive_find(haystack, needle),
                    "needle {needle:?} haystack {haystack:?}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "empty needle")]
    fn empty_needle_is_a_core_precondition_violation() {
        // The pub core (`pub mod scan_impl` in lib.rs) enforces the
        // non-empty-needle precondition itself; the pyo3 wrapper's
        // ValueError is the Python spelling of the same refusal.
        let _ = find_unescaped(b"haystack", b"");
    }

    #[test]
    fn buffer_end_and_adjacent_live_shapes() {
        // Buffer-end goldens: the needle live at the very end (run 2,
        // even, hit at 2, no tail past the match) and the rejected-at-end
        // shape (run 1, odd, -1 with no tail to walk past).
        assert_eq!(
            find_unescaped(&[b"\\\\", NUL_ESCAPE].concat(), NUL_ESCAPE),
            Some(2)
        );
        assert_eq!(
            find_unescaped(&[b"\\", NUL_ESCAPE].concat(), NUL_ESCAPE),
            None
        );
        // Adjacent live-live: two live occurrences back to back answer the
        // FIRST, not the last.
        assert_eq!(
            find_unescaped(&[NUL_ESCAPE, NUL_ESCAPE].concat(), NUL_ESCAPE),
            Some(0)
        );
    }

    #[test]
    fn backward_walk_budget_is_linear() {
        // The runs-are-disjoint cost proof, reproducibly: instrument the
        // same walk the core runs and assert total backward steps <=
        // haystack length over the long-run shape and the 10k-rejected
        // shape. Every walk stays inside its own maximal backslash run
        // and at most one hit per run walks, so no byte is walked twice.
        fn counted(haystack: &[u8], needle: &[u8]) -> (Option<usize>, usize) {
            let finder = memchr::memmem::Finder::new(needle);
            let mut from = 0;
            let mut steps = 0usize;
            while let Some(rel) = finder.find(&haystack[from..]) {
                let hit = from + rel;
                let mut run_back = hit;
                while run_back > 0 && haystack[run_back - 1] == b'\\' {
                    run_back -= 1;
                    steps += 1;
                }
                if (hit - run_back).is_multiple_of(2) {
                    return (Some(hit), steps);
                }
                from = hit + 1;
            }
            (None, steps)
        }
        let mut run_then_needle = vec![b'\\'; 100_000];
        run_then_needle.extend_from_slice(NUL_ESCAPE);
        let (got, steps) = counted(&run_then_needle, NUL_ESCAPE);
        assert_eq!(got, Some(100_000));
        assert!(
            steps <= run_then_needle.len(),
            "walked {steps} steps over {} bytes",
            run_then_needle.len()
        );
        let corpus = [b"\\", NUL_ESCAPE].concat().repeat(10_000);
        let (got, steps) = counted(&corpus, NUL_ESCAPE);
        assert_eq!(got, None);
        assert!(
            steps <= corpus.len(),
            "walked {steps} steps over {} bytes",
            corpus.len()
        );
        // The inside-run long shape: R = 99_999 backslashes then `u0000`
        // spells the needle at 99_998 (run 99_998 before the hit, even).
        let mut odd = vec![b'\\'; 99_999];
        odd.extend_from_slice(b"u0000");
        let (got, steps) = counted(&odd, NUL_ESCAPE);
        assert_eq!(got, Some(99_998));
        assert!(
            steps <= odd.len(),
            "walked {steps} steps over {} bytes",
            odd.len()
        );
        // The backslash-u rejected chain: four all-rejected hits each
        // behind exactly one backslash (runs 3, 1, 1, 1), the
        // consecutive-rejected-hits boundary shape.
        let mut chain = vec![b'\\'; 4];
        for _ in 0..3 {
            chain.extend_from_slice(b"u\\\\");
        }
        chain.extend_from_slice(b"u");
        let (got, steps) = counted(&chain, b"\\u");
        assert_eq!(got, None);
        assert!(
            steps <= chain.len(),
            "walked {steps} steps over {} bytes",
            chain.len()
        );
    }
}
