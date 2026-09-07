//! Multi-pattern substring search, the pure-Rust core of `tors.find_patterns`,
//! of its replacement sibling `replace_many` (below), and of the
//! count spelling `tors.count_matches` (below).
//!
//! Semantics: **leftmost-longest, non-overlapping**: aho-corasick's
//! `MatchKind::LeftmostLongest` (the engine's own words: "reports leftmost
//! matches; when there are multiple possible leftmost matches, the longest
//! match is chosen"). The scan proceeds left to right; at each position the
//! longest matching pattern wins regardless of its position in the list (NOT
//! regex-alternation leftmost-FIRST priority); the scan resumes at the end of
//! each reported match, so matches never overlap and are emitted in strictly
//! increasing start order. Two distinct patterns can tie at a position only by
//! being byte-identical (both must equal the same text), so "longest" always
//! has a strict winner except for exact duplicates, and duplicates report
//! the FIRST index they occupy in the list, made deterministic by a
//! canonical-id remap (below) that does not care which duplicate id the
//! automaton's internals hand back.
//!
//! The engine is `aho-corasick` 1.x (BurntSushi's Aho-Corasick automaton
//! inside Rust's own `regex` crate; dual `Unlicense OR MIT`, MIT elected, the
//! election recorded in the README's dependency table). The pure-Python
//! differential oracle and the golden battery live in
//! tests/test_find_patterns.py; their crate-side mirrors live below so the
//! two sides cannot drift independently.
//!
//! # Byte→char offset mapping
//!
//! The automaton runs on UTF-8 BYTES, so `Match::start()`/`end()` are byte
//! offsets, but the API reports Python `str` indices (codepoints). Two
//! properties make the conversion exact and cheap:
//!
//! * Whole-pattern matches land on character boundaries BY CONSTRUCTION: a
//!   valid UTF-8 sequence cannot begin mid-character (its first byte is a
//!   lead byte, never a continuation byte), so a pattern occurrence in valid
//!   UTF-8 text starts at a char boundary, and the pattern's own bytes being
//!   valid UTF-8 make its end a boundary too. Every byte offset the
//!   conversion pass touches is therefore slice-safe by an invariant, not by
//!   a check.
//! * The matches are non-overlapping and ordered, so ONE boundary-aware
//!   forward pass converts them all: walk the gaps and spans between
//!   successive match boundaries with `str::chars().count()`, carrying a
//!   byte cursor and a char cursor; each byte of the text is visited at
//!   most once across the whole pass.
//!
//! The ASCII fast path: `text.is_ascii()` means every byte offset IS a char
//! offset, so the automaton's offsets pass through unconverted and the whole
//! mapping pass is skipped (the one `is_ascii` scan replaces it; both are
//! single byte-oriented passes over the text). Non-ASCII PATTERNS over ASCII
//! text never match and cannot disturb this; they simply contribute no
//! matches.
//!
//! # The replace core
//!
//! [`replace_many`] (below) applies this module's exact
//! search semantics to replacement: the same one-pass leftmost-longest scan,
//! with each match spliced out and its key's VALUE emitted in its place, the
//! simultaneous-replace primitive that is to `str.replace` what
//! `find_patterns` is to `str.find`. Its full contract, including why it
//! needs no byte→char pass of its own, and the `Cow` identity contract on its
//! answer, is documented on the function.
//!
//! # The compiled spelling
//!
//! The free functions below build their automaton (and the duplicate-id
//! remap, or the first-value map) from the pattern list on EVERY call:
//! the right shape for a one-off call, the wrong shape for a pipeline
//! that runs the SAME fixed vocabulary (a redaction list, a terminology
//! rewrite table) over many texts or many times, where the per-call build
//! is a fixed cost the work keeps re-paying for an automaton it already
//! had. [`CompiledPatterns`] is the `re.compile()` answer: build once,
//! hold the automaton and the remaps behind one `Arc`, and every scan
//! afterwards is the free function's scan CLASSES minus the build (the
//! pyo3 wrapper `tors.CompiledPatterns` in `src/py/compiled_patterns.rs`
//! holds it; its gate is tests/test_compiled_patterns.py, which re-runs
//! the free functions' own batteries through a compiled fixture so the
//! two spellings cannot drift independently, plus the amortization story
//! pinned as construction + N calls vs N free calls at a document-scale
//! corpus).
//!
//! Pure Rust, no pyo3 types: the criterion bench (benches/search.rs) drives
//! this path directly; the pyo3 wrapper in `lib.rs` adds only the argument
//! borrows and the O(matches) return marshalling (see the crate GIL model
//! there).
//!
//! # Preconditions (enforced by the wrapper before this runs)
//!
//! The pattern list may be empty (answers empty, and building a zero-pattern
//! automaton is legal), but every entry must be NON-EMPTY: an empty pattern
//! would match at every position and has no leftmost-longest meaning; the
//! wrapper refuses it with `ValueError("empty pattern")`. The build itself
//! can still fail on engine limits (a pattern longer than the automaton's
//! u32 span budget, or more patterns than its ID space), hence the
//! `Result`, mapped by the wrapper to a `ValueError` carrying the engine's
//! message; both are unreachable for any input a real caller can build.

use std::borrow::Cow;
use std::collections::HashMap;

use aho_corasick::{AhoCorasickBuilder, MatchKind};

/// Re-exported so the pyo3 layer (and any Rust caller of this module) can
/// name the engine's build-failure type: the pub `find_patterns` /
/// `count_matches` / `replace_many` cores return it.
pub use aho_corasick::BuildError;

/// One reported match: `(start, end, pattern_index)` in Python `str` index
/// (codepoint) units: `text[start:end] == patterns[pattern]`, `end` exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatternMatch {
    pub start: usize,
    pub end: usize,
    pub pattern: usize,
}

/// The ONE automaton every spelling in this module drives: built
/// leftmost-longest over the pattern list (see the module docs for the
/// semantics). Shared by the free functions' per-call build and by
/// [`CompiledPatterns`]'s one-time build, so the two spellings drive the
/// identical engine configuration by construction.
fn build_leftmost(patterns: &[&str]) -> Result<aho_corasick::AhoCorasick, BuildError> {
    AhoCorasickBuilder::new()
        .match_kind(MatchKind::LeftmostLongest)
        .build(patterns)
}

/// The canonical-id remap of the find side: one slot per pattern (indexed
/// by automaton id, which is the pattern's position in the list the
/// automaton was built over), holding the FIRST list index that pattern
/// string occupies. For duplicates (and only duplicates can tie, see the
/// module docs), a match reports the FIRST index, and the remap makes
/// that deterministic regardless of which duplicate id the automaton's
/// internals hand back.
fn first_id_remap(patterns: &[&str]) -> Vec<usize> {
    let mut first_id: Vec<usize> = Vec::with_capacity(patterns.len());
    let mut seen: HashMap<&str, usize> = HashMap::with_capacity(patterns.len());
    for (idx, pattern) in patterns.iter().enumerate() {
        first_id.push(*seen.entry(*pattern).or_insert(idx));
    }
    first_id
}

/// The first-value map of the replace side: key string to the FIRST pair's
/// value (a dict cannot produce duplicate keys, but the slice spelling
/// can, and the first pair wins for the value the same way the find
/// side's remap makes the first index win). Every value lookup the
/// replace scans perform routes through this map, so the matched span IS
/// the lookup key.
fn first_values<'a>(replacements: &'a [(&'a str, &'a str)]) -> HashMap<&'a str, &'a str> {
    let mut values: HashMap<&str, &str> = HashMap::with_capacity(replacements.len());
    for (key, value) in replacements {
        values.entry(*key).or_insert(*value);
    }
    values
}

/// The shared scan of the find side: the automaton's matches over `text`,
/// converted from byte offsets to Python `str` indices, with the ASCII
/// fast path and the boundary-aware conversion pass of the module docs.
/// Driven identically by the free spelling (a fresh automaton per call)
/// and the compiled one ([`CompiledPatterns`]: the automaton built once).
fn scan_matches(
    ac: &aho_corasick::AhoCorasick,
    first_id: &[usize],
    text: &str,
) -> Vec<PatternMatch> {
    let mut matches = Vec::new();
    if text.is_ascii() {
        // Fast path: byte offsets ARE char offsets; nothing to convert.
        for m in ac.find_iter(text) {
            matches.push(PatternMatch {
                start: m.start(),
                end: m.end(),
                pattern: first_id[m.pattern().as_usize()],
            });
        }
    } else {
        // One boundary-aware pass: gaps and spans between successive match
        // boundaries partition the text, so the char counts below visit each
        // byte at most once in total. Both cursors sit on char boundaries at
        // every step (0 initially, then each match's end; see the module
        // docs for why match boundaries are boundaries by construction).
        let mut byte_cursor = 0;
        let mut char_cursor = 0;
        for m in ac.find_iter(text) {
            char_cursor += text[byte_cursor..m.start()].chars().count();
            let start = char_cursor;
            char_cursor += text[m.start()..m.end()].chars().count();
            matches.push(PatternMatch {
                start,
                end: char_cursor,
                pattern: first_id[m.pattern().as_usize()],
            });
            byte_cursor = m.end();
        }
    }
    matches
}

/// The leftmost-longest, non-overlapping matches of `patterns` in `text`, in
/// Python `str` index units, in increasing start order; the full contract is
/// the module docs above.
pub fn find_patterns(patterns: &[&str], text: &str) -> Result<Vec<PatternMatch>, BuildError> {
    let ac = build_leftmost(patterns)?;
    let first_id = first_id_remap(patterns);
    Ok(scan_matches(&ac, &first_id, text))
}

/// The number of leftmost-longest, non-overlapping matches of `patterns`
/// in `text`, the count spelling of [`find_patterns`]: the same ONE
/// automaton over the patterns (`MatchKind::LeftmostLongest`), the same
/// `find_iter(text)` scan (left to right, resume at each match's end),
/// with the matches COUNTED instead of collected.
///
/// WHY its own spelling: counting is the common question ("how many
/// occurrences?"), and the list spelling answers it only by materializing
/// the full answer first: a dense corpus fills a `Vec<PatternMatch>` at
/// ~2.5 bytes of match record per input byte (the 100 MiB dense-corpus
/// case fills a ~250 MiB vector today) just to have its `.len()` taken.
/// `find_iter(text).count()` answers the same number in O(1) memory: the
/// automaton and the scan are shared, the per-match record is never
/// built.
///
/// # No offset mapping
///
/// Counting is offset-FREE: the number of matches does not depend on the
/// units the offsets would be reported in, so the byte→char conversion
/// pass of [`find_patterns`] (and the `is_ascii` fast-path decision that
/// gates it) is pure waste here and simply does not run; the automaton's
/// byte offsets are never even read.
///
/// # Duplicates
///
/// No canonical-id remap, and WHY: duplicates are the only patterns that
/// can tie at a position, and they tie only by being byte-identical:
/// whichever duplicate id the automaton reports, the SPAN it reports is
/// the same text span, the scan consumes that span exactly once, and the
/// count therefore cannot see which id won. `find_patterns` remaps ids to
/// make its ANSWER deterministic; a count has no ids in its answer to be
/// non-deterministic about.
///
/// # Preconditions (enforced by the pyo3 wrapper before this runs)
///
/// The same ones as [`find_patterns`]: the pattern list may be empty (a
/// zero-pattern automaton is legal and finds nothing, so the answer is 0),
/// and every entry must be NON-EMPTY; the wrapper refuses an empty pattern
/// with `ValueError("empty pattern")`. The build itself can still fail on
/// engine limits (a pattern longer than the automaton's u32 span budget,
/// or more patterns than its ID space), hence the `Result`, mapped by
/// the wrapper to a `ValueError` carrying the engine's message; both are
/// unreachable for any input a real caller can build.
///
/// # Invariant
///
/// `count_matches(patterns, text)? ==
/// find_patterns(patterns, text)?.len()` for every legal input: the two
/// spellings drive the identical scan, so a disagreement is a BUG, not a
/// tolerance; pinned over the golden battery and the tiny-alphabet
/// exhaustive sweep by the tests below.
pub fn count_matches(patterns: &[&str], text: &str) -> Result<usize, BuildError> {
    let ac = build_leftmost(patterns)?;
    // The same find_iter scan find_patterns drives; `.count()` consumes
    // each match as it is yielded, so no match record ever exists.
    Ok(ac.find_iter(text).count())
}

/// The simultaneous-replace primitive: `text` with every occurrence of any
/// KEY in `replacements` spliced out and that key's VALUE emitted in its
/// place. It applies [`find_patterns`]'s exact search semantics (one automaton over
/// the keys, `MatchKind::LeftmostLongest`, scan left to right, resume at
/// each match's end) with the matched span REPLACED instead of reported.
///
/// This is the multi-key `str.replace` CPython lacks: chained `str.replace`
/// calls are one WHOLE-TEXT pass per key, and each pass rescans the previous
/// pass's output, so earlier replacements can expose later keys to text they
/// were never meant to see. A `re.sub` alternation is leftmost-FIRST instead
/// (the alternation's ORDER decides between same-start candidates, not their
/// lengths) and rescans its own output through the callback. This function
/// does neither: a longer key beats a shorter one at the same position
/// regardless of slice order, matches never overlap (the scan resumes at the
/// match end), and the output is NEVER re-scanned. One pass over the INPUT
/// decides every splice, so a value that itself contains a key does not
/// cascade. The canonical
/// self-feeding case: `{"&": "&amp;"}` over `"&amp;"` answers
/// `"&amp;amp;"`, not an ever-growing `"&amp;amp;amp;"`.
///
/// # Identity-return contract
///
/// The crate's `Cow` identity convention, complete form: when NO key matches, the answer is
/// `Cow::Borrowed(text)` and nothing is allocated; and after the output is
/// built, a final `out == text` comparison (one O(text) pass) catches every
/// replacement whose net effect is the identity: a value equal to its key
/// (`{"a": "a"}`), or splices that mutually cancel (`{"b": "", "a": "ba"}`
/// over `"ba"` deletes the `b` and regrows it inside the `a`), and returns
/// `Cow::Borrowed(text)` for those too. The caller-visible contract is
/// therefore exact: `replace_many(s, m)` IS `s` (borrowed, the same bytes)
/// exactly when `replace_many(s, m) == s`.
///
/// # Duplicates
///
/// A dict cannot produce duplicate keys, but the slice spelling can. The
/// FIRST occurrence of a duplicated key wins, for the automaton id and the
/// value alike: a `HashMap<&str, usize>` of key to first index (the
/// replace-side mirror of `find_patterns`'s canonical-id remap), with every
/// value lookup routed through it, pins that deterministically regardless of
/// which duplicate id the automaton's internals hand back.
///
/// # No offset mapping
///
/// Unlike [`find_patterns`] there is no byte→char pass here: the automaton's
/// byte offsets only ever SLICE `text` at match boundaries, which the module
/// docs' construction already guarantees are character boundaries. They are
/// never reported to a caller, so there is nothing to convert. The output is
/// spliced strings, not reported offsets.
///
/// # Preconditions (enforced by the pyo3 wrapper before this runs)
///
/// Every key must be NON-EMPTY: an empty key would match at every position
/// and has no leftmost-longest meaning; the wrapper refuses it. Keys and
/// values are Python `str` objects borrowed to `&str` (UTF-8 by the borrow,
/// so the splices below cannot panic on slicing). The map may be EMPTY: no
/// key exists, nothing matches, and the input comes back borrowed. The build
/// itself can still fail on engine limits (a key longer than the
/// automaton's u32 span budget, or more keys than its ID space), hence the
/// `Result`, mapped by the wrapper to a `ValueError` carrying the engine's
/// message; unreachable for any input a real caller can build.
///
/// # Complexity
///
/// One automaton build O(total key bytes) + one scan O(text) + one output
/// build O(output): linear in everything it touches.
pub fn replace_many<'a>(
    text: &'a str,
    replacements: &[(&str, &str)],
) -> Result<Cow<'a, str>, BuildError> {
    let keys: Vec<&str> = replacements.iter().map(|(key, _)| *key).collect();
    let ac = build_leftmost(&keys)?;
    let values = first_values(replacements);
    Ok(scan_replace(&ac, &values, text))
}

/// The shared splice of the replace side: [`replace_many`]'s scan over a
/// built automaton and a validated `values` map (key string to the first
/// pair's value; the matched span IS the key's bytes, so the span itself
/// is the lookup key, and the hit is guaranteed by construction: the
/// automaton only matches keys the map carries). Driven identically by
/// the free spelling and the compiled one, with the identity-return
/// contract (the `Cow` convention) intact on both lanes.
fn scan_replace<'a>(
    ac: &aho_corasick::AhoCorasick,
    values: &HashMap<&str, &str>,
    text: &'a str,
) -> Cow<'a, str> {
    // text.len() is a good starting capacity for most workloads (replacement
    // values are typically comparable in length to the keys they replace);
    // a bigger swing still amortizes via the normal growth doubling, but this
    // avoids the small-capacity reallocations that plain `String::new()`
    // pays on every splice-heavy call, the same reservation the crate's
    // other splice-building functions make (url_impl, html_impl,
    // normalize_impl, pipeline_impl).
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    let mut matched = false;
    for m in ac.find_iter(text) {
        matched = true;
        out.push_str(&text[last..m.start()]);
        out.push_str(values[&text[m.start()..m.end()]]);
        last = m.end();
    }
    if !matched {
        return Cow::Borrowed(text);
    }
    out.push_str(&text[last..]);
    // The identity contract's second lane: a replacement whose net effect is
    // the identity (value == key, or mutually-cancelling splices) hands back
    // the input itself rather than a byte-identical copy of it.
    if out == text {
        return Cow::Borrowed(text);
    }
    Cow::Owned(out)
}

/// The length-preserving redaction spelling of [`replace_many`]: the SAME
/// one-automaton leftmost-longest, non-overlapping, never-rescanned scan
/// (the shared [`replace_automaton`] build, the same `find_iter(text)` walk,
/// resume at each match's end, first-duplicate-wins value lookup), but each
/// matched span is replaced by a MASKED value of exactly the span's
/// CHARACTER count, and every non-matching span passes through untouched in
/// place, so `out.chars().count() == text.chars().count()`, always.
///
/// WHY this spelling exists: when text is redacted for logs, training
/// corpora, or PII pipelines, every downstream offset computed BEFORE the
/// redaction (`find_patterns` spans, word/sentence bound indices, diff
/// opcodes) must still be valid on the redacted output. `replace_many`
/// breaks that the moment any value's character length differs from its
/// key's (a splice shifts everything after it); the masked spelling is the
/// one that cannot shift anything.
///
/// # The mask rule (pinned)
///
/// For a matched span of L characters and its key's value V of C
/// characters, exactly ONE of two branches fires: they are mutually
/// exclusive, decided solely by which side of L the value's character count
/// falls:
///
/// * C >= L (value at least as long): TRUNCATION. The value's first L
///   characters are emitted; the mask is never consulted.
/// * C < L (value shorter): PADDING. The whole value is emitted, followed
///   by L − C copies of `mask`.
///
/// Worked examples, the consumer row and its mirrors:
///
/// * `replace_many_masked("the cat sat", &[("cat", "[REDACTED]")], '*')`
///   → `"the [RE sat"`: the span "cat" is 3 chars, the value is 10 ≥ 3, so
///   its FIRST 3 chars "[RE" are emitted; no `'*'` appears in the output.
/// * the same call with `&[("cat", "X")]` → `"the X** sat"`: 1 < 3, so
///   "X" plus two masks.
/// * the same call with `&[("cat", "")]` → `"the *** sat"`: the
///   deletion-shaped value pads to pure mask.
/// * the same call with `&[("cat", "dog")]` → `"the dog sat"`: 3 == 3,
///   neither branch visibly fires.
///
/// # The length proof
///
/// The input's characters partition into the non-matching gaps between
/// matches and the match spans themselves. A gap is copied verbatim, so it
/// contributes exactly its own characters; a span of L characters is
/// replaced by exactly L characters: the truncated value (L of them) or
/// the value plus masks (C + (L − C) = L). Every input character is
/// therefore accounted for by an equal number of output characters, so
/// the character count, and hence every character-unit offset, is preserved.
///
/// The guarantee is in CHARACTERS, the Python-visible unit every offset
/// this crate reports is in. BYTE length may change: a multibyte value or
/// mask replaces the matched key's bytes (`"a"` → `"é"` is 1 char to 1
/// char but 1 byte to 2), so byte-unit arithmetic on the output is NOT
/// preserved. This is by design, because no byte-unit offset is ever reported.
///
/// # Identity-return contract
///
/// The same `Cow` identity convention, complete form: when NO key matches, the
/// answer is `Cow::Borrowed(text)` and nothing is allocated; and the final
/// `out == text` comparison catches every masked replacement whose net
/// effect is the identity: a value equal to its key (the mask never
/// needed), a truncation whose first L chars ARE the key (`("ab", "abcd")`
/// over `"ab"`), a padding that regrows it (`("ab", "a")` with mask `'b'`),
/// returning the input borrowed for those too. The caller-visible
/// contract is exact: `replace_many_masked(s, m, c)` IS `s` (borrowed, the
/// same bytes) exactly when it `== s`.
///
/// # No offset mapping
///
/// The same reasoning as [`replace_many`]: the automaton's byte offsets
/// only ever SLICE `text` at match boundaries (character boundaries by the
/// module docs' construction) and count the span's characters. They are
/// never reported, so there is nothing to convert.
///
/// # Preconditions (enforced by the pyo3 wrapper before this runs)
///
/// Every key NON-EMPTY (an empty key would match at every position and has
/// no leftmost-longest meaning, the same refusal as `replace_many`); the
/// map may be EMPTY (no key exists, nothing matches, the input comes back
/// borrowed); and `mask` is exactly ONE character, by type here (a `char`
/// is one codepoint), enforced on the Python side by the wrapper (a
/// single-character `str`; a multi-char mask would break the L − C padding
/// arithmetic, which counts characters). The build itself can still fail
/// on engine limits (a key longer than the automaton's u32 span budget, or
/// more keys than its ID space), hence the `Result`, mapped by the
/// wrapper to a `ValueError` carrying the engine's message; unreachable
/// for any input a real caller can build.
///
/// # Complexity
///
/// One automaton build O(total key bytes) + one scan O(text) + one output
/// build O(output): the per-match character counts are O(L), the same
/// order the splice itself is; linear in everything it touches.
pub fn replace_many_masked<'a>(
    text: &'a str,
    replacements: &[(&str, &str)],
    mask: char,
) -> Result<Cow<'a, str>, BuildError> {
    let keys: Vec<&str> = replacements.iter().map(|(key, _)| *key).collect();
    let ac = build_leftmost(&keys)?;
    let values = first_values(replacements);
    Ok(scan_replace_masked(&ac, &values, text, mask))
}

/// The shared masked splice: [`replace_many_masked`]'s scan over a built
/// automaton and a validated `values` map, each matched span replaced by
/// the masked value of exactly the span's CHARACTER count (the mask rule
/// documented on [`replace_many_masked`]), driven identically by the free
/// and compiled spellings.
fn scan_replace_masked<'a>(
    ac: &aho_corasick::AhoCorasick,
    values: &HashMap<&str, &str>,
    text: &'a str,
    mask: char,
) -> Cow<'a, str> {
    // Same capacity reservation as scan_replace, for the same reason: the
    // character count is preserved exactly, so text.len() bytes is an even
    // better starting estimate here than in the unmasked splice (it is
    // exact whenever every value and the mask stay within the key's own
    // byte width, e.g. all-ASCII).
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    let mut matched = false;
    for m in ac.find_iter(text) {
        matched = true;
        let span = &text[m.start()..m.end()];
        out.push_str(&text[last..m.start()]);
        let span_chars = span.chars().count();
        let value = values[span];
        let value_chars = value.chars().count();
        if value_chars >= span_chars {
            // Truncation: the value's first L characters; the mask is
            // never consulted on this branch.
            out.extend(value.chars().take(span_chars));
        } else {
            // Padding: the whole value, then L − C copies of the mask.
            out.push_str(value);
            out.extend(std::iter::repeat_n(mask, span_chars - value_chars));
        }
        last = m.end();
    }
    if !matched {
        return Cow::Borrowed(text);
    }
    out.push_str(&text[last..]);
    // The identity contract's second lane: a masked replacement whose net
    // effect is the identity (value == key, a truncation that regrows the
    // key, a padding that regrows it) hands back the input itself rather
    // than a character-identical copy of it.
    if out == text {
        return Cow::Borrowed(text);
    }
    Cow::Owned(out)
}

/// The call-time replacements-validation failure: the dict handed to a
/// compiled replace call must key EXACTLY the compiled pattern set (every
/// pattern paired with a value, no other keys), because the automaton is
/// the compiled part and can only ever match that set; a key outside it
/// could never be honored (the free function would have built it in), and
/// a pattern without a value could never be spliced. `unknown` carries
/// the offending dict keys, `missing` the count of compiled patterns the
/// dict left unvalued.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplacementsMismatch {
    pub unknown: Vec<String>,
    pub missing: usize,
}

impl ReplacementsMismatch {
    /// The `str()` of the `ValueError` the pyo3 layer raises. The key and
    /// pattern names are capped at three shown plus a count, so a huge
    /// vocabulary mismatch cannot build a huge exception message.
    pub fn message(&self) -> String {
        fn show(names: &[String]) -> String {
            let mut shown: Vec<String> = names.iter().take(3).map(|k| format!("{k:?}")).collect();
            if names.len() > 3 {
                shown.push(format!("... and {} more", names.len() - 3));
            }
            shown.join(", ")
        }
        format!(
            "replacements must pair every compiled pattern with a value (no others): \
             unknown keys [{}], {} pattern(s) missing a value",
            show(&self.unknown),
            self.missing
        )
    }
}

/// A pattern list compiled once (the module docs' compiled spelling): the
/// ONE leftmost-longest automaton over the list, the owned pattern
/// strings, the find side's canonical-id remap, and the pattern-string to
/// first-index map the replace side validates call-time dicts against.
/// Immutable after [`CompiledPatterns::build`], so sharing one behind an
/// `Arc` across many calls and threads is sound with no synchronization
/// beyond the refcount itself (the `CompiledLemmaDict` discipline).
///
/// The scans are the free functions' own ([`scan_matches`],
/// [`scan_replace`], [`scan_replace_masked`], and the count spelling's
/// `find_iter(text).count()`), driven over the held automaton: a compiled
/// call is the free call's scan classes minus the build, by construction
/// rather than by reimplementation, which is what the parity gates in
/// this module and in tests/test_compiled_patterns.py pin.
pub struct CompiledPatterns {
    /// The automaton over the pattern list, `MatchKind::LeftmostLongest`.
    ac: aho_corasick::AhoCorasick,
    /// Owned copies of the patterns, list order (the automaton's id space).
    patterns: Vec<String>,
    /// The find side's canonical-id remap: automaton id to first list
    /// index (see [`first_id_remap`]).
    first_id: Vec<usize>,
    /// Pattern string to first list index: the replace side's validation
    /// set (a call-time dict must key exactly this set) and the routing
    /// for its values.
    first_index: HashMap<String, usize>,
}

impl CompiledPatterns {
    /// The one-time build: the automaton, the remaps, and the owned
    /// pattern strings, from a NON-EMPTY-entry pattern list (the empty
    /// list is legal and compiles to a zero-pattern automaton that finds
    /// nothing; the entry-level refusal is the caller's, the same
    /// `ValueError("empty pattern")` contract the free functions' wrapper
    /// enforces).
    pub fn build(patterns: &[&str]) -> Result<Self, BuildError> {
        let ac = build_leftmost(patterns)?;
        let first_id = first_id_remap(patterns);
        let owned: Vec<String> = patterns.iter().map(|p| (*p).to_string()).collect();
        let mut first_index: HashMap<String, usize> = HashMap::with_capacity(owned.len());
        for (idx, pattern) in owned.iter().enumerate() {
            first_index.entry(pattern.clone()).or_insert(idx);
        }
        Ok(CompiledPatterns {
            ac,
            patterns: owned,
            first_id,
            first_index,
        })
    }

    /// The list's length, mirroring `len(patterns)` on the source list
    /// (duplicates included, the same count the automaton's id space has).
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// The find spelling: [`find_patterns`]' scan over the held automaton,
    /// same matches, same canonical duplicate ids.
    pub fn find(&self, text: &str) -> Vec<PatternMatch> {
        scan_matches(&self.ac, &self.first_id, text)
    }

    /// The count spelling: [`count_matches`]' scan over the held
    /// automaton.
    pub fn count(&self, text: &str) -> usize {
        self.ac.find_iter(text).count()
    }

    /// Validates a call-time replacements pairing and builds the value
    /// map the replace scans look spans up in: every dict key must be a
    /// compiled pattern (a key the automaton cannot match could never be
    /// honored), and every compiled pattern must carry a value (a matched
    /// pattern with no value could never be spliced; the free spelling
    /// would simply not have that key in its automaton). Duplicate keys
    /// in the slice spelling route first-pair-wins, [`first_values`]'s
    /// own rule.
    pub fn replace_values<'a>(
        &self,
        pairs: &[(&'a str, &'a str)],
    ) -> Result<HashMap<&'a str, &'a str>, ReplacementsMismatch> {
        let mut values: HashMap<&str, &str> = HashMap::with_capacity(pairs.len());
        let mut unknown: Vec<String> = Vec::new();
        for (key, value) in pairs {
            if !self.first_index.contains_key(*key) {
                unknown.push((*key).to_string());
            }
            values.entry(*key).or_insert(*value);
        }
        // Dict keys are unique, so the in-set key count is the map's size
        // minus the unknown ones, and every compiled pattern is valued
        // exactly when that count reaches the pattern set's size.
        let in_set = values.len() - unknown.len();
        let missing = self.first_index.len().saturating_sub(in_set);
        if !unknown.is_empty() || missing > 0 {
            return Err(ReplacementsMismatch { unknown, missing });
        }
        Ok(values)
    }

    /// The replace spelling: [`replace_many`]'s scan over the held
    /// automaton and the validated call-time values.
    pub fn replace_many<'a>(&self, text: &'a str, values: &HashMap<&str, &str>) -> Cow<'a, str> {
        scan_replace(&self.ac, values, text)
    }

    /// The masked replace spelling: [`replace_many_masked`]'s scan over
    /// the held automaton and the validated call-time values.
    pub fn replace_many_masked<'a>(
        &self,
        text: &'a str,
        values: &HashMap<&str, &str>,
        mask: char,
    ) -> Cow<'a, str> {
        scan_replace_masked(&self.ac, values, text, mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find(patterns: &[&str], text: &str) -> Vec<(usize, usize, usize)> {
        find_patterns(patterns, text)
            .unwrap_or_else(|err| panic!("automaton build failed: {err}"))
            .into_iter()
            .map(|m| (m.start, m.end, m.pattern))
            .collect()
    }

    fn count(patterns: &[&str], text: &str) -> usize {
        count_matches(patterns, text).unwrap_or_else(|err| panic!("automaton build failed: {err}"))
    }

    fn replace<'a>(replacements: &[(&str, &str)], text: &'a str) -> Cow<'a, str> {
        replace_many(text, replacements)
            .unwrap_or_else(|err| panic!("automaton build failed: {err}"))
    }

    /// The brute-force leftmost-longest reference, in CHARACTER space (a
    /// Vec<char> view of the text, `str::starts_with` per position), the
    /// mirror of the Python oracle in tests/test_find_patterns.py, kept in
    /// lockstep with it so the two sides cannot drift independently.
    fn reference(patterns: &[&str], text: &str) -> Vec<(usize, usize, usize)> {
        let chars: Vec<char> = text.chars().collect();
        let pattern_chars: Vec<Vec<char>> = patterns.iter().map(|p| p.chars().collect()).collect();
        let mut out = Vec::new();
        let mut pos = 0usize;
        while pos < chars.len() {
            let mut best: Option<(usize, usize)> = None;
            for (idx, pc) in pattern_chars.iter().enumerate() {
                if best.is_none_or(|(len, _)| pc.len() > len)
                    && !pc.is_empty()
                    && chars[pos..].starts_with(pc)
                {
                    best = Some((pc.len(), idx));
                }
            }
            match best {
                Some((len, idx)) => {
                    out.push((pos, pos + len, idx));
                    pos += len;
                }
                None => pos += 1,
            }
        }
        out
    }

    /// The brute-force leftmost-longest REPLACE reference, in CHARACTER space
    /// (a `Vec<char>` view of the text, `str::starts_with` per position), the
    /// replace-side mirror of `reference`, kept in the same lockstep
    /// discipline. The first occurrence of a duplicated key wins for the
    /// value by the same strict `>` that keeps the first on length ties.
    fn reference_replace(replacements: &[(&str, &str)], text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        let pairs: Vec<(Vec<char>, &str)> = replacements
            .iter()
            .map(|(key, value)| (key.chars().collect(), *value))
            .collect();
        let mut out = String::new();
        let mut pos = 0usize;
        while pos < chars.len() {
            let mut best: Option<(usize, usize)> = None;
            for (idx, (key_chars, _)) in pairs.iter().enumerate() {
                if best.is_none_or(|(len, _)| key_chars.len() > len)
                    && !key_chars.is_empty()
                    && chars[pos..].starts_with(key_chars)
                {
                    best = Some((key_chars.len(), idx));
                }
            }
            match best {
                Some((len, idx)) => {
                    out.push_str(pairs[idx].1);
                    pos += len;
                }
                None => {
                    out.push(chars[pos]);
                    pos += 1;
                }
            }
        }
        out
    }

    #[test]
    fn golden_overlap_cases_are_pinned() {
        // Longest at the same start, independent of list order.
        assert_eq!(find(&["abc", "abcd"], "abcd"), vec![(0, 4, 1)]);
        assert_eq!(find(&["abcd", "abc"], "abcd"), vec![(0, 4, 0)]);
        // Positioning: nothing matches at 0; at 1 the longer "bcd" wins.
        assert_eq!(find(&["bcd", "cd"], "abcd"), vec![(1, 4, 0)]);
        // The scan resumes at the longest match's end.
        assert_eq!(find(&["bcd", "cd"], "abcdcd"), vec![(1, 4, 0), (4, 6, 1)]);
        // A prefix chain across the text: each position takes its longest fit.
        assert_eq!(
            find(&["ab", "abc", "abcd"], "abcdabcab"),
            vec![(0, 4, 2), (4, 7, 1), (7, 9, 0)]
        );
        // The longest that FITS, not the longest listed.
        assert_eq!(find(&["aaaa", "aa"], "aaa"), vec![(0, 2, 1)]);
        // Non-overlap: "aaaa" consumes four, then the tail matches "aa".
        assert_eq!(find(&["aa", "aaaa"], "aaaaaa"), vec![(0, 4, 1), (4, 6, 0)]);
        // Degenerate shapes.
        assert_eq!(find(&["abc"], ""), Vec::<(usize, usize, usize)>::new());
        assert_eq!(find(&["xyz"], "abc"), Vec::<(usize, usize, usize)>::new());
    }

    #[test]
    fn empty_pattern_list_answers_empty() {
        // The wrapper early-exits before this, but the core's own behavior on
        // an empty list is pinned too: an automaton over no patterns finds
        // nothing (and building it is legal).
        assert_eq!(find(&[], "abc"), Vec::<(usize, usize, usize)>::new());
    }

    #[test]
    fn duplicate_patterns_report_the_first_index() {
        // Identical patterns are legal; the FIRST index is the reported one,
        // and the canonical-id remap makes that deterministic regardless of which
        // duplicate id the automaton's internals hand back.
        assert_eq!(find(&["abc", "abc"], "abcabc"), vec![(0, 3, 0), (3, 6, 0)]);
        assert_eq!(find(&["abcd", "abc", "abcd"], "abcd"), vec![(0, 4, 0)]);
        // The same string occupying every slot.
        assert_eq!(
            find(&["needle", "needle", "needle"], "needle needle"),
            vec![(0, 6, 0), (7, 13, 0)]
        );
    }

    #[test]
    fn count_matches_is_the_length_of_find_patterns_on_the_golden_battery() {
        // The count/list invariant over the module's existing golden battery:
        // the overlap pins and the multibyte mapping battery re-run as
        // (patterns, text) pairs, each count checked against BOTH the list
        // spelling and the char-space reference (the count of the oracle's
        // answer is the independent differential: the list spelling's own
        // rows above pin find == oracle, so a disagreement in either lane is
        // a bug in one of the two spellings, not a tolerance to adjust).
        // Plus the two count-specific shapes the golden rows do not cover:
        // the empty list (0, since a zero-pattern automaton finds nothing) and
        // duplicates (the count is id-invariant: byte-identical ties report
        // the same span once, whichever duplicate id wins; no remap needed).
        let family = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}";
        let alternating = format!("é{}é{}é", 'b', 'b'); // é b é b é
        type CountCase<'a> = (Vec<&'a str>, String);
        let cases: Vec<CountCase> = vec![
            // The golden overlap pins, as (patterns, text) pairs.
            (vec!["abc", "abcd"], "abcd".to_string()),
            (vec!["abcd", "abc"], "abcd".to_string()),
            (vec!["bcd", "cd"], "abcd".to_string()),
            (vec!["bcd", "cd"], "abcdcd".to_string()),
            (vec!["ab", "abc", "abcd"], "abcdabcab".to_string()),
            (vec!["aaaa", "aa"], "aaa".to_string()),
            (vec!["aa", "aaaa"], "aaaaaa".to_string()),
            // Degenerate shapes, empty list included.
            (vec!["abc"], String::new()),
            (vec!["xyz"], "abc".to_string()),
            (Vec::new(), "abc".to_string()),
            // Duplicates: the same span is reported once regardless of
            // which duplicate id the automaton hands back, so the count
            // matches the list spelling's remapped answer.
            (vec!["abc", "abc"], "abcabc".to_string()),
            (vec!["abcd", "abc", "abcd"], "abcd".to_string()),
            (
                vec!["needle", "needle", "needle"],
                "needle needle".to_string(),
            ),
            // The multibyte mapping battery: non-ASCII text, where the
            // count spelling skips the byte→char pass the list spelling
            // runs (the count must not notice the pass is missing).
            (vec!["café"], "café café".to_string()),
            (vec!["cafe\u{301}"], "cafe\u{301} ok".to_string()),
            (vec!["東京"], "京都東京大阪".to_string()),
            (vec!["東京", "京都"], "京都東京京都".to_string()),
            (vec![family], format!("hi{family}!")),
            (vec!["ab"], "éabéab".to_string()),
            (vec!["éab"], "東京éab".to_string()),
            (vec!["b", "é"], alternating),
            (vec!["éé", "é"], "ééé".to_string()),
            (vec!["éé"], "éééé".to_string()),
        ];
        for (patterns, text) in &cases {
            assert_eq!(
                count(patterns, text),
                find(patterns, text).len(),
                "count/list disagreement on patterns {patterns:?} over {text:?}"
            );
            assert_eq!(
                count(patterns, text),
                reference(patterns, text).len(),
                "count/oracle disagreement on patterns {patterns:?} over {text:?}"
            );
        }
    }

    #[test]
    fn offsets_are_characters_not_bytes_over_multibyte_text() {
        // The mapping battery: every expectation is in CHARACTER units; a
        // byte-identity bug misreports each one (e.g. "café" at char 0 would
        // end at 5 in bytes instead of 4 in chars). Every row is ALSO checked
        // against the char-space reference: hand-pinned literal expectations
        // are error-prone (a mistyped text or a longest-that-fits miscount is
        // easy to write by hand), so a wrong row is caught by disagreement
        // with the oracle rather than trusted on its own.
        // Texts are built by concatenation where a typed literal would be
        // visually ambiguous (é b é b é, not the mistyped é b é é b).
        let alternating = format!("é{}é{}é", 'b', 'b'); // é b é b é
        let family = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}";
        let crab = "\u{1F980}\u{FE0F}";
        type MappingCase<'a> = (Vec<&'a str>, String, Vec<(usize, usize, usize)>);
        let cases: Vec<MappingCase> = vec![
            (
                vec!["café"],
                "café café".to_string(),
                vec![(0, 4, 0), (5, 9, 0)],
            ),
            (
                vec!["cafe\u{301}"],
                "cafe\u{301} ok".to_string(),
                vec![(0, 5, 0)],
            ),
            (vec!["東京"], "京都東京大阪".to_string(), vec![(2, 4, 0)]),
            (
                vec!["東京", "京都"],
                "京都東京京都".to_string(),
                vec![(0, 2, 1), (2, 4, 0), (4, 6, 1)],
            ),
            // ZWJ family emoji: 5 chars / 18 bytes.
            (vec![family], format!("hi{family}!"), vec![(2, 7, 0)]),
            // Emoji + variation selector (a combining mark): 2 chars / 7 bytes.
            (vec![crab], format!("{crab}!"), vec![(0, 2, 0)]),
            // ASCII pattern over NON-ASCII text: the is_ascii fast path must
            // NOT be taken (byte and char offsets diverge after the é).
            (vec!["ab"], "éabéab".to_string(), vec![(1, 3, 0), (4, 6, 0)]),
            (vec!["éab"], "東京éab".to_string(), vec![(2, 5, 0)]),
            (
                vec!["b", "é"],
                alternating,
                vec![(0, 1, 1), (1, 2, 0), (2, 3, 1), (3, 4, 0), (4, 5, 1)],
            ),
            // Longest-among-multibyte at one start ("éé" is 2 chars / 4 bytes).
            (
                vec!["éé", "é"],
                "ééé".to_string(),
                vec![(0, 2, 0), (2, 3, 1)],
            ),
            // Back-to-back two-char multibyte matches.
            (vec!["éé"], "éééé".to_string(), vec![(0, 2, 0), (2, 4, 0)]),
        ];
        for (patterns, text, expected) in &cases {
            assert_eq!(
                find(patterns, text),
                expected[..],
                "patterns {patterns:?} over {text:?}"
            );
            assert_eq!(
                reference(patterns, text),
                expected[..],
                "oracle disagrees on patterns {patterns:?} over {text:?}"
            );
        }
    }

    #[test]
    fn the_ascii_fast_path_agrees_with_the_conversion_pass() {
        // The same pattern over an ASCII text (offsets pass through) and over
        // a non-ASCII text containing the same ASCII core (conversion pass)
        // must produce the same CHARACTER answer.
        assert_eq!(find(&["abc"], "xabcx"), vec![(1, 4, 0)]);
        assert_eq!(find(&["abc"], "éabcé"), vec![(1, 4, 0)]);
    }

    #[test]
    fn every_pattern_list_over_a_tiny_alphabet_matches_the_reference() {
        // The exhaustive sweep: all pattern lists of size 0-3 over
        // {"a", "ab", "b"} (repetition allowed, duplicates included) × all
        // texts over {"a", "b"} up to length 5, the complete small space of
        // prefix/overlap/duplicate interactions, against the char-space
        // reference. 2,520 pairs, no sampling. Each pair also pins the
        // count/list invariant: the count spelling must agree with
        // the list spelling's length, and therefore with the oracle, over
        // the same complete space.
        let pool = ["a", "ab", "b"];
        let mut pattern_lists: Vec<Vec<&str>> = vec![Vec::new()];
        for size in 1..=3 {
            for combo in 0..pool.len().pow(size) {
                let mut list = Vec::with_capacity(size as usize);
                let mut rest = combo;
                for _ in 0..size {
                    list.push(pool[rest % pool.len()]);
                    rest /= pool.len();
                }
                pattern_lists.push(list);
            }
        }
        let mut texts: Vec<String> = Vec::new();
        for length in 0..=5 {
            for bits in 0..(1usize << length) {
                let text: String = (0..length)
                    .map(|i| if bits & (1 << i) == 0 { 'a' } else { 'b' })
                    .collect();
                texts.push(text);
            }
        }
        for patterns in &pattern_lists {
            for text in &texts {
                let found = find(patterns, text);
                assert_eq!(
                    found,
                    reference(patterns, text),
                    "patterns {patterns:?} over {text:?}"
                );
                assert_eq!(
                    count(patterns, text),
                    found.len(),
                    "count/list disagreement on patterns {patterns:?} over {text:?}"
                );
            }
        }
    }

    #[test]
    fn multibyte_random_texts_match_the_reference() {
        // A deterministic pseudo-random sweep over the multi-byte alphabet
        // (the same width mix the Python hypothesis property uses): an LCG
        // over {"a","b","é","´","東","京","🦀"} building 200 texts of up to 40
        // chars, each searched with a fixed 5-pattern set spanning the widths.
        // Crate-side differential coverage of the byte→char mapping that
        // does not depend on a hypothesis installation.
        let alphabet = ['a', 'b', 'é', '\u{301}', '東', '京', '\u{1F980}'];
        let patterns = ["éa", "東京", "\u{1F980}", "ab", "b\u{301}"];
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for case in 0..200u64 {
            let length = (case % 40) as usize;
            let text: String = (0..length)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    alphabet[(state >> 33) as usize % alphabet.len()]
                })
                .collect();
            let found = find(&patterns, &text);
            assert_eq!(found, reference(&patterns, &text), "case {case}: {text:?}");
            // The count invariant on the same non-ASCII texts: the lane
            // where the count spelling skips the byte→char pass the list
            // spelling runs.
            assert_eq!(
                count(&patterns, &text),
                found.len(),
                "count/list disagreement, case {case}: {text:?}"
            );
        }
    }

    #[test]
    fn golden_replacement_cases_are_pinned() {
        // Every row is checked against the replace oracle as well, the same
        // table discipline as the mapping battery. The "aa"→"b" row is the
        // one that pins leftmost-longest resume-at-match-end: "aaaa" is two
        // non-overlapping "aa"s ("bb"), not an "a"-cascade ("bbbb") and not
        // a single leftmost match ("baa").
        type ReplaceCase<'a> = (Vec<(&'a str, &'a str)>, &'a str, &'a str);
        let cases: Vec<ReplaceCase> = vec![
            // Longest at the same start, independent of list order.
            (vec![("ab", "X"), ("abc", "Y")], "abc", "Y"),
            (vec![("abc", "Y"), ("ab", "X")], "abc", "Y"),
            // Resume at the match end; nothing overlaps.
            (vec![("aa", "b")], "aaaa", "bb"),
            (vec![("aaaa", "z"), ("aa", "b")], "aaaaaa", "zb"),
            // Deletion via an empty value.
            (vec![("a", "")], "banana", "bnn"),
            // The output is NEVER re-scanned: the "ba" emitted for "a" is not
            // itself matched, and the "&amp;" emitted for "&" does not grow.
            (vec![("a", "ba")], "a", "ba"),
            (vec![("&", "&amp;")], "&amp;", "&amp;amp;"),
            (vec![("&", "&amp;")], "a&b&c", "a&amp;b&amp;c"),
            // An overlapping chain: "ab" consumes the shared "b" so "bc"
            // never fires at 0; the middle "c" is a gap, then "bc" fires.
            (vec![("ab", "X"), ("bc", "Y")], "abcbc", "XcY"),
            (vec![("abc", "Z"), ("bcd", "W")], "abcd", "Zd"),
            // Keys that are substrings of other keys, mid-text.
            (vec![("bc", "1"), ("b", "2")], "abcbc", "a11"),
            (vec![("b", "1"), ("abc", "2")], "zabcz", "z2z"),
            // Gaps between matches pass through untouched.
            (vec![("ab", "-")], "1ab2ab3", "1-2-3"),
            // Degenerate shapes.
            (vec![("xyz", "Q")], "abc", "abc"),
            (vec![("abc", "Q")], "", ""),
        ];
        for (pairs, text, expected) in &cases {
            assert_eq!(
                &*replace(pairs, text),
                *expected,
                "pairs {pairs:?} over {text:?}"
            );
            assert_eq!(
                reference_replace(pairs, text),
                *expected,
                "oracle disagrees on pairs {pairs:?} over {text:?}"
            );
        }
    }

    #[test]
    fn identity_answers_come_back_borrowed() {
        // The complete Cow contract: borrowed exactly when the answer equals
        // the input. No match at all: nothing is allocated.
        assert!(matches!(replace(&[("x", "y")], "abc"), Cow::Borrowed(s) if s == "abc"));
        // A value equal to its key: every match splices the same bytes back,
        // and the final out == text comparison hands back the input itself.
        assert!(matches!(replace(&[("a", "a")], "aaa"), Cow::Borrowed(s) if s == "aaa"));
        // Mutually-cancelling splices: delete the "b", regrow it inside the
        // "a"; the net effect over "ba" is the identity. The oracle pins the
        // value; the Cow variant pins the contract's second lane.
        assert_eq!(reference_replace(&[("b", ""), ("a", "ba")], "ba"), "ba");
        assert!(matches!(
            replace(&[("b", ""), ("a", "ba")], "ba"),
            Cow::Borrowed(s) if s == "ba"
        ));
        // And the other lane: a real change is owned.
        assert!(matches!(replace(&[("a", "b")], "aaa"), Cow::Owned(s) if s == "bbb"));
    }

    #[test]
    fn duplicate_keys_report_the_first_pairs_value() {
        // A dict cannot produce duplicate keys, but the slice spelling can;
        // the FIRST pair wins for the value: the remap routes every lookup
        // through the first index, whatever id the automaton hands back.
        assert_eq!(&*replace(&[("a", "1"), ("a", "2")], "aa"), "11");
        assert_eq!(
            &*replace(&[("ab", "X"), ("cd", "Y"), ("ab", "Z")], "abab"),
            "XX"
        );
        assert_eq!(&*replace(&[("a", ""), ("a", "keep")], "banana"), "bnn");
        // Duplicates of a key that LOSES to a longer one are inert either way.
        assert_eq!(
            &*replace(&[("ab", "1"), ("ab", "3"), ("abc", "2")], "abc"),
            "2"
        );
    }

    #[test]
    fn empty_replacement_list_answers_the_input_borrowed() {
        // The wrapper early-exits before this, but the core's own behavior on
        // an empty map is pinned too: an automaton over no keys finds nothing
        // (building it is legal), so the input comes back borrowed.
        for text in ["", "abc", "a longer text with no keys at all"] {
            assert!(matches!(replace(&[], text), Cow::Borrowed(s) if s == text));
        }
    }

    #[test]
    fn every_replacement_set_over_a_tiny_alphabet_matches_the_reference() {
        // The exhaustive sweep, replace side: all replacement lists of size
        // 0-2 over the keys {"a", "ab", "b"} with values from {"X", ""}
        // (repetition allowed, duplicate keys included) × all texts over
        // {"a", "b"} up to length 5, the complete small space of
        // longest-wins / resume / deletion / never-rescanned / identity
        // interactions, against the char-space reference. Each pair also
        // pins the Cow contract itself: borrowed exactly when the oracle's
        // answer equals the input. 2,709 pairs (43 lists × 63 texts), no
        // sampling.
        let keys = ["a", "ab", "b"];
        let values = ["X", ""];
        let per_slot = keys.len() * values.len();
        let mut sets: Vec<Vec<(&str, &str)>> = vec![Vec::new()];
        for size in 1..=2 {
            for combo in 0..per_slot.pow(size) {
                let mut list = Vec::with_capacity(size as usize);
                let mut rest = combo;
                for _ in 0..size {
                    let slot = rest % per_slot;
                    list.push((keys[slot % keys.len()], values[slot / keys.len()]));
                    rest /= per_slot;
                }
                sets.push(list);
            }
        }
        let mut texts: Vec<String> = Vec::new();
        for length in 0..=5 {
            for bits in 0..(1usize << length) {
                let text: String = (0..length)
                    .map(|i| if bits & (1 << i) == 0 { 'a' } else { 'b' })
                    .collect();
                texts.push(text);
            }
        }
        for set in &sets {
            for text in &texts {
                let got = replace(set, text);
                let want = reference_replace(set, text);
                assert_eq!(&*got, want, "pairs {set:?} over {text:?}");
                assert_eq!(
                    matches!(&got, Cow::Borrowed(_)),
                    want == text.as_str(),
                    "Cow contract: pairs {set:?} over {text:?}"
                );
            }
        }
    }

    fn replace_masked<'a>(
        replacements: &[(&str, &str)],
        text: &'a str,
        mask: char,
    ) -> Cow<'a, str> {
        replace_many_masked(text, replacements, mask)
            .unwrap_or_else(|err| panic!("automaton build failed: {err}"))
    }

    /// The pinned mask rule as a test-side spelling: the value's first `len`
    /// characters, or the value plus `len` − value_chars masks. Used to
    /// reconstruct expected SPAN contents in the offsets test; the golden
    /// literals and the oracle below stay independent of it.
    fn masked_value(value: &str, len: usize, mask: char) -> String {
        let value_chars = value.chars().count();
        if value_chars >= len {
            value.chars().take(len).collect()
        } else {
            let mut out = String::from(value);
            out.extend(std::iter::repeat_n(mask, len - value_chars));
            out
        }
    }

    /// The brute-force leftmost-longest MASKED-replace reference, in
    /// CHARACTER space (a `Vec<char>` view of the text, `str::starts_with`
    /// per position), the masked mirror of `reference_replace`, kept in
    /// the same lockstep discipline, with the masking spelled INLINE (not
    /// via `masked_value`) so the differential is against an independent
    /// implementation. The first occurrence of a duplicated key wins for
    /// the value by the same strict `>` that keeps the first on length
    /// ties.
    fn reference_replace_masked(replacements: &[(&str, &str)], text: &str, mask: char) -> String {
        let chars: Vec<char> = text.chars().collect();
        let pairs: Vec<(Vec<char>, &str)> = replacements
            .iter()
            .map(|(key, value)| (key.chars().collect(), *value))
            .collect();
        let mut out = String::new();
        let mut pos = 0usize;
        while pos < chars.len() {
            let mut best: Option<(usize, usize)> = None;
            for (idx, (key_chars, _)) in pairs.iter().enumerate() {
                if best.is_none_or(|(len, _)| key_chars.len() > len)
                    && !key_chars.is_empty()
                    && chars[pos..].starts_with(key_chars)
                {
                    best = Some((key_chars.len(), idx));
                }
            }
            match best {
                Some((len, idx)) => {
                    let mut emitted = 0usize;
                    for ch in pairs[idx].1.chars() {
                        if emitted == len {
                            break;
                        }
                        out.push(ch);
                        emitted += 1;
                    }
                    for _ in emitted..len {
                        out.push(mask);
                    }
                    pos += len;
                }
                None => {
                    out.push(chars[pos]);
                    pos += 1;
                }
            }
        }
        out
    }

    #[test]
    fn golden_masked_cases_are_pinned() {
        // Every row is checked against the masked oracle as well, the same
        // table discipline as the replace battery. The "[RE" row is the one
        // the rule pins hardest: the value is LONGER than the span, so
        // truncation fires and the mask is never consulted ("[RE", not
        // "[RE*" and not "[REDACTED]"); the "X**" row is the mirror.
        type MaskedCase<'a> = (Vec<(&'a str, &'a str)>, &'a str, char, &'a str);
        let cases: Vec<MaskedCase> = vec![
            // Truncation: a 10-char value into a 3-char span → its first 3.
            (
                vec![("cat", "[REDACTED]")],
                "the cat sat",
                '*',
                "the [RE sat",
            ),
            // Padding: a 1-char value into a 3-char span → value + 2 masks.
            (vec![("cat", "X")], "the cat sat", '*', "the X** sat"),
            // Deletion-shaped value: the empty value pads to pure mask.
            (vec![("cat", "")], "the cat sat", '*', "the *** sat"),
            // Exact length: neither branch visibly fires.
            (vec![("cat", "dog")], "the cat sat", '*', "the dog sat"),
            // Multi-match mixed: truncation and padding in one pass, the
            // gaps untouched in place ("sat" is 3 chars, "Z" pads to "Z**").
            (
                vec![("cat", "[REDACTED]"), ("sat", "Z")],
                "the cat sat",
                '*',
                "the [RE Z**",
            ),
            // Never re-scanned: the "ba" value would expose a fresh "a" to
            // a rescanning replace; here each 1-char span truncates to "b".
            (vec![("a", "ba")], "aa", '*', "bb"),
            // Value == key: the mask is never needed; the text survives.
            (vec![("cat", "cat")], "the cat sat", '*', "the cat sat"),
            // Multibyte MASK: padding counts characters, so a 2-byte mask
            // still pads one character per missing slot.
            (vec![("cat", "X")], "the cat sat", 'é', "the Xéé sat"),
            // CJK key over CJK text: a 2-char span, a 1-char value → 1 mask.
            (vec![("東京", "X")], "京都東京大阪", '#', "京都X#大阪"),
            // Multibyte value truncated: 4 chars of value into 2-char spans.
            (vec![("ab", "éééé")], "abab", '*', "éééé"),
            // Multibyte key, 2-char ASCII value truncated to the 1-char span.
            (vec![("é", "XY")], "ééé", '*', "XXX"),
            // Degenerate shapes.
            (vec![("xyz", "Q")], "abc", '*', "abc"),
            (vec![("abc", "Q")], "", '*', ""),
        ];
        for (pairs, text, mask, expected) in &cases {
            assert_eq!(
                &*replace_masked(pairs, text, *mask),
                *expected,
                "pairs {pairs:?} over {text:?} with mask {mask:?}"
            );
            assert_eq!(
                reference_replace_masked(pairs, text, *mask),
                *expected,
                "oracle disagrees on pairs {pairs:?} over {text:?} with mask {mask:?}"
            );
        }
    }

    #[test]
    fn masked_output_preserves_find_patterns_offsets() {
        // The spelling's whole point, made executable through the module's
        // own find_patterns the way a real pipeline runs it: compute the
        // match spans on the ORIGINAL text FIRST, redact, then assert the
        // redacted output sliced at those same spans is the masked value of
        // exactly the span's length, and every character OUTSIDE the spans
        // is the input's own, in place: the pre-redaction offsets stay
        // valid on the post-redaction text.
        type OffsetCase<'a> = (Vec<(&'a str, &'a str)>, &'a str, char);
        let cases: Vec<OffsetCase> = vec![
            (vec![("cat", "[REDACTED]")], "the cat sat", '*'),
            (vec![("bc", "1"), ("b", "22")], "abcbc", '#'),
            (vec![("cat", "X"), ("sat", "")], "the cat sat", 'é'),
            (vec![("東京", "éééé"), ("京都", "")], "京都東京京都", 'é'),
            (vec![("é", "XY")], "ééé", '*'),
        ];
        for (pairs, text, mask) in &cases {
            let keys: Vec<&str> = pairs.iter().map(|(key, _)| *key).collect();
            let matches = find(&keys, text);
            assert!(!matches.is_empty(), "battery row must match: {text:?}");
            let out = replace_masked(pairs, text, *mask);
            let text_chars: Vec<char> = text.chars().collect();
            let out_chars: Vec<char> = out.chars().collect();
            assert_eq!(
                out_chars.len(),
                text_chars.len(),
                "length invariant on pairs {pairs:?} over {text:?}"
            );
            let mut prev_end = 0usize;
            for &(start, end, pattern) in &matches {
                for i in prev_end..start {
                    assert_eq!(
                        out_chars[i], text_chars[i],
                        "gap char moved on pairs {pairs:?} over {text:?} at {i}"
                    );
                }
                let want: Vec<char> = masked_value(pairs[pattern].1, end - start, *mask)
                    .chars()
                    .collect();
                assert_eq!(
                    &out_chars[start..end],
                    &want[..],
                    "span content on pairs {pairs:?} over {text:?} at [{start}:{end}]"
                );
                prev_end = end;
            }
            for i in prev_end..text_chars.len() {
                assert_eq!(
                    out_chars[i], text_chars[i],
                    "gap char moved on pairs {pairs:?} over {text:?} at {i}"
                );
            }
        }
    }

    #[test]
    fn masked_length_invariant_holds_in_chars_and_byte_length_may_change() {
        // The guarantee is CHARACTERS (the Python-visible unit); BYTE length
        // is explicitly NOT preserved, pinned here so nobody mistakes the
        // contract. 1 char in → 1 char out, but 1 byte in → 2 bytes out.
        let out = replace_masked(&[("a", "é")], "a", '*');
        assert_eq!(&*out, "é");
        assert_eq!(out.chars().count(), 1);
        assert_eq!(out.len(), 2); // bytes; the input's was 1
        // Padding with a multibyte mask: 2 chars in → 2 chars out, 2 bytes
        // in → 4 bytes out.
        let out = replace_masked(&[("aa", "")], "aa", 'é');
        assert_eq!(&*out, "éé");
        assert_eq!(out.chars().count(), 2);
        assert_eq!(out.len(), 4); // bytes; the input's was 2
    }

    #[test]
    fn masked_identity_answers_come_back_borrowed() {
        // The complete Cow contract, masked side: borrowed exactly when the
        // answer equals the input. No match at all: nothing is allocated.
        assert!(matches!(
            replace_masked(&[("x", "y")], "abc", '*'),
            Cow::Borrowed(s) if s == "abc"
        ));
        // Value == key: every span splices the same characters back, the
        // mask never needed.
        assert!(matches!(
            replace_masked(&[("a", "a")], "aaa", '*'),
            Cow::Borrowed(s) if s == "aaa"
        ));
        // Truncation net-identity: the value's first L chars ARE the key,
        // including the canonical self-feeding pair, whose every "&" span
        // truncates "&amp;" back to "&".
        assert!(matches!(
            replace_masked(&[("ab", "abcd")], "abab", '*'),
            Cow::Borrowed(s) if s == "abab"
        ));
        assert!(matches!(
            replace_masked(&[("&", "&amp;")], "&amp;", '*'),
            Cow::Borrowed(s) if s == "&amp;"
        ));
        // Padding net-identity: value + masks regrow the key exactly, with
        // an ASCII mask and with a multibyte one.
        assert!(matches!(
            replace_masked(&[("ab", "a")], "ab", 'b'),
            Cow::Borrowed(s) if s == "ab"
        ));
        assert!(matches!(
            replace_masked(&[("東京", "東")], "東京", '京'),
            Cow::Borrowed(s) if s == "東京"
        ));
        // And the other lane: a real change is owned.
        assert!(matches!(
            replace_masked(&[("a", "b")], "aaa", '*'),
            Cow::Owned(s) if s == "bbb"
        ));
        assert!(matches!(
            replace_masked(&[("a", "")], "aaa", '*'),
            Cow::Owned(s) if s == "***"
        ));
    }

    #[test]
    fn empty_masked_replacement_list_answers_the_input_borrowed() {
        // The wrapper early-exits before this, but the core's own behavior
        // on an empty map is pinned too: an automaton over no keys finds
        // nothing (building it is legal), so the input comes back borrowed;
        // the mask is irrelevant when there is nothing to pad.
        for text in ["", "abc", "a longer text with no keys at all"] {
            assert!(matches!(
                replace_masked(&[], text, '*'),
                Cow::Borrowed(s) if s == text
            ));
        }
    }

    #[test]
    fn duplicate_keys_report_the_first_pairs_value_masked() {
        // Mirror of replace_many's duplicate pin: the FIRST pair wins for
        // the value (the shared replace_automaton remap routes the lookup),
        // whatever id the automaton hands back for the duplicated key.
        assert_eq!(
            &*replace_masked(&[("cat", "dog"), ("cat", "XXXXX")], "the cat sat", '*'),
            "the dog sat"
        );
        // Padding from the FIRST pair: ("a", "") pads each 1-char span to
        // one mask; the second pair's "ZZ" never fires.
        assert_eq!(&*replace_masked(&[("a", ""), ("a", "ZZ")], "aa", '#'), "##");
        // Duplicates of a key that LOSES to a longer key are inert either
        // way: "abc" beats "ab", and its own value pads to the 3-char span.
        assert_eq!(
            &*replace_masked(&[("ab", "1"), ("ab", "3"), ("abc", "2")], "abc", '*'),
            "2**"
        );
    }

    #[test]
    fn every_masked_replacement_set_over_a_tiny_alphabet_matches_the_reference_and_preserves_length()
     {
        // The exhaustive sweep, masked replace side: all replacement lists
        // of size 0-2 over the keys {"a", "ab", "b"} with values from
        // {"X", ""} (repetition allowed, duplicate keys included) × all
        // texts over {"a", "b"} up to length 5, one mask char: the same
        // complete small space the replace sweep covers, so the masked
        // spelling is differentially pinned over every longest-wins /
        // resume / deletion / never-rescanned / duplicate interaction it
        // can have there (the sweep's values are all shorter-or-equal, so
        // the TRUNCATION lane is carried by the golden battery and the
        // offsets rows above). Each row asserts three things: the
        // differential against the char-space oracle, the length invariant
        // (out chars == text chars, the spelling's whole point, on every
        // row), and the Cow contract (borrowed exactly when the oracle's
        // answer equals the input). 2,709 pairs (43 lists × 63 texts), no
        // sampling.
        let keys = ["a", "ab", "b"];
        let values = ["X", ""];
        let mask = '#';
        let per_slot = keys.len() * values.len();
        let mut sets: Vec<Vec<(&str, &str)>> = vec![Vec::new()];
        for size in 1..=2 {
            for combo in 0..per_slot.pow(size) {
                let mut list = Vec::with_capacity(size as usize);
                let mut rest = combo;
                for _ in 0..size {
                    let slot = rest % per_slot;
                    list.push((keys[slot % keys.len()], values[slot / keys.len()]));
                    rest /= per_slot;
                }
                sets.push(list);
            }
        }
        let mut texts: Vec<String> = Vec::new();
        for length in 0..=5 {
            for bits in 0..(1usize << length) {
                let text: String = (0..length)
                    .map(|i| if bits & (1 << i) == 0 { 'a' } else { 'b' })
                    .collect();
                texts.push(text);
            }
        }
        for set in &sets {
            for text in &texts {
                let got = replace_masked(set, text, mask);
                let want = reference_replace_masked(set, text, mask);
                assert_eq!(&*got, want, "pairs {set:?} over {text:?}");
                assert_eq!(
                    got.chars().count(),
                    text.chars().count(),
                    "length invariant on pairs {set:?} over {text:?}"
                );
                assert_eq!(
                    matches!(&got, Cow::Borrowed(_)),
                    want == text.as_str(),
                    "Cow contract: pairs {set:?} over {text:?}"
                );
            }
        }
    }

    // --- the compiled spelling (CompiledPatterns) --------------------------
    //
    // The parity gates: every compiled scan is the free function's own
    // scan over the held automaton, so the two spellings must agree
    // everywhere the free batteries reach, plus the call-time validation
    // contract pinned on its own error contents.

    /// Compiles `patterns` once, panicking on nothing (a build failure
    /// here is a bug in the test, not a case).
    fn compiled(patterns: &[&str]) -> CompiledPatterns {
        CompiledPatterns::build(patterns).unwrap_or_else(|err| panic!("build failed: {err}"))
    }

    /// The compiled find in the tuple shape the `find` helper speaks, so
    /// the parity assertions compare like with like.
    fn compiled_find(cp: &CompiledPatterns, text: &str) -> Vec<(usize, usize, usize)> {
        cp.find(text)
            .into_iter()
            .map(|m| (m.start, m.end, m.pattern))
            .collect()
    }

    #[test]
    fn compiled_find_and_count_parity_over_the_tiny_alphabet_sweep() {
        // The exhaustive sweep, compiled side: the same 2,520 pairs the
        // free spelling's own sweep covers (every pattern list of size 0-3
        // over {"a", "ab", "b"} × every text over {"a", "b"} up to length
        // 5), each list compiled ONCE and both find spellings and both
        // count spellings run over it, exact list/number equality with
        // the free functions and the char-space reference.
        let pool = ["a", "ab", "b"];
        let mut pattern_lists: Vec<Vec<&str>> = vec![Vec::new()];
        for size in 1..=3 {
            for combo in 0..pool.len().pow(size) {
                let mut list = Vec::with_capacity(size as usize);
                let mut rest = combo;
                for _ in 0..size {
                    list.push(pool[rest % pool.len()]);
                    rest /= pool.len();
                }
                pattern_lists.push(list);
            }
        }
        let mut texts: Vec<String> = Vec::new();
        for length in 0..=5 {
            for bits in 0..(1usize << length) {
                let text: String = (0..length)
                    .map(|i| if bits & (1 << i) == 0 { 'a' } else { 'b' })
                    .collect();
                texts.push(text);
            }
        }
        for patterns in &pattern_lists {
            let cp = compiled(patterns);
            for text in &texts {
                let found = compiled_find(&cp, text);
                assert_eq!(
                    found,
                    find(patterns, text),
                    "compiled/free find disagreement on {patterns:?} over {text:?}"
                );
                assert_eq!(
                    found,
                    reference(patterns, text),
                    "compiled/reference disagreement on {patterns:?} over {text:?}"
                );
                assert_eq!(
                    cp.count(text),
                    count(patterns, text),
                    "compiled/free count disagreement on {patterns:?} over {text:?}"
                );
            }
        }
    }

    #[test]
    fn compiled_find_parity_over_the_golden_and_multibyte_batteries() {
        // The golden overlap rows and the multibyte mapping battery's own
        // rows (extracted from the pinned expectations above), re-run
        // through one compiled fixture per row: the byte→char conversion
        // pass and the duplicate first-index rule must survive the
        // compiled route exactly.
        let family = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}";
        let alternating = format!("é{}é{}é", 'b', 'b');
        type Case<'a> = (Vec<&'a str>, String);
        let cases: Vec<Case> = vec![
            (vec!["abc", "abcd"], "abcd".to_string()),
            (vec!["abcd", "abc"], "abcd".to_string()),
            (vec!["bcd", "cd"], "abcdcd".to_string()),
            (vec!["ab", "abc", "abcd"], "abcdabcab".to_string()),
            (vec!["abc", "abc"], "abcabc".to_string()),
            (vec!["abcd", "abc", "abcd"], "abcd".to_string()),
            (Vec::new(), "abc".to_string()),
            (vec!["café"], "café café".to_string()),
            (vec!["cafe\u{301}"], "cafe\u{301} ok".to_string()),
            (vec!["東京", "京都"], "京都東京京都".to_string()),
            (vec![family], format!("hi{family}!")),
            (vec!["ab"], "éabéab".to_string()),
            (vec!["b", "é"], alternating),
            (vec!["éé", "é"], "ééé".to_string()),
        ];
        for (patterns, text) in &cases {
            let cp = compiled(patterns);
            assert_eq!(
                compiled_find(&cp, text),
                find(patterns, text),
                "compiled/free disagreement on {patterns:?} over {text:?}"
            );
            assert_eq!(
                cp.count(text),
                count(patterns, text),
                "compiled/free count disagreement on {patterns:?} over {text:?}"
            );
        }
    }

    #[test]
    fn compiled_replace_parity_over_the_tiny_alphabet_sweep() {
        // The exhaustive sweep, replace side: every replacement list of
        // size 0-2 over the keys {"a", "ab", "b"} with values {"X", ""},
        // each list's KEY SET compiled once and the same pairs validated
        // and spliced through the compiled route, exact equality with the
        // free replace and the char-space oracle, plus the Cow contract
        // (borrowed exactly when the oracle's answer equals the input).
        let keys = ["a", "ab", "b"];
        let values = ["X", ""];
        let per_slot = keys.len() * values.len();
        let mut sets: Vec<Vec<(&str, &str)>> = vec![Vec::new()];
        for size in 1..=2 {
            for combo in 0..per_slot.pow(size) {
                let mut list = Vec::with_capacity(size as usize);
                let mut rest = combo;
                for _ in 0..size {
                    let slot = rest % per_slot;
                    list.push((keys[slot % keys.len()], values[slot / keys.len()]));
                    rest /= per_slot;
                }
                sets.push(list);
            }
        }
        let mut texts: Vec<String> = Vec::new();
        for length in 0..=5 {
            for bits in 0..(1usize << length) {
                let text: String = (0..length)
                    .map(|i| if bits & (1 << i) == 0 { 'a' } else { 'b' })
                    .collect();
                texts.push(text);
            }
        }
        for set in &sets {
            let key_list: Vec<&str> = set.iter().map(|(key, _)| *key).collect();
            let cp = compiled(&key_list);
            let values_map = cp
                .replace_values(set)
                .unwrap_or_else(|err| panic!("validation failed on {set:?}: {}", err.message()));
            for text in &texts {
                let got = cp.replace_many(text, &values_map);
                let want = replace(set, text);
                assert_eq!(&*got, &*want, "pairs {set:?} over {text:?}");
                assert_eq!(
                    matches!(&got, Cow::Borrowed(_)),
                    matches!(&want, Cow::Borrowed(_)),
                    "Cow contract: pairs {set:?} over {text:?}"
                );
                let masked = cp.replace_many_masked(text, &values_map, '#');
                let want_masked = replace_masked(set, text, '#');
                assert_eq!(
                    &*masked, &*want_masked,
                    "masked pairs {set:?} over {text:?}"
                );
            }
        }
    }

    #[test]
    fn compiled_replace_validation_is_the_exact_key_set() {
        // The call-time contract, pinned on the error contents: unknown
        // keys and missing values are each refused, the exact key set
        // passes, and duplicates in the compiled LIST collapse to the one
        // dict key (the set semantics, not list-position semantics).
        let cp = compiled(&["cat", "catalogue", "cat"]);
        // The exact set passes (order-free: the pairs are a slice, the
        // dict spelling has no order at all).
        let pairs = [("catalogue", "Y"), ("cat", "X")];
        assert!(cp.replace_values(&pairs).is_ok());
        // An unknown key is refused and named.
        let err = cp
            .replace_values(&[("cat", "X"), ("dog", "Z")])
            .unwrap_err();
        assert_eq!(err.unknown, vec!["dog".to_string()]);
        assert_eq!(err.missing, 1); // "catalogue" left unvalued
        assert!(err.message().contains("\"dog\""));
        assert!(err.message().contains("1 pattern(s) missing a value"));
        // A missing pattern is refused with its count.
        let err = cp.replace_values(&[("cat", "X")]).unwrap_err();
        assert!(err.unknown.is_empty());
        assert_eq!(err.missing, 1);
        // The empty compiled list accepts only the empty map.
        let empty = compiled(&[]);
        assert!(empty.replace_values(&[]).is_ok());
        assert_eq!(
            empty.replace_values(&[("a", "b")]).unwrap_err(),
            ReplacementsMismatch {
                unknown: vec!["a".to_string()],
                missing: 0
            }
        );
        // Duplicates in the pairs slice collapse to the one key (set
        // semantics), and first-pair-wins routing is pinned through the
        // parity sweep above (duplicate keys in the free slice spelling):
        // here the duplicate-only pairing is a MISSING failure, because
        // "catalogue" stays unvalued.
        assert_eq!(
            cp.replace_values(&[("cat", "first"), ("cat", "second")])
                .unwrap_err()
                .missing,
            1
        );
        // len mirrors the source list, duplicates included.
        assert_eq!(cp.pattern_count(), 3);
        assert_eq!(empty.pattern_count(), 0);
    }

    #[test]
    fn compiled_matches_the_free_functions_over_multibyte_random_texts() {
        // The multibyte random sweep, compiled side: the same deterministic
        // LCG texts the free spelling's sweep builds, the compiled fixture
        // built once outside the loop, find/count/replace parity plus the
        // replace-side validation feeding a per-text value map keyed by
        // the fixed pattern set.
        let alphabet = ['a', 'b', 'é', '\u{301}', '東', '京', '\u{1F980}'];
        let patterns = ["éa", "東京", "\u{1F980}", "ab", "b\u{301}"];
        let cp = compiled(&patterns);
        let pairs: Vec<(&str, &str)> = patterns.iter().map(|p| (*p, "X")).collect();
        let values = cp.replace_values(&pairs).expect("the fixed set validates");
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for case in 0..200u64 {
            let length = (case % 40) as usize;
            let text: String = (0..length)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    alphabet[(state >> 33) as usize % alphabet.len()]
                })
                .collect();
            assert_eq!(
                compiled_find(&cp, &text),
                find(&patterns, &text),
                "case {case}: {text:?}"
            );
            assert_eq!(cp.count(&text), count(&patterns, &text), "case {case}");
            assert_eq!(
                &*cp.replace_many(&text, &values),
                &*replace(&pairs, &text),
                "case {case}: {text:?}"
            );
        }
    }
}
