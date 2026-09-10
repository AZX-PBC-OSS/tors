use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use pyo3::{Py, PyAny};
use std::borrow::Cow;

use crate::EagerIter;
use crate::py::_borrow::{EmptyPolicy, borrow_dict_pairs, borrow_str_list};
use crate::py::eager_iter_class;
use crate::search_impl;

/// The GIL-held pattern-list walk + detached run shared by `find_patterns`,
/// `find_patterns_iter`, and `count_matches`: the `_borrow::borrow_str_list`
/// walk (its doc carries the handles-alive-across-the-detach soundness
/// story, true by construction there), patterns refused empty with
/// `ValueError("empty pattern")`, then `run` over the borrows and the text
/// under one `py.detach` and a build failure mapped to a `ValueError`
/// carrying the engine's message. One function, three spellings, so the
/// walk and the run cannot drift apart.
fn run_over_borrowed_patterns<R: Send>(
    py: Python<'_>,
    patterns: &Bound<'_, PyList>,
    text: &str,
    run: impl FnOnce(&[&str], &str) -> Result<R, search_impl::BuildError> + Send,
) -> PyResult<R> {
    borrow_str_list(patterns, EmptyPolicy::Refuse, |_items, borrowed| {
        py.detach(|| run(borrowed, text))
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })
}

/// The `(start, end, pattern_index)` triple shape shared by the list and
/// iterator spellings of the search.
fn matches_into_triples(matches: Vec<search_impl::PatternMatch>) -> Vec<(usize, usize, usize)> {
    matches
        .into_iter()
        .map(|m| (m.start, m.end, m.pattern))
        .collect()
}

eager_iter_class! {
    /// The streaming search (v0.9): the same iterator design over
    /// `find_patterns`' matches, yielding the same `(start, end, pattern_index)`
    /// triples as the list API (pinned to sequence-parity), the whole search
    /// (pattern-list walk, automaton build, scan, byte→char conversion) under
    /// one detach at construction, one 3-tuple of ints per `__next__`. See
    /// `find_patterns_iter`'s docs for the marshalling-caveat answer this is.
    FindPatternsIter, (usize, usize, usize);
}

/// `tors.find_patterns(patterns, text)`: leftmost-longest, non-overlapping
/// multi-pattern substring search in one GIL-released native pass: every
/// occurrence of every pattern, reported as `(start, end, pattern_index)`
/// with `end` exclusive, offsets in Python `str` index (codepoint) units:
/// `text[start:end] == patterns[pattern_index]` for every reported match.
/// The engine is `aho-corasick`'s `MatchKind::LeftmostLongest` (see
/// `src/search_impl.rs` for the semantics and the byte→char offset mapping;
/// the differential-oracle contract is tests/test_find_patterns.py):
///
/// - **leftmost**: a match is reported at the earliest position any pattern
///   matches;
/// - **longest**: among the patterns matching at that position, the longest
///   wins regardless of list order (not regex-alternation leftmost-first
///   priority: a shorter earlier-listed pattern never beats a longer one);
/// - **non-overlapping**: the scan resumes at each match's end; matches come
///   back in strictly increasing start order;
/// - **duplicates report the first index**: identical pattern strings are
///   legal, and a match of that string reports its lowest list index.
///
/// Argument contract, each decision pinned in tests/test_find_patterns.py:
/// `patterns` must be exactly a `list` of `str` (a tuple or a non-`str` entry
/// raises `TypeError`); an empty pattern string raises
/// `ValueError("empty pattern")`, since it would match at every position and
/// has no leftmost-longest meaning; an empty patterns list returns `[]`
/// immediately, without building an automaton; lone surrogates raise
/// `UnicodeEncodeError` at the argument boundary (the standard str-in
/// boundary). An automaton build can still fail on engine limits (a single
/// pattern spanning more than the engine's u32 offset budget: 4 GiB of
/// pattern); that maps to a `ValueError` carrying the engine's message.
///
/// GIL model: the argument side walks the pattern list once under the GIL,
/// borrowing each entry's UTF-8 zero-copy (the standard str-in borrow class,
/// O(patterns) handles; the one-time O(input) UTF-8 materialization applies
/// per non-ASCII pattern object on first call, and to `text` as usual); the
/// whole search (automaton build, scan, and the byte→char offset conversion,
/// with the `is_ascii` fast path) runs under one `py.detach`; the
/// GIL-held residue is the return marshalling, one 3-tuple of ints per match,
/// O(matches), the `word_bounds`/`diff_opcodes` list-shape class. The
/// measured band (sparse vs matches-heavy 12 MiB cells) is recorded in
/// docs/performance.md and pinned by tests/test_gil_release.py.
#[pyfunction]
pub fn find_patterns(
    py: Python<'_>,
    patterns: Bound<'_, PyList>,
    text: &str,
) -> PyResult<Vec<(usize, usize, usize)>> {
    if patterns.is_empty() {
        return Ok(Vec::new());
    }
    let matches = run_over_borrowed_patterns(py, &patterns, text, search_impl::find_patterns)?;
    Ok(matches_into_triples(matches))
}

/// `tors.count_matches(patterns, text)`: the count spelling of
/// `find_patterns` (v0.9): the same leftmost-longest, non-overlapping
/// search, answering just the number: `count_matches(p, t) ==
/// len(find_patterns(p, t))`, pinned. Why a separate function: the count is
/// the common question at corpus scale, and the list spelling materializes
/// the whole match vector to answer it (measured: ~250 MiB of `Vec` for the
/// 100 MiB dense corpus's 10.7M matches); the count core builds the same
/// automaton and counts the scan's matches with O(1) memory, and counting
/// is offset-free, so the byte→char conversion pass does not exist here
/// either. Duplicates cannot change a count (identical patterns report the
/// same span once, whatever id the automaton prefers).
///
/// Argument contract: exactly `find_patterns`'s: exactly a `list` of `str`
/// (a tuple or non-`str` entry raises `TypeError`), an empty pattern string
/// raises `ValueError("empty pattern")`, an empty list returns `0`
/// immediately without building an automaton, lone surrogates raise
/// `UnicodeEncodeError` at the standard str-in boundary.
///
/// GIL model: the `find_patterns` classes exactly, with the one GIL-held
/// pattern-list walk (`borrow_patterns`), the automaton build + scan under
/// one `py.detach`, and a single int return (the `grapheme_count` extreme
/// point: no marshalling class at all).
#[pyfunction]
pub fn count_matches(py: Python<'_>, patterns: Bound<'_, PyList>, text: &str) -> PyResult<usize> {
    if patterns.is_empty() {
        return Ok(0);
    }
    run_over_borrowed_patterns(py, &patterns, text, search_impl::count_matches)
}

/// `tors.find_patterns_iter(patterns, text)`: the streaming spelling of
/// `find_patterns` (v0.9), the `word_bounds_iter` design over matches: a
/// lazy iterator yielding the same `(start, end, pattern_index)` triples,
/// in the same order, as the list API (pinned to sequence-parity). The
/// whole search (the pattern-list walk, automaton build, scan, and the
/// byte→char offset conversion) fills an internal buffer under one
/// `py.detach` at construction (the match buffer is 24 bytes per match
/// against the list shape's Python tuples), and each `__next__` then holds
/// the GIL only to construct one 3-tuple of ints, µs-scale. This is the
/// streaming answer to the O(matches) list-marshalling caveat the v0.7
/// section records: ~13 ms held per 100k matches in the list shape's tuple
/// construction becomes ~nothing per item, and a whole-corpus sweep that
/// would hold ~170 ms holds the ping floor instead.
///
/// Eager-at-construction is the `word_bounds_iter` argument over again:
/// leftmost-longest semantics need the scan's resume-at-match-end state, so
/// the matches cannot be produced lazily without re-scanning per item; the
/// buffer is filled by one whole-text pass up front.
///
/// Argument contract: exactly `find_patterns`'s (an empty list yields an
/// empty iterator without building an automaton; an empty pattern raises
/// `ValueError("empty pattern")`).
#[pyfunction]
pub fn find_patterns_iter(
    py: Python<'_>,
    patterns: Bound<'_, PyList>,
    text: Bound<'_, PyString>,
) -> PyResult<Py<FindPatternsIter>> {
    let s = text.to_str()?;
    let items: Vec<(usize, usize, usize)> = if patterns.is_empty() {
        Vec::new()
    } else {
        matches_into_triples(run_over_borrowed_patterns(
            py,
            &patterns,
            s,
            search_impl::find_patterns,
        )?)
    };
    Py::new(py, FindPatternsIter(EagerIter::new(py, text, items)))
}

/// `tors.replace_many(text, replacements)`: simultaneous multi-pattern
/// replace in one GIL-released native pass (v0.8): every occurrence of every
/// key in the dict is replaced by its value, with `find_patterns`'s exact
/// search semantics: leftmost-longest (among the keys matching at a
/// position, the longest wins regardless of dict order; not regex
/// alternation's leftmost-first priority), non-overlapping (the scan resumes
/// at each match's end), and the replacement output is never re-scanned (a
/// value that itself contains a key does not cascade, the double-replace
/// guard, the `&#38;amp;` discipline). No offset conversion exists here
/// (the output is spliced strings, not reported offsets); one automaton
/// build + one scan + one output build, O(text + output).
///
/// This is the replace primitive CPython does not have: chained
/// `str.replace` calls are N whole-text GIL-held passes, and `re.sub` with
/// an alternation is leftmost-first and rescans its own output, so neither
/// has these semantics. The canonical consumers are redaction and
/// normalization maps (a PII scrub list, a terminology rewrite table),
/// exactly the `find_patterns` + replace pipeline in one call.
///
/// Identity-return contract (the v0.4 idiom, complete form): a map whose
/// keys never match returns the original input object, and so does a map
/// whose net effect is the identity: `tors.replace_many(s, m) is s`
/// exactly when `tors.replace_many(s, m) == s`.
///
/// Argument contract, each pinned in tests/test_replace_many.py: `text`
/// must be exactly `str`; `replacements` must be exactly a `dict` of
/// `str -> str` (a non-`str` key or value raises `TypeError`; a tuple or
/// list of pairs raises `TypeError`, since the dict is the ergonomic shape
/// and keys are unique by construction, which is what makes the semantics
/// order-free); an empty key raises `ValueError("empty pattern")` (it would
/// match at every position, the `find_patterns` contract); an empty dict
/// returns the input object immediately, without building an automaton;
/// lone surrogates raise `UnicodeEncodeError` at the argument boundary (the
/// standard str-in boundary, paid by the text and every key and value).
/// Dict order cannot matter (leftmost-longest, unique keys), pinned.
/// An automaton build can still fail on engine limits (a key spanning more
/// than the engine's u32 offset budget, 4 GiB); that maps to a
/// `ValueError` carrying the engine's message.
///
/// GIL model: the `find_patterns` argument shape over a dict: one
/// GIL-held walk borrowing each key and value (the standard str-in borrow
/// class, O(entries) handles; the one-time O(input) UTF-8 materialization
/// applies per non-ASCII key/value object on first call, and to `text` as
/// usual), then automaton build + scan + splice under one `py.detach`, and
/// either the identity return (zero marshalling) or the O(output) string
/// marshalling. No list-shape class exists: the return is one string.
#[pyfunction]
pub fn replace_many(
    py: Python<'_>,
    text: Bound<'_, PyString>,
    replacements: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    // The str-in boundary first (the crate's contract: the borrow's
    // UnicodeEncodeError for lone surrogates fires before any dict walk,
    // exactly as `find_patterns`'s `text: &str` extraction does).
    let s = text.to_str()?;
    if replacements.is_empty() {
        return Ok(text.into_any().unbind());
    }
    // The shared dict walk (`_borrow.rs`'s soundness story: handles alive
    // across the detach by construction), empty keys refused.
    let out = borrow_dict_pairs(&replacements, |pairs| {
        py.detach(|| search_impl::replace_many(s, pairs))
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    match out {
        Cow::Borrowed(_) => Ok(text.into_any().unbind()),
        Cow::Owned(out) => Ok(out.into_pyobject(py)?.into_any().unbind()),
    }
}

/// `tors.replace_many_masked(text, replacements, mask="*")`: the
/// length-preserving spelling of `replace_many`, the redaction shape: the
/// same leftmost-longest, non-overlapping, never-rescanned scan, but every
/// matched span of L characters is replaced by the value truncated to L
/// characters (value longer) or the value followed by `L − len(value)`
/// copies of `mask` (value shorter), so the output's character length and
/// every non-matching span's offsets are exactly the input's, which is
/// what redaction pipelines need (every offset computed before redaction,
/// find_patterns results, word/sentence bounds, stays valid after it).
/// Byte length may change (a multibyte value/mask replaces the matched
/// bytes); the guarantee is character/offset preservation, the
/// Python-visible one. Worked rows: `{"cat": "[REDACTED]"}` mask `"*"` on
/// `"the cat sat"` → `"the [RE sat"` (truncation, no `*` anywhere);
/// `{"cat": "X"}` → `"the X** sat"` (padding). `mask` must be exactly one
/// character (`ValueError` otherwise: the length arithmetic requires it);
/// the rest of the argument contract is `replace_many`'s exactly (exactly a
/// `dict[str, str]`, empty key `ValueError("empty pattern")`, lone
/// surrogates `UnicodeEncodeError`), and the identity contract holds
/// complete: `replace_many_masked(s, m, c) is s` exactly when `== s`.
///
/// GIL model: `replace_many`'s classes exactly, with one GIL-held dict walk
/// (borrows of every key and value plus the one-character mask), the
/// automaton build + scan + splice under one `py.detach`, and either the
/// identity return or the O(output) string marshalling.
#[pyfunction(signature = (text, replacements, mask = "*"))]
pub fn replace_many_masked(
    py: Python<'_>,
    text: Bound<'_, PyString>,
    replacements: Bound<'_, PyDict>,
    mask: &str,
) -> PyResult<Py<PyAny>> {
    // The str-in boundary first (the crate's contract), then the mask check.
    let s = text.to_str()?;
    let mut mask_char = mask.chars();
    let (Some(mask_char), None) = (mask_char.next(), mask_char.next()) else {
        return Err(PyValueError::new_err(
            "mask must be exactly one character (the length arithmetic requires it)",
        ));
    };
    if replacements.is_empty() {
        return Ok(text.into_any().unbind());
    }
    // The shared dict walk as replace_many (`_borrow.rs`'s soundness
    // story: handles alive across the detach by construction, empty keys
    // refused).
    let out = borrow_dict_pairs(&replacements, |pairs| {
        py.detach(|| search_impl::replace_many_masked(s, pairs, mask_char))
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    match out {
        Cow::Borrowed(_) => Ok(text.into_any().unbind()),
        Cow::Owned(out) => Ok(out.into_pyobject(py)?.into_any().unbind()),
    }
}
