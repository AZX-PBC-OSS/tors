use crate::detached_transform;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::parse_boundary;
use crate::truncate_impl;

/// `tors.truncate_to_bounds(text, max_chars, boundary="word")`: truncate to
/// at most `max_chars` codepoints, cutting at the last word (or, with
/// `boundary="sentence"`, sentence) boundary at or before `max_chars`
/// instead of mid-word/mid-sentence. Composes the crate's own
/// `word_bounds`/`sentence_bounds` segmentation (`src/truncate_impl.rs`), no
/// new dependency. The cut is also never mid-grapheme-cluster: a
/// word/sentence boundary that would split a cluster (e.g. Thai SARA AM,
/// combining accents, ZWJ emoji sequences, regional-indicator flag pairs;
/// `word_bounds` can score a combining mark as its own word-segment even
/// though it renders as one unit with the preceding character) is not a
/// valid cut point, so the result never separates a combining mark from its
/// base. If no boundary fits at or before `max_chars` (a single
/// word/sentence longer than the budget, or `max_chars == 0`), the fallback
/// is a hard cut at the largest grapheme boundary `<= max_chars`, still
/// documented, not a surprise, and still cluster-safe. The result never
/// exceeds `max_chars` codepoints either way, though it can fall short of
/// the budget when respecting a cluster boundary requires backing off
/// further. The cut point is then trimmed of trailing whitespace
/// (`str::trim_end`), since cutting right after a word/sentence boundary can
/// otherwise leave a dangling separator space.
///
/// Identity-return contract: `tors.truncate_to_bounds(s, n) is s` exactly
/// when `s` already has `<= n` codepoints (no truncation happens at all).
///
/// `max_chars < 0` and an unrecognized `boundary` both raise `ValueError`
/// before any work runs.
///
/// GIL model: `detached_transform`'s shape. The str-in argument borrow, the
/// segmentation scan and cut under `py.detach`, then either the original
/// object back (zero marshalling) or the O(output) truncated string.
#[pyfunction(signature = (text, max_chars, boundary = "word"))]
pub fn truncate_to_bounds(
    py: Python<'_>,
    text: Bound<'_, PyString>,
    max_chars: i64,
    boundary: &str,
) -> PyResult<Py<PyAny>> {
    if max_chars < 0 {
        return Err(PyValueError::new_err("max_chars must be >= 0"));
    }
    let boundary = parse_boundary(boundary)?;
    let max_chars = max_chars as usize;
    detached_transform(py, text, move |s| {
        truncate_impl::truncate_to_bounds(s, max_chars, boundary)
    })
}

/// `tors.truncate_ellipsis(text, max_chars)`: the DB-column truncation
/// shape: hard cut to at most `max_chars` codepoints plus a U+2026
/// `…` marker, never mid-grapheme-cluster. Unlike `truncate_to_bounds`
/// there is no word/sentence awareness: a storage bound is positional, not
/// semantic, and the marker tells the reader the value continues.
///
/// Contract (`src/truncate_impl.rs`): `<= max_chars` codepoints comes back
/// unchanged (identity return, `is s` exactly when no truncation happens);
/// otherwise the kept prefix is the largest cluster boundary at or before
/// `max_chars - 1` plus the marker, so the result never exceeds
/// `max_chars` (it falls short when cluster backoff requires it).
/// `max_chars == 0` yields `""` (there is no room for even the marker)
/// and `max_chars < 0` raises `ValueError` before any work runs. No
/// trailing-whitespace trim: the cut is positional.
///
/// GIL model: `detached_transform`'s shape, the same as
/// `truncate_to_bounds`.
#[pyfunction(signature = (text, max_chars))]
pub fn truncate_ellipsis(
    py: Python<'_>,
    text: Bound<'_, PyString>,
    max_chars: i64,
) -> PyResult<Py<PyAny>> {
    if max_chars < 0 {
        return Err(PyValueError::new_err("max_chars must be >= 0"));
    }
    let max_chars = max_chars as usize;
    detached_transform(py, text, move |s| {
        truncate_impl::truncate_ellipsis(s, max_chars)
    })
}
