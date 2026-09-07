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
/// new dependency. The cut is ALSO never mid-grapheme-cluster: a
/// word/sentence boundary that would split a cluster (e.g. Thai SARA AM,
/// combining accents, ZWJ emoji sequences, regional-indicator flag pairs;
/// `word_bounds` can score a combining mark as its own word-segment even
/// though it renders as one unit with the preceding character) is not a
/// valid cut point, so the result never separates a combining mark from its
/// base. If no boundary fits at or before `max_chars` (a single
/// word/sentence longer than the budget, or `max_chars == 0`), the fallback
/// is a hard cut at the largest GRAPHEME boundary `<= max_chars`, still
/// documented, not a surprise, and still cluster-safe. The result NEVER
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
/// segmentation scan and cut under `py.detach`, then either the ORIGINAL
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
