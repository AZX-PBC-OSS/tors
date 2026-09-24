use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::grounding_impl;

/// `tors.highlight(query, text, *, max_snippets=3, max_chars=400)`:
/// the best-matching snippet(s) of `text` for `query`, with CHARACTER
/// offsets into the original string — the span-level provenance a search
/// UI highlights and a citation deep-links to. See `src/grounding_impl.rs`
/// for the algorithm (ROUGE-W F1 over UAX #29-tokenized anchor runs,
/// sentence-bounded) and its bounds.
///
/// Returns a `GroundingResult` dict: `{"snippets": [{"text", "start", "end",
/// "score"}, ...], "score": float}` — `snippets` ordered by position,
/// non-overlapping, each `text[start:end] == snippet["text"]` exactly (the
/// offsets are Python codepoint indices, round-tripping through CJK, accents
/// and emoji), and `score` the best snippet's ROUGE-W F1 in `[0.0, 1.0]`
/// (`0.0` when there is no overlap at all). An empty or token-free query,
/// an empty or token-free text, and `max_snippets=0` all return the empty
/// result — degenerate input is a valid answer, never an error.
/// `max_chars` bounds every snippet's length at token boundaries (a snippet
/// always holds at least one token; a value of 0 is an error).
///
/// GIL model: the argument borrows and the parameter validation under the
/// GIL, the whole tokenize/score/select pass (the O(n·|Q|) scan) under one
/// `py.detach` — the result is plain data, so the residue is only the
/// O(snippets) dict marshalling.
#[pyfunction(signature = (query, text, *, max_snippets = 3, max_chars = 400))]
pub fn highlight(
    py: Python<'_>,
    query: &str,
    text: Bound<'_, PyString>,
    max_snippets: usize,
    max_chars: usize,
) -> PyResult<Py<PyAny>> {
    if max_chars == 0 {
        return Err(PyValueError::new_err("max_chars must be positive"));
    }
    let t = text.to_str()?;
    let result = py.detach(|| grounding_impl::highlight(query, t, max_snippets, max_chars));
    let out = PyDict::new(py);
    let snippets = pyo3::types::PyList::empty(py);
    for s in &result.snippets {
        let d = PyDict::new(py);
        d.set_item("text", &s.text)?;
        d.set_item("start", s.start)?;
        d.set_item("end", s.end)?;
        d.set_item("score", s.score)?;
        snippets.append(d)?;
    }
    out.set_item("snippets", snippets)?;
    out.set_item("score", result.score)?;
    Ok(out.into_pyobject(py)?.into_any().unbind())
}
