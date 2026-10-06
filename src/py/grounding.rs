use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

use crate::grounding_impl;
use crate::py::_borrow::{EmptyPolicy, borrow_str_list, validate_unit_interval};

/// `tors.highlight(query, text, *, max_snippets=3, max_chars=400)`:
/// the best-matching snippet(s) of `text` for `query`, with CHARACTER
/// offsets into the original string — the span-level provenance a search
/// UI highlights and a citation deep-links to. See `src/grounding_impl.rs`
/// for the algorithm (a ROUGE-W-shaped F1 over UAX #29-tokenized anchor
/// runs, sentence-bounded) and its bounds.
///
/// Qualification: the WLCS fill is a monotone max-on-match recurrence
/// (Lin 2004, Eq. 15, with the forced-diagonal branch replaced by max to
/// preserve candidate monotonicity), a greedy-run-weighted alignment
/// score, not the literal weighted-LCS optimum. NOT bit-compatible with
/// the official ROUGE package or rouge-score; the deviation is deliberate
/// (monotonicity) and verified in tests.
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

/// `tors.ground_sentences(text, query, *, max_chars=None)`: EVERY UAX #29
/// sentence of `text`, scored against `query`: the batch bridge primitive
/// a downstream NLI verifier (MiniCheck/SummaC style) consumes. See
/// `src/grounding_impl.rs` for the algorithm (the same ROUGE-W-shaped F1,
/// the qualified monotone max-on-match recurrence, not the official ROUGE
/// package's numbers, that the snippet surface ranks spans with, over the
/// same sentence bounds `sentence_bounds` publishes) and the aggregate's
/// max policy. Note the argument order: `ground_sentences(text, query)`,
/// the OPPOSITE of `highlight(query, text)`.
///
/// Returns a `SentenceGrounding` dict: `{"sentences": [{"text", "start",
/// "end", "score"}, ...], "score": float}`, `sentences` ordered by
/// position (one entry per sentence, token-free sentences included at
/// score 0.0), each `text[start:end] == sentence["text"]` exactly (the
/// offsets are Python codepoint indices, round-tripping through CJK,
/// accents and emoji), and `score` the best sentence's ROUGE-W F1 in
/// `[0.0, 1.0]` (`0.0` when the query matches nothing). An empty or
/// token-free query scores every sentence 0.0 (the segmentation is the
/// answer's shape; the query only drives scores); an empty text returns
/// the empty result: degenerate input is a valid answer, never an error.
/// `max_chars` bounds each sentence's SCORED window (a too-long sentence
/// is scored over its leading token-boundary window; its reported span
/// still covers the whole sentence; the exact window boundary is an
/// implementation detail, deliberately not exposed); `None` scores whole
/// sentences, a value of 0 is an error.
///
/// GIL model: the argument borrows and the parameter validation under the
/// GIL, the whole segment/tokenize/score pass (linear in the text at a
/// bounded query width) under one `py.detach`, the result is plain data,
/// so the residue is only the O(sentences) dict marshalling.
#[pyfunction(signature = (text, query, *, max_chars = None))]
pub fn ground_sentences(
    py: Python<'_>,
    text: &str,
    query: &str,
    max_chars: Option<usize>,
) -> PyResult<Py<PyAny>> {
    if max_chars == Some(0) {
        return Err(PyValueError::new_err("max_chars must be positive"));
    }
    let result = py.detach(|| grounding_impl::ground_sentences(query, text, max_chars));
    let out = PyDict::new(py);
    let sentences = pyo3::types::PyList::empty(py);
    for s in &result.sentences {
        let d = PyDict::new(py);
        d.set_item("text", &s.text)?;
        d.set_item("start", s.start)?;
        d.set_item("end", s.end)?;
        d.set_item("score", s.score)?;
        sentences.append(d)?;
    }
    out.set_item("sentences", sentences)?;
    out.set_item("score", result.score)?;
    Ok(out.into_pyobject(py)?.into_any().unbind())
}

/// `tors.grounding_report(text, sources, query=None, *, threshold=0.85)`:
/// the production grounding-report composition, the pipeline shape the
/// Deepchecks "Grounded in Context" framework makes (Gerner et al. 2025,
/// "Grounded in Context: Retrieval-Based Method for Hallucination
/// Detection", arXiv 2504.15771: decompose the output into statements,
/// score each statement against the context, aggregate into one verdict),
/// with tors's lexical layer in place of the paper's NLI entailment
/// model. See `src/grounding_impl.rs` for the composition's exact
/// definitions.
///
/// Returns the report dict, every key present every time:
/// `{"sentences": [{"text", "start", "end", "best_source", "score",
/// "grounded"}, ...], "aggregate": {"grounded_ratio", "grounded",
/// "sentences", "mean_score", "best_score", "coverage", "query_score"}}`.
/// Each sentence entry is a UAX #29 sentence of `text` in position order
/// (its offsets are `sentence_bounds`' exact tuples, codepoint indices,
/// so `text[start:end] == sentence["text"]` exactly), `best_source` the
/// index of the source holding the sentence's best alignment (`None`
/// when there is none: no sources, or no source shares a token),
/// `score` that alignment's ROUGE-W F1 in `[0.0, 1.0]` (the per-pair
/// score is `highlight`'s own, the sentence scored as the query at the
/// family's default 1-snippet/400-char budget), and `grounded` the
/// threshold verdict (`score >= threshold` AND an alignment exists: no
/// sources grounds nothing, even at `threshold=0.0`). The aggregate:
/// `grounded_ratio` the grounded fraction (`0.0` for no sentences),
/// `grounded`/`sentences` the counts, `mean_score`/`best_score` the
/// per-sentence score mean/max, `coverage` the text-level token
/// utilization of the text against the newline-joined sources
/// (`grounding_coverage`'s recall twin, `0.0` with no sources), and
/// `query_score` the optional query's `ground_sentences` aggregate over
/// the text (the best sentence's F1 against the query: WHICH grounded
/// sentence reads first; `0.0` when `query` is `None`).
///
/// This is the LEXICAL layer, named as such: the paper's step 5 (NLI
/// entailment per claim-context pair) is the handoff a consumer's own
/// model takes over from, the same line `ground_sentences` documents.
/// `threshold` must be in `[0.0, 1.0]`.
///
/// GIL model: ONE detached pass for the WHOLE report (the argument
/// borrows and the threshold validation under the GIL, then the
/// segmentation, every per-(sentence, source) highlight score, the
/// coverage and the query pass inside a single `py.detach`); the
/// underlying primitives already detach their own passes when called
/// directly, and composing them here must not multiply that into one
/// detach per sentence, so the composition runs server-side, detached
/// once; the residue is the O(sentences) dict marshalling.
#[pyfunction(signature = (text, sources, query = None, *, threshold = 0.85))]
pub fn grounding_report(
    py: Python<'_>,
    text: &str,
    sources: Bound<'_, PyList>,
    query: Option<&str>,
    threshold: f64,
) -> PyResult<Py<PyAny>> {
    validate_unit_interval("threshold", threshold, false)?;
    // The bounded collect-handles-then-borrow walk (`borrow_str_list`):
    // each source borrowed zero-copy, the borrows alive across the
    // detach by the walk's own soundness argument (src/py/_borrow.rs).
    // An empty source entry is legal (an empty source aligns nothing,
    // highlight's documented degenerate answer).
    borrow_str_list(&sources, EmptyPolicy::Allow, |_items, borrowed| {
        let result =
            py.detach(|| grounding_impl::grounding_report(text, borrowed, query, threshold));
        let n = result.sentences.len();
        let sentences = PyList::empty(py);
        for s in &result.sentences {
            let d = PyDict::new(py);
            d.set_item("text", &s.text)?;
            d.set_item("start", s.start)?;
            d.set_item("end", s.end)?;
            d.set_item("best_source", s.best_source)?;
            d.set_item("score", s.score)?;
            d.set_item("grounded", s.grounded)?;
            sentences.append(d)?;
        }
        let aggregate = PyDict::new(py);
        aggregate.set_item("grounded_ratio", result.grounded_ratio)?;
        aggregate.set_item("grounded", result.grounded_count)?;
        aggregate.set_item("sentences", n)?;
        aggregate.set_item("mean_score", result.mean_score)?;
        aggregate.set_item("best_score", result.best_score)?;
        aggregate.set_item("coverage", result.coverage)?;
        aggregate.set_item("query_score", result.query_score)?;
        let out = PyDict::new(py);
        out.set_item("sentences", sentences)?;
        out.set_item("aggregate", aggregate)?;
        Ok(out.into_pyobject(py)?.into_any().unbind())
    })
}
