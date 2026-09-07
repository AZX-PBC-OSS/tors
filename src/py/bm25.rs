use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rust_stemmers::Stemmer;

use crate::bm25_impl;
use crate::py::lemma_dict::resolve_lemma_dict;
use crate::tokenize_impl::parse_stemmer_algorithm;

/// `tors.bm25_rank(query, corpus, *, k1=1.5, b=0.75, strip_accents=False,
/// stemmer=None) -> list[tuple[int, float]]`: Okapi BM25 score for every
/// document in `corpus` against `query`, one GIL-released native pass.
/// Returns `(index, score)` pairs for EVERY document (no top-k cutoff baked
/// in, slice/sort the result yourself), sorted by score descending, ties
/// broken by ascending original index. See `src/bm25_impl.rs` for the
/// exact formula (the always-non-negative `+1` IDF variant, not
/// the classic form) and its full worked example.
///
/// **This is a RERANKING primitive, not a search index.** It recomputes
/// corpus statistics from scratch on every call, which is the right shape
/// for scoring a small, already-retrieved candidate set (tens to a few
/// hundred documents) against one query, and the wrong shape for a large
/// corpus queried repeatedly, which wastes that recomputation every call.
/// For that, reach for a real search engine (`tantivy` is the mature
/// choice in Rust); tors does not build persistent index objects, the
/// same scope line that kept a Merkle inclusion-proof API out of this
/// crate. tors makes no claim about retrieval/relevance QUALITY for any
/// particular corpus or query: this is a correct implementation of a
/// well-known ranking FORMULA, not an AI/LLM performance promise.
///
/// `k1` (>= 0, default 1.5) tunes term-frequency saturation; `b` (in
/// `[0, 1]`, default 0.75) tunes length normalization. Both invalid
/// ranges raise `ValueError`. A non-`list` `corpus` or a non-`str` entry
/// raises `TypeError`. An empty `corpus` returns `[]`; an empty `query`
/// scores every document `0.0` (no query terms to accumulate a score
/// over, not an error).
///
/// `strip_accents`/`stemmer`/`lemma_dict` are `tf_idf`'s exact same opt-in
/// normalization knobs (see its docs for the accent-folding/stemming/
/// lemma-substitution details), applied IDENTICALLY to `query` and every
/// `corpus` document: required for the scores to mean anything, not just
/// a style choice. All default off, reproducing the original
/// lowercase-only tokenization. `lemma_dict` accepts a raw
/// `dict[str, str]` or a pre-built `tors.CompiledLemmaDict` (see its docs
/// and `tf_idf`'s for the measured cost this avoids on repeated calls
/// against the same mapping); anything else, or a `dict` with a non-`str`
/// key/value, raises `TypeError`.
///
/// GIL model: the corpus/query list walk (zero-copy `&str` borrows),
/// building the `Stemmer` (cheap, built ONCE, reused for every document),
/// and resolving `lemma_dict` (an `Arc::clone` for a `CompiledLemmaDict`,
/// a fresh `HashMap` build for a raw `dict`, either way ONCE for the whole
/// call) happen under the GIL; tokenization, corpus-statistics build, and
/// scoring all run under one `py.detach`.
#[pyfunction(signature = (query, corpus, *, k1 = 1.5, b = 0.75, strip_accents = false, stemmer = None, lemma_dict = None))]
#[allow(clippy::too_many_arguments)]
pub fn bm25_rank(
    py: Python<'_>,
    query: &str,
    corpus: Bound<'_, PyList>,
    k1: f64,
    b: f64,
    strip_accents: bool,
    stemmer: Option<&str>,
    lemma_dict: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<(usize, f64)>> {
    if k1 < 0.0 || !k1.is_finite() {
        return Err(PyValueError::new_err(format!(
            "k1 must be >= 0.0 and finite, got {k1}"
        )));
    }
    if !(0.0..=1.0).contains(&b) {
        return Err(PyValueError::new_err(format!(
            "b must be in [0.0, 1.0], got {b}"
        )));
    }
    let stemmer = stemmer
        .map(parse_stemmer_algorithm)
        .transpose()
        .map_err(PyValueError::new_err)?
        .map(Stemmer::create);
    let lemma_dict = resolve_lemma_dict(lemma_dict)?;
    let items: Vec<_> = corpus.iter().collect();
    let mut borrowed: Vec<&str> = Vec::with_capacity(items.len());
    for item in &items {
        borrowed.push(item.extract::<&str>()?);
    }
    Ok(py.detach(|| {
        bm25_impl::bm25_rank(
            query,
            &borrowed,
            k1,
            b,
            strip_accents,
            stemmer.as_ref(),
            lemma_dict.as_deref(),
        )
    }))
}
