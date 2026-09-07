use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rust_stemmers::Stemmer;

use crate::py::_borrow::{EmptyPolicy, borrow_str_list};
use crate::py::lemma_dict::resolve_lemma_dict;
use crate::tfidf_impl;
use crate::tokenize_impl::parse_stemmer_algorithm;

/// `tors.tf_idf(corpus: list[str], *, strip_accents: bool = False, stemmer:
/// str | None = None,
/// lemma_dict: dict[str, str] | CompiledLemmaDict | None = None) ->
/// list[list[tuple[str, float]]]`: one GIL-released native pass over the
/// whole corpus. See `src/tfidf_impl.rs` for the exact TF/IDF formulas (raw
/// term count; the scikit-learn-style smoothed IDF `ln((1 + N) / (1 + df)) + 1`)
/// and the tokenization (UAX #29 word segments, non-whitespace only,
/// lowercased). Every TF-IDF crate on crates.io is stale, the math is small
/// enough to hand-roll, and the stdlib has nothing here short of pulling in
/// scikit-learn/numpy for a lightweight pipeline.
///
/// Stateless: no vocabulary/vectorizer object persists between calls, each
/// call computes fresh over exactly the `corpus` given. Output is SPARSE:
/// one `(term, score)` list per document, holding only that document's own
/// terms, sorted alphabetically for determinism, never a
/// vocabulary-size-by-corpus-size dense structure. An empty corpus returns
/// `[]`; an empty-string document returns `[]` at its position (the output
/// always has exactly `len(corpus)` entries, position-matched to the
/// input). A non-`list` argument or a non-`str` entry raises `TypeError`.
///
/// `strip_accents=True` NFD-decomposes each token and drops combining
/// marks (`"café"` -> `"cafe"`) before scoring. See `src/tokenize_impl.rs`
/// for why this is unconditional (a real bug in scikit-learn's own
/// `strip_accents_unicode`, not replicated here). `stemmer` names a
/// Snowball algorithm (`"english"`, `"french"`, ...; see
/// `tokenize_impl::STEMMER_LANGUAGES` for the full list; an unrecognized
/// name raises `ValueError` naming every valid choice) applied after
/// lowercasing/accent-folding. `lemma_dict` is a caller-supplied
/// `word -> lemma` map, applied LAST (after any stemming). tors doesn't
/// bundle a lemma dictionary (that needs a per-language dataset or a POS
/// model, out of scope, the same boundary that kept schema-aware JSON/YAML
/// coercion out of this crate); it just applies one you supply, the same
/// shape `replace_many` already takes a caller-supplied map instead of a
/// bundled one. All three default OFF, reproducing the original
/// lowercase-only tokenization exactly.
///
/// `lemma_dict` accepts either a raw `dict[str, str]` (materialized into a
/// Rust `HashMap` fresh on this call) or a pre-built `tors.CompiledLemmaDict`
/// (`tors.CompiledLemmaDict(mapping)`, built once and reused across many
/// calls). This matters at real scale: a realistic 20,000-entry lemma
/// table (spaCy's and the Lemmatization Lists project's English tables
/// both land in that range) costs roughly 1.5ms to materialize, and a raw
/// `dict` pays that on every call: measured to make `apply_pipeline`
/// slower than an equivalent pure-Python loop at small batch sizes,
/// entirely from this one fixed cost. A `CompiledLemmaDict` pays it once;
/// every call after is an `Arc::clone`. Passing neither a `dict` nor a
/// `CompiledLemmaDict` (nor `None`), or a `dict` with a non-`str` key/
/// value, raises `TypeError`.
///
/// GIL model: the list walk (borrowed `&str`s, zero-copy) and resolving
/// `lemma_dict` (an `Arc::clone` for a `CompiledLemmaDict`, a fresh
/// `HashMap` build for a raw `dict`, either way ONCE for the whole call,
/// the same amortization boundary as building the `Stemmer`) happen under
/// the GIL; tokenization, counting, and scoring for the whole corpus run
/// under one `py.detach`.
#[pyfunction(signature = (corpus, *, strip_accents = false, stemmer = None, lemma_dict = None))]
pub fn tf_idf(
    py: Python<'_>,
    corpus: Bound<'_, PyList>,
    strip_accents: bool,
    stemmer: Option<&str>,
    lemma_dict: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<Vec<(String, f64)>>> {
    let stemmer = stemmer
        .map(parse_stemmer_algorithm)
        .transpose()
        .map_err(PyValueError::new_err)?
        .map(Stemmer::create);
    let lemma_dict = resolve_lemma_dict(lemma_dict)?;
    // The shared walk (`_borrow.rs`'s soundness story: handles alive
    // across the detach by construction), empty documents legal.
    borrow_str_list(&corpus, EmptyPolicy::Allow, |_items, borrowed| {
        Ok(py.detach(|| {
            tfidf_impl::tf_idf(
                borrowed,
                strip_accents,
                stemmer.as_ref(),
                lemma_dict.as_deref(),
            )
        }))
    })
}
