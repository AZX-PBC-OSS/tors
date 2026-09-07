use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use rust_stemmers::Stemmer;

use crate::pipeline_impl;
use crate::py::_borrow::{EmptyPolicy, borrow_str_list};
use crate::py::lemma_dict::resolve_lemma_dict;
use crate::tokenize_impl::parse_stemmer_algorithm;

/// `tors.apply_pipeline(texts: list[str], *, nfd: bool = False, lowercase:
/// bool = False, strip_accents: bool = False, stemmer: str | None = None,
/// lemma_dict: dict[str, str] | CompiledLemmaDict | None = None,
/// collapse_whitespace: bool = False) -> list[str]`: a stateless,
/// GENERAL-PURPOSE batch text preprocessor: every requested step fused
/// into ONE GIL-released pass over the WHOLE list. See
/// `src/pipeline_impl.rs` for the exact, fixed-order pipeline
/// (`nfd` -> `lowercase` -> `strip_accents` -> `stemmer`/`lemma_dict` ->
/// `collapse_whitespace`, each skipped when its flag is off/`None`) and
/// why THE PIPELINE ITSELF is pure function composition, not a
/// `re.compile()`-style compiled-pipeline object: a stateful handle for
/// the whole pipeline adds no benefit over composing the fused steps
/// directly. `lemma_dict` specifically is the one narrow exception: see
/// below: because it alone has a measured, real per-call cost a
/// pre-built handle avoids; that does not reopen the case for a general
/// pipeline object.
///
/// All six steps default off, so `apply_pipeline(texts)` with no other
/// arguments is a true IDENTITY: the original `texts` list OBJECT comes
/// back unchanged (not just content-equal), the same zero-allocation
/// contract `normalize`/`quote`/`replace_many` already give when their own
/// transform is a no-op. An empty `texts` list returns `[]`. A non-`list`
/// argument or a non-`str` element raises `TypeError`; an invalid
/// `stemmer` name raises `ValueError` naming every valid choice
/// (`tokenize_impl::STEMMER_LANGUAGES`).
///
/// `lemma_dict` accepts a raw `dict[str, str]` or a pre-built
/// `tors.CompiledLemmaDict` (see its docs and `tf_idf`'s for the measured
/// per-call materialization cost a compiled handle avoids on repeated
/// calls against the same mapping). Anything else, or a `dict` with a
/// non-`str` key/value, raises `TypeError`.
///
/// **Relationship to `tf_idf`/`bm25_rank`**: those two already fuse the
/// SAME `strip_accents`/`stemmer`/`lemma_dict` knobs directly into their
/// own tokenization. Calling `apply_pipeline` first and then `tf_idf`/
/// `bm25_rank` on the result tokenizes TWICE for no benefit. Reach for
/// their own knobs when they're the only consumer; reach for
/// `apply_pipeline` to preprocess text feeding anything else
/// (`chunk_text`, `find_patterns`, your own logic).
///
/// GIL model: the list walk (borrowed `&str`s, zero-copy), building the
/// `Stemmer`, and resolving `lemma_dict` (an `Arc::clone` for a
/// `CompiledLemmaDict`, a fresh `HashMap` build for a raw `dict`) all
/// happen under the GIL (each done ONCE for the whole call, the same
/// amortization boundary `tf_idf`/`bm25_rank` use); every text's transform
/// runs under one `py.detach` over the whole list.
#[pyfunction(signature = (
    texts, *, nfd = false, lowercase = false, strip_accents = false,
    stemmer = None, lemma_dict = None, collapse_whitespace = false
))]
#[allow(clippy::too_many_arguments)]
pub fn apply_pipeline(
    py: Python<'_>,
    texts: Bound<'_, PyList>,
    nfd: bool,
    lowercase: bool,
    strip_accents: bool,
    stemmer: Option<&str>,
    lemma_dict: Option<&Bound<'_, PyAny>>,
    collapse_whitespace: bool,
) -> PyResult<Py<PyAny>> {
    let stemmer = stemmer
        .map(parse_stemmer_algorithm)
        .transpose()
        .map_err(PyValueError::new_err)?
        .map(Stemmer::create);
    let lemma_dict = resolve_lemma_dict(lemma_dict)?;

    // The argument-contract check (every element genuinely a `str`) runs
    // UNCONDITIONALLY via the shared walk (`_borrow.rs`'s soundness
    // story), before any identity short-circuit: a `TypeError` on a
    // non-`str` element must fire even when every transform step is off,
    // never silently pass through untouched.
    let out = borrow_str_list(&texts, EmptyPolicy::Allow, |_items, borrowed| {
        if !nfd
            && !lowercase
            && !strip_accents
            && stemmer.is_none()
            && lemma_dict.is_none()
            && !collapse_whitespace
        {
            // The identity path: every step is off, so no text can
            // possibly change. Signal the caller, which still owns
            // `texts`, to return the caller's ORIGINAL list object,
            // matching the zero-allocation contract this crate's other
            // identity-returning functions already give
            // (normalize/quote/replace_many), rather than building an
            // equal-but-new list.
            return Ok(None);
        }

        let out = py.detach(|| {
            pipeline_impl::apply_pipeline(
                borrowed,
                nfd,
                lowercase,
                strip_accents,
                stemmer.as_ref(),
                lemma_dict.as_deref(),
                collapse_whitespace,
            )
        });
        let result = PyList::empty(py);
        for s in out {
            result.append(PyString::new(py, &s))?;
        }
        Ok(Some(result.into_any().unbind()))
    })?;
    Ok(match out {
        Some(result) => result,
        None => texts.into_any().unbind(),
    })
}
