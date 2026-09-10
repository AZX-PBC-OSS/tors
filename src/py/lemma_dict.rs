//! `tors.CompiledLemmaDict` and the shared resolver behind `apply_pipeline`/
//! `tf_idf`/`bm25_rank`'s `lemma_dict` parameter.
//!
//! Materializing a Python `dict` into a Rust `HashMap<String, String>`
//! costs real, linear time: measured at roughly 1.45ms for a realistic
//! 20,000-entry lemma table (spaCy's and the Lemmatization Lists project's
//! English tables both land in that range) on a 100-document call. Passing
//! the same raw `dict` to `apply_pipeline`/`tf_idf`/`bm25_rank` pays that
//! cost on every single call; at small batch sizes that fixed cost can
//! exceed the actual tokenize-and-score work, making the native path
//! measurably slower than an equivalent pure-Python loop.
//!
//! `CompiledLemmaDict` is the `re.compile()` answer to that: build the
//! `HashMap` once, hold it in an `Arc`, and pass the handle to any of the
//! three functions instead of the raw `dict`. Every call after the first
//! is an `Arc::clone` (a refcount bump), not a re-walk of the mapping. It
//! is scoped narrowly and explicitly opt-in on purpose: no silent
//! identity-keyed cache is built inside the pyo3 layer, because a caller
//! mutating a `dict` in place between calls (same `id()`, different
//! content) would then silently see a stale mapping with no signal
//! anything is wrong. A `CompiledLemmaDict` is immutable once built and
//! the caller controls exactly when a new one gets created, so there is no
//! staleness question to get wrong.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A `lemma_dict` mapping, pre-built once. Immutable after construction
/// (the `HashMap` is only ever read, never mutated through this handle),
/// so sharing one `Arc` across many calls and threads is sound with no
/// synchronization beyond the refcount itself.
#[pyclass(name = "CompiledLemmaDict", frozen)]
pub struct CompiledLemmaDict {
    pub(crate) map: Arc<HashMap<String, String>>,
}

#[pymethods]
impl CompiledLemmaDict {
    /// `tors.CompiledLemmaDict(mapping)`: extracts `mapping` (a
    /// `dict[str, str]`) into a Rust `HashMap` once, under the GIL (the
    /// same linear-in-size cost `apply_pipeline`/`tf_idf`/`bm25_rank`
    /// already pay per call for a raw `dict`: this handle pays
    /// it here, a single time, instead). A non-`str` key or value raises
    /// `TypeError`, the same contract a raw `lemma_dict` argument has.
    #[new]
    fn new(mapping: &Bound<'_, PyDict>) -> PyResult<Self> {
        let map: HashMap<String, String> = mapping.extract()?;
        Ok(CompiledLemmaDict { map: Arc::new(map) })
    }

    /// The number of entries, mirroring `len(mapping)` on the source
    /// `dict`: a cheap sanity check a caller can run without touching
    /// the (unexposed) underlying mapping directly.
    fn __len__(&self) -> usize {
        self.map.len()
    }

    fn __repr__(&self) -> String {
        format!("CompiledLemmaDict({} entries)", self.map.len())
    }
}

/// Resolves a `lemma_dict` argument that may be `None`, a raw
/// `dict[str, str]`, or a `CompiledLemmaDict`, into one shared shape: an
/// owned `Arc` either way, so every call site downstream (`apply_pipeline`,
/// `tf_idf`, `bm25_rank`) treats the two input forms identically after
/// this point. The raw-`dict` branch pays the materialization cost this
/// call, exactly as before `CompiledLemmaDict` existed; the compiled-handle
/// branch is an `Arc::clone`, not a re-walk. Anything else raises the same
/// `TypeError` a bad `dict` extraction would.
pub fn resolve_lemma_dict(
    arg: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Arc<HashMap<String, String>>>> {
    let Some(arg) = arg else {
        return Ok(None);
    };
    if let Ok(compiled) = arg.cast::<CompiledLemmaDict>() {
        return Ok(Some(compiled.borrow().map.clone()));
    }
    if let Ok(dict) = arg.cast::<PyDict>() {
        let map: HashMap<String, String> = dict.extract()?;
        return Ok(Some(Arc::new(map)));
    }
    Err(PyTypeError::new_err(
        "lemma_dict must be a dict[str, str], a CompiledLemmaDict, or None",
    ))
}
