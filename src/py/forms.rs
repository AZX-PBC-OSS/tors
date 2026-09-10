use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::{Py, PyAny};

use crate::detached_transform;
use crate::forms_impl;

/// `tors.nfc`: `unicodedata.normalize("NFC", text)` in one GIL-released pass,
/// the standalone canonical-composition form, without the v0.1 pipeline's
/// folding/collapsing/strip stages (see `src/forms_impl.rs` for the parity
/// contract; the sweeps live in tests/test_parity.py). GIL model: same as
/// `tors.normalize`. Whole body under `py.detach`, zero-copy argument borrow
/// for ASCII/cached inputs, one-time O(input) UTF-8 materialization on the
/// first non-ASCII call, O(output) return marshalling.
///
/// v0.4 identity-return contract: a quick-check Yes (or an output==input
/// pass) returns the original object: `tors.nfc(s) is s` whenever
/// `tors.nfc(s) == s`, CPython's own `unicodedata.normalize` fast-path
/// behavior.
#[pyfunction]
pub fn nfc(py: Python<'_>, text: Bound<'_, PyString>) -> PyResult<Py<PyAny>> {
    detached_transform(py, text, forms_impl::nfc)
}

/// `tors.nfd`: `unicodedata.normalize("NFD", text)`, same contract (parity,
/// GIL model, v0.4 identity return) as `tors.nfc`.
#[pyfunction]
pub fn nfd(py: Python<'_>, text: Bound<'_, PyString>) -> PyResult<Py<PyAny>> {
    detached_transform(py, text, forms_impl::nfd)
}

/// `tors.nfkc`: `unicodedata.normalize("NFKC", text)`, canonical composition
/// with the compatibility (`<...>`) mappings applied (U+FB01 "ﬁ" -> "fi",
/// fullwidth -> halfwidth). Same contract and GIL model as `tors.nfc`.
#[pyfunction]
pub fn nfkc(py: Python<'_>, text: Bound<'_, PyString>) -> PyResult<Py<PyAny>> {
    detached_transform(py, text, forms_impl::nfkc)
}

/// `tors.nfkd`: `unicodedata.normalize("NFKD", text)`, compatibility
/// decomposition (the K-forms' decomposed spelling). Same contract and GIL
/// model as `tors.nfc`.
#[pyfunction]
pub fn nfkd(py: Python<'_>, text: Bound<'_, PyString>) -> PyResult<Py<PyAny>> {
    detached_transform(py, text, forms_impl::nfkd)
}
