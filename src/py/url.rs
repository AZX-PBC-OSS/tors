use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::{Py, PyAny};

use crate::detached_transform;
use crate::url_impl;

/// `tors.quote(text, safe="/")`: `urllib.parse.quote(text, safe)` byte-exact
/// (UTF-8 percent-encoding, uppercase hex). Never quotes ASCII letters,
/// digits and `_.-~`, plus the ASCII members of `safe`; everything else
/// becomes `%XX` of its UTF-8 bytes. The stdlib quirks are pinned in
/// `tests/test_url_quote.py` (non-ASCII `safe` members are ignored since the
/// stdlib works byte-level; `%` in `safe` is honored) and derived in
/// `src/url_impl.rs`'s docs from Lib/urllib/parse.py. The stdlib spelling
/// is pure Python, a GIL-held whole-text pass for the most-used encoding
/// operation in web/ingestion pipelines; this is one detached native pass.
///
/// GIL model: `detached_transform`'s classes. The str-in borrow, the scan
/// under `py.detach`, then the identity lane (`tors.quote(s) is s` when
/// nothing encodes, strictly stronger than the stdlib, which always
/// re-decodes) or O(output) marshalling.
#[pyfunction(signature = (text, safe = "/"))]
pub fn quote(py: Python<'_>, text: Bound<'_, PyString>, safe: &str) -> PyResult<Py<PyAny>> {
    detached_transform(py, text, |s| url_impl::quote(s, safe))
}

/// `tors.quote_plus(text, safe="")`: `urllib.parse.quote_plus` byte-exact.
/// The quote semantics with every space becoming `+` (and a literal `+`
/// escaping to `%2B` unless caller-safed; the stdlib quotes with space
/// added to `safe`, then swaps, NOT "quote then replace `%20`", a quirk
/// pinned in the gate). Default `safe=""` is the stdlib's own.
#[pyfunction(signature = (text, safe = ""))]
pub fn quote_plus(py: Python<'_>, text: Bound<'_, PyString>, safe: &str) -> PyResult<Py<PyAny>> {
    detached_transform(py, text, |s| url_impl::quote_plus(s, safe))
}

/// `tors.unquote(text)`: `urllib.parse.unquote(text)` byte-exact. `%XX`
/// sequences (either hex case) decode as UTF-8 with `errors="replace"`
/// (invalid sequences → U+FFFD); anything that is not a valid escape
/// (`%zz`, a trailing `%`) stays verbatim; `+` is NOT special here. The
/// fragmentation quirk is pinned in the gate: each ASCII run unquotes
/// independently, so `%C3é%A9` is NOT `éé`. Identity lane: no `%` at all
/// returns the ORIGINAL object (exactly the stdlib's own `'%' not in
/// string` fast path).
#[pyfunction]
pub fn unquote(py: Python<'_>, text: Bound<'_, PyString>) -> PyResult<Py<PyAny>> {
    detached_transform(py, text, url_impl::unquote)
}

/// `tors.unquote_plus(text)`: `urllib.parse.unquote_plus` byte-exact.
/// Every `+` becomes a space FIRST (so `%2B` survives as a literal `+`
/// while a raw `+` becomes a space; order is semantics, pinned), then the
/// unquote semantics. Identity lane: no `+` and no `%` returns the
/// ORIGINAL object.
#[pyfunction]
pub fn unquote_plus(py: Python<'_>, text: Bound<'_, PyString>) -> PyResult<Py<PyAny>> {
    detached_transform(py, text, url_impl::unquote_plus)
}
