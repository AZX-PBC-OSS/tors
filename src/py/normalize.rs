use crate::finalize_impl;
use std::borrow::Cow;

use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::{Py, PyAny};

use crate::controls_impl;
use crate::detached_transform;
use crate::normalize_impl;

/// `tors.normalize` releases the GIL for the whole call via `py.detach`; see the crate
/// GIL model above for the call's GIL-held residue.
///
/// v0.4 identity-return contract: when the COMPLETE pipeline is a no-op (NFC
/// quick-check Yes + no CR + no `[ \t]` before a newline + no 3+ newline run
/// + no strip delta, a handful of SIMD sentinel scans), the ORIGINAL input
/// object comes back: `tors.normalize(s) is s`. Inputs the probe cannot
/// prove clean still run the scan, and an output==input comparison after it
/// extends the same object identity to them, so the contract is complete:
/// `tors.normalize(s) is s` whenever `tors.normalize(s) == s`. A quick-check
/// Yes with a dirty scan skips only the NFC materialization (the input is
/// already NFC; the fold/drop/collapse/strip scan still runs).
#[pyfunction]
pub fn normalize(py: Python<'_>, text: Bound<'_, PyString>) -> PyResult<Py<PyAny>> {
    detached_transform(py, text, normalize_impl::normalize_cow)
}

/// `tors.finalize`: `(normalize(text), sha256-hex-of-the-normalized-utf8)` in the same
/// single GIL-released pass. The hash tail exists so a normalize-then-content-hash
/// tail collapses into one `py.detach` call instead of a second GIL-held `encode` +
/// `hashlib.sha256` walk over the whole text.
/// See the crate GIL model above for the call's GIL-held residue.
///
/// v0.4: the SHA-256 is integrated into the transform's output pass (fed the
/// confirmed output bytes as the scan writes them: one pass, not two), and
/// on the identity path the digest is computed straight from the borrowed
/// input buffer while the STRING element is the ORIGINAL object
/// (`tors.finalize(s)[0] is s`), so no output allocation happens at all.
///
/// Lone surrogates (a `str` CPython can hold but UTF-8 cannot encode, e.g. from a
/// `surrogatepass` decoder) are refused at the argument boundary with `UnicodeEncodeError`
/// ("surrogates not allowed") before any Rust code runs: this is the `to_str` borrow's
/// behavior (the same `PyUnicode_AsUTF8AndSize` call pyo3's `&str` extraction
/// uses), measured and pinned in `tests/test_finalize.py::TestSurrogateBehavior`.
#[pyfunction]
pub fn finalize(py: Python<'_>, text: Bound<'_, PyString>) -> PyResult<(Py<PyAny>, String)> {
    let s = text.to_str()?;
    let (identity, out, digest) = py.detach(|| match finalize_impl::finalize_checked(s) {
        (Cow::Borrowed(_), digest) => (true, String::new(), digest),
        (Cow::Owned(out), digest) => (false, out, digest),
    });
    let normalized = if identity {
        text.into_any().unbind()
    } else {
        out.into_pyobject(py)?.into_any().unbind()
    };
    Ok((normalized, digest))
}

/// `tors.strip_controls`: replace every maximal run of C0 controls
/// (`U+0000`–`U+001F`, tabs and newlines included) and DEL (`U+007F`) with
/// a single ASCII space — the scrub model-authored display text needs
/// before it is stored or served. Byte-identical to
/// `re.compile(r"[\x00-\x1f\x7f]+").sub(" ", text)`; C1 controls
/// (`U+0080`–`U+009F`) pass through untouched (see `src/controls_impl.rs`
/// for why that exclusion is load-bearing, not an oversight).
///
/// Identity-return contract: `tors.strip_controls(s) is s` exactly when
/// `s` holds no C0/DEL character. No strip of the edges: a control run at
/// either end becomes an edge space for the caller to `.strip()`.
///
/// GIL model: `detached_transform`'s shape, the same as `normalize`.
#[pyfunction]
pub fn strip_controls(py: Python<'_>, text: Bound<'_, PyString>) -> PyResult<Py<PyAny>> {
    detached_transform(py, text, controls_impl::strip_controls)
}
