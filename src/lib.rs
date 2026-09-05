mod normalize_impl;

use pyo3::prelude::*;

/// `tors.normalize` releases the GIL for the whole call via `py.detach` (pyo3 0.29's rename
/// of `allow_threads`) — the point of `tors`: Python's `re`/`str` methods never release the
/// GIL regardless of input size, so this is one native pass instead of cennan's
/// GIL-hold-bounding chunked `re.sub` calls.
#[pyfunction]
fn normalize(py: Python<'_>, text: &str) -> String {
    py.detach(|| normalize_impl::normalize(text))
}

#[pymodule]
fn _tors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    Ok(())
}
