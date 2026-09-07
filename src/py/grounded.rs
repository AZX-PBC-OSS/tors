use pyo3::exceptions::{PyTimeoutError, PyValueError};
use pyo3::prelude::*;

use crate::grounded_impl;
use crate::validate_deadline_ms;

/// `tors.is_grounded(claim, source, *, fuzzy=False, threshold=0.85,
/// deadline_ms=None)`: is `claim` grounded in `source`? Does `source`
/// actually contain it (`fuzzy=False`, the default, `source.contains(claim)`
/// exactly) or something close enough to it (`fuzzy=True`, a
/// windowed difflib-ratio scan against `threshold`: see
/// `src/grounded_impl.rs`'s module docs for precisely what the fuzzy score
/// measures and its DoS-bounded windowing; it is a LEXICAL check, not a
/// semantic/NLI one). An empty `claim` is vacuously grounded in anything on
/// both paths. `threshold` must be in `[0.0, 1.0]`. `deadline_ms` (only
/// meaningful, and only accepted, when `fuzzy=True`) bounds the whole fuzzy
/// scan the same way `diff_opcodes`' `deadline_ms` does: `TimeoutError` on
/// expiry, a positive-finite-or-`None` precondition validated before any
/// work runs.
///
/// GIL model: `fuzzy=False` is one `py.detach`'d `str::contains` call.
/// `fuzzy=True` runs the whole windowed scan under `py.detach`; the
/// `TimeoutError` (if any) is constructed after the GIL is reacquired, the
/// same `diff_opcodes` shape.
#[pyfunction(signature = (claim, source, *, fuzzy = false, threshold = 0.85, deadline_ms = None))]
pub fn is_grounded(
    py: Python<'_>,
    claim: &str,
    source: &str,
    fuzzy: bool,
    threshold: f64,
    deadline_ms: Option<f64>,
) -> PyResult<bool> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(PyValueError::new_err("threshold must be in [0.0, 1.0]"));
    }
    if !fuzzy && deadline_ms.is_some() {
        return Err(PyValueError::new_err(
            "deadline_ms is only meaningful when fuzzy=True",
        ));
    }
    if !fuzzy {
        return Ok(py.detach(|| grounded_impl::is_grounded_exact(claim, source)));
    }
    validate_deadline_ms(deadline_ms)?;
    py.detach(|| grounded_impl::is_grounded_fuzzy(claim, source, threshold, deadline_ms))
        .map_err(|err| PyErr::new::<PyTimeoutError, _>(err.message()))
}
