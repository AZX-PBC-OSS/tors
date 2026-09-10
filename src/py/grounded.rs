use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::grounded_impl;
use crate::py::_borrow::{timeout_err, validate_unit_interval};
use crate::validate_deadline_ms;

/// `tors.is_grounded(claim, source, *, fuzzy=False, threshold=0.85,
/// deadline_ms=None)`: is `claim` grounded in `source`? Does `source`
/// actually contain it (`fuzzy=False`, the default, `source.contains(claim)`
/// exactly) or something close enough to it (`fuzzy=True`, a
/// windowed difflib-ratio scan against `threshold`: see
/// `src/grounded_impl.rs`'s module docs for precisely what the fuzzy score
/// measures and its DoS-bounded windowing; it is a lexical check, not a
/// semantic/NLI one). An empty `claim` is vacuously grounded in anything on
/// both paths. `threshold` must be in `[0.0, 1.0]`. `deadline_ms` (only
/// meaningful, and only accepted, when `fuzzy=True`) bounds the whole fuzzy
/// scan the same way `diff_opcodes`' `deadline_ms` does: `TimeoutError` on
/// expiry, a positive-finite-or-`None` precondition validated before any
/// work runs.
///
/// `fuzzy=True` is a superset of `fuzzy=False`: an exact-containment floor
/// (`memmem`, see grounded_impl's module docs for why not std's contains)
/// runs first, so a claim present verbatim is grounded before any
/// windowing: regardless of window alignment, and before `deadline_ms` is
/// even set up (a verbatim substring never times out; `deadline_ms` bounds
/// the windowed scan that runs only when there is no exact match). Every
/// windowed score uses the claim-length denominator `2*L` — a truncated
/// tail window's missing chars are mismatches, never a discounted
/// denominator, so the verdict never depends on where the evidence sits
/// relative to the source's end (issue #40); the one exception is a
/// `source` shorter than the claim, where the whole source is the evidence
/// and the score is one direct difflib `2*M/(m+n)` ratio. Near
/// matches are alignment-independent in the guarantee band: a region with
/// aligned ratio r is detected at any offset whenever
/// r >= max(0.75, threshold + 1/32), via the bounded refinement pass;
/// best-effort below r = 0.75, and evictable from the 64 refinement
/// candidates by adversarial decoys (the regime `deadline_ms` exists for);
/// see `grounded_impl`'s module docs.
///
/// GIL model: `fuzzy=False` is one `py.detach`'d `memmem` containment
/// check. `fuzzy=True` runs the whole windowed scan under `py.detach`; the
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
    validate_unit_interval("threshold", threshold, false)?;
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
        .map_err(|err| timeout_err(err.message()))
}
