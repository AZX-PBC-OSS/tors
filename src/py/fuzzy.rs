use crate::diff_impl;
use pyo3::exceptions::{PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::{Py, PyAny};

use crate::fuzzy_impl;
use crate::validate_deadline_ms;

/// `tors.similarity_ratio(a, b, *, deadline_ms=None)`:
/// `difflib.SequenceMatcher(None, a, b).ratio()`'s shape at native speed:
/// `2.0*M/T` with M the matched total over THE SAME Myers equal-ops
/// `diff_opcodes` emits (see `src/diff_impl.rs`'s similarity docs for the
/// validity-first parity contract: difflib's M is anchoring-dependent, so
/// exact agreement is pinned on the forced-alignment classes and the
/// divergence rows are pinned, never silent). Bounded `[0.0, 1.0]`,
/// symmetric, `1.0` iff `a == b`; the degenerate rows are difflib's own
/// (`("", "")` → `1.0`, `("", "x")` → `0.0`). `deadline_ms` bounds the
/// whole call (`TimeoutError` on expiry; the O(ND) worst case is the same
/// DoS class `diff_opcodes`' deadline exists for, and an enormous-but-finite
/// budget saturates to unbounded).
///
/// GIL model: the `diff_opcodes` classes, meaning two str-in borrows, the Myers
/// search under `py.detach`, a single float out (no marshalling class).
#[pyfunction(signature = (a, b, *, deadline_ms = None))]
pub fn similarity_ratio(
    py: Python<'_>,
    a: &str,
    b: &str,
    deadline_ms: Option<f64>,
) -> PyResult<f64> {
    validate_deadline_ms(deadline_ms)?;
    py.detach(|| diff_impl::similarity_ratio_deadline(a, b, deadline_ms))
        .map_err(|err| PyErr::new::<PyTimeoutError, _>(err.message()))
}

/// `tors.get_close_matches(word, possibilities, n=3, cutoff=0.6, *,
/// deadline_ms=None)`: `difflib.get_close_matches`' shape at native speed:
/// every candidate scoring `similarity_ratio >= cutoff`, best first,
/// truncated to `n`. The sort is difflib's own `heapq.nlargest` tuple order:
/// score descending, then the candidate STRING descending (a stdlib
/// quirk probed and pinned in the gate: `get_close_matches("ab",
/// ["ac", "ca"], 2, 0.5)` → `["ca", "ac"]`); equal (score, string) pairs
/// keep input order. The returned strings are the ORIGINAL candidate
/// objects: references, zero marshalling, exactly what the stdlib hands
/// back. `n` must be `> 0` and `cutoff` in `[0.0, 1.0]` (difflib's own
/// `ValueError` messages, pinned). `deadline_ms` bounds the WHOLE call
/// across every candidate pair (`TimeoutError` on expiry, partial results
/// discarded).
///
/// GIL model: one GIL-held walk borrowing the candidate list (the standard
/// str-in class; empty candidate strings are legal here, unlike search
/// patterns), the whole scoring under `py.detach`, and the O(matches)
/// return marshalling is REFERENCE COUNTING ONLY (the original objects,
/// difflib's own shape).
#[pyfunction(signature = (word, possibilities, n = 3, cutoff = 0.6, *, deadline_ms = None))]
pub fn get_close_matches(
    py: Python<'_>,
    word: &str,
    possibilities: Bound<'_, PyList>,
    n: isize,
    cutoff: f64,
    deadline_ms: Option<f64>,
) -> PyResult<Py<PyAny>> {
    // difflib's own validation, verbatim message shape (difflib.py: "n must
    // be > 0: %r" % (n,)): `n` is taken SIGNED specifically so a negative
    // caller value reaches this check as a normal Python int rather than
    // failing pyo3's argument extraction into an unsigned type first, which
    // would raise OverflowError instead of stdlib's ValueError.
    if n <= 0 {
        return Err(PyValueError::new_err(format!("n must be > 0: {n}")));
    }
    let n = n as usize;
    if !(0.0..=1.0).contains(&cutoff) {
        return Err(PyValueError::new_err(format!(
            "cutoff must be in [0.0, 1.0]: {cutoff}"
        )));
    }
    validate_deadline_ms(deadline_ms)?;
    // The candidate walk mirrors find_patterns' list walk: handles kept
    // alive across the detach (the borrowed &strs point into the str
    // objects' immutable UTF-8 buffers; the list holds them, these handles
    // re-pin that for the compiler), but with NO empty-refusal (an empty
    // candidate is legal, difflib scores it 0.0 unless word is empty too).
    let items: Vec<_> = possibilities.iter().collect();
    let mut candidates: Vec<&str> = Vec::with_capacity(items.len());
    for item in &items {
        candidates.push(item.extract::<&str>()?);
    }
    let indices = py
        .detach(|| diff_impl::close_matches(word, &candidates, n, cutoff, deadline_ms))
        .map_err(|err| PyErr::new::<PyTimeoutError, _>(err.message()))?;
    // Indices → the ORIGINAL candidate objects (references, zero copy).
    let picked = indices
        .into_iter()
        .map(|idx| items[idx].clone())
        .collect::<Vec<_>>();
    Ok(PyList::new(py, picked)?.into_any().unbind())
}

/// `tors.levenshtein(a, b, *, deadline_ms=None)`: the unit-cost edit
/// distance (insert/delete/substitute = 1), CHARACTER-level, as a single
/// int: the metric CPython has no stdlib spelling of (difflib's ratio is
/// not a metric; the third-party spellings are third-party). The classic
/// vectors are pinned crate-side (kitten→sitting = 3) with `strsim 0.11`
/// as the dev-side differential oracle over a generated battery; symmetric;
/// equal operands short-circuit to 0 without the DP. O(n·m) worst case is
/// the reason `deadline_ms` exists: on two 1 MiB strings the DP is minutes
/// of CPU, and the budget is checked once per row (negligible overhead;
/// the uninterruptible-`strsim` route could not meet the crate's DoS bar,
/// which is why the formulas live in this crate). `TimeoutError` on
/// expiry; an enormous-but-finite budget saturates to unbounded.
///
/// GIL model: two str-in borrows, the DP under `py.detach`, a single int
/// out (no marshalling class).
#[pyfunction(signature = (a, b, *, deadline_ms = None))]
pub fn levenshtein(py: Python<'_>, a: &str, b: &str, deadline_ms: Option<f64>) -> PyResult<usize> {
    validate_deadline_ms(deadline_ms)?;
    py.detach(|| fuzzy_impl::levenshtein(a, b, deadline_ms))
        .map_err(|err| PyErr::new::<PyTimeoutError, _>(err.message()))
}

/// `tors.jaro(a, b, *, deadline_ms=None)`: the Jaro similarity
/// (window-matched chars + transpositions, `[0.0, 1.0]`, `("", "")` →
/// `1.0`), CHARACTER-level: pinned to `strsim 0.11` over a generated
/// battery plus the literature vectors (MARTHA/MARHTA ≈ 0.944). The
/// matching window makes the worst case O(n·m); `deadline_ms` bounds it
/// (checked per phase, `TimeoutError` on expiry).
///
/// GIL model: the `levenshtein` classes exactly.
#[pyfunction(signature = (a, b, *, deadline_ms = None))]
pub fn jaro(py: Python<'_>, a: &str, b: &str, deadline_ms: Option<f64>) -> PyResult<f64> {
    validate_deadline_ms(deadline_ms)?;
    py.detach(|| fuzzy_impl::jaro(a, b, deadline_ms))
        .map_err(|err| PyErr::new::<PyTimeoutError, _>(err.message()))
}

/// `tors.jaro_winkler(a, b, *, deadline_ms=None)`: Jaro-Winkler (Jaro plus
/// the common-prefix boost `l ≤ 4`, `p = 0.1`, applied when Jaro > `0.7`,
/// `strsim`'s exact threshold, cited in `src/fuzzy_impl.rs`), the
/// spell-correction similarity, pinned the same way (MARTHA/MARHTA ≈
/// `0.961`). Same window, same O(n·m) worst case, same `deadline_ms`.
///
/// GIL model: the `levenshtein` classes exactly.
#[pyfunction(signature = (a, b, *, deadline_ms = None))]
pub fn jaro_winkler(py: Python<'_>, a: &str, b: &str, deadline_ms: Option<f64>) -> PyResult<f64> {
    validate_deadline_ms(deadline_ms)?;
    py.detach(|| fuzzy_impl::jaro_winkler(a, b, deadline_ms))
        .map_err(|err| PyErr::new::<PyTimeoutError, _>(err.message()))
}
