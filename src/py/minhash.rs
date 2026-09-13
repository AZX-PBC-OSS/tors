use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt};

use crate::minhash_impl;

/// `tors.minhash_signature(text, *, num_perm=128, shingle_size=3, seed=0)
/// -> list[int]`: the MinHash signature of `text`'s word shingles, the
/// recall-side near-duplicate complement to the SimHash family: `num_perm`
/// min-hashes whose agreement fraction estimates the Jaccard similarity of
/// the two documents' shingle sets (the quantity an LSH-banding index --
/// caller state, tors stays stateless -- buckets candidates on). Each
/// element is `min over shingles of (a_i * x + b_i) mod (2^61 - 1)`: `x`
/// the shingle's XXH64 (the frozen-spec algorithm, deterministic across
/// processes/machines/versions), the `(a_i, b_i)` pairs derived from
/// `seed` by a pinned SplitMix64 stream (fixture-grade determinism, not
/// crypto). Tokens are the crate's one tokenizer (the `tf_idf`/
/// `bm25_rank` UAX #29 word stream, lowercased); shingles are
/// consecutive `shingle_size`-token windows. The estimator's standard
/// error is `sqrt(J(1-J)/num_perm)`, ~0.044 at the default 128.
///
/// Empty text, whitespace-only text, or fewer tokens than
/// `shingle_size` is the empty-shingle-set convention: every element the
/// u64 MAX sentinel (2^64 - 1, outside the affine range, so never a real
/// minimum) -- a stable, seed-invariant digest for empty documents.
///
/// Bounds: `num_perm` must be in `[1, 1024]` and `shingle_size` at least
/// 1, each raising `ValueError` (naming the bounds) before any work
/// runs; `seed` is any int, reduced mod 2^64 (two's complement for
/// negatives: `seed=-1` is `seed=2**64-1`), accepted through the
/// `__index__` protocol so int-likes (numpy integers included) work, with
/// `bool` rejected explicitly. A non-str `text` or non-int `seed` raises
/// `TypeError`; text bearing lone surrogates raises `UnicodeEncodeError`
/// (the crate-wide str-borrow contract).
///
/// GIL model: the text borrow and validation under the GIL, then the
/// whole tokenize + shingle + hash + min-sweep under one `py.detach`,
/// then the `num_perm`-element int-list marshalling (O(k), k <= 1024).
/// No `aio` twin: a fast one-shot call.
#[pyfunction(signature = (text, *, num_perm = 128, shingle_size = 3, seed = 0))]
pub fn minhash_signature(
    py: Python<'_>,
    text: &str,
    num_perm: i64,
    shingle_size: i64,
    #[pyo3(from_py_with = normalize_seed)] seed: u64,
) -> PyResult<Vec<u64>> {
    if !(1..=1024).contains(&num_perm) {
        return Err(PyValueError::new_err(format!(
            "num_perm must be between 1 and 1024, not {num_perm}"
        )));
    }
    if shingle_size < 1 {
        return Err(PyValueError::new_err(format!(
            "shingle_size must be at least 1, not {shingle_size}"
        )));
    }
    Ok(py.detach(|| minhash_impl::signature(text, num_perm as usize, shingle_size as usize, seed)))
}

/// `seed`'s house reduction, as the `from_py_with` extractor: any int
/// (arbitrary magnitude, negative two's-complement) to its low 64 bits —
/// `& 0xFFFF_FFFF_FFFF_FFFF` in Python is exactly this. The path is the
/// `__index__` protocol (the `PyNumber_Index` slot, never the instance's
/// own `__and__`: dispatching `__and__` runs arbitrary caller code and
/// rejects `__index__`-only int-likes such as numpy integers), then the
/// `PyLong_AsUnsignedLongLongMask` reduction pyo3's native extractions
/// cannot spell (they cap at i64/u64 and reject either negatives or the
/// full range). `bool` is rejected explicitly first: `True` is an `int`
/// with an `__index__`, but a boolean seed is a caller bug, not a seed.
/// Anything without `__index__` (str, float, None, bytes) is a `TypeError`
/// (the `AttributeError` mapped over), as is an `__index__` that does not
/// return an exact int.
fn normalize_seed(seed: &Bound<'_, PyAny>) -> PyResult<u64> {
    if seed.cast::<PyBool>().is_ok() {
        return Err(PyTypeError::new_err("seed must be an int, not bool"));
    }
    let index = seed
        .call_method0("__index__")
        .map_err(|_| PyTypeError::new_err("seed must be an int"))?;
    if index.cast::<PyInt>().is_err() {
        return Err(PyTypeError::new_err("seed must be an int"));
    }
    // `__index__` contractually returns an exact int, over which the Mask
    // reduction cannot fail (it truncates; only a non-int errors, excluded
    // above), so the sentinel return is unreachable.
    Ok(unsafe { pyo3::ffi::PyLong_AsUnsignedLongLongMask(index.as_ptr()) })
}
