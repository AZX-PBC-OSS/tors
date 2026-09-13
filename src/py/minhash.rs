use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyInt;

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
/// negatives: `seed=-1` is `seed=2**64-1`). A non-str `text` or non-int
/// `seed` raises `TypeError`; text bearing lone surrogates raises
/// `UnicodeEncodeError` (the crate-wide str-borrow contract).
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
/// (arbitrary magnitude, negative two's-complement) to its low 64 bits.
/// `& 0xFFFF_FFFF_FFFF_FFFF` in Python is exactly this, so the mask is
/// applied by calling the int's own `__and__` under the GIL (pyo3's
/// native extractions cap at i64/u64 and reject either negatives or the
/// full range); a non-int argument is a `TypeError` before the mask is
/// ever attempted.
fn normalize_seed(seed: &Bound<'_, PyAny>) -> PyResult<u64> {
    let as_int = seed
        .cast::<PyInt>()
        .map_err(|_| PyTypeError::new_err("seed must be an int"))?;
    let masked = as_int.call_method1("__and__", (u64::MAX,))?;
    masked.extract()
}
