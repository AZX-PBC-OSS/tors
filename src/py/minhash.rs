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
/// Widths past 1024 tokens short-circuit through a retention-free token
/// count when the stream ends short of a full window, so a huge
/// `shingle_size` over a large text answers the sentinel without
/// materializing the stream.
///
/// Bounds: `num_perm` must be in `[1, 1024]` and `shingle_size` at least
/// 1, each raising `ValueError` (naming the bounds) before any work
/// runs; a stream that fills a window wider than 1024 tokens also raises
/// `ValueError` when the sweep it would cost -- `(tokens - shingle_size
/// + 1) × shingle_size` token-hashes, the whole live window re-hashed
/// per step -- exceeds the 2^26 budget (that middle range is otherwise
/// unbounded; the count deciding it is retention-free, capped at
/// `floor(2^26 / shingle_size) + shingle_size` tokens regardless of
/// stream length, under the GIL, and short streams still answer the
/// sentinel as before). `seed` is any int, reduced mod 2^64 (two's
/// complement for
/// negatives: `seed=-1` is `seed=2**64-1`). All three are accepted through
/// the `__index__` protocol so int-likes (numpy integers included) work,
/// with `bool` rejected explicitly in every position (including as an
/// `__index__` result); anything without `__index__` (str, float, None,
/// bytes) is a `TypeError`, and an `__index__` that does not return an
/// exact int is a `TypeError` too. A non-str `text` raises `TypeError`;
/// text bearing lone surrogates raises `UnicodeEncodeError` (the
/// crate-wide str-borrow contract).
///
/// GIL model: the text borrow, the bounds, and the budget gate under the
/// GIL, then the whole tokenize + shingle + hash + min-sweep under one
/// `py.detach`, then the `num_perm`-element int-list marshalling (O(k),
/// k <= 1024). The `aio` twin (`tors.aio.minhash_signature`) is the same
/// call under `asyncio.to_thread`.
#[pyfunction(signature = (text, *, num_perm = 128, shingle_size = 3, seed = 0))]
pub fn minhash_signature(
    py: Python<'_>,
    text: &str,
    #[pyo3(from_py_with = extract_num_perm)] num_perm: i64,
    #[pyo3(from_py_with = extract_shingle_size)] shingle_size: i64,
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
    // The wide-window sweep budget: past WIDE_WINDOW_COUNT_FIRST a stream
    // that fills the window re-hashes the whole live window per token, so
    // the pass is (tokens - shingle_size + 1) * shingle_size token-hashes
    // -- unbounded in exactly the middle range the sentinel short-circuit
    // cannot decide. Shapes past the budget raise here, under the GIL,
    // before the detached pass; the short-stream case (the short-circuit's
    // own, answered by the sentinel) and the width-bounded range at or
    // below 1024 pass through untouched. The gate's own count walk is
    // capped at floor(budget/shingle_size) + shingle_size tokens, so even
    // a many-gigabyte stream is rejected without a GIL-held walk of the
    // whole input, and the reported work is the minimum that shape would
    // spend ("at least" -- the exact count is never walked past the cap).
    if let Some(work) = minhash_impl::sweep_past_budget(text, shingle_size as usize) {
        return Err(PyValueError::new_err(format!(
            "shingle_size {shingle_size} over a stream that fills the window would sweep \
             at least {work} token-hashes, past the {} (2^26) token-hash budget; use a \
             smaller shingle_size",
            minhash_impl::SHINGLE_SWEEP_BUDGET
        )));
    }
    Ok(py.detach(|| minhash_impl::signature(text, num_perm as usize, shingle_size as usize, seed)))
}

/// The shared `__index__`-protocol extraction behind all three int
/// parameters: `bool` is rejected up front (it would otherwise launder to
/// 0/1 through the index), then the `__index__` SLOT is dispatched --
/// `getattr` plus call, never the instance's own `__and__`, so masking
/// cannot alter the value and `__index__`-only int-likes (numpy integers)
/// reduce identically. `__index__` itself IS caller code: it runs with its
/// own side effects, exactly once, and its own failure propagates
/// unchanged rather than masking as a parameter error. Only a MISSING
/// `__index__` (str, float, None, bytes) and an `__index__` result that is
/// not an exact int (including `bool`, the same caller bug one dispatch
/// removed) are `TypeError`.
fn extract_index<'py>(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyInt>> {
    if obj.cast::<PyBool>().is_ok() {
        return Err(PyTypeError::new_err(format!(
            "{name} must be an int, not bool"
        )));
    }
    let index = obj
        .getattr("__index__")
        .map_err(|_| PyTypeError::new_err(format!("{name} must be an int")))?;
    let index = index.call0()?;
    if index.cast::<PyBool>().is_ok() {
        return Err(PyTypeError::new_err(format!(
            "{name} must be an int, not bool"
        )));
    }
    if index.cast::<PyInt>().is_err() {
        return Err(PyTypeError::new_err(format!("{name} must be an int")));
    }
    Ok(index.cast_into::<PyInt>().expect("checked exact int above"))
}

/// `num_perm`'s extraction: the `__index__` protocol, narrowed to the i64
/// range the core casts from. An int outside i64 raises pyo3's own
/// `OverflowError` at extraction (the `truncate_to_bounds`-identical
/// pattern for every i64-typed size argument); range membership itself
/// (`[1, 1024]`) stays a `ValueError` in the body.
fn extract_num_perm(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    extract_index(obj, "num_perm")?.extract::<i64>()
}

/// `shingle_size`'s extraction: same protocol and range narrowing as
/// `num_perm` (`>= 1` stays a `ValueError` in the body).
fn extract_shingle_size(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    extract_index(obj, "shingle_size")?.extract::<i64>()
}

/// `seed`'s house reduction, as the `from_py_with` extractor: any int
/// (arbitrary magnitude, negative two's-complement) to its low 64 bits —
/// `& 0xFFFF_FFFF_FFFF_FFFF` in Python is exactly this. The path is the
/// shared `extract_index` protocol above (the `__index__` slot, never the
/// instance's own `__and__`), then the `PyLong_AsUnsignedLongLongMask`
/// reduction pyo3's native extractions cannot spell (they cap at i64/u64
/// and reject either negatives or the full range). `bool` is rejected
/// explicitly first -- and as the `__index__` result: `True` is an `int`
/// with an `__index__`, but a boolean seed is a caller bug, not a seed.
/// Anything without `__index__` (str, float, None, bytes) is a `TypeError`,
/// as is an `__index__` that does not return an exact int; the caller's
/// own `__index__` failure propagates unchanged.
fn normalize_seed(seed: &Bound<'_, PyAny>) -> PyResult<u64> {
    let index = extract_index(seed, "seed")?;
    // `extract_index` guarantees an exact int, over which the Mask
    // reduction cannot fail (it truncates; only a non-int errors, excluded
    // above), so the sentinel return is unreachable.
    Ok(unsafe { pyo3::ffi::PyLong_AsUnsignedLongLongMask(index.as_ptr()) })
}
