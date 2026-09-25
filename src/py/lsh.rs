use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyList, PySequence, PyString};

use crate::lsh_impl;
use crate::py::_borrow::{extract_index, validate_unit_interval};

/// The `bands`/`rows` extraction shared by all three LSH functions: the
/// `__index__` protocol first (int-likes work, `bool` rejected), narrowed
/// to the i64 range the bodies cast from (`>= 1` and the product bound
/// stay `ValueError` in the bodies).
fn extract_band_param(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<i64> {
    extract_index(obj, name)?.extract::<i64>()
}

fn extract_bands(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    extract_band_param(obj, "bands")
}

fn extract_rows(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    extract_band_param(obj, "rows")
}

/// One signature's extraction: a sequence of exact non-negative ints,
/// each within 2^64 - 1 (the values `minhash_signature` returns). Batch
/// elements are strict, the bytes-element surfaces' convention (`list[
/// bytes]` refuses a bytearray): elements are exact ints, `bool`
/// rejected, no per-element `__index__` dispatch (one dunder hop per
/// element is an O(n * num_perm) interpreter cost no corpus-scale
/// caller should buy; int-likes are the scalar-parameter courtesy,
/// which `bands`/`rows` keep). A negative element is a `ValueError`
/// (MinHash values are unsigned; a negative is a caller bug, the
/// simhash-fingerprint convention) and an element past 2^64 - 1 raises
/// `OverflowError` (the out-of-range-int pattern). A bare `str` is
/// refused before the sequence cast (its characters would silently
/// extract one by one, the `Vec` footgun); any other non-sequence is a
/// `TypeError`.
fn extract_signature(obj: &Bound<'_, PyAny>, position: usize) -> PyResult<Vec<u64>> {
    if obj.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(format!(
            "signatures[{position}] must be a sequence of ints, not str"
        )));
    }
    let seq = obj.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err(format!(
            "signatures[{position}] must be a sequence of ints, not {}",
            obj.get_type()
                .qualname()
                .map(|q| q.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "object".into())
        ))
    })?;
    let mut rows: Vec<u64> = Vec::new();
    for item in seq.try_iter()? {
        let item = item?;
        if item.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(format!(
                "signatures[{position}] elements must be ints, not bool"
            )));
        }
        match item.extract::<u64>() {
            Ok(value) => rows.push(value),
            Err(_) => return Err(signature_element_error(&item, position)),
        }
    }
    Ok(rows)
}

/// The precise error for an element the fast u64 extraction refused:
/// re-derive WHICH contract failed (negative, past 2^64 - 1, or not an
/// int at all) instead of surfacing the generic conversion message.
fn signature_element_error(item: &Bound<'_, PyAny>, position: usize) -> PyErr {
    let type_name = item
        .get_type()
        .qualname()
        .map(|q| q.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "object".into());
    match item.extract::<i128>() {
        Ok(value) if value < 0 => PyValueError::new_err(format!(
            "signatures[{position}] elements must be non-negative ints: MinHash \
             values are unsigned"
        )),
        Ok(value) => PyOverflowError::new_err(format!(
            "signatures[{position}] element {value} is too large for a u64 MinHash \
             value (max 2**64 - 1)"
        )),
        Err(_) => PyTypeError::new_err(format!(
            "signatures[{position}] elements must be ints, not {type_name}"
        )),
    }
}

/// The `signatures` argument's extraction: one walk, each element one
/// signature's `num_perm` rows (see [`extract_signature`]). The walk
/// never reads `__len__` to pre-reserve (the lying-`__len__`
/// capacity-overrun class the bounded walks exist for): the rows vectors
/// grow by pushing.
fn extract_signatures(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<u64>>> {
    if obj.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(
            "signatures must be a sequence of int sequences, not str",
        ));
    }
    let seq = obj.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err(format!(
            "signatures must be a sequence of int sequences, not {}",
            obj.get_type()
                .qualname()
                .map(|q| q.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "object".into())
        ))
    })?;
    let mut signatures: Vec<Vec<u64>> = Vec::new();
    for (position, item) in seq.try_iter()?.enumerate() {
        signatures.push(extract_signature(&item?, position)?);
    }
    Ok(signatures)
}

/// `bands`/`rows`' shared shape validation: each at least 1, and the
/// product within i64 (it becomes the every-signature length contract).
fn validate_bands_rows(bands: i64, rows: i64) -> PyResult<usize> {
    if bands < 1 {
        return Err(PyValueError::new_err(format!(
            "bands must be at least 1, not {bands}"
        )));
    }
    if rows < 1 {
        return Err(PyValueError::new_err(format!(
            "rows must be at least 1, not {rows}"
        )));
    }
    match bands.checked_mul(rows) {
        Some(num_perm) if num_perm <= i64::from(u32::MAX) => Ok(num_perm as usize),
        _ => Err(PyValueError::new_err(format!(
            "bands * rows = {bands} * {rows} overflows the signature length contract; \
             use a shape whose product fits a signature"
        ))),
    }
}

/// `tors.lsh_candidates(signatures, *, bands, rows) -> CandidatePairs`:
/// MinHash LSH banding, the near-duplicate candidate generator. Cuts
/// every signature into `bands` bands of `rows` rows (every signature's
/// length MUST equal `bands * rows`: exactly what `minhash_signature`
/// returns at `num_perm=bands*rows`), hashes each band's rows to a
/// bucket key, and returns every pair of signatures sharing at least one
/// bucket: `{"pairs": [(i, j), ...]}` with `i < j`, deduplicated across
/// bands (a pair sharing several band buckets appears once) and in
/// ascending `(i, j)` order. The candidate relation is a function of the
/// signatures' content alone: permuting the input permutes the same
/// relation (pinned in tests/test_lsh.py).
///
/// This is the stateless pass docs/design.md's LSH scope cut points at:
/// one call over the signatures in hand, no table kept across calls, no
/// insert/query surface. Broder, Glassman, Manasse, and Zweig, "Syntactic
/// Clustering of the Web" (WWW 1997) supply the shingling/resemblance
/// frame; Leskovec, Rajaraman, and Ullman, "Mining of Massive Datasets",
/// ch. 3 the banding probability model; datasketch's persistent
/// `MinHashLSH` the incremental-table API shape tors deliberately does
/// not ship.
///
/// The false-positive contract, stated plainly: dissimilar signatures
/// CAN become candidates. `r` rows agreeing by chance is the S-curve
/// itself (`lsh_probability` gives its probability), and a 64-bit
/// band-key collision over two different row frames is possible at
/// ~k^2/2^65 over k distinct keys. Both channels make the output a
/// recall-biased filter to SCORE downstream (`shingle_jaccard`,
/// `dedup_near_dup`), never a verdict: the same recall-biased contract
/// LSH banding has everywhere.
///
/// Bounds: `bands` and `rows` must each be at least 1 and every
/// signature's length must equal `bands * rows`, each a `ValueError`
/// before any work runs; the two shape parameters ride `__index__`
/// (int-likes work, `bool` rejected); signature elements are STRICT, the
/// bytes-element surfaces' convention: exact non-negative ints within
/// 2**64 - 1 (negative: `ValueError`, past the range: `OverflowError`,
/// non-int including bool: `TypeError`; no per-element `__index__`
/// dispatch, so numpy int-likes convert first). An empty `signatures`
/// list gives `{"pairs": []}`.
///
/// Cost: one pass, `O(n * num_perm)` band hashing plus pair emission
/// ONLY inside shared buckets, which is O(output): nothing is quadratic
/// in the bucket sizes beyond the pairs they contribute. Memory is one
/// band's bucket table plus the dedup set, `O(n + pairs)`.
///
/// GIL model: the `signatures` walk (one int extraction per element, the
/// standard O(n * num_perm) arg-walk class) and the bounds validation
/// under the GIL, then the whole banding pass under one `py.detach`, then
/// the O(pairs) tuple-list marshalling. The `aio` twin
/// (`tors.aio.lsh_candidates`) is the same call under
/// `asyncio.to_thread`.
#[pyfunction(signature = (signatures, *, bands, rows))]
pub fn lsh_candidates(
    py: Python<'_>,
    signatures: Bound<'_, PyAny>,
    #[pyo3(from_py_with = extract_bands)] bands: i64,
    #[pyo3(from_py_with = extract_rows)] rows: i64,
) -> PyResult<Py<PyAny>> {
    let num_perm = validate_bands_rows(bands, rows)?;
    let signatures = extract_signatures(&signatures)?;
    for (position, sig) in signatures.iter().enumerate() {
        if sig.len() != num_perm {
            return Err(PyValueError::new_err(format!(
                "signatures[{position}] has {} rows, expected bands * rows = {num_perm}: \
                 every signature must have exactly bands * rows elements",
                sig.len()
            )));
        }
    }
    let outcome =
        py.detach(|| lsh_impl::lsh_candidates(&signatures, bands as usize, rows as usize));
    let pairs = PyList::new(
        py,
        outcome
            .pairs
            .iter()
            .map(|&(i, j)| PyTuplePair(i, j))
            .collect::<Vec<_>>(),
    )?;
    let result = PyDict::new(py);
    result.set_item("pairs", pairs)?;
    Ok(result.into_any().unbind())
}

/// The `(i, j)` pair as a 2-tuple: `PyTuplePair` marshals through
/// `IntoPyObject` as `(i, j)`.
struct PyTuplePair(usize, usize);

impl<'py> IntoPyObject<'py> for PyTuplePair {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok((self.0, self.1).into_pyobject(py)?.into_any())
    }
}

/// `tors.lsh_probability(s, *, bands, rows) -> float`: the banding
/// S-curve as a pure formula, `1 - (1 - s^rows)^bands` -- the probability
/// that two signatures with Jaccard similarity `s` share at least one of
/// `bands` band buckets of `rows` rows (Leskovec, Rajaraman, and Ullman,
/// "Mining of Massive Datasets", ch. 3; datasketch's parameter-tuning
/// docs spell the same curve). Use it to PICK `bands`/`rows` for a
/// target threshold: choose the similarity `s` a pair must reach to
/// become a candidate, then check the curve catches it, e.g.
/// `lsh_probability(0.8, bands=16, rows=8)` is about 0.947 while
/// `lsh_probability(0.25, bands=16, rows=8)` is about 0.00024.
/// Around this shape the candidate threshold sits near
/// `lsh_threshold(bands=16, rows=8)` (about 0.36): a pair past it
/// becomes very likely a candidate, a pair below it very unlikely.
///
/// `s` must be in [0.0, 1.0] (NaN refused), else `ValueError`; `bands`
/// and `rows` at least 1, else `ValueError`. The ends are exact:
/// `s = 0.0` gives `0.0` and `s = 1.0` gives `1.0` (identical signatures
/// are always candidates). Pure float arithmetic over two integers: no
/// detach, no `aio` twin (the thread hop would cost more than the call,
/// the `simhash_distance` stay-sync class).
#[pyfunction(signature = (s, *, bands, rows))]
pub fn lsh_probability(
    s: f64,
    #[pyo3(from_py_with = extract_bands)] bands: i64,
    #[pyo3(from_py_with = extract_rows)] rows: i64,
) -> PyResult<f64> {
    validate_unit_interval("s", s, false)?;
    let (bands, rows) = validate_shape(bands, rows)?;
    Ok(lsh_impl::lsh_probability(s, bands, rows))
}

/// `tors.lsh_threshold(*, bands, rows) -> float`: the approximate
/// similarity threshold where the S-curve takes its step,
/// `(1/bands)^(1/rows)` (Leskovec, Rajaraman, and Ullman, "Mining of
/// Massive Datasets", ch. 3, the same formulation datasketch's docs
/// carry). An approximation, not an inversion of `lsh_probability`:
/// pairs at this similarity are candidates with probability NEAR the
/// curve's midpoint, not exactly 0.5. Pick a shape here, then verify the
/// actual curve with `lsh_probability`. Pure float arithmetic over two
/// integers: no detach, no `aio` twin (the `simhash_distance`
/// stay-sync class).
#[pyfunction(signature = (*, bands, rows))]
pub fn lsh_threshold(
    #[pyo3(from_py_with = extract_bands)] bands: i64,
    #[pyo3(from_py_with = extract_rows)] rows: i64,
) -> PyResult<f64> {
    let (bands, rows) = validate_shape(bands, rows)?;
    Ok(lsh_impl::lsh_threshold(bands, rows))
}

/// The formula functions' shape validation: bands and rows at least 1
/// (the product bound `lsh_candidates` needs does not apply -- these are
/// pure arithmetic over the two shape integers).
fn validate_shape(bands: i64, rows: i64) -> PyResult<(usize, usize)> {
    if bands < 1 {
        return Err(PyValueError::new_err(format!(
            "bands must be at least 1, not {bands}"
        )));
    }
    if rows < 1 {
        return Err(PyValueError::new_err(format!(
            "rows must be at least 1, not {rows}"
        )));
    }
    Ok((bands as usize, rows as usize))
}
