use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use crate::minhash_impl;
use crate::near_dup_impl;
use crate::py::_borrow::{extract_index, validate_unit_interval};

/// The sweep-budget gate the pair functions share with
/// `minhash_signature`: a `width` past the wide-window bound over a
/// stream that fills the window re-hashes the whole live window per
/// token — the same unbounded middle range and the same 2^26 budget
/// (the shingle unit is the MinHash core's own framing hash, so the cost
/// shape is the sweep's). `ValueError` under the GIL, before any work;
/// widths at or below 1024 are width-bounded and pass through (the
/// documented caller-size lever), and the fixed width 3 the dedup sweep
/// rides never reaches this gate at all.
fn gate_width(text: &str, width: i64) -> PyResult<()> {
    if let Some(work) = minhash_impl::sweep_past_budget(text, width as usize) {
        return Err(PyValueError::new_err(format!(
            "width {width} over a stream that fills the window would sweep at least {work} \
             token-hashes, past the {} (2^26) token-hash budget; use a smaller width",
            minhash_impl::SHINGLE_SWEEP_BUDGET
        )));
    }
    Ok(())
}

/// `width`'s extraction: the shared `__index__` protocol (numpy integers
/// work, bool rejected with TypeError in every position), narrowed to the
/// i64 range the core casts from; `>= 1` stays a `ValueError` in the
/// bodies below.
fn extract_width(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    extract_index(obj, "width")?.extract::<i64>()
}

/// One simhash fingerprint argument's extraction for
/// `simhash_distance`: the shared `__index__` protocol first (the
/// minhash int parameters' convention — a non-int is a `TypeError`, a
/// bool is a `TypeError` in every position), then the sign and range
/// checks the fingerprint domain adds: a negative int is a `ValueError`
/// (simhash fingerprints are unsigned; a negative is a caller bug, not a
/// value) and an int past 2^128 overflows at extraction (pyo3's own
/// `OverflowError`, the truncate_to_bounds-identical pattern for
/// out-of-range ints).
fn extract_fingerprint(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<u128> {
    let index = extract_index(obj, name)?;
    let is_negative = index.lt(0)?;
    if is_negative {
        return Err(PyValueError::new_err(format!(
            "{name} must be a non-negative int: simhash fingerprints are unsigned"
        )));
    }
    index.extract::<u128>()
}

/// `tors.simhash_distance(a, b) -> int`: the Hamming distance between two
/// simhash fingerprints — the values `tors.simhash64`/`tors.simhash128`
/// return — the count of bit positions at which they differ. Zero means
/// an identical fingerprint (the near-dup gate's "same token multiset"
/// equality: same words, any order, any whitespace); small distances are
/// the near-duplicate band whose calibration is corpus-dependent (the
/// measured anchors and their caveats live in `simhash64`'s docs).
///
/// Widths: both spellings are supported, but a pair must come from ONE
/// spelling — mixing a 64-bit with a 128-bit fingerprint raises
/// `ValueError`. The honest mechanics, since a Python int carries no
/// width metadata: the check classifies each argument by magnitude (only
/// a 128-bit fingerprint can be >= 2**64) and refuses a pair split
/// across the line, which catches the real caller bug (passing one
/// `simhash64` value and one `simhash128` value) with probability
/// 1 - 2**-64; a pair both of whose values fit in 64 bits compares
/// correctly either way, because the distance arithmetic itself is
/// width-blind `(a ^ b).bit_count()`.
///
/// Bounds: a non-int argument (including `bool`) raises `TypeError`, a
/// negative int raises `ValueError` (fingerprints are unsigned), and an
/// int past 2**128 raises `OverflowError` (pyo3's own, the
/// out-of-range-int pattern every size argument here shares); both ride
/// the `__index__` protocol, so int-likes (numpy integers) work.
///
/// GIL model: `uuid_parse`'s zero-detach member — the whole work is one
/// xor-and-popcount over two borrowed ints, nanoseconds, strictly less
/// than the argument extraction that precedes it; a `py.detach` around
/// it would cost more than it guards. No `aio` twin (a
/// microsecond-scale call over two small ints, `docs/async.md`'s
/// stay-sync lane).
#[pyfunction]
pub fn simhash_distance(a: Bound<'_, PyAny>, b: Bound<'_, PyAny>) -> PyResult<u32> {
    let a = extract_fingerprint(&a, "a")?;
    let b = extract_fingerprint(&b, "b")?;
    // The width check: exactly one side >= 2^64 means one argument came
    // from simhash128 and the other from simhash64 — refuse the mix.
    if (a >= 1u128 << 64) != (b >= 1u128 << 64) {
        return Err(PyValueError::new_err(
            "a and b must be simhash values of the same width: one argument is a 128-bit \
             simhash (>= 2**64) and the other fits in 64 bits — compare values from the \
             same spelling (both simhash64 or both simhash128)",
        ));
    }
    Ok(near_dup_impl::fingerprint_hamming(a, b))
}

/// `tors.shingle_jaccard(a, b, *, width=3) -> float`: the Jaccard index
/// of the two texts' `width`-token WORD shingle sets,
/// `|A ∩ B| / |A ∪ B|` — Broder 1997's resemblance, the exact quantity
/// `minhash_signature` estimates. Word shingles (not character shingles)
/// for the reason `minhash_signature` documents: near-duplicates
/// preserve word sequence where a reflow shifts character k-grams
/// wholesale. Tokens are the crate's one real-word tokenizer (UAX #29
/// word segments, whitespace-only segments skipped), each normalized
/// with the grounding layer's matching form — lowercased, then
/// canonicalized to NFC — so `"Hello, World!"` and `"hello world"`
/// score 1.0 and NFC-equivalent inputs (NFD "cafe\\u{301}" vs NFC
/// "café") behave identically.
///
/// Identical shingle sets score 1.0, disjoint sets 0.0. The empty-set
/// convention: text with no word tokens (empty, whitespace-only, or
/// fewer tokens than `width`) has an EMPTY shingle set; two empty sets
/// score 1.0 (∅ ⊆ ∅ — two token-free texts are duplicates of each
/// other, which is what `dedup_near_dup(method="shingle")` must conclude
/// too), exactly one empty scores 0.0.
///
/// Bounds: `width` must be >= 1 (`ValueError`; the `__index__` protocol
/// accepts int-likes, `bool` rejected), and a `width` past 1024 over a
/// stream that fills the window raises the same sweep-budget
/// `ValueError` `minhash_signature` raises (the cost shape is identical:
/// each step re-hashes the live window). A non-str argument raises
/// `TypeError`; a lone surrogate raises `UnicodeEncodeError` (the
/// crate-wide str-borrow contract).
///
/// GIL model: the two str borrows and the `width` validation under the
/// GIL, the whole tokenize+shingle+set pass under one `py.detach`, a
/// single float out (no marshalling class).
#[pyfunction(signature = (a, b, *, width = 3))]
pub fn shingle_jaccard(
    py: Python<'_>,
    a: &str,
    b: &str,
    #[pyo3(from_py_with = extract_width)] width: i64,
) -> PyResult<f64> {
    if width < 1 {
        return Err(PyValueError::new_err(format!(
            "width must be at least 1, not {width}"
        )));
    }
    gate_width(a, width)?;
    gate_width(b, width)?;
    Ok(py.detach(|| near_dup_impl::shingle_jaccard(a, b, width as usize)))
}

/// `tors.shingle_dice(a, b, *, width=3) -> float`: the Dice coefficient
/// of the two texts' `width`-token word-shingle sets,
/// `2|A ∩ B| / (|A| + |B|)` — the same agreement Jaccard measures,
/// weighted toward the small-set side (a shared sliver of a huge set
/// moves Dice more than Jaccard). Everything else — tokenization,
/// normalization, the empty-set convention, the `width` and str-borrow
/// bounds, the GIL model — is `shingle_jaccard`'s exactly.
#[pyfunction(signature = (a, b, *, width = 3))]
pub fn shingle_dice(
    py: Python<'_>,
    a: &str,
    b: &str,
    #[pyo3(from_py_with = extract_width)] width: i64,
) -> PyResult<f64> {
    if width < 1 {
        return Err(PyValueError::new_err(format!(
            "width must be at least 1, not {width}"
        )));
    }
    gate_width(a, width)?;
    gate_width(b, width)?;
    Ok(py.detach(|| near_dup_impl::shingle_dice(a, b, width as usize)))
}

/// `tors.dedup_near_dup(texts, *, threshold=0.9, method="simhash")`:
/// greedy keep-first near-duplicate dedup over a list of strings.
/// Returns a dict with three keys, all present every time, all indices
/// into the INPUT order:
///
/// - `kept`: the representatives, ascending — text i is kept iff no
///   EARLIER kept text is within `threshold` of it, so input order is
///   the tie-break by construction and the result is deterministic.
/// - `dropped`: the absorbed texts, ascending.
/// - `groups`: a full partition of the indices, in representative
///   order — each group is one kept representative followed by the
///   texts it absorbed (singleton groups are kept texts with no
///   duplicates). `kept == [g[0] for g in groups]` and
///   `sorted(kept + dropped) == list(range(len(texts)))` always hold.
///
/// Methods (the closed set; an unknown name raises `ValueError` naming
/// every choice): `"simhash"` — a pair is duplicates when the Hamming
/// distance between the two 64-bit fingerprints of the
/// normalization-folded texts is at most `floor((1 - threshold) * 64)`
/// (threshold 0.9 -> 6 bits); `"shingle"` — when the EXACT Jaccard
/// index of the 3-token word-shingle sets is at least `threshold`;
/// `"minhash"` — when the agreement fraction of the two
/// 128-permutation `minhash_signature` signatures (its defaults:
/// shingle_size 3, seed 0) is at least `threshold`, the estimated
/// Jaccard for corpora where the exact sets are too wide to intersect
/// pairwise. All three share the grounding normalization policy
/// (lowercase + NFC; see `src/near_dup_impl.rs`), so a method switch
/// cannot silently change what "same text" means.
///
/// SCOPE — small candidate sets, honestly: the sweep is O(n²) pair
/// checks by design (with the per-method early exits: the popcount is
/// O(1); the shingle check skips a pair whose cardinality ratio already
/// fails the threshold; the MinHash count aborts the moment agreement
/// is unreachable), holding O(total input) of fingerprint state and NO
/// persistent index — the LSH banding table a corpus-scale pipeline
/// builds on these signatures is caller state, the doctrine's scope cut
/// (docs/design.md). The quadratic wall is pinned with an explicit
/// budget in tests/test_scaling_pins.py and benchmarked in
/// benches/near_dup.rs; beyond tens of thousands of candidates, band
/// `minhash_signature` output yourself.
///
/// The empty list gives the empty result (`{"kept": [], "dropped": [],
/// "groups": []}`); all-identical input keeps exactly the first text;
/// threshold must be in [0.0, 1.0] (NaN included in the refusal) and a
/// non-str element raises `TypeError` (the list walk's standard
/// extraction class).
///
/// GIL model: the list extraction (one str copy per element, the
/// standard O(total input) GIL-held class) and the
/// threshold/method validation under the GIL, then the whole
/// fingerprint pass AND the pairwise sweep under one `py.detach`, then
/// the O(n + groups) index-list marshalling after. The `aio` twin
/// (`tors.aio.dedup_near_dup`) is the same call under
/// `asyncio.to_thread`.
#[pyfunction(signature = (texts, *, threshold = 0.9, method = "simhash"))]
pub fn dedup_near_dup(
    py: Python<'_>,
    texts: Vec<String>,
    threshold: f64,
    method: &str,
) -> PyResult<Py<PyAny>> {
    validate_unit_interval("threshold", threshold, true)?;
    let method = near_dup_impl::parse_dedup_method(method).map_err(PyValueError::new_err)?;
    // The fixed sweep widths are all width-bounded (simhash 64 bits,
    // shingle/minhash width 3): no sweep-budget gate applies per text.
    let outcome = py.detach(|| near_dup_impl::dedup_near_dup(&texts, threshold, method));
    let kept = PyList::new(py, &outcome.kept)?;
    let dropped = PyList::new(py, &outcome.dropped)?;
    let groups = PyList::new(
        py,
        outcome
            .groups
            .iter()
            .map(|group| PyList::new(py, group).map(Bound::into_any))
            .collect::<PyResult<Vec<Bound<'_, PyAny>>>>()?,
    )?;
    let result = PyDict::new(py);
    result.set_item("kept", kept)?;
    result.set_item("dropped", dropped)?;
    result.set_item("groups", groups)?;
    Ok(result.into_any().unbind())
}
