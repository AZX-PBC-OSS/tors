use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyString};

use crate::minhash_impl;
use crate::py::_borrow::extract_index;

/// `tors.minhash_signature(text, *, num_perm=128, shingle_size=3, seed=0,
/// method="xxh", bits=None) -> list[int]`: the MinHash signature of
/// `text`'s word shingles, the recall-side near-duplicate complement to
/// the SimHash family: `num_perm` min-hashes whose agreement fraction
/// estimates the Jaccard similarity of the two documents' shingle sets
/// (the quantity an LSH-banding index -- caller state, tors stays
/// stateless -- buckets candidates on). Each
/// element is `min over shingles of (a_i * x + b_i) mod (2^61 - 1)`: `x`
/// the shingle's XXH64 (the frozen-spec algorithm, deterministic across
/// processes/machines/versions), the `(a_i, b_i)` pairs derived from
/// `seed` by a pinned SplitMix64 stream (fixture-grade determinism, not
/// crypto). Tokens are the crate's one tokenizer (the `tf_idf`/
/// `bm25_rank` UAX #29 word stream, lowercased); shingles are
/// consecutive `shingle_size`-token windows. The estimator's standard
/// error is `sqrt(J(1-J)/num_perm)`, ~0.044 at the default 128.
///
/// `method` picks the engine: `"xxh"` (the default) is the classic
/// k-independent-permutations sweep described above, byte-identical to
/// every pre-method signature ever produced; `"superminhash"` is Ertl's
/// SuperMinHash (arXiv 1706.05698), the in-place incremental sampling
/// scheme whose agreement estimator is unbiased like the classic one but
/// carries up to ~half the variance for small sets (the paper's
/// alpha(m, u) factor) and whose sweep is amortized cheaper on large
/// shingle sets. The two engines' signatures are NOT cross-compatible:
/// rows from different methods (or different `num_perm` values -- the
/// SuperMinHash permutation structure depends on m, so it has no prefix
/// property) must never be compared, banded, or mixed; that is a caller
/// error no cheap runtime check can catch (rows are opaque ints), so it
/// is a documentation contract enforced by convention. Each engine is
/// deterministic and self-consistent: identical text at identical
/// parameters gives identical rows, forever, within one tors version.
///
/// `bits` is b-bit compression (Li and König, WDE 2010): `None` (the
/// default) returns the full u64 rows, byte-identical to the unmasked
/// output; an int in [1, 63] keeps the LOWEST `bits` bits of every row --
/// `num_perm` rows shrink to a b-bit fingerprint, and the paired-row
/// agreement estimator needs the 2^-b chance correction
/// (`minhash_jaccard` applies it). `bits` is defined only over the
/// classic engine's rows (uniform over [0, 2^61), whose low bits are
/// exactly uniform); `method="superminhash"` with `bits` set is a
/// `ValueError` -- the SuperMinHash rows are f64 bit patterns, not
/// uniform integers, and the b-bit estimator math does not transfer to
/// them.
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
/// k <= 1024; the b-bit mask rides the marshalled rows, O(k) integer ANDs).
/// The `aio` twin (`tors.aio.minhash_signature`) is the same call under
/// `asyncio.to_thread`.
#[pyfunction(signature = (text, *, num_perm = 128, shingle_size = 3, seed = 0, method = "xxh", bits = None))]
pub fn minhash_signature(
    py: Python<'_>,
    text: &str,
    #[pyo3(from_py_with = extract_num_perm)] num_perm: i64,
    #[pyo3(from_py_with = extract_shingle_size)] shingle_size: i64,
    #[pyo3(from_py_with = normalize_seed)] seed: u64,
    #[pyo3(from_py_with = extract_method)] method: &str,
    #[pyo3(from_py_with = extract_bits)] bits: Option<u32>,
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
    if method != "xxh" && method != "superminhash" {
        return Err(PyValueError::new_err(format!(
            "method must be \"xxh\" or \"superminhash\", not {method:?}"
        )));
    }
    validate_bits(bits)?;
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
    if method == "superminhash" && bits.is_some() {
        return Err(PyValueError::new_err(
            "bits= is defined only over the classic engine's rows (uniform integers, \
             whose low bits the b-bit estimator's math needs); the superminhash rows are \
             f64 bit patterns, so method=\"superminhash\" requires bits=None",
        ));
    }
    let mut rows = py.detach(|| match method {
        "superminhash" => minhash_impl::superminhash_signature(
            text,
            num_perm as usize,
            shingle_size as usize,
            seed,
        ),
        // The default: the classic k-permutation sweep, byte-identical
        // for every caller that does not name a method.
        _ => minhash_impl::signature(text, num_perm as usize, shingle_size as usize, seed),
    });
    if let Some(bits) = bits {
        let mask = (1u64 << bits) - 1;
        for row in &mut rows {
            *row &= mask;
        }
    }
    Ok(rows)
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

/// `seed`'s house reduction for the weighted engine: `None` is the fixed
/// default seed 0 (there is no random entropy anywhere in the engine, so
/// `seed=None` IS `seed=0`, a documented equality not a random draw);
/// everything else rides [`normalize_seed`]. The Option stays an Option
/// through the extraction because the `#[pyfunction]` default
/// (`seed = None`) must type-check against the parameter.
fn normalize_seed_optional(seed: &Bound<'_, PyAny>) -> PyResult<Option<u64>> {
    if seed.is_none() {
        return Ok(None);
    }
    Ok(Some(normalize_seed(seed)?))
}

/// `method`'s extraction: a strict `str` (`bool` and every other type is
/// a `TypeError`); the VALUE membership ("xxh" | "superminhash") stays a
/// `ValueError` in the body, the same split as every range-vs-type bound.
fn extract_method<'a>(obj: &'a Bound<'_, PyAny>) -> PyResult<&'a str> {
    obj.extract::<&str>().map_err(|_| {
        PyTypeError::new_err(format!(
            "method must be a str (\"xxh\" or \"superminhash\"), not {}",
            obj.get_type()
                .qualname()
                .map(|q| q.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "object".into())
        ))
    })
}

/// `bits`' extraction: `None` (the full-row default) or an int in [1, 63]
/// through the `__index__` protocol (`bool` rejected in every position,
/// including as an `__index__` result); the range check stays a
/// `ValueError` in the body.
fn extract_bits(obj: &Bound<'_, PyAny>) -> PyResult<Option<u32>> {
    if obj.is_none() {
        return Ok(None);
    }
    let index = extract_index(obj, "bits")?;
    let bits: i64 = index.extract::<i64>()?;
    Ok(Some(bits as u32))
}

/// The shared `bits` range validation: an int in [1, 63] (63 keeps the
/// u64 range's low 63 bits; the full row is `bits=None`, not 64, so the
/// mask can never be the identity and silently cost the caller its
/// compression for nothing).
fn validate_bits(bits: Option<u32>) -> PyResult<()> {
    match bits {
        None => Ok(()),
        Some(bits) if (1..=63).contains(&bits) => Ok(()),
        Some(bits) => Err(PyValueError::new_err(format!(
            "bits must be between 1 and 63, or None for full rows, not {bits}"
        ))),
    }
}

/// One signature argument's rows for the estimator functions: the strict
/// element convention the `lsh_candidates` walk spells (exact ints within
/// 2^64 - 1; negative: `ValueError`, past the range: `OverflowError`,
/// non-int including bool: `TypeError`; no per-element `__index__`
/// dispatch), with the PARAMETER NAME in every message (`sig_a`/`sig_b`,
/// not a list position -- the caller spelled the names at the call site).
fn extract_rows(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u64>> {
    if obj.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a sequence of ints, not str"
        )));
    }
    let seq = obj.cast::<pyo3::types::PySequence>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{name} must be a sequence of ints, not {}",
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
                "{name} elements must be ints, not bool"
            )));
        }
        match item.extract::<u64>() {
            Ok(value) => rows.push(value),
            Err(_) => return Err(row_element_error(&item, name)),
        }
    }
    Ok(rows)
}

/// The precise error for a row element the fast u64 extraction refused
/// (the `signature_element_error` shape, parameter-named).
fn row_element_error(item: &Bound<'_, PyAny>, name: &str) -> PyErr {
    let type_name = item
        .get_type()
        .qualname()
        .map(|q| q.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "object".into());
    match item.extract::<i128>() {
        Ok(value) if value < 0 => PyValueError::new_err(format!(
            "{name} elements must be non-negative ints: MinHash values are unsigned"
        )),
        Ok(value) => PyOverflowError::new_err(format!(
            "{name} element {value} is too large for a u64 MinHash value (max 2**64 - 1)"
        )),
        Err(_) => PyTypeError::new_err(format!("{name} elements must be ints, not {type_name}")),
    }
}

/// `tors.minhash_jaccard(sig_a, sig_b, *, bits=None) -> float`: the
/// paired-row Jaccard estimator over two `minhash_signature` signatures.
/// `bits=None` counts rows equal at full width and returns the agreement
/// fraction -- the one-expression estimator docs/api.md spells, exact
/// contract over the classic engine's rows and the agreement-fraction
/// estimator over SuperMinHash rows alike. `bits=b` (1..=63) is the
/// b-bit estimator over b-bit-compressed rows (Li and König, WDE 2010):
/// masked rows agree with probability `J + (1 - J) * 2^-b`, so the
/// returned estimate applies the chance correction
/// `(p_hat - 2^-b) / (1 - 2^-b)` -- unbiased, but able to land slightly
/// NEGATIVE at true similarity zero (a finite sample's chance term can
/// exceed the observed agreement); threshold callers should compare raw
/// agreement fractions instead. The variance matches full rows once
/// `b >= log2(1/J)`.
///
/// Both signatures must be equal-length and non-empty, else `ValueError`
/// naming the lengths; elements ride the strict int convention
/// (`lsh_candidates`'). Mixing signatures from different engines,
/// parameters, or seeds is a caller error (see `minhash_signature`):
/// nothing here can detect it, so the contract is documentation.
///
/// O(len(sig_a)) integer comparisons: no detach, no `aio` twin (the
/// `lsh_probability` stay-sync class -- the thread hop costs more than
/// the call).
#[pyfunction(signature = (sig_a, sig_b, *, bits = None))]
pub fn minhash_jaccard(
    sig_a: Bound<'_, PyAny>,
    sig_b: Bound<'_, PyAny>,
    #[pyo3(from_py_with = extract_bits)] bits: Option<u32>,
) -> PyResult<f64> {
    validate_bits(bits)?;
    let a = extract_rows(&sig_a, "sig_a")?;
    let b = extract_rows(&sig_b, "sig_b")?;
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "signature lengths differ: sig_a has {} rows, sig_b has {} rows; estimates \
             read only paired rows",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Err(PyValueError::new_err(
            "empty signatures estimate nothing: num_perm is at least 1 in every \
             minhash_signature call",
        ));
    }
    Ok(minhash_impl::jaccard_estimate(&a, &b, bits))
}

/// `tors.weighted_minhash_signature(text_or_tokens, *, num_perm, seed=None)
/// -> list[int]`: the Consistent Weighted Sampling signature (Ioffe, ICDM
/// 2010; Shrivastava, NeurIPS 2016) of the token MULTISET -- the
/// frequency-aware complement to `minhash_signature`. The binary MinHash
/// reads a document as its token SET (a token repeated thrice is one
/// element, indistinguishable from a single occurrence); the weighted
/// engine reads the counts, and the fraction of permutations whose
/// `(token, active-index)` pairs agree estimates the GENERALIZED Jaccard
/// similarity `sum_k min(w_a, w_b) / sum_k max(w_a, w_b)` -- 'aaa' counts
/// thrice, and a document holding 'aaa' three times is genuinely more
/// similar to one holding it four times than to one holding it once.
///
/// The input spellings, all one multiset:
///
/// - `str`: tokenized by the crate's one UAX #29 word stream (the
///   `tf_idf`/`bm25_rank` tokenizer, lowercased); each token occurrence
///   is a count.
/// - a sequence of `str`: each element ONE occurrence, no
///   re-tokenization -- a caller's domain-specific segmenter weighs
///   exactly as given.
/// - a mapping of `str` to a non-negative finite number: explicit
///   weights (fractional allowed; zero excludes the token; NaN, infinity,
///   and negatives are `ValueError`; `bool` values are `TypeError`).
///
/// Returns `2 * num_perm` rows, the winner pair `(token_hash, t)` per
/// permutation: the token's identity is the crate's one XXH64 contract
/// over the single-token frame (the shingle_size-1 shingle hash), `t` the
/// integer active index emitted as its f64 bit pattern (an injective
/// encoding, so fractional-weight signatures compare exactly too). Empty
/// text, an empty list, an empty mapping, or all-zero weights is the
/// empty-multiset convention: every row the u64 MAX sentinel.
///
/// The engine: per permutation j and token hash h with weight w, five
/// SplitMix64 draws from the XXH64 frame `[seed, j, h]` give
/// `r, c ~ Gamma(2, 1)` and `beta ~ U(0, 1)`; `t = floor(ln(w)/r + beta)`,
/// `y = exp(r*(t - beta))`, `z = y*exp(r)`, `a = c/z`; the permutation's
/// sample is the token minimizing `a` (ties, measure-zero, break to the
/// lowest token hash: tokens sweep in ascending-(hash, weight) order).
/// The estimator's consistency is Ioffe's theorem; the empirical accuracy
/// cell pins it against exact count-vector Jaccards. The engine has NO
/// prefix property and is NOT cross-compatible with the binary engines'
/// signatures -- the generalized Jaccard is a different quantity than the
/// Jaccard, and `minhash_weighted_jaccard` is the only estimator that
/// reads these rows.
///
/// Bounds: `num_perm` must be in `[1, 1024]` (`ValueError` naming the
/// bounds; `__index__` accepted, `bool` rejected); `seed` is any int or
/// None, None meaning the fixed default seed 0 -- there is no random
/// entropy anywhere in the engine, so the same input at the same
/// parameters is the same signature across processes, machines, and
/// versions. Mapping keys must be `str` and list/tuple elements `str`
/// (`TypeError` otherwise). Cost is O(tokens) hashing to count the
/// multiset plus O(num_perm * distinct_tokens) in the ICWS sweep.
///
/// GIL model: the input walk and validation under the GIL, the count +
/// sweep pass under one `py.detach`, then the `2 * num_perm`-element
/// int-list marshalling. The `aio` twin
/// (`tors.aio.weighted_minhash_signature`) is the same call under
/// `asyncio.to_thread`.
#[pyfunction(signature = (text_or_tokens, *, num_perm, seed = None))]
pub fn weighted_minhash_signature(
    py: Python<'_>,
    text_or_tokens: Bound<'_, PyAny>,
    #[pyo3(from_py_with = extract_num_perm)] num_perm: i64,
    #[pyo3(from_py_with = normalize_seed_optional)] seed: Option<u64>,
) -> PyResult<Vec<u64>> {
    let seed = seed.unwrap_or(0);
    if !(1..=1024).contains(&num_perm) {
        return Err(PyValueError::new_err(format!(
            "num_perm must be between 1 and 1024, not {num_perm}"
        )));
    }
    enum Weights {
        Text(String),
        Tokens(Vec<String>),
        Explicit(Vec<(u64, f64)>),
    }
    let input = if text_or_tokens.is_instance_of::<PyString>() {
        // The str path is checked FIRST and its extraction propagates
        // unchanged: text bearing lone surrogates raises the crate-wide
        // UnicodeEncodeError here, not a reclassified TypeError from the
        // fall-through.
        Weights::Text(text_or_tokens.extract::<String>()?)
    } else if let Ok(dict) = text_or_tokens.cast::<PyDict>() {
        let mut items: Vec<(u64, f64)> = Vec::new();
        for (key, value) in dict.iter() {
            let token: String = key.extract().map_err(|_| {
                PyTypeError::new_err(format!(
                    "weight keys must be str (the token), not {}",
                    key.get_type()
                        .qualname()
                        .map(|q| q.to_string_lossy().into_owned())
                        .unwrap_or_else(|_| "object".into())
                ))
            })?;
            if value.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(
                    "weight values must be numbers, not bool",
                ));
            }
            let weight: f64 = value.extract().map_err(|_| {
                PyTypeError::new_err(format!(
                    "weight values must be numbers, not {}",
                    value
                        .get_type()
                        .qualname()
                        .map(|q| q.to_string_lossy().into_owned())
                        .unwrap_or_else(|_| "object".into())
                ))
            })?;
            if !weight.is_finite() || weight < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "weight values must be finite and non-negative, not {weight}"
                )));
            }
            let hash = minhash_impl::token_hash(&token);
            items.push((hash, weight));
        }
        Weights::Explicit(items)
    } else if let Ok(tokens) = text_or_tokens.extract::<Vec<String>>() {
        Weights::Tokens(tokens)
    } else {
        return Err(PyTypeError::new_err(format!(
            "text_or_tokens must be a str, a sequence of str, or a mapping of str to \
             non-negative numbers, not {}",
            text_or_tokens
                .get_type()
                .qualname()
                .map(|q| q.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "object".into())
        )));
    };
    Ok(py.detach(|| match input {
        Weights::Text(text) => {
            minhash_impl::weighted_signature_text(&text, num_perm as usize, seed)
        }
        Weights::Tokens(tokens) => {
            minhash_impl::weighted_signature_tokens(&tokens, num_perm as usize, seed)
        }
        Weights::Explicit(items) => {
            minhash_impl::weighted_signature_from_weights(items, num_perm as usize, seed)
        }
    }))
}

/// `tors.minhash_weighted_jaccard(sig_a, sig_b) -> float`: the
/// generalized-Jaccard estimator over two
/// `weighted_minhash_signature` signatures: the fraction of permutations
/// whose `(token_hash, active-index)` pairs agree (Ioffe's consistency
/// property). Both signatures must be equal-length, even-length, and
/// non-empty (`ValueError` naming the shape -- an odd row count is not an
/// ICWS signature); the result is in [0, 1] and estimates
/// `sum_k min(w_a, w_b) / sum_k max(w_a, w_b)` over the two token
/// multisets. O(len(sig_a) / 2) integer comparisons: no detach, no `aio`
/// twin (the `lsh_probability` stay-sync class).
#[pyfunction(signature = (sig_a, sig_b))]
pub fn minhash_weighted_jaccard(sig_a: Bound<'_, PyAny>, sig_b: Bound<'_, PyAny>) -> PyResult<f64> {
    let a = extract_rows(&sig_a, "sig_a")?;
    let b = extract_rows(&sig_b, "sig_b")?;
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "signature lengths differ: sig_a has {} rows, sig_b has {} rows; estimates \
             read only paired rows",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() || a.len() % 2 != 0 {
        return Err(PyValueError::new_err(format!(
            "not an ICWS signature shape: weighted_minhash_signature returns 2 * num_perm \
             rows ({} here)",
            a.len()
        )));
    }
    Ok(minhash_impl::weighted_jaccard_estimate(&a, &b))
}
