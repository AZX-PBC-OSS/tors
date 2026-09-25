use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet};

use crate::rank_fusion_impl;

/// The `relevant` argument's accepted spellings, checked once up front:
/// exactly a `set` or `frozenset` of ids (the same exactly-list
/// discipline `bm25_rank`'s `corpus` keeps: a `list`/`tuple`/generator
/// is a `TypeError` naming the accepted set, not a silent copy into a
/// fresh set; a caller with a list has `set(...)`). Membership tests
/// then ride the set's own `__contains__` through
/// `PySequence_Contains`: Python's hash/equality semantics, unhashable
/// ids included (the interpreter's own "unhashable type" `TypeError`).
fn check_relevant(relevant: &Bound<'_, PyAny>) -> PyResult<()> {
    if relevant.is_instance_of::<PySet>() || relevant.is_instance_of::<pyo3::types::PyFrozenSet>() {
        Ok(())
    } else {
        Err(PyTypeError::new_err(
            "relevant must be a set (or frozenset) of ids",
        ))
    }
}

/// `k`'s shared validation: at least 1 (the fusion/metric contract),
/// extracted as i64 so an out-of-range int is pyo3's own OverflowError
/// (the `truncate_to_bounds`-identical pattern) and a non-int is a
/// `TypeError`; a negative or zero `k` is a range error, `ValueError`
/// naming the bound.
fn check_k(k: i64) -> PyResult<usize> {
    if k < 1 {
        return Err(PyValueError::new_err(format!("k must be >= 1, got {k}")));
    }
    Ok(k as usize)
}

/// One graded-relevance value's validation: finite and non-negative
/// (the standard nDCG gain domain; a negative or non-finite gain would
/// corrupt the ideal ranking the score normalizes against).
fn check_gain(id: &Bound<'_, PyAny>, gain: f64) -> PyResult<f64> {
    if !gain.is_finite() || gain < 0.0 {
        return Err(PyValueError::new_err(format!(
            "gains values must be finite and >= 0, got {gain} for {id}"
        )));
    }
    Ok(gain)
}

/// `tors.rank_fuse(ranked_lists, *, k=60) -> list[tuple[Hashable, float]]`:
/// Reciprocal Rank Fusion (Cormack, Clarke & Buüttcher, SIGIR 2009,
/// https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf): fuses multiple
/// ranked lists of hashable doc ids into one ranking,
/// `score(d) = sum over lists of 1 / (k + rank(d))`, ranks 1-based,
/// `k=60` (the paper's own default, unchanged across its experiments).
/// Ranks only, never raw scores: the paper's whole point is that raw
/// scores from different retrieval systems are not comparable while
/// ranks are, so a doc absent from a list contributes no vote from it,
/// and a doc ranked twice in one list votes once (at its FIRST
/// occurrence: the dedup-first pass folds duplicates).
///
/// Returns `(id, score)` pairs for every distinct id across all lists,
/// sorted by fused score descending, ties broken by earliest first
/// appearance across the lists in caller order (a point the paper leaves
/// open, pinned here as contract). Returned ids are the original
/// objects (references, zero marshalling); dedup and membership follow
/// Python's own dict/set equality (`1` and `True` are the same id, `1`
/// and `1.0` likewise).
///
/// `k` must be >= 1 (`ValueError`); `ranked_lists` must be a non-empty
/// list of non-empty-or-empty lists (fusing zero lists raises
/// `ValueError`; the `merkle_root` "root of no chunks" precedent: the
/// formula is defined over one-or-more lists, and a zero-list call is
/// almost certainly an upstream bug, while an individual empty list is
/// legal and contributes no votes, the "this retriever returned nothing"
/// shape). An unhashable id raises `TypeError` (Python's own hash error:
/// a dict cannot key it, the same wrong-type-entry contract
/// `bm25_rank`'s corpus walk keeps).
///
/// GIL model: one GIL-held walk of every list (Python-object hashing IS
/// interpreter work: each occurrence is one dict lookup (an existing
/// entry folds a within-list duplicate, a fresh one joins the id table);
/// the same arg-walk class `content_hash`'s object walk is), then the
/// score accumulation + sort (plain arithmetic over dedup indices) under
/// one `py.detach`, then the O(distinct-ids) `(id, score)` tuple
/// marshalling.
#[pyfunction(signature = (ranked_lists, *, k = 60))]
pub fn rank_fuse(py: Python<'_>, ranked_lists: Bound<'_, PyList>, k: i64) -> PyResult<Py<PyAny>> {
    let k = check_k(k)?;
    if ranked_lists.is_empty() {
        return Err(PyValueError::new_err(
            "rank_fuse needs at least one ranked list; fusing zero lists has no \
             defined answer (every call site so far meant an upstream bug)",
        ));
    }
    // The GIL-held dedup pass: one dict (id -> first-appearance index)
    // and one id table in that order. Dict lookups raise Python's own
    // TypeError on an unhashable id; equality semantics are the dict's.
    let mut ids: Vec<Py<PyAny>> = Vec::new();
    let table = PyDict::new(py);
    // Per-id last-seen list index: the dedup is WITHIN one list only;
    // the same id in a different list is a legitimate second vote (that
    // is the whole point of fusion), so a repeat is skipped exactly
    // when its previous occurrence was in THIS list.
    let mut last_list: Vec<u32> = Vec::new();
    let mut lists_idx: Vec<Vec<u32>> = Vec::with_capacity(ranked_lists.len());
    for (list_idx, list_obj) in ranked_lists.iter().enumerate() {
        let list = list_obj.cast::<PyList>().map_err(|_| {
            let type_name = list_obj
                .get_type()
                .qualname()
                .and_then(|name| name.to_str().map(str::to_string))
                .unwrap_or_else(|_| "object".to_string());
            PyTypeError::new_err(format!(
                "ranked_lists entry {list_idx} must be a list of ids, not {type_name}"
            ))
        })?;
        let mut indices = Vec::with_capacity(list.len());
        for item in list {
            if let Some(existing) = table.get_item(item.as_any())? {
                let idx = existing.extract::<u32>()?;
                if last_list[idx as usize] != list_idx as u32 {
                    indices.push(idx);
                    last_list[idx as usize] = list_idx as u32;
                }
                continue;
            }
            let idx = ids.len() as u32;
            table.set_item(item.as_any(), idx)?;
            ids.push(item.as_any().clone().unbind());
            last_list.push(list_idx as u32);
            indices.push(idx);
        }
        lists_idx.push(indices);
    }
    // The detached fusion pass: pure arithmetic over dedup indices.
    let fused = py.detach(|| rank_fusion_impl::rank_fuse(&lists_idx, k as u64, ids.len()));
    // The marshalling: original id objects by index, one tuple each.
    // Scaling-pin note: an injected regression here must be
    // `std::hint::black_box`-wrapped to be measured at all. A
    // `take(pos + 1).count()`-style "re-scan" is not a valid probe:
    // the id vector is an ExactSizeIterator, so LLVM folds the length
    // subtraction to O(1) in release and the injection elides. The
    // black-boxed per-entry recompute over `fused[..pos + 1]` is the
    // elision-proof form; it fails tests/test_rank_fusion.py's 9.0x
    // scaling gate 5/5 at the 10k -> 40k span (measured 13-15x).
    let out = PyList::empty(py);
    for (idx, score) in fused {
        out.append((ids[idx as usize].bind(py), score))?;
    }
    Ok(out.into_any().unbind())
}

/// `tors.ndcg_at_k(ranked, relevant, *, k=None, gains=None) -> float`:
/// normalized discounted cumulative gain at `k` (Järvelin & Kekäläinen,
/// ACM TOIS 20(4), 2002), in `[0.0, 1.0]`. `ranked` is a list of ids
/// (best first); `relevant` is a set of relevant ids (binary relevance
/// 1.0) or, with `gains`, the baseline set whose members a graded
/// `gains` dict overrides: the gain of id `d` is `gains[d]` when the
/// dict contains it, else `1.0` when `d` is in `relevant`, else `0.0`.
/// A duplicated id inside `ranked` counts once, at its first occurrence
/// (the same dedup-first contract `rank_fuse` keeps: a repeat is a
/// malformed ranking, and counting it twice would inflate the DCG past
/// what the ideal can match).
///
/// The DCG uses the paper's log2 discount, rank 1 undiscounted:
/// `DCG@k = sum over i in 1..k of gain_i / log2(i + 1)` over the linear
/// gain function (for binary relevance the paper's exponential
/// `2^rel - 1` variant is identical). The ideal DCG sorts the complete
/// judged pool, every id in `relevant` (at its gain) plus every
/// `gains` key, descending and discounts the same way, so the score is
/// exactly the paper's normalization.
///
/// `k=None` (the default) scores the whole ranking; `k` is clamped to
/// the deduplicated ranking's length. Edge inputs are well-defined zeros: an
/// empty `ranked`, an empty `relevant` (with no `gains`), and the
/// zero-ideal-DCG case (nothing judged relevant) all answer `0.0`.
/// Legal finite gains can be so large the DCG and IDCG sums overflow to
/// `+inf` (three gains of 1e308, or two of 1.7e308); the normalization
/// saturates instead of dividing `inf/inf` (NaN): when either sum is
/// non-finite the score is `1.0` if `DCG >= IDCG` else `0.0`, and a
/// finite ratio clamps to `[0.0, 1.0]` (the monotone-total policy, since
/// the ideal pool contains every ranked gain under the same discount
/// schedule, so an overflowed DCG can at most match the overflowed ideal.
/// `k < 1` and a negative or non-finite `gains` value raise `ValueError`;
/// a non-list `ranked`, a non-set `relevant`, a non-dict `gains`, or a
/// non-numeric `gains` value raise `TypeError` (the extraction failure);
/// an unhashable id raises
/// `TypeError` (Python's own hash error).
///
/// GIL model: the argument checks and the per-position gain walk (one
/// `__contains__`/dict lookup per ranked id, interpreter hashing) under
/// the GIL, the DCG/IDCG arithmetic under one `py.detach`, a single
/// float out (no marshalling class).
#[pyfunction(signature = (ranked, relevant, *, k = None, gains = None))]
pub fn ndcg_at_k(
    py: Python<'_>,
    ranked: Bound<'_, PyList>,
    relevant: Bound<'_, PyAny>,
    k: Option<i64>,
    gains: Option<&Bound<'_, PyDict>>,
) -> PyResult<f64> {
    check_relevant(&relevant)?;
    let k = k.map(check_k).transpose()?;
    // The GIL-held gain walk: per-position gains over `ranked` (the
    // dedup-first contract: a repeat is skipped entirely, the same fold
    // rank_fuse applies), plus the judged pool for the ideal (relevant
    // ids at 1.0 where `gains` does not override, plus every gains key
    // at its value).
    let seen = PySet::empty(py)?;
    let mut position_gains: Vec<f64> = Vec::with_capacity(ranked.len());
    for item in ranked.iter() {
        if seen.contains(item.as_any())? {
            continue;
        }
        seen.add(item.as_any())?;
        let gain = match gains {
            Some(gains) => match gains.get_item(item.as_any())? {
                Some(value) => check_gain(&item, value.extract::<f64>()?)?,
                None => {
                    if relevant.contains(item.as_any())? {
                        1.0
                    } else {
                        0.0
                    }
                }
            },
            None => {
                if relevant.contains(item.as_any())? {
                    1.0
                } else {
                    0.0
                }
            }
        };
        position_gains.push(gain);
    }
    let mut ideal_pool: Vec<f64> = Vec::new();
    if let Some(gains) = gains {
        for (id, value) in gains.iter() {
            ideal_pool.push(check_gain(&id, value.extract::<f64>()?)?);
        }
        for id in relevant.try_iter()? {
            let id = id?;
            if gains.get_item(&id)?.is_none() {
                ideal_pool.push(1.0);
            }
        }
    } else {
        ideal_pool.resize(relevant.len()?, 1.0);
    }
    let k = k.map_or(position_gains.len(), |k| k.min(position_gains.len()));
    Ok(py.detach(|| rank_fusion_impl::ndcg_at_k(&position_gains, ideal_pool, k)))
}

/// `tors.mrr(ranked, relevant) -> float`: mean reciprocal rank of a
/// single ranking: the reciprocal rank of the first relevant result,
/// `1/rank` with ranks 1-based, `0.0` when no ranked result is relevant
/// (and for an empty `ranked`: the same well-defined zero). A
/// duplicated id counts once at its first occurrence (the family's
/// dedup-first contract). `relevant`
/// is a set of relevant ids (exactly a set or frozenset; `TypeError`
/// otherwise); an unhashable id raises `TypeError` (Python's own hash
/// error).
///
/// GIL model: the per-position membership walk (interpreter hashing)
/// under the GIL, the trivial reciprocal under one `py.detach`, a single
/// float out.
#[pyfunction(signature = (ranked, relevant))]
pub fn mrr(py: Python<'_>, ranked: Bound<'_, PyList>, relevant: Bound<'_, PyAny>) -> PyResult<f64> {
    check_relevant(&relevant)?;
    let flags = relevance_flags(&ranked, &relevant)?;
    Ok(py.detach(|| rank_fusion_impl::mrr(&flags)))
}

/// `tors.recall_at_k(ranked, relevant, k) -> float`: the fraction of the
/// relevant set found in the top `k` positions,
/// `|relevant ∩ ranked[:k]| / |relevant|` (the formula assumes deduped
/// input: a duplicate counts once, at its first occurrence). A `k` past
/// the ranking's
/// length simply uses every available position; a duplicated id counts
/// once at its first occurrence (the family's dedup-first contract).
/// The empty-relevant case is the core's own documented `0.0` (pinned
/// crate-side). `k < 1` raises
/// `ValueError`; a non-set `relevant` raises `TypeError`; an unhashable
/// id raises `TypeError` (Python's own hash error).
///
/// GIL model: `mrr`'s classes exactly (membership walk under the GIL,
/// arithmetic detached, single float out).
#[pyfunction(signature = (ranked, relevant, k))]
pub fn recall_at_k(
    py: Python<'_>,
    ranked: Bound<'_, PyList>,
    relevant: Bound<'_, PyAny>,
    k: i64,
) -> PyResult<f64> {
    check_relevant(&relevant)?;
    let k = check_k(k)?;
    let flags = relevance_flags(&ranked, &relevant)?;
    let n_relevant = relevant.len()?;
    Ok(py.detach(|| rank_fusion_impl::recall_at_k(&flags, n_relevant, k)))
}

/// `tors.precision_at_k(ranked, relevant, k) -> float`: the fraction of
/// the top `k` positions that are relevant,
/// `|relevant ∩ ranked[:k]| / min(k, len(ranked))`, trec_eval's own
/// convention for a run shorter than `k`: a system that returned fewer
/// results is not punished for positions it never filled (the formula
/// assumes deduped input: a duplicate counts once, at its first
/// occurrence). A duplicated
/// id counts once at its first occurrence (the family's dedup-first
/// contract), so the denominator is the DEDUPLICATED ranking's length
/// clamped to `k`. Edge inputs
/// are well-defined zeros: an empty `ranked` and an empty `relevant`
/// both answer `0.0`. `k < 1` raises `ValueError`; a non-set `relevant`
/// raises `TypeError`; an unhashable id raises `TypeError` (Python's
/// own hash error).
///
/// GIL model: `mrr`'s classes exactly.
#[pyfunction(signature = (ranked, relevant, k))]
pub fn precision_at_k(
    py: Python<'_>,
    ranked: Bound<'_, PyList>,
    relevant: Bound<'_, PyAny>,
    k: i64,
) -> PyResult<f64> {
    check_relevant(&relevant)?;
    let k = check_k(k)?;
    let flags = relevance_flags(&ranked, &relevant)?;
    Ok(py.detach(|| rank_fusion_impl::precision_at_k(&flags, k)))
}

/// The shared GIL-held membership walk: one `__contains__` dispatch per
/// ranked id (Python's own hash/equality semantics), into the plain
/// flags the detached metric arithmetic consumes. Dedup-first, the same
/// contract `rank_fuse` keeps: a repeat inside `ranked` is skipped
/// entirely; the same document cannot occupy two metric positions.
fn relevance_flags(ranked: &Bound<'_, PyList>, relevant: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
    let py = ranked.py();
    let seen = PySet::empty(py)?;
    let mut flags = Vec::with_capacity(ranked.len());
    for item in ranked.iter() {
        if seen.contains(item.as_any())? {
            continue;
        }
        seen.add(item.as_any())?;
        flags.push(relevant.contains(item.as_any())?);
    }
    Ok(flags)
}
