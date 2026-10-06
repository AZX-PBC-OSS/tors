use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySequence};

use crate::py::rank_fusion::{check_k, check_weights};
use crate::score_fusion_impl;

/// `tors.score_fuse(scored_lists, *, method="combmnz", weights=None, k=None)
/// -> list[tuple[Hashable, float]]`: score-based fusion, the score-space
/// sibling of `rank_fuse` (same id table, same first-appearance
/// tie-break, same emission contract; raw similarity scores consumed
/// instead of ranks). Three methods:
///
/// - `method="combmnz"` (the default), Fox & Shaw, "Combination of
///   Multiple Searches", TREC-2 1994: `score(d) = lists(d) × Σ_lists
///   w_i × norm(score_i(d))`, where `norm` is min-max per list over the
///   list's OWN scores and `lists(d)` counts the lists containing `d`
///   (CombSUM × the consensus count). The best of the score-based
///   family in Cormack, Clarke & Buüttcher's SIGIR 2009 comparison (the
///   RRF paper's own baseline set,
///   https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf), hence the
///   default.
/// - `method="borda"`: the rank-based count, `score(d) = Σ_lists w_i ×
///   (n_i - rank_i(d)) / n_i`, ranks 1-based over the DEDUPLICATED
///   list (a duplicate folds to its first occurrence and later ids move
///   up, `rank_fuse`'s contract). Deliberately RANK-based, not
///   score-based: Borda counts are defined over positions -- each list
///   elects its top document with `n-1` points and its last with 0 --
///   and the `(n - rank)/n` spelling only rescales the count to `[0,
///   1)` so a weight means the same thing over lists of any length;
///   normalizing the scores there would smuggle magnitudes into the
///   one method whose entire point is that only the ordering votes.
/// - `method="linear"`: Elasticsearch's linear-retriever pattern
///   (https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/linear-retriever):
///   `score(d) = Σ_lists w_i × norm(score_i(d))`, no MNZ multiplier,
///   min-max normalized, the same per-retriever `weight` shape weighted
///   RRF carries.
///
/// The normalization conventions (pinned in tests/test_score_fusion.py):
/// min-max runs per list over that list's OWN (deduplicated) scores --
/// a folded duplicate's score does not shape the range; a list whose
/// scores are ALL equal (a single-entry list included) normalizes every
/// entry to the neutral midpoint `0.5` (the list carries order
/// information only; the midpoint invents neither a winner nor a
/// loser); negative scores are LEGAL (min-max maps any finite range
/// onto `[0, 1]`, so the signs of a cosine-similarity list wash out and
/// a BM25 list normalizes to the same interval -- the rejected domain
/// is only non-finite scores, `ValueError` for NaN and both infinites,
/// the `gains` finite-domain discipline, because an infinite min or
/// max makes the range ill-defined); and legal finite scores can span
/// so far the range itself overflows (`-1.7e308` to `1.7e308`), where
/// an overflowing numerator would answer IEEE `inf/inf = NaN` -- the
/// core saturates instead (an overflowing numerator is the top of the
/// scale and answers exactly `1.0`, finite numerators over an infinite
/// range divide to `0.0`-scale values), the monotone, NaN-free policy
/// `ndcg_at_k`'s saturating ratio carries, with the same documented
/// one-sided cost (near-top scores can over-report as exactly `1.0`).
///
/// `scored_lists` is a list of lists of `(id, score)` pairs (scores
/// this time: a BM25 output, a cosine similarity, a click count). A
/// duplicate id folds to its FIRST occurrence per list (its first
/// score stands; the same id in a DIFFERENT list is a legitimate
/// second vote with its own score, the whole point of fusion).
/// `weights` optionally carries one positive finite float per list
/// (the weighted-RRF philosophy extended to score space: a weight
/// re-scales one list's contribution; `weights=None`, the default, is
/// the all-1.0 unweighted fusion EXACTLY, outputs byte-identical to
/// the unweighted spelling, pinned). A zero, negative, NaN, or
/// infinite weight raises `ValueError`; a length mismatch with
/// `scored_lists` raises `ValueError` naming both sides; a
/// non-sequence `weights` (a bare `str` included) or a non-numeric
/// entry raises `TypeError`. `method` must be one of `"combmnz"`,
/// `"borda"`, `"linear"` (`ValueError` naming the accepted set
/// otherwise). `k=None` (the default) returns every distinct id;
/// `k=N` the top N (`ValueError` for `k < 1`).
///
/// Returns `(id, score)` pairs for every distinct id across all lists,
/// sorted by fused score descending, ties broken by earliest first
/// appearance across the lists in caller order (the `rank_fuse`
/// contract, extended to score space). Returned ids are the original
/// objects (references, zero marshalling); dedup and equality follow
/// Python's own dict/set semantics (`1`, `True`, and `1.0` are the
/// same id). Emission is vote-existence, not score positivity: a
/// fused `0.0` (Borda's last place, an underflowed denormal weight)
/// still appears, ordered last. Scores are finite-or-`+inf`, never
/// NaN (all-nonneg sums; the saturating norm above). `scored_lists`
/// must be a non-empty list of lists (fusing zero lists raises
/// `ValueError`, the `merkle_root` "root of no chunks" precedent
/// `rank_fuse` keeps); an individual empty list is legal and
/// contributes no votes. An unhashable id raises `TypeError`
/// (Python's own hash error); a malformed pair (not a 2-element
/// sequence) or a non-numeric score raises `TypeError`.
///
/// GIL model: one GIL-held walk of every pair (Python-object hashing
/// IS interpreter work: the id table is a dict, the `rank_fuse`
/// arg-walk class plus one score extraction per entry), the `weights`
/// walk and validation when supplied, then the per-list min-max,
/// normalization, weighted accumulation, MNZ counts, and sort (plain
/// arithmetic over dedup indices) under one `py.detach`, then the
/// O(distinct-ids) `(id, score)` tuple marshalling.
#[pyfunction(signature = (scored_lists, *, method = "combmnz", weights = None, k = None))]
pub fn score_fuse(
    py: Python<'_>,
    scored_lists: Bound<'_, PyList>,
    method: &str,
    weights: Option<&Bound<'_, PyAny>>,
    k: Option<i64>,
) -> PyResult<Py<PyAny>> {
    // Argument-domain checks before any caller id is hashed (the
    // argument-validation-before-work order every binding in this
    // crate keeps): k, then the method name, then the weights.
    let k = k.map(check_k).transpose()?;
    let method = score_fusion_impl::parse_fusion_method(method).map_err(PyValueError::new_err)?;
    if scored_lists.is_empty() {
        return Err(PyValueError::new_err(
            "score_fuse needs at least one scored list; fusing zero lists has no \
             defined answer (every call site so far meant an upstream bug)",
        ));
    }
    let weights: Vec<f64> = match weights {
        None => Vec::new(),
        Some(any) => check_weights(any, scored_lists.len(), "score_fuse", "scored")?,
    };
    // The GIL-held pair walk: one dict (id -> first-appearance index)
    // and one id table in that order, plus the per-entry score
    // extraction and finite-domain check. Dict lookups raise Python's
    // own TypeError on an unhashable id; equality semantics are the
    // dict's; a within-list duplicate folds to its FIRST occurrence's
    // score (a repeat is skipped entirely).
    let mut ids: Vec<Py<PyAny>> = Vec::new();
    let table = pyo3::types::PyDict::new(py);
    let mut last_list: Vec<u32> = Vec::new();
    let mut lists_scores: Vec<Vec<(u32, f64)>> = Vec::with_capacity(scored_lists.len());
    for (list_idx, list_obj) in scored_lists.iter().enumerate() {
        let list = list_obj.cast::<PyList>().map_err(|_| {
            let type_name = list_obj
                .get_type()
                .qualname()
                .and_then(|name| name.to_str().map(str::to_string))
                .unwrap_or_else(|_| "object".to_string());
            PyTypeError::new_err(format!(
                "scored_lists entry {list_idx} must be a list of (id, score) pairs, not {type_name}"
            ))
        })?;
        let mut pairs = Vec::with_capacity(list.len());
        for (entry_idx, item) in list.iter().enumerate() {
            let pair = item.cast::<PySequence>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "scored_lists entry {list_idx} element {entry_idx} must be an (id, score) pair"
                ))
            })?;
            let len = pair.len().map_err(|_| {
                PyTypeError::new_err(format!(
                    "scored_lists entry {list_idx} element {entry_idx} must be an (id, score) pair"
                ))
            })?;
            if len != 2 {
                return Err(PyTypeError::new_err(format!(
                    "scored_lists entry {list_idx} element {entry_idx} must be an \
                     (id, score) pair, got {len} element(s)"
                )));
            }
            let id = pair.get_item(0)?;
            let score = pair.get_item(1)?.extract::<f64>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "scored_lists entry {list_idx} element {entry_idx} score must be a number"
                ))
            })?;
            if !score.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "scores values must be finite, got {score}"
                )));
            }
            if let Some(existing) = table.get_item(&id)? {
                let idx = existing.extract::<u32>()?;
                if last_list[idx as usize] != list_idx as u32 {
                    pairs.push((idx, score));
                    last_list[idx as usize] = list_idx as u32;
                }
                continue;
            }
            let idx = ids.len() as u32;
            table.set_item(&id, idx)?;
            ids.push(id.clone().unbind());
            last_list.push(list_idx as u32);
            pairs.push((idx, score));
        }
        lists_scores.push(pairs);
    }
    // The detached fusion pass: pure arithmetic over dedup indices.
    // The weights (validated above, one per list) ride in as a slice.
    let fused = py.detach(|| {
        score_fusion_impl::score_fuse(
            &lists_scores,
            method,
            if weights.is_empty() {
                None
            } else {
                Some(&weights)
            },
            k,
        )
    });
    // The marshalling: original id objects by index, one tuple each.
    // Scaling-pin note (the rank_fuse marshalling note's own shape): a
    // regression here must be `std::hint::black_box`-wrapped to be
    // measured at all; the per-entry tuple append is the measured
    // O(distinct-ids) band tests/test_scaling_pins.py's
    // TestScoreFusionScaling holds.
    let out = PyList::empty(py);
    for (idx, score) in fused {
        out.append((ids[idx as usize].bind(py), score))?;
    }
    Ok(out.into_any().unbind())
}
