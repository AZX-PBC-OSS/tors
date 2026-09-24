//! Reciprocal Rank Fusion and the IR ranking metrics: the pure-Rust core
//! of `tors.rank_fuse`, `tors.ndcg_at_k`, `tors.mrr`, `tors.recall_at_k`,
//! and `tors.precision_at_k`.
//!
//! # What this is: rank-space arithmetic over an already-materialized
//! candidate universe
//!
//! Every function here operates on `rank`-space data the pyo3 layer
//! materializes from the caller's Python objects: `rank_fuse` on
//! per-list vectors of dedup indices (the py layer assigns one index per
//! distinct id, in first-appearance order, using Python's own dict
//! equality; the ids themselves never cross into this module), the
//! metrics on per-position relevance flags (or gains). That split is the
//! same one `bm25_impl` draws: Python-object hashing/equality is
//! interpreter work and stays under the GIL in the binding layer, while
//! the arithmetic here is plain Rust over plain numbers and runs under
//! one `py.detach`.
//!
//! # Reciprocal Rank Fusion
//!
//! Cormack, Clarke & Buüttcher, "Reciprocal Rank Fusion outperforms
//! Condorcet and individual Rank Learning Methods", SIGIR 2009
//! (https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf): with `r(d)`
//! a document's rank in one list (ranks begin at 1), fused over `L`
//! lists:
//!
//! ```text
//! score(d) = sum over lists of 1 / (k + r(d))
//! ```
//!
//! RRF consumes RANKS ONLY, never raw similarity scores: the paper's
//! whole point is that raw scores from different retrieval systems are
//! not comparable, while ranks are; the single constant `k` (default 60,
//! the paper's own, used unchanged across its experiments) dampens the
//! weight of top ranks so a list's #1 does not swamp every other list's
//! votes. tors follows the paper exactly: ranks 1-based, one shared `k`
//! over all lists, and (a point the paper leaves open, pinned here as
//! contract) ties broken by EARLIEST FIRST APPEARANCE across the lists
//! in caller order (the natural deterministic reading: earlier evidence
//! outranks later), not by id value or sort-stable accident.
//!
//! A document absent from a list simply contributes no vote from it
//! (there is no "rank N+1 penalty" in RRF); a document appearing twice
//! in ONE list is a malformed ranking: the dedup-first binding pass
//! folds duplicates to their first occurrence, so a duplicate can never
//! vote twice.
//!
//! # The metrics
//!
//! - **nDCG**: Järvelin & Kekäläinen, "Cumulated gain-based evaluation of
//!   IR techniques", ACM TOIS 20(4), 2002: the discounted gain at rank
//!   `i` (1-based) is `gain / log2(i + 1)` (the paper's log2 discount,
//!   which keeps rank 1 undiscounted) and the score is
//!   `DCG@k / IDCG@k`, the ideal DCG computed by sorting the judged
//!   relevance values descending and discounting the same way. tors uses
//!   the paper's linear gain function (for binary relevance the paper's
//!   exponential `2^rel - 1` variant is identical: `2^1 - 1 = 1`).
//! - **MRR**: the reciprocal rank of the first relevant result (0.0 when
//!   no ranked result is relevant).
//! - **recall@k / precision@k**: the standard definitions,
//!   `|relevant ∩ ranked[:k]| / |relevant|` and `|relevant ∩ ranked[:k]| / min(k,
//!   |ranked|)` (trec_eval's convention: a run shorter than `k` is not
//!   punished for positions it never filled). Both formulas assume
//!   deduped input; a duplicate counts once, at its first occurrence.
//!
//! # Edge-input policy (the module's one contract, stated once)
//!
//! Empty DATA is a well-defined zero: an empty `ranked` list, an empty
//! relevant set (no relevant document exists, so no hit is possible),
//! and the zero-ideal-DCG case (nDCG of a ranking with nothing relevant)
//! all answer `0.0`. Empty STRUCTURE (fusing zero lists) is a caller
//! bug and raises (the binding layer's `ValueError`, the `merkle_root`
//! "root of no chunks" precedent: the paper's formula is defined over
//! one-or-more lists, and a zero-list call almost certainly means an
//! upstream bug). Numeric arguments out of range (`k < 1`) raise too;
//! the binding layer owns that validation exactly the way
//! `bm25_impl`'s `k1`/`b` contract does.
//!
//! # The nDCG normalization under overflow (the saturating-ratio policy)
//!
//! The binding layer validates every gain finite and non-negative, but
//! legal finite gains can still be so large the DCG and IDCG sums
//! overflow to `+inf` (three gains of 1e308, or two of 1.7e308, are
//! enough), and IEEE `inf / inf` is NaN, which would break the pinned
//! `[0, 1]` contract on a legal input. The core therefore normalizes
//! with overflow-aware, saturating logic instead of a bare division:
//! when either sum is non-finite the score is `1.0` if `dcg >= idcg`
//! else `0.0`, and a finite ratio is clamped to `[0, 1]`. The policy is
//! the monotone-total one: both sums are sums of non-negative terms
//! under one discount schedule, with the ideal pool a superset of the
//! ranked gains, so an overflowed DCG can at most match, never beat,
//! an overflowed ideal, and `1.0` is the only in-interval answer
//! consistent with the ordering the finite arithmetic reports. The
//! saturation can over-report a ranking whose exact ratio sits below 1
//! once the sums overflow (the low-order terms are lost), and the
//! `0.0` branch (finite DCG facing an infinite ideal) can under-report
//! a near-perfect ranking all the way to 0.0; within that branch the
//! error magnitude is unbounded across [0, 1]. Pinning the exact ratio
//! would cost a scale-normalizing pre-pass over every call for an
//! input class (gains within ~16 orders of magnitude of f64's
//! ceiling) no caller supplies, so the saturation is the documented
//! approximation, not a hidden one.

/// Reciprocal-rank-fuses the deduplicated lists (Cormack, Clarke &
/// Buüttcher, SIGIR 2009: `score(d) = Σ 1/(k + r(d))`, ranks 1-based).
///
/// `lists` holds per-list vectors of dedup indices (the py layer's id
/// table); `n_docs` is the number of distinct ids. Returns one
/// `(index, score)` pair per distinct id, sorted by score descending,
/// ties broken by earliest first appearance across the lists in caller
/// order (tracked here in the same walk that scores), so the tie-break
/// is the paper-shape contract no matter how the caller numbered its
/// indices. `k` is trusted here as already-validated (`k >= 1`): the
/// pyo3 layer's job, matching this crate's usual split of caller-facing
/// validation from the trusted core.
pub fn rank_fuse(lists: &[Vec<u32>], k: u64, n_docs: usize) -> Vec<(u32, f64)> {
    // The sweep needs one slot per index that can receive a vote: the
    // data's own max index bounds that, so a caller-supplied n_docs
    // looser than the data (a sparse numbering with a huge table)
    // costs no allocation: every index above the max is unvoted and
    // filtered below anyway.
    let max_index = lists
        .iter()
        .flat_map(|list| list.iter())
        .copied()
        .max()
        .map_or(0, |m| m as usize + 1);
    let n_docs = n_docs.min(max_index);
    let mut scores = vec![0.0f64; n_docs];
    let mut first_seen = vec![u32::MAX; n_docs];
    let mut seen_count = 0u32;
    for list in lists {
        for (position, &doc) in list.iter().enumerate() {
            let doc = doc as usize;
            // Ranks are 1-based (the paper's own convention): the first
            // entry of a list contributes 1/(k + 1).
            scores[doc] += 1.0 / (k as f64 + position as f64 + 1.0);
            // First appearance, in the same pass: lists in caller
            // order, positions within a list, so the first walk visit
            // IS the earliest appearance.
            if first_seen[doc] == u32::MAX {
                first_seen[doc] = seen_count;
                seen_count += 1;
            }
        }
    }
    let mut ranked: Vec<(u32, f64)> = scores
        .into_iter()
        .enumerate()
        // A zero score means zero votes: an index the caller's dedup
        // table sized but no list ever ranked (unreachable from the
        // binding, which only assigns indices on first sight), not a
        // document to emit. Every voted score is positive.
        .filter(|(_, score)| *score > 0.0)
        .map(|(i, score)| (i as u32, score))
        .collect();
    ranked.sort_by(|(a_idx, a_score), (b_idx, b_score)| {
        b_score
            .partial_cmp(a_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(first_seen[*a_idx as usize].cmp(&first_seen[*b_idx as usize]))
    });
    ranked
}

/// nDCG@k (Järvelin & Kekäläinen, TOIS 2002) over per-position gains:
/// `ranked_gains[i]` is the gain of the i-th ranked result; `ideal_pool`
/// is the complete multiset of judged relevance values (binary: one 1.0
/// per relevant id; graded: every judged gain), from which the ideal
/// ranking (the pool sorted descending, top-`k`) is derived here.
/// `k` is trusted as already-validated (`k >= 1`), and the caller may
/// have clamped it to `ranked_gains.len()`; the discount is
/// `1/log2(i + 2)` over 0-based positions (the paper's `log2(i + 1)`
/// over 1-based ranks: rank 1 undiscounted).
///
/// Returns `DCG@k / IDCG@k` in `[0, 1]`; a zero ideal (nothing judged
/// relevant) is the documented `0.0` answer, not a division by zero.
/// The ratio is computed with overflow-aware, saturating logic (see the
/// module docs' "saturating-ratio policy"): legal finite gains can
/// overflow both sums to `+inf`, where a bare `inf / inf` division
/// would answer NaN and break the pinned interval; a non-finite sum
/// saturates instead (`1.0` when `dcg >= idcg`, else `0.0`), and a
/// finite ratio clamps to `[0, 1]`.
pub fn ndcg_at_k(ranked_gains: &[f64], mut ideal_pool: Vec<f64>, k: usize) -> f64 {
    let dcg: f64 = ranked_gains
        .iter()
        .take(k)
        .enumerate()
        .map(|(position, gain)| gain / (position as f64 + 2.0).log2())
        .sum();
    ideal_pool.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg: f64 = ideal_pool
        .iter()
        .take(k)
        .enumerate()
        .map(|(position, gain)| gain / (position as f64 + 2.0).log2())
        .sum();
    if idcg == 0.0 {
        return 0.0;
    }
    // The sums are non-negative-term sums of binding-validated finite
    // gains, so a non-finite sum is exactly +inf (never NaN): saturate
    // rather than divide inf by inf. An overflowed DCG can at most
    // match the overflowed ideal (the pool is a superset of the ranked
    // gains under the same discount schedule), so the tie answers 1.0.
    if !dcg.is_finite() || !idcg.is_finite() {
        return if dcg >= idcg { 1.0 } else { 0.0 };
    }
    (dcg / idcg).clamp(0.0, 1.0)
}

/// MRR: the reciprocal rank of the first relevant result, `0.0` when
/// every position misses (and for an empty ranking, the same
/// well-defined zero).
pub fn mrr(ranked_relevant: &[bool]) -> f64 {
    match ranked_relevant.iter().position(|hit| *hit) {
        Some(position) => 1.0 / (position as f64 + 1.0),
        None => 0.0,
    }
}

/// recall@k: the fraction of the relevant set found in the top `k`
/// positions. The empty-relevant-set case (n_relevant == 0) is the
/// documented `0.0` answer, guarded HERE: no relevant document exists
/// to find, and the 0/0 spelling of that answer is not one a caller
/// should ever see. A `k` past the ranking's length simply uses every
/// available position.
pub fn recall_at_k(ranked_relevant: &[bool], n_relevant: usize, k: usize) -> f64 {
    if n_relevant == 0 {
        return 0.0;
    }
    let hits = ranked_relevant.iter().take(k).filter(|hit| **hit).count();
    hits as f64 / n_relevant as f64
}

/// precision@k: the fraction of the top `k` positions that are relevant,
/// with the denominator `min(k, ranking length)`, trec_eval's own
/// convention for a run shorter than `k` (a system that returned fewer
/// results is not punished for positions it never filled). `k` is
/// trusted as already-validated (`k >= 1`); an empty ranking answers
/// `0.0` (the empty-denominator convention).
pub fn precision_at_k(ranked_relevant: &[bool], k: usize) -> f64 {
    let width = k.min(ranked_relevant.len());
    if width == 0 {
        return 0.0;
    }
    let hits = ranked_relevant
        .iter()
        .take(width)
        .filter(|hit| **hit)
        .count();
    hits as f64 / width as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    // --- rank_fuse ------------------------------------------------------

    #[test]
    fn hand_computed_two_list_fusion() {
        // Two lists that disagree: d0 = 1/61 (L0 rank 1) + 1/62 (L1
        // rank 2); d1 = 1/62 (L0 rank 2) + 1/61 (L1 rank 1). The scores
        // tie exactly; first appearance breaks the tie (d0 first).
        let lists = [vec![0u32, 1u32], vec![1u32, 0u32]];
        let fused = rank_fuse(&lists, 60, 2);
        assert_eq!(fused[0].0, 0);
        assert!(close(fused[0].1, fused[1].1));
        assert_eq!(fused[0].1, fused[1].1);
        assert!(close(fused[0].1, 1.0 / 61.0 + 1.0 / 62.0));
    }

    #[test]
    fn consensus_document_outranks_single_list_top() {
        // The RRF thesis: a document ranked 2nd by two lists beats a
        // document ranked 1st by one (and absent from the other).
        let lists = [vec![0u32, 1u32], vec![2u32, 1u32]];
        let fused = rank_fuse(&lists, 60, 3);
        // d1: 1/62 + 1/62 = 0.03225...; d0: 1/61; d2: 1/61.
        assert_eq!(fused[0].0, 1);
        // The two single-vote documents tie at 1/61; d0 appeared first.
        assert!(close(fused[1].1, fused[2].1));
        assert_eq!(fused[1].0, 0);
        assert_eq!(fused[2].0, 2);
        assert!(close(fused[1].1, 1.0 / 61.0));
    }

    #[test]
    fn ranks_start_at_one_not_zero() {
        // A single list of three: the top document scores exactly 1/61
        // at k=60 (not 1/60), the third 1/63.
        let lists = [vec![0u32, 1u32, 2u32]];
        let fused = rank_fuse(&lists, 60, 3);
        assert!(close(fused[0].1, 1.0 / 61.0));
        assert!(close(fused[2].1, 1.0 / 63.0));
    }

    #[test]
    fn smaller_k_lowers_the_top_rank_bonus_without_reordering_voting_patterns() {
        // k tunes how sharply the top of a list is weighted; ranks, not
        // scores, are consumed either way.
        let lists = [vec![0u32, 1u32], vec![1u32]];
        let fused_k1 = rank_fuse(&lists, 1, 2);
        // d0: 1/2; d1: 1/3 + 1/2 = 0.8333...; d1 wins (two votes beat one).
        assert_eq!(fused_k1[0].0, 1);
        assert!(close(fused_k1[0].1, 1.0 / 3.0 + 1.0 / 2.0));
        assert!(close(fused_k1[1].1, 1.0 / 2.0));
    }

    #[test]
    fn ties_break_by_earliest_first_appearance_not_id_value() {
        // Both documents score exactly 1/61 (rank 1 of one list each);
        // the HIGHER id appeared first and must outrank the lower.
        let lists = [vec![9u32], vec![3u32]];
        let fused = rank_fuse(&lists, 60, 10);
        assert!(close(fused[0].1, fused[1].1));
        assert_eq!(fused[0].0, 9);
        assert_eq!(fused[1].0, 3);
    }

    #[test]
    fn single_list_preserves_its_order() {
        let lists = [vec![4u32, 0u32, 2u32]];
        let fused = rank_fuse(&lists, 60, 5);
        assert_eq!(
            fused.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![4, 0, 2]
        );
    }

    #[test]
    fn empty_inner_lists_contribute_no_votes() {
        let lists = [vec![1u32], Vec::new(), vec![0u32]];
        let fused = rank_fuse(&lists, 60, 2);
        assert_eq!(
            fused.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![1, 0]
        );
    }

    #[test]
    fn duplicate_positions_in_one_list_would_vote_twice_the_core_assumes_dedup() {
        // The core is trusted with deduped lists (the binding's job);
        // this pins what the core does with its own contract: a repeated
        // index is two votes from one list, exactly as written. The
        // binding's dedup-first pass is what makes this unreachable from
        // Python (pinned py-side, not here).
        let lists = [vec![0u32, 0u32]];
        let fused = rank_fuse(&lists, 60, 1);
        assert!(close(fused[0].1, 1.0 / 61.0 + 1.0 / 62.0));
    }

    #[test]
    fn scores_are_finite_and_positive() {
        let lists = [vec![0u32, 1, 2], vec![2, 1], Vec::new(), vec![1]];
        for (_, score) in rank_fuse(&lists, 1, 3) {
            assert!(score.is_finite() && score > 0.0);
        }
    }

    // --- ndcg_at_k ------------------------------------------------------

    #[test]
    fn perfect_ranking_scores_one() {
        let gains = [1.0, 1.0, 0.0];
        assert!(close(ndcg_at_k(&gains, vec![1.0, 1.0], 3), 1.0));
    }

    #[test]
    fn hand_computed_binary_ndcg() {
        // ranked gains [1, 0, 1]: DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5.
        // ideal [1, 1]: IDCG = 1 + 1/log2(3).
        let dcg = 1.0 + 0.5;
        let idcg = 1.0 + 1.0 / 3.0_f64.log2();
        assert!(close(
            ndcg_at_k(&[1.0, 0.0, 1.0], vec![1.0, 1.0], 3),
            dcg / idcg
        ));
    }

    #[test]
    fn log2_discount_rank_one_is_undiscounted() {
        // A single hit at rank 1: DCG = 1/log2(2) = 1; IDCG = 1 -> 1.0.
        assert!(close(ndcg_at_k(&[1.0], vec![1.0], 1), 1.0));
        // The same hit at rank 2 (one miss above it): 1/log2(3), the
        // rank-1 undiscounted ideal below it.
        let score = ndcg_at_k(&[0.0, 1.0], vec![1.0], 2);
        assert!(close(score, 1.0 / 3.0_f64.log2()));
    }

    #[test]
    fn k_truncates_both_sides() {
        // Two relevant documents, only the first ranked: at k=1 the
        // second cannot be reached, so the score is 1.0 (the ideal at
        // k=1 is also a single hit).
        assert!(close(ndcg_at_k(&[1.0, 0.0], vec![1.0, 1.0], 1), 1.0));
    }

    #[test]
    fn graded_gains_order_the_ideal() {
        // ranked gains [2, 3]: DCG = 2/log2(2) + 3/log2(3).
        // ideal [3, 2]: IDCG = 3/log2(2) + 2/log2(3); the higher grade
        // discounted at the better rank, so score < 1.
        let score = ndcg_at_k(&[2.0, 3.0], vec![2.0, 3.0], 2);
        let dcg = 2.0 / 1.0 + 3.0 / 3.0_f64.log2();
        let idcg = 3.0 / 1.0 + 2.0 / 3.0_f64.log2();
        assert!(close(score, dcg / idcg));
        assert!(score < 1.0);
    }

    #[test]
    fn zero_ideal_answers_zero_not_nan() {
        // Nothing judged relevant: the documented 0.0, never 0/0.
        assert!(close(ndcg_at_k(&[0.0, 0.0], Vec::new(), 2), 0.0));
    }

    #[test]
    fn nothing_relevant_anywhere_scores_zero() {
        assert!(close(ndcg_at_k(&[0.0, 0.0, 0.0], vec![1.0], 3), 0.0));
    }

    // --- ndcg_at_k: the saturating-ratio policy (overflowed sums) -------

    #[test]
    fn three_huge_gains_saturate_at_one_not_nan() {
        // 3 × 1e308: DCG and IDCG each sum past f64's ceiling to +inf;
        // inf/inf is NaN in IEEE; the core saturates instead, and the
        // perfect ranking (ranked gains == ideal pool, same discounts)
        // pins the equality → 1.0 exactly.
        assert_eq!(ndcg_at_k(&[1e308, 1e308, 1e308], vec![1e308; 3], 3), 1.0);
    }

    #[test]
    fn two_huge_gains_saturate_at_one_not_nan() {
        // 2 × 1.7e308 overflows each sum the same way.
        assert_eq!(ndcg_at_k(&[1.7e308, 1.7e308], vec![1.7e308; 2], 2), 1.0);
    }

    #[test]
    fn finite_dcg_under_an_infinite_ideal_saturates_at_zero() {
        // A ranked DCG that stays finite under an ideal that overflows:
        // the true ratio is ~1e-308-scale; the saturating answer is 0.0,
        // inside the pinned interval either way.
        assert_eq!(ndcg_at_k(&[1e308], vec![1e308; 3], 3), 0.0);
    }

    #[test]
    fn mixed_huge_and_small_gains_saturate_monotonically() {
        // Huge and small gains together, both sums overflowing: the
        // equality shape (ranked gains == ideal pool) still pins 1.0
        // exactly, and the documented saturation can over-report the
        // imperfect overflowed shape at exactly 1.0, inside the pinned
        // interval either way.
        assert_eq!(
            ndcg_at_k(
                &[1e308, 1e-300, 1e308, 1e308],
                vec![1e308, 1e308, 1e308, 1e-300],
                4
            ),
            1.0
        );
        let imperfect = ndcg_at_k(
            &[1e308, 1e-300, 1e308, 1e308],
            vec![1e308, 1e308, 1e308, 1e308],
            4,
        );
        assert!((0.0..=1.0).contains(&imperfect));
        // Where the mixed sums stay finite, the ratio is the ordinary
        // one: 1e308 at rank 1 under an ideal of two 1e308s.
        assert!(close(
            ndcg_at_k(&[1e308, 1e-300], vec![1e308, 1e308], 2),
            1e308 / (1e308 + 1e308 / 3.0_f64.log2())
        ));
    }

    #[test]
    fn moderate_ratios_are_untouched_by_the_clamp() {
        // The clamp only guards the extremes: ordinary scores match the
        // hand-computed value exactly as before the policy landed.
        let dcg = 1.0 + 0.5;
        let idcg = 1.0 + 1.0 / 3.0_f64.log2();
        assert!(close(
            ndcg_at_k(&[1.0, 0.0, 1.0], vec![1.0, 1.0], 3),
            dcg / idcg
        ));
    }

    // --- mrr ------------------------------------------------------------

    #[test]
    fn first_position_hit_is_one() {
        assert!(close(mrr(&[true, false]), 1.0));
    }

    #[test]
    fn reciprocal_of_first_relevant_position() {
        assert!(close(mrr(&[false, false, true]), 1.0 / 3.0));
    }

    #[test]
    fn no_hit_and_empty_ranking_answer_zero() {
        assert!(close(mrr(&[false, false]), 0.0));
        assert!(close(mrr(&[]), 0.0));
    }

    #[test]
    fn only_the_first_hit_counts() {
        // MRR is first-hit: later hits never move the score.
        assert!(close(mrr(&[false, true, true, true]), 1.0 / 2.0));
    }

    // --- recall_at_k ------------------------------------------------------

    #[test]
    fn recall_counts_hits_over_relevant_size() {
        // 2 of 3 relevant found in the top 4.
        assert!(close(
            recall_at_k(&[true, false, true, false], 3, 4),
            2.0 / 3.0
        ));
    }

    #[test]
    fn recall_k_past_ranking_length_uses_available_positions() {
        assert!(close(recall_at_k(&[true, false], 2, 10), 0.5));
    }

    #[test]
    fn recall_empty_ranking_is_zero() {
        assert!(close(recall_at_k(&[], 2, 3), 0.0));
    }

    #[test]
    fn recall_empty_relevant_is_zero_not_nan() {
        // The core owns the documented empty-relevant answer (0/0 is NaN
        // without the guard; the sparse-numbering table shape is covered
        // by the fuzz target).
        assert!(close(recall_at_k(&[true, true], 0, 2), 0.0));
        assert!(close(recall_at_k(&[], 0, 2), 0.0));
    }

    // --- precision_at_k ---------------------------------------------------

    #[test]
    fn precision_divides_by_min_k_and_length() {
        assert!(close(precision_at_k(&[true, false, true], 3), 2.0 / 3.0));
        // k past the length: the denominator is the length (trec_eval's
        // convention).
        assert!(close(precision_at_k(&[true, false], 10), 0.5));
    }

    #[test]
    fn precision_empty_ranking_is_zero_not_nan() {
        assert!(close(precision_at_k(&[], 3), 0.0));
    }

    #[test]
    fn precision_one_hit_in_one_position_is_one() {
        assert!(close(precision_at_k(&[true], 1), 1.0));
    }

    #[test]
    fn hand_computed_three_list_rrf_with_k_five() {
        // k=5, three lists over four documents: a fully hand-computed
        // vector (the py-side oracle repeats it with different ids):
        // d0: 1/6 + 1/8 = 0.291666...  (L0 rank 1, L1 rank 3)
        // d1: 1/7 + 1/6 = 0.309523...  (L0 rank 2, L2 rank 1)
        // d2: 1/8 + 1/6 + 1/7 = 0.434523... (L0 rank 3, L1 rank 1, L2 rank 2)
        // d3: 1/7 = 0.142857...        (L1 rank 2)
        let lists = [vec![0u32, 1, 2], vec![2u32, 3, 0], vec![1u32, 2]];
        let fused = rank_fuse(&lists, 5, 4);
        let by_id: std::collections::HashMap<u32, f64> = fused.iter().copied().collect();
        assert!(close(by_id[&0], 1.0 / 6.0 + 1.0 / 8.0));
        assert!(close(by_id[&1], 1.0 / 7.0 + 1.0 / 6.0));
        assert!(close(by_id[&2], 1.0 / 8.0 + 1.0 / 6.0 + 1.0 / 7.0));
        assert!(close(by_id[&3], 1.0 / 7.0));
        assert_eq!(
            fused.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![2, 1, 0, 3]
        );
    }
}
