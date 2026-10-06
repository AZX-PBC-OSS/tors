//! Score-based fusion: the pure-Rust core of `tors.score_fuse`, the
//! score-space sibling of [`rank_fusion_impl`]'s rank-space `rank_fuse`.
//!
//! # What this is: score-space arithmetic over an already-materialized
//! candidate universe
//!
//! The split is `rank_fusion_impl`'s own, drawn one level earlier in the
//! caller's data: the pyo3 layer materializes each list as a vector of
//! `(dedup index, raw score)` pairs (one index per distinct id in
//! first-appearance order, duplicates within a list folded to their
//! first occurrence's score; the ids themselves never cross into this
//! module), while the normalization, weighting, accumulation, and sort
//! here are plain Rust over plain numbers and run under one
//! `py.detach`. Python-object hashing is interpreter work and stays
//! under the GIL in the binding layer, exactly the `rank_fuse` and
//! `bm25_impl` split.
//!
//! # The three methods
//!
//! With `L` scored lists, `w_i` list `i`'s weight (default 1.0), and
//! `norm_i(d)` list `i`'s normalized score for document `d`:
//!
//! - **CombMNZ** (default), Fox & Shaw, "Combination of Multiple
//!   Searches", TREC-2, 1994 -- `score(d) = lists(d) × Σ_i w_i ×
//!   norm_i(d)`, where `lists(d)` counts the lists containing `d` (the
//!   MNZ multiplier is CombSUM × the consensus count; each occurrence
//!   is a vote of confidence in the score itself, not just its rank).
//!   This is the best performer of the score-based family in Cormack,
//!   Clarke & Buüttcher's SIGIR 2009 comparison (the RRF paper's own
//!   baseline set, https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf),
//!   which is why it is the default here.
//! - **Borda**, the rank-based count -- `score(d) = Σ_i w_i × (n_i -
//!   rank_i(d)) / n_i`, with `n_i` the list's distinct-document count
//!   and `rank_i(d)` the 1-based position in the DEDUPLICATED list (the
//!   fold moves later ids up, `rank_fuse`'s contract). The votes are
//!   RANK-based deliberately: Borda counts are defined over
//!   positions, not scores -- each list elects its top document with
//!   `n_i - 1` points and its last with 0, and the
//!   `(n - rank)/n` spelling only rescales that count to `[0, 1)` so a
//!   weight means the same thing over a 3-doc list and a 300-doc one.
//!   Normalizing the SCORES here (min-max, the CombMNZ spelling) would
//!   not be Borda: it would smuggle score magnitudes back into a method
//!   whose entire point is that only the ordering votes.
//! - **Linear**, Elasticsearch's linear-retriever pattern -- `score(d)
//!   = Σ_i w_i × norm_i(d)` (no MNZ multiplier):
//!   https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/linear-retriever,
//!   which sums min-max-normalized scores across retrievers with the
//!   same per-retriever `weight` shape weighted RRF carries. tors
//!   follows that shape: min-max normalization per list, weights
//!   multiplying each list's contribution, no consensus boost.
//!
//! # The normalization conventions (stated once, pinned in tests)
//!
//! `norm_i` is min-max over the list's OWN scores -- the folded
//! entries' scores, the votes the list actually casts (a within-list
//! duplicate's folded-away score does not shape the range, the
//! dedup-first contract applied to the normalizer too). The
//! conventions at the edges:
//!
//! - **Zero-range list**: every score equal (`max == min`) normalizes
//!   to `0.5` -- the midpoint, not 0.0 or 1.0: the list carries order
//!   information only ("all these docs tie"), no magnitude information,
//!   and the midpoint is the neutral value a linear sum can carry
//!   without inventing a winner or a loser. A single-entry list is the
//!   zero-range shape (one score spans an empty range).
//! - **Negative scores are legal**: min-max maps ANY finite range to
//!   `[0, 1]` (`(s - min)/(max - min)` shifts and rescales; the signs
//!   of the raw scores wash out), so a cosine-similarity list (-1..1)
//!   and a BM25 list (0..40) normalize to the same interval and the
//!   raw domains never meet. The rejected domain is only non-finite
//!   scores: NaN poisons every comparison it touches and an infinite
//!   min or max makes `max - min` an ill-defined `inf - inf`/`inf`
//!   range -- both are `ValueError`s at the binding, the `gains`
//!   finite-domain discipline.
//! - **Range overflow (the saturating-norm policy)**: legal finite
//!   scores can span so far the range itself overflows (`min =
//!   -1.7e308, max = +1.7e308` gives `max - min = +inf`), and a
//!   numerator that overflows too (`s - min` at the top of the scale)
//!   would answer IEEE `inf / inf = NaN`, breaking the sort's total
//!   order the same way an overflowing DCG would break nDCG's pinned
//!   interval. The core therefore saturates: a numerator that overflowed
//!   to `+inf` against an infinite range answers exactly `1.0` (the top
//!   of the scale; `max` itself is the first such numerator), while a
//!   finite numerator over an infinite range divides to `0.0`-scale
//!   values. The policy is monotone and NaN-free; the saturation can
//!   over-report near-top scores as exactly `1.0`, the same documented
//!   one-sided cost `ndcg_at_k`'s saturating ratio carries, and it
//!   applies only when the scores sit within ~16 orders of magnitude
//!   of f64's ceiling on BOTH signs.
//!
//! Accumulation is all-nonneg term sums (weights strictly positive,
//! norms in `[0, 1]`), so a sum can overflow to `+inf` but can never
//! produce a NaN: the sort's `partial_cmp` stays total on every legal
//! input. A legal denormal weight (`5e-324`) times a small norm can
//! underflow a doc's every term to exactly `0.0` (Borda's last-place
//! vote is `0.0` by construction too); emission is
//! vote-existence -- every distinct id any list carried appears, a
//! `0.0`-score pair included, ordered last, ties by first appearance
//! -- the same underflow-emission policy weighted `rank_fuse` pins.

/// The three method spellings, for the binding's `ValueError` (and the
/// fuzz target's steering).
pub const SCORE_FUSION_METHODS: [&str; 3] = ["combmnz", "borda", "linear"];

/// The fusion method: parsed at the binding (a bad name is the
/// caller's `ValueError`), consumed as the enum here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionMethod {
    /// CombMNZ: CombSUM × the containing-list count (the default).
    CombMnz,
    /// Borda: rank-based votes `(n - rank)/n`.
    Borda,
    /// Linear: the plain weighted sum of min-max norms (Elastic's
    /// linear retriever).
    Linear,
}

/// The `method=` parameter's three spellings (the
/// `parse_dedup_method` shape: the accepted set named in the error).
pub fn parse_fusion_method(name: &str) -> Result<FusionMethod, String> {
    match name {
        "combmnz" => Ok(FusionMethod::CombMnz),
        "borda" => Ok(FusionMethod::Borda),
        "linear" => Ok(FusionMethod::Linear),
        _ => Err(format!(
            "unrecognized method {name:?}; valid choices are: {}",
            SCORE_FUSION_METHODS.join(", ")
        )),
    }
}

/// Score-fuses the deduplicated `(index, score)` lists per `method`
/// (see the module docs for the three formulas and their sources).
///
/// `lists` holds per-list vectors of `(dedup index, raw score)` pairs
/// (the py layer's id table; duplicates within a list already folded to
/// their first occurrence's score). `weights` is `None` (the all-1.0
/// default) or one positive finite weight per list; a weight slice
/// shorter than `lists` is tolerated at the core (the missing tails
/// answer 1.0, the `rank_fuse` core's own padding contract, which the
/// fuzz target exercises) but the binding validates the exact length
/// before handing the slice over. `k` truncates the output to the top
/// `k` pairs (`None`: every distinct id).
///
/// The table is sized by the data's own max index (every slot a vote
/// can land in), and the first_seen filter below drops any slot no
/// list ever visited (a sparse numbering's gaps), so no separate
/// distinct-id count is consumed -- the binding's dense numbering
/// makes one redundant.
///
/// Returns one `(index, score)` pair per distinct id any list carried,
/// sorted by fused score descending, ties broken by earliest first
/// appearance across the lists in caller order (tracked in the same
/// walk that scores, the `rank_fuse` contract extended to score
/// space). Scores are finite-or-`+inf`, never NaN; a `0.0` score
/// (Borda's last place, an underflowed denormal weight) still emits.
/// `method` and `k` are trusted here as already-validated: the pyo3
/// layer's job, matching this crate's usual split of caller-facing
/// validation from the trusted core.
pub fn score_fuse(
    lists: &[Vec<(u32, f64)>],
    method: FusionMethod,
    weights: Option<&[f64]>,
    k: Option<usize>,
) -> Vec<(u32, f64)> {
    let max_index = lists
        .iter()
        .flat_map(|list| list.iter().map(|(i, _)| *i))
        .max()
        .map_or(0, |m| m as usize + 1);
    let mut scores = vec![0.0f64; max_index];
    let mut counts = vec![0u32; max_index];
    let mut first_seen = vec![u32::MAX; max_index];
    let mut seen_count = 0u32;
    for (list_idx, list) in lists.iter().enumerate() {
        if list.is_empty() {
            continue; // the "this retriever returned nothing" shape
        }
        // The default path's weight is exactly 1.0, so the multiply
        // below is the identical expression the unweighted spelling
        // has always executed (weights=None IS all-1.0, byte for byte).
        let weight = weights
            .and_then(|w| w.get(list_idx).copied())
            .unwrap_or(1.0);
        match method {
            FusionMethod::Borda => {
                // Rank-based votes: 1-based rank of position is
                // position + 1, so the vote is (n - rank)/n. The
                // numerator is an exact small integer either way; the
                // single division is the one rounding step.
                let n = list.len() as f64;
                for (position, (doc, _)) in list.iter().enumerate() {
                    let doc = *doc as usize;
                    let vote = (n - position as f64 - 1.0) / n;
                    scores[doc] += weight * vote;
                    counts[doc] += 1;
                    if first_seen[doc] == u32::MAX {
                        first_seen[doc] = seen_count;
                        seen_count += 1;
                    }
                }
            }
            FusionMethod::CombMnz | FusionMethod::Linear => {
                // Min-max over the list's OWN (folded) scores.
                let mut min = f64::INFINITY;
                let mut max = f64::NEG_INFINITY;
                for (_, score) in list {
                    min = min.min(*score);
                    max = max.max(*score);
                }
                let range = max - min;
                for (doc, score) in list.iter() {
                    let doc = *doc as usize;
                    let norm = if range == 0.0 {
                        // The zero-range convention: every score equal
                        // (a single-entry list included) is the
                        // neutral midpoint 0.5.
                        0.5
                    } else {
                        let num = score - min;
                        // The saturating-norm policy: a numerator that
                        // overflowed against an infinite range is at
                        // the top of the scale (max itself first);
                        // inf/inf would be NaN, 1.0 is the only
                        // in-scale answer consistent with the ordering
                        // the finite arithmetic reports.
                        if num.is_infinite() { 1.0 } else { num / range }
                    };
                    scores[doc] += weight * norm;
                    counts[doc] += 1;
                    if first_seen[doc] == u32::MAX {
                        first_seen[doc] = seen_count;
                        seen_count += 1;
                    }
                }
            }
        }
    }
    if method == FusionMethod::CombMnz {
        // The MNZ multiplier: each list's vote of confidence in the
        // doc, applied to the fused CombSUM (per-list counts, one
        // multiply per doc).
        for (doc, score) in scores.iter_mut().enumerate() {
            *score *= counts[doc] as f64;
        }
    }
    let mut ranked: Vec<(u32, f64)> = scores
        .into_iter()
        .enumerate()
        // Emission is the VOTE-EXISTENCE signal (first_seen assigned in
        // the scoring walk), never score positivity: a 0.0 fused score
        // (Borda's last place, an underflowed denormal weight) still
        // emits, ordered last. The filter's remaining job is the one it
        // always has: a slot the data's max index sized but no list
        // ever carried (a sparse numbering's gap; unreachable from the
        // binding, which assigns indices on first sight) is not a
        // document to emit.
        .filter(|(i, _)| first_seen[*i] != u32::MAX)
        .map(|(i, score)| (i as u32, score))
        .collect();
    ranked.sort_by(|(a_idx, a_score), (b_idx, b_score)| {
        b_score
            .partial_cmp(a_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(first_seen[*a_idx as usize].cmp(&first_seen[*b_idx as usize]))
    });
    if let Some(k) = k {
        ranked.truncate(k);
    }
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    fn ids(fused: &[(u32, f64)]) -> Vec<u32> {
        fused.iter().map(|(i, _)| *i).collect()
    }

    // --- combmnz --------------------------------------------------------

    #[test]
    fn hand_computed_combmnz_two_lists() {
        // L0: d0=1.0, d1=0.5 (norms 1.0, 0.0); L1: d1=1.0, d0=0.0 (norms
        // 1.0, 0.0). Both appear in BOTH lists: CombSUM 1.0 each, MNZ ×2
        // = 2.0 each; d0 appeared first and wins the tie.
        let lists = [
            vec![(0u32, 1.0), (1u32, 0.5)],
            vec![(1u32, 1.0), (0u32, 0.0)],
        ];
        let fused = score_fuse(&lists, FusionMethod::CombMnz, None, None);
        assert_eq!(fused[0].0, 0);
        assert_eq!(fused[1].0, 1);
        assert!(close(fused[0].1, 2.0));
        assert!(close(fused[1].1, 2.0));
    }

    #[test]
    fn combmnz_multiplies_by_the_containing_list_count() {
        // d1 in two lists (norm 1.0 each): (1 + 1) × 2 = 4; d0 in one
        // (norm 1.0): 1 × 1 = 1. The consensus multiplier is the MNZ
        // point, visible exactly here.
        let lists = [
            vec![(0u32, 1.0), (2u32, 0.0)],
            vec![(1u32, 1.0), (2u32, 0.0)],
            vec![(1u32, 1.0), (2u32, 0.0)],
        ];
        let fused = score_fuse(&lists, FusionMethod::CombMnz, None, None);
        assert!(close(fused[0].1, 4.0));
        assert!(close(fused[1].1, 1.0));
    }

    #[test]
    fn combmnz_norms_are_min_max_per_list_over_its_own_scores() {
        // L0's range is [10, 20] (not the universe's [0, 20]): its top
        // scores norm 1.0, its bottom 0.0; L1's own range [0, 0.5]. A
        // list's scores are normalized against the list, never the
        // global extremes.
        let lists = [
            vec![(0u32, 20.0), (1u32, 10.0)],
            vec![(2u32, 0.5), (3u32, 0.0)],
        ];
        let fused = score_fuse(&lists, FusionMethod::CombMnz, None, None);
        let by_id: std::collections::HashMap<u32, f64> = fused.iter().copied().collect();
        assert!(close(by_id[&0], 1.0));
        assert!(close(by_id[&1], 0.0));
        assert!(close(by_id[&2], 1.0));
        assert!(close(by_id[&3], 0.0));
    }

    #[test]
    fn combmnz_top_of_every_list_scores_the_list_count_squared() {
        // Max possible: norm 1.0 in every one of L lists (each list
        // needs a second, bottom entry to span the range) × count L
        // = L².
        let lists = [
            vec![(0u32, 9.0), (1u32, 0.0)],
            vec![(0u32, 7.0), (2u32, 0.0)],
            vec![(0u32, 1.0), (3u32, 0.0)],
        ];
        let fused = score_fuse(&lists, FusionMethod::CombMnz, None, None);
        assert!(close(fused[0].1, 9.0)); // 3 lists × 3 × 1.0
    }

    // --- borda ----------------------------------------------------------

    #[test]
    fn borda_votes_are_rank_based_n_minus_rank_over_n() {
        // One list of three: votes (3-1)/3, (3-2)/3, (3-3)/3 = 2/3,
        // 1/3, 0. The scores are irrelevant (10.0 and 0.001 same order
        // of magnitude separation as 1.0 and 0.0): only positions vote.
        let lists = [vec![(0u32, 10.0), (1u32, 5.0), (2u32, 0.001)]];
        let fused = score_fuse(&lists, FusionMethod::Borda, None, None);
        assert!(close(fused[0].1, 2.0 / 3.0));
        assert!(close(fused[1].1, 1.0 / 3.0));
        assert!(close(fused[2].1, 0.0));
    }

    #[test]
    fn borda_ignores_score_magnitudes_that_min_max_would_amplify() {
        // The Borda/min-max divergence: in L1 the scores are 1000.0 vs
        // 999.0 (a hair apart, min-max would norm 1.0 vs 0.0);
        // Borda votes 2/3 vs 1/3 -- the middle position, not the range.
        // d0/d1/d3 tie at 2/3; first appearance decides (d0 walked
        // first).
        let lists = [
            vec![(0u32, 2.0), (1u32, 1.0), (2u32, 0.0)],
            vec![(3u32, 1000.0), (1u32, 999.0), (2u32, 0.0)],
        ];
        let fused = score_fuse(&lists, FusionMethod::Borda, None, None);
        let by_id: std::collections::HashMap<u32, f64> = fused.iter().copied().collect();
        assert!(close(by_id[&0], 2.0 / 3.0));
        assert!(close(by_id[&3], 2.0 / 3.0));
        assert!(close(by_id[&1], 1.0 / 3.0 + 1.0 / 3.0));
        assert!(close(by_id[&2], 0.0));
        assert_eq!(ids(&fused), vec![0, 1, 3, 2]);
    }

    #[test]
    fn borda_folds_duplicates_before_ranking() {
        // The fold is the BINDING's pass (the core assumes dedup, the
        // rank_fuse contract): handed the folded list, 'a' ranks 1st
        // of TWO distinct entries, voting (2-1)/2, 'b' (2-2)/2.
        let lists = [vec![(0u32, 5.0), (1u32, 1.0)]];
        let fused = score_fuse(&lists, FusionMethod::Borda, None, None);
        assert!(close(fused[0].1, 1.0 / 2.0)); // (2 - 1)/2
        assert!(close(fused[1].1, 0.0)); // (2 - 2)/2
    }

    #[test]
    fn borda_raw_duplicates_vote_twice_the_core_assumes_dedup() {
        // What the core does with its own contract: a repeated id is
        // two POSITIONS (n = 3), voting (3-1)/3 + (3-2)/3; the
        // binding's dedup-first pass is what makes this unreachable
        // from Python (pinned py-side, not here).
        let lists = [vec![(0u32, 5.0), (0u32, 5.0), (1u32, 1.0)]];
        let fused = score_fuse(&lists, FusionMethod::Borda, None, None);
        assert!(close(fused[0].1, 2.0 / 3.0 + 1.0 / 3.0));
        assert!(close(fused[1].1, 0.0));
    }

    #[test]
    fn borda_single_element_list_votes_zero_and_still_emits() {
        // (1 - 1)/1 = 0: the list's only document casts its last-place
        // vote of zero; emission is vote existence, not positivity. d0
        // ties d2 at 0.0 and d0's first appearance (list 0) wins.
        let lists = [vec![(0u32, 42.0)], vec![(1u32, 1.0), (2u32, 0.0)]];
        let fused = score_fuse(&lists, FusionMethod::Borda, None, None);
        assert!(close(fused[0].1, 1.0 / 2.0)); // (2-1)/2
        assert_eq!(ids(&fused), vec![1, 0, 2]);
        assert_eq!(fused[1], (0, 0.0)); // present, ordered last, at 0.0
    }

    // --- linear ---------------------------------------------------------

    #[test]
    fn linear_is_the_plain_weighted_sum_no_mnz_multiplier() {
        // The same shape that gave combmnz (1+1)×2 = 4 for d1: linear
        // answers the bare CombSUM 1 + 1 = 2, d0's single 1.0 loses.
        let lists = [
            vec![(0u32, 1.0), (2u32, 0.0)],
            vec![(1u32, 1.0), (2u32, 0.0)],
            vec![(1u32, 1.0), (2u32, 0.0)],
        ];
        let fused = score_fuse(&lists, FusionMethod::Linear, None, None);
        let by_id: std::collections::HashMap<u32, f64> = fused.iter().copied().collect();
        assert!(close(by_id[&0], 1.0));
        assert!(close(by_id[&1], 2.0));
        assert!(close(by_id[&2], 0.0));
        assert_eq!(ids(&fused), vec![1, 0, 2]);
    }

    #[test]
    fn linear_zero_range_list_normalizes_to_one_half() {
        // The pinned convention: a list whose scores are all equal (a
        // single-entry list included) carries no magnitude information;
        // every entry normalizes to the neutral midpoint 0.5.
        let lists = [vec![(0u32, 7.0), (1u32, 7.0), (2u32, 7.0)]];
        let fused = score_fuse(&lists, FusionMethod::Linear, None, None);
        assert_eq!(ids(&fused), vec![0, 1, 2]); // all tie; first appearance
        assert!(fused.iter().all(|(_, s)| close(*s, 0.5)));
    }

    #[test]
    fn linear_preserves_a_single_lists_order_and_norms() {
        let lists = [vec![(4u32, 3.0), (0u32, 1.0), (2u32, 0.0), (9u32, -5.0)]];
        let fused = score_fuse(&lists, FusionMethod::Linear, None, None);
        assert_eq!(ids(&fused), vec![4, 0, 2, 9]);
        let by_id: std::collections::HashMap<u32, f64> = fused.iter().copied().collect();
        assert!(close(by_id[&4], 1.0));
        assert!(close(by_id[&0], 0.75));
        assert!(close(by_id[&2], 0.625));
        assert!(close(by_id[&9], 0.0));
    }

    #[test]
    fn linear_negative_scores_normalize_without_signs_leaking() {
        // A cosine-similarity list (-1..1): min-max maps the whole
        // signed range onto [0, 1] -- the signs wash out, the ORDER is
        // what survives.
        let lists = [vec![(0u32, 1.0), (1u32, 0.0), (2u32, -1.0)]];
        let fused = score_fuse(&lists, FusionMethod::Linear, None, None);
        assert!(close(fused[0].1, 1.0));
        assert!(close(fused[1].1, 0.5));
        assert!(close(fused[2].1, 0.0));
    }

    // --- the shared contract --------------------------------------------

    #[test]
    fn ties_break_by_earliest_first_appearance_not_id_value() {
        // Both documents norm 1.0 in their own single-entry list (the
        // zero-range shape, 0.5 each after the convention): the HIGHER
        // id appeared first and must outrank the lower.
        let lists = [vec![(9u32, 3.0)], vec![(3u32, 3.0)]];
        for method in [
            FusionMethod::CombMnz,
            FusionMethod::Borda,
            FusionMethod::Linear,
        ] {
            let fused = score_fuse(&lists, method, None, None);
            assert_eq!(ids(&fused), vec![9, 3], "{method:?}");
        }
    }

    #[test]
    fn empty_inner_lists_contribute_no_votes() {
        let lists = [vec![(1u32, 1.0)], Vec::new(), vec![(0u32, 2.0)]];
        for method in [
            FusionMethod::CombMnz,
            FusionMethod::Borda,
            FusionMethod::Linear,
        ] {
            let fused = score_fuse(&lists, method, None, None);
            assert_eq!(ids(&fused), vec![1, 0], "{method:?}");
        }
    }

    #[test]
    fn all_empty_lists_answer_empty() {
        let lists: [Vec<(u32, f64)>; 2] = [Vec::new(), Vec::new()];
        for method in [
            FusionMethod::CombMnz,
            FusionMethod::Borda,
            FusionMethod::Linear,
        ] {
            assert!(
                score_fuse(&lists, method, None, None).is_empty(),
                "{method:?}"
            );
        }
    }

    #[test]
    fn k_truncates_the_top_n() {
        let lists = [vec![(0u32, 3.0), (1u32, 2.0), (2u32, 1.0)]];
        for method in [
            FusionMethod::CombMnz,
            FusionMethod::Borda,
            FusionMethod::Linear,
        ] {
            let fused = score_fuse(&lists, method, None, Some(2));
            assert_eq!(ids(&fused), vec![0, 1], "{method:?}");
            let fused = score_fuse(&lists, method, None, Some(10)); // clamps
            assert_eq!(ids(&fused), vec![0, 1, 2], "{method:?}");
        }
    }

    #[test]
    fn scores_never_nan_even_overflow_adjacent() {
        // The range overflows (max - min = +inf over [−1.7e308,
        // 1.7e308]): the saturating-norm policy keeps every norm in
        // [0, 1] (the top of the scale answers 1.0, finite numerators
        // divide to 0.0-scale), and the all-nonneg sums stay
        // finite-or-+inf. No NaN anywhere, the sort stays total.
        let lists = [
            vec![(0u32, 1.7e308), (1u32, 0.0), (2u32, -1.7e308)],
            vec![(3u32, 1.7e308), (4u32, -1.7e308)],
        ];
        for method in [FusionMethod::CombMnz, FusionMethod::Linear] {
            let fused = score_fuse(&lists, method, None, None);
            for (_, score) in &fused {
                assert!(!score.is_nan(), "{method:?}: NaN leaked");
            }
            let by_id: std::collections::HashMap<u32, f64> = fused.iter().copied().collect();
            assert!(close(by_id[&0], 1.0), "{method:?}"); // the saturated top
            assert_eq!(ids(&fused)[0], 0, "{method:?}");
        }
    }

    #[test]
    fn extreme_magnitudes_normalize_by_the_lists_own_range() {
        // 1e300 over [1e-300, 1e300]: the range is finite (no
        // overflow), the norms are the ordinary ratios -- 1.0 and
        // 0.0 at the ends, and a mid score divides to its real
        // fraction. The magnitudes themselves never leak past the
        // normalization.
        let lists = [vec![(0u32, 1e300), (1u32, 5e299), (2u32, 1e-300)]];
        let fused = score_fuse(&lists, FusionMethod::Linear, None, None);
        assert!(close(fused[0].1, 1.0));
        assert!(close(fused[1].1, 0.5));
        assert!(close(fused[2].1, 0.0));
    }

    #[test]
    fn weights_none_is_byte_identical_to_all_ones() {
        // The default must be the unweighted fusion, not a weighted
        // spelling of it: same scores bit for bit, same order.
        let lists = [
            vec![(0u32, 1.0), (1u32, 0.5)],
            vec![(1u32, 2.0), (0u32, 1.0)],
            vec![(2u32, 7.0)],
        ];
        for method in [
            FusionMethod::CombMnz,
            FusionMethod::Borda,
            FusionMethod::Linear,
        ] {
            let plain = score_fuse(&lists, method, None, None);
            let ones = score_fuse(&lists, method, Some(&[1.0, 1.0, 1.0]), None);
            assert_eq!(plain, ones, "{method:?}");
        }
    }

    #[test]
    fn hand_computed_weighted_combmnz() {
        // weights [2.0, 0.5]. L0's range [5, 10]: d0 norm 1.0, d1 0.0.
        // L1's range [2, 5]: d1 norm 1.0, d0 0.0. CombSUM: d0 = 2×1.0
        // + 0.5×0.0 = 2.0; d1 = 2×0.0 + 0.5×1.0 = 0.5. Both in both
        // lists, count 2 each: d0 = 4.0, d1 = 1.0.
        let lists = [
            vec![(0u32, 10.0), (1u32, 5.0)],
            vec![(1u32, 5.0), (0u32, 2.0)],
        ];
        let fused = score_fuse(&lists, FusionMethod::CombMnz, Some(&[2.0, 0.5]), None);
        let by_id: std::collections::HashMap<u32, f64> = fused.iter().copied().collect();
        assert!(close(by_id[&0], 4.0));
        assert!(close(by_id[&1], 1.0));
    }

    #[test]
    fn weights_can_flip_a_consensus_loss() {
        // The point of the extension: a 3x-weighted single norm-1.0
        // vote outranks the two unweighted 0.4-norm votes it loses at
        // equal weights (3.0 > 0.8 vs 1.0 < 0.8 flipped).
        let shape = [
            vec![(0u32, 1.0), (9u32, 0.0)],
            vec![(1u32, 1.0), (2u32, 0.4), (3u32, 0.0)],
            vec![(1u32, 1.0), (2u32, 0.4), (3u32, 0.0)],
        ];
        let equal = score_fuse(&shape, FusionMethod::Linear, Some(&[1.0, 1.0, 1.0]), None);
        assert_eq!(equal[0].0, 1); // two 0.8-scale votes win at equal weights
        let boosted = score_fuse(&shape, FusionMethod::Linear, Some(&[3.0, 1.0, 1.0]), None);
        assert_eq!(boosted[0].0, 0); // the weighted top now wins
        assert!(close(boosted[0].1, 3.0));
    }

    #[test]
    fn short_weight_slice_defaults_the_tail_to_one() {
        // The core tolerates a loose slice (the binding validates the
        // exact length); the missing tail votes at weight 1.0.
        let lists = [
            vec![(0u32, 1.0), (1u32, 0.0)],
            vec![(2u32, 1.0), (3u32, 0.0)],
        ];
        let fused = score_fuse(&lists, FusionMethod::Linear, Some(&[4.0]), None);
        let by_id: std::collections::HashMap<u32, f64> = fused.iter().copied().collect();
        assert!(close(by_id[&0], 4.0));
        assert!(close(by_id[&2], 1.0));
    }

    #[test]
    fn a_denormal_weight_underflow_still_emits_every_voted_doc() {
        // The emission signal is vote existence, not score positivity:
        // a zero-range list (every norm 0.5) times 5e-324 (the smallest
        // subnormal) underflows every term to exactly 0.0 (0.5 × 5e-324
        // rounds to even, 0.0), and the voted docs still emit --
        // 0.0-score pairs in first-appearance order (all scores tie at
        // 0.0, so the tie-break decides).
        let lists = [vec![(0u32, 1.0), (1u32, 1.0), (2u32, 1.0)]];
        for method in [FusionMethod::CombMnz, FusionMethod::Linear] {
            let fused = score_fuse(&lists, method, Some(&[5e-324]), None);
            assert_eq!(ids(&fused), vec![0, 1, 2], "{method:?}");
            assert!(fused.iter().all(|(_, s)| *s == 0.0), "{method:?}");
        }
        // The mixed shape: d0's term (norm 1.0 × 5e-324) survives as a
        // subnormal, d1's (norm 0.0) is exactly 0.0; both appear, the
        // positive subnormal first, the order deterministic.
        let mixed = [vec![(0u32, 10.0), (1u32, 0.0)]];
        for method in [FusionMethod::CombMnz, FusionMethod::Linear] {
            let fused = score_fuse(&mixed, method, Some(&[5e-324]), None);
            assert_eq!(fused, vec![(0, 5e-324), (1, 0.0)], "{method:?}");
        }
    }

    #[test]
    fn a_loose_n_docs_still_emits_only_carried_indices() {
        // The filter's surviving job: slots a sparse numbering sized
        // (here index 5 in a 6-wide table) but no list ever carried
        // stay unemitted. The binding's dense numbering cannot produce
        // the gap; the fuzz target's remap can't either -- the shape
        // is pinned anyway, rank_fuse's own contract.
        let lists = [vec![(0u32, 1.0), (1u32, 0.5), (2u32, 0.0)]];
        let fused = score_fuse(&lists, FusionMethod::Linear, None, None);
        assert_eq!(ids(&fused), vec![0, 1, 2]);
    }

    #[test]
    fn parse_fusion_method_names_the_accepted_set() {
        for name in SCORE_FUSION_METHODS {
            assert!(parse_fusion_method(name).is_ok(), "{name} must parse");
        }
        let err = parse_fusion_method("combmnz ").unwrap_err();
        assert!(err.contains("combmnz") && err.contains("borda") && err.contains("linear"));
    }
}
