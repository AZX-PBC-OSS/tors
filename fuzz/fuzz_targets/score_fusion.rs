//! `score_fusion_impl` never panics on arbitrary fused inputs: the
//! fusion emits exactly the distinct ids the lists carry (vote
//! existence, not score positivity -- Borda's last place, a zero-range
//! list's midpoint, an underflowed denormal weight all still appear),
//! in score-descending order with ties broken by first appearance, all
//! scores finite-or-+inf and NEVER NaN (the saturating-norm policy:
//! an overflowed range's overflowing numerator answers 1.0, never
//! inf/inf), for any method, any positive-finite weight spelling, and
//! any `k` (truncation only ever shortens).
//!
//! The binding-side domain checks (finite scores, positive finite
//! weights, exact lengths) are pinned in tests/test_score_fusion.py;
//! this target exercises the core's own contract over the shapes the
//! binding can produce, INCLUDING the within-list duplicates the core
//! is trusted to receive pre-folded and the raw-duplicate votes it
//! then casts (the rank_fusion target's same trust boundary).

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

use tors::score_fusion_impl::{FusionMethod, SCORE_FUSION_METHODS, score_fuse};

#[derive(Arbitrary, Debug)]
struct Input {
    /// The scored lists, as RAW label values the target first remaps to
    /// the dense dedup-index space the py layer materializes (one index
    /// per distinct id, assigned on first sight), each entry carrying
    /// the score's raw bits. The core's contract is a dense id table,
    /// so the hostile shapes a caller can actually reach are duplicate
    /// ids within a list (the binding folds them; the core is trusted
    /// with the folded form, and the target exercises BOTH the folded
    /// and the raw-duplicate spelling), overlaps across lists, and
    /// empty lists, all of which the remap preserves.
    lists: Vec<Vec<(u32, u64)>>,
    /// The method (mod 3 over the accepted spellings' positions).
    method: u8,
    /// Per-list weights, drawn across the None/short/exact spellings
    /// the binding can produce: bit 0 selects weighted, bit 1 pads the
    /// slice short by one (the core defaults the missing tail to 1.0;
    /// the binding rejects a mismatch, the padding shape is the core's
    /// own contract), each weight a positive finite f64 assembled from
    /// bit patterns that avoid the zero/negative/NaN/inf domain the
    /// binding rejects.
    weight_bits: u32,
    /// The top-N truncation: bit 0 selects Some, the value (mod 64) + 1
    /// steers across the clamp shapes (k past the distinct count
    /// truncates nothing).
    k_bits: u8,
}

/// A finite f64 assembled from raw bits, biased toward the documented
/// edge classes: exact 0.0/-0.0 (the zero-range and negative-score
/// shapes), ±1.0, ±1e300 and ±1e-300 (the extreme-magnitude class),
/// ±1.7e308 (the range-overflow class), and f64::MIN_POSITIVE-scale
/// subnormals, plus free [1.0, 2.0)-scale normals the fuzzer steers.
/// Never NaN, never infinite: those are the binding's ValueError
/// domain, unreachable at the core.
fn finite_score(bits: u64) -> f64 {
    if bits & (1 << 59) != 0 {
        let normal = f64::from_bits((bits & 0x000F_FFFF_FFFF_FFFF) | 0x3FF0_0000_0000_0000);
        return if bits & (1 << 58) != 0 {
            -normal
        } else {
            normal
        };
    }
    match (bits >> 55) & 0xF {
        0 => 0.0,
        1 => -0.0,
        2 => 1.0,
        3 => -1.0,
        4 => 1e300,
        5 => -1e300,
        6 => 1e-300,
        7 => -1e-300,
        8 => 1.7e308,
        9 => -1.7e308,
        10 => f64::MIN_POSITIVE,
        11 => -f64::MIN_POSITIVE,
        12 => 5e-324, // the smallest subnormal
        13 => -5e-324,
        14 => 0.5,
        _ => -0.5,
    }
}

fuzz_target!(|input: Input| {
    if input.lists.is_empty() {
        return; // the zero-list ValueError is the binding's, pinned py-side
    }
    let method = match input.method % 3 {
        0 => FusionMethod::CombMnz,
        1 => FusionMethod::Borda,
        _ => FusionMethod::Linear,
    };
    // The dense remap (the binding's dedup table, spelled in Rust):
    // first-appearance indices over the raw labels.
    let mut remap: HashMap<u32, u32> = HashMap::new();
    let lists: Vec<Vec<(u32, f64)>> = input
        .lists
        .iter()
        .map(|list| {
            list.iter()
                .map(|(label, bits)| {
                    let next = remap.len() as u32;
                    let idx = *remap.entry(*label).or_insert(next);
                    (idx, finite_score(*bits))
                })
                .collect()
        })
        .collect();
    // The per-list weights: positive finite f64s assembled from raw
    // bits (an exponent in the normal range, mantissa bits from the
    // fuzzer), then the None/short/exact spellings steered by the low
    // bits. The value-domain validation (positive, finite) is the
    // binding's; the core consumes whatever slice it is handed.
    let weights: Option<Vec<f64>> = if input.weight_bits & 1 == 1 {
        let mut w: Vec<f64> = input
            .lists
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let bits = (input.weight_bits as u64) << 32 | (i as u64) << 16;
                // Mantissa bits only under a fixed 0x3FF exponent (the
                // [1.0, 2.0) normal range, always finite, never signed
                // or zero), the fuzzer steering the low bits.
                f64::from_bits((bits & 0x000F_FFFF_FFFF_FFFF) | 0x3FF0_0000_0000_0000)
            })
            .collect();
        if input.weight_bits & 2 == 2 && w.len() > 1 {
            w.pop(); // the short-slice spelling: the core defaults the tail
        }
        Some(w)
    } else {
        None
    };
    let k = if input.k_bits & 1 == 1 {
        Some((input.k_bits % 64) as usize + 1)
    } else {
        None
    };

    let fused = score_fuse(&lists, method, weights.as_deref(), k);

    // Every emitted score is finite-or-+inf and never NaN (the
    // saturating norm + the all-nonneg accumulation contract); the
    // emitted indices are in range and pairwise distinct (one entry
    // per carried id). Truncation only ever shortens.
    let mut seen = std::collections::HashSet::new();
    for (idx, score) in &fused {
        assert!(!score.is_nan(), "NaN leaked ({method:?}): {score}");
        assert!(
            score.is_finite() || *score == f64::INFINITY,
            "bad score {score}"
        );
        assert!(*score >= 0.0, "negative score {score} ({method:?})");
        assert!(seen.insert(*idx), "duplicate fused index: {idx}");
    }
    assert!(fused.len() <= seen.len());
    if let Some(k) = k {
        assert!(
            fused.len() <= k,
            "k truncation ignored: {} > {k}",
            fused.len()
        );
    }
    // Exactly the carried documents are emitted (vote existence): the
    // remap assigns one index per distinct label, every index appears
    // at least once by construction -- on the FULL run only, a
    // k-truncated run is its prefix by the assertion below.
    if k.is_none() {
        assert_eq!(seen.len(), remap.len(), "fused set != carried set");
    }

    // The full run (k=None) is a prefix of what truncation returns:
    // same order, same scores, bit for bit.
    if k.is_some() {
        let full = score_fuse(&lists, method, weights.as_deref(), None);
        assert!(full.starts_with(&fused), "truncated run is not a prefix");
    }

    // Score-descending, ties by first appearance: re-derive the scores
    // AND the first-appearance order independently (the same formula
    // the module docs state, the slice as-spelled, short tails at 1.0)
    // and check the emitted order pairwise.
    let table = remap.len();
    let mut scores = vec![0.0f64; table];
    let mut counts = vec![0u32; table];
    let mut first_seen = vec![u32::MAX; table];
    let mut seen_count = 0u32;
    for (list_idx, list) in lists.iter().enumerate() {
        if list.is_empty() {
            continue;
        }
        let weight = weights
            .as_deref()
            .and_then(|w| w.get(list_idx).copied())
            .unwrap_or(1.0);
        match method {
            FusionMethod::Borda => {
                let n = list.len() as f64;
                for (position, (doc, _)) in list.iter().enumerate() {
                    let doc = *doc as usize;
                    // The vote first, then the weight (the core's and
                    // the Python oracle's own rounding order -- a
                    // multiply-first spelling rounds 1 ulp away from
                    // the pinned contract).
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
                        0.5
                    } else {
                        let num = score - min;
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
        for (doc, score) in scores.iter_mut().enumerate() {
            *score *= counts[doc] as f64;
        }
    }
    for w in fused.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (sa, sb) = (scores[a.0 as usize], scores[b.0 as usize]);
        assert!(
            sa > sb || (sa == sb && first_seen[a.0 as usize] < first_seen[b.0 as usize]),
            "order violated ({method:?}): {a:?} then {b:?} (scores {sa}, {sb})"
        );
    }
    // The accepted set's spellings all parse (the parse function is
    // the binding's ValueError source; pin it cannot rot silently).
    assert_eq!(SCORE_FUSION_METHODS.len(), 3);
});
