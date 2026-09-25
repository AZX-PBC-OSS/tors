//! MinHash LSH banding: the pure-Rust core of `tors.lsh_candidates`,
//! `tors.lsh_probability`, and `tors.lsh_threshold` — the near-duplicate
//! candidate generator `docs/design.md` names as the banding companion
//! to `minhash_signature`. `dedup_near_dup` scores every pair of a SMALL
//! candidate list (its O(n²) sweep is documented, budget-pinned design);
//! banding is the recall-side complement for LARGER corpora: cut each
//! signature into `b` bands of `r` rows, bucket the band hashes, and a
//! pair of documents is a candidate exactly when one band's buckets
//! collide. One stateless pass over the signatures, no table held across
//! calls, no model data: the persistent banding table a corpus-scale
//! pipeline queries incrementally stays caller state (docs/design.md's
//! scope cut), and this module is the one-shot helper beside it.
//!
//! # Prior art
//!
//! - **Shingling + resemblance**: Broder, Glassman, Manasse, and Zweig,
//!   "Syntactic Clustering of the Web" (WWW 1997) — a document's
//!   near-duplicate identity is the set of its k-grams and their
//!   Jaccard-index resemblance, the quantity `minhash_signature`
//!   estimates and banding consumes. The MinHash core
//!   (`minhash_impl`) cites the same work.
//! - **The banding S-curve**: Leskovec, Rajaraman, and Ullman, "Mining
//!   of Massive Datasets", chapter 3 (the LSH section): two signatures
//!   whose similarity is `s` share at least one band's bucket with
//!   probability `P(s) = 1 - (1 - s^r)^b` — flat near 0, a steep step
//!   around the approximate threshold `(1/b)^(1/r)`, flat near 1. The
//!   same curve and threshold formulation appear in datasketch's
//!   `minhash` documentation (the "optimal parameters" use case):
//!   `lsh_probability` is that curve as a pure formula, so a caller can
//!   PICK `bands`/`rows` for a target similarity threshold instead of
//!   copying one from a tutorial.
//! - **API shape, for ergonomics comparison only**: datasketch's
//!   `MinHashLSH` builds a persistent table (`insert` per signature,
//!   `query` per probe) precisely because it serves incremental
//!   workloads. tors stays stateless (docs/design.md), so there is no
//!   table here to insert into or query against: one call takes every
//!   signature in hand and returns every candidate pair, and a caller
//!   needing incremental inserts owns the table themselves.
//!
//! # The band hash (the fixed-seed contract)
//!
//! Each band's `r` rows hash to one 64-bit bucket key with the crate's
//! one hashing contract: XXH64, seed 0, over the same injective
//! length-prefixed little-endian framing `minhash_impl::hash_tokens`
//! spells for token windows (LE64 of the row count, then LE64 of each
//! row), shared as `minhash_impl::hash_u64_frame`. The seed is FIXED —
//! 0, the same frozen seed the shingle hashes ride — and there is no
//! rng handle anywhere in the call: the same signatures band to the same
//! buckets on every call, in every process, on every machine, within one
//! tors version (the `minhash_signature` determinism contract carries
//! over, and its segmentation-table boundary with it: a release bumping
//! those tables can change signatures, so persisted signatures
//! re-baseline on upgrade either way).
//!
//! # The false-positive contract (recall-biased, stated plainly)
//!
//! Two DISSIMILAR signatures can still become candidates, in exactly two
//! ways, and both are by design:
//!
//! - **Genuine band agreement**: `r` rows of the two signatures agree by
//!   chance. This is the S-curve itself: at similarity `s` the pair
//!   becomes a candidate with probability `P(s)`, so low-similarity
//!   pairs leak through at `1 - P(s) - (recall below the step)` — LSH
//!   banding trades precision for recall EVERYWHERE it is used, and the
//!   candidate list is a filter to score downstream
//!   (`shingle_jaccard`, `dedup_near_dup`), never a verdict.
//! - **Band-hash collisions**: two DIFFERENT row frames hashing to the
//!   same 64-bit key. Under any reasonable hash this is probability
//!   ~`k²/2^65` over `k` distinct keys — negligible, but nonzero, and
//!   deliberately NOT engineered away (a second hash pass or a 128-bit
//!   key buys nothing: the recall-biased contract above already
//!   dominates the false-positive budget). The statistical pin in
//!   `tests/test_lsh.py` holds this: disjoint signatures produce no
//!   candidates beyond a generous collision margin.
//!
//! Candidate output NEVER misses a true pair above the step without
//! cause: two signatures identical in at least one full band always
//! collide in that band's frame deterministically (identical frames hash
//! identically — the pin holds this exactly).
//!
//! # Cost and memory (output-sensitive by construction)
//!
//! The pass is one sweep: `O(n · b)` band hashes (each `O(r)`, so
//! `O(n · num_perm)` total — linear in the signature data itself) plus
//! pair emission ONLY inside buckets, which is `O(output)` by
//! definition: the nested loop over a bucket's members emits exactly the
//! candidate pairs that bucket contributes, and nothing pairwise
//! compares signatures that share no bucket. There is no `n²` anywhere
//! except through the output. Resident memory is one band's bucket table
//! (`O(n)`, freed per band) plus the dedup set (`O(pairs)`) — the
//! VmHWM guard in `tests/test_lsh.py` pins the class.

use std::collections::{BTreeSet, HashMap};

use crate::minhash_impl::hash_u64_frame;

/// The banding outcome: every candidate pair as `(i, j)` with `i < j`,
/// deduplicated across bands (two signatures sharing several band
/// buckets appear once) and in ascending `(i, j)` order — the
/// deterministic result-object style `DedupOutcome` shares (every field
/// present every time; the empty corpus gives the empty pair list).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidatePairs {
    pub pairs: Vec<(usize, usize)>,
}

/// The core candidate generator: `bands` bands of `rows` rows over every
/// signature (each signature's length MUST equal `bands * rows` — the
/// pyo3 binding validates this as a `ValueError` and the core asserts
/// it for direct Rust callers), one pass, deduplicated ascending pairs.
/// See the module doc for the hash contract, the false-positive
/// contract, and the output-sensitive cost shape.
pub fn lsh_candidates(signatures: &[Vec<u64>], bands: usize, rows: usize) -> CandidatePairs {
    assert!(bands >= 1, "bands {bands} must be at least 1");
    assert!(rows >= 1, "rows {rows} must be at least 1");
    let num_perm = bands
        .checked_mul(rows)
        .expect("bands * rows overflows usize (the pyo3 binding rejects this shape)");
    for (idx, sig) in signatures.iter().enumerate() {
        assert!(
            sig.len() == num_perm,
            "signature {idx} has {} rows, expected bands * rows = {num_perm}",
            sig.len()
        );
    }
    // BTreeSet: dedup across bands AND the ascending (i, j) order in one
    // structure. Bucket iteration order (std RandomState, per-process
    // seeded) never escapes: only the sorted set crosses out — the same
    // order-independence argument `minhash_impl`'s distinct-set sweep
    // makes.
    let mut pairs: BTreeSet<(usize, usize)> = BTreeSet::new();
    for band in 0..bands {
        let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
        for (idx, sig) in signatures.iter().enumerate() {
            let lo = band * rows;
            let key = hash_u64_frame(&sig[lo..lo + rows]);
            buckets.entry(key).or_default().push(idx);
        }
        for members in buckets.values() {
            // Members accumulated in ascending index order, so the inner
            // loop emits exactly this bucket's candidate pairs, `i < j` —
            // work proportional to the pairs this bucket CONTRIBUTES,
            // never a pairwise scan over non-candidates.
            for (pos, &i) in members.iter().enumerate() {
                for &j in &members[pos + 1..] {
                    pairs.insert((i, j));
                }
            }
        }
    }
    CandidatePairs {
        pairs: pairs.into_iter().collect(),
    }
}

/// The banding S-curve as a pure formula: the probability that two
/// signatures with Jaccard similarity `s` share at least one of `bands`
/// band buckets, `P(s) = 1 - (1 - s^rows)^bands` (Leskovec, Rajaraman,
/// and Ullman, "Mining of Massive Datasets", ch. 3; datasketch's
/// parameter-tuning docs spell the same curve). `s` must be in `[0, 1]`
/// (the pyo3 binding rejects anything else, NaN included, before the
/// core sees it). The curve's ends are exact: `s = 0` gives `0.0`, `s =
/// 1` gives `1.0` — identical signatures are always candidates, disjoint
/// ones only through the module doc's false-positive channels.
pub fn lsh_probability(s: f64, bands: usize, rows: usize) -> f64 {
    assert!(
        s.is_finite() && (0.0..=1.0).contains(&s),
        "s {s} outside [0, 1] (the pyo3 binding rejects this before the core)"
    );
    assert!(bands >= 1, "bands {bands} must be at least 1");
    assert!(rows >= 1, "rows {rows} must be at least 1");
    1.0 - (1.0 - s.powf(rows as f64)).powf(bands as f64)
}

/// The approximate threshold similarity where the S-curve takes its
/// step: `(1/bands)^(1/rows)` (Leskovec, Rajaraman, and Ullman,
/// "Mining of Massive Datasets", ch. 3, the same formulation datasketch's
/// docs carry). An approximation, not an inversion of
/// [`lsh_probability`]: pairs at this similarity are candidates with
/// probability NEAR the curve's midpoint, not exactly 0.5. Pick
/// `bands`/`rows` for a target threshold with this, then verify the
/// actual curve behavior with [`lsh_probability`].
pub fn lsh_threshold(bands: usize, rows: usize) -> f64 {
    assert!(bands >= 1, "bands {bands} must be at least 1");
    assert!(rows >= 1, "rows {rows} must be at least 1");
    (1.0 / bands as f64).powf(1.0 / rows as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The naive spec spelling the BTreeSet core is pinned against:
    /// recompute the candidate relation per band with a fresh map and a
    /// sort-dedup, no shared code beyond the band hash itself.
    fn naive_candidates(signatures: &[Vec<u64>], bands: usize, rows: usize) -> Vec<(usize, usize)> {
        let mut all: Vec<(usize, usize)> = Vec::new();
        for band in 0..bands {
            let mut keys: Vec<(u64, usize)> = signatures
                .iter()
                .enumerate()
                .map(|(idx, sig)| {
                    let lo = band * rows;
                    (hash_u64_frame(&sig[lo..lo + rows]), idx)
                })
                .collect();
            keys.sort_unstable();
            for group in keys.chunk_by(|a, b| a.0 == b.0) {
                for a in 0..group.len() {
                    for b in a + 1..group.len() {
                        let (i, j) = (group[a].1, group[b].1);
                        all.push((i.min(j), i.max(j)));
                    }
                }
            }
        }
        all.sort_unstable();
        all.dedup();
        all
    }

    fn sig(values: &[u64]) -> Vec<u64> {
        values.to_vec()
    }

    #[test]
    fn band_frame_hash_matches_the_manual_xxh64_frame() {
        // The framing pin: LE64(count) then LE64 per row, XXH64 seed 0,
        // the same frame discipline hash_tokens spells for tokens (the
        // minhash test's own oneshot oracle, over u64 rows).
        let rows = [1u64, 2, 3];
        let mut frame = Vec::new();
        frame.extend_from_slice(&3u64.to_le_bytes());
        for row in rows {
            frame.extend_from_slice(&row.to_le_bytes());
        }
        assert_eq!(
            hash_u64_frame(&rows),
            twox_hash::XxHash64::oneshot(0, &frame)
        );
        // The empty frame is well-defined (a zero-row band is refused at
        // validation, but the frame itself is the honest XXH64 of the
        // length prefix).
        assert_eq!(
            hash_u64_frame(&[]),
            twox_hash::XxHash64::oneshot(0, &0u64.to_le_bytes())
        );
        // Framing injectivity the band keys rely on: [1, 2] vs [2, 1]
        // (order carries), and [1] vs [1, 0] (the length prefix carries).
        assert_ne!(hash_u64_frame(&[1, 2]), hash_u64_frame(&[2, 1]));
        assert_ne!(hash_u64_frame(&[1]), hash_u64_frame(&[1, 0]));
    }

    #[test]
    fn identical_signatures_are_always_candidates() {
        // Identical signatures collide in EVERY band, so every pair of
        // them is a candidate, at every (bands, rows) shape.
        let sigs: Vec<Vec<u64>> = (0..5).map(|_| sig(&[7u64; 128])).collect();
        for (bands, rows) in [(16usize, 8usize), (1, 128), (128, 1), (8, 16)] {
            let out = lsh_candidates(&sigs, bands, rows);
            let expected: Vec<(usize, usize)> = (0..5)
                .flat_map(|i| (i + 1..5).map(move |j| (i, j)))
                .collect();
            assert_eq!(
                out,
                CandidatePairs { pairs: expected },
                "bands={bands} rows={rows}"
            );
        }
    }

    #[test]
    fn disjoint_signatures_never_collide() {
        // Fully distinct u64 rows: no two band frames equal, no XXH64
        // collision over this handful of keys, no candidates.
        let sigs: Vec<Vec<u64>> = (0..64u64)
            .map(|d| (0..128u64).map(|r| d * 1000 + r).collect())
            .collect();
        assert!(lsh_candidates(&sigs, 16, 8).pairs.is_empty());
        assert!(lsh_candidates(&sigs, 1, 128).pairs.is_empty());
    }

    #[test]
    fn one_shared_band_makes_a_candidate() {
        // Two signatures agreeing on exactly ONE band's rows: that band's
        // frames are identical, so the pair is a candidate regardless of
        // the other bands (the recall side of the contract, exact).
        let mut a = vec![0u64; 16];
        let mut b = vec![1u64; 16];
        for (a_row, b_row) in a.iter_mut().zip(b.iter_mut()).take(8) {
            *a_row = 42;
            *b_row = 42;
        }
        let out = lsh_candidates(&[a, b], 2, 8);
        assert_eq!(out.pairs, vec![(0, 1)]);
    }

    #[test]
    fn pairs_are_deduplicated_sorted_and_symmetric() {
        // Six signatures in three identical families (two bands each):
        // every within-family pair must appear exactly once, ascending,
        // i < j, symmetric as a relation.
        let fam = |base: u64| vec![base, base + 1, base + 2, base + 3];
        let sigs = vec![fam(0), fam(0), fam(10), fam(0), fam(10), fam(10)];
        let out = lsh_candidates(&sigs, 2, 2);
        assert_eq!(
            out.pairs,
            vec![(0, 1), (0, 3), (1, 3), (2, 4), (2, 5), (4, 5)]
        );
        // Symmetry as a relation: every (i, j) has i < j, and no (j, i)
        // duplicate exists.
        for (i, j) in &out.pairs {
            assert!(i < j);
            assert!(
                !out.pairs.contains(&(*j, *i)),
                "({j}, {i}) back-pair present"
            );
        }
    }

    #[test]
    fn candidate_relation_survives_input_permutation() {
        // The contract under relabeling: permuting the input permutes
        // the SAME candidate relation. pairs(perm)[pi, pj] comes from
        // pairs(orig)[i, j] — the relation is a function of the
        // signatures' content, not their order.
        let fam = |base: u64| (base..base + 8).collect::<Vec<u64>>();
        let sigs = vec![fam(0), fam(100), fam(0), fam(200), fam(100), fam(300)];
        // perm[p] = q means the permuted list's position p holds the
        // original signature q; the relation maps each original index q
        // to the position holding it (the inverse permutation).
        let perm = [3usize, 0, 5, 1, 4, 2];
        let original = lsh_candidates(&sigs, 4, 2);
        let permuted_sigs: Vec<Vec<u64>> = perm.iter().map(|&q| sigs[q].clone()).collect();
        let permuted = lsh_candidates(&permuted_sigs, 4, 2);
        let relabel = |q: usize| perm.iter().position(|&p| p == q).expect("inverse exists");
        let mut expected: Vec<(usize, usize)> = original
            .pairs
            .iter()
            .map(|&(i, j)| {
                let (pi, pj) = (relabel(i), relabel(j));
                (pi.min(pj), pi.max(pj))
            })
            .collect();
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(permuted.pairs, expected);
        // And the permuted spelling is itself deterministic.
        assert_eq!(permuted, lsh_candidates(&permuted_sigs, 4, 2));
    }

    #[test]
    fn core_matches_the_naive_reference_over_a_battery() {
        // The differential pin: the BTreeSet core must agree with the
        // sort-dedup naive spelling over a mixed corpus at several
        // (bands, rows) shapes.
        let sigs: Vec<Vec<u64>> = [
            vec![1, 2, 3, 4, 5, 6],
            vec![1, 2, 3, 4, 5, 6], // identical to 0
            vec![1, 2, 3, 9, 9, 9], // one band shared
            vec![7, 8, 9, 10, 11, 12],
            vec![1, 2, 3, 4, 5, 6], // identical to 0 again
        ]
        .into_iter()
        .collect();
        for (bands, rows) in [(1usize, 6usize), (2, 3), (3, 2), (6, 1)] {
            assert_eq!(
                lsh_candidates(&sigs, bands, rows),
                CandidatePairs {
                    pairs: naive_candidates(&sigs, bands, rows)
                },
                "bands={bands} rows={rows}"
            );
        }
    }

    #[test]
    fn empty_and_singleton_inputs_give_the_empty_result() {
        assert_eq!(lsh_candidates(&[], 16, 8), CandidatePairs { pairs: vec![] });
        assert_eq!(
            lsh_candidates(&[vec![5u64; 128]], 16, 8),
            CandidatePairs { pairs: vec![] }
        );
    }

    #[test]
    #[should_panic(expected = "must be at least 1")]
    fn zero_bands_panics_in_the_core() {
        let _ = lsh_candidates(&[vec![0u64; 8]], 0, 8);
    }

    #[test]
    #[should_panic(expected = "must be at least 1")]
    fn zero_rows_panics_in_the_core() {
        let _ = lsh_candidates(&[vec![0u64; 8]], 8, 0);
    }

    #[test]
    #[should_panic(expected = "expected bands * rows")]
    fn length_mismatch_panics_in_the_core() {
        let _ = lsh_candidates(&[vec![0u64; 8], vec![0u64; 7]], 1, 8);
    }

    #[test]
    fn probability_pins_the_curve_values() {
        // Hand-computed rows: s=0.5, b=16, r=4 is
        // 1 - (1 - (1/2)^4)^16 = 1 - (15/16)^16 ~= 0.6439258695482072.
        let p = lsh_probability(0.5, 16, 4);
        assert!((p - 0.6439258695482072).abs() < 1e-9, "{p}");
        // The ends are exact.
        assert_eq!(lsh_probability(0.0, 16, 8), 0.0);
        assert_eq!(lsh_probability(1.0, 16, 8), 1.0);
        // b=1 collapses to s^r (one band is a single r-row vote).
        assert_eq!(lsh_probability(0.5, 1, 2), 0.25);
        // r=1, b=16: 1 - (1 - s)^16.
        let q = lsh_probability(0.5, 16, 1);
        assert!((q - (1.0 - 0.5f64.powi(16))).abs() < 1e-12, "{q}");
    }

    #[test]
    fn probability_is_monotone_in_s() {
        for (bands, rows) in [(16usize, 8usize), (1, 1), (4, 4), (128, 1)] {
            let mut prev = -1.0f64;
            for step in 0..=20 {
                let s = step as f64 / 20.0;
                let p = lsh_probability(s, bands, rows);
                assert!(p >= prev, "non-monotone at s={s} b={bands} r={rows}");
                assert!((0.0..=1.0).contains(&p));
                prev = p;
            }
        }
    }

    #[test]
    fn threshold_pins_and_directions() {
        // The one-liner: (1/b)^(1/r), pinned literally.
        assert!((lsh_threshold(16, 8) - (1.0f64 / 16.0).powf(1.0 / 8.0)).abs() < 1e-15);
        // (1/16)^(1/8) = 2^(-1/2) = FRAC_1_SQRT_2, pinned to the constant.
        assert!((lsh_threshold(16, 8) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        // More bands LOWER the threshold (recall up); more rows RAISE it.
        assert!(lsh_threshold(32, 8) < lsh_threshold(16, 8));
        assert!(lsh_threshold(16, 16) > lsh_threshold(16, 8));
        // The 1-band edge: threshold (1/1)^(1/r) = 1.
        assert_eq!(lsh_threshold(1, 8), 1.0);
    }

    #[test]
    #[should_panic(expected = "outside [0, 1]")]
    fn probability_asserts_the_s_bounds() {
        let _ = lsh_probability(1.5, 16, 8);
    }

    #[test]
    #[should_panic(expected = "must be at least 1")]
    fn probability_asserts_the_band_bounds() {
        let _ = lsh_probability(0.5, 0, 8);
    }
}
