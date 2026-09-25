//! The near-duplicate comparison layer's core never panics on any input,
//! and the contracts the surface pins hold under raw adversarial bytes:
//! `shingle_jaccard`/`shingle_dice` live in [0, 1] and are symmetric,
//! identical texts score 1.0, `fingerprint_hamming` is symmetric and
//! zero on identical fingerprints, and `dedup_near_dup` is deterministic,
//! partitions the indices exactly (kept ∪ dropped = 0..n, kept == group
//! heads), is threshold-monotone at the pair level, and agrees with a
//! naive from-scratch re-dedup of its own corpus.
//!
//! The locality lane is the MinHash target's discipline applied to the
//! exact layer: a one-byte mutation of a token-DIVERSE text damages only
//! the shingle windows around the edit, so the exact Jaccard stays high
//! — asserted above a distinct-shingle floor that keeps degenerate
//! (repeated-token) inputs out, where the collapse is real and correct.
//!
//! Sizes are capped (the dedup sweep is O(n²) pair checks by design),
//! the `fuzz_targets/minhash.rs` discipline.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tors::near_dup_impl::{self, DedupMethod};
use tors::simhash_impl::simhash64;

#[derive(Arbitrary, Debug)]
enum Input {
    /// Two arbitrary texts at a fuzzed width: panic-freedom, the [0, 1]
    /// range and symmetry of both similarities, and the self-similarity
    /// row. `width` spans 1..=12: the default 3, the unigram edge 1, and
    /// widths past the text's token count (the empty-set convention).
    Pair { a: Vec<u8>, b: Vec<u8>, width: u8 },
    /// The input vs a one-byte flip of itself vs an independent random
    /// text: the exact-Jaccard locality invariant (a single edit of a
    /// diverse text is a high-Jaccard pair; an independent text shares
    /// nothing).
    Mutated {
        data: Vec<u8>,
        flip_at: u8,
        noise_seed: u64,
    },
    /// Exact adversarial strings (WB4 separator-attach corpus, ZWJ emoji,
    /// CJK, regional indicators) at fuzzed widths: the inputs a byte soup
    /// almost never assembles, pinned exactly.
    Exact { index: u8, width: u8 },
    /// An arbitrary corpus (up to 32 texts) at a fuzzed threshold and
    /// method: the dedup structural contract — determinism, the exact
    /// partition, threshold-monotone pair scoring, and agreement with a
    /// naive from-scratch re-dedup.
    Corpus {
        texts: Vec<Vec<u8>>,
        threshold_high: u8,
        threshold_low: u8,
        method: u8,
    },
}

/// The exact-string lane's corpus: every row a segmentation edge (see
/// `src/minhash_impl.rs`'s WB4 note and `fuzz_targets/minhash.rs`'s list).
const EXACT_STRINGS: &[&str] = &[
    "\u{1f}\u{301}",
    "\u{1f}\u{200d}",
    "a \u{1f}\u{301} b c",
    "A\u{1f}\u{301}b",
    "\u{1f469}\u{200d}\u{1f52c} \u{30c6}\u{30b9}\u{30c8}",
    "\u{1f1fa}\u{1f1f8}\u{1f1fa}",
    "\u{1100}\u{1161}\u{11a8} \u{e0}\u{30d}",
    "caf\u{e9} soci\u{e9}t\u{e9} na\u{ef}ve \u{6771}\u{4eac}\u{306f}\u{65e5}\u{672c}",
    "Hello, world! One. Two.",
    "   \t  ",
    "",
];

/// The repo's deterministic u64 LCG (the same Knuth-style constants
/// `tests/reference.py`'s corpus builders use): an independent random
/// text sharing nothing structured with a fuzz input.
fn lcg_text(seed: u64, tokens: usize) -> String {
    let mut state = seed;
    (0..tokens)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            format!("t{:06x}", (state >> 48) as usize & 0xffffff)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The distinct-shingle floor the locality invariant is asserted above
/// (the module doc's diversity guard): a repeated-token text has a
/// degenerate shingle set a single edit really can destroy.
const DIVERSITY_FLOOR: usize = 128;

fn check_pair_contract(a: &str, b: &str, width: usize) {
    // Range and symmetry, both similarities, exact arithmetic.
    let j_ab = near_dup_impl::shingle_jaccard(a, b, width);
    let j_ba = near_dup_impl::shingle_jaccard(b, a, width);
    assert!((0.0..=1.0).contains(&j_ab), "jaccard out of range: {j_ab}");
    assert_eq!(j_ab, j_ba, "jaccard not symmetric");
    let d_ab = near_dup_impl::shingle_dice(a, b, width);
    let d_ba = near_dup_impl::shingle_dice(b, a, width);
    assert!((0.0..=1.0).contains(&d_ab), "dice out of range: {d_ab}");
    assert_eq!(d_ab, d_ba, "dice not symmetric");
    // Dice weights the small set more: dice >= jaccard for every pair.
    assert!(d_ab >= j_ab, "dice {d_ab} < jaccard {j_ab}");
    // Self-similarity: identical texts score 1.0, deterministically.
    assert_eq!(near_dup_impl::shingle_jaccard(a, a, width), 1.0);
    assert_eq!(near_dup_impl::shingle_dice(a, a, width), 1.0);
    // The fingerprint distance: symmetric, zero on identical values.
    let (fa, fb) = (simhash64(a), simhash64(b));
    assert_eq!(
        near_dup_impl::fingerprint_hamming(u128::from(fa), u128::from(fb)),
        near_dup_impl::fingerprint_hamming(u128::from(fb), u128::from(fa))
    );
    assert_eq!(
        near_dup_impl::fingerprint_hamming(u128::from(fa), u128::from(fa)),
        0
    );
}

/// A naive from-scratch re-dedup of `texts` at `threshold` with fresh
/// fingerprints per pair (no shared precompute): the spec spelling the
/// sweep must agree with, riding the crate's public primitives only (the
/// fuzz crate links `tors` as an external crate — the pub(crate) helpers
/// are invisible here, which is the point: agreement must hold through
/// the public surface).
fn naive_dedup(texts: &[String], threshold: f64, method: DedupMethod) -> (Vec<usize>, Vec<usize>) {
    let mut kept: Vec<usize> = Vec::new();
    let mut dropped: Vec<usize> = Vec::new();
    for i in 0..texts.len() {
        let claimed = kept
            .iter()
            .any(|&r| is_duplicate(&texts[i], &texts[r], threshold, method));
        if claimed {
            dropped.push(i);
        } else {
            kept.push(i);
        }
    }
    (kept, dropped)
}

/// The pair scoring spelled out on the public primitives — the same
/// thresholds the dedup core uses, re-derived here so a drift in the
/// core's pair semantics cannot hide behind a shared helper. Every
/// spelling folds its input FIRST (the core's normalization policy —
/// the crash the first fuzz run found: a naive lane on the raw texts
/// disagreed with the folded sweep the moment a corpus held any case).
fn is_duplicate(a: &str, b: &str, threshold: f64, method: DedupMethod) -> bool {
    match method {
        DedupMethod::SimHash => {
            let max_bits = ((1.0 - threshold) * 64.0).floor() as u32;
            near_dup_impl::fingerprint_hamming(
                u128::from(simhash64(&near_dup_impl::fold_text(a))),
                u128::from(simhash64(&near_dup_impl::fold_text(b))),
            ) <= max_bits
        }
        DedupMethod::Shingle => near_dup_impl::shingle_jaccard(a, b, 3) >= threshold,
        DedupMethod::MinHash => {
            let (sig_a, sig_b) = (
                tors::minhash_impl::signature(&near_dup_impl::fold_text(a), 128, 3, 0),
                tors::minhash_impl::signature(&near_dup_impl::fold_text(b), 128, 3, 0),
            );
            // The core's exact agreement form: matches >= ceil(threshold * k)
            // (the naive matches/k >= threshold is the same predicate
            // only when the product is not a float-rounding edge).
            let needed = (threshold * 128.0).ceil() as usize;
            sig_a.iter().zip(&sig_b).filter(|(x, y)| x == y).count() >= needed
        }
    }
}

fuzz_target!(|input: Input| {
    match input {
        Input::Pair { a, b, width } => {
            if a.len() > 16 * 1024 || b.len() > 16 * 1024 {
                return;
            }
            let (a, b) = (String::from_utf8_lossy(&a), String::from_utf8_lossy(&b));
            check_pair_contract(&a, &b, 1 + width as usize % 12);
        }
        Input::Mutated {
            data,
            flip_at,
            noise_seed,
        } => {
            if data.is_empty() || data.len() > 16 * 1024 {
                return;
            }
            let text = String::from_utf8_lossy(&data);
            // The diversity floor: skip degenerate inputs where the
            // collapse is real (the minhash target's guard).
            if tors::minhash_impl::distinct_shingle_count(&text, 3) < DIVERSITY_FLOOR {
                return;
            }
            // The one-byte mutation: xor 0x5A always changes the byte,
            // the position cycles the whole input.
            let mut flipped = data.clone();
            let at = flip_at as usize % data.len();
            flipped[at] ^= 0x5A;
            let mutated = String::from_utf8_lossy(&flipped);
            if tors::minhash_impl::distinct_shingle_count(&mutated, 3) < DIVERSITY_FLOOR {
                return;
            }
            let noise = lcg_text(noise_seed, 256);
            // The exact Jaccard locality invariant: a single edit of a
            // diverse text damages only the windows around it, so the
            // pair stays high (>= 0.5 with a >= 128x margin under the
            // (S-4)/(S+4) bound), and an independent text shares
            // nothing structured (<= 0.1).
            let near = near_dup_impl::shingle_jaccard(&text, &mutated, 3);
            let far = near_dup_impl::shingle_jaccard(&text, &noise, 3);
            assert!(
                near >= 0.5,
                "one-byte mutation scored {near:.3} (< 0.5) against its original"
            );
            assert!(
                far <= 0.1,
                "independent random text scored {far:.3} (> 0.1) against the input"
            );
        }
        Input::Exact { index, width } => {
            let text = EXACT_STRINGS[index as usize % EXACT_STRINGS.len()];
            check_pair_contract(text, text, 1 + width as usize % 12);
            // The identical-text dedup row: one group holding every
            // index, the first text kept.
            let corpus = vec![text.to_string(), text.to_string(), text.to_string()];
            let out = near_dup_impl::dedup_near_dup(&corpus, 1.0, DedupMethod::Shingle);
            assert_eq!(out.kept, vec![0]);
            assert_eq!(out.groups, vec![vec![0, 1, 2]]);
        }
        Input::Corpus {
            texts,
            threshold_high,
            threshold_low,
            method,
        } => {
            if texts.is_empty() || texts.len() > 32 {
                return;
            }
            if texts.iter().any(|t| t.len() > 4 * 1024) {
                return;
            }
            let texts: Vec<String> = texts
                .iter()
                .map(|t| String::from_utf8_lossy(t).into_owned())
                .collect();
            let method = [
                DedupMethod::SimHash,
                DedupMethod::Shingle,
                DedupMethod::MinHash,
            ][method as usize % 3];
            // Thresholds in [0, 1] with high >= low (the monotonicity
            // lane's ordering); low thresholds merge everything, which
            // is the contract, not a defect.
            let t_low = threshold_low as f64 / 255.0;
            let t_high = threshold_high as f64 / 255.0;
            let (low, high) = (t_low.min(t_high), t_low.max(t_high));

            let out = near_dup_impl::dedup_near_dup(&texts, high, method);
            // Determinism: the same corpus, the same outcome.
            let again = near_dup_impl::dedup_near_dup(&texts, high, method);
            assert_eq!(out.kept, again.kept, "dedup not deterministic");
            assert_eq!(out.dropped, again.dropped);
            assert_eq!(out.groups, again.groups);
            // The exact partition: kept + dropped == 0..n, group heads
            // == kept, heads are their group's minimum.
            let mut all: Vec<usize> = out.kept.iter().chain(&out.dropped).copied().collect();
            all.sort_unstable();
            assert_eq!(all, (0..texts.len()).collect::<Vec<_>>());
            assert_eq!(
                out.kept,
                out.groups.iter().map(|g| g[0]).collect::<Vec<_>>()
            );
            for g in &out.groups {
                assert_eq!(g[0], *g.iter().min().unwrap());
            }
            // Agreement with the naive from-scratch re-dedup (the exact
            // core-vs-spec pin), at BOTH thresholds — the sweep's pair
            // semantics must be threshold-monotone the way the pair
            // scoring is, which agreement at two distinct thresholds
            // exercises. (A corpus-level drop-count monotonicity pin
            // would over-claim: greedy keep-first re-associates chains,
            // so only pair-level scoring is exactly monotone in the
            // threshold — pinned in src/near_dup_impl.rs's tests.)
            for threshold in [low, high] {
                let (naive_kept, naive_dropped) = naive_dedup(&texts, threshold, method);
                let out_at = near_dup_impl::dedup_near_dup(&texts, threshold, method);
                assert_eq!(out_at.kept, naive_kept, "core/naive kept disagreement");
                assert_eq!(
                    out_at.dropped, naive_dropped,
                    "core/naive dropped disagreement"
                );
            }
        }
    }
});
