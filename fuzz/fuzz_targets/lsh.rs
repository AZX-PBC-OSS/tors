//! `lsh_candidates`' core never panics on any input, and the contracts
//! the surface pins hold under raw adversarial bytes: exact determinism,
//! the ascending `i < j` deduplicated pair shape, identical signatures
//! always paired, and the permutation-invariance contract (permuting the
//! input permutes the same candidate relation). The differential pin is
//! the naive banding oracle recomputed inline (bucket per band, within-
//! bucket pairs, sort-dedup): small shapes make the quadratic oracle
//! cheap, so every fuzz case checks the core against it exactly.
//!
//! Sizes are capped (at most 48 signatures of at most 64 rows, the
//! `fuzz_targets/minhash.rs` discipline) so the fuzzer explores deep
//! small shapes instead of stalling on huge ones. The formula lane
//! (`lsh_probability`) pins the S-curve against the same
//! `1 - (1 - s^r)^b` expression spelled inline and monotonicity in `s`
//! at the fuzzed point; `lsh_threshold`'s one-liner rides the same lane.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
enum Input {
    /// Raw bytes -> an LCG signature corpus at a fuzzed (bands, rows)
    /// shape: panic-freedom, determinism, the pair shape, the identical-
    /// signature rule, and exact agreement with the naive oracle.
    Raw { data: Vec<u8>, shape: u8, seed: u64 },
    /// The same corpus with a fuzzed pairwise swap applied: the permuted
    /// call's pair set must be the original's relabeled by the swap.
    Permuted { data: Vec<u8>, shape: u8, swap: u8 },
    /// The S-curve lane: a fuzzed `s` in [0, 1] at a fuzzed shape, pinned
    /// against the inline formula, monotone in `s`, exact at the ends.
    Formula { s_bits: u32, shape: u8 },
}

/// The repo's deterministic u64 LCG (the same Knuth-style constants
/// `fuzz_targets/minhash.rs` uses), over u64 rows directly.
fn lcg_rows(seed: u64, count: usize) -> Vec<u64> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        })
        .collect()
}

/// The fuzzed shape: bands 1..=8, rows 1..=8 (num_perm at most 64).
fn shape_of(shape: u8) -> (usize, usize) {
    (1 + (shape as usize) % 8, 1 + ((shape as usize) / 8) % 8)
}

/// The corpus: at most 48 signatures over `bands * rows` rows, every 7th
/// a duplicate of its predecessor (live multi-member buckets, so the
/// pair-emission paths actually run).
fn corpus(seed: u64, n: usize, num_perm: usize) -> Vec<Vec<u64>> {
    let rows = lcg_rows(seed, n * num_perm);
    let mut sigs: Vec<Vec<u64>> = Vec::with_capacity(n);
    for i in 0..n {
        if i % 7 == 1 {
            sigs.push(sigs[i - 1].clone());
        } else {
            sigs.push(rows[i * num_perm..(i + 1) * num_perm].to_vec());
        }
    }
    sigs
}

/// The full candidate contract every non-formula lane asserts.
fn check_candidates(sigs: &[Vec<u64>], bands: usize, rows: usize) {
    let out = tors::lsh_impl::lsh_candidates(sigs, bands, rows);
    // Determinism.
    assert_eq!(
        out,
        tors::lsh_impl::lsh_candidates(sigs, bands, rows),
        "lsh_candidates not deterministic"
    );
    // The shape contract: ascending, i < j, deduplicated.
    assert!(
        out.pairs.windows(2).all(|w| w[0] < w[1]),
        "pairs not strictly ascending: {:?}",
        out.pairs
    );
    assert!(out.pairs.iter().all(|&(i, j)| i < j), "pair with i >= j");
    // Identical signatures are always paired (the S-curve's s = 1 end).
    for a in 0..sigs.len() {
        for b in a + 1..sigs.len() {
            if sigs[a] == sigs[b] {
                assert!(
                    out.pairs.contains(&(a, b)),
                    "identical signatures {a},{b} not candidates"
                );
            }
        }
    }
    // The differential pin: the naive banding oracle, spelled inline.
    let mut expected: Vec<(usize, usize)> = Vec::new();
    for band in 0..bands {
        let mut keys: Vec<(u64, usize)> = sigs
            .iter()
            .enumerate()
            .map(|(idx, sig)| (band_key(&sig[band * rows..(band + 1) * rows]), idx))
            .collect();
        keys.sort_unstable();
        for group in keys.chunk_by(|a, b| a.0 == b.0) {
            for a in 0..group.len() {
                for b in a + 1..group.len() {
                    let (i, j) = (group[a].1.min(group[b].1), group[a].1.max(group[b].1));
                    expected.push((i, j));
                }
            }
        }
    }
    expected.sort_unstable();
    expected.dedup();
    assert_eq!(out.pairs, expected, "core disagrees with the naive oracle");
}

/// The band key the oracle buckets on: XXH64 seed 0 over the
/// length-prefixed little-endian frame (LE64 row count, then LE64 per
/// row), the core's documented contract
/// (`minhash_impl::hash_u64_frame`). Spelled INLINE over the same
/// twox-hash pin, the sha2-oracle discipline: the oracle never calls the
/// code under test, so an internal hash regression cannot hide behind
/// itself. The framing is pinned separately in `src/lsh_impl.rs`'s unit
/// tests against the manual XXH64 spelling.
fn band_key(rows: &[u64]) -> u64 {
    let mut frame = Vec::with_capacity(8 + rows.len() * 8);
    frame.extend_from_slice(&(rows.len() as u64).to_le_bytes());
    for row in rows {
        frame.extend_from_slice(&row.to_le_bytes());
    }
    twox_hash::XxHash64::oneshot(0, &frame)
}

fuzz_target!(|input: Input| {
    match input {
        Input::Raw { data, shape, seed } => {
            if data.len() > 4 * 1024 {
                return;
            }
            let (bands, rows) = shape_of(shape);
            let n = 48.min(data.len() / 8);
            let sigs = corpus(seed, n, bands * rows);
            check_candidates(&sigs, bands, rows);
        }
        Input::Permuted { data, shape, swap } => {
            if data.len() > 4 * 1024 {
                return;
            }
            let (bands, rows) = shape_of(shape);
            let n = 48.min(data.len() / 8);
            if n < 2 {
                return;
            }
            let sigs = corpus(seed_of(&data), n, bands * rows);
            let original = tors::lsh_impl::lsh_candidates(&sigs, bands, rows);
            // One pairwise swap: a relabeling the pair relation must
            // follow exactly (the inverse of the swap itself).
            let a = (swap as usize) % n;
            let b = ((swap as usize) / n) % n;
            let mut permuted = sigs.clone();
            permuted.swap(a, b);
            let moved = tors::lsh_impl::lsh_candidates(&permuted, bands, rows);
            let relabel = |q: usize| {
                if q == a {
                    b
                } else if q == b {
                    a
                } else {
                    q
                }
            };
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
            assert_eq!(
                moved.pairs, expected,
                "swap ({a}, {b}) did not relabel the candidate relation"
            );
        }
        Input::Formula { s_bits, shape } => {
            let (bands, rows) = shape_of(shape);
            let s = f64::from(s_bits) / f64::from(u32::MAX);
            let p = tors::lsh_impl::lsh_probability(s, bands, rows);
            // The inline formula spelling. The margin is the pytest grid
            // pin's (tests/test_lsh.py): rel=1e-12 (libm pow is not
            // required correctly rounded, and the core evaluates through
            // powf's exp/log path where this oracle uses powi's repeated
            // squaring) plus an abs=1e-15 floor — at small p the final
            // `1 - (...)` cancellation leaves an absolute error of a few
            // ulps of 1.0, which no relative bound can hold. The contract
            // is the curve, not powi's rounding.
            let expected = 1.0 - (1.0 - s.powi(rows as i32)).powi(bands as i32);
            assert!(
                (p - expected).abs() <= 1e-12 * expected.abs() + 1e-15,
                "lsh_probability({s}, b={bands}, r={rows}) = {p}, formula says {expected}"
            );
            // Monotonicity in s (a step of one s-bit upward).
            let s_up = f64::from(s_bits.saturating_add(1)) / f64::from(u32::MAX);
            let p_up = tors::lsh_impl::lsh_probability(s_up, bands, rows);
            assert!(
                p_up >= p,
                "S-curve not monotone at s={s} b={bands} r={rows}"
            );
            // The ends are exact.
            assert_eq!(tors::lsh_impl::lsh_probability(0.0, bands, rows), 0.0);
            assert_eq!(tors::lsh_impl::lsh_probability(1.0, bands, rows), 1.0);
            // The threshold one-liner: (1/b)^(1/r), inside the S-curve's
            // near-midpoint band.
            let t = tors::lsh_impl::lsh_threshold(bands, rows);
            let expected_t = (1.0 / bands as f64).powf(1.0 / rows as f64);
            assert!(
                (t - expected_t).abs() <= f64::EPSILON * expected_t.abs(),
                "lsh_threshold(b={bands}, r={rows}) = {t}, formula says {expected_t}"
            );
            let p_t = tors::lsh_impl::lsh_probability(t, bands, rows);
            // The near-midpoint band is the approximation's claim for
            // b >= 2; at one band the threshold degenerates to 1.0 and
            // the S-curve there is exactly the certain candidate.
            if bands >= 2 {
                assert!(
                    (0.2..=0.9).contains(&p_t),
                    "threshold b={bands} r={rows} sits at P={p_t}, outside the near-midpoint band"
                );
            } else {
                assert_eq!(p_t, 1.0, "one-band threshold is s = 1, P = 1");
            }
        }
    }
});

/// The corpus seed from the raw bytes: the first 8 bytes little-endian
/// (short inputs zero-pad — every bytestring maps, none panics), the
/// `fuzz_targets/minhash.rs` seed_from_bytes discipline.
fn seed_of(data: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let n = data.len().min(8);
    buf[..n].copy_from_slice(&data[..n]);
    u64::from_le_bytes(buf)
}
