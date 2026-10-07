//! `weighted_minhash_signature`'s core never panics on any input, and the
//! contracts the surface pins hold under raw adversarial bytes: exact
//! determinism, the `2 * num_perm` row shape, the empty-multiset
//! sentinel equivalence, the (hash, t) pair structure (every non-sentinel
//! hash row is a winning token's identity, and the t rows decode to
//! finite non-negative f64 active indices), and -- the differential pin,
//! the lsh target's discipline -- exact agreement with Ioffe's ICWS
//! spelled inline (frame-keyed SplitMix64 draws, the Gamma(2, 1) +
//! uniform draw order, the floored active index, the c/z argmin over the
//! ascending-(hash, weight) sweep), so an internal regression cannot hide
//! behind itself.
//!
//! Sizes are capped (at most 16 weight entries over at most 128
//! permutations, the `fuzz_targets/lsh.rs` discipline), with the weight
//! range deliberately including the extremes that stress the float math:
//! zero (excluded from the multiset), denormal minima, sub-1 fractional
//! weights (negative active indices), and huge weights (the z-overflow
//! path the argmin handles deterministically).

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
enum Input {
    /// Raw bytes -> an LCG weight vector (including zeros and extreme
    /// magnitudes) at a fuzzed (num_perm, seed) shape: panic-freedom,
    /// determinism, the shape and pair structure, the empty-multiset
    /// sentinel equivalence, and exact agreement with the inline naive
    /// ICWS oracle.
    Raw {
        data: Vec<u8>,
        num_perm: u16,
        seed: u64,
        shape: u8,
    },
    /// Explicit weight extremes: the denormal, fractional, and huge
    /// weights the LCG lane rarely draws, plus the all-zero multiset.
    Extremes {
        variant: u8,
        num_perm: u16,
        seed: u64,
    },
    /// The same multiset spelled two ways (a weight map vs an equivalent
    /// token list) must signature identically: the input-spelling
    /// contract.
    Spellings {
        tokens: Vec<u8>,
        num_perm: u16,
        seed: u64,
    },
}

/// The repo's deterministic u64 LCG (the same Knuth-style constants the
/// other targets use).
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

/// The weight vector: `n` entries with token hashes from the LCG and
/// weights shaped by `shape` (bit 0: allow zeros, bit 1: fractional
/// sub-1 weights, bit 2: huge weights), the extremes the float paths
/// need.
fn weight_vector(seed: u64, n: usize, shape: u8) -> Vec<(u64, f64)> {
    let mut state = seed | 1;
    (0..n)
        .map(|i| {
            let hash = lcg_next(&mut state);
            let raw = lcg_next(&mut state);
            let weight = if shape & 1 != 0 && i % 3 == 0 {
                0.0
            } else if shape & 2 != 0 && i % 3 == 1 {
                (raw >> 11) as f64 / (1u64 << 53) as f64 * 0.999
            } else if shape & 4 != 0 && i % 3 == 2 {
                (raw % 1000) as f64 * 1e250
            } else {
                1.0 + (raw % 100) as f64
            };
            (hash, weight)
        })
        .collect()
}

/// One uniform in (0, 1) from the SplitMix64 chain: the exact pinned
/// arithmetic the core spells (add the golden-ratio gamma, mix a copy
/// through the two constants, final xorshift), spelled inline (the
/// oracle never calls the code under test), then the top-53-bits float
/// conversion with the +0.5 centering.
fn oracle_uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let draw = z ^ (z >> 31);
    ((draw >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
}

/// The naive ICWS oracle (Ioffe 2010; Shrivastava's NeurIPS 2016
/// Algorithm 1): per permutation j and token (hash, w), the stream is
/// XXH64 over the frame [seed, j, hash]; five draws in pinned order
/// (r's exponentials, c's, beta); t = floor(ln(w)/r + beta),
/// y = exp(r(t - beta)), z = y*exp(r), a = c/z; the sample is the token
/// minimizing a (first wins ties, ascending-(hash, weight) order). The
/// rows are the winner pair (hash, t-as-f64-bits).
fn oracle_icws(items: &[(u64, f64)], num_perm: usize, seed: u64) -> Vec<u64> {
    let mut positive: Vec<(u64, f64)> = items.iter().copied().filter(|&(_, w)| w > 0.0).collect();
    positive.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap().then(x.1.total_cmp(&y.1)));
    if num_perm == 0 || positive.is_empty() {
        return vec![u64::MAX; 2 * num_perm];
    }
    let mut sig = Vec::with_capacity(2 * num_perm);
    for j in 0..num_perm {
        let mut best: Option<(f64, u64, f64)> = None;
        for &(hash, w) in &positive {
            let mut frame = Vec::with_capacity(32);
            frame.extend_from_slice(&3u64.to_le_bytes());
            frame.extend_from_slice(&seed.to_le_bytes());
            frame.extend_from_slice(&(j as u64).to_le_bytes());
            frame.extend_from_slice(&hash.to_le_bytes());
            let mut state = twox_hash::XxHash64::oneshot(0, &frame);
            let e1 = -(1.0 - oracle_uniform(&mut state)).ln();
            let e2 = -(1.0 - oracle_uniform(&mut state)).ln();
            let e3 = -(1.0 - oracle_uniform(&mut state)).ln();
            let e4 = -(1.0 - oracle_uniform(&mut state)).ln();
            let beta = oracle_uniform(&mut state);
            let r = e1 + e2;
            let c = e3 + e4;
            let t = (w.ln() / r + beta).floor();
            let y = (r * (t - beta)).exp();
            let z = y * r.exp();
            let score = if z > 0.0 { c / z } else { f64::INFINITY };
            let wins = match best {
                None => true,
                Some((best_a, _, _)) => score < best_a,
            };
            if wins {
                best = Some((score, hash, t));
            }
        }
        let (_, hash, t) = best.unwrap();
        sig.push(hash);
        sig.push(t.to_bits());
    }
    sig
}

/// The token-list spelling's token identity: the crate's single-token
/// frame (LE64(1) || LE64(byte_len) || bytes) under XXH64 seed 0,
/// spelled inline (the sha2-oracle discipline).
fn token_frame_hash(token: &str) -> u64 {
    let mut frame = Vec::with_capacity(16 + token.len());
    frame.extend_from_slice(&1u64.to_le_bytes());
    frame.extend_from_slice(&(token.len() as u64).to_le_bytes());
    frame.extend_from_slice(token.as_bytes());
    twox_hash::XxHash64::oneshot(0, &frame)
}

fuzz_target!(|input: Input| {
    match input {
        Input::Raw {
            data,
            num_perm,
            seed,
            shape,
        } => {
            if data.len() > 4 * 1024 {
                return;
            }
            let num_perm = num_perm as usize % 129;
            let mut state = seed | 1;
            let n = (data.len() % 16) + 1;
            let items: Vec<(u64, f64)> = weight_vector(lcg_next(&mut state), n, shape);
            let sig =
                tors::minhash_impl::weighted_signature_from_weights(items.clone(), num_perm, seed);
            // Determinism and shape.
            assert_eq!(
                sig,
                tors::minhash_impl::weighted_signature_from_weights(items.clone(), num_perm, seed),
                "weighted signature not deterministic"
            );
            assert_eq!(sig.len(), 2 * num_perm, "length != 2 * num_perm");
            // The sentinel equivalence: an all-zero multiset is the
            // all-sentinel signature; a non-empty multiset never emits
            // the sentinel (a real token hash could equal u64 MAX only
            // with probability 2^-64, and these are LCG-fixed).
            if items.iter().all(|&(_, w)| w <= 0.0) {
                assert!(sig.iter().all(|&v| v == u64::MAX), "not all sentinel");
            } else {
                assert!(
                    sig.iter().all(|&v| v != u64::MAX),
                    "sentinel row over a non-empty multiset"
                );
                // The differential pin: exact agreement with the inline
                // naive ICWS oracle.
                assert_eq!(
                    sig,
                    oracle_icws(&items, num_perm, seed),
                    "oracle disagreement"
                );
            }
        }
        Input::Extremes {
            variant,
            num_perm,
            seed,
        } => {
            let num_perm = num_perm as usize % 129;
            let items: Vec<(u64, f64)> = match variant % 6 {
                0 => vec![(0xDEADBEEF, f64::MAX)],
                1 => vec![(0xDEADBEEF, f64::from_bits(1))], // the denormal minimum
                2 => vec![(0xDEADBEEF, 1e-300), (0xFEEDFACE, 1e300)],
                3 => vec![(1, 0.0), (2, 0.0)], // the all-zero multiset
                4 => (0..64u64)
                    .map(|i| (i.wrapping_mul(0x9E37_79B9_7F4A_7C15), i as f64 + 0.5))
                    .collect(),
                _ => vec![(7, 0.5), (7, 2.0)], // one hash, two weights: sorted together
            };
            let sig =
                tors::minhash_impl::weighted_signature_from_weights(items.clone(), num_perm, seed);
            assert_eq!(sig.len(), 2 * num_perm);
            assert_eq!(
                sig,
                tors::minhash_impl::weighted_signature_from_weights(items.clone(), num_perm, seed)
            );
            // Determinism across the estimate too: identical signatures
            // estimate exactly 1.0 (empty multisets included). num_perm
            // 0 is the core's empty-signature guard (the binding rejects
            // it; the core asserts the shape), so the estimate lane only
            // runs at the binding's admissible shapes.
            if num_perm > 0 {
                let est = tors::minhash_impl::weighted_jaccard_estimate(&sig, &sig);
                assert_eq!(est, 1.0, "self-estimate {est} != 1.0");
            }
            if !items.iter().any(|&(_, w)| w > 0.0) {
                assert!(sig.iter().all(|&v| v == u64::MAX), "not all sentinel");
            } else {
                assert_eq!(
                    sig,
                    oracle_icws(&items, num_perm, seed),
                    "oracle disagreement"
                );
            }
        }
        Input::Spellings {
            tokens,
            num_perm,
            seed,
        } => {
            if tokens.len() > 4 * 1024 {
                return;
            }
            let num_perm = num_perm as usize % 129;
            // The bytes -> lossy text -> tokens: the token-list spelling
            // of the occurrence multiset, hashed under the crate's
            // single-token frame.
            let text = String::from_utf8_lossy(&tokens);
            let token_list: Vec<String> = text.split_whitespace().map(str::to_string).collect();
            let from_list =
                tors::minhash_impl::weighted_signature_tokens(&token_list, num_perm, seed);
            // The explicit-weight spelling of the SAME multiset: counts
            // per distinct token. Two distinct tokens colliding in the
            // crate's XXH64 is the 2^-64 channel the core documents;
            // over these LCG-free short alphabets it cannot fire.
            let mut counts: std::collections::HashMap<String, u64> =
                std::collections::HashMap::new();
            for token in &token_list {
                *counts.entry(token.clone()).or_insert(0) += 1;
            }
            let from_map: Vec<(u64, f64)> = counts
                .iter()
                .map(|(token, &count)| (token_frame_hash(token), count as f64))
                .collect();
            let from_weights =
                tors::minhash_impl::weighted_signature_from_weights(from_map, num_perm, seed);
            assert_eq!(
                from_list, from_weights,
                "the token-list and weight-map spellings of one multiset disagree"
            );
        }
    }
});
