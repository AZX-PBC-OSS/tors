//! `minhash_signature`'s core never panics on any input, and the two
//! contracts the surface pins hold under raw adversarial bytes: exact
//! determinism with the length/value shape (every element an affine
//! output below 2^61 - 1, or the whole signature the u64 MAX sentinel
//! exactly when the shingle set is empty), and the locality property
//! MinHash exists for — a one-byte mutation of a shingle-DIVERSE document
//! (a single edit damages at most the shingle windows around it, so the
//! shingle-set Jaccard stays high) agrees with the original far more
//! than an independent random text does.
//!
//! The diversity floor is the load-bearing guard: a repeated-token
//! document has thousands of tokens but ONE distinct shingle, and a
//! single edit there really does destroy most of its (degenerate)
//! shingle set — the estimate correctly collapses, so the invariant is
//! only asserted above 256 distinct shingles, where the worst legal
//! edit (a whitespace flip splitting or merging two tokens at
//! shingle_size 8) leaves J >= (256 - 18)/(256 + 18) ~= 0.87, four
//! sigma above the 0.5 threshold at 64 permutations. The random side is
//! a full-range LCG byte stream (1024 bytes, lossily decoded): its
//! shingle set shares nothing structured with any input, so its
//! agreement sits at the ~0 floor, far under the 0.1 ceiling.
//!
//! Sizes are capped (the sweep is O(shingles x num_perm)), the
//! `fuzz_targets/diff.rs` discipline, so the fuzzer explores deep small
//! shapes instead of stalling on huge ones.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

/// The invariant lane's permutation count: enough concentration that
/// four sigma of estimator noise stays inside the thresholds.
const INVARIANT_PERMS: usize = 64;
/// The distinct-shingle floor the locality invariant is asserted above
/// (the module doc's diversity guard).
const INVARIANT_FLOOR: usize = 256;

#[derive(Arbitrary, Debug)]
enum Input {
    /// Arbitrary raw text and parameters: panic-freedom, determinism,
    /// the length/value shape, and the empty-set <-> sentinel
    /// equivalence.
    Raw {
        data: Vec<u8>,
        num_perm: u8,
        shingle_size: u8,
        seed: u64,
    },
    /// The input vs a one-byte flip of itself vs an independent
    /// LCG-random text: the mutation-agreement locality invariant.
    Mutated {
        data: Vec<u8>,
        flip_at: u8,
        noise_seed: u64,
        shingle_size: u8,
    },
}

fn agreement_fraction(a: &[u64], b: &[u64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len() as f64
}

/// The repo's deterministic u64 LCG (the same Knuth-style constants
/// `tests/reference.py`'s corpus builders use), full-range bytes: an
/// independent random text whose shingle set shares nothing structured
/// with a fuzz input.
fn lcg_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 56) as u8
        })
        .collect()
}

fuzz_target!(|input: Input| {
    match input {
        Input::Raw {
            data,
            num_perm,
            shingle_size,
            seed,
        } => {
            if data.len() > 16 * 1024 {
                return;
            }
            let text = String::from_utf8_lossy(&data);
            let num_perm = 1 + num_perm as usize % 256;
            let shingle_size = 1 + shingle_size as usize % 8;
            // Determinism: the pinned arithmetic is platform-independent,
            // so the same call twice must agree element for element.
            let sig = tors::minhash_impl::signature(&text, num_perm, shingle_size, seed);
            assert_eq!(
                sig,
                tors::minhash_impl::signature(&text, num_perm, shingle_size, seed),
                "signature not deterministic"
            );
            // The length contract.
            assert_eq!(sig.len(), num_perm, "signature length != num_perm");
            // The value contract: every element is either an affine output
            // (below the Mersenne prime) or the empty-set sentinel, and
            // the sentinel appears exactly when the shingle set is empty.
            let distinct = tors::minhash_impl::distinct_shingle_count(&text, shingle_size);
            if distinct == 0 {
                assert!(
                    sig.iter().all(|&v| v == u64::MAX),
                    "empty shingle set but not the all-sentinel signature"
                );
            } else {
                assert!(
                    sig.iter().all(|&v| v < (1 << 61) - 1),
                    "signature element outside the affine range"
                );
            }
        }
        Input::Mutated {
            data,
            flip_at,
            noise_seed,
            shingle_size,
        } => {
            if data.is_empty() || data.len() > 16 * 1024 {
                return;
            }
            let shingle_size = 1 + shingle_size as usize % 8;
            let text = String::from_utf8_lossy(&data);
            // The one-byte mutation: xor 0x5A always changes the byte, and
            // the position cycles the whole input.
            let at = flip_at as usize % data.len();
            let mut flipped = data.clone();
            flipped[at] ^= 0x5A;
            let mutated = String::from_utf8_lossy(&flipped);
            // The independent random text.
            let noise_bytes = lcg_bytes(noise_seed, 1024);
            let noise = String::from_utf8_lossy(&noise_bytes);
            // The diversity floor: skip degenerate inputs (the module
            // doc's guard), on either side of the mutation.
            if tors::minhash_impl::distinct_shingle_count(&text, shingle_size) < INVARIANT_FLOOR
                || tors::minhash_impl::distinct_shingle_count(&mutated, shingle_size)
                    < INVARIANT_FLOOR
            {
                return;
            }
            let base = tors::minhash_impl::signature(&text, INVARIANT_PERMS, shingle_size, 0);
            let near = tors::minhash_impl::signature(&mutated, INVARIANT_PERMS, shingle_size, 0);
            let far = tors::minhash_impl::signature(&noise, INVARIANT_PERMS, shingle_size, 0);
            let near_agreement = agreement_fraction(&base, &near);
            let far_agreement = agreement_fraction(&base, &far);
            // The locality invariant, with the module doc's margins: a
            // single edit of a diverse document is a high-Jaccard pair
            // (> 0.5 at four sigma), an independent random text is a
            // zero-Jaccard pair (< 0.1).
            assert!(
                near_agreement > 0.5,
                "one-byte mutation agreed only {near_agreement:.3} (< 0.5) with its original"
            );
            assert!(
                far_agreement < 0.1,
                "independent random text agreed {far_agreement:.3} (> 0.1) with the input"
            );
        }
    }
});
