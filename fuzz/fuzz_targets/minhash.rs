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
    /// equivalence. `num_perm` spans 0..=1024: 0 exercises the core's
    /// empty-signature guard (the binding rejects it; the core must still
    /// never panic) and the top of the binding range rides the full
    /// coefficient stream; `shingle_size` spans 0..=1029: 0 exercises the
    /// core's malformed-width guard (the binding rejects it; the core
    /// must still never panic) and the large end the short-of-a-window
    /// sentinel (through the wide-window count-first path at the top).
    Raw {
        data: Vec<u8>,
        num_perm: u16,
        shingle_size: u16,
        seed: u64,
    },
    /// The seed as arbitrary bytes (any length, including empty and
    /// over-long): the byte->u64 reduction plus the same contract the Raw
    /// lane asserts, so seed handling is fuzzed as bytes, not just as a
    /// ready-made u64.
    SeedBytes {
        data: Vec<u8>,
        num_perm: u16,
        shingle_size: u16,
        seed_bytes: Vec<u8>,
    },
    /// Exact adversarial strings (the WB4 separator-attach corpus, ZWJ
    /// emoji, CJK, regional indicators) at fuzzed parameters: the inputs
    /// a byte soup almost never assembles, pinned exactly.
    Exact {
        index: u8,
        num_perm: u16,
        shingle_size: u16,
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

/// The exact-string lane's corpus: every row a segmentation or framing
/// edge (see `src/minhash_impl.rs`'s WB4 note and
/// `tests/test_minhash.py`'s adjacency corpus).
const EXACT_STRINGS: &[&str] = &[
    "\u{1f}\u{301}",
    "\u{1f}\u{200d}",
    "\u{1f}\u{ad}",
    "a \u{1f}\u{301} b c",
    "A\u{1f}\u{301}b",
    "\u{1f}",
    "\u{1f}\u{1f}\u{1f}",
    "a\u{1f}b c",
    "\u{1f469}\u{200d}\u{1f52c} \u{30c6}\u{30b9}\u{30c8}",
    "\u{1f1fa}\u{1f1f8}\u{1f1fa}",
    "\u{1100}\u{1161}\u{11a8} \u{e0}\u{30d}",
    "caf\u{e9} soci\u{e9}t\u{e9} na\u{ef}ve \u{6771}\u{4eac}\u{306f}\u{65e5}\u{672c}",
    "Hello, world! One. Two.",
    "   \t  ",
    "",
];

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

/// The shared panic-freedom/determinism/length/value-shape contract
/// every non-mutated lane asserts.
fn check_signature_contract(text: &str, num_perm: usize, shingle_size: usize, seed: u64) {
    // Determinism: the pinned arithmetic is platform-independent,
    // so the same call twice must agree element for element.
    let sig = tors::minhash_impl::signature(text, num_perm, shingle_size, seed);
    assert_eq!(
        sig,
        tors::minhash_impl::signature(text, num_perm, shingle_size, seed),
        "signature not deterministic"
    );
    // The length contract.
    assert_eq!(sig.len(), num_perm, "signature length != num_perm");
    // The value contract: every element is either an affine output
    // (below the Mersenne prime) or the empty-set sentinel, and
    // the sentinel appears exactly when the shingle set is empty.
    let distinct = tors::minhash_impl::distinct_shingle_count(text, shingle_size);
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

/// Arbitrary seed bytes to the u64 the core takes: the first 8 bytes
/// little-endian (short inputs zero-pad, long inputs truncate — every
/// bytestring maps, none panics). A shape-only probe: it exercises the
/// byte→u64 reduction plumbing, not the binding's `__index__`/bool
/// contract (pinned separately in `tests/test_minhash.py`).
fn seed_from_bytes(seed_bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let n = seed_bytes.len().min(8);
    buf[..n].copy_from_slice(&seed_bytes[..n]);
    u64::from_le_bytes(buf)
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
            let num_perm = num_perm as usize % 1025;
            let shingle_size = shingle_size as usize % 1030;
            check_signature_contract(&text, num_perm, shingle_size, seed);
        }
        Input::SeedBytes {
            data,
            num_perm,
            shingle_size,
            seed_bytes,
        } => {
            if data.len() > 16 * 1024 || seed_bytes.len() > 16 {
                return;
            }
            let text = String::from_utf8_lossy(&data);
            let num_perm = num_perm as usize % 1025;
            let shingle_size = shingle_size as usize % 1030;
            check_signature_contract(&text, num_perm, shingle_size, seed_from_bytes(&seed_bytes));
        }
        Input::Exact {
            index,
            num_perm,
            shingle_size,
            seed,
        } => {
            let text = EXACT_STRINGS[index as usize % EXACT_STRINGS.len()];
            let num_perm = num_perm as usize % 1025;
            let shingle_size = shingle_size as usize % 1030;
            check_signature_contract(text, num_perm, shingle_size, seed);
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
