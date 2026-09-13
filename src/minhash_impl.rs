//! MinHash: the pure-Rust core of `tors.minhash_signature`, the recall-side
//! near-duplicate complement to the SimHash family. `simhash64`/
//! `simhash128` are the precision side: one compact fingerprint whose
//! Hamming distance grows slowly with edit distance, cheap to store and
//! compare, but a weak recall instrument at corpus scale (two paraphrased
//! documents sit far apart in Hamming space however related their
//! vocabulary). MinHash is the recall side: a signature of `num_perm`
//! independent min-hashes whose agreement fraction is an unbiased
//! estimator of the Jaccard similarity of the two documents' shingle sets
//! (Broder's MinHash, the primitive behind near-dup detection at web
//! scale — Broder, Andrejczuk, and Dharmanandan's stage-2 shingles at
//! AltaVista, WWW 1997, and Broder's "Syntactic clustering of the Web"
//! report). A 100k-document ingestion pipeline bands the signatures
//! (LSH) and recalls every pair above a similarity threshold; the LSH
//! table itself is caller state — tors stays stateless (docs/design.md's
//! scope cut), and a banding helper is a future question, not a hidden
//! one inside this core.
//!
//! # The estimator contract
//!
//! For two documents' shingle sets `A` and `B`, the fraction of
//! signature positions on which two signatures agree estimates the
//! Jaccard index `J(A, B) = |A ∩ B| / |A ∪ B|`: each position `i` agrees
//! exactly when the shingle achieving the minimum of `h_i` over `A ∪ B`
//! lies in `A ∩ B`, which happens with probability `J`. The standard
//! error of the estimate over `k = num_perm` positions is
//! `sqrt(J(1 - J) / k)` — `O(1/sqrt(k))`, about 0.044 at `k = 128` and
//! the worst case `J = 0.5`, halving with every 4x in `num_perm`. The
//! `h_i` are affine maps over the Mersenne prime field (below), a
//! 2-wise-independent family that approximates the min-wise
//! independence the exact statement needs with negligible residual bias
//! (measured below 0.003 over 2280 controlled pairs, the same battery
//! that compared the candidate shingle hashes).
//!
//! # Tokenization and shingling
//!
//! Tokens are the crate's one word tokenizer — the same UAX #29
//! real-word-segment walk, lowercased, that `tf_idf`/`bm25_rank` ride
//! (`tokenize_impl::normalized_word_tokens` with every knob off): UAX #29
//! word segments, segments made entirely of whitespace skipped, each
//! lowercased with Unicode-correct `str::to_lowercase`. Word shingles
//! (not character shingles) are the recall-side unit because natural-text
//! near-duplicates preserve word sequence far more often than they
//! preserve exact character spans — a paragraph reflowed or a word
//! swapped shifts character k-grams wholesale while word k-grams survive.
//! A shingle is `shingle_size` consecutive tokens joined with U+001F (the
//! ASCII unit separator); the join is injective because no UAX #29 token
//! can contain U+001F (a C0 control is its own word segment), so two
//! distinct token windows never hash the same bytes.
//!
//! # The pinned arithmetic (determinism contract)
//!
//! Every element is fixed, documented arithmetic, platform-independent
//! (integer ops only, no floats, no per-process state), so the same text
//! at the same parameters produces the identical signature across
//! processes, versions, and machines — the same stability requirement
//! `simhash_impl` states for its FNV-1a (a fingerprint that changes
//! between runs breaks every cross-run dedupe built on it):
//!
//! - the shingle hash `x` is XXH64 with seed 0 (the frozen-spec
//!   algorithm via twox-hash; see the Cargo.toml dependency note for the
//!   measured why, and the tests below for the reference vectors);
//! - `p = 2^61 - 1`, the Mersenne prime, the standard MinHash field;
//! - the `(a_i, b_i)` pairs derive from `seed` by a SplitMix64 stream
//!   (Steele/Marsaglia's fixed arithmetic): state starts at `seed`
//!   reduced mod 2^64, and for each permutation two draws are taken,
//!   `a_i` first: `a_i` in `[1, p-1]` (`a_i = 0` would collapse the
//!   permutation to a constant, so zero is excluded) and `b_i` in
//!   `[0, p-1)`. `rand` is deliberately not involved: it is not on
//!   main's dependency tree, and its stream internals are not a
//!   semver-stable contract — the SplitMix64 arithmetic is ~8 lines,
//!   fixture-grade deterministic, and golden-pinned on both the Rust and
//!   Python sides (tests below, `tests/reference.py`,
//!   `tests/test_minhash.py`);
//! - `h_i(x) = (a_i * x + b_i) mod p`, the 128-bit product reduced by
//!   the shift-add Mersenne reduction (three fold rounds plus one
//!   conditional subtract, exact for every input below 2^126; the
//!   `v == p` fixed point of the naive fold loop is why the reduction is
//!   spelled in closed rounds — pinned against a direct `%` oracle in
//!   the tests);
//! - `signature[i] = min` over the document's shingle hashes.
//!
//! This is fixture-grade determinism, not cryptography: nothing here
//! resists an adversary crafting collisions, and nothing needs to (the
//! signature compares documents a caller already holds).
//!
//! # The empty-shingle-set convention
//!
//! Empty text, whitespace-only text, or fewer tokens than
//! `shingle_size` means no shingles, and the signature is `num_perm`
//! copies of the u64 MAX sentinel (2^64 - 1): deterministic, seed-
//! invariant, and unable to collide with any real minimum (the affine
//! outputs live in `[0, 2^61 - 1)`, a disjoint range). Two empty
//! documents agree at every position and two empty-vs-nonempty pairs at
//! none, so the estimator's degenerate cases stay consistent; and an
//! all-sentinel signature is a stable digest an LSH table can bucket
//! empty documents under.
//!
//! # GIL model
//!
//! The pyo3 wrapper (src/py/minhash.rs) borrows the text and validates
//! the bounds under the GIL, then runs the whole tokenize + shingle +
//! hash + min-sweep under one `py.detach`, and marshals the
//! `Vec<u64>` to a `num_perm`-element int list after (O(k), k <= 1024 —
//! the `diff_opcodes` list-marshalling class at a far smaller count). No
//! `aio` twin: a fast one-shot call (hundreds of milliseconds at the
//! whole-file sizes, single-digit at document scale) gains nothing from
//! a thread dispatch.

use std::hash::Hasher as _;

use twox_hash::XxHash64;

use crate::tokenize_impl::normalized_word_tokens;

/// The Mersenne prime the affine permutations live over: p = 2^61 - 1.
const MERSENNE_P: u64 = (1 << 61) - 1;

/// The shingle join separator: U+001F, the ASCII unit separator (the
/// module doc's injectivity note).
const SHINGLE_SEP: &[u8] = b"\x1f";

/// `x mod (2^61 - 1)` for `x < 2^126`, the shift-add Mersenne reduction:
/// three fold rounds `(x & p) + (x >> 61)` (each folds the top bits into
/// the bottom 61 without changing the residue mod `2^61 - 1`, since
/// `2^61 ≡ 1`), then one conditional subtract. The closed-round spelling
/// exists because the naive `while x >= p { x = (x & p) + (x >> 61) }`
/// loop never terminates on `x == p` (a fixed point: `p & p + p >> 61` is
/// `p` again) — see `reduce_matches_direct_modulo` below for the edge
/// battery, including that exact value.
fn mersenne_mod(x: u128) -> u64 {
    let m: u128 = u128::from(MERSENNE_P);
    // x < 2^126 -> r1 < 2^65 + 2^61 -> r2 < 2^61 + 18 -> r3 <= p + 1.
    let mut r = (x & m) + (x >> 61);
    r = (r & m) + (r >> 61);
    r = (r & m) + (r >> 61);
    if r >= m {
        r -= m;
    }
    debug_assert!(r < m, "Mersenne reduction out of range");
    r as u64
}

/// One SplitMix64 step: advance the state by the golden-ratio gamma mod
/// 2^64 and return the mixed copy of the new state. The fixed arithmetic
/// (gamma and both mixer constants) is the standard published SplitMix64
/// and the pinned derivation the coefficient stream rides.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The `(a_i, b_i)` pairs for `num_perm` permutations from `seed` (already
/// reduced mod 2^64 by the caller): two SplitMix64 draws per permutation,
/// `a_i` first, `a_i = draw % (p - 1) + 1` in `[1, p-1]`, `b_i = draw % p`
/// in `[0, p-1)`. The draws happen in permutation order, so a shorter
/// coefficient list is a prefix of a longer one at the same seed (the
/// prefix property pinned by the tests).
fn coefficients(num_perm: usize, seed: u64) -> Vec<(u64, u64)> {
    let mut state = seed;
    let mut pairs = Vec::with_capacity(num_perm);
    for _ in 0..num_perm {
        let a = splitmix64(&mut state) % (MERSENNE_P - 1) + 1;
        let b = splitmix64(&mut state) % MERSENNE_P;
        pairs.push((a, b));
    }
    pairs
}

/// The XXH64 (seed 0) of one shingle window: each token's UTF-8 bytes
/// fed to the streaming hasher with the U+001F separator between them —
/// byte-identical to hashing the joined `String` (XXH64 is a streaming
/// algorithm: the digest depends only on seed and byte sequence),
/// without materializing a per-shingle allocation. The one spelling both
/// the signature sweep and the distinct-shingle counter ride, so the
/// shingle-hash contract cannot drift between them.
fn hash_window(window: &[String]) -> u64 {
    let mut hasher = XxHash64::default();
    for (i, token) in window.iter().enumerate() {
        if i > 0 {
            hasher.write(SHINGLE_SEP);
        }
        hasher.write(token.as_bytes());
    }
    hasher.finish()
}

/// The signature: `num_perm` min-hashes of the document's
/// `shingle_size`-token word shingles. Every element starts at the u64
/// MAX sentinel, which is simultaneously the min-identity (so the sweep
/// overwrites it on the first shingle) and the empty-set convention's
/// answer (no shingles means it survives untouched). Tokens ride the
/// crate's one tokenizer (`normalized_word_tokens` with every knob off,
/// the `tf_idf`/`bm25_rank` stream).
pub fn signature(text: &str, num_perm: usize, shingle_size: usize, seed: u64) -> Vec<u64> {
    let mut sig = vec![u64::MAX; num_perm];
    // shingle_size == 0 is unreachable from the pyo3 surface (the binding
    // validates >= 1) and nonsensical as a window width (std's windows(0)
    // panics); the sentinel answer keeps the core total for direct Rust
    // callers rather than panicking on a malformed width.
    if num_perm == 0 || shingle_size == 0 {
        return sig;
    }
    let tokens = normalized_word_tokens(text, false, None, None);
    if tokens.len() < shingle_size {
        return sig;
    }
    let coefficients = coefficients(num_perm, seed);
    for window in tokens.windows(shingle_size) {
        let x = hash_window(window);
        for (element, &(a, b)) in sig.iter_mut().zip(&coefficients) {
            let h = mersenne_mod(u128::from(a) * u128::from(x) + u128::from(b));
            if h < *element {
                *element = h;
            }
        }
    }
    sig
}

/// The number of distinct shingle hashes in `text` at `shingle_size`: the
/// document's shingle-set cardinality, the quantity the Jaccard estimator
/// is actually about (the min-sweep visits every shingle occurrence, but
/// duplicates cannot change a minimum, so the signature depends on this
/// set, not the occurrence count). Exposed for the Rust surface because
/// it is the natural diversity check an LSH-table builder runs before
/// trusting a document's signature (a repeated-token document has
/// thousands of tokens but one distinct shingle, and its estimates are
/// correspondingly coarse), and because `fuzz_targets/minhash.rs`'s
/// semantic-invariant floor is exactly this count.
pub fn distinct_shingle_count(text: &str, shingle_size: usize) -> usize {
    if shingle_size == 0 {
        return 0;
    }
    let tokens = normalized_word_tokens(text, false, None, None);
    if tokens.len() < shingle_size {
        return 0;
    }
    let mut seen = std::collections::HashSet::with_capacity(tokens.len());
    for window in tokens.windows(shingle_size) {
        seen.insert(hash_window(window));
    }
    seen.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The naive spec spelling the streaming core is differentially pinned
    /// against: materialize the joined shingle `String`s, hash each with
    /// the one-shot XXH64, then take each permutation's minimum from the
    /// collected hash list (permutation-major, opposite loop nesting).
    /// Same spec, opposite structure; agreement is evidence about the
    /// contract, not a shared bug.
    fn naive_signature(text: &str, num_perm: usize, shingle_size: usize, seed: u64) -> Vec<u64> {
        let tokens = normalized_word_tokens(text, false, None, None);
        if tokens.len() < shingle_size {
            return vec![u64::MAX; num_perm];
        }
        let shingles: Vec<String> = (0..=tokens.len() - shingle_size)
            .map(|i| tokens[i..i + shingle_size].join("\u{1f}"))
            .collect();
        let hashes: Vec<u64> = shingles
            .iter()
            .map(|s| XxHash64::oneshot(0, s.as_bytes()))
            .collect();
        coefficients(num_perm, seed)
            .into_iter()
            .map(|(a, b)| {
                hashes
                    .iter()
                    .map(|&x| mersenne_mod(u128::from(a) * u128::from(x) + u128::from(b)))
                    .min()
                    .unwrap_or(u64::MAX)
            })
            .collect()
    }

    #[test]
    fn reduce_matches_direct_modulo_on_the_edges_and_a_battery() {
        // The direct `%` oracle (the exact Python-side semantics the
        // differential oracle applies), over the edges of every bound the
        // reduction's derivation names plus a deterministic value battery.
        let mut edges = vec![
            0u128,
            1,
            u128::from(MERSENNE_P) - 1,
            u128::from(MERSENNE_P), // the naive fold loop's fixed point
            u128::from(MERSENNE_P) + 1,
            2 * u128::from(MERSENNE_P),
            u128::from(u64::MAX),
            (1u128 << 61) * 17 + 12345, // a >2^61 mid-range value
            // the exact maximum the sweep can produce: (p-1)*(u64::MAX) + (p-1)
            u128::from(MERSENNE_P - 1) * u128::from(u64::MAX) + u128::from(MERSENNE_P - 1),
        ];
        let mut state = 0x1234_5678_9abc_def0u64;
        for _ in 0..10_000 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let hi = u128::from(state);
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let lo = u128::from(state);
            edges.push(hi << 64 | lo); // full-range u128 battery values
        }
        // Full u128 range is wider than the sweep's domain; keep the
        // battery honest by also checking every value is < 2^126 where the
        // closed-round derivation's bounds hold... the reduce is in fact
        // exact for all u128 (the derivation only ever folds downward), so
        // assert against % unconditionally.
        for x in edges {
            assert_eq!(
                u128::from(mersenne_mod(x)),
                x % u128::from(MERSENNE_P),
                "reduce disagreement at x={x}"
            );
        }
    }

    #[test]
    fn xxh64_reference_vectors() {
        // The published XXH64 vectors (seed 0), pinning the twox-hash
        // integration against the frozen spec itself: the empty-input
        // digest 0xEF46DB3751D8E999 and the single-'a' digest, the same
        // two values the Python-side xxhash package produced when the
        // oracle was derived (tests/reference.py).
        assert_eq!(XxHash64::oneshot(0, b""), 0xEF46_DB37_51D8_E999);
        assert_eq!(XxHash64::oneshot(0, b"a"), 0xD24E_C4F1_A98C_6E5B);
        // The streaming spelling the core uses must agree with oneshot on
        // a multi-part feed: tokens + separators == the joined string.
        let joined = "the\u{1f}quick\u{1f}brown";
        let mut hasher = XxHash64::default();
        hasher.write(b"the");
        hasher.write(SHINGLE_SEP);
        hasher.write(b"quick");
        hasher.write(SHINGLE_SEP);
        hasher.write(b"brown");
        assert_eq!(hasher.finish(), XxHash64::oneshot(0, joined.as_bytes()));
    }

    #[test]
    fn coefficient_derivation_goldens() {
        // The pinned derivation, the same literals the Python oracle pins
        // (tests/reference.py's derivation was transcribed from this
        // arithmetic; these rows are the cross-language lockstep pin).
        let c0 = coefficients(2, 0);
        assert_eq!(c0[0], (153307352162749886, 1042757494553273847));
        assert_eq!(c0[1], (487617019471545680, 1768710312284684787));
        let c42 = coefficients(1, 42);
        assert_eq!(c42[0], (2150242486686805664, 643983082913198340));
        // Seed reduction: -1 as u64 == 2^64 - 1 gives the same stream.
        assert_eq!(coefficients(4, u64::MAX), coefficients(4, (-1i64) as u64));
    }

    #[test]
    fn coefficients_are_in_range_with_nonzero_multipliers() {
        for &seed in &[0u64, 1, 42, u64::MAX] {
            for &(a, b) in &coefficients(1024, seed) {
                assert!((1..MERSENNE_P).contains(&a), "a out of [1, p-1]: {a}");
                assert!(b < MERSENNE_P, "b out of [0, p-1): {b}");
            }
        }
    }

    #[test]
    fn num_perm_prefix_property() {
        // The SplitMix64 stream draws in permutation order, so k=8's
        // signature is exactly k=128's first 8 elements (the structural
        // consequence the Python battery pins as an equality too).
        let full = signature("the quick brown fox jumps over the lazy dog", 128, 3, 0);
        for k in [1usize, 8, 64, 127] {
            assert_eq!(
                signature("the quick brown fox jumps over the lazy dog", k, 3, 0),
                full[..k]
            );
        }
    }

    #[test]
    fn empty_and_short_inputs_pin_the_sentinel_convention() {
        let sentinel = vec![u64::MAX; 128];
        for text in ["", "   ", "\t\n \r\n\u{a0}", "one two"] {
            assert_eq!(signature(text, 128, 3, 0), sentinel, "{text:?}");
        }
        // Seed-invariant: no shingles, no coefficients applied.
        assert_eq!(signature("", 128, 3, 12345), sentinel);
        // Exactly shingle_size tokens is one shingle, not the empty set.
        let one = signature("one two three", 128, 3, 0);
        assert!(one.iter().all(|&v| v != u64::MAX));
        // num_perm 0 or shingle_size 0: the total-behavior guards (the
        // pyo3 surface rejects both before the core ever sees them).
        assert!(signature("one two three", 0, 3, 0).is_empty());
        assert_eq!(signature("one two three", 4, 0, 0), vec![u64::MAX; 4]);
    }

    #[test]
    fn signature_values_live_in_the_affine_range() {
        let sig = signature("the quick brown fox jumps over the lazy dog", 128, 3, 0);
        assert!(sig.iter().all(|&v| v < MERSENNE_P), "value >= 2^61-1");
    }

    #[test]
    fn case_and_whitespace_respellings_signature_identically() {
        // Downstream of the tokenizer, case and whitespace shape are
        // invisible: same token stream, same shingles, same signature.
        let base = signature("Hello, WORLD! one two three", 64, 3, 0);
        assert_eq!(base, signature("hello, world! one\ttwo\nthree", 64, 3, 0));
    }

    #[test]
    fn determinism_across_calls_and_objects() {
        let text = "the quarterly oil sample interval for field outages";
        let a = signature(text, 128, 3, 0);
        let fresh = text.to_owned();
        assert_eq!(a, signature(&fresh, 128, 3, 0));
        assert_eq!(a, signature(text, 128, 3, 0));
        // Different seeds draw different permutations (deterministic row).
        assert_ne!(a, signature(text, 128, 3, 1));
    }

    #[test]
    fn agrees_with_the_naive_reference_over_a_battery() {
        // Every string over {a, b, space} up to length 5 (363 strings; the
        // space exercises the whitespace skip and WSegSpace joining at
        // every position, and the length-4/5 rows cross the 3-token
        // shingle boundary in both directions), plus the tricky non-ASCII
        // rows, at several parameter shapes.
        let mut battery: Vec<String> = Vec::new();
        let alphabet = ["a", "b", " "];
        let mut frontier: Vec<String> = vec![String::new()];
        for _ in 0..5 {
            let mut next = Vec::new();
            for prefix in &frontier {
                for c in alphabet {
                    let mut s = prefix.clone();
                    s.push_str(c);
                    next.push(s.clone());
                    battery.push(s);
                }
            }
            frontier = next;
        }
        battery.extend([
            "a\r\nb".to_string(),
            "\u{1f469}\u{200d}\u{1f52c} \u{30c6}\u{30b9}\u{30c8}".to_string(),
            "\u{1f1fa}\u{1f1f8}\u{1f1fa}".to_string(),
            "\u{1100}\u{1161}\u{11a8} \u{e0}\u{30d}".to_string(),
            "caf\u{e9} soci\u{e9}t\u{e9} na\u{ef}ve \u{6771}\u{4eac}\u{306f}\u{65e5}\u{672c}"
                .to_string(),
            "a\u{1f}b c".to_string(), // the separator itself as a token
            "Hello, world! One. Two.".to_string(),
            "   \t  ".to_string(),
        ]);
        for text in &battery {
            for (num_perm, shingle_size, seed) in [
                (8usize, 3usize, 0u64),
                (4, 1, 42),
                (16, 4, u64::MAX),
                (8, 2, 7),
            ] {
                assert_eq!(
                    signature(text, num_perm, shingle_size, seed),
                    naive_signature(text, num_perm, shingle_size, seed),
                    "core/naive disagreement for {text:?} at k={num_perm} s={shingle_size} seed={seed}"
                );
            }
        }
    }

    /// The three fixture pairs the Jaccard property pins on the Python
    /// side (tests/test_minhash.py), held here as the crate-side anchors
    /// (the simhash precedent: the impl module's tests hold the measured
    /// anchors, the Python battery mirrors them): near-identical
    /// (agreement 82 of 128, exact J 0.6129), moderately similar (30 of
    /// 128, exact J 0.2000), vocabulary-disjoint (0 of 128, exact J 0).
    #[test]
    fn distinct_shingle_count_pins_the_set_cardinality() {
        // Empty/short/whitespace: the empty set.
        assert_eq!(distinct_shingle_count("", 3), 0);
        assert_eq!(distinct_shingle_count("   ", 3), 0);
        assert_eq!(distinct_shingle_count("one two", 3), 0);
        // n tokens at k: n - k + 1 windows, all distinct here.
        assert_eq!(distinct_shingle_count("one two three", 3), 1);
        assert_eq!(distinct_shingle_count("a b c d e", 3), 3);
        // The repeated-token pathology the fuzz floor exists to exclude:
        // hundreds of token occurrences, ONE distinct shingle (duplicates
        // cannot change a minimum, so the signature -- and any estimate
        // built on it -- rides a single-element set).
        assert_eq!(distinct_shingle_count(&"ab ".repeat(200), 3), 1);
        // Occurrence count vs set: a 3-token-period text has exactly the 3
        // distinct 3-grams of its period, however many times it repeats.
        assert_eq!(distinct_shingle_count(&"ab cd ef ".repeat(50), 3), 3);
    }

    #[test]
    fn fixture_pair_agreement_anchors() {
        let base_sentence = "The quarterly oil sample interval for field outages was adjusted \
                             after the bushing torque specifications changed. Maintenance \
                             windows now close within fourteen days. ";
        let review_sentence = "Review panels approved the revised schedule and the field team \
                               confirmed the plan. ";
        let near_a = base_sentence.repeat(3);
        let near_b = near_a
            .replace("quarterly", "monthly")
            .replace("bushing", "insulator");
        let moderate_a = format!("{base_sentence}{review_sentence}");
        let moderate_b = format!(
            "{review_sentence}Spare parts arrived on site before the storm and the crew \
             replaced the seal. Documentation updates followed the same revision cycle. "
        );
        let disjoint_a = "the quick brown fox jumps over the lazy dog ".repeat(6);
        let disjoint_b =
            "compiler backends schedule instructions over directed acyclic graphs ".repeat(6);
        let agreement = |a: &str, b: &str| {
            let sig_a = signature(a, 128, 3, 0);
            let sig_b = signature(b, 128, 3, 0);
            sig_a.iter().zip(&sig_b).filter(|(x, y)| x == y).count()
        };
        assert_eq!(agreement(&near_a, &near_b), 82);
        assert_eq!(agreement(&moderate_a, &moderate_b), 30);
        assert_eq!(agreement(&disjoint_a, &disjoint_b), 0);
    }
}
