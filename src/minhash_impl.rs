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
//! 2-wise-independent family approximating the min-wise independence the
//! exact statement needs, with residual bias negligible beside the
//! `O(1/sqrt(k))` estimator noise (the fixture pairs pinned in
//! `tests/test_minhash.py` sit inside the `k = 128` 2-sigma band).
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
//! A shingle is `shingle_size` consecutive tokens hashed under an
//! injective length-prefixed framing: the window's token count as one
//! little-endian u64, then per token its UTF-8 byte length as one
//! little-endian u64 followed by the bytes themselves, the whole frame fed
//! to XXH64 (seed 0). The framing is injective by construction
//! (length-prefix codes are uniquely decodable), so a window's XXH64 is a
//! function of the window alone with no separator-injectivity premise at
//! all — deliberately, because "no UAX #29 token can contain U+001F" is
//! FALSE as stated: U+001F is not whitespace, so the segmenter keeps it,
//! and UAX #29 WB4 (ignore Extend/Format/ZWJ) then glues a following
//! combining mark, ZWJ, or SOFT HYPHEN onto it (`tors.word_bounds("a\x1fb")`
//! is the three tokens `["a", "\x1f", "b"]`, but
//! `tors.word_bounds("\x1f\u{301}")` is the single token `["\x1f\u{301}"]`).
//! The old `token + U+001F + token` join therefore rested on the narrower
//! (and segmentation-table-sensitive) claim that U+001F never mixes into a
//! longer segment; the framing rests on nothing the segmenter can take
//! away (the WB4 attach is pinned over a `\x1f`-adjacent mark corpus and
//! the framing's injectivity over a separator-bearing window domain in
//! `tests/test_minhash.py`).
//!
//! # The pinned arithmetic (determinism contract)
//!
//! Every element is fixed, documented arithmetic, platform-independent
//! (integer ops only, no floats, no per-process state), so the same text
//! at the same parameters produces the identical signature across
//! processes, machines, and platforms within one tors version — the
//! same stability requirement `simhash_impl` states for its FNV-1a (a
//! fingerprint that changes between runs breaks every cross-run dedupe
//! built on it). The boundary that claim stops at: the signature is a
//! function of the crate's UAX #29 segmentation tables
//! (`unicode-segmentation`, pinned in `Cargo.lock`) as well as of the
//! frozen arithmetic, so a tors release bumping those tables can change
//! signatures — re-fingerprinting every affected document; a caller
//! persisting signatures or LSH tables across tors versions must
//! re-baseline on upgrade (within a version the arithmetic and XXH64
//! are frozen, nothing varying by process, machine, or platform):
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
//!   conditional subtract, exact for every u128 input — the `v == p` fixed
//!   point of the naive fold loop is why the reduction is spelled in closed
//!   rounds — pinned against a direct `%` oracle over the full u128 range
//!   in the tests);
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

use std::collections::{HashSet, VecDeque};
use std::hash::Hasher as _;

use twox_hash::XxHash64;

#[cfg(test)]
use crate::tokenize_impl::normalized_word_tokens;
use crate::tokenize_impl::normalized_word_tokens_stream;

/// The Mersenne prime the affine permutations live over: p = 2^61 - 1.
const MERSENNE_P: u64 = (1 << 61) - 1;

/// `x mod (2^61 - 1)` for every u128 `x`, the shift-add Mersenne reduction:
/// three fold rounds `(x & p) + (x >> 61)` (each folds the top bits into
/// the bottom 61 without changing the residue mod `2^61 - 1`, since
/// `2^61 ≡ 1`), then one conditional subtract. The closed-round spelling
/// exists because the naive `while x >= p { x = (x & p) + (x >> 61) }`
/// loop never terminates on `x == p` (a fixed point: `p & p + p >> 61` is
/// `p` again) — see `reduce_matches_direct_modulo` below for the edge
/// battery, including that exact value.
fn mersenne_mod(x: u128) -> u64 {
    let m: u128 = u128::from(MERSENNE_P);
    // Full-u128 bound: x < 2^128 -> r1 < 2^67 + 2^61 -> r2 < 2^61 + 2^7
    // (r1 >> 61 <= 65) -> r3 <= 2^61 = m + 1 (r2 >> 61 <= 1), so the one
    // conditional subtract always lands below m. The sweep itself never
    // exceeds (p-1) * u64::MAX + (p-1) < 2^125; the reduction is exact
    // above that anyway, and the test battery pins the whole u128 range.
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

/// The XXH64 (seed 0, spelled explicitly: `Default` would also be seed 0
/// today, but the seed is load-bearing contract, not a default worth
/// inheriting silently) of one shingle window under the injective
/// length-prefixed framing: LE64(window length), then per token
/// LE64(byte length) + the UTF-8 bytes. Length-prefix codes are uniquely
/// decodable, so distinct windows frame distinctly however many U+001F
/// bytes the tokens themselves carry (the module doc's WB4 note) — and
/// the digest is fed streaming, with no per-shingle allocation. The one
/// spelling both the signature sweep and the distinct-shingle counter
/// ride, so the shingle-hash contract cannot drift between them.
fn hash_tokens<'a>(count: u64, tokens: impl Iterator<Item = &'a str>) -> u64 {
    let mut hasher = XxHash64::with_seed(0);
    hasher.write(&count.to_le_bytes());
    for token in tokens {
        hasher.write(
            &u64::try_from(token.len())
                .expect("token longer than u64::MAX bytes")
                .to_le_bytes(),
        );
        hasher.write(token.as_bytes());
    }
    hasher.finish()
}

/// [`hash_tokens`] over a materialized window: the spelling the tests pin
/// the framing through.
#[cfg(test)]
fn hash_window(window: &[String]) -> u64 {
    hash_tokens(
        u64::try_from(window.len()).expect("window longer than u64::MAX tokens"),
        window.iter().map(String::as_str),
    )
}

/// [`hash_tokens`] over the live streaming window: the spelling the sweep
/// rides, so the window never materializes outside the deque.
fn hash_live_window(window: &VecDeque<String>) -> u64 {
    hash_tokens(
        u64::try_from(window.len()).expect("window longer than u64::MAX tokens"),
        window.iter().map(String::as_str),
    )
}

/// Widths above this take the count-first path: a window wider than 1024
/// tokens is past every documented use (the default 3, the bench's
/// widest timing row at 256), so the short-stream case -- the only case
/// where the deque would grow to the token count instead of the width --
/// is decided by a retention-free token count first (see
/// `token_count_up_to`), and the deque never materializes the stream for
/// an answer that is always the empty-set sentinel.
const WIDE_WINDOW_COUNT_FIRST: usize = 1024;

/// The token count up to `limit`, streamed without retaining anything:
/// the retention-free probe wide windows ride before deciding the stream
/// can ever fill one. Returns `min(tokens, limit)` -- callers comparing
/// against `limit` learn exactly "fewer than `limit`" vs "at least
/// `limit`" with at most `limit` tokens walked.
fn token_count_up_to(text: &str, limit: usize) -> usize {
    let mut n = 0usize;
    for _ in normalized_word_tokens_stream(text) {
        n += 1;
        if n >= limit {
            break;
        }
    }
    n
}

/// Initial distinct-hash capacity guess from the byte length: ~6 bytes per
/// token on prose bounds the window count from above, clamped so tiny
/// inputs do not over-reserve and huge inputs do not pre-grab memory the
/// set may never need (the repeated-token pathology needs exactly one
/// slot). A heuristic only -- the set grows by rehash exactly as before,
/// and the answer is bit-identical either way.
fn distinct_capacity_guess(text: &str) -> usize {
    (text.len() / 6).clamp(64, 8192)
}

/// The signature: `num_perm` min-hashes of the document's
/// `shingle_size`-token word shingles. Every element starts at the u64
/// MAX sentinel, which is simultaneously the min-identity (so the sweep
/// overwrites it on the first distinct shingle) and the empty-set
/// convention's answer (no shingles means it survives untouched). Tokens
/// ride the crate's one tokenizer (`normalized_word_tokens_stream`, the
/// `tf_idf`/`bm25_rank` stream with every knob off), streamed through a
/// `shingle_size`-deep window: only the live window is ever resident, not
/// the token list. The window holds at most `shingle_size` tokens --
/// resident window memory is `O(min(tokens, shingle_size))` -- except
/// through the wide-window short-circuit: widths above
/// `WIDE_WINDOW_COUNT_FIRST` first count tokens retention-free, and a
/// stream ending short of a full window answers the sentinel with `O(1)`
/// window memory instead of materializing the whole stream in the deque.
///
/// The sweep is dedup-first: each distinct shingle hash updates the minima
/// once, so the pass is `O(tokens × shingle_size)` hashing (every step
/// re-hashes the whole live window under the length-prefixed framing) plus
/// `O(distinct × num_perm)` in the sweep (a repeated-token document rides
/// its one distinct shingle, not its thousands of occurrences). Minima over
/// occurrences equal minima over the distinct set, so the answer is
/// byte-identical to the naive per-occurrence sweep. The distinct set is
/// pre-sized from `distinct_capacity_guess` (a heuristic; growth rehashes
/// as before). Note on the `HashSet`: it rides the std `RandomState`
/// hasher, whose per-process seed would matter if iteration order
/// escaped — it cannot here (only the per-position minima and the set
/// cardinality escape, both order-independent), so the signature stays
/// fixture-grade deterministic; see also `distinct_shingle_count`.
///
/// `num_perm` carries a core-side sanity ceiling (1M elements, 8 MiB) so a
/// direct Rust caller cannot spell `vec![u64::MAX; usize::MAX]` past the
/// allocator: the pyo3 binding caps at 1024 long before this, and the fuzz
/// target ranges `num_perm` over 0..=1024 to keep the `num_perm == 0`
/// empty-signature guard exercised.
pub fn signature(text: &str, num_perm: usize, shingle_size: usize, seed: u64) -> Vec<u64> {
    assert!(
        num_perm <= 1 << 20,
        "num_perm {num_perm} exceeds the core sanity ceiling (2^20; the binding caps at 1024)"
    );
    let mut sig = vec![u64::MAX; num_perm];
    // shingle_size == 0 is unreachable from the pyo3 surface (the binding
    // validates >= 1) and nonsensical as a window width; the sentinel
    // answer keeps the core total for direct Rust callers rather than
    // panicking on a malformed width.
    if num_perm == 0 || shingle_size == 0 {
        return sig;
    }
    // Wide-window short-circuit: fewer tokens than the width means the
    // empty set, decided here by a retention-free count so the deque below
    // never grows to the token count for an answer fixed in advance. When
    // the count reaches the width the stream CAN fill a window and the
    // sweep runs as usual.
    if shingle_size > WIDE_WINDOW_COUNT_FIRST
        && token_count_up_to(text, shingle_size) < shingle_size
    {
        return sig;
    }
    let coefficients = coefficients(num_perm, seed);
    let mut window: VecDeque<String> = VecDeque::new();
    let mut distinct: HashSet<u64> = HashSet::with_capacity(distinct_capacity_guess(text));
    for token in normalized_word_tokens_stream(text) {
        window.push_back(token);
        if window.len() > shingle_size {
            window.pop_front();
        }
        if window.len() == shingle_size {
            distinct.insert(hash_live_window(&window));
        }
    }
    // No full window ever formed: fewer tokens than shingle_size, the
    // empty-shingle-set convention (the sentinel survives untouched).
    for x in &distinct {
        for (element, &(a, b)) in sig.iter_mut().zip(&coefficients) {
            let h = mersenne_mod(u128::from(a) * u128::from(*x) + u128::from(b));
            if h < *element {
                *element = h;
            }
        }
    }
    sig
}

/// The number of distinct shingle hashes in `text` at `shingle_size`: the
/// document's shingle-set cardinality, the quantity the Jaccard estimator
/// is actually about (duplicates cannot change a minimum, so the signature
/// depends on this set, not the occurrence count — the sweep rides the
/// same set). Streamed like the signature: only the live window plus the
/// hash set is resident, never the token list. Exposed for the Rust surface
/// because it is the natural diversity check an LSH-table builder runs
/// before trusting a document's signature (a repeated-token document has
/// thousands of tokens but one distinct shingle, and its estimates are
/// correspondingly coarse), and because `fuzz_targets/minhash.rs`'s
/// semantic-invariant floor is exactly this count. The `HashSet` rides
/// `RandomState` like the signature's; only the cardinality escapes, so
/// the count is deterministic across processes.
pub fn distinct_shingle_count(text: &str, shingle_size: usize) -> usize {
    if shingle_size == 0 {
        return 0;
    }
    // The same wide-window short-circuit as `signature`: a stream ending
    // short of a full wide window is the empty set, decided retention-free.
    if shingle_size > WIDE_WINDOW_COUNT_FIRST
        && token_count_up_to(text, shingle_size) < shingle_size
    {
        return 0;
    }
    let mut window: VecDeque<String> = VecDeque::new();
    let mut seen: HashSet<u64> = HashSet::with_capacity(distinct_capacity_guess(text));
    for token in normalized_word_tokens_stream(text) {
        window.push_back(token);
        if window.len() > shingle_size {
            window.pop_front();
        }
        if window.len() == shingle_size {
            seen.insert(hash_live_window(&window));
        }
    }
    seen.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The naive spec spelling the streaming core is differentially pinned
    /// against: materialize the length-prefixed shingle frames, hash each
    /// with the one-shot XXH64, then take each permutation's minimum from
    /// the collected hash list (permutation-major, opposite loop nesting;
    /// dedup-free, unlike the core's distinct-set sweep — minima agree
    /// either way). Same spec, opposite structure; agreement is evidence
    /// about the contract, not a shared bug.
    fn naive_signature(text: &str, num_perm: usize, shingle_size: usize, seed: u64) -> Vec<u64> {
        let tokens = normalized_word_tokens(text, false, None, None);
        if tokens.len() < shingle_size {
            return vec![u64::MAX; num_perm];
        }
        let hashes: Vec<u64> = (0..=tokens.len() - shingle_size)
            .map(|i| {
                let window = &tokens[i..i + shingle_size];
                let mut frame = Vec::new();
                frame.extend_from_slice(
                    &u64::try_from(window.len())
                        .expect("window longer than u64::MAX tokens")
                        .to_le_bytes(),
                );
                for token in window {
                    frame.extend_from_slice(
                        &u64::try_from(token.len())
                            .expect("token longer than u64::MAX bytes")
                            .to_le_bytes(),
                    );
                    frame.extend_from_slice(token.as_bytes());
                }
                XxHash64::oneshot(0, &frame)
            })
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
        // a multi-part feed: the length-prefixed frame written part by
        // part == the same frame hashed whole.
        let mut frame = Vec::new();
        frame.extend_from_slice(&3u64.to_le_bytes());
        for token in ["the", "quick", "brown"] {
            frame.extend_from_slice(&(token.len() as u64).to_le_bytes());
            frame.extend_from_slice(token.as_bytes());
        }
        let mut hasher = XxHash64::with_seed(0);
        hasher.write(&3u64.to_le_bytes());
        for token in ["the", "quick", "brown"] {
            hasher.write(&(token.len() as u64).to_le_bytes());
            hasher.write(token.as_bytes());
        }
        assert_eq!(hasher.finish(), XxHash64::oneshot(0, &frame));
    }

    #[test]
    fn shingle_framing_is_length_prefixed() {
        // UAX #29 WB4 attaches Extend/Format/ZWJ to U+001F, so tokens like
        // "\x1f\u{301}" exist and the old token+U+001F+token join is no
        // injective basis. The frame is LE64(window_len) followed by
        // LE64(byte_len)+bytes per token (explicit little-endian: the
        // determinism contract is cross-platform), hashed with XXH64 seed 0.
        let window = vec!["a".to_string(), "\x1f\u{301}".to_string()];
        let mut frame = Vec::new();
        frame.extend_from_slice(&(window.len() as u64).to_le_bytes());
        for token in &window {
            frame.extend_from_slice(&(token.len() as u64).to_le_bytes());
            frame.extend_from_slice(token.as_bytes());
        }
        assert_eq!(hash_window(&window), XxHash64::oneshot(0, &frame));
        // Window boundaries the framing must disambiguate: a one-token
        // window holding the separator-bearing token is a different frame
        // from the two-token window spelling the same bytes apart, and
        // ["ab", "c"] frames differently from ["a", "bc"] (the lengths
        // travel with the bytes, so no boundary can hide).
        assert_ne!(
            hash_window(&["\x1f\u{301}".to_string()]),
            hash_window(&["\x1f".to_string(), "\u{301}".to_string()])
        );
        assert_ne!(
            hash_window(&["ab".to_string(), "c".to_string()]),
            hash_window(&["a".to_string(), "bc".to_string()])
        );
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
        // A shingle wider than the token stream is the empty set, however
        // huge the width: the sentinel with no window blowup (the window
        // deque is grown, never pre-reserved, so this reserves nothing).
        assert_eq!(
            signature("one two three", 4, usize::MAX, 0),
            vec![u64::MAX; 4]
        );
        assert_eq!(distinct_shingle_count("one two three", usize::MAX), 0);
    }

    #[test]
    fn dedup_sweep_agrees_with_naive_on_repeated_tokens() {
        // The dedup-first sweep's load-bearing equality where the
        // occurrence and distinct sets differ most: "ab ".repeat(200) is
        // hundreds of occurrences of ONE distinct shingle, the period-3
        // text hundreds of occurrences of three. Minima over occurrences
        // equal minima over the distinct set, so the core must match the
        // dedup-free naive spelling exactly here, not just on short
        // strings.
        for text in [("ab ".repeat(200)), ("ab cd ef ".repeat(50))] {
            for (num_perm, shingle_size, seed) in
                [(128usize, 3usize, 0u64), (8, 3, 42), (16, 2, u64::MAX)]
            {
                assert_eq!(
                    signature(&text, num_perm, shingle_size, seed),
                    naive_signature(&text, num_perm, shingle_size, seed),
                    "dedup/naive disagreement on repeated tokens at k={num_perm} s={shingle_size}"
                );
            }
        }
    }

    #[test]
    fn wide_window_over_large_text_answers_sentinel() {
        // The count-first path's contract at scale: ~1 MB of prose at a
        // wider-than-stream width is the empty set, decided without ever
        // growing the deque to the token count.
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(24_000);
        assert_eq!(signature(&text, 4, usize::MAX, 0), vec![u64::MAX; 4]);
        assert_eq!(distinct_shingle_count(&text, usize::MAX), 0);
        // The counter itself: 10 word tokens per sentence ("the quick
        // brown fox jumps over the lazy dog" plus its period), all walked
        // retention-free when the limit exceeds the stream.
        assert_eq!(token_count_up_to(&text, usize::MAX), 240_000);
        assert_eq!(token_count_up_to(&text, 10), 10);
    }

    #[test]
    #[should_panic(expected = "core sanity ceiling")]
    fn num_perm_past_the_sanity_ceiling_panics() {
        // A direct Rust caller spelling vec![MAX; usize::MAX] must get a
        // named panic, not an allocator abort: the ceiling sits far above
        // the binding's 1024 cap.
        let _ = signature("one two three", (1 << 20) + 1, 3, 0);
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
    /// (agreement 74 of 128, exact J 0.6129), moderately similar (22 of
    /// 128, exact J 0.2000), vocabulary-disjoint (0 of 128, exact J 0).
    /// All three estimates sit inside the k=128 2-sigma band (0.088).
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
        assert_eq!(agreement(&near_a, &near_b), 74);
        assert_eq!(agreement(&moderate_a, &moderate_b), 22);
        assert_eq!(agreement(&disjoint_a, &disjoint_b), 0);
    }
}
