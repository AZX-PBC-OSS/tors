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
//! scope cut) — and the banding pass that turns signatures into
//! candidate pairs is its own stateless module (`lsh_impl`: one call,
//! one pass, no persistent table), never a hidden handle inside this
//! core.
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
//! the `diff_opcodes` list-marshalling class at a far smaller count).
//! The sweep-budget gate also runs under the GIL before the detach, and
//! is itself bounded: its retention-free count walk stops at
//! ⌊budget/shingle_size⌋ + shingle_size tokens (see `sweep_past_budget`),
//! so even a G many-token stream is rejected without a GIL-held walk of
//! the whole input. The `aio` twin (`tors.aio.minhash_signature`) is the
//! same call dispatched through `asyncio.to_thread` — mechanically the
//! hop the GIL-heartbeat cell pins — for callers that want the wait off
//! the event loop entirely.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hasher as _;

use twox_hash::XxHash64;

use crate::segmentation_impl::real_word_segments;
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
/// ride, so the shingle-hash contract cannot drift between them. Shared
/// with `near_dup_impl`'s shingle-set functions, so the crate has exactly
/// one shingle hashing contract (the near-dup comparison layer's set
/// operations ride the same framing, not a second, weaker one).
pub(crate) fn hash_tokens<'a>(count: u64, tokens: impl Iterator<Item = &'a str>) -> u64 {
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

/// One token's identity hash: the shingle_size-1 shingle hash (the
/// single-token window under the same length-prefixed framing), the
/// weighted engine's token key. Shared `pub(crate)` with the pyo3
/// binding's explicit-weight mapping, whose keys are hashed outside the
/// detached pass but through this same contract.
pub(crate) fn token_hash(token: &str) -> u64 {
    hash_tokens(1, std::iter::once(token))
}

/// The crate's one hashing contract applied to raw u64 values: XXH64
/// (seed 0) over the length-prefixed little-endian frame -- LE64 of the
/// value count, then LE64 of each value -- the same framing discipline
/// [`hash_tokens`] spells for token windows, over rows instead of
/// strings. Shared `pub(crate)` with `lsh_impl`, whose band keys are
/// exactly this frame over one band's `r` signature rows, so the crate
/// keeps exactly one XXH64 contract (one seed, one framing, whatever the
/// unit) and the band keys cannot drift from the shingle hashes' frozen
/// arithmetic.
pub(crate) fn hash_u64_frame(rows: &[u64]) -> u64 {
    let mut hasher = XxHash64::with_seed(0);
    hasher.write(
        &u64::try_from(rows.len())
            .expect("row frame longer than u64::MAX values")
            .to_le_bytes(),
    );
    for row in rows {
        hasher.write(&row.to_le_bytes());
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

/// The token-hash budget one wide fillable-window sweep may spend: the
/// ceiling the pyo3 binding turns into a `ValueError` and the core asserts
/// for direct Rust callers. Every step of the sweep re-hashes the whole
/// live window (`hash_live_window`), so a stream that fills a window past
/// `WIDE_WINDOW_COUNT_FIRST` costs `(tokens - shingle_size + 1) ×
/// shingle_size` token-hashes -- unbounded in exactly the middle range the
/// sentinel short-circuit above cannot decide (the stream CAN fill the
/// window). 2^26 token-hashes measures ~0.4 s at the repro's ~5.8 ns per
/// framed window hash (20k tokens x 10^4 wide -> 583 ms is 10^8), the
/// point where one call stops being the fast one-shot the GIL model
/// promises, orders of magnitude past every documented width, so nothing
/// a caller should be doing is refused. Widths at or below
/// `WIDE_WINDOW_COUNT_FIRST` bound the per-token cost by the width
/// itself and ride the documented caller-size lever instead: no count
/// walk is spent on the default path.
pub(crate) const SHINGLE_SWEEP_BUDGET: u128 = 1 << 26;

/// The token count up to `limit`, streamed without retaining anything:
/// the retention-free probe wide windows ride before deciding the stream
/// can ever fill one. Returns `min(tokens, limit)` -- callers comparing
/// against `limit` learn exactly "fewer than `limit`" vs "at least
/// `limit`" with at most `limit` tokens walked.
///
/// Allocation-free by construction: counts `real_word_segments` directly,
/// never the lowercased `String` stream (the previous spelling ran ~3M
/// transient `String` alloc/free cycles on the gate's 13.5MB probe -- O(1)
/// live, but churn this spelling simply does not pay). The counts are
/// identical because `str::to_lowercase` maps every non-empty segment to a
/// non-empty token (char-wise lowercasing never deletes: each input char
/// yields >= 1 output char), so the stream's `!term.is_empty()` filter
/// never fires on the no-knob path and no token is ever dropped or added.
/// This spelling walks `&str` slices with zero allocation, so the
/// huge-window short-circuit holds only the interpreter + input string
/// resident. The CI #74 memory saga ended with the finding that the
/// ~570-615MB readings once attributed to a glibc RSS peak were a
/// workload-invariant `resource.getrusage` fiction on the ubuntu runners
/// (true peaks ~15-38MB by `/proc` VmHWM; no retaining allocation exists)
/// -- the gate in `tests/test_minhash.py` now reads the kernel high-water
/// mark, and this zero-alloc spelling is what it pins.
fn token_count_up_to(text: &str, limit: usize) -> usize {
    if limit == 0 {
        return 0;
    }
    let mut n = 0usize;
    for _ in real_word_segments(text) {
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

/// The sweep-budget probe both validation layers ride: `Some(work)` -- a
/// token-hash count the pass would spend, proven past
/// [`SHINGLE_SWEEP_BUDGET`] -- when the shape sweeps over budget, `None`
/// when the call may proceed. The None cases: any width at or below
/// `WIDE_WINDOW_COUNT_FIRST` (the per-token cost is width-bounded there,
/// and no count walk is spent on the default path), and a stream whose
/// sweep cannot exceed the budget. The probe is retention-free and bounded
/// -- O(min(tokens, ⌊budget/shingle_size⌋ + shingle_size)) tokens walked,
/// never the O(tokens × shingle_size) pass it gates, and (the same bug
/// class one level up) never O(tokens) either: the walk stops at the cap,
/// so a huge stream is rejected without walking all of it.
///
/// `work` is the exact spend only when the stream ends at or under the
/// cap (where the over-budget case is impossible, so `None`); past the
/// cap the exact count is never walked and `work` is the MINIMUM any
/// shape that deep would spend -- `(⌊budget/s⌋ + 1) × s` at the cap --
/// which is all the reject decision needs (the pyo3 binding reports it
/// as "at least {work}").
pub(crate) fn sweep_past_budget(text: &str, shingle_size: usize) -> Option<u128> {
    if shingle_size <= WIDE_WINDOW_COUNT_FIRST {
        return None;
    }
    let s = shingle_size as u128;
    // The first token count whose sweep must exceed the budget:
    // `(tokens - s + 1) × s > B` has its first integer solution at
    // `tokens = ⌊B/s⌋ + s` (one token earlier the work is at most
    // `⌊B/s⌋ × s <= B` -- the exactly-at-budget boundary included), so a
    // walk ending short of that cap proves the within-budget case
    // outright, and reaching it proves the reject without walking on.
    // The cap is the whole gate's cost bound: at most ⌊B/1025⌋ + 1025 ≈
    // 66.5k tokens for every admissible width, usize::MAX tokens included.
    let cap = SHINGLE_SWEEP_BUDGET / s + s;
    // cap <= usize::MAX on every target, so the try_from cannot fail: for
    // s > B the sum is s itself; for 1025 <= s <= B it is at most
    // B/1025 + B < 2^27. (A saturated fallback here would silently disarm
    // the gate -- the walk would never reach it -- hence the named panic.)
    let cap = usize::try_from(cap)
        .expect("budget/shingle_size + shingle_size fits usize for every admissible width");
    if token_count_up_to(text, cap) < cap {
        // The stream ended short of the cap: the count is exact and the
        // inequality above puts its sweep at or under the budget -- the
        // within-budget case, the exactly-at-budget boundary included.
        return None;
    }
    // The walk reached the cap: tokens >= cap, so the sweep spends at
    // least `(cap - s + 1) × s` token-hashes, which exceeds the budget
    // by construction. Reported at the cap: the minimum provable work.
    Some((cap as u128 - s + 1) * s)
}

/// The core half of the two-sided sweep-budget validation (the pyo3
/// binding's `ValueError` is the other half): a fillable wide window whose
/// sweep would run past the budget is a caller-shape bug, named in the
/// panic (the num_perm ceiling's spelling) instead of silently grinding
/// through the O(tokens × shingle_size) pass. Unreachable from the pyo3
/// surface, which raises the `ValueError` before the detached pass.
fn assert_sweep_within_budget(text: &str, shingle_size: usize) {
    assert!(
        sweep_past_budget(text, shingle_size).is_none(),
        "shingle_size {shingle_size} over a fillable stream would sweep past the \
         {SHINGLE_SWEEP_BUDGET}-token-hash core sanity budget (the pyo3 binding rejects \
         this shape with a ValueError)"
    );
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
/// as before). Past `WIDE_WINDOW_COUNT_FIRST`, a stream that fills the
/// window is budget-bounded: shapes whose `(tokens - shingle_size + 1) ×
/// shingle_size` token-hash cost exceeds `SHINGLE_SWEEP_BUDGET` are a
/// caller-shape bug -- the pyo3 binding rejects them with a `ValueError`
/// and `assert_sweep_within_budget` names them for direct Rust callers --
/// so the middle range the sentinel short-circuit cannot decide cannot
/// sweep unbounded. Note on the `HashSet`: it rides the std `RandomState`
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
    assert_sweep_within_budget(text, shingle_size);
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
    assert_sweep_within_budget(text, shingle_size);
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

// ---------------------------------------------------------------------------
// The SuperMinHash engine (Ertl, arXiv 1706.05698)
// ---------------------------------------------------------------------------

/// One uniform draw in (0, 1) from a SplitMix64 chain state: the top 53
/// bits of the draw, scaled into the open unit interval with the +0.5
/// centering that keeps both endpoints unreachable (`u > 0` makes the
/// logarithmic draws below finite, `u < 1` makes `1 - u` positive). The
/// same spelling every float draw of the new engines rides, so the
/// conversion is pinned in exactly one place.
fn uniform_open(state: &mut u64) -> f64 {
    let draw = splitmix64(state);
    ((draw >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
}

/// One standard exponential draw from a uniform in (0, 1): the inverse
/// CDF `-ln(1 - u)`, strictly positive and finite because
/// `u` is (the `uniform_open` contract).
fn exponential_open(state: &mut u64) -> f64 {
    -(1.0 - uniform_open(state)).ln()
}

/// The per-element pseudorandom stream both new engines draw from: the
/// crate's one XXH64 contract over the little-endian frame of the given
/// words (`hash_u64_frame`), the digest standing in for the paper's
/// "initialize pseudo-random generator with seed d" -- the element itself
/// IS the seed, exactly as Algorithm 4 of arXiv 1706.05698 states it. The
/// caller advances the returned state through `splitmix64` per draw.
fn element_stream(words: &[u64]) -> u64 {
    hash_u64_frame(words)
}

/// The SuperMinHash signature of `text`'s word shingles: Ertl's
/// SuperMinHash algorithm (arXiv 1706.05698, Algorithm 4) over the same
/// shingle set the classic sweep rides. The signature values are REAL
/// numbers in [0, m) -- `r + j` with `r` uniform in (0, 1) and `j` the
/// permutation position -- so each row is emitted as the f64 bit pattern,
/// the exact encoding that keeps "same winning shingle" an exact row
/// equality (finite f64 bit patterns never collide with the u64 MAX
/// sentinel, the empty-set convention below). Two documents agree at
/// position j exactly when the same shingle achieves the minimum, so the
/// agreement-fraction estimator stays unbiased (the paper's equation 2
/// holds for the `r + pi` values) with the paper's STRICTLY SMALLER
/// variance for small sets (section 2.2, the alpha(m, u) factor -- ~half
/// the classic variance when the union cardinality is under the signature
/// size).
///
/// The algorithm is order-sensitive (the in-place permutation state does
/// not commute across elements the way a min does), so the sweep order is
/// pinned: the distinct shingle hashes ASCENDING -- fully deterministic
/// across processes, machines, and platforms (the classic sweep's
/// order-independence argument does not transfer, and no `RandomState`
/// iteration order ever escapes). Duplicates ride the distinct set exactly
/// as the classic sweep does (the paper's section 2.3 note: repeated
/// insertions change no signature state).
///
/// NOT a prefix property: the permutation structure depends on m, so
/// unlike the classic engine a smaller `num_perm` is NOT a prefix of a
/// larger signature, and signatures from different `num_perm` values (or
/// different methods) are not comparable -- the estimator reads only
/// paired rows from signatures produced at identical parameters.
fn superminhash_rows(shingles: &[u64], num_perm: usize, seed: u64) -> Vec<u64> {
    let m = num_perm;
    // Algorithm 4's state: h the signature values (infinite = unfilled),
    // p the in-place permutation array (lazily initialized per element via
    // q, the paper's -1 spelled as usize::MAX), b the histogram of
    // integral parts (b[m-1] counts values >= m-1, the infinities
    // included), a the maximum nonzero histogram index (the early-exit
    // frontier: updates are impossible past it).
    let mut h = vec![f64::INFINITY; m];
    let mut p = vec![0usize; m];
    let mut q = vec![usize::MAX; m];
    let mut b = vec![0usize; m];
    b[m - 1] = m;
    let mut a = m - 1;
    for (i, &d) in shingles.iter().enumerate() {
        let mut state = element_stream(&[seed, d]);
        let mut j = 0usize;
        while j <= a {
            // The draws per iteration, in pinned order: r first, then the
            // swap target k uniform over {j..m-1} (the paper's "uniform
            // random number from {j, ..., m-1}"; the SplitMix64 modulo
            // reduction carries the same negligible 2^-64-scale bias the
            // classic coefficients' reductions carry).
            let r = uniform_open(&mut state);
            let k = j + (splitmix64(&mut state) % (m - j) as u64) as usize;
            if q[j] != i {
                q[j] = i;
                p[j] = j;
            }
            if q[k] != i {
                q[k] = i;
                p[k] = k;
            }
            p.swap(j, k);
            let slot = p[j];
            let candidate = r + j as f64;
            if candidate < h[slot] {
                // f64::INFINITY saturates the as-cast to usize::MAX, so
                // the min() lands every first update at m-1 (the paper's
                // floor(infinity) case).
                let j_old = (h[slot].floor() as usize).min(m - 1);
                h[slot] = candidate;
                if j < j_old {
                    b[j_old] -= 1;
                    b[j] += 1;
                    while b[a] == 0 {
                        a -= 1;
                    }
                }
            }
            j += 1;
        }
    }
    h.iter()
        .map(|&v| if v.is_finite() { v.to_bits() } else { u64::MAX })
        .collect()
}

/// The SuperMinHash surface twin of [`signature`]: the same shingle
/// collection (the streamed window, the dedup-first distinct set, the
/// same budget gates) with the sweep swapped for Ertl's Algorithm 4. The
/// distinct hashes are sorted ascending before the sweep -- the pinned
/// deterministic order the order-sensitive algorithm needs (see
/// [`superminhash_rows`]).
pub fn superminhash_signature(
    text: &str,
    num_perm: usize,
    shingle_size: usize,
    seed: u64,
) -> Vec<u64> {
    assert!(
        num_perm <= 1 << 20,
        "num_perm {num_perm} exceeds the core sanity ceiling (2^20; the binding caps at 1024)"
    );
    if num_perm == 0 || shingle_size == 0 {
        return vec![u64::MAX; num_perm];
    }
    if shingle_size > WIDE_WINDOW_COUNT_FIRST
        && token_count_up_to(text, shingle_size) < shingle_size
    {
        return vec![u64::MAX; num_perm];
    }
    assert_sweep_within_budget(text, shingle_size);
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
    let mut shingles: Vec<u64> = distinct.into_iter().collect();
    shingles.sort_unstable();
    superminhash_rows(&shingles, num_perm, seed)
}

/// The b-bit Jaccard estimator over two signatures of equal length: the
/// paired-row agreement fraction, masked to the lowest `bits` bits when
/// some. `None` is the classic agreement estimator (unbiased over the
/// affine rows' full range). `Some(bits)` is Li and König's b-bit
/// minwise estimator (Li and König, WWW 2010, "b-Bit Minwise
/// Hashing": the agreement probability of masked rows is
/// `J + (1 - J) * 2^-b`, so the unbiased estimate is the corrected
/// fraction `(p_hat - 2^-b) / (1 - 2^-b)` -- which can land slightly
/// NEGATIVE at true similarity zero (the correction subtracts the chance
/// term from a finite sample); that is the unbiased estimator's honest
/// shape, not a bug, and callers thresholding should compare the raw
/// agreement fraction instead. The variance is at most ~3x the full-row
/// estimator's once `b >= log2(1/J)` (below that the chance-agreement
/// term dominates the information the rows carry).
pub fn jaccard_estimate(a: &[u64], b: &[u64], bits: Option<u32>) -> f64 {
    assert_eq!(a.len(), b.len(), "signature lengths differ");
    assert!(!a.is_empty(), "empty signatures estimate nothing");
    let agree = match bits {
        None => a.iter().zip(b).filter(|(x, y)| x == y).count(),
        Some(bits) => {
            let mask = (1u64 << bits) - 1;
            a.iter()
                .zip(b)
                .filter(|&(&x, &y)| (x & mask) == (y & mask))
                .count()
        }
    };
    let p_hat = agree as f64 / a.len() as f64;
    match bits {
        None => p_hat,
        Some(bits) => {
            let chance = 2f64.powi(-(bits as i32));
            (p_hat - chance) / (1.0 - chance)
        }
    }
}

// ---------------------------------------------------------------------------
// The weighted engine (Consistent Weighted Sampling, Ioffe 2010)
// ---------------------------------------------------------------------------

/// Token occurrence counts by shingle hash: the multiset weights the
/// weighted engine rides, keyed by the crate's one hashing contract
/// applied to each token as its own one-token window (`hash_tokens` with
/// count 1 -- the shingle_size-1 shingle hash, so a token's identity is
/// the same XXH64 value the classic engine's narrowest shingles carry).
/// Counting rides hashes, not strings (resident memory stays O(distinct)
/// u64s like the classic sweep's set); two distinct tokens colliding in
/// XXH64 is the ~n^2/2^65 negligible-by-design channel the LSH band keys
/// already document.
fn token_hash_counts(tokens: impl Iterator<Item = impl AsRef<str>>) -> Vec<(u64, f64)> {
    let mut counts: HashMap<u64, u64> = HashMap::new();
    for token in tokens {
        *counts.entry(token_hash(token.as_ref())).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(h, count)| (h, count as f64))
        .collect()
}

/// The ICWS signature of a weighted token multiset: Ioffe's Improved
/// Consistent Weighted Sampling (ICDM 2010, "Improved Consistent
/// Sampling, Weighted MinHash and L1 Sketching"), the active-index scheme
/// Shrivastava's NeurIPS 2016 paper restates as its Algorithm 1. For each
/// permutation j (0-based) and each token hash h with weight w > 0:
///
/// - the per-(j, h) stream is the crate's one XXH64 contract over the
///   frame `[seed, j, h]` (`hash_u64_frame`), advanced by SplitMix64;
/// - five draws in pinned order: `r = e(u1) + e(u2)` and
///   `c = e(u3) + e(u4)` (each the sum of two standard exponentials, i.e.
///   Gamma(2, 1), the paper's distribution), then `beta = u5` uniform;
///   `e(u) = -ln(1 - u)` is the inverse-CDF exponential over
///   `uniform_open`'s (0, 1) draws;
/// - `t = ln(w) / r + beta` (the active index), `y = exp(r * (t - beta))`,
///   `z = y * exp(r)`, `a = c / z`;
/// - the permutation's sample is the token achieving the MINIMUM `a`
///   (ties, measure-zero in exact arithmetic, break to the LOWEST token
///   hash: the sweep runs over tokens sorted ascending by (hash, weight)
///   and only a strictly smaller `a` replaces the incumbent);
/// - the signature rows are the pair `(hash, t)` of the winner, with `t`
///   emitted as its f64 bit pattern (the injective encoding that keeps
///   "same active index" an exact row equality for every positive weight,
///   fractional ones included; finite f64 never produces the u64 MAX
///   sentinel pattern, which stays the empty-multiset convention).
///
/// The pair-agreement fraction of two such signatures estimates the
/// GENERALIZED (weighted) Jaccard similarity
/// `sum(min(w_a, w_b)) / sum(max(w_a, w_b))` -- Ioffe's consistency
/// theorem, the property the empirical accuracy cell pins against exact
/// count-vector Jaccards. The estimator is consistent, not row-wise
/// independent; the sample per permutation reads every token, so the
/// pass is O(num_perm * distinct_tokens).
pub fn weighted_signature_from_weights(
    mut items: Vec<(u64, f64)>,
    num_perm: usize,
    seed: u64,
) -> Vec<u64> {
    assert!(
        num_perm <= 1 << 20,
        "num_perm {num_perm} exceeds the core sanity ceiling (2^20; the binding caps at 1024)"
    );
    // Weights are validated for the direct Rust caller (the binding
    // rejects negatives, NaN, and infinities as ValueErrors before here):
    // zero weights are the absent-token case (a zero-weight token
    // contributes nothing to the generalized Jaccard), positive weights
    // only past this line, so the logarithms below are finite.
    for (_, w) in &items {
        assert!(
            w.is_finite() && *w >= 0.0,
            "weights must be finite and non-negative, not {w}"
        );
    }
    items.retain(|&(_, w)| w > 0.0);
    if num_perm == 0 || items.is_empty() {
        return vec![u64::MAX; 2 * num_perm];
    }
    // The pinned sweep order: ascending (hash, weight). Deterministic
    // across processes (no RandomState order escapes), and the tie-break
    // the argmin's strict comparison rides.
    items.sort_unstable_by(|x, y| x.0.cmp(&y.0).then(x.1.total_cmp(&y.1)));
    let mut sig = vec![u64::MAX; 2 * num_perm];
    for (j, pair) in sig.chunks_mut(2).enumerate() {
        // The incumbent: Option, not an infinity sentinel -- a candidate
        // whose z underflowed to zero scores +inf (c/z over IEEE float
        // division) and must still win when it is the ONLY candidate
        // (the single-token multiset samples its one token at every
        // permutation, whatever its weight's scale); the strict <
        // comparison keeps the lowest-hash tie-break for real races.
        let mut best: Option<(f64, u64, f64)> = None;
        for &(hash, w) in &items {
            let mut state = element_stream(&[seed, j as u64, hash]);
            // Gamma(2, 1) = the sum of two standard exponentials; the
            // draw order is pinned: r's pair, then c's pair, then beta.
            let r = exponential_open(&mut state) + exponential_open(&mut state);
            let c = exponential_open(&mut state) + exponential_open(&mut state);
            let beta = uniform_open(&mut state);
            // The floor is the consistency-bearing step (the survey's
            // equation 7): t is the INTEGER active index -- the weight-
            // independent grid the pair (hash, t) rides, so two documents
            // naming the same token at different weights still agree
            // exactly when the token wins both processes.
            let t = (w.ln() / r + beta).floor();
            let y = (r * (t - beta)).exp();
            let z = y * r.exp();
            let score = c / z;
            let wins = match best {
                None => true,
                Some((best_a, _, _)) => score < best_a,
            };
            if wins {
                best = Some((score, hash, t));
            }
        }
        let (_, best_hash, best_t) = best.expect("the item list is non-empty");
        pair[0] = best_hash;
        pair[1] = best_t.to_bits();
    }
    sig
}

/// The weighted engine's surface twin: `text`'s token counts (the crate's
/// one tokenizer, the same stream the classic sweep rides) as the
/// multiset weights, then [`weighted_signature_from_weights`].
pub fn weighted_signature_text(text: &str, num_perm: usize, seed: u64) -> Vec<u64> {
    let weights = token_hash_counts(normalized_word_tokens_stream(text));
    weighted_signature_from_weights(weights, num_perm, seed)
}

/// The token-list twin: each element one occurrence (the caller's
/// spelling of the multiset -- no re-tokenization, so token lists a
/// caller built with a domain-specific segmenter weight exactly as
/// given).
pub fn weighted_signature_tokens(tokens: &[String], num_perm: usize, seed: u64) -> Vec<u64> {
    let weights = token_hash_counts(tokens.iter().map(String::as_str));
    weighted_signature_from_weights(weights, num_perm, seed)
}

/// The generalized-Jaccard estimator over two ICWS signatures: the
/// fraction of permutations whose `(hash, t)` pairs agree (Ioffe's
/// consistency property). Signatures must be equal-length 2k-row pairs;
/// mismatched or odd lengths are a caller bug named in the panic (the
/// binding validates both as ValueErrors).
pub fn weighted_jaccard_estimate(a: &[u64], b: &[u64]) -> f64 {
    assert_eq!(a.len(), b.len(), "signature lengths differ");
    assert!(
        !a.is_empty() && a.len().is_multiple_of(2),
        "not an ICWS signature shape"
    );
    let num_perm = a.len() / 2;
    let agree = (0..num_perm)
        .filter(|&j| a[2 * j] == b[2 * j] && a[2 * j + 1] == b[2 * j + 1])
        .count();
    agree as f64 / num_perm as f64
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
    fn token_count_up_to_matches_the_token_stream() {
        // The allocation-free rewrite's equivalence pin: counting
        // `real_word_segments` must equal counting the lowercased stream
        // (lowercasing never empties, so the stream's empty-filter never
        // fires). Covers empty/whitespace/case/unicode plus the limit cap.
        let samples = [
            "",
            "   ",
            "one two three",
            "Hello, WORLD! one two three",
            "café société naïve",
            "a\u{1f}b c",
            "\u{1f469}\u{200d}\u{1f52c} test",
        ];
        for text in samples {
            let full = normalized_word_tokens_stream(text).count();
            assert_eq!(token_count_up_to(text, usize::MAX), full, "{text:?}");
            for limit in [0, 1, 2, 3, 10] {
                assert_eq!(
                    token_count_up_to(text, limit),
                    full.min(limit),
                    "{text:?} limit={limit}"
                );
            }
        }
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
    #[should_panic(expected = "core sanity budget")]
    fn wide_fillable_sweep_past_the_budget_panics() {
        // The sweep-budget backstop's core half (the binding's ValueError
        // is the other half): a fillable wide window whose sweep would run
        // past SHINGLE_SWEEP_BUDGET is named and aborted BEFORE the sweep,
        // not ground through -- the probe's retention-free count walk is
        // the only work paid. (20000 - 5000 + 1) * 5000 = 75,005,000 > 2^26.
        let text = "w ".repeat(20_000);
        let _ = signature(&text, 8, 5_000, 0);
    }

    #[test]
    fn sweep_budget_gate_shapes() {
        // The budget gate's None cases, probed directly (no sweep runs):
        // widths at or below WIDE_WINDOW_COUNT_FIRST never walk the count,
        // and a stream whose walk ends short of the gate's cap --
        // floor(2^26/s) + s tokens -- is the within-budget case (the
        // sentinel short-circuit's own short-stream shape included).
        // (20000 - 4000 + 1) * 4000 = 64,004,000 < 2^26 passes: the cap
        // for s=4000 is 16777 + 4000 = 20777 > 20000 tokens, so the walk
        // ends short of it and the within-budget proof applies. The
        // over-budget shape reports its minimum provable work at the cap
        // (floor(2^26/5000) + 5000 = 18421; the walk stops there, never
        // reaching the stream's 20000th token).
        let text = "w ".repeat(20_000);
        assert_eq!(sweep_past_budget(&text, 1_024), None);
        assert_eq!(sweep_past_budget("one two three", usize::MAX), None);
        assert_eq!(sweep_past_budget(&text, 4_000), None);
        assert_eq!(sweep_past_budget(&text, 5_000), Some(67_110_000));
        // The sentinel answers the gate exists to protect are untouched:
        // the wide short-stream and huge-width shapes still return
        // without ever reaching the assert.
        assert_eq!(signature(&text, 8, usize::MAX, 0), vec![u64::MAX; 8]);
        assert_eq!(distinct_shingle_count("one two three", usize::MAX), 0);
    }

    #[test]
    fn sweep_budget_boundary_is_exact_on_both_sides() {
        // H1: the budget boundary itself. s = 4096 divides 2^26, so
        // tokens = 2^26/4096 + 4096 - 1 = 20479 gives work exactly
        // 16384 * 4096 = 2^26 -- the call PROCEEDS (the budget is a
        // ceiling, not an exclusive bound: the gate answers None, so the
        // core backstop cannot fire and the sweep runs -- the release-side
        // execution pin is tests/test_minhash.py's exactly-at golden) --
        // and one more token tips (20480 - 4096 + 1) * 4096 = 67,112,960
        // > 2^26 into the reject, reported at the cap (here the cap IS
        // the stream: 20480 tokens, so the reported minimum is also the
        // exact spend).
        let at_budget = "w ".repeat(20_479);
        assert_eq!(sweep_past_budget(&at_budget, 4_096), None);
        let one_past = "w ".repeat(20_480);
        assert_eq!(sweep_past_budget(&one_past, 4_096), Some(67_112_960));
    }

    #[test]
    fn sweep_budget_reject_walk_is_capped_not_o_tokens() {
        // H2: the reject path's count walk stops at the cap
        // (floor(2^26/5000) + 5000 = 18421 tokens), so a stream ten times
        // deeper than the cap pays the SAME bounded walk and reports the
        // SAME minimum work -- 67,110,000, not the (200000 - 5000 + 1) *
        // 5000 = 975,005,000 a full-stream walk would have computed. The
        // pinned value is the deterministic proof the walk never scales
        // with the stream: pre-fix this probe walked all 200k tokens
        // GIL-held (measured 65.8ms per 1M tokens) before rejecting.
        let deep = "w ".repeat(200_000);
        assert_eq!(sweep_past_budget(&deep, 5_000), Some(67_110_000));
        // The widest admissible width caps the same way: s = 1025 walks at
        // most floor(2^26/1025) + 1025 = 65472 + 1025 = 66497 tokens.
        assert_eq!(sweep_past_budget(&deep, 1_025), Some((65_472 + 1) * 1_025));
    }

    #[test]
    fn sweep_budget_cap_arithmetic_cannot_overflow() {
        // H3: the gate's only multiplication is (floor(B/s) + 1) * s in
        // u128. For s > B the first factor is 1; for s <= B it is at most
        // B/1025 + 1 -- so the product is bounded by ~(2^26/1025 + 1) *
        // usize::MAX < 2^91, u128 headroom to spare (and the pre-cap
        // exact-work spelling was safe too: both factors < 2^64 gives
        // (2^64-1)^2 < 2^128). Pinned across the admissible widths
        // including the usize extremes, with the over-budget property the
        // cap exists to prove.
        for &s in &[1_025usize, 4_096, 65_536, 500_000, usize::MAX] {
            let s128 = s as u128;
            let cap = usize::try_from(SHINGLE_SWEEP_BUDGET / s128 + s128)
                .expect("cap fits usize for every admissible width");
            let work_at_cap = (cap as u128 - s128 + 1) * s128;
            assert!(
                work_at_cap > SHINGLE_SWEEP_BUDGET,
                "cap work for s={s} does not prove the reject"
            );
            assert!(
                work_at_cap < 2u128.pow(91),
                "cap work for s={s} near u128 limits"
            );
        }
        // The exact-work spelling the gate replaced, at the u128 extremes
        // it could have been fed: both factors bounded by usize::MAX, so
        // the product is at most (2^64-1)^2 = 2^128 - 2^65 + 1 -- strictly
        // inside u128, no overflow (debug panic or release wrap) possible.
        let max = usize::MAX as u128; // 2^64 - 1
        assert_eq!(max * max, u128::MAX - (1u128 << 65) + 2);
        assert!((max - 1) * max < u128::MAX);
    }

    #[test]
    fn fuzz_domain_stays_under_the_sweep_budget() {
        // H6: the fuzz target's panic-freedom lane must never reach the
        // core backstop's assert. Its widest legal shape is 16 KiB of
        // single-character tokens -- 8192 tokens -- at the top of its
        // shingle_size range (u16 % 1030 -> 1029): work =
        // (8192 - 1029 + 1) * 1029 = 7,372,756 token-hashes, ~9x under
        // the budget. Widening either fuzz bound re-balances this pin.
        let max_tokens: usize = (16 * 1024_usize).div_ceil(2); // "a b c ...": 2n-1 bytes for n tokens
        let work = (max_tokens as u128 - 1029 + 1) * 1029;
        assert_eq!(max_tokens, 8192);
        assert!(
            work < SHINGLE_SWEEP_BUDGET,
            "fuzz domain reaches the backstop"
        );
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

    #[test]
    fn superminhash_determinism_sentinels_and_row_range() {
        // The determinism contract carries over: same text, same
        // parameters, identical rows across calls and objects; different
        // seed draws a different stream.
        let text = "the quick brown fox jumps over the lazy dog";
        let a = superminhash_signature(text, 128, 3, 0);
        assert_eq!(a, superminhash_signature(text, 128, 3, 0));
        assert_eq!(a, superminhash_signature(text, 128, 3, 0));
        assert_ne!(a, superminhash_signature(text, 128, 3, 1));
        // The empty-shingle-set convention: the u64 MAX sentinel,
        // seed-invariant, exactly the classic engine's shape.
        for text in ["", "   ", "one two"] {
            assert_eq!(superminhash_signature(text, 8, 3, 0), vec![u64::MAX; 8]);
        }
        assert_eq!(superminhash_signature("", 8, 12345, 0), vec![u64::MAX; 8]);
        // Real rows decode to finite values in [0, m) (the paper's r + j
        // range): the f64 bit patterns never touch the sentinel pattern
        // (u64 MAX is a NaN pattern; h is finite).
        for &v in &a {
            assert_ne!(v, u64::MAX);
            let h = f64::from_bits(v);
            assert!(
                h.is_finite() && (0.0..128.0).contains(&h),
                "row {v} decodes to {h}"
            );
        }
        // The classic engine's num_perm prefix property does NOT hold:
        // the permutation structure depends on m, so k=8 is not a prefix
        // of k=128 (pinned so the documented difference cannot silently
        // become an accidental equality claim).
        assert_ne!(
            superminhash_signature(text, 8, 3, 0),
            superminhash_signature(text, 128, 3, 0)[..8]
        );
        // Case and whitespace shape are invisible (same token stream).
        assert_eq!(
            superminhash_signature("Hello, WORLD! one two three", 64, 3, 0),
            superminhash_signature("hello, world! one\ttwo\nthree", 64, 3, 0)
        );
    }

    #[test]
    fn superminhash_agreement_orders_the_fixture_pairs() {
        // The estimator over the same three fixture pairs the classic
        // engine pins: near-duplicates recall high, disjoint pairs recall
        // ~zero, and the ordering is wide. Counts are deterministic (the
        // sweep order is pinned), so these are exact integers.
        let base_sentence = "The quarterly oil sample interval for field outages was adjusted \
                             after the bushing torque specifications changed. Maintenance \
                             windows now close within fourteen days. ";
        let near_a = base_sentence.repeat(3);
        let near_b = near_a
            .clone()
            .replace("quarterly", "monthly")
            .replace("bushing", "insulator");
        let disjoint_a = "the quick brown fox jumps over the lazy dog ".repeat(6);
        let disjoint_b =
            "compiler backends schedule instructions over directed acyclic graphs ".repeat(6);
        let agreement = |a: &str, b: &str| {
            let sig_a = superminhash_signature(a, 128, 3, 0);
            let sig_b = superminhash_signature(b, 128, 3, 0);
            sig_a.iter().zip(&sig_b).filter(|(x, y)| x == y).count()
        };
        // Exact J 0.6129; the classic engine's estimate is 74/128.
        assert_eq!(agreement(&near_a, &near_b), 75);
        // Disjoint sets: zero agreement, deterministically.
        assert_eq!(agreement(&disjoint_a, &disjoint_b), 0);
    }

    #[test]
    fn bbit_estimator_matches_the_manual_correction() {
        let a = vec![0b1010u64, 5, 7];
        let b = vec![0b0010u64, 5, 9];
        // bits=None: the plain agreement fraction.
        assert!((jaccard_estimate(&a, &b, None) - 1.0 / 3.0).abs() < 1e-15);
        // bits=1: masks 0b10|0b11... rows: 0 vs 0, 1 vs 1, 1 vs 1: all
        // agree, p_hat = 1, corrected (1 - 1/2)/(1 - 1/2) = 1.
        assert!((jaccard_estimate(&a, &b, Some(1)) - 1.0).abs() < 1e-15);
        // bits=2: 0b10 vs 0b10, 0b01 vs 0b01, 0b11 vs 0b01: p_hat = 2/3,
        // corrected (2/3 - 1/4)/(3/4) = 5/9.
        assert!((jaccard_estimate(&a, &b, Some(2)) - 5.0 / 9.0).abs() < 1e-12);
        // The corrected estimator may go negative at true zero (the
        // chance term exceeds the observed agreement): honest, documented.
        let neg = jaccard_estimate(&[0u64, 0], &[1u64, 1], Some(1));
        assert!(neg < 0.0, "{neg}");
        // Identical signatures estimate 1.0 at every bits value.
        for bits in [None, Some(1), Some(8), Some(63)] {
            assert!((jaccard_estimate(&a, &a, bits) - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    #[should_panic(expected = "signature lengths differ")]
    fn bbit_estimator_panics_on_a_length_mismatch() {
        let _ = jaccard_estimate(&[1u64], &[1u64, 2], None);
    }

    #[test]
    fn weighted_icws_pins_the_single_token_and_sentinel_conventions() {
        // One token, weight 1: EVERY permutation samples it (the argmin
        // over one candidate), and t = floor(ln(1)/r + beta) = floor(beta)
        // = 0, so the signature is (hash, 0) repeated. h is the
        // shingle_size-1 shingle hash of the token.
        let h = hash_tokens(1, std::iter::once("hello"));
        let sig = weighted_signature_text("hello", 8, 0);
        assert_eq!(sig, vec![h, 0u64, h, 0, h, 0, h, 0, h, 0, h, 0, h, 0, h, 0]);
        // Empty multiset conventions: empty text, whitespace-only text
        // (no tokens), an empty weight list, and an all-zero weight list
        // are the u64 MAX sentinel, 2 rows per permutation.
        let sentinel = vec![u64::MAX; 16];
        assert_eq!(weighted_signature_text("", 8, 0), sentinel);
        assert_eq!(weighted_signature_text("   ", 8, 0), sentinel);
        assert_eq!(weighted_signature_from_weights(vec![], 8, 0), sentinel);
        assert_eq!(
            weighted_signature_from_weights(vec![(h, 0.0)], 8, 0),
            sentinel
        );
        // Seed-invariant on the empty multiset.
        assert_eq!(weighted_signature_text("", 8, 42), sentinel);
        // Determinism across calls; a different seed draws differently
        // over a real multiset.
        let a = weighted_signature_text("the quick brown fox jumps", 64, 0);
        assert_eq!(
            a,
            weighted_signature_text("the quick brown fox jumps", 64, 0)
        );
        assert_ne!(
            a,
            weighted_signature_text("the quick brown fox jumps", 64, 1)
        );
    }

    #[test]
    fn weighted_icws_sees_token_frequency_the_binary_engine_cannot() {
        // The WHAT-it-adds pin: "aaa bbb" and "aaa aaa bbb" hold the SAME
        // token set, so the binary engine (shingle_size 1) sees identical
        // signatures and estimates J = 1 -- frequency is invisible to it.
        // The weighted engine reads the counts (1,1) vs (2,1), whose
        // generalized Jaccard is exactly 2/3, and estimates near it.
        let binary_a = signature("aaa bbb", 128, 1, 0);
        let binary_b = signature("aaa aaa bbb", 128, 1, 0);
        assert_eq!(binary_a, binary_b);
        let a = weighted_signature_text("aaa bbb", 64, 0);
        let b = weighted_signature_text("aaa aaa bbb", 64, 0);
        let est = weighted_jaccard_estimate(&a, &b);
        assert!((est - 2.0 / 3.0).abs() < 0.15, "{est}");
        // Identical multisets estimate exactly 1.0.
        let est_self = weighted_jaccard_estimate(&a, &a);
        assert!((est_self - 1.0).abs() < 1e-15);
        // The estimator averages to the true generalized Jaccard over the
        // fixture count vectors (the empirical anchor the Python RMSE
        // grid scales up): (2,0,1) vs (1,1,1) has true J
        // (1+0+1)/(2+1+1) = 0.5.
        let c = weighted_signature_text("x x z", 256, 0);
        let d = weighted_signature_text("x y z", 256, 0);
        let est_cd = weighted_jaccard_estimate(&c, &d);
        assert!((est_cd - 0.5).abs() < 0.1, "{est_cd}");
    }

    #[test]
    #[should_panic(expected = "signature lengths differ")]
    fn weighted_jaccard_estimate_panics_on_a_length_mismatch() {
        let _ = weighted_jaccard_estimate(&[1u64, 2], &[1u64]);
    }

    #[test]
    #[should_panic(expected = "not an ICWS signature shape")]
    fn weighted_jaccard_estimate_panics_on_an_odd_length() {
        let bad = [1u64, 2, 3];
        let _ = weighted_jaccard_estimate(&bad, &bad);
    }
}
