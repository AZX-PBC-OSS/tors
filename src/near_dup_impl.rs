//! Near-duplicate detection and dedup helpers: the pure-Rust core of
//! `tors.simhash_distance`, `tors.shingle_jaccard`, `tors.shingle_dice`,
//! and `tors.dedup_near_dup` — the comparison layer that sits on top of
//! the similarity primitives this crate already ships.
//!
//! # Prior art
//!
//! - **Shingling + resemblance**: Broder, "Syntactic Clustering of the
//!   Web" (1997) — a document's identity, for near-duplicate purposes, is
//!   the SET of its k-grams ("shingles"), and resemblance is set
//!   similarity (the Jaccard index of the two shingle sets). The shingle
//!   functions below are that definition verbatim; the MinHash core
//!   (`minhash_impl`) is its sampling estimator and cites the same work.
//! - **Proven value at LLM pretraining scale**: Lee et al., "Deduplicating
//!   Training Data Makes Language Models Better"
//!   (<https://arxiv.org/abs/2107.06499>) — near-duplicate removal in a
//!   training corpus improves both training efficiency and memorization
//!   behavior; the paper's pipeline is exactly the shape `dedup_near_dup`
//!   serves (score pairs, keep-first greedy removal) at a scale where the
//!   corpus-side indexing (LSH banding) is caller state.
//! - **Exact + near dedup pipelines**: the BigCode / starcode corpus
//!   preparation practice (exact-hash dedup first, then near-dup passes
//!   with a tuned threshold and human spot-checks) — the operating
//!   procedure `dedup_near_dup(method=...)`'s three methods slot into.
//!
//! # Scope: small candidate sets, no index (the doctrine)
//!
//! `dedup_near_dup` is O(n²) pairwise BY DESIGN and DOCUMENTED — the
//! same small-candidate-set posture as `bm25_rank` (docs/design.md's
//! scope cuts): every call recomputes everything from scratch, holds no
//! table across calls, and exists for the "tens to tens-of-thousands of
//! documents in hand" shape. A corpus-scale near-dup pipeline bands
//! MinHash signatures into an LSH table to avoid paying all pairs; that
//! banding table and candidate store are exactly the persistent index
//! the doctrine cuts out of this crate, so they stay caller state. The
//! quadratic wall is pinned with an explicit budget in
//! `tests/test_scaling_pins.py` and benchmarked at n=100/1k/10k in
//! `benches/near_dup.rs` — documented cost, not a hidden one.
//!
//! # Tokenization and normalization (the house policy)
//!
//! Shingles are WORD shingles: `width` consecutive tokens from the
//! crate's one real-word-token walk (`segmentation_impl::
//! real_word_segments`, the `tf_idf`/`bm25_rank`/`minhash` tokenizer:
//! UAX #29 word segments, whitespace-only segments skipped), and word
//! shingles are the recall-side unit for the same reason
//! `minhash_impl` documents — near-duplicates preserve word sequence
//! (a reflowed or lightly-edited paragraph keeps most of its word
//! k-grams) where character k-grams shift wholesale.
//!
//! Every token is normalized before hashing with the grounding layer's
//! exact matching form (`grounding_impl::fold`, mirrored token for
//! token): each segment's characters case-folded with the full Unicode
//! `char::to_lowercase` mapping, then canonicalized to NFC — lowercasing
//! alone does NOT make NFD "cafe\u{301}" equal NFC "café", and the NFC
//! pass does, so NFC-equivalent inputs produce IDENTICAL token streams
//! and identical scores (pinned below). `dedup_near_dup`'s SimHash
//! method fingerprints the whole folded text for the same reason: all
//! three methods share the one normalization policy, so a method switch
//! cannot silently change what "same text" means. (The raw
//! `simhash64`/`simhash128` surfaces deliberately leave case folding to
//! the caller — `segmentation_impl`'s own note — which is why the dedup
//! core folds first rather than calling them on the raw input.)
//!
//! Each shingle is hashed with the MinHash core's own shingle hash —
//! [`crate::minhash_impl::hash_tokens`], the injective length-prefixed
//! framing over XXH64, reused verbatim so there is exactly one shingle
//! hashing contract in the crate (its WB4/separator-injectivity
//! rationale carries over unchanged). Set operations run over the u64
//! hashes; the differential tests below pin the hash-based sets against
//! exact token-tuple sets, so the 64-bit truncation is measured, not
//! assumed.
//!
//! # The empty-shingle-set convention
//!
//! Empty text, whitespace-only text, or fewer tokens than `width`
//! yields the empty shingle set, and the similarities define the
//! degenerate cases directly: two EMPTY sets are identical (Jaccard of
//! ∅ and ∅ is 1.0 — the empty set is a subset of itself), so two
//! token-free texts dedup together; one empty and one non-empty set
//! score 0.0. The MinHash method's all-sentinel signature convention
//! (`minhash_impl`) lands on the same answers: two sentinels agree at
//! every position, a sentinel against a real signature at none.

use std::collections::HashSet;

use unicode_normalization::UnicodeNormalization;

use crate::minhash_impl::hash_tokens;
use crate::segmentation_impl::real_word_segments;
use crate::simhash_impl::simhash64;

/// The dedup fingerprints' token normalization: the grounding layer's
/// matching form (see the module doc) applied to a whole text — every
/// character lowercased, the result canonicalized to NFC. Public because
/// the fuzz target's naive re-dedup oracle must normalize the way the
/// sweep does (the pyo3 layer's str-in borrows feed the folded text
/// through the same function inside [`dedup_near_dup`]).
pub fn fold_text(text: &str) -> String {
    let lowered: String = text.chars().flat_map(char::to_lowercase).collect();
    lowered.nfc().collect()
}

/// The folded real-word tokens the shingle functions ride: one String per
/// `real_word_segments` segment, each put through [`fold_text`]'s
/// character-level form (lowercase, then NFC). Materialized because the
/// sliding shingle window needs random access; O(tokens) is the input
/// size, so nothing beyond the input is retained.
fn folded_word_tokens(text: &str) -> Vec<String> {
    real_word_segments(text)
        .map(|segment| {
            let lowered: String = segment.chars().flat_map(char::to_lowercase).collect();
            lowered.nfc().collect()
        })
        .collect()
}

/// The set of `width`-token shingle HASHES of `text` (the empty set when
/// `text` yields fewer than `width` tokens — the module doc's empty-set
/// convention). Each window rides the MinHash core's injective framing
/// hash, so a shingle's identity is a function of the window alone.
pub(crate) fn shingle_hashes(text: &str, width: usize) -> HashSet<u64> {
    let mut set = HashSet::new();
    if width == 0 {
        return set;
    }
    let tokens = folded_word_tokens(text);
    if tokens.len() < width {
        return set;
    }
    for window in tokens.windows(width) {
        set.insert(hash_tokens(width as u64, window.iter().map(String::as_str)));
    }
    set
}

/// The shared scoring core of both set similarities: the intersection
/// size of two shingle-hash sets. The callers supply the degenerate-case
/// conventions (which differ: see below), this only counts.
fn intersection_size(a: &HashSet<u64>, b: &HashSet<u64>) -> usize {
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    small.iter().filter(|h| large.contains(*h)).count()
}

/// The Jaccard index of the two texts' `width`-token word-shingle sets:
/// `|A ∩ B| / |A ∪ B|` (Broder 1997's resemblance). Identical shingle
/// sets (identical texts modulo the normalization policy) score 1.0;
/// disjoint sets score 0.0; two EMPTY sets are the 1.0 case (∅ ⊆ ∅) and
/// exactly-one-empty is the 0.0 case. Symmetric by definition; rides the
/// same normalization policy as every surface in this module.
pub fn shingle_jaccard(a: &str, b: &str, width: usize) -> f64 {
    let sa = shingle_hashes(a, width);
    let sb = shingle_hashes(b, width);
    set_jaccard(&sa, &sb)
}

/// [`shingle_jaccard`] over precomputed sets (the dedup sweep's spelling:
/// each text's shingle set is computed once and compared many times).
pub(crate) fn set_jaccard(a: &HashSet<u64>, b: &HashSet<u64>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = intersection_size(a, b);
    inter as f64 / (a.len() + b.len() - inter) as f64
}

/// The Dice coefficient of the two texts' `width`-token word-shingle
/// sets: `2|A ∩ B| / (|A| + |B|)` — the same agreement Jaccard measures,
/// weighted toward the small-set side (a shared sliver of a huge set
/// moves Dice more than Jaccard, which is why dedup pipelines that want
/// a strict "most of the SMALLER document is present" reading prefer it).
/// Identical sets 1.0, disjoint 0.0, two empty sets 1.0,
/// exactly-one-empty 0.0 — the same conventions as [`shingle_jaccard`].
pub fn shingle_dice(a: &str, b: &str, width: usize) -> f64 {
    let sa = shingle_hashes(a, width);
    let sb = shingle_hashes(b, width);
    set_dice(&sa, &sb)
}

/// [`shingle_dice`] over precomputed sets (the dedup sweep's spelling).
pub(crate) fn set_dice(a: &HashSet<u64>, b: &HashSet<u64>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = intersection_size(a, b);
    2.0 * inter as f64 / (a.len() + b.len()) as f64
}

/// The Hamming distance between two simhash fingerprints (`simhash64`'s
/// u64s and `simhash128`'s u128s alike — the comparison is width-blind
/// xor-and-count): the number of bit positions at which they differ.
/// Zero means an identical fingerprint (the near-dup gate's "same token
/// multiset" equality), and the distance is the quantity
/// `dedup_near_dup(method="simhash")` thresholds on. Width checking —
/// refusing to mix a 64-bit with a 128-bit spelling — lives at the pyo3
/// boundary, where the caller's spelling is knowable (a Python int does
/// not carry its width; see `src/py/near_dup.rs`'s honest note).
pub fn fingerprint_hamming(a: u128, b: u128) -> u32 {
    (a ^ b).count_ones()
}

/// The dedup methods' closed set (the `method=` spelling of
/// `dedup_near_dup`; the same closed-set-of-strings convention as
/// `errors=`/`boundary=`). A future method is one arm away — each new
/// one is a permanent similarity-contract commitment, the same discipline
/// every named-set surface here carries.
pub const DEDUP_METHODS: &[&str] = &["simhash", "shingle", "minhash"];

/// `name` -> the [`DedupMethod`] it names, or `Err` naming every valid
/// choice (never a silent no-op on a typo'd method name).
pub fn parse_dedup_method(name: &str) -> Result<DedupMethod, String> {
    match name {
        "simhash" => Ok(DedupMethod::SimHash),
        "shingle" => Ok(DedupMethod::Shingle),
        "minhash" => Ok(DedupMethod::MinHash),
        _ => Err(format!(
            "unrecognized method {name:?}; valid choices are: {}",
            DEDUP_METHODS.join(", ")
        )),
    }
}

/// The `method=` parameter's three spellings. All three share one
/// normalization policy (the module doc's fold) and one greedy
/// keep-first sweep; they differ only in what a pair is scored against:
///
/// - `SimHash`: the Hamming distance between the two 64-bit fingerprints
///   of the folded texts, at most `floor((1 - threshold) * 64)` for a
///   pair to be duplicates (threshold 0.9 -> 6 bits; the measured
///   near-dup band in `simhash_impl`'s tests is the calibration context).
/// - `Shingle`: the EXACT Jaccard index of the folded texts' 3-token
///   word-shingle sets, at least `threshold` for a duplicate pair. The
///   comparison is f64 (the score is an f64 division): a pair whose
///   exact Jaccard rounds up to exactly the threshold merges, at most
///   one ulp of over-merge and never an under-merge.
/// - `MinHash`: the agreement fraction of the two 128-permutation
///   signatures (the `minhash_signature` defaults: `num_perm=128`,
///   `shingle_size=3`, `seed=0`), at least `threshold` — the estimated
///   Jaccard, `O(1/sqrt(128))` standard error, for corpora where the
///   exact sets are too wide to intersect pairwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DedupMethod {
    SimHash,
    Shingle,
    MinHash,
}

/// The MinHash dedup method's fixed signature shape: the
/// `minhash_signature` defaults (see [`DedupMethod`]'s doc). Fixed, not
/// caller-tuned, to keep `dedup_near_dup`'s surface at the doctrine's
/// minimal spelling — a caller needing other shapes calls
/// `minhash_signature` and bands the signatures itself (caller state).
const MINHASH_PERMS: usize = 128;
const MINHASH_SHINGLE_SIZE: usize = 3;
const MINHASH_SEED: u64 = 0;
/// The simhash method's fingerprint width, in bits (a u64 fingerprint).
const SIMHASH_BITS: f64 = 64.0;
/// The shingle method's fixed word-shingle width (the `minhash_signature`
/// default `shingle_size`, so the two shingle-set methods read the same
/// unit; `shingle_jaccard`/`shingle_dice` expose `width=` to callers who
/// want the knob).
const SHINGLE_WIDTH: usize = 3;

/// One text's dedup fingerprint: the precomputed per-text state the pair
/// sweep scores. The sweep is O(n²) PAIR CHECKS but each text's
/// fingerprint is computed exactly once — O(total input) retained, never
/// O(n²) state (the memory pin in `tests/test_near_dup.py` holds the
/// line).
#[derive(Debug)]
enum Fingerprint {
    /// The 64-bit simhash of the folded text.
    Sim(u64),
    /// The folded text's shingle-hash set (the `HashSet` rides the std
    /// `RandomState` hasher like `minhash_impl`'s distinct set; only set
    /// membership and cardinality escape the sweep — both
    /// order-independent — so the outcome stays deterministic).
    Shingle(HashSet<u64>),
    /// The folded text's 128-permutation MinHash signature.
    MinHash(Vec<u64>),
}

impl Fingerprint {
    fn compute(text: &str, method: DedupMethod) -> Self {
        match method {
            DedupMethod::SimHash => Fingerprint::Sim(simhash64(&fold_text(text))),
            DedupMethod::Shingle => Fingerprint::Shingle(shingle_hashes(text, SHINGLE_WIDTH)),
            DedupMethod::MinHash => Fingerprint::MinHash(crate::minhash_impl::signature(
                &fold_text(text),
                MINHASH_PERMS,
                MINHASH_SHINGLE_SIZE,
                MINHASH_SEED,
            )),
        }
    }

    /// Is `self` a near-duplicate of `other` at `threshold`? The pair
    /// check every method's semantics reduce to, with each method's
    /// early-exit where one exists (the simhash popcount is O(1) by
    /// construction; the shingle check's cardinality bound skips the
    /// intersection it cannot pass; the MinHash count aborts the moment
    /// the remaining positions cannot reach the required agreement).
    fn is_duplicate(&self, other: &Fingerprint, threshold: f64) -> bool {
        match (self, other) {
            (Fingerprint::Sim(a), Fingerprint::Sim(b)) => {
                // The similarity-to-distance map, kept monotone and
                // exact at both ends: threshold 1.0 -> 0 bits (fingerprint
                // equality), threshold 0.0 -> 64 (everything matches).
                let max_bits = ((1.0 - threshold) * SIMHASH_BITS).floor() as u32;
                fingerprint_hamming(u128::from(*a), u128::from(*b)) <= max_bits
            }
            (Fingerprint::Shingle(a), Fingerprint::Shingle(b)) => {
                // Early exit: Jaccard is at most min/max (and Dice at
                // most 2*min/(|a|+|b|)), so a cardinality ratio already
                // under threshold decides the pair without touching the
                // sets — the shared-sliver-heavy corpus's cheap reject.
                // Both-empty is the 1.0 case (always a duplicate at any
                // admissible threshold); exactly-one-empty scores 0.0,
                // a duplicate only at threshold 0.0 — both pinned below.
                if a.is_empty() && b.is_empty() {
                    return true;
                }
                if a.is_empty() || b.is_empty() {
                    return threshold == 0.0;
                }
                let bound = a.len().min(b.len()) as f64 / a.len().max(b.len()) as f64;
                if bound < threshold {
                    return false;
                }
                set_jaccard(a, b) >= threshold
            }
            (Fingerprint::MinHash(a), Fingerprint::MinHash(b)) => {
                // agreement >= threshold  <=>  matches >= ceil(threshold * k).
                // Early abort: once matches + remaining < needed, the
                // pair cannot pass however the rest agree.
                let needed = (threshold * MINHASH_PERMS as f64).ceil() as usize;
                let mut matches = 0usize;
                for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                    if x == y {
                        matches += 1;
                    }
                    // Positions left after this one: once even agreeing
                    // everywhere remaining cannot reach `needed`, the
                    // pair cannot pass — the abort the sweep exists for.
                    let remaining = MINHASH_PERMS - i - 1;
                    if matches + remaining < needed {
                        return false;
                    }
                }
                matches >= needed
            }
            // The sweep only ever compares fingerprints of ONE method.
            _ => unreachable!("dedup_near_dup compares fingerprints of one method only"),
        }
    }
}

/// The dedup outcome: three parallel views of one greedy keep-first
/// sweep, all indices into the INPUT order (the result-object style the
/// report surfaces share — every key present every time):
///
/// - `kept`: the representatives, ascending — the first text of each
///   near-duplicate group (greedy keep-first: a text is kept iff no
///   EARLIER kept text is within threshold of it, so input order is the
///   tie-break by construction and the kept list preserves input order).
/// - `dropped`: the absorbed texts, ascending — each dropped text was
///   claimed by the first kept text it matched.
/// - `groups`: a full partition of the indices, in representative order:
///   each group is one kept representative followed by the texts it
///   absorbed (a singleton group is a kept text with no duplicates).
///   `kept == [g[0] for g in groups]` and
///   `sorted(kept + dropped) == range(n)` hold by construction (pinned).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedupOutcome {
    pub kept: Vec<usize>,
    pub dropped: Vec<usize>,
    pub groups: Vec<Vec<usize>>,
}

/// The greedy keep-first near-duplicate dedup over `texts` at
/// `threshold` (in [0, 1]; the pyo3 boundary validates and the core
/// asserts): O(n²) pair checks worst case with the per-method early
/// exits, O(total input) retained — the small-candidate-set scope the
/// module doc states honestly (no LSH banding index; the doctrine
/// forbids one and this core builds none). Deterministic: the same
/// input always yields the same outcome (no hash-order or thread
/// variance), input order is the tie-break, all-identical input keeps
/// exactly the first text, and the empty corpus returns the empty
/// outcome.
pub fn dedup_near_dup(texts: &[String], threshold: f64, method: DedupMethod) -> DedupOutcome {
    assert!(
        threshold.is_finite() && (0.0..=1.0).contains(&threshold),
        "threshold {threshold} outside [0, 1] (the pyo3 binding rejects this before the core)"
    );
    let fingerprints: Vec<Fingerprint> = texts
        .iter()
        .map(|text| Fingerprint::compute(text, method))
        .collect();
    let mut kept: Vec<usize> = Vec::new();
    let mut dropped: Vec<usize> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, fp) in fingerprints.iter().enumerate() {
        // Claimed by the FIRST kept text it matches: scanning `kept` in
        // ascending order makes input order the tie-break, so the sweep
        // is deterministic by construction, not by hash luck.
        let mut claimed: Option<usize> = None;
        for (ki, &rep) in kept.iter().enumerate() {
            if fp.is_duplicate(&fingerprints[rep], threshold) {
                claimed = Some(ki);
                break;
            }
        }
        match claimed {
            Some(ki) => {
                dropped.push(i);
                groups[ki].push(i);
            }
            None => {
                kept.push(i);
                groups.push(vec![i]);
            }
        }
    }
    DedupOutcome {
        kept,
        dropped,
        groups,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The naive exact-set spelling the hash-based core is differentially
    /// pinned against: shingle sets as token TUPLES (no 64-bit hash
    /// truncation anywhere), scored with textbook Jaccard/Dice. Agreement
    /// over the battery is evidence the XXH64 framing hash introduces no
    /// collision on the domain the tests cover, not a shared-bug mirror
    /// (the framing hash itself is pinned to reference vectors in
    /// `minhash_impl`'s tests).
    fn naive_shingle_set(text: &str, width: usize) -> std::collections::HashSet<Vec<String>> {
        let tokens = folded_word_tokens(text);
        if width == 0 || tokens.len() < width {
            return std::collections::HashSet::new();
        }
        tokens.windows(width).map(<[String]>::to_vec).collect()
    }

    fn naive_jaccard(a: &str, b: &str, width: usize) -> f64 {
        let sa = naive_shingle_set(a, width);
        let sb = naive_shingle_set(b, width);
        if sa.is_empty() && sb.is_empty() {
            return 1.0;
        }
        if sa.is_empty() || sb.is_empty() {
            return 0.0;
        }
        let inter = sa.intersection(&sb).count();
        inter as f64 / (sa.len() + sb.len() - inter) as f64
    }

    fn naive_dice(a: &str, b: &str, width: usize) -> f64 {
        let sa = naive_shingle_set(a, width);
        let sb = naive_shingle_set(b, width);
        if sa.is_empty() && sb.is_empty() {
            return 1.0;
        }
        if sa.is_empty() || sb.is_empty() {
            return 0.0;
        }
        let inter = sa.intersection(&sb).count();
        2.0 * inter as f64 / (sa.len() + sb.len()) as f64
    }

    /// The battery: every string over {a, b, space} up to length 5 plus
    /// the crate's tricky non-ASCII rows (NFD accents, CJK, ZWJ emoji,
    /// regional indicators, CRLF, the SARA AM spacing mark), the same
    /// exhaustive domain shape the simhash and minhash cores use.
    fn battery() -> Vec<String> {
        let mut battery: Vec<String> = Vec::new();
        let alphabet = ["a", "b", " "];
        let mut frontier: Vec<String> = vec![String::new()];
        for _ in 0..5 {
            let mut next = Vec::new();
            for c in alphabet {
                for prefix in &frontier {
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
            "Hello, world! One. Two.".to_string(),
            "   \t  ".to_string(),
        ]);
        battery
    }

    #[test]
    fn jaccard_and_dice_agree_with_the_exact_set_reference() {
        // Widths bracket the default (3) and the pair surface's legal
        // range edges (1 and past-the-stream).
        for width in [1usize, 2, 3, 5] {
            for a in &battery() {
                for b in &battery() {
                    let j = shingle_jaccard(a, b, width);
                    let d = shingle_dice(a, b, width);
                    assert!(
                        (j - naive_jaccard(a, b, width)).abs() < 1e-12,
                        "jaccard/naive disagreement for {a:?} vs {b:?} at width {width}: {j}"
                    );
                    assert!(
                        (d - naive_dice(a, b, width)).abs() < 1e-12,
                        "dice/naive disagreement for {a:?} vs {b:?} at width {width}: {d}"
                    );
                    assert!((0.0..=1.0).contains(&j));
                    assert!((0.0..=1.0).contains(&d));
                }
            }
        }
    }

    #[test]
    fn identical_and_disjoint_pin_the_ends() {
        // Identical texts (fresh objects, content-equal): 1.0 both scores.
        for text in [
            "the quick brown fox jumps over the lazy dog",
            "caf\u{e9}",
            "",
        ] {
            assert_eq!(shingle_jaccard(text, text, 3), 1.0);
            assert_eq!(shingle_dice(text, text, 3), 1.0);
            let fresh = text.to_owned();
            assert_eq!(shingle_jaccard(text, &fresh, 3), 1.0);
        }
        // Vocabulary-disjoint texts: 0.0 (no shared word shingles).
        assert_eq!(
            shingle_jaccard("alpha beta gamma", "delta epsilon zeta", 3),
            0.0
        );
        assert_eq!(
            shingle_dice("alpha beta gamma", "delta epsilon zeta", 3),
            0.0
        );
        // Exactly one token-free side: 0.0; both token-free: 1.0.
        assert_eq!(shingle_jaccard("", "alpha beta gamma", 3), 0.0);
        assert_eq!(shingle_jaccard("", "", 3), 1.0);
        assert_eq!(shingle_dice("", "", 3), 1.0);
        assert_eq!(shingle_jaccard("   ", "\t\n", 3), 1.0);
    }

    #[test]
    fn normalization_policy_folds_case_and_nfc() {
        // The grounding policy's two effects, each pinned: case is folded
        // (the tokenizer's lowercase), and NFC-equivalent inputs score
        // identically — the NFD spelling of "café" must behave exactly
        // like its NFC spelling against any third text.
        let base = "The quarterly oil sample interval for field outages";
        let upper = "THE QUARTERLY OIL SAMPLE INTERVAL FOR FIELD OUTAGES";
        assert_eq!(shingle_jaccard(base, upper, 3), 1.0);
        let nfc = "caf\u{e9} soci\u{e9}t\u{e9} na\u{ef}ve";
        let nfd = "cafe\u{301} socie\u{301}te\u{301} na\u{ef}ve";
        assert_ne!(nfc, nfd, "the fixture must be two different spellings");
        assert_eq!(shingle_jaccard(nfc, nfd, 3), 1.0);
        assert_eq!(shingle_dice(nfc, nfd, 3), 1.0);
        // And against a third text, both spellings agree exactly (the
        // "behave identically" half of the invariant, not just the 1.0
        // row).
        let third = "the quarterly oil sample interval for field outages";
        assert_eq!(
            shingle_jaccard(nfc, third, 3),
            shingle_jaccard(nfd, third, 3)
        );
        assert_eq!(shingle_dice(nfc, third, 3), shingle_dice(nfd, third, 3));
        // Whitespace shape is invisible downstream of the segmenter.
        assert_eq!(shingle_jaccard("a b c d e", "a\nb\tc  d e", 3), 1.0);
    }

    #[test]
    fn word_shingles_survive_reflow_but_not_rewrite() {
        // The word-shingle rationale, pinned as a direction: a
        // case/whitespace reflow keeps every word token, so the shingle
        // sets agree exactly; a rewrite does not.
        let base = "the lighthouse keeper checked the lamp every morning before sunrise";
        let reflowed = "THE lighthouse keeper checked the lamp  every\n\tmorning before sunrise";
        assert_eq!(shingle_jaccard(base, reflowed, 3), 1.0);
        let rewritten = "each dawn the warden inspected the wick and trimmed the glass";
        assert!(shingle_jaccard(base, rewritten, 3) < 0.1);
        // The tokenizer's punctuation rule carries over (simhash_impl's
        // own: punctuation-only segments are tokens, deterministic and
        // preserved by small edits): INSERTING punctuation is an edit,
        // not a reflow — it lands between tokens and shifts every window
        // after it. Pinned so the shape is a documented choice, not a
        // surprise.
        let punctuated = "the lighthouse keeper checked the lamp, every morning before sunrise";
        assert!(shingle_jaccard(base, punctuated, 3) < 1.0);
    }

    #[test]
    fn fingerprint_hamming_pins() {
        // Zero and symmetry, the two invariants the API pins at the
        // Python layer too, plus a known-value row.
        let a = crate::simhash_impl::simhash64("the quick brown fox");
        let b = crate::simhash_impl::simhash64("pack my box with five dozen liquor jugs");
        assert_eq!(fingerprint_hamming(u128::from(a), u128::from(a)), 0);
        assert_eq!(
            fingerprint_hamming(u128::from(a), u128::from(b)),
            fingerprint_hamming(u128::from(b), u128::from(a))
        );
        assert_eq!(fingerprint_hamming(0, u128::MAX), 128);
        assert_eq!(fingerprint_hamming(1u128 << 70, 0), 1);
        // Width-blind arithmetic: u64-range values compare exactly as
        // `(a ^ b).count_ones()`.
        assert_eq!(fingerprint_hamming(0b1010, 0b0110), 2);
    }

    #[test]
    fn parse_dedup_method_names_the_accepted_set() {
        for name in DEDUP_METHODS {
            assert!(parse_dedup_method(name).is_ok(), "{name} must parse");
        }
        let err = parse_dedup_method("simhash ").unwrap_err();
        assert!(err.contains("simhash") && err.contains("shingle") && err.contains("minhash"));
    }

    #[test]
    #[should_panic(expected = "outside [0, 1]")]
    fn dedup_core_asserts_the_threshold_bounds() {
        let _ = dedup_near_dup(&["a b c".to_string()], 1.5, DedupMethod::SimHash);
    }

    #[test]
    fn dedup_pins_the_structural_contract() {
        // Empty corpus -> the empty outcome.
        assert_eq!(
            dedup_near_dup(&[], 0.9, DedupMethod::SimHash),
            DedupOutcome {
                kept: vec![],
                dropped: vec![],
                groups: vec![]
            }
        );
        // All-identical -> keep the first only, one group holding every
        // index, for every method.
        let identical: Vec<String> =
            std::iter::repeat_n("same words here".to_string(), 5).collect();
        for method in [
            DedupMethod::SimHash,
            DedupMethod::Shingle,
            DedupMethod::MinHash,
        ] {
            let out = dedup_near_dup(&identical, 0.9, method);
            assert_eq!(out.kept, vec![0], "{method:?}");
            assert_eq!(out.dropped, vec![1, 2, 3, 4], "{method:?}");
            assert_eq!(out.groups, vec![vec![0, 1, 2, 3, 4]], "{method:?}");
        }
        // All-distinct -> every index kept, singleton groups, nothing
        // dropped. The texts share NO word token (unique vocabulary per
        // row), so no shingle overlaps at any width and no method can
        // merge them.
        let distinct: Vec<String> = (0..6)
            .map(|i| format!("doc{i}_tok_a doc{i}_tok_b doc{i}_tok_c doc{i}_tok_d"))
            .collect();
        let out = dedup_near_dup(&distinct, 0.9, DedupMethod::SimHash);
        assert_eq!(out.kept, (0..6).collect::<Vec<_>>());
        assert!(out.dropped.is_empty());
        assert_eq!(out.groups, (0..6).map(|i| vec![i]).collect::<Vec<_>>());
    }

    #[test]
    fn dedup_matches_a_naive_pairwise_reference() {
        // The differential pin: the greedy sweep must agree with the
        // naive O(n²) reference that re-scores each pair from the texts
        // directly (fresh fingerprints per pair, no shared precompute),
        // over a mixed corpus at several thresholds and all three
        // methods.
        let corpus: Vec<String> = [
            "the lighthouse keeper walked the stone steps every morning",
            "The Lighthouse keeper walked the stone steps every morning!",
            "the lighthouse keeper walked the stone steps every evening",
            "compiler backends schedule instructions over graphs",
            "compiler backends schedule instructions over directed graphs",
            "completely unrelated vocabulary about quantum chemistry bonds",
            "",
            "   ",
            "",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        for method in [
            DedupMethod::SimHash,
            DedupMethod::Shingle,
            DedupMethod::MinHash,
        ] {
            for threshold in [0.0f64, 0.5, 0.8, 0.9, 1.0] {
                let out = dedup_near_dup(&corpus, threshold, method);
                // The naive reference, scored from the texts directly.
                let mut naive_kept: Vec<usize> = Vec::new();
                let mut naive_dropped: Vec<usize> = Vec::new();
                let mut naive_groups: Vec<Vec<usize>> = Vec::new();
                for i in 0..corpus.len() {
                    let claimed = naive_kept.iter().position(|&r| {
                        Fingerprint::compute(&corpus[i], method)
                            .is_duplicate(&Fingerprint::compute(&corpus[r], method), threshold)
                    });
                    match claimed {
                        Some(ki) => {
                            naive_dropped.push(i);
                            naive_groups[ki].push(i);
                        }
                        None => {
                            naive_kept.push(i);
                            naive_groups.push(vec![i]);
                        }
                    }
                }
                assert_eq!(
                    (out.kept, out.dropped, out.groups),
                    (naive_kept, naive_dropped, naive_groups),
                    "core/naive disagreement for {method:?} at threshold {threshold}"
                );
            }
        }
    }

    #[test]
    fn dedup_is_deterministic_across_calls_and_input_orders() {
        let mut corpus: Vec<String> = (0..40)
            .map(|i| {
                if i % 3 == 0 {
                    format!("repeated block {i} the same core words repeat here often {i}")
                } else {
                    format!("unique document {i} with vocabulary set number {i} distinct")
                }
            })
            .collect();
        let first = dedup_near_dup(&corpus, 0.85, DedupMethod::SimHash);
        let fresh: Vec<String> = corpus.iter().map(|t| t.clone() + "").collect();
        assert_eq!(first, dedup_near_dup(&fresh, 0.85, DedupMethod::SimHash));
        assert_eq!(first, dedup_near_dup(&corpus, 0.85, DedupMethod::SimHash));
        // Input order is the tie-break: reversing the corpus mirrors the
        // outcome's indices (the same texts, the same groups, renumbered)
        // — the determinism claim's strongest row.
        corpus.reverse();
        let reversed = dedup_near_dup(&corpus, 0.85, DedupMethod::SimHash);
        assert_eq!(reversed.kept.len(), first.kept.len());
        assert_eq!(reversed.dropped.len(), first.dropped.len());
    }

    #[test]
    fn dedup_threshold_monotonicity_fewer_drops_as_threshold_rises() {
        // The monotonicity invariant, at its two honest levels.
        //
        // PAIR level (exact, every method): is_duplicate is monotone in
        // the threshold — a pair that duplicates at t duplicates at
        // every t' <= t (simhash's max_bits only grows as t falls, the
        // shingle bound/needed count only tighten as t rises), so the
        // boolean over an ascending threshold ladder is a single step
        // false -> true, never true -> false.
        let pairs: Vec<(String, String)> = battery()
            .iter()
            .zip(battery().iter().rev())
            .map(|(a, b)| (a.clone(), b.clone()))
            .collect();
        for method in [
            DedupMethod::SimHash,
            DedupMethod::Shingle,
            DedupMethod::MinHash,
        ] {
            for (a, b) in &pairs {
                let (fa, fb) = (
                    Fingerprint::compute(a, method),
                    Fingerprint::compute(b, method),
                );
                let mut seen_false = false;
                for threshold in [0.0f64, 0.3, 0.6, 0.85, 0.95, 1.0] {
                    let dup = fa.is_duplicate(&fb, threshold);
                    if dup {
                        assert!(
                            !seen_false,
                            "{method:?}: {a:?} vs {b:?} duplicated at a \
                             higher threshold after failing a lower one"
                        );
                    } else {
                        seen_false = true;
                    }
                }
            }
        }
        // CORPUS level (the greedy sweep, over a well-separated battery):
        // a higher threshold drops no MORE texts — the drop COUNT is
        // non-increasing as the threshold rises. (The dropped SETS are
        // NOT pinned as nested: greedy keep-first re-associates chains
        // when a representative changes — a text absorbed loosely by one
        // rep at a low threshold can sit strictly inside a different
        // rep's radius at a high one — so set nesting is a property of
        // pair scoring, pinned above, not of the sweep.)
        let corpus: Vec<String> = (0..30)
            .map(|i| match i % 4 {
                0 => format!("duplicate family alpha member {i} shared core text"),
                1 => format!("duplicate family beta member {i} other core text"),
                _ => format!("solo document {i} stands entirely alone here"),
            })
            .collect();
        for method in [
            DedupMethod::SimHash,
            DedupMethod::Shingle,
            DedupMethod::MinHash,
        ] {
            let mut prev = usize::MAX; // the first iteration is unconstrained
            for threshold in [0.0f64, 0.3, 0.6, 0.85, 0.95, 1.0] {
                let out = dedup_near_dup(&corpus, threshold, method);
                assert!(
                    out.dropped.len() <= prev,
                    "{method:?}: threshold {threshold} dropped {} texts, more than the \
                     lower threshold's {prev}",
                    out.dropped.len()
                );
                prev = out.dropped.len();
            }
        }
    }

    #[test]
    fn dedup_partitions_the_indices() {
        // The result-object structural pins: kept == group heads, every
        // index in exactly one group, kept + dropped == 0..n sorted.
        let corpus: Vec<String> = (0..25)
            .map(|i| {
                if i % 5 == 0 {
                    "shared family text that repeats across the corpus".to_string()
                } else {
                    format!("independent row {i} {i} {i} distinct words")
                }
            })
            .collect();
        for method in [
            DedupMethod::SimHash,
            DedupMethod::Shingle,
            DedupMethod::MinHash,
        ] {
            let out = dedup_near_dup(&corpus, 0.9, method);
            assert_eq!(out.kept, {
                let mut v: Vec<usize> = out.groups.iter().map(|g| g[0]).collect();
                v.sort_unstable();
                v
            });
            let mut all: Vec<usize> = out.kept.iter().chain(out.dropped.iter()).copied().collect();
            all.sort_unstable();
            assert_eq!(all, (0..corpus.len()).collect::<Vec<_>>());
            let mut group_flattened: Vec<usize> = out.groups.iter().flatten().copied().collect();
            group_flattened.sort_unstable();
            assert_eq!(group_flattened, all);
            // Each group's head is its minimum (the representative kept
            // first).
            for g in &out.groups {
                assert_eq!(g[0], *g.iter().min().unwrap());
            }
        }
    }

    #[test]
    fn dedup_fingerprints_retain_input_proportional_state() {
        // The memory class the pyo3 doc promises: per-text state is
        // O(input), never O(n²) — a 2x corpus growth must at most double
        // the fingerprint store. Measured via the shingle method's
        // resident set sizes (the widest per-text state of the three).
        let corpus = |n: usize| -> Vec<String> {
            (0..n)
                .map(|i| format!("document {i} with a modest token stream of distinct words"))
                .collect()
        };
        let half = corpus(1_000);
        let full = corpus(2_000);
        // Both sweeps must COMPLETE (no blowup) — and the structural
        // guarantee is by construction: `Fingerprint` holds one value per
        // text. The quadratic pair sweep allocates nothing per pair
        // (popcount / set lookup / signature zip), so there is no n²
        // allocation to measure; the Python-side VmHWM guard in
        // tests/test_near_dup.py pins the end-to-end peak.
        let out_half = dedup_near_dup(&half, 0.9, DedupMethod::Shingle);
        let out_full = dedup_near_dup(&full, 0.9, DedupMethod::Shingle);
        assert_eq!(out_half.kept.len(), 1_000);
        assert_eq!(out_full.kept.len(), 2_000);
    }
}
