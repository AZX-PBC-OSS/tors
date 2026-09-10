use pyo3::prelude::*;

use crate::simhash_impl;

/// `tors.simhash64(text) -> int`: the 64-bit SimHash fingerprint of the
/// text's UAX #29 word segments (the standard token unit). Each token is
/// hashed with a deterministic FNV-1a, not std's per-process-seeded
/// DefaultHasher (a fingerprint that changes between runs breaks every
/// cross-run/cross-machine dedupe use); each of the 64 bits is voted ±1
/// across the tokens, and the bit is set iff the vote is positive (ties → 0).
/// This gives near-duplicate detection that `finalize`'s exact hash cannot
/// answer: Hamming distance `(a ^ b).bit_count()` grows slowly with edit
/// distance, so near-dupes cluster within a few bits. Interpretation
/// guidance is corpus-dependent and must be calibrated per deployment. The
/// measured anchors (pinned in `src/simhash_impl.rs`'s tests):
/// single-word edits at document scale (95-word bases) move the fingerprint
/// ≤ 4 bits; at sentence scale (8-12-word bases) ≤ 14 bits; unrelated
/// sentences sit ≥ 23 bits apart. Order-invariant: a bag-of-words vote, so
/// permuted word order gives the same fingerprint (pinned). Empty or
/// whitespace-only text → 0.
///
/// GIL model: the whole tokenize+hash+vote pass under `py.detach`, then a
/// single int return (no marshalling class, the `grapheme_count` extreme
/// point).
#[pyfunction]
pub fn simhash64(py: Python<'_>, text: &str) -> u64 {
    py.detach(|| simhash_impl::simhash64(text))
}

/// `tors.simhash128(text) -> int`: the 128-bit spelling of `simhash64`,
/// using the same word tokens and the same ±1 vote, with FNV-1a at 128
/// bits (twice the bit positions). What that buys, measured by the same
/// battery and pinned in `src/simhash_impl.rs`'s tests: the unrelated
/// floor widens from 23 bits at 64 bits to 40 at 128, while the near-dup
/// bands grow only sublinearly (document scale 3 bits worst vs 64-bit 4;
/// sentence scale 20 vs 14). The separation between the near-dup band and
/// the unrelated floor widens, which is the property a corpus whose
/// 64-bit bands overlap needs. Same contract otherwise:
/// deterministic across processes (FNV-1a), order-invariant (a
/// bag-of-words vote), empty/whitespace-only text → 0, `(a ^
/// b).bit_count()` at the call site for the distance; thresholds are
/// corpus-dependent, calibrated per deployment.
///
/// GIL model: the `simhash64` classes exactly. The whole
/// tokenize+hash+vote pass under `py.detach`, then a single int return.
#[pyfunction]
pub fn simhash128(py: Python<'_>, text: &str) -> u128 {
    py.detach(|| simhash_impl::simhash128(text))
}
