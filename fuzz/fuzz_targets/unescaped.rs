//! `find_unescaped` never panics on arbitrary backslash-dense bytes, its
//! answer is exactly the naive parity walk's (an independent oracle
//! re-derived here: `bytes.find`-loop over occurrences with a full backward
//! run count per hit — the Python suite's `reference_find_unescaped`,
//! tests/reference.py — no engine reuse for the occurrence search beyond
//! the standard one-shot `memmem::find`, no carried state), and a returned
//! offset really holds the needle behind an even backslash run — the
//! invariants the Python gates pin (tests/test_unescaped_scan.py), checked
//! here at raw-byte depth over an input space the hypothesis strategies
//! cannot shape. The contains spelling's invariant (contains == an offset
//! was found) is the Option-equality's `is_some` half against the same
//! oracle at this level (the core has one scan; the exported pair's
//! agreement is pinned at the Python level, where the two spellings are
//! distinct functions).

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    haystack: Vec<u8>,
    needle: Vec<u8>,
    // Dual-mode bit: folded backslash-dense bytes exercise runs/overlap/
    // adjacency at depth; raw bytes exercise high-byte/NUL/multibyte
    // fragments the 4-symbol fold can never spell (arbitrary bytes are
    // in-contract input).
    raw: bool,
}

// Map arbitrary bytes onto a 4-symbol backslash-dense alphabet: uniform
// random bytes almost never spell escape-shaped content, so every byte is
// folded onto {backslash, u, 0, a} (~25% backslashes) — long runs,
// needle-shaped sequences, and adjacency all occur at real density.
const ALPHABET: [u8; 4] = [b'\\', b'u', b'0', b'a'];

fn fold_dense(bytes: Vec<u8>) -> Vec<u8> {
    bytes
        .into_iter()
        .map(|b| ALPHABET[(b % 4) as usize])
        .collect()
}

// The find-loop oracle (the Python suite's reference_find_unescaped,
// spelled in Rust): `memmem::find` from the resume origin to the next
// occurrence, a full backward run count per hit, resume one byte past each
// rejected hit — O(n + hits * run), the same shape as `bytes.find` in the
// Python reference, so the differential runs unbounded with the fast core
// (the old per-position naive walk was O(n * m) and needed a 4096 cap that
// blinded large inputs to false negatives; libfuzzer speed is kept — the
// wall cells prove the same loop at 12 MiB in Python).
fn oracle_find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let mut pos = 0;
    while pos <= haystack.len().saturating_sub(needle.len()) {
        let rel = memchr::memmem::find(&haystack[pos..], needle)?;
        let hit = pos + rel;
        let run = haystack[..hit]
            .iter()
            .rev()
            .take_while(|&&b| b == b'\\')
            .count();
        if run % 2 == 0 {
            return Some(hit);
        }
        pos = hit + 1;
    }
    None
}

fuzz_target!(|input: Input| {
    // The empty needle is the wrapper's ValueError("empty needle"), not
    // this target's contract (the core documents it as a precondition);
    // everything else is fair game, all-backslash haystacks included.
    let haystack = if input.raw {
        input.haystack
    } else {
        fold_dense(input.haystack)
    };
    let needle = if input.raw {
        input.needle
    } else {
        fold_dense(input.needle)
    };
    if needle.is_empty() {
        return;
    }
    let got = tors::scan_impl::find_unescaped(&haystack, &needle);

    // The structural check, every input size: a returned offset really
    // holds the needle and sits behind an even backslash run.
    if let Some(hit) = got {
        assert!(
            haystack[hit..].starts_with(&needle),
            "offset {hit} does not hold the needle"
        );
        let run = haystack[..hit]
            .iter()
            .rev()
            .take_while(|&&b| b == b'\\')
            .count();
        assert_eq!(run % 2, 0, "offset {hit} sits behind an odd run of {run}");
    }

    // The oracle differential, unbounded: the find-loop oracle is
    // O(n + hits * run), so large inputs verify exactly like small ones.
    // One assertion pins both invariants: the offset equality (find) and,
    // as its is_some half, the contains invariant.
    assert_eq!(
        got,
        oracle_find(&haystack, &needle),
        "find_unescaped diverged from the parity oracle"
    );
});
