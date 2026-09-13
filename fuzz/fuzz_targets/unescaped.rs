//! `find_unescaped` never panics on arbitrary backslash-dense bytes, its
//! answer is exactly the naive parity walk's (an independent oracle
//! re-derived here: every position in order, a full backward run count per
//! occurrence, no engine, no carried state), and a returned offset really
//! holds the needle behind an even backslash run — the invariants the
//! Python gates pin (tests/test_unescaped_scan.py), checked here at
//! raw-byte depth over an input space the hypothesis strategies cannot
//! shape. The contains spelling's invariant (contains == an offset was
//! found) is the Option-equality's `is_some` half against the same oracle
//! at this level (the core has one scan; the exported pair's agreement is
//! pinned at the Python level, where the two spellings are distinct
//! functions).

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    haystack: Vec<u8>,
    needle: Vec<u8>,
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

// The naive oracle (the Python suite's reference_find_unescaped, spelled
// in Rust): every position in order, a full backward run count per
// occurrence, no engine, no carried state — agreement between this and
// `find_unescaped` is evidence about the contract, not a shared bug.
fn naive_find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i + needle.len() <= haystack.len() {
        if &haystack[i..i + needle.len()] == needle {
            let mut run = 0;
            let mut j = i;
            while j > 0 && haystack[j - 1] == b'\\' {
                run += 1;
                j -= 1;
            }
            if run % 2 == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

fuzz_target!(|input: Input| {
    // The empty needle is the wrapper's ValueError("empty needle"), not
    // this target's contract (the core documents it as a precondition);
    // everything else is fair game, all-backslash haystacks included.
    let haystack = fold_dense(input.haystack);
    let needle = fold_dense(input.needle);
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

    // The oracle differential, bounded working set (the search target's
    // cap precedent: the naive walk is O(n·m), so large inputs keep
    // exploring the fast core's crash surface while the oracle verifies
    // the bounded ones). One assertion pins both invariants: the offset
    // equality (find) and, as its is_some half, the contains invariant.
    if haystack.len() < 4096 {
        assert_eq!(
            got,
            naive_find(&haystack, &needle),
            "find_unescaped diverged from the naive parity walk"
        );
    }
});
