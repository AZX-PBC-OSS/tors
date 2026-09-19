//! `chunk_text` and its overlapping sibling never panic, and the
//! partition contract holds over every fuzzer string and every
//! (max_chars, overlap, boundary) shape: the chunks are non-empty,
//! in bounds, byte-exact over the text (non-overlap: a lossless covering
//! partition; overlap: strictly-advancing starts, cover to the end).
//! `chunk_text` is the suite's most-reachable API and had NO fuzz target
//! of its own (the hierarchical harness reaches only its segmented
//! cousins) — this target is the API's own.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tors::chunk_impl::{chunk_text, chunk_text_overlapping};
use tors::truncate_impl::Boundary;

/// A structured (max_chars, overlap) pair from the fuzzer bytes: budgets
/// 1..=96 (0 is refused at the binding; the impl's own floor), overlap
/// 0..=budget-1 (the impl's own rule: overlap < max_chars, no forward
/// progress otherwise) — the shapes a caller can actually spell.
fn shape(bytes: &[u8]) -> (usize, usize) {
    let budget = (bytes.first().copied().unwrap_or(0) % 96 + 1) as usize;
    let overlap = (bytes.get(1).copied().unwrap_or(0) as usize) % budget;
    (budget, overlap)
}

fuzz_target!(|data: &[u8]| {
    let (max_chars, overlap) = shape(data);
    let text = String::from_utf8_lossy(data).into_owned();
    let total = text.chars().count();

    for boundary in [Boundary::Word, Boundary::Sentence] {
        // The non-overlapping spelling: a lossless covering partition.
        let chunks = chunk_text(&text, max_chars, boundary);
        let mut expect = 0usize;
        for &(start, end) in &chunks {
            assert!(start < end, "empty chunk: {chunks:?} text={text:?}");
            assert!(
                start == expect,
                "gap/overlap in a non-overlap partition: {chunks:?} text={text:?}"
            );
            assert!(end <= total, "out of bounds: {chunks:?} text={text:?}");
            expect = end;
        }
        assert_eq!(expect, total, "not covering: {chunks:?} text={text:?}");

        // The overlapping spelling: non-empty, in bounds, strictly
        // advancing starts, cover to the end.
        let overlaid = chunk_text_overlapping(&text, max_chars, overlap, boundary);
        let mut prev_start = 0usize;
        for (idx, &(start, end)) in overlaid.iter().enumerate() {
            assert!(
                start < end,
                "empty overlaid chunk {idx}: {overlaid:?} text={text:?}"
            );
            assert!(end <= total, "out of bounds: {overlaid:?} text={text:?}");
            if idx > 0 {
                assert!(
                    start > prev_start,
                    "start not advancing: {overlaid:?} text={text:?}"
                );
            }
            prev_start = start;
        }
        assert_eq!(
            overlaid.last().map(|&(_, e)| e),
            if total == 0 { None } else { Some(total) },
            // The impl's own documented contract: empty text -> [] (no
            // chunks to make); any non-empty text covers to the end.
            "the overlaid spelling must cover to the end: {overlaid:?} text={text:?}"
        );
    }
});
