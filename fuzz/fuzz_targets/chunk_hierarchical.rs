//! The chunking family's cross-function invariants on arbitrary text:
//! `chunk_hierarchical` (any separator hierarchy, any budget, any
//! overlap), `chunk_by_words`/`chunk_by_sentences` (the unit-count
//! spellings whose merge step shares the same grapheme boundary index),
//! and `chunk_by_paragraphs` (basic contract only: its paragraph spans
//! are line-run edges, documented as not necessarily grapheme-aligned —
//! a combining mark after a newline joins the newline's cluster, and the
//! newline is separator content no paragraph's caller would call "split").
//!
//! For the three cluster-safe chunkers, every chunk start and end must be
//! a grapheme-cluster boundary per `unicode-segmentation` directly — the
//! independent oracle for the shared `GraphemeIndex` bitmap (including
//! its ASCII fast path), not the production code re-answering its own
//! question — and a chunk exceeding `max_chars` is legal ONLY as exactly
//! one whole grapheme cluster (the documented oversized-cluster
//! exception). Starts strictly increase: forward progress at the
//! sequence level.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Arbitrary, Debug)]
struct Input {
    text: String,
    max_chars: std::num::NonZeroU16,
    overlap_raw: u16,
    separators: Option<Vec<String>>,
}

/// The chunk positions that must be grapheme-cluster boundaries for the
/// cluster-safe chunkers, checked against `unicode-segmentation` itself.
fn assert_cluster_safe(chunks: &[(usize, usize)], text: &str, budget: usize, what: &str) {
    // Every cluster start, plus the end-of-text boundary (a final chunk
    // legitimately ends there): the same entry set the production
    // GraphemeIndex encodes, built here by the segmenter itself.
    let mut boundaries: Vec<usize> = text
        .graphemes(true)
        .scan(0usize, |cp, cluster| {
            let at = *cp;
            *cp += cluster.chars().count();
            Some(at)
        })
        .collect();
    boundaries.push(text.chars().count());
    for &(start, end) in chunks {
        let Ok(idx) = boundaries.binary_search(&start) else {
            panic!("{what}: chunk start {start} is mid-cluster: {chunks:?} on {text:?}")
        };
        assert!(
            boundaries.binary_search(&end).is_ok(),
            "{what}: chunk end {end} is mid-cluster: {chunks:?} on {text:?}"
        );
        if end - start > budget {
            assert_eq!(
                boundaries[idx + 1],
                end,
                "{what}: chunk ({start}, {end}) exceeds budget {budget} without being one \
                 whole grapheme cluster: {chunks:?} on {text:?}"
            );
        }
    }
}

fn assert_basic_contract(chunks: &[(usize, usize)], total: usize, what: &str) {
    let mut prev_start = None;
    for &(start, end) in chunks {
        assert!(start < end, "{what}: empty or inverted chunk: ({start}, {end})");
        assert!(end <= total, "{what}: chunk end {end} exceeds text length {total}");
        if let Some(prev) = prev_start {
            assert!(start > prev, "{what}: starts not strictly increasing at {start}");
        }
        prev_start = Some(start);
    }
}

fuzz_target!(|input: Input| {
    let total = input.text.chars().count();
    let max_chars = input.max_chars.get() as usize;
    // The functions' own precondition is `overlap < max_chars` (or
    // `per_chunk`); clamp rather than skip so the harness still explores
    // the boundary.
    let overlap = (input.overlap_raw as usize) % max_chars.max(1);

    let sep_refs: Option<Vec<&str>> = input
        .separators
        .as_ref()
        .map(|v| v.iter().map(String::as_str).collect());
    let sep_slice: Option<&[&str]> = sep_refs.as_deref();

    let chunks = tors::chunk_hierarchical_impl::chunk_hierarchical(
        &input.text,
        max_chars,
        sep_slice,
        overlap,
    );
    assert_basic_contract(&chunks, total, "chunk_hierarchical");
    assert_cluster_safe(&chunks, &input.text, max_chars, "chunk_hierarchical");

    // The unit-count chunkers over the same arbitrary text: same
    // per-chunk/overlap envelope (per_chunk in 1..=u16, overlap clamped),
    // cluster safety for the two merge-based spellings, basic contract
    // for the line-run paragraph heuristic.
    for per_chunk in [1usize, 2, 3, 7, max_chars] {
        let overlap = overlap % per_chunk;
        let words =
            tors::chunk_by_segment_impl::chunk_by_words(&input.text, per_chunk, overlap);
        assert_basic_contract(&words, total, "chunk_by_words");
        assert_cluster_safe(&words, &input.text, usize::MAX, "chunk_by_words");

        let sentences =
            tors::chunk_by_segment_impl::chunk_by_sentences(&input.text, per_chunk, overlap);
        assert_basic_contract(&sentences, total, "chunk_by_sentences");
        assert_cluster_safe(&sentences, &input.text, usize::MAX, "chunk_by_sentences");

        let paragraphs =
            tors::chunk_by_segment_impl::chunk_by_paragraphs(&input.text, per_chunk, overlap);
        assert_basic_contract(&paragraphs, total, "chunk_by_paragraphs");
    }
});
