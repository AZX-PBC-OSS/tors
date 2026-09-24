//! `chunk_to_budget`/`chunk_to_offsets` never panic, and the packing
//! contract holds over every fuzzer string and every (budget, overlap,
//! token-spans) shape: chunks are non-empty and in bounds, starts and
//! ends strictly advance, the last chunk ends at the codepoint length,
//! and every chunk fits the budget by the measurement the packing used —
//! except the documented single-oversized-segment exception (a chunk
//! with no interior word boundary to cut at). The offsets spelling's
//! additive measurement is expressible purely in Rust (contained token
//! pairs), so both spellings drive the real core here: the callback
//! variant through a Rust closure counter (the same shape the pyo3
//! binding wraps a Python callable into), including a non-monotone
//! counter keyed to the span's first byte — the adversarial shape that
//! breaks sum-based or binary-search packers.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tors::chunk_budget_impl::{BudgetError, chunk_to_budget, chunk_to_offsets};
use unicode_segmentation::UnicodeSegmentation;

/// A (budget, overlap) pair from the fuzzer bytes: budgets 1..=96,
/// overlap 0..=budget-1 (the impl's own rule: overlap < budget).
fn shape(bytes: &[u8]) -> (u64, u64) {
    let budget = (bytes.first().copied().unwrap_or(0) % 96 + 1) as u64;
    let overlap = (bytes.get(1).copied().unwrap_or(0) as u64) % budget;
    (budget, overlap)
}

/// Token spans synthesized from the fuzzer bytes: a token covers 1-4
/// codepoints, decided per codepoint from the byte stream (roughly one
/// bit), with whitespace untokenized (gaps — the realistic
/// tokenizer-offsets shape). Sorted and non-overlapping by construction.
fn token_spans(text: &str, bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut token: Option<(usize, usize)> = None;
    for (cp_idx, ch) in text.chars().enumerate() {
        let decision = bytes.get(cp_idx % bytes.len().max(1)).copied().unwrap_or(0);
        if ch.is_whitespace() {
            if let Some((s, e)) = token.take() {
                spans.push((s, e));
            }
            continue;
        }
        token = match token {
            None => Some((cp_idx, cp_idx + 1)),
            Some((s, _)) => Some((s, cp_idx + 1)),
        };
        if decision & 1 == 0
            && let Some((s, e)) = token.take()
        {
            spans.push((s, e));
        }
    }
    if let Some((s, e)) = token.take() {
        spans.push((s, e));
    }
    spans
}

/// A codepoint→byte grid (the packing's offsets are codepoints; Rust
/// slices are bytes).
fn grid(text: &str) -> Vec<usize> {
    text.char_indices()
        .map(|(b, _)| b)
        .chain([text.len()])
        .collect()
}

fn total_of(text: &str) -> usize {
    text.chars().count()
}

fuzz_target!(|data: &[u8]| {
    let (budget, overlap) = shape(data);
    let text = String::from_utf8_lossy(data).into_owned();
    let spans = token_spans(&text, data);

    // The GIL-free spelling: the offsets variant over synthetic spans.
    let chunks = chunk_to_offsets(&text, &spans, budget, overlap)
        .expect("precomputed-offset packing is infallible for valid spans");
    let contained = |start: usize, end: usize| {
        spans
            .iter()
            .filter(|&&(s, e)| s >= start && e <= end)
            .count() as u64
    };
    assert_invariants(&text, &chunks, budget, contained);

    // The callback spelling with a monotone-enough Rust counter (the
    // word-count proxy; never 0, so no ValueError path).
    let text_grid = grid(&text);
    let chunks = chunk_to_budget(&text, budget, overlap, |span| {
        Ok(match span.split_whitespace().count() {
            0 => 1,
            n => n as u64,
        })
    })
    .expect("the word-count proxy never violates the counter contract");
    assert_invariants(&text, &chunks, budget, |s, e| {
        let n = text[text_grid[s]..text_grid[e]].split_whitespace().count();
        if n == 0 { 1 } else { n as u64 }
    });

    // The adversarial spelling: a NON-MONOTONE counter keyed to the
    // span's first byte, unrelated to its length. Progress is anchored
    // to codepoint offsets, never counts, so this must terminate with
    // the same invariants (positive always: no ValueError path).
    let flare = |span: &str| -> Result<u64, BudgetError> {
        Ok(match span.as_bytes().first() {
            Some(&b) if b % 3 == 0 => 80,
            Some(_) => 1,
            None => 1,
        })
    };
    let chunks = chunk_to_budget(&text, budget, overlap, flare)
        .expect("the flare counter is positive always");
    assert_invariants(&text, &chunks, budget, |s, e| {
        match text.as_bytes()[text_grid[s]..text_grid[e]].first() {
            Some(&b) if b % 3 == 0 => 80,
            _ => 1,
        }
    });
});

/// The packing contract, over one chunking: non-empty in-bounds chunks,
/// first start 0, last end the codepoint length, starts and ends
/// strictly advancing, and every chunk fits `budget` by the same
/// measurement the packing used — except the single-oversized-segment
/// exception (a chunk with no interior word boundary to cut at).
fn assert_invariants(
    text: &str,
    chunks: &[(usize, usize)],
    budget: u64,
    measure: impl Fn(usize, usize) -> u64,
) {
    if text.is_empty() {
        assert!(chunks.is_empty(), "empty text yields no chunks: {chunks:?}");
        return;
    }
    assert!(!chunks.is_empty(), "non-empty text yields chunks: {text:?}");
    let grid = grid(text);
    let slice = |s: usize, e: usize| &text[grid[s]..grid[e]];
    let mut prev_start = 0usize;
    let mut prev_end = 0usize;
    for (i, &(start, end)) in chunks.iter().enumerate() {
        assert!(start < end, "empty chunk {i}: {chunks:?} text={text:?}");
        assert!(
            end <= total_of(text),
            "out of bounds: {chunks:?} text={text:?}"
        );
        if i == 0 {
            assert_eq!(start, 0, "first chunk starts at 0: {chunks:?}");
        } else {
            assert!(
                start > prev_start,
                "starts advance: {chunks:?} text={text:?}"
            );
            assert!(end > prev_end, "ends advance: {chunks:?} text={text:?}");
        }
        if measure(start, end) > budget {
            // The documented exception: no interior word boundary.
            assert_eq!(
                slice(start, end).split_word_bound_indices().count(),
                1,
                "budget exceeded without the single-segment exception: \
                 {chunks:?} text={text:?}"
            );
        }
        prev_start = start;
        prev_end = end;
    }
    assert_eq!(
        prev_end,
        total_of(text),
        "covers to the end: {chunks:?} text={text:?}"
    );
}
