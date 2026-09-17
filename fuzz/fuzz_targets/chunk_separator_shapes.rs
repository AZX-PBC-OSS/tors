//! `chunk_hierarchical`'s structural contract on SEPARATOR-SHAPE-hostile
//! hierarchies: the raw byte-drain generator in the `chunk_hierarchical`
//! target derives arbitrary separator strings, which essentially never
//! match arbitrary text — so the shapes where separator handling actually
//! bites (repeats, overlaps, lists whose entries are prefixes of each
//! other, empty strings, `None` splices, entries as long as the budget's
//! window) are reached by accident at best. This target generates the
//! separator LISTS from a shape grammar over a tiny literal alphabet
//! (`"a"`, `"b"`, `"ab"`, `"ba"`, `"abab"`, `"aa"`, `"bb"`) matched
//! against text built from the same alphabet, so matches are dense and
//! every list shape fires at useful rates.
//!
//! Invariants (the documented contract, calibrated to what the impl
//! guarantees — see the note at the bottom for the one material rule
//! deliberately left off):
//!
//! * forward progress: starts strictly increase, ends never move
//!   backward (strictly advance under overlap > 0 — the #83 rule);
//! * every chunk is non-empty, in-range, and either within `max_chars`
//!   or exactly one whole grapheme cluster (the documented
//!   oversized-cluster exception), checked against
//!   `unicode-segmentation` directly, not the production index;
//! * the chunk count is bounded: window starts advance at least one
//!   codepoint per emitted chunk, so `chunks <= total + list + slack` —
//!   a separator-shape pathology that spawns windows without consuming
//!   text trips this.
//!
//! DELIBERATELY NOT ASSERTED (tracked residual of #103's class): "no
//! chunk consists entirely of separator material". The realized-cut path
//! drops separators (#103's fix), but the remainder/hard-cut path still
//! emits a separator's own span as a chunk —
//! `chunk_hierarchical("aaaa", 2, separators=["aa"])` returns `[(2, 4)]`,
//! the separator itself. The documented contract ("the separator itself
//! is dropped between chunks") does not clearly cover the raw-cut
//! remainder, so the differential here pins impl == structure-contract
//! and leaves the material rule off; flip
//! `ASSERT_NO_SEPARATOR_MATERIAL_CHUNKS` when the residual is settled
//! and the corpus canary (tests/test_redteam_corpus.py,
//! `finding-chunk-hier-separator-chunk`) goes green with it.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use unicode_segmentation::UnicodeSegmentation;

/// The one material rule left off (see module header).
#[allow(dead_code)]
const ASSERT_NO_SEPARATOR_MATERIAL_CHUNKS: bool = false;

/// A separator literal over the shared alphabet. `None` splices the
/// default hierarchy (the production `None` entry), `Empty` is the
/// no-op literal the slot construction must keep inert.
#[derive(Arbitrary, Debug)]
enum SepShape {
    None,
    Empty,
    A,
    B,
    Ab,
    Ba,
    Abab,
    Aa,
    Bb,
}

impl SepShape {
    fn literal(&self) -> Option<&'static str> {
        match self {
            SepShape::None => None,
            SepShape::Empty => Some(""),
            SepShape::A => Some("a"),
            SepShape::B => Some("b"),
            SepShape::Ab => Some("ab"),
            SepShape::Ba => Some("ba"),
            SepShape::Abab => Some("abab"),
            SepShape::Aa => Some("aa"),
            SepShape::Bb => Some("bb"),
        }
    }
}

/// A text piece over the same alphabet (plus boundary material), so
/// every generated separator literal matches at useful rates.
#[derive(Arbitrary, Debug)]
enum TextPiece {
    Ab,
    Abab,
    Aa,
    Bb,
    Word,
    Newline,
    Space,
    X, // a non-alphabet letter: never separator material, always content
}

impl TextPiece {
    fn push(&self, out: &mut String) {
        match self {
            TextPiece::Ab => out.push_str("ab"),
            TextPiece::Abab => out.push_str("abab"),
            TextPiece::Aa => out.push_str("aa"),
            TextPiece::Bb => out.push_str("bb"),
            TextPiece::Word => out.push_str("word"),
            TextPiece::Newline => out.push('\n'),
            TextPiece::Space => out.push(' '),
            TextPiece::X => out.push('X'),
        }
    }
}

#[derive(Arbitrary, Debug)]
struct Input {
    text: Vec<TextPiece>,
    max_chars: std::num::NonZeroU16,
    overlap_raw: u16,
    /// Lists up to four entries deep: enough for repeats, prefix chains
    /// (`["a", "ab", "abab"]`), overlaps (`["ab", "ba"]` on `abab`), a
    /// `None` splice in any position, and an empty literal in any
    /// position — the shapes the byte-drain generator misses.
    list_a: Vec<SepShape>,
    list_b: Vec<SepShape>,
}

fn assert_contract(
    chunks: &[(usize, usize)],
    text: &str,
    total: usize,
    budget: usize,
    overlap: usize,
    separators: &[Option<&str>],
    what: &str,
) {
    let mut prev_start = None;
    let mut prev_end = None;
    for &(start, end) in chunks {
        assert!(
            start < end,
            "{what}: empty or inverted chunk ({start}, {end})"
        );
        assert!(
            end <= total,
            "{what}: chunk end {end} exceeds text length {total}"
        );
        if let Some(prev) = prev_start {
            assert!(
                start > prev,
                "{what}: starts not strictly increasing at {start}"
            );
        }
        if let Some(prev) = prev_end {
            assert!(
                if overlap > 0 { end > prev } else { end >= prev },
                "{what}: ends not advancing under overlap={overlap} at {end} after {prev}"
            );
        }
        prev_start = Some(start);
        prev_end = Some(end);
    }
    // Budget conformance with the one-whole-cluster exception, against
    // unicode-segmentation directly (the independent cluster oracle).
    let mut boundaries: Vec<usize> = text
        .graphemes(true)
        .scan(0usize, |cp, cluster| {
            let at = *cp;
            *cp += cluster.chars().count();
            Some(at)
        })
        .collect();
    boundaries.push(total);
    for &(start, end) in chunks {
        if end - start > budget {
            let idx = boundaries
                .binary_search(&start)
                .expect("{what}: over-budget chunk start is mid-cluster");
            assert_eq!(
                boundaries.get(idx + 1),
                Some(&end),
                "{what}: chunk ({start}, {end}) exceeds budget {budget} without being one \
                 whole grapheme cluster"
            );
        }
        assert!(
            boundaries.binary_search(&start).is_ok() && boundaries.binary_search(&end).is_ok(),
            "{what}: chunk ({start}, {end}) is not cluster-aligned"
        );
    }
    // Count bound: every emitted chunk's start strictly advances, so
    // starts are distinct codepoints — plus slack for the list's own
    // spliced levels (a generous constant; the pathology this catches is
    // windows spawning without consuming text, which is unbounded).
    assert!(
        chunks.len() <= total + separators.len() + 2,
        "{what}: chunk count {} is unbounded relative to the text ({total} codepoints, \
         {} separator entries)",
        chunks.len(),
        separators.len()
    );
    if ASSERT_NO_SEPARATOR_MATERIAL_CHUNKS {
        for &(start, end) in chunks {
            assert!(
                !separators.iter().any(|sep| sep
                    .map(|s| !s.is_empty() && text[start..end] == *s)
                    .unwrap_or(false)),
                "{what}: chunk ({start}, {end}) consists entirely of separator material"
            );
        }
    }
}

fuzz_target!(|input: Input| {
    let mut text = String::new();
    for piece in &input.text {
        piece.push(&mut text);
    }
    let total = text.chars().count();
    let raw_budget = input.max_chars.get() as usize;
    // A pressure-shaped budget that forces the multi-window fallback walk
    // (the same shaping the chunk_hierarchical target applies): raw byte
    // drains land at-or-above the text length for almost every input,
    // starving the walk a hierarchy exists for.
    let pressure_budget = 1 + (raw_budget - 1) % total.max(1);

    let lists: [Vec<Option<&str>>; 2] = [
        input.list_a.iter().map(|s| s.literal()).collect(),
        input.list_b.iter().map(|s| s.literal()).collect(),
    ];
    // The single-entry lists are the shape-minimal cases (one repeat, one
    // overlap); the empty list is the documented raw-cut-only spelling.
    let singletons: [Vec<Option<&str>>; 3] = [
        vec![Some("ab")],
        vec![Some("ab"), Some("ba"), Some("abab")],
        vec![],
    ];

    for separators in lists.iter().chain(singletons.iter()) {
        for budget in [raw_budget, pressure_budget, total.max(1)] {
            let overlap = input.overlap_raw as usize % budget.max(1);
            // Both overlap-boundary modes run the same contract (#47's
            // keyword joined the signature; the contract — forward
            // progress, budget and cluster conformance, no separator
            // chunk — is mode-independent, so each mode asserts in full).
            for boundary in [
                tors::chunk_hierarchical_impl::OverlapBoundary::Grapheme,
                tors::chunk_hierarchical_impl::OverlapBoundary::Word,
            ] {
                let chunks = tors::chunk_hierarchical_impl::chunk_hierarchical(
                    &text,
                    budget,
                    Some(separators),
                    overlap,
                    boundary,
                );
                assert_contract(
                    &chunks,
                    &text,
                    total,
                    budget,
                    overlap,
                    separators,
                    "chunk_hierarchical",
                );
            }
        }
    }
});
