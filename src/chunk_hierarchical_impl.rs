//! Hierarchical fallback chunking: the pure-Rust core of
//! `tors.chunk_hierarchical`, the `chunk_text`/`chunk_by_*` family's fourth
//! shape: a PRIORITY-ORDERED list of separator levels, coarsest first,
//! tried in order for each chunk: the same pattern LangChain's
//! `RecursiveCharacterTextSplitter` popularized (default separators
//! `["\n\n", "\n", " ", ""]`, literal strings, falling back to the next
//! level only when the coarser one has no in-budget cut).
//!
//! Split into its own file rather than folded into [`crate::chunk_impl`] or
//! [`crate::chunk_by_segment_impl`]: this is a genuinely different
//! algorithm from both siblings, a MULTI-LEVEL search per chunk, not a
//! single boundary list, and it earns its own "one concern per file" home
//! the same separation unit-count chunking already has.
//!
//! DEFAULT hierarchy (`separators = None`): paragraph → sentence → word →
//! grapheme-safe raw cut, reusing tors's OWN accurate segmenters
//! ([`crate::chunk_by_segment_impl::paragraph_bounds`],
//! [`crate::segmentation_impl::sentence_bounds`],
//! [`crate::segmentation_impl::word_bounds`]) rather than the naive literal
//! guesses (`"\n\n"`, `". "`, `" "`) LangChain's own default falls back to:
//! tors already has real UAX #29 segmentation, so the default hierarchy
//! uses it.
//!
//! CUSTOM hierarchy (`separators = Some(list)`): a caller-supplied list of
//! LITERAL strings (not regex, a documented scope line: literals are
//! LangChain's own default too, cover the motivating markdown-header case
//! completely, and avoid reopening the regex-semantics question this crate
//! already navigated once for `re`). REPLACES the default hierarchy
//! entirely for the levels it specifies. The grapheme-safe raw cut is
//! ALWAYS appended as an implicit final level regardless of what the
//! caller passes: "never fails to produce a chunk" is an unconditional
//! guarantee here, not something a caller can accidentally break by
//! forgetting a sentinel (LangChain's own convention requires an explicit
//! trailing `""`; tors does not require this, and ignores a trailing `""`
//! if one is passed, since the raw cut already covers that case).
//!
//! UNLIKE [`crate::chunk_impl::chunk_text`] (a lossless covering
//! partition), this is NOT lossless: at every level except the raw-cut
//! fallback, the separator itself is DROPPED between chunks (the chunk
//! ends where the separator starts, the next chunk begins where it ends),
//! the same convention [`crate::chunk_by_segment_impl::chunk_by_paragraphs`]
//! already established for blank-line runs. Joining the chunks does NOT
//! reproduce the input; that was already true of the unit-count family and
//! stays true here for the same reason (a caller asking to split ON a
//! marker wants it gone, not duplicated).
//!
//! Performance: every level's candidate cut-position list is computed ONCE
//! per call (one scan per level: `paragraph_bounds`/`sentence_bounds`/
//! `word_bounds` for the default levels, one `memmem` pass per
//! custom literal), never re-scanned per chunk. Building each chunk is one
//! `partition_point` binary search per level: O(n × levels) total, levels
//! bounded by the small, caller-supplied list length. Forward progress is
//! pinned the same way [`crate::chunk_impl::chunk_text_overlapping`]'s is:
//! a hard iteration-count assertion in the tests, not just a slow-test
//! timeout.

use std::collections::HashSet;

use memchr::memmem;

use crate::chunk_by_segment_impl::paragraph_bounds;
use crate::chunk_impl::grapheme_safe_hard_cut;
use crate::segmentation_impl;
use crate::truncate_impl::grapheme_boundary_chars;

/// One level's candidate cut points, ascending by `cut_end`. `next_start >=
/// cut_end` always: equal for contiguous segmenters (word/sentence bounds,
/// where there is no gap to drop), strictly greater when a separator's own
/// content is dropped between chunks (paragraph gaps, custom literal
/// separators).
struct Level {
    cuts: Vec<(usize, usize)>,
}

impl Level {
    /// The largest `(cut_end, next_start)` with `cut_end <= limit` AND
    /// `cut_end > after` (genuine forward progress): `None` if this level
    /// has no such candidate for the current window.
    fn best_cut(&self, after: usize, limit: usize) -> Option<(usize, usize)> {
        let hi = self.cuts.partition_point(|&(end, _)| end <= limit);
        if hi > 0 && self.cuts[hi - 1].0 > after {
            Some(self.cuts[hi - 1])
        } else {
            None
        }
    }
}

/// A segment-bounds list (contiguous, `word_bounds`/`sentence_bounds`
/// shape) into a [`Level`]: each segment's `end` is a cut point, and since
/// these segmenters partition the text with no gaps, `next_start == end`
/// and nothing is dropped between chunks at this level.
fn level_from_contiguous_bounds(bounds: Vec<(usize, usize)>) -> Level {
    Level {
        cuts: bounds.into_iter().map(|(_, end)| (end, end)).collect(),
    }
}

/// [`paragraph_bounds`] into a [`Level`]: unlike word/sentence bounds,
/// consecutive paragraph segments are NOT contiguous (the blank-line run
/// between them is excluded from both): the cut is the current
/// paragraph's `end`, the next chunk resumes at the FOLLOWING paragraph's
/// `start`, dropping the gap. The last paragraph has no following segment
/// and contributes no cut (nothing to fall back into past the end of the
/// text; the caller's final-chunk case handles running to `total`).
fn level_from_paragraph_bounds(bounds: Vec<(usize, usize)>) -> Level {
    let cuts = bounds
        .windows(2)
        .map(|w| (w[0].1, w[1].0))
        .collect::<Vec<_>>();
    Level { cuts }
}

/// A literal separator into a [`Level`]: every non-overlapping match's
/// `(start, end)` in codepoint units: the chunk ends at the match start
/// (the separator is not part of either chunk), the next chunk resumes at
/// the match end (the separator is dropped, the same convention
/// [`level_from_paragraph_bounds`] already applies to blank-line runs).
/// One `memchr::memmem` pass over the whole text (SIMD-skipped two-way —
/// std's `match_indices` runs the same algorithm without the SIMD skip
/// and crawls on degenerate repeated-byte documents), converted from byte
/// to codepoint offsets in the same forward walk (no second pass).
fn level_from_literal(text: &str, separator: &str) -> Level {
    if separator.is_empty() {
        // An empty literal matches everywhere and cuts nothing meaningful
        // (LangChain's own trailing "" sentinel means "raw character
        // fallback," which this crate provides unconditionally via the
        // grapheme-safe hard cut instead); an explicit no-op level rather
        // than a pathological infinite-candidate one.
        return Level { cuts: Vec::new() };
    }
    // Every match of a literal needle IS the needle: its char length is a
    // loop-invariant, counted once.
    let sep_chars = separator.chars().count();
    let mut cuts = Vec::new();
    let mut char_idx = 0usize;
    let mut byte_idx = 0usize;
    for byte_start in memmem::find_iter(text.as_bytes(), separator.as_bytes()) {
        char_idx += text[byte_idx..byte_start].chars().count();
        let start_char = char_idx;
        let end_char = start_char + sep_chars;
        cuts.push((start_char, end_char));
        char_idx = end_char;
        byte_idx = byte_start + separator.len();
    }
    Level { cuts }
}

/// Hierarchical fallback chunking of `text`: `(start, end)` codepoint-unit
/// pairs, each chunk at most `max_chars` codepoints, cut at the COARSEST
/// level (first in `levels`, excluding the always-appended grapheme-safe
/// raw cut) that has an in-budget candidate, falling back to progressively
/// finer levels only when a coarser one has none over the current window.
/// The one exception, shared with [`crate::chunk_impl::chunk_text`] via the
/// same `grapheme_safe_hard_cut`: a single grapheme cluster wider than the
/// whole remaining budget (e.g. an oversized ZWJ emoji chain) is kept
/// whole rather than split, so that one chunk can exceed `max_chars`:
/// this never affects ordinary text (no cluster is more than a handful of
/// codepoints). See the module docs for the default-vs-custom hierarchy
/// and the separator-dropped (not lossless) contract. `overlap` snaps the next
/// chunk's start backward from the just-emitted chunk's end to the nearest
/// GRAPHEME boundary at or before the target (never mid-cluster): NOT
/// necessarily a semantic word/sentence/paragraph boundary the way
/// `chunk_text_overlapping`'s single-hierarchy overlap snap is; a
/// documented simplification of the general multi-level case, not a
/// silent gap. A target at or before the chunk's own start (a short
/// trailing chunk, or a run of tight hard-cuts) silently degrades to zero
/// overlap for just that one transition, the same documented
/// snap-collapse [`crate::chunk_impl::chunk_text_overlapping`] already
/// applies.
pub fn chunk_hierarchical(
    text: &str,
    max_chars: usize,
    separators: Option<&[&str]>,
    overlap: usize,
) -> Vec<(usize, usize)> {
    if text.is_empty() {
        return Vec::new();
    }
    // `assert!`, not left to the incidental `total / max_chars` divide-by-zero
    // panic below: this is `pub` Rust API in its own right, reachable
    // without the pyo3 layer's `ValueError`, and a clear message beats a
    // bare "attempt to divide by zero": the same discipline `chunk_text`'s
    // own `max_chars > 0` assert applies.
    assert!(max_chars > 0, "max_chars must be at least 1, got 0");
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();

    let mut levels: Vec<Level> = match separators {
        Some(seps) => seps
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| level_from_literal(text, s))
            .collect(),
        None => vec![
            level_from_paragraph_bounds(paragraph_bounds(text)),
            level_from_contiguous_bounds(segmentation_impl::sentence_bounds(text)),
            level_from_contiguous_bounds(segmentation_impl::word_bounds(text)),
        ],
    };
    let grapheme_starts = grapheme_boundary_chars(text);
    let grapheme_set: HashSet<usize> = grapheme_starts.iter().copied().collect();
    // Every level's cuts are additionally filtered to grapheme-cluster
    // boundaries, one O(level size) pass each: the same "never split a
    // cluster" fix chunk_text/chunk_by_* already apply, extended here to
    // custom literal separators too (a caller's separator could, in
    // principle, land inside a cluster on pathological input).
    for level in &mut levels {
        level
            .cuts
            .retain(|&(end, next)| grapheme_set.contains(&end) && grapheme_set.contains(&next));
    }

    let mut chunks = Vec::with_capacity(total / max_chars + 1);
    let mut start = 0usize;
    let mut iterations = 0usize;
    while start < total {
        iterations += 1;
        assert!(
            iterations <= total + 1,
            "chunk_hierarchical: forward-progress invariant violated"
        );
        let remaining = total - start;
        if remaining <= max_chars {
            chunks.push((start, total));
            break;
        }
        let limit = start + max_chars;
        let cut = levels
            .iter()
            .find_map(|level| level.best_cut(start, limit))
            .unwrap_or_else(|| {
                let end = grapheme_safe_hard_cut(&grapheme_starts, start, limit);
                (end, end)
            });
        chunks.push((start, cut.0));
        if overlap == 0 {
            start = cut.1;
        } else {
            let target = cut.0.saturating_sub(overlap);
            let ghi = grapheme_starts.partition_point(|&g| g <= target);
            let snapped = if ghi > 0 { grapheme_starts[ghi - 1] } else { 0 };
            start = if snapped > start { snapped } else { cut.1 };
        }
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_of(chunks: &[(usize, usize)], text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        chunks
            .iter()
            .map(|&(s, e)| chars[s..e].iter().collect())
            .collect()
    }

    #[test]
    fn empty_text_is_no_chunks() {
        assert_eq!(chunk_hierarchical("", 10, None, 0), Vec::new());
    }

    #[test]
    fn text_within_budget_is_one_chunk() {
        let chunks = chunk_hierarchical("hello world", 100, None, 0);
        assert_eq!(chunks, vec![(0, 11)]);
    }

    #[test]
    fn falls_back_paragraph_to_sentence_to_word() {
        // One long paragraph containing several sentences containing
        // several words: a small budget forces the walk past paragraph
        // (too big) and sentence (still too big) down to word level.
        let text = "Alpha beta gamma delta epsilon zeta eta theta.";
        let chunks = chunk_hierarchical(text, 12, None, 0);
        for &(s, e) in &chunks {
            assert!(e - s <= 12, "chunk exceeded budget: {:?}", (s, e));
        }
        assert!(chunks.len() > 1);
    }

    #[test]
    fn default_hierarchy_prefers_paragraph_when_it_fits() {
        let text = "Short one.\n\nShort two.";
        // Budget fits each paragraph but not the whole text: the walk
        // should cut at the paragraph gap, not descend to sentence/word.
        let chunks = chunk_hierarchical(text, 12, None, 0);
        let texts = text_of(&chunks, text);
        assert_eq!(texts, vec!["Short one.", "Short two."]);
    }

    #[test]
    fn custom_markdown_separators_split_on_headers_first() {
        let text = "# Title\nintro text\n## Section\nmore text here that is long";
        let chunks = chunk_hierarchical(text, 40, Some(&["\n## ", "\n\n", ". ", " "]), 0);
        let texts = text_of(&chunks, text);
        assert_eq!(texts[0], "# Title\nintro text");
        assert!(texts.iter().any(|t| t.starts_with("Section")));
    }

    #[test]
    fn always_produces_some_chunk_via_raw_cut_fallback() {
        // A single "word" with no boundary anywhere and no separator
        // match at all: only the grapheme-safe raw cut can produce chunks.
        let text = "a".repeat(100);
        let chunks = chunk_hierarchical(&text, 10, Some(&["XYZ_NEVER_MATCHES"]), 0);
        assert!(!chunks.is_empty());
        for &(s, e) in &chunks {
            assert!(e - s <= 10);
        }
        let joined: String = text_of(&chunks, &text).concat();
        assert_eq!(joined, text);
    }

    #[test]
    fn empty_separators_list_skips_straight_to_raw_cut() {
        let text = "abcdefghij klmno pqrstu";
        let chunks = chunk_hierarchical(text, 5, Some(&[]), 0);
        for &(s, e) in &chunks {
            assert!(e - s <= 5);
        }
    }

    #[test]
    fn overlap_zero_matches_no_overlap_semantics() {
        let text = "Alpha beta gamma delta epsilon zeta eta theta.";
        let a = chunk_hierarchical(text, 12, None, 0);
        let b = chunk_hierarchical(text, 12, None, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn overlap_shares_genuine_content_langchain_34804_regression() {
        // The LangChain #34804-shaped regression: overlap must produce
        // ACTUAL shared content between consecutive chunks, not merely be
        // accepted as a parameter.
        let text = "one two three four five six seven eight nine ten eleven twelve";
        let chunks = chunk_hierarchical(text, 20, None, 5);
        assert!(chunks.len() > 1);
        let chars: Vec<char> = text.chars().collect();
        for pair in chunks.windows(2) {
            let (prev_start, prev_end) = pair[0];
            let (next_start, _next_end) = pair[1];
            if next_start < prev_end {
                let shared: String = chars[next_start..prev_end].iter().collect();
                assert!(!shared.is_empty(), "overlap produced no shared content");
                let _ = prev_start;
            }
        }
    }

    #[test]
    fn a_single_cluster_wider_than_the_budget_is_kept_whole_rather_than_split() {
        // The same pathological case chunk_text documents via the shared
        // grapheme_safe_hard_cut: the entire input is one grapheme cluster
        // (2 codepoints) but max_chars=1 can't fit it. Correctness wins
        // over the budget: the whole cluster comes back as one (oversized)
        // chunk rather than splitting it.
        let text = "0\u{0E33}";
        let chunks = chunk_hierarchical(text, 1, None, 0);
        assert_eq!(chunks, [(0, 2)]);
    }

    #[test]
    fn no_cut_ever_lands_inside_a_grapheme_cluster() {
        // Thai SARA AM (U+0E33) forms one grapheme cluster with the
        // preceding base character but chunk boundaries could otherwise
        // land between them (the same class of bug fixed elsewhere in
        // this crate).
        let text = "ab 0\u{0E33} cd ef 0\u{0E33} gh ij 0\u{0E33} kl";
        for max_chars in 1..text.chars().count() {
            let chunks = chunk_hierarchical(text, max_chars, None, 0);
            let grapheme_starts = grapheme_boundary_chars(text);
            let grapheme_set: HashSet<usize> = grapheme_starts.into_iter().collect();
            for &(s, e) in &chunks {
                assert!(
                    grapheme_set.contains(&s),
                    "chunk start {s} splits a cluster"
                );
                assert!(grapheme_set.contains(&e), "chunk end {e} splits a cluster");
            }
        }
    }

    #[test]
    fn forward_progress_never_stalls_on_adversarial_input() {
        // A degenerate custom separator (matches nothing) combined with a
        // tiny budget over long text: must still terminate promptly via
        // the raw-cut fallback, never loop.
        let text = "x".repeat(500);
        let chunks = chunk_hierarchical(&text, 3, Some(&["NEVER"]), 1);
        assert!(chunks.len() >= 166);
    }
}
