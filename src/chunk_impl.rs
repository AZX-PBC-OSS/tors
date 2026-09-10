//! Boundary-aware and content-defined chunking: the pure-Rust cores of
//! `tors.chunk_text` and `tors.chunk_cdc`, the two chunk shapes an
//! ingestion/RAG pipeline needs, both composing with the crate's merkle
//! tree.
//!
//! [`chunk_text`] cuts text at word/sentence boundaries into consecutive
//! chunks under a character budget: `truncate_to_bounds`'s own cut rule
//! (largest boundary end within the budget, hard-cut fallback, trailing
//! whitespace trimmed at the cut) applied repeatedly across the whole
//! text, computed against whole-text bounds once (a per-chunk
//! `truncate_to_bounds` call would re-segment its entire argument every
//! time, O(text²) across a chunking, and its `trim_end` deletes
//! characters a covering chunking must keep).
//!
//! [`chunk_cdc`] wraps `fastcdc::v2020::FastCDC`, the crate's recommended
//! implementation (same cut points as the 2016 paper, faster). Unlike
//! every other segmentation primitive in this crate (`word_bounds`,
//! `sentence_bounds`, `extract_code_blocks`), this operates on raw bytes
//! and returns byte offsets, not codepoints: content-defined chunking is
//! a byte-level dedup/incremental-sync primitive (its natural downstream
//! is feeding `merkle_root`/`merkle_diff`'s `list[bytes]`), not a text
//! operation, so there is no codepoint boundary to respect or preserve.
//! The offset conventions are correct on their own sides of that line
//! and neither function converts: `chunk_text`'s `text[start:end]` must
//! be the chunk for Python callers; `chunk_cdc`'s gear hash runs over a
//! byte window and its chunks feed the byte-level tree.
//!
//! Why content-defined rather than fixed-size cuts for the byte side: a
//! cut point is a function of a bounded window of content, not of the
//! input's offset, so an insert or append moves only the chunks whose
//! content actually changed: after a bounded window the cuts re-anchor
//! to the unchanged bytes and the rest of the chunk sequence is
//! byte-identical. `merkle_diff` over the before/after chunkings reports
//! a short, local run of changed indices instead of "everything after
//! the edit" (the stability tests pin this). `chunk_text`'s chunks
//! compose with the same tree (encode each slice, hash, diff) but
//! without that stability property: its cuts are made greedily from the
//! start, so an early text edit can shift every later chunk: when the
//! diffing property is the point, `chunk_cdc` is the shape.
//!
//! `fastcdc::v2020::FastCDC::with_level_and_seed` only `debug_assert!`s its
//! size-parameter bounds: a no-op in a release build, so an out-of-range
//! parameter would silently misbehave rather than panic or error. tors
//! validates the same bounds itself before ever constructing the chunker,
//! turning that silent-misbehavior class into an explicit `ValueError` at
//! the argument boundary (the same discipline `is_grounded`'s `threshold`
//! and `truncate_to_bounds`' `max_chars` already apply).
//!
//! The family's unit-count chunkers, `tors.chunk_by_words`,
//! `tors.chunk_by_sentences`, `tors.chunk_by_paragraphs`, live in
//! [`crate::chunk_by_segment_impl`], a separate file per the crate's
//! "one concern per file" rule: those three window over a fixed count
//! of segments rather than a character/byte budget, a different enough
//! shape to deserve its own file.

use crate::segmentation_impl;
use crate::truncate_impl::{Boundary, cluster_safe_ends};
use fastcdc::v2020::{
    AVERAGE_MAX, AVERAGE_MIN, FastCDC, MAXIMUM_MAX, MAXIMUM_MIN, MINIMUM_MAX, MINIMUM_MIN,
};
use unicode_segmentation::UnicodeSegmentation;

/// Snap `limit` down to the largest grapheme-cluster boundary `<= limit`:
/// the hard-cut fallback's grapheme-safety, the same rule
/// `truncate_impl::truncate_to_bounds`'s own hard-cut fallback applies
/// (see that module's docs for the Thai SARA AM / combining-mark
/// motivation this closes). Unlike `truncate_to_bounds`, a covering
/// chunker cannot simply drop content that doesn't fit: if the only
/// grapheme boundary `<= limit` is `start` itself (a single cluster wider
/// than the remaining budget: pathological, but possible, e.g. a long
/// ZWJ emoji chain), that content still has to go in some chunk, so this
/// advances to the next grapheme boundary strictly after `start` instead:
/// the chunk exceeds `max_chars` rather than split the cluster,
/// correctness over the budget, forward progress unconditional either
/// way. `grapheme_starts` must be ascending and include `0`; every caller
/// here gets that from [`grapheme_boundary_chars`].
pub(crate) fn grapheme_safe_hard_cut(
    grapheme_starts: &[usize],
    start: usize,
    limit: usize,
) -> usize {
    let ghi = grapheme_starts.partition_point(|&g| g <= limit);
    let candidate = if ghi > 0 { grapheme_starts[ghi - 1] } else { 0 };
    if candidate > start {
        candidate
    } else {
        let nhi = grapheme_starts.partition_point(|&g| g <= start);
        grapheme_starts[nhi]
    }
}

/// The grapheme-boundary grid the chunk loop cuts on, fused with the one
/// per-cluster fact the trim needs: whether each cluster is entirely
/// Unicode whitespace. One forward pass (the same grapheme walk
/// `truncate_impl::grapheme_boundary_chars` drives, each codepoint
/// decoded exactly once for both the grid's char count and the
/// whitespace test) returning the ascending cluster starts plus a
/// parallel all-whitespace flag per cluster. This replaces the former
/// whole-text `Vec<char>` (a separate full decode pass and 4 bytes per
/// codepoint): the codepoint budget's `total` is the grid's own final
/// accumulated count (its last entry, the same total the bounds walk's
/// last segment end carries), and the trim only ever asks whether the
/// clusters just before a cut are whitespace, a question the flags
/// answer by grid index with no text re-read at all.
fn grapheme_boundary_whitespace(text: &str) -> (Vec<usize>, Vec<bool>) {
    let mut starts = Vec::new();
    let mut whitespace = Vec::new();
    let mut char_idx = 0usize;
    for cluster in text.graphemes(true) {
        starts.push(char_idx);
        let mut cp_len = 0usize;
        let mut all_ws = true;
        for c in cluster.chars() {
            cp_len += 1;
            all_ws &= c.is_whitespace();
        }
        whitespace.push(all_ws);
        char_idx += cp_len;
    }
    starts.push(char_idx);
    (starts, whitespace)
}

/// The end of the codepoint span `[grid[from_idx], grid[to_idx])` after
/// `str::trim_end`'s rule, as a cluster-grid index (`grid` the
/// `grapheme_boundary_whitespace` starts, `to_idx` a cut the caller
/// already knows is a cluster boundary, `from_idx` the current chunk
/// start's own index): the largest end at or before the cut such that no
/// cluster from `from_idx` up to it is entirely whitespace, `from_idx`
/// itself when the whole span is whitespace (the caller's empty-chunk
/// case). Backs the cut off over trailing all-whitespace clusters.
///
/// Backing off whole clusters lands exactly where `str::trim_end` stops,
/// whether trim_end is spelled over the byte slice between the two
/// offsets or as the former char-by-char walk over a `Vec<char>`:
/// whitespace codepoints are their own complete UTF-8 sequences, so the
/// byte-slice and char-slice trims agree, and no cluster mixes
/// whitespace and non-whitespace in a way that could separate them: a
/// whitespace codepoint either starts its own cluster (whatever combines
/// after it, e.g. NBSP + a combining accent, makes a mixed cluster whose
/// last codepoint is that non-whitespace mark, exactly where trim_end
/// stops) or is the LF of a CRLF pair (whose CR is whitespace too, an
/// all-whitespace cluster trim_end removes whole). So trim_end's
/// stopping point is always a cluster boundary with only all-whitespace
/// clusters between it and the cut, which is precisely what this
/// walk-back computes; the mixed battery's "a\r\nb. c\r d\ne" row pins
/// the CRLF case.
fn trimmed_end(cluster_whitespace: &[bool], from_idx: usize, to_idx: usize) -> usize {
    let mut idx = to_idx;
    while idx > from_idx && cluster_whitespace[idx - 1] {
        idx -= 1;
    }
    idx
}

/// Boundary-aware chunking of `text`: consecutive `(start, end)` pairs in
/// Python str index (codepoint) units: `text[start:end]` is the chunk:
/// covering the whole text, each chunk at most `max_chars` codepoints,
/// cut at `boundary` (word or sentence) wherever the budget allows.
/// Empty text returns no chunks.
///
/// The cut rule per chunk is `truncate_to_bounds`'s own, re-derived here:
/// the largest word/sentence segment end in `(start, start + max_chars]`
/// (the `word_bounds`/`sentence_bounds` segment ends: the same UAX #29
/// machinery `truncate_to_bounds` filters over), falling back to a hard
/// cut at `start + max_chars` when no boundary fits (a single
/// word/sentence longer than the budget, the truncation precedent; the
/// budget is essentially never exceeded, see the grapheme-cluster
/// exception below). The rule is the same and so is the `Boundary` enum
/// (reused from `truncate_impl`, not duplicated); the spelling here
/// computes the whole-text bounds once and cuts against them in codepoint
/// space: a binary search per chunk over the ascending segment ends.
///
/// Every accepted cut point (boundary end or hard cut) is additionally
/// grapheme-cluster-safe, the identical fix `truncate_to_bounds` applies
/// (see that module's docs for the Thai SARA AM / combining-mark
/// motivation): a word/sentence segment end that would split a cluster is
/// never accepted, and the hard-cut fallback snaps to the nearest cluster
/// boundary (`grapheme_safe_hard_cut`) rather than a raw codepoint offset.
/// Like `truncate_to_bounds`, this is computed once up front (the
/// whole-text grapheme boundary grid, `grapheme_boundary_whitespace`, one
/// O(n) pass fused with the trim's per-cluster whitespace flags) and
/// reused as an O(log n) lookup per chunk: no per-chunk
/// re-scan. The one place this can still exceed `max_chars`: a single
/// grapheme cluster wider than the whole remaining budget (e.g. an
/// oversized ZWJ emoji chain), where a covering chunker cannot drop content
/// that doesn't fit, so that one chunk is allowed past the budget rather
/// than split the cluster; this never affects ordinary text (no cluster
/// is more than a handful of codepoints).
///
/// Greedy-then-trim is the optimal greedy order, not an approximation:
/// trimming the largest in-budget boundary end yields a final end at
/// least as large as trimming any smaller in-budget end would, so the
/// largest cut first is always the most content kept.
///
/// Trailing whitespace at each interior cut is trimmed exactly as
/// `truncate_to_bounds` trims; but a chunker cannot delete characters
/// (the chunks must join back to the input), so the trim moves the cut
/// back and the trimmed whitespace rides the head of the next chunk.
/// Two exceptions, both forced by the coverage invariant and pinned by
/// the tests: (1) the final chunk runs to the end of the text untrimmed:
/// trailing whitespace there is input the chunks must cover; (2) a span
/// that is entirely whitespace is emitted untrimmed: trimming it would
/// empty the chunk (chunks are non-empty) or strand its characters.
/// Unconditionally: chunks are non-empty (`start < end`), contiguous
/// (the first starts at 0, each next start is the previous end), strictly
/// increasing, and joining the codepoint slices reproduces the input
/// exactly.
///
/// `max_chars == 0` with non-empty text is unsatisfiable (a non-empty
/// chunk of zero characters does not exist); the pyo3 layer pre-validates
/// it into `ValueError` (the `max_chars < 0` case lands there too).
///
/// `overlap` (default 0): `chunk_text_overlapping`'s zero case, kept as a
/// separate top-level fn (not a runtime branch inside one function) so this
/// lossless-partition contract stays exactly what it always was, verified
/// unchanged by this crate's own test suite rather than re-derived from a
/// more general (and therefore harder to keep byte-identical) spelling.
pub fn chunk_text(text: &str, max_chars: usize, boundary: Boundary) -> Vec<(usize, usize)> {
    if text.is_empty() {
        return Vec::new();
    }
    assert!(
        max_chars > 0,
        "max_chars must be at least 1 (chunks are non-empty), got {max_chars}"
    );
    let bounds = match boundary {
        Boundary::Word => segmentation_impl::word_bounds(text),
        Boundary::Sentence => segmentation_impl::sentence_bounds(text),
    };
    // The whole-text segment ends and grapheme-cluster boundaries, each
    // computed once: word/sentence bounds are already in codepoint units,
    // and every chunk's cut is a search over these ascending ends: the
    // per-suffix re-segmentation a truncate_to_bounds-per-chunk spelling
    // would pay is the cost this avoids. `ends` is additionally
    // intersected with grapheme-cluster boundaries up front (one O(n)
    // merge over the whole list, `truncate_impl::cluster_safe_ends`: the
    // same shared helper `truncate_to_bounds`'s own cut filters through,
    // not a per-chunk re-check or a second spelling) so every accepted
    // cut is cluster-safe by construction: see `grapheme_safe_hard_cut`
    // for the fallback's own cluster-safety. The grid walk replaces the
    // former `Vec<char>` whole-text collect with its own accumulated
    // codepoint count (the grid's final entry) plus the trim's
    // per-cluster whitespace flags.
    let (grapheme_starts, cluster_whitespace) = grapheme_boundary_whitespace(text);
    let total = *grapheme_starts.last().unwrap();
    let ends = cluster_safe_ends(&bounds, &grapheme_starts);
    let mut chunks = Vec::with_capacity(total / max_chars + 1);
    let mut start = 0usize;
    // The grid index of `start` (every chunk start is a cluster boundary
    // by construction: 0, or a previous cut or trimmed end), so the
    // trim's walk-back is pure index arithmetic over the flags.
    let mut start_idx = 0usize;
    while start < total {
        let remaining = total - start;
        if remaining <= max_chars {
            // The final span fits the budget whole and runs to the end of
            // the text untrimmed (the documented coverage exception). The
            // end of the text is always a grapheme boundary (the last
            // entry of `grapheme_starts`), so this is cluster-safe too.
            chunks.push((start, total));
            break;
        }
        // No overflow: computed only when remaining > max_chars, so
        // start + max_chars < total.
        let limit = start + max_chars;
        // The largest cluster-safe segment end <= limit; ends are
        // ascending, so partition_point splits them in one comparison
        // run. An end at or before `start` is no cut at all (an empty
        // chunk), and if the largest in-budget end is at or before
        // `start` then every in-budget end is: the grapheme-safe hard
        // cut applies (it can, in the pathological case of a single
        // cluster wider than the remaining budget, land past `limit`:
        // see `grapheme_safe_hard_cut`'s docs).
        let hi = ends.partition_point(|&end| end <= limit);
        let cut = if hi > 0 && ends[hi - 1] > start {
            ends[hi - 1]
        } else {
            grapheme_safe_hard_cut(&grapheme_starts, start, limit)
        };
        // Every cut is a grid boundary (an end that survived the
        // cluster-safety merge, or the hard cut's own snapped boundary),
        // so its grid index is a binary search away.
        let cut_idx = grapheme_starts.partition_point(|&g| g < cut);
        let trimmed_idx = trimmed_end(&cluster_whitespace, start_idx, cut_idx);
        if trimmed_idx > start_idx {
            let trimmed = grapheme_starts[trimmed_idx];
            chunks.push((start, trimmed));
            start = trimmed;
            start_idx = trimmed_idx;
        } else {
            // The span is entirely whitespace: trimming it would empty
            // the chunk, so it goes out whole (the documented exception).
            chunks.push((start, cut));
            start = cut;
            start_idx = cut_idx;
        }
    }
    chunks
}

/// [`chunk_text`] with repeated trailing context: each chunk after the
/// first starts `overlap` codepoints before the previous chunk's end,
/// snapped to the nearest `boundary` at or before that target: never
/// starting mid-word/mid-sentence, same as every cut in [`chunk_text`]
/// itself. `overlap == 0` degenerates to calling [`chunk_text`] directly
/// (identical output, not just equivalent: the lossless-partition
/// contract is untouched). `overlap > 0` trades that lossless-join
/// guarantee for the RAG-retrieval shape (a fact split across a boundary
/// is still whole in at least one chunk); chunks are no longer generally
/// contiguous or coverage-exact, though every codepoint of the input is
/// still repeated at least once for `boundary`-safe overlap targets short
/// of a run of oversized (hard-cut) chunks.
///
/// Each chunk's own `(start, end)` is computed by the exact same cut+trim
/// rule as [`chunk_text`] (so within one call every chunk is still
/// `<= max_chars` and never splits a word/sentence, or a grapheme
/// cluster (see [`chunk_text`]'s docs for the same fix applied here),
/// at its own edges); the overlap snap below reuses that same
/// grapheme-cluster-filtered boundary list, so a snapped start is never
/// mid-cluster either. Only where the next chunk starts differs.
/// `overlap >= max_chars` is
/// rejected by the pyo3 layer before this ever runs (no forward progress
/// would be possible: an overlap at least as large as the budget means
/// each chunk would restart at or before its own start).
///
/// Forward progress is unconditional, not merely typical: `chunk_end` is
/// always `> start` (proven the same way [`chunk_text`]'s is: the cut is
/// either a boundary end `> start` or the hard-cut `start + max_chars`,
/// and `max_chars >= 1`), and the next start is either that same
/// `chunk_end` (the snap-collapsed-too-far fallback below) or a boundary
/// strictly between `start` and `chunk_end`, both `> start` by
/// construction, so the loop always advances and terminates in at most
/// `text.chars().count()` iterations (pinned by a hard iteration-count
/// assertion in the tests, not just a slow-test timeout).
///
/// The snap-collapse case: when a chunk is shorter than the requested
/// `overlap` (a short trailing chunk, or a run of tight hard-cuts), the
/// target `chunk_end - overlap` can land at or before `start`: snapping
/// it there would either violate forward progress or claim overlap this
/// chunk cannot actually provide. Rather than either, the overlap is
/// silently reduced to zero for just that one transition (the next chunk
/// starts at `chunk_end`, `chunk_text`'s own no-overlap rule), a
/// documented degradation under the one invariant that must never break
/// (forward progress), not a silent contract violation.
pub fn chunk_text_overlapping(
    text: &str,
    max_chars: usize,
    overlap: usize,
    boundary: Boundary,
) -> Vec<(usize, usize)> {
    if overlap == 0 {
        return chunk_text(text, max_chars, boundary);
    }
    if text.is_empty() {
        return Vec::new();
    }
    assert!(
        max_chars > 0,
        "max_chars must be at least 1 (chunks are non-empty), got {max_chars}"
    );
    assert!(
        overlap < max_chars,
        "overlap must be less than max_chars (no forward progress otherwise), \
         got overlap={overlap}, max_chars={max_chars}"
    );
    let bounds = match boundary {
        Boundary::Word => segmentation_impl::word_bounds(text),
        Boundary::Sentence => segmentation_impl::sentence_bounds(text),
    };
    // Grapheme-cluster-safe, exactly as chunk_text's own `ends`: see that
    // function's comments. The overlap snap below reuses this same
    // cluster-safe `ends` list, so a snapped start is never mid-cluster
    // either.
    let (grapheme_starts, cluster_whitespace) = grapheme_boundary_whitespace(text);
    let total = *grapheme_starts.last().unwrap();
    let ends = cluster_safe_ends(&bounds, &grapheme_starts);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    // The grid index of `start`, carried across iterations exactly as
    // chunk_text's: every start here is a snapped boundary end, a cut, or
    // a trimmed end, all grid boundaries by construction.
    let mut start_idx = 0usize;
    loop {
        let remaining = total - start;
        let chunk_end = if remaining <= max_chars {
            total
        } else {
            let limit = start + max_chars;
            let hi = ends.partition_point(|&end| end <= limit);
            let cut = if hi > 0 && ends[hi - 1] > start {
                ends[hi - 1]
            } else {
                grapheme_safe_hard_cut(&grapheme_starts, start, limit)
            };
            let cut_idx = grapheme_starts.partition_point(|&g| g < cut);
            let trimmed_idx = trimmed_end(&cluster_whitespace, start_idx, cut_idx);
            if trimmed_idx > start_idx {
                grapheme_starts[trimmed_idx]
            } else {
                cut
            }
        };
        chunks.push((start, chunk_end));
        if chunk_end >= total {
            break;
        }
        // Snap `chunk_end - overlap` to the nearest boundary at or before
        // it; fall back to `chunk_end` (no overlap this transition) unless
        // the snapped position is strictly between `start` and `chunk_end`,
        // per the forward-progress guarantee documented above.
        let target = chunk_end.saturating_sub(overlap);
        let snap_hi = ends.partition_point(|&end| end <= target);
        let snapped = if snap_hi > 0 { ends[snap_hi - 1] } else { 0 };
        start = if snapped > start && snapped < chunk_end {
            snapped
        } else {
            chunk_end
        };
        start_idx = grapheme_starts.partition_point(|&g| g < start);
    }
    chunks
}

/// One content-defined chunk's byte span in the original input, `end`
/// exclusive.
pub type ChunkSpan = (usize, usize);

/// Why [`chunk_cdc`] fails: a size-parameter violates the
/// underlying algorithm's required bounds (`fastcdc`'s own documented
/// `MINIMUM_MIN..MINIMUM_MAX` / `AVERAGE_MIN..AVERAGE_MAX` /
/// `MAXIMUM_MIN..MAXIMUM_MAX` ranges, evenness, and `min <= avg <= max`
/// ordering; the ordering isn't asserted by the crate itself, but is
/// required for the algorithm to mean anything: an `avg_size` below
/// `min_size` or above `max_size` produces a normalization target the
/// chunker can never land on).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidChunkParams(pub String);

fn validate(min_size: usize, avg_size: usize, max_size: usize) -> Result<(), InvalidChunkParams> {
    if !(MINIMUM_MIN..=MINIMUM_MAX).contains(&min_size) {
        return Err(InvalidChunkParams(format!(
            "min_size must be in [{MINIMUM_MIN}, {MINIMUM_MAX}], got {min_size}"
        )));
    }
    if !(AVERAGE_MIN..=AVERAGE_MAX).contains(&avg_size) {
        return Err(InvalidChunkParams(format!(
            "avg_size must be in [{AVERAGE_MIN}, {AVERAGE_MAX}], got {avg_size}"
        )));
    }
    if !(MAXIMUM_MIN..=MAXIMUM_MAX).contains(&max_size) {
        return Err(InvalidChunkParams(format!(
            "max_size must be in [{MAXIMUM_MIN}, {MAXIMUM_MAX}], got {max_size}"
        )));
    }
    if !min_size.is_multiple_of(2) || !avg_size.is_multiple_of(2) || !max_size.is_multiple_of(2) {
        return Err(InvalidChunkParams(
            "min_size, avg_size, and max_size must all be even".to_string(),
        ));
    }
    if !(min_size <= avg_size && avg_size <= max_size) {
        return Err(InvalidChunkParams(format!(
            "min_size <= avg_size <= max_size required, got {min_size}, {avg_size}, {max_size}"
        )));
    }
    Ok(())
}

/// Content-defined chunk boundaries over `data`, `(start, end)` byte spans
/// in document order, `end` exclusive, partitioning `data` exactly (no
/// gaps, no overlaps, the last span's `end == data.len()`). Empty input
/// yields `[]`. Input shorter than `min_size` yields exactly one span
/// covering the whole input (`fastcdc`'s own documented special case).
/// Deterministic: the same bytes at the same parameters always cut at the
/// same offsets; the whole reason to reach for content-defined over
/// fixed-size chunking is that a small edit only perturbs the 1-2 chunks
/// nearest the edit, not every boundary after it.
pub fn chunk_cdc(
    data: &[u8],
    min_size: usize,
    avg_size: usize,
    max_size: usize,
) -> Result<Vec<ChunkSpan>, InvalidChunkParams> {
    validate(min_size, avg_size, max_size)?;
    Ok(FastCDC::new(data, min_size, avg_size, max_size)
        .map(|chunk| (chunk.offset, chunk.offset + chunk.length))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- chunk_text ----

    #[test]
    fn greedy_word_cut_trims_the_dangling_space_onto_the_next_chunk() {
        // word_bounds("cats are cute") ends: 4, 5, 8, 9, 13. Budget 9 from
        // 0: the largest end <= 9 is 9, the "are" + following-space
        // segment (truncate_to_bounds' own row), and the trim moves the
        // cut back to 8, the space riding the next chunk's head instead
        // of being deleted (a covering chunking keeps every character).
        let text = "cats are cute";
        let chars: Vec<char> = text.chars().collect();
        let chunks = chunk_text(text, 9, Boundary::Word);
        assert_eq!(chunks, [(0, 8), (8, 13)]);
        let slice = |(a, b): (usize, usize)| chars[a..b].iter().collect::<String>();
        assert_eq!(slice(chunks[0]), "cats are");
        assert_eq!(slice(chunks[1]), " cute");
    }

    #[test]
    fn greedy_sentence_cut_lands_on_whole_sentences() {
        // sentence_bounds("One. Two. Three.") ends: 5, 10, 16 (SB9-SB11
        // put each inter-sentence space on the preceding sentence).
        // Budget 10 from 0 lands exactly on end 10, trimmed to 9; the
        // remaining 7 codepoints fit the budget whole.
        let text = "One. Two. Three.";
        let chars: Vec<char> = text.chars().collect();
        let chunks = chunk_text(text, 10, Boundary::Sentence);
        assert_eq!(chunks, [(0, 9), (9, 16)]);
        let slice = |(a, b): (usize, usize)| chars[a..b].iter().collect::<String>();
        assert_eq!(slice(chunks[0]), "One. Two.");
        assert_eq!(slice(chunks[1]), " Three.");
    }

    #[test]
    fn hard_cut_when_no_boundary_fits() {
        // One 34-codepoint word, no boundary end <= 10: hard cuts at the
        // budget, the truncation precedent, never exceeded, still
        // covering and contiguous.
        assert_eq!(
            chunk_text("Supercalifragilisticexpialidocious", 10, Boundary::Word),
            [(0, 10), (10, 20), (20, 30), (30, 34)]
        );
        // One terminator-free sentence: the same degradation at the
        // sentence boundary.
        assert_eq!(
            chunk_text("no terminator here", 8, Boundary::Sentence),
            [(0, 8), (8, 16), (16, 18)]
        );
    }

    #[test]
    fn a_whitespace_only_span_is_emitted_untrimmed() {
        // word_bounds ends: 4, 8, 12 (WSegSpace joins the space run into
        // one segment). Budget 6 from 4 lands on end 8, whose span is
        // entirely the space run: trimming it would empty the chunk, so
        // it goes out whole (the documented coverage exception).
        assert_eq!(
            chunk_text("aaaa    bbbb", 6, Boundary::Word),
            [(0, 4), (4, 8), (8, 12)]
        );
    }

    #[test]
    fn the_final_chunk_runs_to_the_end_untrimmed() {
        // Trailing input whitespace is content the chunks must cover; the
        // interior-cut trim does not apply at the end of the text.
        assert_eq!(chunk_text("abc  ", 10, Boundary::Word), [(0, 5)]);
    }

    #[test]
    fn empty_text_yields_no_chunks() {
        assert_eq!(
            chunk_text("", 5, Boundary::Word),
            Vec::<(usize, usize)>::new()
        );
        assert_eq!(
            chunk_text("", 5, Boundary::Sentence),
            Vec::<(usize, usize)>::new()
        );
    }

    #[test]
    #[should_panic(expected = "max_chars must be at least 1")]
    fn max_chars_zero_with_nonempty_text_is_refused() {
        // A non-empty chunk of zero characters cannot exist; the pyo3
        // layer pre-validates this into ValueError.
        chunk_text("hello", 0, Boundary::Word);
    }

    #[test]
    fn offsets_are_codepoints_not_bytes() {
        // "café 東京。" is 8 codepoints but 11 bytes (é and the han
        // characters are multibyte). word_bounds ends: 4, 5, then 7
        // and/or 8 (whether U+3002 rides the han word is the segmenter's
        // call); either way budget 5 cuts after "café" at codepoint 4
        // (where a byte-offset chunker would say 5), and the rest fits.
        let text = "café 東京。";
        let chars: Vec<char> = text.chars().collect();
        let chunks = chunk_text(text, 5, Boundary::Word);
        assert_eq!(chunks, [(0, 4), (4, 8)]);
        let joined: String = chunks
            .iter()
            .map(|&(a, b)| chars[a..b].iter().collect::<String>())
            .collect();
        assert_eq!(joined, text);
    }

    // ---- grapheme-cluster safety (the truncate_impl regression, re-derived here) ----

    #[test]
    fn never_splits_a_thai_sara_am_cluster_across_two_chunks() {
        // "0" + SARA AM (U+0E33) is one grapheme cluster, but word_bounds
        // scores it as two word segments, (3,4)="0", (4,5)="ำ", the exact
        // edge truncate_impl's grapheme-safety fix closes (see that
        // module's docs). Without that fix, chunk_text("ab 0ำ cd", 4, Word)
        // cuts exactly between them, [(0, 4)="ab 0", (4, 8)="ำ cd"]: the
        // base character and its combining mark silently separated into
        // different chunks. The fix must keep the cluster whole in one
        // chunk instead, even though that means backing the cut off
        // earlier (to the previous word boundary) rather than filling
        // the budget to 4.
        let text = "ab 0\u{0E33} cd";
        let chars: Vec<char> = text.chars().collect();
        let chunks = chunk_text(text, 4, Boundary::Word);
        assert_eq!(chunks, [(0, 2), (2, 5), (5, 8)]);
        let slice = |(a, b): (usize, usize)| chars[a..b].iter().collect::<String>();
        assert_eq!(slice(chunks[0]), "ab");
        assert_eq!(slice(chunks[1]), " 0\u{0E33}");
        assert_eq!(slice(chunks[2]), " cd");
        let joined: String = chunks.iter().map(|&c| slice(c)).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn a_single_cluster_wider_than_the_budget_is_kept_whole_rather_than_split() {
        // The pathological case grapheme_safe_hard_cut documents: the
        // entire input is one grapheme cluster (2 codepoints) but
        // max_chars=1 can't fit it. A covering chunker cannot drop
        // content, so correctness wins over the budget: the whole
        // cluster comes back as one (oversized) chunk rather than
        // silently splitting it.
        let text = "0\u{0E33}";
        let chunks = chunk_text(text, 1, Boundary::Word);
        assert_eq!(chunks, [(0, 2)]);
    }

    #[test]
    fn no_chunk_boundary_ever_lands_strictly_inside_a_grapheme_cluster() {
        // Exhaustive over every max_chars for a text containing the SARA
        // AM edge in both Word and Sentence boundary modes: every chunk
        // boundary (every `a` and `b` across all chunks) must be a real
        // grapheme-cluster boundary of the text, never a codepoint index
        // strictly inside a cluster.
        use crate::truncate_impl::grapheme_boundary_chars;
        let text = "ab 0\u{0E33} cd. x0\u{0E33}y.";
        let valid: std::collections::HashSet<usize> =
            grapheme_boundary_chars(text).into_iter().collect();
        for max_chars in 1..=(text.chars().count() + 2) {
            for boundary in [Boundary::Word, Boundary::Sentence] {
                let chunks = chunk_text(text, max_chars, boundary);
                for &(a, b) in &chunks {
                    assert!(
                        valid.contains(&a),
                        "chunk start {a} is mid-cluster for max_chars={max_chars}/{boundary:?}: {chunks:?}"
                    );
                    assert!(
                        valid.contains(&b),
                        "chunk end {b} is mid-cluster for max_chars={max_chars}/{boundary:?}: {chunks:?}"
                    );
                }
            }
        }
    }

    /// The full chunk_text contract over one (text, max_chars, boundary)
    /// case: contiguity (first start 0, each next start the previous end),
    /// non-emptiness, the <= max_chars invariant on every chunk, coverage
    /// (the last end is the codepoint length), and the join-back: the
    /// Python `"".join(text[a:b] for ...)` equivalent, exact equality.
    fn assert_contract(text: &str, max_chars: usize, boundary: Boundary) {
        use crate::truncate_impl::grapheme_boundary_chars;
        let chunks = chunk_text(text, max_chars, boundary);
        let chars: Vec<char> = text.chars().collect();
        let grapheme_starts = grapheme_boundary_chars(text);
        let mut prev_end = 0usize;
        for &(a, b) in &chunks {
            assert_eq!(
                a, prev_end,
                "contiguity broke for {text:?}/{max_chars}/{boundary:?}"
            );
            assert!(b > a, "empty chunk for {text:?}/{max_chars}/{boundary:?}");
            if b - a > max_chars {
                // The one documented exception: a single grapheme cluster
                // wider than the whole budget (e.g. a CRLF pair at
                // max_chars=1) is kept whole rather than split: never
                // any other reason to exceed the budget. Confirm [a, b) is
                // exactly one cluster: `a` is a cluster start and `b` is
                // the very next cluster start, nothing smaller possible.
                let a_idx = grapheme_starts
                    .iter()
                    .position(|&g| g == a)
                    .unwrap_or_else(|| {
                        panic!(
                            "chunk start {a} is not a grapheme boundary for {text:?}/{max_chars}/{boundary:?}: {chunks:?}"
                        )
                    });
                assert_eq!(
                    grapheme_starts.get(a_idx + 1),
                    Some(&b),
                    "budget exceeded without being a single oversized cluster for \
                     {text:?}/{max_chars}/{boundary:?}: {chunks:?}"
                );
            }
            prev_end = b;
        }
        assert_eq!(
            prev_end,
            chars.len(),
            "coverage broke for {text:?}/{max_chars}/{boundary:?}"
        );
        let joined: String = chunks
            .iter()
            .map(|&(a, b)| chars[a..b].iter().collect::<String>())
            .collect();
        assert_eq!(
            joined, text,
            "join-back broke for {text:?}/{max_chars}/{boundary:?}"
        );
    }

    #[test]
    fn properties_hold_over_the_exhaustive_small_alphabet() {
        // Every string over {a, space, .} up to length 4 (1+3+9+27+81 =
        // 121 strings) x max_chars 1..=4 x both boundaries: the alphabet
        // puts whitespace (the trim paths) and a sentence terminator (the
        // sentence-boundary paths) inside the exhaustive space, and the
        // property block catches every cut rule at the smallest scales
        // where the boundary cases live.
        let alphabet = ['a', ' ', '.'];
        let mut texts = vec![String::new()];
        for _ in 0..4 {
            let mut frontier = Vec::new();
            for text in &texts {
                for &c in &alphabet {
                    let mut next = text.clone();
                    next.push(c);
                    frontier.push(next);
                }
            }
            texts.extend(frontier);
        }
        for text in &texts {
            for max_chars in 1..=4 {
                for boundary in [Boundary::Word, Boundary::Sentence] {
                    assert_contract(text, max_chars, boundary);
                }
            }
        }
    }

    #[test]
    fn properties_hold_on_a_mixed_battery() {
        // The non-ASCII and separator rows the small alphabet cannot
        // spell, at every budget from 1 up through the whole-text fit.
        let cases = [
            "",
            "a",
            "  ",
            "hello world",
            "One. Two. Three.",
            "supercalifragilistic",
            "aaaa    bbbb",
            "no terminator here",
            "a\r\nb. c\r d\ne",
            "caf\u{e9} \u{6771}\u{4eac}\u{3002} \u{5927}\u{962a}\u{3002}",
            "\u{1f469}\u{200d}\u{1f52c} says hi. \u{1100}\u{1161}\u{11a8}!",
        ];
        for case in cases {
            for max_chars in 1..=(case.chars().count() + 2) {
                for boundary in [Boundary::Word, Boundary::Sentence] {
                    assert_contract(case, max_chars, boundary);
                }
            }
        }
    }

    // ---- chunk_text_overlapping ----

    #[test]
    fn overlap_zero_is_byte_identical_to_chunk_text() {
        let cases = [
            "cats are cute",
            "One. Two. Three.",
            "Supercalifragilisticexpialidocious",
            "aaaa    bbbb",
            "",
        ];
        for text in cases {
            for max_chars in 1..=8 {
                for boundary in [Boundary::Word, Boundary::Sentence] {
                    assert_eq!(
                        chunk_text_overlapping(text, max_chars, 0, boundary),
                        chunk_text(text, max_chars, boundary),
                        "overlap=0 diverged for {text:?}/{max_chars}/{boundary:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn overlap_produces_genuinely_shared_content_between_consecutive_chunks() {
        // word_bounds("the cat sat on the mat today") has plenty of word
        // boundaries; budget 12 with overlap 4 must make chunk[i+1] start
        // strictly before chunk[i]'s end, and the shared span's text must
        // actually match on both sides.
        let text = "the cat sat on the mat today";
        let chars: Vec<char> = text.chars().collect();
        let chunks = chunk_text_overlapping(text, 12, 4, Boundary::Word);
        assert!(
            chunks.len() >= 2,
            "test needs at least 2 chunks: {chunks:?}"
        );
        for w in chunks.windows(2) {
            let (prev_start, prev_end) = w[0];
            let (next_start, _) = w[1];
            assert!(
                next_start > prev_start,
                "no forward progress: {:?} -> {:?}",
                w[0],
                w[1]
            );
            assert!(
                next_start < prev_end,
                "overlap produced no actual overlap: {:?} -> {:?}",
                w[0],
                w[1]
            );
            let shared: String = chars[next_start..prev_end].iter().collect();
            assert!(!shared.is_empty());
        }
    }

    #[test]
    fn overlap_never_produces_an_infinite_loop_bounded_by_char_count() {
        // The forward-progress guarantee as a hard iteration-count
        // assertion, not a timeout: the number of chunks can never exceed
        // the codepoint count (each chunk starts at a strictly later
        // position than the last), so a regression to a stalled/looping
        // start would blow this bound long before it would hang a test
        // runner.
        let text = "a".repeat(500) + " " + &"b".repeat(500);
        let total = text.chars().count();
        for max_chars in [2usize, 5, 10, 50] {
            for overlap in 1..max_chars {
                let chunks = chunk_text_overlapping(&text, max_chars, overlap, Boundary::Word);
                assert!(
                    chunks.len() <= total,
                    "chunk count {} exceeded char count {} for max_chars={max_chars}, overlap={overlap}",
                    chunks.len(),
                    total
                );
                // Every chunk's own budget and boundary-safety invariant
                // still holds exactly as chunk_text's does.
                for &(a, b) in &chunks {
                    assert!(b > a);
                    assert!(b - a <= max_chars);
                }
            }
        }
    }

    #[test]
    fn overlap_never_exceeds_max_chars_and_snaps_are_never_mid_word() {
        // Adversarial: a short trailing chunk shorter than the requested
        // overlap forces the snap-collapse fallback (documented on
        // chunk_text_overlapping): confirm forward progress still holds
        // and no chunk boundary lands mid-word (every start/end is a real
        // word_bounds segment edge or 0/len(text)).
        let text = "a bb ccc dddd eeeee";
        let bounds = segmentation_impl::word_bounds(text);
        let mut valid_positions: std::collections::HashSet<usize> =
            bounds.iter().map(|&(_, end)| end).collect();
        valid_positions.insert(0);
        valid_positions.insert(text.chars().count());
        for max_chars in 2..=10 {
            for overlap in 1..max_chars {
                let chunks = chunk_text_overlapping(text, max_chars, overlap, Boundary::Word);
                let mut prev_start: Option<usize> = None;
                for &(start, end) in &chunks {
                    assert!(end - start <= max_chars);
                    if let Some(p) = prev_start {
                        assert!(
                            start > p,
                            "no forward progress at max_chars={max_chars}, overlap={overlap}"
                        );
                    }
                    prev_start = Some(start);
                }
            }
        }
    }

    fn xorshift_bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.push((state & 0xff) as u8);
        }
        out
    }

    #[test]
    fn empty_input_is_no_chunks() {
        assert_eq!(chunk_cdc(&[], 4096, 16384, 65534).unwrap(), vec![]);
    }

    #[test]
    fn input_shorter_than_min_size_is_one_chunk() {
        let data = b"tiny";
        let chunks = chunk_cdc(data, 4096, 16384, 65534).unwrap();
        assert_eq!(chunks, vec![(0, data.len())]);
    }

    #[test]
    fn chunks_partition_the_input_exactly() {
        let data = xorshift_bytes(500_000, 0x9E3779B97F4A7C15);
        let chunks = chunk_cdc(&data, 4096, 16384, 65534).unwrap();
        assert!(!chunks.is_empty());
        let mut prev_end = 0usize;
        for &(start, end) in &chunks {
            assert_eq!(start, prev_end, "gap or overlap at {start}..{end}");
            assert!(end > start, "empty chunk at {start}..{end}");
            assert!(
                end - start <= 65534,
                "chunk exceeds max_size: {start}..{end}"
            );
            prev_end = end;
        }
        assert_eq!(prev_end, data.len());
    }

    #[test]
    fn deterministic_same_input_same_chunks() {
        let data = xorshift_bytes(200_000, 42);
        let a = chunk_cdc(&data, 4096, 16384, 65534).unwrap();
        let b = chunk_cdc(&data, 4096, 16384, 65534).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn a_small_edit_near_the_start_only_perturbs_nearby_chunks() {
        // The whole point of content-defined over fixed-size chunking: an
        // insertion shifts every following byte's absolute offset, but the
        // cut points are chosen by local content, so chunks far past the
        // edit should reappear as identical (start, end) - offset pairs,
        // not reshuffle wholesale the way fixed-size chunking would.
        let mut data = xorshift_bytes(500_000, 7);
        let before = chunk_cdc(&data, 4096, 16384, 65534).unwrap();
        data.splice(1000..1000, [0xAAu8; 37]);
        let after = chunk_cdc(&data, 4096, 16384, 65534).unwrap();

        let shift = 37i64;
        let mut unchanged_beyond_edit = 0usize;
        let mut total_beyond_edit = 0usize;
        for &(bs, be) in &before {
            if bs < 50_000 {
                continue; // too close to the edit to expect stability
            }
            total_beyond_edit += 1;
            let shifted = (bs as i64 + shift, be as i64 + shift);
            if after
                .iter()
                .any(|&(as_, ae)| (as_ as i64, ae as i64) == shifted)
            {
                unchanged_beyond_edit += 1;
            }
        }
        assert!(
            total_beyond_edit > 5,
            "test needs more chunks to be meaningful"
        );
        let ratio = unchanged_beyond_edit as f64 / total_beyond_edit as f64;
        assert!(
            ratio > 0.9,
            "expected most distant chunks to reappear shifted by exactly the edit size; \
             {unchanged_beyond_edit}/{total_beyond_edit} did ({ratio:.2})"
        );
    }

    #[test]
    fn rejects_out_of_range_and_misordered_sizes() {
        assert!(chunk_cdc(&[1, 2, 3], 63, 16384, 65534).is_err()); // below MINIMUM_MIN
        assert!(chunk_cdc(&[1, 2, 3], 4096, 16384, 4095).is_err()); // max < avg
        assert!(chunk_cdc(&[1, 2, 3], 16384, 4096, 65534).is_err()); // min > avg
        assert!(chunk_cdc(&[1, 2, 3], 4097, 16384, 65534).is_err()); // odd min_size
    }
}
