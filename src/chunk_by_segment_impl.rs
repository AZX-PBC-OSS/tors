//! Unit-count chunking: [`chunk_by_words`], [`chunk_by_sentences`],
//! [`chunk_by_paragraphs`], and [`chunk_by_lines`], the
//! `tors.chunk_text`/`tors.chunk_cdc` family's third shape: instead of a
//! character budget ([`crate::chunk_impl::chunk_text`]) or byte-content
//! anchoring (`chunk_cdc`), each chunk spans a fixed COUNT of consecutive
//! segments from one of the crate's segmenters: real word tokens
//! ([`crate::segmentation_impl::word_bounds`], filtered), UAX #29
//! sentences ([`crate::segmentation_impl::sentence_bounds`]),
//! newline-run-delimited paragraphs ([`paragraph_bounds`], a heuristic: no
//! UAX exists for paragraphs), or newline-terminated lines ([`line_bounds`],
//! the same CR/CRLF-folding convention). Split from `chunk_impl.rs` per the
//! crate's own "one concern per file" rule: [`chunk_text`] and `chunk_cdc`
//! are CHARACTER/BYTE-budget chunkers, these four are UNIT-COUNT chunkers, a
//! different windowing shape sharing only the `Boundary`-safety discipline,
//! not the cut logic.
//!
//! [`chunk_by_segments`] is the one windowing walk behind all four unit
//! chunkers: "N segments per chunk, Y segments of overlap" over whatever
//! `(start, end)` segment list the caller already produced: the DRY point
//! this file exists to keep in one place rather than copied per unit.
//!
//! [`crate::chunk_impl`]

use crate::segmentation_impl;
use crate::truncate_impl::{GraphemeIndex, char_count};

/// Merge adjacent CONTIGUOUS segments (`bounds[i].1 == bounds[i + 1].0`:
/// the `word_bounds`/`sentence_bounds` covering-partition contract) whose
/// shared boundary is NOT a grapheme-cluster boundary, the same
/// SARA-AM-shaped edge `crate::chunk_impl`'s hard-cut fallback guards
/// against: UAX #29 word boundaries occasionally score a combining
/// sequence (e.g. Thai SARA AM, U+0E33) as its own word-segment even
/// though `unicode-segmentation`'s grapheme rules join it to the
/// preceding base character into one cluster. [`chunk_by_segments`]
/// windows over segment EDGES directly, so a chunk boundary landing
/// exactly on such a split would silently divide the cluster between two
/// returned chunks; merging the two segments before windowing removes the
/// cut point rather than special-casing it per chunk. One forward pass,
/// O(n), the membership question answered O(1) by the shared
/// [`GraphemeIndex`] bitmap (the former `HashSet<usize>` built from the
/// whole boundary list cost ~12.6M hashed inserts — ~1.2 s — on a 12 MiB
/// document, the same #22 pathology `chunk_hierarchical` fixed).
fn merge_mid_cluster_boundaries(
    bounds: Vec<(usize, usize)>,
    graphemes: &GraphemeIndex,
) -> Vec<(usize, usize)> {
    // Fast path: when no adjacent pair shares a mid-cluster edge, the
    // merge would return the input unchanged — so it does, the input Vec
    // moving through with no copy. That is the common case (any text
    // without UAX-29/grapheme boundary divergence, e.g. pure-ASCII prose
    // with no CRLF pairs, where the merge pass would otherwise duplicate
    // a multi-MiB bounds list just to push every segment through). One
    // allocation-free scan decides; the differential tests pin both
    // paths to the same output.
    if !bounds
        .windows(2)
        .any(|w| w[0].1 == w[1].0 && !graphemes.is_boundary(w[1].0))
    {
        return bounds;
    }
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(bounds.len());
    for (start, end) in bounds {
        match merged.last_mut() {
            Some(last) if last.1 == start && !graphemes.is_boundary(start) => {
                last.1 = end;
            }
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// `chunk_by_words`'s real-token filter — keep exactly the segments
/// carrying at least one non-whitespace codepoint — as ONE streaming
/// decode pass over `text` instead of the former whole-text `Vec<char>`
/// collect (4 bytes per codepoint materialized just to random-access
/// slice each segment, the same #22 allocation class). `merged` is a
/// contiguous covering partition of `[0, total)` (word_bounds' own
/// contract, preserved by the merge), so a single forward decode crosses
/// every segment edge in sequence: each decoded codepoint folds into the
/// current segment's has-a-non-whitespace flag, and each completed
/// segment is kept or dropped on that flag alone. O(text) time, O(1)
/// memory beyond the output.
fn retain_non_whitespace_segments(text: &str, merged: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut bounds = Vec::with_capacity(merged.len());
    let mut seg = 0usize;
    let mut has_non_ws = false;
    let mut cp = 0usize;
    // The partition's contiguity, debug-pinned: segment `seg` must begin
    // exactly where the decode cursor is when it opens.
    let mut seg_start_cp = 0usize;
    for ch in text.chars() {
        if seg < merged.len() && cp == merged[seg].1 {
            debug_assert_eq!(merged[seg].0, seg_start_cp);
            if has_non_ws {
                bounds.push(merged[seg]);
            }
            seg += 1;
            seg_start_cp = cp;
            has_non_ws = false;
        }
        has_non_ws |= !ch.is_whitespace();
        cp += 1;
    }
    // The final segment completes at the decode's end, not before another
    // codepoint arrives. A partition that stopped short of the decode's
    // end (a contract violation, since word_bounds covers the whole text)
    // leaves a segment unconsumed: caught here in debug.
    if seg < merged.len() && cp == merged[seg].1 {
        debug_assert_eq!(merged[seg].0, seg_start_cp);
        if has_non_ws {
            bounds.push(merged[seg]);
        }
        seg += 1;
    }
    debug_assert_eq!(seg, merged.len());
    bounds
}

/// The shared "N segments per chunk, Y segments of overlap" walk behind
/// all four unit chunkers — [`chunk_by_words`], [`chunk_by_sentences`],
/// [`chunk_by_paragraphs`], and [`chunk_by_lines`]: the only difference
/// between them is which segmenter produced `bounds`, so the windowing
/// logic itself is factored here once rather than duplicated per unit
/// (the `elapsed_exceeds` precedent: factor a second consumer, don't copy
/// it). `bounds` is an ascending, non-overlapping segment list; the four
/// producers split on contiguity — the word/sentence producers
/// (`segmentation_impl::word_bounds`/`sentence_bounds`) are contiguous
/// coverings of the whole text, while the paragraph/line producers
/// ([`paragraph_bounds`]/[`line_bounds`]) are gapped (each excludes the
/// break runs it splits on, so the span between two consecutive segments
/// belongs to neither). The walk only ever reads `bounds[i].0` and
/// `bounds[j - 1].1`, so contiguity is genuinely not required — only the
/// ascending, non-overlapping part of the contract is. This function
/// trusts that contract and does not re-validate it.
///
/// Each chunk spans `per_chunk` consecutive segments, `[bounds[i].0,
/// bounds[i + per_chunk - 1].1)`, except possibly the LAST chunk, which
/// takes whatever remains when the segment count doesn't divide evenly.
/// Consecutive chunks advance by `stride = per_chunk - overlap` segments
/// (`overlap < per_chunk` is the caller's precondition, so
/// `stride >= 1` always: unconditional forward progress by construction,
/// no runtime check needed the way `chunk_text_overlapping`'s
/// character-granularity snapping needs one). Empty `bounds` (empty text)
/// yields `[]`.
fn chunk_by_segments(
    bounds: &[(usize, usize)],
    per_chunk: usize,
    overlap: usize,
) -> Vec<(usize, usize)> {
    if bounds.is_empty() {
        return Vec::new();
    }
    // `assert!`, not `debug_assert!`: this function is `pub` Rust API in its
    // own right (reachable without going through the pyo3 validation these
    // four callers' Python bindings apply), and both preconditions are
    // load-bearing for `stride`'s arithmetic below: with overflow checks
    // off in a release build (the crate's default profile), a violated
    // `overlap < per_chunk` would underflow `per_chunk - overlap` into a
    // huge `usize` silently rather than panic, corrupting the walk instead
    // of failing loudly. The same discipline `chunk_text`/
    // `chunk_text_overlapping`'s own `assert!`s already apply.
    assert!(per_chunk > 0, "per_chunk must be at least 1, got 0");
    assert!(
        overlap < per_chunk,
        "overlap must be less than per_chunk (no forward progress otherwise), \
         got overlap={overlap}, per_chunk={per_chunk}"
    );
    let stride = per_chunk - overlap;
    let n = bounds.len();
    let mut chunks = Vec::with_capacity(n.div_ceil(stride));
    let mut i = 0usize;
    loop {
        let j = (i + per_chunk).min(n);
        chunks.push((bounds[i].0, bounds[j - 1].1));
        if j >= n {
            break;
        }
        i += stride;
    }
    chunks
}

/// Word-count-windowed chunking: each chunk spans `words_per_chunk`
/// consecutive WORD TOKENS, not `word_bounds`' raw segment
/// count. `word_bounds` itself follows UAX #29 exactly, which gives an
/// inter-word space run its OWN segment (`"one two"` is three segments:
/// `"one"`, `" "`, `"two"`), the established convention `word_count`
/// already carries. Grouping RAW segments here would silently mean
/// "`words_per_chunk` roughly halved" for ordinary space-separated
/// prose, the opposite of what a caller reaching for
/// `words_per_chunk=100` (a "~100 word chunk" for an embedding budget)
/// actually wants. So this filters `word_bounds`' output to segments
/// that carry at least one non-whitespace codepoint FIRST, and only
/// then windows over what remains: a "word" here is a real token, and
/// the whitespace between two tokens in one chunk still rides along
/// naturally (the span is a contiguous slice of the ORIGINAL text
/// between two real absolute offsets, not a re-assembly of kept
/// segments), exactly as it would if nothing had been filtered.
///
/// `(start, end)` are codepoint offsets spanning the first included
/// word token's start through the last included token's end (NOT
/// through any trailing whitespace after it: that whitespace belongs
/// to neither this chunk nor the next one's word tokens, so
/// non-overlapping chunks are no longer necessarily contiguous, unlike
/// `chunk_text`'s covering-partition contract; this function makes no
/// such claim). The final chunk may hold fewer than `words_per_chunk`
/// tokens when the total doesn't divide evenly. Empty text, or text
/// with no word tokens at all (pure whitespace), yields `[]`.
/// `words_per_chunk == 0` or `overlap >= words_per_chunk` are the pyo3
/// layer's `ValueError`s (this core trusts its precondition, matching
/// [`chunk_by_segments`]'s own contract).
///
/// Grapheme-cluster-safe at every window edge, the same fix
/// `crate::chunk_impl::chunk_text` applies: `word_bounds` occasionally
/// scores a combining sequence (e.g. Thai SARA AM) as its own
/// word-segment even though it's one grapheme cluster, and a chunk
/// boundary landing there would silently split it:
/// [`merge_mid_cluster_boundaries`] closes this before windowing starts.
pub fn chunk_by_words(text: &str, words_per_chunk: usize, overlap: usize) -> Vec<(usize, usize)> {
    // Merge any word_bounds segment edge that would split a grapheme
    // cluster (the SARA AM edge, see `merge_mid_cluster_boundaries`)
    // BEFORE filtering out whitespace-only segments: the merge relies on
    // `word_bounds`' raw covering-partition contiguity, which the
    // whitespace filter below would otherwise break (it opens gaps).
    let graphemes = GraphemeIndex::build(text, char_count(text));
    let merged = merge_mid_cluster_boundaries(segmentation_impl::word_bounds(text), &graphemes);
    let bounds = retain_non_whitespace_segments(text, merged);
    chunk_by_segments(&bounds, words_per_chunk, overlap)
}

/// [`chunk_by_words`]'s sentence-count twin: each chunk spans
/// `sentences_per_chunk` consecutive UAX #29 sentence segments
/// (`sentence_bounds`), `overlap` sentences repeated. Same contract,
/// same preconditions, same empty-input answer, same
/// grapheme-cluster-safe window edges (see [`chunk_by_words`]'s docs).
pub fn chunk_by_sentences(
    text: &str,
    sentences_per_chunk: usize,
    overlap: usize,
) -> Vec<(usize, usize)> {
    // Same grapheme-cluster merge as chunk_by_words, applied to
    // sentence_bounds' segments (also a covering, contiguous partition,
    // so the merge's contiguity assumption holds directly: no
    // whitespace-filter step exists here to reorder around).
    let graphemes = GraphemeIndex::build(text, char_count(text));
    let bounds = merge_mid_cluster_boundaries(segmentation_impl::sentence_bounds(text), &graphemes);
    chunk_by_segments(&bounds, sentences_per_chunk, overlap)
}

/// Paragraph boundaries: `text` split on maximal runs of 2+ NEWLINE
/// UNITS: `\r\n` counts as ONE unit (matching `normalize`'s own
/// CR/CRLF folding), a lone `\r` or `\n` also one unit each. This is
/// the same "2+ newlines is the surviving paragraph gap" convention
/// `normalize`'s own pipeline already establishes (it collapses 3+
/// consecutive newlines down to exactly 2, never below: see
/// `normalize_impl::flush`). There is NO Unicode Standard segmentation
/// for paragraphs (unlike UAX #29 for words/sentences), so this is a
/// heuristic, stated plainly, not a spec-backed segmenter: a single `\n`
/// is ordinary content here, not a break (`"A\nB"` is one paragraph),
/// and a "blank-looking" line that holds only spaces/tabs between two
/// LONE newlines does NOT qualify: only an actual run of 2+ newline
/// characters does. This operates on `text` as given, not on any prior
/// `normalize` pass.
///
/// Each returned span is one paragraph's content, `(start, end)`
/// codepoint offsets, EXCLUDING the separating run itself (a paragraph's
/// span shouldn't include the gap that separates it from the next one).
/// A leading or trailing qualifying run produces an empty span at that
/// edge, which is DISCARDED rather than emitted: an empty "paragraph"
/// is not a useful chunk. Text with no qualifying run at all yields
/// exactly one paragraph: the whole text. Empty input yields `[]`.
///
/// UNLIKE `word_bounds`/`sentence_bounds`, this split point is
/// structurally grapheme-safe with no merge step needed: every split
/// happens strictly INSIDE a run of `\n`/`\r` characters (`\r\n` is
/// consumed as one unit, matching `normalize`'s own CRLF folding, so a
/// CRLF pair is never itself torn in two), and neither character is a
/// combining mark: a grapheme cluster spanning a newline would require a
/// combining mark to immediately follow it, at which point the newline
/// (never emitted in any paragraph's span: it's discarded as separator
/// content) is a non-printing control character, not text either
/// paragraph's caller would consider "split". No visible content
/// character is ever cut mid-cluster by this function.
pub(crate) fn paragraph_bounds(text: &str) -> Vec<(usize, usize)> {
    // The scan as ONE streaming decode pass — a three-flag state machine
    // (in-run, unit count, pending-CR) instead of the former whole-text
    // `Vec<char>` collect with random access and a lookahead, the same
    // #22 allocation class: O(text) time, O(1) memory beyond the output.
    // The codepoint total the end-of-text close needs is derived inside
    // the walk (`total = cp + 1` per codepoint) rather than paid as a
    // separate whole-text `char_count` pass before it — one decode of
    // the text, not two. The pending-CR flag IS the lookahead: a '\r'
    // counts one unit and stays pending; a following '\n' completes the
    // CRLF pair without adding a unit; anything else leaves the '\r'
    // standing as its own unit (already counted).
    let mut bounds = Vec::new();
    let mut total = 0usize;
    let mut seg_start = 0usize;
    let mut in_run = false;
    let mut run_start = 0usize;
    let mut units = 0usize;
    let mut pending_cr = false;
    for (cp, ch) in text.chars().enumerate() {
        total = cp + 1;
        match ch {
            '\r' => {
                if !in_run {
                    in_run = true;
                    run_start = cp;
                    units = 0;
                }
                units += 1;
                pending_cr = true;
            }
            '\n' => {
                if !in_run {
                    in_run = true;
                    run_start = cp;
                    units = 0;
                }
                if pending_cr {
                    // Completes the pending CRLF pair: one unit total.
                    pending_cr = false;
                } else {
                    units += 1;
                }
            }
            _ => {
                if in_run {
                    if units >= 2 {
                        if seg_start < run_start {
                            bounds.push((seg_start, run_start));
                        }
                        seg_start = cp;
                    }
                    // A single-unit run is ordinary content: no split.
                    in_run = false;
                    pending_cr = false;
                }
            }
        }
    }
    // End of text terminates a trailing run the same way: seg_start moves
    // past the run whenever the run qualifies (the push itself is guarded
    // by seg_start < run_start, the leading-empty-paragraph discard).
    if in_run && units >= 2 {
        if seg_start < run_start {
            bounds.push((seg_start, run_start));
        }
        seg_start = total;
    }
    // The final paragraph is guarded by `seg_start < total` — KEPT,
    // unlike `line_bounds`' content-filter guard: paragraph_bounds has
    // no has-content flag, so the guard is the only thing standing
    // between a trailing qualifying run's `seg_start = total` and a
    // phantom empty paragraph. Empty text falls out with no special
    // case: the loop never runs, `total` stays 0, `0 < 0` is false.
    if seg_start < total {
        bounds.push((seg_start, total));
    }
    bounds
}

/// [`chunk_by_words`]'s paragraph-count twin: each chunk spans
/// `paragraphs_per_chunk` consecutive [`paragraph_bounds`] segments,
/// `overlap` PARAGRAPHS repeated. Same contract, same preconditions,
/// same empty-input answer: see [`paragraph_bounds`] for exactly what
/// counts as a paragraph boundary here (a heuristic, not a Unicode
/// Standard segmentation). UNLIKE the word/line twins, paragraphs have
/// NO content filter: a whitespace-only paragraph IS emitted as a chunk
/// (only fully-empty spans are dropped), so an overlapping pair of
/// chunks can share blank content.
pub fn chunk_by_paragraphs(
    text: &str,
    paragraphs_per_chunk: usize,
    overlap: usize,
) -> Vec<(usize, usize)> {
    let bounds = paragraph_bounds(text);
    chunk_by_segments(&bounds, paragraphs_per_chunk, overlap)
}

/// Line boundaries: `text` split on LINE-BREAK UNITS, where a unit is a
/// `\n`, a lone `\r`, or a `\r\n` pair counted as ONE (the same
/// CR/CRLF-folding convention [`paragraph_bounds`] and `normalize`'s own
/// pipeline already use; the exotic Unicode line separators
/// `str.splitlines` also honors — `\v`, `\f`, NEL, LS, PS — are NOT line
/// breaks here, keeping this family's "what `normalize` folds is what
/// splits" convention). Every break unit terminates exactly one line;
/// each returned span is one line's content, `(start, end)` codepoint
/// offsets EXCLUDING the break unit itself, and a trailing break at end
/// of text yields no trailing empty line (there is no content after it).
///
/// A line counts as a line only when it carries at least one
/// non-whitespace codepoint — the same real-token discipline
/// [`chunk_by_words`] applies to `word_bounds` segments (an inter-word
/// space run is not a word, a blank line is not a line): one message per
/// line (a chat thread), one record per line (a log), one cue per block
/// are the shapes this exists for, and a caller reaching for
/// `lines_per_chunk=200` wants 200 content lines, not "200 lines, of
/// which 40 are blank separators". "Non-whitespace" is definitional
/// here: the Unicode `White_Space` property (`char::is_whitespace`),
/// under which U+001C–U+001F (FS/GS/RS/US) count as CONTENT (Python's
/// `str.isspace()` treats them as whitespace, and `str.splitlines`
/// even breaks on them, so a ported expectation may differ) and NBSP
/// counts as blank. The blank lines between two counted
/// lines of the SAME chunk still ride along inside its span (the span is
/// a contiguous slice of the ORIGINAL text between two absolute offsets,
/// exactly as inter-word whitespace rides along in `chunk_by_words`);
/// they belong to neither chunk when the counted lines land in different
/// chunks. Whitespace-only text, or empty input, yields `[]`.
///
/// UNLIKE the `word_bounds`/`sentence_bounds` spellings (but exactly like
/// [`paragraph_bounds`]), this split point is structurally grapheme-safe
/// with no merge step: every split lands strictly between a break
/// character and adjacent content, and the break characters are never
/// combining marks. The one theoretical divergence — a combining mark
/// immediately after a newline joins the NEWLINE's cluster, so the next
/// line's span would start mid-cluster — is the same documented
/// non-issue [`paragraph_bounds`] carries: the "cluster" is a newline
/// plus an orphan combining mark, not visible content any line's caller
/// would call "split".
pub(crate) fn line_bounds(text: &str) -> Vec<(usize, usize)> {
    // The scan as ONE streaming decode pass, the same shape as
    // `paragraph_bounds`' state machine (no whole-text `Vec<char>`
    // collect, and no `char_count` pre-pass either: the codepoint total
    // the end-of-text close needs is derived inside the walk, `total =
    // cp + 1` per codepoint, so "one pass" is now literally true): a
    // `pending_cr` flag is the CRLF lookahead, and a `has_non_ws` flag
    // carries the real-line filter so the segment list is built in the
    // same pass that finds the breaks. O(text) time, O(lines) memory.
    let mut bounds = Vec::new();
    let mut total = 0usize;
    let mut seg_start = 0usize;
    let mut has_non_ws = false;
    let mut pending_cr = false;
    for (cp, ch) in text.chars().enumerate() {
        total = cp + 1;
        match ch {
            '\r' => {
                // Opens a break unit whether it stands alone or begins a
                // CRLF pair: the current line ends at the '\r' itself.
                if has_non_ws {
                    bounds.push((seg_start, cp));
                }
                seg_start = cp + 1;
                has_non_ws = false;
                pending_cr = true;
            }
            '\n' => {
                if pending_cr {
                    // Completes the CRLF pair: the line already ended at
                    // the '\r'; the next line begins after this '\n'.
                    seg_start = cp + 1;
                    pending_cr = false;
                } else {
                    if has_non_ws {
                        bounds.push((seg_start, cp));
                    }
                    seg_start = cp + 1;
                    has_non_ws = false;
                }
            }
            _ => {
                has_non_ws |= !ch.is_whitespace();
                pending_cr = false;
            }
        }
    }
    // End of text closes the final line, and `has_non_ws` ALONE is the
    // phantom-line guarantee: it resets together with `seg_start` at
    // every break unit, and only a codepoint at `cp >= seg_start` can
    // set it after the last reset, so `has_non_ws` holding at end of
    // text implies a real line at `(seg_start, total)` — the former
    // `seg_start < total` half of the guard was implied by exactly that
    // argument and is gone. A trailing break unit already reset
    // `has_non_ws`; empty text never enters the loop, so the flag stays
    // false and `[]` falls out with no special case.
    if has_non_ws {
        bounds.push((seg_start, total));
    }
    bounds
}

/// [`chunk_by_words`]'s line-count twin: each chunk spans
/// `lines_per_chunk` consecutive [`line_bounds`] segments, `overlap`
/// LINES repeated at the start of the next chunk. Same contract, same
/// preconditions, same empty-input answer as its siblings; see
/// [`line_bounds`] for exactly what counts as a line here (a
/// content-carrying, newline-terminated segment — blank lines neither
/// count nor split a chunk's interior, "content" being the Unicode
/// `White_Space` reading [`line_bounds`] pins, not Python's
/// `str.isspace()` notion).
pub fn chunk_by_lines(text: &str, lines_per_chunk: usize, overlap: usize) -> Vec<(usize, usize)> {
    let bounds = line_bounds(text);
    chunk_by_segments(&bounds, lines_per_chunk, overlap)
}

#[cfg(test)]
mod tests {
    use super::*;
    // The differential oracles below are the pre-#22 spellings verbatim
    // (whole-text `Vec<char>` collects, `HashSet<usize>` boundary sets,
    // the random-access paragraph scan), which is why the tests module
    // re-imports what production no longer uses.
    use std::collections::HashSet;

    use crate::truncate_impl::grapheme_boundary_chars;

    /// The former `merge_mid_cluster_boundaries`, HashSet spelling, kept
    /// verbatim as the differential oracle for the bitmap spelling.
    fn merge_mid_cluster_boundaries_reference(
        bounds: Vec<(usize, usize)>,
        grapheme_set: &HashSet<usize>,
    ) -> Vec<(usize, usize)> {
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(bounds.len());
        for (start, end) in bounds {
            match merged.last_mut() {
                Some(last) if last.1 == start && !grapheme_set.contains(&start) => {
                    last.1 = end;
                }
                _ => merged.push((start, end)),
            }
        }
        merged
    }

    /// The former `chunk_by_words`, verbatim oracle: `Vec<char>` collect,
    /// `HashSet` boundary set, random-access whitespace slices.
    fn chunk_by_words_reference(
        text: &str,
        words_per_chunk: usize,
        overlap: usize,
    ) -> Vec<(usize, usize)> {
        let chars: Vec<char> = text.chars().collect();
        let grapheme_set: HashSet<usize> = grapheme_boundary_chars(text).into_iter().collect();
        let merged = merge_mid_cluster_boundaries_reference(
            segmentation_impl::word_bounds(text),
            &grapheme_set,
        );
        let bounds: Vec<(usize, usize)> = merged
            .into_iter()
            .filter(|&(start, end)| chars[start..end].iter().any(|c| !c.is_whitespace()))
            .collect();
        chunk_by_segments(&bounds, words_per_chunk, overlap)
    }

    /// The former `chunk_by_sentences`, verbatim oracle.
    fn chunk_by_sentences_reference(
        text: &str,
        sentences_per_chunk: usize,
        overlap: usize,
    ) -> Vec<(usize, usize)> {
        let grapheme_set: HashSet<usize> = grapheme_boundary_chars(text).into_iter().collect();
        let bounds = merge_mid_cluster_boundaries_reference(
            segmentation_impl::sentence_bounds(text),
            &grapheme_set,
        );
        chunk_by_segments(&bounds, sentences_per_chunk, overlap)
    }

    /// The former `paragraph_bounds`, verbatim oracle: the whole-text
    /// `Vec<char>` collect with random access and one-codepoint lookahead.
    fn paragraph_bounds_reference(text: &str) -> Vec<(usize, usize)> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }
        let mut bounds = Vec::new();
        let mut seg_start = 0usize;
        let mut i = 0usize;
        while i < n {
            if chars[i] == '\n' || chars[i] == '\r' {
                let run_start = i;
                let mut units = 0usize;
                while i < n && (chars[i] == '\n' || chars[i] == '\r') {
                    if chars[i] == '\r' && i + 1 < n && chars[i + 1] == '\n' {
                        i += 2;
                    } else {
                        i += 1;
                    }
                    units += 1;
                }
                if units >= 2 {
                    if seg_start < run_start {
                        bounds.push((seg_start, run_start));
                    }
                    seg_start = i;
                }
            } else {
                i += 1;
            }
        }
        if seg_start < n {
            bounds.push((seg_start, n));
        }
        bounds
    }

    /// The differential corpus: word/sentence/paragraph shapes — prose,
    /// CRLF and lone-CR runs (every unit-counting case the paragraph
    /// scanner has: CRLF pairs, mixed \n\r, trailing and leading runs),
    /// Thai SARA AM (the merge's reason to exist), whitespace-only and
    /// whitespace-heavy text (the token filter's drop-everything and
    /// ride-along cases), and degenerate runs.
    fn differential_corpus() -> Vec<String> {
        vec![
            String::new(),
            "   ".to_string(),
            "a".to_string(),
            "one two three four five six seven eight".to_string(),
            "Alpha beta gamma delta. Epsilon zeta eta. Theta iota kappa.".to_string(),
            "x0\u{0E33}y0\u{0E33}z".to_string(),
            "One 0\u{0E33} fish. Two 0\u{0E33} fish.".to_string(),
            "a\n\nb\n\n\nc".to_string(),
            "a\r\nb\r\n\r\nc".to_string(),
            "a\rb".to_string(),
            "\r\n\r\n\r\n".to_string(),
            "\n\na".to_string(),
            "a\n\n".to_string(),
            "a\r\rb\n\n\nc".to_string(),
            "para one\n\npara two\nsingle\n\npara three".to_string(),
            " \t \n\n \t ".to_string(),
            "e\u{0301}e\u{0301} words here".to_string(),
            "q".repeat(300),
            "word ".repeat(120),
        ]
    }

    #[test]
    fn streaming_spellings_match_the_former_implementations_exactly() {
        for text in differential_corpus() {
            // paragraph_bounds: the state machine against the random-access
            // scan, over every corpus shape (CRLF pairing, unit counts at
            // the split threshold, leading/trailing runs).
            assert_eq!(
                paragraph_bounds(&text),
                paragraph_bounds_reference(&text),
                "paragraph_bounds divergence on {text:?}"
            );
            // The unit-count chunkers over the validated envelope
            // (per_chunk >= 1, overlap < per_chunk), sweeping the
            // boundary-adjacent values.
            for per_chunk in 1usize..=6 {
                for overlap in [0usize, 1, per_chunk.saturating_sub(1)]
                    .into_iter()
                    .filter(|&o| o < per_chunk)
                {
                    assert_eq!(
                        chunk_by_words(&text, per_chunk, overlap),
                        chunk_by_words_reference(&text, per_chunk, overlap),
                        "chunk_by_words divergence: text={text:?} per={per_chunk} ov={overlap}"
                    );
                    assert_eq!(
                        chunk_by_sentences(&text, per_chunk, overlap),
                        chunk_by_sentences_reference(&text, per_chunk, overlap),
                        "chunk_by_sentences divergence: text={text:?} per={per_chunk} ov={overlap}"
                    );
                    assert_eq!(
                        chunk_by_paragraphs(&text, per_chunk, overlap),
                        chunk_by_segments(&paragraph_bounds_reference(&text), per_chunk, overlap),
                        "chunk_by_paragraphs divergence: text={text:?} per={per_chunk} ov={overlap}"
                    );
                }
            }
        }
    }

    #[test]
    fn paragraph_state_machine_survives_a_deterministic_newline_soup() {
        // Pseudo-random text over exactly the alphabet the paragraph
        // scanner branches on (\r, \n, CRLF pairings, and one ordinary
        // character), so the state machine's unit counting is checked
        // against the random-access oracle on runs no hand-written corpus
        // anticipates — including runs that end at end-of-text.
        let mut state = 0x853C49E6748FEA9Bu64;
        let alphabet = ['x', '\r', '\n'];
        for _ in 0..300 {
            let mut text = String::new();
            for _ in 0..(state % 60 + 1) as usize {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                text.push(alphabet[(state >> 33) as usize % alphabet.len()]);
            }
            assert_eq!(
                paragraph_bounds(&text),
                paragraph_bounds_reference(&text),
                "divergence on {text:?}"
            );
        }
    }

    // ---- grapheme-cluster safety (the truncate_impl regression, re-derived here) ----

    #[test]
    fn chunk_by_words_never_splits_a_thai_sara_am_cluster_across_two_words() {
        // Raw word_bounds("x0ำy0ำz") = [(0,2)="x0", (2,3)="ำ", (3,5)="y0",
        // (5,6)="ำ", (6,7)="z"]: TWO combining sequences each split into
        // a base-segment + a lone-combining-mark segment. Neither "ำ"
        // segment is whitespace, so a whitespace-only filter would
        // NOT catch this: chunk_by_words(text, 1, 0) would silently
        // return a chunk containing only the bare combining mark. The
        // merge step must fuse each pair into one real word first.
        let text = "x0\u{0E33}y0\u{0E33}z";
        let chunks = chunk_by_words(text, 1, 0);
        let slice =
            |(a, b): (usize, usize)| -> String { text.chars().skip(a).take(b - a).collect() };
        assert_eq!(chunks.len(), 3, "chunks: {chunks:?}");
        assert_eq!(slice(chunks[0]), "x0\u{0E33}");
        assert_eq!(slice(chunks[1]), "y0\u{0E33}");
        assert_eq!(slice(chunks[2]), "z");
        // No chunk is a bare, unattached combining mark.
        for &c in &chunks {
            assert_ne!(slice(c), "\u{0E33}");
        }
    }

    #[test]
    fn chunk_by_sentences_never_splits_a_grapheme_cluster_across_two_sentences() {
        // sentence_bounds is coarser than word_bounds and, empirically,
        // does not isolate the SARA AM combining mark into its own
        // sentence segment for ordinary sentence-terminated text, but
        // the merge step runs unconditionally (see chunk_by_sentences'
        // implementation), so this pins that no chunk produced by it
        // ever starts or ends strictly inside a cluster, regardless.
        let text = "One 0\u{0E33} fish. Two 0\u{0E33} fish.";
        let valid: HashSet<usize> = grapheme_boundary_chars(text).into_iter().collect();
        for sentences_per_chunk in 1..=3 {
            let chunks = chunk_by_sentences(text, sentences_per_chunk, 0);
            for &(a, b) in &chunks {
                assert!(valid.contains(&a), "start {a} mid-cluster: {chunks:?}");
                assert!(valid.contains(&b), "end {b} mid-cluster: {chunks:?}");
            }
        }
    }

    // ---- chunk_by_words / chunk_by_sentences ----

    #[test]
    fn chunk_by_words_groups_exact_word_counts() {
        // word_bounds("the cat sat on the mat") segments: the/ /cat/ /sat/
        // /on/ /the/ /mat: 11 raw segments (6 real word tokens + 5
        // inter-word spaces, each its own WB segment), but chunk_by_words
        // filters the whitespace-only segments out FIRST so "2 words per
        // chunk" means 2 real tokens, not 2 raw segments (which would
        // silently be ~1 real word per chunk on ordinary prose). 2 words
        // per chunk, no overlap: 3 chunks of 2 real words each, spans
        // still contiguous slices of the ORIGINAL text (inter-word space
        // inside a chunk rides along naturally).
        let text = "the cat sat on the mat";
        let chunks = chunk_by_words(text, 2, 0);
        let slice =
            |(a, b): (usize, usize)| -> String { text.chars().skip(a).take(b - a).collect() };
        assert_eq!(chunks.len(), 3);
        assert_eq!(slice(chunks[0]), "the cat");
        assert_eq!(slice(chunks[1]), "sat on");
        assert_eq!(slice(chunks[2]), "the mat");
        // Non-overlapping in this overlap=0 case, but chunks are NOT
        // necessarily contiguous any more (the space between "cat" and
        // "sat" belongs to neither chunk): this function makes no
        // covering-partition claim, unlike chunk_text.
        for w in chunks.windows(2) {
            assert!(w[1].0 >= w[0].1);
        }
    }

    #[test]
    fn chunk_by_words_counts_real_tokens_not_raw_word_bounds_segments() {
        // The regression this pins: word_bounds gives an inter-word space
        // run its OWN segment, so a naive "group N raw segments" reading
        // of "words_per_chunk" would silently mean roughly HALF as many
        // real words per chunk on ordinary space-separated prose.
        // "one two three four five six seven" has 7 real word tokens (13
        // raw word_bounds segments, 7 words + 6 spaces): 3 per chunk
        // must yield exactly ceil(7/3) = 3 chunks, the last holding the
        // remaining 1 word, never the wrong (roughly-halved) count a
        // raw-segment grouping would produce.
        let text = "one two three four five six seven";
        let chunks = chunk_by_words(text, 3, 0);
        assert_eq!(chunks.len(), 3, "chunks: {chunks:?}");
        let slice =
            |(a, b): (usize, usize)| -> String { text.chars().skip(a).take(b - a).collect() };
        assert_eq!(slice(chunks[0]), "one two three");
        assert_eq!(slice(chunks[1]), "four five six");
        assert_eq!(slice(chunks[2]), "seven");
    }

    #[test]
    fn chunk_by_words_overlap_repeats_words_at_each_boundary() {
        let text = "one two three four five six seven";
        let chunks = chunk_by_words(text, 3, 1);
        assert!(chunks.len() >= 2);
        for w in chunks.windows(2) {
            assert!(
                w[1].0 < w[0].1,
                "no overlap between {:?} and {:?}",
                w[0],
                w[1]
            );
            assert!(
                w[1].0 > w[0].0,
                "no forward progress: {:?} -> {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn chunk_by_words_overlap_actually_shares_content_langchain_34804_regression() {
        // LangChain issue #34804: chunk_overlap was silently a no-op
        // except when a size-overflow forced a merge: a real, shipped
        // bug in the most popular chunking library. The regression this
        // pins: consecutive chunks must share GENUINE, non-empty text,
        // not merely satisfy a position check that happens to coincide
        // with hitting a size ceiling. 8 words, no chunk here divides
        // evenly to a size ceiling by coincidence; the overlap must still
        // manifest as literal shared text on every transition.
        let text = "alpha beta gamma delta epsilon zeta eta theta";
        let chunks = chunk_by_words(text, 3, 1);
        assert!(chunks.len() >= 2, "chunks: {chunks:?}");
        let slice =
            |(a, b): (usize, usize)| -> String { text.chars().skip(a).take(b - a).collect() };
        for w in chunks.windows(2) {
            let (prev_start, prev_end) = w[0];
            let (next_start, next_end) = w[1];
            let shared: String = text
                .chars()
                .skip(next_start)
                .take(prev_end.saturating_sub(next_start))
                .collect();
            assert!(
                !shared.trim().is_empty(),
                "no genuine shared content between {:?} and {:?}",
                w[0],
                w[1]
            );
            // The shared span reads identically from EITHER chunk's own
            // text (it's the same underlying offsets on both sides).
            let from_prev = &slice((prev_start, prev_end))[(next_start - prev_start)..];
            let from_next = &slice((next_start, next_end))[..(prev_end - next_start)];
            assert_eq!(from_prev, from_next);
            assert_eq!(from_prev, shared);
        }
    }

    #[test]
    fn chunk_by_sentences_groups_exact_sentence_counts() {
        let text = "One. Two. Three. Four. Five.";
        let chunks = chunk_by_sentences(text, 2, 0);
        assert!(chunks.len() >= 2);
        let mut prev_end = 0usize;
        for &(a, b) in &chunks {
            assert_eq!(a, prev_end);
            prev_end = b;
        }
        assert_eq!(prev_end, text.chars().count());
    }

    #[test]
    fn chunk_by_sentences_overlap_repeats_sentences() {
        let text = "One. Two. Three. Four. Five. Six.";
        let chunks = chunk_by_sentences(text, 3, 1);
        assert!(chunks.len() >= 2);
        for w in chunks.windows(2) {
            assert!(w[1].0 < w[0].1);
            assert!(w[1].0 > w[0].0);
        }
    }

    #[test]
    fn chunk_by_sentences_overlap_actually_shares_content_langchain_34804_regression() {
        // Same LangChain #34804 regression as chunk_by_words' twin: the
        // shared span must be real, extractable text, not merely a
        // position check; and this must hold even though 6 sentences
        // over 3-per-chunk/1-overlap doesn't divide to any size ceiling.
        let text = "One. Two. Three. Four. Five. Six.";
        let chunks = chunk_by_sentences(text, 3, 1);
        assert!(chunks.len() >= 2, "chunks: {chunks:?}");
        let chars: Vec<char> = text.chars().collect();
        let slice = |(a, b): (usize, usize)| -> String { chars[a..b].iter().collect() };
        for w in chunks.windows(2) {
            let (_, prev_end) = w[0];
            let (next_start, _) = w[1];
            assert!(
                next_start < prev_end,
                "no overlap: {:?} -> {:?}",
                w[0],
                w[1]
            );
            let shared: String = chars[next_start..prev_end].iter().collect();
            assert!(
                !shared.trim().is_empty(),
                "no genuine shared content between {:?} and {:?}",
                w[0],
                w[1]
            );
            assert!(slice(w[0]).ends_with(&shared));
            assert!(slice(w[1]).starts_with(&shared));
        }
    }

    #[test]
    fn chunk_by_word_or_sentence_empty_text_is_no_chunks() {
        assert_eq!(chunk_by_words("", 3, 0), Vec::<(usize, usize)>::new());
        assert_eq!(chunk_by_sentences("", 3, 0), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn paragraph_bounds_splits_on_blank_line_runs() {
        let text = "First para.\n\nSecond para.\n\nThird para.";
        let bounds = paragraph_bounds(text);
        let chars: Vec<char> = text.chars().collect();
        let pieces: Vec<String> = bounds
            .iter()
            .map(|&(a, b)| chars[a..b].iter().collect())
            .collect();
        assert_eq!(pieces, vec!["First para.", "Second para.", "Third para."]);
    }

    #[test]
    fn paragraph_bounds_a_single_newline_is_not_a_break() {
        let text = "line one\nline two";
        assert_eq!(paragraph_bounds(text), vec![(0, text.chars().count())]);
    }

    #[test]
    fn paragraph_bounds_three_plus_newlines_are_still_one_break() {
        let text = "First.\n\n\n\nSecond.";
        let bounds = paragraph_bounds(text);
        assert_eq!(bounds.len(), 2);
        let chars: Vec<char> = text.chars().collect();
        let second: String = chars[bounds[1].0..bounds[1].1].iter().collect();
        assert_eq!(second, "Second.");
    }

    #[test]
    fn paragraph_bounds_crlf_run_counts_as_two_units() {
        let text = "First.\r\n\r\nSecond.";
        let bounds = paragraph_bounds(text);
        let chars: Vec<char> = text.chars().collect();
        let pieces: Vec<String> = bounds
            .iter()
            .map(|&(a, b)| chars[a..b].iter().collect())
            .collect();
        assert_eq!(pieces, vec!["First.", "Second."]);
    }

    #[test]
    fn paragraph_bounds_lone_cr_run_counts_too() {
        let text = "First.\r\rSecond.";
        let bounds = paragraph_bounds(text);
        assert_eq!(bounds.len(), 2);
    }

    #[test]
    fn paragraph_bounds_leading_and_trailing_blank_runs_are_trimmed() {
        let text = "\n\n\nHello\n\n\n";
        let bounds = paragraph_bounds(text);
        let chars: Vec<char> = text.chars().collect();
        assert_eq!(bounds.len(), 1);
        let piece: String = chars[bounds[0].0..bounds[0].1].iter().collect();
        assert_eq!(piece, "Hello");
    }

    #[test]
    fn paragraph_bounds_no_breaks_is_one_paragraph() {
        let text = "just one paragraph, no blank lines at all";
        assert_eq!(paragraph_bounds(text), vec![(0, text.chars().count())]);
    }

    #[test]
    fn paragraph_bounds_empty_text_is_no_paragraphs() {
        assert_eq!(paragraph_bounds(""), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn chunk_by_paragraphs_groups_exact_paragraph_counts() {
        // Unlike chunk_by_sentences (whose UAX #29 segments already carry
        // their own trailing whitespace, so chunks stay contiguous),
        // paragraph_bounds EXCLUDES the separating blank-line run from
        // each paragraph's span, so, like chunk_by_words, chunks here
        // are not necessarily contiguous; assert on content, not on
        // gapless coverage.
        let text = "P1.\n\nP2.\n\nP3.\n\nP4.\n\nP5.";
        let chunks = chunk_by_paragraphs(text, 2, 0);
        let slice =
            |(a, b): (usize, usize)| -> String { text.chars().skip(a).take(b - a).collect() };
        assert_eq!(chunks.len(), 3);
        assert_eq!(slice(chunks[0]), "P1.\n\nP2.");
        assert_eq!(slice(chunks[1]), "P3.\n\nP4.");
        assert_eq!(slice(chunks[2]), "P5.");
    }

    #[test]
    fn chunk_by_paragraphs_overlap_repeats_paragraphs() {
        let text = "P1.\n\nP2.\n\nP3.\n\nP4.\n\nP5.\n\nP6.";
        let chunks = chunk_by_paragraphs(text, 3, 1);
        assert!(chunks.len() >= 2);
        for w in chunks.windows(2) {
            assert!(w[1].0 < w[0].1);
            assert!(w[1].0 > w[0].0);
        }
    }

    #[test]
    fn chunk_by_paragraphs_overlap_actually_shares_content_langchain_34804_regression() {
        // Same LangChain #34804 regression, paragraph-count sibling: the
        // shared span between consecutive chunks must be real,
        // extractable paragraph text.
        let text = "P1.\n\nP2.\n\nP3.\n\nP4.\n\nP5.\n\nP6.";
        let chunks = chunk_by_paragraphs(text, 3, 1);
        assert!(chunks.len() >= 2, "chunks: {chunks:?}");
        let chars: Vec<char> = text.chars().collect();
        let slice = |(a, b): (usize, usize)| -> String { chars[a..b].iter().collect() };
        for w in chunks.windows(2) {
            let (_, prev_end) = w[0];
            let (next_start, _) = w[1];
            assert!(
                next_start < prev_end,
                "no overlap: {:?} -> {:?}",
                w[0],
                w[1]
            );
            let shared: String = chars[next_start..prev_end].iter().collect();
            assert!(
                !shared.trim().is_empty(),
                "no genuine shared content between {:?} and {:?}",
                w[0],
                w[1]
            );
            assert!(slice(w[0]).ends_with(&shared));
            assert!(slice(w[1]).starts_with(&shared));
        }
    }

    #[test]
    fn chunk_by_paragraphs_empty_text_is_no_chunks() {
        assert_eq!(chunk_by_paragraphs("", 3, 0), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn chunk_by_paragraphs_single_chunk_when_fewer_paragraphs_than_per_chunk() {
        let text = "Only one paragraph here.";
        let chunks = chunk_by_paragraphs(text, 100, 0);
        assert_eq!(chunks, vec![(0, text.chars().count())]);
    }

    #[test]
    fn chunk_by_words_single_chunk_when_fewer_words_than_per_chunk() {
        let text = "hi there";
        let chunks = chunk_by_words(text, 100, 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], (0, text.chars().count()));
    }

    #[test]
    fn chunk_by_words_last_chunk_may_be_partial() {
        // 5 word-level segments spelled out won't divide evenly by a
        // stride of 2 in general; confirm the last chunk just takes
        // whatever remains rather than panicking or dropping content.
        let text = "alpha beta gamma";
        let bounds = segmentation_impl::word_bounds(text);
        let chunks = chunk_by_words(text, 2, 0);
        let last = *chunks.last().unwrap();
        assert_eq!(last.1, bounds.last().unwrap().1);
    }

    #[test]
    #[should_panic(expected = "per_chunk must be at least 1")]
    fn chunk_by_segments_panics_on_zero_per_chunk() {
        // Pins that the `per_chunk > 0` precondition is an `assert!`, not a
        // `debug_assert!` that a release build would silently skip: the
        // three pyo3 wrappers already reject this before it ever reaches
        // here, but this function is reachable directly within the crate,
        // and a regression back to `debug_assert!` would only be caught by
        // a release-mode run, never by `cargo test`'s debug profile. This
        // test can't tell `assert!` from `debug_assert!` either (both panic
        // in a debug test build), but it does pin that the condition and
        // message stay intact, and any regression that inlines/removes the
        // check outright still fails it.
        let bounds = [(0usize, 1usize), (1, 2), (2, 3)];
        chunk_by_segments(&bounds, 0, 0);
    }

    #[test]
    #[should_panic(expected = "overlap must be less than per_chunk")]
    fn chunk_by_segments_panics_when_overlap_at_least_per_chunk() {
        let bounds = [(0usize, 1usize), (1, 2), (2, 3)];
        chunk_by_segments(&bounds, 2, 2);
    }

    #[test]
    fn chunk_by_segments_forward_progress_is_unconditional_by_construction() {
        // per_chunk - overlap >= 1 is the caller's precondition (validated
        // at the pyo3 boundary); confirm the resulting stride actually
        // produces strictly increasing starts over a battery, matching
        // chunk_text_overlapping's own runtime-checked guarantee (this one
        // needs no runtime check since stride >= 1 is structural).
        let text = "one two three four five six seven eight nine ten";
        for per_chunk in 1..6 {
            for overlap in 0..per_chunk {
                let chunks = chunk_by_words(text, per_chunk, overlap);
                let mut prev_start: Option<usize> = None;
                for &(start, _) in &chunks {
                    if let Some(p) = prev_start {
                        assert!(start > p, "per_chunk={per_chunk}, overlap={overlap}");
                    }
                    prev_start = Some(start);
                }
            }
        }
    }

    // ---- chunk_by_lines / line_bounds ----

    /// The random-access `line_bounds` oracle: the whole-text `Vec<char>`
    /// collect with a one-codepoint CRLF lookahead, the pre-#22 spelling
    /// style `paragraph_bounds_reference` keeps — the differential pin for
    /// the streaming state machine above.
    fn line_bounds_reference(text: &str) -> Vec<(usize, usize)> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }
        let is_break = |c: char| c == '\n' || c == '\r';
        // A '\r' ALWAYS opens a unit (lone CR, or the first half of a CRLF
        // pair); a '\n' opens one only when it did NOT ride in as the
        // second half of a pair (no '\r' immediately before it).
        let is_unit_start =
            |i: usize| chars[i] == '\r' || (chars[i] == '\n' && !(i > 0 && chars[i - 1] == '\r'));
        let line_has_content = |a: usize, b: usize| chars[a..b].iter().any(|c| !c.is_whitespace());
        let mut bounds = Vec::new();
        let mut seg_start = 0usize;
        let mut i = 0usize;
        while i < n {
            if is_break(chars[i]) {
                if is_unit_start(i) && line_has_content(seg_start, i) {
                    bounds.push((seg_start, i));
                }
                // Whether this codepoint opens a unit (a lone break) or
                // completes one (the '\n' half of a CRLF pair the '\r'
                // opened), the next line begins after it.
                seg_start = i + 1;
            }
            i += 1;
        }
        if seg_start < n && line_has_content(seg_start, n) {
            bounds.push((seg_start, n));
        }
        bounds
    }

    #[test]
    fn line_bounds_streaming_matches_the_random_access_oracle_exactly() {
        for text in differential_corpus() {
            assert_eq!(
                line_bounds(&text),
                line_bounds_reference(&text),
                "line_bounds divergence on {text:?}"
            );
        }
    }

    #[test]
    fn line_bounds_state_machine_survives_a_deterministic_newline_soup() {
        // The paragraph scanner's own discipline, applied to the line
        // scanner: pseudo-random text over exactly the alphabet the state
        // machine branches on (\r, \n, CRLF pairings, one ordinary
        // character), so the unit counting is checked against the
        // random-access oracle on runs no hand-written corpus anticipates.
        let mut state = 0x9E3779B97F4A7C15u64;
        let alphabet = ['x', '\r', '\n', ' '];
        for _ in 0..300 {
            let mut text = String::new();
            for _ in 0..(state % 60 + 1) as usize {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                text.push(alphabet[(state >> 33) as usize % alphabet.len()]);
            }
            assert_eq!(
                line_bounds(&text),
                line_bounds_reference(&text),
                "divergence on {text:?}"
            );
            // The unit-count windowing over the same arbitrary text: same
            // per_chunk/overlap envelope the paragraph differential sweeps.
            let bounds = line_bounds(&text);
            for per_chunk in 1usize..=6 {
                for overlap in [0usize, 1, per_chunk.saturating_sub(1)]
                    .into_iter()
                    .filter(|&o| o < per_chunk)
                {
                    assert_eq!(
                        chunk_by_lines(&text, per_chunk, overlap),
                        chunk_by_segments(&bounds, per_chunk, overlap),
                        "chunk_by_lines divergence: text={text:?} per={per_chunk} ov={overlap}"
                    );
                }
            }
        }
    }

    #[test]
    fn line_bounds_splits_on_every_break_unit() {
        let text = "one\ntwo\r\nthree\rfour";
        let bounds = line_bounds(text);
        let chars: Vec<char> = text.chars().collect();
        let pieces: Vec<String> = bounds
            .iter()
            .map(|&(a, b)| chars[a..b].iter().collect())
            .collect();
        assert_eq!(pieces, vec!["one", "two", "three", "four"]);
    }

    #[test]
    fn line_bounds_crlf_pair_is_one_unit_and_never_torn() {
        // "a\r\nb": the line ends at the '\r', the next begins after the
        // '\n' — the pair is consumed as one break, and the '\n' is not a
        // second line ending that would mint an empty line between them.
        let text = "a\r\nb";
        assert_eq!(line_bounds(text), vec![(0, 1), (3, 4)]);
    }

    #[test]
    fn line_bounds_blank_lines_are_not_lines() {
        // The real-token discipline: blank lines neither count nor split —
        // "a\n\nb" is TWO lines (the empty line between is dropped), and
        // whitespace-only text is zero lines, the same answer
        // chunk_by_words gives pure-whitespace input.
        assert_eq!(line_bounds("a\n\nb"), vec![(0, 1), (3, 4)]);
        assert_eq!(line_bounds(" \n \t \n  "), Vec::<(usize, usize)>::new());
        assert_eq!(line_bounds(""), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn line_bounds_trailing_break_yields_no_phantom_line() {
        // `"a\n".split('\n')` mints a trailing ""; a break at end of text
        // terminates the last line and produces nothing after it.
        assert_eq!(line_bounds("a\n"), vec![(0, 1)]);
        assert_eq!(line_bounds("a\r\n"), vec![(0, 1)]);
    }

    #[test]
    fn line_bounds_interior_blank_lines_ride_inside_a_chunk_span() {
        // chunk spans are contiguous slices of the ORIGINAL text between
        // the first and last counted line's absolute offsets: the blank
        // line between two counted lines of the SAME chunk rides along
        // (exactly as inter-word whitespace rides along in chunk_by_words).
        let text = "msg one\n\nmsg two";
        let total = text.chars().count();
        let chunks = chunk_by_lines(text, 2, 0);
        assert_eq!(chunks, vec![(0, total)]);
        let chunks = chunk_by_lines(text, 1, 0);
        assert_eq!(chunks, vec![(0, 7), (9, 16)]);
    }

    #[test]
    fn chunk_by_lines_groups_exact_line_counts() {
        let text = "l1\nl2\nl3\nl4\nl5";
        let chunks = chunk_by_lines(text, 2, 0);
        let chars: Vec<char> = text.chars().collect();
        let pieces: Vec<String> = chunks
            .iter()
            .map(|&(a, b)| chars[a..b].iter().collect())
            .collect();
        assert_eq!(pieces, vec!["l1\nl2", "l3\nl4", "l5"]);
    }

    #[test]
    fn chunk_by_lines_final_chunk_may_hold_fewer_lines() {
        let text = "a\nb\nc\nd\ne";
        let chunks = chunk_by_lines(text, 3, 0);
        let chars: Vec<char> = text.chars().collect();
        let pieces: Vec<String> = chunks
            .iter()
            .map(|&(a, b)| chars[a..b].iter().collect())
            .collect();
        assert_eq!(pieces, vec!["a\nb\nc", "d\ne"]);
    }

    #[test]
    fn chunk_by_lines_single_chunk_when_fewer_lines_than_per_chunk() {
        let text = "only line";
        assert_eq!(
            chunk_by_lines(text, 100, 0),
            vec![(0, text.chars().count())]
        );
    }

    #[test]
    fn chunk_by_lines_overlap_repeats_whole_lines() {
        let text = "l1\nl2\nl3\nl4\nl5\nl6";
        let chunks = chunk_by_lines(text, 3, 1);
        assert!(chunks.len() >= 2, "chunks: {chunks:?}");
        let slice =
            |(a, b): (usize, usize)| -> String { text.chars().skip(a).take(b - a).collect() };
        for w in chunks.windows(2) {
            let (_, prev_end) = w[0];
            let (next_start, _) = w[1];
            assert!(
                next_start < prev_end,
                "no overlap: {:?} -> {:?}",
                w[0],
                w[1]
            );
            // The shared span must be real, extractable text (the
            // LangChain #34804 regression, line-count sibling), and it
            // must be WHOLE lines: the overlap starts at a counted line's
            // own start, never inside it.
            let shared = slice((next_start, prev_end));
            assert!(!shared.trim().is_empty());
            assert!(slice(w[0]).ends_with(&shared));
            assert!(slice(w[1]).starts_with(&shared));
        }
    }

    #[test]
    fn chunk_by_lines_empty_or_blank_text_is_no_chunks() {
        assert_eq!(chunk_by_lines("", 3, 0), Vec::<(usize, usize)>::new());
        assert_eq!(
            chunk_by_lines(" \n \t\n  ", 3, 0),
            Vec::<(usize, usize)>::new()
        );
    }
}
