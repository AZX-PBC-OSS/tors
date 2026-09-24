//! Token-budget chunking: the pure-Rust cores of `tors.chunk_to_budget`
//! and `tors.chunk_to_offsets`, the chunking family's measured-budget
//! member. Where [`crate::chunk_impl::chunk_text`] budgets in codepoints
//! (a token proxy), this family budgets in the caller's own tokens: a
//! `max_tokens` measured by a caller-provided counter (`chunk_to_budget`)
//! or by pre-computed token offsets (`chunk_to_offsets`), the two shapes
//! an LLM context-window packer needs. The counter callable is the
//! measured exception to the stateless doctrine (the
//! `CompiledLemmaDict`-style caller-supplied-function lane: the packing
//! itself is native, the token measurement is whatever the caller's
//! tokenizer says), and the precomputed-offsets spelling is the fully
//! GIL-free twin for hot paths.
//!
//! Prior art, and what is borrowed from each: semchunk
//! (https://github.com/isaacus-dev/semchunk) packs whole semantic units
//! (sentences, falling back to words) greedily under a token budget
//! measured by a tokenizer callable, the same segment-then-pack shape
//! implemented here, with tors's own UAX #29 segmenters in place of
//! semchunk's regex split. LangChain's text splitters and LlamaIndex's
//! `SentenceSplitter` popularized the token-budget-with-overlap shape
//! this family ships (`overlap` repeating trailing context into the next
//! chunk). The study at https://arxiv.org/abs/2410.13070 ("Is Semantic
//! Chunking Worth the Computational Cost?", Qu/Tu/Bao 2024) found the
//! expensive semantic-chunking variants not consistently worth their
//! cost over simpler splitters: the reason this surface ships the
//! cheap mechanical contract (boundary-safe packing under an exact
//! token budget) and, like every chunker here, makes no
//! retrieval-quality promise (docs/design.md's own limitation).
//!
//! # The packing rule
//!
//! Segments are UAX #29 sentences ([`crate::segmentation_impl::sentence_bounds`]);
//! a sentence whose own measured count exceeds `max_tokens` is re-cut
//! at UAX #29 word boundaries (each word segment packed the same way),
//! and a single word segment still wider than the whole budget goes out
//! whole as an oversized chunk: a covering chunker cannot drop or
//! split below its own finest boundary, the same
//! correctness-over-the-budget stance [`crate::chunk_impl::chunk_text`]
//! takes for an oversized grapheme cluster. Segments tile
//! `[0, codepoint_len)` exactly, so chunk starts always exist and every
//! codepoint lands in (or under) some chunk.
//!
//! Greedy packing measures the CANDIDATE CHUNK (`text[start..end]` as
//! it would be emitted), not a sum of per-segment counts: token counts
//! are not additive across boundaries (a BPE-style tokenizer merges
//! across spaces; any counter may be non-monotone), and the one
//! invariant that must hold for every counter is per-chunk: each
//! emitted chunk measures `<= max_tokens` by the same measurement the
//! packing used. The cost is one counter call per packing decision
//! (O(segments) calls total: one per sentence up front plus one per
//! greedy extension, each measuring up to one chunk's worth of text).
//! It is the "batch several boundaries per callback" shape: a single call
//! measures a whole candidate span, never one boundary apiece.
//!
//! Sentence counts are measured once up front and cached (reused for
//! the chunk's forced-first measurement and the overlap walk-back's
//! last-segment step); sentence-level counts of 0 are rejected
//! (`BudgetError::Invalid`): a sentence measuring zero tokens makes
//! the budget contract meaningless (sub-sentence spans, the
//! whitespace runs between words, may measure 0 and are tolerated:
//! word-count tokenizers legitimately return 0 there).
//!
//! # Overlap
//!
//! After a chunk closes, the next chunk starts at the trailing segment
//! boundary whose cumulative span to the closed chunk's end measures at
//! least `overlap_tokens` tokens (the walk goes back from the last
//! segment; the first boundary that reaches the requested overlap wins,
//! so the repeated context is the shortest the counter certifies). The
//! overlap is declined for a transition (the next chunk starts at the
//! closed chunk's end, zero overlap) when it cannot buy new context,
//! never stall or emit a contained chunk: the walk only accepts
//! positions strictly inside the chunk (never before its own start),
//! and a chosen position whose own re-cut re-embeds the predecessor's
//! tail (ends at or before its end, possible with non-monotone or
//! tightly-budgeted counters) is discarded and re-cut from the
//! zero-overlap position. Forward progress is unconditional: every
//! accepted chunk starts strictly after its predecessor's start and
//! ends strictly after its predecessor's end, so the chunk count is
//! bounded by the codepoint count (pinned by tests both here and in
//! Python, Hypothesis included).
//!
//! # The two spellings
//!
//! [`chunk_to_budget`] takes the measurement as a closure over a
//! `&str` span (the Python binding's callback shape: the closure
//! re-attaches the GIL per call (see `src/py/chunk_budget.rs`) while
//! the packing between calls stays detached native work).
//! [`chunk_to_offsets`] takes pre-computed token spans
//! `(start, end)` codepoint pairs (sorted, non-overlapping; gaps
//! allowed: untokenized text such as whitespace between tokens simply
//! measures 0) and measures a span's token count as the number of
//! spans fully contained in it: additive, exact, allocation-free, and
//! callable with no Python at all, which is what makes the
//! `chunk_to_offsets` binding a single end-to-end `py.detach`.

use crate::segmentation_impl;
use crate::truncate_impl::char_count;

/// Why a packing call failed: the caller's counter raised (Python; the
/// binding re-raises the captured exception) or a measured count
/// violated the contract (a `ValueError` message).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetError {
    /// The caller's token counter raised. The binding holds the
    /// original `PyErr` alongside and re-raises it; this arm only
    /// tells the two failure classes apart.
    Raised,
    /// A contract violation with its `ValueError` message.
    Invalid(String),
    /// A contract violation with its `TypeError` message (the
    /// counter's return is not an int).
    WrongType(String),
}

impl BudgetError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        BudgetError::Invalid(message.into())
    }
}

/// The codepoint→byte grid's offset width: `pack`'s grid (below) stores
/// byte offsets as `u32`, so text beyond `u32::MAX` bytes would silently
/// truncate them. The Python bindings refuse such text with a clear
/// `ValueError` via [`grid_overflow`], factored out so the unit test
/// below can pin the boundary with a synthetic value, no 4 GiB
/// allocation.
pub(crate) fn grid_overflow(text_bytes: u64) -> bool {
    text_bytes > u32::MAX as u64
}

/// One packing segment: a UAX #29 sentence that fits the budget
/// individually (`cached` holds its phase-A count, reused for the forced
/// first measurement of a chunk and the overlap walk-back's last step;
/// both measure exactly this segment's span, the same measurement phase
/// A already paid), or one of the word-boundary segments an oversized
/// sentence was re-cut into (`cached: None`; word segments may
/// individually exceed the budget and go out whole as oversized chunks;
/// sentence-level ones can't, phase A replaces every one of those).
struct Seg {
    start: usize,
    end: usize,
    cached: Option<u64>,
}

/// [`chunk_to_budget`]'s packing core shared with the offsets spelling:
/// `measure` answers "how many tokens in this candidate span?" for one
/// candidate span, given both its codepoint bounds `(start, end)` and
/// the text it covers (`span`, resolved through the codepoint→byte grid
/// below; offsets are codepoint indices, Rust slices are bytes).
/// `reject_zero_sentences` is the callback spelling's contract (a
/// sentence measuring 0 tokens is a broken counter,
/// `BudgetError::Invalid`); the offsets spelling passes `false`
/// (untokenized text legitimately measures 0).
fn pack(
    text: &str,
    max_tokens: u64,
    overlap_tokens: u64,
    reject_zero_sentences: bool,
    mut measure: impl FnMut(usize, usize, &str) -> Result<u64, BudgetError>,
) -> Result<Vec<(usize, usize)>, BudgetError> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    // The codepoint→byte grid: `grid[i]` is the byte offset of codepoint
    // `i`, with the total byte length as the final entry, so a span's
    // bytes are `grid[s]..grid[e]`. Pure-ASCII text (the common case)
    // skips the grid entirely: bytes are codepoints there, the same
    // fast path `truncate_impl::char_count` takes. Every slice the
    // packing performs (measure spans, the word-boundary fallback's
    // re-segmentation) resolves through this one grid, so a codepoint
    // offset can never drift into a byte slice.
    let grid: Option<Vec<u32>> = if text.is_ascii() {
        None
    } else {
        Some(
            text.char_indices()
                .map(|(byte, _)| byte as u32)
                .chain([text.len() as u32])
                .collect(),
        )
    };
    let slice = |s: usize, e: usize| -> &str {
        match &grid {
            None => &text[s..e],
            Some(grid) => &text[grid[s] as usize..grid[e] as usize],
        }
    };
    // Phase A: sentence segments with cached counts; an oversized
    // sentence is re-cut at word boundaries in place (its word
    // segments replace it in the packing list). One measurement per
    // sentence, paid exactly once.
    let mut segs: Vec<Seg> = Vec::new();
    for (s, e) in segmentation_impl::sentence_bounds(text) {
        let count = measure(s, e, slice(s, e))?;
        if count == 0 && reject_zero_sentences {
            return Err(BudgetError::invalid(format!(
                "token_counter returned 0 for a non-empty sentence (codepoints {s}..{e}): \
                 every sentence must measure at least one token"
            )));
        }
        if count > max_tokens {
            // Word-boundary fallback: this sentence alone exceeds the
            // budget, so its word segments are packed individually
            // (whitespace runs between them are their own segments and
            // ride along inside chunk spans, exactly as
            // chunk_text's trim rides them onto the next chunk).
            for (ws, we) in segmentation_impl::word_bounds(slice(s, e)) {
                segs.push(Seg {
                    start: s + ws,
                    end: s + we,
                    cached: None,
                });
            }
        } else {
            segs.push(Seg {
                start: s,
                end: e,
                cached: Some(count),
            });
        }
    }

    let total = char_count(text);
    let mut chunks: Vec<(usize, usize)> = Vec::new();
    // The segment index the next chunk starts from; the zero-overlap
    // fallback position for the current transition (the segment after
    // the last one the just-closed chunk included).
    let mut si = 0usize;
    let mut fallback_si = 0usize;
    let mut prev_end = 0usize;
    while si < segs.len() {
        let start = segs[si].start;
        // The first segment is forced in (a chunk is never empty);
        // sentence-level segments reuse their phase-A count.
        let mut last = si;
        let mut end = segs[si].end;
        let first_count = match segs[si].cached {
            Some(count) => count,
            None => measure(start, end, slice(start, end))?,
        };
        if first_count <= max_tokens {
            // Greedy extension: keep appending whole segments while the
            // CANDIDATE CHUNK still fits the budget (measured, never
            // summed: the emitted chunk's measurement is the packing's
            // own).
            while last + 1 < segs.len() {
                let candidate = segs[last + 1].end;
                if measure(start, candidate, slice(start, candidate))? > max_tokens {
                    break;
                }
                last += 1;
                end = candidate;
            }
        }
        // `first_count > max_tokens` falls through with last == si: a
        // single segment (word-level; sentence-level can't get here)
        // wider than the whole budget goes out whole, the oversized-
        // chunk exception documented on the module.
        if !chunks.is_empty() && end <= prev_end {
            // The overlap walk-back chose a start whose own chunk
            // re-embeds this predecessor's tail without advancing past
            // it (the same-text-twice shape `chunk_text_overlapping`
            // rejects): decline the overlap for this
            // transition and re-cut from the zero-overlap position,
            // which always ends strictly past `prev_end` (it starts
            // there). Progress unconditional; at most one re-cut per
            // transition.
            si = fallback_si;
            continue;
        }
        chunks.push((start, end));
        prev_end = end;
        if end >= total {
            break;
        }
        // Choose the next chunk's start: walk back from the last
        // segment of the closed chunk to the first boundary whose
        // cumulative span to the chunk's end measures at least
        // `overlap_tokens` (the shortest context the counter certifies;
        // the last segment's step reuses its cached sentence count).
        // The walk never reaches this chunk's own start (j > si), so an
        // accepted overlap start is strictly inside the chunk. If the
        // reached count exceeds the budget (the overlap region alone
        // would not fit), the overlap is declined for this transition.
        let mut next = last + 1;
        if overlap_tokens > 0 {
            for j in (si + 1..=last).rev() {
                let count = match j == last {
                    true => match segs[j].cached {
                        Some(count) => count,
                        None => measure(segs[j].start, end, slice(segs[j].start, end))?,
                    },
                    false => measure(segs[j].start, end, slice(segs[j].start, end))?,
                };
                if count >= overlap_tokens {
                    if count <= max_tokens {
                        next = j;
                    }
                    break;
                }
            }
        }
        fallback_si = last + 1;
        si = next;
    }
    Ok(chunks)
}

/// Callback-spelling packing core: `count` measures one candidate
/// `&str` span of `text`. Sentence counts of 0 are rejected (a sentence
/// measuring no tokens is a broken counter); sub-sentence spans may
/// measure 0. See the module docs for the packing rule and the GIL
/// contract the Python binding layers on top.
pub fn chunk_to_budget(
    text: &str,
    max_tokens: u64,
    overlap_tokens: u64,
    mut count: impl FnMut(&str) -> Result<u64, BudgetError>,
) -> Result<Vec<(usize, usize)>, BudgetError> {
    pack(
        text,
        max_tokens,
        overlap_tokens,
        true,
        |_start, _end, span| count(span),
    )
}

/// Precomputed-offsets packing core: `token_spans` are the tokens'
/// `(start, end)` codepoint spans in `text`, sorted and non-overlapping
/// (gaps allowed); a span's token count is the number of token spans
/// fully contained in it (two binary searches, no allocation, no
/// callback, the GIL-free spelling's whole point). Infallible for
/// sorted input; the binding validates sortedness before detaching.
pub fn chunk_to_offsets(
    text: &str,
    token_spans: &[(usize, usize)],
    max_tokens: u64,
    overlap_tokens: u64,
) -> Result<Vec<(usize, usize)>, BudgetError> {
    pack(
        text,
        max_tokens,
        overlap_tokens,
        false,
        |start, end, _span| {
            let base = token_spans.partition_point(|&span| span.0 < start);
            let top = token_spans.partition_point(|&span| span.1 <= end);
            Ok(top.saturating_sub(base) as u64)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_segmentation::UnicodeSegmentation;

    /// A split-based counter: `span.split_whitespace().count()`, the
    /// word-count token proxy the Python tests also use (no third-party
    /// tokenizer needed).
    fn words(span: &str) -> Result<u64, BudgetError> {
        Ok(span.split_whitespace().count() as u64)
    }

    #[test]
    fn grid_overflow_boundary_is_u32_max_bytes() {
        // The synthetic-bound test for the bindings' text guard: the
        // codepoint→byte grid's u32 offset width, pinned at the boundary
        // without allocating a 4 GiB string.
        assert!(!grid_overflow(u32::MAX as u64));
        assert!(grid_overflow(u32::MAX as u64 + 1));
    }

    #[test]
    fn text_fitting_the_budget_is_one_chunk() {
        let text = "One. Two. Three.";
        let chunks = chunk_to_budget(text, 10, 0, words).unwrap();
        assert_eq!(chunks, [(0, 16)]);
    }

    #[test]
    fn empty_text_yields_no_chunks() {
        assert_eq!(
            chunk_to_budget("", 5, 0, words).unwrap(),
            Vec::<(usize, usize)>::new()
        );
        assert_eq!(
            chunk_to_offsets("", &[(0, 1)], 5, 0).unwrap(),
            Vec::<(usize, usize)>::new()
        );
    }

    #[test]
    fn sentence_cuts_pack_greedily_under_the_budget() {
        // sentence_bounds("One. Two. Three.") ends: 5, 10, 16 (each
        // sentence's trailing space rides it). Counts: 1, 2, 2 words;
        // budget 2 packs [One.] with [Two.] (2), then [Three.].
        let text = "One. Two. Three.";
        let chunks = chunk_to_budget(text, 2, 0, words).unwrap();
        assert_eq!(chunks, [(0, 10), (10, 16)]);
        let slice = |&(a, b): &(usize, usize)| &text[a..b];
        // Each sentence's trailing space rides its own chunk's tail
        // (this packer does not trim; the budget measured the text as
        // it is).
        assert_eq!(slice(&chunks[0]), "One. Two. ");
        assert_eq!(slice(&chunks[1]), "Three.");
    }

    #[test]
    fn an_oversized_sentence_falls_back_to_word_boundaries() {
        // One 30-codepoint terminator-free "sentence" of 6 words:
        // budget 2 (2 words per chunk) must cut it at word boundaries,
        // not emit one oversized chunk.
        let text = "aa bb cc dd ee ff";
        let chunks = chunk_to_budget(text, 2, 0, words).unwrap();
        assert_eq!(chunks, [(0, 6), (6, 12), (12, 17)]);
        assert_eq!(&text[0..6], "aa bb ");
    }

    #[test]
    fn a_single_word_wider_than_the_budget_goes_out_whole() {
        // "aaaa" (1 word, 4 codepoints): the word-fallback's finest
        // boundary can't split it; budget 1 packs it whole as the
        // documented oversized exception.
        let text = "aaaa bb";
        let chunks = chunk_to_budget(text, 1, 0, words).unwrap();
        // The space between the words rides chunk 1's tail (measured
        // free at 1 word); "bb" is chunk 2.
        assert_eq!(chunks, [(0, 5), (5, 7)]);
        assert_eq!(&text[0..5], "aaaa ");
        assert_eq!(&text[5..7], "bb");
    }

    #[test]
    fn a_zero_sentence_count_is_rejected() {
        let err = chunk_to_budget("hello world", 5, 0, |_| Ok(0)).unwrap_err();
        assert!(
            matches!(err, BudgetError::Invalid(ref m) if m.contains("returned 0")),
            "{err:?}"
        );
    }

    #[test]
    fn a_raised_error_propagates() {
        let err = chunk_to_budget("hello", 5, 0, |_| Err(BudgetError::Raised)).unwrap_err();
        assert_eq!(err, BudgetError::Raised);
    }

    #[test]
    fn a_non_monotone_counter_terminates_and_keeps_chunks_in_bounds() {
        // The adversarial shape: counts keyed to the span's first
        // character, unrelated to its length. Progress is anchored to
        // codepoint offsets, never to counts, so this terminates with
        // in-bounds, strictly-advancing, covering chunks.
        let text = format!("{} {}", "a".repeat(200), "b".repeat(200));
        let flare = |span: &str| -> Result<u64, BudgetError> {
            Ok(match span.as_bytes().first() {
                Some(&b'a') => 50,
                Some(&b'b') => 1,
                _ => 1,
            })
        };
        let chunks = chunk_to_budget(&text, 60, 0, flare).unwrap();
        assert!(!chunks.is_empty());
        let mut prev_start = 0usize;
        for &(start, end) in &chunks {
            assert!(start >= prev_start);
            assert!(end > start);
            assert!(end <= text.len());
            prev_start = start;
        }
        assert_eq!(chunks.last().unwrap().1, text.len());
    }

    #[test]
    fn offsets_variant_counts_contained_spans() {
        // Token spans for "ab cd ef" (whitespace untokenized: gaps):
        // (0,2), (3,5), (6,8). Budget 2 tokens packs all of it (3
        // tokens? No: 3 > 2): [ab cd] (2 tokens) then [ef].
        let text = "ab cd ef";
        let spans = [(0, 2), (3, 5), (6, 8)];
        let chunks = chunk_to_offsets(text, &spans, 2, 0).unwrap();
        assert_eq!(chunks, [(0, 6), (6, 8)]);
        assert_eq!(&text[0..6], "ab cd ");
        assert_eq!(&text[6..8], "ef");
    }

    #[test]
    fn offsets_variant_tolerates_straddling_and_untokenized_text() {
        // A token span straddling a would-be chunk boundary is never
        // counted on either side of the cut; wholly-untokenized text
        // measures 0 and still makes progress (offsets, not counts,
        // anchor the walk).
        let text = "abcd";
        let spans = [(0, 4)]; // one token covering everything
        let chunks = chunk_to_offsets(text, &spans, 1, 0).unwrap();
        // The single token can't fit any strict sub-budget: it goes out
        // whole (its containing chunk is the whole text).
        assert_eq!(chunks, [(0, 4)]);
        let chunks = chunk_to_offsets("   ", &[], 1, 0).unwrap();
        assert_eq!(chunks, [(0, 3)]);
    }

    #[test]
    fn overlap_repeats_trailing_context() {
        // 6 sentences of 2 words each ("One Two." ...): budget 2 packs
        // one sentence per chunk; overlap 1 token must start the next
        // chunk at the last sentence boundary (1 repeated token).
        let text = "aa bb. cc dd. ee ff.";
        let chunks = chunk_to_budget(text, 2, 1, words).unwrap();
        assert!(chunks.len() >= 2, "{chunks:?}");
        for window in chunks.windows(2) {
            let (prev_start, prev_end) = window[0];
            let (next_start, _) = window[1];
            assert!(next_start > prev_start, "no forward progress: {chunks:?}");
            if next_start < prev_end {
                // The repeated context must actually match on both sides.
                assert!(!text[next_start..prev_end].trim().is_empty());
            }
        }
    }

    #[test]
    fn overlap_never_produces_a_chunk_contained_in_its_predecessor() {
        // The same-text-twice shape for the budget packer: a chunk shorter than the
        // requested overlap (or a re-cut that lands inside the
        // predecessor) degrades to zero overlap for that transition
        // rather than emit the same span twice.
        let text = "aa bb. cc dd. ee ff.";
        for budget in 1..=4 {
            for overlap in 1..budget {
                let chunks = chunk_to_budget(text, budget, overlap, words).unwrap();
                let mut prev: Option<(usize, usize)> = None;
                for &(start, end) in &chunks {
                    assert!(end > start);
                    if let Some((ps, pe)) = prev {
                        assert!(start > ps, "starts must advance: {chunks:?}");
                        assert!(end > pe, "ends must advance: {chunks:?}");
                        assert!((start, end) != (ps + 1, pe), "contained chunk: {chunks:?}");
                    }
                    prev = Some((start, end));
                }
                assert_eq!(prev.unwrap().1, text.len(), "must cover to the end");
            }
        }
    }

    #[test]
    fn chunk_count_is_bounded_by_the_codepoint_count() {
        // The forward-progress guarantee as a hard bound, the
        // chunk_text_overlapping pin's twin.
        let text = format!("{} {}", "a".repeat(500), "b".repeat(500));
        let total = text.chars().count();
        for budget in [1usize, 2, 5, 50] {
            for overlap in 0..budget {
                let chunks = chunk_to_budget(&text, budget as u64, overlap as u64, words).unwrap();
                assert!(
                    chunks.len() <= total,
                    "chunk count {} exceeded codepoint count {total}",
                    chunks.len()
                );
            }
        }
    }

    #[test]
    fn offsets_variant_overlap_walks_back_by_contained_tokens() {
        let text = "aa bb cc dd ee";
        let spans = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14)];
        // Budget 2 tokens = 1 token per chunk with overlap 1: each next
        // chunk re-includes exactly one trailing token.
        let chunks = chunk_to_offsets(text, &spans, 2, 1).unwrap();
        assert!(chunks.len() >= 2, "{chunks:?}");
        for window in chunks.windows(2) {
            let (next_start, prev_end) = (window[1].0, window[0].1);
            assert!(next_start < prev_end, "no actual overlap: {chunks:?}");
            // The shared region holds at least one whole token span.
            let shared = spans
                .iter()
                .filter(|&&(s, e)| s >= next_start && e <= prev_end)
                .count();
            assert!(shared >= 1, "{chunks:?}");
        }
    }

    /// The full contract over one (text, budget, overlap) case: chunks
    /// non-empty, in bounds, starts and ends strictly advancing, first
    /// start 0, last end the codepoint length, and (zero-overlap) a
    /// contiguous covering partition that joins back to the input.
    fn assert_contract(text: &str, budget: u64, overlap: u64, count: impl Fn(&str) -> u64) {
        let grid: Vec<usize> = text
            .char_indices()
            .map(|(byte, _)| byte)
            .chain([text.len()])
            .collect();
        let slice_cp = |a: usize, b: usize| &text[grid[a]..grid[b]];
        let chunks = match chunk_to_budget(text, budget, overlap, |span| {
            Ok::<u64, BudgetError>(count(span))
        }) {
            // The documented zero-sentence contract: a sentence
            // measuring 0 tokens (a whitespace-only sentence under the
            // word counter) raises before anything is packed.
            Err(BudgetError::Invalid(message)) => {
                assert!(
                    message.contains("returned 0"),
                    "unexpected packing error for {text:?}/{budget}/{overlap}: {message}"
                );
                // The error fires on the first measured sentence, so
                // that sentence must be the 0-count one.
                let (first_start, first_end) = segmentation_impl::sentence_bounds(text)[0];
                assert_eq!(
                    count(slice_cp(first_start, first_end)),
                    0,
                    "0-count error for text whose first sentence measures tokens: {text:?}"
                );
                return;
            }
            Err(err) => panic!("unexpected packing error for {text:?}: {err:?}"),
            Ok(chunks) => chunks,
        };
        if text.is_empty() {
            assert_eq!(chunks, Vec::<(usize, usize)>::new());
            return;
        }
        assert!(!chunks.is_empty());
        let mut prev_start = 0usize;
        let mut prev_end = 0usize;
        for (i, &(start, end)) in chunks.iter().enumerate() {
            assert!(end > start, "empty chunk: {chunks:?}");
            assert!(end <= text.chars().count(), "out of bounds: {chunks:?}");
            if i == 0 {
                assert_eq!(start, 0);
            } else {
                assert!(start > prev_start, "starts must advance: {chunks:?}");
                assert!(end > prev_end, "ends must advance: {chunks:?}");
                if overlap == 0 {
                    assert_eq!(start, prev_end, "zero-overlap must be contiguous");
                }
            }
            // Every chunk individually fits the budget with the same
            // counter, except the documented single-oversized-segment
            // exception (the chunk is exactly one segment).
            if count(slice_cp(start, end)) > budget {
                // A single segment: word-level granularity means the
                // chunk contains no boundary to cut at that the packer
                // didn't already try; assert it is one unsplittable
                // unit (no interior word boundary).
                let interior_words = slice_cp(start, end).split_word_bound_indices().count();
                assert_eq!(
                    interior_words, 1,
                    "budget exceeded without the single-segment exception: {chunks:?} text={text:?}"
                );
            }
            prev_start = start;
            prev_end = end;
        }
        assert_eq!(prev_end, text.chars().count(), "must cover to the end");
        if overlap == 0 {
            let joined: String = chunks
                .iter()
                .map(|&(a, b)| slice_cp(a, b))
                .collect::<Vec<_>>()
                .join("");
            assert_eq!(joined, text, "zero-overlap join-back broke: {chunks:?}");
        }
    }

    #[test]
    fn properties_hold_over_the_exhaustive_small_alphabet() {
        // Every string over {a, space, .} up to length 5 x budgets
        // 1..=3 x overlap 0..=2: the alphabet puts whitespace runs and
        // sentence terminators (the word-fallback and overlap paths)
        // inside the exhaustive space.
        let alphabet = ['a', ' ', '.'];
        let mut texts = vec![String::new()];
        for _ in 0..5 {
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
            for budget in 1..=3u64 {
                for overlap in 0..=2u64.min(budget - 1) {
                    assert_contract(text, budget, overlap, |span| {
                        span.split_whitespace().count() as u64
                    });
                }
            }
        }
    }

    #[test]
    fn properties_hold_on_a_mixed_battery() {
        // Non-ASCII rows (CJK, emoji ZWJ, combining marks): offsets are
        // codepoints, and the round-trip text[start:end] == chunk must
        // hold through multibyte characters (a byte-slice of a
        // codepoint offset would panic or land mid-character; the
        // char-grid slicing here is the Python str[a:b] equivalent the
        // contract promises).
        let cases = [
            "hello world",
            "One. Two. Three.",
            "caf\u{e9} \u{6771}\u{4eac}\u{3002} \u{5927}\u{962a}\u{3002}",
            "\u{1f469}\u{200d}\u{1f52c} says hi. \u{1100}\u{1161}\u{11a8}!",
            "a\u{0301}b. \u{0e01}\u{0e33}\u{0e19}\u{0e14}.",
            "aaaa    bbbb",
            "  leading and trailing  ",
        ];
        let slice_cp = |text: &str, a: usize, b: usize| {
            text.chars()
                .enumerate()
                .filter(|(i, _)| *i >= a && *i < b)
                .map(|(_, c)| c)
                .collect::<String>()
        };
        for case in &cases {
            let cp = |span: &str| span.chars().count() as u64;
            for budget in 1..=4u64 {
                for overlap in 0..budget {
                    assert_contract(case, budget, overlap, |span| {
                        span.split_whitespace().count() as u64
                    });
                    assert_contract(case, budget, overlap, cp);
                    // Round-trip: each chunk slice is exactly the
                    // codepoints between its offsets (char-grid slicing,
                    // the str[a:b] equivalent).
                    let chunks =
                        chunk_to_budget(case, budget, overlap, |span: &str| Ok(cp(span))).unwrap();
                    for &(a, b) in &chunks {
                        assert!(!slice_cp(case, a, b).is_empty());
                    }
                }
            }
        }
    }
}
