//! Hierarchical fallback chunking: the pure-Rust core of
//! `tors.chunk_hierarchical`, the `chunk_text`/`chunk_by_*` family's fourth
//! shape: a priority-ordered list of separator levels, coarsest first,
//! tried in order for each chunk: the same pattern LangChain's
//! `RecursiveCharacterTextSplitter` popularized (default separators
//! `["\n\n", "\n", " ", ""]`, literal strings, falling back to the next
//! level only when the coarser one has no in-budget cut).
//!
//! Split into its own file rather than folded into [`crate::chunk_impl`] or
//! [`crate::chunk_by_segment_impl`]: this is a different
//! algorithm from both siblings, a multi-level search per chunk, not a
//! single boundary list, and it earns its own "one concern per file" home
//! the same separation unit-count chunking already has.
//!
//! Default hierarchy (`separators = None`): paragraph → sentence → word →
//! grapheme-safe raw cut, reusing tors's own accurate segmenters
//! ([`crate::chunk_by_segment_impl::paragraph_bounds`],
//! [`crate::segmentation_impl::sentence_bounds`],
//! [`crate::segmentation_impl::word_bounds`]) rather than the naive literal
//! guesses (`"\n\n"`, `". "`, `" "`) LangChain's own default falls back to:
//! tors already has real UAX #29 segmentation, so the default hierarchy
//! uses it.
//!
//! Custom hierarchy (`separators = Some(list)`): a caller-supplied list of
//! literal strings (not regex, a documented scope line: literals are
//! LangChain's own default too, cover the motivating markdown-header case
//! completely, and avoid reopening the regex-semantics question this crate
//! already navigated once for `re`). Replaces the default hierarchy
//! entirely for the levels it specifies. The grapheme-safe raw cut is
//! always appended as an implicit final level regardless of what the
//! caller passes: "never fails to produce a chunk" is an unconditional
//! guarantee here, not something a caller can accidentally break by
//! forgetting a sentinel (LangChain's own convention requires an explicit
//! trailing `""`; tors does not require this, and ignores a trailing `""`
//! if one is passed, since the raw cut already covers that case).
//!
//! A `None` entry in an otherwise-literal list splices the default
//! hierarchy's three accurate levels in at that position: the mix the
//! all-or-nothing custom list could not express before. The motivating
//! shape is line-oriented text that must never split mid-line but whose
//! oversized lines still deserve accurate fallback cuts: a chat thread,
//! one message per line: `["\n", None]` is line → paragraph → sentence →
//! word → raw cut, where the sentence/word levels below the line level
//! are the real UAX #29 segmenters, not the `". "`/`" "` literal guesses
//! an all-literal `["\n", ". ", " "]` list would pin them to (the guesses
//! cut inside "U.S. team"; the segmenters do not: the exact reason the
//! default hierarchy exists). `[None]` is therefore identical to
//! `separators = None`. Cost: every level (each of the three default
//! walks, each distinct custom literal) pays its one whole-text walk at
//! most once per call, and only when a window
//! actually consults it (#30's lazy levels: the splice contributes
//! three level specs, not three whole-text walks, a literal's slot
//! waits unbuilt the same way, and realization happens on first
//! consultation, the same at-most-once-per-call memoization #24's
//! lazy bitmap applies), so `["\n", None]` over a thread whose every
//! line fits the budget builds none of the spliced levels, and one
//! whose oversized lines fall to the sentence level pays sentence's
//! walk once and word's never. Duplicate entries (a second or later
//! `None`, a repeated literal) are recognized at slot construction
//! and skipped, inert by the `find_map` dominance argument (identical
//! levels can never change the answer: the first occurrence of a level
//! always dominates its duplicate), so neither `[None; 100]` nor
//! `[" "; 100]` is the caller-controlled unbounded cost the per-entry
//! spelling made them (every duplicate re-paid the walks plus a full
//! cut vector, ~45 MiB per duplicate on a 6 MiB document: an OOM
//! shape); the dedups close both spellings under the #21
//! pathological-input discipline, memoized like #24's lazy bitmap. The
//! former spellings were eager on both axes on top of that: the
//! pre-#31 one re-paid all three walks per `None` entry outright, and
//! #31's dedup still paid the three walks once per call even when no
//! window ever consulted them (a single-chunk budget over 6 MiB spent
//! ~176 ms and ~45 MiB on levels that supplied zero cuts); the lazy
//! realization removes the paid-for-nothing cost.
//!
//! Unlike [`crate::chunk_impl::chunk_text`] (a lossless covering
//! partition), this is not lossless: at every level except the raw-cut
//! fallback, the separator itself is dropped between chunks (the chunk
//! ends where the separator starts, the next chunk begins where it ends),
//! the same convention [`crate::chunk_by_segment_impl::chunk_by_paragraphs`]
//! already established for blank-line runs. Joining the chunks does not
//! reproduce the input; that was already true of the unit-count family and
//! stays true here for the same reason (a caller asking to split on a
//! marker wants it gone, not duplicated).
//!
//! Performance: every consulted level's candidate cut-position list is
//! computed at most once per call, at the first window that consults it
//! (#30's lazy levels: one scan per distinct level (`paragraph_bounds`/
//! `sentence_bounds`/`word_bounds` for the default levels, one `memmem`
//! pass per custom literal), never re-scanned per chunk, never scanned
//! at all for a level no window descends to, and never repeated for a
//! duplicate entry the slot-construction dedup already collapsed; a
//! single-chunk budget, `max_chars >= total`, consults no level and
//! walks nothing: one codepoint count, one chunk out). Building each chunk is one
//! `partition_point` binary search per level: O(n × levels) total, levels
//! bounded by the small, caller-supplied list length. The codepoint `total`
//! the budget arithmetic needs is [`char_count`]'s one ASCII-first pass
//! (pure-ASCII text answers its own byte length; non-ASCII text pays the
//! branchless non-continuation-byte count ([`char_count`]'s own docs carry
//! why the split exists), not a materialized `Vec<char>`. The one other
//! whole-text structure is the grapheme boundary
//! index (see [`GraphemeIndex`]): one `graphemes(true)` walk emitting one
//! bit per codepoint on non-ASCII text, or, on pure-ASCII text, an
//! all-ones bitmap plus the CRLF fixup, two SIMD byte scans, no
//! segmentation walk at all. It is built at most once per call and only
//! when a call actually needs it: a realized level with cuts to filter,
//! the raw-cut fallback, or an overlap snap. A call that realizes no level
//! with cuts and never falls back or snaps (the whole-document-budget
//! case, custom hierarchy or default) builds none of it. The former
//! spelling paid an unconditional `Vec<char>` collect plus a
//! `HashSet<usize>` of every grapheme boundary in the document before
//! anything else could run, which dominated document-scale cost and was
//! superlinear on top of it (#22: a 12 MiB document spent ~1.2 s in
//! ~12.6M hashed inserts regardless of chunk budget or separator
//! presence); until #30, every level was additionally built, and every
//! non-empty one filtered, before the first window, so a budget that
//! never consulted them paid for them anyway. Forward progress is pinned
//! the same way [`crate::chunk_impl::chunk_text_overlapping`]'s is: a hard
//! iteration-count assertion in the tests, not just a slow-test timeout.

use std::collections::HashSet;

use memchr::memmem;

use crate::chunk_by_segment_impl::paragraph_bounds;
use crate::segmentation_impl;
use crate::truncate_impl::{GraphemeIndex, char_count};

/// One level's candidate cut points, ascending by `cut_end`. `next_start >=
/// cut_end` always: equal for contiguous segmenters (word/sentence bounds,
/// where there is no gap to drop), strictly greater when a separator's own
/// content is dropped between chunks (paragraph gaps, custom literal
/// separators).
struct Level {
    cuts: Vec<(usize, usize)>,
}

/// One window's outcome from the level search: a genuine cut (the chunk
/// `(start, cut_end)` is emitted, the window resumes at `next_start`), or
/// a separator match at the window's own start (no chunk; the window
/// resumes at the separator's end; the skip, see the loop body). The
/// explicit enum, not an encoded `(cut_end, next_start)` shape, because
/// the skip's cut_end IS the window's own start: an empty chunk no
/// genuine cut can ever produce (`best_cut` requires `cut_end > after`),
/// so the encoding is unambiguous, but the variant name says what the
/// branch means, and the match below is where the two verdicts diverge.
enum Verdict {
    Cut(usize, usize),
    Skip(usize),
}

impl Level {
    /// The largest `(cut_end, next_start)` with `cut_end <= limit` and
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

    /// A separator match beginning exactly at `at`: a cut
    /// `(at, next_start)` with `next_start > at`, as the resume position
    /// past it, `None` when this level has no such cut. Only
    /// separator-dropping levels can answer (`next_start > cut_end` holds
    /// strictly there); a contiguous level's cuts have
    /// `next_start == cut_end`, and a segment end at `at` would be an
    /// empty segment that no contiguous segmenter produces, so a `Some`
    /// from here is exactly the "this window opens on a separator" case
    /// the window loop skips before it ever searches for a cut.
    fn skip_cut(&self, at: usize) -> Option<usize> {
        let lo = self.cuts.partition_point(|&(end, _)| end < at);
        match self.cuts.get(lo) {
            Some(&(end, next)) if end == at && next > at => Some(next),
            _ => None,
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
/// consecutive paragraph segments are not contiguous (the blank-line run
/// between them is excluded from both): the cut is the current
/// paragraph's `end`, the next chunk resumes at the following paragraph's
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
/// One `memchr::memmem` pass over the whole text (SIMD-skipped two-way:
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
    // Every match of a literal needle is the needle: its char length is a
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

/// A not-yet-built level: which walk or scan realizes it, not the walk
/// itself. The default hierarchy's three levels and a `None` splice's
/// copies exist as specs until the first window that consults them;
/// a custom literal exists as a spec until the same first
/// consultation. Realization (one whole-text pass: `paragraph_bounds`,
/// UAX #29 `sentence_bounds`, UAX #29 `word_bounds`, or one `memmem`
/// scan for [`LevelSpec::Literal`]) happens in [`LevelSlot::realize`],
/// never here; building a spec list walks no text at all.
#[derive(Clone, Copy)]
enum LevelSpec<'a> {
    /// The blank-line paragraph walk, gaps dropped between chunks.
    Paragraph,
    /// The UAX #29 sentence walk, contiguous.
    Sentence,
    /// The UAX #29 word walk, contiguous.
    Word,
    /// One `memchr::memmem` scan for this caller-supplied literal.
    Literal(&'a str),
}

/// One hierarchy position: the spec waiting to be realized, plus the
/// memoized [`Level`] once a window has consulted it. `level` stays
/// `None` until the first `best_cut` consultation and holds the build
/// for the rest of the call, the same at-most-once-per-call
/// memoization #24's lazy grapheme bitmap applies, extended (#30) from
/// the bitmap to the levels themselves: a level no window ever
/// descends to is never walked, never filtered, never allocated, which
/// is what removes the eager spelling's paid-for-nothing cost (a
/// single-chunk budget over 6 MiB spent ~176 ms and ~45 MiB building
/// three levels that supplied zero cuts).
struct LevelSlot<'a> {
    spec: LevelSpec<'a>,
    level: Option<Level>,
}

impl LevelSlot<'_> {
    /// The slot's built [`Level`], realizing the spec on first
    /// consultation and memoizing the build for every later window:
    /// at most one realization per slot per call (`get_or_insert_with`
    /// on the slot, the same memoization idiom #24's lazy grapheme
    /// bitmap uses, extended to the levels themselves). Realization is
    /// the level's one whole-text walk or scan, then, for a level with
    /// cuts, the grapheme cut filter, exactly the filter semantics the
    /// former pre-loop pass applied, just paid at build time, so a
    /// level never consulted never builds, never filters, and never
    /// triggers the grapheme bitmap.
    fn realize(
        &mut self,
        text: &str,
        total: usize,
        graphemes: &mut Option<GraphemeIndex>,
    ) -> &Level {
        self.level.get_or_insert_with(|| {
            let mut level = match self.spec {
                LevelSpec::Paragraph => level_from_paragraph_bounds(paragraph_bounds(text)),
                LevelSpec::Sentence => {
                    level_from_contiguous_bounds(segmentation_impl::sentence_bounds(text))
                }
                LevelSpec::Word => {
                    level_from_contiguous_bounds(segmentation_impl::word_bounds(text))
                }
                LevelSpec::Literal(sep) => level_from_literal(text, sep),
            };
            // The realized level's cuts are filtered to grapheme-cluster
            // boundaries, one O(cuts) bitmap pass: the same "never split a
            // cluster" fix chunk_text/chunk_by_* already apply, extended
            // here to custom literal separators too (a caller's separator
            // could, in principle, land inside a cluster on pathological
            // input). The default hierarchy is not exempt:
            // `word_bounds`/`sentence_bounds` follow UAX #29 exactly, which
            // scores some combining sequences (Thai SARA AM, U+0E33) as
            // their own word/sentence segment even though the grapheme
            // rules join them to the preceding base character into one
            // cluster (`truncate_impl`'s module docs document the same
            // divergence), so their cuts need this filter exactly like a
            // custom literal's do. A level with no cuts filters nothing
            // and builds nothing (a never-matching separator costs one
            // scan, no structure).
            if !level.cuts.is_empty() {
                let g = grapheme_index(graphemes, text, total);
                level
                    .cuts
                    .retain(|&(end, next)| g.is_boundary(end) && g.is_boundary(next));
            }
            #[cfg(test)]
            build_seam::bump_levels();
            level
        })
    }
}

/// The default hierarchy's three specs, in coarsest-first order: the
/// splice a `None` entry in a custom `separators` list inserts at its
/// position, and the whole hierarchy when `separators` is `None`.
/// Specs, not levels: no text is walked here, which is the point; the
/// walks happen in [`LevelSlot::realize`], only for slots a window
/// actually consults, at most once per call each.
fn default_level_specs<'a>() -> Vec<LevelSlot<'a>> {
    vec![
        LevelSlot {
            spec: LevelSpec::Paragraph,
            level: None,
        },
        LevelSlot {
            spec: LevelSpec::Sentence,
            level: None,
        },
        LevelSlot {
            spec: LevelSpec::Word,
            level: None,
        },
    ]
}

/// The caller-supplied hierarchy as a spec list: a literal `Some(s)` is
/// one [`LevelSpec::Literal`] waiting for its scan, one slot per
/// distinct literal, a repeated literal skipped as a duplicate the same
/// way a second `None` is (see the arms below for why a duplicate is
/// provably inert); a `None` entry splices [`default_level_specs`] in
/// at its position, at most once per call. An empty literal is a no-op
/// level (dropped, no slot at all, the same filter the all-literal
/// spelling always applied, and no seen-set entry either, since it
/// never produces a slot). Both dedups are provably inert by the same
/// `find_map` dominance argument: the window loop consults `slots`
/// only through the find_map (the first slot supplying a cut wins, and
/// the list is never indexed positionally or counted), and a
/// duplicate's level is bit-identical to its original's, so a
/// duplicate can never change an answer its original didn't already
/// give. Before the dedups, every duplicate was a re-paid whole-text
/// walk plus a full cut vector (~45 MiB per duplicate `None` on a
/// 6 MiB document, the same for a repeated literal that matches often
/// (`[" "; 100]` was the OOM shape `[None; 100]` was)), a
/// caller-controlled unbounded cost the #21 pathological-input
/// discipline closes here, memoized like #24's lazy bitmap. No level
/// is built here; the whole list is specs, realized per slot on
/// first consultation.
fn custom_level_specs<'a>(seps: &[Option<&'a str>]) -> Vec<LevelSlot<'a>> {
    let mut slots = Vec::new();
    let mut spliced = false;
    let mut seen_literals: HashSet<&str> = HashSet::new();
    for entry in seps {
        match entry {
            None => {
                if !spliced {
                    slots.extend(default_level_specs());
                    spliced = true;
                }
                // A duplicate splice is skipped, not re-spliced: the
                // window loop consults `slots` only through the
                // find_map (the first slot supplying a cut wins, and
                // the list is never indexed positionally or counted),
                // and a duplicate's levels are bit-identical to the
                // first splice's, so a duplicate can never change an
                // answer its original didn't already give: inert.
                // Skipping still matters under the lazy spelling: a
                // duplicate slot a window reached would realize (pay
                // its walk a second time) before returning the same
                // None its original just did, and the pre-#31 eager
                // spelling re-paid the three whole-text walks and
                // their ~45 MiB of cut vectors (on a 6 MiB document)
                // per duplicate outright, a caller-controlled
                // unbounded cost (`[None; 100]` is an OOM shape), the
                // same pathological-input discipline #21 established:
                // contribute once per call, memoized like #24's lazy
                // bitmap.
            }
            // Dropped before the seen-set: an empty literal never
            // produces a slot, so it must not occupy a seen-set entry
            // either (a later non-empty literal is still first-seen,
            // and any number of empty literals stays a no-op).
            Some("") => {}
            Some(s) => {
                // A duplicate literal is skipped for the same
                // dominance reason as a duplicate splice: the find_map
                // only ever reaches it after its original returned
                // None for this window, and its level is bit-identical
                // to the original's: the same None again, inert. The
                // walk a duplicate would otherwise re-pay is one
                // `memmem` scan plus a full cut vector, so `[Some("
                // "); N]` is the literal spelling of the `[None; 100]`
                // OOM shape: caller-controlled unbounded allocation.
                if seen_literals.insert(s) {
                    slots.push(LevelSlot {
                        spec: LevelSpec::Literal(s),
                        level: None,
                    });
                }
            }
        }
    }
    slots
}

/// The call's grapheme boundary index, built at the first need: the
/// realization-time cut filter inside [`LevelSlot::realize`], the
/// raw-cut fallback, and the overlap snap all funnel through here, so
/// "at most once per call, only when a call actually needs it" is one
/// line to read (and the test seam below counts every build). The
/// former spelling built the whole per-codepoint structure
/// unconditionally before anything else could run, which is the
/// document-scale cost #22 measured.
fn grapheme_index<'g>(
    graphemes: &'g mut Option<GraphemeIndex>,
    text: &str,
    total: usize,
) -> &'g GraphemeIndex {
    graphemes.get_or_insert_with(|| {
        #[cfg(test)]
        build_seam::bump_graphemes();
        GraphemeIndex::build(text, total)
    })
}

/// The #103 final-exit pre-test: could a separator match OPEN at
/// codepoint index `at`, i.e. could any level's [`Level::skip_cut`]
/// fire there? A necessary-condition scan over the level SPECS — no
/// level is realized here, which is the whole point: the final-chunk
/// exit must answer the skip question (#103) without costing the #30
/// zero-build contract its headline (a whole-document budget over 12 MiB
/// is pinned at < 0.8 ms by tests/test_performance.py; a paragraph walk
/// or one literal scan blows it). Soundness per arm:
///
/// * a REALIZED level answers exactly (its memoized, grapheme-filtered
///   cut list is what the search itself consults — free, O(log cuts));
/// * an unrealized LITERAL can carry a cut at `at` only if an occurrence
///   of the literal begins there. At `at == 0` that is byte-checkable
///   exactly (codepoint 0 IS byte 0) and cheap: `starts_with`. At
///   `at > 0` there is no O(1) byte mapping for a codepoint index (the
///   whole-text structure this module refuses to build, the #22 sin), so
///   the arm answers "maybe" — the caller descends to the search, which
///   realizes the level once (memoized) and answers exactly from then
///   on. Over-approximating is output-invisible: the search decides.
///   (A prior spelling answered `at == 0 && text.starts_with(sep)` —
///   "provably no" at every `at > 0` — which is UNSOUND: a literal
///   separator can begin at any codepoint, and a final window opening on
///   an unrealized fine level's match was pushed whole, emitting
///   pure-separator chunks and breaking the all-separator-zero-chunks
///   contract. The `separator_pretest_literal_at_gt_zero_may_open`
///   pins pin the soundness; the differential's separator pool covers
///   the unrealized-fine-level shape.)
/// * an unrealized PARAGRAPH level can carry a cut at `at` only if a
///   paragraph-gap cut begins there — its cut is
///   `(paragraph[i].end, paragraph[i+1].start)`, the gap a newline-run —
///   and `paragraph_bounds` discards empty leading spans (a text opening
///   on a blank run has its first segment start past the run), so at
///   `at == 0` no paragraph cut can ever begin (the default hierarchy's
///   whole-document budget keeps its zero-build exit); at `at > 0` the
///   arm answers "maybe", the same descend-and-memoize story.
/// * the contiguous levels (sentence, word) are structurally incapable:
///   their cuts have `next == cut_end`, and `skip_cut` requires
///   `next > cut_end` — no realization, no maybe, ever.
///
/// Every `false` is therefore provably no-skip (the final-chunk exit
/// pushes the whole remainder and the differential sweeps — the reference
/// oracle runs the bare search — hold the outputs equal), every `true`
/// merely costs the search a final window would pay anyway.
fn separator_may_open(slots: &[LevelSlot<'_>], text: &str, at: usize) -> bool {
    slots.iter().any(|slot| match &slot.level {
        Some(level) => level.skip_cut(at).is_some(),
        None => match slot.spec {
            // `at > 0` is "maybe" for BOTH unrealized match-carrying specs
            // (a literal can begin at any codepoint, a paragraph gap at
            // any non-zero one); only `at == 0` admits an exact O(1)
            // answer (the codepoint-0 = byte-0 `starts_with`, and the
            // paragraph arm's leading-blank-run discard).
            LevelSpec::Literal(sep) => at > 0 || text.starts_with(sep),
            LevelSpec::Paragraph => at > 0,
            LevelSpec::Sentence | LevelSpec::Word => false,
        },
    })
}

/// #47's word snap: the largest word-bounds cut at or before `snapped`
/// (the grapheme candidate), or `snapped` itself when the word-bounds
/// level has no boundary there (a dense-script run or one long token
/// with no internal boundary — the documented fallback to the plain
/// grapheme snap). The word-bounds level is the hierarchy's own when one
/// exists (the default hierarchy's Word slot, or a `None` splice's
/// copy): realized at the first snap that consults it, memoized, and
/// SHARED with any window that descends to it — the word level is never
/// built twice and never built for a call whose snaps (and windows) never
/// reach it, the #30 laziness discipline. A hierarchy with no word level
/// at all (an all-literal custom list) builds the one-off
/// [`word_fallback`] level at the first snap instead, exactly the build a
/// Word slot's realization produces (the UAX #29 word walk, the cuts
/// filtered to grapheme-cluster boundaries — the snap target must never
/// land mid-cluster, the same invariant the windows' cut filter
/// enforces).
///
/// The boundary set is the word SEGMENTS' ends (a contiguous partition:
/// the largest end at or before `snapped` is `snapped`'s own position
/// when it already sits on a word boundary, and otherwise the start of
/// the segment containing it — mid-word targets snap to their word's
/// first codepoint; mid-space-run targets snap to the run's start, a
/// UAX #29 word boundary like any other segment edge). The candidate
/// returned here is NOT the final start: the caller's decline-the-snap
/// lookahead (#83) runs on it unchanged, so a candidate reaching back to
/// or past the previous chunk's start, or one whose own chunk would not
/// advance past the just-emitted end, is declined exactly as a grapheme
/// candidate would be.
fn word_snap_back(
    levels: &mut [LevelSlot<'_>],
    word_fallback: &mut Option<Level>,
    text: &str,
    total: usize,
    graphemes: &mut Option<GraphemeIndex>,
    snapped: usize,
) -> usize {
    // At most one Word slot exists per call (the splice is once-per-call
    // and deduped; the default hierarchy has exactly one), so the first
    // match is the only one.
    let word_slot = levels
        .iter_mut()
        .find(|slot| matches!(slot.spec, LevelSpec::Word));
    let level = match word_slot {
        Some(slot) => slot.realize(text, total, graphemes),
        None => word_fallback.get_or_insert_with(|| {
            let mut level = level_from_contiguous_bounds(segmentation_impl::word_bounds(text));
            if !level.cuts.is_empty() {
                let g = grapheme_index(graphemes, text, total);
                level
                    .cuts
                    .retain(|&(end, next)| g.is_boundary(end) && g.is_boundary(next));
            }
            #[cfg(test)]
            build_seam::bump_levels();
            level
        }),
    };
    let hi = level.cuts.partition_point(|&(end, _)| end <= snapped);
    if hi > 0 {
        level.cuts[hi - 1].0
    } else {
        snapped
    }
}

/// Where the overlap snap may land the next chunk's start (#47). The
/// default is the historical behavior, so every existing caller is
/// unaffected; `Word` is the opt-in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlapBoundary {
    /// Snap backward to the nearest grapheme-cluster boundary (never
    /// mid-cluster, possibly mid-word): the historical default.
    Grapheme,
    /// Snap backward to the nearest grapheme-cluster boundary first, then
    /// further backward to the nearest UAX #29 word boundary at or before
    /// that candidate: the overlap tail starts at a word edge when one
    /// exists in the snap-back range. Falls back to the plain grapheme
    /// candidate when the word-bounds level has no boundary there (a
    /// dense-script run or one long token with no internal boundary), and
    /// is a no-op at `overlap == 0` (no snap site ever runs).
    Word,
}

/// Hierarchical fallback chunking of `text`: `(start, end)` codepoint-unit
/// pairs, each chunk at most `max_chars` codepoints, cut at the coarsest
/// level (first in `levels`, excluding the always-appended grapheme-safe
/// raw cut) that has an in-budget candidate, falling back to progressively
/// finer levels only when a coarser one has none over the current window.
/// The one exception, the same rule `crate::chunk_impl::chunk_text`
/// applies via its own `grapheme_safe_hard_cut` (replicated here against
/// the [`GraphemeIndex`] bitmap and differential-pinned against the
/// original in the tests): a single grapheme cluster wider than the
/// whole remaining budget (e.g. an oversized ZWJ emoji chain) is kept
/// whole rather than split, so that one chunk can exceed `max_chars`:
/// this never affects ordinary text (no cluster is more than a handful of
/// codepoints). See the module docs for the default-vs-custom hierarchy,
/// the `None`-entry splice, and the separator-dropped (not lossless)
/// contract, a contract that includes a separator whose match begins
/// exactly where the previous window resumed (consecutive separator
/// matches, e.g. `"\n\n"` over `"\n\n\n\n"`, or a declined overlap snap
/// landing there): its "chunk" would be empty, so the window skips the
/// match and resumes at its end BEFORE searching for a cut; no window
/// ever opens on a separator, so the raw-cut fallback can never fill a
/// window with the separator itself. Since #103 that includes the
/// final-chunk exit: the skip question is answered before the whole
/// remainder is emitted whole, so a window that opens on a separator
/// match never comes back as a chunk even there — a trailing separator
/// run survives only as a suffix of a content-bearing chunk (or not at
/// all: an all-separator document chunks to zero chunks).
/// `overlap` snaps the next
/// chunk's start backward from the just-emitted chunk's end to the nearest
/// grapheme boundary at or before the target (never mid-cluster): not
/// necessarily a semantic word/sentence/paragraph boundary the way
/// `chunk_text_overlapping`'s single-hierarchy overlap snap is; a
/// documented simplification of the general multi-level case, not a
/// silent gap. `overlap_boundary` opts into the word-aware snap ([`OverlapBoundary::Word`]:
/// the composition order is grapheme snap, then word snap, then the
/// decline-the-snap lookahead — the lookahead's candidate semantics carry
/// over unchanged, see the snap site below). The snap is declined — zero
/// overlap for just that one
/// transition, the next chunk starting at `cut.1` — when it would not buy
/// new context: a target at or before the chunk's own start (a short
/// trailing chunk, or a run of tight hard-cuts), or a snapped start whose
/// own chunk would end at or before the just-emitted chunk's end (a span
/// strictly contained in its predecessor, the same text re-embedded;
/// #83's decline-the-snap lookahead, the same rule
/// [`crate::chunk_impl::chunk_text_overlapping`] applies). Either way the
/// next chunk's end strictly advances past the current one's.
pub fn chunk_hierarchical(
    text: &str,
    max_chars: usize,
    separators: Option<&[Option<&str>]>,
    overlap: usize,
    overlap_boundary: OverlapBoundary,
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
    // The codepoint count as `char_count`'s ASCII-first pass: pure-ASCII
    // text answers its own byte length (the `is_ascii` gate, which LLVM
    // vectorizes outright), non-ASCII text pays the branchless count of
    // non-continuation bytes, instead of the former whole-text `Vec<char>`
    // collect (4 bytes per codepoint materialized before anything else
    // could run, the same O(source) allocation class #17 removed from
    // `is_grounded_fuzzy` and `chunk_text`'s grapheme grid before that).
    // `total` was the only thing that collect was ever read for; see
    // `char_count`'s own docs for the fast-path split's rationale.
    let total = char_count(text);

    // The hierarchy as specs: no walk paid yet. Every slot is realized
    // by the first window that consults it ([`LevelSlot::realize`], at
    // most once per call per slot), so a budget whose windows never
    // open a level pays nothing for it: the #30 lazy-level extension of
    // #24's lazy-bitmap idiom. The former spelling built every level,
    // and filtered every non-empty one, building the grapheme bitmap,
    // before the first window, so even a single-chunk budget
    // (`max_chars >= total`, this loop's own first-iteration exit) paid
    // all three default walks, ~176 ms and ~45 MiB of cut vectors over
    // a 6 MiB document, for levels that supplied zero cuts.
    let mut levels: Vec<LevelSlot> = match separators {
        Some(seps) => custom_level_specs(seps),
        None => default_level_specs(),
    };
    // The grapheme boundary index: built lazily, at most once per call,
    // at the first site that actually needs it (a realized level's cut
    // filter, the raw-cut fallback, or the overlap snap, every site one
    // [`grapheme_index`] call). The former spelling built the whole
    // per-codepoint structure unconditionally before anything else
    // could run, which is the document-scale cost #22 measured.
    let mut graphemes: Option<GraphemeIndex> = None;

    let mut chunks = Vec::with_capacity(total / max_chars + 1);
    let mut start = 0usize;
    let mut iterations = 0usize;
    // #47's word-bounds fallback for hierarchies with no word level at
    // all (an all-literal custom list): built by the first word snap that
    // needs it, at most once per call, exactly the build a Word slot's
    // own realization produces (the same walk, the same grapheme cut
    // filter). The hierarchy's own Word slot — the default hierarchy's,
    // or a `None` splice's copy — is preferred at the snap site, so the
    // snap shares the windows' memoized build instead of paying a second
    // walk; this fallback exists only when no slot carries word bounds.
    let mut word_fallback: Option<Level> = None;
    while start < total {
        iterations += 1;
        assert!(
            iterations <= total + 1,
            "chunk_hierarchical: forward-progress invariant violated"
        );
        let remaining = total - start;
        let final_window = remaining <= max_chars;
        // The final-chunk exit — but #103's skip question is answered
        // first: a window that OPENS on a separator match is skipped even
        // here (a trailing separator run survives only as a suffix of a
        // content-bearing chunk; an all-separator document chunks to zero
        // chunks). Answering it must not cost the #30 zero-build contract
        // its headline (a whole-document budget over 12 MiB is pinned at
        // < 0.8 ms by tests/test_performance.py — a paragraph walk or one
        // literal scan blows it), so the cheap pre-test below answers
        // "could the skip possibly fire here" over the UNREALIZED SPECS,
        // and only a `true` descends to the real search (whose verdict
        // decides the skip question exactly). The pre-test is a
        // necessary-condition scan, over-approximating on purpose: every
        // `false` is provably no-skip (output-invisible — the reference
        // oracle below runs the bare search and the differential sweeps
        // hold the two equal), every `true` is merely "maybe".
        if final_window && !separator_may_open(&levels, text, start) {
            chunks.push((start, total));
            break;
        }
        let limit = start.saturating_add(max_chars);
        // The window's verdict, searched coarsest-first with the find_map
        // short-circuit preserved exactly (the first slot with a verdict
        // wins; later slots stay unrealized for this window; a later
        // window that reaches them reuses their memoized build; a
        // consulted slot is built even when it supplies no verdict:
        // consultation, not success, is what pays the walk):
        //
        // * Skip: a separator match beginning exactly at `start`, the
        //   window OPENS on a separator (consecutive matches make one
        //   match's drop resume exactly at the next match's start, and a
        //   declined overlap snap can land there). Its "chunk" would be
        //   empty, and the raw-cut fallback would fill the window with
        //   the separator itself; separators are dropped between chunks,
        //   every one of them, so the window skips the match and
        //   restarts at its end. The skip beats the slot's own genuine
        //   cuts (a cut past the match would carry the separator as
        //   content) and stops the search before finer slots are
        //   consulted, so the laziness discipline is untouched: a level
        //   is realized exactly when the window's search descends to it.
        //   Forward progress is unconditional (`skip_cut` answers a
        //   `next_start` strictly past `start`) and the loop-head
        //   iteration counter still bounds the skips by `total + 1`.
        //   Since #103 the skip also preempts the final-chunk exit: the
        //   exit runs only after this search (when the pre-test above
        //   could not rule the skip out), so the whole-remainder push
        //   below happens only for a window that does not open on a
        //   separator match.
        // * Cut: the slot's largest in-budget cut past `start`, exactly
        //   as this search always ran.
        let verdict = levels.iter_mut().find_map(|slot| {
            let level = slot.realize(text, total, &mut graphemes);
            level.skip_cut(start).map(Verdict::Skip).or_else(|| {
                level
                    .best_cut(start, limit)
                    .map(|(end, next)| Verdict::Cut(end, next))
            })
        });
        let cut = match verdict {
            Some(Verdict::Skip(next)) => {
                start = next;
                continue;
            }
            Some(Verdict::Cut(cut_end, next_start)) => (cut_end, next_start),
            None if final_window => {
                // No level supplied any verdict and the whole remainder
                // fits the budget: the final chunk runs to the end
                // untrimmed, before the raw-cut fallback — building the
                // grapheme index for a hard cut this exit would discard
                // (the pre-test said a separator might open here, the
                // search answered no).
                chunks.push((start, total));
                break;
            }
            None => {
                let g = grapheme_index(&mut graphemes, text, total);
                let end = g.hard_cut(start, limit);
                (end, end)
            }
        };
        if final_window {
            // The final-chunk exit: the whole remainder fits the budget,
            // and the window does not open on a separator match (the
            // pre-test plus this search would have skipped it — #103), so
            // the chunk runs to the end untrimmed, the same
            // final-chunk-runs-untrimmed exception `chunk_text`
            // documents. The search's own cut is discarded here: the
            // final chunk is not re-cut.
            chunks.push((start, total));
            break;
        }
        chunks.push((start, cut.0));
        if overlap == 0 {
            start = cut.1;
        } else {
            let target = cut.0.saturating_sub(overlap);
            let g = grapheme_index(&mut graphemes, text, total);
            let snapped = g.last_at_or_before(target);
            // #47's word snap, second in the composition order: the
            // grapheme candidate above lands first (never mid-cluster),
            // then the word snap moves it further back to the nearest
            // word-bounds cut at or before it, and the decline-the-snap
            // lookahead BELOW runs on the word-snapped candidate
            // unchanged — the lookahead's candidate semantics carry over
            // whole (it never sees the pre-word candidate). The word snap
            // may land at or before this chunk's own start; that is not
            // clamped away, it is declined by the lookahead's own
            // `snapped > start` conjunct (zero overlap for the
            // transition), the documented degradation for a candidate
            // that buys no new context — the same rule that declines a
            // grapheme candidate reaching back past the chunk start.
            let snapped = if overlap_boundary == OverlapBoundary::Word {
                word_snap_back(
                    &mut levels,
                    &mut word_fallback,
                    text,
                    total,
                    &mut graphemes,
                    snapped,
                )
            } else {
                snapped
            };
            // Decline-the-snap with lookahead (#83): accept the candidate
            // only when it starts past this chunk's own start, ends before
            // this chunk's end, and the chunk cut from there ends strictly
            // past this chunk's end. `next_end` is the loop body's own cut
            // computation run from `snapped` (the levels are memoized, so
            // this costs one partition_point per level, and the raw-cut
            // fallback the same hard cut): a candidate whose own chunk
            // would end at or before `cut.0` emits a span strictly
            // contained in its predecessor (the same text re-embedded, no
            // new context for the overlap to buy), so the snap is declined
            // and the transition degrades to zero overlap (`start =
            // cut.1`), the documented degradation. The lookahead reads the
            // levels' direct cuts only: a separator match beginning
            // exactly at `snapped` is not consulted as a skip here: the
            // snap is an acceptance heuristic, and the window that
            // eventually runs from `snapped` applies the skip itself, so
            // declining on the direct cut's answer can only forgo an
            // overlap (the documented degradation), never emit a wrong or
            // stalled chunk.
            let next_end = if total - snapped <= max_chars {
                total
            } else {
                let next_limit = snapped + max_chars;
                levels
                    .iter_mut()
                    .find_map(|slot| {
                        slot.realize(text, total, &mut graphemes)
                            .best_cut(snapped, next_limit)
                    })
                    .map(|(end, _next)| end)
                    .unwrap_or_else(|| {
                        let g = grapheme_index(&mut graphemes, text, total);
                        g.hard_cut(snapped, next_limit)
                    })
            };
            start = if snapped > start && snapped < cut.0 && next_end > cut.0 {
                snapped
            } else {
                cut.1
            };
        }
    }
    chunks
}

/// The test-only counting seam behind the lazy-level pins: per-thread
/// counters of the level realizations and grapheme-index builds this
/// module issues, so the tests can assert exactly which levels a call
/// built, how often, and whether it built the bitmap at all, counts
/// the differential sweep cannot see (it pins lazy == eager output;
/// this pins the laziness itself). Per-thread because the suite's
/// tests run in parallel: each test's calls bump its own thread's
/// counters and cannot interfere. Production builds compile none of
/// this, so the release path carries zero counting cost.
#[cfg(test)]
mod build_seam {
    use std::cell::Cell;

    thread_local! {
        static LEVELS_BUILT: Cell<usize> = const { Cell::new(0) };
        static GRAPHEME_INDEX_BUILDS: Cell<usize> = const { Cell::new(0) };
    }

    pub fn reset() {
        LEVELS_BUILT.with(|c| c.set(0));
        GRAPHEME_INDEX_BUILDS.with(|c| c.set(0));
    }

    pub fn levels_built() -> usize {
        LEVELS_BUILT.with(Cell::get)
    }

    pub fn grapheme_index_built() -> usize {
        GRAPHEME_INDEX_BUILDS.with(Cell::get)
    }

    pub fn bump_levels() {
        LEVELS_BUILT.with(|c| c.set(c.get() + 1));
    }

    pub fn bump_graphemes() {
        GRAPHEME_INDEX_BUILDS.with(|c| c.set(c.get() + 1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The differential oracle below is the pre-#22 implementation verbatim,
    // which is why the tests module re-imports the three spellings the
    // production code no longer uses: the `Vec<char>` collect's total, the
    // `grapheme_boundary_chars` + `HashSet<usize>` filter, and
    // `grapheme_safe_hard_cut` over the usize grid.
    use std::collections::HashSet;

    use crate::chunk_impl::grapheme_safe_hard_cut;
    use crate::truncate_impl::grapheme_boundary_chars;

    /// The tests module's grapheme-default spelling: the historical
    /// four-argument shape with `overlap_boundary = Grapheme`, the mode
    /// every pre-#47 pin means (the shadowing definition beats the glob
    /// import, so this module's existing call sites read unchanged).
    /// Word-mode tests call the five-argument production function through
    /// `super::chunk_hierarchical`.
    #[allow(unused)]
    fn chunk_hierarchical(
        text: &str,
        max_chars: usize,
        separators: Option<&[Option<&str>]>,
        overlap: usize,
    ) -> Vec<(usize, usize)> {
        super::chunk_hierarchical(
            text,
            max_chars,
            separators,
            overlap,
            OverlapBoundary::Grapheme,
        )
    }

    fn text_of(chunks: &[(usize, usize)], text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        chunks
            .iter()
            .map(|&(s, e)| chars[s..e].iter().collect())
            .collect()
    }

    /// The pre-#22 implementation, kept verbatim as the differential
    /// oracle for the bitmap/lazy rewrite: every behavior-affecting line
    /// of the former code (the whole-text `Vec<char>` collect, the
    /// unconditional `grapheme_boundary_chars` + `HashSet<usize>` filter
    /// over every level, `grapheme_safe_hard_cut` over the usize grid, the
    /// `partition_point` overlap snap) runs against the new spelling over
    /// a corpus × budget × overlap × hierarchy sweep below. The rewrite
    /// claims bit-identical output; this is the pin. Three deliberate
    /// post-verbatim edits, mirrored on both sides in lockstep: the
    /// decline-the-snap lookahead in the overlap branch (the oracle's
    /// snap carried the same contained-chunk defect the production loop
    /// did), so the pin compares fixed machine against fixed machine; the
    /// separator-at-the-window-start skip in the window loop (the
    /// oracle shared the raw-cut-fallback defect that let a separator
    /// whose match begins exactly at a window start come back as a chunk
    /// of its own), skipped identically on both sides; and since #103 the
    /// final-chunk shortcut moved AFTER the skip question — a window that
    /// opens on a separator match is skipped even at the whole-remainder
    /// exit, on both sides (the oracle runs the bare search there; the
    /// production loop's spec pre-test is an output-invisible laziness
    /// guard, and the sweeps below hold the two spellings equal). #47's
    /// word snap is mirrored too: the oracle builds its word-bounds level
    /// up front (eager) — the boundary set is the same list the
    /// production snap reads through the hierarchy's Word slot or its
    /// one-off fallback (the same UAX #29 walk, the same grapheme cut
    /// filter), so the two need not track which slot carried it.
    fn chunk_hierarchical_reference(
        text: &str,
        max_chars: usize,
        separators: Option<&[Option<&str>]>,
        overlap: usize,
        overlap_boundary: OverlapBoundary,
    ) -> Vec<(usize, usize)> {
        if text.is_empty() {
            return Vec::new();
        }
        assert!(max_chars > 0, "max_chars must be at least 1, got 0");
        let chars: Vec<char> = text.chars().collect();
        let total = chars.len();

        // The oracle keeps its own inline eager level building (not the
        // spec list + [`LevelSlot::realize`] spelling the production
        // code now uses) so the differential sweep below pins the lazy
        // spelling's output against this eager one, the exact pin that
        // lazy == eager, the same way it pins the bitmap machinery:
        // extended for the `None` entry in the former code's own inline
        // style.
        let mut levels: Vec<Level> = match separators {
            Some(seps) => {
                let mut built = Vec::new();
                for entry in seps {
                    match entry {
                        None => {
                            built.push(level_from_paragraph_bounds(paragraph_bounds(text)));
                            built.push(level_from_contiguous_bounds(
                                segmentation_impl::sentence_bounds(text),
                            ));
                            built.push(level_from_contiguous_bounds(
                                segmentation_impl::word_bounds(text),
                            ));
                        }
                        Some("") => {}
                        Some(s) => built.push(level_from_literal(text, s)),
                    }
                }
                built
            }
            None => vec![
                level_from_paragraph_bounds(paragraph_bounds(text)),
                level_from_contiguous_bounds(segmentation_impl::sentence_bounds(text)),
                level_from_contiguous_bounds(segmentation_impl::word_bounds(text)),
            ],
        };
        let grapheme_starts = grapheme_boundary_chars(text);
        let grapheme_set: HashSet<usize> = grapheme_starts.iter().copied().collect();
        for level in &mut levels {
            level
                .cuts
                .retain(|&(end, next)| grapheme_set.contains(&end) && grapheme_set.contains(&next));
        }
        // #47's word-bounds level for the "word" snap mode: built up front
        // (eager, the oracle's idiom) with the same cut filter every level
        // gets. Only built when the mode can consult it.
        let word_level = if overlap_boundary == OverlapBoundary::Word {
            let mut level = level_from_contiguous_bounds(segmentation_impl::word_bounds(text));
            level
                .cuts
                .retain(|&(end, next)| grapheme_set.contains(&end) && grapheme_set.contains(&next));
            Some(level)
        } else {
            None
        };

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
            let final_window = remaining <= max_chars;
            let limit = start.saturating_add(max_chars);
            // The window's verdict, mirrored from the production loop in
            // lockstep (the second deliberate post-verbatim edit): per
            // slot, the separator-at-the-window-start skip beats the
            // slot's own genuine cuts, and the first slot with a verdict
            // wins; a window that opens on a separator skips it before
            // any cut is searched (the separator is dropped between
            // chunks, never emitted as one). Since #103 this includes the
            // final window: the whole-remainder exit below runs only when
            // the search did not answer Skip.
            let verdict = levels.iter().find_map(|level| {
                level.skip_cut(start).map(Verdict::Skip).or_else(|| {
                    level
                        .best_cut(start, limit)
                        .map(|(end, next)| Verdict::Cut(end, next))
                })
            });
            let cut = match verdict {
                Some(Verdict::Skip(next)) => {
                    start = next;
                    continue;
                }
                Some(Verdict::Cut(cut_end, next_start)) => (cut_end, next_start),
                None if final_window => {
                    // The final-chunk exit ahead of the raw-cut fallback:
                    // the oracle never builds a hard cut the exit would
                    // discard.
                    chunks.push((start, total));
                    break;
                }
                None => {
                    let end = grapheme_safe_hard_cut(&grapheme_starts, start, limit);
                    (end, end)
                }
            };
            if final_window {
                // The final-chunk exit (#103): the search above has
                // already answered the skip question — a Skip verdict
                // preempted this exit — and its cut is discarded: the
                // final chunk runs to the end untrimmed.
                chunks.push((start, total));
                break;
            }
            chunks.push((start, cut.0));
            if overlap == 0 {
                start = cut.1;
            } else {
                let target = cut.0.saturating_sub(overlap);
                let ghi = grapheme_starts.partition_point(|&g| g <= target);
                let snapped = if ghi > 0 { grapheme_starts[ghi - 1] } else { 0 };
                // #47's word snap, second in the composition order,
                // mirrored from the production snap: the grapheme
                // candidate lands first, then the largest word-bounds cut
                // at or before it (or the grapheme candidate back when
                // the word level has no boundary there), and the
                // decline-the-snap lookahead below runs on the
                // word-snapped candidate unchanged.
                let snapped = match &word_level {
                    Some(word) => {
                        let hi = word.cuts.partition_point(|&(end, _)| end <= snapped);
                        if hi > 0 { word.cuts[hi - 1].0 } else { snapped }
                    }
                    None => snapped,
                };
                // The production snap's decline-the-snap lookahead (#83),
                // spelled against this oracle's own eager levels: kept in
                // verbatim lockstep with chunk_hierarchical's, or the
                // differential sweep below diverges exactly where one side
                // declines a snap and the other accepts it. `next_end` is
                // the oracle loop's own computation run from `snapped`.
                let next_end = if total - snapped <= max_chars {
                    total
                } else {
                    let next_limit = snapped + max_chars;
                    levels
                        .iter()
                        .find_map(|level| level.best_cut(snapped, next_limit))
                        .map(|(end, _next)| end)
                        .unwrap_or_else(|| {
                            grapheme_safe_hard_cut(&grapheme_starts, snapped, next_limit)
                        })
                };
                start = if snapped > start && snapped < cut.0 && next_end > cut.0 {
                    snapped
                } else {
                    cut.1
                };
            }
        }
        chunks
    }

    /// The differential corpus: every cluster shape the filter and the
    /// hard cut have to get right: plain ASCII, CRLF/blank-line
    /// paragraphs, Thai SARA AM (the combining sequence UAX #29 word/
    /// sentence bounds split but grapheme rules join), ZWJ emoji chains,
    /// regional-indicator pairs, decomposed accents, and the degenerate
    /// repeated-character runs that stress the raw cut.
    fn differential_corpus() -> Vec<String> {
        vec![
            "a".to_string(),
            "hello world".to_string(),
            "Short one.\n\nShort two.".to_string(),
            "Alpha beta gamma delta epsilon zeta eta theta.".to_string(),
            "one two three four five six seven eight nine ten eleven".to_string(),
            "# Title\nintro text\n## Section\nmore text here that is long".to_string(),
            "a\r\nb\r\n\r\nc".to_string(),
            "a\rb".to_string(),
            "\r".to_string(),
            "\r\n".to_string(),
            "ab 0\u{0E33} cd ef 0\u{0E33} gh ij 0\u{0E33} kl".to_string(),
            "0\u{0E33}0\u{0E33}0\u{0E33}".to_string(),
            "\u{0E33}\u{0E33}".to_string(),
            "e\u{0301}e\u{0301}e\u{0301} ".to_string(),
            "thumbs up \u{1F44D}\u{200D}\u{1F3FB} flag \u{1F1FA}\u{1F1F8}".to_string(),
            "q".repeat(200),
            "q".repeat(128),
            "0\u{0E33}".repeat(32),
            "abcdefghijklmnopqrstuvwxyz".repeat(8),
            "\u{4E2D}\u{6587}\u{6587}\u{672C}\u{FF0C}\u{6D4B}\u{8BD5}".to_string(),
            "mixed 0\u{0E33} ascii \u{1F600} \u{4E2D}\u{6587} tail".to_string(),
            // The #103 shapes: trailing and leading separator runs — the
            // final-chunk exit's skip question is exactly what these
            // stress (a trailing window opening on a match, a document
            // that is all matches, a run spanning several skips).
            "aa\n\n\n\n".to_string(),
            "\n\n".to_string(),
            "\n\n\n\n".to_string(),
            "a\n\n\n\n".to_string(),
            "ab\n\n\ncd".to_string(),
        ]
    }

    #[test]
    fn bitmap_lazy_spelling_matches_the_former_implementation_exactly() {
        let separator_cases: Vec<Option<Vec<Option<&str>>>> = vec![
            None,
            Some(vec![]),
            Some(vec![Some("ZZZ_NEVER_MATCHES")]),
            Some(vec![Some("xyz")]),
            Some(vec![Some(" ")]),
            Some(vec![Some(". ")]),
            Some(vec![Some("\n\n"), Some(" ")]),
            Some(vec![Some("\u{0E33}")]),
            Some(vec![Some(""), Some(" ")]),
            // The None-entry splice: alone (== default), under a literal
            // (the line-first shape), above a literal, and mid-list.
            Some(vec![None]),
            Some(vec![Some("\n"), None]),
            Some(vec![Some("\n## "), None, Some("\n")]),
            Some(vec![Some(""), None]),
        ];
        for text in differential_corpus() {
            let total = text.chars().count();
            let budgets = (1..=total.min(48))
                .chain([64, 97])
                .filter(|&m| m <= total.max(1));
            for max_chars in budgets {
                // The pyo3 layer's validated envelope is overlap < max_chars
                // (the fuzz target clamps to the same range); sweep the
                // boundary-adjacent values, not just one mid choice.
                for overlap in [0usize, 1, max_chars.saturating_sub(1)]
                    .into_iter()
                    .filter(|&o| o < max_chars)
                {
                    for seps in &separator_cases {
                        let sep_refs: Option<Vec<Option<&str>>> = seps.as_ref().map(|v| v.to_vec());
                        // Both overlap-boundary modes: the word snap is a
                        // pure grapheme-candidate refinement, so the sweep
                        // pins the word machinery against the oracle's own
                        // word level on every cell the grapheme sweep
                        // already ran (#47).
                        for boundary in [OverlapBoundary::Grapheme, OverlapBoundary::Word] {
                            let new = super::chunk_hierarchical(
                                &text,
                                max_chars,
                                sep_refs.as_deref(),
                                overlap,
                                boundary,
                            );
                            let old = chunk_hierarchical_reference(
                                &text,
                                max_chars,
                                sep_refs.as_deref(),
                                overlap,
                                boundary,
                            );
                            assert_eq!(
                                new, old,
                                "divergence: text={text:?} max_chars={max_chars} overlap={overlap} \
                                 boundary={boundary:?} separators={seps:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn lookahead_memoized_levels_match_the_eager_oracle_over_every_overlap() {
        // H3, attacked: #83's lookahead re-consults the level slots
        // (`realize` + `best_cut`) for the snapped candidate's window
        // before the production cut for that window ever runs, so the
        // question is whether realize-then-query through the lookahead can
        // mutate the memoized state (a level built at lookahead time,
        // filtered differently, or the grapheme index built early) in a
        // way the subsequent production cut then inherits but the eager
        // oracle does not see. `realize` is get_or_insert_with-pure (the
        // build and its cut filter run exactly once, deterministically),
        // so the answer should be no — this sweep pins it by running EVERY
        // overlap value 0..max_chars-1 (not just the boundary-adjacent
        // three the bitmap sweep uses: the accepted/declined alternation
        // the lookahead decides is per-transition, and only the full
        // overlap range walks every branch of it) over the differential
        // corpus x three hierarchy shapes. Overlap > 0 transitions also
        // re-assert #83's ends-advance invariant independently of the
        // oracle, so a lockstep bug on both sides of the differential
        // cannot hide here.
        let separator_cases: Vec<Option<Vec<Option<&str>>>> =
            vec![None, Some(vec![Some("\n"), None]), Some(vec![Some(" ")])];
        for text in differential_corpus() {
            let total = text.chars().count();
            for max_chars in 1..=total.min(24) {
                for overlap in 0..max_chars {
                    for seps in &separator_cases {
                        let sep_refs: Option<Vec<Option<&str>>> = seps.as_ref().map(|v| v.to_vec());
                        for boundary in [OverlapBoundary::Grapheme, OverlapBoundary::Word] {
                            let new = super::chunk_hierarchical(
                                &text,
                                max_chars,
                                sep_refs.as_deref(),
                                overlap,
                                boundary,
                            );
                            let old = chunk_hierarchical_reference(
                                &text,
                                max_chars,
                                sep_refs.as_deref(),
                                overlap,
                                boundary,
                            );
                            assert_eq!(
                                new, old,
                                "lookahead/production divergence: text={text:?} \
                                 max_chars={max_chars} overlap={overlap} boundary={boundary:?} \
                                 separators={seps:?}"
                            );
                            if overlap > 0 {
                                let mut prev_end = 0usize;
                                for &(s, e) in &new {
                                    assert!(
                                        e > prev_end,
                                        "ends not strictly advancing under overlap={overlap}: \
                                         text={text:?} max_chars={max_chars} boundary={boundary:?} \
                                         separators={seps:?}"
                                    );
                                    let _ = s;
                                    prev_end = e;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_separator_at_the_window_start_is_skipped_not_emitted() {
        // Two adjacent "\n\n" matches: the second begins exactly where
        // the first's drop resumed, the separator level supplies no cut
        // past it, and the raw cut used to slice the separator out as a
        // chunk of its own (its own span, no content). The separator is
        // dropped between chunks instead: the chunk sequence is the
        // content only.
        assert_eq!(
            chunk_hierarchical("a\n\n\n\nb", 2, Some(&[Some("\n\n")]), 0),
            vec![(0, 1), (5, 6)]
        );
        // A leading separator (the text opens on a match) is skipped the
        // same way (twice in a row here), its would-be span dropped from
        // the head: no window may open on a separator, whatever a genuine
        // cut farther in could have consumed.
        assert_eq!(
            chunk_hierarchical("\n\n\n\nb", 2, Some(&[Some("\n\n")]), 0),
            vec![(4, 5)]
        );
        // A run of three adjacent matches skips twice in a row.
        assert_eq!(
            chunk_hierarchical("x------y", 3, Some(&[Some("---")]), 0),
            vec![(0, 1), (7, 8)]
        );
        // The skip composes with overlap: the skipped transition carries
        // no snap, the next emitted chunk's overlap is computed from the
        // chunk that actually follows the skipped separators.
        let overlapped = chunk_hierarchical("a\n\n\n\nbbbbbb", 4, Some(&[Some("\n\n")]), 2);
        for w in overlapped.windows(2) {
            assert!(
                w[1].1 > w[0].1,
                "ends must advance under overlap: {overlapped:?}"
            );
        }
        for &(s, e) in &overlapped {
            assert!(
                &"a\n\n\n\nbbbbbb"[s..e] != "\n\n",
                "a separator came back as its own chunk: {overlapped:?}"
            );
        }
    }

    // ---- #103, reopened: the final-chunk exit answers the skip question ----

    #[test]
    fn a_trailing_window_opening_on_a_separator_is_skipped_even_at_the_final_exit() {
        // The reopened ticket's two repros: the `remaining <= max_chars`
        // shortcut used to run BEFORE the separator-at-the-window-start
        // skip, so a trailing window that opens on a match was emitted
        // whole — `'\n\n'` came back as `[(0, 2)]` (the whole chunk the
        // separator), `'aa\n\n\n\n'` as `[(0, 2), (4, 6)]` with `(4, 6)`
        // pure separator. The contract the fix pins: a window that OPENS
        // on a separator match is skipped wherever the skip applies,
        // INCLUDING the final-chunk exit; a trailing separator run
        // survives only as a suffix of a content-bearing chunk; an
        // all-separator document chunks to zero chunks.
        assert_eq!(
            super::chunk_hierarchical(
                "\n\n",
                5,
                Some(&[Some("\n\n")]),
                0,
                OverlapBoundary::Grapheme
            ),
            Vec::<(usize, usize)>::new(),
            "an all-separator document must chunk to zero chunks"
        );
        assert_eq!(
            super::chunk_hierarchical(
                "aa\n\n\n\n",
                2,
                Some(&[Some("\n\n")]),
                0,
                OverlapBoundary::Grapheme
            ),
            vec![(0, 2)],
            "the trailing (4, 6) window opens on a match and must be skipped"
        );
        // A trailing run longer than one match skips repeatedly — each
        // skip advances `start` strictly, so the loop-head iteration
        // counter stays an honest bound and the run ends in zero chunks,
        // not one chunk per match.
        assert_eq!(
            super::chunk_hierarchical(
                "\n\n\n\n\n\n\n\n",
                2,
                Some(&[Some("\n\n")]),
                0,
                OverlapBoundary::Grapheme
            ),
            Vec::<(usize, usize)>::new()
        );
        // The run survives only as a suffix of content: the chunk before
        // it ends inside content... and the chunk AFTER the skips, when
        // content follows the run, is that content alone.
        assert_eq!(
            super::chunk_hierarchical(
                "aa\n\n\n\nbb",
                2,
                Some(&[Some("\n\n")]),
                0,
                OverlapBoundary::Grapheme
            ),
            vec![(0, 2), (6, 8)]
        );
        // The one-residue case, pinned as documented behavior: a trailing
        // '\n' that is NOT a "\n\n" match (the matcher consumed the pair
        // before it) opens no window on a match, so the raw-cut final
        // exit emits it. The contract is match-shaped (`skip_cut`
        // answers "a match begins at `start`"), and the residue is the
        // same lone-character ride-along any too-short remainder gets.
        assert_eq!(
            super::chunk_hierarchical(
                "aa\n\n\n",
                2,
                Some(&[Some("\n\n")]),
                0,
                OverlapBoundary::Grapheme
            ),
            vec![(0, 2), (4, 5)]
        );
    }

    #[test]
    fn separator_pretest_literal_at_gt_zero_may_open() {
        // The final-exit PRE-TEST's own soundness, beyond the
        // single-level shapes above: the unrealized-LITERAL arm used to
        // answer `at == 0 && text.starts_with(sep)` — "provably no" at
        // every `at > 0` — but a literal separator can begin at ANY
        // codepoint. A hierarchy with a fine level left unrealized by
        // the `find_map` short-circuit (here: the `"\n"` level under the
        // `"\n\n"` verdicts) then pushed final windows that OPEN on a
        // match: `"\n"*5` chunked to `[(4, 5)]` (a pure-separator chunk,
        // the #103 relapse) and `"XaX\nbb"` lost the `"\n"` skip at the
        // final exit. The arm answers "maybe" at `at > 0` (the caller
        // descends to the exact, memoized search — the same
        // over-approximate discipline the PARAGRAPH arm has always
        // used), so every shape below matches the bare-search oracle.
        //
        // The unit-level pin on the pre-test itself: an unrealized
        // literal at `at > 0` can never answer "provably no".
        let slots: Vec<LevelSlot<'_>> = ["\n\n", "\n"]
            .into_iter()
            .map(|sep| LevelSlot {
                spec: LevelSpec::Literal(sep),
                level: None,
            })
            .collect();
        assert!(separator_may_open(&slots, "\n\n\n\n\n", 3));
        assert!(separator_may_open(&slots, "XaX\nbb", 3));
        // `at == 0` keeps its exact O(1) answer (the zero-build
        // contract's headline is a whole-document window at `at == 0`).
        assert!(!separator_may_open(&slots, "abc", 0));
        assert!(separator_may_open(&slots, "\nabc", 0));
        // End-to-end: the red-team repros, each previously diverging
        // from the bare-search reference oracle.
        assert_eq!(
            super::chunk_hierarchical(
                "\n\n\n\n\n",
                5000,
                Some(&[Some("\n\n"), Some("\n")]),
                0,
                OverlapBoundary::Grapheme
            ),
            Vec::<(usize, usize)>::new(),
            "an all-separator document chunks to zero chunks even when a fine literal level is unrealized"
        );
        assert_eq!(
            super::chunk_hierarchical(
                "XaX\nbb",
                3,
                Some(&[Some("X"), Some("\n")]),
                0,
                OverlapBoundary::Grapheme
            ),
            vec![(1, 2), (4, 6)],
            "the final window opening on an unrealized level's match is skipped, not emitted"
        );
        assert_eq!(
            super::chunk_hierarchical(
                "XaX\n\n\n",
                3,
                Some(&[Some("X"), Some("\n")]),
                0,
                OverlapBoundary::Grapheme
            ),
            vec![(1, 2)],
            "a trailing separator run survives only as a suffix of a content-bearing chunk"
        );
        // Overlap composition: the trailing pure-separator chunk the
        // pre-test used to emit at the overlap snap's landing.
        assert_eq!(
            super::chunk_hierarchical(
                "babaaababaaababaaababaaababaaababaaabaa",
                3,
                Some(&[Some("ab"), Some("ba"), Some("a"), Some("b")]),
                2,
                OverlapBoundary::Grapheme
            )
            .last()
            .copied(),
            Some((33, 35)),
            "the last chunk ends on content, not on a separator match"
        );
    }

    #[test]
    fn the_final_exit_skip_holds_under_every_separator_shape() {
        // The red-team matrix over the final-exit skip, as a battery:
        // length-1 separators (a match at every separator codepoint),
        // separators wider than the window, runs spanning several
        // windows, custom empty-string separators (a no-op level that
        // must not resurrect the shortcut), and overlap > 0 with
        // separator-final documents (a declined snap landing on a match
        // must not hand the NEXT iteration's final exit a separator
        // chunk — the skip preempts it, #103's lookahead corner). The
        // contract, per cell: no chunk's slice is exactly a separator
        // the caller asked to split on, and under overlap the ends
        // strictly advance.
        let cases: Vec<(&str, &[Option<&str>], usize)> = vec![
            ("a\n\nb", &[Some("\n")], 2),
            ("a\n\n\n\nb", &[Some("\n")], 2),
            ("a\nb\nc\nd", &[Some("\n")], 1),
            ("-----cd", &[Some("-----")], 2),
            ("ab-----cd-----ef", &[Some("-----")], 4),
            ("x------y", &[Some("---")], 3),
            ("\n\n\n--\n\n\n", &[Some("\n\n")], 3),
            ("aa\n\n\n\n", &[Some("\n\n")], 2),
            ("aa\n\n\n\n", &[Some("\n\n")], 3),
            ("a\n\n\n\nb", &[Some("")], 2),
            ("a\n\n\n\nb", &[Some(""), Some("\n\n")], 2),
            ("a\n\n\n\nbbbbbb", &[Some("\n\n")], 4),
        ];
        for (text, seps, max_chars) in cases {
            for overlap in [0usize, 1, max_chars.saturating_sub(1)] {
                let chunks = super::chunk_hierarchical(
                    text,
                    max_chars,
                    Some(seps),
                    overlap,
                    OverlapBoundary::Grapheme,
                );
                for sep in seps.iter().flatten() {
                    for &(s, e) in &chunks {
                        assert!(
                            &text[s..e] != *sep,
                            "separator {sep:?} came back as chunk ({s}, {e}): \
                             text={text:?} m={max_chars} ov={overlap}: {chunks:?}"
                        );
                    }
                }
                let mut prev_end = 0usize;
                for &(s, e) in &chunks {
                    if overlap > 0 {
                        assert!(
                            e > prev_end,
                            "ends not strictly advancing: {chunks:?} \
                             text={text:?} m={max_chars} ov={overlap}"
                        );
                    } else {
                        assert!(e >= prev_end, "ends moved backward: {chunks:?}");
                    }
                    let _ = s;
                    prev_end = e;
                }
            }
        }
    }

    #[test]
    fn a_snap_landing_on_a_separator_is_skipped_by_the_next_window_not_emitted() {
        // The #103 corner the overlap lookahead leaves open BY DESIGN
        // (the lookahead reads the levels' direct cuts only — a separator
        // match at `snapped` is not consulted there): with the fix at the
        // loop head, the window that eventually runs from an accepted
        // `snapped` applies the skip itself, so an overlap transition
        // whose snap lands exactly on a separator match can never emit
        // the match as a chunk. The battery walks every legal overlap
        // over separator-run texts where snaps land on and beside
        // matches; the output never carries the separator alone.
        let text = "abcdefgh\n\nijklmnop\n\nqrstuvwx\n\nyz012345";
        for max_chars in 4..=16 {
            for overlap in 1..max_chars {
                let chunks = super::chunk_hierarchical(
                    text,
                    max_chars,
                    Some(&[Some("\n\n")]),
                    overlap,
                    OverlapBoundary::Grapheme,
                );
                for &(s, e) in &chunks {
                    assert!(
                        &text[s..e] != "\n\n",
                        "a snapped window emitted the separator as chunk ({s}, {e}): \
                         m={max_chars} ov={overlap}: {chunks:?}"
                    );
                }
                let mut prev_end = 0usize;
                for &(_s, e) in &chunks {
                    assert!(e > prev_end, "ends not advancing: {chunks:?}");
                    prev_end = e;
                }
            }
        }
    }

    #[test]
    fn the_default_hierarchys_final_exit_is_unchanged() {
        // The default hierarchy never skipped at a final exit before
        // #103 (a paragraph gap is a cut whose resume is the next
        // paragraph's start, never a second match), and must not start:
        // the whole-remainder exit is unchanged for it, gap runs included.
        assert_eq!(
            super::chunk_hierarchical("ab\n\n\ncd", 2, None, 0, OverlapBoundary::Grapheme),
            vec![(0, 2), (5, 7)]
        );
        assert_eq!(
            super::chunk_hierarchical("ab\n\n\ncd", 2, Some(&[None]), 0, OverlapBoundary::Grapheme),
            vec![(0, 2), (5, 7)]
        );
        // A paragraph gap CAN open a window — via an overlap snap into
        // it — and then the skip applies (the gap is a cut with
        // next_start strictly past it): the final exit after such a snap
        // starts at the next paragraph, not inside the gap.
        let text = "ab\n\ncd";
        let snapped_into_gap =
            super::chunk_hierarchical(text, 6, None, 1, OverlapBoundary::Grapheme);
        for &(s, e) in &snapped_into_gap {
            assert_ne!(
                &text[s..e],
                "\n\n",
                "a paragraph gap came back as its own chunk: {snapped_into_gap:?}"
            );
        }
    }

    // ---- #47: the word-aware overlap snap ----

    #[test]
    fn word_mode_snaps_the_overlap_tail_to_a_word_boundary() {
        // The issue's own motivating shape: the grapheme snap starts the
        // tail mid-word ("uter Interaction"); the word snap moves the
        // candidate back to the word's first codepoint ("Computer ...").
        let text = "...Bachelor of Arts in Human-Computer Interaction, Lakeside \
                    College, 2018\n\nCapstone project: designing a better chunker \
                    for embedding pipelines and retrieval.";
        let seps: &[Option<&str>] = &[Some("\n## "), Some("\n# "), None];
        assert_eq!(
            super::chunk_hierarchical(text, 150, Some(seps), 40, OverlapBoundary::Grapheme),
            vec![(0, 73), (33, 158)]
        );
        assert_eq!(
            super::chunk_hierarchical(text, 150, Some(seps), 40, OverlapBoundary::Word),
            vec![(0, 73), (29, 158)],
            "the word snap must move the tail start from mid-word to the word edge"
        );
        // A smaller budget: both tails word-aligned where the grapheme
        // ones were mid-word ("teraction", "unker").
        assert_eq!(
            super::chunk_hierarchical(text, 60, Some(seps), 20, OverlapBoundary::Grapheme),
            vec![(0, 3), (3, 60), (40, 73), (75, 134), (114, 158)]
        );
        assert_eq!(
            super::chunk_hierarchical(text, 60, Some(seps), 20, OverlapBoundary::Word),
            vec![(0, 3), (3, 60), (38, 73), (75, 134), (112, 158)]
        );
    }

    #[test]
    fn word_mode_with_no_word_boundary_in_range_falls_back_to_the_grapheme_snap() {
        // One long token: the word level is the single segment (0, 100)
        // (its only cut at the text end, past every snap target), so no
        // boundary exists in the snap-back range — the plain grapheme
        // candidate is kept, byte-for-byte the grapheme mode's output.
        let text = format!("{}{}", "a".repeat(60), " b b b b");
        for max_chars in [10usize, 20, 37] {
            for overlap in [1usize, 6, max_chars - 1] {
                assert_eq!(
                    super::chunk_hierarchical(
                        &text,
                        max_chars,
                        None,
                        overlap,
                        OverlapBoundary::Word
                    ),
                    super::chunk_hierarchical(
                        &text,
                        max_chars,
                        None,
                        overlap,
                        OverlapBoundary::Grapheme
                    ),
                    "the word snap invented a boundary inside one long token: \
                     m={max_chars} ov={overlap}"
                );
            }
        }
        // Dense CJK (every Han character its own UAX #29 word) and Thai
        // (no dictionary: one run, no internal boundary) both agree with
        // grapheme mode — the first because word and grapheme boundaries
        // coincide, the second through the documented fallback.
        let cjk = "中文数据段落。中文数据段落。".repeat(5);
        for max_chars in [7usize, 11, 30] {
            for overlap in [1usize, 2, max_chars - 1] {
                assert_eq!(
                    super::chunk_hierarchical(
                        &cjk,
                        max_chars,
                        None,
                        overlap,
                        OverlapBoundary::Word
                    ),
                    super::chunk_hierarchical(
                        &cjk,
                        max_chars,
                        None,
                        overlap,
                        OverlapBoundary::Grapheme
                    ),
                    "CJK word mode diverged from grapheme mode: m={max_chars} ov={overlap}"
                );
            }
        }
        let thai = "กาลครั้งหนึ่งนานาพรบ์มาแล้ว ".repeat(6);
        for max_chars in [13usize, 20] {
            for overlap in [3usize, 5] {
                assert_eq!(
                    super::chunk_hierarchical(
                        &thai,
                        max_chars,
                        None,
                        overlap,
                        OverlapBoundary::Word
                    ),
                    super::chunk_hierarchical(
                        &thai,
                        max_chars,
                        None,
                        overlap,
                        OverlapBoundary::Grapheme
                    ),
                    "Thai word mode diverged from grapheme mode: m={max_chars} ov={overlap}"
                );
            }
        }
    }

    #[test]
    fn word_mode_never_snaps_before_the_previous_chunk_or_contained_in_it() {
        // Constraint (a): the word snap may push the candidate back past
        // the just-emitted chunk's own start; the decline-the-snap
        // lookahead (#83) runs AFTER the word snap (the composition
        // order) and declines exactly that — the transition degrades to
        // zero overlap, no chunk is emitted from a candidate at or before
        // its predecessor's start, and no chunk is ever strictly contained
        // in (or identical to) its predecessor. Swept over every overlap
        // of two budgets on word-run text.
        let text = "aaaa bbbb cccc dddd eeee ffff gggg hhhh";
        for max_chars in [8usize, 9, 10, 12] {
            for overlap in 1..max_chars {
                let chunks = super::chunk_hierarchical(
                    text,
                    max_chars,
                    None,
                    overlap,
                    OverlapBoundary::Word,
                );
                for w in chunks.windows(2) {
                    let (prev_start, prev_end) = (w[0].0, w[0].1);
                    let (next_start, next_end) = (w[1].0, w[1].1);
                    assert!(
                        next_start > prev_start,
                        "starts not strictly increasing: m={max_chars} ov={overlap}: {chunks:?}"
                    );
                    assert!(
                        next_end > prev_end,
                        "ends not strictly advancing: m={max_chars} ov={overlap}: {chunks:?}"
                    );
                    assert!(
                        !(next_start >= prev_start && next_end <= prev_end),
                        "a chunk contained in its predecessor: m={max_chars} \
                         ov={overlap}: {chunks:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn word_mode_is_a_noop_at_overlap_zero_and_inert_at_the_default() {
        // Constraint (b): "word" with overlap=0 is accepted and does
        // nothing (no snap site ever runs, so the output is the
        // zero-overlap answer); the grapheme default is the same function
        // it always was.
        let text = "one two three four five six seven eight nine ten eleven twelve";
        for max_chars in [5usize, 12, 20] {
            let zero =
                super::chunk_hierarchical(text, max_chars, None, 0, OverlapBoundary::Grapheme);
            assert_eq!(
                super::chunk_hierarchical(text, max_chars, None, 0, OverlapBoundary::Word),
                zero,
                "word mode changed a zero-overlap answer: m={max_chars}"
            );
        }
        // And the default-hierarchy word-mode output at overlap > 0 moves
        // at least one tail on word-run text (the cheap drift canary; the
        // exact values are pinned in word_mode_snaps_the_overlap_tail…):
        // the sweep pins the full semantics against the oracle.
        assert_ne!(
            super::chunk_hierarchical(text, 9, None, 8, OverlapBoundary::Word),
            super::chunk_hierarchical(text, 9, None, 8, OverlapBoundary::Grapheme),
            "word mode never moved a tail on word-run text — the snap is inert?"
        );
    }

    #[test]
    fn word_mode_snap_at_a_separator_run_boundary_composes_with_the_103_skip() {
        // The #103/#47 interaction: a word snap landing on (or inside) a
        // separator run, under a budget whose final window then opens on
        // a match — the skip must preempt the final exit exactly as it
        // does in grapheme mode, and the word boundary the snap lands on
        // must not resurrect the separator as a chunk. Swept: separator
        // run texts × every legal overlap × both boundary modes, no
        // chunk is the separator, ends always advance.
        let text = "alpha\n\nbeta\n\ngamma\n\ndelta";
        for max_chars in [6usize, 9, 12] {
            for overlap in 1..max_chars {
                let word = super::chunk_hierarchical(
                    text,
                    max_chars,
                    Some(&[Some("\n\n")]),
                    overlap,
                    OverlapBoundary::Word,
                );
                for &(s, e) in &word {
                    assert!(
                        &text[s..e] != "\n\n",
                        "word mode emitted the separator as a chunk: m={max_chars} \
                         ov={overlap}: {word:?}"
                    );
                }
                let mut prev_end = 0usize;
                for &(_s, e) in &word {
                    assert!(e > prev_end, "ends not advancing: {word:?}");
                    prev_end = e;
                }
            }
        }
        // And the word snap never lands mid-cluster even beside a
        // separator: every chunk start/end is a grapheme boundary (the
        // word level's cuts are the same filtered list the windows cut
        // on).
        let word_starts: HashSet<usize> =
            super::chunk_hierarchical(text, 9, Some(&[Some("\n\n")]), 5, OverlapBoundary::Word)
                .iter()
                .map(|&(s, _)| s)
                .collect();
        for s in word_starts {
            let g = crate::truncate_impl::grapheme_boundary_chars(text);
            assert!(
                g.contains(&s),
                "word-snapped start {s} is not a grapheme boundary"
            );
        }
    }

    #[test]
    fn word_mode_realizes_the_word_level_lazily_and_at_most_once() {
        // The seam pins, #47's performance constraint: the word level is
        // realized only when a snap consults it — never eagerly, never
        // twice. (a) grapheme mode with overlap (snap sites galore)
        // realizes no word level; (b) word mode with overlap realizes it
        // exactly once, shared with any window that descends; (c) word
        // mode at overlap=0 realizes none at all (no snap site runs).
        let text = "one two three four five six seven eight nine ten eleven twelve";
        build_seam::reset();
        let _ = super::chunk_hierarchical(text, 7, None, 2, OverlapBoundary::Grapheme);
        let grapheme_levels = build_seam::levels_built();
        assert!(
            grapheme_levels <= 3,
            "grapheme mode built extra levels: {grapheme_levels}"
        );

        build_seam::reset();
        let _ = super::chunk_hierarchical(text, 7, None, 2, OverlapBoundary::Word);
        let word_levels = build_seam::levels_built();
        assert!(
            word_levels <= grapheme_levels + 1,
            "word mode built the word level more than once (or something \
             beyond it): grapheme={grapheme_levels} word={word_levels}"
        );

        build_seam::reset();
        let zero_word = super::chunk_hierarchical(text, 7, None, 0, OverlapBoundary::Word);
        let zero_word_levels = build_seam::levels_built();
        build_seam::reset();
        let zero_grapheme = super::chunk_hierarchical(text, 7, None, 0, OverlapBoundary::Grapheme);
        assert_eq!(
            build_seam::levels_built(),
            zero_word_levels,
            "word mode at overlap=0 must build exactly what grapheme mode \
             builds (windows only, no snap sites run at overlap=0)"
        );
        assert_eq!(zero_word, zero_grapheme);

        // The grapheme-mode cost is unchanged by the feature existing:
        // the same call the pre-#47 pins ran builds the same levels —
        // the word-bounds walk is behind the mode flag, not in the
        // grapheme path (the whole-document-budget pin above continues
        // to hold: zero builds).
        build_seam::reset();
        let _ = super::chunk_hierarchical(text, 70, None, 0, OverlapBoundary::Grapheme);
        assert_eq!(build_seam::levels_built(), 0);
        build_seam::reset();
        let _ = super::chunk_hierarchical(text, 70, None, 0, OverlapBoundary::Word);
        assert_eq!(
            build_seam::levels_built(),
            0,
            "word mode realized a level for a single-chunk overlap=0 call"
        );
    }

    // ---- The lazy levels (#30): seam-counted structural pins. The
    // differential sweep above pins lazy == eager output; these pin the
    // laziness itself: which levels a call builds, how often, and
    // whether the grapheme bitmap is built at all, through the
    // per-thread `build_seam` counters, reset at each test's start.

    #[test]
    fn whole_document_budget_builds_no_levels_and_no_grapheme_index() {
        // The issue's own headline cell: a single-chunk budget
        // (`max_chars >= total`) pays no level walk at all: the eager
        // spelling built all three default levels (and the grapheme
        // bitmap, via the pre-loop cut filter) before the loop's first
        // iteration could take its own `remaining <= max_chars` exit,
        // ~176 ms and ~45 MiB over a 6 MiB document for levels that
        // supplied zero cuts. Every hierarchy spelling of the same call
        // is equally lazy: default, custom literal, and None-spliced.
        let text = "Para one.\n\nPara two.\n\nPara three.\n\nPara four.";
        let total = text.chars().count();
        for seps in [
            None,
            Some(vec![Some("ZZZ_NEVER_MATCHES")]),
            Some(vec![Some("\n"), None]),
        ] {
            build_seam::reset();
            let chunks = chunk_hierarchical(text, total, seps.as_deref(), 0);
            assert_eq!(chunks, vec![(0, total)], "seps={seps:?}");
            assert_eq!(
                build_seam::levels_built(),
                0,
                "a single-chunk budget built levels: seps={seps:?}"
            );
            assert_eq!(
                build_seam::grapheme_index_built(),
                0,
                "a single-chunk budget built the grapheme index: seps={seps:?}"
            );
        }
        // Overlap cannot resurrect the cost either: the single chunk
        // exits the loop before any snap site runs.
        build_seam::reset();
        assert_eq!(chunk_hierarchical(text, total, None, 4), vec![(0, total)]);
        assert_eq!(build_seam::levels_built(), 0);
        assert_eq!(build_seam::grapheme_index_built(), 0);
    }

    #[test]
    fn a_budget_that_never_descends_builds_only_the_paragraph_level() {
        // A paragraph-sized budget over multi-paragraph text: every
        // window's first consultation is the paragraph level and it
        // supplies every cut, so sentence and word are never consulted,
        // never built, across however many windows the text windows
        // into; the chunk ends landing on paragraph ends is the
        // output-level proof the cuts came from the paragraph level.
        let text = "Short one.\n\nShort two.\n\nShort three.\n\nShort four.";
        build_seam::reset();
        let chunks = chunk_hierarchical(text, 12, None, 0);
        assert_eq!(
            text_of(&chunks, text),
            vec!["Short one.", "Short two.", "Short three.", "Short four."]
        );
        assert_eq!(
            build_seam::levels_built(),
            1,
            "only the paragraph level may build for a never-descending budget"
        );
        // The one built level had cuts to filter, so the bitmap built
        // exactly once too, at realization, not in any pre-loop pass.
        assert_eq!(build_seam::grapheme_index_built(), 1);
    }

    #[test]
    fn a_budget_that_descends_builds_each_level_at_most_once_per_call() {
        // One paragraph, one sentence, many words, a word-sized budget:
        // every window consults paragraph (a lone paragraph has no gap
        // to cut at), then sentence (its only cut sits past the
        // window's limit), then word (which supplies the cut), the
        // full descent, window after window, and the counts stay at
        // three: the first window realized all three levels and every
        // later window reuses the memoized builds.
        let text = "one two three four five six seven eight nine ten eleven twelve";
        build_seam::reset();
        let chunks = chunk_hierarchical(text, 7, None, 0);
        assert!(
            chunks.len() > 10,
            "expected many windows, got {}",
            chunks.len()
        );
        assert_eq!(
            build_seam::levels_built(),
            3,
            "each of paragraph/sentence/word must build exactly once"
        );
        assert_eq!(
            build_seam::grapheme_index_built(),
            1,
            "the filter bitmap is shared: one build, however many levels filter"
        );
        // Overlap adds snap sites, not builds: the bitmap is already
        // realized by the word level's filter, the levels already
        // memoized.
        build_seam::reset();
        let _ = chunk_hierarchical(text, 7, None, 2);
        assert_eq!(build_seam::levels_built(), 3);
        assert_eq!(build_seam::grapheme_index_built(), 1);
        // Per-call scope: a fresh call pays the walks again; the
        // memoization lives in the call's level slots, not a cache.
        build_seam::reset();
        let _ = chunk_hierarchical(text, 7, None, 0);
        assert_eq!(build_seam::levels_built(), 3);
    }

    #[test]
    fn a_never_descended_splice_builds_none_of_its_levels() {
        // The splice cost, seam-pinned from the cheap side:
        // `["\n", None]` over lines that all fit the budget (every
        // window's first consultation is the "\n" literal and it
        // supplies every cut) builds the literal level and none of the
        // three spliced specs. The eager spelling paid paragraph/
        // sentence/word walks for exactly this call shape.
        let text = "l0\nl1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9";
        build_seam::reset();
        let chunks = chunk_hierarchical(text, 5, Some(&[Some("\n"), None]), 0);
        assert!(chunks.len() >= 5);
        for &(s, e) in &chunks {
            assert!(e - s <= 5, "chunk {s}..{e} broke the budget");
        }
        assert_eq!(
            build_seam::levels_built(),
            1,
            "only the literal level may build; the spliced specs must stay unrealized"
        );
        assert_eq!(build_seam::grapheme_index_built(), 1);
    }

    #[test]
    fn raw_cut_fallback_builds_the_grapheme_index_once_not_per_window() {
        // The bitmap's other lazy build site: a hierarchy with no
        // levels at all (an empty separators list) under a small budget
        // builds zero levels and exactly one grapheme index, at the
        // first window's raw-cut fallback, reused by every later window
        // and by the overlap snap.
        let text = "abcdefghij klmno pqrstu";
        build_seam::reset();
        let chunks = chunk_hierarchical(text, 5, Some(&[]), 2);
        assert!(chunks.len() > 1);
        for &(s, e) in &chunks {
            assert!(e - s <= 5);
        }
        assert_eq!(build_seam::levels_built(), 0);
        assert_eq!(build_seam::grapheme_index_built(), 1);
    }

    #[test]
    fn duplicate_entries_build_once_not_once_per_entry_after_consultation() {
        // The dedups' structural pin, seam-counted: the output-equality
        // sweep in the None-splice contract tests pins that duplicates
        // never change an answer, but the undeduped spelling passes
        // that pin too; what it cannot see is the walks a duplicate
        // would re-pay. The find_map only ever
        // reaches a duplicate after its original returned None for the
        // current window, so the pin needs a window that exhausts the
        // whole (deduped) hierarchy to the raw-cut fallback: the
        // unbroken "x" run below supplies it. Eightfold [" "] must then
        // build exactly one literal level (not eight identical scans
        // and cut vectors, the 1.69 GiB shape the dedup closed), and
        // [None, None] exactly the three spliced levels (not six; a
        // second None after a consulted splice adds no builds, the pin
        // the output sweep cannot express). Each duplicated spelling's
        // output must equal its deduped spelling's, the same inertness
        // one level deeper.
        let text = format!("ab cd {}", "x".repeat(40));
        build_seam::reset();
        let eight = chunk_hierarchical(&text, 4, Some(&[Some(" "); 8]), 0);
        assert_eq!(
            build_seam::levels_built(),
            1,
            "eightfold [\" \"] must build one literal level, not one per entry"
        );
        assert_eq!(build_seam::grapheme_index_built(), 1);
        assert_eq!(
            eight,
            chunk_hierarchical(&text, 4, Some(&[Some(" ")]), 0),
            "the duplicated literal list changed the answer"
        );

        build_seam::reset();
        let pair = chunk_hierarchical(&text, 4, Some(&[None, None]), 0);
        assert_eq!(
            build_seam::levels_built(),
            3,
            "[None, None] must build the three spliced levels once, not twice"
        );
        assert_eq!(build_seam::grapheme_index_built(), 1);
        assert_eq!(
            pair,
            chunk_hierarchical(&text, 4, Some(&[None]), 0),
            "the duplicated None list changed the answer"
        );
    }

    #[test]
    fn mixed_duplicate_literals_and_none_entries_collapse_to_each_distinct_entry() {
        // The two dedups together on one list: literals dedup against
        // literals across any distance (the seen-set is whole-list,
        // spanning the spliced specs between them), the None splice
        // against the first splice, and the two rules compose:
        // ["\n", None, "\n", None, " ", None, " "] is exactly
        // ["\n", None, " "] as a slot list: one "\n" literal, the three
        // spliced specs, one " " literal. The line-shaped text (the
        // "\n" matches, so the pin is not vacuous) with an
        // unbroken "x" run forces a window that exhausts the whole
        // hierarchy to the raw cut, consulting every slot, so the seam
        // count is the deduped list's own length: five.
        let text = format!("l0 l1\nl2 {}", "x".repeat(40));
        let mixed: &[Option<&str>] = &[
            Some("\n"),
            None,
            Some("\n"),
            None,
            Some(" "),
            None,
            Some(" "),
        ];
        let deduped: &[Option<&str>] = &[Some("\n"), None, Some(" ")];
        build_seam::reset();
        let chunks = chunk_hierarchical(&text, 5, Some(mixed), 0);
        assert_eq!(
            build_seam::levels_built(),
            5,
            "the mixed duplicate list must build its five distinct levels, \
             not one slot per entry (thirteen)"
        );
        assert_eq!(build_seam::grapheme_index_built(), 1);
        assert_eq!(
            chunks,
            chunk_hierarchical(&text, 5, Some(deduped), 0),
            "the mixed duplicate list changed the answer"
        );
    }

    #[test]
    fn custom_separator_matching_inside_a_cluster_never_cuts_there() {
        // "0" + SARA AM is one grapheme cluster; a literal separator that
        // matches the SARA AM codepoint alone would cut between the two
        // codepoints of that cluster. The filter must drop every such
        // match: the case the filter exists for, pinned because the
        // bitmap membership query is what answers it now.
        let text = "0\u{0E33} 0\u{0E33} 0\u{0E33}";
        let boundaries: HashSet<usize> = grapheme_boundary_chars(text).into_iter().collect();
        for max_chars in 1..text.chars().count() {
            let chunks = chunk_hierarchical(text, max_chars, Some(&[Some("\u{0E33}")]), 0);
            assert!(!chunks.is_empty(), "no chunks at max_chars={max_chars}");
            for &(s, e) in &chunks {
                assert!(
                    boundaries.contains(&s) && boundaries.contains(&e),
                    "chunk ({s}, {e}) splits a cluster at max_chars={max_chars}"
                );
            }
        }
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
        let chunks = chunk_hierarchical(
            text,
            40,
            Some(&[Some("\n## "), Some("\n\n"), Some(". "), Some(" ")]),
            0,
        );
        let texts = text_of(&chunks, text);
        assert_eq!(texts[0], "# Title\nintro text");
        assert!(texts.iter().any(|t| t.starts_with("Section")));
    }

    #[test]
    fn always_produces_some_chunk_via_raw_cut_fallback() {
        // A single "word" with no boundary anywhere and no separator
        // match at all: only the grapheme-safe raw cut can produce chunks.
        let text = "a".repeat(100);
        let chunks = chunk_hierarchical(&text, 10, Some(&[Some("XYZ_NEVER_MATCHES")]), 0);
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

    // ---- The None entry: splicing the accurate hierarchy into a custom list ----

    #[test]
    fn none_entry_alone_reproduces_the_default_hierarchy_exactly() {
        // [None] is separators=None, and a never-matching literal above a
        // None entry changes nothing (it can never supply a cut, so the
        // spliced levels answer every window the default hierarchy would):
        // exact-equality properties over the whole differential corpus and
        // a budget sweep, far stronger than a couple of spot cases.
        // Duplicate inertness (of `None` and of literals, both deduped at
        // slot construction) is the same argument one level deeper: a
        // duplicate's level is identical to its first occurrence's, so
        // the find_map can never reach a duplicate that changes an answer
        // its original didn't already give: [None, None] is [None], and
        // the duplicated line-first list is the unduplicated one (the
        // corpus texts contain "\n", so the literal matches:
        // the pin is meaningful, not vacuous). That inertness is also
        // overlap-independent: the dedup happens in list construction,
        // before any windowing or overlap snap, so every duplicate shape
        // is swept over the boundary-adjacent overlaps the bitmap
        // differential above uses, and an eightfold run pins that longer
        // duplicate lists stay inert, free now that the splice is built
        // once per call regardless of list length. (The dedup's cost,
        // what this output-equality sweep cannot see since the
        // undeduped spelling passes it too, is pinned by the timing
        // cells in tests/test_performance.py.)
        for text in differential_corpus() {
            let total = text.chars().count();
            for max_chars in 1..=total.min(48) {
                for overlap in [0usize, 1, max_chars.saturating_sub(1)]
                    .into_iter()
                    .filter(|&o| o < max_chars)
                {
                    assert_eq!(
                        chunk_hierarchical(&text, max_chars, Some(&[None]), overlap),
                        chunk_hierarchical(&text, max_chars, None, overlap),
                        "text={text:?} max_chars={max_chars} overlap={overlap}"
                    );
                    assert_eq!(
                        chunk_hierarchical(
                            &text,
                            max_chars,
                            Some(&[Some("ZZZ_NEVER_MATCHES"), None]),
                            overlap
                        ),
                        chunk_hierarchical(&text, max_chars, None, overlap),
                        "text={text:?} max_chars={max_chars} overlap={overlap}"
                    );
                    // Duplicate None entries, pairwise and eightfold:
                    // skipped at construction, inert under overlap.
                    assert_eq!(
                        chunk_hierarchical(&text, max_chars, Some(&[None, None]), overlap),
                        chunk_hierarchical(&text, max_chars, Some(&[None]), overlap),
                        "duplicate None changed the answer: text={text:?} \
                         max_chars={max_chars} overlap={overlap}"
                    );
                    assert_eq!(
                        chunk_hierarchical(&text, max_chars, Some(&[None; 8]), overlap),
                        chunk_hierarchical(&text, max_chars, Some(&[None]), overlap),
                        "eightfold None changed the answer: text={text:?} \
                         max_chars={max_chars} overlap={overlap}"
                    );
                    // The duplicated line-first list, duplicate literals
                    // in every position: the same dominance argument for
                    // the `Some(s)` arm the dedup now covers.
                    assert_eq!(
                        chunk_hierarchical(
                            &text,
                            max_chars,
                            Some(&[Some("\n"), None, Some("\n"), None]),
                            overlap
                        ),
                        chunk_hierarchical(&text, max_chars, Some(&[Some("\n"), None]), overlap),
                        "duplicated [\"\\n\", None] changed the answer: text={text:?} \
                         max_chars={max_chars} overlap={overlap}"
                    );
                    assert_eq!(
                        chunk_hierarchical(
                            &text,
                            max_chars,
                            Some(&[Some("\n"), Some("\n")]),
                            overlap
                        ),
                        chunk_hierarchical(&text, max_chars, Some(&[Some("\n")]), overlap),
                        "duplicated literal changed the answer: text={text:?} \
                         max_chars={max_chars} overlap={overlap}"
                    );
                    assert_eq!(
                        chunk_hierarchical(
                            &text,
                            max_chars,
                            Some(&[Some(" "), Some("\n"), Some(" ")]),
                            overlap
                        ),
                        chunk_hierarchical(
                            &text,
                            max_chars,
                            Some(&[Some(" "), Some("\n")]),
                            overlap
                        ),
                        "mid-list duplicated literal changed the answer: text={text:?} \
                         max_chars={max_chars} overlap={overlap}"
                    );
                }
            }
        }
    }

    #[test]
    fn line_first_splice_cuts_an_oversized_line_at_real_sentence_boundaries() {
        // The motivating shape for the None entry: a chat thread, one
        // message per line, never split mid-line, and an oversized
        // message falling back to the accurate UAX #29 sentence level.
        // Every cut this budget produces must land either at a line break
        // or at a real sentence boundary, never anywhere else.
        let text = "Nathan: kicking off.\nPriya: We briefed the U.S. team on the numbers. \
                    They asked for a follow-up meeting. The budget holds.\nNathan: done.";
        let seps: &[Option<&str>] = &[Some("\n"), None];
        let chunks = chunk_hierarchical(text, 60, Some(seps), 0);
        assert!(chunks.len() > 1, "expected the oversized line to split");
        let chars: Vec<char> = text.chars().collect();
        let sentence_ends: Vec<usize> = segmentation_impl::sentence_bounds(text)
            .into_iter()
            .map(|(_, end)| end)
            .collect();
        for &(_, end) in &chunks {
            if end == chars.len() {
                continue; // the final chunk runs to the end untrimmed
            }
            let at_line_break = chars[end] == '\n';
            let at_sentence_end = sentence_ends.contains(&end);
            assert!(
                at_line_break || at_sentence_end,
                "cut at {end} is neither a line break nor a sentence boundary: {chunks:?}"
            );
        }
        // The name "U.S. team" survives whole in some chunk: the sentence
        // level's SB6-SB8 rules do not break after "U.S." the way a naive
        // ends-with-punctuation rule does.
        let joined: Vec<String> = chunks
            .iter()
            .map(|&(s, e)| chars[s..e].iter().collect())
            .collect();
        assert!(
            joined.iter().any(|p| p.contains("U.S. team")),
            "no chunk holds the name whole: {joined:?}"
        );
    }

    #[test]
    fn all_literal_fallbacks_sever_where_the_none_splice_does_not() {
        // The contrast the None entry exists for, pinned from both sides:
        // the same thread and a budget below the first sentence boundary.
        // The all-literal [\"\\n\", \". \", \" \"] list's \". \" level has a
        // match after \"U.S.\" (a mid-name non-sentence), so it severs the
        // name; the spliced [\"\\n\", None] list has no in-budget sentence
        // cut there and falls to the word level, whose cuts are UAX #29
        // word boundaries: every piece still ends at a word boundary, and
        // the word walk never lands inside the name's letter sequence.
        let text = "Nathan: kicking off.\nPriya: We briefed the U.S. team on the numbers. \
                    They asked for a follow-up meeting. The budget holds.\nNathan: done.";
        let naive = chunk_hierarchical(text, 40, Some(&[Some("\n"), Some(". "), Some(" ")]), 0);
        let naive_pieces: Vec<String> = naive
            .iter()
            .map(|&(s, e)| text.chars().skip(s).take(e - s).collect())
            .collect();
        assert!(
            naive_pieces.iter().any(|p| p.ends_with("U.S")),
            "expected the naive literal list to sever the name (the '. ' match \
             after 'U.S.' drops the period as separator): {naive_pieces:?}"
        );

        let spliced = chunk_hierarchical(text, 40, Some(&[Some("\n"), None]), 0);
        let spliced_pieces: Vec<String> = spliced
            .iter()
            .map(|&(s, e)| text.chars().skip(s).take(e - s).collect())
            .collect();
        for piece in &spliced_pieces {
            assert!(
                !piece.ends_with("U.S") && !piece.ends_with("U.S."),
                "the spliced hierarchy severed the name: {spliced_pieces:?}"
            );
        }
        // And the name rides whole inside one piece (the word-level cuts
        // walk past it, never through it).
        assert!(
            spliced_pieces.iter().any(|p| p.contains("U.S. team")),
            "no chunk holds the name whole: {spliced_pieces:?}"
        );
        // Both spellings still respect the budget and make progress.
        for chunks in [&naive, &spliced] {
            for &(s, e) in chunks {
                assert!(e - s <= 40);
            }
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
        // actual shared content between consecutive chunks, not merely be
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
    fn overlapped_chunk_is_never_strictly_inside_its_predecessor() {
        // #83's hierarchical shape, pinned: the snapped start used to
        // resolve back to the same cut, emitting (2, 4) strictly inside
        // (0, 4). The decline-the-snap lookahead now drops the overlap for
        // exactly that transition (start = cut.1).
        assert_eq!(
            chunk_hierarchical("aaa bbbbbbbb", 5, None, 2),
            vec![(0, 4), (4, 9), (7, 12)]
        );
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
        let chunks = chunk_hierarchical(&text, 3, Some(&[Some("NEVER")]), 1);
        assert!(chunks.len() >= 166);
    }
}
