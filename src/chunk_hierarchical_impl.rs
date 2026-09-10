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
//! A `None` ENTRY in an otherwise-literal list splices the default
//! hierarchy's three accurate levels in AT THAT POSITION: the mix the
//! all-or-nothing custom list could not express before. The motivating
//! shape is line-oriented text that must never split mid-line but whose
//! oversized lines still deserve ACCURATE fallback cuts — a chat thread,
//! one message per line: `["\n", None]` is line → paragraph → sentence →
//! word → raw cut, where the sentence/word levels below the line level
//! are the real UAX #29 segmenters, not the `". "`/`" "` literal guesses
//! an all-literal `["\n", ". ", " "]` list would pin them to (the guesses
//! cut inside "U.S. team"; the segmenters do not — the exact reason the
//! default hierarchy exists). `[None]` is therefore identical to
//! `separators = None`. Cost stated plainly: every level — each of the
//! three default walks, each distinct custom literal — pays its one
//! whole-text walk at most once per call, and only when a window
//! actually consults it (#30's lazy levels: the splice contributes
//! three level SPECS, not three whole-text walks, a literal's slot
//! waits unbuilt the same way, and realization happens on first
//! consultation — the same at-most-once-per-call memoization #24's
//! lazy bitmap applies), so `["\n", None]` over a thread whose every
//! line fits the budget builds NONE of the spliced levels, and one
//! whose oversized lines fall to the sentence level pays sentence's
//! walk once and word's never. Duplicate entries — a second or later
//! `None`, a repeated literal — are recognized at slot construction
//! and skipped, inert by the `find_map` dominance argument (identical
//! levels can never change the answer: the first occurrence of a level
//! always dominates its duplicate), so neither `[None; 100]` nor
//! `[" "; 100]` is the caller-controlled unbounded cost the per-entry
//! spelling made them (every duplicate re-paid the walks plus a full
//! cut vector, ~45 MiB per duplicate on a 6 MiB document — an OOM
//! shape); the dedups close both spellings under the #21
//! pathological-input discipline, memoized like #24's lazy bitmap. The
//! former spellings were eager on both axes on top of that: the
//! pre-#31 one re-paid all three walks per `None` entry outright, and
//! #31's dedup still paid the three walks once per call even when no
//! window ever consulted them (a single-chunk budget over 6 MiB spent
//! ~176 ms and ~45 MiB on levels that supplied zero cuts); the lazy
//! realization removes the paid-for-nothing cost.
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
//! Performance: every CONSULTED level's candidate cut-position list is
//! computed at most once per call, at the FIRST window that consults it
//! (#30's lazy levels: one scan per DISTINCT level — `paragraph_bounds`/
//! `sentence_bounds`/`word_bounds` for the default levels, one `memmem`
//! pass per custom literal — never re-scanned per chunk, never scanned
//! at all for a level no window descends to, and never repeated for a
//! duplicate entry the slot-construction dedup already collapsed; a
//! single-chunk budget, `max_chars >= total`, consults no level and
//! walks nothing: one codepoint count, one chunk out). Building each chunk is one
//! `partition_point` binary search per level: O(n × levels) total, levels
//! bounded by the small, caller-supplied list length. The codepoint `total`
//! the budget arithmetic needs is [`char_count`]'s one ASCII-first pass
//! (pure-ASCII text answers its own byte length; non-ASCII text pays the
//! branchless non-continuation-byte count — [`char_count`]'s own docs carry
//! why the split exists), not a materialized `Vec<char>`. The one other
//! whole-text structure is the grapheme boundary
//! index (see [`GraphemeIndex`]): ONE `graphemes(true)` walk emitting one
//! bit per codepoint on non-ASCII text — or, on pure-ASCII text, an
//! all-ones bitmap plus the CRLF fixup, two SIMD byte scans, no
//! segmentation walk at all — built at most once per call and ONLY when a
//! call actually needs it: a realized level with cuts to filter, the
//! raw-cut fallback, or an overlap snap. A call that realizes no level
//! with cuts and never falls back or snaps — the whole-document-budget
//! case, custom hierarchy or default — builds none of it. The former
//! spelling paid an unconditional `Vec<char>` collect plus a
//! `HashSet<usize>` of every grapheme boundary in the document BEFORE
//! anything else could run, which dominated document-scale cost and was
//! superlinear on top of it (#22: a 12 MiB document spent ~1.2 s in
//! ~12.6M hashed inserts regardless of chunk budget or separator
//! presence); until #30, every level was additionally built — and every
//! non-empty one filtered — before the first window, so a budget that
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

/// A not-yet-built level: WHICH walk or scan realizes it, not the walk
/// itself. The default hierarchy's three levels and a `None` splice's
/// copies exist as specs until the first window that consults them;
/// a custom literal exists as a spec until the same first
/// consultation. Realization (one whole-text pass: `paragraph_bounds`,
/// UAX #29 `sentence_bounds`, UAX #29 `word_bounds`, or one `memmem`
/// scan for [`LevelSpec::Literal`]) happens in [`LevelSlot::realize`],
/// never here — building a spec list walks no text at all.
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
/// `None` until the FIRST `best_cut` consultation and holds the build
/// for the rest of the call — the same at-most-once-per-call
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
    /// at most one realization per slot per call — `get_or_insert_with`
    /// on the slot, the same memoization idiom #24's lazy grapheme
    /// bitmap uses, extended to the levels themselves. Realization is
    /// the level's one whole-text walk or scan, then — for a level with
    /// cuts — the grapheme cut filter, exactly the filter semantics the
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
            // input). The default hierarchy is NOT exempt:
            // `word_bounds`/`sentence_bounds` follow UAX #29 exactly, which
            // scores some combining sequences (Thai SARA AM, U+0E33) as
            // their own word/sentence segment even though the grapheme
            // rules join them to the preceding base character into one
            // cluster — `truncate_impl`'s module docs document the same
            // divergence — so their cuts need this filter exactly like a
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

/// The default hierarchy's three specs, in coarsest-first order — the
/// splice a `None` entry in a custom `separators` list inserts at its
/// position, and the whole hierarchy when `separators` is `None`.
/// SPECS, not levels: no text is walked here, which is the point — the
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
/// one [`LevelSpec::Literal`] waiting for its scan — one slot per
/// DISTINCT literal, a repeated literal skipped as a duplicate the same
/// way a second `None` is (see the arms below for why a duplicate is
/// provably inert); a `None` entry splices [`default_level_specs`] in
/// at its position, at most once per call. An empty literal is a no-op
/// level (dropped, no slot at all, the same filter the all-literal
/// spelling always applied — and no seen-set entry either, since it
/// never produces a slot). Both dedups are provably inert by the same
/// `find_map` dominance argument: the window loop consults `slots`
/// only through the find_map (the FIRST slot supplying a cut wins, and
/// the list is never indexed positionally or counted), and a
/// duplicate's level is bit-identical to its original's, so a
/// duplicate can never change an answer its original didn't already
/// give. Before the dedups, every duplicate was a re-paid whole-text
/// walk plus a full cut vector — ~45 MiB per duplicate `None` on a
/// 6 MiB document, the same for a repeated literal that matches often
/// (`[" "; 100]` was the OOM shape `[None; 100]` was) — a
/// caller-controlled unbounded cost the #21 pathological-input
/// discipline closes here, memoized like #24's lazy bitmap. No level
/// is BUILT here — the whole list is specs, realized per slot on
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
                // find_map (the FIRST slot supplying a cut wins, and
                // the list is never indexed positionally or counted),
                // and a duplicate's levels are bit-identical to the
                // first splice's, so a duplicate can never change an
                // answer its original didn't already give — inert.
                // Skipping still matters under the lazy spelling: a
                // duplicate slot a window reached would REALIZE (pay
                // its walk a second time) before returning the same
                // None its original just did, and the pre-#31 eager
                // spelling re-paid the three whole-text walks and
                // their ~45 MiB of cut vectors (on a 6 MiB document)
                // per duplicate outright — a caller-controlled
                // unbounded cost (`[None; 100]` is an OOM shape) — the
                // same pathological-input discipline #21 established:
                // contribute once per call, memoized like #24's lazy
                // bitmap.
            }
            // Dropped BEFORE the seen-set: an empty literal never
            // produces a slot, so it must not occupy a seen-set entry
            // either (a later non-empty literal is still first-seen,
            // and any number of empty literals stays a no-op).
            Some("") => {}
            Some(s) => {
                // A duplicate literal is skipped for the same
                // dominance reason as a duplicate splice: the find_map
                // only ever reaches it after its original returned
                // None for this window, and its level is bit-identical
                // to the original's — the same None again, inert. The
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

/// The call's grapheme boundary index, built at the FIRST need: the
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

/// Hierarchical fallback chunking of `text`: `(start, end)` codepoint-unit
/// pairs, each chunk at most `max_chars` codepoints, cut at the COARSEST
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
/// contract. `overlap` snaps the next
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
    separators: Option<&[Option<&str>]>,
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
    // The codepoint count as `char_count`'s ASCII-first pass — pure-ASCII
    // text answers its own byte length (the `is_ascii` gate, which LLVM
    // vectorizes outright), non-ASCII text pays the branchless count of
    // non-continuation bytes — instead of the former whole-text `Vec<char>`
    // collect (4 bytes per codepoint materialized before anything else
    // could run, the same O(source) allocation class #17 removed from
    // `is_grounded_fuzzy` and `chunk_text`'s grapheme grid before that).
    // `total` was the only thing that collect was ever read for; see
    // `char_count`'s own docs for the fast-path split's rationale.
    let total = char_count(text);

    // The hierarchy as SPECS — no walk paid yet. Every slot is realized
    // by the FIRST window that consults it ([`LevelSlot::realize`], at
    // most once per call per slot), so a budget whose windows never
    // open a level pays nothing for it: the #30 lazy-level extension of
    // #24's lazy-bitmap idiom. The former spelling built every level —
    // and filtered every non-empty one, building the grapheme bitmap —
    // before the first window, so even a single-chunk budget
    // (`max_chars >= total`, this loop's own first-iteration exit) paid
    // all three default walks, ~176 ms and ~45 MiB of cut vectors over
    // a 6 MiB document, for levels that supplied zero cuts.
    let mut levels: Vec<LevelSlot> = match separators {
        Some(seps) => custom_level_specs(seps),
        None => default_level_specs(),
    };
    // The grapheme boundary index: built lazily, AT MOST ONCE per call,
    // at the first site that actually needs it — a realized level's cut
    // filter, the raw-cut fallback, or the overlap snap, every site one
    // [`grapheme_index`] call. The former spelling built the whole
    // per-codepoint structure unconditionally before anything else
    // could run, which is the document-scale cost #22 measured.
    let mut graphemes: Option<GraphemeIndex> = None;

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
        // The find_map short-circuit is preserved exactly: the FIRST
        // slot supplying a cut wins, and later slots stay unrealized for
        // THIS window (a LATER window that reaches them reuses their
        // memoized build). Realization is the closure's first act, so a
        // consulted slot is built even when it supplies no cut for the
        // window — consultation, not success, is what pays the walk.
        let cut = levels
            .iter_mut()
            .find_map(|slot| {
                slot.realize(text, total, &mut graphemes)
                    .best_cut(start, limit)
            })
            .unwrap_or_else(|| {
                let g = grapheme_index(&mut graphemes, text, total);
                let end = g.hard_cut(start, limit);
                (end, end)
            });
        chunks.push((start, cut.0));
        if overlap == 0 {
            start = cut.1;
        } else {
            let target = cut.0.saturating_sub(overlap);
            let g = grapheme_index(&mut graphemes, text, total);
            let snapped = g.last_at_or_before(target);
            start = if snapped > start { snapped } else { cut.1 };
        }
    }
    chunks
}

/// The test-only counting seam behind the lazy-level pins: per-THREAD
/// counters of the level realizations and grapheme-index builds this
/// module issues, so the tests can assert exactly WHICH levels a call
/// built, how often, and whether it built the bitmap at all — counts
/// the differential sweep cannot see (it pins lazy == eager OUTPUT;
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
    /// UNCONDITIONAL `grapheme_boundary_chars` + `HashSet<usize>` filter
    /// over every level, `grapheme_safe_hard_cut` over the usize grid, the
    /// `partition_point` overlap snap) runs against the new spelling over
    /// a corpus × budget × overlap × hierarchy sweep below. The rewrite
    /// claims bit-identical output; this is the pin.
    fn chunk_hierarchical_reference(
        text: &str,
        max_chars: usize,
        separators: Option<&[Option<&str>]>,
        overlap: usize,
    ) -> Vec<(usize, usize)> {
        if text.is_empty() {
            return Vec::new();
        }
        assert!(max_chars > 0, "max_chars must be at least 1, got 0");
        let chars: Vec<char> = text.chars().collect();
        let total = chars.len();

        // The oracle keeps its OWN inline EAGER level building (not the
        // spec list + [`LevelSlot::realize`] spelling the production
        // code now uses) so the differential sweep below pins the lazy
        // spelling's output against this eager one — the exact pin that
        // lazy == eager — the same way it pins the bitmap machinery:
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

    /// The differential corpus: every cluster shape the filter and the
    /// hard cut have to get right — plain ASCII, CRLF/blank-line
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
                        let new =
                            chunk_hierarchical(&text, max_chars, sep_refs.as_deref(), overlap);
                        let old = chunk_hierarchical_reference(
                            &text,
                            max_chars,
                            sep_refs.as_deref(),
                            overlap,
                        );
                        assert_eq!(
                            new, old,
                            "divergence: text={text:?} max_chars={max_chars} \
                             overlap={overlap} separators={seps:?}"
                        );
                    }
                }
            }
        }
    }

    // ---- The lazy levels (#30): seam-counted structural pins. The
    // differential sweep above pins lazy == eager OUTPUT; these pin the
    // laziness itself — WHICH levels a call builds, how often, and
    // whether the grapheme bitmap is built at all — through the
    // per-thread `build_seam` counters, reset at each test's start.

    #[test]
    fn whole_document_budget_builds_no_levels_and_no_grapheme_index() {
        // The issue's own headline cell: a single-chunk budget
        // (`max_chars >= total`) pays NO level walk at all — the eager
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
        // window's FIRST consultation is the paragraph level and it
        // supplies every cut, so sentence and word are never consulted,
        // never built, across however many windows the text windows
        // into — the chunk ends landing on paragraph ends is the
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
        // exactly once too — at realization, not in any pre-loop pass.
        assert_eq!(build_seam::grapheme_index_built(), 1);
    }

    #[test]
    fn a_budget_that_descends_builds_each_level_at_most_once_per_call() {
        // One paragraph, one sentence, many words, a word-sized budget:
        // EVERY window consults paragraph (a lone paragraph has no gap
        // to cut at), then sentence (its only cut sits past the
        // window's limit), then word (which supplies the cut) — the
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
        // Per-CALL scope: a fresh call pays the walks again — the
        // memoization lives in the call's level slots, not a cache.
        build_seam::reset();
        let _ = chunk_hierarchical(text, 7, None, 0);
        assert_eq!(build_seam::levels_built(), 3);
    }

    #[test]
    fn a_never_descended_splice_builds_none_of_its_levels() {
        // The splice cost, seam-pinned from the cheap side:
        // `["\n", None]` over lines that all fit the budget — every
        // window's first consultation is the "\n" literal and it
        // supplies every cut — builds the LITERAL level and none of the
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
        // The bitmap's OTHER lazy build site: a hierarchy with no
        // levels at all (an empty separators list) under a small budget
        // builds zero levels and exactly one grapheme index — at the
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
        // never change an ANSWER, but the UNdeduped spelling passes
        // that pin too — what it cannot see is the WALKS a duplicate
        // would re-pay. The find_map only ever
        // reaches a duplicate after its original returned None for the
        // current window, so the pin needs a window that exhausts the
        // whole (deduped) hierarchy to the raw-cut fallback: the
        // unbroken "x" run below supplies it. Eightfold [" "] must then
        // build exactly ONE literal level (not eight identical scans
        // and cut vectors — the 1.69 GiB shape the dedup closed), and
        // [None, None] exactly the three spliced levels (not six — a
        // second None after a CONSULTED splice adds no builds, the pin
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
        // The two dedups TOGETHER on one list: literals dedup against
        // literals across any distance (the seen-set is whole-list,
        // spanning the spliced specs between them), the None splice
        // against the first splice, and the two rules compose —
        // ["\n", None, "\n", None, " ", None, " "] is exactly
        // ["\n", None, " "] as a slot list: one "\n" literal, the three
        // spliced specs, one " " literal. The line-shaped text (the
        // "\n" genuinely matches, so the pin is not vacuous) with an
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
        // "0" + SARA AM is ONE grapheme cluster; a literal separator that
        // matches the SARA AM codepoint alone would cut between the two
        // codepoints of that cluster. The filter must drop every such
        // match — the case the filter exists for, pinned because the
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
        // [None] IS separators=None, and a never-matching literal above a
        // None entry changes nothing (it can never supply a cut, so the
        // spliced levels answer every window the default hierarchy would):
        // exact-equality properties over the whole differential corpus and
        // a budget sweep, far stronger than a couple of spot cases.
        // Duplicate inertness — of `None` AND of literals, both deduped at
        // slot construction — is the same argument one level deeper: a
        // duplicate's level is identical to its first occurrence's, so
        // the find_map can never reach a duplicate that changes an answer
        // its original didn't already give — [None, None] IS [None], and
        // the duplicated line-first list is the unduplicated one (the
        // corpus texts contain "\n", so the literal genuinely matches:
        // the pin is meaningful, not vacuous). That inertness is also
        // overlap-INDEPENDENT — the dedup happens in list construction,
        // before any windowing or overlap snap — so every duplicate shape
        // is swept over the boundary-adjacent overlaps the bitmap
        // differential above uses, and an eightfold run pins that longer
        // duplicate lists stay inert, free now that the splice is built
        // once per call regardless of list length. (The dedup's COST —
        // what this output-equality sweep cannot see, since the
        // undeduped spelling passes it too — is pinned by the timing
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
        // message per line, never split mid-line — and an oversized
        // message falling back to the ACCURATE UAX #29 sentence level.
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
        // the SAME thread and a budget below the first sentence boundary.
        // The all-literal [\"\\n\", \". \", \" \"] list's \". \" level has a
        // match after \"U.S.\" (a mid-name non-sentence), so it severs the
        // name; the spliced [\"\\n\", None] list has no in-budget sentence
        // cut there and falls to the WORD level, whose cuts are UAX #29
        // word boundaries — every piece still ends at a word boundary, and
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
        // And the name rides WHOLE inside one piece (the word-level cuts
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
        let chunks = chunk_hierarchical(&text, 3, Some(&[Some("NEVER")]), 1);
        assert!(chunks.len() >= 166);
    }
}
