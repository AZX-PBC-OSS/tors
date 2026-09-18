//! The chunking family's cross-function invariants on arbitrary text:
//! `chunk_hierarchical` (any separator hierarchy, any budget, any
//! overlap), `chunk_by_words`/`chunk_by_sentences` (the unit-count
//! spellings whose merge step shares the same grapheme boundary index),
//! and `chunk_by_paragraphs` (basic contract only: its paragraph spans
//! are line-run edges, documented as not necessarily grapheme-aligned: a
//! combining mark after a newline joins the newline's cluster, and the
//! newline is separator content no paragraph's caller would call "split").
//!
//! For the three cluster-safe chunkers, every chunk start and end must be
//! a grapheme-cluster boundary per `unicode-segmentation` directly (the
//! independent oracle for the shared `GraphemeIndex` bitmap, including
//! its ASCII fast path, not the production code re-answering its own
//! question), and a chunk exceeding `max_chars` is legal only as exactly
//! one whole grapheme cluster (the documented oversized-cluster
//! exception). Starts strictly increase, and — since #83 — ends strictly
//! advance past the previous chunk's end whenever overlap > 0: an
//! overlapping window can no longer legitimately re-offer the same cut to
//! two consecutive chunks (that emitted a chunk strictly inside its
//! predecessor; the snap is now declined instead), so under overlap a
//! failing-to-advance end is a bug. At overlap == 0 ends never move
//! backward, non-strictly: only a regressing end is a bug there. Forward
//! progress at the sequence level.
//!
//! `chunk_by_lines` and `chunk_by_paragraphs` carry more than
//! structure: an inline random-access reference oracle each (the same
//! whole-text-`Vec<char>` spellings the Rust unit tests keep as their
//! own `#[cfg(test)]` oracles; tors-core is a path dep, so those are
//! unreachable from this crate and inlined here instead) plus a
//! windower mirroring `chunk_by_segments`'s documented contract, a
//! differential pin: wrong CRLF folding, a blank-line miscount, or a
//! paragraph run-qualification drift now panics the fuzzer with the
//! reproducing input, not just violates structure.
//!
//! Separator hierarchies fuzz in a two-shape x two-budget grid over the
//! same body: the `SepEntry` alphabet below (shaped so the
//! `None`-splice's interesting region is reached at useful rates) and
//! the raw arbitrary needles (extraction shapes, odd literals, the
//! plain `None` spelling), each at both the raw byte-drain `max_chars`
//! and a pressure-shaped budget in 1..=total: the raw budget lands at
//! or above the text length for almost every input (struct fields drain
//! the byte stream in order and `text` eats most of it, so the leftover
//! bytes behind `max_chars` are few), and a budget that swallows the
//! whole text never opens a second window, starving the multi-window
//! fallback walk a hierarchy exists for; the shaped budget forces that
//! walk. The unit-count sweep is budget-independent of the grid and
//! runs once.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Arbitrary, Debug)]
struct Input {
    text: String,
    max_chars: std::num::NonZeroU16,
    overlap_raw: u16,
    // The shaped list derives before the raw one: struct fields drain the
    // byte stream in order and `text` eats most of it, so the leftover
    // bytes should first buy the shapes that reach the splice's
    // interesting region (short lists, each entry a one-discriminant
    // variant, so even scraps yield [Line, Splice] / [Never, Splice] /
    // [Splice, Splice]), leaving the remainders for the raw needles,
    // whose multi-byte junk rarely matched anything anyway (the
    // documented reason the shaped field exists).
    shaped_separators: Option<Vec<SepEntry>>,
    separators: Option<Vec<Option<String>>>,
}

/// Separators drawn from a shaped alphabet so the None-splice's
/// semantically interesting region (a literal that actually fires
/// above or below a `None` entry, on line-shaped text, at a budget
/// that forces fallback) is reached at useful rates. Raw arbitrary
/// string needles essentially never match raw arbitrary text (the
/// multi-byte junk a byte-drain produces), so the unshaped field
/// alone left the splice path unfuzzed where it matters. The same
/// shaping argument covers two more literals: the empty literal (the
/// no-op level production drops at slot construction; this entry
/// pins it inert in every position, one discriminant away instead of
/// a zero-length-String derive the byte drain only happens to hit)
/// and Thai SARA AM (a multi-byte literal that matches inside a
/// grapheme cluster, the exact shape the grapheme cut filter exists
/// to drop; the Rust unit test covers that literal directly, but the
/// fuzzer could not reach the cut-filter region without a shaped
/// entry, since a raw needle must derive U+0E33's exact bytes and the
/// text must carry them too).
#[derive(Arbitrary, Debug)]
enum SepEntry {
    /// `None`: splice the default hierarchy's levels here.
    Splice,
    /// "\n": the line-first shape ([Some("\n"), None]).
    Line,
    /// "\n\n": the paragraph-run literal.
    BlankRun,
    /// ". ": the naive sentence guess.
    DotSpace,
    /// " ": the naive word guess.
    Space,
    /// "ZZZ_NEVER_MATCHES": a level that can never supply a cut.
    Never,
    /// "": the no-op level production drops at slot construction (an
    /// empty literal would match everywhere and cut nothing). Pins
    /// that it stays inert in every position (above, below, and
    /// between other levels) rather than trusting the raw field's
    /// byte drain to derive an empty String.
    Empty,
    /// "\u{0E33}" (Thai SARA AM): a multi-byte literal whose matches
    /// land inside grapheme clusters ("0" + SARA AM is one cluster:
    /// the UAX #29 word/sentence divergence the cut filter exists
    /// for). The Rust unit test pins this literal directly; a raw
    /// byte drain essentially never derives the needle, so this entry
    /// is what makes the cut-filter region reachable with a literal
    /// that actually fires there.
    SaraAm,
    /// "heading": the #63 sentinel — the heading LEVEL (bounding ATX
    /// heading-line cuts), not a literal split on the word. The
    /// gate-closed texts (no '#' byte) pin its inertness; the
    /// heading-bearing texts the raw `text` field occasionally
    /// derives exercise the gate-open side.
    Heading,
}

impl SepEntry {
    /// The separator entry this shape stands for: `None` splices the
    /// default hierarchy's three accurate levels in at this position, a
    /// literal is the literal itself. Duplicates stay derivable exactly
    /// the same way ([Splice, Splice] pins duplicate-None inertness,
    /// [Never, Splice] the never-matching literal above the splice), so
    /// the shaped list keeps the None/duplicate coverage the raw field's
    /// byte drain only happened to hit.
    fn separator(&self) -> Option<&'static str> {
        match self {
            SepEntry::Splice => None,
            SepEntry::Line => Some("\n"),
            SepEntry::BlankRun => Some("\n\n"),
            SepEntry::DotSpace => Some(". "),
            SepEntry::Space => Some(" "),
            SepEntry::Never => Some("ZZZ_NEVER_MATCHES"),
            SepEntry::Empty => Some(""),
            SepEntry::SaraAm => Some("\u{0E33}"),
            SepEntry::Heading => Some("heading"),
        }
    }
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

/// The #63 heading-cut oracle, inlined (the same whole-text-`Vec<char>`
/// spelling the Rust unit tests keep as `heading_cuts_reference`; tors-core
/// is a path dep, so this copy is inlined like the line/paragraph oracles
/// above): one cut `(gap_start, heading_start)` per ATX heading line —
/// 1-3 leading spaces, 1-6 `#`, then space/tab/EOL — with the pre-heading
/// newline run dropped, the fence state tracked, and a heading at offset 0
/// not recorded. The line-ending set is LF, CRLF, lone CR.
fn heading_cuts_reference(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut cuts = Vec::new();
    let mut open_fence: Option<(char, usize)> = None;
    let mut line_start = 0usize;
    let mut last_content_end = 0usize;
    while line_start < n {
        let mut line_end = n;
        let mut next_start = n;
        let mut j = line_start;
        while j < n {
            if chars[j] == '\n' {
                line_end = j;
                next_start = j + 1;
                break;
            }
            if chars[j] == '\r' {
                line_end = j;
                next_start = if j + 1 < n && chars[j + 1] == '\n' {
                    j + 2
                } else {
                    j + 1
                };
                break;
            }
            j += 1;
        }
        let line: String = chars[line_start..line_end].iter().collect();
        match open_fence {
            Some((fence_char, fence_len)) => {
                let indent = line.chars().take_while(|&c| c == ' ').count().min(3);
                let rest: Vec<char> = line.chars().skip(indent).collect();
                let run = rest.iter().take_while(|&&c| c == fence_char).count();
                let trailing_ok = rest[run..]
                    .iter()
                    .all(|&c| c == ' ' || c == '\t' || c == '\r');
                if run >= fence_len && trailing_ok {
                    open_fence = None;
                }
            }
            None => {
                let indent = line.chars().take_while(|&c| c == ' ').count().min(3);
                let rest: Vec<char> = line.chars().skip(indent).collect();
                let fence_char = rest.first().copied();
                if fence_char == Some('`') || fence_char == Some('~') {
                    let fence_len = rest
                        .iter()
                        .take_while(|&&c| c == fence_char.unwrap())
                        .count();
                    if fence_len >= 3 {
                        let info: String = rest[fence_len..].iter().collect();
                        if !(fence_char == Some('`') && info.trim().contains('`')) {
                            open_fence = Some((fence_char.unwrap(), fence_len));
                        }
                    }
                } else if line_start > 0 {
                    let hashes = rest.iter().take_while(|&&c| c == '#').count();
                    let after = rest.get(hashes).copied();
                    if (1..=6).contains(&hashes) && matches!(after, None | Some(' ') | Some('\t')) {
                        cuts.push((last_content_end, line_start));
                    }
                }
            }
        }
        if line_end > line_start {
            last_content_end = line_end;
        }
        line_start = next_start;
    }
    cuts
}

fn assert_basic_contract(
    chunks: &[(usize, usize)],
    total: usize,
    what: &str,
    overlap: usize,
    text: &str,
) {
    let mut prev_start = None;
    let mut prev_end = None;
    for &(start, end) in chunks {
        assert!(
            start < end,
            "{what}: empty or inverted chunk: ({start}, {end})"
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
        // Ends strictly advance past the previous chunk's end when
        // overlap > 0 (#83: the overlap re-offer that could emit a chunk
        // strictly inside its predecessor — one cut serving two
        // consecutive windows, a "\n\n" firing at 22 under budget 22 and
        // overlap 4 yielding (0, 22), (18, 22) — is declined by the
        // windowers now, not blessed). At overlap == 0 the looser
        // non-decreasing bound is kept. The unit windowers are not a
        // second justification for the loose bound: chunk_by_segments'
        // loop breaks right after its single clamped final chunk, and
        // every window before it advances `stride >= 1` segments, so
        // unit-chunker ends are always strictly increasing; they merely
        // happen to satisfy it. What must never happen anywhere is an end
        // moving backward.
        if let Some(prev) = prev_end {
            assert!(
                if overlap > 0 { end > prev } else { end >= prev },
                "{what}: ends not advancing under overlap={overlap} at {end} after {prev} \
                 text={text:?} chunks={chunks:?}"
            );
        }
        prev_start = Some(start);
        prev_end = Some(end);
    }
}

/// The random-access `line_bounds` oracle, inlined from the same
/// whole-text-`Vec<char>` spelling the Rust unit tests keep as their
/// own `#[cfg(test)]` oracle (that one is invisible to this path-dep
/// crate: the reason this copy exists): the whole-text `Vec<char>`
/// collect with a one-codepoint CRLF lookahead. A break unit is a `\n`
/// (unless directly after a `\r`, whose CRLF pair it completes) or a
/// `\r` (always opens a unit); a line is the maximal run between break
/// units, kept only when it carries a non-whitespace codepoint;
/// trailing content after the last break is a line iff non-whitespace,
/// and a trailing break yields no phantom line. Caveat: the oracle
/// shares production's `char::is_whitespace()` content filter, so the
/// differential below pins the two machines (CRLF folding, break
/// counting, windowing) against each other, not the whitespace
/// definition itself; that one the Python suite pins
/// (tests/test_chunk_text.py, the White_Space-vs-`str.isspace` cells).
fn line_bounds_reference(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    if n == 0 {
        return Vec::new();
    }
    let is_break = |c: char| c == '\n' || c == '\r';
    // A '\r' always opens a unit (a lone CR, or the first half of a CRLF
    // pair); a '\n' opens one only when it did not ride in as the
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

/// The random-access `paragraph_bounds` oracle, inlined from the same
/// whole-text-`Vec<char>` spelling the Rust unit tests keep as their
/// own `#[cfg(test)]` oracle (that one is invisible to this path-dep
/// crate: the same reason `line_bounds_reference` above is inlined):
/// paragraphs split on maximal runs of 2+ newline units, a `\r\n` pair
/// counting as one unit (the CR/CRLF folding every scanner in this
/// crate shares); a lone unit is ordinary content, and the empty spans
/// a qualifying run would mint at the text's edges are discarded.
/// Caveat, the line oracle's twin: the oracle shares production's
/// 2+-unit/run-walk shape, so the differential pins the two machines
/// (byte-level run walking, unit counting, windowing) against each
/// other; the heuristic itself (no Unicode Standard behind "2+
/// newlines is a paragraph gap") is a contract stated in prose, not
/// something either spelling could cross-check.
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

/// The tiny windower mirroring `chunk_by_segments`'s documented contract
/// (src/chunk_by_segment_impl.rs, inlined for the same `#[cfg(test)]`
/// reachability reason as `line_bounds_reference`): chunk i spans
/// `[bounds[i].0, bounds[min(i + per_chunk, n) - 1].1)`, consecutive
/// chunks advance `stride = per_chunk - overlap` segments, empty bounds
/// yield `[]`. The two `assert!`s mirror the Rust side's own
/// load-bearing preconditions verbatim: `stride`'s arithmetic
/// underflows without them.
fn reference_window(
    bounds: &[(usize, usize)],
    per_chunk: usize,
    overlap: usize,
) -> Vec<(usize, usize)> {
    if bounds.is_empty() {
        return Vec::new();
    }
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

fuzz_target!(|input: Input| {
    let total = input.text.chars().count();
    // The chunk spans are CODEPOINT indices; this target's whole-document
    // oracle compares a span's text against the separator literals, which
    // needs the codepoint space, not a byte slice (a byte slice of a
    // codepoint index is mid-char for any multi-byte head — the smoke
    // crash: text "\u{5a5}", chunk (0, 1), `&text[..1]` not a char
    // boundary). One collect, shared by every oracle below.
    let codepoints: Vec<char> = input.text.chars().collect();
    // The raw byte-drain budget: struct fields drain the stream in
    // order and `text` eats most of it, so the leftover bytes behind
    // `max_chars` are few and it lands at or above the text length
    // for almost every input: a budget that swallows the whole text
    // never opens a second window, so the multi-window fallback walk
    // (the level consultation, a literal firing under pressure, the
    // overlap snap) ran in only a few percent of corpus inputs. The
    // pressure-shaped budget below closes that.
    let raw_budget = input.max_chars.get() as usize;
    // The pressure shape, in 1..=total by construction: the dividend
    // `max_chars - 1` is non-negative (`NonZeroU16`), the modulus
    // `total.max(1)` never divides by zero, and `1 + x % total` with
    // `x % total < total` cannot exceed `total`. It lands on `total`
    // (the one value in range with no budget pressure, since the
    // first window's `remaining <= max_chars` then swallows the text
    // in one chunk) exactly when `max_chars` is a positive multiple
    // of `total` (`1 + (k*total - 1) % total == total`); every other
    // derivation is strictly below `total`, so the first window has
    // `remaining > max_chars` and the multi-window walk must run.
    // Any input with `total > 1` therefore reaches the walk under at
    // least one of the two budgets in ~all derivations; the residue
    // is the exact-multiple corner and single-codepoint texts.
    let pressure_budget = 1 + (raw_budget - 1) % total.max(1);

    // The line oracle's bounds list is text-only, so it is computed once
    // here and shared by every windowing below (each per_chunk value
    // re-windows the list, never re-scans the text); the paragraph
    // oracle's list rides the same one-compute discipline.
    let reference_lines = line_bounds_reference(&input.text);
    let reference_paragraphs = paragraph_bounds_reference(&input.text);
    let reference_heading_cuts = heading_cuts_reference(&input.text);

    // The hierarchical body, parameterized on (separators, budget): the
    // functions' own precondition is `overlap < max_chars`, clamped
    // rather than skipped so the harness still explores the boundary,
    // derived per budget, since the two budgets share only the raw
    // byte. `assert_cluster_safe` gets the same budget the call used:
    // the oversized-cluster exception is budget-relative.
    let run_hierarchical = |separators: Option<&[Option<&str>]>, budget: usize| {
        let overlap = input.overlap_raw as usize % budget.max(1);
        let chunks = tors::chunk_hierarchical_impl::chunk_hierarchical(
            &input.text,
            budget,
            separators,
            overlap,
            tors::chunk_hierarchical_impl::OverlapBoundary::Grapheme,
        );
        assert_basic_contract(&chunks, total, "chunk_hierarchical", overlap, &input.text);
        assert_cluster_safe(&chunks, &input.text, budget, "chunk_hierarchical");
        // The whole-document-budget oracle, folded in at every budget
        // that can only ever emit the single first window: `max_chars
        // >= total` makes the loop's first `remaining <= max_chars`
        // exit fire, so the answer is [(0, total)] (or [] on empty
        // text) regardless of separators and overlap — with the #103
        // narrowing: a window that OPENS on a separator match is
        // skipped even at that exit, so an all-separator document (or
        // one whose every window before content opens on a match)
        // comes back [] (or a single (start, total) window past the
        // leading matches). What stays exactly pinned — and what the
        // codepoint total (`char_count`, pub(crate) and unreachable
        // from this crate) directly feeds, the reason this row exists
        // as near-equality and not another invariant — is the shape: at
        // most one chunk, ending exactly at `total` (a byte-count
        // regression would emit (0, byte_total) and fail the end
        // check), never a chunk that IS a separator match (the #103
        // symptom, exact-equality-checked against every literal in the
        // hierarchy), never more than one chunk from a budget that
        // swallows the document. The #63 narrowing: a hierarchy that
        // SPELLS the heading level (the `None` default, or a list with
        // the `"heading"` sentinel) demotes the whole-remainder exit at
        // every heading cut in range — the sections come back separate
        // — so the at-most-one pin applies only to hierarchies without
        // the level (the pinned property is the pre-#63 machine's, and
        // the exact section-split answer for the heading-active
        // default hierarchy is the dedicated overlap-0 row below).
        let heading_active = match separators {
            None => true,
            Some(list) => list
                .iter()
                .any(|entry| entry.is_none() || entry.as_deref() == Some("heading")),
        };
        if budget >= total && !(heading_active && !reference_heading_cuts.is_empty()) {
            assert!(
                chunks.len() <= 1,
                "whole-document budget {budget} must emit at most one chunk: \
                 text={:?} separators={separators:?} overlap={overlap} chunks={chunks:?}",
                input.text
            );
            if let Some(&(start, end)) = chunks.first() {
                assert_eq!(
                    end, total,
                    "whole-document chunk must run to the codepoint total: \
                     text={:?} separators={separators:?} chunks={chunks:?}",
                    input.text
                );
                if let Some(list) = separators {
                    // The no-separator-chunk assert is enforceable only for
                    // SINGLE-LITERAL, no-splice, no-heading-sentinel lists:
                    // there the one separator level's skip_cut is consulted
                    // first at every window, so a window opening on the
                    // match always skips and the whole-document exit never
                    // runs (the #103 contract, including the
                    // all-separator-zero-chunks shape). Three exceptions
                    // break the theorem, each fuzz-discovered and each
                    // pinned oracle-equal (production and the reference
                    // agree — the machine's documented coarsest-first
                    // precedence, not a regression):
                    // * a `None` SPLICE: the default hierarchy's paragraph
                    //   level owns blank-run cuts, and "\n" beside the
                    //   default triple chunks [(0, 1)] on both sides;
                    // * the "heading" SENTINEL: the heading level's cut is
                    //   coarser by design (text "#", [None, "#"] ->
                    //   [(0, 1)], the sentence cut);
                    // * TWO OR MORE literals: an earlier literal's CUT
                    //   preempts a later one's skip, and the final exit
                    //   pushes the remainder untrimmed — which can equal
                    //   the later separator's whole match
                    //   (["\n\u{1a}]\0", "\t\n\u{1a}]\0"] over the 5-char
                    //   text chunks (0, 5) on BOTH sides).
                    // For the enforceable case the pin compares against the
                    // separator's own non-overlapping MATCH SPANS, in the
                    // codepoint space the chunks are spanned in — not the
                    // literal: the untrimmed push of a window that opens on
                    // NO match can emit a remainder that merely SLICES a
                    // separator shape at a position the level never matched
                    // (the same remainder class the
                    // chunk_separator_shapes target's header documents as
                    // deliberately unasserted).
                    let spliced_or_heading_or_multi = list.len() > 1
                        || list
                            .iter()
                            .any(|entry| entry.is_none() || entry.as_deref() == Some("heading"));
                    if !spliced_or_heading_or_multi {
                        for entry in list {
                            let Some(sep) = *entry else { continue };
                            if sep.is_empty() {
                                continue; // the no-op literal production drops at slot construction
                            }
                            let sep_chars: Vec<char> = sep.chars().collect();
                            let mut matches_at = Vec::new();
                            let mut pos = 0usize;
                            while pos + sep_chars.len() <= total {
                                if codepoints[pos..pos + sep_chars.len()] == sep_chars[..] {
                                    matches_at.push(pos);
                                    pos += sep_chars.len();
                                } else {
                                    pos += 1;
                                }
                            }
                            assert!(
                                !matches_at
                                    .iter()
                                    .any(|&m| start == m && end == m + sep_chars.len()),
                                "whole-document budget emitted a chunk that IS the separator \
                                 {sep:?}'s own match (#103): text={:?} chunks={chunks:?}",
                                input.text
                            );
                        }
                    }
                }
            }
        }
    };

    // The grid: both separator shapes at both budgets. The shaped
    // alphabet first (the point of the fix: the splice region, now
    // under real pressure), then the raw arbitrary needles (extraction
    // shapes, odd literals, and the plain `None` default-hierarchy
    // spelling the raw field's Option carries). The two budgets
    // coincide exactly when the raw one already sits in 1..=total (the
    // shape is the identity there), so the grid's only duplicated work
    // re-runs a combination one budget already covered, never a new
    // cost class. The third row is the whole-document budget
    // `total.max(1)`: the pressure row lands on `total` only for exact
    // multiples, so without this row the single-window path (and the
    // char_count total it rests on) ran in a few percent of inputs;
    // `.max(1)` keeps the empty-text case a legal budget (its oracle
    // answer is []).
    let shaped: Option<Vec<Option<&'static str>>> = input
        .shaped_separators
        .as_ref()
        .map(|v| v.iter().map(|entry| entry.separator()).collect());
    let raw: Option<Vec<Option<&str>>> = input
        .separators
        .as_ref()
        .map(|v| v.iter().map(|entry| entry.as_deref()).collect());
    for budget in [raw_budget, pressure_budget, total.max(1)] {
        run_hierarchical(shaped.as_deref(), budget);
        run_hierarchical(raw.as_deref(), budget);
    }

    // The #63 row, exact: the DEFAULT hierarchy (the heading level
    // spelled) at a whole-document budget and overlap 0 — the demotion
    // cuts at every heading cut in range (progress is unconditional
    // here: overlap 0 opens windows only at heading starts, and the
    // cut ends ascend), so the answer IS the section split, the
    // reference heading cuts' own chunking. The differential pins
    // production's byte-level fence/ATX scan against the inlined
    // char-grid oracle line for line — the fence state machine, the
    // shape clauses, the gap dropping, and the offset-0 discard all
    // have to agree exactly, on arbitrary text.
    if total > 0 {
        let mut expected: Vec<(usize, usize)> = Vec::new();
        let mut resume = 0usize;
        for &(gap_start, heading_start) in &reference_heading_cuts {
            // A cut whose gap start has no room before it (the text opens
            // on the dropped run: gap_start == resume == 0) contributes
            // no chunk — the window at the run's start is skipped to the
            // heading (#103's skip machinery), never an empty span.
            if gap_start > resume {
                expected.push((resume, gap_start));
            }
            resume = heading_start;
        }
        expected.push((resume, total));
        let chunks = tors::chunk_hierarchical_impl::chunk_hierarchical(
            &input.text,
            total,
            None,
            0,
            tors::chunk_hierarchical_impl::OverlapBoundary::Grapheme,
        );
        assert_eq!(
            chunks, expected,
            "whole-document default hierarchy diverged from the section split: \
             text={:?} heading_cuts={reference_heading_cuts:?}",
            input.text
        );
    }

    // The unit-count chunkers over the same arbitrary text: same
    // per-chunk/overlap envelope (per_chunk in 1..=u16, overlap
    // clamped), cluster safety for the two merge-based spellings, and
    // the differential pin for both gapped scanners: the line scanner
    // against its inline reference oracle (any divergence between the
    // streaming state machine and the random-access spelling: wrong
    // CRLF folding, a blank-line miscount, a windowing drift) panics
    // here with the reproducing parameters in the message, and the
    // paragraph scanner against its own (same machine class: byte-level
    // run walking and unit counting against the random-access
    // spelling; the paragraph spans are break-run edges, documented as
    // not necessarily grapheme-aligned: a combining mark after a
    // newline joins the newline's cluster, and the newline is
    // separator content no paragraph's caller would call "split"), but
    // the oracle checks where those edges are, not just that they
    // satisfy structure. This sweep is budget-independent of the grid
    // above (its per_chunk envelope is its own), so it runs once, not
    // per budget and not per shape, and the grid's added executions stay
    // hierarchical-only.
    let overlap = (input.overlap_raw as usize) % raw_budget.max(1);
    for per_chunk in [1usize, 2, 3, 7, raw_budget] {
        let overlap = overlap % per_chunk;
        let words = tors::chunk_by_segment_impl::chunk_by_words(&input.text, per_chunk, overlap);
        assert_basic_contract(&words, total, "chunk_by_words", overlap, &input.text);
        assert_cluster_safe(&words, &input.text, usize::MAX, "chunk_by_words");

        let sentences =
            tors::chunk_by_segment_impl::chunk_by_sentences(&input.text, per_chunk, overlap);
        assert_basic_contract(
            &sentences,
            total,
            "chunk_by_sentences",
            overlap,
            &input.text,
        );
        assert_cluster_safe(&sentences, &input.text, usize::MAX, "chunk_by_sentences");

        let paragraphs =
            tors::chunk_by_segment_impl::chunk_by_paragraphs(&input.text, per_chunk, overlap);
        assert_basic_contract(
            &paragraphs,
            total,
            "chunk_by_paragraphs",
            overlap,
            &input.text,
        );
        assert_eq!(
            paragraphs,
            reference_window(&reference_paragraphs, per_chunk, overlap),
            "chunk_by_paragraphs diverged from the reference oracle: text={:?} \
             per_chunk={per_chunk} overlap={overlap}",
            input.text
        );

        let lines = tors::chunk_by_segment_impl::chunk_by_lines(&input.text, per_chunk, overlap);
        assert_basic_contract(&lines, total, "chunk_by_lines", overlap, &input.text);
        assert_eq!(
            lines,
            reference_window(&reference_lines, per_chunk, overlap),
            "chunk_by_lines diverged from the reference oracle: text={:?} \
             per_chunk={per_chunk} overlap={overlap}",
            input.text
        );
    }
});
