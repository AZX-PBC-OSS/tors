//! Boundary-safe truncation, the pure-Rust core of `tors.truncate_to_bounds`.
//!
//! Composes the crate's OWN segmentation tables (`segmentation_impl::word_bounds`/
//! `sentence_bounds`, the UAX #29 machinery already shipped) rather than
//! reimplementing any boundary logic: cut at the last word/sentence boundary
//! at or before `max_chars`, so a context-window budget is never enforced by
//! slicing a codepoint out of the middle of a word or sentence. No new
//! dependency, no new algorithm: a composition of two primitives already in
//! the crate, for the LLM context-window/token-budget fitting case, where
//! the naive `text[:max_chars]` a pipeline reaches for today risks cutting
//! mid-word.
//!
//! A word/sentence boundary is NOT automatically a grapheme-cluster boundary:
//! `word_bounds` gives some combining sequences (e.g. Thai SARA AM, U+0E33)
//! their own word-segment even though `unicode-segmentation`'s grapheme rules
//! join them to the preceding base character into one cluster: cutting at
//! such a word boundary would silently split the cluster and drop/mangle the
//! combining mark. Every cut point below is additionally intersected with
//! grapheme-cluster boundaries (`unicode_segmentation::graphemes`, the same
//! table backing `grapheme_count`), including the hard-cut fallback, so the
//! result never ends mid-cluster regardless of which boundary kind is asked
//! for.

use std::borrow::Cow;

use unicode_segmentation::UnicodeSegmentation;

use crate::segmentation_impl;

/// The codepoint-index STARTS of every grapheme cluster in `text`, plus one
/// final entry at `text`'s total codepoint length (the end-of-text
/// boundary): the complete, sorted-ascending set of positions a cut is
/// allowed to land on without splitting a cluster. One forward pass over
/// `text.graphemes(true)`, O(n) total (each codepoint is counted exactly
/// once across all clusters), not O(n) per cluster.
pub(crate) fn grapheme_boundary_chars(text: &str) -> Vec<usize> {
    let mut boundaries = Vec::new();
    let mut char_idx = 0usize;
    for cluster in text.graphemes(true) {
        boundaries.push(char_idx);
        char_idx += cluster.chars().count();
    }
    boundaries.push(char_idx);
    boundaries
}

/// Where `truncate_to_bounds` is allowed to cut: at a word boundary (UAX #29
/// WB1-WB999) or a sentence boundary (UAX #29 SB1-SB999), the same two
/// segmenters `word_bounds`/`sentence_bounds` already expose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    Word,
    Sentence,
}

/// Every grapheme-cluster boundary at or before `max_chars`, in BOTH
/// index units at once: parallel ascending arrays of each boundary's
/// codepoint index and byte offset, from ONE forward pass that stops at
/// the budget. Every position a `truncate_to_bounds` cut can land on (a
/// cluster-safe segment end, the hard-cut fallback, and the byte offset
/// the final slice needs) is a cluster boundary inside that range, so
/// the walk never passes it: a small budget on a huge corpus walks only
/// the prefix instead of the whole text. The caller has already
/// established the text exceeds `max_chars` codepoints, so the walk
/// always stops at the first cluster boundary past the budget; 0 always
/// records.
fn grapheme_boundary_offsets_within(text: &str, max_chars: usize) -> (Vec<usize>, Vec<usize>) {
    let mut char_starts = Vec::with_capacity(max_chars);
    let mut byte_starts = Vec::with_capacity(max_chars);
    let mut char_idx = 0usize;
    let mut byte_idx = 0usize;
    for cluster in text.graphemes(true) {
        if char_idx > max_chars {
            break;
        }
        char_starts.push(char_idx);
        byte_starts.push(byte_idx);
        char_idx += cluster.chars().count();
        byte_idx += cluster.len();
    }
    (char_starts, byte_starts)
}

/// The word/sentence segment ends that are ALSO grapheme-cluster
/// boundaries (the cluster-safety intersection every cut in this module
/// and `chunk_impl` filters through), as one ascending `Vec<usize>`:
/// `bounds`'s ends merged against `grapheme_starts`, both ascending, one
/// two-pointer pass. This is the shared helper behind both modules' cut
/// machinery (one copy, owned here with the rest of the boundary/cut
/// rule, imported by `chunk_impl`), and it replaces the `HashSet<usize>`
/// both spellings formerly built: a 12 MiB prose input holds millions of
/// boundaries, where a multi-million-entry hash set costs an order of
/// magnitude more memory than this flat `Vec<usize>` while answering the
/// same membership question; the callers that need a RANGE (the largest
/// end at or before a budget) binary-search it with `partition_point`,
/// O(log n) per cut. `grapheme_starts` chooses the range: `chunk_text`
/// passes every boundary of the text, `truncate_to_bounds` only those at
/// or before its budget, which bounds the merge output exactly where its
/// own `<= max_chars` filter would have cut anyway.
pub(crate) fn cluster_safe_ends(
    bounds: &[(usize, usize)],
    grapheme_starts: &[usize],
) -> Vec<usize> {
    let mut ends = Vec::with_capacity(bounds.len());
    let mut gi = 0usize;
    for &(_, end) in bounds {
        while gi < grapheme_starts.len() && grapheme_starts[gi] < end {
            gi += 1;
        }
        if gi < grapheme_starts.len() && grapheme_starts[gi] == end {
            ends.push(end);
        }
    }
    ends
}

/// Truncate `text` to at most `max_chars` codepoints, cutting at the last
/// `boundary` at or before `max_chars` rather than mid-word/mid-sentence,
/// and never mid-grapheme-cluster either (see the module docs).
///
/// * If `text` already has `<= max_chars` codepoints, it comes back
///   UNCHANGED, per the crate's `Cow` identity convention:
///   `tors.truncate_to_bounds(s, n) is s` exactly when no truncation
///   happens.
/// * Otherwise the cut point is the largest `word_bounds`/`sentence_bounds`
///   segment end `<= max_chars` that is ALSO a grapheme-cluster boundary
///   (a segment end that would split a cluster, e.g. a combining mark
///   `word_bounds` scores as its own word-segment, is not a valid cut
///   point, even if it's otherwise `<= max_chars`); if no such boundary
///   exists (a single word/sentence/cluster longer than `max_chars`, or
///   `max_chars == 0`), the cut falls back to the largest GRAPHEME
///   boundary `<= max_chars`. This is a hard cut in codepoint terms, but
///   still cluster-safe, so a lone combining mark is never separated from
///   its base character. The one invariant that never breaks either way:
///   the result never exceeds `max_chars` codepoints (it can fall short of
///   the budget when respecting a cluster boundary requires backing off
///   further; correctness takes priority over filling the last codepoint).
/// * The result is then trimmed of trailing Unicode whitespace at the cut
///   point (`str::trim_end`). Cutting exactly after a word boundary would
///   otherwise leave a dangling separator space (word boundaries include
///   the inter-word space as its own segment; sentence boundaries carry a
///   trailing space on the PRECEDING sentence per UAX #29 SB9-SB11: see
///   `segmentation_impl::sentence_bounds`'s docs), which this trims away.
///   Trimming can only ever shrink the result further, so it cannot violate
///   the `max_chars` invariant above.
pub fn truncate_to_bounds(text: &str, max_chars: usize, boundary: Boundary) -> Cow<'_, str> {
    if text.chars().count() <= max_chars {
        return Cow::Borrowed(text);
    }
    // The text exceeds the budget, so the boundary walk below stops at
    // the first cluster boundary past it and never touches the rest of
    // the text.
    let (grapheme_starts, byte_starts) = grapheme_boundary_offsets_within(text, max_chars);
    let bounds = match boundary {
        Boundary::Word => segmentation_impl::word_bounds(text),
        Boundary::Sentence => segmentation_impl::sentence_bounds(text),
    };
    let ends = cluster_safe_ends(&bounds, &grapheme_starts);
    let hi = ends.partition_point(|&end| end <= max_chars);
    let cut_chars = if hi > 0 {
        ends[hi - 1]
    } else {
        // No word/sentence boundary is also cluster-safe within budget
        // (or none fit at all): fall back to the largest grapheme
        // boundary <= max_chars, the bounded walk's last recorded start
        // (0 always records, so this always exists).
        *grapheme_starts.last().unwrap()
    };
    // The cut is one of the walk's recorded boundaries by construction
    // (an end that survived the cluster-safety merge, or the fallback's
    // own last record), so its byte offset is the parallel array's entry
    // at the same index: no `char_indices` re-walk of the text to turn a
    // codepoint index back into a byte offset.
    let cut_idx = grapheme_starts.partition_point(|&g| g < cut_chars);
    let byte_cut = byte_starts[cut_idx];
    Cow::Owned(text[..byte_cut].trim_end().to_string())
}

/// The marker `truncate_ellipsis` appends to a cut value: U+2026 HORIZONTAL
/// ELLIPSIS, one codepoint. The same marker `ta_sync`'s column-bound
/// `truncate` uses (`value[:limit - 1] + "…"`, stored length exactly the
/// bound), so adopting the tors spelling keeps stored values byte-identical
/// on every input where the naive cut does not land mid-cluster.
pub const ELLIPSIS: char = '\u{2026}';

/// Truncate `text` to at most `max_chars` codepoints with an ellipsis
/// marker: a hard cut (no word/sentence awareness, unlike
/// `truncate_to_bounds` — this is the DB-column shape, where the bound is a
/// storage limit, not a reading break), made grapheme-cluster-safe.
///
/// * If `text` already has `<= max_chars` codepoints, it comes back
///   UNCHANGED, per the crate's `Cow` identity convention:
///   `tors.truncate_ellipsis(s, n) is s` exactly when no truncation
///   happens.
/// * Otherwise the kept prefix is the largest grapheme-cluster boundary at
///   or before `max_chars - 1` codepoints (one codepoint of budget is the
///   marker itself), plus `ELLIPSIS`. The result never exceeds `max_chars`
///   codepoints; it falls short when cluster backoff requires it (a cut
///   landing inside a ZWJ sequence or combining cluster snaps back past
///   the whole cluster), correctness over filling the last codepoint.
/// * `max_chars == 0` yields empty (there is no room for even the marker;
///   the naive `value[:0] + "…"` spelling answers `"…"` here, exceeding a
///   zero bound — this does not repeat that).
/// * No trailing-whitespace trim: the cut is positional, not semantic, and
///   the caller asked for exactly the bound. `"abc   "` at `max_chars=5`
///   keeps its spaces then the marker.
///
/// Composes the same `grapheme_boundary_offsets_within` walk
/// `truncate_to_bounds` uses (prefix-only: a small bound on a huge value
/// walks only the prefix), no new dependency, no new algorithm.
pub fn truncate_ellipsis(text: &str, max_chars: usize) -> Cow<'_, str> {
    if text.chars().count() <= max_chars {
        return Cow::Borrowed(text);
    }
    if max_chars == 0 {
        return Cow::Owned(String::new());
    }
    // The text exceeds the budget, so the walk stops at the first cluster
    // boundary past `max_chars - 1` and never touches the rest of it. The
    // last recorded start is the largest cluster boundary within budget (0
    // always records, so this always exists), and every recorded start is
    // a char boundary, so the slice below cannot panic.
    let (_char_starts, byte_starts) = grapheme_boundary_offsets_within(text, max_chars - 1);
    let byte_cut = *byte_starts.last().unwrap();
    let mut out = String::with_capacity(byte_cut + ELLIPSIS.len_utf8());
    out.push_str(&text[..byte_cut]);
    out.push(ELLIPSIS);
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, max_chars: usize) -> String {
        truncate_to_bounds(text, max_chars, Boundary::Word).into_owned()
    }

    fn sentence(text: &str, max_chars: usize) -> String {
        truncate_to_bounds(text, max_chars, Boundary::Sentence).into_owned()
    }

    #[test]
    fn no_op_when_already_within_budget() {
        let text = "short text";
        assert!(matches!(
            truncate_to_bounds(text, text.chars().count(), Boundary::Word),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            truncate_to_bounds(text, 1000, Boundary::Word),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn cuts_at_the_last_word_boundary_and_trims_the_dangling_space() {
        // "cats are cute" word-segments as cats/ /are/ /cute (WB-space
        // segments included); cutting at max_chars=9 lands exactly after
        // the "are" + following-space segment, which trim_end removes.
        assert_eq!(word("cats are cute", 9), "cats are");
        assert_eq!(word("cats are cute", 13), "cats are cute");
        assert_eq!(word("cats are cute", 100), "cats are cute");
    }

    #[test]
    fn falls_back_to_a_hard_cut_when_no_boundary_fits() {
        // A single word far longer than the budget: no word-boundary end is
        // <= max_chars, so the hard-cut fallback applies. The result must
        // never exceed max_chars codepoints.
        let text = "Supercalifragilisticexpialidocious";
        let got = word(text, 10);
        assert_eq!(got.chars().count(), 10);
        assert_eq!(got, &text[..10]);
    }

    #[test]
    fn max_chars_zero_is_empty() {
        assert_eq!(word("hello", 0), "");
        assert_eq!(sentence("hello", 0), "");
    }

    #[test]
    fn empty_input_is_identity() {
        assert!(matches!(
            truncate_to_bounds("", 5, Boundary::Word),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn sentence_boundary_cuts_whole_sentences() {
        let text = "One. Two. Three.";
        // "One. " (5) + "Two. " would be 10, but only a FULL sentence
        // boundary end counts, so max_chars=8 (mid "Two.") falls back to a
        // hard cut, never exceeding 8 chars; max_chars=10 lands exactly on
        // the "Two." sentence boundary end and keeps the whole thing.
        assert_eq!(sentence(text, 10), "One. Two.");
        assert!(sentence(text, 8).chars().count() <= 8);
    }

    #[test]
    fn never_splits_a_thai_sara_am_combining_cluster() {
        // U+0E33 (SARA AM) combines with the preceding base into ONE
        // grapheme cluster, but word_bounds scores it as its OWN
        // word-segment. Budget 1 can't fit the 2-codepoint cluster at all,
        // so the correct, cluster-safe answer is empty; a codepoint-only
        // fallback would instead silently mangle the cluster by keeping
        // just its base character.
        let text = "0\u{0E33}";
        assert_eq!(word(text, 1), "");
        // Budget 2 fits the whole cluster.
        assert_eq!(word(text, 2), text);
    }

    #[test]
    fn hard_cut_fallback_backs_off_rather_than_splitting_a_combining_accent() {
        // "ab" + COMBINING ACUTE ACCENT + "cd", no spaces: word_bounds
        // treats it as one long word-segment (no boundary end <= 2), so
        // the hard-cut fallback applies, and must land on a grapheme
        // boundary (0 or 1, since "b́" is one cluster spanning chars
        // 1-3), never splitting the accent from "b". The safe answer at
        // max_chars=2 is "a" (char 1 is INSIDE the b+accent cluster).
        let text = "ab\u{0301}cd";
        assert_eq!(word(text, 2), "a");
        // A budget landing exactly ON a cluster boundary still works.
        assert_eq!(word(text, 3), "ab\u{0301}");
    }

    #[test]
    fn never_splits_a_zwj_emoji_sequence() {
        // WOMAN + ZWJ + MICROSCOPE is one grapheme cluster (4 codepoints);
        // word_bounds may or may not agree, but the cluster must never be
        // split regardless. A tight budget backs off to before it.
        let text = "hi \u{1f469}\u{200d}\u{1f52c} there".to_string();
        for max_chars in 0..=text.chars().count() {
            let got = word(&text, max_chars);
            assert!(
                got.chars().count() <= max_chars,
                "exceeded budget for {max_chars}: {got:?}"
            );
            // The cluster is 3 codepoints wide as a `char` count (surrogate
            // pairs aren't a Rust `char` concern); if it appears at all in
            // `got`, it must appear WHOLE (never just the ZWJ or just one
            // endpoint).
            let has_zwj = got.contains('\u{200d}');
            let has_woman = got.contains('\u{1f469}');
            let has_scope = got.contains('\u{1f52c}');
            assert_eq!(
                has_zwj,
                has_woman && has_scope,
                "ZWJ sequence split for max_chars={max_chars}: {got:?}"
            );
        }
    }

    #[test]
    fn never_splits_a_regional_indicator_flag_pair() {
        // Two regional-indicator codepoints (e.g. the US flag) are one
        // grapheme cluster; a budget landing between them must back off.
        let text = "a\u{1f1fa}\u{1f1f8}b";
        for max_chars in 0..=text.chars().count() {
            let got = word(text, max_chars);
            let has_first = got.contains('\u{1f1fa}');
            let has_second = got.contains('\u{1f1f8}');
            assert_eq!(
                has_first, has_second,
                "regional-indicator pair split for max_chars={max_chars}: {got:?}"
            );
        }
    }

    #[test]
    fn never_exceeds_max_chars_over_a_battery() {
        let cases = [
            "",
            "a",
            "hello world",
            "One. Two. Three.",
            "supercalifragilistic",
        ];
        for text in cases {
            for max_chars in 0..=text.chars().count() + 2 {
                for boundary in [Boundary::Word, Boundary::Sentence] {
                    let got = truncate_to_bounds(text, max_chars, boundary);
                    assert!(
                        got.chars().count() <= max_chars,
                        "exceeded max_chars={max_chars} for {text:?}/{boundary:?}: {got:?}"
                    );
                }
            }
        }
    }

    fn ellipsis(text: &str, max_chars: usize) -> String {
        truncate_ellipsis(text, max_chars).into_owned()
    }

    #[test]
    fn ellipsis_no_op_when_already_within_budget() {
        let text = "short text";
        assert!(matches!(
            truncate_ellipsis(text, text.chars().count()),
            Cow::Borrowed(_)
        ));
        assert!(matches!(truncate_ellipsis(text, 1000), Cow::Borrowed(_)));
        assert!(matches!(truncate_ellipsis("", 0), Cow::Borrowed(_)));
    }

    #[test]
    fn ellipsis_cuts_to_exact_bound_plus_marker() {
        // The ta_sync column-bound shape: kept prefix of max_chars - 1 plus
        // U+2026, stored length exactly the bound on plain text.
        assert_eq!(ellipsis("hello world", 6), "hello\u{2026}");
        assert_eq!(ellipsis("hello world", 11), "hello world");
        assert_eq!(ellipsis("abcdef", 1), "\u{2026}");
    }

    #[test]
    fn ellipsis_max_chars_zero_is_empty() {
        // NOT "…": the marker alone would exceed a zero bound.
        assert_eq!(ellipsis("hello", 0), "");
    }

    #[test]
    fn ellipsis_does_not_trim_trailing_whitespace() {
        // Positional cut, not semantic: spaces survive, then the marker.
        assert_eq!(ellipsis("abc   def", 5), "abc \u{2026}");
    }

    #[test]
    fn ellipsis_backs_off_a_combining_cluster() {
        // "ab" + COMBINING ACUTE + "cd": budget 3 (prefix 2) lands inside
        // the b+accent cluster, so the cut snaps back to "a".
        assert_eq!(ellipsis("ab\u{0301}cd", 3), "a\u{2026}");
        assert_eq!(ellipsis("ab\u{0301}cd", 4), "ab\u{0301}\u{2026}");
    }

    #[test]
    fn ellipsis_never_splits_a_zwj_sequence_or_flag() {
        let text = "hi \u{1f469}\u{200d}\u{1f52c} there";
        for max_chars in 0..=text.chars().count() + 2 {
            let got = ellipsis(text, max_chars);
            assert!(
                got.chars().count() <= max_chars,
                "exceeded {max_chars}: {got:?}"
            );
            // Marker excluded, the kept prefix must hold whole clusters.
            let kept = got.strip_suffix('\u{2026}').unwrap_or(&got);
            let has_zwj = kept.contains('\u{200d}');
            let has_woman = kept.contains('\u{1f469}');
            let has_scope = kept.contains('\u{1f52c}');
            assert_eq!(
                has_zwj,
                has_woman && has_scope,
                "ZWJ sequence split for max_chars={max_chars}: {got:?}"
            );
        }
        let flag = "a\u{1f1fa}\u{1f1f8}b";
        for max_chars in 0..=flag.chars().count() {
            let got = ellipsis(flag, max_chars);
            assert!(got.chars().count() <= max_chars);
            let kept = got.strip_suffix('\u{2026}').unwrap_or(&got);
            assert_eq!(
                kept.contains('\u{1f1fa}'),
                kept.contains('\u{1f1f8}'),
                "flag split for max_chars={max_chars}: {got:?}"
            );
        }
    }

    #[test]
    fn ellipsis_never_exceeds_max_chars_over_a_battery() {
        let cases = [
            "",
            "a",
            "hello world",
            "caf\u{e9} \u{1f600}",
            "\u{2026}\u{2026}\u{2026}",
        ];
        for text in cases {
            for max_chars in 0..=text.chars().count() + 2 {
                let got = ellipsis(text, max_chars);
                assert!(
                    got.chars().count() <= max_chars,
                    "exceeded max_chars={max_chars} for {text:?}: {got:?}"
                );
                // Either untouched, or cut with the marker at the end (the
                // kept prefix may itself contain U+2026; only the suffix
                // is pinned here).
                if got != text {
                    assert!(
                        max_chars == 0 || got.ends_with('\u{2026}'),
                        "marker shape wrong for {text:?}/{max_chars}: {got:?}"
                    );
                }
            }
        }
    }
}
