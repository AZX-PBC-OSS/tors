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
//!
//! This module is also the crate's shared home for the boundary-set
//! MACHINERY the chunking family builds on: [`char_count`] (the
//! allocation-free codepoint count), [`GraphemeIndex`] (the boundary set
//! as one bit per codepoint — membership, largest-at-or-before, and
//! first-after queries without the usize grid or hash set a
//! document-scale input would otherwise pay for), and
//! [`cluster_safe_ends`] (the single-ended intersection `chunk_text`
//! filters its cuts through).

use std::borrow::Cow;

use unicode_segmentation::UnicodeSegmentation;

use crate::segmentation_impl;

/// The codepoint-index STARTS of every grapheme cluster in `text`, plus one
/// final entry at `text`'s total codepoint length (the end-of-text
/// boundary): the complete, sorted-ascending set of positions a cut is
/// allowed to land on without splitting a cluster. One forward pass over
/// `text.graphemes(true)`, O(n) total (each codepoint is counted exactly
/// once across all clusters), not O(n) per cluster.
///
/// Test-only since the [`GraphemeIndex`] bitmap replaced every production
/// consumer: this Vec spelling is the differential ORACLE the bitmap (and
/// `chunk_impl::grapheme_safe_hard_cut`'s rule via
/// `GraphemeIndex::hard_cut`) is pinned against — the semantic definition
/// in entries, the bitmap the compressed production spelling.
#[cfg(test)]
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

/// A text's codepoint count, ASCII-first: pure-ASCII text answers its
/// own byte length (every ASCII byte is the one-byte UTF-8 encoding of
/// exactly one codepoint, so bytes ARE codepoints there), and
/// non-ASCII text pays the branchless byte pass — every UTF-8 codepoint
/// begins at a byte that is not a continuation byte (`0b10xxxxxx`), so
/// counting non-continuation bytes IS counting codepoints. The
/// allocation-free spelling of `text.chars().count()`'s answer for the
/// callers that need the count WITHOUT the `Vec<char>`-class
/// whole-text materialization a collect would pay (`chunk_hierarchical`'s
/// budget arithmetic, the chunkers' grapheme index below): one pass,
/// zero allocation.
///
/// WHY the split, when the predicate count already answers every case:
/// the two passes are not the same speed on the input that dominates
/// the callers' real traffic. `str::is_ascii` early-exits at the FIRST
/// non-ASCII byte, so pure-ASCII text (the overwhelming case: logs,
/// transcripts, source code) pays one pass that LLVM vectorizes
/// outright and returns the byte length — the same shape as std's own
/// `chars().count()` ASCII path, which measures ~43 GB/s where the
/// filter-and-count predicate measures ~5.1 GB/s (the per-byte
/// `(b & 0xC0) != 0x80` test defeats the auto-vectorizer's lane
/// packing) — and
/// `chunk_by_words`/`chunk_by_sentences`/`chunk_hierarchical` call this
/// on every document-scale text BEFORE their real work, so the count
/// was a measurable fixed tax on exactly the fastest-input case.
/// Non-ASCII text pays a short prefix scan up to its first non-ASCII
/// byte and then the existing exact predicate, so its cost is
/// unchanged to within ~4% (the one extra failed `is_ascii` pass).
/// The same `is_ascii()` fast-path idiom the neighboring
/// [`GraphemeIndex::build`] already uses for its bitmap (with its own
/// exhaustive table-level pin, the 128×128 adjacency test below); the
/// correctness pin HERE is `char_count_matches_chars_count_over_the_corpus`,
/// whose corpus carries the non-ASCII shapes that would expose a
/// miscount (Thai SARA AM, CJK, astral emoji, mixed-script soup).
pub(crate) fn char_count(text: &str) -> usize {
    if text.is_ascii() {
        text.len()
    } else {
        text.as_bytes()
            .iter()
            .filter(|&b| (b & 0xC0) != 0x80)
            .count()
    }
}

/// A text's grapheme-cluster boundary set as one bit per codepoint: bit
/// `i` set iff codepoint index `i` begins a cluster, bit 0 always set, and
/// the end-of-text boundary `total` set too — exactly the entry set
/// [`grapheme_boundary_chars`] materializes as a `Vec<usize>` (~8 bytes
/// per codepoint), here as `total / 64 + 1` words of `u64` (~1.6 MB at a
/// 12 MiB document, two orders of magnitude less) with O(1) cache-friendly
/// membership instead of a hash. That set is the membership question
/// every cut-safety filter in the chunking family asks, and the two
/// positional queries (largest boundary at or before `x`, first boundary
/// after `x`) are word scans over the same bits. Shared by
/// `chunk_hierarchical` (its cut filter, raw-cut fallback, and overlap
/// snap — built lazily, at most once per call) and `chunk_by_segment`
/// (the mid-cluster segment merge), replacing the `HashSet<usize>` both
/// formerly built from the whole `Vec<usize>`: a 12 MiB document holds
/// ~12.6M boundaries, and ~12.6M hashed inserts cost ~1.2 s while being
/// superlinear on top of it (#22).
///
/// WHY a bitmap when [`cluster_safe_ends`] spells the cluster-safety
/// intersection as a two-pointer merge over the `Vec<usize>`: that helper
/// answers one question shape — which single-ended bounds are
/// cluster-safe — while the chunkers' cuts are `(end, next_start)` PAIRS
/// and their merge edges need arbitrary membership, and one bitmap
/// answers every question those callers ask without the ~100 MB usize
/// grid a 12 MiB document would materialize. `grapheme_safe_hard_cut`
/// (in `chunk_impl`) stays the Vec spelling's home; the rule it encodes
/// is replicated (and differential-pinned against it) here as
/// [`GraphemeIndex::hard_cut`].
pub(crate) struct GraphemeIndex {
    /// Bit `i` = "codepoint index `i` is a grapheme-cluster boundary".
    /// `words.len() == total / 64 + 1`: one word PAST bit `total`'s own
    /// word, so the end-of-text boundary bit is always representable
    /// (`total.div_ceil(64)` would be one word short whenever `total` is
    /// an exact multiple of 64, since bit `total` lives in word
    /// `total >> 6`, not `total - 1 >> 6`).
    words: Vec<u64>,
    /// The text's codepoint count: the highest boundary index there is.
    total: usize,
}

impl GraphemeIndex {
    /// One `graphemes(true)` walk setting a bit per cluster start plus the
    /// final end-of-text bit (the same entries `grapheme_boundary_chars`
    /// pushes, as bits instead of usizes). `total` is the caller's
    /// independently-computed codepoint count ([`char_count`]); the debug
    /// asserts pin the two countings and the fast path's claim to each
    /// other.
    ///
    /// Pure-ASCII text skips the segmentation walk: GB3 (CRLF) is the only
    /// grapheme rule that joins two ASCII codepoints — no ASCII byte is
    /// Extend/ZWJ/SpacingMark/Prepend/Regional-Indicator, so every
    /// codepoint starts a cluster except the LF of each CRLF pair — which
    /// makes the bitmap all-ones with the CRLF LF bits cleared, buildable
    /// from one `is_ascii` pass plus one SIMD `\r` scan (char index and
    /// byte index coincide in ASCII). The sufficiency claim is pinned
    /// EXHAUSTIVELY against the real segmenter by the 128×128 adjacency
    /// test in this module's tests, so a unicode-segmentation table
    /// change that ever touched ASCII clustering fails loudly instead of
    /// silently mis-bitting.
    pub(crate) fn build(text: &str, total: usize) -> Self {
        if text.is_ascii() {
            let mut words = vec![u64::MAX; total / 64 + 1];
            let bytes = text.as_bytes();
            let mut crlf_pairs = 0usize;
            for cr in memchr::memchr_iter(b'\r', bytes) {
                // A trailing CR has no LF to join and clears nothing.
                if cr + 1 < bytes.len() && bytes[cr + 1] == b'\n' {
                    let lf = cr + 1;
                    words[lf >> 6] &= !(1u64 << (lf & 63));
                    crlf_pairs += 1;
                }
            }
            // The all-ones fill sets bits above `total` in the top word:
            // mask to the representable range, bits `0..=total & 63` of
            // word `total >> 6` (the `r == 63` case avoids a shift past
            // `u64`'s width).
            let r = total & 63;
            words[total >> 6] &= if r == 63 {
                u64::MAX
            } else {
                (1u64 << (r + 1)) - 1
            };
            debug_assert_eq!(
                text.graphemes(true).count(),
                total - crlf_pairs,
                "ASCII fast path diverged from the grapheme tables"
            );
            Self { words, total }
        } else {
            let mut words = vec![0u64; total / 64 + 1];
            let mut char_idx = 0usize;
            for cluster in text.graphemes(true) {
                words[char_idx >> 6] |= 1u64 << (char_idx & 63);
                char_idx += cluster.chars().count();
            }
            // The end-of-text boundary, the Vec spelling's final entry.
            words[char_idx >> 6] |= 1u64 << (char_idx & 63);
            debug_assert_eq!(char_idx, total);
            Self { words, total }
        }
    }

    /// Is codepoint index `i` a cluster boundary? `false` past `total`
    /// (a defensive answer, not an indexing panic: no cut past `total`
    /// can exist in the callers' arithmetic, and dropping one only falls
    /// through to a finer level or the raw cut, never an unsafe output).
    pub(crate) fn is_boundary(&self, i: usize) -> bool {
        i <= self.total && (self.words[i >> 6] >> (i & 63)) & 1 == 1
    }

    /// The largest cluster boundary `<= x`. Bit 0 is always set, so the
    /// backward scan always terminates; `x` is clamped to `total`, at or
    /// past which the answer is `total` itself.
    pub(crate) fn last_at_or_before(&self, x: usize) -> usize {
        let x = x.min(self.total);
        let r = x & 63;
        let mut wi = x >> 6;
        // Bits `0..=r` of word `wi`; the `r == 63` case spelled out to
        // avoid shifting past `u64`'s width.
        let mut bits = self.words[wi]
            & if r == 63 {
                u64::MAX
            } else {
                (1u64 << (r + 1)) - 1
            };
        loop {
            if bits != 0 {
                return (wi << 6) + (63 - bits.leading_zeros()) as usize;
            }
            wi -= 1;
            bits = self.words[wi];
        }
    }

    /// The smallest cluster boundary strictly `> x`. The caller guarantees
    /// `x < total` (the chunk loop's `start < total` invariant), and bit
    /// `total` is always set, so the forward scan always terminates within
    /// the words.
    pub(crate) fn first_after(&self, x: usize) -> usize {
        debug_assert!(x < self.total);
        let r = x & 63;
        let mut wi = x >> 6;
        let mut bits = self.words[wi] & if r == 63 { 0 } else { !((1u64 << (r + 1)) - 1) };
        loop {
            if bits != 0 {
                return (wi << 6) + bits.trailing_zeros() as usize;
            }
            wi += 1;
            bits = self.words[wi];
        }
    }

    /// The raw-cut fallback's end for the window `[start, limit]`:
    /// `chunk_impl::grapheme_safe_hard_cut`'s exact rule against the
    /// bitmap — the largest boundary `<= limit` when that is genuine
    /// forward progress past `start`, otherwise the first boundary after
    /// `start` (a single cluster wider than the whole remaining budget is
    /// kept whole rather than split, the same documented exception
    /// `chunk_text`'s hard cut carries: one chunk may exceed `max_chars`).
    /// `start < total` and `limit < total` are the chunk loop's own
    /// invariants.
    pub(crate) fn hard_cut(&self, start: usize, limit: usize) -> usize {
        let candidate = self.last_at_or_before(limit);
        if candidate > start {
            candidate
        } else {
            self.first_after(start)
        }
    }
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

    /// The cluster shapes the boundary machinery has to get right: plain
    /// ASCII, CRLF and lone-CR (the ASCII fast path's one join rule and
    /// its near miss), Thai SARA AM (the combining sequence UAX #29
    /// word/sentence bounds split but grapheme rules join), ZWJ emoji
    /// chains, regional-indicator pairs, decomposed accents, CJK, and the
    /// repeated-character runs that stress positional queries. Shared by
    /// the `GraphemeIndex` tests below.
    fn boundary_corpus() -> Vec<String> {
        vec![
            "a".to_string(),
            "hello world".to_string(),
            "Short one.\n\nShort two.".to_string(),
            "a\r\nb\r\n\r\nc".to_string(),
            "a\rb".to_string(),
            "\r".to_string(),
            "\r\n".to_string(),
            "x".repeat(62) + "\r\n",
            "q".repeat(128),
            "0\u{0E33}".repeat(32),
            "ab 0\u{0E33} cd ef 0\u{0E33} gh ij 0\u{0E33} kl".to_string(),
            "e\u{0301}e\u{0301}e\u{0301} ".to_string(),
            "thumbs up \u{1F44D}\u{200D}\u{1F3FB} flag \u{1F1FA}\u{1F1F8}".to_string(),
            "q".repeat(200),
            "abcdefghijklmnopqrstuvwxyz".repeat(8),
            "\u{4E2D}\u{6587}\u{6587}\u{672C}\u{FF0C}\u{6D4B}\u{8BD5}".to_string(),
            "mixed 0\u{0E33} ascii \u{1F600} \u{4E2D}\u{6587} tail".to_string(),
        ]
    }

    #[test]
    fn grapheme_index_answers_every_query_the_vec_spelling_does() {
        for text in boundary_corpus() {
            let total = text.chars().count();
            let starts = grapheme_boundary_chars(&text);
            let index = GraphemeIndex::build(&text, total);
            for x in 0..=total {
                assert_eq!(
                    index.is_boundary(x),
                    starts.contains(&x),
                    "is_boundary({x}) on {text:?}"
                );
                let expected = starts.iter().rev().find(|&&g| g <= x).copied().unwrap();
                assert_eq!(
                    index.last_at_or_before(x),
                    expected,
                    "last_at_or_before({x}) on {text:?}"
                );
                if x < total {
                    let expected = starts.iter().find(|&&g| g > x).copied().unwrap();
                    assert_eq!(
                        index.first_after(x),
                        expected,
                        "first_after({x}) on {text:?}"
                    );
                }
            }
            // Past-the-end clamping and the defensive membership answer.
            assert_eq!(index.last_at_or_before(total + 12345), total);
            assert!(!index.is_boundary(total + 1));
        }
        // A 66-codepoint run of TWO-codepoint clusters ("0" + SARA AM),
        // so boundaries sit at every EVEN index only: the multi-word
        // arithmetic (mask edges, the r == 63 seam, forward/backward
        // scans crossing a word boundary) is exercised on its own shape.
        // The 64-codepoint variant pins the exact-multiple-of-64 seam
        // where bit `total` lives one word past the last cluster's own
        // word (a `div_ceil` length is one word short there, and this
        // test is what catches that class).
        for clustered in ["0\u{0E33}".repeat(33), "0\u{0E33}".repeat(32)] {
            let total = clustered.chars().count();
            let index = GraphemeIndex::build(&clustered, total);
            for x in 0..=total {
                let expected_last = if x % 2 == 0 { x } else { x - 1 };
                assert_eq!(index.last_at_or_before(x), expected_last);
                assert_eq!(index.is_boundary(x), x % 2 == 0);
                if x < total {
                    let expected_next = if x % 2 == 0 { x + 2 } else { x + 1 };
                    assert_eq!(index.first_after(x), expected_next);
                }
            }
        }
    }

    #[test]
    fn ascii_fast_path_condition_is_exhaustively_the_grapheme_tables_answer() {
        // `GraphemeIndex::build`'s ASCII fast path rests on one claim: GB3
        // (CRLF) is the ONLY grapheme rule that joins two ASCII
        // codepoints, so a break between adjacent ASCII chars happens
        // everywhere except between \r and \n. Prove it EXHAUSTIVELY
        // against the actual unicode-segmentation tables in use — every
        // ordered pair of ASCII bytes, embedded in fixed ASCII context
        // (the context bytes cannot join anything themselves, so the only
        // possible cluster spanning the pair's seam is the pair itself) —
        // so a future table change that ever touched ASCII clustering
        // fails here loudly instead of silently mis-bitting the fast
        // path.
        for a in 0u8..128 {
            for b in 0u8..128 {
                let text = format!("xy{}{}zw", a as char, b as char);
                let starts = grapheme_boundary_chars(&text);
                let joined = a == b'\r' && b == b'\n';
                // Position 3 is b's own index: a cluster spanning the
                // a/b seam would have to be the pair itself.
                assert_eq!(
                    starts.contains(&3usize),
                    !joined,
                    "ASCII pair ({a:#04x}, {b:#04x}) {} the tables' answer",
                    if joined {
                        "joined but tables say break at"
                    } else {
                        "breaks but tables join at"
                    }
                );
                // And the fast-path-built index answers every query the
                // same way on this input.
                let index = GraphemeIndex::build(&text, text.chars().count());
                for x in 0..=text.chars().count() {
                    assert_eq!(index.is_boundary(x), starts.contains(&x));
                }
            }
        }
    }

    #[test]
    fn char_count_matches_chars_count_over_the_corpus() {
        // The byte pass's answer is the decode's answer, pinned over every
        // corpus shape (the multi-byte entries are the ones a broken
        // continuation-byte mask would miscount).
        for text in boundary_corpus() {
            assert_eq!(char_count(&text), text.chars().count(), "{text:?}");
        }
        // And a deterministic pseudo-random byte soup: mixed ASCII and
        // multi-byte sequences in every alignment, so the mask's
        // auto-vectorized lane boundaries get crossed too.
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut soup = String::new();
        for _ in 0..2000 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            soup.push(match state % 7 {
                0..=2 => char::from((state >> 8) as u8 % 128), // ASCII lane
                3 => '\u{0E33}',                               // 2-byte SARA AM
                4 => '\u{4E2D}',                               // 3-byte CJK
                5 => '\u{1F600}',                              // 4-byte emoji
                _ => '\u{0301}',                               // 2-byte combining
            });
        }
        assert_eq!(char_count(&soup), soup.chars().count());
    }

    #[test]
    fn grapheme_index_survives_a_deterministic_soup_sweep() {
        // Pseudo-random mixed-script text (the LCG from char_count's
        // soup test), so the index is checked against the Vec spelling on
        // inputs no hand-written corpus anticipates — every position,
        // every query, plus the hard-cut rule against
        // chunk_impl::grapheme_safe_hard_cut where its preconditions hold.
        use crate::chunk_impl::grapheme_safe_hard_cut;
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut alphabet: Vec<char> = ('a'..='z').collect();
        alphabet.extend([
            '\r',
            '\n',
            ' ',
            '.',
            '0',
            '\u{0E33}',
            '\u{0301}',
            '\u{1F600}',
        ]);
        for _ in 0..40 {
            let mut text = String::new();
            for _ in 0..(state % 200 + 1) as usize {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                text.push(alphabet[(state >> 33) as usize % alphabet.len()]);
            }
            let total = char_count(&text);
            let starts = grapheme_boundary_chars(&text);
            let index = GraphemeIndex::build(&text, total);
            for x in 0..=total {
                assert_eq!(index.is_boundary(x), starts.contains(&x), "{text:?}@{x}");
                assert_eq!(
                    index.last_at_or_before(x),
                    starts.iter().rev().find(|&&g| g <= x).copied().unwrap(),
                    "{text:?}@{x}"
                );
                if x < total {
                    assert_eq!(
                        index.first_after(x),
                        starts.iter().find(|&&g| g > x).copied().unwrap(),
                        "{text:?}@{x}"
                    );
                }
            }
            // The hard-cut rule is grapheme_safe_hard_cut's rule: pin it
            // against the original over a strided (start, limit) window
            // sweep (dense enough to hit every seam class — cluster
            // interiors, tight windows, wide windows — without the full
            // O(total^2) cross product).
            for start in (0..total).step_by(3) {
                for limit in (start..total).step_by(5).chain([total - 1]) {
                    assert_eq!(
                        index.hard_cut(start, limit),
                        grapheme_safe_hard_cut(&starts, start, limit),
                        "{text:?}@({start},{limit})"
                    );
                }
            }
        }
    }

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
