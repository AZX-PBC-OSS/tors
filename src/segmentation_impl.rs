//! UAX #29 text segmentation, the pure-Rust core of `tors.grapheme_count`,
//! `tors.word_bounds`, `tors.sentence_bounds`, plus the count spellings of
//! those two list surfaces, `tors.word_count` and `tors.sentence_count`, via
//! the `unicode-segmentation` crate (1.13.3, Unicode 17.0.0 tables): extended
//! grapheme clusters (rules GB1-GB999), word boundaries (rules WB1-WB999),
//! and sentence boundaries (rules SB1-SB999). The stdlib has no equivalent
//! for any of the three; the sentence rules are the same stdlib gap filled
//! by the same crate, and the gap is the point of this surface.
//!
//! There is no stdlib oracle to parity-pin against, so the pins are
//! hand-derived. The Python-side contract gate (tests/test_segmentation.py)
//! pins the grapheme/word table of the tricky cases (ZWJ emoji sequences,
//! combining-mark chains, CRLF as one cluster, regional-indicator pairing,
//! Hangul jamo, and the SARA AM edge (the one character class where a
//! cluster legitimately spans a word boundary) plus structural properties
//! over hypothesis text: monotonic, covering, round-tripping, and the
//! additivity of grapheme counts over word segments wherever that holds (all
//! text except U+0E33/U+0EB3, the Other_Letter spacing marks the normative
//! property files split across the two segmenters). The sentence side is
//! pinned here, in this module's tests, as a battery whose every row was
//! derived from its cited UAX #29 rule before running, then verified against
//! the crate, plus the same structural properties (monotonic, covering,
//! round-tripping) over mixed non-ASCII text.
//!
//! One known sentence presentation quirk, verified against the rules and
//! pinned by the tests: per SB9-SB11 the boundary after a terminator only
//! lands after the terminator's trailing Close* Sp* (SB9/SB10 keep the
//! closing punctuation and spaces attached, SB11 then breaks), so trailing
//! spaces after a sentence terminator belong to the preceding sentence:
//! "One. Two." segments as "One. " + "Two.", the first sentence carrying
//! the inter-sentence space. Callers wanting trimmed sentences must strip
//! their own ends; this function reports what UAX #29 says the boundary is.
//!
//! The offsets are Python str indices (codepoints), not Rust byte offsets:
//! `word_bounds` and `sentence_bounds` each convert the segment byte spans
//! to codepoint spans so `text[start:end]` in Python is the segment. Slicing
//! a non-ASCII text with byte offsets would silently mis-segment for every
//! Python caller.
//!
//! Pure Rust, no pyo3 types: the pyo3 wrapper in `lib.rs` adds only the
//! argument borrow and the return marshalling. That marshalling has a real
//! cost, documented in the crate GIL model and docs/performance.md: returning the
//! full bounds list holds the GIL for O(number-of-segments) tuple
//! construction, fine for ordinary documents but a real cost at whole-file
//! sizes (428-497ms at 12 MiB, 3.67M segments). `tors.word_bounds_iter`
//! (the pyo3 iterator in `lib.rs`) addresses this: it drives this function
//! under one detach at construction and yields the same sequence with
//! µs-scale GIL holds per item, pinned to sequence-parity with the list API
//! by tests/test_segmentation.py, with its GIL band pinned by
//! tests/test_gil_release.py (worst gap 15.4ms at 12 MiB, inside the
//! suite's shared budgets). The list API stays for small inputs and batch
//! work. The sentence surface reuses this exact shape: the pyo3 layer
//! exposes `tors.sentence_bounds` (list) and `tors.sentence_bounds_iter`
//! (streaming, the same sequence), both driving `sentence_bounds` below,
//! one core shared by both spellings, the same pattern as `word_bounds`.
//! Sentences are far sparser than words in prose, so the list API's GIL
//! residue is proportionally smaller, but the iterator exists for symmetry
//! and the same whole-file case. The count spellings (`tors.word_count`,
//! `tors.sentence_count`, driven by the cores below) close the memory
//! story at the other end of the same families: counting is the common
//! question, and the count cores answer it in O(1) memory with a single
//! int return. No per-item marshalling exists, so there is no
//! iterator/count split to ship and no GIL residue to budget; the pyo3
//! layer detaches once and returns a number.

use unicode_segmentation::UnicodeSegmentation;

/// The number of extended grapheme clusters in `text` (UAX #29 GB1-GB999,
/// `is_extended=true`, the only spelling anyone means by "grapheme"
/// post-Unicode 11).
pub fn grapheme_count(text: &str) -> usize {
    text.graphemes(true).count()
}

/// The word-boundary segments of `text` (UAX #29 WB1-WB999) as
/// `(start, end)` pairs in Python str index (codepoint) units: `text[start..end]`
/// is the segment, the bounds cover [0, codepoint_len), and joining the
/// slices reproduces the input. The crate yields byte spans; each segment's
/// codepoint span is accumulated in one pass (its own `chars().count()`, so
/// the total work stays O(text)).
pub fn word_bounds(text: &str) -> Vec<(usize, usize)> {
    // Capacity heuristic: prose measures ~3.4 bytes per segment (12 MiB ->
    // 3.67M segments), so len/4 lands close to the final count. Measured on
    // this box (Linux/glibc) at parity, ±0.5% (p>0.1, criterion 12 MiB
    // cells): glibc realloc serves the multi-MiB growth via mremap, so the
    // exponential-growth copying the heuristic targets is already near-free
    // here. mremap is a Linux allocator behavior, not a guarantee though
    // (heap growth on other allocators pays real copies), and the
    // reservation removes ~21 grow reallocations regardless. Kept as the
    // standard reservation idiom with the parity recorded, not claimed as a
    // measured win: some reservations in this crate measurably reduce
    // allocation cost, others are merely neutral, and this one is neutral.
    // Capped at 1<<20 tuples (16 MiB of 16-byte tuples): an uncapped len/4
    // reserve of a single-token input is 4x its byte size in reserved
    // tuples (a 500MB
    // one-token string would reserve ~2GB for one bound), while the cap
    // costs ordinary inputs nothing: 12 MiB of prose needs 3.67M tuples
    // (~59MB), so the cap only trades ~21 saved reallocs for standard
    // exponential growth past 16 MiB of bounds, the same growth every
    // un-reserved Vec pays everywhere.
    // Shared by the list API and `word_bounds_iter`, whose construction pass
    // calls this.
    let mut bounds = Vec::with_capacity((text.len() / 4).min(1 << 20));
    let mut cp_start = 0usize;
    for (_byte_start, segment) in text.split_word_bound_indices() {
        let cp_len = segment.chars().count();
        bounds.push((cp_start, cp_start + cp_len));
        cp_start += cp_len;
    }
    bounds
}

/// The number of word segments in `text` (UAX #29 WB1-WB999), the count
/// spelling of [`word_bounds`]: the same `unicode-segmentation` word
/// tables, the same single pass the pyo3 layer runs under one detach, and
/// a single int return, the `grapheme_count` precedent applied to the
/// word family. Why its own spelling: counting is the common question,
/// and `word_bounds(text).len()` answers it only by materializing the
/// full answer (the 12 MiB document cell builds ~3.67M tuples, ~59 MB,
/// just to have a number taken off its `len()`); counting the same
/// segmentation keeps none of it: O(1) memory, O(text) time, one number.
///
/// # Invariant
///
/// `word_count(text) == word_bounds(text).len()` for every input: the
/// two spellings run the identical UAX #29 word segmentation, so a
/// disagreement is a bug, not a tolerance; pinned over the mixed battery
/// by the tests below.
pub fn word_count(text: &str) -> usize {
    // `split_word_bounds` is the same segmentation walk `word_bounds`
    // drives via its `_indices` sibling (same tables, same rules), with
    // the segment strings yielded instead of (byte offset, segment)
    // pairs; counting needs neither the offsets nor the spans, so this
    // is the cheapest spelling of the identical pass.
    text.split_word_bounds().count()
}

/// The raw "real word segment" walk shared by every term-frequency-style
/// primitive in this crate: UAX #29 word segments (`split_word_bounds`, the
/// same walk `word_bounds`/`word_count` drive), restricted to segments
/// carrying at least one non-whitespace codepoint. A "word" is a real
/// token, not a raw `word_bounds` segment (which gives an inter-word
/// whitespace run its own segment, per the WSegSpace rule). Exposed
/// separately so `tokenize_impl`'s richer, opt-in-normalizing tokenizer
/// (accent-folding, stemming) can build on the same segment walk without
/// re-deriving it.
pub(crate) fn real_word_segments(text: &str) -> impl Iterator<Item = &str> {
    text.split_word_bounds()
        .filter(|segment| !segment.chars().all(char::is_whitespace))
}

/// `real_word_segments`, each lowercased via Rust's Unicode-correct
/// `str::to_lowercase` (not an ASCII-only fold), the near-universal
/// term-matching convention: `"Cat"` and `"cat"` are the same term.
/// `simhash64` does not reuse this (its bag-of-words
/// fingerprint leaves case folding to the caller).
///
/// Production code now calls `tokenize_impl::normalized_word_tokens(text,
/// false, None, None)` instead (the richer, opt-in-normalizing tokenizer
/// `tf_idf`/`bm25_rank` share, with accent-folding/stemming/lemma
/// substitution all off). This simpler, independent implementation is kept
/// `#[cfg(test)]`-only as the reference oracle proving the defaults
/// reproduce it exactly.
#[cfg(test)]
pub(crate) fn lowercased_word_tokens(text: &str) -> Vec<String> {
    real_word_segments(text).map(str::to_lowercase).collect()
}

/// The sentence-boundary segments of `text` (UAX #29 SB1-SB999) as
/// `(start, end)` pairs in Python str index (codepoint) units, the same
/// convention as `word_bounds`: `text[start..end]` is the sentence, the
/// bounds cover [0, codepoint_len), and joining the slices reproduces the
/// input. The crate yields byte spans; each segment's codepoint span is
/// accumulated in one pass (its own `chars().count()`, so the total work
/// stays O(text)). One presentation quirk the caller must know, pinned by
/// the tests below: SB9-SB11 attach a terminator's trailing Close* Sp* to
/// the preceding sentence (the boundary lands only before the next
/// non-space), so segments may carry trailing spaces: "One. Two." is
/// `"One. "` + `"Two."`. That is what the rules say the boundary is, not a
/// crate artifact; trim at the call site if trimmed sentences are wanted.
pub fn sentence_bounds(text: &str) -> Vec<(usize, usize)> {
    // Capacity heuristic: sentences are far sparser than words. Prose runs
    // on the order of 30+ bytes per sentence, so len/24 lands in the right
    // decade where word_bounds uses len/4 (its ~3.4 bytes/segment cell).
    // Unlike word_bounds' heuristic this is not a measured parity (no
    // sentence benchmark cells exist yet); it follows the word_bounds
    // precedent as a heuristic, capped at 1<<20 tuples (16 MiB of 16-byte
    // tuples) so a degenerate single-sentence input never reserves a
    // multiple of its own byte size, and needle-neutral at worst: a wrong
    // guess only pays the standard exponential growth every un-reserved Vec
    // pays everywhere. Kept as the standard reservation idiom, not claimed
    // as a measured win.
    // Shared by the list API and the streaming iterator spelling, whose
    // construction pass calls this, the same word_bounds pattern.
    let mut bounds = Vec::with_capacity((text.len() / 24).min(1 << 20));
    let mut cp_start = 0usize;
    for (_byte_start, segment) in text.split_sentence_bound_indices() {
        let cp_len = segment.chars().count();
        bounds.push((cp_start, cp_start + cp_len));
        cp_start += cp_len;
    }
    bounds
}

/// The number of sentence segments in `text` (UAX #29 SB1-SB999), the
/// count spelling of [`sentence_bounds`]: the same `unicode-segmentation`
/// sentence tables, the same single pass the pyo3 layer runs under one
/// detach, and a single int return, the `grapheme_count` precedent
/// applied to the sentence family. Why its own spelling: the same
/// memory-efficiency argument as [`word_count`] (a count answered by
/// `sentence_bounds(text).len()` materializes the full bounds list to
/// have a number taken), one degree sparser: sentences are the sparsest
/// of the three families, but the waste is proportional, not different in
/// kind. The SB9-SB11 trailing-space quirk documented on
/// [`sentence_bounds`] does not affect a count: it shapes where the
/// boundaries land, and the count only asks how many landed.
///
/// # Invariant
///
/// `sentence_count(text) == sentence_bounds(text).len()` for every input:
/// the two spellings run the identical UAX #29 sentence segmentation,
/// so a disagreement is a bug, not a tolerance; pinned over the mixed
/// battery by the tests below.
pub fn sentence_count(text: &str) -> usize {
    // The same sentence segmentation walk `sentence_bounds` drives via
    // its `_indices` sibling; counting needs neither offsets nor spans.
    text.split_sentence_bounds().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZWJ: &str = "\u{200d}";
    const WOMAN: &str = "\u{1f469}";
    const MICROSCOPE: &str = "\u{1f52c}";
    const RI_U: &str = "\u{1f1fa}";
    const RI_S: &str = "\u{1f1f8}";

    #[test]
    fn sentence_bounds_matches_the_cited_uax29_rules() {
        // Every row's expected value was derived from the cited rule before
        // running, then pinned to the crate's verified split (see the module
        // docs). No row disagreed with its derivation.
        // SB3: CR × LF; the CRLF never splits. SB4 then breaks after the
        // LF, so the separator rides the first sentence and "b." is its own.
        assert_eq!(sentence_bounds("a\r\nb."), [(0, 3), (3, 5)]);
        // SB4: a break after each paragraph separator (CR, LF, Sep). The
        // separator stays with the preceding text, no terminator required.
        assert_eq!(sentence_bounds("a\rb."), [(0, 2), (2, 4)]);
        assert_eq!(sentence_bounds("line one\nline two"), [(0, 9), (9, 17)]);
        assert_eq!(sentence_bounds("one\u{2029}two"), [(0, 4), (4, 7)]);
        // SB6: ATerm × Numeric; the '.' before a digit is a decimal point.
        assert_eq!(sentence_bounds("3.4 percent"), [(0, 11)]);
        // SB7: (Upper | Lower) ATerm × Upper; initials stay inside the word.
        assert_eq!(sentence_bounds("U.S.A"), [(0, 5)]);
        // SB8: ATerm Close* Sp* × (¬(OLetter|Upper|Lower|ParaSep|SATerm))*
        // Lower: an ambiguous '.' before lowercase continues the sentence.
        assert_eq!(sentence_bounds("etc. and so on"), [(0, 14)]);
        // SB9/SB10/SB11: the terminator's Close (plain quote and the U+201D
        // RIGHT DOUBLE QUOTATION MARK, Line_Break=Quotation) and Sp* attach
        // to the preceding sentence; SB11 breaks before "Now" (Upper, so
        // SB8's lowercase continuation does not apply).
        assert_eq!(
            sentence_bounds("He said \"stop.\" Now go."),
            [(0, 16), (16, 23)]
        );
        assert_eq!(
            sentence_bounds("He said \u{201d}stop.\u{201d} Now go."),
            [(0, 16), (16, 23)]
        );
        // The trailing-space presentation quirk, rule-derived and pinned:
        // SB10 keeps the inter-sentence space with sentence 1, and SB11
        // breaks only before the non-space "T": "One. " + "Two.". The
        // first segment carries the space; that is the rule, not a bug.
        assert_eq!(sentence_bounds("One. Two."), [(0, 5), (5, 9)]);
        // SB11 over an ideographic terminator: U+3002 is STerm
        // (Sentence_Terminal=Yes), so each 。 ends its sentence in place.
        assert_eq!(
            sentence_bounds("\u{6771}\u{4eac}\u{3002}\u{5927}\u{962a}\u{3002}"),
            [(0, 3), (3, 6)]
        );
        // SB8a: SATerm Close* Sp* × (SContinue | SATerm): "?!" is one
        // terminator run, not two sentences; SB10/SB11 then attach the
        // space and break before "Yes".
        assert_eq!(sentence_bounds("Wow?! Yes."), [(0, 6), (6, 10)]);
        assert_eq!(sentence_bounds("Stop! Go."), [(0, 6), (6, 9)]);
        // Degenerates: empty is empty; spaces-only and terminator-free text
        // are each one segment: SB998 joins Any × Any, and no interior
        // rule ever fires, so only SB1/SB2 bound the text.
        assert_eq!(sentence_bounds(""), Vec::<(usize, usize)>::new());
        assert_eq!(sentence_bounds("   "), [(0, 3)]);
        assert_eq!(sentence_bounds("no terminator here"), [(0, 18)]);
    }

    #[test]
    fn sentence_bounds_round_trip_and_structure_hold_on_a_mixed_battery() {
        let cases = [
            "",
            "   ",
            "no terminator here",
            "One. Two. Three.",
            "He said \"stop.\" Now go.",
            "U.S.A is not three sentences",
            "etc. and so on",
            "Wow?! Yes. 3.4 percent.",
            "a\r\nb. c\r d\ne",
            &format!(
                "caf\u{e9} \u{6771}\u{4eac}\u{3002} \u{5927}\u{962a}\u{3002} {WOMAN}{ZWJ}{MICROSCOPE}"
            ),
            &format!("{RI_U}{RI_S} \u{1100}\u{1161}\u{11a8}\u{3002} e\u{0301}\u{0301}?"),
        ];
        for case in cases {
            let cp: Vec<char> = case.chars().collect();
            let bounds = sentence_bounds(case);
            // Structure: bounds start at 0, are contiguous, strictly
            // increasing and non-overlapping (each start == the previous
            // end, which is stronger than non-overlap), every segment is
            // non-empty, and the last end is the codepoint len (covering).
            // Empty input: no bounds, the loop is vacuous, prev_end stays 0.
            let mut prev_end = 0usize;
            for &(a, b) in &bounds {
                assert!(a == prev_end && b > a, "structure broke for {case:?}");
                prev_end = b;
            }
            assert_eq!(prev_end, cp.len(), "coverage broke for {case:?}");
            // Round trip: joining the codepoint slices reproduces the input,
            // the Python contract that text[start:end] is the segment.
            let joined: String = bounds
                .iter()
                .map(|&(a, b)| cp[a..b].iter().collect::<String>())
                .collect();
            assert_eq!(joined, case, "round trip broke for {case:?}");
        }
    }

    #[test]
    fn grapheme_count_counts_the_tricky_sequences() {
        // GB9: combining marks join their base.
        assert_eq!(grapheme_count("e\u{0301}\u{0301}\u{0301}"), 1);
        // GB4: CRLF is one cluster; CR CR is two.
        assert_eq!(grapheme_count("\r\n"), 1);
        assert_eq!(grapheme_count("\r\r"), 2);
        assert_eq!(grapheme_count("a\r\nb"), 3);
        // GB12/GB13: regional indicators pair.
        assert_eq!(grapheme_count(&format!("{RI_U}{RI_S}")), 1);
        assert_eq!(grapheme_count(&format!("{RI_U}{RI_S}{RI_U}")), 2);
        // GB11: ZWJ emoji sequences are one cluster; ZWJ between letters is not.
        assert_eq!(grapheme_count(&format!("{WOMAN}{ZWJ}{MICROSCOPE}")), 1);
        assert_eq!(grapheme_count(&format!("a{ZWJ}b")), 2);
        // GB9: emoji modifiers are Extend (Emoji_Modifier=Yes is in the
        // Extend property value; the old GB10/E_Base/E_Modifier classes are
        // obsolete in current UAX #29).
        assert_eq!(grapheme_count("\u{1f44d}\u{1f3fd}"), 1);
        // GB6/GB7/GB8: Hangul; GB8 is a join: (LVT | T) x T, the
        // trailing T is part of the syllable term L*(V+|LV V*|LVT)T*.
        assert_eq!(grapheme_count("\u{1100}\u{1161}\u{11a8}"), 1);
        assert_eq!(grapheme_count("\u{ac00}\u{11a8}"), 1); // GB7: LV x T
        assert_eq!(grapheme_count("\u{ac01}\u{11a8}"), 1); // GB8: LVT x T joins
        assert_eq!(grapheme_count(""), 0);
        assert_eq!(grapheme_count("plain text"), 10);
    }

    #[test]
    fn word_bounds_uses_codepoint_indices_and_covers_the_text() {
        // ASCII: byte == codepoint, so this row also pins the segment shapes.
        assert_eq!(
            word_bounds("Hello, world!"),
            [(0, 5), (5, 6), (6, 7), (7, 12), (12, 13)]
        );
        assert_eq!(word_bounds("can't"), [(0, 5)]);
        assert_eq!(word_bounds("3.14"), [(0, 4)]);
        assert_eq!(word_bounds(""), Vec::<(usize, usize)>::new());
        // CRLF is one segment (WB3), in codepoint units.
        assert_eq!(word_bounds("a\r\nb"), [(0, 1), (1, 3), (3, 4)]);
        // Non-ASCII: codepoint indices, not byte indices. The emoji row is
        // 3 codepoints (0..3), where the byte span would be 0..11.
        assert_eq!(word_bounds(&format!("{WOMAN}{ZWJ}{MICROSCOPE}")), [(0, 3)]);
        assert_eq!(
            word_bounds(&format!("caf\u{e9} {RI_U}{RI_S}")),
            [(0, 4), (4, 5), (5, 7)]
        );
        // GB12/GB13 odd-count regional-indicator run (WB15/WB16): pairs as
        // (RI RI)(RI); a 3rd, unpaired RI must not merge into the prior
        // pair or split off wrong; grapheme_count's sibling test pins this
        // count (2 clusters), this pins the actual word-segment shape.
        assert_eq!(
            word_bounds(&format!("{RI_U}{RI_S}{RI_U}")),
            [(0, 2), (2, 3)]
        );
    }

    #[test]
    fn word_bounds_round_trip_and_additivity_hold_on_a_mixed_battery() {
        let cases = [
            "",
            "plain text",
            "Hello, world!",
            "can't stop 3.14 a\r\nb",
            &format!("{WOMAN}{ZWJ}{MICROSCOPE}\u{30c6}\u{30b9}\u{30c8}"),
            &format!("\u{1100}\u{1161}\u{11a8} {RI_U}{RI_S} e\u{0301}"),
        ];
        for case in cases {
            // Byte-level round trip: the segments' UTF-8 concatenation is the input.
            let joined: String = word_bounds(case)
                .iter()
                .map(|&(a, b)| {
                    let cp: Vec<char> = case.chars().collect();
                    cp[a..b].iter().collect::<String>()
                })
                .collect();
            assert_eq!(joined, case, "round trip broke for {case:?}");
            // Additivity: cluster counts of the segments sum to the whole.
            let cp: Vec<char> = case.chars().collect();
            let per_segment: usize = word_bounds(case)
                .iter()
                .map(|&(a, b)| grapheme_count(&cp[a..b].iter().collect::<String>()))
                .sum();
            assert_eq!(
                per_segment,
                grapheme_count(case),
                "additivity broke for {case:?}"
            );
        }
    }

    #[test]
    fn word_count_and_sentence_count_are_the_bounds_lengths_over_the_battery() {
        // The count/list invariants over the two mixed batteries above:
        // every count row is checked against its own bounds spelling, the
        // invariant the two pub docs pin (a disagreement here is a bug in
        // one of the two spellings, not a tolerance to adjust).
        let word_cases = [
            "",
            "plain text",
            "Hello, world!",
            "can't stop 3.14 a\r\nb",
            &format!("{WOMAN}{ZWJ}{MICROSCOPE}\u{30c6}\u{30b9}\u{30c8}"),
            &format!("\u{1100}\u{1161}\u{11a8} {RI_U}{RI_S} e\u{0301}"),
        ];
        for case in word_cases {
            assert_eq!(
                word_count(case),
                word_bounds(case).len(),
                "word count/list disagreement for {case:?}"
            );
        }
        let sentence_cases = [
            "",
            "   ",
            "no terminator here",
            "One. Two. Three.",
            "He said \"stop.\" Now go.",
            "U.S.A is not three sentences",
            "etc. and so on",
            "Wow?! Yes. 3.4 percent.",
            "a\r\nb. c\r d\ne",
            &format!(
                "caf\u{e9} \u{6771}\u{4eac}\u{3002} \u{5927}\u{962a}\u{3002} {WOMAN}{ZWJ}{MICROSCOPE}"
            ),
            &format!("{RI_U}{RI_S} \u{1100}\u{1161}\u{11a8}\u{3002} e\u{0301}\u{0301}?"),
        ];
        for case in sentence_cases {
            assert_eq!(
                sentence_count(case),
                sentence_bounds(case).len(),
                "sentence count/list disagreement for {case:?}"
            );
        }
        // Named degenerates, values derived from the rules before running:
        // empty input has no segments (SB1/SB2 and WB1/WB2 bound nothing
        // when there is nothing to bound); whitespace-only is one word
        // segment (the WSegSpace rule, WSegSpace × WSegSpace, joins
        // consecutive spaces) and one sentence (SB998, Any × Any);
        // "Hello, world!" is 5 word segments (Hello, comma, space, world,
        // bang; the word battery's own row); "One. Two." is 2 sentences
        // (the sentence battery's own row).
        assert_eq!(word_count(""), 0);
        assert_eq!(sentence_count(""), 0);
        assert_eq!(word_count("   "), 1);
        assert_eq!(sentence_count("   "), 1);
        assert_eq!(word_count("Hello, world!"), 5);
        assert_eq!(sentence_count("One. Two."), 2);
    }
}
