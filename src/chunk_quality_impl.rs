//! Intrinsic chunk-quality metrics: the pure-Rust core of
//! `tors.chunk_quality`.
//!
//! Two intrinsic (no gold labels, no embeddings, no retrieval loop)
//! measures of how well a caller's own chunk spans
//! (`chunk_text`/`chunk_hierarchical`/`chunk_by_*`'s `(start, end)`
//! tuples, or any hand-built spans) respect the text's structure, from
//! the adaptive-chunking metrics study (de Moura Júnior, Lelong &
//! Blangero, "Adaptive Chunking: Optimizing Chunking-Method Selection
//! for RAG", LREC 2026, arXiv 2603.25333, the
//! reference implementation's metric suite):
//!
//! - **integrity**: the study's Block Integrity (BI), the fraction of
//!   gold blocks NOT crossed by a chunk boundary. The gold blocks here
//!   are the suite's own UAX #29 sentence spans (`grounding_impl`'s
//!   segmentation, the same bounds `sentence_bounds` publishes), not a
//!   human-annotated block set: a chunk boundary "crosses" a sentence
//!   when it falls strictly inside the sentence's span, farther than the
//!   tolerance `tau` (codepoints, default 0) from BOTH edges. A boundary
//!   within `tau` of an edge is snapped-close-enough, the tolerance the
//!   study's metric carries for segmenter jitter. No sentence, no
//!   crossing, no damage: the score is the fraction of sentences with
//!   zero crossing boundaries.
//! - **cohesion**: the study's Intra-Chunk Cohesion (ICC), the mean
//!   within-chunk sentence-to-chunk similarity. The paper's ICC scores
//!   each sentence against its chunk with EMBEDDING cosine similarity;
//!   tors's version is the dependency-free LEXICAL PROXY: the Dice
//!   coefficient (`shingle_dice`'s exact quantity, 2|A ∩ B| / (|A| + |B|)
//!   over the width-3 word-shingle sets the near-duplicate family
//!   defines, the same UAX #29 tokenization and case-fold+NFC matching
//!   form) between the sentence's shingle set and its containing chunk's.
//!   That is a different, weaker signal than the paper's embeddings: it
//!   rewards chunks whose sentences share surface vocabulary and
//!   penalizes chunks that stitch unrelated sentences together, but it
//!   cannot see topical continuity that shares no words (a pronoun-heavy
//!   continuation scores low), and a sentence shorter than the shingle
//!   width has an empty shingle set (the shingle family's own convention:
//!   exactly-one-empty scores 0.0, two empty sets 1.0). A sentence is
//!   assigned to a chunk by CONTAINMENT (the chunk's span covers the
//!   sentence's span); a sentence no chunk contains (the mid-sentence
//!   cuts `integrity` already penalized) contributes to neither metric's
//!   numerator, and a text with no (sentence, chunk) pair at all pins
//!   cohesion to the conservative 0.0, the same 0/0 convention
//!   `grounding_coverage` pins.
//!
//! Both metrics are pure functions of (chunks, text): no model, no
//! network, no dependencies beyond the crate's own segmenters and
//! shingle hashes, so a caller can gate a chunking-config sweep on them
//! the way the study's selector does, per document, in milliseconds.

use std::collections::HashSet;

use crate::grounding_impl::sentence_spans;
use crate::near_dup_impl::shingle_hashes;

/// The shingle width for the cohesion proxy: the near-duplicate family's
/// own documented default (`shingle_jaccard`/`shingle_dice`'s
/// `width=3`), so the proxy scores with exactly the quantity those
/// surfaces publish. Deliberately not a parameter: one number, one
/// convention, no second knob to drift against `shingle_dice`.
const SHINGLE_WIDTH: usize = 3;

/// The two metrics over one chunk set, both in `[0.0, 1.0]`, both always
/// present (the house result-object style; the binding marshals the
/// pair as a dict with exactly these keys).
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkQuality {
    pub integrity: f64,
    pub cohesion: f64,
}

/// The Dice coefficient of two shingle-hash sets with the shingle
/// family's documented empty-set conventions (two empty sets are
/// duplicates of each other: 1.0; exactly one empty: 0.0).
fn dice(a: &HashSet<u64>, b: &HashSet<u64>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count();
    2.0 * inter as f64 / (a.len() + b.len()) as f64
}

/// The two metrics over the caller's `chunks` (codepoint spans into
/// `text`) and `text`, `tau` the crossing tolerance in codepoints. Never
/// panics on any input: an empty text, an empty chunk list, and
/// token-free text all return the pinned degenerate shape
/// (integrity 1.0: no chunk boundary crosses any sentence, vacuously;
/// cohesion 0.0: no (sentence, chunk) pair exists to average).
pub fn chunk_quality(chunks: &[(usize, usize)], text: &str, tau: usize) -> ChunkQuality {
    // The gold blocks: the suite's own UAX #29 sentence spans. Built
    // once; the same segmentation every grounding surface publishes.
    let sentences = sentence_spans(text);
    let n = sentences.len();

    // --- integrity: the fraction of sentences (gold blocks) no chunk
    // boundary crosses. A boundary p (either edge of any chunk) crosses
    // the sentence spanning (s, e) iff s < p < e and p sits farther than
    // tau from both edges; the vacuous case (no sentences) is 1.0, the
    // honest reading of "no boundary crosses anything".
    let integrity = if n == 0 {
        1.0
    } else {
        let crossed = sentences
            .iter()
            .filter(|&&(s, e, _, _)| {
                chunks.iter().any(|&(cs, ce)| {
                    let crosses = |p: usize| s < p && p < e && p - s > tau && e - p > tau;
                    crosses(cs) || crosses(ce)
                })
            })
            .count();
        1.0 - crossed as f64 / n as f64
    };

    // --- cohesion: the mean Dice similarity between each contained
    // sentence's shingle set and its containing chunk's. Every sentence
    // set is computed exactly once (up front); a sentence contained in
    // several chunks of an overlapping caller's set contributes once per
    // containing chunk, the honest mean over the (sentence, chunk) pairs
    // that exist. The chunk's span is sliced through a
    // char-index-to-byte-offset table (the spans are codepoint offsets;
    // the shingle tokenizer wants a &str, and re-walking chars per chunk
    // would be quadratic in the chunk count).
    let mut byte_at: Vec<usize> = Vec::with_capacity(text.len() + 1);
    for (byte, _) in text.char_indices() {
        byte_at.push(byte);
    }
    byte_at.push(text.len());

    let sentence_sets: Vec<HashSet<u64>> = sentences
        .iter()
        .map(|&(_, _, bs, be)| shingle_hashes(&text[bs..be], SHINGLE_WIDTH))
        .collect();

    let mut total = 0.0f64;
    let mut count = 0usize;
    for &(cs, ce) in chunks {
        if cs > ce || ce > text.len() {
            // Unreachable through the binding (the wrapper validates
            // every span), the core's defensive contract: skip rather
            // than panic, a bad span simply contains nothing.
            continue;
        }
        let chunk_set = shingle_hashes(&text[byte_at[cs]..byte_at[ce]], SHINGLE_WIDTH);
        for (i, &(s, e, _, _)) in sentences.iter().enumerate() {
            if cs <= s && e <= ce {
                total += dice(&sentence_sets[i], &chunk_set);
                count += 1;
            }
        }
    }
    let cohesion = if count == 0 {
        0.0
    } else {
        total / count as f64
    };

    ChunkQuality {
        integrity,
        cohesion,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentence_sized_chunks_score_perfect_on_both() {
        let text = "One two three. Four five six. Seven eight nine.";
        // The exact sentence spans, one chunk per sentence: every
        // boundary lands on a sentence edge (nothing crossed) and every
        // chunk IS its sentence (Dice 1.0).
        let chunks: Vec<(usize, usize)> = sentence_spans(text)
            .iter()
            .map(|&(s, e, _, _)| (s, e))
            .collect();
        assert_eq!(chunks.len(), 3);
        let q = chunk_quality(&chunks, text, 0);
        assert_eq!(q.integrity, 1.0);
        assert_eq!(q.cohesion, 1.0);
    }

    #[test]
    fn the_hand_built_known_answer_vector() {
        // Three 3-word sentences, one chunk covering everything: no
        // boundary inside any sentence (integrity 1.0), and each
        // sentence's shingle set rides entirely inside the chunk's. The
        // shingle tokenizer keeps punctuation segments (only whitespace
        // is filtered: `real_word_segments`' documented filter), so each
        // sentence is 4 tokens (3 words + the period) -> 2 width-3
        // shingles, the chunk is 12 tokens -> 10 shingles, and every
        // sentence shingle is one of them: Dice = 2*2/(2+10) = 1/3 per
        // sentence, cohesion 1/3. (Integrity stays 1.0: the one big
        // chunk's edges are the text's own edges, no sentence is cut.)
        let text = "One two three. Four five six. Seven eight nine.";
        let q = chunk_quality(&[(0, text.chars().count())], text, 0);
        assert_eq!(q.integrity, 1.0);
        assert!((q.cohesion - 1.0 / 3.0).abs() < 1e-12, "got {}", q.cohesion);
    }

    #[test]
    fn mid_sentence_cuts_damage_integrity_and_tau_forgives_edge_cuts() {
        // Sentence spans (UAX #29 attaches the terminator's trailing
        // space to the preceding sentence): (0, 15) and (15, 29).
        let text = "One two three. Four five six.";
        // A boundary 4 codepoints inside the first sentence (past
        // "One t|wo"): crossed at tau 0 and still crossed at tau 3 (4
        // and 11 codepoints from the two edges).
        let q0 = chunk_quality(&[(0, 4), (4, text.len())], text, 0);
        assert_eq!(q0.integrity, 0.5);
        let q3 = chunk_quality(&[(0, 4), (4, text.len())], text, 3);
        assert_eq!(q3.integrity, 0.5);
        // A boundary exactly at the second sentence's start edge (15):
        // not a crossing at any tau.
        let q_edge = chunk_quality(&[(0, 15), (15, text.len())], text, 0);
        assert_eq!(q_edge.integrity, 1.0);
        // A boundary 1 codepoint past that edge: crossed at tau 0,
        // forgiven once tau reaches 1.
        let q_strict = chunk_quality(&[(0, 16), (16, text.len())], text, 0);
        assert_eq!(q_strict.integrity, 0.5);
        let q_tol = chunk_quality(&[(0, 16), (16, text.len())], text, 1);
        assert_eq!(q_tol.integrity, 1.0);
    }

    #[test]
    fn degenerate_inputs_pin_their_shapes() {
        let text = "One two three. Four five six.";
        // No chunks: no boundary crosses anything (integrity 1.0, the
        // vacuous reading), no (sentence, chunk) pair exists (cohesion
        // 0.0, the conservative 0/0 pin).
        let q = chunk_quality(&[], text, 0);
        assert_eq!(
            q,
            ChunkQuality {
                integrity: 1.0,
                cohesion: 0.0
            }
        );
        // Empty text: no sentences either.
        assert_eq!(
            chunk_quality(&[(0, 0)], "", 0),
            ChunkQuality {
                integrity: 1.0,
                cohesion: 0.0
            }
        );
        // Token-free text: one sentence span exists (a whitespace run
        // segments), contained, and both shingle sets are empty, so the
        // shingle family's own two-empty-sets convention scores the pair
        // 1.0 (two token-free spans are duplicates of each other), the
        // same answer `shingle_dice("    ", "    ")` publishes.
        assert_eq!(
            chunk_quality(&[(0, 4)], "    ", 0),
            ChunkQuality {
                integrity: 1.0,
                cohesion: 1.0
            }
        );
        // A single chunk exactly covering the text's single sentence:
        // cohesion 1.0 (the chunk's shingle set IS the sentence's).
        let one = chunk_quality(&[(0, 14)], "One two three.", 0);
        assert_eq!(one.cohesion, 1.0);
    }
}
