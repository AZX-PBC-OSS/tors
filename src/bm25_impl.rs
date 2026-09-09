//! Okapi BM25: the pure-Rust core of `tors.bm25_rank`.
//!
//! # What this is: a RERANKING primitive, not a search index
//!
//! `bm25_rank` scores every document in a caller-supplied `corpus` against
//! one `query`, recomputing everything from scratch on every call. That is
//! the right shape for the common RAG pattern of reranking a SMALL,
//! already-retrieved candidate set (tens to a few hundred documents: a
//! vector-search step's top-k, say) against one query: no state to manage,
//! composes with the rest of the crate's flat, stateless primitives, and
//! cheap enough at that scale to recompute per call.
//!
//! It is NOT a search engine: a corpus with thousands of
//! documents queried repeatedly wants a real inverted index built once and
//! queried many times: tokenize-and-score-from-scratch on every call would
//! waste the corpus-statistics work every single time. For that, reach for
//! a real search engine (`tantivy` is the mature, dominant choice in Rust);
//! tors does not build persistent index objects, the same scope boundary
//! that kept a Merkle inclusion-proof API out of this crate too.
//!
//! `tors` makes no claim about retrieval or relevance QUALITY for any
//! particular corpus or query: BM25 is a well-specified ranking FORMULA,
//! correctly implemented here, not a model-quality promise.
//!
//! # The formula
//!
//! For query `Q` (tokenized to a set of DISTINCT terms: a repeated query
//! word does not multiply its own contribution, the standard Robertson/
//! Spärck-Jones formulation) and document `D`:
//!
//! ```text
//! score(D, Q) = sum over t in Q of IDF(t) * f(t,D) * (k1 + 1)
//!                                   -----------------------------------
//!                                   f(t,D) + k1 * (1 - b + b * |D| / avgdl)
//!
//! IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )
//! ```
//!
//! `N` = corpus size, `n(t)` = number of documents containing `t`, `f(t,D)`
//! = `t`'s occurrence count in `D`, `|D|` = `D`'s token count, `avgdl` =
//! the corpus's mean document length. `IDF` is the "+1" (Lucene-since-2011)
//! variant, chosen DELIBERATELY over the classic `ln((N-n(t)+0.5)/(n(t)+0.5))`:
//! the classic form goes NEGATIVE for a term appearing in more than half the
//! corpus, which can make a document's score fall as it gains an extra
//! occurrence of a very common term: a surprising, unwanted answer for a
//! reranking primitive with no stopword list to filter such terms out
//! first. The `+1` variant is always `>= 0`, so every score is `>= 0` too.
//!
//! `k1` (>= 0) tunes term-frequency saturation: how much repeat
//! occurrences of a term keep adding to the score (0 = a term matters only
//! as present/absent; the conventional default 1.5 is Lucene/Elasticsearch's
//! own default). `b` (in `[0, 1]`) tunes length normalization: how much a
//! longer-than-average document is penalized (0 = no length penalty at all;
//! 1 = full normalization by relative length; the conventional default is
//! 0.75, again matching Lucene/Elasticsearch).
//!
//! # Tokenization
//!
//! `tokenize_impl::normalized_word_tokens`: the same "real word token,
//! lowercased, optionally accent-folded/stemmed" convention `tf_idf` uses
//! (UAX #29 word segments, whitespace-only segments skipped,
//! Unicode-correct lowercasing), shared rather than reimplemented here.
//! Unlike `simhash64` (a bag-of-words fingerprint, where case folding is
//! the caller's call), a ranking function comparing a query against a
//! corpus should not silently miss "Rust" against "rust"; case
//! sensitivity is not a knob this primitive exposes. `strip_accents`/
//! `stemmer` ARE knobs (both default off), applied IDENTICALLY to the
//! query and every corpus document, since scoring a query normalized
//! differently from its corpus produces meaningless scores, not just
//! imprecise ones.
//!
//! # No `deadline_ms`
//!
//! Every deadline-bearing primitive in this crate protects against
//! ADVERSARIAL-INPUT superlinear blowup (Levenshtein/Jaro's O(n·m) DP
//! tables, the Myers scans behind `similarity_ratio`/`get_close_matches`).
//! `bm25_rank` has no
//! such shape: cost is linear in total corpus token count plus
//! `corpus_size * distinct_query_terms` for scoring: both driven directly
//! and proportionally by the SIZES of the caller's own arguments, not by
//! adversarial structure within them. A caller already controls the one
//! lever that bounds the cost (how large a `corpus` they pass), so no
//! separate timeout knob is warranted.

use std::collections::HashMap;

use rust_stemmers::Stemmer;

use crate::tokenize_impl::normalized_word_tokens;

/// Term occurrence counts within one document: built once per document,
/// reused for every query term looked up against it (the `O(query terms)`
/// per-document lookup cost the module doc promises, rather than
/// re-scanning the document's token list once per query term).
fn term_counts(tokens: &[String]) -> HashMap<&str, usize> {
    let mut counts = HashMap::new();
    for token in tokens {
        *counts.entry(token.as_str()).or_insert(0) += 1;
    }
    counts
}

/// BM25-ranks every document in `corpus` against `query`: `(index, score)`
/// pairs for EVERY document (no top-k cutoff: the caller composes that,
/// same "flat primitive" convention `find_patterns` follows), sorted by
/// score descending, ties broken by original index ascending (a stable,
/// documented, deterministic order: not sort-implementation-dependent).
///
/// An empty `corpus` returns `[]`. An empty (or all-whitespace, or
/// no-real-tokens) `query` scores every document `0.0`: there are no
/// query terms to accumulate a score over, which is the correct,
/// unsurprising answer, not an error. `k1`/`b` are trusted here as
/// already-validated (`k1 >= 0.0`, `0.0 <= b <= 1.0`): the pyo3 layer's
/// job, matching this crate's usual split of caller-facing validation from
/// the trusted core.
///
/// `strip_accents`/`stemmer` are `tokenize_impl::normalized_word_tokens`'s
/// opt-in knobs, applied IDENTICALLY to `query` and every `corpus`
/// document: a query normalized differently from the corpus it's scored
/// against would make every score meaningless, not just imprecise.
pub fn bm25_rank(
    query: &str,
    corpus: &[&str],
    k1: f64,
    b: f64,
    strip_accents: bool,
    stemmer: Option<&Stemmer>,
    lemma_dict: Option<&HashMap<String, String>>,
) -> Vec<(usize, f64)> {
    if corpus.is_empty() {
        return Vec::new();
    }
    let doc_tokens: Vec<Vec<String>> = corpus
        .iter()
        .map(|doc| normalized_word_tokens(doc, strip_accents, stemmer, lemma_dict))
        .collect();
    let doc_lens: Vec<usize> = doc_tokens.iter().map(Vec::len).collect();
    let n = corpus.len();
    let avgdl = doc_lens.iter().sum::<usize>() as f64 / n as f64;

    // Document frequency per term: how many documents contain it at least
    // once (NOT total occurrences): one pass per document over its
    // distinct terms.
    let mut doc_freq: HashMap<&str, usize> = HashMap::new();
    let doc_term_counts: Vec<HashMap<&str, usize>> = doc_tokens
        .iter()
        .map(|tokens| term_counts(tokens))
        .collect();
    for counts in &doc_term_counts {
        for term in counts.keys() {
            *doc_freq.entry(term).or_insert(0) += 1;
        }
    }

    // Distinct query terms: a repeated query word contributes its IDF
    // term ONCE (the module doc's Robertson/Spärck-Jones convention), not
    // once per occurrence.
    let query_tokens = normalized_word_tokens(query, strip_accents, stemmer, lemma_dict);
    let mut seen = std::collections::HashSet::new();
    let query_terms: Vec<&str> = query_tokens
        .iter()
        .map(String::as_str)
        .filter(|t| seen.insert(*t))
        .collect();

    let mut scores: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let doc_len = doc_lens[i] as f64;
            let len_norm = 1.0 - b + b * (doc_len / avgdl);
            // `Iterator::sum`'s f64 identity is IEEE-754 `-0.0` (faithful to
            // the standard, but a needlessly surprising thing for a
            // documented "scores 0.0" case to print); the trailing `+ 0.0`
            // normalizes an all-zero-terms sum back to `+0.0` (IEEE-754:
            // `-0.0 + 0.0 == 0.0`) without touching any nonzero result.
            let score: f64 = query_terms
                .iter()
                .map(|term| {
                    let n_t = doc_freq.get(term).copied().unwrap_or(0) as f64;
                    let idf = ((n as f64 - n_t + 0.5) / (n_t + 0.5) + 1.0).ln();
                    let f_td = doc_term_counts[i].get(term).copied().unwrap_or(0) as f64;
                    if f_td == 0.0 {
                        0.0
                    } else {
                        idf * (f_td * (k1 + 1.0)) / (f_td + k1 * len_norm)
                    }
                })
                .sum::<f64>()
                + 0.0;
            (i, score)
        })
        .collect();

    scores.sort_by(|(i_a, s_a), (i_b, s_b)| {
        s_b.partial_cmp(s_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(i_a.cmp(i_b))
    });
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn empty_corpus_is_no_scores() {
        assert_eq!(
            bm25_rank("anything", &[], 1.5, 0.75, false, None, None),
            Vec::new()
        );
    }

    #[test]
    fn empty_query_scores_everything_zero_and_keeps_index_order() {
        let corpus = ["the cat sat", "a dog ran", "quantum entanglement"];
        assert_eq!(
            bm25_rank("", &corpus, 1.5, 0.75, false, None, None),
            vec![(0, 0.0), (1, 0.0), (2, 0.0)]
        );
    }

    #[test]
    fn zero_scores_are_positive_zero_not_negative_zero() {
        // `Iterator::sum`'s f64 identity is IEEE-754 `-0.0` for an empty
        // sequence: `-0.0 == 0.0` so `assert_eq!` above can't catch a
        // regression here; check the actual sign bit.
        for (_, score) in bm25_rank("", &["a", "b"], 1.5, 0.75, false, None, None) {
            assert!(!score.is_sign_negative(), "{score} has a negative sign bit");
        }
    }

    #[test]
    fn a_document_containing_the_query_term_outranks_one_that_does_not() {
        let corpus = ["the cat sat on the mat", "a dog ran in the park"];
        let ranked = bm25_rank("cat", &corpus, 1.5, 0.75, false, None, None);
        assert_eq!(ranked[0].0, 0);
        assert!(ranked[0].1 > 0.0);
        assert_eq!(ranked[1], (1, 0.0));
    }

    #[test]
    fn scores_are_non_negative_for_the_plus_one_idf_variant() {
        // A term in EVERY document (n(t) == N) is the classic-formula
        // negative-IDF case; the +1 variant must still stay >= 0.
        let corpus = [
            "common word here",
            "common word there",
            "common word everywhere",
        ];
        for (_, score) in bm25_rank("common", &corpus, 1.5, 0.75, false, None, None) {
            assert!(score >= 0.0, "{score} < 0");
        }
    }

    #[test]
    fn single_document_corpus_does_not_panic_and_scores_sensibly() {
        // avgdl == that document's own length, so the b-normalization term
        // is exactly 1.0 (no length penalty possible with nothing to
        // compare against); must not divide by zero or misbehave.
        let corpus = ["the only document here"];
        let ranked = bm25_rank("document", &corpus, 1.5, 0.75, false, None, None);
        assert_eq!(ranked.len(), 1);
        assert!(ranked[0].1 > 0.0);
    }

    #[test]
    fn query_term_absent_from_every_document_contributes_nothing() {
        let corpus = ["the cat sat", "a dog ran"];
        assert_eq!(
            bm25_rank("nonexistentword", &corpus, 1.5, 0.75, false, None, None),
            vec![(0, 0.0), (1, 0.0)]
        );
    }

    #[test]
    fn repeated_query_term_counts_once_not_per_occurrence() {
        let corpus = ["cat cat cat", "cat dog bird"];
        let once = bm25_rank("cat", &corpus, 1.5, 0.75, false, None, None);
        let repeated = bm25_rank("cat cat cat cat cat", &corpus, 1.5, 0.75, false, None, None);
        assert_eq!(once, repeated);
    }

    #[test]
    fn ties_break_by_ascending_original_index() {
        let corpus = ["cat cat", "cat cat", "cat cat"];
        let ranked = bm25_rank("cat", &corpus, 1.5, 0.75, false, None, None);
        let scores: Vec<f64> = ranked.iter().map(|(_, s)| *s).collect();
        assert!(close(scores[0], scores[1]) && close(scores[1], scores[2]));
        assert_eq!(
            ranked.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn determinism_same_input_same_output() {
        let corpus = [
            "the quick brown fox",
            "jumps over the lazy dog",
            "the fox is quick",
        ];
        assert_eq!(
            bm25_rank("quick fox", &corpus, 1.5, 0.75, false, None, None),
            bm25_rank("quick fox", &corpus, 1.5, 0.75, false, None, None)
        );
    }

    #[test]
    fn case_insensitive_matching() {
        let corpus = ["Rust is great", "python is fine"];
        let ranked = bm25_rank("RUST", &corpus, 1.5, 0.75, false, None, None);
        assert_eq!(ranked[0].0, 0);
        assert!(ranked[0].1 > 0.0);
    }

    #[test]
    fn hand_computed_two_document_vector() {
        // corpus[0]="cat dog" (len 2), corpus[1]="cat cat cat" (len 3).
        // avgdl = 2.5. N=2. query="cat": n(t)=2 (both docs contain it).
        // IDF = ln((2 - 2 + 0.5)/(2 + 0.5) + 1) = ln(0.5/2.5 + 1) = ln(1.2).
        let idf = (0.2_f64 + 1.0).ln();
        // doc0: f=1, |D|=2, len_norm = 1 - 0.75 + 0.75*(2/2.5) = 0.25 + 0.6 = 0.85
        // score0 = idf * (1*2.5) / (1 + 1.5*0.85) = idf * 2.5 / 2.275
        let len_norm0 = 0.25 + 0.75 * (2.0 / 2.5);
        let expected0 = idf * (1.0 * 2.5) / (1.0 + 1.5 * len_norm0);
        // doc1: f=3, |D|=3, len_norm = 0.25 + 0.75*(3/2.5) = 0.25 + 0.9 = 1.15
        let len_norm1 = 0.25 + 0.75 * (3.0 / 2.5);
        let expected1 = idf * (3.0 * 2.5) / (3.0 + 1.5 * len_norm1);
        let corpus = ["cat dog", "cat cat cat"];
        let ranked = bm25_rank("cat", &corpus, 1.5, 0.75, false, None, None);
        let by_index: HashMap<usize, f64> = ranked.into_iter().collect();
        assert!(
            close(by_index[&0], expected0),
            "{} vs {}",
            by_index[&0],
            expected0
        );
        assert!(
            close(by_index[&1], expected1),
            "{} vs {}",
            by_index[&1],
            expected1
        );
    }
}
