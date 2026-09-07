//! Stateless TF-IDF, the pure-Rust core of `tors.tf_idf`.
//!
//! Every TF-IDF crate on crates.io is stale or effectively abandoned (the
//! module-decisions survey: `rust-tfidf` last released 2021, `tfidf` 2016,
//! `tfidf-text-summarizer` narrow and unmaintained). The math itself is a
//! few dozen lines with no ML machinery, squarely in this crate's
//! surgical-hand-roll tradition (`html_impl`, `fence_impl`), not a
//! dependency pull. The real gap this fills: Python's stdlib has no TF-IDF
//! at all, and `scikit-learn`'s `TfidfVectorizer` drags in numpy/scipy for
//! a lightweight pipeline that just wants document similarity or keyword
//! extraction.
//!
//! Stateless by design: one call, one corpus, no persisted
//! vectorizer/vocabulary object between calls: matching this crate's flat,
//! single-call primitive model rather than a `fit`/`transform` API.
//!
//! **Tokenization**: UAX #29 word segments (`segmentation_impl::word_bounds`,
//! the same machinery `chunk_by_words` already filters through) restricted
//! to segments carrying at least one non-whitespace codepoint. A "term" is
//! a real token, not a `word_bounds` segment (which gives inter-word
//! whitespace its own segment). Terms are lowercased via Rust's
//! Unicode-correct `str::to_lowercase` (not an ASCII-only fold) before
//! counting, the near-universal TF-IDF convention: `"Cat"` and `"cat"` are
//! the same term.
//!
//! **TF** (per document `d`, term `t`): the RAW count of `t` in `d`,
//! `tf(t, d) = count of t in d`. Not length-normalized; a caller wanting
//! `tf / len(d)` divides the returned raw count themselves (the raw count
//! is the more broadly reusable primitive; normalizing here would silently
//! discard information a caller might want, e.g. total term count is
//! itself useful).
//!
//! **IDF** (per term `t`, corpus size `N`, document frequency `df(t)` = the
//! number of documents `t` appears in at least once): the SMOOTHED formula
//! `idf(t) = ln((1 + N) / (1 + df(t))) + 1`, scikit-learn's own default
//! (`smooth_idf=True`: "the constant 1 is added to the numerator and
//! denominator... as if an extra document was seen containing every term
//! in the collection exactly once") rather than the textbook
//! `ln(N / df(t))`. The unsmoothed formula gives a term
//! appearing in EVERY document an IDF of exactly `ln(1) = 0`, so its
//! TF-IDF score is `0` regardless of how often it occurs, an unhelpfully
//! sharp cliff for exactly the "practically universal term" case a caller
//! most wants a small-but-nonzero weight for. Smoothing keeps that case
//! strictly positive (`ln((1+N)/(1+N)) + 1 = 1` exactly, for the
//! universal-term case `df(t) = N`) while still monotonically favoring
//! rarer terms.
//!
//! **score**(t, d) = `tf(t, d) * idf(t)`.
//!
//! **Output**: sparse, one `Vec<(String, f64)>` per input document, only
//! terms that actually occur in that document (never a
//! vocabulary-size-by-corpus-size dense structure). Term order within each
//! document's list is alphabetical (`String`'s own `Ord`), a deterministic
//! tie-break independent of any hash-map iteration order.
//!
//! What this is NOT: no stemming, no lemmatization, no stop-word removal,
//! no n-grams. It is a lightweight keyword-weighting/document-similarity
//! primitive over real (lowercased) word tokens, not a full NLP pipeline
//! stage.

use std::collections::HashMap;

use rust_stemmers::Stemmer;

use crate::tokenize_impl::normalized_word_tokens;

/// Tokenize one document into its lowercased term-count map: the per-term
/// raw frequency `tf(t, d)` before any corpus-wide IDF is known. Shares
/// `tokenize_impl::normalized_word_tokens` with `bm25_rank` rather than
/// re-deriving the same "real word token, lowercased, optionally
/// accent-folded/stemmed" walk.
fn term_counts(
    text: &str,
    strip_accents: bool,
    stemmer: Option<&Stemmer>,
    lemma_dict: Option<&HashMap<String, String>>,
) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for term in normalized_word_tokens(text, strip_accents, stemmer, lemma_dict) {
        *counts.entry(term).or_insert(0) += 1;
    }
    counts
}

/// `tf_idf(corpus) -> Vec<Vec<(term, score)>>`, one entry per input
/// document, in input order; each document's list holds only the terms
/// that occur in it, sorted alphabetically. Empty corpus -> `[]`; an
/// empty-string document -> `[]` at its position (still present, so the
/// output always has `corpus.len()` entries).
///
/// `strip_accents`/`stemmer` are `tokenize_impl::normalized_word_tokens`'s
/// opt-in normalization knobs, both OFF by default (`false`/`None`
/// reproduce the original lowercase-only tokenization exactly).
pub fn tf_idf(
    corpus: &[&str],
    strip_accents: bool,
    stemmer: Option<&Stemmer>,
    lemma_dict: Option<&HashMap<String, String>>,
) -> Vec<Vec<(String, f64)>> {
    if corpus.is_empty() {
        return Vec::new();
    }
    let n = corpus.len();
    let per_doc: Vec<HashMap<String, usize>> = corpus
        .iter()
        .map(|doc| term_counts(doc, strip_accents, stemmer, lemma_dict))
        .collect();

    let mut document_frequency: HashMap<String, usize> = HashMap::new();
    for doc_counts in &per_doc {
        for term in doc_counts.keys() {
            *document_frequency.entry(term.clone()).or_insert(0) += 1;
        }
    }

    let idf: HashMap<String, f64> = document_frequency
        .into_iter()
        .map(|(term, df)| {
            let idf = ((1.0 + n as f64) / (1.0 + df as f64)).ln() + 1.0;
            (term, idf)
        })
        .collect();

    per_doc
        .into_iter()
        .map(|doc_counts| {
            let mut scored: Vec<(String, f64)> = doc_counts
                .into_iter()
                .map(|(term, tf)| {
                    let score = tf as f64 * idf[&term];
                    (term, score)
                })
                .collect();
            scored.sort_by(|a, b| a.0.cmp(&b.0));
            scored
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_corpus_is_empty() {
        assert_eq!(
            tf_idf(&[], false, None, None),
            Vec::<Vec<(String, f64)>>::new()
        );
    }

    #[test]
    fn empty_document_yields_no_terms_but_keeps_its_slot() {
        let result = tf_idf(&["cat sat", ""], false, None, None);
        assert_eq!(result.len(), 2);
        assert_eq!(result[1], Vec::new());
    }

    #[test]
    fn case_folding_merges_terms() {
        let result = tf_idf(&["Cat cat CAT"], false, None, None);
        assert_eq!(result[0].len(), 1);
        assert_eq!(result[0][0].0, "cat");
        // tf = 3, df = 1 of N = 1 -> idf = ln((1+1)/(1+1)) + 1 = 1.0.
        let expected_idf = ((1.0 + 1.0f64) / (1.0 + 1.0)).ln() + 1.0;
        assert!((result[0][0].1 - 3.0 * expected_idf).abs() < 1e-12);
    }

    #[test]
    fn rare_term_outscores_a_universal_term_at_equal_raw_frequency() {
        // "cat" in every doc (universal); "dog" in only doc 0. Same raw
        // count (1) in doc 0 for both -> the IDF difference alone must
        // make "dog" score higher there.
        let result = tf_idf(&["cat dog", "cat bird", "cat fish"], false, None, None);
        let doc0: HashMap<&str, f64> = result[0].iter().map(|(t, s)| (t.as_str(), *s)).collect();
        assert!(doc0["dog"] > doc0["cat"]);
    }

    #[test]
    fn universal_term_scores_strictly_positive_not_zero() {
        let result = tf_idf(&["cat", "cat", "cat"], false, None, None);
        assert!(result[0][0].1 > 0.0);
    }

    #[test]
    fn terms_are_alphabetically_sorted_within_a_document() {
        let result = tf_idf(&["zebra apple mango"], false, None, None);
        let terms: Vec<&str> = result[0].iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(terms, vec!["apple", "mango", "zebra"]);
    }

    #[test]
    fn determinism() {
        let corpus = ["the quick brown fox", "the lazy dog", "the fox and the dog"];
        assert_eq!(
            tf_idf(&corpus, false, None, None),
            tf_idf(&corpus, false, None, None)
        );
    }

    #[test]
    fn all_scores_are_non_negative() {
        let corpus = ["a b c", "b c d", "c d e", "a a a a"];
        for doc in tf_idf(&corpus, false, None, None) {
            for (_, score) in doc {
                assert!(score >= 0.0);
            }
        }
    }
}
