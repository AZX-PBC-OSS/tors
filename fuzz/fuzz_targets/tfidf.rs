//! `tf_idf` never panics on arbitrary corpora, returns one entry per input
//! document in input order, each document's terms sorted alphabetically
//! with all scores finite.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    corpus: Vec<String>,
    strip_accents: bool,
}

fuzz_target!(|input: Input| {
    let corpus: Vec<&str> = input.corpus.iter().map(String::as_str).collect();
    if corpus.is_empty() {
        return; // the empty-corpus early return is pinned by crate tests
    }
    let scored = tors::tfidf_impl::tf_idf(&corpus, input.strip_accents, None, None);
    assert_eq!(scored.len(), corpus.len(), "one entry per document");
    for (doc_idx, terms) in scored.iter().enumerate() {
        for w in terms.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "terms not sorted alphabetically in doc {doc_idx}: {w:?}"
            );
        }
        // No non-emptiness assert on terms: a word segment made only of
        // combining marks is not whitespace, so `real_word_segments` keeps
        // it, and `strip_accents` folds it to the empty string (found by
        // this very target; contract-consistent with the documented fold,
        // reported rather than asserted against).
        for (_term, score) in terms {
            assert!(score.is_finite(), "non-finite score in doc {doc_idx}");
        }
    }
});
