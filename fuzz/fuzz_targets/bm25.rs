//! `bm25_rank` never panics on arbitrary queries and corpora within the
//! validated parameter domain (`k1 >= 0`, `0 <= b <= 1`, derived here from
//! the input), ranks EVERY document (no top-k cutoff), in score-descending
//! order with ties broken by original index ascending, all scores finite.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    query: String,
    corpus: Vec<String>,
}

fuzz_target!(|input: Input| {
    let corpus: Vec<&str> = input.corpus.iter().map(String::as_str).collect();
    if corpus.is_empty() {
        return; // the empty-corpus early return is pinned by crate tests
    }
    // k1/b inside the pyo3 layer's validated domain, derived from the
    // input so the fuzzer steers them.
    let k1 = (input.query.as_bytes().first().copied().unwrap_or(0) % 40) as f64;
    let b = (input.query.as_bytes().get(1).copied().unwrap_or(0) % 101) as f64 / 100.0;

    let ranked = tors::bm25_impl::bm25_rank(&input.query, &corpus, k1, b, false, None, None);
    assert_eq!(ranked.len(), corpus.len(), "every document must be ranked");
    for w in ranked.windows(2) {
        assert!(w[0].1 >= w[1].1, "scores not descending: {w:?}");
        if w[0].1 == w[1].1 {
            assert!(
                w[0].0 < w[1].0,
                "ties must break by original index ascending: {w:?}"
            );
        }
    }
    let mut indices: Vec<usize> = ranked.iter().map(|(i, _)| *i).collect();
    indices.sort_unstable();
    assert_eq!(
        indices,
        (0..corpus.len()).collect::<Vec<_>>(),
        "ranked indices are not a permutation of the corpus"
    );
    for (idx, score) in &ranked {
        assert!(score.is_finite(), "non-finite score at index {idx}");
        assert!(*idx < corpus.len(), "index out of range: {idx}");
    }
});
