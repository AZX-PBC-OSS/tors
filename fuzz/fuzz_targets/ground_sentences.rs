//! `ground_sentences`' core never panics on any input, and the two
//! invariants a consumer's correctness stands on hold under raw
//! adversarial input:
//!
//! 1. The offset round-trip: every returned sentence's CHARACTER offsets
//!    slice the ORIGINAL text (via Python codepoint indexing, which Rust
//!    models as `chars().collect()[start..end]`) to EXACTLY the returned
//!    text — the property the FFI's other side pins.
//! 2. Batch hygiene: the sentences are in position order (pairwise
//!    non-overlapping), every score is in `[0, 1]`, and the aggregate is
//!    exactly the max per-sentence score — whatever the query/text
//!    geometry. Degenerate inputs are valid answers (never a panic, never
//!    an error): the empty text yields the empty result; an empty or
//!    token-free query scores every sentence 0.0.
//! 3. The max_chars budget bounds the scored WINDOW, never the report:
//!    under any `max_chars` the returned spans are the unclamped spans.
//!
//! Sizes are capped so the fuzzer explores deep small shapes instead of
//! stalling on huge ones, the same discipline `fuzz_targets/grounding.rs`
//! applies.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Params {
    text: String,
    query: String,
    max_chars: Option<u16>,
}

fuzz_target!(|input: Params| {
    if input.text.chars().count() + input.query.chars().count() > 4096 {
        return;
    }
    let max_chars = input
        .max_chars
        .map(|mc| (mc % 512) as usize)
        .filter(|mc| *mc > 0);
    let result = tors::grounding_impl::ground_sentences(&input.query, &input.text, max_chars);

    let chars: Vec<char> = input.text.chars().collect();
    let mut max_score = 0.0f64;
    for (earlier, later) in result.sentences.iter().zip(result.sentences.iter().skip(1)) {
        assert!(
            earlier.start < later.start && earlier.end <= later.start,
            "sentences overlap or misordered: {earlier:?} / {later:?}"
        );
    }
    for sentence in &result.sentences {
        assert!(
            sentence.start < sentence.end && sentence.end <= chars.len(),
            "offsets out of range: {sentence:?} for {} chars",
            chars.len()
        );
        let sliced: String = chars[sentence.start..sentence.end].iter().collect();
        assert_eq!(
            sliced, sentence.text,
            "offset round-trip broken for query {:?}",
            input.query
        );
        assert!(
            (0.0..=1.0).contains(&sentence.score),
            "score out of range: {sentence:?}"
        );
        max_score = max_score.max(sentence.score);
    }
    assert!(
        (0.0..=1.0).contains(&result.score),
        "aggregate score out of range: {:?}",
        result.score
    );
    assert_eq!(
        result.score, max_score,
        "the aggregate is not the max per-sentence score"
    );
    // The budget bounds the scored window, never the reported spans:
    // under a different max_chars the same text/query yields the same
    // sentence spans (the spans are the segmentation's).
    if max_chars.is_some() {
        let whole = tors::grounding_impl::ground_sentences(&input.query, &input.text, None);
        assert_eq!(
            result
                .sentences
                .iter()
                .map(|s| (s.start, s.end, s.text.as_str()))
                .collect::<Vec<_>>(),
            whole
                .sentences
                .iter()
                .map(|s| (s.start, s.end, s.text.as_str()))
                .collect::<Vec<_>>(),
            "max_chars changed the reported spans"
        );
    }
});
