//! `highlight`'s core never panics on any input, and the two invariants a
//! consumer's correctness stands on hold under raw adversarial input:
//!
//! 1. The offset round-trip: every returned snippet's CHARACTER offsets
//!    slice the ORIGINAL text (via Python codepoint indexing, which Rust
//!    models as `chars().collect()[start..end]`) to EXACTLY the returned
//!    text — the property the FFI's other side pins.
//! 2. Selection hygiene: snippets are pairwise non-overlapping and in
//!    position order, whatever the candidate geometry.
//!
//! Sizes are capped so the fuzzer explores deep small shapes instead of
//! stalling on huge ones, the same discipline `fuzz_targets/diff.rs`
//! applies.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Params {
    query: String,
    text: String,
    max_snippets: u8,
    max_chars: u16,
}

fuzz_target!(|input: Params| {
    if input.query.chars().count() + input.text.chars().count() > 4096 {
        return;
    }
    let max_snippets = (input.max_snippets % 8) as usize;
    let max_chars = (input.max_chars % 512) as usize;
    let result =
        tors::grounding_impl::highlight(&input.query, &input.text, max_snippets, max_chars);

    let chars: Vec<char> = input.text.chars().collect();
    for (earlier, later) in result.snippets.iter().zip(result.snippets.iter().skip(1)) {
        assert!(
            earlier.end <= later.start,
            "snippets overlap: {earlier:?} / {later:?}"
        );
    }
    for snippet in &result.snippets {
        assert!(
            snippet.start < snippet.end && snippet.end <= chars.len(),
            "offsets out of range: {snippet:?} for {} chars",
            chars.len()
        );
        let sliced: String = chars[snippet.start..snippet.end].iter().collect();
        assert_eq!(
            sliced, snippet.text,
            "offset round-trip broken for query {:?}",
            input.query
        );
        assert!(
            (0.0..=1.0).contains(&snippet.score),
            "score out of range: {snippet:?}"
        );
    }
    assert!(
        (0.0..=1.0).contains(&result.score),
        "overall score out of range: {:?}",
        result.score
    );
});
