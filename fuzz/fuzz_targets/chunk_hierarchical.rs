//! `chunk_hierarchical` never panics on arbitrary text, arbitrary
//! caller-supplied separator lists (including empty strings, overlapping
//! separators, and separators that never match), and arbitrary
//! `max_chars`/`overlap` within the validated range.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    text: String,
    max_chars: std::num::NonZeroU16,
    overlap_raw: u16,
    separators: Option<Vec<String>>,
}

fuzz_target!(|input: Input| {
    let max_chars = input.max_chars.get() as usize;
    // The function's own precondition is `overlap < max_chars`; clamp
    // rather than skip so the harness still explores the boundary.
    let overlap = (input.overlap_raw as usize) % max_chars.max(1);
    let sep_refs: Option<Vec<&str>> = input
        .separators
        .as_ref()
        .map(|v| v.iter().map(String::as_str).collect());
    let sep_slice: Option<&[&str]> = sep_refs.as_deref();

    let chunks = tors::chunk_hierarchical_impl::chunk_hierarchical(
        &input.text,
        max_chars,
        sep_slice,
        overlap,
    );

    // Coverage: chunks must be non-decreasing, in-bounds, and never empty
    // (a zero-length chunk would be a real bug — forward progress
    // guarantees no chunk can be empty).
    let total = input.text.chars().count();
    for (start, end) in &chunks {
        assert!(start < end, "empty or inverted chunk: ({start}, {end})");
        assert!(*end <= total, "chunk end {end} exceeds text length {total}");
    }
});
