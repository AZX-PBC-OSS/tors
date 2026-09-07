//! `extract_code_blocks`/`strip_code_fences`/`dedent` never panic on
//! arbitrary strings, including pathological fence-like sequences
//! (thousands of backticks, malformed nesting, mixed tilde/backtick runs).

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|s: &str| {
    let blocks = tors::fence_impl::extract_code_blocks(s, None);
    // Every block's span must be valid char-boundary indices into `s`,
    // never out of bounds and never end < start.
    for block in &blocks {
        assert!(block.start <= block.end, "block span inverted: {block:?}");
        assert!(
            s.get(byte_offset(s, block.start)..byte_offset(s, block.end))
                .is_some(),
            "block span not a valid slice: {block:?}"
        );
    }

    let _ = tors::fence_impl::strip_code_fences(s);
    let _ = tors::fence_impl::dedent(s);
});

/// `extract_code_blocks` reports codepoint offsets; convert to a byte
/// offset for the `s.get(..)` boundary check (panics here would themselves
/// be a real bug — an out-of-range `nth` returns `None`, not a panic, so
/// this stays a benign check on failure).
fn byte_offset(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
}
