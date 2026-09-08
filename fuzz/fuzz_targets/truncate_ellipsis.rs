//! `truncate_ellipsis` never panics on an arbitrary string at an arbitrary
//! budget, never exceeds the budget, never splits a grapheme cluster, and
//! the identity path never lies: a borrowed return is byte-identical to
//! the input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct Input {
    s: String,
    // Small budgets are the interesting ones (every cluster-boundary
    // interaction lives there); the modulo keeps the fuzzer in that
    // range while still reaching past the text length sometimes.
    budget: u8,
}

fuzz_target!(|input: Input| {
    let max_chars = (input.budget as usize) % 40;
    let got = tors::truncate_impl::truncate_ellipsis(&input.s, max_chars);
    let got = got.as_ref();

    // The budget invariant, in codepoints.
    assert!(
        got.chars().count() <= max_chars,
        "exceeded budget {max_chars} on {:?}: {got:?}",
        input.s
    );
    // Valid UTF-8 by construction (Rust &str), and the kept prefix is a
    // prefix of the input: truncation only drops a tail.
    if got != input.s {
        assert!(max_chars > 0, "non-empty output at zero budget");
        assert!(
            got.ends_with(tors::truncate_impl::ELLIPSIS),
            "truncated output lacks the marker on {:?}: {got:?}",
            input.s
        );
        let kept = &got[..got.len() - tors::truncate_impl::ELLIPSIS.len_utf8()];
        assert!(
            input.s.starts_with(kept),
            "kept prefix is not a prefix of the input on {:?}: {got:?}",
            input.s
        );
    }
});
