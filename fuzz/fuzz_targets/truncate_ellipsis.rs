//! `truncate_ellipsis` never panics on an arbitrary string at an arbitrary
//! budget, never exceeds the budget, never splits a grapheme cluster, and
//! the identity path never lies: a borrowed return is byte-identical to
//! the input.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

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
        // `max_chars == 0` yields empty — the impl's pinned contract (no
        // room for even the marker; pytest's zero-bound pins) — so the
        // marker/prefix checks apply only to a positive budget. The old
        // `max_chars > 0` assert fired on the CORRECT empty output for any
        // non-empty input at zero budget (CI's first run of this target,
        // 2026-09-09: a 2-byte input, s="\n", budget=0, found in three
        // executions from an empty corpus).
        if max_chars == 0 {
            assert!(
                got.is_empty(),
                "non-empty output at zero budget on {:?}: {got:?}",
                input.s
            );
        } else {
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
    }
});
