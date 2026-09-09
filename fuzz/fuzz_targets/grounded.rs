//! `is_grounded`'s core never panics on any input, and the contract the
//! exact-containment floor added holds under raw adversarial input: a
//! claim spliced VERBATIM into a source at ANY offset is grounded by the
//! fuzzy path at ANY threshold — including under a deadline tight enough
//! that the pre-floor windowed scan would have expired (the floor runs
//! before deadline setup). The raw-pair lane pins the two directions of
//! the threshold-1.0 equivalence — fuzzy@1.0 is EXACTLY containment, no
//! window scores 1.0 without the claim verbatim — plus threshold
//! monotonicity. This is the target that would have caught the original
//! bug: a verbatim claim at an offset unaligned with the stride-`L/2`
//! windows scored ~0.75 < 0.85 and was wrongly reported ungrounded.
//!
//! Sizes are capped (the windowed Myers scan is superlinear on hard
//! pairs), the same discipline `fuzz_targets/diff.rs` applies, so the
//! fuzzer explores deep small shapes instead of stalling on huge ones.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
enum Input {
    /// Arbitrary raw claim/source pair: panic-freedom, the threshold-1.0
    /// equivalence, and monotonicity.
    Raw { claim: String, source: String },
    /// A claim embedded verbatim at an arbitrary offset: the superset
    /// invariant, at every threshold and under an effectively-zero budget.
    Spliced { claim: String, lead: u16, tail: u8 },
    /// A claim with ONE character mutated, embedded at an arbitrary offset:
    /// the refinement guarantee (r = (L-1)/L >= 0.85 + 1/32 whenever the
    /// claim is 12+ chars), wherever the region sits.
    NearSpliced {
        claim: String,
        lead: u16,
        tail: u8,
        at: u8,
        sub: u8,
    },
}

fuzz_target!(|input: Input| {
    match input {
        Input::Raw { claim, source } => {
            if claim.chars().count() + source.chars().count() > 4096 {
                return;
            }
            // Panic-freedom on any pair, budgeted or not.
            let _ = tors::grounded_impl::is_grounded_fuzzy(&claim, &source, 0.85, None);
            let _ = tors::grounded_impl::is_grounded_fuzzy(&claim, &source, 1.0, Some(0.001));
            // Threshold 1.0 is EXACTLY containment, both directions: a
            // verbatim claim must clear it (the floor), and nothing else
            // ever may (a window scores 1.0 only by equaling the claim,
            // which is containment; truncated tail windows and the
            // source-shorter direct comparison are both strictly below).
            let at_one = tors::grounded_impl::is_grounded_fuzzy(&claim, &source, 1.0, None);
            assert_eq!(
                at_one,
                Ok(source.contains(&claim)),
                "threshold-1.0 is not containment"
            );
            // Monotonicity: clearing a higher threshold clears every lower
            // one (a single scalar score compared against threshold).
            let hi = tors::grounded_impl::is_grounded_fuzzy(&claim, &source, 0.9, None);
            let lo = tors::grounded_impl::is_grounded_fuzzy(&claim, &source, 0.4, None);
            if hi == Ok(true) {
                assert_eq!(lo, Ok(true), "cleared 0.9 but not 0.4");
            }
        }
        Input::Spliced { claim, lead, tail } => {
            if claim.chars().count() > 1024 {
                return;
            }
            let source = format!(
                "{}{}{}",
                "q".repeat(lead as usize % 1024),
                claim,
                "q".repeat(tail as usize % 128)
            );
            // The splice is verbatim by construction; the floor must
            // ground it at EVERY threshold, whatever the offset.
            assert!(source.contains(&claim));
            for threshold in [1.0, 0.85, 0.5] {
                let got = tors::grounded_impl::is_grounded_fuzzy(&claim, &source, threshold, None);
                assert_eq!(
                    got,
                    Ok(true),
                    "verbatim claim ungrounded at threshold {threshold}"
                );
            }
            // And before the deadline applies: an effectively-zero budget
            // that would expire the windowed scan must still ground a
            // verbatim claim — never DeadlineExceeded.
            let got = tors::grounded_impl::is_grounded_fuzzy(&claim, &source, 1.0, Some(0.0001));
            assert_eq!(
                got,
                Ok(true),
                "verbatim claim timed out under a tiny budget"
            );
        }
        Input::NearSpliced {
            claim,
            lead,
            tail,
            at,
            sub,
        } => {
            let chars: Vec<char> = claim.chars().collect();
            if chars.len() > 1024 || chars.is_empty() {
                return;
            }
            // One deterministic substitution (from a 3-char alphabet; a
            // no-op substitution just degenerates to the verbatim lane).
            let i = at as usize % chars.len();
            let mut near: Vec<char> = chars.clone();
            near[i] = ['q', 'z', '9'][sub as usize % 3];
            let near: String = near.into_iter().collect();
            let source = format!(
                "{}{}{}",
                "q".repeat(lead as usize % 1024),
                near,
                "q".repeat(tail as usize % 128)
            );
            // r = (L-1)/L >= 0.85 + 1/32 whenever L >= 12: the refinement
            // pass must find the near-match region at ANY offset.
            if chars.len() >= 12 {
                let got = tors::grounded_impl::is_grounded_fuzzy(&claim, &source, 0.85, None);
                assert_eq!(
                    got,
                    Ok(true),
                    "one-substitution near match ungrounded at lead {}",
                    lead as usize % 1024
                );
            }
            // Panic-freedom on the near shape at any size, budgeted or not.
            let _ = tors::grounded_impl::is_grounded_fuzzy(&claim, &source, 0.85, Some(0.001));
        }
    }
});
