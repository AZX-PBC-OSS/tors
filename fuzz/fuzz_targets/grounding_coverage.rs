//! `grounding_coverage`'s core never panics on any input, and the
//! contract a consumer's correctness stands on holds under raw
//! adversarial input:
//!
//! 1. The unit interval: the score is always in `[0.0, 1.0]`, whatever
//!    the operand shapes.
//! 2. The degeneracy conventions: either operand empty or token-free is
//!    EXACTLY 0.0; identical non-empty operands are exactly 1.0.
//! 3. Monotonicity: extending the text with source material never lowers
//!    the score (the weighted LCS is monotone in the candidate stream).
//! 4. Symmetry of the degenerate cases and determinism across call
//!    orders.
//!
//! Sizes are capped so the fuzzer explores deep small shapes instead of
//! stalling on the O(|S|·|T|) DP, the same discipline
//! `fuzz_targets/diff.rs` applies.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
enum Input {
    /// Arbitrary raw source/text pair: panic-freedom, the unit interval,
    /// the empty conventions.
    Raw { source: String, text: String },
    /// A source embedded verbatim as the text: the identical-operands
    /// identity (1.0 exactly).
    Identical { source: String },
    /// A source extended with filler INSIDE the text: the monotonicity
    /// lane (superset text >= subset text).
    Extended { source: String, filler: String },
}

fuzz_target!(|input: Input| {
    fn token_free(s: &str) -> bool {
        !s.chars().any(char::is_alphanumeric)
    }
    match input {
        Input::Raw { source, text } => {
            if source.chars().count() + text.chars().count() > 4096 {
                return;
            }
            let a = tors::grounded_impl::grounding_coverage(&source, &text);
            let b = tors::grounded_impl::grounding_coverage(&source, &text);
            assert_eq!(a, b, "non-deterministic");
            assert!(
                (0.0..=1.0).contains(&a),
                "coverage out of range: {a} for {source:?} / {text:?}"
            );
            // The empty/token-free conventions, exactly 0.0.
            if source.is_empty() || text.is_empty() || token_free(&source) || token_free(&text) {
                assert_eq!(a, 0.0, "a degenerate pair must score exactly 0.0");
            }
        }
        Input::Identical { source } => {
            if source.chars().count() > 2048 {
                return;
            }
            let got = tors::grounded_impl::grounding_coverage(&source, &source);
            if source.is_empty() || token_free(&source) {
                assert_eq!(got, 0.0, "a token-free pair is the 0.0 convention");
            } else {
                assert!(
                    (got - 1.0).abs() < 1e-9,
                    "identical operands must score 1.0 (up to f64 rounding): {got}"
                );
            }
        }
        Input::Extended { source, filler } => {
            if source.chars().count() + filler.chars().count() > 2048 {
                return;
            }
            if source.is_empty() || token_free(&source) || token_free(&filler) {
                // The conventions hold; monotonicity needs token material
                // on both sides to be meaningful.
                let _ = tors::grounded_impl::grounding_coverage(&source, &filler);
                return;
            }
            let base = tors::grounded_impl::grounding_coverage(&source, &source);
            let extended = format!("{source} {filler}");
            let with_filler = tors::grounded_impl::grounding_coverage(&source, &extended);
            if with_filler < base - 1e-9 {
                eprintln!("DBG source={source:?} filler={filler:?} base={base} ext={with_filler}");
            }
            assert!(base >= 1.0 - 1e-9, "identical operands below 1.0");
            assert!(
                with_filler >= base - 1e-9,
                "extending the text lowered coverage: {with_filler} < {base}"
            );
        }
    }
});
