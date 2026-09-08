//! Lexical claim-grounding check: the pure-Rust core of `tors.is_grounded`.
//!
//! Answers one narrow question: does `source` actually CONTAIN `claim`, or
//! something close enough to it? This is a LEXICAL check, not semantic:
//! there is no NLI/entailment model here, and the fuzzy path's score is a
//! difflib-SHAPED character ratio (the `2*M/T` formula) against a bounded
//! scan of `source`, not a general string-similarity oracle and NOT a
//! bit-exact difflib differential: `M` comes from `similar`'s Myers engine
//! (the same maximal-LCS alignment `similarity_ratio` uses), which can pick
//! a different, equally valid `M` than difflib's own anchored recursion on
//! repeated-character inputs (`tests/test_similarity.py`'s "validity-first,
//! not exact parity" contract applies identically here: the divergence is
//! expected, not a bug: see `tests/test_grounded.py`'s pinned
//! `"010"`/`"120"` regression). Be exact about what it measures; see
//! [`is_grounded_fuzzy`]'s docs for the limits.
//!
//! `fuzzy=False` is `source.contains(claim)`: `str::contains`'s own
//! substring search (a Two-Way-algorithm scan via `memchr`'s SIMD-accelerated
//! search under the hood), already the right primitive; no new dependency.
//!
//! `fuzzy=True` reuses the ONLY diffing engine already in the crate
//! (`similar`, backing `diff_opcodes`) rather than adding a fuzzy-string
//! crate: `claim` is compared against overlapping same-length windows of
//! `source` (stride `claim`'s length / 2), and the BEST window's difflib
//! ratio (`2 * matched_chars / (len(claim) + len(window))`) is the score.
//! Windowing, not a single whole-string diff of `claim` against all of
//! `source`, is the DoS-discipline choice: the realistic
//! RAG-grounding shape is a short claim against a long retrieved passage, so
//! bounding each diff's operand sizes to ~`claim`'s length keeps the total
//! work roughly linear in `source`'s length instead of the O(source *
//! claim) a single unwindowed diff would cost. `deadline_ms` bounds the
//! whole scan on top of that, the same discretionary escape hatch
//! `diff_opcodes` already has for adversarial input: checked once per
//! window (a window's own Myers search is itself deadline-bounded via
//! `similar`'s `capture_diff_slices_deadline`, so even one huge window
//! cannot blow through the budget uninterrupted). An exact-containment
//! floor runs first: when `source.contains(claim)` the answer is `true`
//! before any windowing or deadline setup, so a claim present verbatim is
//! always grounded regardless of window alignment or `deadline_ms` (a
//! verbatim substring is grounded by definition — the windowed ratio is
//! only consulted when there is no exact match to find).

use std::time::Instant;

use similar::{Algorithm, DiffOp, capture_diff_slices_deadline};

use crate::diff_impl::{budget_from_ms, elapsed_exceeds};

/// The whole fuzzy scan exceeded its caller-supplied budget; the (possibly
/// approximated, definitely incomplete) in-progress verdict is discarded:
/// never a silently degraded answer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeadlineExceeded {
    pub deadline_ms: f64,
    pub elapsed_ms: f64,
}

impl DeadlineExceeded {
    pub fn message(&self) -> String {
        format!(
            "is_grounded fuzzy scan deadline exceeded: elapsed {:.1}ms > deadline_ms {:.1}ms",
            self.elapsed_ms, self.deadline_ms
        )
    }
}

/// `source.contains(claim)` exactly: Rust's own substring search. An empty
/// `claim` is vacuously contained in anything (`"".contains("")` and
/// `s.contains("")` are both `true` in Rust already; no special-casing
/// needed).
pub fn is_grounded_exact(claim: &str, source: &str) -> bool {
    source.contains(claim)
}

/// The difflib `2*M/T` ratio (`M` = total matched-char length across every
/// `Equal` op, `T` = the combined length of both slices) between two char
/// slices, `similar`'s Myers engine under `deadline` (an absolute instant,
/// `None` = unbounded for THIS window: the caller still enforces the
/// overall budget between windows). `1.0` for two empty slices, the same
/// convention `difflib.SequenceMatcher([], []).ratio()` uses.
fn ratio(a: &[char], b: &[char], deadline: Option<Instant>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let matched: usize = capture_diff_slices_deadline(Algorithm::Myers, a, b, deadline)
        .into_iter()
        .map(|op| match op {
            DiffOp::Equal { len, .. } => len,
            _ => 0,
        })
        .sum();
    2.0 * matched as f64 / (a.len() + b.len()) as f64
}

/// Is `claim` fuzzily grounded in `source`: does the BEST-matching
/// same-length window of `source` reach `threshold` ratio against `claim`
/// (see the module docs for exactly what that ratio measures, and why the
/// scan is windowed rather than one whole-string diff)? `claim` empty is
/// vacuously `Ok(true)` (nothing to find, no scan needed). `deadline_ms`
/// (`None` = unbounded) bounds the WHOLE scan; on expiry the in-progress
/// verdict is discarded and [`DeadlineExceeded`] is returned: the caller
/// (the pyo3 layer) is guaranteed the same positive-finite precondition
/// `diff_opcodes` already validates before calling.
pub fn is_grounded_fuzzy(
    claim: &str,
    source: &str,
    threshold: f64,
    deadline_ms: Option<f64>,
) -> Result<bool, DeadlineExceeded> {
    if claim.is_empty() {
        return Ok(true);
    }
    // Exact-containment floor: a claim present VERBATIM in `source` is
    // grounded by definition — it is exactly what `fuzzy=False` reports — so
    // short-circuit before the windowed scan. Without this, a verbatim claim
    // at an offset unaligned with the stride-`L/2` windows overlaps its
    // nearest window by only ~3L/4, scores ~0.75 < the 0.85 default, and is
    // wrongly reported ungrounded. This runs BEFORE deadline setup, so a
    // verbatim substring is grounded even under a tight `deadline_ms` —
    // intentional: exact containment holds independent of the budget.
    if source.contains(claim) {
        return Ok(true);
    }
    let started = Instant::now();
    let deadline = deadline_ms.and_then(|ms| started.checked_add(budget_from_ms(ms)));
    let claim_chars: Vec<char> = claim.chars().collect();
    let source_chars: Vec<char> = source.chars().collect();

    let mut best = if source_chars.len() <= claim_chars.len() {
        // source no longer than claim: one direct comparison, no windowing
        // needed (there is nothing bigger to window over).
        ratio(&claim_chars, &source_chars, deadline)
    } else {
        let window = claim_chars.len();
        let stride = (window / 2).max(1);
        let mut best = 0.0f64;
        let mut start = 0usize;
        loop {
            let end = (start + window).min(source_chars.len());
            best = best.max(ratio(&claim_chars, &source_chars[start..end], deadline));
            if let Some(exceeded) = exceeded_after(started, deadline_ms) {
                return Err(exceeded);
            }
            if best >= threshold || end == source_chars.len() {
                break;
            }
            start += stride;
        }
        best
    };
    if let Some(exceeded) = exceeded_after(started, deadline_ms) {
        return Err(exceeded);
    }
    // Clamp: floating-point summation over pathological inputs stays within
    // [0, 1] by construction (matched <= min(a.len(), b.len()) <= either
    // length), but a defensive clamp costs nothing and keeps the caller-
    // facing invariant airtight regardless.
    best = best.clamp(0.0, 1.0);
    Ok(best >= threshold)
}

fn exceeded_after(started: Instant, deadline_ms: Option<f64>) -> Option<DeadlineExceeded> {
    elapsed_exceeds(started, deadline_ms).map(|(deadline_ms, elapsed_ms)| DeadlineExceeded {
        deadline_ms,
        elapsed_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_is_plain_substring_containment() {
        assert!(is_grounded_exact("cat", "the cat sat"));
        assert!(!is_grounded_exact("dog", "the cat sat"));
        assert!(is_grounded_exact("", "anything"));
        assert!(is_grounded_exact("", ""));
        assert!(!is_grounded_exact("x", ""));
    }

    #[test]
    fn fuzzy_reports_a_verbatim_substring_at_any_offset() {
        // Exact containment is a floor on the fuzzy verdict: a claim present
        // verbatim in source must be grounded at any offset, even where
        // stride-L/2 windowing would otherwise straddle it and score below
        // threshold.
        for claim in [
            "the quick brown fox jumps over lazy dog",
            "café über naïve résumé — 速い茶色の狐",
        ] {
            for lead in 0..41 {
                let source = format!("{}{}{}", "x".repeat(lead), claim, "x".repeat(30));
                assert_eq!(is_grounded_fuzzy(claim, &source, 0.85, None), Ok(true));
                assert_eq!(is_grounded_fuzzy(claim, &source, 1.0, None), Ok(true));
            }
        }
        // The floor runs before deadline setup, so a verbatim claim is
        // grounded even under an effectively-zero budget — never DeadlineExceeded.
        let big = format!("{}cat{}", "y".repeat(200_000), "y".repeat(200_000));
        assert_eq!(is_grounded_fuzzy("cat", &big, 1.0, Some(0.0001)), Ok(true));
    }

    #[test]
    fn fuzzy_exact_match_scores_one() {
        assert_eq!(
            is_grounded_fuzzy("the cat sat", "the cat sat", 1.0, None),
            Ok(true)
        );
    }

    #[test]
    fn fuzzy_empty_claim_is_vacuously_true() {
        assert_eq!(
            is_grounded_fuzzy("", "anything at all", 1.0, None),
            Ok(true)
        );
        assert_eq!(is_grounded_fuzzy("", "", 1.0, None), Ok(true));
    }

    #[test]
    fn fuzzy_finds_a_near_match_inside_a_much_longer_source() {
        let source = "Lorem ipsum dolor sit amet. The cats sit on mats today. Consectetur.";
        // "the cat sat" vs "The cats sit" (window-scanned): close but not
        // identical, so it should clear a moderate threshold, not a strict one.
        assert_eq!(
            is_grounded_fuzzy("the cat sat", source, 0.6, None),
            Ok(true)
        );
        assert_eq!(
            is_grounded_fuzzy("the cat sat", source, 0.999, None),
            Ok(false)
        );
    }

    #[test]
    fn fuzzy_unrelated_text_scores_low() {
        assert_eq!(
            is_grounded_fuzzy(
                "quantum entanglement",
                "a recipe for banana bread",
                0.5,
                None
            ),
            Ok(false)
        );
    }

    #[test]
    fn fuzzy_source_shorter_than_claim_still_compares_directly() {
        // No windowing possible; still a well-defined ratio.
        assert_eq!(
            is_grounded_fuzzy("hello world", "hello", 0.99, None),
            Ok(false)
        );
        assert_eq!(
            is_grounded_fuzzy("hello world", "hello", 0.5, None),
            Ok(true)
        );
    }

    #[test]
    fn fuzzy_threshold_zero_is_always_grounded() {
        assert_eq!(
            is_grounded_fuzzy("anything", "completely different", 0.0, None),
            Ok(true)
        );
    }

    #[test]
    fn fuzzy_deadline_exceeded_is_reported() {
        let claim = "x".repeat(2000);
        let source = "y".repeat(200_000);
        let err = is_grounded_fuzzy(&claim, &source, 1.0, Some(0.001))
            .expect_err("an effectively-zero budget must expire");
        assert_eq!(err.deadline_ms, 0.001);
        assert!(err.message().contains("deadline"));
    }

    #[test]
    fn fuzzy_generous_deadline_never_expires() {
        assert_eq!(
            is_grounded_fuzzy("the cat sat", "the cat sat on the mat", 0.5, Some(60_000.0)),
            Ok(true)
        );
    }
}
