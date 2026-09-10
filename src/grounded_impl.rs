//! Lexical claim-grounding check: the pure-Rust core of `tors.is_grounded`.
//!
//! Answers one narrow question: does `source` actually contain `claim`, or
//! something close enough to it? This is a lexical check, not semantic:
//! there is no NLI/entailment model here, and the fuzzy path's score is a
//! difflib-shaped character ratio (the `2*M/T` formula) against a bounded
//! scan of `source`, not a general string-similarity oracle and not a
//! bit-exact difflib differential: `M` comes from `similar`'s Myers engine
//! (the same maximal-LCS alignment `similarity_ratio` uses), which can pick
//! a different, equally valid `M` than difflib's own anchored recursion on
//! repeated-character inputs (`tests/test_similarity.py`'s "validity-first,
//! not exact parity" contract applies identically here: the divergence is
//! expected, not a bug: see `tests/test_grounded.py`'s pinned
//! `"010"`/`"120"` regression). Be exact about what it measures; see
//! [`is_grounded_fuzzy`]'s docs for the limits.
//!
//! `fuzzy=False` is exact substring containment: the `memchr` crate's
//! `memmem` search (two-way with SIMD-accelerated skipping, already a
//! dependency of this crate). std's `str::contains` runs the same two-way
//! algorithm without the SIMD skip and measured 7x slower than CPython's
//! `in` on a degenerate single-byte-run source (239us vs 33us over 256
//! KiB, the exact lane of tests/test_grounded_performance.py); a
//! byte-level find is UTF-8-boundary-safe (a valid-UTF-8 needle can only
//! match at a char boundary, UTF-8 being self-synchronizing).
//!
//! `fuzzy=True` reuses the only diffing engine already in the crate
//! (`similar`, backing `diff_opcodes`) rather than adding a fuzzy-string
//! crate: `claim` is compared against overlapping windows of `source`
//! (stride `claim`'s length / 2), and the best window's region score
//! against the claim-length denominator — the difflib ratio
//! `2 * matched_chars / (len(claim) + len(claim))` over equal-length
//! operands — is the score. A window shorter than the claim
//! (the source's truncated tail window) is scored as the claim-length
//! region it truncates: the chars the source fails to provide past its
//! end are mismatches, never the discounted `len(claim) + len(window)`
//! denominator, which would inflate a source that just ends partway
//! through the evidence above the same evidence sitting mid-source and
//! make the verdict depend on where the evidence sits (issue #40: a
//! 31-of-41-char tail scored 0.861 flush with the end, 0.756 in the
//! interior). The one deliberate exception is the no-windowing regime,
//! `source` shorter than `claim`: there is nothing to window over, so the
//! whole source is the evidence and the score is one direct difflib
//! `2*M/(m+n)` ratio — the convention the unwindowed difflib-parity
//! contract pins; at `len(source) == len(claim)` the two formulas agree,
//! so the boundary is continuous in score.
//! Windowing, not a single whole-string diff of `claim` against all of
//! `source`, is the DoS-discipline choice: the realistic
//! RAG-grounding shape is a short claim against a long retrieved passage, so
//! bounding each diff's operand sizes to ~`claim`'s length keeps the total
//! work roughly linear in `source`'s length instead of the O(source *
//! claim) a single unwindowed diff would cost. `deadline_ms` bounds the
//! whole scan on top of that, the same discretionary escape hatch
//! `diff_opcodes` already has for adversarial input: checked after every
//! window diff, coarse and refinement alike (a window's own Myers search
//! is itself deadline-bounded via `similar`'s `capture_diff_slices_deadline`,
//! so even one huge window cannot blow through the budget uninterrupted).
//! An exact-containment floor runs first: when `source` contains `claim`
//! answer is `true` before any windowing or deadline setup, so a claim
//! present verbatim is always grounded regardless of window alignment or
//! `deadline_ms` (a verbatim substring is grounded by definition: the
//! windowed ratio is only consulted when there is no exact match to find).
//!
//! A bounded refinement pass closes the window-straddle recall gap the floor
//! cannot (a near-match at an offset unaligned with the stride-`L/2` grid
//! overlaps its nearest window by as little as ~`3L/4`, scoring ~`r - 1/4`
//! for an aligned ratio `r`: one typo in a 41-char claim straddled to
//! ~0.73 and fell below the 0.85 default). The coarse scan tracks its
//! best-scoring windows (the top [`REFINE_CANDIDATES`] windows scoring at
//! least [`CANDIDATE_MIN_SCORE`], the truncated tail window competing like
//! any other at its region score: for a region flush with the source's
//! end, the tail window or the last full grid window between them always
//! clear the entry bound, and every candidate's fine range is clamped to
//! reach `n - L`, whose window `[n - L, n)` holds any sub-`L` suffix of
//! the source — so end-truncated evidence scores exactly like the same
//! evidence mid-source, the region-score invariance above); if the coarse
//! best falls short of `threshold`, each
//! candidate is re-scanned at a fine stride (`L / `[`REFINE_STRIDE_DIV`]`)
//! across its neighborhood. The guarantee, from the grid arithmetic and
//! validated by a brute-force differential geometry search (see
//! `fuzzy_the_guarantee_band_survives_the_tail_region_score`'s comment for
//! the post-#40 re-validation): a same-length source region with aligned
//! ratio `r` is detected at any offset whenever
//! `r >= max(0.75, threshold + 1/32)`: one substitution clears the 0.85
//! default for any claim of 9+ characters, wherever it sits. Below
//! `r = 0.75` detection is best-effort. The refinement's worst case is a
//! constant number of extra window diffs (candidates x fine windows,
//! independent of source length), so the scan's linear-in-`source` cost
//! shape is unchanged; adversarially flooding the candidate band can
//! still evict a genuine region from the top scores, the regime
//! `deadline_ms` exists for.
//!
//! Memory is O(claim), not O(source): windows slide through one reusable
//! exact-capacity char buffer instead of materializing the whole source
//! (a 12 MiB passage costs kilobytes, and an early exit stops consuming
//! input mid-source instead of paying the full collect first).

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::time::Instant;

use memchr::memmem;
use similar::{Algorithm, DiffOp, capture_diff_slices_deadline};

use crate::diff_impl::{budget_from_ms, elapsed_exceeds};

/// Refinement candidates: the highest-scoring coarse windows that the
/// refinement pass may re-scan. Constant, so the refinement's worst-case
/// add-on is bounded independently of source length: at most
/// `REFINE_CANDIDATES` windows' worth of fine-stride diffs beyond the
/// coarse scan.
const REFINE_CANDIDATES: usize = 64;

/// The minimum coarse score for a full window to enter the candidate set.
/// Entry bound: a same-length region with aligned ratio `r` has its
/// nearest full grid window scoring at least `r - 1/4` (the grid's spacing
/// is `L/2`, so the nearest window misses at most `L/4` of the region), so
/// every region with `r >= 0.75` is certain to enter: independent of
/// `threshold`, which keeps the verdict monotonic in `threshold`.
/// Random/unrelated text scores far below this, so ordinary ungrounded
/// scans collect no candidates and pay nothing.
const CANDIDATE_MIN_SCORE: f64 = 0.5;

/// The refinement stride divisor: fine windows step `max(1, L / 16)` chars,
/// so a region's aligned window is never more than `1/32` of `L` away from
/// a scanned start: the `threshold + 1/32` term of the guarantee.
const REFINE_STRIDE_DIV: usize = 16;

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

/// `source` contains `claim` exactly: `memchr::memmem` (see the module
/// docs for why not std's `str::contains`). An empty `claim` is vacuously
/// contained in anything (`memmem` finds an empty needle at 0, matching
/// `"".contains("")` and `s.contains("")`).
pub fn is_grounded_exact(claim: &str, source: &str) -> bool {
    memmem::find(source.as_bytes(), claim.as_bytes()).is_some()
}

/// `M`: the total matched-char length across every `Equal` op of a diff of
/// two char slices, `similar`'s Myers engine under `deadline` (an absolute
/// instant, `None` = unbounded for this window: the caller still enforces
/// the overall budget between windows). The minimal-edit-script consequence
/// of the Myers search is `M == LCS(a, b)` exactly, the invariant the
/// Python oracle's independent DP checks.
fn matched_len(a: &[char], b: &[char], deadline: Option<Instant>) -> usize {
    capture_diff_slices_deadline(Algorithm::Myers, a, b, deadline)
        .into_iter()
        .map(|op| match op {
            DiffOp::Equal { len, .. } => len,
            _ => 0,
        })
        .sum()
}

/// The difflib `2*M/T` ratio (`T` = the combined length of both slices)
/// between two char slices: the score of the one direct comparison the
/// `n < L` source-shorter-than-claim path makes, where there is nothing to
/// window over and the whole source is the evidence (the convention the
/// unwindowed difflib-parity contract pins). `1.0` for two empty slices,
/// the same convention `difflib.SequenceMatcher([], []).ratio()` uses.
fn ratio(a: &[char], b: &[char], deadline: Option<Instant>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    2.0 * matched_len(a, b, deadline) as f64 / (a.len() + b.len()) as f64
}

/// The windowed scan's score: one window of `source` against the
/// claim-length denominator `2*L` — `2*M / (L + L)`, identical to [`ratio`]
/// for a full window (whose length is `L`) and deliberately NOT the
/// `2*M / (L + len(window))` a truncated tail window would get from
/// [`ratio`]. The tail window is scored as the claim-length region it
/// truncates: the chars the source fails to provide past its end are
/// mismatches, never a discounted denominator, which would inflate a
/// source that just ends partway through the evidence above the same
/// evidence sitting mid-source (issue #40: a 31-of-41-char tail scored
/// 0.861 flush with the end and 0.756 in the interior, flipping the
/// verdict at the 0.85 default on position alone). Position-invariance is
/// the point: the verdict depends on the evidence, not on where the source
/// stops.
fn region_ratio(claim: &[char], window: &[char], deadline: Option<Instant>) -> f64 {
    matched_len(claim, window, deadline) as f64 / claim.len() as f64
}

/// A refinement candidate: one coarse window worth re-scanning at fine
/// stride: its score (for priority and the top-K cut), its char start
/// (for the fine-range arithmetic) and byte start (the reposition anchor).
/// The truncated tail window competes like any other (score-gated, top-K),
/// with its range clamped to the last possible region start `n - L`.
struct Candidate {
    score: f64,
    start: usize,
    byte: usize,
}

/// Insert into the bounded candidate set: the top `REFINE_CANDIDATES`
/// entries by `(score, start)`: the streaming keep-K-largest discipline
/// (evict the minimum only when the newcomer beats it), which maintains
/// exactly the top-K of everything seen under the strict total order
/// `(score, start)` (starts are unique, so the order is strict with no
/// ties to resolve by fiat).
fn keep_candidate(candidates: &mut Vec<Candidate>, new: Candidate) {
    let cap = REFINE_CANDIDATES;
    if candidates.len() < cap {
        candidates.push(new);
        return;
    }
    let (min_idx, min) = candidates
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(Ordering::Equal)
                .then(a.start.cmp(&b.start))
        })
        .expect("the candidate set is at capacity, so non-empty");
    let keeps_seat = new.score > min.score || (new.score == min.score && new.start > min.start);
    if keeps_seat {
        candidates[min_idx] = new;
    }
}

/// A sliding same-length window stream over `source`'s characters with
/// O(window) memory: one reusable char buffer and a byte-offset deque, both
/// exact-capacity and allocated once, sliding `stride` chars per window via
/// a front drain (the drain's memmove is ~`L - stride` chars: tens of
/// nanoseconds against the microsecond Myers diff the window feeds). The
/// byte offsets exist so each window's start can be recovered as an anchor
/// (the refinement pass repositions on a candidate's byte start instead of
/// re-walking the source from its beginning).
struct CharWindows<'a> {
    src: &'a str,
    byte: usize,
    target: usize,
    window: Vec<char>,
    offsets: VecDeque<usize>,
    first: bool,
    done: bool,
}

impl<'a> CharWindows<'a> {
    fn new(src: &'a str, window_len: usize) -> Self {
        Self {
            src,
            byte: 0,
            target: window_len,
            window: Vec::with_capacity(window_len),
            offsets: VecDeque::with_capacity(window_len),
            first: true,
            done: false,
        }
    }

    /// Reset to a byte anchor (a char boundary): the next window starts
    /// there. The buffers are reused, not reallocated.
    fn reposition(&mut self, byte: usize) {
        self.byte = byte;
        self.window.clear();
        self.offsets.clear();
        self.first = true;
        self.done = false;
    }

    /// The next window: `(chars, byte offset of its first char, truncated)`.
    /// The first window after construction/reposition starts at the cursor;
    /// each later one advances the start by `stride` (the caller passes the
    /// same advance it used for its own start arithmetic, so a capped grid
    /// walk cannot desync the window content from the caller's offsets).
    /// `None` once the source is exhausted (a truncated window is always
    /// the last).
    fn next(&mut self, stride: usize) -> Option<(&[char], usize, bool)> {
        if self.done {
            return None;
        }
        if !self.first {
            if self.window.len() <= stride {
                self.window.clear();
                self.offsets.clear();
            } else {
                self.window.drain(0..stride);
                for _ in 0..stride {
                    self.offsets.pop_front();
                }
            }
        }
        self.first = false;
        let mut consumed_to = self.byte;
        for (rel, ch) in self.src[self.byte..].char_indices() {
            if self.window.len() == self.target {
                break;
            }
            let abs = self.byte + rel;
            self.window.push(ch);
            self.offsets.push_back(abs);
            consumed_to = abs + ch.len_utf8();
        }
        self.byte = consumed_to;
        let truncated = self.window.len() < self.target;
        if truncated {
            self.done = true;
        }
        let start = *self.offsets.front()?;
        Some((self.window.as_slice(), start, truncated))
    }

    /// Whether the cursor sits at the source's end: the most recent full
    /// window was the last window there is. (A truncated window sets
    /// `done` itself.)
    fn at_end(&self) -> bool {
        self.byte >= self.src.len()
    }
}

/// The byte offset of the char boundary `chars_back` positions before `i`
/// (itself a char boundary): step back one char at a time. Bounded by the
/// refinement neighborhood (<= `L` chars), so O(L) byte checks.
fn back_char_boundary(source: &str, mut i: usize, mut chars_back: usize) -> usize {
    while chars_back > 0 {
        i -= 1;
        while !source.is_char_boundary(i) {
            i -= 1;
        }
        chars_back -= 1;
    }
    i
}

/// Is `claim` fuzzily grounded in `source`: does the best-matching window
/// of `source` reach `threshold`'s region score against `claim` (see the
/// module docs for exactly what that score measures — the claim-length
/// denominator, the truncated tail window included, and the
/// source-shorter-than-claim direct-comparison exception — why the
/// scan is windowed rather than one whole-string diff, and the bounded
/// refinement pass that makes a near-match's verdict independent of where
/// it sits)? `claim` empty is vacuously `Ok(true)` (nothing to find, no
/// scan needed). `deadline_ms` (`None` = unbounded) bounds the whole scan:
/// checked after every window diff, coarse and refinement alike; on
/// expiry the in-progress verdict is discarded and [`DeadlineExceeded`] is
/// returned: the caller (the pyo3 layer) is guaranteed the same
/// positive-finite precondition `diff_opcodes` already validates before
/// calling.
pub fn is_grounded_fuzzy(
    claim: &str,
    source: &str,
    threshold: f64,
    deadline_ms: Option<f64>,
) -> Result<bool, DeadlineExceeded> {
    if claim.is_empty() {
        return Ok(true);
    }
    // Exact-containment floor: a claim present verbatim in `source` is
    // grounded by definition (it is exactly what `fuzzy=False` reports) so
    // short-circuit before the windowed scan. Without this, a verbatim claim
    // at an offset unaligned with the stride-`L/2` windows overlaps its
    // nearest window by only ~3L/4, scores ~0.75 < the 0.85 default, and is
    // wrongly reported ungrounded. This runs before deadline setup, so a
    // verbatim substring is grounded even under a tight `deadline_ms`:
    // intentional: exact containment holds independent of the budget.
    if is_grounded_exact(claim, source) {
        return Ok(true);
    }
    let started = Instant::now();
    let deadline = deadline_ms.and_then(|ms| started.checked_add(budget_from_ms(ms)));
    let claim_len = claim.chars().count();
    // Exact-capacity, one allocation; no incremental growth. The window
    // buffer below is the only other allocation on any path.
    let mut claim_chars = Vec::with_capacity(claim_len);
    claim_chars.extend(claim.chars());

    // The first window fill doubles as the source-length probe, so no
    // upfront O(source) char count and no second collection on any path:
    // a truncated first fill means the source holds fewer than `L` chars
    // and the window buffer already is the whole source: the
    // no-windowing direct comparison (`ratio`, the whole-source difflib
    // convention). A source of exactly `L` chars fills
    // the window full and takes the windowed path; its single window
    // scores `M/L`, the same value the direct comparison would produce
    // (`2*M/(L+L)`), so the verdict is the same either way and the
    // boundary between the two conventions is continuous in score.
    // An empty source yields no window at all: nothing to match, `best`
    // stays 0.0 (still `true` at threshold 0.0, the empty-anything
    // convention).
    let l = claim_len;
    let stride = (l / 2).max(1);
    let mut walker = CharWindows::new(source, l);
    let mut candidates: Vec<Candidate> = Vec::with_capacity(REFINE_CANDIDATES);
    let mut best = 0.0f64;
    match walker.next(stride) {
        None => {}
        Some((first_window, _first_byte, true)) => {
            // Direct comparison: the buffer holds the whole source.
            best = ratio(&claim_chars, first_window, deadline);
        }
        Some((first_window, first_byte, false)) => {
            let score = ratio(&claim_chars, first_window, deadline);
            best = best.max(score);
            if let Some(exceeded) = exceeded_after(started, deadline_ms) {
                return Err(exceeded);
            }
            if score >= CANDIDATE_MIN_SCORE {
                keep_candidate(
                    &mut candidates,
                    Candidate {
                        score,
                        start: 0,
                        byte: first_byte,
                    },
                );
            }
            let mut start = 0usize;
            while best < threshold && !walker.at_end() {
                start += stride;
                let Some((window, byte_start, truncated)) = walker.next(stride) else {
                    break;
                };
                // Full windows and the truncated tail alike: the region
                // score (the tail's missing chars are mismatches, never a
                // discounted denominator). The tail competes as an
                // ordinary candidate (its range clamps to `n - L` in the
                // refinement below).
                let score = region_ratio(&claim_chars, window, deadline);
                best = best.max(score);
                if let Some(exceeded) = exceeded_after(started, deadline_ms) {
                    return Err(exceeded);
                }
                if score >= CANDIDATE_MIN_SCORE {
                    // Full windows and the truncated final window alike:
                    // the tail competes as an ordinary candidate (its
                    // range clamps to `n - L` in the refinement below).
                    keep_candidate(
                        &mut candidates,
                        Candidate {
                            score,
                            start,
                            byte: byte_start,
                        },
                    );
                }
                if best >= threshold || truncated {
                    break;
                }
            }
            if best < threshold {
                Refinement {
                    claim_chars: &claim_chars,
                    source,
                    threshold,
                    started,
                    deadline_ms,
                    deadline,
                }
                .run(&mut walker, candidates, &mut best)?;
            }
        }
    }
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

/// The bounded refinement pass: re-scan each candidate's neighborhood at
/// fine stride `max(1, L / REFINE_STRIDE_DIV)`: starts across
/// `[w - L/2, min(w + L/2, n - L)]` (a truncated tail candidate's range
/// clamps to `n - L`, the last possible region start). Every fine window
/// is full-length and deadline-checked like a coarse one, and each range's
/// upper end is always scanned (the grid alone could leave a `fine - 1`
/// gap at `hi`, doubling the misalignment margin). What this buys, from
/// the grid arithmetic: a same-length region with aligned ratio `r` has
/// its nearest full grid window scoring at least `r - 1/4` (candidate
/// entry for `r >= 0.75`) and its aligned window within `1/32 * L` of a
/// scanned fine start, so `r >= max(0.75, threshold + 1/32)` is detected
/// at any offset. The guarantee band is where the verdict is
/// alignment-invariant; below `r = 0.75` detection is best-effort (the
/// coarse pass or a candidate's neighborhood may still find it).
struct Refinement<'a> {
    claim_chars: &'a [char],
    source: &'a str,
    threshold: f64,
    started: Instant,
    deadline_ms: Option<f64>,
    deadline: Option<Instant>,
}

impl Refinement<'_> {
    fn run(
        &self,
        walker: &mut CharWindows<'_>,
        candidates: Vec<Candidate>,
        best: &mut f64,
    ) -> Result<(), DeadlineExceeded> {
        let l = self.claim_chars.len();
        // The source's char count, needed only for the range clamps. Paid
        // here rather than up front: the refinement runs at most once, and
        // only after a coarse scan that already traversed the whole
        // source, so this pass is noise on every path that reaches it:
        // and the early-exit paths never pay it at all.
        let n = self.source.chars().count();
        let mut all = candidates;
        // Best-first: highest coarse scores are the most likely regions, so
        // the early break trips soonest. (The verdict is order-independent;
        // best only rises, so this is a speed choice, not a semantic one.)
        all.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then(a.start.cmp(&b.start))
        });
        let fine = (l / REFINE_STRIDE_DIV).max(1);
        for cand in &all {
            if *best >= self.threshold {
                break;
            }
            let lo = cand.start.saturating_sub(l / 2);
            let hi = (cand.start + l / 2).min(n - l);
            if lo > hi {
                continue;
            }
            let anchor = back_char_boundary(self.source, cand.byte, cand.start - lo);
            walker.reposition(anchor);
            // Fine starts: the `fine` grid over [lo, hi], with `hi` itself
            // always scanned: the advance passed to the walker is the
            // caller's own step (capped at hi), so window content and
            // start arithmetic cannot desync.
            let mut fstart = lo;
            let mut advance = fine; // ignored until the second window
            while let Some((window, _, _)) = walker.next(advance) {
                *best = best.max(region_ratio(self.claim_chars, window, self.deadline));
                if let Some(exceeded) = exceeded_after(self.started, self.deadline_ms) {
                    return Err(exceeded);
                }
                if *best >= self.threshold || fstart >= hi {
                    break;
                }
                let next = (fstart + fine).min(hi);
                advance = next - fstart;
                fstart = next;
            }
        }
        Ok(())
    }
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
        // grounded even under an effectively-zero budget: never DeadlineExceeded.
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
    fn fuzzy_empty_source_is_ungrounded_except_at_threshold_zero() {
        // The empty-source convention (the walker yields no window, `best`
        // stays 0.0: the same value the direct comparison produced): a
        // nonempty claim over "" is ungrounded at any positive threshold,
        // and vacuously grounded at exactly 0.0, where everything is.
        assert_eq!(is_grounded_fuzzy("x", "", 0.85, None), Ok(false));
        assert_eq!(is_grounded_fuzzy("claim", "", 0.5, None), Ok(false));
        assert_eq!(is_grounded_fuzzy("x", "", 0.0, None), Ok(true));
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
        // A near-match, deliberately not a verbatim substring, so the scan
        // still traverses the windowed path under the budget (the
        // exact-containment floor would return before deadline setup and
        // the budget would never be exercised: the Python twin's own
        // comment explains the same trap).
        assert_eq!(
            is_grounded_fuzzy("the cet sat", "the cat sat on the mat", 0.5, Some(60_000.0)),
            Ok(true)
        );
    }

    #[test]
    fn fuzzy_the_refinement_pass_respects_a_generous_deadline() {
        // The straddled one-typo geometry (lead 10: coarse best ~0.73 <
        // 0.85, only the refinement finds it) under a generous budget: the
        // refinement's own per-window deadline checks must all pass and the
        // verdict arrive: the deadline plumbing on the refinement path,
        // which the zero-budget tests never reach (they expire at the
        // coarse stage). 60s against microsecond-scale work is
        // machine-speed-immune.
        let claim = "the bushing torque specifications changed";
        let near = "the bushing torqxe specifications changed";
        let source = format!("{}{}{}", "q".repeat(10), near, "q".repeat(60));
        assert_eq!(
            is_grounded_fuzzy(claim, &source, 0.85, Some(60_000.0)),
            Ok(true)
        );
    }

    #[test]
    fn fuzzy_a_near_exact_non_substring_clears_the_default_but_never_one() {
        // One substitution, so the exact-containment floor cannot
        // short-circuit and the windowed scorer must produce the verdict:
        // near-exact content clears the default, and only a verbatim match
        // may clear 1.0 (restores the scorer coverage the floor removed
        // from `fuzzy_exact_match_scores_one`, whose identical operands now
        // return at the floor before any window is diffed).
        assert_eq!(
            is_grounded_fuzzy("the cet sat", "the cat sat on the mat", 0.85, None),
            Ok(true)
        );
        assert_eq!(
            is_grounded_fuzzy("the cet sat", "the cat sat on the mat", 1.0, None),
            Ok(false)
        );
    }

    #[test]
    fn fuzzy_one_typo_near_match_at_any_offset_clears_the_default() {
        // The refinement pass's headline guarantee: a same-length source
        // region matching the claim with ratio r is detected at any offset
        // whenever r >= max(0.75, threshold + 1/32): one substitution at
        // L=41 gives r = 40/41 = 0.976, comfortably above 0.85 + 1/32, so
        // every offset must clear the default threshold. Before refinement,
        // stride-L/2 windowing straddled the region at unaligned offsets
        // (nearest window overlapping only ~3L/4, score ~0.73) and wrongly
        // rejected it: the same defect class the exact-containment floor
        // fixed for the verbatim case, here fixed for near matches.
        let claim = "the bushing torque specifications changed"; // 41 chars
        let near = "the bushing torqxe specifications changed"; // one substitution
        assert!(!near.contains(claim));
        for lead in 0..=40 {
            let source = format!("{}{}{}", "q".repeat(lead), near, "q".repeat(60));
            assert_eq!(
                is_grounded_fuzzy(claim, &source, 0.85, None),
                Ok(true),
                "one-typo near match rejected at lead {lead}"
            );
        }
    }

    #[test]
    fn fuzzy_three_typos_at_the_worst_alignment_still_clear_the_default() {
        // r = 38/41 = 0.927 >= 0.85 + 1/32: still inside the guaranteed
        // band, even at the worst stride alignment.
        let claim = "the bushing torque specifications changed";
        let near: String = claim
            .chars()
            .enumerate()
            .map(|(i, c)| if matches!(i, 5 | 18 | 33) { 'Z' } else { c })
            .collect();
        let source = format!("{}{}{}", "q".repeat(10), near, "q".repeat(60));
        assert_eq!(is_grounded_fuzzy(claim, &source, 0.85, None), Ok(true));
    }

    #[test]
    fn fuzzy_the_detection_margin_is_pinned() {
        // The guarantee is sufficient, not necessary: r >= 0.85 + 1/32 is
        // the worst-case fine-grid misalignment bound, and at L=41 the fine
        // stride is 2, so a region whose start lands on the grid is scored
        // exactly aligned. Pinned at the measured edge: six substitutions
        // (r = 35/41 = 0.854, below the worst-case line) still clears 0.85
        // here; seven (r = 34/41 = 0.829, below the threshold outright)
        // does not, and clears 0.8 where the guarantee covers it. Positions
        // k * (41 / d) for k in 0..d: the same rows the Python twin and
        // the differential oracle pin.
        let claim = "the bushing torque specifications changed";
        for (d, at_default, at_08) in [(6, true, true), (7, false, true)] {
            let mut chars: Vec<char> = claim.chars().collect();
            let step = 41 / d;
            for k in 0..d {
                chars[k * step] = 'Z';
            }
            let near: String = chars.into_iter().collect();
            let source = format!("{}{}{}", "q".repeat(10), near, "q".repeat(60));
            assert_eq!(
                is_grounded_fuzzy(claim, &source, 0.85, None),
                Ok(at_default),
                "d={d} at the default"
            );
            assert_eq!(
                is_grounded_fuzzy(claim, &source, 0.8, None),
                Ok(at_08),
                "d={d} at 0.8"
            );
        }
    }

    #[test]
    fn fuzzy_near_tail_regions_are_detected_through_the_tail_window_candidate() {
        // The two tail geometries, pinned behaviorally with the oracle's
        // agreement: a region hugging the source's end either clears via
        // the truncated tail window's own region score (the flush case:
        // the tail holds 40 of the region's chars, M = 35, 35/41 = 0.854
        // >= the default) or enters the candidate set with it (the
        // adjacent case: the tail holds 40 chars with M = 34, 0.829 <
        // 0.85, while the nearest full grid window [60, 101) scores only
        // 17/41 = 0.41, below the 0.5 entry bound, so the tail candidate
        // is the region's only lane into the refinement) and is found by
        // its clamped [w - L/2, n - L] range, whose last full window
        // [79, 120) scores the region's aligned r = 35/41 = 0.854. A
        // brute-force differential search over geometries (5 claim lengths
        // x 4 thresholds) found no guarantee-band region needing a
        // dedicated forced/extended tail mechanism: one existed here
        // briefly and was removed as unpinnable dead weight; these rows
        // pin the guarantee where the tail actually delivers it.
        let claim = "the bushing torque specifications changed";
        // Six typos packed into the head (r = 35/41 = 0.854), region
        // starting one char before the truncated tail window: n = 120,
        // grid 0/20/40/60 full, tail [80, 120), region [79, 120).
        let packed: String = claim
            .chars()
            .enumerate()
            .map(|(i, c)| {
                if matches!(i, 3 | 7 | 11 | 15 | 19 | 23) {
                    'Z'
                } else {
                    c
                }
            })
            .collect();
        let tail_adjacent = format!("{}{}", "q".repeat(79), packed);
        assert_eq!(tail_adjacent.chars().count(), 120);
        assert!(!tail_adjacent.contains(claim));
        assert_eq!(
            is_grounded_fuzzy(claim, &tail_adjacent, 0.85, None),
            Ok(true)
        );
        // The spread-typo variant further from the tail (16 chars past
        // 79 the last full grid window): found via the ordinary full-window
        // candidate's range.
        let spread: String = claim
            .chars()
            .enumerate()
            .map(|(i, c)| {
                if matches!(i, 3 | 11 | 19 | 27 | 35) {
                    'Z'
                } else {
                    c
                }
            })
            .collect();
        let via_band = format!("{}{}{}", "q".repeat(76), spread, "q".repeat(3));
        assert_eq!(is_grounded_fuzzy(claim, &via_band, 0.85, None), Ok(true));
    }

    #[test]
    fn fuzzy_partial_tail_evidence_scores_position_invariant() {
        // Issue #40's repro, pinned: the same 31-of-41 chars of evidence
        // must produce the same verdict wherever it sits. The pre-fix
        // truncated tail window was scored with the difflib
        // `2*M/(len(claim) + len(window))` denominator, so a source that
        // just ended partway through the evidence scored
        // `2*31/(41+31)` = 0.861 (clearing the 0.85 default) while the
        // identical evidence mid-source scored `2*31/(41+41)` = 0.756 and
        // was rejected. The tail window is now scored as the claim-length
        // region it truncates (missing chars are mismatches: `M/L`), so
        // every placement scores 0.756: ungrounded at the default, and a
        // 37-of-41 tail (0.902) grounded at every placement alike: the
        // legit tail match keeps its verdict, only the position-dependent
        // inflation loses its.
        let claim = "the bushing torque specifications changed"; // 41 chars
        assert_eq!(claim.chars().count(), 41);
        for k in [31usize, 37] {
            let frag: String = claim.chars().take(k).collect();
            let at_end = format!("{}{}", "q".repeat(60), frag);
            let mid = format!("{}{}{}", "q".repeat(60), frag, "q".repeat(60));
            let at_start = format!("{}{}", frag, "q".repeat(60));
            let expected = Ok(k as f64 / 41.0 >= 0.85);
            for source in [&at_end, &mid, &at_start] {
                assert!(!source.contains(claim)); // evidence, not containment
                assert_eq!(
                    is_grounded_fuzzy(claim, source, 0.85, None),
                    expected,
                    "k={k}"
                );
                // threshold=1.0 was never affected: nothing but verbatim
                // containment scores 1.0 (the tail's M <= k < 41).
                assert_eq!(is_grounded_fuzzy(claim, source, 1.0, None), Ok(false));
            }
        }
    }

    #[test]
    fn fuzzy_the_tail_truncation_band_is_pinned() {
        // The trigger band swept: evidence of k of the claim's 41 chars
        // (k = 29..40) flush with the source's end (lead 60 lands the tail
        // window exactly on the evidence for every k <= 40) vs the same
        // evidence mid-source. Post-fix both placements score exactly
        // `k/41`, so the verdict is `k/41 >= 0.85` at every position:
        // k = 31..34 (0.756..0.829) flip from grounded-at-the-end to
        // ungrounded everywhere (pre-fix `2*34/(41+34)` = 0.907), k >= 35
        // (0.854+) stays grounded everywhere.
        let claim = "the bushing torque specifications changed";
        for k in 29..=40usize {
            let frag: String = claim.chars().take(k).collect();
            let at_end = format!("{}{}", "q".repeat(60), frag);
            let mid = format!("{}{}{}", "q".repeat(60), frag, "q".repeat(60));
            let expected = Ok(k as f64 / 41.0 >= 0.85);
            assert_eq!(
                is_grounded_fuzzy(claim, &at_end, 0.85, None),
                expected,
                "k={k} at the end"
            );
            assert_eq!(
                is_grounded_fuzzy(claim, &mid, 0.85, None),
                expected,
                "k={k} mid-source"
            );
        }
    }

    #[test]
    fn fuzzy_a_source_shorter_than_the_claim_keeps_the_direct_difflib_ratio() {
        // The boundary judgement call, pinned on both sides: n < L is the
        // whole-source question, answered by one direct difflib
        // `2*M/(m+n)` ratio (the unwindowed difflib-parity contract pins
        // exact parity with difflib there, so its shorter denominator is
        // the pinned convention, not the tail bug); n >= L is the
        // containment question, where every window scores against the full
        // `2*L` denominator. The same 31-char evidence: alone as the whole
        // source it grounds at 0.85 (`2*31/(41+31)` = 0.861), padded into
        // the windowed regime it does not (31/41 = 0.756).
        let claim = "the bushing torque specifications changed";
        let short = &claim[..31]; // n = 31 < 41: the direct comparison
        let padded = format!("{}{}", "q".repeat(60), short); // n = 91: windowed
        assert_eq!(is_grounded_fuzzy(claim, short, 0.85, None), Ok(true));
        assert_eq!(is_grounded_fuzzy(claim, short, 0.87, None), Ok(false));
        assert_eq!(is_grounded_fuzzy(claim, &padded, 0.85, None), Ok(false));
    }

    #[test]
    fn fuzzy_the_guarantee_band_survives_the_tail_region_score() {
        // Post-#40 re-validation of the refinement guarantee under the tail
        // window's region score: the deflation could in principle have
        // starved end-flush regions of their candidate entry, but the entry
        // arithmetic holds -- the tail window and the last full grid window
        // between them always clear the 0.5 entry bound for r >= 0.75 (the
        // two windows' region overlaps tile 3L/2 of the region, so at least
        // one holds >= (r - 1/4) * L matched chars), and every candidate's
        // fine range is clamped to reach n - L, whose window [n - L, n)
        // contains any sub-L suffix of the source: end-truncated evidence
        // scores exactly like the same evidence mid-source. Brute-force
        // sweep, the shape of the pre-#40 6k-geometry search: 5 claim
        // lengths x 4 thresholds x two typo densities x every lead 0..=40,
        // three geometries each -- the region interior, flush with the
        // source's end, and truncated by it (k = L - 1 of the region's
        // chars as the whole evidence, with its interior twin agreeing:
        // the position-invariance the fix bought). Every row must ground.
        for l in [12usize, 17, 25, 33, 41] {
            // A claim over an alphabet disjoint from the 'q' pad and 'Z'
            // typo chars, so matched chars are exactly the region's.
            let claim: String = (0..l).map(|i| char::from(b'a' + (i % 13) as u8)).collect();
            for threshold in [0.85f64, 0.8, 0.75, 0.7] {
                for d in [1usize, l / 4] {
                    let r = (l - d) as f64 / l as f64;
                    if r < (0.75f64).max(threshold + 1.0 / 32.0) {
                        continue; // outside the guarantee band: best-effort
                    }
                    let near: String = claim
                        .chars()
                        .enumerate()
                        .map(|(i, c)| if d > 0 && i % (l / d) == 0 { 'Z' } else { c })
                        .collect();
                    assert!(!near.contains(claim.as_str()));
                    for lead in 0..=40 {
                        let interior = format!("{}{}{}", "q".repeat(lead), near, "q".repeat(50));
                        let flush = format!("{}{}", "q".repeat(lead), near);
                        for (what, source) in [("interior", &interior), ("flush", &flush)] {
                            assert_eq!(
                                is_grounded_fuzzy(&claim, source, threshold, None),
                                Ok(true),
                                "l={l} t={threshold} d={d} lead={lead} {what}"
                            );
                        }
                    }
                    // The truncated-tail geometry: the region's first L-1
                    // chars are the whole evidence, flush with the end (n >
                    // L required: the direct-comparison regime below n is a
                    // different, pinned convention). Its interior twin must
                    // agree: same evidence, same verdict.
                    let k = l - 1;
                    let partial: String = near.chars().take(k).collect();
                    for lead in 1..=40 {
                        let at_end = format!("{}{}", "q".repeat(lead), partial);
                        let mid = format!("{}{}{}", "q".repeat(lead), partial, "q".repeat(50));
                        let got_end = is_grounded_fuzzy(&claim, &at_end, threshold, None);
                        let got_mid = is_grounded_fuzzy(&claim, &mid, threshold, None);
                        assert_eq!(got_end, got_mid, "l={l} t={threshold} d={d} lead={lead}");
                        // In the band only when the truncated region's own
                        // ratio clears it; the r filter above used the full
                        // region, so re-check the partial's (L-1-d)/(L)
                        // against the band before demanding Ok(true).
                        if ((l - 1 - d) as f64 / l as f64) >= (0.75f64).max(threshold + 1.0 / 32.0)
                        {
                            assert_eq!(
                                got_end,
                                Ok(true),
                                "l={l} t={threshold} d={d} lead={lead} partial"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fuzzy_candidate_eviction_at_the_64th_band_window_is_the_pinned_flood_limit() {
        // The 64-candidate cap is the documented adversarial limit: the
        // top (score, start) band windows are all the refinement will ever
        // see, so a real near-match whose straddled coarse score (~0.73
        // here) ranks below 64 decoys scoring above it (~0.829, seven-typo
        // variants placed at grid-aligned offsets) is evicted and the
        // verdict is false despite r = 40/41 >= the guarantee: the regime
        // `deadline_ms` exists for. The boundary is exact: 63 decoys (64
        // band windows counting the real one's) still find it.
        let claim = "the bushing torque specifications changed";
        let real = "the bushing torqxe specifications changed"; // 1 typo
        // The measured decoy: seven substitutions at 2/8/14/20/26/32/38,
        // r = 34/41 = 0.829: in the candidate band, above the real
        // region's straddled ~0.73, below the 0.85 coarse break.
        let decoy: String = claim
            .chars()
            .enumerate()
            .map(|(i, c)| {
                if matches!(i, 2 | 8 | 14 | 20 | 26 | 32 | 38) {
                    'Z'
                } else {
                    c
                }
            })
            .collect();
        let build = |n: usize| {
            let mut parts = vec![format!("{}{}", "q".repeat(10), real)];
            let mut at = 60usize; // every decoy starts at a multiple of 20
            for _ in 0..n {
                let have: usize = parts.iter().map(|p| p.chars().count()).sum();
                parts.push(format!("{}{}", "q".repeat(at - have), decoy));
                at += 60;
            }
            parts.push("q".repeat(60));
            parts.join("")
        };
        assert_eq!(is_grounded_fuzzy(claim, &build(63), 0.85, None), Ok(true));
        assert_eq!(is_grounded_fuzzy(claim, &build(64), 0.85, None), Ok(false));
    }
}
