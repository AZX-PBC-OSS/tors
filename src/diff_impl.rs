//! Character-level opcode diffs: the pure-Rust core of `tors.diff_opcodes`.
//!
//! The output is `difflib`'s opcode shape exactly: `(tag, i1, i2, j1, j2)`
//! tuples, `tag` in {equal, replace, delete, insert}, ranges monotone,
//! contiguous and covering both sides, adjacent delete+insert merged into
//! replace the way `difflib.get_opcodes` presents it. Indices are Python
//! `str` indices (codepoints): the diff runs over `char`s; a Rust `char` IS
//! a Unicode scalar value, exactly what a Python `str` index addresses, so
//! `a[i1:i2]` is the slice an op describes. Why character level:
//! `difflib.SequenceMatcher(None, a, b)` on `str` operands diffs characters,
//! and it is the parity oracle: callers wanting line-level diffs split the
//! operands themselves.
//!
//! The engine is `similar`'s Myers (Apache-2.0), whose 3.x default entrypoint
//! is a Git-style bounded middle-snake search with a preflight pass that
//! removes unmatched and confusing records before the search. Two documented
//! consequences, both pinned on the Python side
//! (tests/test_diff_opcodes.py):
//!
//! * on hard inputs the bounded search accepts a good non-minimal split, so
//!   the edit script is not guaranteed minimal; it is always VALID (the
//!   opcodes reconstruct both sides; the structural property is pinned by
//!   hypothesis over arbitrary pairs);
//! * where the input is ambiguous, Myers + run-maximization can pick a
//!   different valid alignment than difflib's longest-match recursion (the
//!   pinned boundary cases: `"a"` vs `"baa"`, an insertion split around
//!   difflib's anchored middle match vs one contiguous insertion slid to the
//!   side; and `"ppp"` vs `"pwpp"`, the equal-run boundary sliding across a
//!   repeated-flank insertion point): exact agreement is only promised, and
//!   pinned, on the unambiguous classes (pure insert/delete with differing
//!   flanks, all-equal, single-run replace, empty operands).
//!
//! The superlinear worst case (the `deadline_ms` parameter): on inputs with
//! few anchorable unique records (a character-level permutation of prose is
//! the measured shape), the bounded search's work grows ~n^2 with
//! size (measured on the dev box via `reference.diff_pair_char_shuffled`,
//! ambient load 2.6-3.7: 50k chars 0.32 s, 200k 3.67 s, 400k 13.81 s, 1M
//! 183.6 s; the ladder the README's diff section records).
//! `diff_opcodes_deadline` bounds the whole call with similar's deadline
//! mechanism (`capture_diff_slices_deadline`, which makes the search BAIL
//! at the deadline instead of running on) plus an expiry check of our own:
//! similar 3.2.0 has NO error surface for expiry (verified in its source:
//! `deadline_exceeded` only makes the algorithm fall back to an
//! approximation), so the core checks the elapsed wall against the budget
//! after the call returns and reports [`DeadlineExceeded`]; the pyo3 layer
//! maps that to `TimeoutError`, constructed after the GIL is reacquired.
//!
//! The line-level spelling (v0.8): [`diff_opcodes_lines_deadline`] diffs the
//! SAME operands as LINE sequences (`split_inclusive('\n')`; each line
//! keeps its terminator, the last line may lack one) and reports the same
//! opcode shape with LINE indices, so `a.splitlines(keepends=True)`-style
//! reconstruction composes in Python. The char-level spelling remains the
//! difflib-parity oracle; the line-level one is the shape document/version
//! tooling wants (`difflib` on `splitlines` operands), at the same native
//! speed with the same deadline machinery: one engine, two tokenizations.
//! A line-level diff has far fewer tokens than its char-level twin, so its
//! walls sit proportionally lower at the same corpus size.
//!
//! Pure Rust, no pyo3 types: the criterion bench (benches/diff.rs) drives this
//! path directly; the pyo3 wrapper in `lib.rs` adds only the `&str` argument
//! borrows and the O(ops) return marshalling (see the crate GIL model there).

use std::convert::Infallible;
use std::time::{Duration, Instant};

use similar::algorithms::{DiffHook, diff_slices_deadline};
use similar::{Algorithm, DiffOp, capture_diff_slices_deadline};

/// The difflib opcode tags. `as_str` is the exact spelling `difflib` puts in
/// the tuples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpcodeTag {
    Equal,
    Replace,
    Delete,
    Insert,
}

impl OpcodeTag {
    pub const fn as_str(self) -> &'static str {
        match self {
            OpcodeTag::Equal => "equal",
            OpcodeTag::Replace => "replace",
            OpcodeTag::Delete => "delete",
            OpcodeTag::Insert => "insert",
        }
    }
}

/// One difflib opcode: `a[i1:i2]` ↔ `b[j1:j2]` under `tag`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Opcode {
    pub tag: OpcodeTag,
    pub i1: usize,
    pub i2: usize,
    pub j1: usize,
    pub j2: usize,
}

/// The deadline outcome: the whole call exceeded its caller-supplied budget,
/// so the (incomplete, approximated) result is discarded. Carries both
/// numbers the message names: the deadline and the elapsed cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeadlineExceeded {
    pub deadline_ms: f64,
    pub elapsed_ms: f64,
}

impl DeadlineExceeded {
    /// The `str()` of the `TimeoutError` the pyo3 layer raises.
    pub fn message(&self) -> String {
        format!(
            "diff_opcodes deadline exceeded: elapsed {:.1}ms > deadline_ms {:.1}ms",
            self.elapsed_ms, self.deadline_ms
        )
    }
}

fn map_op(op: DiffOp) -> Opcode {
    match op {
        DiffOp::Equal {
            old_index,
            new_index,
            len,
        } => Opcode {
            tag: OpcodeTag::Equal,
            i1: old_index,
            i2: old_index + len,
            j1: new_index,
            j2: new_index + len,
        },
        DiffOp::Delete {
            old_index,
            old_len,
            new_index,
        } => Opcode {
            tag: OpcodeTag::Delete,
            i1: old_index,
            i2: old_index + old_len,
            j1: new_index,
            j2: new_index,
        },
        DiffOp::Insert {
            old_index,
            new_index,
            new_len,
        } => Opcode {
            tag: OpcodeTag::Insert,
            i1: old_index,
            i2: old_index,
            j1: new_index,
            j2: new_index + new_len,
        },
        DiffOp::Replace {
            old_index,
            old_len,
            new_index,
            new_len,
        } => Opcode {
            tag: OpcodeTag::Replace,
            i1: old_index,
            i2: old_index + old_len,
            j1: new_index,
            j2: new_index + new_len,
        },
    }
}

/// Saturating milliseconds → `Duration`. The pyo3 layer validates
/// positive-and-finite before calling, but an enormous-but-finite budget
/// (e.g. `1e300` ms) overflows `Duration::from_secs_f64` and PANICS
/// (`cannot convert float seconds to Duration`). Saturation
/// maps any out-of-range value to `Duration::MAX` (unbounded by the heat
/// death of the universe, the only sensible reading of a caller-supplied
/// 1e300 ms budget), which makes the core panic-free for every `f64`. NaN
/// and negative values (reachable only through the pub Rust spelling, whose
/// doc states the positive-finite precondition) saturate the same way
/// rather than panicking.
pub(crate) fn budget_from_ms(ms: f64) -> Duration {
    Duration::try_from_secs_f64(ms / 1000.0).unwrap_or(Duration::MAX)
}

/// The elapsed-vs-budget check shared by every `deadline_ms`-bearing
/// primitive in the crate (`diff_opcodes`, `diff_opcodes_lines`,
/// `is_grounded`'s fuzzy scan): the wall clock against the same saturating
/// budget, `(deadline_ms, elapsed_ms)` on expiry. `None` means either no
/// budget was set (can never expire) or the budget hasn't been exceeded yet.
/// Callers wrap the pair into their own `DeadlineExceeded`-shaped type so
/// each keeps a distinct, function-named `message()`.
pub(crate) fn elapsed_exceeds(started: Instant, deadline_ms: Option<f64>) -> Option<(f64, f64)> {
    let ms = deadline_ms?;
    let elapsed = started.elapsed();
    if elapsed > budget_from_ms(ms) {
        Some((ms, elapsed.as_secs_f64() * 1000.0))
    } else {
        None
    }
}

/// The shared expiry tail of both diff spellings: the elapsed wall against
/// the same saturating budget, both message numbers packed. `None` means no
/// budget was set, which can never expire.
fn exceeded_after(started: Instant, deadline_ms: Option<f64>) -> Option<DeadlineExceeded> {
    elapsed_exceeds(started, deadline_ms).map(|(deadline_ms, elapsed_ms)| DeadlineExceeded {
        deadline_ms,
        elapsed_ms,
    })
}

/// The difflib-shaped opcode list for the character-level diff of `a` and `b`.
///
/// Identical operands short-circuit to the single-equal list (one native
/// equality scan; no diff search, no token vectors); the empty pair returns
/// the empty list, difflib's own answer (its sentinel-only matching blocks
/// produce no opcodes).
pub fn diff_opcodes(a: &str, b: &str) -> Vec<Opcode> {
    diff_opcodes_deadline(a, b, None).expect("a None deadline can never be exceeded")
}

/// The deadline-bounded spelling of [`diff_opcodes`]: `deadline_ms` bounds the
/// WHOLE call (the `Vec<char>` materialization, the Myers search, and the
/// identical-input equality scan alike: one clock, started at entry). The
/// caller guarantees a positive finite value where `Some` (the pyo3 layer
/// validates before calling). On expiry the approximated result similar's
/// deadline mechanism produces is DISCARDED and [`DeadlineExceeded`] is
/// returned: an expired budget is a caller-visible `TimeoutError`, never a
/// silently degraded diff. `None` is exactly [`diff_opcodes`]. A
/// finite-but-out-of-range budget saturates to unbounded (the core is
/// panic-free for every `f64`); the positive-finite precondition is the
/// documented contract, not a panic boundary.
pub fn diff_opcodes_deadline(
    a: &str,
    b: &str,
    deadline_ms: Option<f64>,
) -> Result<Vec<Opcode>, DeadlineExceeded> {
    let started = Instant::now();
    // similar takes an absolute Instant (its search bails there); the expiry
    // verdict afterwards compares the elapsed wall against the same budget:
    // one clock, one arithmetic. Saturating: an out-of-range budget means
    // unbounded, and if even the saturated instant overflows the clock,
    // similar gets no deadline at all, the same unbounded semantics.
    let deadline = deadline_ms.and_then(|ms| started.checked_add(budget_from_ms(ms)));
    let opcodes = if a == b {
        // The identical-input short-circuit: one O(n) equality scan, exact,
        // and inside any sane budget: no search, no deadline machinery.
        let len = a.chars().count();
        if len == 0 {
            Vec::new()
        } else {
            vec![Opcode {
                tag: OpcodeTag::Equal,
                i1: 0,
                i2: len,
                j1: 0,
                j2: len,
            }]
        }
    } else {
        let old: Vec<char> = a.chars().collect();
        let new: Vec<char> = b.chars().collect();
        capture_diff_slices_deadline(Algorithm::Myers, &old, &new, deadline)
            .into_iter()
            .map(map_op)
            .collect()
    };
    if let Some(err) = exceeded_after(started, deadline_ms) {
        return Err(err);
    }
    Ok(opcodes)
}

/// Split `text` into lines, each keeping its own `'\n'` terminator: a line
/// is the span through its own `'\n'`, and the LAST line may lack one; `""`
/// yields no lines. This is `str.splitlines(keepends=True)` restricted to
/// the `'\n'` terminator, the single convention document tooling actually
/// wants (`difflib` users diff `splitlines(keepends=True)` operands), kept
/// explicit and simple rather than tolerant of every Unicode line boundary.
pub fn split_keepend_lines(text: &str) -> Vec<&str> {
    text.split_inclusive('\n').collect()
}

/// The line-level spelling of [`diff_opcodes_deadline`]: the same engine
/// (similar's Myers), the same deadline machinery (one clock bounding the
/// whole call; the identical-input fast path and the empty pair behave the
/// same), but the operands are tokenized as LINES ([`split_keepend_lines`])
/// and the returned [`Opcode`] indices are LINE indices: `i1/i2` address
/// `a`'s line vector and `j1/j2` address `b`'s, so
/// `a.splitlines(keepends=True)[i1:i2]`-style reconstruction composes in
/// Python. `map_op` is token-type-agnostic (old/new indices and lengths), so
/// the mapping is shared verbatim with the char-level spelling.
pub fn diff_opcodes_lines_deadline(
    a: &str,
    b: &str,
    deadline_ms: Option<f64>,
) -> Result<Vec<Opcode>, DeadlineExceeded> {
    let started = Instant::now();
    // The same saturating clock as the char-level spelling (see the comment
    // there): one budget, one deadline, one expiry verdict.
    let deadline = deadline_ms.and_then(|ms| started.checked_add(budget_from_ms(ms)));
    let opcodes = if a == b {
        // The identical-input short-circuit, in line units: one O(lines)
        // scan, exact, inside any sane budget: no search.
        let len = split_keepend_lines(a).len();
        if len == 0 {
            Vec::new()
        } else {
            vec![Opcode {
                tag: OpcodeTag::Equal,
                i1: 0,
                i2: len,
                j1: 0,
                j2: len,
            }]
        }
    } else {
        let old = split_keepend_lines(a);
        let new = split_keepend_lines(b);
        capture_diff_slices_deadline(Algorithm::Myers, &old, &new, deadline)
            .into_iter()
            .map(map_op)
            .collect()
    };
    if let Some(err) = exceeded_after(started, deadline_ms) {
        return Err(err);
    }
    Ok(opcodes)
}

// --- the similarity surface (difflib's ratio / get_close_matches shape) ---
//
// `difflib.SequenceMatcher.ratio()` and
// `difflib.get_close_matches()` at native speed over the same vendored
// Myers engine: difflib's spellings are pure Python and O(n^2)-worst-case,
// a famous pain point. The governing caveat: difflib's ratio is 2.0*M/T
// over ITS OWN anchored alignment (M the sum of the matching blocks'
// sizes, difflib.py:597-620), an anchoring-dependent number, NOT a
// canonical metric; this surface's contract is
// validity-first, the diff_opcodes boundary-class lesson over again:
// `similarity_ratio` is 2.0*M/T over THE SAME Myers equal-ops this crate's
// `diff_opcodes` emits (M the sum of the equal ops' lengths), with exact
// difflib parity pinned only on the forced-alignment classes, namely
// identical operands (1.0), the empty-pair classes, disjoint alphabets
// (0.0), and pure insert/delete with differing flanks; the divergent
// repeated-flank rows are pinned as divergent (both numbers valid, both
// in [0,1]). Every
// difflib value cited in the docs and tests below was cross-checked
// against the running stdlib (python3.12 difflib) before pinning.
//
// M is accumulated by a counting hook driven through the SAME Myers
// entry point the capture spelling uses (see [`matched_chars`]): a scored
// pair allocates only its two `Vec<char>` tokenizations (no op vector, no
// per-op enum traffic), which is what makes a 10k-candidate
// `close_matches` scan pay the search itself, not 10k op-vector
// materializations. The bulk spelling (`close_matches`) layers a provable
// length-ratio prefilter on top: a candidate whose ratio ceiling is
// STRICTLY under the cutoff is skipped before any tokenization (see
// [`ratio_ceiling`]), so the all-miss scan class runs no searches at all.

/// difflib's `_calculate_ratio` verbatim (difflib.py:39-42): `2.0*M/T`,
/// with the zero-total guard returning `1.0`. That is difflib's own answer
/// for the empty pair (`SequenceMatcher(None, "", "").ratio() == 1.0`,
/// probe-verified), so the guard is parity, not a division-avoidance hack
/// of ours.
fn ratio_from_matches(matches: usize, total: usize) -> f64 {
    if total > 0 {
        2.0 * matches as f64 / total as f64
    } else {
        1.0
    }
}

/// The provable ceiling on a pair's ratio from the two CHAR lengths alone:
/// `2.0 * min(len_a, len_b) / (len_a + len_b)`. The proof is three lines:
/// a match consumes one char from each side, so the matched total M can
/// never exceed the shorter operand, `M <= min(len_a, len_b)`; the ratio
/// is `2.0*M/T` with `T = len_a + len_b` (difflib.py:39-42); therefore
/// `ratio <= 2.0*min(len_a, len_b)/T`, the ceiling. The skip rule in
/// [`close_matches`] is STRICT: `ceiling < cutoff` skips (every
/// realizable ratio is strictly under the cutoff, so the `>=` gate at
/// difflib.py:706 can never fire), `ceiling >= cutoff` searches, because
/// at equality the candidate can still qualify: `M == min_len` realizes
/// `ratio == ceiling == cutoff` exactly, and the gate keeps it. The
/// both-empty pair (`0/0`) is NaN, which compares false against
/// `< cutoff` and so is always searched, which is correct: its ratio is
/// difflib's own 1.0 (the zero-total guard), which clears any cutoff.
fn ratio_ceiling(len_a: usize, len_b: usize) -> f64 {
    2.0 * len_a.min(len_b) as f64 / (len_a + len_b) as f64
}

/// The M-sink: a [`DiffHook`] accumulating exactly the one number the
/// similarity surface is defined over: the sum of the equal ops' lengths,
/// the matched-char total M. `delete`/`insert`/`replace` contribute nothing
/// to M and `finish` carries no payload, so the trait's default no-op
/// implementations ARE the hook's behavior there; only `equal` is
/// overridden. The error type is `Infallible` because a counting sink
/// cannot fail: the driven diff's `Result` is a formality.
struct CountEqualHook(usize);

impl DiffHook for CountEqualHook {
    type Error = Infallible;

    fn equal(
        &mut self,
        _old_index: usize,
        _new_index: usize,
        len: usize,
    ) -> Result<(), Self::Error> {
        self.0 += len;
        Ok(())
    }
}

/// The shared M computation, the spine of both similarity spellings: the
/// matched-char total over THE SAME Myers equal-ops [`diff_opcodes`] emits,
/// accumulated by [`CountEqualHook`] through `diff_slices_deadline`: the
/// very dispatcher call (`algorithms::diff_deadline` on the full `0..len`
/// ranges, hence the same `myers::diff_deadline` search, preflight, and
/// deadline-bail checks) that `capture_diff_slices_deadline` drives its
/// `Compact`/`Replace`/`Capture` chain through, so the count IS the capture
/// spelling's equal-op sum without materializing the op vector: one pair
/// costs its two `Vec<char>` tokenizations and nothing else (the capture
/// path's presentation hooks only merge and shift ops, which preserves the
/// equal-length total; a differential battery pins the agreement). The
/// identical-input fast path counts `a`'s chars: exactly the single equal
/// op's length, no search. `deadline` is similar's absolute bail Instant,
/// passed through verbatim; the expiry verdict is the CALLER's, because the
/// two spellings budget differently: one clock per pair for
/// [`similarity_ratio_deadline`], ONE shared clock for the whole candidate
/// scan in [`close_matches`].
fn matched_chars(a: &str, b: &str, deadline: Option<Instant>) -> usize {
    if a == b {
        a.chars().count()
    } else {
        let old: Vec<char> = a.chars().collect();
        let new: Vec<char> = b.chars().collect();
        let mut hook = CountEqualHook(0);
        diff_slices_deadline(Algorithm::Myers, &mut hook, &old, &new, deadline)
            .expect("the counting sink cannot fail");
        hook.0
    }
}

/// The per-pair spelling of [`matched_chars`]: the same one-clock-per-call
/// shape as [`diff_opcodes_deadline`] (the saturating budget, similar's
/// bail Instant, and the expiry verdict after the search; see the comment
/// there), yielding the matched total or [`DeadlineExceeded`].
fn matched_chars_deadline(
    a: &str,
    b: &str,
    deadline_ms: Option<f64>,
) -> Result<usize, DeadlineExceeded> {
    let started = Instant::now();
    let deadline = deadline_ms.and_then(|ms| started.checked_add(budget_from_ms(ms)));
    let matches = matched_chars(a, b, deadline);
    if let Some(err) = exceeded_after(started, deadline_ms) {
        return Err(err);
    }
    Ok(matches)
}

/// `difflib.SequenceMatcher(None, a, b).ratio()` at native speed: `2.0*M/T`
/// with `T` the total char length of both operands and `M` the matched
/// total over THE SAME Myers equal-ops [`diff_opcodes`] emits on the pair
/// (the shared [`matched_chars`] spine). Bounded to `[0.0, 1.0]`, symmetric,
/// and exactly `1.0` iff `a == b`.
///
/// The degenerate rows are difflib's own answers, probe-verified against
/// the 3.12 source: the empty pair is `1.0`: `_calculate_ratio`
/// (difflib.py:39-42) returns 1.0 for a zero total rather than dividing.
/// Empty vs nonempty is `0.0` (`M = 0`, `T > 0`). Identical operands
/// short-circuit to `1.0` without a search (one native equality scan, the
/// same fast path the opcode spellings take).
///
/// The parity caveat: difflib's ratio is anchoring-dependent, not a
/// canonical metric. Its `M` sums ITS OWN longest-match blocks
/// (difflib.py:597-620), so on repeated-flank/ambiguous inputs the two
/// algorithms' `M` can differ (both valid alignments, both ratios in
/// `[0,1]`). Exact parity holds where the alignment is FORCED (identical
/// operands, the empty-pair classes, disjoint alphabets, pure
/// insert/delete with differing flanks), and where the placement of
/// matches diverges but the total does not, the ratios still agree (the
/// pinned `"a"` vs `"baa"` row: both `M = 1` → `0.5`). The pinned
/// divergence rows: `"ppp"` vs `"pwpp"`, where difflib anchors `"pp"` and
/// starves the flank (`M = 2` → `4/7` ≈ 0.5714) while the Myers one-insert
/// script matches `1 + 2 = 3` → `6/7`; and `"qpqpq"` vs `"qpwqpq"`, where
/// difflib anchors `"qpq"` (`M = 3` → `6/11` ≈ 0.5455) while
/// prefix/suffix trimming leaves a one-insert middle (`M = 5` → `10/11`).
pub fn similarity_ratio(a: &str, b: &str) -> f64 {
    similarity_ratio_deadline(a, b, None).expect("a None deadline can never be exceeded")
}

/// The deadline-bounded spelling of [`similarity_ratio`]: `deadline_ms`
/// bounds the WHOLE call (the `Vec<char>` materialization, the Myers
/// search, and the identical-input fast path alike, one clock started at
/// entry), with the same saturating arithmetic and expiry tail as
/// [`diff_opcodes_deadline`] ([`budget_from_ms`]/[`exceeded_after`], the
/// same bail-then-verdict two-step over similar's deadline mechanism). The
/// caller guarantees a positive finite value where `Some` (the pyo3 layer
/// validates before calling). On expiry the approximated M similar's
/// deadline mechanism produces is DISCARDED and [`DeadlineExceeded`] is
/// returned: the `TimeoutError` contract, never a silently degraded
/// score. `None` is exactly [`similarity_ratio`]. The O(ND) worst case on
/// hard pairs (the character-permutation shape, the same DoS class the
/// opcode spellings document) is the whole reason this spelling exists.
pub fn similarity_ratio_deadline(
    a: &str,
    b: &str,
    deadline_ms: Option<f64>,
) -> Result<f64, DeadlineExceeded> {
    let matches = matched_chars_deadline(a, b, deadline_ms)?;
    Ok(ratio_from_matches(
        matches,
        a.chars().count() + b.chars().count(),
    ))
}

/// difflib's ordering-and-truncation tail (difflib.py:707, difflib.py:710):
/// `heapq.nlargest` over the `(ratio, candidate)` tuples, whose documented
/// equivalence is `sorted(iterable, reverse=True)[:n]`: score descending,
/// then the candidate itself descending, exact duplicates (equal score AND
/// equal string) stable in input order, truncated to `n` and projected to
/// the candidate indices. Shared by [`close_matches`] and the test
/// module's unfiltered differential spelling, so the two can differ ONLY
/// in the prefilter.
fn best_first(mut scored: Vec<(f64, usize)>, candidates: &[&str], n: usize) -> Vec<usize> {
    scored.sort_by(|&(score_a, idx_a), &(score_b, idx_b)| {
        score_b
            .total_cmp(&score_a)
            .then_with(|| candidates[idx_b].cmp(candidates[idx_a]))
            .then_with(|| idx_a.cmp(&idx_b))
    });
    scored.truncate(n);
    scored.into_iter().map(|(_, idx)| idx).collect()
}

/// The test-only counting seam behind the skip-fires pins: a per-THREAD
/// counter of the `matched_chars` calls [`close_matches`] issues, so the
/// tests can assert exactly which candidates were searched. Per-thread
/// because the suite's tests run in parallel: each test's calls bump its
/// own thread's counter and cannot interfere. Production builds compile
/// none of this.
#[cfg(test)]
mod search_seam {
    use std::cell::Cell;

    thread_local! {
        static SEARCHES: Cell<usize> = const { Cell::new(0) };
    }

    pub fn reset() {
        SEARCHES.with(|c| c.set(0));
    }

    pub fn counted() -> usize {
        SEARCHES.with(Cell::get)
    }

    pub fn bump() {
        SEARCHES.with(|c| c.set(c.get() + 1));
    }
}

/// `difflib.get_close_matches(word, possibilities, n, cutoff)` at native
/// speed, returning the INDICES of the matched candidates (best-first)
/// rather than the strings: the pyo3 layer maps indices back to the
/// original `str` objects, zero-copy. Every candidate is scored by
/// [`similarity_ratio`] over char tokens; candidates scoring `>= cutoff`
/// are kept (difflib.py:706; a score exactly equal to the cutoff is
/// kept), ordered best-first, and truncated to `n`.
///
/// The scan is prefILTERED by a provable ceiling before any search (see
/// [`ratio_ceiling`] for the three-line proof): a candidate whose length
/// ratio `2.0*min(len_word, len_cand)/(len_word + len_cand)` is STRICTLY
/// below `cutoff` can never reach the cutoff, so its Myers search is pure
/// waste, exactly the under-cutoff miss class a bulk scan against a
/// short-word dictionary measures (the bench's own shape: a 41-char query
/// against a dictionary whose longest entry is 17 chars, where every
/// single candidate is under-ceiling at the 0.6 default). A skipped
/// candidate costs one `chars().count()` pass over its bytes and
/// allocates nothing: the two `Vec<char>` tokenizations a searched pair
/// pays inside [`matched_chars`] never happen, so a 10k-candidate miss
/// scan runs zero searches and zero token-vector allocations.
///
/// The order is difflib's ACTUAL rule, not input order: difflib appends
/// `(ratio, x)` pairs (difflib.py:707) and takes `heapq.nlargest(n, ...)`
/// (difflib.py:710, the import at difflib.py:33), whose documented
/// equivalence is `sorted(iterable, reverse=True)[:n]`: the WHOLE tuple
/// compares, so equal scores are broken by the candidate itself in
/// DESCENDING order (for `str`, code-point order; Rust's `str` ordering is
/// the same bytes-equal-codepoints order), and exact duplicates (equal
/// score AND equal string) stay in input order. Probe-verified against the
/// running stdlib: `get_close_matches("ab", ["ac", "ca"], 2, 0.5)` →
/// `['ca', 'ac']`. Input order is NOT the tie rule.
///
/// One `deadline_ms` bounds the WHOLE scan: every candidate's search bails
/// at the same absolute Instant, and the expiry verdict is re-checked after
/// each candidate, searched or skipped alike: a budget spent mid-list STOPS
/// the scan, the partial result list is DISCARDED, and [`DeadlineExceeded`]
/// is returned
/// (the `TimeoutError` contract: a truncated close-matches list would look
/// like a real answer). `None` is unbounded.
///
/// Preconditions the PYO3 layer enforces (difflib itself raises ValueError
/// there, difflib.py:695-698): `n > 0` and `cutoff` in `[0.0, 1.0]`. The
/// Rust spelling does not re-validate: `n == 0` truncates to the empty
/// list, and an out-of-range cutoff merely changes which scores clear the
/// gate.
pub fn close_matches(
    word: &str,
    candidates: &[&str],
    n: usize,
    cutoff: f64,
    deadline_ms: Option<f64>,
) -> Result<Vec<usize>, DeadlineExceeded> {
    let started = Instant::now();
    // The same saturating clock as the pair spellings, but ONE budget for
    // the whole scan: every candidate's search bails at the same absolute
    // Instant, and the expiry verdict is re-checked after each candidate.
    let deadline = deadline_ms.and_then(|ms| started.checked_add(budget_from_ms(ms)));
    let word_len = word.chars().count();
    let mut scored: Vec<(f64, usize)> = Vec::new();
    for (idx, candidate) in candidates.iter().enumerate() {
        // The cheap length pass FIRST: one scan of the candidate's bytes,
        // no allocation, feeding the ceiling test before any tokenization.
        let cand_len = candidate.chars().count();
        let score = if ratio_ceiling(word_len, cand_len) < cutoff {
            // The provable skip: this candidate's ratio cannot reach the
            // cutoff (see [`ratio_ceiling`]), so its Myers search is pure
            // waste. A skipped candidate allocates NOTHING, not even the
            // two `Vec<char>` tokenizations a searched pair pays.
            None
        } else {
            #[cfg(test)]
            search_seam::bump();
            let matches = matched_chars(word, candidate, deadline);
            Some(ratio_from_matches(matches, word_len + cand_len))
        };
        if let Some(err) = exceeded_after(started, deadline_ms) {
            // A budget spent mid-list stops the scan, skipped candidate or
            // searched alike; the partial result list is discarded, never a
            // silently truncated answer.
            return Err(err);
        }
        if let Some(score) = score.filter(|&s| s >= cutoff) {
            scored.push((score, idx));
        }
    }
    Ok(best_first(scored, candidates, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ops(a: &str, b: &str) -> Vec<(&'static str, usize, usize, usize, usize)> {
        diff_opcodes(a, b)
            .into_iter()
            .map(|op| (op.tag.as_str(), op.i1, op.i2, op.j1, op.j2))
            .collect()
    }

    /// The structural contract the Python-side hypothesis property pins over
    /// arbitrary pairs, checked here over a generated battery: every pair of
    /// strings over {'a', 'b'} up to length 4 (625 pairs): the mapping must
    /// produce a valid, difflib-shaped cover of both sides (contiguity,
    /// coverage, alternation, per-tag nonemptiness, equal-content equals,
    /// full reconstruction).
    fn assert_valid(a: &str, b: &str) {
        let old: Vec<char> = a.chars().collect();
        let new: Vec<char> = b.chars().collect();
        let mut rebuilt_a = String::new();
        let mut rebuilt_b = String::new();
        let mut prev = (0usize, 0usize);
        let mut prev_equal: Option<bool> = None;
        for op in diff_opcodes(a, b) {
            assert_eq!(
                (op.i1, op.j1),
                prev,
                "contiguity broken on {a:?} vs {b:?}: {op:?}"
            );
            let is_equal = op.tag == OpcodeTag::Equal;
            assert!(
                prev_equal != Some(is_equal),
                "tags do not alternate on {a:?} vs {b:?}: {op:?}"
            );
            match op.tag {
                OpcodeTag::Equal => {
                    assert!(op.i1 < op.i2 && op.j1 < op.j2, "empty equal op: {op:?}");
                    assert_eq!(
                        op.i2 - op.i1,
                        op.j2 - op.j1,
                        "unequal equal lengths: {op:?}"
                    );
                    let left: String = old[op.i1..op.i2].iter().collect();
                    let right: String = new[op.j1..op.j2].iter().collect();
                    assert_eq!(left, right, "equal content mismatch: {op:?}");
                    rebuilt_a.push_str(&left);
                    rebuilt_b.push_str(&right);
                }
                OpcodeTag::Delete => {
                    assert!(op.i1 < op.i2, "empty delete op: {op:?}");
                    rebuilt_a.push_str(&old[op.i1..op.i2].iter().collect::<String>());
                }
                OpcodeTag::Insert => {
                    assert!(op.j1 < op.j2, "empty insert op: {op:?}");
                    rebuilt_b.push_str(&new[op.j1..op.j2].iter().collect::<String>());
                }
                OpcodeTag::Replace => {
                    assert!(op.i1 < op.i2 && op.j1 < op.j2, "empty replace op: {op:?}");
                    rebuilt_a.push_str(&old[op.i1..op.i2].iter().collect::<String>());
                    rebuilt_b.push_str(&new[op.j1..op.j2].iter().collect::<String>());
                }
            }
            prev = (op.i2, op.j2);
            prev_equal = Some(is_equal);
        }
        assert_eq!(
            prev,
            (old.len(), new.len()),
            "coverage broken on {a:?} vs {b:?}"
        );
        assert_eq!(rebuilt_a, a, "a not reconstructed from {a:?} vs {b:?}");
        assert_eq!(rebuilt_b, b, "b not reconstructed from {a:?} vs {b:?}");
    }

    fn all_strings(len: usize, cur: &mut String, out: &mut Vec<String>) {
        if cur.len() == len {
            out.push(cur.clone());
            return;
        }
        for c in ['a', 'b'] {
            cur.push(c);
            all_strings(len, cur, out);
            cur.pop();
        }
    }

    #[test]
    fn identical_inputs_short_circuit_to_one_equal_op() {
        assert_eq!(ops("abc", "abc"), vec![("equal", 0, 3, 0, 3)]);
        // Non-ASCII: the fast path counts chars (Python str indices), not bytes.
        assert_eq!(ops("café", "café"), vec![("equal", 0, 4, 0, 4)]);
    }

    #[test]
    fn the_empty_pair_returns_the_empty_list() {
        // difflib's own answer for ("", ""): the sentinel-only matching
        // blocks produce no opcodes, not a zero-length equal op.
        assert_eq!(
            ops("", ""),
            Vec::<(&str, usize, usize, usize, usize)>::new()
        );
    }

    #[test]
    fn empty_versus_nonempty_covers_the_whole_operand_in_one_op() {
        assert_eq!(ops("", "abc"), vec![("insert", 0, 0, 0, 3)]);
        assert_eq!(ops("abc", ""), vec![("delete", 0, 3, 0, 0)]);
    }

    #[test]
    fn adjacent_delete_and_insert_present_as_one_replace() {
        // The merged presentation difflib's get_opcodes emits for a
        // same-position change: never a delete next to an insert.
        assert_eq!(
            ops("abXcd", "abYcd"),
            vec![
                ("equal", 0, 2, 0, 2),
                ("replace", 2, 3, 2, 3),
                ("equal", 3, 5, 3, 5)
            ]
        );
    }

    #[test]
    fn the_boundary_case_opcodes_are_pinned() {
        // The documented, intentional divergences from difflib (whose
        // longest-match recursion anchors a match and splits the change
        // around it, where Myers + run-maximization emits contiguous runs
        // slid to one side): "a" vs "baa", where the anchored middle 'a'
        // splits the insertion; and "ppp" vs "pwpp" (plus its delete-class
        // twin), where the equal-run boundary slides across a repeated-flank
        // insertion
        // point. Pinned crate-side so the presentation cannot drift
        // independently of the Python-side pins.
        assert_eq!(
            ops("a", "baa"),
            vec![("insert", 0, 0, 0, 2), ("equal", 0, 1, 2, 3)]
        );
        assert_eq!(
            ops("ppp", "pwpp"),
            vec![
                ("equal", 0, 1, 0, 1),
                ("insert", 1, 1, 1, 2),
                ("equal", 1, 3, 2, 4)
            ]
        );
        assert_eq!(
            ops("pXpp", "ppp"),
            vec![
                ("equal", 0, 1, 0, 1),
                ("delete", 1, 2, 1, 1),
                ("equal", 2, 4, 1, 3)
            ]
        );
    }

    #[test]
    fn every_pair_over_a_two_letter_alphabet_up_to_length_four_is_valid() {
        let mut strings = Vec::new();
        for len in 0..=4 {
            let mut bucket = Vec::new();
            all_strings(len, &mut String::new(), &mut bucket);
            strings.extend(bucket);
        }
        for a in &strings {
            for b in &strings {
                assert_valid(a, b);
            }
        }
    }

    #[test]
    fn non_ascii_pairs_diff_by_codepoint_not_by_byte() {
        // Combining-mark text: indices must be codepoint offsets (a Python
        // str index), which is what the reconstruction over chars guarantees.
        assert_eq!(
            ops("cafe\u{301}", "cafe"),
            vec![("equal", 0, 4, 0, 4), ("delete", 4, 5, 4, 4)]
        );
        assert_eq!(
            ops("égal", "egal"),
            vec![("replace", 0, 1, 0, 1), ("equal", 1, 4, 1, 4)]
        );
    }

    // --- the deadline (the pyo3 layer's deadline_ms) ----------------------
    //
    // The hard shape a deadline exists for: a character permutation of a
    // small-alphabet text has almost no anchorable unique records, so the
    // bounded search's work grows superlinearly (the README's diff section
    // records the measured ladder). Built here with a deterministic
    // xorshift Fisher-Yates over a 3-char alphabet: small enough to build
    // in microseconds, hard enough that its unbounded diff costs far more
    // than the deadline the tests set.
    fn char_permutation(text: &str) -> String {
        let mut chars: Vec<char> = text.chars().collect();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in (1..chars.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let j = (state % (i as u64 + 1)) as usize;
            chars.swap(i, j);
        }
        chars.into_iter().collect()
    }

    #[test]
    fn a_none_deadline_is_the_unbounded_diff() {
        // The additive-parameter contract: None (the default) is exactly the
        // plain spelling: same opcodes, no expiry possible.
        let a = "abcabcabcXdefdefdef";
        let b = "abcabcabcYdefdefdef";
        assert_eq!(diff_opcodes_deadline(a, b, None), Ok(diff_opcodes(a, b)));
    }

    #[test]
    fn a_generous_deadline_yields_the_identical_opcodes() {
        // A budget the search never reaches changes nothing: similar's
        // preflight only SKIPS work once a deadline has expired (verified in
        // its 3.2.0 source), so a far-future deadline is the same code path.
        let a = "the quick brown fox";
        let b = "the quick brown dog";
        assert_eq!(
            diff_opcodes_deadline(a, b, Some(60_000.0)),
            Ok(diff_opcodes(a, b))
        );
    }

    #[test]
    fn an_out_of_range_finite_budget_saturates_to_unbounded() {
        // 1e300 ms is positive and finite (so it passes the pyo3 layer's
        // validation) but overflows Duration::from_secs_f64, which would
        // PANICK ("cannot convert float seconds to Duration"). Saturation
        // maps it to Duration::MAX (unbounded), so the opcodes are
        // exactly the plain diff's.
        let a = "abcabcabcXdefdefdef";
        let b = "abcabcabcYdefdefdef";
        assert_eq!(
            diff_opcodes_deadline(a, b, Some(1e300)),
            Ok(diff_opcodes(a, b))
        );
    }

    #[test]
    fn a_nan_budget_saturates_to_unbounded_too() {
        // NaN is reachable only through the pub Rust spelling (the pyo3
        // layer rejects it); the documented positive-finite precondition
        // maps it to the same saturation rather than a panic.
        let a = "abcabcabcXdefdefdef";
        let b = "abcabcabcYdefdefdef";
        assert_eq!(
            diff_opcodes_deadline(a, b, Some(f64::NAN)),
            Ok(diff_opcodes(a, b))
        );
    }

    #[test]
    fn an_expired_deadline_reports_both_numbers_and_discards_the_result() {
        let base: String = "abc".repeat(20_000);
        let a = base.as_str();
        let b = char_permutation(a);
        // The unbounded diff of this pair cannot finish inside the 5ms
        // budget (the nearest measured anchor: a 50k-char character
        // permutation cost 0.33s on the dev box at ambient load 2.7; the
        // 3-char alphabet here is no easier), so the deadline fires; the
        // assert below is the proof, not the anchor.
        let err = diff_opcodes_deadline(a, &b, Some(5.0)).unwrap_err();
        assert_eq!(err.deadline_ms, 5.0);
        assert!(err.elapsed_ms > 5.0, "elapsed must exceed the budget");
        // The message names both numbers: the str() the TimeoutError carries.
        let message = err.message();
        assert!(message.contains("deadline_ms 5.0ms"), "{message}");
        assert!(message.contains("elapsed "), "{message}");
        // And the deadline actually bounded the work: the call returned in
        // well under the unbounded cost (loose bound for a loaded runner).
        let started = Instant::now();
        assert!(diff_opcodes_deadline(a, &b, Some(5.0)).is_err());
        assert!(started.elapsed().as_secs_f64() < 10.0);
    }

    #[test]
    fn identical_inputs_under_a_deadline_short_circuit_without_a_search() {
        // The equality scan stays inside any sane budget and is exact, so the
        // deadline machinery never fires for it.
        let a = "abc".repeat(1000);
        assert_eq!(
            diff_opcodes_deadline(&a, &a, Some(5_000.0)),
            Ok(vec![Opcode {
                tag: OpcodeTag::Equal,
                i1: 0,
                i2: 3000,
                j1: 0,
                j2: 3000
            }])
        );
        assert_eq!(diff_opcodes_deadline("", "", Some(5_000.0)), Ok(Vec::new()));
    }

    // --- the line-level spelling (v0.8) -------------------------------------
    //
    // The same engine and structural contract as the char-level battery
    // above, with the operands tokenized as split_keepend_lines lines and
    // every index a LINE index.

    fn line_ops(a: &str, b: &str) -> Vec<(&'static str, usize, usize, usize, usize)> {
        diff_opcodes_lines_deadline(a, b, None)
            .expect("a None deadline can never be exceeded")
            .into_iter()
            .map(|op| (op.tag.as_str(), op.i1, op.i2, op.j1, op.j2))
            .collect()
    }

    #[test]
    fn split_keepend_lines_keeps_each_terminator_and_empties_to_no_lines() {
        // A line is the span through its own '\n' (the last line may lack
        // one); the empty string is zero lines, not one empty line.
        assert_eq!(split_keepend_lines(""), Vec::<&str>::new());
        assert_eq!(split_keepend_lines("a"), vec!["a"]);
        assert_eq!(split_keepend_lines("a\n"), vec!["a\n"]);
        assert_eq!(split_keepend_lines("a\nb"), vec!["a\n", "b"]);
        assert_eq!(split_keepend_lines("\n\n"), vec!["\n", "\n"]);
    }

    #[test]
    fn split_keepend_lines_diverges_from_str_splitlines_on_non_lf_terminators() {
        // The pinned narrower-than-splitlines contract (see the
        // module docs and docs/api.md): only '\n' is a terminator. Python's
        // str.splitlines(keepends=True) also breaks on lone '\r', '\v',
        // '\f', U+2028, U+2029, etc.; text using ONLY those never splits
        // here, where a caller assuming full splitlines() semantics would
        // expect multiple lines. Regression-pinned so a future change to
        // this behavior is a visible diff, not silent.
        assert_eq!(
            split_keepend_lines("line1\rline2\rline3"),
            vec!["line1\rline2\rline3"],
            "lone CR is not a line terminator for tors, unlike str.splitlines()"
        );
        assert_eq!(
            split_keepend_lines("a\u{2028}b\u{2029}c"),
            vec!["a\u{2028}b\u{2029}c"],
            "Unicode line/paragraph separators are not line terminators for tors"
        );
        // CRLF *does* split correctly, but only because the '\n' half of it
        // is a real terminator: the '\r' rides along as trailing content on
        // the previous line, not because CRLF is specially recognized.
        assert_eq!(
            split_keepend_lines("a\r\nb"),
            vec!["a\r\n", "b"],
            "CRLF splits on its '\\n', with '\\r' as ordinary trailing content"
        );
    }

    #[test]
    fn line_opcodes_pin_the_golden_shapes() {
        // Identical operands: one equal op over the LINE count (the fast
        // path counts lines, not chars).
        assert_eq!(
            line_ops("l1\nl2\n", "l1\nl2\n"),
            vec![("equal", 0, 2, 0, 2)]
        );
        // The empty pair: difflib's own answer, no opcodes at all.
        assert_eq!(
            line_ops("", ""),
            Vec::<(&str, usize, usize, usize, usize)>::new()
        );
        // Empty vs nonempty: one op covering the whole operand, in lines.
        assert_eq!(line_ops("", "a\nb\n"), vec![("insert", 0, 0, 0, 2)]);
        assert_eq!(line_ops("a\nb\n", ""), vec![("delete", 0, 2, 0, 0)]);
        // A one-line replacement: the merged presentation, never a delete
        // next to an insert.
        assert_eq!(line_ops("x\n", "y\n"), vec![("replace", 0, 1, 0, 1)]);
        // An insert and a delete at DIFFERENT positions (the "abXcd"-style
        // shape at line grain): delete 'b' early, insert 'Z' at the far
        // end, the shared flanks equal.
        assert_eq!(
            line_ops("a\nb\nc\nd\ne\n", "a\nc\nd\ne\nZ\n"),
            vec![
                ("equal", 0, 1, 0, 1),
                ("delete", 1, 2, 1, 1),
                ("equal", 2, 5, 1, 4),
                ("insert", 5, 5, 4, 5)
            ]
        );
        // Adjacent delete+insert still merges into one replace at line
        // grain (old lines 'b','X' out, new line 'Y' in, same position).
        assert_eq!(
            line_ops("a\nb\nX\nc\nd\n", "a\nY\nc\nd\n"),
            vec![
                ("equal", 0, 1, 0, 1),
                ("replace", 1, 3, 1, 2),
                ("equal", 3, 5, 2, 4)
            ]
        );
    }

    #[test]
    fn the_line_level_boundary_case_is_pinned() {
        // The line-level instance of the documented char-level boundary
        // class ("ppp" vs "pwpp"): Myers + run-maximization slides the
        // equal-run boundary across a repeated-flank insertion point: the
        // insert lands after the first 'p' line and the remaining 'p' lines
        // match as one run, where difflib's anchored middle match would
        // split the alignment differently.
        assert_eq!(
            line_ops("p\np\np\n", "p\nw\np\np\n"),
            vec![
                ("equal", 0, 1, 0, 1),
                ("insert", 1, 1, 1, 2),
                ("equal", 1, 3, 2, 4)
            ]
        );
    }

    /// The line-level twin of [`assert_valid`]: the same structural
    /// contract (contiguity, coverage, alternation, per-tag nonemptiness,
    /// equal-content equal-length equals, full reconstruction of both
    /// sides), checked over the CANONICAL line tokenization of the texts:
    /// the indices address `split_keepend_lines` output, so the ground
    /// truth is that split, not any equivalent segmentation.
    fn assert_valid_lines(text_a: &str, text_b: &str) {
        let a = split_keepend_lines(text_a);
        let b = split_keepend_lines(text_b);
        let mut rebuilt_a = String::new();
        let mut rebuilt_b = String::new();
        let mut prev = (0usize, 0usize);
        let mut prev_equal: Option<bool> = None;
        for op in diff_opcodes_lines_deadline(text_a, text_b, None)
            .expect("a None deadline can never be exceeded")
        {
            assert_eq!(
                (op.i1, op.j1),
                prev,
                "contiguity broken on {text_a:?} vs {text_b:?}: {op:?}"
            );
            let is_equal = op.tag == OpcodeTag::Equal;
            assert!(
                prev_equal != Some(is_equal),
                "tags do not alternate on {text_a:?} vs {text_b:?}: {op:?}"
            );
            match op.tag {
                OpcodeTag::Equal => {
                    assert!(op.i1 < op.i2 && op.j1 < op.j2, "empty equal op: {op:?}");
                    assert_eq!(
                        op.i2 - op.i1,
                        op.j2 - op.j1,
                        "unequal equal lengths: {op:?}"
                    );
                    let left = a[op.i1..op.i2].concat();
                    let right = b[op.j1..op.j2].concat();
                    assert_eq!(left, right, "equal content mismatch: {op:?}");
                    rebuilt_a.push_str(&left);
                    rebuilt_b.push_str(&right);
                }
                OpcodeTag::Delete => {
                    assert!(op.i1 < op.i2, "empty delete op: {op:?}");
                    rebuilt_a.push_str(&a[op.i1..op.i2].concat());
                }
                OpcodeTag::Insert => {
                    assert!(op.j1 < op.j2, "empty insert op: {op:?}");
                    rebuilt_b.push_str(&b[op.j1..op.j2].concat());
                }
                OpcodeTag::Replace => {
                    assert!(op.i1 < op.i2 && op.j1 < op.j2, "empty replace op: {op:?}");
                    rebuilt_a.push_str(&a[op.i1..op.i2].concat());
                    rebuilt_b.push_str(&b[op.j1..op.j2].concat());
                }
            }
            prev = (op.i2, op.j2);
            prev_equal = Some(is_equal);
        }
        assert_eq!(
            prev,
            (a.len(), b.len()),
            "coverage broken on {text_a:?} vs {text_b:?}"
        );
        assert_eq!(
            rebuilt_a, text_a,
            "a not reconstructed: {text_a:?} vs {text_b:?}"
        );
        assert_eq!(
            rebuilt_b, text_b,
            "b not reconstructed: {text_a:?} vs {text_b:?}"
        );
    }

    fn all_line_lists(len: usize, cur: &mut Vec<&'static str>, out: &mut Vec<Vec<&'static str>>) {
        if cur.len() == len {
            out.push(cur.clone());
            return;
        }
        for line in ["a\n", "b\n", "ab\n"] {
            cur.push(line);
            all_line_lists(len, cur, out);
            cur.pop();
        }
    }

    #[test]
    fn every_line_pair_over_a_three_line_alphabet_up_to_length_four_is_valid() {
        // Note the alphabet collides on concatenation ("ab\n" vs
        // "a\nb\n"): distinct line lists can produce equal texts, which
        // exercise the identical-input fast path over line counts; the
        // canonical split is the ground truth either way.
        let mut lists = Vec::new();
        for len in 0..=4 {
            let mut bucket = Vec::new();
            all_line_lists(len, &mut Vec::new(), &mut bucket);
            lists.extend(bucket);
        }
        for a in &lists {
            for b in &lists {
                assert_valid_lines(&a.concat(), &b.concat());
            }
        }
    }

    #[test]
    fn a_generous_line_deadline_yields_the_identical_opcodes() {
        // The additive-parameter contract at line grain, including the
        // saturated unbounded spelling (the same budget helper serves both
        // tokenizations).
        let a = "the quick brown fox\njumps over the lazy dog\n";
        let b = "the quick brown fox\nleaps over the lazy dog\n";
        let expected = diff_opcodes_lines_deadline(a, b, None);
        assert_eq!(diff_opcodes_lines_deadline(a, b, Some(60_000.0)), expected);
        assert_eq!(diff_opcodes_lines_deadline(a, b, Some(1e300)), expected);
    }

    #[test]
    fn an_expired_line_deadline_reports_deadline_exceeded() {
        // The hard shape at line grain, built with the same deterministic
        // permutation helper: shuffling "a\nb\nc\n" repeats moves the
        // newlines too, so both sides tokenize to ~30k lines over a tiny
        // line alphabet: almost no anchorable unique records, the same
        // superlinear wall the char-level ladder measures, reached at a
        // fraction of the corpus size.
        let base = "a\nb\nc\n".repeat(10_000);
        let b = char_permutation(&base);
        let err = diff_opcodes_lines_deadline(&base, &b, Some(5.0)).unwrap_err();
        assert_eq!(err.deadline_ms, 5.0);
        assert!(err.elapsed_ms > 5.0, "elapsed must exceed the budget");
    }

    // --- the similarity surface (difflib's ratio / get_close_matches) ------
    //
    // Every difflib value cited below was cross-checked by computing it
    // with the running stdlib (python3.12 difflib) BEFORE pinning: the
    // rows are derived, not laundered.

    #[test]
    fn ratio_pins_the_forced_alignment_classes() {
        // Identical operands: the fast path, and difflib's 1.0.
        assert_eq!(similarity_ratio("abc", "abc"), 1.0);
        assert_eq!(similarity_ratio("café", "café"), 1.0);
        // The empty pair: difflib's own answer is 1.0 (the _calculate_ratio
        // zero-total guard, difflib.py:39-42, probe-verified), not 0.0
        // and not a panic.
        assert_eq!(similarity_ratio("", ""), 1.0);
        // Empty vs nonempty: M = 0, T > 0 → 0.0 (difflib agrees).
        assert_eq!(similarity_ratio("", "abc"), 0.0);
        assert_eq!(similarity_ratio("abc", ""), 0.0);
        // Disjoint alphabets: no equal op exists → 0.0, forced for both
        // algorithms.
        assert_eq!(similarity_ratio("abc", "xyz"), 0.0);
        // Pure insert with differing flanks: the alignment is forced,
        // M = 3, T = 9 → 2/3 (difflib: 0.6666666666666666,
        // probe-verified), and the delete-class twin is the same number.
        assert_eq!(similarity_ratio("abc", "abcXYZ"), 2.0 * 3.0 / 9.0);
        assert_eq!(similarity_ratio("abcXYZ", "abc"), 2.0 * 3.0 / 9.0);
        // The difflib docstring's own example:
        // SequenceMatcher(None, "abcd", "bcde").ratio() == 0.75.
        assert_eq!(similarity_ratio("abcd", "bcde"), 0.75);
    }

    #[test]
    fn ratio_agrees_with_difflib_where_only_the_match_placement_differs() {
        // "a" vs "baa" is a PINNED opcode boundary case above (difflib
        // anchors the middle 'a' and splits the insertion around it;
        // Myers slides one contiguous insertion to the side), but the
        // matched TOTAL is forced to 1 either way, so both algorithms'
        // ratio is 0.5 (probe-verified): placement divergence, score
        // agreement.
        assert_eq!(similarity_ratio("a", "baa"), 0.5);
        assert_eq!(similarity_ratio("baa", "a"), 0.5);
    }

    #[test]
    fn the_ratio_boundary_rows_where_difflibs_anchored_m_diverges() {
        // The divergence rows, the ratio twin of the opcode boundary pins:
        // difflib's M is anchoring-dependent, ours is the Myers equal-op
        // total; both valid, both ratios in [0,1], the NUMBERS differ.
        //
        // "ppp" vs "pwpp": difflib anchors the earliest "pp" at
        // a[0..2] ↔ b[2..4] and starves the flanks (M = 2 → 4/7 ≈ 0.5714,
        // probe-verified); our pinned opcodes above match 1 + 2 = 3
        // (M = 3 → 6/7 ≈ 0.8571): the Myers one-insert script keeps the
        // whole of "ppp".
        assert_eq!(similarity_ratio("ppp", "pwpp"), 6.0 / 7.0);
        assert_eq!(similarity_ratio("pwpp", "ppp"), 6.0 / 7.0);
        // "qpqpq" vs "qpwqpq": difflib anchors "qpq" at a[0..3] ↔ b[3..6]
        // and starves both flanks (M = 3 → 6/11 ≈ 0.5455,
        // probe-verified); prefix/suffix trimming leaves a one-insert
        // middle, so Myers matches the whole of a (M = 5 → 10/11 ≈
        // 0.9091).
        assert_eq!(similarity_ratio("qpqpq", "qpwqpq"), 10.0 / 11.0);
        assert_eq!(similarity_ratio("qpwqpq", "qpqpq"), 10.0 / 11.0);
    }

    #[test]
    fn every_generated_pair_has_a_bounded_symmetric_identity_iff_equal_ratio() {
        // The ratio twin of the every-pair opcode battery: over all pairs
        // of strings on {'a', 'b'} up to length 4 (625 pairs): the ratio
        // stays in [0, 1], is symmetric, is exactly 1.0 iff the operands
        // are equal, and (the contract tie) decomposes as 2.0*M/T with M
        // the equal-op length sum of diff_opcodes' OWN output on the pair.
        let mut strings = Vec::new();
        for len in 0..=4 {
            let mut bucket = Vec::new();
            all_strings(len, &mut String::new(), &mut bucket);
            strings.extend(bucket);
        }
        for a in &strings {
            for b in &strings {
                let r = similarity_ratio(a, b);
                assert!((0.0..=1.0).contains(&r), "{a:?} vs {b:?}: {r}");
                assert_eq!(r, similarity_ratio(b, a), "asymmetric: {a:?} vs {b:?}");
                assert_eq!(r == 1.0, a == b, "1.0 iff equal: {a:?} vs {b:?}");
                let m: usize = diff_opcodes(a, b)
                    .into_iter()
                    .map(|op| {
                        if op.tag == OpcodeTag::Equal {
                            op.i2 - op.i1
                        } else {
                            0
                        }
                    })
                    .sum();
                let t = a.chars().count() + b.chars().count();
                assert_eq!(
                    r,
                    ratio_from_matches(m, t),
                    "not the diff_opcodes matched total: {a:?} vs {b:?}"
                );
            }
        }
    }

    #[test]
    fn the_hook_count_is_the_capture_sum_everywhere_the_batteries_reach() {
        // The differential gate on the counting-hook spelling of M: the OLD
        // spelled-out computation (materialize the full Vec<DiffOp> from
        // capture_diff_slices_deadline, sum the equal lengths) must equal
        // matched_chars' hook count on every input: the two paths drive the
        // SAME myers::diff_deadline on the same operands, and the capture
        // path's Compact/Replace presentation hooks only merge/shift ops,
        // which cannot change the equal-length total. Asserted over the
        // exhaustive two-letter-alphabet all-pairs sweep AND the golden rows
        // (the opcode boundary pins, the ratio parity/divergence rows, the
        // non-ASCII pairs, and an identical pair through the fast path).
        let capture_m = |a: &str, b: &str| {
            let old: Vec<char> = a.chars().collect();
            let new: Vec<char> = b.chars().collect();
            similar::capture_diff_slices_deadline(similar::Algorithm::Myers, &old, &new, None)
                .into_iter()
                .map(|op| match op {
                    similar::DiffOp::Equal { len, .. } => len,
                    _ => 0,
                })
                .sum::<usize>()
        };
        let mut strings = Vec::new();
        for len in 0..=4 {
            let mut bucket = Vec::new();
            all_strings(len, &mut String::new(), &mut bucket);
            strings.extend(bucket);
        }
        for a in &strings {
            for b in &strings {
                assert_eq!(
                    matched_chars(a, b, None),
                    capture_m(a, b),
                    "hook vs capture M divergence: {a:?} vs {b:?}"
                );
            }
        }
        let golden = [
            ("a", "baa"),
            ("baa", "a"),
            ("ppp", "pwpp"),
            ("pwpp", "ppp"),
            ("pXpp", "ppp"),
            ("qpqpq", "qpwqpq"),
            ("qpwqpq", "qpqpq"),
            ("abcd", "bcde"),
            ("abc", "abcXYZ"),
            ("abcXYZ", "abc"),
            ("abc", "xyz"),
            ("", "abc"),
            ("abc", ""),
            ("", ""),
            ("abXcd", "abYcd"),
            ("cafe\u{301}", "cafe"),
            ("égal", "egal"),
            ("café", "café"),
        ];
        for (a, b) in golden {
            assert_eq!(
                matched_chars(a, b, None),
                capture_m(a, b),
                "hook vs capture M divergence: {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn close_matches_pins_the_difflib_docstring_example() {
        // get_close_matches("appel", ["ape", "apple", "peach", "puppy"]) →
        // ['apple', 'ape'] (the stdlib's own docstring): apple scores 0.8,
        // ape 0.75, peach and puppy 0.4; under the default 0.6 cutoff only
        // the first two survive, best-first. We return the INDICES: apple
        // is 1, ape is 0.
        let candidates = ["ape", "apple", "peach", "puppy"];
        assert_eq!(
            close_matches("appel", &candidates, 3, 0.6, None),
            Ok(vec![1, 0])
        );
        // n truncates best-first: the single best is apple.
        assert_eq!(
            close_matches("appel", &candidates, 1, 0.6, None),
            Ok(vec![1])
        );
    }

    #[test]
    fn close_matches_breaks_score_ties_by_candidate_descending() {
        // difflib's ACTUAL tie rule (difflib.py:707 appends (ratio, x);
        // difflib.py:710 takes heapq.nlargest over the pairs, documented as
        // sorted(iterable, reverse=True)[:n]): the whole tuple compares, so
        // equal scores break by the candidate itself DESCENDING, NOT input
        // order. Probe: get_close_matches("ab", ["ac", "ca"], 2, 0.5) →
        // ['ca', 'ac']: "ca" wins the tie despite being second.
        let candidates = ["ac", "ca"];
        assert_eq!(
            close_matches("ab", &candidates, 2, 0.5, None),
            Ok(vec![1, 0])
        );
        // Exact duplicates (equal score AND equal string) keep input order
        // : the stability of the descending sort. Probe:
        // get_close_matches("ab", ["ac", "ac"], 2, 0.5) → ['ac', 'ac'].
        let duplicates = ["ac", "ac"];
        assert_eq!(
            close_matches("ab", &duplicates, 2, 0.5, None),
            Ok(vec![0, 1])
        );
        // A mixed row, probe-verified end-to-end: "abcd" vs its three
        // one-tail-char variants all tie at 0.75 (each an M = 3 replace of
        // the last char); the largest string sorts first:
        // get_close_matches("abcd", ["abcd", "abce", "abcf", "abcg"], 2,
        // 0.7) → ['abcd', 'abcg'].
        let variants = ["abcd", "abce", "abcf", "abcg"];
        assert_eq!(
            close_matches("abcd", &variants, 2, 0.7, None),
            Ok(vec![0, 3])
        );
    }

    #[test]
    fn close_matches_keeps_scores_equal_to_the_cutoff() {
        // The gate is >= (difflib.py:706): "ab" vs "ac" scores exactly 0.5,
        // which a 0.5 cutoff keeps and a 0.51 cutoff drops.
        let candidates = ["ac"];
        assert_eq!(close_matches("ab", &candidates, 3, 0.5, None), Ok(vec![0]));
        assert_eq!(
            close_matches("ab", &candidates, 3, 0.51, None),
            Ok(Vec::<usize>::new())
        );
        // The empty-word row, probe-verified: ratio("", "") is 1.0 (the
        // zero-total guard), so the empty candidate clears a 0.6 cutoff
        // while "a" scores 0.0: get_close_matches("", ["", "a"], 3, 0.6)
        // → [''].
        let empty_row = ["", "a"];
        assert_eq!(close_matches("", &empty_row, 3, 0.6, None), Ok(vec![0]));
    }

    #[test]
    fn a_none_ratio_deadline_is_the_plain_ratio() {
        // The additive-parameter contract, ratio spelling.
        let a = "abcabcabcXdefdefdef";
        let b = "abcabcabcYdefdefdef";
        assert_eq!(
            similarity_ratio_deadline(a, b, None),
            Ok(similarity_ratio(a, b))
        );
    }

    #[test]
    fn a_generous_or_saturated_ratio_budget_changes_nothing() {
        let a = "the quick brown fox";
        let b = "the quick brown dog";
        assert_eq!(
            similarity_ratio_deadline(a, b, Some(60_000.0)),
            Ok(similarity_ratio(a, b))
        );
        // The shared saturating budget helper: an out-of-range finite
        // budget is unbounded, exactly as in the opcode spellings.
        assert_eq!(
            similarity_ratio_deadline(a, b, Some(1e300)),
            Ok(similarity_ratio(a, b))
        );
    }

    #[test]
    fn an_expired_ratio_deadline_reports_both_numbers_and_discards() {
        // The hard shape, the same class as the opcode deadline test: a
        // character permutation has no anchorable unique records, so the
        // unbounded search costs far more than the 5ms budget.
        let base: String = "abc".repeat(20_000);
        let a = base.as_str();
        let b = char_permutation(a);
        let err = similarity_ratio_deadline(a, &b, Some(5.0)).unwrap_err();
        assert_eq!(err.deadline_ms, 5.0);
        assert!(err.elapsed_ms > 5.0, "elapsed must exceed the budget");
        // Identical inputs under a deadline: the fast path, no search,
        // always inside the budget: 1.0, including the empty pair's 1.0.
        assert_eq!(similarity_ratio_deadline(a, a, Some(5_000.0)), Ok(1.0));
        assert_eq!(similarity_ratio_deadline("", "", Some(5_000.0)), Ok(1.0));
    }

    #[test]
    fn an_expired_close_matches_deadline_discards_the_partial_results() {
        // ONE budget bounds the whole scan: the identical first candidate
        // scores 1.0 through the fast path (kept even at cutoff 0.0); the
        // hard second candidate's search blows the budget: the scan
        // STOPS, the partial list is discarded, and DeadlineExceeded is
        // returned (a truncated close-matches list would look like a real
        // answer).
        let base: String = "abc".repeat(20_000);
        let hard = char_permutation(&base);
        let candidates = [base.as_str(), hard.as_str()];
        let err = close_matches(&base, &candidates, 3, 0.0, Some(5.0)).unwrap_err();
        assert_eq!(err.deadline_ms, 5.0);
        assert!(err.elapsed_ms > 5.0, "elapsed must exceed the budget");
        // And the budget actually bounded the whole call, hard pair and all
        // (loose bound for a loaded runner).
        let started = Instant::now();
        assert!(close_matches(&base, &candidates, 3, 0.0, Some(5.0)).is_err());
        assert!(started.elapsed().as_secs_f64() < 10.0);
    }

    #[test]
    fn a_generous_close_matches_budget_changes_nothing() {
        let candidates = ["ape", "apple", "peach", "puppy"];
        assert_eq!(
            close_matches("appel", &candidates, 3, 0.6, Some(60_000.0)),
            close_matches("appel", &candidates, 3, 0.6, None)
        );
    }

    // --- the close-matches prefilter (the provable length-ratio skip) ---
    //
    // Three gates: the differential (the prefiltered scan equals the
    // UNFILTERED spelling everywhere, exact index lists), the seam count
    // (the skip actually fires), and the strictness boundary (a candidate
    // at exactly the ceiling is still searched, and can still qualify).

    /// The UNFILTERED spelling `close_matches` had before the prefilter:
    /// every candidate searched and scored, the identical gate, the shared
    /// [`best_first`] tail. Kept test-side so the differential battery
    /// isolates the prefilter's effect exactly: any disagreement between
    /// the two spellings is the prefilter and nothing else.
    fn close_matches_unfiltered(
        word: &str,
        candidates: &[&str],
        n: usize,
        cutoff: f64,
    ) -> Vec<usize> {
        let word_len = word.chars().count();
        let mut scored: Vec<(f64, usize)> = Vec::new();
        for (idx, candidate) in candidates.iter().enumerate() {
            let matches = matched_chars(word, candidate, None);
            let score = ratio_from_matches(matches, word_len + candidate.chars().count());
            if score >= cutoff {
                scored.push((score, idx));
            }
        }
        best_first(scored, candidates, n)
    }

    /// A deterministic xorshift string over {'a', 'b', 'c'} (the same
    /// shift triple the `char_permutation` helper uses), seeded per call
    /// so the word, the candidates, and distinct lengths all get distinct
    /// strings.
    fn xorshift_string(len: usize, seed: u64) -> String {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                match state % 3 {
                    0 => 'a',
                    1 => 'b',
                    _ => 'c',
                }
            })
            .collect()
    }

    #[test]
    fn the_prefiltered_scan_equals_the_unfiltered_spelling_over_a_generated_battery() {
        // The load-bearing differential gate on the prefilter: word
        // lengths 3 through 60 against candidate lists holding one
        // candidate of every length 1 through 80 (deterministic
        // three-letter-alphabet strings), at cutoffs {0.4, 0.6, 0.8} and
        // both n regimes (difflib's default 3 and the whole-list
        // no-truncation n). The battery saturates under-ceiling rows in
        // BOTH directions (short candidates under long words, long
        // candidates under short words) and includes exact ceiling ==
        // cutoff boundaries (word 3 vs candidate 12 at cutoff 0.4, word
        // 60 vs candidate 15 at 0.4: both 2*3/15 and 30/75 are exactly
        // 0.4), where the strictness of the skip rule is the whole
        // question. Exact index-list equality.
        for word_len in 3..=60 {
            let word = xorshift_string(word_len, 0x9E37_79B9_7F4A_7C15);
            let candidates: Vec<String> = (1..=80)
                .map(|len| xorshift_string(len, 0x0123_4567_89AB_CDEF ^ (len as u64)))
                .collect();
            let refs: Vec<&str> = candidates.iter().map(String::as_str).collect();
            for cutoff in [0.4, 0.6, 0.8] {
                for n in [3, refs.len()] {
                    assert_eq!(
                        close_matches(&word, &refs, n, cutoff, None),
                        Ok(close_matches_unfiltered(&word, &refs, n, cutoff)),
                        "prefilter disagreement at word_len {word_len}, cutoff {cutoff}, n {n}"
                    );
                }
            }
        }
    }

    #[test]
    fn under_ceiling_candidates_are_skipped_without_a_search() {
        // The seam-measured pin that the skip actually fires: the counter
        // tallies the matched_chars calls close_matches issues, so a
        // battery mixing over-ceiling and impossible-length candidates
        // must search EXACTLY the over-ceiling ones. "zqx" against a
        // 40-char word has ceiling 6/43 ≈ 0.14 and the empty candidate
        // 0.0, both under every cutoff in use; the word itself and a
        // second 40-char string have ceiling 1.0.
        search_seam::reset();
        let word = xorshift_string(40, 0x9E37_79B9_7F4A_7C15);
        let other = xorshift_string(40, 0x2545_F491_4F6C_DD1D);
        let candidates = [word.as_str(), "zqx", "", other.as_str()];
        let filtered = close_matches(&word, &candidates, 4, 0.4, None);
        assert_eq!(
            search_seam::counted(),
            2,
            "only the two 40-char (ceiling 1.0) candidates may be searched"
        );
        // And what survived is exactly what the unfiltered spelling
        // returns: the skip changed nothing about the answer.
        assert_eq!(
            filtered,
            Ok(close_matches_unfiltered(&word, &candidates, 4, 0.4))
        );
    }

    #[test]
    fn a_candidate_at_exactly_the_ceiling_is_searched_and_can_qualify() {
        // The strictness boundary of the skip rule: "abc" (3 chars) vs a
        // 12-char candidate containing it has ceiling 2*3/15 = 0.4
        // EXACTLY, and M = 3 (the whole word matched) realizes ratio ==
        // 0.4, which the >= gate (difflib.py:706) keeps at a 0.4 cutoff.
        // An off-by-one skip rule (skipping at equality) would silently
        // drop this candidate; the seam confirms the search happened.
        let word = "abc";
        let candidates = ["abczzzzzzzzz"]; // "abc" + nine 'z's = 12 chars
        search_seam::reset();
        assert_eq!(close_matches(word, &candidates, 3, 0.4, None), Ok(vec![0]));
        assert_eq!(
            search_seam::counted(),
            1,
            "the at-ceiling candidate must be searched"
        );
        // A hair over the line and the SAME candidate is skipped (ceiling
        // 0.4 < 0.41), its would-be score 0.4 failing the gate anyway, so
        // the counter must not move.
        assert_eq!(
            close_matches(word, &candidates, 3, 0.41, None),
            Ok(Vec::<usize>::new())
        );
        assert_eq!(
            search_seam::counted(),
            1,
            "the under-ceiling re-run must be skipped"
        );
    }
}
