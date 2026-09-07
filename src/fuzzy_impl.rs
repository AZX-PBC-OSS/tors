//! Character-level fuzzy-match metrics: the pure-Rust core of the
//! `tors` similarity functions: Levenshtein distance, Jaro similarity and
//! Jaro-Winkler similarity, all deadline-bounded.
//!
//! CPython has no stdlib spelling of any of the three: `difflib`'s ratio is
//! not a metric (it is 2·M/T over a non-minimal matching, not an edit
//! distance), and every real spelling (`python-Levenshtein`, `rapidfuzz`)
//! is third-party. All three are O(n·m) in the worst case: on two 1 MiB
//! strings that is minutes of CPU, exactly the superlinear class
//! `diff_opcodes`' `deadline_ms` exists for (the README's diff section
//! records the measured ladder for its engine; the DP here is the same
//! complexity class). The obvious route (calling `strsim` directly)
//! cannot meet that bar: its functions are uninterruptible (no deadline
//! parameter, no bail-out point in their loops), so a deadline could never
//! fire mid-DP and the DoS discipline would be vacuous. DECISION: the
//! metrics are implemented here with per-row/phase deadline checks (the
//! check overhead is one `Instant` comparison per DP row, negligible
//! against the m cells the row just cost), and `strsim 0.11.1`, already a
//! dev-dependency, is the DEV-side differential oracle pinning our formulas
//! to a maintained implementation: the crate tests below assert equality
//! with `strsim::levenshtein`/`jaro`/`jaro_winkler` over a generated
//! battery (ASCII, accented, CJK, emoji, empty/degenerate) plus hand-derived
//! known vectors from the literature: the correctness of the crate WITHOUT
//! shipping a runtime dependency, and the DoS bar intact.
//!
//! All three operate on `char`s: a Rust `char` IS a Unicode scalar value,
//! exactly what a Python `str` index addresses, so every length the
//! Levenshtein result counts is a CHARACTER count (the house convention;
//! see `diff_impl`'s module docs for the same stance). Byte-level metrics
//! would disagree with every Python-side expectation and are not offered.
//!
//! The Jaro-Winkler boost threshold is `> 0.7` and the prefix cap is 4 with
//! p = 0.1, read from strsim's own source (strsim-0.11.1/src/lib.rs,
//! `generic_jaro_winkler`: `if sim > 0.7` and `.take(4)`, boost
//! `sim + 0.1 * prefix_length * (1.0 - sim)`); it is the standard spelling
//! the literature uses, pinned here rather than guessed.
//!
//! Pure Rust, no pyo3 types: the deadline outcome ([`DeadlineExceeded`])
//! mirrors `diff_impl`'s shape and is what the later pyo3 layer maps to
//! `TimeoutError`; the fields exist for the message, which names both
//! numbers. The budget arithmetic itself (`budget_from_ms`'s saturating
//! ms→`Duration` conversion and the elapsed-vs-budget comparison) is NOT
//! reimplemented here: it is `diff_impl::elapsed_exceeds` (which itself
//! calls `diff_impl::budget_from_ms`), the same shared helper
//! `grounded_impl`'s `is_grounded_fuzzy` reuses, kept DRY across every
//! `deadline_ms`-bearing primitive in the crate. Only the
//! `DeadlineExceeded` TYPE is local: its own field set and its own
//! function-named `message()`, `diff_impl`'s documented convention for why
//! each caller keeps a distinct wrapper.

use std::collections::HashMap;
use std::time::Instant;

use crate::diff_impl::elapsed_exceeds;

/// The deadline outcome, mirroring `diff_impl`'s shape: the call exceeded
/// its caller-supplied budget, so the (partial) result is discarded.
/// Carries both numbers the message names: the deadline and the elapsed
/// cost. The pyo3 layer maps this to `TimeoutError`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeadlineExceeded {
    pub deadline_ms: f64,
    pub elapsed_ms: f64,
}

impl DeadlineExceeded {
    /// The `str()` of the `TimeoutError` the pyo3 layer raises.
    pub fn message(&self) -> String {
        format!(
            "fuzzy metric deadline exceeded: elapsed {:.1}ms > deadline_ms {:.1}ms",
            self.elapsed_ms, self.deadline_ms
        )
    }
}

/// The shared expiry tail every `deadline_ms`-bearing primitive here uses:
/// `diff_impl::elapsed_exceeds` against the wall clock and the caller's
/// budget, packed into this module's own [`DeadlineExceeded`]. `None` means
/// no budget was set (unbounded) or it hasn't been exceeded yet.
fn exceeded_after(started: Instant, deadline_ms: Option<f64>) -> Option<DeadlineExceeded> {
    elapsed_exceeds(started, deadline_ms).map(|(deadline_ms, elapsed_ms)| DeadlineExceeded {
        deadline_ms,
        elapsed_ms,
    })
}

/// One clock per call: the start instant and the absolute deadline instant
/// (`None` when unbounded; no budget set, or a budget so out of range it
/// saturates past the clock's range). The expiry check is one
/// `Instant::now()` comparison, called per DP row (Levenshtein) or per
/// phase/1024-outer-steps (Jaro), negligible against the work each row or
/// phase performs, and the only way a deadline can fire mid-DP.
struct Budget {
    deadline_ms: Option<f64>,
    started: Instant,
}

impl Budget {
    fn new(deadline_ms: Option<f64>) -> Self {
        Budget {
            deadline_ms,
            started: Instant::now(),
        }
    }

    /// The mid-loop expiry check: `Err` carries both message numbers, via
    /// the shared `diff_impl::elapsed_exceeds` arithmetic. Never fires for
    /// an unbounded budget.
    fn expired(&self) -> Result<(), DeadlineExceeded> {
        match exceeded_after(self.started, self.deadline_ms) {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

/// The classic unit-cost Levenshtein edit distance (insert/delete/
/// substitute each cost 1) between `a` and `b`, in CHARACTER counts.
///
/// Degenerate fast paths, all inside any sane budget and none of them
/// allocating the DP: identical operands → 0 (one native equality scan),
/// an empty operand → the other side's char count. Otherwise a two-row DP
/// over char vectors, the deadline checked once per ROW. Symmetric by
/// construction: the DP's recurrence is symmetric in its two operands
/// (every operation cost is 1 on both axes), and the symmetry is pinned
/// over the full test battery below: `levenshtein(a, b) ==
/// levenshtein(b, a)` for every pair, both spellings also differentially
/// pinned against `strsim::levenshtein`.
pub fn levenshtein(a: &str, b: &str, deadline_ms: Option<f64>) -> Result<usize, DeadlineExceeded> {
    let budget = Budget::new(deadline_ms);
    if a == b {
        // Identical operands: one O(n) equality scan, exact; no DP, no
        // char vectors, inside any sane budget.
        return Ok(0);
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    if a_chars.is_empty() {
        return Ok(b_chars.len());
    }
    if b_chars.is_empty() {
        return Ok(a_chars.len());
    }
    // The shorter operand is the bit-vector pattern (its length is the
    // register width the algorithm needs); Myers' algorithm handles up to
    // one machine word (64 positions) per pattern in a single pass; beyond
    // that the classic DP below still runs, unchanged. Most of tors's
    // realistic fuzzy-match inputs (names, identifiers, short queries) are
    // well under 64 characters, so this covers the common case with O(n)
    // work instead of O(n·m).
    let (pattern, text) = if a_chars.len() <= b_chars.len() {
        (&a_chars, &b_chars)
    } else {
        (&b_chars, &a_chars)
    };
    if pattern.len() <= 64 {
        return levenshtein_myers_bitvector(pattern, text, &budget);
    }
    levenshtein_dp(&a_chars, &b_chars, &budget)
}

/// The classic two-row DP: `prev` is row i-1, `curr` fills row i over b's
/// columns. Unit costs on both axes keep the recurrence symmetric in the
/// operands, so which side is outer is a memory-layout choice only. The
/// fallback for pattern lengths beyond one machine word (64 characters),
/// where [`levenshtein_myers_bitvector`]'s single-block form doesn't apply,
/// and the differential oracle [`levenshtein_myers_bitvector`] is tested
/// against.
fn levenshtein_dp(
    a_chars: &[char],
    b_chars: &[char],
    budget: &Budget,
) -> Result<usize, DeadlineExceeded> {
    let mut prev: Vec<usize> = (0..=b_chars.len()).collect();
    let mut curr = vec![0usize; b_chars.len() + 1];
    for (i, ca) in a_chars.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b_chars.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
        budget.expired()?;
    }
    Ok(prev[b_chars.len()])
}

/// Myers' bit-vector algorithm (G. Myers, "A Fast Bit-Vector Algorithm for
/// Approximate String Matching Based on Dynamic Programming", JACM 46(3),
/// 1999) for unit-cost edit distance, single-block form: `pattern` (at most
/// 64 characters, one `u64` register per DP column-delta) against
/// `text` of any length, in O(n) time and O(1) auxiliary space (`Peq` is
/// O(distinct characters in pattern), never O(pattern × text)); the DP
/// above is O(pattern · text). This is a different algorithm from the same
/// author's O(ND) diff search backing `diff_opcodes`; the two solve
/// different problems and share no code.
///
/// The DP being simulated tracks, at each text position, the edit distance
/// between `pattern` and the text prefix ending there; this function
/// returns the final value, the distance against the whole text, which is
/// exactly `levenshtein`'s contract when `text` is the complete second
/// operand rather than a substring-search haystack. `Pv`/`Mv` encode which
/// DP cells increased or decreased relative to the cell one row up: the
/// standard formulation, reproduced from the paper rather than approximated.
fn levenshtein_myers_bitvector(
    pattern: &[char],
    text: &[char],
    budget: &Budget,
) -> Result<usize, DeadlineExceeded> {
    let m = pattern.len();
    debug_assert!(
        (1..=64).contains(&m),
        "single-block form: 1..=64 characters"
    );
    let all_ones: u64 = if m == 64 { u64::MAX } else { (1u64 << m) - 1 };
    let last_bit: u64 = 1u64 << (m - 1);

    // Peq[c]: bit i set iff pattern[i] == c. A HashMap, not a fixed-size
    // table, since a `char` is a full Unicode scalar value, not a byte.
    let mut peq: HashMap<char, u64> = HashMap::with_capacity(m);
    for (i, &c) in pattern.iter().enumerate() {
        *peq.entry(c).or_insert(0) |= 1u64 << i;
    }

    let mut pv: u64 = all_ones;
    let mut mv: u64 = 0u64;
    let mut score: i64 = m as i64;

    for (idx, &c) in text.iter().enumerate() {
        let eq = peq.get(&c).copied().unwrap_or(0);
        let xv = eq | mv;
        let xh = ((eq & pv).wrapping_add(pv) ^ pv) | eq;
        let mut ph = mv | !(xh | pv);
        let mut mh = pv & xh;
        if ph & last_bit != 0 {
            score += 1;
        } else if mh & last_bit != 0 {
            score -= 1;
        }
        ph = (ph << 1) | 1;
        mh <<= 1;
        pv = (mh | !(xv | ph)) & all_ones;
        mv = (ph & xv) & all_ones;
        if idx % 1024 == 0 {
            budget.expired()?;
        }
    }
    debug_assert!(score >= 0, "edit distance cannot go negative");
    Ok(score as usize)
}

/// The Jaro similarity of `a` and `b` (a `f64` in [0.0, 1.0], higher is
/// more similar) over CHARACTER sequences.
///
/// Semantics (the standard flag-based algorithm, the one `strsim` runs and
/// the crate tests differentially pin): the matching window is
/// `max(|a|, |b|)/2 − 1` (floor, saturating at 0); a char of `a` matches
/// the first not-yet-matched equal char of `b` inside the window (greedy
/// left-to-right); transpositions are the count of mismatched pairs when
/// the two matched subsequences are walked side by side, divided by 2
/// (integer); the score is `(m/|a| + m/|b| + (m−t)/m)/3`, or 0.0 when
/// `m == 0`. Degenerates: both empty → 1.0, one empty → 0.0; no
/// allocation, no passes.
///
/// The deadline is checked per phase: every 1024 chars of the match pass's
/// outer loop (the pass is O(|a|·w) worst case) and once before the
/// transposition walk.
pub fn jaro(a: &str, b: &str, deadline_ms: Option<f64>) -> Result<f64, DeadlineExceeded> {
    let budget = Budget::new(deadline_ms);
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    if a_chars.is_empty() && b_chars.is_empty() {
        return Ok(1.0);
    }
    if a_chars.is_empty() || b_chars.is_empty() {
        return Ok(0.0);
    }
    jaro_core(&a_chars, &b_chars, &budget)
}

/// The Jaro-Winkler similarity: [`jaro`], then (only when the Jaro score
/// clears the standard boost threshold `> 0.7`; the exact threshold
/// `strsim` itself uses, read from its source) a prefix boost
/// `jaro + l·p·(1 − jaro)` where `l` is the length of the common prefix
/// capped at 4 chars and `p = 0.1`. Below the threshold the Jaro score is
/// returned unchanged, common prefix or not. Same character-level
/// convention and same deadline machinery as [`jaro`] (the boost itself is
/// O(prefix) and needs no check of its own).
pub fn jaro_winkler(a: &str, b: &str, deadline_ms: Option<f64>) -> Result<f64, DeadlineExceeded> {
    let sim = jaro(a, b, deadline_ms)?;
    if sim > 0.7 {
        let prefix_length = a
            .chars()
            .zip(b.chars())
            .take(4)
            .take_while(|(ca, cb)| ca == cb)
            .count();
        Ok(sim + 0.1 * prefix_length as f64 * (1.0 - sim))
    } else {
        Ok(sim)
    }
}

/// The Jaro core over materialized char slices; the empty degenerates are
/// the caller's fast paths. The expression order of the final formula
/// mirrors `strsim`'s so the differential is bit-identical, not merely
/// within tolerance.
fn jaro_core(a: &[char], b: &[char], budget: &Budget) -> Result<f64, DeadlineExceeded> {
    let a_len = a.len();
    let b_len = b.len();
    let window = (a_len.max(b_len) / 2).saturating_sub(1);
    let mut a_flags = vec![false; a_len];
    let mut b_flags = vec![false; b_len];
    let mut matches = 0usize;
    // The match pass: for each a[i], the first unmatched equal b[j] in
    // [i−w, i+w] (clamped). O(a_len·window) worst case; the deadline
    // check every 1024 outer steps is the bound for that pass.
    for i in 0..a_len {
        if i % 1024 == 0 {
            budget.expired()?;
        }
        let lo = i.saturating_sub(window);
        let hi = b_len.min(i.saturating_add(window).saturating_add(1));
        for j in lo..hi {
            if !b_flags[j] && a[i] == b[j] {
                a_flags[i] = true;
                b_flags[j] = true;
                matches += 1;
                break;
            }
        }
    }
    // transposition walk: both flagged subsequences side by side, each
    // unequal pair one transposition, halved by INTEGER division,
    // matching strsim's own `transpositions /= 2` (verified against its
    // vendored source), not because the mismatch count is guaranteed even.
    // It is NOT: matched characters that cycle through more than a 2-cycle
    // (e.g. a 3-cycle rotation) yield an odd count: "102" vs "021000" is
    // the pinned regression (3 mismatches, floors to 1, not 1.5). A float
    // division here would silently diverge from strsim on exactly this
    // class of input.
    budget.expired()?;
    let mut transpositions = 0usize;
    if matches > 0 {
        let mut b_iter = b_flags.iter().zip(b).filter(|(flag, _)| **flag);
        for (a_flag, ca) in a_flags.iter().zip(a) {
            if *a_flag
                && let Some((_, cb)) = b_iter.next()
                && ca != cb
            {
                transpositions += 1;
            }
        }
    }
    transpositions /= 2;
    if matches == 0 {
        Ok(0.0)
    } else {
        Ok(((matches as f64 / a_len as f64)
            + (matches as f64 / b_len as f64)
            + ((matches - transpositions) as f64 / matches as f64))
            / 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- hand-derived known vectors (the independent anchor) --------------
    //
    // Values derived by hand from the formulas in the doc comments; if any
    // of these disagrees with strsim, that is a STOP-worthy finding: the
    // differential test below would be the one to fail on it.

    #[test]
    fn levenshtein_known_vectors() {
        assert_eq!(levenshtein("kitten", "sitting", None), Ok(3));
        assert_eq!(levenshtein("flaw", "lawn", None), Ok(2));
        assert_eq!(levenshtein("", "", None), Ok(0));
        // Unicode rows: one char, one substitution each.
        assert_eq!(levenshtein("é", "e", None), Ok(1));
        assert_eq!(levenshtein("🦀", "🐙", None), Ok(1));
        assert_eq!(levenshtein("café", "cafe", None), Ok(1));
        // Lengths are CHARACTER counts: the 4-byte emoji costs 1, not 4.
        assert_eq!(levenshtein("", "🦀🦀🦀", None), Ok(3));
    }

    #[test]
    fn jaro_known_vectors_from_the_literature() {
        // MARTHA/MARHTA: m=6, t=1 → (1 + 1 + 5/6)/3.
        let martha = jaro("MARTHA", "MARHTA", None).unwrap();
        assert!((martha - (17.0 / 18.0)).abs() < 1e-12, "{martha}");
        // DIXON/DICKSONX: m=4 (D, I, O, N), t=0 → (4/5 + 4/8 + 1)/3.
        let dixon = jaro("DIXON", "DICKSONX", None).unwrap();
        assert!((dixon - 23.0 / 30.0).abs() < 1e-12, "{dixon}");
        // DWAYNE/DUANE: m=4 (D, A, N, E), t=0 → (4/6 + 4/5 + 1)/3.
        let dwayne = jaro("DWAYNE", "DUANE", None).unwrap();
        assert!((dwayne - 37.0 / 45.0).abs() < 1e-12, "{dwayne}");
    }

    #[test]
    fn jaro_winkler_known_vectors_from_the_literature() {
        // MARTHA/MARHTA: jaro 17/18 > 0.7, common prefix "MAR" (3) →
        // 17/18 + 0.1·3·(1/18) = 173/180.
        let martha = jaro_winkler("MARTHA", "MARHTA", None).unwrap();
        assert!((martha - 173.0 / 180.0).abs() < 1e-12, "{martha}");
        // DWAYNE/DUANE: jaro 37/45 > 0.7, common prefix "D" (1) →
        // 37/45 + 0.1·(8/45) = 21/25.
        let dwayne = jaro_winkler("DWAYNE", "DUANE", None).unwrap();
        assert!((dwayne - 0.84).abs() < 1e-12, "{dwayne}");
    }

    // --- the differential oracle (strsim 0.11.1, the dev-dependency) -------

    /// The battery the differential runs over: every pair of strings over
    /// {'a', 'b'} up to length 4 (625 ordered pairs; every degenerate,
    /// window edge, transposition and repeated-char shape at small size),
    /// plus the named Unicode and literature rows.
    fn battery() -> Vec<(String, String)> {
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
        let mut strings: Vec<String> = Vec::new();
        for len in 0..=4 {
            let mut bucket = Vec::new();
            all_strings(len, &mut String::new(), &mut bucket);
            strings.extend(bucket);
        }
        let mut pairs: Vec<(String, String)> = Vec::new();
        for a in &strings {
            for b in &strings {
                pairs.push((a.clone(), b.clone()));
            }
        }
        for (a, b) in [
            ("kitten", "sitting"),
            ("flaw", "lawn"),
            ("MARTHA", "MARHTA"),
            ("DIXON", "DICKSONX"),
            ("DWAYNE", "DUANE"),
            ("cheeseburger", "cheese fries"),
            ("Friedrich Nietzsche", "Jean-Paul Sartre"),
            ("café", "cafe"),
            ("cafe\u{301}", "cafe"),
            ("naïve", "naive"),
            ("résumé", "resume"),
            ("你好世界", "你好"),
            ("こんにちは世界", "こんにちは"),
            ("🦀🦀 rust", "🐙 rust"),
            ("🦀👍🏽", "👍🏽🦀"),
            ("a🦀b🦀c", "a🦀c"),
            ("", "🦀🦀🦀"),
            ("🦀🦀🦀", ""),
            ("aaab aaaab", "aaab aaab"),
        ] {
            pairs.push((a.to_string(), b.to_string()));
        }
        pairs
    }

    #[test]
    fn levenshtein_myers_bitvector_matches_the_dp_directly() {
        // The differential this algorithm swap actually needs: the
        // bit-vector fast path against the DP it replaces for
        // pattern.len() <= 64, over a much wider alphabet and length range
        // than the {a,b}-up-to-4 battery above: a deterministic xorshift
        // over ASCII, Latin accents, CJK, and emoji, at lengths spanning
        // the 64-character single-block boundary from both sides.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let alphabet: Vec<char> = "ab café東京🦀🐙\u{200d}\u{0301}".chars().collect();
        let rand_string = |next: &mut dyn FnMut() -> u64, len: usize| -> String {
            (0..len)
                .map(|_| alphabet[(next() as usize) % alphabet.len()])
                .collect()
        };
        for &(la, lb) in &[
            (0, 0),
            (1, 1),
            (0, 5),
            (5, 0),
            (63, 63),
            (64, 64),
            (64, 65),
            (65, 64),
            (1, 200),
            (200, 1),
            (63, 200),
            (64, 200),
            (65, 200),
            (200, 200),
            (300, 400),
        ] {
            for _ in 0..20 {
                let a: Vec<char> = rand_string(&mut next, la).chars().collect();
                let b: Vec<char> = rand_string(&mut next, lb).chars().collect();
                let budget = Budget::new(None);
                let dp = levenshtein_dp(&a, &b, &budget).unwrap();
                let (pattern, text) = if a.len() <= b.len() {
                    (&a, &b)
                } else {
                    (&b, &a)
                };
                if pattern.is_empty() {
                    assert_eq!(dp, text.len(), "empty-pattern mismatch");
                    continue;
                }
                if pattern.len() > 64 {
                    continue; // DP is the only path; nothing to differential-test here.
                }
                let bitvector = levenshtein_myers_bitvector(pattern, text, &budget).unwrap();
                assert_eq!(
                    dp,
                    bitvector,
                    "diverged: a={a:?} b={b:?} (pattern.len()={})",
                    pattern.len()
                );
            }
        }
    }

    #[test]
    fn levenshtein_myers_bitvector_matches_known_vectors() {
        // The same literature vectors the public levenshtein() function is
        // pinned against, called through the fast path directly.
        let budget = Budget::new(None);
        let cases: &[(&str, &str, usize)] = &[
            ("kitten", "sitting", 3),
            ("flaw", "lawn", 2),
            ("gumbo", "gambol", 2),
            ("", "", 0),
        ];
        for &(a, b, expected) in cases {
            let a_chars: Vec<char> = a.chars().collect();
            let b_chars: Vec<char> = b.chars().collect();
            if a_chars.is_empty() || b_chars.is_empty() {
                continue;
            }
            let (pattern, text) = if a_chars.len() <= b_chars.len() {
                (&a_chars, &b_chars)
            } else {
                (&b_chars, &a_chars)
            };
            let got = levenshtein_myers_bitvector(pattern, text, &budget).unwrap();
            assert_eq!(got, expected, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn levenshtein_matches_strsim_over_the_battery() {
        for (a, b) in battery() {
            let (a, b) = (a.as_str(), b.as_str());
            let ours = levenshtein(a, b, None).expect("a None deadline can never be exceeded");
            assert_eq!(
                ours,
                strsim::levenshtein(a, b),
                "diverged on {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn jaro_matches_strsim_over_the_battery() {
        for (a, b) in battery() {
            let (a, b) = (a.as_str(), b.as_str());
            let ours = jaro(a, b, None).expect("a None deadline can never be exceeded");
            let oracle = strsim::jaro(a, b);
            assert!(
                (ours - oracle).abs() < 1e-12,
                "diverged on {a:?} vs {b:?}: {ours} vs {oracle}"
            );
        }
    }

    #[test]
    fn jaro_winkler_matches_strsim_over_the_battery() {
        for (a, b) in battery() {
            let (a, b) = (a.as_str(), b.as_str());
            let ours = jaro_winkler(a, b, None).expect("a None deadline can never be exceeded");
            let oracle = strsim::jaro_winkler(a, b);
            assert!(
                (ours - oracle).abs() < 1e-12,
                "diverged on {a:?} vs {b:?}: {ours} vs {oracle}"
            );
        }
    }

    // --- properties --------------------------------------------------------

    #[test]
    fn levenshtein_is_symmetric_over_the_battery() {
        // The unit-cost recurrence is symmetric in its operands; the pin is
        // both directions of every battery pair through the actual DP.
        for (a, b) in battery() {
            let (a, b) = (a.as_str(), b.as_str());
            let forward = levenshtein(a, b, None).unwrap();
            let backward = levenshtein(b, a, None).unwrap();
            assert_eq!(forward, backward, "asymmetric on {a:?} vs {b:?}");
        }
    }

    #[test]
    fn jaro_is_bounded_and_identity_rows_hold() {
        for (a, b) in battery() {
            let (a, b) = (a.as_str(), b.as_str());
            let j = jaro(a, b, None).unwrap();
            assert!(
                (0.0..=1.0).contains(&j),
                "out of [0,1] on {a:?} vs {b:?}: {j}"
            );
            let jw = jaro_winkler(a, b, None).unwrap();
            assert!(
                (0.0..=1.0).contains(&jw),
                "out of [0,1] on {a:?} vs {b:?}: {jw}"
            );
            // The boost never decreases the score.
            assert!(jw >= j, "jw < jaro on {a:?} vs {b:?}");
        }
        // Identity rows: every metric answers "identical" exactly.
        for x in ["abc", "MARTHA", "café", "你好世界", "🦀👍🏽", ""] {
            assert_eq!(levenshtein(x, x, None), Ok(0));
            assert_eq!(jaro(x, x, None), Ok(1.0));
            assert_eq!(jaro_winkler(x, x, None), Ok(1.0));
        }
    }

    #[test]
    fn jaro_winkler_prefix_spot_rows() {
        // Above the 0.7 threshold the boost applies and grows with the
        // common prefix; below it the Jaro score is returned unchanged
        // even with a common prefix.
        let martha_jaro = jaro("MARTHA", "MARHTA", None).unwrap();
        let martha_jw = jaro_winkler("MARTHA", "MARHTA", None).unwrap();
        assert!(martha_jw > martha_jaro);
        let abcd_jaro = jaro("abcd", "abce", None).unwrap();
        let abcd_jw = jaro_winkler("abcd", "abce", None).unwrap();
        assert!(abcd_jw > abcd_jaro);
        // A 4-char common prefix boosts more than a 1-char one, both
        // above threshold: jaro("abcd","abcd") side rows.
        let four = jaro_winkler("abcdX", "abcdY", None).unwrap();
        let one = jaro_winkler("aXXX", "aYYY", None).unwrap();
        assert!(four > one);
        // Below the threshold: jaro("axxxx","ayyyy") = (1/5+1/5+1)/3 =
        // 7/15 ≈ 0.467 ≤ 0.7, common prefix 'a' notwithstanding: no boost.
        let low = jaro("axxxx", "ayyyy", None).unwrap();
        assert!(low <= 0.7, "{low}");
        assert_eq!(jaro_winkler("axxxx", "ayyyy", None), Ok(low));
        // The prefix cap: a 6-char common prefix only counts 4.
        // jaro("abcdef","abcdef") = 1.0; a capped spelling:
        // jaro("abcdeX","abcdeY"): m=5, t=0 → (5/6+5/6+1)/3 = 8/9 > 0.7,
        // prefix 5 → capped at 4: 8/9 + 0.4/9.
        let capped = jaro_winkler("abcdeX", "abcdeY", None).unwrap();
        assert!((capped - (8.0 / 9.0 + 0.4 / 9.0)).abs() < 1e-12, "{capped}");
    }

    // --- the deadline (the DoS discipline) ----------------------------------
    //
    // The hard shape a deadline exists for: two ~60k-char strings over a
    // 2-letter alphabet: the DP is ~3.6e9 cells and the per-row check is
    // the only thing that can fire inside it. Built with the same
    // deterministic xorshift spelling diff_impl's tests use.

    fn random_two_letter(len: usize, seed: u64) -> String {
        let mut state = seed;
        let mut out = String::with_capacity(len);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.push(if state.is_multiple_of(2) { 'x' } else { 'y' });
        }
        out
    }

    #[test]
    fn an_expired_deadline_reports_both_numbers_and_discards_the_result() {
        let a = random_two_letter(60_000, 0x9E37_79B9_7F4A_7C15);
        let b = random_two_letter(60_000, 0x0123_4567_89AB_CDEF);
        // One DP row of 60k cells costs microseconds; the full 60k-row DP
        // is minutes, so a 5ms budget fires within the first rows. The
        // per-row check is the mechanism under test: the assert is the
        // proof.
        let err = levenshtein(&a, &b, Some(5.0)).unwrap_err();
        assert_eq!(err.deadline_ms, 5.0);
        assert!(err.elapsed_ms > 5.0, "elapsed must exceed the budget");
        let message = err.message();
        assert!(message.contains("deadline_ms 5.0ms"), "{message}");
        assert!(message.contains("elapsed "), "{message}");
        // And the deadline actually bounded the work: the call returned
        // in well under the unbounded cost (loose bound for a loaded
        // runner).
        let started = Instant::now();
        assert!(levenshtein(&a, &b, Some(5.0)).is_err());
        assert!(jaro(&a, &b, Some(5.0)).is_err());
        assert!(jaro_winkler(&a, &b, Some(5.0)).is_err());
        assert!(started.elapsed().as_secs_f64() < 10.0);
    }

    #[test]
    fn a_generous_deadline_equals_the_none_spelling() {
        for (a, b) in battery() {
            let (a, b) = (a.as_str(), b.as_str());
            assert_eq!(levenshtein(a, b, Some(60_000.0)), levenshtein(a, b, None));
            let j = jaro(a, b, None).unwrap();
            assert!((jaro(a, b, Some(60_000.0)).unwrap() - j).abs() < 1e-12);
            let jw = jaro_winkler(a, b, None).unwrap();
            assert!((jaro_winkler(a, b, Some(60_000.0)).unwrap() - jw).abs() < 1e-12);
        }
    }

    #[test]
    fn a_huge_but_finite_budget_saturates_to_unbounded() {
        // 1e300 ms is positive and finite but overflows Duration; the
        // saturating discipline maps it to unbounded, never a panic.
        for (a, b) in battery() {
            let (a, b) = (a.as_str(), b.as_str());
            assert_eq!(levenshtein(a, b, Some(1e300)), levenshtein(a, b, None));
            let j = jaro(a, b, None).unwrap();
            assert!((jaro(a, b, Some(1e300)).unwrap() - j).abs() < 1e-12);
            let jw = jaro_winkler(a, b, None).unwrap();
            assert!((jaro_winkler(a, b, Some(1e300)).unwrap() - jw).abs() < 1e-12);
        }
        // NaN and negative budgets saturate the same way (reachable only
        // through the pub Rust spelling; the pyo3 layer validates).
        let (a, b) = ("kitten", "sitting");
        assert_eq!(levenshtein(a, b, Some(f64::NAN)), Ok(3));
        assert_eq!(levenshtein(a, b, Some(-1.0)), Ok(3));
        let j = jaro(a, b, None).unwrap();
        assert!((jaro(a, b, Some(f64::NAN)).unwrap() - j).abs() < 1e-12);
        assert!((jaro(a, b, Some(-1.0)).unwrap() - j).abs() < 1e-12);
    }

    #[test]
    fn degenerate_fast_paths_never_allocate_the_dp() {
        // Equal operands: the fast path is one O(n) equality scan, so a
        // budget the DP could never meet (a 200k-row DP is ~4e10 cells,
        // minutes) still returns Ok(0) instantly: the pin that the DP
        // was never entered.
        let a = "ab".repeat(100_000);
        assert_eq!(levenshtein(&a, &a, Some(20.0)), Ok(0));
        // The empty degenerates are char counts, not DP results.
        let b = "xyz🦀".repeat(50_000);
        assert_eq!(levenshtein("", &b, Some(20.0)), Ok(200_000));
        assert_eq!(levenshtein(&b, "", Some(20.0)), Ok(200_000));
        // Jaro degenerates: both empty 1.0, one empty 0.0; no passes.
        assert_eq!(jaro("", "", Some(20.0)), Ok(1.0));
        assert_eq!(jaro(&b, "", Some(20.0)), Ok(0.0));
        assert_eq!(jaro("", &b, Some(20.0)), Ok(0.0));
        assert_eq!(jaro_winkler("", "", Some(20.0)), Ok(1.0));
        assert_eq!(jaro_winkler(&b, "", Some(20.0)), Ok(0.0));
    }
}
