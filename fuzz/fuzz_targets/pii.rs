//! `scrub_pii` never panics on an arbitrary string, and redaction
//! completeness holds in the exact shape the contract allows:
//!
//! * No phone match of the input survives verbatim in the pass-one
//!   output — unconditionally: every token carries a `~` (and email
//!   tokens an `@` head) that breaks digit runs, and unmatched text
//!   cannot hold a full run (the scan is total), so a surviving match
//!   string would itself be a phone match of the output, and none can
//!   be constructed across a token boundary.
//! * Email-match strings CAN reappear in the pass-one output — by
//!   reconstruction, not by survival: a token's digest hex is
//!   local-part material, so a token immediately followed by `@`-shaped
//!   text can spell out an input match by coincidence (the hex suffix
//!   plays the local part). Two reachable shapes: two ADJACENT email
//!   matches (their tokens land back-to-back), and a match whose end is
//!   followed by an unmatched `@`-run whose own local part the match
//!   consumed. Both are exactly why the surface documents re-scrubbing:
//!   pass two eats every `@` that local-part material can reach.
//! * After the second pass (the documented convergence point) NO match
//!   of the input survives, absolutely: every `@` in the converged
//!   output is preceded by a non-local character or has no valid domain
//!   (else pass two would have fired on it), so the converged output
//!   holds no email match at all — and no phone match either, because a
//!   replacement can never remove a digit-run breaker without inserting
//!   one (`@` or `~`) in its place. The completeness invariant is
//!   asserted there, in its absolute form, alongside the structural
//!   convergence claim (a third pass is the identity).
//! * The identity path never lies: a borrowed return implies no match
//!   existed (a match that fired without allocating is silent
//!   under-redaction by definition).
//! * Phone-only is strictly idempotent, and its pass-one output is
//!   unconditionally free of input phone-match survivors (no email pass
//!   runs, so the reconstruction shape cannot arise).
//!
//! The completeness checks need a matcher, and the transform is not one,
//! so this target carries its own: a char-space, per-position
//! transcription of the two quoted grammars (the regex engine's own
//! order of operations — try every start, greedy runs, backtracked
//! split/final-digit — spelled nothing like the byte scanners the
//! transform drives). Agreement between the two spellings is exactly
//! what the invariants assert. Both reachable reconstruction shapes
//! were found by exactly this harness (an independent re-derivation of
//! the body over corpus plus 500k deterministic random compositions)
//! before the target ever ran under libFuzzer.

#![no_main]

use std::borrow::Cow;

use libfuzzer_sys::fuzz_target;
use tors::pii_impl::{PiiRules, is_nd, scrub_pii};

fn is_local_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '%' | '+' | '-')
}

fn is_domain_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '-')
}

fn is_phone_class_char(c: char) -> bool {
    is_nd(c) || matches!(c, '-' | '.' | ' ' | '(' | ')')
}

/// The email grammar at one start position: the maximal local run, a
/// literal `@`, then the domain's greedy split (longest middle first,
/// the largest dot with a two-plus-letter tail inside the maximal
/// domain run). Returns the match END (char index) on success.
fn email_match_at(chars: &[char], start: usize) -> Option<usize> {
    let mut i = start;
    while i < chars.len() && is_local_char(chars[i]) {
        i += 1;
    }
    if i == start || i >= chars.len() || chars[i] != '@' {
        return None;
    }
    let run_start = i + 1;
    let mut run_end = run_start;
    while run_end < chars.len() && is_domain_char(chars[run_end]) {
        run_end += 1;
    }
    let run = &chars[run_start..run_end];
    for len in (1..run.len()).rev() {
        if run[len] == '.' {
            let mut j = len + 1;
            while j < run.len() && run[j].is_ascii_alphabetic() {
                j += 1;
            }
            if j - len - 1 >= 2 {
                return Some(run_start + j);
            }
        }
    }
    None
}

/// The phone grammar at one start position: a literal `+`, an Nd digit,
/// then the greedy class run backtracked to its last Nd digit at char
/// index 6 or beyond (the middle holds at least six chars). Returns the
/// match END (char index) on success.
fn phone_match_at(chars: &[char], start: usize) -> Option<usize> {
    if chars[start] != '+' || start + 1 >= chars.len() || !is_nd(chars[start + 1]) {
        return None;
    }
    let mut run_end = start + 2;
    while run_end < chars.len() && is_phone_class_char(chars[run_end]) {
        run_end += 1;
    }
    // The final digit sits at the largest index in [start + 8, run_end)
    // holding an Nd char (index start+8 makes the middle exactly six).
    ((start + 8)..run_end)
        .rev()
        .find(|&idx| is_nd(chars[idx]))
        .map(|idx| idx + 1)
}

/// The non-overlapping leftmost match set over `s` for one grammar, in
/// the regex engine's own scanning order. Char indices, not bytes: the
/// phone grammar's Nd digits are multi-byte, and the two grammars must
/// agree on units with each other (never with the transform's byte
/// offsets).
fn matches_of(s: &str, match_at: fn(&[char], usize) -> Option<usize>) -> Vec<(usize, usize)> {
    let chars: Vec<char> = s.chars().collect();
    let mut found = Vec::new();
    let mut pos = 0;
    while pos < chars.len() {
        match match_at(&chars, pos) {
            Some(end) => {
                found.push((pos, end));
                pos = end;
            }
            None => pos += 1,
        }
    }
    found
}

/// Whether any of the input's matches survives verbatim in `out`.
fn any_survivor(s: &str, out: &str, emails: &[(usize, usize)], phones: &[(usize, usize)]) -> bool {
    for (start, end) in emails.iter().chain(phones.iter()) {
        let matched: String = s.chars().skip(*start).take(end - start).collect();
        if out.contains(matched.as_str()) {
            return true;
        }
    }
    false
}

fuzz_target!(|s: &str| {
    let emails = matches_of(s, email_match_at);
    let phones = matches_of(s, phone_match_at);

    for salt in ["", tors::pii_impl::DEFAULT_SALT] {
        let out = scrub_pii(s, PiiRules::BOTH, salt);
        let got = out.as_ref();

        // Pass one: phone matches never survive verbatim (a survivor
        // would be a phone match of the output, and no digit run can be
        // constructed across a token boundary: see the module docs).
        for (start, end) in &phones {
            let matched: String = s.chars().skip(*start).take(end - start).collect();
            assert!(
                !got.contains(matched.as_str()),
                "a phone match of the input survived pass one on {s:?}: {matched:?}"
            );
        }

        // The identity path never lies, in the stronger direction: a
        // borrowed return means NO match existed (a match that fired
        // without allocating is silent under-redaction by definition).
        match out {
            Cow::Borrowed(_) => {
                assert!(
                    emails.is_empty() && phones.is_empty(),
                    "identity return on {s:?} but a match existed"
                );
            }
            Cow::Owned(_) => {}
        }

        // The converged output (pass two, the documented fixed point):
        // no match of the input survives at all, and a third pass is
        // the identity — convergence, pinned structurally rather than
        // by value equality alone.
        let twice = scrub_pii(got, PiiRules::BOTH, salt);
        let twice = twice.as_ref();
        assert!(
            !any_survivor(s, twice, &emails, &phones),
            "a match of the input survived into the converged output on {s:?}"
        );
        assert!(
            matches!(scrub_pii(twice, PiiRules::BOTH, salt), Cow::Borrowed(_)),
            "the converged output is not a fixed point on {s:?}"
        );
    }

    // Phone-only: no email pass runs, so the reconstruction shape
    // cannot arise, no phone-match survivor is possible at all, and the
    // pass is strictly idempotent (a phone token's `~` breaks every
    // digit run three code points in).
    let phone_only = PiiRules {
        email: false,
        phone: true,
    };
    let once = scrub_pii(s, phone_only, "");
    let once = once.as_ref();
    assert!(
        !any_survivor(s, once, &[], &phones),
        "a phone match survived phone-only pass one on {s:?}"
    );
    assert!(
        matches!(scrub_pii(once, phone_only, ""), Cow::Borrowed(_)),
        "phone-only is not idempotent on {s:?}"
    );
});
