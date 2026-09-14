//! `scrub_pii` never panics on an arbitrary string, and its output is
//! EXACTLY the pipeline the contract describes — asserted as a
//! full-output differential, the strongest shape:
//!
//! * The oracle side of this harness transcribes the two quoted
//!   grammars plus the domestic extension (char-space, per-position,
//!   the regex engine's own order of operations — try every start,
//!   greedy runs, backtracked split/final-digit — spelled nothing like
//!   the byte scanners the transform drives) with its own Nd table,
//!   then computes the expected output by running the transform's own
//!   pipeline order: the email pass over the input, then the phone
//!   grammar over the email pass's result, token breakers and all. The
//!   transform's output must be byte-identical at both salts, for the
//!   BOTH-rules pipeline and for phone-only. This subsumes every
//!   accounting corner the earlier survivor-counting shape could not
//!   express: an email pass eating a phone-shaped local part whole
//!   (`440..1.0III0@…` consuming `…440..1`), an email token's verbatim
//!   domain carrying a phone shape the phone pass cannot scrub (glued
//!   to hex-alphabet letters), and the digest-hex reconstruction
//!   corners (a token's hex tail is local-part material, so a token
//!   followed by `@`-shaped text can re-spell an eaten match) — all of
//!   those are exact output, not counted absence.
//! * Convergence, structurally: the second pass over the differential
//!   output is a fixed point (a third pass is the identity), and
//!   phone-only is strictly idempotent.
//! * The identity path never lies: a borrowed return is exactly the
//!   expected output being the input (no match existed on either side
//!   of the differential).
//!
//! The oracle's digests are spelled independently (sha2 + const-hex,
//! the same hand-synced-to-root pin discipline the normalize target's
//! oracle uses) so the token construction is a transcription too, not
//! a call into the code under test.

#![no_main]

use std::borrow::Cow;

use libfuzzer_sys::fuzz_target;
use tors::pii_impl::{PiiRules, scrub_pii};

/// Independent Nd table for the oracle side of this harness: an
/// own transcription of Unicode 16.0.0 Nd ranges, deliberately NOT
/// imported from `pii_impl::is_nd` (which it checks). If the two
/// tables ever disagree, the phone-match invariants below fail and
/// the harness — not the transform — names the drift.
const ORACLE_ND_RANGES: [(u32, u32); 71] = [
    (0x0030, 0x0039),
    (0x0660, 0x0669),
    (0x06F0, 0x06F9),
    (0x07C0, 0x07C9),
    (0x0966, 0x096F),
    (0x09E6, 0x09EF),
    (0x0A66, 0x0A6F),
    (0x0AE6, 0x0AEF),
    (0x0B66, 0x0B6F),
    (0x0BE6, 0x0BEF),
    (0x0C66, 0x0C6F),
    (0x0CE6, 0x0CEF),
    (0x0D66, 0x0D6F),
    (0x0DE6, 0x0DEF),
    (0x0E50, 0x0E59),
    (0x0ED0, 0x0ED9),
    (0x0F20, 0x0F29),
    (0x1040, 0x1049),
    (0x1090, 0x1099),
    (0x17E0, 0x17E9),
    (0x1810, 0x1819),
    (0x1946, 0x194F),
    (0x19D0, 0x19D9),
    (0x1A80, 0x1A89),
    (0x1A90, 0x1A99),
    (0x1B50, 0x1B59),
    (0x1BB0, 0x1BB9),
    (0x1C40, 0x1C49),
    (0x1C50, 0x1C59),
    (0xA620, 0xA629),
    (0xA8D0, 0xA8D9),
    (0xA900, 0xA909),
    (0xA9D0, 0xA9D9),
    (0xA9F0, 0xA9F9),
    (0xAA50, 0xAA59),
    (0xABF0, 0xABF9),
    (0xFF10, 0xFF19),
    (0x0104A0, 0x0104A9),
    (0x010D30, 0x010D39),
    (0x010D40, 0x010D49),
    (0x011066, 0x01106F),
    (0x0110F0, 0x0110F9),
    (0x011136, 0x01113F),
    (0x0111D0, 0x0111D9),
    (0x0112F0, 0x0112F9),
    (0x011450, 0x011459),
    (0x0114D0, 0x0114D9),
    (0x011650, 0x011659),
    (0x0116C0, 0x0116C9),
    (0x0116D0, 0x0116E3),
    (0x011730, 0x011739),
    (0x0118E0, 0x0118E9),
    (0x011950, 0x011959),
    (0x011BF0, 0x011BF9),
    (0x011C50, 0x011C59),
    (0x011D50, 0x011D59),
    (0x011DA0, 0x011DA9),
    (0x011F50, 0x011F59),
    (0x016130, 0x016139),
    (0x016A60, 0x016A69),
    (0x016AC0, 0x016AC9),
    (0x016B50, 0x016B59),
    (0x016D70, 0x016D79),
    (0x01CCF0, 0x01CCF9),
    (0x01D7CE, 0x01D7FF),
    (0x01E140, 0x01E149),
    (0x01E2F0, 0x01E2F9),
    (0x01E4F0, 0x01E4F9),
    (0x01E5F1, 0x01E5FA),
    (0x01E950, 0x01E959),
    (0x01FBF0, 0x01FBF9),
];

#[inline]
fn oracle_is_nd(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    let cp = c as u32;
    ORACLE_ND_RANGES
        .binary_search_by(|&(lo, hi)| {
            if cp < lo {
                std::cmp::Ordering::Greater
            } else if cp > hi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

fn is_local_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '%' | '+' | '-')
}

fn is_domain_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '-')
}

fn is_phone_class_char(c: char) -> bool {
    oracle_is_nd(c) || matches!(c, '-' | '.' | ' ' | '(' | ')')
}

fn is_phone_sep_char(c: char) -> bool {
    matches!(c, '-' | '.' | ' ' | '(' | ')')
}

fn is_token_hex(c: char) -> bool {
    c.is_ascii_digit() || matches!(c, 'a'..='f')
}

/// A token span in char space: `~` + 12 digest hex, exactly as the
/// transform's byte-level `token_span_end_at` defines it (a prefix, a
/// `~`, twelve lowercase-hex digest chars). Returns the span END (char
/// index, exclusive).
fn token_span_end_at_chars(chars: &[char], tilde: usize) -> Option<usize> {
    if chars.get(tilde) != Some(&'~') {
        return None;
    }
    const TOKEN_HEX: usize = 12;
    let end = tilde + 1 + TOKEN_HEX;
    if end > chars.len() {
        return None;
    }
    if chars[tilde + 1..end].iter().all(|&c| is_token_hex(c)) {
        Some(end)
    } else {
        None
    }
}

/// The token span covering `off` (strictly inside), or `None` — the
/// char-space twin of the transform's `token_span_containing`: a run
/// starting inside a digest is digest tail, not a number's head.
fn token_span_containing_chars(chars: &[char], off: usize) -> Option<usize> {
    const TOKEN_HEX: usize = 12;
    let lo = off.saturating_sub(TOKEN_HEX);
    let hi = off.min(chars.len().saturating_sub(1));
    for tilde in lo..=hi {
        if chars.get(tilde) != Some(&'~') {
            continue;
        }
        if let Some(end) = token_span_end_at_chars(chars, tilde)
            && tilde <= off
            && off < end
        {
            return Some(end);
        }
    }
    None
}

/// Whether a token span ends exactly at `pos` — the char-space twin of
/// the transform's `token_ends_at`: the byte after a finished token is
/// a clean boundary.
fn token_ends_at_chars(chars: &[char], pos: usize) -> bool {
    const TOKEN_HEX: usize = 12;
    pos > TOKEN_HEX
        && chars.get(pos - 1 - TOKEN_HEX) == Some(&'~')
        && chars[pos - TOKEN_HEX..pos].iter().all(|&c| is_token_hex(c))
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

/// The INTERNATIONAL phone grammar at one start position: a literal `+`,
/// an Nd digit, then the greedy class run backtracked to its last Nd
/// digit at char index 6 or beyond (the middle holds at least six
/// chars). Returns the match END (char index) on success.
fn intl_match_at(chars: &[char], start: usize) -> Option<usize> {
    if chars[start] != '+' || start + 1 >= chars.len() || !oracle_is_nd(chars[start + 1]) {
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
        .find(|&idx| oracle_is_nd(chars[idx]))
        .map(|idx| idx + 1)
}

/// The full phone match set: the international grammar at every `+`,
/// and the domestic extension over un-plussed runs — a FULL class run
/// of exactly ten Nd digits, or eleven with an ASCII leading `1`,
/// carrying at least one separator inside the match span (the bare-run
/// cut), never behind a `+` (international territory), leading spaces
/// skipped — with the transform's token-span semantics transcribed
/// exactly: the scan steps over `~` + 12-hex token spans (a run never
/// forms from digest material, so a number following a token keeps its
/// own clean run and matches), and a match starting exactly at a
/// token's end is clean (the finished token is a word boundary). Char
/// indices, never bytes: the grammars must agree on units with each
/// other (never with the transform's byte offsets).
fn phone_matches_of(s: &str) -> Vec<(usize, usize)> {
    let chars: Vec<char> = s.chars().collect();
    let mut found = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        // A token head at the cursor: step over the whole span.
        if chars[i] == '~'
            && let Some(end) = token_span_end_at_chars(&chars, i)
        {
            i = end;
            continue;
        }
        if chars[i] == '+' {
            if let Some(end) = intl_match_at(&chars, i) {
                found.push((i, end));
                i = end;
                continue;
            }
            i += 1;
            continue;
        }
        if !is_phone_class_char(chars[i]) {
            i += 1;
            continue;
        }
        // A run starting inside a token digest is digest tail: resume
        // after the token so the composed run never forms and the real
        // number following it keeps its own clean run.
        if let Some(end) = token_span_containing_chars(&chars, i) {
            i = end;
            continue;
        }
        // A class run starting at i (never at a `+`: those broke above).
        let run_start = i;
        let mut j = i;
        let mut digits = 0;
        let mut first_digit: Option<char> = None;
        let mut last_digit_idx = None;
        let mut seen_non_space = false;
        let mut leading_spaces = 0;
        while j < chars.len() && is_phone_class_char(chars[j]) {
            if oracle_is_nd(chars[j]) {
                digits += 1;
                if first_digit.is_none() {
                    first_digit = Some(chars[j]);
                }
                last_digit_idx = Some(j);
            }
            if !seen_non_space {
                if chars[j] == ' ' {
                    leading_spaces += 1;
                } else {
                    seen_non_space = true;
                }
            }
            j += 1;
        }
        let plussed = run_start > 0 && chars[run_start - 1] == '+';
        if !plussed {
            let nanp = digits == 10 || (digits == 11 && first_digit == Some('1'));
            if nanp {
                let start = run_start + leading_spaces;
                let end = last_digit_idx.unwrap() + 1;
                // The clean-boundary rule, exactly as the transform
                // spells it: the match's first char not glued to a `~`
                // or a lowercase a-f (the token-interior alphabet) —
                // UNLESS the match starts exactly where a token span
                // ends (the finished token is a word boundary).
                let clean = start == 0
                    || !matches!(chars[start - 1], '~' | 'a'..='f')
                    || token_ends_at_chars(&chars, start);
                if clean && chars[start..end].iter().any(|&c| is_phone_sep_char(c)) {
                    found.push((start, end));
                    i = end;
                    continue;
                }
            }
        }
        i = j; // a non-matching run is spent whole: no partial matches
    }
    found
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

/// The token digest, spelled independently of the transform: sha256
/// (salt || matched) truncated to 12 lowercase hex chars, via sha2 +
/// const-hex directly (the same hand-synced pin discipline as the
/// normalize target's oracle) — the transform routes through its own
/// token_digest, so the construction is a transcription too.
fn token_digest(salt: &str, matched: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(salt.as_bytes());
    hasher.update(matched.as_bytes());
    let digest = hasher.finalize();
    const_hex::encode(&digest.as_slice()[..6])
}

/// Substitute every span (leftmost, non-overlapping, in order) with the
/// token the token_of closure emits for it, char-space.
fn substitute(
    chars: &[char],
    spans: &[(usize, usize)],
    token_of: impl Fn(&str) -> String,
) -> String {
    let mut out = String::with_capacity(chars.len());
    let mut pos = 0;
    for &(start, end) in spans {
        if start > pos {
            out.extend(chars[pos..start].iter());
        }
        let matched: String = chars[start..end].iter().collect();
        out.push_str(&token_of(&matched));
        pos = end;
    }
    out.extend(chars[pos..].iter());
    out
}

/// The email token for a matched address: `@domain~digest`, the domain
/// being the match's own domain portion verbatim (the single `@` in
/// the span splits local from domain — the local class excludes `@`).
fn email_token(salt: &str, matched: &str) -> String {
    let at = matched
        .find('@')
        .expect("an email match holds exactly one @");
    let domain = &matched[at + 1..];
    format!("@{domain}~{}", token_digest(salt, matched))
}

/// The phone token for a matched number: the match's first three code
/// points, `~`, the digest.
fn phone_token(salt: &str, matched: &str) -> String {
    let prefix: String = matched.chars().take(3).collect();
    format!("{prefix}~{}", token_digest(salt, matched))
}

/// The expected BOTH-rules output: the email pass over the input, then
/// the phone grammar over the email pass's result — the transform's own
/// pipeline order, token breakers and all (the phone match set is
/// computed on the intermediate exactly as the transform does).
fn expected_both(s: &str, salt: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let emails = matches_of(s, email_match_at);
    let mid = substitute(&chars, &emails, |m| email_token(salt, m));
    let mid_chars: Vec<char> = mid.chars().collect();
    let phones = phone_matches_of(&mid);
    substitute(&mid_chars, &phones, |m| phone_token(salt, m))
}

/// The expected phone-only output: the phone grammar over the raw
/// input, no email pass.
fn expected_phone_only(s: &str, salt: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let phones = phone_matches_of(s);
    substitute(&chars, &phones, |m| phone_token(salt, m))
}

fuzz_target!(|s: &str| {
    let phone_only = PiiRules {
        email: false,
        phone: true,
    };
    for salt in ["", tors::pii_impl::DEFAULT_SALT] {
        // The full-output differential: the transcription's own pipeline
        // (email pass, then the phone grammar over the email output,
        // token breakers and all) must produce byte-identical output to
        // the transform, at both salts. This subsumes survivor
        // accounting: email-eaten phone shapes, domain-borne phone
        // shapes the phone pass cannot scrub, and the digest-hex
        // reconstruction corners are all exact output, not counted
        // absence. A disagreement here is either a transform bug or an
        // oracle drift — the panic names the input either way.
        let out = scrub_pii(s, PiiRules::BOTH, salt);
        assert_eq!(
            out.as_ref(),
            &expected_both(s, salt),
            "the BOTH-rules pipeline diverged from the oracle at salt {salt:?}"
        );

        // Convergence, structurally: the second pass over the
        // differential output is a fixed point (a third pass is the
        // identity).
        let twice = scrub_pii(out.as_ref(), PiiRules::BOTH, salt);
        assert!(
            matches!(
                scrub_pii(twice.as_ref(), PiiRules::BOTH, salt),
                Cow::Borrowed(_)
            ),
            "the converged output is not a fixed point on {s:?}"
        );

        // Phone-only: the phone grammar over the raw input, and the
        // pass is strictly idempotent.
        let once = scrub_pii(s, phone_only, salt);
        assert_eq!(
            once.as_ref(),
            &expected_phone_only(s, salt),
            "the phone-only pipeline diverged from the oracle at salt {salt:?}"
        );
        assert!(
            matches!(scrub_pii(once.as_ref(), phone_only, salt), Cow::Borrowed(_)),
            "phone-only is not idempotent on {s:?}"
        );
    }
});
