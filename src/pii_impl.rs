//! Contact-material scrub, the pure-Rust core of `tors.scrub_pii`.
//!
//! Replaces email addresses and `+`-led phone numbers inside free text
//! with correlation tokens: the scrub an error excerpt, rejection
//! message, or response-body excerpt needs before it reaches telemetry,
//! because telemetry is the one store a data purge cannot reach — a
//! candidate scrubbed from the primary store must not leave their
//! address behind in log retention. This is a port of a private
//! consumer's telemetry-safety module, pinned byte-identical to it at
//! `salt=""` (the quoted-pin oracle in `tests/reference.py` is the
//! transcription; `tests/test_scrub_pii_parity.py` is the differential).
//!
//! The contract, exactly as the source states it:
//!
//! * **Email rule** — the pattern
//!   `[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}`: a deliberately
//!   permissive local part (this is a redactor, so over-matching costs a
//!   token where a literal string would have read fine, while
//!   under-matching leaks an address), a domain of ASCII letters, digits,
//!   dots, and hyphens, and a two-plus-letter ASCII tail after the last
//!   dot the greedy backtracking can reach — which is why `a@b.co9`
//!   scrubs `a@b.co` and leaves the `9`, and `a@b.c.d` (a one-letter
//!   tail) never matches at all. Domains are kept verbatim in the token,
//!   leading and doubled dots included (`a@.b.co` → `@.b.co~…`), because
//!   the domain is the coarse, non-identifying half an operator reasons
//!   about. No special treatment for URLs, `mailto:` links, or code
//!   spans: whatever email sits inside them matches.
//! * **Phone rule** — the pattern `\+\d[\d\-. ()]{6,}\d`: anchored on a
//!   literal `+` (a bare digit run is an order number, byte count, or
//!   timestamp, and redacting those would destroy the diagnostic this
//!   scrubber exists to preserve), at least eight digits total with the
//!   separators humans and upstream APIs spell numbers with, and the
//!   digit class in the Python-regex sense: every Unicode Nd decimal
//!   digit (Arabic-Indic, Devanagari, fullwidth, …), not ASCII-only. A
//!   run of class characters between two numbers is ONE match
//!   (`+4712345678 1234567890` scrubs as a unit), the match ends at the
//!   run's last digit (trailing separators survive), and a `~` or a
//!   second `+` breaks it.
//! * **Pass order** — email substitution over the whole string first,
//!   then phone substitution over its result, each exactly once, no
//!   cascade: an email's local part may itself contain a `+`-led digit
//!   run (`user+14155552671@example.com` is one email), so the phone
//!   rule must see the email tokens, never the addresses that produced
//!   them.
//! * **Tokens** — `@domain~<digest>` for an email match, and
//!   `prefix~<digest>` for a phone match, where `prefix` is the match's
//!   first three CODE POINTS (a canonical E.164's country code — `"+47"`
//!   compact, `"+1 "` for a domestic spelling where the third code point
//!   is the space) and `<digest>` is the first 12 hex chars of
//!   `sha256(salt + match)`. A token is a correlation handle, not a
//!   secret: the E.164 space is small enough to enumerate, so the digest
//!   lets an operator tie two log lines to the same number without the
//!   record holding the number, nothing more.
//!
//! Two behaviors documented rather than hidden:
//!
//! * **Idempotence, as it true is.** Tokens are individually fixed
//!   points, and the output is a fixed point unless an email token is
//!   immediately followed by `@`-shaped text — the one re-fire corner,
//!   reachable in two shapes: two ADJACENT email matches (the first
//!   ends exactly where the second's local part begins, e.g.
//!   `a@b.co9@x.yz`, so the tokens land back-to-back), and a match
//!   immediately followed by an unmatched `@`-run whose own local part
//!   the match consumed (e.g. `x@b.co@w.vu`, where the second `@` never
//!   matched on its own). The corner exists because a token's digest
//!   hex is local-part material, so pass two sees a fresh match there
//!   (the hex plays the local part) and fires once more — and that
//!   second output is a fixed point, always: every `@` in it is
//!   preceded by a non-local character or holds no valid domain (else
//!   pass two would have fired on it), so scrubbing twice always
//!   converges. `tests/test_scrub_pii.py` pins both shapes literally
//!   and the fuzz target asserts convergence over arbitrary input.
//!   Phone-only is strictly idempotent (a phone token's `~` breaks
//!   every digit run three code points in), and a phone token can
//!   never land flush against an email token because the email pass's
//!   greedy local consumption guarantees a non-class character before
//!   every email token while a phone match ends on a digit, which is
//!   class.
//! * **The salt.** `DEFAULT_SALT` is tors's own constant — the source
//!   chain digests unsalted, and re-publishing that as a default would
//!   re-publish its documented weakness (an enumerated E.164 space
//!   confirms a candidate list against unsalted digests). The default is
//!   a fixed, non-secret, versioned domain-separation tag, frozen
//!   because changing it would silently change every deployment's token
//!   values; `salt=""` is the unsalted spelling (byte-identical tokens
//!   with the source chain, the migration lane), and deployments that
//!   care pass their own secret. A KNOWN salt — the public default
//!   included — still leaves candidate-list confirmation possible: the
//!   tokens are redaction, not pseudonymization crypto.
//!
//! No new dependency: hand-rolled `memchr`-anchored scanners over the
//! input bytes (the crate's charter cuts regex engines at runtime; see
//! `docs/design.md`'s scope cuts), one linear pass per rule (the domain
//! split is a single backward sweep with a running letter count, never a
//! quadratic rescan), `sha2` for the digests, and a committed Nd range
//! table (Unicode 16.0.0, the same UCD the crate's other tables pin, and
//! the UCD CPython 3.14's own `re` module matches on). Allocating only
//! when a rule actually fires: `tors.scrub_pii(s, ...) is s` exactly
//! when no active rule matches (the crate's `Cow` identity convention).
//!
//! The failure mode this module exists to prevent is SILENT
//! UNDER-REDACTION: where the source is permissive, this port is
//! permissive; every documented non-match of the source (`+`-less digit
//! runs, one-letter TLDs, non-ASCII domains, a separator right after
//! the `+`) is a pinned non-match here, never tightened.

use std::borrow::Cow;

use memchr::memchr;
use sha2::{Digest, Sha256};

/// tors's documented default digest salt. A fixed, non-secret,
/// versioned domain-separation tag, frozen: changing it would silently
/// change every deployment's token values. Mirrored in
/// `tests/reference.py` (`SCRUB_PII_DEFAULT_SALT`), which the salt=None
/// differential lane pins byte-equal to this constant.
pub const DEFAULT_SALT: &str = "tors/scrub_pii/v1";

/// The digest half of every token: 12 lowercase hex chars.
const TOKEN_HEX: usize = 12;

/// The Unicode Nd (decimal digit) codepoint ranges, Unicode 16.0.0 —
/// the same UCD the crate's normalization/segmentation tables pin, and
/// the UCD CPython 3.14's own `re` digit class matches on. This is the
/// exact set the source chain's `\d` accepts (a Unicode digit is an Nd
/// codepoint: superscripts and fractions are No, Roman numerals Nl, and
/// neither matches). Nd assignments are append-only across Unicode
/// versions, so this table also covers every older interpreter's digit
/// set. Sorted, and binary-searched below.
const ND_RANGES: [(u32, u32); 71] = [
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

/// Whether `c` is a Unicode Nd decimal digit (the `re` `\d` class). The
/// ASCII range is the hot path and answered directly; everything else is
/// a binary search over the committed table.
#[inline]
pub fn is_nd(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    let cp = c as u32;
    ND_RANGES
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

/// The email local-part class: `[A-Za-z0-9._%+\-]`. ASCII only, so the
/// scanner walks raw bytes.
#[inline]
fn is_local_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'%' | b'+' | b'-')
}

/// The email domain class: `[A-Za-z0-9.\-]`. ASCII only.
#[inline]
fn is_domain_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-')
}

#[inline]
fn is_letter_byte(b: u8) -> bool {
    b.is_ascii_alphabetic()
}

/// The phone separator class beyond digits: `[\d\-. ()]` minus the digit
/// half. Non-ASCII digits make this a char-level check, not bytewise.
#[inline]
fn is_phone_separator(c: char) -> bool {
    matches!(c, '-' | '.' | ' ' | '(' | ')')
}

/// The digest half of a token: the first `TOKEN_HEX` hex chars of
/// `sha256(salt || match)`. `salt=""` is the source chain's unsalted
/// digest exactly, which is why the salt is a plain concatenation (any
/// separator would break that identity).
fn token_digest(salt: &str, matched: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(salt.as_bytes());
    hasher.update(matched.as_bytes());
    let digest = hasher.finalize();
    const_hex::encode(&digest.as_slice()[..TOKEN_HEX / 2])
}

/// The email pass: every leftmost match of the email pattern becomes
/// `@domain~digest`. One linear scan anchored on `@` (memchr), with the
/// local part walked back over class bytes and the domain's greedy
/// split found in a single backward sweep that carries the ASCII-letter
/// run length after each candidate dot — the exact backtracking order
/// of the regex (longest middle first), without its quadratic rescan.
/// `Cow::Borrowed` when nothing matches.
fn scrub_email_pass<'a>(text: &'a str, salt: &str) -> Cow<'a, str> {
    let bytes = text.as_bytes();
    let mut pos = 0; // where the search for the next `@` resumes
    let mut emitted = 0; // the prefix of `text` already pushed to `out`
    let mut out: Option<String> = None;
    while let Some(rel) = memchr(b'@', &bytes[pos..]) {
        let at = pos + rel;
        // The local part: the maximal run of class bytes immediately
        // before `@`, clipped at the previous match's end (a new match
        // can never start inside a consumed span). The regex engine
        // would try every start position and fail on the same run; the
        // clip reproduces the leftmost surviving start.
        let mut local_start = at;
        while local_start > emitted && is_local_byte(bytes[local_start - 1]) {
            local_start -= 1;
        }
        if local_start == at {
            pos = at + 1;
            continue; // no local part at this `@`
        }
        // The domain: the maximal run of domain-class bytes after `@`.
        // A dot right after the run is impossible (a dot IS a class
        // byte), so every candidate split dot sits inside it.
        let dom_start = at + 1;
        let mut dom_end = dom_start;
        while dom_end < bytes.len() && is_domain_byte(bytes[dom_end]) {
            dom_end += 1;
        }
        let r = &bytes[dom_start..dom_end];
        // The greedy split, longest middle first: the largest `len` in
        // 1..r.len() with `r[len] == '.'` and at least two ASCII letters
        // at `len + 1`. One backward sweep: `letter_run` holds the
        // consecutive-letter count starting at `len + 1` for the `len`
        // being tried, then folds `r[len]` in for the next round.
        let mut match_end = None;
        let mut letter_run = 0; // letters starting at r.len() (off-array)
        for len in (1..r.len()).rev() {
            if r[len] == b'.' && letter_run >= 2 {
                // The match ends after the letter run following the dot.
                match_end = Some(dom_start + len + 1 + letter_run);
                break;
            }
            letter_run = if is_letter_byte(r[len]) {
                letter_run + 1
            } else {
                0
            };
        }
        let Some(end) = match_end else {
            pos = at + 1;
            continue; // no valid split at this `@`: a documented non-match
        };
        // Emit: the untouched span up to the match, then the token. The
        // domain field is the match's own domain portion, verbatim.
        let matched = &text[local_start..end];
        let domain = &text[at + 1..end];
        let token = format!("@{domain}~{}", token_digest(salt, matched));
        let buf = out.get_or_insert_with(|| String::with_capacity(text.len()));
        buf.push_str(&text[emitted..local_start]);
        buf.push_str(&token);
        emitted = end;
        pos = end;
    }
    match out {
        None => Cow::Borrowed(text),
        Some(mut buf) => {
            buf.push_str(&text[emitted..]);
            Cow::Owned(buf)
        }
    }
}

/// The phone pass: every leftmost match of the phone pattern becomes
/// `prefix~digest` (`prefix` = the match's first three code points, the
/// E.164 dialling prefix). One linear scan anchored on `+` (memchr); the
/// class run after the leading digit is walked once, recording the last
/// Nd digit at char index 6 or beyond — the exact landing point of the
/// regex's greedy `{6,}` backtracking. `Cow::Borrowed` when nothing
/// matches.
fn scrub_phone_pass<'a>(text: &'a str, salt: &str) -> Cow<'a, str> {
    let bytes = text.as_bytes();
    let mut pos = 0;
    let mut emitted = 0;
    let mut out: Option<String> = None;
    while let Some(rel) = memchr(b'+', &bytes[pos..]) {
        let plus = pos + rel;
        // The leading digit: any Nd char (ASCII or not) directly after
        // the `+`; anything else is a documented non-match at this `+`.
        let Some(d1) = text[plus + 1..].chars().next() else {
            break; // the `+` is the last byte: nothing more to find
        };
        if !is_nd(d1) {
            pos = plus + 1;
            continue;
        }
        // The class run after the leading digit, walked once: the run's
        // end and the last qualifying digit land together (a digit at
        // char index >= 6 is a match end; later digits overwrite, so the
        // largest index wins, the backtracking's first success).
        let run_start = plus + 1 + d1.len_utf8();
        let mut offset = run_start;
        let mut best_end: Option<usize> = None;
        for (char_idx, c) in text[run_start..].chars().enumerate() {
            let nd = is_nd(c);
            if !nd && !is_phone_separator(c) {
                break;
            }
            if char_idx >= 6 && nd {
                best_end = Some(offset + c.len_utf8());
            }
            offset += c.len_utf8();
        }
        let Some(end) = best_end else {
            pos = plus + 1;
            continue; // fewer than 8 digits in the run: a non-match here
        };
        // The token prefix: the match's first three code points. The
        // match holds at least nine (`+`, the leading digit, six middle
        // chars, the final digit), so three always exist.
        let mut prefix_end = plus;
        for _ in 0..3 {
            prefix_end += text[prefix_end..].chars().next().unwrap().len_utf8();
        }
        let matched = &text[plus..end];
        let token = format!(
            "{}~{}",
            &text[plus..prefix_end],
            token_digest(salt, matched)
        );
        let buf = out.get_or_insert_with(|| String::with_capacity(text.len()));
        buf.push_str(&text[emitted..plus]);
        buf.push_str(&token);
        emitted = end;
        pos = end;
    }
    match out {
        None => Cow::Borrowed(text),
        Some(mut buf) => {
            buf.push_str(&text[emitted..]);
            Cow::Owned(buf)
        }
    }
}

/// Which rules a call applies. `scrub_pii` with neither rule is the
/// identity (the caller's `rules=[]`).
#[derive(Clone, Copy)]
pub struct PiiRules {
    pub email: bool,
    pub phone: bool,
}

impl PiiRules {
    /// The default `rules=None`: both rules, the source chain's canonical
    /// set.
    pub const BOTH: PiiRules = PiiRules {
        email: true,
        phone: true,
    };
}

/// Scrub `text` of contact material: the email substitution over the
/// whole string, then the phone substitution over its result (the
/// source chain's canonical order — each rule exactly once, no
/// cascade), or whichever subset `rules` selects. `Cow::Borrowed` — the
/// identity path — exactly when no active rule matches.
pub fn scrub_pii<'a>(text: &'a str, rules: PiiRules, salt: &str) -> Cow<'a, str> {
    let after_email = if rules.email {
        scrub_email_pass(text, salt)
    } else {
        Cow::Borrowed(text)
    };
    if !rules.phone {
        return after_email;
    }
    match after_email {
        Cow::Borrowed(t) => scrub_phone_pass(t, salt),
        // The email pass already allocated, so the phone pass's borrow
        // of the intermediate folds back into it: fired, its own output;
        // unfired, the intermediate itself. Neither branch copies.
        Cow::Owned(mid) => match scrub_phone_pass(&mid, salt) {
            Cow::Borrowed(_) => Cow::Owned(mid),
            Cow::Owned(fin) => Cow::Owned(fin),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scrub(text: &str, rules: PiiRules, salt: &str) -> String {
        scrub_pii(text, rules, salt).into_owned()
    }

    fn digest(salt: &str, matched: &str) -> String {
        token_digest(salt, matched)
    }

    #[test]
    fn clean_input_is_identity() {
        let text = "plain prose, café, emoji \u{1f600}, digits 4096 and 1200";
        assert!(matches!(
            scrub_pii(text, PiiRules::BOTH, DEFAULT_SALT),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn empty_rules_is_identity_even_with_contacts() {
        let text = "a@b.co +14155552671";
        assert!(matches!(
            scrub_pii(
                text,
                PiiRules {
                    email: false,
                    phone: false
                },
                ""
            ),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn email_tokens_keep_the_domain_verbatim() {
        let rules = PiiRules {
            email: true,
            phone: false,
        };
        assert_eq!(
            scrub("a@b.co", rules, ""),
            format!("@b.co~{}", digest("", "a@b.co"))
        );
        assert_eq!(
            scrub("a@.b.co", rules, ""),
            format!("@.b.co~{}", digest("", "a@.b.co"))
        );
        assert_eq!(
            scrub("A@B.CO", rules, ""),
            format!("@B.CO~{}", digest("", "A@B.CO"))
        );
    }

    #[test]
    fn the_domain_split_backtracks_like_the_regex() {
        let rules = PiiRules {
            email: true,
            phone: false,
        };
        // The largest dot with a two-letter tail wins; the tail's own
        // trailing class bytes survive the match.
        assert_eq!(
            scrub("a@b.co9", rules, ""),
            format!("@b.co~{}9", digest("", "a@b.co"))
        );
        assert_eq!(
            scrub("a@b.co.", rules, ""),
            format!("@b.co~{}.", digest("", "a@b.co"))
        );
        assert_eq!(
            scrub("a@b.co.uk", rules, ""),
            format!("@b.co.uk~{}", digest("", "a@b.co.uk"))
        );
        // One letter after every dot: never a match.
        assert_eq!(scrub("a@b.c.d", rules, ""), "a@b.c.d");
    }

    #[test]
    fn the_local_run_is_clipped_at_the_previous_match() {
        let rules = PiiRules {
            email: true,
            phone: false,
        };
        // The first match ends before "9"; the resume makes "9" the next
        // local part.
        assert_eq!(
            scrub("a@b.co9@x.yz", rules, ""),
            format!(
                "@b.co~{}@x.yz~{}",
                digest("", "a@b.co"),
                digest("", "9@x.yz")
            )
        );
    }

    #[test]
    fn non_ascii_domain_bytes_break_the_run() {
        let rules = PiiRules {
            email: true,
            phone: false,
        };
        assert_eq!(scrub("a@b.cö", rules, ""), "a@b.cö");
        assert_eq!(scrub("a@ö.co", rules, ""), "a@ö.co");
    }

    #[test]
    fn a_degenerate_domain_of_alternating_dots_stays_linear() {
        // The split's backward sweep is O(run), not O(run²): a domain of
        // 100_000 alternating "a." bytes resolves without rescanning, on
        // both sides of the match line. The single-letter tail makes the
        // whole address a documented non-match (no dot anywhere has two
        // letters after it); the "zz" tail makes the LAST dot the match.
        let rules = PiiRules {
            email: true,
            phone: false,
        };
        let dots: String = "a.".repeat(50_000);
        let non_match = format!("x@{dots}a");
        assert!(matches!(scrub_pii(&non_match, rules, ""), Cow::Borrowed(_)));
        let matched = format!("x@{dots}zz");
        let expected = format!("@{dots}zz~{}", digest("", &matched));
        assert_eq!(scrub(&matched, rules, ""), expected);
    }

    #[test]
    fn phone_tokens_keep_the_first_three_code_points() {
        let rules = PiiRules {
            email: false,
            phone: true,
        };
        assert_eq!(
            scrub("+14155552671", rules, ""),
            format!("+14~{}", digest("", "+14155552671"))
        );
        // The domestic spelling's third code point is the space.
        assert_eq!(
            scrub("+1 (415) 555-2671", rules, ""),
            format!("+1 ~{}", digest("", "+1 (415) 555-2671"))
        );
        // Non-ASCII digits carry their own spelling into the prefix.
        let arabic = "+\u{0661}\u{0662}\u{0663}\u{0664}\u{0665}\u{0666}\u{0667}\u{0668}";
        assert_eq!(
            scrub(arabic, rules, ""),
            format!("+\u{0661}\u{0662}~{}", digest("", arabic))
        );
    }

    #[test]
    fn the_phone_anchoring_is_documented_non_matches() {
        let rules = PiiRules {
            email: false,
            phone: true,
        };
        assert_eq!(
            scrub("read 4096 bytes in 1200 ms", rules, ""),
            "read 4096 bytes in 1200 ms"
        );
        assert_eq!(scrub("+ (415) 555-2671", rules, ""), "+ (415) 555-2671");
        assert_eq!(scrub("+1234567", rules, ""), "+1234567");
        assert_eq!(scrub("+12\u{00b3}45678", rules, ""), "+12\u{00b3}45678"); // No, not Nd
    }

    #[test]
    fn a_class_run_spans_two_numbers_split_by_one_space() {
        let rules = PiiRules {
            email: false,
            phone: true,
        };
        let matched = "+4712345678 1234567890";
        assert_eq!(
            scrub(matched, rules, ""),
            format!("+47~{}", digest("", matched))
        );
    }

    #[test]
    fn the_match_ends_at_the_last_digit_of_the_run() {
        let rules = PiiRules {
            email: false,
            phone: true,
        };
        assert_eq!(
            scrub("call +1 (415) 555-2671 , ok", rules, ""),
            format!("call +1 ~{} , ok", digest("", "+1 (415) 555-2671"))
        );
    }

    #[test]
    fn a_second_plus_starts_the_match() {
        let rules = PiiRules {
            email: false,
            phone: true,
        };
        assert_eq!(
            scrub("+1+4155552671", rules, ""),
            format!("+1+41~{}", digest("", "+4155552671"))
        );
    }

    #[test]
    fn the_email_pass_runs_before_the_phone_pass() {
        // One email whose local part is a whole E.164 number: the email
        // rule eats it, and nothing remains for the phone rule.
        let matched = "user+14155552671@example.com";
        assert_eq!(
            scrub(matched, PiiRules::BOTH, ""),
            format!("@example.com~{}", digest("", matched))
        );
    }

    #[test]
    fn the_default_salt_is_not_the_unsalted_digest() {
        let unsalted = scrub("a@b.co", PiiRules::BOTH, "");
        let default = scrub("a@b.co", PiiRules::BOTH, DEFAULT_SALT);
        assert_ne!(unsalted, default);
        assert_eq!(default, format!("@b.co~{}", digest(DEFAULT_SALT, "a@b.co")));
    }

    #[test]
    fn phone_only_is_strictly_idempotent() {
        let rules = PiiRules {
            email: false,
            phone: true,
        };
        let once = scrub("+14155552671 +14155552672", rules, "");
        assert!(matches!(scrub_pii(&once, rules, ""), Cow::Borrowed(_)));
    }

    #[test]
    fn adjacent_email_matches_re_fire_once_then_converge() {
        let rules = PiiRules {
            email: true,
            phone: false,
        };
        let once = scrub("a@b.co9@x.yz", rules, "");
        let twice = scrub(&once, rules, "");
        assert_ne!(once, twice);
        assert!(matches!(scrub_pii(&twice, rules, ""), Cow::Borrowed(_)));
    }

    #[test]
    fn a_match_followed_by_an_unmatched_at_run_re_fires_once() {
        // The corner's second shape: "x@b.co" consumed the local-part
        // material before the second `@`, so "@w.vu" never matched on
        // its own — but the first token's digest hex is local-part
        // material, and pass two fires on [hex + "@w.vu"].
        let rules = PiiRules {
            email: true,
            phone: false,
        };
        let once = scrub("x@b.co@w.vu", rules, "");
        let twice = scrub(&once, rules, "");
        assert_ne!(once, twice);
        assert!(matches!(scrub_pii(&twice, rules, ""), Cow::Borrowed(_)));
    }

    #[test]
    fn tokens_are_fixed_points() {
        for token in [
            format!("@b.co~{}", digest("", "a@b.co")),
            format!("+14~{}", digest("", "+14155552671")),
            format!("+1 ~{}", digest("", "+1 (415) 555-2671")),
        ] {
            assert!(
                matches!(scrub_pii(&token, PiiRules::BOTH, ""), Cow::Borrowed(_)),
                "{token}"
            );
        }
    }

    #[test]
    fn the_nd_table_agrees_with_the_ascii_fast_path() {
        for b in b'0'..=b'9' {
            assert!(is_nd(b as char));
        }
        for b in b'a'..=b'z' {
            assert!(!is_nd(b as char));
        }
        // One representative per script beyond ASCII, plus the non-Nd
        // numerics that must never match.
        for c in '\u{0660}'..='\u{0669}' {
            assert!(is_nd(c), "Arabic-Indic {c:?}");
        }
        assert!(is_nd('\u{0966}')); // Devanagari
        assert!(is_nd('\u{FF10}')); // fullwidth
        assert!(is_nd('\u{1D7CE}')); // mathematical bold
        assert!(is_nd('\u{1E950}')); // Adlam
        assert!(is_nd('\u{1FBF7}')); // segmented digit seven: Nd, not No
        assert!(!is_nd('\u{00B2}')); // superscript two: No
        assert!(!is_nd('\u{2169}')); // Roman numeral: Nl
    }
}
