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
//! * **Phone rule, two matchers** —
//!   * *international*: the source's `\+\d[\d\-. ()]{6,}\d`, anchored on a
//!     literal `+` (a bare digit run is an order number, byte count, or
//!     timestamp, and redacting those would destroy the diagnostic this
//!     scrubber exists to preserve), a digit directly after the `+`, six
//!     or more middle class characters, and a final digit — the middle
//!     `{6,}` counts separators, so a long spelling matches on seven
//!     digits (`+1 415 555 ` after an email pass eats its tail digits)
//!     while `+1234567` never matches; the digit class is every Unicode
//!     Nd codepoint, one space-bridged run is ONE match, the match ends
//!     at the run's last digit, and a `~` or a second `+` breaks it. A
//!     `+` before a run marks international intent for the whole run:
//!     the grammar matches or the run is a documented non-match, and
//!     the domestic matcher never fires behind a `+` (which is why
//!     `+ (415) 555-2671` stays untouched, exactly as the source leaves
//!     it).
//!   * *domestic* (the extension past the source): un-plussed NANP
//!     shapes, a FULL run of exactly ten digits, or eleven with an ASCII
//!     leading `1`, in any `[\d\-. ()]` spelling — `(XXX) XXX-XXXX`,
//!     `XXX-XXX-XXXX`, `XXX.XXX.XXXX`, `XXX XXX XXXX`, the 1-prefixed
//!     variants — with three discipline rules the international
//!     anchor's rationale demands. First, the match must carry at least
//!     one separator: a BARE unseparated digit run is an order number
//!     or id even at exactly ten digits, the same reasoning that
//!     anchors the international matcher on `+`, and (the structural
//!     consequence) it is what keeps a token's own digest hex
//!     unmatchable from inside, so phone-only stays strictly idempotent
//!     and scrub-twice convergence cannot be chained adversarially
//!     through chosen digests. Second, the match must start at a CLEAN
//!     boundary — its first char glued to neither `~` nor a lowercase
//!     `a`-`f` (the token-interior alphabet: hex `a`-`f` + `~`, exactly;
//!     `g`/`z`/`A`-`F` are clean and still match) — so a run starting inside
//!     an email token's digest can never flow out through a separator
//!     into following text and compose a fresh "number" out of hex
//!     digits (`…~e292cb255128 4096` stays untouched); in real text a
//!     digit run glued to a letter or tilde is an identifier fragment,
//!     the same reasoning as the bare cut. Third, no partial match
//!     inside a longer run: twelve-plus digits is an id, not a phone.
//!     Leading spaces are skipped (word separation); other leading
//!     separators are part of the spelling; trailing separators
//!     survive past the last digit, same as international; and non-NANP
//!     un-plussed domestic (`020 …` shapes) is out of scope, the `+`
//!     form being the international spelling of those.
//! * **Pass order** — email substitution over the whole string first,
//!   then phone substitution over its result, each exactly once, no
//!   cascade: an email's local part may itself contain a `+`-led digit
//!   run (`user+14155552671@example.com` is one email), so the phone
//!   rule must see the email tokens, never the addresses that produced
//!   them. One reachable interaction is documented rather than fixed:
//!   an email token whose DOMAIN spells a domestic number
//!   (`@5551234567.co~…`) has its digit half re-tokenized by the phone
//!   pass — over-redaction in the safe direction, converging on the
//!   second scrub like every other shape.
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
//!   Phone-only is strictly idempotent, and the domestic matcher keeps
//!   it so by construction: every phone match, international or
//!   domestic, ENDS at its run's last digit (the run's remainder is
//!   separator-only, holding no new match), no run can span a token
//!   boundary (the `~` is not class), and a token's interior can hold
//!   no domestic match — the prefix is at most three codepoints (under
//!   ten digits), the digest hex carries no separator, and the
//!   clean-boundary rule keeps a run that starts inside a digest from
//!   flowing out into following text. The same rules are what make the
//!   both-rules email-token corners converge: the domain that spells a
//!   number leaves only letter-bearing fragments behind, and the
//!   digest-tail-plus-adjacent-digits composition is unreachable.
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
//!
//! Threat model: diagnostic-preserving, NOT adversarial-robust. The
//! narrow grammars are the parity contract at `salt=""` (widening them
//! silently would break byte-identity), so attacker-controlled
//! formatting bypasses by design — `/ : , ;` split runs, fullwidth
//! U+FF0B and fullwidth spaces bypass, IDN/non-ASCII domains leak whole
//! (idna-to-punycode before scrub for IDN threat), IP-literal
//! `user@[192.168.1.1]` and dotted-quad `user@192.168.1.1` leak whole,
//! RFC quoted-string locals (`"user@name"@example.com`) leak whole (the
//! quote before `@` blocks the match — not a fragment-leak),
//! RFC local chars outside `[A-Za-z0-9._%+-]` fragment-leak (`a!`
//! survives), bare 10/11-digit and short `+`-led runs never match,
//! extensions (`x1234`) survive past the last digit, NPA/NXX unvalidated,
//! the digest is 48 bits (~2.5% merge at ~119k, frozen-for-stability) over
//! plain `salt||match` (boundaries can alias). For
//! adversarial threat, map Zs/Zl/Zp plus `\t\n\r\f\v` to U+0020 and
//! canonicalize separators/domains before scrub (`tors.nfkc` alone is
//! insufficient); see `docs/api.md`'s scrub_pii section for the full
//! residual-risk list and the canonicalization code block.
//!
//! Performance: one linear `memchr`-anchored pass per rule, `Cow::Borrowed`
//! identity when nothing matches, `py.detach` around the whole scan on
//! the Python side, and `sha2` digests computed only for spans that
//! actually matched (never per candidate). The degenerate-domain bench
//! (`a.` × 50k) pins the linear domain split.

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

/// One maximal phone-class run, with the facts both phone grammars
/// classify it by. `run_end` is one past the run's last class char;
/// `last_digit_end` one past its last Nd digit and
/// `last_digit_char_idx` that digit's index within the run (the
/// international grammar's true test: the regex
/// `\+\d[\d\-. ()]{6,}\d` needs a digit at the eighth class character
/// or beyond — the middle `{6,}` counts separators too, so a long
/// spelling matches on seven digits); `first_digit` is the run's first
/// digit; `starts_with_digit` whether the run's first char is one (a
/// digit directly after the `+`); `leading_spaces` the space chars at
/// the head of the run, before the first non-space class char (a
/// domestic match skips them: word separation, not spelling).
struct PhoneRun {
    run_end: usize,
    digits: usize,
    last_digit_end: usize,
    last_digit_char_idx: usize,
    first_digit: Option<char>,
    starts_with_digit: bool,
    leading_spaces: usize,
}

/// The next phone-class char (an Nd digit or a phone separator) at or
/// after `from`, as a byte offset. ASCII answers directly; a non-ASCII
/// byte costs one char decode, and only for the Nd check (every phone
/// separator is ASCII).
fn next_phone_class(text: &str, from: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut i = from;
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii() {
            if b.is_ascii_digit() || matches!(b, b'-' | b'.' | b' ' | b'(' | b')') {
                return Some(i);
            }
            i += 1;
        } else {
            let c = text[i..].chars().next().unwrap();
            if is_nd(c) {
                return Some(i);
            }
            i += c.len_utf8();
        }
    }
    None
}

/// The maximal class run starting at `run_start`, walked once.
fn scan_phone_run(text: &str, run_start: usize) -> PhoneRun {
    let mut run = PhoneRun {
        run_end: run_start,
        digits: 0,
        last_digit_end: run_start,
        last_digit_char_idx: 0,
        first_digit: None,
        starts_with_digit: false,
        leading_spaces: 0,
    };
    let bytes = text.as_bytes();
    let mut i = run_start;
    let mut char_idx = 0;
    let mut seen_non_space = false;
    while i < bytes.len() {
        let c = text[i..].chars().next().unwrap();
        let nd = is_nd(c);
        if !nd && !is_phone_separator(c) {
            break;
        }
        if char_idx == 0 {
            run.starts_with_digit = nd;
        }
        if !seen_non_space {
            if c == ' ' {
                run.leading_spaces += 1;
            } else {
                seen_non_space = true;
            }
        }
        i += c.len_utf8();
        run.run_end = i;
        if nd {
            run.digits += 1;
            if run.first_digit.is_none() {
                run.first_digit = Some(c);
            }
            run.last_digit_end = i;
            run.last_digit_char_idx = char_idx;
        }
        char_idx += 1;
    }
    run
}

/// Whether the match span `[start, end)` carries a phone separator: the
/// domestic grammar's bare-run cut. Walked only for runs whose digit
/// count already qualifies, so the amortized cost stays one pass.
fn has_separator_in(text: &str, start: usize, end: usize) -> bool {
    text[start..end].chars().any(is_phone_separator)
}

/// The phone pass: every leftmost match of the two phone grammars
/// becomes `prefix~digest` (`prefix` = the match's first three code
/// points). One linear scan over the class runs: a run preceded by a
/// `+` is international territory (the `+` plus the run matches when
/// the run starts with a digit and its last digit sits at the eighth
/// class character or beyond — the regex's own `{6,}` middle counting
/// separators, so a long spelling matches on seven digits; anything
/// else is a documented non-match, and the domestic grammar never
/// fires behind a `+`); an un-plussed run of exactly ten digits, or
/// eleven with an ASCII leading `1`, carrying a separator, is a
/// domestic match. Non-matching runs are spent whole — no partial
/// match inside a longer run. `Cow::Borrowed` when nothing matches.
fn scrub_phone_pass<'a>(text: &'a str, salt: &str) -> Cow<'a, str> {
    let bytes = text.as_bytes();
    let mut pos = 0;
    let mut emitted = 0;
    let mut out: Option<String> = None;
    while pos < bytes.len() {
        let Some(run_start) = next_phone_class(text, pos) else {
            break;
        };
        let run = scan_phone_run(text, run_start);
        let plussed = run_start > 0 && bytes[run_start - 1] == b'+';
        let span = if plussed {
            if run.starts_with_digit && run.last_digit_char_idx >= 7 {
                Some((run_start - 1, run.last_digit_end))
            } else {
                None
            }
        } else {
            // Domestic: a full un-plussed run of exactly ten digits, or
            // eleven with an ASCII leading `1`, carrying a separator (a
            // bare digit run is an id, not a phone), and starting at a
            // CLEAN boundary: the match's first char glued to neither `~`
            // or a lowercase `a`-`f` (hex `a`-`f` + `~`, exactly) — so
            // no run starting inside a token's digest can flow out
            // through a separator into following text and compose a
            // fresh "number" out of hex digits. In real text a digit
            // run glued to hex/tilde is an identifier fragment,
            // the same reasoning as the bare cut.
            let nanp_shape = run.digits == 10 || (run.digits == 11 && run.first_digit == Some('1'));
            let match_start = run_start + run.leading_spaces;
            let clean_start =
                match_start == 0 || !matches!(bytes[match_start - 1], b'~' | b'a'..=b'f');
            if nanp_shape && clean_start && has_separator_in(text, match_start, run.last_digit_end)
            {
                Some((match_start, run.last_digit_end))
            } else {
                None
            }
        };
        let Some((start, end)) = span else {
            pos = run.run_end;
            continue;
        };
        // The token prefix: the match's first three code points. Every
        // match holds at least nine (a `+`, a digit, six more class
        // chars, a final digit), so three always exist.
        let mut prefix_end = start;
        for _ in 0..3 {
            prefix_end += text[prefix_end..].chars().next().unwrap().len_utf8();
        }
        let matched = &text[start..end];
        let token = format!(
            "{}~{}",
            &text[start..prefix_end],
            token_digest(salt, matched)
        );
        let buf = out.get_or_insert_with(|| String::with_capacity(text.len()));
        buf.push_str(&text[emitted..start]);
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
    fn rfc_quoted_locals_and_ip_domains_leak_whole() {
        // C1/C2: RFC quoted-string locals and IP-literal/dotted-quad
        // domains never match — the whole address survives. No grammar
        // widening (parity): canonicalize before scrub.
        let rules = PiiRules { email: true, phone: false };
        for text in [
            r#""user@name"@example.com"#,
            r#""a@b"@x.co"#,
            "user@[192.168.1.1]",
            "user@192.168.1.1",
        ] {
            assert!(matches!(scrub_pii(text, rules, ""), Cow::Borrowed(_)), "{text}");
        }
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

    // --- The domestic matcher (the extension past the source) --------------------

    fn phone_only() -> PiiRules {
        PiiRules {
            email: false,
            phone: true,
        }
    }

    #[test]
    fn domestic_shapes_match_in_every_separator_spelling() {
        for (text, matched) in [
            ("(415) 555-2671", "(415) 555-2671"),
            ("415-555-2671", "415-555-2671"),
            ("415.555.2671", "415.555.2671"),
            ("415 555 2671", "415 555 2671"),
            ("1-415-555-2671", "1-415-555-2671"),
            ("1 (415) 555-2671", "1 (415) 555-2671"),
            ("1 415 555 2671", "1 415 555 2671"),
            ("1415 555 2671", "1415 555 2671"),
        ] {
            assert_eq!(
                scrub(text, phone_only(), ""),
                format!("{}~{}", &matched[..3], digest("", matched)),
                "{text}"
            );
        }
    }

    #[test]
    fn domestic_matches_skip_leading_spaces_and_spare_trailing_separators() {
        assert_eq!(
            scrub("call 415-555-2671 ok", phone_only(), ""),
            format!("call 415~{} ok", digest("", "415-555-2671"))
        );
        // A leading structural separator is part of the spelling.
        assert_eq!(
            scrub("x -415-555-2671 , ok", phone_only(), ""),
            format!("x -41~{} , ok", digest("", "-415-555-2671"))
        );
        // Extensions survive past the last digit, same as international.
        assert_eq!(
            scrub("415-555-2671 x1234", phone_only(), ""),
            format!("415~{} x1234", digest("", "415-555-2671"))
        );
    }

    #[test]
    fn bare_digit_runs_never_match_even_at_ten_or_eleven() {
        // The documented cut: a bare unseparated digit run is an order
        // number or id; the separator requirement is also what keeps a
        // token's own digest hex unmatchable (strict idempotence).
        for text in [
            "4155552671",
            "14155552671",
            "ref 4155552671 x",
            "id 4155552671 ",
            "order 1234567890 closed",
        ] {
            assert_eq!(scrub(text, phone_only(), ""), text, "{text}");
        }
    }

    #[test]
    fn the_width_discipline_rejects_nine_twelve_and_wrong_eleven() {
        for text in [
            "415-555-267",               // nine digits
            "415-555-267123",            // twelve: an id, not a phone
            "415-555-2671-555-123-4567", // twenty in ONE run: an id
            "915-555-26712",             // eleven, but not led by ASCII '1'
        ] {
            assert_eq!(scrub(text, phone_only(), ""), text, "{text}");
        }
        // Eleven in Nd digits whose first is the Arabic-Indic ONE, not
        // the ASCII '1': not the NANP trunk-prefix shape.
        let arabic_eleven = "\u{0661}\u{0661}\u{0665} \u{0665}\u{0665}\u{0665} \u{0662}\u{0666}\u{0667}\u{0661}\u{0662}";
        assert_eq!(scrub(arabic_eleven, phone_only(), ""), arabic_eleven);
    }

    #[test]
    fn unicode_nd_digits_spell_domestic_numbers() {
        // Ten Arabic-Indic digits, space-separated: a domestic match
        // whose token prefix carries the script's own spelling.
        let arabic =
            "\u{0664}\u{0661}\u{0665} \u{0665}\u{0665}\u{0665} \u{0662}\u{0666}\u{0667}\u{0661}";
        let prefix: String = arabic.chars().take(3).collect();
        assert_eq!(
            scrub(arabic, phone_only(), ""),
            format!("{prefix}~{}", digest("", arabic))
        );
    }

    #[test]
    fn a_plus_before_a_run_is_international_territory_never_domestic() {
        // Ten digits behind a + is the INTERNATIONAL match; a separator
        // right after the +, or a run too short for the {6,} middle, is
        // the source's documented non-match — never a domestic fallback.
        assert_eq!(
            scrub("+415-555-2671", phone_only(), ""),
            format!("+41~{}", digest("", "+415-555-2671"))
        );
        assert_eq!(
            scrub("+ (415) 555-2671", phone_only(), ""),
            "+ (415) 555-2671"
        );
        assert_eq!(scrub("+415-555", phone_only(), ""), "+415-555");
    }

    #[test]
    fn a_long_spelling_matches_on_seven_digits() {
        // The regex's {6,} middle counts separators: the run
        // "1 415 555 " holds seven digits across nine class chars, and
        // its last digit sits at the eighth char or beyond — a match,
        // the exact shape the parity corpus catches when an email pass
        // eats a phone piece's tail digits.
        let matched = "+1 415 555";
        assert_eq!(
            scrub("ring +1 415 555 now", phone_only(), ""),
            format!("ring +1 ~{} now", digest("", matched))
        );
        // The short spelling of the same seven digits never matches.
        assert_eq!(scrub("+1415555", phone_only(), ""), "+1415555");
    }

    #[test]
    fn two_domestic_matches_with_a_break_both_scrub() {
        let (a, b) = ("415-555-2671", "415-555-2672");
        assert_eq!(
            scrub("415-555-2671, 415-555-2672", phone_only(), ""),
            format!("415~{}, 415~{}", digest("", a), digest("", b))
        );
    }

    #[test]
    fn an_email_token_domain_that_spells_a_number_is_re_tokenized() {
        // The documented over-redaction corner: the email pass tokens
        // the whole address, and the phone pass re-tokens the domestic
        // shape the DOMAIN spelled (the dot between the digit groups is
        // the separator that makes it matchable; digits before a final
        // ".co" alone are not). Safe direction; converges.
        let matched = "user@555.1234567.co";
        let once = scrub(matched, PiiRules::BOTH, "");
        assert_eq!(
            once,
            format!(
                "@555~{}.co~{}",
                digest("", "555.1234567"),
                digest("", matched)
            )
        );
        let twice = scrub(&once, PiiRules::BOTH, "");
        assert!(matches!(
            scrub_pii(&twice, PiiRules::BOTH, ""),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn a_match_start_glued_to_hex_or_tilde_is_an_identifier_fragment() {
        // The clean-boundary rule: a domestic match's first char may not
        // be glued to neither `~` nor a lowercase a-f — the token-interior
        // alphabet (hex a-f + `~`, exactly; NOT "letters" in general) —
        // so a run starting inside a token's digest can never flow out
        // into following text and compose a fresh "number" out of hex
        // digits.
        for text in [
            "job255128-4096",
            "value~255-123-4567",
            "ref c415-555-2671",
            "ref a415-555-2671",
            "ref f415-555-2671",
        ] {
            assert_eq!(scrub(text, phone_only(), ""), text, "{text}");
        }
        // A clean boundary is anything else — g/z, uppercase A-F, and
        // every word-separated spelling. Only lowercase a-f and `~` are
        // dirty by design.
        for (text, matched, prefix_glue) in [
            ("jobg415-555-2671", "415-555-2671", "jobg"),
            ("jobz415-555-2671", "415-555-2671", "jobz"),
            ("jobG415-555-2671", "415-555-2671", "jobG"),
            ("jobA415-555-2671", "415-555-2671", "jobA"),
            ("jobF415-555-2671", "415-555-2671", "jobF"),
        ] {
            assert_eq!(
                scrub(text, phone_only(), ""),
                format!("{prefix_glue}415~{}", digest("", matched)),
                "{text}"
            );
        }
        assert_eq!(
            scrub("item 415-555-2671", phone_only(), ""),
            format!("item 415~{}", digest("", "415-555-2671"))
        );
    }

    #[test]
    fn an_email_tokens_digest_cannot_compose_a_domestic_number() {
        // The parity suite's falsifier, verbatim: three adjacent
        // addresses whose email pass leaves tokens whose digest tails
        // plus the following numbers could spell ten-digit runs — the
        // clean-boundary rule keeps the phone pass off them, exactly
        // two email tokens fire, and the numbers survive untouched.
        let text = "fungai.chetima@example.comfungai.chetima@example.com\
fungai.chetima@example.comread 4096 bytes in 1200 ms";
        let once = scrub(text, PiiRules::BOTH, "");
        assert_eq!(once.matches('~').count(), 2, "{once}");
        assert!(once.ends_with(" 4096 bytes in 1200 ms"), "{once}");
        let twice = scrub(&once, PiiRules::BOTH, "");
        assert!(matches!(
            scrub_pii(&twice, PiiRules::BOTH, ""),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn phone_only_stays_strictly_idempotent_with_domestic_matches() {
        let once = scrub("(415) 555-2671 415-555-2672 +14155552673", phone_only(), "");
        assert!(matches!(
            scrub_pii(&once, phone_only(), ""),
            Cow::Borrowed(_)
        ));
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
        // Table hygiene: sorted, non-overlapping, and the committed
        // Unicode 16.0.0 count (760). The per-interpreter exhaustive pin
        // lives in tests/test_scrub_pii.py::TestNdExhaustive (unicodedata
        // as oracle, observed through `+`-anchored matches).
        assert_eq!(ND_RANGES.len(), 71);
        let mut total = 0u32;
        let mut prev_hi = 0u32;
        for &(lo, hi) in &ND_RANGES {
            assert!(lo <= hi, "{lo:#X}..={hi:#X}");
            assert!(lo > prev_hi, "unsorted/overlapping at {lo:#X}");
            total += hi - lo + 1;
            prev_hi = hi;
        }
        assert_eq!(total, 760);
        assert!(!is_nd('\u{FF0B}')); // fullwidth plus: not a digit
        assert!(!is_nd('~'));
        assert!(!is_nd('+'));
    }
}
