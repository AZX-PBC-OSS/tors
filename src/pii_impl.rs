//! Contact- and credential-material scrub, the pure-Rust core of
//! `tors.scrub_pii`.
//!
//! Replaces email addresses, `+`-led phone numbers, and provider/platform
//! API-key material inside free text with correlation tokens: the scrub
//! an error excerpt, rejection message, or response-body excerpt needs
//! before it reaches telemetry, because telemetry is the one store a
//! data purge cannot reach — a candidate scrubbed from the primary store
//! must not leave their address behind in log retention, and a rotated
//! credential must not survive in one. The contact rules are a port of a
//! private consumer's telemetry-safety module, pinned byte-identical to
//! it at `salt=""` (the quoted-pin oracle in `tests/reference.py` is the
//! transcription; `tests/test_scrub_pii_parity.py` is the differential);
//! the keys rule and the domestic phone matcher are extensions past that
//! contract, the same maintainer-directed posture.
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
//!     digits (`…~e292cb255128 4096` stays untouched) — and a token span
//!     itself (``~`` + 12 digest hex) is a breaker: runs never start
//!     inside a digest (the scan resumes after the token instead of
//!     spending a composed digest-plus-number run whole, which would
//!     silently swallow the real number after it), and the byte after a
//!     token is a clean boundary even when the digest ends hex-dirty, so
//!     an adjacent number still scrubs exactly. In real text a
//!     digit run glued to a letter or tilde is an identifier fragment,
//!     the same reasoning as the bare cut. Third, no partial match
//!     inside a longer run: twelve-plus digits is an id, not a phone.
//!     Leading spaces are skipped (word separation); other leading
//!     separators are part of the spelling; trailing separators
//!     survive past the last digit, same as international; and eleven
//!     digits not led by ASCII `1` are excluded (the NANP trunk-prefix
//!     shape) — ten-digit runs carry no leading-digit check at all, so a
//!     `0`-led ten-digit shape (`020-794-6095`) scrubs like any other,
//!     only its eleven-digit `0`-led spelling staying out (the `+` form
//!     being the international spelling of those).
//! * **Keys rule** (the credential extension past the source) — the
//!   evidence-backed closed set of provider/platform key families, each
//!   a literal prefix plus a minimal `[A-Za-z0-9_-]` tail consumed
//!   maximally: OpenAI `sk-`/`sk-proj-`/`sk-svcacct-` (20+), Anthropic
//!   `sk-ant-` (20+), Google `AIza` (35+), Fireworks `fw-`/`fw_` (20+),
//!   Modal `ak-`/`wk-` (20+), GitHub `ghp_` (36+) and `github_pat_`
//!   (22+), the minted shapes `azxdev_` (20+), `wd-` (43+), `w-` (43+),
//!   `cn-` (20+), and MARKER-SCOPED JWTs — `Bearer eyJ` plus three
//!   maximal base64url segments, single-dot separated (a bare `eyJ`
//!   never matches: one consumer's API legitimately carries eyJ-shaped
//!   non-secret cursors, and redacting those would destroy the
//!   diagnostic this scrubber exists to preserve). The leak vector is
//!   the error text itself: provider and platform error strings can
//!   quote the credential back — five private consumers evidenced, the
//!   strongest a platform whose own code comments that a vendor auth
//!   failure "can quote the key" and keeps the full text in an
//!   admin-served ledger. Slack `xox`, Stripe, and AWS `AKIA` shapes are
//!   deliberately absent (zero evidence): growing the set is a
//!   new-evidence decision, never a drive-by. Three discipline rules:
//!   the prefixes are tried LONGEST-FIRST with fall-through (a
//!   too-short `sk-ant-` tail falls through to the bare `sk-` family,
//!   whose own tail swallows the `ant-` spelling — still scrubbed, with
//!   the generic prefix); a prefix glued to a preceding key-charset char
//!   is MID-TOKEN and never fires (`xak-…` — the same reasoning as the
//!   phone rule's clean-boundary cut, and what keeps a second key glued
//!   to a token's digest hex from firing); and the tail run is MAXIMAL,
//!   so a key glued to further charset material is one long key —
//!   over-redaction in the safe direction.
//! * **Pass order** — the keys substitution over the whole string FIRST,
//!   then the email substitution over its result, then the phone
//!   substitution over that, each exactly once, no cascade. The keys
//!   pass must run before the contact passes because a key's tail can
//!   spell a dash-separated ten-digit run (a domestic phone match if
//!   the phone pass saw it first) and a whole key can spell an email
//!   local part (`sk-…@x.co` would be one email match); the email rule
//!   must run before the phone rule because an email's local part may
//!   itself contain a `+`-led digit run (`user+14155552671@example.com`
//!   is one email), so the phone rule must see the email tokens, never
//!   the addresses that produced them. Three reachable interactions are
//!   documented rather than fixed: the two below (an email token whose
//!   DOMAIN spells a domestic number has its digit half re-tokenized by
//!   the phone pass, and an email local removal can trim a too-long
//!   digit run into exactly ten (or eleven-with-`1`) digits that then
//!   scrub), plus the keys-before-email corner — the keys pass eats a
//!   key-shaped local part and leaves `<family>~<digest>@domain`, whose
//!   digest hex is itself local-part material, so the email pass tokens
//!   `hex@domain` (the family prefix survives, the domain tokens too) —
//!   all over-redaction in the safe direction, converging on the second
//!   scrub like every other shape.
//! * **Tokens** — `@domain~<digest>` for an email match, `prefix~<digest>`
//!   for a phone match, where `prefix` is the match's first three CODE
//!   POINTS (a canonical E.164's country code — `"+47"` compact, `"+1 "`
//!   for a domestic spelling where the third code point is the space),
//!   and `<family-prefix>~<digest>` for a key match, where the family
//!   prefix is kept VERBATIM (`sk-`, `sk-ant-`, `github_pat_`, `AIza`,
//!   `Bearer`) — the non-secret half that tells the operator WHICH
//!   credential to rotate. In every rule `<digest>` is the first 12 hex
//!   chars of `sha256(salt + match)`, and for a key the match span is
//!   the FULL key text (prefix + tail). A token is a correlation handle,
//!   not a secret: the E.164 space is small enough to enumerate, so the
//!   digest lets an operator tie two log lines to the same number without the
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
//!   boundary (the `~` is not class, and a `~` + 12-hex span is skipped
//!   whole), and a token's interior can hold no domestic match — the
//!   prefix is at most three codepoints (under ten digits), the digest
//!   hex carries no separator, and a run starting inside a digest
//!   resumes after the token instead of flowing out into following
//!   text. The same rules are what make the both-rules email-token
//!   corners converge: the domain that spells a number leaves only
//!   letter-bearing fragments behind, and a digest tail followed by an
//!   adjacent number splits at the token — the tail never composes, and
//!   the number scrubs exactly. Key tokens are fixed points by
//!   construction, every family: the prefix ends in `-` or `_` (or is
//!   `AIza`/`Bearer`), the byte after it is `~`, never tail charset, so
//!   no family can re-fire at the token's own head — and no family
//!   prefix can be spelled inside 12 lowercase digest hex (the
//!   distinctive second characters — `z` in `AIza`/`azxdev_`, `k` in
//!   `ak-`, `n` in `cn-`, `w` in `fw-`/`wk-`, `h` in `ghp_`, the space in
//!   `Bearer eyJ` — are all outside `[0-9a-f]`), so the digest half is
//!   inert too. A key token's `~` + 12 hex is a token span for the phone
//!   pass's existing breaker, so a number after it keeps its clean run.
//!   `tests/test_scrub_pii.py` and the unit tests below pin the fixed
//!   point per family.
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
//!   tokens are redaction, not pseudonymization crypto. `salt=None`
//!   resolves PER RULE: the contact rules keep `DEFAULT_SALT` and the
//!   keys rule digests with its own `KEYS_DEFAULT_SALT` tag — the split
//!   is load-bearing, keeping a key digest from ever aliasing a contact
//!   digest at the default settings — while an explicit string salts
//!   every rule alike.
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
//! plain `salt||match` (boundaries can alias). The keys rule adds its
//! own documented cuts: punctuation inside a key splits the tail
//! (`sk-…/…-rest` never matches whole), an unlisted provider's key
//! shape leaks whole (the family set is closed on evidence — the
//! exclusion is deliberate, and a new family is a new-evidence decision),
//! and the kept family prefix is a coarse provider label, not a
//! credential. For
//! adversarial threat, map Zs/Zl/Zp plus `\t\n\r\f\v` to U+0020 and
//! canonicalize separators/domains before scrub (`tors.nfkc` alone is
//! insufficient); see `docs/api.md`'s scrub_pii section for the full
//! residual-risk list and the canonicalization code block.
//!
//! Performance: one linear pass per rule — `memchr`-anchored for the
//! `@` and phone-class scans, a first-byte-dispatched table walk for the
//! key families (one `matches!` per byte on prose, at most fifteen
//! prefix compares on an anchor hit) — `Cow::Borrowed`
//! identity when nothing matches, `py.detach` around the whole scan on
//! the Python side (the keys pass rides that same single detach; no new
//! GIL class), and `sha2` digests computed only for spans that
//! actually matched (never per candidate). The degenerate-domain bench
//! (`a.` × 50k) pins the linear domain split.

use std::borrow::Cow;

use memchr::memchr;
use sha2::{Digest, Sha256};

/// tors's documented default digest salt for the CONTACT rules (email,
/// phone). A fixed, non-secret, versioned domain-separation tag, frozen:
/// changing it would silently change every deployment's token values.
/// Mirrored in `tests/reference.py` (`SCRUB_PII_DEFAULT_SALT`), which the
/// salt=None differential lane pins byte-equal to this constant.
pub const DEFAULT_SALT: &str = "tors/scrub_pii/v1";

/// The keys rule's own documented default digest salt — the same
/// frozen-tag discipline as `DEFAULT_SALT`, and a SEPARATE tag because
/// `salt=None` resolves per rule: key digests must never alias contact
/// digests at the default settings (an operator correlating a token
/// across log lines must not have to wonder which rule produced it).
/// An explicit salt string salts every rule alike; `salt=""` is unsalted
/// for every rule (the migration lane). Pinned in
/// `tests/test_scrub_pii.py`'s salt lanes.
pub const KEYS_DEFAULT_SALT: &str = "tors/scrub_keys/v1";

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

/// Whether `b` is a token-digest hex char: lowercase `[0-9a-f]`, exactly
/// the alphabet `const_hex::encode` emits. Uppercase `A`-`F` never opens
/// a token span (tokens are lowercase-only), so natural text holding one
/// keeps the pre-breaker behavior.
#[inline]
fn is_token_hex(b: u8) -> bool {
    matches!(b, b'0'..=b'9' | b'a'..=b'f')
}

/// The end offset of the token span opening at `tilde` (`~` + 12 digest
/// hex chars), or `None`. Both email and phone tokens end this way, and
/// the digest half never holds a separator — so a span found here is
/// always token interior, never a phone spelling.
fn token_span_end_at(text: &str, tilde: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(tilde) != Some(&b'~') {
        return None;
    }
    let end = tilde + 1 + TOKEN_HEX;
    if end > bytes.len() {
        return None;
    }
    if bytes[tilde + 1..end].iter().all(|&b| is_token_hex(b)) {
        Some(end)
    } else {
        None
    }
}

/// The end offset of the token span covering `off`, or `None`: `off`
/// sits strictly inside `~` + 12 hex. A phone-class run starting there
/// is digest tail, not a number's head — the caller resumes after the
/// token instead of spending the composed run whole (which would swallow
/// the real number following it).
fn token_span_containing(text: &str, off: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let lo = off.saturating_sub(TOKEN_HEX);
    let hi = off.min(bytes.len().saturating_sub(1));
    for tilde in lo..=hi {
        if bytes.get(tilde) != Some(&b'~') {
            continue;
        }
        if let Some(end) = token_span_end_at(text, tilde)
            && tilde <= off
            && off < end
        {
            return Some(end);
        }
    }
    None
}

/// Whether a token span ends exactly at `pos`: the byte after a token
/// is a clean boundary (the token is a finished unit, like a word
/// boundary), even when the digest's own last char is `~`-dirty hex.
fn token_ends_at(text: &str, pos: usize) -> bool {
    pos > TOKEN_HEX
        && text.as_bytes().get(pos - 1 - TOKEN_HEX) == Some(&b'~')
        && text.as_bytes()[pos - TOKEN_HEX..pos]
            .iter()
            .all(|&b| is_token_hex(b))
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
/// domestic match — with token spans (`~` + 12 digest hex, either
/// rule's token) acting as breakers: a run starting inside a digest
/// resumes after the token (never a composed digest-plus-number run),
/// and the byte after a token is a clean boundary. Non-matching runs
/// are spent whole — no partial match inside a longer run.
/// `Cow::Borrowed` when nothing matches.
fn scrub_phone_pass<'a>(text: &'a str, salt: &str) -> Cow<'a, str> {
    let bytes = text.as_bytes();
    let mut pos = 0;
    let mut emitted = 0;
    let mut out: Option<String> = None;
    while pos < bytes.len() {
        // A token span (`~` + 12 digest hex) is a breaker: runs never
        // start inside a digest, so step over the whole span. The bytes
        // stay in the `emitted` prefix and pass through verbatim.
        if bytes[pos] == b'~'
            && let Some(end) = token_span_end_at(text, pos)
        {
            pos = end;
            continue;
        }
        let Some(run_start) = next_phone_class(text, pos) else {
            break;
        };
        // A run starting inside a token digest is digest tail, not a
        // number's head: resume after the token so the composed run
        // (digest hex flowing through a separator into following text)
        // never forms and the real number keeps its own clean run.
        if let Some(end) = token_span_containing(text, run_start) {
            pos = end;
            continue;
        }
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
            // fresh "number" out of hex digits — unless the match starts
            // exactly where a token span ends (the token is a finished
            // unit, so the number after it is a new word, not an
            // identifier fragment). In real text a digit
            // run glued to hex/tilde is an identifier fragment,
            // the same reasoning as the bare cut.
            let nanp_shape = run.digits == 10 || (run.digits == 11 && run.first_digit == Some('1'));
            let match_start = run_start + run.leading_spaces;
            let clean_start = match_start == 0
                || !matches!(bytes[match_start - 1], b'~' | b'a'..=b'f')
                || token_ends_at(text, match_start);
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

// --- The keys rule: the credential scanner (the extension past the
// source's contact contract) -------------------------------------------

/// The key-tail charset every family shares (and the JWT segments'
/// base64url): `[A-Za-z0-9_-]`. ASCII only, so the scanner walks raw
/// bytes.
#[inline]
fn is_key_tail_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-')
}

/// The family table: (literal prefix, minimum tail length), ordered
/// LONGEST-PREFIX-FIRST — at one scan position the entries are tried in
/// this order and the first whose own grammar holds wins, so `sk-ant-`
/// outranks bare `sk-`, and a too-short `sk-ant-` tail FALLS THROUGH to
/// the `sk-` family, whose tail swallows the `ant-` spelling (still
/// scrubbed, the generic prefix). Entries whose prefixes share no head
/// (`github_pat_` vs `ghp_`) cannot tie at one position; the length-desc
/// order is the table's one total order anyway. This is the
/// evidence-backed closed set — the leaked-credential shapes five
/// private consumers evidenced; Slack `xox`, Stripe, and AWS `AKIA` are
/// deliberately absent (zero evidence), and growing the set is a
/// new-evidence decision, never a drive-by.
const KEY_FAMILIES: &[(&[u8], usize)] = &[
    (b"github_pat_", 22),
    (b"sk-svcacct-", 20),
    (b"sk-proj-", 20),
    (b"sk-ant-", 20),
    (b"azxdev_", 20),
    (b"ghp_", 36),
    (b"AIza", 35),
    (b"fw-", 20),
    (b"fw_", 20),
    (b"ak-", 20),
    (b"wk-", 20),
    (b"wd-", 43),
    (b"cn-", 20),
    (b"sk-", 20),
    (b"w-", 43),
];

/// The first bytes any family prefix (or the JWT marker) can start with:
/// the per-byte dispatch that keeps the walk linear-cheap on prose (one
/// `matches!` per byte; an anchor hit pays at most fifteen prefix
/// compares). Every table prefix and the `Bearer` marker begin with one
/// of these, so nothing is missed by the filter.
#[inline]
fn is_key_anchor(b: u8) -> bool {
    matches!(b, b'g' | b's' | b'a' | b'A' | b'f' | b'w' | b'c' | b'B')
}

/// The JWT family at one position: the literal marker `Bearer eyJ`,
/// then the segment grammar — three MAXIMAL `[A-Za-z0-9_-]+` runs
/// separated by single dots, the marker having consumed the first
/// segment's `eyJ` head (so the grammar's one-or-more needs at least one
/// more charset char before the first dot: the degenerate
/// `Bearer eyJ.a.b` is a non-match, and a second dot after a segment
/// ends the match attempt — an empty segment never matches). A bare
/// `eyJ` never matches anywhere: the family is MARKER-SCOPED because one
/// consumer's API legitimately carries eyJ-shaped non-secret cursors,
/// and redacting those would destroy the diagnostic this scrubber
/// exists to preserve. Returns the match END on success.
fn jwt_match_at(bytes: &[u8], start: usize) -> Option<usize> {
    const MARKER: &[u8] = b"Bearer eyJ";
    if !bytes[start..].starts_with(MARKER) {
        return None;
    }
    let mut i = start + MARKER.len();
    for seg in 0..3 {
        let run_start = i;
        while i < bytes.len() && is_key_tail_byte(bytes[i]) {
            i += 1;
        }
        if i == run_start {
            return None; // an empty segment: the grammar's `+` is one-or-more
        }
        if seg < 2 {
            if i >= bytes.len() || bytes[i] != b'.' {
                return None; // the single dot into the next segment
            }
            i += 1;
        }
    }
    Some(i)
}

/// The keys pass: every leftmost match of a family grammar becomes
/// `<family prefix>~<digest>` — the prefix VERBATIM (the non-secret half
/// that tells the operator WHICH credential to rotate: `sk-` vs
/// `sk-ant-` vs `github_pat_`), the digest over the FULL match (prefix +
/// tail). One linear walk: a byte no prefix can start with advances one
/// byte; an anchor byte pays the boundary check first — a prefix glued
/// to a preceding key-charset char is MID-TOKEN and never fires
/// (`xak-…`: in real text a key glued to a word is that word's fragment,
/// the same reasoning as the phone rule's clean-boundary cut, and it is
/// what keeps a second key glued to a token's digest hex from firing) —
/// then the table longest-first with fall-through, then the JWT marker
/// grammar (its `B` head shares no prefix with any table family). The
/// tail run is MAXIMAL: a key glued to further charset material is one
/// long key, over-redaction in the safe direction. `Cow::Borrowed` when
/// nothing matches.
fn scrub_keys_pass<'a>(text: &'a str, salt: &str) -> Cow<'a, str> {
    let bytes = text.as_bytes();
    let mut pos = 0;
    let mut emitted = 0;
    let mut out: Option<String> = None;
    while pos < bytes.len() {
        let b = bytes[pos];
        if !is_key_anchor(b) {
            pos += 1;
            continue;
        }
        if pos > 0 && is_key_tail_byte(bytes[pos - 1]) {
            pos += 1; // a mid-token prefix: the boundary rule
            continue;
        }
        let mut hit: Option<(&'static [u8], usize)> = None;
        for &(prefix, min_tail) in KEY_FAMILIES {
            if prefix[0] != b || !bytes[pos..].starts_with(prefix) {
                continue;
            }
            let tail_start = pos + prefix.len();
            let mut tail_end = tail_start;
            while tail_end < bytes.len() && is_key_tail_byte(bytes[tail_end]) {
                tail_end += 1;
            }
            if tail_end - tail_start >= min_tail {
                hit = Some((prefix, tail_end));
                break;
            }
            // A too-short tail falls through to the shorter prefixes.
        }
        if hit.is_none() && b == b'B' {
            hit = jwt_match_at(bytes, pos).map(|end| (b"Bearer".as_slice(), end));
        }
        let Some((prefix, end)) = hit else {
            pos += 1;
            continue;
        };
        let matched = &text[pos..end];
        let token = format!(
            "{}~{}",
            std::str::from_utf8(prefix).expect("family prefixes are ASCII"),
            token_digest(salt, matched)
        );
        let buf = out.get_or_insert_with(|| String::with_capacity(text.len()));
        buf.push_str(&text[emitted..pos]);
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

/// Which rules a call applies. `scrub_pii` with no rule is the
/// identity (the caller's `rules=[]`).
#[derive(Clone, Copy)]
pub struct PiiRules {
    pub email: bool,
    pub phone: bool,
    pub keys: bool,
}

impl PiiRules {
    /// The default `rules=None`: every rule in the canonical order —
    /// keys first, then email, then phone.
    pub const BOTH: PiiRules = PiiRules {
        email: true,
        phone: true,
        keys: true,
    };
}

/// One pipeline stage's `Cow` fold: an inactive stage passes its input
/// through untouched; an active stage over a `Borrowed` input runs on
/// the borrow, and over an `Owned` intermediate folds back into it —
/// fired, its own output; unfired, the intermediate itself. Neither
/// branch copies, the zero-copy discipline the two-stage spelling paid
/// for, kept whole as the pipeline grew to three stages.
fn fold_stage<'a>(
    mid: Cow<'a, str>,
    active: bool,
    pass: for<'x, 'y> fn(&'x str, &'y str) -> Cow<'x, str>,
    salt: &str,
) -> Cow<'a, str> {
    match mid {
        Cow::Borrowed(t) => {
            if active {
                pass(t, salt)
            } else {
                Cow::Borrowed(t)
            }
        }
        Cow::Owned(s) => {
            if !active {
                return Cow::Owned(s);
            }
            match pass(&s, salt) {
                Cow::Borrowed(_) => Cow::Owned(s),
                Cow::Owned(fin) => Cow::Owned(fin),
            }
        }
    }
}

/// Scrub `text` of credential and contact material: the keys
/// substitution over the whole string FIRST (a key's tail can spell a
/// dash-separated domestic phone run and a whole key an email local
/// part, so the credential must be eaten before the contact passes
/// scan), then the email substitution over its result, then the phone
/// substitution over that — each exactly once, no cascade — or
/// whichever subset `rules` selects. The contact rules digest with
/// `contact_salt` and the keys rule with `keys_salt`: the `salt=None`
/// per-rule defaults (`DEFAULT_SALT` / `KEYS_DEFAULT_SALT`) never alias
/// a contact digest with a key digest, an explicit string salts every
/// rule alike, and `""` is unsalted for every rule. `Cow::Borrowed` —
/// the identity path — exactly when no active rule matches.
pub fn scrub_pii<'a>(
    text: &'a str,
    rules: PiiRules,
    contact_salt: &str,
    keys_salt: &str,
) -> Cow<'a, str> {
    let after_keys = fold_stage(Cow::Borrowed(text), rules.keys, scrub_keys_pass, keys_salt);
    let after_email = fold_stage(after_keys, rules.email, scrub_email_pass, contact_salt);
    fold_stage(after_email, rules.phone, scrub_phone_pass, contact_salt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scrub(text: &str, rules: PiiRules, salt: &str) -> String {
        // (salt, salt): the unit lanes salt every rule alike — the
        // per-rule None split has its own pins below.
        scrub_pii(text, rules, salt, salt).into_owned()
    }

    fn digest(salt: &str, matched: &str) -> String {
        token_digest(salt, matched)
    }

    #[test]
    fn clean_input_is_identity() {
        let text = "plain prose, café, emoji \u{1f600}, digits 4096 and 1200";
        assert!(matches!(
            scrub_pii(text, PiiRules::BOTH, DEFAULT_SALT, KEYS_DEFAULT_SALT),
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
                    phone: false,
                    keys: false
                },
                "",
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
            keys: false,
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
            keys: false,
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
            keys: false,
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
            keys: false,
        };
        assert_eq!(scrub("a@b.cö", rules, ""), "a@b.cö");
        assert_eq!(scrub("a@ö.co", rules, ""), "a@ö.co");
    }

    #[test]
    fn rfc_quoted_locals_and_ip_domains_leak_whole() {
        // C1/C2: RFC quoted-string locals and IP-literal/dotted-quad
        // domains never match — the whole address survives. No grammar
        // widening (parity): canonicalize before scrub.
        let rules = PiiRules {
            email: true,
            phone: false,
            keys: false,
        };
        for text in [
            r#""user@name"@example.com"#,
            r#""a@b"@x.co"#,
            "user@[192.168.1.1]",
            "user@192.168.1.1",
        ] {
            assert!(
                matches!(scrub_pii(text, rules, "", ""), Cow::Borrowed(_)),
                "{text}"
            );
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
            keys: false,
        };
        let dots: String = "a.".repeat(50_000);
        let non_match = format!("x@{dots}a");
        assert!(matches!(
            scrub_pii(&non_match, rules, "", ""),
            Cow::Borrowed(_)
        ));
        let matched = format!("x@{dots}zz");
        let expected = format!("@{dots}zz~{}", digest("", &matched));
        assert_eq!(scrub(&matched, rules, ""), expected);
    }

    #[test]
    fn phone_tokens_keep_the_first_three_code_points() {
        let rules = PiiRules {
            email: false,
            phone: true,
            keys: false,
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
            keys: false,
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
            keys: false,
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
            keys: false,
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
            keys: false,
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
            keys: false,
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
            scrub_pii(&twice, PiiRules::BOTH, "", ""),
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
            scrub_pii(&twice, PiiRules::BOTH, "", ""),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn an_email_token_is_a_breaker_before_an_adjacent_domestic_number() {
        // P0: the email pass emits `@domain~<12hex>`; the digest tail
        // must not compose with a following domestic number into one
        // spent-whole run (silent under-redaction). The token is a
        // breaker: each number scrubs exactly, and the output converges.
        for (text, matched) in [
            (
                "candidate ada+tag@azx.io 415-555-2671 no answer",
                "415-555-2671",
            ),
            ("fungai.chetima@example.com-415-555-2671", "-415-555-2671"),
        ] {
            let once = scrub(text, PiiRules::BOTH, "");
            assert!(
                !once.contains(matched),
                "digest tail swallowed the adjacent number in {once:?}"
            );
            assert!(
                once.contains(&format!("{}~", &matched[..3])),
                "the number did not scrub exactly in {once:?}"
            );
            let twice = scrub(&once, PiiRules::BOTH, "");
            assert!(
                matches!(scrub_pii(&twice, PiiRules::BOTH, "", ""), Cow::Borrowed(_)),
                "no convergence in {twice:?}"
            );
        }
    }

    #[test]
    fn phone_only_stays_strictly_idempotent_with_domestic_matches() {
        let once = scrub("(415) 555-2671 415-555-2672 +14155552673", phone_only(), "");
        assert!(matches!(
            scrub_pii(&once, phone_only(), "", ""),
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
            keys: false,
        };
        let once = scrub("+14155552671 +14155552672", rules, "");
        assert!(matches!(scrub_pii(&once, rules, "", ""), Cow::Borrowed(_)));
    }

    #[test]
    fn adjacent_email_matches_re_fire_once_then_converge() {
        let rules = PiiRules {
            email: true,
            phone: false,
            keys: false,
        };
        let once = scrub("a@b.co9@x.yz", rules, "");
        let twice = scrub(&once, rules, "");
        assert_ne!(once, twice);
        assert!(matches!(scrub_pii(&twice, rules, "", ""), Cow::Borrowed(_)));
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
            keys: false,
        };
        let once = scrub("x@b.co@w.vu", rules, "");
        let twice = scrub(&once, rules, "");
        assert_ne!(once, twice);
        assert!(matches!(scrub_pii(&twice, rules, "", ""), Cow::Borrowed(_)));
    }

    #[test]
    fn fuzz_crash_adjacent_embedded_email_match_converges() {
        // fuzz-smoke crash-effd9780 (`A@a.Az.A@a.Az`, raw bytes with a
        // trailing invalid-UTF-8 `\xccA`): adjacent email matches where
        // the second span's string (`.A@a.Az`) embeds the first's
        // (`A@a.Az`). The transform was correct — the abort came from
        // the harness's per-string survivor accounting, which ignored
        // cross-string coverage — pinned here so the shape never
        // regresses: exact tokens, convergence, and phone-only identity
        // (no digits, so the phone pass must stay borrowed).
        let once = scrub("A@a.Az.A@a.Az", PiiRules::BOTH, "");
        assert_eq!(
            once,
            format!(
                "@a.Az~{}@a.Az~{}",
                digest("", "A@a.Az"),
                digest("", ".A@a.Az")
            )
        );
        let twice = scrub(&once, PiiRules::BOTH, "");
        assert!(
            matches!(scrub_pii(&twice, PiiRules::BOTH, "", ""), Cow::Borrowed(_)),
            "no convergence in {twice:?}"
        );
        assert!(matches!(
            scrub_pii("A@a.Az.A@a.Az", phone_only(), "", ""),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn tokens_are_fixed_points() {
        for token in [
            format!("@b.co~{}", digest("", "a@b.co")),
            format!("+14~{}", digest("", "+14155552671")),
            format!("+1 ~{}", digest("", "+1 (415) 555-2671")),
        ] {
            assert!(
                matches!(scrub_pii(&token, PiiRules::BOTH, "", ""), Cow::Borrowed(_)),
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

    // --- The keys rule (the credential extension past the source) --------

    fn keys_only() -> PiiRules {
        PiiRules {
            email: false,
            phone: false,
            keys: true,
        }
    }

    /// A deterministic 62-char-alphabet cycle, letters-only below 53: no
    /// accidental contact shape rides inside a battery key.
    fn key_tail(n: usize) -> String {
        const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        (0..n)
            .map(|i| ALPHABET[i % ALPHABET.len()] as char)
            .collect()
    }

    const JWT: &str = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.\
eyJzdWIiOiIxMjM0NTY3ODkwIn0.\
dozjgNryP4J3jVmNHc0FKW3YtV9zZ2YwXqR8uT1aB5cDe";

    fn key_vectors() -> Vec<(String, &'static str)> {
        let t48 = key_tail(48);
        vec![
            (format!("sk-{t48}"), "sk-"),
            (format!("sk-proj-{t48}"), "sk-proj-"),
            (format!("sk-svcacct-{t48}"), "sk-svcacct-"),
            (format!("sk-ant-api03-{}", key_tail(95)), "sk-ant-"),
            (format!("AIza{}", key_tail(35)), "AIza"),
            (format!("fw-{t48}"), "fw-"),
            (format!("fw_{t48}"), "fw_"),
            (format!("ak-{t48}"), "ak-"),
            (format!("wk-{t48}"), "wk-"),
            (format!("ghp_{}", key_tail(36)), "ghp_"),
            (format!("github_pat_{}", key_tail(22)), "github_pat_"),
            (format!("azxdev_{}", key_tail(20)), "azxdev_"),
            (format!("wd-{}", key_tail(43)), "wd-"),
            (format!("w-{}", key_tail(43)), "w-"),
            (format!("cn-{}", key_tail(20)), "cn-"),
            (JWT.to_string(), "Bearer"),
        ]
    }

    #[test]
    fn every_key_family_scrubs_with_its_verbatim_prefix() {
        for (key, prefix) in key_vectors() {
            assert_eq!(
                scrub(&key, keys_only(), ""),
                format!("{prefix}~{}", digest("", &key)),
                "{key}"
            );
            // The full pipeline composes identically: keys first, and no
            // contact shape rides inside a battery key.
            assert_eq!(
                scrub(&key, PiiRules::BOTH, ""),
                format!("{prefix}~{}", digest("", &key))
            );
        }
    }

    #[test]
    fn key_non_matches_stay_identity() {
        let t48 = key_tail(48);
        for text in [
            format!("sk-{}", key_tail(19)),  // one under
            "sk-".to_string(),               // the bare prefix
            format!("SKI-{t48}"),            // uppercase
            format!("AIza{}", key_tail(34)), // one under
            format!("ghp_{}", key_tail(35)), // one under
            format!("github_pat_{}", key_tail(21)),
            format!("azxdev_{}", key_tail(19)),
            format!("wd-{}", key_tail(42)),
            format!("w-{}", key_tail(42)),
            format!("cn-{}", key_tail(19)),
            format!("fw-{}", key_tail(19)),
            format!("ak-{}", key_tail(19)),
            format!("wk-{}", key_tail(19)),
            format!("xak-{t48}"), // mid-token prefix (the boundary rule)
            "bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.c2ln".to_string(), // lowercase marker
            // The non-secret cursor class: eyJ-shaped, never behind Bearer.
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.c2ln.dozj".to_string(),
            // Two segments only.
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.c2ln".to_string(),
            // The degenerate first segment (no charset char after eyJ).
            "Bearer eyJ.a.b.c".to_string(),
            // An empty middle segment (the double dot).
            "Bearer eyJhbGciOiJIUzI1Ni..dozjgNryP4J3".to_string(),
        ] {
            assert!(
                matches!(scrub_pii(&text, PiiRules::BOTH, "", ""), Cow::Borrowed(_)),
                "{text}"
            );
        }
    }

    #[test]
    fn a_fourth_jwt_segment_survives_the_three_segment_grammar() {
        // The grammar is exactly three maximal runs: `Bearer eyJa.b.c.d`
        // scrubs through the third segment and `.d` survives (the tail
        // beyond the match is not the grammar's business).
        assert_eq!(
            scrub("Bearer eyJa.b.c.d", keys_only(), ""),
            format!("Bearer~{}.d", digest("", "Bearer eyJa.b.c"))
        );
    }

    #[test]
    fn the_longest_prefix_wins_and_falls_through() {
        // `sk-ant-` outranks bare `sk-` when its own grammar holds...
        let key = format!("sk-ant-{}", key_tail(40));
        assert_eq!(
            scrub(&key, keys_only(), ""),
            format!("sk-ant-~{}", digest("", &key))
        );
        // ...and one under the Anthropic minimum falls through to the
        // bare `sk-` family, whose tail swallows the `ant-` spelling.
        let short = format!("sk-ant-{}", key_tail(19));
        assert_eq!(
            scrub(&short, keys_only(), ""),
            format!("sk-~{}", digest("", &short))
        );
    }

    #[test]
    fn a_prefix_glued_to_a_charset_char_is_mid_token() {
        let first = scrub(&format!("sk-{}", key_tail(48)), keys_only(), "");
        let second = format!("ghp_{}", key_tail(36));
        // A key glued to a preceding word (`xak-...`) is that token's
        // fragment; a key glued to a token's digest hex is mid-token the
        // same way — conservative, and what keeps the pass idempotent
        // against composed input. Word-separated keys both fire.
        assert_eq!(
            scrub(&format!("xak-{}", key_tail(48)), keys_only(), ""),
            format!("xak-{}", key_tail(48))
        );
        assert_eq!(
            scrub(&format!("{first}{second}"), keys_only(), ""),
            format!("{first}{second}")
        );
        assert_eq!(
            scrub(&format!("{first} {second}"), keys_only(), ""),
            format!("{first} ghp_~{}", digest("", &second))
        );
    }

    #[test]
    fn the_tail_run_is_maximal() {
        // A second key glued to the first is charset material for its
        // tail: ONE long match, over-redaction in the safe direction.
        let a = format!("sk-{}", key_tail(48));
        let b = format!("ghp_{}", key_tail(36));
        assert_eq!(
            scrub(&format!("{a}{b}"), keys_only(), ""),
            format!("sk-~{}", digest("", &format!("{a}{b}")))
        );
    }

    #[test]
    fn key_tokens_are_fixed_points_every_family() {
        // The idempotence-by-construction argument, verified per family:
        // the token's prefix ends `-`/`_` (or is `AIza`/`Bearer`), the
        // byte after it is `~` (never tail charset), and no family
        // prefix can be spelled inside 12 lowercase digest hex — so the
        // full pipeline never fires on a key token.
        for (key, prefix) in key_vectors() {
            let once = scrub(&key, PiiRules::BOTH, "");
            assert_eq!(once, format!("{prefix}~{}", digest("", &key)));
            assert!(
                matches!(scrub_pii(&once, PiiRules::BOTH, "", ""), Cow::Borrowed(_)),
                "{once}"
            );
        }
    }

    #[test]
    fn keys_run_before_email_and_phone() {
        // The pass-order pins: a key-shaped email local part is eaten by
        // the keys pass (the email pass then tokens the key token's
        // digest hex as a fresh local part, the documented safe corner),
        // and a key's dash-separated ten-digit run never surfaces to the
        // domestic phone matcher (the phone pass sees only the token).
        let key = format!("sk-{}", key_tail(48));
        let at = format!("{key}@x.co");
        let hex12 = digest("", &key);
        assert_eq!(
            scrub(&at, PiiRules::BOTH, ""),
            format!("sk-~@x.co~{}", digest("", &format!("{hex12}@x.co")))
        );
        let phonekey = format!("sk-proj-415-555-2671{}", key_tail(20));
        assert_eq!(
            scrub(
                &format!("leaked {phonekey} in an error"),
                PiiRules::BOTH,
                ""
            ),
            format!("leaked sk-proj-~{} in an error", digest("", &phonekey))
        );
        // The contrast: without the keys rule the same text loses its
        // digit run to the domestic matcher — the order is load-bearing.
        let phone_only_out = scrub(
            &format!("leaked {phonekey} in an error"),
            PiiRules {
                email: false,
                phone: true,
                keys: false,
            },
            "",
        );
        assert!(phone_only_out.contains("-41~"));
    }

    #[test]
    fn a_number_after_a_key_token_keeps_its_clean_run() {
        // A key token's `~` + 12 hex is a token span for the phone
        // pass's existing breaker, and the byte after it is a clean
        // boundary: the number after a scrubbed key scrubs exactly.
        let key = format!("fw-{}", key_tail(48));
        let text = format!("{key} 415-555-2671");
        assert_eq!(
            scrub(&text, PiiRules::BOTH, ""),
            format!(
                "fw-~{} 415~{}",
                digest("", &key),
                digest("", "415-555-2671")
            )
        );
    }

    #[test]
    fn the_default_salts_split_per_rule() {
        // salt=None resolves per rule: the contact tag and the keys tag
        // are different constants, and the pipeline threads them
        // independently — a key digest can never alias a contact digest
        // at the default settings.
        assert_ne!(DEFAULT_SALT, KEYS_DEFAULT_SALT);
        let key = format!("sk-{}", key_tail(48));
        let text = format!("a@b.co {key}");
        let out = scrub_pii(
            text.as_str(),
            PiiRules::BOTH,
            DEFAULT_SALT,
            KEYS_DEFAULT_SALT,
        )
        .into_owned();
        assert_eq!(
            out,
            format!(
                "@b.co~{} sk-~{}",
                digest(DEFAULT_SALT, "a@b.co"),
                digest(KEYS_DEFAULT_SALT, &key)
            )
        );
        // An explicit string salts every rule alike; "" is unsalted for
        // every rule.
        let site = scrub_pii(text.as_str(), PiiRules::BOTH, "site", "site").into_owned();
        assert_eq!(
            site,
            format!(
                "@b.co~{} sk-~{}",
                digest("site", "a@b.co"),
                digest("site", &key)
            )
        );
    }
}
