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
//!   a literal prefix plus a minimal tail consumed maximally (the shared
//!   `[A-Za-z0-9_-]` alphabet except where the family names its own):
//!   OpenAI `sk-`/`sk-proj-`/`sk-svcacct-` (20+), Anthropic `sk-ant-`
//!   (20+), Google `AIza` (35+), Fireworks `fw-`/`fw_` (20+), Modal
//!   `ak-`/`wk-` (20+), GitHub `ghp_` (36+) and `github_pat_` (22+),
//!   the minted shapes `azxdev_` (20+), `wd-` (43+), `w-` (43+),
//!   `cn-` (20+), MARKER-SCOPED JWTs — `Bearer eyJ` plus three maximal
//!   base64url segments, single-dot separated (a bare `eyJ` never
//!   matches: one consumer's API legitimately carries eyJ-shaped
//!   non-secret cursors, and redacting those would destroy the
//!   diagnostic this scrubber exists to preserve) — AWS `AKIA`/`ASIA` +
//!   `[0-9A-Z]{16,}` (the access-key ID; AWS SECRET keys carry no
//!   prefix and stay a documented exclusion), xAI `xai-` (20+), GCP
//!   OAuth `ya29.` (20+), the PEM SPAN family (`-----BEGIN <words>
//!   PRIVATE KEY-----` … `-----END <same words> PRIVATE KEY-----`, both
//!   markers required; the PKCS#8 bare header carries no algorithm
//!   words and stays a documented exclusion), and Azure `AccountKey=` +
//!   `[A-Za-z0-9+/=]{40,}` (Azure client secrets carry no distinctive
//!   prefix and stay a documented exclusion, Mistral keys with them).
//!   The rule takes a per-family selection (`families=`: `None` is
//!   every family this version knows; a list selects exactly those —
//!   the binding walks the names into a private mask, and an unselected
//!   family whose grammar holds is detected but preserved verbatim).
//!   The leak vector is
//!   the error text itself: provider and platform error strings can
//!   quote the credential back — five private consumers evidenced, the
//!   strongest a platform whose own code comments that a vendor auth
//!   failure "can quote the key" and keeps the full text in an
//!   admin-served ledger. Slack `xox` and Stripe shapes are deliberately
//!   absent (zero evidence): growing the set is a new-evidence decision,
//!   never a drive-by. Three discipline rules:
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
//!   construction, every family: the prefix ends in `-`/`_`/`=`/`.`
//!   (or is a bare head — `AIza`, `AKIA`/`ASIA`, `Bearer`, `PEM`), the
//!   byte after it is `~`, never tail charset, so no family can re-fire
//!   at the token's own head — and no family prefix can be spelled
//!   inside 12 lowercase digest hex (the distinctive characters — `z`
//!   in `AIza`/`azxdev_`, `k` in `ak-`, `n` in `cn-`, `w` in `fw-`/`wk-`,
//!   `h` in `ghp_`, `y` in `ya29.`, the uppercase in `AKIA`/`ASIA`/
//!   `PEM`/`AccountKey=`, the space in `Bearer eyJ`, the `-`/`.`/`=`
//!   tails — are all outside `[0-9a-f]`), so the digest half is inert
//!   too. A key token's `~` + 12 hex is a token span for the phone
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
//! exclusion is deliberate, and a new family is a new-evidence
//! decision; the named exclusions are prefix-less shapes — AWS secret
//! keys, Azure client secrets, Mistral keys — plus Slack `xox` and
//! Stripe, all on zero distinctive-prefix evidence), and the kept
//! family prefix is a coarse provider label, not a credential. For
//! adversarial threat, map Zs/Zl/Zp plus `\t\n\r\f\v` to U+0020 and
//! canonicalize separators/domains before scrub (`tors.nfkc` alone is
//! insufficient); see `docs/api.md`'s scrub_pii section for the full
//! residual-risk list and the canonicalization code block.
//!
//! Performance: one linear pass per rule — `memchr`-anchored for the
//! `@` and phone-class scans, a first-byte-dispatched table walk for the
//! key families (one `matches!` per byte on prose, at most twenty
//! prefix compares on an anchor hit) plus the marker grammars on their
//! disjoint heads — `Cow::Borrowed`
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

/// One api-key family: the evidence-backed closed set's identity, in the
/// canonical order the `families=` names and the `tors.KEY_FAMILIES`
/// tuple mirror. The discriminant IS the selection-mask bit (bit i is
/// `ALL[i]`): thirteen families fit a `u16` with room to grow, and the
/// binding walks names into that mask — an implementation detail, never
/// a public bit arithmetic surface.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KeyFamily {
    OpenAi,
    Anthropic,
    Google,
    Fireworks,
    Modal,
    GitHub,
    Minted,
    Jwt,
    Aws,
    Xai,
    GcpOauth,
    Pem,
    Azure,
}

impl KeyFamily {
    /// Every family in the canonical order: the scanner table's order,
    /// the `families=` closed set's order, the `KEY_FAMILIES` tuple's
    /// order — one order everywhere, so a new family has exactly one
    /// place to land.
    pub const ALL: [KeyFamily; 13] = [
        KeyFamily::OpenAi,
        KeyFamily::Anthropic,
        KeyFamily::Google,
        KeyFamily::Fireworks,
        KeyFamily::Modal,
        KeyFamily::GitHub,
        KeyFamily::Minted,
        KeyFamily::Jwt,
        KeyFamily::Aws,
        KeyFamily::Xai,
        KeyFamily::GcpOauth,
        KeyFamily::Pem,
        KeyFamily::Azure,
    ];

    /// The `families=` name: lowercase, the tuple's spelling.
    pub const fn name(&self) -> &'static str {
        match self {
            KeyFamily::OpenAi => "openai",
            KeyFamily::Anthropic => "anthropic",
            KeyFamily::Google => "google",
            KeyFamily::Fireworks => "fireworks",
            KeyFamily::Modal => "modal",
            KeyFamily::GitHub => "github",
            KeyFamily::Minted => "minted",
            KeyFamily::Jwt => "jwt",
            KeyFamily::Aws => "aws",
            KeyFamily::Xai => "xai",
            KeyFamily::GcpOauth => "gcp_oauth",
            KeyFamily::Pem => "pem",
            KeyFamily::Azure => "azure",
        }
    }

    /// The selection-mask bit for this family.
    pub(crate) const fn bit(&self) -> u16 {
        1u16 << (*self as u16)
    }
}

/// The canonical family-name tuple in `ALL` order: the single source
/// the binding's `KEY_FAMILIES` export and the unknown-name `ValueError`
/// both spell from, so the message can never drift from the tuple.
pub const KEY_FAMILY_NAMES: [&str; 13] = [
    "openai",
    "anthropic",
    "google",
    "fireworks",
    "modal",
    "github",
    "minted",
    "jwt",
    "aws",
    "xai",
    "gcp_oauth",
    "pem",
    "azure",
];

/// Every family this version knows: the `families=None` selection and
/// the `PiiRules::BOTH` default. A new family sets one more bit here —
/// and nowhere else in the selection path.
pub const KEY_FAMILY_MASK_ALL: u16 = (1u16 << KeyFamily::ALL.len()) - 1;

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
fn email_pass_impl<'a>(text: &'a str, salt: &str, rec: Option<&mut PassRec>) -> Cow<'a, str> {
    let bytes = text.as_bytes();
    let mut pos = 0; // where the search for the next `@` resumes
    let mut emitted = 0; // the prefix of `text` already pushed to `out`
    let mut out: Option<String> = None;
    let mut rec = rec;
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
        if let Some(r) = rec.as_deref_mut() {
            // The verbatim head is `@domain` itself, mapping to the
            // match's own tail `[at..end)`.
            r.edits.push(PassEdit {
                start: local_start,
                end,
                token: token.clone(),
                verbatim_src: at,
                verbatim_len: end - at,
            });
            r.kinds.push(SpanKind::Email);
        }
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
fn phone_pass_impl<'a>(text: &'a str, salt: &str, rec: Option<&mut PassRec>) -> Cow<'a, str> {
    let bytes = text.as_bytes();
    let mut pos = 0;
    let mut emitted = 0;
    let mut out: Option<String> = None;
    let mut rec = rec;
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
        if let Some(r) = rec.as_deref_mut() {
            // The verbatim head is the match's own first three code
            // points, mapping to the match's head.
            r.edits.push(PassEdit {
                start,
                end,
                token: token.clone(),
                verbatim_src: start,
                verbatim_len: prefix_end - start,
            });
            r.kinds.push(SpanKind::Phone);
        }
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

/// The tail alphabet a family consumes maximally. Most families share
/// the key-tail charset; the marker families carry their own — the
/// access-key ID alphabet (`[0-9A-Z]`, no lowercase anywhere in it) and
/// the connection-string secret alphabet (`[A-Za-z0-9+/=]`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum KeyTailClass {
    Shared,
    Aws,
    Azure,
}

/// One tail-class predicate per class, ASCII-only like the shared one.
#[inline]
fn is_aws_tail_byte(b: u8) -> bool {
    b.is_ascii_uppercase() || b.is_ascii_digit()
}

#[inline]
fn is_azure_tail_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=')
}

#[inline]
fn tail_predicate(class: KeyTailClass) -> fn(u8) -> bool {
    match class {
        KeyTailClass::Shared => is_key_tail_byte,
        KeyTailClass::Aws => is_aws_tail_byte,
        KeyTailClass::Azure => is_azure_tail_byte,
    }
}

/// The family table: (literal prefix, family, minimum tail length, tail
/// class), ordered LONGEST-PREFIX-FIRST — at one scan position the
/// entries are tried in this order and the first whose own grammar holds
/// wins, so `sk-ant-` outranks bare `sk-`, and a too-short `sk-ant-`
/// tail FALLS THROUGH to the `sk-` family, whose tail swallows the
/// `ant-` spelling (still scrubbed, the generic prefix). Entries whose
/// prefixes share no head (`github_pat_` vs `ghp_`) cannot tie at one
/// position; the length-desc order is the table's one total order
/// anyway. This is the evidence-backed closed set — the
/// leaked-credential shapes the consumers evidenced; Slack `xox` and
/// Stripe stay deliberately absent (zero evidence), and growing the set
/// is a new-evidence decision, never a drive-by. The JWT and PEM
/// families live outside this table (their grammars are marker/span
/// shapes, not prefix-plus-tail), tried after it on their disjoint head
/// bytes (`B`/`-`, which no table prefix starts with).
const KEY_FAMILIES: &[(&[u8], KeyFamily, usize, KeyTailClass)] = &[
    (b"github_pat_", KeyFamily::GitHub, 22, KeyTailClass::Shared),
    (b"sk-svcacct-", KeyFamily::OpenAi, 20, KeyTailClass::Shared),
    (b"AccountKey=", KeyFamily::Azure, 40, KeyTailClass::Azure),
    (b"sk-proj-", KeyFamily::OpenAi, 20, KeyTailClass::Shared),
    (b"sk-ant-", KeyFamily::Anthropic, 20, KeyTailClass::Shared),
    (b"azxdev_", KeyFamily::Minted, 20, KeyTailClass::Shared),
    (b"ya29.", KeyFamily::GcpOauth, 20, KeyTailClass::Shared),
    (b"ghp_", KeyFamily::GitHub, 36, KeyTailClass::Shared),
    (b"AIza", KeyFamily::Google, 35, KeyTailClass::Shared),
    (b"AKIA", KeyFamily::Aws, 16, KeyTailClass::Aws),
    (b"ASIA", KeyFamily::Aws, 16, KeyTailClass::Aws),
    (b"xai-", KeyFamily::Xai, 20, KeyTailClass::Shared),
    (b"fw-", KeyFamily::Fireworks, 20, KeyTailClass::Shared),
    (b"fw_", KeyFamily::Fireworks, 20, KeyTailClass::Shared),
    (b"ak-", KeyFamily::Modal, 20, KeyTailClass::Shared),
    (b"wk-", KeyFamily::Modal, 20, KeyTailClass::Shared),
    (b"wd-", KeyFamily::Minted, 43, KeyTailClass::Shared),
    (b"cn-", KeyFamily::Minted, 20, KeyTailClass::Shared),
    (b"sk-", KeyFamily::OpenAi, 20, KeyTailClass::Shared),
    (b"w-", KeyFamily::Minted, 43, KeyTailClass::Shared),
];

/// The first bytes any family prefix (or the JWT/PEM marker) can start
/// with: the per-byte dispatch that keeps the walk linear-cheap on
/// prose (one `matches!` per byte; an anchor hit pays at most twenty
/// prefix compares). Every table prefix and both markers begin with one
/// of these, so nothing is missed by the filter.
#[inline]
fn is_key_anchor(b: u8) -> bool {
    matches!(
        b,
        b'g' | b's' | b'a' | b'A' | b'f' | b'w' | b'c' | b'B' | b'x' | b'y' | b'-'
    )
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

/// The PEM label word class: `[A-Za-z0-9]`, ASCII only, so the marker
/// parse walks raw bytes.
#[inline]
fn is_pem_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric()
}

/// One PEM label production — `<words> PRIVATE KEY-----` — at
/// `words_start` (the first byte after `-----BEGIN ` or `-----END `):
/// one-or-more `[A-Za-z0-9]+` words, single-space separated, then the
/// fixed ` PRIVATE KEY-----` tail, the FIRST position where a complete
/// word is followed by the tail winning (so `RSA` in
/// `RSA PRIVATE KEY-----` is the words; a doubled space, a non-word
/// byte, or a missing tail fails the marker — and the PKCS#8 bare
/// `BEGIN PRIVATE KEY` header with it: no algorithm words, a
/// new-evidence decision like any other family shape). Returns the words
/// end and the marker end.
fn pem_marker_end(bytes: &[u8], words_start: usize) -> Option<(usize, usize)> {
    const TAIL: &[u8] = b" PRIVATE KEY-----";
    let mut p = words_start;
    loop {
        let w = p;
        while p < bytes.len() && is_pem_word_byte(bytes[p]) {
            p += 1;
        }
        if w == p {
            return None; // an empty word: the label ran out or doubled its space
        }
        if bytes[p..].starts_with(TAIL) {
            return Some((p, p + TAIL.len()));
        }
        if p >= bytes.len() || bytes[p] != b' ' {
            return None;
        }
        p += 1;
    }
}

/// The PEM family at one position, over a hoisted END-literal index
/// plus a per-words failure memo: `-----BEGIN ` + algorithm words +
/// ` PRIVATE KEY-----`, then any bytes including newlines (the key
/// body), then `-----END ` + the SAME words + ` PRIVATE KEY-----`.
/// Both markers required — an unterminated BEGIN is a documented
/// non-match — and the first `-----END ` at or past the body start
/// whose words parse AND equal the BEGIN's terminates the block (an
/// unparseable or mismatched END is skipped, a later matching one
/// still terminates). The whole block is the match; the token prefix
/// is the constant `PEM` (never input material, so the report's offset
/// map collapses the whole token to the replaced span's end). Returns
/// the match END on success.
/// `index` is the pass's one `-----END ` sweep (see `pem_end_index`),
/// built lazily here on the first VALID header (headers are rare;
/// dashes are not — an eager sweep would tax every dash-bearing
/// input). Each BEGIN then binary-searches its body start and verifies
/// only true literals in increasing position order — the same candidate
/// order as a forward scan, without the per-anchor re-scan. `failed`
/// maps each seen words value to the furthest END index already
/// verified-and-failed for it: verification is deterministic (same
/// bytes, same words, same verdict), and BEGINs arrive in increasing
/// position order, so a later BEGIN resumes past its words' failures
/// instead of re-verifying them — each (words, candidate) pair is
/// verified at most once, and a mismatched-END flood costs O(ENDs),
/// not O(BEGINs × ENDs). A verifying END returns immediately without
/// touching the memo (a later BEGIN may legitimately re-verify it).
fn pem_match_at(
    bytes: &[u8],
    start: usize,
    index: &mut Option<Vec<usize>>,
    failed: &mut Vec<(Vec<u8>, usize)>,
) -> Option<usize> {
    const BEGIN: &[u8] = b"-----BEGIN ";
    const END_HEAD: &[u8] = b"-----END ";
    if !bytes[start..].starts_with(BEGIN) {
        return None;
    }
    let (words_end, body_start) = pem_marker_end(bytes, start + BEGIN.len())?;
    let words = &bytes[start + BEGIN.len()..words_end];
    let ends = index.get_or_insert_with(|| pem_end_index(bytes));
    let resume = failed
        .iter()
        .find(|(w, _)| w.as_slice() == words)
        .map(|(_, i)| i + 1)
        .unwrap_or(0);
    let mut idx = ends.partition_point(|&e| e < body_start).max(resume);
    while idx < ends.len() {
        let cand = ends[idx];
        if let Some((end_words_end, end)) = pem_marker_end(bytes, cand + END_HEAD.len())
            && &bytes[cand + END_HEAD.len()..end_words_end] == words
        {
            return Some(end);
        }
        // Unparseable or mismatched END: record the failure for these
        // words and keep searching.
        match failed.iter_mut().find(|(w, _)| w.as_slice() == words) {
            Some(slot) => slot.1 = idx,
            None => failed.push((words.to_vec(), idx)),
        }
        idx += 1;
    }
    None
}

/// The pass's one `-----END ` sweep: every position where the END
/// literal opens, in increasing order. Built lazily on the first valid
/// PEM header (headers are rare; dashes are not — an eager sweep would
/// tax every dash-bearing input), then shared by every BEGIN in the
/// pass: the per-anchor cost drops from a full-suffix re-scan to a
/// binary search plus one verification per true literal.
fn pem_end_index(bytes: &[u8]) -> Vec<usize> {
    const END_HEAD: &[u8] = b"-----END ";
    let mut ends = Vec::new();
    let mut q = 0;
    while q < bytes.len() {
        let Some(rel) = memchr(b'-', &bytes[q..]) else {
            break;
        };
        let cand = q + rel;
        q = cand + 1; // one-char steps: overlapping markers stay exact
        if bytes[cand..].starts_with(END_HEAD) {
            ends.push(cand);
        }
    }
    ends
}

/// The keys pass: every leftmost match of a family grammar becomes
/// `<family prefix>~<digest>` — the prefix VERBATIM (the non-secret half
/// that tells the operator WHICH credential to rotate: `sk-` vs
/// `sk-ant-` vs `github_pat_`), the digest over the FULL match (prefix +
/// tail) — except PEM, whose token prefix is the constant `PEM`. One
/// linear walk: a byte no prefix can start with advances one byte; an
/// anchor byte pays the boundary check first — a prefix glued to a
/// preceding key-charset char is MID-TOKEN and never fires (`xak-…`: in
/// real text a key glued to a word is that word's fragment, the same
/// reasoning as the phone rule's clean-boundary cut, and it is what
/// keeps a second key glued to a token's digest hex from firing) — then
/// the table longest-first with fall-through (a too-short tail falls
/// through to the shorter prefixes), then the JWT marker grammar (its
/// `B` head shares no prefix with any table family), then the PEM span
/// grammar (its `-` head likewise). The tail run is MAXIMAL: a key
/// glued to further charset material is one long key, over-redaction in
/// the safe direction. `Cow::Borrowed` when nothing matches.
/// The report's span kind: the contact rule that fired, or the key
/// family that did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpanKind {
    Email,
    Phone,
    Key(KeyFamily),
}

/// One substitution a recorded pass made, in the pass's OWN input
/// coordinates: the replaced span, the token emitted, and the token's
/// verbatim head — the token's `[0..verbatim_len)` maps affinely to
/// `[verbatim_src..verbatim_src + verbatim_len)` (the bytes the token
/// kept: a key's family prefix, a phone token's three code points, an
/// email token's `@domain`), while the `~` + digest half has no
/// preimage and collapses to the replaced span's end (see `map_back`).
/// PEM tokens keep nothing (`verbatim_len` 0: `PEM` is a constant, and
/// no later match can land inside the head anyway — every token's `~`
/// blocks the email walk-back).
pub struct PassEdit {
    pub start: usize,
    pub end: usize,
    pub token: String,
    pub verbatim_src: usize,
    pub verbatim_len: usize,
}

/// What a recorded pass hands the report: its substitutions (in
/// pass-input coordinates) with their kinds, plus — keys pass only —
/// the per-family SKIPPED counts (detected-but-unselected matches, in
/// `ALL` order; zeros elsewhere).
#[derive(Default)]
pub struct PassRec {
    pub edits: Vec<PassEdit>,
    pub kinds: Vec<SpanKind>,
    pub skipped: [usize; 13],
}

/// The keys pass core: every leftmost match of a family grammar over
/// `text`, `mask`-selected. A match whose family is selected becomes
/// its token (recorded into `rec` when present); a match whose family
/// is NOT selected is DETECTED but not redacted — the span is spent
/// whole and preserved verbatim (counted in `rec.skipped`, never
/// re-scanned inside: the preserved span stays whole, the "we preserved
/// a JWT, log it separately" semantic) — and a family whose grammar
/// fails (too-short tail) falls through to the shorter prefixes exactly
/// as before. `Cow::Borrowed` when nothing matched (selected or not: a
/// skipped span still passes its bytes through untouched).
fn keys_pass_impl<'a>(
    text: &'a str,
    salt: &str,
    mask: u16,
    rec: Option<&mut PassRec>,
) -> Cow<'a, str> {
    let bytes = text.as_bytes();
    let mut pos = 0;
    let mut emitted = 0;
    let mut out: Option<String> = None;
    let mut rec = rec;
    let mut pem_ends: Option<Vec<usize>> = None;
    let mut pem_failed: Vec<(Vec<u8>, usize)> = Vec::new();
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
        // (family, verbatim prefix length in the token, match end). The
        // verbatim length is the matched head's own length for the
        // table/JWT families (the token keeps the head verbatim) and 0
        // for PEM (the token's `PEM` is a constant).
        let mut hit: Option<(KeyFamily, usize, usize)> = None;
        for &(prefix, family, min_tail, class) in KEY_FAMILIES {
            if prefix[0] != b || !bytes[pos..].starts_with(prefix) {
                continue;
            }
            let tail_start = pos + prefix.len();
            let mut tail_end = tail_start;
            let class_ok = tail_predicate(class);
            while tail_end < bytes.len() && class_ok(bytes[tail_end]) {
                tail_end += 1;
            }
            if tail_end - tail_start >= min_tail {
                hit = Some((family, prefix.len(), tail_end));
                break;
            }
            // A too-short tail falls through to the shorter prefixes.
        }
        if hit.is_none() && b == b'B' {
            hit = jwt_match_at(bytes, pos).map(|end| (KeyFamily::Jwt, b"Bearer".len(), end));
        }
        if hit.is_none() && b == b'-' {
            hit = pem_match_at(bytes, pos, &mut pem_ends, &mut pem_failed)
                .map(|end| (KeyFamily::Pem, 0, end));
        }
        let Some((family, verbatim_len, end)) = hit else {
            pos += 1;
            continue;
        };
        if mask & family.bit() == 0 {
            if let Some(r) = rec.as_deref_mut() {
                r.skipped[family as usize] += 1;
            }
            pos = end; // spent whole, preserved verbatim
            continue;
        }
        let matched = &text[pos..end];
        let head = if family == KeyFamily::Pem {
            "PEM"
        } else {
            &text[pos..pos + verbatim_len]
        };
        let token = format!("{head}~{}", token_digest(salt, matched));
        if let Some(r) = rec.as_deref_mut() {
            r.edits.push(PassEdit {
                start: pos,
                end,
                token: token.clone(),
                verbatim_src: pos,
                verbatim_len,
            });
            r.kinds.push(SpanKind::Key(family));
        }
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
/// identity (the caller's `rules=[]`). `key_families` scopes the keys
/// rule to a subset of families (the `families=` selection as a private
/// mask over the scanner's family table — an implementation detail, no
/// public bit arithmetic); it is meaningless when `keys` is false, and
/// with `keys` true a zero mask selects nothing (the identity — the
/// Python binding refuses that spelling outright).
#[derive(Clone, Copy)]
pub struct PiiRules {
    pub email: bool,
    pub phone: bool,
    pub keys: bool,
    pub key_families: u16,
}

impl PiiRules {
    /// The default `rules=None`: every rule in the canonical order —
    /// keys first, then email, then phone — over every key family this
    /// version knows.
    pub const BOTH: PiiRules = PiiRules {
        email: true,
        phone: true,
        keys: true,
        key_families: KEY_FAMILY_MASK_ALL,
    };
}

/// One pipeline stage's `Cow` fold: an inactive stage passes its input
/// through untouched; an active stage over a `Borrowed` input runs on
/// the borrow, and over an `Owned` intermediate folds back into it —
/// fired, its own output; unfired, the intermediate itself. Neither
/// branch copies, the zero-copy discipline the two-stage spelling paid
/// for, kept whole as the pipeline grew to three stages. Generic over
/// the pass so the keys stage can close over its family mask.
fn fold_stage<'a, P>(mid: Cow<'a, str>, active: bool, mut pass: P, salt: &str) -> Cow<'a, str>
where
    P: for<'x, 'y> FnMut(&'x str, &'y str) -> Cow<'x, str>,
{
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
/// whichever subset `rules` selects (the keys rule over whichever
/// subset `rules.key_families` selects). The contact rules digest with
/// `contact_salt` and the keys rule with `keys_salt`: the `salt=None`
/// per-rule defaults (`DEFAULT_SALT` / `KEYS_DEFAULT_SALT`) never alias
/// a contact digest with a key digest, an explicit string salts every
/// rule alike, and `""` is unsalted for every rule. `Cow::Borrowed` —
/// the identity path — exactly when no active rule matches (a skipped,
/// unselected family still passes its bytes through untouched).
pub fn scrub_pii<'a>(
    text: &'a str,
    rules: PiiRules,
    contact_salt: &str,
    keys_salt: &str,
) -> Cow<'a, str> {
    let mask = rules.key_families;
    let after_keys = fold_stage(
        Cow::Borrowed(text),
        rules.keys,
        |t, s| keys_pass_impl(t, s, mask, None),
        keys_salt,
    );
    let after_email = fold_stage(
        after_keys,
        rules.email,
        |t, s| email_pass_impl(t, s, None),
        contact_salt,
    );
    fold_stage(
        after_email,
        rules.phone,
        |t, s| phone_pass_impl(t, s, None),
        contact_salt,
    )
}

/// Map a position in a pass's OUTPUT back to the pass's input, through
/// that pass's recorded edits: a position in copied material shifts by
/// the cumulative delta; a position inside a token's verbatim head maps
/// affinely to its source bytes (the head IS those bytes), INCLUDING
/// the head-end boundary itself (a span END at the `~` slot consumed
/// the head through its last byte, so its preimage end is the head
/// end); a position strictly past the head — the derived digest half,
/// which has no preimage — collapses to the replaced span's end.
/// Monotone by construction, so mapped spans stay ordered and
/// well-formed: a later match that began inside a token's digest is
/// recorded from that token's input end (the first position whose
/// material is real input — exactly the keys-before-email corner's
/// shape), and a match that ran into a token's verbatim head maps back
/// onto the producing span (the input bytes fed two tokens; both spans
/// are recorded).
/// The offset-map cursor: a monotone sweep over one pass's edits for
/// a NON-DECREASING position stream. Each pass emits its spans
/// left-to-right, and every map is monotone, so every mapping stream
/// in the report qualifies — one sweep per pass, O(edits + spans)
/// total instead of O(edits × spans). Private to the report assembly;
/// callers must feed non-decreasing positions.
struct BackMap<'e> {
    edits: &'e [PassEdit],
    idx: usize,
    delta: isize,
}

impl<'e> BackMap<'e> {
    fn new(edits: &'e [PassEdit]) -> Self {
        BackMap {
            edits,
            idx: 0,
            delta: 0,
        }
    }

    fn map(&mut self, pos: usize) -> usize {
        while self.idx < self.edits.len() {
            let e = &self.edits[self.idx];
            let new_start = (e.start as isize + self.delta) as usize;
            if pos < new_start {
                break; // before this edit: copied material, identity by delta
            }
            let new_end = new_start + e.token.len();
            if pos < new_end {
                let off = pos - new_start;
                return if off <= e.verbatim_len {
                    e.verbatim_src + off
                } else {
                    e.end
                };
            }
            self.delta += e.token.len() as isize - (e.end - e.start) as isize;
            self.idx += 1;
        }
        (pos as isize - self.delta) as usize
    }
}

/// The single-position spelling of the cursor above: a fresh sweep per
/// call. Test-only (the unit pins below exercise the arithmetic
/// directly); the report sweeps each pass once via `BackMap`.
#[cfg(test)]
fn map_back(edits: &[PassEdit], pos: usize) -> usize {
    BackMap::new(edits).map(pos)
}

/// One redaction span, INPUT byte coordinates (the binding renders
/// codepoint indices): the rule that fired, or the key family that did.
pub struct ReportSpan {
    pub kind: SpanKind,
    pub start: usize,
    pub end: usize,
}

/// The `scrub_pii_report` accounting, in pipeline order: the scrubbed
/// text, the per-rule redacted counts, the per-family redacted counts
/// (`key_counts`, `ALL` order) and skipped counts (`skipped_counts`,
/// families NOT in the active selection that would have matched anyway
/// — detection ran, redaction did not), and the redaction spans in
/// input coordinates, ordered by start.
pub struct ScrubReport {
    pub text: String,
    pub email_count: usize,
    pub phone_count: usize,
    pub key_counts: [usize; 13],
    pub skipped_counts: [usize; 13],
    pub spans: Vec<ReportSpan>,
}

/// Scrub `text` exactly as `scrub_pii` would for the same arguments,
/// and account for every redaction: the keys substitution over the
/// input (its spans are already input coordinates), then the email
/// substitution over its result, then the phone substitution over that
/// — each pass recorded, each later span mapped back through the
/// earlier passes' substitutions by `map_back`, the three span sets
/// merged in start order. Counts and spans cover only ACTIVE rules; a
/// skipped (unselected but matching) family counts in `skipped_counts`
/// and leaves no span.
pub fn scrub_pii_report(
    text: &str,
    rules: PiiRules,
    contact_salt: &str,
    keys_salt: &str,
) -> ScrubReport {
    // The same Cow fold as `scrub_pii` (one owned string out, zero
    // transient copies on clean input), each stage recorded.
    let mask = rules.key_families;
    let mut krec = PassRec::default();
    let mut erec = PassRec::default();
    let mut prec = PassRec::default();
    let after_keys = fold_stage(
        Cow::Borrowed(text),
        rules.keys,
        |t, s| keys_pass_impl(t, s, mask, Some(&mut krec)),
        keys_salt,
    );
    let after_email = fold_stage(
        after_keys,
        rules.email,
        |t, s| email_pass_impl(t, s, Some(&mut erec)),
        contact_salt,
    );
    let final_text = fold_stage(
        after_email,
        rules.phone,
        |t, s| phone_pass_impl(t, s, Some(&mut prec)),
        contact_salt,
    )
    .into_owned();
    let mut key_counts = [0usize; 13];
    for kind in &krec.kinds {
        if let SpanKind::Key(f) = kind {
            key_counts[*f as usize] += 1;
        }
    }
    let mut spans: Vec<ReportSpan> =
        Vec::with_capacity(krec.edits.len() + erec.edits.len() + prec.edits.len());
    for (e, kind) in krec.edits.iter().zip(krec.kinds.iter()) {
        spans.push(ReportSpan {
            kind: *kind,
            start: e.start,
            end: e.end,
        });
    }
    // One sweep per pass: every span stream is left-to-right, every map
    // monotone — O(edits + spans), never O(edits × spans).
    let mut kmap = BackMap::new(&krec.edits);
    for e in &erec.edits {
        spans.push(ReportSpan {
            kind: SpanKind::Email,
            start: kmap.map(e.start),
            end: kmap.map(e.end),
        });
    }
    let mut emap = BackMap::new(&erec.edits);
    let mut kmap2 = BackMap::new(&krec.edits);
    for e in &prec.edits {
        let s2 = emap.map(e.start);
        let e2 = emap.map(e.end);
        spans.push(ReportSpan {
            kind: SpanKind::Phone,
            start: kmap2.map(s2),
            end: kmap2.map(e2),
        });
    }
    spans.sort_by_key(|s| s.start); // stable: start ties keep keys<email<phone
    ScrubReport {
        text: final_text,
        email_count: erec.edits.len(),
        phone_count: prec.edits.len(),
        key_counts,
        skipped_counts: krec.skipped,
        spans,
    }
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
                    keys: false,
                    key_families: KEY_FAMILY_MASK_ALL,
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
            key_families: KEY_FAMILY_MASK_ALL,
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
            key_families: KEY_FAMILY_MASK_ALL,
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
            key_families: KEY_FAMILY_MASK_ALL,
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
            key_families: KEY_FAMILY_MASK_ALL,
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
            key_families: KEY_FAMILY_MASK_ALL,
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
            key_families: KEY_FAMILY_MASK_ALL,
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
            key_families: KEY_FAMILY_MASK_ALL,
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
            key_families: KEY_FAMILY_MASK_ALL,
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
            key_families: KEY_FAMILY_MASK_ALL,
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
            key_families: KEY_FAMILY_MASK_ALL,
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
            key_families: KEY_FAMILY_MASK_ALL,
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
            key_families: KEY_FAMILY_MASK_ALL,
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
            key_families: KEY_FAMILY_MASK_ALL,
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
            key_families: KEY_FAMILY_MASK_ALL,
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
            key_families: KEY_FAMILY_MASK_ALL,
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
            key_families: KEY_FAMILY_MASK_ALL,
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

    /// An AWS-tail cycle: uppercase letters and digits only (the
    /// access-key ID alphabet — no lowercase anywhere in it).
    fn aws_tail(n: usize) -> String {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        (0..n)
            .map(|i| ALPHABET[i % ALPHABET.len()] as char)
            .collect()
    }

    /// An Azure-tail cycle: the connection-string secret alphabet.
    fn azure_tail(n: usize) -> String {
        const ALPHABET: &[u8] =
            b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/=";
        (0..n)
            .map(|i| ALPHABET[i % ALPHABET.len()] as char)
            .collect()
    }

    /// A PEM block: the BEGIN marker with algorithm words, base64 body
    /// lines (contact-inert by construction — no separators, no `@`),
    /// and the END marker with the same words.
    fn pem_block(words: &str) -> String {
        format!(
            "-----BEGIN {words} PRIVATE KEY-----\n\
             MIIEpAIBAAKCAQEA7b\n\
             qY4sLk2MnOpQrStUvW\n\
             xYz0123456789ABCD\n\
             -----END {words} PRIVATE KEY-----"
        )
    }

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
            ("AKIAIOSFODNN7EXAMPLE".to_string(), "AKIA"),
            (format!("ASIA{}", aws_tail(16)), "ASIA"),
            (format!("xai-{}", key_tail(20)), "xai-"),
            (format!("ya29.{}", key_tail(20)), "ya29."),
            (pem_block("RSA"), "PEM"),
            (format!("AccountKey={}", azure_tail(44)), "AccountKey="),
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
            format!("AKIA{}", aws_tail(15)), // one under the access-key ID
            format!("akia{}", aws_tail(16)), // lowercase: not the prefix
            format!("ASIA{}", aws_tail(15)),
            format!("xai-{}", key_tail(19)),
            "XAI-".to_string() + &key_tail(20),
            format!("ya29.{}", key_tail(19)),
            "YA29.".to_string() + &key_tail(20),
            format!("AccountKey={}", azure_tail(39)),
            "Accountkey=".to_string() + &azure_tail(44),
            format!("xAKIA{}", aws_tail(16)), // mid-token prefix
            format!("xxai-{}", key_tail(20)),
            format!("xya29.{}", key_tail(20)),
            format!("xAccountKey={}", azure_tail(40)),
            {
                let full = pem_block("RSA");
                full[..full.len() - "-----END RSA PRIVATE KEY-----".len()].to_string()
            }, // unterminated BEGIN
            pem_block("RSA").replace(
                "-----END RSA PRIVATE KEY-----",
                "-----END EC PRIVATE KEY-----",
            ), // mismatched words
            "-----BEGIN PRIVATE KEY-----\nMIIEpAIBAAKCAQEA7b\n-----END PRIVATE KEY-----"
                .to_string(), // no algorithm words
            "-----begin rsa private key-----\nMIIEpAIBAAKCAQEA7b\n-----end rsa private key-----"
                .to_string(), // lowercase markers
            "-----BEGIN RSA  PRIVATE KEY-----\nMIIE\n-----END RSA  PRIVATE KEY-----".to_string(), // doubled space
            "-----BEGIN RSA-EC PRIVATE KEY-----\nMIIE\n-----END RSA-EC PRIVATE KEY-----"
                .to_string(), // hyphen word
            "-----BEGIN RSA_EC PRIVATE KEY-----\nMIIE\n-----END RSA_EC PRIVATE KEY-----"
                .to_string(), // underscore word
            // Split across the concatenation: the joined shape trips
            // push protection (a synthetic vector, not a secret).
            "xoxb-".to_string() + "123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUv", // Slack: excluded
            "sk_test_51MZABCDefghijklmnOP0123456789abcdefghiJ".to_string(), // Stripe: excluded
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
                key_families: KEY_FAMILY_MASK_ALL,
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

    // --- Per-family selection and the report ---------------------------------

    fn keys_with_mask(mask: u16) -> PiiRules {
        PiiRules {
            email: false,
            phone: false,
            keys: true,
            key_families: mask,
        }
    }

    #[test]
    fn the_table_is_longest_prefix_first() {
        // The table's one total order, pinned structurally: whenever
        // one prefix is a proper prefix of another, the longer comes
        // first — so a 14th nesting family has exactly one place to
        // land, and drift fails here instead of as a fall-through
        // mystery. (The known nestings are the `sk-*` group and
        // `github_pat_`/`ghp_`.)
        for (i, (a, _, _, _)) in KEY_FAMILIES.iter().enumerate() {
            for (b, _, _, _) in &KEY_FAMILIES[i + 1..] {
                // A shorter prefix before a longer one it opens is the
                // misorder (the longer would never win); the reverse —
                // the `sk-*` group, `github_pat_`/`ghp_` — is the
                // discipline working.
                assert!(
                    !(b.starts_with(a) && a.len() < b.len()),
                    "misordered nesting: {a:?} at {i} opens {b:?}"
                );
            }
        }
    }

    #[test]
    fn the_family_names_agree_with_all_in_order() {
        // The single order everywhere: ALL[i].name() is NAMES[i], the
        // discriminant is the mask bit, and ALL is the full mask. The
        // Python battery pins the literal tuple; this pins the wiring.
        assert_eq!(KeyFamily::ALL.len(), KEY_FAMILY_NAMES.len());
        for (i, f) in KeyFamily::ALL.iter().enumerate() {
            assert_eq!(f.name(), KEY_FAMILY_NAMES[i]);
            assert_eq!(f.bit(), 1u16 << i);
            assert_eq!(*f as usize, i);
        }
        assert_eq!(KEY_FAMILY_MASK_ALL, (1u16 << 13) - 1);
    }

    #[test]
    fn a_five_part_jwe_keeps_parts_four_and_five_verbatim() {
        // The grammar is exactly three maximal segments: a 5-part JWE
        // behind Bearer scrubs through the third segment and keeps
        // parts 4-5 verbatim (the residual-risk red-team pin).
        assert_eq!(
            scrub("Bearer eyJa.b.c.d.e", keys_only(), ""),
            format!("Bearer~{}.d.e", digest("", "Bearer eyJa.b.c"))
        );
    }

    #[test]
    fn an_unselected_family_is_detected_not_redacted() {
        // The mask consults AFTER the grammar holds: a holding but
        // unselected family spends its span whole and preserves it
        // verbatim (no fall-through past a hold, no re-scan inside).
        let jwt_bit = KeyFamily::Jwt.bit();
        let no_jwt = KEY_FAMILY_MASK_ALL & !jwt_bit;
        let key = format!("sk-{}", key_tail(48));
        let text = format!("{JWT} {key}");
        assert_eq!(
            scrub(&text, keys_with_mask(no_jwt), ""),
            format!("{JWT} sk-~{}", digest("", &key))
        );
        // The single-family lane: only JWTs fire, an sk- key glued
        // beside one survives whole.
        assert_eq!(
            scrub(&text, keys_with_mask(jwt_bit), ""),
            format!("Bearer~{} {key}", digest("", JWT))
        );
    }

    #[test]
    fn pem_blocks_match_whole_and_span_lines() {
        // The whole multi-line block is one match; the token is the
        // constant PEM head over the full block's digest.
        let block = pem_block("EC");
        assert_eq!(
            scrub(&block, keys_only(), ""),
            format!("PEM~{}", digest("", &block))
        );
        // An empty body is still both markers: a match with nothing
        // between them.
        let empty = "-----BEGIN RSA PRIVATE KEY----------END RSA PRIVATE KEY-----";
        assert_eq!(
            scrub(empty, keys_only(), ""),
            format!("PEM~{}", digest("", empty))
        );
        // The search skips a mismatched END for a later matching one:
        // the stranger's END line is body material, and the block runs
        // whole to its own END.
        let nested = "-----BEGIN RSA PRIVATE KEY-----\n\
             MIIEpAIBAAKCAQEA7b\n\
             -----END EC PRIVATE KEY-----\n\
             -----END RSA PRIVATE KEY-----";
        assert_eq!(
            scrub(nested, keys_only(), ""),
            format!("PEM~{}", digest("", nested))
        );
    }

    #[test]
    fn azure_tokens_keep_the_marker_and_span_connection_strings() {
        // The realistic context: the marker plus its secret inside a
        // storage connection string — the match is the marker and its
        // maximal tail only, the `;`-separated fields around it survive.
        let secret = azure_tail(44);
        let text = format!(
            "DefaultEndpointsProtocol=https;AccountName=acct;AccountKey={secret};EndpointSuffix=core"
        );
        let matched = format!("AccountKey={secret}");
        assert_eq!(
            scrub(&text, keys_only(), ""),
            format!(
                "DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=~{};EndpointSuffix=core",
                digest("", &matched)
            )
        );
    }

    #[test]
    fn map_back_is_identity_without_edits() {
        assert_eq!(map_back(&[], 0), 0);
        assert_eq!(map_back(&[], 41), 41);
    }

    #[test]
    fn map_back_folds_verbatim_affine_and_collapses_digests() {
        // One keys edit: "sk-KEY(48)" [0,51) became "sk-~hex12" [0,16).
        // Copied material before shifts by nothing; the verbatim head
        // maps affinely INCLUDING its end boundary (a span END at the
        // `~` slot ate the head through its last byte); strictly past
        // the head the `~`+hex collapses to the replaced end; the
        // material after shifts by the delta.
        let edits = vec![PassEdit {
            start: 0,
            end: 51,
            token: format!("sk-~{}", "a".repeat(12)),
            verbatim_src: 0,
            verbatim_len: 3,
        }];
        assert_eq!(map_back(&edits, 0), 0);
        assert_eq!(map_back(&edits, 2), 2); // inside "sk-": affine
        assert_eq!(map_back(&edits, 3), 3); // head end: affine (span END)
        assert_eq!(map_back(&edits, 9), 51); // mid-hex: collapse
        assert_eq!(map_back(&edits, 16), 51); // token end: the replaced end
        assert_eq!(map_back(&edits, 17), 52); // past the token: delta -35
        assert_eq!(map_back(&edits, 30), 65);
    }

    #[test]
    fn the_report_counts_spans_and_sorts() {
        // The assembly contract, byte coordinates: keys spans direct,
        // email spans mapped through the keys edits, phone spans through
        // both — merged in start order with per-rule and per-family
        // counts, skipped in ALL order.
        let key = format!("sk-{}", key_tail(48));
        let text = format!("a@b.co {key} 415-555-2671");
        let rep = scrub_pii_report(&text, PiiRules::BOTH, "", "");
        assert_eq!(rep.text, scrub(&text, PiiRules::BOTH, ""));
        assert_eq!(rep.email_count, 1);
        assert_eq!(rep.phone_count, 1);
        assert_eq!(rep.key_counts[KeyFamily::OpenAi as usize], 1);
        assert_eq!(rep.key_counts.iter().sum::<usize>(), 1);
        assert_eq!(rep.skipped_counts, [0; 13]);
        // Sorted by start: the email span [0,6) first, then the key
        // span, then the phone span (mapped through the keys edit's
        // delta back to its input position).
        let key_start = 7;
        let phone_start = key_start + key.len() + 1;
        assert_eq!(rep.spans.len(), 3);
        assert!(matches!(rep.spans[0].kind, SpanKind::Email));
        assert_eq!((rep.spans[0].start, rep.spans[0].end), (0, 6));
        assert!(matches!(
            rep.spans[1].kind,
            SpanKind::Key(KeyFamily::OpenAi)
        ));
        assert_eq!(
            (rep.spans[1].start, rep.spans[1].end),
            (key_start, key_start + key.len())
        );
        assert!(matches!(rep.spans[2].kind, SpanKind::Phone));
        assert_eq!(
            (rep.spans[2].start, rep.spans[2].end),
            (phone_start, phone_start + 12)
        );
        assert_eq!(&text[phone_start..phone_start + 12], "415-555-2671");
    }

    #[test]
    fn the_report_sweeps_two_edits_for_a_late_email() {
        // The delta accumulates over every edit, not just the first: an
        // email after TWO keys maps through both tokens' deltas to its
        // input position.
        let a = format!("sk-{}", key_tail(48));
        let b = format!("fw-{}", key_tail(48));
        let text = format!("{a} {b} a@b.co");
        let rep = scrub_pii_report(&text, PiiRules::BOTH, "", "");
        assert_eq!(rep.spans.len(), 3);
        let email_at = a.len() + 1 + b.len() + 1;
        assert!(matches!(rep.spans[2].kind, SpanKind::Email));
        assert_eq!(
            (rep.spans[2].start, rep.spans[2].end),
            (email_at, email_at + 6)
        );
        assert_eq!(&text[email_at..email_at + 6], "a@b.co");
    }

    #[test]
    fn the_report_skips_unselected_families_without_spans() {
        let key = format!("sk-{}", key_tail(48));
        let text = format!("{JWT} {key}");
        let no_jwt = KEY_FAMILY_MASK_ALL & !KeyFamily::Jwt.bit();
        let rep = scrub_pii_report(
            &text,
            PiiRules {
                email: false,
                phone: false,
                keys: true,
                key_families: no_jwt,
            },
            "",
            "",
        );
        assert_eq!(rep.key_counts[KeyFamily::OpenAi as usize], 1);
        assert_eq!(rep.skipped_counts[KeyFamily::Jwt as usize], 1);
        assert_eq!(rep.skipped_counts.iter().sum::<usize>(), 1);
        assert_eq!(rep.spans.len(), 1);
        assert!(matches!(
            rep.spans[0].kind,
            SpanKind::Key(KeyFamily::OpenAi)
        ));
    }
}
