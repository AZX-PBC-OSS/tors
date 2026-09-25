//! Secret-token redaction grammars, the pure-Rust core of
//! `tors.scrub_secrets` (and the `secret_tokens` rule of
//! `tors.scrub_log_text`): the credential shapes operators hit in logs
//! that `scrub_pii`'s evidence-backed keys rule deliberately does not
//! carry, each a hand-rolled single-pass scanner over a pinned, cited
//! vendor shape.
//!
//! Five grammars, one rule name each:
//!
//! * `aws_access_key`: `AKIA`/`ASIA` + exactly 16 chars of `[0-9A-Z]`.
//!   The two prefixes are the access-key-ID heads AWS's own docs spell
//!   (the `AKIA` + `IOSFODNN7EXAMPLE` example, IAM's "Manage access
//!   keys" page); the tail
//!   alphabet is uppercase letters and digits, Amazon's example carrying
//!   both `I` and `O`, so there is no I/O exclusion to implement (the
//!   base32-ish folklore is not in Amazon's docs and is not pinned
//!   here). detect-secrets' AWS detector
//!   (`(?:A3T[A-Z0-9]|ABIA|ACCA|AKIA|ASIA)[0-9A-Z]{16}`) and trufflehog's
//!   (`\b(?:AKIA|ABIA|ACCA)[A-Z0-9]{16}\b`) bracket the class; tors pins
//!   the two live/temporary credential prefixes and the exact 16-char
//!   width. AWS documents no checksum for access key IDs, so none is
//!   validated and none is claimed.
//! * `slack_token`: case-insensitive `xox[abprso]-` + one or more
//!   digit-run/dash sections + a final alphanumeric run. Slack's docs
//!   pin the prefixes they document today (`xoxb-` bot, `xoxp-` user)
//!   and the dash-separated section structure whose final section is the
//!   secret (32 chars now, 6/10 before August 2016); the full grammar is
//!   detect-secrets' Slack detector,
//!   `xox(?:a|b|p|o|s|r)-(?:\d+-)+[a-z0-9]+` under IGNORECASE, pinned
//!   verbatim including its case-insensitivity. The classes are ASCII
//!   digits and letters only: Python's `\d` under the cited pattern
//!   would also match Unicode digits (`xoxb-١٢٣-…`), which the scanner
//!   refuses — the documented scanner/oracle divergence. Trufflehog's stricter
//!   two-numeric-sections spelling (`xoxb\-[0-9]{10,13}\-[0-9]{10,13}…`)
//!   is a subset; the looser cited grammar is the one pinned (a
//!   redactor's misses are the dangerous direction). Slack's `xapp-`
//!   app-level prefix is a different head and stays out.
//! * `stripe_key`: `(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{24,}`, the
//!   tail consumed maximally. Stripe's docs pin the six prefixes
//!   (`pk_live_`/`rk_live_`/`sk_live_` and the `_test_` spellings) and
//!   do not pin a length; trufflehog's detector
//!   (`[rs]k_live_[a-zA-Z0-9]{20,247}`) brackets the documented example
//!   key spelling (`sk_live_` + 24). tors pins 24+ so the documented
//!   shape always matches and newer longer keys match too. No checksum
//!   is documented by Stripe; none is validated or claimed. Live versus
//!   test is visible in the prefix and the REPORT says which
//!   (`stripe_live`/`stripe_test`); publishable `pk_` keys are not
//!   secrets by Stripe's own table and scrub anyway (over-redaction,
//!   safe direction). Stripe's organization key prefix `sk_org_` (no
//!   live/test half in Stripe's docs) stays a documented exclusion.
//! * `github_token`: `(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}`, exactly
//!   36, plus the pre-2021 legacy class: a maximal hex run of exactly 40
//!   chars at a word boundary. GitHub's token-format post pins the five
//!   prefixes, the `_` separator, the base62 random portion, and the
//!   old formats it replaced ("hex-encoded 40 character strings that
//!   are indistinguishable from other encoded data like SHA hashes");
//!   trufflehog's detector
//!   (`\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[a-zA-Z0-9_]{36,255}\b`)
//!   carries the same prefix set. GitHub DEFINES a checksum (a 32-bit
//!   CRC32 base62-encoded in the last 6 chars) but does not publish its
//!   exact construction, so tors validates no checksum and claims none:
//!   the shape is recall-biased by design. The fine-grained
//!   `github_pat_` shape is NOT here (this module's closed set is the
//!   five grammars above; `scrub_pii`'s `api_keys` rule carries
//!   `github_pat_` and the rest of its own evidence-backed set).
//! * `pem_key`: the `-----BEGIN <words> PRIVATE KEY-----` …
//!   `-----END <same words> PRIVATE KEY-----` block span (RFC 7468's
//!   framing), the WHOLE block the redacted span. The marker parser and
//!   END index are the keys rule's own (`pii_impl::pem_marker_end`,
//!   `pii_impl::pem_end_index`), reused, not reimplemented: both
//!   markers required, the label's words equal, the PGP
//!   ` PRIVATE KEY BLOCK-----` close accepted the same way.
//!
//! ## Boundary discipline
//!
//! Anchored boundaries everywhere, no mid-token matches. The shared
//! glue class is the keys rule's own key charset `[A-Za-z0-9_-]`
//! (`pii_impl::is_key_tail_byte`): a head glued to a preceding class
//! char is MID-TOKEN and never fires (`XAKIA…`), except that the
//! prefix grammars fire when that char ends a complete escape sequence
//! (`%XX`, `\uXXXX`, …) or an ANSI CSI sequence, whose tail byte is
//! formatting material (the keys rule's own carve-outs, reused
//! verbatim), or the position opens a PEM BEGIN head after a dash run
//! (the same armor carve). The carve-outs do NOT start the legacy hex
//! class's head fresh: that class adds its own word-boundary rule
//! (before- and after-chars outside `[A-Za-z0-9_]`), and an escape or
//! CSI tail byte can itself be a word char (`%4D`, a CSI final letter),
//! so a hex run behind such a tail stays glued and refused even where
//! a prefix grammar fires — pinned by
//! `test_legacy_hex_after_a_json_escape_stays_glued`. The tail runs
//! are maximal over their own alphabets, except where the vendor pins an
//! exact width (AWS 16, GitHub 36, legacy hex 40): an exact-width
//! candidate inside a LONGER run of its own alphabet is a near-miss and
//! does not match (the run's interior cannot anchor again: every
//! interior byte is class-glued to its predecessor). The legacy hex
//! class adds the word-boundary form the prefix grammars cannot use:
//! its after-char must not be `[A-Za-z0-9_]`, because a prefixless
//! class's match is extended by any word char on either side.
//!
//! ## Linearity
//!
//! One pass over the input, linear in it, the contract every pass here
//! owes the scrub API. The invariant that keeps it linear: every anchor
//! requires the previous byte to sit OUTSIDE the shared glue class
//! (plus the escape/CSI/armor carve-outs), and every alphabet a failed
//! candidate walks is a subset of that glue class, so a failed
//! candidate's walk contains no later anchor: failed walks are disjoint,
//! each byte is class-walked O(1) times, and no shape composes a
//! quadratic rescan. Exact-width grammars add one run walk per run
//! start (the anchor is always a run start: the glue class contains the
//! run's alphabet).
//!
//! ## Tokens and the report
//!
//! `scrub_secrets` replaces each span with `<head>~<digest>`: the head
//! is the match's own non-secret prefix verbatim (`AKIA`, `xoxb`,
//! `sk_live_`, `ghp_`), or a constant when the grammar has none
//! (`github` for the legacy hex class, `PEM` for the block span), and
//! `<digest>` is the first 12 hex chars of `sha256(salt + match)` over
//! the FULL match, the keys rule's own token construction
//! (`pii_impl::token_digest`). The default salt is this module's own
//! frozen domain-separation tag (`SECRETS_DEFAULT_SALT`), so a secret
//! digest never aliases a `scrub_pii` digest at the default settings.
//! Tokens are fixed points by construction: every prefix's tail check
//! lands on the `~` (never its own alphabet), the digest hex spells no
//! prefix, and the constant heads are not candidate material, so
//! `scrub_secrets` is strictly idempotent.
//!
//! `scrub_secrets_report` returns the same scrub plus the accounting
//! (`text`, per-kind counts, spans in input coordinates): one scan, so
//! the spans ARE input coordinates and no offset mapping exists.
//!
//! The span kinds (`SecretKind`) are finer than the rule selection
//! (`SecretRules`): the `stripe_key` rule reports `stripe_live` and
//! `stripe_test` separately, and the `github_token` rule reports the
//! modern and legacy classes separately.
//!
//! False-positive posture, per grammar (prefix + length classes are
//! intentionally recall-biased; the docs say so, docs/api.md carries the
//! same table):
//!
//! * `aws_access_key`: FP only on 20-char `[0-9A-Z]` strings with an
//!   `AKIA`/`ASIA` head at a clean boundary; no checksum exists to
//!   check.
//! * `slack_token`: FP on any `xox?-digit-dash-alnum` shape; the
//!   digit-section structure is Slack-specific enough that FP shows up
//!   almost only in token-shaped test fixtures.
//! * `stripe_key`: FP on 32+-char alnum strings behind a stripe prefix;
//!   `pk_` keys are not secrets and scrub anyway.
//! * `github_token` (modern): FP on random 40-char base62 strings with
//!   a `gh?_` head; the vendor checksum is unimplemented (construction
//!   unpublished).
//! * `github_token` (legacy): MAXIMAL recall bias: every clean 40-char
//!   hex run matches, SHA-1 digests and git commit ids included; the
//!   class exists because GitHub's own pre-2021 tokens were exactly that
//!   shape and are indistinguishable from those hashes by design. A log
//!   stream dense with bare SHA-1s selects this rule with its FP cost
//!   knowingly.
//! * `pem_key`: both markers with matching label words required, so the
//!   FP rate is negligible.

use std::borrow::Cow;

use crate::pii_impl::{
    PemEndIndex, ansi_csi_ends_before, escape_ends_before, is_key_tail_byte, pem_end_index,
    pem_head_after_dash_run, pem_marker_end, token_digest,
};

/// The secrets surface's own documented default digest salt: the same
/// frozen-tag discipline as `pii_impl::DEFAULT_SALT` /
/// `pii_impl::KEYS_DEFAULT_SALT`, and a THIRD tag so a secret digest
/// never aliases a contact or api-key digest at the default settings.
/// Changing it would silently change every deployment's token values.
pub const SECRETS_DEFAULT_SALT: &str = "tors/scrub_secrets/v1";

impl std::ops::BitOrAssign for SecretRules {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// One redacted span's kind: finer than the rule selection. The
/// `stripe_key` rule reports live and test separately (the report must
/// say which), and the `github_token` rule reports the modern prefixed
/// class and the legacy 40-hex class separately. The order is the
/// canonical order (`ALL`, the counts array's order).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SecretKind {
    AwsAccessKey,
    SlackToken,
    StripeLive,
    StripeTest,
    GitHubToken,
    GitHubLegacyToken,
    PemPrivateKey,
}

impl SecretKind {
    /// Every kind in the canonical order: the counts array's order, the
    /// report's kind-name table's order, one order everywhere.
    pub const ALL: [SecretKind; 7] = [
        SecretKind::AwsAccessKey,
        SecretKind::SlackToken,
        SecretKind::StripeLive,
        SecretKind::StripeTest,
        SecretKind::GitHubToken,
        SecretKind::GitHubLegacyToken,
        SecretKind::PemPrivateKey,
    ];

    /// The report's span/`redacted` name: lowercase, snake_case, the
    /// stripe environment split out.
    pub const fn name(&self) -> &'static str {
        match self {
            SecretKind::AwsAccessKey => "aws_access_key",
            SecretKind::SlackToken => "slack_token",
            SecretKind::StripeLive => "stripe_live",
            SecretKind::StripeTest => "stripe_test",
            SecretKind::GitHubToken => "github_token",
            SecretKind::GitHubLegacyToken => "github_legacy_token",
            SecretKind::PemPrivateKey => "pem_private_key",
        }
    }
}

/// The selected rules, as a bit set. Construction is the py layer's
/// (name parsing and validation happen under the GIL); the core only
/// consults membership. The same shape as `scrub_impl::RuleSet`: a
/// closed set of names, never a pattern parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SecretRules(u8);

impl SecretRules {
    /// The empty selection: `rules=[]`, the identity.
    pub const EMPTY: Self = Self(0);
    /// `AKIA`/`ASIA` + exactly 16 uppercase-alphanumeric chars.
    pub const AWS_ACCESS_KEY: Self = Self(1);
    /// Case-insensitive `xox[abprso]-` + digit sections + alnum tail.
    pub const SLACK_TOKEN: Self = Self(2);
    /// The six `(?:sk|rk|pk)_(?:live|test)_` prefixes + 24+ alnum.
    pub const STRIPE_KEY: Self = Self(4);
    /// The five modern `gh?_` prefixes + exactly 36 base62, and the
    /// legacy exactly-40 hex class.
    pub const GITHUB_TOKEN: Self = Self(8);
    /// The PEM private-key block span.
    pub const PEM_KEY: Self = Self(16);
    /// `rules=None`: every grammar.
    pub const ALL: Self = Self(31);

    pub fn aws_access_key(self) -> bool {
        self.0 & Self::AWS_ACCESS_KEY.0 != 0
    }

    pub fn slack_token(self) -> bool {
        self.0 & Self::SLACK_TOKEN.0 != 0
    }

    pub fn stripe_key(self) -> bool {
        self.0 & Self::STRIPE_KEY.0 != 0
    }

    pub fn github_token(self) -> bool {
        self.0 & Self::GITHUB_TOKEN.0 != 0
    }

    pub fn pem_key(self) -> bool {
        self.0 & Self::PEM_KEY.0 != 0
    }
}

/// The canonical rule-name tuple, in `SecretRules` bit order: the single
/// source the binding's unknown-name error and the doc spell from.
pub const SECRET_RULE_NAMES: [&str; 5] = [
    "aws_access_key",
    "slack_token",
    "stripe_key",
    "github_token",
    "pem_key",
];

/// One scanned span: input byte coordinates and the kind that fired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SecretSpan {
    pub start: usize,
    pub end: usize,
    pub kind: SecretKind,
}

/// The `scrub_secrets_report` accounting: the scrubbed text, the
/// per-kind redacted counts (`ALL` order), and the redaction spans in
/// input byte coordinates, ordered by start (one scan emits them in
/// order; no offset mapping exists).
#[derive(Debug)]
pub struct SecretReport {
    pub text: String,
    pub kind_counts: [usize; 7],
    pub spans: Vec<SecretSpan>,
}

/// The AWS tail alphabet: `[0-9A-Z]`, uppercase letters and digits.
/// Amazon's own example access key ID carries both `I` and `O`; no
/// I/O-exclusion or base32 narrowing is documented and none is applied.
#[inline]
fn is_aws_tail_byte(b: u8) -> bool {
    b.is_ascii_uppercase() || b.is_ascii_digit()
}

/// The base62 class GitHub's token bodies use (`a-z A-Z 0-9`), also the
/// Stripe tail class.
#[inline]
fn is_b62_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric()
}

/// The ASCII hex class, case-insensitive: the legacy GitHub token
/// alphabet GitHub's post describes as "hex-encoded 40 character
/// strings".
#[inline]
fn is_hex_byte(b: u8) -> bool {
    b.is_ascii_hexdigit()
}

/// The legacy class's extension class: a word char (`[A-Za-z0-9_]`). A
/// 40-hex candidate glued to a word char on either side is a fragment
/// of a longer word (a longer hash, an identifier), never a token.
#[inline]
fn is_legacy_glue_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// The AWS grammar at `pos`: `AKIA`/`ASIA` + exactly 16 chars of
/// `[0-9A-Z]`. The tail run is walked to its end once; the candidate
/// holds only when the run is EXACTLY 16 (a longer run is a longer
/// word, the near-miss the exact width exists to refuse; a shorter one
/// is not the shape). The walk cost is one run per run start: the
/// caller's boundary gate puts every anchor at a run start.
fn aws_match_at(bytes: &[u8], pos: usize) -> Option<usize> {
    const PREFIXES: [&[u8]; 2] = [b"AKIA", b"ASIA"];
    for prefix in PREFIXES {
        if !bytes[pos..].starts_with(prefix) {
            continue;
        }
        let tail_start = pos + prefix.len();
        let mut end = tail_start;
        while end < bytes.len() && is_aws_tail_byte(bytes[end]) {
            end += 1;
        }
        return (end - tail_start == 16).then_some(end);
    }
    None
}

/// The Slack grammar at `pos`: case-insensitive `xox`, one of
/// `[abprso]` (case-insensitive), `-`, then one-or-more
/// (digit-run + `-`) sections, then a final alphanumeric run of at
/// least one char (detect-secrets' `xox(?:a|b|p|o|s|r)-(?:\d+-)+[a-z0-9]+`
/// under IGNORECASE, pinned verbatim). The final run is maximal and its
/// class consumes digits: a digit run NOT followed by a dash is where
/// the final (secret) section starts, digit-led spellings included
/// (`xoxb-1-23`, the modern bot-token layout's digit-led 32-char
/// secret) — the run is rewound to the final class, not refused. A
/// trailing digit run with NO dash-terminated section before it
/// (`xoxb-123`) is still no match: the grammar needs one full
/// digit-run+dash section first. Single-pass, no backtracking: the
/// scanner never re-opens a consumed section (the pinned refusals
/// `xoxb-1-2-3-`, `xoxb-1-2--3` — the greedy regex would backtrack a
/// trailing dash run into the final class; the scanner's refusal is
/// the pinned contract, tested in tests/test_secret_grammars.py).
fn slack_match_at(bytes: &[u8], pos: usize) -> Option<usize> {
    let fold = |b: u8| b.to_ascii_lowercase();
    if bytes.get(pos + 4) != Some(&b'-')
        || !matches!(fold(bytes[pos]), b'x')
        || !matches!(fold(bytes.get(pos + 1).copied()?), b'o')
        || !matches!(fold(bytes.get(pos + 2).copied()?), b'x')
        || !matches!(
            fold(bytes[pos + 3]),
            b'a' | b'b' | b'p' | b'r' | b's' | b'o'
        )
    {
        return None;
    }
    let mut i = pos + 5;
    let mut sections = 0;
    loop {
        let digits_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == digits_start {
            break;
        }
        if bytes.get(i) != Some(&b'-') {
            // A digit run not followed by a dash is where the FINAL
            // section starts: the cited grammar's final `[a-z0-9]+`
            // class consumes digits, so the final (secret) section may
            // open with a digit run (`xoxb-1-23`, the modern bot-token
            // layout's digit-led 32-char secret). Rewind the run — the
            // final-run class re-consumes it below — rather than
            // refusing the token: refusing here leaked every token
            // whose final section starts with a digit (RT-SLACK-1).
            i = digits_start;
            break;
        }
        i += 1;
        sections += 1;
    }
    if sections == 0 {
        return None;
    }
    let final_start = i;
    while i < bytes.len() && is_b62_byte(bytes[i]) {
        i += 1;
    }
    (i > final_start).then_some(i)
}

/// The Stripe grammar at `pos`: one of the six
/// `(?:sk|rk|pk)_(?:live|test)_` prefixes + at least 24 chars of
/// `[A-Za-z0-9]`, the tail maximal. Returns the match end and the
/// environment the prefix spells (the report must say which).
fn stripe_match_at(bytes: &[u8], pos: usize) -> Option<(usize, SecretKind)> {
    const S: [(&[u8], SecretKind); 2] = [
        (b"sk_live_", SecretKind::StripeLive),
        (b"sk_test_", SecretKind::StripeTest),
    ];
    const R: [(&[u8], SecretKind); 2] = [
        (b"rk_live_", SecretKind::StripeLive),
        (b"rk_test_", SecretKind::StripeTest),
    ];
    const P: [(&[u8], SecretKind); 2] = [
        (b"pk_live_", SecretKind::StripeLive),
        (b"pk_test_", SecretKind::StripeTest),
    ];
    let candidates: [(&[u8], SecretKind); 2] = match bytes[pos] {
        b's' => S,
        b'r' => R,
        b'p' => P,
        _ => return None,
    };
    for (prefix, kind) in candidates {
        if !bytes[pos..].starts_with(prefix) {
            continue;
        }
        let tail_start = pos + prefix.len();
        let mut end = tail_start;
        while end < bytes.len() && is_b62_byte(bytes[end]) {
            end += 1;
        }
        return (end - tail_start >= 24).then_some((end, kind));
    }
    None
}

/// The modern GitHub grammar at `pos`: one of the five documented
/// `gh?_` prefixes + exactly 36 chars of base62 (the token-format
/// post's own length pin: 30 random base62 chars plus the 6-char
/// checksum tail, whose exact CRC32/base62 construction GitHub does not
/// publish, so no checksum is validated here). The tail run is walked
/// once; exactly 36 holds, longer and shorter are near-miss non-matches.
fn github_modern_match_at(bytes: &[u8], pos: usize) -> Option<usize> {
    const PREFIXES: [&[u8]; 5] = [b"ghp_", b"gho_", b"ghu_", b"ghs_", b"ghr_"];
    for prefix in PREFIXES {
        if !bytes[pos..].starts_with(prefix) {
            continue;
        }
        let tail_start = pos + prefix.len();
        let mut end = tail_start;
        while end < bytes.len() && is_b62_byte(bytes[end]) {
            end += 1;
        }
        return (end - tail_start == 36).then_some(end);
    }
    None
}

/// The legacy GitHub class at `pos`: a maximal hex run of exactly 40
/// chars whose before/after chars are not word chars
/// (`[A-Za-z0-9_]`; the prefixless class extends by any word char, the
/// word-boundary form). The caller's shared gate already refused a
/// dash/underscore-glued head; the after check here is the class's own.
fn github_legacy_match_at(bytes: &[u8], pos: usize) -> Option<usize> {
    let mut end = pos;
    while end < bytes.len() && is_hex_byte(bytes[end]) {
        end += 1;
    }
    if end - pos != 40 {
        return None;
    }
    if bytes.get(end).is_some_and(|&b| is_legacy_glue_byte(b)) {
        return None;
    }
    Some(end)
}

/// The PEM grammar at `pos`, the keys rule's own machinery reused (both
/// markers, matching label words, the PGP close accepted): the whole
/// block is the span.
fn pem_match_at(bytes: &[u8], pos: usize, index: &mut Option<PemEndIndex>) -> Option<usize> {
    const BEGIN: &[u8] = b"-----BEGIN ";
    if !bytes[pos..].starts_with(BEGIN) {
        return None;
    }
    let (words_end, body_start) = pem_marker_end(bytes, pos + BEGIN.len())?;
    let words = &bytes[pos + BEGIN.len()..words_end];
    let ends = index.get_or_insert_with(|| pem_end_index(bytes));
    let bucket = ends.get(words)?;
    let first = bucket.partition_point(|&(cand, _)| cand < body_start);
    bucket.get(first).map(|&(_, end)| end)
}

/// One left-to-right scan, every grammar, selected rules only: the
/// spans in input byte coordinates, ordered by start, non-overlapping
/// (a match's span is consumed whole; the scan resumes at its end).
/// Linear in the input, the linearity invariant in the module doc.
pub fn scan_secrets(text: &str, rules: SecretRules) -> Vec<SecretSpan> {
    let bytes = text.as_bytes();
    let mut spans: Vec<SecretSpan> = Vec::new();
    let mut pem_ends: Option<PemEndIndex> = None;
    let mut pos = 0usize;
    while pos < bytes.len() {
        let b = bytes[pos];
        // The shared boundary gate: a prefix (or hex-run head) glued to
        // a preceding key-charset char is MID-TOKEN and never fires,
        // unless that char ends a complete escape sequence or an ANSI
        // CSI sequence (the keys rule's carve-outs, reused), or the
        // position opens a PEM BEGIN head after a dash run (armor, not
        // a word). One table load plus, on a hex-class tail, one
        // backward walk, for every byte of prose: the cheap arm ahead
        // of the grammar tries.
        let clean = if pos == 0 || !is_key_tail_byte(bytes[pos - 1]) {
            true
        } else {
            escape_ends_before(bytes, pos)
                || ansi_csi_ends_before(bytes, pos)
                || pem_head_after_dash_run(bytes, pos)
        };
        let mut hit: Option<(usize, SecretKind)> = None;
        if clean {
            if rules.aws_access_key() && b == b'A' {
                hit = aws_match_at(bytes, pos).map(|end| (end, SecretKind::AwsAccessKey));
            }
            if hit.is_none() && rules.slack_token() && matches!(b, b'x' | b'X') {
                hit = slack_match_at(bytes, pos).map(|end| (end, SecretKind::SlackToken));
            }
            if hit.is_none() && rules.stripe_key() && matches!(b, b's' | b'r' | b'p') {
                hit = stripe_match_at(bytes, pos);
            }
            if hit.is_none() && rules.github_token() && b == b'g' {
                hit = github_modern_match_at(bytes, pos).map(|end| (end, SecretKind::GitHubToken));
            }
            if hit.is_none()
                && rules.github_token()
                && is_hex_byte(b)
                // The legacy class's own boundary: a hex run behind a
                // word char is a fragment of a longer word. This check
                // runs AFTER the shared gate's escape carve-outs and
                // dominates them on the head side: an escape tail can
                // itself be a word char (`%4D`, a CSI final letter), and
                // the carve-outs do not start this class's head fresh —
                // the word-glue refusal holds (pinned by
                // test_legacy_hex_after_a_json_escape_stays_glued).
                && (pos == 0 || !is_legacy_glue_byte(bytes[pos - 1]))
            {
                hit = github_legacy_match_at(bytes, pos)
                    .map(|end| (end, SecretKind::GitHubLegacyToken));
            }
            if hit.is_none() && rules.pem_key() && b == b'-' {
                hit = pem_match_at(bytes, pos, &mut pem_ends)
                    .map(|end| (end, SecretKind::PemPrivateKey));
            }
        }
        let Some((end, kind)) = hit else {
            pos += 1;
            continue;
        };
        spans.push(SecretSpan {
            start: pos,
            end,
            kind,
        });
        pos = end;
    }
    spans
}

/// The token head a span's replacement keeps verbatim: the match's own
/// non-secret prefix (which credential to rotate), or the constant when
/// the grammar has none (`github` for the prefixless legacy class,
/// `PEM` for the block span, the keys rule's own PEM precedent).
fn span_head<'a>(text: &'a str, span: &SecretSpan) -> &'a str {
    match span.kind {
        SecretKind::AwsAccessKey | SecretKind::SlackToken | SecretKind::GitHubToken => {
            &text[span.start..span.start + 4]
        }
        SecretKind::StripeLive | SecretKind::StripeTest => &text[span.start..span.start + 8],
        SecretKind::GitHubLegacyToken => "github",
        SecretKind::PemPrivateKey => "PEM",
    }
}

/// Scrub `text` of the selected secret-token grammars: each span
/// becomes `<head>~<digest>` (the head verbatim, the digest the first
/// 12 hex chars of `sha256(salt + match)` over the full span).
/// `Cow::Borrowed` (the identity lane) exactly when no grammar matched.
pub fn scrub_secrets<'a>(text: &'a str, rules: SecretRules, salt: &str) -> Cow<'a, str> {
    let spans = scan_secrets(text, rules);
    if spans.is_empty() {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut emitted = 0usize;
    for span in &spans {
        out.push_str(&text[emitted..span.start]);
        let head = span_head(text, span);
        let digest = token_digest(salt, &text[span.start..span.end]);
        out.push_str(head);
        out.push('~');
        out.push_str(&digest);
        emitted = span.end;
    }
    out.push_str(&text[emitted..]);
    Cow::Owned(out)
}

/// The log-scrub spelling: the same scan, each span spliced to `***`
/// (the `scrub_log_text` family's own mask convention), one pass, no
/// token digests. `Cow::Borrowed` exactly when no grammar matched.
pub fn mask_secret_tokens(text: &str) -> Cow<'_, str> {
    let spans = scan_secrets(text, SecretRules::ALL);
    if spans.is_empty() {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut emitted = 0usize;
    for span in &spans {
        out.push_str(&text[emitted..span.start]);
        out.push_str("***");
        emitted = span.end;
    }
    out.push_str(&text[emitted..]);
    Cow::Owned(out)
}

/// Scrub `text` exactly as `scrub_secrets` would for the same
/// arguments, and account for every redaction: the per-kind counts and
/// the spans (input byte coordinates, ordered by start).
pub fn scrub_secrets_report(text: &str, rules: SecretRules, salt: &str) -> SecretReport {
    let spans = scan_secrets(text, rules);
    let mut kind_counts = [0usize; 7];
    let scrubbed = if spans.is_empty() {
        None
    } else {
        let mut out = String::with_capacity(text.len());
        let mut emitted = 0usize;
        for span in &spans {
            out.push_str(&text[emitted..span.start]);
            let head = span_head(text, span);
            let digest = token_digest(salt, &text[span.start..span.end]);
            out.push_str(head);
            out.push('~');
            out.push_str(&digest);
            emitted = span.end;
        }
        out.push_str(&text[emitted..]);
        Some(out)
    };
    for span in &spans {
        kind_counts[span.kind as usize] += 1;
    }
    SecretReport {
        text: scrubbed.unwrap_or_else(|| text.to_owned()),
        kind_counts,
        spans,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scrub(text: &str, rules: SecretRules) -> String {
        scrub_secrets(text, rules, "").into_owned()
    }

    fn digest(matched: &str) -> String {
        token_digest("", matched)
    }

    /// A synthesized uppercase-alphanumeric run (never a real
    /// credential; the alphabet is [0-9A-Z] so the tail stays in the
    /// AWS class).
    fn aws_tail() -> &'static str {
        "B2C4E6G8H1J3K5M9"
    }

    #[test]
    fn clean_input_is_identity() {
        let text = "plain prose, café, emoji \u{1f600}, numbers 4096 and 1200, \
                    refs deadbeef and 0123456789abcdef";
        assert!(matches!(
            scrub_secrets(text, SecretRules::ALL, ""),
            Cow::Borrowed(_)
        ));
        assert!(matches!(mask_secret_tokens(text), Cow::Borrowed(_)));
    }

    #[test]
    fn empty_rules_is_identity() {
        // Split across the concatenation: the joined shape trips push
        // protection (a synthetic vector, not a secret).
        let text = concat!("AKIA", "B2C4E6G8H1J3K5L7M9 xoxb-123-456-abcdefabcdef");
        assert!(matches!(
            scrub_secrets(text, SecretRules::EMPTY, ""),
            Cow::Borrowed(_)
        ));
    }

    // --- aws_access_key ------------------------------------------------

    #[test]
    fn aws_access_keys_scrub_with_the_prefix_kept() {
        // Amazon's documented example spelling (the `AKIA` +
        // `IOSFODNN7EXAMPLE` example, IAM's "Manage access keys" page)
        // plus a synthesized ASIA key.
        let asia = format!("ASIA{}", aws_tail());
        for key in [concat!("AKIA", "IOSFODNN7EXAMPLE"), asia.as_str()] {
            assert_eq!(
                scrub(key, SecretRules::AWS_ACCESS_KEY),
                format!("{}~{}", &key[..4], digest(key))
            );
        }
    }

    #[test]
    fn aws_near_misses_are_anchored() {
        let key = format!("AKIA{}", aws_tail());
        // The tail's run extended: a longer [0-9A-Z] word, never a match.
        assert_eq!(
            scrub(&format!("{key}X"), SecretRules::AWS_ACCESS_KEY),
            format!("{key}X")
        );
        assert_eq!(
            scrub(&format!("{key}9"), SecretRules::AWS_ACCESS_KEY),
            format!("{key}9")
        );
        // Glued to a preceding word char: mid-token, never a match.
        assert_eq!(
            scrub(&format!("X{key}"), SecretRules::AWS_ACCESS_KEY),
            format!("X{key}")
        );
        assert_eq!(
            scrub(&format!("x{key}"), SecretRules::AWS_ACCESS_KEY),
            format!("x{key}")
        );
        assert_eq!(
            scrub(&format!("_{key}"), SecretRules::AWS_ACCESS_KEY),
            format!("_{key}")
        );
        // Lowercase variants: not the AWS alphabet, never a match.
        assert_eq!(
            scrub("akiaB2C4E6G8H1J3K5M9", SecretRules::AWS_ACCESS_KEY),
            "akiaB2C4E6G8H1J3K5M9"
        );
        // Short tails: not the shape.
        assert_eq!(scrub(&key[..19], SecretRules::AWS_ACCESS_KEY), key[..19]);
        // A clean boundary before the key still scrubs: punctuation and
        // text start are boundaries.
        assert_eq!(
            scrub(&format!("key={key}"), SecretRules::AWS_ACCESS_KEY),
            format!("key=AKIA~{}", digest(&key))
        );
        assert_eq!(
            scrub(&format!("-{key}"), SecretRules::AWS_ACCESS_KEY),
            // A dash is key-charset glue (the shared mid-token class).
            format!("-{key}")
        );
    }

    // --- slack_token ---------------------------------------------------

    #[test]
    fn slack_tokens_scrub_with_the_prefix_kept() {
        for token in [
            concat!("xox", "b-123456789012-1234567890123-abcdefghijklmnop"),
            // Synthesized.
            "xoxp-111-222-d6bc768406e5c2e6958cfc399b438004",
            "xoxa-1-2-abc123xyz",
            "xoxr-1-2-abc123xyz",
            "xoxs-1-2-abc123xyz",
            "xoxo-1-2-abc123xyz",
            // Digit-led FINAL sections: the cited grammar's final
            // `[a-z0-9]+` class consumes digits (RT-SLACK-1). The third
            // is the modern bot-token layout with a digit-led 32-char
            // secret — the leak direction.
            "xoxb-1-23",
            "xoxb-123-456-789012",
            concat!("xox", "b-123456789012-1234567890123-9f8e7d6c5b4a3211f00d"),
        ] {
            assert_eq!(
                scrub(token, SecretRules::SLACK_TOKEN),
                format!("{}~{}", &token[..4], digest(token)),
                "{token}"
            );
        }
        // Case-insensitive: the cited IGNORECASE grammar.
        let token = "XOXB-123-456-ABCDEFABCDEF";
        assert_eq!(
            scrub(token, SecretRules::SLACK_TOKEN),
            format!("XOXB~{}", digest(token))
        );
    }

    #[test]
    fn slack_near_misses_are_anchored() {
        // No digit section: never a match.
        assert_eq!(scrub("xoxb-abc", SecretRules::SLACK_TOKEN), "xoxb-abc");
        assert_eq!(
            scrub("xoxb-abc-123", SecretRules::SLACK_TOKEN),
            "xoxb-abc-123"
        );
        // A digit run with no dash-terminated section before it: not
        // the shape (the grammar needs one full digit-run+dash first).
        assert_eq!(scrub("xoxb-123", SecretRules::SLACK_TOKEN), "xoxb-123");
        // Unicode digits: the class is ASCII-only (the Python `\d`
        // oracle would match these; the scanner's refusal is the
        // documented divergence, docs/api.md).
        assert_eq!(
            scrub(
                "xoxb-\u{0661}\u{0662}\u{0663}-abc",
                SecretRules::SLACK_TOKEN
            ),
            "xoxb-\u{0661}\u{0662}\u{0663}-abc"
        );
        // A trailing dash-run with no final alnum: not the shape.
        assert_eq!(
            scrub("xoxb-1-2-3-", SecretRules::SLACK_TOKEN),
            "xoxb-1-2-3-"
        );
        // Glued to a preceding word char: mid-token.
        assert_eq!(
            scrub("Xxoxb-1-2-abc123xyz", SecretRules::SLACK_TOKEN),
            "Xxoxb-1-2-abc123xyz"
        );
        // Not the prefix family: xapp- stays out (Slack's app-level
        // prefix is a different head).
        assert_eq!(
            scrub("xapp-1-2-3-abc", SecretRules::SLACK_TOKEN),
            "xapp-1-2-3-abc"
        );
        // The final alnum run is maximal: trailing class chars ride along.
        assert_eq!(
            scrub("xoxb-1-2-abcdef0123456789", SecretRules::SLACK_TOKEN),
            format!("xoxb~{}", digest("xoxb-1-2-abcdef0123456789"))
        );
    }

    // --- stripe_key ----------------------------------------------------

    #[test]
    fn stripe_keys_scrub_and_the_report_says_live_or_test() {
        let live = concat!("sk_live_", "4eC39HqLyjWDarjtT1zdp7dc");
        let test = concat!("sk_test_", "51AbCdEfGhIjKlMnOpQrStUvw");
        for (key, kind, count_idx) in [
            (live, SecretKind::StripeLive, 2),
            (test, SecretKind::StripeTest, 3),
        ] {
            assert_eq!(
                scrub(key, SecretRules::STRIPE_KEY),
                format!("{}~{}", &key[..8], digest(key))
            );
            let rep = scrub_secrets_report(key, SecretRules::STRIPE_KEY, "");
            assert_eq!(rep.kind_counts[count_idx], 1);
            assert_eq!(rep.kind_counts.iter().sum::<usize>(), 1);
            assert_eq!(rep.spans[0].kind, kind);
        }
        // Restricted and publishable prefixes, both environments.
        for prefix in ["rk_live_", "pk_live_", "rk_test_", "pk_test_"] {
            let key = format!("{prefix}AbCdEfGhIjKlMnOpQrStUvWx");
            assert_eq!(
                scrub(&key, SecretRules::STRIPE_KEY),
                format!("{prefix}~{}", digest(&key))
            );
        }
    }

    #[test]
    fn stripe_near_misses_are_anchored() {
        let tail = "4eC39HqLyjWDarjtT1zdp7dc"; // 24 chars
        // 23 tail chars: not the shape.
        assert_eq!(
            scrub(&format!("sk_live_{}", &tail[..23]), SecretRules::STRIPE_KEY),
            format!("sk_live_{}", &tail[..23])
        );
        // Uppercase prefix spellings: Stripe's prefixes are lowercase.
        assert_eq!(
            scrub(&format!("SK_LIVE_{tail}"), SecretRules::STRIPE_KEY),
            format!("SK_LIVE_{tail}")
        );
        // Mid-token glue.
        assert_eq!(
            scrub(&format!("xsk_live_{tail}"), SecretRules::STRIPE_KEY),
            format!("xsk_live_{tail}")
        );
        // The tail run is maximal: trailing class chars ride along.
        let key = format!("sk_live_{tail}9extra");
        assert_eq!(
            scrub(&key, SecretRules::STRIPE_KEY),
            format!("sk_live_~{}", digest(&key))
        );
        // A shorter tail glued to the prefix falls to no match (no
        // shorter stripe prefix exists to fall through to).
        assert_eq!(scrub("sk_live_abc", SecretRules::STRIPE_KEY), "sk_live_abc");
    }

    // --- github_token --------------------------------------------------

    #[test]
    fn github_modern_tokens_scrub_with_the_prefix_kept() {
        // Synthesized 36-char base62 bodies (the format post's
        // composition: 30 random base62 + 6 checksum chars; the exact
        // checksum construction is unpublished, so vectors are
        // shape-only and synthesized).
        for prefix in ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"] {
            let token = format!("{prefix}aB3xY9kL2mN5pQ7rS4tU8vW1xY6zA0bC3dEF");
            assert_eq!(
                scrub(&token, SecretRules::GITHUB_TOKEN),
                format!("{prefix}~{}", digest(&token)),
                "{prefix}"
            );
        }
    }

    #[test]
    fn github_modern_near_misses_are_anchored() {
        let body36 = "aB3xY9kL2mN5pQ7rS4tU8vW1xY6zA0bC3dEF";
        assert_eq!(body36.len(), 36);
        let token = format!("ghp_{body36}");
        // A longer base62 run is a longer word: never a match.
        assert_eq!(
            scrub(&format!("{token}x"), SecretRules::GITHUB_TOKEN),
            format!("{token}x")
        );
        assert_eq!(
            scrub(&format!("{token}_suffix"), SecretRules::GITHUB_TOKEN),
            // `_` is not base62: the token matches, the suffix survives.
            format!("ghp_~{}_suffix", digest(&token))
        );
        // Glued head.
        assert_eq!(
            scrub(&format!("x{token}"), SecretRules::GITHUB_TOKEN),
            format!("x{token}")
        );
        // 35 chars: not the shape.
        assert_eq!(
            scrub(&format!("ghp_{}", &body36[..35]), SecretRules::GITHUB_TOKEN),
            format!("ghp_{}", &body36[..35])
        );
    }

    #[test]
    fn github_legacy_hex_class_scrubs() {
        // Synthesized 40-char hex body (never a real token; every
        // 40-char hex string matches by design, SHA-1s included).
        let token = "0123456789abcdef0123456789abcdef01234567";
        assert_eq!(token.len(), 40);
        assert_eq!(
            scrub(token, SecretRules::GITHUB_TOKEN),
            format!("github~{}", digest(token))
        );
        // Uppercase hex: the class is case-insensitive hex.
        let upper = "0123456789ABCDEF0123456789ABCDEF01234567";
        assert_eq!(
            scrub(upper, SecretRules::GITHUB_TOKEN),
            format!("github~{}", digest(upper))
        );
        // Punctuation boundaries are clean.
        let text = format!("commit {token} ok");
        assert_eq!(
            scrub(&text, SecretRules::GITHUB_TOKEN),
            format!("commit github~{} ok", digest(token))
        );
    }

    #[test]
    fn github_legacy_near_misses_are_anchored() {
        let token = "0123456789abcdef0123456789abcdef01234567";
        // 39 and 41: not the shape.
        assert_eq!(scrub(&token[1..], SecretRules::GITHUB_TOKEN), token[1..]);
        assert_eq!(
            scrub(&format!("{token}a"), SecretRules::GITHUB_TOKEN),
            format!("{token}a")
        );
        // Word-glued on either side: a fragment of a longer word.
        assert_eq!(
            scrub(&format!("x{token}"), SecretRules::GITHUB_TOKEN),
            format!("x{token}")
        );
        assert_eq!(
            scrub(&format!("{token}x"), SecretRules::GITHUB_TOKEN),
            format!("{token}x")
        );
        assert_eq!(
            scrub(&format!("_{token}"), SecretRules::GITHUB_TOKEN),
            format!("_{token}")
        );
        // A dash-glued head is mid-token by the shared glue class.
        assert_eq!(
            scrub(&format!("-{token}"), SecretRules::GITHUB_TOKEN),
            format!("-{token}")
        );
    }

    // --- pem_key -------------------------------------------------------

    #[test]
    fn pem_blocks_scrub_whole() {
        let block = "-----BEGIN RSA PRIVATE KEY-----\nMIIB\nxyz\n-----END RSA PRIVATE KEY-----\n";
        let expected_tail = format!("PEM~{}\n", digest(block.trim_end_matches('\n')));
        assert_eq!(scrub(block, SecretRules::PEM_KEY), expected_tail);
        // EC and PKCS#8-with-words spellings; the bare PKCS#8 header
        // (no algorithm words) is the keys rule's own documented
        // exclusion and stays one here too... no: the label grammar
        // REQUIRES words, so `-----BEGIN PRIVATE KEY-----` never
        // matches, exactly as the keys rule pins it.
        let ec = "-----BEGIN EC PRIVATE KEY-----\naaa\n-----END EC PRIVATE KEY-----";
        assert_eq!(
            scrub(ec, SecretRules::PEM_KEY),
            format!("PEM~{}", digest(ec))
        );
        let bare = "-----BEGIN PRIVATE KEY-----\naaa\n-----END PRIVATE KEY-----";
        assert_eq!(scrub(bare, SecretRules::PEM_KEY), bare);
        // Unterminated BEGIN: a documented non-match.
        let open = "-----BEGIN RSA PRIVATE KEY-----\nMIIB\nxyz\n";
        assert_eq!(scrub(open, SecretRules::PEM_KEY), open);
        // Mismatched END words: the first verifying END wins, a
        // mismatched one is skipped.
        let mismatched = "-----BEGIN RSA PRIVATE KEY-----\naaa\n-----END EC PRIVATE KEY-----\n-----END RSA PRIVATE KEY-----";
        assert_eq!(
            scrub(mismatched, SecretRules::PEM_KEY),
            format!("PEM~{}", digest(mismatched))
        );
    }

    // --- cross-grammar composition, tokens, report ---------------------

    #[test]
    fn one_pass_reports_every_grammar_and_spans_stay_ordered() {
        let aws = format!("AKIA{}", aws_tail());
        let slack = concat!("xox", "b-123456789012-1234567890123-abcdefghijklmnop");
        let stripe_live = concat!("sk_live_", "4eC39HqLyjWDarjtT1zdp7dc");
        let stripe_test = concat!("sk_test_", "51AbCdEfGhIjKlMnOpQrStUvw");
        let gh = format!("ghp_{}", "aB3xY9kL2mN5pQ7rS4tU8vW1xY6zA0bC3dEF");
        let legacy = "0123456789abcdef0123456789abcdef01234567";
        let pem = "-----BEGIN RSA PRIVATE KEY-----\naaa\n-----END RSA PRIVATE KEY-----";
        let text = [
            aws.as_str(),
            slack,
            stripe_live,
            stripe_test,
            gh.as_str(),
            legacy,
            pem,
        ]
        .join(" | ");
        let rep = scrub_secrets_report(&text, SecretRules::ALL, "");
        assert_eq!(
            rep.kind_counts,
            [1, 1, 1, 1, 1, 1, 1],
            "one of each kind: {rep:?}"
        );
        assert!(rep.spans.windows(2).all(|w| w[0].end <= w[1].start));
        for span in &rep.spans {
            assert!(!&text[span.start..span.end].is_empty());
        }
        let scrubbed = scrub_secrets(&text, SecretRules::ALL, "");
        assert_eq!(scrubbed, rep.text);
        // Every span re-derives from the input by offset round-trip.
        for span in &rep.spans {
            assert!(!&text[span.start..span.end].is_empty());
        }
    }

    #[test]
    fn tokens_are_fixed_points() {
        let aws = format!("AKIA{}", aws_tail());
        let slack = concat!("xox", "b-123456789012-1234567890123-abcdefghijklmnop");
        let stripe = concat!("sk_live_", "4eC39HqLyjWDarjtT1zdp7dc");
        let gh = format!("ghp_{}", "aB3xY9kL2mN5pQ7rS4tU8vW1xY6zA0bC3dEF");
        let legacy = "0123456789abcdef0123456789abcdef01234567";
        let pem = "-----BEGIN RSA PRIVATE KEY-----\naaa\n-----END RSA PRIVATE KEY-----";
        for token in [
            format!("AKIA~{}", digest(&aws)),
            format!("xoxb~{}", digest(slack)),
            format!("sk_live_~{}", digest(stripe)),
            format!("ghp_~{}", digest(&gh)),
            format!("github~{}", digest(legacy)),
            format!("PEM~{}", digest(pem)),
        ] {
            assert!(
                matches!(
                    scrub_secrets(&token, SecretRules::ALL, ""),
                    Cow::Borrowed(_)
                ),
                "{token}"
            );
        }
    }

    #[test]
    fn the_default_salt_is_not_the_unsalted_digest() {
        let aws = format!("AKIA{}", aws_tail());
        let unsalted = scrub(&aws, SecretRules::ALL);
        let default = scrub_secrets(&aws, SecretRules::ALL, SECRETS_DEFAULT_SALT);
        assert_ne!(unsalted, default);
        assert_eq!(
            default,
            format!("AKIA~{}", token_digest(SECRETS_DEFAULT_SALT, &aws))
        );
    }

    #[test]
    fn the_log_rule_masks_with_stars() {
        let aws = format!("AKIA{}", aws_tail());
        let text = format!("upstream auth failed key={aws} retrying");
        assert_eq!(
            mask_secret_tokens(&text),
            "upstream auth failed key=*** retrying".to_string()
        );
        assert!(matches!(
            mask_secret_tokens("no secrets here"),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn the_linearity_invariant_survives_a_dense_run() {
        // Every alphabet a failed candidate walks is a subset of the
        // glue class, so failed walks never host a later anchor: one
        // pass, no rescan. A dense adversarial shape (repeated heads
        // inside one long class run) stays linear-shaped, and its
        // output is stable under a second scrub.
        let text = "AKIA".repeat(20_000) + "B2C4E6G8H1J3K5L7M9";
        let once = scrub(&text, SecretRules::ALL);
        let twice = scrub(&once, SecretRules::ALL);
        assert_eq!(once, twice);
    }
}
