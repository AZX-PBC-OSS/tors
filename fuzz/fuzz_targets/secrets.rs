//! `scrub_secrets` never panics on an arbitrary string, and its output
//! is EXACTLY the pipeline the contract describes — asserted as a
//! full-output differential, the strongest shape (the `pii` target's
//! discipline over the secret-token grammars):
//!
//! * The oracle side of this harness transcribes the five grammars in
//!   char space, per position, in the scanner's own order (the shared
//!   boundary gate, then the prefix table, then the legacy hex class,
//!   then PEM): the AWS exact-16 tail run, the Slack section grammar
//!   transcribed from the CITED detect-secrets detector as a chunk
//!   classification (`xox(?:a|b|p|o|s|r)-(?:\d+-)+[a-z0-9]+` under
//!   IGNORECASE: dash-terminated digit-run chunks are the `(?:\d+-)+`
//!   repetitions, the first remaining chunk starts the maximal
//!   `[a-z0-9]+` final run — the final class consumes digits, so a
//!   digit-led final section matches), the Stripe six-prefix table
//!   with the maximal 24+ tail, the GitHub modern exactly-36 run and
//!   the legacy exactly-40 hex run at a word boundary, and the PEM
//!   span (the label words parse, the first verifying END wins). The
//!   transform's output must be byte-identical to the oracle's splice
//!   for the all-rules pipeline and for each single-rule pipeline.
//!   This subsumes the accounting corners: span consumption, near-miss
//!   exact-width refusals, the boundary gate's mid-token refusals, and
//!   the escape carve-outs (which the oracle transcribes too). The
//!   oracle is anchored to the cited detector spellings, NOT to the
//!   implementation — a bug-for-bug transcription cannot catch grammar
//!   drift, which is how RT-SLACK-1 (the digit-led final section the
//!   section loop refused) survived 629k runs of the old oracle.
//! * The structured lane: a token-grammar-aware generator (a valid
//!   head, the section skeleton, boundary jitter) so COMPLETE token
//!   spellings — digit-led slack final sections included — occur at
//!   meaningful rates instead of the flat alphabet's combinatorial
//!   mercy. The jitter is boundary-clean (no key-charset chars, no
//!   `%`/`\` spellings), where oracle parity is exact; the
//!   hostile-glue and escape shapes ride the raw lane, whose oracle
//!   transcribes the carve-outs.
//! * The report-vs-scrub consistency: `scrub_secrets_report`'s text is
//!   the scrub's text for the same args, and every span re-derives from
//!   the INPUT (one pass, input coordinates).
//! * Strict idempotence: the second pass over any output is the
//!   identity (tokens are fixed points by construction).
//! * The identity path never lies: a borrowed return is exactly the
//!   expected output being the input.
//!
//! The oracle's digests are spelled independently (sha2 + const-hex,
//! the same hand-synced-to-root pin discipline the pii target's oracle
//! uses) so the token construction is a transcription too, not a call
//! into the code under test.

#![no_main]

use std::borrow::Cow;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tors::secret_impl::{
    SECRETS_DEFAULT_SALT, SecretKind, SecretRules, scrub_secrets, scrub_secrets_report,
};

// --- the oracle's class helpers (char space) ----------------------------

fn is_key_tail(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-')
}

fn is_aws_tail(c: char) -> bool {
    c.is_ascii_uppercase() || c.is_ascii_digit()
}

fn is_hex(c: char) -> bool {
    c.is_ascii_hexdigit()
}

fn is_legacy_glue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// The run of backslashes ending just before `at` (never crossing the
/// string start) — the odd-backslash discipline's counter.
fn backslash_run_before(chars: &[char], at: usize) -> usize {
    let mut run = 0usize;
    while run < at && chars[at - 1 - run] == '\\' {
        run += 1;
    }
    run
}

/// Whether the glue char at `i - 1` ends a complete escape sequence —
/// the char-space transcription of `pii_impl::escape_ends_before`:
/// `%XX`, `\uXXXX`, `\UHHHHHHHH`, `\xHH` (odd backslash run before the
/// `x`), `\NNN` (1-3 octal digits, maximal munch, odd run), `\X` (odd
/// run), so escaped text is a clean boundary and a partial escape is
/// not.
fn escape_ends_before_at(chars: &[char], i: usize) -> bool {
    if i >= 3
        && chars[i - 3] == '%'
        && chars[i - 2].is_ascii_hexdigit()
        && chars[i - 1].is_ascii_hexdigit()
    {
        return true;
    }
    if i >= 6
        && chars[i - 6] == '\\'
        && chars[i - 5] == 'u'
        && chars[i - 4].is_ascii_hexdigit()
        && chars[i - 3].is_ascii_hexdigit()
        && chars[i - 2].is_ascii_hexdigit()
        && chars[i - 1].is_ascii_hexdigit()
    {
        return true;
    }
    if i >= 10
        && chars[i - 10] == '\\'
        && chars[i - 9] == 'U'
        && chars[i - 8..i].iter().all(|&c| c.is_ascii_hexdigit())
    {
        return true;
    }
    if i >= 4
        && chars[i - 4] == '\\'
        && chars[i - 3] == 'x'
        && chars[i - 2].is_ascii_hexdigit()
        && chars[i - 1].is_ascii_hexdigit()
        && backslash_run_before(chars, i - 3) % 2 == 1
    {
        return true;
    }
    let mut digits = 0usize;
    while digits < i && matches!(chars[i - 1 - digits], '0'..='7') {
        digits += 1;
    }
    if (1..=3).contains(&digits)
        && digits < i
        && chars[i - 1 - digits] == '\\'
        && backslash_run_before(chars, i - digits) % 2 == 1
    {
        return true;
    }
    backslash_run_before(chars, i - 1) % 2 == 1
}

/// Whether the glue char at `i - 1` ends an ANSI CSI sequence (`ESC [`
/// params final) — the char-space transcription of
/// `pii_impl::ansi_csi_ends_before`.
fn ansi_csi_ends_before_at(chars: &[char], i: usize) -> bool {
    if !('\u{40}'..='\u{7e}').contains(&chars[i - 1]) {
        return false;
    }
    let mut j = i - 1;
    while j > 0 && ('\u{20}'..='\u{3f}').contains(&chars[j - 1]) {
        j -= 1;
    }
    j >= 2 && chars[j - 1] == '[' && chars[j - 2] == '\u{1b}'
}

/// Whether `i` opens the PEM BEGIN head after a dash run or a shared
/// close — the char-space transcription of
/// `pii_impl::pem_head_after_dash_run` (armor, not a word).
fn pem_head_after_dash_run_at(chars: &[char], i: usize) -> bool {
    let open: &str = "-----BEGIN ";
    if !open
        .chars()
        .enumerate()
        .all(|(k, pc)| chars.get(i + k) == Some(&pc))
    {
        return false;
    }
    if chars[i - 1] == '-' {
        return true;
    }
    if !chars[i - 1].is_ascii_alphanumeric() {
        return false;
    }
    let mut w = i - 1;
    while w > 0 && chars[w - 1].is_ascii_alphanumeric() {
        w -= 1;
    }
    true
}

// --- the five grammars at one position (char space) --------------------

fn starts_with_at(chars: &[char], at: usize, prefix: &str) -> bool {
    prefix
        .chars()
        .enumerate()
        .all(|(k, pc)| chars.get(at + k) == Some(&pc))
}

fn aws_match_at(chars: &[char], pos: usize) -> Option<usize> {
    for prefix in ["AKIA", "ASIA"] {
        if !starts_with_at(chars, pos, prefix) {
            continue;
        }
        let tail_start = pos + 4;
        let mut end = tail_start;
        while end < chars.len() && is_aws_tail(chars[end]) {
            end += 1;
        }
        return (end - tail_start == 16).then_some(end);
    }
    None
}

/// The Slack grammar at `pos`, transcribed from the CITED
/// detect-secrets detector (`xox(?:a|b|p|o|s|r)-(?:\d+-)+[a-z0-9]+`
/// under IGNORECASE) as a CHUNK CLASSIFICATION -- an independent
/// spelling of the section grammar, not the implementation's
/// pointer-walk shape: the post-head remainder splits on `-`; a chunk
/// that is exactly a digit run AND dash-terminated is one `(\d+-)`
/// repetition; the first remaining chunk starts the final `[a-z0-9]+`
/// run, maximal over its class (the final class CONSUMES digits, so a
/// digit-led final section matches: the RT-SLACK-1 shape). One
/// dash-terminated digit section is required and the final run must be
/// non-empty. Single pass, no backtracking (the pinned refusals: a
/// trailing dash run is not re-absorbed into the final class --
/// `xoxb-1-2-3-` stays unredacted, tests/test_secret_grammars.py).
fn slack_match_at(chars: &[char], pos: usize) -> Option<usize> {
    let fold = |c: char| c.to_ascii_lowercase();
    if chars.get(pos + 4) != Some(&'-')
        || !matches!(chars.get(pos).copied().map(fold), Some('x'))
        || chars.get(pos + 1).copied().map(fold) != Some('o')
        || chars.get(pos + 2).copied().map(fold) != Some('x')
        || !matches!(
            chars.get(pos + 3).copied().map(fold),
            Some('a' | 'b' | 'p' | 'r' | 's' | 'o')
        )
    {
        return None;
    }
    let mut i = pos + 5;
    let mut sections = 0usize;
    let final_run: Option<usize> = loop {
        // One chunk: up to the next dash (or the end).
        let chunk_start = i;
        let mut j = i;
        while j < chars.len() && chars[j] != '-' {
            j += 1;
        }
        // A chunk that is a NON-EMPTY digit run and dash-terminated is
        // one `(\d+-)` repetition (the emptiness gates the all-digits
        // test: an empty chunk between two dashes is vacuously
        // digit-free material for the final run, never a section).
        if chunk_start < j
            && chars[chunk_start..j].iter().all(|&c| c.is_ascii_digit())
            && chars.get(j) == Some(&'-')
        {
            sections += 1;
            i = j + 1;
            continue;
        }
        // The first non-section chunk starts the final `[a-z0-9]+`
        // run, maximal (it stops at the chunk's own dash or the end).
        let mut end = chunk_start;
        while end < chars.len() && chars[end].is_ascii_alphanumeric() {
            end += 1;
        }
        break (end > chunk_start).then_some(end);
    };
    if sections == 0 {
        return None;
    }
    final_run
}

fn stripe_match_at(chars: &[char], pos: usize) -> Option<(usize, SecretKind)> {
    const S: [(&str, SecretKind); 2] = [
        ("sk_live_", SecretKind::StripeLive),
        ("sk_test_", SecretKind::StripeTest),
    ];
    const R: [(&str, SecretKind); 2] = [
        ("rk_live_", SecretKind::StripeLive),
        ("rk_test_", SecretKind::StripeTest),
    ];
    const P: [(&str, SecretKind); 2] = [
        ("pk_live_", SecretKind::StripeLive),
        ("pk_test_", SecretKind::StripeTest),
    ];
    let candidates = match chars[pos] {
        's' => S,
        'r' => R,
        'p' => P,
        _ => return None,
    };
    for (prefix, kind) in candidates {
        if !starts_with_at(chars, pos, prefix) {
            continue;
        }
        let tail_start = pos + 8;
        let mut end = tail_start;
        while end < chars.len() && chars[end].is_ascii_alphanumeric() {
            end += 1;
        }
        return (end - tail_start >= 24).then_some((end, kind));
    }
    None
}

fn github_modern_match_at(chars: &[char], pos: usize) -> Option<usize> {
    for prefix in ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"] {
        if !starts_with_at(chars, pos, prefix) {
            continue;
        }
        let tail_start = pos + 4;
        let mut end = tail_start;
        while end < chars.len() && chars[end].is_ascii_alphanumeric() {
            end += 1;
        }
        return (end - tail_start == 36).then_some(end);
    }
    None
}

fn github_legacy_match_at(chars: &[char], pos: usize) -> Option<usize> {
    let mut end = pos;
    while end < chars.len() && is_hex(chars[end]) {
        end += 1;
    }
    if end - pos != 40 {
        return None;
    }
    if chars.get(end).copied().is_some_and(is_legacy_glue) {
        return None;
    }
    Some(end)
}

/// Parse a PEM label's words at `i`: `word( SP word)*` closed by
/// ` PRIVATE KEY-----` or ` PRIVATE KEY BLOCK-----`, the
/// terminator-first transcription of `pii_impl::pem_marker_end`.
/// Returns the words' span and the index past the close.
fn pem_label_at(chars: &[char], mut i: usize) -> Option<((usize, usize), usize)> {
    const CLOSE: &str = " PRIVATE KEY-----";
    const BLOCK_CLOSE: &str = " PRIVATE KEY BLOCK-----";
    let wstart = i;
    loop {
        let mut j = i;
        while j < chars.len() && chars[j].is_ascii_alphanumeric() {
            j += 1;
        }
        if j == i {
            return None;
        }
        if starts_with_at(chars, j, BLOCK_CLOSE) {
            return Some(((wstart, j), j + BLOCK_CLOSE.chars().count()));
        }
        if starts_with_at(chars, j, CLOSE) {
            return Some(((wstart, j), j + CLOSE.chars().count()));
        }
        if j < chars.len() && chars[j] == ' ' {
            i = j + 1;
            continue;
        }
        return None;
    }
}

fn pem_match_at(chars: &[char], start: usize) -> Option<usize> {
    const BEGIN: &str = "-----BEGIN ";
    const MARKER: &str = "-----END ";
    if !starts_with_at(chars, start, BEGIN) {
        return None;
    }
    let ((wstart, wend), mut k) = pem_label_at(chars, start + BEGIN.chars().count())?;
    while k + MARKER.chars().count() <= chars.len() {
        if starts_with_at(chars, k, MARKER)
            && let Some(((wstart2, wend2), end)) = pem_label_at(chars, k + MARKER.chars().count())
            && chars[wstart2..wend2] == chars[wstart..wend]
        {
            return Some(end);
        }
        k += 1;
    }
    None
}

// --- the oracle's scan + splice (char space) ----------------------------

struct Hit {
    start: usize,
    end: usize,
    kind: SecretKind,
}

fn scan_of(s: &str, rules: SecretRules) -> Vec<Hit> {
    let chars: Vec<char> = s.chars().collect();
    let mut found = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        let clean = if i == 0 || !is_key_tail(chars[i - 1]) {
            true
        } else {
            escape_ends_before_at(&chars, i)
                || ansi_csi_ends_before_at(&chars, i)
                || pem_head_after_dash_run_at(&chars, i)
        };
        let mut hit: Option<(usize, SecretKind)> = None;
        if clean {
            if rules.aws_access_key() && c == 'A' {
                hit = aws_match_at(&chars, i).map(|end| (end, SecretKind::AwsAccessKey));
            }
            if hit.is_none() && rules.slack_token() && matches!(c, 'x' | 'X') {
                hit = slack_match_at(&chars, i).map(|end| (end, SecretKind::SlackToken));
            }
            if hit.is_none() && rules.stripe_key() && matches!(c, 's' | 'r' | 'p') {
                hit = stripe_match_at(&chars, i);
            }
            if hit.is_none() && rules.github_token() && c == 'g' {
                hit = github_modern_match_at(&chars, i).map(|end| (end, SecretKind::GitHubToken));
            }
            if hit.is_none()
                && rules.github_token()
                && is_hex(c)
                && (i == 0 || !is_legacy_glue(chars[i - 1]))
            {
                hit = github_legacy_match_at(&chars, i)
                    .map(|end| (end, SecretKind::GitHubLegacyToken));
            }
            if hit.is_none() && rules.pem_key() && c == '-' {
                hit = pem_match_at(&chars, i).map(|end| (end, SecretKind::PemPrivateKey));
            }
        }
        match hit {
            Some((end, kind)) => {
                found.push(Hit {
                    start: i,
                    end,
                    kind,
                });
                i = end;
            }
            None => i += 1,
        }
    }
    found
}

/// The token digest, spelled independently of the transform: sha256
/// (salt || matched) truncated to 12 lowercase hex chars (the pii
/// target's own hand-synced pin discipline).
fn token_digest(salt: &str, matched: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(salt.as_bytes());
    hasher.update(matched.as_bytes());
    let digest = hasher.finalize();
    const_hex::encode(&digest.as_slice()[..6])
}

fn head_of(s: &str, hit: &Hit) -> String {
    let chars: Vec<char> = s.chars().collect();
    match hit.kind {
        SecretKind::AwsAccessKey | SecretKind::SlackToken | SecretKind::GitHubToken => {
            chars[hit.start..hit.start + 4].iter().collect()
        }
        SecretKind::StripeLive | SecretKind::StripeTest => {
            chars[hit.start..hit.start + 8].iter().collect()
        }
        SecretKind::GitHubLegacyToken => "github".to_owned(),
        SecretKind::PemPrivateKey => "PEM".to_owned(),
    }
}

fn expected_secrets(s: &str, rules: SecretRules, salt: &str) -> (String, Vec<(usize, usize)>) {
    let chars: Vec<char> = s.chars().collect();
    let hits = scan_of(s, rules);
    let mut out = String::with_capacity(chars.len());
    let mut pos = 0usize;
    let mut spans = Vec::new();
    for hit in &hits {
        if hit.start > pos {
            out.extend(chars[pos..hit.start].iter());
        }
        let matched: String = chars[hit.start..hit.end].iter().collect();
        out.push_str(&head_of(s, hit));
        out.push('~');
        out.push_str(&token_digest(salt, &matched));
        spans.push((hit.start, hit.end));
        pos = hit.end;
    }
    out.extend(chars[pos..].iter());
    (out, spans)
}

// --- the structured lane: per-grammar token skeletons --------------------

/// Which grammar a structured case spells.
#[derive(Clone, Copy, Debug, Arbitrary)]
enum Grammar {
    Slack,
    Aws,
    Stripe,
    GithubModern,
    GithubLegacy,
}

/// Boundary-clean jitter alphabet: punctuation and CJK only — nothing
/// in the shared glue class, no `%`/`\` escape spellings, no dash — so
/// a generated token sits at a clean boundary, where oracle parity is
/// exact (the hostile-glue and escape shapes ride the raw lane).
const JITTER: &[u8] = " \t=,;:.!?()[]{}\"'`|\u{e9}\u{3042}\u{1f600}".as_bytes();
const DIGITS: &[u8] = b"0123456789";
const B62: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const AWS_TAIL: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const HEX: &[u8] = b"0123456789abcdefABCDEF";

/// Spell `len` chars of `alphabet` from the entropy slice (deterministic
/// in the bytes; falls back to a cycling filler when it runs dry).
fn spell(entropy: &mut &[u8], alphabet: &[u8], len: usize) -> String {
    (0..len)
        .map(|k| match entropy.first() {
            Some(&e) => {
                *entropy = &entropy[1..];
                alphabet[e as usize % alphabet.len()] as char
            }
            None => alphabet[k % alphabet.len()] as char,
        })
        .collect()
}

/// A token-grammar-aware case: a complete per-grammar token skeleton
/// (valid head + section structure) with boundary jitter, so complete
/// token spellings — digit-led slack final sections (the RT-SLACK-1
/// leak shape) included — occur at meaningful rates, not at the flat
/// alphabet's combinatorial mercy (629k raw runs never spelled one).
#[derive(Debug, Arbitrary)]
struct TokenCase {
    grammar: Grammar,
    /// Slack digit-section count bias and section width bias.
    sections: u8,
    width: u8,
    /// Slack final-run length bias.
    final_len: u8,
    /// Slack final run starts with a digit (the leak direction).
    digit_led: bool,
    pre_len: u8,
    post_len: u8,
    entropy: Vec<u8>,
}

impl TokenCase {
    /// The bare token and the token in boundary-clean jitter: two
    /// inputs, both run through every rule lane.
    fn expand(self) -> (String, String) {
        let mut ent = self.entropy.as_slice();
        let token = match self.grammar {
            Grammar::Slack => {
                const HEADS: &[&str] = &[
                    "xoxb", "xoxp", "xoxa", "xoxo", "xoxs", "xoxr", "XOXB", "XOXP",
                ];
                let head = match ent.first() {
                    Some(&e) => {
                        ent = &ent[1..];
                        HEADS[e as usize % HEADS.len()]
                    }
                    None => HEADS[0],
                };
                let n_sections = 1 + (self.sections % 4) as usize;
                let section_width = 1 + (self.width % 13) as usize;
                let section_list: Vec<String> = (0..n_sections)
                    .map(|_| spell(&mut ent, DIGITS, section_width))
                    .collect();
                let final_width = 1 + (self.final_len % 40) as usize;
                let final_run = if self.digit_led {
                    spell(&mut ent, DIGITS, 1) + &spell(&mut ent, B62, final_width - 1)
                } else {
                    spell(&mut ent, B62, final_width)
                };
                format!("{head}-{}-{final_run}", section_list.join("-"))
            }
            Grammar::Aws => {
                let head = match ent.first() {
                    Some(&e) => {
                        ent = &ent[1..];
                        if e & 1 == 0 { "AKIA" } else { "ASIA" }
                    }
                    None => "AKIA",
                };
                format!("{head}{}", spell(&mut ent, AWS_TAIL, 16))
            }
            Grammar::Stripe => {
                const PREFIXES: &[&str] = &[
                    "sk_live_", "sk_test_", "rk_live_", "rk_test_", "pk_live_", "pk_test_",
                ];
                let prefix = match ent.first() {
                    Some(&e) => {
                        ent = &ent[1..];
                        PREFIXES[e as usize % PREFIXES.len()]
                    }
                    None => PREFIXES[0],
                };
                let tail_width = 24 + (self.width % 24) as usize;
                format!("{prefix}{}", spell(&mut ent, B62, tail_width))
            }
            Grammar::GithubModern => {
                const PREFIXES: &[&str] = &["ghp_", "gho_", "ghu_", "ghs_", "ghr_"];
                let prefix = match ent.first() {
                    Some(&e) => {
                        ent = &ent[1..];
                        PREFIXES[e as usize % PREFIXES.len()]
                    }
                    None => PREFIXES[0],
                };
                format!("{prefix}{}", spell(&mut ent, B62, 36))
            }
            Grammar::GithubLegacy => spell(&mut ent, HEX, 40),
        };
        let pre = spell(&mut ent, JITTER, (self.pre_len % 5) as usize);
        let post = spell(&mut ent, JITTER, (self.post_len % 5) as usize);
        (token.clone(), format!("{pre}{token}{post}"))
    }
}

/// One structured or raw case: the raw lane keeps the hostile shapes
/// (glue, escapes, partial spellings); the token lane guarantees
/// complete grammar spellings.
#[derive(Debug, Arbitrary)]
enum FuzzInput {
    Raw(String),
    Token(TokenCase),
}

fn run_lanes(s: &str) {
    let aws_slack = {
        let mut r = SecretRules::EMPTY;
        r |= SecretRules::AWS_ACCESS_KEY;
        r |= SecretRules::SLACK_TOKEN;
        r
    };
    let rules_lanes = [
        SecretRules::ALL,
        aws_slack,
        SecretRules::STRIPE_KEY,
        SecretRules::GITHUB_TOKEN,
        SecretRules::PEM_KEY,
    ];
    for rules in rules_lanes {
        let (expected, _) = expected_secrets(s, rules, SECRETS_DEFAULT_SALT);
        let out = scrub_secrets(s, rules, SECRETS_DEFAULT_SALT);
        assert_eq!(
            out.as_ref(),
            expected.as_str(),
            "the secrets scrub diverged from the oracle for rules {:?}",
            rules
        );

        // The report-vs-scrub consistency + the span round-trip: the
        // report's text is the scrub's text, and every span re-derives
        // a non-empty shape from the INPUT (one pass, input coords;
        // char indices here — the oracle's units).
        let rep = scrub_secrets_report(s, rules, SECRETS_DEFAULT_SALT);
        assert_eq!(rep.text, out.as_ref());
        let mut cursor = 0usize;
        let mut rebuilt = String::new();
        for span in &rep.spans {
            assert!(span.start >= cursor);
            rebuilt.push_str(&s[cursor..span.start]);
            rebuilt.push_str(&s[span.start..span.end]);
            cursor = span.end;
        }
        rebuilt.push_str(&s[cursor..]);
        assert_eq!(rebuilt, s, "the report's spans do not tile the input");

        // Strict idempotence: the second pass is the identity object.
        let once = out.into_owned();
        let twice = scrub_secrets(&once, rules, SECRETS_DEFAULT_SALT);
        assert!(
            matches!(twice, Cow::Borrowed(_)) && twice == once,
            "scrub_secrets is not idempotent for rules {:?} on {s:?}",
            rules
        );
    }
}

fuzz_target!(|input: FuzzInput| {
    match input {
        FuzzInput::Raw(s) => run_lanes(&s),
        FuzzInput::Token(case) => {
            let (token, jittered) = case.expand();
            run_lanes(&token);
            run_lanes(&jittered);
        }
    }
});
