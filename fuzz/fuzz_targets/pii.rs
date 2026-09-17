//! `scrub_pii` never panics on an arbitrary string, and its output is
//! EXACTLY the pipeline the contract describes — asserted as a
//! full-output differential, the strongest shape:
//!
//! * The oracle side of this harness transcribes the two quoted
//!   grammars plus both extensions past the source (the domestic
//!   matcher and the api-key families; char-space, per-position, the
//!   regex engine's own order of operations — try every start, greedy
//!   runs, backtracked split/final-digit — spelled nothing like the
//!   byte scanners the transform drives) with its own Nd table, then
//!   computes the expected output by running the transform's own
//!   pipeline order: the keys pass over the input (its own salt —
//!   `salt=None` resolves per rule, so the defaults lane digests keys
//!   with `KEYS_DEFAULT_SALT` and contacts with `DEFAULT_SALT`), then
//!   the email pass over the keys result, then the phone grammar over
//!   the email pass's result, token breakers and all. The transform's
//!   output must be byte-identical at every salt lane, for the
//!   all-rules pipeline and for each single-rule pipeline. This
//!   subsumes every accounting corner the earlier survivor-counting
//!   shape could not express: an email pass eating a phone-shaped
//!   local part whole (`440..1.0III0@…` consuming `…440..1`), an email
//!   token's verbatim domain carrying a phone shape the phone pass
//!   cannot scrub (glued to hex-alphabet letters), the digest-hex
//!   reconstruction corners (a token's hex tail is local-part material,
//!   so a token followed by `@`-shaped text can re-spell an eaten
//!   match), and the keys-rule corners (a key tail swallowing a
//!   phone-shaped digit run, a key token's digest hex feeding the email
//!   pass) — all of those are exact output, not counted absence.
//! * Convergence, structurally: the second pass over the differential
//!   output is a fixed point (a third pass is the identity), and
//!   phone-only and keys-only are strictly idempotent.
//! * The identity path never lies: a borrowed return is exactly the
//!   expected output being the input (no match existed on either side
//!   of the differential).
//!
//! The keys grammar here transcribes the scanner EXACTLY — the family
//! table in longest-prefix-first order with fall-through on a too-short
//! tail, the per-family tail alphabets (the shared key charset, the AWS
//! uppercase-only run, the Azure base64-plus-`=` run), the PEM span
//! (words-pinned BEGIN/END markers, the first verifying END wins) and
//! the JWT segment grammar behind their markers, the prefix-boundary
//! rule (a prefix glued to a preceding key-charset char is mid-token,
//! before any family), and the mask selection (the first holding
//! grammar wins; an unselected span is spent whole and preserved
//! verbatim) — so a drift on either side fails as a differential
//! mismatch naming the input.
//!
//! The oracle's digests are spelled independently (sha2 + const-hex,
//! the same hand-synced-to-root pin discipline the normalize target's
//! oracle uses) so the token construction is a transcription too, not
//! a call into the code under test.

#![no_main]

use std::borrow::Cow;

use libfuzzer_sys::fuzz_target;
use tors::pii_impl::{KEY_FAMILY_MASK_ALL, KeyFamily, PiiRules, scrub_pii, scrub_pii_report};

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

/// Whether `s` starts with `prefix` at char index `at` — the char-space
/// prefix compare the key family table drives.
fn starts_with_at(chars: &[char], at: usize, prefix: &str) -> bool {
    prefix
        .chars()
        .enumerate()
        .all(|(k, pc)| chars.get(at + k) == Some(&pc))
}

/// The key-tail charset every family shares (and the JWT segments'
/// base64url): `[A-Za-z0-9_-]`, char-space twin of the scanner's byte
/// class.
fn is_key_tail_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-')
}

/// The run of backslashes ending just before `at` (never crossing the
/// string start) — the char-space twin of the scanner's
/// `backslash_run_before`, the odd-backslash discipline's counter.
fn backslash_run_before_at(chars: &[char], at: usize) -> usize {
    let mut run = 0usize;
    while run < at && chars[at - 1 - run] == '\\' {
        run += 1;
    }
    run
}

/// Whether the key-charset char at `i - 1` ends a complete escape
/// sequence, the char-space twin of the scanner's `escape_ends_before`:
/// the same grammar, one spelling per arm — `%XX` (`%` + two hex
/// digits), `\uXXXX` (`\` `u` + four hex digits, no odd-backslash
/// recount: the documented released over-trigger on an escaped
/// backslash directly before the spelling), `\UHHHHHHHH` (`\` `U` +
/// eight hex digits, the same position-pinned discipline), `\xHH`
/// (`\` `x` + two hex digits, the backslash run before the `x` ODD),
/// `\NNN` (`\` + 1-3 octal digits, maximal munch, the run ending exactly
/// here and the backslash run before it ODD), and `\X` (any byte after
/// an ODD backslash run). Escaped text is a CLEAN boundary, the escape's
/// tail byte being formatting material, not a word; a partial escape is
/// not a boundary, and an escaped backslash is a literal keeping its
/// neighbor mid-token.
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
        && backslash_run_before_at(chars, i - 3) % 2 == 1
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
        && backslash_run_before_at(chars, i - digits) % 2 == 1
    {
        return true;
    }
    backslash_run_before_at(chars, i - 1) % 2 == 1
}

/// Whether the key-charset char at `i - 1` ends an ANSI CSI escape
/// sequence (`ESC [ params final`) — the raw-ESC arm of the escape
/// grammar, the char-space twin of the scanner's `ansi_csi_ends_before`:
/// the final byte `U+0040..=U+007E`, the walk back over the
/// parameter/intermediate class (`U+0020..=U+003F`), and the `ESC [`
/// head directly before the walked run (a `[` in prose without the ESC
/// byte never carves).
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

/// The family indices into `KeyFamily::ALL` — the spec's name order
/// (`openai`, `anthropic`, `google`, `fireworks`, `modal`, `github`,
/// `minted`, `jwt`, `aws`, `xai`, `gcp_oauth`, `pem`, `azure`,
/// `gitlab`), the same hand-synced pin discipline as the Nd table
/// above; bit `i` of the selection mask is family `ALL[i]`
/// (`KEY_FAMILY_MASK_ALL` is all fourteen).
const FAM_OPENAI: usize = 0;
const FAM_ANTHROPIC: usize = 1;
const FAM_GOOGLE: usize = 2;
const FAM_FIREWORKS: usize = 3;
const FAM_MODAL: usize = 4;
const FAM_GITHUB: usize = 5;
const FAM_MINTED: usize = 6;
const FAM_JWT: usize = 7;
const FAM_AWS: usize = 8;
const FAM_XAI: usize = 9;
const FAM_GCP_OAUTH: usize = 10;
const FAM_PEM: usize = 11;
const FAM_AZURE: usize = 12;
const FAM_GITLAB: usize = 13;

/// The per-family tail alphabet: most families share the key charset,
/// AWS access-key IDs are uppercase-plus-digits only, Azure storage
/// secrets run the base64-plus-`=` alphabet. The span families (JWT,
/// PEM) carry no tail run and never appear in this table — a new
/// tail-run family is one row below plus one arm in `tail_run_end`,
/// nothing else moves.
#[derive(Clone, Copy)]
enum TailClass {
    Key,
    Aws,
    Azure,
}

/// The AWS tail alphabet `[0-9A-Z]`: lowercase ends the run (an `AKIA`
/// head glued to lowercase tail material is a shorter, non-matching
/// head — the scanner's own maximal-run spelling).
fn is_aws_tail_char(c: char) -> bool {
    c.is_ascii_digit() || matches!(c, 'A'..='Z')
}

/// The Azure tail alphabet `[A-Za-z0-9+/=]`: the connection-string
/// secret alphabet, padding and trailing `=` consumed into the run.
fn is_azure_tail_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=')
}

/// The maximal tail run for one alphabet class from `at` — the
/// char-space twin of the scanner's byte run.
fn tail_run_end(chars: &[char], at: usize, class: TailClass) -> usize {
    let mut end = at;
    while end < chars.len()
        && match class {
            TailClass::Key => is_key_tail_char(chars[end]),
            TailClass::Aws => is_aws_tail_char(chars[end]),
            TailClass::Azure => is_azure_tail_char(chars[end]),
        }
    {
        end += 1;
    }
    end
}

/// The family table, the scanner's own order: (literal prefix, minimum
/// tail length, family index, tail alphabet), LONGEST-PREFIX-FIRST with
/// fall-through on a too-short tail. An independent transcription of
/// `pii_impl::KEY_FAMILIES` — the same hand-synced pin discipline as
/// the Nd table above. Rows whose prefixes share no head byte can never
/// tie at one position, so the order among them is immaterial; only the
/// `sk-` prefix chain (`sk-svcacct-`/`sk-proj-`/`sk-ant-` over bare
/// `sk-`) exercises the fall-through, and the `gl…` rows are one
/// provider's documented token-prefix enumeration (no row a prefix of
/// another).
const KEY_FAMILIES: &[(&str, usize, usize, TailClass)] = &[
    ("_gitlab_session=", 40, FAM_GITLAB, TailClass::Azure),
    ("github_pat_", 22, FAM_GITHUB, TailClass::Key),
    ("sk-svcacct-", 20, FAM_OPENAI, TailClass::Key),
    ("AccountKey=", 40, FAM_AZURE, TailClass::Azure),
    ("sk-proj-", 20, FAM_OPENAI, TailClass::Key),
    ("sk-ant-", 20, FAM_ANTHROPIC, TailClass::Key),
    ("azxdev_", 20, FAM_MINTED, TailClass::Key),
    ("glpat-", 20, FAM_GITLAB, TailClass::Key),
    ("glagent-", 20, FAM_GITLAB, TailClass::Key),
    ("glsoat-", 20, FAM_GITLAB, TailClass::Key),
    ("glrtr-", 20, FAM_GITLAB, TailClass::Key),
    ("glcbt-", 20, FAM_GITLAB, TailClass::Key),
    ("glptt-", 20, FAM_GITLAB, TailClass::Key),
    ("glimt-", 20, FAM_GITLAB, TailClass::Key),
    ("gloas-", 20, FAM_GITLAB, TailClass::Key),
    ("glft-", 20, FAM_GITLAB, TailClass::Key),
    ("gldt-", 20, FAM_GITLAB, TailClass::Key),
    ("glrt-", 20, FAM_GITLAB, TailClass::Key),
    ("glwt-", 20, FAM_GITLAB, TailClass::Key),
    ("glffct-", 20, FAM_GITLAB, TailClass::Key),
    ("ya29.", 20, FAM_GCP_OAUTH, TailClass::Key),
    ("1//", 20, FAM_GCP_OAUTH, TailClass::Key),
    ("ghp_", 36, FAM_GITHUB, TailClass::Key),
    ("gho_", 36, FAM_GITHUB, TailClass::Key),
    ("ghu_", 36, FAM_GITHUB, TailClass::Key),
    ("ghs_", 36, FAM_GITHUB, TailClass::Key),
    ("ghr_", 36, FAM_GITHUB, TailClass::Key),
    ("AIza", 35, FAM_GOOGLE, TailClass::Key),
    ("AKIA", 16, FAM_AWS, TailClass::Aws),
    ("ASIA", 16, FAM_AWS, TailClass::Aws),
    ("A3T", 17, FAM_AWS, TailClass::Aws),
    ("AGPA", 16, FAM_AWS, TailClass::Aws),
    ("AIDA", 16, FAM_AWS, TailClass::Aws),
    ("AIPA", 16, FAM_AWS, TailClass::Aws),
    ("ANPA", 16, FAM_AWS, TailClass::Aws),
    ("ANVA", 16, FAM_AWS, TailClass::Aws),
    ("AROA", 16, FAM_AWS, TailClass::Aws),
    ("xai-", 20, FAM_XAI, TailClass::Key),
    ("fw-", 20, FAM_FIREWORKS, TailClass::Key),
    ("fw_", 20, FAM_FIREWORKS, TailClass::Key),
    ("ak-", 20, FAM_MODAL, TailClass::Key),
    ("wk-", 20, FAM_MODAL, TailClass::Key),
    ("wd-", 43, FAM_MINTED, TailClass::Key),
    ("cn-", 20, FAM_MINTED, TailClass::Key),
    ("sk-", 20, FAM_OPENAI, TailClass::Key),
    ("w-", 43, FAM_MINTED, TailClass::Key),
];

/// The JWT family at one position, char-space: the `Bearer eyJ` marker,
/// then three maximal `[A-Za-z0-9_-]+` segments single-dot separated
/// (the marker consumed the first segment's `eyJ` head, so at least one
/// more charset char is required before the first dot). The match END,
/// or None.
fn jwt_match_at_chars(chars: &[char], start: usize) -> Option<usize> {
    const MARKER: &str = "Bearer eyJ";
    if !starts_with_at(chars, start, MARKER) {
        return None;
    }
    let mut i = start + MARKER.len();
    for seg in 0..3 {
        let run_start = i;
        while i < chars.len() && is_key_tail_char(chars[i]) {
            i += 1;
        }
        if i == run_start {
            return None; // an empty segment: the grammar's `+` is one-or-more
        }
        if seg < 2 {
            if i >= chars.len() || chars[i] != '.' {
                return None; // the single dot into the next segment
            }
            i += 1;
        }
    }
    Some(i)
}

/// Parse a PEM label's words at `i`: `word( SP word)*` closed by
/// ` PRIVATE KEY BLOCK-----` (the PGP label's own) or
/// ` PRIVATE KEY-----`, the greedy terminator-first transcription — at
/// each word end the closes are tried (BLOCK first) before the single
/// space into the next word, so `RSA PRIVATE KEY-----` closes with
/// words `RSA` while bare `PRIVATE KEY-----` (words would have to be
/// empty) and double-spaced labels fail. Returns the words' span (for
/// the BEGIN/END equality check) and the index past the closing dashes.
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
            return None; // an empty word: a leading or double space
        }
        if starts_with_at(chars, j, BLOCK_CLOSE) {
            return Some(((wstart, j), j + BLOCK_CLOSE.chars().count()));
        }
        if starts_with_at(chars, j, CLOSE) {
            return Some(((wstart, j), j + CLOSE.chars().count()));
        }
        if j < chars.len() && chars[j] == ' ' {
            i = j + 1; // exactly one single space into the next word
            continue;
        }
        return None;
    }
}

/// The PEM family at one position, char-space: `-----BEGIN `, the words
/// grammar, any body chars including newlines, then the FIRST
/// `-----END ` whose words parse AND equal the BEGIN words (an earlier
/// END naming other words is skipped, not fatal; an END whose words
/// fail to parse is skipped the same way — mismatch or absence alike
/// is a non-match). The match END, or None.
///
/// Adversarial-time note: entry is gated on the literal `-----BEGIN `
/// (prose pays one prefix compare per position, nothing more), and the
/// END search is a single monotonic forward pass — each `-----END `
/// candidate is visited once, stepping one char so overlapping markers
/// stay exact. There is deliberately NO body-length cap: "any chars" is
/// transcribed literally, so `B` BEGINs with no verifying END cost
/// `O(B·n)`; at fuzz scale that is noise, the same class as the
/// oracle's existing maximal-tail rescans. A cap would be a semantic
/// choice the differential must catch, not one smuggled into the oracle.
fn pem_match_at_chars(chars: &[char], start: usize) -> Option<usize> {
    const OPEN: &str = "-----BEGIN ";
    const MARKER: &str = "-----END ";
    if !starts_with_at(chars, start, OPEN) {
        return None;
    }
    let ((wstart, wend), mut k) = pem_label_at(chars, start + OPEN.chars().count())?;
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

/// One key-span hit: char-space `(start, end)`, the token-prefix length
/// (`plen` — 3 for PEM, whose token prefix is the literal `PEM`, not
/// input material), the family index into `KeyFamily::ALL`, and whether
/// the lane's mask selects it. Unselected hits are spent whole and
/// preserved verbatim downstream: never redacted, never re-entered.
struct KeyHit {
    start: usize,
    end: usize,
    plen: usize,
    fam: usize,
    selected: bool,
}

/// Whether `i` opens the PEM BEGIN marker head (`-----BEGIN `) directly
/// after a dash run or a shared close — the char-space twin of the
/// scanner's `pem_head_after_dash_run`: a preceding block's
/// `-----END …-----` close is a dash run of key-tail chars, and the
/// next block's head glued to it is armor-glued, not word-glued, so it
/// is a clean boundary (`END CERTIFICATE----------BEGIN …` must scan).
/// Two armor spellings carve, both pinned: a DASH directly before the
/// head (the close's run beyond the head's own five), and the SHARED
/// CLOSE — the head's dash run entirely the previous close's, a PEM
/// word byte directly before it (`…CERTIFICATE-----BEGIN …`): the
/// lookback walks the marker's own word class backward from that byte
/// (BEGIN heads never overlap, each ends in a space, so the walk never
/// crosses a previous head), and the carve only opens the boundary —
/// the PEM match downstream still requires both markers with the same
/// words, so only a self-validating block can redact.
fn pem_head_after_dash_run_at(chars: &[char], i: usize) -> bool {
    if !starts_with_at(chars, i, "-----BEGIN ") {
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

/// The key match set at one lane mask: every position whose grammar
/// HOLDS, in the scanner's leftmost walk. Transcribes the scanner
/// exactly — the prefix-boundary rule first (a prefix glued to a
/// preceding key-charset char is mid-token and never fires, before any
/// family; a PEM BEGIN head after a dash run is the carved-out clean
/// boundary, the twin of `pem_head_after_dash_run`), then the family
/// table longest-first with fall-through, then
/// the span families (their `-`/`B` heads share no byte with any table
/// prefix, so trying them after the table IS longest-first). The FIRST
/// holding grammar wins its span even when its family is unselected —
/// there is no fall-through past a hold on mask grounds; the mask only
/// decides redact versus spent-whole-verbatim, recorded per hit. The
/// tail run is maximal; a non-matching position advances one char.
fn key_matches_of(s: &str, mask: u16) -> Vec<KeyHit> {
    let chars: Vec<char> = s.chars().collect();
    let mut found = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if i > 0
            && is_key_tail_char(chars[i - 1])
            && !escape_ends_before_at(&chars, i)
            && !ansi_csi_ends_before_at(&chars, i)
            && !pem_head_after_dash_run_at(&chars, i)
        {
            i += 1; // a mid-token prefix: the boundary rule (an escape
            // sequence's tail char is formatting, not a word; a PEM head
            // after a dash run is armor, not one either)
            continue;
        }
        let mut hit = None;
        for &(prefix, min_tail, fam, class) in KEY_FAMILIES {
            if !starts_with_at(&chars, i, prefix) {
                continue;
            }
            let tail_start = i + prefix.chars().count();
            let tail_end = tail_run_end(&chars, tail_start, class);
            if tail_end - tail_start >= min_tail {
                hit = Some((tail_end, prefix.chars().count(), fam));
                break;
            }
            // A too-short tail falls through to the shorter prefixes.
        }
        if hit.is_none() {
            hit = pem_match_at_chars(&chars, i).map(|end| (end, "PEM".len(), FAM_PEM));
        }
        if hit.is_none() {
            hit = jwt_match_at_chars(&chars, i).map(|end| (end, "Bearer".len(), FAM_JWT));
        }
        match hit {
            Some((end, plen, fam)) => {
                found.push(KeyHit {
                    start: i,
                    end,
                    plen,
                    fam,
                    selected: mask & (1u16 << fam) != 0,
                });
                i = end;
            }
            None => i += 1,
        }
    }
    found
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

/// The phone token for a matched number: a `+`-led match keeps its
/// first three code points (the dialling prefix), `~`, the digest; any
/// other spelling gets the digest alone: its head digits are the area
/// code, the identifying half a visible prefix would surface (the
/// oracle's `_scrub_phone_token` rule).
fn phone_token(salt: &str, matched: &str) -> String {
    if matched.starts_with('+') {
        let prefix: String = matched.chars().take(3).collect();
        format!("{prefix}~{}", token_digest(salt, matched))
    } else {
        format!("~{}", token_digest(salt, matched))
    }
}

/// The key token for a matched credential: the family prefix verbatim
/// (the first `prefix_len` chars of the match — the non-secret half
/// that says which credential to rotate), `~`, the digest of the FULL
/// match.
fn key_token(salt: &str, matched: &str, prefix_len: usize) -> String {
    let prefix: String = matched.chars().take(prefix_len).collect();
    format!("{prefix}~{}", token_digest(salt, matched))
}

/// Substitute every key hit (leftmost, non-overlapping, in order):
/// selected hits become their token — the family prefix verbatim (the
/// first `plen` chars of the match, via `key_token`), except PEM whose
/// token prefix is the literal `PEM` — while unselected hits are spent
/// whole and preserved verbatim (the skipped semantics: no redaction
/// inside, the scan resumes after the end), char-space.
fn substitute_keys(chars: &[char], hits: &[KeyHit], salt: &str) -> String {
    let mut out = String::with_capacity(chars.len());
    let mut pos = 0;
    for hit in hits {
        if hit.start > pos {
            out.extend(chars[pos..hit.start].iter());
        }
        if hit.selected {
            let matched: String = chars[hit.start..hit.end].iter().collect();
            if hit.fam == FAM_PEM {
                out.push_str(&format!("PEM~{}", token_digest(salt, &matched)));
            } else {
                out.push_str(&key_token(salt, &matched, hit.plen));
            }
        } else {
            out.extend(chars[hit.start..hit.end].iter());
        }
        pos = hit.end;
    }
    out.extend(chars[pos..].iter());
    out
}

/// The expected all-rules output at one lane mask: the keys pass over
/// the input (the keys rule's own salt, the lane's family mask —
/// unselected spans flow through preserved verbatim), then the email
/// pass over the keys result, then the phone grammar over the email
/// pass's result — the transform's own pipeline order, token breakers
/// and all (each pass's match set is computed on the intermediate
/// exactly as the transform does).
fn expected_both(s: &str, contact_salt: &str, keys_salt: &str, mask: u16) -> String {
    let chars: Vec<char> = s.chars().collect();
    let keys = key_matches_of(s, mask);
    let after_keys = substitute_keys(&chars, &keys, keys_salt);
    let after_keys_chars: Vec<char> = after_keys.chars().collect();
    let emails = matches_of(&after_keys, email_match_at);
    let mid = substitute(&after_keys_chars, &emails, |m| email_token(contact_salt, m));
    let mid_chars: Vec<char> = mid.chars().collect();
    let phones = phone_matches_of(&mid);
    substitute(&mid_chars, &phones, |m| phone_token(contact_salt, m))
}

/// The expected phone-only output: the phone grammar over the raw
/// input, no other pass (the contact salt is the only one it reads).
fn expected_phone_only(s: &str, contact_salt: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let phones = phone_matches_of(s);
    substitute(&chars, &phones, |m| phone_token(contact_salt, m))
}

/// The expected keys-only output at one lane mask: the key grammar
/// over the raw input, no other pass (the keys salt is the only one it
/// reads) — selected spans tokenized, unselected spans verbatim.
fn expected_keys_only(s: &str, keys_salt: &str, mask: u16) -> String {
    let chars: Vec<char> = s.chars().collect();
    let keys = key_matches_of(s, mask);
    substitute_keys(&chars, &keys, keys_salt)
}

fuzz_target!(|s: &str| {
    // The family-selection bits, resolved off `KeyFamily::ALL` by name
    // (bit `i` is family `ALL[i]`): the mask lanes are mask-ALL (today's
    // three salt lanes, behavior unchanged), all-but-jwt, and
    // single-family aws. Lanes are multiplicative, so each lane computes
    // each output once and reuses it across its asserts.
    let bit = |name: &str| {
        1u16 << KeyFamily::ALL
            .iter()
            .position(|f| f.name() == name)
            .expect("the oracle names families off KeyFamily::ALL")
    };
    let masks = [
        KEY_FAMILY_MASK_ALL,
        KEY_FAMILY_MASK_ALL ^ bit("jwt"),
        bit("aws"),
    ];
    for mask in masks {
        // The salt lanes: unsalted ("" for every rule — the source-parity
        // spelling), an explicit string (one salt for EVERY rule — the
        // binding's explicit-salt semantic), and the per-rule defaults
        // (salt=None's resolution: contacts DEFAULT_SALT, keys
        // KEYS_DEFAULT_SALT — the lane that pins the split).
        for (contact_salt, keys_salt) in [
            ("", ""),
            (tors::pii_impl::DEFAULT_SALT, tors::pii_impl::DEFAULT_SALT),
            (
                tors::pii_impl::DEFAULT_SALT,
                tors::pii_impl::KEYS_DEFAULT_SALT,
            ),
        ] {
            let rules = PiiRules {
                email: true,
                phone: true,
                keys: true,
                key_families: mask,
            };
            // The full-output differential: the transcription's own pipeline
            // (keys pass at the lane mask, email pass over its result, phone
            // grammar over that, token breakers and all) must produce
            // byte-identical output to the transform, at every salt lane and
            // every mask lane. This subsumes survivor accounting: email-eaten
            // phone shapes, domain-borne phone shapes the phone pass cannot
            // scrub, the digest-hex reconstruction corners, and the keys-rule
            // corners (a key tail swallowing a phone-shaped digit run; a key
            // token's digest hex feeding the email pass; an unselected span
            // flowing verbatim into the contact passes) are all exact output,
            // not counted absence. A disagreement here is either a transform
            // bug or an oracle drift — the panic names the input either way.
            let out = scrub_pii(s, rules, contact_salt, keys_salt);
            assert_eq!(
                out.as_ref(),
                &expected_both(s, contact_salt, keys_salt, mask),
                "the all-rules pipeline diverged from the oracle at salts \
                 {contact_salt:?}/{keys_salt:?} mask {mask:#06x}"
            );

            // The report-vs-scrub consistency: the report's text is the
            // scrub's text for the same args, in every lane (pure Rust, no
            // marshalling — the report path cannot drift from the text path).
            let rep = scrub_pii_report(s, rules, contact_salt, keys_salt);
            assert_eq!(
                rep.text,
                out.as_ref(),
                "scrub_pii_report text diverged from scrub_pii at salts \
                 {contact_salt:?}/{keys_salt:?} mask {mask:#06x}"
            );

            // Convergence, structurally: the second pass over the
            // differential output is a fixed point (a third pass is the
            // identity).
            let twice = scrub_pii(out.as_ref(), rules, contact_salt, keys_salt);
            assert!(
                matches!(
                    scrub_pii(twice.as_ref(), rules, contact_salt, keys_salt),
                    Cow::Borrowed(_)
                ),
                "the converged output is not a fixed point on {s:?}"
            );

            // Keys-only: the key grammar over the raw input at the lane mask
            // (the keys salt), and the pass is strictly idempotent — a key
            // token is a fixed point by construction (the prefix ends
            // `-`/`_`/`=`/`.` or is `AIza`/`AKIA`/`ASIA`/`PEM`/`Bearer`,
            // the next byte is `~`, and no family prefix is spellable
            // inside 12 lowercase digest hex: every prefix carries a
            // distinctive char outside `[0-9a-f]` — an uppercase
            // `A`/`K`/`I`/`S`/`P`/`E`/`M`/`B`, a lowercase `y`, or a
            // `-`, `_`, `.`, or `=`).
            let keys_only = PiiRules {
                email: false,
                phone: false,
                keys: true,
                key_families: mask,
            };
            let konce = scrub_pii(s, keys_only, contact_salt, keys_salt);
            assert_eq!(
                konce.as_ref(),
                &expected_keys_only(s, keys_salt, mask),
                "the keys-only pipeline diverged from the oracle at salt \
                 {keys_salt:?} mask {mask:#06x}"
            );
            assert!(
                matches!(
                    scrub_pii(konce.as_ref(), keys_only, contact_salt, keys_salt),
                    Cow::Borrowed(_)
                ),
                "keys-only is not idempotent on {s:?}"
            );
            let krep = scrub_pii_report(s, keys_only, contact_salt, keys_salt);
            assert_eq!(
                krep.text,
                konce.as_ref(),
                "scrub_pii_report text diverged from keys-only scrub_pii at salt \
                 {keys_salt:?} mask {mask:#06x}"
            );

            // Phone-only: the phone grammar over the raw input, and the
            // pass is strictly idempotent. The mask is inert with keys
            // off, so this runs only on the mask-ALL lanes (today's
            // asserts, unchanged).
            if mask == KEY_FAMILY_MASK_ALL {
                let phone_only = PiiRules {
                    email: false,
                    phone: true,
                    keys: false,
                    key_families: mask,
                };
                let once = scrub_pii(s, phone_only, contact_salt, contact_salt);
                assert_eq!(
                    once.as_ref(),
                    &expected_phone_only(s, contact_salt),
                    "the phone-only pipeline diverged from the oracle at salt {contact_salt:?}"
                );
                assert!(
                    matches!(
                        scrub_pii(once.as_ref(), phone_only, contact_salt, contact_salt),
                        Cow::Borrowed(_)
                    ),
                    "phone-only is not idempotent on {s:?}"
                );
            }
        }
    }
});
