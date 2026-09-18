//! `scrub_pii`'s keys pass agrees with its escape-grammar twin on
//! ESCAPE-DENSE inputs: raw fuzzer strings essentially never spell the
//! escape shapes the boundary rule has to answer (`%XX`, `\uXXXX`,
//! backslash runs of every parity and length 0-6, and every accepted
//! spellings `\xHH` and octal among them), so the existing `pii` target's byte-drain
//! generator starves exactly the region where the boundary rule lives.
//! This target assembles inputs from a grammar: escape spellings glued
//! directly to key-shaped tokens (`sk-` + a key-charset tail whose
//! length brackets the openai family's 20-char minimum), with
//! key-tail-byte fillers between them, so nearly every input lands on a
//! boundary-rule question.
//!
//! The oracle is a full-output differential against a hand-synced twin
//! of the keys pass, the `pii` target's discipline narrowed to one
//! pipeline stage: the prefix-boundary rule (a prefix glued to a
//! preceding key-charset char is mid-token UNLESS a recognized escape
//! sequence ends right before it), the family table with fall-through,
//! the span families, the mask lanes, and the token digests (sha2 +
//! const-hex, spelled independently). The keys-only lane isolates the
//! stage: no email/phone interaction is possible in either machine.
//!
//! The escape grammar is the impl's own contract table, transcribed arm
//! for arm (`%XX`, `\uXXXX`, `\UHHHHHHH`, `\xHH`, octal `\NNN` maximal
//! munch, the odd-count backslash run, and the ANSI CSI spelling) — the
//! former KNOWN-UNFIXED holes (`\xHH`, octal, ANSI: the impl pinned them
//! as leaks while #100's fix carried only the first three arms) closed
//! when the impl learned those arms, and this twin learned them in the
//! same sweep, exactly the header's sync condition. The differential
//! pins impl == twin on every arm, including the escaped-backslash
//! negatives (`\\x41` stays mid-token in both machines).
//!
//! Beyond the differential: the keys-only pass is strictly idempotent
//! (a second pass is a borrowed fixed point) at every salt lane, and
//! the report twin's text is byte-identical to the scrub's.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tors::pii_impl::{KEY_FAMILY_MASK_ALL, KeyFamily, PiiRules, scrub_pii, scrub_pii_report};

// --- the twin: a hand-synced transcription of the keys pass -------------
// (copied piecewise from the `pii` target's oracle — same table, same
// boundary rule, same token construction — narrowed to keys-only lanes;
// the email/phone passes are never armed below, so their grammars are
// not transcribed here).

fn is_key_tail_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-')
}

/// The odd-backslash discipline's counter: the run of backslashes
/// ending just before `at` (the `pii` target's twin, ported).
fn backslash_run_before_at(chars: &[char], at: usize) -> usize {
    let mut run = 0usize;
    while run < at && chars[at - 1 - run] == '\\' {
        run += 1;
    }
    run
}

/// Whether the key-charset char at `i - 1` ends a complete escape
/// sequence, the boundary rule's escape recognition transcribed exactly
/// from the impl's grammar (one spelling per arm): `%XX`, `\uXXXX`
/// (position-pinned; the documented released over-trigger on an escaped
/// backslash directly before the spelling), `\UHHHHHHH`, `\xHH` and
/// maximal-munch octal `\NNN` (both pinned backslashes unescaped — an
/// odd backslash run before the spelling's first letter/digit), and the
/// odd-count backslash run `\X`. The former KNOWN-UNFIXED hole (`\xHH`,
/// octal, ANSI) closed when the impl learned those arms — the module
/// header's own condition for syncing this twin.
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
    let mut slashes = 0usize;
    while slashes + 2 <= i && chars[i - 2 - slashes] == '\\' {
        slashes += 1;
    }
    slashes % 2 == 1
}

/// The ANSI CSI spelling (`ESC [`, parameter bytes `0x20..=0x3F`, final
/// byte `0x40..=0x7E`): a colored-log quote before a key head is a clean
/// boundary. A `[31m` without the ESC is literal, mid-token.
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

#[derive(Clone, Copy)]
enum TailClass {
    Key,
    Aws,
    Azure,
}

fn tail_run_end(chars: &[char], at: usize, class: TailClass) -> usize {
    let mut end = at;
    while end < chars.len()
        && match class {
            TailClass::Key => is_key_tail_char(chars[end]),
            TailClass::Aws => chars[end].is_ascii_digit() || matches!(chars[end], 'A'..='Z'),
            TailClass::Azure => {
                chars[end].is_ascii_alphanumeric() || matches!(chars[end], '+' | '/' | '=')
            }
        }
    {
        end += 1;
    }
    end
}

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

fn starts_with_at(chars: &[char], at: usize, prefix: &str) -> bool {
    prefix
        .chars()
        .enumerate()
        .all(|(k, pc)| chars.get(at + k) == Some(&pc))
}

/// The JWT family at one position (the span families stay transcribed so
/// an accidentally-spelled `Bearer eyJ…` in glued material is scored the
/// same way the impl scores it, not skipped by the twin).
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
            return None;
        }
        if seg < 2 {
            if i >= chars.len() || chars[i] != '.' {
                return None;
            }
            i += 1;
        }
    }
    Some(i)
}

/// The PEM label's words grammar: `word( SP word)*` closed by the PGP or
/// plain ` PRIVATE KEY( BLOCK)?-----` tail, greedy terminator-first.
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

struct KeyHit {
    start: usize,
    end: usize,
    plen: usize,
    fam: usize,
    selected: bool,
}

/// The PEM BEGIN head after a dash run (the carve's twin): a head
/// opening after a dash of its own, or after a PEM word run (the
/// shared-close spelling), is a clean boundary.
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
            i += 1;
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

fn token_digest(salt: &str, matched: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(salt.as_bytes());
    hasher.update(matched.as_bytes());
    let digest = hasher.finalize();
    const_hex::encode(&digest.as_slice()[..6])
}

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
                let prefix: String = matched.chars().take(hit.plen).collect();
                out.push_str(&format!("{prefix}~{}", token_digest(salt, &matched)));
            }
        } else {
            out.extend(chars[hit.start..hit.end].iter());
        }
        pos = hit.end;
    }
    out.extend(chars[pos..].iter());
    out
}

// --- the grammar-aware generator ---------------------------------------

/// One input piece. The set is chosen so the byte-drain economy of the
/// raw `pii` generator is inverted: every piece either IS a boundary-rule
/// question or supplies the key-tail byte that makes the next key one.
#[derive(arbitrary::Arbitrary, Debug)]
enum Piece {
    /// `sk-` + a key-charset tail whose length brackets the openai
    /// family's 20-char minimum (19 = one short, 20 = exact, 21 = one
    /// over; the length rides the tail byte's low bits).
    Key,
    /// `%XY` with hex X/Y (a recognized escape), or `%zq` with the low
    /// bit choosing non-hex letters (NOT an escape: `%` itself is not a
    /// key-tail char, but the trailing letter is, so the glued key after
    /// it answers the mid-token rule through the non-escape arm).
    Pct(bool),
    /// `\uXXXX` with hex digits (recognized), or `\u00zq` (not).
    Uxx(bool),
    /// `\xHH` — the literal hex-escape spelling the impl does NOT
    /// recognize (the tracked grammar hole; the twin agrees with the
    /// impl's leak rather than wishing otherwise).
    Xhh,
    /// `\NNN` octal — also unrecognized.
    Octal,
    /// A backslash run of length 0..=6 (parity is the boundary rule's
    /// whole question: odd = one escape, even = escaped backslashes).
    Backslashes(u8),
    /// A raw ESC control byte (U+001B): not a key-tail char, so the
    /// glued key after it is a clean boundary through the plain arm.
    RawEsc,
    /// ` ` / `:` / `_` fillers (space is a clean non-tail boundary; `:`
    /// and `_` are key-tail bytes that force the mid-token arm).
    Filler(u8),
}

impl Piece {
    fn push(&self, out: &mut String, tail_byte: u8) {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        match self {
            Piece::Key => {
                let tail_len = 18 + (tail_byte as usize % 5); // 18..=22 around the 20 minimum
                out.push_str("sk-");
                for k in 0..tail_len {
                    out.push(HEX[(tail_byte as usize + k) % 16] as char);
                }
            }
            Piece::Pct(hex) => {
                out.push('%');
                if *hex {
                    out.push(HEX[tail_byte as usize % 16] as char);
                    out.push(HEX[(tail_byte as usize + 3) % 16] as char);
                } else {
                    out.push('z');
                    out.push('q');
                }
            }
            Piece::Uxx(hex) => {
                out.push_str("\\u");
                if *hex {
                    for k in 0..4 {
                        out.push(HEX[(tail_byte as usize + k) % 16] as char);
                    }
                } else {
                    out.push_str("00zq");
                }
            }
            Piece::Xhh => {
                out.push_str("\\x");
                out.push(HEX[tail_byte as usize % 16] as char);
                out.push(HEX[(tail_byte as usize + 7) % 16] as char);
            }
            Piece::Octal => {
                out.push('\\');
                for k in 1..4 {
                    out.push(char::from(b'0' + ((tail_byte as usize + k) % 8) as u8));
                }
            }
            Piece::Backslashes(n) => {
                for _ in 0..(n % 7) {
                    out.push('\\');
                }
            }
            Piece::RawEsc => out.push('\u{1b}'),
            Piece::Filler(k) => out.push(match k % 3 {
                0 => ' ',
                1 => ':',
                _ => '_',
            }),
        }
    }
}

#[derive(arbitrary::Arbitrary, Debug)]
struct Input {
    pieces: Vec<Piece>,
    /// Per-piece tail-byte source (keeps the tails varied at useful rates
    /// without spending input bytes on a per-piece byte).
    tail: u8,
}

fuzz_target!(|input: Input| {
    let mut s = String::new();
    for (idx, piece) in input.pieces.iter().enumerate() {
        piece.push(&mut s, input.tail.wrapping_add(idx as u8));
    }
    // Every generated input is non-degenerate at useful rates: pieces are
    // cheap, so even a 2-piece input usually carries a key glued to an
    // escape; nothing to shape further.

    // The keys-only lanes: mask-ALL and the single-family openai lane
    // (unselected spans are spent whole and preserved verbatim — the
    // mask decides redact vs verbatim, never detection, so the single-
    // family lane exercises the selected=false substitute arm).
    let bit = |name: &str| {
        1u16 << KeyFamily::ALL
            .iter()
            .position(|f| f.name() == name)
            .expect("the oracle names families off KeyFamily::ALL")
    };
    for mask in [KEY_FAMILY_MASK_ALL, bit("openai")] {
        let rules = PiiRules {
            email: false,
            phone: false,
            keys: true,
            key_families: mask,
        };
        for keys_salt in [
            "",
            tors::pii_impl::DEFAULT_SALT,
            tors::pii_impl::KEYS_DEFAULT_SALT,
        ] {
            let out = scrub_pii(&s, rules, "", keys_salt);
            let chars: Vec<char> = s.chars().collect();
            let expected = substitute_keys(&chars, &key_matches_of(&s, mask), keys_salt);
            assert_eq!(
                out.as_ref(),
                &expected,
                "the keys-only pass diverged from the escape-grammar twin at salt \
                 {keys_salt:?} mask {mask:#06x}"
            );

            // The report twin's text is byte-identical to the scrub's.
            let rep = scrub_pii_report(&s, rules, "", keys_salt);
            assert_eq!(
                rep.text,
                out.as_ref(),
                "scrub_pii_report text diverged from scrub_pii at salt {keys_salt:?}"
            );

            // Keys-only is strictly idempotent: a second pass is a
            // borrowed fixed point (a key token's prefix ends `-` and its
            // tail is digest hex; no family re-matches inside it).
            assert!(
                matches!(
                    scrub_pii(out.as_ref(), rules, "", keys_salt),
                    std::borrow::Cow::Borrowed(_)
                ),
                "keys-only is not idempotent on {s:?}"
            );
        }
    }
});
