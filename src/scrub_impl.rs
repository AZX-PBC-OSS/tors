//! Named-rule log and exception-text scrubbing, the pure-Rust core of
//! `tors.scrub_log_text`: a hand-rolled, byte-identical port of the TaskQ
//! exception-text chain (`src/taskq/obs/_redact_exc.py`), the scrub a worker
//! applies to `str(exc)`/`repr(exc)`/rendered tracebacks before any of it
//! reaches a log line, a span, or an exported attribute.
//!
//! Three rules, one name each, pinned to the consumer's exact semantics
//! (the four compiled regexes this module ports are quoted in
//! `tests/reference.py` and re-synced against the live TaskQ source by
//! `tests/test_scrub_log_text_parity.py`):
//!
//! * `pg_detail_lines`: PostgreSQL `DETAIL:` lines quote caller-supplied
//!   row values, so the whole line is dropped. Two segmenters under one
//!   name — real-newline lines (`^[ \t]*DETAIL:.*$` under MULTILINE: line
//!   content deleted, the newline kept, so a blank line is left behind,
//!   and a CRLF line's `\r` is consumed with the content), and the
//!   `repr()`-flattened runs a traceback's final line carries
//!   (`\n[ \t]*DETAIL:...` with the literal two-character `\n`/`\r\n`
//!   separators, consumed up to the next escaped separator or the closing
//!   quote, which is preserved). Two behaviors the source chain treats as
//!   non-matches are pinned as specified behavior, not quietly fixed: a
//!   run with no closing quote and no trailing escaped newline is left
//!   alone, and one terminated by a real newline with no quote before it
//!   is left alone (both unreachable from `repr()` output, pinned by the
//!   differential suite so a future change that silently alters redaction
//!   behavior is a test failure).
//! * `uri_userinfo`: `scheme://user:password@host` →
//!   `scheme://user:***@host`. Scheme and username preserved verbatim,
//!   empty username handled (the `*`-quantified username class), password
//!   ending at the first `@`, and the `\b` word boundary before the scheme
//!   honored exactly (see `WORD_DEMOTE_RANGES` for the Unicode seam that
//!   makes a naive `is_alphanumeric` check under-redact).
//! * `uri_query_creds`: `[?&](password|passphrase|passwd|pwd)=value` →
//!   `[?&]name=***`. Name preserved, exact lowercase, value running to
//!   whitespace, `&`, or `@`.
//!
//! Canonical order (the chain's own application order, not a caller
//! choice): `pg_detail_lines` (real pass, then escaped pass), then
//! `uri_userinfo`, then `uri_query_creds`, each rule a whole pass over the
//! current text before the next begins. The order is a contract because
//! the rules interact: a DETAIL deletion can eat the `@` a userinfo mask
//! anchors on, and the userinfo password class claims text a param value
//! would otherwise mask separately — rule interaction is why order is
//! pinned, the same discipline `replace_many`'s order-freedom argument
//! inverts.
//!
//! The passes never rescan their own output (`re.sub`'s no-cascade
//! semantics) and only allocate when they fire, so the crate-wide `Cow`
//! identity convention holds one level up: `tors.scrub_log_text(s, rules)`
//! returns the original object exactly when no rule fires — including the
//! `***` fixed points, where a rule fires and splices to an equal value:
//! those return a fresh, equal string (fired is fired; the contract is
//! identity exactly when nothing fired, pinned in
//! `tests/test_scrub_log_text.py`).
//!
//! Two classification seams between CPython's `re` and Rust's std are
//! closed by hand, both empirically derived and both load-bearing for
//! byte-identity (a divergence in either direction is silent
//! under-redaction or over-redaction):
//!
//! * `\w`/`\b`: CPython's `\w` is `L* ∪ N* ∪ _`, while Rust's
//!   `char::is_alphanumeric` is the Unicode `Alphabetic` property (`L* ∪
//!   Nl ∪ Other_Alphabetic`) plus `N*`. The difference — the 6,167
//!   Other_Alphabetic codepoints that are neither letters nor numbers
//!   (Devanagari vowel signs, Hebrew points, circled letters, …) — is
//!   spelled out in `WORD_DEMOTE_RANGES` below; a mark directly before a
//!   scheme is a word boundary for the chain (the mask fires) even though
//!   a naive Rust check would call it a word char and skip the mask.
//! * `\s`: CPython's `\s` additionally accepts the four file-separator
//!   controls `U+001C..U+001F`, which the Unicode `White_Space` property
//!   Rust's `is_whitespace` implements does not; `is_python_space` adds
//!   them back, so a `\x1c` inside a username or param value stops the
//!   class walk exactly where the chain's walk stops.
//!
//! No regex engine at runtime and no new dependency: the scanners are
//! `memchr` (newline and delimiter positions), `memmem` (the literal
//! `"://"`, `b"\n"`, and `"DETAIL:"` needles), and small class walks. The
//! source chain's lookahead is the reason this cannot be a literal-pattern
//! API port; hand-rolled is the point. Every pass is linear in its input,
//! a contract the escaped DETAIL pass upholds by memoizing its per-line
//! scan state across the needle run — that pass's doc carries the war
//! story of why the naive per-needle recomputation was not.

use std::borrow::Cow;
use std::cmp::Ordering;

use memchr::{memchr, memchr2_iter, memmem, memrchr};

/// Chars Rust's std classifies as alphanumeric but CPython's `re` `\w`
/// (`L* ∪ N* ∪ _`) does not: the Other_Alphabetic codepoints that are
/// neither letters nor numbers. 6,167 codepoints in 273 sorted,
/// non-adjacent ranges, covering U+0345 through U+33479.
///
/// Provenance: generated 2026-09-12 by intersecting a rustc enumeration of
/// `char::is_alphanumeric()` with the running CPython 3.14's `re.compile(
/// "\\w")` (Unicode 16.0.0, the UCD revision this crate's own tables pin;
/// the crate doctrine applies — on an older interpreter the only possible
/// divergence is on codepoints that revision leaves unassigned). Regen
/// recipe, run from the repo root against a new UCD:
///
/// ```text
/// rustc -O enum.rs   # for cp in 0..=0x10FFFF: if char::from_u32(cp)
///                   # .is_some_and(|c| c.is_alphanumeric()) { println!("{cp}") }
/// python - <<'EOF'   # rust set minus re \w set, folded to ranges
/// import re
/// word = re.compile(r"\w")
/// rust = frozenset(int(l) for l in open("rust_alnum.txt"))
/// demote = sorted((rust | {0x5F}) - {cp for cp in range(0x110000) if word.match(chr(cp))})
/// # ... fold to (lo, hi) ranges and diff against this table
/// EOF
/// ```
///
/// A rustc that adopts a newer UCD can classify newly-assigned
/// Other_Alphabetic codepoints this table does not list; the crate-side
/// spot-check pins below and the TaskQ-gated differential lane are the
/// tripwires.
const WORD_DEMOTE_RANGES: &[(u32, u32)] = &[
    (0x0345, 0x0345),
    (0x0363, 0x036F),
    (0x05B0, 0x05BD),
    (0x05BF, 0x05BF),
    (0x05C1, 0x05C2),
    (0x05C4, 0x05C5),
    (0x05C7, 0x05C7),
    (0x0610, 0x061A),
    (0x064B, 0x0657),
    (0x0659, 0x065F),
    (0x0670, 0x0670),
    (0x06D6, 0x06DC),
    (0x06E1, 0x06E4),
    (0x06E7, 0x06E8),
    (0x06ED, 0x06ED),
    (0x0711, 0x0711),
    (0x0730, 0x073F),
    (0x07A6, 0x07B0),
    (0x0816, 0x0817),
    (0x081B, 0x0823),
    (0x0825, 0x0827),
    (0x0829, 0x082C),
    (0x088F, 0x088F),
    (0x0897, 0x0897),
    (0x08D4, 0x08DF),
    (0x08E3, 0x08E9),
    (0x08F0, 0x0903),
    (0x093A, 0x093B),
    (0x093E, 0x094C),
    (0x094E, 0x094F),
    (0x0955, 0x0957),
    (0x0962, 0x0963),
    (0x0981, 0x0983),
    (0x09BE, 0x09C4),
    (0x09C7, 0x09C8),
    (0x09CB, 0x09CC),
    (0x09D7, 0x09D7),
    (0x09E2, 0x09E3),
    (0x0A01, 0x0A03),
    (0x0A3E, 0x0A42),
    (0x0A47, 0x0A48),
    (0x0A4B, 0x0A4C),
    (0x0A51, 0x0A51),
    (0x0A70, 0x0A71),
    (0x0A75, 0x0A75),
    (0x0A81, 0x0A83),
    (0x0ABE, 0x0AC5),
    (0x0AC7, 0x0AC9),
    (0x0ACB, 0x0ACC),
    (0x0AE2, 0x0AE3),
    (0x0AFA, 0x0AFC),
    (0x0B01, 0x0B03),
    (0x0B3E, 0x0B44),
    (0x0B47, 0x0B48),
    (0x0B4B, 0x0B4C),
    (0x0B56, 0x0B57),
    (0x0B62, 0x0B63),
    (0x0B82, 0x0B82),
    (0x0BBE, 0x0BC2),
    (0x0BC6, 0x0BC8),
    (0x0BCA, 0x0BCC),
    (0x0BD7, 0x0BD7),
    (0x0C00, 0x0C04),
    (0x0C3E, 0x0C44),
    (0x0C46, 0x0C48),
    (0x0C4A, 0x0C4C),
    (0x0C55, 0x0C56),
    (0x0C5C, 0x0C5C),
    (0x0C62, 0x0C63),
    (0x0C81, 0x0C83),
    (0x0CBE, 0x0CC4),
    (0x0CC6, 0x0CC8),
    (0x0CCA, 0x0CCC),
    (0x0CD5, 0x0CD6),
    (0x0CDC, 0x0CDC),
    (0x0CE2, 0x0CE3),
    (0x0CF3, 0x0CF3),
    (0x0D00, 0x0D03),
    (0x0D3E, 0x0D44),
    (0x0D46, 0x0D48),
    (0x0D4A, 0x0D4C),
    (0x0D57, 0x0D57),
    (0x0D62, 0x0D63),
    (0x0D81, 0x0D83),
    (0x0DCF, 0x0DD4),
    (0x0DD6, 0x0DD6),
    (0x0DD8, 0x0DDF),
    (0x0DF2, 0x0DF3),
    (0x0E31, 0x0E31),
    (0x0E34, 0x0E3A),
    (0x0E4D, 0x0E4D),
    (0x0EB1, 0x0EB1),
    (0x0EB4, 0x0EB9),
    (0x0EBB, 0x0EBC),
    (0x0ECD, 0x0ECD),
    (0x0F71, 0x0F83),
    (0x0F8D, 0x0F97),
    (0x0F99, 0x0FBC),
    (0x102B, 0x1036),
    (0x1038, 0x1038),
    (0x103B, 0x103E),
    (0x1056, 0x1059),
    (0x105E, 0x1060),
    (0x1062, 0x1064),
    (0x1067, 0x106D),
    (0x1071, 0x1074),
    (0x1082, 0x108D),
    (0x108F, 0x108F),
    (0x109A, 0x109D),
    (0x1712, 0x1713),
    (0x1732, 0x1733),
    (0x1752, 0x1753),
    (0x1772, 0x1773),
    (0x17B6, 0x17C8),
    (0x1885, 0x1886),
    (0x18A9, 0x18A9),
    (0x1920, 0x192B),
    (0x1930, 0x1938),
    (0x1A17, 0x1A1B),
    (0x1A55, 0x1A5E),
    (0x1A61, 0x1A74),
    (0x1ABF, 0x1AC0),
    (0x1ACC, 0x1ACE),
    (0x1B00, 0x1B04),
    (0x1B35, 0x1B43),
    (0x1B80, 0x1B82),
    (0x1BA1, 0x1BA9),
    (0x1BAC, 0x1BAD),
    (0x1BE7, 0x1BF1),
    (0x1C24, 0x1C36),
    (0x1DD3, 0x1DF4),
    (0x24B6, 0x24E9),
    (0x2DE0, 0x2DFF),
    (0xA674, 0xA67B),
    (0xA69E, 0xA69F),
    (0xA7CE, 0xA7CF),
    (0xA7D2, 0xA7D2),
    (0xA7D4, 0xA7D4),
    (0xA7F1, 0xA7F1),
    (0xA802, 0xA802),
    (0xA80B, 0xA80B),
    (0xA823, 0xA827),
    (0xA880, 0xA881),
    (0xA8B4, 0xA8C3),
    (0xA8C5, 0xA8C5),
    (0xA8FF, 0xA8FF),
    (0xA926, 0xA92A),
    (0xA947, 0xA952),
    (0xA980, 0xA983),
    (0xA9B4, 0xA9BF),
    (0xA9E5, 0xA9E5),
    (0xAA29, 0xAA36),
    (0xAA43, 0xAA43),
    (0xAA4C, 0xAA4D),
    (0xAA7B, 0xAA7D),
    (0xAAB0, 0xAAB0),
    (0xAAB2, 0xAAB4),
    (0xAAB7, 0xAAB8),
    (0xAABE, 0xAABE),
    (0xAAEB, 0xAAEF),
    (0xAAF5, 0xAAF5),
    (0xABE3, 0xABEA),
    (0xFB1E, 0xFB1E),
    (0x10376, 0x1037A),
    (0x10940, 0x10959),
    (0x10A01, 0x10A03),
    (0x10A05, 0x10A06),
    (0x10A0C, 0x10A0F),
    (0x10D24, 0x10D27),
    (0x10D69, 0x10D69),
    (0x10EAB, 0x10EAC),
    (0x10EC5, 0x10EC7),
    (0x10EFA, 0x10EFC),
    (0x11000, 0x11002),
    (0x11038, 0x11045),
    (0x11073, 0x11074),
    (0x11080, 0x11082),
    (0x110B0, 0x110B8),
    (0x110C2, 0x110C2),
    (0x11100, 0x11102),
    (0x11127, 0x11132),
    (0x11145, 0x11146),
    (0x11180, 0x11182),
    (0x111B3, 0x111BF),
    (0x111CE, 0x111CF),
    (0x1122C, 0x11234),
    (0x11237, 0x11237),
    (0x1123E, 0x1123E),
    (0x11241, 0x11241),
    (0x112DF, 0x112E8),
    (0x11300, 0x11303),
    (0x1133E, 0x11344),
    (0x11347, 0x11348),
    (0x1134B, 0x1134C),
    (0x11357, 0x11357),
    (0x11362, 0x11363),
    (0x113B8, 0x113C0),
    (0x113C2, 0x113C2),
    (0x113C5, 0x113C5),
    (0x113C7, 0x113CA),
    (0x113CC, 0x113CD),
    (0x11435, 0x11441),
    (0x11443, 0x11445),
    (0x114B0, 0x114C1),
    (0x115AF, 0x115B5),
    (0x115B8, 0x115BE),
    (0x115DC, 0x115DD),
    (0x11630, 0x1163E),
    (0x11640, 0x11640),
    (0x116AB, 0x116B5),
    (0x1171D, 0x1172A),
    (0x1182C, 0x11838),
    (0x11930, 0x11935),
    (0x11937, 0x11938),
    (0x1193B, 0x1193C),
    (0x11940, 0x11940),
    (0x11942, 0x11942),
    (0x119D1, 0x119D7),
    (0x119DA, 0x119DF),
    (0x119E4, 0x119E4),
    (0x11A01, 0x11A0A),
    (0x11A35, 0x11A39),
    (0x11A3B, 0x11A3E),
    (0x11A51, 0x11A5B),
    (0x11A8A, 0x11A97),
    (0x11B60, 0x11B67),
    (0x11C2F, 0x11C36),
    (0x11C38, 0x11C3E),
    (0x11C92, 0x11CA7),
    (0x11CA9, 0x11CB6),
    (0x11D31, 0x11D36),
    (0x11D3A, 0x11D3A),
    (0x11D3C, 0x11D3D),
    (0x11D3F, 0x11D41),
    (0x11D43, 0x11D43),
    (0x11D47, 0x11D47),
    (0x11D8A, 0x11D8E),
    (0x11D90, 0x11D91),
    (0x11D93, 0x11D96),
    (0x11DB0, 0x11DDB),
    (0x11DE0, 0x11DE9),
    (0x11EF3, 0x11EF6),
    (0x11F00, 0x11F01),
    (0x11F03, 0x11F03),
    (0x11F34, 0x11F3A),
    (0x11F3E, 0x11F40),
    (0x1611E, 0x1612E),
    (0x16EA0, 0x16EB8),
    (0x16EBB, 0x16ED3),
    (0x16F4F, 0x16F4F),
    (0x16F51, 0x16F87),
    (0x16F8F, 0x16F92),
    (0x16FF0, 0x16FF6),
    (0x187F8, 0x187FF),
    (0x18D09, 0x18D1E),
    (0x18D80, 0x18DF2),
    (0x1BC9E, 0x1BC9E),
    (0x1E000, 0x1E006),
    (0x1E008, 0x1E018),
    (0x1E01B, 0x1E021),
    (0x1E023, 0x1E024),
    (0x1E026, 0x1E02A),
    (0x1E08F, 0x1E08F),
    (0x1E6C0, 0x1E6DE),
    (0x1E6E0, 0x1E6F5),
    (0x1E6FE, 0x1E6FF),
    (0x1E947, 0x1E947),
    (0x1F130, 0x1F149),
    (0x1F150, 0x1F169),
    (0x1F170, 0x1F189),
    (0x2B73A, 0x2B73F),
    (0x2CEA2, 0x2CEAD),
    (0x323B0, 0x33479),
];

/// The selected rules, as a bit set. Construction is the py layer's
/// (name parsing and validation happen under the GIL); the core only
/// consults membership.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuleSet(u8);

impl RuleSet {
    /// The empty selection: `rules=[]`, the identity.
    pub const EMPTY: Self = Self(0);
    /// Drop PostgreSQL DETAIL lines (both spellings of the separator).
    pub const PG_DETAIL_LINES: Self = Self(1);
    /// Mask `scheme://user:password@host` userinfo passwords.
    pub const URI_USERINFO: Self = Self(2);
    /// Mask password-family query parameters.
    pub const URI_QUERY_CREDS: Self = Self(4);
    /// `rules=None`: the full chain, in canonical order.
    pub const ALL: Self = Self(7);

    fn pg_detail_lines(self) -> bool {
        self.0 & Self::PG_DETAIL_LINES.0 != 0
    }

    fn uri_userinfo(self) -> bool {
        self.0 & Self::URI_USERINFO.0 != 0
    }

    fn uri_query_creds(self) -> bool {
        self.0 & Self::URI_QUERY_CREDS.0 != 0
    }
}

impl std::ops::BitOrAssign for RuleSet {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// CPython `re`'s `\w` for one char: `_`, or alphanumeric under Rust's std
/// MINUS the Other_Alphabetic codepoints Python's letter/number classes
/// reject (see [`WORD_DEMOTE_RANGES`]).
fn is_python_word(c: char) -> bool {
    c == '_' || (c.is_alphanumeric() && !word_demoted(c as u32))
}

fn word_demoted(cp: u32) -> bool {
    WORD_DEMOTE_RANGES
        .binary_search_by(|&(lo, hi)| {
            if cp < lo {
                Ordering::Greater
            } else if cp > hi {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        })
        .is_ok()
}

/// CPython `re`'s `\s` for one char: the Unicode `White_Space` property
/// Rust's `is_whitespace` implements, plus the four file-separator
/// controls `U+001C..U+001F` that CPython accepts and the property does
/// not (the empirically-verified whole of the disagreement).
fn is_python_space(c: char) -> bool {
    matches!(c, '\u{1c}'..='\u{1f}') || c.is_whitespace()
}

/// A byte of the scheme grammar `[a-zA-Z0-9+.-]` (the same grammar the
/// autolink parser in `gfm_strip_impl` uses; ASCII, so a byte check is a
/// char check here).
fn is_scheme_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.')
}

/// Drop every real-newline line whose content starts with optional
/// spaces/tabs and the literal `DETAIL:` — the whole line to (not
/// including) its newline, so a blank line is left behind; a CRLF line's
/// `\r` is part of the deleted content (`.` consumes it). Byte-identical
/// to `re.compile(r"^[ \t]*DETAIL:.*$", re.MULTILINE).sub("", text)`.
fn drop_detail_lines(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();
    let mut out: Option<String> = None;
    // End of the last emitted region: deletions are the line contents, so
    // a deletion advances the cursor to the line's newline, which stays.
    let mut cursor = 0usize;
    let mut line_start = 0usize;
    loop {
        let line_end = memchr(b'\n', &bytes[line_start..]).map_or(bytes.len(), |i| line_start + i);
        let mut prefix = line_start;
        while prefix < line_end && matches!(bytes[prefix], b' ' | b'\t') {
            prefix += 1;
        }
        if text[prefix..line_end].starts_with("DETAIL:") {
            out.get_or_insert_with(|| String::with_capacity(text.len()))
                .push_str(&text[cursor..line_start]);
            cursor = line_end;
        }
        if line_end == bytes.len() {
            break;
        }
        line_start = line_end + 1;
    }
    match out {
        Some(mut o) => {
            o.push_str(&text[cursor..]);
            Cow::Owned(o)
        }
        None => Cow::Borrowed(text),
    }
}

/// The first index of `[line_start, line_end]`'s trailing Python-whitespace
/// run: everything from it to the line's end is `\s`, so a `\s*` that
/// starts at or after it reaches the `$` (the newline or end of text) and
/// the closing-quote lookahead alternative succeeds. `line_end` itself
/// when the line has no trailing whitespace.
fn trailing_ws_start(text: &str, line_start: usize, line_end: usize) -> usize {
    let mut start = line_end;
    for (i, c) in text[line_start..line_end].char_indices().rev() {
        if is_python_space(c) {
            start = line_start + i;
        } else {
            break;
        }
    }
    start
}

/// Drop the `repr()`-flattened DETAIL runs: from a literal `\n` (or
/// `\r\n`) separator, through optional spaces/tabs and the literal
/// `DETAIL:`, up to — not including — the first position where the chain's
/// lookahead succeeds: another escaped separator, or a quote whose
/// optional `)` is followed by only whitespace to the end of the line
/// (which is what preserves a repr's closing `')"`). The run cannot cross
/// a real newline (`.` does not match one), so a run with no terminator
/// before its line's end is left alone — one of the two pinned
/// non-matches. Byte-identical to
/// `re.compile(r"(?:\\r)?\\n[ \t]*DETAIL:.*?(?=(?:\\r)?\\n|['\"]\)?\s*$)",
/// re.MULTILINE).sub("", text)`.
///
/// Linear in the input — the contract this pass's needle loop owes the
/// scrub API, and the war story behind the line-state memo below. The
/// terminator scan needs three per-LINE values: the line's end (the run
/// cannot cross it), its start, and its trailing-whitespace run's start
/// (the quote lookahead's `\s*$` reach). The naive spelling recomputed all
/// three per needle — but the needles are not per-line: a flattened
/// traceback line can carry thousands of `\nDETAIL:` needles, and on that
/// one-line shape each recomputation is a full-line scan (the `memchr`
/// runs to the text's end, the `memrchr` to its start: quadratic, ~480ms
/// pre-fix on 704KB of chained needles where the chain's own lazy scan
/// takes ~11ms) while a long trailing-whitespace run adds a per-needle
/// backward walk over the same M chars (K·M). Byte-identity hid the whole
/// cliff — the differential suite was green throughout — which on a scrub
/// API is the worst kind of bug: nothing fails, the attacker-shaped log
/// line just stops being answerable.
///
/// The memo is exact, not approximate, so match semantics are untouched:
/// needles arrive sorted, and a needle that passes the `DETAIL:` check
/// owns a needle/spaces/`DETAIL:` region holding no other needle's
/// backslash, so content positions strictly increase — a memoized
/// `line_end` (the first `\n` at or after an earlier content position in
/// the same line) is therefore also the first `\n` at or after any later
/// content position still inside it, and reusing `(line_start, line_end,
/// ws_start)` while `content <= line_end` yields exactly the values the
/// per-needle recomputation would produce. Recompute windows —
/// `[prev_line_end, content)` for the start, `[content, line_end)` for
/// the end — tile the text disjointly, so every byte of line state is
/// scanned O(1) times total: the chain's own linearity class, pinned by
/// the parity suite's needle-chain lane and its timing cell.
fn drop_detail_escaped(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();
    let mut out: Option<String> = None;
    let mut cursor = 0usize;
    // The line state every needle's terminator scan consults, carried
    // across the needle run instead of recomputed per needle: the
    // (line_start, line_end) of the line holding the last needle's
    // content, and that line's trailing-ws start once a quote candidate
    // has first needed it. Valid to reuse while the current needle's
    // content lies inside the memoized line (see the doc above for why
    // that reuse is exact); reset per line crossing, the ws memo with it.
    let mut line: Option<(usize, usize)> = None;
    let mut ws_memo: Option<usize> = None;
    for nl in memmem::find_iter(text.as_bytes(), b"\\n") {
        // `nl` is the backslash of a literal `\n`. A literal `\r` directly
        // before it is consumed too: the chain's earliest-start preference
        // makes the match begin at the `\r` whenever the pair is there.
        let start = if nl >= 2 && &bytes[nl - 2..nl] == b"\\r" {
            nl - 2
        } else {
            nl
        };
        if start < cursor {
            continue;
        }
        let mut i = nl + 2;
        while i < bytes.len() && matches!(bytes[i], b' ' | b'\t') {
            i += 1;
        }
        if !text[i..].starts_with("DETAIL:") {
            continue;
        }
        let content = i + "DETAIL:".len();
        // The lazy run stops at the first lookahead hit; it cannot cross
        // the line's real newline. Same line as the memo (content inside
        // the memoized line): reuse. A later line (content past the
        // memoized end — a real `\n` was crossed): recompute, with the
        // backward start-scan bounded by the newline the memo already
        // proved, so the recompute windows stay disjoint.
        let (line_start, line_end) = match line {
            Some(cached) if content <= cached.1 => cached,
            prev => {
                let line_end =
                    memchr(b'\n', &bytes[content..]).map_or(bytes.len(), |k| content + k);
                let line_start = match prev {
                    Some((_, prev_end)) => memrchr(b'\n', &bytes[prev_end + 1..content])
                        .map_or(prev_end + 1, |k| prev_end + k + 2),
                    None => memrchr(b'\n', &bytes[..content]).map_or(0, |k| k + 1),
                };
                ws_memo = None; // a new line's trailing-ws run is a new value
                line = Some((line_start, line_end));
                (line_start, line_end)
            }
        };
        let mut q = content;
        let mut terminator = None;
        while q <= line_end {
            // Alternative 1: an escaped separator begins here (`\n`, or
            // `\r\n` whose `\r` the lookahead's optional also accepts).
            if bytes.get(q..q + 2) == Some(b"\\n") || bytes.get(q..q + 4) == Some(b"\\r\\n") {
                terminator = Some(q);
                break;
            }
            // Alternative 2: a quote, an optional `)`, then whitespace
            // only up to the end of the line. When the `)` is there the
            // without-`)` path would need the `)` itself to be `\s`, so
            // testing past it (when present) is the whole backtracking.
            if matches!(bytes.get(q), Some(b'\'') | Some(b'"')) {
                let mut after_quote = q + 1;
                if bytes.get(after_quote) == Some(&b')') {
                    after_quote += 1;
                }
                let ws_start =
                    *ws_memo.get_or_insert_with(|| trailing_ws_start(text, line_start, line_end));
                if after_quote >= ws_start {
                    terminator = Some(q);
                    break;
                }
            }
            q += text[q..].chars().next().map_or(1, char::len_utf8);
        }
        let Some(q) = terminator else { continue };
        out.get_or_insert_with(|| String::with_capacity(text.len()))
            .push_str(&text[cursor..start]);
        cursor = q;
    }
    match out {
        Some(mut o) => {
            o.push_str(&text[cursor..]);
            Cow::Owned(o)
        }
        None => Cow::Borrowed(text),
    }
}

/// Mask the password of `scheme://user:password@host` shapes: the scheme
/// run ending at a `://` (its first char a letter behind a word boundary —
/// a boundary the run's own `+`/`-`/`.` chars also create), the username
/// up to the first `:`/`/`/`@`/whitespace, and the password up to the
/// first whitespace or `@` (empty usernames and empty passwords keep the
/// chain's exact treatment: masked, and not masked at all). Byte-identical
/// to `re.compile(r"(\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s:/@]*):([^\s@]+)@")
/// .sub(r"\1:***@", text)`.
fn mask_uri_userinfo(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();
    let mut out: Option<String> = None;
    let mut cursor = 0usize;
    for anchor in memmem::find_iter(text.as_bytes(), b"://") {
        if anchor < cursor {
            continue; // inside a previous match (a password may hold "://")
        }
        // The scheme char run ending at the anchor; the match can only
        // start at a letter inside it that sits behind a word boundary —
        // at the run's start that boundary depends on the preceding char
        // (Python `\w`), inside the run only on `+`/`-`/`.` (the run's
        // letters and digits are word chars, so no boundary).
        let mut run_start = anchor;
        while run_start > 0 && is_scheme_byte(bytes[run_start - 1]) {
            run_start -= 1;
        }
        let mut p = None;
        for cand in run_start..anchor {
            if !bytes[cand].is_ascii_alphabetic() {
                continue;
            }
            let boundary = if cand == run_start {
                run_start == 0
                    || !text[..run_start]
                        .chars()
                        .next_back()
                        .is_some_and(is_python_word)
            } else {
                matches!(bytes[cand - 1], b'+' | b'-' | b'.')
            };
            if boundary {
                p = Some(cand);
                break;
            }
        }
        let Some(p) = p else { continue };
        // Username: up to the first `:` (the separator the mask needs),
        // `/`, `@`, or whitespace; empty is a real shape.
        let mut colon = None;
        for (idx, c) in text[anchor + 3..].char_indices() {
            if c == ':' {
                colon = Some(anchor + 3 + idx);
                break;
            }
            if is_python_space(c) || c == '/' || c == '@' {
                break;
            }
        }
        let Some(colon) = colon else { continue };
        // Password: one or more chars that are neither whitespace nor `@`,
        // and then the `@` the mask anchors on.
        let mut end = colon + 1;
        let mut nonempty = false;
        while let Some(c) = text[end..].chars().next() {
            if is_python_space(c) || c == '@' {
                break;
            }
            nonempty = true;
            end += c.len_utf8();
        }
        if !nonempty || !text[end..].starts_with('@') {
            continue;
        }
        let out = out.get_or_insert_with(|| String::with_capacity(text.len()));
        out.push_str(&text[cursor..p]);
        // Group 1 (scheme://user) verbatim, then the `:` the chain's
        // template re-emits, the mask, and the `@`.
        out.push_str(&text[p..colon]);
        out.push_str(":***@");
        cursor = end + 1;
    }
    match out {
        Some(mut o) => {
            o.push_str(&text[cursor..]);
            Cow::Owned(o)
        }
        None => Cow::Borrowed(text),
    }
}

/// The password-family parameter names, exact lowercase. At most one can
/// match at any one position (their prefixes diverge by the fifth char),
/// so the scan order among them is free.
const PARAM_NAMES: [&str; 4] = ["password", "passphrase", "passwd", "pwd"];

/// Mask the values of `[?&](password|passphrase|passwd|pwd)=` parameters:
/// name and delimiter kept verbatim, value (`[^\s&@]+`, greedy — an inner
/// `?`/`=` rides along, an empty value is not a mask) replaced with `***`.
/// Byte-identical to
/// `re.compile(r"([?&](?:password|passphrase|passwd|pwd)=)([^\s&@]+)")
/// .sub(r"\1***", text)`.
fn mask_uri_query_creds(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();
    let mut out: Option<String> = None;
    let mut cursor = 0usize;
    for d in memchr2_iter(b'?', b'&', bytes) {
        if d < cursor {
            continue; // inside a previous value (values may hold ? and &)
        }
        let mut value_start = None;
        for name in PARAM_NAMES {
            let after_name = d + 1 + name.len();
            if text[d + 1..].starts_with(name) && bytes.get(after_name) == Some(&b'=') {
                value_start = Some(after_name + 1);
                break;
            }
        }
        let Some(v0) = value_start else { continue };
        let mut end = v0;
        let mut nonempty = false;
        while let Some(c) = text[end..].chars().next() {
            if is_python_space(c) || c == '&' || c == '@' {
                break;
            }
            nonempty = true;
            end += c.len_utf8();
        }
        if !nonempty {
            continue;
        }
        let out = out.get_or_insert_with(|| String::with_capacity(text.len()));
        out.push_str(&text[cursor..d]);
        out.push_str(&text[d..v0]);
        out.push_str("***");
        cursor = end;
    }
    match out {
        Some(mut o) => {
            o.push_str(&text[cursor..]);
            Cow::Owned(o)
        }
        None => Cow::Borrowed(text),
    }
}

/// The full chain, in canonical order: the DETAIL rule's two passes, then
/// the userinfo mask, then the query-param mask, each pass over the
/// current text, no pass rescanning another's output. A pass that fires
/// moves the chain onto its owned output; a pass that does not fire
/// returns the borrow, and the chain stays where it was. The result is
/// borrowed — the identity lane — exactly when no rule fired.
pub fn scrub_log_text(text: &str, rules: RuleSet) -> Cow<'_, str> {
    type Pass = fn(&str) -> Cow<'_, str>;
    let passes: [Option<Pass>; 4] = [
        rules.pg_detail_lines().then_some(drop_detail_lines as Pass),
        rules.pg_detail_lines().then_some(drop_detail_escaped),
        rules.uri_userinfo().then_some(mask_uri_userinfo),
        rules.uri_query_creds().then_some(mask_uri_query_creds),
    ];
    let mut owned: Option<String> = None;
    for pass in passes.into_iter().flatten() {
        let src: &str = owned.as_deref().unwrap_or(text);
        if let Cow::Owned(out) = pass(src) {
            owned = Some(out);
        }
    }
    owned.map_or(Cow::Borrowed(text), Cow::Owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scrub(text: &str, rules: RuleSet) -> String {
        scrub_log_text(text, rules).into_owned()
    }

    #[test]
    fn word_demote_table_is_sorted_disjoint_and_populated() {
        assert!(WORD_DEMOTE_RANGES.len() == 273);
        for pair in WORD_DEMOTE_RANGES.windows(2) {
            assert!(pair[0].1 < pair[1].0, "overlapping or unsorted ranges");
        }
    }

    #[test]
    fn word_classification_spot_checks() {
        // Demoted (Other_Alphabetic, non-letter/non-number): the seam the
        // table exists for.
        for c in ['\u{0345}', '\u{093e}', '\u{24b6}', '\u{1f170}'] {
            assert!(
                word_demoted(c as u32),
                "U+{:04X} should be demoted",
                c as u32
            );
            assert!(!is_python_word(c));
        }
        // Word under both: letters, digits, `_`, CJK numerals (Numeric_Type
        // letters), Nl, No.
        for c in ['a', 'Z', '5', '_', 'é', '五', '\u{2167}', '\u{00b2}'] {
            assert!(
                is_python_word(c),
                "U+{:04X} should be a word char",
                c as u32
            );
        }
        // Non-word under both: punctuation, whitespace, controls.
        for c in ['-', '.', '+', ' ', '\t', '@', '\u{1c}'] {
            assert!(!is_python_word(c));
        }
    }

    #[test]
    fn space_classification_spot_checks() {
        // The file separators: Python \s yes, Rust is_whitespace no.
        for c in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
            assert!(is_python_space(c) && !c.is_whitespace());
        }
        // Shared: the ASCII set, NBSP, U+2028/2029.
        for c in [' ', '\t', '\n', '\r', '\u{a0}', '\u{2028}', '\u{2029}'] {
            assert!(is_python_space(c));
        }
        // Neither: ZWSP, word chars.
        for c in ['\u{200b}', 'a', '@'] {
            assert!(!is_python_space(c));
        }
    }

    #[test]
    fn detail_lines_drop_content_keep_newline() {
        assert_eq!(
            scrub("duplicate key\nDETAIL:  Key (id)=(9) exists.", RuleSet::ALL),
            "duplicate key\n"
        );
        assert_eq!(
            scrub("duplicate key\r\nDETAIL: v\r\nHINT: x", RuleSet::ALL),
            "duplicate key\r\n\nHINT: x"
        );
        assert_eq!(scrub("\t DETAIL: v\nnext", RuleSet::ALL), "\nnext");
        assert_eq!(scrub("DETAIL: x\ny", RuleSet::ALL), "\ny");
        assert_eq!(scrub("x\nDETAIL: tail", RuleSet::ALL), "x\n");
        assert_eq!(
            scrub("detail: v\nDETAILX: w", RuleSet::ALL),
            "detail: v\nDETAILX: w"
        );
    }

    #[test]
    fn detail_escaped_pinned_edges() {
        assert_eq!(
            scrub(
                "PostgresError('msg\\nDETAIL:  Key (id)=(9) exists.')",
                RuleSet::ALL
            ),
            "PostgresError('msg')"
        );
        assert_eq!(
            scrub("PostgresError('msg\\r\\nDETAIL: secret')", RuleSet::ALL),
            "PostgresError('msg')"
        );
        assert_eq!(
            scrub("E('a\\nDETAIL: one\\nDETAIL: two')", RuleSet::ALL),
            "E('a')"
        );
        assert_eq!(
            scrub("E('a\\nDETAIL: Key (x)=('val') exists.')", RuleSet::ALL),
            "E('a')"
        );
        // The pinned non-matches.
        assert_eq!(
            scrub("E('a\\nDETAIL: leaks", RuleSet::ALL),
            "E('a\\nDETAIL: leaks"
        );
        assert_eq!(
            scrub("E('a\\nDETAIL: leaks\nnext", RuleSet::ALL),
            "E('a\\nDETAIL: leaks\nnext"
        );
        assert_eq!(
            scrub("E('a\\nDETAIL: v')  tail", RuleSet::ALL),
            "E('a\\nDETAIL: v')  tail"
        );
        // Real tab prefix consumed; literal `\t` is not a prefix.
        assert_eq!(scrub("E('a\\n\tDETAIL: v')", RuleSet::ALL), "E('a')");
        assert_eq!(
            scrub("E('a\\n\\tDETAIL: v')", RuleSet::ALL),
            "E('a\\n\\tDETAIL: v')"
        );
        // Trailing-whitespace and real-newline terminations.
        assert_eq!(
            scrub("E('a\\nDETAIL: v')  \nnext", RuleSet::ALL),
            "E('a')  \nnext"
        );
        assert_eq!(
            scrub("E('a\\nDETAIL: v')\nnext", RuleSet::ALL),
            "E('a')\nnext"
        );
    }

    #[test]
    fn userinfo_pinned_edges() {
        assert_eq!(
            scrub("postgresql://worker:hunter2@db/prod", RuleSet::ALL),
            "postgresql://worker:***@db/prod"
        );
        assert_eq!(
            scrub("postgresql://:SECRET@host/db", RuleSet::ALL),
            "postgresql://:***@host/db"
        );
        assert_eq!(scrub("a://u:p@ss@h", RuleSet::ALL), "a://u:***@ss@h");
        assert_eq!(scrub("a://u:p:q@h", RuleSet::ALL), "a://u:***@h");
        assert_eq!(scrub("-st://u:pw@h", RuleSet::ALL), "-st://u:***@h");
        // No matches: empty password, digit-led run, word char before.
        for text in [
            "postgresql://user:@host/db",
            "1st://u:pw@h",
            "0https://u:pw@h",
            "éhttps://u:pw@h",
            "_https://u:pw@h",
            "a://u/v:pw@h",
            "a://u\u{a0}v:pw@h",
            "a://u\x1cv:pw@h",
            "a://u:pw host",
        ] {
            assert_eq!(scrub(text, RuleSet::ALL), text, "{text:?}");
        }
        // The demote seam: a mark before the scheme IS a boundary.
        assert_eq!(
            scrub("\u{093e}https://u:pw@h", RuleSet::ALL),
            "\u{093e}https://u:***@h"
        );
        // A scheme inside the password is masked with it.
        assert_eq!(scrub("a://u:p://q@h", RuleSet::ALL), "a://u:***@h");
    }

    #[test]
    fn query_creds_pinned_edges() {
        assert_eq!(
            scrub("?password=x&passphrase=y&passwd=z&pwd=w", RuleSet::ALL),
            "?password=***&passphrase=***&passwd=***&pwd=***"
        );
        assert_eq!(
            scrub("?password=a b?pwd=c@d&passwd=e", RuleSet::ALL),
            "?password=*** b?pwd=***@d&passwd=***"
        );
        assert_eq!(scrub("?password=a=b?c", RuleSet::ALL), "?password=***");
        assert_eq!(scrub("a?password=1?pwd=2", RuleSet::ALL), "a?password=***");
        assert_eq!(
            scrub("?password=a\x1cb", RuleSet::ALL),
            "?password=***\x1cb"
        );
        for text in ["?Password=x&passwords=y&pwd=", "?password=&x=1"] {
            assert_eq!(scrub(text, RuleSet::ALL), text, "{text:?}");
        }
    }

    #[test]
    fn canonical_order_is_the_chain_order() {
        // The DETAL deletion eats the `@`; with the full chain the userinfo
        // rule then has nothing to anchor on (under-redaction by design of
        // the chain, pinned), with userinfo alone the whole password goes.
        let text = "pg://u:p\\nDETAIL:x@h')";
        assert_eq!(scrub(text, RuleSet::ALL), "pg://u:p')");
        assert_eq!(scrub(text, RuleSet::URI_USERINFO), "pg://u:***@h')");
        assert_eq!(scrub(text, RuleSet::PG_DETAIL_LINES), "pg://u:p')");
    }

    #[test]
    fn identity_is_borrowed_exactly_when_nothing_fires() {
        assert!(matches!(
            scrub_log_text("plain text", RuleSet::ALL),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            scrub_log_text("plain", RuleSet::EMPTY),
            Cow::Borrowed(_)
        ));
        for rules in [
            RuleSet::PG_DETAIL_LINES,
            RuleSet::URI_USERINFO,
            RuleSet::URI_QUERY_CREDS,
        ] {
            assert!(matches!(
                scrub_log_text("plain text", rules),
                Cow::Borrowed(_)
            ));
        }
        // The *** fixed points fire (and allocate) even though the spliced
        // output equals the input.
        assert_eq!(scrub_log_text("a://u:***@h", RuleSet::ALL), "a://u:***@h");
        assert!(matches!(
            scrub_log_text("a://u:***@h", RuleSet::ALL),
            Cow::Owned(_)
        ));
    }

    #[test]
    fn the_chain_is_value_idempotent() {
        for text in [
            "a\nDETAIL: v\npg://u:pw@h/db?password=x",
            "E('x\\nDETAIL: K=(v)')",
            "pg://u:p\\nDETAIL:x@h')",
            "a://u:***@h",
            "?password=***",
        ] {
            let once = scrub(text, RuleSet::ALL);
            assert_eq!(scrub(&once, RuleSet::ALL), once, "{text:?}");
        }
    }

    #[test]
    fn empty_input_is_identity() {
        assert!(matches!(scrub_log_text("", RuleSet::ALL), Cow::Borrowed(_)));
    }
}
