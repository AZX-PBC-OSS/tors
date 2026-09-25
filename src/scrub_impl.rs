//! Named-rule log and exception-text scrubbing, the pure-Rust core of
//! `tors.scrub_log_text`: a hand-rolled scanner over a documented
//! five-rule grammar, for scrubbing `str(exc)`/`repr(exc)`/rendered
//! tracebacks before any of it reaches a log line, a span, or an
//! exported attribute.
//!
//! Five rules, one name each, four pinned to the documented grammar (the
//! four compiled regexes this module implements are quoted in
//! `tests/reference.py` and differentially enforced by
//! `tests/test_scrub_log_text_parity.py`; the fifth, `secret_tokens`,
//! delegates to `secret_impl`'s cited vendor grammars and splices each
//! span to `***`):
//!
//! * `pg_detail_lines`: PostgreSQL `DETAIL:` lines quote caller-supplied
//!   row values, so the whole line is dropped. Two segmenters under one
//!   name — real-newline lines (`^(?:[ \t]*[|+][ \t]*)*[ \t]*DETAIL:.*$`
//!   under MULTILINE: line content deleted, the newline kept, so a blank
//!   line is left behind, a CRLF line's `\r` is consumed with the content,
//!   and the `(?:[ \t]*[|+][ \t]*)*` gutter group absorbs the `| `/`+ `
//!   indentation `traceback.format_exception` renders for every line of an
//!   `ExceptionGroup`/`except*` sub-exception, one layer per nesting
//!   level), and the `repr()`-flattened runs a traceback's final line
//!   carries (`\n[ \t]*DETAIL:...` with the literal two-character
//!   `\n`/`\r\n` separators, consumed up to the first position where the
//!   chain's lookahead succeeds: another escaped separator, or the repr
//!   tail — a quote followed by the run of `)`/`]` closers `repr()` ends
//!   with (`')` plain, `')])` once the exception sits in an
//!   ExceptionGroup's list, one more `])` per nesting level) with only
//!   whitespace to the end of the line, which is what preserves a repr's
//!   trailing `')"`.
//!
//!   > [!WARNING]
//!   > SECURITY-POLICY CHANGE (issue #107), inverting the pinned 0.7.0
//!   > behavior: the lookahead's final bare `$` leg is FAIL-CLOSED. An
//!   > escaped DETAIL run whose tail matches NEITHER safe delimiter — an
//!   > unterminated repr (no closing quote), or one embedded mid-line with
//!   > more text after the quote — is scrubbed THROUGH END OF LINE rather
//!   > than left alone. 0.7.0 pinned the old chain's two non-matches
//!   > ("an unterminated run is left alone; one terminated by a real
//!   > newline with no quote before it is left alone") as specified
//!   > behavior; both are subsumed by the fail-closed leg and both now
//!   > scrub to end of line. The consumer chain made the delimiter-miss
//!   > policy explicit: a lookahead miss must delete MORE text, never
//!   > less of the secret — the 0.7.0 shape let a repr that
//!   > `traceback`/`repr` never actually renders (or hand-built text)
//!   > ship its DETAIL payload verbatim, which is the under-redaction
//!   > direction a scrubber may not take. Both old non-matches were
//!   > unreachable from real `repr()` output; nothing that 0.7.0 scrubbed
//!   > differently survives in the new chain's output.
//!
//! * `uri_userinfo`: `scheme://user:password@host` →
//!   `scheme://user:***@host`. Scheme and username preserved verbatim,
//!   empty username handled (the `*`-quantified username class), password
//!   ending at the first `@`, and the `\b` word boundary before the scheme
//!   honored exactly (see `WORD_DEMOTE_RANGES` for the Unicode seam that
//!   makes a naive `is_alphanumeric` check under-redact).
//! * `uri_query_creds`: the URI-query anchor of the conninfo credential
//!   pass — `[?&]name=value` → `[?&]name=***` — over the shared grammar
//!   documented under `libpq_conninfo_creds` below.
//! * `libpq_conninfo_creds`: the keyword/value anchor of the SAME pass —
//!   `name=value` where the name is not the tail of a longer word (the
//!   live chain's `(?<![A-Za-z0-9_])` lookbehind), so libpq conninfo text
//!   (`host=db password='hun ter2'` — no `://`, no `?`) masks too, and
//!   `cpwd=` is not mistaken for `pwd=`. The two names select the two
//!   anchor grammars of ONE conninfo pass, exactly the live chain's single
//!   combined regex (anchor alternation `[?&]|(?<![A-Za-z0-9_])`); the
//!   default chain (both selected) is that combined leftmost-first pass,
//!   never two sequential substitutions — a value's `***` splice must not
//!   become a new anchor for a second pass. Selection is per anchor
//!   grammar, so `["uri_query_creds"]` alone masks only `?`/`&`-anchored
//!   params and `["libpq_conninfo_creds"]` alone only keyword-anchored
//!   ones.
//!
//!   The shared value grammar, both anchors: the five credential parameter
//!   names (`password`, `passphrase`, `passwd`, `pwd`, `sslpassword` —
//!   `sslpassword` is the client-TLS key's passphrase, a credential in its
//!   own right) matched CASE-INSENSITIVELY (libpq names are
//!   case-insensitive and operators' DSNs echo back whatever casing was
//!   written; the one classification seam: CPython `re.IGNORECASE` also
//!   folds exotic Unicode variants of ASCII letters — `ſ` U+017F folds to
//!   `s`, `K` U+212A to `k` — where the scanner folds ASCII only; the
//!   same accepted-seam class as `WORD_DEMOTE_RANGES`, hyp-lane policed),
//!   and the value is either a libpq single-quoted string — which may
//!   carry spaces and honors the `\'` and `\\` escapes, so the quote run
//!   must be consumed whole or the tail of the secret rides along after
//!   the `***` — or an unquoted token running to whitespace or `&`. The
//!   token deliberately does NOT stop at `@` (issue #107's tail-leak row:
//!   a password may legally contain an unencoded `@`, and the 0.7.0
//!   class stopped there, leaving the tail riding after the `***`).
//!
//! Canonical order (the chain's own application order, not a caller
//! choice): `pg_detail_lines` (real pass, then escaped pass), then
//! `uri_userinfo`, then the conninfo credential pass (both anchor
//! grammars under their two names), then `secret_tokens` last (its
//! spans ride no other rule's anchors and no established rule sees its
//! masks), each rule a whole pass over the
//! current text before the next begins. The order is a contract because
//! the rules interact: a DETAIL deletion can eat the `@` a userinfo mask
//! anchors on, and the userinfo password class claims text a param value
//! would otherwise mask separately — rule interaction is why order is
//! pinned, the same discipline `replace_many`'s order-freedom argument
//! inverts.
//!
//! > [!WARNING]
//! > The default chain can leave a credential fragment by design:
//! > `pg://u:p\nDETAIL:x@h')` scrubs to `pg://u:p')` (the DETAIL deletion
//! > eats the `@`, the userinfo mask then has nothing to anchor on). The
//! > order is the documented contract; do NOT reorder to
//! > "fix" it. Safe pattern when credential removal outranks DETAIL
//! > handling: run `uri_userinfo` separately (it gives `pg://u:***@h')`
//! > here), trading the DETAIL deletion for the mask, visibly at the call
//! > site.
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

use memchr::{memchr, memchr_iter, memmem, memrchr};

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
/// recipe (both files vendored under tools/): run step 1 against the
/// pinned rustc, then step 2 to fold and diff:
///
/// ```text
/// rustc -O tools/enum.rs -o /tmp/enum && /tmp/enum > /tmp/rust_alnum.txt
/// python3 tools/gen_word_demote_table.py /tmp/rust_alnum.txt
/// ```
///
/// Pinned inputs: rustc 1.98.1, UCD 16.0.0 (see PINNED_RUSTC /
/// PINNED_UNIDATA in the script and the version-pin test in
/// tests/test_scrub_log_text_parity.py — a toolchain or UCD jump that moves
/// the recomputed ranges is a deliberate re-sync of the table, the pins,
/// and the test together, never a silent edit.
///
/// A rustc that adopts a newer UCD can classify newly-assigned
/// Other_Alphabetic codepoints this table does not list; the crate-side
/// spot-check pins below are the
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
    /// Mask password-family query parameters (`[?&]`-anchored).
    pub const URI_QUERY_CREDS: Self = Self(4);
    /// Mask password-family conninfo keywords (libpq lookbehind-anchored;
    /// see the module docs — the two conninfo names select the two anchor
    /// grammars of ONE pass).
    pub const LIBPQ_CONNINFO_CREDS: Self = Self(8);
    /// Mask secret-token material (the `secret_impl` grammars: AWS
    /// access keys, Slack tokens, Stripe keys, GitHub tokens, PEM
    /// private-key blocks), each span spliced to `***`.
    pub const SECRET_TOKENS: Self = Self(16);
    /// `rules=None`: the full chain, in canonical order.
    pub const ALL: Self = Self(31);

    fn pg_detail_lines(self) -> bool {
        self.0 & Self::PG_DETAIL_LINES.0 != 0
    }

    fn uri_userinfo(self) -> bool {
        self.0 & Self::URI_USERINFO.0 != 0
    }

    fn uri_query_creds(self) -> bool {
        self.0 & Self::URI_QUERY_CREDS.0 != 0
    }

    fn libpq_conninfo_creds(self) -> bool {
        self.0 & Self::LIBPQ_CONNINFO_CREDS.0 != 0
    }

    fn secret_tokens(self) -> bool {
        self.0 & Self::SECRET_TOKENS.0 != 0
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

/// Drop every real-newline line whose content starts with the ExceptionGroup
/// gutter run — repetitions of (spaces/tabs, one `|` or `+`, spaces/tabs) —
/// then optional spaces/tabs and the literal `DETAIL:` — the whole line to
/// (not including) its newline, so a blank line is left behind; a CRLF line's
/// `\r` is part of the deleted content (`.` consumes it). Byte-identical to
/// `re.compile(r"^(?:[ \t]*[|+][ \t]*)*[ \t]*DETAIL:.*$", re.MULTILINE)
/// .sub("", text)`.
///
/// The gutter group (`#107`): `traceback.format_exception` indents every
/// line of an `ExceptionGroup`/`except*` sub-exception with repeated `| `
/// markers (and `+` on the group's own header/separator lines), one layer
/// per nesting level, so a DETAIL line inside a grouped exception reads
/// `"    +   | DETAIL: row-848"` and the bare `^[ \t]*` anchor never
/// reached past the marker. The walk is the group grammar directly: skip
/// whitespace, take a gutter char, repeat — a gutter char may only follow
/// whitespace, so `DETAIL: a|b`'s interior `|` (no whitespace before it,
/// inside the line's payload) can never be part of a prefix, and the
/// prefix is consumed only when the walk lands on `DETAIL:` — a header
/// line such as `  | ExceptionGroup: ...` does not start with `DETAIL:`
/// and is not touched.
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
        loop {
            while prefix < line_end && matches!(bytes[prefix], b' ' | b'\t') {
                prefix += 1;
            }
            if prefix < line_end && matches!(bytes[prefix], b'|' | b'+') {
                prefix += 1;
            } else {
                break;
            }
        }
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
/// lookahead succeeds: another escaped separator, or the repr tail (a
/// quote, then the run of `)`/`]` closers `repr()` ends with — `')` plain,
/// `')])` inside an ExceptionGroup's list — with only whitespace to the
/// end of the line), or — the FAIL-CLOSED leg (#107's security-policy
/// change, inverting 0.7.0's pinned "unterminated run is left alone") —
/// the end of the line itself. The run cannot cross a real newline (`.`
/// does not match one), so a run with no delimiter at all scrubs through
/// end of line: a delimiter miss must delete more text, never less of the
/// secret (the consumer chain's own stated policy). Byte-identical to
/// `re.compile(r"(?:\\r)?\\n[ \t]*DETAIL:.*?(?=(?:\\r)?\\n|['\"][)\]]*\s*$|$)",
/// re.MULTILINE).sub("", text)`.
///
/// The lazy run stops at the FIRST lookahead hit; it cannot cross the
/// line's real newline. `$` in MULTILINE matches before a real newline
/// and at end of text, and `\s*` inside the closing alternative cannot
/// cross a real newline either without landing on another `$` position —
/// both reduce to "everything between the closer run and the line's real
/// newline is whitespace", the form the scan checks. Alternatives are
/// tried at every position in the chain's order (escaped separator, then
/// closer run, then line end), which is the lazy-quantifier contract.
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
            // Alternative 2: a quote, then the `)`/`]` closer run a repr's
            // tail is made of (`')` plain, `')])` inside an ExceptionGroup's
            // list, one more `])` per nesting level), then whitespace only
            // up to the end of the line. The run is maximal: a shorter run
            // would put a `)`/`]` (not `\s`, not a `$` position) where the
            // alternative needs `\s*$` to start or hold, so backtracking
            // can never prefer one. When the run is there the without-run
            // path would need a closer char itself to be `\s`, so testing
            // past the run (when present) is the whole backtracking.
            if matches!(bytes.get(q), Some(b'\'') | Some(b'"')) {
                let mut after_closers = q + 1;
                while matches!(bytes.get(after_closers), Some(b')') | Some(b']')) {
                    after_closers += 1;
                }
                let ws_start =
                    *ws_memo.get_or_insert_with(|| trailing_ws_start(text, line_start, line_end));
                if after_closers >= ws_start {
                    terminator = Some(q);
                    break;
                }
            }
            // Alternative 3, the fail-closed leg: the end of the line
            // itself (`$`: before the real newline, or end of text — the
            // positions `q` reaches exactly, `.` not matching a newline
            // and `line_end` being a char boundary). A delimiter miss
            // scrubs through end of line; see the doc above.
            if q == line_end {
                terminator = Some(q);
                break;
            }
            q += text[q..].chars().next().map_or(1, char::len_utf8);
        }
        // The fail-closed leg makes the line end itself a terminator, so
        // the walk above cannot fall out empty (q advances by whole chars
        // and lands exactly on line_end); the `else` is kept loud anyway.
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
///
/// Linear in the input — the contract this pass owes the scrub API. The
/// cursor advances only on match, so K failed anchors sharing one tail
/// would each re-walk it: the password class `[^\s@]+` permits `:`/`/`/`?`
/// `=`, so `"a://u:"*K + "p"*M` (no `@` anywhere) pays O(K*M) — ~486ms
/// pre-fix at K=2000/M=200k where the chain's own `re` pays ~4s (also
/// quadratic; availability wins over matching its complexity class). The
/// fix is failure memoization: `fail_end` is the farthest tail position a
/// failed anchor has already proven holds no `@` before its whitespace/end
/// terminator, and any later anchor starting before it cannot match (its
/// own password scan would stop at the same terminator without seeing an
/// `@`, since passwords cannot cross whitespace). Anchors `< fail_end`
/// are skipped; every byte of failed-tail scan is then charged once total
/// (recompute windows tile disjointly, the escaped pass's memo discipline),
/// while matches still advance `cursor` past the `@`. Skips are exact, not
/// approximate: a skipped anchor's `@`-before-terminator set is a subset
/// of the already-proven empty set, so parity with the chain is untouched.
fn mask_uri_userinfo(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();
    let mut out: Option<String> = None;
    let mut cursor = 0usize;
    let mut fail_end = 0usize;
    for anchor in memmem::find_iter(text.as_bytes(), b"://") {
        if anchor < cursor {
            continue; // inside a previous match (a password may hold "://")
        }
        if anchor < fail_end {
            continue; // inside a previously failed tail (no @ before its end)
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
        // `/`, `@`, or whitespace; empty is a real shape. `username_end`
        // is the failure position the skip-ahead charges when no colon is
        // found (a `://` holds a `:`, so a colon-less username region holds
        // no later anchor either — the skip is vacuous there, exact).
        let mut colon = None;
        let mut username_end = text.len();
        for (idx, c) in text[anchor + 3..].char_indices() {
            let pos = anchor + 3 + idx;
            if c == ':' {
                colon = Some(pos);
                username_end = pos;
                break;
            }
            if is_python_space(c) || c == '/' || c == '@' {
                username_end = pos;
                break;
            }
        }
        let Some(colon) = colon else {
            fail_end = fail_end.max(username_end);
            continue;
        };
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
            // No `@` in (colon, end): any anchor starting before `end`
            // would hit the same terminator first (passwords cannot cross
            // whitespace), so it cannot match either — skip them. Lemma: a
            // skipped anchor's colon lies at or after this anchor's colon
            // because the username class excludes `:` and `/`, so no second
            // `://` can start inside the username before its colon.
            // debug_assert below documents the bound the skip relies on:
            // the failed tail end never precedes its colon.
            debug_assert!(end > colon);
            fail_end = fail_end.max(end);
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

/// The password-family parameter names, lowercased. Matching is
/// CASE-INSENSITIVE (libpq parameter names are case-insensitive, and
/// psql, ORMs and operator-typed DSNs echo back whatever casing was
/// written — `#107`'s IGNORECASE change; the 0.7.0 port matched exact
/// lowercase and shipped every other casing's value verbatim). Suffix
/// overlap: `password` is a trailing substring of `sslpassword` — the
/// scanner resolves a candidate `=` by the longest name first, which is
/// the live chain's leftmost-match preference (the longer name's match
/// starts earlier; the shorter suffix can never fire when the longer
/// one's anchor fails, the char between them being a word char by
/// construction). No name is a PREFIX of another and they diverge by the
/// 6th char at the latest (`pwd` diverges at the 2nd, `passphrase` at
/// the 5th, `password` vs `passwd` at the 6th), so at most one name can
/// match at any one position and the order among equals is free.
///
/// The one accepted case-folding seam: CPython `re.IGNORECASE` also folds
/// exotic Unicode variants of ASCII letters (`ſ` U+017F → `s`,
/// `K` U+212A → `k`); the scanner folds ASCII only (`eq_ignore_ascii_case`)
/// — the same accepted-seam class as `WORD_DEMOTE_RANGES`, policed by the
/// hypothesis lanes.
const PARAM_NAMES: [&str; 5] = ["sslpassword", "passphrase", "password", "passwd", "pwd"];

/// Is `c` in the lookbehind class `[A-Za-z0-9_]` — explicitly ASCII (the
/// live chain spells the class; a Unicode letter such as `é` is NOT in
/// it, so `épassword=x` masks, where the `\b`-style Unicode word check
/// the userinfo rule needs would disagree).
fn is_conninfo_word_byte(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Mask the values of the password-family connection parameters, the
/// `#107` re-sync of the 0.7.0 `uri_query_creds` rule: one pass over the
/// two anchor grammars the live chain's single combined regex
/// (`((?:[?&]|(?<![A-Za-z0-9_]))(?:password|passphrase|passwd|pwd|sslpassword)=)('(?:[^'\\]|\\.)*'|[^\s&]+)`,
/// IGNORECASE) scans with — `query_anchor` selects the URI query `[?&]`
/// anchor (the `uri_query_creds` rule), `lookbehind_anchor` the libpq
/// keyword anchor (the `libpq_conninfo_creds` rule); the default chain
/// selects both, which is the combined leftmost-first pass, never two
/// sequential substitutions (a value's `***` splice must not become a
/// new anchor for a second pass).
///
/// Name and delimiter are kept verbatim — the masked form still names
/// which setting carried the credential — and the value is replaced with
/// `***`. The value is either a libpq single-quoted string, which may
/// carry spaces and honors the `\'` and `\\` escapes (so the quote run
/// must be consumed whole or the tail of the secret rides along after
/// the `***`), or an unquoted token running to whitespace or `&` —
/// deliberately NOT stopping at `@` (`#107`'s tail-leak fix: a password
/// may legally contain an unencoded `@`, and the 0.7.0 class stopped
/// there, leaving the tail riding after the `***`). An empty value is
/// not a mask (`+`/the quoted alternatives all need at least one char).
///
/// The quoted walk is greedy without backtracking, which is exact for
/// this grammar: inside the quotes every `\` pairs with the following
/// char (`\'`, `\\`; a `\` whose follower is a real newline cannot pair —
/// `.` in the chain's `\\.` does not match a newline — and the
/// alternative fails, falling back to the unquoted token, which stops at
/// that same newline), a `'` closes, and no decomposition other than the
/// greedy one exists (a `\` can never start a shorter unit).
///
/// Anchoring is `=`-driven, not delimiter-driven: every match contains
/// exactly one `name=`, so the scan iterates `=` positions (memchr) and
/// reads the parameter name off the chars before it — this covers both
/// anchor grammars in one left-to-right pass, the live regex's own scan
/// order. A candidate fires when (a) the chars before the `=` equal one
/// of the names case-insensitively (longest first, the leftmost-start
/// preference the suffix overlap `password`/`sslpassword` needs), (b) the
/// anchor immediately before the name is selected: `?`/`&` under
/// `query_anchor`, a non-`[A-Za-z0-9_]` char (or start of text) under
/// `lookbehind_anchor`, and (c) a value follows. Matches never overlap:
/// the cursor jumps past each consumed value, and a later name's anchor
/// cannot reach back into a consumed value (a value ends only at
/// whitespace, `&`, or a closing quote — every one of which is a valid,
/// non-word anchor position for a FOLLOWING name, never a straddling
/// one). Linear in the input, the contract every pass here owes the
/// scrub API: each `=` costs O(longest name) plus its own value walk.
fn mask_conninfo_creds(text: &str, query_anchor: bool, lookbehind_anchor: bool) -> Cow<'_, str> {
    debug_assert!(query_anchor || lookbehind_anchor);
    let bytes = text.as_bytes();
    let mut out: Option<String> = None;
    let mut cursor = 0usize;
    for eq in memchr_iter(b'=', bytes) {
        if eq < cursor {
            continue; // inside a previous value (values may hold =)
        }
        // (a) The parameter name ending at the `=`, longest first (the
        // `eq >= name.len()` guard: the text can open with `pwd=` before
        // any longer name could fit).
        let mut name_start = None;
        for name in PARAM_NAMES {
            if eq >= name.len() {
                let start = eq - name.len();
                if start >= cursor && bytes[start..eq].eq_ignore_ascii_case(name.as_bytes()) {
                    name_start = Some(start);
                    break;
                }
            }
        }
        let Some(name_start) = name_start else {
            continue;
        };
        // (b) The anchor immediately before the name, per selection.
        // `eq > name_start >= cursor` bounds the char decode: the anchor
        // position is at or after the last committed cursor, so the
        // backward char walk cannot cross into a consumed value.
        let anchored = if name_start == 0 {
            lookbehind_anchor // no preceding char: the lookbehind succeeds
        } else {
            let mut prev = name_start - 1;
            while !text.is_char_boundary(prev) {
                prev -= 1;
            }
            match text[prev..].chars().next() {
                Some(c @ ('?' | '&')) => {
                    query_anchor || (lookbehind_anchor && !is_conninfo_word_byte(c))
                }
                Some(c) => lookbehind_anchor && !is_conninfo_word_byte(c),
                None => false,
            }
        };
        if !anchored {
            continue;
        }
        // (c) The value: a libpq single-quoted run (escapes honored) or
        // an unquoted token to whitespace/`&`.
        let mut end = eq + 1;
        let quoted_end = if bytes.get(end) == Some(&b'\'') {
            let mut i = end + 1;
            loop {
                match bytes.get(i) {
                    Some(b'\'') => break Some(i + 1),
                    Some(b'\\') if i + 1 < bytes.len() && bytes[i + 1] != b'\n' => i += 2,
                    // Unpairable escape (`\` at end or before a real
                    // newline, the chain's `\\.` needing a non-newline
                    // follower) or the closing quote never comes (end of
                    // text): the quoted leg fails and the unquoted-token
                    // leg takes the same start.
                    Some(b'\\') | None => break None,
                    Some(_) => i += 1,
                }
            }
        } else {
            None
        };
        match quoted_end {
            Some(vend) => end = vend,
            None => {
                // Unquoted token (also the quoted leg's fallback): one or
                // more chars that are neither Python-whitespace nor `&`.
                end = eq + 1;
                while let Some(c) = text[end..].chars().next() {
                    if is_python_space(c) || c == '&' {
                        break;
                    }
                    end += c.len_utf8();
                }
                if end == eq + 1 {
                    continue; // empty value: not a mask
                }
            }
        }
        let out = out.get_or_insert_with(|| String::with_capacity(text.len()));
        out.push_str(&text[cursor..name_start]);
        out.push_str(&text[name_start..=eq]);
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
/// the userinfo mask, then the conninfo credential pass (both anchor
/// grammars under their two names — one pass, the combined leftmost-first
/// scan the live chain's single regex performs), then the secret-token
/// mask (the `secret_impl` grammars, spans spliced to `***`, LAST so the
/// established chain's contract is untouched), each pass over the
/// current text, no pass rescanning another's output. A pass that fires
/// moves the chain onto its owned output; a pass that does not fire
/// returns the borrow, and the chain stays where it was. The result is
/// borrowed — the identity lane — exactly when no rule fired.
pub fn scrub_log_text(text: &str, rules: RuleSet) -> Cow<'_, str> {
    type Pass = fn(&str) -> Cow<'_, str>;
    let passes: [Option<Pass>; 3] = [
        rules.pg_detail_lines().then_some(drop_detail_lines as Pass),
        rules.pg_detail_lines().then_some(drop_detail_escaped),
        rules.uri_userinfo().then_some(mask_uri_userinfo),
    ];
    let mut owned: Option<String> = None;
    for pass in passes.into_iter().flatten() {
        let src: &str = owned.as_deref().unwrap_or(text);
        if let Cow::Owned(out) = pass(src) {
            owned = Some(out);
        }
    }
    if rules.uri_query_creds() || rules.libpq_conninfo_creds() {
        let src: &str = owned.as_deref().unwrap_or(text);
        let out = mask_conninfo_creds(src, rules.uri_query_creds(), rules.libpq_conninfo_creds());
        if let Cow::Owned(out) = out {
            owned = Some(out);
        }
    }
    if rules.secret_tokens() {
        let src: &str = owned.as_deref().unwrap_or(text);
        if let Cow::Owned(out) = crate::secret_impl::mask_secret_tokens(src) {
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
    fn space_table_exhaustive_definition_holds() {
        // Definition self-check over 0..0x110000, not UCD coverage: it is
        // tautological against the definition by construction — its job is
        // only to fail if a future edit touches the definition without
        // updating the seam docs. UCD coverage lives in the split-out pins:
        // the Python-side exhaustive re-vs-isspace pin and the
        // file-separator spot checks (both directions of the seam).
        for cp in 0..=0x10FFFFu32 {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            assert_eq!(
                is_python_space(c),
                c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}'),
                "U+{cp:04X}"
            );
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
        // The fail-closed leg (#107's policy change, inverting 0.7.0's
        // pinned "unterminated run is left alone"): a delimiter miss
        // scrubs THROUGH END OF LINE — both of the two non-matches 0.7.0
        // pinned (no closing quote; real-newline termination without a
        // quote) are subsumed by the bare `$` lookahead leg.
        assert_eq!(scrub("E('a\\nDETAIL: leaks", RuleSet::ALL), "E('a");
        assert_eq!(
            scrub("E('a\\nDETAIL: leaks\nnext", RuleSet::ALL),
            "E('a\nnext"
        );
        // A quote not at end of line is not a terminator either — the run
        // continues past it and the fail-closed leg takes the line end.
        assert_eq!(scrub("E('a\\nDETAIL: v')  tail", RuleSet::ALL), "E('a");
        // The `)`/`]` closer RUN a nested repr ends with (`')` plain,
        // `')])` inside an ExceptionGroup's list, one more `])` per
        // nesting level); a `]` before the quote is payload, consumed.
        assert_eq!(scrub("E('m\\nDETAIL: v')])", RuleSet::ALL), "E('m')])");
        assert_eq!(scrub("E('m\\nDETAIL: v]')", RuleSet::ALL), "E('m')");
        assert_eq!(scrub("E('m\\nDETAIL: v)]')", RuleSet::ALL), "E('m')");
        assert_eq!(scrub("E('m\\nDETAIL: v')])')", RuleSet::ALL), "E('m')");
        // Trailing gutter text inside the payload does not terminate it,
        // and with no quote-closer at EOL the fail-closed leg takes the
        // line end.
        assert_eq!(scrub("E('m\\nDETAIL: v|x", RuleSet::ALL), "E('m");
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
    fn detail_lines_absorb_the_exception_group_gutters() {
        // #107: the `(?:[ \t]*[|+][ \t]*)*` gutter group — one `| `/`+ `
        // layer per ExceptionGroup nesting level.
        assert_eq!(scrub("    +   | DETAIL: row-848", RuleSet::ALL), "");
        assert_eq!(scrub("  | DETAIL: v\nnext", RuleSet::ALL), "\nnext");
        assert_eq!(scrub("+\t+ DETAIL: v", RuleSet::ALL), "");
        assert_eq!(scrub("||DETAIL: v", RuleSet::ALL), "");
        assert_eq!(scrub("\t | \t + DETAIL: v", RuleSet::ALL), "");
        // A header line that is not a DETAIL line is untouched; an interior
        // gutter with no whitespace before it is not a prefix.
        assert_eq!(
            scrub("  | ExceptionGroup: x\n", RuleSet::ALL),
            "  | ExceptionGroup: x\n"
        );
        assert_eq!(scrub("DETAIL: a|b", RuleSet::ALL), "");
        assert_eq!(scrub("x | DETAIL: v", RuleSet::ALL), "x | DETAIL: v");
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
        // Uppercase/IP/port shape: host grammar is untouched by the mask.
        assert_eq!(
            scrub("http://u:p@192.168.1.1:8080/x", RuleSet::ALL),
            "http://u:***@192.168.1.1:8080/x"
        );
    }

    #[test]
    fn query_creds_pinned_edges() {
        assert_eq!(
            scrub("?password=x&passphrase=y&passwd=z&pwd=w", RuleSet::ALL),
            "?password=***&passphrase=***&passwd=***&pwd=***"
        );
        // #107: the value class no longer stops at `@` (the tail-leak fix:
        // a password may legally contain an unencoded `@`), names are
        // CASE-INSENSITIVE, and `sslpassword` joined the name set.
        assert_eq!(
            scrub("?password=a b?pwd=c@d&passwd=e", RuleSet::ALL),
            "?password=*** b?pwd=***&passwd=***"
        );
        assert_eq!(scrub("?password=a@b", RuleSet::ALL), "?password=***");
        assert_eq!(
            scrub("?password=a@b@c&x=1", RuleSet::ALL),
            "?password=***&x=1"
        );
        assert_eq!(
            scrub("?Password=x&PASSWORD=y&passwords=z", RuleSet::ALL),
            "?Password=***&PASSWORD=***&passwords=z"
        );
        assert_eq!(
            scrub("postgresql://h/db?sslpassword=p", RuleSet::ALL),
            "postgresql://h/db?sslpassword=***"
        );
        assert_eq!(
            scrub("?SSLPassword=s&key=k", RuleSet::ALL),
            "?SSLPassword=***&key=k"
        );
        assert_eq!(scrub("?password=a=b?c", RuleSet::ALL), "?password=***");
        assert_eq!(scrub("a?password=1?pwd=2", RuleSet::ALL), "a?password=***");
        assert_eq!(
            scrub("?password=a\x1cb", RuleSet::ALL),
            "?password=***\x1cb"
        );
        // The libpq conninfo keyword anchor (the fourth named rule's
        // grammar): no `?`/`&` needed, a non-`[A-Za-z0-9_]` char (or text
        // start) before the name, single-quoted values carrying spaces,
        // `cpwd=` not mistaken for `pwd=` (a longer-word tail is not an
        // anchor).
        assert_eq!(
            scrub("host=h password=p", RuleSet::ALL),
            "host=h password=***"
        );
        assert_eq!(
            scrub("host=db PASSWORD='hun ter2'", RuleSet::ALL),
            "host=db PASSWORD=***"
        );
        assert_eq!(
            scrub("host='db host' password='p w' user=u", RuleSet::ALL),
            "host='db host' password=*** user=u"
        );
        assert_eq!(scrub("cpwd=x pwd=y", RuleSet::ALL), "cpwd=x pwd=***");
        assert_eq!(scrub("apassword=x", RuleSet::ALL), "apassword=x");
        assert_eq!(scrub("_password=x", RuleSet::ALL), "_password=x");
        // The lookbehind class is explicitly ASCII: a Unicode letter before
        // the name does NOT block the anchor.
        assert_eq!(scrub("épassword=x", RuleSet::ALL), "épassword=***");
        // Quoted values: escaped quotes and backslashes consumed whole, an
        // unterminated quote falls back to the unquoted token (which keeps
        // the leading quote in the mask), a real newline inside the quotes
        // rides along.
        assert_eq!(
            scrub("?password='a b'&x=1", RuleSet::ALL),
            "?password=***&x=1"
        );
        assert_eq!(
            scrub("?password='a\\'b'&x=1", RuleSet::ALL),
            "?password=***&x=1"
        );
        assert_eq!(
            scrub("?password='a\\\\'&x=1", RuleSet::ALL),
            "?password=***&x=1"
        );
        assert_eq!(
            scrub("?password='unterminated", RuleSet::ALL),
            "?password=***"
        );
        assert_eq!(
            scrub("?password='unterminated &password=x", RuleSet::ALL),
            "?password=*** &password=***"
        );
        assert_eq!(
            scrub("?password='multi\nline real-nl'&x=1", RuleSet::ALL),
            "?password=***&x=1"
        );
        assert_eq!(
            scrub("password='a\\'\\'' x", RuleSet::ALL),
            "password=*** x"
        );
        // Anchor-grammar selection: the keyword rule alone reaches
        // lookbehind-anchored names — and a `?` before a name is itself a
        // non-word char, so the lookbehind grammar masks that shape too;
        // the `[?&]` rule alone masks only its anchor grammar.
        assert_eq!(
            scrub(
                "host=h password=p ?password=q",
                RuleSet::LIBPQ_CONNINFO_CREDS
            ),
            "host=h password=*** ?password=***"
        );
        assert_eq!(
            scrub("host=h password=p ?password=q", RuleSet::URI_QUERY_CREDS),
            "host=h password=p ?password=***"
        );
        for text in ["?password=&x=1", "?pwd="] {
            assert_eq!(scrub(text, RuleSet::ALL), text, "{text:?}");
        }
    }

    #[test]
    fn param_names_are_prefix_free() {
        for (i, a) in PARAM_NAMES.iter().enumerate() {
            for (j, b) in PARAM_NAMES.iter().enumerate() {
                if i != j {
                    assert!(!b.starts_with(a), "{a:?} is a prefix of {b:?}");
                }
            }
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
            RuleSet::LIBPQ_CONNINFO_CREDS,
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
    fn a_param_mask_can_unblock_a_userinfo_match_on_pass_two() {
        // The one non-idempotence class, cross-rule (CI fuzz-found,
        // crash-a2d92f3d): pass 1's param value eats the `/` capping the
        // userinfo user run, so pass 2's user class spans the `***` and
        // the `&` and the userinfo rule fires; pass 3 re-matches the
        // already-`***` password to itself — the fixed point.
        let text = "x://u?pwd=a/b&:pw@h";
        let once = scrub(text, RuleSet::ALL);
        assert_eq!(once, "x://u?pwd=***&:pw@h");
        let twice = scrub(&once, RuleSet::ALL);
        assert_eq!(twice, "x://u?pwd=***&:***@h");
        assert_eq!(scrub(&twice, RuleSet::ALL), twice);
    }

    #[test]
    fn empty_input_is_identity() {
        assert!(matches!(scrub_log_text("", RuleSet::ALL), Cow::Borrowed(_)));
    }
}
