//! `scrub_log_text` never panics, is value-idempotent on the full chain and
//! on every single-rule lane, the borrowed (identity) lane never lies, and
//! the redaction-completeness invariant holds: a payload legal for the
//! password/value classes, planted in the userinfo, query-param, and
//! DETAIL-line shapes, never survives the scrub — asserted as the exact
//! masked template shape, the Rust twin of the battery's hypothesis
//! invariants, over raw fuzzer strings. The DETAIL shape has both
//! spellings of the segmenter: the real-newline line template and the
//! repr-flattened escaped-run template (`\nDETAIL:...`), the two passes
//! the one rule name routes to.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tors::scrub_impl::{RuleSet, scrub_log_text};

/// A char run the chain's password/value classes accept in full: no
/// Python-`\s` char, no `&`, no `@`. The `\s` spelling must match the
/// chain's, not Rust std's: `is_whitespace` misses the U+001C..U+001F
/// file separators CPython's `re` accepts (the seam `scrub_impl`'s
/// `is_python_space` closes in production code; spelled inline here
/// because the fuzz target answers its own question, not the code's).
fn legal_payload(s: &str) -> String {
    s.chars()
        .filter(|c| {
            !(c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}') || matches!(c, '&' | '@'))
        })
        .collect()
}

fuzz_target!(|s: &str| {
    // Idempotence, on the full chain and on every single-rule lane.
    let once = scrub_log_text(s, RuleSet::ALL);
    assert_eq!(
        scrub_log_text(once.as_ref(), RuleSet::ALL).as_ref(),
        once.as_ref()
    );
    for rule in [
        RuleSet::PG_DETAIL_LINES,
        RuleSet::URI_USERINFO,
        RuleSet::URI_QUERY_CREDS,
    ] {
        let once = scrub_log_text(s, rule);
        assert_eq!(scrub_log_text(once.as_ref(), rule).as_ref(), once.as_ref());
    }

    // The identity lane never lies: a borrowed return is byte-identical.
    if let std::borrow::Cow::Borrowed(clean) = scrub_log_text(s, RuleSet::ALL) {
        assert_eq!(clean, s, "identity path borrowed a changed value");
    }

    // Redaction completeness, the exact masked shapes. The userinfo
    // template's outer match always fires (fixed scheme/username/
    // separator, the `@` right after the payload) and no earlier pass can
    // break it; the param template carries no `@` at all, so the userinfo
    // pass can never fire inside it; the DETAIL template's whole line
    // goes, leaving exactly the surrounding lines and the blank line.
    let payload = legal_payload(s);
    let userinfo_in = format!("a://u:{payload}@h");
    assert_eq!(
        scrub_log_text(&userinfo_in, RuleSet::ALL).as_ref(),
        if payload.is_empty() {
            "a://u:@h" // an empty password is not a mask
        } else {
            "a://u:***@h"
        }
    );
    let param_in = format!("?password={payload}&x=1");
    assert_eq!(
        scrub_log_text(&param_in, RuleSet::ALL).as_ref(),
        if payload.is_empty() {
            "?password=&x=1" // an empty value is not a mask
        } else {
            "?password=***&x=1"
        }
    );
    let detail_payload = s.replace('\n', "");
    let detail_in = format!("x\nDETAIL:{detail_payload}\ny");
    assert_eq!(scrub_log_text(&detail_in, RuleSet::ALL).as_ref(), "x\n\ny");

    // The DETAIL template's escaped-segmenter twin: the repr()-flattened
    // run. The payload may not hold a Python-`\s` char (a real newline
    // ends the run's line, and other whitespace could hand an interior
    // quote the `\s*$` lookahead tail) or a backslash (an interior escaped
    // separator would end the run early and let the rest of the payload
    // survive); quotes and `)` stay legal — inside the run the template's
    // own closing `')` is the only quote whose tail is whitespace-to-EOL,
    // so the run always dies whole and the closing `')` (the repr shape
    // the lookahead's `\s*$` alternative exists to preserve) is what is
    // left. An empty payload is the same shape: the run is still the whole
    // `\nDETAIL:` stretch up to the quote.
    let escaped_payload: String = s
        .chars()
        .filter(|c| !(c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}') || *c == '\\'))
        .collect();
    let escaped_in = format!("E('x\\nDETAIL:{escaped_payload}')");
    assert_eq!(scrub_log_text(&escaped_in, RuleSet::ALL).as_ref(), "E('x')");

    // Coverage pins (fixed shapes, input-independent): the red-team battery.
    // \b demote-mark: an Other_Alphabetic mark before the scheme IS a
    // boundary, so the mask fires (a naive is_alphanumeric check would skip).
    assert_eq!(
        scrub_log_text("\u{093e}https://u:pw@h", RuleSet::ALL).as_ref(),
        "\u{093e}https://u:***@h"
    );
    // \x1c..\x1f are Python-\s: they stop the username/password/value
    // classes exactly where the chain stops.
    assert_eq!(
        scrub_log_text("a://u\x1cv:pw@h", RuleSet::ALL).as_ref(),
        "a://u\x1cv:pw@h"
    );
    assert_eq!(
        scrub_log_text("?password=a\x1cb", RuleSet::ALL).as_ref(),
        "?password=***\x1cb"
    );
    // \r\n escaped DETAIL run twin.
    assert_eq!(
        scrub_log_text("E('x\\r\\nDETAIL:secret')", RuleSet::ALL).as_ref(),
        "E('x')"
    );
    // DETAIL-eats-@: canonical order pin — the escaped deletion eats the
    // `@`, the userinfo mask then has nothing to anchor on (kept for
    // parity, NOT reordered).
    assert_eq!(
        scrub_log_text("pg://u:p\\nDETAIL:x@h')", RuleSet::ALL).as_ref(),
        "pg://u:p')"
    );
    assert_eq!(
        scrub_log_text("pg://u:p\\nDETAIL:x@h')", RuleSet::URI_USERINFO).as_ref(),
        "pg://u:***@h')"
    );
    // Empty user is a real shape (masked); empty password is not a mask.
    assert_eq!(
        scrub_log_text("postgresql://:SECRET@host/db", RuleSet::ALL).as_ref(),
        "postgresql://:***@host/db"
    );
    assert_eq!(
        scrub_log_text("postgresql://user:@host/db", RuleSet::ALL).as_ref(),
        "postgresql://user:@host/db"
    );
    assert_eq!(
        scrub_log_text("?password=&x=1", RuleSet::ALL).as_ref(),
        "?password=&x=1"
    );
    // Uppercase scheme alphabet.
    assert_eq!(
        scrub_log_text("HTTP://U:PW@H", RuleSet::ALL).as_ref(),
        "HTTP://U:***@H"
    );
    // Uppercase scheme carrying the fuzzer payload.
    let upper_in = format!("HTTP://U:{payload}@H");
    assert_eq!(
        scrub_log_text(&upper_in, RuleSet::ALL).as_ref(),
        if payload.is_empty() {
            "HTTP://U:@H"
        } else {
            "HTTP://U:***@H"
        }
    );
});
