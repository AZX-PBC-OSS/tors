//! `scrub_log_text` never panics, is value-idempotent on the full chain and
//! on every single-rule lane, the borrowed (identity) lane never lies, and
//! the redaction-completeness invariant holds: a payload legal for the
//! password/value classes, planted in the userinfo, query-param, and
//! DETAIL-line shapes, never survives the scrub — asserted as the exact
//! masked template shape, the Rust twin of the battery's hypothesis
//! invariants, over raw fuzzer strings.

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
});
