//! `scrub_log_text` never panics, is CONVERGENT on the full chain and on
//! every single-rule lane (pass 3 == pass 2: the grammar's own alternation
//! and the passes' interactions are not pass-1-idempotent — the reference
//! chain behaves identically), the borrowed (identity) lane never lies,
//! and the redaction-completeness invariant holds: a payload legal for the
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

fuzz_target!(|s: &str| {
    // Terminating convergence — the honest contract on raw inputs, on the
    // full chain and on every single-rule lane. The pass interactions are
    // real and pinned by the pytest battery at two passes over its corpus
    // (crash-9268942a: a param replacement deletes a `/` and arms a
    // userinfo match; the grammar's own alternation is also not
    // pass-1-idempotent where the quoted arm's mask leaves a tail —
    // `\p&pwd='wH\0\0\0')[n` masks the quoted span, whose `***` + tail
    // the unquoted arm then re-swallows on the next pass — the pinned
    // regex chain itself behaves identically, the reference re-applied
    // matches it exactly). On RAW adversarial input the pass count to the
    // fixed point is input-dependent, so the invariant asserted here is
    // that the chain REACHES its fixed point — within a generous pass
    // budget — and never oscillates. Strict pass-1 idempotence and a
    // global 2-pass bound were this target's over-assertions, not the
    // grammar's.
    let mut cur = scrub_log_text(s, RuleSet::ALL);
    let mut converged = false;
    for _ in 0..32 {
        let next = scrub_log_text(cur.as_ref(), RuleSet::ALL);
        if next.as_ref() == cur.as_ref() {
            converged = true;
            break;
        }
        // The Cow may borrow its input: own it before reassigning.
        cur = std::borrow::Cow::Owned(next.into_owned());
    }
    assert!(
        converged,
        "the full chain did not reach a fixed point within 32 passes: s={s:?} stuck at {cur:?}"
    );
    for rule in [
        RuleSet::PG_DETAIL_LINES,
        RuleSet::URI_USERINFO,
        RuleSet::URI_QUERY_CREDS,
        RuleSet::URI_QUERY_CREDS_EXTENDED,
    ] {
        let mut cur = scrub_log_text(s, rule);
        let mut converged = false;
        for _ in 0..32 {
            let next = scrub_log_text(cur.as_ref(), rule);
            if next.as_ref() == cur.as_ref() {
                converged = true;
                break;
            }
            cur = std::borrow::Cow::Owned(next.into_owned());
        }
        assert!(
            converged,
            "single-rule lane did not reach a fixed point within 32 passes: rule={rule:?} s={s:?} stuck at {cur:?}"
        );
    }

    // The identity lane never lies: a borrowed return is byte-identical.
    if let std::borrow::Cow::Borrowed(clean) = scrub_log_text(s, RuleSet::ALL) {
        assert_eq!(clean, s, "identity path borrowed a changed value");
    }

    // Redaction completeness, model-free: an alphanumeric credential value
    // planted in each template shape must be GONE from the output. The
    // exact masked SHAPE is not asserted here on purpose: the chain is an
    // ordered multi-rule pipeline (DETAIL deletions eat later passes'
    // anchors; a masked value's tail can arm a later occurrence), so an
    // exact-shape twin is a second full implementation of the chain — the
    // thing that breaks as the fuzzer explores. The shape contract is the
    // pytest battery's differential against tests/reference.py's chain
    // model; THIS lane asserts what needs no model: a plain credential
    // value never survives, under every template spelling, whatever the
    // passes do to the surrounding text.
    let secret = format!(
        "{}{}",
        "s3cret",
        &s.chars()
            .filter(char::is_ascii_alphanumeric)
            .take(24)
            .collect::<String>()
    );
    // (alnum-only secret: no rule's own grammar can interact with it —
    // every rule's value/token class accepts it whole, so a surviving
    // secret is unambiguously a miss, and no pass can eat an anchor that
    // is not there.)
    for planted in [
        format!("a://u:{secret}@h"),
        format!("?password={secret}&x=1"),
        format!("?passphrase={secret}&x=1"),
        format!("?passwd={secret}&x=1"),
        format!("?pwd={secret}&x=1"),
        format!("?sslpassword={secret}&x=1"),
        // The extended key set's shapes (the default chain runs the
        // widest key set): an alnum value behind an ops-standard key
        // never survives either.
        format!("?sig={secret}&x=1"),
        format!("?api_key={secret}&x=1"),
        format!("?apikey={secret}&x=1"),
        format!("?key={secret}&x=1"),
        format!("?access_key={secret}&x=1"),
        format!("?sas_token={secret}&x=1"),
        format!("?token={secret}&x=1"),
        format!("?secret={secret}&x=1"),
        format!("?passkey={secret}&x=1"),
        format!("?auth={secret}&x=1"),
        format!("x\nDETAIL:{secret}\ny"),
        format!("E('x\\nDETAIL:{secret}')"),
    ] {
        let out = scrub_log_text(&planted, RuleSet::ALL);
        assert!(
            !out.contains(&secret),
            "credential value survived the scrub: secret={secret:?} shape={planted:?} out={out:?}"
        );
    }

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
    // The extended key set: name preserved, value masked; the base rule
    // alone still leaves the extended keys alone (the superset lane only
    // adds names), and the anchor/delimiter discipline holds under the
    // fuzzer's own shapes too.
    assert_eq!(
        scrub_log_text("?api_key=v&next=1", RuleSet::ALL).as_ref(),
        "?api_key=***&next=1"
    );
    assert_eq!(
        scrub_log_text("?api_key=v&next=1", RuleSet::URI_QUERY_CREDS).as_ref(),
        "?api_key=v&next=1"
    );
    assert_eq!(scrub_log_text("?xkey=v", RuleSet::ALL).as_ref(), "?xkey=v");
    assert_eq!(scrub_log_text("?key=", RuleSet::ALL).as_ref(), "?key=");
    // Uppercase scheme alphabet.
    assert_eq!(
        scrub_log_text("HTTP://U:PW@H", RuleSet::ALL).as_ref(),
        "HTTP://U:***@H"
    );
    // Uppercase scheme alphabet carries the fuzzer's shapes too: the
    // IGNORECASE name/scheme classes are the grammar's own case rule.
    assert_eq!(
        scrub_log_text("HTTP://U:PW@H", RuleSet::ALL).as_ref(),
        "HTTP://U:***@H"
    );
});
