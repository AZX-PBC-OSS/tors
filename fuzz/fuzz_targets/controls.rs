//! `strip_controls` never panics on an arbitrary string, emits no C0/DEL
//! character, is idempotent, and the identity path never lies: a borrowed
//! return is byte-identical to the input.

#![no_main]

use libfuzzer_sys::fuzz_target;

fn has_scrubbed(s: &str) -> bool {
    s.chars().any(|c| matches!(c, '\0'..='\x1f' | '\x7f'))
}

fuzz_target!(|s: &str| {
    let got = tors::controls_impl::strip_controls(s);
    let got = got.as_ref();

    // The scrub contract: no C0/DEL character survives.
    assert!(
        !has_scrubbed(got),
        "scrubbed output still holds a control on {s:?}: {got:?}"
    );
    if matches!(
        tors::controls_impl::strip_controls(s),
        std::borrow::Cow::Borrowed(_)
    ) {
        assert_eq!(got, s, "identity path borrowed a changed value");
    } else {
        assert!(has_scrubbed(s), "allocated on clean input {s:?}");
    }

    // Idempotence: the output is already clean, so a second pass borrows.
    assert!(
        matches!(
            tors::controls_impl::strip_controls(got),
            std::borrow::Cow::Borrowed(_)
        ),
        "strip_controls is not idempotent on {s:?}"
    );
});
