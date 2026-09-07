//! `html_unescape` never panics on arbitrary strings, including deeply
//! malformed/adversarial numeric character references (huge digit runs,
//! nested-looking entity prefixes, boundary codepoints).

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|s: &str| {
    let unescaped = tors::html_impl::unescape(s);
    // Output must always be a real `str` (trivially true for `Cow<str>`,
    // asserted explicitly as the harness's own invariant check).
    let _ = unescaped.len();

    // The int-max-str-digits-checked path must also never panic, for any
    // limit including the degenerate `None` (no limit).
    let _ = tors::html_impl::unescape_checked(s, Some(4300));
    let _ = tors::html_impl::unescape_checked(s, None);

    // The percent-encoding pair round-trips over any valid str with the
    // empty safe set: quote percent-encodes every non-always-safe byte, so
    // unquote's decode recovers exactly the original bytes.
    let quoted = tors::url_impl::quote(s, "");
    let unquoted = tors::url_impl::unquote(quoted.as_ref());
    assert_eq!(unquoted.as_ref(), s, "unquote(quote(s)) diverged on {s:?}");
});
