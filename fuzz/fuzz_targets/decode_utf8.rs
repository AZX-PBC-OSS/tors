//! Never panics on arbitrary bytes, and `utf8_is_valid` never disagrees
//! with `decode_strict`'s own success/failure (the soundness relationship
//! the crate documents between the two functions).

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let strict = tors::decode_impl::decode_strict(data);
    let valid = tors::utf8_impl::is_valid(data);
    assert_eq!(
        strict.is_ok(),
        valid,
        "utf8_is_valid disagreed with decode_strict on {data:?}"
    );
    if let Ok(s) = &strict {
        // A successful decode must itself be valid UTF-8 (trivially true
        // for a `Cow<str>`, but assert it explicitly as the harness's own
        // sanity check rather than trusting the type system silently).
        assert!(std::str::from_utf8(s.as_bytes()).is_ok());
    }

    // errors="replace" must never panic and must always produce valid UTF-8.
    let replaced = tors::decode_impl::decode_replace(data);
    assert!(std::str::from_utf8(replaced.as_bytes()).is_ok());
});
