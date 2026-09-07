//! `b64_decode` never panics on arbitrary strings, not just valid base64
//! alphabet — including strings containing lone surrogates encoded as
//! Rust `&str` cannot hold, so this is bounded to whatever `arbitrary`
//! produces as a valid `String` (still exercises non-alphabet characters,
//! malformed padding, and adversarial whitespace).

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    s: String,
    strict: bool,
}

fuzz_target!(|input: Input| {
    let _ = tors::b64_impl::decode(&input.s, input.strict);
});
