//! `detect_encoding` never panics on arbitrary bytes and never disagrees
//! with the crate's own UTF-8 soundness relationship: a `utf-8` verdict on
//! input that `decode_strict` rejects is the one impossible answer (the
//! detector claims the bytes ARE utf-8; the decoder is the authority on
//! that same question). The verdict space is closed (the detector returns
//! `&'static str` — the type system bounds it); the harness pins the
//! relationship and the no-panic contract over raw fuzzer bytes, with the
//! `tld` hint swept across a small fixed pool (the hint changes the
//! tie-breaks, never the soundness).

#![no_main]

use libfuzzer_sys::fuzz_target;

/// The fixed tld pool: the real callers' shapes (a hostname tail, the dot
/// spelling, mixed case, empty) — enough to exercise the hint's
/// normalization branches without a second arbitrary input.
const TLDS: [Option<&str>; 5] = [None, Some("com"), Some(".com"), Some("ORG"), Some("")];

fuzz_target!(|data: &[u8]| {
    let utf8_is_valid = tors::utf8_impl::is_valid(data);
    for tld in TLDS {
        let verdict = tors::encoding_impl::detect(data, tld);
        if verdict == "utf-8" {
            assert!(
                utf8_is_valid,
                "detect_encoding claimed utf-8 on invalid UTF-8: tld={tld:?} data={data:?}"
            );
        }
    }
    // The tld hint changes tie-breaks, never the no-panic contract: one
    // extra sweep with a hint derived FROM the input (arbitrary strings
    // reach the normalization branches, not just the pool's shapes).
    if let Ok(hint) = std::str::from_utf8(data) {
        let tld = hint.get(..hint.len().min(16)).unwrap_or(hint);
        let _ = tors::encoding_impl::detect(data, Some(tld));
    }
});
