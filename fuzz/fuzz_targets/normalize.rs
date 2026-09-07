//! `normalize`/`finalize` never panic on arbitrary strings, output is
//! always valid UTF-8, normalize is idempotent, finalize's digest is the
//! sha256 of normalize's output (checked against an independent hasher),
//! and the zero-cost identity path is never a false
//! positive: whenever `normalize_cow` borrows the input unchanged, the
//! borrowed value really must equal a hypothetical full-scan result. Since
//! a full independent oracle isn't available in this harness, the checked
//! identity invariant is the weaker but still real one: `normalize_cow(s) == s`
//! exactly when it borrows: a false-positive identity return would violate
//! this immediately, since `Cow::Borrowed(x) == s` is definitionally true,
//! but pairing it with a second independent call to `normalize` (the
//! always-scanning owned-output spelling) catches any divergence between
//! the fast path's decision and the real scan.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|s: &str| {
    let cow = tors::normalize_impl::normalize_cow(s);
    let scanned = tors::normalize_impl::normalize(s);

    // The identity path and the always-scanning path must never disagree
    // on the resulting VALUE, regardless of which one borrowed.
    assert_eq!(
        cow.as_ref(),
        scanned.as_str(),
        "normalize_cow and normalize disagree on {s:?}"
    );
    if matches!(cow, std::borrow::Cow::Borrowed(_)) {
        // A borrowed result must be byte-identical to the input: the
        // strongest, most direct check against a false-positive identity
        // return (the class of bug that would be silent data corruption).
        assert_eq!(cow.as_ref(), s, "identity path borrowed a changed value");
    }

    // Idempotence: a second pass over the output changes nothing.
    assert_eq!(
        tors::normalize_impl::normalize(&scanned),
        scanned,
        "normalize is not idempotent on {s:?}"
    );

    let (finalized, digest) = tors::finalize_impl::finalize(s);
    assert_eq!(
        finalized, scanned,
        "finalize's text half disagrees with normalize"
    );
    // The digest is the sha256 of normalize's output, the documented
    // contract; checked against an independent hasher here.
    use sha2::Digest as _;
    let expected = const_hex::encode(sha2::Sha256::digest(scanned.as_bytes()));
    assert_eq!(
        digest, expected,
        "finalize digest is not sha256(normalize output) on {s:?}"
    );
    assert_eq!(digest.len(), 64, "sha256 hex digest must be 64 chars");
    assert!(
        digest
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    );
});
