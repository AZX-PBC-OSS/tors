//! `hash_impl`'s five one-shot cores never panic on arbitrary bytes,
//! always emit the exact digest-length output in each spelling (lowercase
//! hex of the exact length, raw digest bytes of the exact length), and
//! never drift off the pinned known-answer constants: the empty-input
//! digests (the canary a wiring regression would move first, since the
//! empty input is in every fuzzer's reachable space, pinned in BOTH
//! spellings) and the RFC 4231 case-2 HMAC vector (the one `str`-shaped
//! vector, key "Jefe", data "what do ya want for nothing?", pinned in
//! both spellings). Plus the key/data cross-invariant spelling:
//! hmac with key and data fed the SAME arbitrary bytes must hold the
//! same shape contract (crash-freedom + 64 lowercase hex chars / 32
//! raw bytes).
//!
//! Stateful one-shot only: there is no incremental-update surface to
//! cross-check against a fresh instance (the stateless-by-charter
//! scope cut, docs/design.md), so the invariants are the shape and
//! known-answer ones above.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    data: Vec<u8>,
    key: Vec<u8>,
}

fuzz_target!(|input: Input| {
    let data = input.data.as_slice();
    let key = input.key.as_slice();

    // Crash-freedom + the output-shape contract: exact length, lowercase
    // hex charset, for every core over every input.
    for (digest, len) in [
        (tors::hash_impl::md5_hex(data), 32),
        (tors::hash_impl::sha1_hex(data), 40),
        (tors::hash_impl::sha256_hex(data), 64),
        (tors::hash_impl::sha512_hex(data), 128),
    ] {
        assert_eq!(digest.len(), len);
        assert!(digest.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(digest, digest.to_ascii_lowercase());
    }

    // The raw-digest mirror: crash-freedom + exact byte lengths
    // (16/20/32/64 for the four digests, 32 for HMAC-SHA-256) over every
    // input. The hex loop above pins the hex spelling; this loop pins the
    // digest spelling the `_digest` twins return, so a length or
    // crash regression on that path cannot hide behind hex-only fuzzing.
    assert_eq!(tors::hash_impl::md5_digest(data).len(), 16);
    assert_eq!(tors::hash_impl::sha1_digest(data).len(), 20);
    assert_eq!(tors::hash_impl::sha256_digest(data).len(), 32);
    assert_eq!(tors::hash_impl::sha512_digest(data).len(), 64);

    // The empty-input known-answer canary: the empty input is trivially
    // reachable, and these four constants (RFC 1321 / FIPS 180-4) are
    // what a wiring or padding regression would move first.
    if data.is_empty() {
        assert_eq!(
            tors::hash_impl::md5_hex(b""),
            "d41d8cd98f00b204e9800998ecf8427e"
        );
        assert_eq!(
            tors::hash_impl::sha1_hex(b""),
            "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        );
        assert_eq!(
            tors::hash_impl::sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            tors::hash_impl::sha512_hex(b""),
            "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce\
             47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
        );
        // The digest-spelling mirror of the same canary: the raw bytes a
        // wiring regression would move first, hex-decoded from the same
        // RFC 1321 / FIPS 180-4 constants. (The decode result sits on the
        // left of each compare: std implements `PartialEq<[u8; N]> for
        // Vec<u8>`, not the reverse.)
        assert_eq!(
            const_hex::decode("d41d8cd98f00b204e9800998ecf8427e").unwrap(),
            tors::hash_impl::md5_digest(b"")
        );
        assert_eq!(
            const_hex::decode("da39a3ee5e6b4b0d3255bfef95601890afd80709").unwrap(),
            tors::hash_impl::sha1_digest(b"")
        );
        assert_eq!(
            const_hex::decode("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855").unwrap(),
            tors::hash_impl::sha256_digest(b"")
        );
        assert_eq!(
            const_hex::decode(
                "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce\
                 47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
            ).unwrap(),
            tors::hash_impl::sha512_digest(b"")
        );
    }

    // hmac: crash-freedom + shape over arbitrary key/data (any key length
    // is legal, empty included), the key=data cross spelling, and the RFC
    // 4231 case-2 canary when the fuzzer happens to build exactly that
    // key/data pair — in BOTH output spellings, so a digest-path
    // regression cannot hide behind hex-only coverage.
    let mac = tors::hash_impl::hmac_sha256_hex(key, data);
    assert_eq!(mac.len(), 64);
    assert!(mac.bytes().all(|b| b.is_ascii_hexdigit()));
    let mac_raw = tors::hash_impl::hmac_sha256_digest(key, data);
    assert_eq!(mac_raw.len(), 32);
    let cross = tors::hash_impl::hmac_sha256_hex(data, data);
    assert_eq!(cross.len(), 64);
    assert!(cross.bytes().all(|b| b.is_ascii_hexdigit()));
    let cross_raw = tors::hash_impl::hmac_sha256_digest(data, data);
    assert_eq!(cross_raw.len(), 32);
    if key == b"Jefe" && data == b"what do ya want for nothing?" {
        assert_eq!(
            mac,
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        assert_eq!(
            const_hex::decode("5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843").unwrap(),
            mac_raw
        );
    }
});
