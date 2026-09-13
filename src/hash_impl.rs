//! One-shot text/byte hashing: the pure-Rust cores of `tors.md5_hex`,
//! `tors.sha1_hex`, `tors.sha256_hex`, `tors.sha512_hex`, and
//! `tors.hmac_sha256_hex`, and of their raw-digest twins `tors.md5_digest`,
//! `tors.sha1_digest`, `tors.sha256_digest`, `tors.sha512_digest`, and
//! `tors.hmac_sha256_digest`.
//!
//! # Why this module exists
//!
//! The request-signing and content-check primitives a text-operations
//! library keeps getting asked for, in one place: webhook signature
//! verification and API auth (HMAC-SHA-256, the GitHub/Stripe/Slack
//! convention), ETag and Content-MD5 checks against object stores (md5),
//! rsync-style quick content compares and legacy-interop digests (sha1),
//! and dedup-cache keys / general content addressing (sha256/sha512).
//! `finalize` already hashes, but only as the tail of its normalize
//! pipeline; this module is the same digest computation without the
//! transform, over exactly the bytes (or UTF-8 of the str) the caller
//! supplies. Each algorithm computes its digest ONCE and offers two
//! output spellings over that one computation: the raw bytes (the
//! `*_digest` functions — the call sites that base64-encode a signature,
//! chain a digest back in as a key, or slice a stable int off the front)
//! and the lowercase hex (the `*_hex` functions, the cache-key/ETag/
//! request-ID spelling).
//!
//! # Engines: RustCrypto, nothing hand-rolled
//!
//! Every algorithm is the maintained RustCrypto implementation (the
//! RustCrypto org's hash crates: `md-5`, `sha1`, `sha2`, `hmac`), the same family `sha2`
//! already comes from for `finalize` and `merkle_root`. A digest
//! implementation is the worst kind of code to hand-roll: every line is a
//! maintenance burden and a silent-corruption risk the maintained crates
//! already carry primary-source test vectors against (this module's
//! crate-side tests pin those vectors, RFC 1321 / FIPS 180-4 / RFC 4231,
//! against the exact functions the pyo3 wrappers drive).
//!
//! # Security scope (md5 and sha1)
//!
//! `md5_hex`/`md5_digest` and `sha1_hex`/`sha1_digest` are
//! checksum/legacy-interop primitives only: Content-MD5, S3 ETags,
//! cache-busting, quick equality checks. Both are broken for security
//! purposes and have been since the 2000s (practical md5 collisions
//! since 2004; sha1's first public collision in 2017). Never use either
//! for signatures, certificates, or password handling:
//! `sha256`/`sha512`/`hmac_sha256` (either spelling) are the security
//! side of this module. Every user-facing doc surface that names them
//! carries this note.
//!
//! # Stateless one-shot only
//!
//! Each function hashes its whole input in one call and returns the
//! digest — raw bytes or lowercase hex. There is deliberately no
//! streaming update surface and no hash object: tors is stateless by
//! charter (docs/design.md), and a `hashlib`-style constructor object is
//! exactly the persistent-handle shape that charter cuts (the two
//! measured exceptions, `CompiledPatterns`/`CompiledLemmaDict`, exist for
//! per-call re-materialization costs a digest object does not have:
//! `hashlib.sha256()` construction is O(1)). A caller hashing a stream
//! hashes chunk digests and combines them (the `merkle_root` shape), or
//! uses `hashlib` directly — its object API is the right tool for
//! incremental feeding and is not duplicated here.
//!
//! # GIL model
//!
//! The pyo3 wrappers (src/py/hash.rs) borrow the arguments under the GIL
//! (the standard str-in class for a `str`, the zero-copy immutable
//! `PyBytes` borrow for `bytes`) and run the whole digest computation —
//! update and finalize, plus the O(digest-size) hex formatting on the
//! `*_hex` spellings — under one `py.detach`. The GIL-held residue is
//! the argument borrow plus O(output) marshalling: one short `String`
//! for the hex spellings, one fixed-size `PyBytes` for the digest
//! spellings (the `b64_decode` bytes-return class, 16/20/32/64 bytes
//! fixed by algorithm); see the crate GIL model in src/lib.rs.

use hmac::{Hmac, KeyInit, Mac};
use md5::Md5;
use sha1::Sha1;
use sha2::{Digest, Sha256, Sha512};

/// HMAC-SHA-256: the request-signing primitive (webhook signatures,
/// AWS SigV4-style HMAC chains, API auth).
type HmacSha256 = Hmac<Sha256>;

/// MD5 raw digest of `data`, 16 bytes. CHECKSUM/ETAG/LEGACY-INTEROP
/// ONLY, never security: see the module doc.
pub fn md5_digest(data: &[u8]) -> [u8; 16] {
    Md5::digest(data).into()
}

/// MD5 of `data`, lowercase hex: `md5_digest`, hex-encoded.
pub fn md5_hex(data: &[u8]) -> String {
    const_hex::encode(md5_digest(data))
}

/// SHA-1 raw digest of `data`, 20 bytes. CHECKSUM/LEGACY-INTEROP ONLY,
/// never security: see the module doc.
pub fn sha1_digest(data: &[u8]) -> [u8; 20] {
    Sha1::digest(data).into()
}

/// SHA-1 of `data`, lowercase hex: `sha1_digest`, hex-encoded.
pub fn sha1_hex(data: &[u8]) -> String {
    const_hex::encode(sha1_digest(data))
}

/// SHA-256 raw digest of `data`, 32 bytes: the same engine `finalize`'s
/// tail and `merkle_root`'s leaves use, byte-identical to
/// `hashlib.sha256(data).digest()`.
pub fn sha256_digest(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

/// SHA-256 of `data`, lowercase hex: `sha256_digest`, hex-encoded —
/// byte-identical to `hashlib.sha256(data).hexdigest()`.
pub fn sha256_hex(data: &[u8]) -> String {
    const_hex::encode(sha256_digest(data))
}

/// SHA-512 raw digest of `data`, 64 bytes.
pub fn sha512_digest(data: &[u8]) -> [u8; 64] {
    Sha512::digest(data).into()
}

/// SHA-512 of `data`, lowercase hex: `sha512_digest`, hex-encoded.
pub fn sha512_hex(data: &[u8]) -> String {
    const_hex::encode(sha512_digest(data))
}

/// HMAC-SHA-256 raw digest of `data` under `key`, 32 bytes: byte-identical
/// to `hmac.new(key, data, hashlib.sha256).digest()`. Any key length is
/// legal, empty included (parity with the stdlib spelling): HMAC pads
/// short keys and hashes long ones (RFC 2104), so `new_from_slice` cannot
/// fail for `Hmac` — the same expectation hmac's own `KeyInit::new` impl
/// carries (verified against the vendored hmac 0.13.0 source, not just
/// its docs). The key is held in memory for the call and is not zeroized
/// on return — the same posture as the stdlib `hmac`/`hashlib` spelling,
/// which likewise keeps key material in ordinary memory.
pub fn hmac_sha256_digest(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts keys of any length");
    mac.update(data);
    mac.finalize().into_bytes().into()
}

/// HMAC-SHA-256 of `data` under `key`, lowercase hex:
/// `hmac_sha256_digest`, hex-encoded — byte-identical to
/// `hmac.new(key, data, hashlib.sha256).hexdigest()`. Key handling is
/// `hmac_sha256_digest`'s contract verbatim (any length legal, empty
/// included; held in memory for the call, not zeroized — stdlib posture).
pub fn hmac_sha256_hex(key: &[u8], data: &[u8]) -> String {
    const_hex::encode(hmac_sha256_digest(key, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    // RFC 1321 A.5, the md5 vectors this module's engines are pinned
    // against (crate-side twins of tests/test_hash.py's pins).
    #[test]
    fn md5_rfc1321_vectors() {
        assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex(b"a"), "0cc175b9c0f1b6a831c399e269772661");
        assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(
            md5_hex(b"message digest"),
            "f96b697d7cb7938d525a2f31aaf161d0"
        );
        assert_eq!(
            md5_hex(b"abcdefghijklmnopqrstuvwxyz"),
            "c3fcd3d76192e4007dfb496cca67e13b"
        );
        assert_eq!(
            md5_hex(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
            "d174ab98d277d9f5a5611c2c9f419d9f"
        );
        assert_eq!(
            md5_hex(
                b"123456789012345678901234567890123456789012345678901234567890\
                  12345678901234567890"
            ),
            "57edf4a22be3c955ac49da2e2107b67a"
        );
        assert_eq!(
            md5_hex(b"The quick brown fox jumps over the lazy dog"),
            "9e107d9d372bb6826bd81d3542a419d6"
        );
    }

    // FIPS 180-4 / RFC 3174 sha1 vectors.
    #[test]
    fn sha1_fips180_vectors() {
        assert_eq!(sha1_hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(
            sha1_hex(b"a"),
            "86f7e437faa5a7fce15d1ddcb9eaeaea377667b8"
        );
        assert_eq!(
            sha1_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
        assert_eq!(
            sha1_hex(
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno\
                  ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
            ),
            "a49b2446a02c645bf419f995b67091253a04a259"
        );
        assert_eq!(
            sha1_hex(b"The quick brown fox jumps over the lazy dog"),
            "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12"
        );
    }

    // FIPS 180-4 sha256/sha512 vectors, including the two-block and
    // million-'a' examples.
    #[test]
    fn sha256_fips180_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"a"),
            "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            sha256_hex(
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno\
                  ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
            ),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
        );
        assert_eq!(
            sha256_hex(&[b'a'; 1_000_000]),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn sha512_fips180_vectors() {
        assert_eq!(
            sha512_hex(b""),
            "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce\
             47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
        );
        assert_eq!(
            sha512_hex(b"abc"),
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a\
             2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
        );
        assert_eq!(
            sha512_hex(b"a"),
            "1f40fc92da241694750979ee6cf582f2d5d7d28e18335de05abc54d0560e0f53\
             02860c652bf08d560252aa5e74210546f369fbbbce8c12cfc7957b2652fe9a75"
        );
        assert_eq!(
            sha512_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "204a8fc6dda82f0a0ced7beb8e08a41657c16ef468b228a8279be331a703c335\
             96fd15c13b1b07f9aa1d3bea57789ca031ad85c7a71dd70354ec631238ca3445"
        );
        assert_eq!(
            sha512_hex(
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno\
                  ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
            ),
            "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018\
             501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909"
        );
        assert_eq!(
            sha512_hex(&[b'a'; 1_000_000]),
            "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973eb\
             de0ff244877ea60a4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b"
        );
    }

    // RFC 4231's HMAC-SHA-256 cases, verbatim (5 is the truncation case,
    // not applicable to a full-digest function).
    #[test]
    fn hmac_sha256_rfc4231_vectors() {
        assert_eq!(
            hmac_sha256_hex(&[0x0b; 20], b"Hi There"),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        assert_eq!(
            hmac_sha256_hex(b"Jefe", b"what do ya want for nothing?"),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        assert_eq!(
            hmac_sha256_hex(&[0xaa; 20], &[0xdd; 50]),
            "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe"
        );
        assert_eq!(
            hmac_sha256_hex(&(1u8..26).collect::<Vec<u8>>(), &[0xcd; 50]),
            "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b"
        );
        assert_eq!(
            hmac_sha256_hex(
                &[0xaa; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            ),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
        assert_eq!(
            hmac_sha256_hex(
                &[0xaa; 131],
                b"This is a test using a larger than block-size key and a larger \
                  than block-size data. The key needs to be hashed before being \
                  used by the HMAC algorithm."
            ),
            "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2"
        );
    }

    #[test]
    fn empty_key_is_legal_and_is_the_zero_padded_key() {
        // Parity with stdlib hmac.new(b"", ...): RFC 2104 pads a key
        // shorter than the block size with zero bytes, so the empty key
        // IS the 64-zero-byte key, and the digest is a pinned constant.
        assert_eq!(
            hmac_sha256_hex(b"", b"data"),
            "e528c4d99e6177f5841f712a143b90843299a4aa181a06501422d9ca862bd2a5"
        );
        assert_eq!(
            hmac_sha256_hex(&[0u8; 64], b"data"),
            hmac_sha256_hex(b"", b"data")
        );
    }

    #[test]
    fn hex_spellings_are_the_digest_spellings_hex_encoded() {
        // The one-digest-two-spellings invariant this module's shape
        // promises (the hex path consumes the digest path, no second
        // computation): pinned crate-side over a small battery, the
        // Python-side hypothesis lane pins it differentially.
        let inputs: [&[u8]; 5] = [
            b"",
            b"a",
            b"abc",
            &[0x00, 0xff, 0x7f],
            b"The quick brown fox jumps over the lazy dog",
        ];
        for raw in inputs {
            assert_eq!(md5_hex(raw), const_hex::encode(md5_digest(raw)));
            assert_eq!(sha1_hex(raw), const_hex::encode(sha1_digest(raw)));
            assert_eq!(sha256_hex(raw), const_hex::encode(sha256_digest(raw)));
            assert_eq!(sha512_hex(raw), const_hex::encode(sha512_digest(raw)));
            assert_eq!(
                hmac_sha256_hex(b"key", raw),
                const_hex::encode(hmac_sha256_digest(b"key", raw))
            );
        }
    }

    #[test]
    fn digest_spellings_pin_the_primary_source_vectors_raw() {
        // The empty-input and "abc" vectors in their raw-byte form: the
        // fixed-size return types are the length pins (16/20/32/64), and
        // these bytes are the same RFC 1321 / FIPS 180-4 / RFC 4231
        // constants the hex tests pin, decoded. (The decode result sits
        // on the left of each compare: std implements
        // `PartialEq<[u8; N]> for Vec<u8>`, not the reverse.)
        assert_eq!(
            const_hex::decode("d41d8cd98f00b204e9800998ecf8427e").unwrap(),
            md5_digest(b"")
        );
        assert_eq!(
            const_hex::decode("a9993e364706816aba3e25717850c26c9cd0d89d").unwrap(),
            sha1_digest(b"abc")
        );
        assert_eq!(
            const_hex::decode("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
                .unwrap(),
            sha256_digest(b"")
        );
        assert_eq!(
            const_hex::decode(
                "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a\
                 2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
            )
            .unwrap(),
            sha512_digest(b"abc")
        );
        assert_eq!(
            const_hex::decode("5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843")
                .unwrap(),
            hmac_sha256_digest(b"Jefe", b"what do ya want for nothing?")
        );
    }
}
