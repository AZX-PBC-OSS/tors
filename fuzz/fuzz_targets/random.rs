//! `random_impl` never panics on arbitrary (seed, size, alphabet) inputs,
//! and its seeded outputs carry the family's structural contracts: charset
//! membership, exact lengths at ANY length (odd hex lengths and
//! non-multiple-of-4 b64url lengths are first-class — the token contract
//! has no alignment constraint), the UUID field bits and canonical
//! charset, the four token spellings' delegation to `random_string` (one
//! char-sampling engine), and same-input-same-output determinism. The fuzz
//! bytes drive a FIXED-seed ChaCha stream (they choose the seed, the
//! sizes, and the alphabet), so every assertion is deterministic — the
//! unseeded OS path is deliberately not fuzzed (its output is not
//! reproducible, so there is nothing to assert beyond the crash-freedom
//! the seeded path already exercises); the Python suite owns the unseeded
//! shape contracts.
//!
//! What died with the length-first refactor, so nobody re-pins it: the
//! b64url padding-math invariant (the `padded=` parameter is gone —
//! padding was an encoding concept, not a token concept) and the
//! byte-fill length coupling (2n hex chars per n bytes drawn). Both are
//! replaced by the plain exact-length contract over the token alphabets.
//!
//! `arbitrary`'s `String` is valid UTF-8 (no lone surrogates — the same
//! bound `b64_decode`'s target documents), and it may be EMPTY: the
//! empty-alphabet refusal is one of the asserted contracts here.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::time::{SystemTime, UNIX_EPOCH};
use tors::random_impl::{self, RandomError};

#[derive(Arbitrary, Debug)]
struct Input {
    seed: u64,
    length: u8,
    alphabet: String,
}

// Spelled in full, not imported: these are the independent charset pins
// the assertions below check the crate's constants against.
const HEX: &[u8] = b"0123456789abcdef";
const B62: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const B64URL: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// The canonical-UUID field checks shared by the v4 and v7 shapes: 36
/// chars, hyphens at 8/13/18/23, lowercase hex everywhere else, the version
/// nibble at index 14, the RFC 4122 variant nibble at index 19.
fn assert_uuid_shape(value: &str, version: u8) {
    let bytes = value.as_bytes();
    assert_eq!(bytes.len(), 36, "uuid is not 36 chars: {value}");
    assert_eq!(&bytes[8..9], b"-");
    assert_eq!(&bytes[13..14], b"-");
    assert_eq!(&bytes[18..19], b"-");
    assert_eq!(&bytes[23..24], b"-");
    assert_eq!(bytes[14], version, "wrong version nibble in {value}");
    assert!(
        matches!(bytes[19], b'8' | b'9' | b'a' | b'b'),
        "wrong variant nibble in {value}"
    );
    assert!(
        value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase() || b == b'-'),
        "non-canonical character in {value}"
    );
}

fuzz_target!(|input: Input| {
    let Input {
        seed,
        length,
        alphabet,
    } = input;
    let length = length as usize;
    let seed = Some(seed);

    // random_string: the empty alphabet is the documented refusal; any
    // non-empty alphabet yields exactly `length` of its own characters,
    // deterministically.
    match random_impl::random_string(length, &alphabet, seed) {
        Err(RandomError::EmptyAlphabet) => assert!(alphabet.is_empty()),
        Ok(first) => {
            assert!(!alphabet.is_empty());
            let second = random_impl::random_string(length, &alphabet, seed).unwrap();
            assert_eq!(first, second, "seeded random_string is not deterministic");
            assert_eq!(first.chars().count(), length, "wrong character count");
            assert!(
                first.chars().all(|c| alphabet.contains(c)),
                "output character outside the alphabet"
            );
            // The b62 delegation: one engine, so the outputs must be equal.
            let b62 = random_impl::random_b62(length, seed).unwrap();
            assert_eq!(
                b62,
                random_impl::random_string(length, random_impl::BASE62_CHARS, seed).unwrap()
            );
            assert!(b62.bytes().all(|b| B62.contains(&b)));
        }
        Err(other) => panic!("unexpected error shape: {other:?}"),
    }

    // random_hex: the string engine over the hex alphabet — exact length
    // (odd included), lowercase hex charset, deterministic.
    let hex_first = random_impl::random_hex(length, seed).unwrap();
    assert_eq!(
        hex_first,
        random_impl::random_hex(length, seed).unwrap(),
        "seeded random_hex is not deterministic"
    );
    assert_eq!(
        hex_first,
        random_impl::random_string(length, random_impl::HEX_CHARS, seed).unwrap(),
        "random_hex is not the string engine"
    );
    assert_eq!(hex_first.len(), length);
    assert!(hex_first.bytes().all(|b| HEX.contains(&b)));

    // random_b64url: the string engine over the 64-char urlsafe alphabet —
    // exact length at ANY length (no alignment constraint; the padding
    // math died with the encoding contract), deterministic, '=' never.
    let b64_first = random_impl::random_b64url(length, seed).unwrap();
    assert_eq!(
        b64_first,
        random_impl::random_b64url(length, seed).unwrap(),
        "seeded random_b64url is not deterministic"
    );
    assert_eq!(
        b64_first,
        random_impl::random_string(length, random_impl::B64URL_CHARS, seed).unwrap(),
        "random_b64url is not the string engine"
    );
    assert_eq!(b64_first.len(), length, "wrong b64url token length");
    assert!(
        b64_first.bytes().all(|b| B64URL.contains(&b)),
        "non-urlsafe character in the token"
    );
    assert!(
        !b64_first.contains('='),
        "'=' can never appear: no padded spelling exists"
    );

    // uuid4: deterministic under seed, canonical v4 shape.
    let v4_first = random_impl::uuid4(seed).unwrap();
    assert_eq!(
        v4_first,
        random_impl::uuid4(seed).unwrap(),
        "seeded uuid4 is not deterministic"
    );
    assert_uuid_shape(&v4_first, b'4');

    // uuid7: unseeded and timestamp-carried, so only the shape and a
    // generous clock sanity (the 48-bit field — the first 12 hex digits,
    // which in the canonical string are u[:8] (the high 32 bits) and
    // u[9..13] (the low 16) — within a day of now; a badly skewed host
    // clock is an environment finding, not a crash, and the day window
    // tolerates any sane CI runner).
    let v7 = random_impl::uuid7().unwrap();
    assert_uuid_shape(&v7, b'7');
    let ts = (u64::from_str_radix(&v7[..8], 16).unwrap() << 16)
        | u64::from_str_radix(&v7[9..13], 16).unwrap();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("fuzz host clock before the Unix epoch")
        .as_millis() as u64;
    assert!(
        ts.abs_diff(now) < 24 * 60 * 60 * 1000,
        "uuid7 timestamp {ts} is not within a day of now {now}"
    );
});
