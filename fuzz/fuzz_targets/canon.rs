//! `content_hash`'s emitter under adversarial value trees: crash-freedom,
//! determinism, the two-sink agreement (streaming digest == SHA-256 over
//! the collected canonical bytes), and the `ensure_ascii` structural
//! invariants -- the canonical form is pure ASCII and no control or DEL
//! byte ever appears raw -- over trees whose STRINGS come from two
//! adversarial sources: the raw fuzz bytes themselves (every byte-level
//! edge of the escape table: controls, DEL, quotes, backslash runs,
//! invalid sequences rendered U+FFFD by the lossy conversion) and
//! arbitrary u32 codepoints (the whole valid scalar space: BMP
//! non-ASCII, the surrogate-block boundaries, astral codepoints and
//! U+10FFFF, where the surrogate-pair arithmetic lives).
//!
//! The Python-boundary differential (tests/test_content_hash.py, against
//! `json.dumps` itself) is the real parity net; this target is the other
//! bug class, per the fuzz crate's own doctrine: raw adversarial shapes
//! a hypothesis strategy never draws, straight into the pure-Rust core.

#![no_main]

use libfuzzer_sys::fuzz_target;
use sha2::{Digest, Sha256};
use tors::canon_impl::{self, Canon};

const FLOAT_SPELLINGS: [&str; 6] = ["0.1", "1e+16", "-0.0", "NaN", "Infinity", "5e-324"];

fn next(data: &[u8], cursor: &mut usize) -> u8 {
    let b = data.get(*cursor).copied().unwrap_or(0);
    *cursor += 1;
    b
}

fn next_slice<'a>(data: &'a [u8], cursor: &mut usize, len: usize) -> &'a [u8] {
    let start = (*cursor).min(data.len());
    let end = (start + len).min(data.len());
    *cursor = end;
    &data[start..end]
}

/// A raw-bytes string: the lossy conversion guarantees valid UTF-8 while
/// keeping every byte-level escape edge reachable (and exercises U+FFFD
/// itself, a non-ASCII escape).
fn raw_string(data: &[u8], cursor: &mut usize) -> String {
    let len = (next(data, cursor) as usize) % 97;
    String::from_utf8_lossy(next_slice(data, cursor, len)).into_owned()
}

/// A codepoint string: raw u32s (little-endian) filtered to valid scalar
/// values, so the whole space -- controls, DEL, `"`, `\`, BMP non-ASCII,
/// the astral range and its extremes -- reaches the escape table.
fn codepoint_string(data: &[u8], cursor: &mut usize) -> String {
    let len = (next(data, cursor) as usize) % 33;
    let mut out = String::new();
    for chunk in next_slice(data, cursor, len * 4).chunks(4) {
        let mut raw = [0u8; 4];
        raw[..chunk.len()].copy_from_slice(chunk);
        if let Some(ch) = char::from_u32(u32::from_le_bytes(raw)) {
            out.push(ch);
        }
    }
    out
}

/// The adversarial tree: a byte-tagged builder over the fuzz input, depth
/// capped at 256 (the iterative emitter and the iterative `Drop` handle
/// any depth -- pinned at 100k in the crate-side tests -- so the fuzzer's
/// job is the SHAPE surface, not the depth claim).
fn build(data: &[u8], cursor: &mut usize, depth: usize) -> Canon {
    if depth >= 256 {
        return Canon::Null;
    }
    match next(data, cursor) % 12 {
        0 => Canon::Null,
        1 => Canon::Bool(true),
        2 => Canon::Bool(false),
        3 => {
            // Byte-by-byte (a short input pads with zeros): the builder is
            // total over ANY byte string, including 1-2 byte fuzzer inputs
            // -- a length-checked copy_from_slice here was the first crash
            // this target ever found, in this builder itself.
            let mut raw = [0u8; 8];
            for slot in &mut raw {
                *slot = next(data, cursor);
            }
            Canon::Int(i64::from_le_bytes(raw)) // every bit pattern incl. MIN/MAX
        }
        4 => Canon::Int((next(data, cursor) as i64) - 128), // small ints, negatives
        5 => {
            // A big-int spelling: decimal digits only, the shape the walk
            // materializes via Python's int->str.
            let len = (next(data, cursor) as usize) % 65;
            let digits: String = next_slice(data, cursor, len)
                .iter()
                .map(|&b| char::from(b'0' + b % 10))
                .collect();
            let signed = if next(data, cursor) % 2 == 0 { "-" } else { "" };
            Canon::BigInt(format!("{signed}{digits}"))
        }
        6 => {
            let idx = (next(data, cursor) as usize) % FLOAT_SPELLINGS.len();
            Canon::Float(FLOAT_SPELLINGS[idx].to_string())
        }
        7 => Canon::Str(raw_string(data, cursor)),
        8 => Canon::Str(codepoint_string(data, cursor)),
        9 => {
            let len = (next(data, cursor) as usize) % 9;
            Canon::Seq((0..len).map(|_| build(data, cursor, depth + 1)).collect())
        }
        _ => {
            let len = (next(data, cursor) as usize) % 9;
            Canon::Map(
                (0..len)
                    .map(|_| {
                        // Keys take the scalar shapes too (coerced string
                        // forms: digits, true/false, spellings, and the
                        // two adversarial string sources).
                        let key = match next(data, cursor) % 5 {
                            0 => (next(data, cursor) as i64).to_string(),
                            1 => "true".to_string(),
                            2 => "null".to_string(),
                            3 => raw_string(data, cursor),
                            _ => codepoint_string(data, cursor),
                        };
                        (key, build(data, cursor, depth + 1))
                    })
                    .collect(),
            )
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut cursor = 0usize;
    let tree = build(data, &mut cursor, 0);

    // Crash-freedom plus determinism: the same tree hashes identically,
    // twice, through the streaming path.
    let digest = canon_impl::digest_hex(&tree);
    assert_eq!(digest, canon_impl::digest_hex(&tree));

    // The hex shape: 64 lowercase hex characters.
    assert_eq!(digest.len(), 64, "digest is not 64 chars: {digest}");
    assert!(
        digest
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "digest is not lowercase hex: {digest}"
    );

    // The ensure_ascii structural invariants over the collected bytes:
    // pure ASCII, and no control or DEL byte ever raw (every one of them
    // must arrive as escape TEXT, never as the byte itself).
    let bytes = canon_impl::canonical_bytes(&tree);
    assert!(bytes.is_ascii(), "canonical form is not pure ASCII");
    assert!(
        !bytes.iter().any(|&b| b < 0x20 || b == 0x7F),
        "a raw control or DEL byte leaked into the canonical form"
    );

    // The two-sink agreement: the streaming digest is SHA-256 over
    // exactly the collected canonical bytes (the same sha2/const-hex the
    // main crate pins, hand-synced in this crate's Cargo.toml).
    assert_eq!(
        digest,
        const_hex::encode(Sha256::digest(&bytes)),
        "streaming digest disagrees with the digest over the collected bytes"
    );
});
