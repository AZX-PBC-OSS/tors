//! The uuid text surface never panics on arbitrary strings, and the
//! three spellings agree with each other: `parse_canonical`'s Ok bytes
//! feed `uuid_version`'s nibble and `v7_timestamp_ms`'s read, the
//! version nibble is a v7's iff the timestamp read succeeds under the
//! 48-bit field, and the canonical lower-hex spelling of parsed bytes
//! re-parses to the same bytes (the grammar's fixed point). The strict
//! grammar (36 chars, canonical hyphenated hex ONLY: the stdlib's
//! brace/urn/hyphenless tolerances are tors' documented divergences) is
//! exercised over raw fuzzer strings.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tors::uuid_impl::{parse_canonical, v7_timestamp_ms, version_nibble};

fuzz_target!(|s: &str| {
    if let Ok(bytes) = parse_canonical(s) {
        let version = version_nibble(&bytes);
        if version == 7 {
            // A v7's timestamp is the 48-bit ms-since-epoch field: the
            // read cannot fail on bytes that really are v7.
            let ts = v7_timestamp_ms(&bytes).expect("v7_timestamp_ms failed on a v7-parsed UUID");
            assert!(
                ts < (1u64 << 48),
                "v7 timestamp over the 48-bit field: {ts}"
            );
        }
        // Round-trip: the canonical lower-hex spelling of the parsed
        // bytes re-parses to the same bytes (the grammar's fixed point).
        let canonical = uuid_fmt(&bytes);
        let reparsed = parse_canonical(&canonical).expect("canonical spelling re-parse failed");
        assert_eq!(reparsed, bytes, "round-trip drift on {s:?}");
    }
});

/// The canonical `8-4-4-4-12` lower-hex spelling (the shape
/// `parse_canonical` accepts; hand-spelled to keep this target's
/// dependency footprint at the impl, not a re-export chain).
fn uuid_fmt(bytes: &[u8; 16]) -> String {
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}
