//! `json_is_valid` never panics on arbitrary bytes, and its whitespace
//! contract is structural: appending JSON whitespace to any document
//! never changes the answer (trailing-whitespace acceptance is part of
//! the RFC 8259 grammar the scanner implements — a scanner that reads
//! one byte past a document's end, or mishandles the ws-before-EOF
//! states, breaks this invariant at raw-byte depth). There is no
//! acceptance-set oracle in this crate to differential against (the
//! orjson-facing equality is the Python suite's job,
//! tests/test_json_is_valid.py — a Rust-side oracle would be a second
//! hand-rolled scanner agreeing with itself), so the panic hunt plus the
//! whitespace invariant are this target's two checks; the scanner's
//! SWAR plain-run skip, the fixed 128-byte container bitset, and the
//! UTF-8/escape index arithmetic are exactly the code class where a
//! panic lurks that unit tests never shape.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let first = tors::json_valid_impl::is_valid(data);

    // The whitespace invariant: the grammar accepts trailing JSON
    // whitespace unconditionally, so the answer is invariant under it.
    for pad in [
        &b" "[..],
        b"\t",
        b"\n",
        b"\r",
        b"  \t\r\n ",
        b"\t\t\t\t\t\t\t\t", // an 8-byte pad lands the tail in the SWAR window
    ] {
        let mut padded = data.to_vec();
        padded.extend_from_slice(pad);
        assert_eq!(
            tors::json_valid_impl::is_valid(&padded),
            first,
            "appending {:?} changed the answer for {data:?}",
            pad
        );
    }
});
