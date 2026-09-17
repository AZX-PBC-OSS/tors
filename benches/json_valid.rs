//! Criterion benches for `json_valid_impl::is_valid` — the RFC 8259
//! validity scanner behind `tors.json_is_valid` (#61). The corpus shapes
//! are #61's measured prototype's own (the reference points the issue
//! records: orjson.loads discard vs the scanner), generated
//! deterministically here so the lanes are reproducible without a corpus
//! file:
//!
//! - `list_of_dicts`: 64 KiB and 1 MiB lists of small dicts — the
//!   consumer's shape (TaskQ's terminal write, 64 KiB default / 1 MiB
//!   ceiling), where the object-tree cost the scan skips is largest.
//!   Issue reference points: orjson 160 µs / 3.4 ms, scanner 42 µs /
//!   0.66 ms.
//! - `single_big_string`: 64 KiB of string content in one value — the
//!   scan's worst shape relative to a full parse (orjson is already
//!   near-linear over a single string), kept because it pins the string
//!   path (the SWAR plain-run skip) instead of the container loop.
//!   Issue reference point: orjson 10.5 µs, scanner 5.7 µs.
//! - `invalid_tail`: the 1 MiB valid document with its final byte
//!   replaced — the same whole-input-pass discipline as the utf8 bench's
//!   invalid corpus (a failure anywhere earlier would stop the scan and
//!   measure less than the full pass).
//!
//! These guard tors against its own regressions; the cross-implementation
//! claims are owned by the Python suite (tests/test_json_is_valid.py's
//! differential battery) and the orjson-facing acceptance-set equality by
//! the same file. Run locally with `cargo bench --no-default-features
//! --bench json_valid`; CI only compiles it (`cargo bench --no-run`).

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::json_valid_impl;

/// A deterministic list-of-small-dicts document of ~`target_bytes` bytes:
/// the issue's 64 KiB / 1 MiB prototype corpus shape.
fn list_of_dicts(target_bytes: usize) -> Vec<u8> {
    let mut doc = Vec::with_capacity(target_bytes + 64);
    doc.extend_from_slice(b"[");
    let mut i = 0u64;
    while doc.len() < target_bytes {
        if i > 0 {
            doc.extend_from_slice(b",");
        }
        doc.extend_from_slice(
            format!(
                r#"{{"id": {i}, "src": "worker-{}", "ok": true, "score": {}.{}, "msg": "result payload {i}"}}"#,
                i % 8,
                i % 1000,
                i % 10,
            )
            .as_bytes(),
        );
        i += 1;
    }
    doc.extend_from_slice(b"]");
    doc
}

/// A 64 KiB single big string (the scan's string-path shape).
fn single_big_string(target_bytes: usize) -> Vec<u8> {
    let unit = "The quarterly oil sample interval was adjusted after the torque specifications changed, with \\\"quotes\\\" and \\n escapes mixed in. ";
    let mut doc = Vec::with_capacity(target_bytes + 2);
    doc.push(b'"');
    while doc.len() < target_bytes {
        doc.extend_from_slice(unit.as_bytes());
    }
    doc.push(b'"');
    doc
}

fn bench(c: &mut Criterion, name: &str, doc: &[u8]) {
    let mut group = c.benchmark_group("json_is_valid");
    group.throughput(Throughput::Bytes(doc.len() as u64));
    group.bench_with_input(name, doc, |b, d| {
        b.iter(|| black_box(json_valid_impl::is_valid(black_box(d))))
    });
    group.finish();
}

fn bench_json_is_valid(c: &mut Criterion) {
    bench(c, "list_of_dicts_64KiB", &list_of_dicts(64 * 1024));
    let mib = list_of_dicts(1024 * 1024);
    bench(c, "list_of_dicts_1MiB", &mib);
    bench(c, "single_big_string_64KiB", &single_big_string(64 * 1024));
    // The 1 MiB valid document with its last byte clobbered: same linear
    // pass, failing at the very end (the utf8 bench's invalid-corpus rule).
    let mut invalid = mib.clone();
    let last = invalid.len() - 1;
    invalid[last] = b'x';
    assert!(!json_valid_impl::is_valid(&invalid));
    bench(c, "invalid_tail_1MiB", &invalid);
}

criterion_group!(benches, bench_json_is_valid);
criterion_main!(benches);
