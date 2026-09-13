//! Criterion benches for the one-shot hashing surface
//! (`hash_impl::md5_hex`/`sha1_hex`/`sha256_hex`/`sha512_hex`/
//! `hmac_sha256_hex` and their raw-digest `_digest` twins), the
//! request-signing / content-check primitives.
//!
//! `md5_*`/`sha1_*` (either spelling) are checksum/legacy-interop only,
//! never security (see `hash_impl`'s scope section); their rows below are
//! the Content-MD5/ETag/quick-compare jobs.
//!
//! The digest ladder (1 KiB / 1 MiB / 12 MiB) measures the engines at the
//! sizes the Python-side wall cells and docs/performance.md quote
//! (throughput sizes, where per-call overhead is amortized), and the hmac
//! group measures the request-signature shapes: short payloads where the
//! one-call overhead IS the cost (the RFC 4231 case-1 and case-6
//! key/data shapes, 20B/8B and 131B/54B, plus a 1 MiB payload row for
//! the engine-size end) in BOTH output spellings — the `_digest` twin is
//! the same computation minus the hex tail, and the `(32, 256)` digest
//! row is the request shape `tests/test_performance.py`'s HMAC wall cell
//! asserts, so the bench covers what the wall cell gates.
//!
//! No Rust-side opponent: the honest stdlib comparison (OpenSSL-backed
//! hashlib, hardware SHA extensions) is a Python-interpreter
//! measurement and lives in tests/test_performance.py's cells and
//! docs/performance.md's table, not here; racing the same sha2 crate
//! through a second spelling would be a fake opponent.
//!
//! Corpora: the shared prose recipe (`benches/common/mod.rs`, pinned
//! byte-identical to `tests/reference.py` by
//! tests/test_bench_corpus_parity.py), hashed as its UTF-8 bytes — the
//! same bytes the Python-side cells measure via `reference.corpus_utf8`.
//!
//! Run locally with `cargo bench --no-default-features --bench hashing`.
//! CI only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`).

#[expect(dead_code)] // `decomposed` and `crlf` have no hashing-bench cell
// (normalize.rs/bytes.rs/text.rs bench them), so within THIS bench's
// compilation the shared module's builders are dead — allowed here only;
// the expectation is deliberate and self-retiring. Same pattern as
// benches/integrity.rs and benches/search.rs.
mod common;

use common::prose;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::hash_impl;

fn bench_digest(c: &mut Criterion) {
    // The four one-shot digests over the same byte ladder, one group per
    // algorithm so criterion's per-algorithm throughput lines are directly
    // comparable with the wall cells and docs/performance.md.
    for (group_name, core) in [
        ("md5_hex", hash_impl::md5_hex as fn(&[u8]) -> String),
        ("sha1_hex", hash_impl::sha1_hex as fn(&[u8]) -> String),
        ("sha256_hex", hash_impl::sha256_hex as fn(&[u8]) -> String),
        ("sha512_hex", hash_impl::sha512_hex as fn(&[u8]) -> String),
    ] {
        let mut group = c.benchmark_group(group_name);
        for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024] {
            let text = prose(target_bytes);
            let bytes = text.as_bytes();
            group.throughput(Throughput::Bytes(bytes.len() as u64));
            group.bench_with_input(
                BenchmarkId::new("prose", format!("{}B", bytes.len())),
                bytes,
                |bench, bytes| bench.iter(|| core(black_box(bytes))),
            );
        }
        group.finish();
    }
}

fn bench_hmac(c: &mut Criterion) {
    // The request-signing shapes: the RFC 4231 case-1 (20-byte key, 8-byte
    // data) and case-6 (131-byte key, 54-byte data) key/data pairs — the
    // webhook/API-auth shapes where the one-call overhead is the cost —
    // plus a 1 MiB payload row for the engine-size end.
    let mut group = c.benchmark_group("hmac_sha256_hex");
    for (key_len, data_len) in [(20, 8), (131, 54), (32, 1024 * 1024)] {
        let key = vec![0x0bu8; key_len];
        // prose(n) quantizes DOWN to a whole unit (floor division), so the
        // slice needs one unit of slack to always cover data_len.
        let data = prose(data_len + 1024).as_bytes()[..data_len].to_vec();
        group.throughput(Throughput::Bytes(data_len as u64));
        group.bench_with_input(
            BenchmarkId::new("sign", format!("k{key_len}d{data_len}")),
            &(key, data),
            |bench, (key, data)| {
                bench.iter(|| hash_impl::hmac_sha256_hex(black_box(key), black_box(data)))
            },
        );
    }
    group.finish();
}

fn bench_hmac_digest(c: &mut Criterion) {
    // The raw-digest twin of the request-signing shapes: the same keyed
    // computation without the hex tail. The (32, 256) row is recorded, not
    // wall-gated: the wall cell (`test_hmac_sha256_hex_beats_the_fastest_
    // stdlib_hmac_spelling`) asserts the `_hex` spelling of this shape, and
    // the digest spelling is that same computation minus the hex tail, so
    // the overhead-dominated shape is benched on both spellings while only
    // the hex one carries a gate.
    let mut group = c.benchmark_group("hmac_sha256_digest");
    {
        let (key_len, data_len) = (32, 256);
        let key = vec![0x0bu8; key_len];
        let data = prose(data_len + 1024).as_bytes()[..data_len].to_vec();
        group.throughput(Throughput::Bytes(data_len as u64));
        group.bench_with_input(
            BenchmarkId::new("sign", format!("k{key_len}d{data_len}")),
            &(key, data),
            |bench, (key, data)| {
                bench.iter(|| hash_impl::hmac_sha256_digest(black_box(key), black_box(data)))
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_digest, bench_hmac, bench_hmac_digest);
criterion_main!(benches);
