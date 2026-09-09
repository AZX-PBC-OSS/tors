//! Criterion benches for the content-addressing/dedup cluster:
//! `merkle_impl::merkle_root`/`merkle_diff` (RFC 6962-style domain-separated
//! SHA-256 Merkle tree over `list[bytes]` chunks) and `simhash_impl::simhash64`
//! (the 64-bit locality-sensitive fingerprint — near-duplicate detection,
//! the fuzzy sibling to `finalize`'s exact-hash dedupe gate).
//!
//! `merkle_root`/`merkle_diff` take pre-chunked `&[&[u8]]`, not raw bytes —
//! the realistic input is `chunk_cdc`'s own output (see `benches/chunking.rs`),
//! so the corpus here is the prose corpus split into FIXED-size byte chunks
//! (not re-running `chunk_cdc` itself inside this bench — that would
//! conflate the two costs; chunking throughput is `chunking.rs`'s job, tree
//! throughput given ALREADY-CHUNKED input is this file's).
//!
//! `simhash64` runs directly over the prose corpus (it's a whole-text
//! primitive, not chunk-shaped).
//!
//! Run locally with `cargo bench --no-default-features --bench integrity`.
//! CI only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`).

#[expect(dead_code)] // `decomposed` and `crlf` have no integrity-bench cell
// (normalize.rs/bytes.rs/text.rs bench them), so within THIS bench's
// compilation the shared module's builders are dead — allowed here only;
// the expectation is deliberate and self-retiring. Same pattern as
// benches/search.rs and benches/diff.rs.
mod common;

use common::prose;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::merkle_impl;
use tors::simhash_impl;

const CHUNK_SIZE: usize = 4096;

/// Split `bytes` into `CHUNK_SIZE`-byte pieces — a deterministic, fixed-size
/// stand-in for `chunk_cdc`'s output, sized so the chunk COUNT scales with
/// input size the way a real chunked document would.
fn fixed_chunks(bytes: &[u8]) -> Vec<&[u8]> {
    bytes.chunks(CHUNK_SIZE).collect()
}

fn bench_merkle_root(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_root");
    for target_bytes in [1024 * 1024, 12 * 1024 * 1024, 100 * 1024 * 1024] {
        let text = prose(target_bytes);
        let bytes = text.as_bytes();
        let chunks = fixed_chunks(bytes);
        group.throughput(Throughput::Elements(chunks.len() as u64));
        if target_bytes == 100 * 1024 * 1024 {
            group.sample_size(10);
        }
        group.bench_with_input(
            BenchmarkId::new("chunks", format!("{}", chunks.len())),
            &chunks,
            |bench, chunks| bench.iter(|| merkle_impl::merkle_root(black_box(chunks))),
        );
    }
    group.finish();
}

fn bench_merkle_diff(c: &mut Criterion) {
    // The near-identical-with-one-edit shape: every chunk equal except one,
    // near the middle — the realistic "did this document change" scan a
    // sync/dedup pipeline runs.
    let mut group = c.benchmark_group("merkle_diff");
    for target_bytes in [1024 * 1024, 12 * 1024 * 1024] {
        let text_a = prose(target_bytes);
        let bytes_a = text_a.into_bytes();
        let mut bytes_b = bytes_a.clone();
        let mid = bytes_b.len() / 2;
        bytes_b[mid] = bytes_b[mid].wrapping_add(1);
        let chunks_a = fixed_chunks(&bytes_a);
        let chunks_b = fixed_chunks(&bytes_b);
        group.throughput(Throughput::Elements(chunks_a.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("one_edit", format!("{}", chunks_a.len())),
            &(chunks_a, chunks_b),
            |bench, (a, b)| bench.iter(|| merkle_impl::merkle_diff(black_box(a), black_box(b))),
        );
    }
    group.finish();
}

fn bench_simhash64(c: &mut Criterion) {
    let mut group = c.benchmark_group("simhash64");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024] {
        let text = prose(target_bytes);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("prose", format!("{}B", text.len())),
            &text,
            |bench, text| bench.iter(|| simhash_impl::simhash64(black_box(text))),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_merkle_root,
    bench_merkle_diff,
    bench_simhash64
);
criterion_main!(benches);
