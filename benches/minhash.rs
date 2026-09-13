//! Criterion benches for `minhash_impl::signature`: a document-size ladder
//! (1 KiB / 100 KiB / 1 MiB, the same sizes the Python-side wall cells in
//! `tests/test_performance.py` measure, so the bench numbers and the cell
//! numbers cross-reference) crossed with `num_perm` {64, 128, 512} (the
//! default, half, and 4x: the sweep is O(shingles x num_perm), so the
//! perm axis is the cost knob the ladder isolates).
//!
//! Run locally with `cargo bench --no-default-features --bench minhash`.
//! CI only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`).

#[expect(dead_code)] // `decomposed` and `crlf` have no minhash bench cell
// (the signature pass is normalization-shape-insensitive to first order:
// the tokenizer walk dominates, not the NFC pass), so within THIS bench's
// compilation the shared module's builders are dead — allowed here only;
// the expectation is deliberate and self-retiring. Same pattern as
// benches/search.rs and benches/retrieval.rs.
mod common;

use common::prose;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::minhash_impl;

fn bench_minhash_signature(c: &mut Criterion) {
    let mut group = c.benchmark_group("minhash_signature");
    for size_bytes in [1024usize, 100 * 1024, 1024 * 1024] {
        let text = prose(size_bytes);
        group.throughput(Throughput::Bytes(size_bytes as u64));
        for num_perm in [64usize, 128, 512] {
            group.bench_with_input(
                BenchmarkId::new(format!("k{num_perm}"), size_bytes),
                &text,
                |bench, text| {
                    bench.iter(|| {
                        minhash_impl::signature(black_box(text), black_box(num_perm), 3, 0)
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_minhash_signature);
criterion_main!(benches);
