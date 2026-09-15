//! Criterion benches for `minhash_impl::signature`: a document-size ladder
//! (1 KiB / 100 KiB / 1 MiB, the same sizes the Python-side wall cells in
//! `tests/test_performance.py` measure, so the bench numbers and the cell
//! numbers cross-reference) crossed with `num_perm` {64, 128, 512} (the
//! default, half, and 4x: the sweep is O(distinct x num_perm), so the
//! perm axis is the cost knob the ladder isolates), plus the
//! distinct-rich worst-case row below.
//!
//! Run locally with `cargo bench --no-default-features --bench minhash`.
//! CI only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`).
//!
//! The worst-case row (`distinct_1MiB`, k=128/1024) runs the sweep where
//! the prose ladder does not: the repeated-sentence corpus rides a handful
//! of distinct shingles, so its sweep is trivial at any `num_perm`; the
//! hex-counter corpus below is ~every token unique, so the
//! O(distinct x num_perm) sweep actually runs (~100M affine ops at
//! k=1024). Its own group with few samples: each iteration is hundreds of
//! milliseconds, and CI never runs it anyway.

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

/// Deterministic distinct-rich corpus: zero-padded hex-counter words,
/// every token unique, so the distinct-shingle set is ~the token count
/// (the same construction `tests/test_minhash.py`'s worst-case cell
/// uses, so the bench and cell cross-reference).
fn distinct_text(target_bytes: usize) -> String {
    let n = target_bytes / 9;
    let mut out = String::with_capacity(n * 9);
    for i in 0..n {
        use std::fmt::Write as _;
        write!(out, "w{i:07x} ").expect("write to String cannot fail");
    }
    out
}

fn bench_minhash_distinct_worst_case(c: &mut Criterion) {
    let mut group = c.benchmark_group("minhash_signature_distinct");
    group.sample_size(10);
    let text = distinct_text(1024 * 1024);
    group.throughput(Throughput::Bytes(text.len() as u64));
    for num_perm in [128usize, 1024] {
        group.bench_with_input(
            BenchmarkId::new(format!("k{num_perm}"), "1MiB-distinct"),
            &text,
            |bench, text| {
                bench.iter(|| minhash_impl::signature(black_box(text), black_box(num_perm), 3, 0))
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_minhash_signature,
    bench_minhash_distinct_worst_case
);
criterion_main!(benches);
