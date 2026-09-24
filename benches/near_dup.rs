//! Criterion benches for the near-duplicate comparison layer
//! (`near_dup_impl`): pairwise shingle similarity at several text sizes
//! (the `shingle_jaccard` cost ladder), and `dedup_near_dup`'s documented
//! O(n²) pair sweep at n = 100 / 1k / 10k documents.
//!
//! Run locally with `cargo bench --no-default-features --bench near_dup`.
//! CI only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`).
//!
//! The dedup sizes are the honest statement of the doctrine's
//! small-candidate-set scope: the simhash sweep is 100/1k/10k documents
//! (the 10k row is ~50M pair checks — seconds, sampled sparingly), while
//! the shingle and minhash methods stop at 1k (every shingle-method pair
//! check is a set intersection / signature zip, ~50x the popcount's
//! cost; at 10k documents one iteration would take minutes). The
//! quadratic wall itself is pinned with an explicit budget in
//! `tests/test_scaling_pins.py`.

#[expect(dead_code)] // `decomposed` and `crlf` have no near-dup bench cell
// (the shingle pass is normalization-shape-insensitive to first order:
// the tokenizer walk dominates, not the NFC pass), so within THIS bench's
// compilation the shared module's builders are dead — allowed here only;
// the expectation is deliberate and self-retiring. Same pattern as
// benches/minhash.rs.
mod common;

use common::prose;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::near_dup_impl;

/// A deterministic corpus of `n` documents of `doc_bytes` bytes each:
/// prose bodies (the shared `prose` recipe) made pairwise DISTINCT by a
/// zero-padded index token block, so the greedy dedup accumulates kept
/// representatives and walks its full O(n²) pair ladder (a shared-body
/// corpus would merge into one group and exit every pair check on the
/// first representative).
fn distinct_corpus(n: usize, doc_bytes: usize) -> Vec<String> {
    let body = prose(doc_bytes / 2);
    (0..n)
        .map(|i| {
            // ~10 unique tokens per document: enough that the simhash
            // fingerprints differ, a rounding error against the body.
            let tag = (0..10)
                .map(|j| format!("u{i:06x}x{j:03x}"))
                .collect::<Vec<_>>();
            format!("{body} {} tail", tag.join(" "))
        })
        .collect()
}

fn bench_shingle_pairwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("shingle_jaccard");
    for size_bytes in [1024usize, 100 * 1024, 1024 * 1024] {
        let a = prose(size_bytes);
        let b = prose(size_bytes); // content-equal copy: the full-intersection shape
        group.throughput(Throughput::Bytes(2 * size_bytes as u64));
        for width in [3usize, 8] {
            group.bench_with_input(
                BenchmarkId::new(format!("w{width}"), size_bytes),
                &(a.clone(), b.clone()),
                |bench, (a, b)| {
                    bench.iter(|| {
                        near_dup_impl::shingle_jaccard(black_box(a), black_box(b), black_box(width))
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_dedup_pair_sweep(c: &mut Criterion) {
    // The simhash method's pair sweep: one popcount per pair, so the
    // ladder reaches the documented 10k-document ceiling.
    let mut group = c.benchmark_group("dedup_near_dup_simhash");
    group.sample_size(10);
    for n in [100usize, 1_000, 10_000] {
        let corpus = distinct_corpus(n, 1024);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("docs", n), &corpus, |bench, corpus| {
            bench.iter(|| {
                near_dup_impl::dedup_near_dup(
                    black_box(corpus),
                    black_box(0.9),
                    black_box(near_dup_impl::DedupMethod::SimHash),
                )
            })
        });
    }
    group.finish();

    // The set-intersection and signature-zip methods: the same sweep at
    // the sizes where one iteration stays seconds-scale (see the module
    // doc for why 10k is simhash-only).
    let mut group = c.benchmark_group("dedup_near_dup_sets");
    group.sample_size(10);
    for (method_name, method) in [
        ("shingle", near_dup_impl::DedupMethod::Shingle),
        ("minhash", near_dup_impl::DedupMethod::MinHash),
    ] {
        for n in [100usize, 1_000] {
            let corpus = distinct_corpus(n, 1024);
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(
                BenchmarkId::new(method_name, n),
                &corpus,
                |bench, corpus| {
                    bench.iter(|| {
                        near_dup_impl::dedup_near_dup(black_box(corpus), black_box(0.9), method)
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(near_dup, bench_shingle_pairwise, bench_dedup_pair_sweep);
criterion_main!(near_dup);
