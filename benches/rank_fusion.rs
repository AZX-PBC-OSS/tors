//! Criterion benches for the rank-fusion family: `rank_fusion_impl::rank_fuse`
//! plus the four IR metrics. Like `benches/retrieval.rs`, these drive the
//! pure-Rust cores directly over plain data (dedup indices / relevance
//! flags): the Python-object hashing walk that precedes them is interpreter
//! work the bench cannot see, and the cores are what a Rust consumer (or the
//! detached pass of the pyo3 wrapper) actually pays for.
//!
//! `rank_fuse` is benched at small/medium/large totals over a shared
//! id space (every list draws references into one pool, the realistic
//! fusion shape, where the same documents recur across lists, unlike a
//! corpus of disjoint strings), with throughput reported in total
//! entries; the weighted spelling (per-list weights, the Elasticsearch
//! weighted-RRF extension) benches the same shapes under a mixed
//! weight vector, its cell directly comparable to the unweighted one.
//! The metrics are benched over a relevance-flag vector of the
//! same large scale: each is a linear sweep, and the interesting
//! question is the constant, so one size per metric is enough.
//!
//! Run locally with `cargo bench --no-default-features --bench rank_fusion`.
//! CI only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`).

#[expect(dead_code)] // the shared module has no other consumer in this bench
mod common;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::rank_fusion_impl::{mrr, ndcg_at_k, precision_at_k, rank_fuse, recall_at_k};
/// The fusion workload: `n_lists` ranked lists of dedup indices over a
/// shared id space of `id_space` documents; every list ranks entries
/// drawn from the same pool (stride-sampled so the votes genuinely
/// overlap), the shape fusion exists for.
fn fusion_lists(total_entries: usize, n_lists: usize, id_space: usize) -> Vec<Vec<u32>> {
    let per_list = total_entries / n_lists;
    (0..n_lists)
        .map(|j| {
            (0..per_list)
                .map(|i| ((j * 7 + i * 3) % id_space) as u32)
                .collect()
        })
        .collect()
}

fn bench_rank_fuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank_fuse");
    for total in [10_000usize, 100_000, 400_000] {
        let lists = fusion_lists(total, 5, total / 2);
        group.throughput(Throughput::Elements(total as u64));
        group.bench_with_input(
            BenchmarkId::new("total_entries", total),
            &lists,
            |bench, lists| {
                bench.iter(|| rank_fuse(black_box(lists), 60, black_box(total / 2), None))
            },
        );
    }
    group.finish();
}

/// The weighted spelling over the same workload: one weight per list
/// (the 2.0/1.0/1.0/1.0/0.5 hybrid shape), so the weighted cell's wall
/// is directly comparable to the unweighted one at the same sizes —
/// the extension must cost a multiply per vote, nothing else.
fn bench_weighted_rank_fuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank_fuse_weighted");
    for total in [10_000usize, 100_000, 400_000] {
        let lists = fusion_lists(total, 5, total / 2);
        let weights = [2.0f64, 1.0, 1.0, 1.0, 0.5];
        group.throughput(Throughput::Elements(total as u64));
        group.bench_with_input(
            BenchmarkId::new("total_entries", total),
            &lists,
            |bench, lists| {
                bench.iter(|| rank_fuse(black_box(lists), 60, black_box(total / 2), Some(&weights)))
            },
        );
    }
    group.finish();
}

/// The metrics over one large ranking: a linear membership sweep with a
/// relevance hit every third position (the reranking-scale shape), k =
/// the full length.
fn bench_metrics(c: &mut Criterion) {
    let n = 100_000usize;
    let flags: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
    let gains: Vec<f64> = flags
        .iter()
        .map(|hit| if *hit { 1.0 } else { 0.0 })
        .collect();
    let pool: Vec<f64> = (0..n / 3).map(|_| 1.0).collect();

    let mut group = c.benchmark_group("rank_metrics");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("ndcg_at_k", |bench| {
        bench.iter(|| ndcg_at_k(black_box(&gains), black_box(pool.clone()), n))
    });
    group.bench_function("mrr", |bench| bench.iter(|| mrr(black_box(&flags))));
    group.bench_function("recall_at_k", |bench| {
        bench.iter(|| recall_at_k(black_box(&flags), n / 3, n))
    });
    group.bench_function("precision_at_k", |bench| {
        bench.iter(|| precision_at_k(black_box(&flags), n))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_rank_fuse,
    bench_weighted_rank_fuse,
    bench_metrics
);
criterion_main!(benches);
