//! Criterion benches for the agent/RAG-pipeline pair: `truncate_impl::truncate_to_bounds`
//! (cut to a context-window character budget at a real word/sentence
//! boundary, composing `word_bounds`/`sentence_bounds`) and
//! `grounded_impl::is_grounded_exact`/`is_grounded_fuzzy` (is a generated
//! claim actually present in a retrieved source passage — exact substring,
//! or a bounded windowed-ratio fuzzy scan over the `similar` engine).
//!
//! `truncate_to_bounds` is benched over the prose corpus at a budget well
//! inside the corpus (so every size actually exercises the segmentation +
//! cut-search work, not the zero-cost identity return for already-short
//! input).
//!
//! `is_grounded_fuzzy` is the DoS-relevant one (bounded, but still a scan
//! over `source`): benched with a short realistic claim against sources at
//! several sizes, both grounded (claim present, early exit via the
//! threshold check) and ungrounded (claim absent, full scan) shapes.
//!
//! Run locally with `cargo bench --no-default-features --bench redaction`.
//! CI only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`).

#[expect(dead_code)] // `decomposed` and `crlf` have no redaction-bench cell
// (normalize.rs/bytes.rs/text.rs bench them), so within THIS bench's
// compilation the shared module's builders are dead — allowed here only;
// the expectation is deliberate and self-retiring. Same pattern as
// benches/search.rs and benches/diff.rs.
mod common;

use common::prose;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::grounded_impl;
use tors::truncate_impl::{self, Boundary};

fn bench_truncate_to_bounds(c: &mut Criterion) {
    let mut group = c.benchmark_group("truncate_to_bounds");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024] {
        let text = prose(target_bytes);
        let max_chars = (text.chars().count() / 4).max(1);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("word_quarter_budget", format!("{}B", text.len())),
            &text,
            |bench, text| {
                bench.iter(|| {
                    truncate_impl::truncate_to_bounds(black_box(text), max_chars, Boundary::Word)
                })
            },
        );
    }
    group.finish();
}

// A realistic short claim, present verbatim near the middle of the prose
// corpus (drawn from PROSE_SENTENCE in benches/common/mod.rs) — the
// grounded shape; a same-length claim built from characters the corpus
// never contains is the ungrounded shape (forces the full scan).
const GROUNDED_CLAIM: &str = "the bushing torque specifications changed";
const UNGROUNDED_CLAIM: &str = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";

fn bench_is_grounded_exact(c: &mut Criterion) {
    let mut group = c.benchmark_group("is_grounded_exact");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024] {
        let source = prose(target_bytes);
        group.throughput(Throughput::Bytes(source.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("grounded", format!("{}B", source.len())),
            &source,
            |bench, source| {
                bench.iter(|| grounded_impl::is_grounded_exact(black_box(GROUNDED_CLAIM), source))
            },
        );
    }
    group.finish();
}

fn bench_is_grounded_fuzzy(c: &mut Criterion) {
    let mut group = c.benchmark_group("is_grounded_fuzzy");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024] {
        let source = prose(target_bytes);
        group.throughput(Throughput::Bytes(source.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("grounded", format!("{}B", source.len())),
            &source,
            |bench, source| {
                bench.iter(|| {
                    grounded_impl::is_grounded_fuzzy(black_box(GROUNDED_CLAIM), source, 0.85, None)
                        .expect("no deadline set")
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("ungrounded", format!("{}B", source.len())),
            &source,
            |bench, source| {
                bench.iter(|| {
                    grounded_impl::is_grounded_fuzzy(
                        black_box(UNGROUNDED_CLAIM),
                        source,
                        0.85,
                        None,
                    )
                    .expect("no deadline set")
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_truncate_to_bounds,
    bench_is_grounded_exact,
    bench_is_grounded_fuzzy
);
criterion_main!(benches);
