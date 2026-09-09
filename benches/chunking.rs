//! Criterion benches for `chunk_impl`'s two chunk shapes — `chunk_text`
//! (boundary-aware, codepoint offsets, feeds embedding/RAG pipelines) and
//! `chunk_cdc` (content-defined, byte offsets, feeds `merkle_root`/
//! `merkle_diff`'s dedup/incremental-sync use). Benched together because
//! the module they live in (`src/chunk_impl.rs`) frames them as the two
//! halves of one chunking story, not because they share an engine — see
//! that file's own module docs for why one is codepoint-based and the
//! other byte-based and neither converts.
//!
//! Corpus: the shared prose recipe (`benches/common/mod.rs`).
//! `chunk_cdc`'s min/avg/max sizes are the crate's own shipped defaults
//! (4096/16384/65534 — lifted from `fastcdc`'s doc examples, not invented;
//! see `chunk_impl.rs`), so the bench measures the shape a caller gets
//! with no tuning, the realistic case.
//!
//! Run locally with `cargo bench --no-default-features --bench chunking`.
//! CI only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`).

#[expect(dead_code)] // `decomposed` and `crlf` have no chunking-bench cell
// (normalize.rs/bytes.rs/text.rs bench them), so within THIS bench's
// compilation the shared module's builders are dead — allowed here only;
// the expectation is deliberate and self-retiring. Same pattern as
// benches/search.rs and benches/diff.rs.
mod common;

use common::prose;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::chunk_impl;
use tors::truncate_impl::Boundary;

fn bench_chunk_text(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_text");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024] {
        let text = prose(target_bytes);
        group.throughput(Throughput::Bytes(text.len() as u64));
        // A 500-char budget: several sentences per chunk, well above any
        // single word/sentence bound in the prose corpus (no degenerate
        // one-boundary-per-chunk shape).
        group.bench_with_input(
            BenchmarkId::new("word_500", format!("{}B", text.len())),
            &text,
            |bench, text| {
                bench.iter(|| chunk_impl::chunk_text(black_box(text), 500, Boundary::Word))
            },
        );
    }
    group.finish();
}

fn bench_chunk_cdc(c: &mut Criterion) {
    // Shipped defaults (see chunk_impl.rs): min 4096, avg 16384, max 65534.
    const MIN: usize = 4096;
    const AVG: usize = 16384;
    const MAX: usize = 65534;
    let mut group = c.benchmark_group("chunk_cdc");
    for target_bytes in [1024 * 1024, 12 * 1024 * 1024, 100 * 1024 * 1024] {
        let text = prose(target_bytes);
        let bytes = text.as_bytes();
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        if target_bytes == 100 * 1024 * 1024 {
            group.sample_size(10);
        }
        group.bench_with_input(
            BenchmarkId::new("default_params", format!("{}B", bytes.len())),
            &bytes,
            |bench, bytes| {
                bench.iter(|| {
                    chunk_impl::chunk_cdc(black_box(bytes), MIN, AVG, MAX)
                        .expect("valid chunk params")
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_chunk_text, bench_chunk_cdc);
criterion_main!(benches);
