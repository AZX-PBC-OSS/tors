//! Criterion benches for the chunking family: `chunk_impl`'s two chunk
//! shapes — `chunk_text` (boundary-aware, codepoint offsets, feeds
//! embedding/RAG pipelines) and `chunk_cdc` (content-defined, byte
//! offsets, feeds `merkle_root`/`merkle_diff`'s dedup/incremental-sync
//! use) — plus the document-scale regression lane for
//! `chunk_hierarchical` and the unit-count chunkers (`chunk_by_words`/
//! `chunk_by_sentences`), the cells issue #22 measured: a per-call cost
//! that used to be dominated by unconditional per-codepoint structures
//! (a whole-text `Vec<char>` plus a `HashSet<usize>` of every grapheme
//! boundary) and is now the segmentation walks themselves. The first two
//! groups bench together because the module they live in
//! (`src/chunk_impl.rs`) frames them as the two halves of one chunking
//! story, not because they share an engine — see that file's own module
//! docs for why one is codepoint-based and the other byte-based and
//! neither converts.
//!
//! Corpus: the shared prose recipe (`benches/common/mod.rs`).
//! `chunk_cdc`'s min/avg/max sizes are the crate's own shipped defaults
//! (4096/16384/65534 — lifted from `fastcdc`'s doc examples, not invented;
//! see `chunk_impl.rs`), so the bench measures the shape a caller gets
//! with no tuning, the realistic case. The #22 lane's sizes (1/12 MiB)
//! and budgets (2000 chars, 200 words, 10 sentences) mirror the issue's
//! own cells so the criterion numbers stay comparable with the
//! `tools/bench_chunking.py` wall-time table.
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
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use tors::chunk_by_segment_impl;
use tors::chunk_hierarchical_impl;
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

/// The #22 document-scale lane: `chunk_hierarchical` at both hierarchy
/// kinds plus the two merge-based unit-count chunkers, at the issue's own
/// cell shapes. The `never-match` cells (a custom separator list that
/// matches nothing, whole-document budget) isolate the per-call machinery
/// — one count pass, one scan pass, no grapheme structure; the
/// `default_2000` cells are the segmentation walks the function exists to
/// provide; `by_words`/`by_sentences` are the unit-count spellings that
/// carry the same boundary index. 12 MiB cells drop the sample count (the
/// `chunk_cdc` 100 MiB precedent): ~300 ms per iteration does not need
/// criterion's default 100 samples to hold a stable line.
fn bench_chunk_hierarchical(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_hierarchical");
    for target_bytes in [1024 * 1024, 12 * 1024 * 1024] {
        let text = prose(target_bytes);
        group.throughput(Throughput::Bytes(text.len() as u64));
        if target_bytes == 12 * 1024 * 1024 {
            group.sample_size(10);
        }
        group.bench_with_input(
            BenchmarkId::new("default_2000", format!("{}B", text.len())),
            &text,
            |bench, text| {
                bench.iter(|| {
                    chunk_hierarchical_impl::chunk_hierarchical(black_box(text), 2000, None, 0)
                })
            },
        );
    }
    // The degenerate single-character run, a never-matching custom
    // separator, and a whole-document budget: pure per-call machinery.
    let q = "q".repeat(12 * 1024 * 1024);
    group.throughput(Throughput::Bytes(q.len() as u64));
    group.sample_size(20);
    group.bench_with_input(
        BenchmarkId::new("custom_never_match", format!("{}B", q.len())),
        &q,
        |bench, q| {
            let budget = 12 * 1024 * 1024;
            bench.iter(|| {
                chunk_hierarchical_impl::chunk_hierarchical(black_box(q), budget, Some(&["xyz"]), 0)
            })
        },
    );
    group.finish();

    let mut group = c.benchmark_group("chunk_by_segment");
    let text = prose(12 * 1024 * 1024);
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("by_words_200", format!("{}B", text.len())),
        &text,
        |bench, text| bench.iter(|| chunk_by_segment_impl::chunk_by_words(black_box(text), 200, 0)),
    );
    group.bench_with_input(
        BenchmarkId::new("by_sentences_10", format!("{}B", text.len())),
        &text,
        |bench, text| {
            bench.iter(|| chunk_by_segment_impl::chunk_by_sentences(black_box(text), 10, 0))
        },
    );
    group.finish();
}

criterion_group!(
    benches,
    bench_chunk_text,
    bench_chunk_cdc,
    bench_chunk_hierarchical
);
criterion_main!(benches);
