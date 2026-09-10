//! Criterion benches for the chunking family: `chunk_impl`'s two chunk
//! shapes, `chunk_text` (boundary-aware, codepoint offsets, feeds
//! embedding/RAG pipelines) and `chunk_cdc` (content-defined, byte
//! offsets, feeds `merkle_root`/`merkle_diff`'s dedup/incremental-sync
//! use), plus the document-scale regression lane for
//! `chunk_hierarchical` and the unit-count chunkers (`chunk_by_words`/
//! `chunk_by_sentences`/`chunk_by_paragraphs`/`chunk_by_lines`), the
//! cells issue #22 measured, joined by `chunk_by_lines`, the family's
//! newest spelling, for the
//! same story: a per-call cost that used to be dominated by
//! unconditional per-codepoint structures (a whole-text `Vec<char>` plus
//! a `HashSet<usize>` of every grapheme boundary) and is now the
//! segmentation walks themselves. The `chunk_by_segment` group also
//! carries issue #30's line/paragraph-scan density lane: the same two
//! scanners over a realistic-density log corpus and a break-soup
//! corpus, the two regimes `line_bounds`/`paragraph_bounds`'
//! density-guarded memchr hop trades between (the bench cells' own
//! comments carry the numbers). The first two
//! groups bench together because the module they live in
//! (`src/chunk_impl.rs`) frames them as the two halves of one chunking
//! story, not because they share an engine; see that file's own module
//! docs for why one is codepoint-based and the other byte-based and
//! neither converts.
//!
//! Corpus: the shared prose recipe (`benches/common/mod.rs`).
//! `chunk_cdc`'s min/avg/max sizes are the crate's own shipped defaults
//! (4096/16384/65534: lifted from `fastcdc`'s doc examples, not invented;
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
// (normalize.rs/bytes.rs/text.rs bench them), so within this bench's
// compilation the shared module's builders are dead (allowed here only;
// the expectation is deliberate and self-retiring). Same pattern as
// benches/search.rs and benches/diff.rs.
mod common;

use common::prose;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
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

/// The #30 line/paragraph-scan density lane's realistic corpus: a
/// synthetic syslog shape, one 81-byte content line (80 payload bytes +
/// `'\n'`) times three, then the blank line that closes the paragraph,
/// the "one break per ~80 bytes" density the issue's per-segment
/// prototype measured 2.6–2.9× on (17.4 → 6.1 ms on a 21 MB log), and a
/// qualifying 2-unit paragraph break every third line (~244 bytes) so
/// the same corpus drives `chunk_by_paragraphs` too. Local to this
/// bench, the `benches/text.rs` precedent for recipes no other bench
/// shares: the shared three (`prose`/`decomposed`/`crlf`) stay in
/// `benches/common/mod.rs` where the corpus-parity test pins them, and
/// neither shape here has a Python-side twin to stay byte-identical to.
/// Same quantization idiom as the common module's `repeat_to` (UTF-8
/// byte-length floor division with a 1-unit floor).
fn log_lines(target_bytes: usize) -> String {
    const LOG_LINE: &str =
        "2026-09-09T12:34:56.789Z INFO worker.12 heartbeat ok latency_ms=41 queue_depth=0\n";
    let unit = format!("{}{}", LOG_LINE.repeat(3), "\n");
    unit.repeat((target_bytes / unit.len()).max(1))
}

/// The density lane's degenerate corpus: `"x\r\n\r\n"` repeated, one
/// content codepoint then a two-unit CRLF run, a break unit every 2.5
/// bytes, the "break soup" density where the issue measured the naive
/// memchr-per-segment loop 2.2× slower than the per-char decoder it
/// replaced (one memchr call and its setup per 1–2 scanned bytes), i.e.
/// the no-regression pin for the scanners' density guard. One corpus
/// serves both scanners: `chunk_by_lines` sees a content line every 5
/// bytes (the blank line inside each run is dropped by the real-line
/// filter), `chunk_by_paragraphs` a one-codepoint paragraph every 5
/// bytes (every run qualifies).
fn break_soup(target_bytes: usize) -> String {
    let unit = "x\r\n\r\n";
    unit.repeat((target_bytes / unit.len()).max(1))
}

/// The #22 document-scale lane: `chunk_hierarchical` at both hierarchy
/// kinds plus the unit-count chunkers, the merge-based pair
/// (`by_words`/`by_sentences`) at the issue's own cell shapes,
/// `by_paragraphs` at the wall-time tool's own row, and the merge-free
/// `by_lines` sibling at a corpus-derived one (no issue cell exists for
/// it to mirror; the cell's own comment derives the budget).
/// The `never-match` cells (a custom separator list that
/// matches nothing, whole-document budget) isolate the per-call machinery
/// (one count pass, one scan pass, no grapheme structure); the
/// `default_2000` cells are the segmentation walks the function exists
/// to provide; `by_words`/`by_sentences` are the unit-count spellings that
/// carry the same boundary index, `by_paragraphs` and `by_lines` the
/// siblings that carry none (their newline-run breaks are structurally
/// grapheme-safe, no merge step). 12 MiB cells
/// drop the sample count (the `chunk_cdc` 100 MiB precedent): ~300 ms
/// per iteration does not need criterion's default 100 samples to hold
/// a stable line. The `_log`/`_soup` cells (same budgets, the two
/// density-lane corpora above) are #30's: realistic density is where the
/// scanners' memchr hop is supposed to win, soup is where a naive
/// per-segment memchr loop measurably lost.
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
                chunk_hierarchical_impl::chunk_hierarchical(
                    black_box(q),
                    budget,
                    Some(&[Some("xyz")]),
                    0,
                )
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
    // The wall-time tool's own "by-paragraphs 5" row, mirrored here so
    // the criterion group keeps a paragraph line (no issue-#22 cell to
    // mirror either; the budget is the tool's, keeping the criterion
    // and wall-time tables comparable): the shared prose corpus is one
    // paragraph per 666-byte unit (18,893 paragraphs in this 12 MiB
    // text), so 5 per chunk keeps the windowing meaningful (3,779
    // chunks; a 200-paragraph window would collapse the same text
    // to 95).
    group.bench_with_input(
        BenchmarkId::new("by_paragraphs_5", format!("{}B", text.len())),
        &text,
        |bench, text| {
            bench.iter(|| chunk_by_segment_impl::chunk_by_paragraphs(black_box(text), 5, 0))
        },
    );
    // No issue-#22 cell to mirror (words/sentences above have those),
    // and the shared prose corpus is paragraph-shaped (one content
    // line per 666-byte unit, 18,893 content lines in this 12 MiB
    // text), so the budget is corpus-derived: 50 lines per chunk keeps
    // the windowing meaningful (378 chunks; a 2000-line window would
    // collapse the same text to 10).
    group.bench_with_input(
        BenchmarkId::new("by_lines_50", format!("{}B", text.len())),
        &text,
        |bench, text| bench.iter(|| chunk_by_segment_impl::chunk_by_lines(black_box(text), 50, 0)),
    );
    // #30's density lane, same budgets as the prose cells so density is
    // the only variable: the log corpus (one content line per 81 bytes, a
    // paragraph break every third line) is the realistic shape the
    // scanners' memchr hop exists for, the soup corpus (a break unit per
    // 2.5 bytes) the degenerate shape where the naive per-segment memchr
    // loop measured 2.2× slower than the per-char decoder, the
    // no-regression pin for the density guard.
    let log = log_lines(12 * 1024 * 1024);
    group.throughput(Throughput::Bytes(log.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("by_lines_50_log", format!("{}B", log.len())),
        &log,
        |bench, log| bench.iter(|| chunk_by_segment_impl::chunk_by_lines(black_box(log), 50, 0)),
    );
    group.bench_with_input(
        BenchmarkId::new("by_paragraphs_10_log", format!("{}B", log.len())),
        &log,
        |bench, log| {
            bench.iter(|| chunk_by_segment_impl::chunk_by_paragraphs(black_box(log), 10, 0))
        },
    );
    let soup = break_soup(12 * 1024 * 1024);
    group.throughput(Throughput::Bytes(soup.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("by_lines_50_soup", format!("{}B", soup.len())),
        &soup,
        |bench, soup| bench.iter(|| chunk_by_segment_impl::chunk_by_lines(black_box(soup), 50, 0)),
    );
    group.bench_with_input(
        BenchmarkId::new("by_paragraphs_10_soup", format!("{}B", soup.len())),
        &soup,
        |bench, soup| {
            bench.iter(|| chunk_by_segment_impl::chunk_by_paragraphs(black_box(soup), 10, 0))
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
