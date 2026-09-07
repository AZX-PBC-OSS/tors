//! Criterion benches for the fenced-code-block surface: `fence_impl`'s
//! `extract_code_blocks`/`strip_code_fences`/`dedent` — the hand-rolled
//! CommonMark §4.5 grammar, no Markdown-parser dependency (see the module
//! docs on why: LLM chat output is rarely deeply-nested Markdown, so a full
//! CommonMark engine's extra correctness is mostly wasted weight for this).
//!
//! Corpus: a synthetic fenced-block document built from the shared prose
//! sentence (`benches/common/mod.rs`), alternating prose paragraphs with
//! ` ```text ` fenced blocks — the realistic shape (an LLM response mixing
//! prose and code), not one giant fence or bare prose (which `extract_code_blocks`
//! would degenerate to a single-block or zero-block scan respectively,
//! under-measuring the fence-detection work).
//!
//! `dedent` is benched separately over plain indented prose (its own,
//! narrower job — the longest-common-leading-whitespace reduction — has
//! nothing to do with fence scanning).
//!
//! Run locally with `cargo bench --no-default-features --bench fence`. CI
//! only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`).

#[expect(dead_code)] // `decomposed` and `crlf` have no fence-bench cell
// (normalize.rs/bytes.rs/text.rs bench them), so within THIS bench's
// compilation the shared module's builders are dead — allowed here only;
// the expectation is deliberate and self-retiring. Same pattern as
// benches/search.rs and benches/diff.rs.
mod common;

use common::prose;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use tors::fence_impl;

/// One prose paragraph, one fenced code block — repeated to the target
/// size. The fence's content is itself prose-shaped (irrelevant to a
/// generic code-fence scanner, but keeps the corpus deterministic and
/// reuses the shared sentence rather than inventing a second recipe).
fn fenced_document(target_bytes: usize) -> String {
    let paragraph = common::prose(256);
    let block = format!("```text\n{paragraph}```\n\n");
    let unit_len = paragraph.len() + block.len();
    let mut out = String::with_capacity(target_bytes + unit_len);
    while out.len() < target_bytes {
        out.push_str(&paragraph);
        out.push_str(&block);
    }
    out
}

fn bench_extract_code_blocks(c: &mut Criterion) {
    let mut group = c.benchmark_group("extract_code_blocks");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024] {
        let text = fenced_document(target_bytes);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("mixed", format!("{}B", text.len())),
            &text,
            |bench, text| bench.iter(|| fence_impl::extract_code_blocks(black_box(text), None)),
        );
    }
    group.finish();
}

fn bench_strip_code_fences(c: &mut Criterion) {
    // strip_code_fences' own job: the whole-response-is-one-fence case —
    // benched over that shape specifically (the mixed-document corpus above
    // is NOT single-block, so it would only measure the no-op identity
    // path, not the actual unwrap-and-dedent work).
    let mut group = c.benchmark_group("strip_code_fences");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024] {
        let body = prose(target_bytes);
        let text = format!("```text\n{body}```\n");
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("single_block", format!("{}B", text.len())),
            &text,
            |bench, text| bench.iter(|| fence_impl::strip_code_fences(black_box(text))),
        );
    }
    group.finish();
}

fn bench_dedent(c: &mut Criterion) {
    let mut group = c.benchmark_group("dedent");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024] {
        // Every line indented by 4 spaces — the common-margin reduction has
        // real work to do (a flush-left corpus would take the identity
        // fast path and under-measure the scan).
        let indented: String = prose(target_bytes)
            .lines()
            .map(|line| format!("    {line}\n"))
            .collect();
        group.throughput(Throughput::Bytes(indented.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("indented", format!("{}B", indented.len())),
            &indented,
            |bench, text| bench.iter(|| fence_impl::dedent(black_box(text))),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_extract_code_blocks,
    bench_strip_code_fences,
    bench_dedent
);
criterion_main!(benches);
