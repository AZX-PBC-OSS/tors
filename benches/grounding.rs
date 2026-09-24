//! Criterion benches for the grounding core: `grounding_impl::highlight`,
//! the snippet-provenance pass the search surface calls per retrieved chunk.
//! The shapes are the primary consumer's own: a 60-token query against
//! chunks of 500 / 2k / 10k tokens (the asymmetric case the module docs
//! justify the algorithm against), in a Latin-script and a CJK variant
//! (CJK tokenizes per character, so the same nominal token count is ~3x
//! the characters and every token is a single codepoint), plus a
//! no-overlap cell (zero anchors — the cheapest path, pinned so an
//! accidental regression to whole-text scoring shows up) and a dense cell
//! (every token matches — the adversarial-anchor ceiling the
//! MAX_CANDIDATES cap bounds).
//!
//! Run locally with `cargo bench --no-default-features --bench grounding`.
//! CI only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`); the wall-clock gate a CI leg CAN run is
//! `tests/test_grounding_performance.py` (machine-speed-immune ratios).

#[expect(dead_code)] // `repeat_to`/`prose`/`decomposed`/`crlf` have no
// grounding-bench cell (normalize.rs/bytes.rs/text.rs bench them), so
// within THIS bench's compilation the shared module's other builders are
// dead — allowed here only; the expectation is deliberate and
// self-retiring. Same pattern as benches/retrieval.rs.
mod common;

use common::PROSE_SENTENCE;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tors::grounding_impl;

/// A 60-token query whose terms partly occur in the prose corpus (the
/// realistic lexical-overlap shape a retrieved chunk sees: some anchor
/// material, some not).
const LATIN_QUERY: &str = "the quarterly oil sample interval for field outages was adjusted \
 after the bushing torque specifications changed maintenance windows now close within \
 fourteen days of each outage review cycle and the transformer oil analysis report \
 lists dielectric strength moisture content and dissolved gas concentrations for every \
 sampled unit";

/// ~60 CJK characters (each its own token under the grounding tokenizer):
/// a realistic unspaced CJK query length. Punctuation segments drop out of
/// the tokenizer, so the token count is the alphanumeric characters.
const CJK_QUERY: &str = "油田の定期検査は四半期ごとに行われ、絶縁油のサンプル採取間隔と絶縁耐力の測定結果に基づいて調整される。";

/// Build a Latin chunk of roughly `tokens` whitespace-separated tokens:
/// distinct-enough sentences (numbered) so document-shape degeneracies
/// (one giant repeated string) don't mask per-window work, with the
/// query's own terms present every third sentence — the scattered-anchor
/// shape selection actually has to rank.
fn latin_chunk(tokens: usize) -> String {
    let sentence = PROSE_SENTENCE.trim_end();
    let mut out = String::with_capacity(tokens * 7);
    let mut count = 0usize;
    while out.split_whitespace().count() < tokens {
        if count.is_multiple_of(3) {
            out.push_str(sentence);
        } else {
            out.push_str(&format!(
                "Unrelated filler sentence number {count} walks on. "
            ));
        }
        count += 1;
        out.push(' ');
    }
    out
}

/// Build a CJK chunk of roughly `tokens` CJK characters (each character is
/// one token under the grounding tokenizer).
fn cjk_chunk(tokens: usize) -> String {
    let unit = "変電所の絶縁油検査は四半期ごとに行われる。";
    let filler = "その他の無関係な記述がここに入る。";
    let mut out = String::with_capacity(tokens * 3);
    let mut count = 0usize;
    while out.chars().count() < tokens {
        if count.is_multiple_of(3) {
            out.push_str(unit);
        } else {
            out.push_str(filler);
        }
        count += 1;
    }
    out
}

fn bench_grounding(c: &mut Criterion) {
    let mut group = c.benchmark_group("highlight");
    for tokens in [500usize, 2_000, 10_000] {
        let latin = latin_chunk(tokens);
        let cjk = cjk_chunk(tokens);
        group.throughput(criterion::Throughput::Bytes(latin.len() as u64));
        group.bench_with_input(BenchmarkId::new("latin", tokens), &latin, |b, text| {
            b.iter(|| grounding_impl::highlight(black_box(LATIN_QUERY), black_box(text), 3, 400))
        });
        group.throughput(criterion::Throughput::Bytes(cjk.len() as u64));
        group.bench_with_input(BenchmarkId::new("cjk", tokens), &cjk, |b, text| {
            b.iter(|| grounding_impl::highlight(black_box(CJK_QUERY), black_box(text), 3, 400))
        });
    }
    // The no-overlap cell: zero anchors, so the anchor walk is the whole
    // cost — the floor the selection machinery adds nothing to.
    let latin = latin_chunk(2_000);
    group.bench_with_input(
        BenchmarkId::new("latin_no_overlap", 2_000),
        &latin,
        |b, text| {
            b.iter(|| {
                grounding_impl::highlight(
                    black_box("zebra quantum xylophone"),
                    black_box(text),
                    3,
                    400,
                )
            })
        },
    );
    // The dense cell: the query is one repeated term, so every token is an
    // anchor — the MAX_CANDIDATES pre-rank's adversarial ceiling.
    let dense = "relevant term ".repeat(3_500);
    group.bench_with_input(BenchmarkId::new("latin_dense", 7_000), &dense, |b, text| {
        b.iter(|| grounding_impl::highlight(black_box("relevant term"), black_box(text), 3, 400))
    });
    group.finish();
}

criterion_group!(benches, bench_grounding);
criterion_main!(benches);
