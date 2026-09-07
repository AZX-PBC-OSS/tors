//! Criterion benches for the retrieval cluster: `tfidf_impl::tf_idf` and
//! `bm25_impl::bm25_rank`. Both take a `&[&str]` corpus already split into
//! documents (the split itself is the caller's job — `chunk_by_paragraphs`/
//! `chunk_by_sentences` if the caller wants tors to do it, out of scope for
//! this bench), so the corpus here is a fixed set of paragraph-sized prose
//! documents, not one giant blob.
//!
//! Each function is benched bare (defaults: no accent-stripping, no
//! stemming, no lemma substitution) and with every normalization knob
//! enabled, so the marginal cost of `strip_accents`/`stemmer`/`lemma_dict`
//! is visible on its own line rather than folded into one number.
//!
//! Run locally with `cargo bench --no-default-features --bench retrieval`.
//! CI only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`).

#[expect(dead_code)] // `decomposed` and `crlf` have no retrieval-bench cell
// (normalize.rs/bytes.rs/text.rs bench them), so within THIS bench's
// compilation the shared module's builders are dead — allowed here only;
// the expectation is deliberate and self-retiring. Same pattern as
// benches/search.rs and benches/integrity.rs.
mod common;

use std::collections::HashMap;

use common::PROSE_SENTENCE;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use rust_stemmers::{Algorithm, Stemmer};
use tors::bm25_impl;
use tors::tfidf_impl;

/// `doc_count` distinct paragraph-sized documents — distinct, not
/// repetitions of one string, so document frequency actually varies
/// (a corpus of one repeated document makes every term's `df` degenerate
/// to either 1 or `N`, which is not the realistic shape a corpus scan
/// exercises).
fn corpus(doc_count: usize) -> Vec<String> {
    (0..doc_count)
        .map(|i| format!("Document {i}. {}", PROSE_SENTENCE.repeat(3)))
        .collect()
}

fn corpus_refs(docs: &[String]) -> Vec<&str> {
    docs.iter().map(String::as_str).collect()
}

fn english_stemmer() -> Stemmer {
    Stemmer::create(Algorithm::English)
}

/// A small lemma map covering a few of the prose corpus's own inflected
/// forms, so the `lemma_dict` path actually substitutes something rather
/// than paying HashMap-lookup cost for zero hits.
fn sample_lemma_dict() -> HashMap<String, String> {
    HashMap::from([
        ("adjusted".to_string(), "adjust".to_string()),
        ("changed".to_string(), "change".to_string()),
        ("outages".to_string(), "outage".to_string()),
        ("windows".to_string(), "window".to_string()),
    ])
}

fn bench_tf_idf(c: &mut Criterion) {
    let mut group = c.benchmark_group("tf_idf");
    let stemmer = english_stemmer();
    let lemma_dict = sample_lemma_dict();
    for doc_count in [10usize, 100, 500] {
        let docs = corpus(doc_count);
        let refs = corpus_refs(&docs);
        group.throughput(Throughput::Elements(doc_count as u64));

        group.bench_with_input(BenchmarkId::new("bare", doc_count), &refs, |bench, refs| {
            bench.iter(|| tfidf_impl::tf_idf(black_box(refs), false, None, None))
        });
        group.bench_with_input(
            BenchmarkId::new("strip_accents", doc_count),
            &refs,
            |bench, refs| bench.iter(|| tfidf_impl::tf_idf(black_box(refs), true, None, None)),
        );
        group.bench_with_input(
            BenchmarkId::new("stemmer", doc_count),
            &refs,
            |bench, refs| {
                bench.iter(|| tfidf_impl::tf_idf(black_box(refs), false, Some(&stemmer), None))
            },
        );
        group.bench_with_input(
            BenchmarkId::new("lemma_dict", doc_count),
            &refs,
            |bench, refs| {
                bench.iter(|| tfidf_impl::tf_idf(black_box(refs), false, None, Some(&lemma_dict)))
            },
        );
    }
    group.finish();
}

fn bench_bm25_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25_rank");
    let stemmer = english_stemmer();
    let lemma_dict = sample_lemma_dict();
    let query = "quarterly oil sample outages";
    for doc_count in [10usize, 100, 500] {
        let docs = corpus(doc_count);
        let refs = corpus_refs(&docs);
        group.throughput(Throughput::Elements(doc_count as u64));

        group.bench_with_input(BenchmarkId::new("bare", doc_count), &refs, |bench, refs| {
            bench.iter(|| {
                bm25_impl::bm25_rank(
                    black_box(query),
                    black_box(refs),
                    1.5,
                    0.75,
                    false,
                    None,
                    None,
                )
            })
        });
        group.bench_with_input(
            BenchmarkId::new("strip_accents", doc_count),
            &refs,
            |bench, refs| {
                bench.iter(|| {
                    bm25_impl::bm25_rank(
                        black_box(query),
                        black_box(refs),
                        1.5,
                        0.75,
                        true,
                        None,
                        None,
                    )
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("stemmer", doc_count),
            &refs,
            |bench, refs| {
                bench.iter(|| {
                    bm25_impl::bm25_rank(
                        black_box(query),
                        black_box(refs),
                        1.5,
                        0.75,
                        false,
                        Some(&stemmer),
                        None,
                    )
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("lemma_dict", doc_count),
            &refs,
            |bench, refs| {
                bench.iter(|| {
                    bm25_impl::bm25_rank(
                        black_box(query),
                        black_box(refs),
                        1.5,
                        0.75,
                        false,
                        None,
                        Some(&lemma_dict),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_tf_idf, bench_bm25_rank);
criterion_main!(benches);
