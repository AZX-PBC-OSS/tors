//! Criterion benches for tors's two native paths: `normalize_impl::normalize` and
//! `finalize_impl::finalize` (normalize + sha256), over the same deterministic corpus
//! recipes as the Python test suite (`tests/reference.py` — the deterministic
//! prose-corpus recipe): plain prose at 1 KiB / 1 MiB / 12 MiB / 100 MiB, plus the
//! decomposed-accent and CRLF variants at 12 MiB. The shared recipes live once in
//! `benches/common/mod.rs` (criterion's shared-module pattern) and are pinned
//! byte-identical to `tests/reference.py` by `tests/test_bench_corpus_parity.py`.
//! These guard tors against their own regressions across versions — the
//! cross-implementation wall-time claim (tors vs the pure-Python pipeline) is owned by
//! `tests/test_performance.py` instead, because only the Python side can run the
//! reference.
//!
//! Run locally with `cargo bench --no-default-features --bench normalize` — the
//! `--no-default-features` is required because `extension-module` deliberately does not
//! link libpython, which a bench binary needs. CI only compiles it
//! (`cargo bench --no-run`, equally with `--no-default-features`). Sample sizes are
//! criterion's defaults.
//!
//! Results are recorded in the README (performance table) from a local run on the
//! shared dev box (load disclosed there, incl. one load-9 cell); re-run quiet before
//! quoting release numbers.

mod common;

use common::{crlf, decomposed, prose};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use tors::finalize_impl;
use tors::normalize_impl;

fn bench_both_paths(c: &mut Criterion, kind: &str, corpus: &str) {
    let mut group = c.benchmark_group(kind.to_string());
    group.throughput(Throughput::Bytes(corpus.len() as u64));
    let size_label = format!("{}B", corpus.len());
    group.bench_with_input(
        BenchmarkId::new("normalize", &size_label),
        corpus,
        |b, text| b.iter(|| normalize_impl::normalize(black_box(text))),
    );
    group.bench_with_input(
        BenchmarkId::new("finalize", &size_label),
        corpus,
        |b, text| b.iter(|| finalize_impl::finalize(black_box(text))),
    );
    group.finish();
}

fn bench_tors(c: &mut Criterion) {
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024, 100 * 1024 * 1024] {
        bench_both_paths(c, "prose", &prose(target_bytes));
    }
    bench_both_paths(c, "decomposed", &decomposed(12 * 1024 * 1024));
    bench_both_paths(c, "crlf", &crlf(12 * 1024 * 1024));
    bench_identity_path(c);
}

/// The v0.4 identity path over ALREADY-NORMALIZED prose (the pipeline's own
/// output — `normalize_cow` of the prose corpus, the same bytes
/// `reference_normalize` produces on the Python side, pinned equal by the
/// suite): `normalize_cow` / `finalize_checked` short-circuit to the identity
/// probe (+ SHA-256 over the borrowed input, detached, for finalize) and
/// return the input borrowed. These cells measure that zero-allocation lane;
/// the `normalize`/`finalize` cells above measure the allocating scan over
/// scan-dirty corpora. The quick-check skip also shows up ABOVE: the prose
/// and crlf `normalize`/`finalize` cells (quick-check-Yes corpora) no longer
/// pay the NFC collect — measured ~127ms -> ~40ms at 12 MiB prose.
fn bench_identity_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("clean");
    for target_bytes in [12 * 1024 * 1024, 100 * 1024 * 1024] {
        let corpus = normalize_impl::normalize_cow(&prose(target_bytes)).into_owned();
        group.throughput(Throughput::Bytes(corpus.len() as u64));
        let label = format!("{}B", corpus.len());
        group.bench_with_input(
            BenchmarkId::new("normalize_cow", &label),
            &corpus,
            |b, text| b.iter(|| normalize_impl::normalize_cow(black_box(text))),
        );
        group.bench_with_input(
            BenchmarkId::new("finalize_checked", &label),
            &corpus,
            |b, text| b.iter(|| finalize_impl::finalize_checked(black_box(text))),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_tors);
criterion_main!(benches);
