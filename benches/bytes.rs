//! Criterion benches for the v0.2 bytes-in paths: `decode_impl::decode_strict`
//! (the strict UTF-8 scan — on these valid corpora it is the same work
//! `decode_replace`'s lossy scan does), `finalize_impl::finalize_utf8_strict`
//! (decode + normalize + sha256 in one pass), and `b64_impl::encode`. Corpora are
//! the same deterministic recipes as `benches/normalize.rs` and the Python suite
//! (`tests/reference.py`) rendered to UTF-8 bytes — the shared recipes live once
//! in `benches/common/mod.rs` and the identity (including this rendering) is
//! enforced by `tests/test_bench_corpus_parity.py`. Ladder: prose at
//! 1 KiB / 1 MiB / 12 MiB / 100 MiB, plus the decomposed-accent and CRLF variants
//! at 12 MiB.
//!
//! These guard tors against its own regressions across versions — the
//! cross-implementation wall-time claims are owned by tests/test_performance.py
//! (only the Python side can run the stdlib/reference), and the GIL-release
//! claims by tests/test_gil_release.py.
//!
//! Run locally with `cargo bench --no-default-features --bench bytes` — the
//! `--no-default-features` is required because `extension-module` deliberately
//! does not link libpython, which a bench binary needs. CI only compiles it
//! (`cargo bench --no-run`, equally with `--no-default-features`). Sample sizes
//! are criterion's defaults.
//!
//! Results are recorded in the README (performance table) from a local run on
//! the shared dev box with load disclosed; re-run quiet before quoting release
//! numbers.

mod common;

use common::{crlf, decomposed, prose};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::b64_impl;
use tors::decode_impl;
use tors::finalize_impl;

fn bench_bytes_paths(c: &mut Criterion, kind: &str, corpus: &str) {
    // The bytes-in corpora are the str corpora rendered to UTF-8 — exactly
    // `reference.corpus_utf8` on the Python side (pinned by the corpus test).
    let bytes = corpus.as_bytes();
    let mut group = c.benchmark_group(kind.to_string());
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    let size_label = format!("{}B", bytes.len());
    group.bench_with_input(
        BenchmarkId::new("decode_utf8", &size_label),
        bytes,
        |b, raw| {
            b.iter(|| {
                decode_impl::decode_strict(black_box(raw)).expect("bench corpus is valid UTF-8")
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("finalize_utf8", &size_label),
        bytes,
        |b, raw| {
            b.iter(|| {
                finalize_impl::finalize_utf8_strict(black_box(raw))
                    .expect("bench corpus is valid UTF-8")
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("b64_encode_bytes", &size_label),
        bytes,
        |b, raw| b.iter(|| b64_impl::encode(black_box(raw))),
    );
    group.finish();
}

fn bench_tors_bytes(c: &mut Criterion) {
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024, 100 * 1024 * 1024] {
        bench_bytes_paths(c, "prose", &prose(target_bytes));
    }
    bench_bytes_paths(c, "decomposed", &decomposed(12 * 1024 * 1024));
    bench_bytes_paths(c, "crlf", &crlf(12 * 1024 * 1024));
}

criterion_group!(benches, bench_tors_bytes);
criterion_main!(benches);
