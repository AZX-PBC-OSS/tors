//! Criterion benches for `utf8_impl::is_valid` — the SIMD UTF-8 validity scan.
//! Corpora are the same deterministic recipes as every other bench and the
//! Python suite (`tests/reference.py`) rendered to UTF-8 bytes — the shared
//! recipes live once in `benches/common/mod.rs` and the identity (including
//! this rendering) is enforced by `tests/test_bench_corpus_parity.py`.
//! Ladder: prose at 1 KiB / 1 MiB / 12 MiB / 100 MiB, the decomposed-accent
//! and CRLF variants at 12 MiB, plus an INVALID prose corpus at 12 MiB — the
//! valid bytes with the final byte replaced by 0xFF, so the scan still
//! traverses the whole corpus before failing (an invalid byte anywhere
//! earlier would let the validator stop early and measure less than the
//! full-corpus pass; end placement pins the whole-input scan cost).
//!
//! These guard tors against its own regressions across versions — the
//! cross-implementation wall-time claims are owned by the Python-side tests
//! (tests/test_utf8_is_valid.py records the absolute throughput: the stdlib
//! has no boolean validity oracle to race), and the GIL-release claims by
//! tests/test_gil_release.py.
//!
//! Run locally with `cargo bench --no-default-features --bench utf8` — the
//! `--no-default-features` is required because `extension-module`
//! deliberately does not link libpython, which a bench binary needs. CI only
//! compiles it (`cargo bench --no-run`, equally with `--no-default-features`).
//! Sample sizes are criterion's defaults.
//!
//! Results are recorded in the README (performance table) from a local run on
//! the shared dev box with load disclosed; re-run quiet before quoting
//! release numbers.

mod common;

use common::{crlf, decomposed, prose};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::utf8_impl;

fn bench_utf8_is_valid(c: &mut Criterion, kind: &str, bytes: &[u8]) {
    let mut group = c.benchmark_group(kind.to_string());
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    let size_label = format!("{}B", bytes.len());
    group.bench_with_input(
        BenchmarkId::new("utf8_is_valid", &size_label),
        bytes,
        |b, raw| b.iter(|| utf8_impl::is_valid(black_box(raw))),
    );
    group.finish();
}

fn bench_tors_utf8(c: &mut Criterion) {
    // The bytes-in corpora are the str corpora rendered to UTF-8 — exactly
    // `reference.corpus_utf8` on the Python side (pinned by the corpus test).
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024, 100 * 1024 * 1024] {
        bench_utf8_is_valid(c, "prose", prose(target_bytes).as_bytes());
    }
    bench_utf8_is_valid(c, "decomposed", decomposed(12 * 1024 * 1024).as_bytes());
    bench_utf8_is_valid(c, "crlf", crlf(12 * 1024 * 1024).as_bytes());
    // The invalid corpus (see the module docs): the 12 MiB prose bytes with
    // an invalid lead byte at the very end.
    let mut invalid = prose(12 * 1024 * 1024).into_bytes();
    let last = invalid.len() - 1;
    invalid[last] = 0xFF;
    bench_utf8_is_valid(c, "invalid-prose", &invalid);
}

criterion_group!(benches, bench_tors_utf8);
criterion_main!(benches);
