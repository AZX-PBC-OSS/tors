//! Criterion benches for the random-generation family (`random_impl`): the
//! uuid4 full path against the uuid crate's own `new_v4()` constructor, and
//! the three token spellings (hex / b62 / b64url) over a size ladder.
//!
//! These are the only benches in the tree with no corpus at all: the
//! functions generate their own bytes (the input IS the output size), so
//! there is nothing to share with `benches/common/mod.rs` and nothing for
//! `tests/test_bench_corpus_parity.py` to pin — the throughput numbers are
//! bytes-drawn (hex/b64url) or characters-emitted (b62) per call, labeled
//! per size.
//!
//! The uuid4 pair is the comparison the family's design turns on:
//! `tors_osrng_per_call` is the full shipped path — a FRESH `OsRng` fill
//! per call (one getrandom syscall, no process or thread RNG state, hence
//! fork-safe) plus the builder and the canonical formatting — against
//! `uuid_crate_new_v4`, the uuid crate's own `Uuid::new_v4()` spelling.
//! Note what that comparator actually is in uuid 1.26 (verified in its
//! source, `src/rng.rs`): the `v4` feature's `rng` backend is ALSO
//! getrandom-per-call — not a thread-cached engine (that is the separate
//! opt-in `fast-rng` feature, the stateful shape the family's secrets
//! contract declines). The pair therefore measures the wrapper tax of
//! tors's fill+builder+format path over the crate's own equivalent
//! per-call-OS-entropy constructor, and the recorded result is ~parity
//! (~1.06 vs ~1.05 us on the dev box): zero tax for the GIL release and
//! the pyo3 marshalling being the only additions.
//!
//! These guard tors against its own regressions across versions — the
//! cross-implementation wall-time claims are owned by
//! tests/test_performance.py (only the Python side can run the stdlib),
//! and the GIL-release claims by tests/test_gil_release.py.
//!
//! Run locally with `cargo bench --no-default-features --bench random` —
//! the `--no-default-features` is required because `extension-module`
//! deliberately does not link libpython, which a bench binary needs. CI
//! only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`). Sample sizes are criterion's defaults.
//!
//! Results are recorded in docs/performance.md from a local run on the
//! dev box with load disclosed; re-run quiet before quoting release
//! numbers.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::random_impl;

/// The size ladder: token/key sizes real callers ask for (a 16-byte API key,
/// 32-token, 256-byte session id, 1 KiB batch, 64 KiB bulk), plus the 22-char
/// b62 id length the wall cells use.
const BYTE_LADDER: [usize; 5] = [16, 64, 256, 1024, 64 * 1024];
const B62_LADDER: [usize; 5] = [22, 64, 256, 1024, 64 * 1024];

fn bench_uuid_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("uuid4");
    group.bench_function("tors_osrng_per_call", |b| {
        b.iter(|| black_box(random_impl::uuid4(None).expect("os entropy")))
    });
    group.bench_function("uuid_crate_new_v4", |b| {
        b.iter(|| black_box(uuid::Uuid::new_v4().to_string()))
    });
    group.finish();

    let mut group = c.benchmark_group("uuid7");
    group.bench_function("tors_osrng_per_call", |b| {
        b.iter(|| black_box(random_impl::uuid7().expect("os entropy")))
    });
    group.finish();
}

fn bench_token_ladders(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_hex");
    for &n_bytes in &BYTE_LADDER {
        group.throughput(Throughput::Bytes(n_bytes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_bytes), &n_bytes, |b, &n| {
            b.iter(|| black_box(random_impl::random_hex(n, None).expect("os entropy")))
        });
    }
    group.finish();

    let mut group = c.benchmark_group("random_b64url");
    for &n_bytes in &BYTE_LADDER {
        group.throughput(Throughput::Bytes(n_bytes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_bytes), &n_bytes, |b, &n| {
            b.iter(|| black_box(random_impl::random_b64url(n, false, None).expect("os entropy")))
        });
    }
    group.finish();

    let mut group = c.benchmark_group("random_b62");
    for &length in &B62_LADDER {
        group.throughput(Throughput::Bytes(length as u64));
        group.bench_with_input(BenchmarkId::from_parameter(length), &length, |b, &len| {
            b.iter(|| black_box(random_impl::random_b62(len, None).expect("os entropy")))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_uuid_paths, bench_token_ladders);
criterion_main!(benches);
