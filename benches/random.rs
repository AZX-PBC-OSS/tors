//! Criterion benches for the random-generation family (`random_impl`): the
//! uuid4 full path against the uuid crate's own `new_v4()` constructor, and
//! the three token spellings (hex / b62 / b64url) over a size ladder.
//!
//! These are the only benches in the tree with no corpus at all: the
//! functions generate their own output (the argument IS the output
//! length), so there is nothing to share with `benches/common/mod.rs` and
//! nothing for `tests/test_bench_corpus_parity.py` to pin — the
//! throughput numbers are characters-emitted per call, labeled per size.
//!
//! The hex/b64url ladders are output-equivalent to the pre-length-first-
//! refactor byte ladders: the old rungs drew n bytes and emitted their
//! encoding (hex 2n chars, b64url ceil(4n/3) chars), the new engine takes
//! the output length directly, so each rung keeps the output size the old
//! one produced at the same point ([16, 64, 256, 1024, 64K] bytes ->
//! hex [32, 128, 512, 2048, 131072] chars and b64url [22, 86, 342, 1366,
//! 87382] chars). That makes the cross-refactor comparison honest — same
//! tokens, new spelling — at the cost of the engine-class change the
//! comparison records: char sampling consumes one u64 (8 stream bytes)
//! per character where byte-fill+encode consumed ~0.5-0.75, so the
//! measured delta is the price of uniform-per-character output
//! (docs/performance.md's random table records it).
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

/// The size ladders: token/key sizes real callers ask for, spelled as the
/// output lengths they are (a 32-char hex key, a 22-char b62 or b64url id,
/// 1 KiB-class and 64 KiB-class bulk) — the hex/b64url rungs are the
/// output equivalents of the old byte ladder (see the module docs).
const HEX_LADDER: [usize; 5] = [32, 128, 512, 2048, 131_072];
const B64URL_LADDER: [usize; 5] = [22, 86, 342, 1366, 87_382];
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
    for &length in &HEX_LADDER {
        group.throughput(Throughput::Bytes(length as u64));
        group.bench_with_input(BenchmarkId::from_parameter(length), &length, |b, &len| {
            b.iter(|| black_box(random_impl::random_hex(len, None).expect("os entropy")))
        });
    }
    group.finish();

    let mut group = c.benchmark_group("random_b64url");
    for &length in &B64URL_LADDER {
        group.throughput(Throughput::Bytes(length as u64));
        group.bench_with_input(BenchmarkId::from_parameter(length), &length, |b, &len| {
            b.iter(|| black_box(random_impl::random_b64url(len, None).expect("os entropy")))
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
