//! Criterion benches for `lsh_impl::lsh_candidates`: the signature-count
//! ladder (1k / 10k signatures, the corpus-scale shapes the API docs
//! name) crossed with the band axis `b {8, 16, 32}` at a fixed
//! `num_perm` of 128 (rows 16 / 8 / 4 -- the shape a caller tunes
//! through the S-curve, and the knob that decides how many bucket
//! tables the sweep builds). Signatures are random full-range u64s:
//! no bucket sharing, so the bench measures the hashing sweep and the
//! per-band bucket tables, the cost class the scaling pins in
//! `tests/test_scaling_pins.py` hold linear.
//!
//! Run locally with `cargo bench --no-default-features --bench lsh`.
//! CI only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::lsh_impl;

/// The repo's deterministic u64 LCG (the same Knuth-style constants the
/// fuzz targets and `tests/reference.py`'s corpus builders use): a
/// full-range random signature, distinct from every other.
fn lcg_u64(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn signatures(n: usize, num_perm: usize) -> Vec<Vec<u64>> {
    let mut state = 0x2026_0924u64;
    (0..n)
        .map(|_| (0..num_perm).map(|_| lcg_u64(&mut state)).collect())
        .collect()
}

fn bench_lsh_candidates(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_candidates");
    for n in [1_000usize, 10_000] {
        let sigs = signatures(n, 128);
        // Throughput = the signature rows the sweep hashes (n x num_perm),
        // the input the pass is linear in.
        group.throughput(Throughput::Elements((n * 128) as u64));
        for bands in [8usize, 16, 32] {
            let rows = 128 / bands;
            group.bench_with_input(
                BenchmarkId::new(format!("b{bands}r{rows}"), n),
                &sigs,
                |bench, sigs| {
                    bench.iter(|| {
                        lsh_impl::lsh_candidates(black_box(sigs), black_box(bands), black_box(rows))
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(lsh, bench_lsh_candidates);
criterion_main!(lsh);
