//! Criterion benches for the v0.7 search surface: `search_impl::find_patterns`
//! — the Rust core alone (automaton build, scan, byte→char offset conversion
//! with the `is_ascii` fast path, and the `Vec<PatternMatch>` fill), with no
//! pyo3 layer; the argument borrows and the O(matches) tuple marshalling are
//! measured Python-side by `tests/test_gil_release.py` (the marshalling band,
//! ~0.13 µs per match) and the wall cells in `tests/test_find_patterns.py`.
//!
//! Corpus shapes, both over the shared prose recipe in `benches/common/mod.rs`
//! (byte-identical to `tests/reference.py`'s — the corpus identity is pinned
//! by `tests/test_bench_corpus_parity.py`):
//!
//! - `sparse`: the three-word terminology-scan set that never occurs in the
//!   prose corpus — the PURE SCAN shape (0 matches at every size; the whole
//!   iteration is automaton build + scan + the `is_ascii` fast path's
//!   nothing-to-convert). The Python-side GIL cell's few-matches shape runs
//!   this pattern set over the diff near-identical pair's edited corpus
//!   instead (where `"monthly"` occurs once); the bench's job is scan
//!   throughput in isolation, so it stays on the plain corpus.
//! - `dense`: the seventeen-word prose set — every word matches once per
//!   sentence, so the corpus fills the match vector (68 matches per 666-byte
//!   unit; 1,284,724 at 12 MiB). The match-reporting shape.
//!
//! Each iteration is the WHOLE core call — automaton build included, the
//! same shape as the Python call (build is µs-scale for these pattern
//! counts, invisible next to the scan at every ladder size; the table's
//! flat sparse throughput is the evidence).
//!
//! Ladder: 256 KiB / 1 MiB / 12 MiB / 100 MiB for both shapes — the scan is
//! linear, so unlike the diff bench's superlinear shuffled shape, both
//! shapes carry the 100 MiB leg (the dense 100 MiB cell fills a ~10.7M-entry
//! match vector, ~250 MiB of `Vec`, and runs with criterion's 10-sample
//! minimum because its ~hundreds-of-ms iterations would make the default
//! 100-sample cell a minute-plus; every other cell uses default sampling).
//!
//! The v0.8 `replace_many` group: the dense set's words each mapped to the
//! `[REDACTED]` redaction token — the redaction-map shape
//! `tests/test_gil_release.py`'s dense replace cell drives at the pyo3 layer
//! — over the same prose ladder, each iteration again the WHOLE core call:
//! automaton build + scan + the splice of every match into ONE output
//! string (the no-list-shape class: the return is a single ~corpus-sized
//! string, not a match vector). Every dense word matches once per sentence,
//! so the 100 MiB leg splices ~10.7M matches into one ~95 MiB output — the
//! dense find cell's slow-iteration regime, so it takes the same 10-sample
//! reduction there (default elsewhere).
//!
//! The `replace_many_masked` group: the length-preserving spelling of the
//! same dense redaction map — the SAME DENSE_PATTERNS → `[REDACTED]` map
//! the replace_many cell drives (built from the pinned array the same way,
//! so the two maps cannot drift) plus the mask char that pads a masked
//! value out to its span's CHARACTER count (with truncation clipping the
//! token where the span is shorter — the dense set exercises both
//! branches: `specifications` and `Maintenance` pad, the fifteen shorter
//! words truncate), so the output's char count equals the input's and every
//! pre-redaction offset — `find_patterns` spans, word/sentence bounds —
//! stays valid on the redacted text: the redaction shape for logs,
//! training corpora, and PII pipelines. Same ladder, same whole-core-call
//! shape, and the same slow-iteration regime at 100 MiB (the ~10.7M masked
//! matches) — 10 samples there, default elsewhere.
//!
//! Run locally with `cargo bench --no-default-features --bench search` — the
//! `--no-default-features` is required because `extension-module`
//! deliberately does not link libpython, which a bench binary needs. CI only
//! compiles it (`cargo bench --no-run`, equally with `--no-default-features`).
//! Results are recorded in the README (performance table) from a local run on
//! the shared dev box with load disclosed.

#[expect(dead_code)] // `decomposed` and `crlf` have no search-bench cell (normalize.rs and
// bytes.rs/text.rs bench them), so within THIS bench's compilation the shared
// module's builders are dead — allowed here only; the expectation is
// deliberate and self-retiring (a future cell makes it unfulfilled, and the
// lint says so). Same pattern as benches/diff.rs.
mod common;

use common::prose;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::search_impl;

// The two pattern sets, mirroring tests/reference.py's SEARCH_SPARSE_PATTERNS
// and SEARCH_DENSE_PATTERNS — cross-checked against the reference tuples by
// tests/test_bench_corpus_parity.py, so the bench numbers and the Python-side
// GIL/wall cell numbers cross-reference on the same patterns.
const SPARSE_PATTERNS: [&str; 3] = ["monthly", "weekly", "annually"];

const DENSE_PATTERNS: [&str; 17] = [
    "quarterly",
    "oil",
    "sample",
    "interval",
    "field",
    "outages",
    "adjusted",
    "bushing",
    "torque",
    "specifications",
    "changed",
    "Maintenance",
    "windows",
    "close",
    "within",
    "fourteen",
    "days",
];

// The v0.8 redaction token — the same "[REDACTED]" tests/test_gil_release.py's
// dense replace cell maps the dense set's words to.
const REDACTION_TOKEN: &str = "[REDACTED]";

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_patterns");
    for target_bytes in [256 * 1024, 1024 * 1024, 12 * 1024 * 1024, 100 * 1024 * 1024] {
        let text = prose(target_bytes);
        group.throughput(Throughput::Bytes(text.len() as u64));
        // The pure-scan shape: nothing matches, so the iteration is build +
        // scan + the fast path's no-op conversion.
        group.bench_with_input(
            BenchmarkId::new("sparse", format!("{}B", text.len())),
            &text,
            |bench, text| {
                bench.iter(|| {
                    search_impl::find_patterns(&SPARSE_PATTERNS, black_box(text))
                        .expect("automaton build failed")
                })
            },
        );
        // The match-reporting shape: the scan fills the match vector (68
        // matches per corpus unit). The 100 MiB leg's ~10.7M-match vector
        // makes its iterations slow enough that default sampling would turn
        // the cell into minutes — 10 samples there (the diff bench's
        // shuffled-12 MiB precedent), default elsewhere.
        if target_bytes == 100 * 1024 * 1024 {
            group.sample_size(10);
        }
        group.bench_with_input(
            BenchmarkId::new("dense", format!("{}B", text.len())),
            &text,
            |bench, text| {
                bench.iter(|| {
                    search_impl::find_patterns(&DENSE_PATTERNS, black_box(text))
                        .expect("automaton build failed")
                })
            },
        );
    }
    group.finish();
}

fn bench_replace_many(c: &mut Criterion) {
    // The dense set's words each mapped to the redaction token — built from
    // the pinned DENSE_PATTERNS (not re-listed) so the map cannot drift from
    // the dense find cell's pattern set.
    let replacements: Vec<(&str, &str)> = DENSE_PATTERNS
        .iter()
        .map(|&word| (word, REDACTION_TOKEN))
        .collect();
    let mut group = c.benchmark_group("replace_many");
    for target_bytes in [256 * 1024, 1024 * 1024, 12 * 1024 * 1024, 100 * 1024 * 1024] {
        let text = prose(target_bytes);
        group.throughput(Throughput::Bytes(text.len() as u64));
        // The redaction shape: every dense word matches once per sentence and
        // the splice writes one ~corpus-sized output string, so the 100 MiB
        // leg's ~10.7M-match splice lands in the dense find cell's
        // slow-iteration regime — 10 samples there (the same precedent),
        // default elsewhere.
        if target_bytes == 100 * 1024 * 1024 {
            group.sample_size(10);
        }
        group.bench_with_input(
            BenchmarkId::new("dense", format!("{}B", text.len())),
            &text,
            |bench, text| {
                bench.iter(|| {
                    search_impl::replace_many(black_box(text), &replacements)
                        .expect("automaton build failed")
                })
            },
        );
    }
    group.finish();
}

fn bench_replace_many_masked(c: &mut Criterion) {
    // The same dense redaction map the replace_many cell drives — built
    // from the pinned DENSE_PATTERNS the same way, so the two cells' maps
    // cannot drift — plus the mask char that pads a masked value out to
    // its span's character count.
    let replacements: Vec<(&str, &str)> = DENSE_PATTERNS
        .iter()
        .map(|&word| (word, REDACTION_TOKEN))
        .collect();
    let mut group = c.benchmark_group("replace_many_masked");
    for target_bytes in [256 * 1024, 1024 * 1024, 12 * 1024 * 1024, 100 * 1024 * 1024] {
        let text = prose(target_bytes);
        group.throughput(Throughput::Bytes(text.len() as u64));
        // The length-preserving twin of the replace_many dense cell: the
        // same scan and the same ~10.7M matches at 100 MiB, but each match
        // is masked to its span's CHARACTER count (the token truncated
        // where the span is shorter — fifteen of the seventeen dense words
        // — or padded with the mask where it is longer — "specifications"
        // and "Maintenance"), so the output's char count equals the
        // input's. The same slow-iteration regime at 100 MiB, so the same
        // 10-sample reduction there (default elsewhere).
        if target_bytes == 100 * 1024 * 1024 {
            group.sample_size(10);
        }
        group.bench_with_input(
            BenchmarkId::new("dense", format!("{}B", text.len())),
            &text,
            |bench, text| {
                bench.iter(|| {
                    search_impl::replace_many_masked(black_box(text), &replacements, '*')
                        .expect("automaton build failed")
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_search,
    bench_replace_many,
    bench_replace_many_masked
);
criterion_main!(benches);
