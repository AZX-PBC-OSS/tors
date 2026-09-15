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
//! The `first_invalid_charset` group benches the batch codepoint-set
//! validator's core (`charset_impl::first_invalid_charset`, the scan
//! surface's batch-only companion) over the identifier rule (TaskQ's
//! `_IDENT_RE` shape: letters and underscore at position 0, digits
//! joining after) on 1 / 10 / 100-item batches of identifier-shaped
//! strings — the sizes bracketing the motivating consumer's batches
//! (a 100-tag enqueue) and the bulk pre-flight shape. Every item is
//! valid, so each iteration is the full-pass worst case (no
//! short-circuit), and each iteration is the WHOLE core call, set builds
//! included — the same shape as the Python call, whose per-call cost at
//! these sizes is the wall race's measured band (tests/test_performance.py:
//! ~0.5 µs at 10 items, ~2 µs at 100, scaling per item after). It lives
//! in this file because no per-area bench file fits a validator (the
//! search/integrity/text benches are all whole-corpus transforms), and
//! the issue's direction names this as the fallback home.
//!
//! The `unescaped_scan` group (the issue #50 surface,
//! `scan_impl::find_unescaped`): the escape-parity byte scan over the same
//! prose ladder, driven with the six-byte escape-text needle
//! `UNESCAPED_NEEDLE` (mirroring `reference.UNESCAPED_NEEDLE`, cross-checked
//! by tests/test_bench_corpus_parity.py so the bench numbers and the
//! Python-side GIL/wall cells cross-reference on the same bytes). Two
//! shapes:
//!
//! - `sparse`: the plain prose corpus as bytes — the needle never occurs
//!   (the corpus holds no backslash at all), so the iteration is the pure
//!   memmem scan with the parity walk never taken.
//! - `dense`: the false-positive corpus — one literal escape TEXT per
//!   sentence (`FALSE_POSITIVE_LITERAL`, the backslash itself escaped, so
//!   every one of the needle's occurrences sits behind a single backslash:
//!   an odd run, REJECTED) — the full scan plus the per-hit parity work
//!   with no early exit, the workload a confirm-by-reparse walk existed
//!   for. A corpus of REAL escapes would answer at its first hit and
//!   measure an early exit, not the scan, which is why the hit-dense shape
//!   is all-rejected by construction.
//!
//! Each iteration is the WHOLE core call, `find_unescaped` including the
//! Finder build (the same shape as the Python call; the build is ns-scale
//! for a six-byte needle). Iterations stay sub-10ms at every ladder size
//! (memchr-class throughput), so every cell keeps default sampling.
//!
//! The `utf8_byte_len` group (the issue #52 companion core,
//! `scan_impl::utf8_byte_len`): the answer to "how many UTF-8 bytes is
//! this str?" versus the cost of the stdlib expression it replaces. Two
//! cells per size:
//!
//! - `core`: `scan_impl::utf8_byte_len(&text)` — the borrowed `&str`'s
//!   `len()`, ONE FIELD READ, flat ns at every size. That flatness is the
//!   whole result: the core's throughput column is deliberately
//!   meaningless (an O(1) read over n input bytes reports absurd TiB/s)
//!   and exists only to make the flat line visible next to the baseline.
//! - `memcpy_floor`: `text.as_bytes().to_vec()` — one allocation plus
//!   one memcpy of the full corpus, the dominant cost of
//!   `len(s.encode("utf-8"))` on ASCII input (CPython's ASCII encode fast
//!   path is exactly this copy) and the floor model of it on non-ASCII
//!   input (where a cold encode additionally runs the ucs-to-UTF-8
//!   encoder pass, several× the copy). The true end-to-end lanes — ASCII
//!   zero-copy alias, non-ASCII materialize-once-then-cached, all raced
//!   against the live expression — are measured Python-side by the
//!   `utf8_byte_len` cells in tests/test_performance.py; a Rust bench
//!   cannot see the pyo3 borrow where those lanes live.
//!
//! Ladder: 64 KiB / 1 MiB / 12 MiB — the TaskQ result cap
//! (`MAX_RESULT_BYTES`, the size the function's motivating double pass
//! pays), the wall cells' mid leg, and the suite's canonical size. No
//! 100 MiB leg (the sibling groups' 100 MiB exists to show a linear
//! scan's throughput stability; a flat O(1) read and a trivial memcpy
//! have nothing new to show there). Corpus: the shared prose recipe
//! only — the core is representation-independent (the ASCII-alias vs
//! materialize-once distinction lives at the pyo3 borrow, not here), and
//! prose is already corpus-pinned by tests/test_bench_corpus_parity.py
//! through the common module, so this group adds no bench-local
//! constants for that file to pin. Iterations are ns-scale (core) and
//! µs-scale (baseline), so every cell keeps default sampling.
//!
//! The `utf16_byte_len` group (the interop twin's core, #52): the same
//! two-cell shape over the same ladder, with both cells carrying real
//! work this time — the twin's `core` cell is a flat field read, this
//! one is the chunked byte-class scan (`UTF16_COUNT_CHUNK`-wide, the
//! auto-vectorizing shape; see `src/scan_impl.rs` for the width's
//! measurement and for the scalar spelling it exists to avoid, which
//! measured slower than the expression the function replaces):
//!
//! - `core`: `scan_impl::utf16_byte_len(&text)` — the derived
//!   arithmetic (the lead-byte and 4-byte-lead counts, doubled), one
//!   pass over the corpus, no allocation.
//! - `rust_utf16_shape`: `text.encode_utf16().collect::<Vec<u16>>()` —
//!   the 2n allocation plus the per-codepoint encode pass, the
//!   Rust-side shape model of `len(s.encode("utf-16-le"))`'s cost
//!   (CPython's codec is this shape: an ASCII widen pass or a UCS2
//!   near-memcpy, always with the full output allocation — not a
//!   CPython timing claim). The true end-to-end lanes — the pyo3
//!   borrow's cache classes — are measured Python-side by the
//!   `utf16_byte_len` cells in tests/test_performance.py, same split
//!   as the utf8 group.
//!
//! Same corpus reasoning as the utf8 group (representation-independent
//! core over the pinned prose recipe), and iterations are µs-scale at
//! the ladder top, so default sampling everywhere.
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

use common::{PROSE_SENTENCE, prose, repeat_to};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::{charset_impl, scan_impl, search_impl};

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

// The escape-parity scan's needle (the issue #50 surface), mirroring
// tests/reference.py's UNESCAPED_NEEDLE — the six-byte escape text a JSON
// serializer renders a NUL codepoint as. Cross-checked against the
// reference constant by tests/test_bench_corpus_parity.py.
const UNESCAPED_NEEDLE: &[u8] = b"\\u0000";

// The false-positive corpus's injection unit, mirroring reference.py's
// _ESCAPE_LITERAL_TEXT: the literal six-character TEXT of the same spelling
// as a serializer renders it — the backslash itself escaped, seven bytes —
// so the needle occurs once per injection, at +1, behind one backslash (an
// odd run: rejected). Same cross-check.
const FALSE_POSITIVE_LITERAL: &str = "\\\\u0000";

// The false-positive corpus: the shared prose sentence with one literal
// escape text appended per sentence, unit-quantized like every common
// recipe — byte-identical to reference.unescaped_false_positive (pinned by
// tests/test_bench_corpus_parity.py, which rebuilds it in Python from the
// parsed constants).
fn unescaped_false_positive(target_bytes: usize) -> Vec<u8> {
    let unit = format!("{}{}", PROSE_SENTENCE, FALSE_POSITIVE_LITERAL).repeat(4) + "\n\n";
    repeat_to(target_bytes, &unit).into_bytes()
}

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

fn bench_first_invalid_charset(c: &mut Criterion) {
    // The identifier rule's two halves (the dense-patterns set's own
    // provenance: mirrored from the TaskQ _IDENT_RE shape the Python-side
    // race in tests/test_performance.py drives) and a deterministic
    // 100-item identifier batch (the job/queue/worker/tag spellings an
    // enqueue path validates; the _ident_items builders on the Python
    // side use the same shapes), all valid: the full-pass worst case.
    const IDENT_FIRST: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_";
    const IDENT_REST: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_";
    let items: Vec<String> = (0..100)
        .map(|n| match n % 4 {
            0 => format!("job_{n}"),
            1 => format!("queue_eu_{n}"),
            2 => format!("worker_{n}"),
            _ => format!("tag_{n}"),
        })
        .collect();
    let refs: Vec<&str> = items.iter().map(String::as_str).collect();
    let mut group = c.benchmark_group("first_invalid_charset");
    for count in [1, 10, 100] {
        let batch = &refs[..count];
        group.throughput(Throughput::Elements(count as u64));
        // The whole core call per iteration — set builds included, the
        // same shape as the Python call — over the first `count` items.
        group.bench_with_input(
            BenchmarkId::new("valid", format!("{count}items")),
            &batch,
            |bench, batch| {
                bench.iter(|| {
                    black_box(charset_impl::first_invalid_charset(
                        black_box(batch),
                        Some(IDENT_FIRST),
                        IDENT_REST,
                    ))
                })
            },
        );
    }
    group.finish();
}

fn bench_unescaped_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("unescaped_scan");
    for target_bytes in [256 * 1024, 1024 * 1024, 12 * 1024 * 1024, 100 * 1024 * 1024] {
        // The pure-scan shape: no occurrence, the parity walk never taken.
        let sparse = prose(target_bytes).into_bytes();
        group.throughput(Throughput::Bytes(sparse.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("sparse", format!("{}B", sparse.len())),
            &sparse,
            |bench, data| {
                bench.iter(|| {
                    scan_impl::find_unescaped(black_box(data), black_box(UNESCAPED_NEEDLE))
                })
            },
        );
        // The hit-dense all-rejected shape: every occurrence behind an odd
        // run, so the scan runs to the end doing the per-hit parity work —
        // the false-positive workload the surface exists for (a live-hit
        // corpus would early-exit at its first occurrence and measure
        // nothing).
        let dense = unescaped_false_positive(target_bytes);
        group.throughput(Throughput::Bytes(dense.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("dense", format!("{}B", dense.len())),
            &dense,
            |bench, data| {
                bench.iter(|| {
                    scan_impl::find_unescaped(black_box(data), black_box(UNESCAPED_NEEDLE))
                })
            },
        );
    }
    group.finish();
}

fn bench_utf8_byte_len(c: &mut Criterion) {
    let mut group = c.benchmark_group("utf8_byte_len");
    for target_bytes in [64 * 1024, 1024 * 1024, 12 * 1024 * 1024] {
        let text = prose(target_bytes);
        group.throughput(Throughput::Bytes(text.len() as u64));
        // The core: the borrowed &str's len() — one field read, flat ns at
        // every size (the throughput column's absurd TiB/s is the point).
        group.bench_with_input(
            BenchmarkId::new("core", format!("{}B", text.len())),
            &text,
            |bench, text| {
                bench.iter(|| scan_impl::utf8_byte_len(black_box(text)));
            },
        );
        // The memcpy floor: the alloc+memcpy the replaced expression
        // pays — CPython's ASCII encode fast path is exactly this copy
        // (the Python-side cells measure the live expression and the
        // non-ASCII cold/warm lanes this cannot see).
        group.bench_with_input(
            BenchmarkId::new("memcpy_floor", format!("{}B", text.len())),
            &text,
            |bench, text| {
                bench.iter(|| black_box(text.as_bytes().to_vec()));
            },
        );
    }
    group.finish();
}

fn bench_utf16_byte_len(c: &mut Criterion) {
    let mut group = c.benchmark_group("utf16_byte_len");
    for target_bytes in [64 * 1024, 1024 * 1024, 12 * 1024 * 1024] {
        let text = prose(target_bytes);
        group.throughput(Throughput::Bytes(text.len() as u64));
        // The core: the chunked byte-class scan — the derived arithmetic,
        // one pass, no allocation (the twin's flat-field-read core is the
        // contrast this group exists next to).
        group.bench_with_input(
            BenchmarkId::new("core", format!("{}B", text.len())),
            &text,
            |bench, text| {
                bench.iter(|| scan_impl::utf16_byte_len(black_box(text)));
            },
        );
        // The Rust-side shape model: the 2n allocation plus the
        // per-codepoint encode pass the replaced expression pays
        // (CPython's utf-16-le codec is this shape; the Python-side
        // cells measure the live expression and the borrow's cache
        // lanes this cannot see).
        group.bench_with_input(
            BenchmarkId::new("rust_utf16_shape", format!("{}B", text.len())),
            &text,
            |bench, text| {
                bench.iter(|| black_box(text.encode_utf16().collect::<Vec<u16>>()));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_search,
    bench_replace_many,
    bench_replace_many_masked,
    bench_first_invalid_charset,
    bench_unescaped_scan,
    bench_utf8_byte_len,
    bench_utf16_byte_len
);
criterion_main!(benches);
