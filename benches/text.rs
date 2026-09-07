//! Criterion benches for the v0.3 str surface: `html_impl::unescape_checked`
//! (the limit-aware spelling the pyo3 layer calls — benched with the default
//! 4300 limit, the value a CPython 3.11+ call reads), `forms_impl::nfc`/
//! `nfkd`, `segmentation_impl::grapheme_count`/
//! `word_bounds`, and `b64_impl::decode`. The shared corpus recipes (prose,
//! decomposed, crlf) live once in `benches/common/mod.rs`, pinned
//! byte-identical to `tests/reference.py` by `tests/test_bench_corpus_parity.py`;
//! the v0.3/v0.4-only recipes below are the entity-bearing prose corpus
//! (`entities`) and the compat corpus, plus the b64-decode corpus (the prose
//! corpus rendered through the same
//! `base64::engine::general_purpose::STANDARD` engine `b64_impl::encode`
//! uses — i.e. `reference.corpus_b64` on the Python side). Ladder: entities
//! at 1 KiB / 1 MiB / 12 MiB / 100 MiB plus the no-`&` prose cell for the
//! unescape fast path; prose at the same ladder for the forms/segmentation
//! cells plus the decomposed and compat variants at 12 MiB.
//!
//! v0.4 note on what the forms cells measure: the quick-check fast path
//! changed the corpus→work mapping. The PROSE cells now measure the fast
//! lane (ASCII quick-checks Yes under every form, so `nfc`/`nfkd` return the
//! input borrowed — the scan alone, ~2.4ms at 12 MiB); the DECOMPOSED cell
//! still measures NFC's full pass (combining marks keep the quick check at
//! Maybe) but is the identity under NFKD (no compatibility mappings); the
//! COMPAT corpus (decomposed accents plus U+FB01 and U+FF10 per unit) is the
//! one that still pays BOTH forms' full passes — the direct-runtime
//! regression cells for the K-forms.
//!
//! v0.8 note: the `sentence_bounds` group benches the sentence-segmentation
//! core — the `split_sentence_bound_indices` pass plus the bounds-vector
//! fill — over the plain prose corpus at the 1 KiB / 1 MiB / 12 MiB /
//! 100 MiB ladder: the word_bounds cells' list shape on the far sparser
//! sentence class (~170k tuples at 12 MiB where word_bounds fills ~3.67M),
//! so like every other cell in this file it keeps criterion's default
//! sampling at every leg.
//!
//! The `quote`/`unquote` groups bench the percent-encoding pair over the
//! plain prose corpus at the same 1 KiB / 1 MiB / 12 MiB / 100 MiB ladder:
//! `url_impl::quote(text, "/")` (the stdlib's default `safe`) measures the
//! owned-branch encode — the corpus's spaces and newlines are exactly the
//! bytes that become `%XX` under it — and `url_impl::unquote` measures the
//! decode pass over the percent-encoded variant of the same corpus, built
//! once per bench setup the way the b64 corpus is pre-rendered (the `quote`
//! output itself, so every escape the decode resolves is a real one). Both
//! are linear single-pass cores, so every leg keeps this file's default
//! sampling.
//!
//! These guard tors against its own regressions across versions — the
//! cross-implementation wall-time claims are owned by tests/test_performance.py
//! (only the Python side can run the stdlib/reference), and the GIL-release
//! claims by tests/test_gil_release.py (including the word_bounds
//! marshalling-band finding, which lives at the pyo3 layer this bench
//! deliberately does not measure).
//!
//! Run locally with `cargo bench --no-default-features --bench text` — the
//! `--no-default-features` is required because `extension-module` deliberately
//! does not link libpython, which a bench binary needs. CI only compiles it
//! (`cargo bench --no-run`, equally with `--no-default-features`). Sample
//! sizes are criterion's defaults.
//!
//! Results are recorded in the README (performance table) from a local run on
//! the shared dev box with load disclosed; re-run quiet before quoting
//! release numbers.

#[expect(dead_code)] // `crlf` has no text-bench cell (normalize.rs and bytes.rs bench
// it), so within THIS bench's compilation the shared module's
// crlf builder is dead — allowed here only; the expectation is
// deliberate and self-retiring (a future crlf cell makes it
// unfulfilled, and the lint says so).
mod common;

use base64::Engine;
use common::{decomposed, prose};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use tors::b64_impl;
use tors::forms_impl;
use tors::html_impl;
use tors::segmentation_impl;
use tors::url_impl;

// The decomposed sentence plus a compatibility ligature (U+FB01) and a
// fullwidth digit (U+FF10) — mirrors tests/reference.py's `_COMPAT_SENTENCE`:
// the corpus that still pays NFKC/NFKD's full pass under the v0.4
// quick-check fast paths (plain decomposed prose quick-checks Yes for the
// K-forms and comes back borrowed).
const COMPAT_SENTENCE: &str = "The quarte\u{0301}rly oil sa\u{0301}mple interval \u{fb01}eld outa\u{0301}ges was adjusted \u{ff10} days after the bushing torque specifications changed. Maintenance windows now close within fourteen days. ";

// Entity-bearing prose — mirrors tests/reference.py's `_ENTITY_SENTENCE`
// (nine HTML5 refs per sentence): the html_unescape corpus.
const ENTITY_SENTENCE: &str = "The quarterly &amp; field &lt;outage&gt; interval &quot;adjusted&quot; after &#233; the bushing &copy; changed &nbsp; for torque &there4; specs. ";

fn repeat_to(target_bytes: usize, unit: &str) -> String {
    unit.repeat((target_bytes / unit.len()).max(1))
}

fn compat(target_bytes: usize) -> String {
    repeat_to(
        target_bytes,
        &format!("{}{}", COMPAT_SENTENCE.repeat(4), "\n\n"),
    )
}

fn entities(target_bytes: usize) -> String {
    repeat_to(
        target_bytes,
        &format!("{}{}", ENTITY_SENTENCE.repeat(4), "\n\n"),
    )
}

fn bench_html(c: &mut Criterion) {
    let mut group = c.benchmark_group("html_unescape");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024, 100 * 1024 * 1024] {
        let corpus = entities(target_bytes);
        group.throughput(Throughput::Bytes(corpus.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("entities", format!("{}B", corpus.len())),
            &corpus,
            |b, text| {
                b.iter(|| {
                    html_impl::unescape_checked(black_box(text), black_box(Some(4300)))
                        .expect("bench corpus carries no over-limit refs")
                })
            },
        );
    }
    // The no-'&' fast path: a borrowed return, the scan-for-'&' cost alone.
    let corpus = prose(12 * 1024 * 1024);
    group.throughput(Throughput::Bytes(corpus.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("no-amp", format!("{}B", corpus.len())),
        &corpus,
        |b, text| {
            b.iter(|| {
                html_impl::unescape_checked(black_box(text), black_box(Some(4300)))
                    .expect("no-'&' corpus carries no refs at all")
            })
        },
    );
    group.finish();
}

fn bench_forms(c: &mut Criterion) {
    for (kind, corpus) in [
        ("prose", prose(12 * 1024 * 1024)),
        ("decomposed", decomposed(12 * 1024 * 1024)),
        ("compat", compat(12 * 1024 * 1024)),
    ] {
        let mut group = c.benchmark_group(kind.to_string());
        group.throughput(Throughput::Bytes(corpus.len() as u64));
        let label = format!("{}B", corpus.len());
        group.bench_with_input(BenchmarkId::new("nfc", &label), &corpus, |b, text| {
            b.iter(|| forms_impl::nfc(black_box(text)))
        });
        group.bench_with_input(BenchmarkId::new("nfkd", &label), &corpus, |b, text| {
            b.iter(|| forms_impl::nfkd(black_box(text)))
        });
        group.finish();
    }
}

fn bench_segmentation(c: &mut Criterion) {
    for (kind, corpus) in [
        ("prose", prose(12 * 1024 * 1024)),
        ("decomposed", decomposed(12 * 1024 * 1024)),
    ] {
        let mut group = c.benchmark_group(kind.to_string());
        group.throughput(Throughput::Bytes(corpus.len() as u64));
        let label = format!("{}B", corpus.len());
        group.bench_with_input(
            BenchmarkId::new("grapheme_count", &label),
            &corpus,
            |b, text| b.iter(|| segmentation_impl::grapheme_count(black_box(text))),
        );
        group.bench_with_input(
            BenchmarkId::new("word_bounds", &label),
            &corpus,
            |b, text| b.iter(|| segmentation_impl::word_bounds(black_box(text))),
        );
        group.finish();
    }
}

fn bench_sentence_bounds(c: &mut Criterion) {
    let mut group = c.benchmark_group("sentence_bounds");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024, 100 * 1024 * 1024] {
        let corpus = prose(target_bytes);
        group.throughput(Throughput::Bytes(corpus.len() as u64));
        // The word_bounds cells' list shape on the sentence class: the
        // split_sentence_bound_indices pass plus the bounds-vector fill, one
        // whole-core call per iteration — default sampling (this file's
        // convention; the 100 MiB leg is the same linear pass, just longer).
        group.bench_with_input(
            BenchmarkId::new("prose", format!("{}B", corpus.len())),
            &corpus,
            |b, text| b.iter(|| segmentation_impl::sentence_bounds(black_box(text))),
        );
    }
    group.finish();
}

fn bench_b64_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("b64_decode");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024, 100 * 1024 * 1024] {
        // The decode corpus is the prose corpus rendered through the same
        // STANDARD engine `b64_impl::encode` uses — `reference.corpus_b64`
        // on the Python side (the rendering is pinned textually by
        // tests/test_bench_corpus_parity.py).
        let corpus = prose(target_bytes);
        let b64 = base64::engine::general_purpose::STANDARD.encode(corpus.as_bytes());
        group.throughput(Throughput::Bytes(b64.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("prose-b64", format!("{}B", b64.len())),
            &b64,
            |b, text| b.iter(|| b64_impl::decode(black_box(text), true)),
        );
    }
    group.finish();
}

fn bench_quote(c: &mut Criterion) {
    let mut group = c.benchmark_group("quote");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024, 100 * 1024 * 1024] {
        let corpus = prose(target_bytes);
        group.throughput(Throughput::Bytes(corpus.len() as u64));
        // The owned-branch encode: the corpus's spaces and newlines are
        // exactly the bytes that become %XX under safe="/" — every unit
        // pays the table walk plus the hex emit.
        group.bench_with_input(
            BenchmarkId::new("prose", format!("{}B", corpus.len())),
            &corpus,
            |b, text| b.iter(|| url_impl::quote(black_box(text), black_box("/"))),
        );
    }
    group.finish();
}

fn bench_unquote(c: &mut Criterion) {
    let mut group = c.benchmark_group("unquote");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024, 100 * 1024 * 1024] {
        let corpus = prose(target_bytes);
        // The percent-encoded variant of the same corpus, built once per
        // setup the way the b64 corpus is: the quote output itself, so
        // every escape the decode pass resolves is a real one (and the
        // pair round-trips by construction).
        let encoded = url_impl::quote(&corpus, "/").into_owned();
        group.throughput(Throughput::Bytes(encoded.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("prose-pct", format!("{}B", encoded.len())),
            &encoded,
            |b, text| b.iter(|| url_impl::unquote(black_box(text))),
        );
    }
    group.finish();
}

fn bench_tors_text(c: &mut Criterion) {
    bench_html(c);
    bench_forms(c);
    bench_segmentation(c);
    bench_sentence_bounds(c);
    bench_b64_decode(c);
    bench_quote(c);
    bench_unquote(c);
}

criterion_group!(benches, bench_tors_text);
criterion_main!(benches);
