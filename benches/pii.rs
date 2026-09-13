//! Criterion benches for `pii_impl::scrub_pii`: the contact-material
//! scrub over the contact-bearing prose corpus (one email and one
//! human-spelled E.164 number per sentence, plus a bare digit run the
//! `+` anchoring must leave alone), at the sizes the Python-side cells
//! measure — 1 KiB and ~100 KiB (the wall cells, the quoted chain
//! comparison in tests/test_performance.py), 1 MiB, and 12 MiB (the
//! GIL cell's size in tests/test_gil_release.py). Both rules with the
//! documented default salt, the default-args shape a caller actually
//! spells; the unsalted and subset lanes are contract questions, pinned
//! by the differential gates, not throughput questions.
//!
//! The corpus recipe is this bench's own (the `benches/text.rs`
//! precedent for a bench-only kind): it mirrors
//! `tests/reference.py`'s `_CONTACTS_SENTENCE`/`contacts` byte-for-byte,
//! pinned there by `tests/test_bench_corpus_parity.py`, so the bench
//! numbers and the Python-side wall/GIL cell numbers cross-reference on
//! the same bytes.
//!
//! Run locally with `cargo bench --no-default-features --bench pii`.
//! CI only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::pii_impl::{PiiRules, scrub_pii};

// Contact-bearing prose — mirrors tests/reference.py's
// `_CONTACTS_SENTENCE`: the error-excerpt shape the scrub exists for,
// with the bare "ticket 4096" digit run riding along as the deliberate
// non-match.
const CONTACTS_SENTENCE: &str = "The intake desk rang fungai.chetima@example.com at +1 (415) 555-2671 twice about ticket 4096, no answer. ";

fn repeat_to(target_bytes: usize, unit: &str) -> String {
    unit.repeat((target_bytes / unit.len()).max(1))
}

fn contacts(target_bytes: usize) -> String {
    repeat_to(
        target_bytes,
        &format!("{}{}", CONTACTS_SENTENCE.repeat(4), "\n\n"),
    )
}

fn bench_scrub_pii(c: &mut Criterion) {
    let mut group = c.benchmark_group("scrub_pii");
    for target_bytes in [1024, 100 * 1024, 1024 * 1024, 12 * 1024 * 1024] {
        let text = contacts(target_bytes);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("both_rules_default_salt", format!("{}B", text.len())),
            &text,
            |bench, text| {
                bench.iter(|| {
                    scrub_pii(
                        black_box(text),
                        PiiRules::BOTH,
                        tors::pii_impl::DEFAULT_SALT,
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_scrub_pii);
criterion_main!(benches);
