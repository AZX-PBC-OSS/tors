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
                        tors::pii_impl::KEYS_DEFAULT_SALT,
                    )
                })
            },
        );
    }
    // Keys-rule lanes (the contact corpus above never fires the keys
    // pass — these do): one key per sentence across the table families
    // (shared-charset tails, an AWS ID, an Azure marker, a GCP dot, a
    // full multi-line PEM block), plus the two adversarial shapes — a
    // dash-dense non-match input (the PEM `-` anchor filter on prose
    // that never opens a header) and a run of unterminated BEGINs (the
    // END-index path: one sweep shared by every BEGIN, never a
    // per-anchor re-scan).
    let key_sentence = format!(
        "rotated {} and {} and {} ok. ",
        "sk-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV",
        // Split across the concatenation: the joined shape trips push
        // protection (Amazon's documented example, not a secret).
        concat!("AKIA", "IOSFODNN7EXAMPLE"),
        "AccountKey=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX",
    );
    let pem_block =
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA7b\n-----END RSA PRIVATE KEY-----\n";
    for (id, text) in [
        (
            "keys_dense",
            repeat_to(100 * 1024, &format!("{key_sentence}{pem_block}")),
        ),
        (
            "dashes_no_header",
            repeat_to(100 * 1024, "well-known - state-of-the-art - up-to-date - "),
        ),
        (
            "unterminated_begins",
            repeat_to(
                100 * 1024,
                "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA7b\n",
            ),
        ),
    ] {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new(id, format!("{}B", text.len())),
            &text,
            |bench, text| {
                bench.iter(|| {
                    scrub_pii(
                        black_box(text),
                        PiiRules::BOTH,
                        tors::pii_impl::DEFAULT_SALT,
                        tors::pii_impl::KEYS_DEFAULT_SALT,
                    )
                })
            },
        );
    }
    // Mismatched-END flood at scaling block counts: N RSA BEGINs each
    // followed by N EC ENDs no BEGIN can terminate at. All BEGINs share
    // words, so every END lands in one bucket and each BEGIN
    // binary-searches past the flood — the series must scale ~linearly
    // in N (a 4x-per-doubling series is the END index failing).
    for blocks in [500usize, 1000, 2000, 4000, 8000] {
        let begins = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA7b\n".repeat(blocks);
        let ends = "-----END EC PRIVATE KEY-----\n".repeat(blocks);
        let text = format!("{begins}{ends}");
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("mismatched_ends", format!("{blocks}blocks")),
            &text,
            |bench, text| {
                bench.iter(|| {
                    scrub_pii(
                        black_box(text),
                        PiiRules::BOTH,
                        tors::pii_impl::DEFAULT_SALT,
                        tors::pii_impl::KEYS_DEFAULT_SALT,
                    )
                })
            },
        );
    }
    // The same flood with DISTINCT words per BEGIN and per END (the
    // #92 shape the shared-words lane cannot see): every BEGIN's bucket
    // is empty — each BEGIN pays one hash probe and a miss. The old
    // per-words failure memo degraded super-linearly here (~2.87
    // exponent measured; 67ms/659ms/3724ms/30053ms at
    // 500/1000/2000/4000 blocks through the Python API). The series
    // must scale ~linearly in N.
    for blocks in [500usize, 1000, 2000, 4000, 8000] {
        let begins: String = (0..blocks)
            .map(|i| format!("-----BEGIN K{i} PRIVATE KEY-----\n"))
            .collect();
        let ends: String = (0..blocks)
            .map(|i| format!("-----END L{i} PRIVATE KEY-----\n"))
            .collect();
        let text = format!("{begins}{ends}");
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("distinct_begin_words", format!("{blocks}blocks")),
            &text,
            |bench, text| {
                bench.iter(|| {
                    scrub_pii(
                        black_box(text),
                        PiiRules::BOTH,
                        tors::pii_impl::DEFAULT_SALT,
                        tors::pii_impl::KEYS_DEFAULT_SALT,
                    )
                })
            },
        );
    }
    // Degenerate-domain guard (pins the linear domain split): 50k `a.`
    // pairs, both the non-match (`…a`) and the match (`…zz`) spellings.
    // The backward sweep is O(run), never O(run²); sha2 runs only for
    // spans that actually match (the non-match allocates nothing).
    let dots = "a.".repeat(50_000);
    for (id, text) in [
        ("degenerate_domain_non_match", format!("x@{dots}a")),
        ("degenerate_domain_match", format!("x@{dots}zz")),
    ] {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new(id, format!("{}B", text.len())),
            &text,
            |bench, text| {
                bench.iter(|| {
                    scrub_pii(
                        black_box(text),
                        PiiRules::BOTH,
                        tors::pii_impl::DEFAULT_SALT,
                        tors::pii_impl::KEYS_DEFAULT_SALT,
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_scrub_pii);
criterion_main!(benches);
