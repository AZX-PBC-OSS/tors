//! Criterion benches for `secret_impl`: the secret-token grammars over
//! LOG-SHAPED text, with and without tokens — the shape the family
//! exists for (a scrub over a worker's log lines before they reach a
//! log line), and the shape its per-byte head dispatch is tuned for
//! (six grammar heads walked over every byte of token-free prose; the
//! with-tokens lane adds the match walks, the splices, and the token
//! digests).
//!
//! Run locally with `cargo bench --no-default-features --bench secret`.
//! CI only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::secret_impl::{SecretRules, scrub_secrets};

// Token-dense log text: one credential shape every other line, every
// grammar represented, all credentials synthesized. The GIL cell's
// corpus shape (tests/test_gil_release.py::_secrets_corpus), built
// independently here.
// Push protection scans the pushed blobs for the contiguous token
// shapes, so every full-shape vendor token is split across the
// concat! (the joined value is what the scanner sees).
const SECRETS_UNIT: &str = concat!(
    "INFO deploy worker key=AKIA",
    "B2C4E6G8H1J3K5M9 accepted\n",
    "INFO poll health ok latency=12ms\n",
    "WARN slack api xox",
    "b-123456789012-1234567890123-abcdefghijklmnop rate\n",
    "INFO poll queue depth 0\n",
    "ERR stripe charge failed key=",
    "sk_live_",
    "4eC39HqLyjWDarjtT1zdp7dc retry\n",
    "INFO poll health ok latency=11ms\n",
    "INFO github webhook token ghp_",
    "aB3xY9kL2mN5pQ7rS4tU8vW1xY6zA0bC3dEF ok\n",
    "INFO poll queue depth 1\n",
    "INFO git sha 0123456789abcdef0123456789abcdef01234567 checked\n",
    "INFO poll health ok latency=10ms\n",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIB\n-----END RSA PRIVATE KEY-----\n",
    "INFO poll rotate complete\n",
);

// Token-free log text: the same line shape, no credential anywhere —
// every byte pays the head dispatch and no grammar ever fires (the
// `Cow::Borrowed` identity lane, the fast path the with-tokens lane is
// measured against).
const CLEAN_UNIT: &str = concat!(
    "INFO deploy worker key=accepted\n",
    "INFO poll health ok latency=12ms\n",
    "WARN slack api rate limit reached\n",
    "INFO poll queue depth 0\n",
    "ERR stripe charge failed retry\n",
    "INFO poll health ok latency=11ms\n",
    "INFO github webhook token ok\n",
    "INFO poll queue depth 1\n",
    "INFO git sha checked\n",
    "INFO poll health ok latency=10ms\n",
    "-----END of the rotated log marker-----\n",
    "INFO poll rotate complete\n",
);

fn corpus(target_bytes: usize, unit: &str) -> String {
    unit.repeat((target_bytes / unit.len()).max(1))
}

fn bench_scrub_secrets(c: &mut Criterion) {
    for (kind, unit) in [("token_dense", SECRETS_UNIT), ("token_free", CLEAN_UNIT)] {
        let mut group = c.benchmark_group("scrub_secrets");
        for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024] {
            let text = corpus(target_bytes, unit);
            group.throughput(Throughput::Bytes(text.len() as u64));
            group.bench_with_input(
                BenchmarkId::new(kind, format!("{}B", text.len())),
                &text,
                |bench, text| {
                    bench.iter(|| {
                        scrub_secrets(
                            black_box(text),
                            SecretRules::ALL,
                            tors::secret_impl::SECRETS_DEFAULT_SALT,
                        )
                    })
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, bench_scrub_secrets);
criterion_main!(benches);
