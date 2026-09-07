//! Criterion bench for `encoding_impl::detect`: `tors.detect_encoding`'s
//! core, a single `chardetng::EncodingDetector::feed` + `guess` pass over
//! arbitrary bytes — the heuristic sniff a `utf8_is_valid`-failed input
//! reaches for next (see `encoding_impl.rs`'s module docs for the intended
//! pipeline shape).
//!
//! Corpus: the shared prose recipe rendered to UTF-8 bytes (the same
//! bytes-in corpus `benches/bytes.rs` uses) — `detect` is a single forward
//! scan regardless of what it eventually guesses, so a well-formed-UTF-8
//! corpus exercises the same scan cost a legacy-encoded one would; no need
//! to construct a second, mis-encoded corpus purely for this throughput
//! measurement (correctness across real encodings is `tests/test_encoding.py`'s
//! job, not this bench's).
//!
//! Run locally with `cargo bench --no-default-features --bench encoding`.
//! CI only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`).

#[expect(dead_code)] // `decomposed` and `crlf` have no encoding-bench cell
// (normalize.rs/bytes.rs/text.rs bench them), so within THIS bench's
// compilation the shared module's builders are dead — allowed here only;
// the expectation is deliberate and self-retiring. Same pattern as
// benches/search.rs and benches/diff.rs.
mod common;

use common::prose;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use tors::encoding_impl;

fn bench_detect_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("detect_encoding");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024] {
        let text = prose(target_bytes);
        let bytes = text.as_bytes();
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("prose_utf8", format!("{}B", bytes.len())),
            &bytes,
            |bench, bytes| bench.iter(|| encoding_impl::detect(black_box(bytes), None)),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_detect_encoding);
criterion_main!(benches);
