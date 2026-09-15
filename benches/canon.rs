//! Criterion benches for `tors.content_hash`'s detached half: the
//! canonical-form emitter plus SHA-256 over an owned `Canon` tree (the
//! `pyo3` walk that builds the tree is GIL-held interpreter work, measured
//! Python-side by `tests/test_gil_release.py` and
//! `tests/test_performance.py` -- the same split every bench in this
//! directory makes: the pure-Rust core here, the binding-layer costs in
//! the Python suites).
//!
//! Two corpus shapes over an object-size ladder (sizes count the
//! CANONICAL form's bytes, the same sizing `reference.content_object`
//! uses; the trees MIRROR that corpus's shape -- same fields, notes
//! carrying the shared prose sentence twice -- without a byte-parity
//! pin, which the bench-corpus-parity mechanism reserves for the str
//! corpora):
//!
//! - `records`: the str-heavy JSON shape the Python-side cells measure,
//!   where the escape table is an idle branch (prose is printable ASCII)
//!   and the work is raw-run copying plus SHA-256.
//! - `escape_heavy`: the same records shape with the note replaced by a
//!   cycling run of EVERY valid Unicode codepoint (controls, DEL, `"` and
//!   `\`, BMP non-ASCII, astral), so the escape table -- the five short
//!   escapes, `\u00XX`, `\uXXXX`, and the surrogate pairs -- is the whole
//!   workload: the adversarial side of the emitter's cost, the shape the
//!   fuzz target attacks.
//!
//! Run locally with `cargo bench --no-default-features --bench canon`.
//! CI only compiles it (`cargo bench --no-run`, equally with
//! `--no-default-features`).

#[expect(dead_code)] // `decomposed` and `crlf` have no canon-bench cell
// (normalize.rs/bytes.rs/text.rs bench them), so within THIS bench's
// compilation the shared module's builders are dead — allowed here only;
// the expectation is deliberate and self-retiring. Same pattern as
// benches/integrity.rs and benches/search.rs.
mod common;

use common::PROSE_SENTENCE;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::canon_impl::{self, Canon};

/// One records-corpus record, mirroring `reference._content_record`'s
/// field set (sorted keys, the shape a serialized API row has). The score
/// spelling mimics Python's repr for the k/8 values (Rust's `Display`
/// prints `1.0` as `1`); the bench corpus mirrors the shape, it is not
/// byte-parity-pinned with the Python corpus.
fn record(i: usize, note: &str) -> Canon {
    let score = (i % 40) as f64 * 0.125;
    let mut score_spelling = format!("{score}");
    if !score_spelling.contains('.') {
        score_spelling.push_str(".0");
    }
    let tags = if i.is_multiple_of(2) {
        vec![Canon::Str("alpha".into()), Canon::Str("beta".into())]
    } else {
        vec![]
    };
    Canon::Map(vec![
        ("active".into(), Canon::Bool(i.is_multiple_of(3))),
        ("id".into(), Canon::Int(i as i64)),
        ("name".into(), Canon::Str(format!("record-{i:06}"))),
        ("note".into(), Canon::Str(note.to_string())),
        ("score".into(), Canon::Float(score_spelling)),
        ("tags".into(), Canon::Seq(tags)),
    ])
}

/// The records document at `target_bytes` of canonical form: whole-record
/// quantization, the same idiom `reference.content_object` uses.
fn records_tree(target_bytes: usize, note: &str) -> Canon {
    let unit = canon_impl::canonical_bytes(&record(0, note)).len() + 1;
    let n = (target_bytes / unit).max(1);
    Canon::Map(vec![
        ("count".into(), Canon::Int(n as i64)),
        (
            "records".into(),
            Canon::Seq((0..n).map(|i| record(i, note)).collect()),
        ),
        ("schema".into(), Canon::Int(2)),
    ])
}

/// A note-length run cycling every valid Unicode codepoint (surrogates
/// are not valid scalar values and are skipped by `char::from_u32`):
/// controls and DEL, `"` and `\`, BMP non-ASCII, and astral codepoints,
/// so every branch of the escape table fires per note.
fn codepoint_note(target_len: usize) -> String {
    let mut out = String::with_capacity(target_len + 4);
    for cp in (0u32..0x110000).cycle().filter_map(char::from_u32) {
        if out.len() >= target_len {
            break;
        }
        out.push(cp);
    }
    out
}

fn bench_content_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("content_hash");
    let note = format!("{PROSE_SENTENCE}{PROSE_SENTENCE}");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024] {
        let tree = records_tree(target_bytes, &note);
        let canonical_len = canon_impl::canonical_bytes(&tree).len();
        group.throughput(Throughput::Bytes(canonical_len as u64));
        group.bench_with_input(
            BenchmarkId::new("records", format!("{canonical_len}B")),
            &tree,
            |bench, tree| bench.iter(|| canon_impl::digest_hex(black_box(tree))),
        );
    }
    let hot_note = codepoint_note(PROSE_SENTENCE.len() * 2);
    for target_bytes in [1024 * 1024, 12 * 1024 * 1024] {
        let tree = records_tree(target_bytes, &hot_note);
        let canonical_len = canon_impl::canonical_bytes(&tree).len();
        group.throughput(Throughput::Bytes(canonical_len as u64));
        group.bench_with_input(
            BenchmarkId::new("escape_heavy", format!("{canonical_len}B")),
            &tree,
            |bench, tree| bench.iter(|| canon_impl::digest_hex(black_box(tree))),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_content_hash);
criterion_main!(benches);
