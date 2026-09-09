//! Criterion benches for the LLM-JSON repair surface: `tors::json_repair`
//! (the Rust port of json_repair, upstream
//! https://github.com/mangiucugna/json_repair by Stefano Baccianella, MIT,
//! pinned at commit 251d141786d0f6ff561f6ec04d90188a338e2470 — see
//! src/json_repair/mod.rs for the port's full provenance and scope notes).
//!
//! Four cells over a prose-derived JSON document ladder (1 KiB / 1 MiB /
//! 12 MiB), one per pipeline stage, so each stage's regressions surface in
//! their own row instead of hiding inside one end-to-end number:
//!
//! - `valid_fast_path` (group `repair_json`): strictly valid JSON — the
//!   COMMON case for machine-emitted JSON, answered by the fence pre-pass
//!   scan plus the json.loads-parity strict parser without ever entering
//!   the heuristic engine. The regression to watch is the fast path
//!   growing slower than json.loads-equivalent work.
//! - `malformed` (group `repair_json`): the same document shape with a
//!   fixed rotation of the classic LLM defects on every other object —
//!   the heuristic repair engine (parse_string's quote inference, the
//!   object/array recovery, the truncation salvage) doing its real job.
//! - `fenced` (group `repair_json`): the valid corpus wrapped in one
//!   `` ```json `` fence — the CommonMark pre-pass composed with the fast
//!   path (unwrap, then strict-parse the payload); the fence scanner
//!   alone has its own bench (fence.rs).
//! - `coerce` (group `repair_json_schema`): a strictly valid but
//!   schema-noncompliant payload against a typed schema — the alignment
//!   layer (validate, coerce `"1"`→1 and `"yes"`→true, fill the missing
//!   default, re-validate).
//! - `merge_chain_198` / `comma_chain_199` / `seq_merges_2k` (group
//!   `repair_json_continuations`): the continuation-merge recursions at
//!   their deepest ADMISSIBLE sizes (one slot under MAX_NESTING) plus a
//!   2_000-deep same-level sequential merge — the cells that keep the
//!   depth guards honest: a guard that over-counts turns the first two
//!   into raises (and this bench's setup asserts Ok, so it fails loudly),
//!   and any per-merge overhead beyond the two integer ops shows up as a
//!   regression against the recorded baseline.
//!
//! Corpus builders are deterministic (no RNG): a top-level object wrapping
//! a list of records whose string values are the shared prose sentence
//! (`benches/common/mod.rs`) embedded as escaped JSON string content — its
//! `\n\n` tail becomes `\n` escapes, and a fixed note value carries real
//! `\"`/`\\` escapes — realistic escaped-string-heavy payloads, the shape
//! LLM JSON output actually has. Defects are applied by index arithmetic
//! (every other object, cycling a fixed five-entry list), so every leg of
//! the ladder carries the same defect mix.
//!
//! Run locally with `cargo bench --no-default-features --bench json_repair` —
//! the `--no-default-features` is required because `extension-module`
//! deliberately does not link libpython, which a bench binary needs. CI only
//! compiles it (`cargo bench --no-run`, equally with `--no-default-features`)
//! — the CI timing GATE for this surface is the Python lane instead
//! (tests/test_json_repair_performance.py, `-m timing`, 3.12 leg): its cells
//! assert output equality against the oracle and wall ratios, so a
//! regression this bench would catch fails CI through that lane. The setup
//! asserts below fire whenever the bench IS run, so a depth-accounting
//! regression can never silently time the error path.

#[expect(dead_code)] // `decomposed` and `crlf` have no json_repair-bench cell
// (normalize.rs/bytes.rs/text.rs bench them), so within THIS bench's
// compilation the shared module's builders are dead — allowed here only;
// the expectation is deliberate and self-retiring. Same pattern as
// benches/fence.rs and benches/text.rs.
mod common;

use common::prose;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tors::json_repair::{RepairConfig, loads_strict, repair};

/// The corpus's fixed note value, hand-written in ALREADY-ESCAPED form
/// (the exact bytes that sit between the value delimiters in the document
/// text): a quote pair and a backslash, so every object carries real
/// string escapes beyond the prose newlines without a second escaper pass.
const NOTE_JSON: &str = "He said \\\"replace the gasket\\\" and left \\\\ the spec on the shelf.";

/// Embed a corpus string into a JSON string value: the always-escape set
/// (quote, backslash) plus the whitespace controls CPython escapes —
/// everything the pinned prose sentence can need (its only escape-bearing
/// character is the newline; the rest is future-proofing if the shared
/// sentence ever grows). Local to this bench, not
/// `tors::json_repair::dumps`, because the bench builds document TEXT
/// (corpus input), not a `Value` tree.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

/// The per-object defect rotation for the malformed corpus: the five
/// classic LLM-output failure modes, applied one per every-other object in
/// this fixed cyclic order (index arithmetic, no RNG).
#[derive(Clone, Copy)]
enum Defect {
    /// The object's closing `}` dropped — the next object runs into it
    /// (and the merge trips the parser's duplicate-key splice).
    DropClosingBrace,
    /// A bare, unquoted object key (`id:` — Python-dict style).
    UnquoteKey,
    /// Single-quoted string VALUES, keys still double-quoted — the
    /// half-remembered-Python-dict shape.
    SingleQuoteStrings,
    /// The comma between two members dropped.
    DropComma,
    /// The object's text cut at 7/8 of its length — a mid-object token
    /// cutoff. The rotation's per-object spelling of the max-tokens
    /// truncation, so truncation-salvage work spreads through the corpus
    /// instead of concentrating at the document tail.
    Truncate,
}

/// One list element of the corpus document: an object whose string values
/// are the shared prose (embedded escaped) plus the fixed note — strictly
/// valid JSON in the clean spelling (`None`), which each defect variant
/// then bends in exactly one place.
fn object_json(index: usize, defect: Option<Defect>) -> String {
    let body = escape_json_string(&prose(256));
    // Value delimiters: single quotes under the single-quote defect (the
    // note's hand-written `\"`/`\\` escape sequences ride along inside
    // them — malformed text for the engine to chew through, not
    // understand).
    let quote = match defect {
        Some(Defect::SingleQuoteStrings) => '\'',
        _ => '"',
    };
    let clean = format!(
        "{{\"id\": {index}, \"body\": {quote}{body}{quote}, \"note\": {quote}{NOTE_JSON}{quote}}}"
    );
    match defect {
        None => clean,
        Some(Defect::DropClosingBrace) => {
            let mut object = clean;
            object.truncate(object.len() - 1); // the object's own closing brace
            object
        }
        Some(Defect::UnquoteKey) => clean.replacen("\"id\":", "id:", 1),
        Some(Defect::SingleQuoteStrings) => clean,
        Some(Defect::DropComma) => clean.replacen(", \"body\":", " \"body\":", 1),
        Some(Defect::Truncate) => {
            let cut = clean.chars().count() * 7 / 8;
            clean.chars().take(cut).collect()
        }
    }
}

/// The `valid_fast_path` corpus: a top-level object wrapping a list of
/// clean corpus objects, grown to the target size one object at a time —
/// strictly valid JSON by construction, so this cell measures the fast
/// path, never the repair engine.
fn valid_json(target_bytes: usize) -> String {
    // Target plus one object's headroom, so the final overshoot object
    // doesn't reallocate.
    let mut out = String::with_capacity(target_bytes + 1024);
    out.push_str("{\"records\": [");
    let mut index = 0usize;
    while out.len() < target_bytes {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&object_json(index, None));
        index += 1;
    }
    out.push_str("]}");
    out
}

/// The `malformed` corpus: the same document shape with the fixed defect
/// rotation — every OTHER object (even indices) carries one defect,
/// cycling through the five-entry list in order, so each defect class is
/// spread uniformly through the document and every leg exercises the
/// engine's full range rather than one lucky path. Never strictly valid
/// (object 0 is already defective), so the fast path never answers this
/// cell.
fn malformed_llm_output(target_bytes: usize) -> String {
    const DEFECTS: [Defect; 5] = [
        Defect::DropClosingBrace,
        Defect::UnquoteKey,
        Defect::SingleQuoteStrings,
        Defect::DropComma,
        Defect::Truncate,
    ];
    let mut out = String::with_capacity(target_bytes + 1024);
    out.push_str("{\"records\": [");
    let mut index = 0usize;
    while out.len() < target_bytes {
        if index > 0 {
            out.push(',');
        }
        let defect = if index.is_multiple_of(2) {
            Some(DEFECTS[(index / 2) % DEFECTS.len()])
        } else {
            None
        };
        out.push_str(&object_json(index, defect));
        index += 1;
    }
    out.push_str("]}");
    out
}

/// The `fenced` corpus: the valid corpus wrapped in one `` ```json ``
/// fence — the whole-input-is-one-block shape the pre-pass unwraps before
/// the fast path strict-parses the payload: the pre-pass COMPOSITION cell.
fn fenced_json(target_bytes: usize) -> String {
    // The inner document is a single line (its newlines live inside string
    // values as escapes), so the closing fence needs its own line —
    // CommonMark requires a fence close at line start.
    format!("```json\n{}\n```", valid_json(target_bytes))
}

/// The typed schema for the `coerce` cell: an object with string and
/// boolean properties, an array of objects (integer `id`, string `body`,
/// boolean `passed`), and an optional string with a default — exactly the
/// knobs the coercion corpus exercises.
const SCHEMA_STR: &str = r#"{
  "type": "object",
  "properties": {
    "site": {"type": "string"},
    "active": {"type": "boolean"},
    "records": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "integer"},
          "body": {"type": "string"},
          "passed": {"type": "boolean"}
        },
        "required": ["id", "body"]
      }
    },
    "priority": {"type": "string", "default": "normal"}
  },
  "required": ["site", "records"]
}"#;

/// The `coerce` corpus: strictly valid JSON that is schema-NONcompliant
/// where it matters — every record carries its integer as a string
/// (`"id": "7"`) and its boolean as `"yes"`, the top-level `active` is
/// `"yes"`, and the optional-with-default `priority` is simply absent —
/// so the schema fast path loads it, validation fails, and the alignment
/// layer does its full coerce-and-refill pass before re-validating.
fn schema_payload(target_bytes: usize) -> String {
    let mut out = String::with_capacity(target_bytes + 1024);
    out.push_str("{\"site\": \"");
    out.push_str(&escape_json_string(&prose(256)));
    out.push_str("\", \"active\": \"yes\", \"records\": [");
    let mut index = 0usize;
    while out.len() < target_bytes {
        if index > 0 {
            out.push(',');
        }
        let record = format!(
            "{{\"id\": \"{index}\", \"body\": \"{}\", \"passed\": \"yes\"}}",
            escape_json_string(&prose(256))
        );
        out.push_str(&record);
        index += 1;
    }
    out.push_str("]}");
    out
}

/// One measured repair call, opaque to the optimizer on both the inputs
/// (black_boxed at the call site) and the result: each arm's payload is
/// black_boxed separately — their types differ, so a value-typed match
/// would not unify — which keeps the Ok value, the diagnostics vec, and
/// the Err message all materialized.
fn repair_opaque(s: &str, cfg: &RepairConfig) {
    match repair(s, cfg) {
        Ok((value, diagnostics)) => {
            black_box((value, diagnostics));
        }
        Err(message) => {
            black_box(message);
        }
    }
}

fn bench_repair_json(c: &mut Criterion) {
    // Built once outside the measured closures — repair() borrows the
    // config on every call.
    let cfg = RepairConfig::default();
    let mut group = c.benchmark_group("repair_json");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024] {
        let valid = valid_json(target_bytes);
        group.throughput(Throughput::Bytes(valid.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("valid_fast_path", format!("{}B", valid.len())),
            &valid,
            |bench, text| bench.iter(|| repair_opaque(black_box(text), black_box(&cfg))),
        );

        let malformed = malformed_llm_output(target_bytes);
        group.throughput(Throughput::Bytes(malformed.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("malformed", format!("{}B", malformed.len())),
            &malformed,
            |bench, text| bench.iter(|| repair_opaque(black_box(text), black_box(&cfg))),
        );

        let fenced = fenced_json(target_bytes);
        group.throughput(Throughput::Bytes(fenced.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("fenced", format!("{}B", fenced.len())),
            &fenced,
            |bench, text| bench.iter(|| repair_opaque(black_box(text), black_box(&cfg))),
        );
    }
    group.finish();
}

fn bench_repair_json_schema(c: &mut Criterion) {
    // Setup, not measured: parse the schema once per group via the strict
    // parser (a bad SCHEMA_STR is a bench bug, so the setup unwrap is
    // legitimate — benches may panic on setup errors), and build the
    // config once; repair() borrows it per call.
    let schema = loads_strict(SCHEMA_STR).expect("SCHEMA_STR is valid JSON");
    let cfg = RepairConfig {
        schema: Some(schema),
        ..RepairConfig::default()
    };
    let mut group = c.benchmark_group("repair_json_schema");
    for target_bytes in [1024, 1024 * 1024, 12 * 1024 * 1024] {
        let payload = schema_payload(target_bytes);
        group.throughput(Throughput::Bytes(payload.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("coerce", format!("{}B", payload.len())),
            &payload,
            |bench, text| bench.iter(|| repair_opaque(black_box(text), black_box(&cfg))),
        );
    }
    group.finish();
}

fn bench_repair_json_continuations(c: &mut Criterion) {
    // The continuation-merge cells: chains sized one slot under the
    // MAX_NESTING cap so the guarded recursion runs at its deepest
    // admissible extent, and a same-level sequential merge run an order
    // of magnitude past it (sequential merges balance enter/leave per
    // continuation, so depth never accrues). Setup asserts each payload
    // actually parses Ok — a depth-accounting regression that turns these
    // into raises fails the bench here instead of silently benchmarking
    // the error path.
    let cfg = RepairConfig {
        skip_json_loads: true,
        ..RepairConfig::default()
    };
    let merge_chain = format!("{}{}1]", r#"{"a":[0],"#, r#"["b":[0],"#.repeat(198));
    assert!(repair(&merge_chain, &cfg).is_ok());
    let comma_chain = format!("{}{}", r#"{"a":1}"#, r#", "k":1}"#.repeat(199));
    assert!(repair(&comma_chain, &cfg).is_ok());
    let seq_merges = format!("{}{}", r#"{"a":[1]"#, ", [2]".repeat(2_000));
    assert!(repair(&seq_merges, &cfg).is_ok());

    let mut group = c.benchmark_group("repair_json_continuations");
    for (name, payload) in [
        ("merge_chain_198", merge_chain.as_str()),
        ("comma_chain_199", comma_chain.as_str()),
        ("seq_merges_2k", seq_merges.as_str()),
    ] {
        group.throughput(Throughput::Bytes(payload.len() as u64));
        group.bench_with_input(
            BenchmarkId::new(name, format!("{}B", payload.len())),
            payload,
            |bench, text| bench.iter(|| repair_opaque(black_box(text), black_box(&cfg))),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_repair_json,
    bench_repair_json_schema,
    bench_repair_json_continuations
);
criterion_main!(benches);
