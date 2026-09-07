//! Criterion benches for the v0.6 diff surface: `diff_impl::diff_opcodes` —
//! the Rust core alone (the `Vec<char>` materialization of both operands and
//! the Myers search), with no pyo3 layer; the argument borrows and the
//! O(ops) tuple marshalling are measured Python-side by
//! `tests/test_gil_release.py` (the marshalling band) and
//! `tests/test_performance`-style cells in `tests/test_diff_opcodes.py`
//! (the difflib race).
//!
//! Corpus shapes, both built from the shared prose recipe in
//! `benches/common/mod.rs` (byte-identical to `tests/reference.py`'s — the
//! constants the two sides share are pinned by
//! `tests/test_bench_corpus_parity.py`):
//!
//! - `near-identical`: the prose corpus vs itself with six scattered
//!   line-level edits (four word-swapped replacement lines, one deleted
//!   line, one inserted word-swapped line) at NON-dyadic line fractions —
//!   dyadic positions collide with the Myers divide-and-conquer's bisection
//!   midpoints at every recursion depth and fragment the op stream
//!   (measured: the same six edits at n/4, n/2, 3n/4 produce 18,917 ops at
//!   12 MiB where these positions produce 235). The re-diff-an-edited-
//!   document shape.
//! - `shuffled`: the prose corpus with every paragraph line numbered (the
//!   recipe's paragraph lines are byte-identical, so an un-numbered line
//!   shuffle would be a no-op) vs the same numbered lines permuted by the
//!   deterministic u64-LCG Fisher-Yates — the many-opcode shape that makes
//!   the Python-side marshalling band measurable (103,421 opcodes at
//!   12 MiB).
//!
//! Ladder: near-identical at 256 KiB / 1 MiB / 12 MiB / 100 MiB (the 100 MiB
//! leg measured 708 ms / 53 opcodes / ~1.1 GiB peak RSS on the dev box —
//! linear-ish in the corpus, so it carries the scaling story); shuffled at
//! 256 KiB / 1 MiB / 12 MiB — deliberately NOT 100 MiB: the bounded search's
//! work on hard inputs grows superlinearly (measured ~1.6 s per iteration at
//! 12 MiB), so a 100 MiB shuffled cell would sit at minutes per iteration,
//! out of bench budget; near-identical carries that leg. There is no 1 KiB
//! cell: the six-edit shape needs ≥ ~250 lines for the fraction positions to
//! be distinct (small-pair behavior is the correctness suite's business, not
//! a throughput ladder's).
//!
//! The v0.8 `diff_opcodes_lines` group: the line-level spelling of the same
//! engine — `split_keepend_lines` tokenization of both operands plus the
//! Myers `capture_diff_slices_deadline` pass under no deadline — over the
//! near-identical pair at the same 256 KiB / 1 MiB / 12 MiB / 100 MiB
//! ladder. The operands are line tokens (a few hundred thousand at 100 MiB,
//! not ~100M chars), and the near-identical shape's op count stays tiny at
//! every leg, so the cells are plain default-sampling cells (the 100 MiB
//! leg included — nothing approaches the shuffled cell's regime).
//!
//! The `metrics` group: the fuzzy-similarity core over ONE fixed
//! near-identical pair — the prose sentence's first 64 chars vs itself with
//! five scattered single-char substitutions, the fuzzy-matching workload
//! shape (dedup/spell-check's "two long tokens differing in a few scattered
//! spots") — with `fuzzy_impl::levenshtein` (the two-row DP),
//! `fuzzy_impl::jaro_winkler` (the match/transposition passes plus the
//! prefix boost), and `diff_impl::similarity_ratio` (the Myers search plus
//! the 2M/T arithmetic) each driven as one whole-core call per iteration on
//! the SAME pair, so the three numbers cross-reference. Fixed-size by
//! nature — the workload is a token pair, not a corpus — so there is no
//! ladder; default sampling.
//!
//! The `get_close_matches` group: the difflib close-matches core at its
//! documented defaults (`n=3`, `cutoff=0.6`) — a 41-char compound of five
//! dense prose words against a 10k-entry dictionary built once per setup
//! from the corpus's own vocabulary (the pinned corpus repeats ~23
//! distinct whitespace tokens; each entry is a vocabulary word plus the
//! base-26 rendering of its index, unique by construction). The
//! long-token-against-short-words shape forces the under-cutoff miss scan —
//! no candidate can clear 0.6 (`2M/T` caps at ~0.59 for a 17-char word
//! against the 41-char query), so the cell measures the pure per-candidate
//! scoring pass (a Myers search plus ratio arithmetic per word), the
//! close-matches twin of the search bench's sparse find cell. Default
//! sampling.
//!
//! Run locally with `cargo bench --no-default-features --bench diff` — the
//! `--no-default-features` is required because `extension-module`
//! deliberately does not link libpython, which a bench binary needs. CI only
//! compiles it (`cargo bench --no-run`, equally with `--no-default-features`).
//! Results are recorded in the README (performance table) from a local run on
//! the shared dev box with load disclosed.

#[expect(dead_code)] // `decomposed` and `crlf` have no diff-bench cell (normalize.rs and
// bytes.rs/text.rs bench them), so within THIS bench's compilation the shared
// module's builders are dead — allowed here only; the expectation is
// deliberate and self-retiring (a future cell makes it unfulfilled, and the
// lint says so). Same pattern as benches/text.rs.
mod common;

use common::{PROSE_SENTENCE, prose};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use tors::diff_impl;
use tors::fuzzy_impl;

// The word swap applied to the near-identical pair's replacement lines (and
// its inserted line) — mirrors tests/reference.py's `_DIFF_WORD_SWAP`.
const WORD_SWAP_OLD: &str = "quarterly";
const WORD_SWAP_NEW: &str = "monthly";

// The near-identical pair's edit positions as (numerator, denominator) line
// fractions — mirrors tests/reference.py's `_DIFF_REPLACE_FRACTIONS`,
// `_DIFF_DELETE_FRACTION`, `_DIFF_INSERT_FRACTION` (see the module docs for
// why they are non-dyadic).
const REPLACE_FRACTIONS: [(usize, usize); 4] = [(1, 5), (1, 3), (2, 3), (7, 9)];
const DELETE_FRACTION: (usize, usize) = (1, 7);
const INSERT_FRACTION: (usize, usize) = (4, 9);

// The shuffled pair's deterministic u64 LCG and numbering width — mirrors
// tests/reference.py's `_DIFF_LCG_SEED` / `_DIFF_LCG_MUL` / `_DIFF_LCG_INC`
// / `_DIFF_LINE_NUMBER_WIDTH`.
const LCG_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const LCG_MUL: u64 = 6364136223846793005;
const LCG_INC: u64 = 1442695040888963407;
const LINE_NUMBER_WIDTH: usize = 6;

fn near_identical_pair(target_bytes: usize) -> (String, String) {
    let a = prose(target_bytes);
    // split('\n') — NOT str::lines(): lines() drops the final trailing-empty
    // elements and would diverge from the Python twin's split("\n") corpus.
    let lines: Vec<&str> = a.split('\n').collect();
    let n = lines.len();
    let replace_at: [usize; 4] = REPLACE_FRACTIONS.map(|(num, den)| num * n / den);
    let delete_at = DELETE_FRACTION.0 * n / DELETE_FRACTION.1;
    let insert_at = INSERT_FRACTION.0 * n / INSERT_FRACTION.1;
    let mut out: Vec<String> = Vec::with_capacity(n + 1);
    for (idx, line) in lines.iter().enumerate() {
        if replace_at.contains(&idx) {
            out.push(line.replace(WORD_SWAP_OLD, WORD_SWAP_NEW));
        } else if idx == delete_at {
            continue;
        } else {
            out.push((*line).to_string());
        }
        if idx == insert_at {
            out.push(PROSE_SENTENCE.replace(WORD_SWAP_OLD, WORD_SWAP_NEW));
        }
    }
    (a, out.join("\n"))
}

fn shuffled_pair(target_bytes: usize) -> (String, String) {
    let a = prose(target_bytes);
    let mut lines: Vec<String> = a
        .split('\n')
        .enumerate()
        .map(|(idx, line)| {
            if line.is_empty() {
                String::new()
            } else {
                format!("{idx:0w$} {line}", w = LINE_NUMBER_WIDTH)
            }
        })
        .collect();
    let first = lines.join("\n");
    // Deterministic Fisher-Yates over the numbered lines — the exact loop
    // tests/reference.py mirrors (textually pinned by the corpus-parity test).
    let mut state = LCG_SEED;
    for i in (1..lines.len()).rev() {
        state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
        let j = (state % (i as u64 + 1)) as usize;
        lines.swap(i, j);
    }
    (first, lines.join("\n"))
}

fn bench_diff(c: &mut Criterion) {
    let mut group = c.benchmark_group("diff_opcodes");
    for target_bytes in [256 * 1024, 1024 * 1024, 12 * 1024 * 1024, 100 * 1024 * 1024] {
        let (a, b) = near_identical_pair(target_bytes);
        group.throughput(Throughput::Bytes(a.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("near-identical", format!("{}B", a.len())),
            &(a, b),
            |bench, (a, b)| bench.iter(|| diff_impl::diff_opcodes(black_box(a), black_box(b))),
        );
    }
    for target_bytes in [256 * 1024, 1024 * 1024, 12 * 1024 * 1024] {
        let (a, b) = shuffled_pair(target_bytes);
        group.throughput(Throughput::Bytes(a.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("shuffled", format!("{}B", a.len())),
            &(a, b),
            |bench, (a, b)| bench.iter(|| diff_impl::diff_opcodes(black_box(a), black_box(b))),
        );
    }
    group.finish();
}

fn bench_diff_opcodes_lines(c: &mut Criterion) {
    let mut group = c.benchmark_group("diff_opcodes_lines");
    for target_bytes in [256 * 1024, 1024 * 1024, 12 * 1024 * 1024, 100 * 1024 * 1024] {
        let (a, b) = near_identical_pair(target_bytes);
        group.throughput(Throughput::Bytes(a.len() as u64));
        // The near-identical cells' shape at the line spelling: the whole
        // core call — split_keepend_lines tokenization plus the Myers search,
        // no deadline. The op count is line-level and tiny at every size, so
        // a plain default-sampling cell suffices — no leg needs a reduction.
        group.bench_with_input(
            BenchmarkId::new("near-identical", format!("{}B", a.len())),
            &(a, b),
            |bench, (a, b)| {
                bench.iter(|| {
                    diff_impl::diff_opcodes_lines_deadline(black_box(a), black_box(b), None)
                        .expect("no deadline is set, so the search cannot expire")
                })
            },
        );
    }
    group.finish();
}

// The metrics pair's edit positions: five scattered single-char
// substitutions at odd letter offsets — every position lands inside a word
// (none on a boundary space), so each edit is a letter swap a fuzzy metric
// must find.
const METRICS_EDIT_POSITIONS: [usize; 5] = [7, 19, 31, 45, 57];

// The metrics pair's size: the fuzzy-matching workload's ~64-char token
// pair, not a corpus.
const METRICS_PAIR_CHARS: usize = 64;

// The get_close_matches query: a 41-char compound of five dense prose
// words — the long-token-against-a-dictionary shape.
const CLOSE_MATCHES_WORD: &str = "quarterlyintervalspecificationschangedoil";

// The dictionary size: the bulk candidate scan at a size where the
// per-candidate scoring pass is the measurable cost.
const CLOSE_MATCHES_DICTIONARY: usize = 10_000;

fn metrics_pair() -> (String, String) {
    let base: String = PROSE_SENTENCE.chars().take(METRICS_PAIR_CHARS).collect();
    let mut edited: Vec<char> = base.chars().collect();
    for at in METRICS_EDIT_POSITIONS {
        // 'x' unless the position already holds one (then 'q'): every edit
        // changes the char, so all five edits are real.
        edited[at] = if edited[at] == 'x' { 'q' } else { 'x' };
    }
    (base, edited.into_iter().collect())
}

// The dictionary: the prose corpus's own vocabulary (the pinned corpus
// repeats ~23 distinct whitespace tokens), each entry a vocabulary word
// plus the base-26 rendering of its index — three letters cover
// 26^3 = 17,576 > 10,000, so every (word, suffix) pair is unique by
// construction (whatever the vocabulary size): no generator, no dedup
// pass, and the dictionary is the same bytes on every run. A
// random-draw-and-dedup builder is deliberately avoided: the LCG's
// consecutive states are affinely related, so tuples drawn from them cover
// only a fraction of the (word, suffix) space and the dedup loop stalls
// short of 10k forever.
fn close_matches_dictionary() -> Vec<String> {
    let mut vocabulary: Vec<&str> = Vec::new();
    let corpus = prose(64 * 1024);
    for word in corpus.split_whitespace() {
        if !vocabulary.contains(&word) {
            vocabulary.push(word);
        }
    }
    (0..CLOSE_MATCHES_DICTIONARY)
        .map(|i| {
            let base = vocabulary[i % vocabulary.len()];
            let mut word = String::from(base);
            for digit in [i / 676, (i / 26) % 26, i % 26] {
                word.push((b'a' + digit as u8) as char);
            }
            word
        })
        .collect()
}

fn bench_metrics(c: &mut Criterion) {
    // The pair bound ONCE: every cell below drives the same tuple, so the
    // three numbers cross-reference on identical operands.
    let pair = metrics_pair();
    let mut group = c.benchmark_group("metrics");
    group.throughput(Throughput::Bytes((pair.0.len() + pair.1.len()) as u64));
    let label = format!("{}-char-pair", METRICS_PAIR_CHARS);
    // One whole-core call per iteration on the SAME pair for all three
    // metrics, so the numbers cross-reference: the two-row DP
    // (levenshtein), the match/transposition passes plus the prefix boost
    // (jaro_winkler), and the Myers search plus the 2M/T arithmetic
    // (similarity_ratio). Fixed-size by nature — the workload is a token
    // pair — so no ladder; default sampling.
    group.bench_with_input(
        BenchmarkId::new("levenshtein", &label),
        &pair,
        |bench, (a, b)| {
            bench.iter(|| {
                fuzzy_impl::levenshtein(black_box(a), black_box(b), None)
                    .expect("a None deadline can never be exceeded")
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("jaro_winkler", &label),
        &pair,
        |bench, (a, b)| {
            bench.iter(|| {
                fuzzy_impl::jaro_winkler(black_box(a), black_box(b), None)
                    .expect("a None deadline can never be exceeded")
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("similarity_ratio", &label),
        &pair,
        |bench, (a, b)| bench.iter(|| diff_impl::similarity_ratio(black_box(a), black_box(b))),
    );
    group.finish();
}

fn bench_close_matches(c: &mut Criterion) {
    let dictionary = close_matches_dictionary();
    let candidates: Vec<&str> = dictionary.iter().map(String::as_str).collect();
    let mut group = c.benchmark_group("get_close_matches");
    group.throughput(Throughput::Bytes(
        (CLOSE_MATCHES_WORD.len() + dictionary.iter().map(String::len).sum::<usize>()) as u64,
    ));
    // The long-token-against-short-words shape is the under-cutoff miss
    // scan: no candidate can clear 0.6 (2M/T caps at ~0.59 for a 17-char
    // word against the 41-char query), so the iteration is the pure
    // per-candidate scoring pass — a Myers search plus ratio arithmetic
    // per word — at difflib's documented n=3/cutoff=0.6 defaults. Default
    // sampling.
    group.bench_with_input(
        BenchmarkId::new(
            "prose",
            format!("{}-word-dictionary", CLOSE_MATCHES_DICTIONARY),
        ),
        &candidates,
        |bench, candidates| {
            bench.iter(|| {
                diff_impl::close_matches(
                    black_box(CLOSE_MATCHES_WORD),
                    black_box(candidates),
                    3,
                    0.6,
                    None,
                )
                .expect("a None deadline can never be exceeded")
            })
        },
    );
    group.finish();
}

criterion_group!(
    benches,
    bench_diff,
    bench_diff_opcodes_lines,
    bench_metrics,
    bench_close_matches
);
criterion_main!(benches);
