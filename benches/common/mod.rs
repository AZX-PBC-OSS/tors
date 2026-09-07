//! The shared deterministic corpus recipes — the single Rust-side source for
//! every bench file (`normalize.rs`, `bytes.rs`, `text.rs` all do `mod
//! common;`), so the previously triplicated copies cannot drift apart. The
//! Python↔Rust identity is enforced by `tests/test_bench_corpus_parity.py`,
//! which parses THIS file (not each bench) and asserts byte-equality of both
//! the sentence constants and the built corpora against `tests/reference.py`
//! — the suite's single source for the same recipes.
//!
//! Same recipes as the Python suite (`tests/reference.py`, the deterministic
//! prose-corpus idiom): plain prose, decomposed-accent prose, and the CRLF
//! variant. Sizes are UTF-8 bytes, unit-quantized (a 12 MiB target lands
//! within one unit of 12 MiB). The text-bench-only recipes (`compat`,
//! `entities`) stay in `benches/text.rs`, the one bench that uses them.

pub const PROSE_SENTENCE: &str = "The quarterly oil sample interval for field outages was adjusted after the bushing torque specifications changed. Maintenance windows now close within fourteen days. ";

// Same sentence with three decomposed accents (base letter + U+0301) — mirrors
// tests/reference.py's `_DECOMPOSED_SENTENCE` so bench and test numbers are comparable.
const DECOMPOSED_SENTENCE: &str = "The quarte\u{0301}rly oil sa\u{0301}mple interval for field outa\u{0301}ges was adjusted after the bushing torque specifications changed. Maintenance windows now close within fourteen days. ";

fn repeat_to(target_bytes: usize, unit: &str) -> String {
    unit.repeat((target_bytes / unit.len()).max(1))
}

pub fn prose(target_bytes: usize) -> String {
    repeat_to(
        target_bytes,
        &format!("{}{}", PROSE_SENTENCE.repeat(4), "\n\n"),
    )
}

pub fn decomposed(target_bytes: usize) -> String {
    repeat_to(
        target_bytes,
        &format!("{}{}", DECOMPOSED_SENTENCE.repeat(4), "\n\n"),
    )
}

pub fn crlf(target_bytes: usize) -> String {
    repeat_to(
        target_bytes,
        &format!("{}{}", PROSE_SENTENCE.repeat(4), " \t\r\n\r\n"),
    )
}
