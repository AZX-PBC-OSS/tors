//! `tors::json_repair` (the json_repair port; see src/json_repair/mod.rs
//! for provenance) never panics on arbitrary strings. The panic surfaces
//! are the repair parser's `Vec<char>` splices (the duplicate-key split
//! and the empty-object reparse both rewrite the input buffer in place),
//! the negative-wrap index arithmetic of `Parser::get`, and
//! parse_string's lookahead scans — plus the output invariant: a
//! successful repair's value must render through `dumps` to "" (the
//! nothing-recoverable sentinel's spelling) or to text `loads_strict`
//! accepts, on every config spelling that produced it. Err results are
//! fine (strict-mode and depth-cap ValueErrors); only panics and
//! non-loadable outputs are bugs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tors::json_repair::{RepairConfig, Value, dumps, loads_strict, repair};

fuzz_target!(|s: &str| {
    // Default config: the strict fast path for valid JSON, the heuristic
    // engine otherwise. Err is fine; a successful repair must satisfy the
    // output invariant.
    if let Ok((value, _)) = repair(s, &RepairConfig::default()) {
        assert_output_invariant(&value);
    }

    // The strict parser alone: Err is its whole job, panic is never one.
    let _ = loads_strict(s);

    // skip_json_loads disables the whole-input fast path, forcing the same
    // input through the repair engine's splices and scans — never panics,
    // Err fine, and its output meets the same invariant.
    let engine_only = RepairConfig {
        skip_json_loads: true,
        ..RepairConfig::default()
    };
    if let Ok((value, _)) = repair(s, &engine_only) {
        assert_output_invariant(&value);
    }
});

/// The output invariant: a successful repair's value must render via
/// `dumps` to "" or to text `loads_strict` accepts — repair's output is
/// always loadable JSON. (The empty spelling is the py layer's
/// nothing-recoverable sentinel; `dumps` itself never emits fewer bytes
/// than a quoted empty string, so this arm states the contract, not the
/// serializer's current floor.)
fn assert_output_invariant(value: &Value) {
    let rendered = dumps(value, true);
    assert!(
        rendered.is_empty() || loads_strict(&rendered).is_ok(),
        "repair output is not strict-loadable: {rendered}"
    );
}
