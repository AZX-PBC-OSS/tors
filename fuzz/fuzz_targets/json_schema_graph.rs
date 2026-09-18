//! `repair_json`'s schema walk never panics on a schema-graph-hostile
//! input, terminates within the documented walk caps, and surfaces every
//! refusal as a catchable `Err` — fuzzed from the Rust core
//! (`tors::json_repair::repair`), where the schema is a `Value` tree the
//! fuzzer builds directly through a builder with its own depth, width,
//! and aliasing knobs.
//!
//! A TYPE-SYSTEM NOTE that scopes this target: the shared-reference and
//! CYCLE amplification of issue #113 is a Python-object-graph phenomenon
//! (`py_to_value` walks the caller's dict/list graph, where refs and
//! cycles exist). A Rust-side `Value` tree is owned, acyclic, and
//! clone-duplicated by construction, so no Rust target can express the
//! cycle shape; the Python-side graph generator (shared refs, true
//! cycles, doubling builders) lives in the pytest suite instead
//! (tests/test_memory_spike_guards.py). What THIS target fuzzes against
//! the same walk caps is the tree-shaped neighborhood of that class:
//! deep nesting past the 200-unit cap (which must refuse with a clean
//! Err, never recurse to a stack overflow), wide property maps, alias
//! shapes (one subtree duplicated into many slots — the expanded form a
//! shared ref yields on the Python side), wide enums, and deep-vs-wide
//! combinations, each against arbitrary document text.
//!
//! DELIBERATELY NOT ASSERTED: deadline compliance of the enum-suggestion
//! path. Issue #115 is LIVE on this tree — `closest_enum_member` scores
//! every member with `jaro_winkler` and never consults the armed clock,
//! so an enum-miss repair runs past an armed `deadline_ms` (measured:
//! 2000 members × 1500-char strings → TimeoutError ~160x past a 5 ms
//! budget; the repro is in tests/redteam_corpus/, `issue-115`). Once the
//! fix lands, add the deadline invariant here (assert the call's own
//! wall stays under a generous multiple of the budget on wide enums) —
//! a failing assert before the fix would make this a known-crashing
//! target, the time-bomb shape the Makefile's documents_markdown note
//! bans from the run lists.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tors::json_repair::{NumericLocale, RepairConfig, Value, repair};

/// The documented schema-nesting cap (src/json_schema_impl.rs
/// MAX_SCHEMA_DEPTH = 200 nesting units). The builder's Deep shape costs
/// two units per level (`properties` -> the named subschema), so past
/// MAX_DEPTH_UNITS / 2 builder levels the refusal must fire.
const MAX_DEPTH_UNITS: usize = 200;

#[derive(Arbitrary, Debug)]
struct Input {
    /// The raw document text (arbitrary JSON-ish bytes: the repair parser
    /// is the other half of every call).
    doc: String,
    shape: Shape,
    /// Builder knobs, modded to useful ranges at build time.
    knob_a: u16,
    knob_b: u16,
    /// Whether the config arms a small deadline (the live #115 hole means
    /// this must NOT gate any assertion yet; the lane exists so the
    /// no-panic/termination contract holds under BOTH armaments, and so
    /// the deadline invariant slots in without reshaping the target).
    arm_deadline: bool,
    salvage: bool,
    diagnostics: bool,
}

#[derive(Arbitrary, Debug)]
enum Shape {
    /// `properties`-nested `{"a": …}` `d` levels deep, `d` spanning both
    /// sides of the 200-unit cap.
    Deep,
    /// `w` properties of alternating scalar/subschema types.
    Wide,
    /// One property subschema cloned into `k` slots (the expanded form a
    /// Python shared ref yields), with `w` distinct neighbors around it.
    Aliased,
    /// A wide `enum` list (the #115 scorer's input shape, kept small).
    EnumWide,
    /// Deep nesting INSIDE each of `w` wide properties.
    DeepWide,
}

fn scalar_schema(kind: usize) -> Value {
    let ty = match kind % 5 {
        0 => "string",
        1 => "integer",
        2 => "number",
        3 => "boolean",
        _ => "null",
    };
    Value::Object(vec![("type".to_string(), Value::Str(ty.to_string()))])
}

fn deep_schema(levels: usize) -> Value {
    let mut node = scalar_schema(0);
    for _ in 0..levels {
        node = Value::Object(vec![(
            "properties".to_string(),
            Value::Object(vec![("a".to_string(), node)]),
        )]);
    }
    node
}

fn build_schema(shape: &Shape, knob_a: u16, knob_b: u16) -> (Value, usize) {
    // (schema, the builder's own nesting-unit estimate for the Deep shape
    // — 0 for shapes that cannot exceed the cap at these knob ranges).
    match shape {
        Shape::Deep => {
            let levels = (knob_a as usize % 260) + 60; // 60..=320 levels: 120..=640 units
            (deep_schema(levels), levels * 2)
        }
        Shape::Wide => {
            let w = (knob_a as usize % 96) + 1;
            let props: Vec<(String, Value)> = (0..w)
                .map(|i| (format!("k{i}"), scalar_schema(i)))
                .collect();
            (
                Value::Object(vec![
                    ("type".to_string(), Value::Str("object".to_string())),
                    ("properties".to_string(), Value::Object(props)),
                ]),
                0,
            )
        }
        Shape::Aliased => {
            let w = (knob_a as usize % 32) + 1;
            let k = (knob_b as usize % 8) + 1;
            let mut props: Vec<(String, Value)> = (0..w)
                .map(|i| (format!("n{i}"), scalar_schema(i)))
                .collect();
            // The alias shape: ONE subtree's value cloned into k slots —
            // content-identical members, the expanded form a shared ref
            // presents on the Python side of the walk.
            let shared = deep_schema(knob_b as usize % 24);
            for j in 0..k {
                props.push((format!("alias{j}"), shared.clone()));
            }
            (
                Value::Object(vec![
                    ("type".to_string(), Value::Str("object".to_string())),
                    ("properties".to_string(), Value::Object(props)),
                ]),
                0,
            )
        }
        Shape::EnumWide => {
            let w = (knob_a as usize % 64) + 1;
            let len = (knob_b as usize % 24) + 1;
            let members: Vec<Value> = (0..w)
                .map(|i| Value::Str(format!("{}{}", "a".repeat(len), i)))
                .collect();
            (
                Value::Object(vec![("enum".to_string(), Value::Array(members))]),
                0,
            )
        }
        Shape::DeepWide => {
            let w = (knob_a as usize % 24) + 1;
            let d = (knob_b as usize % 80) + 60; // stays under the cap (120 units)
            let props: Vec<(String, Value)> =
                (0..w).map(|i| (format!("k{i}"), deep_schema(d))).collect();
            (
                Value::Object(vec![
                    ("type".to_string(), Value::Str("object".to_string())),
                    ("properties".to_string(), Value::Object(props)),
                ]),
                0,
            )
        }
    }
}

fuzz_target!(|input: Input| {
    let (schema, deep_units) = build_schema(&input.shape, input.knob_a, input.knob_b);
    let cfg = RepairConfig {
        skip_json_loads: false,
        strict: false,
        salvage: input.salvage,
        schema: Some(schema),
        diagnostics: input.diagnostics,
        locale: NumericLocale::Auto,
        deadline_ms: input.arm_deadline.then_some(5.0),
        deadline_clock: None,
    };

    // The core contract: the call TERMINATES (implicit — a hang fails the
    // run's own timeout) and answers with Ok or a catchable Err, never a
    // panic, for arbitrary document text against every builder shape.
    let outcome = repair(&input.doc, &cfg);

    // The walk-cap refusal: a schema past the nesting cap must be a clean
    // Err (upstream surfaces deep schemas as errors from validation, never
    // a recursion crash) — and the builder's own unit count is the oracle
    // for "past the cap".
    if deep_units > MAX_DEPTH_UNITS {
        assert!(
            outcome.is_err(),
            "a schema {deep_units} nesting units deep (cap {MAX_DEPTH_UNITS}) repaired \
             without refusing: doc={:?}",
            input.doc
        );
    }

    // The diagnostics lane records at most the actions it took: nothing
    // here may panic on either lane, and the Ok value (when present) is
    // inspectable without asserts — the run's own invariants above are
    // the contract; this arm exists to keep the diagnostics machinery
    // exercised on the same inputs.
    let _ = outcome;
});
