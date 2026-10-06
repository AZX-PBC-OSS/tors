//! `tors::json_repair::StreamingRepairer` (the streaming repairer; see
//! src/json_repair/streaming.rs for the contract) never panics on
//! arbitrary chunk sequences, and its closed renders stay loadable. The
//! input is an arbitrary string replayed as a chunk sequence (several
//! deterministic split sets per input, boundaries mid-escape and
//! mid-surrogate-pair included); the asserts:
//!
//! - panic-freedom through push -> push -> end (the primary; Err is
//!   fine: the depth cap's normalized ValueError, after which the
//!   machine's state is unspecified and that splitting just stops);
//! - on every error-free path: every snapshot (and the end) is the
//!   empty sentinel or strict-loadable JSON, and snapshot == end on the
//!   same prefix;
//! - chunking invariance: any split set ends at the one-push answer;
//! - on the strict-loadable class (valid JSON text, within the depth
//!   cap), the streamed end() equals the whole-text engine's repair
//!   byte for byte — the same output invariant the whole-text target
//!   pins, plus the streaming agreement.
//!
//! Panics, non-loadable renders, snapshot/end disagreement, or
//! chunking drift are bugs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tors::json_repair::{RepairConfig, StreamingRepairer, Value, dumps, loads_strict, repair};

fuzz_target!(|s: &str| {
    // Several deterministic split sets per input: one-push (no splits),
    // three-char chunks (escape/pair straddling), single-char pushes,
    // and a strided scatter. Every assertion below runs per split set.
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let split_sets: Vec<Vec<usize>> = vec![
        Vec::new(),
        (0..n).step_by(3).collect(),
        (0..n).collect(),
        (0..n).filter(|i| i % 7 == 3).collect(),
    ];

    // The whole-text spelling's answer: the empty-string sentinel
    // renders bare (the repaired value Str("") is the nothing-
    // recoverable spelling), exactly what repair_json returns.
    let engine_out = match repair(s, &RepairConfig::default()) {
        Ok((Value::Str(empty), _)) if empty.is_empty() => Some(String::new()),
        Ok((v, _)) => Some(dumps(&v, true)),
        Err(_) => None,
    };
    // The one-push answer (None when the depth cap fired: the state is
    // unspecified after an error, so error-free splittings compare only
    // against a None-free reference).
    let one_out = (|| {
        let mut once = StreamingRepairer::new(true);
        once.push(s).ok()?;
        once.end().ok()
    })();

    for splits in &split_sets {
        let mut r = StreamingRepairer::new(true);
        let mut prev = 0usize;
        let mut errored = false;
        for &p in splits {
            let p = p.min(n);
            if p < prev {
                continue;
            }
            let chunk: String = chars[prev..p].iter().collect();
            prev = p;
            if r.push(&chunk).is_err() {
                errored = true;
                break;
            }
            // Snapshot is loadable-or-sentinel at every point, and equals
            // end on the same prefix (the machine's own render, pure).
            let snap = r.snapshot();
            assert!(
                snap.is_empty() || loads_strict(&snap).is_ok(),
                "snapshot is not strict-loadable: {snap}"
            );
            assert_eq!(r.end().expect("end never fails"), snap, "snapshot != end");
        }
        if errored {
            continue;
        }
        let tail: String = chars[prev..].iter().collect();
        if r.push(&tail).is_err() {
            continue;
        }
        let final_text = r.end().expect("end never fails");
        assert!(
            final_text.is_empty() || loads_strict(&final_text).is_ok(),
            "end() is not strict-loadable: {final_text}"
        );
        if let Some(want) = &one_out {
            assert_eq!(&final_text, want, "end() drifted across chunkings");
        }
    }

    // The strict-loadable class: valid JSON text (within the depth cap,
    // where the engine itself succeeds) streams to the whole-text
    // engine's exact answer under every split set.
    if loads_strict(s).is_ok()
        && let Some(want) = engine_out
    {
        for splits in &split_sets {
            let mut r = StreamingRepairer::new(true);
            let mut prev = 0usize;
            let mut ok = true;
            for &p in splits {
                let p = p.min(n);
                if p < prev {
                    continue;
                }
                let chunk: String = chars[prev..p].iter().collect();
                prev = p;
                if r.push(&chunk).is_err() {
                    // Valid JSON past the 200-container cap: the engine's
                    // own normalized guard fired on both sides; skip.
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            let tail: String = chars[prev..].iter().collect();
            if r.push(&tail).is_err() {
                continue;
            }
            assert_eq!(
                r.end().expect("end never fails"),
                want,
                "streamed valid JSON != engine on split set {splits:?}"
            );
        }
    }
});
