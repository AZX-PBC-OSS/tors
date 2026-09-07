//! `diff_opcodes` never panics on any pair of strings, and the opcodes
//! reconstruct both sides: contiguous, covering, alternating, per-tag
//! non-empty, equal ops holding equal content (the structural contract the
//! Python hypothesis property pins, checked here at raw-byte depth).
//!
//! Input size is capped (the Myers search is superlinear on hard pairs),
//! so the fuzzer explores deep small pairs instead of stalling on huge
//! ones.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    a: String,
    b: String,
}

fuzz_target!(|input: Input| {
    if input.a.len() + input.b.len() > 4096 {
        return;
    }
    let ops = tors::diff_impl::diff_opcodes(&input.a, &input.b);
    let a: Vec<char> = input.a.chars().collect();
    let b: Vec<char> = input.b.chars().collect();
    let mut rebuilt_a = String::with_capacity(input.a.len());
    let mut rebuilt_b = String::with_capacity(input.b.len());
    let mut prev = (0, 0);
    let mut prev_equal: Option<bool> = None;
    for op in &ops {
        assert_eq!((op.i1, op.j1), prev, "contiguity broken: {op:?}");
        assert!(
            op.i2 <= a.len() && op.j2 <= b.len(),
            "op out of bounds: {op:?}"
        );
        let is_equal = op.tag == tors::diff_impl::OpcodeTag::Equal;
        assert!(
            prev_equal != Some(is_equal),
            "tags do not alternate: {op:?}"
        );
        match op.tag {
            tors::diff_impl::OpcodeTag::Equal => {
                assert!(op.i1 < op.i2 && op.j1 < op.j2, "empty equal op: {op:?}");
                assert_eq!(
                    op.i2 - op.i1,
                    op.j2 - op.j1,
                    "unequal equal lengths: {op:?}"
                );
                let left: String = a[op.i1..op.i2].iter().collect();
                let right: String = b[op.j1..op.j2].iter().collect();
                assert_eq!(left, right, "equal content mismatch: {op:?}");
                rebuilt_a.push_str(&left);
                rebuilt_b.push_str(&right);
            }
            tors::diff_impl::OpcodeTag::Delete => {
                assert!(op.i1 < op.i2, "empty delete op: {op:?}");
                rebuilt_a.push_str(&a[op.i1..op.i2].iter().collect::<String>());
            }
            tors::diff_impl::OpcodeTag::Insert => {
                assert!(op.j1 < op.j2, "empty insert op: {op:?}");
                rebuilt_b.push_str(&b[op.j1..op.j2].iter().collect::<String>());
            }
            tors::diff_impl::OpcodeTag::Replace => {
                assert!(op.i1 < op.i2 && op.j1 < op.j2, "empty replace op: {op:?}");
                rebuilt_a.push_str(&a[op.i1..op.i2].iter().collect::<String>());
                rebuilt_b.push_str(&b[op.j1..op.j2].iter().collect::<String>());
            }
        }
        prev = (op.i2, op.j2);
        prev_equal = Some(is_equal);
    }
    assert_eq!(prev, (a.len(), b.len()), "coverage broken");
    assert_eq!(rebuilt_a, input.a, "a not reconstructed");
    assert_eq!(rebuilt_b, input.b, "b not reconstructed");
});
