//! `first_invalid_charset`/`first_invalid_offender` never panic and the
//! answers agree with the single-item question: an item the scan calls
//! valid, run alone under the same sets, must come back valid, and the
//! items BEFORE the named offender must all be valid alone (the scan's
//! own first-invalid monotonicity). The set grammar: a literal charset
//! string, an optional `first` set layered over the `rest` set —
//! exercised over raw fuzzer strings for both sets and the items.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tors::charset_impl::{FirstInvalid, scan_first_invalid};

fuzz_target!(|data: &[u8]| {
    // Split the fuzzer bytes into the three arguments: the charset spell
    // (rest set), an optional first-set spell, and 1-2 items.
    let text = String::from_utf8_lossy(data).into_owned();
    let words: Vec<&str> = text.split('\u{0}').take(4).collect();
    if words.len() < 2 {
        return; // need at least a charset spell and one item
    }
    let (rest, first, items) = match words.len() {
        2 => (words[0], None, vec![words[1]]),
        3 => (words[0], Some(words[1]), vec![words[2]]),
        _ => (words[0], Some(words[1]), vec![words[2], words[3]]),
    };
    let items_ref: Vec<&str> = items.iter().copied().collect();
    let verdict = scan_first_invalid(&items_ref, first, rest);

    // The answer cross-checks the single-item question item by item.
    for (idx, item) in items_ref.iter().enumerate() {
        let alone = scan_first_invalid(std::slice::from_ref(item), first, rest);
        match (&verdict, &alone) {
            (FirstInvalid::Clean, FirstInvalid::Clean) => {}
            (FirstInvalid::Clean, FirstInvalid::Offender { .. }) => {
                panic!(
                    "valid list, but item {idx} alone is invalid: item {idx} alone answered invalid"
                )
            }
            (FirstInvalid::Offender { item: v_item, .. }, FirstInvalid::Clean)
                if *v_item == idx =>
            {
                panic!("the scan names item {idx} invalid but it is valid alone: the verdict")
            }
            _ => {}
        }
    }
    // The first-invalid INDEX: the items before the named one are valid
    // alone (the scan's own monotonicity contract).
    if let FirstInvalid::Offender { item: bad, .. } = &verdict {
        for (idx, item) in items_ref.iter().enumerate().take(*bad) {
            assert!(
                matches!(
                    scan_first_invalid(std::slice::from_ref(item), first, rest),
                    FirstInvalid::Clean
                ),
                "an item BEFORE the named offender is itself invalid: item {idx} in the verdict"
            );
        }
    }
});
