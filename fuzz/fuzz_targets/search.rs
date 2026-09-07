//! `find_patterns`/`replace_many`/`replace_many_masked` never panic on
//! arbitrary non-empty pattern lists and arbitrary text; every reported
//! match really holds its pattern at its reported codepoint span, matches
//! never overlap or regress, and the masked spelling preserves the char
//! count exactly (the invariants the Python gates pin, checked here at
//! raw-byte depth).

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    text: String,
    patterns: Vec<String>,
}

fuzz_target!(|input: Input| {
    // Empty patterns have no leftmost-longest meaning (the wrapper refuses
    // them); everything else is fair game, including duplicates.
    let patterns: Vec<&str> = input
        .patterns
        .iter()
        .map(String::as_str)
        .filter(|p| !p.is_empty())
        .collect();
    if patterns.is_empty() {
        return;
    }

    let matches = tors::search_impl::find_patterns(&patterns, &input.text)
        .expect("engine limits are unreachable at fuzz sizes");
    let char_len = input.text.chars().count();
    let mut prev_end = 0;
    for m in &matches {
        assert!(m.start >= prev_end, "matches overlap or regress: {m:?}");
        assert!(m.start < m.end, "empty match: {m:?}");
        assert!(m.end <= char_len, "match end past text end: {m:?}");
        assert!(
            m.pattern < patterns.len(),
            "pattern index out of range: {m:?}"
        );
        prev_end = m.end;
    }
    // The span-content check slices the text by codepoint index; cap it so
    // large inputs keep exploring the scan instead of re-verifying one
    // huge reconstruction.
    if input.text.len() < 4096 {
        let chars: Vec<char> = input.text.chars().collect();
        for m in &matches {
            let matched: String = chars[m.start..m.end].iter().collect();
            assert_eq!(
                matched, patterns[m.pattern],
                "match span does not hold its pattern: {m:?}"
            );
        }
    }

    // The replace spellings share the automaton; values rotate the pattern
    // list so key/value length skew is exercised too.
    let replacements: Vec<(&str, &str)> = patterns
        .iter()
        .enumerate()
        .map(|(i, p)| (*p, patterns[(i + 1) % patterns.len()]))
        .collect();
    let _ = tors::search_impl::replace_many(&input.text, &replacements)
        .expect("engine limits are unreachable at fuzz sizes");
    let mask = input.text.chars().next().unwrap_or('*');
    let masked = tors::search_impl::replace_many_masked(&input.text, &replacements, mask)
        .expect("engine limits are unreachable at fuzz sizes");
    assert_eq!(
        masked.chars().count(),
        input.text.chars().count(),
        "masked replace changed the char count"
    );
});
