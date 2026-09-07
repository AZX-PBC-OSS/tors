//! `word_bounds`/`sentence_bounds` never panic on arbitrary text, cover
//! [0, char_len) exactly (contiguous, no gaps, no overlaps, no empty
//! segments), and joining the segments reproduces the input.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|s: &str| {
    check_bounds(&tors::segmentation_impl::word_bounds(s), s, "word");
    check_bounds(&tors::segmentation_impl::sentence_bounds(s), s, "sentence");
});

fn check_bounds(bounds: &[(usize, usize)], s: &str, kind: &str) {
    let char_len = s.chars().count();
    let mut prev_end = 0;
    for &(start, end) in bounds {
        assert_eq!(
            start, prev_end,
            "{kind} bounds not contiguous at ({start}, {end})"
        );
        assert!(start < end, "empty {kind} segment: ({start}, {end})");
        assert!(
            end <= char_len,
            "{kind} segment end {end} past char_len {char_len}"
        );
        prev_end = end;
    }
    assert_eq!(prev_end, char_len, "{kind} bounds do not cover the text");
    // Join-back walks the text's chars once per segment; cap it so large
    // inputs keep exploring segmentation instead of re-verifying one huge
    // rebuild.
    if s.len() < 4096 {
        let mut rebuilt = String::with_capacity(s.len());
        let mut chars = s.chars();
        for &(start, end) in bounds {
            for _ in start..end {
                rebuilt.push(
                    chars
                        .next()
                        .expect("bounds outran the text's char iterator"),
                );
            }
        }
        assert_eq!(rebuilt, s, "{kind} join-back diverged from the input");
    }
}
