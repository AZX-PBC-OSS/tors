//! Escape-parity byte scan, the pure-Rust core of `tors.contains_unescaped`
//! and `tors.find_unescaped`: find the first occurrence of a needle that is
//! not itself escaped.
//!
//! Semantics: an occurrence of `needle` at offset `i` in `haystack` is
//! **live** exactly when the maximal run of `\` immediately before `i` has
//! even length. An empty run is even, so an occurrence at offset 0 is live.
//! An odd run means the run's backslash pairs escape each other and the
//! leftover one escapes the needle's first byte: the occurrence is literal
//! text, not the sequence — **rejected** — and the scan resumes ONE BYTE
//! past the hit, not past the whole match, so self-overlapping needles stay
//! correct: the two-byte needle `00` in a haystack of one backslash then
//! `000` rejects the hit at 1 (one backslash before it) and finds the
//! overlapping live hit at 2 (the byte before it is the rejected match's
//! own `0`). memmem's `find_iter` resumes at match end and would skip that
//! hit entirely, which is why this core drives `Finder::find` over
//! re-slices in its own loop instead of taking the iterator.
//!
//! # Why this core exists, and why it is public
//!
//! The motivating case is JSON, and it is byte-ambiguous by construction:
//! an encoder (orjson is the measured consumer) renders a real NUL
//! codepoint U+0000 as the six-byte escape text `\u0000` and the literal
//! six-character text of the same spelling as seven bytes (the backslash
//! itself escaped), and the second contains the first at offset +1 — so a
//! plain substring test answers "yes" for both. PostgreSQL settles it
//! downstream: a real NUL is fatal in a `jsonb` column (SQLSTATE 22P05,
//! `jsonb_in` cannot put it in the decoded text), the literal text is
//! fine, so a pipeline that binds serialized JSON must know which one it
//! holds before the INSERT. Get it wrong in either direction and you
//! either reject legal text or ship a value that fails the bind. The
//! consumer's original guard confirmed every prefilter hit by re-parsing
//! the whole value and recursively walking it — the walk existed only
//! because nothing could answer "is this occurrence escaped?" — and the
//! parity rule (proven there against the re-parse walk over an adversarial
//! matrix of backslash runs and 20k randomized payloads) answers it
//! directly from the raw bytes. It landed here, as a public GIL-free
//! primitive, because a hand-rolled loop in one consumer's private module
//! is usable by nobody else and holds the GIL for the whole scan.
//!
//! # No JSON knowledge lives here
//!
//! Parity is the mechanism; "the needle is an escape sequence, so even
//! means live and odd means literal" is the caller's reading of it. Any
//! backslash-escaped grammar (printf format strings, shell quotes, regex
//! sources) can drive the same scan. The tree already had the idiom twice,
//! both internal and both unfactorable into this shape:
//! `json_repair::parser`'s `skip_to_character` is char-unit and
//! cursor-relative (and telemetry-coupled, via `note_scan_distance`), and
//! `gfm_strip_impl`'s `find_unescaped_bracket_close` walks a char slice
//! against a precomputed escaped-at table. A fresh byte-level core over
//! arbitrary (haystack, needle) pairs is smaller than an abstraction that
//! would span those two, and it serves the general case.
//!
//! # The engine and the parity walk
//!
//! The occurrence scan is `memchr::memmem::Finder` (memchr is already a
//! direct dependency; the Finder holds the needle's search strategy, so
//! the per-hit loop reuses one build), and the parity work is a backward
//! walk over the backslash run immediately before each hit:
//!
//! * Cost: the walk for a hit covers that hit's maximal backslash run,
//!   and at most one hit per maximal run carries a nonzero walk. A hit
//!   with a nonzero walk sits at a maximal run's end or strictly inside
//!   it, and the two placements are mutually exclusive per needle: a
//!   hit at a run's end needs the needle's first byte to be a
//!   non-backslash (the byte there is what ends the run), while a hit
//!   strictly inside needs it to be a backslash, which forces the
//!   needle's backslash prefix to end exactly at the run's end — one
//!   position per run either way. Every walk stays inside its own run,
//!   and runs are disjoint, so every byte is walked backward at most
//!   once across the whole scan. An all-backslash needle never walks at
//!   all: its first hit sits at a maximal run's start — an even, empty
//!   run before it — and answers the scan there.
//! * No state is carried between hits. An earlier revision of this scan
//!   bounded each walk at the previous hit and carried the run's start
//!   forward; both devices were vestigial. The bound never bounded: the
//!   run before a hit never reaches back to the previous hit, because
//!   an all-backslash gap between two hits forces an all-backslash
//!   needle — the gap either swallows the needle whole, or its length
//!   is a period of the needle (both hits spell the needle over the
//!   same gap bytes) and a backslash-prefixed periodic needle is
//!   backslashes all the way through — and an all-backslash needle's
//!   first hit is always live at a run start, ending the scan before a
//!   second hit exists. With no gap ever all backslashes, the bounded
//!   walk and this walk take the same steps on every input, and the
//!   carried start's join branch never fired past the first hit, where
//!   it answered `0` — what the walk's own `run_back` already held.
//!   Verified before the removal by an instrumented differential over the
//!   exhaustive small-alphabet sweep plus a backslash-dense adversarial
//!   sweep (zero join firings past the first hit, zero re-walks, the two
//!   loops step-for-step identical); the linear-walk bound itself is pinned
//!   reproducibly by `backward_walk_budget_is_linear` below (total backward
//!   steps <= haystack length over the long-run and many-hit shapes), so no
//!   historical pair count is load-bearing here.
//!
//! # Preconditions (enforced by the wrapper before this runs)
//!
//! `needle` must be non-empty: an empty needle would match at every
//! position and has no parity meaning (there is no occurrence to stand
//! before); the wrapper refuses it with `ValueError("empty needle")`, the
//! same rationale as `find_patterns`' empty pattern. The haystack may be
//! empty or shorter than the needle (no occurrence, answer `None`). No
//! other input is invalid: arbitrary bytes are in-contract (the function
//! has no JSON assumptions to violate), which is what the fuzz target and
//! the arbitrary-bytes differential both lean on.
//!
//! Pure Rust, no pyo3 types: the criterion bench (the `unescaped_scan`
//! group in benches/search.rs) drives this path directly; the pyo3 wrapper
//! in `src/py/scan.rs` adds only the argument borrows, the GIL release,
//! and the `bool`/`int` return marshalling (see the crate GIL model in
//! `src/lib.rs`).

/// The byte offset of the first live (even-run) occurrence of `needle` in
/// `haystack` — the first occurrence whose maximal preceding backslash run
/// has even length — or `None` when no live occurrence exists (the wrapper
/// maps `None` to Python `-1`, `bytes.find`'s own sentinel). `needle` must
/// be non-empty (see the module docs' preconditions); the haystack's bytes
/// are never assumed to be anything but bytes.
///
/// Rejected (odd-run) hits advance the search one byte past the hit, not
/// past the whole match, so self-overlapping needles stay correct; each
/// hit's walk covers the backslash run immediately before it, and at
/// most one hit per maximal run ever walks, so every byte of the
/// haystack is walked backward at most once across the whole scan (the
/// module docs' cost argument).
pub fn find_unescaped(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    assert!(!needle.is_empty(), "empty needle");
    let finder = memchr::memmem::Finder::new(needle);
    // `from` is the next occurrence search's origin: 0, then one byte
    // past each rejected hit — the resume rule that keeps self-overlapping
    // needles correct.
    let mut from = 0;
    while let Some(rel) = finder.find(&haystack[from..]) {
        let hit = from + rel;
        // The backward run count: walk the maximal backslash run ending
        // at the hit down to its start (a non-backslash byte, or offset
        // 0). No state is carried between hits — the module docs' second
        // bullet is why that is safe: at most one hit per maximal run
        // ever walks, so no byte is walked twice.
        let mut run_back = hit;
        while run_back > 0 && haystack[run_back - 1] == b'\\' {
            run_back -= 1;
        }
        if (hit - run_back).is_multiple_of(2) {
            return Some(hit);
        }
        from = hit + 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const NUL_ESCAPE: &[u8] = b"\\u0000";

    /// The independent oracle (`reference_find_unescaped`, the Python
    /// suite's, spelled in Rust): every position in order, a full backward
    /// run walk per occurrence, no carried state, no engine — agreement
    /// between this and `find_unescaped` over the exhaustive sweep below is
    /// evidence about the contract, not a shared bug.
    fn naive_find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        let mut i = 0;
        while i + needle.len() <= haystack.len() {
            if &haystack[i..i + needle.len()] == needle {
                let mut run = 0;
                let mut j = i;
                while j > 0 && haystack[j - 1] == b'\\' {
                    run += 1;
                    j -= 1;
                }
                if run % 2 == 0 {
                    return Some(i);
                }
            }
            i += 1;
        }
        None
    }

    #[test]
    fn runs_zero_through_seven_decide_liveness() {
        // The parity ladder: a run of k backslashes before the occurrence,
        // live iff k is even. k=0 is the offset-0 case (nothing before the
        // occurrence, an even empty run).
        for k in 0..=7usize {
            let mut haystack = vec![b'\\'; k];
            haystack.extend_from_slice(NUL_ESCAPE);
            haystack.extend_from_slice(b" tail");
            let want = if k % 2 == 0 { Some(k) } else { None };
            assert_eq!(find_unescaped(&haystack, NUL_ESCAPE), want, "run {k}");
        }
    }

    #[test]
    fn json_real_and_literal_renderings_in_both_orders() {
        // The motivating ambiguity, as it sits on the wire inside a JSON
        // string value: the real NUL ("a\u0000b"), the literal text
        // ("a\\u0000b"), and both orders in one document.
        assert_eq!(find_unescaped(b"\"a\\u0000b\"", NUL_ESCAPE), Some(2));
        assert_eq!(find_unescaped(b"\"a\\\\u0000b\"", NUL_ESCAPE), None);
        assert_eq!(
            find_unescaped(b"\"a\\\\u0000b\"\"c\\u0000d\"", NUL_ESCAPE),
            Some(13)
        );
        assert_eq!(
            find_unescaped(b"\"c\\u0000d\"\"a\\\\u0000b\"", NUL_ESCAPE),
            Some(2)
        );
    }

    #[test]
    fn adjacent_literals_are_all_rejected_then_the_real_one_is_found() {
        // The miniature false-positive corpus (three literal renderings
        // back to back), then the same with a real escape appended: the
        // scan walks past every rejection to the live occurrence.
        let literals_only = [b"\\", NUL_ESCAPE].concat().repeat(3);
        assert_eq!(find_unescaped(&literals_only, NUL_ESCAPE), None);
        let mut with_real = literals_only.clone();
        with_real.extend_from_slice(NUL_ESCAPE);
        assert_eq!(find_unescaped(&with_real, NUL_ESCAPE), Some(21));
    }

    #[test]
    fn rejected_hits_advance_one_byte_not_the_whole_match() {
        // The self-overlapping battery: the resume rule's own rows. `00` in
        // `\000` is the canonical one — the live hit at 2 starts INSIDE the
        // rejected match at 1, so any resume-at-match-end scan (memmem's
        // find_iter) answers None here.
        assert_eq!(find_unescaped(b"\\000", b"00"), Some(2));
        assert_eq!(find_unescaped(b"\\0000", b"000"), Some(2));
        assert_eq!(find_unescaped(b"\\a\\aa", b"a"), Some(4));
    }

    #[test]
    fn all_backslash_needles_answer_at_the_run_start() {
        // An all-backslash needle's first occurrence always sits at a
        // maximal run's start (an even, empty run before it — the leftmost
        // window of k backslashes begins where the run does), so those
        // scans answer at the first hit and never walk.
        assert_eq!(find_unescaped(b"\\\\", b"\\"), Some(0));
        assert_eq!(find_unescaped(&[b'\\'; 8], b"\\\\"), Some(0));
        let haystack = [b"x", &[b'\\'; 3][..], b"y"].concat();
        assert_eq!(find_unescaped(&haystack, b"\\\\"), Some(1));
    }

    #[test]
    fn offsets_index_bytes_over_multibyte_content() {
        // bytes in, BYTE offsets out: the needle after a two-byte char
        // (byte 6, char 5), after a four-byte emoji (byte 4, char 1), and
        // the -1 shape with multibyte bytes still traversed past two
        // rejected hits.
        let acute = "caf\u{e9} ".as_bytes();
        let mut haystack = acute.to_vec();
        haystack.extend_from_slice(NUL_ESCAPE);
        assert_eq!(find_unescaped(&haystack, NUL_ESCAPE), Some(6));
        let emoji = "\u{1f600}".as_bytes();
        let mut haystack = emoji.to_vec();
        haystack.extend_from_slice(NUL_ESCAPE);
        assert_eq!(find_unescaped(&haystack, NUL_ESCAPE), Some(4));
        // CJK row, mirroring the Python byte-offset battery: two CJK chars
        // are six UTF-8 bytes, so the needle sits at byte 6 (char 2).
        let cjk = "\u{6771}\u{4eac}".as_bytes();
        let mut haystack = cjk.to_vec();
        haystack.extend_from_slice(NUL_ESCAPE);
        assert_eq!(find_unescaped(&haystack, NUL_ESCAPE), Some(6));
        let mut haystack = [b"\\", NUL_ESCAPE].concat().repeat(2);
        haystack.extend_from_slice(b"caf");
        haystack.extend_from_slice("\u{e9}".as_bytes());
        assert_eq!(find_unescaped(&haystack, NUL_ESCAPE), None);
    }

    #[test]
    fn degenerate_shapes_are_answerable_not_errors() {
        assert_eq!(find_unescaped(b"", NUL_ESCAPE), None);
        assert_eq!(find_unescaped(NUL_ESCAPE, NUL_ESCAPE), Some(0));
        assert_eq!(find_unescaped(b"\\u00", NUL_ESCAPE), None);
        assert_eq!(find_unescaped(b"plain text, no escapes", NUL_ESCAPE), None);
    }

    #[test]
    fn long_runs_and_many_hits_stay_correct() {
        // The long-run and many-hit shapes at scale. Careful with the run
        // arithmetic (the mistake this row originally pinned against
        // itself): for R backslashes then `u0000`, the needle's occurrence
        // starts at R-1 — its first byte IS the run's last backslash — so
        // the run BEFORE the hit is R-1, live iff R-1 is even.
        // R = 99_999: run 99_998, even, live at 99_998.
        let mut odd = vec![b'\\'; 99_999];
        odd.extend_from_slice(b"u0000");
        assert_eq!(find_unescaped(&odd, NUL_ESCAPE), Some(99_998));
        // R = 100_000: run 99_999, odd, rejected, and nothing else occurs.
        let mut even = vec![b'\\'; 100_000];
        even.extend_from_slice(b"u0000");
        assert_eq!(find_unescaped(&even, NUL_ESCAPE), None);
        // The needle's own leading backslash behind a whole even run: the
        // occurrence at the run's end is live.
        let mut run_then_needle = vec![b'\\'; 100_000];
        run_then_needle.extend_from_slice(NUL_ESCAPE);
        assert_eq!(find_unescaped(&run_then_needle, NUL_ESCAPE), Some(100_000));
        // Ten thousand all-rejected hits (the false-positive corpus's
        // shape, where a resume or walk bug would compound across hits).
        let corpus = [b"\\", NUL_ESCAPE].concat().repeat(10_000);
        assert_eq!(find_unescaped(&corpus, NUL_ESCAPE), None);
    }

    #[test]
    fn every_needle_over_a_tiny_alphabet_matches_the_naive_walk() {
        // The exhaustive sweep, the Python suite's tiny-alphabet idiom:
        // every 1-2 byte needle over {\\, u, 0} (self-overlapping ones
        // included) crossed with every haystack over the same alphabet up
        // to length 7 — 39,360 pairs (12 needles x 3,280 haystacks), the
        // complete small space of run/overlap/adjacency interactions at
        // that size, no sampling. Level-by-level: each length built once
        // from the previous length's snapshot, so every haystack occurs
        // exactly once (a cumulative extend-from-all would duplicate short
        // haystacks into 16,384 entries for the same 3,280 distinct).
        let alphabet = *b"\\u0";
        let mut needles: Vec<Vec<u8>> = Vec::new();
        for &a in &alphabet {
            needles.push(vec![a]);
            for &b in &alphabet {
                needles.push(vec![a, b]);
            }
        }
        let mut haystacks: Vec<Vec<u8>> = vec![Vec::new()];
        let mut level: Vec<Vec<u8>> = vec![Vec::new()];
        for _ in 0..7 {
            let mut next = Vec::new();
            for base in &level {
                for &c in &alphabet {
                    let mut extended = base.clone();
                    extended.push(c);
                    next.push(extended);
                }
            }
            haystacks.extend(next.iter().cloned());
            level = next;
        }
        assert_eq!(haystacks.len(), 3_280);
        for needle in &needles {
            for haystack in &haystacks {
                assert_eq!(
                    find_unescaped(haystack, needle),
                    naive_find(haystack, needle),
                    "needle {needle:?} haystack {haystack:?}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "empty needle")]
    fn empty_needle_is_a_core_precondition_violation() {
        // The pub core (`pub mod scan_impl` in lib.rs) enforces the
        // non-empty-needle precondition itself; the pyo3 wrapper's
        // ValueError is the Python spelling of the same refusal.
        let _ = find_unescaped(b"haystack", b"");
    }

    #[test]
    fn buffer_end_and_adjacent_live_shapes() {
        // Buffer-end goldens: the needle live at the very end (run 2,
        // even, hit at 2, no tail past the match) and the rejected-at-end
        // shape (run 1, odd, -1 with no tail to walk past).
        assert_eq!(
            find_unescaped(&[b"\\\\", NUL_ESCAPE].concat(), NUL_ESCAPE),
            Some(2)
        );
        assert_eq!(
            find_unescaped(&[b"\\", NUL_ESCAPE].concat(), NUL_ESCAPE),
            None
        );
        // Adjacent live-live: two live occurrences back to back answer the
        // FIRST, not the last.
        assert_eq!(
            find_unescaped(&[NUL_ESCAPE, NUL_ESCAPE].concat(), NUL_ESCAPE),
            Some(0)
        );
    }

    #[test]
    fn backward_walk_budget_is_linear() {
        // The runs-are-disjoint cost proof, reproducibly: instrument the
        // same walk the core runs and assert total backward steps <=
        // haystack length over the long-run shape and the 10k-rejected
        // shape. Every walk stays inside its own maximal backslash run
        // and at most one hit per run walks, so no byte is walked twice.
        fn counted(haystack: &[u8], needle: &[u8]) -> (Option<usize>, usize) {
            let finder = memchr::memmem::Finder::new(needle);
            let mut from = 0;
            let mut steps = 0usize;
            while let Some(rel) = finder.find(&haystack[from..]) {
                let hit = from + rel;
                let mut run_back = hit;
                while run_back > 0 && haystack[run_back - 1] == b'\\' {
                    run_back -= 1;
                    steps += 1;
                }
                if (hit - run_back).is_multiple_of(2) {
                    return (Some(hit), steps);
                }
                from = hit + 1;
            }
            (None, steps)
        }
        let mut run_then_needle = vec![b'\\'; 100_000];
        run_then_needle.extend_from_slice(NUL_ESCAPE);
        let (got, steps) = counted(&run_then_needle, NUL_ESCAPE);
        assert_eq!(got, Some(100_000));
        assert!(
            steps <= run_then_needle.len(),
            "walked {steps} steps over {} bytes",
            run_then_needle.len()
        );
        let corpus = [b"\\", NUL_ESCAPE].concat().repeat(10_000);
        let (got, steps) = counted(&corpus, NUL_ESCAPE);
        assert_eq!(got, None);
        assert!(
            steps <= corpus.len(),
            "walked {steps} steps over {} bytes",
            corpus.len()
        );
        // The inside-run long shape: R = 99_999 backslashes then `u0000`
        // spells the needle at 99_998 (run 99_998 before the hit, even).
        let mut odd = vec![b'\\'; 99_999];
        odd.extend_from_slice(b"u0000");
        let (got, steps) = counted(&odd, NUL_ESCAPE);
        assert_eq!(got, Some(99_998));
        assert!(
            steps <= odd.len(),
            "walked {steps} steps over {} bytes",
            odd.len()
        );
        // The backslash-u rejected chain: four all-rejected hits each
        // behind exactly one backslash (runs 3, 1, 1, 1), the
        // consecutive-rejected-hits boundary shape.
        let mut chain = vec![b'\\'; 4];
        for _ in 0..3 {
            chain.extend_from_slice(b"u\\\\");
        }
        chain.extend_from_slice(b"u");
        let (got, steps) = counted(&chain, b"\\u");
        assert_eq!(got, None);
        assert!(
            steps <= chain.len(),
            "walked {steps} steps over {} bytes",
            chain.len()
        );
    }
}
