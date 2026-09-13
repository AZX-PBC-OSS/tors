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
//! # The engine and the amortized run state
//!
//! The occurrence scan is `memchr::memmem::Finder` (memchr is already a
//! direct dependency; the Finder holds the needle's search strategy, so
//! the per-hit loop reuses one build), and the parity work is a backward
//! walk over the backslash run immediately before each hit, with the run
//! state carried forward across hits so no run is re-walked:
//!
//! * Invariant: `run_start` is the start of the maximal backslash run
//!   ending at `scanned` (empty at 0), maintained exactly by induction —
//!   the walk below either stops at a non-backslash (a fresh maximal run)
//!   or reaches `scanned`, in which case the run genuinely continues into
//!   the carried `[run_start, scanned)` and its start is already known.
//! * Cost: the walk for a hit covers only bytes of the gap since the
//!   previous hit, and gaps tile (each byte belongs to exactly one
//!   inter-hit gap), so every byte is walked backward at most once across
//!   the whole scan — the same bounding argument `skip_to_character`
//!   writes down for its bulk-skipped runs: the walk-back's total cost is
//!   bounded by the runs the forward scan skipped past. A naive per-hit
//!   walk, by contrast, re-walks a shared run's bytes for every hit at its
//!   tail (up to needle-backslash-prefix-length hits per run, each walking
//!   the full run: a long run followed by `u0000`-style tails pays the run
//!   once per hit).
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
/// past the whole match, so self-overlapping needles stay correct; the
/// run state carried between hits keeps every byte of the haystack walked
/// backward at most once across the whole scan (the module docs'
/// amortization argument).
pub fn find_unescaped(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let finder = memchr::memmem::Finder::new(needle);
    // The carried run state (the module docs' invariant): `run_start` is
    // the start of the maximal backslash run ending at `scanned`, and
    // `scanned` is the previous hit's offset (0 before the first). `from`
    // is the next occurrence search's origin: 0, then one byte past each
    // rejected hit — the resume rule that keeps self-overlapping needles
    // correct.
    let mut scanned = 0;
    let mut run_start = 0;
    let mut from = 0;
    while let Some(rel) = finder.find(&haystack[from..]) {
        let hit = from + rel;
        // The backward run count, bounded by `scanned`: walk the gap's
        // trailing backslash run. Stopping at a non-backslash settles a
        // fresh maximal run; reaching `scanned` means the run continues
        // into the carried `[run_start, scanned)`, whose start is already
        // known — the jump that keeps the walk regions inside disjoint
        // inter-hit gaps, so no run is ever re-walked.
        let mut run_back = hit;
        while run_back > scanned && haystack[run_back - 1] == b'\\' {
            run_back -= 1;
        }
        let hit_run_start = if run_back == scanned {
            run_start
        } else {
            run_back
        };
        // Carry the classified prefix forward: the maximal run ending at
        // `hit` is now the invariant's subject for the next hit.
        scanned = hit;
        run_start = hit_run_start;
        if (hit - hit_run_start).is_multiple_of(2) {
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

    /// The independent oracle (the Python suite's `reference_find_unescaped`
    ///, spelled in Rust): every position in order, a full backward run walk
    /// per occurrence, no carried state, no engine — agreement between this
    /// and `find_unescaped` over the exhaustive sweep below is evidence
    /// about the contract, not a shared bug.
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
        // The parity ladder: run k before the occurrence, live iff k is
        // even (0 included: the occurrence at offset k with nothing before
        // the run's backslashes... k=0 IS the offset-0 case).
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
        // The carried-run-state shapes at scale. Careful with the run
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
        // shape, where a carried-state bug would compound across hits).
        let corpus = [b"\\", NUL_ESCAPE].concat().repeat(10_000);
        assert_eq!(find_unescaped(&corpus, NUL_ESCAPE), None);
    }

    #[test]
    fn every_needle_over_a_tiny_alphabet_matches_the_naive_walk() {
        // The exhaustive sweep, the Python suite's tiny-alphabet idiom:
        // every 1-2 byte needle over {\\, u, 0} (self-overlapping ones
        // included) crossed with every haystack over the same alphabet up
        // to length 7 — 39,348 pairs, the complete small space of
        // run/overlap/adjacency interactions at that size, no sampling.
        let alphabet = *b"\\u0";
        let mut needles: Vec<Vec<u8>> = Vec::new();
        for &a in &alphabet {
            needles.push(vec![a]);
            for &b in &alphabet {
                needles.push(vec![a, b]);
            }
        }
        let mut haystacks: Vec<Vec<u8>> = vec![Vec::new()];
        for _ in 0..7 {
            let mut next = Vec::new();
            for base in &haystacks {
                for &c in &alphabet {
                    let mut extended = base.clone();
                    extended.push(c);
                    next.push(extended);
                }
            }
            haystacks.extend(next);
        }
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
}
