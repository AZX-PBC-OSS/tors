//! Unit-count chunking: [`chunk_by_words`], [`chunk_by_sentences`],
//! [`chunk_by_paragraphs`], and [`chunk_by_lines`], the
//! `tors.chunk_text`/`tors.chunk_cdc` family's third shape: instead of a
//! character budget ([`crate::chunk_impl::chunk_text`]) or byte-content
//! anchoring (`chunk_cdc`), each chunk spans a fixed COUNT of consecutive
//! segments from one of the crate's segmenters: real word tokens
//! ([`crate::segmentation_impl::word_bounds`], filtered), UAX #29
//! sentences ([`crate::segmentation_impl::sentence_bounds`]),
//! newline-run-delimited paragraphs ([`paragraph_bounds`], a heuristic: no
//! UAX exists for paragraphs), or newline-terminated lines ([`line_bounds`],
//! the same CR/CRLF-folding convention). Split from `chunk_impl.rs` per the
//! crate's own "one concern per file" rule: [`chunk_text`] and `chunk_cdc`
//! are CHARACTER/BYTE-budget chunkers, these four are UNIT-COUNT chunkers, a
//! different windowing shape sharing only the `Boundary`-safety discipline,
//! not the cut logic.
//!
//! [`chunk_by_segments`] is the one windowing walk behind all four unit
//! chunkers: "N segments per chunk, Y segments of overlap" over whatever
//! `(start, end)` segment list the caller already produced: the DRY point
//! this file exists to keep in one place rather than copied per unit.
//!
//! [`crate::chunk_impl`]

use crate::segmentation_impl;
use crate::truncate_impl::{GraphemeIndex, char_count};
use memchr::memchr2;

/// Merge adjacent CONTIGUOUS segments (`bounds[i].1 == bounds[i + 1].0`:
/// the `word_bounds`/`sentence_bounds` covering-partition contract) whose
/// shared boundary is NOT a grapheme-cluster boundary, the same
/// SARA-AM-shaped edge `crate::chunk_impl`'s hard-cut fallback guards
/// against: UAX #29 word boundaries occasionally score a combining
/// sequence (e.g. Thai SARA AM, U+0E33) as its own word-segment even
/// though `unicode-segmentation`'s grapheme rules join it to the
/// preceding base character into one cluster. [`chunk_by_segments`]
/// windows over segment EDGES directly, so a chunk boundary landing
/// exactly on such a split would silently divide the cluster between two
/// returned chunks; merging the two segments before windowing removes the
/// cut point rather than special-casing it per chunk. One forward pass,
/// O(n), the membership question answered O(1) by the shared
/// [`GraphemeIndex`] bitmap (the former `HashSet<usize>` built from the
/// whole boundary list cost ~12.6M hashed inserts — ~1.2 s — on a 12 MiB
/// document, the same #22 pathology `chunk_hierarchical` fixed).
fn merge_mid_cluster_boundaries(
    bounds: Vec<(usize, usize)>,
    graphemes: &GraphemeIndex,
) -> Vec<(usize, usize)> {
    // Fast path: when no adjacent pair shares a mid-cluster edge, the
    // merge would return the input unchanged — so it does, the input Vec
    // moving through with no copy. That is the common case (any text
    // without UAX-29/grapheme boundary divergence, e.g. pure-ASCII prose
    // with no CRLF pairs, where the merge pass would otherwise duplicate
    // a multi-MiB bounds list just to push every segment through). One
    // allocation-free scan decides; the differential tests pin both
    // paths to the same output.
    if !bounds
        .windows(2)
        .any(|w| w[0].1 == w[1].0 && !graphemes.is_boundary(w[1].0))
    {
        return bounds;
    }
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(bounds.len());
    for (start, end) in bounds {
        match merged.last_mut() {
            Some(last) if last.1 == start && !graphemes.is_boundary(start) => {
                last.1 = end;
            }
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// `chunk_by_words`'s real-token filter — keep exactly the segments
/// carrying at least one non-whitespace codepoint — as ONE streaming
/// decode pass over `text` instead of the former whole-text `Vec<char>`
/// collect (4 bytes per codepoint materialized just to random-access
/// slice each segment, the same #22 allocation class). `merged` is a
/// contiguous covering partition of `[0, total)` (word_bounds' own
/// contract, preserved by the merge), so a single forward decode crosses
/// every segment edge in sequence: each decoded codepoint folds into the
/// current segment's has-a-non-whitespace flag, and each completed
/// segment is kept or dropped on that flag alone. O(text) time, O(1)
/// memory beyond the output.
fn retain_non_whitespace_segments(text: &str, merged: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut bounds = Vec::with_capacity(merged.len());
    let mut seg = 0usize;
    let mut has_non_ws = false;
    let mut cp = 0usize;
    // The partition's contiguity, debug-pinned: segment `seg` must begin
    // exactly where the decode cursor is when it opens.
    let mut seg_start_cp = 0usize;
    for ch in text.chars() {
        if seg < merged.len() && cp == merged[seg].1 {
            debug_assert_eq!(merged[seg].0, seg_start_cp);
            if has_non_ws {
                bounds.push(merged[seg]);
            }
            seg += 1;
            seg_start_cp = cp;
            has_non_ws = false;
        }
        has_non_ws |= !ch.is_whitespace();
        cp += 1;
    }
    // The final segment completes at the decode's end, not before another
    // codepoint arrives. A partition that stopped short of the decode's
    // end (a contract violation, since word_bounds covers the whole text)
    // leaves a segment unconsumed: caught here in debug.
    if seg < merged.len() && cp == merged[seg].1 {
        debug_assert_eq!(merged[seg].0, seg_start_cp);
        if has_non_ws {
            bounds.push(merged[seg]);
        }
        seg += 1;
    }
    debug_assert_eq!(seg, merged.len());
    bounds
}

/// The shared "N segments per chunk, Y segments of overlap" walk behind
/// all four unit chunkers — [`chunk_by_words`], [`chunk_by_sentences`],
/// [`chunk_by_paragraphs`], and [`chunk_by_lines`]: the only difference
/// between them is which segmenter produced `bounds`, so the windowing
/// logic itself is factored here once rather than duplicated per unit
/// (the `elapsed_exceeds` precedent: factor a second consumer, don't copy
/// it). `bounds` is an ascending, non-overlapping segment list; the four
/// producers split on contiguity — the word/sentence producers
/// (`segmentation_impl::word_bounds`/`sentence_bounds`) are contiguous
/// coverings of the whole text, while the paragraph/line producers
/// ([`paragraph_bounds`]/[`line_bounds`]) are gapped (each excludes the
/// break runs it splits on, so the span between two consecutive segments
/// belongs to neither). The walk only ever reads `bounds[i].0` and
/// `bounds[j - 1].1`, so contiguity is genuinely not required — only the
/// ascending, non-overlapping part of the contract is. This function
/// trusts that contract and does not re-validate it.
///
/// Each chunk spans `per_chunk` consecutive segments, `[bounds[i].0,
/// bounds[i + per_chunk - 1].1)`, except possibly the LAST chunk, which
/// takes whatever remains when the segment count doesn't divide evenly.
/// Consecutive chunks advance by `stride = per_chunk - overlap` segments
/// (`overlap < per_chunk` is the caller's precondition, so
/// `stride >= 1` always: unconditional forward progress by construction,
/// no runtime check needed the way `chunk_text_overlapping`'s
/// character-granularity snapping needs one). Empty `bounds` (empty text)
/// yields `[]`.
fn chunk_by_segments(
    bounds: &[(usize, usize)],
    per_chunk: usize,
    overlap: usize,
) -> Vec<(usize, usize)> {
    if bounds.is_empty() {
        return Vec::new();
    }
    // `assert!`, not `debug_assert!`: this function is `pub` Rust API in its
    // own right (reachable without going through the pyo3 validation these
    // four callers' Python bindings apply), and both preconditions are
    // load-bearing for `stride`'s arithmetic below: with overflow checks
    // off in a release build (the crate's default profile), a violated
    // `overlap < per_chunk` would underflow `per_chunk - overlap` into a
    // huge `usize` silently rather than panic, corrupting the walk instead
    // of failing loudly. The same discipline `chunk_text`/
    // `chunk_text_overlapping`'s own `assert!`s already apply.
    assert!(per_chunk > 0, "per_chunk must be at least 1, got 0");
    assert!(
        overlap < per_chunk,
        "overlap must be less than per_chunk (no forward progress otherwise), \
         got overlap={overlap}, per_chunk={per_chunk}"
    );
    let stride = per_chunk - overlap;
    let n = bounds.len();
    let mut chunks = Vec::with_capacity(n.div_ceil(stride));
    let mut i = 0usize;
    loop {
        let j = (i + per_chunk).min(n);
        chunks.push((bounds[i].0, bounds[j - 1].1));
        if j >= n {
            break;
        }
        i += stride;
    }
    chunks
}

/// Word-count-windowed chunking: each chunk spans `words_per_chunk`
/// consecutive WORD TOKENS, not `word_bounds`' raw segment
/// count. `word_bounds` itself follows UAX #29 exactly, which gives an
/// inter-word space run its OWN segment (`"one two"` is three segments:
/// `"one"`, `" "`, `"two"`), the established convention `word_count`
/// already carries. Grouping RAW segments here would silently mean
/// "`words_per_chunk` roughly halved" for ordinary space-separated
/// prose, the opposite of what a caller reaching for
/// `words_per_chunk=100` (a "~100 word chunk" for an embedding budget)
/// actually wants. So this filters `word_bounds`' output to segments
/// that carry at least one non-whitespace codepoint FIRST, and only
/// then windows over what remains: a "word" here is a real token, and
/// the whitespace between two tokens in one chunk still rides along
/// naturally (the span is a contiguous slice of the ORIGINAL text
/// between two real absolute offsets, not a re-assembly of kept
/// segments), exactly as it would if nothing had been filtered.
///
/// `(start, end)` are codepoint offsets spanning the first included
/// word token's start through the last included token's end (NOT
/// through any trailing whitespace after it: that whitespace belongs
/// to neither this chunk nor the next one's word tokens, so
/// non-overlapping chunks are no longer necessarily contiguous, unlike
/// `chunk_text`'s covering-partition contract; this function makes no
/// such claim). The final chunk may hold fewer than `words_per_chunk`
/// tokens when the total doesn't divide evenly. Empty text, or text
/// with no word tokens at all (pure whitespace), yields `[]`.
/// `words_per_chunk == 0` or `overlap >= words_per_chunk` are the pyo3
/// layer's `ValueError`s (this core trusts its precondition, matching
/// [`chunk_by_segments`]'s own contract).
///
/// Grapheme-cluster-safe at every window edge, the same fix
/// `crate::chunk_impl::chunk_text` applies: `word_bounds` occasionally
/// scores a combining sequence (e.g. Thai SARA AM) as its own
/// word-segment even though it's one grapheme cluster, and a chunk
/// boundary landing there would silently split it:
/// [`merge_mid_cluster_boundaries`] closes this before windowing starts.
pub fn chunk_by_words(text: &str, words_per_chunk: usize, overlap: usize) -> Vec<(usize, usize)> {
    // Merge any word_bounds segment edge that would split a grapheme
    // cluster (the SARA AM edge, see `merge_mid_cluster_boundaries`)
    // BEFORE filtering out whitespace-only segments: the merge relies on
    // `word_bounds`' raw covering-partition contiguity, which the
    // whitespace filter below would otherwise break (it opens gaps).
    let graphemes = GraphemeIndex::build(text, char_count(text));
    let merged = merge_mid_cluster_boundaries(segmentation_impl::word_bounds(text), &graphemes);
    let bounds = retain_non_whitespace_segments(text, merged);
    chunk_by_segments(&bounds, words_per_chunk, overlap)
}

/// [`chunk_by_words`]'s sentence-count twin: each chunk spans
/// `sentences_per_chunk` consecutive UAX #29 sentence segments
/// (`sentence_bounds`), `overlap` sentences repeated. Same contract,
/// same preconditions, same empty-input answer, same
/// grapheme-cluster-safe window edges (see [`chunk_by_words`]'s docs).
pub fn chunk_by_sentences(
    text: &str,
    sentences_per_chunk: usize,
    overlap: usize,
) -> Vec<(usize, usize)> {
    // Same grapheme-cluster merge as chunk_by_words, applied to
    // sentence_bounds' segments (also a covering, contiguous partition,
    // so the merge's contiguity assumption holds directly: no
    // whitespace-filter step exists here to reorder around).
    let graphemes = GraphemeIndex::build(text, char_count(text));
    let bounds = merge_mid_cluster_boundaries(segmentation_impl::sentence_bounds(text), &graphemes);
    chunk_by_segments(&bounds, sentences_per_chunk, overlap)
}

/// Paragraph boundaries: `text` split on maximal runs of 2+ NEWLINE
/// UNITS: `\r\n` counts as ONE unit (matching `normalize`'s own
/// CR/CRLF folding), a lone `\r` or `\n` also one unit each. This is
/// the same "2+ newlines is the surviving paragraph gap" convention
/// `normalize`'s own pipeline already establishes (it collapses 3+
/// consecutive newlines down to exactly 2, never below: see
/// `normalize_impl::flush`). There is NO Unicode Standard segmentation
/// for paragraphs (unlike UAX #29 for words/sentences), so this is a
/// heuristic, stated plainly, not a spec-backed segmenter: a single `\n`
/// is ordinary content here, not a break (`"A\nB"` is one paragraph),
/// and a "blank-looking" line that holds only spaces/tabs between two
/// LONE newlines does NOT qualify: only an actual run of 2+ newline
/// characters does. This operates on `text` as given, not on any prior
/// `normalize` pass.
///
/// Each returned span is one paragraph's content, `(start, end)`
/// codepoint offsets, EXCLUDING the separating run itself (a paragraph's
/// span shouldn't include the gap that separates it from the next one).
/// A leading or trailing qualifying run produces an empty span at that
/// edge, which is DISCARDED rather than emitted: an empty "paragraph"
/// is not a useful chunk. Text with no qualifying run at all yields
/// exactly one paragraph: the whole text. Empty input yields `[]`.
///
/// UNLIKE `word_bounds`/`sentence_bounds`, this split point is
/// structurally grapheme-safe with no merge step needed: every split
/// happens strictly INSIDE a run of `\n`/`\r` characters (`\r\n` is
/// consumed as one unit, matching `normalize`'s own CRLF folding, so a
/// CRLF pair is never itself torn in two), and neither character is a
/// combining mark: a grapheme cluster spanning a newline would require a
/// combining mark to immediately follow it, at which point the newline
/// (never emitted in any paragraph's span: it's discarded as separator
/// content) is a non-printing control character, not text either
/// paragraph's caller would consider "split". No visible content
/// character is ever cut mid-cluster by this function.
/// The density-guard window [`next_break`] scans inline before it will
/// pay a `memchr2` call. 8 bytes: see [`next_break`]'s docs for the
/// measurement-backed reasoning.
const BREAK_WINDOW: usize = 8;

/// The byte-level break search [`paragraph_bounds`] uses: the offset of
/// the next `\r` or `\n` at or after `from`, or `None` when none
/// remains. Every other scanner in this crate already finds its anchor
/// bytes through memchr/memmem over the raw bytes (`fence_impl`'s
/// fence-line scan, `diff_impl`'s terminator scan, `normalize_impl`'s
/// stage sentinels, `truncate_impl`'s CRLF bit-clearing pass,
/// `chunk_hierarchical_impl::level_from_literal`'s separator walk);
/// `line_bounds` and `paragraph_bounds` were the last two per-`char`
/// `match` scanners, and this hop retires that spelling — but NOT as a
/// naive `memchr2` call per segment: on break soup (a break unit every
/// ~2 bytes) one memchr2 call and its per-call setup per 1–2 scanned
/// bytes loses to a plain byte loop, while the same per-segment
/// formulation is a large win at realistic densities (a break every
/// ~80 bytes). The density guard makes both regimes the cheap one: a
/// bounded inline byte scan of [`BREAK_WINDOW`] first, `memchr2` (SIMD)
/// from the window's end only when that comes up empty. A next break
/// found inside the window costs zero memchr overhead — on soup the
/// window finds every break within the first byte or two, the search
/// degenerates to roughly the old per-byte cost, and no per-segment
/// memchr2 call happens at all; a window that comes up empty pays the
/// inline bytes as pure overhead on sparse text, which is where the
/// constant's size is decided: 8 because the window's scalar bytes are
/// the dominant per-segment cost on realistic densities once the SIMD
/// fold takes the rest, while soup's breaks arrive within a byte or
/// two, far inside any window in the 8–32 range. The trade 8 buys:
/// densities between 8 and 16 bytes per break — very short lines —
/// pay one hop per segment, at roughly the OLD per-`char` decoder's
/// cost (never worse, just not won); everything denser stays inline,
/// everything sparser was paying the hop anyway. [`line_bounds`] fuses
/// this same window into its
/// per-segment fold (same constant, same reasoning — a separate search
/// would re-touch the window bytes the fold is already touching); this
/// bare window-then-hop spelling is paragraph_bounds', whose
/// per-segment work is only a codepoint count and needs no fold.
fn next_break(bytes: &[u8], from: usize) -> Option<usize> {
    let limit = (from + BREAK_WINDOW).min(bytes.len());
    if let Some(rel) = bytes[from..limit]
        .iter()
        .position(|&b| matches!(b, b'\r' | b'\n'))
    {
        return Some(from + rel);
    }
    memchr2(b'\r', b'\n', &bytes[limit..]).map(|rel| limit + rel)
}

/// How far one batch [`AsciiCert::certify`] attempt scans before giving
/// the next attempt a fresh start: 4 KiB. The batch exists to amortize
/// the per-CALL cost of `is_ascii` — the certification itself is
/// memory-bound and vectorized, but a slice call pays setup, bounds,
/// and SIMD prologue/epilogue per invocation, and at realistic
/// one-break-per-~80-bytes densities that per-call overhead dominates
/// the per-segment cost of the byte scans; one batch per ~50 segments
/// removes it. (A whole-text gate would amortize further but forfeits
/// the byte paths entirely on the first non-ASCII byte — the trade
/// this certificate exists to avoid.) 4 KiB keeps every batch inside
/// L1 and the failure cost (a batch that stops at a non-ASCII byte
/// plus the locating scan) bounded to a couple of cache lines past the
/// offending byte.
const CERT_CHUNK: usize = 4096;

/// A SLIDING ASCII certificate for the two byte-level scanners: the
/// range `lo..hi` is proven all-ASCII, advanced by ONE kind of evidence
/// — explicit batched `is_ascii` calls over [`CERT_CHUNK`]-sized
/// strides (`certify`). The query "is `a..b` pure ASCII" is then a
/// two-compare range check (`covers`) instead of a slice call, which is
/// what retires the per-segment `is_ascii`/`char_count` call the
/// pre-certificate spelling paid per segment — at realistic densities
/// that per-call overhead dominated the per-segment cost, and batching
/// removes it while buying the same amortization a whole-text
/// `is_ascii()` gate would, without the gate's forfeiture: a gate
/// collapses the WHOLE text to the per-`char` machine on the first
/// non-ASCII byte, this certificate degrades one 4 KiB stride and
/// keeps every ASCII segment on the byte path.
///
/// (Advancing by examination — the run walks vouching for the break
/// bytes they `match` — is deliberately NOT done: on break soup the
/// run walk fires per ~2-byte segment, so per-run bookkeeping costs
/// more than the batches it would save, which already amortize 4096×
/// without help; and every extra live value in a hot loop is a
/// register the loop state can lose. Batch-only: one mechanism, fewer
/// live values, and `certify`'s miss path is the only path that
/// touches bytes.)
///
/// The range is deliberately NOT a prefix-invariant ("everything below
/// `hi` is ASCII"): mixed text makes that unsound, because a known
/// non-ASCII byte at `p` would block every later batch from starting at
/// a frozen prefix forever. Instead the range slides: when a batch fails
/// inside a queried range, the first non-ASCII byte `p` is located (one
/// early-exit scan), the range is reset to the EMPTY range at `p + 1`
/// (`bytes[p]` is known-bad; no future batch may start on it), and the
/// next query — whose cursor has by then moved past `p` — restarts
/// cleanly from its own segment. Queries are monotone (the scanners'
/// byte cursor only moves forward) and every query start is inside or
/// ahead of the previous range's end, so a single sliding range is
/// always sufficient and stale ranges behind the cursor are simply
/// discarded on restart.
struct AsciiCert {
    /// Invariant: every byte in `bytes[lo..hi]` is ASCII (`lo <= hi`).
    lo: usize,
    hi: usize,
}

impl AsciiCert {
    /// The empty range at 0: nothing certified, every query misses and
    /// pays one batch.
    #[inline]
    fn new() -> Self {
        AsciiCert { lo: 0, hi: 0 }
    }

    /// Is `a..b` proven ASCII? A two-compare range check, no bytes
    /// touched. The empty range (`a == b`) is trivially covered whenever
    /// `a` itself is inside the certificate.
    #[inline]
    fn covers(&self, a: usize, b: usize) -> bool {
        self.lo <= a && b <= self.hi
    }

    /// Certify `bytes[a..b]` (`a <= b <= bytes.len()`): `true` iff the
    /// range is pure ASCII, answering from `covers` when it can and
    /// otherwise extending the sliding range with batched `is_ascii`
    /// strides until `b` is covered or a non-ASCII byte is proven inside
    /// `a..b` itself (a non-ASCII byte BEYOND `b` does not fail the
    /// query: the locating scan certifies `a..p` on the way, and `p >=
    /// b` answers the query affirmatively). On a genuine failure the
    /// range resets to the empty range past the offending byte so no
    /// later batch can start on it — the one known-bad byte is skipped,
    /// not forever-blocked (see the type docs for why that matters on
    /// mixed text).
    fn certify(&mut self, bytes: &[u8], a: usize, b: usize) -> bool {
        debug_assert!(a <= b && b <= bytes.len());
        if self.covers(a, b) {
            return true;
        }
        // Extend from `hi` when the range already covers `a` (contiguous
        // extension, `a..hi` already proven); otherwise start a fresh
        // range at `a` and let the loop below set `lo` on success.
        let extends = self.lo <= a && a <= self.hi;
        let mut start = if extends { self.hi } else { a };
        while start < b {
            let chunk_end = (start + CERT_CHUNK).min(bytes.len());
            debug_assert!(chunk_end > start);
            if bytes[start..chunk_end].is_ascii() {
                start = chunk_end;
                continue;
            }
            // The batch stopped: locate the first non-ASCII byte. The
            // scan itself certifies everything it walks past, so `p >=
            // b` still answers this query — only a bad byte INSIDE
            // `a..b` is a failure.
            let p = start
                + bytes[start..chunk_end]
                    .iter()
                    .position(|&x| x >= 0x80)
                    .expect("is_ascii failed, so a non-ASCII byte exists");
            if p >= b {
                if !extends {
                    self.lo = a;
                }
                self.hi = p;
                return true;
            }
            // A genuine failure: `a..b` holds a non-ASCII byte. Reset to
            // the empty range past it — everything the failed batch and
            // the locating scan walked is behind the cursor's future and
            // will be re-derived by the fallback path that answers THIS
            // segment exactly.
            self.lo = p + 1;
            self.hi = p + 1;
            return false;
        }
        // `start` reached `b`: the whole range is certified.
        if !extends {
            self.lo = a;
        }
        self.hi = start;
        true
    }
}

pub(crate) fn paragraph_bounds(text: &str) -> Vec<(usize, usize)> {
    // The scan as ONE byte-level walk over `text.as_bytes()` — the
    // memchr-family discipline every other scanner in the crate already
    // follows (see `next_break`'s docs for why this is NOT the naive
    // memchr-per-segment loop): [`next_break`]'s density-guarded hop
    // between break bytes, each inter-break segment's codepoint count
    // through the sliding ASCII certificate ([`AsciiCert`]: byte length
    // on the certified path, the continuation-byte predicate otherwise,
    // no decode, no whole-text `Vec<char>` collect — the #22 allocation
    // class stays retired — and certification batched one 4 KiB
    // `is_ascii` stride per ~50 segments, never the per-segment
    // `char_count` slice call), and each break RUN
    // walked unit-by-unit at byte level with the CRLF pairing done by
    // direct lookahead (`bytes[i] == b'\r' && bytes[i + 1] == b'\n'`)
    // instead of the former `pending_cr` flag — a flag was only ever
    // the streaming decode's substitute for the one-codepoint lookahead
    // the random-access oracle always had, and at byte level the
    // lookahead is free. O(text) time, O(1) memory beyond the output.
    //
    // The output is bit-identical to the former per-`char` state machine
    // by construction: the unit counting (CRLF pair = one, everything
    // else = one each), the 2+-unit run qualification, the
    // leading/trailing empty-span discards (`seg_start < run_start` and
    // the end-of-text close) are the same statements over the same
    // units — only the spelling of "find the next unit" changed. The
    // differential oracles in this module's tests (the corpus sweep and
    // both soup generators, ASCII and non-ASCII — the non-ASCII sweeps
    // exist to pin the codepoint-offset bookkeeping this walk now does
    // in bytes) hold that equivalence.
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut bounds = Vec::new();
    // The sliding ASCII certificate (see `AsciiCert`): certified
    // segments answer their codepoint count as the byte length with no
    // call at all, and certification itself is one batched `is_ascii`
    // stride per ~CERT_CHUNK bytes instead of the per-segment
    // `char_count` slice call the pre-certificate spelling paid — at
    // realistic densities that per-call overhead was the scan's dominant
    // per-segment cost (the same amortization on soup: the inlined
    // `covers` check answers every ~2-byte segment with no call at all,
    // and a batch fires only every 4 KiB).
    let mut cert = AsciiCert::new();
    // The byte cursor and the codepoint cursor advance together: every
    // segment pays its count, every break unit advances cp by its
    // byte count (1, or 2 for a CRLF pair — `\r` and `\n` are ASCII,
    // one codepoint each), so at loop exit cp IS the whole text's
    // codepoint total, derived inside the walk rather than paid as a
    // separate whole-text count pass — the same one-pass property the
    // decode spelling had (`total = cp + 1` per codepoint).
    let mut byte = 0usize;
    let mut cp = 0usize;
    let mut seg_start = 0usize;
    while byte < n {
        let brk = match next_break(bytes, byte) {
            Some(brk) => brk,
            None => {
                // No further break: the final segment runs to end of
                // text. Count it and let the close below emit the final
                // paragraph — a trailing qualifying run already moved
                // `seg_start` to cp (= total), which the close's
                // `seg_start < cp` guard discards. Same covers-fast-path
                // and same first-byte pre-probe as the per-segment
                // count below: a trailing segment is usually inside the
                // certificate already; the miss pays one final batch,
                // and a non-ASCII first byte skips straight to the
                // predicate count.
                cp += if bytes[byte] < 0x80
                    && (cert.covers(byte, n) || cert.certify(bytes, byte, n))
                {
                    n - byte
                } else {
                    bytes[byte..].iter().filter(|&x| (x & 0xC0) != 0x80).count()
                };
                break;
            }
        };
        // The segment's codepoint count: byte length on the certified
        // path (one ASCII byte IS one codepoint), the continuation-byte
        // predicate otherwise — inlined rather than delegated to
        // `char_count`, which would re-attempt the very certification
        // that just failed before reaching its own predicate branch.
        // The `covers` pre-check is the soup path's cheapness: this
        // loop runs once per ~2-byte segment on break soup (2.5M calls
        // at 12 MiB), and an `certify` CALL per segment — even one that
        // answers from its own internal covers check — costs more than
        // the `char_count` call it replaced; the inlined two-compare
        // makes the hit path call-free and leaves `certify`'s batch
        // machinery for the once-per-4 KiB miss.
        // The `bytes[byte] < 0x80` pre-probe: a non-ASCII byte at the
        // segment's FIRST byte makes both arms behind it guaranteed
        // failures — `covers` can only answer for a start already
        // inside the proven range (an in-range byte IS ASCII), and
        // `certify`'s batch would stop at byte 0, its locating scan
        // find byte 0, and the query fail and reset — so on
        // non-ASCII-starting segments (a CJK log's every segment) one
        // compare skips the whole attempt and lands directly in the
        // continuation-byte count the failure would have chosen
        // anyway. Skipping the failed call is unobservable in the
        // certificate's state: the reset it would perform moves the
        // range to the empty range at `byte + 1`, and the next query
        // starts past this whole segment — beyond both the stale and
        // the reset range ends — so `covers`/`extends` answer
        // identically either way (a reset only matters to a query at
        // or before the skipped byte, and the cursor never returns
        // there). Both queries here are non-empty (`brk > byte`
        // always: the cursor sits on a non-break byte), so the empty-
        // range `covers` corner (a == b == hi, where `hi` itself is
        // NOT proven) cannot be reached behind the probe.
        cp += if bytes[byte] < 0x80 && (cert.covers(byte, brk) || cert.certify(bytes, byte, brk)) {
            brk - byte
        } else {
            bytes[byte..brk]
                .iter()
                .filter(|&x| (x & 0xC0) != 0x80)
                .count()
        };
        // The run of break units from `brk`: a `\r` immediately followed
        // by `\n` is ONE unit (the CRLF folding, matching `normalize`'s
        // own), every other break byte one unit each. `i` lands on the
        // first non-break byte or end of text — always a codepoint
        // start, since break bytes are ASCII and a multi-byte sequence
        // never begins with a continuation byte after one — and cp
        // keeps pace, so cp at the run's end is the first post-run
        // codepoint's index, the same index the per-`char` machine's
        // `seg_start = cp` close used.
        let run_start = cp;
        let mut i = brk;
        let mut units = 0usize;
        while i < n && matches!(bytes[i], b'\r' | b'\n') {
            if bytes[i] == b'\r' && i + 1 < n && bytes[i + 1] == b'\n' {
                cp += 2;
                i += 2;
            } else {
                cp += 1;
                i += 1;
            }
            units += 1;
        }
        // Run qualification, the same statements as before: 2+ units
        // split — the span before the run is emitted unless it is empty
        // (the leading-run discard, `seg_start < run_start`) — and a
        // single-unit run is ordinary content, no split. A run that ran
        // to end of text is qualified HERE, inside the loop, which is
        // the former spelling's end-of-text close: nothing between the
        // run's last byte and the loop's exit reads this state.
        if units >= 2 {
            if seg_start < run_start {
                bounds.push((seg_start, run_start));
            }
            seg_start = cp;
        }
        byte = i;
    }
    // The final paragraph, guarded by `seg_start < cp` — KEPT, unlike
    // `line_bounds`' content-filter guard: paragraph_bounds has no
    // has-content flag, so this guard is the only thing standing between
    // a trailing qualifying run's `seg_start = cp` and a phantom empty
    // paragraph. Empty text falls out with no special case: the loop
    // never runs, cp stays 0, `0 < 0` is false.
    if seg_start < cp {
        bounds.push((seg_start, cp));
    }
    bounds
}

/// [`chunk_by_words`]'s paragraph-count twin: each chunk spans
/// `paragraphs_per_chunk` consecutive [`paragraph_bounds`] segments,
/// `overlap` PARAGRAPHS repeated. Same contract, same preconditions,
/// same empty-input answer: see [`paragraph_bounds`] for exactly what
/// counts as a paragraph boundary here (a heuristic, not a Unicode
/// Standard segmentation). UNLIKE the word/line twins, paragraphs have
/// NO content filter: a whitespace-only paragraph IS emitted as a chunk
/// (only fully-empty spans are dropped), so an overlapping pair of
/// chunks can share blank content.
pub fn chunk_by_paragraphs(
    text: &str,
    paragraphs_per_chunk: usize,
    overlap: usize,
) -> Vec<(usize, usize)> {
    let bounds = paragraph_bounds(text);
    chunk_by_segments(&bounds, paragraphs_per_chunk, overlap)
}

/// Line boundaries: `text` split on LINE-BREAK UNITS, where a unit is a
/// `\n`, a lone `\r`, or a `\r\n` pair counted as ONE (the same
/// CR/CRLF-folding convention [`paragraph_bounds`] and `normalize`'s own
/// pipeline already use; the exotic Unicode line separators
/// `str.splitlines` also honors — `\v`, `\f`, NEL, LS, PS — are NOT line
/// breaks here, keeping this family's "what `normalize` folds is what
/// splits" convention). Every break unit terminates exactly one line;
/// each returned span is one line's content, `(start, end)` codepoint
/// offsets EXCLUDING the break unit itself, and a trailing break at end
/// of text yields no trailing empty line (there is no content after it).
///
/// A line counts as a line only when it carries at least one
/// non-whitespace codepoint — the same real-token discipline
/// [`chunk_by_words`] applies to `word_bounds` segments (an inter-word
/// space run is not a word, a blank line is not a line): one message per
/// line (a chat thread), one record per line (a log), one cue per block
/// are the shapes this exists for, and a caller reaching for
/// `lines_per_chunk=200` wants 200 content lines, not "200 lines, of
/// which 40 are blank separators". "Non-whitespace" is definitional
/// here: the Unicode `White_Space` property (`char::is_whitespace`),
/// under which U+001C–U+001F (FS/GS/RS/US) count as CONTENT (Python's
/// `str.isspace()` treats them as whitespace, and `str.splitlines`
/// even breaks on them, so a ported expectation may differ) and NBSP
/// counts as blank. The blank lines between two counted
/// lines of the SAME chunk still ride along inside its span (the span is
/// a contiguous slice of the ORIGINAL text between two absolute offsets,
/// exactly as inter-word whitespace rides along in `chunk_by_words`);
/// they belong to neither chunk when the counted lines land in different
/// chunks. Whitespace-only text, or empty input, yields `[]`.
///
/// UNLIKE the `word_bounds`/`sentence_bounds` spellings (but exactly like
/// [`paragraph_bounds`]), this split point is structurally grapheme-safe
/// with no merge step: every split lands strictly between a break
/// character and adjacent content, and the break characters are never
/// combining marks. The one theoretical divergence — a combining mark
/// immediately after a newline joins the NEWLINE's cluster, so the next
/// line's span would start mid-cluster — is the same documented
/// non-issue [`paragraph_bounds`] carries: the "cluster" is a newline
/// plus an orphan combining mark, not visible content any line's caller
/// would call "split".
/// The ASCII half of `char::is_whitespace` as one byte-membership
/// table: `ASCII_WS[b]` is whether the codepoint `b` encodes is
/// White_Space, true exactly for 0x09..=0x0D and 0x20 — the set
/// identity [`line_bounds`]'s docs state, built by the const block in
/// exactly that spelling so the table cannot drift from the predicate
/// it replaces (the break bytes 0x0A/0x0D are members but never reach
/// the fold — the window stops at them first; the non-ASCII fallback
/// below is what keeps the answer exact above 0x7F). A table, not the
/// `matches!` range check, because that check compiles to a
/// five-instruction compare/setcc chain per byte and in the hot window
/// that chain is the largest per-content-byte cost on break soup; the
/// one-load lookup hangs off the side of the compare-based break
/// check, never on the loop's carried chain — routing the break check
/// itself through a table load would put a load→load latency exactly
/// on that chain.
const ASCII_WS: [bool; 256] = {
    let mut table = [false; 256];
    let mut b = b'\t';
    while b <= b'\r' {
        table[b as usize] = true;
        b += 1;
    }
    table[b' ' as usize] = true;
    table
};

pub(crate) fn line_bounds(text: &str) -> Vec<(usize, usize)> {
    // The scan as ONE byte-level walk over `text.as_bytes()`, the same
    // restructure [`paragraph_bounds`] above got, split into two regimes
    // around the density guard (see `next_break`'s docs for the guard's
    // measurement-backed reasoning):
    //
    // HOT — the no-call loop every segment starts in: the fold and the
    // break search are the SAME inline pass over the first
    // [`BREAK_WINDOW`] segment bytes, so a dense or short segment (a
    // break within the window — soup's every segment, a short line's)
    // is answered entirely by that one pass: no second scan, no slice,
    // no call overhead, and back-to-back break units (a line's break
    // run, soup's whole shape) are consumed by one inner walk — the
    // first unit ends the line just folded, every further unit ends an
    // empty line the real-line filter drops by construction, so the
    // walk pushes nothing. Keeping every call out of this loop is a
    // necessity, not style: a `memchr2` call site anywhere in it makes
    // the compiler spill the loop state, and soup pays that spill per
    // push — the hot/cold split exists for exactly that reason. (A
    // flat per-byte loop — the old machine's rhythm with the escape
    // check as its only per-segment logic — is no better: the per-
    // segment state it must carry across the back edge (the segment's
    // byte start, the non-ASCII flag) pushes the register allocation
    // over its edge and every unit begins reloading `bytes`, `n` and
    // `seg_start` from the stack; the window spelling keeps them in
    // registers by scoping the per-segment locals to one iteration.)
    //
    // COLD — where a window comes up empty (a long segment, the
    // realistic-density shape): one `memchr2` hop finds the break, and
    // the fold restarts over the WHOLE segment with the SIMD tools —
    // the sliding certificate (`AsciiCert`) answers the purity question
    // from batched 4 KiB `is_ascii` strides (one call per ~50 segments,
    // extending from wherever the sliding range ended — the
    // per-segment `is_ascii()` slice call this replaces paid its
    // per-call setup on every segment, which dominated the lane), an
    // early-exit `any()` finds the real-line answer at the first
    // content byte (a content line answers within a byte or two; the
    // discarded ≤BREAK_WINDOW window bytes are bounded waste, re-scanned
    // L1-hot), and the codepoint count on that path is just the byte
    // length. Cold then does its own line-close bookkeeping and hands
    // the break byte back to hot, which begins ON a break byte, folds
    // an empty segment for free, and walks the unit run — so the walk
    // exists in exactly one place.
    //
    // The byte-level whitespace check rests on a set identity:
    // `char::is_whitespace` (the Unicode `White_Space` property the
    // contract above pins) holds for an ASCII codepoint exactly when
    // its byte is `\t`..=`\r` (0x09–0x0D) or a space — note 0x0B (VT)
    // IS `White_Space`, so std's `u8::is_ascii_whitespace` (which omits
    // it) would silently mis-classify vertical-tab-only lines as
    // content, and U+001C–U+001F are NOT (they stay content, the
    // documented divergence from Python's `str.isspace()`). A segment
    // that is not pure ASCII takes the per-codepoint fallback at its
    // break (or at the end-of-text close): `char::is_whitespace` is
    // asked directly over the whole segment, exactness winning over
    // skipping a rare second pass; non-ASCII is detected per segment,
    // so mixed text keeps the byte paths on its ASCII segments (no
    // whole-text `is_ascii()` gate to forfeit them on any mixed
    // document). The codepoint count is byte arithmetic on the ASCII
    // paths (a pure-ASCII segment's byte length IS its codepoint
    // length) and a decode recount in the fallbacks — every
    // non-continuation byte starts exactly one UTF-8 codepoint, so no
    // path ever needs a separate counting pass.
    //
    // CRLF pairing is direct byte lookahead, not the former `pending_cr`
    // flag (the flag was the streaming decode's substitute for the
    // lookahead the random-access oracle always had): a `\r` immediately
    // followed by `\n` is ONE unit, the line ending at the `\r` and the
    // next beginning after the `\n`; a trailing `\r` with no following
    // byte, or followed by anything but `\n`, stands alone.
    //
    // Bit-identical to the former per-`char` state machine by
    // construction: the same break units (a `\r` ALWAYS opens one,
    // whether it stands alone or begins a CRLF pair; a `\n` opens one
    // only when it did not ride in as a pair's second half — at byte
    // level that is simply "the pair is consumed together"), the same
    // `has_non_ws` resets at the same unit boundaries, the same
    // end-of-text close. The differential oracles (corpus sweep plus
    // the ASCII and non-ASCII soup generators, the latter pinning the
    // fallback path and the codepoint bookkeeping) hold the equivalence.
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut bounds = Vec::new();
    // The byte cursor and the codepoint cursor advance together; cp at
    // each break is that break byte's own codepoint index, exactly
    // where the per-`char` machine pushed the line's end. The table is
    // held by reference so its address is one register held across the
    // loops instead of a `lea` re-materialized per byte.
    let ascii_ws = &ASCII_WS;
    // The sliding ASCII certificate (see `AsciiCert`): cold segments
    // answer `is this segment pure ASCII` as a two-compare range check
    // served by one batched 4 KiB `is_ascii` stride per ~50 segments —
    // the pre-certificate spelling paid the slice call per segment,
    // whose per-call overhead dominated the realistic-density lanes.
    // ONLY the cold path touches the certificate: the hot loop never
    // does (its per-segment register budget is the tighter resource,
    // see the window's comment), so dense text pays no certificate
    // overhead at all.
    let mut cert = AsciiCert::new();
    let mut byte = 0usize;
    let mut cp = 0usize;
    let mut seg_start = 0usize;
    let mut has_non_ws = false;
    'outer: loop {
        // ---- hot: segments whose break falls inside the window ----
        // The live set here is deliberately minimal (fold straight into
        // `has_non_ws`, no separate segment flag, no continuation
        // counter — the per-segment locals are scoped to one iteration
        // so they never cross the back edge): this loop runs once per
        // soup segment, and every extra live value is a register the
        // whole loop state can no longer stay in.
        while byte < n {
            // The fused window (the density guard): fold each byte and
            // break-check it in one pass, up to BREAK_WINDOW bytes. The
            // break check is COMPARE-based (two immediates against the
            // loaded byte) and the fold's table load hangs off the side
            // of it — routing more of the loop through the table loses
            // either way: a class-table `match` compiles to an indirect
            // jump per byte, and a table-first spelling that skipped
            // the compares for content bytes puts the load on the
            // carried chain (the long-segment corpus runs every one of
            // the BREAK_WINDOW content bytes through the window; soup
            // runs one or two).
            // `has_non_ws` is false at every segment start (reset below
            // per break run, by cold, and at entry), so folding straight
            // into it is the segment's fold; non-ASCII bytes only raise
            // the fallback flag — they never fold, so the provisional
            // answer is always the ASCII bytes' exact contribution and
            // the fallback's recompute can only OR the same-or-truer
            // value in.
            let limit = (byte + BREAK_WINDOW).min(n);
            let mut i = byte;
            let mut saw_non_ascii = false;
            let mut brk = None;
            while i < limit {
                let b = bytes[i];
                if matches!(b, b'\r' | b'\n') {
                    brk = Some(i);
                    break;
                }
                if b < 0x80 {
                    has_non_ws |= !ascii_ws[b as usize];
                } else {
                    saw_non_ascii = true;
                }
                i += 1;
            }
            // Deliberately NO certificate touch here: this loop runs
            // once per soup segment, and every extra live value (the
            // certificate's lo/hi) is a register the whole loop state
            // can no longer stay in — the same failure mode the
            // hot/cold split itself exists to avoid. The window's bytes
            // are re-derived for free where they matter: cold's batch
            // extends across them inside one vectorized stride, and a
            // batch only fires once per ~CERT_CHUNK bytes regardless,
            // so advancing here would save nothing.
            let Some(brk) = brk else {
                // Window exhausted without a break: a long segment.
                // Escape to the cold path below (one pass through
                // 'outer's tail) and let the SIMD tools fold it. Nothing
                // has been folded into cp yet — cold recomputes the
                // whole segment — so the window's partial work is simply
                // dropped.
                break;
            };
            if saw_non_ascii {
                // The non-ASCII fallback: one decode pass answers the
                // filter exactly (`char::is_whitespace`, NBSP and the
                // U+2000..U+3000 spaces included) and recounts the
                // codepoints — the continuation arithmetic the ASCII
                // path gets for free (byte length IS codepoint length
                // there) is folded into the same rare pass instead of
                // being paid per byte in the hot loop above.
                let mut non_ws = false;
                let mut cps = 0usize;
                for ch in text[byte..brk].chars() {
                    non_ws |= !ch.is_whitespace();
                    cps += 1;
                }
                has_non_ws |= non_ws;
                cp += cps;
            } else {
                cp += brk - byte;
            }
            if has_non_ws {
                bounds.push((seg_start, cp));
            }
            // The break-unit walk: consume the unit at `brk` and any
            // further back-to-back units — a `\r` immediately followed
            // by `\n` is ONE unit of two bytes (both codepoints), every
            // other break byte a single-byte unit. Each further unit
            // ends an EMPTY line (`has_non_ws` is false for it until the
            // reset below, the same reset the per-`char` machine applied
            // per unit), so nothing is pushed inside the walk; the next
            // segment starts at a non-break byte by the exit check, so
            // hot never folds an empty segment except after cold hands
            // it a break byte. (Branches, deliberately: a branchless
            // unit-length spelling forces the pair lookahead's loads
            // unconditionally and tangles the walk's exit check into
            // them.)
            let mut j = brk;
            loop {
                if bytes[j] == b'\r' && j + 1 < n && bytes[j + 1] == b'\n' {
                    cp += 2;
                    j += 2;
                } else {
                    cp += 1;
                    j += 1;
                }
                if j >= n || !matches!(bytes[j], b'\r' | b'\n') {
                    break;
                }
            }
            seg_start = cp;
            has_non_ws = false;
            byte = j;
        }
        if byte >= n {
            // Hot ran to end of text (a trailing short segment or break
            // run); the close below emits the final line if it earned
            // one. Cold cannot run: there is no segment left.
            break 'outer;
        }
        // ---- cold: the window came up empty — a long segment ----
        // One vectorized hop for the break, then the fold over the
        // whole segment (cp has not moved yet — cold recomputes the
        // whole segment's count — and `has_non_ws |=` over the whole
        // segment merely re-ORs the window's partial contribution, the
        // same bytes) so the answers come from the SIMD pass instead of
        // 80+ scalar iterations. `seg_end` is the hop's hit or end of
        // text. The purity question is the certificate's (`certify`,
        // one batched 4 KiB stride per ~50 segments, extending from
        // wherever the sliding range ended so the window bytes ride
        // inside the stride), not a per-segment `is_ascii()` slice call
        // — the per-call overhead of that spelling dominated the
        // realistic-density lanes, and the certificate answers most
        // segments with no call at all. The `bytes[byte] < 0x80`
        // pre-probe is `paragraph_bounds`' segment count's (see there
        // for the soundness argument): a non-ASCII first byte makes the
        // `certify` attempt a guaranteed failure, and on a CJK-dense
        // log (every segment starting with a multi-byte lead byte)
        // that is every segment — one compare lands straight in the
        // decode fold without paying the batch call, the locating scan,
        // and the reset a failed attempt performs. The query is
        // non-empty (`seg_end > byte` always — the window exhausted
        // without a break, so at least one byte remains), so the
        // empty-range `covers` corner cannot hide behind the probe
        // here either.
        let limit = (byte + BREAK_WINDOW).min(n);
        let seg_end = memchr2(b'\r', b'\n', &bytes[limit..]).map_or(n, |rel| limit + rel);
        if bytes[byte] < 0x80 && cert.certify(bytes, byte, seg_end) {
            // The segment is certified pure ASCII (no break byte exists
            // in it — the window saw none and the hop found none — and
            // no byte above 0x7F), so the table's non-whitespace answer
            // is exactly the filter's; the early exit answers within a
            // byte or two.
            let seg = &bytes[byte..seg_end];
            has_non_ws |= seg.iter().any(|&b| !ascii_ws[b as usize]);
            cp += seg.len();
        } else {
            let mut non_ws = false;
            let mut cps = 0usize;
            for ch in text[byte..seg_end].chars() {
                non_ws |= !ch.is_whitespace();
                cps += 1;
            }
            has_non_ws |= non_ws;
            cp += cps;
        }
        if seg_end == n {
            // No further break: the final segment is folded, cp is the
            // whole text's codepoint total, and the close below emits
            // the line it earned (if any).
            break 'outer;
        }
        if has_non_ws {
            bounds.push((seg_start, cp));
        }
        // The line-close bookkeeping hot's walk tail applies, applied
        // here too; the break run at `seg_end` is left for hot, whose
        // next iteration begins ON a break byte and folds an empty
        // segment — free, and the one place the walk lives.
        seg_start = cp;
        has_non_ws = false;
        byte = seg_end;
    }
    // End of text closes the final line, and `has_non_ws` ALONE is the
    // phantom-line guarantee, the same argument as before in byte
    // terms: the flag resets together with `seg_start` at every break
    // run, and only a codepoint at or after `seg_start` can set it
    // after the last reset (the final segment's fold), so the flag
    // holding at end of text implies a real line at `(seg_start, cp)`.
    // A trailing break unit already reset it; empty text never enters
    // the loop; `[]` falls out with no special case.
    if has_non_ws {
        bounds.push((seg_start, cp));
    }
    bounds
}

/// [`chunk_by_words`]'s line-count twin: each chunk spans
/// `lines_per_chunk` consecutive [`line_bounds`] segments, `overlap`
/// LINES repeated at the start of the next chunk. Same contract, same
/// preconditions, same empty-input answer as its siblings; see
/// [`line_bounds`] for exactly what counts as a line here (a
/// content-carrying, newline-terminated segment — blank lines neither
/// count nor split a chunk's interior, "content" being the Unicode
/// `White_Space` reading [`line_bounds`] pins, not Python's
/// `str.isspace()` notion).
pub fn chunk_by_lines(text: &str, lines_per_chunk: usize, overlap: usize) -> Vec<(usize, usize)> {
    let bounds = line_bounds(text);
    chunk_by_segments(&bounds, lines_per_chunk, overlap)
}

#[cfg(test)]
mod tests {
    use super::*;
    // The differential oracles below are the pre-#22 spellings verbatim
    // (whole-text `Vec<char>` collects, `HashSet<usize>` boundary sets,
    // the random-access paragraph scan), which is why the tests module
    // re-imports what production no longer uses.
    use std::collections::HashSet;

    use crate::truncate_impl::grapheme_boundary_chars;

    /// The former `merge_mid_cluster_boundaries`, HashSet spelling, kept
    /// verbatim as the differential oracle for the bitmap spelling.
    fn merge_mid_cluster_boundaries_reference(
        bounds: Vec<(usize, usize)>,
        grapheme_set: &HashSet<usize>,
    ) -> Vec<(usize, usize)> {
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(bounds.len());
        for (start, end) in bounds {
            match merged.last_mut() {
                Some(last) if last.1 == start && !grapheme_set.contains(&start) => {
                    last.1 = end;
                }
                _ => merged.push((start, end)),
            }
        }
        merged
    }

    /// The former `chunk_by_words`, verbatim oracle: `Vec<char>` collect,
    /// `HashSet` boundary set, random-access whitespace slices.
    fn chunk_by_words_reference(
        text: &str,
        words_per_chunk: usize,
        overlap: usize,
    ) -> Vec<(usize, usize)> {
        let chars: Vec<char> = text.chars().collect();
        let grapheme_set: HashSet<usize> = grapheme_boundary_chars(text).into_iter().collect();
        let merged = merge_mid_cluster_boundaries_reference(
            segmentation_impl::word_bounds(text),
            &grapheme_set,
        );
        let bounds: Vec<(usize, usize)> = merged
            .into_iter()
            .filter(|&(start, end)| chars[start..end].iter().any(|c| !c.is_whitespace()))
            .collect();
        chunk_by_segments(&bounds, words_per_chunk, overlap)
    }

    /// The former `chunk_by_sentences`, verbatim oracle.
    fn chunk_by_sentences_reference(
        text: &str,
        sentences_per_chunk: usize,
        overlap: usize,
    ) -> Vec<(usize, usize)> {
        let grapheme_set: HashSet<usize> = grapheme_boundary_chars(text).into_iter().collect();
        let bounds = merge_mid_cluster_boundaries_reference(
            segmentation_impl::sentence_bounds(text),
            &grapheme_set,
        );
        chunk_by_segments(&bounds, sentences_per_chunk, overlap)
    }

    /// The former `paragraph_bounds`, verbatim oracle: the whole-text
    /// `Vec<char>` collect with random access and one-codepoint lookahead.
    fn paragraph_bounds_reference(text: &str) -> Vec<(usize, usize)> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }
        let mut bounds = Vec::new();
        let mut seg_start = 0usize;
        let mut i = 0usize;
        while i < n {
            if chars[i] == '\n' || chars[i] == '\r' {
                let run_start = i;
                let mut units = 0usize;
                while i < n && (chars[i] == '\n' || chars[i] == '\r') {
                    if chars[i] == '\r' && i + 1 < n && chars[i + 1] == '\n' {
                        i += 2;
                    } else {
                        i += 1;
                    }
                    units += 1;
                }
                if units >= 2 {
                    if seg_start < run_start {
                        bounds.push((seg_start, run_start));
                    }
                    seg_start = i;
                }
            } else {
                i += 1;
            }
        }
        if seg_start < n {
            bounds.push((seg_start, n));
        }
        bounds
    }

    /// The differential corpus: word/sentence/paragraph shapes — prose,
    /// CRLF and lone-CR runs (every unit-counting case the paragraph
    /// scanner has: CRLF pairs, mixed \n\r, trailing and leading runs),
    /// Thai SARA AM (the merge's reason to exist), whitespace-only and
    /// whitespace-heavy text (the token filter's drop-everything and
    /// ride-along cases), degenerate runs, and the #30 non-ASCII shapes
    /// (precomposed and combining accents, NBSP, an astral char, CJK)
    /// that pin the byte-path scanners' per-segment non-ASCII fallback
    /// and codepoint-offset bookkeeping — NBSP-only lines are blank to
    /// `line_bounds` (White_Space) but ordinary content to
    /// `paragraph_bounds` (no content filter), astral chars make byte
    /// length != codepoint count, and the long mixed entry's ~19-byte
    /// segments cross the density-guard window so the memchr hop and
    /// the non-ASCII fold co-occur.
    fn differential_corpus() -> Vec<String> {
        vec![
            String::new(),
            "   ".to_string(),
            "a".to_string(),
            "one two three four five six seven eight".to_string(),
            "Alpha beta gamma delta. Epsilon zeta eta. Theta iota kappa.".to_string(),
            "x0\u{0E33}y0\u{0E33}z".to_string(),
            "One 0\u{0E33} fish. Two 0\u{0E33} fish.".to_string(),
            "a\n\nb\n\n\nc".to_string(),
            "a\r\nb\r\n\r\nc".to_string(),
            "a\rb".to_string(),
            "\r\n\r\n\r\n".to_string(),
            "\n\na".to_string(),
            "a\n\n".to_string(),
            "a\r\rb\n\n\nc".to_string(),
            "para one\n\npara two\nsingle\n\npara three".to_string(),
            " \t \n\n \t ".to_string(),
            "e\u{0301}e\u{0301} words here".to_string(),
            "\u{00E9}x\n\u{00E9}\u{0301}y\r\n\u{1F600}".to_string(),
            "x\n\u{0301}y".to_string(),
            "\u{00A0}\n\u{00A0}\u{00A0}\n".to_string(),
            "\u{4E2D}\u{6587}\n\n\u{4E2D}\u{6587}".to_string(),
            "\u{00E9}\u{0301} \u{4E2D}\u{00A0}\u{1F600}x\r\r\n\n\u{00E9}".to_string(),
            "\u{00E9}\u{0301}\u{4E2D}x words here\r\n".repeat(40),
            // The ASCII whitespace set's two members a `matches!` range
            // check or std's `is_ascii_whitespace` (which omits VT)
            // would drop: a VT-only and an FF-only line are BLANK lines
            // (White_Space), pinned here because the mutation "drop
            // 0x0B/0x0C from ASCII_WS" changes `chunk_by_lines` output
            // and nothing else in this module's corpus would catch it.
            "a\n\u{0B}\nb".to_string(),
            "a\n\u{0C}\nb".to_string(),
            "\u{0B}\n\u{0C}\n\u{0B}\u{0C}".to_string(),
            "q".repeat(300),
            "word ".repeat(120),
        ]
    }

    #[test]
    fn streaming_spellings_match_the_former_implementations_exactly() {
        for text in differential_corpus() {
            // paragraph_bounds: the state machine against the random-access
            // scan, over every corpus shape (CRLF pairing, unit counts at
            // the split threshold, leading/trailing runs).
            assert_eq!(
                paragraph_bounds(&text),
                paragraph_bounds_reference(&text),
                "paragraph_bounds divergence on {text:?}"
            );
            // The unit-count chunkers over the validated envelope
            // (per_chunk >= 1, overlap < per_chunk), sweeping the
            // boundary-adjacent values.
            for per_chunk in 1usize..=6 {
                for overlap in [0usize, 1, per_chunk.saturating_sub(1)]
                    .into_iter()
                    .filter(|&o| o < per_chunk)
                {
                    assert_eq!(
                        chunk_by_words(&text, per_chunk, overlap),
                        chunk_by_words_reference(&text, per_chunk, overlap),
                        "chunk_by_words divergence: text={text:?} per={per_chunk} ov={overlap}"
                    );
                    assert_eq!(
                        chunk_by_sentences(&text, per_chunk, overlap),
                        chunk_by_sentences_reference(&text, per_chunk, overlap),
                        "chunk_by_sentences divergence: text={text:?} per={per_chunk} ov={overlap}"
                    );
                    assert_eq!(
                        chunk_by_paragraphs(&text, per_chunk, overlap),
                        chunk_by_segments(&paragraph_bounds_reference(&text), per_chunk, overlap),
                        "chunk_by_paragraphs divergence: text={text:?} per={per_chunk} ov={overlap}"
                    );
                    assert_eq!(
                        chunk_by_lines(&text, per_chunk, overlap),
                        chunk_by_segments(&line_bounds_reference(&text), per_chunk, overlap),
                        "chunk_by_lines divergence: text={text:?} per={per_chunk} ov={overlap}"
                    );
                }
            }
        }
    }

    #[test]
    fn paragraph_state_machine_survives_a_deterministic_newline_soup() {
        // Pseudo-random text over exactly the alphabet the paragraph
        // scanner branches on (\r, \n, CRLF pairings, whitespace riding
        // inside segments, and one ordinary character), so the unit
        // counting is checked against the random-access oracle on runs
        // no hand-written corpus anticipates — including runs that end
        // at end-of-text.
        let mut state = 0x853C49E6748FEA9Bu64;
        let alphabet = ['x', ' ', '\t', '\r', '\n'];
        for _ in 0..300 {
            let mut text = String::new();
            for _ in 0..(state % 60 + 1) as usize {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                text.push(alphabet[(state >> 33) as usize % alphabet.len()]);
            }
            assert_eq!(
                paragraph_bounds(&text),
                paragraph_bounds_reference(&text),
                "divergence on {text:?}"
            );
        }
        // The #30 byte-scan extension: the same soup discipline over a
        // non-ASCII alphabet — a precomposed accent, a combining mark
        // (also right after breaks, the documented orphan-cluster
        // non-issue), NBSP (White_Space above 0x7F), an astral char
        // (byte length != codepoint count) and CJK — so the per-segment
        // codepoint-offset bookkeeping the byte walk now does in bytes
        // is differentially pinned on shapes the ASCII alphabet cannot
        // reach, with the unit-count chunker riding the same sweep over
        // the oracle's bounds. Astral chars also stretch ordinary runs
        // past the density-guard window, so the memchr-hop path
        // and the inline path both take their turn under the oracle.
        let mut state = 0x0D1B54A32B970E89u64;
        let alphabet = [
            'x',
            '\r',
            '\n',
            '\u{00E9}',
            '\u{0301}',
            '\u{00A0}',
            '\u{1F600}',
            '\u{4E2D}',
        ];
        for _ in 0..300 {
            let mut text = String::new();
            for _ in 0..(state % 60 + 1) as usize {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                text.push(alphabet[(state >> 33) as usize % alphabet.len()]);
            }
            assert_eq!(
                paragraph_bounds(&text),
                paragraph_bounds_reference(&text),
                "divergence on {text:?}"
            );
            for per_chunk in 1usize..=6 {
                for overlap in [0usize, 1, per_chunk.saturating_sub(1)]
                    .into_iter()
                    .filter(|&o| o < per_chunk)
                {
                    assert_eq!(
                        chunk_by_paragraphs(&text, per_chunk, overlap),
                        chunk_by_segments(&paragraph_bounds_reference(&text), per_chunk, overlap),
                        "chunk_by_paragraphs divergence: text={text:?} per={per_chunk} ov={overlap}"
                    );
                }
            }
        }
    }

    // ---- grapheme-cluster safety (the truncate_impl regression, re-derived here) ----

    #[test]
    fn chunk_by_words_never_splits_a_thai_sara_am_cluster_across_two_words() {
        // Raw word_bounds("x0ำy0ำz") = [(0,2)="x0", (2,3)="ำ", (3,5)="y0",
        // (5,6)="ำ", (6,7)="z"]: TWO combining sequences each split into
        // a base-segment + a lone-combining-mark segment. Neither "ำ"
        // segment is whitespace, so a whitespace-only filter would
        // NOT catch this: chunk_by_words(text, 1, 0) would silently
        // return a chunk containing only the bare combining mark. The
        // merge step must fuse each pair into one real word first.
        let text = "x0\u{0E33}y0\u{0E33}z";
        let chunks = chunk_by_words(text, 1, 0);
        let slice =
            |(a, b): (usize, usize)| -> String { text.chars().skip(a).take(b - a).collect() };
        assert_eq!(chunks.len(), 3, "chunks: {chunks:?}");
        assert_eq!(slice(chunks[0]), "x0\u{0E33}");
        assert_eq!(slice(chunks[1]), "y0\u{0E33}");
        assert_eq!(slice(chunks[2]), "z");
        // No chunk is a bare, unattached combining mark.
        for &c in &chunks {
            assert_ne!(slice(c), "\u{0E33}");
        }
    }

    #[test]
    fn chunk_by_sentences_never_splits_a_grapheme_cluster_across_two_sentences() {
        // sentence_bounds is coarser than word_bounds and, empirically,
        // does not isolate the SARA AM combining mark into its own
        // sentence segment for ordinary sentence-terminated text, but
        // the merge step runs unconditionally (see chunk_by_sentences'
        // implementation), so this pins that no chunk produced by it
        // ever starts or ends strictly inside a cluster, regardless.
        let text = "One 0\u{0E33} fish. Two 0\u{0E33} fish.";
        let valid: HashSet<usize> = grapheme_boundary_chars(text).into_iter().collect();
        for sentences_per_chunk in 1..=3 {
            let chunks = chunk_by_sentences(text, sentences_per_chunk, 0);
            for &(a, b) in &chunks {
                assert!(valid.contains(&a), "start {a} mid-cluster: {chunks:?}");
                assert!(valid.contains(&b), "end {b} mid-cluster: {chunks:?}");
            }
        }
    }

    // ---- chunk_by_words / chunk_by_sentences ----

    #[test]
    fn chunk_by_words_groups_exact_word_counts() {
        // word_bounds("the cat sat on the mat") segments: the/ /cat/ /sat/
        // /on/ /the/ /mat: 11 raw segments (6 real word tokens + 5
        // inter-word spaces, each its own WB segment), but chunk_by_words
        // filters the whitespace-only segments out FIRST so "2 words per
        // chunk" means 2 real tokens, not 2 raw segments (which would
        // silently be ~1 real word per chunk on ordinary prose). 2 words
        // per chunk, no overlap: 3 chunks of 2 real words each, spans
        // still contiguous slices of the ORIGINAL text (inter-word space
        // inside a chunk rides along naturally).
        let text = "the cat sat on the mat";
        let chunks = chunk_by_words(text, 2, 0);
        let slice =
            |(a, b): (usize, usize)| -> String { text.chars().skip(a).take(b - a).collect() };
        assert_eq!(chunks.len(), 3);
        assert_eq!(slice(chunks[0]), "the cat");
        assert_eq!(slice(chunks[1]), "sat on");
        assert_eq!(slice(chunks[2]), "the mat");
        // Non-overlapping in this overlap=0 case, but chunks are NOT
        // necessarily contiguous any more (the space between "cat" and
        // "sat" belongs to neither chunk): this function makes no
        // covering-partition claim, unlike chunk_text.
        for w in chunks.windows(2) {
            assert!(w[1].0 >= w[0].1);
        }
    }

    #[test]
    fn chunk_by_words_counts_real_tokens_not_raw_word_bounds_segments() {
        // The regression this pins: word_bounds gives an inter-word space
        // run its OWN segment, so a naive "group N raw segments" reading
        // of "words_per_chunk" would silently mean roughly HALF as many
        // real words per chunk on ordinary space-separated prose.
        // "one two three four five six seven" has 7 real word tokens (13
        // raw word_bounds segments, 7 words + 6 spaces): 3 per chunk
        // must yield exactly ceil(7/3) = 3 chunks, the last holding the
        // remaining 1 word, never the wrong (roughly-halved) count a
        // raw-segment grouping would produce.
        let text = "one two three four five six seven";
        let chunks = chunk_by_words(text, 3, 0);
        assert_eq!(chunks.len(), 3, "chunks: {chunks:?}");
        let slice =
            |(a, b): (usize, usize)| -> String { text.chars().skip(a).take(b - a).collect() };
        assert_eq!(slice(chunks[0]), "one two three");
        assert_eq!(slice(chunks[1]), "four five six");
        assert_eq!(slice(chunks[2]), "seven");
    }

    #[test]
    fn chunk_by_words_overlap_repeats_words_at_each_boundary() {
        let text = "one two three four five six seven";
        let chunks = chunk_by_words(text, 3, 1);
        assert!(chunks.len() >= 2);
        for w in chunks.windows(2) {
            assert!(
                w[1].0 < w[0].1,
                "no overlap between {:?} and {:?}",
                w[0],
                w[1]
            );
            assert!(
                w[1].0 > w[0].0,
                "no forward progress: {:?} -> {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn chunk_by_words_overlap_actually_shares_content_langchain_34804_regression() {
        // LangChain issue #34804: chunk_overlap was silently a no-op
        // except when a size-overflow forced a merge: a real, shipped
        // bug in the most popular chunking library. The regression this
        // pins: consecutive chunks must share GENUINE, non-empty text,
        // not merely satisfy a position check that happens to coincide
        // with hitting a size ceiling. 8 words, no chunk here divides
        // evenly to a size ceiling by coincidence; the overlap must still
        // manifest as literal shared text on every transition.
        let text = "alpha beta gamma delta epsilon zeta eta theta";
        let chunks = chunk_by_words(text, 3, 1);
        assert!(chunks.len() >= 2, "chunks: {chunks:?}");
        let slice =
            |(a, b): (usize, usize)| -> String { text.chars().skip(a).take(b - a).collect() };
        for w in chunks.windows(2) {
            let (prev_start, prev_end) = w[0];
            let (next_start, next_end) = w[1];
            let shared: String = text
                .chars()
                .skip(next_start)
                .take(prev_end.saturating_sub(next_start))
                .collect();
            assert!(
                !shared.trim().is_empty(),
                "no genuine shared content between {:?} and {:?}",
                w[0],
                w[1]
            );
            // The shared span reads identically from EITHER chunk's own
            // text (it's the same underlying offsets on both sides).
            let from_prev = &slice((prev_start, prev_end))[(next_start - prev_start)..];
            let from_next = &slice((next_start, next_end))[..(prev_end - next_start)];
            assert_eq!(from_prev, from_next);
            assert_eq!(from_prev, shared);
        }
    }

    #[test]
    fn chunk_by_sentences_groups_exact_sentence_counts() {
        let text = "One. Two. Three. Four. Five.";
        let chunks = chunk_by_sentences(text, 2, 0);
        assert!(chunks.len() >= 2);
        let mut prev_end = 0usize;
        for &(a, b) in &chunks {
            assert_eq!(a, prev_end);
            prev_end = b;
        }
        assert_eq!(prev_end, text.chars().count());
    }

    #[test]
    fn chunk_by_sentences_overlap_repeats_sentences() {
        let text = "One. Two. Three. Four. Five. Six.";
        let chunks = chunk_by_sentences(text, 3, 1);
        assert!(chunks.len() >= 2);
        for w in chunks.windows(2) {
            assert!(w[1].0 < w[0].1);
            assert!(w[1].0 > w[0].0);
        }
    }

    #[test]
    fn chunk_by_sentences_overlap_actually_shares_content_langchain_34804_regression() {
        // Same LangChain #34804 regression as chunk_by_words' twin: the
        // shared span must be real, extractable text, not merely a
        // position check; and this must hold even though 6 sentences
        // over 3-per-chunk/1-overlap doesn't divide to any size ceiling.
        let text = "One. Two. Three. Four. Five. Six.";
        let chunks = chunk_by_sentences(text, 3, 1);
        assert!(chunks.len() >= 2, "chunks: {chunks:?}");
        let chars: Vec<char> = text.chars().collect();
        let slice = |(a, b): (usize, usize)| -> String { chars[a..b].iter().collect() };
        for w in chunks.windows(2) {
            let (_, prev_end) = w[0];
            let (next_start, _) = w[1];
            assert!(
                next_start < prev_end,
                "no overlap: {:?} -> {:?}",
                w[0],
                w[1]
            );
            let shared: String = chars[next_start..prev_end].iter().collect();
            assert!(
                !shared.trim().is_empty(),
                "no genuine shared content between {:?} and {:?}",
                w[0],
                w[1]
            );
            assert!(slice(w[0]).ends_with(&shared));
            assert!(slice(w[1]).starts_with(&shared));
        }
    }

    #[test]
    fn chunk_by_word_or_sentence_empty_text_is_no_chunks() {
        assert_eq!(chunk_by_words("", 3, 0), Vec::<(usize, usize)>::new());
        assert_eq!(chunk_by_sentences("", 3, 0), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn paragraph_bounds_splits_on_blank_line_runs() {
        let text = "First para.\n\nSecond para.\n\nThird para.";
        let bounds = paragraph_bounds(text);
        let chars: Vec<char> = text.chars().collect();
        let pieces: Vec<String> = bounds
            .iter()
            .map(|&(a, b)| chars[a..b].iter().collect())
            .collect();
        assert_eq!(pieces, vec!["First para.", "Second para.", "Third para."]);
    }

    #[test]
    fn paragraph_bounds_a_single_newline_is_not_a_break() {
        let text = "line one\nline two";
        assert_eq!(paragraph_bounds(text), vec![(0, text.chars().count())]);
    }

    #[test]
    fn paragraph_bounds_three_plus_newlines_are_still_one_break() {
        let text = "First.\n\n\n\nSecond.";
        let bounds = paragraph_bounds(text);
        assert_eq!(bounds.len(), 2);
        let chars: Vec<char> = text.chars().collect();
        let second: String = chars[bounds[1].0..bounds[1].1].iter().collect();
        assert_eq!(second, "Second.");
    }

    #[test]
    fn paragraph_bounds_crlf_run_counts_as_two_units() {
        let text = "First.\r\n\r\nSecond.";
        let bounds = paragraph_bounds(text);
        let chars: Vec<char> = text.chars().collect();
        let pieces: Vec<String> = bounds
            .iter()
            .map(|&(a, b)| chars[a..b].iter().collect())
            .collect();
        assert_eq!(pieces, vec!["First.", "Second."]);
    }

    #[test]
    fn paragraph_bounds_lone_cr_run_counts_too() {
        let text = "First.\r\rSecond.";
        let bounds = paragraph_bounds(text);
        assert_eq!(bounds.len(), 2);
    }

    #[test]
    fn paragraph_bounds_leading_and_trailing_blank_runs_are_trimmed() {
        let text = "\n\n\nHello\n\n\n";
        let bounds = paragraph_bounds(text);
        let chars: Vec<char> = text.chars().collect();
        assert_eq!(bounds.len(), 1);
        let piece: String = chars[bounds[0].0..bounds[0].1].iter().collect();
        assert_eq!(piece, "Hello");
    }

    #[test]
    fn paragraph_bounds_no_breaks_is_one_paragraph() {
        let text = "just one paragraph, no blank lines at all";
        assert_eq!(paragraph_bounds(text), vec![(0, text.chars().count())]);
    }

    #[test]
    fn paragraph_bounds_empty_text_is_no_paragraphs() {
        assert_eq!(paragraph_bounds(""), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn chunk_by_paragraphs_groups_exact_paragraph_counts() {
        // Unlike chunk_by_sentences (whose UAX #29 segments already carry
        // their own trailing whitespace, so chunks stay contiguous),
        // paragraph_bounds EXCLUDES the separating blank-line run from
        // each paragraph's span, so, like chunk_by_words, chunks here
        // are not necessarily contiguous; assert on content, not on
        // gapless coverage.
        let text = "P1.\n\nP2.\n\nP3.\n\nP4.\n\nP5.";
        let chunks = chunk_by_paragraphs(text, 2, 0);
        let slice =
            |(a, b): (usize, usize)| -> String { text.chars().skip(a).take(b - a).collect() };
        assert_eq!(chunks.len(), 3);
        assert_eq!(slice(chunks[0]), "P1.\n\nP2.");
        assert_eq!(slice(chunks[1]), "P3.\n\nP4.");
        assert_eq!(slice(chunks[2]), "P5.");
    }

    #[test]
    fn chunk_by_paragraphs_overlap_repeats_paragraphs() {
        let text = "P1.\n\nP2.\n\nP3.\n\nP4.\n\nP5.\n\nP6.";
        let chunks = chunk_by_paragraphs(text, 3, 1);
        assert!(chunks.len() >= 2);
        for w in chunks.windows(2) {
            assert!(w[1].0 < w[0].1);
            assert!(w[1].0 > w[0].0);
        }
    }

    #[test]
    fn chunk_by_paragraphs_overlap_actually_shares_content_langchain_34804_regression() {
        // Same LangChain #34804 regression, paragraph-count sibling: the
        // shared span between consecutive chunks must be real,
        // extractable paragraph text.
        let text = "P1.\n\nP2.\n\nP3.\n\nP4.\n\nP5.\n\nP6.";
        let chunks = chunk_by_paragraphs(text, 3, 1);
        assert!(chunks.len() >= 2, "chunks: {chunks:?}");
        let chars: Vec<char> = text.chars().collect();
        let slice = |(a, b): (usize, usize)| -> String { chars[a..b].iter().collect() };
        for w in chunks.windows(2) {
            let (_, prev_end) = w[0];
            let (next_start, _) = w[1];
            assert!(
                next_start < prev_end,
                "no overlap: {:?} -> {:?}",
                w[0],
                w[1]
            );
            let shared: String = chars[next_start..prev_end].iter().collect();
            assert!(
                !shared.trim().is_empty(),
                "no genuine shared content between {:?} and {:?}",
                w[0],
                w[1]
            );
            assert!(slice(w[0]).ends_with(&shared));
            assert!(slice(w[1]).starts_with(&shared));
        }
    }

    #[test]
    fn chunk_by_paragraphs_empty_text_is_no_chunks() {
        assert_eq!(chunk_by_paragraphs("", 3, 0), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn chunk_by_paragraphs_single_chunk_when_fewer_paragraphs_than_per_chunk() {
        let text = "Only one paragraph here.";
        let chunks = chunk_by_paragraphs(text, 100, 0);
        assert_eq!(chunks, vec![(0, text.chars().count())]);
    }

    #[test]
    fn chunk_by_words_single_chunk_when_fewer_words_than_per_chunk() {
        let text = "hi there";
        let chunks = chunk_by_words(text, 100, 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], (0, text.chars().count()));
    }

    #[test]
    fn chunk_by_words_last_chunk_may_be_partial() {
        // 5 word-level segments spelled out won't divide evenly by a
        // stride of 2 in general; confirm the last chunk just takes
        // whatever remains rather than panicking or dropping content.
        let text = "alpha beta gamma";
        let bounds = segmentation_impl::word_bounds(text);
        let chunks = chunk_by_words(text, 2, 0);
        let last = *chunks.last().unwrap();
        assert_eq!(last.1, bounds.last().unwrap().1);
    }

    #[test]
    #[should_panic(expected = "per_chunk must be at least 1")]
    fn chunk_by_segments_panics_on_zero_per_chunk() {
        // Pins that the `per_chunk > 0` precondition is an `assert!`, not a
        // `debug_assert!` that a release build would silently skip: the
        // three pyo3 wrappers already reject this before it ever reaches
        // here, but this function is reachable directly within the crate,
        // and a regression back to `debug_assert!` would only be caught by
        // a release-mode run, never by `cargo test`'s debug profile. This
        // test can't tell `assert!` from `debug_assert!` either (both panic
        // in a debug test build), but it does pin that the condition and
        // message stay intact, and any regression that inlines/removes the
        // check outright still fails it.
        let bounds = [(0usize, 1usize), (1, 2), (2, 3)];
        chunk_by_segments(&bounds, 0, 0);
    }

    #[test]
    #[should_panic(expected = "overlap must be less than per_chunk")]
    fn chunk_by_segments_panics_when_overlap_at_least_per_chunk() {
        let bounds = [(0usize, 1usize), (1, 2), (2, 3)];
        chunk_by_segments(&bounds, 2, 2);
    }

    #[test]
    fn chunk_by_segments_forward_progress_is_unconditional_by_construction() {
        // per_chunk - overlap >= 1 is the caller's precondition (validated
        // at the pyo3 boundary); confirm the resulting stride actually
        // produces strictly increasing starts over a battery, matching
        // chunk_text_overlapping's own runtime-checked guarantee (this one
        // needs no runtime check since stride >= 1 is structural).
        let text = "one two three four five six seven eight nine ten";
        for per_chunk in 1..6 {
            for overlap in 0..per_chunk {
                let chunks = chunk_by_words(text, per_chunk, overlap);
                let mut prev_start: Option<usize> = None;
                for &(start, _) in &chunks {
                    if let Some(p) = prev_start {
                        assert!(start > p, "per_chunk={per_chunk}, overlap={overlap}");
                    }
                    prev_start = Some(start);
                }
            }
        }
    }

    // ---- chunk_by_lines / line_bounds ----

    /// The random-access `line_bounds` oracle: the whole-text `Vec<char>`
    /// collect with a one-codepoint CRLF lookahead, the pre-#22 spelling
    /// style `paragraph_bounds_reference` keeps — the differential pin for
    /// the streaming state machine above.
    fn line_bounds_reference(text: &str) -> Vec<(usize, usize)> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }
        let is_break = |c: char| c == '\n' || c == '\r';
        // A '\r' ALWAYS opens a unit (lone CR, or the first half of a CRLF
        // pair); a '\n' opens one only when it did NOT ride in as the
        // second half of a pair (no '\r' immediately before it).
        let is_unit_start =
            |i: usize| chars[i] == '\r' || (chars[i] == '\n' && !(i > 0 && chars[i - 1] == '\r'));
        let line_has_content = |a: usize, b: usize| chars[a..b].iter().any(|c| !c.is_whitespace());
        let mut bounds = Vec::new();
        let mut seg_start = 0usize;
        let mut i = 0usize;
        while i < n {
            if is_break(chars[i]) {
                if is_unit_start(i) && line_has_content(seg_start, i) {
                    bounds.push((seg_start, i));
                }
                // Whether this codepoint opens a unit (a lone break) or
                // completes one (the '\n' half of a CRLF pair the '\r'
                // opened), the next line begins after it.
                seg_start = i + 1;
            }
            i += 1;
        }
        if seg_start < n && line_has_content(seg_start, n) {
            bounds.push((seg_start, n));
        }
        bounds
    }

    #[test]
    fn line_bounds_streaming_matches_the_random_access_oracle_exactly() {
        for text in differential_corpus() {
            assert_eq!(
                line_bounds(&text),
                line_bounds_reference(&text),
                "line_bounds divergence on {text:?}"
            );
        }
    }

    #[test]
    fn line_bounds_state_machine_survives_a_deterministic_newline_soup() {
        // The paragraph scanner's own discipline, applied to the line
        // scanner: pseudo-random text over exactly the alphabet the state
        // machine branches on (\r, \n, CRLF pairings, one ordinary
        // character), so the unit counting is checked against the
        // random-access oracle on runs no hand-written corpus anticipates.
        let mut state = 0x9E3779B97F4A7C15u64;
        // VT and FF ride the alphabet alongside the space and tab: the
        // ASCII whitespace members whose omission from the fold (a
        // `matches!` range slip, or std's `is_ascii_whitespace`, which
        // drops 0x0B) would silently reclassify blank lines as content —
        // the exact mutation the corpus's VT/FF entries pin at
        // hand-written shapes and this soup pins under arbitrary density
        // and adjacency.
        let alphabet = ['x', '\r', '\n', ' ', '\t', '\u{0B}', '\u{0C}'];
        for _ in 0..300 {
            let mut text = String::new();
            for _ in 0..(state % 60 + 1) as usize {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                text.push(alphabet[(state >> 33) as usize % alphabet.len()]);
            }
            assert_eq!(
                line_bounds(&text),
                line_bounds_reference(&text),
                "divergence on {text:?}"
            );
            // The unit-count windowing over the same arbitrary text: same
            // per_chunk/overlap envelope the paragraph differential sweeps.
            let bounds = line_bounds(&text);
            for per_chunk in 1usize..=6 {
                for overlap in [0usize, 1, per_chunk.saturating_sub(1)]
                    .into_iter()
                    .filter(|&o| o < per_chunk)
                {
                    assert_eq!(
                        chunk_by_lines(&text, per_chunk, overlap),
                        chunk_by_segments(&bounds, per_chunk, overlap),
                        "chunk_by_lines divergence: text={text:?} per={per_chunk} ov={overlap}"
                    );
                }
            }
        }
        // The #30 byte-scan extension, the paragraph soup's twin: the
        // same discipline over a non-ASCII alphabet (with ' ' kept in, so
        // ASCII whitespace and NBSP hit the filter from the same texts)
        // — pinning the non-ASCII fallback path (NBSP blank, accents and
        // the astral char content) and the byte-vs-codepoint offset
        // bookkeeping against the random-access oracle, with the
        // unit-count chunker riding the sweep.
        let mut state = 0x2545F4914F6CDD1Du64;
        let alphabet = [
            'x',
            '\r',
            '\n',
            ' ',
            '\u{00E9}',
            '\u{0301}',
            '\u{00A0}',
            '\u{1F600}',
            '\u{4E2D}',
        ];
        for _ in 0..300 {
            let mut text = String::new();
            for _ in 0..(state % 60 + 1) as usize {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                text.push(alphabet[(state >> 33) as usize % alphabet.len()]);
            }
            assert_eq!(
                line_bounds(&text),
                line_bounds_reference(&text),
                "divergence on {text:?}"
            );
            let bounds = line_bounds(&text);
            for per_chunk in 1usize..=6 {
                for overlap in [0usize, 1, per_chunk.saturating_sub(1)]
                    .into_iter()
                    .filter(|&o| o < per_chunk)
                {
                    assert_eq!(
                        chunk_by_lines(&text, per_chunk, overlap),
                        chunk_by_segments(&bounds, per_chunk, overlap),
                        "chunk_by_lines divergence: text={text:?} per={per_chunk} ov={overlap}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore] // the pre-PR long-run sweep: `cargo test -- --ignored` runs it
    fn both_scanners_survive_a_200k_case_whitespace_bearing_newline_soup() {
        // The committed long-run artifact for the byte scans: a
        // long-run claim needs a committed test, not a prose note.
        // 200k pseudo-random
        // cases over a WHITESPACE-BEARING alphabet — ' ' and '\t' join
        // the breaks and the ordinary character, so blank runs, mixed
        // content/blank segments, and every break density collide with
        // the real-line filter's whitespace handling, the ASCII
        // certificate's batch machinery, and the CRLF pairing
        // at arbitrary adjacencies — with BOTH scanners asserted
        // against their random-access oracles per case. The in-suite
        // soups above run the same generator logic at 300 cases each;
        // this is the same contract at a length only a deliberate long
        // run pays for.
        let mut state = 0x853C49E6748FEA9Bu64;
        let alphabet = ['x', ' ', '\t', '\r', '\n'];
        for case in 0..200_000 {
            let mut text = String::new();
            for _ in 0..(state % 60 + 1) as usize {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                text.push(alphabet[(state >> 33) as usize % alphabet.len()]);
            }
            assert_eq!(
                paragraph_bounds(&text),
                paragraph_bounds_reference(&text),
                "paragraph divergence on case {case}: {text:?}"
            );
            assert_eq!(
                line_bounds(&text),
                line_bounds_reference(&text),
                "line divergence on case {case}: {text:?}"
            );
        }
    }

    #[test]
    fn line_bounds_splits_on_every_break_unit() {
        let text = "one\ntwo\r\nthree\rfour";
        let bounds = line_bounds(text);
        let chars: Vec<char> = text.chars().collect();
        let pieces: Vec<String> = bounds
            .iter()
            .map(|&(a, b)| chars[a..b].iter().collect())
            .collect();
        assert_eq!(pieces, vec!["one", "two", "three", "four"]);
    }

    #[test]
    fn line_bounds_crlf_pair_is_one_unit_and_never_torn() {
        // "a\r\nb": the line ends at the '\r', the next begins after the
        // '\n' — the pair is consumed as one break, and the '\n' is not a
        // second line ending that would mint an empty line between them.
        let text = "a\r\nb";
        assert_eq!(line_bounds(text), vec![(0, 1), (3, 4)]);
    }

    #[test]
    fn line_bounds_blank_lines_are_not_lines() {
        // The real-token discipline: blank lines neither count nor split —
        // "a\n\nb" is TWO lines (the empty line between is dropped), and
        // whitespace-only text is zero lines, the same answer
        // chunk_by_words gives pure-whitespace input.
        assert_eq!(line_bounds("a\n\nb"), vec![(0, 1), (3, 4)]);
        assert_eq!(line_bounds(" \n \t \n  "), Vec::<(usize, usize)>::new());
        assert_eq!(line_bounds(""), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn line_bounds_trailing_break_yields_no_phantom_line() {
        // `"a\n".split('\n')` mints a trailing ""; a break at end of text
        // terminates the last line and produces nothing after it.
        assert_eq!(line_bounds("a\n"), vec![(0, 1)]);
        assert_eq!(line_bounds("a\r\n"), vec![(0, 1)]);
    }

    #[test]
    fn line_bounds_interior_blank_lines_ride_inside_a_chunk_span() {
        // chunk spans are contiguous slices of the ORIGINAL text between
        // the first and last counted line's absolute offsets: the blank
        // line between two counted lines of the SAME chunk rides along
        // (exactly as inter-word whitespace rides along in chunk_by_words).
        let text = "msg one\n\nmsg two";
        let total = text.chars().count();
        let chunks = chunk_by_lines(text, 2, 0);
        assert_eq!(chunks, vec![(0, total)]);
        let chunks = chunk_by_lines(text, 1, 0);
        assert_eq!(chunks, vec![(0, 7), (9, 16)]);
    }

    #[test]
    fn chunk_by_lines_groups_exact_line_counts() {
        let text = "l1\nl2\nl3\nl4\nl5";
        let chunks = chunk_by_lines(text, 2, 0);
        let chars: Vec<char> = text.chars().collect();
        let pieces: Vec<String> = chunks
            .iter()
            .map(|&(a, b)| chars[a..b].iter().collect())
            .collect();
        assert_eq!(pieces, vec!["l1\nl2", "l3\nl4", "l5"]);
    }

    #[test]
    fn chunk_by_lines_final_chunk_may_hold_fewer_lines() {
        let text = "a\nb\nc\nd\ne";
        let chunks = chunk_by_lines(text, 3, 0);
        let chars: Vec<char> = text.chars().collect();
        let pieces: Vec<String> = chunks
            .iter()
            .map(|&(a, b)| chars[a..b].iter().collect())
            .collect();
        assert_eq!(pieces, vec!["a\nb\nc", "d\ne"]);
    }

    #[test]
    fn chunk_by_lines_single_chunk_when_fewer_lines_than_per_chunk() {
        let text = "only line";
        assert_eq!(
            chunk_by_lines(text, 100, 0),
            vec![(0, text.chars().count())]
        );
    }

    #[test]
    fn chunk_by_lines_overlap_repeats_whole_lines() {
        let text = "l1\nl2\nl3\nl4\nl5\nl6";
        let chunks = chunk_by_lines(text, 3, 1);
        assert!(chunks.len() >= 2, "chunks: {chunks:?}");
        let slice =
            |(a, b): (usize, usize)| -> String { text.chars().skip(a).take(b - a).collect() };
        for w in chunks.windows(2) {
            let (_, prev_end) = w[0];
            let (next_start, _) = w[1];
            assert!(
                next_start < prev_end,
                "no overlap: {:?} -> {:?}",
                w[0],
                w[1]
            );
            // The shared span must be real, extractable text (the
            // LangChain #34804 regression, line-count sibling), and it
            // must be WHOLE lines: the overlap starts at a counted line's
            // own start, never inside it.
            let shared = slice((next_start, prev_end));
            assert!(!shared.trim().is_empty());
            assert!(slice(w[0]).ends_with(&shared));
            assert!(slice(w[1]).starts_with(&shared));
        }
    }

    #[test]
    fn chunk_by_lines_empty_or_blank_text_is_no_chunks() {
        assert_eq!(chunk_by_lines("", 3, 0), Vec::<(usize, usize)>::new());
        assert_eq!(
            chunk_by_lines(" \n \t\n  ", 3, 0),
            Vec::<(usize, usize)>::new()
        );
    }

    // ---- the committed audit harnesses ----
    //
    // The red-team differential audit (~1.2M probe cases) ran its
    // harnesses as adhoc /tmp scripts against the installed extension;
    // zero divergences found, but zero-divergence coverage that lives
    // outside the repo protects nothing after /tmp is wiped. The tests
    // below convert the audit's harness SHAPES into committed gates over
    // the same two random-access oracles the corpus and soups already
    // use: exhaustive short-string enumeration (every string over a
    // break-bearing alphabet up to a length budget), certificate-boundary
    // surgicals (non-ASCII bytes at exact CERT_CHUNK-multiple offsets on
    // multi-KiB texts, the region the sub-4096-byte corpus cannot reach),
    // a window-phase/density sweep, and the systematic White_Space zoo.
    // The runtime budgets are measured, not guessed: the whole block is
    // sized to keep `cargo test`'s debug-build wall time in the same
    // ballpark as the suite it joins.

    /// The audit's exhaustive-alphabet walker: every string over
    /// `alphabet` from length 0 through `max_len`, shortlex order —
    /// for {'\r','\n'} at 16 that is 2^0 + ... + 2^16 = 131,071
    /// strings, the exact count the audit's A-exh-CR-LF probe ran. The
    /// buffer is reused across cases so the sweep's own allocation cost
    /// stays out of the differential measurement.
    fn for_each_string_over(alphabet: &[char], max_len: usize, mut case: impl FnMut(&str)) {
        let mut buf = String::new();
        for len in 0..=max_len {
            // An odometer over alphabet indices; the carry out of the
            // first digit means every string of this length was visited.
            let mut digits = vec![0usize; len];
            loop {
                buf.clear();
                buf.extend(digits.iter().map(|&d| alphabet[d]));
                case(&buf);
                let mut carry = true;
                for d in digits.iter_mut().rev() {
                    if !carry {
                        break;
                    }
                    *d += 1;
                    carry = *d == alphabet.len();
                    if carry {
                        *d = 0;
                    }
                }
                if carry {
                    break;
                }
            }
        }
    }

    /// The one assertion every audit shape reduces to: both byte
    /// scanners against their random-access oracles.
    fn assert_both_scanners_match_the_references(text: &str) {
        assert_eq!(
            line_bounds(text),
            line_bounds_reference(text),
            "line_bounds divergence on {text:?}"
        );
        assert_eq!(
            paragraph_bounds(text),
            paragraph_bounds_reference(text),
            "paragraph_bounds divergence on {text:?}"
        );
    }

    /// The unit-count windowers against `chunk_by_segments` over the
    /// ORACLE bounds — the same differential spelling the corpus sweep
    /// uses (the windower is shared, already-swept code; the SCANNER is
    /// the thing under test) — over the audit's per_chunk 1..=3 with
    /// overlap 0 and per-1.
    fn assert_windowing_matches_over_oracle_bounds(text: &str) {
        let lines = line_bounds_reference(text);
        let paras = paragraph_bounds_reference(text);
        for per_chunk in 1usize..=3 {
            let overlaps: &[usize] = if per_chunk == 1 {
                &[0]
            } else {
                &[0, per_chunk - 1]
            };
            for &overlap in overlaps {
                assert_eq!(
                    chunk_by_lines(text, per_chunk, overlap),
                    chunk_by_segments(&lines, per_chunk, overlap),
                    "chunk_by_lines divergence: text={text:?} per={per_chunk} ov={overlap}"
                );
                assert_eq!(
                    chunk_by_paragraphs(text, per_chunk, overlap),
                    chunk_by_segments(&paras, per_chunk, overlap),
                    "chunk_by_paragraphs divergence: text={text:?} per={per_chunk} ov={overlap}"
                );
            }
        }
    }

    /// The audit's surgical-event builder: `events` are (byte offset,
    /// token) pairs at strictly increasing byte offsets with 'a' fill
    /// between them and up to `total`, so a two-byte 'é' placed at
    /// offset k*4096 starts at EXACTLY that byte — the placement
    /// precision the certificate-boundary shapes exist for (an é one
    /// byte off is a different case, not this one).
    fn surgical_text(events: &[(usize, &str)], total: usize) -> String {
        let mut text = String::with_capacity(total);
        let mut prev = 0usize;
        for &(off, token) in events {
            text.push_str(&"a".repeat(off - prev));
            text.push_str(token);
            prev = off + token.len();
        }
        text.push_str(&"a".repeat(total - prev));
        text
    }

    #[test]
    fn both_scanners_match_the_references_on_every_cr_lf_string_up_to_16_codepoints() {
        // The audit's A-exh-CR-LF shape: EVERY string over the two break
        // bytes through length 16 — every CRLF-pairing case (\r\n vs
        // \n\r vs \r\r), every 2+-unit run-qualification case, leading
        // and trailing runs, runs ending at end of text — so any
        // unit-counting mutation with a witness of 16 codepoints or
        // fewer is caught by CONSTRUCTION, not by the luck of a
        // 300-case soup draw sampling the same space. The count is
        // asserted so a regressed alphabet or length budget fails the
        // sweep itself. Windowing rides every 97th case: the walk over
        // the oracle bounds is scanner-orthogonal, so a bounded sample
        // buys the windowing differential without paying 131k × 5
        // windower calls.
        let mut cases = 0usize;
        for_each_string_over(&['\r', '\n'], 16, |text| {
            cases += 1;
            assert_both_scanners_match_the_references(text);
            if cases.is_multiple_of(97) {
                assert_windowing_matches_over_oracle_bounds(text);
            }
        });
        assert_eq!(cases, 131_071, "the sweep's own size regressed");
    }

    #[test]
    fn both_scanners_match_the_references_on_every_three_symbol_string_up_to_9_codepoints() {
        // The audit's B-exh-3 shape: every string over {\r, \n, x}
        // through length 9 (29,524 cases) — the CR/LF alphabet with one
        // CONTENT byte, so break runs and content segments interleave at
        // every adjacency, including the content-bearing single-unit
        // runs the two-symbol alphabet cannot express (a lone \n between
        // x's is a kept LINE and ordinary paragraph content at once).
        let mut cases = 0usize;
        for_each_string_over(&['\r', '\n', 'x'], 9, |text| {
            cases += 1;
            assert_both_scanners_match_the_references(text);
            if cases.is_multiple_of(97) {
                assert_windowing_matches_over_oracle_bounds(text);
            }
        });
        assert_eq!(cases, 29_524, "the sweep's own size regressed");
    }

    #[test]
    fn both_scanners_match_the_references_on_every_four_symbol_string_up_to_7_codepoints() {
        // The audit's C-exh-4 shape: every string over {\r, \n, x, VT}
        // through length 7 (21,845 cases) — VT joins the alphabet
        // because it is the one ASCII_WS member whose omission from the
        // fold table is the corpus's own cited mutation pin, and here it
        // rides at ARBITRARY adjacency to the break bytes and the
        // content byte, not just the hand-written "a\n\x0b\nb" shape:
        // blank-looking VT lines next to real lines, VT inside break
        // runs, VT as a whole blank paragraph.
        let mut cases = 0usize;
        for_each_string_over(&['\r', '\n', 'x', '\u{0B}'], 7, |text| {
            cases += 1;
            assert_both_scanners_match_the_references(text);
            if cases.is_multiple_of(97) {
                assert_windowing_matches_over_oracle_bounds(text);
            }
        });
        assert_eq!(cases, 21_845, "the sweep's own size regressed");
    }

    #[test]
    fn both_scanners_match_the_references_on_two_five_symbol_alphabets_up_to_6_codepoints() {
        // The audit's C-exh-4 extension the runtime budget allows: the
        // four-symbol alphabet grown to five twice — once with FF (the
        // OTHER ASCII_WS member below 0x20 the table must carry) and
        // once with ' ' (the member at 0x20, so the fold answers from a
        // different table cell than the 0x09..=0x0D run) — at length 6,
        // 19,531 strings each. The five-symbol product grows too fast to
        // carry the full 7-length budget of the smaller alphabets; 6
        // keeps the whole exhaustive block inside its measured seconds.
        for alphabet in [
            &['\r', '\n', 'x', '\u{0B}', '\u{0C}'][..],
            &['\r', '\n', 'x', '\u{0B}', ' '][..],
        ] {
            let mut cases = 0usize;
            for_each_string_over(alphabet, 6, |text| {
                cases += 1;
                assert_both_scanners_match_the_references(text);
                if cases.is_multiple_of(97) {
                    assert_windowing_matches_over_oracle_bounds(text);
                }
            });
            assert_eq!(cases, 19_531, "the sweep's own size regressed");
        }
    }

    #[test]
    fn both_scanners_match_the_references_on_certificate_boundary_straddles() {
        // The AsciiCert arithmetic the sub-4096-byte corpus cannot see:
        // a non-ASCII byte at k*4096 ± 2 straddles the exact stride a
        // certify batch ends on, so the batch fails, the locating scan
        // runs, and the reset-to-past-the-bad-byte (or the p >= b
        // early-true) branch fires at every alignment a 4 KiB stride
        // has — the audit's X-structured sweep, scaled to several k: k=1
        // and k=2 are the first and second batch boundaries, k=31 puts
        // the straddle ~128 KiB deep, past 31 batch boundaries the
        // sliding range crossed by extension and reset in turn. Breaks
        // sit ~104 bytes past the NEXT batch boundary so the
        // post-straddle segment is itself a second certificate case,
        // and the trailing "\n\n" gives the paragraph scanner a real
        // second paragraph; é (2 bytes), an astral emoji (4 bytes) and
        // CJK (3 bytes) stress the byte-vs-codepoint bookkeeping at
        // three different token widths.
        for k in [1usize, 2, 3, 31] {
            for ch in ["\u{00E9}", "\u{1F600}", "\u{4E2D}"] {
                for delta in [-2isize, -1, 0, 1, 2] {
                    let off = (k * CERT_CHUNK).saturating_add_signed(delta);
                    let brk = (k + 1) * CERT_CHUNK + 104;
                    let mut text = surgical_text(&[(off, ch)], brk);
                    text.push('\n');
                    text.push_str(&"b".repeat(100));
                    text.push_str("\n\n");
                    text.push_str(&"c".repeat(50));
                    assert_both_scanners_match_the_references(&text);
                }
            }
        }
    }

    #[test]
    fn both_scanners_match_the_references_on_certificate_reset_and_alternation_surgicals() {
        // The certificate's reset/restart and alternation regimes at
        // multi-KiB scale, the audit's S4/S5/S6/S7/S8 shapes:
        // (1) the p >= b extend path — a long pure-ASCII segment
        // certifies PAST the break, so the very next segment starts
        // inside the certified range and its é must fail `covers` and
        // re-batch from a mid-range start (pre swept across the
        // 4095/4096/4097 and 8191/8192 batch-boundary values);
        // (2) reset-then-long-run-then-restart, repeated: é, 4100 a's,
        // é, 8192 a's, a break, then two more é's 4104 bytes apart —
        // each rep forces a failed batch, a reset, and a restart
        // against a range the previous rep left behind;
        // (3) alternation at the period the audit found interesting:
        // é every 4104 bytes (4096+8, one batch plus the window) and
        // every 4097, sustained for 32 periods (~131 KiB) so the
        // certificate fails, resets, and re-derives continuously;
        // (4) alternation WITH breaks — a 4100-byte segment, é, 3 a's,
        // a break, 30 times — the reset/restart interleaved with the
        // window and hop paths; and the extend-heavy twin (8200 a's,
        // break, é, 10 a's, break) where the certificate SUCCEEDS on
        // the long runs between hits.
        for pre in [4095usize, 4096, 4097, 8191, 8192] {
            let text = format!(
                "{}\n\u{00E9}{}\n\n{}",
                "a".repeat(pre),
                "b".repeat(4100),
                "c".repeat(9)
            );
            assert_both_scanners_match_the_references(&text);
        }
        let reset_runs = format!(
            "{}{}{}{}\n{}{}{}\n\n{}",
            "\u{00E9}",
            "a".repeat(4100),
            "\u{00E9}",
            "a".repeat(8192),
            "x".repeat(4104),
            "\u{00E9}",
            "x".repeat(4104),
            "y".repeat(7)
        )
        .repeat(6);
        assert_both_scanners_match_the_references(&reset_runs);
        let alt_4104 = format!("{}\u{00E9}", "a".repeat(4102)).repeat(32);
        assert_both_scanners_match_the_references(&alt_4104);
        let alt_4097 = format!("{}\u{00E9}", "a".repeat(4095)).repeat(32);
        assert_both_scanners_match_the_references(&alt_4097);
        let alt_with_breaks =
            format!("{}\u{00E9}{}\n", "a".repeat(4100), "a".repeat(3)).repeat(30) + "end";
        assert_both_scanners_match_the_references(&alt_with_breaks);
        let alt_extend = format!("{}\n\u{00E9}{}\n", "a".repeat(8200), "a".repeat(10)).repeat(15);
        assert_both_scanners_match_the_references(&alt_extend);
    }

    #[test]
    fn both_scanners_match_the_references_on_batch_multiple_segments_and_crlf_straddles() {
        // Segments whose lengths are EXACTLY the batch arithmetic's
        // interesting values — 4095/4096/4097 (one stride minus, on,
        // plus) and 8191/8192/8193 (two strides minus, on, plus) — under
        // all three break-unit spellings (\n lines, \n\n paragraphs,
        // \r\n\r\n CRLF-pair paragraphs), so a certify query's end lands
        // on every side of a batch boundary the stride math has; plus
        // the é-after-exact-length shape (the certificate succeeds on
        // the whole segment, then the very next byte is non-ASCII). The
        // CRLF straddles put a \r at byte 4095 with its \n at 4096 (a
        // PAIR torn across the batch boundary in byte terms, one unit
        // in codepoint terms) and the same at 8191/8192, repeated to
        // ~128 KiB so the sliding range crosses dozens of straddles.
        for seglen in [4095usize, 4096, 4097, 8191, 8192, 8193] {
            let lines = format!("{}\n", "a".repeat(seglen)).repeat(4) + "end";
            assert_both_scanners_match_the_references(&lines);
            let paras = format!("{}\n\n", "a".repeat(seglen)).repeat(4) + "end";
            assert_both_scanners_match_the_references(&paras);
            let crlf_paras = format!("{}\r\n\r\n", "a".repeat(seglen)).repeat(3) + "end";
            assert_both_scanners_match_the_references(&crlf_paras);
            let after = format!(
                "{}\u{00E9}{}\n{}",
                "a".repeat(seglen),
                "a".repeat(10),
                "b".repeat(9)
            );
            assert_both_scanners_match_the_references(&after);
        }
        let straddle_4096 = format!(
            "a{}\r\nb{}\r\nc{}\nd{}",
            "a".repeat(4094),
            "a".repeat(4095),
            "a".repeat(4096),
            "3"
        )
        .repeat(8);
        assert_both_scanners_match_the_references(&straddle_4096);
        let straddle_8192 = format!(
            "a{}\r\nb{}\n\nc{}",
            "a".repeat(8190),
            "b".repeat(10),
            "5".repeat(5)
        );
        assert_both_scanners_match_the_references(&straddle_8192);
        let crlf_para_straddle = format!(
            "a{}\r\n\r\nb{}\r\rc{}",
            "a".repeat(4094),
            "b".repeat(99),
            "7".repeat(6)
        );
        assert_both_scanners_match_the_references(&crlf_para_straddle);
    }

    #[test]
    fn both_scanners_match_the_references_on_window_edge_and_post_break_non_ascii() {
        // The BREAK_WINDOW edge from the non-ASCII side: é inside the
        // first 8 window bytes (offsets 0..=9, both sides of the
        // window's edge) with the break ~4200 bytes beyond, so the hot
        // fold raises `saw_non_ascii` and the cold path's certificate
        // must then fail the very range the window already folded — the
        // audit's third X-structured sweep. Then non-ASCII immediately
        // AFTER a break run at cert scale (the first byte of the
        // post-run segment is the bad byte, the certificate starts its
        // very first batch on it), the short-segment-after-p shape (a
        // 10-byte segment squeezed between two 5000-byte ones), and the
        // cover-p/cover-p-para shapes: a query whose segment ends
        // exactly at a known-bad byte (line) and its paragraph twin
        // (the audit's S14/S14b/S14c/S14d).
        for off in 0..=9usize {
            let text = format!(
                "{}\u{00E9}{}\n{}",
                "a".repeat(off),
                "a".repeat(4200),
                "b".repeat(9)
            );
            assert_both_scanners_match_the_references(&text);
        }
        let after_break_run =
            format!("{}\n\n\u{00E9}{}\n\n", "a".repeat(4096), "a".repeat(4096)).repeat(15) + "zz";
        assert_both_scanners_match_the_references(&after_break_run);
        let short_mid = format!(
            "a{}\n\u{00E9}b{}\nc{}\nd{}",
            "a".repeat(4999),
            "b".repeat(9),
            "c".repeat(4999),
            "d".repeat(9)
        );
        assert_both_scanners_match_the_references(&short_mid);
        let cover_p = format!(
            "a{}\nb{}\n\u{00E9}c{}",
            "a".repeat(4999),
            "b".repeat(99),
            "c".repeat(9)
        );
        assert_both_scanners_match_the_references(&cover_p);
        let cover_p_para = format!(
            "a{}\n\nb{}\n\n\u{00E9}c{}",
            "a".repeat(4999),
            "b".repeat(99),
            "c".repeat(9)
        );
        assert_both_scanners_match_the_references(&cover_p_para);
    }

    #[test]
    fn both_scanners_survive_an_eight_mib_single_line_with_surgical_non_ascii_bytes() {
        // The ONE multi-MiB case the audit budget allows (the probe's
        // S11 20-MiB family cut to 8 MiB, the mission budget's floor:
        // measured to keep this test under ~1 s in a debug build): a
        // single unbroken line — no break byte anywhere, so the line
        // scanner pays one window, one memchr2 hop over 8 MiB, and
        // 2048 certify batches; the paragraph scanner the same — with
        // three surgical é's at 2 MiB+4095, 4 MiB+1 and 6 MiB+4097:
        // two batch boundaries the sliding range must RESET on, with
        // ~2 MiB (512 batches) of pure-ASCII extension between them.
        // The codepoint total is 8 MiB - 3 (three two-byte é's), so any
        // byte-vs-codepoint drift anywhere in the walk lands in the
        // final (0, total) span both scanners must emit.
        let mib = 1024 * 1024;
        let e1 = 2 * mib + 4095;
        let e2 = 4 * mib + 1;
        let e3 = 6 * mib + 4097;
        let text = surgical_text(
            &[(e1, "\u{00E9}"), (e2, "\u{00E9}"), (e3, "\u{00E9}")],
            8 * mib,
        );
        assert_both_scanners_match_the_references(&text);
    }

    #[test]
    fn both_scanners_match_the_references_across_break_phase_fill_and_unit_density() {
        // The window-phase/density sweep: the first break at byte
        // `phase` from the segment start for every phase 0..=24 — the
        // density-guard window is 8 bytes, so phases 7/8/9 are the
        // phases where the break crosses the window edge and the scan
        // flips between the inline fold and the memchr2 hop, and the
        // odd/even phases around them pin the edge under CRLF pairs
        // (\r the last byte inside the window, \n the first beyond it)
        // — × the ASCII_WS fills as CONTENT ('x' the control, ' ',
        // '\t', VT, FF) × the unit shapes (\n, \r, \r\n, and mixed
        // runs of 2-5 units: the pairing and the 2+-unit paragraph
        // threshold at every adjacency) × a short within-window tail
        // and a long beyond-window tail with a second break. Then the
        // density shape itself: (fill*k + unit)*6 — segments of
        // exactly k fill bytes for k 1..=12, the densities at and
        // around the window, six in a row.
        let fills = ['x', ' ', '\t', '\u{0B}', '\u{0C}'];
        let units = [
            "\n", "\r", "\r\n", "\n\n", "\r\r", "\n\r", "\r\n\r\n", "\n\r\n", "\r\n\n", "\r\r\n",
            "\n\n\n", "\r\n\r", "\n\n\r\r",
        ];
        let mut cases = 0usize;
        for phase in 0..=24usize {
            for fill in fills {
                for unit in units {
                    for tail_len in [1usize, 40] {
                        let mut text: String = std::iter::repeat_n(fill, phase).collect();
                        text.push_str(unit);
                        text.push_str(&"b".repeat(tail_len));
                        if tail_len == 40 {
                            text.push_str("\nccc");
                        }
                        assert_both_scanners_match_the_references(&text);
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 3_250, "the sweep's own size regressed");
        for k in 1..=12usize {
            for fill in fills {
                for unit in ["\n", "\r\n", "\n\r\n", "\r\n\r\n", "\r\r", "\n\n"] {
                    let unit_text: String = std::iter::repeat_n(fill, k).collect::<String>() + unit;
                    let text = unit_text.repeat(6) + "b";
                    assert_both_scanners_match_the_references(&text);
                }
            }
        }
    }

    #[test]
    fn line_bounds_classifies_the_full_white_space_zoo_and_its_content_lookalikes() {
        // The systematic White_Space zoo, the audit's WS-ZOO category:
        // each of the 25 Unicode White_Space codepoints as a line's
        // SOLE content is a BLANK line (dropped by the real-line
        // filter — the ASCII_WS table answers below 0x7F, the
        // char::is_whitespace fallback above it), while U+001C..U+001F
        // (FS/GS/RS/US — Python's str.isspace/splitlines treat them as
        // whitespace/breaks; the Unicode White_Space property does not),
        // ZWSP U+200B, U+180E (White_Space only up to Unicode 6.3) and
        // the BOM U+FEFF are CONTENT, each its own counted line. The
        // corpus pins VT/FF/NBSP at three hand-written shapes; this
        // sweep is the systematic superset — every member once, in the
        // sole-content position AND as the blank line between two real
        // lines, so a table or fallback mutation anywhere in the
        // 25-member set lands here. The paragraph scanner has no
        // content filter, so every zoo member is ordinary paragraph
        // content: [(0, 2)] throughout, stated alongside so the
        // no-filter contract is pinned by the same sweep. Expectations
        // are STATED, not derived from char::is_whitespace (the
        // references share production's predicate and would happily
        // agree with a mutated one); the reference agreement asserted
        // after pins the non-ASCII members' fallback-path machinery,
        // which the stated answers alone do not reach.
        let white_space = [
            '\u{0009}', '\u{000A}', '\u{000B}', '\u{000C}', '\u{000D}', '\u{0020}', '\u{0085}',
            '\u{00A0}', '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}',
            '\u{2005}', '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200A}', '\u{2028}',
            '\u{2029}', '\u{202F}', '\u{205F}', '\u{3000}',
        ];
        assert_eq!(white_space.len(), 25, "the White_Space set itself drifted");
        // The two BREAK bytes are White_Space members too, but they are
        // the delimiters, not "a line's sole content": '{c}\n' with
        // c='\n' is a 2-unit break run (a paragraph SPLIT, the shape the
        // 131k exhaustive sweep pins at every length), not a blank
        // paragraph. The zoo walks the 23 non-break members; the break
        // pair's own arithmetic is the whole point of the CR/LF
        // exhaustive sweep above.
        let blank_content: Vec<char> = white_space
            .iter()
            .copied()
            .filter(|&c| !matches!(c, '\n' | '\r'))
            .collect();
        assert_eq!(blank_content.len(), 23);
        for c in blank_content {
            let text = format!("{c}\n");
            assert_eq!(
                line_bounds(&text),
                Vec::<(usize, usize)>::new(),
                "White_Space {c:?} as sole line content must be blank"
            );
            assert_eq!(
                paragraph_bounds(&text),
                vec![(0, 2)],
                "paragraph_bounds has no content filter: {c:?}"
            );
            let between = format!("a\n{c}\nb");
            assert_eq!(
                line_bounds(&between),
                vec![(0, 1), (4, 5)],
                "a blank {c:?} line between real lines must vanish"
            );
            assert_eq!(
                paragraph_bounds(&between),
                vec![(0, 5)],
                "single lone newlines do not split paragraphs: {c:?}"
            );
            assert_both_scanners_match_the_references(&text);
            assert_both_scanners_match_the_references(&between);
        }
        for c in [
            '\u{001C}', '\u{001D}', '\u{001E}', '\u{001F}', '\u{200B}', '\u{180E}', '\u{FEFF}',
        ] {
            let text = format!("{c}\n");
            assert_eq!(
                line_bounds(&text),
                vec![(0, 1)],
                "content lookalike {c:?} must be a counted line"
            );
            assert_eq!(paragraph_bounds(&text), vec![(0, 2)], "{c:?}");
            assert_both_scanners_match_the_references(&text);
        }
    }
}
