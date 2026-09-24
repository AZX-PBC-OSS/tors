//! Snippet-provenance grounding: the pure-Rust core of `tors.highlight`.
//!
//! Answers a different question from [`super::grounded_impl`]: not "is this
//! claim grounded?" but "WHERE in this chunk does the query's evidence sit?" —
//! the span-level provenance a search UI highlights and a citation deep-links
//! to. Given a query and a chunk of text, it returns the best-matching,
//! non-overlapping snippets as CHARACTER offsets into the original text plus
//! a per-snippet overlap score, so a consumer can slice the original `str`
//! (`text[start:end] == snippet["text"]`, pinned by tests on both sides of
//! the FFI) without re-deriving any alignment in Python.
//!
//! # Metric choice, from the literature
//!
//! **ROUGE-W F1, not BLEU-n** (Lin 2004, "ROUGE: A Package for Automatic
//! Evaluation of Summaries", defines ROUGE-L over the longest common
//! subsequence and ROUGE-W over a length-weighted LCS; Papineni et al. 2002,
//! "BLEU: a Method for Automatic Evaluation of Machine Translation", defines
//! modified n-gram precision with a brevity penalty). The two papers' own
//! properties decide the roles here:
//!
//! - BLEU is PRECISION-oriented: it scores a candidate translation against a
//!   fixed reference and penalizes short candidates with an explicit brevity
//!   penalty that needs a reference length. A snippet selector has no
//!   reference — and a precision-only objective against the query terms is
//!   gamed by a one-word window (one query term, perfect precision), exactly
//!   the degenerate highlight a citation consumer cannot use. BLEU-4's
//!   contiguous 4-gram matching is also unmeasurable against a 10-60-token
//!   query: the modified n-gram statistics are too sparse to rank spans
//!   differently. BLEU's legitimate role — "is this span about the query
//!   terms" — is carried here instead by the anchor pre-filter (below),
//!   which admits only spans that actually contain query terms.
//! - ROUGE-L is RECALL-oriented (the LCS counts how much of the query's
//!   token sequence the candidate covers, in order): "does this span cover
//!   the query's content", which is the provenance claim. As the reference
//!   ROUGE toolkit and the RAG snippet pickers that followed it do, raw LCS
//!   recall is combined with an LCS precision into an F1 so a span the size
//!   of the whole text cannot win on recall alone.
//! - Within ROUGE, **ROUGE-W rather than plain ROUGE-L**: Lin's §3.2 shapes
//!   the LCS credit through `f(k) = k^1.2`, so a run of CONTIGUOUS matched
//!   tokens scores superlinearly more than the same matches spread through
//!   filler. For snippets that is the right bias, and it usually is: two
//!   spans containing the same query terms rank better when the terms occur
//!   as a contiguous run ("embedding model" over "embedding ... model"),
//!   which is what a highlight and a citation read best. Plain LCS is
//!   indifferent to that difference; the anchored construction below makes
//!   matched material locally dense but cannot make the query's own terms
//!   contiguous inside the candidate, so the shaping still decides real
//!   rankings. F combines the shaped recall and precision (`2RP/(R+P)`,
//!   Lin's balance parameter at 1), with the paper's Equation 15
//!   normalization — `R = f^-1(WLCS/f(m))`, the shaping function's
//!   INVERSE — keeping every score in `[0, 1]` (applying f rather than
//!   f^-1 over-scores long full runs past 1.0; the fuzz target caught
//!   that before this shipped).
//!
//! # Algorithm and its memory shape
//!
//! 1. Tokenize the query into normalized terms (see [`tokens`]) — the token
//!    sequence `Q`, capped at [`MAX_QUERY_TOKENS`].
//! 2. Tokenize the text the same way, keeping each token's character AND
//!    byte span, capped at the first [`MAX_TEXT_TOKENS`] tokens (the
//!    pathological-chunk cap: tokens past it are simply not evidence for
//!    this call — the same bounded-scan discipline `grounded_impl`'s
//!    windowing applies).
//! 3. **Anchor runs**: the text tokens whose normalized form equals some
//!    query term (a hash set — O(1) membership) are matched tokens; maximal
//!    runs of consecutive matched tokens are the evidence anchors. Runs are
//!    merged while the merged CHARACTER span still fits `max_chars` (a
//!    snippet is by definition a span that fits the budget, so this merge
//!    rule IS the budget's semantics, not a separate knob).
//! 4. **Candidate expansion and the cap**: each run expands to its
//!    UAX #29 sentence span when that sentence fits `max_chars` — sentence
//!    bounds are the citation unit the attribution literature converged on
//!    (ALCE's own snippet-mode baselines feed "the first two sentences" of a
//!    passage and measure near-ceiling citation quality on them: Gao et al.
//!    2023, "Enabling Large Language Models to Generate Text with
//!    Citations"; a snippet ending mid-sentence reads as truncated, one
//!    covering the sentence reads as quoted). A sentence larger than the
//!    budget is skipped (the run itself is the candidate), and every
//!    candidate is then clamped to `max_chars` at token boundaries. At most
//!    [`MAX_CANDIDATES`] candidates, pre-ranked by matched-token density
//!    (matched count desc, position asc — a cheap O(1)-per-run key), reach
//!    the DP; the rest are discarded, which bounds the total DP work
//!    independently of how match-dense an adversarial text is.
//! 5. **Scoring**: each candidate's token sequence against `Q` through the
//!    ROUGE-W F1 DP — two score rows and one run-length row, each
//!    `|C|+1` wide, allocated ONCE and reused across candidates. The DP is
//!    the classic O(|Q|·|C|) recurrence; `|C|` is bounded by the tokens a
//!    `max_chars` span can hold, so no allocation is ever proportional to
//!    n·m (the full-matrix LCS this module must not be — see below).
//! 6. **Selection**: candidates sort by `(score desc, start asc)`, a greedy
//!    pass accepts only non-overlapping CHARACTER spans up to
//!    `max_snippets` (multiple snippets are the attribution-report shape —
//!    RARR's per-passage evidence list, Gao et al. 2022, "RARR: Researching
//!    and Revising What Language Models Say, Using Language Models":
//!    different regions of a chunk support different parts of an answer),
//!    and the accepted set is returned in position order. The overall
//!    `score` is the best accepted snippet's F1, `0.0` when there are none.
//!
//! Why the LCS machinery is *not* Hirschberg's (Hirschberg 1975, "A linear
//! space algorithm for computing maximal common subsequences") nor
//! Hunt–Szymanski's (Hunt & Szymanski 1977, "A fast algorithm for computing
//! longest common subsequences"): Hirschberg's divide-and-conquer exists to
//! RECOVER the subsequence itself in O(min(n,m)) space, at roughly twice
//! the fill's time. This module never needs the subsequence — spans are
//! already anchored and the DP produces a scalar score — so the two-row
//! fill delivers O(min) DP space without the recursion. Hunt–Szymanski
//! wins when matching PAIRS are sparse, O((r+n)·log n) in their count r;
//! step 3's anchor pre-filter is the same insight applied one level up
//! (only dense-match regions reach the DP at all), in the regime that
//! actually holds here: a short query (10-60 tokens) against a long chunk
//! (up to ~2k tokens), the asymmetric case Myers' O((N+M)·D) diff (Myers
//! 1986, "An O(ND) difference algorithm and its variations") also targets —
//! but a diff produces an edit script for aligning two whole texts, not a
//! score for ranking candidate spans, and the maintained diff engine
//! (`similar`) already backs `grounded_impl`'s verdict path where exactly
//! that is needed.
//!
//! # Why the ROUGE-W DP is not the `similar` crate (span-scoring library
//! check, 2026-09-24)
//!
//! `similar` is already a dependency (3.2, behind `grounded_impl`), so the
//! fair question is whether ROUGE-W SPAN SCORING could ride it too. It
//! cannot, for reasons specific to scoring rather than to the
//! alignment-recovery argument above. `similar` computes an EDIT SCRIPT — an
//! ordered sequence of Equal/Insert/Delete ops between two whole texts — and
//! what span scoring needs is not in one: (1) the weighted-LCS RUN
//! structure. ROUGE-W credits `f(k) - f(k-1)` per diagonal run of length k,
//! and the run decomposition of the WEIGHTED optimum is in general different
//! from every plain-LCS alignment `similar` can emit (the superlinear shaping
//! prefers fewer longer runs at equal matched counts, so the weighted LCS is
//! not a function of the plain LCS length) — recovering the run lengths from
//! an edit script of a different, unweighted optimization would mean re-doing
//! the DP to get numbers the DP already yields. (2) Lin's Equation 15
//! normalization, which needs the scalar WLCS alongside `f(|Q|)` and `f(|C|)`
//! to produce an F1 in `[0, 1]` — a summary no diff-script API exposes. (3)
//! The calling shape: up to [`MAX_CANDIDATES`] candidate spans scored against
//! ONE short query with a single reused two-row scratch
//! ([`Scratch`]) — a fill per candidate, not one fill per document pair.
//! The custom two-row fill stays.

//!
//! A candidate scoring 0 (no shared token) is never a snippet: no overlap,
//! no provenance claim (RARR credits a span with attribution only when the
//! evidence supports the claim; a zero-overlap span supports nothing). The
//! score is a RANKING signal, not an answerability verdict: Joren et al.
//! 2024, "Sufficient Context: A New Lens on Retrieval Augmented Generation
//! Systems", draw the line this module must not blur — whether a context
//! suffices to ANSWER is a semantic judgment no lexical overlap score can
//! make; consumers needing that call should run a model over the snippet.
//!
//! # The sentence batch: `ground_sentences` (per-sentence scores)
//!
//! [`ground_sentences`] is the batch bridge primitive a downstream NLI
//! verifier (MiniCheck/SummaC style) consumes: segment `text` with the SAME
//! UAX #29 sentence bounds `sentence_bounds` publishes ([`sentence_spans`],
//! shared with `highlight`'s expansion step, so the batch's offsets are
//! by construction those bounds), score EVERY sentence against `query`
//! with the same ROUGE-W F1, and return every sentence with its score in
//! position order. The citation unit is the sentence (the unit the
//! attribution literature converged on; ALCE: Gao et al. 2023, "Enabling
//! Large Language Models to Generate Text with Citations"; see the
//! candidate-expansion step below for how `highlight` already uses it),
//! and the per-sentence score is a RANKING signal, not an answerability
//! verdict (Joren et al. 2024, "Sufficient Context": sufficiency is a
//! semantic judgment a lexical overlap cannot make, the honest-limitations
//! line the highlight docs carry applies verbatim here; consumers needing
//! the verdict run their NLI model over the top-scored sentences). The
//! aggregate is the MAX per-sentence score, not the mean: it is the
//! retrieval signal the bridge needs ("does SOME sentence carry the
//! evidence"), it mirrors `highlight`'s best-snippet aggregate, and it is
//! stable under irrelevant additions, see [`ground_sentences`]'s docs for
//! the full justification.
//!
//! # Token normalization
//!
//! Tokens come from UAX #29 word boundaries (the `unicode-segmentation`
//! crate, the same segmentation `word_bounds`/`word_count` expose), NOT a
//! whitespace/regex split: accents stay attached through their combining
//! marks (NFD "cafe\u{301}" is one token folding to the same norm as NFC
//! "café"), ZWJ emoji sequences survive as single tokens, RTL text yields
//! the same logical-order tokens any scanner should. One refinement the
//! UAX #29 tables do not make themselves: every CJK character (Han,
//! Hiragana, Katakana, Hangul — see [`is_cjk`]) inside a word segment
//! becomes its OWN token. UAX #29 splits Han and Hiragana per character
//! but keeps Katakana and Hangul runs joined (verified against this
//! crate's own `word_bounds`: `日本語のテキスト` →
//! `日|本|語|の|テキスト`), and unspaced CJK morphemes are the standard
//! IR per-character fallback (Lin 2004's Chinese evaluations tokenize per
//! character for the same reason): without it a Katakana query term could
//! never partially match inside a longer Katakana run. The sub-split walks
//! grapheme clusters, never bytes or bare chars, so a CJK base with
//! combining marks stays one token. Matching is case-folded per character
//! (the full `char::to_lowercase` mapping; a few foldings change length,
//! which is harmless — the fold feeds matching only, offsets always come
//! from the original text).
//!
//! # Offsets
//!
//! Every offset is a CHARACTER index (Unicode scalar value / Python
//! codepoint index), counted while walking the text — never a byte offset.
//! That is the only coordinate Python `str` slicing accepts, and tors is
//! worked over the Python string's own encoding (pyo3's `&str` extraction
//! is the UTF-8 view of the same object, no lossy re-interpretation
//! anywhere on the path), so the round-trip `text[start:end] ==
//! snippet.text` holds for CJK, accents, emoji (including multi-codepoint
//! sequences — offsets land on token boundaries, and UAX #29 word
//! boundaries never split a grapheme cluster: WB4 removes Extend/ZWJ from
//! the break decision before any rule fires), and mixed scripts alike,
//! pinned in `tests/test_grounding.py` and, under raw adversarial bytes,
//! `fuzz_targets/grounding.rs`.

use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

/// The pathological-chunk cap: at most this many text tokens are scanned. A
/// 16k-token cap covers a ~60k-character chunk at English average word
/// length — past the primary consumer's own `whole_document_max_chars`
/// ceiling — and keeps the anchor walk's worst case linear in that bound.
pub const MAX_TEXT_TOKENS: usize = 16_384;

/// The query's own cap: a "query" longer than 128 tokens is a document, not
/// a query; its excess terms are ignored the way a search engine's term cap
/// ignores them.
pub const MAX_QUERY_TOKENS: usize = 128;

/// How many candidates (after the density pre-rank) may reach the ROUGE-W
/// DP. Bounds the total DP work independently of match density: 64
/// candidates x (|Q| <= 128) x (|C| bounded by the tokens `max_chars` can
/// hold) is a few million DP cells in the absolute worst case,
/// milliseconds; the realistic 3-snippet call touches a handful.
const MAX_CANDIDATES: usize = 64;

/// One best-matching snippet: `text[start:end]` of the original text
/// (CHARACTER offsets, codepoint indices) plus the candidate's ROUGE-W F1
/// against the query (`score`, in `[0.0, 1.0]`).
#[derive(Debug, Clone, PartialEq)]
pub struct Snippet {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub score: f64,
}

/// The whole result: up to `max_snippets` non-overlapping snippets in
/// position order, and the best snippet's score (`0.0` when empty).
#[derive(Debug, Clone, PartialEq)]
pub struct Grounding {
    pub snippets: Vec<Snippet>,
    pub score: f64,
}

/// Is this character a CJK ideograph/syllable (Hiragana, Katakana, Han,
/// Hangul, and the compatibility blocks)? CJK text carries little word
/// structure UAX #29 can use (see the module docs), so the tokenizer hands
/// each such character out as its own token.
fn is_cjk(c: char) -> bool {
    matches!(
        c,
        '\u{3040}'..='\u{30FF}'       // Hiragana + Katakana
            | '\u{3400}'..='\u{4DBF}' // CJK Extension A
            | '\u{4E00}'..='\u{9FFF}' // CJK Unified Ideographs
            | '\u{AC00}'..='\u{D7AF}' // Hangul syllables
            | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
    )
}

/// One text token: its CHARACTER span (`start`/`end`, codepoint indices into
/// the source) and BYTE span (`byte_start`/`byte_end`) — the char span is
/// what crosses the FFI, the byte span is what slices the `&str` for the
/// snippet text without any re-indexing. `norm` is the case-folded matching
/// form; the spans are the truth.
#[derive(Debug, Clone)]
struct Token {
    start: usize,
    end: usize,
    byte_start: usize,
    byte_end: usize,
    norm: String,
}

/// The token cap, applied while building: once full, the scan stops
/// emitting (the pathological-chunk discipline — text past the cap is not
/// evidence for this call).
fn push_token(out: &mut Vec<Token>, token: Token) {
    if out.len() < MAX_TEXT_TOKENS {
        out.push(token);
    }
}

/// A token's matching form: the segment's characters case-folded per
/// character (`char::to_lowercase`, the full Unicode mapping), then
/// canonicalized to NFC — `char::to_lowercase` alone does NOT make NFD
/// "cafe\u{301}" equal NFC "café" (it leaves the decomposed form
/// decomposed), and the normalization does, using the same
/// `unicode-normalization` tables the `nfc` surface exposes.
fn fold(segment: &str) -> String {
    let folded: String = segment.chars().flat_map(char::to_lowercase).collect();
    folded.nfc().collect()
}

/// Tokenize `text`: UAX #29 word segments (the `unicode-segmentation`
/// crate), each kept as one token unless it contains CJK — see the module
/// docs for the per-character CJK refinement and why it walks grapheme
/// clusters. Case-folded and NFC-canonicalized for matching; character and
/// byte spans recorded for slicing. Segments with no alphanumeric character
/// (whitespace runs, punctuation, standalone emoji) are dropped — they
/// cannot match a term, and the query side drops them identically; the same
/// rule applies to every run the CJK sub-split flushes, so CJK-range
/// punctuation (U+30FB, U+3099) stays token-free too. At most
/// [`MAX_TEXT_TOKENS`] tokens are emitted — the scan just stops there.
fn tokens(text: &str) -> Vec<Token> {
    let mut out: Vec<Token> = Vec::new();
    let mut cp = 0usize; // running codepoint offset across the whole text
    for (byte_start, segment) in text.split_word_bound_indices() {
        let seg_cps = segment.chars().count();
        if segment.chars().any(is_cjk) {
            // Sub-split at CJK: a new token starts at every grapheme
            // cluster whose first char is CJK; non-CJK clusters attach to
            // the current token (or start one when the segment opens with
            // them). Clusters, not chars, so combining marks stay put.
            let mut byte = byte_start;
            let mut tok_cp = cp;
            let mut tok_byte = byte_start;
            let mut norm = String::new();
            for cluster in segment.graphemes(true) {
                let cluster_cps = cluster.chars().count();
                let starts_cjk = cluster.chars().next().is_some_and(is_cjk);
                if starts_cjk && !norm.is_empty() {
                    // The SAME no-alphanumeric drop rule the non-CJK branch
                    // applies to whole segments, applied per flushed run:
                    // CJK-range punctuation (U+30FB katakana middle dot,
                    // U+3099) lands in this branch and must stay token-free
                    // like any other punctuation.
                    if norm.chars().any(char::is_alphanumeric) {
                        push_token(
                            &mut out,
                            Token {
                                start: tok_cp,
                                end: cp,
                                byte_start: tok_byte,
                                byte_end: byte,
                                norm: fold(&norm),
                            },
                        );
                    }
                    norm.clear();
                }
                if starts_cjk || norm.is_empty() {
                    tok_cp = cp;
                    tok_byte = byte;
                }
                norm.push_str(cluster);
                cp += cluster_cps;
                byte += cluster.len();
            }
            if !norm.is_empty() && norm.chars().any(char::is_alphanumeric) {
                push_token(
                    &mut out,
                    Token {
                        start: tok_cp,
                        end: cp,
                        byte_start: tok_byte,
                        byte_end: byte,
                        norm: fold(&norm),
                    },
                );
            }
        } else if segment.chars().any(char::is_alphanumeric) {
            push_token(
                &mut out,
                Token {
                    start: cp,
                    end: cp + seg_cps,
                    byte_start,
                    byte_end: byte_start + segment.len(),
                    norm: fold(segment),
                },
            );
            cp += seg_cps;
        } else {
            cp += seg_cps;
        }
    }
    out
}

/// Lin 2004's ROUGE-W shaping function, `f(k) = k^1.2` (the paper's
/// recommended exponent): a contiguous matched run of length k credits
/// superlinearly, so contiguity beats spread at equal matched counts.
fn shape(k: f64) -> f64 {
    k.powf(1.2)
}

/// The shaping function's inverse (Lin 2004 uses `f^-1` in Equation 15's
/// normalization, so the recall/precision ratios stay in `[0, 1]`).
fn shape_inv(x: f64) -> f64 {
    x.powf(1.0 / 1.2)
}

/// ROUGE-W F1 between the query's normalized terms `q` and one candidate's
/// normalized terms `c`: the weighted-LCS fill (two score rows + one
/// run-length row, each `c.len() + 1` wide, over `q`'s rows — the caller's
/// reusable scratch, so repeated calls never reallocate), then Lin's
/// recall/precision shaping and their F1 (`2RP/(R+P)`, β = 1).
///
/// Recurrence (Lin 2004 §3.2's WLCS fill, in its max-on-match spelling):
/// a match extends the current contiguous run `l`, crediting
/// `f(l) - f(l-1)`, but only when the extension actually beats the skip
/// options, `max(cell above, cell to the left)`; a non-match takes the
/// better of those two with the run reset. The max is not decoration:
/// Lin's Figure 3 spells the forced-diagonal variant (a match cell ALWAYS
/// extends the run), and that spelling is not monotone in the candidate:
/// a forced diagonal can strand accumulated credit when a later token
/// re-matches (the coverage fuzz target caught a score DROPPING when the
/// text gained source material: `[o, o]` scored against `[o, o, uua, o]`
/// read 2.0 off the last cell where the optimal alignment held 2.297).
/// A scoring function offered MORE evidence cannot report LESS: the
/// max-on-match spelling restores that (it computes the max-on-match
/// recurrence, a greedy-run-weighted alignment score, NOT the literal
/// weighted-LCS optimum over alignments: the two-row DP cannot represent
/// Pareto (value, trailing-run) states, so a brute-force oracle beats it on
/// rare inputs, a documented, deliberate deviation, pinned in the test
/// suite), and it reduces to Figure 3's value on every
/// contiguous-run shape, which is what the shaping rewards. Only the
/// score matters (never the alignment; see the module docs), so two
/// rows suffice; no allocation is proportional to `q.len() * c.len()`.
fn rouge_w_f1(q: &[String], c: &[&str], scratch: &mut Scratch) -> f64 {
    if q.is_empty() || c.is_empty() {
        return 0.0;
    }
    let wlcs = rouge_w_wlcs(q, c, scratch);
    if wlcs <= 0.0 {
        return 0.0;
    }
    // Lin 2004's Equation 15: the weighted score normalizes through the
    // shaping function's INVERSE, R = f^-1(WLCS / f(m)), not through f
    // itself. f^-1 is what keeps the ratio in [0, 1] (WLCS <= f(m) by
    // superadditivity, so WLCS/f(m) <= 1 and f^-1 is monotone); applying f
    // instead over-scores long matches past 1.0 (the fuzz target caught
    // exactly that: a full-run candidate scored 1.18).
    let r = shape_inv(wlcs / shape(q.len() as f64));
    let p = shape_inv(wlcs / shape(c.len() as f64));
    // Defensive clamp, the same shape `grounded_impl`'s verdict path ends
    // with: the F1 of two factors in [0, 1] is in [0, 1] mathematically,
    // but the DP's increment accumulation can round a full-match score to
    // 1 + 2e-16 (the fuzz target caught exactly that under the
    // max-on-match spelling), and the caller-facing invariant, every
    // score in [0.0, 1.0], must be airtight.
    (2.0 * r * p / (r + p)).clamp(0.0, 1.0)
}

/// Equation 15's recall normalization alone, `f^-1(WLCS / f(|q|))`, from a
/// caller-computed WLCS, `grounded_impl::grounding_coverage`'s denominator
/// step (it drives [`rouge_w_wlcs`] over interned id streams and normalizes
/// through here, so the shaping constants live in exactly one place).
pub(crate) fn rouge_w_recall_from_wlcs(q_len: usize, wlcs: f64) -> f64 {
    if q_len == 0 || wlcs <= 0.0 {
        return 0.0;
    }
    shape_inv(wlcs / shape(q_len as f64))
}

/// The weighted-LCS fill over interned token-id streams (u32 machine-word
/// compares): `grounded_impl::grounding_coverage`'s engine: it needs the
/// raw WLCS (to normalize by the SOURCE's own length, not the id call's
/// first argument), not the pre-shaped recall.
pub(crate) fn rouge_w_wlcs_ids(q: &[u32], c: &[u32], scratch: &mut Scratch) -> f64 {
    rouge_w_wlcs(q, c, scratch)
}

/// The weighted-LCS fill itself (the recurrence above, Lin 2004 §3.2's
/// WLCS fill in the max-on-match spelling, see [`rouge_w_f1`]'s docs for
/// why not Figure 3's forced diagonal), shared by the F1 and recall
/// spellings: two score rows + one run-length row, each
/// `c.len() + 1` wide, over `q`'s rows, the caller's reusable scratch, so
/// repeated calls never reallocate. Generic over the token pair because
/// two callers score with it: the in-module spellings pass the query's
/// normalized terms (`String`) against a candidate's (`&str`), and
/// `grounded_impl::grounding_coverage` passes two interned id streams
/// (`u32`; machine-word compares, the per-cell cost that matters at a
/// two-document DP's ~10^8 cells). The recurrence only ever compares a
/// row token to a width token, so `X: PartialEq<Y>` is the whole story.
fn rouge_w_wlcs<X, Y>(q: &[X], c: &[Y], scratch: &mut Scratch) -> f64
where
    X: PartialEq<Y>,
{
    let width = c.len() + 1;
    scratch.score_a.clear();
    scratch.score_a.resize(width, 0.0);
    scratch.score_b.clear();
    scratch.score_b.resize(width, 0.0);
    scratch.run_a.clear();
    scratch.run_a.resize(width, 0);
    scratch.run_b.clear();
    scratch.run_b.resize(width, 0);
    for x in q {
        // Row i-1 in score_a/run_a, row i accumulates into score_b/run_b.
        for (j, y) in c.iter().enumerate() {
            let (up, left) = (scratch.score_a[j + 1], scratch.score_b[j]);
            let val = if x == y {
                let run = scratch.run_a[j] + 1; // diagonal run length + 1
                let ext = scratch.score_a[j] + shape(run as f64) - shape(run as f64 - 1.0);
                if up > ext || left > ext {
                    // The skip options beat the run extension: take them
                    // and reset the run (the max-on-match spelling; see
                    // `rouge_w_f1`'s docs for why the forced diagonal is
                    // not an option).
                    scratch.run_b[j + 1] = 0;
                    up.max(left)
                } else {
                    scratch.run_b[j + 1] = run;
                    ext
                }
            } else {
                scratch.run_b[j + 1] = 0;
                up.max(left)
            };
            scratch.score_b[j + 1] = val;
        }
        std::mem::swap(&mut scratch.score_a, &mut scratch.score_b);
        std::mem::swap(&mut scratch.run_a, &mut scratch.run_b);
    }
    scratch.score_a[c.len()]
}

/// The reused DP scratch (see [`rouge_w_f1`]): allocated once per
/// `highlight` call, reused across every candidate. `pub(crate)` so
/// `grounded_impl`'s coverage core can drive the same fill with its own
/// scratch, the fields stay private to this module, the fill is the only
/// thing that touches them.
#[derive(Default)]
pub(crate) struct Scratch {
    score_a: Vec<f64>,
    score_b: Vec<f64>,
    run_a: Vec<usize>,
    run_b: Vec<usize>,
}

/// One candidate span: the token range `[tok_s, tok_e)` it scores over,
/// and the span it PRESENTS (the sentence's when sentence-expanded, the
/// token span otherwise) — `start`/`end` in CHARACTER offsets,
/// `byte_start`/`byte_end` the matching byte span.
struct Candidate {
    tok_s: usize,
    tok_e: usize,
    start: usize,
    end: usize,
    byte_start: usize,
    byte_end: usize,
}

/// Index of the last token whose `end` <= `cp`, plus one (the first token
/// overlapping or after `cp`); `toks` is sorted by position, so binary
/// search is sound.
fn token_at_or_after(toks: &[Token], cp: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = toks.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if toks[mid].end <= cp {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Index of the first token whose `start` is strictly greater than `cp`
/// (equivalently: the count of tokens with `start <= cp` — the sentence's
/// last token is the one before it, so as an EXCLUSIVE range end this is
/// the sentence's token span); `toks` sorted by position.
fn token_strictly_after(toks: &[Token], cp: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = toks.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if toks[mid].start <= cp {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The text's UAX #29 sentence spans: `(cp_start, cp_end, byte_start,
/// byte_end)` per sentence, in position order, the char spans are what
/// cross the FFI (Python codepoint indices), the byte spans what slice the
/// `&str` for the sentence's text without re-indexing. The same
/// segmentation `sentence_bounds` exposes (and `highlight`'s
/// sentence-expansion step uses), built once and shared by both surfaces
/// so the batch API's per-sentence offsets are BY CONSTRUCTION the bounds
/// `sentence_bounds(text)` returns.
fn sentence_spans(text: &str) -> Vec<(usize, usize, usize, usize)> {
    let mut sentences: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut cp = 0usize;
    for (byte_start, segment) in text.split_sentence_bound_indices() {
        sentences.push((
            cp,
            cp + segment.chars().count(),
            byte_start,
            byte_start + segment.len(),
        ));
        cp += segment.chars().count();
    }
    sentences
}

/// Compute the best-matching snippets of `text` for `query` — see the
/// module docs for the algorithm and its bounds. Never panics on any
/// input: empty or token-free query/text, and `max_snippets == 0`, all
/// return the empty result.
pub fn highlight(query: &str, text: &str, max_snippets: usize, max_chars: usize) -> Grounding {
    let empty = Grounding {
        snippets: Vec::new(),
        score: 0.0,
    };
    if query.is_empty() || text.is_empty() || max_snippets == 0 || max_chars == 0 {
        return empty;
    }
    let q: Vec<String> = tokens(query)
        .into_iter()
        .map(|t| t.norm)
        .take(MAX_QUERY_TOKENS)
        .collect();
    if q.is_empty() {
        return empty;
    }
    let q_set: std::collections::HashSet<&str> = q.iter().map(String::as_str).collect();
    let toks = tokens(text);
    if toks.is_empty() {
        return empty;
    }
    let norms: Vec<&str> = toks.iter().map(|t| t.norm.as_str()).collect();

    // --- anchor runs: maximal runs of consecutive matched tokens, merged
    // while the merged character span still fits max_chars.
    let mut runs: Vec<(usize, usize)> = Vec::new(); // token ranges
    let mut matched_counts: Vec<usize> = Vec::new();
    let mut i = 0usize;
    while i < toks.len() {
        if !q_set.contains(norms[i]) {
            i += 1;
            continue;
        }
        let (s, mut e, mut count) = (i, i + 1, 1usize);
        // Extend the run over consecutive matches while the span fits.
        while e < toks.len() && q_set.contains(norms[e]) && toks[e].end - toks[s].start <= max_chars
        {
            e += 1;
            count += 1;
        }
        runs.push((s, e));
        matched_counts.push(count);
        i = e;
    }
    if runs.is_empty() {
        return empty;
    }

    // --- density pre-rank: the top MAX_CANDIDATES runs by (matched desc,
    // position asc) reach the DP; the rest are discarded (the documented
    // adversarial-density cap).
    let mut order: Vec<usize> = (0..runs.len()).collect();
    order.sort_by(|&a, &b| {
        matched_counts[b]
            .cmp(&matched_counts[a])
            .then(runs[a].0.cmp(&runs[b].0))
    });
    order.truncate(MAX_CANDIDATES);

    // --- candidate expansion: sentence bounds (UAX #29, the citation unit
    // the ALCE snippet baselines use) when the sentence fits the budget;
    // then the max_chars clamp at token boundaries. Sentences keep their
    // own byte spans (built here rather than via `sentence_bounds`, whose
    // char-only tuples cannot slice the text) so an expanded snippet can
    // present the sentence's punctuation in its returned text: the offsets
    // are sentence bounds, not token bounds, by design.
    let sentences = sentence_spans(text);
    let mut scratch = Scratch::default();
    let mut candidates: Vec<Candidate> = Vec::with_capacity(order.len());
    for &r in &order {
        let (s, e) = runs[r];
        let (mut cs, mut ce) = (s, e);
        let mut presented = (
            toks[s].start,
            toks[e - 1].end,
            toks[s].byte_start,
            toks[e - 1].byte_end,
        );
        // The enclosing sentence, by the run's first token's start.
        if let Some(&(ss, se, bs, be)) = sentences
            .iter()
            .find(|&&(ss, se, _, _)| ss <= toks[s].start && toks[s].start < se)
            && se - ss <= max_chars
        {
            let s_lo = token_at_or_after(&toks, ss);
            let s_hi = token_strictly_after(&toks, se.saturating_sub(1));
            if s_lo < s_hi {
                cs = s_lo;
                ce = s_hi;
                presented = (ss, se, bs, be);
            }
        }
        // Clamp to max_chars at token boundaries. One shrink pass suffices:
        // the loop exits only when `cs + 1 == ce` (a single token left — the
        // documented keep-at-least-one-token floor) or the span already fits
        // `max_chars`, and the second condition (shrinking from the start)
        // requires BOTH to be false, so a start-shrink pass can never run.
        // (A sentence-expanded candidate fits by construction; the clamp
        // only ever bites run-only candidates.)
        while cs + 1 < ce && toks[ce - 1].end - toks[cs].start > max_chars {
            ce -= 1;
        }
        candidates.push(Candidate {
            tok_s: cs,
            tok_e: ce,
            start: presented.0,
            end: presented.1,
            byte_start: presented.2,
            byte_end: presented.3,
        });
    }

    // --- score, select, present.
    let mut scored: Vec<(f64, usize, usize, usize, usize, usize)> = candidates
        .into_iter()
        .map(|c| {
            let norm_slice: Vec<&str> = norms[c.tok_s..c.tok_e].to_vec();
            let score = rouge_w_f1(&q, &norm_slice, &mut scratch);
            (score, c.start, c.end, c.byte_start, c.byte_end, c.tok_s)
        })
        .filter(|&(score, _, _, _, _, _)| score > 0.0)
        .collect();
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    let mut chosen: Vec<(usize, usize, usize, usize, f64)> = Vec::new(); // (start, end, byte_start, byte_end, score)
    for &(score, start, end, byte_start, byte_end, _) in &scored {
        if chosen.len() == max_snippets {
            break;
        }
        if !chosen.iter().any(|&(s, e, _, _, _)| s < end && start < e) {
            chosen.push((start, end, byte_start, byte_end, score));
        }
    }
    if chosen.is_empty() {
        return empty;
    }
    let mut snippets: Vec<Snippet> = chosen
        .into_iter()
        .map(|(start, end, byte_start, byte_end, score)| Snippet {
            text: text[byte_start..byte_end].to_owned(),
            start,
            end,
            score,
        })
        .collect();
    // Position order for presentation: the selection above is score order.
    snippets.sort_by_key(|s| s.start);
    let score = snippets.iter().map(|s| s.score).fold(0.0, f64::max);
    Grounding { snippets, score }
}

/// One scored sentence of the batch: `text[start:end]` of the original
/// text (CHARACTER offsets, codepoint indices, the sentence's exact
/// bounds, so slicing the original always round-trips) plus the sentence's
/// ROUGE-W F1 against the query (`score`, in `[0.0, 1.0]`; `0.0` when the
/// sentence shares no token with the query).
#[derive(Debug, Clone, PartialEq)]
pub struct SentenceScore {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub score: f64,
}

/// The batch result: EVERY UAX #29 sentence of `text` in position order,
/// each with its score, and the aggregate (`max`, see [`ground_sentences`]'s
/// docs).
#[derive(Debug, Clone, PartialEq)]
pub struct SentenceGrounding {
    pub sentences: Vec<SentenceScore>,
    pub score: f64,
}

/// The grounding tokenizer's normalized token stream as plain strings,
/// `grounded_impl::grounding_coverage`'s input (the coverage core lives
/// beside the difflib verdict it twins but scores over the SAME
/// tokenization the grounding family scores with, so the three surfaces
/// never disagree about what a token is). At most [`MAX_TEXT_TOKENS`]
/// tokens, the same bounded-scan discipline every consumer here applies.
pub(crate) fn normalized_tokens(text: &str) -> Vec<String> {
    tokens(text).into_iter().map(|t| t.norm).collect()
}

/// Score EVERY UAX #29 sentence of `text` against `query`: the batch
/// bridge primitive a downstream NLI verifier (MiniCheck/SummaC-style)
/// consumes: per-sentence `{start, end, text, score}` in position order
/// (offsets are CHARACTER indices into `text`, `text[s.start:s.end] ==
/// s.text` exactly) plus an aggregate score. Citation unit = sentence, the
/// ALCE baselines' own unit (Gao et al. 2023, "Enabling Large Language
/// Models to Generate Text with Citations"); the per-sentence score is the
/// same ROUGE-W F1 the snippet surface ranks spans with (Lin 2004; see
/// the module docs).
///
/// # Aggregate policy: MAX, not mean
///
/// The aggregate is the best sentence's F1 (`0.0` when there are no
/// sentences or the query matches nothing). Three reasons, in decreasing
/// weight: (1) it is the retrieval signal the downstream consumer needs:
/// "does SOME sentence carry this query's evidence" is the question an
/// NLI bridge answers per sentence, and the max is exactly the ranking
/// key that picks the evidence sentence; (2) it mirrors [`highlight`]'s
/// own aggregate (the best snippet's score), keeping the family coherent;
/// (3) it is stable under irrelevant additions: a long document with one
/// relevant sentence must not read as ungrounded because the document is
/// long, which a mean rewards forgetting (and ALCE's sentence-level
/// citation selection is likewise a max/argmax over units, not an
/// average). The mean is deliberately NOT offered: it answers "how much
/// of this document is about the query", a different question, and one
/// `sentence_bounds` + a fold in Python composes trivially from the
/// per-sentence scores this returns.
///
/// # Bounds
///
/// Total DP work is `sum_i |Q| * |s_i| <= |Q| * N` (N = the text's tokens,
/// capped at [`MAX_TEXT_TOKENS`]; the query capped at [`MAX_QUERY_TOKENS`])
/// is linear in the text at a bounded query width, one reusable
/// [`Scratch`] never wider than the longest sentence's token count. A
/// sentence whose tokens fall past the cap (a pathological single-token-
/// flood chunk) scores `0.0`: its tokens are not evidence for this call,
/// the same discipline `highlight` applies, while its offsets and text
/// stay exact. `max_chars` (when set) bounds the SCORED WINDOW of a
/// sentence: a sentence longer than the budget is scored over its leading
/// token-boundary window that fits (at least one token, `highlight`'s
/// documented floor), its returned `start`/`end`/`text` still covering the
/// WHOLE sentence: the budget bounds the DP, never the report. A value of
/// 0 is rejected at the binding; `None` (the default) scores whole
/// sentences.
///
/// Empty/token-free `query`, or empty `text`, returns the matching empty
/// shape (no sentences / all-zero scores): degenerate input is a valid
/// answer, never an error, the same convention [`highlight`] pins.
pub fn ground_sentences(query: &str, text: &str, max_chars: Option<usize>) -> SentenceGrounding {
    let mut sentences: Vec<SentenceScore> = Vec::new();
    let mut best = 0.0f64;
    if text.is_empty() {
        return SentenceGrounding {
            sentences,
            score: 0.0,
        };
    }
    let q: Vec<String> = tokens(query)
        .into_iter()
        .map(|t| t.norm)
        .take(MAX_QUERY_TOKENS)
        .collect();
    let toks = tokens(text);
    let norms: Vec<&str> = toks.iter().map(|t| t.norm.as_str()).collect();
    let mut scratch = Scratch::default();
    for &(ss, se, bs, be) in &sentence_spans(text) {
        // The sentence's token range: binary searches, the same pair
        // `highlight`'s sentence-expansion step uses (tokens never straddle
        // a sentence boundary in practice, and if one ever did the offsets
        // below are the SENTENCE's, so the round-trip holds regardless).
        let lo = token_at_or_after(&toks, ss);
        let mut hi = token_strictly_after(&toks, se.saturating_sub(1));
        let mut score = 0.0f64;
        if !q.is_empty() && lo < hi {
            // The max_chars clamp at token boundaries, at least one token
            // kept (the documented floor): bounds the DP width on the
            // pathological one-giant-sentence shape.
            if let Some(mc) = max_chars {
                while hi > lo + 1 && toks[hi - 1].end - toks[lo].start > mc {
                    hi -= 1;
                }
            }
            score = rouge_w_f1(&q, &norms[lo..hi], &mut scratch);
        }
        best = best.max(score);
        sentences.push(SentenceScore {
            text: text[bs..be].to_owned(),
            start: ss,
            end: se,
            score,
        });
    }
    SentenceGrounding {
        sentences,
        score: best,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(text: &str, g: &Grounding) {
        for s in &g.snippets {
            // THE load-bearing property: slicing the original by the
            // returned CHARACTER offsets yields exactly the snippet text.
            // Rust's own `text[s.start..s.end]` is NOT that check — Rust
            // slices by UTF-8 BYTES and these offsets are Python codepoint
            // indices (they coincide only in ASCII) — so the slice goes
            // through `chars()`; the Python-side byte-free `str[start:end]`
            // is pinned by tests/test_grounding.py.
            let sliced: String = text.chars().collect::<Vec<char>>()[s.start..s.end]
                .iter()
                .collect();
            assert_eq!(sliced, s.text);
        }
    }

    #[test]
    fn the_exact_match_is_found_and_slices_back() {
        let text = "The quick brown fox jumps over the lazy dog.";
        // A tight max_chars: the snippet is exactly the query's tokens, so
        // the ROUGE-W F1 against a candidate = the query itself is 1.0.
        let g = highlight("lazy dog", text, 3, 10);
        assert_eq!(g.snippets.len(), 1);
        let s = &g.snippets[0];
        roundtrip(text, &g);
        assert_eq!(s.text, "lazy dog");
        assert!(
            s.score > 0.99,
            "a candidate holding the whole contiguous query scores ~1.0"
        );
        assert_eq!(g.score, s.score);
    }

    #[test]
    fn a_snippet_expands_to_its_sentence_when_the_sentence_fits_the_budget() {
        // The ALCE citation-unit behavior: with room for the sentence, the
        // snippet is the whole sentence, not the bare term.
        let text = "The quick brown fox jumps over the lazy dog.";
        let g = highlight("lazy dog", text, 3, 400);
        assert_eq!(g.snippets.len(), 1);
        assert_eq!(g.snippets[0].text, text);
        roundtrip(text, &g);
    }

    #[test]
    fn snippets_are_character_offsets_that_roundtrip() {
        // Multi-codepoint emoji, accents, CJK: the offsets must be codepoint
        // indices (Rust chars), which Python str slicing shares.
        let text = "café ☕ summary — 日本語のテキスト continues 🎉thonk end";
        let g = highlight("日本語のテキスト", text, 3, 400);
        assert_eq!(g.snippets.len(), 1);
        let s = &g.snippets[0];
        assert!(s.text.contains("日本語"));
        roundtrip(text, &g);
    }

    #[test]
    fn cjk_text_is_tokenized_per_character_so_runs_can_align() {
        // No spaces: unspaced CJK morphemes must become individual tokens
        // or no partial query could ever score.
        let text = "検索対象の文書には重要な情報が含まれています。";
        // Tight budget: the snippet is exactly the matched run, so the
        // candidate equals the query and scores ~1.0.
        let g = highlight("重要な情報", text, 3, 12);
        assert_eq!(g.snippets.len(), 1);
        assert!(g.snippets[0].text.contains("重要な情報"));
        assert!(g.snippets[0].score > 0.5);
        roundtrip(text, &g);
    }

    #[test]
    fn katakana_runs_subsplit_so_partial_terms_match() {
        // UAX #29 keeps Katakana runs joined (テキスト is ONE word segment);
        // the CJK refinement splits them per character, so a query term
        // shorter than the run still matches inside it.
        let text = "このテキストを検索する";
        let g = highlight("キスト", text, 3, 400);
        assert!(!g.snippets.is_empty());
        assert!(g.snippets[0].text.contains("キスト"));
        roundtrip(text, &g);
    }

    #[test]
    fn multiple_scattered_matches_yield_multiple_non_overlapping_snippets() {
        let text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima";
        let g = highlight("alpha kilo", text, 3, 12);
        assert_eq!(g.snippets.len(), 2);
        // Non-overlapping, position-ordered.
        assert!(g.snippets[0].end <= g.snippets[1].start);
        assert!(g.snippets[0].text.contains("alpha"));
        assert!(g.snippets[1].text.contains("kilo"));
        for s in &g.snippets {
            assert!(s.text.chars().count() <= 12, "max_chars respected");
        }
        roundtrip(text, &g);
    }

    #[test]
    fn no_overlap_means_no_snippets_and_a_zero_score() {
        let g = highlight("zebra", "the quick brown fox", 3, 400);
        assert!(g.snippets.is_empty());
        assert_eq!(g.score, 0.0);
    }

    #[test]
    fn degenerate_inputs_return_the_empty_result_without_panicking() {
        for (q, t) in [
            ("", "some text"),
            ("some", ""),
            ("", ""),
            ("   !!!   ", "some text"),
            ("some", "   !!!   "),
        ] {
            let g = highlight(q, t, 3, 400);
            assert!(g.snippets.is_empty());
            assert_eq!(g.score, 0.0);
        }
        assert!(highlight("q", "text", 0, 400).snippets.is_empty());
        assert!(highlight("q", "text", 3, 0).snippets.is_empty());
    }

    #[test]
    fn max_chars_caps_every_snippet_at_token_boundaries() {
        let text = "one two three four five six seven eight nine ten eleven twelve";
        let g = highlight("seven eight nine ten", text, 3, 10);
        assert!(!g.snippets.is_empty());
        for s in &g.snippets {
            assert!(s.text.chars().count() <= 10);
            // Token-boundary clean: no snippet starts or ends mid-word.
            assert!(s.text.chars().all(|c| c.is_alphanumeric() || c == ' '));
        }
        roundtrip(text, &g);
    }

    #[test]
    fn pathological_input_stays_bounded() {
        // A 560k-character chunk of adversarial repeats: the caps keep the
        // pass linear-ish and the test merely proves it completes and stays
        // correct (run under `cargo test`, this is the DoS regression guard
        // — an unbounded formulation would hang here). Identical disjoint
        // regions beyond the first stay non-overlapping snippets (up to
        // max_snippets), each slicing back exactly.
        let text = "relevant term ".repeat(40_000);
        let g = highlight("relevant term", &text, 3, 400);
        assert_eq!(g.snippets.len(), 3);
        for s in &g.snippets {
            assert_eq!(&text[s.start..s.end], s.text);
            assert!(s.text.contains("relevant term"));
            assert!(s.score > 0.0);
        }
        roundtrip(&text, &g);
    }

    #[test]
    fn ties_break_by_earliest_position_deterministically() {
        let text = "tick tock tick tock tick tock tick tock";
        let a = highlight("tick", text, 1, 400);
        let b = highlight("tick", text, 1, 400);
        assert_eq!(a, b);
        assert_eq!(a.snippets[0].start, 0);
    }

    #[test]
    fn case_folding_and_punctuation_do_not_block_matching() {
        let text = "The EMBEDDING model processes documents quickly.";
        let g = highlight("embedding model", text, 3, 400);
        assert_eq!(g.snippets.len(), 1);
        assert!(g.snippets[0].text.contains("EMBEDDING model"));
    }

    #[test]
    fn mixed_scripts_align_in_one_pass() {
        let text = "Le café français 東京タワー theEmbeddingPipeline 🎉 finale";
        let g = highlight("東京タワー café", text, 3, 400);
        assert_eq!(g.snippets.len(), 1);
        let s = &g.snippets[0];
        roundtrip(text, &g);
        assert!(s.text.contains("東京タワー"));
        assert!(s.text.contains("café"));
    }

    #[test]
    fn snippet_spans_never_split_a_multi_codepoint_character() {
        // ZWJ emoji family: a byte-offset bug splits inside the sequence and
        // produces an un-sliceable span; a char-offset implementation cannot.
        let text = "prefix 👨‍👩‍👧‍👦 family suffix";
        let g = highlight("family", text, 3, 400);
        let s = &g.snippets[0];
        roundtrip(text, &g);
        assert!(s.text.contains("family"));
    }

    #[test]
    fn combining_marks_stay_attached_through_nfd() {
        // NFD "café" = "cafe\u{301}": the combining mark must ride with its
        // base (UAX #29 WB4), and the offsets must still slice the original.
        let text = "The cafe\u{301} is open — the cafe\u{301} au lait too.";
        let g = highlight("café", text, 3, 400);
        assert!(!g.snippets.is_empty());
        assert!(g.snippets[0].text.contains("cafe\u{301}"));
        roundtrip(text, &g);
    }

    #[test]
    fn rtl_text_matches_in_logical_order() {
        let text = "المادة رقم ٥ من القانون تنص على أن العقد ملزم";
        let g = highlight("العقد ملزم", text, 3, 400);
        assert!(!g.snippets.is_empty());
        roundtrip(text, &g);
        assert!(g.snippets[0].text.contains("العقد ملزم"));
    }

    #[test]
    fn astral_plane_characters_offset_correctly() {
        // Astral (4-byte UTF-8) chars on BOTH sides of the match: a
        // byte-index bug drifts the offsets; a codepoint index cannot. (The
        // mathematical-alphanumeric run is a DIFFERENT term from ASCII
        // "alignment" — exact-term matching, not fuzzy — so the query is
        // the ASCII word alone; the assertion is the offsets, not fuzzy
        // recall.)
        let text = "𝕌𝕟𝕚𝕔𝕠𝕕𝕖 target 𝕒𝕝𝕚𝕘𝕟𝕞𝕖𝕟𝕥 𝕌𝕟𝕚𝕔𝕠𝕕𝕖";
        let g = highlight("target", text, 3, 400);
        assert_eq!(g.snippets.len(), 1);
        roundtrip(text, &g);
        assert!(g.snippets[0].text.contains("target"));
        assert!(g.snippets[0].text.contains("𝕌𝕟𝕚𝕔𝕠𝕕𝕖"));
    }

    #[test]
    fn a_single_repeated_codepoint_never_panics() {
        // One 10k-char "word" of 'a': the term "a" does not occur as a
        // word in it (exact-term matching), so nothing anchors — while the
        // whole string AS the query matches its own token exactly.
        let text = "a".repeat(10_000);
        let g = highlight("a", &text, 3, 400);
        assert!(g.snippets.is_empty());
        assert_eq!(g.score, 0.0);
        let q = "a".repeat(10_000);
        let g2 = highlight(&q, &text, 3, 400);
        assert_eq!(g2.snippets.len(), 1);
        roundtrip(&text, &g2);
    }

    #[test]
    fn a_query_longer_than_the_chunk_still_anchors() {
        let g = highlight(
            "the quick brown fox jumps over the lazy dog",
            "lazy",
            3,
            400,
        );
        assert!(!g.snippets.is_empty());
        assert_eq!(g.snippets[0].text, "lazy");
        roundtrip("lazy", &g);
    }

    #[test]
    fn a_snippet_always_keeps_at_least_one_token() {
        // A budget smaller than any token: the clamp stops at one token
        // rather than returning an empty span (documented graceful
        // degeneracy — a snippet is at least a token, never a fragment).
        let text = "some content here";
        let g = highlight("content", text, 3, 1);
        assert_eq!(g.snippets.len(), 1);
        assert_eq!(g.snippets[0].text, "content");
        roundtrip(text, &g);
    }

    #[test]
    fn rouge_w_prefers_the_contiguous_candidate() {
        // The shaping's reason to exist, pinned: two candidates holding the
        // same two query terms, one adjacent, one spread through filler —
        // the contiguous one scores strictly higher (plain LCS would tie).
        let q = vec!["alpha".to_string(), "beta".to_string()];
        let contiguous = vec!["alpha", "beta", "filler"];
        let spread = vec!["alpha", "x", "y", "z", "beta"];
        let mut s = Scratch::default();
        let a = rouge_w_f1(&q, &contiguous, &mut s);
        let b = rouge_w_f1(&q, &spread, &mut s);
        assert!(a > b, "contiguous {a} must outrank spread {b}");
    }

    #[test]
    fn rouge_w_f1_matches_a_hand_computed_value() {
        // Oracle check on the DP itself, hand-computed with Lin 2004's
        // Equation 15 normalization (f(k) = k^1.2, R = f^-1(WLCS/f(m))):
        // q = [a, b], c = [a, x, b] — two runs of one match each, so
        // wlcs = f(1) + f(1) = 2.0.
        let q = vec!["a".to_string(), "b".to_string()];
        let c = vec!["a", "x", "b"];
        let mut s = Scratch::default();
        let got = rouge_w_f1(&q, &c, &mut s);
        let f = |k: f64| k.powf(1.2);
        let finv = |x: f64| x.powf(1.0 / 1.2);
        let r = finv(2.0 / f(2.0));
        let p = finv(2.0 / f(3.0));
        let want = 2.0 * r * p / (r + p);
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        // No shared term scores exactly 0.
        assert_eq!(rouge_w_f1(&q, &["x", "y"], &mut s), 0.0);
    }

    #[test]
    fn the_fill_is_monotone_in_the_candidate_where_the_forced_diagonal_was_not() {
        // The coverage fuzz target's find, pinned at the unit level: Lin's
        // Figure 3 spelling (a match cell FORCES the diagonal) reads
        // [o, o] vs [o, o, uua, o] off the last cell as two stranded
        // single matches (wlcs 2.0) where the optimal alignment holds one
        // run of two (wlcs f(2) = 2.297), a LONGER candidate scoring
        // LESS. The max-on-match fill restores the optimum and with it
        // candidate-monotonicity: the identical-pair wlcs f(2) survives
        // appending tokens untouched.
        let ids = |norms: &[&str]| -> Vec<u32> {
            let mut out = Vec::new();
            let mut vocab: Vec<&str> = Vec::new();
            for n in norms {
                match vocab.iter().position(|v| v == n) {
                    Some(i) => out.push(i as u32),
                    None => {
                        vocab.push(n);
                        out.push((vocab.len() - 1) as u32)
                    }
                }
            }
            out
        };
        let mut s = Scratch::default();
        let base = ids(&["o", "o"]);
        let extended = ids(&["o", "o", "uua", "o"]);
        let wlcs_base = rouge_w_wlcs(&base, &base, &mut s);
        let wlcs_ext = rouge_w_wlcs(&base, &extended, &mut s);
        let f2 = 2.0f64.powf(1.2);
        assert!((wlcs_base - f2).abs() < 1e-12);
        assert!((wlcs_ext - f2).abs() < 1e-12, "wlcs {wlcs_ext} vs {f2}");
        assert!(wlcs_ext >= wlcs_base - 1e-12);
    }

    #[test]
    fn every_score_stays_in_the_unit_interval() {
        // The fuzz-found over-score shape, pinned at the unit level: the
        // densest possible candidates (every token a match, runs of every
        // length) must all score <= 1.0.
        let q = vec!["a".to_string(), "b".to_string()];
        let mut s = Scratch::default();
        for run in 1..12 {
            for extra in 0..12 {
                let c: Vec<&str> = (0..run)
                    .map(|i| if i % 2 == 0 { "a" } else { "b" })
                    .chain(std::iter::repeat_n("z", extra))
                    .collect();
                let score = rouge_w_f1(&q, &c, &mut s);
                assert!(
                    (0.0..=1.0).contains(&score),
                    "run {run} extra {extra}: {score}"
                );
            }
        }
        // Full contiguous match, query == candidate: exactly 1.0.
        let c = vec!["a", "b"];
        assert!((rouge_w_f1(&q, &c, &mut s) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn token_cap_stops_the_scan_and_offsets_stay_sound() {
        // Text with far more tokens than MAX_TEXT_TOKENS: the scan stops at
        // the cap and the emitted tokens' offsets still slice exactly.
        let text = "word ".repeat(30_000);
        let toks = tokens(&text);
        assert_eq!(toks.len(), MAX_TEXT_TOKENS);
        for t in &toks {
            assert_eq!(&text[t.byte_start..t.byte_end], &text[t.start..t.end]);
        }
    }

    // --- ground_sentences: the batch core's unit pins -------------------

    fn roundtrip_sentences(text: &str, g: &SentenceGrounding) {
        for s in &g.sentences {
            // The same codepoint-slice check the snippet round-trip makes.
            let sliced: String = text.chars().collect::<Vec<char>>()[s.start..s.end]
                .iter()
                .collect();
            assert_eq!(sliced, s.text);
        }
    }

    #[test]
    fn every_sentence_is_scored_in_position_order() {
        let text = "The quick brown fox jumps. Second sentence has the fox again. Done.";
        let g = ground_sentences("fox", text, None);
        assert_eq!(g.sentences.len(), 3);
        assert!(g.sentences[0].score > 0.0);
        assert!(g.sentences[1].score > 0.0);
        assert_eq!(g.sentences[2].score, 0.0);
        // Position order: starts strictly increasing, ends non-overlapping.
        for pair in g.sentences.windows(2) {
            assert!(pair[0].start < pair[1].start);
            assert!(pair[0].end <= pair[1].start);
        }
        // Aggregate = max (the documented policy).
        assert_eq!(g.score, 0.0f64.max(g.sentences[0].score));
        assert_eq!(
            g.score,
            g.sentences.iter().map(|s| s.score).fold(0.0, f64::max)
        );
        roundtrip_sentences(text, &g);
    }

    #[test]
    fn identical_text_and_query_scores_one_point_zero() {
        let text = "The bushing torque specifications changed.";
        let g = ground_sentences(text, text, None);
        assert_eq!(g.sentences.len(), 1);
        assert!((g.sentences[0].score - 1.0).abs() < 1e-12);
        assert_eq!(g.score, g.sentences[0].score);
    }

    #[test]
    fn a_whole_query_sentence_outranks_a_partial_overlap() {
        // Monotonicity, pinned: the sentence holding EVERY query token
        // (contiguously) must outscore the sentence holding only part.
        let text = "Partial torque only. The full bushing torque spec here. Nothing at all.";
        let g = ground_sentences("full bushing torque spec", text, None);
        assert!(g.sentences[1].score > g.sentences[0].score);
        assert!(g.sentences[1].score > g.sentences[2].score);
    }

    #[test]
    fn sentence_spans_match_the_published_sentence_bounds() {
        // The batch's offsets ARE `sentence_bounds`' tuples, by construction
        // (the shared `sentence_spans` builder), pinned against the
        // segmentation surface the docs publish.
        let text = "One. Two!! Three\r\nfour. 3.4 percent stays one.";
        let bounds = crate::segmentation_impl::sentence_bounds(text);
        let g = ground_sentences("three", text, None);
        let got: Vec<(usize, usize)> = g.sentences.iter().map(|s| (s.start, s.end)).collect();
        assert_eq!(got, bounds);
    }

    #[test]
    fn max_chars_clamps_the_scored_window_but_not_the_report() {
        let text = "A very long sentence holding the torque term deep inside its middle.";
        let g = ground_sentences("torque", text, Some(10));
        assert_eq!(g.sentences.len(), 1);
        let s = &g.sentences[0];
        // The report still covers the whole sentence; the scored window was
        // clamped (score 0: the term sits past the leading window).
        assert_eq!(text[s.start..s.end] /* char-safe: ASCII */, s.text);
        assert_eq!(s.text, text);
        assert_eq!(s.score, 0.0);
        // Unclamped, the term is inside the scored window.
        let whole = ground_sentences("torque", text, None);
        assert!(whole.sentences[0].score > 0.0);
    }

    #[test]
    fn degenerate_batch_inputs_are_valid_answers() {
        // Empty text: no sentences at all.
        let g = ground_sentences("query", "", None);
        assert!(g.sentences.is_empty());
        assert_eq!(g.score, 0.0);
        // Empty/token-free query: every sentence still reported, all zero.
        for q in ["", "   !!!   "] {
            let g = ground_sentences(q, "Two sentences here. Another one.", None);
            assert_eq!(g.sentences.len(), 2);
            assert!(g.sentences.iter().all(|s| s.score == 0.0));
            assert_eq!(g.score, 0.0);
        }
    }

    #[test]
    fn the_batch_round_trips_through_every_script() {
        for text in [
            "検索対象の文書には重要な情報が含まれています。次の文もある。",
            "cafe\u{0301} au lait is served. NFD accents stay attached.",
            "prefix 👨‍👩‍👧‍👦 family sentence. Suffix sentence.",
            "المادة رقم ٥ من القانون تنص على أن العقد ملزم. وتابع.",
        ] {
            let g = ground_sentences("検索 重要 family العقد café", text, None);
            roundtrip_sentences(text, &g);
            assert!(g.sentences.iter().all(|s| (0.0..=1.0).contains(&s.score)));
        }
    }

    #[test]
    fn the_batch_is_deterministic() {
        let text = "tick tock one. tick tock two. tick tock three.";
        let a = ground_sentences("tick", text, None);
        let b = ground_sentences("tick", text, None);
        assert_eq!(a, b);
    }

    #[test]
    fn rouge_w_recall_reaches_one_only_on_full_contiguous_coverage() {
        // The ids API `grounded_impl::grounding_coverage` drives: intern
        // the streams into ONE shared vocabulary (equal norms -> equal ids;
        // separate vocabularies would collide ids across streams), fill,
        // normalize by the reference stream's length.
        fn side<'a>(
            vocab: &mut std::collections::HashMap<&'a str, u32>,
            next: &mut u32,
            norms: &[&'a str],
        ) -> Vec<u32> {
            norms
                .iter()
                .map(|n| {
                    *vocab.entry(n).or_insert_with(|| {
                        let v = *next;
                        *next += 1;
                        v
                    })
                })
                .collect()
        }
        let recall = |q: &[&str], c: &[&str]| {
            let mut vocab: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
            let mut next = 0u32;
            let qi = side(&mut vocab, &mut next, q);
            let ci = side(&mut vocab, &mut next, c);
            let mut s = Scratch::default();
            let wlcs = rouge_w_wlcs_ids(&qi, &ci, &mut s);
            rouge_w_recall_from_wlcs(qi.len(), wlcs)
        };
        let q = ["a", "b", "c"];
        assert!((recall(&q, &["a", "b", "c"]) - 1.0).abs() < 1e-12);
        // Partial coverage: recall < 1 but > 0, and stays in the unit
        // interval for every coverage shape.
        for c in [
            vec!["a", "x", "b"],
            vec!["a", "b", "c", "extra"],
            vec!["x", "y"],
        ] {
            let r = recall(&q, &c);
            assert!((0.0..=1.0).contains(&r), "{c:?}: {r}");
        }
        assert_eq!(recall(&q, &["x", "y"]), 0.0);
        // Recalled CONTIGUOUS coverage outranks the same tokens scattered
        // (the shaping's bias, on the recall axis too).
        let contiguous = recall(&q, &["a", "b", "c", "z"]);
        let spread = recall(&q, &["a", "z", "b", "z", "c", "z"]);
        assert!(contiguous > spread);
        // The family identity on the full-containment shape: a candidate
        // holding EXACTLY the query's tokens has precision 1, so F1 equals
        // recall there (a candidate with extra tokens deflates precision
        // and F1 below recall; the F1/recall gap is the precision cost).
        let exact: Vec<&str> = q.to_vec();
        let mut s = Scratch::default();
        let f1 = rouge_w_f1(
            &q.iter().map(|s| (*s).to_string()).collect::<Vec<_>>(),
            &exact,
            &mut s,
        );
        assert!((f1 - recall(&q, &exact)).abs() < 1e-9, "p=1 case: f1 {f1}");
        // Symmetry of the fill across operand order (the ids API's caller
        // exploits it to put the shorter stream in the rows): the same
        // raw WLCS either way.
        let mut vocab: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
        let mut next = 0u32;
        let qi = side(&mut vocab, &mut next, &q);
        let zi = side(&mut vocab, &mut next, &["a", "z", "b"]);
        let mut s2 = Scratch::default();
        let wlcs_ab = rouge_w_wlcs_ids(&qi, &zi, &mut s2);
        let mut s3 = Scratch::default();
        let wlcs_ba = rouge_w_wlcs_ids(&zi, &qi, &mut s3);
        assert!((wlcs_ab - wlcs_ba).abs() < 1e-12);
        assert_eq!(recall(&q, &["a", "z", "b"]), recall(&["a", "z", "b"], &q));
    }
}
