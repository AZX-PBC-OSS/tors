"""Scaling pins for the hot paths the suite had not yet pinned: one
growth-ratio cell per (surface, adversarial shape) that had NO scaling
pin when the superlinear-defect wave (#91 #92 #101 #102 #111) landed.
The pattern is test_chunk_text_overlap_scaling.py's, simplified to its
two-size core: each cell times a small size n and a large one (2n or
4n) with min-of-samples walls after a warmup and asserts the growth
ratio stays below the complexity class's bound with a generous margin
— a linear path doubles ~2.0-2.3x per doubling on a quiet box, so the
3.0x-per-doubling gate (the parity suite's own C1 wording: "doubling
... less than triples") sits ~1.3x above the measured band while any
quadratic (4x per doubling) blows straight through it. Ratios are
wall-clock only, never absolute times, so a shared-runner slowdown
inflates both sizes together.

Every shape below is one of the documented defect classes: PEM-flood
and escape-dense scrub_pii (#92's shape, fixed and now pinned),
separator-heavy chunk_hierarchical hierarchies (#103's neighborhood),
traceback-shaped scrub_log_text corpora (the parity suite's C1 pin
covers the userinfo fail-chain; this adds the DETAIL/traceback shape),
deep and wide repair_json schemas (#113's neighborhood; the WIDE axis
is deliberately xfail — see the class docstring), match-dense
replace_many (#101's surface, unmasked), deeply-nested documents HTML,
and the retrieval trio at their documented cost ceilings (#91's
neighborhood).

Cells that measure multi-millisecond walls are marked ``timing`` (the
suite's slow/load-sensitive lane, run once in CI on the 3.12 leg;
`make test` runs everything locally). Sub-millisecond large cells are
left unmarked: their min-of-5 floor is only robust against noise
because both sizes inflate together, and the absolute cost is
microseconds.
"""

from __future__ import annotations

import json
from time import monotonic

import pytest

import tors

# The growth gate per doubling: a linear path measures ~2.0-2.3x on a
# quiet box (the chunk_text pin's own post-fix band: 2.03-2.26x); 3.0x
# is ~1.3x above that band while a quadratic's 4x trips it. Shaped
# cells whose measured band runs higher carry their own gate with the
# band recorded in the docstring.
LINEAR_GATE_PER_DOUBLING = 3.0


def _min_wall_ms(fn, *, samples: int = 5) -> float:
    """Min-of-N wall milliseconds after one warmup call: the fastest of
    several draws approximates the uncontended cost, the measurement
    discipline test_chunk_text_overlap_scaling.py's _min_wall_ms and
    test_performance.py's timing cells share."""
    fn()
    best = float("inf")
    for _ in range(samples):
        started = monotonic()
        fn()
        best = min(best, monotonic() - started)
    return best * 1e3


def _assert_linear_per_doubling(small_ms: float, large_ms: float, factor: int, gate: float) -> None:
    """The shared ratio assert: the large wall must stay under the gate
    per doubling of the input. factor is the large/small SIZE ratio."""
    import math

    doublings = math.log2(factor)
    allowed = gate**doublings
    assert large_ms < allowed * small_ms, (
        f"cost grew {small_ms:.2f}ms -> {large_ms:.2f}ms for a {factor}x input "
        f"({large_ms / small_ms:.2f}x, allowed {allowed:.1f}x at {gate:.1f}x per "
        "doubling): the path grew superlinear in the input"
    )


# --- chunk_hierarchical: separator-heavy custom hierarchies ------------------------
#
# The LangChain-pattern surface takes ANY caller literal list; a list of
# alphabet literals that all match dense "ab"-run text at high rates is
# the adversarial shape (repeats, overlaps, prefix chains), the one where
# per-level match rescans would show a superlinear add-on.


def _chunk_hier_shape(n: int) -> list[tuple[int, int]]:
    text = "ab" * n
    separators = ["a", "ab", "ba", "abab", "b"]
    return tors.chunk_hierarchical(text, 64, separators=separators, overlap=8)


class TestChunkHierarchicalSeparatorScaling:
    @pytest.mark.timing
    def test_separator_heavy_lists_stay_linear(self) -> None:
        """Dense-match separators (every list entry matches most
        positions) at 10k -> 40k codepoints (4x): measured 0.2ms ->
        0.6ms, ratio 4.2 (linear; 2.05x per doubling), gate 3.0x per
        doubling. A per-window rescan of the separator list (the
        quadratic shape #102's cousin) measures ~4x per doubling here."""
        small, large = _min_wall_ms(lambda: _chunk_hier_shape(10_000)), _min_wall_ms(
            lambda: _chunk_hier_shape(40_000)
        )
        _assert_linear_per_doubling(small, large, 4, LINEAR_GATE_PER_DOUBLING)


# --- scrub_pii: PEM flood + escape-dense text --------------------------------------
#
# #92's flood shape: every header a different algorithm word (the
# attacker's choice that defeated the memo), no verifying END. Fixed
# (index-by-words); the pin holds the line.


def _pem_flood(n: int) -> str:
    begins = "".join(f"-----BEGIN K{i} PRIVATE KEY-----\n" for i in range(n))
    ends = "".join(f"-----END L{i} PRIVATE KEY-----\n" for i in range(n))
    return begins + ends


class TestScrubPiiFloodScaling:
    @pytest.mark.timing
    def test_pem_flood_stays_linear(self) -> None:
        """"25k -> 100k headers (4x, ~3MB text): measured 17.8ms -> 68.8ms,
        ratio 3.9 (linear; ~1.97x per doubling), gate 3.0x per doubling.
        The pre-fix cubic measured 8x the input -> a few hundred x the
        time (~4.7x per doubling of the header count at these sizes)."""
        small, large = (
            _min_wall_ms(lambda: tors.scrub_pii(_pem_flood(25_000))),
            _min_wall_ms(lambda: tors.scrub_pii(_pem_flood(100_000))),
        )
        _assert_linear_per_doubling(small, large, 4, LINEAR_GATE_PER_DOUBLING)

    @pytest.mark.timing
    def test_escape_dense_text_stays_linear(self) -> None:
        """\\uXXXX-escaped prose with keys sprinkled every 97 units — the
        #100 shape the boundary rule must answer per position. 25k ->
        100k escapes (4x): measured 3.0ms -> 12.5ms, ratio 4.1 (linear;
        ~2.03x per doubling), gate 3.0x per doubling."""
        def shape(n: int) -> str:
            pieces = []
            for i in range(n):
                pieces.append("\\u%04x" % (0x41 + (i % 26)))
                if i % 97 == 0:
                    pieces.append("sk-abcdefghijklmnopqrstuvwx")
            return "".join(pieces)

        small, large = (
            _min_wall_ms(lambda: tors.scrub_pii(shape(25_000))),
            _min_wall_ms(lambda: tors.scrub_pii(shape(100_000))),
        )
        _assert_linear_per_doubling(small, large, 4, LINEAR_GATE_PER_DOUBLING)


# --- scrub_log_text: traceback-shaped corpora --------------------------------------
#
# The parity suite's C1 pin covers the userinfo fail-chain; this cell
# pins the OTHER corpus shape the chain's DETAIL rule walks: repr-flattened
# traceback frames, one DSN-bearing exception per unit.


def _traceback_corpus(n: int) -> str:
    frames = []
    for i in range(n):
        frames.append(f'  File "/app/pkg/mod{i % 50}.py", line {i}, in handler{i % 7}')
        frames.append(
            f"    raise ValueError('db dsn postgres://admin:sekret{i}"
            f"@db-{i % 3}.internal/main failed')\\nDETAIL:  op={i} trace=traceback\\n"
        )
    return "".join(frames)


class TestScrubLogTextTracebackScaling:
    @pytest.mark.timing
    def test_traceback_shaped_corpus_stays_linear(self) -> None:
        """12k -> 48k frame pairs (4x, ~2.6MB): measured 3.7ms -> 18.3ms,
        ratio 5.0 — ~2.24x per doubling, mildly above the clean-2.0 band
        (the chain's per-line regex work plus O(matches) output
        marshalling), so this cell carries its own 2.5x-per-doubling
        gate. The quadratic relapse (~4x per doubling) trips it."""
        small, large = (
            _min_wall_ms(lambda: tors.scrub_log_text(_traceback_corpus(12_000))),
            _min_wall_ms(lambda: tors.scrub_log_text(_traceback_corpus(48_000))),
        )
        _assert_linear_per_doubling(small, large, 4, 2.5)


# --- repair_json: deep + wide schemas ----------------------------------------------


def _deep_schema(levels: int) -> dict:
    schema: dict = {}
    node = schema
    for _ in range(levels):
        node["properties"] = {"a": {"type": "object"}}
        node = node["properties"]["a"]
    return schema


class TestRepairJsonSchemaScaling:
    @pytest.mark.timing
    def test_deep_schema_walk_stays_linear_in_depth(self) -> None:
        """40 -> 80 nested levels (2x, both under the documented 200-unit
        walk cap; past it the walk raises ValueError — that contract is
        pinned corpus-side): measured 0.15ms -> 0.32ms, ratio 2.1
        (linear), gate 3.0x per doubling. A per-level rescan of the
        accumulated schema path (the quadratic shape) measures 4x. Load-
        sensitive sub-millisecond ratios: the timing lane, the same
        discipline as the wide-schema sibling below."""
        small = _min_wall_ms(lambda: tors.repair_json_loads("{}", schema=_deep_schema(40)))
        large = _min_wall_ms(lambda: tors.repair_json_loads("{}", schema=_deep_schema(80)))
        _assert_linear_per_doubling(small, large, 2, LINEAR_GATE_PER_DOUBLING)

    @pytest.mark.timing
    def test_wide_schema_stays_linear_in_properties(self) -> None:
        """GREEN since the F3 fix landed (the lazy schema-side property
        index + the document-side EntryIndex in json_schema_impl.rs): the
        per-key alignment scans became lookups. Pre-fix this cell measured
        ~4-4.6x per doubling (quadratic) at 2500 -> 40000 properties;
        post-fix ~2.1-2.3x per doubling, byte-identical output."""
        """The WIDE axis: a properties map scaling with the document's
        keys (the natural schema+doc shape) at 2500 -> 5000 properties
        (2x). Currently RED — see the xfail reason; this cell is the
        pin that goes green the day the quadratic is fixed."""
        def shape(w: int) -> None:
            schema = {
                "type": "object",
                "properties": {f"k{i}": {"type": "string"} for i in range(w)},
            }
            doc = json.dumps({f"k{i}": f"v{i}" for i in range(w)})
            tors.repair_json_loads(doc, schema=schema)

        small, large = _min_wall_ms(lambda: shape(2_500)), _min_wall_ms(lambda: shape(5_000))
        _assert_linear_per_doubling(small, large, 2, LINEAR_GATE_PER_DOUBLING)


# --- replace_many: match-dense inputs ----------------------------------------------


class TestReplaceManyMatchDenseScaling:
    @pytest.mark.timing
    def test_match_dense_input_stays_linear(self) -> None:
        """Every position a match (two keys, short values — the
        time-class shape of #101's surface, unmasked; the memory class
        is the guards' job, tests/test_memory_spike_guards.py). 50k ->
        200k keys (4x): measured 3.6ms -> 14.1ms, ratio 3.9 (linear;
        ~1.98x per doubling), gate 3.0x per doubling."""
        def shape(n: int) -> str:
            return tors.replace_many("ab" * n, {"a": "x", "b": "yy"})

        small, large = _min_wall_ms(lambda: shape(50_000)), _min_wall_ms(lambda: shape(200_000))
        _assert_linear_per_doubling(small, large, 4, LINEAR_GATE_PER_DOUBLING)


# --- documents.to_text: deeply-nested HTML -----------------------------------------


class TestDocumentsDeepHtmlScaling:
    @pytest.mark.timing
    def test_deeply_nested_html_stays_linear(self) -> None:
        """500 -> 2000 nested <div>s (4x): the html engine's nesting
        handling must stay linear in depth (measured ~0.1ms -> ~0.5ms;
        the cell's floor is scheduler noise, so the generous 3.0x gate
        per doubling stands). A depth-tax relapse (per-level rescans)
        measures 4x per doubling."""
        import tors.documents as documents

        def shape(depth: int) -> str:
            body = (
                b"<html><body>" + b"<div>" * depth + b"payload text here" + b"</div>" * depth
                + b"</body></html>"
            )
            return documents.to_text(data=body, format="html")

        small, large = _min_wall_ms(lambda: shape(500), samples=5), _min_wall_ms(
            lambda: shape(2000), samples=5
        )
        _assert_linear_per_doubling(small, large, 4, LINEAR_GATE_PER_DOUBLING)

    @pytest.mark.timing
    def test_backtick_runs_stay_linear(self) -> None:
        """#111's shape: backtick runs of increasing length, each followed
        by a letter, through the documents Auto/HTML lane. GREEN since the
        near-linear closer index (`index_backtick_runs` + binary-search
        `find_equal_run`, #111's fix): per INPUT BYTE the cost is flat
        (~6 us/KB at every size — 0.25ms @ 40KB, 24.9ms @ 4MB).

        ACCOUNTING NOTE (the reason this cell's sizes look odd): the
        increasing-runs shape's input is THETA(runs^2) — the runs sum
        1+2+...+n backtick characters — so doubling the run count
        QUADRUPLES the input (two doublings, allowed 9x at the 3.0x/doubling
        gate; the pre-fix quadratic measured 16.6x for 200 -> 400 runs =
        4x input = 2 doublings, ~4.1x per input doubling). The sizes below
        are 200 -> 400 runs = 40KB -> 160KB, passed to the assert as
        factor=4 (two input doublings, allowed 9x at the gate)."""
        import tors.documents as documents

        def shape(runs: int) -> str:
            body = (
                b"<html><body><p>"
                + b"".join((b"`" * k + b"x") for k in range(1, runs))
                + b"</p></body></html>"
            )
            return documents.to_text(data=body, format="html")

        small, large = _min_wall_ms(lambda: shape(200), samples=3), _min_wall_ms(
            lambda: shape(400), samples=3
        )
        _assert_linear_per_doubling(small, large, 4, LINEAR_GATE_PER_DOUBLING)


# --- the retrieval trio at their documented ceilings -------------------------------


def _tokens(n: int) -> str:
    return " ".join(f"w{t % 997}" for t in range(n))


class TestRetrievalCeilingScaling:
    """minhash_signature's documented cost is O(tokens x shingle_size)
    under the token-hash budget (past it: ValueError — the #91 fix's own
    contract, pinned corpus-side); tf_idf/bm25_rank recompute corpus
    statistics per call, documented as linear in total corpus tokens.
    Each axis is pinned at its ceiling shape."""

    @pytest.mark.timing
    def test_minhash_tokens_axis_stays_linear(self) -> None:
        """20k -> 40k tokens at shingle_size=1000 (2x; both inside the
        2^26 token-hash budget): measured 108ms -> 217ms, ratio 2.0
        (linear), gate 3.0x per doubling. The pre-#91 shape (a fillable
        shingle re-hashing the whole window per token) grows with the
        PRODUCT of the axes, not either alone."""
        small = _min_wall_ms(
            lambda: tors.minhash_signature(_tokens(20_000), num_perm=8, shingle_size=1000)
        )
        large = _min_wall_ms(
            lambda: tors.minhash_signature(_tokens(40_000), num_perm=8, shingle_size=1000)
        )
        _assert_linear_per_doubling(small, large, 2, LINEAR_GATE_PER_DOUBLING)

    @pytest.mark.timing
    def test_minhash_shingle_axis_stays_linear(self) -> None:
        """shingle_size 1000 -> 2000 at 20k tokens (2x, both inside the
        budget): measured 107ms -> 206ms, ratio 1.9 (linear), gate 3.0x
        per doubling."""
        small = _min_wall_ms(
            lambda: tors.minhash_signature(_tokens(20_000), num_perm=8, shingle_size=1000)
        )
        large = _min_wall_ms(
            lambda: tors.minhash_signature(_tokens(20_000), num_perm=8, shingle_size=2000)
        )
        _assert_linear_per_doubling(small, large, 2, LINEAR_GATE_PER_DOUBLING)

    @pytest.mark.timing
    def test_tfidf_corpus_axis_stays_linear(self) -> None:
        """100 -> 200 documents x 500 words (2x in total tokens):
        measured 21ms -> 43ms, ratio 2.0, gate 3.0x per doubling."""
        def shape(docs: int) -> list:
            corpus = [
                " ".join(f"w{d % 50}_{t % 500}" for t in range(500)) for d in range(docs)
            ]
            return tors.tf_idf(corpus)

        small, large = _min_wall_ms(lambda: shape(100)), _min_wall_ms(lambda: shape(200))
        _assert_linear_per_doubling(small, large, 2, LINEAR_GATE_PER_DOUBLING)

    @pytest.mark.timing
    def test_bm25_corpus_axis_stays_linear(self) -> None:
        """The bm25 twin of the tf_idf pin: measured 12ms -> 26ms for
        100 -> 200 documents x 500 words, ratio 2.1, gate 3.0x per
        doubling."""
        def shape(docs: int) -> list:
            corpus = [
                " ".join(f"w{d % 50}_{t % 500}" for t in range(500)) for d in range(docs)
            ]
            return tors.bm25_rank("w0_1 w1_2", corpus)

        small, large = _min_wall_ms(lambda: shape(100)), _min_wall_ms(lambda: shape(200))
        _assert_linear_per_doubling(small, large, 2, LINEAR_GATE_PER_DOUBLING)


# --- rank_fuse: fusion at scale ------------------------------------------------------


def _fusion_lists(total_entries: int, n_lists: int = 5) -> list[list[str]]:
    """The GIL-release cell's deterministic workload (one shared
    definition, the same no-RNG idiom): n_lists ranked lists whose
    entries are REFERENCES into one shared pool of half the entries;
    distinct string objects built once and reused across lists, the
    realistic fusion shape (the same document retrieved by several
    systems) and the shape whose per-object str hashes are cached, so
    the pin measures the algorithm, not first-hash costs."""
    pool = [f"id_{i}" for i in range(total_entries // 2)]
    per_list = total_entries // n_lists
    return [
        [pool[(j * 7 + i * 3) % len(pool)] for i in range(per_list)]
        for j in range(n_lists)
    ]


class TestRankFusionScaling:
    @pytest.mark.timing
    def test_rank_fuse_stays_linear_in_total_list_length(self) -> None:
        """10k -> 40k total entries across 5 lists (4x; the output is
        bounded by the distinct-id count, here half the entries): the
        dedup walk is one dict op per entry and the detached pass is the
        score sweep plus an O(distinct log distinct) sort, so the whole
        call is linear (up to the sort's log factor) in TOTAL list
        length, the sum across lists, not the longest one, is the cost
        driver, because every entry is walked and voted. Measured
        1.26ms -> 5.72ms, ratio 4.6 (~2.15x per doubling, ambient load
        ~5-20; the box's large-dict cache-miss band sits ~2.2-2.4x per
        doubling; a pure-Python dict walk over the same shapes measures
        the same), gate 3.0x per doubling. A per-list rescan of the
        accumulated id table (the quadratic shape) would measure ~4x per
        doubling here."""
        small, large = (
            _min_wall_ms(lambda: tors.rank_fuse(_fusion_lists(10_000))),
            _min_wall_ms(lambda: tors.rank_fuse(_fusion_lists(40_000))),
        )
        _assert_linear_per_doubling(small, large, 4, LINEAR_GATE_PER_DOUBLING)

    @pytest.mark.timing
    def test_ndcg_at_k_stays_linear_in_ranking_length(self) -> None:
        """25k -> 100k ranked ids (4x; the relevant set scales with it):
        the membership walk is a constant number of set ops per id, the
        detached arithmetic tail is O(n). Measured 3.24ms -> 18.31ms,
        ratio 5.6 (~2.4x per doubling, ambient load ~5-20; the same
        large-set cache-miss band the rank_fuse cell records), gate 3.0x
        per doubling.

        Dev-box correction (this box, ambient load ~150): the cell
        measured 5.7-7.2x per 4x fresh and inflated past the shared
        3.0x-per-doubling gate under load/heap state (9.66x observed
        in-suite, twice in six full-lane runs) — the inflation is NOT
        proportional (the large cell is hit harder), so the ratio does
        not cancel and no span move fixes it (10k -> 40k measured
        10.04x under the same conditions). This cell therefore carries
        its own gate, the file's documented idiom for a band that runs
        higher: 3.5x per doubling (12.25x per 4x) sits above the loaded
        honest band (~10x worst observed) while a quadratic's 16x per
        4x still blows through."""
        def shape(n: int) -> float:
            ranked = [f"id_{i}" for i in range(n)]
            relevant = {ranked[i] for i in range(0, n, 3)}
            return tors.ndcg_at_k(ranked, relevant)

        small, large = _min_wall_ms(lambda: shape(25_000)), _min_wall_ms(lambda: shape(100_000))
        _assert_linear_per_doubling(small, large, 4, 3.5)


# --- ground_sentences / grounding_coverage: the grounding batch -------------
#
# ground_sentences' documented cost is O(sentences x rouge_w DP): the total
# DP work is |Q| x N (N = the text's tokens, capped at 16384), linear in the
# text at a bounded query width. grounding_coverage's documented cost is the
# classic O(|S| x |T|) weighted-LCS DP (its own docs): time grows with the
# PRODUCT of the operands, memory with the MINIMUM (two rows, never an n*m
# matrix), the product axis is pinned at its documented 4x-per-doubling
# band, the one-sided axis (doubling one operand only) at the linear gate.


def _ground_sentences_shape(tokens: int) -> object:
    text = ("Word. " * (tokens // 2))[:-1]
    return tors.ground_sentences(text, "word")


def _coverage_shape(tokens: int) -> float:
    return tors.grounding_coverage(("word " * tokens)[: 4 * tokens], ("word " * tokens))


class TestGroundingBatchScaling:
    @pytest.mark.timing
    def test_ground_sentences_stays_linear_in_the_text(self) -> None:
        """8k -> 16k -> 32k tokens (2x each): measured 1.4ms -> 2.7ms ->
        5.5ms, ratios ~2.0 (linear), gate 3.0x per doubling. A per-sentence
        rescan of the token stream (the quadratic shape) measures ~4x per
        doubling here."""
        small = _min_wall_ms(lambda: _ground_sentences_shape(8_000))
        large = _min_wall_ms(lambda: _ground_sentences_shape(16_000))
        _assert_linear_per_doubling(small, large, 2, LINEAR_GATE_PER_DOUBLING)

    @pytest.mark.timing
    def test_coverage_one_sided_doubling_stays_linear(self) -> None:
        """Doubling the TEXT (the candidate stream) at a fixed source:
        the DP's rows double, the width is fixed (measured ~2x, gate 3.0x
        per doubling."""
        source = "word " * 4_000
        small = _min_wall_ms(lambda: tors.grounding_coverage(source, "word " * 2_000))
        large = _min_wall_ms(lambda: tors.grounding_coverage(source, "word " * 4_000))
        _assert_linear_per_doubling(small, large, 2, LINEAR_GATE_PER_DOUBLING)

    @pytest.mark.timing
    def test_coverage_two_sided_doubling_stays_at_the_product(self) -> None:
        """Doubling BOTH operands (4x the DP cells): measured ~4x, gate
        5.0x per doubling, the documented quadratic-product time class,
        pinned so an accidental CUBIC formulation (per-cell reallocation,
        an n·m matrix) blows through."""
        small = _min_wall_ms(lambda: _coverage_shape(2_000))
        large = _min_wall_ms(lambda: _coverage_shape(4_000))
        _assert_linear_per_doubling(small, large, 2, 5.0)


# --- dedup_near_dup: the documented quadratic pair sweep --------------------------
#
# The near-dup dedup is O(n^2) pair checks BY DESIGN (docs/design.md's
# small-candidate-set scope: no LSH banding index, every call from
# scratch). The pin here holds that documented cost CLASS: the sweep may
# be quadratic, and must be nothing WORSE than quadratic, with an
# explicit absolute wall budget for the documented candidate-set size.


def _dedup_corpus(n: int) -> list:
    # Fingerprint-distinct documents (disjoint vocabulary per row) so the
    # greedy sweep accumulates kept representatives and walks the full
    # pair ladder instead of exiting every check on the first one.
    return [" ".join(f"tok{i}_{j}" for j in range(40)) for i in range(n)]


def _assert_quadratic_per_doubling(
    small_ms: float, large_ms: float, factor: int, gate: float
) -> None:
    """The quadratic-class gate: cost may grow up to `gate` per doubling
    (4.0 is the quadratic bound itself; the gate sits above it for
    measurement noise, while any cubic shape blows through)."""
    import math

    doublings = math.log2(factor)
    allowed = gate**doublings
    assert large_ms < allowed * small_ms, (
        f"cost grew {small_ms:.2f}ms -> {large_ms:.2f}ms for a {factor}x input "
        f"({large_ms / small_ms:.2f}x, allowed {allowed:.1f}x at {gate:.1f}x per "
        "doubling): the sweep grew worse than its documented quadratic class"
    )


class TestDedupNearDupPairSweepScaling:
    @pytest.mark.timing
    def test_pair_sweep_stays_within_the_quadratic_class(self) -> None:
        """1k -> 4k documents (4x, fingerprint-distinct, simhash method):
        measured 14ms -> 74ms, ratio 5.3 (the linear fingerprint pass
        dilutes the quadratic sweep at these sizes; 4x input means 16x
        pair checks at full quadratic), gate 4.5x per doubling -- the
        quadratic bound 4.0 plus measurement margin, far under any
        cubic's 64x."""
        small, large = _min_wall_ms(lambda: tors.dedup_near_dup(_dedup_corpus(1_000))), (
            _min_wall_ms(lambda: tors.dedup_near_dup(_dedup_corpus(4_000)))
        )
        _assert_quadratic_per_doubling(small, large, 4, 4.5)

    @pytest.mark.timing
    def test_documented_candidate_set_has_an_explicit_wall_budget(self) -> None:
        """The budget pin, absolute, not relative: the documented
        small-candidate-set shape (1k documents, the size the API docs
        name as the comfortable ceiling) must complete in well under a
        second for the default method -- measured 14ms (simhash),
        273ms (the shingle method's set intersections), 44ms (minhash).
        2.0s is ~7x the worst measured method and still an honest
        'sub-second-scale call' contract; a regression past it is a
        defect, not noise."""
        corpus = _dedup_corpus(1_000)
        assert _min_wall_ms(lambda: tors.dedup_near_dup(corpus, method="simhash")) < 2_000.0
        assert _min_wall_ms(lambda: tors.dedup_near_dup(corpus, method="shingle")) < 2_000.0
        assert _min_wall_ms(lambda: tors.dedup_near_dup(corpus, method="minhash")) < 2_000.0
