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
    def test_deep_schema_walk_stays_linear_in_depth(self) -> None:
        """40 -> 80 nested levels (2x, both under the documented 200-unit
        walk cap; past it the walk raises ValueError — that contract is
        pinned corpus-side): measured 0.15ms -> 0.32ms, ratio 2.1
        (linear), gate 3.0x per doubling. A per-level rescan of the
        accumulated schema path (the quadratic shape) measures 4x."""
        small = _min_wall_ms(lambda: tors.repair_json_loads("{}", schema=_deep_schema(40)))
        large = _min_wall_ms(lambda: tors.repair_json_loads("{}", schema=_deep_schema(80)))
        _assert_linear_per_doubling(small, large, 2, LINEAR_GATE_PER_DOUBLING)

    @pytest.mark.xfail(
        reason=(
            "LIVE DEFECT (this harness's finding, #113's cost class): the schema-aware "
            "repair is superlinear in the joint property x document-key count. Measured "
            "on this tree, min-of-5: w=2500 -> 15.7ms, w=5000 -> 57.2ms (3.7x), "
            "w=10000 -> 251.3ms (4.4x), w=20000 -> 686.6ms, w=40000 -> 3140ms — "
            "~4-4.6x per doubling (quadratic) when document keys match schema "
            "properties, and ~2.5-2.7x per doubling even with an empty document. "
            "Flip to a green pin when the fix lands (expected: <3.0x per doubling; "
            "the sizes here are deliberately small so the red cell costs ~70ms)."
        ),
        strict=False,
    )
    @pytest.mark.timing
    def test_wide_schema_stays_linear_in_properties(self) -> None:
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

    @pytest.mark.xfail(
        reason=(
            "LIVE DEFECT #111: the gfm code-span lifter (find_equal_run) rescans to "
            "end-of-line per backtick opener, so backtick runs of increasing length "
            "are quadratic — measured on this tree at 200 -> 400 -> 800 runs: "
            "37.5ms -> 622.7ms (16.6x) -> 9664.9ms (15.5x) per doubling, reachable "
            "from documents.to_text on any HTML/PDF/office payload that emits "
            "backticks. Flip to a green pin when the fix lands (expected: <3.0x per "
            "doubling; sizes kept small so the red cell costs ~0.7s)."
        ),
        strict=False,
    )
    @pytest.mark.timing
    def test_backtick_runs_stay_linear(self) -> None:
        """#111's shape: backtick runs of increasing length, each followed
        by a letter, through the documents Auto/HTML lane. Currently RED
        — see the xfail reason; this cell is the pin that goes green the
        day the code-span quadratic is fixed."""
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
        _assert_linear_per_doubling(small, large, 2, LINEAR_GATE_PER_DOUBLING)


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
