"""Contract gate for ``tors.grounding_report``: the production grounding-
report composition, the pipeline shape the Deepchecks "Grounded in
Context" framework makes (Gerner et al. 2025, "Grounded in Context:
Retrieval-Based Method for Hallucination Detection", arXiv 2504.15771:
decompose the output into statements, score each statement against the
context, aggregate into one verdict), with tors's lexical layer in place
of the paper's NLI entailment model (the named handoff).

The differential oracle is the composition of the published primitives
themselves (``tors.sentence_bounds`` + ``tors.highlight`` +
``tors.grounding_coverage`` + ``tors.ground_sentences``), which is
exactly what the Rust core drives under ONE py.detach: the reference
below is the Python spelling of the same composition, and the gates pin
the two to exact equality (same primitives, same order, identical
floats). Span round-trips (``text[start:end] == sentence["text"]``),
degenerate-shape pins, idempotence, heartbeat cells at document scale
(tests/loop_harness.py), scaling pins, and hypothesis properties
(spans in-range and sorted; the aggregate's ratios in [0, 1]) complete
the file.
"""

from __future__ import annotations

import math
from time import monotonic

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import tors
import tors.aio
from loop_harness import assert_heartbeat_clean, min_wall_ms

# Arbitrary Unicode for the offset round-trip property: the property must
# hold for ANY text the pipeline can see (surrogates excluded, never
# valid in decoded str anyway).
_ANY_TEXT = st.text(max_size=300)
_ANY_SOURCES = st.lists(st.text(max_size=60), max_size=3)
_ANY_QUERY = st.none() | st.text(max_size=40)


def reference_report(
    text: str, sources: list[str], query: str | None = None, threshold: float = 0.85
) -> dict:
    """The naive composition over the published primitives, the report
    contract's Python spelling (see the module docstring)."""

    sentences = []
    for start, end in tors.sentence_bounds(text):
        claim = text[start:end]
        best_score, best_source = 0.0, None
        for index, source in enumerate(sources):
            score = tors.highlight(claim, source, max_snippets=1, max_chars=400)["score"]
            if score > best_score:
                best_score, best_source = score, index
        grounded = best_source is not None and best_score >= threshold
        sentences.append(
            {
                "text": claim,
                "start": start,
                "end": end,
                "best_source": best_source,
                "score": best_score,
                "grounded": grounded,
            }
        )
    n = len(sentences)
    grounded = sum(1 for entry in sentences if entry["grounded"])
    scores = [entry["score"] for entry in sentences]
    coverage = 0.0
    if sources and text:
        coverage = tors.grounding_coverage(text, "\n".join(sources))
    query_score = 0.0
    if query:
        query_score = tors.ground_sentences(text, query)["score"]
    return {
        "sentences": sentences,
        "aggregate": {
            "grounded_ratio": grounded / n if n else 0.0,
            "grounded": grounded,
            "sentences": n,
            "mean_score": sum(scores) / n if n else 0.0,
            "best_score": max(scores, default=0.0),
            "coverage": coverage,
            "query_score": query_score,
        },
    }


class TestOracleDifferential:
    """The Rust composition against the Python spelling of the same
    primitive composition: exact equality, floats included."""

    def test_the_hand_case_matches_exactly(self) -> None:
        text = "The pump failed. The bushing torque spec was 42 Nm. Replaced."
        sources = ["Service log: the bushing torque spec was 42 Nm."]
        for kwargs in (
            {},
            {"query": "torque"},
            {"threshold": 0.3},
            {"query": "pump", "threshold": 0.1},
        ):
            assert tors.grounding_report(text, sources, **kwargs) == reference_report(
                text, sources, **kwargs
            )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        text=st.text(min_size=1, max_size=150),
        sources=_ANY_SOURCES,
        query=_ANY_QUERY,
        threshold=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_matches_under_arbitrary_unicode(
        self, text: str, sources: list[str], query: str | None, threshold: float
    ) -> None:
        assert tors.grounding_report(text, sources, query, threshold=threshold) == reference_report(
            text, sources, query, threshold
        )

    def test_it_agrees_with_ground_sentences_per_sentence_scores(self) -> None:
        # The per-sentence spans/offsets are ground_sentences' own (the
        # shared segmentation), and a sentence whose tokens the source
        # carries grounds once the threshold reaches its alignment score
        # (0.714 here, measured; the exact per-pair score is pinned by
        # the oracle differential above).
        text = "The quick brown fox jumps. Something entirely different here."
        source = "the quick brown fox jumps over the lazy dog"
        report = tors.grounding_report(text, [source], threshold=0.7)
        batch = tors.ground_sentences(text, "irrelevant anchor")
        assert [s["start"] for s in report["sentences"]] == [s["start"] for s in batch["sentences"]]
        assert [s["end"] for s in report["sentences"]] == [s["end"] for s in batch["sentences"]]
        assert report["sentences"][0]["grounded"] is True
        assert report["sentences"][1]["grounded"] is False


class TestSpanRoundTrip:
    """THE load-bearing property of every span-returning surface here:
    slicing the original by the returned offsets yields exactly the
    sentence text."""

    @settings(max_examples=200)
    @given(text=_ANY_TEXT, sources=_ANY_SOURCES, query=_ANY_QUERY)
    def test_offsets_slice_the_original_text(
        self, text: str, sources: list[str], query: str | None
    ) -> None:
        report = tors.grounding_report(text, sources, query)
        for entry in report["sentences"]:
            start, end = entry["start"], entry["end"]
            assert 0 <= start < end <= len(text)
            assert text[start:end] == entry["text"]

    def test_every_script_case_round_trips(self) -> None:
        # The boundary battery test_ground_sentences.py pins, reused:
        # CJK (unspaced), NFC and NFD accents, ZWJ emoji, RTL, astral.
        cases = [
            "検索対象の文書には重要な情報が含まれています。次の文もある。",
            "café au lait and the café again. Second sentence.",
            "cafe\u0301 au lait (NFD form). Second sentence.",
            "prefix 👨‍👩‍👧‍👦 family sentence. Suffix sentence.",
            "المادة رقم ٥ من القانون تنص. وتابع.",
            "𝕌𝕟𝕚𝕔𝕠𝕕𝕖 target 𝕒𝕝𝕚𝕘𝕟𝕞𝕖𝕟𝕥. Second sentence.",
            "emoji 🎉🎉🎉 then the term. Trailing sentence.",
        ]
        for text in cases:
            report = tors.grounding_report(text, ["café 検索 family term"])
            for entry in report["sentences"]:
                assert text[entry["start"] : entry["end"]] == entry["text"]

    @settings(max_examples=150)
    @given(text=_ANY_TEXT, sources=_ANY_SOURCES)
    def test_spans_are_sorted_and_disjoint(self, text: str, sources: list[str]) -> None:
        report = tors.grounding_report(text, sources)
        starts = [entry["start"] for entry in report["sentences"]]
        ends = [entry["end"] for entry in report["sentences"]]
        assert starts == sorted(starts)
        assert all(a < b for a, b in zip(starts, starts[1:], strict=False))
        assert all(a <= b for a, b in zip(ends, ends[1:], strict=False))
        assert all(s < e for s, e in zip(starts, ends, strict=False))


class TestVerdict:
    def test_verbatim_claims_ground_at_the_documented_default(self) -> None:
        text = "The bushing torque spec was 42 Nm."
        # The source carries the claim nearly whole (one prefix token of
        # slack): the alignment scores 0.933, over the 0.85 default.
        source = "log: The bushing torque spec was 42 Nm."
        report = tors.grounding_report(text, [source])
        assert report["sentences"][0]["grounded"] is True
        assert report["sentences"][0]["best_source"] == 0
        assert report["aggregate"]["grounded_ratio"] == 1.0

    def test_unsupported_sentences_do_not_ground(self) -> None:
        text = "The moon is made of green cheese. The sky is blue today."
        source = "The sky is blue today, meteorologists said."
        # The supported sentence's alignment scores 0.833 (measured): the
        # cell runs at 0.8 so the split verdict is the point.
        report = tors.grounding_report(text, [source], threshold=0.8)
        assert report["sentences"][0]["grounded"] is False
        assert report["sentences"][1]["grounded"] is True
        assert 0.0 < report["aggregate"]["grounded_ratio"] < 1.0

    def test_the_threshold_is_parameterized(self) -> None:
        text = "The bushing torque spec was 42 Nm exactly."
        source = "the torque spec was about 42"
        report = tors.grounding_report(text, [source])
        # The alignment scores ~0.61 (measured 0.6126): either side of it
        # the threshold flips the verdict, the parameterization's teeth.
        assert report["sentences"][0]["score"] == pytest.approx(0.6126, abs=1e-3)
        strict = tors.grounding_report(text, [source], threshold=0.7)
        assert strict["sentences"][0]["grounded"] is False
        loose = tors.grounding_report(text, [source], threshold=0.5)
        assert loose["sentences"][0]["grounded"] is True
        # threshold=0.0 still refuses the no-alignment shape: grounded
        # requires an alignment to exist.
        no_source = tors.grounding_report(text, [], threshold=0.0)
        assert no_source["sentences"][0]["grounded"] is False

    def test_the_best_source_is_the_argmax_with_earliest_ties(self) -> None:
        text = "The bushing torque spec was 42 Nm."
        weaker = "unrelated maintenance notes about filters"
        stronger = "log: the bushing torque spec was 42 Nm confirmed"
        report = tors.grounding_report(text, [weaker, stronger])
        assert report["sentences"][0]["best_source"] == 1
        twin = tors.grounding_report(text, [stronger, stronger + " (copy)"])
        assert twin["sentences"][0]["best_source"] == 0


class TestDegeneratePins:
    def test_empty_text_is_the_all_empty_shape(self) -> None:
        report = tors.grounding_report("", ["a source"])
        assert report == {
            "sentences": [],
            "aggregate": {
                "grounded_ratio": 0.0,
                "grounded": 0,
                "sentences": 0,
                "mean_score": 0.0,
                "best_score": 0.0,
                "coverage": 0.0,
                "query_score": 0.0,
            },
        }

    def test_no_sources_grounds_nothing(self) -> None:
        text = "The pump failed. Replaced."
        report = tors.grounding_report(text, [])
        assert len(report["sentences"]) == 2
        for entry in report["sentences"]:
            assert entry["best_source"] is None
            assert entry["score"] == 0.0
            assert entry["grounded"] is False
        assert report["aggregate"]["grounded_ratio"] == 0.0
        assert report["aggregate"]["coverage"] == 0.0
        # Even at threshold 0.0: no sources, no alignment, no grounding.
        assert tors.grounding_report(text, [], threshold=0.0)["sentences"][0]["grounded"] is False

    def test_no_query_leaves_the_query_lens_at_zero(self) -> None:
        report = tors.grounding_report("One sentence.", ["One sentence."])
        assert report["aggregate"]["query_score"] == 0.0
        with_query = tors.grounding_report("One sentence.", ["One sentence."], "sentence")
        assert with_query["aggregate"]["query_score"] > 0.0

    def test_out_of_range_threshold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            tors.grounding_report("text.", ["source"], threshold=1.5)
        with pytest.raises(ValueError, match="threshold"):
            tors.grounding_report("text.", ["source"], threshold=-0.1)


class TestIdempotence:
    def test_repeat_calls_are_identical(self) -> None:
        text = "The pump failed. The bushing torque spec was 42 Nm. Replaced."
        sources = ["Service log: the bushing torque spec was 42 Nm."]
        first = tors.grounding_report(text, sources, query="torque")
        for _ in range(3):
            assert tors.grounding_report(text, sources, query="torque") == first


class TestProperties:
    @settings(max_examples=150)
    @given(text=_ANY_TEXT, sources=_ANY_SOURCES, query=_ANY_QUERY)
    def test_the_aggregate_stays_in_range(
        self, text: str, sources: list[str], query: str | None
    ) -> None:
        report = tors.grounding_report(text, sources, query)
        agg = report["aggregate"]
        assert set(agg) == {
            "grounded_ratio",
            "grounded",
            "sentences",
            "mean_score",
            "best_score",
            "coverage",
            "query_score",
        }
        assert 0.0 <= agg["grounded_ratio"] <= 1.0
        assert 0.0 <= agg["mean_score"] <= 1.0
        assert 0.0 <= agg["best_score"] <= 1.0
        assert 0.0 <= agg["coverage"] <= 1.0
        assert 0.0 <= agg["query_score"] <= 1.0
        assert agg["sentences"] == len(report["sentences"])
        assert 0 <= agg["grounded"] <= agg["sentences"]
        assert agg["grounded_ratio"] == (
            agg["grounded"] / agg["sentences"] if agg["sentences"] else 0.0
        )

    @settings(max_examples=150)
    @given(text=_ANY_TEXT, sources=_ANY_SOURCES)
    def test_best_source_is_in_range_or_none(self, text: str, sources: list[str]) -> None:
        report = tors.grounding_report(text, sources)
        for entry in report["sentences"]:
            if entry["best_source"] is None:
                assert entry["score"] == 0.0
                assert entry["grounded"] is False
            else:
                assert 0 <= entry["best_source"] < len(sources)
                assert entry["score"] >= 0.0


class TestGilHeartbeat:
    """The composition is ONE detached pass at document scale: a 10ms
    heartbeat keeps ticking with worst gaps under the shared budgets
    while the whole report runs (tests/loop_harness.py's discipline)."""

    def test_heartbeat_stays_clean_at_document_scale(self) -> None:
        sentence = "The quarterly oil sample interval was adjusted after the review. "
        text = sentence * 120  # ~9.5 KB, ~120 sentences
        sources = [sentence * 40, "Unrelated maintenance notes. " * 40]
        # The awaitable spelling IS the heartbeat cell's op: the sync
        # call blocks its own coroutine's turn on the loop by
        # definition (loop_harness's awaitable-factory shape), and this
        # is exactly the input scale the aio twin exists for.
        assert_heartbeat_clean(
            lambda: tors.aio.grounding_report(text, sources, query="oil sample"),
            subject="grounding_report at document scale",
        )


class TestScaling:
    @pytest.mark.timing
    def test_linear_in_sentences_times_sources(self) -> None:
        """The report's work is the (sentence, source) pair count:
        doubling sentences AND doubling sources (4x pairs) stays under
        the 3.0x-per-doubling gate's 2-doubling allowance (9x)."""
        sentence = "The bushing torque spec was 42 Nm and the pump failed. "

        def run(sentences: int, sources: int) -> dict:
            text = sentence * sentences
            srcs = [sentence * 5] * sources
            return tors.grounding_report(text, srcs)

        small_ms = min_wall_ms(run, 25, 1, samples=5)
        large_ms = min_wall_ms(run, 50, 2, samples=5)
        # 4x the pairs is 2 doublings of work: allowed 3^2 = 9x.
        assert large_ms < 9.0 * small_ms, (
            f"{small_ms:.2f}ms -> {large_ms:.2f}ms for 4x the (sentence, source) pairs: "
            "grounding_report grew superlinear in sentences x sources"
        )


def test_heartbeat_smoke_is_sane() -> None:
    # A cheap sanity cell (not a timing gate): the document-scale wall is
    # finite and the report is well-formed at that scale.
    sentence = "The quarterly oil sample interval was adjusted after the review. "
    text = sentence * 120
    started = monotonic()
    report = tors.grounding_report(text, [sentence * 40])
    wall = monotonic() - started
    assert math.isfinite(wall)
    assert report["aggregate"]["sentences"] > 0
