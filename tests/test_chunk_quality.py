"""Contract gate for ``tors.chunk_quality``: the two intrinsic
chunk-quality metrics of the LREC 2026 adaptive-chunking study (Madan et
al. 2026, "Adaptive Chunking", arXiv 2603.25333) over the caller's own
chunk spans: Block Integrity (the fraction of the text's UAX #29
sentences no chunk boundary crosses, within the tolerance ``tau``) and
Intra-Chunk Cohesion (the mean within-chunk sentence-to-chunk Dice
similarity over width-3 word shingles, tors's dependency-free LEXICAL
PROXY for the study's embedding-based ICC).

The differential oracle is the published-primitive composition itself:
the reference implementation below rebuilds both metrics from
``tors.sentence_bounds`` and ``tors.shingle_dice`` (the exact quantity
the cohesion proxy rides), so the gate pins the core to the surfaces it
is documented to share primitives with. Known-answer vectors,
degenerate-shape pins, idempotence, and hypothesis properties round the
gate out; the honest lexical-proxy caveat lives in docs/api.md and
src/chunk_quality_impl.rs.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import tors
from loop_harness import first_clean, min_wall_ms

# Arbitrary Unicode for the span/structure properties: any text the
# chunk family can emit (surrogates excluded, never valid in decoded
# str anyway).
_ANY_TEXT = st.text(min_size=1, max_size=200)


def reference_chunk_quality(
    chunks: list[tuple[int, int]], text: str, tau: int = 0
) -> dict[str, float]:
    """The naive composition, over the published primitives only:
    ``tors.sentence_bounds`` for the gold blocks, ``tors.shingle_dice``
    (width 3, the shingle family's default) for the cohesion proxy."""

    def dice(a: str, b: str) -> float:
        return tors.shingle_dice(a, b, width=3)

    spans = tors.sentence_bounds(text)
    n = len(spans)
    if n == 0:
        integrity = 1.0
    else:
        crossed = 0
        for s, e in spans:
            hit = False
            for cs, ce in chunks:
                for p in (cs, ce):
                    if s < p < e and p - s > tau and e - p > tau:
                        hit = True
            crossed += hit
        integrity = 1.0 - crossed / n
    total = 0.0
    count = 0
    for cs, ce in chunks:
        chunk_text = text[cs:ce]
        for s, e in spans:
            if cs <= s and e <= ce:
                total += dice(text[s:e], chunk_text)
                count += 1
    cohesion = total / count if count else 0.0
    return {"integrity": integrity, "cohesion": cohesion}


class TestKnownAnswers:
    """The hand-computed vectors (mirrored in the Rust core's unit
    tests; pinned here against the binding too)."""

    def test_sentence_sized_chunks_score_perfect(self) -> None:
        text = "One two three. Four five six. Seven eight nine."
        chunks = tors.chunk_by_sentences(text, 1)
        assert tors.chunk_quality(chunks, text) == {"integrity": 1.0, "cohesion": 1.0}

    def test_one_big_chunk_is_the_documented_hand_vector(self) -> None:
        # Each sentence's shingle set rides inside the chunk's: the
        # tokenizer keeps punctuation segments, so each sentence is 4
        # tokens -> 2 width-3 shingles, the chunk 12 tokens -> 10, Dice
        # 2*2/(2+10) = 1/3 per sentence.
        text = "One two three. Four five six. Seven eight nine."
        out = tors.chunk_quality([(0, len(text))], text)
        assert out["integrity"] == 1.0
        assert out["cohesion"] == pytest.approx(1.0 / 3.0, abs=1e-12)

    def test_mid_sentence_cuts_damage_integrity(self) -> None:
        text = "One two three. Four five six."
        # Sentence spans: (0, 15) and (15, 29); a cut at 4 is strictly
        # inside the first.
        assert tors.chunk_quality([(0, 4), (4, len(text))], text)["integrity"] == 0.5
        # A cut exactly at the sentence edge is not a crossing.
        assert tors.chunk_quality([(0, 15), (15, len(text))], text)["integrity"] == 1.0

    def test_tau_forgives_edge_adjacent_cuts(self) -> None:
        text = "One two three. Four five six."
        # A cut 1 codepoint past the second sentence's start edge.
        assert tors.chunk_quality([(0, 16), (16, len(text))], text, tau=0)["integrity"] == 0.5
        assert tors.chunk_quality([(0, 16), (16, len(text))], text, tau=1)["integrity"] == 1.0


class TestDegeneratePins:
    def test_empty_chunk_list(self) -> None:
        out = tors.chunk_quality([], "One two three. Four five six.")
        assert out == {"integrity": 1.0, "cohesion": 0.0}

    def test_empty_text(self) -> None:
        assert tors.chunk_quality([(0, 0)], "") == {"integrity": 1.0, "cohesion": 0.0}

    def test_token_free_text_follows_the_shingle_conventions(self) -> None:
        # One whitespace-run sentence, contained, both shingle sets
        # empty: the shingle family's two-empty-sets convention (1.0).
        assert tors.chunk_quality([(0, 4)], "    ") == {"integrity": 1.0, "cohesion": 1.0}

    def test_bad_spans_raise_value_error(self) -> None:
        text = "One two three."
        with pytest.raises(ValueError, match="0 <= start <= end"):
            tors.chunk_quality([(-1, 5)], text)
        with pytest.raises(ValueError, match="0 <= start <= end"):
            tors.chunk_quality([(5, 2)], text)
        with pytest.raises(ValueError, match="0 <= start <= end"):
            tors.chunk_quality([(0, 999)], text)

    def test_bad_tau_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="tau"):
            tors.chunk_quality([(0, 5)], "One two three.", tau=-1)

    def test_a_bare_str_chunks_argument_is_refused(self) -> None:
        # The bounded walk's refusal of the char-split footgun.
        with pytest.raises(TypeError):
            tors.chunk_quality("ab", "One two three.")


class TestIdempotence:
    def test_repeat_calls_are_identical(self) -> None:
        text = "The pump failed. The bushing torque spec was 42 Nm. Replaced."
        chunks = tors.chunk_text(text, 30)
        first = tors.chunk_quality(chunks, text)
        for _ in range(3):
            assert tors.chunk_quality(chunks, text) == first
        # And the same spans under the same tau from a fresh list too.
        assert tors.chunk_quality(list(chunks), text, tau=2) == tors.chunk_quality(
            list(chunks), text, tau=2
        )


class TestOracleDifferential:
    @settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
    @given(
        text=st.text(min_size=1, max_size=120),
        seed=st.integers(0, 2**32),
        tau=st.integers(0, 4),
    )
    def test_matches_the_published_primitive_composition(
        self, text: str, seed: int, tau: int
    ) -> None:
        # Spans from the family itself (always in range by contract),
        # plus a deterministic pseudo-random jitter so the reference sees
        # crossed sentences and edge-adjacent cuts, not only clean cuts.
        chunks = [span for span in tors.chunk_text(text, max(1, len(text) // 2 + 1))]
        rng = seed
        jittered: list[tuple[int, int]] = []
        for s, e in chunks:
            rng = (rng * 6364136223846793005 + 1442695040888963407) % 2**64
            delta = (rng % 5) - 2
            start = max(0, min(len(text), s + delta))
            end = max(start, min(len(text), e + delta))
            jittered.append((start, end))
        assert tors.chunk_quality(jittered, text, tau=tau) == reference_chunk_quality(
            jittered, text, tau
        )

    def test_the_hand_case_matches_too(self) -> None:
        text = "The pump failed. The bushing torque spec was 42 Nm. Replaced."
        for chunks in ([(0, 30), (30, 61)], [(0, 25), (25, 61)], tors.chunk_by_sentences(text, 1)):
            for tau in (0, 1, 3):
                assert tors.chunk_quality(chunks, text, tau=tau) == reference_chunk_quality(
                    chunks, text, tau
                )


class TestProperties:
    @settings(max_examples=200)
    @given(
        text=st.text(min_size=1, max_size=200),
        max_chars=st.integers(8, 60),
        tau=st.integers(0, 6),
    )
    def test_metrics_stay_in_the_unit_interval(self, text: str, max_chars: int, tau: int) -> None:
        out = tors.chunk_quality(tors.chunk_text(text, max_chars), text, tau=tau)
        assert 0.0 <= out["integrity"] <= 1.0
        assert 0.0 <= out["cohesion"] <= 1.0
        assert set(out) == {"integrity", "cohesion"}

    @settings(max_examples=100)
    @given(text=_ANY_TEXT)
    def test_sentence_aligned_chunks_never_lose_integrity_to_overlap(self, text: str) -> None:
        # chunk_by_sentences' spans (one sentence per chunk) cross
        # nothing, however exotic the text, when the segmentation is the
        # same one integrity scores against.
        chunks = tors.chunk_by_sentences(text, 1)
        assert tors.chunk_quality(chunks, text)["integrity"] == 1.0


class TestScaling:
    @pytest.mark.timing
    def test_chunk_quality_stays_linear_in_the_text(self) -> None:
        """The whole pass is one segmentation + one shingle pass per
        chunk and sentence: doubling the text at most ~doubles the wall
        (measured ~1.4x per doubling at these sizes), gate 3.0x per
        doubling (test_scaling_pins.py's gate)."""
        sentence = "The quarterly oil sample interval was adjusted after the review. "
        small = sentence * 100
        large = sentence * 400

        def run(text: str) -> dict[str, float]:
            chunks = tors.chunk_text(text, 300)
            return tors.chunk_quality(chunks, text)

        small_ms = min_wall_ms(run, small, samples=5)
        large_ms = min_wall_ms(run, large, samples=5)
        factor = len(large) / len(small)
        assert large_ms < (3.0 ** math.log2(factor)) * small_ms, (
            f"{small_ms:.2f}ms -> {large_ms:.2f}ms for a {factor:.0f}x text: "
            "chunk_quality grew superlinear in the text"
        )


def test_the_loop_harness_smoke() -> None:
    # The scaling cell's first_clean discipline, smoke-checked so the
    # import stays honest (tests/test_loop_harness.py owns the red side).
    first_clean(lambda: tors.chunk_quality([(0, 5)], "hello"), lambda out: None, samples=1)
