"""The highlight wall lane: pin the grounding pass's cost shape with
machine-speed-immune ratios (never absolute times), each cell asserting its
verdict in-cell so a perf gate can never silently diverge from the
correctness contract. The absolute numbers live in ``benches/grounding.rs``
(criterion; CI compiles that, it does not run it — this lane is the wall
gate CI runs).

The shapes, from src/grounding_impl.rs's contract:

- linear-in-chunk: the anchor walk and tokenization are linear in the
  chunk, and the DP is capped (64 candidates x query-bounded widths), so
  doubling the chunk at most ~triples the wall — a superlinear selection
  step would blow past a 4x slack on doubling.
- cjk-vs-latin: the same nominal token count in unspaced CJK (per-character
  tokens, ~3x the UTF-8 bytes) must not cost more than a small constant of
  the Latin wall: the per-character sub-split walks grapheme clusters, but
  the DP shapes are identical.
- no-overlap floor: zero anchors skips the DP entirely; its wall is the
  tokenization + anchor walk floor, bounded by the same linear gate.
- realistic budget: a 60-token query against a 2k-token chunk (the
  primary consumer's per-hit shape) must complete inside a wall an async
  service can thread-hop without thinking about it — pinned as a generous
  absolute ceiling (100 ms) precisely because the ratios above carry the
  regression sensitivity; this cell only catches a catastrophic qualitative
  break (e.g. an accidental whole-text DP).
"""

from __future__ import annotations

from time import monotonic

import pytest

import tors

pytestmark = pytest.mark.timing

_LATIN_SENTENCE = (
    "The quarterly oil sample interval for field outages was adjusted after the "
    "bushing torque specifications changed. "
)
_CJK_UNIT = "変電所の絶縁油検査は四半期ごとに行われる。"
_QUERY = (
    "the quarterly oil sample interval for field outages was adjusted after the "
    "bushing torque specifications changed maintenance windows now close within "
    "fourteen days of each outage review cycle and the transformer oil analysis "
    "report lists dielectric strength moisture content and dissolved gas "
    "concentrations for every sampled unit"
)
_SAMPLES = 5


def _latin_chunk(tokens: int) -> str:
    parts: list[str] = []
    count = 0
    while sum(len(p.split()) for p in parts) < tokens:
        parts.append(_LATIN_SENTENCE if count % 3 == 0 else f"Filler sentence {count} walks on.")
        count += 1
    return " ".join(parts)


def _cjk_chunk(tokens: int) -> str:
    parts: list[str] = []
    count = 0
    while sum(len(p) for p in parts) < tokens:
        parts.append(_CJK_UNIT if count % 3 == 0 else "その他の無関係な記述がここに入る。")
        count += 1
    return "".join(parts)


def _wall(query: str, text: str) -> float:
    best = float("inf")
    for _ in range(_SAMPLES):
        start = monotonic()
        tors.highlight(query, text, max_snippets=3, max_chars=400)
        best = min(best, monotonic() - start)
    return best


class TestHighlightWall:
    def test_full_scan_stays_linear_in_the_chunk(self) -> None:
        walls = {
            tokens: _wall(_QUERY, _latin_chunk(tokens))
            for tokens in (500, 1_000, 2_000, 4_000, 10_000)
        }
        for small, large in ((500, 1_000), (1_000, 2_000), (2_000, 4_000), (4_000, 10_000)):
            ratio = walls[large] / max(walls[small], 1e-9)
            growth = large / small
            assert ratio < growth * 1.6, (
                f"{small}->{large} tokens: wall grew {ratio:.2f}x for {growth:.0f}x the "
                f"chunk ({walls}): a superlinear step leaked past the caps"
            )

    def test_cjk_costs_a_small_constant_of_latin_at_equal_token_counts(self) -> None:
        latin = _wall(_QUERY, _latin_chunk(2_000))
        cjk = _wall(_QUERY, _cjk_chunk(2_000))
        assert cjk < latin * 3.0, f"cjk {cjk * 1e3:.1f}ms vs latin {latin * 1e3:.1f}ms"

    def test_the_no_overlap_floor_stays_linear(self) -> None:
        small = _wall("zebra quantum xylophone", _latin_chunk(2_000))
        large = _wall("zebra quantum xylophone", _latin_chunk(10_000))
        assert large / max(small, 1e-9) < 10_000 / 2_000 * 1.6, (
            f"no-overlap floor grew {large / small:.2f}x for 5x the chunk"
        )

    def test_the_realistic_per_hit_shape_completes_inside_the_thread_hop_budget(self) -> None:
        # 60-token query x 2k-token chunk: the consumer's per-hit shape.
        # Generous absolute ceiling on purpose - the linear and cjk gates
        # above carry the regression sensitivity; this one only catches a
        # qualitative break (an accidental whole-text DP would blow past it).
        wall = _wall(_QUERY, _latin_chunk(2_000))
        assert wall < 0.1, f"2k-token chunk took {wall * 1e3:.1f}ms"


def _batch_wall(query: str, text: str) -> float:
    best = float("inf")
    for _ in range(_SAMPLES):
        start = monotonic()
        tors.ground_sentences(text, query)
        best = min(best, monotonic() - start)
    return best


class TestGroundSentencesWall:
    """The batch twin of the highlight lane: ground_sentences' documented
    cost is O(sentences x rouge_w DP): the total DP work is |Q| x N (N =
    the text's tokens, capped at 16384), linear in the text at a bounded
    query width, one reused scratch never wider than the longest sentence
    (src/grounding_impl.rs). Same machine-speed-immune ratio gates, same
    in-cell verdicts."""

    def test_full_batch_stays_linear_in_the_chunk(self) -> None:
        walls = {
            tokens: _batch_wall(_QUERY, _latin_chunk(tokens))
            for tokens in (500, 1_000, 2_000, 4_000, 10_000)
        }
        for small, large in ((500, 1_000), (1_000, 2_000), (2_000, 4_000), (4_000, 10_000)):
            ratio = walls[large] / max(walls[small], 1e-9)
            growth = large / small
            assert ratio < growth * 1.6, (
                f"{small}->{large} tokens: batch wall grew {ratio:.2f}x for "
                f"{growth:.0f}x the chunk ({walls}): a superlinear step leaked "
                "past the caps"
            )

    def test_batch_cjk_costs_a_small_constant_of_latin_at_equal_token_counts(self) -> None:
        latin = _batch_wall(_QUERY, _latin_chunk(2_000))
        cjk = _batch_wall(_QUERY, _cjk_chunk(2_000))
        assert cjk < latin * 3.0, f"cjk {cjk * 1e3:.1f}ms vs latin {latin * 1e3:.1f}ms"

    def test_the_batch_realistic_shape_completes_inside_the_thread_hop_budget(self) -> None:
        # 60-token query x 2k-token chunk, every sentence scored: the
        # consumer's per-document shape (the NLI bridge's own input).
        # Generous absolute ceiling on purpose; the linear gate above
        # carries the regression sensitivity; this catches a qualitative
        # break (an accidental whole-text DP or per-sentence reallocation).
        wall = _batch_wall(_QUERY, _latin_chunk(2_000))
        assert wall < 0.5, f"2k-token chunk batch took {wall * 1e3:.1f}ms"

    def test_the_batch_no_overlap_floor_stays_linear(self) -> None:
        small = _batch_wall("zebra quantum xylophone", _latin_chunk(2_000))
        large = _batch_wall("zebra quantum xylophone", _latin_chunk(10_000))
        assert large / max(small, 1e-9) < 10_000 / 2_000 * 1.6, (
            f"batch no-overlap floor grew {large / small:.2f}x for 5x the chunk"
        )
