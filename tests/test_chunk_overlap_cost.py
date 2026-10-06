"""Contract gate for ``tors.chunk_overlap_cost``: the chunk family's
overlap knob, priced. The 2026 systematic chunking study (Bennani &
Moslonka 2026, "A Systematic Analysis of Chunking Strategies for
Reliable Question Answering", arXiv 2601.14123, Finding F1) measured
that 10-20% overlap did not improve retrieval (|Delta BERTScore|
<= 0.004, EM differences <= 0.001) while chunk count and index size
inflate by exactly 1/(1 - r); this function returns that factor, and
the family's overlap=0 default recommendation is documented on it.

The known-answer vectors below are the study's own example (r = 0.2 ->
1.25x) plus the exact arithmetic corners; the refusal battery pins the
[0.0, 1.0) domain including the asymptote naming (overlap >= 1 is
infinite inflation, the ValueError says so) and the NaN rejection.
"""

from __future__ import annotations

import math

import pytest

import tors


class TestKnownAnswers:
    @pytest.mark.parametrize(
        ("overlap", "expected"),
        [
            (0.0, 1.0),  # the documented default: no inflation at all
            (0.2, 1.25),  # the study's own worked example (Finding F1)
            (0.5, 2.0),
            (0.75, 4.0),
            (0.9, 10.0),
            (0.99, 100.0),
        ],
    )
    def test_exact_factors(self, overlap: float, expected: float) -> None:
        assert tors.chunk_overlap_cost(overlap) == pytest.approx(expected, rel=1e-12)

    def test_it_is_the_reciprocal_identity(self) -> None:
        for overlap in (0.0, 0.1, 1 / 3, 0.5, 0.618, 0.9999):
            assert tors.chunk_overlap_cost(overlap) == pytest.approx(
                1.0 / (1.0 - overlap), rel=1e-12
            )

    def test_monotone_in_the_overlap(self) -> None:
        prev = -math.inf
        for i in range(1000):
            value = tors.chunk_overlap_cost(i / 1000)
            assert value > prev
            prev = value


class TestRefusals:
    @pytest.mark.parametrize("overlap", [-0.001, -1.0, 1.0, 1.2, 47.0, math.inf])
    def test_out_of_range_values_raise_value_error(self, overlap: float) -> None:
        with pytest.raises(ValueError, match=r"overlap must be in \[0\.0, 1\.0\)"):
            tors.chunk_overlap_cost(overlap)

    def test_the_asymptote_is_named_in_the_message(self) -> None:
        with pytest.raises(ValueError, match="asymptote"):
            tors.chunk_overlap_cost(1.0)

    @pytest.mark.parametrize("overlap", [float("nan"), -float("nan")])
    def test_nan_is_rejected_with_the_out_of_range_values(self, overlap: float) -> None:
        with pytest.raises(ValueError, match=r"overlap must be in \[0\.0, 1\.0\)"):
            tors.chunk_overlap_cost(overlap)

    def test_the_domain_is_open_below_one(self) -> None:
        # Just under the asymptote is finite and enormous, and legal.
        assert tors.chunk_overlap_cost(1.0 - 1e-12) == pytest.approx(1e12, rel=1e-3)
