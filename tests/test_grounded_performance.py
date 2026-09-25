"""The is_grounded wall lane: pin the fuzzy scan's cost shape with
machine-speed-immune ratios (never absolute times), each cell asserting its
verdict in-cell so a perf gate can never silently diverge from the
correctness contract.

The shapes, from src/grounded_impl.rs's contract:

- early-exit: a one-typo near-match at a grid-aligned offset (its aligned
  coarse window scores r = 40/41, so the coarse scan breaks within a few
  windows). The sliding-window walker means the call consumes only the
  prefix it diffed: the only O(source) work left is the exact-containment
  floor's own containment pass. Gate: the fuzzy call's wall stays within a
  small factor of a bare `claim in source` on the same bytes; before this
  PR's search-engine and windowing work the same cell measured 7.6x that
  factor at 256 KiB (247us, an O(source) Vec<char> collect plus std
  contains' non-SIMD two-way scan).
- full-scan: an unrelated claim (nothing matches, no candidates collected,
  no refinement): the linear-in-source coarse scan. Gate: quadrupling the
  source quadruples the wall (linear), never superlinear.
- band-flood: the adversarial shape: seven-typo decoys at grid-aligned
  offsets keep every coarse window in the [0.5, 0.85) candidate band, so
  the refinement runs its full constant budget (64 candidates x ~17 fine
  windows) on top of the coarse scan. Gate: quadrupling the source still
  quadruples the wall: the refinement's add-on must stay constant, not
  per-window (a per-window refinement would blow past the slack a
  quadrupling allows). The cell's verdict is asserted too: past 64 band
  windows the real region is evicted: the pinned adversarial limit
  (test_grounded.py's eviction pin carries the oracle agreement on this
  shape at tractable sizes).
- floor: a verbatim claim returns at the exact-containment floor: one
  `contains` pass, no windowing. Gate: the fuzzy call's wall stays within
  a small factor of a bare `claim in source`.

This lane is the CI wall gate for this surface (the criterion suite is
compile-only in CI by design; `pytest -m "not timing"` deselects this lane
in the matrix legs, the dedicated timing step runs it).
"""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic

import pytest

import tors

pytestmark = pytest.mark.timing

_CLAIM = "the bushing torque specifications changed"  # 41 chars, stride 20
_REAL_NEAR = "the bushing torqxe specifications changed"  # 1 typo, r = 0.976
_SUB_CHARS = ("Z", "Y", "X")
_SAMPLES = 5
_KIB = 1024


def _decoy(i: int) -> str:
    """One band-flood decoy region: seven substitutions (r = 0.829, in the
    candidate band, below the 0.85 break), pairwise DISTINCT across `i` —
    rotated typo positions plus a rotating substitution char. Distinctness
    is load-bearing: windows with identical content dedup at admission
    (one region), so only distinct decoys fill the 64-candidate cap and
    exercise the refinement's full budget this cell measures."""
    sub = _SUB_CHARS[i % 3]
    return "".join(
        sub if any((i + 7 * k) % 41 == j for k in range(7)) else c
        for j, c in enumerate(_CLAIM)
    )


def _min_wall_us(fn: Callable[..., object], *args: object) -> float:
    fn(*args)  # warmup: extension init
    best = float("inf")
    for _ in range(_SAMPLES):
        started = monotonic()
        fn(*args)
        best = min(best, monotonic() - started)
    return best * 1e6


def _fuzzy(claim: str, source: str) -> bool:
    return tors.is_grounded(claim, source, fuzzy=True)


def _contains(claim: str, source: str) -> bool:
    return claim in source


def _flood(n_decoys: int) -> str:
    """The band-flood source: the real one-typo region first (straddled,
    coarse ~0.73), then `n_decoys` grid-aligned DISTINCT decoys at 60-char
    spacing (every decoy start a multiple of the stride, 20)."""
    parts = [("q" * 10) + _REAL_NEAR]
    at = 60
    for i in range(n_decoys):
        have = sum(len(p) for p in parts)
        parts.append(("q" * (at - have)) + _decoy(i))
        at += 60
    parts.append("q" * 60)
    return "".join(parts)


def _unrelated(kib: int) -> str:
    return "q" * (kib * _KIB)


def test_early_exit_costs_little_more_than_the_floor_contains() -> None:
    """A near-match at a grid-aligned offset breaks the coarse scan within
    a few windows; with the sliding-window walker the only O(source) work
    left is the floor's `contains`. The pre-refinement implementation
    collected the whole source into a Vec<char> before windowing and
    measured ~6x this factor at 256 KiB."""
    source = ("q" * 20) + _REAL_NEAR + ("q" * (256 * _KIB))
    assert tors.is_grounded(_CLAIM, source, fuzzy=True) is True  # in-cell verdict
    fuzzy_us = _min_wall_us(_fuzzy, _CLAIM, source)
    contains_us = _min_wall_us(_contains, _CLAIM, source)
    assert fuzzy_us < 3.0 * contains_us, (
        f"early-exit fuzzy {fuzzy_us:.1f}us vs bare contains {contains_us:.1f}us"
        f" (ratio {fuzzy_us / contains_us:.2f}x): O(source) work crept back"
        " ahead of the windowed scan's exit"
    )


@pytest.mark.parametrize(
    ("name", "build", "claim"),
    [
        ("full_scan", _unrelated, "x" * 41),
        ("band_flood", lambda kib: _flood((kib * _KIB) // 60), _CLAIM),
    ],
)
def test_linear_shapes_stay_linear(name: str, build: Callable[[int], str], claim: str) -> None:
    """Quadrupling the source must quadruple the wall (within cache slack)
    on both the plain full scan (no candidates, no refinement) and the
    adversarial band flood (the refinement's full constant budget runs on
    every call): a per-window refinement or a superlinear coarse pass
    would blow the 6x slack a quadrupling allows."""
    small, large = build(64), build(256)
    if name == "band_flood":
        assert (256 * _KIB) // 60 >= 64  # the flood actually fills the cap
    # In-cell verdicts: nothing grounds the unrelated claim; the flood
    # evicts the real region past 64 band windows (the pinned limit).
    assert tors.is_grounded(claim, small, fuzzy=True) is False
    assert tors.is_grounded(claim, large, fuzzy=True) is False
    small_us = _min_wall_us(_fuzzy, claim, small)
    large_us = _min_wall_us(_fuzzy, claim, large)
    assert large_us < 6.0 * small_us, (
        f"{name}: 256 KiB {large_us:.0f}us vs 64 KiB {small_us:.0f}us"
        f" (ratio {large_us / small_us:.1f}x; linear is ~4x) — the scan"
        " grew superlinearly in the source"
    )


def test_the_floor_is_one_contains_pass() -> None:
    """A verbatim claim returns at the exact-containment floor before any
    windowing: the fuzzy call must track a bare `claim in source` (the
    pyo3 boundary is the only difference). If windowing ever creeps ahead
    of the floor, this ratio blows up."""
    source = ("q" * 128) + _CLAIM + ("q" * (256 * _KIB))
    assert tors.is_grounded(_CLAIM, source, fuzzy=True) is True  # in-cell verdict
    fuzzy_us = _min_wall_us(_fuzzy, _CLAIM, source)
    contains_us = _min_wall_us(_contains, _CLAIM, source)
    assert fuzzy_us < 3.0 * contains_us, (
        f"floor fuzzy {fuzzy_us:.2f}us vs bare contains {contains_us:.2f}us"
        f" (ratio {fuzzy_us / contains_us:.2f}x): work is running ahead of"
        " the exact-containment floor"
    )


class TestGroundingCoverageWall:
    """The recall twin's wall lane: grounding_coverage's documented cost is
    the classic weighted-LCS DP over the two token streams: O(|S| x |T|)
    TIME (the product both operands own, the same class levenshtein
    documents) with O(min(|S|, |T|)) MEMORY (two reused rows, never a
    materialized n*m matrix), each operand capped at its first 16384
    tokens (src/grounded_impl.rs). The one-sided axis (doubling ONE
    operand) is linear; the two-sided axis is the documented product,
    pinned at its 4x-per-doubling band so an accidental cubic formulation
    (per-cell reallocation, an n*m matrix) blows through the gate."""

    def _wall(self, *args: str) -> float:
        return _min_wall_us(tors.grounding_coverage, *args) / 1e6  # -> seconds

    def test_one_sided_doubling_stays_linear(self) -> None:
        source = "word " * 4_000
        small = self._wall(source, "word " * 2_000)
        large = self._wall(source, "word " * 4_000)
        assert large / max(small, 1e-9) < 2 * 1.6, (
            f"one-sided doubling grew {large / small:.2f}x (allowed < 3.2x)"
        )

    def test_two_sided_doubling_stays_at_the_product(self) -> None:
        small = self._wall("word " * 2_000, "word " * 2_000)
        large = self._wall("word " * 4_000, "word " * 4_000)
        # 4x cells for 2x per side: the documented product class; a
        # cubic-per-doubling step (>= 8x) trips.
        assert large / max(small, 1e-9) < 4.0 * 1.6, (
            f"two-sided doubling grew {large / small:.2f}x (allowed < 6.4x)"
        )

    def test_the_realistic_rag_shape_completes_inside_the_thread_hop_budget(self) -> None:
        # The utilization lane's real shape: a 4k-token retrieved context
        # against a 300-word response. Generous absolute ceiling on
        # purpose; the ratio gates above carry the regression
        # sensitivity; this catches only a qualitative break (an
        # accidental n*m matrix allocation). The ceiling rides over the
        # dev-profile build the local suite runs (a release wheel scores
        # the same shape an order of magnitude under it).
        wall = self._wall("word " * 4_000, "word " * 300)
        assert wall < 0.5, f"4k-token context vs 300-word response took {wall * 1e3:.1f}ms"
