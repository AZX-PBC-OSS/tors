"""The harness's own teeth, proved mechanically (the suite's red-side
discipline): every helper in tests/loop_harness.py must FAIL on a
deliberately dirty op, so a future refactor of the harness cannot silently
disarm the cells that lean on it.

- ``assert_heartbeat_clean`` red side: a GIL-held whole-text C call
  (``re.sub`` over a large string, the suite's documented red side) blocks
  the loop for its whole duration, reads ratio ~1.0, and must fail the
  assert in every sample.
- ``assert_bounded`` red side: a deliberately slow op (a sleep-injected
  call, the scratch-copy regression injection) blows the ceiling in every
  sample and must fail; the same op with the sleep removed passes, proving
  the failure is the wall, not the wrapper.
- ``first_clean`` red side: a check that never passes fails with every
  sample's detail; a check that passes on a later sample returns that
  sample's result (the retry discipline itself).
"""

from __future__ import annotations

import re
import time

import pytest

import tors  # noqa: F401 -- the harness's subjects are tors ops in the real cells
from loop_harness import assert_bounded, assert_heartbeat_clean, first_clean, gap_and_wall


def test_heartbeat_harness_fails_a_gil_held_pass() -> None:
    """Red side: a pure-Python CPU loop in a thread holds the GIL; the
    heartbeat must see it (measured: worst gap ~= the whole wall, ratio
    ~1.0, every sample) and ``assert_heartbeat_clean`` must raise."""

    def gil_held_c_call() -> str:
        return re.sub("x", "y", "x" * 30_000_000)

    with pytest.raises(AssertionError, match="missed the heartbeat budgets"):
        assert_heartbeat_clean(
            lambda: __import__("asyncio").to_thread(gil_held_c_call),
            samples=2,
            subject="the red-side GIL-held pass",
        )


def test_bounded_wall_fails_a_sleep_injected_op_and_passes_the_fast_one() -> None:
    """Red side: the deliberate-regression injection (a sleep inside the
    measured call) must fail ``assert_bounded`` in every sample; the same
    call without the sleep must pass the same ceiling, so the failure is
    the wall and not the wrapper."""
    calls = {"n": 0}

    def slow() -> int:
        calls["n"] += 1
        time.sleep(0.02)
        return calls["n"]

    with pytest.raises(AssertionError, match="every one over"):
        assert_bounded(slow, 0.005, samples=3, label="the sleep-injected call")
    assert calls["n"] == 3, "the red side must have drawn all three samples"

    fast = assert_bounded(lambda: 7, 0.005, samples=3, label="the fast call")
    assert fast == 7


def test_bounded_wall_propagates_an_unexpected_exception_immediately() -> None:
    """An exception the cell did not expect (anything but the expected
    deadline abort) is a correctness failure, not a slow sample: it must
    propagate on the first sample, not be retried into a wall verdict."""

    def broken() -> None:
        raise RuntimeError("not the expected exception")

    with pytest.raises(RuntimeError, match="not the expected exception"):
        assert_bounded(broken, 5.0, samples=3)


def test_first_clean_retries_until_clean_and_reports_every_dirty_sample() -> None:
    """The retry discipline: pass on the first clean sample and return its
    result; when no sample is clean, raise with every sample's detail."""
    attempts = {"n": 0}

    def flaky() -> float:
        attempts["n"] += 1
        return float(attempts["n"])

    def clean_on_third(value: float) -> None:
        assert value >= 3.0, f"attempt {value:.0f} was below the bar"

    assert first_clean(flaky, clean_on_third, samples=3, label="the retry") == 3.0
    assert attempts["n"] == 3

    with pytest.raises(AssertionError, match="dirty in every one of 2 samples") as miss:
        first_clean(lambda: 1.0, clean_on_third, samples=2, label="the always-dirty cell")
    assert "sample 1/2" in str(miss.value) and "sample 2/2" in str(miss.value)


def test_gap_and_wall_measures_a_blocking_call_honestly() -> None:
    """The measurement primitive: a blocking ``time.sleep`` in the loop's
    own task shows up as the worst gap and the wall (a green-side sanity
    anchor for the two asserts above)."""
    worst, wall = gap_and_wall(lambda: time.sleep(0.05))
    assert wall >= 0.05, wall
    assert worst >= 0.05, worst
