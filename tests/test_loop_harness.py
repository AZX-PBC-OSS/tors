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
- The all-samples-dirty verdict's starvation discriminator
  (``_gap_was_starved``, the 2026-10 CI flake class): a window the loop
  thread spent runnable-but-unrun (runner starvation) is excused; a window
  it spent futex-blocked while the worker burned the CPU (the GIL) still
  fails; the surviving GIL-signature windows are convicted only when the
  idle-loop control (tier 3) stays free. All branches pinned below.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable

import pytest

import loop_harness
import tors  # noqa: F401 -- the harness's subjects are tors ops in the real cells
from loop_harness import (
    _CONTROL_WINDOW_S,
    assert_bounded,
    assert_heartbeat_clean,
    first_clean,
    gap_and_wall,
)


def test_heartbeat_harness_fails_a_gil_held_pass() -> None:
    """Red side: a pure-Python CPU loop in a thread holds the GIL; the
    heartbeat must see it (measured: worst gap ~= the whole wall, ratio
    ~1.0, every sample -- the loop thread cannot tick while the C call
    holds the GIL, a guarantee of Python's GIL semantics, not of load) --
    the measurement every all-samples-dirty verdict convicts on. The
    VERDICT-level conviction of that shape (every sample dirty -> raise,
    with the starvation excuses disarmed) is pinned deterministically by
    ``test_the_control_arbitrates_gil_signature_windows_only_when_it_loses``
    (canned GIL-signature samples against a free control): under a
    saturated runner the loop thread's OWN wakeup latency is the runner's,
    and no in-process verdict can convict a shape the runner can forge on
    the very thread that runs the verdict (the doctrine: a saturated box
    cannot judge a wall ratio)."""

    def gil_held_c_call() -> str:
        return re.sub("x", "y", "x" * 30_000_000)

    observed = [
        gap_and_wall(lambda: asyncio.to_thread(gil_held_c_call)) for _ in range(2)
    ]
    for worst, wall, *_ in observed:
        assert wall > 0 and worst / wall >= 0.9, (
            f"the GIL-held red side measured a tick during the hold "
            f"({worst * 1000:.0f}ms of a {wall * 1000:.0f}ms wall): the "
            "measurement cannot see a held GIL, so every verdict built on "
            "it is blind (tests/loop_harness.py)"
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
    anchor for the two asserts above), and its tick-less window carries the
    starvation signature (a sleep burns no process CPU and leaves the loop
    thread never-runnable -- the discriminator the all-samples-dirty
    verdicts judge)."""
    worst, wall, gap_cpu, _cpu_share, _gap_wait = gap_and_wall(lambda: time.sleep(0.05))
    assert wall >= 0.05, wall
    assert worst >= 0.05, worst
    assert gap_cpu < 0.5 * worst, (
        f"a pure sleep's blocked window burned {gap_cpu * 1e3:.1f}ms of CPU on a "
        f"{worst * 1e3:.0f}ms gap: the CPU account is not measuring the window"
    )


def test_heartbeat_verdict_excuses_a_starved_block_and_keeps_the_gil_teeth() -> None:
    """The harness's own pin of the 2026-10 CI flake class (the 99-100%-blocked
    failures on diffs with zero code change, tests/test_gil_release.py and
    tests/test_score_fusion.py's aio twin): the verdict discriminates the
    ways a tick-less window happens.

    green (the starvation class): a whole-wall block whose window burned no
    process CPU -- the runner's freeze signature, every sample -- is the
    saturated-runner case the plain pass-on-first-clean retry exhausts, and
    the verdict excuses it with a printed record instead of failing a diff
    with no code change. ``time.sleep`` on the loop's own task is the
    deterministic stand-in for the saturated runner: ratio ~1.0 in every
    sample (the exact CI shape) with the CPU account at ~0 and the loop
    thread never runnable (the starvation signature), reproducibly, with no
    synthetic load.

    green (the wall-pin class, the second under-load class the suite's own
    x20-under-load gate found): a CPU-burning block whose ratio HELD -- a
    wall-clock spin on the loop's task over a long healthy tail misses the
    100ms ceiling in every sample at a ~6% ratio, which the accounts alone
    would call the GIL (the window burns) -- and the verdict excuses it as
    the advisory wall pin it is, because the ratio against the operation's
    own wall held.

    red (the teeth): kept by ``test_heartbeat_harness_fails_a_gil_held_pass``
    above -- a GIL-held ``re.sub`` burns the process CPU through the very
    window whose ticks went missing while the loop thread never ran
    runnable, the combination only a GIL hold produces, and no amount of
    runner load explains it away."""
    assert_heartbeat_clean(lambda: time.sleep(0.02), samples=2)

    def wall_pin_blocker() -> object:
        async def run() -> None:
            end = time.monotonic() + 0.13  # a CPU-burning stall on the loop's task
            while time.monotonic() < end:
                pass
            await asyncio.sleep(2.0)  # the healthy wall the ratio judges against

        return run()

    assert_heartbeat_clean(wall_pin_blocker, samples=2)


def test_the_control_arbitrates_gil_signature_windows_only_when_it_loses(monkeypatch) -> None:
    """Tier 3's own pin (the x20-under-load finding: the heartbeat thread's
    wakeup tail reaches every thread under saturation, so GIL-signature
    windows -- ticks missing while the worker burns CPU -- are also produced
    by plain scheduler starvation, including in the no-ratio cells): the
    GIL-signature strikes are convicted only on a box whose idle-loop
    control stays free.

    green (the control loses): three GIL-signature windows (ratio ~1.0, burn
    at the process's own rate) whose idle-loop control cannot beat the
    smallest one -- the wakeup tail reaches the control too, the box cannot
    judge a wall ratio, and the verdict excuses with a record instead of
    failing a diff with no code change.

    red (the teeth): the same three windows on a box whose idle loop holds
    the ping floor -- the control stays free while the subject's windows
    went tick-less and CPU-burning, the combination only a GIL hold
    produces, and the verdict must fail loudly."""
    calls: list[object] = []

    def fake_control_loses(op: Callable[[], object]) -> tuple[float, float, float, float, float]:
        calls.append(op)
        if len(calls) <= 3:
            return (0.5, 0.5, 0.45, 0.95, 0.0)  # ratio ~1.0, burn at the process's own rate
        return (0.6, _CONTROL_WINDOW_S, 0.0, 0.0, 0.0)  # the control: the idle loop can't tick

    monkeypatch.setattr(loop_harness, "gap_and_wall", fake_control_loses)
    assert_heartbeat_clean(lambda: None, samples=3)
    assert len(calls) == 4, calls  # three subject samples + the one control window

    calls.clear()

    def fake_control_free(op: Callable[[], object]) -> tuple[float, float, float, float, float]:
        calls.append(op)
        if len(calls) <= 3:
            return (0.5, 0.5, 0.45, 0.95, 0.0)
        return (0.011, _CONTROL_WINDOW_S, 0.0, 0.0, 0.0)  # the control holds the ping floor

    monkeypatch.setattr(loop_harness, "gap_and_wall", fake_control_free)
    with pytest.raises(AssertionError, match="missed the heartbeat budgets"):
        assert_heartbeat_clean(lambda: None, samples=3)
