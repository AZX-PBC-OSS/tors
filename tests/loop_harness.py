"""The suite's shared load-robustness harness for timing cells.

Every cell that asserts on WALL TIME or a MEASURED RATIO must be robust to
ambient runner load (the evidence: cells calibrated with raw absolute-wall
asserts pass on quiet runners and fail at ambient load 8-24). The proven
methodology lives in tests/test_gil_release.py (multi-sample, pass on the
first clean sample; a sample is judged against both a ratio budget over the
operation's OWN wall and an absolute ceiling; heartbeat and switch-interval
derived budgets where the claim is loop responsiveness). This module makes
that methodology importable, so a cell hardens by wrapping its measurement
instead of restating the pattern:

- ``gap_and_wall``: run an op with a 10ms heartbeat; return the worst tick
  gap and the op's own wall (captured at op completion, before the
  heartbeat join, so no drain ping is absorbed).
- ``assert_heartbeat_clean``: the pass-on-first-clean loop-responsiveness
  assert over up to ``samples`` measurements (a sample is clean exactly
  when the worst gap is under BOTH the ratio budget and the absolute
  ceiling). A GIL-held whole pass misses the budgets in every sample and
  fails; transient whole-process CPU starvation hits only the sample it
  lands in and is retried.
- ``first_clean``: the generic retry discipline for cells whose clean
  predicate is bespoke (the OR-shaped ``gap < ratio * wall or gap <
  ceiling`` budget some families use): run the measurement up to
  ``samples`` times, return on the first run whose ``check`` passes, raise
  with every run's detail otherwise.
- ``min_wall_ms``: min-of-N wall after warmup, the suite's standard
  load-fair measurement primitive.
- ``assert_bounded``: the load-robust replacement for the single-shot
  absolute-wall assert (``start = ...; op(); assert elapsed < X``): run
  ``call`` up to ``samples`` times and pass on the first sample whose wall
  is under the ceiling. A real regression inflates EVERY sample (the
  same dirt-in-every-sample property the heartbeat harness relies on),
  so the ceiling keeps its teeth while one scheduler hit no longer fails
  the cell. ``call`` may raise (the deadline cells' expected
  ``TimeoutError``): the timer wraps the call whatever it does, and a
  missing expected exception propagates immediately.

The red side is proved per helper in tests/test_loop_harness.py: a
deliberately slow or GIL-held op fails each one.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# The shared budgets (tests/test_gil_release.py's defaults): the ratio
# budget against the op's own wall, plus the absolute ceiling.
DEFAULT_RATIO_BUDGET = 0.30
DEFAULT_CEILING_S = 0.100
DEFAULT_SAMPLES = 3

_PING_S = 0.01


def gap_and_wall(op: Callable[[], object]) -> tuple[float, float]:
    """Run ``op`` (sync, or an awaitable factory) with a 10ms heartbeat and
    return ``(worst_tick_gap, op_wall)``.

    The leading ``sleep(0)`` guarantees the heartbeat's first tick lands
    before the operation starts, so a fully blocking operation shows its
    whole duration as the worst gap. The wall ends the moment ``op``
    returns, before the heartbeat join, so it is the operation's wall, not
    the operation plus up to one heartbeat-drain ping (the
    test_gil_release.py methodology)."""
    ticks: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        while True:
            ticks.append(time.monotonic())
            if stop.is_set():
                return
            await asyncio.sleep(_PING_S)

    async def run() -> tuple[float, float]:
        hb = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        started = time.monotonic()
        try:
            result = op()
            if asyncio.iscoroutine(result):
                await result
            elif hasattr(result, "__await__"):
                await result  # type: ignore[arg-type]
            end = time.monotonic()
        finally:
            stop.set()
            await hb
        wall = end - started
        worst = max((b - a for a, b in itertools.pairwise(ticks)), default=0.0)
        return worst, wall

    return asyncio.run(run())


def _both_budget_misses(
    gap: float, wall: float, ratio_budget: float, ceiling_s: float
) -> list[str]:
    """What a sample missed under the AND semantics (both budgets must
    hold): the single source for what "clean" means in
    ``assert_heartbeat_clean``, so the check and the failure message can
    never disagree on the boundary."""
    missed: list[str] = []
    if gap >= ceiling_s:
        missed.append(f"the {ceiling_s * 1000:.0f}ms ceiling")
    if gap >= ratio_budget * wall:
        missed.append(f"the {ratio_budget:.0%} ratio budget")
    return missed


def assert_heartbeat_clean(
    op: Callable[[], object],
    *,
    samples: int = DEFAULT_SAMPLES,
    ratio_budget: float = DEFAULT_RATIO_BUDGET,
    ceiling_s: float = DEFAULT_CEILING_S,
    subject: str = "the operation",
) -> None:
    """Assert ``op`` leaves the event loop schedulable, over up to
    ``samples`` measurements, passing on the first clean one (the
    test_gil_release.py pattern). A GIL-held whole pass misses the budgets
    in EVERY sample and fails; whole-process CPU starvation does not, so
    one starved sample retries instead of failing the cell. A sample is
    clean exactly when the worst tick gap stays under BOTH the ratio
    budget (a held whole pass reads ~1.0) and the absolute ceiling."""
    observed: list[tuple[float, float]] = []
    for _ in range(samples):
        worst, wall = gap_and_wall(op)
        observed.append((worst, wall))
        if not _both_budget_misses(worst, wall, ratio_budget, ceiling_s):
            return
    detail = "; ".join(
        f"worst {worst * 1000:.1f}ms of a {wall * 1000:.0f}ms operation "
        f"({(worst / wall if wall else 0):.0%}, over "
        f"{' and '.join(_both_budget_misses(worst, wall, ratio_budget, ceiling_s))})"
        for worst, wall in observed
    )
    raise AssertionError(
        f"{subject} missed the heartbeat budgets in every one of {samples} "
        f"samples ({detail}): the py.detach release is not freeing the loop, "
        "or the return marshalling regressed out of its band "
        "(tests/test_gil_release.py)"
    )


def first_clean(
    run: Callable[[], T],
    check: Callable[[T], None],
    *,
    samples: int = DEFAULT_SAMPLES,
    label: str = "the cell",
) -> T:
    """The generic pass-on-first-clean retry: run ``run()`` up to
    ``samples`` times, after each run apply ``check`` (which raises
    AssertionError when that run is dirty), and return the first clean
    run's result. When every run is dirty, raise with each run's
    AssertionError detail. Bespoke budgets (a family's OR-shaped
    ``gap < ratio * wall or gap < ceiling``, say) stay with the cell;
    only the retry discipline is shared."""
    failures: list[str] = []
    for index in range(samples):
        result = run()
        try:
            check(result)
        except AssertionError as miss:
            failures.append(f"sample {index + 1}/{samples}: {miss}")
        else:
            return result
    raise AssertionError(
        f"{label} was dirty in every one of {samples} samples:\n"
        + "\n".join(failures)
    )


def min_wall_ms(
    fn: Callable[..., object], *args: object, samples: int = 3, warmup: int = 1
) -> float:
    """Min-of-``samples`` wall in milliseconds after ``warmup`` calls: the
    fastest of several draws approximates the uncontended cost, the suite's
    standard load-fair measurement primitive."""
    for _ in range(warmup):
        fn(*args)
    best = float("inf")
    for _ in range(samples):
        started = time.monotonic()
        fn(*args)
        best = min(best, time.monotonic() - started)
    return best * 1000.0


def assert_bounded(
    call: Callable[[], T],
    ceiling_s: float,
    *,
    samples: int = DEFAULT_SAMPLES,
    label: str = "the call",
) -> T | None:
    """The load-robust absolute-wall band: run ``call`` up to ``samples``
    times, pass on the first sample whose wall is under ``ceiling_s``,
    return that sample's result. The single-shot assert this replaces
    (``start = ...; call(); assert elapsed < ceiling``) fails when one
    scheduler hit lands on its only sample; a real regression inflates
    every sample, so the ceiling keeps its teeth (red-side proof:
    tests/test_loop_harness.py). ``call`` may raise its own expected
    exception (a deadline cell's ``TimeoutError``); anything else
    propagates immediately."""
    walls: list[float] = []
    for _ in range(samples):
        started = time.monotonic()
        try:
            result = call()
        except BaseException:
            walls.append(time.monotonic() - started)
            raise
        wall = time.monotonic() - started
        if wall < ceiling_s:
            return result
        walls.append(wall)
    detail = ", ".join(f"{w * 1000:.0f}ms" for w in walls)
    raise AssertionError(
        f"{label} took {detail} over {samples} samples, every one over the "
        f"{ceiling_s * 1000:.0f}ms ceiling: the bounded pass regressed "
        "(not a transient scheduler hit; see tests/loop_harness.py)"
    )
