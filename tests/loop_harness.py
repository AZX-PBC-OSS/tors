"""The shared event-loop heartbeat harness.

One implementation of the measurement every GIL-claim cell repeats: run
``op`` on the loop while a 10ms heartbeat task records tick times, and
return the worst tick gap and the call's wall time. ``op`` is
callable-or-awaitable (a sync lambda returning a coroutine, an
``asyncio.to_thread`` future, or a bare awaitable all work).

The three copies this replaces (test_rank_fusion.py's
``TestAioTwins._gap_and_wall``, test_ground_sentences.py's
``_gap_and_wall``/``_assert_heartbeat_clean``, test_chunk_to_budget.py's
``_gap_and_wall``) disagreed on nothing: same 10ms cadence, same
pairwise-worst-gap reduction, same pass-on-first-clean retry discipline.
Cells keep their own budgets; this module owns the measurement and the
budget shape.

The red side is pinned in the files that use it (a GIL-held C call
measures a worst gap ~= the whole wall): the harness exists to detect a
held pass, and ``assert_heartbeat_clean`` must fail on one.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Awaitable, Callable

# The defaults test_ground_sentences.py's constants pinned: a held whole
# pass reads a ~1.0 ratio; the detached pass's marshalling residue rides
# the wall, so the ceiling catches a bounded pathological hold.
DEFAULT_RATIO_BUDGET = 0.35
DEFAULT_CEILING_S = 0.5
DEFAULT_SAMPLES = 3


async def heartbeat_gap_and_wall(
    op: Callable[[], Awaitable[object]] | Awaitable[object],
) -> tuple[float, float]:
    """Run ``op`` under a 10ms heartbeat; return ``(worst_gap_s, wall_s)``.

    ``op`` may be a zero-arg callable whose result is awaited (sync
    lambdas returning coroutines, ``asyncio.to_thread`` futures) or an
    already-created awaitable. The worst gap is the largest tick-to-tick
    heartbeat interval (``0.0`` when fewer than two ticks landed); the
    wall is the op's own start-to-end time.
    """
    ticks: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        while True:
            ticks.append(time.monotonic())
            if stop.is_set():
                return
            await asyncio.sleep(0.01)

    task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    started = time.monotonic()
    try:
        await (op() if callable(op) else op)
        end = time.monotonic()
    finally:
        stop.set()
        await task
    wall = end - started
    worst = max((b - a for a, b in itertools.pairwise(ticks)), default=0.0)
    return worst, wall


async def assert_heartbeat_clean(
    op: Callable[[], Awaitable[object]] | Awaitable[object],
    ratio: float = DEFAULT_RATIO_BUDGET,
    ceiling: float = DEFAULT_CEILING_S,
    samples: int = DEFAULT_SAMPLES,
) -> None:
    """Assert ``op`` leaves the event loop schedulable, over up to
    ``samples`` measurements, passing on the first clean one (the
    harness's ``test_gil_release.py`` pattern): a GIL-held whole pass
    misses its budgets in EVERY sample, scheduler starvation of the
    heartbeat task does not, so one starved sample retries instead of
    failing the cell. A sample is clean exactly when the worst tick gap
    stays under BOTH the ratio budget (a fraction of the wall) and the
    absolute ceiling in seconds."""
    observed: list[tuple[float, float]] = []
    for _ in range(samples):
        worst, wall = await heartbeat_gap_and_wall(op)
        assert wall > 0.02, (
            f"the op finished in {wall * 1000:.1f}ms, under the 20ms "
            "floor a heartbeat measurement can read"
        )
        observed.append((worst, wall))
        if worst < ratio * wall and worst < ceiling:
            return
    detail = "; ".join(
        f"worst {worst * 1000:.0f}ms of a {wall * 1000:.0f}ms operation ({worst / wall:.0%})"
        for worst, wall in observed
    )
    raise AssertionError(
        f"the heartbeat missed its budgets in every one of {samples} samples "
        f"({detail}): the op held the loop past the pinned band "
        f"(ratio {ratio}, ceiling {ceiling})"
    )
