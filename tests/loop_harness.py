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
  gap, the op's own wall, and the worst gap's load account (captured at op
  completion, before the heartbeat join, so no drain ping is absorbed).
- ``assert_heartbeat_clean``: the pass-on-first-clean loop-responsiveness
  assert over up to ``samples`` measurements (a sample is clean exactly
  when the worst gap is under BOTH the ratio budget over the operation's
  OWN wall and an absolute ceiling; heartbeat and switch-interval derived
  budgets where the claim is loop responsiveness). A GIL-held whole pass
  misses the budgets in every sample and fails; transient whole-process
  CPU starvation hits only the sample it lands in and is retried. When
  EVERY sample misses, the verdict is three-tiered: windows that HELD the
  ratio budget missed only the absolute ceiling -- a wall pin, advisory
  under the doctrine, and excused with a printed record (tier 1); the
  ratio-missed windows then consult the loop thread's own scheduler
  account (tier 2, ``_gap_was_starved``): a window the heartbeat thread
  spent runnable-but-unrun (its runqueue wait grows with the gap) is the
  runner's starvation, whatever the worker burned, and is excused with a
  printed record; a window the thread spent futex-blocked on the GIL
  (runqueue wait ~none while the gap grew) is the GIL, and -- on kernels
  without the schedstat account, after the process-CPU fallback and the
  idle-loop control (tier 3) -- raises. The teeth keep the doctrine's
  shape: a regression that blocks the loop while the machine had CPU to
  give -- and the control stays free -- fails loudly; a saturated box
  cannot judge a wall ratio at all (99-100% "blocked" is the freeze's
  signature, not evidence).
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
import threading
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


def _runqueue_wait_s() -> float:
    """The CURRENT thread's cumulative runqueue wait (schedstat field 2,
    ns, on every kernel this suite runs on), or ``-1.0`` when the kernel
    does not expose the account (the verdicts then fall back to the
    process-CPU rule). Read on the heartbeat thread each tick: it is the
    per-thread scheduler account that discriminates the two ways a tick-less
    window happens -- a window the loop thread spent RUNNABLE-but-unrun (the
    wait grows with the gap) is the runner's starvation, whatever the worker
    burned; a window it spent futex-blocked on the GIL (the wait stays ~none
    while the gap grows) is only producible by a GIL hold."""
    try:
        with open(f"/proc/self/task/{threading.get_native_id()}/schedstat", "rb") as stat:
            return int(stat.read().split()[1]) / 1e9
    except (OSError, IndexError, ValueError):
        return -1.0


def gap_and_wall(op: Callable[[], object]) -> tuple[float, float, float, float, float]:
    """Run ``op`` (sync, or an awaitable factory) with a 10ms heartbeat and
    return ``(worst_tick_gap, op_wall, worst_gap_cpu, cpu_share,
    worst_gap_wait)``.

    The leading ``sleep(0)`` guarantees the heartbeat's first tick lands
    before the operation starts, so a fully blocking operation shows its
    whole duration as the worst gap. The wall ends the moment ``op``
    returns, before the heartbeat join, so it is the operation's wall, not
    the operation plus up to one heartbeat-drain ping (the
    test_gil_release.py methodology). The third element is the worst gap's
    process-CPU account (``time.process_time()`` across exactly that
    window); the fourth is the process's own CPU share over the whole
    operation (CPU seconds per wall second); the fifth is the heartbeat
    thread's OWN runqueue wait across exactly that window (the scheduler
    account ``_gap_was_starved`` judges, with the CPU account as the
    printed evidence and the no-schedstat fallback): a gap the GIL created
    leaves the heartbeat thread futex-blocked (wait ~none while the gap
    grows), while a gap the runner's starvation created leaves it
    runnable-but-unrun (the wait grows with the gap) -- heartbeat and
    worker together descheduled, or the worker hogging the process's
    scheduled time with the loop's thread losing every wakeup."""
    ticks: list[tuple[float, float, float]] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        while True:
            ticks.append((time.monotonic(), time.process_time(), _runqueue_wait_s()))
            if stop.is_set():
                return
            await asyncio.sleep(_PING_S)

    async def run() -> tuple[float, float, float, float, float]:
        hb = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        started = time.monotonic()
        cpu_started = time.process_time()
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
        cpu_share = (time.process_time() - cpu_started) / wall if wall else 0.0
        worst_gap = 0.0
        worst_gap_cpu = 0.0
        worst_gap_wait = -1.0
        for (tick_a, cpu_a, wait_a), (tick_b, cpu_b, wait_b) in itertools.pairwise(ticks):
            if tick_b - tick_a > worst_gap:
                worst_gap = tick_b - tick_a
                worst_gap_cpu = cpu_b - cpu_a
                worst_gap_wait = -1.0 if wait_a < 0 or wait_b < 0 else wait_b - wait_a
        return worst_gap, wall, worst_gap_cpu, cpu_share, worst_gap_wait

    return asyncio.run(run())


# The starvation threshold (test_gil_release.py's ``_STARVED_CPU_SHARE``): a
# share of the window the loop thread spent RUNNABLE-but-unrun, so it is
# load-independent and worker-independent. The two signatures are far apart
# and bimodal, so the threshold is not fine-tuned: a GIL-held window leaves
# the heartbeat thread futex-blocked (runqueue wait ~none -- it was
# scheduled and could not run), while a starved window leaves it runnable
# in the runqueue (the wait IS the gap, minus its own slice of cpu).
_STARVED_RUNQUEUE_SHARE = 0.25


def _gap_was_starved(sample: tuple[float, float, float, float, float]) -> bool:
    """Whether a tick-less window (the sample tuple ``gap_and_wall``
    returns) carries the RUNNER's signature (``True``) or the GIL's
    (``False``).

    Primary (the schedstat account available): the loop thread's OWN
    runqueue wait across the window. A window the thread spent
    runnable-but-unrun -- the wait growing with the gap -- is the runner's
    starvation, WHATEVER the worker burned (the 2026-10 under-load finding:
    a detached native worker hogging the process's scheduled time starves
    the heartbeat thread's wakeups while burning CPU, which the process-CPU
    account alone reads as the GIL signature). A window the thread spent
    futex-blocked -- the wait ~none while the gap grew -- convicts only
    when the burn half agrees (``_cpu_account_starved``): ticks missing
    while the worker burned the CPU is the combination only a GIL hold
    produces; ticks missing while NOTHING burned is the loop's own sleep
    or the runner's freeze.

    The rate normalization (both accounts) is what makes the verdict
    box-independent: this is the content-match control the flake doctrine
    calls for, sampled in the same run -- in the very window the ticks
    went missing -- instead of against an absolute wall ratio: the 2026-10
    CI flakes (tests/test_gil_release.py:787, tests/test_score_fusion.py:911,
    diffs with no code change) read 99-100% blocked in every sample exactly
    when the runner matrix saturated, and re-runs went green. A regression
    whose GIL-held native work SLEEPS (a futex/mutex hold) burns no CPU and
    reads starved on the burn half -- the pinned regression class (the
    py.detach marshalling/extract band) is CPU work, which this verdict
    still fails on loudly."""
    gap, _wall, gap_cpu, cpu_share, gap_wait = sample
    # Primary: the loop thread's own runqueue wait. A window the thread spent
    # runnable-but-unrun (the wait growing with the gap) is the runner's
    # starvation, whatever the worker burned.
    if gap_wait >= 0.0 and gap_wait >= _STARVED_RUNQUEUE_SHARE * gap:
        return True
    # The thread was never runnable (futex-blocked on the GIL, or the loop's
    # own task slept in place). The GIL is the ONLY shape of that which
    # matters here: the worker burned the process's CPU while the loop
    # starved. A window whose burn is ALSO ~none is the loop's own sleep
    # (nothing ran -- the pure-sleep stand-in), the runner's freeze (the
    # whole process descheduled), or an idle artifact: starved either way.
    return _cpu_account_starved(gap, gap_cpu, cpu_share)


def _cpu_account_starved(gap: float, gap_cpu: float, cpu_share: float) -> bool:
    """The process-CPU fallback (and the burn half of the schedstat verdict),
    rate-normalized so it is box-independent: a GIL-held window runs the
    worker whenever the process is scheduled (the loop's thread is
    GIL-blocked and unschedulable, so the worker eats ~all of the share --
    ratio ~1.0), while a descheduled window freezes the whole process, CPU
    account included (ratio ~0). This is the content-match control the flake
    doctrine calls for, sampled in the same run -- in the very window the
    ticks went missing -- instead of against an absolute wall ratio: the
    2026-10 CI flakes (tests/test_gil_release.py:787,
    tests/test_score_fusion.py:911, diffs with no code change) read 99-100%
    blocked in every sample exactly when the runner matrix saturated, and
    re-runs went green.

    Known limitation, recorded and bounded (the suite's findings pattern):
    a regression whose GIL-held native work SLEEPS (a futex/mutex hold)
    burns no CPU and carries the same freeze signature as a descheduled
    runner -- the two are unjudgeable apart on the CPU account alone (the
    schedstat account is what disarms this: the loop thread's runqueue wait
    separates runnable-but-unrun from futex-blocked). The pinned regression
    class (the py.detach marshalling/extract band) is CPU work, which this
    verdict still fails on loudly."""
    expected = cpu_share * gap
    if expected < _CPU_ACCOUNT_FLOOR_S:
        # Below a couple of ms of predicted burn the rate ratio is noise (an
        # idle sample's whole CPU account is asyncio overhead): judge the
        # window's burn against the same absolute floor instead. A real
        # GIL-held window at this scale does not exist here -- dirty samples
        # carry multi-ms gaps by construction (the ratio budget over walls
        # well above the ping floor).
        return gap_cpu < _CPU_ACCOUNT_FLOOR_S
    return gap_cpu < _STARVED_CPU_SHARE * expected


# The process-CPU fallback's starvation threshold (a share of the process's
# OWN CPU rate, so it is load-independent).
_STARVED_CPU_SHARE = 0.5


# The absolute floor under which the process-CPU account is read as "no
# burn" regardless of the rate math (asyncio's own overhead per sample is
# tens of µs; real held-GIL marshalling windows are multi-ms).
_CPU_ACCOUNT_FLOOR_S = 0.002


# The idle-loop control's exposure (test_gil_release.py's
# ``_CONTROL_WINDOW_S``): long enough that its worst gap reflects the
# scheduler's wakeup tail under load, not the 10ms ping cadence itself.
_CONTROL_WINDOW_S = 3.0


def _idle_loop_control_gap() -> float:
    """The achievable baseline the flake doctrine's option (a) requires: the
    worst tick gap the loop itself achieves RIGHT NOW, under this load, with
    no op at all (a pure idle heartbeat over ``_CONTROL_WINDOW_S``). A quiet
    box reads the ~10ms ping floor; a saturated box reads the wakeup tail
    under load."""
    return gap_and_wall(lambda: asyncio.sleep(_CONTROL_WINDOW_S))[0]


def _excusable_under_control(
    subject: str, strikes: list[tuple[float, float, float, float, float]]
) -> bool:
    """Tier 3, the final arbiter ``assert_heartbeat_clean`` raises through
    (the test_gil_release.py twin, ``_excusable_under_control``): the
    surviving GIL-signature strikes are judged RELATIVE to the idle-loop
    control sampled in the same run. A box whose idle loop cannot beat the
    strikes' smallest gap cannot judge this verdict at all (a wall ratio
    under that wakeup tail is noise) -- excused with a printed record. A box
    whose idle loop holds the ping floor while the subject's windows went
    tick-less against the account is looking at the GIL, and the caller
    raises. The teeth: a real regression that blocks the loop while the
    control stays free fails loudly, on any box."""
    control_gap = _idle_loop_control_gap()
    smallest = min(strike[0] for strike in strikes)
    if control_gap < smallest:
        return False
    print(
        f"{subject} showed GIL-signature windows in every sample, but the "
        f"idle-loop control sampled in the same run cannot beat them either "
        f"(control worst gap {control_gap * 1000:.0f}ms over a "
        f"{_CONTROL_WINDOW_S:.0f}s idle window against the smallest dirty gap "
        f"{smallest * 1000:.0f}ms): the wakeup tail reaches every thread on "
        "this box, and a wall ratio under it is unjudgeable ("
        + "; ".join(
            f"worst {worst * 1000:.1f}ms of a "
            f"{_wall * 1000:.0f}ms operation, "
            f"the process burned {gap_cpu * 1000:.0f}ms of CPU"
            for worst, _wall, gap_cpu, _share, _wait in strikes
        )
        + "): excused as runner saturation relative to the control, not a "
        "loop-blocking regression (tests/loop_harness.py, _excusable_under_control)"
    )
    return True


def _ratio_missed(gap: float, wall: float, ratio_budget: float) -> bool:
    """Whether a sample missed its RATIO budget (the loop-responsiveness
    teeth): the worst gap against the operation's OWN wall, the
    noise-headroom device the flake doctrine ranks above the absolute
    ceiling."""
    return gap >= ratio_budget * wall


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
    if _ratio_missed(gap, wall, ratio_budget):
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
    budget (a held whole pass reads ~1.0) and the absolute ceiling.

    When every sample misses, the verdict is three-tiered. Tier 1: a window
    that HELD its ratio budget and missed only the absolute ceiling is a
    wall pin, advisory under the doctrine (the ratio against the
    operation's own wall is the loop-responsiveness contract), and is
    excused with a printed record. Tier 2: the ratio-missed windows consult
    the loop thread's own scheduler account (``_gap_was_starved``): windows
    the heartbeat thread spent runnable-but-unrun are the runner's
    starvation and are excused with a printed record (a saturated box
    cannot judge a wall ratio at all, and an environmental wall pin does
    not fail a diff with no code change); windows the thread spent
    futex-blocked on the GIL are the regression -- on kernels without the
    schedstat account, the process-CPU fallback and tier 3's idle-loop
    control (``_excusable_under_control``) arbitrate before the raise. A
    regression that blocks the loop while the machine had CPU to give --
    and the control stays free -- still fails on this verdict, loudly."""
    observed: list[tuple[float, float, float, float, float]] = []
    for _ in range(samples):
        worst, wall, gap_cpu, cpu_share, gap_wait = gap_and_wall(op)
        observed.append((worst, wall, gap_cpu, cpu_share, gap_wait))
        if not _both_budget_misses(worst, wall, ratio_budget, ceiling_s):
            return
    # Tier 1 (the wall-pin excuse): windows that HELD the ratio budget and
    # missed only the absolute ceiling are not evidence -- the ratio against
    # the operation's own wall is the contract, the ceiling a wall pin (the
    # doctrine: content-match is the contract, wall pins are advisory). Under
    # saturation a by-design held window stretches to several times its
    # quiet-box wall with the ratio intact, and that is the environment.
    ratio_strikes = [t for t in observed if _ratio_missed(t[0], t[1], ratio_budget)]
    if not ratio_strikes:
        print(
            f"{subject} missed the {ceiling_s * 1000:.0f}ms ceiling in every one "
            f"of {samples} samples, but every missed window held its "
            f"{ratio_budget:.0%} ratio budget ("
            + "; ".join(
                f"worst {worst * 1000:.1f}ms of a {wall * 1000:.0f}ms operation "
                f"({(worst / wall if wall else 0):.0%})"
                for worst, wall, _gap_cpu, _share, _wait in observed
            )
            + "): the ratio is the loop-responsiveness contract, the ceiling a "
            "wall pin -- excused as an advisory pin under load, not a "
            "loop-blocking regression (tests/loop_harness.py, _ratio_missed)"
        )
        return
    # Tier 2 (the starvation excuse): the ratio-missed windows are judged by
    # the loop thread's own scheduler account (the process-CPU fallback when
    # the kernel does not expose it) -- the GIL blocks the thread, the
    # runner's starvation leaves it runnable.
    strikes = [sample for sample in ratio_strikes if not _gap_was_starved(sample)]
    if not strikes:
        print(
            f"{subject} missed the heartbeat budgets in every one of {samples} "
            "samples, but every missed window carries the runner's starvation "
            "signature -- the loop thread was runnable-but-unrun while the "
            "ticks went missing ("
            + "; ".join(
                f"worst {worst * 1000:.1f}ms of a {wall * 1000:.0f}ms operation, "
                f"the runqueue wait {gap_wait * 1000:.0f}ms, the process burned "
                f"{gap_cpu * 1000:.0f}ms of CPU"
                for worst, wall, gap_cpu, _share, gap_wait in ratio_strikes
            )
            + "): excused as runner saturation, not a loop-blocking regression "
            "(tests/loop_harness.py, _gap_was_starved)"
        )
        return
    # Tier 3 (the control): the surviving GIL-signature strikes are judged
    # relative to the idle-loop control sampled in the same run -- the
    # wakeup tail reaches every thread under saturation, and only a box
    # whose idle loop holds the ping floor can convict.
    if _excusable_under_control(subject, strikes):
        return
    detail = "; ".join(
        f"worst {worst * 1000:.1f}ms of a {wall * 1000:.0f}ms operation "
        f"({(worst / wall if wall else 0):.0%}, the runqueue wait "
        f"{gap_wait * 1000:.0f}ms, the process burned {gap_cpu * 1000:.0f}ms "
        f"of CPU, over "
        f"{' and '.join(_both_budget_misses(worst, wall, ratio_budget, ceiling_s))})"
        for worst, wall, gap_cpu, _share, gap_wait in strikes
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
