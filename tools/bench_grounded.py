"""Honest measurement: tors.is_grounded against the pure-stdlib composition
a Python developer writes without it — ``claim in source`` for the exact
path, and a windowed ``difflib.SequenceMatcher`` ratio scan for the fuzzy
path (the same windowing recipe the crate documents: same-length windows,
stride ``len(claim) // 2``, truncated tail window, early exit once the best
ratio reaches the threshold; no exact-containment floor and no refinement
pass — neither exists in the naive composition, and the verbatim cells
price exactly that difference).

Correctness is parity-gated against the FULL contract model —
``tests/reference.py``'s ``reference_is_grounded_fuzzy`` (floor, windowing,
bounded refinement, all modeled in pure Python with an LCS oracle) — not
against the naive composition: tors's refinement detects near-matches the
naive scan misses by design, so the naive lane would report those as
divergences. The parity gate runs tors vs the contract model at thresholds
1.0 / 0.85 / 0.6 plus the exact lane vs ``in``; a cell whose verdicts
differ is reported as divergent with its first differing claim and is NOT
timed. --check runs only that verification pass.

Three corpus shapes, each five deterministic claims against prose sources
at three sizes (4 KiB, 64 KiB, 256 KiB):

- verbatim: claims present word-for-word in the source — the
  exact-containment floor's lane (tors returns at the floor; the naive
  composition pays the full window scan).
- near: each claim with one transposed letter pair, deliberately NOT a
  substring — the windowed-ratio lane; tors's refinement also finds these
  wherever they straddle the coarse grid.
- unrelated: same-length claims built from words the corpus never contains
  — the full-scan (nothing matches) DoS shape.

Timing is time.perf_counter min-of-N over five-claim passes, N sized so a
cell accumulates at least 50 ms (at least 2 passes, GC disabled while
measuring), reported as per-call microseconds (one call = one
is_grounded(claim, source)). autojunk never engages: difflib only applies
it to sequences of 200+ chars, and no window ever exceeds the claim length.
Run with `uv run --no-sync python tools/bench_grounded.py`.
"""

from __future__ import annotations

import argparse
import difflib
import gc
import math
import sys
import time
from functools import partial
from pathlib import Path

import tors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from reference import reference_is_grounded_fuzzy  # noqa: E402

KIB = 1024
SIZES = ((4 * KIB, "4 KiB"), (64 * KIB, "64 KiB"), (256 * KIB, "256 KiB"))
SHAPES = ("verbatim", "near", "unrelated")
CLAIMS_PER_CELL = 5
THRESHOLDS = (1.0, 0.85, 0.6)
FUZZY_THRESHOLD = 0.85
MIN_CELL_SECONDS = 0.050
MAX_PASSES = 4000

# The corpus sentences: plain deterministic prose, the claim phrases carried
# verbatim (the verbatim shape draws them unchanged; the near shape
# transposes one letter pair inside each).
_CLAIM_SENTENCE = (
    "The quarterly oil sample interval for field outages was adjusted after "
    "the bushing torque specifications changed."
)
_SENTENCES = (
    _CLAIM_SENTENCE,
    "Maintenance windows now close within fourteen days of the adjustment.",
    "Field crews retested the bushing torque specifications after every outage.",
    "Sample intervals follow the field outage log without exception.",
    "The adjustment record lists every changed specification for review.",
)

# The verbatim claims: five phrases the corpus carries word-for-word.
_VERBATIM_CLAIMS = (
    "the bushing torque specifications changed",
    "quarterly oil sample interval",
    "windows now close within fourteen days",
    "crews retested the bushing torque specifications",
    "intervals follow the field outage log",
)

# One transposed letter pair per verbatim claim: near-matches, NOT substrings.
# The transposition is guaranteed (an adjacent DIFFERING pair at or past the
# midpoint), so every near claim provably differs from its verbatim original
# and cannot short-circuit tors's exact-containment floor.


def _transposed(claim: str) -> str:
    chars = list(claim)
    i = max(1, len(chars) // 2)
    while i + 1 < len(chars) and chars[i - 1] == chars[i]:
        i += 1
    assert chars[i - 1] != chars[i], f"no differing pair at/past midpoint: {claim!r}"
    chars[i - 1], chars[i] = chars[i], chars[i - 1]
    return "".join(chars)


_NEAR_CLAIMS = tuple(_transposed(claim) for claim in _VERBATIM_CLAIMS)

# Same-length unrelated claims over words the corpus never contains.
_UNRELATED_CLAIMS = (
    "uniform victor whiskey xray yankee zulu xray yankee",
    "uniform victor whiskey victor uniform zulu yankee",
    "whiskey xray yankee zulu uniform victor xray zulu",
    "victor uniform zulu yankee whiskey xray victor yankee",
    "yankee zulu uniform whiskey victor xray zulu whiskey",
)


def _repeat_to(target_bytes: int) -> str:
    unit = " ".join(_SENTENCES)
    return (unit + " ") * max(1, target_bytes // len(unit))


def _claims(shape: str) -> list[str]:
    return {
        "verbatim": list(_VERBATIM_CLAIMS),
        "near": list(_NEAR_CLAIMS),
        "unrelated": list(_UNRELATED_CLAIMS),
    }[shape]


def upstream_fuzzy(claim: str, source: str, threshold: float) -> bool:
    """The naive stdlib composition (the TIMING baseline): difflib ratios
    over the documented windows, early exit at the threshold. No
    exact-containment floor, no refinement pass — neither exists in the
    composition a Python developer writes, and the verbatim/near cells
    price exactly what tors's floor and refinement buy."""
    if not claim:
        return True
    m, n = len(claim), len(source)
    if n <= m:
        return difflib.SequenceMatcher(None, claim, source).ratio() >= threshold
    stride = max(m // 2, 1)
    best = 0.0
    start = 0
    while True:
        end = min(start + m, n)
        best = max(best, difflib.SequenceMatcher(None, claim, source[start:end]).ratio())
        if best >= threshold or end == n:
            break
        start += stride
    return best >= threshold


def verify_cell(shape: str, label: str, source: str, claims: list[str]) -> str | None:
    """tors vs the full contract model (and `in`) on every claim; None = all
    agree."""
    # The shapes' own semantics, self-checked: verbatim claims are present
    # word-for-word, near and unrelated claims are absent (a near claim that
    # accidentally survived its transposition verbatim would silently turn
    # the cell into a floor lane and pollute the timing).
    for claim in claims:
        if (claim in source) != (shape == "verbatim"):
            return (
                f"{shape} / {label}: BAD SHAPE (claim {claim!r} presence"
                f" {(claim in source)!r} contradicts the {shape} shape)"
            )
    for threshold in THRESHOLDS:
        for claim in claims:
            got = tors.is_grounded(claim, source, fuzzy=True, threshold=threshold)
            want = reference_is_grounded_fuzzy(claim, source, threshold)
            if got != want:
                return (
                    f"{shape} / {label}: DIVERGENT (threshold {threshold}, claim"
                    f" {claim!r}: tors={got}, contract model={want})"
                )
    exact = tors.is_grounded(claims[0], source)  # the fuzzy=False lane
    want = claims[0] in source
    if exact != want:
        return f"{shape} / {label}: DIVERGENT (exact lane: tors={exact}, `in`={want})"
    return None


def _py_in(claim: str, source: str) -> bool:
    return claim in source


def time_cell(source: str, claims: list[str], fn) -> float:
    """Per-call microseconds: min-of-N five-claim passes, N sized for >= 50 ms."""
    fn(claims[0], source)  # warmup: extension init
    gc_on = gc.isenabled()
    gc.disable()
    try:
        best = math.inf
        total = 0.0
        passes = 0
        while True:
            start = time.perf_counter()
            for claim in claims:
                fn(claim, source)
            elapsed = time.perf_counter() - start
            best = min(best, elapsed)
            total += elapsed
            passes += 1
            if (total >= MIN_CELL_SECONDS and passes >= 2) or passes >= MAX_PASSES:
                break
    finally:
        if gc_on:
            gc.enable()
    return best / len(claims) * 1e6


def _fmt_us(value: float) -> str:
    if value >= 10_000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _fmt_ratio(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}x"
    if value >= 10:
        return f"{value:.1f}x"
    return f"{value:.2f}x"


_TABLE_HEADER = (
    f"{'shape':<10} {'size':>7} {'src chars':>9} {'tors exact':>11} {'py in':>10} "
    f"{'exact x':>8} {'tors fuzzy':>11} {'difflib':>11} {'fuzzy x':>8}  check"
)


def _row(shape: str, label: str, chars: int, times: list[float] | None) -> str:
    if times is None:
        return (
            f"{shape:<10} {label:>7} {chars:>9,} {'-':>11} {'-':>10} {'-':>8}"
            f" {'-':>11} {'-':>11} {'-':>8}  divergent"
        )
    tors_exact, py_in, tors_fuzzy, upstream = times
    return (
        f"{shape:<10} {label:>7} {chars:>9,} {_fmt_us(tors_exact):>11} {_fmt_us(py_in):>10} "
        f"{_fmt_ratio(py_in / tors_exact):>8} {_fmt_us(tors_fuzzy):>11} {_fmt_us(upstream):>11} "
        f"{_fmt_ratio(upstream / tors_fuzzy):>8}  ok"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-head benchmark: tors.is_grounded vs the stdlib composition."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify tors-vs-stdlib verdict parity per cell (no timing)",
    )
    args = parser.parse_args()
    context = (
        f"tors vs stdlib (in / difflib) | python {sys.version.split()[0]} | "
        f"{CLAIMS_PER_CELL} deterministic claims per cell | per-call us, min-of-N passes"
    )
    print(context)
    if args.check:
        print(
            "parity check: tors vs the contract model on every cell"
            " (thresholds 1.0/0.85/0.6 + exact)"
        )
        divergent = 0
        for shape in SHAPES:
            for target, label in SIZES:
                source = _repeat_to(target)
                report = verify_cell(shape, label, source, _claims(shape))
                if report is None:
                    print(f"{shape} / {label}: ok ({CLAIMS_PER_CELL} claims x thresholds)")
                else:
                    divergent += 1
                    print(report)
        print(f"summary: {len(SHAPES) * len(SIZES)} cells, {divergent} divergent")
        return
    print(_TABLE_HEADER)
    print("-" * len(_TABLE_HEADER))
    for shape in SHAPES:
        for target, label in SIZES:
            source = _repeat_to(target)
            claims = _claims(shape)
            report = verify_cell(shape, label, source, claims)
            chars = len(source)
            if report is not None:
                print(_row(shape, label, chars, None))
                print(report)
                continue
            times = [
                time_cell(source, claims, tors.is_grounded),
                time_cell(source, claims, _py_in),
                time_cell(
                    source, claims, partial(tors.is_grounded, fuzzy=True, threshold=FUZZY_THRESHOLD)
                ),
                time_cell(source, claims, partial(upstream_fuzzy, threshold=FUZZY_THRESHOLD)),
            ]
            print(_row(shape, label, chars, times))
    print(
        "exact x = py 'in' us/call / tors exact us/call; fuzzy x = the naive"
        " difflib composition's us/call / tors fuzzy us/call; check = verdicts equal"
        " to the contract model at every threshold. verbatim fuzzy: tors returns at"
        " the exact-containment floor while the naive composition scans every"
        " window; near: tors's refinement also finds straddled near-matches the"
        " naive scan misses."
    )


if __name__ == "__main__":
    main()
