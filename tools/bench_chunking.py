"""Honest measurement: the chunking family's document-scale cost, the
cells issue #22 reported plus the sibling unit-count chunkers that turned
out to carry the same pathology, with each cell's PEAK RSS measured
alongside its wall time.

Cells (12 MiB unless noted, the issue's own scale):

- hierarchical-custom-miss: ``chunk_hierarchical`` over a degenerate
  single-character run with a never-matching custom separator list and a
  whole-document budget -- the pure "per-call machinery" cost: one count
  pass plus one scan pass, no grapheme structure at all after the fix.
- hierarchical-custom-2000: the same input at a 2000-codepoint budget --
  every window falls to the raw cut, the lazily-built grapheme index
  (ASCII fast path) plus ~6.3K hard cuts.
- hierarchical-default-2000: real prose, the default paragraph -> word
  hierarchy at a 2000-codepoint budget -- the segmentation walks the
  function exists to provide, plus the cut filter and chunk walk.
- by-words / by-sentences / by-paragraphs: the unit-count chunkers over
  the same prose at their natural window sizes.
- by-lines: the merge-free sibling at 50 lines (mirroring the criterion
  group's corpus-derived ``by_lines_50`` budget): one fused scan, no
  segmentation walk and no boundary index.
- walks: the component scans (grapheme/word/sentence count) as reference
  rows, the "what the residual IS" numbers.

Wall timing is time.perf_counter min-of-N inside a FRESH CHILD PROCESS
per cell (N sized so a cell accumulates >= 50 ms, at least 2 passes, at
most 20, GC disabled while measuring, one warm-up pass before); each
child hands its best time back through a temp file and the parent reaps
it with os.wait4, so every row's peak RSS is that child's own high-water
mark -- the interpreter plus the corpus string plus one call's transient
allocations, with the corpus-only baseline row subtracted in the reported
number. Fresh child per cell is what makes the RSS number per-call: a
long-lived parent's high-water mark would carry every earlier cell's
transients. Run with `uv run --no-sync python tools/bench_chunking.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

MIB = 1024 * 1024
MIN_CELL_SECONDS = 0.050
MAX_PASSES = 20

# The prose recipe: sentences and blank-line paragraph gaps, the shape the
# default hierarchy exists for. Deterministic, no filesystem dependency.
# (Not tests/reference.py's corpus: this tool runs outside pytest's
# pythonpath, and the cell inputs are internal to the measurement.)
_PROSE_UNIT = (
    "The quarterly oil sample interval for field outages was adjusted "
    "after the bushing torque specifications changed. Maintenance windows "
    "now close within fourteen days. "
) * 4 + "\n\n"

# name, corpus kind, size, callable source (evaluated in the child)
CELLS: list[tuple[str, str, int, str]] = [
    (
        "hierarchical-custom-miss 12MiB",
        "q",
        12 * MIB,
        "lambda s: tors.chunk_hierarchical(s, 12 * 1024 * 1024, ['xyz'])",
    ),
    (
        "hierarchical-custom-miss 4MiB",
        "q",
        4 * MIB,
        "lambda s: tors.chunk_hierarchical(s, 4 * 1024 * 1024, ['xyz'])",
    ),
    (
        "hierarchical-custom-miss 1MiB",
        "q",
        1 * MIB,
        "lambda s: tors.chunk_hierarchical(s, 1 * 1024 * 1024, ['xyz'])",
    ),
    (
        "hierarchical-custom-2000 12MiB",
        "q",
        12 * MIB,
        "lambda s: tors.chunk_hierarchical(s, 2000, ['xyz'])",
    ),
    (
        "hierarchical-default-2000 12MiB",
        "prose",
        12 * MIB,
        "lambda s: tors.chunk_hierarchical(s, 2000)",
    ),
    (
        "hierarchical-default-2000-overlap 12MiB",
        "prose",
        12 * MIB,
        "lambda s: tors.chunk_hierarchical(s, 2000, overlap=200)",
    ),
    (
        "by-words 200 12MiB",
        "prose",
        12 * MIB,
        "lambda s: tors.chunk_by_words(s, 200)",
    ),
    (
        "by-sentences 10 12MiB",
        "prose",
        12 * MIB,
        "lambda s: tors.chunk_by_sentences(s, 10)",
    ),
    (
        "by-paragraphs 5 12MiB",
        "prose",
        12 * MIB,
        "lambda s: tors.chunk_by_paragraphs(s, 5)",
    ),
    (
        "by-lines 50 12MiB",
        "prose",
        12 * MIB,
        "lambda s: tors.chunk_by_lines(s, 50)",
    ),
    ("walk: grapheme_count 12MiB", "prose", 12 * MIB, "tors.grapheme_count"),
    ("walk: word_count 12MiB", "prose", 12 * MIB, "tors.word_count"),
    ("walk: sentence_count 12MiB", "prose", 12 * MIB, "tors.sentence_count"),
]

_CHILD = r"""
import gc, sys, time
import tors

kind, size, line, out_path, min_seconds, max_passes = sys.argv[1:7]
size, min_seconds, max_passes = int(size), float(min_seconds), int(max_passes)
if kind == "prose":
    unit = ("The quarterly oil sample interval for field outages was adjusted "
            "after the bushing torque specifications changed. Maintenance windows "
            "now close within fourteen days. ") * 4 + "\n\n"
    corpus = unit * (size // len(unit) + 1)
else:
    corpus = "q" * size
fn = eval(line)
fn(corpus)  # warm-up, outside the measured passes
gc.disable()
best = float("inf")
passes = 0
while passes < 2 or (passes < max_passes and passes * best < min_seconds):
    t0 = time.perf_counter()
    fn(corpus)
    best = min(best, time.perf_counter() - t0)
    passes += 1
gc.enable()
with open(out_path, "w") as f:
    f.write(f"{best * 1000:.2f}\n")
"""


def _run_cell(kind: str, size: int, line: str) -> tuple[float, int]:
    """One fresh child: (min-of-N wall ms, that child's peak RSS in KiB).
    The timing crosses via a temp file and the child is reaped with
    os.wait4 (Popen's own wait machinery is bypassed by marking the
    already-reaped process done)."""
    with tempfile.NamedTemporaryFile("r", suffix=".bench", delete=True) as tmp:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _CHILD,
                kind,
                str(size),
                line,
                tmp.name,
                str(MIN_CELL_SECONDS),
                str(MAX_PASSES),
            ],
            stdout=subprocess.DEVNULL,
        )
        _, status, rusage = os.wait4(proc.pid, 0)
        # Reaped via wait4: tell Popen so its finalizer does not wait again.
        proc.returncode = os.waitstatus_to_exitcode(status)
        if proc.returncode != 0:
            raise RuntimeError(f"cell child failed: {kind}/{size} exit {proc.returncode}")
        best_ms = float(tmp.read().strip())
    return best_ms, int(rusage.ru_maxrss)


def main() -> None:
    # One no-call baseline child per DISTINCT (kind, size) corpus, so each
    # cell's "over baseline" number subtracts a same-corpus high-water
    # mark (a 1 MiB corpus child is naturally smaller than a 12 MiB one).
    baselines: dict[tuple[str, int], int] = {}
    for _, kind, size, _ in CELLS:
        if (kind, size) not in baselines:
            _, rss = _run_cell(kind, size, "lambda s: None")
            baselines[(kind, size)] = rss
    print(f"{'cell':44s} {'wall (min-of-N)':>16s} {'peak RSS over corpus':>20s}")
    print("-" * 84)
    for name, kind, size, line in CELLS:
        ms, rss = _run_cell(kind, size, line)
        over = (rss - baselines[(kind, size)]) / 1024
        print(f"{name:44s} {ms:>13.2f} ms {over:>17.1f} MiB")


if __name__ == "__main__":
    main()
