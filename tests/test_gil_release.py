"""THE GIL-release claim, as a test: ``tors.finalize``'s whole native pass runs under
``py.detach``, so a thread running it leaves the event loop schedulable at heartbeat
granularity while megabytes of text are transformed and hashed.

Methodology ported from the specification's loop-safety harness: a heartbeat
task appends monotonic ticks every 10ms while the operation runs; the assertion is on the
worst tick gap, and a sample is clean only when that gap is under BOTH budgets: the
ratio budget (worst gap < a fraction of the operation's wall) and a generous absolute
ceiling. The test takes up to 3 samples and passes on the first clean one. A GIL-held
whole-text pass blocks the loop in EVERY sample (the loop's thread cannot acquire the GIL
while the C call holds it, so worst gap ~= wall, ratio ~= 1.0, every time), while
transient whole-process CPU starvation on a shared runner reads as a block only in the
sample it hits, so a single starved sample is retried rather than failing the test, and
the multi-sample design is load-robust without weakening the red side (a genuine
regression dirties every sample, so it still fails). This file uses no
async pytest plugin (tors has none and needs none): each test owns its loop via
``asyncio.run``.

Measured on the dev box this test was written on (WSL2, 28 logical cores, CPython 3.12.7;
5 samples per cell, prose and decomposed corpora from ``tests/reference.py``; the prose
cells at ambient load ~5, the decomposed cells (added with the corpus-kind parametrize)
at ambient load 1.6-4.0; walls as the corrected harness measures them, the operation's
own wall, captured at op completion before the heartbeat join, so no drain ping is
absorbed):

- ``tors.finalize`` via ``asyncio.to_thread``: worst gap / wall:
  12 MiB: 10.3-14.0ms of 109-147ms walls (ratio 0.082-0.095);
  32 MiB: 16.3-20.7ms of 353-372ms walls (ratio 0.044-0.058).
  The gap is the ~10ms ping floor plus pyo3's return marshalling of the two output
  strings (O(output), size-coupled: ~0-4ms over the floor at 12 MiB, ~6-11ms at 32 MiB),
  i.e. the marshalling band, not the transform. The 12/32 MiB pair proves the residual stays
  in that band as the corpus grows 2.7x.
- ``tors.finalize`` on the decomposed (non-ASCII) corpus, the first-call residue: the
  first sample pays pyo3's one-time O(input) UTF-8 materialization of the argument under
  the GIL (``&str`` extraction builds and caches the UTF-8 copy); every later sample on
  the same object borrows that copy zero-copy, and the prose cells' first calls sit in
  the same band as their cached samples (the ASCII borrow is a zero-copy alias, so
  there is no materialization to pay):
  12 MiB: first-call gap 13.9-15.6ms of 149-158ms walls (ratio 0.088-0.105) against a
  10.2-14.9ms cached band: the GIL-held materialization itself is ~5-8ms (a fresh-object
  ``str.encode("utf-8")`` of the same corpus, the same conversion, costs 7.7-8.6ms);
  32 MiB: first-call gap 37.6-44.1ms of 403-406ms walls (ratio 0.093-0.110) against a
  18.6-31.7ms cached band: materialization ~16-24ms (encode anchor 21.4-23.5ms).
  Even the 32 MiB first call is clean on both budgets (worst ratio 0.110 vs the 0.30
  budget, worst gap 44.1ms vs the 100ms ceiling), so these cells pass on sample 1; the
  pass-on-first-clean design also tolerates a loaded-box first sample, since samples 2+
  sit in the cached band. The first call's WALL is ~20ms slower at 12 MiB (~30ms at
  32 MiB) than cached calls, but that extra is non-GIL first-call warm-up of the
  detached pass, so the loop never sees it; the gap delta and the encode
  anchor, not the wall delta, are the materialization's measure.
- The pure-Python reference finalize (``reference_finalize``) in the SAME to_thread
  placement, the red side: same work, same thread, but its whole-text GIL-held C calls
  (``unicodedata.normalize``, ``str.replace`` ×2, ``re.sub`` ×2, ``encode``) each hold
  the GIL for their full duration; only ``hashlib.sha256`` releases it (CPython drops
  the GIL above 2047-byte updates), which cannot save the loop because the worst gap is
  set by the largest ``re.sub``:
  12 MiB: 91.6-107.5ms gaps (ratio 0.591-0.613): fails the 0.30 ratio budget in every
  sample; 32 MiB: 249.9-271.5ms gaps (ratio 0.539-0.555), also blowing the 100ms ceiling
  by ~2.5-2.7x. Run inline on the loop instead, it is ratio ~1.0 (151-161ms at 12 MiB,
  460-505ms at 32 MiB).
- Budget margins: 0.30 sits ~3.2x above tors's worst measured ratio and ~1.8x below the
  reference's best; the 100ms ceiling sits ~4.8x above tors's worst measured gap (the
  32 MiB gap, under ambient load ~5; at 12 MiB it is ~7x).
- What this suite pins: BOTH bands of the crate GIL model (src/lib.rs), the ASCII
  marshalling band via the prose cells and the non-ASCII first-call materialization via
  the decomposed cells, not just the ASCII band.

(quick-check fast paths + identity returns + the streaming iterator), and what
they did to these cells, measured on the dev box, ambient load 4.4-5.7 unless noted:

- The identity path's band (the zero-cost lane): on ALREADY-NORMALIZED 12 MiB prose
  (``reference_normalize(prose(...))``, the pipeline's own output), ``tors.normalize``
  walls collapse to ~3-4ms and ``tors.finalize`` to ~9-10ms (the quick-check scan plus,
  for finalize, SHA-256 over the borrowed input, measured 2.89ms/7.72ms min-of-5 at
  load 4.6; the build measured 123.0ms/126.3ms at load 12.1, min-of-N robust on
  this box). Those walls sit AT or UNDER the 10ms ping floor, so their gap/wall ratios
  (~1.2-3.0) are the documented sub-ping artifact: the identity-path cells assert the
  100ms ceiling only (>=9x margin), exactly the b64 12 MiB precedent.
- The QC-Yes-but-scan-dirty prose corpus keeps paying the scan but skips the NFC pass:
  ``finalize`` prose walls shrink ~3x (42-46ms at 12 MiB, 111-123ms at 32 MiB,
  measured) while the GIL-held residue (the O(output) marshalling) is unchanged,
  so the residue/wall FRACTION structurally rose at 12 MiB: measured ratios 0.27-0.34.
  The shared 0.30 budget is therefore no longer attainable with margin on that one
  cell, and a budget that flakes is worse than a budget derived from the band: the
  12 MiB prose cells (finalize, finalize_utf8) move to a bespoke 0.60 ratio budget,
  ~1.8x above the worst measured ratio (0.34) and ~40% below the ~1.0 a detach
  regression shows (the whole 42-45ms transform held would fail it by far; the 100ms
  ceiling holds ~2.3x margin). The 32 MiB prose cells stay on the shared budget
  (measured 0.09-0.17).
- The D-forms' full pass lost its corpus: plain decomposed prose carries NO
  compatibility mappings, so under NFKC/NFKD it now quick-checks Yes and comes back
  as the input object; ``nfkd(decomposed)`` walls collapse from 90-98ms to 10-23ms
  (measured; the residue is the str-in first-call materialization plus the quick-check
  scan). The forms cells therefore moved to the NEW ``compat`` corpus (decomposed
  accents + U+FB01 ligature + U+FF10 fullwidth digit per unit; the compatibility
  mappings keep the quick check at No), whose measured bands sit inside BOTH shared
  budgets (nfc 12 MiB 16.8-17.4ms of 127-145ms walls, 0.10-0.13; nfkd 12 MiB
  14.7-17.8ms of 87-104ms, 0.14-0.20; nfc 32 MiB 36.3-40.1ms of 346-391ms, 0.10-0.12;
  nfkd 32 MiB 35.7-38.2ms of 240-263ms, 0.15-0.16). A separate ceiling-only cell pins
  the D-form FAST PATH on the decomposed corpus (walls at the ping floor → ratio
  artifact, the b64 12 MiB precedent again).
- The word_bounds streaming answer: ``tors.word_bounds_iter`` full drain at 12 MiB
  (3.67M bounds) measured worst gaps 15.4ms of 327-358ms walls (0.04-0.05), inside
  both SHARED budgets, against the list shape's structurally-unattainable 428-567ms
  band. The wall surprise, recorded: the full drain measured 347ms min-of-3
  against the list API's 724ms in the same process; the per-``__next__`` path is
  FASTER per bound than the list conversion (~47ns vs ~150ns per bound by
  subtraction of the shared ~175ms detached core), so the iterator wins on both axes
  at this size; the list API stays for small inputs and one-shot batch work.

Corpus sizes start at 12 MiB because the ratio only means something when the wall exceeds
the 10ms ping floor by a wide margin (at 1 MiB the whole finalize is ~10ms: gap ~= ping
floor ~= wall, ratio ~1.0 by artifact, measured, not evidence of blocking).

bytes-in cells (``finalize_utf8``, ``b64_encode_bytes``; measured on the same box,
ambient load 4.3-5.6, 5 samples per cell, corpora from ``reference.corpus_utf8``, the
str corpora rendered to UTF-8 bytes):

- ``tors.finalize_utf8`` via ``asyncio.to_thread``: the no-materialization claim, on
  both corpus kinds: prose 12 MiB 10.3-13.0ms of 113-143ms walls (0.08-0.10); decomposed
  12 MiB 10.3-13.0ms of 116-131ms (0.09-0.10); prose 32 MiB 13.5-20.7ms of 345-370ms
  (0.04-0.06); decomposed 32 MiB 17.6-26.7ms of 350-405ms (0.04-0.07). The decomposed
  cells' SAMPLE-1 gaps sit in the same band as samples 2+: the one-time O(input) UTF-8
  materialization that the str-in ``finalize`` cells pay on their first sample
  (13.9-15.6ms at 12 MiB, 37.6-44.1ms at 32 MiB, above) does not exist for bytes-in:
  pyo3's ``&[u8]`` extraction is a zero-copy borrow of the immutable buffer (verified in
  pyo3 0.29's source; see src/lib.rs's GIL model). The 12/32 MiB pair proves the residue
  stays in the marshalling band as the corpus grows.
- ``tors.b64_encode_bytes`` via ``asyncio.to_thread``: the residue is the marshalling
  of the 4/3x-sized ASCII output: 12 MiB worst gaps 10.1-11.2ms against 5-9ms walls
  (wall under the ping floor: the ratio there is the sub-ping artifact, hence a ceiling-only
  cell); 32 MiB 27.5-29.3ms of 48-50ms (0.57-0.58) against the b64-specific 0.80 ratio
  budget; 48 MiB 37.0-48.8ms of 68-80ms (0.54-0.61) and 96 MiB 67.2-78.3ms of
  139-161ms (0.46-0.50), measured and unasserted (ceiling margin ~1.3x under load).
  The b64 budget derivation: 0.80 is ~1.3x above the worst measured ratio (0.61) and
  ~20% below the ~1.0 a GIL-held pass shows in every sample; the shared 0.30 is
  structurally unattainable for b64 because the fast encode makes the marshalling the
  dominant share of the wall (gap/wall -> marshal/(encode+marshal) ~ 0.6-0.75 as size
  grows, measured). The red side: `base64.b64encode` holds the GIL, with 122.3-127.6ms
  held at 96 MiB, reproducing the ~150ms@100MB GIL-held b64encode observation that
  motivated the function; at 12/32 MiB the red expression measures ratio 0.63-0.66
  (window-dependent at 12) and passes those cells' budgets; the red-side cell below
  records the structural reason and asserts the sizes where the budgets do
  discriminate.

diff_opcodes cells (the 12 MiB near-identical and shuffled pairs from
``reference.diff_pair_near_identical`` / ``reference.diff_pair_shuffled``;
measured on the dev box, ambient load 5.3-6.6, 3 samples per cell):

- Near-identical (the six scattered line edits -> 235 opcodes): worst gaps
  10.5-11.2ms of 76-88ms walls (ratio 0.12-0.14): the ping floor plus the two
  zero-copy ASCII argument borrows and a ~0.05ms 235-tuple marshalling; inside
  both SHARED budgets with ~2x ratio margin. The native diff over a
  near-identical corpus is fast enough that the ping floor dominates the
  ratio; if walls ever collapse toward the ping floor on a quiet box, this
  cell follows the b64 12 MiB ceiling-only recalibration precedent rather
  than pretending the 0.30 ratio is meaningful there.
- Shuffled (103,421 opcodes; the O(ops) marshalling band made visible, the
  ``word_bounds`` list-shape class): worst gaps 20.4-26.3ms of 1554-1626ms
  walls (0.013-0.017). The marshalling itself is the delta over the
  near-identical cell's floor band: ~10-15ms for 103,421 5-tuples, ~0.1-0.15µs
  per opcode (one tuple plus up to four fresh PyLongs each; the four tag
  strings are constructed once per call and shared by reference into every
  tuple, difflib's own interned-tag shape; a per-op PyString would
  multiply the band several-fold). The cell pins the band with a bespoke 60ms
  gap ceiling (~2.3x above the measured worst, catching a several-fold
  marshalling blowout that the shared 100ms ceiling would let through at
  this op count) plus the shared 0.30 ratio (a detach regression holds the
  whole ~1.6s wall). The op count is REPORTED, not asserted: it is the
  algorithm's business (similar's bounded search trades minimality for speed
  on hard inputs, and a crate bump may change the count without changing
  validity).

find_patterns cells (the sparse and dense 12 MiB shapes from
``reference.SEARCH_SPARSE_PATTERNS`` / ``SEARCH_DENSE_PATTERNS``; measured on
the dev box, ambient load 6.9-7.3, 3 samples per cell):

- Sparse (the diff near-identical pair's edited corpus, the terminology-scan
  shape: ``"monthly"`` occurs exactly once, the other two terms never):
  worst gaps 10.3-10.7ms of 6-7ms walls: the ping floor plus ~nothing (one
  3-tuple). The pure scan is ~2 GiB/s, so the WALL sits under the 10ms ping
  floor and any gap/wall ratio is the suite's documented sub-ping artifact;
  the cell asserts the 100ms ceiling only (the b64 12 MiB / utf8_is_valid
  precedent). A 6ms scan held or released is invisible under the
  floor either way, so this cell cannot discriminate a detach regression;
  the dense cell below carries that (a held dense pass shows ratio ~1.0
  against its ~215ms wall); the sparse cell pins that the no-match scan
  leaves the loop at the floor at all.
- Dense (17 prose words over the plain 12 MiB prose corpus, 1,284,724
  matches; the O(matches) marshalling band made visible, the
  ``word_bounds`` list-shape class): worst gaps 171.0-174.7ms of 214-224ms
  walls (ratio 0.76-0.80). The marshalling itself is the delta over the
  sparse cell's floor band: ~160-165ms for 1.28M 3-tuples, ~0.13µs per match,
  the same per-element band diff_opcodes measures per opcode. The cell
  pins the band with a bespoke 400ms gap ceiling (~2.3x above the measured
  worst; the word_bounds-precedent derivation, catching a several-fold
  marshalling blowout the shared 100ms ceiling is not sized for at this
  match count) plus a 0.90 ratio budget as the detach discriminator. Why
  0.90 and not word_bounds' 0.85: this cell's band sits structurally
  HIGHER than word_bounds' 0.72-0.74, because the search core is ~3.5x
  faster than segmentation, so the marshalling is a larger share of the
  wall; 0.90 sits ~1.125x above the worst measured ratio and ~10% below
  the ~1.0 a detach regression shows (the whole ~215ms wall held). The
  match count is REPORTED, not asserted (it is the corpus's business).
  Guidance, the word_bounds finding's shape: the list-returning API
  holds ~0.13µs per match under the GIL: ~13ms at 100k matches (document
  scale), ~170ms at 1.28M (whole-corpus keyword sweeps); callers producing
  millions of matches are the streaming-API question the word_bounds
  finding already raised, recorded again in the README's section.

cells (``replace_many`` dense, ``sentence_bounds`` list,
``diff_opcodes_lines`` near-identical, all at 12 MiB; measured on the dev
box, ambient load 2.0, 5 samples per cell, corpora from
``tests/reference.py``):

- ``replace_many`` dense (the 17 SEARCH_DENSE_PATTERNS words each mapped
  to a redaction token over the plain prose corpus, the redaction-map
  shape the function exists for): worst gaps 10.3-10.9ms of 47-60ms walls
  (ratio 0.18-0.23): the 10ms ping floor plus the O(entries) argument
  walk and the O(output) marshalling of ONE ~10 MiB string, exactly the
  crate GIL model's prediction: the scan+splice is one detached pass and
  the return is a single string, so no list-shape class exists at all.
  The wall clears the ping floor ~5x, so the ratio is NOT the sub-ping
  artifact, and the cell takes the SHARED budgets (0.30 sits ~1.3x above the
  worst measured ratio and ~70% below the ~1.0 a detach regression shows
  against these ~50ms walls; the 100ms ceiling holds ~9x).
- ``sentence_bounds`` list (the word_bounds pair's list shape, sentences
  being far sparser: 170,037 segments measured at 12 MiB, ~1/22nd of
  word_bounds' 3.67M, even sparser than the ~1/10th design estimate in
  src/lib.rs): worst gaps 17.6-23.8ms of 185-190ms walls (ratio
  0.09-0.13): the ping floor plus the O(sentences) 2-tuple marshalling.
  Every sample inside both SHARED budgets (~2.3x ratio margin, ~4.2x
  ceiling margin); a detach regression holds the whole ~190ms wall at
  ratio ~1.0 and fails by far. The list shape MEETS the shared budgets
  here (unlike word_bounds' 428-497ms band) precisely because the tuple
  count is ~20x smaller; the per-shape guidance, not a reason to
  skip the streaming spelling for whole-corpus sweeps.
- ``diff_opcodes_lines`` on the near-identical pair (5 line-opcodes
  measured, the line-level Myers over ~37.8k mostly-distinct lines
  anchored to near-nothing): worst gaps 10.5-12.0ms against walls of only
  1.9-2.6ms: the wall sits an order of magnitude UNDER the 10ms ping
  floor, so the gap/wall ratio (4.2-6.4) is the suite's documented
  sub-ping artifact and the cell asserts the 100ms ceiling only (the b64
  12 MiB / utf8_is_valid / find_patterns-sparse precedent, ~8x margin).
  The line-level spelling is ~30x cheaper in wall than its char-level
  twin on the same pair (76-88ms, the cell) because the diff runs
  over lines, not characters; a limitation in the sparse-search
  cell's shape: a ~2.5ms diff held or released is invisible under the floor
  either way, so this cell pins that the line-split + Myers pass leaves the
  loop at the floor at all.

cells (``count_matches`` dense and ``find_patterns_iter`` dense drain,
both at 12 MiB, the 17-word ``SEARCH_DENSE_PATTERNS`` set over the plain
prose corpus, 1,284,724 matches; measured on the dev box, ambient load 4.4,
5 samples per cell):

- ``count_matches`` dense (the same scan the list shape pays 171.0-174.7ms
  of GIL hold to marshal; here with no match vector, no byte→char
  conversion pass, and a single int return): worst gaps 10.4-11.2ms of
  31.0-31.8ms walls: the ping floor plus ~nothing, the grapheme_count
  precedent's no-marshalling band. The budget consequence: the wall
  clears the ping floor only ~3x, so gap/wall sits at 0.336-0.353 BY
  ARITHMETIC and the shared 0.30 ratio budget is unattainable-by-
  construction for this wall (the QC-Yes 12 MiB situation verbatim: a fast
  detached core makes the floor a large fraction of the wall); the cell
  takes a bespoke 0.60 ratio budget: ~1.7x above the worst measured ratio,
  ~40% below the ~1.0 a detach regression shows (a held scan pins the whole
  ~31ms wall), plus the shared 100ms ceiling (~9x). The inverse of the
  find_patterns-sparse limitation: a held ~31ms scan
  passes the 100ms ceiling easily, so the RATIO budget is this cell's only
  detach discriminator.
- ``find_patterns_iter`` dense drain (construction plus every ``__next__``,
  the streaming answer to the list shape's 171-175ms band): worst gaps
  15.4ms of 129.0-132.0ms walls (ratio 0.116-0.119), inside BOTH shared
  budgets (~2.5x ratio margin, ~6.5x ceiling margin), the word_bounds_iter
  band exactly (15.4ms there: the ping floor plus the two-thread GIL
  contention of the draining thread's per-``__next__`` bytecode). The wall
  contrast, measured in the same process (min-of-3, ambient load 3.1): full
  drain 117.4ms against the list API's 211.1ms and the count core's
  28.3ms: the drain is the ~28-31ms construction scan plus ~86-90ms of
  1.28M per-``__next__`` 3-tuple handoffs (~70ns per match, against the
  list shape's ~0.13µs per match of GIL-HELD marshalling), so the iterator
  wins BOTH axes at this size (the word_bounds_iter finding's shape:
  347ms vs 724ms there). A discrimination caveat: the construction pass is
  only ~31ms of the ~130ms wall, so a construction-only detach regression
  shows ~41ms gaps at ratio ~0.32; the 0.30 budget catches it, but
  thinly; the grosser classes (a per-``__next__`` re-scan, per-next chunk
  marshalling) blow through both budgets.
"""

from __future__ import annotations

import asyncio
import base64
import difflib
import itertools
import string
import time
import urllib.parse
from collections.abc import Awaitable, Callable

import pytest

import tors
from reference import (
    SEARCH_DENSE_PATTERNS,
    SEARCH_SPARSE_PATTERNS,
    compat,
    corpus_b64,
    corpus_utf8,
    decomposed,
    diff_pair_near_identical,
    diff_pair_shuffled,
    entities,
    prose,
    reference_finalize,
    reference_normalize,
)

# The timing lane: every test in this module is a MEASUREMENT cell (worst
# heartbeat-gap bands over multi-MiB corpora), slow and load-sensitive, so
# CI's matrix legs deselect it (`-m "not timing"`) and ONE dedicated step on
# the 3.12 leg runs it (`-m timing`): the GIL contracts still gate every
# push, once, instead of being paid on all five legs. Locally a bare
# ``pytest`` (and ``make test``) run everything, marker or not.
pytestmark = pytest.mark.timing

_PING_S = 0.01
_RATIO_BUDGET = 0.30
_ABS_CEILING_S = 0.100
_SAMPLES = 3

_MIB = 1024 * 1024

# b64_encode_bytes's own ratio budget; see the module docstring's b64
# discussion for why it is not the shared 0.30: the output is 4/3x the input, so the O(output)
# marshalling residue is a structurally larger fraction of b64's wall than of
# finalize's (whose transform dwarfs its residue). 0.80 sits ~1.3x above the worst
# measured ratio (0.61, 48 MiB under ambient load ~5.5) and ~20% below the ~1.0 a
# GIL-held pass shows in every sample.
_B64_RATIO_BUDGET = 0.80

# word_bounds's marshalling-band regression ceiling: NOT the suite's 100ms
# ceiling, which the list-returning API shape cannot meet at whole-file sizes
# (measured: 428-497ms worst gaps at 12 MiB, 3.67M segments; see the cell's
# docstring and the README's streaming-API answer). 1.0s is ~2x above
# the measured band, so it catches marshalling blowouts without pretending
# the 100ms budget is attainable here. The streaming sibling
# (word_bounds_iter, cell below) DOES meet the shared budgets; this cell
# keeps the list shape's band pinned.
_WORD_BOUNDS_CEILING_S = 1.0

# word_bounds's detach-regression discriminator: the measured marshalling
# band sits at ratio 0.72-0.74 (gap/wall), while
# a detach regression (the segmentation itself GIL-held) shows ratio ~1.0
# with a gap of only ~0.4-0.6s at 12 MiB, which the 1.0s ceiling alone lets
# through (a simulated regression measured 0.37-0.40s gaps at ratio 100%
# PASSED the old ceiling-only assertion). 0.85 sits ~1.15x above the band's
# worst measured ratio and well below the ~1.0 a detach regression shows.
_WORD_BOUNDS_RATIO_BUDGET = 0.85

# The 12 MiB QC-Yes prose cells' ratio budget (finalize / finalize_utf8), a
# recalibration: the quick-check skip shrank those walls ~3x (42-46ms,
# measured) while the O(output) marshalling residue is unchanged, so the
# shared 0.30 became a coin flip (measured ratios 0.27-0.34). 0.60 sits ~1.8x
# above the worst measured ratio and ~40% below the ~1.0 a detach regression
# shows (the whole transform held); the 100ms ceiling independently holds
# ~2.3x margin. Same derivation shape as _B64_RATIO_BUDGET below.
_QC_YES_12MIB_RATIO_BUDGET = 0.60

# Both corpus kinds the GIL model names: prose (ASCII) pins the O(output) marshalling
# band; decomposed (non-ASCII) pins the one-time O(input) first-call UTF-8
# materialization (samples 2+ on the same object borrow the cached copy zero-copy).
_CORPORA: dict[str, Callable[[int], str]] = {"prose": prose, "decomposed": decomposed}


async def _gap_and_wall_during(op: Callable[[], Awaitable[object]]) -> tuple[float, float]:
    """Run ``op`` concurrently with a heartbeat; return ``(worst_tick_gap, op_wall)``.

    The leading ``sleep(0)`` guarantees the heartbeat's first tick lands BEFORE the
    operation starts, so a fully blocking operation shows its whole duration as the worst
    gap. The wall ends the moment ``op`` returns, inside the ``try``, before the
    ``finally`` joins the heartbeat, so it is the operation's wall, not the operation
    plus up to one 10ms heartbeat-drain ping (capturing it after the join deflated every
    measured ratio by up to ~9% at 12 MiB)."""
    ticks: list[float] = []
    stop = asyncio.Event()

    async def _heartbeat() -> None:
        while True:
            ticks.append(time.monotonic())
            if stop.is_set():
                return
            await asyncio.sleep(_PING_S)

    heartbeat = asyncio.create_task(_heartbeat())
    await asyncio.sleep(0)
    started = time.monotonic()
    try:
        await op()
        end = time.monotonic()
    finally:
        stop.set()
        await heartbeat
    wall = end - started
    worst_gap = max((b - a for a, b in itertools.pairwise(ticks)), default=0.0)
    return worst_gap, wall


def _budget_misses(gap: float, wall: float, ratio_budget: float | None) -> list[str]:
    """The budgets a sample missed: the single source for what "clean" means. A sample
    is clean exactly when this list is empty (worst gap under BOTH the absolute
    ceiling and the ratio budget), and the failure detail reports the same list, so
    the check and the message can never disagree on the boundary. ``ratio_budget``
    is ``None`` only for cells whose wall sits under the 10ms ping floor, where ANY
    ratio is an artifact (see the module docstring); there the ceiling alone is the
    assertion."""
    missed: list[str] = []
    if gap >= _ABS_CEILING_S:
        missed.append(f"the {_ABS_CEILING_S * 1000:.0f}ms ceiling")
    if ratio_budget is not None and gap >= ratio_budget * wall:
        missed.append(f"the {ratio_budget:.0%} ratio budget")
    return missed


async def _assert_loop_stays_responsive(
    op: Callable[[], Awaitable[object]], ratio_budget: float | None = _RATIO_BUDGET
) -> None:
    """Assert ``op`` leaves the event loop schedulable, over up to ``_SAMPLES``
    measurements, passing on the first clean one (a sample is clean exactly
    when ``_budget_misses`` returns an empty list: the worst tick gap under BOTH
    budgets). A genuine GIL-held whole-text pass reproduces in every sample;
    whole-process CPU starvation does not, so a single starved sample is retried
    instead of failing the test outright. The absolute ceiling catches pathological
    regressions independently of the ratio; both budgets derived from the measured
    bands in this module's docstring; ``ratio_budget`` is per-cell (see
    ``_B64_RATIO_BUDGET``)."""
    observed: list[tuple[float, float]] = []
    for _ in range(_SAMPLES):
        worst_gap, wall = await _gap_and_wall_during(op)
        observed.append((worst_gap, wall))
        if not _budget_misses(worst_gap, wall, ratio_budget):
            return
    detail = "; ".join(
        f"blocked {gap * 1000:.0f}ms of a {wall_ * 1000:.0f}ms operation "
        f"({(gap / wall_ if wall_ else 0):.0%}, "
        f"over {' and '.join(_budget_misses(gap, wall_, ratio_budget))})"
        for gap, wall_ in observed
    )
    raise AssertionError(
        f"the event loop was blocked in every one of {_SAMPLES} samples ({detail}): "
        "tors's py.detach release is not freeing the loop while it runs, or "
        "the return marshalling regressed out of its band (src/lib.rs, "
        "tests/test_gil_release.py)"
    )


@pytest.mark.parametrize(
    ("corpus_kind", "size_bytes", "ratio_budget"),
    [
        ("prose", 12 * _MIB, _QC_YES_12MIB_RATIO_BUDGET),
        ("prose", 32 * _MIB, _RATIO_BUDGET),
        ("decomposed", 12 * _MIB, _RATIO_BUDGET),
        ("decomposed", 32 * _MIB, _RATIO_BUDGET),
    ],
    ids=["prose-12MiB", "prose-32MiB", "decomposed-12MiB", "decomposed-32MiB"],
)
def test_finalize_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    corpus_kind: str, size_bytes: int, ratio_budget: float
) -> None:
    """The claim under test: with the GIL released for the whole native pass, a caller
    that off-loads ``tors.finalize`` to a thread (the wrap extraction-pipeline callers
    keep for the GIL-held boundary residue) gets a loop that ticks every ~10ms through
    the whole transform+hash, where the same work as pure Python starves the loop for
    its largest whole-text C call (91.6-271.5ms at these sizes, measured; see the
    module docstring). Parametrized over both corpus kinds the GIL model names:
    ``prose`` (ASCII, pins the O(output) marshalling band) and ``decomposed``
    (non-ASCII, pins the one-time O(input) first-call UTF-8 materialization; even
    sample 1 stays inside both budgets, and samples 2+ hit the cached borrow).

    note on the prose-12MiB cell's budget: the quick-check skip removed the
    NFC pass from this QC-Yes corpus, shrinking the wall ~3x (42-46ms measured)
    while the marshalling residue is unchanged; the per-cell 0.60 budget is
    derived from the new band (see ``_QC_YES_12MIB_RATIO_BUDGET``); a detach
    regression still fails it at ratio ~1.0."""
    corpus = _CORPORA[corpus_kind](size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.finalize, corpus),
            ratio_budget=ratio_budget,
        )
    )


@pytest.mark.parametrize(
    ("corpus_kind", "size_bytes", "ratio_budget"),
    [
        ("prose", 12 * _MIB, _QC_YES_12MIB_RATIO_BUDGET),
        ("prose", 32 * _MIB, _RATIO_BUDGET),
        ("decomposed", 12 * _MIB, _RATIO_BUDGET),
        ("decomposed", 32 * _MIB, _RATIO_BUDGET),
    ],
    ids=["prose-12MiB", "prose-32MiB", "decomposed-12MiB", "decomposed-32MiB"],
)
def test_finalize_utf8_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    corpus_kind: str, size_bytes: int, ratio_budget: float
) -> None:
    """The bytes-in claim: ``tors.finalize_utf8``'s ``PyBytes`` argument
    extraction is a zero-copy borrow of the immutable buffer, so there is NO
    argument-materialization class at all, so even the non-ASCII (decomposed)
    corpus's FIRST call sits in the plain marshalling band, where the str-in
    ``finalize`` twin pays its one-time O(input) materialization (13.9-15.6ms
    first-call gap at 12 MiB, 37.6-44.1ms at 32 MiB, measured; see the
    module docstring). Measured bands (ambient load 4.3-5.6, 5 samples per cell):
    prose 12 MiB: 10.3-13.0ms of 113-143ms walls (ratio 0.08-0.10);
    decomposed 12 MiB: 10.3-13.0ms of 116-131ms (0.09-0.10), sample 1 in the
    same band as samples 2+;
    prose 32 MiB: 13.5-20.7ms of 345-370ms (0.04-0.06);
    decomposed 32 MiB: 17.6-26.7ms of 350-405ms (0.04-0.07), again sample 1
    indistinguishable from the cached band, against the str-in twin's
    37.6-44.1ms first call.

    note on the prose-12MiB cell's budget: same recalibration as the
    str-in ``finalize`` cell (see ``_QC_YES_12MIB_RATIO_BUDGET``): the
    quick-check skip shrank the QC-Yes prose wall ~3x (34-45ms measured)
    while the bytes-in marshalling residue is unchanged; measured ratios
    0.27-0.34 against the new 0.60 budget. A detach regression still fails
    it at ratio ~1.0."""
    corpus = corpus_utf8(corpus_kind, size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.finalize_utf8, corpus),
            ratio_budget=ratio_budget,
        )
    )


@pytest.mark.parametrize(
    "size_bytes, ratio_budget",
    [(12 * _MIB, None), (32 * _MIB, _B64_RATIO_BUDGET)],
    ids=["12MiB-ceiling-only", "32MiB-both-budgets"],
)
def test_b64_encode_bytes_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int, ratio_budget: float | None
) -> None:
    """The b64 claim, in two cells with different jobs. The GIL-held residue
    is the marshalling of the 4/3x-sized ASCII output, structurally a LARGER
    fraction of the wall than finalize's residue, because the encode itself is so
    fast. Measured (ambient load 4.3-5.6, 5 samples per cell, prose bytes):

    - 12 MiB: worst gaps 10.1-11.2ms (the ping floor plus ~1ms of marshalling a
      16 MiB output) against walls of only 5-9ms, and the wall sits UNDER the 10ms
      ping floor, so the gap/wall ratio (1.0-2.3) is the artifact this suite
      already documents for sub-ping walls, not evidence of blocking; the cell
      asserts the 100ms ceiling (>=9x margin) and records the band.
    - 32 MiB: worst gaps 27.5-29.3ms of 48-50ms walls (ratio 0.57-0.58): the
      43 MiB output's marshalling band at ~2.4GB/s under this load. This is the
      regression-detecting cell: the 0.80 b64 ratio budget holds with ~1.4x
      margin, while a detach regression (GIL-held encode) shows ratio ~1.0 in
      every sample and fails it; the ceiling holds with ~3.4x margin.
    - Larger sizes, measured and unasserted: 48 MiB 37.0-48.8ms of
      68-80ms (0.54-0.61); 96 MiB 67.2-78.3ms of 139-161ms (0.46-0.50); the
      ceiling margin shrinks to ~1.3x under load, below this suite's tolerance.
    - The red side, same placement: ``base64.b64encode(...).decode("ascii")``
      holds the GIL for the encode: measured 122.7-127.6ms of 188-199ms walls
      at 96 MiB, reproducing the ~150ms GIL-held b64encode observation at
      a 100MB document cap that motivated the function. The red expression's
      ratio at 12 MiB is window-dependent (1.00 in the quiet window the band
      was first recorded in; 0.63 under load) and at 32 MiB measures
      0.64-0.66, and both PASS this cell's budgets, a structural property of the
      two-call expression (the eval loop can tick at the bytecode boundary
      between ``b64encode`` and ``decode("ascii")``, so the worst gap is the
      encode alone); the red-side cell below asserts the discriminating sizes
      mechanically."""
    corpus = corpus_utf8("prose", size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.b64_encode_bytes, corpus),
            ratio_budget=ratio_budget,
        )
    )


def _stdlib_b64_expression(raw: bytes) -> str:
    """The stdlib expression ``tors.b64_encode_bytes`` replaces (the red side)."""
    return base64.b64encode(raw).decode("ascii")


@pytest.mark.parametrize(
    ("cell", "size_bytes", "ratio_budget"),
    [
        ("reference-finalize", 12 * _MIB, _RATIO_BUDGET),
        ("reference-finalize", 32 * _MIB, _RATIO_BUDGET),
        ("stdlib-b64-encode", 96 * _MIB, _B64_RATIO_BUDGET),
    ],
    ids=["ref-finalize-12MiB", "ref-finalize-32MiB", "stdlib-b64-96MiB"],
)
def test_the_gil_held_red_sides_fail_their_budgets_in_every_sample(
    cell: str, size_bytes: int, ratio_budget: float
) -> None:
    """The RED side, asserted mechanically: the same
    budgets tors's cells above pass must be FAILED by the GIL-held
    expressions those cells replace; otherwise the budgets would have no
    discriminating power and a tors regression into GIL-held behavior would
    sail through the same numbers. Every sample of every parametrized red
    cell must miss at least one budget, judged by the SAME
    ``_budget_misses`` list the green cells use. Measured on the dev box
    (ambient load ~3.5-5, 3 samples per cell):

    - ``reference_finalize`` (the pure-Python pipeline) in the same
      to_thread placement: at 12 MiB, 95.1-98.0ms gaps of 161-165ms walls
      (ratio 0.58-0.60), missing the 0.30 ratio budget by ~2x in every
      sample; at 32 MiB, 238-259ms of 447-481ms (0.53-0.55), missing BOTH the
      ratio budget and the 100ms ceiling (~2.4x over).
    - The stdlib b64 expression at 96 MiB: 122.3-127.6ms of 188-194ms
      (ratio 0.65-0.66): the single C ``b64encode`` call alone exceeds the
      100ms ceiling in every sample (the motivating ~150ms@100MB
      observation's size class, reproduced).

    Measured and NOT asserted: the 32 MiB b64 red side, 41.5-42.5ms of 63-65ms
    walls (ratio 0.64-0.66), and 12 MiB 15.3-16.5ms of 24-26ms (0.63);
    both PASS the b64 cells' budgets. The structural reason, and why the
    tors cells still discriminate: the stdlib expression is TWO C calls
    with an eval-loop bytecode boundary between them, so the loop ticks
    between encode and ``decode("ascii")`` and the worst gap is the encode
    alone, under both budgets at those sizes. A tors detach regression is
    ONE C call (encode and marshalling held together, no bytecode
    boundary), which pins the whole wall: the 32 MiB tors cell's measured
    band is 0.57-0.58 against its 0.80 budget, and a one-call hold of such
    a wall shows ratio ~1.0, the shape the single-C-call reds above
    demonstrate directly. The b64 budget's discriminating power for tors's
    own shape is real, but it rests on the one-call structure, not on the
    stdlib red side's two-call shape at small sizes; recorded here so the
    next reader does not mistake the 12/32 MiB b64 red sides for
    regression-proof."""
    corpus = corpus_utf8("prose", size_bytes) if cell == "stdlib-b64-encode" else prose(size_bytes)
    red = _stdlib_b64_expression if cell == "stdlib-b64-encode" else reference_finalize
    observed = [
        asyncio.run(_gap_and_wall_during(lambda: asyncio.to_thread(red, corpus)))
        for _ in range(_SAMPLES)
    ]
    for gap, wall in observed:
        assert _budget_misses(gap, wall, ratio_budget), (
            f"the {cell} red side at {size_bytes // _MIB} MiB measured a CLEAN "
            f"sample ({gap * 1000:.0f}ms of a {wall * 1000:.0f}ms wall): the "
            "budgets tors's cells pass no longer discriminate against the "
            "GIL-held expression they replace, and a detach regression could "
            "pass them too (tests/test_gil_release.py)"
        )


@pytest.mark.parametrize("form_name", ["nfc", "nfkd"])
@pytest.mark.parametrize("size_bytes", [12 * _MIB, 32 * _MIB], ids=["12MiB", "32MiB"])
def test_standalone_forms_full_pass_on_the_compat_corpus_keep_the_loop_at_heartbeat_granularity(
    form_name: str, size_bytes: int
) -> None:
    """The standalone-forms GIL claim, on the corpus that still
    pays it: the whole transform runs under ``py.detach`` (same pattern as
    ``tors.normalize``), so the residue is the str-in call's two known bands:
    the O(output) return marshalling, plus the ONE-TIME O(input) UTF-8
    materialization on the first non-ASCII call. Measured on the dev box
    (ambient load 4.4-5.7, 3 samples per cell, compat corpus: decomposed
    accents PLUS a compatibility ligature and fullwidth digit per unit, the
    input that keeps NFKC/NFKD's quick check at No):

    - ``nfc`` 12 MiB: 16.8-17.4ms of 127-145ms walls (0.10-0.13): the
      sample-1 gap carries the materialization, the same class every str-in
      function pays.
    - ``nfkd`` 12 MiB: 14.7-17.8ms of 87-104ms walls (0.14-0.20).
    - ``nfc`` 32 MiB: 36.3-40.1ms of 346-391ms walls (0.10-0.12).
    - ``nfkd`` 32 MiB: 35.7-38.2ms of 240-263ms walls (0.15-0.16).

    Every measured sample sits under BOTH shared budgets (worst ratio 0.20 vs
    the 0.30 budget; worst gap 40.1ms vs the 100ms ceiling), and the red side
    is the whole reason the functions exist: ``unicodedata.normalize`` is one
    GIL-held C call for the entire text (the reference finalize's 91.6-271.5ms
    gaps in the module docstring are set by exactly that call).

    Why this corpus replaced the decomposed one (a finding, pinned by
    the fast-path cell below): plain decomposed prose carries no compatibility
    mappings, so under NFKC/NFKD it quick-checks Yes and the identity-return
    fast path hands it back untouched, so the D-forms' transform band cannot be
    measured on it anymore."""
    corpus = compat(size_bytes)
    form = {"nfc": tors.nfc, "nfkd": tors.nfkd}[form_name]
    asyncio.run(_assert_loop_stays_responsive(lambda: asyncio.to_thread(form, corpus)))


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_d_form_fast_path_on_the_decomposed_corpus_is_ceiling_only(
    size_bytes: int,
) -> None:
    """The quick-check fast path's own band: under NFKD the decomposed
    corpus (decomposed accents, ASCII otherwise, NO compatibility mappings)
    quick-checks Yes, so ``tors.nfkd`` returns the INPUT OBJECT; the
    transform is the quick-check scan (2.4ms at 12 MiB, measured) plus the
    str-in first-call O(input) UTF-8 materialization, and nothing else.
    Measured on the dev box (ambient load 4.4-5.7, 3 samples): worst gaps
    10.4-13.7ms of 10-23ms walls: the wall sits AT the 10ms ping floor, so
    the gap/wall ratio (0.59-1.13) is the documented sub-ping artifact (the
    b64 12 MiB precedent), and the cell asserts the 100ms ceiling (~9x
    margin) only. ``nfc`` on the same corpus still pays its full pass (the
    combining marks keep NFC's quick check at Maybe); that band is covered
    by the compat cell above, whose nfc leg runs the same pass shape."""
    corpus = decomposed(size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.nfkd, corpus),
            ratio_budget=None,
        )
    )


@pytest.mark.parametrize("fn_name", ["normalize", "finalize"])
@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_identity_paths_on_already_normalized_prose_are_ceiling_only(
    fn_name: str, size_bytes: int
) -> None:
    """The identity-return contract's GIL band, the zero-cost path in
    the crate GIL model: on ALREADY-NORMALIZED text (this corpus is the
    pipeline's own output: ``reference_normalize(prose(...))``), the complete
    transform is provably a no-op (quick-check Yes + every scan stage's
    fingerprint absent), so the ORIGINAL object comes back (no allocation,
    no copy, no marshalling), and the only GIL-held residue is the argument
    borrow itself (zero-copy for the ASCII corpus). ``finalize`` additionally
    computes SHA-256 over the borrowed input, detached.

    Measured on the dev box (ambient load 4.4-4.6, 3 samples per cell):
    ``normalize`` worst gaps 10.4-10.9ms of 3-4ms walls, ``finalize``
    10.9ms of ~9ms walls, and the wall is AT or UNDER the 10ms ping floor (the
    whole call is now cheaper than one heartbeat), so the ratio is the
    documented sub-ping artifact and the cell asserts the 100ms ceiling
    (~9-10x margin) only. The wall collapse this pins, for the record:
    normalize 123.0ms -> 2.89ms, finalize 126.3ms -> 7.72ms (min-of-5,
    build vs build; see the README's table for loads)."""
    corpus = reference_normalize(prose(size_bytes))
    fn = {"normalize": tors.normalize, "finalize": tors.finalize}[fn_name]
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(fn, corpus),
            ratio_budget=None,
        )
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB, 32 * _MIB], ids=["12MiB", "32MiB"])
def test_html_unescape_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The html claim: the whole entity scan runs under ``py.detach``, where
    ``html.unescape`` is a GIL-held ``re.sub`` over the whole string
    with a PYTHON callback invoked per entity. Measured on the dev box
    (ambient load 10.0, the heaviest window in this module's tables; 3
    samples per cell, entities corpus):

    - 12 MiB: worst gaps 16.4-17.6ms of 79-88ms walls (ratio 0.19-0.22):
      the ping floor plus the marshalling of the ~12 MiB decoded output;
      every sample inside both shared budgets (~1.4x ratio margin under this
      load, more when quiet).
    - 32 MiB: worst gaps 28.6-40.9ms of 223-225ms walls (ratio 0.13-0.18).

    The red side, same placement, is NOT ratio ~1.0 like the other stdlib
    red sides, and the reason is the function's own structure:
    ``html.unescape``'s per-entity ``_replace_charref`` callback is PYTHON
    bytecode, and the eval loop checks the GIL drop request between
    bytecodes, so ~800k callbacks per 12 MiB give the interpreter frequent
    (if tiny) yield points; measured worst gaps 35.7-49.4ms of 236-250ms
    walls (ratio 0.15-0.20). tors still roughly HALVES the worst gap
    (16.4-17.6 vs 35.7-49.4ms) and is ~3x faster in wall time (the
    callbacks are the cost); the accurate statement is "half the block time,
    a third of the wall", not "unblocked vs fully blocked"."""
    corpus = entities(size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(lambda: asyncio.to_thread(tors.html_unescape, corpus))
    )


@pytest.mark.parametrize(
    "size_bytes, ratio_budget",
    [(12 * _MIB, None), (32 * _MIB, _B64_RATIO_BUDGET)],
    ids=["12MiB-ceiling-only", "32MiB-both-budgets"],
)
def test_b64_decode_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int, ratio_budget: float | None
) -> None:
    """The decode claim (input: the b64-encoded corpora from
    ``reference.corpus_b64``, a 4/3x-sized ASCII str; output: the decoded
    bytes). The GIL-held residue is the marshalling of the decoded bytes
    (O(output), 3/4 of the input text), the same structural shape as
    ``b64_encode_bytes``'s output marshalling, so it uses the same b64 budget
    derivation (the shared 0.30 is unattainable for the same reason: the scan
    is fast, so the marshalling is a large wall share). Measured on the dev box
    (ambient load 6.0, 3 samples per cell, prose):

    - 12 MiB decoded (16 MiB b64 text): worst gaps 11.0-11.7ms against
      14.0-23.2ms walls (ratio 0.50-0.78): the wall sits near the 10ms ping
      floor, so the ratio is the documented sub-ping artifact; the cell asserts
      the 100ms ceiling (~9x margin) and records the band.
    - 32 MiB decoded (43 MiB b64 text): worst gaps 16.6-22.7ms of 59.1-65.3ms
      walls (ratio 0.28-0.35), comfortably inside both the 0.80 b64 budget
      (~2.3x margin) and the 100ms ceiling (~4.4x margin). A detach regression
      shows ratio ~1.0 (the red side below) and fails this cell.
    - The red side, same placement: ``base64.b64decode`` holds the GIL for the
      whole C decode: 14.8-27.5ms gaps of 15.0-27.8ms walls (ratio 0.98-0.99)
      in EVERY 12 MiB sample, measured alongside the tors cells above."""
    corpus = corpus_b64("prose", size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.b64_decode, corpus),
            ratio_budget=ratio_budget,
        )
    )


@pytest.mark.parametrize("corpus_kind", ["prose", "decomposed"])
@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_grapheme_count_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    corpus_kind: str, size_bytes: int
) -> None:
    """The grapheme claim: the whole cluster scan runs under
    ``py.detach`` and the return is a single int, so no list marshalling at all.
    Measured on the dev box (ambient load 8.3, 3 samples per cell):

    - prose 12 MiB: worst gap 10.4ms of a 129ms wall (0.08): the ping floor
      plus the int return; nothing else to hold.
    - decomposed 12 MiB: first call 12.8ms of 134ms (0.10, the str-in
      one-time O(input) UTF-8 materialization on top of the floor, the same
      class every str-in function pays), cached 10.5ms of 122ms (0.09).

    Every sample sits deep inside both shared budgets (~3x on the ratio,
    ~8x on the ceiling)."""
    corpus = _CORPORA[corpus_kind](size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.grapheme_count, corpus)
        )
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_word_bounds_marshalling_band_is_pinned_against_regression(
    size_bytes: int,
) -> None:
    """The word_bounds MARSHALLING FINDING, as a regression ceiling:
    NOT the suite's 100ms ceiling, which this API shape cannot
    meet at whole-file sizes, and saying so is the point of this cell.

    The measurement (dev box, ambient load 8.3, 3 samples, prose 12 MiB =
    12.58M chars = 3,665,242 word segments): worst gaps 428-497ms of
    591-671ms walls (ratio 0.72-0.74): the GIL-held construction of 3.67M
    2-tuples and 7.33M ints DOMINATES the call. The segmentation itself runs
    detached; it is the return marshalling that holds the loop, O(number of
    segments), so the 100ms ceiling is out of reach by ~4.5x for the
    list-returning shape at this size; a real, reported cost, not one to
    threshold away or hide.

    The API question this raises (recorded in the README and the
    report): a streaming shape for large inputs, either a lazy iterator
    (pyo3 `PyIterator` yielding `(start, end)` tuples in chunks, so the GIL
    is held only per-chunk), or an explicit `word_bounds_into(text, chunk)`
    callback/`count_only` fast path. Until such an API exists, the guidance is
    the measured band: fine at document scale (a 100KB chunk is ~30k
    segments, ~4ms held), a half-second GIL hold at 12 MiB.

    The cell pins the band against regression with BOTH budgets, pass-on-
    first-clean like every other cell in this file (a genuine regression
    dirties every sample; a starved box dirties only the sample it hits):
    a sample is clean when its worst gap is under BOTH the 1.0s
    marshalling-band ceiling (~2x above the measured 0.43-0.50s band; the
    marshalling is allocation-bound and stable under load) AND the 0.85 ratio
    budget of its own wall. The ratio budget is the detach-regression
    discriminator: the measured band sits at
    ratio 0.72-0.74, while a detach regression (the segmentation itself
    GIL-held) holds essentially the whole wall (ratio ~1.0) with a gap of
    only ~0.4-0.6s at this size, which the 1.0s ceiling alone would let
    through; a simulated-regression probe measured exactly that
    (0.37-0.40s gaps at ratio 100%, passing the old ceiling-only assertion
    and failing this one)."""
    corpus = prose(size_bytes)
    observed = [
        asyncio.run(_gap_and_wall_during(lambda: asyncio.to_thread(tors.word_bounds, corpus)))
        for _ in range(_SAMPLES)
    ]
    for gap, wall in observed:
        if gap < _WORD_BOUNDS_CEILING_S and gap < _WORD_BOUNDS_RATIO_BUDGET * wall:
            return
    detail = "; ".join(
        f"blocked {gap * 1000:.0f}ms of a {wall * 1000:.0f}ms operation "
        f"({gap / wall:.0%}, over the 1.0s ceiling and/or the "
        f"{_WORD_BOUNDS_RATIO_BUDGET:.0%} ratio budget)" if wall else "n/a"
        for gap, wall in observed
    )
    raise AssertionError(
        f"the word_bounds band regressed in every one of {_SAMPLES} samples ({detail}): "
        "either the bounds-list marshalling blew past its measured band or the "
        "segmentation lost its detach (ratio ~1.0) (src/segmentation_impl.rs, "
        "tests/test_gil_release.py)"
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_word_bounds_iter_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The streaming answer to the word_bounds marshalling finding: the
    FULL consumption of ``tors.word_bounds_iter(text)`` over the same 12 MiB
    prose corpus (3.67M segments; construction plus every ``__next__``)
    against the suite's SHARED budgets (0.30 ratio, 100ms ceiling), which the
    list-returning shape structurally could not meet (428-497ms held, the
    cell above).

    The design (src/lib.rs, ``WordBoundsIter``): the whole segmentation
    (the same detached core pass the list API runs) computes the bounds
    Vec up front under ONE ``py.detach`` (GIL-free for its full duration;
    the Vec is 16 bytes per segment, ~59 MiB at this size, versus the list
    API's ~hundreds of MiB of Python tuples), and each ``__next__`` then
    holds the GIL only to construct ONE 2-tuple (µs-scale), so the worst
    heartbeat gap collapses back to the ping-floor band every other
    str-in cell sits in.

    The tradeoff, recorded in the README's performance section,
    measured the opposite way round from the design's expectation: the
    full drain measured 347ms (min-of-3) against the list API's 724ms in
    the same process; the per-``__next__`` tuple path is FASTER per bound
    than the list-return conversion, so the iterator wins on BOTH axes at
    this size (gap band above, wall ~2.1x). The list API stays the right
    shape for small inputs and one-shot batch work; the iterator is the
    shape for whole-file segmentation on a live loop.

    Measured on the dev box (ambient load 4.4-4.6, 3 samples): construction
    plus full drain: worst gaps 15.4ms of 327-358ms walls (0.04-0.05),
    inside both SHARED budgets; the construction's own detached pass is the
    same ~175ms core the list API runs."""
    corpus = prose(size_bytes)

    def consume() -> int:
        return sum(1 for _ in tors.word_bounds_iter(corpus))

    asyncio.run(_assert_loop_stays_responsive(lambda: asyncio.to_thread(consume)))


def _invalid_utf8_corpus(size_bytes: int) -> bytes:
    """The prose corpus as UTF-8 bytes with its FINAL byte replaced by 0xFF,
    an invalid lead byte at the very end, so the validity scan still traverses
    the whole corpus before answering False (an invalid byte earlier in the
    buffer would let the validator stop early and hold proportionally less of
    the loop; end placement makes the cell pin the whole-input scan's detach)."""
    raw = corpus_utf8("prose", size_bytes)
    return raw[:-1] + b"\xff"


@pytest.mark.parametrize(
    ("corpus_kind", "size_bytes", "ratio_budget"),
    [
        ("valid", 12 * _MIB, None),
        ("valid", 32 * _MIB, None),
        ("invalid", 12 * _MIB, None),
        ("invalid", 32 * _MIB, None),
    ],
    ids=[
        "valid-12MiB-ceiling-only",
        "valid-32MiB-ceiling-only",
        "invalid-12MiB-ceiling-only",
        "invalid-32MiB-ceiling-only",
    ],
)
def test_utf8_is_valid_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    corpus_kind: str, size_bytes: int, ratio_budget: float | None
) -> None:
    """The validity claim: the whole SIMD scan runs under ``py.detach``,
    and the call's GIL-held residue is the argument borrow ALONE: the bytes-in
    family's extreme point (a ``bool`` return, so no marshalling class at all)
    with no exception path either (invalid input answers False; nothing ever
    raises), so the VALID and INVALID corpora's cells must sit in the same
    band. Both cells are ceiling-only by the b64 12 MiB precedent: validation
    is SIMD-fast, so the wall at these sizes sits far UNDER the 10ms ping
    floor and any gap/wall ratio is the suite's documented sub-ping artifact;
    the 100ms ceiling alone is the assertion.

    Measured on the dev box (ambient load 1.7, 3 samples per cell): valid
    12 MiB worst gaps 10.9-11.5ms of 0.7-1.3ms walls; valid 32 MiB 11.3-11.5ms
    of 1.1-1.4ms; invalid 12 MiB 10.9-11.0ms of 0.7ms; invalid 32 MiB
    11.5-11.7ms of 1.3-1.4ms. Every gap is the 10ms ping floor plus ~1-1.7ms
    of to_thread dispatch and the argument borrow (there is no marshalling
    class to pay at all), and the walls (the SIMD scan plus the dispatch;
    the inline call itself measures 0.09-0.55ms) sit an order of magnitude
    under the ping floor, so the cells assert the 100ms ceiling (~9x margin)
    only, at both sizes and both corpus kinds."""
    corpus = (
        corpus_utf8("prose", size_bytes)
        if corpus_kind == "valid"
        else _invalid_utf8_corpus(size_bytes)
    )
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.utf8_is_valid, corpus),
            ratio_budget=ratio_budget,
        )
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_diff_opcodes_near_identical_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The diff claim on the few-opcode shape: the WHOLE diff (both
    operands' ``Vec<char>`` materialization and the Myers search) runs under
    ``py.detach``, so a near-identical 12 MiB pair (six scattered line edits,
    235 opcodes) leaves the loop ticking at heartbeat granularity while
    ~80ms of native diff work runs. Measured on the dev box (ambient load
    5.3, 3 samples): worst gaps 10.5-11.2ms of 76-88ms walls (ratio
    0.12-0.14): the ping floor plus the two zero-copy ASCII argument borrows
    and a ~0.05ms 235-tuple marshalling. Inside both shared budgets with ~2x
    ratio margin; a detach regression (the diff itself GIL-held) holds the
    whole wall at ratio ~1.0 and fails by ~3x. If walls ever collapse toward
    the ping floor on a quiet box, recalibrate ceiling-only per the b64 12
    MiB precedent (see the module docstring's b64 discussion)."""
    a, b = diff_pair_near_identical(size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(lambda: asyncio.to_thread(tors.diff_opcodes, a, b))
    )


# The shuffled cell's marshalling-band regression ceiling: NOT the suite's
# 100ms ceiling, which a several-fold marshalling blowout at this op count
# would still clear (measured band 20.4-26.3ms; ~5x the ~10-15ms marshalling
# share lands at ~60-75ms < 100ms). 60ms sits ~2.3x above the measured worst
# gap and ~5x above the marshalling share itself; the word_bounds-precedent
# derivation (a bespoke ceiling ~2x above the measured band), sized to catch
# the blowout class (per-op PyString construction, per-op tuple churn) while
# tolerating ping-floor wobble under load. The shared 0.30 ratio budget is
# the detach-regression discriminator alongside it (a held diff shows ratio
# ~1.0 against these ~1.6s walls).
_DIFF_SHUFFLED_CEILING_S = 0.060


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_diff_opcodes_shuffled_pair_marshalling_band_is_pinned_against_regression(
    size_bytes: int,
) -> None:
    """The diff claim on the many-opcode shape: the O(ops) return
    marshalling made visible (the ``word_bounds`` list-shape class, applied to
    a function whose output is a list by necessity): the character-level diff
    of the 12 MiB shuffled pair (same content, different line order, the
    corpus that makes the opcode count large). The diff itself is detached
    (walls 1554-1626ms measured); the GIL-held residue is the construction of
    103,421 5-tuples with up to four fresh PyLongs each, the four tag strings
    pre-built once per call and shared by reference into every tuple.

    Measured on the dev box (ambient load 5.3-6.6, 3 samples): worst gaps
    20.4-26.3ms of 1554-1626ms walls (ratio 0.013-0.017). The marshalling
    itself is the delta over the near-identical cell's floor band:
    ~10-15ms for 103,421 opcodes, ~0.1-0.15µs per opcode.

    The cell pins the band pass-on-first-clean over both budgets: the 60ms
    bespoke ceiling (the word_bounds-precedent marshalling-band pin, catching
    a several-fold marshalling blowout the shared 100ms ceiling would let
    through at this op count) AND the shared 0.30 ratio (a detach regression
    holds the whole ~1.6s wall). The op count is REPORTED in the failure
    detail, never asserted; it is the algorithm's business: similar's
    bounded search trades minimality for speed on hard inputs, and a crate
    bump may legitimately change the count without changing validity (the
    structural contract lives in tests/test_diff_opcodes.py)."""
    a, b = diff_pair_shuffled(size_bytes)
    op_counts: list[int] = []

    def op() -> list[tuple[str, int, int, int, int]]:
        opcodes = tors.diff_opcodes(a, b)
        op_counts.append(len(opcodes))
        return opcodes

    observed = [
        asyncio.run(_gap_and_wall_during(lambda: asyncio.to_thread(op))) for _ in range(_SAMPLES)
    ]
    for gap, wall in observed:
        if gap < _DIFF_SHUFFLED_CEILING_S and gap < _RATIO_BUDGET * wall:
            return
    detail = "; ".join(
        f"blocked {gap * 1000:.0f}ms of a {wall_ * 1000:.0f}ms operation "
        f"({(gap / wall_ if wall_ else 0):.0%}, "
        f"over the {_DIFF_SHUFFLED_CEILING_S * 1000:.0f}ms marshalling ceiling "
        f"and/or the {_RATIO_BUDGET:.0%} ratio budget)"
        for gap, wall_ in observed
    )
    raise AssertionError(
        f"the diff_opcodes marshalling band regressed in every one of {_SAMPLES} "
        f"samples ({detail}; {op_counts[0]} opcodes in the pair): either the "
        "opcode-list marshalling blew past its measured band (~10-15ms above "
        "the ping floor for ~103k opcodes) or the diff lost its detach "
        "(ratio ~1.0) (src/diff_impl.rs, tests/test_gil_release.py)"
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_find_patterns_sparse_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The sparse-search claim: the WHOLE search (automaton build, scan,
    and the byte→char conversion, idle here since the corpus is ASCII) runs
    under ``py.detach``, over the diff near-identical pair's edited corpus
    with the sparse terminology-scan pattern set: ``"monthly"`` occurs
    exactly once (the inserted line; the diff builder's four replace
    positions land on the corpus's empty separator lines at this size), the
    other two terms never. Measured on the dev box (ambient load 6.9-7.3, 3
    samples): worst gaps 10.3-10.7ms of 6-7ms walls: the ping floor plus
    one 3-tuple.

    Ceiling-only by the b64 12 MiB / utf8_is_valid precedent: the pure scan
    is ~2 GiB/s, so the wall sits UNDER the 10ms ping floor and any
    gap/wall ratio is the suite's documented sub-ping artifact; the 100ms
    ceiling alone is the assertion (~10x margin). A limitation, said
    plainly: a ~6ms scan held or released is invisible under the floor
    either way, so this cell cannot by itself discriminate a detach
    regression; the dense cell below carries that (a held dense pass shows
    ratio ~1.0 against its ~215ms wall); this cell pins that the no-match
    scan leaves the loop at the floor at all."""
    edited = diff_pair_near_identical(size_bytes)[1]
    patterns = list(SEARCH_SPARSE_PATTERNS)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.find_patterns, patterns, edited),
            ratio_budget=None,
        )
    )


# The dense cell's budgets: the word_bounds-precedent marshalling-band pin,
# with the ratio recalibrated to this cell's structurally higher marshalling
# share: the search core is ~3.5x faster than segmentation, so the O(matches)
# marshalling is a larger fraction of the wall (measured 0.76-0.80, against
# word_bounds' 0.72-0.74). 0.90 sits ~1.125x above the worst measured ratio
# and ~10% below the ~1.0 a detach regression shows; the same derivation
# shape as _WORD_BOUNDS_RATIO_BUDGET, sized for this band. The 400ms ceiling
# is ~2.3x above the measured 171.0-174.7ms worst gaps (the diff-shuffled
# cell's margin), sized to catch a several-fold marshalling blowout the
# shared 100ms ceiling is not sized for at 1.28M matches.
_SEARCH_DENSE_CEILING_S = 0.400
_SEARCH_DENSE_RATIO_BUDGET = 0.90


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_find_patterns_dense_marshalling_band_is_pinned_against_regression(
    size_bytes: int,
) -> None:
    """The dense-search claim: the O(matches) return marshalling made
    visible (the ``word_bounds``/``diff_opcodes`` list-shape class, applied
    to a function whose output is a list by necessity): the 17-word dense
    pattern set over the plain 12 MiB prose corpus, 1,284,724 matches. The
    search itself is detached (walls 214-224ms measured); the GIL-held
    residue is the construction of 1.28M 3-tuples of ints.

    Measured on the dev box (ambient load 6.9-7.3, 3 samples): worst gaps
    171.0-174.7ms of 214-224ms walls (ratio 0.76-0.80). The marshalling
    itself is the delta over the sparse cell's floor band: ~160-165ms for
    1.28M 3-tuples, ~0.13µs per match.

    The cell pins the band pass-on-first-clean over both budgets: the 400ms
    bespoke ceiling (the word_bounds-precedent marshalling-band pin) AND the
    0.90 ratio budget (the detach discriminator: a held search shows ratio
    ~1.0 against these ~215ms walls; see the constant's comment for why 0.90
    and not word_bounds' 0.85). The match count is REPORTED in the failure
    detail, never asserted; it is the corpus's business, not a contract.

    Guidance (the word_bounds finding's shape, recorded in the
    README's section): ~0.13µs of GIL hold per match: ~13ms at 100k
    matches (document scale), ~170ms at 1.28M (whole-corpus keyword sweeps);
    callers producing millions of matches are the streaming-API question the
    word_bounds finding already raised."""
    text = prose(size_bytes)
    patterns = list(SEARCH_DENSE_PATTERNS)
    match_counts: list[int] = []

    def op() -> list[tuple[int, int, int]]:
        matches = tors.find_patterns(patterns, text)
        match_counts.append(len(matches))
        return matches

    observed = [
        asyncio.run(_gap_and_wall_during(lambda: asyncio.to_thread(op))) for _ in range(_SAMPLES)
    ]
    for gap, wall in observed:
        if gap < _SEARCH_DENSE_CEILING_S and gap < _SEARCH_DENSE_RATIO_BUDGET * wall:
            return
    detail = "; ".join(
        f"blocked {gap * 1000:.0f}ms of a {wall_ * 1000:.0f}ms operation "
        f"({(gap / wall_ if wall_ else 0):.0%}, "
        f"over the {_SEARCH_DENSE_CEILING_S * 1000:.0f}ms marshalling ceiling "
        f"and/or the {_SEARCH_DENSE_RATIO_BUDGET:.0%} ratio budget)"
        for gap, wall_ in observed
    )
    raise AssertionError(
        f"the find_patterns marshalling band regressed in every one of {_SAMPLES} "
        f"samples ({detail}; {match_counts[0]} matches in the corpus): either the "
        "match-list marshalling blew past its measured band (~160-165ms above "
        "the ping floor for ~1.28M matches) or the search lost its detach "
        "(ratio ~1.0) (src/search_impl.rs, tests/test_gil_release.py)"
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_replace_many_dense_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The replace claim on the redaction-map shape the function exists
    for: the 17 dense prose words each mapped to a redaction token, over the
    plain 12 MiB prose corpus: one automaton build, one scan, one splice,
    all detached, and the return is ONE ~10 MiB string, so the GIL-held
    residue is the O(entries) argument walk plus that single string's
    marshalling: the no-list-shape-class prediction of the crate GIL
    model, measured to be exactly the ping-floor band.

    Measured on the dev box (ambient load 2.0, 5 samples): worst gaps
    10.3-10.9ms of 47-60ms walls (ratio 0.18-0.23). The wall clears the
    10ms ping floor ~5x, so the ratio is NOT the sub-ping artifact, and the
    cell takes the SHARED budgets: the 0.30 ratio sits ~1.3x above the
    worst measured ratio and ~70% below the ~1.0 a detach regression shows
    (a held scan+splice pins the whole ~50ms wall), and the 100ms ceiling
    holds ~9x over the worst gap. The residue band is the structural
    contrast with the find_patterns dense cell above (171.0-174.7ms for
    the same 1.28M matches): reporting the matches as a LIST costs
    ~0.13µs of GIL hold per match, while splicing them into one output
    string costs ~nothing the loop can see."""
    text = prose(size_bytes)
    replacements = {word: "[REDACTED]" for word in SEARCH_DENSE_PATTERNS}
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.replace_many, text, replacements)
        )
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_sentence_bounds_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The sentence claim, list spelling: the whole UAX #29 sentence
    segmentation runs under ``py.detach`` (the word_bounds pair's GIL
    model exactly), and the GIL-held residue is the O(sentences) 2-tuple
    marshalling: 170,037 segments measured at 12 MiB, ~1/22nd of
    word_bounds' 3.67M on the same corpus (even sparser than the ~1/10th
    design estimate in src/lib.rs), which is why the list shape MEETS the
    shared budgets here where word_bounds' 428-497ms band could not.

    Measured on the dev box (ambient load 2.0, 5 samples): worst gaps
    17.6-23.8ms of 185-190ms walls (ratio 0.09-0.13): the ping floor
    plus ~8-14ms of 2-tuple construction for 170k segments (~0.05-0.08µs
    per segment, the word_bounds per-element band). Every sample inside
    both SHARED budgets (~2.3x ratio margin, ~4.2x ceiling margin); a
    detach regression (the segmentation itself GIL-held) shows ratio ~1.0
    against these ~190ms walls and fails by far. The streaming
    ``sentence_bounds_iter`` exists for whole-corpus sweeps that never
    materialize the list; its sequence parity is pinned in
    tests/test_sentence_bounds.py, and the shared-budget band here is the
    per-shape guidance for callers choosing between the two."""
    corpus = prose(size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(lambda: asyncio.to_thread(tors.sentence_bounds, corpus))
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_diff_opcodes_lines_near_identical_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The line-diff claim on the near-identical pair: the WHOLE pass
    (both operands' line split and the Myers search over the ~37.8k lines)
    runs under ``py.detach``, and the return is a 5-tuple list measured at
    just 5 opcodes for the pair's six scattered line edits (the line-level
    diff collapses the char-level twin's 235 opcodes to the edit blocks'
    line granularity), so the marshalling is µs-scale.

    Measured on the dev box (ambient load 2.0, 5 samples): worst gaps
    10.5-12.0ms against walls of only 1.9-2.6ms: the wall sits an order
    of magnitude UNDER the 10ms ping floor (the line-level spelling is
    ~30x cheaper in wall than its char-level twin's 76-88ms on the same
    pair, the cell, because the Myers search runs over lines, not
    characters), so the gap/wall ratio (4.2-6.4) is the suite's
    documented sub-ping artifact and the cell asserts the 100ms ceiling
    only (~8x margin; the b64 12 MiB / utf8_is_valid /
    find_patterns-sparse precedent). A limitation, the sparse cell's
    shape: a ~2.5ms diff held or released is invisible under the floor
    either way, so this cell cannot discriminate a detach regression by
    itself; it pins that the line-split + Myers pass leaves the loop at
    the floor at all."""
    a, b = diff_pair_near_identical(size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.diff_opcodes_lines, a, b),
            ratio_budget=None,
        )
    )


# count_matches's ratio budget: NOT the shared 0.30, which this wall cannot
# attain by construction: the count core is a single detached scan with an
# int return (no marshalling class at all), so the GIL-held residue is the
# 10ms ping floor alone, but the wall (~31ms at 12 MiB dense) clears the
# floor only ~3x, making gap/wall ≈ 0.34 by arithmetic. The QC-Yes 12 MiB
# derivation verbatim (_QC_YES_12MIB_RATIO_BUDGET's comment): 0.60 sits
# ~1.7x above the worst measured ratio (0.353) and ~40% below the ~1.0 a
# detach regression shows (a held scan pins the whole ~31ms wall). The 100ms
# ceiling holds ~9x over the measured gaps, and it is NOT a
# detach discriminator at this size (a held ~31ms wall passes it easily);
# the ratio budget is this cell's only one.
_COUNT_MATCHES_RATIO_BUDGET = 0.60


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_count_matches_dense_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The count claim: the whole count pass (automaton build, scan,
    and nothing else: no match vector, no byte→char conversion pass, a
    single int return) runs under ``py.detach``, so the GIL-held residue
    is the ping floor alone, the grapheme_count precedent's no-marshalling
    band. The corpus is the find_patterns dense cell's own (the 17-word
    dense set over the plain prose corpus, 1,284,724 matches): the count
    answers the same question the list shape marshalled 171.0-174.7ms of
    GIL hold for, at zero measurable hold.

    Measured on the dev box (ambient load 4.4, 5 samples): worst gaps
    10.4-11.2ms of 31.0-31.8ms walls (ratio 0.336-0.353). The budget
    consequence: the wall clears the ping floor only ~3x,
    so the gap/wall ratio sits at 0.34-0.35 BY ARITHMETIC; the shared 0.30
    budget is unattainable for any sub-40ms wall with a floor-plus-nothing
    residue (a budget that flakes is worse than a budget derived from the
    band; see ``_QC_YES_12MIB_RATIO_BUDGET`` for the same recalibration
    shape). The cell takes the bespoke 0.60 ratio budget (a detach
    regression holds the whole ~31ms wall at ratio ~1.0 and fails by far)
    plus the shared 100ms ceiling (~9x margin). The inverse of the sparse
    cell's limitation: a held ~31ms scan passes the ceiling easily,
    so the RATIO budget is this cell's only detach discriminator."""
    text = prose(size_bytes)
    patterns = list(SEARCH_DENSE_PATTERNS)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.count_matches, patterns, text),
            ratio_budget=_COUNT_MATCHES_RATIO_BUDGET,
        )
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_find_patterns_iter_dense_drain_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The streaming answer to the find_patterns dense marshalling band
    (171.0-174.7ms of GIL hold for 1.28M 3-tuples, the cell above): the
    FULL consumption of ``tors.find_patterns_iter(patterns, text)`` over the
    same dense corpus (construction plus every ``__next__``) against the
    suite's SHARED budgets, which the list-returning shape structurally
    could not meet.

    The design (the word_bounds_iter cell's, verbatim): the whole search
    (automaton build, scan, conversion, buffer fill) runs under ONE
    ``py.detach`` when the iterator is constructed, and each ``__next__``
    then holds the GIL only to construct ONE 3-tuple (µs-scale), so the
    worst heartbeat gap collapses back to the ping-floor band.

    Measured on the dev box (ambient load 4.4, 5 samples): worst gaps
    15.4ms of 129.0-132.0ms walls (ratio 0.116-0.119): the word_bounds_iter
    band exactly (15.4ms there: the ping floor plus the two-thread GIL
    contention of the draining thread's per-``__next__`` bytecode), inside
    both shared budgets (~2.5x ratio margin, ~6.5x ceiling margin).

    The wall contrast, measured in the same process (min-of-3, ambient load
    3.1): full drain 117.4ms against the list API's 211.1ms and the count
    core's 28.3ms: the drain is the ~28-31ms construction scan plus
    ~86-90ms of 1.28M per-``__next__`` 3-tuple handoffs (~70ns per match,
    against the list shape's ~0.13µs per match of GIL-HELD marshalling), so
    the iterator wins BOTH axes at this size (the word_bounds_iter
    finding's shape: 347ms vs 724ms there). A discrimination caveat: the
    construction pass is only ~31ms of the ~130ms wall, so a
    construction-only detach regression shows ~41ms gaps at ratio ~0.32;
    the 0.30 budget catches it, but thinly; the grosser classes (a
    per-``__next__`` re-scan, per-next chunk marshalling) blow through both
    budgets."""
    text = prose(size_bytes)
    patterns = list(SEARCH_DENSE_PATTERNS)

    def consume() -> int:
        return sum(1 for _ in tors.find_patterns_iter(patterns, text))

    asyncio.run(_assert_loop_stays_responsive(lambda: asyncio.to_thread(consume)))


# tors.quote's 32 MiB ratio budget: NOT the shared 0.30, which this wall
# cannot attain: the output is ~5/4 the input (roughly every sixth prose
# char is a space escaping to %20), so the O(output) marshalling residue
# is a structurally large share of a fast-scan wall, the b64_encode_bytes
# shape (see _B64_RATIO_BUDGET's comment). Measured band (ambient load
# ~5): worst gaps 24.3-28.8ms of 55-60ms walls (ratio 0.44-0.48). 0.60
# sits ~1.25x above the worst measured ratio and ~40% below the ~1.0 a
# detach regression shows (the whole wall held); a several-fold
# marshalling blowout crosses 0.60 or the 100ms ceiling (~3.5x
# above the band). If a crate bump ever speeds the detached scan enough
# to collapse the wall toward the marshalling share, recalibrate per the
# QC-Yes 12 MiB precedent rather than letting the budget flake.
_QUOTE_32MIB_RATIO_BUDGET = 0.60


@pytest.mark.parametrize(
    ("size_bytes", "ratio_budget"),
    [(12 * _MIB, None), (32 * _MIB, _QUOTE_32MIB_RATIO_BUDGET)],
    ids=["12MiB-ceiling-only", "32MiB-both-budgets"],
)
def test_quote_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int, ratio_budget: float | None
) -> None:
    """The percent-encode claim: the whole escape scan runs under
    ``py.detach``, and the GIL-held residue is the argument borrow plus
    the marshalling of ONE output string (~5/4 the input at this corpus),
    the no-list-shape class, ``replace_many``'s band shape.

    Measured on the dev box (ambient load 0.4-5.2, 3 samples per cell,
    prose corpus):

    - 12 MiB: worst gaps 10.1-10.8ms against walls of 10.1-23.5ms: the
      wall STRADDLES the 10ms ping floor (a fast native pass: ~20ms
      inline), so any gap/wall ratio is the suite's documented sub-ping
      artifact and the cell asserts the 100ms ceiling only (the b64
      12 MiB precedent). A limitation: a held ~20ms encode passes
      the ceiling too, so the RATIO discriminator for this function lives
      in the 32 MiB cell and the red-side cells below.
    - 32 MiB: worst gaps 24.3-28.8ms of 55.2-59.7ms walls (ratio
      0.44-0.48): the ping floor plus the ~43 MiB output's marshalling
      band, the b64 structural shape (the output is larger than the
      input, so the marshalling is a larger wall share than finalize's).
      The 0.60 cell budget sits ~1.25x above the worst measured ratio;
      a detach regression (the scan itself GIL-held) shows ratio ~1.0
      and fails by far (see ``_QUOTE_32MIB_RATIO_BUDGET``'s comment).

    The red side, same corpus and placement, is the whole reason the
    function exists: ``urllib.parse.quote`` is pure Python; its
    measured bands and the structural surprise (the ≥200KB CHUNKED path
    gh-95865 added) are pinned by the red-side cells below."""
    corpus = prose(size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.quote, corpus),
            ratio_budget=ratio_budget,
        )
    )


@pytest.mark.parametrize(
    ("size_bytes", "ratio_budget"),
    [(12 * _MIB, None), (32 * _MIB, _RATIO_BUDGET)],
    ids=["12MiB-ceiling-only", "32MiB-both-budgets"],
)
def test_unquote_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int, ratio_budget: float | None
) -> None:
    """The percent-decode claim, on the encoder's own output (the corpus
    is ``tors.quote(prose(...))``, the canonical decoder input, its
    round trip pinned in tests/test_url_quote.py): the whole fragment
    walk + replace-decode runs under ``py.detach``, the residue is the
    marshalling of ONE decoded string (3/4 the encoded size).

    Measured on the dev box (ambient load 0.4-5.2, 3 samples per cell):

    - 12 MiB decoded (16 MiB encoded): worst gaps 10.8-11.1ms against
      walls of 14.3-23.5ms: the wall straddles the ping floor, the
      sub-ping artifact; the cell asserts the 100ms ceiling only (the
      b64 12 MiB precedent).
    - 32 MiB decoded (43 MiB encoded): worst gaps 15.5-16.5ms of
      77.0-77.8ms walls (ratio 0.20-0.21): the floor plus the ~32 MiB
      output's marshalling. Every sample inside BOTH shared budgets
      (~1.4x ratio margin, ~6x ceiling margin); a detach regression
      holds the whole ~77ms wall at ratio ~1.0 and fails by far.

    The red side: ``urllib.parse.unquote`` over the same encoded corpus
    measured worst gaps 50.5-52.0ms of 217.6-221.5ms walls (ratio
    0.23-0.24): it PASSES both budgets (recorded in the red-side cells
    below, with the structural reason), so the statement is the
    wall (~12x: ~18ms tors inline vs ~223ms stdlib) and the worst gap
    (~5x), not "blocked vs unblocked" (the html.unescape red-side
    finding's shape)."""
    corpus = tors.quote(prose(size_bytes))
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.unquote, corpus),
            ratio_budget=ratio_budget,
        )
    )


def _close_matches_corpus() -> tuple[str, list[str]]:
    """One query × 13,900 candidates for the bulk get_close_matches
    cells: the prose corpus's own four-sentence unit (664 chars) with
    the diff pair's word swap as the query, and every single-character
    substitution of the unit (lowercase letters at the lowercase
    positions) as the candidate pool: the corpus's own text at a
    length where difflib's quadratic char matcher costs seconds
    (measured ~3.4s) and tors's per-candidate Myers passes are real
    work (~65ms). The short-candidate alternative (every mutation of
    the 23-word vocabulary (~7.9k × ~8 chars)) measures BOTH engines at
    ~7.5ms (difflib is fast on short candidates; the wall race's
    seconds live at sentence length), so the bulk cell's pool is this
    one."""
    sentence = prose(64 * 1024).split("\n\n")[0]
    query = sentence.replace("quarterly", "monthly", 1)
    pool = list(
        dict.fromkeys(
            sentence[:i] + c + sentence[i + 1 :]
            for i in range(len(sentence))
            if sentence[i].isalpha() and sentence[i].islower()
            for c in string.ascii_lowercase
            if c != sentence[i]
        )
    )
    return query, pool


def test_get_close_matches_bulk_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity() -> None:
    """The bulk-similarity claim: one query × 13,900 664-char candidates
    (the corpus builder above): the sweep's per-candidate Myers passes
    run under ``py.detach``, and the GIL-held residue is the O(candidates)
    argument walk plus the ping floor: the returned elements are the
    ORIGINAL candidate objects (references, zero marshalling per hit,
    pinned in tests/test_similarity.py), so there is no list-shape class
    at all.

    Measured on the dev box (ambient load 4.1-5.2, 3 samples): worst
    gaps 10.6-11.3ms of 64.2-71.7ms walls (ratio 0.15-0.18): inside
    BOTH shared budgets (~1.7x ratio margin, ~9x ceiling margin); a
    detach regression (the sweep itself GIL-held) pins the whole ~65ms
    wall at ratio ~1.0 and fails by far. The wall race against difflib
    on the same input (~50x, single sample; the difflib precedent) is
    the cell below."""
    query, pool = _close_matches_corpus()
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.get_close_matches, query, pool, 5)
        )
    )


def _quote_unchunked_core(text: str) -> str:
    """``urllib.parse.quote``'s per-byte core statement: the
    ``''.join(map(quoter, bytes))`` that ``quote_from_bytes`` runs for
    inputs UNDER 200 KB, with the quoter table built here rather than
    reaching into ``urllib.parse``'s private ``_byte_quoter_factory``
    (absent on 3.10, measured): the stdlib's ``_Quoter`` is a dict
    subclass whose ``__getitem__`` this fully-populated table's is. The
    join over a C-callable mapper holds the GIL for its whole duration,
    i.e. the GIL-held whole-text pass the stdlib spelling IS below the
    chunking threshold; the value-parity assert in the red-side cell
    proves the table is the stdlib's own."""
    always_safe = frozenset(string.ascii_letters + string.digits + "_.-~" + "/")
    table = {
        byte: chr(byte) if chr(byte) in always_safe else f"%{byte:02X}"
        for byte in range(256)
    }
    return "".join(map(table.__getitem__, text.encode("utf-8")))


def test_the_gil_held_url_red_sides_fail_their_budgets_in_every_sample() -> None:
    """The RED side, asserted mechanically: the per-byte join that IS
    ``urllib.parse.quote``'s core (the ``<200 KB`` statement; see
    ``_quote_unchunked_core``) holds the GIL for its whole duration, so
    the budgets tors's quote cells above pass must be FAILED by it in
    every sample, judged by the same ``_budget_misses`` list the green
    cells use. Measured on the dev box (ambient load 4.1-11.8, 3
    samples, 12 MiB prose): worst gaps 179.8-187.8ms of 180.0-188.0ms
    walls: ratio 0.999, the whole-text hold, missing BOTH budgets in
    every sample (the single-C-hold shape the b64 96 MiB red side
    demonstrates for ``b64encode``).

    Measured and NOT asserted: the two stdlib red
    expressions that PASS the budgets at this size, with the structural
    reasons (the b64 12/32 MiB red-side precedent: record the shape,
    assert only what discriminates):

    - ``urllib.parse.quote(text)`` itself, same corpus and placement:
      worst gaps 15.6-22.0ms of 163.9-183.0ms walls (ratio 0.09-0.12;
      ambient load 0.4-32 across the measurement windows).
      The surprise vs the whole-text-hold expectation: inputs ≥200 KB
      take the CHUNKED path gh-95865 added (``chunk_size =
      isqrt(len)``), whose per-chunk joins hold only µs-bursts between
      the chunk comprehension's bytecode; the eval loop drops the GIL
      between chunks, so the worst gap is a burst, not the wall. The
      unchunked core above is what the function runs below the
      threshold and ran at every size before the chunking; the race in
      WALL is unchanged (~8x: tors ~20ms inline vs stdlib 158-183ms).
    - ``urllib.parse.unquote(encoded)`` (the 12 MiB percent-encoded
      corpus): worst gaps 50.5-65.0ms of 217.6-267.0ms walls (ratio
      0.23-0.25). Its ``_generate_unquoted_parts`` generator yields
      bytecode per ASCII run (eval-loop GIL drops), and the worst gap
      is set by the per-run C work between those yield points: the
      html.unescape red-side class ("a fifth of the block time, an
      eighth of the wall" there; ~5x/~12x here).

    The live measurements of both recorded legs run in the sister cell
    below, in the same to_thread cell style, with their value parity
    asserted as a 12 MiB differential anchor."""
    corpus = prose(12 * _MIB)
    # The table is the stdlib's: the core statement produces quote()'s
    # own output, and tors agrees with both (the 12 MiB parity anchor).
    assert _quote_unchunked_core(corpus) == tors.quote(corpus) == urllib.parse.quote(corpus)
    observed = [
        asyncio.run(_gap_and_wall_during(lambda: asyncio.to_thread(_quote_unchunked_core, corpus)))
        for _ in range(_SAMPLES)
    ]
    for gap, wall in observed:
        assert _budget_misses(gap, wall, _RATIO_BUDGET), (
            f"the unchunked quote core measured a CLEAN sample ({gap * 1000:.0f}ms of a "
            f"{wall * 1000:.0f}ms wall): the budgets tors's quote cells pass no longer "
            "discriminate against the GIL-held whole-text join they replace, and a "
            "detach regression could pass them too (tests/test_gil_release.py)"
        )


def _measure_red(red: Callable[[str], str], corpus: str) -> list[tuple[float, float]]:
    """Measure a stdlib red expression in the green cells' own cell style
    (``asyncio.to_thread`` under the heartbeat), ``_SAMPLES`` samples."""
    return [
        asyncio.run(_gap_and_wall_during(lambda: asyncio.to_thread(red, corpus)))
        for _ in range(_SAMPLES)
    ]


def test_urllib_quote_and_unquote_red_sides_measured_value_parity_and_recorded() -> None:
    """The stdlib red sides, measured LIVE in the same to_thread cell
    style, recorded, NOT budget-asserted (both PASS the shared budgets
    at this size; the structural reasons are in the asserted red-side
    cell's docstring above). What IS asserted is the VALUE parity (the
    12 MiB differential anchor for both directions, the same equality
    the hypothesis gates in tests/test_url_quote.py prove at small
    sizes, here at corpus scale), and the bands are printed so every
    run's log carries the recorded divergence shape."""
    text = prose(12 * _MIB)
    encoded = tors.quote(text)
    # The 12 MiB parity anchors (one inline call each side).
    assert urllib.parse.quote(text) == tors.quote(text)
    assert urllib.parse.unquote(encoded) == tors.unquote(encoded)

    note = "urllib red side (recorded, not asserted, passes the budgets; see the red-side cell): "
    for name, red, corpus in (
        ("urllib.parse.quote", urllib.parse.quote, text),
        ("urllib.parse.unquote", urllib.parse.unquote, encoded),
    ):
        observed = _measure_red(red, corpus)
        for gap, wall in observed:
            print(
                f"{note}{name} blocked {gap * 1000:.0f}ms of a {wall * 1000:.0f}ms "
                f"operation ({gap / wall:.0%})"
            )


# The bulk wall race's tolerance margin: tors's measured wall (~65ms)
# against difflib's (~3.4s) is a ratio of ~0.02, so 0.25 leaves >10x
# headroom; the _DIFF_WALL_MARGIN precedent (the assertion pins the
# quadratic-vs-native relationship, not a close race).
_GCM_WALL_MARGIN = 0.25


def test_get_close_matches_beats_difflib_on_the_bulk_corpus() -> None:
    """The wall race at the bulk shape: difflib's per-candidate
    ``SequenceMatcher`` is quadratic in the common region, and 664-char
    candidates with a one-word-different query put nearly the whole
    candidate in that region: ~0.24ms per candidate, ~3.4s for the
    13,900-candidate sweep (measured once, single sample: noise only
    ever adds time, so one sample is conservative for the denominator;
    the difflib precedent test_diff_opcodes.py set at its 256 KiB
    race). tors runs the same sweep's Myers passes natively: measured
    64.2-71.7ms walls (ambient load 4.1-5.2), ratio ~0.02 against the
    0.25 margin.

    difflib's own GIL band, measured in the same to_thread placement:
    worst gaps 15.5-20.5ms of 3,346-3,931ms walls (~0.5%): the sweep
    is pure-Python bytecode, so the eval loop drops the GIL at switch
    intervals and the loop stays schedulable; the WALL is the finding
    (~50x), not the blocking; recorded here so the race is not
    misread as a GIL claim. tors's GIL band is the green bulk cell
    above."""
    query, pool = _close_matches_corpus()
    tors_walls = [
        asyncio.run(
            _gap_and_wall_during(lambda: asyncio.to_thread(tors.get_close_matches, query, pool, 5))
        )
        for _ in range(_SAMPLES)
    ]
    tors_wall = min(wall for _, wall in tors_walls)
    difflib_gap, difflib_wall = asyncio.run(
        _gap_and_wall_during(lambda: asyncio.to_thread(difflib.get_close_matches, query, pool, 5))
    )
    print(
        f"difflib.get_close_matches red side (recorded): blocked {difflib_gap * 1000:.0f}ms "
        f"of a {difflib_wall * 1000:.0f}ms wall ({difflib_gap / difflib_wall:.1%}): "
        "pure-Python eval-loop yields, the html.unescape red-side class; the WALL is the finding"
    )
    assert tors_wall < _GCM_WALL_MARGIN * difflib_wall, (
        f"bulk get_close_matches: tors {tors_wall * 1000:.0f}ms vs difflib "
        f"{difflib_wall * 1000:.0f}ms (ratio {tors_wall / difflib_wall:.4f}): the native "
        "sweep lost more than the tolerance margin to the quadratic stdlib matcher"
    )
