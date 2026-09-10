"""Wall-time proof: tors's native passes versus the expressions they replace,
measured in the same process so both sides pay identical machine conditions.
This is load-fair by construction: a shared-runner slowdown inflates both
sides together, which is why the threshold can stay tight.

``tors.finalize`` vs the pure-Python reference finalize, measured on the dev
box this test was written on (WSL2, 28 logical cores, ambient load ~3.5, CPython
3.12.7), min-of-5 per side after one warm-up call:

    corpus      size   tors        reference   tors/ref
    prose       1 KiB  0.007ms     0.008ms     0.80  (wins, but thin: µs-scale call
                                                      overhead dominates; recorded here,
                                                      not CI-asserted)
    prose       1 MiB  8.61ms      12.82ms     0.67
    prose       12 MiB 105.8ms     163.0ms     0.65
    decomposed  1 MiB  9.06ms      16.94ms     0.53
    decomposed  12 MiB 109.8ms     246.3ms     0.45
    crlf        1 MiB  8.85ms      13.58ms     0.65
    crlf        12 MiB 107.3ms     162.8ms     0.66

The CI assertion covers 1 MiB and 12 MiB over all three corpora with a tolerant
``tors < 0.9 x reference`` margin: the measured ratios sit at 0.45-0.67, so the margin
absorbs a loaded 2-vCPU CI runner (min-of-3 on both sides; noise only ever adds time).
If tors ever fails to beat the reference at an asserted size, that is a design failure
of the native pass, not something to threshold away; see the numbers above for what
healthy looks like.

The bytes-in functions use the same methodology, with corpora from
``reference.corpus_utf8`` (the same recipes as UTF-8 bytes; ambient load 5.4;
min-of-5 after warmup):

    ``tors.b64_encode_bytes`` vs ``base64.b64encode(raw).decode("ascii")``: a win at
    every cell, asserted with the same 0.9 margin:

        corpus      size   tors        stdlib     tors/stdlib
        prose       1 MiB  1.07ms      1.32ms     0.81
        decomposed  1 MiB  0.30ms      0.87ms     0.34
        crlf        1 MiB  0.30ms      0.86ms     0.34
        prose       12 MiB 14.60ms     17.46ms    0.84  (a contended outlier: the
                                                         same cell measured 4.2ms vs
                                                         10.9ms, ratio 0.38, at load
                                                         ~5.6 in an adjacent pass;
                                                         recorded as the worst observed)
        decomposed  12 MiB 4.76ms      11.28ms    0.42
        crlf        12 MiB 5.60ms      18.15ms    0.31

    ``tors.finalize_utf8`` vs the decode-plus-reference tail it replaces
    (``raw.decode`` + the pure-Python reference finalize): a win at every cell,
    asserted with the same margin:

        corpus      size   tors        decode+ref  tors/ref
        prose       1 MiB  10.0ms      13.4ms      0.75
        decomposed  1 MiB  9.6ms       20.1ms      0.48
        crlf        1 MiB  9.3ms       14.3ms      0.65
        prose       12 MiB 113.1ms     166.3ms     0.68
        decomposed  12 MiB 121.7ms     263.4ms     0.46
        crlf        12 MiB 120.2ms     170.5ms     0.70

    ``tors.decode_utf8`` vs ``raw.decode("utf-8")``: not asserted, and reported as a
    measured loss, ratios 1.07-1.60 across the six cells (e.g. prose 12 MiB 0.61ms vs
    0.45ms; decomposed 12 MiB 3.82ms vs 2.81ms). CPython's UTF-8 decode is a C
    fast-path with no Python-level overhead, and tors marshals its result through the
    same str construction, so there is no wall-time case for ``decode_utf8`` standalone.
    Its value is the GIL release (pinned in tests/test_gil_release.py), CPython-parity
    guarantees (tests/test_decode_utf8.py), and its role inside ``finalize_utf8``.
    Deliberately recorded, not thresholded away.
"""

from __future__ import annotations

import base64
import html
import time
from collections.abc import Callable

import pytest

import tors
from reference import corpus_b64, corpus_utf8, crlf, decomposed, entities, prose, reference_finalize
from tors import (
    chunk_by_lines,
    chunk_by_paragraphs,
    chunk_by_sentences,
    chunk_by_words,
    chunk_hierarchical,
)

# The timing lane: every test in this module is a measurement cell (wall-time
# bands and races over multi-MiB corpora), slow and load-sensitive, so CI's
# matrix legs deselect it (`-m "not timing"`) and one dedicated step on the
# 3.12 leg runs it (`-m timing`). The wall contracts still gate every push,
# once, instead of being paid on all five legs. Locally a bare ``pytest``
# (and ``make test``) run everything, marker or not.
pytestmark = pytest.mark.timing

_MIB = 1024 * 1024
_MARGIN = 0.9
_SAMPLES = 3


def _min_wall_ms(
    op: Callable[[str | bytes], object], corpus: str | bytes, warmup: int = 1
) -> float:
    """Min-of-``_SAMPLES`` wall after warmup. The corpus parameter is ``str |
    bytes`` because the suite measures both str-in functions (``finalize`` over
    the reference corpora) and bytes-in functions (the surface over
    ``corpus_utf8``)."""
    for _ in range(warmup):
        op(corpus)
    best = float("inf")
    for _ in range(_SAMPLES):
        started = time.monotonic()
        op(corpus)
        best = min(best, time.monotonic() - started)
    return best * 1000.0


def _stdlib_b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _stdlib_b64_decode(s: str) -> bytes:
    """The stdlib expression ``tors.b64_decode`` replaces."""
    return base64.b64decode(s)


def _stdlib_html_unescape(text: str) -> str:
    """The stdlib expression ``tors.html_unescape`` replaces."""
    return html.unescape(text)


def _stdlib_decode_finalize(raw: bytes) -> tuple[str, str]:
    """The replaced tail today: GIL-held decode plus the pure-Python reference
    finalize (normalize + sha256), what ``tors.finalize_utf8`` collapses into one
    detached call."""
    return reference_finalize(raw.decode("utf-8"))


@pytest.mark.parametrize("corpus_kind", ["prose", "decomposed", "crlf"])
@pytest.mark.parametrize("size_bytes", [1 * _MIB, 12 * _MIB], ids=["1MiB", "12MiB"])
def test_finalize_beats_the_reference_pipeline_on_the_same_corpus(
    corpus_kind: str, size_bytes: int
) -> None:
    corpus = {"prose": prose, "decomposed": decomposed, "crlf": crlf}[corpus_kind](size_bytes)
    tors_ms = _min_wall_ms(tors.finalize, corpus)
    ref_ms = _min_wall_ms(reference_finalize, corpus)
    assert tors_ms < _MARGIN * ref_ms, (
        f"{corpus_kind} {size_bytes // _MIB}MiB: tors {tors_ms:.1f}ms vs "
        f"reference {ref_ms:.1f}ms (ratio {tors_ms / ref_ms:.2f}): the native pass lost "
        "more than the tolerance margin to the pure-Python pipeline"
    )


@pytest.mark.parametrize("corpus_kind", ["prose", "decomposed", "crlf"])
@pytest.mark.parametrize("size_bytes", [1 * _MIB, 12 * _MIB], ids=["1MiB", "12MiB"])
def test_b64_encode_bytes_beats_the_stdlib_expression_on_the_same_corpus(
    corpus_kind: str, size_bytes: int
) -> None:
    """The content-addressing swap's wall case: the stdlib expression holds the GIL for
    ``b64encode`` and pays a second 4/3-sized str construction in ``decode("ascii")``;
    tors encodes detached and marshals one output. Measured ratios 0.31-0.84 (the
    worst a contended prose 12 MiB cell; see the module docstring)."""
    corpus = corpus_utf8(corpus_kind, size_bytes)
    tors_ms = _min_wall_ms(tors.b64_encode_bytes, corpus)
    std_ms = _min_wall_ms(_stdlib_b64, corpus)
    assert tors_ms < _MARGIN * std_ms, (
        f"{corpus_kind} {size_bytes // _MIB}MiB: tors {tors_ms:.1f}ms vs "
        f"stdlib {std_ms:.1f}ms (ratio {tors_ms / std_ms:.2f}): the native b64 pass "
        "lost more than the tolerance margin to the stdlib expression"
    )


@pytest.mark.parametrize("corpus_kind", ["prose", "decomposed", "crlf"])
@pytest.mark.parametrize("size_bytes", [1 * _MIB, 12 * _MIB], ids=["1MiB", "12MiB"])
def test_finalize_utf8_beats_the_decode_plus_reference_pipeline_on_the_same_corpus(
    corpus_kind: str, size_bytes: int
) -> None:
    """The one-call shape's wall case: decode, normalize, and hash in a single detached
    pass vs the tail it replaces (a GIL-held decode plus the pure-Python
    reference finalize). Measured ratios 0.46-0.75."""
    corpus = corpus_utf8(corpus_kind, size_bytes)
    tors_ms = _min_wall_ms(tors.finalize_utf8, corpus)
    ref_ms = _min_wall_ms(_stdlib_decode_finalize, corpus)
    assert tors_ms < _MARGIN * ref_ms, (
        f"{corpus_kind} {size_bytes // _MIB}MiB: tors {tors_ms:.1f}ms vs "
        f"decode+reference {ref_ms:.1f}ms (ratio {tors_ms / ref_ms:.2f}): the "
        "one-call native pass lost more than the tolerance margin to the "
        "decode-then-finalize pipeline"
    )


@pytest.mark.parametrize("corpus_kind", ["prose", "decomposed", "crlf"])
@pytest.mark.parametrize("size_bytes", [1 * _MIB, 12 * _MIB], ids=["1MiB", "12MiB"])
def test_b64_decode_wall_time_vs_the_stdlib_is_measured_not_asserted(
    corpus_kind: str, size_bytes: int
) -> None:
    """``tors.b64_decode`` vs ``base64.b64decode`` (the b64-encoded corpora from
    ``reference.corpus_b64``), measured and not asserted for the
    same reason as the ``decode_utf8`` case: CPython's ``a2b_base64`` is
    a C fast-path with no Python-level overhead, and tors marshals its result
    through the same ``PyBytes`` construction, so there is no wall-time case
    for the standalone decode. Measured on the dev box (ambient load 6.0,
    min-of-3 after warmup):

        corpus      size   tors        stdlib     tors/stdlib
        prose       1 MiB  0.99ms      0.96ms     1.03
        decomposed  1 MiB  1.18ms      1.21ms     0.97
        crlf        1 MiB  1.04ms      1.02ms     1.03
        prose       12 MiB 13.04ms     15.05ms    0.87
        decomposed  12 MiB 13.32ms     13.96ms    0.95
        crlf        12 MiB 13.19ms     13.98ms    0.94

    Even-parity at 1 MiB, a thin win at 12 MiB (0.87-0.95, under the 0.9
    assertion margin on two of three corpora, which is why nothing is
    asserted). The value is the GIL release (pinned in tests/test_gil_release.py:
    stdlib ratio 0.98-0.99 held in every sample vs tors's 0.50-0.78 band at
    12 MiB), the error-path parity contract (tests/test_b64_decode.py), and
    symmetry with ``b64_encode_bytes``. Recorded, not thresholded away."""
    corpus = corpus_b64(corpus_kind, size_bytes)
    tors_ms = _min_wall_ms(tors.b64_decode, corpus)
    std_ms = _min_wall_ms(_stdlib_b64_decode, corpus)
    print(
        f"b64_decode {corpus_kind} {size_bytes // _MIB}MiB: tors {tors_ms:.2f}ms "
        f"stdlib {std_ms:.2f}ms ratio {tors_ms / std_ms:.2f}"
    )


@pytest.mark.parametrize("size_bytes", [1 * _MIB, 12 * _MIB], ids=["1MiB", "12MiB"])
def test_html_unescape_beats_the_stdlib_on_entity_bearing_prose(
    size_bytes: int,
) -> None:
    """The wall headline: ``tors.html_unescape`` vs ``html.unescape`` on
    the entities corpus (~7% entity density). The stdlib's cost is structural:
    a ``re.sub`` whose every match invokes the Python callback
    ``_replace_charref``, so the native scan wins by ~3x regardless of load
    (the callback count, not the machine, dominates). Measured on the dev box
    (ambient load 10.0, min-of-3 after warmup):

        size    tors        stdlib      tors/stdlib
        1 MiB   6.3ms       18.1ms      0.346
        12 MiB  82.5ms      257.9ms     0.320

    Asserted with the same 0.9 margin as the other wall cells. The measured
    ratios (0.32-0.35, under the heaviest load window in these tables) leave
    ~2.8x of headroom."""
    corpus = entities(size_bytes)
    tors_ms = _min_wall_ms(tors.html_unescape, corpus)
    std_ms = _min_wall_ms(_stdlib_html_unescape, corpus)
    assert tors_ms < _MARGIN * std_ms, (
        f"entities {size_bytes // _MIB}MiB: tors {tors_ms:.1f}ms vs "
        f"stdlib {std_ms:.1f}ms (ratio {tors_ms / std_ms:.2f}): the native "
        "entity scan lost more than the tolerance margin to the regex+callback"
    )


def test_html_unescape_no_ampersand_path_is_measured_not_asserted() -> None:
    """The degenerate path, measured and not asserted: on text
    with no ``&`` at all, ``html.unescape`` is a single C ``in`` check
    (~0.10ms at 12 MiB) while tors pays the same class of scan through a
    bytewise ``contains`` (~0.58ms at 12 MiB, min-of-3 at load 10.0), a
    measured loss by ~5x on a path that is microseconds-vs-milliseconds in
    absolute terms either way. The zero-copy identity return (the wrapper
    returns the input object, exactly as CPython's ``return s`` does;
    measured: ``tors.html_unescape(s) is s``) removes the O(n) marshalling
    copy this path would otherwise pay (1.17ms before it), leaving only the
    scan itself. Recorded, not thresholded away: no caller routes
    megabytes of ampersand-free text through an entity decoder for its speed,
    and the wall cells that matter are the entity-bearing ones above."""
    corpus = prose(12 * _MIB)
    tors_ms = _min_wall_ms(tors.html_unescape, corpus)
    std_ms = _min_wall_ms(_stdlib_html_unescape, corpus)
    print(
        f"html_unescape no-& prose 12MiB: tors {tors_ms:.2f}ms "
        f"stdlib {std_ms:.2f}ms ratio {tors_ms / std_ms:.2f}"
    )


@pytest.mark.parametrize("corpus_kind", ["prose", "decomposed"])
@pytest.mark.parametrize("size_bytes", [1 * _MIB, 12 * _MIB], ids=["1MiB", "12MiB"])
def test_grapheme_count_absolute_band_holds(corpus_kind: str, size_bytes: int) -> None:
    """``tors.grapheme_count`` has no stdlib comparator (the gap is the
    feature), so its wall cell is an absolute band, a regression ceiling
    with generous margin, not a race. Measured on the dev box (ambient load
    8.3, min-of-3 after warmup):

        corpus      size    time        throughput
        prose       1 MiB   ~10ms       ~100 MiB/s
        decomposed  1 MiB   9.7ms       ~105 MiB/s
        prose       12 MiB  122.8ms     ~102 MiB/s
        decomposed  12 MiB  117.6ms     ~104 MiB/s

    The band: 12 MiB must complete under 400ms (~3.3x above the measured
    118-123ms; the scan is a linear walk, stable under load, so a regression to
    a quadratic or per-cluster-allocation path blows straight through it).
    The GIL-release side of this function is pinned separately in
    tests/test_gil_release.py (worst gaps 10.4-12.8ms at 12 MiB, a single
    int return with no marshalling class at all)."""
    corpus = {"prose": prose, "decomposed": decomposed}[corpus_kind](size_bytes)
    took_ms = _min_wall_ms(tors.grapheme_count, corpus)
    assert took_ms < 400.0, (
        f"grapheme_count {corpus_kind} {size_bytes // _MIB}MiB took {took_ms:.0f}ms, "
        "outside the absolute band (measured ~118-123ms at 12 MiB, ceiling 400ms "
        "with 3.3x margin); the cluster scan regressed"
    )


# ---------------------------------------------------------------------------
# The chunking family's document-scale cost shape (#22, then #30): the
# per-call cost must be the segmentation walks the function is FOR, not
# per-codepoint structures built unconditionally — and, since #30's lazy
# levels, only the walks a call actually CONSULTS. Load-fair ratios (both
# sides measured in the same process, min-of-3 after warmup), so a
# shared-runner slowdown inflates both sides together.
#
# Measured on the dev box (Linux, CPython 3.12, min-of-3 after warmup),
# before the fix -> after:
#
#     chunk_hierarchical, 12 MiB degenerate single-char run, custom
#     never-matching separators, whole-document budget: ~1048ms -> ~2.6ms
#     after #22 (the unconditional grapheme HashSet + Vec<char> collect;
#     ~400x); after #30's lazy levels the same call pays only the
#     codepoint count — the literal scan itself is skipped, since no
#     window ever opens a level
#
#     chunk_hierarchical, 12 MiB prose, default hierarchy: ~2996ms ->
#     ~351ms after #22; -> ~1.2ms at a 2000-char budget after #30's lazy
#     levels (every window is served by the paragraph level alone on this
#     corpus, so the sentence/word walks are never built), ~0.1ms under a
#     whole-document budget (no level consulted at all; ~340ms before,
#     ~176ms at 6 MiB), and ~396ms at a 100-char budget that genuinely
#     descends to the word level — the word (~132ms) plus sentence
#     (~188ms) walks it actually uses, the accurate UAX #29 hierarchy
#     the function exists to provide
#
#     chunk_by_words, 12 MiB prose, 200 words/chunk: ~1923ms -> ~165ms
#     chunk_by_sentences, 12 MiB prose, 10 sentences/chunk:
#     ~1328ms -> ~200ms
#
# Follow-up pass (lazy level builds, byte-level line/paragraph scans,
# char_count ASCII fast path): default hierarchy @2000 12 MiB prose
# ~351ms -> ~1.2ms (a paragraph-scale budget consults only the
# paragraph level; the sentence/word walks never run), whole-document
# budgets ~340ms -> ~0.1ms (one codepoint count, no level consulted,
# and the count itself is the ASCII fast path), by-lines @50 12 MiB
# ~8ms -> ~0.5ms and by-paras on a log-density corpus ~7.5ms -> ~1.4ms
# (density-guarded byte scan with a sliding ASCII certificate;
# non-ASCII segments keep the byte path per segment). The
# dedup/laziness contract cells (min-of-3 after warmup): default
# hierarchy @2000 over 12 MiB ~1.5ms vs ~1.5ms for a ["\n\n"]-only
# custom hierarchy (ratio ~1.0; the eager pre-#30 spelling measured
# ~351ms on the same corpus); and [None]*8 / [" "]*8 at a descending
# 8-codepoint budget over 6 MiB each ~1.0x their lone-entry spellings
# with the slot-construction dedups in the built extension (the
# duplicated [" "] measured ~1.7x pre-dedup -- the literal spelling of
# the [None]*8 walk re-payment the dedup closes). The docstrings below
# carry the detail.
# ---------------------------------------------------------------------------


def test_chunk_hierarchical_custom_no_match_is_scan_cost_not_per_char_structures() -> None:
    """The whole-document-budget custom-hierarchy cell from #22: a
    never-matching separator list under a budget that swallows the whole
    text must cost ONE CODEPOINT COUNT and nothing else. Under the
    lazy-level pass the level is never consulted (the first window fits
    the entire document), so it is never built and never scanned -- the
    literal's scan does not run at all, where the pre-lazy spelling paid
    one count pass plus one scan pass. The reference stays CPython's own
    ``in``, a C-speed scan of the same text, because the class of
    regression this cell guards (an unconditional per-codepoint
    structure or scan built before the budget is even examined) shows up
    as wall time against exactly that reference. With the char_count
    ASCII fast path the count lane itself drops to ~0.1ms at 12 MiB, so
    the measured ratio against ``in`` is ~0.05 (count pass only; ~2
    after #22, when the scan still ran on top). The pre-#22 spelling
    measured ~700x.

    The absolute ceiling (0.8ms) is the fast path's OWN pin, which the
    8x scan race above cannot provide: a char_count reverted to the
    predicate-only spelling (no ``is_ascii`` gate) measures ~2.5ms on
    this corpus (red-proofed: that revert fails this cell) — which still
    passes 8x an ~8ms scan, so the race is blind to exactly the
    fast-path loss. 0.8ms sits ~3-5x above the measured band
    (0.15-0.25ms, min-of-3 after warmup on the box this ceiling was
    calibrated on) and ~3x below the predicate-only spelling, so the
    gate's loss FAILS this cell while CI load does not."""
    q = "q" * (12 * _MIB)
    tors_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 12 * _MIB, ["xyz"]), q)
    scan_ms = _min_wall_ms(lambda s: "xyz" in s, q)
    assert tors_ms < 8.0 * scan_ms, (
        f"chunk_hierarchical no-match 12MiB took {tors_ms:.1f}ms against an "
        f"{scan_ms:.1f}ms bare scan ({tors_ms / scan_ms:.0f}x); the lazily-built "
        "grapheme machinery regressed to an unconditional structure"
    )
    assert tors_ms < 0.8, (
        f"chunk_hierarchical no-match 12MiB took {tors_ms:.2f}ms, over the "
        "whole-budget absolute ceiling (measured ~0.15-0.25ms with the char_count "
        "ASCII fast path, ceiling 0.8ms; the predicate-only spelling measures "
        "~2.5ms and must fail this cell); the char_count ASCII fast path regressed"
    )


def test_chunk_hierarchical_custom_no_match_skips_the_grapheme_walk_on_non_ascii() -> None:
    """The laziness contract, on the input where it is load-bearing: for
    NON-ASCII text a whole-text level build is a full segmentation walk
    (the grapheme index ~150ms at 12 MiB), so a call that never consults
    a level (never-matching separators, whole-document budget: no cuts
    to filter, no raw-cut window, no overlap snap) must build nothing.
    Under the lazy-level pass the literal is never even scanned -- the
    whole call is one codepoint count (an O(n) walk on non-ASCII text)
    plus marshalling, where the pre-lazy spelling paid the scan pass too.
    The reference is the same ``in`` scan; measured ~2.6ms vs ~1.4ms on
    decomposed prose (ratio ~1.9) on this box (Linux, CPython 3.13,
    ambient load 42-85, min-of-3 after warmup), and both builds sit in
    the same 2.4-3.6ms noise-bound band -- no speedup claimed, the
    cheaper structure is the point. ~75x if the index were built
    unconditionally -- the ASCII fast path masks that regression on
    single-byte corpora, which is why this cell runs decomposed text."""
    corpus = decomposed(12 * _MIB)
    tors_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, len(s), ["xyz"]), corpus)
    scan_ms = _min_wall_ms(lambda s: "xyz" in s, corpus)
    assert tors_ms < 8.0 * scan_ms, (
        f"chunk_hierarchical no-match non-ASCII 12MiB took {tors_ms:.1f}ms against "
        f"an {scan_ms:.1f}ms bare scan ({tors_ms / scan_ms:.0f}x); the grapheme "
        "index is being built on a call that cannot use it"
    )


def test_chunk_hierarchical_whole_document_budget_pays_no_level_walks() -> None:
    """The #30 lazy-level headline cell: a whole-document budget consults
    no level at all -- the loop's first iteration takes its own
    ``remaining <= max_chars`` exit before any level is realized -- so
    the DEFAULT hierarchy must cost the same nothing the never-matching
    custom one does: one codepoint count, one chunk out. Ratioed against
    CPython's own ``in`` (a C-speed scan of the same text): measured
    ~0.1ms vs ~2ms (ratio ~0.05) after the lazy levels; the eager
    spelling measured ~340ms here (~170x, and ~176ms at 6 MiB) -- the
    three default walks paid for levels that supplied zero cuts.

    The absolute ceiling (0.8ms) is the char_count ASCII fast path's pin
    on this lane, the no-match cell's twin rationale: the 8x scan race
    cannot see the fast path's loss (a predicate-only count ~2.5ms at
    12 MiB still passes 8x an ~8ms scan), while 0.8ms sits ~2.4-5x above
    the measured band (0.16-0.33ms, min-of-3 after warmup on the box
    this ceiling was calibrated on) and ~3x below the predicate-only
    spelling (red-proofed: that revert fails this cell), so the
    ``is_ascii`` gate's loss FAILS this cell."""
    corpus = prose(12 * _MIB)
    tors_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, len(s)), corpus)
    scan_ms = _min_wall_ms(lambda s: "xyz" in s, corpus)
    assert tors_ms < 8.0 * scan_ms, (
        f"chunk_hierarchical whole-document default 12MiB took {tors_ms:.1f}ms "
        f"against an {scan_ms:.1f}ms bare scan ({tors_ms / scan_ms:.0f}x); "
        "levels are being built on a call that consults none of them"
    )
    assert tors_ms < 0.8, (
        f"chunk_hierarchical whole-document default 12MiB took {tors_ms:.2f}ms, "
        "over the whole-budget absolute ceiling (measured ~0.16-0.33ms with the "
        "char_count ASCII fast path, ceiling 0.8ms; the predicate-only spelling "
        "measures ~2.5ms and must fail this cell); the char_count ASCII fast "
        "path regressed"
    )


def test_chunk_hierarchical_default_hierarchy_is_its_own_segmentation_walks() -> None:
    """The default hierarchy's per-call cost contract: no more than its
    own UAX #29 walks (word_count + sentence_count on the same corpus,
    measured in-process), at a budget that genuinely descends to the
    word level so every walk is consulted and the contract has teeth.
    The hierarchy IS those walks; everything around them -- level
    realization, the cut filter, the chunk loop, marshalling -- must be
    marginal. Measured ~396ms against a ~310ms reference sum (ratio
    ~1.3) at a 100-codepoint budget (at a 2000-codepoint budget on this
    corpus every window is served by the paragraph level alone, so the
    same call drops to ~1.2ms -- the sentence/word walks are never
    built; the dedicated ["\\n\\n"]-ratio cell below pins that property
    against a tighter reference); the pre-#22 spelling measured ~9.4x
    (the grapheme hash set dominated the segmentation it was
    filtering). The assertion is the standing ceiling in the other
    direction: whatever levels a budget DOES consult, the machinery
    around the walks must not dominate them."""
    corpus = prose(12 * _MIB)
    tors_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 100), corpus)
    walks_ms = _min_wall_ms(tors.word_count, corpus) + _min_wall_ms(tors.sentence_count, corpus)
    assert tors_ms < 2.0 * walks_ms, (
        f"chunk_hierarchical default 12MiB took {tors_ms:.0f}ms against "
        f"{walks_ms:.0f}ms of its own segmentation walks "
        f"({tors_ms / walks_ms:.1f}x); per-call machinery is dominating the walks"
    )


def test_chunk_hierarchical_default_hierarchy_builds_only_the_levels_the_budget_consults() -> None:
    """The lazy-level contract at the budget where it pays: at a
    paragraph-scale budget (2000 codepoints over 12 MiB of prose) every
    window is answered by the paragraph level, so the sentence and word
    walks -- the expensive lower levels of the default hierarchy -- must
    never run at all. The load-fair reference is ``["\\n\\n"]``, a custom
    hierarchy whose single literal supplies an equivalent top level
    WITHOUT the default hierarchy's lower levels: both sides pay the
    same paragraph scan plus the same windowing, so the default spelling
    may only add marginal spec-construction cost, never two more
    whole-text walks. Measured (min-of-3 after warmup): ~1.5ms default
    vs ~1.5ms literal (ratio ~1.0). The eager pre-#30 spelling measured
    ~351ms on the same corpus (the sentence walk ~188ms and the word
    walk ~132ms ran on a call whose budget never consulted them, ~200x
    this reference) -- the exact regression class the 3.0x ceiling is
    positioned to catch."""
    corpus = prose(12 * _MIB)
    default_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 2000), corpus)
    literal_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 2000, ["\n\n"]), corpus)
    assert default_ms < 3.0 * literal_ms, (
        f"chunk_hierarchical default @2000 12MiB took {default_ms:.1f}ms against "
        f"{literal_ms:.1f}ms for the paragraph-literal-only hierarchy "
        f"({default_ms / literal_ms:.1f}x); a budget the lower levels never "
        "answer is building them anyway"
    )


def test_chunk_hierarchical_duplicate_separator_entries_are_deduped_not_rebuilt() -> None:
    """The dedup contract at slot construction, load-fair ratios (both
    sides in the same process, min-of-3 after warmup): duplicate entries
    in a custom separator list are collapsed when the slot list is
    built -- a second ``None`` never re-splices the default hierarchy, a
    repeated literal never re-pays its scan -- so ``[None] * 8`` must
    cost one spliced hierarchy and ``[" "] * 8`` one literal level, not
    eight of either. The budget is DESCENDING (8 codepoints: windows
    inside long words exhaust the ``" "`` level down to the raw cut),
    which is where a lazy spelling without the dedup would still re-pay
    a duplicate -- the find_map only reaches one after its original
    returned None for that window, so a budget whose every window is
    answered by the first copy can never see the duplicates. Measured
    (min-of-3 after warmup; 6 MiB prose): ``[None]`` ~350ms vs ``[None] * 8`` ~355ms
    (ratio ~1.0), ``[" "]`` ~170ms vs ``[" "] * 8`` ~170ms (ratio
    1.0) with the dedup in the built extension -- the same ``[" "]``
    pair measures ~1.7x (~150ms vs ~258ms) without the dedup, the
    seven extra duplicate scans and cut vectors the descent re-paid.
    The eager spelling measured ~8x here outright (the ``[None]`` pair
    1347.1ms vs 169.2ms on the pre-dedup build), and the
    ``[" "] * 100`` spelling was a 1.69 GiB peak OOM shape
    (``[None] * 100`` its splice twin) without them. The
    structural teeth wall time cannot see -- that a duplicate after a
    CONSULTED level adds no BUILDS at all -- are the Rust seam pins
    (the build_seam LEVELS_BUILT counts in
    src/chunk_hierarchical_impl.rs)."""
    corpus = prose(6 * _MIB)
    lone_none_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 8, [None]), corpus)
    eight_none_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 8, [None] * 8), corpus)
    assert eight_none_ms < 1.5 * lone_none_ms, (
        f"chunk_hierarchical [None]*8 @8 6MiB took {eight_none_ms:.1f}ms against "
        f"{lone_none_ms:.1f}ms for [None] ({eight_none_ms / lone_none_ms:.1f}x); "
        "duplicate separator entries are being rebuilt per slot instead of deduped"
    )
    lone_space_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 8, [" "]), corpus)
    eight_space_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 8, [" "] * 8), corpus)
    assert eight_space_ms < 1.5 * lone_space_ms, (
        f"chunk_hierarchical [' ']*8 @8 6MiB took {eight_space_ms:.1f}ms against "
        f"{lone_space_ms:.1f}ms for [' '] ({eight_space_ms / lone_space_ms:.1f}x); "
        "duplicate separator entries are being rebuilt per slot instead of deduped"
    )


def test_chunk_hierarchical_none_splice_duplicates_are_inert_at_documented_budgets() -> None:
    """The ``[None] * 100`` dedup contract at the budget the docs disclose
    (the README's and api.md's "~0.5 ms at a 2000-codepoint budget over
    6 MiB of prose, the paragraph walk alone" parenthetical): at a
    paragraph-scale budget every window is answered by the FIRST paragraph
    slot, so the 99 duplicate splices are never consulted at all (the
    find_map dominance argument — the FIRST slot supplying a cut wins) and
    the ``*100`` spelling must cost exactly the lone spelling: one spliced
    hierarchy, one paragraph walk. Measured (min-of-3 after warmup, 6 MiB
    prose, the box this cell was written on — Linux, 32 logical cores,
    ambient load ~5): ~0.50 ms for BOTH spellings, ratio 1.00-1.01.

    That same dominance argument is the @2000 leg's limit, said plainly:
    a lost dedup is INVISIBLE at this budget, because the duplicate slots
    a lost dedup would leave in the list are never consulted — measured
    directly, with the slot-construction dedup disabled in a scratch
    build, the @2000 pair still measures ratio ~1.0 (green). The teeth
    therefore live in the second leg, the DESCENDING budget the sibling
    cell above established (@8, ``[None] * 8``): at 8 codepoints over
    1 MiB of prose, windows inside long words exhaust every spliced level
    down to the raw cut, so the find_map walks PAST every slot and each
    duplicate splice re-pays the three walks — with the dedup disabled,
    that scratch build measured the ``*100`` spelling at ~3.6 s against
    ~48 ms (ratio ~75x) at this exact leg, where the pristine tree
    measures ~48 ms for both spellings (ratio ~1.0). The ``*100`` scale
    (not the sibling's ``*8``) is the docs' own OOM shape: the pre-dedup
    eager spelling measured 17.2 s and +3,120 MiB of peak RSS at 6 MiB.
    The 1 MiB corpus (not the sibling's 6) keeps the leg at ~0.2 s of
    measurement while the disabled-dedup red side still lands at ~75x.

    Budgets: the @2000 leg takes the docs-claim ratio (1.3x, ~1.3x above
    the worst measured ratio) plus an absolute 2.5 ms ceiling (~5x above
    the measured 0.50 ms, CI-load headroom; the regression this leg IS
    positioned to catch — levels built EAGERLY again, every duplicate
    paying its walks up front, the 17.2 s class — blows through it by
    ~7000x). The descending leg takes the sibling cell's 1.5x margin.
    The @2000 wall is the paragraph walk, so a paragraph-scanner retune
    moves the ceiling's headroom, not the ratio legs (both sides pay the
    same walk)."""
    corpus = prose(6 * _MIB)
    lone_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 2000, [None]), corpus)
    hundred_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 2000, [None] * 100), corpus)
    assert hundred_ms < 1.3 * lone_ms, (
        f"chunk_hierarchical [None]*100 @2000 6MiB took {hundred_ms:.2f}ms against "
        f"{lone_ms:.2f}ms for [None] ({hundred_ms / lone_ms:.2f}x); duplicate None "
        "splices are costing more than one spliced hierarchy at a budget that "
        "never consults them"
    )
    assert hundred_ms < 2.5, (
        f"chunk_hierarchical [None]*100 @2000 6MiB took {hundred_ms:.2f}ms, over the "
        "absolute ceiling (measured ~0.50ms, ceiling 2.5ms with ~5x load headroom); "
        "duplicates are being built eagerly instead of deduped at slot construction"
    )
    small = prose(1 * _MIB)
    lone8_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 8, [None]), small)
    hundred8_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 8, [None] * 100), small)
    assert hundred8_ms < 1.5 * lone8_ms, (
        f"chunk_hierarchical [None]*100 @8 1MiB took {hundred8_ms:.1f}ms against "
        f"{lone8_ms:.1f}ms for [None] ({hundred8_ms / lone8_ms:.1f}x); duplicate "
        "None splices are being rebuilt per slot instead of deduped on the "
        "descending budget that consults them"
    )


def test_chunk_by_words_is_its_own_word_walk() -> None:
    """chunk_by_words' per-call cost contract: the word walk it is FOR,
    plus only marginal machinery (the grapheme index build, the merge, the
    streaming token filter). Measured ~165ms against ~132ms of word_count
    (ratio ~1.25) after the fix; the former spelling measured ~14.6x."""
    corpus = prose(12 * _MIB)
    tors_ms = _min_wall_ms(lambda s: chunk_by_words(s, 200), corpus)
    walk_ms = _min_wall_ms(tors.word_count, corpus)
    assert tors_ms < 2.5 * walk_ms, (
        f"chunk_by_words 12MiB took {tors_ms:.0f}ms against a {walk_ms:.0f}ms "
        f"word walk ({tors_ms / walk_ms:.1f}x); the merge/filter machinery is "
        "dominating the segmentation it windows"
    )


def test_chunk_by_sentences_is_its_own_sentence_walk() -> None:
    """chunk_by_sentences' cost contract, same shape as chunk_by_words':
    measured ~200ms against ~188ms of sentence_count (ratio ~1.07) after
    the fix; the former spelling measured ~7.1x."""
    corpus = prose(12 * _MIB)
    tors_ms = _min_wall_ms(lambda s: chunk_by_sentences(s, 10), corpus)
    walk_ms = _min_wall_ms(tors.sentence_count, corpus)
    assert tors_ms < 2.5 * walk_ms, (
        f"chunk_by_sentences 12MiB took {tors_ms:.0f}ms against a "
        f"{walk_ms:.0f}ms sentence walk ({tors_ms / walk_ms:.1f}x); the merge "
        "machinery is dominating the segmentation it windows"
    )


def test_chunk_by_paragraphs_absolute_band_holds() -> None:
    """``chunk_by_paragraphs`` has no stdlib comparator (a paragraph here
    is tors's own documented 2+-newline heuristic), so like
    ``grapheme_count`` its wall cell is an absolute regression ceiling
    with generous margin, not a race. Measured (min-of-3 after warmup,
    the helper's own sample count): ~0.5ms at 12 MiB of prose against
    the current byte-level scanner (the pre-fast-path per-char spelling
    measured ~10.4ms at the same corpus on the box that calibrated this
    cell, and 12.8ms on a CRLF-dense log there). The ceiling is 10ms
    (~20x margin over the current measurement), positioned so that a
    regression back to the per-char decode loop FAILS this cell: the old
    spelling's ~10.4ms blows straight through 10ms, where the former
    30ms ceiling would have absorbed it silently. This cell exists
    because the #28 codegen change moved this function's wall by +23-28%
    (8.55ms -> 10.96ms on the same corpus, measured at the criterion
    level) without any existing cell noticing -- the chunking-family
    section had no paragraph row at all; a future change of that class
    should at least have a ceiling to argue against."""
    corpus = prose(12 * _MIB)
    took_ms = _min_wall_ms(lambda s: chunk_by_paragraphs(s, 200), corpus)
    assert took_ms < 10.0, (
        f"chunk_by_paragraphs 12MiB took {took_ms:.1f}ms, outside the absolute "
        "band (measured ~0.5ms at 12 MiB, ceiling 10ms; the pre-fast-path "
        "per-char spelling measured ~10.4ms and must fail this "
        "cell); the paragraph scan regressed"
    )


def test_chunk_by_lines_absolute_band_holds() -> None:
    """``chunk_by_lines``' cell, same absolute-band shape as
    ``chunk_by_paragraphs``' (no stdlib comparator, one fused linear
    scan): measured ~0.6ms at 12 MiB of prose (min-of-3 after warmup,
    the helper's own sample count) against the current byte-level
    scanner; the pre-fast-path per-char spelling measured ~13.6ms at
    the same corpus on the box that calibrated this cell (a CRLF-dense
    log ran 12.8ms there). Ceiling 10ms (~16x margin over the current
    measurement), positioned so that a regression back to the per-char
    decode loop FAILS it: the old spelling's ~13.6ms blows straight
    through 10ms,
    where the former 40ms ceiling would have absorbed it silently. The
    scan and the real-line filter are one pass with O(1) memory beyond
    the output, so only a class regression (a per-line allocation, a
    lost fusion, a lost fast path) reaches the ceiling."""
    corpus = prose(12 * _MIB)
    took_ms = _min_wall_ms(lambda s: chunk_by_lines(s, 200), corpus)
    assert took_ms < 10.0, (
        f"chunk_by_lines 12MiB took {took_ms:.1f}ms, outside the absolute band "
        "(measured ~0.6ms at 12 MiB, ceiling 10ms; the pre-fast-path per-char "
        "spelling measured ~13.6ms and must fail this cell); the "
        "line scan regressed"
    )
