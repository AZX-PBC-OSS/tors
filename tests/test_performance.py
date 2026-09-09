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
from tors import chunk_by_sentences, chunk_by_words, chunk_hierarchical

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
    corpus = {"prose": prose, "decomposed": decomposed, "crlf": crlf}[corpus_kind](
        size_bytes
    )
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
    print(f"b64_decode {corpus_kind} {size_bytes // _MIB}MiB: tors {tors_ms:.2f}ms "
          f"stdlib {std_ms:.2f}ms ratio {tors_ms / std_ms:.2f}")


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
    print(f"html_unescape no-& prose 12MiB: tors {tors_ms:.2f}ms "
          f"stdlib {std_ms:.2f}ms ratio {tors_ms / std_ms:.2f}")


@pytest.mark.parametrize("corpus_kind", ["prose", "decomposed"])
@pytest.mark.parametrize("size_bytes", [1 * _MIB, 12 * _MIB], ids=["1MiB", "12MiB"])
def test_grapheme_count_absolute_band_holds(
    corpus_kind: str, size_bytes: int
) -> None:
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
# The chunking family's document-scale cost shape (#22): the per-call cost
# must be the segmentation walks the function is FOR, not per-codepoint
# structures built unconditionally. Load-fair ratios (both sides measured
# in the same process, min-of-3 after warmup), so a shared-runner slowdown
# inflates both sides together.
#
# Measured on the dev box (Linux, CPython 3.12, min-of-3 after warmup),
# before the fix -> after:
#
#     chunk_hierarchical, 12 MiB degenerate single-char run, custom
#     never-matching separators, whole-document budget: ~1048ms -> ~2.6ms
#     (the unconditional grapheme HashSet + Vec<char> collect; ~400x)
#
#     chunk_hierarchical, 12 MiB prose, default hierarchy, 2000-char
#     budget: ~2996ms -> ~351ms; the residual is the word walk (~132ms)
#     plus the sentence walk (~188ms) -- the accurate UAX #29 hierarchy
#     the function exists to provide
#
#     chunk_by_words, 12 MiB prose, 200 words/chunk: ~1923ms -> ~165ms
#     chunk_by_sentences, 12 MiB prose, 10 sentences/chunk:
#     ~1328ms -> ~200ms
# ---------------------------------------------------------------------------


def test_chunk_hierarchical_custom_no_match_is_scan_cost_not_per_char_structures() -> None:
    """The whole-document-budget custom-hierarchy cell from #22: a
    never-matching separator list must cost one count pass plus one scan
    pass (ratioed against CPython's own ``in``, a C-speed scan of the same
    text), not the unconditional per-codepoint grapheme structure the
    former spelling built before anything else could run. Measured ratio
    ~2 after the fix (the count pass plus marshalling rides on top of the
    scan); the former spelling measured ~700x."""
    q = "q" * (12 * _MIB)
    tors_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 12 * _MIB, ["xyz"]), q)
    scan_ms = _min_wall_ms(lambda s: "xyz" in s, q)
    assert tors_ms < 8.0 * scan_ms, (
        f"chunk_hierarchical no-match 12MiB took {tors_ms:.1f}ms against an "
        f"{scan_ms:.1f}ms bare scan ({tors_ms / scan_ms:.0f}x); the lazily-built "
        "grapheme machinery regressed to an unconditional structure"
    )


def test_chunk_hierarchical_custom_no_match_skips_the_grapheme_walk_on_non_ascii() -> None:
    """The laziness contract, on the input where it is load-bearing: for
    NON-ASCII text the grapheme index is a full segmentation walk
    (~150ms at 12 MiB), so a call that never needs it (never-matching
    separators, whole-document budget: no cuts to filter, no raw-cut
    window, no overlap snap) must not build it. The reference is the same
    ``in`` scan; measured ~5ms vs ~2ms on decomposed prose (ratio ~2.5)
    when lazy, ~75x if the index were built unconditionally — the ASCII
    fast path masks that regression on single-byte corpora, which is why
    this cell runs decomposed text."""
    corpus = decomposed(12 * _MIB)
    tors_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, len(s), ["xyz"]), corpus)
    scan_ms = _min_wall_ms(lambda s: "xyz" in s, corpus)
    assert tors_ms < 8.0 * scan_ms, (
        f"chunk_hierarchical no-match non-ASCII 12MiB took {tors_ms:.1f}ms against "
        f"an {scan_ms:.1f}ms bare scan ({tors_ms / scan_ms:.0f}x); the grapheme "
        "index is being built on a call that cannot use it"
    )


def test_chunk_hierarchical_default_hierarchy_is_its_own_segmentation_walks() -> None:
    """The default hierarchy's per-call cost contract: no more than its
    own UAX #29 walks (word_count + sentence_count on the same corpus,
    measured in-process). The hierarchy IS those walks; everything around
    them -- level construction, the cut filter, the chunk loop,
    marshalling -- must be marginal. Measured ~351ms against a ~320ms
    reference sum (ratio ~1.1) after the fix; the former spelling measured
    ~9.4x (the grapheme hash set dominated the segmentation it was
    filtering)."""
    corpus = prose(12 * _MIB)
    tors_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 2000), corpus)
    walks_ms = _min_wall_ms(tors.word_count, corpus) + _min_wall_ms(
        tors.sentence_count, corpus
    )
    assert tors_ms < 2.0 * walks_ms, (
        f"chunk_hierarchical default 12MiB took {tors_ms:.0f}ms against "
        f"{walks_ms:.0f}ms of its own segmentation walks "
        f"({tors_ms / walks_ms:.1f}x); per-call machinery is dominating the walks"
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
