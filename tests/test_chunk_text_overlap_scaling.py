"""The chunk_text overlap path's bounded-work contract: the overlap-heavy
whitespace-run shape (a long blank-line run under a budget with
``overlap = max_chars - 1``) must stay within a linear factor of the
same-size zero-overlap partition.

The shape that went quadratic: on ``"\\n" * n`` every newline is its own
word segment, so the snap target ``chunk_end - overlap`` lands one
codepoint past the current window's start and ``start`` advances one
cluster per iteration, ~n windows. Each window's cut ends on a
trailing-whitespace run the trim then walked back cluster by cluster
(the ``trimmed_end`` walk in ``src/chunk_impl.rs``), O(the run behind the
window) per window: O(n^2) overall. The trim is now a precomputed
per-cluster prefix lookup (``grapheme_boundary_whitespace``'s
``last_non_ws_end``, one O(1) lookup per cut), so the whole call is the
segmentation walk plus O(n log n) window arithmetic, the same class the
zero-overlap partition already pays.

Measured through the public API (min-of-5 after warmup, Linux, CPython
3.12, otherwise-idle box), ``chunk_text("\\n" * n, n // 2, overlap=n // 2 - 1)``:

    n=20_000   51.0 ms  ->  3.0 ms   (~17x)
    n=40_000  180.8 ms  ->  6.1 ms   (~30x)
    n=80_000 709.6 ms  -> 12.5 ms   (~57x)
    n=160_000    n/a     -> 28.2 ms   (pre-fix projects ~2.8 s)

per-doubling wall ratios 3.55-3.92x before (quadratic), 2.03-2.26x after
(linear); the chunk spans are identical before and after (the same
``n - max_chars + 1`` windows), so this is purely a cost-shape fix.

The cells below are ratio-shaped on purpose (never absolute times): each
size's overlap wall is divided by that same size's zero-overlap wall
measured in the same process, so a shared-runner slowdown inflates both
sides together (the load-fair pattern test_performance.py's chunking
cells and test_grounded_performance.py's linear-shape cells use). Both
ratio cells time only multi-millisecond walls (the smallest is the 200k
zero-overlap base at ~5 ms): a sub-scheduler-slice wall lets a single
preemption set a side's minimum, the fragility test_performance.py's
own fast-cell note names. The quadratic relapse the gates guard against
measures ~843x at 200k and ~4x growth per quadruple (far outside the
gates), and was red-proofed against the pre-fix build through the same
public API (the before-numbers above).
"""

from __future__ import annotations

from time import monotonic

import pytest

import tors

# The pathological shape: a pure whitespace run, a half-text budget, and
# overlap one under it (the window advance collapses to one cluster per
# iteration, every window's trim sits on the whole run behind it).


def _min_wall_ms(fn, *, samples: int = 5, **kwargs) -> float:
    """Min-of-N wall milliseconds after one warmup call (extension init,
    allocator, cache): the fastest of several draws approximates the
    uncontended cost, the same measurement discipline the repo's other
    wall cells apply."""
    fn(**kwargs)
    best = float("inf")
    for _ in range(samples):
        started = monotonic()
        fn(**kwargs)
        best = min(best, monotonic() - started)
    return best * 1e3


def _run(n: int) -> list[tuple[int, int]]:
    m = n // 2
    return tors.chunk_text("\n" * n, m, overlap=m - 1)


def _assert_overlap_contract(chunks: list[tuple[int, int]], n: int) -> None:
    """The overlap contract on the whitespace-run shape, asserted in-cell
    so a cost regression can never be confused with a semantics change:
    the full window count, every window within budget, starts strictly
    advancing, consecutive windows genuinely sharing content."""
    m = n // 2
    assert len(chunks) == n - m + 1
    for start, end in chunks:
        assert 0 < end - start <= m
    for (prev_start, prev_end), (next_start, next_end) in zip(chunks, chunks[1:], strict=False):
        assert next_start > prev_start, "starts must strictly advance"
        assert next_end > prev_end, "ends must strictly advance"
        assert prev_start < next_start < prev_end, "no shared content"


@pytest.mark.timing
def test_high_overlap_stays_within_a_linear_factor_of_no_overlap() -> None:
    """The bounded-work cell: at a fixed size, the overlap-heavy call's
    wall must stay within a linear factor of the same-size zero-overlap
    partition measured in the same process. Both sides pay the same
    segmentation walk and grid build; the overlap path's own add-on (the
    per-window snap arithmetic) is constant per window post-fix, so the
    ratio sits ~4.5-5x and is flat across sizes. Measured, three
    consecutive cell runs on this box, across three builds of the
    in-flight tree (min of 5 per side, each): ratio 4.45-5.02x, overlap
    23-26 ms against base 4.6-7.4 ms, so the 8x gate sits ~1.6-1.8x
    above the measured band.
    The pre-fix trim walk made the
    overlap side quadratic (~843x at this size, red-proofed through this
    API): a superlinear add-on (the walk or anything like it) blows
    straight through the 8x gate."""
    n = 200_000
    m = n // 2
    text = "\n" * n
    overlap_chunks = _run(n)
    _assert_overlap_contract(overlap_chunks, n)
    overlap_ms = _min_wall_ms(tors.chunk_text, text=text, max_chars=m, overlap=m - 1)
    base_ms = _min_wall_ms(tors.chunk_text, text=text, max_chars=m, overlap=0)
    assert overlap_ms < 8.0 * base_ms, (
        f"overlap {overlap_ms:.1f}ms vs no-overlap {base_ms:.1f}ms "
        f"(ratio {overlap_ms / base_ms:.0f}x; the flat post-fix band is "
        "~4-5x): the overlap path grew superlinear in the text"
    )


@pytest.mark.timing
def test_high_overlap_whitespace_run_stays_linear_across_sizes() -> None:
    """The scaling curve, load-fair: normalize each size's overlap wall by
    its own same-process zero-overlap wall, then require the normalized
    factor to stay flat as the input quadruples. Post-fix the factor is
    ~4.45-5.1x at 200k and ~5.1-5.7x at 800k (growth 1.06-1.21x across
    three consecutive cell runs on this box, three builds of the
    in-flight tree, well under the 2.0x gate);
    the pre-fix trim walk grew the factor with the input (measured
    ~212x -> ~843x over 50k -> 200k, growth ~4x, red-proofed; the same
    quadratic, projected from the 709.6 ms pre-fix row at 80k in the
    table above, puts these sizes at ~820x -> ~3,400x, growth ~4x); any
    per-window cost that scales with the text behind the window trips
    this.

    The sizes are the measurement's own robustness knob: the smallest
    wall this cell times is the 200k zero-overlap base (~4.6-7.4 ms), a
    multi-scheduler-slice sample whose minimum-of-5 no single preemption
    can set. The 50k base it used to time was ~1.1 ms, sub-slice, and a
    deliberately starved run (nice 19 against 32 busy cores) reddened
    the cell through exactly that skew: a side whose every sample fits
    inside one scheduler burst keeps its clean floor while the multi-
    burst side pays quantized bursts, and the factor moves for scheduler
    reasons, not cost reasons."""
    small, large = 200_000, 800_000
    _assert_overlap_contract(_run(small), small)
    _assert_overlap_contract(_run(large), large)

    def factor(n: int) -> float:
        m = n // 2
        text = "\n" * n
        overlap_ms = _min_wall_ms(tors.chunk_text, text=text, max_chars=m, overlap=m - 1)
        base_ms = _min_wall_ms(tors.chunk_text, text=text, max_chars=m, overlap=0)
        return overlap_ms / base_ms

    small_factor, large_factor = factor(small), factor(large)
    assert large_factor < 2.0 * small_factor, (
        f"overlap-to-base factor grew {small_factor:.1f}x -> {large_factor:.1f}x "
        f"as the input quadrupled (flat post-fix at ~1.1x growth): the "
        "overlap path's per-window cost scales with the text"
    )


def test_whitespace_run_overlap_windows_are_exact_at_pin_size() -> None:
    """The exact-output anchor the cost fix must not move: at a size small
    enough to read whole, the overlap-heavy whitespace run is the full
    ladder of m+1 windows, (i, i + m) for every i, with the final window
    running to the text's end. The trim rewrite is output-identical to the
    walk-back it replaced (pinned exhaustively Rust-side, prefix against
    walk-back over every span); this anchors the same guarantee at the
    Python boundary, on the exact shape the scaling cells time."""
    n, m = 40, 20
    assert _run(n) == [(i, min(i + m, n)) for i in range(n - m + 1)]


def test_mixed_whitespace_and_content_overlap_windows_are_exact() -> None:
    """The trim's corners at the Python boundary: CRLF pairs (an
    all-whitespace cluster trim_end removes whole), a whitespace-only
    interior span (emitted untrimmed, the documented exception), word and
    sentence cuts, exact outputs the O(1) trim must reproduce unchanged
    from the walk-back era."""
    cases = [
        ("a\r\nb. c\r d\ne", 3, 0, "word", [(0, 1), (1, 4), (4, 7), (7, 10), (10, 12)]),
        ("a\r\nb. c\r d\ne", 3, 1, "word", [(0, 1), (1, 4), (3, 5), (4, 7), (7, 10), (9, 12)]),
        ("a\r\nb. c\r d\ne", 5, 1, "word", [(0, 5), (4, 7), (6, 10), (9, 12)]),
        ("a\r\nb. c\r d\ne", 5, 4, "sentence", [(0, 1), (1, 3), (3, 7), (7, 12)]),
        ("aaaa    bbbb", 3, 2, "sentence", [(0, 3), (3, 4), (4, 7), (7, 10), (10, 12)]),
        ("aaaa    bbbb", 5, 0, "word", [(0, 4), (4, 8), (8, 12)]),
        ("aaaa    bbbb", 8, 7, "word", [(0, 4), (4, 12)]),
        (
            "the cat sat on the mat today",
            8,
            1,
            "word",
            [(0, 7), (4, 11), (8, 14), (12, 18), (15, 22), (22, 28)],
        ),
        (
            "the cat sat on the mat today",
            5,
            0,
            "sentence",
            [(0, 5), (5, 10), (10, 14), (14, 18), (18, 22), (22, 27), (27, 28)],
        ),
    ]
    for text, max_chars, overlap, boundary, expected in cases:
        got = tors.chunk_text(text, max_chars, overlap=overlap, boundary=boundary)
        assert got == expected, f"{text!r} m={max_chars} ov={overlap} {boundary}"
