"""Adversarial attack suite for ``tors.chunk_to_budget`` /
``tors.chunk_to_offsets``.

Every attack is a test; passing tests are regression pins that document
robustness.

Attack classes covered (on top of the contract's seven):

- BUDGET INVARIANT under NON-ADDITIVE counters: BPE-style merging that
  measures a merged span LESS than its parts; a counter that measures any
  span containing a space as len+2 (counter(whole) > counter(a)+counter(b));
  whitespace-run-hostile counters; a counter that returns exactly
  max_tokens for the oversized single word (can a second word smuggle in?).
- CALLBACK ADVERSARIAL: side-effecting/stateful counters, counters that
  raise mid-pack (traceback must surface intact), non-int returns
  (bool/float/None/huge), counters that call tors functions RE-ENTRANTLY
  (detach-within-attach), counters that release the GIL themselves
  (time.sleep), and the "one call per CANDIDATE CHUNK, never per boundary"
  call-count claim (docs/api.md) instrumented.
- OVERLAP SEMANTICS: a 0-or-1 counter at overlap == max_tokens-1 (the
  no-forward-progress stress shape), overlap larger than a whole chunk,
  float ratios near 1.0 vs the equivalent int (the floor(ratio*budget)
  int/float path equivalence), chunks strictly increasing and pairwise
  distinct.
- GIL CLAIM AUDIT: hostile heartbeat cells: a counter that releases the
  GIL itself, a re-entrant counter (tors call inside the callback), and
  the extraction band under 1-span-per-codepoint offsets.
- OFFSET/ROUND-TRIP torture: CJK, emoji ZWJ, combining marks, RTL with
  bidi marks, whitespace-only, text of exactly max_tokens tokens, empty
  text / empty offsets consistency, degenerate span shapes.
"""

from __future__ import annotations

import asyncio
import itertools
import sys
import time
import traceback

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors

# ---- helpers -----------------------------------------------------------------


def word_counter(text: str) -> int:
    return len(text.split())


def word_offsets(text: str) -> list[tuple[int, int]]:
    return [(s, e) for s, e in tors.word_bounds(text) if text[s:e].strip()]


def assert_full_contract(text, chunks, budget, counter, spans=None):
    """The whole packing contract, re-derived independently of the impl:
    bounds, strict advancement, coverage, and the per-chunk budget
    invariant (the single-oversized-segment exception is the only out)."""
    if not text:
        assert chunks == []
        return
    assert chunks, "non-empty text must yield chunks"
    prev_s, prev_e = -1, 0
    for i, (s, e) in enumerate(chunks):
        assert 0 <= s < e <= len(text), f"bounds: {chunks}"
        if i == 0:
            assert s == 0
        else:
            assert s > prev_s, f"starts must strictly advance: {chunks}"
            assert e > prev_e, f"ends must strictly advance: {chunks}"
        if spans is not None:
            measured = sum(1 for a, b in spans if a >= s and b <= e)
        else:
            measured = counter(text[s:e])
        if measured > budget:
            interior = len(tors.word_bounds(text[s:e]))
            assert interior <= 1, (
                f"budget exceeded without the single-segment exception: "
                f"{chunks} {text!r} measured={measured} budget={budget}"
            )
        prev_s, prev_e = s, e
    assert prev_e == len(text), "must cover to the end"


# ---- 1. budget invariant under non-additive counters -------------------------


COUNTER_BATTERY = {
    # BPE-style merging: a merged span measures LESS than its parts would
    # sum to; sum-based packers would under-fill; this packer must still
    # never emit a chunk that over-measures.
    "bpe_merge_less": lambda s: max(1, len(s.split()) // 2) if " " in s else len(s),
    # counter(whole) > counter(a) + counter(b): +2 for any span with a space.
    "space_penalty": lambda s: len(s) + 2 if " " in s else len(s),
    # whitespace runs measure huge.
    "ws_huge": lambda s: len(s) + 100 * s.count(" "),
    # non-monotone both directions (first/last codepoint keyed).
    "flare_first": lambda s: (len(s) * 7 % 11) + 1,
    "flare_last": lambda s: (ord(s[-1]) % 5) + 1,
    # everything measures 1 (the all-max liveness shape at budget=1).
    "one": lambda s: 1,
}

TEXTS = [
    "", " ", "   ", ".", "a", "aa bb", "aa. bb. cc.", "a b c d e f g h i j",
    "One. Two. Three. Four.", "  lead", "trail  ", "a    b", "東京。大阪。京都。",
    "a\u0301 b\u0301. \U0001F469\u200D\U0001F52C x.", "x" * 100, "ab " * 40,
    "no terminator sentence that just runs on and on with many words",
]


@pytest.mark.parametrize("counter_name", sorted(COUNTER_BATTERY))
def test_budget_invariant_survives_non_additive_counters(counter_name: str) -> None:
    """ATTACK: every counter in the battery is non-additive or non-monotone
    in a way that breaks sum-based packers. If the packer ever sums instead
    of measuring the candidate chunk, some (text, budget, overlap) cell
    emits a chunk that over-measures without the single-segment excuse."""
    counter = COUNTER_BATTERY[counter_name]
    for text in TEXTS:
        for budget in (1, 2, 3, 5, 9):
            for overlap in range(budget):
                try:
                    chunks = tors.chunk_to_budget(
                        text, counter, max_tokens=budget, overlap=overlap
                    )
                except ValueError as err:
                    # only the documented zero-sentence contract may raise
                    assert "returned 0" in str(err), (counter_name, text, str(err))
                    continue
                assert_full_contract(text, chunks, budget, counter)


def test_oversized_word_at_exactly_max_tokens_smuggles_no_second_word() -> None:
    """ATTACK: a counter that returns max_tokens EXACTLY for the oversized
    single word (and for anything bigger). The documented exception lets
    ONE segment out whole; the packer must not extend past it (extension
    requires first_count <= max_tokens, and the candidate measurement,
    not a sum, gates every accepted append)."""
    text = "aaaaa bbbbb ccccc"

    def exact(s: str) -> int:
        # the whole span always measures exactly the budget: an extension
        # would only be possible if the packer skipped measurement
        return 10

    chunks = tors.chunk_to_budget(text, exact, max_tokens=10)
    # whole text measures 10 <= budget: one chunk, nothing oversized.
    assert chunks == [(0, len(text))]

    # now the true oversized shape: the first word alone measures 11
    def first_word_huge(s: str) -> int:
        if "aaaaa" in s and "bbbbb" in s:
            return 100  # any span containing a second word over-measures
        if "aaaaa" in s:
            return 11  # the oversized word itself, above the budget
        return 1

    chunks = tors.chunk_to_budget(text, first_word_huge, max_tokens=10)
    # the oversized word must go out ALONE (no second word smuggled in):
    assert text[chunks[0][0] : chunks[0][1]] == "aaaaa"
    for i, (s, e) in enumerate(chunks):
        assert 0 <= s < e <= len(text)
        if i:
            assert s > chunks[i - 1][0] and e > chunks[i - 1][1]
    assert chunks[-1][1] == len(text)


# ---- 2. overlap semantics -----------------------------------------------------


def test_zero_or_one_counter_at_overlap_max_minus_one_terminates() -> None:
    """ATTACK: a counter that measures everything 0-or-1 with
    overlap == max_tokens-1; the walk-back can never certify the
    requested overlap, so every transition must decline to zero overlap
    and still make unconditional progress."""
    text = "aa. bb. cc. dd. ee. ff. gg. hh."
    for budget in (1, 2, 3, 5):
        for overlap in range(budget):

            def zero_one(s: str) -> int:
                return 1 if s.strip() else 0

            chunks = tors.chunk_to_budget(
                text, zero_one, max_tokens=budget, overlap=overlap
            )
            starts = [s for s, _ in chunks]
            assert len(starts) == len(set(starts)), "duplicate starts"
            assert len(chunks) <= len(text), "chunk count bounded by codepoints"
            assert_full_contract(text, chunks, budget, zero_one)


def test_overlap_larger_than_a_whole_chunk_never_duplicates() -> None:
    """ATTACK: overlap requests bigger than any single chunk can satisfy;
    the next chunk must never be empty, never equal its predecessor."""
    text = "a. b. c."
    for budget in (1, 2):
        for overlap in range(budget):
            chunks = tors.chunk_to_budget(text, word_counter, max_tokens=budget, overlap=overlap)
            assert all(text[s:e] for s, e in chunks)
            pairwise = list(zip(chunks, chunks[1:], strict=False))
            for prev, nxt in pairwise:
                assert prev != nxt, "identical consecutive chunks"
                assert nxt[0] > prev[0] and nxt[1] > prev[1]


def test_float_ratio_near_one_matches_the_equivalent_int_overlap() -> None:
    """ATTACK: ratios whose floor(ratio*max_tokens) sits at the int/float
    path boundary; the two spellings must resolve identically."""
    import math

    text = "the cat sat on the mat today and the dog ran far away"
    for budget in (1, 2, 4, 7, 13):
        for ratio in (0.9999999, 0.999, 0.5, 0.001):
            tokens = math.floor(ratio * budget)
            if tokens >= budget:
                continue
            a = tors.chunk_to_budget(text, word_counter, max_tokens=budget, overlap=ratio)
            b = tors.chunk_to_budget(text, word_counter, max_tokens=budget, overlap=tokens)
            assert a == b, (budget, ratio, a, b)


# ---- 3. callback adversarial --------------------------------------------------


def test_counter_raising_on_the_third_call_surfaces_intact() -> None:
    """ATTACK: an exception raised mid-pack (3rd counter call) must
    propagate with its original type, message, and traceback frame: no
    Rust panic, no 'original exception was lost' substitute, no poisoned
    state (a follow-up call still works)."""
    calls = {"n": 0}

    class Boom(Exception):
        pass

    def counter(s: str) -> int:
        calls["n"] += 1
        if calls["n"] == 3:
            raise Boom("third call dies")
        return max(len(s.split()), 1)

    with pytest.raises(Boom, match="third call dies") as info:
        tors.chunk_to_budget("One. Two. Three. Four. Five. Six.", counter, max_tokens=2)
    tb = traceback.format_exception(info.type, info.value, info.value.__traceback__)
    assert any("counter" in frame for frame in tb), "traceback lost the raising frame"
    assert calls["n"] == 3
    # no poisoned state: the next call with a healthy counter is normal
    chunks = tors.chunk_to_budget("a b c", word_counter, max_tokens=2)
    assert_full_contract("a b c", chunks, 2, word_counter)


def test_stateful_side_effecting_counter_terminates_deterministically() -> None:
    """ATTACK: a counter with side effects (mutates a captured list) whose
    answers depend on the call index. Progress is anchored to offsets, so
    this terminates; starts must never repeat."""
    calls: list[int] = []

    def stateful(s: str) -> int:
        calls.append(len(s))
        return 1 if len(calls) % 3 else 500

    text = "aa bb. cc dd. ee ff. gg hh."
    seen = []
    for _ in range(50):
        calls.clear()
        chunks = tors.chunk_to_budget(text, stateful, max_tokens=4, overlap=2)
        starts = [s for s, _ in chunks]
        assert len(starts) == len(set(starts))
        seen.append(tuple(chunks))
    assert len(set(seen)) == 1, "same call sequence must give the same chunks"


def test_counter_return_shapes_are_gated() -> None:
    """ATTACK: bool (int subclass) is accepted; float/None/str/list are
    TypeErrors; 2**70 is a ValueError; negatives are ValueErrors."""
    text = "One. Two. Three."
    assert tors.chunk_to_budget(text, lambda _: True, max_tokens=3) == [(0, 16)]
    for bad in (12.7, None, "3", [2], 2.0):
        with pytest.raises(TypeError, match="must return an int"):
            tors.chunk_to_budget(text, lambda _, b=bad: b, max_tokens=3)  # noqa: B023
    with pytest.raises(ValueError, match="unreasonably large"):
        tors.chunk_to_budget(text, lambda _: 2**70, max_tokens=3)
    with pytest.raises(ValueError, match="negative"):
        tors.chunk_to_budget(text, lambda _: -(2**40), max_tokens=3)


def test_counter_may_call_tors_functions_reentrantly() -> None:
    """ATTACK (GIL stress): a counter that calls tors.chunk_to_offsets /
    tors.chunk_to_budget INSIDE the callback (detach-within-attach). No
    deadlock, no panic, and the outer packing still satisfies its contract."""
    text = "One. Two. Three. Four. Five. Six. Seven. Eight."

    def reentrant(s: str) -> int:
        spans = [(i, i + 1) for i in range(len(s))]
        tors.chunk_to_offsets(s, spans, max_tokens=3)
        tors.chunk_to_budget(s, lambda t: 1, max_tokens=2)
        return max(len(s.split()), 1)

    chunks = tors.chunk_to_budget(text, reentrant, max_tokens=2, overlap=1)
    assert_full_contract(text, chunks, 2, word_counter)


def test_counter_may_release_the_gil_itself() -> None:
    """ATTACK: a counter that calls time.sleep (releases the GIL from
    inside the native caller's attach window). The call must complete
    correctly; the heartbeat cell below checks the loop's accounting."""

    def sleepy(s: str) -> int:
        time.sleep(0.001)
        return max(len(s.split()), 1)

    chunks = tors.chunk_to_budget(
        "One. Two. Three. Four. Five. Six.", sleepy, max_tokens=2, overlap=1
    )
    assert_full_contract("One. Two. Three. Four. Five. Six.", chunks, 2, word_counter)


def test_call_count_is_per_candidate_span_not_per_boundary() -> None:
    """ATTACK on the docs claim ('one candidate chunk's text per packing
    decision, never once per boundary'): instrument the counter and pin
    the call count to O(segments). A per-boundary counter on this text
    would see ~2 calls per word; the candidate-span shape is a small
    multiple of the chunk count. Also: every measured span is at most one
    chunk's worth of text, and no span is ever empty."""
    text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten."
    spans_seen: list[str] = []

    def counting(s: str) -> int:
        spans_seen.append(s)
        return max(len(s.split()), 1)

    chunks = tors.chunk_to_budget(text, counting, max_tokens=2, overlap=1)
    assert len(chunks) >= 5
    # 10 sentences + extensions + walk-backs: a small multiple of chunks,
    # far below a per-boundary counter's ~2-per-word (text has 20 words).
    assert len(spans_seen) <= 4 * len(chunks), (len(spans_seen), len(chunks))
    # every measured span is real text (never empty, always a text slice)
    for span in spans_seen:
        assert span
        assert span in text
    # and no measured span exceeds one chunk's worth plus one segment (a
    # REJECTED extension candidate is the chunk-to-be plus one segment;
    # the doc's "at most one chunk's worth" is honored to that tolerance)
    longest_chunk = max(e - s for s, e in chunks)
    longest_sentence = max(e - s for s, e in tors.sentence_bounds(text))
    assert max(len(s) for s in spans_seen) <= longest_chunk + longest_sentence


# ---- 4. offset / round-trip torture -------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "こんにちは世界。東京タワー。",  # CJK
        # RTL scripts (Hebrew, Arabic):
        "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd. \u0645\u0631\u062d\u0628\u0627.",
        "peace \u05e9\u05dc\u05d5\u05dd\u200f ok\u200e end.",  # bidi marks
        "\U0001F469\u200D\U0001F52C\U0001F468\u200D\U0001F4BB x. y.",  # ZWJ sequences
        "a\u0301b\u0327\u0301. n\u0303.",  # stacking combining marks
        "cafe\u0301 strasse\u0300. done.",
    ],
)
def test_round_trip_through_hostile_scripts(text: str) -> None:
    """text[start:end] must be exactly the codepoints between the offsets
    through CJK, RTL + bidi marks, ZWJ emoji, and combining marks, for
    both spellings, over budgets and overlaps."""
    for budget in (1, 2, 3, 5):
        for overlap in range(budget):
            for chunks in (
                tors.chunk_to_budget(text, word_counter, max_tokens=budget, overlap=overlap),
                tors.chunk_to_offsets(
                    text, word_offsets(text), max_tokens=budget, overlap=overlap
                ),
            ):
                for s, e in chunks:
                    assert text[s:e], f"empty slice at {(s, e)} in {chunks}"
                assert_full_contract(text, chunks, budget, word_counter)


def test_text_of_exactly_max_tokens_is_one_chunk() -> None:
    text = "one two three"
    assert tors.chunk_to_budget(text, word_counter, max_tokens=3) == [(0, 13)]
    assert tors.chunk_to_offsets(text, word_offsets(text), max_tokens=3) == [(0, 13)]


def test_empty_offsets_on_non_empty_text_is_covering_chunks() -> None:
    """Pinned either way: empty offsets
    on non-empty text measure 0 everywhere (tolerated, gaps allowed), and
    the result is a covering chunk set anchored to offsets, not counts.
    Empty text with empty offsets is []."""
    assert tors.chunk_to_offsets("hello world", [], max_tokens=5) == [(0, 11)]
    assert tors.chunk_to_offsets("   ", [], max_tokens=5) == [(0, 3)]
    assert tors.chunk_to_offsets("", [], max_tokens=5) == []
    # the callback spelling's twin shape (a counter measuring 0 for the
    # whole first sentence) is the documented ValueError; the two
    # spellings' zero-measure contracts differ, and both are pinned:
    with pytest.raises(ValueError, match="returned 0"):
        tors.chunk_to_budget("   ", word_counter, max_tokens=5)


def test_offsets_degenerate_span_shapes_are_rejected_or_handled() -> None:
    """Panic hunting at the offsets boundary: zero-width, unsorted,
    overlapping, out-of-bounds, 3-tuples, non-pair items."""
    text = "abcd ef"
    with pytest.raises(ValueError):
        tors.chunk_to_offsets(text, [(1, 1)], max_tokens=2)  # zero-width
    with pytest.raises(ValueError):
        tors.chunk_to_offsets(text, [(0, 4), (3, 5)], max_tokens=2)  # overlapping
    with pytest.raises(ValueError):
        tors.chunk_to_offsets(text, [(3, 5), (0, 4)], max_tokens=2)  # unsorted
    with pytest.raises(ValueError):
        tors.chunk_to_offsets(text, [(0, 99)], max_tokens=2)  # out of bounds
    with pytest.raises(ValueError):
        tors.chunk_to_offsets(text, [(0, 4, 9)], max_tokens=2)  # 3-tuple
    # degenerate-but-valid: one span covering the whole text at budget 1
    assert tors.chunk_to_offsets(text, [(0, 7)], max_tokens=1) == [(0, 7)]
    # adjacent singleton spans on a word-separated text: each token is its
    # own chunk (whitespace rides each chunk's tail)
    spans = [(0, 1), (2, 3), (4, 5)]
    chunks = tors.chunk_to_offsets("a b c", spans, max_tokens=1)
    assert_full_contract("a b c", chunks, 1, word_counter, spans=spans)
    assert chunks == [(0, 2), (2, 4), (4, 5)]


def test_huge_int_arguments_fail_clean_not_panicky() -> None:
    """A Python int too large for i64 must raise a clean Python error
    (OverflowError from extraction), never a Rust panic."""
    with pytest.raises(OverflowError):
        tors.chunk_to_budget("a b", word_counter, max_tokens=10**30)
    with pytest.raises(OverflowError):
        tors.chunk_to_budget("a b", word_counter, max_tokens=2, overlap=10**30)


# ---- 5. Hypothesis sweeps with adversarial counters ---------------------------


@settings(max_examples=300, deadline=None)
@given(
    text=st.text(max_size=100),
    budget=st.integers(1, 12),
    flare=st.integers(2, 60),
)
def test_flare_counter_never_breaks_the_contract(text: str, budget: int, flare: int) -> None:
    """The adversarial shape that breaks sum-based or binary-search
    packers: counts keyed to the span's FIRST codepoint, unrelated to
    length. Progress must stay unconditional and the contract intact."""
    def flare_counter(s: str) -> int:
        if not s:
            return 1
        return flare if ord(s[0]) % 3 == 0 else 1

    chunks = tors.chunk_to_budget(text, flare_counter, max_tokens=budget, overlap=0)
    assert len(chunks) <= max(len(text), 1)
    for s, e in chunks:
        assert 0 <= s < e <= len(text)


@settings(max_examples=200, deadline=None)
@given(
    text=st.text(max_size=80),
    budget=st.integers(1, 8),
)
def test_space_penalty_counter_never_over_emits(text: str, budget: int) -> None:
    """counter(whole) > counter(a) + counter(b): the superadditive shape.
    Emitted chunks must still fit the budget per the same counter (or be
    the single-segment exception)."""
    counter = COUNTER_BATTERY["space_penalty"]
    chunks = tors.chunk_to_budget(text, counter, max_tokens=budget, overlap=0)
    assert_full_contract(text, chunks, budget, counter)


@settings(max_examples=200, deadline=None)
@given(
    text=st.text(max_size=80),
    budget=st.integers(1, 8),
    span_bits=st.data(),
)
def test_generated_offsets_never_break_the_offsets_contract(
    text: str, budget: int, span_bits
) -> None:
    """ GENERATED token offsets (not just text): random sorted
    non-overlapping spans with gaps, driving the offsets packing through
    the same contract: the fuzz target's shape, at the Python layer."""
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        if span_bits.draw(st.booleans()) and not text[i].isspace():
            j = min(len(text), i + span_bits.draw(st.integers(1, 4)))
            if not any(text[k].isspace() for k in range(i, j)):
                spans.append((i, j))
                i = j
                continue
        i += 1
    chunks = tors.chunk_to_offsets(text, spans, max_tokens=budget, overlap=0)

    def contained(s: int, e: int) -> int:
        return sum(1 for a, b in spans if a >= s and b <= e)

    if not text:
        assert chunks == []
        return
    prev_s, prev_e = -1, 0
    for k, (s, e) in enumerate(chunks):
        assert 0 <= s < e <= len(text)
        if k == 0:
            assert s == 0
        else:
            assert s > prev_s and e > prev_e
        if contained(s, e) > budget:
            assert len(tors.word_bounds(text[s:e])) <= 1
        prev_s, prev_e = s, e
    assert prev_e == len(text)


# ---- 6. GIL claim audit (hostile cells, modest sizes for suite speed) ---------


async def _gap_and_wall(op):
    ticks: list[float] = []
    stop = asyncio.Event()

    async def heartbeat():
        while True:
            ticks.append(time.monotonic())
            if stop.is_set():
                return
            await asyncio.sleep(0.01)

    hb = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    started = time.monotonic()
    try:
        await op()
    finally:
        stop.set()
        await hb
    wall = time.monotonic() - started
    worst = max((b - a for a, b in itertools.pairwise(ticks)), default=0.0)
    return worst, wall


def test_gil_heartbeat_survives_a_counter_that_releases_the_gil() -> None:
    """A counter that itself releases the GIL (time.sleep per call): the
    worker's handoffs must still interleave; the worst heartbeat gap
    stays a small fraction of the wall, never the whole call."""

    def sleepy(s: str) -> int:
        time.sleep(0.002)
        return max(len(s.split()), 1)

    text = "One. " * 200
    worst, wall = asyncio.run(
        _gap_and_wall(lambda: asyncio.to_thread(tors.chunk_to_budget, text, sleepy, max_tokens=2))
    )
    assert wall > 0.2  # the sleeps dominate: the call is genuinely long
    assert worst < 0.10, f"loop blocked {worst * 1000:.0f}ms of {wall * 1000:.0f}ms"


def test_gil_heartbeat_survives_reentrant_tors_calls_in_the_counter() -> None:
    """The hostile re-entrancy cell: the counter calls tors.chunk_to_offsets
    (its own py.detach) from inside the packer's attach window. No
    deadlock; the loop stays schedulable between callbacks."""
    unit = "speaker: message with a few words and a number 42.\n"
    corpus = unit * (200_000 // len(unit))

    def reentrant(s: str) -> int:
        spans = [(i, i + 1) for i in range(min(len(s), 5_000))]
        tors.chunk_to_offsets(s, spans, max_tokens=50)
        return max(len(s.split()), 1)

    worst, wall = asyncio.run(
        _gap_and_wall(
            lambda: asyncio.to_thread(tors.chunk_to_budget, corpus, reentrant, max_tokens=100)
        )
    )
    assert worst < 0.30 * wall or worst < 0.100, (
        f"loop blocked {worst * 1000:.0f}ms of {wall * 1000:.0f}ms"
    )


def test_gil_offsets_extraction_band_at_one_span_per_codepoint() -> None:
    """The extraction-band attack: one span per codepoint (~1.5M spans on
    a 2MB corpus): the O(tokens) GIL-held argument walk. The documented
    bespoke budget (0.60 ratio, the measured-band ratio-budget shape) plus
    the 100ms ceiling must hold."""
    unit = "speaker: message with a few words and a number 42.\n"
    corpus = unit * (2 * 1024 * 1024 // len(unit))
    spans = [(i, i + 1) for i in range(len(corpus)) if not corpus[i].isspace()]
    worst, wall = asyncio.run(
        _gap_and_wall(
            lambda: asyncio.to_thread(tors.chunk_to_offsets, corpus, spans, max_tokens=200)
        )
    )
    assert worst < 0.60 * wall or worst < 0.100, (
        f"loop blocked {worst * 1000:.0f}ms of {wall * 1000:.0f}ms "
        f"(ratio {worst / wall:.2f})"
    )


# ---- 7. aio twins --------------------------------------------------------------


def test_aio_twins_match_their_sync_spellings() -> None:
    """The aio twins exist and return exactly what the sync spellings
    return (test_aio.py's await-correctness sweep covers them too; this
    cell pins the values directly)."""
    import tors.aio

    async def run():
        text = "One. Two. Three. Four. Five."
        a = await tors.aio.chunk_to_budget(text, word_counter, max_tokens=2, overlap=1)
        b = await tors.aio.chunk_to_offsets(text, word_offsets(text), max_tokens=4, overlap=2)
        return a, b

    a, b = asyncio.run(run())
    assert a == tors.chunk_to_budget(
        "One. Two. Three. Four. Five.", word_counter, max_tokens=2, overlap=1
    )
    assert b == tors.chunk_to_offsets(
        "One. Two. Three. Four. Five.", word_offsets("One. Two. Three. Four. Five."),
        max_tokens=4, overlap=2,
    )


# ---- 9. the slow-counter heartbeat claim ---------------------------------------
#
# test_gil_release.py::test_chunk_to_budget_slow_counter_blocks_only_for_
# its_callbacks pins "the worst gap tracks the callbacks themselves, never
# the whole call" with a busy counter time-budgeted at ~3x the GIL switch
# interval per call, so every callback stays above sys.getswitchinterval()
# (0.005s) on any interpreter (3.10-3.14). The mechanism, measured: with a
# callback-dominated wall on a small text
# (microsecond detach windows), a callback shorter than the switch interval
# starves the event loop for the WHOLE call: the worker drops and
# re-acquires the GIL faster than the woken loop thread can take it, and
# gil_drop_request (which forces the fair handoff) only fires for
# callbacks that straddle the interval. Measured here: ~1ms callbacks ->
# 100% blocked, ~4ms -> 79-100%, ~8ms+ -> 3-6% (clean). The Rust code's
# GIL claim itself holds (detach-between-callbacks is real; the 8/16ms
# cells and the fast-counter cell all pass): the loop is schedulable
# BETWEEN callbacks for callback-dominated calls when each callback
# exceeds the switch interval or the native windows between them are
# substantial, and the sub-switch-interval starvation is the honest
# boundary documented on every surface (docs/api.md, the .pyi,
# tors/aio.py, the binding docstrings). The green cell below pins the
# qualified claim's own boundary:
# super-interval callbacks on a small text are schedulable between. The
# sub-interval shape is NOT asserted in either direction (starvation is a
# timing property of the interpreter, not a contract); it lives in the
# docs. Zero xfail in this file.


def test_loop_schedulable_between_callbacks_above_the_switch_interval() -> None:
    """GREEN regression witness: the qualified claim's own boundary. A
    GIL-held callback time-budgeted at ~3x ``sys.getswitchinterval()``
    (so it stays above the interval on every interpreter 3.10-3.14)
    straddles the interval and fires ``gil_drop_request``'s fair
    handoff, so the loop stays schedulable between callbacks on a SMALL
    text too: the worst gap tracks the callbacks, never the whole call.
    The sub-switch-interval caveat (a fast GIL-held callback on a small
    text CAN starve the loop for the whole call) is the documented
    boundary (docs/api.md, the ``.pyi``, and the binding docstrings),
    deliberately not asserted here in either direction: it is a timing
    property of the interpreter, not a contract."""

    def busy(s: str) -> int:
        deadline = time.monotonic() + 3 * sys.getswitchinterval()
        n = 0
        i = 0
        while time.monotonic() < deadline:
            for _ in range(1_000):
                i += 1
                n += i * i
        return max(len(s.split()), 1)

    text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. Eleven. Twelve."
    worst, wall = asyncio.run(
        _gap_and_wall(
            lambda: asyncio.to_thread(tors.chunk_to_budget, text, busy, max_tokens=2, overlap=1)
        )
    )
    assert worst < 0.30 * wall or worst < 0.100, (
        f"loop starved: blocked {worst * 1000:.0f}ms of {wall * 1000:.0f}ms "
        f"({worst / wall:.0%}); callbacks held the GIL for ~3x the switch "
        f"interval each, so gil_drop_request's fair handoff should have "
        f"scheduled the loop between them"
    )


# ---- 8. documented edge contracts pinned from the adversarial side ------------


def test_whitespace_only_text_under_word_counter_is_a_value_error() -> None:
    """The zero-sentence contract: whitespace-only text is one sentence
    that IS a whitespace run, it measures 0 tokens under a word-count
    tokenizer, and the call raises the documented ValueError."""
    with pytest.raises(ValueError, match="returned 0"):
        tors.chunk_to_budget("   ", word_counter, max_tokens=5)


def test_whitespace_run_can_go_out_whole_as_the_oversized_exception() -> None:
    """ATTACK: a counter that measures whitespace runs as huge turns
    inter-word whitespace into its own word-level segment, which can go
    out WHOLE as an 'oversized chunk' (a whitespace-only chunk). Within
    the letter of the documented exception (a single UAX #29 word-bound
    unit), but surprising: pinned here so the behavior is a decision, not
    an accident."""
    counter = COUNTER_BATTERY["ws_huge"]
    text = "aa    bb.  cc dd."
    chunks = tors.chunk_to_budget(text, counter, max_tokens=5, overlap=0)
    assert_full_contract(text, chunks, 5, counter)
    assert any(not text[s:e].strip() for s, e in chunks), "whitespace-only chunk emitted"
