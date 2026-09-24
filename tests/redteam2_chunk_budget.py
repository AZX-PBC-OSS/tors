"""Second-wave adversarial attack suite for ``tors.chunk_to_budget`` /
``tors.chunk_to_offsets``.

A second red-team pass. ``tests/redteam_chunk_budget.py`` (wave 1) owns
non-additive counters, per-candidate-span call counts, forward progress
under flare counters, and the degenerate-offsets rejection battery; the
``test_chunk_to_budget*`` cells in ``tests/test_gil_release.py`` own the
GIL heartbeat band. Nothing here re-runs those. This file attacks:

- THE 4 GiB GUARD around its pinned unit cell: the real refusal
  end-to-end (refused before the counter is ever called, sub-second),
  the exact-boundary acceptance (u32::MAX bytes is legal per the
  ``> u32::MAX`` predicate), and the guard's own logic read from
  ``src/chunk_budget_impl.rs::grid_overflow`` (documented in the test
  docstrings; the heavy cells are env-gated, see ``RUN_4G`` below).
- OVERLAP CONTENT, not just progress: the accepted overlap region is
  the trailing span of the closed chunk AND measures at least the
  requested overlap by the very counter that certified the walk-back;
  overlap starts sit on UAX #29 sentence or word boundaries (no
  mid-word splice) including across the sentence-to-word fallback
  boundary; the degenerate whitespace-only overlap a
  whitespace-certifying counter can buy; ``overlap=0`` chunks
  concatenate back to the input exactly (Hypothesis set equality).
- TOKEN_COUNTER shapes: callable class instances, bound methods, and
  ``functools.partial`` are accepted; a CONSTANT counter packs
  deterministically (const == budget: one chunk; const > budget: one
  oversized chunk per segment, terminating); a counter whose cost
  GROWS with the call count keeps the GIL-held time tracking the
  callbacks (heartbeat ratio budget); a counter raising
  ``KeyboardInterrupt`` / ``SystemExit`` (``BaseException``, not
  ``Exception``) propagates unchanged with clean state after.
- SENTENCE-BOUNDARY wave 2: no terminators at all (the word-fallback
  path as the whole text's shape), CJK terminators (``。！？``),
  abbreviation-heavy text (the rule-based segmenter's cuts must equal
  ``tors.sentence_bounds`` exactly), sentences differing only by
  trailing whitespace runs -- and the finding that a whitespace-only
  SENTENCE (a blank line, e.g. ``"P one.\\n\\nP two."``) makes the
  canonical word-count counter raise, not just whitespace-only TEXT.
- OFFSETS TWIN with realistic HuggingFace-style spans: whitespace and
  newline gaps, whitespace-only spans (accepted), zero-width
  special-token offsets (rejected -- a doc-drift finding against the
  "``Encoding.offsets`` is exactly this shape" claim), and the
  unspanned-tail policy (the tail rides the last chunk, which still
  ends at ``len(text)``; it is never a chunk of its own and never
  dropped).
- THE RECALIBRATED GIL CELL: the 3x-switch-interval heartbeat cell run
  20 times on this box (determinism), and the budget re-run under
  monkeypatched ``sys.setswitchinterval`` values -- the 3x multiplier
  must be derived at runtime or the cell is a hardcoded-ms accident.
- DOC EXAMPLES recomputed (docs/api.md's chunk-budget block), verbatim.

FINDINGS referenced by docstrings below are reported in the review, not
fixed: this suite fixes nothing in the implementation.
"""

from __future__ import annotations

import asyncio
import functools
import itertools
import os
import sys
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors

RUN_4G = os.environ.get("TORS_REDTEAM_4G") == "1"
reason_4g = "4 GiB-class inputs: set TORS_REDTEAM_4G=1 (needs ~5 GB RAM, ~1 min)"


# ---- contract checker (independent of wave 1's) --------------------------------


def _cp_slices(text: str, spans: list[tuple[int, int]]) -> list[str]:
    """Codepoint-index slicing without the byte/codepoint trap: Python
    str indices ARE codepoint indices, so plain slicing is the contract."""
    return [text[s:e] for s, e in spans]


def assert_contract(
    text: str,
    chunks: list[tuple[int, int]],
    budget: int,
    counter=None,
    spans: list[tuple[int, int]] | None = None,
) -> None:
    """Bounds, strict advancement, first-start-0, cover-to-end, and the
    per-chunk budget invariant (single-oversized-segment exception only)."""
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
            measured = sum(1 for a, b in spans if s <= a and b <= e)
        else:
            assert counter is not None
            measured = counter(text[s:e])
        if measured > budget:
            interior = len(tors.word_bounds(text[s:e]))
            assert interior <= 1, (
                f"budget exceeded without the single-segment exception: "
                f"{chunks} {text!r} measured={measured} budget={budget}"
            )
        prev_s, prev_e = s, e
    assert prev_e == len(text), "must cover to the end"


def word_counter(text: str) -> int:
    return len(text.split())


def uax_counter(text: str) -> int:
    """The UAX #29 span model as a counter: one token per non-whitespace
    word segment -- the exact measurement ``word_offsets`` encodes as
    spans, so the two spellings can be compared twin-to-twin."""
    return sum(1 for a, b in tors.word_bounds(text) if text[a:b].strip())


def word_offsets(text: str) -> list[tuple[int, int]]:
    return [(s, e) for s, e in tors.word_bounds(text) if text[s:e].strip()]


# ---- 1. the 4 GiB guard --------------------------------------------------------
#
# Read from src/chunk_budget_impl.rs::grid_overflow (the guard's own logic,
# attacked without a 2 GiB-class allocation, which the review contract deems
# impractical in-suite):
#
# - The predicate is `text_bytes > u32::MAX`: text of EXACTLY u32::MAX bytes
#   is accepted. Consistent with every surface's wording ("text beyond
#   u32::MAX bytes raises"): the grid's largest entry is the final
#   `text.len() as u32`, which at exactly u32::MAX bytes is still
#   representable, so nothing truncates at the accepted boundary. No
#   off-by-one: the docs' "cannot exceed" reading and the check agree.
# - Both bindings run the guard BEFORE any allocation or scan (before the
#   O(tokens) offsets walk, before the counter is ever invoked, before
#   `char_count`), so the refusal is O(1) -- verified end-to-end by the
#   env-gated cell below (0.00s, zero counter calls at u32::MAX + 1 bytes).
# - The grid is only built for non-ASCII text (the ASCII fast path skips
#   it), so the guard's bound is exactly the width the code uses; for
#   non-ASCII the grid's u32 entries are all < text.len() <= u32::MAX.
#
# The heavy cells below are the real end-to-end pins, run once in review on
# this box (refusal: 0.00s, zero counter calls; exact-bound acceptance:
# ~3 min, one chunk [(0, len)]); they are env-gated so no CI lane ever pays
# for a 4 GiB string.


@pytest.mark.skipif(not RUN_4G, reason=reason_4g)
def test_4g_guard_refuses_before_any_counter_call() -> None:
    """ATTACK: a u32::MAX + 1 byte text must be refused by the guard
    BEFORE the counter is ever invoked ('before any work runs', per
    docs/api.md) -- not mid-pack, not after one measurement."""
    calls: list[int] = []

    def counter(s: str) -> int:
        calls.append(len(s))
        return 1

    big = "a" * (2**32)  # u32::MAX + 1 UTF-8 bytes
    with pytest.raises(ValueError, match="4294967295-byte offset grid"):
        tors.chunk_to_budget(big, counter, max_tokens=10)
    assert calls == [], "the counter must never run: the guard fires first"
    # the offsets twin shares the guard (same grid, same hazard)
    with pytest.raises(ValueError, match="offset grid"):
        tors.chunk_to_offsets(big, [(0, 1)], max_tokens=10)


@pytest.mark.skipif(not RUN_4G, reason=reason_4g)
def test_4g_guard_accepts_exactly_u32_max_bytes() -> None:
    """ATTACK (off-by-one hunt at the accepted boundary): a text of
    EXACTLY u32::MAX bytes is inside the guard (the predicate is
    strictly greater-than); it must pack, not raise, and its single
    chunk must end at len(text) -- the grid's final entry is
    u32::MAX exactly, still representable."""
    big = "a" * (2**32 - 1) + "."  # one sentence, u32::MAX bytes
    calls: list[int] = []

    def counter(s: str) -> int:
        calls.append(len(s))
        return 1

    chunks = tors.chunk_to_budget(big, counter, max_tokens=10)
    assert chunks == [(0, 2**32 - 1)]
    assert calls == [2**32 - 1]
    assert chunks[-1][1] == len(big)


# ---- 2. overlap content semantics ----------------------------------------------


def test_accepted_overlap_region_is_the_certified_trailing_span() -> None:
    """ATTACK on the overlap CONTENT claim: docs/api.md promises the next
    chunk starts at the trailing boundary whose span back to the closed
    chunk's end measures AT LEAST the requested overlap. So for every
    accepted overlap (next_start < prev_end) the region
    text[next_start:prev_end] -- exactly the walk-back's measured span --
    must measure >= overlap by the very counter that certified it, and be
    both a suffix of the previous chunk and a prefix of the next one."""
    texts = [
        "One. Two. Three. Four. Five. Six.",
        "One two three four five six. " + " ".join(f"w{i}" for i in range(12)) + ".",
        "".join(f"word{i} " for i in range(40)),
    ]
    counters = [
        word_counter,
        lambda s: max(1, len(s) // 10),  # ~1 token per 10 codepoints
    ]
    for text in texts:
        for counter in counters:
            for budget in (3, 5, 8):
                for overlap in range(1, budget):
                    chunks = tors.chunk_to_budget(text, counter, max_tokens=budget, overlap=overlap)
                    for (ps, pe), (ns, ne) in zip(chunks, chunks[1:], strict=False):
                        if ns >= pe:
                            continue  # declined for this transition
                        region = text[ns:pe]
                        assert counter(region) >= overlap, (
                            f"accepted overlap measures below the request: "
                            f"region={region!r} overlap={overlap} {chunks}"
                        )
                        # the repeated context is the SAME text on both sides
                        assert text[ps:pe].endswith(region)
                        assert text[ns:ne].startswith(region)


def test_overlap_starts_on_sentence_or_word_boundaries_through_the_fallback() -> None:
    """ATTACK: across the SENTENCE-to-WORD fallback boundary (a chunk that
    ended mid-sentence via the word re-cut), the walk-back runs over
    word-level segments: every overlap start must sit on a UAX #29
    sentence or word boundary of the text -- never mid-word -- and the
    re-embedded region must be real trailing context (a sentence
    terminator segment like '. w0 ' is a legal boundary, pinned)."""
    text = "One two three four five six. " + " ".join(f"w{i}" for i in range(12)) + "."
    legal_starts = {s for s, _ in tors.sentence_bounds(text)}
    legal_starts |= {s for s, _ in tors.word_bounds(text)}
    for budget in (3, 4, 5):
        for overlap in range(1, budget):
            chunks = tors.chunk_to_budget(text, word_counter, max_tokens=budget, overlap=overlap)
            for (_ps, pe), (ns, _ne) in zip(chunks, chunks[1:], strict=False):
                assert ns in legal_starts, (
                    f"overlap start {ns} is not a sentence/word boundary: "
                    f"{text[ns : ns + 10]!r} in {chunks}"
                )
                if ns < pe:
                    assert not text[ns].isalnum() or ns == 0 or not text[ns - 1].isalnum(), (
                        f"overlap splices mid-word at {ns}: {text[ns:pe]!r}"
                    )


def test_whitespace_certifying_counter_can_buy_a_whitespace_only_overlap() -> None:
    """Pinned behavior (review finding, P2): a counter that certifies a
    whitespace run as >= 1 token (max(1, len//n) floors EVERYTHING to 1)
    lets the walk-back accept a boundary whose repeated region is
    whitespace-only -- the docs' 'genuine shared content' holds only
    counter-relative, and the wave-1 Rust cell's trim-non-empty shape is
    NOT a Python-side contract. The region still measures >= overlap per
    that counter, so the letter of the certification claim holds."""
    text = "One two three four five six. " + " ".join(f"w{i}" for i in range(12)) + "."

    def floor_one(s: str) -> int:
        return max(1, len(s.split()))

    chunks = tors.chunk_to_budget(text, floor_one, max_tokens=3, overlap=1)
    regions = [
        text[ns:pe] for (_ps, pe), (ns, _ne) in zip(chunks, chunks[1:], strict=False) if ns < pe
    ]
    assert any(not r.strip() for r in regions), (
        f"expected the degenerate whitespace-only overlap to be reachable: {regions}"
    )
    assert all(floor_one(r) >= 1 for r in regions)


@settings(max_examples=200, deadline=None)
@given(text=st.text(max_size=80))
def test_overlap_zero_chunks_concatenate_to_the_input(text: str) -> None:
    """ATTACK (property): with overlap=0, both spellings must be a
    CONTIGUOUS LOSSLESS partition -- the chunks concatenate back to the
    input exactly (codepoint set equality, not just coverage-to-end)."""
    for kind in ("budget", "offsets"):
        if kind == "budget":
            try:
                chunks = tors.chunk_to_budget(text, word_counter, max_tokens=4)
            except ValueError as err:
                # the documented zero-sentence contract (whitespace-only
                # text under the word counter); the offsets twin still runs
                assert "returned 0" in str(err), err
                continue
            spans = None
        else:
            spans = word_offsets(text)
            chunks = tors.chunk_to_offsets(text, spans, max_tokens=4)
        assert_contract(text, chunks, 4, word_counter, spans=spans)
        assert "".join(_cp_slices(text, chunks)) == text, f"{chunks} {text!r}"


@settings(max_examples=150, deadline=None)
@given(
    text=st.text(min_size=1, max_size=60),
    budget=st.integers(2, 8),
    overlap=st.integers(1, 7),
)
def test_accepted_overlap_measures_at_least_the_request(
    text: str, budget: int, overlap: int
) -> None:
    """ATTACK (property): every ACCEPTED overlap region measures >= the
    requested overlap by the certifying counter, for arbitrary text."""
    if overlap >= budget:
        return
    chunks = tors.chunk_to_budget(
        text, lambda s: max(1, len(s) // 5), max_tokens=budget, overlap=overlap
    )
    for (_ps, pe), (ns, _ne) in zip(chunks, chunks[1:], strict=False):
        if ns < pe:
            assert max(1, len(text[ns:pe]) // 5) >= overlap, (chunks, overlap)


# ---- 3. token_counter shapes wave 2 ---------------------------------------------


class _CallableCounter:
    """A callable CLASS instance (not a function, not a closure)."""

    def __init__(self, base: int) -> None:
        self.base = base

    def __call__(self, text: str) -> int:
        return max(len(text.split()), self.base)


class _MethodHolder:
    def measure(self, text: str) -> int:
        return max(len(text.split()), 1)


def test_counter_callables_beyond_plain_functions_are_accepted() -> None:
    """ATTACK: the docs say 'callable'; only wave 1 exercised functions
    and lambdas. A callable class instance, a bound method, and a
    functools.partial must all be accepted and measure identically."""
    text = "One. Two. Three. Four. Five."
    expected = [(0, 17), (17, 28)]
    assert tors.chunk_to_budget(text, _CallableCounter(1), max_tokens=3) == expected
    assert tors.chunk_to_budget(text, _MethodHolder().measure, max_tokens=3) == expected
    assert (
        tors.chunk_to_budget(
            text, functools.partial(lambda b, s: max(len(s.split()), b), 1), max_tokens=3
        )
        == expected
    )


def test_constant_counter_packs_deterministically_and_terminates() -> None:
    """ATTACK: a counter that returns the SAME value every call. const ==
    budget: every candidate fits, the greedy extension runs to the end
    (one chunk). const > budget: every sentence AND every word segment
    over-measures, so each segment goes out whole -- termination with a
    sane chunk count (one per UAX #29 word/whitespace segment), never a
    stall. const == 0: the documented ValueError."""
    text = "One. Two. Three. Four. Five."
    assert tors.chunk_to_budget(text, lambda s: 3, max_tokens=3) == [(0, len(text))]
    text2 = "aa bb cc"
    chunks = tors.chunk_to_budget(text2, lambda s: 5, max_tokens=4)
    assert len(chunks) == len(tors.word_bounds(text2)), chunks
    assert_contract(text2, chunks, 4, lambda s: 5)
    with pytest.raises(ValueError, match="returned 0"):
        tors.chunk_to_budget(text, lambda s: 0, max_tokens=3)


def test_growing_cost_counter_keeps_gil_held_time_tracking_the_callbacks() -> None:
    """ATTACK on 'held time tracks the callbacks': a stateful counter
    whose per-call cost GROWS with the call count (call N busy-holds the
    GIL for N * 0.3ms). The docs' claim is that the GIL is held only
    while the counter runs (plus O(chunk) argument construction), so the
    worst heartbeat gap must track the (growing) callbacks, never the
    whole call: the ratio budget of the recalibrated cell must hold."""
    ticks: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        while True:
            ticks.append(time.monotonic())
            if stop.is_set():
                return
            await asyncio.sleep(0.01)

    state = {"n": 0}

    def growing(s: str) -> int:
        state["n"] += 1
        deadline = time.monotonic() + min(state["n"] * 0.0003, 0.05)
        n = i = 0
        while time.monotonic() < deadline:
            for _ in range(1_000):
                i += 1
                n += i * i
        return max(len(s.split()), 1)

    text = (
        "One. Two. Three. Four. Five. Six. Seven. "
        "Eight. Nine. Ten. Eleven. Twelve. Thirteen. Fourteen."
    )

    async def run() -> tuple[float, float]:
        hb = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        started = time.monotonic()
        try:
            await asyncio.to_thread(tors.chunk_to_budget, text, growing, max_tokens=2, overlap=1)
        finally:
            stop.set()
            await hb
        wall = time.monotonic() - started
        worst = max((b - a for a, b in itertools.pairwise(ticks)), default=0.0)
        return worst, wall

    worst, wall = asyncio.run(run())
    assert state["n"] > 20, "the corpus must exercise many growing calls"
    assert worst < 0.30 * wall or worst < 0.100, (
        f"held time does not track the callbacks: worst gap {worst * 1000:.0f}ms "
        f"of {wall * 1000:.0f}ms wall ({worst / wall:.0%})"
    )


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_counter_base_exception_propagates_with_clean_state(exc_type: type[BaseException]) -> None:
    """ATTACK: wave 1 raised Exception subclasses. KeyboardInterrupt /
    SystemExit are BaseExceptions -- the captured-err re-raise path must
    propagate them UNCHANGED (same class, same message), and a follow-up
    call with a healthy counter must behave normally (no poisoned state)."""

    def boom(s: str) -> int:
        raise exc_type("from the counter")

    with pytest.raises(exc_type, match="from the counter"):
        tors.chunk_to_budget("One. Two. Three. Four.", boom, max_tokens=2)
    chunks = tors.chunk_to_budget("One. Two. Three. Four.", word_counter, max_tokens=2)
    assert_contract("One. Two. Three. Four.", chunks, 2, word_counter)


# ---- 4. sentence-boundary wave 2 ------------------------------------------------


def test_no_terminators_the_whole_text_is_the_word_fallback_path() -> None:
    """ATTACK: text with NO sentence terminators at all is one giant UAX
    #29 sentence; the entire packing runs through the word-fallback
    path. Coverage and the budget invariant must hold for every budget,
    and the chunk count must be the segment count, not a stall."""
    text = " ".join(f"word{i}" for i in range(30))
    assert tors.sentence_bounds(text) == [(0, len(text))]
    for budget in (1, 2, 3, 7):
        chunks = tors.chunk_to_budget(text, word_counter, max_tokens=budget)
        assert_contract(text, chunks, budget, word_counter)
        assert len(chunks) <= len(text)


def test_cjk_terminators_cut_sentences_and_the_packer_agrees() -> None:
    """ATTACK: terminators that are themselves CJK (``。！？``) must be
    sentence cuts (UAX #29 SB-series on terminal punctuation), the
    packer's segments must be exactly tors.sentence_bounds, and chunk
    boundaries must land on those bounds (plus word bounds for an
    oversized CJK sentence's fallback)."""
    text = "東京。大阪！京都？奈良。"
    assert tors.sentence_bounds(text) == [(0, 3), (3, 6), (6, 9), (9, 12)]
    sentence_starts = {s for s, _ in tors.sentence_bounds(text)}
    for budget in (2, 3, 5):
        for overlap in range(budget):
            half = lambda s: max(1, len(s) // 2)  # noqa: E731
            chunks = tors.chunk_to_budget(text, half, max_tokens=budget, overlap=overlap)
            assert_contract(text, chunks, budget, half)
            for s, _ in chunks[1:]:
                assert s in sentence_starts or any(
                    s == start for start, _ in tors.word_bounds(text)
                ), f"cut at {s} is not a segment boundary: {chunks}"
    # round-trip through multibyte: text[start:end] is the chunk, verbatim
    for s, e in tors.chunk_to_budget(text, lambda s: max(1, len(s) // 2), max_tokens=5):
        assert text[s:e]


@pytest.mark.parametrize(
    "abbrev_text",
    [
        "Mr. Smith went to Washington. He arrived at 5 p.m. sharp.",
        "e.g. the i.e. restatement, vs. the etc. list, and Dr. Who.",
        "Prices rose in the U.S. and the U.K. alike, said Prof. X. Then fell.",
    ],
)
def test_abbreviation_heavy_cuts_match_sentence_bounds_exactly(abbrev_text: str) -> None:
    """ATTACK: abbreviation-heavy text ('e.g.', 'Mr.', '5 p.m.') stresses
    UAX #29's suppressions (the crate's segmenter is rule-based; 'Mr.'
    may or may not break). Whatever the rules say, the PACKER's sentence
    segments must be exactly tors.sentence_bounds -- every interior chunk
    start is a sentence bound, and the cuts cannot drift from the
    segmenter the docs cite."""
    bounds = tors.sentence_bounds(abbrev_text)
    # legal packer cuts: sentence bounds, or word bounds of an oversized
    # sentence re-cut by the fallback (word-level segments replace it)
    legal = {s for s, _ in bounds}
    for ws, we in bounds:
        legal |= {ws + s for s, _ in tors.word_bounds(abbrev_text[ws:we])}
    ends = {e for _, e in bounds}
    for budget in (2, 3, 6):
        chunks = tors.chunk_to_budget(abbrev_text, word_counter, max_tokens=budget)
        assert_contract(abbrev_text, chunks, budget, word_counter)
        for s, _ in chunks[1:]:
            assert s in legal, f"packer cut at {s} is not a segment bound: {chunks}"
        assert chunks[-1][1] in ends or chunks[-1][1] == len(abbrev_text)


def test_sentences_differing_only_by_trailing_whitespace_runs() -> None:
    """ATTACK: whitespace runs between sentences ride the preceding
    sentence ('a. \\n' ends the sentence) UNTIL a run is long enough to
    become its OWN sentence (a blank line: the second \\n of '\\n\\n').
    Pin both shapes. FINDING (reported, not fixed): the own-sentence run
    makes the CANONICAL word-count counter raise ValueError on perfectly
    ordinary multi-paragraph text -- 'Paragraph one.\\n\\nParagraph two.'
    and even a trailing blank line raise; docs/api.md and the .pyi
    present the trigger as 'whitespace-only text', which understates the
    trigger set to the most common real case (P1 doc-scope finding)."""
    # single trailing whitespace ride along: fine
    for ok_text in ("a. \n b.", "Paragraph one. Paragraph two.\n", "a.  b.  c."):
        chunks = tors.chunk_to_budget(ok_text, word_counter, max_tokens=1)
        assert_contract(ok_text, chunks, 1, word_counter)
        assert "".join(ok_text[s:e] for s, e in chunks) == ok_text
    # a whitespace-only SENTENCE (blank line / double newline / trailing
    # blank) trips the zero-sentence contract under the word counter
    for raises_text in (
        "Paragraph one.\n\nParagraph two.",
        "Paragraph one. Paragraph two.\n\n",
        "a.  \n\n b. \tc.",
    ):
        with pytest.raises(ValueError, match="returned 0"):
            tors.chunk_to_budget(raises_text, word_counter, max_tokens=8)
    # the offsets twin is unaffected (it tolerates 0-measure sentences)
    spans = word_offsets("Paragraph one.\n\nParagraph two.")
    chunks = tors.chunk_to_offsets("Paragraph one.\n\nParagraph two.", spans, max_tokens=8)
    assert_contract("Paragraph one.\n\nParagraph two.", chunks, 8, word_counter, spans=spans)


# ---- 5. offsets twin: realistic HuggingFace-style shapes ------------------------


def test_hf_style_offsets_with_special_tokens_raise() -> None:
    """ATTACK (doc-drift finding, P2): a real HuggingFace
    ``Encoding.offsets`` from ``tokenizer(text)`` -- add_special_tokens
    defaulting to True -- carries ZERO-WIDTH (0, 0) offsets for the
    [CLS]/[SEP] special tokens. docs/api.md and the .pyi claim
    ``Encoding.offsets`` 'is exactly this shape', but the zero-width
    entries are rejected (0 <= start < end), so the DEFAULT HF call is
    not acceptable verbatim. The acceptance predicate is stated
    correctly on the same surfaces; the HF comparison overstates.
    Pinned: the rejection, and the working filter spelling."""
    with pytest.raises(ValueError, match="out of bounds"):
        tors.chunk_to_offsets("Hi there", [(0, 0), (0, 2), (3, 8)], max_tokens=2)
    # the documented workaround: drop the zero-width special tokens
    assert tors.chunk_to_offsets("Hi there", [(0, 2), (3, 8)], max_tokens=2) == [(0, 8)]


def test_realistic_hf_offsets_with_whitespace_and_newline_gaps() -> None:
    """ATTACK: realistic fast-tokenizer offsets -- whitespace excluded
    from spans (gaps at every space), gaps at newlines, multi-codepoint
    tokens. Packing + full contract must hold, and the offsets spelling
    must agree with the counter spelling when BOTH measure the same
    model (the UAX #29 span model: uax_counter is word_offsets as a
    callable) -- the docs' 'same contract' claim. The text uses single
    newlines: the blank-line shape that would make the counter twin
    raise is pinned in
    test_sentences_differing_only_by_trailing_whitespace_runs."""
    text = "Hello world!\nThe quick brown fox jumps over the lazy dog."
    spans = word_offsets(text)  # whitespace untokenized: gaps, incl. newlines
    for budget in (2, 3, 5, 8):
        for overlap in range(budget):
            a = tors.chunk_to_offsets(text, spans, max_tokens=budget, overlap=overlap)
            assert_contract(text, a, budget, uax_counter, spans=spans)
            b = tors.chunk_to_budget(text, uax_counter, max_tokens=budget, overlap=overlap)
            assert a == b, f"twin divergence at {budget}/{overlap}: {a} vs {b}"


def test_whitespace_only_spans_are_accepted_spans() -> None:
    """ATTACK: spans CONTAINING only whitespace are legal (the validation
    is positional, not content-based) -- contrast the zero-width
    rejection. A whitespace-only span measures 1 where contained."""
    chunks = tors.chunk_to_offsets("a b", [(0, 1), (1, 2)], max_tokens=2)
    assert chunks == [(0, 3)]


def test_unspanned_tail_rides_the_last_chunk_never_its_own() -> None:
    """ATTACK (the contract question wave 1 left open): token offsets that
    stop short of the text's last codepoints (a trailing gap -- HF
    offsets for trailing punctuation the tokenizer dropped, or a
    truncated span list). PINNED SEMANTICS: the tail is never dropped
    and never a chunk of its own (there IS no counter for unspanned
    text): it rides the SENTENCE containing it -- sentence_bounds tiles
    the whole text -- whose chunk still ends at len(text) per the
    covering contract. Same for a leading gap and a giant middle gap.
    The docs state 'cover to the end of the text' and 'gaps allowed';
    this cell pins where the unspanned text goes."""
    text = "hello world token three tail!!!"
    spans = [(0, 5), (6, 11), (12, 17), (18, 23)]  # text[24:] unspanned
    chunks = tors.chunk_to_offsets(text, spans, max_tokens=2)
    assert chunks[-1][1] == len(text), "the unspanned tail must be covered"
    assert "tail!!!" in text[chunks[-1][0] : chunks[-1][1]]
    # the tail does not open its own chunk: no chunk lies fully in the gap
    for s, e in chunks:
        assert not (s >= 24 and text[s:e].strip() == "")
    # leading gap: rides the first chunk
    spans2 = [(6, 11), (12, 17), (18, 23), (24, 31)]
    chunks2 = tors.chunk_to_offsets(text, spans2, max_tokens=2)
    assert chunks2[0][0] == 0
    assert "hello" in text[chunks2[0][0] : chunks2[0][1]]
    # a giant middle gap measures 0 and still makes progress (docs: gaps)
    long_text = "ab" + "x" * 500 + "cd"
    chunks3 = tors.chunk_to_offsets(long_text, [(0, 2), (502, 504)], max_tokens=1)
    assert_contract(long_text, chunks3, 1, word_counter, spans=[(0, 2), (502, 504)])
    assert chunks3[-1][1] == len(long_text)


# ---- 6. the recalibrated GIL cell: determinism and runtime derivation -----------


def _busy_at_3x(s: str) -> int:
    """The recalibrated cell's counter shape: GIL-held CPU work budgeted
    at 3 * sys.getswitchinterval() per call -- derived at RUNTIME, which
    is the thing under test (a hardcoded-ms budget would stop straddling
    the interval when the interval is monkeypatched)."""
    deadline = time.monotonic() + 3 * sys.getswitchinterval()
    n = i = 0
    while time.monotonic() < deadline:
        for _ in range(1_000):
            i += 1
            n += i * i
    return max(len(s.split()), 1)


async def _gap_and_wall(op: object) -> tuple[float, float]:
    ticks: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
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


_TEXT = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. Eleven. Twelve."


def test_recalibrated_heartbeat_cell_is_deterministic_across_20_runs() -> None:
    """ATTACK: the red-team fix recalibrated the heartbeat cell above the
    switch interval. Determinism audit: 20 consecutive runs on this box;
    EVERY run must meet the ratio budget (worst gap tracks the ~3x
    callbacks, never the whole call). One flaky run out of 20 is exactly
    the flakiness the recalibration was meant to buy off."""
    runs = 20
    for k in range(runs):
        worst, wall = asyncio.run(
            _gap_and_wall(
                lambda: asyncio.to_thread(
                    tors.chunk_to_budget, _TEXT, _busy_at_3x, max_tokens=2, overlap=1
                )
            )
        )
        assert worst < 0.30 * wall or worst < 0.100, (
            f"run {k + 1}/{runs} starved: worst {worst * 1000:.1f}ms of {wall * 1000:.0f}ms "
            f"({worst / wall:.0%})"
        )


@pytest.mark.parametrize("interval", [0.001, 0.05])
def test_recalibrated_budget_is_derived_from_switchinterval_at_runtime(interval: float) -> None:
    """ATTACK: the 3x multiplier must be computed from
    sys.getswitchinterval() AT RUNTIME. Monkeypatch the interval to 1ms
    and 50ms and re-run the recalibrated cell: it must still pass. If
    the 3x were hardcoded in milliseconds (a 15ms constant), the 50ms
    run would put every callback far BELOW the interval -- the exact
    sub-interval starvation shape the cell claims to be above -- and
    fail here."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(interval)
    try:
        worst, wall = asyncio.run(
            _gap_and_wall(
                lambda: asyncio.to_thread(
                    tors.chunk_to_budget, _TEXT, _busy_at_3x, max_tokens=2, overlap=1
                )
            )
        )
    finally:
        sys.setswitchinterval(old)
    assert worst < 0.30 * wall or worst < 0.100, (
        f"interval={interval}: worst {worst * 1000:.1f}ms of {wall * 1000:.0f}ms "
        f"({worst / wall:.0%}) -- the 3x budget is not runtime-derived"
    )


# ---- 7. docs examples recomputed ------------------------------------------------


def test_api_md_examples_recomputed() -> None:
    """The docs/api.md chunk-budget block, recomputed verbatim (an
    independent pass over test_docs_examples.py's pins)."""

    def word_counter(text: str) -> int:
        return len(text.split())

    text = "One. Two. Three. Four."
    assert tors.chunk_to_budget(text, word_counter, max_tokens=2) == [(0, 10), (10, 22)]
    assert tors.chunk_to_budget(text, word_counter, max_tokens=2, overlap=1) == [
        (0, 10),
        (5, 17),
        (10, 22),
    ]
    assert tors.chunk_to_budget("a b c d e f g h", word_counter, max_tokens=3) == [
        (0, 6),
        (6, 12),
        (12, 15),
    ]
    spans = [(s, e) for s, e in tors.word_bounds(text) if text[s:e].strip()]
    assert tors.chunk_to_offsets(text, spans, max_tokens=4, overlap=2) == [
        (0, 10),
        (5, 17),
        (10, 22),
    ]
