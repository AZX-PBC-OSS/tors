"""Contract gate for ``tors.chunk_to_budget`` and ``tors.chunk_to_offsets``:
token-budget chunking measured by the caller's own token counter (the
``CompiledLemmaDict``-style measured exception: a Python callable inside the
packing), and its GIL-free twin over pre-computed token spans.

The properties pinned here:

- the budget invariant: every chunk individually fits ``max_tokens`` as
  measured by the SAME counter the packing used (the one documented
  exception: a single word wider than the whole budget goes out whole;
  a covering chunker cannot split below its finest boundary);
- the coverage invariant: chunks are non-empty, start and end strictly
  advance, the first chunk starts at 0 and the last ends at
  ``len(text)``; with ``overlap=0`` the chunks are a contiguous lossless
  covering partition (``"".join(text[s:e] for ...) == text``);
- codepoint correctness: ``text[start:end]`` is the chunk through CJK,
  astral emoji, and combining marks (a byte-offset chunker would panic
  or slice mid-character);
- overlap: int token counts and float ratios both produce genuine shared
  content between consecutive chunks, and the decline-the-snap rule
  (a transition that cannot buy new context degrades to zero overlap)
  keeps chunks out of their predecessors;
- validation: exhaustive, at the argument boundary: ``max_tokens < 1``,
  an int ``overlap`` outside ``[0, max_tokens)``, a float ratio outside
  ``[0, 1)``, a non-callable counter, and counters returning 0/negative/
  huge/non-int all raise before any packing runs;
- liveness: pathological counters (non-monotone, all-max, all-min-but-
  positive) terminate; chunk counts stay bounded by the codepoint
  count, pinned with Hypothesis;
- determinism: the same (text, counter, arguments) always yields the
  same chunks;
- scaling: growth stays linear on the text-size axis (the
  ``test_scaling_pins.py`` two-size pattern);
- memory: no superlinear accumulation on the packing pass (peak-RSS
  guard, the ``test_memory_spike_guards.py`` discipline);
- GIL honesty: ``chunk_to_offsets`` passes the heartbeat budgets (its
  whole native pass is one ``py.detach``); ``chunk_to_budget`` is
  honestly NOT GIL-free (its counter is Python) and what IS true is
  pinned here: the per-callback GIL handoffs keep the loop schedulable
  between callbacks, including under counters that release the GIL
  themselves or re-enter tors. The family's harness cells live in
  ``tests/test_gil_release.py``.

Adversarial coverage beyond the base contract:

- non-additive and non-monotone counters: BPE-style merging that
  measures a merged span less than its parts, a superadditive space
  penalty (``counter(whole) > counter(a) + counter(b)``), whitespace-run-
  hostile counters, flare counters keyed to a span's first or last
  codepoint, and a counter returning exactly ``max_tokens`` for the
  oversized single word;
- callback adversarial: side-effecting and stateful counters, counters
  that raise mid-pack (the traceback surfaces intact),
  ``BaseException`` subclasses, re-entrant counters that call tors
  functions inside the callback, counters that release the GIL
  themselves, and the per-candidate-span (never per-boundary)
  call-count claim instrumented;
- overlap content, not just progress: the accepted overlap region is
  the trailing span of the closed chunk and measures at least the
  requested overlap by the certifying counter, overlap starts sit on
  UAX #29 sentence or word boundaries including across the
  sentence-to-word fallback, and a counter that certifies whitespace as
  a token can buy a whitespace-only overlap;
- the 4 GiB offset-grid guard around its pinned unit cell (the heavy
  cells are env-gated: ``TORS_REDTEAM_4G=1``);
- sentence-boundary shapes: no terminators at all, CJK terminators,
  abbreviation-heavy text, sentences differing only by trailing
  whitespace runs, and the whitespace-only SENTENCE trigger (a blank
  line) that makes the canonical word-count counter raise;
- the offsets twin over realistic HuggingFace-style spans: whitespace
  and newline gaps, whitespace-only spans, zero-width special-token
  offsets (rejected), and the unspanned-tail policy;
- the docs' examples, recomputed verbatim.
"""

from __future__ import annotations

import asyncio
import functools
import itertools
import os
import sys
import time
import traceback

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from loop_harness import heartbeat_gap_and_wall

import tors

RUN_4G = os.environ.get("TORS_REDTEAM_4G") == "1"
reason_4g = "4 GiB-class inputs: set TORS_REDTEAM_4G=1 (needs ~5 GB RAM, ~1 min)"

# ---- helpers ----------------------------------------------------------------


def word_counter(text: str) -> int:
    """The docs' counter: ``len(text.split())``, no third-party tokenizer."""
    return len(text.split())


def char_counter(text: str) -> int:
    """A codepoint counter: every codepoint is one token."""
    return len(text)


def uax_counter(text: str) -> int:
    """The UAX #29 span model as a counter: one token per non-whitespace
    word segment -- the exact measurement ``word_offsets`` encodes as
    spans, so the two spellings can be compared twin-to-twin."""
    return sum(1 for a, b in tors.word_bounds(text) if text[a:b].strip())


def word_offsets(text: str) -> list[tuple[int, int]]:
    """Token spans aligned to the word counter, via tors's own UAX #29
    word segmentation filtered to non-whitespace segments: the realistic
    tokenizer-offsets shape for ``chunk_to_offsets`` (each span is one
    token, whitespace between them is untokenized gap)."""
    return [(s, e) for s, e in tors.word_bounds(text) if text[s:e].strip()]


def assert_full_contract(
    text: str,
    chunks: list[tuple[int, int]],
    budget: int,
    counter=None,
    spans: list[tuple[int, int]] | None = None,
) -> None:
    """The whole packing contract, re-derived independently of the impl:
    bounds, strict advancement, first-start-0, cover-to-end, and the
    per-chunk budget invariant (the single-oversized-segment exception is
    the only out). With ``spans``, the budget invariant is measured the
    offsets spelling's own way: fully-contained token pairs."""
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


def _cp_slices(text: str, spans: list[tuple[int, int]]) -> list[str]:
    """Codepoint-index slicing without the byte/codepoint trap: Python
    str indices ARE codepoint indices, so plain slicing is the contract."""
    return [text[s:e] for s, e in spans]


def _flare_counter(high: int, low: int):
    """A deterministic NON-MONOTONE counter: its answer depends only on
    the span's first codepoint's parity, not its length: the shape that
    breaks sum-based or binary-search packers. Positive always (a 0
    return is a documented ValueError, not an adversarial shape)."""

    def counter(text: str) -> int:
        first = ord(text[0]) if text else 1
        return high if first % 3 == 0 else low

    return counter


# ---- validation -------------------------------------------------------------


class TestValidation:
    TEXT = "One. Two. Three. Four."

    @pytest.mark.parametrize("max_tokens", [0, -1, -100])
    def test_max_tokens_below_one_is_refused(self, max_tokens: int) -> None:
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            tors.chunk_to_budget(self.TEXT, word_counter, max_tokens=max_tokens)
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            tors.chunk_to_offsets(self.TEXT, word_offsets(self.TEXT), max_tokens=max_tokens)

    @pytest.mark.parametrize("overlap", [-1, -7, 2, 5])
    def test_int_overlap_outside_range_is_refused(self, overlap: int) -> None:
        # overlap must be in [0, max_tokens): 2 and 5 with max_tokens=2
        # have no forward progress.
        with pytest.raises(ValueError, match="overlap"):
            tors.chunk_to_budget(self.TEXT, word_counter, max_tokens=2, overlap=overlap)
        with pytest.raises(ValueError, match="overlap"):
            tors.chunk_to_offsets(
                self.TEXT, word_offsets(self.TEXT), max_tokens=2, overlap=overlap
            )

    @pytest.mark.parametrize("overlap", [-0.5, 1.0, 1.5, 37.0])
    def test_float_overlap_outside_unit_range_is_refused(self, overlap: float) -> None:
        with pytest.raises(ValueError, match="overlap ratio"):
            tors.chunk_to_budget(self.TEXT, word_counter, max_tokens=2, overlap=overlap)
        with pytest.raises(ValueError, match="overlap ratio"):
            tors.chunk_to_offsets(
                self.TEXT, word_offsets(self.TEXT), max_tokens=2, overlap=overlap
            )

    def test_nan_overlap_is_refused(self) -> None:
        nan = float("nan")
        with pytest.raises(ValueError, match="overlap ratio"):
            tors.chunk_to_budget(self.TEXT, word_counter, max_tokens=2, overlap=nan)

    @pytest.mark.parametrize("overlap", ["1", [1], b"1"])
    def test_non_numeric_overlap_is_a_type_error(self, overlap: object) -> None:
        with pytest.raises(TypeError, match="overlap"):
            tors.chunk_to_budget(self.TEXT, word_counter, max_tokens=2, overlap=overlap)

    def test_none_overlap_is_the_zero_overlap_spelling(self) -> None:
        # The runtime default's own spelling (see the text_signature
        # note on the binding): None resolves as zero overlap.
        assert tors.chunk_to_budget(self.TEXT, word_counter, max_tokens=2, overlap=None) == (
            tors.chunk_to_budget(self.TEXT, word_counter, max_tokens=2)
        )

    @pytest.mark.parametrize("not_callable", [42, "counter", None, [word_counter]])
    def test_non_callable_counter_is_a_type_error(self, not_callable: object) -> None:
        with pytest.raises(TypeError, match="token_counter must be callable"):
            tors.chunk_to_budget(self.TEXT, not_callable, max_tokens=2)

    @pytest.mark.parametrize("returned", [0, -1, -42, -(2**40)])
    def test_zero_or_negative_counts_are_value_errors(self, returned: int) -> None:
        # 0 for a whole sentence means the budget contract is meaningless;
        # negative is never a count. The first sentence is measured before
        # anything is packed.
        with pytest.raises(ValueError, match="token_counter returned"):
            tors.chunk_to_budget(self.TEXT, lambda _: returned, max_tokens=2)

    def test_huge_counts_are_value_errors(self) -> None:
        with pytest.raises(ValueError, match="unreasonably large"):
            tors.chunk_to_budget(self.TEXT, lambda _: 10**30, max_tokens=2)

    @pytest.mark.parametrize("returned", [1.5, "3", None, [2]])
    def test_non_int_counts_are_type_errors(self, returned: object) -> None:
        with pytest.raises(TypeError, match="token_counter must return an int"):
            tors.chunk_to_budget(self.TEXT, lambda _: returned, max_tokens=2)  # type: ignore[arg-type, return-value]

    def test_counter_exceptions_propagate_unchanged(self) -> None:
        class Boom(Exception):
            pass

        def counter(_: str) -> int:
            raise Boom("tokenizer down")

        with pytest.raises(Boom, match="tokenizer down"):
            tors.chunk_to_budget(self.TEXT, counter, max_tokens=2)

    def test_offsets_structure_is_validated(self) -> None:
        text = "One. Two."
        with pytest.raises(ValueError, match="token_offsets"):
            tors.chunk_to_offsets(text, [(0, 4), (3, 5)], max_tokens=2)  # overlapping
        with pytest.raises(ValueError, match="token_offsets"):
            tors.chunk_to_offsets(text, [(4, 4)], max_tokens=2)  # empty span
        with pytest.raises(ValueError, match="token_offsets"):
            tors.chunk_to_offsets(text, [(0, 99)], max_tokens=2)  # out of bounds
        with pytest.raises(ValueError, match="token_offsets"):
            tors.chunk_to_offsets(text, [(-1, 4)], max_tokens=2)  # negative
        with pytest.raises(ValueError, match="token_offsets"):
            tors.chunk_to_offsets(text, ["ab"], max_tokens=2)  # not a pair
        with pytest.raises(TypeError, match="token_offsets must be a sequence"):
            tors.chunk_to_offsets(text, 42, max_tokens=2)  # type: ignore[arg-type]


# ---- the base contract ------------------------------------------------------


class TestBaseContract:
    def test_text_fitting_the_budget_is_one_chunk(self) -> None:
        text = "One. Two. Three."
        assert tors.chunk_to_budget(text, word_counter, max_tokens=99) == [(0, 16)]
        assert tors.chunk_to_offsets(text, word_offsets(text), max_tokens=99) == [(0, 16)]

    def test_empty_text_is_no_chunks(self) -> None:
        assert tors.chunk_to_budget("", word_counter, max_tokens=5) == []
        assert tors.chunk_to_offsets("", [], max_tokens=5) == []

    def test_sentence_cuts_pack_greedily(self) -> None:
        # Each sentence's trailing space rides its own chunk's tail
        # (this packer does not trim; the budget measured the text as
        # it is).
        text = "One. Two. Three. Four."
        assert tors.chunk_to_budget(text, word_counter, max_tokens=2) == [
            (0, 10),
            (10, 22),
        ]
        assert [text[s:e] for s, e in tors.chunk_to_budget(text, word_counter, max_tokens=2)] == [
            "One. Two. ",
            "Three. Four.",
        ]

    def test_an_oversized_sentence_falls_back_to_word_boundaries(self) -> None:
        # One terminator-free sentence of six words: budget 2 words per
        # chunk must cut it at word boundaries, not emit one oversized
        # chunk.
        text = "aa bb cc dd ee ff"
        assert tors.chunk_to_budget(text, word_counter, max_tokens=2) == [
            (0, 6),
            (6, 12),
            (12, 17),
        ]

    def test_a_single_word_wider_than_the_budget_goes_out_whole(self) -> None:
        # The one budget exception: the word-fallback's finest boundary
        # cannot split a single word; it goes out whole rather than drop
        # content.
        text = "aaaa bb"
        chunks = tors.chunk_to_budget(text, word_counter, max_tokens=1)
        assert chunks == [(0, 5), (5, 7)]
        assert [text[s:e] for s, e in chunks] == ["aaaa ", "bb"]

    def test_cjk_and_multibyte_round_trip(self) -> None:
        # Codepoint offsets through multibyte characters: a byte-offset
        # chunker would panic slicing or land mid-character.
        text = "café 東京。 大阪。"
        for chunks in (
            tors.chunk_to_budget(text, word_counter, max_tokens=1),
            tors.chunk_to_budget(text, word_counter, max_tokens=2),
            tors.chunk_to_offsets(text, word_offsets(text), max_tokens=1),
        ):
            for start, end in chunks:
                assert text[start:end]  # slicing is char-boundary safe
                assert (start, end) != (end, start)
        joined = "".join(
            text[s:e] for s, e in tors.chunk_to_budget(text, word_counter, max_tokens=1)
        )
        assert joined == text

    def test_zwj_emoji_and_combining_marks_round_trip(self) -> None:
        text = "\U0001F469\u200D\U0001F52C says hi. a\u0301b. ok."
        chunks = tors.chunk_to_budget(text, word_counter, max_tokens=3)
        assert "".join(text[s:e] for s, e in chunks) == text

    def test_offsets_variant_counts_contained_spans(self) -> None:
        text = "ab cd ef"
        spans = [(0, 2), (3, 5), (6, 8)]
        assert tors.chunk_to_offsets(text, spans, max_tokens=2) == [(0, 6), (6, 8)]
        assert [text[s:e] for s, e in tors.chunk_to_offsets(text, spans, max_tokens=2)] == [
            "ab cd ",
            "ef",
        ]

    def test_offsets_variant_tolerates_untokenized_text(self) -> None:
        # Whitespace between tokens is untokenized gap (measures 0) and
        # still makes progress: offsets, not counts, anchor the walk.
        text = "  a  b  "
        spans = [(2, 3), (5, 6)]
        assert tors.chunk_to_offsets(text, spans, max_tokens=1) == [(0, 5), (5, 8)]
        assert [text[s:e] for s, e in tors.chunk_to_offsets(text, spans, max_tokens=1)] == [
            "  a  ",
            "b  ",
        ]

    def test_offsets_variant_straddling_token_is_never_double_counted(self) -> None:
        # A token span straddling a would-be boundary counts on the side
        # that fully contains it and on no other.
        text = "abcd ef"
        spans = [(0, 7)]  # one token covering everything
        assert tors.chunk_to_offsets(text, spans, max_tokens=7) == [(0, 7)]
        assert tors.chunk_to_offsets(text, spans, max_tokens=1) == [(0, 7)]  # goes out whole

    def test_determinism(self) -> None:
        text = "the cat sat on the mat today and the dog ran far away"
        for overlap in (0, 1, 2, 0.5):
            a = tors.chunk_to_budget(text, word_counter, max_tokens=4, overlap=overlap)
            b = tors.chunk_to_budget(text, word_counter, max_tokens=4, overlap=overlap)
            assert a == b
        spans = word_offsets(text)
        a = tors.chunk_to_offsets(text, spans, max_tokens=4, overlap=2)
        b = tors.chunk_to_offsets(text, spans, max_tokens=4, overlap=2)
        assert a == b


# ---- overlap ----------------------------------------------------------------


class TestOverlap:
    TEXT = "the cat sat on the mat today and the dog ran far away"

    def test_int_overlap_repeats_trailing_context(self) -> None:
        chunks = tors.chunk_to_budget(self.TEXT, word_counter, max_tokens=3, overlap=1)
        assert len(chunks) >= 2
        for (prev_start, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
            assert next_start > prev_start
            assert next_start < prev_end, "no actual overlap between consecutive chunks"
            shared = self.TEXT[next_start:prev_end]
            assert shared.strip(), "the shared region is real content"

    def test_float_overlap_is_a_ratio_of_the_budget(self) -> None:
        # 0.5 of a 4-token budget is 2 tokens of trailing context: the
        # same chunking an int overlap of 2 asks for.
        text = self.TEXT
        assert tors.chunk_to_budget(text, word_counter, max_tokens=4, overlap=0.5) == (
            tors.chunk_to_budget(text, word_counter, max_tokens=4, overlap=2)
        )
        assert tors.chunk_to_offsets(
            text, word_offsets(text), max_tokens=4, overlap=0.5
        ) == tors.chunk_to_offsets(text, word_offsets(text), max_tokens=4, overlap=2)

    def test_zero_ratio_is_zero_overlap(self) -> None:
        assert tors.chunk_to_budget(self.TEXT, word_counter, max_tokens=4, overlap=0.0) == (
            tors.chunk_to_budget(self.TEXT, word_counter, max_tokens=4, overlap=0)
        )

    def test_a_chunk_shorter_than_the_overlap_degrades_to_zero_overlap(self) -> None:
        # The decline-the-snap rule: a transition that cannot buy new
        # context degrades to zero overlap for that transition rather
        # than stall, loop, or emit a chunk contained in its predecessor.
        text = "aa bb. cc dd. ee ff."
        for budget in range(1, 5):
            for overlap in range(1, budget):
                chunks = tors.chunk_to_budget(
                    text, word_counter, max_tokens=budget, overlap=overlap
                )
                for (prev_start, prev_end), (start, end) in zip(
                    chunks, chunks[1:], strict=False
                ):
                    assert start > prev_start, "starts must advance"
                    assert end > prev_end, "ends must advance (no contained chunks)"

    def test_offsets_variant_overlap_shares_whole_tokens(self) -> None:
        text = "aa bb cc dd ee ff"
        spans = word_offsets(text)
        chunks = tors.chunk_to_offsets(text, spans, max_tokens=2, overlap=1)
        assert len(chunks) >= 2
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
            assert next_start < prev_end
            shared_tokens = [
                (s, e) for s, e in spans if s >= next_start and e <= prev_end
            ]
            assert shared_tokens, "the shared region holds at least one whole token"


# ---- the budget invariant under non-additive counters ------------------------


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


# ---- overlap semantics --------------------------------------------------------


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


# ---- callback adversarial -------------------------------------------------------


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
    correctly; the heartbeat cells below check the loop's accounting."""

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


# ---- offset / round-trip torture ------------------------------------------------


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


# ---- adversarial counters (Hypothesis) --------------------------------------


class TestAdversarialCounters:
    @settings(max_examples=200, deadline=None)
    @given(
        text=st.text(max_size=120),
        budget=st.integers(1, 12),
        overlap=st.integers(0, 11),
    )
    def test_non_monotone_counters_terminate_and_keep_the_contract(
        self, text: str, budget: int, overlap: int
    ) -> None:
        counter = _flare_counter(3, 1)
        overlap = min(overlap, budget - 1)
        chunks = tors.chunk_to_budget(text, counter, max_tokens=budget, overlap=overlap)
        assert len(chunks) <= max(len(text), 1), "chunk count bounded by the codepoint count"
        for start, end in chunks:
            assert 0 <= start < end <= len(text)

    @settings(max_examples=200, deadline=None)
    @given(
        text=st.text(
            alphabet=st.sampled_from("ab .\u3002\u4e01\u0301\U0001F469"), max_size=60
        ),
        budget=st.integers(1, 8),
        overlap=st.integers(0, 7),
    )
    def test_word_counter_contract_over_arbitrary_text(
        self, text: str, budget: int, overlap: int
    ) -> None:
        counter = lambda s: max(len(s.split()), 1)  # noqa: E731
        overlap = min(overlap, budget - 1)
        chunks = tors.chunk_to_budget(text, counter, max_tokens=budget, overlap=overlap)
        assert_full_contract(text, chunks, budget, counter)

    @settings(max_examples=150, deadline=None)
    @given(
        text=st.text(max_size=80),
        budget=st.integers(1, 10),
        overlap=st.integers(0, 9),
    )
    def test_char_counter_contract_over_arbitrary_text(
        self, text: str, budget: int, overlap: int
    ) -> None:
        overlap = min(overlap, budget - 1)
        chunks = tors.chunk_to_budget(text, char_counter, max_tokens=budget, overlap=overlap)
        assert_full_contract(text, chunks, budget, char_counter)

    @settings(max_examples=150, deadline=None)
    @given(
        text=st.text(max_size=80),
        budget=st.integers(1, 10),
        overlap=st.integers(0, 9),
    )
    def test_offsets_variant_contract_over_arbitrary_text(
        self, text: str, budget: int, overlap: int
    ) -> None:
        overlap = min(overlap, budget - 1)
        spans = word_offsets(text)
        chunks = tors.chunk_to_offsets(text, spans, max_tokens=budget, overlap=overlap)
        assert_full_contract(text, chunks, budget, char_counter, spans=spans)

    def test_a_small_text_always_fits_into_one_chunk(self) -> None:
        # When the whole text measures <= budget, exactly one chunk.
        texts = ["", "a", "a b c", "One. Two. Three.", "  spaces  "]
        for text in texts:
            for counter in (word_counter, char_counter):
                total = counter(text)
                if total == 0:
                    # Whitespace-only text under the word counter raises
                    # ValueError (a sentence measuring no tokens makes the
                    # budget contract meaningless, covered by its own test
                    # below).
                    continue
                chunks = tors.chunk_to_budget(text, counter, max_tokens=max(total, 1))
                assert len(chunks) == 1, (text, chunks)

    def test_whitespace_only_text_under_the_word_counter_is_a_value_error(self) -> None:
        """The zero-sentence contract: whitespace-only text is one sentence
        that IS a whitespace run, it measures 0 tokens under a word-count
        tokenizer, and the call raises the documented ValueError. The
        trigger is any zero-measure sentence, not just whitespace-only
        text: the blank-line shapes that put a whitespace-only sentence
        inside ordinary multi-paragraph text are pinned in
        test_sentences_differing_only_by_trailing_whitespace_runs."""
        with pytest.raises(ValueError, match="returned 0"):
            tors.chunk_to_budget("   ", word_counter, max_tokens=5)

    def test_liveness_under_an_all_max_counter(self) -> None:
        # Every measurement returns a huge-but-legal count: every chunk
        # is a forced single segment; progress is anchored to offsets,
        # never counts, so this terminates.
        text = "a b c d e f g h " * 20
        chunks = tors.chunk_to_budget(text, lambda _: 1_000_000, max_tokens=10)
        assert len(chunks) <= len(text)
        for start, end in chunks:
            assert end > start


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


# ---- the 4 GiB guard ------------------------------------------------------------
#
# Read from src/chunk_budget_impl.rs::grid_overflow (the guard's own logic,
# exercised without a 4 GiB-class allocation in the default run):
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
# The heavy cells below are the real end-to-end pins (refusal: 0.00s, zero
# counter calls; exact-bound acceptance: ~3 min, one chunk [(0, len)]);
# they are env-gated so no CI lane ever pays for a 4 GiB string.


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


# ---- overlap content semantics ---------------------------------------------------


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
    """Pinned behavior: a counter that certifies a whitespace run as >= 1
    token (max(1, len//n) floors EVERYTHING to 1) lets the walk-back
    accept a boundary whose repeated region is whitespace-only: the docs'
    'genuine shared content' is counter-relative (a counter that
    certifies whitespace as a token makes the overlap a whitespace run),
    and the region still measures >= overlap per that counter, so the
    letter of the certification claim holds."""
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
                # the documented zero-sentence contract (a zero-measure
                # sentence under the word counter); the offsets twin still
                # runs
                assert "returned 0" in str(err), err
                continue
            spans = None
        else:
            spans = word_offsets(text)
            chunks = tors.chunk_to_offsets(text, spans, max_tokens=4)
        assert_full_contract(text, chunks, 4, word_counter, spans=spans)
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


# ---- token_counter shapes ---------------------------------------------------------


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
    """ATTACK: the docs say 'callable'; a callable class instance, a
    bound method, and a functools.partial must all be accepted and
    measure identically."""
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
    assert_full_contract(text2, chunks, 4, lambda s: 5)
    with pytest.raises(ValueError, match="returned 0"):
        tors.chunk_to_budget(text, lambda s: 0, max_tokens=3)


def test_growing_cost_counter_keeps_gil_held_time_tracking_the_callbacks() -> None:
    """ATTACK on 'held time tracks the callbacks': a stateful counter
    whose per-call cost GROWS with the call count (call N busy-holds the
    GIL for N * 0.3ms). The docs' claim is that the GIL is held only
    while the counter runs (plus O(chunk) argument construction), so the
    worst heartbeat gap must track the (growing) callbacks, never the
    whole call: the heartbeat cells' ratio budget must hold."""
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
    """ATTACK: KeyboardInterrupt / SystemExit are BaseExceptions, not
    Exceptions -- the captured-err re-raise path must propagate them
    UNCHANGED (same class, same message), and a follow-up call with a
    healthy counter must behave normally (no poisoned state)."""

    def boom(s: str) -> int:
        raise exc_type("from the counter")

    with pytest.raises(exc_type, match="from the counter"):
        tors.chunk_to_budget("One. Two. Three. Four.", boom, max_tokens=2)
    chunks = tors.chunk_to_budget("One. Two. Three. Four.", word_counter, max_tokens=2)
    assert_full_contract("One. Two. Three. Four.", chunks, 2, word_counter)


# ---- sentence-boundary shapes ----------------------------------------------------


def test_no_terminators_the_whole_text_is_the_word_fallback_path() -> None:
    """ATTACK: text with NO sentence terminators at all is one giant UAX
    #29 sentence; the entire packing runs through the word-fallback
    path. Coverage and the budget invariant must hold for every budget,
    and the chunk count must be the segment count, not a stall."""
    text = " ".join(f"word{i}" for i in range(30))
    assert tors.sentence_bounds(text) == [(0, len(text))]
    for budget in (1, 2, 3, 7):
        chunks = tors.chunk_to_budget(text, word_counter, max_tokens=budget)
        assert_full_contract(text, chunks, budget, word_counter)
        assert len(chunks) <= len(text)


def test_cjk_terminators_cut_sentences_and_the_packer_agrees() -> None:
    """ATTACK: terminators that are themselves CJK (``。！？``) must be
    sentence cuts (UAX #29 SB-series on terminal punctuation), the
    packer's sentence segments must be exactly tors.sentence_bounds, and
    chunk boundaries must land on those bounds (plus word bounds for an
    oversized CJK sentence's fallback)."""
    text = "東京。大阪！京都？奈良。"
    assert tors.sentence_bounds(text) == [(0, 3), (3, 6), (6, 9), (9, 12)]
    sentence_starts = {s for s, _ in tors.sentence_bounds(text)}
    for budget in (2, 3, 5):
        for overlap in range(budget):
            half = lambda s: max(1, len(s) // 2)  # noqa: E731
            chunks = tors.chunk_to_budget(text, half, max_tokens=budget, overlap=overlap)
            assert_full_contract(text, chunks, budget, half)
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
        assert_full_contract(abbrev_text, chunks, budget, word_counter)
        for s, _ in chunks[1:]:
            assert s in legal, f"packer cut at {s} is not a segment bound: {chunks}"
        assert chunks[-1][1] in ends or chunks[-1][1] == len(abbrev_text)


def test_sentences_differing_only_by_trailing_whitespace_runs() -> None:
    """ATTACK: whitespace runs between sentences ride the preceding
    sentence ('a. \\n' ends the sentence) UNTIL a run is long enough to
    become its OWN sentence (a blank line: the second \\n of '\\n\\n').
    Pin both shapes. The own-sentence run makes the CANONICAL word-count
    counter raise ValueError on perfectly ordinary multi-paragraph text:
    'Paragraph one.\\n\\nParagraph two.' and even a trailing blank line
    raise, because UAX #29 makes the whitespace run a sentence of its own
    and a sentence measuring no tokens makes the budget contract
    meaningless. The trigger is the zero-measure sentence, not
    whitespace-only text; the offsets twin is unaffected (it tolerates
    0-measure sentences) and is pinned below."""
    # single trailing whitespace ride along: fine
    for ok_text in ("a. \n b.", "Paragraph one. Paragraph two.\n", "a.  b.  c."):
        chunks = tors.chunk_to_budget(ok_text, word_counter, max_tokens=1)
        assert_full_contract(ok_text, chunks, 1, word_counter)
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
    assert_full_contract("Paragraph one.\n\nParagraph two.", chunks, 8, word_counter, spans=spans)


# ---- the offsets twin: realistic HuggingFace-style shapes ------------------------


def test_hf_style_offsets_with_special_tokens_raise() -> None:
    """ATTACK: a real HuggingFace ``Encoding.offsets`` from
    ``tokenizer(text)`` -- add_special_tokens defaulting to True --
    carries ZERO-WIDTH (0, 0) offsets for the [CLS]/[SEP] special tokens.
    chunk_to_offsets rejects them (0 <= start < end), so the DEFAULT HF
    call is not acceptable verbatim: ``Encoding.offsets`` matches the
    expected shape only after filtering zero-width spans. Pinned: the
    rejection, and the working filter spelling."""
    with pytest.raises(ValueError, match="out of bounds"):
        tors.chunk_to_offsets("Hi there", [(0, 0), (0, 2), (3, 8)], max_tokens=2)
    # the workaround: drop the zero-width special tokens
    assert tors.chunk_to_offsets("Hi there", [(0, 2), (3, 8)], max_tokens=2) == [(0, 8)]


def test_realistic_hf_offsets_with_whitespace_and_newline_gaps() -> None:
    """ATTACK: realistic fast-tokenizer offsets -- whitespace excluded
    from spans (gaps at every space), gaps at newlines, multi-codepoint
    tokens. Packing + full contract must hold, and the offsets spelling
    must agree with the counter spelling when BOTH measure the same
    model (the UAX #29 span model: uax_counter is word_offsets as a
    callable) -- the docs' 'same contract' claim. The text uses single
    newlines: the blank-line shape that makes the counter twin raise is
    pinned in test_sentences_differing_only_by_trailing_whitespace_runs."""
    text = "Hello world!\nThe quick brown fox jumps over the lazy dog."
    spans = word_offsets(text)  # whitespace untokenized: gaps, incl. newlines
    for budget in (2, 3, 5, 8):
        for overlap in range(budget):
            a = tors.chunk_to_offsets(text, spans, max_tokens=budget, overlap=overlap)
            assert_full_contract(text, a, budget, uax_counter, spans=spans)
            b = tors.chunk_to_budget(text, uax_counter, max_tokens=budget, overlap=overlap)
            assert a == b, f"twin divergence at {budget}/{overlap}: {a} vs {b}"


def test_whitespace_only_spans_are_accepted_spans() -> None:
    """ATTACK: spans CONTAINING only whitespace are legal (the validation
    is positional, not content-based) -- contrast the zero-width
    rejection. A whitespace-only span measures 1 where contained."""
    chunks = tors.chunk_to_offsets("a b", [(0, 1), (1, 2)], max_tokens=2)
    assert chunks == [(0, 3)]


def test_unspanned_tail_rides_the_last_chunk_never_its_own() -> None:
    """ATTACK: token offsets that stop short of the text's last codepoints
    (a trailing gap -- HF offsets for trailing punctuation the tokenizer
    dropped, or a truncated span list). PINNED SEMANTICS: the tail is
    never dropped and never a chunk of its own (there IS no counter for
    unspanned text): it rides the SENTENCE containing it -- sentence_bounds
    tiles the whole text -- whose chunk still ends at len(text) per the
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
    assert_full_contract(long_text, chunks3, 1, word_counter, spans=[(0, 2), (502, 504)])
    assert chunks3[-1][1] == len(long_text)


# ---- GIL claim audit (hostile heartbeat cells) -----------------------------------
# The measurement is tests/loop_harness.py's shared heartbeat_gap_and_wall
# (the one copy of the 10ms heartbeat / worst-gap reduction the suite's
# GIL cells repeat); the budgets below are this file's own.


def _busy_at_3x(s: str) -> int:
    """The heartbeat cell's counter shape: GIL-held CPU work budgeted at
    3 * sys.getswitchinterval() per call -- derived at RUNTIME, which is
    the thing under test (a hardcoded-ms budget would stop straddling
    the interval when the interval is monkeypatched)."""
    deadline = time.monotonic() + 3 * sys.getswitchinterval()
    n = i = 0
    while time.monotonic() < deadline:
        for _ in range(1_000):
            i += 1
            n += i * i
    return max(len(s.split()), 1)


_TEXT = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. Eleven. Twelve."


def test_gil_heartbeat_survives_a_counter_that_releases_the_gil() -> None:
    """A counter that itself releases the GIL (time.sleep per call): the
    worker's handoffs must still interleave; the worst heartbeat gap
    stays a small fraction of the wall, never the whole call."""

    def sleepy(s: str) -> int:
        time.sleep(0.002)
        return max(len(s.split()), 1)

    text = "One. " * 200
    worst, wall = asyncio.run(
        heartbeat_gap_and_wall(
            lambda: asyncio.to_thread(tors.chunk_to_budget, text, sleepy, max_tokens=2)
        )
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
        heartbeat_gap_and_wall(
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
        heartbeat_gap_and_wall(
            lambda: asyncio.to_thread(tors.chunk_to_offsets, corpus, spans, max_tokens=200)
        )
    )
    assert worst < 0.60 * wall or worst < 0.100, (
        f"loop blocked {worst * 1000:.0f}ms of {wall * 1000:.0f}ms "
        f"(ratio {worst / wall:.2f})"
    )


# The slow-counter heartbeat claim, and where the schedulable band sits.
# test_gil_release.py::test_chunk_to_budget_slow_counter_blocks_only_for_
# its_callbacks pins "the worst gap tracks the callbacks themselves, never
# the whole call" with a busy counter time-budgeted at ~3x the GIL switch
# interval per call, so every callback stays above sys.getswitchinterval()
# (0.005s) on any interpreter (3.10-3.14). The mechanism: with a
# callback-dominated wall on a small text (microsecond detach windows), a
# callback shorter than the switch interval starves the event loop for the
# WHOLE call (the worker drops and re-acquires the GIL faster than the
# woken loop thread can take it, and gil_drop_request only fires for
# callbacks that straddle the interval), while callbacks of ~8ms and up
# measure a clean 3-6% worst gap. The Rust code's GIL claim itself holds
# (detach-between-callbacks is real; the 8/16ms cells and the
# fast-counter cell all pass): the loop is schedulable BETWEEN callbacks
# for callback-dominated calls when each callback exceeds the switch
# interval or the native windows between them are substantial, and the
# sub-switch-interval starvation is the honest boundary documented on
# every surface (docs/api.md, the .pyi, tors/aio.py, the binding
# docstrings). The green cell below pins the qualified claim's own
# boundary: super-interval callbacks on a small text are schedulable
# between. The sub-interval shape is NOT asserted in either direction
# (starvation is a timing property of the interpreter, not a contract);
# it lives in the docs.


def test_loop_schedulable_between_callbacks_above_the_switch_interval() -> None:
    """The qualified claim's own boundary. A GIL-held callback
    time-budgeted at ~3x ``sys.getswitchinterval()`` (so it stays above
    the interval on every interpreter 3.10-3.14) straddles the interval
    and fires ``gil_drop_request``'s fair handoff, so the loop stays
    schedulable between callbacks on a SMALL text too: the worst gap
    tracks the callbacks, never the whole call. The sub-switch-interval
    caveat (a fast GIL-held callback on a small text CAN starve the loop
    for the whole call) is the documented boundary (docs/api.md, the
    ``.pyi``, and the binding docstrings), deliberately not asserted
    here in either direction: it is a timing property of the
    interpreter, not a contract."""

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
        heartbeat_gap_and_wall(
            lambda: asyncio.to_thread(tors.chunk_to_budget, text, busy, max_tokens=2, overlap=1)
        )
    )
    assert worst < 0.30 * wall or worst < 0.100, (
        f"loop starved: blocked {worst * 1000:.0f}ms of {wall * 1000:.0f}ms "
        f"({worst / wall:.0%}); callbacks held the GIL for ~3x the switch "
        f"interval each, so gil_drop_request's fair handoff should have "
        f"scheduled the loop between them"
    )


def test_heartbeat_cell_is_deterministic_across_20_runs() -> None:
    """ATTACK: the heartbeat cell budgets each callback at 3x
    ``sys.getswitchinterval()``, above the interval. Determinism audit:
    20 consecutive runs; EVERY run must meet the ratio budget (worst gap
    tracks the ~3x callbacks, never the whole call). One flaky run out
    of 20 is a broken pin, not a pass."""
    runs = 20
    for k in range(runs):
        worst, wall = asyncio.run(
            heartbeat_gap_and_wall(
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
def test_heartbeat_budget_is_derived_from_switchinterval_at_runtime(interval: float) -> None:
    """ATTACK: the 3x multiplier must be computed from
    sys.getswitchinterval() AT RUNTIME. Monkeypatch the interval to 1ms
    and 50ms and re-run the heartbeat cell: it must still pass. If the
    3x were hardcoded in milliseconds (a 15ms constant), the 50ms run
    would put every callback far BELOW the interval -- the exact
    sub-interval starvation shape the cell claims to be above -- and
    fail here."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(interval)
    try:
        worst, wall = asyncio.run(
            heartbeat_gap_and_wall(
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


# ---- aio twins --------------------------------------------------------------


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


# ---- the docs' examples (pinned in test_docs_examples.py too) ----------------


class TestDocsExample:
    def test_readme_style_example(self) -> None:
        text = "One. Two. Three. Four."
        assert tors.chunk_to_budget(text, word_counter, max_tokens=2) == [
            (0, 10),
            (10, 22),
        ]
        assert tors.chunk_to_budget(text, word_counter, max_tokens=2, overlap=1) == [
            (0, 10),
            (5, 17),
            (10, 22),
        ]

    def test_api_md_examples_recomputed(self) -> None:
        """The docs/api.md chunk-budget block, recomputed verbatim (an
        independent pass over test_docs_examples.py's pins)."""
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
        spans = word_offsets(text)
        assert tors.chunk_to_offsets(text, spans, max_tokens=4, overlap=2) == [
            (0, 10),
            (5, 17),
            (10, 22),
        ]


# ---- scaling + memory -------------------------------------------------------


class TestScaling:
    @pytest.mark.timing
    def test_offsets_packing_stays_linear_in_the_text_size(self) -> None:
        def wall(text: str) -> float:
            spans = word_offsets(text)
            best = float("inf")
            for _ in range(5):
                started = time.monotonic()
                tors.chunk_to_offsets(text, spans, max_tokens=200)
                best = min(best, time.monotonic() - started)
            return best * 1000

        small = "one two three. " * 20_000  # ~300k chars
        large = "one two three. " * 80_000  # ~1.2M chars
        ratio = wall(large) / wall(small)
        # 4x the text, linear packing ~4x the wall (the pin pattern's
        # two-size core); a quadratic pass would measure ~16x. The gate
        # is 10.0: ~2.5x above the linear band, far below quadratic.
        assert ratio < 10.0, f"growth ratio {ratio:.2f} per 4x is superlinear"


class TestMemoryGuard:
    def test_offsets_packing_peak_memory_stays_bounded(self) -> None:
        # The amplification lane runs in a subprocess and reads VmHWM
        # (peak RSS since start), the test_memory_spike_guards.py
        # discipline: a packing pass that accumulated per-chunk state
        # superlinearly would show peak RSS far past the input's own
        # footprint.
        import subprocess

        code = (
            "import tors, resource\n"
            "text = ('the quick brown fox jumps over the lazy dog. ' * 30_000)\n"
            "spans = [(s, e) for s, e in tors.word_bounds(text) if text[s:e].strip()]\n"
            "chunks = tors.chunk_to_offsets(text, spans, max_tokens=200)\n"
            "assert chunks\n"
            "hwm = 0\n"
            "with open('/proc/self/status') as status:\n"
            "    for line in status:\n"
            "        if line.startswith('VmHWM:'):\n"
            "            hwm = int(line.split()[1])\n"
            "print(f'RESULT|{hwm}')\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert done.returncode == 0, done.stderr
        peak_kib = int(done.stdout.strip().split("|")[1])
        # The text is ~1.6 MB; the spans list ~1.3M pairs (~30 MB as
        # Python tuples). A superlinear accumulator would blow well past
        # 6x that working set.
        assert peak_kib < 6 * 60_000, f"peak RSS {peak_kib} KiB"
