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
  honestly NOT GIL-free (its counter is Python) and the cell here pins
  what IS true: the per-callback GIL handoffs keep the loop
  schedulable between callbacks when hopped to a thread. Both cells live
  in ``tests/test_gil_release.py`` with the family's harness.
"""

from __future__ import annotations

from time import monotonic

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors

# ---- helpers ----------------------------------------------------------------


def word_counter(text: str) -> int:
    """The docs' counter: ``len(text.split())``, no third-party tokenizer."""
    return len(text.split())


def char_counter(text: str) -> int:
    """A codepoint counter: every codepoint is one token."""
    return len(text)


def word_offsets(text: str) -> list[tuple[int, int]]:
    """Token spans aligned to the word counter, via tors's own UAX #29
    word segmentation filtered to non-whitespace segments: the realistic
    tokenizer-offsets shape for ``chunk_to_offsets`` (each span is one
    token, whitespace between them is untokenized gap)."""
    return [(s, e) for s, e in tors.word_bounds(text) if text[s:e].strip()]


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


# ---- adversarial counters (Hypothesis) --------------------------------------


def _flare_counter(high: int, low: int):
    """A deterministic NON-MONOTONE counter: its answer depends only on
    the span's first codepoint's parity, not its length: the shape that
    breaks sum-based or binary-search packers. Positive always (a 0
    return is a documented ValueError, not an adversarial shape)."""

    def counter(text: str) -> int:
        first = ord(text[0]) if text else 1
        return high if first % 3 == 0 else low

    return counter


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
        self._assert_full_contract(
            text, lambda s: max(len(s.split()), 1), budget, min(overlap, budget - 1)
        )

    @settings(max_examples=150, deadline=None)
    @given(
        text=st.text(max_size=80),
        budget=st.integers(1, 10),
        overlap=st.integers(0, 9),
    )
    def test_char_counter_contract_over_arbitrary_text(
        self, text: str, budget: int, overlap: int
    ) -> None:
        self._assert_full_contract(text, char_counter, budget, min(overlap, budget - 1))

    @settings(max_examples=150, deadline=None)
    @given(
        text=st.text(max_size=80),
        budget=st.integers(1, 10),
        overlap=st.integers(0, 9),
    )
    def test_offsets_variant_contract_over_arbitrary_text(
        self, text: str, budget: int, overlap: int
    ) -> None:
        spans = word_offsets(text)
        self._assert_full_contract(
            text, char_counter, budget, min(overlap, budget - 1), spans=spans
        )

    def _assert_full_contract(
        self,
        text: str,
        counter,
        budget: int,
        overlap: int,
        spans: list[tuple[int, int]] | None = None,
    ) -> None:
        if spans is None:
            chunks = tors.chunk_to_budget(text, counter, max_tokens=budget, overlap=overlap)
        else:
            chunks = tors.chunk_to_offsets(text, spans, max_tokens=budget, overlap=overlap)
        if not text:
            assert chunks == []
            return
        assert chunks, "non-empty text yields at least one chunk"
        prev_start = -1
        prev_end = 0
        for i, (start, end) in enumerate(chunks):
            assert 0 <= start < end <= len(text), f"bounds: {chunks}"
            if i == 0:
                assert start == 0
            else:
                assert start > prev_start, f"starts advance: {chunks}"
                assert end > prev_end, f"ends advance: {chunks}"
            if spans is not None:
                # The budget invariant under the offsets spelling's own
                # measurement: fully-contained token pairs.
                measured = sum(1 for s, e in spans if s >= start and e <= end)
            else:
                measured = counter(text[start:end])
            if measured > budget:
                # The single-oversized-segment exception: the chunk must
                # be exactly one word-bound unit (nothing finer to cut).
                assert len(tors.word_bounds(text[start:end])) <= 1, (
                    f"budget exceeded without the single-segment exception: {chunks} {text!r}"
                )
            prev_start, prev_end = start, end
        assert prev_end == len(text), "covers to the end"
        if overlap == 0 and spans is None:
            assert "".join(text[s:e] for s, e in chunks) == text, "lossless join-back"

    def test_a_small_text_always_fits_into_one_chunk(self) -> None:
        # When the whole text measures <= budget, exactly one chunk.
        texts = ["", "a", "a b c", "One. Two. Three.", "  spaces  "]
        for text in texts:
            for counter in (word_counter, char_counter):
                total = counter(text)
                if total == 0:
                    # Whitespace-only text under the word counter raises
                    # ValueError (the documented zero-sentence contract,
                    # covered by its own test below).
                    continue
                chunks = tors.chunk_to_budget(text, counter, max_tokens=max(total, 1))
                assert len(chunks) == 1, (text, chunks)

    def test_whitespace_only_text_under_the_word_counter_is_a_value_error(self) -> None:
        # The documented zero-sentence contract: a whitespace-only text
        # measures 0 tokens under a word counter (a sentence with no
        # tokens makes the budget contract meaningless).
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


# ---- scaling + memory -------------------------------------------------------


class TestScaling:
    @pytest.mark.timing
    def test_offsets_packing_stays_linear_in_the_text_size(self) -> None:
        def wall(text: str) -> float:
            spans = word_offsets(text)
            best = float("inf")
            for _ in range(5):
                started = monotonic()
                tors.chunk_to_offsets(text, spans, max_tokens=200)
                best = min(best, monotonic() - started)
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
        import sys

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


# ---- the docs' example (pinned in test_docs_examples.py too) ----------------


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
