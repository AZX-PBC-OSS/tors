"""Contract gate for the text-chunking family: ``tors.chunk_text`` (with its
``overlap`` parameter) and the segment-count-windowed ``tors.chunk_by_words``
/ ``tors.chunk_by_sentences`` / ``tors.chunk_by_paragraphs`` /
``tors.chunk_by_lines``.

``chunk_text`` with ``overlap=0`` (the default) is a LOSSLESS COVERING
PARTITION: chunks are non-empty, contiguous, strictly increasing, cover the
whole text, and joining them reproduces the input exactly; the properties
pinned below must hold for every case, matching the Rust-side contract this
was unchanged from. ``overlap > 0`` trades that lossless-join guarantee for
genuine overlap between consecutive chunks (the RAG-retrieval shape: a fact
split across a cut is still whole in at least one chunk), snapped to a real
word/sentence boundary, never mid-word/mid-sentence.

``chunk_by_words``/``chunk_by_sentences`` measure chunks in UAX #29 segment
COUNT rather than a character budget: each chunk spans exactly N consecutive
word/sentence segments (the last chunk may hold fewer), with Y segments of
overlap repeated at the start of the next chunk.

``chunk_by_lines`` is the line-count sibling of that windowing (the
transcript/log shape): a break is a ``\n``, a lone ``\r``, or a ``\r\n`` pair
counted as ONE unit, and a line counts only when it carries content, so
blank lines ride along inside a chunk's span rather than counting.

Forward progress (no infinite loop) is the one invariant enforced most
aggressively here: every property test below either bounds the chunk count
by the input length, or asserts strictly-increasing chunk starts directly,
a regression to a stalled/looping start fails fast rather than hanging.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import (
    chunk_by_lines,
    chunk_by_lines_iter,
    chunk_by_paragraphs,
    chunk_by_paragraphs_iter,
    chunk_by_sentences,
    chunk_by_sentences_iter,
    chunk_by_words,
    chunk_by_words_iter,
    chunk_text,
    chunk_text_iter,
    grapheme_count,
)

# ---------------------------------------------------------------------------
# chunk_text: overlap=0, the original lossless-partition contract
# ---------------------------------------------------------------------------


class TestChunkTextNoOverlap:
    def test_empty_text_is_no_chunks(self) -> None:
        assert chunk_text("", 5) == []

    def test_text_within_budget_is_one_chunk(self) -> None:
        assert chunk_text("hello", 100) == [(0, 5)]

    def test_worked_example(self) -> None:
        # word_bounds("cats are cute and cats are fun") lands a budget-12 cut
        # at "cats are" (8) with the trailing space trimmed onto the next
        # chunk's head.
        text = "cats are cute and cats are fun"
        chunks = chunk_text(text, 12)
        assert chunks == [(0, 8), (8, 17), (17, 26), (26, 30)]
        joined = "".join(text[a:b] for a, b in chunks)
        assert joined == text

    def test_sentence_boundary(self) -> None:
        text = "One. Two. Three."
        chunks = chunk_text(text, 10, boundary="sentence")
        assert chunks == [(0, 9), (9, 16)]

    def test_hard_cut_when_no_boundary_fits(self) -> None:
        text = "Supercalifragilisticexpialidocious"
        chunks = chunk_text(text, 10)
        assert chunks == [(0, 10), (10, 20), (20, 30), (30, 34)]
        for a, b in chunks:
            assert b - a <= 10

    @given(
        text=st.text(alphabet="ab .", max_size=40),
        max_chars=st.integers(min_value=1, max_value=10),
        boundary=st.sampled_from(["word", "sentence"]),
    )
    @settings(max_examples=300)
    def test_covering_partition_contract(self, text: str, max_chars: int, boundary: str) -> None:
        chunks = chunk_text(text, max_chars, boundary=boundary)
        prev_end = 0
        for a, b in chunks:
            assert a == prev_end, "contiguity broke"
            assert b > a, "empty chunk"
            assert b - a <= max_chars, "budget exceeded"
            prev_end = b
        assert prev_end == len(text), "coverage broke"
        joined = "".join(text[a:b] for a, b in chunks)
        assert joined == text, "join-back broke"

    def test_never_splits_a_thai_sara_am_cluster_across_two_chunks(self) -> None:
        # "0" + SARA AM (U+0E33) is ONE grapheme cluster, but the UAX #29
        # word segmenter scores it as TWO word segments; the same edge
        # tors.truncate_to_bounds guards against. A boundary-aware chunker
        # must never split the base character from its combining mark
        # into two different chunks, even when that means backing the cut
        # off before the character budget is fully used.
        text = "ab 0ำ cd"
        chunks = chunk_text(text, 4)
        pieces = [text[a:b] for a, b in chunks]
        assert "".join(pieces) == text
        cluster = "0ำ"
        containing = [p for p in pieces if cluster in p]
        assert len(containing) == 1, f"cluster split across chunks: {pieces!r}"
        for p in pieces:
            assert p != "ำ", f"a lone combining mark escaped as its own chunk: {pieces!r}"


# ---------------------------------------------------------------------------
# chunk_text: overlap > 0
# ---------------------------------------------------------------------------


class TestChunkTextOverlap:
    def test_worked_example(self) -> None:
        text = "cats are cute and cats are fun"
        chunks = chunk_text(text, 12, overlap=3)
        assert chunks[0] == (0, 8)
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
            assert next_start < prev_end, "no actual overlap"

    def test_overlap_zero_matches_no_overlap_path_exactly(self) -> None:
        text = "cats are cute and cats are fun"
        for max_chars in range(1, 15):
            for boundary in ("word", "sentence"):
                assert chunk_text(text, max_chars, overlap=0, boundary=boundary) == chunk_text(
                    text, max_chars, boundary=boundary
                )

    def test_overlap_equal_to_or_above_max_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be < max_chars"):
            chunk_text("hello world", 5, overlap=5)
        with pytest.raises(ValueError, match="overlap must be < max_chars"):
            chunk_text("hello world", 5, overlap=6)

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            chunk_text("hello world", 5, overlap=-1)

    def test_shared_content_between_consecutive_chunks_is_real(self) -> None:
        text = "the cat sat on the mat today near the door"
        chunks = chunk_text(text, 12, overlap=4)
        assert len(chunks) >= 2
        for (prev_start, prev_end), (next_start, _next_end) in zip(
            chunks, chunks[1:], strict=False
        ):
            assert next_start > prev_start, "no forward progress"
            assert next_start < prev_end, "overlap produced no actual overlap"
            shared = text[next_start:prev_end]
            assert shared == text[next_start:prev_end]
            assert shared, "shared span must be non-empty"

    @given(
        text=st.text(alphabet="ab .", max_size=40),
        max_chars=st.integers(min_value=2, max_value=10),
        boundary=st.sampled_from(["word", "sentence"]),
        data=st.data(),
    )
    @settings(max_examples=300)
    def test_forward_progress_and_per_chunk_budget(
        self, text: str, max_chars: int, boundary: str, data: st.DataObject
    ) -> None:
        overlap = data.draw(st.integers(min_value=1, max_value=max_chars - 1))
        chunks = chunk_text(text, max_chars, overlap=overlap, boundary=boundary)
        # Forward-progress-as-a-hard-bound: strictly more chunks than the
        # text has codepoints would mean a stalled/looping start; this
        # fails fast instead of hanging the test runner.
        assert len(chunks) <= len(text) + 1
        prev_start = -1
        for a, b in chunks:
            assert b > a
            assert b - a <= max_chars
            assert a > prev_start, "no forward progress"
            prev_start = a

    def test_short_trailing_chunk_degrades_overlap_rather_than_stalling(self) -> None:
        # A short final chunk (shorter than the requested overlap) forces
        # the documented snap-collapse fallback: the transition into it
        # loses its overlap rather than violating forward progress.
        text = "aaaaaaaaaa b"  # a 10-char run, a space, one more char
        chunks = chunk_text(text, 10, overlap=8)
        prev_start = -1
        for a, _b in chunks:
            assert a > prev_start
            prev_start = a
        assert chunks[-1][1] == len(text)


# ---------------------------------------------------------------------------
# chunk_by_words
# ---------------------------------------------------------------------------


class TestChunkByWords:
    def test_empty_text_is_no_chunks(self) -> None:
        assert chunk_by_words("", 3) == []

    def test_worked_example_no_overlap(self) -> None:
        # "the cat sat on the mat" -> 6 real word tokens, 2 per chunk ->
        # 3 chunks. Chunks are NOT necessarily contiguous (the space
        # between "cat" and "sat" belongs to neither chunk), so chunk_by_words
        # makes no covering-partition claim, unlike chunk_text.
        text = "the cat sat on the mat"
        chunks = chunk_by_words(text, 2)
        assert [text[a:b] for a, b in chunks] == ["the cat", "sat on", "the mat"]

    def test_counts_real_word_tokens_not_raw_word_bounds_segments(self) -> None:
        # The regression this pins: word_bounds gives an inter-word space
        # run its OWN segment, so a naive "group N raw segments" reading
        # of words_per_chunk would silently deliver roughly HALF as many
        # real words per chunk as requested on ordinary prose.
        text = "one two three four five six seven"
        chunks = chunk_by_words(text, 3)
        assert [text[a:b] for a, b in chunks] == [
            "one two three",
            "four five six",
            "seven",
        ]

    def test_worked_example_with_overlap(self) -> None:
        text = "one two three four five six seven"
        chunks = chunk_by_words(text, 3, overlap=1)
        assert len(chunks) >= 2
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
            assert next_start < prev_end

    def test_fewer_words_than_per_chunk_is_one_chunk(self) -> None:
        text = "hi there"
        chunks = chunk_by_words(text, 100)
        assert chunks == [(0, len(text))]

    def test_last_chunk_may_be_partial(self) -> None:
        text = "alpha beta gamma"
        chunks = chunk_by_words(text, 2)
        assert chunks[-1][1] == len(text)

    def test_words_per_chunk_zero_or_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="words_per_chunk must be >= 1"):
            chunk_by_words("hello world", 0)
        with pytest.raises(ValueError, match="words_per_chunk must be >= 1"):
            chunk_by_words("hello world", -1)

    def test_overlap_equal_to_or_above_words_per_chunk_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be < words_per_chunk"):
            chunk_by_words("one two three", 2, overlap=2)

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            chunk_by_words("one two three", 2, overlap=-1)

    @given(
        text=st.text(alphabet="ab ", max_size=40),
        per_chunk=st.integers(min_value=1, max_value=6),
        data=st.data(),
    )
    @settings(max_examples=200)
    def test_forward_progress_over_a_battery(
        self, text: str, per_chunk: int, data: st.DataObject
    ) -> None:
        overlap = data.draw(st.integers(min_value=0, max_value=per_chunk - 1))
        chunks = chunk_by_words(text, per_chunk, overlap=overlap)
        assert len(chunks) <= len(text) + 1
        prev_start = -1
        for a, b in chunks:
            assert b > a
            assert a > prev_start
            assert b <= len(text)
            prev_start = a
        # NOT a covering-partition contract (unlike chunk_text): trailing
        # whitespace, or text with no word tokens at all, means the last
        # chunk's end can legitimately fall short of len(text), or there
        # can be no chunks at all; only forward progress and in-bounds
        # spans are guaranteed here.

    def test_never_splits_a_thai_sara_am_cluster_across_two_words(self) -> None:
        # Raw word segmentation of "x0ำy0ำz" splits each "0" + SARA AM
        # (U+0E33) combining sequence into a base-segment and a lone
        # combining-mark segment (neither is whitespace), so the
        # word-token filter alone would NOT catch this: without the
        # grapheme-cluster merge, chunk_by_words(text, 1) would silently
        # return a chunk containing only the bare combining mark.
        text = "x0ำy0ำz"
        chunks = chunk_by_words(text, 1)
        pieces = [text[a:b] for a, b in chunks]
        assert pieces == ["x0ำ", "y0ำ", "z"]
        assert "ำ" not in pieces

    def test_offsets_are_codepoint_offsets_on_astral_text(self) -> None:
        # Each emoji is one word token: a single Python codepoint but
        # four UTF-8 bytes, so a byte-offset regression would read
        # (0, 4), (5, 9), (10, 14) here instead of the codepoint offsets
        # below. Slicing the chunks back out must return whole emoji,
        # never a torn surrogate half.
        text = "\U0001f600 \U0001f601 \U0001f602"  # 5 codepoints, 14 UTF-8 bytes
        assert chunk_by_words(text, 1) == [(0, 1), (2, 3), (4, 5)]
        assert [text[a:b] for a, b in chunk_by_words(text, 1)] == [
            "\U0001f600",
            "\U0001f601",
            "\U0001f602",
        ]
        assert chunk_by_words(text, 2) == [(0, 3), (4, 5)]


# ---------------------------------------------------------------------------
# chunk_by_sentences
# ---------------------------------------------------------------------------


class TestChunkBySentences:
    def test_empty_text_is_no_chunks(self) -> None:
        assert chunk_by_sentences("", 2) == []

    def test_worked_example_no_overlap(self) -> None:
        text = "One. Two. Three. Four. Five."
        chunks = chunk_by_sentences(text, 2)
        assert len(chunks) == 3
        prev_end = 0
        for a, b in chunks:
            assert a == prev_end
            prev_end = b
        assert prev_end == len(text)

    def test_worked_example_with_overlap(self) -> None:
        text = "One. Two. Three. Four. Five. Six."
        chunks = chunk_by_sentences(text, 3, overlap=1)
        assert len(chunks) >= 2
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
            assert next_start < prev_end

    def test_fewer_sentences_than_per_chunk_is_one_chunk(self) -> None:
        text = "Only one sentence here."
        chunks = chunk_by_sentences(text, 100)
        assert chunks == [(0, len(text))]

    def test_sentences_per_chunk_zero_or_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="sentences_per_chunk must be >= 1"):
            chunk_by_sentences("One. Two.", 0)

    def test_overlap_equal_to_or_above_sentences_per_chunk_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be < sentences_per_chunk"):
            chunk_by_sentences("One. Two. Three.", 2, overlap=2)

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            chunk_by_sentences("One. Two.", 2, overlap=-1)

    def test_offsets_are_codepoint_offsets_on_astral_text(self) -> None:
        # Each emoji is a single Python codepoint but four UTF-8 bytes,
        # so a byte-offset regression would read (0, 56), (56, 76) here
        # instead of the codepoint offsets below (the first chunk's 50
        # codepoints include two emoji, its trailing space included by
        # the segmenter's fold-into-preceding-sentence convention).
        # Slicing the chunks back out must return whole emoji, never a
        # torn surrogate half.
        text = (
            "First sentence \U0001f600 here. "
            "Second sentence \U0001f601 follows. "
            "Third \U0001f602 one ends."
        )
        chunks = chunk_by_sentences(text, 2)
        assert chunks == [(0, 50), (50, 67)]
        assert [text[a:b] for a, b in chunks] == [
            "First sentence \U0001f600 here. Second sentence \U0001f601 follows. ",
            "Third \U0001f602 one ends.",
        ]


# ---------------------------------------------------------------------------
# chunk_by_paragraphs: a HEURISTIC boundary (no Unicode Standard for
# paragraphs), stated plainly: a run of 2+ consecutive newlines (\r\n
# counts as one unit) is a paragraph break, matching tors.normalize's own
# "2+ newlines survive as the paragraph gap" convention. A single \n is
# ordinary content, not a break.
# ---------------------------------------------------------------------------


class TestChunkByParagraphs:
    def test_empty_text_is_no_chunks(self) -> None:
        assert chunk_by_paragraphs("", 2) == []

    def test_worked_example_no_overlap(self) -> None:
        text = "First para.\n\nSecond para.\n\nThird para."
        chunks = chunk_by_paragraphs(text, 2)
        pieces = [text[a:b] for a, b in chunks]
        assert pieces == ["First para.\n\nSecond para.", "Third para."]

    def test_worked_example_with_overlap(self) -> None:
        text = "P1.\n\nP2.\n\nP3.\n\nP4.\n\nP5.\n\nP6."
        chunks = chunk_by_paragraphs(text, 3, overlap=1)
        assert len(chunks) >= 2
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
            assert next_start < prev_end

    def test_fewer_paragraphs_than_per_chunk_is_one_chunk(self) -> None:
        text = "Only one paragraph here."
        assert chunk_by_paragraphs(text, 100) == [(0, len(text))]

    def test_a_single_newline_is_not_a_paragraph_break(self) -> None:
        text = "line one\nline two"
        assert chunk_by_paragraphs(text, 100) == [(0, len(text))]

    def test_three_or_more_newlines_are_still_one_break(self) -> None:
        text = "First.\n\n\n\nSecond."
        chunks = chunk_by_paragraphs(text, 1)
        pieces = [text[a:b] for a, b in chunks]
        assert pieces == ["First.", "Second."]

    def test_crlf_separators_are_recognized(self) -> None:
        text = "First.\r\n\r\nSecond."
        chunks = chunk_by_paragraphs(text, 1)
        pieces = [text[a:b] for a, b in chunks]
        assert pieces == ["First.", "Second."]

    def test_leading_and_trailing_blank_runs_are_trimmed_not_emitted(self) -> None:
        text = "\n\n\nHello\n\n\n"
        chunks = chunk_by_paragraphs(text, 100)
        assert chunks == [(3, 8)]
        assert text[3:8] == "Hello"

    def test_a_whitespace_only_interior_paragraph_is_emitted_not_filtered(self) -> None:
        # docs/api.md's own claim, pinned: "Unlike the word/line twins,
        # paragraphs have NO content filter here: a whitespace-only
        # paragraph IS emitted as a chunk (only fully-empty spans are
        # dropped)". The middle paragraph of "x\n\n \n\ny" is a lone
        # space -- a content filter like the word/line twins' would drop
        # it; the paragraph heuristic emits it as a real chunk.
        text = "x\n\n \n\ny"
        chunks = chunk_by_paragraphs(text, 1)
        assert chunks == [(0, 1), (3, 4), (6, 7)]
        assert [text[a:b] for a, b in chunks] == ["x", " ", "y"]

    def test_overlapping_chunks_can_share_blank_content(self) -> None:
        # The doc sentence's second half: "so an overlapping pair of
        # chunks can share blank content" -- the overlap window repeats
        # the whitespace-only paragraph itself, and the two chunks'
        # shared span is exactly the lone space.
        text = "x\n\n \n\ny"
        chunks = chunk_by_paragraphs(text, 2, overlap=1)
        assert chunks == [(0, 4), (3, 7)]
        assert [text[a:b] for a, b in chunks] == ["x\n\n ", " \n\ny"]
        shared = text[chunks[1][0] : chunks[0][1]]
        assert shared == " "

    def test_offsets_are_codepoint_offsets_on_astral_text(self) -> None:
        # Each emoji is a single Python codepoint but four UTF-8 bytes,
        # so a byte-offset regression would read (0, 4), (6, 10) here
        # instead. Offsets are str slicing units, and slicing the chunks
        # back out must return whole emoji, never a torn surrogate half.
        text = "\U0001f600\n\n\U0001f601"  # 4 codepoints, 10 UTF-8 bytes
        assert chunk_by_paragraphs(text, 1) == [(0, 1), (3, 4)]
        assert [text[a:b] for a, b in chunk_by_paragraphs(text, 1)] == [
            "\U0001f600",
            "\U0001f601",
        ]

    def test_paragraphs_per_chunk_zero_or_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="paragraphs_per_chunk must be >= 1"):
            chunk_by_paragraphs("a\n\nb", 0)

    def test_overlap_equal_to_or_above_paragraphs_per_chunk_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be < paragraphs_per_chunk"):
            chunk_by_paragraphs("a\n\nb\n\nc", 2, overlap=2)

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            chunk_by_paragraphs("a\n\nb", 2, overlap=-1)

    @given(st.text(alphabet="ab \n\r", max_size=60), st.integers(min_value=1, max_value=5))
    @settings(max_examples=300)
    def test_forward_progress_never_stalls(self, text: str, per_chunk: int) -> None:
        """No stalled/looping start over arbitrary newline-heavy input,
        for every valid overlap below ``per_chunk``, the fast-failing
        guarantee this whole module docstring names as the top priority."""
        for overlap in range(per_chunk):
            chunks = chunk_by_paragraphs(text, per_chunk, overlap=overlap)
            starts = [a for a, _ in chunks]
            assert all(b > a for a, b in zip(starts, starts[1:], strict=False))


# ---------------------------------------------------------------------------
# chunk_by_lines: the transcript/log shape. A break is a \n, a lone \r,
# or a \r\n pair counted as ONE unit (the same CR/CRLF folding as
# chunk_by_paragraphs; str.splitlines' exotic separators are NOT breaks).
# A line counts only when it carries content, so blank lines ride inside
# a chunk's span rather than counting, and a chunk ends at its last
# line's end, never through the trailing break (non-covering, like
# chunk_by_words).
# ---------------------------------------------------------------------------


class TestChunkByLines:
    def test_empty_text_is_no_chunks(self) -> None:
        assert chunk_by_lines("", 3) == []

    def test_whitespace_only_text_is_no_chunks(self) -> None:
        # No line carries a non-whitespace codepoint, so there is nothing
        # to window: not even a blank-line chunk.
        assert chunk_by_lines("   \n\t\n\r\n", 3) == []

    def test_worked_example_no_overlap(self) -> None:
        text = "l1\nl2\nl3\nl4\nl5"
        chunks = chunk_by_lines(text, 2)
        assert [text[a:b] for a, b in chunks] == ["l1\nl2", "l3\nl4", "l5"]

    def test_crlf_pair_is_one_unit_and_is_never_torn(self) -> None:
        # \r\n folds into ONE break, so a 1-line window never treats the
        # pair as a "\r" break plus a "\n" break and tears it across two
        # chunks' gap.
        text = "a\r\nb"
        assert chunk_by_lines(text, 1) == [(0, 1), (3, 4)]
        assert chunk_by_lines(text, 2) == [(0, 4)]

    def test_lone_carriage_return_is_a_break(self) -> None:
        text = "a\rb"
        assert chunk_by_lines(text, 1) == [(0, 1), (2, 3)]

    def test_splitlines_exotic_separators_are_not_breaks(self) -> None:
        # str.splitlines() would also break on \v \f NEL LS PS; the line
        # contract here (like chunk_by_paragraphs) recognizes only \n,
        # lone \r, and \r\n, so vertical tab, form feed, NEL, LS, and PS
        # all ride as ordinary content inside one line.
        text = "a\vb\fc"
        assert chunk_by_lines(text, 1) == [(0, 5)]
        assert chunk_by_lines("a\x85b\u2028c\u2029d", 1) == [(0, 7)]

    def test_blank_line_judgement_is_the_unicode_white_space_property_not_str_isspace(self) -> None:
        # WHICH lines carry content is judged by the Unicode White_Space
        # property (Rust's char::is_whitespace), not str.isspace(): an
        # NBSP-only line is blank (White_Space=Yes, it rides inside a
        # chunk's span without counting, exactly like an empty line),
        # while the FS-US separator controls \x1c-\x1f are White_Space=No
        # even though str.isspace() accepts them, so an FS-only line
        # carries CONTENT and counts toward the window like a visible
        # character. The one deliberate str.isspace divergence, pinned
        # truthfully as-is.
        assert chunk_by_lines("a\n\u00a0\nb", 1) == [(0, 1), (4, 5)]
        assert chunk_by_lines("a\n\x1c\nb", 1) == [(0, 1), (2, 3), (4, 5)]

    def test_offsets_are_codepoint_offsets_on_astral_text(self) -> None:
        # One emoji per line: each is a single Python codepoint but four
        # UTF-8 bytes, so a byte-offset regression would read (0, 4) and
        # (8, 12) here instead. Offsets are str slicing units, and slicing
        # the chunks back out must return whole emoji, never a torn
        # surrogate half.
        text = "\U0001f600\n\U0001f601"  # 3 codepoints, 9 UTF-8 bytes
        assert chunk_by_lines(text, 1) == [(0, 1), (2, 3)]
        assert [text[a:b] for a, b in chunk_by_lines(text, 1)] == ["\U0001f600", "\U0001f601"]
        assert chunk_by_lines(text, 2) == [(0, 3)]

    def test_blank_lines_do_not_count_but_ride_inside_a_chunk_span(self) -> None:
        # A line counts only when it carries content: the blank line
        # between the two messages neither counts toward the window nor
        # splits a chunk's interior — it rides along inside the chunk's
        # span, exactly as inter-word whitespace rides along in
        # chunk_by_words.
        text = "msg one\n\nmsg two"
        assert chunk_by_lines(text, 2) == [(0, 16)]  # one window holds both content lines
        assert chunk_by_lines(text, 1) == [(0, 7), (9, 16)]

    def test_trailing_break_yields_no_phantom_empty_line(self) -> None:
        # A chunk ends at its last line's end, never through the trailing
        # break, and that trailing break does not materialize an extra
        # empty final chunk.
        text = "l1\n"
        assert chunk_by_lines(text, 1) == [(0, 2)]
        assert text[0:2] == "l1"

    def test_fewer_lines_than_per_chunk_is_one_chunk(self) -> None:
        text = "one\ntwo"
        assert chunk_by_lines(text, 100) == [(0, len(text))]

    def test_last_chunk_may_be_partial(self) -> None:
        text = "l1\nl2\nl3\nl4\nl5"
        chunks = chunk_by_lines(text, 2)
        assert chunks[-1] == (12, 14)
        assert chunks[-1][1] == len(text)

    def test_worked_example_with_overlap(self) -> None:
        # The LangChain #34804-shaped regression the words/sentences/
        # paragraphs classes above already pin, mirrored here: overlap
        # must repeat WHOLE lines (genuine shared content), never merely
        # accept the parameter while sharing nothing or only break chars.
        text = "l1\nl2\nl3\nl4\nl5"
        chunks = chunk_by_lines(text, 2, overlap=1)
        assert [text[a:b] for a, b in chunks] == ["l1\nl2", "l2\nl3", "l3\nl4", "l4\nl5"]
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
            assert next_start < prev_end, "no actual overlap"
            shared = text[next_start:prev_end]
            assert shared, "shared span must be non-empty"
            assert shared.strip(), "shared span carries no line content"

    def test_lines_per_chunk_zero_or_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="lines_per_chunk must be >= 1, got 0"):
            chunk_by_lines("a\nb", 0)
        with pytest.raises(ValueError, match="lines_per_chunk must be >= 1, got -1"):
            chunk_by_lines("a\nb", -1)

    def test_overlap_equal_to_or_above_lines_per_chunk_raises(self) -> None:
        with pytest.raises(
            ValueError, match="overlap must be < lines_per_chunk, got overlap=2, lines_per_chunk=2"
        ):
            chunk_by_lines("a\nb\nc", 2, overlap=2)

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be >= 0, got -1"):
            chunk_by_lines("a\nb", 2, overlap=-1)

    @given(st.text(alphabet="ab \n\r", max_size=60), st.integers(min_value=1, max_value=5))
    @settings(max_examples=300)
    def test_forward_progress_never_stalls(self, text: str, per_chunk: int) -> None:
        """No stalled/looping start over arbitrary break-heavy input,
        for every valid overlap below ``per_chunk``, the fast-failing
        guarantee this whole module docstring names as the top priority."""
        for overlap in range(per_chunk):
            chunks = chunk_by_lines(text, per_chunk, overlap=overlap)
            starts = [a for a, _ in chunks]
            assert all(b > a for a, b in zip(starts, starts[1:], strict=False))


# ---------------------------------------------------------------------------
# Composition sanity: pipeline shape these primitives exist to serve
# ---------------------------------------------------------------------------


class TestComposesAsDocumented:
    def test_chunk_text_slices_are_directly_usable_with_python_slicing(self) -> None:
        text = "The quick brown fox jumps over the lazy dog and keeps running."
        chunks = chunk_text(text, 20, overlap=5)
        pieces = [text[a:b] for a, b in chunks]
        assert all(pieces)
        assert all(len(p) <= 20 for p in pieces)

    def test_chunk_by_words_and_chunk_by_sentences_agree_on_a_single_segment_input(
        self,
    ) -> None:
        text = "Solo."
        assert chunk_by_words(text, 10) == [(0, len(text))]
        assert chunk_by_sentences(text, 10) == [(0, len(text))]


# ---------------------------------------------------------------------------
# Streaming twins: chunk_text_iter / chunk_by_words_iter /
# chunk_by_sentences_iter / chunk_by_paragraphs_iter / chunk_by_lines_iter
# ---------------------------------------------------------------------------


class TestStreamingIterParity:
    """Each `_iter` twin must yield exactly the same sequence, in the same
    order, as its list-returning counterpart, the same guarantee
    word_bounds/word_bounds_iter and find_patterns/find_patterns_iter
    already hold. The iterator is eager at construction (the whole scan
    runs once, up front), so this is a marshalling-shape change, not a
    computation-shape change: the two APIs must never disagree."""

    def test_chunk_text_iter_matches_the_list(self) -> None:
        text = "abc def ghi jkl mno pqr"
        assert list(chunk_text_iter(text, 8)) == chunk_text(text, 8)

    def test_chunk_by_words_iter_matches_the_list(self) -> None:
        text = "one two three four five six seven"
        assert list(chunk_by_words_iter(text, 3, overlap=1)) == chunk_by_words(text, 3, overlap=1)

    def test_chunk_by_sentences_iter_matches_the_list(self) -> None:
        text = "One. Two. Three. Four."
        assert list(chunk_by_sentences_iter(text, 2)) == chunk_by_sentences(text, 2)

    def test_chunk_by_lines_iter_matches_the_list(self) -> None:
        text = "l1\nl2\nl3\nl4\nl5"
        assert list(chunk_by_lines_iter(text, 2, overlap=1)) == chunk_by_lines(text, 2, overlap=1)

    def test_chunk_by_paragraphs_iter_matches_the_list(self) -> None:
        # the docs' worked example (docs/api.md's chunk_by_paragraphs_iter
        # section, pinned byte-exact in tests/test_docs_examples.py): four
        # paragraphs, two per chunk, then the overlap spelling repeating
        # one whole paragraph -- the literals, not just list parity.
        text = (
            "Attendees: Ada, Grace, Edsger.\n\n"
            "Grace: parser rewrite halves latency.\n\n"
            "Edsger: spec drift question, unresolved.\n\n"
            "Next sync moves to Thursday."
        )
        assert list(chunk_by_paragraphs_iter(text, 2)) == chunk_by_paragraphs(text, 2)
        assert list(chunk_by_paragraphs_iter(text, 2)) == [(0, 69), (71, 141)]
        assert list(chunk_by_paragraphs_iter(text, 2, overlap=1)) == chunk_by_paragraphs(
            text, 2, overlap=1
        )
        assert list(chunk_by_paragraphs_iter(text, 2, overlap=1)) == [
            (0, 69),
            (32, 111),
            (71, 141),
        ]

    def test_chunk_by_paragraphs_iter_is_an_iterator_validating_at_construction(self) -> None:
        # The iter-twin-specific bits beyond list parity: the constructor
        # returns a real iterator (iter() of it is itself, ready for for
        # loops and unpacking), and validation is EAGER -- a bad count
        # raises at construction, before the first __next__, the same
        # fail-fast shape every eager _iter twin has.
        it = chunk_by_paragraphs_iter("a\n\nb", 2)
        assert iter(it) is it
        assert it.__length_hint__() == 1
        with pytest.raises(ValueError, match="paragraphs_per_chunk must be >= 1"):
            chunk_by_paragraphs_iter("a\n\nb", 0)

    def test_empty_text_is_an_empty_iterator_not_an_error(self) -> None:
        assert list(chunk_text_iter("", 5)) == []
        assert list(chunk_by_words_iter("", 3)) == []
        assert list(chunk_by_sentences_iter("", 3)) == []
        assert list(chunk_by_paragraphs_iter("", 3)) == []
        assert list(chunk_by_lines_iter("", 3)) == []

    def test_length_hint_counts_down_as_the_iterator_drains(self) -> None:
        it = chunk_text_iter("abc def ghi jkl mno", 8)
        remaining = it.__length_hint__()
        drained = 0
        for _ in it:
            drained += 1
            assert it.__length_hint__() == remaining - drained

    @given(
        text=st.text(alphabet="ab .", max_size=40),
        max_chars=st.integers(min_value=1, max_value=10),
        boundary=st.sampled_from(["word", "sentence"]),
    )
    @settings(max_examples=200)
    def test_chunk_text_iter_matches_the_list_property(
        self, text: str, max_chars: int, boundary: str
    ) -> None:
        assert list(chunk_text_iter(text, max_chars, boundary=boundary)) == chunk_text(
            text, max_chars, boundary=boundary
        )

    @given(
        text=st.text(alphabet="ab ", max_size=40),
        per_chunk=st.integers(min_value=1, max_value=6),
        data=st.data(),
    )
    @settings(max_examples=200)
    def test_chunk_by_words_iter_matches_the_list_property(
        self, text: str, per_chunk: int, data: object
    ) -> None:
        overlap = data.draw(st.integers(min_value=0, max_value=per_chunk - 1))  # type: ignore[attr-defined]
        assert list(chunk_by_words_iter(text, per_chunk, overlap=overlap)) == chunk_by_words(
            text, per_chunk, overlap=overlap
        )

    @given(
        text=st.text(alphabet="ab \n\r", max_size=40),
        per_chunk=st.integers(min_value=1, max_value=6),
        data=st.data(),
    )
    @settings(max_examples=200)
    def test_chunk_by_paragraphs_iter_matches_the_list_property(
        self, text: str, per_chunk: int, data: object
    ) -> None:
        # the same newline-run alphabet the lines property test uses: a
        # paragraph needs a 2+ newline run, which "ab \n\r" generates
        # freely, so this exercises multi-paragraph inputs, not just
        # single-paragraph degenerates.
        overlap = data.draw(st.integers(min_value=0, max_value=per_chunk - 1))  # type: ignore[attr-defined]
        assert list(chunk_by_paragraphs_iter(text, per_chunk, overlap=overlap)) == (
            chunk_by_paragraphs(text, per_chunk, overlap=overlap)
        )

    @given(
        text=st.text(alphabet="ab \n\r", max_size=40),
        per_chunk=st.integers(min_value=1, max_value=6),
        data=st.data(),
    )
    @settings(max_examples=200)
    def test_chunk_by_lines_iter_matches_the_list_property(
        self, text: str, per_chunk: int, data: object
    ) -> None:
        overlap = data.draw(st.integers(min_value=0, max_value=per_chunk - 1))  # type: ignore[attr-defined]
        assert list(chunk_by_lines_iter(text, per_chunk, overlap=overlap)) == chunk_by_lines(
            text, per_chunk, overlap=overlap
        )

    def test_raises_the_same_value_errors_as_the_list_functions(self) -> None:
        with pytest.raises(ValueError, match="max_chars must be >= 1"):
            chunk_text_iter("abc", 0)
        with pytest.raises(ValueError, match="words_per_chunk must be >= 1"):
            chunk_by_words_iter("abc", 0)
        with pytest.raises(ValueError, match="sentences_per_chunk must be >= 1"):
            chunk_by_sentences_iter("abc", 0)
        with pytest.raises(ValueError, match="paragraphs_per_chunk must be >= 1"):
            chunk_by_paragraphs_iter("a\n\nb", 0)
        with pytest.raises(ValueError, match="lines_per_chunk must be >= 1"):
            chunk_by_lines_iter("a\nb", 0)

    def test_raises_the_same_overlap_value_errors_as_the_list_functions(self) -> None:
        # The argument contract's other two branches, which the
        # per-chunk=0 rows above do not reach: negative overlap, and
        # overlap >= the per-chunk count. Each iter twin rejects them
        # with the list spelling's own message, values included where
        # the family's wording carries them (chunk_by_lines' newer
        # messages name the numbers; the older siblings' do not) -- the
        # same messages the list-level classes above pin.
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            chunk_text_iter("abc def", 5, overlap=-1)
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            chunk_by_words_iter("one two three", 2, overlap=-1)
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            chunk_by_sentences_iter("One. Two.", 2, overlap=-1)
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            chunk_by_paragraphs_iter("a\n\nb", 2, overlap=-1)
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            chunk_by_lines_iter("a\nb", 2, overlap=-1)
        with pytest.raises(ValueError, match="overlap must be < max_chars"):
            chunk_text_iter("abc def", 5, overlap=5)
        with pytest.raises(ValueError, match="overlap must be < words_per_chunk"):
            chunk_by_words_iter("one two three", 2, overlap=2)
        with pytest.raises(ValueError, match="overlap must be < sentences_per_chunk"):
            chunk_by_sentences_iter("One. Two.", 2, overlap=2)
        with pytest.raises(ValueError, match="overlap must be < paragraphs_per_chunk"):
            chunk_by_paragraphs_iter("a\n\nb\n\nc", 2, overlap=2)
        with pytest.raises(
            ValueError, match="overlap must be < lines_per_chunk, got overlap=2, lines_per_chunk=2"
        ):
            chunk_by_lines_iter("a\nb\nc", 2, overlap=2)


# ---------------------------------------------------------------------------
# Error precedence, family-wide: when a call is wrong on TWO axes at once
# (a text that cannot cross the UTF-8 argument boundary, and an invalid
# count), WHICH error the caller sees first is part of the contract, and
# the list spelling and its _iter twin must agree on it.
# ---------------------------------------------------------------------------

# A str CPython can build (lone surrogates survive str literals and
# concatenation) but UTF-8 cannot encode: the input every str-in tors
# function refuses at its argument boundary with UnicodeEncodeError.
_LONE_SURROGATE_TEXT = "bad \udcff text"

# The five list/iter pairs, one parametrize row each, so every precedence
# test below runs against the whole family.
_LIST_ITER_PAIRS = [
    pytest.param(chunk_text, chunk_text_iter, id="chunk_text"),
    pytest.param(chunk_by_words, chunk_by_words_iter, id="chunk_by_words"),
    pytest.param(chunk_by_sentences, chunk_by_sentences_iter, id="chunk_by_sentences"),
    pytest.param(chunk_by_paragraphs, chunk_by_paragraphs_iter, id="chunk_by_paragraphs"),
    pytest.param(chunk_by_lines, chunk_by_lines_iter, id="chunk_by_lines"),
]


class TestErrorPrecedence:
    """The #30 item-4 pin: a lone-surrogate text plus an invalid count used
    to raise DIFFERENT exceptions by spelling -- the list functions take
    ``text`` as a pyo3 ``&str`` argument, so the conversion's
    ``UnicodeEncodeError`` fires before the body (and its count checks)
    even runs, while the ``_iter`` twins validated the counts first and
    answered ``ValueError``. The list spelling is the older, shipped
    contract and cannot change, so the iter twins now borrow the text
    FIRST: argument-conversion errors beat count/overlap ``ValueError``s
    in every pair, identically. ``UnicodeEncodeError`` IS a ``ValueError``
    subclass, so a plain ``pytest.raises(ValueError)`` would pass either
    way -- the exact-type assertions are the actual pin, the same vacuity
    guard tests/test_b64_decode.py's lone-surrogate gate names."""

    @pytest.mark.parametrize(("list_fn", "iter_fn"), _LIST_ITER_PAIRS)
    def test_bad_text_beats_bad_count_in_both_spellings(
        self, list_fn: Callable[..., object], iter_fn: Callable[..., object]
    ) -> None:
        for fn in (list_fn, iter_fn):
            with pytest.raises(UnicodeEncodeError) as excinfo:
                fn(_LONE_SURROGATE_TEXT, 0)
            assert type(excinfo.value) is UnicodeEncodeError

    @pytest.mark.parametrize(("list_fn", "iter_fn"), _LIST_ITER_PAIRS)
    def test_bad_count_on_valid_text_is_exactly_value_error(
        self, list_fn: Callable[..., object], iter_fn: Callable[..., object]
    ) -> None:
        # UnicodeEncodeError is a ValueError subclass, so "raises
        # ValueError" is not enough: the type must be EXACTLY ValueError.
        for fn in (list_fn, iter_fn):
            with pytest.raises(ValueError, match=" must be >= 1, got 0") as excinfo:
                fn("a\n\nb\nc", 0)
            assert type(excinfo.value) is ValueError

    @pytest.mark.parametrize(("list_fn", "iter_fn"), _LIST_ITER_PAIRS)
    def test_bad_text_with_a_valid_count_is_unicode_encode_error(
        self, list_fn: Callable[..., object], iter_fn: Callable[..., object]
    ) -> None:
        for fn in (list_fn, iter_fn):
            with pytest.raises(UnicodeEncodeError) as excinfo:
                fn(_LONE_SURROGATE_TEXT, 2)
            assert type(excinfo.value) is UnicodeEncodeError

    def test_chunk_text_boundary_errors_wait_their_turn(self) -> None:
        # chunk_text's body order, shared by both spellings: the text
        # conversion, then the count/overlap checks, then parse_boundary.
        # An unrecognized boundary string is the LAST error to fire, so a
        # call that is also text-invalid raises UnicodeEncodeError and a
        # call that is also count-invalid raises the count ValueError --
        # never the boundary ValueError, in either spelling.
        for fn in (chunk_text, chunk_text_iter):
            with pytest.raises(UnicodeEncodeError) as excinfo:
                fn(_LONE_SURROGATE_TEXT, 5, boundary="paragraph")
            assert type(excinfo.value) is UnicodeEncodeError
        for fn in (chunk_text, chunk_text_iter):
            with pytest.raises(ValueError, match="max_chars must be >= 1") as excinfo:
                fn("hello world", 0, boundary="paragraph")
            assert type(excinfo.value) is ValueError

    def test_both_surrogates_corner_raises_the_type_in_both_spellings(self) -> None:
        # The both-bad corner the family's precedence contract does NOT
        # claim to settle at the message level: a lone surrogate in the
        # TEXT and a DIFFERENT lone surrogate in the BOUNDARY argument.
        # The deliverable, asserted exactly: both spellings raise
        # UnicodeEncodeError -- the two spellings of a function never
        # disagree on which error TYPE a bad call raises (the docs' scoped
        # wording). The message PROVENANCE differs by spelling, though:
        # pyo3 extracts the list spelling's `text: &str` first (argument
        # order), so it reports the text's surrogate, while the iter
        # twin's `text` is an unconverted Bound[PyString] and its
        # `boundary: &str` is extracted ahead of the body, so it reports
        # the boundary's. Frozen as-is -- differ-or-match, not "fixed" --
        # because the corner is a pyo3 extraction-order artifact, not a
        # contract worth code to rearrange, and the loose provenance pin
        # below tolerates a future pyo3 that unifies the messages while
        # still catching a provenance SWAP while they differ.
        bad_boundary = "wor\udced"
        with pytest.raises(UnicodeEncodeError) as list_excinfo:
            chunk_text(_LONE_SURROGATE_TEXT, 5, boundary=bad_boundary)
        with pytest.raises(UnicodeEncodeError) as iter_excinfo:
            chunk_text_iter(_LONE_SURROGATE_TEXT, 5, boundary=bad_boundary)
        assert type(list_excinfo.value) is UnicodeEncodeError
        assert type(iter_excinfo.value) is UnicodeEncodeError
        list_msg = str(list_excinfo.value)
        iter_msg = str(iter_excinfo.value)
        # Current behavior: they differ, the list naming the text's
        # surrogate (\udcff) and the iter the boundary's (\udced).
        assert list_msg == iter_msg or ("\\udcff" in list_msg and "\\udced" in iter_msg), (
            f"the both-surrogates corner's message provenance moved: "
            f"list {list_msg!r}, iter {iter_msg!r}"
        )


# ---------------------------------------------------------------------------
# Grapheme-boundary alignment for the unit-count chunkers, the invariant
# their merge step (against the shared grapheme boundary index) must not
# lose on arbitrary text: every chunk edge lands on a cluster boundary.
# chunk_by_paragraphs is deliberately NOT here: its spans are line-run
# edges, documented as not necessarily cluster-aligned (a combining mark
# after a newline joins the newline's own cluster, and the newline is
# separator content no paragraph's caller would call "split").
# ---------------------------------------------------------------------------


def _is_grapheme_boundary(text: str, p: int) -> bool:
    # Splitting at a cluster boundary counts the same clusters on both
    # sides; a cluster spanning p is counted once per side.
    return grapheme_count(text[:p]) + grapheme_count(text[p:]) == grapheme_count(text)


_CLUSTER_ALPHABET = st.text(
    alphabet=st.sampled_from(["0", "ำ", "ก", " ", "-", ".", "\r", "\n"]), max_size=80
)


@given(
    text=_CLUSTER_ALPHABET,
    per_chunk=st.integers(min_value=1, max_value=5),
    data=st.data(),
)
@settings(max_examples=150)
def test_chunk_by_words_edges_are_grapheme_boundaries(
    text: str, per_chunk: int, data: object
) -> None:
    overlap = data.draw(st.integers(min_value=0, max_value=per_chunk - 1))  # type: ignore[attr-defined]
    for s, e in chunk_by_words(text, per_chunk, overlap=overlap):
        assert _is_grapheme_boundary(text, s), f"start {s} mid-cluster on {text!r}"
        assert _is_grapheme_boundary(text, e), f"end {e} mid-cluster on {text!r}"


@given(
    text=_CLUSTER_ALPHABET,
    per_chunk=st.integers(min_value=1, max_value=5),
    data=st.data(),
)
@settings(max_examples=150)
def test_chunk_by_sentences_edges_are_grapheme_boundaries(
    text: str, per_chunk: int, data: object
) -> None:
    overlap = data.draw(st.integers(min_value=0, max_value=per_chunk - 1))  # type: ignore[attr-defined]
    for s, e in chunk_by_sentences(text, per_chunk, overlap=overlap):
        assert _is_grapheme_boundary(text, s), f"start {s} mid-cluster on {text!r}"
        assert _is_grapheme_boundary(text, e), f"end {e} mid-cluster on {text!r}"
