"""Contract gate for ``tors.chunk_hierarchical``: priority-ordered fallback
chunking, the LangChain ``RecursiveCharacterTextSplitter`` pattern (cut at
the coarsest level that fits the budget, fall back to progressively finer
levels only when the coarser one has no in-budget cut), except the default
hierarchy (``separators=None``) uses tors's own accurate UAX #29 segmenters
(paragraph -> sentence -> word) rather than literal guesses, and a
custom hierarchy (``separators=[...]``) takes caller-supplied literal
strings (not regex, a scope line documented in ``src/chunk_hierarchical_impl.rs``).
A ``None`` entry in a custom list splices that same accurate default
hierarchy in at its position (``["\n", None]`` = line -> paragraph ->
sentence -> word -> raw cut), so a caller-supplied literal shape keeps
tors's real segmenters as its oversized-segment fallback instead of
naive ``". "`` / ``" "`` literal guesses.

unlike ``chunk_text``, this is not a lossless covering partition: the
separator itself is dropped between chunks at every level except the final
grapheme-safe raw cut, the same convention ``chunk_by_paragraphs`` already
applies to blank-line runs.

Forward progress (no infinite loop) is enforced aggressively: every
hypothesis property below either bounds the chunk count by the input
length or runs under pytest's own timeout, so a regression to a
stalled/looping start fails fast rather than hanging the suite.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tors import chunk_hierarchical, grapheme_count, sentence_bounds

# ---------------------------------------------------------------------------
# Argument contract
# ---------------------------------------------------------------------------


class TestArgumentContract:
    def test_empty_text_is_no_chunks(self) -> None:
        assert chunk_hierarchical("", 10) == []

    def test_max_chars_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_chars"):
            chunk_hierarchical("hello", 0)

    def test_max_chars_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_chars"):
            chunk_hierarchical("hello", -1)

    def test_overlap_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            chunk_hierarchical("hello", 5, overlap=-1)

    def test_overlap_equal_to_max_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            chunk_hierarchical("hello world", 5, overlap=5)

    def test_empty_separators_list_is_legal_and_uses_raw_cut(self) -> None:
        # No boundary source at all: every chunk comes from the
        # grapheme-safe raw-cut fallback alone.
        chunks = chunk_hierarchical("abcdefghij klmno pqrstu", 5, separators=[])
        assert chunks
        for s, e in chunks:
            assert e - s <= 5

    @pytest.mark.parametrize(
        "separators",
        [[1, None], ["a", 2], [b"\n"], "abc"],
        ids=["int-entry", "int-entry-later", "bytes-entry", "bare-str-not-a-list"],
    )
    def test_junk_separator_entries_raise_type_error(self, separators: object) -> None:
        # The separators argument is exactly ``list[str | None]`` (or
        # ``None``): a non-str non-None entry -- or a bare ``str``, which
        # is not a list at all -- is rejected with ``TypeError`` by pyo3's
        # extraction before any Rust code runs, the same str-exactly
        # argument boundary ``find_patterns``' pattern list follows. No
        # message match: pyo3's wording is an implementation detail, the
        # type is the contract.
        with pytest.raises(TypeError):
            chunk_hierarchical("hello", 5, separators=separators)  # type: ignore[arg-type]

    def test_text_within_budget_is_one_chunk(self) -> None:
        assert chunk_hierarchical("hello world", 100) == [(0, 11)]


class TestBoundedSeparatorsExtraction:
    """``separators=`` never sizes its argument from ``__len__`` (issue
    #112's residual on this binding): pyo3's ``Option<Vec<Option<String>>>``
    extraction reserved ``Vec::with_capacity(__len__())`` before
    iterating, so a ``Sequence`` whose ``__len__`` lied (2**62) died as a
    ``PanicException`` (capacity overflow) -- ``except Exception`` cannot
    catch it. The parameter now extracts through a bounded manual walk
    (``bounded_str_list``, src/py/_borrow.rs -- the twin of
    ``scrub_pii``'s ``rules=``/``families=`` walk in src/py/pii.rs, same
    cap, same refusal bytes), and every pre-existing boundary behavior is
    preserved byte-identically: the accepted surface (list, tuple, any
    honest ``Sequence``), the refusals (bare ``str``, non-sequences,
    non-``str`` items), mid-iteration error propagation, and a
    ``__len__`` that lies LOW (the walk iterates; it never reserves).
    The one behavior change is the bomb's: a sequence yielding past the
    cap (100_000 items) dies as a catchable ``ValueError``. The hostile
    cases run in a subprocess: a regression to ``PanicException`` would
    otherwise kill this runner (it is not an ``Exception`` subclass)
    instead of failing the cell."""

    @staticmethod
    def _probe(expr: str) -> str:
        done = subprocess.run(
            [sys.executable, "-c", f"import tors\n{expr}"],
            capture_output=True,
            text=True,
            timeout=60.0,
        )
        return f"rc={done.returncode}\n{done.stdout}\n{done.stderr}"

    _LYING_HUGE = (
        "from collections.abc import Sequence\n"
        "class LyingHuge(Sequence):\n"
        "    def __len__(self): return 2**62\n"
        "    def __getitem__(self, i): raise IndexError\n"
    )

    _FLOOD = (
        "from collections.abc import Sequence\n"
        "class Flood(Sequence):\n"
        "    def __len__(self): return 2**62\n"
        "    def __getitem__(self, i):\n"
        "        if i >= 150_000: raise IndexError\n"
        "        return ' '\n"
    )

    def test_a_len_that_lies_huge_is_never_a_panic(self) -> None:
        # The reported repro, exact: pyo3_runtime.PanicException
        # (capacity overflow), not an Exception subclass. The walk never
        # reads __len__, so the empty yield is just an empty selection
        # (raw cut only, the empty-list semantics).
        done = self._probe(
            self._LYING_HUGE
            + "try:\n"
            "    out = tors.chunk_hierarchical('x', 1, separators=LyingHuge())\n"
            "    print('OK', out)\n"
            "except PanicException as e:\n"
            "    print('PANIC', e)\n"
        )
        assert "PANIC" not in done, f"the bomb is back:\n{done}"
        assert "OK" in done, f"the honest empty yield broke:\n{done}"

    def test_a_sequence_yielding_past_the_cap_is_a_value_error(self) -> None:
        # The cap (src/py/_borrow.rs MAX_LIST_ITEMS): past 100_000
        # yielded items the walk refuses with a catchable ValueError --
        # never a PanicException.
        done = self._probe(
            self._FLOOD
            + "try:\n"
            "    tors.chunk_hierarchical('a b c', 3, separators=Flood())\n"
            "    print('NO RAISE')\n"
            "except ValueError as e:\n"
            "    print('VALUEERROR', 'too many items' in str(e))\n"
            "except PanicException as e:\n"
            "    print('PANIC', e)\n"
        )
        assert "PANIC" not in done, f"the bomb is back:\n{done}"
        assert "VALUEERROR True" in done, f"not a catchable ValueError:\n{done}"

    def test_an_honest_100k_item_list_succeeds_and_100001_refuses(self) -> None:
        # The cap's edge: exactly 100_000 entries is a legitimate call
        # (deduped to nothing real), 100_001 is over the line and
        # refused -- as a catchable Exception subclass either way.
        chunks = chunk_hierarchical("a b c", 3, separators=[" "] * 100_000)
        assert chunks == [(0, 3), (4, 5)]
        with pytest.raises(ValueError) as exc:
            chunk_hierarchical("a b c", 3, separators=[" "] * 100_001)  # type: ignore[arg-type]
        assert isinstance(exc.value, Exception)
        assert "too many items" in str(exc.value)

    def test_a_len_that_lies_low_takes_every_yielded_item(self) -> None:
        # __len__ = 2 while __getitem__ yields 5: the walk iterates and
        # never reserves, so all five arrive (the pre-fix extraction's
        # own behavior, pinned so a "trust the prefix length" regression
        # shows here).
        class LowLen(Sequence):  # type: ignore[type-arg]
            def __len__(self) -> int:
                return 2

            def __getitem__(self, i: int) -> str | None:
                if i >= 5:
                    raise IndexError
                return "\n" if i % 2 == 0 else " "

        assert chunk_hierarchical("a b\nc d\ne", 3, LowLen()) == [  # type: ignore[arg-type]
            (0, 3),
            (4, 7),
            (8, 9),
        ]

    def test_a_getitem_raising_mid_iteration_propagates(self) -> None:
        # The pre-fix extraction propagated the Sequence's own error; the
        # walk iterates the same way, so it still does.
        class ExplodesMid(Sequence):  # type: ignore[type-arg]
            def __len__(self) -> int:
                return 3

            def __getitem__(self, i: int) -> str:
                if i == 1:
                    raise RuntimeError("boom mid-iteration")
                if i > 1:
                    raise IndexError
                return " "

        with pytest.raises(RuntimeError, match="boom mid-iteration"):
            chunk_hierarchical("a b c", 3, ExplodesMid())  # type: ignore[arg-type]

    def test_the_error_is_never_a_panic_exception_class(self) -> None:
        # The catchable-error-class assertion: every refusal on this
        # boundary is an Exception (ValueError/TypeError), the class
        # `except Exception` catches -- PanicException derives straight
        # from BaseException and was the defect's whole point.
        class LyingHuge(Sequence):  # type: ignore[type-arg]
            def __len__(self) -> int:
                return 2**62

            def __getitem__(self, i: int) -> str:
                if i >= 150_000:
                    raise IndexError
                return " "

        with pytest.raises(ValueError) as exc:
            chunk_hierarchical("a b c", 3, LyingHuge())  # type: ignore[arg-type]
        assert isinstance(exc.value, Exception)
        assert "too many items" in str(exc.value)


# ---------------------------------------------------------------------------
# Default hierarchy: paragraph -> sentence -> word -> raw cut
# ---------------------------------------------------------------------------


class TestDefaultHierarchy:
    def test_prefers_paragraph_boundary_when_it_fits(self) -> None:
        text = "Short one.\n\nShort two."
        chunks = chunk_hierarchical(text, 12)
        pieces = [text[s:e] for s, e in chunks]
        assert pieces == ["Short one.", "Short two."]

    def test_falls_back_to_sentence_when_paragraph_too_big(self) -> None:
        # One paragraph, two sentences: a budget that fits neither
        # paragraph nor one sentence-and-a-half forces a sentence cut.
        # sentence_bounds' own convention (used as-is here) folds the
        # trailing space after sentence-ending punctuation into the
        # preceding sentence's segment, so the cut lands right before the
        # next sentence's first character, not right after the period.
        text = "This is sentence number one. This is sentence number two."
        chunks = chunk_hierarchical(text, 32)
        pieces = [text[s:e] for s, e in chunks]
        assert pieces[0] == "This is sentence number one. "

    def test_falls_back_to_word_when_sentence_too_big(self) -> None:
        text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda."
        chunks = chunk_hierarchical(text, 12)
        for s, e in chunks:
            assert e - s <= 12
        assert len(chunks) > 1

    def test_falls_back_to_raw_cut_for_one_unbroken_token(self) -> None:
        text = "a" * 50
        chunks = chunk_hierarchical(text, 7)
        for s, e in chunks:
            assert e - s <= 7
        assert "".join(text[s:e] for s, e in chunks) == text


# ---------------------------------------------------------------------------
# Custom (literal) separator hierarchies
# ---------------------------------------------------------------------------


class TestCustomSeparators:
    def test_markdown_headers_split_before_sentences(self) -> None:
        md = "# Title\nintro text\n## Section\nmore text here that is long"
        chunks = chunk_hierarchical(md, 40, separators=["\n## ", "\n\n", ". ", " "])
        pieces = [md[s:e] for s, e in chunks]
        assert pieces[0] == "# Title\nintro text"
        assert any(p.startswith("Section") for p in pieces)

    def test_separator_content_itself_is_dropped(self) -> None:
        text = "aaa---bbb---ccc"
        chunks = chunk_hierarchical(text, 3, separators=["---"])
        pieces = [text[s:e] for s, e in chunks]
        assert "---" not in "".join(pieces)
        assert pieces == ["aaa", "bbb", "ccc"]

    def test_separator_matching_nothing_falls_through_to_raw_cut(self) -> None:
        text = "abcdefghij"
        chunks = chunk_hierarchical(text, 3, separators=["ZZZ_NEVER_MATCHES"])
        for s, e in chunks:
            assert e - s <= 3
        assert "".join(text[s:e] for s, e in chunks) == text

    def test_trailing_empty_string_sentinel_is_accepted_and_ignored(self) -> None:
        # tors does not require LangChain's own "" sentinel (the raw cut is
        # always appended regardless); a caller supplying one anyway must
        # not break anything.
        text = "aaa bbb ccc"
        with_sentinel = chunk_hierarchical(text, 4, separators=[" ", ""])
        without_sentinel = chunk_hierarchical(text, 4, separators=[" "])
        assert with_sentinel == without_sentinel


# ---------------------------------------------------------------------------
# None entries: a custom hierarchy may splice the accurate default
# hierarchy (paragraph -> sentence -> word, UAX #29) in at a position,
# keeping a caller-supplied literal shape without inheriting naive
# ". " / " " literal guesses as the oversized-segment fallback.
# ---------------------------------------------------------------------------


class TestNoneEntrySplice:
    def test_a_lone_none_entry_is_identical_to_the_default_hierarchy(self) -> None:
        # [None] splices the whole default hierarchy in as the only
        # level: indistinguishable from not passing separators at all.
        texts = [
            "One. Two. Three Four Five.",
            "First para.\n\nSecond para with more words in it than the first.",
            "a\nb\nc d e f g h i j k l m n o p",
        ]
        for text in texts:
            for max_chars in (5, 9, 17, 40):
                assert chunk_hierarchical(text, max_chars, separators=[None]) == chunk_hierarchical(
                    text, max_chars
                )
            assert chunk_hierarchical(text, 12, separators=[None], overlap=3) == chunk_hierarchical(
                text, 12, overlap=3
            )

    def test_a_never_matching_literal_above_a_none_entry_changes_nothing(self) -> None:
        # A literal that can never supply a cut is dead weight above the
        # splice: the hierarchy behaves exactly like the default one.
        text = "One. Two. Three Four Five. Six seven eight."
        for max_chars in (6, 11, 24):
            assert chunk_hierarchical(
                text, max_chars, separators=["ZZZ_NEVER_MATCHES", None]
            ) == chunk_hierarchical(text, max_chars)

    def test_line_then_splice_never_splits_mid_line_on_a_thread(self) -> None:
        # ["\n", None] is the chat-thread shape: one message per line, the
        # "\n" literal keeping whole lines whole whenever they fit, and
        # the spliced UAX #29 sentence level (not naive ". " literals) as
        # the oversized-line fallback. At this budget every cut lands
        # either at a line break or at a real sentence boundary,
        # verified directly against sentence_bounds.
        text = (
            "Nathan: kicking off.\n"
            "Priya: We briefed the U.S. team on the numbers. They asked for a "
            "follow-up meeting. The budget holds.\n"
            "Nathan: done."
        )
        chunks = chunk_hierarchical(text, 60, separators=["\n", None])
        pieces = [text[s:e] for s, e in chunks]
        # whole lines stay whole when they fit the budget
        assert pieces[0] == "Nathan: kicking off."
        assert pieces[-1] == "Nathan: done."
        sentence_edges = {p for s, e in sentence_bounds(text) for p in (s, e)}
        # a cut at the "\n" level consumes the break itself, so a boundary
        # position lands at a line break iff it is flush against one
        break_positions = {i for i, ch in enumerate(text) if ch in "\r\n"}
        for s, e in chunks:
            for p in (s, e):
                if p in (0, len(text)):
                    continue
                assert p in sentence_edges or p in break_positions or (p - 1) in break_positions, (
                    f"cut {p} is neither at a line break nor a sentence boundary: "
                    f"{text[max(0, p - 4) : p + 4]!r}"
                )

    def test_offsets_are_codepoint_offsets_on_astral_text(self) -> None:
        # Each emoji is a single Python codepoint but four UTF-8 bytes,
        # so a byte-offset regression would read (0, 9), (9, 14) here
        # instead of the codepoint offsets below. The "\n" literal never
        # fires on this text, so the spliced default hierarchy's word
        # level supplies the cut -- and the offsets are str slicing
        # units at every level, so slicing the chunks back out must
        # return whole emoji, never a torn surrogate half.
        text = "\U0001f600 \U0001f601 \U0001f602"  # 5 codepoints, 14 UTF-8 bytes
        chunks = chunk_hierarchical(text, 3, ["\n", None])
        assert chunks == [(0, 3), (3, 5)]
        assert [text[s:e] for s, e in chunks] == ["\U0001f600 \U0001f601", " \U0001f602"]

    def test_spliced_sentence_fallback_keeps_the_us_team_whole(self) -> None:
        # The contrast that motivates the splice: on this thread with
        # max_chars=40, the naive literal hierarchy ["\n", ". ", " "] cuts
        # right after "U.S": the ". " matcher treats the period ending
        # "U.S." as a separator, severing the name and dropping the period
        #: while the spliced hierarchy's word level walks past the name,
        # keeping "U.S. team" whole inside one piece.
        text = (
            "Nathan: kicking off.\n"
            "Priya: We briefed the U.S. team on the numbers. They asked for a "
            "follow-up meeting. The budget holds.\n"
            "Nathan: done."
        )
        naive = [text[s:e] for s, e in chunk_hierarchical(text, 40, separators=["\n", ". ", " "])]
        spliced = [text[s:e] for s, e in chunk_hierarchical(text, 40, separators=["\n", None])]
        assert any(p.endswith("U.S") for p in naive), "naive hierarchy no longer severs the name?"
        assert not any("U.S. team" in p for p in naive)
        assert any("U.S. team" in p for p in spliced)
        assert not any(p.endswith("U.S") for p in spliced)

    def test_a_none_entry_between_literals_splices_at_its_position(self) -> None:
        # Position is respected: ["---", None] puts the literal above the
        # splice (it supplies the coarser cut and is consumed as a
        # separator), while [None, "---"] puts the same literal below the
        # whole spliced hierarchy, where it can never fire (word-level
        # cuts already exist), so the "---" rides inside a piece as plain
        # text.
        text = "One two three.---Four five six seven eight nine ten eleven twelve."
        above = [text[s:e] for s, e in chunk_hierarchical(text, 14, separators=["---", None])]
        below = [text[s:e] for s, e in chunk_hierarchical(text, 14, separators=[None, "---"])]
        assert above[0] == "One two three."
        assert all("---" not in p for p in above)
        assert any("---" in p for p in below)


# ---------------------------------------------------------------------------
# Overlap, including the LangChain #34804-shaped regression already pinned
# for the sibling chunk functions: overlap must produce genuine shared
# content, not merely be accepted as a parameter.
# ---------------------------------------------------------------------------


class TestOverlap:
    def test_overlap_zero_is_the_default_and_produces_no_overlap(self) -> None:
        text = "one two three four five six seven eight nine ten"
        assert chunk_hierarchical(text, 15) == chunk_hierarchical(text, 15, overlap=0)

    def test_overlap_produces_genuine_shared_content(self) -> None:
        text = "one two three four five six seven eight nine ten eleven twelve"
        chunks = chunk_hierarchical(text, 20, overlap=5)
        assert len(chunks) > 1
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
            if next_start < prev_end:
                shared = text[next_start:prev_end]
                assert shared, "overlap accepted but produced no shared text"
        # The #83 invariant: every chunk's end advances strictly past the
        # previous chunk's end, overlap or not — a snapped start that would
        # re-emit the previous cut is declined, never emitted as a span
        # strictly inside its predecessor.
        prev_end = -1
        for _s, e in chunks:
            assert e > prev_end, "end did not advance past the previous chunk's end"
            prev_end = e

    def test_short_trailing_chunk_degrades_overlap_rather_than_stall(self) -> None:
        # A short final chunk shorter than the requested overlap must not
        # cause a stall or a negative-progress start; it silently drops
        # to zero overlap for that one transition (documented behavior,
        # mirrored from chunk_text_overlapping).
        text = "a very long sentence indeed that keeps going. x"
        chunks = chunk_hierarchical(text, 10, overlap=8)
        starts = [s for s, _ in chunks]
        assert starts == sorted(set(starts))
        assert all(b > a for a, b in zip(starts, starts[1:], strict=False))
        prev_end = -1
        for _s, e in chunks:
            assert e > prev_end, "end did not advance past the previous chunk's end"
            prev_end = e

    def test_chunk_is_never_strictly_inside_its_predecessor(self) -> None:
        # #83 regression: with overlap, the snapped start used to resolve
        # back to the same cut, emitting (2, 4) strictly inside (0, 4).
        # The snap is now declined for exactly that transition. Pinned
        # output, hand-traced against the decline-the-snap rule.
        assert chunk_hierarchical("aaa bbbbbbbb", 5, overlap=2) == [(0, 4), (4, 9), (7, 12)]


# ---------------------------------------------------------------------------
# overlap_boundary (#47): the opt-in word-aware overlap snap. The default
# "grapheme" is the function's whole historical behavior; "word" moves the
# grapheme candidate further back to the nearest UAX #29 word boundary,
# falling back to the grapheme candidate when the word level has no
# boundary in the snap-back range (one long token, dense-script runs).
# ---------------------------------------------------------------------------


class TestOverlapBoundary:
    def test_unknown_value_raises_value_error_naming_the_closed_set(self) -> None:
        # The families= discipline: the closed set is named in the message
        # and the validation is unconditional at the argument boundary
        # (an irrelevant knob never errors late — even overlap=0, where
        # the value can do nothing).
        for value in ("phrase", "WORD", "Grapheme", "", "graphemes"):
            with pytest.raises(ValueError, match=r"overlap_boundary.*grapheme.*word"):
                chunk_hierarchical("hello world", 5, overlap=1, overlap_boundary=value)  # type: ignore[arg-type]

    def test_word_with_overlap_zero_is_accepted_and_a_noop(self) -> None:
        # No snap site ever runs at overlap=0, so the mode is inert: the
        # output is the zero-overlap answer exactly.
        text = "one two three four five six seven eight nine ten eleven twelve"
        for max_chars in (5, 12, 20):
            assert chunk_hierarchical(text, max_chars, overlap=0) == chunk_hierarchical(
                text, max_chars, overlap=0, overlap_boundary="word"
            )

    def test_grapheme_is_the_default_and_unchanged(self) -> None:
        # The explicit "grapheme" spelling is the same function it always
        # was, and omitting the keyword agrees.
        text = "one two three four five six seven eight nine ten eleven twelve"
        for max_chars in (7, 12, 20):
            for overlap in (0, 2, max_chars - 1):
                assert chunk_hierarchical(text, max_chars, overlap=overlap) == (
                    chunk_hierarchical(
                        text, max_chars, overlap=overlap, overlap_boundary="grapheme"
                    )
                ), f"m={max_chars} ov={overlap}"

    def test_word_mode_starts_the_overlap_tail_at_a_word_edge(self) -> None:
        # The issue's motivating shape: the grapheme snap starts the tail
        # mid-word ("uter Interaction"); word mode moves it back to the
        # word's first codepoint ("Computer Interaction"). Pinned exact,
        # both budgets.
        text = (
            "...Bachelor of Arts in Human-Computer Interaction, Lakeside "
            "College, 2018\n\nCapstone project: designing a better chunker "
            "for embedding pipelines and retrieval."
        )
        seps = ["\n## ", "\n# ", None]
        assert chunk_hierarchical(text, 150, separators=seps, overlap=40) == [
            (0, 73),
            (33, 158),
        ]
        assert chunk_hierarchical(
            text, 150, separators=seps, overlap=40, overlap_boundary="word"
        ) == [(0, 73), (29, 158)]
        assert chunk_hierarchical(text, 60, separators=seps, overlap=20) == [
            (0, 3),
            (3, 60),
            (40, 73),
            (75, 134),
            (114, 158),
        ]
        assert chunk_hierarchical(
            text, 60, separators=seps, overlap=20, overlap_boundary="word"
        ) == [(0, 3), (3, 60), (38, 73), (75, 134), (112, 158)]

    def test_word_mode_falls_back_where_no_word_boundary_exists(self) -> None:
        # One long token (no internal UAX #29 boundary), dense CJK (word
        # and grapheme boundaries coincide), and Thai without a dictionary
        # (one run, no internal boundary): the grapheme candidate is kept,
        # byte-for-byte the grapheme mode's output.
        token = "a" * 60 + " b b b b"
        cjk = "中文数据段落。中文数据段落。" * 5
        thai = "กาลครั้งหนึ่งนานาพรบ์มาแล้ว " * 6
        for text in (token, cjk, thai):
            for max_chars in (13, 20):
                for overlap in (2, 5, max_chars - 1):
                    assert chunk_hierarchical(
                        text, max_chars, overlap=overlap, overlap_boundary="word"
                    ) == chunk_hierarchical(text, max_chars, overlap=overlap), (
                        f"word mode invented a boundary: m={max_chars} ov={overlap} "
                        f"text={text[:24]!r}"
                    )

    def test_word_mode_never_lands_mid_cluster(self) -> None:
        # The word level's cuts are the same grapheme-filtered list the
        # windows cut on, so a ZWJ emoji family (one 5-codepoint cluster)
        # never gains an interior boundary under word mode.
        emoji = "\U0001F468‍\U0001F469‍\U0001F467 \U0001F468‍\U0001F469‍\U0001F467 end"
        for max_chars in range(2, 9):
            for overlap in (0, 2, max_chars - 1):
                if overlap >= max_chars:
                    continue  # outside the validated envelope
                chunks = chunk_hierarchical(
                    emoji, max_chars, overlap=overlap, overlap_boundary="word"
                )
                for s, e in chunks:
                    for p in (s, e):
                        if p >= len(emoji):
                            continue  # the end-of-text boundary
                        assert emoji[p] != "‍" and not (
                            0 < p < len(emoji) and emoji[p - 1] == "‍"
                        ), f"boundary {p} lands inside a ZWJ family: {chunks}"

    def test_word_mode_keeps_the_103_skip_invariants_beside_separators(self) -> None:
        # The #103/#47 interaction: a word snap landing on or inside a
        # separator run — the skip preempts the final exit exactly as in
        # grapheme mode, ends strictly advance, and the separator never
        # comes back as a chunk.
        text = "alpha\n\nbeta\n\ngamma\n\ndelta"
        for max_chars in (6, 9, 12):
            for overlap in range(1, max_chars):
                chunks = chunk_hierarchical(
                    text, max_chars, separators=["\n\n"], overlap=overlap,
                    overlap_boundary="word",
                )
                for start, end in chunks:
                    assert text[start:end] != "\n\n", (
                        f"separator as chunk: m={max_chars} ov={overlap}: {chunks}"
                    )
                for (p_s, p_e), (n_s, n_e) in zip(chunks, chunks[1:], strict=False):
                    assert n_e > p_e, f"ends not advancing: m={max_chars} ov={overlap}"
                    assert n_s > p_s, f"starts not increasing: m={max_chars} ov={overlap}"


# ---------------------------------------------------------------------------
# Grapheme-cluster safety, the class of bug already fixed elsewhere in this
# crate (Thai SARA AM combining with the preceding base character into one
# cluster that UAX #29 word/sentence boundaries can still split).
# ---------------------------------------------------------------------------


class TestGraphemeSafety:
    def test_never_splits_a_thai_sara_am_cluster(self) -> None:
        text = "ab 0ำ cd ef 0ำ gh ij 0ำ kl"
        for max_chars in range(1, len(text)):
            chunks = chunk_hierarchical(text, max_chars)
            joined_positions = {p for s, e in chunks for p in (s, e)}
            # A cluster of "0" + SARA AM must never have its interior
            # position appear as a chunk boundary.
            sara_am_interior = text.index("ำ")
            assert sara_am_interior not in joined_positions

    def test_never_splits_a_cluster_with_custom_separators(self) -> None:
        text = "x0ำy---z0ำw"
        chunks = chunk_hierarchical(text, 2, separators=["---"])
        joined_positions = {p for s, e in chunks for p in (s, e)}
        for idx, ch in enumerate(text):
            if ch == "ำ":
                assert idx not in joined_positions

    def test_a_single_cluster_wider_than_the_budget_is_kept_whole(self) -> None:
        # "0" + SARA AM is one grapheme cluster (2 codepoints); max_chars=1
        # can't fit it. A covering-at-the-raw-cut-level chunker cannot drop
        # content, so correctness wins over the budget: the whole cluster
        # comes back as one oversized chunk rather than being split; the
        # same documented exception chunk_text's own hard-cut fallback has
        # (both share grapheme_safe_hard_cut).
        chunks = chunk_hierarchical("0ำ", 1)
        assert chunks == [(0, 2)]


# ---------------------------------------------------------------------------
# Hypothesis properties
# ---------------------------------------------------------------------------

# The newline is deliberate: "\n" is category Cc, which the category
# whitelist below never draws, so without it in the sampled set the
# spliced-hierarchy property's ["\n", None] shape would carry a line
# literal that can never fire -- dead weight above the splice, the same
# shape test_a_never_matching_literal_above_a_none_entry_changes_nothing
# pins deliberately, exercising no line cut at all. Sampling "\n" (and
# "." and " ", already reachable through the categories but weighted up
# here) lets every shape's literals actually match, so the splice
# exercises its line level.
_TEXT = st.text(
    alphabet=st.sampled_from(["\n", ".", " "])
    | st.characters(whitelist_categories=("L", "N", "Zs", "P"), max_codepoint=0x2FFF),
    max_size=300,
)


@given(text=_TEXT, max_chars=st.integers(min_value=1, max_value=50))
@settings(max_examples=200)
def test_no_chunk_exceeds_max_chars_default_hierarchy(text: str, max_chars: int) -> None:
    # The one documented exception (shared with chunk_text via the same
    # grapheme_safe_hard_cut): a single grapheme cluster wider than the
    # whole remaining budget is kept whole rather than split, so that one
    # chunk is allowed past max_chars. A chunk exceeding the budget is only
    # valid when it is exactly one grapheme cluster.
    for s, e in chunk_hierarchical(text, max_chars):
        if e - s > max_chars:
            assert grapheme_count(text[s:e]) == 1, (
                f"chunk exceeded max_chars={max_chars} without being a single "
                f"oversized grapheme cluster: {text[s:e]!r}"
            )


@given(text=_TEXT, max_chars=st.integers(min_value=1, max_value=50))
@settings(max_examples=150)
def test_no_chunk_exceeds_max_chars_spliced_hierarchies(text: str, max_chars: int) -> None:
    # The default hierarchy's budget property (above) extended to the
    # None-entry splice shapes: a lone splice ([None]), a dead literal
    # above it, and the chat-thread shape (["\n", None]) must all keep the
    # same one documented exception -- a chunk may exceed max_chars only
    # when it is exactly one grapheme cluster.
    for seps in ([None], ["-", None], ["\n", None]):
        for s, e in chunk_hierarchical(text, max_chars, separators=seps):
            if e - s > max_chars:
                assert grapheme_count(text[s:e]) == 1, (
                    f"separators={seps}: chunk exceeded max_chars={max_chars} without "
                    f"being a single oversized grapheme cluster: {text[s:e]!r}"
                )


@given(text=_TEXT, max_chars=st.integers(min_value=1, max_value=50))
@settings(max_examples=200)
def test_chunk_count_is_bounded_forward_progress(text: str, max_chars: int) -> None:
    # A hard ceiling on chunk count proportional to input length: a
    # regression to zero forward progress would blow this bound (or hang
    # before ever reaching this assertion).
    chunks = chunk_hierarchical(text, max_chars)
    assert len(chunks) <= len(text) + 1


@given(
    text=_TEXT,
    max_chars=st.integers(min_value=2, max_value=50),
)
@settings(max_examples=150)
def test_chunk_starts_are_strictly_increasing(text: str, max_chars: int) -> None:
    chunks = chunk_hierarchical(text, max_chars, overlap=0)
    starts = [s for s, _ in chunks]
    assert starts == sorted(set(starts))
    assert all(b > a for a, b in zip(starts, starts[1:], strict=False))


@given(
    text=_TEXT,
    max_chars=st.integers(min_value=2, max_value=50),
    data=st.data(),
)
@settings(max_examples=150)
def test_chunk_starts_and_ends_advance_under_overlap(
    text: str, max_chars: int, data: st.DataObject
) -> None:
    # The #83 invariant under overlap: a snapped start that would re-emit
    # the previous chunk's cut — a chunk strictly inside its predecessor —
    # is declined, so both starts and ends strictly advance on every
    # transition. The overlap=0 property above pins the starts half alone;
    # overlap is where the re-offer lived.
    overlap = data.draw(st.integers(min_value=1, max_value=max_chars - 1))
    chunks = chunk_hierarchical(text, max_chars, overlap=overlap)
    prev_start = -1
    prev_end = -1
    for s, e in chunks:
        assert s > prev_start, "starts not strictly increasing under overlap"
        assert e > prev_end, "ends not strictly advancing under overlap"
        prev_start = s
        prev_end = e


# ---------------------------------------------------------------------------
# Grapheme-boundary alignment, the invariant the #22 rewrite (the shared
# GraphemeIndex bitmap behind the cut filter, the raw-cut fallback, and the
# overlap snap) must not lose: every chunk edge lands on a cluster boundary
# for both hierarchies and under overlap. A codepoint index p is a cluster
# boundary iff splitting there counts the same clusters on both sides: a
# cluster spanning p would be counted once per side.
# ---------------------------------------------------------------------------


def _is_grapheme_boundary(text: str, p: int) -> bool:
    return grapheme_count(text[:p]) + grapheme_count(text[p:]) == grapheme_count(text)


# SARA AM is a category-L Thai letter, so the existing _TEXT alphabet can
# already pair it with a base character; this dedicated alphabet makes the
# pairing frequent, and adds CRLF (the ASCII fast path's one join rule).
_CLUSTER_ALPHABET = st.text(
    alphabet=st.sampled_from(["0", "ำ", "ก", " ", "-", ".", "\r", "\n"]), max_size=80
)


@given(text=_CLUSTER_ALPHABET, max_chars=st.integers(min_value=1, max_value=30))
@settings(max_examples=150)
def test_chunk_edges_are_grapheme_boundaries_default_hierarchy(text: str, max_chars: int) -> None:
    for s, e in chunk_hierarchical(text, max_chars):
        assert _is_grapheme_boundary(text, s), f"start {s} mid-cluster on {text!r}"
        assert _is_grapheme_boundary(text, e), f"end {e} mid-cluster on {text!r}"


@given(
    text=_CLUSTER_ALPHABET,
    max_chars=st.integers(min_value=2, max_value=30),
    overlap=st.integers(min_value=0, max_value=29),
)
@settings(max_examples=150)
def test_chunk_edges_are_grapheme_boundaries_custom_separators_and_overlap(
    text: str, max_chars: int, overlap: int
) -> None:
    assume(overlap < max_chars)
    # A clean literal, a literal that matches inside the SARA AM cluster,
    # a multi-level list, and a None-spliced list: the custom-hierarchy
    # filter's whole reason to exist is the second one, and the spliced
    # default hierarchy must survive the same filter unchanged.
    for seps in (["-"], ["ำ"], ["-", " "], ["-", None]):
        for s, e in chunk_hierarchical(text, max_chars, separators=seps, overlap=overlap):
            assert _is_grapheme_boundary(text, s), f"start {s} mid-cluster on {text!r}"
            assert _is_grapheme_boundary(text, e), f"end {e} mid-cluster on {text!r}"


# The #83 lookahead's own cluster alphabet: ZWJ emoji chains (a multi-codepoint
# cluster the ASCII fast path cannot spell), decomposed accents, Thai SARA AM,
# and CRLF, under overlap — the decline-the-snap lookahead re-runs the raw-cut
# fallback and the level cut filter for the snapped candidate's window, and
# neither the accepted nor the declined route may split a cluster or regress
# an end.
_ZWJ_ALPHABET = st.text(
    alphabet=st.sampled_from(
        ["a", "\U0001f469", "\u200d", "\U0001f52c", "e", "\u0301", "\u0e33", " ", "\r", "\n"]
    ),
    max_size=40,
)


@given(
    text=_ZWJ_ALPHABET,
    max_chars=st.integers(min_value=2, max_value=30),
    data=st.data(),
)
@settings(max_examples=200)
def test_zwj_and_combining_edges_survive_the_overlap_lookahead(
    text: str, max_chars: int, data: st.DataObject
) -> None:
    overlap = data.draw(st.integers(min_value=1, max_value=max_chars - 1))
    chunks = chunk_hierarchical(text, max_chars, overlap=overlap)
    prev_start, prev_end = -1, -1
    for s, e in chunks:
        assert _is_grapheme_boundary(text, s), f"start {s} mid-cluster on {text!r}"
        assert _is_grapheme_boundary(text, e), f"end {e} mid-cluster on {text!r}"
        assert s > prev_start, f"start stalled on {text!r}"
        assert e > prev_end, f"end regressed on {text!r}"
        prev_start, prev_end = s, e


# ---------------------------------------------------------------------------
# The #63 heading level: bounding ATX heading-line cuts on the default
# hierarchy (heading → paragraph → sentence → word → raw cut), the
# "heading" sentinel opt-in for custom hierarchies, and the whole-document
# demotion (a heading-bearing document under a whole-document budget comes
# back as its sections, not one giant chunk — the zero-build pin in
# tests/test_performance.py is on heading-FREE input and says so).
# ---------------------------------------------------------------------------


class TestHeadingLevel:
    def test_whole_document_budget_over_sections_returns_the_sections(self) -> None:
        text = "# Title\n\nintro text\n\n## Section\n\nmore text\n\n### Sub\n\ntail"
        chunks = chunk_hierarchical(text, len(text))
        pieces = [text[s:e] for s, e in chunks]
        assert pieces == [
            "# Title\n\nintro text",
            "## Section\n\nmore text",
            "### Sub\n\ntail",
        ]
        # The cut is before the heading, never after it: each chunk after
        # the first starts exactly at its heading line's first codepoint.
        assert text[chunks[1][0] : chunks[1][0] + 2] == "##"
        assert text[chunks[2][0] : chunks[2][0] + 3] == "###"

    def test_heading_free_text_keeps_the_single_chunk_exit(self) -> None:
        # The '#' in text gate: without the byte, no ATX heading line can
        # exist, the level is never realized, and the whole-document exit
        # is byte-for-byte the pre-#63 one.
        text = "Para one.\n\nPara two.\n\nPara three."
        assert chunk_hierarchical(text, len(text)) == [(0, len(text))]

    def test_hash_without_heading_line_still_one_chunk(self) -> None:
        # The gate opens (a '#' byte exists) but the scan finds no heading
        # line: the exit is still the untrimmed one.
        text = "a #b c\n\nd #e f"
        assert chunk_hierarchical(text, len(text)) == [(0, len(text))]

    def test_no_chunk_spans_a_heading_cut_at_any_budget(self) -> None:
        # The bounding semantics: every chunk's interior is free of heading
        # cuts (the budget bounds oversized sections; the heading bounds
        # ordinary ones). Swept over budgets 1..len on a sectioned doc.
        text = "# a\nalpha beta gamma\n\n## b\ndelta epsilon zeta\n\n## c\neta iota theta"
        starts = [i for i in range(1, len(text)) if text[i:].startswith("## ")]
        for max_chars in range(1, len(text) + 1):
            for s, e in chunk_hierarchical(text, max_chars):
                for h in starts:
                    assert not (s < h < e), (
                        f"chunk ({s}, {e}) spans the heading cut at {h} "
                        f"(budget {max_chars})"
                    )

    def test_atx_only_scope_exclusions(self) -> None:
        # The documented v1 scope (verified against the engines' emitters:
        # all three emit ATX only): none of these lines is a heading — a
        # whole-document budget over each doc must come back as one chunk
        # (any stray cut would emit two).
        cases = [
            "#no-space\nbody",
            "####### seven\nbody",
            "\\# escaped\nbody",
            "> # quoted\nbody",
            "- # item\nbody",
            "    # indented code\nbody",
            "\t# tabbed\nbody",
            "mid # line\nbody",
        ]
        for case in cases:
            assert chunk_hierarchical(case, len(case)) == [(0, len(case))], case

    def test_atx_scope_inclusions_and_fence_tracking(self) -> None:
        # The positive edges (1-6 hashes, the 1-3 space indent, trailing
        # closing hashes, tab after the run, CRLF and lone-CR line
        # endings), and the fence state machine: heading-shaped lines
        # inside a fence never cut, an unclosed fence runs to the end.
        text = "intro\n# one\nbody"
        assert [text[s:e] for s, e in chunk_hierarchical(text, len(text))] == [
            "intro",
            "# one\nbody",
        ]
        text = "intro\n## closed ##\nbody"
        assert [text[s:e] for s, e in chunk_hierarchical(text, len(text))] == [
            "intro",
            "## closed ##\nbody",
        ]
        text = "# a\r\n## b\r\nbody"
        assert [text[s:e] for s, e in chunk_hierarchical(text, len(text))] == [
            "# a",
            "## b\r\nbody",
        ]
        fenced = "# Title\n\n```python\n# not a heading\nx = 1\n```\n\n## Real\n\nbody"
        pieces = [fenced[s:e] for s, e in chunk_hierarchical(fenced, len(fenced))]
        assert len(pieces) == 2 and pieces[1].startswith("## Real")
        unclosed = "# Title\n\n```\n# not a heading\n## nor this\n"
        assert chunk_hierarchical(unclosed, len(unclosed)) == [(0, len(unclosed))]

    def test_heading_sentinel_is_the_level_not_a_literal(self) -> None:
        text = "# a\nalpha\n\n## b\nbeta"
        # The sentinel + splice is the default hierarchy spelled out.
        assert chunk_hierarchical(text, 10, separators=["heading", None]) == (
            chunk_hierarchical(text, 10)
        )
        # Either order around the splice (dedup inertness), and repeats.
        assert chunk_hierarchical(text, 10, separators=[None, "heading"]) == (
            chunk_hierarchical(text, 10)
        )
        assert chunk_hierarchical(text, 10, separators=["heading"] * 8) == (
            chunk_hierarchical(text, 10, separators=["heading"])
        )
        # The sentinel alone: heading cuts bound the sections, and the
        # oversized remainder falls to the raw cut (no paragraph level
        # below it).
        alone = chunk_hierarchical(text, 10, separators=["heading"])
        assert text[alone[0][0] : alone[0][0] + 2] == "# "

    def test_unicode_headings_cut_at_codepoint_starts(self) -> None:
        text = "précis\n\n## Ünïcodé heading ✓\n\nbodytext"
        pieces = [text[s:e] for s, e in chunk_hierarchical(text, len(text))]
        assert pieces == ["précis", "## Ünïcodé heading ✓\n\nbodytext"]

    def test_seven_hundred_sections_never_merge(self) -> None:
        # Scale shape: 700 sections, one chunk each under a whole-document
        # budget (the heading level's scan is one linear pass; the
        # per-window cost is the usual binary search over its cuts).
        text = "".join(f"## Section {i}\n\nBody paragraph {i}.\n\n" for i in range(700))
        chunks = chunk_hierarchical(text, len(text))
        assert len(chunks) == 700
        for s, _e in chunks:
            assert text[s : s + 3] == "## "
