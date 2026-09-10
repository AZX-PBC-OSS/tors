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
