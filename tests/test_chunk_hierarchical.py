"""Contract gate for ``tors.chunk_hierarchical``: priority-ordered fallback
chunking, the LangChain ``RecursiveCharacterTextSplitter`` pattern (cut at
the coarsest level that fits the budget, fall back to progressively finer
levels only when the coarser one has no in-budget cut), except the DEFAULT
hierarchy (``separators=None``) uses tors's own accurate UAX #29 segmenters
(paragraph -> sentence -> word) rather than literal guesses, and a
CUSTOM hierarchy (``separators=[...]``) takes caller-supplied LITERAL
strings (not regex, a scope line documented in ``src/chunk_hierarchical_impl.rs``).

UNLIKE ``chunk_text``, this is NOT a lossless covering partition: the
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

from tors import chunk_hierarchical, grapheme_count

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
        # PRECEDING sentence's segment, so the cut lands right before the
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

_TEXT = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs", "P"), max_codepoint=0x2FFF),
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
# for BOTH hierarchies and under overlap. A codepoint index p is a cluster
# boundary iff splitting there counts the same clusters on both sides — a
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
    # A clean literal, a literal that MATCHES INSIDE the SARA AM cluster,
    # and a multi-level list: the custom-hierarchy filter's whole reason
    # to exist is the second one.
    for seps in (["-"], ["ำ"], ["-", " "]):
        for s, e in chunk_hierarchical(text, max_chars, separators=seps, overlap=overlap):
            assert _is_grapheme_boundary(text, s), f"start {s} mid-cluster on {text!r}"
            assert _is_grapheme_boundary(text, e), f"end {e} mid-cluster on {text!r}"
