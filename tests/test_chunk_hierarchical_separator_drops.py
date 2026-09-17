"""The chunk_hierarchical separator-drop contract at the window boundary:
a separator match that begins exactly where the previous window resumed
must be dropped between chunks like any other, never sliced out as a
chunk of its own by the grapheme-safe raw-cut fallback.

The defect this pins: with consecutive separator matches (e.g.
``"\\n\\n"`` over ``"a\\n\\n\\n\\nb"``, two adjacent matches), the first
match's drop resumes exactly at the second match's start. That window
opens on a separator, so the level search has no genuine cut in it (the
chunk would be empty), and the raw-cut fallback used to fill the window
with the separator itself: the chunk coming back as exactly the
separator text, though the docs (and the module's own
``test_separator_content_itself_is_dropped``) say separators are dropped
between chunks. The window loop now recognizes a separator match at the
window's own start (`Level::skip_cut` in ``src/chunk_hierarchical_impl.rs``):
per level, the skip beats the level's own genuine cuts (a cut past the
match would carry the separator as content), and the first level with a
verdict wins the search, so the laziness discipline is untouched (a
finer level is realized only when the search actually descends to it).
No chunk is emitted for the skip; the window resumes at the separator's
end; the loop's iteration counter still bounds the skips (forward
progress, one strictly advancing start per skip).

Since #103 (reopened): the skip applies at the FINAL-CHUNK exit too —
the ``remaining <= max_chars`` shortcut used to run before the skip, so a
trailing window that opened on a match was emitted whole. The contract
pinned at the bottom of this file: a window that opens on a separator
match is skipped wherever the skip would otherwise apply, including the
final-chunk exit; a trailing separator run survives only as a suffix of a
content-bearing chunk; an all-separator document chunks to zero chunks.

The boundary the fix does not cross: a separator match that starts
strictly inside a window rides that chunk as content (the
leftmost-longest rule cuts at the largest in-budget boundary, so an
earlier separator can sit inside the chunk it ends); those rows are
pinned too, so a future redesign cannot drift through them by accident.
"""

from __future__ import annotations

import pytest

from tors import chunk_hierarchical

SEPS = ["\n\n"]


def test_consecutive_separator_matches_are_dropped_not_emitted() -> None:
    """The repro shape: two adjacent ``\\n\\n`` matches. The second begins
    exactly where the first's drop resumed; the chunk list is the content
    only (``"a"`` then ``"b"``), with no chunk carrying the separator."""
    text = "a\n\n\n\nb"
    assert chunk_hierarchical(text, 2, separators=SEPS, overlap=0) == [(0, 1), (5, 6)]


def test_no_separator_text_ever_comes_back_as_a_whole_chunk() -> None:
    """The contract as a property over the adjacent-match battery: every
    budget, both overlap poles, no chunk's slice is exactly a separator:
    the fallback may fill a window with raw content, never with the
    separator the caller asked to split on."""
    cases = [
        ("a\n\n\n\nb", SEPS),
        ("\n\n\n\nb", SEPS),
        ("x------y", ["---"]),
        ("a======b======c======d", ["=="]),
        ("a\n\n\n\nb\n\nc\n\nd\n\ne", SEPS),
    ]
    for text, seps in cases:
        total = len(text)
        for max_chars in range(1, total + 2):
            for overlap in (0, max_chars - 1):
                chunks = chunk_hierarchical(text, max_chars, separators=seps, overlap=overlap)
                for sep in seps:
                    for start, end in chunks:
                        assert text[start:end] != sep, (
                            f"separator {sep!r} came back as chunk {start}:{end} "
                            f"for {text!r} m={max_chars} ov={overlap}: {chunks}"
                        )


def test_skipped_separator_transitions_keep_the_overlap_invariants() -> None:
    """Skips under overlap: the skipped transition carries no snap (no
    chunk is emitted for the separator), the emitted sequence still
    strictly advances its starts and ends, and no chunk is contained in
    its predecessor; the same invariants the decline-the-snap discipline
    keeps for genuine cuts."""
    text = "a\n\n\n\nbbbbbb"
    chunks = chunk_hierarchical(text, 4, separators=SEPS, overlap=2)
    assert chunks == [(0, 3), (5, 9), (7, 11)]
    for (prev_start, prev_end), (next_start, next_end) in zip(chunks, chunks[1:], strict=False):
        assert next_start > prev_start
        assert next_end > prev_end
        assert not (next_start >= prev_start and next_end <= prev_end)


def test_a_separator_riding_a_genuine_cut_is_not_the_skip_case() -> None:
    """The leftmost-longest boundary, pinned: a separator match that
    starts strictly INSIDE a window rides that chunk as content. (0, 3)
    spans "a" plus the first match; the cut consumed was the second
    match's (the chunk ends exactly where it starts, it is dropped). The
    skip fires only for a match at the window's own start; a match one
    codepoint in is content, by the same rule that lets any pre-cut
    separator ride a chunk's interior."""
    text = "a\n\n\n\nb c"
    assert chunk_hierarchical(text, 3, separators=SEPS, overlap=0) == [(0, 3), (5, 8)]
    assert chunk_hierarchical(text, 4, separators=SEPS, overlap=0) == [(0, 3), (5, 8)]


def test_the_default_hierarchy_is_unchanged_blank_runs_are_one_gap() -> None:
    """The default hierarchy never had the defect: paragraph bounds make
    the blank-line run ONE gap, so no window can resume at a second
    separator's start. Pinned beside the fix so the literal-separator
    skip cannot leak into the default path (a paragraph gap is a cut with
    next_start strictly past it and no adjacent match to skip)."""
    assert chunk_hierarchical("a\n\n\n\nb", 2, overlap=0) == [(0, 1), (5, 6)]
    assert chunk_hierarchical("a\n\n\n\nb c", 3, overlap=0) == [(0, 1), (5, 8)]
    assert chunk_hierarchical("a\n\n\n\nb c", 3, separators=[None], overlap=0) == [(0, 1), (5, 8)]


@pytest.mark.parametrize("overlap", [0, 1, 2])
def test_three_adjacent_matches_skip_repeatedly(overlap: int) -> None:
    """A run of three adjacent matches (two skip transitions in a row)
    skips repeatedly until the window lands on content, under every legal
    overlap; the separator never comes back."""
    text = "x------y"  # "---" matches [1,4) and [4,7): adjacent
    chunks = chunk_hierarchical(text, 3, separators=["---"], overlap=overlap)
    assert (0, 1) in chunks
    for start, end in chunks:
        assert text[start:end] != "---"
    if overlap == 0:
        assert chunks == [(0, 1), (7, 8)]


# ---------------------------------------------------------------------------
# #103, reopened: the final-chunk exit answers the skip question. The
# `remaining <= max_chars` shortcut used to run BEFORE the
# separator-at-the-window-start skip, so a trailing window that opened on
# a match was emitted whole: `'\n\n'` came back `[(0, 2)]` (the whole
# chunk the separator), `'aa\n\n\n\n'` `[(0, 2), (4, 6)]` with `(4, 6)`
# pure separator. The contract the reopened ticket pins: a window that
# OPENS on a separator match is skipped wherever the skip would otherwise
# apply, INCLUDING the final-chunk exit; a trailing separator run may
# survive only as a suffix of a content-bearing chunk; an all-separator
# document chunks to zero chunks.
# ---------------------------------------------------------------------------


def test_the_reopened_repro_chunks_are_gone() -> None:
    """Both repros from the reopened ticket, exact: the all-separator
    document to zero chunks, the trailing pure-separator window dropped
    from the tail."""
    assert chunk_hierarchical("\n\n", 5, separators=SEPS, overlap=0) == []
    assert chunk_hierarchical("aa\n\n\n\n", 2, separators=SEPS, overlap=0) == [(0, 2)]


def test_the_final_exit_skip_holds_under_every_budget_and_overlap() -> None:
    """The repro battery over every budget 1..=len and both overlap poles:
    no chunk is exactly a separator, and an all-separator document is
    zero chunks at every budget (the final-exit skip fires repeatedly
    down the run; the forward-progress counter bounds it)."""
    for text, seps in [
        ("\n\n", SEPS),
        ("aa\n\n\n\n", SEPS),
        ("\n\n\n\n\n\n\n\n", SEPS),
        ("a\n\n\n\nb", SEPS),
        ("-----cd", ["-----"]),
        ("x------y", ["---"]),
        ("a\n\n\n\nb", [""]),
    ]:
        total = len(text)
        for max_chars in range(1, total + 2):
            for overlap in (0, max_chars - 1):
                chunks = chunk_hierarchical(text, max_chars, separators=seps, overlap=overlap)
                for sep in seps:
                    for start, end in chunks:
                        assert text[start:end] != sep, (
                            f"separator {sep!r} came back as chunk {start}:{end} "
                            f"at the final exit for {text!r} m={max_chars} "
                            f"ov={overlap}: {chunks}"
                        )
                if set(text) <= {"\n"} and seps == SEPS:
                    assert chunks == [], (
                        f"an all-separator document must chunk to zero chunks: "
                        f"m={max_chars}: {chunks}"
                    )


def test_a_trailing_run_survives_only_as_a_content_chunk_suffix() -> None:
    """The contract's positive half: when content follows the run, the
    post-skip window emits that content alone; when the remainder after
    the last match is not itself a match (a lone separator codepoint),
    it is a raw-cut ride-along like any too-short trailing content —
    the skip is match-shaped (`skip_cut` answers 'a match begins at
    start'), so a residual non-match codepoint is not dropped."""
    assert chunk_hierarchical("aa\n\n\n\nbb", 2, separators=SEPS, overlap=0) == [
        (0, 2),
        (6, 8),
    ]
    assert chunk_hierarchical("aa\n\n\n", 2, separators=SEPS, overlap=0) == [(0, 2), (4, 5)]
    # Under overlap the skipped transition carries no snap, same as any
    # skip: ends strictly advance, no separator chunk. The overlap=2 row
    # is the #103 lookahead corner made concrete: main's snap landed the
    # next window exactly on the (4, 6) match and its final-chunk exit
    # emitted it whole — `[(0, 2), (4, 6)]` with `(4, 6)` pure separator;
    # the skip now preempts that exit and the transition degrades.
    assert chunk_hierarchical("aa\n\n\n\n", 3, separators=SEPS, overlap=1) == [
        (0, 2),
        (1, 4),
        (3, 6),
    ]
    assert chunk_hierarchical("aa\n\n\n\n", 3, separators=SEPS, overlap=2) == [(0, 2)]


def test_a_snap_landing_on_a_separator_is_skipped_by_the_next_window() -> None:
    """#103's overlap corner: the decline-the-snap lookahead reads the
    levels' direct cuts only (a separator match at `snapped` is not
    consulted there, by design), so an accepted snap can land exactly on
    a match — and the window that then runs from it must apply the skip
    itself, at the final exit too. Swept over every legal overlap of
    several budgets: the separator never comes back as a chunk and ends
    always strictly advance."""
    text = "abcdefgh\n\nijklmnop\n\nqrstuvwx\n\nyz012345"
    for max_chars in range(4, 17):
        for overlap in range(1, max_chars):
            chunks = chunk_hierarchical(text, max_chars, separators=SEPS, overlap=overlap)
            for start, end in chunks:
                assert text[start:end] != "\n\n", (
                    f"a snapped window emitted the separator as chunk "
                    f"{start}:{end}: m={max_chars} ov={overlap}: {chunks}"
                )
            for (p_s, p_e), (n_s, n_e) in zip(chunks, chunks[1:], strict=False):
                assert n_e > p_e, f"ends not strictly advancing: m={max_chars} ov={overlap}"
                assert n_s > p_s, f"starts not strictly increasing: m={max_chars} ov={overlap}"


def test_the_default_hierarchys_final_exit_is_unchanged() -> None:
    """The default hierarchy never skipped at a final exit before #103 (a
    paragraph gap is a cut whose resume is the next paragraph's start,
    never a second match) and must not start now; the None-spliced
    spelling agrees."""
    assert chunk_hierarchical("ab\n\n\ncd", 2, overlap=0) == [(0, 2), (5, 7)]
    assert chunk_hierarchical("ab\n\n\ncd", 2, separators=[None], overlap=0) == [(0, 2), (5, 7)]
