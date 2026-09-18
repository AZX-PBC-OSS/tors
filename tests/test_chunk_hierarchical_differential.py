"""Differential gate for ``tors.chunk_hierarchical`` against a committed
pure-Python reference model, the port of the red-team audit's agent-B
reference (an eager-semantics verbatim reading of
``src/chunk_hierarchical_impl.rs``: the pre-#30 spelling, kept because
eager and lazy builds are output-equal by #30's own differential, so the
eager model pins the current function while describing the simpler
machine): paragraph ``windows(2)`` cuts, sentence/word cuts from
``tors.sentence_bounds``/``tors.word_bounds``, non-overlapping literal
matches with the literal dropped, the grapheme-cut filter, the
first-level-with-a-verdict window walk (a separator match at the
window's own start is a skip verdict: no chunk for the separator, the
window resumes at its end -- and since #103 the skip also preempts the
final-chunk exit, so an all-separator document chunks to zero chunks;
production answers that exit's skip question through an output-invisible
necessary-condition pre-test that this bare-search reference deliberately
does not have, and the sweep holds the two spellings equal), the
grapheme-safe hard cut, and the overlap snap (both ``overlap_boundary``
modes since #47: the reference's word model is the same UAX #29
word-bounds list production reads, with the decline-the-snap lookahead
running after the word snap).

Restricted to ASCII corpora the reference's grapheme model is exact:
below the extended-grapheme additions the only joining rule is GB3 (a
``\r`` immediately before an ``\n``), the same claim
``GraphemeIndex::build``'s ASCII fast path makes (pinned exhaustively by
the 128x128 adjacency test in ``truncate_impl.rs``), so for ASCII text
(CRLF included) the Python boundary model is bit-exact with the Rust
bitmap and the comparison is a real differential, not a restatement.

Duplicates are deliberately not deduped in the reference (``[None] * 8``
splices the default hierarchy eight times, ``[" "] * 8`` keeps eight
identical literal levels): a duplicate level is bit-identical to its
original, so the first-match level walk answers identically, and the
reference thereby simultaneously pins the dedup inertness the Rust side
relies on (a wrong dedup that dropped a level it should not, or kept one
that changed the answer, breaks this equality).

Since #63 the reference carries the heading level too: bounding ATX
heading-line cuts (one cut ``(gap_start, heading_start)`` per heading
line, the pre-heading newline run dropped, the first in-range cut wins —
no chunk spans a heading), the ``'#' in text`` gate (a gate-closed text
carries no heading level at all), the ``"heading"`` sentinel spelling, the
whole-document demotion (progress-gated: a demotion whose cut cannot end
past the previous chunk's end demotes nothing), and the lookahead mirror
that keeps the snap acceptance honest. The heading shape generators below
sweep heading-dense markdown, the exclusion shapes (fences, blockquotes,
escapes, the 7-hash run, no-space markers, indented code), and CRLF and
lone-CR heading line endings.

This is the pyo3/marshalling-layer pin the Rust-side differential cannot
see: the Rust unit tests diff core against core, and the fuzz target
diffs the path-dep crate's public Rust API, but only this file checks
that the boundary function (argument extraction, the separators
``list[str | None]`` shape, the tuple list marshalled back out) answers
exactly what the semantics say.
"""

from __future__ import annotations

import random
import unicodedata
from bisect import bisect_right

import tors

# The audit's seed (probe.py's SEED), carried into the committed gate so
# the corpus is reproducible run-for-run.
SEED = 20260909

# ---------------------------------------------------------------------------
# The eager-semantics reference (ported from the audit's agent-B
# ref_impl.py, with the best-cut level scan served by bisect over each
# level's cut-end array instead of a linear walk: same answer, the
# runtime budget the audit ran at needs it).
# ---------------------------------------------------------------------------


def _ascii_grapheme_boundaries(text: str) -> list[bool]:
    """``gb[i]`` iff codepoint index ``i`` starts a grapheme cluster; for
    ASCII text GB3 (CRLF) is the only joining rule, so this is exact."""
    gb = [True] * (len(text) + 1)
    prev = ""
    for i, ch in enumerate(text):
        if prev == "\r" and ch == "\n":
            gb[i] = False
        prev = ch
    return gb


def _paragraph_bounds_ref(text: str) -> list[tuple[int, int]]:
    """The random-access paragraph oracle, the same spelling the Rust
    tests module keeps as ``paragraph_bounds_reference``: runs of 2+
    newline units (CRLF = one unit), a lone unit is content, empty
    edge spans discarded."""
    n = len(text)
    if n == 0:
        return []
    bounds = []
    seg_start = 0
    i = 0
    while i < n:
        if text[i] in ("\n", "\r"):
            run_start = i
            units = 0
            while i < n and text[i] in ("\n", "\r"):
                if text[i] == "\r" and i + 1 < n and text[i + 1] == "\n":
                    i += 2
                else:
                    i += 1
                units += 1
            if units >= 2:
                if seg_start < run_start:
                    bounds.append((seg_start, run_start))
                seg_start = i
        else:
            i += 1
    if seg_start < n:
        bounds.append((seg_start, n))
    return bounds


def _para_level_cuts(text: str) -> list[tuple[int, int]]:
    bounds = _paragraph_bounds_ref(text)
    return [(bounds[k][1], bounds[k + 1][0]) for k in range(len(bounds) - 1)]


def _contig_level_cuts(bounds: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(end, end) for _start, end in bounds]


def _literal_level_cuts(text: str, sep: str) -> list[tuple[int, int]]:
    cuts = []
    pos = 0
    while True:
        i = text.find(sep, pos)
        if i < 0:
            return cuts
        cuts.append((i, i + len(sep)))
        pos = i + len(sep)


# ---------------------------------------------------------------------------
# The #63 heading level's oracle scan: one cut (gap_start, heading_start)
# per ATX heading line, independently spelled on the char grid (the
# production scan is a byte walk with fence_impl's §4.5 machine; this one
# re-derives the same clauses on chars so the pin is a differential):
#   * ATX shape only (verified from the engines' emitters: pdf_oxide's
#     markdown_prefix, anydoc's renderer, html-to-markdown-rs's default
#     HeadingStyle all emit ATX): 1-3 leading spaces, 1-6 '#', then a
#     space, a tab, or end of line. Setext underlines are excluded by
#     design (documented in src/chunk_hierarchical_impl.rs).
#   * the CommonMark §4.5 fence state machine tracks fenced code blocks:
#     a heading-shaped line inside a fence is code, never a cut.
#   * line endings are LF, CRLF, and lone CR (CommonMark's set).
#   * the cut's gap start is the end of the last non-empty content line
#     before the heading (the newline run is dropped between chunks, the
#     paragraph gap's own convention), and a heading at offset 0 records
#     no cut (nothing precedes it to bound).
# ---------------------------------------------------------------------------
def _is_fence_opener_ref(line: str) -> tuple[str, int] | None:
    indent = 0
    for c in line:
        if c == " " and indent < 3:
            indent += 1
        else:
            break
    rest = line[indent:]
    if not rest or rest[0] not in "`~":
        return None
    fence_char = rest[0]
    fence_len = len(rest) - len(rest.lstrip(fence_char))
    if fence_len < 3:
        return None
    info = rest[fence_len:].strip()
    if fence_char == "`" and "`" in info:
        return None
    return (fence_char, fence_len)


def _is_fence_closer_ref(line: str, fence_char: str, fence_len: int) -> bool:
    indent = 0
    for c in line:
        if c == " " and indent < 3:
            indent += 1
        else:
            break
    rest = line[indent:]
    run = 0
    for c in rest:
        if c == fence_char:
            run += 1
        else:
            break
    return run >= fence_len and all(c in " \t\r" for c in rest[run:])


def _is_atx_heading_line_ref(line: str) -> bool:
    i = 0
    while i < 3 and i < len(line) and line[i] == " ":
        i += 1
    if i >= len(line) or line[i] != "#":
        return False
    hashes = 0
    while i < len(line) and line[i] == "#" and hashes < 7:
        i += 1
        hashes += 1
    if hashes > 6:
        return False
    return i >= len(line) or line[i] in " \t"


def _heading_level_cuts(text: str) -> list[tuple[int, int]]:
    n = len(text)
    cuts: list[tuple[int, int]] = []
    open_fence: tuple[str, int] | None = None
    line_start = 0
    last_content_end = 0
    while line_start < n:
        line_end = n
        next_start = n
        j = line_start
        while j < n:
            if text[j] == "\n":
                line_end, next_start = j, j + 1
                break
            if text[j] == "\r":
                line_end = j
                next_start = j + 2 if (j + 1 < n and text[j + 1] == "\n") else j + 1
                break
            j += 1
        line = text[line_start:line_end]
        if open_fence is not None:
            if _is_fence_closer_ref(line, *open_fence):
                open_fence = None
        else:
            opened = _is_fence_opener_ref(line)
            if opened is not None:
                open_fence = opened
            elif line_start > 0 and _is_atx_heading_line_ref(line):
                cuts.append((last_content_end, line_start))
        if line_end > line_start:
            last_content_end = line_end
        line_start = next_start
    return cuts


def _first_cut_in(cuts: list[tuple[int, int]], after: int, limit: int) -> tuple[int, int] | None:
    """A bounding level's answer (#63): the FIRST cut with end in
    ``(after, limit]`` — a chunk may not span a heading cut — or None."""
    idx = 0
    while idx < len(cuts) and cuts[idx][0] <= after:
        idx += 1
    if idx < len(cuts) and cuts[idx][0] <= limit:
        return cuts[idx]
    return None


def _ref_chunk_hierarchical(
    text: str,
    max_chars: int,
    separators: list[str | None] | None = None,
    overlap: int = 0,
    overlap_boundary: str = "grapheme",
) -> list[tuple[int, int]]:
    if text == "":
        return []
    total = len(text)
    gb = _ascii_grapheme_boundaries(text)
    # The #63 heading level: built (eagerly, the reference's idiom)
    # exactly when the hierarchy SPELLS it (separators=None, a None
    # splice, or the "heading" sentinel) and the production gate would
    # ever realize it ('#' in text — the memchr probe's oracle spelling).
    # The held-out copy feeds the demotion check; duplicates are inert,
    # so every spelling after the first re-appends the same list.
    gate_open = "#" in text
    heading_cuts_maybe = _heading_level_cuts(text) if gate_open else None
    spelled = False
    # The FIRST heading level's index in ``levels`` (duplicates are
    # inert; the prediction below walks the slots up to and including
    # this one), and per-level bounding flags (#63): the heading level
    # cuts at its FIRST in-budget candidate (a chunk may not span a
    # heading cut); every other level groups (the largest in-budget cut
    # wins).
    heading_idx: int | None = None
    bounding: list[bool] = []
    if separators is None:
        spelled = True
        if heading_cuts_maybe is not None:
            heading_idx = 0
            bounding.append(True)
        levels = (
            ([list(heading_cuts_maybe)] if heading_cuts_maybe is not None else [])
            + [
                _para_level_cuts(text),
                _contig_level_cuts(tors.sentence_bounds(text)),
                _contig_level_cuts(tors.word_bounds(text)),
            ]
        )
        bounding += [False, False, False]
    else:
        levels = []
        for entry in separators:
            if entry is None:
                if heading_cuts_maybe is not None:
                    spelled = True
                    if heading_idx is None:
                        heading_idx = len(levels)
                    levels.append(list(heading_cuts_maybe))
                    bounding.append(True)
                levels.append(_para_level_cuts(text))
                bounding.append(False)
                levels.append(_contig_level_cuts(tors.sentence_bounds(text)))
                bounding.append(False)
                levels.append(_contig_level_cuts(tors.word_bounds(text)))
                bounding.append(False)
            elif entry == "":
                continue
            elif entry == "heading":
                if heading_cuts_maybe is not None:
                    spelled = True
                    if heading_idx is None:
                        heading_idx = len(levels)
                    levels.append(list(heading_cuts_maybe))
                    bounding.append(True)
            else:
                levels.append(_literal_level_cuts(text, entry))
                bounding.append(False)
    # The grapheme-cut filter: a level's cut is usable only when both its
    # end and its next-start are cluster boundaries.
    levels = [[(e, nx) for e, nx in cuts if gb[e] and gb[nx]] for cuts in levels]
    # The demotion check's input (#63): the spelled heading level's
    # post-filter cuts (the same list production's memoized slot
    # carries — the sentinel's level may sit mid-list, so the held-out
    # copy is filtered separately), or None when the hierarchy spells
    # no heading level: the pre-#63 behavior, whatever the text looks
    # like.
    demotion_cuts: list[tuple[int, int]] | None = None
    if spelled and heading_cuts_maybe is not None:
        demotion_cuts = [(e, nx) for e, nx in heading_cuts_maybe if gb[e] and gb[nx]]
    # Parallel strictly-increasing end arrays for the bisect below.
    ends = [[e for e, _ in cuts] for cuts in levels]
    # The skip maps: per level, separator matches by their cut end, only
    # for separator-dropping levels (next strictly past the end; a
    # contiguous level's cuts have next == end and never skip).
    skips = [{e: nx for e, nx in cut_list if nx > e} for cut_list in levels]
    # #47's word-boundary end array for the "word" snap mode: the word
    # level's own filtered cut ends, whichever hierarchy slot carried
    # them (the boundary set is the same UAX #29 word-bounds list either
    # way). Empty (never a boundary to snap to) when the mode is off.
    word_ends: list[int] = []
    if overlap_boundary == "word":
        word_level = _contig_level_cuts(tors.word_bounds(text))
        word_ends = [e for e, nx in word_level if gb[e] and gb[nx]]

    def last_at_or_before(x: int) -> int:
        for i in range(min(x, total), -1, -1):
            if gb[i]:
                return i
        return 0

    def first_after(x: int) -> int:
        for i in range(x + 1, total + 1):
            if gb[i]:
                return i
        return total

    def _demotion_predicted_cut(after: int, limit: int) -> tuple[int, int] | None:
        # The two-step prediction, mirrored from production's
        # demotion_predicted_cut: (a) the STRUCTURAL TRIGGER — the
        # heading level's own in-range cut (a closed gate or an
        # unspelled hierarchy carries no heading level at all:
        # demotion_cuts stays None, and heading-free text keeps the
        # pre-#63 exit byte-for-byte; an earlier slot's cut must never
        # demote on its own); (b) the PREDICTED VERDICT — the first
        # verdict among the slots up to and including the heading's, in
        # slot order: a slot COARSER than the heading (an earlier
        # literal) legitimately preempts the heading's cut (the
        # first-slot-with-any-verdict walk), and predicting from the
        # heading alone would mispredict the chunk the window actually
        # emits. A Skip verdict predicts "no push at all" — the
        # demotion question is moot, None.
        if demotion_cuts is None or heading_idx is None:
            return None
        if _first_cut_in(demotion_cuts, after, limit) is None:
            return None
        for lvl in range(heading_idx + 1):
            cut_list, end_list, skip_map, is_bounding = (
                levels[lvl],
                ends[lvl],
                skips[lvl],
                bounding[lvl],
            )
            if after in skip_map:
                return None
            if is_bounding:
                got = _first_cut_in(cut_list, after, limit)
                if got is not None:
                    return got
            else:
                idx = bisect_right(end_list, limit) - 1
                if idx >= 0 and cut_list[idx][0] > after:
                    return cut_list[idx]
        return None

    chunks = []
    start = 0
    while start < total:
        remaining = total - start
        # #63's final-window demotion, mirrored from production in
        # lockstep: a heading cut in range bounds the final chunk (the
        # whole-remainder exit is a SIZE concession; the heading level's
        # structural contract wins), gated on strict progress past the
        # previous chunk's end (a demotion whose predicted cut cannot
        # advance would emit a chunk contained in its predecessor — the
        # #83 violation; the untrimmed exit keeps the ends advancing).
        final_window = remaining <= max_chars
        if final_window:
            prev_end = chunks[-1][1] if chunks else 0
            got = _demotion_predicted_cut(start, start + max_chars)
            if got is not None and got[0] > prev_end:
                final_window = False
        limit = start + max_chars
        # The window's verdict, mirroring production's fused per-slot
        # search: per level, the separator-at-the-window-start skip beats
        # the level's own cuts (a cut past the match would carry the
        # separator as content), and the first level with a verdict
        # (skip or cut) wins (later levels stay unconsulted). The skip:
        # a window that opens on a separator match has no genuine cut,
        # and the raw cut would slice the separator out as a chunk of its
        # own; the separator is dropped between chunks, so the window
        # resumes at its end, no chunk emitted. Since #103 this includes
        # the final window: the whole-remainder exit below runs only when
        # the search did not answer Skip. (Production wraps this search
        # in an output-invisible necessary-condition pre-test so a
        # whole-document budget still builds no levels; this reference
        # runs the bare search, and the sweep holds the two equal.)
        cut = None
        skip_to = None
        for cut_list, end_list, skip_map, is_bounding in zip(
            levels, ends, skips, bounding, strict=True
        ):
            if start in skip_map:
                skip_to = skip_map[start]
                break
            if is_bounding:
                # The #63 heading level's bounding answer: the FIRST
                # in-budget cut past the window's start (a chunk may not
                # span a heading cut), not the largest.
                got = _first_cut_in(cut_list, start, limit)
                if got is not None:
                    cut = got
                    break
            else:
                idx = bisect_right(end_list, limit) - 1
                if idx >= 0 and cut_list[idx][0] > start:
                    cut = cut_list[idx]
                    break
        if skip_to is not None:
            start = skip_to
            continue
        if cut is None and final_window:
            chunks.append((start, total))
            break
        if cut is None:
            end = last_at_or_before(limit)
            if end <= start:
                end = first_after(start)
            cut = (end, end)
        if final_window:
            # The final-chunk exit (#103): the search above answered the
            # skip question (a Skip verdict preempted this exit); the
            # cut is discarded — the final chunk runs to the end
            # untrimmed.
            chunks.append((start, total))
            break
        chunks.append((start, cut[0]))
        if overlap == 0:
            start = cut[1]
        else:
            target = max(cut[0] - overlap, 0)
            snapped = last_at_or_before(target)
            # #47's word snap, second in the composition order: the
            # grapheme candidate lands first, then the largest word-level
            # cut end at or before it (or the grapheme candidate back
            # when the word level has no boundary there); the
            # decline-the-snap lookahead below runs on the word-snapped
            # candidate unchanged.
            if word_ends:
                wi = bisect_right(word_ends, snapped) - 1
                if wi >= 0:
                    snapped = word_ends[wi]
            # Decline-the-snap with lookahead (#83), mirroring production:
            # the candidate is taken only when the chunk cut from it ends
            # strictly past this chunk's end; otherwise the transition
            # degrades to the zero-overlap cut.
            if total - snapped <= max_chars:
                # The next window would be final — mirrored from the
                # production lookahead: the #63 demotion may cut it at a
                # heading first, and the acceptance condition needs the
                # end the window would actually take (the same
                # two-step prediction the loop head runs, so the two
                # never disagree).
                next_end = total
                got = _demotion_predicted_cut(snapped, total)
                if got is not None and got[0] > cut[0]:
                    next_end = got[0]
            else:
                next_limit = snapped + max_chars
                next_cut = None
                for cut_list, end_list, is_bounding in zip(
                    levels, ends, bounding, strict=True
                ):
                    if is_bounding:
                        got = _first_cut_in(cut_list, snapped, next_limit)
                        if got is not None:
                            next_cut = got
                            break
                    else:
                        idx = bisect_right(end_list, next_limit) - 1
                        if idx >= 0 and cut_list[idx][0] > snapped:
                            next_cut = cut_list[idx]
                            break
                if next_cut is None:
                    end = last_at_or_before(next_limit)
                    if end <= snapped:
                        end = first_after(snapped)
                    next_cut = (end, end)
                next_end = next_cut[0]
            start = (
                snapped
                if snapped > start and snapped < cut[0] and next_end > cut[0]
                else cut[1]
            )
    return chunks


# ---------------------------------------------------------------------------
# The ASCII corpus shapes (the audit's harness categories, as committed
# generators): markdown, chat-log, CRLF prose, repeated-literal,
# degenerate runs, and a mixed ASCII soup.
# ---------------------------------------------------------------------------

_WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
_NAMES = ["Nathan:", "Priya:", "Wei:", "Amara:", "Okoye:", "Silas:"]


def _words(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def _sentences(rng: random.Random, n: int) -> str:
    return " ".join(_words(rng, rng.randint(2, 6)) + "." for _ in range(n))


def _markdown(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randint(2, 5)):
        parts.append("#" * rng.randint(1, 3) + " " + _words(rng, rng.randint(1, 4)))
        parts.append(_sentences(rng, rng.randint(1, 3)))
        if rng.random() < 0.4:
            parts.append("## " + _words(rng, rng.randint(1, 3)))
    return "\n\n".join(parts)


def _heading_doc(rng: random.Random) -> str:
    """Heading-dense markdown (#63): an ATX heading line every couple of
    paragraphs, headings directly followed by headings (empty sections),
    trailing-hash closings, and a trailing heading (no section after
    it) — the bounding cut, the pre-heading gap drop, and the
    whole-document demotion all have live inputs on every cell."""
    parts = []
    for _ in range(rng.randint(2, 6)):
        parts.append("#" * rng.randint(1, 6) + " " + _words(rng, rng.randint(1, 3)))
        if rng.random() < 0.3:
            parts.append("#" * rng.randint(1, 3) + " " + _words(rng, 1))
        if rng.random() < 0.6:
            parts.append(_sentences(rng, rng.randint(1, 3)))
    if rng.random() < 0.3:
        parts.append("## trailing heading")
    return "\n".join(parts)


def _markdown_redteam(rng: random.Random) -> str:
    """The heading scan's exclusion shapes, interleaved with real
    headings: fenced code blocks (heading-shaped comment lines inside),
    `#no-space`, the 7-hash run, an escaped marker, a blockquote's and a
    list item's heading, 4-space indented code, a tab-indented line, and
    mid-line hashes. None of the excluded shapes may cut; the real
    headings must."""
    parts = [
        "# Title",
        "```python",
        "# not a heading",
        "## nor this",
        "```",
        "#no-space",
        "####### seven hashes",
        "\\# escaped",
        "> # quoted",
        "- # item",
        "    # indented code",
        "\t# tabbed",
        "hash # mid-line",
        "## closed ##",
        _sentences(rng, rng.randint(1, 2)),
        "## real one",
    ]
    order = parts[:]
    rng.shuffle(order)
    return "\n".join(order)


def _markdown_crlf_headings(rng: random.Random) -> str:
    """CRLF (and lone-CR) line endings around heading lines: the full
    line-ending set must cut, and the dropped CRLF pair must go whole."""
    lines = []
    for _ in range(rng.randint(2, 5)):
        lines.append("## " + _words(rng, rng.randint(1, 3)))
        lines.append(_sentences(rng, rng.randint(1, 2)))
    sep = rng.choice(["\r\n", "\r", "\n"])
    return sep.join(lines)


def _chat_log(rng: random.Random) -> str:
    lines = []
    for _ in range(rng.randint(3, 8)):
        lines.append(rng.choice(_NAMES) + " " + _sentences(rng, rng.randint(1, 2)))
        if rng.random() < 0.3:
            lines.append("")
    return "\n".join(lines)


def _crlf_prose(rng: random.Random) -> str:
    return "\r\n".join(_sentences(rng, rng.randint(1, 3)) for _ in range(rng.randint(2, 6)))


def _repeated_literal(rng: random.Random) -> str:
    sep = rng.choice(["---", "***", "|", "==", "END", ", "])
    return sep.join(_words(rng, rng.randint(2, 5)) for _ in range(rng.randint(3, 8)))


def _degenerate(rng: random.Random) -> str:
    unit = rng.choice(["\n", "\r", "\r\n", "\n\r", " ", "\t", "\r\n\r\n"])
    return unit * rng.randint(1, 30) + rng.choice(["", "x", "\n", "end"])


def _ascii_soup(rng: random.Random) -> str:
    alphabet = "abcXY .,;:!?'\"()-\t\r\n"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 220)))


_SHAPES = [
    _markdown,
    _heading_doc,
    _markdown_redteam,
    _markdown_crlf_headings,
    _chat_log,
    _crlf_prose,
    _repeated_literal,
    _degenerate,
    _ascii_soup,
]

# The separator pool: the default spelling, the empty hierarchy, plain
# literals (matching and never-matching), None splices at every
# interesting position, the LangChain-style "" sentinel, and the
# duplicate shapes (8x None, 8x " ", 3x "\n") whose dedup inertness the
# reference's no-dedup model pins.
_SEPARATOR_POOL: list[list[str | None] | None] = [
    None,
    [],
    ["\n"],
    ["\n\n"],
    [". "],
    [" "],
    ["---"],
    ["ZZZ_NEVER_MATCHES"],
    [None],
    ["\n", None],
    ["ZZZ_NEVER_MATCHES", None],
    ["---", None],
    [None, "---"],
    [None] * 8,
    [" "] * 8,
    ["\n"] * 3,
    ["", " "],
    ["\n", ". ", " "],
    ["\n## ", "\n\n", ". ", " "],
    ["\r\n", None, " "],
    # The unrealized-fine-level shapes: hierarchies where a COARSER
    # literal supplies the verdicts and a finer (or mutually overlapping)
    # literal stays unrealized at the final window — the pre-test's
    # literal arm answered "provably no" at `at > 0` there and pushed
    # pure-separator chunks (the `separator_pretest_literal_at_gt_zero_
    # may_open` pins; these pool entries keep the randomized sweep on
    # the shape).
    ["\n\n", "\n"],
    ["aa", "a"],
    ["X", "\n"],
    ["ab", "ba", "a", "b"],
    # The #63 heading level: the sentinel alone (heading → raw cut),
    # sentinel + splice (== the default hierarchy), the sentinel in
    # either position around a splice (dedup inertness), a heading
    # level under a line literal, and the sentinel with a never-match
    # above it.
    ["heading"],
    ["heading", None],
    [None, "heading"],
    ["\n", "heading", None],
    ["ZZZ_NEVER_MATCHES", "heading"],
]

_RANDOM_CASES = 3_000


class TestAsciiDifferential:
    def test_randomized_ascii_shapes_match_the_eager_reference_exactly(self) -> None:
        # The committed differential sweep: every shape x budget x overlap
        # x separator-list combination is an exact list equality against
        # the eager reference, so any divergence in the pyo3 layer, the
        # level construction, the cut filter, the window walk, the hard
        # cut, or the overlap snap fails with the full case in the
        # message. Budgets cover 1..~80 plus the whole-document budget
        # (the single-window path); overlaps cover 0, 1, 2, and max-1
        # (the largest legal snap-back). The sweep runs both
        # overlap_boundary modes (#47): the reference's word model is the
        # same UAX #29 word-bounds list production reads through the
        # hierarchy's word slot (or its one-off fallback for hierarchies
        # with no word level), so the word-mode cells pin the snap, the
        # decline-after-word-snap composition order, and the #103
        # final-exit skip against it exactly as the grapheme cells do.
        rng = random.Random(SEED)
        for _ in range(_RANDOM_CASES):
            text = rng.choice(_SHAPES)(rng)
            if rng.random() < 0.15:
                max_chars = max(1, len(text))
            else:
                max_chars = rng.randint(1, 80)
            overlap = rng.choice(
                [o for o in (0, 1, 2, max_chars - 1) if 0 <= o < max_chars]
            )
            separators = rng.choice(_SEPARATOR_POOL)
            overlap_boundary = rng.choice(["grapheme", "word"])
            chunks = tors.chunk_hierarchical(
                text,
                max_chars,
                separators=separators,
                overlap=overlap,
                overlap_boundary=overlap_boundary,
            )
            want = _ref_chunk_hierarchical(
                text, max_chars, separators, overlap, overlap_boundary
            )
            assert chunks == want, (
                f"chunk_hierarchical diverged from the reference: text={text!r} "
                f"max_chars={max_chars} overlap={overlap} "
                f"overlap_boundary={overlap_boundary!r} separators={separators!r}\n"
                f"  got : {chunks}\n  want: {want}"
            )

    def test_crlf_and_degenerate_text_survive_every_budget_one_through_twelve(self) -> None:
        # A deterministic dense sweep over the two shapes whose grapheme
        # model is non-trivial in ASCII (CRLF: GB3 joins every pair, so
        # the cut filter and the hard-cut snap must skip mid-pair
        # positions; degenerate runs: no levels fire at all, every cut is
        # the hard cut) x every small budget, where the window walk is
        # most sensitive, x the default and line-literal hierarchies.
        texts = [
            "one\r\ntwo\r\n\r\nthree\r\nfour",
            "\r\n\r\n\r\n",
            "a\r\r\nb\n\r\nc",
            "\n\n\n\n",
            "alpha\r\nbeta\r\ngamma\r\n\r\ndelta epsilon zeta eta theta.",
        ]
        for text in texts:
            for max_chars in range(1, 13):
                for separators in (None, ["\r\n"], ["\n", None]):
                    for overlap_boundary in ("grapheme", "word"):
                        chunks = tors.chunk_hierarchical(
                            text,
                            max_chars,
                            separators=separators,
                            overlap_boundary=overlap_boundary,
                        )
                        want = _ref_chunk_hierarchical(
                            text, max_chars, separators, 0, overlap_boundary
                        )
                        assert chunks == want, (
                            f"divergence: text={text!r} max_chars={max_chars} "
                            f"separators={separators!r} "
                            f"overlap_boundary={overlap_boundary!r}: "
                            f"{chunks} != {want}"
                        )


# ---------------------------------------------------------------------------
# Non-ASCII invariant checks. The ASCII reference's grapheme model does
# not hold here, so these are invariant pins (cluster-safe boundaries,
# budget modulo the documented oversized-cluster exception) rather than
# exact differentials; the cluster-safety predicate is deliberately
# conservative and independent of tors's own grapheme machinery: a
# boundary that would start mid-cluster is caught by the Unicode
# category of the character at the boundary (a combining mark,
# Mn/Mc/Me, never starts a cluster; neither does a ZWJ; and GB3's
# CRLF join is checked directly), with shape-specific exact pins where
# the cluster units are known by construction.
# ---------------------------------------------------------------------------


def _assert_cluster_safe(text: str, chunks: list[tuple[int, int]]) -> None:
    for s, e in chunks:
        for p in (s, e):
            if 0 < p < len(text):
                ch = text[p]
                prev = text[p - 1]
                # GB3: a CRLF pair is one cluster, so the '\n' half never
                # starts one.
                if prev == "\r" and ch == "\n":
                    raise AssertionError(f"boundary {p} splits a CRLF pair: {chunks}")
                # A backwards-joining character (a combining mark, or a
                # ZWJ) at a boundary is mid-cluster unless GB4 just forced
                # a break: after a Control/CR/LF codepoint the mark is an
                # orphan that legitimately starts its own cluster (the
                # documented newline-adjacent-mark non-issue), so the
                # exemption is exactly "the previous codepoint is Cc".
                joins_back = unicodedata.category(ch).startswith("M") or ch == "\u200d"
                if joins_back and unicodedata.category(prev) != "Cc":
                    raise AssertionError(
                        f"boundary {p} starts mid-cluster ({ch!r} after {prev!r}): {chunks}"
                    )


class TestNonAsciiInvariants:
    def test_thai_sara_am_runs_keep_every_cluster_whole(self) -> None:
        # "0" + SARA AM (U+0E33) is one cluster of exactly 2 codepoints
        # (the UAX #29 word/sentence divergence the merge steps and the
        # cut filter exist for): in a pure run of that unit every chunk
        # boundary must land at an even codepoint position (the exact
        # expectation, stronger than the conservative check), and a
        # chunk may exceed max_chars only by being exactly one cluster.
        text = "0ำ" * 40
        for max_chars in range(1, 12):
            for separators in (None, ["-"], ["ำ"], ["\n", None]):
                chunks = tors.chunk_hierarchical(
                    text, max_chars, separators=separators
                )
                for s, e in chunks:
                    assert s % 2 == 0 and e % 2 == 0, (
                        f"boundary off the 2-codepoint cluster grid: {chunks}"
                    )
                    if e - s > max_chars:
                        assert text[s:e] == "0ำ", (
                            f"oversized chunk is not one cluster: {text[s:e]!r}"
                        )
                _assert_cluster_safe(text, chunks)

    def test_zwj_emoji_sequences_keep_every_cluster_whole(self) -> None:
        # A ZWJ emoji family is one cluster of 5 codepoints (3 bases, 2
        # ZWJ); at budgets below 5 the oversized-cluster exception is
        # the only legal way past max_chars, and the excepted chunk must
        # be exactly the emoji unit.
        unit = "👨‍👩‍👧"
        text = unit + " " + unit + " end " + unit * 3
        for max_chars in range(1, 8):
            for overlap in (0, 2):
                if overlap >= max_chars:
                    continue
                chunks = tors.chunk_hierarchical(text, max_chars, overlap=overlap)
                _assert_cluster_safe(text, chunks)
                for s, e in chunks:
                    if e - s > max_chars:
                        assert text[s:e] == unit, (
                            f"oversized chunk is not one emoji cluster: {text[s:e]!r}"
                        )

    def test_cjk_text_respects_budgets_with_no_oversized_exception(self) -> None:
        # Every CJK character is its own cluster, so the documented
        # oversized-cluster exception has no room to fire: the budget
        # holds unconditionally, the strongest budget statement any
        # non-ASCII text can make.
        text = "中文数据段落。中文数据段落。" * 6 + "\n\n第二段落中文。"
        for max_chars in range(1, 12):
            for overlap in (0, 1, 3):
                if overlap >= max_chars:
                    continue
                chunks = tors.chunk_hierarchical(text, max_chars, overlap=overlap)
                _assert_cluster_safe(text, chunks)
                for s, e in chunks:
                    assert e - s <= max_chars, (
                        f"budget exceeded with no oversized cluster possible: {chunks}"
                    )

    def test_mixed_non_ascii_soup_keeps_boundaries_cluster_safe(self) -> None:
        # Thai, ZWJ emoji, CJK, CRLF, and a stray combining mark in one
        # seeded soup: the conservative predicate over arbitrary
        # adjacency, plus the forward-progress structure the fuzz target
        # also checks (starts strictly increase).
        rng = random.Random(SEED + 1)
        units = ["0ำ", "👨‍👩‍👧", "中", "文", "\r\n", " ", "e", "́", "\n\n"]
        for _ in range(120):
            text = "".join(rng.choice(units) for _ in range(rng.randint(1, 30)))
            max_chars = rng.randint(1, 20)
            chunks = tors.chunk_hierarchical(text, max_chars)
            _assert_cluster_safe(text, chunks)
            starts = [s for s, _e in chunks]
            assert starts == sorted(set(starts)), f"starts not strictly increasing: {chunks}"
