"""Differential gate for ``tors.chunk_hierarchical`` against a committed
pure-Python reference model, the port of the red-team audit's agent-B
reference (an eager-semantics verbatim reading of
``src/chunk_hierarchical_impl.rs``: the pre-#30 spelling, kept because
eager and lazy builds are output-equal by #30's own differential, so the
eager model pins the current function while describing the simpler
machine): paragraph ``windows(2)`` cuts, sentence/word cuts from
``tors.sentence_bounds``/``tors.word_bounds``, non-overlapping literal
matches with the literal dropped, the grapheme-cut filter, the
first-level-with-a-cut window walk, the grapheme-safe hard cut, and the
overlap snap.

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


def _ref_chunk_hierarchical(
    text: str,
    max_chars: int,
    separators: list[str | None] | None = None,
    overlap: int = 0,
) -> list[tuple[int, int]]:
    if text == "":
        return []
    total = len(text)
    gb = _ascii_grapheme_boundaries(text)
    if separators is None:
        levels = [
            _para_level_cuts(text),
            _contig_level_cuts(tors.sentence_bounds(text)),
            _contig_level_cuts(tors.word_bounds(text)),
        ]
    else:
        levels = []
        for entry in separators:
            if entry is None:
                levels.append(_para_level_cuts(text))
                levels.append(_contig_level_cuts(tors.sentence_bounds(text)))
                levels.append(_contig_level_cuts(tors.word_bounds(text)))
            elif entry == "":
                continue
            else:
                levels.append(_literal_level_cuts(text, entry))
    # The grapheme-cut filter: a level's cut is usable only when both its
    # end and its next-start are cluster boundaries.
    levels = [[(e, nx) for e, nx in cuts if gb[e] and gb[nx]] for cuts in levels]
    # Parallel strictly-increasing end arrays for the bisect below.
    ends = [[e for e, _ in cuts] for cuts in levels]

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

    chunks = []
    start = 0
    while start < total:
        remaining = total - start
        if remaining <= max_chars:
            chunks.append((start, total))
            break
        limit = start + max_chars
        # The first level whose largest in-limit cut is also past `start`
        # supplies the cut; later levels stay unconsulted.
        cut = None
        for cut_list, end_list in zip(levels, ends, strict=True):
            idx = bisect_right(end_list, limit) - 1
            if idx >= 0 and cut_list[idx][0] > start:
                cut = cut_list[idx]
                break
        if cut is None:
            end = last_at_or_before(limit)
            if end <= start:
                end = first_after(start)
            cut = (end, end)
        chunks.append((start, cut[0]))
        if overlap == 0:
            start = cut[1]
        else:
            target = max(cut[0] - overlap, 0)
            snapped = last_at_or_before(target)
            start = snapped if snapped > start else cut[1]
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


_SHAPES = [_markdown, _chat_log, _crlf_prose, _repeated_literal, _degenerate, _ascii_soup]

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
        # (the largest legal snap-back).
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
            chunks = tors.chunk_hierarchical(
                text, max_chars, separators=separators, overlap=overlap
            )
            want = _ref_chunk_hierarchical(text, max_chars, separators, overlap)
            assert chunks == want, (
                f"chunk_hierarchical diverged from the reference: text={text!r} "
                f"max_chars={max_chars} overlap={overlap} separators={separators!r}\n"
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
                    chunks = tors.chunk_hierarchical(
                        text, max_chars, separators=separators
                    )
                    want = _ref_chunk_hierarchical(text, max_chars, separators, 0)
                    assert chunks == want, (
                        f"divergence: text={text!r} max_chars={max_chars} "
                        f"separators={separators!r}: {chunks} != {want}"
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
