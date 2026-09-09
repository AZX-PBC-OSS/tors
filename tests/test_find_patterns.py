"""Contract gate for ``tors.find_patterns``: leftmost-longest multi-pattern
search at native speed, GIL-released.

``tors.find_patterns(patterns, text)`` finds the occurrences of a set of
substring patterns in one pass, reporting one triple per match:
``(start, end, pattern_index)`` where ``end`` is EXCLUSIVE, the offsets are
PYTHON STR INDICES (codepoints), and ``text[start:end] == patterns[index]``.

Semantics, pinned precisely (aho-corasick ``MatchKind::LeftmostLongest``, the
engine's documented "leftmost matches; when there are multiple possible
leftmost matches, the longest match is chosen"):

1. **Leftmost**: the scan proceeds left to right; a match is reported at the
   earliest position any pattern matches.
2. **Longest**: among the patterns matching at that position, the longest
   wins, regardless of its position in the ``patterns`` list (NOT regex
   alternation's leftmost-FIRST priority; a shorter earlier-listed pattern
   never beats a longer one). Two distinct patterns can only tie at a
   position by being byte-identical (both must equal the same text), so
   "longest" is always a strict winner except for exact duplicates.
3. **Non-overlapping**: the scan resumes at the END of each reported match;
   no two reported matches overlap, and they are reported in strictly
   increasing start order.
4. **Duplicates report the first index**: identical pattern strings are legal,
   and a match of that string reports the LOWEST index it occupies in the
   list.
5. **Offsets are characters, not bytes**: the automaton runs on UTF-8 bytes;
   whole-pattern matches land on character boundaries by construction (a
   valid UTF-8 sequence cannot begin mid-character), and a boundary-aware
   conversion pass maps the byte offsets to ``str`` indices. An ASCII
   ``text`` skips the pass (byte offset == char offset). The non-ASCII
   battery and the multi-byte hypothesis differentials below are the proof
   this mapping is right: the class of bug a byte-identity implementation
   would ship (``"café"`` matching at char 0 reporting end 5, not 4).

Contract decisions at the argument boundary (each pinned below):

- an empty pattern STRING raises ``ValueError("empty pattern")``: it would
  match at every position and has no leftmost-longest meaning;
- an empty patterns LIST returns ``[]`` immediately (no automaton build);
- non-``str`` entries in ``patterns`` and a non-``str`` ``text`` raise
  ``TypeError`` (``patterns`` must be exactly a ``list``, the annotation's
  type, so a tuple raises too);
- lone surrogates (a ``str`` CPython can hold but UTF-8 cannot encode) are
  refused with ``UnicodeEncodeError`` before any Rust code runs, the
  standard str-in boundary every tors function pays.

No stdlib oracle exists for these semantics: ``re`` alternation is
leftmost-FIRST (pattern priority), and leftmost-longest is precisely the
property it does not promise, so the contract is proven three ways, the
module decision's prescribed shape: (a) structural validity over arbitrary
generated inputs, (b) a brute-force pure-Python leftmost-longest reference
(the shared oracle, ``reference.reference_find_patterns`` in
tests/reference.py) over hypothesis-driven small alphabets (including
multi-byte ones, which is where the byte→char mapping is actually tested)
and (c) golden overlap cases with exact expected lists.

The GIL-release claim (automaton build + scan + offset conversion under one
``py.detach``; the O(matches) 3-tuple return marshalling measured on the
dense 12 MiB cell, the ``word_bounds`` list-shape precedent) is pinned in
``tests/test_gil_release.py``; the criterion ladder for the Rust core alone
is ``benches/search.rs``.

The count and streaming spellings ride the SAME contract, no new
semantics of their own; pinned below by parity with ``find_patterns``
itself (the strongest available oracle, the ``word_bounds_iter`` precedent):

- ``tors.count_matches(patterns, text) -> int``: the count spelling:
  ``count_matches(p, t) == len(find_patterns(p, t))`` over every input class
  below, with the SAME argument boundary (empty pattern string, list-exactly
  ``patterns``, str-exactly ``text``, the surrogate refusal) and the same
  leftmost-longest non-overlapping semantics, but O(1) memory: no match
  vector is built and no byte→char conversion pass runs (the count needs
  only the automaton and the scan, and returns one int (the
  ``grapheme_count`` no-marshalling shape, its GIL band the ping floor,
  measured in the cells of tests/test_gil_release.py).
- ``tors.find_patterns_iter(patterns, text)``: the streaming spelling: the
  whole scan runs eagerly under ONE detached pass when the iterator is
  CONSTRUCTED, the buffer is drained one 3-tuple per ``__next__``
  (µs-scale GIL holds), the yielded sequence is the list API's EXACT
  sequence, and ``__length_hint__`` reports the REMAINING count.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reference import SEARCH_DENSE_PATTERNS, prose, reference_find_patterns
from tors import count_matches, find_patterns, find_patterns_iter

_MIB = 1024 * 1024


def _assert_matches_are_valid(
    patterns: list[str], text: str, matches: list[tuple[int, int, int]]
) -> None:
    """Contract part (a), the structural validity every answer must satisfy
    regardless of semantics: in-range offsets, in-range pattern indices,
    strictly ordered non-overlapping spans, and content equality:
    ``text[start:end]`` IS ``patterns[idx]``."""
    prev_end = 0
    for start, end, idx in matches:
        assert 0 <= start < end <= len(text), f"span {(start, end)} outside the text"
        assert 0 <= idx < len(patterns), f"pattern index {idx} outside the pattern list"
        assert start >= prev_end, f"match {(start, end)} overlaps or precedes end {prev_end}"
        assert text[start:end] == patterns[idx], (
            f"content mismatch: text[{start}:{end}] != patterns[{idx}]"
        )
        prev_end = end


# --- Part (c): the golden overlap battery -------------------------------------------
#
# Both golden batteries live at module scope (the tests/test_segmentation.py
# ``_CASES`` convention) so the count/iter classes below can cross-check
# against the SAME rows without duplicating them.


_GOLDEN_OVERLAP_CASES: list[tuple[list[str], str, list[tuple[int, int, int]]]] = [
    (["abc", "abcd"], "abcd", [(0, 4, 1)]),
    (["abcd", "abc"], "abcd", [(0, 4, 0)]),
    (["bcd", "cd"], "abcd", [(1, 4, 0)]),
    (["bcd", "cd"], "abcdcd", [(1, 4, 0), (4, 6, 1)]),
    (["ab", "abc", "abcd"], "abcdabcab", [(0, 4, 2), (4, 7, 1), (7, 9, 0)]),
    (["abc", "abc"], "abcabc", [(0, 3, 0), (3, 6, 0)]),
    (["abcd", "abc", "abcd"], "abcd", [(0, 4, 0)]),
    (["aa", "aaaa"], "aaaaaa", [(0, 4, 1), (4, 6, 0)]),
    (["aaaa", "aa"], "aaa", [(0, 2, 1)]),
    (["a"], "aba", [(0, 1, 0), (2, 3, 0)]),
    ([], "abc", []),
    (["abc"], "", []),
    (["xyz"], "abc", []),
]

_GOLDEN_OVERLAP_IDS = [
    "longest-at-same-start",
    "longest-is-order-independent",
    "longer-beats-shorter-at-later-start",
    "scan-resumes-at-longest-end",
    "prefix-chain-across-text",
    "duplicates-report-first-index",
    "duplicated-longest-reports-first-index",
    "non-overlap-resumes-at-match-end",
    "longest-that-fits",
    "match-at-final-character",
    "empty-patterns-list",
    "empty-text",
    "no-matches",
]


@pytest.mark.parametrize(
    ("patterns", "text", "expected"),
    _GOLDEN_OVERLAP_CASES,
    ids=_GOLDEN_OVERLAP_IDS,
)
def test_golden_overlap_battery(
    patterns: list[str], text: str, expected: list[tuple[int, int, int]]
) -> None:
    """The fixed anchor of the contract: every golden case asserts the EXACT
    expected list and the oracle's agreement, so a hand-computed expectation
    that disagreed with the brute-force reference would fail loudly here
    rather than silently laundering a wrong pin into the suite."""
    matches = find_patterns(patterns, text)
    assert matches == expected
    assert matches == reference_find_patterns(patterns, text)
    _assert_matches_are_valid(patterns, text, matches)


# --- The byte→char offset-mapping battery (the correctness crux) --------------------


_E_ACUTE = chr(0xE9)  # precomposed é: 2 UTF-8 bytes
_COMBINING_ACUTE = chr(0x301)  # decomposed accent: 2 UTF-8 bytes, 1 char
_ZWJ = chr(0x200D)  # zero-width joiner: 3 bytes, 1 char
_VS16 = chr(0xFE0F)  # variation selector-16: 3 bytes, 1 char (a combining mark)
_FAMILY = (
    "\U0001f468" + _ZWJ + "\U0001f469" + _ZWJ + "\U0001f467"
)  # man-ZWJ-woman-ZWJ-girl: 5 chars, 18 bytes


# --- The byte→char offset-mapping battery (the correctness crux) --------------------


_MULTIBYTE_MAPPING_CASES: list[tuple[list[str], str, list[tuple[int, int, int]]]] = [
    (["café"], "café café", [(0, 4, 0), (5, 9, 0)]),
    (["cafe" + _COMBINING_ACUTE], "cafe" + _COMBINING_ACUTE + " ok", [(0, 5, 0)]),
    (["東京"], "京都東京大阪", [(2, 4, 0)]),
    (["東京", "京都"], "京都東京京都", [(0, 2, 1), (2, 4, 0), (4, 6, 1)]),
    ([_FAMILY], "hi" + _FAMILY + "!", [(2, 7, 0)]),
    (["\U0001f980" + _VS16], "\U0001f980" + _VS16 + "!", [(0, 2, 0)]),
    (["ab"], "éabéab", [(1, 3, 0), (4, 6, 0)]),
    (["éab"], "東京éab", [(2, 5, 0)]),
    # é b é b é, built by concatenation: a typed literal is visually
    # ambiguous (the crate-side battery's original "ébééb" pin was
    # mistyped é b é é b, caught by the oracle cross-check).
    (["b", "é"], "éb" * 2 + "é", [(0, 1, 1), (1, 2, 0), (2, 3, 1), (3, 4, 0), (4, 5, 1)]),
    (["éé", "é"], "ééé", [(0, 2, 0), (2, 3, 1)]),
    (["éé"], "éééé", [(0, 2, 0), (2, 4, 0)]),
]

_MULTIBYTE_MAPPING_IDS = [
    "precomposed-accents",
    "decomposed-accents",
    "cjk",
    "cjk-alternating-two-char-patterns",
    "emoji-zwj-family-five-chars",
    "emoji-with-variation-selector",
    "ascii-pattern-over-non-ascii-text",
    "multibyte-prefix-before-match",
    "multibyte-between-single-char-matches",
    "longest-two-char-over-multibyte",
    "back-to-back-multibyte-matches",
]


@pytest.mark.parametrize(
    ("patterns", "text", "expected"),
    _MULTIBYTE_MAPPING_CASES,
    ids=_MULTIBYTE_MAPPING_IDS,
)
def test_offsets_are_characters_not_bytes_over_multibyte_text(
    patterns: list[str], text: str, expected: list[tuple[int, int, int]]
) -> None:
    """THE mapping battery: every expected value was computed in CHARACTER
    units by hand; a byte-identity implementation (the bug class the module
    decision warns about) reports every one of these wrongly: e.g. ``"café"``
    matching at char 0 would report end 5 (bytes), not 4 (chars), and the
    ZWJ family (5 chars / 18 bytes) would report a 16-char span. The ASCII
    pattern over non-ASCII text row additionally proves the ``is_ascii`` fast
    path is correctly NOT taken (byte offsets and char offsets diverge there
    even though the patterns are pure ASCII)."""
    assert not text.isascii(), "the mapping battery's texts must all be non-ASCII"
    matches = find_patterns(patterns, text)
    assert matches == expected
    assert matches == reference_find_patterns(patterns, text)
    _assert_matches_are_valid(patterns, text, matches)


def test_the_ascii_fast_path_agrees_with_the_conversion_pass() -> None:
    """The fast path and the conversion pass must answer identically: the same
    pattern over an ASCII text (fast path, offsets pass through) and over a
    non-ASCII text containing the same ASCII core (conversion pass) produce
    the same CHARACTER answer."""
    assert find_patterns(["abc"], "xabcx") == [(1, 4, 0)]
    assert find_patterns(["abc"], "éabcé") == [(1, 4, 0)]


# --- The argument-boundary contract --------------------------------------------------


class TestArgumentContract:
    def test_empty_pattern_string_raises_value_error(self) -> None:
        """An empty pattern would match at every position, refused up front
        with exactly ``ValueError("empty pattern")``, wherever it sits in the
        list."""
        with pytest.raises(ValueError, match="^empty pattern$"):
            find_patterns(["ok", ""], "some text")
        with pytest.raises(ValueError, match="^empty pattern$"):
            find_patterns([""], "some text")

    def test_empty_patterns_list_returns_empty_list_without_scanning(self) -> None:
        """The early exit: no automaton is built, nothing is scanned, ``[]``
        for any text."""
        assert find_patterns([], "any text at all") == []

    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), 123, None],
        ids=["bytes", "bytearray", "int", "none"],
    )
    def test_non_str_pattern_entries_raise_type_error(self, not_str: object) -> None:
        """Exactly ``str`` entries (the str-exactly rule every tors str argument
        follows): pyo3's
        extraction rejects everything else with ``TypeError`` before any Rust
        code runs."""
        with pytest.raises(TypeError):
            find_patterns(["ok", not_str], "text")  # type: ignore[list-item]

    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), 123, None],
        ids=["bytes", "bytearray", "int", "none"],
    )
    def test_non_str_text_raises_type_error(self, not_str: object) -> None:
        with pytest.raises(TypeError):
            find_patterns(["ok"], not_str)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "not_list",
        [("ok", "abc"), "abc", b"abc", 123, None],
        ids=["tuple", "str", "bytes", "int", "none"],
    )
    def test_non_list_patterns_raise_type_error(self, not_list: object) -> None:
        """``patterns`` is exactly ``list[str]``, the annotation's type. A
        tuple of the right strings still raises ``TypeError``: the function
        takes the list shape it is annotated with (an iterable-friendly
        widening is a non-goal, like ``b64_decode``'s str-only
        spelling)."""
        with pytest.raises(TypeError):
            find_patterns(not_list, "text")  # type: ignore[arg-type]

    def test_lone_surrogates_are_refused_at_the_argument_boundary(self) -> None:
        """Lone surrogates (a ``str`` CPython can hold but UTF-8 cannot
        encode, e.g. from a ``surrogatepass`` decoder) are refused with
        ``UnicodeEncodeError`` before any Rust code runs, the standard
        str-in boundary (the ``finalize``/forms/diff pin), paid by the
        pattern entries and the text alike."""
        with pytest.raises(UnicodeEncodeError):
            find_patterns(["abc\ud800"], "abc")
        with pytest.raises(UnicodeEncodeError):
            find_patterns(["abc"], "abc\ud800")


# --- Parts (a)+(b): validity + the differential oracle over generated inputs --------


_SMALL_ALPHABET = "ab" + _E_ACUTE + _COMBINING_ACUTE + "東京🦀"


@st.composite
def _patterns_and_text(
    draw: st.DrawFn,
) -> tuple[list[str], str]:
    """Pattern lists (1-5 entries, duplicates allowed (the first-index
    contract needs them generated, not just pinned) and texts over an
    alphabet spanning every UTF-8 width: 1-byte ``ab``, 2-byte ``é`` and the
    combining accent, 3-byte CJK, 4-byte emoji. This is the strategy that
    makes the byte→char offset mapping face real coverage: matches start
    and end at every width combination."""
    patterns = draw(
        st.lists(
            st.text(alphabet=_SMALL_ALPHABET, min_size=1, max_size=4),
            min_size=1,
            max_size=5,
        )
    )
    text = draw(st.text(alphabet=_SMALL_ALPHABET, max_size=40))
    return patterns, text


@given(_patterns_and_text())
@settings(max_examples=500)
def test_matches_the_leftmost_longest_reference_over_multibyte_alphabets(
    patterns_text: tuple[list[str], str],
) -> None:
    """The differential proof of the offset mapping (part b): over the
    multi-byte alphabet, tors's answer must equal the brute-force char-space
    oracle EXACTLY (list equality, not just validity) for every generated
    pattern set and text. An offset-mapping bug of any kind (a boundary
    miscount, a fast path taken wrongly, a byte-for-char swap) breaks this
    property; the golden battery pins the individual shapes it finds."""
    patterns, text = patterns_text
    matches = find_patterns(patterns, text)
    _assert_matches_are_valid(patterns, text, matches)
    assert matches == reference_find_patterns(patterns, text)


@st.composite
def _ascii_text_patterns_and_text(
    draw: st.DrawFn,
) -> tuple[list[str], str]:
    """The fast path's own generator: ASCII TEXT (where the ``is_ascii`` bail
    skips the conversion pass) with patterns still drawn from the multi-byte
    alphabet: non-ASCII patterns over ASCII text can never match, and ASCII
    ones match with offsets that pass straight through, so both fast-path
    outcomes face the oracle."""
    patterns = draw(
        st.lists(
            st.text(alphabet=_SMALL_ALPHABET, min_size=1, max_size=4),
            min_size=1,
            max_size=5,
        )
    )
    text = draw(st.text(alphabet="abc", max_size=40))
    return patterns, text


@given(_ascii_text_patterns_and_text())
@settings(max_examples=200)
def test_matches_the_reference_when_the_ascii_fast_path_is_taken(
    patterns_text: tuple[list[str], str],
) -> None:
    """The ASCII fast path's differential: with a pure-ASCII text the offsets
    pass through unconverted, and the answers must still be exactly what the
    char-space oracle computes; the fast path cannot diverge from the
    conversion pass's semantics."""
    patterns, text = patterns_text
    assert text.isascii()
    matches = find_patterns(patterns, text)
    _assert_matches_are_valid(patterns, text, matches)
    assert matches == reference_find_patterns(patterns, text)


@st.composite
def _substring_patterns_and_text(
    draw: st.DrawFn,
) -> tuple[list[str], str]:
    """Arbitrary-Unicode texts (hypothesis's full ``st.text`` alphabet: any
    script, any marks, no alphabet bias) with pattern lists biased toward
    SUBSTRINGS of the text, so matches actually occur over text no small
    alphabet can generate."""
    text = draw(st.text(max_size=50))
    patterns: list[str] = []
    for _ in range(draw(st.integers(1, 4))):
        if text and draw(st.booleans()):
            i = draw(st.integers(0, len(text) - 1))
            j = draw(st.integers(i + 1, len(text)))
            patterns.append(text[i:j])
        else:
            patterns.append(draw(st.text(min_size=1, max_size=4)))
    return patterns, text


@given(_substring_patterns_and_text())
@settings(max_examples=300)
def test_matches_the_reference_over_arbitrary_unicode_with_substring_patterns(
    patterns_text: tuple[list[str], str],
) -> None:
    """The arbitrary-Unicode differential: substring-biased patterns over
    hypothesis's full text strategy (any codepoint class, mixed widths,
    combining marks anywhere), the coverage the fixed alphabets cannot
    reach, still under exact list equality with the oracle."""
    patterns, text = patterns_text
    matches = find_patterns(patterns, text)
    _assert_matches_are_valid(patterns, text, matches)
    assert matches == reference_find_patterns(patterns, text)


def test_every_pattern_list_over_a_tiny_alphabet_matches_the_reference() -> None:
    """The deterministic sweep (the suite's exhaustive-small-alphabet idiom):
    every pattern list of size 0-3 over ``{"a", "ab", "b"}`` (with
    repetition, duplicates included) crossed with every text over
    ``{"a", "b"}`` up to length 5: 2,520 pairs, the complete small space of
    prefix/overlap/duplicate interactions, no sampling at all."""
    pattern_pool = ["a", "ab", "b"]
    pattern_lists: list[list[str]] = [[]]
    for size in 1, 2, 3:
        pattern_lists.extend(list(combo) for combo in itertools.product(pattern_pool, repeat=size))
    texts: list[str] = []
    for length in range(6):
        texts.extend("".join(combo) for combo in itertools.product("ab", repeat=length))
    for patterns in pattern_lists:
        for text in texts:
            assert find_patterns(patterns, text) == reference_find_patterns(patterns, text)


# --- Wall cells ----------------------------------------------------------------------
#
# No stdlib racer exists (semantics above), so the wall discipline is absolute
# bands, the diff-test 1 MiB precedent: measured numbers with load disclosed,
# a ceiling with generous margin that any accidental quadratic or
# per-match-allocation regression blows straight through.


def _min_wall_ms(op: Callable[[], object], samples: int = 3, warmup: int = 1) -> float:
    """Min-of-``samples`` wall after ``warmup`` runs (the suite's shared
    methodology)."""
    for _ in range(warmup):
        op()
    best = float("inf")
    for _ in range(samples):
        started = time.perf_counter()
        op()
        best = min(best, time.perf_counter() - started)
    return best * 1000.0


# The 1 MiB dense-search absolute band: measured 16.2 ms min-of-3 after
# warm-up # (ambient load 7.3 on the dev box; 107,032 matches, so the wall is
# marshalling-dominated even at this size, ~0.13 µs per match, the same
# per-element band diff_opcodes measures). The 80 ms ceiling keeps ~5x
# margin over the measured wall, room for a loaded 2-vCPU CI runner, while
# an accidental per-match allocation regression (a 2x per-match cost) or a
# lost fast path blows straight through it.
_SEARCH_1MIB_CEILING_MS = 80.0


def test_find_patterns_absolute_wall_band_holds_at_1mib_dense() -> None:
    """The dense shape at 1 MiB (17 prose words, 107,032 matches): one native
    call: automaton build, scan, offset conversion (ASCII fast path), match
    vector fill, tuple marshalling. Measured on the dev box (WSL2, 28 logical
    cores, CPython 3.12; ambient load 7.3): 16.2 ms min-of-3 after warm-up
    against the 80 ms ceiling (~5x margin). No stdlib racer exists for these
    semantics (the module docstring's no-oracle rationale), so the wall
    discipline is the absolute band, the diff-test 1 MiB precedent."""
    text = prose(_MIB)
    patterns = list(SEARCH_DENSE_PATTERNS)
    tors_ms = _min_wall_ms(lambda: find_patterns(patterns, text))
    assert tors_ms < _SEARCH_1MIB_CEILING_MS, (
        f"1 MiB dense search: tors took {tors_ms:.1f}ms, outside the absolute "
        f"band (measured ~16ms, ceiling {_SEARCH_1MIB_CEILING_MS:.0f}ms with "
        "~5x margin); the native search pass regressed"
    )


# --- The count spelling ---------------------------------------------------------


class TestCountMatches:
    """The count spelling of the find contract: ``count_matches(p, t) ==
    len(find_patterns(p, t))``, the SAME leftmost-longest non-overlapping
    semantics under the SAME argument boundary, with none of the list API's
    O(matches) marshalling: the automaton build and scan run detached, no
    match vector is filled, no byte→char conversion pass runs, and the whole
    return is one int (the ``grapheme_count`` no-marshalling shape; its GIL
    band is the ping floor, measured in the cells of
    tests/test_gil_release.py).

    The oracle is ``find_patterns`` itself, the strongest available
    differential (the list API's own correctness is proven against
    ``reference_find_patterns`` above): count parity over the same hypothesis
    strategies, the same deterministic tiny-alphabet sweep, and the same
    golden batteries, plus the argument contract mirrored method-for-method
    from ``TestArgumentContract``."""

    @given(_patterns_and_text())
    @settings(max_examples=500)
    def test_count_equals_the_list_length_over_multibyte_alphabets(
        self, patterns_text: tuple[list[str], str]
    ) -> None:
        """The multibyte differential: the count spelling must agree with the
        list spelling over every generated pattern set and text; the count
        inherits the byte→char offset mapping's correctness obligations
        (a mis-mapped match end shifts where the non-overlapping scan resumes,
        so it changes the COUNT too, not just the offsets)."""
        patterns, text = patterns_text
        assert count_matches(patterns, text) == len(find_patterns(patterns, text))

    @given(_ascii_text_patterns_and_text())
    @settings(max_examples=200)
    def test_count_equals_the_list_length_when_the_ascii_fast_path_is_taken(
        self, patterns_text: tuple[list[str], str]
    ) -> None:
        """The ASCII fast path's differential: the count over pure-ASCII text
        (where the list API's offsets pass through unconverted) must equal the
        list length, and the count spelling must match the list spelling's
        choice of fast path, whatever that choice is."""
        patterns, text = patterns_text
        assert text.isascii()
        assert count_matches(patterns, text) == len(find_patterns(patterns, text))

    @given(_substring_patterns_and_text())
    @settings(max_examples=300)
    def test_count_equals_the_list_length_over_arbitrary_unicode(
        self, patterns_text: tuple[list[str], str]
    ) -> None:
        """The arbitrary-Unicode differential: substring-biased patterns over
        hypothesis's full text strategy, the coverage class the fixed
        alphabets cannot reach, still under exact count parity."""
        patterns, text = patterns_text
        assert count_matches(patterns, text) == len(find_patterns(patterns, text))

    def test_count_equals_the_reference_over_the_tiny_alphabet_sweep(self) -> None:
        """The deterministic sweep, the list API's own exhaustive idiom: every
        pattern list of size 0-3 over ``{"a", "ab", "b"}`` (duplicates
        included) crossed with every text over ``{"a", "b"}`` up to length 5:
        the count equals BOTH the list API's length and the brute-force
        oracle's length for all 2,520 pairs, no sampling at all."""
        pattern_pool = ["a", "ab", "b"]
        pattern_lists: list[list[str]] = [[]]
        for size in 1, 2, 3:
            pattern_lists.extend(
                list(combo) for combo in itertools.product(pattern_pool, repeat=size)
            )
        texts: list[str] = []
        for length in range(6):
            texts.extend("".join(combo) for combo in itertools.product("ab", repeat=length))
        for patterns in pattern_lists:
            for text in texts:
                count = count_matches(patterns, text)
                assert count == len(find_patterns(patterns, text))
                assert count == len(reference_find_patterns(patterns, text))

    @pytest.mark.parametrize(
        ("patterns", "text", "expected"),
        _GOLDEN_OVERLAP_CASES + _MULTIBYTE_MAPPING_CASES,
        ids=_GOLDEN_OVERLAP_IDS + _MULTIBYTE_MAPPING_IDS,
    )
    def test_count_matches_the_golden_batteries_expected_lengths(
        self, patterns: list[str], text: str, expected: list[tuple[int, int, int]]
    ) -> None:
        """The golden cross-check: every hand-pinned row of BOTH batteries
        (the overlap semantics and the byte→char mapping crux) must count to
        exactly the pinned list's length."""
        assert count_matches(patterns, text) == len(expected)
        assert count_matches(patterns, text) == len(find_patterns(patterns, text))

    def test_empty_pattern_string_raises_value_error(self) -> None:
        """Mirrored from ``TestArgumentContract``: an empty pattern would
        match at every position, refused up front with exactly
        ``ValueError("empty pattern")``, wherever it sits in the list."""
        with pytest.raises(ValueError, match="^empty pattern$"):
            count_matches(["ok", ""], "some text")
        with pytest.raises(ValueError, match="^empty pattern$"):
            count_matches([""], "some text")

    def test_empty_patterns_list_returns_zero_without_scanning(self) -> None:
        """The early exit: no automaton is built, nothing is scanned, ``0``
        for any text."""
        assert count_matches([], "any text at all") == 0

    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), 123, None],
        ids=["bytes", "bytearray", "int", "none"],
    )
    def test_non_str_pattern_entries_raise_type_error(self, not_str: object) -> None:
        with pytest.raises(TypeError):
            count_matches(["ok", not_str], "text")  # type: ignore[list-item]

    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), 123, None],
        ids=["bytes", "bytearray", "int", "none"],
    )
    def test_non_str_text_raises_type_error(self, not_str: object) -> None:
        with pytest.raises(TypeError):
            count_matches(["ok"], not_str)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "not_list",
        [("ok", "abc"), "abc", b"abc", 123, None],
        ids=["tuple", "str", "bytes", "int", "none"],
    )
    def test_non_list_patterns_raise_type_error(self, not_list: object) -> None:
        """``patterns`` is exactly ``list[str]``, the annotation's type, the
        list API's own boundary, mirrored verbatim."""
        with pytest.raises(TypeError):
            count_matches(not_list, "text")  # type: ignore[arg-type]

    def test_lone_surrogates_are_refused_at_the_argument_boundary(self) -> None:
        """The standard str-in boundary, paid by the pattern entries and the
        text alike (the ``finalize``/forms/diff pin)."""
        with pytest.raises(UnicodeEncodeError):
            count_matches(["abc\ud800"], "abc")
        with pytest.raises(UnicodeEncodeError):
            count_matches(["abc"], "abc\ud800")


# --- The streaming spelling -----------------------------------------------------


class TestFindPatternsIter:
    """The streaming spelling of the find contract (the ``word_bounds_iter``
    design, reused verbatim): the whole search (automaton build, scan, offset
    conversion, match buffer fill) runs under ONE detached pass when the
    iterator is CONSTRUCTED, and each ``__next__`` then holds the GIL only to
    hand back ONE ``(start, end, pattern_index)`` 3-tuple. The yielded
    sequence is the list API's EXACT sequence, and ``__length_hint__`` reports
    the REMAINING count, pinned against ``find_patterns`` itself, the
    strongest available oracle, over the same strategies, sweeps, and golden
    batteries as the list API (the count class above adds the same parity for
    ``count_matches``). The GIL band of construction-plus-drain is measured
    in the cells of tests/test_gil_release.py."""

    @given(_patterns_and_text())
    @settings(max_examples=500)
    def test_sequence_parity_over_multibyte_alphabets(
        self, patterns_text: tuple[list[str], str]
    ) -> None:
        """The multibyte differential: the materialized iterator sequence must
        equal the list API's answer EXACTLY (list equality, not just element
        validity) for every generated pattern set and text."""
        patterns, text = patterns_text
        assert list(find_patterns_iter(patterns, text)) == find_patterns(patterns, text)

    @given(_ascii_text_patterns_and_text())
    @settings(max_examples=200)
    def test_sequence_parity_when_the_ascii_fast_path_is_taken(
        self, patterns_text: tuple[list[str], str]
    ) -> None:
        """The ASCII fast path's differential: same parity requirement over
        pure-ASCII text, where the offsets pass through unconverted."""
        patterns, text = patterns_text
        assert text.isascii()
        assert list(find_patterns_iter(patterns, text)) == find_patterns(patterns, text)

    @given(_substring_patterns_and_text())
    @settings(max_examples=300)
    def test_sequence_parity_over_arbitrary_unicode_with_substring_patterns(
        self, patterns_text: tuple[list[str], str]
    ) -> None:
        """The arbitrary-Unicode differential: substring-biased patterns over
        hypothesis's full text strategy, the coverage class the fixed
        alphabets cannot reach."""
        patterns, text = patterns_text
        assert list(find_patterns_iter(patterns, text)) == find_patterns(patterns, text)

    def test_sequence_parity_over_the_tiny_alphabet_sweep(self) -> None:
        """The deterministic sweep, the list API's own exhaustive idiom: every
        pattern list of size 0-3 over ``{"a", "ab", "b"}`` crossed with every
        text over ``{"a", "b"}`` up to length 5: the streamed sequence
        equals the list API's answer for all 2,520 pairs."""
        pattern_pool = ["a", "ab", "b"]
        pattern_lists: list[list[str]] = [[]]
        for size in 1, 2, 3:
            pattern_lists.extend(
                list(combo) for combo in itertools.product(pattern_pool, repeat=size)
            )
        texts: list[str] = []
        for length in range(6):
            texts.extend("".join(combo) for combo in itertools.product("ab", repeat=length))
        for patterns in pattern_lists:
            for text in texts:
                assert list(find_patterns_iter(patterns, text)) == find_patterns(patterns, text)

    @pytest.mark.parametrize(
        ("patterns", "text", "expected"),
        _GOLDEN_OVERLAP_CASES + _MULTIBYTE_MAPPING_CASES,
        ids=_GOLDEN_OVERLAP_IDS + _MULTIBYTE_MAPPING_IDS,
    )
    def test_yields_the_list_apis_exact_sequence_on_the_golden_batteries(
        self, patterns: list[str], text: str, expected: list[tuple[int, int, int]]
    ) -> None:
        """The golden cross-check: every hand-pinned row of BOTH batteries
        must stream back exactly the pinned list."""
        assert list(find_patterns_iter(patterns, text)) == find_patterns(patterns, text) == expected

    def test_the_iterator_protocol_shapes(self) -> None:
        it = find_patterns_iter(["ab", "b"], "abab")
        expected = find_patterns(["ab", "b"], "abab")
        assert iter(it) is it  # an iterator: iter() hands back the same object
        assert next(it) == expected[0]
        # Mid-iteration resumption: the remainder is exactly the list API's
        # remainder.
        assert list(it) == expected[1:]
        with pytest.raises(StopIteration):
            next(it)  # exhaustion is stable, not one-shot

    def test_length_hint_tracks_partial_consumption(self) -> None:
        """``__length_hint__`` (what ``list()``/``tuple()`` preallocation and
        the interpreter's own optimizations consult) is the REMAINING count:
        the full count at construction, decremented by each ``next()``, zero
        at exhaustion, never the original length after consumption."""
        patterns = ["a", "b"]
        text = "ababab"
        expected = find_patterns(patterns, text)
        it = find_patterns_iter(patterns, text)
        assert it.__length_hint__() == len(expected)
        for consumed, _ in enumerate(expected, start=1):
            next(it)
            assert it.__length_hint__() == len(expected) - consumed
        assert it.__length_hint__() == 0
        with pytest.raises(StopIteration):
            next(it)
        assert it.__length_hint__() == 0  # stable at exhaustion

    def test_each_yield_is_a_three_tuple_not_a_list(self) -> None:
        """The marshalling shape each ``__next__`` pays for: a built-in
        3-tuple of three ints, never a list (a list yield would make
        ``==`` comparisons against the list API's tuples fail, and would
        signal a per-next list construction path)."""
        yielded = list(find_patterns_iter(["ab", "b"], "abab"))
        assert yielded == find_patterns(["ab", "b"], "abab")
        for match in yielded:
            assert type(match) is tuple
            assert not isinstance(match, list)
            assert len(match) == 3
            assert all(type(part) is int for part in match)

    def test_empty_patterns_list_yields_an_empty_iterator(self) -> None:
        """The early exit's streaming spelling: an empty iterator with a
        length hint of 0: no automaton is built, nothing is scanned."""
        it = find_patterns_iter([], "any text at all")
        assert list(it) == []
        assert it.__length_hint__() == 0

    def test_corpus_shaped_text_parities_with_the_list_api(self) -> None:
        # A prose-shaped slab (the corpus unit the GIL/wall cells measure),
        # non-ASCII included: the streaming and list spellings must agree on
        # realistic sizes, not just table rows.
        slab = "caf\u00e9 quarterly \u00e9ab sample. " * 400 + "\n\n"
        patterns = ["quarterly", "sample", "\u00e9ab", "caf\u00e9"]
        assert list(find_patterns_iter(patterns, slab)) == find_patterns(patterns, slab)

    def test_empty_pattern_string_raises_value_error(self) -> None:
        """The list API's boundary, mirrored verbatim: an empty pattern is
        refused at CONSTRUCTION time (the eager pass happens then), exactly
        ``ValueError("empty pattern")``."""
        with pytest.raises(ValueError, match="^empty pattern$"):
            find_patterns_iter(["ok", ""], "some text")
        with pytest.raises(ValueError, match="^empty pattern$"):
            find_patterns_iter([""], "some text")

    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), 123, None],
        ids=["bytes", "bytearray", "int", "none"],
    )
    def test_non_str_pattern_entries_raise_type_error(self, not_str: object) -> None:
        with pytest.raises(TypeError):
            find_patterns_iter(["ok", not_str], "text")  # type: ignore[list-item]

    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), 123, None],
        ids=["bytes", "bytearray", "int", "none"],
    )
    def test_non_str_text_raises_type_error(self, not_str: object) -> None:
        with pytest.raises(TypeError):
            find_patterns_iter(["ok"], not_str)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "not_list",
        [("ok", "abc"), "abc", b"abc", 123, None],
        ids=["tuple", "str", "bytes", "int", "none"],
    )
    def test_non_list_patterns_raise_type_error(self, not_list: object) -> None:
        """``patterns`` is exactly ``list[str]``, the annotation's type, the
        list API's own boundary, mirrored verbatim (the eager detached pass
        never runs: extraction fails first)."""
        with pytest.raises(TypeError):
            find_patterns_iter(not_list, "text")  # type: ignore[arg-type]

    def test_lone_surrogates_are_refused_at_the_argument_boundary(self) -> None:
        """The standard str-in boundary, paid by the pattern entries and the
        text alike (the ``finalize``/forms/diff pin)."""
        with pytest.raises(UnicodeEncodeError):
            find_patterns_iter(["abc\ud800"], "abc")
        with pytest.raises(UnicodeEncodeError):
            find_patterns_iter(["abc"], "abc\ud800")
