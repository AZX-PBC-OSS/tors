"""Contract gate for the sentence surface: ``tors.sentence_bounds`` and
``tors.sentence_bounds_iter``: UAX #29 sentence-boundary segmentation, the
capability the stdlib LACKS (the gap is the point: ``unicodedata`` exposes
the properties but no segmenter, exactly as for graphemes and words; the
crate backing this surface, unicode-segmentation 1.13.3, implements rules
SB1-SB999 over Unicode 17.0.0 tables, and every rule cited below has been
stable across Unicode 11-17).

No stdlib oracle exists, so the pins are the segmentation gate's shape
(tests/test_segmentation.py): a HAND-DERIVED table of expected values for
the tricky cases, each derived from its cited UAX #29 rule BEFORE running
it against the implementation (every row below was also verified against
the crate-side battery in src/segmentation_impl.rs; no row disagreed),
plus STRUCTURAL properties over hypothesis-generated text that must hold
for any conforming segmenter, plus the list/iter sequence parity.

Presentation quirks and limits, pinned rather than hidden:

- **Trailing spaces belong to the PRECEDING sentence** (SB9-SB11): the
  boundary after a terminator lands only after the terminator's trailing
  ``Close* Sp*``; SB9/SB10 keep closing punctuation and spaces attached,
  SB11 then breaks before the next non-space. ``"One. Two."`` segments as
  ``"One. "`` + ``"Two."``, the first sentence CARRYING the inter-sentence
  space. That is what the rules say the boundary IS, not a crate artifact;
  trim at the call site if trimmed sentences are wanted.
- **Rule-based only**: UAX #29 sentence segmentation is rule-driven and
  carries no dictionary segmentation for spaceless scripts
  (Thai/Khmer/Burmese/Japanese sentence-adjacent shapes are a
  dictionary-based problem ICU4X addresses with a heavyweight model); the
  limitation note shared with ``word_bounds``, recorded here so no caller
  mistakes this for an ICU-grade segmenter.

The API contract decisions pinned here (from the segmentation
precedents, reused verbatim):

- OFFSETS, never string lists: ``(start, end)`` pairs in PYTHON STR INDEX
  units (codepoints), so ``text[start:end]`` IS the sentence;
- ``sentence_bounds_iter`` is the streaming spelling (the
  ``word_bounds_iter`` design): one detached whole-text pass at
  construction, the SAME sequence as the list API per ``__next__``, and
  ``__length_hint__`` reporting the REMAINING count, pinned to exact
  sequence parity below, its GIL band in tests/test_gil_release.py.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import sentence_bounds, sentence_bounds_iter, sentence_count

# Built from codepoints (pure-ASCII source, per the suite's convention).
_PS = chr(0x2029)  # PARAGRAPH SEPARATOR: SB4's Sep class
_RDQ = chr(0x201D)  # RIGHT DOUBLE QUOTATION MARK: Close in SB9's sense
_IDEOGRAPHIC_STOP = chr(0x3002)  # 。: STerm (Sentence_Terminal=Yes)
_TOKYO = chr(0x6771) + chr(0x4EAC)  # 東京
_OSAKA = chr(0x5927) + chr(0x962A)  # 大阪
_CJK_ROW = _TOKYO + _IDEOGRAPHIC_STOP + _OSAKA + _IDEOGRAPHIC_STOP  # 東京。大阪。

# (text, expected bounds): every row's value derived from the cited rule
# BEFORE running it against the implementation, then verified crate-side
# (src/segmentation_impl.rs pins the same rows); no row disagreed.
_CASES: list[tuple[str, list[tuple[int, int]]]] = [
    # SB3: CR × LF: the CRLF never splits; SB4 then breaks AFTER the LF,
    # so the separator rides the first sentence and "b." is its own.
    ("a\r\nb.", [(0, 3), (3, 5)]),
    # SB4: a break after each paragraph separator (CR, LF, Sep): the
    # separator stays with the preceding text, no terminator required.
    ("a\rb.", [(0, 2), (2, 4)]),
    ("line one\nline two", [(0, 9), (9, 17)]),
    ("one" + _PS + "two", [(0, 4), (4, 7)]),
    # SB6: ATerm × Numeric: the '.' before a digit is a decimal point.
    ("3.4 percent", [(0, 11)]),
    # SB7: (Upper | Lower) ATerm × Upper: initials stay inside the word.
    ("U.S.A", [(0, 5)]),
    # SB8: ATerm Close* Sp* × (¬(OLetter|Upper|Lower|...))* Lower: an
    # ambiguous '.' before lowercase continues the sentence.
    ("etc. and so on", [(0, 14)]),
    # SB9/SB10/SB11: the terminator's Close (the plain quote here, the
    # U+201D right double quotation mark below, Line_Break=Quotation, so
    # Close in SB9's sense) and Sp* attach to the PRECEDING sentence; SB11
    # breaks before "Now" (Upper, so SB8's lowercase continuation does not
    # apply). Both rows span (0, 16) / (16, 23): the quote at 14 and the
    # space at 15 ride sentence 1.
    ('He said "stop." Now go.', [(0, 16), (16, 23)]),
    ("He said " + _RDQ + "stop." + _RDQ + " Now go.", [(0, 16), (16, 23)]),
    # The trailing-space presentation quirk, rule-derived and pinned:
    # SB10 keeps the inter-sentence space with sentence 1, and SB11 breaks
    # only before the non-space "T": "One. " + "Two.". The first segment
    # CARRIES the space; that is the rule, not a bug.
    ("One. Two.", [(0, 5), (5, 9)]),
    # SB11 over an ideographic terminator: U+3002 is STerm
    # (Sentence_Terminal=Yes), so each 。 ends its sentence in place.
    (_CJK_ROW, [(0, 3), (3, 6)]),
    # SB8a: SATerm Close* Sp* × (SContinue | SATerm): "?!" is ONE
    # terminator run, not two sentences; SB10/SB11 then attach the space
    # and break before "Yes".
    ("Wow?! Yes.", [(0, 6), (6, 10)]),
    ("Stop! Go.", [(0, 6), (6, 9)]),
    # Degenerates: empty is empty; spaces-only and terminator-free text
    # are each ONE segment: SB998 joins Any × Any, and no interior rule
    # ever fires, so only SB1/SB2 bound the text.
    ("", []),
    ("   ", [(0, 3)]),
    ("no terminator here", [(0, 18)]),
]


class TestHandDerivedTable:
    @pytest.mark.parametrize(
        ("text", "bounds"),
        _CASES,
        ids=[
            f"{t[:12]!a}" if len(t) <= 12 else f"{t[:6]!a}...{len(t)}" for t, _ in _CASES
        ],
    )
    def test_matches_the_uax29_derived_table(
        self, text: str, bounds: list[tuple[int, int]]
    ) -> None:
        """Every hand-derived row: the full sentence-bounds list (Python str
        indices; ``text[a:b]`` is the sentence) must equal what the cited
        UAX #29 rule produces for that input. Derived from the rules, not
        from the implementation: if the crate disagrees with a row, either
        the derivation or the crate is wrong, and this test is where that
        becomes visible instead of silent."""
        assert sentence_bounds(text) == bounds, f"sentence_bounds for {text!a}"
        assert list(sentence_bounds_iter(text)) == bounds, f"sentence_bounds_iter for {text!a}"


class TestStructuralProperties:
    @given(st.text(alphabet=st.characters(exclude_categories=("Cn", "Cs")), max_size=48))
    @settings(max_examples=400)
    def test_bounds_are_monotonic_covering_and_round_trip(self, text: str) -> None:
        """The structural promise any conforming segmenter makes: bounds
        start at 0 and end at len(text), are strictly increasing (each
        segment non-empty, none overlapping), adjacent bounds touch (each
        start == the previous end, which is stronger than non-overlap),
        and joining the sliced segments reproduces the input byte-for-byte,
        the round-trip that makes the offsets API trustworthy
        (``text[a:b]`` really is the sentence)."""
        bounds = sentence_bounds(text)
        assert all(a < b for a, b in bounds)
        assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))
        if text:
            assert bounds[0][0] == 0
            assert bounds[-1][1] == len(text)
        assert "".join(text[a:b] for a, b in bounds) == text


class TestArgumentContract:
    @pytest.mark.parametrize(
        "fn_name", ["sentence_bounds", "sentence_bounds_iter", "sentence_count"]
    )
    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), memoryview(b"abc"), 123, None],
        ids=["bytes", "bytearray", "memoryview", "int", "none"],
    )
    def test_non_str_arguments_raise_type_error(
        self, fn_name: str, not_str: object
    ) -> None:
        fn = {
            "sentence_bounds": sentence_bounds,
            "sentence_bounds_iter": sentence_bounds_iter,
            "sentence_count": sentence_count,
        }[fn_name]
        with pytest.raises(TypeError):
            fn(not_str)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "fn_name", ["sentence_bounds", "sentence_bounds_iter", "sentence_count"]
    )
    def test_lone_surrogates_are_refused_at_the_argument_boundary(
        self, fn_name: str
    ) -> None:
        """Same pyo3 ``&str`` boundary as every str-in function (pinned for
        ``finalize`` in tests/test_finalize.py, for the segmentation
        surface in tests/test_segmentation.py)."""
        fn = {
            "sentence_bounds": sentence_bounds,
            "sentence_bounds_iter": sentence_bounds_iter,
            "sentence_count": sentence_count,
        }[fn_name]
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            fn("ok\ud800ok")  # type: ignore[arg-type]


class TestSentenceBoundsIter:
    """The streaming spelling of ``sentence_bounds`` (the
    ``word_bounds_iter`` design, reused verbatim): the whole segmentation
    runs under one detached pass when the iterator is constructed, and each
    ``__next__`` holds the GIL only for one 2-tuple. Correctness is pinned
    against the list API itself: same bounds sequence over every battery
    row, arbitrary hypothesis text, and a corpus-shaped slab, the strongest
    available oracle given no stdlib segmenter exists."""

    @pytest.mark.parametrize(
        ("text", "bounds"),
        _CASES,
        ids=[f"case{i}" for i in range(len(_CASES))],
    )
    def test_yields_the_list_apis_exact_sequence_on_every_tricky_row(
        self, text: str, bounds: list[tuple[int, int]]
    ) -> None:
        assert list(sentence_bounds_iter(text)) == sentence_bounds(text) == bounds

    def test_the_iterator_protocol_shapes(self) -> None:
        it = sentence_bounds_iter("One. Two.")
        assert iter(it) is it  # an iterator: iter() hands back the same object
        assert next(it) == (0, 5)
        # Mid-iteration resumption: the remainder is exactly the list API's
        # remainder.
        assert list(it) == sentence_bounds("One. Two.")[1:]
        with pytest.raises(StopIteration):
            next(it)  # exhaustion is stable, not one-shot

    def test_length_hint_tracks_partial_consumption(self) -> None:
        """``__length_hint__`` (what ``list()``/``tuple()`` preallocation and
        the interpreter's own optimizations consult) is the REMAINING count:
        the full count at construction, decremented by each ``next()``, zero
        at exhaustion, never the original length after consumption."""
        expected = sentence_bounds("One. Two. Three.")
        it = sentence_bounds_iter("One. Two. Three.")
        assert it.__length_hint__() == len(expected)
        for consumed, _ in enumerate(expected, start=1):
            next(it)
            assert it.__length_hint__() == len(expected) - consumed
        assert it.__length_hint__() == 0
        with pytest.raises(StopIteration):
            next(it)
        assert it.__length_hint__() == 0  # stable at exhaustion

    def test_empty_text_yields_nothing(self) -> None:
        assert list(sentence_bounds_iter("")) == []
        assert sentence_bounds_iter("").__length_hint__() == 0

    def test_corpus_shaped_text_parities_with_the_list_api(self) -> None:
        # A prose-shaped slab (the corpus unit the GIL/wall cells measure),
        # non-ASCII included: the streaming and list spellings must agree on
        # realistic sizes, not just table rows.
        slab = ("caf\u00e9 na\u00efve quarter\u0301 sample. " * 400) + "\n\n"
        assert list(sentence_bounds_iter(slab)) == sentence_bounds(slab)

    @given(st.text(alphabet=st.characters(exclude_categories=("Cn", "Cs")), max_size=48))
    @settings(max_examples=400)
    def test_sequence_parity_over_arbitrary_text(self, text: str) -> None:
        assert list(sentence_bounds_iter(text)) == sentence_bounds(text)

    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), memoryview(b"abc"), 123, None],
        ids=["bytes", "bytearray", "memoryview", "int", "none"],
    )
    def test_non_str_arguments_raise_type_error(self, not_str: object) -> None:
        with pytest.raises(TypeError):
            sentence_bounds_iter(not_str)  # type: ignore[arg-type]

    def test_lone_surrogates_are_refused_at_the_argument_boundary(self) -> None:
        """Same pyo3 ``&str`` boundary as every str-in function (pinned for
        ``finalize`` in tests/test_finalize.py)."""
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            sentence_bounds_iter("ok\ud800ok")  # type: ignore[arg-type]


class TestSentenceCount:
    """The count spelling of the sentence contract: ``sentence_count(t)
    == len(sentence_bounds(t))``, the SAME UAX #29 sentence segmentation
    under the SAME argument boundary, with none of the list API's
    O(sentences) marshalling: the whole rule walk runs detached and the
    return is one int (the ``grapheme_count`` no-marshalling shape; its GIL
    band is the ping floor, the grapheme_count cells' band, measured in
    tests/test_gil_release.py).

    The oracle is ``sentence_bounds`` itself, the strongest available
    differential given no stdlib segmenter exists (the list API's own
    correctness is proven against the hand-derived UAX #29 table above):
    count parity over every tricky row, arbitrary hypothesis text, and the
    corpus shapes, plus the degenerates and the argument contract (mirrored
    through ``TestArgumentContract``'s shared parametrizations, which
    ``sentence_count`` joins), the ``word_count`` gate's shape, reused
    verbatim."""

    @pytest.mark.parametrize(
        ("text", "bounds"),
        _CASES,
        ids=[
            f"{t[:12]!a}" if len(t) <= 12 else f"{t[:6]!a}...{len(t)}" for t, _ in _CASES
        ],
    )
    def test_count_equals_the_list_length_on_every_tricky_row(
        self, text: str, bounds: list[tuple[int, int]]
    ) -> None:
        """Every hand-derived UAX #29 row: the count spelling must agree with
        the table's pinned bounds length; the tricky shapes (CRLF, the
        SB6-SB8 ambiguity rules, the Close* Sp* attachment, the ideographic
        STerm rows) are exactly where a count fast path could diverge from
        the segmentation the list API pins."""
        assert sentence_count(text) == len(bounds), f"sentence_count for {text!a}"
        assert sentence_count(text) == len(sentence_bounds(text)), (
            f"sentence_count for {text!a}"
        )

    @given(st.text(alphabet=st.characters(exclude_categories=("Cn", "Cs")), max_size=48))
    @settings(max_examples=400)
    def test_count_equals_the_list_length_over_arbitrary_text(self, text: str) -> None:
        """The arbitrary-Unicode differential over the module's existing text
        strategy: the count equals the list API's length for every generated
        text."""
        assert sentence_count(text) == len(sentence_bounds(text))

    def test_the_degenerates(self) -> None:
        """The fixed degenerate anchors, the list battery's own rows: empty
        is 0; spaces-only and terminator-free text are each ONE segment
        (SB998 joins Any × Any, and no interior rule ever fires, so only
        SB1/SB2 bound the text; the count spelling must not silently
        re-scope "sentence" to "terminator-delimited run", which would call
        both of these 0)."""
        assert sentence_count("") == 0
        assert sentence_count("   ") == 1
        assert sentence_count("no terminator here") == 1

    def test_corpus_shaped_text_parities_with_the_list_api(self) -> None:
        # A prose-shaped slab (the corpus unit the GIL/wall cells measure),
        # non-ASCII included: the count and list spellings must agree on
        # realistic sizes, not just table rows.
        slab = ("caf\u00e9 na\u00efve quarter\u0301 sample. " * 400) + "\n\n"
        assert sentence_count(slab) == len(sentence_bounds(slab))
