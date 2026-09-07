"""Contract gate for the segmentation surface: ``tors.grapheme_count``
and ``tors.word_bounds``: UAX #29 extended-grapheme-cluster counting and
word-boundary segmentation, the capability the stdlib LACKS (the gap is the
point: ``unicodedata`` exposes the properties but no segmenter, and every
pure-Python grapheme/word segmenter is either a third-party dependency or a
slow charclass walk).

No stdlib oracle exists, so the pins are of a different shape than the parity
gates: a HAND-DERIVED table of expected values for the tricky cases, each
derived from the Unicode segmentation rules (Unicode Standard Annex #29,
"Unicode Text Segmentation"; grapheme-cluster rules GB1-GB999, word-boundary
rules WB1-WB999; the crate backing this surface, unicode-segmentation 1.13.3,
implements them over Unicode 17.0.0 tables, and every rule cited below has
been stable across Unicode 11-17), plus STRUCTURAL properties over
hypothesis-generated text that must hold for any conforming segmenter:
monotonic strictly-increasing bounds, covering [0, len], a byte-exact
round-trip (joining the sliced segments reproduces the input), and
ADDITIVITY (grapheme clusters never span a word boundary, so the per-segment
grapheme counts sum to the whole-string count, the cross-function
invariant).

The API contract decisions pinned here (from the spec):
- OFFSETS, never string lists: ``word_bounds`` returns ``(start, end)`` pairs
  in PYTHON STR INDEX units (codepoints), so ``text[start:end]`` IS the
  segment (marshalling thousands of small PyStrings under the GIL would eat
  the win, and byte offsets would make Python slicing wrong on non-ASCII).
- ``grapheme_count`` counts EXTENDED grapheme clusters (``is_extended=true``
  in the crate, the only spelling anyone means by "grapheme" post-Unicode 11).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import grapheme_count, word_bounds, word_bounds_iter, word_count

# Built from codepoints (pure-ASCII source, per the suite's convention).
_CRLF = "\r\n"
_ZWJ = chr(0x200D)
_VS16 = chr(0xFE0F)
_COMBINING_ACUTE = chr(0x0301)
_KEYCAP = chr(0x20E3)
_WOMAN = chr(0x1F469)
_MICROSCOPE = chr(0x1F52C)
_MAN = chr(0x1F468)
_GIRL = chr(0x1F467)
_BOY = chr(0x1F466)
_THUMBS_UP = chr(0x1F44D)
_SKIN_TONE_3 = chr(0x1F3FD)
_RI_U = chr(0x1F1FA)
_RI_S = chr(0x1F1F8)
_RI_G = chr(0x1F1EC)
_RI_B = chr(0x1F1E7)
_L_JAMO = chr(0x1100)  # HANGUL CHOSEONG KIYEOK (L)
_V_JAMO = chr(0x1161)  # HANGUL JUNGSEONG A (V)
_T_JAMO = chr(0x11A8)  # HANGUL JONGSEONG KIYEOK (T)
_SYL_GA = chr(0xAC00)  # HANGUL SYLLABLE GA (LV)
_SYL_GAK = chr(0xAC01)  # HANGUL SYLLABLE GAK (LVT)
_SUN = chr(0x2600)

# (text, expected grapheme count, expected word_bounds): every row's values
# derived from the cited UAX #29 rules BEFORE running them against the crate.
_CASES: list[tuple[str, int, list[tuple[int, int]]]] = [
    # --- simple text (GB999 / WB5-WB999) ---
    ("", 0, []),
    ("a", 1, [(0, 1)]),
    ("plain text", 10, [(0, 5), (5, 6), (6, 10)]),
    # --- combining-mark chains: one cluster per base (GB9: x Extend) ---
    ("e" + _COMBINING_ACUTE, 1, [(0, 2)]),  # WB4: the Extend joins its base's word
    ("e" + _COMBINING_ACUTE * 3, 1, [(0, 4)]),
    # --- CRLF is ONE grapheme (GB4: CR x LF) and ONE word segment (WB3) ---
    (_CRLF, 1, [(0, 2)]),
    ("a" + _CRLF + "b", 3, [(0, 1), (1, 3), (3, 4)]),
    ("\r\r", 2, [(0, 1), (1, 2)]),  # GB4/GB5: CR breaks before and after CR
    # --- regional-indicator flags pair up (GB12/GB13; WB15/WB16, the RI pair
    # rules, numbered WB13c/WB13d before the renumbering) ---
    (_RI_U + _RI_S, 1, [(0, 2)]),  # the US flag is one cluster and one word
    (_RI_U + _RI_S + _RI_G + _RI_B, 2, [(0, 2), (2, 4)]),  # two flags
    (_RI_U + _RI_S + _RI_U, 2, [(0, 2), (2, 3)]),  # a pair, then an odd one out
    (_RI_U, 1, [(0, 1)]),  # a lone RI is its own cluster
    # --- ZWJ emoji sequences (GB11: ExtPict x ZWJ x ExtPict; WB3c) ---
    (_WOMAN + _ZWJ + _MICROSCOPE, 1, [(0, 3)]),  # woman-scientist: one cluster, one word
    (_MAN + _ZWJ + _WOMAN + _ZWJ + _GIRL + _ZWJ + _BOY, 1, [(0, 7)]),  # family
    # The grapheme/word ASYMMETRY on ZWJ: GB11 requires ExtPict on BOTH
    # sides, so 'a' ZWJ 'b' is TWO clusters (GB9 keeps the ZWJ with 'a';
    # GB999 breaks before 'b'), but WB4 collapses the ZWJ into its base
    # (X (Extend | Format | ZWJ)* -> X) and WB5 then joins the letters, so
    # it is ONE word segment.
    ("a" + _ZWJ + "b", 2, [(0, 3)]),
    # an emoji sequence followed by a flag is two clusters (GB999 between them)
    (_WOMAN + _ZWJ + _MICROSCOPE + _RI_U + _RI_S, 2, [(0, 3), (3, 5)]),
    # --- emoji modifiers and variation selectors are Extend (GB9) ---
    (_THUMBS_UP + _SKIN_TONE_3, 1, [(0, 2)]),  # GB9: emoji modifiers are Extend
    (_SUN + _VS16, 1, [(0, 2)]),  # VS16 is Extend
    ("1" + _VS16 + _KEYCAP, 1, [(0, 3)]),  # the keycap sequence: two Extends
    # --- Hangul jamo (GB6: L x V; GB7: (LV|V) x T; GB8: (LVT|T) x T: a JOIN,
    # the trailing T is part of the syllable term L*(V+|LV V*|LVT)T* in
    # UAX #29 Table 1c) ---
    (_SYL_GA, 1, [(0, 1)]),  # precomposed LV syllable
    (_SYL_GAK, 1, [(0, 1)]),  # precomposed LVT syllable
    (_L_JAMO + _V_JAMO, 1, [(0, 2)]),  # GB6: L x V
    (_L_JAMO + _V_JAMO + _T_JAMO, 1, [(0, 3)]),  # GB6 then GB7: (LV|V) x T
    (_SYL_GA + _T_JAMO, 1, [(0, 2)]),  # GB7: LV x T
    (_SYL_GAK + _T_JAMO, 1, [(0, 2)]),  # GB8: (LVT | T) x T; the T joins
    (_L_JAMO + _V_JAMO + _L_JAMO + _V_JAMO, 2, [(0, 4)]),  # graphemes: V x L
    # breaks (no GB rule joins them); words: every jamo and syllable is
    # ALetter (Table 3: Alphabetic minus Hiragana/Katakana scripts), so WB5
    # joins the whole run into ONE word.
    (_SYL_GA + _SYL_GAK, 2, [(0, 2)]),  # two clusters (LV x LVT breaks), one
    # word (ALetter x ALetter, WB5)
    # --- classic word-boundary shapes (WB5-WB999) ---
    ("Hello, world!", 13, [(0, 5), (5, 6), (6, 7), (7, 12), (12, 13)]),
    ("Hello,world", 11, [(0, 5), (5, 6), (6, 11)]),  # no space: comma still splits
    ("can't", 5, [(0, 5)]),  # WB6/WB7: AHLetter x MidNumLetQ x AHLetter
    ("Don't stop", 10, [(0, 5), (5, 6), (6, 10)]),
    ("3.14", 4, [(0, 4)]),  # WB11/WB12: Numeric x (MidNum|MidNumLetQ) x Numeric
    ("abc123", 6, [(0, 6)]),  # WB9/WB10: letters and digits join
    ("New York", 8, [(0, 3), (3, 4), (4, 8)]),
    # Hiragana has no joining rule (only Katakana does, WB13)...
    ("\u3053\u3093\u306b\u3061\u306f", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
    # ...so each kana is its own word segment, while Katakana joins:
    ("\u30c6\u30b9\u30c8", 3, [(0, 3)]),
    # --- the SARA AM edge: a grapheme cluster that SPANS a word boundary ---
    # U+0E33 is General_Category=Other_Letter but GraphemeBreakProperty lists
    # it as SpacingMark (GB9a: joins '0' into one cluster), while
    # WordBreakProperty gives it no entry (ALetter excludes Complex_Context
    # scripts), so WB999 splits it from '0'. One cluster, TWO word segments:
    # the one shape where the two segmenters disagree, found by the
    # structural property and confirmed against the normative data files.
    ("0" + chr(0x0E33), 1, [(0, 1), (1, 2)]),
]


class TestHandDerivedTable:
    @pytest.mark.parametrize(
        ("text", "graphemes", "bounds"),
        _CASES,
        ids=[
            f"{t[:12]!a}" if len(t) <= 12 else f"{t[:6]!a}...{len(t)}"
            for t, _, _ in _CASES
        ],
    )
    def test_matches_the_uax29_derived_table(
        self, text: str, graphemes: int, bounds: list[tuple[int, int]]
    ) -> None:
        """Every hand-derived row: the grapheme count and the full word-bounds
        list (Python str indices; ``text[a:b]`` is the segment) must equal
        what the UAX #29 rules in the module docstring produce for that input.
        Derived from the rules, not from the implementation: if the crate
        disagrees with a row, either the derivation or the crate is wrong, and
        this test is where that becomes visible instead of silent."""
        assert grapheme_count(text) == graphemes, f"graphemes for {text!a}"
        assert word_bounds(text) == bounds, f"word_bounds for {text!a}"


class TestStructuralProperties:
    # The COMPLETE measured set of characters where a grapheme cluster CAN
    # span a word boundary in UAX #29: three families, all for the same
    # structural reason (the two segmenters' property files disagree about
    # the character's class):
    #
    # - the GB9b PREPEND class (GraphemeBreakProperty.txt Prepend: the mark
    #   joins the FOLLOWING character into one cluster) whose WordBreak class
    #   is Format/Other and splits anyway: the Arabic prepended number/sign
    #   marks (U+0600..U+0605, U+06DD, U+0890/U+0891, U+08E2), Syriac
    #   abbreviation mark (U+070F), Kaithi number signs (U+110BD/U+110CD),
    #   and the Kawi sign repha (U+11F02);
    # - the Other_Letter spacing marks the two property files classify
    #   differently (SARA AM: U+0E33/U+0EB3 are SpacingMark to graphemes,
    #   GB9a joins them to the PRECEDING cluster, but no WordBreak entry,
    #   so WB999 splits them: '0' + SARA AM is ONE cluster and TWO word
    #   segments);
    # - the Indic signs that are Prepend to graphemes but letters to words
    #   (Sharada jihvamuliya/upadhmaniya U+111C2/U+111C3, Malayalam dot reph
    #   U+0D4E, Dives Akuru U+1193F/U+11941, Soyombo U+11A84..U+11A89,
    #   Masaram Gondi repha U+11D46, and U+113D1, assigned in the crate's
    #   Unicode 17 tables).
    #
    # DERIVED, not guessed: swept every non-surrogate codepoint with the two
    # probes '<cp>:' and 'a<cp>:b' against the shipped crate (29 violators,
    # the list below), including U+0D4E, which is not part of the SARA AM
    # pair. Every UAX #29-conforming segmenter has these edges; a crate
    # bump that changes the set must update this pin knowingly.
    _CLUSTER_SPANNING_WORDS = tuple(
        chr(cp)
        for cp in (
            0x0600, 0x0601, 0x0602, 0x0603, 0x0604, 0x0605, 0x06DD, 0x070F,
            0x0890, 0x0891, 0x08E2, 0x0D4E, 0x0E33, 0x0EB3, 0x110BD, 0x110CD,
            0x111C2, 0x111C3, 0x113D1, 0x1193F, 0x11941, 0x11A84, 0x11A85,
            0x11A86, 0x11A87, 0x11A88, 0x11A89, 0x11D46, 0x11F02,
        )
    )

    @given(st.text(alphabet=st.characters(exclude_categories=("Cn", "Cs")), max_size=48))
    @settings(max_examples=400)
    def test_word_bounds_are_monotonic_covering_and_round_trip(
        self, text: str
    ) -> None:
        """The structural promise any conforming segmenter makes: bounds are
        strictly increasing (each segment non-empty, none overlapping), they
        cover exactly [0, len(text)] (first starts at 0, last ends at the
        end, adjacent bounds touch), and joining the sliced segments
        reproduces the input byte-for-byte, the round-trip that makes the
        offsets API trustworthy (``text[a:b]`` really is the segment)."""
        bounds = word_bounds(text)
        assert all(a < b for a, b in bounds)
        assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))
        if text:
            assert bounds[0][0] == 0
            assert bounds[-1][1] == len(text)
        assert "".join(text[a:b] for a, b in bounds) == text

    @given(st.text(alphabet=st.characters(exclude_categories=("Cn", "Cs")), max_size=48))
    @settings(max_examples=400)
    def test_splitting_into_word_segments_never_loses_grapheme_clusters(
        self, text: str
    ) -> None:
        """Unconditional weak form: summing the grapheme counts of the word
        segments is always >= the whole string's count (a cluster that spans
        a segment boundary (a cluster-spanning character, see
        ``_CLUSTER_SPANNING_WORDS``) is counted in both
        segments, never in neither)."""
        assert sum(grapheme_count(text[a:b]) for a, b in word_bounds(text)) >= grapheme_count(text)
        assert grapheme_count(text) <= len(text)
        if text:
            assert grapheme_count(text) >= 1

    @given(
        st.text(
            alphabet=st.characters(
                exclude_categories=("Cn", "Cs"),
                exclude_characters=_CLUSTER_SPANNING_WORDS,
            ),
            max_size=48,
        )
    )
    @settings(max_examples=400)
    def test_grapheme_count_is_additive_over_word_segments(
        self, text: str
    ) -> None:
        """The strong cross-function invariant, scoped where it is true: for
        text free of the cluster-spanning set (see
        ``_CLUSTER_SPANNING_WORDS``, the complete measured set: the GB9b
        Prepend marks, the SARA AM spacing marks, and the Indic
        Prepend-to-grapheme/letter-to-words signs), a grapheme cluster never
        spans a word boundary, so the grapheme counts of the word segments
        sum EXACTLY to the whole string's count, and each word segment
        therefore contains at least one grapheme. The scoping is a measured
        UAX #29 property, not a dodge: the excluded set is exactly the
        codepoints where the normative property files make the two
        segmenters disagree, derived by sweeping every codepoint against the
        shipped crate, including U+0D4E, which is not part of the SARA AM
        pair."""
        assert grapheme_count(text) == sum(grapheme_count(text[a:b]) for a, b in word_bounds(text))
        assert grapheme_count(text) >= len(word_bounds(text))


class TestArgumentContract:
    @pytest.mark.parametrize("fn_name", ["grapheme_count", "word_bounds", "word_count"])
    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), memoryview(b"abc"), 123, None],
        ids=["bytes", "bytearray", "memoryview", "int", "none"],
    )
    def test_non_str_arguments_raise_type_error(
        self, fn_name: str, not_str: object
    ) -> None:
        fn = {
            "grapheme_count": grapheme_count,
            "word_bounds": word_bounds,
            "word_count": word_count,
        }[fn_name]
        with pytest.raises(TypeError):
            fn(not_str)  # type: ignore[arg-type]

    @pytest.mark.parametrize("fn_name", ["grapheme_count", "word_bounds", "word_count"])
    def test_lone_surrogates_are_refused_at_the_argument_boundary(
        self, fn_name: str
    ) -> None:
        """Same pyo3 ``&str`` boundary as every str-in function (pinned for
        ``finalize`` in tests/test_finalize.py)."""
        fn = {
            "grapheme_count": grapheme_count,
            "word_bounds": word_bounds,
            "word_count": word_count,
        }[fn_name]
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            fn("ok\ud800ok")  # type: ignore[arg-type]


class TestWordBoundsIter:
    """The streaming spelling of ``word_bounds``: a lazy iterator yielding
    the SAME ``(start, end)`` sequence, so whole-file segmentation stops paying
    the list API's O(segments) GIL-held marshalling (measured at 428-497ms at
    12 MiB). The segmentation runs under one detached pass when the
    iterator is constructed, and each ``__next__`` holds the GIL only for one
    tuple (µs-scale; the band is pinned in tests/test_gil_release.py).

    Correctness is pinned against the list API itself: same bounds sequence
    over every UAX #29 tricky row, arbitrary hypothesis text, and the corpus
    shapes, which is the strongest available oracle given no stdlib
    segmenter exists."""

    @pytest.mark.parametrize(
        ("text", "graphemes", "bounds"),
        _CASES,
        ids=[f"case{i}" for i in range(len(_CASES))],
    )
    def test_yields_the_list_apis_exact_sequence_on_every_tricky_row(
        self, text: str, graphemes: int, bounds: list[tuple[int, int]]
    ) -> None:
        assert list(word_bounds_iter(text)) == word_bounds(text) == bounds

    def test_the_iterator_protocol_shapes(self) -> None:
        it = word_bounds_iter("Hello, world!")
        assert iter(it) is it  # an iterator: iter() hands back the same object
        assert next(it) == (0, 5)
        # Mid-iteration resumption: the remainder is exactly the list API's
        # remainder.
        assert list(it) == word_bounds("Hello, world!")[1:]
        with pytest.raises(StopIteration):
            next(it)  # exhaustion is stable, not one-shot

    def test_length_hint_tracks_partial_consumption(self) -> None:
        """``__length_hint__`` (what ``list()``/``tuple()`` preallocation and
        the interpreter's own optimizations consult) is the REMAINING count:
        the full count at construction, decremented by each ``next()``, zero
        at exhaustion, never the original length after consumption."""
        expected = word_bounds("Hello, world!")
        it = word_bounds_iter("Hello, world!")
        assert it.__length_hint__() == len(expected)
        for consumed, _ in enumerate(expected, start=1):
            next(it)
            assert it.__length_hint__() == len(expected) - consumed
        assert it.__length_hint__() == 0
        with pytest.raises(StopIteration):
            next(it)
        assert it.__length_hint__() == 0  # stable at exhaustion

    def test_empty_text_yields_nothing(self) -> None:
        assert list(word_bounds_iter("")) == []

    def test_corpus_shaped_text_parities_with_the_list_api(self) -> None:
        # A prose-shaped slab (the corpus unit the GIL/wall cells measure),
        # non-ASCII included: the streaming and list spellings must agree on
        # realistic sizes, not just table rows.
        slab = ("caf\u00e9 na\u00efve quarter\u0301 sample. " * 400) + "\n\n"
        assert list(word_bounds_iter(slab)) == word_bounds(slab)

    @given(st.text(alphabet=st.characters(exclude_categories=("Cn", "Cs")), max_size=48))
    @settings(max_examples=400)
    def test_sequence_parity_over_arbitrary_text(self, text: str) -> None:
        assert list(word_bounds_iter(text)) == word_bounds(text)

    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), memoryview(b"abc"), 123, None],
        ids=["bytes", "bytearray", "memoryview", "int", "none"],
    )
    def test_non_str_arguments_raise_type_error(self, not_str: object) -> None:
        with pytest.raises(TypeError):
            word_bounds_iter(not_str)  # type: ignore[arg-type]

    def test_lone_surrogates_are_refused_at_the_argument_boundary(self) -> None:
        """Same pyo3 ``&str`` boundary as every str-in function (pinned for
        ``finalize`` in tests/test_finalize.py)."""
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            word_bounds_iter("ok\ud800ok")  # type: ignore[arg-type]


class TestWordCount:
    """The count spelling of the word contract: ``word_count(t) ==
    len(word_bounds(t))``, the SAME UAX #29 word segmentation under the SAME
    argument boundary, with none of the list API's O(segments) marshalling:
    the whole cluster walk runs detached and the return is one int (the
    ``grapheme_count`` no-marshalling shape; its GIL band is the ping floor,
    the grapheme_count cells' band, measured in tests/test_gil_release.py).

    The oracle is ``word_bounds`` itself, the strongest available
    differential given no stdlib segmenter exists (the list API's own
    correctness is proven against the hand-derived UAX #29 table above):
    count parity over every tricky row, arbitrary hypothesis text, and the
    corpus shapes, plus the degenerates and the argument contract (mirrored
    through ``TestArgumentContract``'s shared parametrizations, which
    ``word_count`` joins)."""

    @pytest.mark.parametrize(
        ("text", "graphemes", "bounds"),
        _CASES,
        ids=[
            f"{t[:12]!a}" if len(t) <= 12 else f"{t[:6]!a}...{len(t)}"
            for t, _, _ in _CASES
        ],
    )
    def test_count_equals_the_list_length_on_every_tricky_row(
        self, text: str, graphemes: int, bounds: list[tuple[int, int]]
    ) -> None:
        """Every hand-derived UAX #29 row: the count spelling must agree with
        the table's pinned bounds length; the tricky shapes (CRLF, RI pairs,
        ZWJ sequences, the SARA AM cluster-spanning split, the trailing
        punctuation rows) are exactly where a count fast path could diverge
        from the segmentation the list API pins."""
        assert word_count(text) == len(bounds), f"word_count for {text!a}"
        assert word_count(text) == len(word_bounds(text)), f"word_count for {text!a}"

    @given(st.text(alphabet=st.characters(exclude_categories=("Cn", "Cs")), max_size=48))
    @settings(max_examples=400)
    def test_count_equals_the_list_length_over_arbitrary_text(self, text: str) -> None:
        """The arbitrary-Unicode differential over the module's existing text
        strategy (any codepoint class, mixed widths, combining marks
        anywhere): the count equals the list API's length for every generated
        text."""
        assert word_count(text) == len(word_bounds(text))

    def test_the_degenerates(self) -> None:
        """The fixed degenerate anchors: empty is 0; ``"Hello, world!"`` is
        FIVE segments (the two words plus the comma, space, and bang,
        ``word_bounds``'s own pinned row); whitespace-only is ONE segment
        (WB999 joins Any × Any, so only WB1/WB2 bound the text; the count
        spelling must not silently re-scope "word" to "non-whitespace
        run")."""
        assert word_count("") == 0
        assert word_count("Hello, world!") == 5
        assert word_count("   ") == 1

    def test_corpus_shaped_text_parities_with_the_list_api(self) -> None:
        # A prose-shaped slab (the corpus unit the GIL/wall cells measure),
        # non-ASCII included: the count and list spellings must agree on
        # realistic sizes, not just table rows.
        slab = ("caf\u00e9 na\u00efve quarter\u0301 sample. " * 400) + "\n\n"
        assert word_count(slab) == len(word_bounds(slab))
