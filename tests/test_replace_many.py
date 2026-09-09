"""Contract gate for ``tors.replace_many``: simultaneous multi-pattern
replace, leftmost-longest, GIL-released.

``tors.replace_many(text, replacements)`` applies a ``dict[str, str]`` of
substring replacements in ONE native pass, with ``find_patterns``'s exact
search semantics (the same aho-corasick ``LeftmostLongest`` engine, one
automaton over the keys):

1. **Leftmost-longest**: the scan proceeds left to right; at each position
   the LONGEST matching key wins, regardless of its position in the dict
   (NOT regex alternation's leftmost-FIRST priority; a shorter key never
   beats a longer one, and dict ORDER cannot matter: two distinct keys of
   the same length can never both match at one position, so "longest" is
   always a strict, order-free winner).
2. **Non-overlapping**: the scan resumes at the END of each consumed span
   (resume-at-match-end), so consumed spans never overlap and unmatched
   characters between adjacent matches pass through untouched.
3. **No cascade**: replacement OUTPUT is never re-scanned: a value that
   itself contains a key is emitted verbatim and the scan moves on (the
   ``&#38;amp;`` double-replace discipline; ``re.sub`` rescans its own
   output and does not promise this).

This is the replace primitive CPython does not have: chained
``str.replace`` calls are N whole-text GIL-held passes with
sequential-dependence semantics (a later pass sees earlier outputs), and
``re.sub`` with an alternation is leftmost-FIRST and output-rescanning,
neither has these semantics, so (as with ``find_patterns``) no stdlib
oracle exists and the contract is proven three ways: (a) golden battery
rows with exact expected strings, every row cross-checked against the
brute-force pure-Python oracle (``reference_replace_many``, in
tests/reference.py, character-space, dict-order-agnostic by
construction), (b) hypothesis differentials over multi-byte alphabets and
arbitrary Unicode with substring-biased keys, plus the dict-ORDER
irrelevance property, and (c) the identity-return contract pinned as
object identity.

The identity contract (the idiom, complete form):
``replace_many(s, m) is s`` EXACTLY when ``replace_many(s, m) == s``: a
map whose keys never match returns the ORIGINAL input object, and so does
a map whose net effect is the identity (every match replaced by output
that reconstructs the input, e.g. ``{"aa": "aa"}`` over ``"aaa"``); a map
that changes anything returns a fresh string.

Contract decisions at the argument boundary (each pinned below):

- an empty key raises ``ValueError("empty pattern")``: it would match at
  every position (the ``find_patterns`` contract), wherever it sits in the
  dict;
- an empty dict returns the input OBJECT immediately (no automaton build);
- ``text`` must be exactly ``str`` and ``replacements`` exactly ``dict``:
  a non-``str`` key or value raises ``TypeError``, and a tuple or list of
  pairs raises ``TypeError`` too (the dict is the ergonomic shape and its
  unique keys are what make the semantics order-free);
- lone surrogates (text, key, or value) are refused with
  ``UnicodeEncodeError`` before any Rust code runs, the standard str-in
  boundary every tors function pays.

The GIL-release claim (automaton build + scan + splice under one
``py.detach``; the GIL-held residue is the O(entries) argument walk plus
the O(output) marshalling of ONE string, so no list-shape class exists) is
pinned in tests/test_gil_release.py.
"""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reference import reference_replace_many
from tors import find_patterns, replace_many, replace_many_masked

# Built from codepoints (pure-ASCII source, per the suite's convention).
_E_ACUTE = chr(0xE9)  # precomposed é: 2 UTF-8 bytes
_COMBINING_ACUTE = chr(0x301)  # decomposed accent: 2 UTF-8 bytes, 1 char
_ZWJ = chr(0x200D)
_TOKYO = chr(0x6771) + chr(0x4EAC)  # 東京: two 3-byte chars
_KYOTO = chr(0x4EAC) + chr(0x90FD)  # 京都
_OSAKA = chr(0x5927) + chr(0x962A)  # 大阪
_FAMILY = (
    "\U0001f468" + _ZWJ + "\U0001f469" + _ZWJ + "\U0001f467"
)  # man-ZWJ-woman-ZWJ-girl: 5 chars, 18 bytes


# --- The golden battery ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "replacements", "expected"),
    [
        # Longest wins at the same start regardless of dict order, the
        # row the argument order is rotated on below.
        ("the catalogue", {"cat": "X", "catalogue": "Y"}, "the Y"),
        ("the catalogue", {"catalogue": "Y", "cat": "X"}, "the Y"),
        # Resume-at-match-end: "aa" consumed at 0-2 and 2-4, never
        # overlapping into a leftover "a".
        ("aaaa", {"aa": "b"}, "bb"),
        # The chain that looks wrong but is right: "ab" matches at 0-2,
        # the scan resumes at 2; the middle "c" is a GAP (neither "ab"
        # nor "bc" matches there), passes through, and "bc" matches at
        # 3-5. NOT "XY": leftmost-longest never backs up to re-match the
        # "bc" a "b"-ending earlier match would have left behind.
        ("abcbc", {"ab": "X", "bc": "Y"}, "XcY"),
        # Deletion via empty value.
        ("banana", {"an": ""}, "ba"),
        # No-rescan: the value's own "a" is emitted verbatim, never
        # re-replaced ("baba" would be the cascade tors does not do).
        ("a", {"a": "ba"}, "ba"),
        # A value CONTAINING a key does not cascade either.
        ("cat", {"cat": "catalogue"}, "catalogue"),
        ("the cat sat", {"cat": "catalogue"}, "the catalogue sat"),
        # Adjacent matches, mixed lengths.
        ("abcdabcab", {"ab": "[ab]", "abc": "[abc]", "abcd": "[abcd]"}, "[abcd][abc][ab]"),
        ("one two one", {"one": "1"}, "1 two 1"),
        ("nothing here", {"xyz": "Q"}, "nothing here"),
    ],
    ids=[
        "longest-at-same-start",
        "longest-is-order-independent",
        "resume-at-match-end",
        "chain-with-gap-passes-through",
        "deletion-via-empty-value",
        "no-rescan-of-output",
        "value-containing-a-key-does-not-cascade",
        "value-containing-a-key-over-longer-text",
        "prefix-chain-rotating-lengths",
        "repeated-key-occurrences",
        "no-matches",
    ],
)
def test_golden_battery(text: str, replacements: dict[str, str], expected: str) -> None:
    """The fixed anchor of the contract: every golden case asserts the EXACT
    expected string and the oracle's agreement, so a hand-computed
    expectation that disagreed with the brute-force reference would fail
    loudly here rather than silently laundering a wrong pin into the suite
    (the discipline that caught the crate-side battery's mistyped é-b-é
    row, per tests/test_find_patterns.py)."""
    result = replace_many(text, replacements)
    assert result == expected
    assert result == reference_replace_many(text, replacements)


# --- The multi-byte battery ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "replacements", "expected"),
    [
        ("café café", {"café": "CAFE"}, "CAFE CAFE"),
        ("cafe" + _COMBINING_ACUTE + " ok", {"cafe" + _COMBINING_ACUTE: "café"}, "café ok"),
        ("ééé", {"é": "e"}, "eee"),
        ("éabé", {"é": ""}, "ab"),
        ("ab", {"b": "é"}, "aé"),
        ("京都東京大阪", {"東京": "京都"}, "京都京都大阪"),
        ("京都東京京都", {"東京": "T", "京都": "K"}, "KTK"),
        ("hi" + _FAMILY + "!", {_FAMILY: "FAMILY"}, "hiFAMILY!"),
        ("éabéab", {"ab": "XY"}, "éXYéXY"),
        ("東京éab", {"éab": "OK"}, "東京OK"),
    ],
    ids=[
        "precomposed-accents",
        "decomposed-accents",
        "multibyte-key-to-ascii",
        "multibyte-deletion",
        "ascii-key-to-multibyte-value",
        "cjk",
        "cjk-two-keys",
        "emoji-zwj-family-key",
        "ascii-key-over-non-ascii-text",
        "multibyte-prefix-before-match",
    ],
)
def test_multibyte_keys_values_and_text(
    text: str, replacements: dict[str, str], expected: str
) -> None:
    """The multi-byte battery: keys, values, and texts spanning every UTF-8
    width (1-byte ASCII, 2-byte é and the combining accent, 3-byte CJK,
    4-byte emoji plus the 3-byte ZWJ inside the family key). The engine
    runs on UTF-8 bytes; a key or span miscount of any kind breaks one of
    these rows, and every row is cross-checked against the character-space
    oracle."""
    assert not (
        text.isascii() and all(k.isascii() and v.isascii() for k, v in replacements.items())
    ), "every row of the multi-byte battery must carry non-ASCII somewhere"
    result = replace_many(text, replacements)
    assert result == expected
    assert result == reference_replace_many(text, replacements)


# --- The identity-return contract ----------------------------------------------------


class TestIdentityContract:
    def test_no_match_map_returns_the_same_object(self) -> None:
        """The zero-cost lane: a map whose keys never match returns the
        ORIGINAL input object (no allocation, no copy, no marshalling
        (``Cow::Borrowed`` on the crate side)."""
        s = "nothing to see here"
        assert replace_many(s, {"xyz": "Q"}) is s

    def test_net_identity_map_returns_the_same_object(self) -> None:
        """The complete form of the contract: a map whose keys DO match but
        whose net effect is the identity still returns the ORIGINAL input
        object: the output is reconstructed equal and the implementation
        hands back the borrowed input rather than a fresh equal string.
        ``{"aa": "aa"}`` over ``"aaa"`` matches at 0-2, re-emits "aa",
        then passes the final "a" through: output == input, so ``is``."""
        s = "aaa"
        assert replace_many(s, {"aa": "aa"}) is s
        assert replace_many(s, {"a": "a"}) is s

    def test_net_identity_holds_over_multibyte_and_multi_key(self) -> None:
        """The same net-identity lane over non-ASCII text and a multi-key
        map (some keys matching, some not)."""
        s = "café 東京"
        assert replace_many(s, {"é": "é", "東京": "東京", "zzz": "QQQ"}) is s

    def test_empty_dict_returns_the_same_object_immediately(self) -> None:
        """The early exit: no automaton is built, nothing is scanned, the
        input object itself comes back for any text."""
        s = "any text at all"
        assert replace_many(s, {}) is s

    def test_a_map_that_changes_anything_returns_a_new_object(self) -> None:
        """The contrapositive, pinned as an object-level fact: when the
        output differs from the input, the result is a FRESH string (the
        input object is never mutated or handed back unequal)."""
        result = replace_many("a", {"a": "b"})
        assert result == "b"
        assert result != "a"
        deleted = replace_many("banana", {"an": ""})
        assert deleted == "ba"


# --- The argument-boundary contract --------------------------------------------------


class TestArgumentContract:
    def test_empty_key_raises_value_error(self) -> None:
        """An empty key would match at every position, refused up front
        with exactly ``ValueError("empty pattern")`` (the ``find_patterns``
        contract), wherever it sits in the dict."""
        with pytest.raises(ValueError, match="^empty pattern$"):
            replace_many("some text", {"ok": "x", "": "y"})
        with pytest.raises(ValueError, match="^empty pattern$"):
            replace_many("some text", {"": "y"})

    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), memoryview(b"abc"), 123, None],
        ids=["bytes", "bytearray", "memoryview", "int", "none"],
    )
    def test_non_str_text_raises_type_error(self, not_str: object) -> None:
        with pytest.raises(TypeError):
            replace_many(not_str, {"a": "b"})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), memoryview(b"abc"), 123, None],
        ids=["bytes", "bytearray", "memoryview", "int", "none"],
    )
    def test_non_str_keys_raise_type_error(self, not_str: object) -> None:
        """Exactly ``str`` keys (the str-exactly rule every tors str argument
        follows): pyo3's extraction
        rejects everything else with ``TypeError``. The unhashable shapes
        (bytearray, memoryview) are refused by the dict itself before the
        call, so either way the boundary never sees a non-``str`` key."""
        with pytest.raises(TypeError):
            replace_many("text", {not_str: "b"})  # type: ignore[dict-item]

    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), memoryview(b"abc"), 123, None],
        ids=["bytes", "bytearray", "memoryview", "int", "none"],
    )
    def test_non_str_values_raise_type_error(self, not_str: object) -> None:
        with pytest.raises(TypeError):
            replace_many("text", {"a": not_str})  # type: ignore[dict-item]

    @pytest.mark.parametrize(
        "not_dict",
        [(("a", "b"),), [("a", "b")], "abc", 123, None],
        ids=["tuple-of-pairs", "list-of-pairs", "str", "int", "none"],
    )
    def test_non_dict_replacements_raise_type_error(self, not_dict: object) -> None:
        """``replacements`` is exactly ``dict[str, str]``, the annotation's
        type, and the shape whose unique keys make the semantics order-free.
        A tuple or list of the right pairs still raises ``TypeError`` (the
        non-goal, like ``find_patterns``' list-only
        ``patterns``)."""
        with pytest.raises(TypeError):
            replace_many("text", not_dict)  # type: ignore[arg-type]

    def test_lone_surrogates_are_refused_at_the_argument_boundary(self) -> None:
        """Lone surrogates (a ``str`` CPython can hold but UTF-8 cannot
        encode) are refused with ``UnicodeEncodeError`` before any Rust code
        runs, the standard str-in boundary, paid by the text and every key
        and value alike (the text's borrow fires first, before the dict
        walk)."""
        with pytest.raises(UnicodeEncodeError):
            replace_many("abc\ud800", {"a": "b"})
        with pytest.raises(UnicodeEncodeError):
            replace_many("abc", {"a\ud800": "b"})
        with pytest.raises(UnicodeEncodeError):
            replace_many("abc", {"a": "b\ud800"})


# --- The hypothesis differentials ----------------------------------------------------

_SMALL_ALPHABET = "ab" + _E_ACUTE + _COMBINING_ACUTE + "東京🦀"


@st.composite
def _replacements_and_text(
    draw: st.DrawFn,
) -> tuple[str, dict[str, str]]:
    """Replacement dicts (1-5 entries, values possibly empty (the deletion
    lane) or key-containing (the cascade lane)) and texts over an alphabet
    spanning every UTF-8 width: 1-byte ``ab``, 2-byte ``é`` and the
    combining accent, 3-byte CJK, 4-byte emoji. Keys and values drawn from
    the SAME alphabet as the text, so matches, deletions, and
    value-contains-key shapes all occur at every width combination."""
    replacements = draw(
        st.dictionaries(
            keys=st.text(alphabet=_SMALL_ALPHABET, min_size=1, max_size=4),
            values=st.text(alphabet=_SMALL_ALPHABET, max_size=4),
            min_size=1,
            max_size=5,
        )
    )
    text = draw(st.text(alphabet=_SMALL_ALPHABET, max_size=40))
    return text, replacements


@given(_replacements_and_text())
@settings(max_examples=500)
def test_matches_the_reference_over_multibyte_alphabets(
    text_replacements: tuple[str, dict[str, str]],
) -> None:
    """The differential proof over the multi-byte alphabet: tors's answer
    must equal the brute-force character-space oracle EXACTLY (string
    equality) for every generated dict and text. A span miscount, a
    wrongly-ordered key preference, a cascade slip, or a resume-position
    bug of any kind breaks this property; the golden battery pins the
    individual shapes it finds."""
    text, replacements = text_replacements
    assert replace_many(text, replacements) == reference_replace_many(text, replacements)


@st.composite
def _substring_key_replacements_and_text(
    draw: st.DrawFn,
) -> tuple[str, dict[str, str]]:
    """Arbitrary-Unicode texts (hypothesis's full ``st.text`` alphabet: any
    script, any marks, no alphabet bias) with keys biased toward SUBSTRINGS
    of the text, so matches actually occur over text no small alphabet can
    generate; values are arbitrary short strings (empty allowed)."""
    text = draw(st.text(max_size=50))
    replacements: dict[str, str] = {}
    for _ in range(draw(st.integers(1, 4))):
        if text and draw(st.booleans()):
            i = draw(st.integers(0, len(text) - 1))
            j = draw(st.integers(i + 1, len(text)))
            replacements[text[i:j]] = draw(st.text(max_size=4))
        else:
            replacements[draw(st.text(min_size=1, max_size=4))] = draw(st.text(max_size=4))
    return text, replacements


@given(_substring_key_replacements_and_text())
@settings(max_examples=300)
def test_matches_the_reference_over_arbitrary_unicode_with_substring_keys(
    text_replacements: tuple[str, dict[str, str]],
) -> None:
    """The arbitrary-Unicode differential: substring-biased keys over
    hypothesis's full text strategy (any codepoint class, mixed widths,
    combining marks anywhere), the coverage the fixed alphabets cannot
    reach, still under exact string equality with the oracle."""
    text, replacements = text_replacements
    assert replace_many(text, replacements) == reference_replace_many(text, replacements)


@st.composite
def _key_value_pairs(
    draw: st.DrawFn,
) -> list[tuple[str, str]]:
    """2-6 distinct (key, value) pairs, the raw material for the
    order-irrelevance property, held as a LIST so the same mapping can be
    rebuilt in different insertion orders."""
    pairs = draw(
        st.dictionaries(
            keys=st.text(alphabet=_SMALL_ALPHABET, min_size=1, max_size=3),
            values=st.text(alphabet=_SMALL_ALPHABET, max_size=3),
            min_size=2,
            max_size=6,
        )
    )
    return list(pairs.items())


@given(_key_value_pairs())
@settings(max_examples=200)
def test_dict_insertion_order_cannot_matter(pairs: list[tuple[str, str]]) -> None:
    """The order-freedom property, structural on the semantics: the same
    mapping built in two different insertion orders (forward and reversed, with
    >= 2 entries these differ) produces the IDENTICAL result string,
    and both agree with the oracle. Leftmost-longest over unique keys makes
    this true by construction; a leftmost-FIRST implementation (regex
    alternation's priority, ``re.sub``'s semantics) breaks it on the first
    prefix-pair it generates."""
    assert len(pairs) >= 2
    text = "".join(k for k, _ in pairs) + "ab東京é"
    forward = dict(pairs)
    backward = dict(reversed(pairs))
    assert forward == backward  # the same mapping, different insertion order
    result_forward = replace_many(text, forward)
    result_backward = replace_many(text, backward)
    assert result_forward == result_backward
    assert result_forward == reference_replace_many(text, forward)


def test_every_replacement_map_over_a_tiny_alphabet_matches_the_reference() -> None:
    """The deterministic sweep (the suite's exhaustive-small-alphabet idiom):
    every replacement dict over the keys ``{"a", "ab", "b"}`` (the prefix /
    overlap interactions) with values from ``{"X", "", "ba"}`` (a plain
    value, the deletion lane, and a value CONTAINING a key (the cascade
    lane)): 64 dicts including the empty one, crossed with every text over
    ``{"a", "b"}`` up to length 5 (63 texts), the complete small space of
    prefix/overlap/duplicate-shaped interactions, no sampling at all."""
    keys = ["a", "ab", "b"]
    values = ["X", "", "ba"]
    dicts: list[dict[str, str]] = [{}]
    for size in 1, 2, 3:
        for key_combo in itertools.combinations(keys, size):
            for value_combo in itertools.product(values, repeat=size):
                dicts.append(dict(zip(key_combo, value_combo, strict=True)))
    texts: list[str] = []
    for length in range(6):
        texts.extend("".join(combo) for combo in itertools.product("ab", repeat=length))
    for replacements in dicts:
        for text in texts:
            assert replace_many(text, replacements) == reference_replace_many(text, replacements)


# --- replace_many_masked: the length-preserving redaction spelling --------------------
#
# ``tors.replace_many_masked(text, replacements, mask="*")`` is
# ``replace_many``'s exact scan (the same leftmost-longest,
# resume-at-match-end, no-cascade automaton: one engine, one pass)
# with the LENGTH GUARANTEE the redaction shape needs: every matched
# span of L characters is replaced by its value truncated to L (longer
# values) or right-padded with the mask (shorter), so the output's
# CHARACTER count equals the input's, and every NON-matching offset
# addresses the same character it addressed before, so pre-computed
# offsets (``find_patterns`` spans, tokenized positions) stay valid
# after redaction. Chained ``str.replace`` cannot promise this (each
# pass shifts every offset after its edit), and neither can
# ``replace_many`` itself (values pass through at their own lengths).
class TestReplaceManyMasked:
    @pytest.mark.parametrize(
        ("text", "replacements", "mask", "expected"),
        [
            # The task-shaped rows: truncation and padding to the span.
            ("the cat sat", {"cat": "[REDACTED]"}, "*", "the [RE sat"),
            ("the cat sat", {"cat": "X"}, "*", "the X** sat"),
            ("cat", {"cat": "XYZ"}, "*", "XYZ"),
            # The deletion shape: an empty value pads to the full mask.
            ("the cat sat", {"cat": ""}, "*", "the *** sat"),
            ("banana", {"an": ""}, "*", "b****a"),
            # Multi-match, each span sized independently.
            ("one two one", {"one": "1"}, "*", "1** two 1**"),
            ("one two one", {"one": "ONE!"}, "*", "ONE two ONE"),
            ("aaaa", {"aa": "b"}, "*", "b*b*"),
            # Leftmost-longest carries over: the 9-char span pads a 1-char
            # value to nine.
            ("the catalogue", {"cat": "X", "catalogue": "Y"}, "*", "the Y********"),
            # Resume-at-match-end with mixed truncation and padding.
            ("abcbc", {"ab": "X", "bc": "YYYY"}, "*", "X*cYY"),
            # Mid-string truncation: the non-matching neighbors keep
            # their offsets ("a" and "c" unchanged around the 1-char span).
            ("abc", {"b": "XY"}, "*", "aXc"),
            # Multi-byte text, keys, and values: the arithmetic is
            # CHARACTER-level on every UTF-8 width.
            ("café café", {"café": "C"}, "*", "C*** C***"),
            ("京都東京", {"東京": "TOKYO"}, "*", "京都TO"),
            (
                "cafe" + _COMBINING_ACUTE + " ok",
                {"cafe" + _COMBINING_ACUTE: "café"},
                "*",
                "café* ok",
            ),
            # A multi-byte MASK pads with its own single character.
            ("éaé cat", {"cat": "X"}, _E_ACUTE, "éaé X" + _E_ACUTE + _E_ACUTE),
            ("cat", {"cat": "XY"}, "東", "XY東"),
            # The no-match and empty lanes.
            ("nothing here", {"xyz": "Q"}, "*", "nothing here"),
            ("", {"cat": "X"}, "*", ""),
        ],
        ids=[
            "value-truncated-to-span",
            "value-padded-to-span",
            "exact-length-value",
            "deletion-shape-empty-value",
            "deletion-shape-embedded",
            "multi-match-padding",
            "multi-match-truncation",
            "adjacent-spans-sized-independently",
            "longest-wins-then-pads",
            "resume-at-match-end-mixed",
            "mid-string-truncation-preserves-neighbors",
            "multibyte-key-padding",
            "cjk-truncation",
            "decomposed-accent-key",
            "multibyte-mask",
            "multibyte-mask-padding",
            "no-matches",
            "empty-text",
        ],
    )
    def test_golden_battery(
        self, text: str, replacements: dict[str, str], mask: str, expected: str
    ) -> None:
        """The fixed anchor of the length guarantee: every row asserts the
        EXACT expected string, the output's character length equals the
        input's, and the scan agrees with the brute-force oracle's spans
        (``reference_replace_many`` with each value pre-truncated or
        pre-padded to its key's length reproduces the output; the scan
        is ``replace_many``'s, only the values are reshaped)."""
        result = replace_many_masked(text, replacements, mask)
        assert result == expected
        assert len(result) == len(text)
        reshaped = {
            key: value[: len(key)]
            if len(value) >= len(key)
            else value + mask * (len(key) - len(value))
            for key, value in replacements.items()
        }
        assert result == reference_replace_many(text, reshaped)

    @given(_replacements_and_text())
    @settings(max_examples=300)
    def test_offsets_and_spans_are_preserved_over_multibyte_alphabets(
        self, text_replacements: tuple[str, dict[str, str]]
    ) -> None:
        """The property the function exists for, over the module's own
        multi-byte strategy: every ``find_patterns`` span (the same
        leftmost-longest scan, CHAR offsets) addresses in the output the
        truncated-or-padded value of its key, and every character NOT
        inside a matched span is identical at its own offset, so
        pre-computed offsets stay valid after redaction, by construction
        (the length arithmetic), not by coincidence. The mask is the
        2-byte é, so the pad lane runs at a non-ASCII width too."""
        text, replacements = text_replacements
        out = replace_many_masked(text, replacements, _E_ACUTE)
        assert len(out) == len(text)
        keys = list(replacements)
        matched: set[int] = set()
        for start, end, pattern_index in find_patterns(keys, text):
            key = keys[pattern_index]
            value = replacements[key]
            span_length = end - start
            expected_span = (
                value[:span_length]
                if len(value) >= span_length
                else value + _E_ACUTE * (span_length - len(value))
            )
            assert out[start:end] == expected_span
            matched.update(range(start, end))
        for offset in range(len(text)):
            if offset not in matched:
                assert out[offset] == text[offset]

    @given(_replacements_and_text())
    @settings(max_examples=300)
    def test_the_length_invariant_over_multibyte_alphabets(
        self, text_replacements: tuple[str, dict[str, str]]
    ) -> None:
        """The length invariant over the module's multi-byte hypothesis
        strategy (every UTF-8 width, deletions, cascades): the output's
        CHARACTER count equals the input's, always, the property a
        chained-``str.replace`` redaction cannot keep."""
        text, replacements = text_replacements
        assert len(replace_many_masked(text, replacements)) == len(text)

    @given(_substring_key_replacements_and_text())
    @settings(max_examples=200)
    def test_the_length_invariant_over_arbitrary_unicode(
        self, text_replacements: tuple[str, dict[str, str]]
    ) -> None:
        """The same invariant over arbitrary Unicode with substring-biased
        keys (matches actually occur over text no fixed alphabet can
        generate): char in, char out, whatever the widths."""
        text, replacements = text_replacements
        assert len(replace_many_masked(text, replacements)) == len(text)

    def test_mask_must_be_exactly_one_character(self) -> None:
        """The length arithmetic requires a one-character mask (the pad
        is ``mask * (L - len(value))``); anything else (empty, two, a
        whole word, two multi-byte chars) is refused up front with the
        exact message."""
        for bad in ("", "**", "abc", "éé", "東京"):
            with pytest.raises(ValueError, match="^mask must be exactly one character"):
                replace_many_masked("cat", {"cat": "X"}, bad)

    def test_identity_contract_is_the_net_identity_idiom(self) -> None:
        """``replace_many_masked(s, m, mask) is s`` EXACTLY when the
        output equals the input: the no-match map (nothing consumed),
        the empty dict (no automaton at all), and (the masked family's
        own interesting lane) a map whose values reconstruct the input
        through the length arithmetic (a value equal to its key pads
        and truncates to the key itself), all return the ORIGINAL
        object; a map that changes anything returns a fresh string."""
        s = "the cat sat"
        assert replace_many_masked(s, {"zzz": "QQQ"}) is s
        assert replace_many_masked(s, {}) is s
        s2 = "the cat sat"
        assert replace_many_masked(s2, {"cat": "cat"}) is s2
        s3 = "abab"
        assert replace_many_masked(s3, {"ab": "ab"}) is s3
        s4 = "café 東京"
        assert replace_many_masked(s4, {"é": "é", "東京": "東京"}) is s4
        s5 = "the cat sat"
        changed = replace_many_masked(s5, {"cat": "X"})
        assert changed == "the X** sat"
        assert changed is not s5

    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), memoryview(b"abc"), 123, None],
        ids=["bytes", "bytearray", "memoryview", "int", "none"],
    )
    def test_non_str_text_raises_type_error(self, not_str: object) -> None:
        with pytest.raises(TypeError):
            replace_many_masked(not_str, {"a": "b"})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), memoryview(b"abc"), 123, None],
        ids=["bytes", "bytearray", "memoryview", "int", "none"],
    )
    def test_non_str_keys_and_values_raise_type_error(self, not_str: object) -> None:
        """The dict contract is ``replace_many``'s verbatim: exactly
        ``str`` keys and values (the unhashable shapes are refused by
        the dict itself before the call, so either way the boundary never
        sees them)."""
        with pytest.raises(TypeError):
            replace_many_masked("text", {not_str: "b"})  # type: ignore[dict-item]
        with pytest.raises(TypeError):
            replace_many_masked("text", {"a": not_str})  # type: ignore[dict-item]

    @pytest.mark.parametrize(
        "not_dict",
        [(("a", "b"),), [("a", "b")], "abc", 123, None],
        ids=["tuple-of-pairs", "list-of-pairs", "str", "int", "none"],
    )
    def test_non_dict_replacements_raise_type_error(self, not_dict: object) -> None:
        with pytest.raises(TypeError):
            replace_many_masked("text", not_dict)  # type: ignore[arg-type]

    def test_empty_key_raises_value_error(self) -> None:
        """The ``find_patterns``/``replace_many`` contract carried over:
        an empty key would match at every position, refused with
        exactly ``ValueError("empty pattern")``."""
        with pytest.raises(ValueError, match="^empty pattern$"):
            replace_many_masked("some text", {"ok": "x", "": "y"})

    def test_lone_surrogates_are_refused_at_the_argument_boundary(self) -> None:
        """The standard str-in boundary, paid by the text, every key,
        every value, AND the mask alike (the mask is a str-in argument
        of the same class; the length arithmetic runs on characters)."""
        with pytest.raises(UnicodeEncodeError):
            replace_many_masked("abc\ud800", {"a": "b"})
        with pytest.raises(UnicodeEncodeError):
            replace_many_masked("abc", {"a\ud800": "b"})
        with pytest.raises(UnicodeEncodeError):
            replace_many_masked("abc", {"a": "b\ud800"})
        with pytest.raises(UnicodeEncodeError):
            replace_many_masked("abc", {"a": "b"}, mask="\ud800")

    def test_an_invalid_mask_still_raises_with_an_empty_replacements_dict(self) -> None:
        """The Rust wrapper (src/py/search.rs) validates ``mask`` BEFORE the
        empty-dict early return: ``replace_many_masked("abc", {}, mask="**")``
        raises ``ValueError``, not the identity-return every other empty-dict
        case gets (pinned above). Mask validation fires eagerly regardless
        of whether there's anything to redact, so this pins that ordering
        against silent regression in either direction."""
        for bad in ("", "**"):
            with pytest.raises(ValueError, match="^mask must be exactly one character"):
                replace_many_masked("abc", {}, mask=bad)

    @pytest.mark.parametrize(
        "not_str",
        [b"*", bytearray(b"*"), 123, None],
        ids=["bytes", "bytearray", "int", "none"],
    )
    def test_non_str_mask_raises_type_error(self, not_str: object) -> None:
        """``mask`` is an ordinary ``&str`` pyo3 parameter exactly like
        ``text``/keys/values, all three of which get a dedicated non-str
        ``TypeError`` test above; ``mask``'s TYPE (as opposed to its
        content/encoding, both already covered) never had one."""
        with pytest.raises(TypeError):
            replace_many_masked("abc", {"a": "b"}, mask=not_str)  # type: ignore[arg-type]
