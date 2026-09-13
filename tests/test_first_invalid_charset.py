"""Contract gate for ``tors.first_invalid_charset``: batch codepoint-set
validation of identifier-style rules, one GIL-released pass per batch.

``tors.first_invalid_charset(items, *, first=None, rest)`` answers the index
into ``items`` of the first item that is not built entirely from the
caller's two sets — ``first`` the set of codepoints allowed at position 0,
``rest`` the set allowed at every position after it — or ``-1`` when every
item passes. The motivating consumer (TaskQ's enqueue path) validates
identifier-shaped strings with anchored regexes (schema identifiers, queue
names, keyed-ref names, tags), each check measured at 84-950 ns and the
whole per-enqueue cluster under a single ``py.detach`` round trip: per-item
tors calls would be slower than the regexes they replace, so this function
is batch-only by design — one detach, one pass over all items, the only
winnable shape (the honest sizing recorded in docs/api.md's section).

Semantics, pinned precisely:

1. **Membership is per codepoint**: an item passes when its first codepoint
   is in ``first`` (or in ``rest`` when ``first`` is ``None``, the uniform
   spelling: one set governs every position) and every later codepoint is
   in ``rest``. Never per byte.
2. **The sets are data, not patterns**: plain strings of permitted
   codepoints — no ranges, no escapes, no classes (an ``^[a-z_][a-z0-9_]*$``
   rule is spelled by listing the codepoints; Unicode-category classes like
   ``\\w`` would need property tables, the charter boundary already drawn
   for lexical data in docs/design.md). Duplicate codepoints in a set
   spelling are harmless and their order is irrelevant (a set, however
   spelled).
3. **An empty item is an offender**, wherever it sits and whatever the sets
   allow: there is no codepoint at position 0 to check.
4. ``first=""`` allows nothing at position 0, so every item offends;
   ``rest=""`` allows nothing after position 0, so under the uniform
   spelling every item offends, and with a non-empty ``first`` only
   single-codepoint items drawn from ``first`` can pass (the positional
   rule applied literally, pinned below).
5. **The return is the first offending item's index** — an index into
   ``items``, never a position within an item — and the scan short-circuits
   there: no promise about work done past the first offender (though the
   GIL-held argument walk does traverse the whole list: a bad entry
   anywhere raises at the boundary, past a first offender or not).
6. An empty ``items`` sequence answers ``-1`` (vacuously valid), including
   under spellings where every item would offend.

Contract decisions at the argument boundary (each pinned below):

- ``items`` is a sequence of ``str``: a ``list`` or ``tuple`` (any
  ``Sequence``) is accepted; a bare ``str`` is refused with ``TypeError``
  (it would silently validate its own characters, the ``separators=``
  precedent); non-sequence inputs (``dict``, ``set``, generators, scalars)
  raise ``TypeError``; a non-``str`` entry raises ``TypeError`` (pyo3
  extraction conventions);
- ``first`` and ``rest`` must be exactly ``str`` (``TypeError`` otherwise),
  both keyword-only, ``rest`` required and ``first`` defaulting to ``None``;
- lone surrogates (a ``str`` CPython can hold but UTF-8 cannot encode) are
  refused with ``UnicodeEncodeError`` before any Rust code runs, the
  standard str-in boundary, paid by every item and by ``first``/``rest``
  alike.

No single stdlib primitive has these semantics, so the contract is proven
the module decision's prescribed three ways: (a) a pure-Python membership
loop (the shared oracle, ``reference.reference_first_invalid_charset`` in
tests/reference.py) over hypothesis-driven alphabets including multi-byte
and astral codepoints, (b) the equivalent anchored regexes rebuilt from the
same set halves over the three real TaskQ rule shapes (a second, independent
oracle: the ``re`` engine itself), and (c) golden cases with exact expected
indices. The GIL-release claim (one ``py.detach`` around the whole batch
pass; the GIL-held residue is the O(items) argument walk plus a single int
return) is pinned ceiling-only by tests/test_gil_release.py (the
``utf8_is_valid`` cell class: at realistic batch sizes the whole call sits
far under the 10 ms ping floor); the criterion ladder for the Rust core
alone is the ``first_invalid_charset`` group in benches/search.rs.
"""

from __future__ import annotations

import itertools
import re
import string
from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reference import reference_first_invalid_charset
from tors import first_invalid_charset

# --- The three real TaskQ rule shapes ----------------------------------------
#
# TaskQ validates identifier-shaped strings with three anchored regexes:
# _IDENT_RE (taskq.constants: schema/table/column names, ~100 call sites),
# _QUEUE_NAME_RE (backend/_protocol: per enqueue), and _KEYED_KEY_RE
# (taskq.constants: per rate-limit acquire). Their character classes,
# re-expressed as the two halves this API takes — spelled from string's own
# classes rather than importing TaskQ, since tors's tests stay
# self-contained; the regexes below are rebuilt from the same halves, so the
# differentials pin the equivalence itself, not a transcription.

IDENT_FIRST = string.ascii_letters + "_"
IDENT_REST = string.ascii_letters + string.digits + "_"
QUEUE_FIRST = string.ascii_letters + string.digits + "_"
QUEUE_REST = QUEUE_FIRST + ".-"
KEYED_REST = string.ascii_letters + string.digits + "_-:."

# (first, rest) for each rule; the keyed-key rule is uniform (one set for
# every position: first=None), which is exactly how a first/rest API
# expresses it.
_RULES: tuple[tuple[str | None, str], ...] = (
    (IDENT_FIRST, IDENT_REST),
    (QUEUE_FIRST, QUEUE_REST),
    (None, KEYED_REST),
)


def _class(chars: str) -> str:
    """A regex character class over ``chars``, each codepoint escaped so
    only the set spelling flows into the pattern (no ranges, no classes:
    exactly what the API itself accepts)."""
    return "[" + "".join(re.escape(ch) for ch in chars) + "]"


def _anchored(first: str | None, rest: str) -> re.Pattern[str]:
    """The regex equivalent of one rule, rebuilt from the same halves: one
    first-class codepoint then zero-plus rest-class codepoints, ``\\A``/``\\Z``
    anchored (TaskQ's own anchoring: ``$`` also matches immediately before a
    trailing newline, which is why its regexes spell ``\\A``/``\\Z``)."""
    head = _class(rest if first is None else first)
    return re.compile(rf"\A{head}{_class(rest)}*\Z")


def _regex_first_invalid(pattern: re.Pattern[str], items: Sequence[str]) -> int:
    """The same first-offender-index contract over the regex oracle."""
    for idx, item in enumerate(items):
        if pattern.match(item) is None:
            return idx
    return -1


# The shared rule battery: valid shapes for each rule, the shapes that split
# them (a leading digit splits identifier from queue-name; ":" and "-"
# split queue-name from keyed-key; "." splits identifier from queue-name),
# the TaskQ trap item (a trailing newline), non-ASCII, and an empty item.
_RULE_ITEMS: list[str] = [
    "taskq",
    "jobs",
    "worker_id",
    "queue_eu",
    "job_42",
    "_private",
    "default",
    "events.v2",
    "tag-1",
    "base_name:key",
    "9lives",
    "1st_floor",
    "café",
    "bad name",
    "taskq\n",
    "DROP;TABLE",
    "",
    "a" * 40,
    "東京",
]


class TestTaskQRuleShapes:
    """The motivating claim: the three real rules express directly through
    the API, each pinned against the anchored regex rebuilt from the same
    halves (a second, independent oracle — the ``re`` engine — beyond the
    membership-loop reference), with the first offender's index
    hand-derived per rule."""

    def test_identifier_rule_equals_the_anchored_regex(self) -> None:
        # Hand-derived first offender over _RULE_ITEMS under the identifier
        # rule: everything through "default" is a plain identifier, and
        # "events.v2" is the first item with a codepoint (".") outside the
        # identifier sets.
        got = first_invalid_charset(_RULE_ITEMS, first=IDENT_FIRST, rest=IDENT_REST)
        assert got == 7
        assert got == _regex_first_invalid(_anchored(IDENT_FIRST, IDENT_REST), _RULE_ITEMS)
        assert got == reference_first_invalid_charset(_RULE_ITEMS, IDENT_FIRST, IDENT_REST)

    def test_queue_name_rule_equals_the_anchored_regex(self) -> None:
        # Under the queue-name rule the dot and hyphen items pass; the first
        # offender is "base_name:key" — ":" is the load-bearing exclusion
        # (TaskQ: a queue named "foo:eu" would collide with the flat
        # "taskq:global:queue:foo:eu" concurrency-cap namespace).
        got = first_invalid_charset(_RULE_ITEMS, first=QUEUE_FIRST, rest=QUEUE_REST)
        assert got == 9
        assert got == _regex_first_invalid(_anchored(QUEUE_FIRST, QUEUE_REST), _RULE_ITEMS)
        assert got == reference_first_invalid_charset(_RULE_ITEMS, QUEUE_FIRST, QUEUE_REST)

    def test_keyed_key_rule_equals_the_anchored_regex(self) -> None:
        # The keyed-key rule is uniform — one set for every position — which
        # is the first=None spelling; ":" and a leading digit are legal, so
        # the first offender is "café" (non-ASCII, outside the ASCII sets).
        got = first_invalid_charset(_RULE_ITEMS, rest=KEYED_REST)
        assert got == 12
        assert got == _regex_first_invalid(_anchored(None, KEYED_REST), _RULE_ITEMS)
        assert got == reference_first_invalid_charset(_RULE_ITEMS, None, KEYED_REST)

    @given(st.lists(st.text(max_size=8), max_size=8))
    @settings(max_examples=200)
    def test_each_rule_equals_its_anchored_regex_over_generated_items(
        self, items: list[str]
    ) -> None:
        # The full-coverage differential: arbitrary generated item lists
        # (any script, any marks, any widths) under each of the three rules,
        # tors's answer against the regex oracle's — the equivalence the
        # three batteries above pin at hand-derived points, proven over the
        # whole generated space.
        for first, rest in _RULES:
            assert first_invalid_charset(items, first=first, rest=rest) == (
                _regex_first_invalid(_anchored(first, rest), items)
            )

    def test_a_trailing_newline_is_an_offender_under_every_rule(self) -> None:
        # TaskQ's trap: "^...$" also matches immediately before a trailing
        # newline, so "taskq\n" once passed a queue-name check; its regexes
        # moved to \A/\Z. The charset rule has no anchoring question at all
        # — "\n" is simply not in any of the three sets — so the trap item
        # is an offender here by construction, and the \A/\Z regex agrees.
        for first, rest in _RULES:
            assert first_invalid_charset(["taskq\n"], first=first, rest=rest) == 0
            assert _anchored(first, rest).match("taskq\n") is None

    def test_the_unicode_category_boundary_is_real(self) -> None:
        # TaskQ's _TAG_RE is \w-based — a Unicode-category class — which is
        # exactly the expressiveness this API declines (property tables are
        # outside the charter, docs/design.md's lexical-data boundary). The
        # demonstration: "café" is a valid tag under \w but an offender
        # under an ASCII set spelling — and a caller who wants "é" allowed
        # simply spells it into the set, which IS expressible (per-codepoint
        # data, not patterns).
        ascii_tag_rest = string.ascii_letters + string.digits + "_-"
        assert first_invalid_charset(["café"], rest=ascii_tag_rest) == 0
        assert first_invalid_charset(["café"], rest=ascii_tag_rest + "é") == -1


# --- The golden semantics battery ---------------------------------------------
#
# Every expected index hand-derived; the membership-loop oracle cross-checks
# each row so a wrong pin fails loudly instead of laundering through.

_GOLDEN_CASES: list[tuple[list[str], str | None, str, int]] = [
    ([], None, "a", -1),
    ([], "", "a", -1),
    (["a"], None, "a", -1),
    (["b"], None, "a", 0),
    (["a", "b", "a"], None, "a", 1),
    (["a", "a", "b"], None, "a", 2),
    (["b", "c"], None, "a", 0),
    ([""], None, "a", 0),
    (["a", "", "a"], None, "a", 1),
    ([""], "", "a", 0),
    (["a"], "", "a", 0),
    (["a", "b"], "", "a", 0),
    (["a"], None, "", 0),
    (["a", "aa"], "a", "", 1),
    (["a"], "a", "", -1),
    (["Aaa"], "AB", "ab", -1),
    (["aAa"], "AB", "ab", 0),
    (["A"], "AB", "ab", -1),
    (["aa"], "a", "ab", -1),
    (["ba"], "a", "ab", 0),
    (["ab"], "a", "a", 0),
]

_GOLDEN_IDS = [
    "empty-batch-is-vacuously-valid",
    "empty-batch-stays-valid-under-first-empty",
    "single-valid-item",
    "single-invalid-item",
    "first-offender-mid-list",
    "offender-at-the-end",
    "first-of-several-offenders",
    "empty-item-is-an-offender",
    "empty-item-mid-list",
    "empty-item-offends-regardless-of-first",
    "first-empty-allows-nothing-at-position-0",
    "first-empty-makes-every-item-the-first-offender",
    "rest-empty-allows-nothing-anywhere-uniform",
    "rest-empty-with-a-first-set-allows-single-codepoint-items-only",
    "rest-empty-single-codepoint-item-from-first-passes",
    "first-only-codepoint-at-position-0",
    "first-only-codepoint-fails-at-position-1",
    "single-first-only-codepoint-item",
    "first-and-rest-are-independent-sets",
    "rest-only-codepoint-not-allowed-at-position-0",
    "return-is-the-item-index-not-the-char-position",
]


@pytest.mark.parametrize(("items", "first", "rest", "expected"), _GOLDEN_CASES, ids=_GOLDEN_IDS)
def test_golden_battery(
    items: list[str], first: str | None, rest: str, expected: int
) -> None:
    """The fixed anchor of the contract: every golden index is hand-derived
    and cross-checked against the membership-loop oracle, so a pin and the
    oracle disagreeing fails here rather than silently laundering a wrong
    expectation into the suite."""
    assert first_invalid_charset(items, first=first, rest=rest) == expected
    assert expected == reference_first_invalid_charset(items, first, rest)


# --- The per-codepoint battery (the bytes-vs-codepoints crux) -----------------

_E_ACUTE = chr(0xE9)  # precomposed é: 2 UTF-8 bytes, 1 codepoint
_MATH_X = chr(0x1D54F)  # mathematical double-struck X: 4 bytes, > U+FFFF
_CRAB = chr(0x1F980)  # crab emoji: 4 bytes, > U+FFFF

_MULTIBYTE_CASES: list[tuple[list[str], str | None, str, int]] = [
    ([_E_ACUTE], None, _E_ACUTE, -1),
    (["e" + _E_ACUTE], None, _E_ACUTE, 0),
    ([_E_ACUTE, _E_ACUTE], None, _E_ACUTE, -1),
    (["東京"], None, "東京", -1),
    (["東b京"], None, "東京", 0),
    ([_CRAB + _CRAB], None, _CRAB, -1),
    ([_MATH_X + "x"], None, _MATH_X, 0),
    ([_E_ACUTE + "東" + _CRAB], None, _E_ACUTE + "東" + _MATH_X, 0),
]

_MULTIBYTE_IDS = [
    "two-byte-codepoint-membership",
    "ascii-codepoint-outside-the-set",
    "two-byte-run",
    "three-byte-codepoint-membership",
    "ascii-between-multibyte-members",
    "four-byte-astral-codepoint-membership",
    "astral-then-ascii-tail-offends",
    "mixed-widths-one-member-missing",
]


@pytest.mark.parametrize(
    ("items", "first", "rest", "expected"), _MULTIBYTE_CASES, ids=_MULTIBYTE_IDS
)
def test_membership_is_per_codepoint_over_multibyte_sets(
    items: list[str], first: str | None, rest: str, expected: int
) -> None:
    """The bytes-vs-codepoints crux: every set and item here is non-ASCII,
    spanning 2-, 3-, and 4-byte UTF-8 widths (the astral plane included), so
    a byte-level membership check answers a different question on every
    row; the expected indices are hand-derived in codepoint units and
    cross-checked against the oracle, which spells the same rule per
    codepoint by construction."""
    assert first_invalid_charset(items, first=first, rest=rest) == expected
    assert expected == reference_first_invalid_charset(items, first, rest)


# --- The set-spelling contract ------------------------------------------------


def test_duplicate_codepoints_in_a_set_spelling_are_harmless() -> None:
    """A set spelled with duplicates is the same set: "aabb__" admits
    exactly what "ab_" admits (the first row: both items valid), and
    excludes exactly what it excludes (the second: "xa" offends at
    position 0 either way)."""
    assert first_invalid_charset(["ab_1", "_x"], first="aabb__", rest="aabb__11") == -1
    assert first_invalid_charset(["ab_1", "xa"], first="aabb__", rest="aabb__11") == 1


def test_set_spelling_order_is_irrelevant() -> None:
    """Three permuted spellings of the same two sets answer identically;
    the expected index (1: "xa" is the first item "x" breaks) is
    hand-derived, not computed from the call."""
    items = ["ab_1", "xa", "1a"]
    answers = [
        first_invalid_charset(items, first=fs, rest=rs)
        for fs, rs in (("ab_", "ab_1"), ("_ba", "1_ba"), ("a_b", "ba_1"))
    ]
    assert answers == [1, 1, 1]


# --- The argument-boundary contract -------------------------------------------


class TestArgumentContract:
    def test_a_tuple_of_items_is_accepted_like_a_list(self) -> None:
        """``items`` is a sequence: the tuple spelling answers exactly what
        the list spelling answers (the ``separators=`` Sequence precedent),
        pinned on the same hand-derived index."""
        assert first_invalid_charset(("job_42", "9bad"), first=IDENT_FIRST, rest=IDENT_REST) == 1
        assert first_invalid_charset(["job_42", "9bad"], first=IDENT_FIRST, rest=IDENT_REST) == 1

    @pytest.mark.parametrize(
        "not_items",
        ["abc", b"abc", bytearray(b"abc"), 123, None, {"a": 1}, {"a"}, range(3), (c for c in ())],
        ids=[
            "bare-str",
            "bytes",
            "bytearray",
            "int",
            "none",
            "dict",
            "set",
            "range",
            "generator",
        ],
    )
    def test_non_sequence_items_raise_type_error(self, not_items: object) -> None:
        """``items`` is a sequence of ``str``. A bare ``str`` is refused on
        purpose: it would otherwise validate its own characters one by one,
        silently answering a question nobody asked (the ``separators=``
        precedent's bare-str refusal). Sequences whose entries are not
        ``str`` (bytes, bytearray, range), non-sequences (dict, set,
        scalars), and generators all raise ``TypeError``. No message match:
        the type is the contract."""
        with pytest.raises(TypeError):
            first_invalid_charset(not_items, rest="a")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_entry",
        [b"x", bytearray(b"x"), 123, None],
        ids=["bytes-entry", "bytearray-entry", "int-entry", "none-entry"],
    )
    def test_non_str_entries_raise_type_error(self, bad_entry: object) -> None:
        with pytest.raises(TypeError):
            first_invalid_charset(["ok", bad_entry], rest="ok")  # type: ignore[list-item]

    def test_non_str_first_and_rest_raise_type_error(self) -> None:
        with pytest.raises(TypeError):
            first_invalid_charset(["a"], first=1, rest="a")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            first_invalid_charset(["a"], first=b"x", rest="a")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            first_invalid_charset(["a"], rest=1)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            first_invalid_charset(["a"], rest=b"x")  # type: ignore[arg-type]

    def test_first_is_keyword_only_rest_is_required_items_may_be_keyword(self) -> None:
        """The spec's shape: ``first``/``rest`` keyword-only (a misread
        positional argument cannot silently swap the two sets), ``rest``
        required (there is no sensible default set), ``first`` defaulting
        to ``None`` (the uniform spelling), ``items`` passable by name."""
        with pytest.raises(TypeError):
            first_invalid_charset(["a"], "a", "a")  # type: ignore[misc]
        with pytest.raises(TypeError):
            first_invalid_charset(["a"], first="a")  # type: ignore[call-arg]
        assert first_invalid_charset(items=["a"], rest="a") == -1
        # The default and an explicit None are the same call.
        assert first_invalid_charset(["a"], rest="a") == first_invalid_charset(
            ["a"], first=None, rest="a"
        )

    def test_lone_surrogates_are_refused_at_the_argument_boundary(self) -> None:
        """Lone surrogates (a ``str`` CPython can hold but UTF-8 cannot
        encode) are refused with ``UnicodeEncodeError`` before any Rust
        code runs, the standard str-in boundary, paid by every item and by
        ``first``/``rest`` alike."""
        with pytest.raises(UnicodeEncodeError):
            first_invalid_charset(["abc\ud800"], rest="abc")
        with pytest.raises(UnicodeEncodeError):
            first_invalid_charset(["ok"], first="a\ud800", rest="a")
        with pytest.raises(UnicodeEncodeError):
            first_invalid_charset(["ok"], rest="a\ud800")

    def test_the_items_walk_validates_the_whole_list_before_the_scan(self) -> None:
        """The scan short-circuits at the first offender; the GIL-held
        argument walk does not (the find_patterns pattern-walk discipline:
        every entry is borrowed up front). A bad entry anywhere in the list
        raises at the boundary even when an earlier item already offends —
        the short-circuit is a scan property, never an
        argument-validation one."""
        with pytest.raises(UnicodeEncodeError):
            first_invalid_charset(["bad item", "x\ud800"], rest="abc")
        with pytest.raises(TypeError):
            first_invalid_charset(["bad item", 123], rest="abc")  # type: ignore[list-item]


# --- Parts (a)+(b): the differential oracle over generated inputs -------------


# An alphabet spanning every UTF-8 width: 1-byte "ab1_", 2-byte é, 3-byte
# CJK, 4-byte emoji (the astral plane included), so generated items and sets
# exercise per-codepoint membership at every width combination.
_ALPHABET = "ab1_" + _E_ACUTE + "東京" + _CRAB


@st.composite
def _set_spelling(draw: st.DrawFn, alphabet: list[str]) -> str:
    """A set spelling over ``alphabet``: a drawn subset, re-spelled with
    every codepoint duplicated and the order permuted, so one set arrives
    under many spellings (the duplicate/order-irrelevance contract
    exercised on every draw, and the empty set reachable when the subset
    draws empty)."""
    chosen = draw(st.lists(st.sampled_from(alphabet), min_size=0, max_size=len(alphabet)))
    doubled = chosen + chosen
    return "".join(draw(st.permutations(doubled)))


@st.composite
def _items_first_rest(draw: st.DrawFn) -> tuple[list[str], str | None, str]:
    """Item lists over the multi-byte alphabet (empty items included) with
    independently drawn ``first`` (``None`` half the time: the uniform
    spelling faces the oracle as often as the two-set spelling) and
    ``rest``, both under permuted, duplicate-bearing spellings."""
    items = draw(st.lists(st.text(alphabet=_ALPHABET, max_size=6), max_size=6))
    first = draw(st.one_of(st.none(), _set_spelling(list(_ALPHABET))))
    rest = draw(_set_spelling(list(_ALPHABET)))
    return items, first, rest


@given(_items_first_rest())
@settings(max_examples=500)
def test_matches_the_membership_reference_over_multibyte_alphabets(
    items_first_rest: tuple[list[str], str | None, str],
) -> None:
    """The differential proof over the multi-byte alphabet: tors's answer
    must equal the brute-force membership loop exactly for every generated
    batch, set pair, and spelling. Any semantics bug (a byte-for-codepoint
    swap, a first/rest confusion, an empty-item or empty-set mishandling,
    a spelling-order sensitivity) breaks this property."""
    items, first, rest = items_first_rest
    assert first_invalid_charset(items, first=first, rest=rest) == (
        reference_first_invalid_charset(items, first, rest)
    )


@st.composite
def _arbitrary_items_first_rest(draw: st.DrawFn) -> tuple[list[str], str | None, str]:
    """Arbitrary-Unicode items (hypothesis's full ``st.text``: any script,
    any marks, no alphabet bias) with sets drawn from the codepoints the
    items themselves use plus random extras, so membership hits and misses
    both occur over text no fixed alphabet can generate."""
    items = draw(st.lists(st.text(max_size=6), max_size=5))
    pool = {ch for item in items for ch in item}
    pool.update(ch for ch in draw(st.text(min_size=1, max_size=6)))
    alphabet = sorted(pool) or ["x"]
    first = draw(st.one_of(st.none(), _set_spelling(alphabet)))
    rest = draw(_set_spelling(alphabet))
    return items, first, rest


@given(_arbitrary_items_first_rest())
@settings(max_examples=300)
def test_matches_the_membership_reference_over_arbitrary_unicode(
    items_first_rest: tuple[list[str], str | None, str],
) -> None:
    """The arbitrary-Unicode differential: the coverage class the fixed
    alphabet cannot reach (any codepoint class, mixed widths, combining
    marks anywhere), still under exact index equality with the oracle."""
    items, first, rest = items_first_rest
    assert first_invalid_charset(items, first=first, rest=rest) == (
        reference_first_invalid_charset(items, first, rest)
    )


def _valid(item: str, first: str | None, rest: str) -> bool:
    """The positional rule, spelled inline (a per-item restatement of the
    contract the oracle spells as a loop): non-empty, first codepoint in
    the first set (or rest, when uniform), every later codepoint in rest."""
    allowed_first = rest if first is None else first
    return bool(item) and item[0] in allowed_first and all(ch in rest for ch in item[1:])


@given(_items_first_rest())
@settings(max_examples=300)
def test_minus_one_iff_all_valid_and_the_reported_index_is_the_first_offender(
    items_first_rest: tuple[list[str], str | None, str],
) -> None:
    """The structural invariant, stated on its own: ``-1`` exactly when
    every item is valid; any other answer is an index whose item is invalid
    and whose every predecessor is valid (the first-offender property the
    short-circuit promises)."""
    items, first, rest = items_first_rest
    got = first_invalid_charset(items, first=first, rest=rest)
    assert (got == -1) == all(_valid(item, first, rest) for item in items)
    if got != -1:
        assert not _valid(items[got], first, rest)
        assert all(_valid(item, first, rest) for item in items[:got])


def test_every_small_items_list_and_set_spelling_matches_the_reference() -> None:
    """The deterministic sweep (the suite's exhaustive-small-alphabet
    idiom): every items list of size 0-3 over ``{"", "a", "b", "ab"}``
    crossed with ``first`` in ``{None, "", "a", "ab"}`` and ``rest`` in
    ``{"", "a", "ab", "b"}`` — 85 x 16 pairs, the complete small space of
    empty/short-item and empty/partial-set interactions, no sampling at
    all."""
    pool = ["", "a", "b", "ab"]
    items_lists: list[list[str]] = [[]]
    for size in (1, 2, 3):
        items_lists.extend(list(combo) for combo in itertools.product(pool, repeat=size))
    for items in items_lists:
        for first in (None, "", "a", "ab"):
            for rest in ("", "a", "ab", "b"):
                assert first_invalid_charset(items, first=first, rest=rest) == (
                    reference_first_invalid_charset(items, first, rest)
                ), (items, first, rest)


# --- Batch-scale batteries ------------------------------------------------------


def _ident_batch(count: int, poison_at: int | None = None) -> list[str]:
    """``count`` deterministic identifier-shaped items (the job/queue/worker/
    tag spellings an enqueue path validates), optionally with one invalid
    item ("bad name": a space) spliced in at ``poison_at``."""
    shapes = ("job_{n}", "queue_eu_{n}", "worker_{n}", "tag_{n}")
    items = [shapes[n % 4].format(n=n) for n in range(count)]
    if poison_at is not None:
        items[poison_at] = "bad name"
    return items


class TestBatchScale:
    """The batch-only shape the function exists for: hundreds of items under
    one call, all-valid (the full-pass worst case) and offender-at-the-end
    (the no-early-exit worst case), at the sizes the motivating consumer's
    tag batches and bulk pre-flights actually reach."""

    @pytest.mark.parametrize("count", [100, 1000], ids=["100-items", "1000-items"])
    def test_all_valid_batches_answer_minus_one(self, count: int) -> None:
        items = _ident_batch(count)
        assert first_invalid_charset(items, first=IDENT_FIRST, rest=IDENT_REST) == -1
        assert reference_first_invalid_charset(items, IDENT_FIRST, IDENT_REST) == -1

    @pytest.mark.parametrize("count", [100, 1000], ids=["100-items", "1000-items"])
    def test_an_offender_at_the_end_of_a_large_batch_is_found(self, count: int) -> None:
        items = _ident_batch(count, poison_at=count - 1)
        assert first_invalid_charset(items, first=IDENT_FIRST, rest=IDENT_REST) == count - 1

    def test_a_second_offender_past_the_first_is_never_reported(self) -> None:
        items = _ident_batch(100, poison_at=42)
        items[43] = "also bad"
        assert first_invalid_charset(items, first=IDENT_FIRST, rest=IDENT_REST) == 42


# --- The docs' worked example, pinned -------------------------------------------


class TestDocsExamples:
    """docs/api.md's first_invalid_charset section, pinned the
    test_docs_examples.py discipline: the literals the doc shows are
    re-derived here against the built extension, so a behavior change that
    would turn the documented example into a lie fails here first."""

    def test_identifier_rule_example(self) -> None:
        ident_first = string.ascii_letters + "_"
        ident_rest = ident_first + string.digits
        assert (
            first_invalid_charset(
                ["job_42", "queue_eu", "9bad"], first=ident_first, rest=ident_rest
            )
            == 2
        )
        assert (
            first_invalid_charset(["job_42", "queue_eu"], first=ident_first, rest=ident_rest)
            == -1
        )

    def test_uniform_rule_example(self) -> None:
        ident_rest = string.ascii_letters + "_" + string.digits
        assert first_invalid_charset(["worker:01", "tag name"], rest=ident_rest + "-:.") == 1
