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

The offender-detail spelling — ``tors.first_invalid_offender(items, *,
first=None, rest) -> tuple[int, int, str] | None`` — is the SAME scan
answering the question a rejection UX asks (the integration survey's
finding: TaskQ's per-character rejection messages name the losing
character and its position, which an item index alone cannot): it returns
``(item_index, char_position, offending_char)`` for the first offending
item's FIRST offending position, ``None`` when every item passes.
``char_position`` counts CODEPOINTS within the item (the family's data
model, membership per codepoint), never UTF-8 byte offsets, and the
offending char is that codepoint as a 1-char ``str``. The empty item
reports ``(i, 0, "")`` — an empty item has no offending character; the
char field is empty exactly when the item is. The consistency invariant
against the int spelling (``None`` iff ``-1``; the item index equal) and
the byte-identical argument boundary (the same shared walk) are pinned
below, in the offender section.

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
- the argument walk is BOUNDED (``content_hash``'s protocol-walk cap, the
  same DoS backstop): the walk materializes every item's handle under the
  GIL before the detached scan runs, so a sequence whose ``__iter__``
  never stops would hold the GIL growing the batch until the process
  dies: past the walk's ceiling the call aborts with a generic
  ``ValueError`` (no cap value leaked), both spellings alike (one shared
  walk, byte-identical refusals).

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

The pinned common alphabets (``tors.CHARSET_B62``, ``CHARSET_B64URL``,
``CHARSET_HEX_LOWER``/``_UPPER``/``_MIXED``) are module data for this
engine, and this file pins them the way the engine itself is pinned: the
scope question — should b62/b64/hex/UUID validators ship? — is answered in
code by constants, not wrapper functions (N wrappers delegating to the
single engine would be pure API surface and maintenance cost), so what
ships is the data: the alphabets worth pinning, byte-exact, plus the
length/uniqueness/subset algebra that makes the family coherent and the
use shapes (base62 ids, JWT segments, hex digests) they exist for.
"""

from __future__ import annotations

import itertools
import re
import string
import subprocess
import sys
from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors
from reference import reference_first_invalid_charset, reference_first_invalid_offender
from tors import first_invalid_charset, first_invalid_offender

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

    def test_nfc_and_nfd_forms_get_different_verdicts_normalize_first(self) -> None:
        """The normalization bypass, pinned: the engine is per-scalar (one
        codepoint at a time, no NFC/NFD folding), so precomposed é (U+00E9,
        one codepoint) and decomposed e+U+0301 (two codepoints) get
        different verdicts under ``rest="é"`` — correct per the data
        model, surprising when NFC/NFD must agree. A caller who needs
        the two forms to agree normalizes first (``tors.normalize`` /
        ``tors.nfc``) before validating. Normalization does not fold
        confusables (visually similar but distinct codepoints stay
        distinct); allow-list exactly the codepoints you mean."""
        assert first_invalid_charset(["é"], rest="é") == -1
        assert first_invalid_charset(["é"], rest="é") == 0
        assert reference_first_invalid_charset(["é"], None, "é") == -1
        assert reference_first_invalid_charset(["é"], None, "é") == 0
        # After NFC both forms are the one-codepoint é, so both pass.
        assert first_invalid_charset([tors.normalize("é")], rest="é") == -1
        assert first_invalid_charset([tors.normalize("é")], rest="é") == -1
        # Confusables stay distinct under NFC, pinned: Cyrillic А (U+0410)
        # looks like Latin A but is a different scalar, and normalize does
        # not fold it — a Latin-only allow-list rejects it before and after
        # normalization; allow-list exactly the codepoints you mean.
        assert first_invalid_charset(["\u0410"], rest="A") == 0
        assert first_invalid_charset([tors.normalize("\u0410")], rest="A") == 0
        assert first_invalid_charset(["\u0410"], rest="\u0410") == -1
        assert reference_first_invalid_charset(["\u0410"], None, "A") == 0

    def test_membership_is_per_codepoint_not_per_grapheme_cluster(self) -> None:
        """Codepoints, not grapheme clusters: a ZWJ family emoji (5
        codepoints), a flag pair (2 regional indicators), and a base +
        combining mark (2 codepoints) each pass only when every
        constituent codepoint is in the set — one missing joiner, one
        missing indicator, one missing combining mark offends, even
        though the item is a single grapheme either way."""
        zwj = "👨\u200d👩\u200d👧"  # 5 codepoints, 1 grapheme
        flag = "🇫🇷"  # 2 regional indicators, 1 grapheme
        assert first_invalid_charset([zwj], rest=zwj) == -1
        assert first_invalid_charset([zwj], rest=zwj.replace("\u200d", "")) == 0
        assert first_invalid_charset([flag], rest=flag) == -1
        assert first_invalid_charset([flag], rest="🇫") == 0
        assert first_invalid_charset(["é"], rest="é") == -1
        assert first_invalid_charset(["é"], rest="é") == 0
        assert first_invalid_charset(["é"], rest="é") == -1
        assert reference_first_invalid_charset([zwj], None, zwj) == -1
        assert reference_first_invalid_charset([flag], None, flag) == -1

    def test_a_10k_non_ascii_set_spelling_validates_correctly(self) -> None:
        """The huge-set build: 10_000 distinct non-ASCII codepoints spell
        a set whose tail sorts/dedups (O(set log set), inside the
        detach) and still validates per codepoint — members pass,
        one missing codepoint offends. Hoist huge spellings to module
        constants (define once, reuse); the per-call build is part of the
        measured band for the few-dozen-codepoint ASCII rules (~64 ns
        fixed), while the 10k-tail sort cost is deferred (correctness
        pinned here, unmeasured beyond the ASCII band)."""
        huge = "".join(chr(cp) for cp in range(0x1000, 0x1000 + 10_000))
        member = chr(0x1000) + chr(0x1000 + 9_999)
        assert first_invalid_charset([member], rest=huge) == -1
        assert first_invalid_charset([member + "e"], rest=huge) == 0
        assert first_invalid_charset(["e"], rest=huge) == 0
        assert reference_first_invalid_charset([member], None, huge) == -1


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
    position 0 either way). Cross-checked against the oracle so a wrong
    hand-derived pin fails here instead of laundering through."""
    assert first_invalid_charset(["ab_1", "_a"], first="aabb__", rest="aabb__11") == -1
    assert reference_first_invalid_charset(["ab_1", "_a"], "aabb__", "aabb__11") == -1
    assert first_invalid_charset(["ab_1", "xa"], first="aabb__", rest="aabb__11") == 1
    assert reference_first_invalid_charset(["ab_1", "xa"], "aabb__", "aabb__11") == 1


def test_set_spelling_order_is_irrelevant() -> None:
    """Three permuted spellings of the same two sets answer identically;
    the expected index (1: "xa" is the first item "x" breaks) is
    hand-derived, not computed from the call, and cross-checked against
    the oracle."""
    items = ["ab_1", "xa", "1a"]
    answers = [
        first_invalid_charset(items, first=fs, rest=rs)
        for fs, rs in (("ab_", "ab_1"), ("_ba", "1_ba"), ("a_b", "ba_1"))
    ]
    assert answers == [1, 1, 1]
    for fs, rs in (("ab_", "ab_1"), ("_ba", "1_ba"), ("a_b", "ba_1")):
        assert reference_first_invalid_charset(items, fs, rs) == 1


def test_regex_special_codepoints_are_plain_data_not_patterns() -> None:
    """The data-not-patterns rule at its sharpest edge, pinned
    deterministically: every regex metacharacter is an ordinary member
    codepoint, spelled and matched literally — a set spelled with the
    whole specials string admits items built from it (no escaping
    semantics anywhere), dropping one metacharacter from the spelling
    makes items carrying it offend, and a metacharacter is a legal
    position-0 codepoint under a ``first`` spelling. The membership
    oracle cross-checks every row, so the pin cannot launder a
    regex-flavored misreading of the set (the hypothesis differentials
    cover this statistically; this row makes it deterministic)."""
    specials = ".*+[](){}|^$-\\"
    items = [".*+", "|^$-\\", "[a]", "x.*"]
    assert first_invalid_charset(items, rest=specials) == 2
    assert reference_first_invalid_charset(items, None, specials) == 2
    without_backslash = specials.replace("\\", "")
    assert first_invalid_charset(["\\"], rest=without_backslash) == 0
    assert reference_first_invalid_charset(["\\"], None, without_backslash) == 0
    assert first_invalid_charset([".*x"], first=".", rest="*x") == -1
    assert reference_first_invalid_charset([".*x"], ".", "*x") == -1


# --- The argument-boundary contract -------------------------------------------


class TestArgumentContract:
    def test_a_tuple_of_items_is_accepted_like_a_list(self) -> None:
        """``items`` is a sequence: the tuple spelling answers exactly what
        the list spelling answers (the ``separators=`` Sequence precedent),
        pinned on the same hand-derived index."""
        assert first_invalid_charset(("job_42", "9bad"), first=IDENT_FIRST, rest=IDENT_REST) == 1
        assert first_invalid_charset(["job_42", "9bad"], first=IDENT_FIRST, rest=IDENT_REST) == 1

    def test_any_sequence_abc_instance_is_accepted_like_a_list(self) -> None:
        """The annotation is ``Sequence[str]`` and the walk casts to the
        sequence protocol, so any ``collections.abc.Sequence`` instance —
        not just ``list``/``tuple`` — answers exactly the list spelling's
        answer. The bare-``str`` refusal is an ``isinstance`` refusal, so a
        ``str`` subclass as ``items`` is refused with it: it would silently
        validate its own characters exactly as a bare ``str`` would."""
        class Boxed(Sequence):
            def __init__(self, xs: list[str]) -> None:
                self._xs = xs

            def __getitem__(self, i: int) -> str:
                return self._xs[i]

            def __len__(self) -> int:
                return len(self._xs)

        assert (
            first_invalid_charset(Boxed(["job_42", "9bad"]), first=IDENT_FIRST, rest=IDENT_REST)
            == 1
        )

        class StrSub(str):
            pass

        with pytest.raises(TypeError):
            first_invalid_charset(StrSub("job_42"), rest=IDENT_REST)  # type: ignore[arg-type]

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


# --- The offender detail spelling: tors.first_invalid_offender -------------------
#
# The integration survey's finding, and the reason this spelling exists:
# TaskQ's rejection UX names the losing CHARACTER and POSITION
# (backend/_protocol.py's _queue_name_offender builds per-character
# messages), and the index-only return blocks that migration — an item
# index says WHICH item lost, never where inside it or on what codepoint.
# The offender spelling is the same engine answering that question:
# ``(item_index, char_position, offending_char)`` for the first offending
# item's FIRST offending position, ``None`` when every item passes. The
# Rust core's one scan produces the detail; the int spelling is the index
# projection of the same scan (the consistency invariant below).
#
# No separate bench, wall, or GIL cells for this spelling, and the
# reasoning is recorded here as the docs record it: the engine, the
# GIL-held argument walk, and the one-detach batch pass are the int
# spelling's exactly, and the only delta is the return — one small tuple,
# built only when an offender is found — so the int spelling's cells
# (tests/test_performance.py's race, the ``first_invalid_charset``
# criterion group, tests/test_gil_release.py's ceiling cell) carry the
# family's performance contract unchanged.

_OFFENDER_CASES: list[tuple[list[str], str | None, str, tuple[int, int, str] | None]] = [
    ([], None, "a", None),
    ([], "", "a", None),
    (["a"], None, "a", None),
    (["b"], None, "a", (0, 0, "b")),
    (["a", "b", "a"], None, "a", (1, 0, "b")),
    (["a", "a", "b"], None, "a", (2, 0, "b")),
    ([""], None, "a", (0, 0, "")),
    (["a", "", "a"], None, "a", (1, 0, "")),
    ([""], "", "a", (0, 0, "")),
    (["a"], "", "a", (0, 0, "a")),
    (["a", "b"], "", "a", (0, 0, "a")),
    (["a"], None, "", (0, 0, "a")),
    (["a", "aa"], "a", "", (1, 1, "a")),
    (["a"], "a", "", None),
    (["Aaa"], "AB", "ab", None),
    (["aAa"], "aAB", "ab", (0, 1, "A")),
    (["A"], "AB", "ab", None),
    (["aa"], "a", "ab", None),
    (["ba"], "a", "ab", (0, 0, "b")),
    (["ab"], "a", "a", (0, 1, "b")),
    (["axc"], None, "abc", (0, 1, "x")),
    ([_E_ACUTE], None, _E_ACUTE, None),
    (["e" + _E_ACUTE], None, _E_ACUTE, (0, 0, "e")),
    ([_CRAB + "x"], None, _CRAB, (0, 1, "x")),
    ([_E_ACUTE + "東" + _CRAB], None, _E_ACUTE + "東" + _MATH_X, (0, 2, _CRAB)),
    (["\U0001F1EB\U0001F1F7"], None, "\U0001F1EB", (0, 1, "\U0001F1F7")),
]

_OFFENDER_IDS = [
    "empty-batch-is-vacuously-valid",
    "empty-batch-stays-valid-under-first-empty",
    "single-valid-item",
    "single-item-head-offender-names-position-zero",
    "first-offender-mid-list",
    "offender-at-the-end",
    "empty-item-offender-is-index-position-zero-empty-char",
    "empty-item-mid-list",
    "empty-item-offends-regardless-of-first",
    "first-empty-head-offender-names-position-zero",
    "first-empty-makes-every-item-the-first-offender",
    "rest-empty-uniform-head-offender",
    "rest-empty-offender-at-position-one",
    "rest-empty-single-codepoint-item-from-first-passes",
    "first-only-codepoint-at-position-0-passes",
    "first-only-codepoint-at-position-1-is-the-offender",
    "single-first-only-codepoint-item",
    "first-and-rest-are-independent-sets",
    "rest-only-codepoint-not-allowed-at-position-zero",
    "the-detail-names-the-char-position-not-just-the-item",
    "uniform-offender-past-the-head",
    "two-byte-codepoint-member-passes",
    "ascii-head-outside-a-non-ascii-set",
    "astral-member-then-ascii-offender-at-codepoint-position-one",
    "mixed-widths-offender-past-multibyte-codepoints",
    "flag-partial-offender-is-mid-grapheme",
]


@pytest.mark.parametrize(
    ("items", "first", "rest", "expected"), _OFFENDER_CASES, ids=_OFFENDER_IDS
)
def test_offender_golden_battery(
    items: list[str], first: str | None, rest: str, expected: tuple[int, int, str] | None
) -> None:
    """The fixed anchor of the offender contract: every tuple hand-derived
    (the item index, the codepoint position the rule broke at, the
    codepoint there) and cross-checked against the offender oracle, so a
    wrong pin fails loudly instead of laundering through. The empty-item
    rows pin the spelling decision: ``(i, 0, "")`` — an empty item has no
    offending character; the char field is empty exactly when the item
    is."""
    assert first_invalid_offender(items, first=first, rest=rest) == expected
    assert expected == reference_first_invalid_offender(items, first, rest)


def test_char_position_is_a_codepoint_index_not_a_byte_offset() -> None:
    """The position the tuple reports indexes CODEPOINTS within the item —
    the family's data model, membership per codepoint — never UTF-8 byte
    offsets: over items whose offenders sit past multibyte codepoints the
    codepoint position and the byte offset disagree on every row, and the
    pinned answers are the codepoint ones (a byte-counted-position
    regression fails here). The offending char is that codepoint as a
    1-char str, a 4-byte astral codepoint included."""
    # "x" sits at codepoint position 1 of an item whose head is a 4-byte
    # codepoint: its byte offset would be 4.
    assert first_invalid_offender([_CRAB + "x"], rest=_CRAB) == (0, 1, "x")
    # the crab at codepoint position 2, past a 2-byte and a 3-byte
    # codepoint: its byte offset would be 5.
    item = _E_ACUTE + "東" + _CRAB
    assert first_invalid_offender([item], rest=_E_ACUTE + "東" + _MATH_X) == (0, 2, _CRAB)
    # the offending char field is a 1-char str even for an astral codepoint
    got = first_invalid_offender(["ok_" + _CRAB], rest="ok_")
    assert got == (0, 3, _CRAB)
    assert len(got[2]) == 1


def test_offender_position_may_land_inside_a_grapheme_cluster_dont_slice() -> None:
    """The mid-grapheme pin: a flag pair is one grapheme but two codepoints,
    so the partial flag under ``rest="🇫"`` offends at codepoint 1 — inside
    the grapheme. Do not slice the item at that position for display (it
    would split the grapheme); build messages from ``(item, char)`` — the
    tuple already carries the losing codepoint."""
    flag = "\U0001F1EB\U0001F1F7"  # 🇫🇷: 2 regional indicators, 1 grapheme
    assert first_invalid_offender([flag], rest="\U0001F1EB") == (0, 1, "\U0001F1F7")
    assert first_invalid_offender([flag], rest="\U0001F1EB") == reference_first_invalid_offender(
        [flag], None, "\U0001F1EB"
    )
    assert first_invalid_charset([flag], rest="\U0001F1EB") == 0


def test_empty_batch_is_minus_one_even_when_every_item_would_offend() -> None:
    """The vacuous-validity edge, pinned explicitly: an empty batch answers
    ``-1``/``None`` even under spellings where every item would offend
    (``first=""``, ``rest=""``) — the api.md clause."""
    for first, rest in (("", "a"), ("a", ""), ("", ""), (None, "")):
        assert first_invalid_charset([], first=first, rest=rest) == -1
        assert first_invalid_offender([], first=first, rest=rest) is None


def test_set_arg_errors_beat_the_items_walk_first_beats_rest() -> None:
    """The both-bad precedence pin (pyo3 left-to-right extraction order):
    ``first``/``rest`` conversion errors fire before the GIL-held items walk
    validates entries, and ``first`` beats ``rest`` — so a call that is wrong
    on two axes at once reports the set argument, deterministically."""
    # A bad entry plus a bad rest type: the rest TypeError wins (not the
    # entry's TypeError).
    with pytest.raises(TypeError, match="not an instance of 'str'"):
        first_invalid_charset(["ok", 123], rest=123)  # type: ignore[arg-type,list-item]
    # A bad entry plus a bad first type: the first TypeError wins.
    with pytest.raises(TypeError, match="not an instance of 'str'"):
        first_invalid_charset(["ok", 123], first=123, rest="a")  # type: ignore[arg-type,list-item]
    # first beats rest: the reported value is first's (int vs bytes
    # disambiguate the two otherwise-identical messages).
    try:
        first_invalid_charset(["a"], first=123, rest=b"x")  # type: ignore[arg-type]
    except TypeError as exc:
        assert "'int'" in str(exc)
    else:
        raise AssertionError("expected TypeError for first=123")
    try:
        first_invalid_charset(["a"], first=b"x", rest=123)  # type: ignore[arg-type]
    except TypeError as exc:
        assert "'bytes'" in str(exc)
    else:
        raise AssertionError("expected TypeError for first=b'x'")
    # A lone surrogate in the sets beats one in the items (rest's position
    # 3 reported, not the item's position 1).
    with pytest.raises(UnicodeEncodeError) as excinfo:
        first_invalid_charset(["a\ud800"], rest="xyz\ud800")
    assert "position 3" in str(excinfo.value)
    # The offender spelling shares the walk, so the same precedence holds.
    with pytest.raises(TypeError, match="not an instance of 'str'"):
        first_invalid_offender(["ok", 123], rest=123)  # type: ignore[arg-type,list-item]


def test_offender_never_equals_minus_one_spell_the_check_is_none() -> None:
    """The -1 trap, pinned on both branches: a tuple never equals -1, so
    an ``!= -1`` invalidity guard ported from the int spelling fires on
    EVERY batch (clean included) and an ``== -1`` validity guard never
    does, both silently. The clean spelling is ``is None`` /
    ``is not None``."""
    clean = first_invalid_offender(["job_42"], first=IDENT_FIRST, rest=IDENT_REST)
    assert clean is None
    assert (clean != -1) is True  # the trap: True even though the batch is clean
    assert (clean == -1) is False  # ... and never True, even here
    offending = first_invalid_offender(["job_42", "9bad"], first=IDENT_FIRST, rest=IDENT_REST)
    assert offending == (1, 0, "9")
    assert (offending != -1) is True  # True here too: the guard cannot discriminate
    assert (offending == -1) is False
    assert (offending is None) is False
    assert (clean is None) is True


@given(_items_first_rest())
@settings(max_examples=500)
def test_offender_is_none_iff_the_int_spelling_answers_minus_one(
    items_first_rest: tuple[list[str], str | None, str],
) -> None:
    """The consistency invariant between the siblings, stated on its own:
    the tuple spelling is ``None`` exactly when the int spelling answers
    ``-1``; when not ``None`` its first field IS the int spelling's answer
    (the same scan, two projections); and the whole tuple equals the
    offender oracle's — the (position, char) detail verified against the
    reference over the same generated space the int differential covers
    (astral offenders, first-only codepoints at 0 vs 1, the ``first=""``/
    ``rest=""`` corners, empty items, empty lists, single items)."""
    items, first, rest = items_first_rest
    detail = first_invalid_offender(items, first=first, rest=rest)
    index = first_invalid_charset(items, first=first, rest=rest)
    assert (detail is None) == (index == -1)
    if detail is not None:
        assert detail[0] == index
    assert detail == reference_first_invalid_offender(items, first, rest)


@given(_arbitrary_items_first_rest())
@settings(max_examples=300)
def test_offender_matches_the_reference_over_arbitrary_unicode(
    items_first_rest: tuple[list[str], str | None, str],
) -> None:
    """The arbitrary-Unicode differential for the offender spelling: exact
    tuple equality with the oracle over any codepoint class (combining
    marks, scripts, widths the fixed alphabet cannot generate), plus the
    None-iff-minus-one invariant against the int spelling on the same
    draws."""
    items, first, rest = items_first_rest
    assert first_invalid_offender(items, first=first, rest=rest) == (
        reference_first_invalid_offender(items, first, rest)
    )
    assert (first_invalid_offender(items, first=first, rest=rest) is None) == (
        first_invalid_charset(items, first=first, rest=rest) == -1
    )


def test_every_small_items_list_and_set_spelling_matches_the_offender_reference() -> None:
    """The deterministic sweep for the offender spelling, the int
    spelling's own sweep exactly (85 x 16 pairs: every items list of size
    0-3 over the same pool crossed with the same ``first``/``rest``
    spellings), asserting full tuple equality with the oracle — no
    sampling at all."""
    pool = ["", "a", "b", "ab"]
    items_lists: list[list[str]] = [[]]
    for size in (1, 2, 3):
        items_lists.extend(list(combo) for combo in itertools.product(pool, repeat=size))
    for items in items_lists:
        for first in (None, "", "a", "ab"):
            for rest in ("", "a", "ab", "b"):
                assert first_invalid_offender(items, first=first, rest=rest) == (
                    reference_first_invalid_offender(items, first, rest)
                ), (items, first, rest)


class TestOffenderArgumentContract:
    """The offender spelling's argument boundary is the int spelling's
    exactly — the same shared walk — so every refusal is byte-identical,
    pinned here by raising both siblings on the same bad input and
    comparing the messages: the migration promise is that swapping the
    int call for the tuple call changes nothing about what raises or what
    it says. No new error classes exist to test; the taxonomy is the
    sibling's (TypeError / UnicodeEncodeError at the same boundaries, no
    negative-index handling anywhere in the family)."""

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
    def test_items_refusals_are_byte_identical_to_the_int_spelling(
        self, not_items: object
    ) -> None:
        with pytest.raises(TypeError) as int_raises:
            first_invalid_charset(not_items, rest="a")  # type: ignore[arg-type]
        with pytest.raises(TypeError) as offender_raises:
            first_invalid_offender(not_items, rest="a")  # type: ignore[arg-type]
        assert str(offender_raises.value) == str(int_raises.value)

    @pytest.mark.parametrize(
        "bad_entry",
        [b"x", bytearray(b"x"), 123, None],
        ids=["bytes-entry", "bytearray-entry", "int-entry", "none-entry"],
    )
    def test_non_str_entry_refusals_are_byte_identical(self, bad_entry: object) -> None:
        with pytest.raises(TypeError) as int_raises:
            first_invalid_charset(["ok", bad_entry], rest="ok")  # type: ignore[list-item]
        with pytest.raises(TypeError) as offender_raises:
            first_invalid_offender(["ok", bad_entry], rest="ok")  # type: ignore[list-item]
        assert str(offender_raises.value) == str(int_raises.value)

    def test_non_str_first_and_rest_refusals_are_byte_identical(self) -> None:
        for kwargs in (
            {"first": 1, "rest": "a"},
            {"first": b"x", "rest": "a"},
            {"rest": 1},
            {"rest": b"x"},
        ):
            with pytest.raises(TypeError) as int_raises:
                first_invalid_charset(["a"], **kwargs)  # type: ignore[arg-type]
            with pytest.raises(TypeError) as offender_raises:
                first_invalid_offender(["a"], **kwargs)  # type: ignore[arg-type]
            assert str(offender_raises.value) == str(int_raises.value)

    def test_lone_surrogate_refusals_are_byte_identical(self) -> None:
        """The standard str-in boundary, paid identically: the same
        ``UnicodeEncodeError`` message from both siblings for a lone
        surrogate in an item, in ``first``, or in ``rest``."""
        for items, kwargs in (
            (["abc\ud800"], {"rest": "abc"}),
            (["ok"], {"first": "a\ud800", "rest": "a"}),
            (["ok"], {"rest": "a\ud800"}),
        ):
            with pytest.raises(UnicodeEncodeError) as int_raises:
                first_invalid_charset(items, **kwargs)
            with pytest.raises(UnicodeEncodeError) as offender_raises:
                first_invalid_offender(items, **kwargs)
            assert str(offender_raises.value) == str(int_raises.value)

    def test_the_items_walk_validates_the_whole_list_before_the_scan(self) -> None:
        """The walk-first precedence is the sibling's: a bad entry anywhere
        raises at the boundary even when an earlier item already offends —
        the short-circuit is a scan property, never an argument-validation
        one, in the offender spelling too."""
        with pytest.raises(TypeError):
            first_invalid_offender(["bad item", 123], rest="abc")  # type: ignore[list-item]
        with pytest.raises(UnicodeEncodeError):
            first_invalid_offender(["bad item", "x\ud800"], rest="abc")

    def test_first_is_keyword_only_rest_is_required_items_may_be_keyword(self) -> None:
        """The signature shape is the sibling's: both set arguments
        keyword-only, ``rest`` required, ``first`` defaulting to ``None``
        (the uniform spelling), ``items`` passable by name; the default
        and an explicit ``None`` are the same call."""
        with pytest.raises(TypeError):
            first_invalid_offender(["a"], "a", "a")  # type: ignore[misc]
        with pytest.raises(TypeError):
            first_invalid_offender(["a"], first="a")  # type: ignore[call-arg]
        assert first_invalid_offender(items=["a"], rest="a") is None
        assert first_invalid_offender(["a"], rest="a") == first_invalid_offender(
            ["a"], first=None, rest="a"
        )

    def test_any_sequence_is_accepted_like_a_list(self) -> None:
        """``items`` is a sequence: the tuple spelling answers exactly what
        the list spelling answers (and what the int spelling answers for
        the same batch, the consistency invariant at the boundary)."""
        expected = (1, 0, "9")
        assert (
            first_invalid_offender(("job_42", "9bad"), first=IDENT_FIRST, rest=IDENT_REST)
            == expected
        )
        assert (
            first_invalid_offender(["job_42", "9bad"], first=IDENT_FIRST, rest=IDENT_REST)
            == expected
        )


def test_building_a_taskq_style_rejection_message_from_the_tuple() -> None:
    """The use-case pin: the tuple carries exactly what TaskQ's
    per-character rejection messages are built from (the losing character
    and its position, plus the item index to name the item), so the
    message is one f-string off the tuple — the shape the docs' "Building
    rejection messages" example shows, pinned here with all three
    branches: a character offender, the empty item (whose ``(i, 0, "")``
    tuple has no character to name, so the message says the item is
    empty), and the all-valid batch (``None``, no message at all)."""
    queue_first = string.ascii_letters + string.digits + "_"
    queue_rest = queue_first + ".-"

    def rejection(
        names: list[str], offender: tuple[int, int, str] | None
    ) -> str | None:
        if offender is None:
            return None
        item, position, char = offender
        if not char:  # the empty item: no offending character to name
            return f"queue name {names[item]!r} is invalid: it is empty"
        return (
            f"queue name {names[item]!r} is invalid: {char!r} at position {position} is not allowed"
        )

    names = ["jobs_eu", "foo:eu", "queue_us"]
    offender = first_invalid_offender(names, first=queue_first, rest=queue_rest)
    assert offender == (1, 3, ":")
    assert rejection(names, offender) == (
        "queue name 'foo:eu' is invalid: ':' at position 3 is not allowed"
    )
    empty_at = ["jobs_eu", "", "queue_us"]
    assert first_invalid_offender(empty_at, first=queue_first, rest=queue_rest) == (1, 0, "")
    assert rejection(empty_at, (1, 0, "")) == "queue name '' is invalid: it is empty"
    valid = ["jobs_eu", "queue_us"]
    assert first_invalid_offender(valid, first=queue_first, rest=queue_rest) is None
    assert rejection(valid, None) is None


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

    def test_a_100_000_item_batch_answers_correctly(self) -> None:
        """The GIL cells drive 100k/1M-item batches but assert only loop
        responsiveness, and the scale battery otherwise stops at 1000: this
        cell pins the ANSWER at the 100k size (all-valid and
        offender-at-the-end, both hand-derived), so a size-dependent
        regression — an index overflow, a walk/scan mismatch at scale —
        fails here instead of passing under a heartbeat-only check."""
        items = _ident_batch(100_000)
        assert first_invalid_charset(items, first=IDENT_FIRST, rest=IDENT_REST) == -1
        items[-1] = "bad name"
        assert first_invalid_charset(items, first=IDENT_FIRST, rest=IDENT_REST) == 99_999

    def test_a_100_000_item_batch_reports_the_offender_detail_at_scale(self) -> None:
        """The offender spelling's twin of the cell above: the tuple pinned
        at the 100k size (offender spliced at the very end), hand-derived —
        index 99_999, the space in "bad name" at codepoint position 3 — so
        a size-dependent detail regression fails here, not only under the
        int spelling's cell."""
        items = _ident_batch(100_000, poison_at=99_999)
        assert first_invalid_offender(items, first=IDENT_FIRST, rest=IDENT_REST) == (
            99_999,
            3,
            " ",
        )


class TestItemsWalkIsBounded:
    """The argument walk's DoS backstop: the walk pulls the whole sequence
    under the GIL (every handle borrowed before the detached scan runs),
    so an unbounded sequence (a ``Sequence`` whose ``__iter__`` never
    stops; the walk iterates, it never consults ``__len__``) would hold
    the GIL growing the handle vector until the process OOMs. The walk
    carries ``content_hash``'s protocol-walk cap: past the ceiling the
    call aborts with a generic ``ValueError`` instead of spinning. The
    endless probe runs in a disposable subprocess so a regression hangs
    the child (killed at the timeout, reported as a failure) instead of
    hanging the suite; both spellings are probed because the cap lives on
    the walk they share, and the refusal messages must stay
    byte-identical (the argument-boundary migration promise). Green,
    measured on this tree: the probe child refuses in ~0.15 s end to
    end (interpreter startup plus the capped walk), so the 60 s backstop
    sits ~400x above the green wall; the pre-fix hang dies at it."""

    @staticmethod
    def _endless_probe_child() -> str:
        return (
            "from collections.abc import Sequence\n"
            "import tors\n"
            "class Endless(Sequence):\n"
            "    def __len__(self):\n"
            "        return 0\n"
            "    def __getitem__(self, i):\n"
            "        raise IndexError(i)\n"
            "    def __iter__(self):\n"
            "        n = 0\n"
            "        while True:\n"
            "            yield f'job_{n}'\n"
            "            n += 1\n"
            "messages = []\n"
            "for call in (\n"
            "    lambda: tors.first_invalid_charset(Endless(), rest='a'),\n"
            "    lambda: tors.first_invalid_offender(Endless(), rest='a'),\n"
            "):\n"
            "    try:\n"
            "        call()\n"
            "    except ValueError as exc:\n"
            "        messages.append(str(exc))\n"
            "    else:\n"
            "        print('RETURNED')\n"
            "        raise SystemExit(3)\n"
            "print('REFUSED ' + ' || '.join(messages))\n"
        )

    def test_an_endless_sequence_aborts_with_value_error_instead_of_hanging(self) -> None:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", self._endless_probe_child()],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(
                "first_invalid_charset consumed an endless __iter__ until the "
                "60s timeout: the argument walk is unbounded (the GIL-held "
                "materialization needs content_hash's protocol-walk cap)"
            )
        assert proc.returncode == 0, f"the probe child failed: {proc.stderr[-800:]}"
        head, _, messages = proc.stdout.strip().partition(" ")
        assert head == "REFUSED", proc.stdout
        # The cap lives on the shared walk: both spellings refuse, with
        # the byte-identical message every other argument refusal carries
        # (the migration promise: swapping the int call for the tuple
        # call changes nothing about what raises or what it says).
        int_message, _, offender_message = messages.partition(" || ")
        assert int_message and int_message == offender_message, proc.stdout
        # Generic on purpose (canon's discipline): the bound is a DoS
        # backstop, not a contract: no cap value leaks in the message.
        assert "1000000" not in int_message
        assert "1_000_000" not in int_message


# --- The pinned common alphabets ------------------------------------------------
#
# The scope question — "is it worthwhile adding any other charset validators,
# for b62, b64, hex, UUID?" — answered in code: no wrapper functions ship (N
# wrappers delegating to the same core would add pure API surface and
# maintenance cost for zero performance gain; the generic engine stays the
# single engine), but the worthwhile part does: the alphabets genuinely
# tedious to spell, published once as module constants. A base62 alphabet
# transcribed wrong at a call site still validates *something*, silently —
# the constants kill that failure mode, and the byte-exact pins below are
# the net for a typo in either direction (module or pin).


class TestPinnedAlphabetContents:
    """The five constants, byte-exact: the single place a content typo can
    hide. A constant's characters are contract — the standard spellings, in
    their conventional orders (base62 digits-then-upper-then-lower, the
    RFC 4648 §5 url-safe order, hex digits then letters) — not detail."""

    def test_charset_b62(self) -> None:
        assert (
            tors.CHARSET_B62
            == "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )

    def test_charset_b64url(self) -> None:
        assert (
            tors.CHARSET_B64URL
            == "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        )

    def test_charset_hex_lower(self) -> None:
        assert tors.CHARSET_HEX_LOWER == "0123456789abcdef"

    def test_charset_hex_upper(self) -> None:
        assert tors.CHARSET_HEX_UPPER == "0123456789ABCDEF"

    def test_charset_hex_mixed(self) -> None:
        assert tors.CHARSET_HEX_MIXED == "0123456789abcdefABCDEF"


class TestPinnedAlphabetProperties:
    """The structural invariants a byte-exact pin alone does not state:
    each alphabet is a set however spelled (no duplicate codepoints), each
    has its nominal length, and the family is coherent as set algebra —
    both hex spellings are subsets of HEX_MIXED (which is exactly their
    union), and B64URL is exactly B62 plus the two url-safe punctuation
    codepoints, so the base62 ids and the JWT segments share one lineage."""

    @pytest.mark.parametrize(
        ("name", "length"),
        [
            ("CHARSET_B62", 62),
            ("CHARSET_B64URL", 64),
            ("CHARSET_HEX_LOWER", 16),
            ("CHARSET_HEX_UPPER", 16),
            ("CHARSET_HEX_MIXED", 22),
        ],
    )
    def test_each_alphabet_is_the_right_length_with_unique_codepoints(
        self, name: str, length: int
    ) -> None:
        constant = getattr(tors, name)
        assert len(constant) == length, f"{name}: expected {length} codepoints"
        assert len(set(constant)) == len(constant), f"{name}: duplicate codepoints"

    def test_b64url_is_exactly_b62_plus_the_urlsafe_punctuation(self) -> None:
        assert set(tors.CHARSET_B62) <= set(tors.CHARSET_B64URL)
        assert set(tors.CHARSET_B64URL) == set(tors.CHARSET_B62) | {"-", "_"}

    def test_hex_mixed_is_exactly_the_union_of_both_hex_spellings(self) -> None:
        assert set(tors.CHARSET_HEX_LOWER) <= set(tors.CHARSET_HEX_MIXED)
        assert set(tors.CHARSET_HEX_UPPER) <= set(tors.CHARSET_HEX_MIXED)
        assert set(tors.CHARSET_HEX_MIXED) == (
            set(tors.CHARSET_HEX_LOWER) | set(tors.CHARSET_HEX_UPPER)
        )


class TestPinnedAlphabetUse:
    """The constants in their intended seats. The uniform spelling
    (``first=None``, one set at every position) makes each one a single
    argument — ``first_invalid_charset(items, rest=tors.CHARSET_B62)`` —
    over the three shapes they exist for: base62 ids, unpadded base64url
    (JWT) segments, and hex digests in the fixed- and mixed-case
    spellings."""

    def test_base62_id_batch(self) -> None:
        # "7xK9mQ2pZv" and "0Zz8" are pure base62; "bad!" offends on "!"
        assert (
            first_invalid_charset(["7xK9mQ2pZv", "0Zz8", "bad!"], rest=tors.CHARSET_B62)
            == 2
        )
        assert first_invalid_charset(["7xK9mQ2pZv", "0Zz8"], rest=tors.CHARSET_B62) == -1

    def test_jwt_segments(self) -> None:
        # the three segments of a JWS — header, payload, signature — all
        # unpadded base64url; the signature spelling carries "_", exactly
        # the url-safe punctuation the alphabet exists to cover
        segments = [
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ",
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        ]
        assert first_invalid_charset(segments, rest=tors.CHARSET_B64URL) == -1
        # the padding mistake the unpadded alphabet is documented to catch:
        # "=" is positional structure (terminal only), so a padded segment
        # is an offender at its own index
        padded = [
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0=",
        ]
        assert first_invalid_charset(padded, rest=tors.CHARSET_B64URL) == 1

    def test_hex_digests_lower_and_mixed(self) -> None:
        # sha256("abc"), the canonical 64-char lowercase digest
        digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        assert first_invalid_charset([digest], rest=tors.CHARSET_HEX_LOWER) == -1
        # the uppercase spelling of the same digest offends under the
        # lower-only alphabet and passes under the case-insensitive one
        assert first_invalid_charset([digest.upper()], rest=tors.CHARSET_HEX_LOWER) == 0
        assert (
            first_invalid_charset([digest, digest.upper()], rest=tors.CHARSET_HEX_LOWER)
            == 1
        )
        assert (
            first_invalid_charset([digest, digest.upper()], rest=tors.CHARSET_HEX_MIXED)
            == -1
        )

    def test_empty_item_and_empty_batch_semantics_are_unchanged(self) -> None:
        # the constants are set data only: the empty-item rule (an empty
        # item is an offender wherever it sits, whatever the sets allow)
        # and the empty-batch answer (-1, vacuously valid) are the
        # engine's, identical under a published alphabet
        assert first_invalid_charset(["", "0Zz8"], rest=tors.CHARSET_B62) == 0
        assert first_invalid_charset([], rest=tors.CHARSET_B62) == -1


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

    def test_common_alphabets_example(self) -> None:
        """docs/api.md's Common alphabets subsection, pinned the same way:
        the literals the doc shows (a base62 id batch, then a JWT-segment
        pair whose second segment is mistakenly padded) are re-derived
        here against the built extension."""
        ids = ["7xK9mQ2pZv", "0Zz8", "bad!"]
        assert first_invalid_charset(ids, rest=tors.CHARSET_B62) == 2
        segments = [
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0=",
        ]
        assert first_invalid_charset(segments, rest=tors.CHARSET_B64URL) == 1

    def test_rejection_message_example(self) -> None:
        """docs/api.md's "Building rejection messages" subsection, pinned
        the same way: the literals the doc shows (the queue-name batch,
        the ``(1, 3, ":")`` tuple, the message the example builds from it)
        are re-derived here against the built extension."""
        queue_first = string.ascii_letters + string.digits + "_"
        queue_rest = queue_first + ".-"
        names = ["jobs_eu", "foo:eu", "queue_us"]
        assert (
            first_invalid_offender(names, first=queue_first, rest=queue_rest)
            == (1, 3, ":")
        )
        item, position, char = 1, 3, ":"
        assert (
            f"queue name {names[item]!r} is invalid: {char!r} at position {position} is not allowed"
            == "queue name 'foo:eu' is invalid: ':' at position 3 is not allowed"
        )
