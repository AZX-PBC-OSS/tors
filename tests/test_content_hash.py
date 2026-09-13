"""Contract gate for ``tors.content_hash``: the content-addressing hash over
Python objects, byte-identical with the stdlib expression it replaces.

``content_hash(obj: str | int | float | bool | None | list | tuple | dict) ->
str`` is the lowercase-hex SHA-256 of the object's canonical form, where the
canonical form is EXACTLY ``json.dumps(obj, sort_keys=True, separators=(",",":"))``
with the defaults ``ensure_ascii=True`` and ``allow_nan``. The oracle is
therefore total and always available, the same shape every other
differential in this suite pins against:

    hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

Every contract below is a differential pin against that oracle, plus
byte-level literal pins (the canonical bytes themselves, sha256'd here) so
the emitter's output shape is pinned independently of agreeing-with-itself
differential evidence.

The pinned contracts, each with its own class:

- **Escape table** (``TestEscapeBattery``): inside a string, exactly ``"``,
  ``\\``, and every codepoint outside printable ASCII (U+0020-U+007E) are
  escaped -- the five short escapes ``\\b`` ``\\t`` ``\\n`` ``\\f`` ``\\r``
  for U+0008/9/A/C/D, ``\\"``/``\\\\`` for quote/backslash, and ``\\u00XX``
  LOWERCASE-hex for the other 27 controls below U+0020 plus DEL (U+007F,
  which is outside printable ASCII); every non-ASCII codepoint becomes
  ``\\uXXXX`` lowercase, astral codepoints as surrogate-pair escapes
  (``chr(0x1F600)`` -> ``"\\ud83d\\ude00"``). The forward slash is never
  escaped. Every codepoint U+0000-U+007F and the BMP/astral boundary
  codepoints are pinned as literals and differentially, as values and as
  dict keys.
- **Dict keys** (``TestDictKeys``): json.dumps's coercion (``1`` -> ``"1"``,
  ``True`` -> ``"true"``, ``None`` -> ``"null"``, ``1.0`` -> ``"1.0"``,
  ``nan``/``inf`` -> ``"NaN"``/``"Infinity"``), sorting BEFORE
  stringification (all-int keys sort numerically then stringify: 2 < 10;
  the same digits as str keys sort lexicographically: "10" < "2"), and
  error parity for the shapes json.dumps rejects (mixed unsortable key
  types raise the sort's own TypeError, byte-identical, because tors
  delegates that sort to CPython's; non-coercible key types raise TypeError
  naming the type).
- **Floats** (``TestFloatZoo``): repr'd via Python's own float repr, exact
  by construction (tors never reimplements float formatting); non-finite
  values use json's ``allow_nan`` literals ``NaN``/``Infinity``/
  ``-Infinity``. The zoo: ``-0.0``, ``0.1``, ``1e16``, ``1e-5``, ``1e100``,
  ``5e-324``, ``sys.float_info.max``, ``nan``, ``inf``, ``-inf``, and the
  repr-boundary spellings around them, as values and as keys.
- **Ints** (``TestInts``): arbitrary precision -- the i64 fast path covers
  ``-(2**63)`` .. ``2**63-1`` and everything beyond falls back to Python's
  own int->str (so the interpreter's ``sys.set_int_max_str_digits`` limit
  raises on both sides identically).
- **tuple == list** (``TestTupleListEquivalence``): tuples serialize as
  lists, recursively.
- **Subclass containers** (``TestContainerSubclassIterationParity``):
  json.dumps never walks a container subclass's concrete storage -- a
  dict subclass goes through its OVERRIDABLE ``.items()`` (materialized
  into a snapshot, sorted with CPython's own timsort, each pair required
  to be a 2-tuple) and a list/tuple subclass through its ``__iter__``
  (materialized) -- so tors delegates to the same interpreter calls:
  hiding/faking/reordering/emptying subclasses hash identically on both
  sides, a subclass dict with empty concrete storage emits ``{}``
  without ever calling ``.items()``, and the runaway lane (hooks that
  yield ever-fresh subclasses, which no circular marker can catch) dies
  by ``RecursionError`` on both sides -- json by its C recursion guard,
  tors by a protocol-frame cap at ``sys.getrecursionlimit()`` -- while
  EXACT containers stay uncapped (the deep-nesting superset lane).
- **The surrogate divergence** (``TestSurrogateDivergence``): a str holding
  lone surrogates (value or key) raises ``UnicodeEncodeError`` from the
  standard str borrow -- the crate-wide boundary every str-in surface
  documents -- where json.dumps SUCCEEDS (it emits ``\\udXXX`` escapes).
  A documented, pinned divergence, not a bug.
- **Errors** (``TestErrorParity``): non-serializable values raise TypeError
  naming the type; circular references raise ValueError on both sides;
  huge ints hit the interpreter's digit limit on both sides.
- **Determinism** (``TestDeterminism``): any dict key order -> the same
  hash; equal-value list/tuple -> the same hash; repeated calls -> the
  same hash; 64 lowercase hex chars.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from enum import Enum
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reference import content_object
from tors import content_hash

_MIB = 1024 * 1024


def _canonical(obj: Any) -> str:
    """The canonical form, spelled exactly as the contract defines it."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _oracle(obj: Any) -> str:
    """The full stdlib expression ``tors.content_hash`` replaces."""
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _assert_parity(obj: Any) -> None:
    """Differential parity, raising shapes included: an object the oracle
    refuses (mixed unsortable key types; an int over the interpreter's
    digit limit) must be refused by tors with the SAME exception type --
    the differential covers the error domain, not just the success
    domain."""
    try:
        expected = _oracle(obj)
    except (TypeError, ValueError) as oracle_exc:
        with pytest.raises(type(oracle_exc)):
            content_hash(obj)
        return
    assert content_hash(obj) == expected


def _assert_bytes(obj: Any, canonical: bytes) -> None:
    """The literal pin: the canonical bytes themselves, sha256'd here."""
    assert content_hash(obj) == hashlib.sha256(canonical).hexdigest()


# The deterministic u64 LCG (the reference.py diff-corpus idiom) for the
# key-order shuffles: reproducible permutation sequences, no rng module.
_LCG_SEED = 0x9E3779B97F4A7C15
_LCG_MUL = 6364136223846793005
_LCG_INC = 1442695040888963407
_U64_MASK = (1 << 64) - 1


def _shuffled_order(n: int, salt: int) -> list[int]:
    """A deterministic permutation of ``range(n)``: Fisher-Yates driven by
    the LCG seeded per ``salt``, so each test's shuffles are reproducible."""
    order = list(range(n))
    state = (_LCG_SEED ^ (salt * 0x2545F4914F6CDD1D)) & _U64_MASK
    for i in range(n - 1, 0, -1):
        state = (state * _LCG_MUL + _LCG_INC) & _U64_MASK
        j = state % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


# The differential strategies (module level: the recursive strategy's
# extend lambda cannot see class scope). Keys: every coercible type, floats
# unbounded (nan/inf keys exercise the delegated sort; distinct NaN
# objects coexist as dict keys). Values: full recursion over lists,
# tuples (via .map(tuple), so nesting mixes both spellings), and dicts.
_KEYS = st.one_of(
    st.text(),
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.booleans(),
    st.none(),
)
_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
)
_VALUES = st.recursive(
    _SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=6),
        st.lists(children, max_size=4).map(tuple),
        st.dictionaries(_KEYS, children, max_size=6),
    ),
    max_leaves=25,
)


class TestScalarParity:
    """The leaf types over a literal battery: parity plus literal canonical
    bytes for each, so the emitter's scalar spellings are pinned without
    relying on the oracle alone."""

    @pytest.mark.parametrize(
        ("obj", "canonical"),
        [
            (None, b"null"),
            (True, b"true"),
            (False, b"false"),
            (0, b"0"),
            (-1, b"-1"),
            (1, b"1"),
            (9223372036854775807, b"9223372036854775807"),
            (-9223372036854775808, b"-9223372036854775808"),
            (0.5, b"0.5"),
            (-0.0, b"-0.0"),
            (1e16, b"1e+16"),
            ("", b'""'),
            ("a", b'"a"'),
            (" a ", b'" a "'),
            ("~", b'"~"'),
            ([], b"[]"),
            ((), b"[]"),
            ({}, b"{}"),
            ([None], b"[null]"),
            ([1, "a"], b'[1,"a"]'),
            ([1.5, 2], b"[1.5,2]"),
            ({"a": 1}, b'{"a":1}'),
            ({"b": 2, "a": 1}, b'{"a":1,"b":2}'),
            ([[], [[]]], b"[[],[[]]]"),
            (({"k": (1, [2])},), b'[{"k":[1,[2]]}]'),
            ({True: "x"}, b'{"true":"x"}'),
            ({None: 1}, b'{"null":1}'),
            ({1: "x"}, b'{"1":"x"}'),
            ({1.0: "x"}, b'{"1.0":"x"}'),
            ({float("nan"): 1}, b'{"NaN":1}'),
            ({float("inf"): 1}, b'{"Infinity":1}'),
        ],
        ids=[
            "none",
            "true",
            "false",
            "zero",
            "neg-one",
            "one",
            "i64-max",
            "i64-min",
            "half",
            "neg-zero",
            "1e16",
            "empty-str",
            "a",
            "padded-str",
            "tilde",
            "empty-list",
            "empty-tuple",
            "empty-dict",
            "null-list",
            "int-str-list",
            "float-int-list",
            "single-pair",
            "sorted-pair",
            "nested-empties",
            "tuple-in-dict-in-list",
            "bool-key",
            "none-key",
            "int-key",
            "float-key",
            "nan-key",
            "inf-key",
        ],
    )
    def test_scalar_battery(self, obj: Any, canonical: bytes) -> None:
        _assert_parity(obj)
        _assert_bytes(obj, canonical)

    def test_the_five_short_escapes_and_no_others(self) -> None:
        """Exactly U+0008/0009/000A/000C/000D take the short escapes among
        the controls; the other 27 codepoints below U+0020 take ``\\u00XX``
        lowercase, DEL (U+007F) is escaped as ``\\u007f`` (it is outside
        printable ASCII: the raw range is exactly U+0020-U+007E), and the
        raw set is every printable ASCII codepoint except ``"`` and ``\\``."""
        _assert_bytes("\b", b'"\\b"')
        _assert_bytes("\t", b'"\\t"')
        _assert_bytes("\n", b'"\\n"')
        _assert_bytes("\f", b'"\\f"')
        _assert_bytes("\r", b'"\\r"')
        _assert_bytes("\x00", b'"\\u0000"')
        _assert_bytes("\x0b", b'"\\u000b"')
        _assert_bytes("\x1f", b'"\\u001f"')
        _assert_bytes("\x7f", b'"\\u007f"')
        _assert_bytes('"', b'"\\""')
        _assert_bytes("\\", b'"\\\\"')
        short = {0x08: "b", 0x09: "t", 0x0A: "n", 0x0C: "f", 0x0D: "r"}
        for cp in range(0x20):
            expected = f'"\\{short[cp]}"' if cp in short else f'"\\u{cp:04x}"'
            assert _canonical(chr(cp)) == expected
        raw = [cp for cp in range(0x20, 0x7F) if cp not in (0x22, 0x5C)]
        assert len(raw) == 93
        for cp in raw:
            _assert_bytes(chr(cp), b'"' + bytes([cp]) + b'"')

    def test_forward_slash_is_never_escaped(self) -> None:
        _assert_bytes("a/b", b'"a/b"')
        _assert_parity("\\/\\")


class TestEscapeBattery:
    """Every codepoint U+0000-U+007F plus the non-ASCII boundary codepoints,
    as values and as dict keys: parity against the oracle (which pins the
    exact escape spelling) plus structural assertions on the canonical form
    itself (ASCII-only output, lowercase hex, astral as surrogate pairs)."""

    # Every BMP range boundary, the astral boundary, the extremes, and
    # samples from each plane (Hangul right after the surrogate block's end,
    # private use, variation selectors, the last codepoints).
    _BOUNDARY_CPS = [
        0x00,
        0x07,
        0x08,
        0x0B,
        0x1F,
        0x20,
        0x21,
        0x22,
        0x23,
        0x5B,
        0x5C,
        0x5D,
        0x7E,
        0x7F,
        0x80,
        0x7FF,
        0x800,
        0x9FF,
        0xAC00,
        0xD7FF,  # last codepoint before the surrogate block
        0xE000,  # first private-use codepoint after it
        0xF900,
        0xFFFE,
        0xFFFF,
        0x10000,  # first astral
        0x10001,
        0x1F600,
        0x20000,
        0x2FA1D,
        0x30000,
        0xE0001,
        0xE0100,
        0xF0000,
        0xFFFFE,
        0x100000,
        0x10FFFE,
        0x10FFFF,  # the last codepoint
    ]

    @pytest.mark.parametrize("cp", range(0x80), ids=[f"U+{cp:04X}" for cp in range(0x80)])
    def test_ascii_codepoint_as_value(self, cp: int) -> None:
        _assert_parity(chr(cp))

    @pytest.mark.parametrize("cp", _BOUNDARY_CPS, ids=[f"U+{cp:05X}" for cp in _BOUNDARY_CPS])
    def test_boundary_codepoint_as_value(self, cp: int) -> None:
        _assert_parity(chr(cp))

    @pytest.mark.parametrize("cp", _BOUNDARY_CPS, ids=[f"U+{cp:05X}" for cp in _BOUNDARY_CPS])
    def test_boundary_codepoint_as_key(self, cp: int) -> None:
        _assert_parity({chr(cp): 1})

    def test_astral_codepoints_emit_lowercase_surrogate_pair_escapes(self) -> None:
        _assert_bytes(chr(0x1F600), b'"\\ud83d\\ude00"')
        _assert_bytes(chr(0x10FFFF), b'"\\udbff\\udfff"')
        _assert_bytes(chr(0x10000), b'"\\ud800\\udc00"')

    def test_non_ascii_escapes_are_lowercase_hex(self) -> None:
        for cp in (0xE9, 0x80, 0x7FF, 0x800, 0xFFFD, 0xFEFF, 0x2028, 0x2029, 0x85, 0xA0, 0xAD):
            assert _canonical(chr(cp)) == f'"\\u{cp:04x}"'

    def test_backslash_and_quote_runs(self) -> None:
        for s in [
            "\\\\",
            '""""',
            "\\n\\t\\r",
            'say "hi" \\ ok',
            "\\",
            '"',
            '"\\"',
            "\\u0041",
            "a\\/b",
        ]:
            _assert_parity(s)
        _assert_bytes('"\\"', b'"\\"\\\\\\""')
        _assert_bytes("\\\\", b'"\\\\\\\\"')

    def test_control_run_mixed_with_astral_and_bmp(self) -> None:
        s = "a\x00\b\x1f\x7fé😀\t"
        _assert_parity(s)
        _assert_parity({s: s})
        assert _canonical(s) == '"a\\u0000\\b\\u001f\\u007f\\u00e9\\ud83d\\ude00\\t"'

    def test_escaped_output_is_always_ascii(self) -> None:
        """The whole canonical form is pure ASCII for any input (the
        ensure_ascii contract): the emitted bytes can never exceed 0x7F."""
        obj = {"clé": ["héllo", "wörld", chr(0x10FFFF)], "😀": chr(0xE01F0)}
        assert _canonical(obj).isascii()
        _assert_parity(obj)


class TestFloatZoo:
    """Finite floats use Python's own repr spelling (tors calls it; it never
    reimplements float formatting); non-finites use json's allow_nan
    literals. Every spelling below is what repr() actually returns, pinned
    as literal canonical bytes."""

    @pytest.mark.parametrize(
        ("f", "canonical"),
        [
            (-0.0, b"-0.0"),
            (0.0, b"0.0"),
            (0.1, b"0.1"),
            (0.5, b"0.5"),
            (1.5, b"1.5"),
            (2.0, b"2.0"),
            (100.0, b"100.0"),
            (1e15, b"1000000000000000.0"),
            (1e16, b"1e+16"),
            (1e-4, b"0.0001"),
            (1e-5, b"1e-05"),
            (1e100, b"1e+100"),
            (5e-324, b"5e-324"),  # the smallest subnormal
            (sys.float_info.max, b"1.7976931348623157e+308"),
            (-sys.float_info.max, b"-1.7976931348623157e+308"),
            (123456789012345678.0, b"1.2345678901234568e+17"),
            (float("nan"), b"NaN"),
            (float("inf"), b"Infinity"),
            (float("-inf"), b"-Infinity"),
        ],
        ids=[
            "neg-zero",
            "zero",
            "tenth",
            "half",
            "one-five",
            "two",
            "hundred",
            "1e15",
            "1e16",
            "1e-4",
            "1e-5",
            "1e100",
            "min-subnormal",
            "float-max",
            "float-neg-max",
            "rounding-17",
            "nan",
            "inf",
            "neg-inf",
        ],
    )
    def test_float_spelling_as_value(self, f: float, canonical: bytes) -> None:
        _assert_parity(f)
        _assert_bytes(f, canonical)

    @pytest.mark.parametrize(
        "f",
        [
            -0.0,
            0.1,
            1e16,
            1e-5,
            1e100,
            5e-324,
            sys.float_info.max,
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
        ids=[
            "neg-zero",
            "tenth",
            "1e16",
            "1e-5",
            "1e100",
            "min-subnormal",
            "float-max",
            "nan",
            "inf",
            "neg-inf",
        ],
    )
    def test_float_spelling_as_key(self, f: float) -> None:
        _assert_parity({f: "v"})

    def test_float_key_sort_is_numeric_across_the_zoo(self) -> None:
        zoo = [5e-324, -0.0, 1e-5, 0.1, 1.5, 1e16, 1e100, sys.float_info.max, float("inf")]
        _assert_parity({f: i for i, f in enumerate(zoo)})

    def test_the_whole_zoo_in_a_list_keeps_order(self) -> None:
        zoo = [
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.0,
            0.1,
            1e16,
            1e-5,
            5e-324,
            sys.float_info.max,
        ]
        _assert_parity(zoo)
        _assert_parity([zoo, zoo])  # shared sibling lists, no false circular


class TestInts:
    """Arbitrary precision: the i64 fast path at its exact boundaries, the
    repr fallback beyond them (both directions), and the interpreter's
    int->str digit limit raising identically on both sides."""

    @pytest.mark.parametrize(
        "i",
        [
            0,
            1,
            -1,
            2**31 - 1,
            -(2**31),
            2**53,
            2**62,
            2**63 - 1,
            -(2**63),
            2**63,
            -(2**63) - 1,
            2**64,
            2**100,
            -(2**100),
            10**40,
            -(10**40),
        ],
        ids=[
            "zero",
            "one",
            "neg-one",
            "i32-max",
            "i32-min",
            "2^53",
            "2^62",
            "i64-max",
            "i64-min",
            "i64-max-plus",
            "i64-min-minus",
            "2^64",
            "2^100",
            "neg-2^100",
            "10^40",
            "neg-10^40",
        ],
    )
    def test_int_boundaries_as_values(self, i: int) -> None:
        _assert_parity(i)
        _assert_bytes(i, str(i).encode("ascii"))

    def test_magnitude_ladder(self) -> None:
        ladder = [2**k for k in (0, 31, 32, 62, 63, 64, 65, 100, 431)]
        _assert_parity(ladder)
        _assert_parity([-x for x in ladder])

    def test_big_ints_as_keys_sort_numerically(self) -> None:
        keys = [2**100, -(2**100), 0, 2**63, -(2**63), 17, 2**40]
        _assert_parity({k: i for i, k in enumerate(keys)})

    def test_big_int_mixed_with_float_keys_sorts_exactly(self) -> None:
        """2**63 vs 9.3e18 vs 2**63+1: int/float key comparison is exact in
        Python (neither side rounds), pinned here at the magnitudes where a
        naive as-f64 comparison would misorder."""
        keys = {2**63: "a", 9.3e18: "b", 2**63 + 1: "c", 1e19: "d", -(2**63): "e"}
        _assert_parity(keys)

    def test_huge_int_hits_the_interpreter_digit_limit_identically(self) -> None:
        """10**5001 exceeds the default 4300-digit int->str limit on 3.11+,
        so BOTH sides raise ValueError; on 3.10 (no limit) both sides hash.
        Whichever interpreter runs this gate, the two sides agree."""
        huge = 10**5001
        try:
            expected = _oracle(huge)
        except ValueError:
            with pytest.raises(ValueError):
                content_hash(huge)
        else:
            assert content_hash(huge) == expected


class TestDictKeys:
    """json.dumps's key coercion, sort-before-stringification, and the
    rejected shapes (mixed unsortable types; non-coercible key types)."""

    def test_key_coercion_pins(self) -> None:
        _assert_bytes({1: "x"}, b'{"1":"x"}')
        _assert_bytes({True: "x"}, b'{"true":"x"}')
        _assert_bytes({False: "x"}, b'{"false":"x"}')
        _assert_bytes({None: "x"}, b'{"null":"x"}')
        _assert_bytes({1.0: "x"}, b'{"1.0":"x"}')
        _assert_bytes({2.5: "x"}, b'{"2.5":"x"}')
        _assert_bytes({-1: "x"}, b'{"-1":"x"}')
        _assert_bytes({"": "x"}, b'{"":"x"}')

    def test_sort_before_stringification_int_keys_sort_numerically(self) -> None:
        """{10, 2, 33} -> "2", "10", "33": numeric order THEN stringify. If
        the keys were stringified first, "10" would sort before "2"."""
        _assert_bytes({10: "a", 2: "b", 33: "c", -5: "d"}, b'{"-5":"d","2":"b","10":"a","33":"c"}')

    def test_str_keys_with_digit_spelling_sort_lexicographically(self) -> None:
        """The contrast that proves sorting happens on the original keys:
        the same digits as str keys sort as strings ("10" < "2")."""
        _assert_bytes({"10": "a", "2": "b"}, b'{"10":"a","2":"b"}')

    def test_bool_and_int_keys_sort_together_numerically(self) -> None:
        _assert_bytes({True: "t", 10: "a", 2: "b"}, b'{"true":"t","2":"b","10":"a"}')
        _assert_bytes({False: "f", True: "t"}, b'{"false":"f","true":"t"}')

    def test_int_and_float_keys_sort_together_numerically(self) -> None:
        _assert_bytes({1: "a", 1.5: "b", 0.5: "c"}, b'{"0.5":"c","1":"a","1.5":"b"}')

    def test_str_keys_sort_by_codepoint(self) -> None:
        obj = {"b": 1, "a": 2, "é": 3, "Z": 4, "~": 5, "😀": 6, "0": 7}
        _assert_parity(obj)
        # ASCII first (by codepoint), then U+00E9, then the astral emoji.
        assert list(json.loads(_canonical(obj))) == ["0", "Z", "a", "b", "~", "é", "😀"]

    def test_non_finite_key_coercion(self) -> None:
        _assert_bytes({float("nan"): 1}, b'{"NaN":1}')
        _assert_bytes({float("inf"): 1, float("-inf"): 0}, b'{"-Infinity":0,"Infinity":1}')

    def test_nan_key_dict_corners_match_the_oracle_exactly(self) -> None:
        """NaN keys make the sort comparator inconsistent (nan < x is False
        both ways), so the output order is CPython's timsort behavior, not a
        mathematical property. tors pins parity by delegating that sort to
        CPython's own ``list.sort`` (the same comparisons, the same
        algorithm), so it matches whatever the running interpreter does --
        including the two-distinct-NaN-keys dict (nan != nan lets both
        coexist as keys) and NaN mixed with sortable keys."""
        nan_a, nan_b = float("nan"), float("nan")
        _assert_parity({nan_a: "a"})
        _assert_parity({nan_a: "a", nan_b: "b"})
        _assert_parity({nan_b: "b", nan_a: "a"})
        _assert_parity({nan_a: "a", 1: "b"})
        _assert_parity({1: "b", nan_a: "a"})
        _assert_parity({nan_a: "a", 1: "b", 2.5: "c"})
        _assert_parity({2.5: "c", nan_a: "a", 1: "b"})

    def test_key_order_invariance(self) -> None:
        """The same pairs in any insertion order hash identically (the
        canonical form is sorted, so the hash cannot see insertion order)."""
        base = {f"k{i:03d}": i for i in range(50)}
        reference_hash = content_hash(base)
        for salt in range(1, 6):
            order = _shuffled_order(50, salt)
            shuffled = {f"k{i:03d}": i for i in order}
            assert content_hash(shuffled) == reference_hash

    def test_wide_str_key_dict(self) -> None:
        obj = {f"key-{i:05d}": i for i in _shuffled_order(10_000, salt=7)}
        _assert_parity(obj)

    def test_wide_int_key_dict_sorts_numerically(self) -> None:
        obj = {i: str(i) for i in _shuffled_order(10_000, salt=8)}
        _assert_parity(obj)
        assert list(json.loads(_canonical(obj))) == [str(i) for i in range(10_000)]

    def test_wide_bool_int_float_key_dict(self) -> None:
        keys: list[Any] = [False, True, 2, 10, 10.5, -3.25, 7]
        obj = {k: str(k) for k in keys}
        _assert_parity(obj)

    def test_wide_bool_int_key_set_sorts_numerically_through_the_fast_path(self) -> None:
        """The all-exact int/bool fast path's BOOL lane, widened to life:
        the original whole-set gate (``is_exact_instance_of::<PyInt>``)
        is FALSE for ``True``/``False`` -- bool's type is ``bool``, not
        ``int`` -- so every bool-bearing key set silently took the
        delegated timsort and the fast path the docstring described
        never ran. Bool IS its 0/1 int value (and bool cannot be
        subclassed; ``True``/``1`` and ``False``/``0`` cannot coexist as
        dict keys, so no tie is possible), so the numeric i64 sort is
        exact for the mixed set. This pin hashed identically through the
        delegated path before the widening and through the fast path
        after -- which is exactly why it needs pinning both sides of the
        gate."""
        obj = {
            False: "f",
            True: "t",
            -5: "d",
            2: "b",
            10: "a",
            9223372036854775807: "m",
            -9223372036854775808: "n",
        }
        _assert_parity(obj)
        _assert_bytes(
            obj,
            b'{"-9223372036854775808":"n","-5":"d","false":"f",'
            b'"true":"t","2":"b","10":"a","9223372036854775807":"m"}',
        )

    def test_single_key_of_every_coercible_type_needs_no_sort(self) -> None:
        for key in [
            "s",
            "",
            1,
            -1,
            2**100,
            True,
            False,
            None,
            1.0,
            2.5,
            float("nan"),
            float("inf"),
        ]:
            _assert_parity({key: "v"})


class TestTupleListEquivalence:
    """Tuples serialize as lists: an equal-value list and tuple hash the
    same, recursively, at every nesting depth."""

    @pytest.mark.parametrize(
        ("seq", "tup"),
        [
            ([1, 2], (1, 2)),
            ([], ()),
            ([1, "a", None, 2.5], (1, "a", None, 2.5)),
            ([[1], [2, 3]], ((1,), (2, 3))),
            ([{"a": (1,)}], ({"a": [1]},)),
            ([[[[1]]]], ((((1,),),),)),
        ],
        ids=["flat", "empty", "mixed", "nested", "dict-wrapped", "deep"],
    )
    def test_equal_value_list_and_tuple_hash_identically(self, seq: list, tup: tuple) -> None:
        assert content_hash(seq) == content_hash(tup)
        _assert_parity(seq)
        _assert_parity(tup)

    def test_tuple_mixed_into_lists_and_dicts(self) -> None:
        _assert_parity([1, (2, [3, (4,)])])
        _assert_parity({"t": (1, (2,)), "l": [1, [2]]})


class TestDeterminism:
    def test_repeated_calls_are_identical(self) -> None:
        obj = {"a": [1, 2.5, "x", None, True], "b": (1,)}
        assert content_hash(obj) == content_hash(obj) == content_hash(obj)

    def test_output_is_64_lowercase_hex_chars(self) -> None:
        h = content_hash({"any": ["object"]})
        assert len(h) == 64
        assert h == h.lower()
        int(h, 16)  # valid hex

    def test_distinct_objects_hash_distinctly(self) -> None:
        assert content_hash({"a": 1}) != content_hash({"a": 2})
        assert content_hash([1, 2]) != content_hash([2, 1])  # order-sensitive
        assert content_hash({"a": 1, "b": 2}) != content_hash({"a": 2, "b": 1})


class TestHypothesisDifferential:
    """The total differential: hypothesis-generated objects over every
    accepted type, nesting, and key-type zoo, each checked against the
    oracle. The strategy never generates lone surrogates (``st.text`` cannot
    by construction), so every generated object is inside the shared domain
    of both implementations."""

    @given(_VALUES)
    @settings(max_examples=300)
    def test_arbitrary_objects_match_the_oracle(self, obj: Any) -> None:
        _assert_parity(obj)

    @given(st.dictionaries(_KEYS, _SCALARS, max_size=10))
    @settings(max_examples=300)
    def test_arbitrary_key_zoos_match_the_oracle(self, obj: dict) -> None:
        _assert_parity(obj)

    @given(st.lists(_SCALARS, min_size=1, max_size=6))
    @settings(max_examples=150)
    def test_equal_value_list_and_tuple_always_hash_identically(self, items: list) -> None:
        assert content_hash(items) == content_hash(tuple(items))

    def test_deep_nesting_ladder(self) -> None:
        for depth in (10, 100, 300):
            obj: Any = None
            for _ in range(depth):
                obj = [obj]
            _assert_parity(obj)
            obj = None
            for _ in range(depth):
                obj = {"k": obj}
            _assert_parity(obj)
            obj = None
            for _ in range(depth):
                obj = ({"k": [obj]},)
            _assert_parity(obj)

    def test_beyond_jsons_recursion_boundary_the_walk_stays_iterative(self) -> None:
        """The deep-nesting divergence lane, pinned Python-side: the walk is
        ITERATIVE (an explicit frame stack, ``src/py/canon.rs``), so tree depth
        costs heap, never the call stack, and ``content_hash`` accepts nesting
        far deeper than ``json.dumps``, which ``RecursionError``s at an
        interpreter-version- and stack-size-dependent depth (measured on the
        dev box's 3.14: between 104590 and 104591 list levels -- the C
        recursion budget is stack-proportional there, NOT the ~1000 of the
        Python recursion limit; older interpreters fail far shallower). This
        pin does not touch the oracle at all (the boundary moves per
        interpreter and per box): it pins tors's own side of the documented
        lane -- the same 100k/50k depths the crate-side emitter and iterative
        ``Drop`` pins hold in ``src/canon_impl.rs`` -- through the REAL pyo3
        walk, which no other test covers past depth 300. A regression to a
        recursive walk would pass the 300-step ladder and only fail here.
        The literal bytes follow the crate-side formula: a ``d``-deep list of
        ``None`` is exactly ``b"[" * d + b"null" + b"]" * d``."""
        for depth in (100_000, 50_000):
            obj: Any = None
            for _ in range(depth):
                obj = [obj]
            _assert_bytes(obj, b"[" * depth + b"null" + b"]" * depth)
            obj = None
            for _ in range(depth):
                obj = {"k": obj}
            _assert_bytes(obj, b'{"k":' * depth + b"null" + b"}" * depth)

    def test_1mib_object_parity(self) -> None:
        """The ``reference.content_object`` corpus (the same tree the GIL
        and wall cells measure) at 1 MiB of canonical form."""
        _assert_parity(content_object(1 * _MIB))


class TestErrorParity:
    """Both sides raise for exactly the same objects, with the same
    exception types; tors's own TypeError wording names the offending type
    (values and non-coercible keys), the mixed-key sort error is
    byte-identical with json.dumps's (delegated to CPython's own sort), and
    circular references raise ValueError on both sides."""

    class Widget:
        pass

    class Color(Enum):
        RED = 1

    @pytest.mark.parametrize(
        "value",
        [
            set(),
            frozenset(),
            b"bytes",
            bytearray(),
            Widget(),
            object(),
            type,
            complex(1, 2),
            range(3),
            memoryview(b"x"),
            Color.RED,
            {1: [1]}.keys(),
            iter([]),
        ],
        ids=[
            "set",
            "frozenset",
            "bytes",
            "bytearray",
            "custom-class",
            "object",
            "type-object",
            "complex",
            "range",
            "memoryview",
            "plain-enum",
            "dict-keys-view",
            "list-iterator",
        ],
    )
    def test_non_serializable_values_raise_type_error_on_both_sides(self, value: Any) -> None:
        with pytest.raises(TypeError, match=re.escape(type(value).__name__)):
            content_hash({"v": value})
        with pytest.raises(TypeError):
            _oracle({"v": value})

    def test_type_error_names_the_offending_type_top_level_and_nested(self) -> None:
        with pytest.raises(TypeError, match="set"):
            content_hash(set())
        with pytest.raises(TypeError, match="Widget"):
            content_hash([1, [2, {"deep": TestErrorParity.Widget()}]])
        with pytest.raises(TypeError, match="bytes"):
            content_hash({"k": b"x"})

    @pytest.mark.parametrize(
        "key",
        [(1, 2), b"b", 1.5j, Widget()],
        ids=["tuple", "bytes", "complex", "custom-class"],
    )
    def test_non_coercible_keys_raise_type_error_on_both_sides(self, key: Any) -> None:
        # (bytearray and other unhashables never reach either
        # implementation: the dict itself refuses them at construction.)
        with pytest.raises(TypeError, match="keys must be str, int, float, bool, or None"):
            content_hash({key: 1})
        with pytest.raises(TypeError):
            _oracle({key: 1})

    def test_decimal_key_is_rejected_like_json_rejects_it(self) -> None:
        from decimal import Decimal

        with pytest.raises(TypeError, match="Decimal"):
            content_hash({Decimal("1.5"): 1})
        with pytest.raises(TypeError):
            _oracle({Decimal("1.5"): 1})

    @pytest.mark.parametrize(
        "obj",
        [
            {1: "a", "b": 2},
            {"b": 2, 1: "a"},
            {None: 1, 2: 3},
            {"a": 1, (1, 2): 2},
            {"a": set(), (1, 2): 1},  # sort error fires before the value's
        ],
        ids=["int-str", "str-int", "none-int", "str-tuple", "doubly-bad"],
    )
    def test_mixed_unsortable_keys_raise_the_sorts_own_type_error(self, obj: dict) -> None:
        """The delegated sort reproduces json.dumps's comparison error
        byte-for-byte (same timsort, same operands), and it fires BEFORE
        any value is walked, exactly as in json.dumps's own encode order."""
        with pytest.raises(TypeError) as tors_exc:
            content_hash(obj)
        with pytest.raises(TypeError) as oracle_exc:
            _oracle(obj)
        assert str(tors_exc.value) == str(oracle_exc.value)

    def test_key_type_error_fires_before_value_errors_for_single_key_dicts(self) -> None:
        with pytest.raises(TypeError, match="keys must be"):
            content_hash({(1, 2): set()})
        with pytest.raises(TypeError):
            _oracle({(1, 2): set()})

    def test_first_bad_value_in_emission_order_wins(self) -> None:
        with pytest.raises(TypeError, match="not set"):
            content_hash({"a": set(), "b": frozenset()})
        with pytest.raises(TypeError, match="not frozenset"):
            content_hash({"a": 1, "b": frozenset(), "c": set()})

    def test_circular_references_raise_value_error_on_both_sides(self) -> None:
        a: list[Any] = []
        a.append(a)
        with pytest.raises(ValueError, match="[Cc]ircular reference detected"):
            content_hash(a)
        with pytest.raises(ValueError):
            _oracle(a)

        d: dict[str, Any] = {}
        d["self"] = d
        with pytest.raises(ValueError, match="[Cc]ircular reference detected"):
            content_hash(d)
        with pytest.raises(ValueError):
            _oracle(d)

    def test_shared_non_circular_siblings_hash_fine(self) -> None:
        """The same object appearing twice as a sibling is NOT circular
        (json's markers are enter/exit, and so is tors's walk)."""
        shared = [1, {"k": "v"}]
        _assert_parity([shared, shared, shared])
        _assert_parity({"a": shared, "b": shared})


class TestSurrogateDivergence:
    """The documented divergence lane: json.dumps accepts lone surrogates
    (it emits ``\\udXXX`` escapes for them), while every str-in surface in
    this crate refuses them at the borrow (``UnicodeEncodeError``,
    "surrogates not allowed"). Pinned loudly so nobody mistakes it for a
    bug: if these ever pass, the crate-wide str-borrow contract has
    regressed, not content_hash."""

    @pytest.mark.parametrize(
        "s",
        ["\ud800", "\udfff", "a\ud800b", "\ud800abc\udfff", "\ud83d\ude00"],
        ids=["d800", "dfff", "mid", "both-ends", "surrogate-pair-spelling"],
    )
    def test_lone_surrogate_values_raise_where_json_succeeds(self, s: str) -> None:
        assert _canonical(s)  # json.dumps SUCCEEDS: the divergence is real
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            content_hash(s)

    def test_lone_surrogate_keys_raise_where_json_succeeds(self) -> None:
        assert _canonical({"\ud800": 1})
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            content_hash({"\ud800": 1})

    def test_lone_surrogates_nested_deep_raise(self) -> None:
        obj = {"ok": [1, "fine"], "bad": ["\ud800"]}
        assert _canonical(obj)
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            content_hash(obj)

    def test_clean_astral_text_is_fine_on_both_sides(self) -> None:
        """The neighboring non-divergent lane: real astral text (valid
        surrogate pairs in UTF-16 terms) hashes with full parity."""
        _assert_parity("emoji \U0001F600 and \U0001D538 math")
        _assert_parity({"😀": ["🎉", chr(0x10FFFF)]})


class TestSubclassComparisonParity:
    """The red-team corners around subclass keys, pinned: json.dumps's sort
    honors OVERRIDDEN rich comparison on str/int-subclass dict keys (a
    flipped ``__lt__`` reorders the output; an ``__eq__``-lying subclass
    falls the tuple tiebreak to the VALUES), so tors's fast sorts are
    gated on EXACT instances and every subclass-keyed dict delegates to
    CPython's own timsort over the dict's own (key, value) items. The
    flipped-``__lt__`` int-subclass case was a real red-team find: a
    numeric fast path silently sorted ``I(1)`` before ``I(3)`` where
    json.dumps honored the override (I(3) first); the gate and the
    (key, value) delegation close the whole class."""

    class FlippedInt(int):
        def __repr__(self) -> str:
            return "OVERRIDDEN"

        def __lt__(self, other: object) -> bool:
            return int(self) > int(other)

        def __le__(self, other: object) -> bool:
            return int(self) >= int(other)

    class FlippedStr(str):
        def __lt__(self, other: str) -> bool:
            return str(self) > str(other)

    class LyingEqStr(str):
        # Always-equal with a per-object hash: two of these legally
        # coexist as dict keys, and the items-sort tiebreak lands on the
        # VALUES, exactly json.dumps's behavior.

        def __eq__(self, other: object) -> bool:
            return True

        def __ne__(self, other: object) -> bool:
            return False

        def __hash__(self) -> int:
            return hash(str(self)) ^ id(self)

    class ReprStr(str):
        def __repr__(self) -> str:
            return "OVERRIDDEN"

    class ReprFloat(float):
        def __repr__(self) -> str:
            return "OVERRIDDEN"

    def test_flipped_lt_int_subclass_keys_match_the_oracle(self) -> None:
        cls = TestSubclassComparisonParity
        i1, i2, i3 = cls.FlippedInt(1), cls.FlippedInt(2), cls.FlippedInt(3)
        _assert_parity({i3: "a", i1: "b"})
        _assert_parity({i3: "a", i1: "b", i2: "c"})
        _assert_parity({i1: "a", i3: "b"})

    def test_flipped_lt_str_subclass_keys_match_the_oracle(self) -> None:
        cls = TestSubclassComparisonParity
        a, b = cls.FlippedStr("a"), cls.FlippedStr("b")
        _assert_parity({b: 1, a: 2})
        _assert_parity({a: 1, b: 2})

    def test_eq_lying_keys_fall_the_tiebreak_to_the_values(self) -> None:
        cls = TestSubclassComparisonParity
        obj = {cls.LyingEqStr("a"): 5, cls.LyingEqStr("b"): 3}
        _assert_parity(obj)

    def test_mixed_exact_and_subclass_keys_delegate(self) -> None:
        cls = TestSubclassComparisonParity
        _assert_parity({1: "plain", cls.FlippedInt(9): "sub"})
        _assert_parity({"a": 1, cls.ReprStr("b"): 2})
        _assert_parity({"b": 1, cls.FlippedStr("a"): 2})

    def test_subclass_values_and_keys_use_base_spellings(self) -> None:
        """json.dumps spells int/float subclass VALUES and KEYS via the
        BASE type's repr (an IntEnum dumps as its number; a
        repr-overriding float subclass dumps as 0.5) and str subclasses by
        their CONTENT; tors matches all three."""
        from enum import IntEnum

        class Big(IntEnum):
            SMALL = 5
            HUGE = 2**70

        _assert_parity({"v": TestSubclassComparisonParity.ReprStr("actual")})
        _assert_parity({TestSubclassComparisonParity.ReprStr("k"): 1})
        _assert_parity({"v": 7.5, "w": TestSubclassComparisonParity.ReprFloat(0.5)})
        _assert_parity({TestSubclassComparisonParity.ReprFloat(2.5): 1})
        _assert_parity([Big.SMALL, Big.HUGE])
        _assert_parity({Big.SMALL: "x", Big.HUGE: "y"})


class TestContainerSubclassIterationParity:
    """The red-team P1 lane, pinned: json.dumps does NOT walk a container
    subclass's concrete storage. A non-exact dict is iterated through
    ``PyMapping_Items`` -- the OVERRIDABLE ``.items()``, materialized into
    a snapshot and sorted -- and a non-exact list/tuple through
    ``PyObject_GetIter`` (its ``__iter__``), materialized
    ``PySequence_Fast``-style (Modules/_json.c, stable 3.10 through
    3.14). A subclass that hides, fakes, reorders, or empties its content
    through those hooks therefore changes json's hash, and tors's walk
    delegates to the same interpreter calls so the two agree on every
    shape below. EXACT instances keep the concrete-storage fast path
    (json's own exact-dict ``sort_keys`` branch materializes the same
    items, byte-equivalent there), so the deep-nesting superset lane --
    ``TestHypothesisDifferential``'s 100k-level pin -- is untouched.

    The runaway lane: json's C encoder enters a recursive call at every
    container, so hooks yielding ever-fresh subclasses (which no
    circular marker can catch) die by ``RecursionError`` when the
    interpreter's recursion budget runs out. tors's walk is iterative,
    so the same input is capped instead: PROTOCOL frames -- subclass
    containers only -- are counted against ``sys.getrecursionlimit()``
    (read at walk start), and past the cap the walk raises
    ``RecursionError`` itself. A finite chain well inside the limit
    hashes with full parity on both sides.
    """

    class HidingDict(dict):
        def items(self):
            return [(k, v) for k, v in dict.items(self) if k != "secret"]

    class FakingDict(dict):
        def items(self):
            return list(dict.items(self)) + [("a", 1)]

    class EmptyFakingDict(dict):
        calls = 0

        def items(self):
            type(self).calls += 1
            return [("a", 1)]

    class IterOnlyDict(dict):
        def __iter__(self):
            return iter(["lie"])

    class SkippingList(list):
        def __iter__(self):
            return iter([x for x in list.__iter__(self) if x != 2])

    class FakingList(list):
        def __iter__(self):
            return iter([1, 2, 99])

    class EmptyIterTuple(tuple):
        def __new__(cls, xs):
            return tuple.__new__(cls, xs)

        def __iter__(self):
            return iter(())

    class MutatingIter(list):
        def __iter__(self):
            self.append(99)  # during materialization: captured by the snapshot
            return iter(list.__iter__(self))

    class SelfYieldList(list):
        def __iter__(self):
            yield self

    class SelfYieldDict(dict):
        def __init__(self):
            super().__init__(x=1)

        def items(self):
            return [("k", self)]

    class FreshChainList(list):
        def __iter__(self):
            yield type(self)()

    class FreshChainDict(dict):
        def __init__(self):
            super().__init__(x=1)

        def items(self):
            return [("k", type(self)())]

    class ChainList(list):
        def __init__(self, inner):
            super().__init__()
            self._inner = inner

        def __iter__(self):
            yield self._inner

    def test_dict_subclass_items_is_called_through_the_interpreter(self) -> None:
        cls = TestContainerSubclassIterationParity
        _assert_parity(cls.HidingDict({"a": 1, "secret": 2}))
        _assert_parity(cls.FakingDict({"x": 0}))
        _assert_parity(cls.FakingDict({"z": 0, "m": 3}))

    def test_empty_storage_dict_subclass_never_calls_items(self) -> None:
        """json's ``{}`` gate reads the CONCRETE storage size before
        anything else, so a subclass dict with empty storage emits ``{}``
        without ever calling ``.items()``: a faking ``.items()`` on an
        empty dict is a no-op on both sides, pinned with the call
        counter."""
        cls = TestContainerSubclassIterationParity
        obj = cls.EmptyFakingDict()
        assert _canonical(obj) == "{}"
        assert cls.EmptyFakingDict.calls == 0
        _assert_bytes(obj, b"{}")
        assert cls.EmptyFakingDict.calls == 0

    def test_iter_only_dict_subclass_ignores_the_iter_override(self) -> None:
        """The dict lane is ``.items()``, never ``__iter__``: an
        ``__iter__``-only override on a dict subclass changes nothing on
        either side (json never calls it for dicts)."""
        cls = TestContainerSubclassIterationParity
        _assert_parity(cls.IterOnlyDict({"b": 2, "a": 1}))

    def test_list_and_tuple_subclasses_iterate_via_the_protocol(self) -> None:
        cls = TestContainerSubclassIterationParity
        _assert_parity(cls.SkippingList([1, 2, 3]))
        _assert_parity(cls.FakingList([1]))
        _assert_parity(cls.EmptyIterTuple((1, 2)))
        # Snapshot semantics: a mutation during materialization is
        # captured identically on both sides. The mutation is per-CALL
        # state (every __iter__ invocation appends again), so each
        # engine gets its own fresh instance: json on one, tors on
        # another, and both snapshots are [1,2,99].
        assert _canonical(cls.MutatingIter([1, 2])) == "[1,2,99]"
        assert content_hash(cls.MutatingIter([1, 2])) == _oracle([1, 2, 99])

    def test_protocol_yielded_items_re_enter_the_type_dispatch(self) -> None:
        """Whatever ``__iter__`` yields is walked by the same per-object
        dispatch: exact containers, subclass containers (which delegate
        again), and rejected leaves raise identically."""
        cls = TestContainerSubclassIterationParity

        class YieldsContainers(list):
            def __iter__(self):
                yield cls.HidingDict({"a": 1, "secret": 2})
                yield cls.SkippingList([1, 2, 3])
                yield {"plain": (1, cls.FakingList([1]))}

        _assert_parity(YieldsContainers())

        class YieldsBadLeaf(list):
            def __iter__(self):
                yield set()

        with pytest.raises(TypeError, match="set"):
            content_hash(YieldsBadLeaf())
        with pytest.raises(TypeError):
            _oracle(YieldsBadLeaf())

    def test_self_referential_protocol_shapes_raise_circular_value_error(self) -> None:
        """The literal self-yielders are TRUE cycles on both sides (the
        circular markers are enter/exit around the materialized
        children), so both engines raise the circular ValueError -- the
        runaway lane is the fresh-yielder below, not these."""
        cls = TestContainerSubclassIterationParity
        for obj in (cls.SelfYieldList(), cls.SelfYieldDict()):
            with pytest.raises(ValueError, match="[Cc]ircular reference detected"):
                content_hash(obj)
            with pytest.raises(ValueError):
                _oracle(obj)

    def test_the_runaway_protocol_chain_raises_recursion_error_on_both_sides(self) -> None:
        """Hooks yielding ever-fresh subclasses defeat the circular
        markers, so only a recursion budget can stop the descent: json
        dies by its C recursion guard, tors by the protocol-frame cap.
        Same exception class on both sides (the runaway guard's pin)."""
        cls = TestContainerSubclassIterationParity
        for obj in (cls.FreshChainList(), cls.FreshChainDict()):
            with pytest.raises(RecursionError):
                _oracle(obj)
            with pytest.raises(RecursionError):
                content_hash(obj)

    def test_a_deep_but_finite_protocol_chain_hashes_with_parity(self) -> None:
        """500 levels of list subclasses each overriding ``__iter__``,
        ending in a ``None`` leaf: well inside ``sys.getrecursionlimit()``
        (the default 1000) and inside json's budget on every supported
        interpreter, so BOTH engines hash it -- or, on an interpreter
        whose budget is tighter, both raise ``RecursionError``. Parity
        either way, independent of where the boundary sits."""
        obj: Any = None
        for _ in range(500):
            obj = TestContainerSubclassIterationParity.ChainList(obj)
        try:
            expected = _oracle(obj)
        except RecursionError:
            with pytest.raises(RecursionError):
                content_hash(obj)
        else:
            assert content_hash(obj) == expected

    def test_items_must_return_2_tuples_and_the_sort_runs_first(self) -> None:
        """json's own ``items()`` contract, byte-identical: the sort runs
        over the pairs AS YIELDED (an unsortable mix raises the sort's
        own TypeError first, before any validation), then each pair must
        be a 2-sized tuple -- subclass-tolerant, read from concrete
        storage, an overriding ``__getitem__`` ignored -- or the shared
        ``ValueError: items must return 2-tuples`` fires. And validation
        is LAZY, in json's encode order: a bad VALUE at pair i raises
        before pair i+1 is validated."""
        from collections import namedtuple

        class ItemsDict(dict):
            def __init__(self, items):
                super().__init__(x=1)  # non-empty storage: the {} gate is off
                self._items = items

            def items(self):
                return self._items

        for bad in (
            [("z", 1, 1), ("a", 2, 2)],  # 3-tuples
            [["a", 1]],  # list pairs: not tuples
            ["not-a-pair"],  # not a sequence
        ):
            obj = ItemsDict(bad)
            with pytest.raises(ValueError) as tors_exc:
                content_hash(obj)
            with pytest.raises(ValueError) as oracle_exc:
                _oracle(obj)
            assert str(tors_exc.value) == str(oracle_exc.value)

        # the delegated sort fires before any pair is validated
        obj = ItemsDict([("a", 1), 5])
        with pytest.raises(TypeError) as tors_exc:
            content_hash(obj)
        with pytest.raises(TypeError) as oracle_exc:
            _oracle(obj)
        assert str(tors_exc.value) == str(oracle_exc.value)

        # lazy order: the set value at sorted position 0 raises before
        # the 3-tuple at position 1 is even validated
        obj = ItemsDict([("a", set()), ("z", 1, 1)])
        with pytest.raises(TypeError, match="set"):
            content_hash(obj)
        with pytest.raises(TypeError):
            _oracle(obj)

        # tuple-subclass pairs are accepted and read from concrete storage
        Pair = namedtuple("Pair", "k v")

        class GetItemLiar(tuple):
            def __new__(cls, k, v):
                return tuple.__new__(cls, (k, v))

            def __getitem__(self, i):
                return "LIE"

        _assert_parity(ItemsDict([Pair("b", 1), ("a", 2)]))
        _assert_parity(ItemsDict([GetItemLiar("b", 1), ("a", 2)]))

    def test_plain_and_standard_container_subclasses_match_the_oracle(self) -> None:
        """(Re-scoped from ``test_container_subclasses_walk_native_storage``,
        whose name asserted the WRONG contract -- native storage -- for
        containers json itself never walks natively.) The boring
        subclasses, through the protocol lane: plain list/dict
        subclasses, OrderedDict, namedtuple, and an ``__iter__``-faithful
        list subclass."""
        from collections import OrderedDict, namedtuple

        class PlainListSub(list):
            pass

        class PlainDictSub(dict):
            pass

        class FaithfulIterList(list):
            def __iter__(self):
                return iter(list.__iter__(self))

        Point = namedtuple("Point", "x y")

        _assert_parity(OrderedDict([("b", 1), ("a", 2)]))
        _assert_parity({"sub": PlainListSub([3, 1, 2])})
        _assert_parity(PlainDictSub({"z": 1, "a": 2}))
        _assert_parity(FaithfulIterList([1, {"k": PlainListSub([2])}]))
        _assert_parity(Point(1, 2))
        _assert_parity([Point(1, 2), OrderedDict(a=1)])


class TestCanonicalByteLiterals:
    """Whole-tree canonical forms pinned as literal bytes (sha256'd here),
    independent of the oracle: the emitter's structural spellings -- compact
    separators, sorted keys, nested containers -- in one place."""

    @pytest.mark.parametrize(
        ("obj", "canonical"),
        [
            ({}, b"{}"),
            ([], b"[]"),
            ([1, 2, 3], b"[1,2,3]"),
            ((1, 2, 3), b"[1,2,3]"),
            ({"a": []}, b'{"a":[]}'),
            ({"b": 1, "a": 2}, b'{"a":2,"b":1}'),
            ([{"x": 1}, {"y": [2, 3]}], b'[{"x":1},{"y":[2,3]}]'),
            (
                {"nested": {"deep": {"deeper": [True, False, None]}}},
                b'{"nested":{"deep":{"deeper":[true,false,null]}}}',
            ),
            ({"s": "with spaces and:colons,commas"}, b'{"s":"with spaces and:colons,commas"}'),
            ({"a": [1.5, "x", {"b": ()}]}, b'{"a":[1.5,"x",{"b":[]}]}'),
        ],
        ids=[
            "empty-dict",
            "empty-list",
            "int-list",
            "int-tuple",
            "empty-nested",
            "two-sorted",
            "list-of-dicts",
            "deep-dict",
            "separator-chars-in-strings",
            "kitchen-sink",
        ],
    )
    def test_tree_literals(self, obj: Any, canonical: bytes) -> None:
        _assert_bytes(obj, canonical)
        _assert_parity(obj)
