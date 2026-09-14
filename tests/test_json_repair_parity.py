"""Differential parity + fuzz-shaped invariants for the tors json_repair port.

Provenance: the behavior oracle is json_repair by Stefano Baccianella (MIT,
https://github.com/mangiucugna/json_repair) pinned to json-repair==0.63.4
(commit 251d141786d0f6ff561f6ec04d90188a338e2470). The pin is deliberate: a
version bump is a parity re-sync request, not a drive-by upgrade. Mapping:
``tors.repair_json(raw, **kw) == json_repair.repair_json(raw, **kw)`` (and
likewise for ``repair_json_loads``). This file owns the design §10
differential + hypothesis lanes: corpus equality over repair shapes, schema
and strict-mode differentials, and seed-free hypothesis invariants over
small deterministic mutations. Tors-native divergences (design §9) are never
oracle-compared here; each exclusion cites its §9 number.
"""

from __future__ import annotations

import itertools
import json
import random
import re
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import tors

json_repair_lib = pytest.importorskip("json_repair")  # pin: json-repair==0.63.4

_BASE_RAWS: list[str] = [
    '{"a": 1 "b": 2}',  # missing comma: object members
    "[1 2 3]",  # missing comma: array items
    '{"a": 1, "b": 2 "c": 3}',  # missing comma: later member
    '{"a": 1, "b": 2',  # missing bracket: unclosed object
    "[1, 2, 3",  # missing bracket: unclosed array
    '{"a": "hello}',  # missing quote: unterminated string value
    '{"a: 1}',  # missing quote: unterminated key
    '{"a": {"b": [1, 2',  # truncated: nested containers
    '[{"a": 1}, {"b":',  # truncated: array of objects
    '"hello wo',  # truncated: top-level string
    "{'a': 1}",  # single quotes: object
    "['a', 'b']",  # single quotes: array
    "{'a': 'hi'}",  # single quotes: key and value
    "{a: 1}",  # unquoted key: single
    "{a: 1, b: 2}",  # unquoted keys: multiple
    '{name: "bob"}',  # unquoted key: string value
    '{"a": /*x*/ 1}',  # comment: inline block comment
    '// lead\n{"a": 1}',  # comment: leading line comment
    '/*x*/{"a": 1}',  # comment: leading block comment
    'Answer: {"a": 1}',  # prose prefix: object payload
    "Here: [1, 2]",  # prose prefix: array payload
    'Result: {"a": true}',  # prose prefix: boolean member
    "(1, 2, 3)",  # tuple: top-level parenthesized
    '{"a": (1, 2)}',  # tuple: parenthesized value
    "(1,)",  # tuple: single element
    '{"a": True}',  # Python literal: True
    '{"a": False}',  # Python literal: False
    '{"a": None}',  # Python literal: None
    "[True, False, None]",  # Python literals: mixed array
    '{"a": True, "b": None}',  # Python literals: mixed object
    '""hi""',  # doubled quotes: top-level string
    '{"a": ""hi""}',  # doubled quotes: member value
    '{"a": "x\\qy"}',  # escape: invalid \\q escape
    '{"a": "\\u41"}',  # escape: truncated \\u escape
    '[{\\"\\":]w]"',  # escaped-key splice: ']' kept in value (issue #39)
    '[{\\"\\":]x]}"',  # escaped-key splice: ']' and '}' kept in value (issue #39)
    '```json\n{"a": 1}\n```',  # fenced payload: object (no top-level scalar)
    "```json\n[1, 2]\n```",  # fenced payload: array (no top-level scalar)
    '~~~json\n{"a": 1}\n~~~',  # fenced payload: tilde fence object
    '```\n{"a": 1}\n```',  # fenced payload: untagged fence object
    '{"a": {"b": 1 "c": 2}}',  # nested damage: missing comma deep
    '{"a": [1 2]}',  # nested damage: array missing comma
    "{'a': {'b': 1}}",  # nested damage: single quotes deep
    "{a: {b: 2}}",  # nested damage: unquoted keys deep
    '{"a": {"b": {',  # nested damage: truncated deep
    '[{"a": [1, {"b": 2',  # nested damage: truncated mixed stack
    '{"a": 1,}',  # trailing comma: object
    "[1, 2,]",  # trailing comma: array
    '{"a" 1}',  # missing colon after key
    '{"a": 1,, "b": 2}',  # doubled comma between members
    "",  # empty input: nothing-recoverable sentinel
    "   ",  # whitespace only: sentinel
    '""',  # valid empty string: the sentinel spelling both sides share
    '{"a": }',  # missing value after colon
    '{"a": "b" "c": "d"}',  # missing comma: string values
    '[1, {"a": True}, None]',  # nested damage: Python literals in array
    '{"a": [1, 2}',  # mismatched closer: brace for bracket
    "not json at all",  # prose only: sentinel
    '{"a": "x", "b": }',  # missing value: second member
    "{a: 'x', b: None}",  # combined: unquoted + single + None
    '{"a":1}{"b":2}',  # concatenated objects
    "{" + r"{\"k\": 1}" * 64 + "}",  # escaped-delimiter run in a string
    # body: the escape normalizer's incremental undo record stays
    # byte-identical to the oracle here.
    r'{"bs": "\\\\", "m": "\\"k\\" \u201e x"}',  # backslash run +
    # delimiter unescape + smart quote in one body: exercises every
    # acc_pop repair arm against the oracle.
    "[" + r"{\"k\": \"v\"}" * 48 + "]",  # escaped key and value run
    # in an array body: the heaviest escape-repair density, pinned.
    '{"n": {"d": {"x": True}}}',  # nested damage: deep Python literal
    'Answer is: [1, {"a": None}]',  # prose prefix: nested literal array
    '["' + "]" * 64 + '" x',  # array-context `]` run in a string body: the
    # memoized-lookahead O(n^2) fix stays byte-identical to the oracle here.
    r"""[{"a": "]}\\"x"}]""",  # mixed `]`/`}`/`\\`/`"` in an array-of-object
    # string body: exercises the shared `[outer]` memo across the `]` and `}`
    # sites and pins it byte-identical to the oracle.
    '["' + ("]" + "\\\\") * 32 + '" x',  # interleaved `]`/even-backslash-run
    # string body: the incremental escape-tail rewrite (upstream rebuilds the
    # accumulator per normalization) stays byte-identical to the oracle.
    '["' + 'a"' * 64 + '"]',  # internal-quote run in an array string: the
    # pairing-walk outcome memo (upstream re-walks per quote candidate).
    '{"a": "' + "}" * 64 + '"' + "y" * 64 + '"z',  # object-value `}` run with a
    # long quote-free gap: the `}`-branch's lstring lookahead memo.
    '{"a": "[' + 'x"' * 64 + '"}',  # quote run under an open regex character
    # class: the whitespace-flag + memoized `]` lookahead rewrite.
    "{" + "a:b," * 64 + "}",  # unquoted-key member run: the parser-level
    # lookahead memo shared across the run's many short string parses.
]

# note (§9.4): no fenced top-level scalar lives in _BASE_RAWS: tors recovers
# them while the oracle returns "", so they can never be oracle-compared.
CORPUS: list[tuple[str, dict[str, Any]]] = [(raw, {}) for raw in _BASE_RAWS] + [
    (raw, {"skip_json_loads": True}) for raw in _BASE_RAWS
]

# Deliberately excluded from every schema differential below (§9.7 tors-native
# behavior changes, pinned tors-side elsewhere, never oracle-compared here):
# - typo keys remapped via the key-normalization/fuzzy ladder (§6.0-1): §9.7
# - date/date-time format normalization schemas (§6.3): §9.7
# - comma-split strings ("1, 2, 3" -> [1, 2, 3], oracle raises) (§6.1b): §9.7
# - digit-group separator strings ("1,234" -> 1234, oracle raises) (§6.1c): §9.7
# - string-enum near-misses (tors appends "Did you mean ...") (§6.2): §9.7
# - percent / currency / prose single-number extraction ("50%" -> 0.5 or 50
#   by declared type, "USD 50" -> 50, oracle raises) (§6.1c tiers): §9.7
# - separator-ambiguous numerics under Auto (assume-en-US schema-checked;
#   "1,234" on a number field -> 1234 + disclosure, oracle raises): §9.7
SCHEMA_CORPUS: list[tuple[str, dict[str, Any]]] = [
    # coercion: integer 1 -> string "1"
    ('{"a": 1}', {"type": "object", "properties": {"a": {"type": "string"}}}),
    # coercion: numeric string "42" -> integer 42
    ('{"a": "42"}', {"type": "object", "properties": {"a": {"type": "integer"}}}),
    # coercion: numeric string "3.5" -> number 3.5
    ('{"a": "3.5"}', {"type": "object", "properties": {"a": {"type": "number"}}}),
    # coercion: integer 1 -> boolean True
    ('{"a": 1}', {"type": "object", "properties": {"a": {"type": "boolean"}}}),
    # fill: missing key completed from default
    (
        "{}",
        {"type": "object", "properties": {"a": {"type": "string", "default": "hi"}}},
    ),
    # fill: required key already present passes through
    (
        '{"a": 1}',
        {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
    ),
    # coercion: array items "1"/"2" -> integers
    (
        '{"a": ["1", "2"]}',
        {
            "type": "object",
            "properties": {"a": {"type": "array", "items": {"type": "integer"}}},
        },
    ),
    # unwrap: JSON-string container "[1, 2]" -> array with coerced items
    (
        '{"a": "[1, 2]"}',
        {
            "type": "object",
            "properties": {"a": {"type": "array", "items": {"type": "integer"}}},
        },
    ),
    # union: oneOf picks the validating branch for "1"
    (
        '{"a": "1"}',
        {
            "type": "object",
            "properties": {"a": {"oneOf": [{"type": "integer"}, {"type": "string"}]}},
        },
    ),
    # $ref: local "#/$defs/n" pointer resolves to integer
    (
        '{"a": 1}',
        {
            "$defs": {"n": {"type": "integer"}},
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/n"}},
        },
    ),
    # enum: integer member passes with no suggestion text involved
    ('{"a": 2}', {"type": "object", "properties": {"a": {"enum": [1, 2, 3]}}}),
    # boolean schema True allows anything
    ("true", True),  # type: ignore[list-item]
]

SCHEMA_VALUE_RAISE_CORPUS: list[tuple[str, dict[str, Any]]] = [
    # pattern: "abc" violates "^\\d+$": both raise ValueError; the validation
    # message wording differs (§9.5 jsonschema-crate texts), so only the raise
    # itself is compared, never the message.
    (
        '{"a": "abc"}',
        {"type": "object", "properties": {"a": {"type": "string", "pattern": "^\\d+$"}}},
    ),
]

_DEEP_DEPTH = 500
_DEEP_RAW = '{"x": ' * _DEEP_DEPTH + "1" + "}" * _DEEP_DEPTH


def _deep_schema(depth: int) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "integer"}
    for _ in range(depth):
        schema = {"type": "object", "properties": {"x": schema}}
    return schema


DEEP_SCHEMA: dict[str, Any] = _deep_schema(_DEEP_DEPTH)

STRICT_CORPUS: list[str] = [
    # The real upstream strict corpus (test_strict_mode.py): strict mode
    # rejects structural ambiguity, not ordinary repairable damage:
    # single quotes, trailing commas, truncation and plain-object
    # duplicate keys all repair fine upstream and are corpus cases, not
    # strict errors.
    '{"key":"value"}["value"]',  # multiple top-level values
    '{"key":"value"}, {"key":"value_after"}',  # comma-separated same shape
    '{"key":"value"}{"key":"value_after"}',  # adjacent same-shape objects
    "[1][2]",  # adjacent same-shape arrays
    '[{"key": "first", "key": "second"}]',  # duplicate key inside array
    '{"" : "value"}',  # empty key
    '{"missing" "colon"}',  # missing ':' after key
    '{"key": , "key2": "value2"}',  # empty value
    '{"dangling"}',  # empty object with extra characters
]


class TestDifferentialParity:
    @pytest.mark.parametrize(
        ("raw", "kwargs"), CORPUS, ids=[f"corpus-{i}" for i in range(len(CORPUS))]
    )
    def test_str_parity(self, raw: str, kwargs: dict[str, Any]) -> None:
        got = tors.repair_json(raw, **kwargs)
        want = json_repair_lib.repair_json(raw, **kwargs)
        assert got == want

    @pytest.mark.parametrize(
        ("raw", "kwargs"), CORPUS, ids=[f"corpus-{i}" for i in range(len(CORPUS))]
    )
    def test_loads_parity(self, raw: str, kwargs: dict[str, Any]) -> None:
        got = tors.repair_json_loads(raw, **kwargs)
        want = json_repair_lib.loads(raw, **kwargs)
        assert got == want

    @pytest.mark.parametrize(
        ("raw", "schema"),
        SCHEMA_CORPUS,
        ids=[f"schema-{i}" for i in range(len(SCHEMA_CORPUS))],
    )
    def test_schema_parity(self, raw: str, schema: dict[str, Any] | bool) -> None:
        pytest.importorskip("jsonschema")
        got = tors.repair_json(raw, schema=schema)
        want = json_repair_lib.repair_json(raw, schema=schema)
        assert got == want
        got_loads = tors.repair_json_loads(raw, schema=schema)
        want_loads = json_repair_lib.loads(raw, schema=schema)
        assert got_loads == want_loads

    @pytest.mark.parametrize(
        ("raw", "schema"),
        SCHEMA_VALUE_RAISE_CORPUS,
        ids=[f"schema-raise-{i}" for i in range(len(SCHEMA_VALUE_RAISE_CORPUS))],
    )
    def test_schema_raise_parity(self, raw: str, schema: dict[str, Any]) -> None:
        pytest.importorskip("jsonschema")
        with pytest.raises(ValueError):
            tors.repair_json(raw, schema=schema)
        with pytest.raises(ValueError):
            json_repair_lib.repair_json(raw, schema=schema)

    def test_pattern_properties_extras_keep_input_order(self) -> None:
        # Extras emit at their input positions (upstream's value.items()
        # pass): pattern folds inline with kept extras, never regrouped.
        schema: dict[str, Any] = {"patternProperties": {"^x-": {"type": "string"}}}
        raw = '{"b": 1, "x-1": "a"}'
        assert tors.repair_json(raw, schema=schema, skip_json_loads=True) == (
            json_repair_lib.repair_json(raw, schema=schema, skip_json_loads=True)
        )
        assert tors.repair_json_loads(raw, schema=schema, skip_json_loads=True) == (
            json_repair_lib.loads(raw, schema=schema, skip_json_loads=True)
        )

    @pytest.mark.parametrize(
        ("raw", "prop"),
        [
            ('{"n": "12345678901234567890123"}', {"type": "integer"}),
            ('{"n": "12345678901234567890123.0"}', {"type": "number"}),
            ('{"n": 12345678901234567890123}', {"type": "string"}),
        ],
    )
    def test_unbounded_int_semantics_match(self, raw: str, prop: dict[str, Any]) -> None:
        # Python's int()/str() are exact at any magnitude; the port's
        # BigInt path must match byte-for-byte on both lanes.
        schema: dict[str, Any] = {"type": "object", "properties": {"n": prop}}
        assert tors.repair_json(raw, schema=schema, skip_json_loads=True) == (
            json_repair_lib.repair_json(raw, schema=schema, skip_json_loads=True)
        )
        assert tors.repair_json_loads(raw, schema=schema, skip_json_loads=True) == (
            json_repair_lib.loads(raw, schema=schema, skip_json_loads=True)
        )

    def test_deep_recursion_both_raise(self) -> None:
        # §9.6: tors normalizes deep nesting to ValueError at depth 200 while
        # the oracle raises RecursionError (uncaught) near the interpreter
        # limit: both raise, types differ, so only the both-raise shape is
        # compared, never messages.
        pytest.importorskip("jsonschema")
        with pytest.raises(ValueError):
            tors.repair_json(_DEEP_RAW, schema=DEEP_SCHEMA)
        with pytest.raises((ValueError, RecursionError)):
            json_repair_lib.repair_json(_DEEP_RAW, schema=DEEP_SCHEMA)

    def test_continuation_chain_recursion_both_raise(self) -> None:
        # §9.6 both-raise pin for the array-continuation merge chain
        # (`{"a":[0],` + `["b":[0],` * N): tors caps it at MAX_NESTING
        # fragments and the oracle at its own recursion limit, so at a size
        # far past both thresholds each engine raises a ValueError: the
        # differential signal that neither crashes. (A tors build without
        # the continuation guard dies with SIGSEGV here, which kills the
        # process rather than failing the assertion: the native suite's
        # sub-2k sizes fail cleanly instead.)
        merge_chain = '{"a":[0],' + '["b":[0],' * 2_000 + "1]"
        with pytest.raises(ValueError):
            tors.repair_json(merge_chain, skip_json_loads=True)
        with pytest.raises(ValueError):
            json_repair_lib.repair_json(merge_chain, skip_json_loads=True)

    @pytest.mark.parametrize(
        "raw", STRICT_CORPUS, ids=[f"strict-{i}" for i in range(len(STRICT_CORPUS))]
    )
    def test_error_parity(self, raw: str) -> None:
        # skip_json_loads=True matches upstream's own strict-test spellings:
        # duplicate-keys-in-array and empty-key inputs are valid json.loads
        # input (last-wins duplicate handling; an empty key is legal JSON),
        # so the strict raise only fires from the repair parser: the fast
        # path would return them repaired on both sides. Types only, never
        # messages (§9.5 wording).
        with pytest.raises(ValueError):
            tors.repair_json(raw, strict=True, skip_json_loads=True)
        with pytest.raises(ValueError):
            json_repair_lib.repair_json(raw, strict=True, skip_json_loads=True)
        with pytest.raises(ValueError):
            tors.repair_json_loads(raw, strict=True, skip_json_loads=True)
        with pytest.raises(ValueError):
            json_repair_lib.loads(raw, strict=True, skip_json_loads=True)


class TestLoneSurrogateEscapeDivergence:
    r"""The lone-surrogate-escape class, pinned on both engines (api.md "Lone surrogates").

    A `\uXXXX` escape decoding to an unpaired surrogate (a high half not
    directly followed by a `\udCXX` low-half escape, or a bare low half) is
    the one string-value class where byte-parity with the oracle is
    architecturally out of reach, so it is classified — api.md Divergences
    → "Lone surrogates" entry; the pinned unit tests in
    src/json_repair/string.rs (the escape decoder) and
    src/json_repair/strict.rs (the strict fast path); policy thread
    https://github.com/AZX-PBC-OSS/tors/issues/57 (CLOSED, keep-U+FFFD lean)
    — instead of oracle-compared for equality. (Review shorthand for this
    class is api.md "Lone surrogates"; the traceable pins are the api.md entry, the two Rust
    unit pins, and #57.) Mechanism: upstream decodes every `\u`
    escape with chr(int(hex, 16)) and a Python str carries the raw lone
    surrogate onward — json.dumps re-emits it as the identical escape text
    on the str lane, and the raw surrogate itself comes back on the loads
    lane (and with ensure_ascii=False). A Rust String cannot hold
    U+D800..U+DFFF (char::from_u32 rejects the block), so the port's escape
    decoder maps a lone half to U+FFFD. Preserving the escape TEXT in the
    value instead would match the str lane but corrupt the value (six
    characters of escape text where the oracle holds one surrogate), still
    diverge on the loads lane (pyo3 marshals strings through Rust str), and
    break the str/loads internal consistency pinned below — a full fix
    needs a raw-slice value representation, a maintainer decision, not a
    parser patch.

    This pin holds the classified line on the exact minimal shapes so a
    change on EITHER side — a json-repair re-sync that alters the class, or
    a tors policy change — fails here and forces re-classification, the
    same tripwire role the §9.4/§9.5/§9.6 pins above play for their
    classes. Reachability note: the hypothesis mutation lane CAN draw this
    class even though _base_text excludes the surrogate block and
    backslash — json.dumps reintroduces `\uXXXX` escapes (and `\"`)
    over that alphabet and _mutate corrupts the dumped text: the value
    ["A" + chr(0x0D80) + "1"] dumps as '["A\u0d801"]' and
    _mutate(..., seed=43064841) deletes the 0 to yield '["A\ud801"]'
    (issue #57), pinned as test_mutation_reachable_ud801_diverges below.
    The mutation differential therefore skips surrogate-shaped inputs
    explicitly (see test_differential_on_mutations); Hypothesis replays its
    example DB on every run, so a recorded hit replays into that skip
    instead of failing as a mystery flake.
    """

    # (raw, oracle str-lane spelling, tors str-lane spelling). The oracle
    # preserves the escape text (json.dumps of the raw surrogate re-emits
    # it, lowercased); tors maps every lone half to U+FFFD (api.md "Lone surrogates").
    _LONE_STR_CASES: list[tuple[str, str, str]] = [
        ('["\\ud800"]', '["\\ud800"]', '["\\ufffd"]'),
        ('["\\udc00"]', '["\\udc00"]', '["\\ufffd"]'),
        ('["\\udbff"]', '["\\udbff"]', '["\\ufffd"]'),  # high-half range end
        ('["\\udfff"]', '["\\udfff"]', '["\\ufffd"]'),  # low-half range end
        ('["\\uD800"]', '["\\ud800"]', '["\\ufffd"]'),  # uppercase: dumps lowercases
        ('{"k":"\\ud800"}', '{"k": "\\ud800"}', '{"k": "\\ufffd"}'),
        ('{"\\ud800": 1}', '{"\\ud800": 1}', '{"\\ufffd": 1}'),
        (
            '{"\\udc00\\ud800": 1}',
            '{"\\udc00\\ud800": 1}',
            '{"\\ufffd\\ufffd": 1}',
        ),  # key: low-then-high
        (
            '{"k":"\\udc00\\ud800"}',
            '{"k": "\\udc00\\ud800"}',
            '{"k": "\\ufffd\\ufffd"}',
        ),  # value: low-then-high
        ('["\\ud83d x"]', '["\\ud83d x"]', '["\\ufffd x"]'),
        ('["a\\ud800b"]', '["a\\ud800b"]', '["a\\ufffdb"]'),
        (
            '["\\udc00\\ud800"]',
            '["\\udc00\\ud800"]',
            '["\\ufffd\\ufffd"]',
        ),  # reversed pair
        (
            '["\\ud800\\ud800"]',
            '["\\ud800\\ud800"]',
            '["\\ufffd\\ufffd"]',
        ),  # doubled high half
        (
            '["\\ud800\\u0041"]',
            '["\\ud800A"]',
            '["\\ufffdA"]',
        ),  # high + valid non-low: the \\u0041 still decodes to A
        (
            '["\\ud800\\u41"]',
            '["\\ud800\\\\u41"]',
            '["\\ufffd\\\\u41"]',
        ),  # high + truncated: the short escape stays literal text
        (
            '["A\\ud801"]',
            '["A\\ud801"]',
            '["A\\ufffd"]',
        ),  # seed-43064841 product shape (issue #57)
    ]

    # (raw, oracle loads-lane value, tors loads-lane value): the oracle
    # carries the raw surrogate in the Python str; tors carries U+FFFD.
    _LONE_LOADS_CASES: list[tuple[str, list[str], list[str]]] = [
        ('["\\ud800"]', ["\ud800"], ["\ufffd"]),
        ('["a\\ud800b"]', ["a\ud800b"], ["a\ufffdb"]),
        ('["\\udbff"]', ["\udbff"], ["\ufffd"]),
        ('["\\udc00\\ud800"]', ["\udc00\ud800"], ["\ufffd\ufffd"]),
        ('["\\ud800\\u0041"]', ["\ud800A"], ["\ufffdA"]),
    ]

    @pytest.mark.parametrize(
        ("raw", "want_oracle", "want_tors"),
        _LONE_STR_CASES,
        ids=[
            "bare-high",
            "bare-low",
            "high-range-end",
            "low-range-end",
            "uppercase-hex",
            "object-value",
            "object-key",
            "key-low-then-high",
            "value-low-then-high",
            "high-then-space",
            "embedded",
            "reversed-pair",
            "doubled-high",
            "high-then-valid-non-low",
            "high-then-truncated",
            "seed-43064841-shape",
        ],
    )
    def test_oracle_preserves_the_lone_surrogate(
        self, raw: str, want_oracle: str, want_tors: str
    ) -> None:
        # The str lane: json.dumps re-emits the raw lone surrogate as the
        # identical escape text, on the repair lane (skip_json_loads) and
        # the json.loads fast lane alike. want_tors is asserted alongside
        # so a silent oracle re-sync that alters the class fails here too.
        assert json_repair_lib.repair_json(raw, skip_json_loads=True) == want_oracle
        assert json_repair_lib.repair_json(raw) == want_oracle
        assert tors.repair_json(raw, skip_json_loads=True) == want_tors
        assert tors.repair_json(raw) == want_tors

    @pytest.mark.parametrize(
        ("raw", "want_oracle", "want_tors"),
        _LONE_STR_CASES,
        ids=[
            "bare-high",
            "bare-low",
            "high-range-end",
            "low-range-end",
            "uppercase-hex",
            "object-value",
            "object-key",
            "key-low-then-high",
            "value-low-then-high",
            "high-then-space",
            "embedded",
            "reversed-pair",
            "doubled-high",
            "high-then-valid-non-low",
            "high-then-truncated",
            "seed-43064841-shape",
        ],
    )
    def test_tors_maps_lone_halves_to_the_replacement_char(
        self, raw: str, want_oracle: str, want_tors: str
    ) -> None:
        # api.md "Lone surrogates": the escape decoder's lone-surrogate arm (string.rs) and the
        # strict fast path's (strict.rs) agree — U+FFFD on every spelling,
        # on both lanes.
        assert tors.repair_json(raw, skip_json_loads=True) == want_tors
        assert tors.repair_json(raw) == want_tors
        assert json_repair_lib.repair_json(raw, skip_json_loads=True) == want_oracle
        assert json_repair_lib.repair_json(raw) == want_oracle

    @pytest.mark.parametrize(
        ("raw", "want_oracle", "want_tors"),
        _LONE_LOADS_CASES,
        ids=[
            "bare-high",
            "embedded",
            "high-range-end",
            "reversed-pair",
            "high-then-valid-non-low",
        ],
    )
    def test_loads_lane_split(self, raw: str, want_oracle: list[str], want_tors: list[str]) -> None:
        # The loads lane: the raw lone surrogate inside the Python str on
        # the oracle side (both lanes — the fast lane is stdlib json.loads
        # semantics), U+FFFD on the tors side.
        assert json_repair_lib.loads(raw, skip_json_loads=True) == want_oracle
        assert json_repair_lib.loads(raw) == want_oracle
        assert tors.repair_json_loads(raw, skip_json_loads=True) == want_tors
        assert tors.repair_json_loads(raw) == want_tors

    @pytest.mark.parametrize(
        ("raw", "want_oracle", "want_tors"),
        _LONE_STR_CASES,
        ids=[
            "bare-high",
            "bare-low",
            "high-range-end",
            "low-range-end",
            "uppercase-hex",
            "object-value",
            "object-key",
            "key-low-then-high",
            "value-low-then-high",
            "high-then-space",
            "embedded",
            "reversed-pair",
            "doubled-high",
            "high-then-valid-non-low",
            "high-then-truncated",
            "seed-43064841-shape",
        ],
    )
    @pytest.mark.parametrize("lane", ["default", "skip_json_loads"])
    def test_ensure_ascii_false_lane_split(
        self, raw: str, want_oracle: str, want_tors: str, lane: str
    ) -> None:
        # ensure_ascii=False on both lanes (not just skip_json_loads): the
        # raw surrogate itself, unescaped, vs U+FFFD — full 16-shape matrix
        # (low, ends, reversed, doubled, truncated, key shapes), not a
        # single bare-high probe.
        kwargs: dict[str, Any] = {} if lane == "default" else {"skip_json_loads": True}
        got_oracle = json_repair_lib.repair_json(raw, ensure_ascii=False, **kwargs)
        got_tors = tors.repair_json(raw, ensure_ascii=False, **kwargs)
        assert got_oracle == json.dumps(json.loads(want_oracle), ensure_ascii=False)
        assert got_tors == json.dumps(json.loads(want_tors), ensure_ascii=False)
        assert got_oracle != got_tors
        assert "\ufffd" in got_tors

    @pytest.mark.parametrize(
        ("raw", "want_oracle", "want_tors"),
        _LONE_STR_CASES,
        ids=[
            "bare-high",
            "bare-low",
            "high-range-end",
            "low-range-end",
            "uppercase-hex",
            "object-value",
            "object-key",
            "key-low-then-high",
            "value-low-then-high",
            "high-then-space",
            "embedded",
            "reversed-pair",
            "doubled-high",
            "high-then-valid-non-low",
            "high-then-truncated",
            "seed-43064841-shape",
        ],
    )
    @pytest.mark.parametrize("lane", ["default", "skip_json_loads"])
    def test_strict_lane_split(
        self, raw: str, want_oracle: str, want_tors: str, lane: str
    ) -> None:
        # strict=True shows the same classified split as the repair parser,
        # on both lanes and every shape — including high + valid-non-low,
        # where the strict decoder must leave the non-low escape unconsumed
        # for normal decoding (same as the string path: '["\\ufffdA"]').
        kwargs: dict[str, Any] = {} if lane == "default" else {"skip_json_loads": True}
        assert json_repair_lib.repair_json(raw, strict=True, **kwargs) == want_oracle
        assert tors.repair_json(raw, strict=True, **kwargs) == want_tors

    @pytest.mark.parametrize(
        ("raw", "want_oracle", "want_tors"),
        _LONE_STR_CASES,
        ids=[
            "bare-high",
            "bare-low",
            "high-range-end",
            "low-range-end",
            "uppercase-hex",
            "object-value",
            "object-key",
            "key-low-then-high",
            "value-low-then-high",
            "high-then-space",
            "embedded",
            "reversed-pair",
            "doubled-high",
            "high-then-valid-non-low",
            "high-then-truncated",
            "seed-43064841-shape",
        ],
    )
    @pytest.mark.parametrize("lane", ["default", "skip_json_loads"])
    def test_schema_lane_split(
        self, raw: str, want_oracle: str, want_tors: str, lane: str
    ) -> None:
        # A surrogate under a schema diverges the same way on both lanes;
        # the schema itself is orthogonal to the api.md "Lone surrogates"
        # arm, so a permissive True schema pins the split on every shape
        # (low, ends, reversed, doubled, truncated, key shapes).
        pytest.importorskip("jsonschema")
        kwargs: dict[str, Any] = {} if lane == "default" else {"skip_json_loads": True}
        assert json_repair_lib.repair_json(raw, schema=True, **kwargs) == want_oracle
        assert tors.repair_json(raw, schema=True, **kwargs) == want_tors

    @pytest.mark.parametrize(
        "raw",
        [case[0] for case in _LONE_STR_CASES],
        ids=[
            "bare-high",
            "bare-low",
            "high-range-end",
            "low-range-end",
            "uppercase-hex",
            "object-value",
            "object-key",
            "key-low-then-high",
            "value-low-then-high",
            "high-then-space",
            "embedded",
            "reversed-pair",
            "doubled-high",
            "high-then-valid-non-low",
            "high-then-truncated",
            "seed-43064841-shape",
        ],
    )
    @pytest.mark.parametrize("lane", ["default", "skip_json_loads"])
    def test_diagnostics_agree_with_loads(self, raw: str, lane: str) -> None:
        # The diagnostics spelling is the loads spelling on every shape and
        # lane — the full 16x2 matrix — AND the oracle split is pinned here,
        # not just tors↔tors internal consistency: tors value != oracle
        # loads on every shape. Diagnostics stay empty on this arm (P1
        # telescope note: distinct lone halves collapse to one U+FFFD with
        # no diagnostic emitted — documented at call time).
        kwargs: dict[str, Any] = {} if lane == "default" else {"skip_json_loads": True}
        value, diags = tors.repair_json_diagnostics(raw, **kwargs)
        assert value == tors.repair_json_loads(raw, **kwargs)
        assert value != json_repair_lib.loads(raw, **kwargs)
        assert diags == []

    def test_fffd_telescope_collapses_distinct_halves_with_empty_diagnostics(self) -> None:
        # P1 pre-existing telescope pin: distinct lone halves collapse to
        # one U+FFFD spelling with empty diagnostics — information loss
        # documented at call time (no surrogate-arm diagnostic exists).
        assert tors.repair_json('["\\ud800"]', skip_json_loads=True) == '["\\ufffd"]'
        assert tors.repair_json('["\\udc00"]', skip_json_loads=True) == '["\\ufffd"]'
        assert tors.repair_json_loads('["\\ud800"]', skip_json_loads=True) == ["\ufffd"]
        assert tors.repair_json_loads('["\\udc00"]', skip_json_loads=True) == ["\ufffd"]
        for raw in ('["\\ud800"]', '["\\udc00"]'):
            value, diags = tors.repair_json_diagnostics(raw, skip_json_loads=True)
            assert value == ["\ufffd"]
            assert diags == []

    @pytest.mark.parametrize("lane", ["default", "skip_json_loads"])
    def test_schema_pattern_enum_flip_on_fffd(self, lane: str) -> None:
        # P3 pin: a schema constraining the FFFD spelling flips validation
        # against the oracle, because the repaired VALUE differs. Tors holds
        # U+FFFD so pattern "^\\ufffd$" / enum ["\ufffd"] pass; the oracle
        # holds the raw surrogate so both raise ValueError.
        pytest.importorskip("jsonschema")
        kwargs: dict[str, Any] = {} if lane == "default" else {"skip_json_loads": True}
        raw = '{"a": "\\ud800"}'
        for schema in (
            {"type": "object", "properties": {"a": {"type": "string", "pattern": "^\\ufffd$"}}},
            {"type": "object", "properties": {"a": {"enum": ["\ufffd"]}}},
        ):
            assert tors.repair_json(raw, schema=schema, **kwargs) == '{"a": "\\ufffd"}'
            with pytest.raises(ValueError):
                json_repair_lib.repair_json(raw, schema=schema, **kwargs)

    def test_mutation_reachable_ud801_diverges(self) -> None:
        # Issue #57's exact route into this class: the value
        # ["A" + chr(0x0D80) + "1"] dumps as '["A\\u0d801"]' and
        # _mutate(..., seed=43064841) deletes the 0 to yield '["A\\ud801"]'
        # — a lone high surrogate reached through the hypothesis mutation
        # lane even though _base_text excludes Cs and backslash. Classified
        # api.md "Lone surrogates": tors maps to U+FFFD, the oracle preserves the escape text.
        value = ["A" + chr(0x0D80) + "1"]
        mutated = _mutate(json.dumps(value), 43064841)
        assert mutated == '["A\\ud801"]'
        assert tors.repair_json(mutated, skip_json_loads=True) == '["A\\ufffd"]'
        assert json_repair_lib.repair_json(mutated, skip_json_loads=True) == ('["A\\ud801"]')

    @pytest.mark.parametrize(
        "raw",
        [
            '["\\ud83d\\ude00"]',
            '["\\ud83d\\ude00",]',
            '["\\uD83D\\uDE00"]',
        ],
        ids=["pair", "pair-trailing-comma", "pair-uppercase-hex"],
    )
    def test_well_formed_pairs_stay_byte_identical_on_the_str_lane(self, raw: str) -> None:
        # A well-formed pair is NOT the divergence (api.md "Lone surrogates"): the port combines
        # the halves into the astral scalar and re-emits the identical pair
        # bytes (lowercased: the uppercase spelling normalizes), and the
        # oracle's raw halves serialize to the same text — byte-identical on
        # both lanes, the repair lane included (the trailing comma forces
        # it). A regression in the pair-vs-lone boundary
        # (decode_surrogate_pair) diverges here.
        assert tors.repair_json(raw, skip_json_loads=True) == '["\\ud83d\\ude00"]'
        assert tors.repair_json(raw, skip_json_loads=True) == (
            json_repair_lib.repair_json(raw, skip_json_loads=True)
        )
        # The loads lane is the pair face of the same architecture line
        # (api.md "Lone surrogates"): tors combines (stdlib json.loads semantics, pinned
        # natively in test_json_repair_native.py) while the oracle's repair
        # parser leaves the two raw halves in the value. On the default
        # lane both take the stdlib fast path and agree.
        assert tors.repair_json_loads('["\\ud83d\\ude00",]', skip_json_loads=True) == ["\U0001f600"]
        assert json_repair_lib.loads('["\\ud83d\\ude00",]', skip_json_loads=True) == [
            "\ud83d\ude00"
        ]
        assert tors.repair_json_loads('["\\ud83d\\ude00"]') == ["\U0001f600"]
        assert json_repair_lib.loads('["\\ud83d\\ude00"]') == ["\U0001f600"]

    @pytest.mark.parametrize(
        "raw",
        [
            '["\\ud800"]',
            '["\\udc00"]',
            '["\\udbff"]',
            '["\\udfff"]',
            '{"k":"\\ud800"}',
            '{"\\ud800": 1}',
            '{"\\udc00\\ud800": 1}',
            '["\\ud83d x"]',
            '["a\\ud800b"]',
            '["\\udc00\\ud800"]',
            '["\\ud800\\ud800"]',
            '["\\ud800\\u0041"]',
            '["\\ud83d\\ude00",]',
            '"\\ud800"',
        ],
        ids=[
            "bare-high",
            "bare-low",
            "high-range-end",
            "low-range-end",
            "object-value",
            "object-key",
            "key-low-then-high",
            "high-then-space",
            "embedded",
            "reversed-pair",
            "doubled-high",
            "high-then-valid-non-low",
            "well-formed-pair",
            "top-level-bare-string",
        ],
    )
    @pytest.mark.parametrize("lane", ["default", "skip_json_loads"])
    def test_tors_spellings_stay_internally_consistent(self, raw: str, lane: str) -> None:
        # The invariant a naive "preserve the escape text in the value"
        # fix would break: the loads spelling is exactly json.loads of the
        # str spelling, on every shape and lane.
        kwargs: dict[str, Any] = {} if lane == "default" else {"skip_json_loads": True}
        s = tors.repair_json(raw, **kwargs)
        if s == "":
            # Only the top-level bare-string repair lane collapses to the
            # "" sentinel (see test_top_level_bare_string_sentinel_faces):
            # there is no str spelling to round-trip, so the loads spelling
            # must be that same sentinel — asserted, not skipped.
            assert raw == '"\\ud800"'
            assert kwargs == {"skip_json_loads": True}
            assert tors.repair_json_loads(raw, **kwargs) == ""
        else:
            assert tors.repair_json_loads(raw, **kwargs) == json.loads(s)

    def test_top_level_bare_string_sentinel_faces(self) -> None:
        # The repair lane collapses a top-level lone-surrogate string to
        # the '' sentinel on BOTH engines (the same collapse as '""'), so
        # the top-level spelling is the one agreeing face of the class.
        assert json_repair_lib.repair_json('"\\ud800"', skip_json_loads=True) == ""
        assert tors.repair_json('"\\ud800"', skip_json_loads=True) == ""
        assert json_repair_lib.loads('"\\ud800"', skip_json_loads=True) == ""
        assert tors.repair_json_loads('"\\ud800"', skip_json_loads=True) == ""
        # The default lane takes the json.loads fast path instead, where
        # the api.md "Lone surrogates" arm of the strict decoder shows the same classified
        # split as the repair parser.
        assert json_repair_lib.repair_json('"\\ud800"') == '"\\ud800"'
        assert tors.repair_json('"\\ud800"') == '"\\ufffd"'


_base_text = st.text(
    # BMP-only (max_codepoint=0xFFFF: no astral chars, so json.dumps never
    # emits a surrogate-pair escape here), no raw surrogates (Cs+) and no
    # backslash — but json.dumps reintroduces `\uXXXX` escapes for non-ASCII
    # BMP chars (plus `\"` for quotes) and _mutate corrupts that dumped
    # text, so a delete landing inside an escape CAN yield a surrogate-range
    # half: ["A" + chr(0x0D80) + "1"] dumps as '["A\u0d801"]' and seed
    # 43064841 deletes the 0 to give '["A\ud801"]' (issue #57) — the api.md "Lone surrogates"
    # lone-surrogate class (tors U+FFFD vs oracle preservation), reachable,
    # not unreachable. The mutation differential therefore skips
    # surrogate-shaped inputs explicitly (see
    # test_differential_on_mutations) instead of relying on this alphabet,
    # which only keeps the RAW-value side surrogate-free. Escape handling
    # itself stays covered by the corpus's dedicated escaping
    # cases (both sides, pinned).
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0xFFFF,
        exclude_categories=("Cs",),
        exclude_characters="\\",
    ),
    max_size=12,
)
_key_text = st.text(
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0xFFFF,
        exclude_categories=("Cs",),
        exclude_characters="\\",
    ),
    max_size=6,
)
_ascii_text = st.text(alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E), max_size=12)
_ascii_key = st.text(alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E), max_size=6)


def _json_strategy(
    text: st.SearchStrategy[str], keys: st.SearchStrategy[str]
) -> st.SearchStrategy[Any]:
    leaf = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-(2**53), max_value=2**53),
        st.floats(allow_nan=False, allow_infinity=False),
        text,
    )
    return st.recursive(
        leaf,
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(keys, children, max_size=4),
        ),
        max_leaves=8,
    )


json_strategy = _json_strategy(_base_text, _key_text)
ascii_json_strategy = _json_strategy(_ascii_text, _ascii_key)

_INSERT_CHARS = [",", ":", "{", "}", "[", "]", "`", " ", "#"]
_QUOTE_CHARS = ["'", '"', "`"]
_QUOTEISH = ("'", '"', "`")


def _mutate(s: str, seed: int) -> str:
    """Apply one small deterministic damage step driven by ``seed``."""
    if not s:
        return s
    rng = random.Random(seed)
    op = rng.randrange(5)
    if op == 0:  # delete a char at a random index
        idx = rng.randrange(len(s))
        return s[:idx] + s[idx + 1 :]
    if op == 1:  # replace a random quote-ish char with one of '"`'
        spots = [i for i, c in enumerate(s) if c in _QUOTEISH]
        if not spots:
            return s
        idx = rng.choice(spots)
        return s[:idx] + rng.choice(_QUOTE_CHARS) + s[idx + 1 :]
    if op == 2:  # truncate at a random position
        idx = rng.randrange(len(s) + 1)
        return s[:idx]
    if op == 3:  # insert a structural/prose char at a random position
        idx = rng.randrange(len(s) + 1)
        return s[:idx] + rng.choice(_INSERT_CHARS) + s[idx:]
    # swap a random ',' with ':'
    spots = [i for i, c in enumerate(s) if c == ","]
    if not spots:
        return s
    idx = rng.choice(spots)
    return s[:idx] + ":" + s[idx + 1 :]


def _parses(s: str) -> bool:
    try:
        json.loads(s)
    except ValueError:
        return False
    return True


# The exhaustive structural sweep alphabet: every char that steers the
# lookahead memos (brackets, delimiters, escape runs) plus one ordinary
# filler. All strings up to _SWEEP_MAXLEN over this alphabet that contain a
# delimiter and a bracket exercise every memo-site interaction at
# exhaustively small sizes. Intentionally blind to the full `\X` escape set
# (not just `\uXXXX`): the alphabet has no escape-letter chars (u, n, t, r,
# b, f, v, 0-9, a-f, d, /), so no sweep raw forms `\u`/`\n`/`\t`-shaped
# escapes at all — surrogate or otherwise; that class (api.md "Lone
# surrogates") is pinned separately in TestLoneSurrogateEscapeDivergence,
# including the uppercase pair spelling.
_SWEEP_ALPHABET = ["[", "]", "{", "}", '"', "\\", "x"]
_SWEEP_MAXLEN = 6


def _sweep_raws() -> list[str]:
    raws: list[str] = []
    for length in range(1, _SWEEP_MAXLEN + 1):
        for tup in itertools.product(_SWEEP_ALPHABET, repeat=length):
            s = "".join(tup)
            if '"' in s and ("]" in s or "}" in s):
                raws.append(s)
    return raws


@pytest.mark.sweep
class TestExhaustiveStructuralSweep:
    """Every short string over the structural alphabet, both engines.

    This is the committed form of the exhaustive `]`/`}`/`\\`/`"` sweep the
    lookahead-memo fixes were verified with: sharing one memo key across the
    `}`/`]`/ObjectKey/comma-classify sites, dropping upstream's
    backslash-adjacent write guard, and caching pairing-walk outcomes are
    each only exact for anchored scan starts; and the anchored-start
    argument is over exactly these chars. A memo bug that flips one verdict
    diverges tors from the oracle on at least one of these raws.
    """

    @pytest.mark.parametrize("raw", _sweep_raws())
    def test_engine_lane_parity(self, raw: str) -> None:
        got = tors.repair_json(raw, skip_json_loads=True)
        want = json_repair_lib.repair_json(raw, skip_json_loads=True)
        assert got == want
        got_loads = tors.repair_json_loads(raw, skip_json_loads=True)
        want_loads = json_repair_lib.loads(raw, skip_json_loads=True)
        assert got_loads == want_loads


# The issue #39 escaped-key-splice grid: the three ingredients around the
# minimal repro — escaped object keys (enter the normalization-splice
# path), a bracket wrapper (give the ']' scan its context), and a bracket
# in the value (the cached target) — every combination.
_SPLICE_OPENERS = ["[", "{", "x [", "[["]
_SPLICE_VALUES = ["]w", "w]", "]]", "]x]", "w", "]", "}"]
_SPLICE_CLOSERS = ["]", "}", '"', "", ']"', '}"', '"}', '}']


def _splice_grid_raws() -> list[str]:
    return [
        opener + '{\\"\\":' + value + closer
        for opener, value, closer in itertools.product(
            _SPLICE_OPENERS, _SPLICE_VALUES, _SPLICE_CLOSERS
        )
    ]


@pytest.mark.sweep
class TestEscapedObjectSpliceGrid:
    """The escaped-key normalization splice against the oracle.

    The committed form of the issue #39 differential: the splice rewrites
    the char buffer in place, so a parser-level lookahead-memo entry that
    survives it (its positions are absolute) reads a stale offset and
    silently drops the bracket from the value. 27 of these 224 shapes
    diverged before the splice learned to clear the memo; any future
    memo/splice interaction regression diverges on at least one of them.
    Intentionally blind to the full `\\X` escape set like the structural
    sweep above (no escape-letter chars in the grid ingredients): the
    api.md "Lone surrogates" surrogate class is pinned separately in
    TestLoneSurrogateEscapeDivergence.
    """

    @pytest.mark.parametrize("raw", _splice_grid_raws())
    def test_engine_lane_parity(self, raw: str) -> None:
        got = tors.repair_json(raw, skip_json_loads=True)
        want = json_repair_lib.repair_json(raw, skip_json_loads=True)
        assert got == want
        got_loads = tors.repair_json_loads(raw, skip_json_loads=True)
        want_loads = json_repair_lib.loads(raw, skip_json_loads=True)
        assert got_loads == want_loads


# A `\uXXXX` escape in the surrogate range (U+D800..U+DFFF): the api.md "Lone surrogates"
# lone-surrogate class. json.dumps emits these only for raw surrogates
# (which _base_text excludes) — but _mutate corrupts the dumped text, so a
# delete landing inside a `\u0d8x`-shaped escape can CREATE one (seed
# 43064841 on ["A" + chr(0x0D80) + "1"]: '["A\u0d801"]' minus the 0 is
# '["A\ud801"]', issue #57). The mutation differential skips these inputs
# explicitly so the next hit is a classified skip, not a mystery flake.
# Anchored skip (H1): a bare `\\u[dD]...` search is over-broad — it fires on
# well-formed pairs (`\ud83d\ude00`, byte-identical on the str lane) and on
# escaped literals (`\\ud800`, backslash-escaped text, not an escape). So the
# skip counts preceding backslashes (odd = escaped literal, not an escape)
# and excludes a high half directly followed by an unescaped `\udc..`-`\udf..`
# low half (the non-divergent pair): only an unescaped LONE half skips.
_UNICODE_ESCAPE_RE = re.compile(r"\\u[0-9a-fA-F]{4}")


def _count_preceding_backslashes(s: str, pos: int) -> int:
    n = 0
    i = pos - 1
    while i >= 0 and s[i] == "\\":
        n += 1
        i -= 1
    return n


def _has_lone_surrogate_escape(s: str) -> bool:
    r"""True iff ``s`` holds an unescaped lone-surrogate `\uXXXX` escape.

    Well-formed high+low pairs and backslash-escaped literals return False.
    """
    matches = list(_UNICODE_ESCAPE_RE.finditer(s))
    # (start, end, codepoint) for unescaped escapes only.
    live: list[tuple[int, int, int]] = []
    for m in matches:
        if _count_preceding_backslashes(s, m.start()) % 2 == 1:
            continue  # escaped literal text, not an escape
        try:
            cp = int(m.group(0)[2:], 16)
        except ValueError:
            continue
        live.append((m.start(), m.end(), cp))
    for idx, (start, end, cp) in enumerate(live):
        is_high = 0xD800 <= cp <= 0xDBFF
        is_low = 0xDC00 <= cp <= 0xDFFF
        if not (is_high or is_low):
            continue
        if is_high:
            # Valid pair: directly followed by an unescaped low half.
            if idx + 1 < len(live) and live[idx + 1][0] == end:
                nxt = live[idx + 1][2]
                if 0xDC00 <= nxt <= 0xDFFF:
                    continue
            return True
        # Low half: part of a pair only when directly preceded by an
        # unescaped high half; otherwise lone.
        if idx > 0 and live[idx - 1][1] == start:
            prev = live[idx - 1][2]
            if 0xD800 <= prev <= 0xDBFF:
                continue
        return True
    return False


_SURROGATE_ESCAPE_RE = _UNICODE_ESCAPE_RE  # legacy alias: use _has_lone_surrogate_escape


class TestHypothesisInvariants:
    @given(json_strategy, st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=200, deadline=None)
    def test_never_raises_and_output_is_valid(self, value: Any, seed: int) -> None:
        mutated = _mutate(json.dumps(value), seed)
        out = tors.repair_json(mutated)  # default args: must never raise
        assert out == "" or _parses(out)

    @given(ascii_json_strategy)
    @settings(max_examples=200, deadline=None)
    def test_valid_input_serializer_parity(self, value: Any) -> None:
        # why byte-exact: json.dumps with default separators (", ", ": ") and
        # ensure_ascii=True already emits canonical JSON, and tors
        # re-serializes through its own canonical dumps, so valid canonical
        # input round-trips unchanged.
        # The top-level EMPTY string is excluded: its canonical form '""'
        # repairs to the bare '' sentinel: the nothing-recoverable
        # spelling: which is upstream's own ambiguity (json_repair's
        # `if parsed_json == "": return ""` shortcut returns '' for '""'
        # too), not a serializer defect.
        assume(value != "")
        s = json.dumps(value)
        assert tors.repair_json(s) == s

    @given(json_strategy, st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=200, deadline=None)
    def test_loads_matches_str_roundtrip(self, value: Any, seed: int) -> None:
        mutated = _mutate(json.dumps(value), seed)
        repaired = tors.repair_json(mutated)
        if repaired == "" or not _parses(repaired):
            return
        assert tors.repair_json_loads(mutated) == json.loads(repaired)

    @given(json_strategy, st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=100, deadline=None)
    def test_differential_on_mutations(self, value: Any, seed: int) -> None:
        # red-green protocol: a failure here is either a port bug (fix in the
        # Rust repair parser/serializer) or a newly-classified divergence
        # (move the shape to an explicit exclusion with a §9 comment, the way
        # fenced scalars are excluded below for the §9.4 split).
        mutated = _mutate(json.dumps(value), seed)
        assume("```json" not in mutated)  # §9.4 fenced-scalar divergence
        # api.md "Lone surrogates" divergence (api.md Divergences entry;
        # src/json_repair/string.rs + strict.rs; policy thread
        # https://github.com/AZX-PBC-OSS/tors/issues/57, CLOSED): tors maps
        # a surrogate-range half to U+FFFD while the oracle preserves it,
        # so any surrogate-shaped mutation is a classified skip, never a
        # port-bug signal. Hypothesis replays its example DB on every run:
        # a recorded hit (e.g. seed 43064841) replays into this assume as a
        # skip, not a mystery flake.
        assume(not _has_lone_surrogate_escape(mutated))
        assert tors.repair_json(mutated) == json_repair_lib.repair_json(mutated)


class TestAnchoredSurrogateSkip:
    """H1 pin: the mutation skip fires only on observed-divergent shapes."""

    @pytest.mark.parametrize(
        ("raw", "skips"),
        [
            ('["\\ud800"]', True),  # lone high: divergent, skip
            ('["\\udc00"]', True),  # lone low: divergent, skip
            ('["\\ud800\\u0041"]', True),  # high + valid non-low: lone, skip
            ('["\\ud800\\u41"]', True),  # high + truncated: lone, skip
            ('["\\ud83d\\ude00"]', False),  # valid pair: str-lane agrees, no skip
            ('["\\uD83D\\uDE00"]', False),  # valid pair uppercase: no skip
            ('["\\\\ud800"]', False),  # escaped literal: not an escape, no skip
            ('["plain"]', False),
        ],
    )
    def test_skip_is_anchored_not_text_shaped(self, raw: str, skips: bool) -> None:
        assert _has_lone_surrogate_escape(raw) is skips
