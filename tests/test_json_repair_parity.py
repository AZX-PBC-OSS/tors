"""Differential parity + fuzz-shaped invariants for the tors json_repair port.

Provenance: the behavior oracle is json_repair by Stefano Baccianella (MIT,
https://github.com/mangiucugna/json_repair) pinned to json-repair==0.63.4
(commit 251d141786d0f6ff561f6ec04d90188a338e2470). The pin is deliberate: a
version bump is a parity re-sync request, not a drive-by upgrade. Mapping:
``tors.repair_json(raw, **kw) == json_repair.repair_json(raw, **kw)`` (and
likewise for ``repair_json_loads``). This file owns the DESIGN §10
differential + hypothesis lanes: corpus equality over repair shapes, schema
and strict-mode differentials, and seed-free hypothesis invariants over
small deterministic mutations. Tors-native divergences (DESIGN §9) are never
oracle-compared here; each exclusion cites its §9 number.
"""

from __future__ import annotations

import itertools
import json
import random
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import tors

json_repair_lib = pytest.importorskip("json_repair")  # pin: json-repair==0.63.4

_BASE_RAWS: list[str] = [
    '{"a": 1 "b": 2}',  # missing comma: object members
    '[1 2 3]',  # missing comma: array items
    '{"a": 1, "b": 2 "c": 3}',  # missing comma: later member
    '{"a": 1, "b": 2',  # missing bracket: unclosed object
    '[1, 2, 3',  # missing bracket: unclosed array
    '{"a": "hello}',  # missing quote: unterminated string value
    '{"a: 1}',  # missing quote: unterminated key
    '{"a": {"b": [1, 2',  # truncated: nested containers
    '[{"a": 1}, {"b":',  # truncated: array of objects
    '"hello wo',  # truncated: top-level string
    "{'a': 1}",  # single quotes: object
    "['a', 'b']",  # single quotes: array
    "{'a': 'hi'}",  # single quotes: key and value
    '{a: 1}',  # unquoted key: single
    '{a: 1, b: 2}',  # unquoted keys: multiple
    '{name: "bob"}',  # unquoted key: string value
    '{"a": /*x*/ 1}',  # comment: inline block comment
    '// lead\n{"a": 1}',  # comment: leading line comment
    '/*x*/{"a": 1}',  # comment: leading block comment
    'Answer: {"a": 1}',  # prose prefix: object payload
    'Here: [1, 2]',  # prose prefix: array payload
    'Result: {"a": true}',  # prose prefix: boolean member
    '(1, 2, 3)',  # tuple: top-level parenthesized
    '{"a": (1, 2)}',  # tuple: parenthesized value
    '(1,)',  # tuple: single element
    '{"a": True}',  # Python literal: True
    '{"a": False}',  # Python literal: False
    '{"a": None}',  # Python literal: None
    '[True, False, None]',  # Python literals: mixed array
    '{"a": True, "b": None}',  # Python literals: mixed object
    '""hi""',  # doubled quotes: top-level string
    '{"a": ""hi""}',  # doubled quotes: member value
    '{"a": "x\\qy"}',  # escape: invalid \\q escape
    '{"a": "\\u41"}',  # escape: truncated \\u escape
    "```json\n" '{"a": 1}\n' "```",  # fenced payload: object (no top-level scalar)
    "```json\n" "[1, 2]\n" "```",  # fenced payload: array (no top-level scalar)
    "~~~json\n" '{"a": 1}\n' "~~~",  # fenced payload: tilde fence object
    "```\n" '{"a": 1}\n' "```",  # fenced payload: untagged fence object
    '{"a": {"b": 1 "c": 2}}',  # nested damage: missing comma deep
    '{"a": [1 2]}',  # nested damage: array missing comma
    "{'a': {'b': 1}}",  # nested damage: single quotes deep
    '{a: {b: 2}}',  # nested damage: unquoted keys deep
    '{"a": {"b": {',  # nested damage: truncated deep
    '[{"a": [1, {"b": 2',  # nested damage: truncated mixed stack
    '{"a": 1,}',  # trailing comma: object
    '[1, 2,]',  # trailing comma: array
    '{"a" 1}',  # missing colon after key
    '{"a": 1,, "b": 2}',  # doubled comma between members
    '',  # empty input: nothing-recoverable sentinel
    '   ',  # whitespace only: sentinel
    '\"\"',  # valid empty string: the sentinel spelling both sides share
    '{"a": }',  # missing value after colon
    '{"a": "b" "c": "d"}',  # missing comma: string values
    '[1, {"a": True}, None]',  # nested damage: Python literals in array
    '{"a": [1, 2}',  # mismatched closer: brace for bracket
    'not json at all',  # prose only: sentinel
    '{"a": "x", "b": }',  # missing value: second member
    "{a: 'x', b: None}",  # combined: unquoted + single + None
    '{"a":1}{"b":2}',  # concatenated objects
    '{' + r'{\"k\": 1}' * 64 + '}',  # escaped-delimiter run in a string
    # body: the escape normalizer's incremental undo record stays
    # byte-identical to the oracle here.
    r'{"bs": "\\\\", "m": "\\"k\\" \u201e x"}',  # backslash run +
    # delimiter unescape + smart quote in one body: exercises every
    # acc_pop repair arm against the oracle.
    '[' + r'{\"k\": \"v\"}' * 48 + ']',  # escaped key AND value run
    # in an array body: the heaviest escape-repair density, pinned.
    '{"n": {"d": {"x": True}}}',  # nested damage: deep Python literal
    'Answer is: [1, {"a": None}]',  # prose prefix: nested literal array
    '["' + "]" * 64 + '" x',  # array-context `]` run in a string body: the
    # memoized-lookahead O(n^2) fix stays byte-identical to the oracle here.
    r'''[{"a": "]}\\"x"}]''',  # mixed `]`/`}`/`\\`/`"` in an array-of-object
    # string body: exercises the shared `[outer]` memo across the `]` and `}`
    # sites and pins it byte-identical to the oracle.
    '["' + (']' + '\\\\') * 32 + '" x',  # interleaved `]`/even-backslash-run
    # string body: the incremental escape-tail rewrite (upstream rebuilds the
    # accumulator per normalization) stays byte-identical to the oracle.
    '["' + 'a"' * 64 + '"]',  # internal-quote run in an array string: the
    # pairing-walk outcome memo (upstream re-walks per quote candidate).
    '{"a": "' + '}' * 64 + '"' + 'y' * 64 + '"z',  # object-value `}` run with a
    # long quote-free gap: the `}`-branch's lstring lookahead memo.
    '{"a": "[' + 'x"' * 64 + '"}',  # quote run under an open regex character
    # class: the whitespace-flag + memoized `]` lookahead rewrite.
    '{' + 'a:b,' * 64 + '}',  # unquoted-key member run: the parser-level
    # lookahead memo shared across the run's many short string parses.
]

# NOTE (§9.4): no fenced TOP-LEVEL SCALAR lives in _BASE_RAWS — tors recovers
# them while the oracle returns "", so they can never be oracle-compared.
CORPUS: list[tuple[str, dict[str, Any]]] = [(raw, {}) for raw in _BASE_RAWS] + [
    (raw, {"skip_json_loads": True}) for raw in _BASE_RAWS
]

# Deliberately EXCLUDED from every schema differential below (§9.7 tors-native
# behavior changes, pinned tors-side elsewhere, never oracle-compared here):
# - typo keys remapped via the key-normalization/fuzzy ladder (§6.0-1) — §9.7
# - date/date-time format normalization schemas (§6.3) — §9.7
# - comma-split strings ("1, 2, 3" -> [1, 2, 3], oracle raises) (§6.1b) — §9.7
# - digit-group separator strings ("1,234" -> 1234, oracle raises) (§6.1c) — §9.7
# - string-enum near-misses (tors appends "Did you mean ...") (§6.2) — §9.7
# - percent / currency / prose single-number extraction ("50%" -> 0.5 or 50
#   by declared type, "USD 50" -> 50, oracle raises) (§6.1c tiers) — §9.7
# - separator-ambiguous numerics under Auto (assume-en-US schema-checked;
#   "1,234" on a number field -> 1234 + disclosure, oracle raises) — §9.7
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
        '{}',
        {'type': 'object', 'properties': {'a': {'type': 'string', 'default': 'hi'}}},
    ),
    # fill: required key already present passes through
    (
        '{"a": 1}',
        {'type': 'object', 'properties': {'a': {'type': 'integer'}}, 'required': ['a']},
    ),
    # coercion: array items "1"/"2" -> integers
    (
        '{"a": ["1", "2"]}',
        {
            'type': 'object',
            'properties': {'a': {'type': 'array', 'items': {'type': 'integer'}}},
        },
    ),
    # unwrap: JSON-string container "[1, 2]" -> array with coerced items
    (
        '{"a": "[1, 2]"}',
        {
            'type': 'object',
            'properties': {'a': {'type': 'array', 'items': {'type': 'integer'}}},
        },
    ),
    # union: oneOf picks the validating branch for "1"
    (
        '{"a": "1"}',
        {
            'type': 'object',
            'properties': {'a': {'oneOf': [{'type': 'integer'}, {'type': 'string'}]}},
        },
    ),
    # $ref: local "#/$defs/n" pointer resolves to integer
    (
        '{"a": 1}',
        {
            '$defs': {'n': {'type': 'integer'}},
            'type': 'object',
            'properties': {'a': {'$ref': '#/$defs/n'}},
        },
    ),
    # enum: integer member passes with no suggestion text involved
    ('{"a": 2}', {'type': 'object', 'properties': {'a': {'enum': [1, 2, 3]}}}),
    # boolean schema True allows anything
    ('true', True),  # type: ignore[list-item]
]

SCHEMA_VALUE_RAISE_CORPUS: list[tuple[str, dict[str, Any]]] = [
    # pattern: "abc" violates "^\\d+$" — both raise ValueError; the validation
    # message WORDING differs (§9.5 jsonschema-crate texts), so only the raise
    # itself is compared, never the message.
    (
        '{"a": "abc"}',
        {'type': 'object', 'properties': {'a': {'type': 'string', 'pattern': '^\\d+$'}}},
    ),
]

_DEEP_DEPTH = 500
_DEEP_RAW = '{"x": ' * _DEEP_DEPTH + '1' + '}' * _DEEP_DEPTH


def _deep_schema(depth: int) -> dict[str, Any]:
    schema: dict[str, Any] = {'type': 'integer'}
    for _ in range(depth):
        schema = {'type': 'object', 'properties': {'x': schema}}
    return schema


DEEP_SCHEMA: dict[str, Any] = _deep_schema(_DEEP_DEPTH)

STRICT_CORPUS: list[str] = [
    # The REAL upstream strict corpus (test_strict_mode.py): strict mode
    # rejects structural ambiguity, not ordinary repairable damage —
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
        ('raw', 'kwargs'), CORPUS, ids=[f'corpus-{i}' for i in range(len(CORPUS))]
    )
    def test_str_parity(self, raw: str, kwargs: dict[str, Any]) -> None:
        got = tors.repair_json(raw, **kwargs)
        want = json_repair_lib.repair_json(raw, **kwargs)
        assert got == want

    @pytest.mark.parametrize(
        ('raw', 'kwargs'), CORPUS, ids=[f'corpus-{i}' for i in range(len(CORPUS))]
    )
    def test_loads_parity(self, raw: str, kwargs: dict[str, Any]) -> None:
        got = tors.repair_json_loads(raw, **kwargs)
        want = json_repair_lib.loads(raw, **kwargs)
        assert got == want

    @pytest.mark.parametrize(
        ('raw', 'schema'),
        SCHEMA_CORPUS,
        ids=[f'schema-{i}' for i in range(len(SCHEMA_CORPUS))],
    )
    def test_schema_parity(self, raw: str, schema: dict[str, Any] | bool) -> None:
        pytest.importorskip('jsonschema')
        got = tors.repair_json(raw, schema=schema)
        want = json_repair_lib.repair_json(raw, schema=schema)
        assert got == want
        got_loads = tors.repair_json_loads(raw, schema=schema)
        want_loads = json_repair_lib.loads(raw, schema=schema)
        assert got_loads == want_loads

    @pytest.mark.parametrize(
        ('raw', 'schema'),
        SCHEMA_VALUE_RAISE_CORPUS,
        ids=[f'schema-raise-{i}' for i in range(len(SCHEMA_VALUE_RAISE_CORPUS))],
    )
    def test_schema_raise_parity(self, raw: str, schema: dict[str, Any]) -> None:
        pytest.importorskip('jsonschema')
        with pytest.raises(ValueError):
            tors.repair_json(raw, schema=schema)
        with pytest.raises(ValueError):
            json_repair_lib.repair_json(raw, schema=schema)

    def test_pattern_properties_extras_keep_input_order(self) -> None:
        # Extras emit at their input positions (upstream's value.items()
        # pass) — pattern folds inline with kept extras, never regrouped.
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
        # limit — both raise, types differ, so only the both-raise shape is
        # compared, never messages.
        pytest.importorskip('jsonschema')
        with pytest.raises(ValueError):
            tors.repair_json(_DEEP_RAW, schema=DEEP_SCHEMA)
        with pytest.raises((ValueError, RecursionError)):
            json_repair_lib.repair_json(_DEEP_RAW, schema=DEEP_SCHEMA)

    def test_continuation_chain_recursion_both_raise(self) -> None:
        # §9.6 both-raise pin for the array-continuation merge chain
        # (`{"a":[0],` + `["b":[0],` * N): tors caps it at MAX_NESTING
        # fragments and the oracle at its own recursion limit, so at a size
        # far past both thresholds each engine raises a ValueError — the
        # differential signal that neither crashes. (A tors build without
        # the continuation guard dies with SIGSEGV here, which kills the
        # process rather than failing the assertion — the native suite's
        # sub-2k sizes fail cleanly instead.)
        merge_chain = '{"a":[0],' + '["b":[0],' * 2_000 + '1]'
        with pytest.raises(ValueError):
            tors.repair_json(merge_chain, skip_json_loads=True)
        with pytest.raises(ValueError):
            json_repair_lib.repair_json(merge_chain, skip_json_loads=True)

    @pytest.mark.parametrize(
        'raw', STRICT_CORPUS, ids=[f'strict-{i}' for i in range(len(STRICT_CORPUS))]
    )
    def test_error_parity(self, raw: str) -> None:
        # skip_json_loads=True matches upstream's own strict-test spellings:
        # duplicate-keys-in-array and empty-key inputs are VALID json.loads
        # input (last-wins duplicate handling; an empty key is legal JSON),
        # so the strict raise only fires from the repair parser — the fast
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


_base_text = st.text(
    # BMP-only, no backslash: astral chars dump as surrogate-PAIR escapes
    # and a mutation landing inside one leaves a LONE surrogate escape —
    # the documented §9.2 divergence (Rust str cannot hold lone surrogates;
    # upstream preserves and re-emits them), unreachable by design here so
    # the mutation differential stays an honest port-bug detector. Escape
    # handling itself stays covered by the corpus's dedicated escaping
    # cases (both sides, pinned).
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0xFFFF,
        exclude_categories=('Cs',),
        exclude_characters='\\',
    ),
    max_size=12,
)
_key_text = st.text(
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0xFFFF,
        exclude_categories=('Cs',),
        exclude_characters='\\',
    ),
    max_size=6,
)
_ascii_text = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E), max_size=12
)
_ascii_key = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E), max_size=6
)


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

_INSERT_CHARS = [',', ':', '{', '}', '[', ']', '`', ' ', '#']
_QUOTE_CHARS = ["'", '"', '`']
_QUOTEISH = ("'", '"', '`')


def _mutate(s: str, seed: int) -> str:
    """Apply ONE small deterministic damage step driven by ``seed``."""
    if not s:
        return s
    rng = random.Random(seed)
    op = rng.randrange(5)
    if op == 0:  # delete a char at a random index
        idx = rng.randrange(len(s))
        return s[:idx] + s[idx + 1:]
    if op == 1:  # replace a random quote-ish char with one of '"`'
        spots = [i for i, c in enumerate(s) if c in _QUOTEISH]
        if not spots:
            return s
        idx = rng.choice(spots)
        return s[:idx] + rng.choice(_QUOTE_CHARS) + s[idx + 1:]
    if op == 2:  # truncate at a random position
        idx = rng.randrange(len(s) + 1)
        return s[:idx]
    if op == 3:  # insert a structural/prose char at a random position
        idx = rng.randrange(len(s) + 1)
        return s[:idx] + rng.choice(_INSERT_CHARS) + s[idx:]
    # swap a random ',' with ':'
    spots = [i for i, c in enumerate(s) if c == ',']
    if not spots:
        return s
    idx = rng.choice(spots)
    return s[:idx] + ':' + s[idx + 1:]


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
# exhaustively small sizes.
_SWEEP_ALPHABET = ['[', ']', '{', '}', '"', '\\', 'x']
_SWEEP_MAXLEN = 6


def _sweep_raws() -> list[str]:
    raws: list[str] = []
    for length in range(1, _SWEEP_MAXLEN + 1):
        for tup in itertools.product(_SWEEP_ALPHABET, repeat=length):
            s = ''.join(tup)
            if '"' in s and (']' in s or '}' in s):
                raws.append(s)
    return raws


class TestExhaustiveStructuralSweep:
    """Every short string over the structural alphabet, both engines.

    This is the committed form of the exhaustive `]`/`}`/`\\`/`"` sweep the
    lookahead-memo fixes were verified with: sharing one memo key across the
    `}`/`]`/ObjectKey/comma-classify sites, dropping upstream's
    backslash-adjacent write guard, and caching pairing-walk outcomes are
    each only exact for ANCHORED scan starts — and the anchored-start
    argument is over exactly these chars. A memo bug that flips one verdict
    diverges tors from the oracle on at least one of these raws.
    """

    @pytest.mark.parametrize('raw', _sweep_raws())
    def test_engine_lane_parity(self, raw: str) -> None:
        got = tors.repair_json(raw, skip_json_loads=True)
        want = json_repair_lib.repair_json(raw, skip_json_loads=True)
        assert got == want
        got_loads = tors.repair_json_loads(raw, skip_json_loads=True)
        want_loads = json_repair_lib.loads(raw, skip_json_loads=True)
        assert got_loads == want_loads


class TestHypothesisInvariants:
    @given(json_strategy, st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=200, deadline=None)
    def test_never_raises_and_output_is_valid(self, value: Any, seed: int) -> None:
        mutated = _mutate(json.dumps(value), seed)
        out = tors.repair_json(mutated)  # default args: must never raise
        assert out == '' or _parses(out)

    @given(ascii_json_strategy)
    @settings(max_examples=200, deadline=None)
    def test_valid_input_serializer_parity(self, value: Any) -> None:
        # WHY byte-exact: json.dumps with default separators (", ", ": ") and
        # ensure_ascii=True already emits canonical JSON, and tors
        # re-serializes through its own canonical dumps, so valid canonical
        # input round-trips unchanged.
        # The top-level EMPTY STRING is excluded: its canonical form '""'
        # repairs to the bare '' sentinel — the nothing-recoverable
        # spelling — which is upstream's own ambiguity (json_repair's
        # `if parsed_json == "": return ""` shortcut returns '' for '""'
        # too), not a serializer defect.
        assume(value != '')
        s = json.dumps(value)
        assert tors.repair_json(s) == s

    @given(json_strategy, st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=200, deadline=None)
    def test_loads_matches_str_roundtrip(self, value: Any, seed: int) -> None:
        mutated = _mutate(json.dumps(value), seed)
        repaired = tors.repair_json(mutated)
        if repaired == '' or not _parses(repaired):
            return
        assert tors.repair_json_loads(mutated) == json.loads(repaired)

    @given(json_strategy, st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=100, deadline=None)
    def test_differential_on_mutations(self, value: Any, seed: int) -> None:
        # RED-GREEN PROTOCOL: a failure here is either a port bug (fix in the
        # Rust repair parser/serializer) or a newly-classified divergence
        # (move the shape to an explicit exclusion with a §9 comment, the way
        # fenced scalars are excluded below for the §9.4 split).
        mutated = _mutate(json.dumps(value), seed)
        assume('```json' not in mutated)  # §9.4 fenced-scalar divergence
        assert tors.repair_json(mutated) == json_repair_lib.repair_json(mutated)
