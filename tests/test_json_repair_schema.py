"""Schema-guided repair corpus ported from json_repair to tors.

Provenance: json_repair by Stefano Baccianella (MIT),
https://github.com/mangiucugna/json_repair, commit
251d141786d0f6ff561f6ec04d90188a338e2470 (= 0.63.4).
Upstream source: json_repair's tests/test_schema_guided_parse.py
at commit 251d141.
Contract: design-json-repair-port.md sections 4, 6, 8, 9.

Mapping: ``repair_json(raw, schema=s, skip_json_loads=True, return_objects=True)``
is ``tors.repair_json_loads(raw, schema=s, skip_json_loads=True)``;
``schema_repair_mode="salvage"`` is ``salvage=True`` (default is standard).
``logging=`` variants, pydantic-model schemas, and monkeypatched internals are
never ported (design section 9).
"""

from __future__ import annotations

from typing import Any

import pytest

from tors import repair_json, repair_json_diagnostics, repair_json_loads

# The beyond-u64 integer spelling the refusal tests share (10**25: far
# past u64::MAX, and lossy in f64 — 1e25's neighbors step by 2**20).
_BIG = 10**25


class TestSchemaStandard:
    def test_missing_value_type_defaults(self) -> None:
        # Upstream: test_schema_guides_missing_value_type_defaults.
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "flag": {"type": "boolean"},
                "items": {"type": "array", "items": {"type": "string"}},
                "payload": {"type": "object"},
                "nothing": {"type": "null"},
            },
            "required": ["text", "count", "ratio", "flag", "items", "payload", "nothing"],
        }
        raw = '{ "text": , "count": , "ratio": , "flag": , "items": , "payload": , "nothing": }'
        assert repair_json_loads(raw, schema=schema, skip_json_loads=True) == {
            "text": "",
            "count": 0,
            "ratio": 0,
            "flag": False,
            "items": [],
            "payload": {},
            "nothing": None,
        }

    def test_missing_required_property_raises(self) -> None:
        # Upstream: test_schema_missing_required_property_raises.
        schema = {
            "type": "object",
            "properties": {"required_value": {"type": "integer", "default": 1}},
            "required": ["required_value"],
        }
        with pytest.raises(ValueError, match="Missing required properties"):
            repair_json_loads("{}", schema=schema, skip_json_loads=True)

    def test_optional_default_is_inserted(self) -> None:
        # Upstream: test_schema_optional_default_is_inserted.
        schema = {"type": "object", "properties": {"note": {"type": "string", "default": "n/a"}}}
        assert repair_json_loads("{}", schema=schema, skip_json_loads=True) == {"note": "n/a"}

    def test_applies_to_valid_json_coerces_scalars(self) -> None:
        # Upstream: test_schema_applies_to_valid_json_without_skip_json_loads.
        schema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
        assert repair_json_loads('{"value": "1"}', schema=schema) == {"value": 1}
        assert repair_json_loads('"1"', schema={"type": "integer"}) == 1
        assert repair_json_loads("true", schema={"type": "string"}) == ""

    def test_pattern_mismatch_raises(self) -> None:
        # Upstream: test_schema_applies_to_valid_json_without_skip_json_loads.
        # Tors enforces pattern via its validator; loose match may need tightening at red-green.
        with pytest.raises(ValueError, match="pattern|does not match"):
            repair_json_loads('"bbb"', schema={"type": "string", "pattern": "^a+$"})

    def test_unwraps_double_serialized_object_standard(self) -> None:
        # Upstream: test_schema_unwraps_double_serialized_object_in_all_modes.
        schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string"},
                        "confidence": {"type": "string"},
                    },
                    "required": ["verdict", "confidence"],
                }
            },
            "required": ["summary"],
        }
        raw = '{"summary": "{\\"verdict\\": \\"malicious\\", \\"confidence\\": \\"high\\"}"}'
        assert repair_json_loads(raw, schema=schema, salvage=False) == {
            "summary": {"verdict": "malicious", "confidence": "high"}
        }

    def test_malformed_double_serialized_object_standard_raises(self) -> None:
        # Upstream: test_schema_salvage_repairs_malformed_double_serialized_object_string.
        schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string"},
                        "confidence": {"type": "string"},
                    },
                    "required": ["verdict", "confidence"],
                }
            },
            "required": ["summary"],
        }
        raw = '{"summary": "{verdict: malicious, confidence: high}"}'
        with pytest.raises(ValueError, match=r"Expected object at \$\.summary, got str\."):
            repair_json_loads(raw, schema=schema, salvage=False)

    def test_unwraps_double_serialized_array_standard(self) -> None:
        # Upstream: test_schema_unwraps_double_serialized_array_in_all_modes.
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"],
        }
        assert repair_json_loads('{"items": "[\\"a\\", \\"b\\"]"}', schema=schema) == {
            "items": ["a", "b"]
        }

    def test_malformed_double_serialized_array_standard_fallback(self) -> None:
        # Upstream: test_schema_salvage_repairs_malformed_double_serialized_array_string.
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"],
        }
        assert repair_json_loads('{"items": "[a, b]"}', schema=schema, salvage=False) == {
            "items": ["[a, b]"]
        }

    def test_object_string_unwrap_failures_preserved(self) -> None:
        # Upstream: test_schema_object_string_unwrap_preserves_existing_failures.
        schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "object",
                    "properties": {"verdict": {"type": "string"}},
                    "required": ["verdict"],
                }
            },
            "required": ["summary"],
        }
        with pytest.raises(ValueError, match=r"Expected object at \$\.summary, got str\."):
            repair_json_loads('{"summary": "not json"}', schema=schema)
        with pytest.raises(ValueError, match=r"Expected object at \$\.summary, got str\."):
            repair_json_loads('{"summary": "[1, 2]"}', schema=schema)

    def test_array_string_unwrap_fallbacks(self) -> None:
        # Upstream: test_schema_array_string_unwrap_preserves_existing_fallbacks.
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"],
        }
        assert repair_json_loads('{"items": "not json"}', schema=schema) == {"items": ["not json"]}
        assert repair_json_loads('{"items": "{\\"a\\": 1}"}', schema=schema) == {
            "items": ['{"a": 1}']
        }
        assert repair_json_loads('{"items": "{a: 1}"}', schema=schema, salvage=True) == {
            "items": ["{a: 1}"]
        }

    def test_prefixed_valid_json_preserved(self) -> None:
        # Upstream: test_schema_preserves_prefixed_valid_json_string_content.
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "id": {"type": "integer"},
                        },
                        "required": ["text", "id"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
        }
        raw = 'Preamble\n{"items": [{"text": "a\\n, extra: 1", "id": 8}]}'
        assert repair_json_loads(raw, schema=schema) == {
            "items": [{"text": "a\n, extra: 1", "id": 8}]
        }

    def test_empty_string_schema(self) -> None:
        # Upstream: test_schema_applies_to_valid_empty_string.
        assert repair_json_loads('""', schema={"type": "string"}) == ""

    def test_skip_json_loads_scalar_paths(self) -> None:
        # Upstream: test_schema_skip_json_loads_keeps_parser_path_for_scalars.
        assert repair_json_loads("True", schema={"type": "string"}, skip_json_loads=True) == ""
        with pytest.raises(ValueError, match="is not of type"):
            repair_json_loads('"1"', schema={"type": "integer"}, skip_json_loads=True)

    def test_defs_anyof_union_success_and_null_raise(self) -> None:
        # Upstream: test_schema_union_branch_keeps_root_defs_scope_during_repair.
        schema = {
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "pattern": "^example$"}},
                    "required": ["name"],
                }
            },
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "array", "items": {"$ref": "#/$defs/Item"}},
                        {"type": "null"},
                    ]
                }
            },
            "required": ["value"],
        }
        assert repair_json_loads('{"value": [{"name": "example"}],}', schema=schema) == {
            "value": [{"name": "example"}]
        }
        with pytest.raises(ValueError, match="Expected null"):
            repair_json_loads('{"value": [{"name": "invalid"}],}', schema=schema)

    def test_type_union_maxlength_applies_per_branch(self) -> None:
        # A type-union schema validates each branch with ITS OWN compiled
        # validator: the branches are synthesized per iteration as stack
        # locals whose addresses a pointer-keyed cache reused, so the
        # integer branch here was validated against the string branch's
        # maxLength validator and a valid ``42`` raised ``42 is not of
        # type "string"``. maxLength does not apply to integers: the
        # integer branch passes, and the string branch's own invalid
        # values still raise the ordinary ValueError.
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": ["string", "integer"], "maxLength": 1},
                "b": {"type": "integer"},
            },
        }
        assert repair_json_loads('{"a": 42, "b": "7"}', schema=schema) == {
            "a": 42,
            "b": 7,
        }
        with pytest.raises(ValueError):
            repair_json_loads('{"a": "toolong"}', schema=schema)

    def test_circular_ref_raises(self) -> None:
        # Upstream: test_schema_circular_ref_raises_definition_error.
        schema = {"$ref": "#/definitions/a", "definitions": {"a": {"$ref": "#/definitions/a"}}}
        with pytest.raises(ValueError, match=r"Circular \$ref detected"):
            repair_json_loads("{}", schema=schema)

    def test_non_string_ref_raises(self) -> None:
        # Upstream: test_schema_non_string_ref_raises_definition_error.
        with pytest.raises(ValueError, match=r"\$ref must be a string"):
            repair_json_loads("{}", schema={"$ref": 123})

    def test_unresolvable_ref_raises(self) -> None:
        # No direct upstream fn; pinned by design section 8 catalog entry.
        schema = {"$ref": "#/definitions/missing"}
        with pytest.raises(ValueError, match=r"Unresolvable \$ref"):
            repair_json_loads("{}", schema=schema)

    def test_deep_allof_raises(self) -> None:
        # Upstream: test_deep_allof_schema_raises_value_error_instead_of_recursion_error.
        schema: dict = {"type": "object", "properties": {"value": {"type": "string"}}}
        for _ in range(550):
            schema = {"allOf": [schema]}
        with pytest.raises(ValueError, match="schema recursion depth"):
            repair_json_loads('{"value": "ok"}', schema=schema)

    def test_deep_properties_raises(self) -> None:
        # Upstream: test_deep_properties_schema_raises_value_error_instead_of_recursion_error.
        schema2: dict = {"type": "string"}
        for depth in range(550):
            schema2 = {"type": "object", "properties": {f"level_{depth}": schema2}}
        with pytest.raises(ValueError, match="schema recursion depth"):
            repair_json_loads("{}", schema=schema2)


class TestSchemaSalvage:
    def test_selects_first_matching_fragment(self) -> None:
        # Upstream: test_schema_salvage_selects_first_matching_top_level_fragment.
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        raw = 'Here is an example: {"foo": 1}\n\nFinal answer:\n```json\n{"name": "Alice", "a'
        raw += 'ge": 30}\n```\n\nAlternative: {"name": "Bob", "age": 40}'
        assert repair_json_loads(raw, schema=schema, skip_json_loads=True, salvage=True) == {
            "name": "Alice",
            "age": 30,
        }

    def test_skips_list_fragment(self) -> None:
        # Upstream: test_schema_salvage_skips_list_fragment_before_matching_object.
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        raw = 'The options are ["a", "b"].\n\n{"name": "Alice", "age": 30}'
        assert repair_json_loads(raw, schema=schema, skip_json_loads=True, salvage=True) == {
            "name": "Alice",
            "age": 30,
        }

    def test_standard_still_rejects_first_fragment(self) -> None:
        # Upstream: test_schema_standard_still_rejects_invalid_first_top_level_fragment.
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        with pytest.raises(ValueError, match="Missing required properties"):
            repair_json_loads('{"foo": 1}\n{"name": "Alice", "age": 30}', schema=schema)

    def test_no_match_raises(self) -> None:
        # Upstream: test_schema_salvage_raises_when_no_top_level_fragment_matches.
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        with pytest.raises(ValueError, match="Missing required properties"):
            repair_json_loads('{"foo": 1}\n{"bar": 2}', schema=schema, salvage=True)
        with pytest.raises(ValueError, match="Missing required properties"):
            repair_json_loads('{"foo": 1} trailing prose', schema=schema, salvage=True)
        with pytest.raises(ValueError, match="Missing required properties"):
            repair_json_loads('{"foo": 1} // trailing comment', schema=schema, salvage=True)
        with pytest.raises(ValueError, match="is not of type"):
            repair_json_loads("", schema=schema, skip_json_loads=True, salvage=True)

    def test_real_array_not_item_selected(self) -> None:
        # Upstream: test_schema_salvage_does_not_select_an_item_from_a_real_top_level_array.
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        with pytest.raises(ValueError, match="Expected object"):
            repair_json_loads(
                '[{"foo": 1}, {"name": "Alice", "age": 30}]',
                schema=schema,
                skip_json_loads=True,
                salvage=True,
            )

    def test_drops_invalid_array_items(self) -> None:
        # Upstream: test_schema_salvage_mode_drops_invalid_array_items.
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "score": {"type": "number"},
                        },
                        "required": ["id", "score"],
                    },
                }
            },
            "required": ["items"],
        }
        raw = '{"items":[{"id":1,"score":85.6},{"id":2,"score":"N/A"}]}'
        with pytest.raises(ValueError, match="Expected number"):
            repair_json_loads(raw, schema=schema, skip_json_loads=True, salvage=False)
        assert repair_json_loads(raw, schema=schema, skip_json_loads=True, salvage=True) == {
            "items": [{"id": 1, "score": 85.6}]
        }

    def test_min_items_enforced(self) -> None:
        # Upstream: test_schema_salvage_mode_still_enforces_min_items.
        schema = {"type": "array", "items": {"type": "integer"}, "minItems": 2}
        with pytest.raises(ValueError, match="minItems"):
            repair_json_loads('["1", "bad"]', schema=schema, skip_json_loads=True, salvage=True)

    def test_bogus_type_raises(self) -> None:
        # Upstream: test_schema_salvage_mode_does_not_hide_schema_definition_errors.
        # salvage mode: a broken schema is the caller's bug, never
        # salvageable data: the drop sites re-raise it.
        with pytest.raises(ValueError, match="Unsupported schema type bogus"):
            repair_json_loads(
                "[1]",
                schema={"type": "array", "items": {"type": "bogus"}},
                salvage=True,
                skip_json_loads=True,
            )

    def test_maps_list_to_object(self) -> None:
        # Upstream: test_schema_salvage_mode_maps_list_to_object_when_unambiguous.
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "tags"],
        }
        with pytest.raises(ValueError, match="Expected object"):
            repair_json_loads('["hello", ["a", "b"]]', schema=schema, skip_json_loads=True)
        assert repair_json_loads(
            '["hello", ["a", "b"]]', schema=schema, skip_json_loads=True, salvage=True
        ) == {"name": "hello", "tags": ["a", "b"]}

    def test_mapping_rejects_length_mismatch(self) -> None:
        # Upstream: test_schema_salvage_mode_mapping_rejects_length_mismatch.
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "tags"],
        }
        with pytest.raises(ValueError, match="Expected object"):
            repair_json_loads('["hello"]', schema=schema, skip_json_loads=True, salvage=True)

    def test_mapping_rejects_type_mismatch(self) -> None:
        # Upstream: test_schema_salvage_mode_mapping_rejects_type_mismatch.
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "tags"],
        }
        with pytest.raises(ValueError, match="Expected object"):
            repair_json_loads('[["a", "b"], "hello"]', schema=schema, salvage=True)

    def test_set_like_members_to_null(self) -> None:
        # Upstream: test_schema_salvage_mode_maps_set_like_object_members_to_null_valued_keys.
        schema = {"type": "object"}
        with pytest.raises(ValueError, match="Expected object"):
            repair_json_loads('{"a", "b"}', schema=schema, skip_json_loads=True, salvage=False)
        assert repair_json_loads(
            '{"a", "b"}', schema=schema, skip_json_loads=True, salvage=True
        ) == {
            "a": None,
            "b": None,
        }
        assert repair_json('{"a", "b"}', schema=schema, skip_json_loads=True, salvage=True) == (
            '{"a": null, "b": null}'
        )

    def test_mixed_object_array_keeps_array(self) -> None:
        # Upstream: test_schema_salvage_mode_set_like_members_do_not_override_mixed_object_array.
        schema = {"type": ["object", "array"], "items": {"type": "string"}}
        assert repair_json_loads(
            '{"a", "b"}', schema=schema, skip_json_loads=True, salvage=True
        ) == [
            "a",
            "b",
        ]
        assert repair_json('{"a", "b"}', schema=schema, skip_json_loads=True, salvage=True) == (
            '["a", "b"]'
        )

    def test_boolean_schema_required_raises(self) -> None:
        # Upstream: test_schema_salvage_mode_missing_required_boolean_schema_still_raises.
        schema = {"type": "object", "properties": {"payload": True}, "required": ["payload"]}
        with pytest.raises(ValueError, match="Missing required properties"):
            repair_json_loads("{}", schema=schema, salvage=True)

    def test_root_unwrap_and_required_array_fill(self) -> None:
        # Upstream: test_schema_salvage_mode_unwraps_root_single_item_
        # array_and_fills_required_array.
        schema = {
            "type": "object",
            "properties": {
                "type": {"const": "food_sport_card"},
                "content": {
                    "type": "object",
                    "required": ["food", "sports"],
                    "properties": {
                        "food": {"type": "array", "items": {"type": "string"}},
                        "sports": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "required": ["type", "content"],
        }
        raw = '[{"type": "food_sport_card", "content": {"food": ["mantou"]}}]'
        with pytest.raises(ValueError, match=r"Expected object at \$, got list\."):
            repair_json_loads(raw, schema=schema, salvage=False)
        assert repair_json_loads(raw, schema=schema, salvage=True) == {
            "type": "food_sport_card",
            "content": {"food": ["mantou"], "sports": []},
        }

    def test_fills_required_with_safe_inference(self) -> None:
        # Upstream: test_schema_salvage_mode_fills_required_with_safe_inference_sources.
        schema = {
            "type": "object",
            "properties": {
                "from_default": {"default": "x"},
                "from_const": {"const": 7},
                "from_enum": {"enum": ["first", "second"]},
                "from_array_shape": {"items": {"type": "integer"}},
                "from_object_shape": {"properties": {"nested": {"type": "string"}}},
            },
            "required": [
                "from_default",
                "from_const",
                "from_enum",
                "from_array_shape",
                "from_object_shape",
            ],
        }
        assert repair_json_loads("{}", schema=schema, salvage=True) == {
            "from_default": "x",
            "from_const": 7,
            "from_enum": "first",
            "from_array_shape": [],
            "from_object_shape": {},
        }

    def test_missing_required_without_property_schema_raises(self) -> None:
        # Upstream: test_schema_salvage_mode_missing_required_without_property_schema_still_raises.
        schema = {"type": "object", "properties": {}, "required": ["missing"]}
        with pytest.raises(ValueError, match="Missing required properties"):
            repair_json_loads("{}", schema=schema, salvage=True)

    def test_missing_required_scalar_raises(self) -> None:
        # Upstream: test_schema_salvage_mode_missing_required_scalar_still_raises.
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        with pytest.raises(ValueError, match="Missing required properties"):
            repair_json_loads("[{}]", schema=schema, salvage=True)


class TestRefEscapesAndUnions:
    """$ref pointer escapes, union degenerate forms, tuple validation."""

    def test_ref_pointer_escapes_unescape_in_order(self) -> None:
        # `~1` (slash) before `~0` (tilde): upstream's unescape order. The
        # defs carry literal "a/b" and "a~b" keys.
        schema: dict[str, Any] = {
            "$defs": {"a/b": {"type": "integer"}, "a~b": {"type": "string"}},
            "type": "object",
            "properties": {
                "n": {"$ref": "#/$defs/a~1b"},
                "s": {"$ref": "#/$defs/a~0b"},
            },
            "required": ["n", "s"],
        }
        raw = '{"n": "42", "s": 5}'
        assert repair_json_loads(raw, schema=schema) == {"n": 42, "s": "5"}

    def test_empty_one_of_and_type_list_both_raise(self) -> None:
        # Degenerate unions: both validators reject the schema itself at
        # the API level (upstream: Python jsonschema's wording; tors: the
        # Rust crate's): the §8 validation-boundary divergence class, so
        # the pin is the shared raise, not the message. ("No schema
        # matched the value" is repair_value's internal spelling, reached
        # only when a validator lets the schema through.)
        for schema in ({"oneOf": []}, {"type": []}):
            with pytest.raises(ValueError):
                repair_json_loads("1", schema=schema, skip_json_loads=True)

    def test_allof_enum_chain_folds(self) -> None:
        # allOf members apply conjunctively: the enum rides the type.
        schema: dict[str, Any] = {"allOf": [{"type": "string"}, {"enum": ["red", "green"]}]}
        assert repair_json_loads('"red"', schema=schema) == "red"
        with pytest.raises(ValueError):
            repair_json_loads("5", schema=schema)

    def test_draft07_tuple_items_with_additional_items(self) -> None:
        # Draft-07 tuple form: fixed-position schemas plus additionalItems
        # for the tail; validation enforces both (additionalItems: false
        # drops the tail in the repair parser, pinned natively).
        schema: dict[str, Any] = {
            "type": "array",
            "items": [{"type": "integer"}, {"type": "string"}],
            "additionalItems": {"type": "boolean"},
        }
        raw = '[1, "x", true, false]'
        assert repair_json_loads(raw, schema=schema, skip_json_loads=True) == [
            1,
            "x",
            True,
            False,
        ]
        with pytest.raises(ValueError):
            repair_json_loads('["no", "x"]', schema=schema, skip_json_loads=True)


class TestConstrainedBigintRefusal:
    """Issue #119: a constrained schema position carrying an integer beyond
    u64 used to compile into the validator as f64, whose neighbors alias —
    ``enum: [10**25, 10**25 + 10**10]`` accepted ``10**25 + 5`` and
    ``minimum: 10**25`` accepted ``10**25 - 1`` (silent wrong-accept, the
    sharp direction). The fail-closed contract now: the schema refuses with
    a catchable ``ValueError`` naming the position's JSON pointer, at any
    depth, ``allOf``/``$ref``-reachable positions included. Honest integers
    (the exact i64/u64 span), floats, and the document's own huge integers
    are untouched.
    """

    POINTER_CASES = [
        ("enum", {"enum": [_BIG, 3]}, "/enum/0"),
        ("const", {"const": _BIG}, "/const"),
        ("minimum", {"minimum": _BIG}, "/minimum"),
        ("maximum", {"maximum": _BIG}, "/maximum"),
        ("exclusiveMinimum", {"exclusiveMinimum": _BIG}, "/exclusiveMinimum"),
        ("exclusiveMaximum", {"exclusiveMaximum": _BIG}, "/exclusiveMaximum"),
        ("multipleOf", {"multipleOf": _BIG}, "/multipleOf"),
    ]

    def test_every_constrained_position_refuses_naming_the_pointer(self) -> None:
        for _keyword, schema, pointer in self.POINTER_CASES:
            with pytest.raises(ValueError, match=f"Schema constraint at {pointer}\\b"):
                repair_json_loads("5", schema=schema)

    def test_nested_all_of_and_ref_reachable_positions_refuse(self) -> None:
        # properties-nested: the pointer is the full path.
        schema = {
            "type": "object",
            "properties": {"x": {"minimum": _BIG}},
        }
        with pytest.raises(ValueError, match=r"Schema constraint at /properties/x/minimum\b"):
            repair_json_loads('{"x": 5}', schema=schema)
        # allOf-reachable: the fold's members are tree nodes.
        schema = {"allOf": [{"maximum": _BIG}]}
        with pytest.raises(ValueError, match=r"Schema constraint at /allOf/0/maximum\b"):
            repair_json_loads("5", schema=schema)
        # $ref-reachable: $defs targets live in the root tree, so the walk
        # finds what the ref resolves to; the pointer names the def
        # position.
        schema = {
            "$defs": {"big": {"enum": [_BIG]}},
            "oneOf": [{"$ref": "#/$defs/big"}],
        }
        with pytest.raises(ValueError, match=r"Schema constraint at /\$defs/big/enum/0\b"):
            repair_json_loads(str(_BIG), schema=schema)

    def test_the_refusal_fires_before_any_repair_or_salvage(self) -> None:
        # The gate runs where the schema enters, so the refusal is total:
        # every spelling raises, valid input or not, salvage mode or not.
        schema = {"minimum": _BIG}
        for call in (
            lambda: repair_json("5", schema=schema),
            lambda: repair_json_loads("5", schema=schema),
            lambda: repair_json_diagnostics("5", schema=schema),
            lambda: repair_json_loads("5", schema=schema, salvage=True),
            lambda: repair_json_loads(str(_BIG), schema=schema),  # even a matching value
        ):
            with pytest.raises(ValueError, match=r"/minimum\b"):
                call()

    def test_honest_in_range_values_never_refuse(self) -> None:
        # 10**18 is past f64's exact-integer grid but inside u64: exactly
        # what the gate must keep admitting, exactly.
        assert repair_json_loads(str(10**18), schema={"enum": [10**18]}) == 10**18
        # The u64 lane is exact: 2**63 (past i64, inside u64) as a minimum
        # rejects 2**63 - 1 — the honesty the gate protects (the repair
        # lane answers the nothing-recoverable sentinel, never a wrong
        # accept).
        schema = {"minimum": 2**63}
        assert repair_json_loads(str(2**63), schema=schema) == 2**63
        assert repair_json_loads(str(2**63 - 1), schema=schema) == ""
        # i64::MIN and schema floats keep their exact / documented-lossy
        # contracts (the float spelling never refuses; a document above it
        # validates through).
        assert repair_json_loads(str(-(2**63)), schema={"minimum": -(2**63)}) == -(2**63)
        assert repair_json_loads(str(10**26), schema={"minimum": 1e25}) == 10**26

    def test_unconstrained_bigints_never_refuse(self) -> None:
        # Positions that constrain nothing pass, and the document's own
        # huge integers flow as before (the gate is schema-side only).
        assert repair_json_loads("5", schema={"default": _BIG}) == 5
        assert repair_json_loads("5", schema={"examples": [_BIG]}) == 5
        assert (
            repair_json_loads("5", schema={"description": f"values near {_BIG} exist"}) == 5
        )
        # A huge integer nested INSIDE the propertyNames subschema, at its
        # own unconstrained `default` position: not refused.
        assert repair_json_loads("5", schema={"propertyNames": {"default": _BIG}}) == 5
        assert repair_json_loads(str(_BIG + 5), schema={"type": "integer"}) == _BIG + 5

    def test_property_names_big_int_value_is_not_the_gates_refusal(self) -> None:
        # The red-team shape: a huge integer AS the propertyNames value.
        # It is not a constrained position, so the gate stays silent — the
        # call may still fail later for the pre-existing reason (a scalar
        # subschema does not compile), but never with the constraint
        # refusal.
        with pytest.raises(ValueError) as excinfo:
            repair_json_loads("5", schema={"propertyNames": _BIG})
        assert "Schema constraint at" not in str(excinfo.value)

    def test_pointer_escapes_in_reached_keys(self) -> None:
        # A property key with pointer metacharacters escapes ~0/~1 in the
        # reported pointer (the same order resolve_chain unescapes).
        schema = {"properties": {"a/b": {"enum": [_BIG]}}}
        with pytest.raises(ValueError, match=r"Schema constraint at /properties/a~1b/enum/0\b"):
            repair_json_loads("5", schema=schema)
