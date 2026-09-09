"""Schema-guided repair corpus ported from json_repair to tors.

Provenance: json_repair by Stefano Baccianella (MIT),
https://github.com/mangiucugna/json_repair, commit
251d141786d0f6ff561f6ec04d90188a338e2470 (= 0.63.4).
Upstream source: json_repair's tests/test_schema_guided_parse.py
(the machine-local clone is /tmp/opencode/json_repair at commit 251d141).
Contract: DESIGN-json-repair-port.md sections 4, 6, 8, 9.

Mapping: ``repair_json(raw, schema=s, skip_json_loads=True, return_objects=True)``
is ``tors.repair_json_loads(raw, schema=s, skip_json_loads=True)``;
``schema_repair_mode="salvage"`` is ``salvage=True`` (default is standard).
``logging=`` variants, pydantic-model schemas, and monkeypatched internals are
never ported (DESIGN section 9).
"""

from __future__ import annotations

from typing import Any

import pytest

from tors import repair_json, repair_json_loads


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
        # No direct upstream fn; pinned by DESIGN section 8 catalog entry.
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
        # SALVAGE mode: a broken schema is the caller's bug, never
        # salvageable data — the drop sites re-raise it.
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
        # `~1` (slash) before `~0` (tilde) — upstream's unescape order. The
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
        # Degenerate unions: BOTH validators reject the schema itself at
        # the API level (upstream: Python jsonschema's wording; tors: the
        # Rust crate's) — the §8 validation-boundary divergence class, so
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
