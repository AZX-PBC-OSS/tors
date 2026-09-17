"""tors-native behavior pins for the JSON-repair surface.

These tests pin tors's own extensions beyond upstream ``json_repair``: no
upstream parity is asserted here. For context, upstream is ``json_repair``
0.63.4 by Stefano Baccianella (MIT,
https://github.com/mangiucugna/json_repair); tors's divergences from it are
the documented classes in ``design-json-repair-port.md`` §9 and are re-pinned
below as intentional behavior, not parity cases.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from typing import Any

import pytest

from tors import repair_json, repair_json_diagnostics, repair_json_loads

_KEY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"first_name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["first_name", "age"],
    "additionalProperties": False,
}
_ITEMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {"type": "integer"}}},
    "required": ["items"],
    "additionalProperties": False,
}
_COUNT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"count": {"type": "integer"}},
    "required": ["count"],
    "additionalProperties": False,
}
_DATE_SCHEMA: dict[str, Any] = {"type": "string", "format": "date"}
_DATETIME_SCHEMA: dict[str, Any] = {"type": "string", "format": "date-time"}


class TestFenceIntegration:
    """Fence pre-pass (design §7): one wrapping fence unwraps before repair."""

    def test_backtick_json_fence_unwraps(self) -> None:
        assert repair_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_tilde_json_fence_unwraps(self) -> None:
        assert repair_json("~~~json\n[1, 2]\n~~~") == "[1, 2]"

    def test_four_tick_fence_unwraps(self) -> None:
        # CommonMark §4.5: the closer must run at least as long as the opener.
        assert repair_json('````json\n{"a": 1}\n````') == '{"a": 1}'

    def test_indented_fence_unwraps(self) -> None:
        assert repair_json('  ```json\n  {"a": 1}\n  ```') == '{"a": 1}'

    def test_crlf_fence_content_repairs(self) -> None:
        # The unwrap preserves the CRLF raw span; repair normalizes it away.
        assert repair_json('```json\r\n{"a": 1}\r\n```') == '{"a": 1}'

    def test_prose_before_fence_garbage_skips_to_object(self) -> None:
        # No unwrap (the fence does not span the whole input); the repair
        # parser's garbage-skip reaches the object instead.
        assert repair_json('Here is your JSON:\n```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_two_fenced_blocks_yield_multi_value_list(self) -> None:
        text = '```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```'
        assert repair_json_loads(text) == [{"a": 1}, {"b": 2}]

    def test_fenced_top_level_scalar_is_recovered(self) -> None:
        # §9.4 divergence: json_repair returns "" here because its
        # garbage-skip only reaches containers, while tors's fence pre-pass
        # unwraps first so the strict fast path sees the bare scalar '"hi"'.
        assert repair_json('```json\n"hi"\n```') == '"hi"'


class TestKeyLadder:
    """Key-normalization ladder + fuzzy remap tier (design §6.0–6.1)."""

    @pytest.mark.parametrize("typo", ["First Name", "first-name", "FIRST_NAME", "firstname"])
    def test_normalization_ladder_remaps_case_separator_variants(self, typo: str) -> None:
        value, diags = repair_json_diagnostics(
            json.dumps({typo: "Ada", "age": 30}), schema=_KEY_SCHEMA
        )
        assert value == {"first_name": "Ada", "age": 30}
        remaps = [d for d in diags if d["action"] == "remap_key"]
        assert len(remaps) == 1
        assert remaps[0]["from"] == typo
        assert remaps[0]["to"] == "first_name"

    def test_fuzzy_tier_remaps_close_typo(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
            "additionalProperties": False,
        }
        assert repair_json_loads('{"nam": "Ada", "age": 30}', schema=schema) == {
            "name": "Ada",
            "age": 30,
        }

    def test_ambiguous_key_is_not_remapped(self) -> None:
        # "abcf" is equidistant from "abcd" and "abce": the best score is not
        # unique within the 0.05 margin, so no remap fires and the key stays.
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"abcd": {"type": "string"}, "abce": {"type": "string"}},
        }
        value, diags = repair_json_diagnostics('{"abcf": "x"}', schema=schema)
        assert value == {"abcf": "x"}
        assert all(d["action"] != "remap_key" for d in diags)

    def test_below_threshold_key_is_not_remapped(self) -> None:
        value, diags = repair_json_diagnostics(
            '{"first_name": "Ada", "age": 30, "zzz": 1}', schema=_KEY_SCHEMA
        )
        assert value == {"first_name": "Ada", "age": 30}
        assert all(d["action"] != "remap_key" for d in diags)

    def test_no_remap_when_target_present(self) -> None:
        value, diags = repair_json_diagnostics(
            '{"first_name": "Ada", "First Name": "Bo", "age": 30}',
            schema=_KEY_SCHEMA,
        )
        assert value["first_name"] == "Ada"
        assert all(d["action"] != "remap_key" for d in diags)

    def test_no_remap_when_additional_properties_true(self) -> None:
        # The two-tier split: the mechanical fold tier remaps even on
        # permissive schemas (deterministic match, data would otherwise be
        # stranded on a dead key); the fuzzy tier is a guess and stays
        # gated: "nam" against a permissive, non-required "name" keeps the
        # key and only suggests.
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": True,
        }
        mechanical, diags = repair_json_diagnostics('{"Na Me": "Ada"}', schema=schema)
        assert mechanical == {"name": "Ada"}
        assert any(d["action"] == "remap_key" for d in diags)
        fuzzy, diags = repair_json_diagnostics('{"nam": "Ada"}', schema=schema)
        assert fuzzy == {"nam": "Ada"}
        assert any(d["action"] == "suggest" for d in diags)
        assert all(d["action"] != "remap_key" for d in diags)

    def test_mechanical_remap_on_valid_json_fast_path(self) -> None:
        # Valid JSON + permissive schema still repairs separator slop: the
        # fast path runs the mechanical key pass before the validity
        # shortcut, so no repair-parser roundtrip is needed to fix it.
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"first_name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["age"],
        }
        assert repair_json_loads('{"First Name": "Ada", "age": 30}', schema=schema) == {
            "first_name": "Ada",
            "age": 30,
        }
        # And the kebab/spellings family through the same lane.
        assert repair_json_loads('{"first-name": "Ada", "age": 30}', schema=schema) == {
            "first_name": "Ada",
            "age": 30,
        }


class TestEnumSuggestion:
    def test_close_enum_value_suggests_member(self) -> None:
        # The suggestion surfaces where the value survives parsing (inside
        # an object): a top-level bare '"blu"' is dropped to "" by the
        # repair parser's top-level skip (upstream does the same), so the
        # validator's own message raises there instead.
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"color": {"type": "string", "enum": ["blue", "green"]}},
            "required": ["color"],
        }
        with pytest.raises(ValueError, match=re.escape("Did you mean 'blue'?")):
            repair_json_loads('{"color": "blu"}', schema=schema)


class TestDateNormalization:
    """Date/datetime normalization (design §6.3)."""

    @pytest.mark.parametrize(
        ("raw", "want"),
        [
            ("2024/03/15", "2024-03-15"),
            ("March 15, 2024", "2024-03-15"),
            ("15 Mar 2024", "2024-03-15"),
            ("13/04/2024", "2024-04-13"),
        ],
    )
    def test_date_forms_normalize(self, raw: str, want: str) -> None:
        assert repair_json_loads(json.dumps(raw), schema=_DATE_SCHEMA) == want

    def test_slash_date_emits_format_date_diagnostic(self) -> None:
        value, diags = repair_json_diagnostics(json.dumps("2024/03/15"), schema=_DATE_SCHEMA)
        assert value == "2024-03-15"
        assert any(d["action"] == "format_date" for d in diags)

    def test_already_normalized_date_has_no_diagnostic(self) -> None:
        value, diags = repair_json_diagnostics(json.dumps("2024-03-15"), schema=_DATE_SCHEMA)
        assert value == "2024-03-15"
        assert all(d["action"] != "format_date" for d in diags)

    def test_ambiguous_numeric_date_is_unchanged_with_suggest(self) -> None:
        value, diags = repair_json_diagnostics(json.dumps("03/04/2024"), schema=_DATE_SCHEMA)
        assert value == "03/04/2024"
        assert any(d["action"] == "suggest" for d in diags)

    def test_invalid_calendar_date_is_unchanged(self) -> None:
        assert repair_json_loads(json.dumps("2024-02-30"), schema=_DATE_SCHEMA) == "2024-02-30"

    @pytest.mark.parametrize(
        ("raw", "want"),
        [
            ("2024-03-15 14:30", "2024-03-15T14:30:00"),
            # Offset-bearing input normalizes to its utc instant (jiff's
            # rendering): 14:30+05:30 is 09:00Z.
            ("2024-03-15T14:30:00+0530", "2024-03-15T09:00:00Z"),
            ("2024-03-15T14:30:00Z", "2024-03-15T14:30:00Z"),
            ("2024-03-15T14:30:00.123Z", "2024-03-15T14:30:00.123Z"),
        ],
    )
    def test_datetime_forms_normalize(self, raw: str, want: str) -> None:
        assert repair_json_loads(json.dumps(raw), schema=_DATETIME_SCHEMA) == want

    def test_time_format_normalizes(self) -> None:
        # format: time: seconds always present, no offset invented.
        schema: dict[str, Any] = {"type": "string", "format": "time"}
        assert repair_json_loads(json.dumps("14:30"), schema=schema) == "14:30:00"
        assert repair_json_loads(json.dumps("14:30:00"), schema=schema) == "14:30:00"

    def test_uuid_format_lowercases(self) -> None:
        # format: uuid: shape-gated canonical lowercase; non-uuids pass
        # through untouched for validation to judge.
        schema: dict[str, Any] = {"type": "string", "format": "uuid"}
        assert (
            repair_json_loads(json.dumps("A1B2C3D4-0000-1111-2222-333344445555"), schema=schema)
            == "a1b2c3d4-0000-1111-2222-333344445555"
        )
        assert repair_json_loads(json.dumps("not a uuid"), schema=schema) == "not a uuid"


class TestCommaSplit:
    """Comma-split array recovery (design §6.1b)."""

    def test_comma_separated_string_splits_to_typed_array(self) -> None:
        # §9.7 divergence class: upstream raises here, while tors's
        # validation-gated split wins because [1, 2, 3] validates against the
        # items-integer schema.
        assert repair_json_loads('{"items": "1, 2, 3"}', schema=_ITEMS_SCHEMA) == {
            "items": [1, 2, 3]
        }

    def test_comma_free_string_keeps_upstream_wrap_or_raise(self) -> None:
        # No commas, so the split cannot win: whichever the corpus pins
        # (wrap-singleton or raise) stands: accept either without re-pinning
        # the corpus case itself.
        try:
            got = repair_json_loads('{"items": "not json"}', schema=_ITEMS_SCHEMA)
        except ValueError:
            pass
        else:
            assert got == {"items": ["not json"]}


class TestSeparators:
    def test_digit_group_separators_are_stripped_for_integers(self) -> None:
        assert repair_json_loads('{"count": "1,234"}', schema=_COUNT_SCHEMA) == {"count": 1234}
        assert repair_json_loads('{"count": "82_461_110"}', schema=_COUNT_SCHEMA) == {
            "count": 82461110
        }


class TestDiagnosticsShape:
    """Structured-diagnostics contract (design §6.4)."""

    _ACTIONS = frozenset(
        {
            "coerce",
            "fill",
            "insert_default",
            "remap_key",
            "suggest",
            "drop_property",
            "drop_item",
            "unwrap_string",
            "wrap_array",
            "fill_required",
            "format_date",
            "skip_fragment",
            "map_array_to_object",
            "unwrap_root_array",
        }
    )

    def _probes(self) -> list[tuple[str, dict[str, Any]]]:
        suggest_schema: dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": True,
        }
        return [
            (json.dumps({"First Name": "Ada", "age": 30}), {"schema": _KEY_SCHEMA}),
            (json.dumps("2024/03/15"), {"schema": _DATE_SCHEMA}),
            (json.dumps("03/04/2024"), {"schema": _DATE_SCHEMA}),
            ('{"items": "1, 2, 3"}', {"schema": _ITEMS_SCHEMA}),
            ('{"nam": "Ada"}', {"schema": suggest_schema}),
            ('{"count": "1,234"}', {"schema": _COUNT_SCHEMA}),
        ]

    def test_diagnostic_dicts_carry_all_six_keys(self) -> None:
        for raw, kwargs in self._probes():
            _, diags = repair_json_diagnostics(raw, **kwargs)
            assert diags
            for diag in diags:
                assert set(diag) == {
                    "action",
                    "path",
                    "detail",
                    "from",
                    "to",
                    "suggestion",
                }

    def test_actions_come_from_the_closed_vocabulary(self) -> None:
        for raw, kwargs in self._probes():
            for diag in repair_json_diagnostics(raw, **kwargs)[1]:
                assert diag["action"] in self._ACTIONS, diag["action"]

    def test_unused_from_to_slots_are_none(self) -> None:
        # The ambiguous-date suggest rewrites no value, so from/to stay None.
        _, diags = repair_json_diagnostics(json.dumps("03/04/2024"), schema=_DATE_SCHEMA)
        suggests = [d for d in diags if d["action"] == "suggest"]
        assert suggests
        for diag in suggests:
            assert diag["from"] is None
            assert diag["to"] is None

    def test_value_matches_loads_and_valid_json_has_no_diagnostics(self) -> None:
        raw = '{"items": "1, 2, 3"}'
        value, _ = repair_json_diagnostics(raw, schema=_ITEMS_SCHEMA)
        assert value == repair_json_loads(raw, schema=_ITEMS_SCHEMA)
        valid, valid_diags = repair_json_diagnostics('{"a": 1}')
        assert valid == {"a": 1}
        # v1 scope: schema-free repair of valid JSON emits no diagnostics.
        assert valid_diags == []


class TestArgumentContracts:
    @pytest.mark.parametrize("fn", [repair_json, repair_json_loads, repair_json_diagnostics])
    def test_non_str_input_raises_type_error(self, fn: Any) -> None:
        with pytest.raises(TypeError):
            fn(1)  # type: ignore[arg-type]

    def test_strict_and_schema_together_raise(self) -> None:
        with pytest.raises(
            ValueError, match=re.escape("schema and strict cannot be used together.")
        ):
            repair_json('{"a": 1}', strict=True, schema={"type": "object"})

    def test_salvage_without_schema_raises(self) -> None:
        with pytest.raises(ValueError, match=re.escape("salvage=True requires schema.")):
            repair_json_loads('{"a": 1}', salvage=True)

    def test_non_dict_schema_raises(self) -> None:
        # Plain non-dict, non-model schemas are rejected (pydantic v2 models
        # are accepted: see the e2e suite).
        with pytest.raises(
            ValueError,
            match=re.escape(
                "schema must be a JSON Schema dict, boolean schema, or pydantic v2 model."
            ),
        ):
            repair_json_loads('{"a": 1}', schema=42)  # type: ignore[arg-type]


_NUM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"n": {"type": "number"}},
    "required": ["n"],
}


class TestLocale:
    """The `locale=` knob: known conventions beat the en-US assumption."""

    def test_tag_and_dict_forms_override_the_en_us_assumption(self) -> None:
        raw = '{"n": "1,234"}'
        assert repair_json_loads(raw, schema=_NUM_SCHEMA, locale="de-DE") == {"n": 1.234}
        assert repair_json_loads(raw, schema=_NUM_SCHEMA, locale="de_de") == {"n": 1.234}
        assert repair_json_loads(
            raw, schema=_NUM_SCHEMA, locale={"decimal": ",", "grouping": "."}
        ) == {"n": 1.234}
        # The default assumption for the same input:
        assert repair_json_loads(raw, schema=_NUM_SCHEMA) == {"n": 1234}
        assert repair_json_loads(raw, schema=_NUM_SCHEMA, locale="en-US") == {"n": 1234}

    def test_known_locale_has_no_disclosure_diagnostic(self) -> None:
        _, diags = repair_json_diagnostics('{"n": "1,234"}', schema=_NUM_SCHEMA, locale="de-DE")
        assert all(d["action"] != "suggest" for d in diags)

    @pytest.mark.parametrize(
        "bad",
        [
            "en-IN",  # Lakh grouping is not supported
            "en-PK",
            "1x",
            5,
            ["de"],
            {"decimal": ","},  # missing grouping
            {"grouping": "."},  # missing decimal
            {"decimal": "ab", "grouping": "."},  # multi-char value
            {"decimal": 5, "grouping": "."},  # non-str value
        ],
    )
    def test_bad_locale_raises_value_error(self, bad: object) -> None:
        with pytest.raises(ValueError, match="locale"):
            repair_json_loads('{"n": "1,234"}', schema=_NUM_SCHEMA, locale=bad)  # type: ignore[arg-type]


class TestTier4Ambiguity:
    """Auto-mode separator ambiguity: assume en-US, schema-checked, disclosed."""

    def test_number_field_assumes_en_us_with_disclosure(self) -> None:
        value, diags = repair_json_diagnostics('{"n": "1,234"}', schema=_NUM_SCHEMA)
        assert value == {"n": 1234}
        suggests = [d for d in diags if d["action"] == "suggest"]
        assert len(suggests) == 1
        # The suggestion names the discarded reading's override: suggesting
        # the winner's own locale would be a no-op.
        assert suggests[0]["suggestion"] == "locale='de-DE'"

    def test_schema_decides_the_ambiguous_reading_without_disclosure(self) -> None:
        capped: dict[str, Any] = {
            "type": "object",
            "properties": {"n": {"type": "number", "maximum": 1000}},
            "required": ["n"],
        }
        value, diags = repair_json_diagnostics('{"n": "1,234"}', schema=capped)
        assert value == {"n": 1.234}
        assert all(d["action"] != "suggest" for d in diags)

    def test_integer_field_is_type_disambiguated_silently(self) -> None:
        # "1,234" has exactly one integral reading (1234); 1.234 is not an
        # integer: the declared type alone resolves it.
        value, diags = repair_json_diagnostics('{"count": "1,234"}', schema=_COUNT_SCHEMA)
        assert value == {"count": 1234}
        assert all(d["action"] != "suggest" for d in diags)

    def test_unresolvable_ambiguity_refusal_carries_the_locale_hint(self) -> None:
        # A single separated number whose readings all fail the declared
        # type: the knob could change the outcome, so it is named.
        with pytest.raises(ValueError, match="pass locale='en-US' or 'de-DE'"):
            repair_json_loads('{"count": "1,23"}', schema=_COUNT_SCHEMA)

    def test_prose_without_a_single_number_gets_the_plain_refusal(self) -> None:
        # Two numbers (or none): no locale could change the outcome, so
        # upstream's bare message is kept.
        for raw in ('{"count": "between 10 and 20"}', '{"count": "0x10"}'):
            with pytest.raises(ValueError) as excinfo:
                repair_json_loads(raw, schema=_COUNT_SCHEMA)
            assert str(excinfo.value) == "Expected integer at $.count."


class TestExtractionTiers:
    """tors-native single-number extraction and percent-by-type."""

    def test_percent_by_declared_type(self) -> None:
        assert repair_json_loads('{"n": "50%"}', schema=_NUM_SCHEMA) == {"n": 0.5}
        assert repair_json_loads('{"count": "50%"}', schema=_COUNT_SCHEMA) == {"count": 50}

    def test_currency_and_prose_single_number_extraction(self) -> None:
        assert repair_json_loads('{"n": "$50.0"}', schema=_NUM_SCHEMA) == {"n": 50.0}
        assert repair_json_loads('{"n": "USD 50"}', schema=_NUM_SCHEMA) == {"n": 50}
        assert repair_json_loads('{"n": "value is 42 units"}', schema=_NUM_SCHEMA) == {"n": 42}

    def test_whitespaced_number_keeps_the_declared_type(self) -> None:
        # Python's int()/float() accept surrounding whitespace: " 5" is an
        # integer on integer fields and a float on number fields.
        assert repair_json_loads('{"count": " 5"}', schema=_COUNT_SCHEMA) == {"count": 5}
        assert repair_json_loads('{"n": " 5"}', schema=_NUM_SCHEMA) == {"n": 5.0}

    def test_leading_decimal_point_is_never_dropped(self) -> None:
        # ".5" must never coerce to 5 on an integer field (the grammar
        # would extract "5" and drop the decimal marker).
        with pytest.raises(ValueError, match="Expected integer at"):
            repair_json_loads('{"count": ".5"}', schema=_COUNT_SCHEMA)
        # Number fields read it as the decimal it is.
        assert repair_json_loads('{"n": ".5"}', schema=_NUM_SCHEMA) == {"n": 0.5}


class TestExactNumbers:
    """Python's unbounded int() semantics: never a saturating cast."""

    _INT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
    }

    def test_big_integer_strings_stay_exact(self) -> None:
        raw = '{"n": "12345678901234567890123"}'
        assert repair_json_loads(raw, schema=self._INT_SCHEMA) == {"n": 12345678901234567890123}

    def test_big_integral_floats_convert_to_their_exact_decimal(self) -> None:
        # int(1e30) in Python is the exact value of the binary float.
        raw = '{"n": 1e30}'
        assert repair_json_loads(raw, schema=self._INT_SCHEMA, skip_json_loads=True) == {
            "n": 1000000000000000019884624838656
        }

    def test_big_int_to_string_coercion(self) -> None:
        raw = '{"id": 12345678901234567890123}'
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        }
        assert repair_json_loads(raw, schema=schema) == {"id": "12345678901234567890123"}

    def test_non_finite_to_string_uses_pythons_repr_spellings(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"v": {"type": "string"}},
            "required": ["v"],
        }
        assert repair_json_loads('{"v": NaN}', schema=schema) == {"v": "nan"}
        assert repair_json_loads('{"v": Infinity}', schema=schema) == {"v": "inf"}
        assert repair_json_loads('{"v": -Infinity}', schema=schema) == {"v": "-inf"}

    def test_non_finite_numbers_reject_under_numeric_schemas(self) -> None:
        # §9.5: JSON Schema validation cannot represent non-finite numbers.
        for raw in ('{"n": NaN}', '{"n": Infinity}'):
            with pytest.raises(ValueError):
                repair_json_loads(raw, schema=_NUM_SCHEMA)


class TestRobustness:
    """Adversarial-input regressions from the red-team review."""

    def test_garbage_separated_comment_runs_raise_instead_of_crashing(self) -> None:
        # '/x' chains parse_json <-> parse_comment without unwinding: 2
        # stack frames per 2 chars; without the depth guard this segfaults
        # near 11k pairs.
        for pattern in ("/x", "/*", "a/"):
            with pytest.raises(
                ValueError, match="Input nesting exceeds the supported parser recursion depth"
            ):
                repair_json(pattern * 12_000, skip_json_loads=True)

    def test_comma_merged_object_fragments_raise_instead_of_crashing(self) -> None:
        # `{"a":1}` + `, "k":1}` * N is handled by complete_object_parse's
        # comma-merge continuation, which recurses into parse_object per
        # fragment; without the depth guard this overflowed the native stack
        # (an uncatchable SIGSEGV) at a few thousand fragments on a worker
        # stack. The guard caps it and raises the same catchable ValueError as
        # the other deep-recursion paths.
        with pytest.raises(
            ValueError, match="Input nesting exceeds the supported parser recursion depth"
        ):
            repair_json('{"a":1}' + ', "k":1}' * 2_000, skip_json_loads=True)
        # the schema-guided path flows `schema` through the same guarded site:
        with pytest.raises(
            ValueError, match="Input nesting exceeds the supported parser recursion depth"
        ):
            repair_json(
                '{"a":1}' + ', "k":1}' * 2_000,
                schema={"type": "object"},
                skip_json_loads=True,
            )

    def test_comma_merged_fragments_below_the_cap_still_merge(self) -> None:
        # The guard must fire only past MAX_NESTING, never on an ordinary
        # merge chain: a regression that over-counts depth would raise early
        # and silently change behavior on inputs upstream handles: the exact
        # parity-risk class this guard is scoped to avoid.
        payload = '{"a":1}' + "".join(f', "k{i}":1}}' for i in range(150))
        merged = repair_json_loads(payload, skip_json_loads=True)
        assert merged == {"a": 1, **{f"k{i}": 1 for i in range(150)}}

    def test_merged_array_continuation_chains_raise_instead_of_crashing(self) -> None:
        # `{"a":[0],` + `["b":[0],` * N nests through the array-continuation
        # merge: a '[' at the key position merges into the previous
        # array-valued member, and the merged array's first item: a string
        # followed by ':': is a missing object start parsed by parse_object
        # directly, whose key scan sees another '[' and merges again. That
        # cycle had no depth guard anywhere on it: it grew the native stack
        # per fragment and overflowed: an uncatchable SIGSEGV around 8k
        # fragments (main thread; fewer on worker-sized stacks): instead of
        # the documented catchable ValueError. The continuation guard caps
        # it like every other deep-recursion path.
        payload = '{"a":[0],' + '["b":[0],' * 2_000 + "1]"
        with pytest.raises(
            ValueError, match="Input nesting exceeds the supported parser recursion depth"
        ):
            repair_json(payload, skip_json_loads=True)
        # schema-guided and salvage parsing flow through the same guarded site:
        with pytest.raises(
            ValueError, match="Input nesting exceeds the supported parser recursion depth"
        ):
            repair_json(payload, schema={"type": "object"}, skip_json_loads=True)
        with pytest.raises(
            ValueError, match="Input nesting exceeds the supported parser recursion depth"
        ):
            repair_json(payload, schema={"type": "object"}, salvage=True, skip_json_loads=True)

    def test_merged_array_continuations_below_the_cap_still_merge(self) -> None:
        # The guard must fire only past MAX_NESTING, never on an ordinary
        # merge chain: nested chains below the cap still merge every
        # fragment (an over-counting regression would raise early), and
        # same-level sequential merges never accrue depth at all:
        # enter/leave is balanced per continuation.
        assert repair_json_loads('{"a":[0],["b":[0],["b":[0],1]', skip_json_loads=True) == {
            "a": [0, {"b": [0, {"b": [0], "1": ""}]}]
        }
        expected: dict[str, Any] = {"b": [0], "1": ""}
        for _ in range(149):
            expected = {"b": [0, expected]}
        payload = '{"a":[0],' + '["b":[0],' * 150 + "1]"
        assert repair_json_loads(payload, skip_json_loads=True) == {"a": [0, expected]}
        assert repair_json_loads('{"a":[1], [2], [3]}', skip_json_loads=True) == {"a": [1, 2, 3]}

    def test_continuation_chains_cap_at_max_nesting_exactly(self) -> None:
        # Both continuation recursions share the MAX_NESTING budget with
        # structural nesting. The comma chain spends 1 (the initial `{`) +
        # 1 per fragment (scalar values add nothing): 199 fragments parse
        # (depth 200), the 200th raises. The array-merge chain spends the
        # same 1 + 1 per fragment plus 1 for the innermost fragment's
        # `[0]` value (a container nested inside every merge): 198
        # fragments parse, the 199th raises. Pinning the exact edges
        # catches future accounting drift in either direction:
        # over-counting an edge rejects inputs the cap admits, missing one
        # reopens the crash.
        comma_ok = '{"a":1}' + ', "k":1}' * 199
        assert repair_json_loads(comma_ok, skip_json_loads=True) == {"a": 1, "k": 1}
        with pytest.raises(
            ValueError, match="Input nesting exceeds the supported parser recursion depth"
        ):
            repair_json('{"a":1}' + ', "k":1}' * 200, skip_json_loads=True)
        expected: dict[str, Any] = {"b": [0], "1": ""}
        for _ in range(197):
            expected = {"b": [0, expected]}
        merge_ok = '{"a":[0],' + '["b":[0],' * 198 + "1]"
        assert repair_json_loads(merge_ok, skip_json_loads=True) == {"a": [0, expected]}
        with pytest.raises(
            ValueError, match="Input nesting exceeds the supported parser recursion depth"
        ):
            repair_json('{"a":[0],' + '["b":[0],' * 199 + "1]", skip_json_loads=True)

    def test_related_recursion_shapes_route_through_guarded_edges(self) -> None:
        # Siblings of the continuation chains that do pass guarded edges on
        # every cycle: string-colon objects nested inside arrays (`["b": [`
        # per level, each through parse_json's '[' branch) and salvage-mode
        # comma-merging (every salvage fragment re-enters parse_json).
        # Pinning them keeps a future refactor from quietly rerouting these
        # shapes past the guards.
        with pytest.raises(
            ValueError, match="Input nesting exceeds the supported parser recursion depth"
        ):
            repair_json("[" + '"b": [' * 2_000, skip_json_loads=True)
        with pytest.raises(
            ValueError, match="Input nesting exceeds the supported parser recursion depth"
        ):
            repair_json(
                '{"a":1}' + ', "k":1}' * 2_000,
                schema={"type": "object"},
                salvage=True,
                skip_json_loads=True,
            )

    def test_strict_mode_has_no_continuation_merges(self) -> None:
        # Both continuation merges are repairs and never fire in strict
        # mode: the comma shape surfaces strict's own multiple-elements
        # error, and the array-merge shape parses without merging.
        with pytest.raises(ValueError, match="Multiple top-level JSON elements"):
            repair_json('{"a":1}, "k":1}', strict=True, skip_json_loads=True)
        assert repair_json_loads('{"a":[0],["b":[0],1]', strict=True, skip_json_loads=True) == {
            "a": [0],
            "b": [0],
        }

    def test_escaped_delimiter_run_in_a_string_body_is_not_quadratic(self) -> None:
        # `{` + `{\"k\": 1}` * n + `}` puts 2n escaped quotes through
        # scan_string_body's escape normalizer; every pop-then-push repair
        # used to rebuild the brace/class counters by rescanning the whole
        # accumulator (O(n^2): ~1.2s at 16k fragments, minutes at the MiB
        # scale: 2.05e9 chars scanned for a 160 KB document, measured).
        # The one-level undo record makes each repair O(1). Absolute wall
        # bound with a large margin over the linear cost (~8ms at 32k) and
        # far under the quadratic (~5s at 32k); the shape is pinned too, so
        # a fast-but-wrong path cannot pass on the wall bound alone.
        import time as _time

        n = 32_000
        payload = "{" + r"{\"k\": 1}" * n + "}"
        start = _time.perf_counter()
        result = repair_json_loads(payload, skip_json_loads=True)
        elapsed = _time.perf_counter() - start
        assert elapsed < 1.5, (
            f"escaped-delimiter run took {elapsed:.2f}s at {n} fragments — "
            "the whole-accumulator rescan is back"
        )
        assert result == {}

    def test_recursion_class_grammar_sweep_stays_total(self) -> None:
        # A seeded sweep over a grammar of every stack-growing construct the
        # parser has: both continuation merges (the array-merge chain
        # needs its array-valued head member `{"a":[0],` to arm the merge
        # hook; the bare fragment chain parses iteratively), structural
        # nesting, string-colon objects, comment runs, escaped keys, parens
        #: at fragment counts far past every cap and past the measured
        # unguarded-crash thresholds (the array-merge chain SIGSEGVs around
        # 8k fragments on the main thread, the comma chain around 15k).
        # Every input must either parse or raise ValueError: a parse build
        # with an unguarded cycle anywhere in this grammar kills the
        # process (which is exactly the loud signal this pin exists to
        # send). Over-aggressive guarding is not this pin's job: the
        # below-cap and boundary tests assert the parses it would break.
        rng = random.Random(20260908)
        chains: list[tuple[str, Callable[[int], str]]] = [
            ("array_merge", lambda n: '{"a":[0],' + '["b":[0],' * n + "1]"),
            ("comma_merge", lambda n: '{"a":1}' + ', "k":1}' * n),
            ("strcolon_nest", lambda n: "[" + '"b": [' * n),
            ("brace_nest", lambda n: '{"a":' * n),
            ("bracket_nest", lambda n: "[" * n),
            ("paren_nest", lambda n: "(" * n),
        ]
        junks = ["", " ", "\n", "/*x*/", "/x", "junk ", ' "s",', "1,", "}"]
        for case in range(48):
            name, build = chains[case % len(chains)]
            count = rng.randrange(250, 20_000)
            # Two thirds pure chains (the crash shapes), one third with
            # junk spliced between fragments: the chains break, but the
            # junk-with-fragments interaction stays covered at scale.
            if case % 3 == 2:
                fragment = {"array_merge": '["b":[0],', "comma_merge": ', "k":1}'}.get(
                    name, build(1)
                )
                payload = (fragment + rng.choice(junks)) * (count // 8)
            else:
                payload = build(count)
            try:
                repair_json(payload, skip_json_loads=True)
            except ValueError:
                pass  # the capped, documented outcome for runaway chains

    def test_backslash_run_before_array_close_is_not_quadratic(self) -> None:
        # `'["' + ']'*n + '\\\\' + '" x'` (an even backslash run makes the
        # close backslash-adjacent) still drove O(n^2) after the memoized
        # `]` lookahead: cached_skip_to_character's `s[m-1] != '\\'` write
        # guard suppressed the memo for exactly those matches, so every `]`
        # rescanned the remaining input (~12s at n=200k before the guard was
        # lifted; the interleaved `']' + '\\\\'` spelling is the same class).
        import time as _time

        n = 200_000
        start = _time.perf_counter()
        result = repair_json_loads('["' + "]" * n + '\\\\" x', skip_json_loads=True)
        assert _time.perf_counter() - start < 3.0
        # pin the shape (byte-identical to json-repair 0.63.4 at every n):
        # the even run halves to nothing, the close survives as content.
        assert result == [("]" * n) + '" x']

    def test_objval_close_run_with_delimiter_gap_is_not_quadratic(self) -> None:
        # `'{"a": "' + '}'*n + '"' + 'y'*n + '"z'`: every `}` in the run ran
        # the `}`-branch's UNmemoized `skip_to_character(&[lstring_delimiter])`
        # over the same long quote-free gap (~5.5s at n=100k before the scan
        # was memoized; the memoized `}` lookahead one line above it already
        # made the rest of the branch linear).
        import time as _time

        n = 100_000
        start = _time.perf_counter()
        result = repair_json_loads('{"a": "' + "}" * n + '"' + "y" * n + '"z', skip_json_loads=True)
        assert _time.perf_counter() - start < 3.0
        assert result == {"a": ("}" * n) + '"' + ("y" * n) + '"z'}

    def test_regex_character_class_quote_run_is_not_quadratic(self) -> None:
        # `'{"a": "[' + 'x"'*n + '"}'`: with a regex character class open and
        # no `]` anywhere ahead, every closing-quote candidate ran
        # quote_belongs_to_regex_character_class's UNmemoized
        # `skip_to_character(&[']'])` to the end of input (~12s at n=100k
        # before the scan was memoized through the string state).
        import time as _time

        n = 100_000
        start = _time.perf_counter()
        result = repair_json_loads('{"a": "[' + 'x"' * n + '"}', skip_json_loads=True)
        assert _time.perf_counter() - start < 3.0
        # The parity of the quote run decides the closer (the last quote
        # closes on odd runs, stays content on even ones): even n keeps
        # every pair as content: `{"a": "[" + 'x"'*n}`.
        assert result == {"a": "[" + ('x"' * n)}

    def test_object_key_colon_run_is_not_quadratic(self) -> None:
        # `'{' + 'a:b,'*n + '}'`: unquoted object keys put the scan in
        # ObjectKey context, where every `:` ran two UNmemoized
        # skip_to_character lookaheads over the whole remaining member run
        # (~25s at n=100k before both scans were memoized). The repaired
        # value is n-independent (the duplicate key splits collapse), so the
        # shape pin is a constant.
        import time as _time

        n = 100_000
        start = _time.perf_counter()
        result = repair_json_loads("{" + "a:b," * n + "}", skip_json_loads=True)
        assert _time.perf_counter() - start < 3.0
        assert result == {"a": "b"}

    def test_internal_quote_run_in_array_string_is_not_quadratic(self) -> None:
        # `'["' + 'a"'*n + '"]'`: every internal quote candidate in an
        # array-context string body walked handle_right_delimiter_candidate's
        # delimiter-pairing loop over all remaining quotes (~7s at n=100k
        # before the walk outcomes were cached). The pin holds for every n:
        # the quote pairing keeps every internal quote as content and the
        # final quote closes the string.
        import time as _time

        n = 100_000
        start = _time.perf_counter()
        result = repair_json_loads('["' + 'a"' * n + '"]', skip_json_loads=True)
        assert _time.perf_counter() - start < 3.0
        assert result == ['a"' * n]

    def test_interleaved_close_and_escape_run_is_not_quadratic(self) -> None:
        # `'["' + (']' + '\\\\')*k + '" x'`: the interleaved even-backslash
        # runs made the escape normalizer rewrite the accumulator tail once
        # per pair, and each rewrite rebuilt the whole accumulator
        # (rebuild_unmatched_opening_braces): O(k) per pair, O(k^2) total
        # (~12s at k=100k). The rewrite now pops the counter-neutral
        # backslash and appends through the incremental bookkeeping, O(1)
        # per pair. (Upstream rebuilds per rewrite and stays quadratic.)
        import time as _time

        k = 100_000
        start = _time.perf_counter()
        result = repair_json_loads('["' + ("]" + "\\\\") * k + '" x', skip_json_loads=True)
        assert _time.perf_counter() - start < 3.0
        # Each pair but the last contributes `]\` (the `]` is kept, the
        # even run halves to one backslash); the last pair's run collapses
        # entirely before the closing quote, and the tail rides along.
        assert result == [("]" + "\\") * (k - 1) + "]" + '" x']

    def test_well_formed_surrogate_pairs_survive(self) -> None:
        # A legal \udXXX\udCXX pair is the astral char it encodes, and
        # ensure_ascii re-emits the identical pair bytes.
        assert repair_json('["\\ud83d\\ude00",]') == '["\\ud83d\\ude00"]'
        assert repair_json('["\\ud83d\\ude00",]', ensure_ascii=False) == '["😀"]'
        assert repair_json_loads('["\\ud83d\\ude00",]') == ["😀"]

    def test_lone_surrogates_in_input_raise_unicode_encode_error(self) -> None:
        with pytest.raises(UnicodeEncodeError):
            repair_json('"\ud800"')


class TestFenceSameLinePayload:
    """Same-line fence content is an info string; the payload is still found."""

    @pytest.mark.parametrize(
        ("raw", "want"),
        [
            ("```[1,2]", [1, 2]),
            ("```json[1,2]", [1, 2]),
            ("``` [1,2]", [1, 2]),
            ("```[1,2]\n```", [1, 2]),
        ],
    )
    def test_same_line_container_payload_is_recovered(self, raw: str, want: list[Any]) -> None:
        assert repair_json_loads(raw) == want


class TestFoldRevert:
    """The mechanical fold never turns valid input into a failure."""

    def test_incompatible_value_keeps_its_original_key(self) -> None:
        # "Data" folds to "data", but 5 is not an object: renaming would
        # turn a valid document into a coercion failure, so the key stays.
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"data": {"type": "object"}},
        }
        assert repair_json_loads('{"Data": 5}', schema=schema) == {"Data": 5}

    def test_compatible_value_still_folds(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"first_name": {"type": "string"}},
        }
        assert repair_json_loads('{"First Name": "Ada"}', schema=schema) == {"first_name": "Ada"}


class TestDedup:
    """CPython set/dict.fromkeys collapses the reviewer's duplicate shapes."""

    def test_set_object_collapses_duplicate_members(self) -> None:
        assert repair_json_loads("{'a', 'b', 'a'}", schema={"type": "object"}, salvage=True) == {
            "a": None,
            "b": None,
        }

    def test_duplicate_required_entries_collapse_in_errors(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a", "a"],
        }
        with pytest.raises(ValueError) as excinfo:
            repair_json_loads("{}", schema=schema)
        assert str(excinfo.value) == "Missing required properties at $: a"


class TestAllOfPrePass:
    """allOf guidance reaches the fast path like the repair lane."""

    _SCHEMA: dict[str, Any] = {
        "allOf": [{"properties": {"d": {"type": "string", "format": "date"}}}]
    }
    _KEY_SCHEMA: dict[str, Any] = {"allOf": [{"properties": {"first_name": {"type": "string"}}}]}

    def test_allof_wrapped_formats_normalize_on_both_paths(self) -> None:
        raw = '{"d": "2024/03/15"}'
        assert repair_json_loads(raw, schema=self._SCHEMA) == {"d": "2024-03-15"}
        assert repair_json_loads(raw, schema=self._SCHEMA, skip_json_loads=True) == {
            "d": "2024-03-15"
        }

    def test_allof_wrapped_keys_rename_on_both_paths(self) -> None:
        raw = '{"First Name": "Ada"}'
        assert repair_json_loads(raw, schema=self._KEY_SCHEMA) == {"first_name": "Ada"}
        assert repair_json_loads(raw, schema=self._KEY_SCHEMA, skip_json_loads=True) == {
            "first_name": "Ada"
        }


class TestUnionDiagnosticsRollback:
    """A losing union branch leaves no diagnostics behind."""

    def test_failed_branch_actions_are_not_reported(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "v": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "x": {"type": "integer"},
                                "z": {"type": "string"},
                            },
                            "required": ["z"],
                        },
                        {"type": "object"},
                    ]
                }
            },
            "required": ["v"],
        }
        raw = '{"v": {"x": "5", "y": 2}}'
        value, diags = repair_json_diagnostics(raw, schema=schema, skip_json_loads=True)
        # Branch 1 would coerce "5" -> 5 but fails on the required "z";
        # branch 2 wins unchanged, and the losing branch's records are
        # rolled back: the diagnostics describe the returned value only.
        assert value == {"v": {"x": "5", "y": 2}}
        assert diags == []


class TestDateShapesExtended:
    """The extended date-shape accept list."""

    @pytest.mark.parametrize(
        ("fmt", "raw", "want"),
        [
            ("date-time", "2024/03/15 14:30", "2024-03-15T14:30:00"),
            ("date-time", "2024/03/15 14:30:05", "2024-03-15T14:30:05"),
            ("date", "15 March, 2024", "2024-03-15"),
            ("date", "15 Mar, 2024", "2024-03-15"),
            ("date", "2024/3/5", "2024-03-05"),  # strtime shapes are padding-tolerant
        ],
    )
    def test_extended_shapes_normalize(self, fmt: str, raw: str, want: str) -> None:
        schema: dict[str, Any] = {"type": "string", "format": fmt}
        assert repair_json_loads(json.dumps(raw), schema=schema) == want

    def test_iso_dashes_require_two_digit_padding(self) -> None:
        # jiff's iso parser is strict on padding: "2024-3-5" stays
        # untouched for validation to judge.
        assert repair_json_loads(json.dumps("2024-3-5"), schema=_DATE_SCHEMA) == "2024-3-5"


class TestDiagnosticsVocabularyCompleteness:
    """Every §6.4 action fires somewhere: the vocabulary is real."""

    def test_wrap_array_is_recorded(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"],
        }
        _, diags = repair_json_diagnostics('{"items": "not json"}', schema=schema)
        assert any(d["action"] == "wrap_array" for d in diags)

    def test_insert_default_is_recorded_on_the_malformed_path(self) -> None:
        # The parse-side default insertion (object.rs finalize) records too,
        # not just the fast path's schema-layer insertion.
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"x": {"type": "integer", "default": 1}},
        }
        value, diags = repair_json_diagnostics("{", schema=schema, skip_json_loads=True)
        assert value == {"x": 1}
        assert any(d["action"] == "insert_default" for d in diags)

    def test_drop_property_is_recorded_on_the_parser_path(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": False,
        }
        raw = "{a: 1, extra: 2}"
        value, diags = repair_json_diagnostics(raw, schema=schema, skip_json_loads=True)
        assert value == {"a": 1}
        assert any(d["action"] == "drop_property" and d["path"] == "$.extra" for d in diags)

    def test_drop_item_is_recorded(self) -> None:
        schema: dict[str, Any] = {
            "type": "array",
            "items": [{"type": "integer"}, {"type": "integer"}],
            "additionalItems": False,
        }
        raw = "[1, 2, 3]"
        value, diags = repair_json_diagnostics(raw, schema=schema, skip_json_loads=True)
        assert value == [1, 2]
        assert any(d["action"] == "drop_item" for d in diags)

    def test_skip_fragment_is_recorded_in_salvage(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        }
        raw = "garbage {b: 2} more {a: 1}"
        value, diags = repair_json_diagnostics(
            raw, schema=schema, salvage=True, skip_json_loads=True
        )
        assert value == {"a": 1}
        assert any(d["action"] == "skip_fragment" for d in diags)

    def test_fill_and_unwrap_string_actions(self) -> None:
        # fill: a missing value (a member whose value position holds a
        # separator) takes the schema's const.
        fill_schema: dict[str, Any] = {
            "type": "object",
            "properties": {"kind": {"const": "sum"}},
            "required": ["kind"],
        }
        value, diags = repair_json_diagnostics('{"kind":}', schema=fill_schema)
        assert value == {"kind": "sum"}
        assert [d["action"] for d in diags] == ["fill"]
        # unwrap_string: a JSON-encoded string unwrapped to an array.
        unwrap_schema: dict[str, Any] = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"],
        }
        assert any(
            d["action"] == "unwrap_string"
            for d in repair_json_diagnostics(
                '{"items": "[\\"a\\", \\"b\\"]"}', schema=unwrap_schema
            )[1]
        )


class TestNothingRecoverableUnderSchema:
    """Under ``schema=``, the ""-nothing-recoverable value is itself validated
    against the schema instead of escaping as the sentinel return the
    schema-free spelling documents: a non-string-typed schema turns "nothing
    recoverable" into the same ``ValueError`` every other nonconformant value
    raises, while a string-typed schema accepts it (an empty string is a valid
    string). Consumers build degrade paths on the raise, so it is pinned."""

    def test_nothing_recoverable_raises_under_a_typed_schema(self) -> None:
        with pytest.raises(ValueError):
            repair_json_loads("no JSON anywhere in this prose", schema=_COUNT_SCHEMA)

    def test_empty_input_raises_under_a_typed_schema(self) -> None:
        with pytest.raises(ValueError):
            repair_json_loads("", schema=_COUNT_SCHEMA)

    def test_object_schema_names_the_expected_type_in_the_error(self) -> None:
        with pytest.raises(ValueError, match="object"):
            repair_json_loads("plain prose, no braces at all", schema=_KEY_SCHEMA)

    def test_string_schema_accepts_the_sentinel(self) -> None:
        # A string-typed schema legitimately accepts "": the raise is the
        # typed-schema behavior, not an unconditional one.
        assert repair_json_loads("no JSON anywhere", schema={"type": "string"}) == ""


class TestRepairDeadline:
    """deadline_ms bounds the repair against pathological O(n^2) parser
    shapes (each shared with upstream json_repair): a bounded abort, not a
    speed-up, and a strict no-op when unset."""

    # Two distinct quadratics; unbounded, each runs for tens of seconds at
    # n=200k. dup-key and empty-object are bounded through the parse_json
    # dispatch loop. (Two more went linear as their classes were fixed:
    # the escaped-object-key splice rescan when #19 landed, the
    # backslash-adjacent string-scan when this branch lifted the memo's
    # write guard; both are pinned below / in TestRobustness as wall-time
    # guards instead.)
    _DUP_KEY = "[{" + '"a":1 "a":1 ' * 200_000 + "}]"
    _EMPTY_OBJ = "[" + "{ }" * 200_000 + "]"

    @pytest.mark.parametrize(
        "raw",
        [_DUP_KEY, _EMPTY_OBJ],
        ids=["dup-key", "empty-object"],
    )
    def test_a_pathological_input_is_bounded_by_the_deadline(self, raw: str) -> None:
        import time as _time

        start = _time.perf_counter()
        with pytest.raises(TimeoutError, match="deadline"):
            repair_json(raw, deadline_ms=100)
        # A real bound: the tens-of-seconds unbounded run is cut short well
        # under 2s.
        assert _time.perf_counter() - start < 2.0

    @pytest.mark.parametrize(
        "raw",
        [
            "[{" + '"a":1 "a":1 ' * 50 + "}]",
            "[" + "{ }" * 50 + "]",
            '["' + "]" * 50 + '\\\\" x',
        ],
        ids=["dup-key", "empty-object", "string-scan"],
    )
    def test_a_generous_deadline_does_not_change_output(self, raw: str) -> None:
        # Below the deadline the result is byte-identical to the unbounded call.
        assert repair_json(raw, deadline_ms=60_000) == repair_json(raw)

    def test_the_backslash_string_scan_stays_linear(self) -> None:
        # `'["' + ']'*n + '\\\\' + '" x'` was the third bounded quadratic
        # (tens of seconds at n=200k) until the lookahead memo's write
        # guard was lifted: backslash-adjacent matches memoize exactly for
        # anchored starts. Pin the wall so it stays that way (~4ms at
        # n=200k; TestRobustness carries the same shape with its
        # oracle-pinned output).
        import time as _time

        raw = '["' + "]" * 200_000 + '\\\\" x'
        start = _time.perf_counter()
        repair_json(raw)
        assert _time.perf_counter() - start < 2.0

    def test_the_escaped_object_shape_stays_linear(self) -> None:
        # '[' + '{\\"k\\":1 ' * n + ']' was the fourth quadratic (30s+ at
        # n=200k) until #19's continuation work made the empty-object
        # reparse bounded per fragment. Pin the wall so it stays that way
        # (~20ms at n=200k; the old quadratic would need tens of seconds).
        import time as _time

        raw = "[" + '{\\"k\\":1 ' * 200_000 + "]"
        start = _time.perf_counter()
        repair_json(raw)
        assert _time.perf_counter() - start < 2.0

    def test_a_large_valid_input_does_not_trip_a_generous_deadline(self) -> None:
        # The deadline distinguishes pathological shape from benign size: a
        # multi-MB well-formed document parses far under a generous budget
        # (an input-size cap could not tell the two apart).
        big = "[" + ",".join(f'{{"k{i}": {i}}}' for i in range(100_000)) + "]"
        assert len(big) > 1_000_000
        repair_json(big, deadline_ms=5_000)  # must not raise

    @pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf")])
    def test_non_positive_or_non_finite_deadline_raises_value_error(self, bad: float) -> None:
        with pytest.raises(ValueError):
            repair_json("{}", deadline_ms=bad)

    def test_all_three_spellings_honor_the_deadline(self) -> None:
        raw = "[{" + '"a":1 "a":1 ' * 200_000 + "}]"
        with pytest.raises(TimeoutError):
            repair_json(raw, deadline_ms=100)
        with pytest.raises(TimeoutError):
            repair_json_loads(raw, deadline_ms=100)
        with pytest.raises(TimeoutError):
            repair_json_diagnostics(raw, deadline_ms=100)

    @pytest.mark.parametrize(
        "name,call",
        [
            ("repair_json", repair_json),
            ("repair_json_loads", repair_json_loads),
            ("repair_json_diagnostics", repair_json_diagnostics),
        ],
    )
    def test_the_timeout_message_names_the_called_spelling(self, name: str, call) -> None:
        # The same wording as diff_opcodes' TimeoutError, fronted with the
        # called spelling's own name (a small input under a 1ms budget
        # aborts on the first dispatch-loop check).
        raw = "[{" + '"a":1 "a":1 ' * 5_000 + "}]"
        with pytest.raises(
            TimeoutError,
            match=rf"^{name} deadline exceeded: elapsed \d+\.\dms > deadline_ms 1\.0ms$",
        ):
            call(raw, deadline_ms=1)

    def test_the_budget_includes_the_strict_fast_path(self) -> None:
        # The clock starts at the top of repair(), so the strict fast path
        # (json.loads attempt) burns the budget too: a multi-MB document
        # whose fast path fails at the truncated tail must report the whole
        # attempt as elapsed, not start a fresh clock at the repair parser.
        raw = "[" + ",".join(f'{{"k{i}": {i}}}' for i in range(400_000))[:-1]
        pattern = r"elapsed (\d+\.\d)ms > deadline_ms 1\.0ms"
        with pytest.raises(TimeoutError, match=pattern) as excinfo:
            repair_json(raw, deadline_ms=1)
        elapsed = float(re.search(r"elapsed (\d+\.\d)ms", str(excinfo.value)).group(1))
        # The fast-path scan of ~4MB is tens of ms; a parser-only clock
        # would report ~1ms. 30ms sits far from both.
        assert elapsed >= 30.0

    def test_completed_fast_path_work_is_returned_not_aborted(self) -> None:
        # The deadline stops further work; it does not nullify done work:
        # a valid document whose fast path completes past a tiny budget
        # still returns its parse, byte-identical to the unbounded call
        # (the same shape as diff_opcodes, where a completed diff returns).
        raw = "[" + ",".join(f'{{"k{i}": {i}}}' for i in range(100_000)) + "]"
        assert repair_json(raw, deadline_ms=1) == repair_json(raw)

    def test_a_live_deadline_does_not_recolor_strict_errors(self) -> None:
        # The deadline discriminates by payload, not by timing: a
        # strict-mode violation under a generous live budget is still the
        # documented ValueError, never a TimeoutError.
        with pytest.raises(ValueError, match="strict mode"):
            repair_json('{"a" 1}', strict=True, deadline_ms=60_000)

    def test_schema_and_salvage_paths_honor_the_deadline(self) -> None:
        # The schema-guided and salvage fragment loops route through the
        # same dispatch-loop check; a small pathological input under a 1ms
        # budget aborts on both.
        raw = "[{" + '"a":1 "a":1 ' * 5_000 + "}]"
        with pytest.raises(TimeoutError):
            repair_json_loads(raw, schema={"type": "array"}, deadline_ms=1)
        with pytest.raises(TimeoutError):
            repair_json_loads(raw, schema={"type": "array"}, salvage=True, deadline_ms=1)

    def test_salvage_unwrap_nested_repair_inherits_the_deadline(self) -> None:
        # H2 escape, fixed: the salvage unwrap of a double-serialized
        # container ran the nested repair() with a FRESH config that dropped
        # the deadline — a parser-quadratic string content (the dup-key
        # shape) ran its full unbounded quadratic (~4s at n=100k) inside a
        # 100ms budget before the outer clock could fire. The nested call
        # now inherits the caller's clock and its abort propagates.
        nested = "[{" + '"a":1 "a":1 ' * 100_000 + "}]"
        raw = '{"payload": ' + '"' + nested.replace('\\', '\\\\').replace('"', '\\"') + '"}'
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"payload": {"type": "object"}},
            "required": ["payload"],
        }
        import time as _time

        start = _time.perf_counter()
        with pytest.raises(TimeoutError):
            repair_json_loads(raw, schema=schema, salvage=True, deadline_ms=100)
        # Unbounded this shape needs ~4s; bounded it aborts at the budget
        # plus one unwound frame (~0.1s). 1.5s clears both margins.
        assert _time.perf_counter() - start < 1.5

    @pytest.mark.parametrize(
        "name,call",
        [
            ("repair_json", repair_json),
            ("repair_json_loads", repair_json_loads),
            ("repair_json_diagnostics", repair_json_diagnostics),
        ],
    )
    def test_the_salvage_unwrap_escape_is_bounded_in_all_spellings(
        self, name: str, call
    ) -> None:
        # The same escape through the other two spellings (all three route
        # through repair(), so the shared-clock fix covers them; pinned
        # spellings-by-spellings so a per-spelling fast path cannot reintroduce
        # a leaky config).
        nested = "[{" + '"a":1 "a":1 ' * 20_000 + "}]"
        raw = '{"payload": ' + '"' + nested.replace('\\', '\\\\').replace('"', '\\"') + '"}'
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"payload": {"type": "object"}},
            "required": ["payload"],
        }
        import time as _time

        start = _time.perf_counter()
        with pytest.raises(TimeoutError):
            call(raw, schema=schema, salvage=True, deadline_ms=100)
        assert _time.perf_counter() - start < 1.5

    def test_allof_wrapping_anyof_stays_bounded(self) -> None:
        # H2 composition: anyOf (whose loop head is the forced check) nested
        # inside allOf (whose members ride the per-call entry sample) inside
        # an object property: the abort must fire through both wrappers.
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "value": {
                    "allOf": [
                        {"anyOf": [{"type": "string", "enum": ["nope"]} for _ in range(20_000)]},
                    ]
                }
            },
            "required": ["value"],
        }
        import time as _time

        start = _time.perf_counter()
        with pytest.raises(TimeoutError):
            repair_json_loads('{"value": 123}', schema=schema, deadline_ms=1)
        assert _time.perf_counter() - start < 2.0

    def test_ref_fanout_over_a_huge_array_stays_bounded(self) -> None:
        # H2: $ref-driven work over a large instance — every item's schema
        # is a one-hop $ref, so the chain walk repeats per item; the
        # per-item repair_value_d entry sample must keep the walk bounded.
        schema: dict[str, Any] = {
            "$defs": {"item": {"type": "integer"}},
            "type": "array",
            "items": {"$ref": "#/$defs/item"},
        }
        raw = "[" + ",".join(f'"{i}"' for i in range(300_000)) + "]"
        import time as _time

        start = _time.perf_counter()
        with pytest.raises(TimeoutError):
            repair_json_loads(raw, schema=schema, deadline_ms=1)
        assert _time.perf_counter() - start < 2.0

    # The schema layer's own alignment work (key ladder, union retries,
    # coercion, fill-missing, validation) samples the same clock: before
    # #79's fix none of it did, and a schema whose fast path SUCCEEDS
    # swallowed the expiry whole (the post-fast-path check only runs when
    # the fast path falls through).
    _LADDER_SCHEMA: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {f"property_{i:06}": {"type": "integer"} for i in range(1000)},
    }
    _LADDER_KEYS = 20_000

    def test_the_schema_key_ladder_is_bounded_by_the_deadline(self) -> None:
        # A 1000-property schema with near-miss keys drives the fuzzy
        # ladder (an O(properties) jaro sweep + sort per unknown key)
        # through repair_value; unbounded, 20k keys run ~5.5s. Under a 1ms
        # budget the call must abort well under a second.
        raw = "{" + ",".join(f'"propertx_{i:06}": {i}' for i in range(self._LADDER_KEYS)) + "}"
        import time as _time

        start = _time.perf_counter()
        with pytest.raises(TimeoutError):
            repair_json_loads(raw, schema=self._LADDER_SCHEMA, deadline_ms=1)
        assert _time.perf_counter() - start < 1.0

    def test_the_schema_union_branch_loop_is_bounded_by_the_deadline(self) -> None:
        # A union whose winning branch sits at the end of a long anyOf:
        # the losing branches burn the budget branch by branch, and the
        # late win returns through the fast path (no fall-through check
        # left to catch the expiry). The loop head must sample the clock.
        schema: dict[str, Any] = {
            "anyOf": [{"type": "string", "enum": ["nope"]} for _ in range(20_000)]
            + [{"type": "string"}],
        }
        with pytest.raises(TimeoutError):
            repair_json_loads("123", schema=schema, deadline_ms=1)

    def test_a_generous_deadline_does_not_change_schema_path_output(self) -> None:
        # Below the deadline the schema-path result is identical to the
        # unbounded call (the schema-path mirror of the parser-path pin).
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "first_name": {"type": "string"},
                **{f"property_{i:06}": {"type": "integer"} for i in range(50)},
            },
        }
        raw = '{"First_Name": "Ada", "propertx_000000": 1}'
        assert repair_json_loads(raw, schema=schema, deadline_ms=60_000) == repair_json_loads(
            raw, schema=schema
        )

    # ---- Red-team pins: every schema-layer sample point is load-bearing.
    # Each test below was mutation-verified: temporarily deleting the check
    # it guards (the entry sample in repair_value_d, the forced union /
    # type-union / ladder loop-head checks, the normalize_keys /
    # prenormalize_dates / suggest_scan walk samples) makes exactly one of
    # these fail, so a future refactor cannot silently disarm the clock.

    def test_the_allof_member_fold_is_bounded_by_the_deadline(self) -> None:
        # A long allOf member list whose fold work per member is trivial
        # (boolean-true members) and whose branch bodies fire no other
        # check: the per-member entry sample in repair_value_d is the only
        # reader. Delete it and the fold sweeps all members and the fast
        # path returns (no TimeoutError, ~140ms), so pytest.raises is the
        # discriminator.
        schema: dict[str, Any] = {
            "allOf": [{"properties": {"y": {"type": "integer"}}}] + [True] * 100_000,
        }
        import time as _time

        start = _time.perf_counter()
        with pytest.raises(TimeoutError):
            repair_json_loads('{"y": "7"}', schema=schema, deadline_ms=1)
        assert _time.perf_counter() - start < 2.0

    def test_the_allof_fold_is_bounded_on_the_parser_path_too(self) -> None:
        # The fold pin above rides the fast path, where the advisory walkers'
        # own samples cover the same walk; with skip_json_loads=True the
        # walkers never run and repair_value_d's per-member ENTRY sample is
        # the only clock reader in the fold (mutation-verified: deleting it
        # lets the 100k-member fold complete and the call return, masking
        # the expiry — the tiny document gives the parser's separate counter
        # no read, and validate's entry check is one increment past 100k
        # unchecked calls, still short of a 256th).
        schema: dict[str, Any] = {
            "allOf": [{"properties": {"y": {"type": "integer"}}}] + [True] * 100_000,
        }
        import time as _time

        start = _time.perf_counter()
        with pytest.raises(TimeoutError):
            repair_json_loads('{"y": "7"}', schema=schema, skip_json_loads=True, deadline_ms=1)
        assert _time.perf_counter() - start < 2.0

    def test_the_allof_fold_over_a_huge_value_is_bounded_on_the_parser_path(self) -> None:
        # The discriminator the 100k-member pin above cannot be: there the
        # mutant is caught by the validator-compilation cost tripping the
        # post-fast-path check (compile ~ fold for trivial members). Here
        # the members compile free (booleans) and the EXPENSE lives in the
        # fold itself: every Chain::True member re-runs
        # normalize_missing_values over the whole value (O(members x
        # value)), so with the walkers skipped the per-member ENTRY sample
        # in repair_value_d is the only reader that can bound it. Mutation-
        # verified: deleting the entry check walks all members (~seconds)
        # and the call RETURNS (validate's single entry tick never reaches
        # a 256th read), so pytest.raises is the discriminator.
        big = "[" + ",".join(str(i % 10) for i in range(300_000)) + "]"
        schema: dict[str, Any] = {"allOf": [True] * 4_000}
        import time as _time

        start = _time.perf_counter()
        with pytest.raises(TimeoutError):
            repair_json_loads(big, schema=schema, skip_json_loads=True, deadline_ms=50)
        # Unbounded the fold walks 4000 x ~2MB (~1-4s); the sampled entry
        # check latches within the first few hundred members (~a few
        # hundred MB walked). 1.5s clears both margins.
        assert _time.perf_counter() - start < 1.5

    def test_union_branch_clones_cannot_stampede_past_the_budget(self) -> None:
        # The forced union loop-head check bounds the loop past expiry to
        # a small slice of the clone+repair+validate work: measured on a
        # quiet box the 1ms budget burns inside the strict fast path and
        # the reported elapsed is a fraction of the plain parse (the
        # schema-less repair of the same 5MB value under a generous
        # deadline, ~7ms), where the unbounded failure mode (no forced
        # check at all) runs all 120 branch clones over the 5MB value,
        # ~2.8s, ~400x that parse unit. The gate is machine-scaled against
        # the parse unit measured on THIS runner (min-of-3; the bounded
        # call runs twice and the smaller reported elapsed wins, so a load
        # spike between the measurements cannot flip the ratio): 4x the
        # unit, measured behavior at ~0.4x and the unbounded mode ~400x.
        raw = '"' + "a" * 5_000_000 + '"'
        schema: dict[str, Any] = {"anyOf": [{"type": "integer"}] * 120}
        import time as _time

        def parse_unit_ms() -> float:
            start = _time.perf_counter()
            repair_json(raw, deadline_ms=60_000)
            return (_time.perf_counter() - start) * 1000.0

        def bounded_elapsed_ms() -> float:
            with pytest.raises(TimeoutError, match=r"elapsed (\d+\.\d)ms") as excinfo:
                repair_json_loads(raw, schema=schema, deadline_ms=1)
            return float(re.search(r"elapsed (\d+\.\d)ms", str(excinfo.value)).group(1))

        unit = min(parse_unit_ms() for _ in range(3))
        elapsed = min(bounded_elapsed_ms() for _ in range(2))
        assert elapsed < 4 * unit, (
            f"the union loop ran {elapsed:.0f}ms against a parse unit of "
            f"{unit:.0f}ms: the forced loop-head check is not bounding the clones"
        )

    def test_the_union_head_check_keeps_expired_branches_free(self) -> None:
        # Mutation pin for the union loop head's CHECK half (the force half
        # is pinned by the stampede test above): the head check must fire
        # BEFORE a branch's clone, not inside it. Removing only the check
        # (keeping the force) still aborts — one check deeper, at the
        # branch body's entry — but then EVERY remaining branch pays a full
        # value.clone() before its sticky check: 5000 branches x 2MB is
        # ~0.5-1s of pure clones past a 1ms budget. With the check, the
        # abort lands at the first head check (~the parse time).
        raw = '"' + "a" * 2_000_000 + '"'
        schema: dict[str, Any] = {"anyOf": [{"type": "integer"}] * 5_000}
        with pytest.raises(TimeoutError, match=r"elapsed (\d+\.\d)ms") as excinfo:
            repair_json_loads(raw, schema=schema, deadline_ms=1)
        elapsed = float(re.search(r"elapsed (\d+\.\d)ms", str(excinfo.value)).group(1))
        assert elapsed < 300.0

    def test_the_type_union_head_check_keeps_expired_branches_free(self) -> None:
        # The repair_type_union mirror of the pin above: removing only the
        # loop head's CHECK (keeping the force) still aborts one check
        # deeper, at the branch body's own forced coerce entry — but then
        # EVERY remaining kind pays a full value.clone() before that check:
        # 40000 kinds x 2MB is ~1s of pure clones past a 1ms budget. With
        # the check, the abort lands at the first head check (~the parse).
        raw = '"' + "a" * 2_000_000 + '"'
        schema: dict[str, Any] = {"type": ["integer"] * 40_000}
        with pytest.raises(TimeoutError, match=r"elapsed (\d+\.\d)ms") as excinfo:
            repair_json_loads(raw, schema=schema, deadline_ms=1)
        elapsed = float(re.search(r"elapsed (\d+\.\d)ms", str(excinfo.value)).group(1))
        assert elapsed < 300.0

    def test_the_type_union_branch_loop_is_bounded_by_the_deadline(self) -> None:
        # repair_type_union's loop head (force + check) with branch bodies
        # that fire no other check (coerce fails before validate, and the
        # kind count stays under the 256-sample period): deleting the head
        # check lets all 150 kinds clone the heavy value (~4s elapsed);
        # with it the abort lands at the first kind past expiry (~3ms).
        raw = '"' + "a" * 5_000_000 + '"'
        schema: dict[str, Any] = {"type": ["integer"] * 150}
        with pytest.raises(TimeoutError, match=r"elapsed (\d+\.\d)ms") as excinfo:
            repair_json_loads(raw, schema=schema, deadline_ms=1)
        elapsed = float(re.search(r"elapsed (\d+\.\d)ms", str(excinfo.value)).group(1))
        assert elapsed < 100.0

    def test_the_ladder_sweep_cannot_stampede_past_the_budget(self) -> None:
        # key_ladder's forced check bounds the loop past expiry to a
        # sampled slice of the sweep, not the whole of it: measured on a
        # quiet box the abort lands ~30 sweeps past the 1ms budget (each
        # sweep ~15ms, one O(properties) jaro pass over 1000 names of
        # 2000 chars), where the unbounded failure mode (no forced check
        # at all) runs all 500 keys, ~500 sweeps. The gate is
        # machine-scaled, never an absolute wall: one unknown key under a
        # generous deadline measures the sweep's unit cost on THIS runner,
        # the bounded abort must land under 100x that unit (measured ~30x,
        # a 3x margin; the unbounded mode is 500x, 5x past the gate), and
        # min-of-N on both sides keeps a load spike between the
        # measurements from flipping the ratio.
        props = {f"property_{i:06}" + "x" * 2000: {"type": "integer"} for i in range(1000)}
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": props,
        }
        one_key = '{"propertx_000000": 0}'
        raw = "{" + ",".join(f'"propertx_{i:06}": {i}' for i in range(500)) + "}"
        import time as _time

        def unit() -> float:
            start = _time.perf_counter()
            repair_json_loads(one_key, schema=schema, deadline_ms=60_000)
            return _time.perf_counter() - start

        def bounded() -> float:
            start = _time.perf_counter()
            with pytest.raises(TimeoutError):
                repair_json_loads(raw, schema=schema, deadline_ms=1)
            return _time.perf_counter() - start

        unit_cost = min(unit() for _ in range(3))
        elapsed = min(bounded() for _ in range(2))
        assert elapsed < 100 * unit_cost, (
            f"the ladder sweep ran {elapsed:.3f}s against a one-sweep unit of "
            f"{unit_cost:.3f}s ({elapsed / unit_cost:.0f}x): the forced "
            "loop-head check is not bounding the sweep"
        )
        assert elapsed < 60.0  # hang backstop, not a performance gate

    def test_the_key_walk_cannot_sweep_a_huge_object_past_the_budget(self) -> None:
        # The rename pass's per-key sample (the walker's top-of-function
        # check fires once per CALL, so it cannot bound a single object's
        # intra-call key loop): one flat object with 200k unknown keys
        # against a 1000-property schema is O(keys x properties) fold
        # scans; without the per-key sample the pass sweeps them all
        # (~0.6-2s) before any later check consults the clock, and the
        # wall blows the budget. With it the walk stops within ~256 keys
        # of the expiry and the call aborts via the sticky error. The gate
        # is machine-scaled against a 2000-key run of the same walk under
        # a generous deadline (the per-scan unit cost on THIS runner): the
        # bounded abort costs less than that run, the unbounded sweep is
        # ~100x it, and the gate sits at 10x with min-of-N on both sides.
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {f"property_{i:06}": {"type": "integer"} for i in range(1000)},
        }
        small = "{" + ",".join(f'"propertx_{i:06}": {i}' for i in range(2_000)) + "}"
        raw = "{" + ",".join(f'"propertx_{i:06}": {i}' for i in range(200_000)) + "}"
        import time as _time

        def unit() -> float:
            start = _time.perf_counter()
            repair_json_loads(small, schema=schema, deadline_ms=60_000)
            return _time.perf_counter() - start

        def bounded() -> float:
            start = _time.perf_counter()
            with pytest.raises(TimeoutError):
                repair_json_loads(raw, schema=schema, deadline_ms=1)
            return _time.perf_counter() - start

        unit_cost = min(unit() for _ in range(3))
        elapsed = min(bounded() for _ in range(2))
        assert elapsed < 10 * unit_cost, (
            f"the key walk ran {elapsed:.3f}s against a 2000-key unit of "
            f"{unit_cost:.3f}s: the per-key sample is not bounding the sweep"
        )
        assert elapsed < 60.0  # hang backstop, not a performance gate

    @pytest.mark.parametrize(
        "items_schema",
        [{"type": "string"}, {"type": "string", "format": "date-time"}],
        ids=["plain-strings", "date-strings"],
    )
    def test_the_schema_array_walks_sample_the_clock(self, items_schema: dict[str, Any]) -> None:
        # The advisory walkers (normalize_keys, prenormalize_dates,
        # suggest_scan) recursed through arrays of scalars with no clock
        # read at all: a million-item array walked to completion (~0.2-0.3s)
        # under a 1ms budget and the call returned. The per-call sample at
        # the top of each walker latches the sticky error mid-walk, and the
        # next phase-boundary check (is_valid's entry) turns it into the
        # TimeoutError.
        schema: dict[str, Any] = {"type": "array", "items": items_schema}
        raw = (
            "["
            + ",".join(
                '"2024-01-01T00:00:00Z"' if items_schema.get("format") else f'"s{i}"'
                for i in range(500_000)
            )
            + "]"
        )
        import time as _time

        start = _time.perf_counter()
        with pytest.raises(TimeoutError):
            repair_json_loads(raw, schema=schema, deadline_ms=1)
        assert _time.perf_counter() - start < 2.0

    def test_the_date_prenormalize_walk_stops_within_the_budget(self) -> None:
        # Tightness pin for prenormalize_dates' own top-of-call sample
        # (mutation-verified: the array-walks pin above passes even with
        # the check deleted, because normalize_keys' identical sample
        # covers the same call graph — but then the bound is one FULL date
        # walk, not one sample period). The walk is made expensive per
        # item without growing the input: 100 allOf members per item, each
        # re-run through the date normalizer (~70us/item), so the unchecked
        # walk of 100k items is ~7s while the sampled walk latches within
        # its first ~1ms.
        schema: dict[str, Any] = {
            "type": "array",
            "items": {"allOf": [{"type": "string", "format": "date-time"}] * 100},
        }
        raw = "[" + ",".join('"2024-01-01T00:00:00Z"' for _ in range(100_000)) + "]"
        import time as _time

        start = _time.perf_counter()
        with pytest.raises(TimeoutError):
            repair_json_loads(raw, schema=schema, deadline_ms=1)
        assert _time.perf_counter() - start < 0.5

    def test_completed_schema_work_past_the_budget_is_returned(self) -> None:
        # The soft-semantics contract on the schema path: work that
        # COMPLETES past an expired budget still returns its answer (the
        # deadline stops further work, it does not nullify done work). A
        # 10k-property schema burns the 1ms budget compiling/walking, but
        # the sampled-call count stays under the 256 period, so no clock
        # read ever fires and the fast path returns the valid value.
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {f"p{i:06}": {"type": "integer"} for i in range(10_000)},
        }
        assert repair_json_loads("{}", schema=schema, deadline_ms=1) == {}

    def test_validate_on_a_huge_valid_document_is_bounded(self) -> None:
        # A huge valid document under a tiny budget: the key walks sample
        # the clock, latch the sticky error, and the call aborts well under
        # the wall (measured ~40ms) instead of paying the full validation.
        # Under a budget sized for the document, the same call completes.
        big = "[" + ",".join(f'{{"k{i}": {i}}}' for i in range(200_000)) + "]"
        schema: dict[str, Any] = {"type": "array", "items": {"type": "object"}}
        with pytest.raises(TimeoutError):
            repair_json_loads(big, schema=schema, deadline_ms=1)
        assert repair_json_loads(big, schema=schema, deadline_ms=5_000) == repair_json_loads(
            big, schema=schema
        )

    def test_the_wide_union_validation_gate_is_bounded(self) -> None:
        # H2 escape, fixed: the fast path's is_valid rode the crate's
        # ERROR-CONSTRUCTING validate, and a wide failing union constructs
        # one ValidationError per branch with the instance cloned into
        # each: 5000 branches x a 2MB string measured ~7s INSIDE one
        # validate() call, between sampled checks, under any budget (the
        # crate's boolean is_valid is ~0ms on the same instance). is_valid
        # now rides the boolean API and validate() is a force point (the
        # clock is read before the opaque crate call), so the gate aborts
        # at the budget instead.
        raw = '"' + "a" * 2_000_000 + '"'
        schema: dict[str, Any] = {"anyOf": [{"type": "integer"}] * 5_000}
        for budget in (1, 100):
            import time as _time

            start = _time.perf_counter()
            with pytest.raises(TimeoutError):
                repair_json_loads(raw, schema=schema, deadline_ms=budget)
            assert _time.perf_counter() - start < 1.0
        # A wide union the value DOES satisfy still validates (the boolean
        # path changed no outcome).
        assert repair_json_loads("123", schema={"anyOf": [{"type": "integer"}] * 5_000}) == 123
        # And the error message is preserved for ordinary failing shapes.
        with pytest.raises(ValueError, match="anyOf"):
            repair_json_loads('"abc"', schema={"anyOf": [{"type": "integer"}] * 3})

    def test_suggest_scan_honors_the_budget(self) -> None:
        # The diagnostics spelling on a valid document with a heavy ladder:
        # bounded, the call aborts (the walks' sticky error surfaces; the
        # suggest-only scan itself soft-stops but can no longer mask the
        # expiry), and unbounded it reports a hint for every unknown key
        # with the output unchanged.
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": True,
            "properties": {f"property_{i:06}": {"type": "integer"} for i in range(300)},
        }
        raw = "{" + ",".join(f'"propertx_{i:06}": {i}' for i in range(5_000)) + "}"
        with pytest.raises(TimeoutError):
            repair_json_diagnostics(raw, schema=schema, deadline_ms=1)
        value, hints = repair_json_diagnostics(raw, schema=schema)
        assert len(hints) == 5_000
        assert value == json.loads(raw)

    @pytest.mark.parametrize(
        "name,call",
        [
            ("repair_json", repair_json),
            ("repair_json_loads", repair_json_loads),
            ("repair_json_diagnostics", repair_json_diagnostics),
        ],
    )
    def test_the_schema_path_timeout_message_names_the_called_spelling(
        self, name: str, call
    ) -> None:
        # H3: the schema layer's abort rides the same DEADLINE_TAG payload
        # and the same TimeoutError translation as the parser's, fronted
        # with the called spelling's name (the ladder shape aborts on the
        # schema layer's own checks under a 1ms budget).
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {f"property_{i:06}": {"type": "integer"} for i in range(1000)},
        }
        raw = "{" + ",".join(f'"propertx_{i:06}": {i}' for i in range(20_000)) + "}"
        with pytest.raises(
            TimeoutError,
            match=rf"^{name} deadline exceeded: elapsed \d+\.\dms > deadline_ms 1\.0ms$",
        ):
            call(raw, schema=schema, deadline_ms=1)

    def test_deadline_state_does_not_leak_across_calls(self) -> None:
        # H5: the SchemaRepairer (and its deadline triple) is per-call: a
        # call that timed out must not poison the next one, in either
        # order.
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {f"property_{i:06}": {"type": "integer"} for i in range(1000)},
        }
        raw = "{" + ",".join(f'"propertx_{i:06}": {i}' for i in range(20_000)) + "}"
        with pytest.raises(TimeoutError):
            repair_json_loads(raw, schema=schema, deadline_ms=1)
        assert repair_json_loads('{"property_000001": 5}', schema=schema) == {
            "property_000001": 5
        }
        # And a completing call before a timing-out one leaves the clock
        # armed and the abort intact.
        assert repair_json_loads('{"property_000001": 5}', schema=schema) == {
            "property_000001": 5
        }
        with pytest.raises(TimeoutError):
            repair_json_loads(raw, schema=schema, deadline_ms=1)

    def test_a_generous_deadline_does_not_change_date_and_union_shapes(self) -> None:
        # H4 parity on the schema-heavy shapes: date pre-normalization,
        # union retry, and the allOf fold produce byte-identical output
        # under a generous deadline.
        date_schema: dict[str, Any] = {
            "type": "array",
            "items": {"type": "string", "format": "date-time"},
        }
        date_raw = "[" + ",".join('"2024-01-01T00:00:00Z"' for _ in range(50_000)) + "]"
        assert repair_json_loads(
            date_raw, schema=date_schema, deadline_ms=60_000
        ) == repair_json_loads(date_raw, schema=date_schema)

        union_schema: dict[str, Any] = {
            "anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "object"}],
        }
        assert repair_json_loads(
            '{"a": 1}', schema=union_schema, deadline_ms=60_000
        ) == repair_json_loads('{"a": 1}', schema=union_schema)

        all_of_schema: dict[str, Any] = {
            "allOf": [{"properties": {"y": {"type": "integer"}}}, {"type": "object"}],
        }
        assert repair_json_loads(
            '{"y": "7"}', schema=all_of_schema, deadline_ms=60_000
        ) == repair_json_loads('{"y": "7"}', schema=all_of_schema)


class TestEnumSuggestionDeadline:
    """The enum suggestion loop is ON the clock (issue #115): the pre-fix
    walk scored every member with an unbounded per-comparison budget and
    never read the clock, so a wide enum of long members answered in 1.3s
    against a 5ms ``deadline_ms`` (the TimeoutError arriving only from a
    later phase's check). Both axes are bounded now — a forced clock read
    per member, and the clock's remaining budget handed to each
    jaro-winkler comparison — and an expired clock RAISES, never falls out
    of the loop as a silent ``None`` suggestion that would mask the
    timeout as a plain data error. The wall pins below carry generous
    room (50ms at a 5ms budget, measured ~8ms) for CI load."""

    _WIDE_ENUM = [("a" * 1500) + str(i) for i in range(2000)]

    def test_the_reported_wide_enum_raises_within_the_budget(self) -> None:
        import time

        started = time.perf_counter()
        with pytest.raises(TimeoutError, match="deadline"):
            repair_json_loads(
                json.dumps("b" * 1500),
                schema={"enum": self._WIDE_ENUM},
                deadline_ms=5,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        assert elapsed_ms < 50.0, f"the enum walk ran {elapsed_ms:.1f}ms past a 5ms budget"

    def test_a_budget_expired_before_the_enum_still_raises(self) -> None:
        # The ordering red-team: expiry before the loop and expiry inside
        # it must BOTH raise (a pre-loop-expired clock that returned None
        # would surface the miss as "does not match enum" — the timeout,
        # masked). The 1ms budget is blown in the parse phases; the enum
        # loop's forced check raises anyway.
        with pytest.raises(TimeoutError, match="deadline"):
            repair_json_loads(
                json.dumps("b" * 1500),
                schema={"enum": self._WIDE_ENUM},
                deadline_ms=1,
            )

    def test_a_single_million_char_member_is_bounded_by_the_comparison_budget(self) -> None:
        # The one-very-long-comparison axis: the remaining-budget handoff
        # into jaro_winkler bounds the member's materialization and scan.
        # The member is sized so its own materialization must overrun a
        # 5ms budget (measured: TimeoutError at ~6-8ms; a 10^6-char member
        # sometimes fits the budget and answers the plain miss, which is
        # the bound working, not failing). The miss sits where the
        # suggestion loop runs — an object property's enum, the path that
        # scores members.
        import time

        schema = {
            "type": "object",
            "properties": {"c": {"enum": ["a" * 10**7]}},
            "required": ["c"],
        }
        started = time.perf_counter()
        with pytest.raises(TimeoutError, match="deadline"):
            repair_json_loads('{"c": "b"}', schema=schema, deadline_ms=5)
        elapsed_ms = (time.perf_counter() - started) * 1000
        assert elapsed_ms < 100.0, f"one 10^7-char member ran {elapsed_ms:.1f}ms"

    def test_an_affordable_enum_suggests_identically_armed_or_not(self) -> None:
        # The suggestion-hint path is byte-identical with the clock armed
        # and unarmed: the remaining-budget handoff only ever cuts work at
        # real expiry, so a healthy enum cannot tell the difference (the
        # exact suggestion string is pinned both spellings).
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"color": {"type": "string", "enum": ["blue", "green"]}},
            "required": ["color"],
        }
        expected = "Did you mean 'blue'?"
        with pytest.raises(ValueError, match=re.escape(expected)):
            repair_json_loads('{"color": "blu"}', schema=schema)
        with pytest.raises(ValueError, match=re.escape(expected)):
            repair_json_loads('{"color": "blu"}', schema=schema, deadline_ms=60_000)

    def test_an_empty_enum_misses_plainly(self) -> None:
        # No members, nothing to score, and no deadline interaction: the
        # loop body never runs, so the miss is the plain refusal (the
        # top-level scalar shape answers from the crate's own enum
        # validation; the object-property shape is the suggestion loop's).
        schema = {
            "type": "object",
            "properties": {"c": {"enum": []}},
            "required": ["c"],
        }
        with pytest.raises(ValueError, match="does not match enum"):
            repair_json_loads('{"c": "x"}', schema=schema, deadline_ms=60_000)
