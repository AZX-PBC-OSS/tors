"""tors-native behavior pins for the JSON-repair surface.

These tests pin tors's OWN extensions beyond upstream ``json_repair`` — no
upstream parity is asserted here. For context, upstream is ``json_repair``
0.63.4 by Stefano Baccianella (MIT,
https://github.com/mangiucugna/json_repair); tors's divergences from it are
the documented classes in ``DESIGN-json-repair-port.md`` §9 and are re-pinned
below as intentional behavior, not parity cases.
"""

from __future__ import annotations

import json
import re
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
    """Fence pre-pass (DESIGN §7): one wrapping fence unwraps before repair."""

    def test_backtick_json_fence_unwraps(self) -> None:
        assert repair_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_tilde_json_fence_unwraps(self) -> None:
        assert repair_json('~~~json\n[1, 2]\n~~~') == '[1, 2]'

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
    """Key-normalization ladder + fuzzy remap tier (DESIGN §6.0–6.1)."""

    @pytest.mark.parametrize(
        "typo", ["First Name", "first-name", "FIRST_NAME", "firstname"]
    )
    def test_normalization_ladder_remaps_case_separator_variants(
        self, typo: str
    ) -> None:
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
        # The two-tier split: the MECHANICAL fold tier remaps even on
        # permissive schemas (deterministic match, data would otherwise be
        # stranded on a dead key); the FUZZY tier is a guess and stays
        # gated — "nam" against a permissive, non-required "name" keeps the
        # key and only suggests.
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": True,
        }
        mechanical, diags = repair_json_diagnostics(
            '{"Na Me": "Ada"}', schema=schema
        )
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
        assert repair_json_loads(
            '{"First Name": "Ada", "age": 30}', schema=schema
        ) == {"first_name": "Ada", "age": 30}
        # And the kebab/spellings family through the same lane.
        assert repair_json_loads(
            '{"first-name": "Ada", "age": 30}', schema=schema
        ) == {"first_name": "Ada", "age": 30}


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
    """Date/datetime normalization (DESIGN §6.3)."""

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
        value, diags = repair_json_diagnostics(
            json.dumps("2024/03/15"), schema=_DATE_SCHEMA
        )
        assert value == "2024-03-15"
        assert any(d["action"] == "format_date" for d in diags)

    def test_already_normalized_date_has_no_diagnostic(self) -> None:
        value, diags = repair_json_diagnostics(
            json.dumps("2024-03-15"), schema=_DATE_SCHEMA
        )
        assert value == "2024-03-15"
        assert all(d["action"] != "format_date" for d in diags)

    def test_ambiguous_numeric_date_is_unchanged_with_suggest(self) -> None:
        value, diags = repair_json_diagnostics(
            json.dumps("03/04/2024"), schema=_DATE_SCHEMA
        )
        assert value == "03/04/2024"
        assert any(d["action"] == "suggest" for d in diags)

    def test_invalid_calendar_date_is_unchanged(self) -> None:
        assert (
            repair_json_loads(json.dumps("2024-02-30"), schema=_DATE_SCHEMA)
            == "2024-02-30"
        )

    @pytest.mark.parametrize(
        ("raw", "want"),
        [
            ("2024-03-15 14:30", "2024-03-15T14:30:00"),
            # Offset-bearing input normalizes to its UTC INSTANT (jiff's
            # rendering): 14:30+05:30 IS 09:00Z.
            ("2024-03-15T14:30:00+0530", "2024-03-15T09:00:00Z"),
            ("2024-03-15T14:30:00Z", "2024-03-15T14:30:00Z"),
            ("2024-03-15T14:30:00.123Z", "2024-03-15T14:30:00.123Z"),
        ],
    )
    def test_datetime_forms_normalize(self, raw: str, want: str) -> None:
        assert repair_json_loads(json.dumps(raw), schema=_DATETIME_SCHEMA) == want

    def test_time_format_normalizes(self) -> None:
        # format: time — seconds always present, no offset invented.
        schema: dict[str, Any] = {"type": "string", "format": "time"}
        assert repair_json_loads(json.dumps("14:30"), schema=schema) == "14:30:00"
        assert repair_json_loads(json.dumps("14:30:00"), schema=schema) == "14:30:00"

    def test_uuid_format_lowercases(self) -> None:
        # format: uuid — shape-gated canonical lowercase; non-uuids pass
        # through untouched for validation to judge.
        schema: dict[str, Any] = {"type": "string", "format": "uuid"}
        assert (
            repair_json_loads(json.dumps("A1B2C3D4-0000-1111-2222-333344445555"), schema=schema)
            == "a1b2c3d4-0000-1111-2222-333344445555"
        )
        assert repair_json_loads(json.dumps("not a uuid"), schema=schema) == "not a uuid"


class TestCommaSplit:
    """Comma-split array recovery (DESIGN §6.1b)."""

    def test_comma_separated_string_splits_to_typed_array(self) -> None:
        # §9.7 divergence class: upstream raises here, while tors's
        # validation-gated split wins because [1, 2, 3] validates against the
        # items-integer schema.
        assert repair_json_loads('{"items": "1, 2, 3"}', schema=_ITEMS_SCHEMA) == {
            "items": [1, 2, 3]
        }

    def test_comma_free_string_keeps_upstream_wrap_or_raise(self) -> None:
        # No commas, so the split cannot win: whichever the corpus pins
        # (wrap-singleton or raise) stands — accept either without re-pinning
        # the corpus case itself.
        try:
            got = repair_json_loads('{"items": "not json"}', schema=_ITEMS_SCHEMA)
        except ValueError:
            pass
        else:
            assert got == {"items": ["not json"]}


class TestSeparators:
    def test_digit_group_separators_are_stripped_for_integers(self) -> None:
        assert repair_json_loads('{"count": "1,234"}', schema=_COUNT_SCHEMA) == {
            "count": 1234
        }
        assert repair_json_loads(
            '{"count": "82_461_110"}', schema=_COUNT_SCHEMA
        ) == {"count": 82461110}


class TestDiagnosticsShape:
    """Structured-diagnostics contract (DESIGN §6.4)."""

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
        _, diags = repair_json_diagnostics(
            json.dumps("03/04/2024"), schema=_DATE_SCHEMA
        )
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
    @pytest.mark.parametrize(
        "fn", [repair_json, repair_json_loads, repair_json_diagnostics]
    )
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
        # are accepted — see the e2e suite).
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
        # The suggestion names the DISCARDED reading's override — suggesting
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
        # integer — the declared type alone resolves it.
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
        # integer on integer fields and a FLOAT on number fields.
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
    """Python's unbounded int() semantics — never a saturating cast."""

    _INT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
    }

    def test_big_integer_strings_stay_exact(self) -> None:
        raw = '{"n": "12345678901234567890123"}'
        assert repair_json_loads(raw, schema=self._INT_SCHEMA) == {
            "n": 12345678901234567890123
        }

    def test_big_integral_floats_convert_to_their_exact_decimal(self) -> None:
        # int(1e30) in Python is the EXACT value of the binary float.
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
        # '/x' chains parse_json <-> parse_comment without unwinding — 2
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
        # and silently change behavior on inputs upstream handles — the exact
        # parity-risk class this guard is scoped to avoid.
        payload = '{"a":1}' + "".join(f', "k{i}":1}}' for i in range(150))
        merged = repair_json_loads(payload, skip_json_loads=True)
        assert merged == {"a": 1, **{f"k{i}": 1 for i in range(150)}}

    def test_merged_array_continuation_chains_raise_instead_of_crashing(self) -> None:
        # `{"a":[0],` + `["b":[0],` * N nests through the array-continuation
        # merge: a '[' at the key position merges into the previous
        # array-valued member, and the merged array's first item — a string
        # followed by ':' — is a missing object start parsed by parse_object
        # directly, whose key scan sees another '[' and merges again. That
        # cycle had no depth guard anywhere on it: it grew the native stack
        # per fragment and overflowed — an uncatchable SIGSEGV around 8k
        # fragments (main thread; fewer on worker-sized stacks) — instead of
        # the documented catchable ValueError. The continuation guard caps
        # it like every other deep-recursion path.
        payload = '{"a":[0],' + '["b":[0],' * 2_000 + '1]'
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
            repair_json(
                payload, schema={"type": "object"}, salvage=True, skip_json_loads=True
            )

    def test_merged_array_continuations_below_the_cap_still_merge(self) -> None:
        # The guard must fire only past MAX_NESTING, never on an ordinary
        # merge chain: nested chains below the cap still merge every
        # fragment (an over-counting regression would raise early), and
        # same-level sequential merges never accrue depth at all —
        # enter/leave is balanced per continuation.
        assert repair_json_loads(
            '{"a":[0],["b":[0],["b":[0],1]', skip_json_loads=True
        ) == {"a": [0, {"b": [0, {"b": [0], "1": ""}]}]}
        expected: dict[str, Any] = {"b": [0], "1": ""}
        for _ in range(149):
            expected = {"b": [0, expected]}
        payload = '{"a":[0],' + '["b":[0],' * 150 + '1]'
        assert repair_json_loads(payload, skip_json_loads=True) == {"a": [0, expected]}
        assert repair_json_loads('{"a":[1], [2], [3]}', skip_json_loads=True) == {
            "a": [1, 2, 3]
        }

    def test_continuation_chains_cap_at_max_nesting_exactly(self) -> None:
        # Both continuation recursions share the MAX_NESTING budget with
        # structural nesting. The comma chain spends 1 (the initial `{`) +
        # 1 per fragment (scalar values add nothing): 199 fragments parse
        # (depth 200), the 200th raises. The array-merge chain spends the
        # same 1 + 1 per fragment PLUS 1 for the innermost fragment's
        # `[0]` value (a container nested inside every merge): 198
        # fragments parse, the 199th raises. Pinning the exact edges
        # catches future accounting drift in either direction —
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
        merge_ok = '{"a":[0],' + '["b":[0],' * 198 + '1]'
        assert repair_json_loads(merge_ok, skip_json_loads=True) == {"a": [0, expected]}
        with pytest.raises(
            ValueError, match="Input nesting exceeds the supported parser recursion depth"
        ):
            repair_json('{"a":[0],' + '["b":[0],' * 199 + '1]', skip_json_loads=True)

    def test_related_recursion_shapes_route_through_guarded_edges(self) -> None:
        # Siblings of the continuation chains that DO pass guarded edges on
        # every cycle: string-colon objects nested inside arrays (`["b": [`
        # per level, each through parse_json's '[' branch) and salvage-mode
        # comma-merging (every salvage fragment re-enters parse_json).
        # Pinning them keeps a future refactor from quietly rerouting these
        # shapes past the guards.
        with pytest.raises(
            ValueError, match="Input nesting exceeds the supported parser recursion depth"
        ):
            repair_json('[' + '"b": [' * 2_000, skip_json_loads=True)
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
        assert repair_json_loads(
            '{"a":[0],["b":[0],1]', strict=True, skip_json_loads=True
        ) == {"a": [0], "b": [0]}

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
        assert repair_json_loads('{"First Name": "Ada"}', schema=schema) == {
            "first_name": "Ada"
        }


class TestDedup:
    """CPython set/dict.fromkeys collapses the reviewer's duplicate shapes."""

    def test_set_object_collapses_duplicate_members(self) -> None:
        assert repair_json_loads(
            "{'a', 'b', 'a'}", schema={"type": "object"}, salvage=True
        ) == {"a": None, "b": None}

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
        # rolled back — the diagnostics describe the returned value only.
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
        # jiff's ISO parser is strict on padding: "2024-3-5" stays
        # untouched for validation to judge.
        assert repair_json_loads(json.dumps("2024-3-5"), schema=_DATE_SCHEMA) == "2024-3-5"


class TestDiagnosticsVocabularyCompleteness:
    """Every §6.4 action fires somewhere — the vocabulary is real."""

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
        # fill: a MISSING value (a member whose value position holds a
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
        # A string-typed schema legitimately accepts "" — the raise is the
        # typed-schema behavior, not an unconditional one.
        assert repair_json_loads("no JSON anywhere", schema={"type": "string"}) == ""
