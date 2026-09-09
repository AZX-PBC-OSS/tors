"""Tors-native end-to-end suite for the pydantic round-trip.

Provenance: this file is tors-native (no upstream twin). It proves the
real-world agent pattern end to end: a pydantic v2 model goes straight into
``schema=`` (the runtime calls ``Model.model_json_schema()`` itself),
damaged LLM output is repaired against it, and the result round-trips
through ``Model.model_validate``. Upstream ``json_repair`` 0.63.4 by Stefano
Baccianella (MIT, https://github.com/mangiucugna/json_repair) is context
only: its ``schema=`` accepts pydantic v2 models too, but the agent
round-trip pattern is tors-native scope (tors additionally injects field
defaults/default-factories into the generated schema and skips the phantom
alias-name entries upstream creates — see DESIGN §9). Contract:
DESIGN-json-repair-port.md §§4, 6, 8.
"""

from __future__ import annotations

import enum
import json
from typing import Any, Literal

import pytest

from tors import repair_json, repair_json_diagnostics, repair_json_loads

pydantic = pytest.importorskip("pydantic")


class ProductReview(pydantic.BaseModel):
    """Typical single-object agent payload: rating coerces, rest default.

    `extra="forbid"` is the strict-agent convention (the generated schema
    carries `additionalProperties: false`) and is what makes the key-repair
    ladder remap unknown spellings instead of keeping them as extras.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    name: str
    rating: int = pydantic.Field(ge=1, le=5)
    tags: list[str] = pydantic.Field(default_factory=list)
    verified: bool = False
    review_title: str = ""


class LineItem(pydantic.BaseModel):
    """One nested row of the order payload."""

    sku: str
    qty: int
    price: float


class Order(pydantic.BaseModel):
    """Typical nested agent payload: a list of models plus a total."""

    order_id: str
    items: list[LineItem]
    total: float


class ColorPick(pydantic.BaseModel):
    """Single-Literal model, for the enum-suggestion path."""

    color: Literal["red", "green", "blue"]


class SimpleNote(pydantic.BaseModel):
    """One bare required str: no safe salvage fill exists for it."""

    note: str


_DIAGNOSTIC_ACTIONS: frozenset[str] = frozenset(
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


class TestModelAsSchema:
    def test_model_accepted_directly(self) -> None:
        # Pattern: the model class goes straight into schema=, no manual call.
        value = repair_json_loads('{"name": "x", "rating": "4"}', schema=ProductReview)
        assert value == {
            "name": "x",
            "rating": 4,
            "tags": [],
            "verified": False,
            "review_title": "",
        }
        model = ProductReview.model_validate(value)
        assert model.rating == 4
        assert isinstance(model.rating, int)

    def test_fenced_model_output(self) -> None:
        # Pattern: prose + one fence still repairs against the model.
        raw = 'Here is your review:\n```json\n{"name": "x", "rating": "4",}\n```\nHope this helps!'
        value = repair_json_loads(raw, schema=ProductReview)
        model = ProductReview.model_validate(value)
        assert model.name == "x"
        assert model.rating == 4

    def test_unfenced_prose_output(self) -> None:
        # Pattern: prose prefix + truncated JSON is completed and coerced.
        raw = 'Sure! {"name": "x", "rating": 5, "tags": ["a", "b"]'
        value = repair_json_loads(raw, schema=ProductReview)
        model = ProductReview.model_validate(value)
        assert model.rating == 5
        assert model.tags == ["a", "b"]
        assert model.verified is False


class TestKeyRepair:
    def test_key_typos_fixed_then_validated(self) -> None:
        # Pattern: case typos remap first, then the model validates.
        value = repair_json_loads('{"Name": "x", "Rating": "3"}', schema=ProductReview)
        assert value["name"] == "x"
        assert value["rating"] == 3
        ProductReview.model_validate(value)
        # Pattern: the kebab variant folds onto the snake_case field.
        kebab = repair_json_loads(
            '{"name": "x", "rating": 4, "review-title": "T"}',
            schema=ProductReview,
        )
        assert kebab["review_title"] == "T"
        ProductReview.model_validate(kebab)


class TestNestedPayloads:
    def test_nested_order(self) -> None:
        # Pattern: deep scalars coerce inside nested models, fenced or not.
        bare = '{order_id: "A-1", items: [{sku: "w-1", qty: "2", price: 3.5},], total: "19.99",}'
        fenced = "```json\n" + bare + "\n```"
        for raw in (bare, fenced):
            value = repair_json_loads(raw, schema=Order)
            assert value == {
                "order_id": "A-1",
                "items": [{"sku": "w-1", "qty": 2, "price": 3.5}],
                "total": 19.99,
            }
            order = Order.model_validate(value)
            assert order.total == 19.99
            assert isinstance(order.total, float)
            assert isinstance(order.items[0], LineItem)
            assert order.items[0].qty == 2
            assert isinstance(order.items[0].qty, int)


class TestSchemaErrors:
    def test_enum_suggestion_e2e(self) -> None:
        # Pattern: a near-miss Literal value errors with a "Did you mean".
        with pytest.raises(ValueError, match="Did you mean"):
            repair_json_loads('{"color": "blu"}', schema=ColorPick)

    def test_missing_required_reported(self) -> None:
        # Pattern: a bare required str cannot be filled, even via salvage.
        with pytest.raises(ValueError, match="Missing required properties"):
            repair_json_loads("{}", schema=SimpleNote)
        with pytest.raises(ValueError, match="Missing required properties"):
            repair_json_loads("{}", schema=SimpleNote, salvage=True)


class TestObservability:
    def test_diagnostics_tell_the_story(self) -> None:
        # Pattern: diagnostics narrate the same repair that loads returns.
        raw = 'Here is your review:\n```json\n{"name": "x", "rating": "4",}\n```\nHope this helps!'
        value, diags = repair_json_diagnostics(raw, schema=ProductReview)
        assert value == repair_json_loads(raw, schema=ProductReview)
        assert any(d["action"] == "coerce" for d in diags)
        assert all(d["action"] in _DIAGNOSTIC_ACTIONS for d in diags)

    def test_str_spelling_roundtrip(self) -> None:
        # Pattern: the str spelling serializes exactly the loads value.
        raw = 'Here is your review:\n```json\n{"name": "x", "rating": "4",}\n```\nHope this helps!'
        text = repair_json(raw, schema=ProductReview)
        assert json.loads(text) == repair_json_loads(raw, schema=ProductReview)


class TestModelEdgeCases:
    """Pydantic model edges at the schema boundary."""

    def test_v1_style_model_is_rejected_gracefully(self) -> None:
        class V1Style:
            """Duck-typed: has .schema() (the v1 spelling), no v2 machinery."""

            @classmethod
            def schema(cls) -> dict[str, Any]:  # pragma: no cover - never called
                return {"type": "object"}

        with pytest.raises(ValueError, match="pydantic v2 model"):
            repair_json_loads('{"a": 1}', schema=V1Style)  # type: ignore[arg-type]

    def test_aliased_field_default_pins_the_tors_divergence(self) -> None:
        # Upstream's schema_from_input creates a phantom entry for the
        # FIELD NAME next to the alias; tors injects the default under the
        # ALIAS only (the name validators actually accept) — DESIGN §9.
        class Aliased(pydantic.BaseModel):
            name: str = pydantic.Field(default="x", alias="userName")

        assert repair_json_loads("{", schema=Aliased, skip_json_loads=True) == {"userName": "x"}

    def test_enum_member_default_uses_the_member_value(self) -> None:
        class Color(enum.Enum):
            RED = "red"

        class Pick(pydantic.BaseModel):
            c: Color = Color.RED

        # The Enum-member default is injected as its member VALUE (the
        # JSON spelling), not the member's repr.
        assert repair_json_loads("{", schema=Pick, skip_json_loads=True) == {"c": "red"}
