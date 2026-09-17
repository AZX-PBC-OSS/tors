"""Contract gate for ``repair_json`` under schema hostility: a schema that
mutates itself (or raises) mid-walk must surface a normal, catchable
Python outcome, never an interpreter panic.

The ``schema=`` argument is walked into the repairer's own tree before the
repair runs, and one step of that walk calls back into Python: an ``int``
beyond i64 is spelled through ``PyObject_Str``, which resolves a SUBCLASS's
``__str__``. That hook is arbitrary user code: it can mutate the very
schema dict being walked, which used to trip pyo3's hard ``panic!`` in
``PyDictIterator`` (``dictionary changed size during iteration``),
surfacing as ``pyo3_runtime.PanicException``: a ``BaseException`` subclass
no ``except Exception`` can catch. The contract now: the walk snapshots
the dict's pairs before processing any of them (the ``_borrow.rs``
collect-handles-then-borrow shape), so a hook's mutation is invisible to
the walk (snapshot semantics, the same posture the ``content_hash``
walk documents for its subclass lane), and a hook that raises propagates
its own ordinary exception.

A second gate lives here because it walks the same hostile boundary: a
type-union schema whose string branch carries ``maxLength`` must validate
each union branch with ITS OWN compiled validator. The per-branch schema
is synthesized per iteration as a stack local, whose address the next
iteration's branch reuses; an address-keyed validator cache served the
first branch's validator to the second (a valid integer rejected with
``42 is not of type "string"``). The correct behavior: valid data under a
``["string","integer"]`` union repairs through the integer branch, and
invalid data still raises the ordinary ``ValueError``.

All hostility here is pure Python: cheap, bounded, in-process.
"""

from __future__ import annotations

from typing import Any

import pytest

from tors import repair_json, repair_json_diagnostics, repair_json_loads

# The union schema the branch-aliasing gate is about: the string branch's
# maxLength must never reach the integer branch's validator.
_UNION_MAXLEN = {"type": ["string", "integer"], "maxLength": 1}
_UNION_ROOT = {
    "type": "object",
    "properties": {"a": _UNION_MAXLEN, "b": {"type": "integer"}},
    "required": ["a"],
}


class TestSchemaMutatingStrHooks:
    """A schema value's ``__str__`` mutating the schema dict mid-walk:
    the walk completes with snapshot semantics, no panic."""

    def test_str_popping_from_the_schema_dict_is_tolerated(self) -> None:
        # The canonical repro's shape: the BigInt spelling runs __str__, which
        # pops a not-yet-walked key out of the schema dict.
        schema: dict[str, Any] = {}

        class Big(int):
            def __str__(self) -> str:
                schema.pop("b", None)
                return "0"

        schema["a"] = Big(2**70)
        schema["b"] = 1
        assert repair_json("{}", schema=schema) == "{}"
        # The caller's dict really was mutated by its own hook (the walk
        # tolerates it; the hook's side effect is the hook's business).
        assert schema == {"a": Big(2**70)}

    def test_str_growing_the_schema_dict_is_tolerated(self) -> None:
        # The mirror shape: the hook ADDS a key mid-walk. Snapshot
        # semantics: the added key is invisible to this call.
        schema: dict[str, Any] = {}

        class Grow(int):
            def __str__(self) -> str:
                schema["z"] = 9
                return "0"

        schema["a"] = Grow(2**70)
        assert repair_json("{}", schema=schema) == "{}"

    def test_str_raising_propagates_as_an_ordinary_exception(self) -> None:
        # A hook that raises surfaces ITS exception (a normal
        # RuntimeError any ``except Exception`` catches), not a
        # PanicException.
        class Boom(int):
            def __str__(self) -> str:
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            repair_json("{}", schema={"a": Boom(2**70)})

    def test_the_loads_and_diagnostics_spellings_share_the_walk(self) -> None:
        # All three spellings go through the same schema walk; none may
        # panic on the same hostile schema.
        schema: dict[str, Any] = {}

        class Big(int):
            def __str__(self) -> str:
                schema.pop("b", None)
                return "0"

        schema["a"] = Big(2**70)
        schema["b"] = 1
        assert repair_json_loads("{}", schema=schema) == {}
        value, diagnostics = repair_json_diagnostics("{}", schema=dict(schema))
        assert value == {}
        assert diagnostics == []


class TestTypeUnionMaxLengthValidatesEachBranch:
    """A ``["string","integer"]`` union with a string-branch ``maxLength``:
    valid integer data passes, invalid data still raises."""

    def test_valid_integer_passes_under_union_with_maxlength(self) -> None:
        # 42 is a valid integer for "a"; maxLength does not apply to
        # integers. The pre-fix build rejected it with the string branch's
        # validator (``42 is not of type "string"``).
        assert repair_json_loads('{"a": 42, "b": "7"}', schema=_UNION_ROOT) == {
            "a": 42,
            "b": 7,
        }

    def test_the_union_still_coerces_the_string_branch(self) -> None:
        # The union keeps both branches: a one-character string is valid
        # on the string branch (maxLength 1 satisfied), and a longer one
        # is invalid on every branch (the ordinary ValueError).
        schema = {"type": "object", "properties": {"a": _UNION_MAXLEN}}
        assert repair_json_loads('{"a": "x"}', schema=schema) == {"a": "x"}
        with pytest.raises(ValueError):
            repair_json_loads('{"a": "toolong"}', schema=schema)

    def test_each_array_item_validates_with_its_own_branch(self) -> None:
        # The per-value lane: every element of an array re-enters the
        # union's branch loop, so a stale cache (one branch's validator
        # aliased onto the next) rejects the second valid integer.
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"a": _UNION_MAXLEN},
                "required": ["a"],
            },
        }
        assert repair_json_loads('[{"a": 1}, {"a": 2}, {"a": 3}]', schema=schema) == [
            {"a": 1},
            {"a": 2},
            {"a": 3},
        ]
        with pytest.raises(ValueError):
            repair_json_loads('[{"a": 1}, {"a": 2}, {"a": "xy"}]', schema=schema)

    def test_repeated_calls_stay_valid(self) -> None:
        # The validator cache lives per call; two calls with the same
        # schema must agree.
        schema = {"type": "object", "properties": {"a": _UNION_MAXLEN}}
        first = repair_json_loads('{"a": 42}', schema=schema)
        second = repair_json_loads('{"a": 42}', schema=schema)
        assert first == second == {"a": 42}
