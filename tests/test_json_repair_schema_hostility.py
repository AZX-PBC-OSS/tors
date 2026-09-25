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

import json
import time
from typing import Any

import pytest

from loop_harness import assert_bounded
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


class TestSharedReferenceNodeCap:
    """A schema built from SHARED references must not expand exponentially
    in the walk (issue #113): ``py_to_value`` follows every path, so 48
    nested shared lists — depth 48, no leaf anywhere — are ~2^48 container
    visits while sitting far under ``MAX_SCHEMA_WALK_DEPTH``. The pre-cap
    walk hung on it (measured: >30 s, no completion). The walk now threads
    one visit counter (per container visit, shared refs re-entered per
    path) and refuses past ``MAX_SCHEMA_WALK_NODES`` — 2_000_000, the canon
    walk's own ceiling — with the schema-authoring error class
    (``ValueError``), in about a tenth of a second. The count is of
    CONTAINER visits, deliberately: a legitimate FLAT schema of the same
    node count is linear in the caller's own input and must keep working,
    so the pair below (shared-ref refuses fast / flat-of-the-same-order
    converts) is the ceiling's honest statement, both pinned.

    A self-referential cycle (a list containing itself) is bounded by the
    DEPTH cap (the walk recurses until 200 and refuses there); the node
    cap never gets a chance to fire on it — pinned so neither bound can
    regress to a hang or a stack overflow."""

    def test_the_reported_shared_list_repro_refuses_quickly(self) -> None:
        x = None
        for _ in range(48):
            x = [x, x]

        def fire() -> None:
            with pytest.raises(ValueError, match="too many objects") as excinfo:
                repair_json("{}", schema={"enum": x})
            assert isinstance(excinfo.value, Exception)  # the catchable class

        # Load-robust spelling (tests/loop_harness.py): min-of-3
        # pass-on-first-clean, measured ~0.2s, ceiling 5s.
        assert_bounded(fire, 5.0, samples=3, label="the shared-list refusal")

    def test_shared_dict_refs_refuse_the_same_way(self) -> None:
        # The dict branch composes with its own mutation-snapshot fix: the
        # counter rides past the snapshot, per VISIT, so the same 2^n
        # shape over dicts refuses identically.
        d: dict[str, Any] = {"type": "string"}
        for _ in range(40):
            d = {"allOf": [d, d]}

        def fire() -> None:
            with pytest.raises(ValueError, match="too many objects"):
                repair_json("{}", schema=d)

        assert_bounded(fire, 5.0, samples=3, label="the shared-dict refusal")

    def test_a_self_referential_cycle_is_the_depth_refusal(self) -> None:
        x: list[Any] = []
        x.append(x)
        with pytest.raises(ValueError, match="recursion depth"):
            repair_json("{}", schema={"enum": x})

    def test_a_legitimately_large_flat_schema_still_converts(self) -> None:
        # The ceiling's other side, at the same node count the exponential
        # cannot fake: ~1.9M real leaves (plus their containers) under the
        # cap, flat — the walk converts them all (the instance then fails
        # the enum honestly, the repair layer's own answer, not the
        # boundary's). Load-robust spelling (tests/loop_harness.py):
        # min-of-3 pass-on-first-clean.
        members = [f"member{i}" for i in range(1_900_000)]
        schema = {"properties": {"absent": {"type": "string", "enum": members}}}

        def fire() -> None:
            assert repair_json('{"p": "zzz"}', schema=schema) == '{"p": "zzz"}'

        assert_bounded(fire, 30.0, samples=3, label="the 1.9M-leaf flat schema conversion")


class TestWideSchemaPropertyLookups:
    """``repair_json``'s schema-aware repair against a WIDE schema (issue
    F3, the same cost class the #111/#102 fixes bounded): the alignment
    passes consulted the schema's properties map and the document's key
    set by name per key/property, so each pass was O(properties x
    document keys). Measured through the public API, the reported repro —
    a 40k-property schema, a matching 40k-key document — ran:

        n=40_000   6411 ms  (~127x the 2500-prop cell; ~4.5x/doubling)
        n=20_000   1064 ms
        n=10_000    504 ms  (16x the input, ~200x the time)

    and an EMPTY document still paid 6.9 -> 208 ms over 16x (the schema
    preparation floor). The fix gives both sides the repo's wide-object
    index shape (``ObjectBuilder``'s lazy map, same threshold): the
    schema config indexes its properties by exact and folded key (built
    lazily on the first lookup), and the walkers index the document's
    entries once per container (``EntryIndex``) — every pass is now one
    O(1) name lookup per key/property pair.

    Measured, min-of-5 (Linux, CPython 3.12, otherwise-idle box), the
    same repro:

        n=32_000   192 ms  (empty-doc same size: 160 ms)
        n=16_000    73 ms  (empty-doc same size:  67 ms)
        n= 8_000    31 ms  (empty-doc same size:  24 ms)
        n= 4_000    12 ms  (empty-doc same size: 9.5 ms)

    The pre-fix per-key work is isolated by the LOAD-FAIR ratio the
    chunking scaling cell established (each size's matching-document wall
    divided by that same process's empty-document wall — the shared
    schema-preparation cost inflates both sides together, so the ratio
    carries only the per-key work): flat ~1.1-1.3x after (8k vs 32k),
    growing ~2.7x -> ~30x before over the same sizes — ~4.5x per
    doubling, far outside the gates below.

    The residual per-doubling growth that remains (both cells ride
    ~2.1-2.7x/doubling) is the schema-preparation floor itself — the
    jsonschema crate's validator compile over the wide schema plus the
    draft-normalization pass, paid even by the empty document; that
    floor is what the second cell pins, so a dep bump that regresses it
    is a reviewed event too. All cells are ratio-shaped (never absolute
    times) and min-of-N after warmup."""

    @staticmethod
    def _min_wall_ms(fn, /, *args, samples: int = 5, **kwargs) -> float:
        """Min-of-N wall milliseconds after one warmup call (extension
        init, allocator, cache): the fastest of several draws approximates
        the uncontended cost, the same measurement discipline the repo's
        other wall cells apply."""
        fn(*args, **kwargs)
        best = float("inf")
        for _ in range(samples):
            started = time.perf_counter()
            fn(*args, **kwargs)
            best = min(best, time.perf_counter() - started)
        return best * 1000

    @staticmethod
    def _repro(n: int, *, empty: bool) -> None:
        """The reported repro at size n: a matching-keys document under a
        matching-properties schema (or the empty document: the floor)."""
        schema = {
            "type": "object",
            "properties": {f"k{i}": {"type": "string"} for i in range(n)},
        }
        doc = "{}" if empty else json.dumps({f"k{i}": "x" for i in range(n)})
        repair_json(doc, schema=schema)

    @pytest.mark.timing
    def test_matching_document_walls_stay_load_fair_across_sizes(self) -> None:
        """The scaling curve, load-fair: normalize each size's
        matching-document wall by that same process's empty-document wall
        (both pay the identical schema-preparation floor), then require
        the factor to stay small and flat as the input quadruples.
        Post-fix the factor is ~1.1-1.3x at both sizes (flatness growth
        1.0-1.1x across three consecutive cell runs on this box, well
        under the 1.5x gate; the 2.5x absolute gate sits ~2x above the
        band). The pre-fix per-key rescans grew the factor with the input
        (measured ~2.7x at 2.5k -> ~30.7x at 40k, ~4.5x/doubling,
        red-proofed through this API before the fix): any pass that goes
        back to consulting the schema properties (or the document keys)
        by linear scan per key trips this."""
        walls = {
            empty: {
                n: self._min_wall_ms(self._repro, n, empty=empty)
                for n in (8_000, 32_000)
            }
            for empty in (False, True)
        }
        factor_small = walls[False][8_000] / walls[True][8_000]
        factor_big = walls[False][32_000] / walls[True][32_000]
        assert factor_big < 2.5, (
            f"matching-doc factor {factor_big:.2f}x at 32k (band ~1.1-1.3x): "
            "the per-key work grew superlinear in the schema/document size"
        )
        assert factor_big < 1.5 * factor_small, (
            f"factor grew {factor_small:.2f}x -> {factor_big:.2f}x from 8k to 32k "
            "(flat post-fix): a per-key linear scan is back"
        )

    @pytest.mark.timing
    def test_the_empty_document_schema_floor_stays_subquadratic(self) -> None:
        """The floor cell: even the EMPTY document pays the schema
        preparation (the crate's validator compile + the draft
        normalization), mildly superlinear (~2.1-2.7x/doubling measured,
        an O(n log n) shape we do not own). Pinned as a growth gate at
        ~3x/doubling (9x across the 4x cell; measured ~6.6x) so a dep
        bump or a preparation change that relapses it toward quadratic
        (16x) is caught, with the honest caveat in the class docs: this
        is the residual, not the fix."""
        small = self._min_wall_ms(self._repro, 8_000, empty=True)
        big = self._min_wall_ms(self._repro, 32_000, empty=True)
        assert big < 9.0 * small, (
            f"empty-doc floor grew {small:.1f}ms -> {big:.1f}ms over 4x the schema "
            f"(~{(big / small) ** 0.5:.1f}x/doubling; band ~2.1-2.7x): the schema "
            "preparation went quadratic"
        )
