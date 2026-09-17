"""The ``content_hash`` NaN-key doc carve-out, as an executable contract.

``docs/api.md``'s ``content_hash`` section promised "any dict key order
yields the same hash", which is false for dicts with DISTINCT NaN keys:
NaN never compares equal to NaN (not even to itself), so the delegated
``list.sort`` output order depends on the caller's insertion order, and
the serialized (hashed) key order follows it. ``json.dumps`` with
``sort_keys=True`` behaves the same way, and ``content_hash`` is
byte-identical to that oracle BY DESIGN, so the code is correct and the
doc sentence overclaimed. The fix is the carve-out in the wording, and
this file runs it (docs-contract testing: a promise that fails is a docs
lie or a code divergence, either way a finding):

- the docs sentence carries the carve-out (static half, the stale-phrase
  sweep style: the unqualified "any dict key order" claim is gone, the
  distinct-NaN-key exception is present and points at the delegated-sort
  paragraph that explains it);
- the carve-out's truth (dynamic half): each ordering of a
  distinct-NaN-key dict hashes exactly as the ``json.dumps`` oracle
  spells it: the parity that matters, pinned on the canonical two-key
  repro in both insertion orders;
- the carve-out's scope: non-NaN keys keep the any-order same-hash
  promise unchanged (the carve-out must not swallow the contract).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import tors

REPO = Path(__file__).resolve().parents[1]
API_MD = REPO / "docs" / "api.md"


def _oracle(obj: object) -> str:
    """The documented oracle, verbatim from docs/api.md: sha256 over
    json.dumps(sort_keys=True, separators=(",", ":")) with the stdlib
    defaults ensure_ascii=True and allow_nan."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _api_text() -> str:
    return " ".join(API_MD.read_text(encoding="utf-8").split())


# --- Static: the sentence carries the carve-out ------------------------------


def test_the_unqualified_any_key_order_claim_is_gone() -> None:
    """The stale phrase ("any dict key order yields the same hash" with
    no qualifier) must not survive in the docs: it is the overclaim the
    distinct-NaN-key repro falsifies."""
    text = _api_text()
    assert "any dict key order yields the same hash, and" not in text, (
        "docs/api.md's content_hash section still claims any dict key "
        "order yields the same hash, unqualified: false for dicts with "
        "distinct NaN keys (NaN never compares equal, so the delegated "
        "timsort's output order is insertion-order-dependent)"
    )


def test_the_nan_carve_out_is_present_and_points_at_the_delegated_sort() -> None:
    """The carve-out is IN the determinism sentence (not a distant
    footnote): the determinism promise and its distinct-NaN-key exception
    in one breath, with the delegated-sort paragraph named as the
    mechanism."""
    text = _api_text()
    marker = "Deterministic by construction: any dict key order yields the same hash"
    assert marker in text, (
        "docs/api.md's content_hash determinism sentence vanished or was "
        "reworded past recognition; the carve-out pin needs it as the anchor"
    )
    start = text.index(marker)
    tail = text[start : text.index("hash identically", start)]
    assert "NaN" in tail and "timsort" in tail, (
        f"the determinism sentence must carve out the distinct-NaN-key "
        f"dicts and name timsort's order as the mechanism:\n{tail}"
    )


# --- Dynamic: the carve-out is true ------------------------------------------


def test_each_nan_key_ordering_hashes_exactly_as_the_json_oracle_spells_it() -> None:
    """The canonical repro, both insertion orders, each against the oracle:
    content_hash({n: 0, 1: 1}) is sha256 over json.dumps's OWN spelling
    of that dict, and likewise for the reversed insertion order;
    whatever order json.dumps's timsort emits, tors matches it
    byte-for-byte. (The two orderings may or may not hash differently:
    that is timsort's behavior, not a mathematical property, and the
    carve-out deliberately promises only oracle parity.)"""
    n = float("nan")
    # Homogeneous key sorts only: a NaN key mixed with str keys is the
    # doubly-documented unsortable case (both sides raise the sort's own
    # TypeError), pinned separately below.
    for obj in (
        {n: 0, 1: 1},
        {1: 1, n: 0},
        {n: 0, 1: 1, 2: 3},
        {2: 3, 1: 1, n: 0},
        {n: 0, 2.5: "x"},
        {2.5: "x", n: 0},
    ):
        assert tors.content_hash(obj) == _oracle(obj), obj


def test_a_nan_key_mixed_with_str_keys_raises_the_sort_error_on_both_sides() -> None:
    """The carve-out never softens the sort's own refusal: a NaN key
    mixed with str keys is unsortable, and the sort's TypeError is
    byte-identical with json.dumps's on both insertion orders."""
    n = float("nan")
    for obj in ({n: 0, "a": 1}, {"a": 1, n: 0}):
        with pytest.raises(TypeError):
            tors.content_hash(obj)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            _oracle(obj)


def test_distinct_nan_keys_can_coexist_and_hash_like_the_oracle() -> None:
    """Two distinct NaN objects legally coexist as dict keys (NaN != NaN),
    the delegated-sort paragraph's own corner, and the hash is the
    oracle's over both."""
    nan_a = float("nan")
    nan_b = float("nan")
    assert nan_a != nan_b
    obj = {nan_a: "first", nan_b: "second"}
    assert len(obj) == 2
    assert tors.content_hash(obj) == _oracle(obj)


# --- Dynamic: the carve-out's scope -------------------------------------------


def test_non_nan_keys_keep_the_any_order_same_hash_promise() -> None:
    """The carve-out is scoped to distinct NaN keys: NaN-free dicts still
    hash the same under any key insertion order (the promise the sentence
    exists for). Key sorts are homogeneous per dict (a mixed str/int
    key set is the documented unsortable refusal, not an order question),
    and each pair covers a different sort class: ints (numeric order),
    strs (lexicographic), floats (numeric with the delegated sort)."""
    for a, b in (
        ({1: "x", 2: "y", 10: "z"}, {10: "z", 2: "y", 1: "x"}),
        ({"a": "x", "b": "y", "c": "z"}, {"c": "z", "b": "y", "a": "x"}),
        ({1.5: "x", 0.5: "y"}, {0.5: "y", 1.5: "x"}),
    ):
        assert tors.content_hash(a) == tors.content_hash(b) == _oracle(a), (a, b)


def test_equal_nan_values_hash_like_the_oracle_too() -> None:
    """NaN as a VALUE was never in question (values serialize as json's
    allow_nan literals, order-free); pinned next to the key carve-out so
    the two NaN roles cannot be confused."""
    a = {"k": float("nan")}
    b = {"k": float("nan")}
    assert tors.content_hash(a) == tors.content_hash(b) == _oracle(a)
