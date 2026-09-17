"""Contract gate for ``tors.content_hash`` over hostile dict keys: a key
whose ``__hash__`` misbehaves must never take the interpreter down.

A dict key whose ``__hash__`` returns different values across calls can put
TWO entries into one dict that share the same key object AND the same value
object (the reinsertion lands in a different slot because the hash moved):

    class S(str):
        h = 0
        def __hash__(self): return S.h
    s = S("a"); d = {}
    d[s] = None
    S.h = 1
    d[s] = None          # same object, now a different slot: len(d) == 2

``json.dumps(d, sort_keys=True)`` SUCCEEDS on that dict: it sorts the two
identical ``(key, value)`` pairs and emits ``{"a": null, "a": null}``, so
the documented contract (docs/api.md: byte-identical with the stdlib
expression wherever json.dumps succeeds) says ``content_hash`` returns the
oracle hash of the doubled pair, not an error and never a crash. The walk's
delegated sort reads its permutation back by ``(key ptr, value ptr)``
identity, which folds the two entries into one slot; the pre-fix build
panicked there (``pyo3_runtime.PanicException: the sort returned a pair we
did not build``), a Rust panic wrapped as a ``BaseException`` subclass no
``except Exception`` can catch. The pins assert the correct behavior: the
call completes, and the hash is the oracle's, doubled pair included.

Hostile keys here are pure Python: cheap, bounded, in-process.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from tors import content_hash


def _canonical(obj: Any) -> str:
    """The canonical form, spelled exactly as the contract defines it."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _oracle(obj: Any) -> str:
    """The full stdlib expression ``tors.content_hash`` replaces."""
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


class TestHashShiftingKeys:
    """A key whose ``__hash__`` changes between calls: json parity, never
    a panic."""

    def test_hash_shifting_key_hashes_as_the_oracle(self) -> None:
        # The two folded entries are content-identical (same key object,
        # same value object), so the delegated sort's order between them is
        # unobservable and the output must be the oracle's: the doubled
        # pair, exactly as json.dumps emits it.
        class S(str):
            h = 0

            def __hash__(self) -> int:
                return S.h

        s = S("a")
        d: dict[Any, None] = {}
        d[s] = None
        S.h = 1
        d[s] = None
        assert len(d) == 2  # the fold precondition: two entries, one object
        assert content_hash(d) == _oracle(d)

    def test_the_folded_output_is_the_doubled_pair(self) -> None:
        # The literal pin: the canonical bytes carry the pair twice, the
        # stdlib spelling's own output for this dict.
        class S(str):
            h = 0

            def __hash__(self) -> int:
                return S.h

        s = S("a")
        d: dict[Any, None] = {}
        d[s] = None
        S.h = 1
        d[s] = None
        assert content_hash(d) == hashlib.sha256(b'{"a":null,"a":null}').hexdigest()
        assert _canonical(d) == '{"a":null,"a":null}'

    def test_hash_shifting_key_folds_three_entries(self) -> None:
        # Two shifts, three identical entries: every one survives the
        # identity read-back (the pre-fix build dropped all but one and
        # panicked on the second remove).
        class S(str):
            h = 0

            def __hash__(self) -> int:
                return S.h

        s = S("a")
        d: dict[Any, None] = {}
        d[s] = None
        S.h = 1
        d[s] = None
        S.h = 2
        d[s] = None
        assert len(d) == 3
        assert content_hash(d) == _oracle(d)
        assert _canonical(d) == '{"a":null,"a":null,"a":null}'

    def test_hash_shifting_key_with_distinct_values(self) -> None:
        # Same key object, distinct value objects: the identity pairs
        # differ (no fold), the dict legitimately holds two entries, and
        # the output is the oracle's: both entries, in sorted order.
        class S(str):
            h = 0

            def __hash__(self) -> int:
                return S.h

        s = S("a")
        d: dict[Any, int] = {}
        d[s] = 1
        S.h = 1
        d[s] = 2
        assert len(d) == 2
        assert content_hash(d) == _oracle(d)
        assert _canonical(d) == '{"a":1,"a":2}'

    def test_repeated_calls_stay_deterministic(self) -> None:
        # A shifted-hash dict hashes the same on every call (and the fold
        # handling does not poison later delegated sorts in the process).
        class S(str):
            h = 0

            def __hash__(self) -> int:
                return S.h

        s = S("a")
        d: dict[Any, None] = {}
        d[s] = None
        S.h = 1
        d[s] = None
        first = content_hash(d)
        assert content_hash(d) == first
        assert first == _oracle(d)
        # An ordinary exotic-key dict still hashes through the same lane.
        assert content_hash({"b": 1, "a": 2}) == _oracle({"b": 1, "a": 2})
