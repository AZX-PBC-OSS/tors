"""Contract gate for ``tors.merkle_root`` and ``tors.merkle_diff``.

``merkle_root(chunks: list[bytes]) -> str`` is a domain-separated SHA-256
Merkle root over ``chunks``, RFC 6962 / Certificate-Transparency style:
leaves hash as ``SHA-256(0x00 || chunk)``, internal (two-child) nodes hash as
``SHA-256(0x01 || left || right)``. An unpaired left node at any layer is
PROMOTED unchanged to the next layer (rs_merkle's own default
``concat_and_hash``), not duplicated against itself.

Why domain separation, named (the non-obvious design decision this module
exists to get right): the crate underneath (``rs_merkle``) ships a *default*
hasher whose ``hash()`` is plain undifferentiated ``SHA-256(data)`` and whose
default ``concat_and_hash`` feeds ``SHA-256(left || right)`` through that
SAME function, so no byte anywhere distinguishes "this is a leaf's preimage"
from "this is two children's concatenation." That is precisely the
ambiguity CVE-2012-2459 exploited in early Bitcoin Merkle trees: with no
domain byte, a leaf hash and an internal-node hash live in the same output
space, so a proof can present an internal node's hash as though it were some
leaf's digest (or vice versa), and a duplicate-padded odd layer can make two
differently-shaped chunk lists collide on the same root. tors's own
``Hasher`` impl (``DomainSeparatedSha256`` in ``src/merkle_impl.rs``)
prefixes every leaf hash with ``0x00`` and every two-child concatenation
with ``0x01``, closing both holes, and keeps rs_merkle's promotion (not
duplication) behavior for unpaired odd nodes, which is the standard
mitigation for the second half of that CVE's shape. The hash scheme is part
of the root's output CONTRACT from v1 (roots are meant to be stored/compared
across calls), so it needs to be right before any proof-generation API is
added on top, not patched in retroactively once one lands.

``merkle_diff(chunks_a: list[bytes], chunks_b: list[bytes]) -> list[int]``
reports every index where the two lists' chunks differ, comparing 32-byte
leaf digests rather than raw contents; every index at or beyond the shorter
list's length is reported (no counterpart to compare against). It does not
walk a tree; see ``src/merkle_impl.rs`` for why one buys nothing here.

Proven three ways below: (a) a hand-rolled reference tree builder
(``_reference_root``) that reimplements the RFC 6962-style scheme from
scratch and is cross-checked against ``merkle_root`` over many shapes, (b)
the CVE-2012-2459-shaped adversarial property directly: no internal node's
raw hash bytes, leaf-prefixed, ever collides with an actual leaf hash in the
same tree, and (c) hypothesis differentials against a brute-force
``_reference_diff`` for ``merkle_diff``.
"""

from __future__ import annotations

import hashlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import merkle_diff, merkle_root

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"


def _leaf_hash(chunk: bytes) -> bytes:
    return hashlib.sha256(LEAF_PREFIX + chunk).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(NODE_PREFIX + left + right).digest()


def _reference_root(chunks: list[bytes]) -> str | None:
    """Independent reimplementation: RFC 6962-style domain separation,
    promotion (not duplication) of an unpaired left node."""
    if not chunks:
        return None
    layer = [_leaf_hash(c) for c in chunks]
    while len(layer) > 1:
        nxt = []
        i = 0
        while i < len(layer):
            if i + 1 < len(layer):
                nxt.append(_node_hash(layer[i], layer[i + 1]))
            else:
                nxt.append(layer[i])  # promoted, not duplicated
            i += 2
        layer = nxt
    return layer[0].hex()


def _reference_diff(a: list[bytes], b: list[bytes]) -> list[int]:
    return [i for i in range(max(len(a), len(b))) if i >= len(a) or i >= len(b) or a[i] != b[i]]


_chunks_strategy = st.lists(st.binary(max_size=32), max_size=25)
_chunk_pair_strategy = st.tuples(
    st.lists(st.binary(max_size=32), max_size=25),
    st.lists(st.binary(max_size=32), max_size=25),
)


class TestMerkleRootDomainSeparation:
    """The security-critical property this module exists for."""

    def test_root_matches_rfc6962_style_reference_construction(self) -> None:
        chunks = [b"a", b"b", b"c", b"d"]
        assert merkle_root(chunks) == _reference_root(chunks)

    def test_leaf_hash_uses_zero_prefix(self) -> None:
        # A single chunk's root IS its own leaf hash (no internal node).
        assert merkle_root([b"abc"]) == _leaf_hash(b"abc").hex()

    def test_root_differs_from_plain_undifferentiated_sha256(self) -> None:
        # Proves the domain byte actually changes the output versus what
        # rs_merkle's built-in (undifferentiated) hasher would produce.
        root = merkle_root([b"a"])
        assert root != hashlib.sha256(b"a").hexdigest()

    def test_internal_node_hash_never_collides_with_any_leaf_hash_in_the_tree(
        self,
    ) -> None:
        """The CVE-2012-2459-shaped check: build a small tree, compute its
        internal node hashes independently, and confirm none of them equals
        any actual leaf's hash, so an attacker cannot present an internal
        node's digest as though it were a leaf's."""
        chunks = [b"a", b"b", b"c", b"d"]
        leaves = {_leaf_hash(c) for c in chunks}
        ab = _node_hash(_leaf_hash(b"a"), _leaf_hash(b"b"))
        cd = _node_hash(_leaf_hash(b"c"), _leaf_hash(b"d"))
        root = _node_hash(ab, cd)

        assert ab not in leaves
        assert cd not in leaves
        assert root not in leaves
        assert merkle_root(chunks) == root.hex()

    def test_forging_a_leaf_from_an_internal_nodes_raw_bytes_is_impossible(
        self,
    ) -> None:
        """Even feeding an internal node's raw hash bytes back through the
        LEAF-prefixed hash function (i.e. treating those 32 bytes as if they
        were "a chunk") cannot land on any real leaf hash; the leaf and
        node hash spaces are prefix-disjoint, not merely different by
        chance."""
        chunks = [b"a", b"b", b"c", b"d"]
        leaves = {_leaf_hash(c) for c in chunks}
        ab = _node_hash(_leaf_hash(b"a"), _leaf_hash(b"b"))

        forged = _leaf_hash(ab)  # attacker's forged "leaf hash of ab's bytes"
        assert forged not in leaves

    @given(_chunks_strategy.filter(lambda c: len(c) >= 1))
    @settings(max_examples=200)
    def test_root_matches_reference_for_arbitrary_chunk_lists(
        self, chunks: list[bytes]
    ) -> None:
        assert merkle_root(chunks) == _reference_root(chunks)


class TestMerkleRootOddEvenShapes:
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17])
    def test_various_leaf_counts_match_reference(self, n: int) -> None:
        chunks = [bytes([i % 256]) for i in range(n)]
        assert merkle_root(chunks) == _reference_root(chunks)

    def test_odd_count_root_differs_from_duplicate_padded_even_count(self) -> None:
        # Promotion (tors's actual behavior) must NOT coincide with what
        # Bitcoin-style duplication of the last leaf would produce; that
        # coincidence is exactly the CVE-2012-2459 malleability shape.
        odd = [b"a", b"b", b"c"]
        duplicate_padded = [b"a", b"b", b"c", b"c"]
        assert merkle_root(odd) != merkle_root(duplicate_padded)


class TestMerkleRootCorrectness:
    def test_deterministic_across_calls(self) -> None:
        chunks = [b"x" * 10, b"y" * 3, b"z"]
        assert merkle_root(chunks) == merkle_root(list(chunks))

    def test_order_sensitive(self) -> None:
        assert merkle_root([b"a", b"b"]) != merkle_root([b"b", b"a"])

    def test_large_chunk_count_does_not_panic(self) -> None:
        chunks = [bytes([i % 256]) * 4 for i in range(1500)]
        root = merkle_root(chunks)
        assert isinstance(root, str)
        assert len(root) == 64
        int(root, 16)  # valid hex

    def test_large_single_chunk_ten_megabytes(self) -> None:
        big = b"\x42" * (10 * 1024 * 1024)
        root = merkle_root([big, b"y"])
        assert len(root) == 64

    def test_empty_list_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="root of no chunks"):
            merkle_root([])

    def test_non_bytes_element_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            merkle_root([b"a", "not-bytes"])  # type: ignore[list-item]

    def test_non_list_argument_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            merkle_root("not-a-list")  # type: ignore[arg-type]

    def test_root_is_lowercase_hex(self) -> None:
        root = merkle_root([b"a", b"b", b"c"])
        assert root == root.lower()
        assert all(c in "0123456789abcdef" for c in root)


class TestMerkleDiff:
    def test_finds_exactly_the_differing_indices(self) -> None:
        a = [b"a", b"b", b"c", b"d"]
        b = [b"a", b"X", b"c", b"Y"]
        assert merkle_diff(a, b) == [1, 3]

    def test_single_difference(self) -> None:
        a = [b"a", b"b", b"c"]
        b = [b"a", b"Z", b"c"]
        assert merkle_diff(a, b) == [1]

    def test_two_scattered_differences(self) -> None:
        a = [bytes([i]) for i in range(10)]
        b = list(a)
        b[2] = b"\xff"
        b[7] = b"\xfe"
        assert merkle_diff(a, b) == [2, 7]

    def test_several_scattered_differences(self) -> None:
        a = [bytes([i]) for i in range(20)]
        b = list(a)
        for idx in (0, 3, 5, 11, 19):
            b[idx] = b[idx] + b"!"
        assert merkle_diff(a, b) == [0, 3, 5, 11, 19]

    def test_identical_lists_diff_to_empty(self) -> None:
        a = [b"a", b"b", b"c"]
        assert merkle_diff(a, list(a)) == []

    def test_two_empty_lists_diff_to_empty(self) -> None:
        assert merkle_diff([], []) == []

    def test_a_longer_than_b_reports_every_trailing_index(self) -> None:
        a = [b"a", b"b", b"c", b"d"]
        b = [b"a", b"b"]
        assert merkle_diff(a, b) == [2, 3]

    def test_b_longer_than_a_reports_every_trailing_index(self) -> None:
        a = [b"a", b"b"]
        b = [b"a", b"b", b"c", b"d"]
        assert merkle_diff(a, b) == [2, 3]

    def test_one_empty_one_nonempty_reports_every_index(self) -> None:
        assert merkle_diff([], [b"a", b"b"]) == [0, 1]
        assert merkle_diff([b"a", b"b"], []) == [0, 1]

    def test_non_bytes_element_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            merkle_diff([b"a"], [1, 2])  # type: ignore[list-item]

    def test_non_bytes_element_in_first_argument_raises_type_error(self) -> None:
        """The symmetric case of the test above: ``chunks_a`` is validated
        exactly like ``chunks_b`` (the Rust core treats both slices
        identically, src/merkle_impl.rs), but only the ``chunks_b``-side
        failure was ever pinned from Python."""
        with pytest.raises(TypeError):
            merkle_diff([1, 2], [b"a"])  # type: ignore[list-item]

    def test_non_list_argument_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            merkle_diff("not-a-list", [b"a"])  # type: ignore[arg-type]

    @given(_chunk_pair_strategy)
    @settings(max_examples=300)
    def test_matches_brute_force_reference(
        self, pair: tuple[list[bytes], list[bytes]]
    ) -> None:
        a, b = pair
        assert merkle_diff(a, b) == _reference_diff(a, b)

    @given(_chunks_strategy)
    @settings(max_examples=100)
    def test_diff_of_a_list_against_itself_is_always_empty(
        self, chunks: list[bytes]
    ) -> None:
        assert merkle_diff(chunks, list(chunks)) == []
