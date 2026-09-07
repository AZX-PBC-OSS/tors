"""Contract gate for ``tors.b64_encode_bytes``: byte-exact parity with
``base64.b64encode(raw).decode("ascii")`` (RFC 4648 standard alphabet, padded),
over arbitrary bytes.

Why this parity is the product: the OCR content-addressing path this replaces holds
the GIL for ~150ms of ``b64encode`` on a 100MB document (the measured motivating
case); the swap only works if the
encoded string is indistinguishable from the stdlib expression's, because it is
stored and compared (content-addressing), not just displayed.

Parity is pinned three ways, complementary by construction: hypothesis over
arbitrary bytes (unknown-shape guard), an exhaustive every-byte-value sweep (the
256 single bytes and all 65,536 two-byte pairs: every 6-bit symbol pair appears
as a data position, and every padding shape appears at lengths 1 and 2), and the
RFC 4648 §10 vectors pinned literally (known-answer proof that cannot pass by
accidental agreement with the local stdlib). Crate-side, the same semantics are
pinned by the RFC vectors in ``src/b64_impl.rs``.
"""

from __future__ import annotations

import base64

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import b64_encode_bytes


def _stdlib_b64(raw: bytes) -> str:
    """The contract expression, spelled out: the exact stdlib call being replaced."""
    return base64.b64encode(raw).decode("ascii")


class TestParityOverArbitraryBytes:
    @given(st.binary(max_size=64))
    @settings(max_examples=500)
    def test_equals_the_stdlib_expression(self, raw: bytes) -> None:
        assert b64_encode_bytes(raw) == _stdlib_b64(raw)

    @given(st.binary(min_size=1, max_size=6))
    @settings(max_examples=300)
    def test_output_is_ascii_only_and_4_ceil_3_len(self, raw: bytes) -> None:
        # Structural consequences of the contract that hold independent of the
        # exact symbols: ASCII-only output, length exactly ceil(n/3)*4.
        encoded = b64_encode_bytes(raw)
        assert encoded.isascii()
        assert len(encoded) == 4 * ((len(raw) + 2) // 3)


class TestExhaustiveSweeps:
    def test_every_single_byte_value(self) -> None:
        for value in range(256):
            raw = bytes([value])
            assert b64_encode_bytes(raw) == _stdlib_b64(raw), f"byte 0x{value:02x}"

    def test_every_two_byte_pair(self) -> None:
        # All 65,536 pairs: every ordered pair of 6-bit symbols lands in a data
        # position, and both one-byte padding shapes (== and =) appear.
        for hi in range(256):
            for lo in range(256):
                raw = bytes([hi, lo])
                assert b64_encode_bytes(raw) == _stdlib_b64(raw), f"pair 0x{hi:02x}0x{lo:02x}"

    @pytest.mark.parametrize("length", range(65))
    def test_every_length_up_to_64(self, length: int) -> None:
        # 0..64 covers every padding shape and several 3-byte group boundaries
        # with data that varies per length (i % 251 breaks alignment repetition).
        raw = bytes((i * 251 + 7) % 256 for i in range(length))
        assert b64_encode_bytes(raw) == _stdlib_b64(raw)

    @pytest.mark.parametrize(
        "length",
        [3000, 3001, 3002, 30000, 30001, 30002],
        ids=["3000", "3001", "3002", "30000", "30001", "30002"],
    )
    def test_each_residue_mod_three_at_realistic_sizes(self, length: int) -> None:
        # The padding shapes at sizes a real payload produces, not just toy
        # lengths: one cell per residue class mod 3 at ~3KB and ~30KB.
        raw = bytes((i * 131 + 11) % 256 for i in range(length))
        assert b64_encode_bytes(raw) == _stdlib_b64(raw)


class TestRfc4648Vectors:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (b"", ""),
            (b"f", "Zg=="),
            (b"fo", "Zm8="),
            (b"foo", "Zm9v"),
            (b"foob", "Zm9vYg=="),
            (b"fooba", "Zm9vYmE="),
            (b"foobar", "Zm9vYmFy"),
        ],
        ids=["empty", "f", "fo", "foo", "foob", "fooba", "foobar"],
    )
    def test_rfc_4648_section_10_vectors_verbatim(self, raw: bytes, expected: str) -> None:
        """The RFC's own test vectors, pinned literally; a wrong alphabet or
        padding wiring cannot hide behind parity with the local stdlib alone."""
        assert b64_encode_bytes(raw) == expected

    def test_a_four_byte_emoji_encodes_to_its_known_encoding(self) -> None:
        assert b64_encode_bytes("\U0001f600".encode("utf-8")) == "8J+YgA=="


class TestBytesOnlyArgumentContract:
    """Exactly ``bytes``, the same immutable zero-copy borrow contract as
    ``decode_utf8`` (see tests/test_decode_utf8.py's module docstring): writable
    or non-bytes inputs would either race a GIL-released read or not be bytes at
    all."""

    @pytest.mark.parametrize(
        "not_bytes",
        [bytearray(b"abc"), memoryview(b"abc"), "abc", 123, None],
        ids=["bytearray", "memoryview", "str", "int", "none"],
    )
    def test_non_bytes_arguments_raise_type_error(self, not_bytes: object) -> None:
        with pytest.raises(TypeError):
            b64_encode_bytes(not_bytes)  # type: ignore[arg-type]
