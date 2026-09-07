"""Contract gate for the UTF-16 bytes-in surface: ``tors.decode_utf16`` and
``tors.utf16_is_valid`` must be indistinguishable from the stdlib expressions
they replace: ``raw.decode("utf-16", errors=...)`` / ``"utf-16-le"`` /
``"utf-16-be"``, over ARBITRARY bytes, the same byte-exact bar
``decode_utf8`` already holds.

Unlike UTF-8, UTF-16 has FOUR distinct ``UnicodeDecodeError`` reason strings
CPython produces depending on exactly where and how a code-unit sequence goes
wrong (see ``src/utf16_impl.rs``'s module docs for the full derivation,
verified against a running interpreter, not assumed from the codec's
documentation):

- ``"truncated data"``: a lone trailing byte, one short of a full code unit,
  with no pending high surrogate. Span: that one byte.
- ``"unexpected end of data"``: a high surrogate with fewer than two bytes
  following it. Span: the surrogate through the end of the input.
- ``"illegal UTF-16 surrogate"``: a high surrogate followed by a full code
  unit that is not a valid low surrogate. Span: the high surrogate's two
  bytes alone.
- ``"illegal encoding"``: a low surrogate reached other than as the second
  half of a valid pair. Span: its own two bytes alone.

``byteorder="native"`` (the default, matching the plain ``"utf-16"`` codec
name) sniffs a leading BOM and strips it from the output, falling back to
the host's own endianness when none is present; ``"little"``/``"big"``
(matching ``"utf-16-le"``/``"utf-16-be"``) never sniff or strip a BOM; a
leading BOM-like byte pair decodes as the literal U+FEFF character instead.
``.encoding`` on a raised error is always the RESOLVED label
(``"utf-16-le"``/``"utf-16-be"``), never the bare ``"utf-16"`` name, matching
CPython exactly.

No ``encode_utf16``/``finalize_utf16``: see ``src/utf16_impl.rs``'s module
docs for why.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import decode_utf16, utf16_is_valid

_UNPAIRED_HIGH_ALONE = b"\x00\xd8"
_UNPAIRED_HIGH_THEN_CHAR = b"\x00\xd8A\x00"
_UNPAIRED_HIGH_THEN_HIGH = b"\x00\xd8\x00\xd8"
_UNPAIRED_LOW_ALONE = b"\x00\xdc"
_UNPAIRED_LOW_THEN_CHAR = b"\x00\xdcA\x00"
_LONE_TRAILING_BYTE = b"h\x00i"
_VALID_PAIR_LE = "\U0001f600".encode("utf-16-le")

_MALFORMED_BATTERY: list[tuple[bytes, str]] = [
    (_LONE_TRAILING_BYTE, "lone trailing byte"),
    (_UNPAIRED_HIGH_ALONE, "high surrogate with nothing following"),
    (_UNPAIRED_HIGH_ALONE + b"Z", "high surrogate with one dangling byte"),
    (_UNPAIRED_HIGH_THEN_CHAR, "high surrogate then a non-surrogate unit"),
    (_UNPAIRED_HIGH_THEN_HIGH, "high surrogate then another high surrogate"),
    (_UNPAIRED_LOW_ALONE, "lone low surrogate"),
    (_UNPAIRED_LOW_THEN_CHAR, "low surrogate then a non-surrogate unit"),
    (_UNPAIRED_LOW_ALONE + b"Z", "low surrogate with one dangling byte"),
]


def _codec_name(byteorder: str) -> str:
    return {"native": "utf-16", "little": "utf-16-le", "big": "utf-16-be"}[byteorder]


class TestStrictParityOverArbitraryBytes:
    @given(st.binary(max_size=64), st.sampled_from(["native", "little", "big"]))
    @settings(max_examples=1500)
    def test_strict_decode_matches_stdlib_or_raises_the_same_exception(
        self, raw: bytes, byteorder: str
    ) -> None:
        codec = _codec_name(byteorder)
        try:
            expected = raw.decode(codec)
        except UnicodeDecodeError as exc:
            with pytest.raises(UnicodeDecodeError) as info:
                decode_utf16(raw, byteorder=byteorder)
            got = info.value
            assert got.encoding == exc.encoding
            assert got.start == exc.start
            assert got.end == exc.end
            assert got.reason == exc.reason
            assert got.object == exc.object
            return
        assert decode_utf16(raw, byteorder=byteorder) == expected

    @pytest.mark.parametrize(
        ("raw", "label"), _MALFORMED_BATTERY, ids=[b for _, b in _MALFORMED_BATTERY]
    )
    def test_malformed_battery_matches_stdlib_exactly(self, raw: bytes, label: str) -> None:
        del label
        for byteorder, codec in (
            ("native", "utf-16"),
            ("little", "utf-16-le"),
            ("big", "utf-16-be"),
        ):
            try:
                expected = raw.decode(codec)
            except UnicodeDecodeError as exc:
                with pytest.raises(UnicodeDecodeError) as info:
                    decode_utf16(raw, byteorder=byteorder)
                got = info.value
                assert (got.encoding, got.start, got.end, got.reason) == (
                    exc.encoding,
                    exc.start,
                    exc.end,
                    exc.reason,
                )
                continue
            assert decode_utf16(raw, byteorder=byteorder) == expected


class TestReplaceParityOverArbitraryBytes:
    @given(st.binary(max_size=64), st.sampled_from(["native", "little", "big"]))
    @settings(max_examples=1500)
    def test_replace_matches_stdlib_exactly(self, raw: bytes, byteorder: str) -> None:
        codec = _codec_name(byteorder)
        expected = raw.decode(codec, errors="replace")
        assert decode_utf16(raw, errors="replace", byteorder=byteorder) == expected

    @pytest.mark.parametrize(
        ("raw", "label"), _MALFORMED_BATTERY, ids=[b for _, b in _MALFORMED_BATTERY]
    )
    def test_malformed_battery_replace_matches_stdlib(self, raw: bytes, label: str) -> None:
        del label
        for byteorder, codec in (
            ("native", "utf-16"),
            ("little", "utf-16-le"),
            ("big", "utf-16-be"),
        ):
            expected = raw.decode(codec, errors="replace")
            assert decode_utf16(raw, errors="replace", byteorder=byteorder) == expected


class TestBomHandling:
    def test_native_sniffs_and_strips_a_leading_bom(self) -> None:
        assert decode_utf16(b"\xff\xfeh\x00i\x00") == "hi"
        assert decode_utf16(b"\xfe\xff\x00h\x00i") == "hi"

    def test_native_with_no_bom_falls_back_to_host_endianness(self) -> None:
        raw = b"h\x00i\x00"
        assert decode_utf16(raw) == raw.decode("utf-16")

    def test_explicit_byteorder_never_strips_a_bom(self) -> None:
        assert decode_utf16(b"\xff\xfeh\x00i\x00", byteorder="little") == "﻿hi"
        assert decode_utf16(b"\xfe\xff\x00h\x00i", byteorder="big") == "﻿hi"

    def test_error_encoding_field_is_always_the_resolved_label(self) -> None:
        with pytest.raises(UnicodeDecodeError) as info:
            decode_utf16(b"h\x00i")
        assert info.value.encoding in ("utf-16-le", "utf-16-be")
        assert info.value.encoding != "utf-16"


class TestValidity:
    @given(st.binary(max_size=48), st.sampled_from(["native", "little", "big"]))
    @settings(max_examples=800)
    def test_utf16_is_valid_agrees_with_decode_success(self, raw: bytes, byteorder: str) -> None:
        try:
            decode_utf16(raw, byteorder=byteorder)
            expect_valid = True
        except UnicodeDecodeError:
            expect_valid = False
        assert utf16_is_valid(raw, byteorder=byteorder) == expect_valid

    def test_empty_bytes_are_valid(self) -> None:
        assert utf16_is_valid(b"") is True
        assert decode_utf16(b"") == ""

    def test_valid_surrogate_pair_round_trips(self) -> None:
        assert decode_utf16(_VALID_PAIR_LE, byteorder="little") == "\U0001f600"
        assert utf16_is_valid(_VALID_PAIR_LE, byteorder="little") is True


class TestArgumentContract:
    def test_errors_outside_the_closed_set_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="errors must be one of"):
            decode_utf16(b"", errors="ignore")

    def test_byteorder_outside_the_closed_set_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="byteorder must be one of"):
            decode_utf16(b"", byteorder="middle")
        with pytest.raises(ValueError, match="byteorder must be one of"):
            utf16_is_valid(b"", byteorder="middle")

    @pytest.mark.parametrize("bad", [bytearray(b"hi"), memoryview(b"hi"), "hi", 1, None])
    def test_non_bytes_input_raises_type_error(self, bad: object) -> None:
        with pytest.raises(TypeError):
            decode_utf16(bad)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            utf16_is_valid(bad)  # type: ignore[arg-type]
