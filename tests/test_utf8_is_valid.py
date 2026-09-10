"""Contract gate for ``tors.utf8_is_valid``: the boolean UTF-8 validity check
CPython's stdlib has no primitive for. The only stdlib way to ask "will decoding
succeed?" is decode-and-catch, which materializes the whole ``str`` on the
success path and pays exception construction and flow on the failure path;
``utf8_is_valid`` answers the question directly: SIMD validation (the
``simdutf8`` crate), GIL-released, ``bool`` out, no exception flow on either
path.

The contract, as two oracle-equal properties over arbitrary bytes:

- ``utf8_is_valid(raw)`` is ``True`` exactly when ``raw.decode("utf-8")``
  succeeds, the stdlib's own notion of well-formedness, via decode-and-catch;
- ``utf8_is_valid(raw)`` is ``True`` exactly when ``tors.decode_utf8(raw)``
  does not raise, cross-surface consistency with tors's own decoder: the two
  must never disagree about what is valid.

The deterministic battery pins every ill-formed class from
``tests/test_decode_utf8.py`` (truncated multi-byte tails, overlong encodings,
lone continuation bytes, surrogate/CESU-8 encodings, the invalid lead bytes
FC/FD/FF, out-of-range and maximal-subpart shapes) as ``False``, and the
boundary codepoints around each exclusion zone (U+D7FF/U+E000 around the
surrogate block, U+10FFFF at the ceiling, plus U+FFFF, a noncharacter but a
valid encoding) as ``True``.

The argument contract is the bytes-in surface's: exactly ``bytes``
(``bytearray`` / ``memoryview`` / ``str`` -> ``TypeError``), the same
zero-copy immutable-borrow-under-detach rationale as ``decode_utf8`` (a
writable buffer mutated by another thread mid-read is a data race, not a
semantic difference).

Measured wall (the dev box this suite runs on, WSL2, 28 logical cores,
ambient load 1.7; min-of-5 after one warm-up, ``perf_counter`` around the
inline call; the stdlib has no boolean validity oracle to race, so the
record is absolute throughput; corpora are
``reference.corpus_utf8("prose", ...)`` and its invalid twin):

    corpus   size     utf8_is_valid   throughput
    valid    12 MiB   0.123 ms        95.3 GiB/s
    valid    32 MiB   0.553 ms        56.5 GiB/s
    valid    100 MiB  2.556 ms        38.2 GiB/s
    invalid  12 MiB   0.088 ms        133.5 GiB/s
    invalid  32 MiB   0.578 ms        54.1 GiB/s

Throughput falls as the corpus outgrows this box's L3 (95 -> 56 -> 38
GiB/s): past 32 MiB the scan is memory-bound, the shape of a SIMD
scan, not a flat GiB/s claim. The invalid 12 MiB cell measures faster than
its valid twin (the trailing 0xFF lets the validator return before the
final block's fixup work), recorded, not thresholded. The criterion ladder
(benches/utf8.rs) holds the same story on the Rust core alone, including
the ~3x ASCII-vs-multibyte lane difference (prose 101 GiB/s vs decomposed
33.9 GiB/s at 12 MiB, reduced sampling, ambient load 3-5).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import decode_utf8, utf8_is_valid

# The ill-formed battery: every class tests/test_decode_utf8.py pins for the
# raising decoder, each named for the shape it pins here. Values built from
# ints, pure ASCII source.
_ILL_FORMED: list[tuple[bytes, str]] = [
    (b"a\xc3", "truncated 2-byte tail"),
    (b"\xf0\x9f", "truncated 4-byte tail"),
    (b"\xf0\x9f\x98\x80\xf0\x9f", "truncated tail after a valid emoji"),
    (b"\xe0\xa0", "valid-prefix pair truncated at end"),
    (b"\xc0\x80", "overlong 2-byte encoding of NUL"),
    (b"\xe0\x80\x80", "overlong 3-byte encoding"),
    (b"\xf0\x80\x80\x80", "overlong 4-byte encoding"),
    (b"\xed\xa0\x80", "surrogate-encoded D800 (CESU-8 high)"),
    (b"\xed\xb0\x80", "surrogate-encoded DC00 (CESU-8 low)"),
    (b"\xed\xa0\x80\xed\xb0\x80", "CESU-8 surrogate pair"),
    (b"\xed\xbf\xbf", "surrogate-encoded DFFF"),
    (b"\x80", "lone continuation byte"),
    (b"a\x80\x80b", "two lone continuations mid-text"),
    (b"\xfc\x84\x80\x80\x80\x80", "legacy 5-byte lead FC"),
    (b"\xfd\x84\x80\x80\x80\x80", "legacy 6-byte lead FD"),
    (b"\xff", "invalid lead byte FF"),
    (b"ok\xffok", "invalid lead mid-ASCII"),
    (b"\xf4\x90", "out-of-range continuation after F4"),
    (b"\xf0\x9f\x41", "maximal subpart before a non-continuation"),
    (b"fo\xd8o", "2-byte lead then non-continuation"),
]

# The valid battery: the everyday shapes plus one anchor on each side of every
# exclusion zone (2/3/4-byte range boundaries, the surrogate block's edges,
# the U+10FFFF ceiling), the cases a validator most easily gets wrong by
# rejecting too much.
_VALID: list[tuple[bytes, str]] = [
    (b"hello world, plain ASCII text.", "plain ASCII"),
    ("caf\u00e9 na\u00efve \U0001f600 text".encode("utf-8"), "valid multibyte mix"),
    (b"\x7f", "U+007F, the last 1-byte codepoint"),
    (b"\xc2\x80", "U+0080, the first 2-byte codepoint"),
    (b"\xdf\xbf", "U+07FF, the last 2-byte codepoint"),
    (b"\xe0\xa0\x80", "U+0800, the first 3-byte codepoint"),
    (b"\xed\x9f\xbf", "U+D7FF, the last codepoint before the surrogate block"),
    (b"\xee\x80\x80", "U+E000, the first codepoint after the surrogate block"),
    (b"\xef\xbf\xbf", "U+FFFF, a noncharacter but a VALID encoding"),
    (b"\xf0\x90\x80\x80", "U+10000, the first 4-byte codepoint"),
    (b"\xf4\x8f\xbf\xbf", "U+10FFFF, the maximum codepoint"),
]


def _decodes_cleanly(raw: bytes) -> bool:
    """The stdlib's only boolean answer to "is this valid UTF-8": decode and
    catch (the expression ``utf8_is_valid`` exists so callers never pay it)."""
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


class TestValidityContract:
    @given(st.binary(max_size=64))
    @settings(max_examples=500)
    def test_matches_the_stdlib_decode_and_catch_oracle(self, raw: bytes) -> None:
        """The whole contract in one property: for any bytes, ``utf8_is_valid``
        says True exactly when the running interpreter's own decoder accepts
        them, with no false negatives (rejecting decodable bytes), no false
        positives (accepting bytes that raise)."""
        assert utf8_is_valid(raw) is _decodes_cleanly(raw)

    @given(st.binary(max_size=64))
    @settings(max_examples=300)
    def test_agrees_with_tors_decode_utf8s_raise_behavior(self, raw: bytes) -> None:
        """Cross-surface consistency: the validator and the decoder share one
        notion of well-formedness: ``utf8_is_valid(raw)`` is True exactly when
        ``tors.decode_utf8(raw)`` returns rather than raising, so a caller that
        gates on the boolean can always decode afterwards."""
        try:
            decode_utf8(raw)
        except UnicodeDecodeError:
            decodes = False
        else:
            decodes = True
        assert utf8_is_valid(raw) is decodes


class TestIllFormedBattery:
    @pytest.mark.parametrize(
        ("raw", "because"),
        [(raw, because) for raw, because in _ILL_FORMED],
        ids=[because for _, because in _ILL_FORMED],
    )
    def test_every_ill_formed_class_is_invalid(self, raw: bytes, because: str) -> None:
        """One pin per ill-formed shape: the validator answers False, and the
        running interpreter's decoder agrees the bytes do not decode (so the
        battery cannot silently drift away from what CPython calls invalid)."""
        assert utf8_is_valid(raw) is False
        assert _decodes_cleanly(raw) is False


class TestValidBattery:
    @pytest.mark.parametrize(
        ("raw", "because"),
        [(raw, because) for raw, because in _VALID],
        ids=[because for _, because in _VALID],
    )
    def test_boundary_and_everyday_valid_inputs_are_valid(self, raw: bytes, because: str) -> None:
        """The valid twin of the ill-formed battery: the everyday shapes plus
        the boundary anchors around each exclusion zone validate True; a
        validator that rejects too much (the classic overlong/surrogate
        overreach) fails here, not just on the invalid side."""
        assert utf8_is_valid(raw) is True
        assert _decodes_cleanly(raw) is True


class TestBytesOnlyArgumentContract:
    """tors takes exactly ``bytes``, the same zero-copy immutable-borrow
    rationale as ``decode_utf8`` (see that test module's docstring). The pin:
    anything else is a ``TypeError`` naming the argument."""

    @pytest.mark.parametrize(
        "not_bytes",
        [
            bytearray(b"abc"),
            memoryview(b"abc"),
            "abc",
            123,
            None,
            ["a", "b"],
        ],
        ids=["bytearray", "memoryview", "str", "int", "none", "list"],
    )
    def test_non_bytes_arguments_raise_type_error(self, not_bytes: object) -> None:
        with pytest.raises(TypeError):
            utf8_is_valid(not_bytes)  # type: ignore[arg-type]


class TestBehavioralCases:
    def test_empty_bytes_are_valid(self) -> None:
        # The vacuous case every decoder agrees on: b"".decode("utf-8") == "".
        assert utf8_is_valid(b"") is True

    def test_the_return_is_a_genuine_bool_on_both_paths(self) -> None:
        # The bool return is the design: no str materialized, no exception
        # flow, pinned as the exact type on the valid and the invalid path.
        assert type(utf8_is_valid(b"abc")) is bool
        assert type(utf8_is_valid(b"\xff")) is bool
