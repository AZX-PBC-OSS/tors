"""Contract gate for the bytes-in surface: ``tors.decode_utf8`` and
``tors.finalize_utf8`` must be indistinguishable from the stdlib expressions they
replace: ``raw.decode("utf-8", errors=...)`` and
``finalize(raw.decode("utf-8", errors=...))``, over arbitrary bytes, including
every ill-formed class (truncated tails, overlongs, lone continuations,
surrogate-encoded sequences).

Why byte-exact parity is the whole product: extraction pipelines swap their
``bytes.decode`` + normalize + hash tail onto these calls, so any divergence (a
decoded character, a U+FFFD count, or the fields of the raised
``UnicodeDecodeError``) would silently change ingested text and content hashes.

Measured evidence this parity exists (CPython 3.12 vs the Rust core, pre-implementation):
a 54-case malformed battery + 14,069 biased-random inputs for ``replace`` (0
divergences: Rust's ``from_utf8_lossy`` and CPython's decoder implement the same
maximal-subpart substitution) and 13,658 invalid inputs for the strict error
spans/reasons (0 mismatches: ``start`` = Rust's ``valid_up_to``, ``end`` =
``start + error_len`` or the input length when truncated, and the reason is the
lead-byte test: 0xC2..=0xF4 at the error position → "invalid continuation byte",
anything else → "invalid start byte", no ``error_len`` → "unexpected end of
data"). The hypothesis pins below re-run that differential on every CI matrix
leg, so a future interpreter or std change that breaks it fails loudly instead of
silently changing stored digests.

Decisions pinned here (the errors= contract, from the spec):
- An ``errors`` value outside ``{"strict", "replace"}`` raises ``ValueError``:
  the closed-set-of-strings convention of ``unicodedata.normalize`` ("invalid
  normalization form"), not ``TypeError``: the value has the right type.
- ``finalize_utf8`` accepts both ``"strict"`` (default) and ``"replace"``: the
  spec's own rationale is that this is "the shape an extraction pipeline wants
  (tolerant text reads use ``errors="replace"``)"; a strict-only ``finalize_utf8``
  would force replace reads back into the two-call shape the function exists to
  collapse. (The spec's signature annotation wrote the default as
  ``errors: "strict" = "strict"``; the deviation is documented in the report.)
- The argument must be exactly ``bytes`` (``bytearray`` / ``memoryview`` / ``str``
  → ``TypeError``): pyo3's ``&[u8]`` extraction is a zero-copy borrow of an
  immutable ``PyBytes`` buffer, and the whole pass runs GIL-released, so a writable
  buffer could be mutated by another thread mid-read, which is a data race, not a
  semantic difference. (``bytes.decode`` accepting bytearray is a GIL-held call;
  tors is not.)
"""

from __future__ import annotations

import hashlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reference import (  # noqa: I001 -- the shared oracle module (tests/reference.py)
    pathological_text,
    reference_finalize,
)
from tors import decode_utf8, finalize, finalize_utf8

# The malformed battery from the measured parity work: every ill-formed class,
# each named for the CPython behavior it pins. (Values built from ints, pure
# ASCII source.)
_TRUNCATED_TAIL = b"\xf0\x9f"
_TRUNCATED_TAIL_AFTER_VALID = b"\xf0\x9f\x98\x80\xf0\x9f"
_OVERLONG_TWO = b"\xc0\x80"
_OVERLONG_THREE = b"\xe0\x80\x80"
_OVERLONG_FOUR = b"\xf0\x80\x80\x80"
_SURROGATE_ENCODED = b"\xed\xa0\x80"
_CESU8_SURROGATE_PAIR = b"\xed\xa0\x80\xed\xb0\x80"
_LONE_CONTINUATION = b"\x80"
_INVALID_LEAD = b"\xff"
_LEGACY_FIVE_BYTE_LEAD = b"\xfc\x84\x80\x80\x80\x80"
_MAXIMAL_SUBPART_BEFORE_NONCONTINUATION = b"\xf0\x9f\x41"
_TWO_BYTE_LEAD_BAD_CONTINUATION = b"fo\xd8o"

_MALFORMED_BATTERY: list[tuple[bytes, str]] = [
    (b"a\xc3", "truncated 2-byte tail"),
    (_TRUNCATED_TAIL, "truncated 4-byte tail"),
    (_TRUNCATED_TAIL_AFTER_VALID, "truncated tail after a valid emoji"),
    (_OVERLONG_TWO, "overlong 2-byte encoding of NUL"),
    (_OVERLONG_THREE, "overlong 3-byte encoding"),
    (_OVERLONG_FOUR, "overlong 4-byte encoding"),
    (_SURROGATE_ENCODED, "surrogate-encoded D800 (CESU-8)"),
    (_CESU8_SURROGATE_PAIR, "CESU-8 surrogate pair"),
    (b"\xed\xbf\xbf", "surrogate-encoded DFFF"),
    (_LONE_CONTINUATION, "lone continuation byte"),
    (b"a\x80\x80b", "two lone continuations mid-text"),
    (_INVALID_LEAD, "invalid lead byte 0xFF"),
    (_LEGACY_FIVE_BYTE_LEAD, "legacy 5-byte lead (rejected as invalid start)"),
    (_MAXIMAL_SUBPART_BEFORE_NONCONTINUATION, "maximal subpart before a non-continuation"),
    (b"\xf0\x9f\x98\x41", "3-byte maximal subpart before 'A'"),
    (b"\xe0\xa0", "valid-prefix pair truncated at end"),
    (_TWO_BYTE_LEAD_BAD_CONTINUATION, "2-byte lead then non-continuation"),
    (b"\xf4\x90", "out-of-range continuation after F4"),
    (b"ok\xffok", "invalid lead mid-ASCII"),
    (b"\xed\x9f\xbf" + b"\xed\xa0\x80", "valid U+D7FF then surrogate-encoded D800"),
]


class TestStrictParityOverArbitraryBytes:
    @given(st.binary(max_size=64))
    @settings(max_examples=500)
    def test_strict_decode_equals_stdlib_decode_or_raises_the_same_exception(
        self, raw: bytes
    ) -> None:
        """The strict contract in one property: for any bytes, either both the
        stdlib and tors return the identical str, or both raise ``UnicodeDecodeError``
        with identical ``start`` / ``end`` / ``reason`` / ``str()`` / ``object`` /
        ``encoding``: byte-exact results and exception-exact failures."""
        try:
            expected = raw.decode("utf-8")
        except UnicodeDecodeError as expected_exc:
            with pytest.raises(UnicodeDecodeError) as excinfo:
                decode_utf8(raw)
            got = excinfo.value
            assert (got.start, got.end, got.reason) == (
                expected_exc.start,
                expected_exc.end,
                expected_exc.reason,
            )
            assert str(got) == str(expected_exc)
            assert got.object == expected_exc.object == raw
            assert got.encoding == expected_exc.encoding == "utf-8"
        else:
            assert decode_utf8(raw) == expected

    @given(st.binary(max_size=64))
    @settings(max_examples=500)
    def test_replace_decode_is_byte_exact_with_stdlib_replace(self, raw: bytes) -> None:
        """``errors="replace"`` parity: identical str, which pins the U+FFFD count
        and placement (the maximal-subpart semantics), not just "some
        replacement happened"."""
        assert decode_utf8(raw, errors="replace") == raw.decode("utf-8", "replace")


class TestMalformedShapeBattery:
    @pytest.mark.parametrize(
        ("raw", "because"),
        [(raw, because) for raw, because in _MALFORMED_BATTERY],
        ids=[because for _, because in _MALFORMED_BATTERY],
    )
    def test_strict_raises_the_interpreters_own_unicode_decode_error(
        self, raw: bytes, because: str
    ) -> None:
        """Every ill-formed class, one pin per shape: tors's exception matches the
        running interpreter's own ``decode`` exception field-for-field (so each
        CI matrix leg pins its own CPython's behavior), with the type pinned
        explicitly; raising some other exception, or returning wrong data, fails
        here."""
        with pytest.raises(UnicodeDecodeError) as stdlib_exc:
            raw.decode("utf-8")
        with pytest.raises(UnicodeDecodeError) as tors_exc:
            decode_utf8(raw)
        got, expected = tors_exc.value, stdlib_exc.value
        assert (got.start, got.end, got.reason, str(got)) == (
            expected.start,
            expected.end,
            expected.reason,
            str(expected),
        )
        assert got.object == raw

    def test_truncated_tail_reason_and_span_are_cpythons_verbatim(self) -> None:
        # Literally-pinned spans/reasons for the three reason classes: the
        # classification table from the measured parity work, so a regression
        # reads as "reason X changed", not just "some field differs".
        with pytest.raises(UnicodeDecodeError, match="unexpected end of data") as excinfo:
            decode_utf8(_TRUNCATED_TAIL)
        assert (excinfo.value.start, excinfo.value.end) == (0, 2)

        with pytest.raises(UnicodeDecodeError, match="invalid start byte") as excinfo:
            decode_utf8(_LONE_CONTINUATION)
        assert (excinfo.value.start, excinfo.value.end) == (0, 1)

        with pytest.raises(UnicodeDecodeError, match="invalid continuation byte") as excinfo:
            decode_utf8(_SURROGATE_ENCODED)
        assert (excinfo.value.start, excinfo.value.end) == (0, 1)

    @pytest.mark.parametrize(
        ("raw", "because"),
        [(raw, because) for raw, because in _MALFORMED_BATTERY],
        ids=[because for _, because in _MALFORMED_BATTERY],
    )
    def test_finalize_utf8_strict_raises_the_same_error_as_decode_utf8(
        self, raw: bytes, because: str
    ) -> None:
        """``finalize_utf8`` relies only on the 300-example hypothesis
        property for its adversarial coverage, plausible but not
        guaranteed to reproduce every specific named ill-formed shape (the
        CESU-8 surrogate pair, the legacy 5-byte lead, ...). This runs the
        same deterministic battery ``decode_utf8`` is pinned against, so a
        regression scoped to ``finalize_utf8``'s own wrapper can't hide
        behind "the shared core is tested via the sibling function"."""
        with pytest.raises(UnicodeDecodeError) as decode_exc:
            decode_utf8(raw)
        with pytest.raises(UnicodeDecodeError) as finalize_exc:
            finalize_utf8(raw)
        got, expected = finalize_exc.value, decode_exc.value
        assert (got.start, got.end, got.reason, str(got)) == (
            expected.start,
            expected.end,
            expected.reason,
            str(expected),
        )

    @pytest.mark.parametrize(
        ("raw", "because"),
        [(raw, because) for raw, because in _MALFORMED_BATTERY],
        ids=[because for _, because in _MALFORMED_BATTERY],
    )
    def test_finalize_utf8_replace_matches_finalize_of_the_replace_decode(
        self, raw: bytes, because: str
    ) -> None:
        assert finalize_utf8(raw, errors="replace") == finalize(raw.decode("utf-8", "replace"))

    def test_replace_counts_replacement_chars_like_cpython(self) -> None:
        # The measured CPython layouts: constraint violations (overlong,
        # surrogate, out-of-range) emit one U+FFFD per byte; maximal subparts
        # emit one for the whole subpart and keep the terminating byte.
        assert decode_utf8(_OVERLONG_THREE, errors="replace") == "\ufffd" * 3
        assert decode_utf8(_SURROGATE_ENCODED, errors="replace") == "\ufffd" * 3
        assert decode_utf8(_CESU8_SURROGATE_PAIR, errors="replace") == "\ufffd" * 6
        assert decode_utf8(_LEGACY_FIVE_BYTE_LEAD, errors="replace") == "\ufffd" * 6
        assert decode_utf8(_MAXIMAL_SUBPART_BEFORE_NONCONTINUATION, errors="replace") == "\ufffdA"
        assert decode_utf8(_TRUNCATED_TAIL, errors="replace") == "\ufffd"
        assert decode_utf8(_TWO_BYTE_LEAD_BAD_CONTINUATION, errors="replace") == "fo\ufffdo"


class TestBehavioralCases:
    def test_empty_bytes_decode_to_the_empty_string(self) -> None:
        assert decode_utf8(b"") == ""

    def test_ascii_bytes_decode_unchanged(self) -> None:
        # The fast path real extraction corpora hit almost everywhere.
        assert decode_utf8(b"hello world, plain ASCII text.") == "hello world, plain ASCII text."

    def test_valid_multibyte_round_trips(self) -> None:
        raw = "caf\u00e9 na\u00efve \U0001f600 text".encode("utf-8")
        assert decode_utf8(raw) == raw.decode("utf-8")

    def test_errors_defaults_to_strict(self) -> None:
        # No errors= → strict: invalid input raises rather than silently replacing.
        with pytest.raises(UnicodeDecodeError):
            decode_utf8(_INVALID_LEAD)

    def test_positional_errors_argument_is_rejected(self) -> None:
        # errors= is keyword-only, matching bytes.decode's spelling of the same
        # parameter (positional decode args are the encoding there, and tors has
        # none; refusing the position keeps the parameter free for a future
        # signature extension).
        with pytest.raises(TypeError):
            decode_utf8(b"", "replace")  # type: ignore[misc]

    def test_finalize_utf8_positional_errors_argument_is_rejected(self) -> None:
        """``finalize_utf8`` shares the identical ``*, errors: Literal[...]``
        signature (python/tors/__init__.pyi), pinned for ``decode_utf8``
        above but never for its sibling."""
        with pytest.raises(TypeError):
            finalize_utf8(b"", "replace")  # type: ignore[misc]


class TestErrorsParameterContract:
    @pytest.mark.parametrize(
        "errors", ["ignore", "backslashreplace", "surrogateescape", "", "STRICT"]
    )
    def test_unknown_error_names_raise_value_error(self, errors: str) -> None:
        """A closed set of strings: right type, wrong value → ValueError, the
        ``unicodedata.normalize`` convention for exactly this kind of parameter
        (vs LookupError for codecs' arbitrary handler registry, which tors is
        not; it implements two decoders, not a handler lookup)."""
        with pytest.raises(ValueError, match="errors"):
            decode_utf8(b"", errors=errors)
        with pytest.raises(ValueError, match="errors"):
            finalize_utf8(b"", errors=errors)

    def test_both_documented_error_names_are_accepted(self) -> None:
        assert decode_utf8(b"abc", errors="strict") == "abc"
        assert decode_utf8(b"abc", errors="replace") == "abc"
        assert finalize_utf8(b"abc", errors="strict") == finalize("abc")
        assert finalize_utf8(b"abc", errors="replace") == finalize("abc")


class TestFinalizeUtf8Contract:
    @given(st.binary(max_size=64))
    @settings(max_examples=300)
    def test_strict_equals_finalize_of_stdlib_decode_or_raises_the_same_error(
        self, raw: bytes
    ) -> None:
        """``finalize_utf8(raw) == finalize(raw.decode("utf-8"))`` for valid input
        (the composition the one-call shape promises, cross-checked against the
        pure-Python oracle), and the stdlib's own ``UnicodeDecodeError`` (fields
        and all) for invalid input."""
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as expected_exc:
            with pytest.raises(UnicodeDecodeError) as excinfo:
                finalize_utf8(raw)
            got = excinfo.value
            assert (got.start, got.end, got.reason, str(got)) == (
                expected_exc.start,
                expected_exc.end,
                expected_exc.reason,
                str(expected_exc),
            )
        else:
            assert finalize_utf8(raw) == finalize(text)
            assert finalize_utf8(raw) == reference_finalize(text)

    @given(st.binary(max_size=64))
    @settings(max_examples=300)
    def test_replace_equals_finalize_of_stdlib_replace_decode(self, raw: bytes) -> None:
        text = raw.decode("utf-8", "replace")
        assert finalize_utf8(raw, errors="replace") == finalize(text)
        assert finalize_utf8(raw, errors="replace") == reference_finalize(text)

    @given(st.text(max_size=200))
    @settings(max_examples=300)
    def test_bytes_in_agrees_with_str_in_for_any_text_the_old_api_accepts(self, text: str) -> None:
        """Cross-surface consistency: for every str tors.normalize/finalize accept,
        the bytes-in path over its UTF-8 encoding returns the identical pair,
        a caller can switch either way without changing any stored digest."""
        assert finalize_utf8(text.encode("utf-8")) == finalize(text)

    @given(pathological_text())
    @settings(max_examples=300)
    def test_bytes_in_agrees_with_str_in_on_pathological_whitespace_text(self, text: str) -> None:
        assert finalize_utf8(text.encode("utf-8")) == finalize(text)

    def test_finalize_utf8_hash_is_of_the_normalized_decoded_text(self) -> None:
        # The tail must hash the pipeline output over the decoded text, not the
        # raw bytes: "a \t\n" decodes and normalizes to "a", so the digest is
        # sha256("a"), pinned against the stdlib expression.
        normalized, digest = finalize_utf8(b"a \t\n")
        assert normalized == "a"
        assert digest == hashlib.sha256(b"a").hexdigest()

    def test_replace_flavor_is_one_call_not_decode_then_finalize(self) -> None:
        # The one-call shape an extraction pipeline wants: replace-reads collapse
        # into a single native call, pinned by parity with the two-step expression.
        raw = b"caf\xc3\xa9 \xe0\x80\x80 trailing \xff"
        assert finalize_utf8(raw, errors="replace") == finalize(raw.decode("utf-8", "replace"))


class TestBytesOnlyArgumentContract:
    """tors takes exactly ``bytes``; see the module docstring for why writable or
    non-bytes buffers are refused (zero-copy immutable borrow across a
    GIL-released pass). The pin: anything else is a ``TypeError`` naming the
    argument, the same class of failure as passing a str to bytes.decode's
    receiver."""

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
            decode_utf8(not_bytes)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            finalize_utf8(not_bytes)  # type: ignore[arg-type]
