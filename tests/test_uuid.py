"""Contract gate for the UUIDv7 helper trio: ``tors.uuid7_timestamp_ms``,
``tors.uuid_version``, ``tors.uuid_parse`` (issue #54).

The consumer shape these exist for: a store keyed by UUIDv7 IDs (TaskQ's
enqueue path and its admin keyset cursors are the measured instance) ends up
reimplementing the same three bit operations everywhere -- pull the 48-bit
unix-millisecond timestamp out of the leading six bytes for time-bucketed
queries and keyset pagination, check the version nibble before trusting that
timestamp, and re-parse canonical text IDs into raw bytes at the boundary.
Each is trivial bit work (16 bytes in, integer out), which is exactly why it
gets reimplemented slightly wrong each time; this module pins tors's spelling
of them against a stdlib oracle.

The oracle is local to this file (the ``test_merkle.py`` precedent: a
reference only one gate consumes lives beside the gate, not in
``tests/reference.py``). It has two halves:

- Manual bit extraction over constructed buffers (the primary oracle, the
  issue's own instruction): v7 buffers built from known timestamps
  ``((ms << 80) | (7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b)`` per
  RFC 9562 section 5.7's layout, cross-checked against
  ``int.from_bytes(b[:6], "big")``; arbitrary buffers for the version nibble
  against ``b[6] >> 4``.
- The stdlib ``uuid`` module where it is strict enough to be an oracle:
  ``str(uuid.UUID(bytes=b))`` is always canonical lowercase-hyphenated text
  (the round-trip invariant ``uuid_parse(str(uuid.UUID(bytes=b))) == b``
  holds for arbitrary 16-byte buffers), and ``uuid.UUID(s).bytes`` parses
  exactly the canonical forms tors accepts. ``UUID.version`` is NOT used as
  an oracle except for real ``uuid.uuid4()`` values: the property is
  variant-gated on newer stdlibs and version-dependent on older ones, while
  tors's ``uuid_version`` answers the field question (the nibble) for every
  UUID regardless of variant -- a deliberate contract difference, not parity.

Strict-canonical contract, pinned as PINS NOT PARITY (the
``test_b64_decode.py`` documented-divergence doctrine): the stdlib
``uuid.UUID`` accepts braced text, ``urn:uuid:`` prefixes, hyphen-less hex,
and uppercase (all verified against the running interpreter below);
``uuid_parse`` and the str spellings of the int-out pair reject every one of
those, ``ValueError`` naming the problem and the accepted form. The
validation-primitive contract: a caller using ``uuid_parse`` as a gate wants
one grammar, the canonical one, not the stdlib's permissive union.
"""

from __future__ import annotations

import datetime
import time
import uuid

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import uuid7_timestamp_ms, uuid_parse, uuid_version

# The fixed UUIDv7 of the docs/api.md example (pinned in TestDocExamples):
# timestamp field 1_750_000_000_000 ms (2025-06-15T17:06:40Z), version 7,
# RFC 4122 variant, fixed rand fields -- deterministic, so the doc's outputs
# are literals, not elided. Built by the same RFC 9562 layout the oracle
# builders below use.
DOC_V7_TS_MS = 1_750_000_000_000
DOC_V7_TEXT = "01977420-dc00-7abc-9def-98765432100f"
DOC_V7_BYTES = bytes.fromhex("01977420dc007abc9def98765432100f")


def _v7_bytes(ms: int, rand_a: int = 0xABC, rand_b: int = 0x1DEF98765432100F) -> bytes:
    """A UUIDv7 buffer from its parts, RFC 9562 section 5.7's layout: 48-bit
    big-endian unix-millisecond field, version 7, variant 0b10, caller-fixed
    rand fields (deterministic: same inputs, same UUID)."""
    assert 0 <= ms < 2**48
    assert 0 <= rand_a < 2**12
    assert 0 <= rand_b < 2**62
    value = (ms << 80) | (7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return value.to_bytes(16, "big")


def _version_bytes(version: int, variant: int = 0b10) -> bytes:
    """A 16-byte buffer with the given version nibble (and RFC 4122 variant
    by default): the version-field probe shape, everything else zero."""
    assert 0 <= version <= 15
    value = (version << 76) | (variant << 62)
    return value.to_bytes(16, "big")


# --- the shared hypothesis strategies ----------------------------------------


_ms_strategy = st.integers(min_value=0, max_value=2**48 - 1)
_rand_a_strategy = st.integers(min_value=0, max_value=2**12 - 1)
_rand_b_strategy = st.integers(min_value=0, max_value=2**62 - 1)
_version_strategy = st.integers(min_value=0, max_value=15)
_arbitrary_uuid_bytes = st.binary(min_size=16, max_size=16)


class TestUuid7TimestampExtraction:
    """The 48-bit big-endian unix-millisecond field: the keyset-pagination /
    time-bucketed-query primitive."""

    @given(ms=_ms_strategy, rand_a=_rand_a_strategy, rand_b=_rand_b_strategy)
    @settings(max_examples=300)
    def test_returns_the_timestamp_the_buffer_was_built_from(
        self, ms: int, rand_a: int, rand_b: int
    ) -> None:
        buffer = _v7_bytes(ms, rand_a, rand_b)
        assert uuid7_timestamp_ms(buffer) == ms

    @given(ms=_ms_strategy, rand_a=_rand_a_strategy, rand_b=_rand_b_strategy)
    @settings(max_examples=200)
    def test_agrees_with_the_manual_big_endian_read_of_the_leading_six_bytes(
        self, ms: int, rand_a: int, rand_b: int
    ) -> None:
        # The independent spelling of the same field: int.from_bytes(b[:6],
        # "big") over the hypothesis-generated v7 buffers.
        buffer = _v7_bytes(ms, rand_a, rand_b)
        assert uuid7_timestamp_ms(buffer) == int.from_bytes(buffer[:6], "big")

    def test_boundary_zero_milliseconds(self) -> None:
        assert uuid7_timestamp_ms(_v7_bytes(0)) == 0

    def test_boundary_max_48_bit_field(self) -> None:
        # 2**48 - 1 ms is ~year 8921: the far-future edge of the field.
        assert uuid7_timestamp_ms(_v7_bytes(2**48 - 1)) == 2**48 - 1

    def test_a_real_now_timestamp_round_trips(self) -> None:
        now_ms = int(time.time() * 1000)
        assert 0 <= now_ms < 2**48  # the guard every real date sits inside
        assert uuid7_timestamp_ms(_v7_bytes(now_ms)) == now_ms

    def test_the_doc_examples_fixed_v7(self) -> None:
        assert uuid7_timestamp_ms(DOC_V7_BYTES) == DOC_V7_TS_MS

    @given(ms=_ms_strategy)
    @settings(max_examples=100)
    def test_canonical_text_input_agrees_with_bytes_input(self, ms: int) -> None:
        buffer = _v7_bytes(ms)
        text = str(uuid.UUID(bytes=buffer))
        assert uuid7_timestamp_ms(text) == uuid7_timestamp_ms(buffer) == ms

    def test_the_timestamp_field_reads_as_a_utc_datetime(self) -> None:
        # The time-bucket query the primitive exists for, spelled out: the
        # field is unix milliseconds, so fromtimestamp(ms / 1000) is the
        # ID's creation instant (docs/api.md's example, pinned).
        moment = datetime.datetime.fromtimestamp(DOC_V7_TS_MS / 1000, datetime.UTC)
        assert moment == datetime.datetime(2025, 6, 15, 17, 6, 40, tzinfo=datetime.UTC)


class TestUuid7TimestampVersionGuard:
    """The version guard: the 48-bit field is only defined for version 7, and
    asking for it of any other version is a ValueError naming the version
    found (the parse_errors_mode shape: the problem and the accepted form)."""

    @pytest.mark.parametrize("version", [v for v in range(16) if v != 7])
    def test_every_non_v7_version_raises_naming_the_version(self, version: int) -> None:
        with pytest.raises(ValueError, match=f"got version {version}"):
            uuid7_timestamp_ms(_version_bytes(version))

    def test_the_nil_uuid_raises_version_zero_not_zero_timestamp(self) -> None:
        # All-zero bytes: the timestamp field reads 0, but the version nibble
        # is 0 too, so the guard fires rather than answering a meaningless 0.
        with pytest.raises(ValueError, match="got version 0"):
            uuid7_timestamp_ms(bytes(16))

    def test_the_max_uuid_raises_version_fifteen(self) -> None:
        with pytest.raises(ValueError, match="got version 15"):
            uuid7_timestamp_ms(b"\xff" * 16)

    @pytest.mark.parametrize("version", [v for v in range(16) if v != 7])
    def test_the_text_spelling_of_a_non_v7_uuid_raises_the_same_error(
        self, version: int
    ) -> None:
        text = str(uuid.UUID(bytes=_version_bytes(version)))
        with pytest.raises(ValueError, match=f"got version {version}"):
            uuid7_timestamp_ms(text)

    def test_the_error_names_the_accepted_form(self) -> None:
        with pytest.raises(ValueError, match="must be a version 7 UUID"):
            uuid7_timestamp_ms(_version_bytes(4))


class TestUuidVersion:
    """The version nibble (byte 6's high half): the field itself, answered for
    any UUID of any version and any variant. No variant check -- the variant
    is a different field (byte 8's top two bits) and out of scope."""

    @pytest.mark.parametrize("version", range(16))
    def test_every_version_nibble_round_trips(self, version: int) -> None:
        assert uuid_version(_version_bytes(version)) == version

    @pytest.mark.parametrize("version", range(16))
    def test_every_version_nibble_in_canonical_text_form(self, version: int) -> None:
        text = str(uuid.UUID(bytes=_version_bytes(version)))
        assert uuid_version(text) == version

    def test_the_nil_uuid_is_version_zero(self) -> None:
        assert uuid_version(bytes(16)) == 0

    def test_the_max_uuid_is_version_fifteen(self) -> None:
        assert uuid_version(b"\xff" * 16) == 15

    @given(buffer=_arbitrary_uuid_bytes)
    @settings(max_examples=300)
    def test_matches_the_manual_nibble_extraction_over_arbitrary_buffers(
        self, buffer: bytes
    ) -> None:
        assert uuid_version(buffer) == buffer[6] >> 4

    @given(buffer=_arbitrary_uuid_bytes)
    @settings(max_examples=200)
    def test_text_form_agrees_with_bytes_form_over_arbitrary_buffers(
        self, buffer: bytes
    ) -> None:
        text = str(uuid.UUID(bytes=buffer))
        assert uuid_version(text) == uuid_version(buffer) == buffer[6] >> 4

    def test_real_uuid4_values_agree_with_the_stdlib_version_property(self) -> None:
        # The one stdlib .version differential that is stable on every
        # supported interpreter (version 4, RFC 4122 variant: universally 4).
        for _ in range(50):
            u = uuid.uuid4()
            assert uuid_version(u.bytes) == u.version == 4
            assert uuid_version(str(u)) == 4

    def test_a_v7_agrees_with_the_stdlib_version_property_on_this_interpreter(self) -> None:
        # Recorded, not relied on: uuid.UUID(bytes=...).version is variant-
        # gated on newer stdlibs; this pins the v7 case where both answer.
        u = uuid.UUID(bytes=DOC_V7_BYTES)
        assert u.version == 7
        assert uuid_version(DOC_V7_BYTES) == 7

    def test_no_variant_check_is_performed(self) -> None:
        # Same version nibble, four different variants (including the
        # reserved ones the stdlib's .version property refuses to answer):
        # the nibble comes back unchanged, the field question only.
        for variant in (0b00, 0b01, 0b10, 0b11):
            assert uuid_version(_version_bytes(9, variant)) == 9


class TestUuidParse:
    """Canonical text to the 16 raw bytes, strict: the validation-primitive
    direction (a caller gating IDs at a boundary wants one grammar)."""

    def test_the_fixed_v7_vector(self) -> None:
        assert uuid_parse(DOC_V7_TEXT) == DOC_V7_BYTES

    @given(buffer=_arbitrary_uuid_bytes)
    @settings(max_examples=300)
    def test_round_trips_the_stdlib_canonical_rendering_of_arbitrary_buffers(
        self, buffer: bytes
    ) -> None:
        # str(uuid.UUID(bytes=b)) is canonical lowercase-hyphenated text for
        # every 16-byte buffer, so the round trip is an invariant over
        # arbitrary bytes, not just well-formed UUIDs.
        text = str(uuid.UUID(bytes=buffer))
        assert uuid_parse(text) == buffer

    @given(buffer=_arbitrary_uuid_bytes)
    @settings(max_examples=200)
    def test_agrees_with_the_stdlib_parse_where_the_stdlib_is_strict_enough(
        self, buffer: bytes
    ) -> None:
        # uuid.UUID(s).bytes over the canonical grammar: the parity lane.
        # Where the stdlib is LOOSER than tors (braces, urn, no hyphens,
        # uppercase) the divergence is a pinned rejection below, not parity.
        text = str(uuid.UUID(bytes=buffer))
        assert uuid_parse(text) == uuid.UUID(text).bytes

    def test_output_is_exactly_sixteen_bytes(self) -> None:
        assert len(uuid_parse(DOC_V7_TEXT)) == 16

    def test_parse_then_version_then_timestamp_compose(self) -> None:
        raw = uuid_parse(DOC_V7_TEXT)
        assert uuid_version(raw) == 7
        assert uuid7_timestamp_ms(raw) == DOC_V7_TS_MS


class TestStrictCanonicalRejections:
    """The rejection battery: exactly one grammar is accepted -- 36
    characters, hyphens at 8/13/18/23, lowercase hex elsewhere -- and every
    other shape raises ValueError naming the problem (positions are
    0-based). Each message is pinned with `match=` so the error path cannot
    drift into vagueness."""

    @pytest.mark.parametrize(
        "bad_text",
        [
            "{" + DOC_V7_TEXT + "}",  # braces: the stdlib accepts, tors does not
            "urn:uuid:" + DOC_V7_TEXT,  # the URN prefix: likewise
            DOC_V7_TEXT.replace("-", ""),  # 32-char hyphen-less hex: likewise
            DOC_V7_TEXT.upper(),  # uppercase: likewise (the loudest divergence)
            DOC_V7_TEXT + "0",  # one character long
            DOC_V7_TEXT[:-1],  # one character short
            "",  # empty
            DOC_V7_TEXT + "\n",  # trailing whitespace
            " " + DOC_V7_TEXT,  # leading whitespace
            DOC_V7_TEXT[:5] + "g" + DOC_V7_TEXT[6:],  # non-hex at a hex slot
            "-" + DOC_V7_TEXT[1:],  # stray hyphen where a hex digit belongs
            DOC_V7_TEXT[:13] + "a" + DOC_V7_TEXT[14:],  # missing hyphen at 13
            DOC_V7_TEXT[:8] + "a" + DOC_V7_TEXT[9:],  # missing hyphen at 8
            DOC_V7_TEXT[:18] + "0" + DOC_V7_TEXT[19:],  # missing hyphen at 18
            DOC_V7_TEXT[:23] + "0" + DOC_V7_TEXT[24:],  # missing hyphen at 23
            DOC_V7_TEXT[:10] + "C" + DOC_V7_TEXT[11:],  # single uppercase letter
            DOC_V7_TEXT.replace("-", ":"),  # wrong separator everywhere
        ],
        ids=[
            "braces",
            "urn-prefix",
            "no-hyphens",
            "uppercase",
            "37-chars",
            "35-chars",
            "empty",
            "trailing-newline",
            "leading-space",
            "non-hex-char",
            "stray-hyphen",
            "missing-hyphen-13",
            "missing-hyphen-8",
            "missing-hyphen-18",
            "missing-hyphen-23",
            "single-uppercase",
            "colons-as-separators",
        ],
    )
    @pytest.mark.parametrize("parse", [uuid_parse, uuid_version, uuid7_timestamp_ms])
    def test_every_non_canonical_text_raises_value_error(
        self, parse: object, bad_text: str
    ) -> None:
        with pytest.raises(ValueError):
            parse(bad_text)  # type: ignore[operator]

    def test_wrong_length_message_names_the_accepted_form_and_the_count(self) -> None:
        with pytest.raises(ValueError, match=r"exactly 36 characters .* got 35"):
            uuid_parse(DOC_V7_TEXT[:-1])
        with pytest.raises(ValueError, match="got 45"):
            uuid_parse("urn:uuid:" + DOC_V7_TEXT)
        with pytest.raises(ValueError, match="got 32"):
            uuid_parse(DOC_V7_TEXT.replace("-", ""))

    def test_missing_hyphen_message_names_the_position_and_the_found_character(self) -> None:
        with pytest.raises(ValueError, match=r"expected '-' at position 13, found 'a'"):
            uuid_parse(DOC_V7_TEXT[:13] + "a" + DOC_V7_TEXT[14:])

    def test_stray_hyphen_message_names_the_position(self) -> None:
        with pytest.raises(ValueError, match=r"found '-' at position 0, where a hex digit"):
            uuid_parse("-" + DOC_V7_TEXT[1:])

    def test_uppercase_message_says_uppercase_and_that_the_stdlib_is_looser(self) -> None:
        # The most surprising rejection (uuid.UUID accepts it!) gets the
        # most explicit message: the found character, its position, and the
        # deliberate divergence from the stdlib.
        with pytest.raises(
            ValueError, match=r"uppercase 'C' at position 10 .* stdlib .* does not"
        ):
            uuid_parse(DOC_V7_TEXT[:10] + "C" + DOC_V7_TEXT[11:])

    def test_non_hex_message_names_the_character_and_position(self) -> None:
        with pytest.raises(ValueError, match=r"found 'g' at position 5"):
            uuid_parse(DOC_V7_TEXT[:5] + "g" + DOC_V7_TEXT[6:])

    @pytest.mark.parametrize("length", [0, 1, 15, 17, 32, 35, 37, 64])
    def test_wrong_bytes_length_message_names_the_count(self, length: int) -> None:
        with pytest.raises(ValueError, match=f"exactly 16 bytes for a UUID, got {length}"):
            uuid_version(b"\x07" * length)

    def test_wrong_bytes_length_raises_for_all_three_entry_points(self) -> None:
        # The int-out pair share the bytes-argument contract with uuid_parse's
        # cousins: 15 bytes is a ValueError wherever bytes enter.
        with pytest.raises(ValueError, match="got 15"):
            uuid_version(bytes(15))
        with pytest.raises(ValueError, match="got 15"):
            uuid7_timestamp_ms(bytes(15))


class TestArgumentContract:
    """Exactly ``bytes`` or ``str`` for the int-out pair (the bytes-in
    surface's exactly-``bytes`` contract, test_b64.py's precedent, extended
    by the str spelling), exactly ``str`` for ``uuid_parse``; anything else
    is TypeError naming what was received."""

    @pytest.mark.parametrize(
        "not_value",
        [123, None, 1.5, bytearray(DOC_V7_BYTES), memoryview(DOC_V7_BYTES), [DOC_V7_BYTES]],
        ids=["int", "none", "float", "bytearray", "memoryview", "list"],
    )
    @pytest.mark.parametrize("call", [uuid_version, uuid7_timestamp_ms])
    def test_non_bytes_or_str_raises_type_error_for_the_int_out_pair(
        self, call: object, not_value: object
    ) -> None:
        # bytearray/memoryview are deliberate: the bytes-in surface borrows
        # the immutable bytes buffer zero-copy, and writable views would
        # race the detached read (the test_b64.py TestBytesOnlyArgumentContract
        # doctrine, unchanged by the union).
        with pytest.raises(TypeError, match="value must be bytes or str"):
            call(not_value)  # type: ignore[operator]

    def test_type_error_names_the_received_type(self) -> None:
        with pytest.raises(TypeError, match="not bytearray"):
            uuid_version(bytearray(DOC_V7_BYTES))
        with pytest.raises(TypeError, match="not int"):
            uuid_version(123)

    @pytest.mark.parametrize(
        "not_str",
        [DOC_V7_BYTES, bytearray(DOC_V7_BYTES), memoryview(DOC_V7_BYTES), 123, None],
        ids=["bytes", "bytearray", "memoryview", "int", "none"],
    )
    def test_uuid_parse_rejects_non_str_with_type_error(self, not_str: object) -> None:
        # uuid_parse is the text->bytes direction only: bytes in is a
        # TypeError, not an identity (the identity is the caller's to spell).
        with pytest.raises(TypeError):
            uuid_parse(not_str)  # type: ignore[arg-type]

    def test_lone_surrogate_str_raises_unicode_encode_error(self) -> None:
        # The str-in borrow's standard behavior (the chunk_text_iter
        # precedent): a str that cannot be UTF-8-borrowed fails the borrow
        # itself, before any grammar check could run.
        for call in (uuid_version, uuid7_timestamp_ms, uuid_parse):
            with pytest.raises(UnicodeEncodeError):
                call("\ud800" * 36)  # type: ignore[arg-type]


class TestStdlibDivergencePins:
    """Where the stdlib is looser than tors, the divergence is recorded as a
    pin, not parity: each stdlib-accepted loose form is proven accepted by
    the stdlib AND rejected by tors in the same test, so the divergence is
    deliberate and visible, never an accident of implementation."""

    @pytest.mark.parametrize(
        "loose_form",
        [
            "{" + DOC_V7_TEXT + "}",
            "urn:uuid:" + DOC_V7_TEXT,
            DOC_V7_TEXT.replace("-", ""),
            DOC_V7_TEXT.upper(),
        ],
        ids=["braces", "urn", "no-hyphens", "uppercase"],
    )
    def test_stdlib_accepts_what_tors_rejects(self, loose_form: str) -> None:
        # The stdlib side of the pin: uuid.UUID parses it to the same bytes.
        assert uuid.UUID(loose_form).bytes == DOC_V7_BYTES
        # The tors side: the same text is a canonical-grammar rejection.
        with pytest.raises(ValueError):
            uuid_parse(loose_form)
        with pytest.raises(ValueError):
            uuid_version(loose_form)
        with pytest.raises(ValueError):
            uuid7_timestamp_ms(loose_form)

    def test_the_stdlib_hex_property_is_a_tors_rejection(self) -> None:
        # u.hex (32 lowercase hex chars, no hyphens) is another stdlib-accepted
        # spelling tors refuses: canonical means hyphenated.
        u = uuid.UUID(bytes=DOC_V7_BYTES)
        assert uuid.UUID(u.hex).bytes == DOC_V7_BYTES
        with pytest.raises(ValueError, match="got 32"):
            uuid_parse(u.hex)


class TestDocExamples:
    """docs/api.md's UUIDv7-helper example, pinned: the doc's literals are
    re-derived here against the built extension (the test_docs_examples.py
    discipline, kept local to this gate since that file is scoped to the
    #28 features). If these fail after an intentional change, the doc and
    this pin move together, in the same commit."""

    def test_the_api_md_example(self) -> None:
        u = uuid.UUID(DOC_V7_TEXT)
        assert uuid_version(u.bytes) == 7
        assert uuid7_timestamp_ms(u.bytes) == DOC_V7_TS_MS
        assert uuid7_timestamp_ms(str(u)) == DOC_V7_TS_MS
        assert uuid_parse(str(u)) == u.bytes
        with pytest.raises(ValueError, match="uppercase"):
            uuid_parse(str(u).upper())
        moment = datetime.datetime.fromtimestamp(uuid7_timestamp_ms(u.bytes) / 1000, datetime.UTC)
        assert moment == datetime.datetime(2025, 6, 15, 17, 6, 40, tzinfo=datetime.UTC)
