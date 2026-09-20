"""Contract gate for the random-generation family: ``random_string``,
``random_hex``, ``random_b62``, ``random_b64url``, ``uuid4``, ``uuid7``,
and the uuids' bytes spellings ``uuid4_bytes``/``uuid7_bytes``.

The security contract under test, stated once here because every surface
repeats it: unseeded calls draw fresh OS-entropy bytes on every call (no
process or thread RNG state, so fork-safe, matching ``secrets``' semantics)
and are safe for keys/tokens/secrets; ``seed=`` switches to a deterministic
ChaCha20 stream that is a pure function of (seed, arguments) and therefore
fully predictable — a test/fixture tool, never safe for secrets.

The family is length-first: every token spelling takes the OUTPUT length
directly ("I want a base62 id X characters long" is the whole call), and
``random_hex``/``random_b62``/``random_b64url`` are all ``random_string``
over their fixed alphabets — one char-sampling engine (Lemire, no modulo
bias) for the four string spellings, one byte-fill engine (the uuid
crate's builders) for ``uuid4``/``uuid7``. ``random_b64url`` is the
opaque-token contract — uniform over the 64-char urlsafe alphabet, every
position unconstrained — NOT a base64 encoding of random bytes (an
encoding's final character is constrained, and padding is an encoding
concept with no spelling here).

Determinism is pinned by an independent oracle, not by frozen literals
alone: ``_oracle_string``/``_oracle_uuid4`` re-implement the DOCUMENTED
construction in pure Python — rand_core's ``seed_from_u64`` derivation
(PCG32; see its comment for the spec-vs-source finding), the RFC 8439
ChaCha20 block stream, and Lemire's nearly-divisionless unbiased sampling
— so a seeded pin asserts the extension equals the construction the docs
promise, never merely "whatever the implementation emitted". The oracle
shrank with the length-first refactor: hex/b64url were byte-fill+encode
transcriptions before it and are string-engine spellings now, so their
pins run through ``_oracle_string`` over the respective alphabets, and
only ``uuid4`` keeps the byte view (``_oracle_bytes``). Committed literal
pins (recomputed from the oracle BEFORE the implementation changed, so
they pin the construction, not the code) then guard cross-version
stability independently; the b64 family's RFC 4648 vectors are the
precedent for that double pinning.

Two cross-references the old byte-fill engines supported are dead, noted
here so nobody re-pins them by accident: "uuid4(seed=0)'s first hex
digits are hex(8, seed=0)'s" and "b64url(3, seed=0) decodes to
hex(8, seed=0)'s first bytes" were one-shared-stream artifacts of
hex/b64url byte-filling — hex char-samples now, the byte path is the
uuids' alone, and uuid4's own goldens pin it. The "same distribution,
never literal equality" hex-vs-string subtlety died the same way: hex IS
the string engine, and the equality is pinned literally in
``TestOneEngineDelegation``.
"""

from __future__ import annotations

import base64
import binascii
import enum
import hashlib
import os
import secrets
import struct
import time
import uuid as stdlib_uuid
from collections.abc import Callable, Iterator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import (
    random_b62,
    random_b64url,
    random_hex,
    random_string,
    uuid4,
    uuid4_bytes,
    uuid7,
    uuid7_bytes,
)

# The charset literals this file pins against, defined locally and spelled in
# full rather than imported: the canonical-charset constants on the sibling
# CHARSET_* branch are not a dependency of this one, and the two branches must
# merge without import coupling (merge note: if that branch lands first, keep
# these literals as the pin and let its constants be cross-checked against
# them, not the other way around).
HEX_CHARS = "0123456789abcdef"
BASE62_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
B64URL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

_MASK64 = (1 << 64) - 1


# --- The oracle: the documented construction, re-implemented in Python -------
#
# Every piece below is transcribed from the resolved crate sources (registry
# paths cited per function), not from memory; the oracle's whole value is that
# it is an independent second implementation of the spec the docs states.

_PCG_MUL = 6364136223846793005
_PCG_INC = 11634580027462260723


def _seed_from_u64(seed: int) -> bytes:
    """rand_core's ``SeedableRng::seed_from_u64`` default derivation, verified
    against rand_core's source: a PCG32 (XSH-RR) generator keyed by the seed
    emits the 32-byte key as 8 little-endian u32s, advancing once per word.
    (The feature spec called this a SplitMix64 derivation; the resolved source
    says PCG32 — a spec-vs-source finding recorded in docs/api.md, and the
    oracle follows the source, which is what the extension actually runs.)
    Documented-stable upstream: the derivation is part of rand_core's
    semver-stable surface, so seeded output is reproducible across versions.
    """
    state = seed & _MASK64
    out = bytearray()
    for _ in range(8):
        state = (state * _PCG_MUL + _PCG_INC) & _MASK64
        xorshifted = (((state >> 18) ^ state) >> 27) & 0xFFFFFFFF
        rot = (state >> 59) & 31
        x = ((xorshifted >> rot) | (xorshifted << (32 - rot))) & 0xFFFFFFFF
        out += x.to_bytes(4, "little")
    return bytes(out)


_CHACHA_CONSTANTS = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)


def _rotl32(x: int, n: int) -> int:
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def _quarter_round(s: list[int], a: int, b: int, c: int, d: int) -> None:
    s[a] = (s[a] + s[b]) & 0xFFFFFFFF
    s[d] = _rotl32(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & 0xFFFFFFFF
    s[b] = _rotl32(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b]) & 0xFFFFFFFF
    s[d] = _rotl32(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & 0xFFFFFFFF
    s[b] = _rotl32(s[b] ^ s[c], 7)


def _chacha20_block(key: bytes, counter: int) -> bytes:
    """One 64-byte ChaCha20 block, the RFC 8439 core over rand_chacha's
    ChaCha20Rng layout (verified in its source): constants, 8 key words
    (little-endian from the 32-byte key), a 64-bit block counter (words 12-13),
    and the 64-bit stream id in words 14-15 (always zero for ``seed_from_u64``
    seeding, which never opens a stream)."""
    key_words = struct.unpack("<8I", key)
    state = [
        *_CHACHA_CONSTANTS,
        *key_words,
        counter & 0xFFFFFFFF,
        (counter >> 32) & 0xFFFFFFFF,
        0,
        0,
    ]
    working = state.copy()
    for _ in range(10):
        _quarter_round(working, 0, 4, 8, 12)
        _quarter_round(working, 1, 5, 9, 13)
        _quarter_round(working, 2, 6, 10, 14)
        _quarter_round(working, 3, 7, 11, 15)
        _quarter_round(working, 0, 5, 10, 15)
        _quarter_round(working, 1, 6, 11, 12)
        _quarter_round(working, 2, 7, 8, 13)
        _quarter_round(working, 3, 4, 9, 14)
    return struct.pack("<16I", *((working[i] + state[i]) & 0xFFFFFFFF for i in range(16)))


def _oracle_bytes(n: int, seed: int) -> bytes:
    """The first ``n`` bytes of the seeded stream: block 0, block 1, ... in
    order — exactly what one ``fill_bytes`` over an n-byte buffer consumes.
    The uuids' byte view (the only byte-fill consumers left in the family
    after the length-first refactor)."""
    key = _seed_from_u64(seed)
    out = bytearray()
    counter = 0
    while len(out) < n:
        out += _chacha20_block(key, counter)
        counter += 1
    return bytes(out[:n])


def _oracle_words(seed: int) -> Iterator[int]:
    """The seeded stream as u64 words, little-endian, in stream order — the
    sampler's consumption contract (each word = 8 consecutive stream bytes)."""
    key = _seed_from_u64(seed)
    block = b""
    pos = 0
    counter = 0
    while True:
        if pos + 8 > len(block):
            block = _chacha20_block(key, counter)
            counter += 1
            pos = 0
        yield int.from_bytes(block[pos : pos + 8], "little")
        pos += 8


def _lemire_below(words: Iterator[int], n: int) -> int:
    """Lemire's nearly-divisionless unbiased draw from [0, n): multiply a
    fresh u64 by n, keep the high 64 bits of the 128-bit product, and reject
    the draw when the low 64 bits fall below 2^64 mod n (the only values that
    make some outputs one u64-heavier than others). Rejection probability is
    n/2^64 (< 2^-58 for any alphabet a str can hold), so a redraw never
    happens in practice but the no-modulo-bias property holds by
    construction, not by luck."""
    threshold = (1 << 64) % n
    while True:
        x = next(words)
        m = x * n
        low = m & _MASK64
        if low >= n:
            return m >> 64
        if low >= threshold:
            return m >> 64


def _oracle_string(length: int, alphabet: str, seed: int) -> str:
    chars = list(alphabet)
    words = _oracle_words(seed)
    return "".join(chars[_lemire_below(words, len(chars))] for _ in range(length))


def _oracle_uuid4_bytes(seed: int) -> bytes:
    """The seeded uuid4's 16 pre-format bytes: the first 16 stream bytes
    with the version-4 and RFC 4122 variant nibbles set — the uuid crate's
    ``Builder::from_random_bytes`` field layout, transcribed (byte 6
    ``(b & 0x0f) | 0x40``, byte 8 ``(b & 0x3f) | 0x80``). This is the
    construction ``uuid4_bytes`` owes (and ``uuid4`` formats)."""
    raw = bytearray(_oracle_bytes(16, seed))
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return bytes(raw)


def _oracle_uuid4(seed: int) -> str:
    return str(stdlib_uuid.UUID(bytes=_oracle_uuid4_bytes(seed)))


# --- Shared shape helpers -----------------------------------------------------


def _assert_uuid_shape(value: str, version: str) -> None:
    assert len(value) == 36
    assert value[8] == value[13] == value[18] == value[23] == "-"
    assert value[14] == version
    assert value[19] in "89ab"
    assert all(c in HEX_CHARS for c in value.replace("-", ""))


# The four token spellings as one callable shape, for the argument-contract
# parametrizations below (they share the length-first argument, so they share
# every length-contract test).
_TOKEN_CALLS: list[tuple[str, Callable[[int], str]]] = [
    ("random_string", lambda length: random_string(length, "ab")),
    ("random_hex", random_hex),
    ("random_b62", random_b62),
    ("random_b64url", random_b64url),
]


class _IndexLike:
    """A non-int object presenting an int through ``__index__`` (numpy's
    typed-scalar shape, ``IntEnum`` minus the inheritance): the int-like
    convention the family's int-ish params accept — pinned in
    ``TestSeedDomain``, where the seed's alignment to it lives."""

    def __init__(self, value: int) -> None:
        self._value = value

    def __index__(self) -> int:
        return self._value


class IntEnumLike(enum.IntEnum):
    """The int-instance side of the int-like gate: subclasses int."""

    X = 8


class TestSeededConstructionMatchesTheOracle:
    """Every seeded function equals the documented construction at the pin
    parameters: seeds 0, 1 (the edge goldens), and 42, across sizes that
    exercise each engine's paths (empty, single block, multi-block, odd hex
    lengths — a 31-char hex id is legal —, non-multiple-of-4 b64url lengths
    including the 43-char JWT-sig shape, multibyte and single-char
    alphabets)."""

    @pytest.mark.parametrize("seed", [0, 1, 42], ids=["seed0", "seed1", "seed42"])
    @pytest.mark.parametrize(
        "length",
        [0, 1, 7, 8, 16, 31, 32, 64, 65, 512],
        ids=["0", "1", "7", "8", "16", "31", "32", "64", "65", "512"],
    )
    def test_hex(self, length: int, seed: int) -> None:
        # The string engine over the hex alphabet: odd lengths are first-class
        # (31, 65 in the ladder), no byte pairs anywhere.
        assert random_hex(length, seed=seed) == _oracle_string(length, HEX_CHARS, seed)

    @pytest.mark.parametrize("seed", [0, 1, 42], ids=["seed0", "seed1", "seed42"])
    @pytest.mark.parametrize(
        ("length", "alphabet"),
        [
            (0, "ab"),
            (1, "ab"),
            (16, "ab"),
            (32, HEX_CHARS),
            (22, BASE62_CHARS),
            (9, "éüß漢"),
            (8, "x"),
            (200, BASE62_CHARS),
        ],
        ids=["empty", "one-char", "binary", "hex", "b62", "multibyte", "single", "long"],
    )
    def test_string(self, length: int, alphabet: str, seed: int) -> None:
        assert random_string(length, alphabet, seed=seed) == _oracle_string(length, alphabet, seed)

    @pytest.mark.parametrize("seed", [0, 1, 42], ids=["seed0", "seed1", "seed42"])
    @pytest.mark.parametrize("length", [0, 1, 16, 32, 128], ids=["0", "1", "16", "32", "128"])
    def test_b62_delegates_to_the_string_engine(self, length: int, seed: int) -> None:
        # ONE engine: b62 is the random_string core over the base62 alphabet,
        # so it must equal both the direct spelling and the oracle.
        assert random_b62(length, seed=seed) == random_string(length, BASE62_CHARS, seed=seed)
        assert random_b62(length, seed=seed) == _oracle_string(length, BASE62_CHARS, seed)

    @pytest.mark.parametrize("seed", [0, 1, 42], ids=["seed0", "seed1", "seed42"])
    @pytest.mark.parametrize(
        "length",
        [0, 1, 2, 3, 4, 22, 43, 64, 65, 512],
        ids=["0", "1", "2", "3", "4", "22", "43", "64", "65", "512"],
    )
    def test_b64url(self, length: int, seed: int) -> None:
        # Every residue mod 4 present (43 = 3, 22 = 2, 65 = 1): the token
        # contract has no alignment constraint at all.
        assert random_b64url(length, seed=seed) == _oracle_string(length, B64URL_CHARS, seed)

    @pytest.mark.parametrize("seed", [0, 1, 42], ids=["seed0", "seed1", "seed42"])
    def test_uuid4(self, seed: int) -> None:
        assert uuid4(seed=seed) == _oracle_uuid4(seed)

    @pytest.mark.parametrize("seed", [0, 1, 42], ids=["seed0", "seed1", "seed42"])
    def test_uuid4_bytes(self, seed: int) -> None:
        # The byte path's pre-format buffer, the same construction the
        # string spelling formats: 16 stream bytes, fields set.
        assert uuid4_bytes(seed=seed) == _oracle_uuid4_bytes(seed)


class TestSeededGoldenLiterals:
    """The computed outputs at chosen parameters, committed as exact
    literals and sha256 digests — cross-version regression pins that stand
    independently of the oracle (every literal below was computed from the
    oracle BEFORE the length-first refactor landed and then matched by the
    implementation; a ChaCha/rand_core semantic change breaks these loudly,
    which is the point: it is a finding, not a flake).

    What these pins do NOT prove, stated so nobody over-reads them: they
    pin DETERMINISM (same seed, same output, cross-version) — never
    unbiasedness. A modulo-biased transcription (`x % n` on the same
    words) reproduces most of these literals' prefixes without ever
    rejecting, because the rejection region is < 2^-58 per draw; the
    unbiasedness claim is owned by ``TestUniformityAtScale`` (chi-square
    bounds plus exact-count digests that a ``%`` transcription fails in
    full) and the crate-side reject-path unit, not by any golden here.

    The internal cross-references the byte-fill engines used to carry are
    gone with them: "uuid4(seed=0)'s hex digits are hex(8, seed=0)'s" and
    "b64url(3, seed=0) decodes to hex(8, seed=0)'s first bytes" were
    one-shared-stream artifacts of hex/b64url consuming raw stream bytes —
    hex char-samples now, so the relationship is false and stays unpinned
    (uuid4's own goldens below still pin the byte path). What replaces them
    as the family's structural pin is the delegation equality in
    ``TestOneEngineDelegation``.
    """

    def test_hex_edge_goldens(self) -> None:
        assert random_hex(0, seed=0) == ""
        assert random_hex(0, seed=42) == ""
        assert random_hex(8, seed=0) == "0fd2e314"
        assert random_hex(8, seed=1) == "9286e6a4"
        assert random_hex(16, seed=42) == "861225d7151bf9b1"
        # Odd lengths are legal and pinned: a 31-char hex id is a real shape.
        assert random_hex(31, seed=5) == "9fddbe3cca89ec73270d1f133677747"
        assert (
            random_hex(64, seed=7)
            == "08f4267dbf6ea8fbab86463bb680c70710e85e4f03affac31420c55574847728"
        )

    def test_hex_1kib_golden_via_digest(self) -> None:
        # 1 KiB of hex output (1024 characters): pinned by digest, the b64
        # family's large-vector idiom.
        digest = hashlib.sha256(random_hex(1024, seed=7).encode("ascii")).hexdigest()
        assert digest == "c2ffd75057a19d125653cb1bfbb94d59d81a80e1389676f41e12c74ca054f3ae"

    def test_string_goldens(self) -> None:
        assert random_string(8, "x", seed=99) == "xxxxxxxx"
        assert random_string(16, "ab", seed=1) == "babababaababbbaa"
        assert random_string(32, HEX_CHARS, seed=42) == "861225d7151bf9b14a3617ab9b534d19"
        assert random_string(9, "éüß漢", seed=2) == "éüßßé漢漢éé"

    def test_b62_goldens(self) -> None:
        assert random_b62(0, seed=0) == ""
        assert random_b62(22, seed=0) == "1yrBtE6FUlG59Zjj3K2vVn"
        assert random_b62(22, seed=1) == "c9XQvNdIRcHcYnEMMFqaNP"
        digest = hashlib.sha256(random_b62(4096, seed=3).encode("ascii")).hexdigest()
        assert digest == "122e60ff5ff0014f68a63dccc3c5928b919cf44e907220458e8b88a4db2c51db"

    def test_b64url_goldens(self) -> None:
        assert random_b64url(0, seed=0) == ""
        assert random_b64url(4, seed=0) == "B-3L"
        assert random_b64url(22, seed=42) == "gaGKKW0cEXGu8nuERoMZFe"
        # The 43-char JWT-sig-shaped token (what token_urlsafe(32) formats as).
        assert random_b64url(43, seed=7) == "Bi8SLaf0s_a4pi-vqthbTaOstZjDweDcEC5hW7S_CNp"
        digest = hashlib.sha256(random_b64url(65536, seed=5).encode("ascii")).hexdigest()
        assert digest == "3431c0493a841fd2dd334956fd90ebe6e0fc447a0506af6f7fb9e2953cf00779"

    def test_uuid4_goldens(self) -> None:
        assert uuid4(seed=0) == "b2f7f581-d6de-4c06-a822-fd6e7e8265fb"
        assert uuid4(seed=1) == "9a374450-4560-439e-8670-b7a17d492b27"
        assert uuid4(seed=42) == "7848b5d7-11bc-4883-9963-17a3f9c90269"

    def test_the_hex_and_string_goldens_are_one_engine(self) -> None:
        # The direct spelling of the delegation pin at the golden points:
        # hex(32, seed=42) IS string(32, hex, seed=42) — the same literal
        # the string goldens commit, not a coincidence of two engines.
        assert random_hex(32, seed=42) == random_string(32, HEX_CHARS, seed=42)
        assert random_hex(32, seed=42) == "861225d7151bf9b14a3617ab9b534d19"


class TestOneEngineDelegation:
    """The engine story after the length-first refactor, pinned literally:
    ``random_hex``/``random_b62``/``random_b64url`` are ``random_string``
    over their alphabets — one char-sampling engine for all four token
    spellings — so the same seed gives the SAME output through the named
    spelling and the explicit ``random_string`` spelling (literal equality,
    by construction: the named spellings delegate). The uuids are the
    family's other engine (byte fills handed to the uuid crate's builders),
    pinned by their own goldens above.

    What died here, noted so nobody re-pins it: the old
    "random_hex(n, seed=s) and random_string(2n, hex, seed=s) agree in
    distribution but never in value" subtlety (byte-fill vs char-sampling
    stream consumption) is obsolete — hex IS the string engine, at the same
    length. The bucket-count statistics that proved the two engines agreed
    in distribution died with it; uniformity is pinned structurally by the
    oracle equality above (the oracle transcribes Lemire's rejection, so
    any modulo shortcut breaks it).
    """

    @pytest.mark.parametrize("seed", [0, 1, 42, 2**64 - 1], ids=["0", "1", "42", "max"])
    @pytest.mark.parametrize("length", [0, 1, 16, 128], ids=["0", "1", "16", "128"])
    def test_hex_is_random_string_over_the_hex_alphabet(self, length: int, seed: int) -> None:
        assert random_hex(length, seed=seed) == random_string(length, HEX_CHARS, seed=seed)

    @pytest.mark.parametrize("seed", [0, 1, 42, 2**64 - 1], ids=["0", "1", "42", "max"])
    @pytest.mark.parametrize("length", [0, 1, 16, 128], ids=["0", "1", "16", "128"])
    def test_b64url_is_random_string_over_the_urlsafe_alphabet(
        self, length: int, seed: int
    ) -> None:
        assert random_b64url(length, seed=seed) == random_string(length, B64URL_CHARS, seed=seed)

    def test_seeded_output_is_prefix_continuous_across_lengths(self) -> None:
        # The char engine consumes u64 words in order, so a longer seeded
        # draw extends a shorter one character-for-character (the property
        # the old byte-fill engines pinned on the byte path; the byte path
        # is uuid-only now, and uuid4's fixed 16-byte draw has no length
        # ladder to pin).
        assert random_hex(64, seed=5)[:32] == random_hex(32, seed=5)
        assert random_b64url(43, seed=9)[:22] == random_b64url(22, seed=9)
        assert random_b62(64, seed=5)[:16] == random_b62(16, seed=5)


class TestUuidBytesSpellings:
    """``uuid4_bytes``/``uuid7_bytes``: the uuids' pre-format 16 bytes as
    first-class returns — the same buffer the string spellings format, no
    canonical formatting (no hyphens, no lowercase-hex dance), exposed
    because the surveyed consumers re-wrap the str back into exactly this
    shape anyway (``UUID(bytes=...)`` at the id-assignment call sites,
    ``.hex()[:12]`` timestamp slicing): one native draw instead of
    draw-format-reparse. The uuid paths are fixed-size (16 bytes), so the
    length-scaled allocation class (TestMemoryBound) does not exist here."""

    def test_uuid4_bytes_is_the_buffer_behind_the_uuid4_string(self) -> None:
        # The DRY pin: uuid4(seed=s) IS the canonical hyphenated form of
        # uuid4_bytes(seed=s) — same 16 bytes, one construction; the string
        # spelling formats what the bytes spelling returns.
        for seed in (0, 1, 42, 2**64 - 1):
            raw = uuid4_bytes(seed=seed)
            assert str(stdlib_uuid.UUID(bytes=raw)) == uuid4(seed=seed)

    def test_uuid4_bytes_goldens(self) -> None:
        # The uuid4 goldens' own bytes, unhyphenated — the same literals
        # the string pins commit, pinned as the buffer itself.
        assert uuid4_bytes(seed=0).hex() == "b2f7f581d6de4c06a822fd6e7e8265fb"
        assert uuid4_bytes(seed=1).hex() == "9a3744504560439e8670b7a17d492b27"
        assert uuid4_bytes(seed=42).hex() == "7848b5d711bc4883996317a3f9c90269"

    def test_unseeded_uuid4_bytes_shapes(self) -> None:
        # Version nibble at byte 6's high half, RFC 4122 variant at byte
        # 8's high nibble (8..=0xb) — the field layout, on the raw buffer.
        for _ in range(64):
            b = uuid4_bytes()
            assert len(b) == 16
            assert b[6] >> 4 == 4
            assert 8 <= b[8] >> 4 <= 11

    def test_unseeded_uuid7_bytes_shapes(self) -> None:
        for _ in range(64):
            b = uuid7_bytes()
            assert len(b) == 16
            assert b[6] >> 4 == 7
            assert 8 <= b[8] >> 4 <= 11

    def test_uuid7_bytes_timestamp_is_the_callers_now(self) -> None:
        # The first 6 bytes big-endian are the 48-bit Unix-epoch
        # millisecond field — uuid7()'s own caller-visible contract, on
        # the bytes spelling.
        for _ in range(8):
            before = time.time() * 1000
            b = uuid7_bytes()
            after = time.time() * 1000
            ts = int.from_bytes(b[:6], "big")
            assert before - 5_000 <= ts <= after + 5_000

    def test_the_uuid_constructor_consumer_shape(self) -> None:
        # The surveyed re-wrap shape, pinned as a docs example too: the
        # stdlib constructor over the raw bytes carries version, variant,
        # and (for v7) the timestamp through intact.
        u7 = stdlib_uuid.UUID(bytes=uuid7_bytes())
        assert u7.version == 7
        assert u7.variant == stdlib_uuid.RFC_4122
        u4 = stdlib_uuid.UUID(bytes=uuid4_bytes())
        assert u4.version == 4
        assert u4.variant == stdlib_uuid.RFC_4122

    def test_the_hex_slice_consumer_shape(self) -> None:
        # The other surveyed shape: tors.uuid7_bytes().hex()[:12] is the
        # timestamp prefix — the 12 hex digits consumers slice off the
        # canonical string (u[:8] + u[9:13] there, .hex()[:12] here).
        before = time.time() * 1000
        prefix = uuid7_bytes().hex()[:12]
        after = time.time() * 1000
        ts = int(prefix, 16)
        assert before - 5_000 <= ts <= after + 5_000

    def test_unseeded_bytes_are_distinct(self) -> None:
        # Birthday arithmetic unchanged from the string spellings (122
        # random bits for v4; 74 per millisecond for v7): any repeat among
        # 2048 draws is a broken engine, not bad luck.
        assert len({uuid4_bytes() for _ in range(2048)}) == 2048
        assert len({uuid7_bytes() for _ in range(2048)}) == 2048

    def test_uuid4_bytes_seed_contract_matches_uuid4(self) -> None:
        # TypeError parity with uuid4: a non-int-like seed names the
        # parameter; the parameter is keyword-only (uuid4's own shape).
        with pytest.raises(TypeError, match=r"seed must be int-like \(__index__\) or None"):
            uuid4_bytes(seed="42")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            uuid4_bytes(1)  # type: ignore[misc,call-arg]

    def test_uuid7_bytes_takes_no_seed(self) -> None:
        # uuid7's own rationale, inherited: the timestamp is external
        # state, so there is no seed parameter to refuse or accept.
        with pytest.raises(TypeError):
            uuid7_bytes(seed=1)  # type: ignore[call-arg]


class TestSeedDomain:
    """The seed argument contract: any int-LIKE — an ``int`` instance (bools
    and IntEnums ride along as the ints they are) or any object implementing
    ``__index__`` (the house convention for int-ish params: ``length``
    already accepts these through pyo3's ``i64`` extraction, and the
    chunkers' size params and documents' ``max_bytes=`` keep the same rule)
    — reduced mod 2^64, so huge and negative seeds are legal and defined;
    anything else is a TypeError naming the parameter."""

    def test_same_seed_same_output_repeatably(self) -> None:
        for _ in range(3):
            assert random_hex(32, seed=123456789) == random_hex(32, seed=123456789)
            assert uuid4(seed=987654321) == uuid4(seed=987654321)

    def test_different_seeds_different_outputs(self) -> None:
        outputs = {random_hex(32, seed=s) for s in range(8)}
        assert len(outputs) == 8

    @pytest.mark.parametrize(
        ("seed", "equivalent"),
        [
            (42, 42 + (1 << 64)),
            (-1, (1 << 64) - 1),
            (7, 7 + (1 << 100)),
            (-(1 << 64) + 7, 7),
        ],
        ids=["plus-2^64", "negative-two-complement", "plus-2^100", "negative-wrap"],
    )
    def test_seed_is_reduced_mod_2_64(self, seed: int, equivalent: int) -> None:
        # Documented reduction: the 64-bit two's-complement value feeds the
        # derivation, exactly what Python's own & (2**64-1) computes.
        assert random_hex(32, seed=seed) == random_hex(32, seed=equivalent)
        assert random_string(16, BASE62_CHARS, seed=seed) == random_string(
            16, BASE62_CHARS, seed=equivalent
        )

    def test_bool_seed_is_the_int_it_is(self) -> None:
        # bool is an int in Python; True is 1 under the documented reduction.
        assert random_hex(8, seed=True) == random_hex(8, seed=1)
        assert random_hex(8, seed=False) == random_hex(8, seed=0)

    @pytest.mark.parametrize(
        "not_an_int",
        ["42", 3.5, b"123", [1], object()],
        ids=["str", "float", "bytes", "list", "object"],
    )
    def test_non_int_seed_raises_type_error(self, not_an_int: object) -> None:
        with pytest.raises(TypeError, match=r"seed must be int-like \(__index__\) or None"):
            random_hex(8, seed=not_an_int)  # type: ignore[arg-type]

    def test_index_like_seed_is_the_int_it_indexes_to(self) -> None:
        # The alignment pin (the finding: `length` accepted `__index__`
        # objects through pyo3's i64 extraction while `seed`'s strict
        # `cast::<PyInt>` refused them — the family's two int-ish params
        # disagreed). The house convention is acceptance (every size param
        # in the crate keeps it), so seed now rides the same gate: a
        # non-int object presenting an int through `__index__` (numpy's
        # scalar shape) is the seed it indexes to, on every seeded surface.
        assert random_hex(8, seed=_IndexLike(0)) == random_hex(8, seed=0)
        assert random_hex(8, seed=_IndexLike(42)) == random_hex(8, seed=42)
        assert random_string(16, BASE62_CHARS, seed=_IndexLike(42)) == random_string(
            16, BASE62_CHARS, seed=42
        )
        assert uuid4(seed=_IndexLike(42)) == uuid4(seed=42)

    def test_index_like_seed_reduces_mod_2_64_like_the_plain_int(self) -> None:
        # The mask path through the same gate: an `__index__` wider than
        # i64 (or negative) reduces mod 2^64 exactly like the plain-int
        # spelling of the same value.
        assert random_hex(8, seed=_IndexLike(7 + (1 << 100))) == random_hex(8, seed=7)
        assert random_hex(8, seed=_IndexLike(-1)) == random_hex(8, seed=-1)

    def test_int_enum_seed_is_the_int_it_is(self) -> None:
        # The int-instance side of the same gate: IntEnum subclasses int,
        # so it has always ridden along and still does.
        assert random_hex(8, seed=IntEnumLike.X) == random_hex(8, seed=8)

    def test_real_numpy_scalar_seed_is_the_int_it_holds(self) -> None:
        # L1: the runtime shape `_IndexLike` only models — a real numpy
        # integer (the motivating `__index__`-only scalar) reduces through
        # the same gate to the int it holds. Skipped where numpy is not
        # installed (no new dependency); the fakes above always run.
        np = pytest.importorskip("numpy")
        assert random_hex(8, seed=np.int64(42)) == random_hex(8, seed=42)
        assert uuid4(seed=np.int64(42)) == uuid4(seed=42)

    def test_a_raising_index_surfaces_its_own_error(self) -> None:
        # An `__index__` that RAISES surfaces its own error — the length
        # param's own behavior under pyo3 extraction — not a TypeError
        # about seed: the gate called the protocol and the protocol
        # answered.
        class Raising:
            def __index__(self) -> int:
                raise ValueError("no int today")

        with pytest.raises(ValueError, match="no int today"):
            random_hex(8, seed=Raising())  # type: ignore[arg-type]


class TestValueErrors:
    """The repo's size-argument taxonomy, now uniform across the four token
    spellings (they share the length-first argument, so they share its
    error): negative counts are ValueError naming the parameter and the
    accepted form (the chunkers' style), zero is legal and returns the
    empty string (secrets.token_hex(0)'s own shape), and there is no size
    cap by design — the memory bound is a catchable MemoryError
    (TestMemoryBound below), never a cap and never an abort."""

    @pytest.mark.parametrize("name", [name for name, _ in _TOKEN_CALLS])
    @pytest.mark.parametrize("length", [-1, -1000], ids=["-1", "-1000"])
    def test_negative_length_raises_value_error_naming_it(self, name: str, length: int) -> None:
        call = dict(_TOKEN_CALLS)[name]
        with pytest.raises(ValueError, match=f"length must be >= 0, got {length}"):
            call(length)

    def test_empty_alphabet_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="alphabet must be a non-empty str"):
            random_string(8, "")

    def test_zero_is_the_empty_string_everywhere(self) -> None:
        # secrets.token_hex(0) == "" is the parity anchor for this shape.
        assert secrets.token_hex(0) == ""
        assert random_hex(0) == ""
        assert random_b62(0) == ""
        assert random_string(0, "ab") == ""
        assert random_b64url(0) == ""

    @pytest.mark.parametrize("name", [name for name, _ in _TOKEN_CALLS])
    @pytest.mark.parametrize(
        "not_an_int", ["8", 3.5, b"8", None, [8]], ids=["str", "float", "bytes", "none", "list"]
    )
    def test_non_int_length_raises_type_error(self, name: str, not_an_int: object) -> None:
        call = dict(_TOKEN_CALLS)[name]
        with pytest.raises(TypeError):
            call(not_an_int)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "not_a_str",
        [123, None, b"ab", ["a"], {"a": 1}],
        ids=["int", "none", "bytes", "list", "dict"],
    )
    def test_non_str_alphabet_raises_type_error(self, not_a_str: object) -> None:
        with pytest.raises(TypeError):
            random_string(8, not_a_str)  # type: ignore[arg-type]

    def test_lone_surrogate_alphabet_raises_unicode_encode_error(self) -> None:
        # The repo-wide str contract: the standard to_str borrow rejects lone
        # surrogates with UnicodeEncodeError before any work runs.
        with pytest.raises(UnicodeEncodeError):
            random_string(8, "ab\ud800cd")

    @pytest.mark.parametrize("name", [name for name, _ in _TOKEN_CALLS])
    def test_the_length_ceiling_overflow_at_py_ssize_t(self, name: str) -> None:
        # MEDIUM-3's wording fix, pinned on every token spelling (4x2: the
        # four length-first calls, OverflowError at the Py_ssize_t boundary
        # plus MemoryError just inside it) — "no size cap" never meant
        # unbounded: the argument is Py_ssize_t (2^63 - 1 on 64-bit), and
        # memory is the bound inside it. 2^63 itself never reaches the core
        # (OverflowError at extraction); 2^63 - 1 reaches the reserve and
        # comes back as the catchable MemoryError.
        call = dict(_TOKEN_CALLS)[name]
        with pytest.raises(OverflowError):
            call(2**63)

    @pytest.mark.parametrize("name", [name for name, _ in _TOKEN_CALLS])
    def test_the_length_ceiling_memory_just_inside_py_ssize_t(self, name: str) -> None:
        call = dict(_TOKEN_CALLS)[name]
        with pytest.raises(MemoryError):
            call(2**63 - 1)

    def test_huge_alphabet_single_draw_cost_shape(self) -> None:
        # M4's cost pin at small scale: O(alphabet) to materialize plus
        # O(length) draws (documented in the core, the binding, and
        # docs/api.md) — a 100k-char alphabet with length 1 builds exactly
        # one output char, deterministically under seed=.
        alphabet = "".join(chr(0xE000 + i) for i in range(100_000))
        out = random_string(1, alphabet, seed=0)
        assert len(out) == 1
        assert out in alphabet
        assert out == random_string(1, alphabet, seed=0)

    def test_huge_outputs_complete_no_cap_by_design(self) -> None:
        # 512 KiB of hex output from one call: no size cap exists (the bound
        # is a catchable MemoryError, pinned in TestMemoryBound), and the
        # seeded spelling is digest-stable AND digest-pinned (the
        # large-vector idiom).
        out = random_hex(512 * 1024)
        assert len(out) == 512 * 1024
        assert set(out) <= set(HEX_CHARS)

        def seeded_digest() -> str:
            return hashlib.sha256(random_hex(512 * 1024, seed=3).encode()).hexdigest()

        assert seeded_digest() == seeded_digest()
        assert seeded_digest() == "516ff0a3860664d6bfc6dc5201012acef43da36cf7171203f888ed91e912f664"


class TestMemoryBound:
    """The oversized-length contract: no size cap exists, and the memory
    bound is a CATCHABLE ``MemoryError`` — the Python convention (``'x' * n``
    and ``secrets.token_hex(n)`` raise it too; the anchors are pinned below)
    — never a process abort. The abort is the bug this class pins dead: the
    core's old ``String::with_capacity`` raised SIGABRT on allocation
    failure (Rust's default handler, uncatchable, takes the interpreter
    with it); the reserve now runs through ``try_reserve``, which refuses
    oversized requests as a plain error the binding maps to ``MemoryError``
    — before any allocation is attempted, so these pins run in-process in
    the suite safely. The uuid paths are fixed-size (16-byte buffers) and
    stand outside this class entirely."""

    @pytest.mark.parametrize("name", [name for name, _ in _TOKEN_CALLS])
    def test_an_impossible_length_raises_memory_error_not_an_abort(self, name: str) -> None:
        # 2**62 output characters is past any 64-bit allocator's reach (the
        # userspace address space alone), so the reserve refuses and the
        # call raises, in-process, catchable. If this fix ever regresses the
        # process dies at the first parametrized case — loud, not subtle.
        call = dict(_TOKEN_CALLS)[name]
        with pytest.raises(MemoryError):
            call(2**62)

    def test_a_saturating_worst_case_is_the_same_catchable_shape(self) -> None:
        # A 4-byte-per-char alphabet at 2**61 saturates the worst-case byte
        # computation to usize::MAX; the reserve's own capacity check
        # refuses it without attempting any allocation — same MemoryError,
        # and the arithmetic never panics on the way there.
        with pytest.raises(MemoryError):
            random_string(2**61, "😀")

    def test_the_python_convention_anchors_raise_it_too(self) -> None:
        # The convention being matched, pinned: the stdlib spellings raise
        # the same catchable error for the same request, never abort.
        with pytest.raises(MemoryError):
            _ = "x" * (2**62)
        with pytest.raises(MemoryError):
            secrets.token_hex(2**62)


class TestUnseededOutputShape:
    """The unseeded contract: charset membership and exact lengths at any
    length (odd hex lengths, non-multiple-of-4 b64url lengths — the token
    contract has no alignment constraint), the distributional relationship
    to the stdlib spellings, and the canonical UUID field layout —
    properties of the output distribution, never pinned values (OS entropy
    is the source)."""

    @pytest.mark.parametrize(
        "length",
        [1, 2, 7, 16, 31, 32, 64, 1000],
        ids=["1", "2", "7", "16", "31", "32", "64", "1000"],
    )
    def test_hex_is_exact_length_lowercase_hex(self, length: int) -> None:
        out = random_hex(length)
        assert len(out) == length
        assert set(out) <= set(HEX_CHARS)

    @pytest.mark.parametrize("length", [1, 2, 16, 32, 64, 1000])
    def test_b62_is_exact_length_over_the_base62_alphabet(self, length: int) -> None:
        out = random_b62(length)
        assert len(out) == length
        assert set(out) <= set(BASE62_CHARS)

    @pytest.mark.parametrize(
        "length",
        [1, 2, 3, 4, 22, 42, 43, 44, 1000],
        ids=["1", "2", "3", "4", "22", "42", "43", "44", "1000"],
    )
    def test_b64url_is_exact_length_over_the_urlsafe_alphabet(self, length: int) -> None:
        out = random_b64url(length)
        assert len(out) == length
        assert set(out) <= set(B64URL_CHARS)
        # Padding was an encoding concept; the parameter is gone, so '=' can
        # never appear (also implied by the charset check — pinned explicitly
        # because the parameter's removal is the contract change).
        assert "=" not in out

    def test_token_hex_distributional_parity_with_secrets(self) -> None:
        # secrets.token_hex(n) and random_hex(2n) are the SAME uniform
        # distribution over 2n-char lowercase hex strings (every string
        # equally likely); they are independent draws from the OS source, so
        # never equal values — format parity is what is pinned, exactly the
        # split the old format-parity test stated. The 2n mapping is the
        # whole relationship: token_hex thinks in bytes, random_hex in
        # characters.
        for n in (1, 16, 32, 64):
            tors_out = random_hex(2 * n)
            stdlib_out = secrets.token_hex(n)
            assert len(tors_out) == len(stdlib_out) == 2 * n
            assert set(tors_out) <= set(HEX_CHARS)
            assert set(stdlib_out) <= set(HEX_CHARS)

    def test_the_jwt_shaped_length_matches_token_urlsafe_format(self) -> None:
        # secrets.token_urlsafe(32) formats as 43 urlsafe chars — the same
        # LENGTH class as random_b64url(43), the practical migration shape
        # (a caller generating 43-char JWT-material tokens today). Same
        # alphabet, same length; the contract difference is pinned below.
        assert len(secrets.token_urlsafe(32)) == 43
        assert len(random_b64url(43)) == 43
        assert set(secrets.token_urlsafe(32)) <= set(B64URL_CHARS)

    def test_b64url_is_a_token_not_an_encoding(self) -> None:
        # The honest boundary, pinned: random_b64url(L) is a uniform random
        # string over the 64-char urlsafe alphabet, every position
        # unconstrained — NOT "a valid base64 encoding of N random bytes".
        # Two proofs: a 41-char output (41 % 4 == 1) is legal here and is
        # not a decodable base64 payload at all; and at 43 chars (decodable-
        # shaped), the FINAL character ranges over the whole alphabet where
        # an encoding of 32 bytes could only ever show 16 distinct values
        # there (the last char carries the final byte's low 4 bits, shifted
        # into place; the 31-byte encoding is the 4-distinct case, its last
        # char carrying just the low 2 bits). Callers wanting encodable
        # random material should take random_hex of even length (byte-exact
        # via hex) — docs/api.md states the same boundary.
        out41 = random_b64url(41)
        assert len(out41) == 41
        with pytest.raises(binascii.Error):
            base64.urlsafe_b64decode(out41 + "===")
        last_chars = {random_b64url(43, seed=s)[-1] for s in range(64)}
        # Deterministic given the seeds (measured 41 distinct); an
        # encoding-shaped regression caps at 16 and fails this hard.
        assert len(last_chars) >= 32

    def test_the_encoding_side_final_char_arithmetic_is_what_the_docs_state(
        self,
    ) -> None:
        # The corrected arithmetic, pinned empirically against the stdlib
        # encoder (the red-team finding was this very number being wrong in
        # the docs): 5000 fresh 32-byte draws — coupon-collector arithmetic
        # says all 16 finals appear with overwhelming certainty (the chance
        # any one is missing is ~16 * (15/16)**5000, ~1e-215) — so exactly
        # 16 distinct final characters, each the final byte's low 4 bits
        # shifted; and the 31-byte case, the 4-distinct one, alongside it.
        finals32 = {
            base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=")[-1] for _ in range(5000)
        }
        assert len(finals32) == 16
        # Every one of the 16 is the low nibble shifted left two: the char
        # index is a multiple of 4 in the RFC 4648 §5 alphabet.
        b64url_bytes = B64URL_CHARS.encode("ascii")
        assert all(b64url_bytes.index(c) % 4 == 0 for c in finals32)
        finals31 = {
            base64.urlsafe_b64encode(os.urandom(31)).rstrip(b"=")[-1] for _ in range(2000)
        }
        assert len(finals31) == 4

    def test_uuid4_shape_on_unseeded_draws(self) -> None:
        for _ in range(64):
            _assert_uuid_shape(uuid4(), "4")

    def test_uuid7_shape(self) -> None:
        for _ in range(64):
            _assert_uuid_shape(uuid7(), "7")

    def test_ten_thousand_uuid4s_are_distinct(self) -> None:
        # Birthday arithmetic at 122 random bits: a collision among 10k draws
        # has probability ~1e-29 (n^2 / 2^123); any repeat is a broken
        # engine, not bad luck. (The same ~1e-29 the core docs state —
        # one numeral, pinned once here and stated once there.)
        assert len({uuid4() for _ in range(10_000)}) == 10_000

    def test_uuid7_timestamp_is_the_callers_now_within_5s(self) -> None:
        # The caller-visible uuid7 contract: the 48-bit timestamp field (the
        # first 12 hex digits, which in the canonical string are the two
        # dash-free groups u[:8] + u[9:13]) decodes to the Unix-epoch
        # millisecond timestamp of the call. NOT a monotonic counter —
        # same-millisecond calls differ only in the random tail, and clock
        # skew backwards flows straight through (uuid_utils' strict
        # monotonicity is a different product promise; see docs/api.md).
        # The window is ±5s: the timestamp is read inside the call between
        # the two host reads, so anything wider is a broken clock or a
        # broken builder, not tolerance.
        for _ in range(8):
            before = time.time() * 1000
            value = uuid7()
            after = time.time() * 1000
            ts = int(value[:8] + value[9:13], 16)
            assert before - 5_000 <= ts <= after + 5_000

    def test_uuid7_takes_no_seed_and_rejects_one(self) -> None:
        # The timestamp is external state: a seeded uuid7 would still vary
        # with time, so the parameter does not exist (documented in the
        # docstring and docs/api.md).
        with pytest.raises(TypeError):
            uuid7(seed=1)  # type: ignore[call-arg]


class TestHypothesisProperties:
    """The contract over arbitrary shapes: seeded determinism at any
    length/seed (draw-verified twice per example), arbitrary alphabets
    (multibyte included), the hex/b64url spellings as string-engine
    transcriptions of the oracle, and the unseeded b64url token shape."""

    @given(
        length=st.integers(min_value=0, max_value=64),
        seed=st.integers(min_value=-(2**70), max_value=2**70),
    )
    @settings(max_examples=150)
    def test_seeded_string_is_a_pure_function_of_seed_and_args(
        self, length: int, seed: int
    ) -> None:
        first = random_string(length, BASE62_CHARS, seed=seed)
        second = random_string(length, BASE62_CHARS, seed=seed)
        assert first == second
        assert first == _oracle_string(length, BASE62_CHARS, seed & _MASK64)

    @given(
        alphabet=st.text(min_size=1, max_size=12),
        length=st.integers(min_value=0, max_value=48),
        seed=st.integers(min_value=0, max_value=2**64 - 1),
    )
    @settings(max_examples=150)
    def test_arbitrary_alphabets_stay_inside_the_alphabet(
        self, alphabet: str, length: int, seed: int
    ) -> None:
        out = random_string(length, alphabet, seed=seed)
        assert len(out) == length
        assert set(out) <= set(alphabet)
        assert out == _oracle_string(length, alphabet, seed)

    @given(
        length=st.integers(min_value=0, max_value=96),
        seed=st.integers(min_value=0, max_value=2**64 - 1),
    )
    @settings(max_examples=150)
    def test_seeded_hex_and_b64url_are_the_string_engine(self, length: int, seed: int) -> None:
        # The oracle transcribes the char-sampling construction, so these
        # lanes pin both spellings against the documented engine at
        # arbitrary (length, seed) — including odd hex lengths and every
        # b64url residue mod 4.
        assert random_hex(length, seed=seed) == _oracle_string(length, HEX_CHARS, seed)
        assert random_b64url(length, seed=seed) == _oracle_string(length, B64URL_CHARS, seed)

    @given(length=st.integers(min_value=0, max_value=256))
    @settings(max_examples=100)
    def test_unseeded_b64url_is_exact_length_over_the_alphabet(self, length: int) -> None:
        # The old round-trip-through-the-decoder lane died with the encoding
        # contract: a uniform token is not an encoding of anything. What the
        # unseeded spelling owes is its own shape, at any length.
        out = random_b64url(length)
        assert len(out) == length
        assert set(out) <= set(B64URL_CHARS)
        assert "=" not in out

    @given(seed=st.integers(min_value=0, max_value=2**64 - 1))
    @settings(max_examples=75)
    def test_seeded_uuid4_matches_the_oracle(self, seed: int) -> None:
        # The byte path's only seeded consumer: 16 stream bytes, version and
        # variant nibbles set, canonical formatting.
        assert uuid4(seed=seed) == _oracle_uuid4(seed)

    @given(seed=st.integers(min_value=0, max_value=2**64 - 1))
    @settings(max_examples=75)
    def test_seeded_uuid4_bytes_matches_the_oracle(self, seed: int) -> None:
        # The same construction, pre-formatting: the buffer the string
        # spelling formats, at arbitrary seeds.
        assert uuid4_bytes(seed=seed) == _oracle_uuid4_bytes(seed)


class TestDocstringSecurityContract:
    """The security contract must be LOUD on every seeded surface: the
    docstring pins force the warning to survive refactors (the
    documents-surface TestDocstringHonesty discipline)."""

    @pytest.mark.parametrize(
        "func",
        [random_string, random_hex, random_b62, random_b64url, uuid4, uuid4_bytes],
        ids=[
            "random_string",
            "random_hex",
            "random_b62",
            "random_b64url",
            "uuid4",
            "uuid4_bytes",
        ],
    )
    def test_seeded_surfaces_carry_the_predictability_warning(self, func: object) -> None:
        # Whitespace-normalized: the docstring's phrases span the Rust doc
        # comment's line wraps, so the match is on words, not on wrap points.
        doc = " ".join((func.__doc__ or "").lower().split())
        assert "fully predictable from the seed" in doc
        assert "never safe for secrets" in doc
        assert "fresh bytes from the operating system" in doc
        assert "fork-safe" in doc

    def test_uuid7_documents_the_no_seed_rationale(self) -> None:
        doc = " ".join((uuid7.__doc__ or "").lower().split())
        assert "external state" in doc
        assert "probabilistically unique" in doc

    def test_uuid7_bytes_documents_the_no_seed_rationale(self) -> None:
        # The bytes spelling inherits the rationale and must carry it
        # itself: the timestamp is external state, so no seed parameter.
        doc = " ".join((uuid7_bytes.__doc__ or "").lower().split())
        assert "external state" in doc
        assert "probabilistically unique" in doc


class TestForkSafety:
    # The fork itself emits Python 3.12's DeprecationWarning (forking a
    # process whose thread pool has run) — the warning is the POINT: the
    # cell proves the post-fork child draws independent bytes anyway.
    # Silenced HERE, not globally: the suite's only warning, owned by the
    # one test that forks.
    pytestmark = pytest.mark.filterwarnings(
        "ignore::DeprecationWarning:.*os\\.fork.*",
    )

    """HIGH-1's positive control: the unseeded spelling must not replay
    across ``os.fork()`` — the failure a cached userspace RNG would show.

    Negative control, recorded here rather than run (it would poison the
    suite's own process with a cache): a ``ThreadRng``/``SmallRng``/``StdRng``
    parked in a thread-local or ``lazy_static``/``OnceLock`` replays the
    parent's stream in the child — the first N post-fork child draws
    equal the parent's next N draws. This test asserts the opposite for
    tors (disjoint parent/child token sets), and
    ``TestNoRngStateGuard`` pins the absence of the cache that would
    make the negative control pass.
    """

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="no os.fork on this platform")
    def test_fork_child_does_not_replay_the_parent_stream(self) -> None:
        n = 64
        parent_tokens = {random_hex(32) for _ in range(n)}
        assert len(parent_tokens) == n
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # child: draw, report over the pipe, exit cleanly.
            try:
                os.close(read_fd)
                child_tokens = [random_hex(32) for _ in range(n)]
                os.write(write_fd, "\n".join(child_tokens).encode("ascii"))
            finally:
                # os._exit, never sys.exit: no pytest teardown, no
                # stdio flush, no exception delivery in the child.
                os._exit(0)
        os.close(write_fd)
        chunks = []
        while True:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        child_tokens = b"".join(chunks).decode("ascii").split("\n")
        assert len(child_tokens) == n
        assert len(set(child_tokens)) == n
        assert set(child_tokens).isdisjoint(parent_tokens)


class TestNoRngStateGuard:
    """The grep guard behind the fork test: ``src/random_impl.rs`` must
    contain no thread- or process-level RNG cache — the state whose
    absence IS the fork-safety argument (inspection-only: no test can
    observe a cache it must not have, so the suite pins the source
    text instead). The ``fast-rng``/``lazy`` spellings are checked in
    prose form too, because a comment-only reference is how such a
    cache usually arrives (a "perf: cache the rng" follow-up)."""

    def test_no_cached_rng_state_in_the_core(self) -> None:
        import pathlib

        # H1: the no-state discipline lives in two source files plus the
        # manifest — a "perf: cache the rng" follow-up could land the cache
        # in the binding or the feature list as easily as in the core, so
        # the guard scans all three. Cargo.toml prose documents the declined
        # shapes (thread-rng/fast-rng named as rejected), so its scan is
        # declaration-scoped (tomllib, below) rather than textual.
        for rel in ("src/random_impl.rs", "src/py/random.rs"):
            source = (pathlib.Path(__file__).parent.parent / rel).read_text(encoding="utf-8")
            for forbidden in (
                "thread_rng",
                "ThreadRng",
                "SmallRng",
                "StdRng",
                "fast-rng",
                "fast_rng",
                "lazy_static",
                "LazyLock",
                "thread_local",
                "OnceLock",
            ):
                assert forbidden not in source, f"cached RNG state in {rel}: {forbidden}"

    def test_uuid_enables_no_rng_or_fast_rng_features(self) -> None:
        # H1's manifest half: the published crate's uuid must stay the
        # zero-feature builder set (no `rng`/`fast-rng` engine features —
        # the builders are feature-free; the crate's own-rng constructors
        # are what those features gate, and tors never calls them).
        # Parsed as TOML, not grepped as text: Cargo.toml prose names the
        # declined features in comments, so a substring scan would false-
        # positive on its own documentation. The dev-dependency's `v4`
        # (the bench's `new_v4()` comparator) unifies only into test/bench
        # builds — `cargo tree -e normal --no-default-features` shows the
        # published graph's uuid with no children; the lockfile's uuid ->
        # getrandom edge is that dev unification, not the library path.
        import pathlib
        import re

        cargo_text = (pathlib.Path(__file__).parent.parent / "Cargo.toml").read_text(
            encoding="utf-8"
        )
        # tomllib is 3.11+; on 3.10 fall back to tomli when present,
        # else a minimal section-scoped manual parse (keeps the guard
        # effective on all versions without a new dependency).
        try:
            import tomllib as _toml
        except ModuleNotFoundError:  # Python 3.10: no stdlib tomllib.
            try:
                import tomli as _toml  # type: ignore[no-redef]
            except ModuleNotFoundError:
                _toml = None  # type: ignore[assignment]

        if _toml is not None:
            manifest = _toml.loads(cargo_text)
            uuid_dep = manifest["dependencies"]["uuid"]
            assert uuid_dep.get("default-features") is False
            for feat in uuid_dep.get("features", []):
                assert feat not in ("rng", "fast-rng", "v4", "v7"), f"uuid feature: {feat}"
            rand_dep = manifest["dependencies"]["rand"]
            assert rand_dep.get("default-features") is False
            assert "thread-rng" not in rand_dep.get("features", [])
            assert "fast-rng" not in rand_dep.get("features", [])
            assert "small_rng" not in rand_dep.get("features", [])
            dev_uuid = manifest["dev-dependencies"]["uuid"]
            for feat in dev_uuid.get("features", []):
                assert feat not in ("fast-rng",), f"dev uuid feature: {feat}"
        else:

            def _dep_fields(section: str, name: str) -> tuple[bool | None, list[str]]:
                current: str | None = None
                for line in cargo_text.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("[") and stripped.endswith("]"):
                        current = stripped[1:-1].strip()
                        continue
                    if current != section:
                        continue
                    m = re.match(rf"{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", stripped)
                    if not m:
                        continue
                    body = m.group(1)
                    default_features: bool | None = None
                    dm = re.search(r"default-features\s*=\s*(true|false)", body)
                    if dm:
                        default_features = dm.group(1) == "true"
                    feats: list[str] = []
                    fm = re.search(r"features\s*=\s*\[([^\]]*)\]", body)
                    if fm:
                        feats = re.findall(r'"([^"]+)"', fm.group(1))
                    return default_features, feats
                raise AssertionError(f"dependency not found: [{section}] {name}")

            default_features, feats = _dep_fields("dependencies", "uuid")
            assert default_features is False
            for feat in feats:
                assert feat not in ("rng", "fast-rng", "v4", "v7"), f"uuid feature: {feat}"
            default_features, feats = _dep_fields("dependencies", "rand")
            assert default_features is False
            assert "thread-rng" not in feats
            assert "fast-rng" not in feats
            assert "small_rng" not in feats
            _, dev_feats = _dep_fields("dev-dependencies", "uuid")
            for feat in dev_feats:
                assert feat not in ("fast-rng",), f"dev uuid feature: {feat}"

    def test_os_failures_map_to_runtime_error_in_the_binding(self) -> None:
        # The Os-error half of the mapping contract, pinned as a
        # co-occurrence (not exact arm text) for the same
        # un-triggerability reason: a real getrandom failure is
        # effectively impossible post-boot, so no test can raise it on
        # demand — but the mapping arm must survive refactors. The pin
        # holds whenever a RandomError::Os match arm maps to
        # PyRuntimeError, whatever the arm's exact spelling or the
        # arms' order.
        import pathlib
        import re

        binding = pathlib.Path(__file__).parent.parent / "src" / "py" / "random.rs"
        source = binding.read_text(encoding="utf-8")
        assert re.search(
            r"RandomError::Os\(_\)[^#\n]*=>\s*PyRuntimeError", source
        ), "the RandomError::Os arm must map to PyRuntimeError"


class TestRfc8439Anchors:
    """MEDIUM-1: the oracle's ChaCha transcription anchored to RFC 8439
    itself, not just to rand_chacha's output (which the seeded goldens
    already pin — a transcription bug shared by oracle and extension
    would sail through those).

    What is pinned at which level, honestly: the quarter-round (§2.1.1)
    and quarter-round-on-state (§2.2.1) vectors run THROUGH the oracle's
    ``_quarter_round`` — the exact primitive the stream is built from.
    The full §2.3.2 block vector canNOT run through ``_chacha20_block``:
    RFC 8439 uses the IETF layout (32-bit counter word 12, 96-bit nonce
    words 13-15 — its test nonce sets word 14 to 0x4a000000), while
    rand_chacha 0.9, and therefore this family's stream, uses the
    original layout (64-bit counter words 12-13, 64-bit stream id words
    14-15, always zero under ``seed_from_u64``) — no (key, counter)
    argument to ``_chacha20_block`` can express word 14 != 0. Pinning
    §2.3.2's bytes here would fail, correctly. What anchors the block
    composition instead: ``seed_from_u64(0)``'s first 16 stream bytes
    as a committed literal (the same bytes the crate-side
    ``chacha20_seed_zero_first_block_is_pinned`` asserts through
    rand_chacha itself, and the pre-mask prefix of the uuid4(seed=0)
    golden) — the derivation-to-stream joint, pinned twice.
    """

    def test_quarter_round_matches_rfc_2_1_1(self) -> None:
        state = [0x11111111, 0x01020304, 0x9B8D6F43, 0x01234567]
        _quarter_round(state, 0, 1, 2, 3)
        assert state == [0xEA2A92F4, 0xCB1CF8CE, 0x4581472E, 0x5881C4BB]

    def test_state_quarter_round_matches_rfc_2_2_1(self) -> None:
        state = [
            0x879531E0,
            0xC5ECF37D,
            0x516461B1,
            0xC9A62F8A,
            0x44C20EF3,
            0x3390AF7F,
            0xD9FC690B,
            0x2A5F714C,
            0x53372767,
            0xB00A5631,
            0x974C541A,
            0x359E9963,
            0x5C971061,
            0x3D631689,
            0x2098D9D6,
            0x91DBD320,
        ]
        _quarter_round(state, 2, 7, 8, 13)
        assert state[2] == 0xBDB886DC
        assert state[7] == 0xCFACAFD2
        assert state[8] == 0xE46BEA80
        assert state[13] == 0xCCC07C79

    def test_seed_zero_first_stream_bytes_are_pinned(self) -> None:
        # seed_from_u64(0) -> block 0 -> first 16 bytes, committed: the
        # derivation-to-stream joint both implementations share.
        assert _oracle_bytes(16, 0).hex() == "b2f7f581d6de3c06a822fd6e7e8265fb"


class TestDuplicateAlphabetsAreWeighted:
    """MEDIUM-2: duplicate characters in the alphabet are weighted, not
    deduplicated — each position is an independent uniform draw over the
    alphabet's character POSITIONS, so ``"aaab"`` yields ``a`` with
    probability 3/4. Callers wanting uniform-over-distinct-characters
    must dedupe first (documented in the core, the binding, and
    docs/api.md)."""

    def test_aaab_is_three_to_one_seeded(self) -> None:
        # 256 draws, seed 0: exact pin (187 a's, 69 b's) — a dedupe
        # transcription would print ~128/128 and fail here.
        out = random_string(256, "aaab", seed=0)
        assert len(out) == 256
        assert out.count("a") == 187
        assert out.count("b") == 69
        assert set(out) == {"a", "b"}

    def test_the_ratio_holds_across_seeds(self) -> None:
        # The 3:1 shape, statistically: 4096 draws stay within ±10% of
        # the 3072/1024 expectation at every probed seed (stddev ~28,
        # so the band is ~±11 sigma — a uniform-over-distinct
        # regression lands at ~2048/2048 and fails by ~1000).
        for seed in (0, 1, 42):
            out = random_string(4096, "aaab", seed=seed)
            assert 2764 <= out.count("a") <= 3380
            assert len(out) - out.count("a") == out.count("b")


class TestSamplingIsOverScalarValues:
    """The scalar-values contract (api.md, the binding, and the core):
    sampling is over Unicode scalar values, NOT grapheme clusters — a
    combining mark in the alphabet is its own draw position and can
    land without its base. A cluster-aware (grapheme) sampler would
    never emit the bare mark; the engine does."""

    def test_a_combining_mark_lands_without_its_base(self) -> None:
        # "a" + U+0301 COMBINING ACUTE ACCENT: two scalar values. Under
        # seed 1 the mark is drawn alone (no base, exactly 1 char); under
        # seed 0 the base is. Both are legal outputs of the scalar-values
        # sampler and impossible shapes for a cluster-aware one.
        alphabet = "a\u0301"
        assert random_string(1, alphabet, seed=0) == "a"
        assert random_string(1, alphabet, seed=1) == "\u0301"

    def test_marks_stack_without_their_base(self) -> None:
        # Two draws both landing on the mark (seed 4): a bare double
        # stack — no cluster sampler can produce it from this alphabet.
        assert random_string(2, "a\u0301", seed=4) == "\u0301\u0301"


class TestUniformityAtScale:
    """HIGH-2's Python half: the seeded stream is uniform at a scale
    where eyeballing goldens proves nothing — and the pins fail under
    a ``x % n`` transcription (whose digests differ in full: the
    high-bits Lemire map and the low-bits modulo map are different
    functions of the same words, diverging from draw 0).

    Calibration, measured: n=62, 300k draws, seed 0 — chi-square ~51.9
    over 61 df (99.9% critical value ~109; bound set at 110), bucket
    extremes 4706/4940 around the 4838.7 mean (window set at ±500);
    n=3, 300k draws, seed 7 — chi-square ~3.6 over 2 df (bound 15),
    counts (100266, 100221, 99513). The bounds prove uniformity; the
    EXACT digests prove the mapping (a ``%`` transcription is uniform
    too, so bounds alone cannot catch it — the digests can)."""

    def test_chi_square_bounds_at_300k_draws(self) -> None:
        out = random_string(300_000, BASE62_CHARS, seed=0)
        mean = 300_000 / 62
        counts = [out.count(ch) for ch in BASE62_CHARS]
        chi2 = sum((c - mean) ** 2 / mean for c in counts)
        assert chi2 < 110.0
        assert all(abs(c - mean) < 500 for c in counts)

    def test_exact_digests_fail_under_modulo(self) -> None:
        out62 = random_string(300_000, BASE62_CHARS, seed=0)
        assert (
            hashlib.sha256(out62.encode("ascii")).hexdigest()
            == "65c258660e5ac3b11e3b873075436e7c3a4ac6b70af8ee49742dd56f321a17ed"
        )
        out3 = random_string(300_000, "abc", seed=7)
        assert (
            hashlib.sha256(out3.encode("ascii")).hexdigest()
            == "0155a34eebe428eed943620ded5bb16cb841a9b50b2ab909fae03d86ae4a0687"
        )
        # The modulo-mapped stream over the SAME words digests
        # differently — this is the assertion a `%` transcription
        # fails. One negative control pinned as a literal (not just
        # comment-recorded or inequality-asserted): the n=62 modulo digest
        # below is what the naive `word % n` map over the same oracle words
        # produces, so a regression to `%` fails BOTH the Lemire digest
        # above and this pin (they cannot coincide — the maps diverge at
        # draw 0).
        assert (
            hashlib.sha256(_modulo_string(300_000, BASE62_CHARS, 0).encode("ascii")).hexdigest()
            == "7d80549778e25f58604a6858351a84404c7779c18e2c68d5afbbf8664019684b"
        )
        assert _modulo_string(300_000, BASE62_CHARS, 0) != out62
        assert _modulo_string(300_000, "abc", 7) != out3


def _modulo_string(length: int, alphabet: str, seed: int) -> str:
    """The naive transcription under test: low-bits ``word % n`` over
    the oracle's own word stream — uniform in distribution, a different
    function of the words, and therefore a different (wrong) pinned
    output. Exists only for ``TestUniformityAtScale``'s divergence
    pins."""
    chars = list(alphabet)
    words = _oracle_words(seed)
    return "".join(chars[next(words) % len(chars)] for _ in range(length))
