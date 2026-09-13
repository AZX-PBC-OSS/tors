"""Contract gate for the random-generation family: ``random_string``,
``random_hex``, ``random_b62``, ``random_b64url``, ``uuid4``, ``uuid7``.

The security contract under test, stated once here because every surface
repeats it: unseeded calls draw fresh OS-entropy bytes on every call (no
process or thread RNG state, so fork-safe, matching ``secrets``' semantics)
and are safe for keys/tokens/secrets; ``seed=`` switches to a deterministic
ChaCha20 stream that is a pure function of (seed, arguments) and therefore
fully predictable — a test/fixture tool, never safe for secrets.

Determinism is pinned by an independent oracle, not by frozen literals alone:
``_oracle_hex``/``_oracle_string``/``_oracle_b64url``/``_oracle_uuid4``
re-implement the DOCUMENTED construction in pure Python — rand_core's
``seed_from_u64`` derivation (PCG32; see its comment for the spec-vs-source
finding), the RFC 8439 ChaCha20 block stream, and Lemire's nearly-divisionless
unbiased sampling — so a seeded pin asserts the extension equals the
construction the docs promise, never merely "whatever the implementation
emitted". Committed literal pins (added with the implementation, computed
from it and cross-checked against the oracle) then guard cross-version
stability independently; the b64 family's RFC 4648 vectors are the precedent
for that double pinning.

One subtlety the invariants class pins honestly: ``random_hex(n, seed=s)``
and ``random_string(2n, "0123456789abcdef", seed=s)`` draw from the SAME
distribution but NOT the same stream — byte-fill consumes n bytes while
char-sampling consumes at least 2n u64 draws (>= 16n bytes) — so the same
seed gives different outputs, and the distributional equivalence is pinned
statistically (seeded bucket counts), never as literal equality.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import struct
import time
import uuid as stdlib_uuid
from collections.abc import Iterator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import random_b62, random_b64url, random_hex, random_string, uuid4, uuid7

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
# it is an independent second implementation of the spec the docs state.

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
    order — exactly what one ``fill_bytes`` over an n-byte buffer consumes."""
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


def _oracle_hex(n: int, seed: int) -> str:
    return _oracle_bytes(n, seed).hex()


def _oracle_b64url(n: int, seed: int, *, padded: bool) -> str:
    encoded = base64.urlsafe_b64encode(_oracle_bytes(n, seed)).decode("ascii")
    return encoded if padded else encoded.rstrip("=")


def _oracle_uuid4(seed: int) -> str:
    raw = bytearray(_oracle_bytes(16, seed))
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return str(stdlib_uuid.UUID(bytes=bytes(raw)))


# --- Shared shape helpers -----------------------------------------------------


def _assert_uuid_shape(value: str, version: str) -> None:
    assert len(value) == 36
    assert value[8] == value[13] == value[18] == value[23] == "-"
    assert value[14] == version
    assert value[19] in "89ab"
    assert all(c in HEX_CHARS for c in value.replace("-", ""))


class TestSeededConstructionMatchesTheOracle:
    """Every seeded function equals the documented construction at the pin
    parameters: seeds 0, 1 (the edge goldens), and 42, across sizes that
    exercise each engine's paths (empty, single block, multi-block, every
    b64url residue class mod 3, multibyte and single-char alphabets)."""

    @pytest.mark.parametrize("seed", [0, 1, 42], ids=["seed0", "seed1", "seed42"])
    @pytest.mark.parametrize("n_bytes", [0, 1, 7, 64, 65, 512])
    def test_hex(self, n_bytes: int, seed: int) -> None:
        assert random_hex(n_bytes, seed=seed) == _oracle_hex(n_bytes, seed)

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
        assert random_string(length, alphabet, seed=seed) == _oracle_string(
            length, alphabet, seed
        )

    @pytest.mark.parametrize("seed", [0, 1, 42], ids=["seed0", "seed1", "seed42"])
    @pytest.mark.parametrize(
        "length", [0, 1, 16, 32, 128], ids=["0", "1", "16", "32", "128"]
    )
    def test_b62_delegates_to_the_string_engine(self, length: int, seed: int) -> None:
        # ONE engine: b62 is the random_string core over the base62 alphabet,
        # so it must equal both the direct spelling and the oracle.
        assert random_b62(length, seed=seed) == random_string(length, BASE62_CHARS, seed=seed)
        assert random_b62(length, seed=seed) == _oracle_string(length, BASE62_CHARS, seed)

    @pytest.mark.parametrize("seed", [0, 1, 42], ids=["seed0", "seed1", "seed42"])
    @pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
    @pytest.mark.parametrize("n_bytes", [0, 1, 2, 3, 4, 6, 64, 65, 512])
    def test_b64url(self, n_bytes: int, padded: bool, seed: int) -> None:
        # 0, 1, and 2 mod 3 all present, plus a 64-byte block boundary.
        assert random_b64url(n_bytes, padded=padded, seed=seed) == _oracle_b64url(
            n_bytes, seed, padded=padded
        )

    @pytest.mark.parametrize("seed", [0, 1, 42], ids=["seed0", "seed1", "seed42"])
    def test_uuid4(self, seed: int) -> None:
        assert uuid4(seed=seed) == _oracle_uuid4(seed)


class TestSeededGoldenLiterals:
    """The computed outputs at chosen parameters, committed as exact
    literals and sha256 digests — cross-version regression pins that stand
    independently of the oracle (every literal below was ALSO verified
    against the oracle-equality tests above before being committed; a
    ChaCha/rand_core semantic change breaks these loudly, which is the
    point: it is a finding, not a flake). The internal cross-references are
    themselves pins: uuid4(seed=0)'s first two groups are hex(8, seed=0)'s
    first 16 digits with the version nibble set, and b64url(3, seed=0)
    decodes to hex(8, seed=0)'s first 3 bytes — one stream, three
    consumers."""

    def test_hex_edge_goldens(self) -> None:
        assert random_hex(0, seed=0) == ""
        assert random_hex(0, seed=42) == ""
        assert random_hex(8, seed=0) == "b2f7f581d6de3c06"
        assert random_hex(8, seed=1) == "9a3744504560639e"
        assert random_hex(16, seed=42) == "7848b5d711bc9883996317a3f9c90269"
        assert (
            random_hex(64, seed=7)
            == "19454a27b752f905909507d6160ddc888e2df8b773098ef3f7bcd321a7caa748"
            "3a9afa8c98415d2fde7ae061aed1ef6821fb9ab3e89e9c7d07e32aa9c034fcd2"
        )

    def test_hex_1kib_golden_via_digest(self) -> None:
        # 1 KiB of hex output (512 bytes drawn): pinned by digest, the b64
        # family's large-vector idiom.
        digest = hashlib.sha256(random_hex(512, seed=7).encode("ascii")).hexdigest()
        assert digest == "9f42910e2d2c1b816c8f95a58d0606d3b4bcca0f88bd5e2308840901db075fcf"

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
        assert random_b64url(3, seed=0) == "svf1"
        assert random_b64url(1, padded=True, seed=42) == "eA=="
        assert random_b64url(2, padded=True, seed=42) == "eEg="
        assert random_b64url(7, seed=9) == "G92VZnZGBA"
        digest = hashlib.sha256(random_b64url(65536, seed=5).encode("ascii")).hexdigest()
        assert digest == "71881e3b69226086d5e408c049465e4a7c9c842a9a07ca4da68420007476ec2a"

    def test_uuid4_goldens(self) -> None:
        assert uuid4(seed=0) == "b2f7f581-d6de-4c06-a822-fd6e7e8265fb"
        assert uuid4(seed=1) == "9a374450-4560-439e-8670-b7a17d492b27"
        assert uuid4(seed=42) == "7848b5d7-11bc-4883-9963-17a3f9c90269"

    def test_the_goldens_cross_reference_one_stream(self) -> None:
        # The byte-fill consumers read the same stream: uuid4(seed=0)'s
        # first 16 hex digits (its first three dash-free groups,
        # u[:8]+u[9:13]+u[14:18]) are hex(8, seed=0)'s 16 digits with digit
        # 12 replaced by the version nibble (3 -> 4), and b64url(3, seed=0)
        # is the urlsafe encoding of hex(8, seed=0)'s first three bytes.
        hex8 = random_hex(8, seed=0)
        v4 = uuid4(seed=0)
        assert v4[:8] == hex8[:8]
        assert v4[9:13] == hex8[8:12]
        assert v4[14] == "4"
        assert v4[15:18] == hex8[13:16]
        urlsafe = random_b64url(3, seed=0)
        pad = "=" * ((4 - len(urlsafe) % 4) % 4)
        assert base64.urlsafe_b64decode(urlsafe + pad) == bytes.fromhex(hex8[:6])


class TestSeedDomain:
    """The seed argument contract: any Python int (reduced mod 2^64, so
    huge and negative seeds are legal and defined), bools ride along as the
    ints they are, and anything else is a TypeError naming the parameter."""

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
        with pytest.raises(TypeError, match="seed must be an int or None"):
            random_hex(8, seed=not_an_int)  # type: ignore[arg-type]


class TestValueErrors:
    """The repo's size-argument taxonomy: negative counts are ValueError
    naming the parameter and the accepted form (the chunkers' style), zero is
    legal and returns the empty string (secrets.token_hex(0)'s own shape),
    and there is no size cap by design."""

    @pytest.mark.parametrize("length", [-1, -1000], ids=["-1", "-1000"])
    def test_negative_length_raises_value_error_naming_it(self, length: int) -> None:
        with pytest.raises(ValueError, match=f"length must be >= 0, got {length}"):
            random_string(length, "ab")
        with pytest.raises(ValueError, match=f"length must be >= 0, got {length}"):
            random_b62(length)

    @pytest.mark.parametrize("n_bytes", [-1, -1000], ids=["-1", "-1000"])
    def test_negative_n_bytes_raises_value_error_naming_it(self, n_bytes: int) -> None:
        with pytest.raises(ValueError, match=f"n_bytes must be >= 0, got {n_bytes}"):
            random_hex(n_bytes)
        with pytest.raises(ValueError, match=f"n_bytes must be >= 0, got {n_bytes}"):
            random_b64url(n_bytes)

    def test_empty_alphabet_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="alphabet must be a non-empty str"):
            random_string(8, "")

    @pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
    def test_zero_is_the_empty_string_everywhere(self, padded: bool) -> None:
        # secrets.token_hex(0) == "" is the parity anchor for this shape.
        assert secrets.token_hex(0) == ""
        assert random_hex(0) == ""
        assert random_b62(0) == ""
        assert random_string(0, "ab") == ""
        assert random_b64url(0, padded=padded) == ""

    @pytest.mark.parametrize("not_an_int", ["8", 3.5, b"8", None, [8]], ids=str)
    def test_non_int_length_and_n_bytes_raise_type_error(self, not_an_int: object) -> None:
        with pytest.raises(TypeError):
            random_string(not_an_int, "ab")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            random_b62(not_an_int)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            random_hex(not_an_int)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            random_b64url(not_an_int)  # type: ignore[arg-type]

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

    def test_huge_outputs_complete_no_cap_by_design(self) -> None:
        # 1 MiB of hex output from one call: no size cap exists (memory is
        # the only bound), and the seeded spelling is digest-stable.
        out = random_hex(512 * 1024)
        assert len(out) == 1024 * 1024
        assert set(out) <= set(HEX_CHARS)

        def seeded_digest() -> str:
            return hashlib.sha256(random_hex(512 * 1024, seed=3).encode()).hexdigest()

        assert seeded_digest() == seeded_digest()


class TestUnseededOutputShape:
    """The unseeded contract: charset membership, exact lengths (the padding
    math pinned at every residue mod 3), and the canonical UUID field layout —
    properties of the output distribution, never pinned values (OS entropy is
    the source)."""

    @pytest.mark.parametrize("n_bytes", [1, 2, 3, 16, 32, 64, 1000])
    def test_hex_is_2n_lowercase_hex(self, n_bytes: int) -> None:
        out = random_hex(n_bytes)
        assert len(out) == 2 * n_bytes
        assert set(out) <= set(HEX_CHARS)

    @pytest.mark.parametrize("length", [1, 2, 16, 32, 64, 1000])
    def test_b62_is_exact_length_over_the_base62_alphabet(self, length: int) -> None:
        out = random_b62(length)
        assert len(out) == length
        assert set(out) <= set(BASE62_CHARS)

    @pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
    @pytest.mark.parametrize(
        "n_bytes",
        [1, 2, 3, 4, 5, 6, 7, 64, 3000, 3001, 3002],
        ids=["n=1", "n=2", "n=3", "n=4", "n=5", "n=6", "n=7", "n=64", "3000", "3001", "3002"],
    )
    def test_b64url_lengths_and_padding_math(self, n_bytes: int, padded: bool) -> None:
        out = random_b64url(n_bytes, padded=padded)
        assert set(out) <= set(B64URL_CHARS) | {"="}
        body = out.rstrip("=")
        assert set(body) <= set(B64URL_CHARS)
        pad = len(out) - len(body)
        if padded:
            # RFC 4648 §5: '=' tail only, count = (3 - n mod 3) mod 3, and
            # the total is 4 * ceil(n/3).
            assert pad == (3 - n_bytes % 3) % 3
            assert len(out) == 4 * ((n_bytes + 2) // 3)
        else:
            assert pad == 0
            # ceil(4n/3) exactly.
            assert len(out) == (4 * n_bytes + 2) // 3

    def test_token_hex_format_parity_with_secrets(self) -> None:
        # Format parity (length + charset class), never value parity: the two
        # spellings draw independently from the same OS source.
        for n in (1, 16, 32, 64):
            assert len(random_hex(n)) == len(secrets.token_hex(n)) == 2 * n
            assert set(random_hex(n)) <= set(HEX_CHARS)
            assert set(secrets.token_hex(n)) <= set(HEX_CHARS)

    def test_token_urlsafe_format_parity_with_secrets(self) -> None:
        # secrets.token_urlsafe(n) is urlsafe-b64 of n bytes, unpadded: the
        # exact format random_b64url(n) produces (padded=False default).
        for n in (1, 16, 32, 64, 3001):
            tors_out = random_b64url(n)
            secrets_out = secrets.token_urlsafe(n)
            assert len(tors_out) == len(secrets_out) == (4 * n + 2) // 3
            assert set(tors_out) <= set(B64URL_CHARS)
            assert set(secrets_out) <= set(B64URL_CHARS)

    def test_uuid4_shape_on_unseeded_draws(self) -> None:
        for _ in range(64):
            _assert_uuid_shape(uuid4(), "4")

    def test_uuid7_shape(self) -> None:
        for _ in range(64):
            _assert_uuid_shape(uuid7(), "7")

    def test_ten_thousand_uuid4s_are_distinct(self) -> None:
        # Birthday arithmetic at 122 random bits: a collision among 10k draws
        # has probability ~1e-23; any repeat is a broken engine, not bad luck.
        assert len({uuid4() for _ in range(10_000)}) == 10_000

    def test_uuid7_timestamp_is_the_callers_now_within_60s(self) -> None:
        # The caller-visible uuid7 contract: the 48-bit timestamp field (the
        # first 12 hex digits, which in the canonical string are the two
        # dash-free groups u[:8] + u[9:13]) decodes to the Unix-epoch
        # millisecond timestamp of the call. NOT a monotonic counter —
        # same-millisecond calls differ only in the random tail, and clock
        # skew backwards flows straight through (uuid_utils' strict
        # monotonicity is a different product promise; see docs/api.md).
        for _ in range(8):
            before = time.time() * 1000
            value = uuid7()
            after = time.time() * 1000
            ts = int(value[:8] + value[9:13], 16)
            assert before - 60_000 <= ts <= after + 60_000

    def test_uuid7_takes_no_seed_and_rejects_one(self) -> None:
        # The timestamp is external state: a seeded uuid7 would still vary
        # with time, so the parameter does not exist (documented in the
        # docstring and docs/api.md).
        with pytest.raises(TypeError):
            uuid7(seed=1)  # type: ignore[call-arg]


class TestStreamsDifferButDistributionsMatch:
    """The engine subtlety, pinned honestly: the byte-fill engines (hex,
    b64url, uuid4) consume exactly n stream bytes, while char-sampling
    (random_string/b62) consumes u64 words — the same seed therefore gives
    DIFFERENT outputs for random_hex(n, s) vs random_string(2n, hex, s)
    (literal equality would indicate a wiring bug), and the equivalence of
    the two OUTPUT distributions is what the bucket pins prove, statistically.

    Statistics cannot prove the no-modulo-bias property itself (u64 modulo
    bias over a <=64-char alphabet is ~2^-58, invisible at any sample size);
    that property is pinned structurally, by the oracle equality above (the
    oracle implements Lemire's rejection, so any modulo shortcut breaks it).
    """

    def test_same_seed_hex_vs_string_are_not_equal(self) -> None:
        assert random_hex(16, seed=42) != random_string(32, HEX_CHARS, seed=42)

    def test_bucket_counts_match_across_engines(self) -> None:
        # 20 seeds x 4096 bytes: 163,840 hex chars per engine, per-char
        # expectation 10,240. Both engines' counts must sit within 4 sigma
        # (sigma ~= 98) of expectation — deterministic given the seeds, so
        # the tolerance is robustness against future crate-version drift,
        # not a flake allowance.
        n, n_seeds = 4096, 20
        total = n_seeds * n * 2
        expected = total / 16
        sigma = (total * (1 / 16) * (15 / 16)) ** 0.5
        tolerance = 4 * sigma
        hex_counts = {
            c: "".join(random_hex(n, seed=s) for s in range(n_seeds)).count(c) for c in HEX_CHARS
        }
        string_counts = {
            c: "".join(random_string(2 * n, HEX_CHARS, seed=s) for s in range(n_seeds)).count(c)
            for c in HEX_CHARS
        }
        for engine, counts in (("hex", hex_counts), ("string", string_counts)):
            assert len(counts) == 16, f"{engine}: a hex char went missing"
            for char, count in counts.items():
                assert abs(count - expected) <= tolerance, (
                    f"{engine} char {char!r}: {count} vs {expected} (tolerance {tolerance})"
                )


class TestHypothesisProperties:
    """The contract over arbitrary shapes: seeded determinism at any
    length/seed (draw-verified twice per example), arbitrary alphabets
    (multibyte included), stdlib-parity b64url over the oracle bytes, and
    unseeded UUID shape."""

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
        n_bytes=st.integers(min_value=0, max_value=96),
        seed=st.integers(min_value=0, max_value=2**64 - 1),
        padded=st.booleans(),
    )
    @settings(max_examples=150)
    def test_b64url_equals_the_stdlib_expression_over_oracle_bytes(
        self, n_bytes: int, seed: int, padded: bool
    ) -> None:
        # The stdlib expression itself (base64.urlsafe_b64encode + strip) over
        # the oracle's bytes: the encoder is pinned against the stdlib, the
        # byte source against the construction.
        assert random_b64url(n_bytes, padded=padded, seed=seed) == _oracle_b64url(
            n_bytes, seed, padded=padded
        )

    @given(n_bytes=st.integers(min_value=0, max_value=256))
    @settings(max_examples=100)
    def test_unseeded_b64url_round_trips_through_the_stdlib_decoder(self, n_bytes: int) -> None:
        out = random_b64url(n_bytes)
        # unpadded needs its '=' tail restored for urlsafe_b64decode.
        decoded = base64.urlsafe_b64decode(out + "=" * ((3 - n_bytes % 3) % 3))
        assert len(decoded) == n_bytes

    @given(seed=st.integers(min_value=0, max_value=2**64 - 1))
    @settings(max_examples=75)
    def test_seeded_hex_and_uuid4_match_the_oracle(self, seed: int) -> None:
        assert random_hex(96, seed=seed) == _oracle_hex(96, seed)
        assert uuid4(seed=seed) == _oracle_uuid4(seed)


class TestDocstringSecurityContract:
    """The security contract must be LOUD on every seeded surface: the
    docstring pins force the warning to survive refactors (the
    documents-surface TestDocstringHonesty discipline)."""

    @pytest.mark.parametrize(
        "func",
        [random_string, random_hex, random_b62, random_b64url, uuid4],
        ids=["random_string", "random_hex", "random_b62", "random_b64url", "uuid4"],
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
