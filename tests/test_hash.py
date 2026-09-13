"""Contract gate for tors's one-shot hashing surface: ``md5_hex``,
``sha1_hex``, ``sha256_hex``, ``sha512_hex``, and ``hmac_sha256_hex``, and
their raw-digest twins ``md5_digest``, ``sha1_digest``, ``sha256_digest``,
``sha512_digest``, and ``hmac_sha256_digest``.

The five algorithms are the request-signing / content-check primitives a
text-operations library keeps getting asked for: webhook signature
verification (HMAC-SHA-256 is the GitHub/Slack/Stripe convention), S3/HTTP
ETag and Content-MD5 checks (md5), rsync-style quick content compares and
legacy-interop digests (sha1), dedup-cache keys and general content
addressing (sha256/sha512, the ``finalize`` family's hash without the
normalize stage). Each algorithm comes in two output spellings over ONE
digest computation: lowercase hex (the ``_hex`` names) and the raw digest
bytes (the ``_digest`` names — the call sites that want the bytes
themselves: webhook schemes that base64-encode the signature, key
derivation chains that feed a digest back in as a key, digest-sliced
advisory-lock ints, content thumbprints). Every one is a stateless
one-shot: bytes (or a str, taken as its UTF-8 bytes) in, the whole digest
computation under one ``py.detach``.

Parity is the product and is pinned three ways:

- exact differential against ``hashlib``/``hmac`` over hypothesis corpora
  (arbitrary bytes including NUL and high bytes; arbitrary Unicode text
  including multibyte, emoji, and combining marks; both str and bytes
  spellings; ~39,000 generated cases across the cells below, both output
  spellings of every algorithm) — the stdlib is the always-available
  oracle, and RustCrypto (tors's engines) is an independent implementation
  from CPython's OpenSSL backend, so agreement is two-implementation
  agreement, not self-consistency;
- known-answer vectors transcribed verbatim from the primary sources:
  RFC 1321 (md5), FIPS 180-4 / RFC 3174 (sha1), FIPS 180-4 (sha256,
  sha512, including the two-block and million-'a' examples), and RFC 4231
  (every HMAC-SHA-256 case: 1, 2, 3, 4, 6, and 7, the 131-byte-key cases
  included) — a wrong engine wiring cannot hide behind parity with the
  local stdlib alone;
- the block-boundary battery: every length 0..131 (both digest-block
  sizes plus every padding-edge shape: 55/56 for the 64-byte-block
  family, 111/112 and 119/120 for sha512's 128-byte blocks) and a 12 MiB
  single-shot, each differentially pinned.

Two crate-wide contracts carry over verbatim and are pinned here: a
``str`` argument is its UTF-8 bytes (``tors.sha256_hex(s) ==
hashlib.sha256(s.encode("utf-8")).hexdigest()``, the convenience
``hashlib`` deliberately refuses — it raises TypeError on str — provided
deliberately and documented loudly), and a lone surrogate therefore
raises UnicodeEncodeError at the argument boundary (pyo3's UTF-8 borrow,
the same contract ``normalize``/``finalize`` pin). The bytes side is the
exactly-``bytes`` doctrine of the bytes-in family
(tests/test_b64.py::TestBytesOnlyArgumentContract): ``bytearray`` and
``memoryview`` raise TypeError, because the GIL-released read wants an
immutable buffer.

SECURITY: md5 and sha1, in either spelling (``*_hex`` and ``*_digest``),
are checksum/legacy-interop primitives only (Content-MD5, S3 ETags,
cache-busting, rsync-style quick compares). Both are broken for security
purposes and have been since the 2000s (md5 collisions since 2004, sha1's
first practical collision 2017): never use either for signatures,
certificates, or password handling. Every doc surface that names them
carries this note; it is repeated here because this file is the contract
of record.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_module

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reference import corpus_utf8  # noqa: I001 -- the shared oracle module (tests/reference.py)
from tors import (
    hmac_sha256_digest,
    hmac_sha256_hex,
    md5_digest,
    md5_hex,
    sha1_digest,
    sha1_hex,
    sha256_digest,
    sha256_hex,
    sha512_digest,
    sha512_hex,
)

_MIB = 1024 * 1024

# The stdlib oracle each spelling must equal, keyed by the tors spelling:
# the hex spellings compare against ``.hexdigest()``, the digest spellings
# against ``.digest()`` — one table, so every battery below (the hypothesis
# lanes, the block edges, the 0..131 sweep, the 12 MiB shot, the surrogate
# and argument contracts) runs over both output spellings of the same five
# algorithms.
_STDLIB_DIGEST = {
    md5_hex: lambda raw: hashlib.md5(raw).hexdigest(),
    sha1_hex: lambda raw: hashlib.sha1(raw).hexdigest(),
    sha256_hex: lambda raw: hashlib.sha256(raw).hexdigest(),
    sha512_hex: lambda raw: hashlib.sha512(raw).hexdigest(),
    md5_digest: lambda raw: hashlib.md5(raw).digest(),
    sha1_digest: lambda raw: hashlib.sha1(raw).digest(),
    sha256_digest: lambda raw: hashlib.sha256(raw).digest(),
    sha512_digest: lambda raw: hashlib.sha512(raw).digest(),
}
_HEX_LENGTH = {md5_hex: 32, sha1_hex: 40, sha256_hex: 64, sha512_hex: 128}
# The raw-digest spellings' exact byte lengths (hmac's digest is
# sha256-sized, pinned in its own test below).
_DIGEST_LENGTH = {md5_digest: 16, sha1_digest: 20, sha256_digest: 32, sha512_digest: 64}
# The hex spelling of each algorithm over its digest spelling: the DRY
# shape the Rust cores implement (ONE digest computation, two outputs),
# pinned from the Python side in the battery below.
_HEX_OF_DIGEST = {
    md5_hex: md5_digest,
    sha1_hex: sha1_digest,
    sha256_hex: sha256_digest,
    sha512_hex: sha512_digest,
}


def _stdlib_hmac(key: bytes, data: bytes) -> str:
    return hmac_module.new(key, data, hashlib.sha256).hexdigest()


def _stdlib_hmac_digest(key: bytes, data: bytes) -> bytes:
    return hmac_module.new(key, data, hashlib.sha256).digest()


# The two HMAC spellings over their stdlib oracles: the request-signing
# primitive in both output shapes, hex and raw bytes alike.
_HMAC_SPELLINGS = {
    hmac_sha256_hex: _stdlib_hmac,
    hmac_sha256_digest: _stdlib_hmac_digest,
}


class TestKnownAnswerVectors:
    """The primary-source vectors, pinned verbatim: a locally consistent
    wrong digest (tors and the local hashlib agreeing because both were
    built wrong) cannot pass these the way it can pass a pure differential."""

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (b"", "d41d8cd98f00b204e9800998ecf8427e"),
            (b"a", "0cc175b9c0f1b6a831c399e269772661"),
            (b"abc", "900150983cd24fb0d6963f7d28e17f72"),
            (b"message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
            (b"abcdefghijklmnopqrstuvwxyz", "c3fcd3d76192e4007dfb496cca67e13b"),
            (
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                "d174ab98d277d9f5a5611c2c9f419d9f",
            ),
            (
                b"123456789012345678901234567890123456789012345678901234567890"
                b"12345678901234567890",
                "57edf4a22be3c955ac49da2e2107b67a",
            ),
            (b"The quick brown fox jumps over the lazy dog", "9e107d9d372bb6826bd81d3542a419d6"),
        ],
        ids=[
            "empty",
            "a",
            "abc",
            "message-digest",
            "alpha",
            "alpha-digits",
            "digits-80",
            "fox",
        ],
    )
    def test_md5_rfc1321_vectors_verbatim(self, data: bytes, expected: str) -> None:
        assert md5_hex(data) == expected

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (b"", "da39a3ee5e6b4b0d3255bfef95601890afd80709"),
            (b"abc", "a9993e364706816aba3e25717850c26c9cd0d89d"),
            (b"a", "86f7e437faa5a7fce15d1ddcb9eaeaea377667b8"),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "84983e441c3bd26ebaae4aa1f95129e5e54670f1",
            ),
            (
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
                b"ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
                "a49b2446a02c645bf419f995b67091253a04a259",
            ),
            (
                b"The quick brown fox jumps over the lazy dog",
                "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12",
            ),
        ],
        ids=["empty", "abc", "a", "448-bit", "896-bit", "fox"],
    )
    def test_sha1_fips180_vectors_verbatim(self, data: bytes, expected: str) -> None:
        assert sha1_hex(data) == expected

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            (b"a", "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
            (
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
                b"ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
                "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
            ),
        ],
        ids=["empty", "abc", "a", "448-bit", "896-bit"],
    )
    def test_sha256_fips180_vectors_verbatim(self, data: bytes, expected: str) -> None:
        assert sha256_hex(data) == expected

    def test_sha256_million_a_fips_vector(self) -> None:
        # The FIPS 180-4 / CDT million-'a' message: the multi-block stress
        # vector the standard itself publishes.
        assert (
            sha256_hex(b"a" * 1_000_000)
            == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        )

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (
                b"",
                "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
                "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
            ),
            (
                b"abc",
                "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
                "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
            ),
            (
                b"a",
                "1f40fc92da241694750979ee6cf582f2d5d7d28e18335de05abc54d0560e0f53"
                "02860c652bf08d560252aa5e74210546f369fbbbce8c12cfc7957b2652fe9a75",
            ),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "204a8fc6dda82f0a0ced7beb8e08a41657c16ef468b228a8279be331a703c335"
                "96fd15c13b1b07f9aa1d3bea57789ca031ad85c7a71dd70354ec631238ca3445",
            ),
        ],
        ids=["empty", "abc", "a", "448-bit"],
    )
    def test_sha512_fips180_vectors_verbatim(self, data: bytes, expected: str) -> None:
        assert sha512_hex(data) == expected

    def test_sha512_896_bit_and_million_a_fips_vectors(self) -> None:
        assert (
            sha512_hex(
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
                b"ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
            )
            == "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018"
            "501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909"
        )
        assert (
            sha512_hex(b"a" * 1_000_000)
            == "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973eb"
            "de0ff244877ea60a4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b"
        )

    @pytest.mark.parametrize(
        ("key", "data", "expected"),
        [
            (
                b"\x0b" * 20,
                b"Hi There",
                "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
            ),
            (
                b"Jefe",
                b"what do ya want for nothing?",
                "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
            ),
            (
                b"\xaa" * 20,
                b"\xdd" * 50,
                "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe",
            ),
            (
                bytes(range(1, 26)),
                b"\xcd" * 50,
                "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b",
            ),
            (
                b"\xaa" * 131,
                b"Test Using Larger Than Block-Size Key - Hash Key First",
                "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54",
            ),
            (
                b"\xaa" * 131,
                b"This is a test using a larger than block-size key and a larger "
                b"than block-size data. The key needs to be hashed before being "
                b"used by the HMAC algorithm.",
                "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2",
            ),
        ],
        ids=[
            "rfc4231-case1",
            "rfc4231-case2",
            "rfc4231-case3",
            "rfc4231-case4",
            "rfc4231-case6",
            "rfc4231-case7",
        ],
    )
    def test_hmac_sha256_rfc4231_vectors_verbatim(
        self, key: bytes, data: bytes, expected: str
    ) -> None:
        # Every RFC 4231 HMAC-SHA-256 case (5 is the truncation case, not
        # applicable to a full-digest function): the short-key cases, the
        # key+data-over-64-byte cases, and both 131-byte-key cases.
        assert hmac_sha256_hex(key, data) == expected

    @pytest.mark.parametrize(
        ("tors_fn", "data", "expected"),
        [
            (md5_digest, b"", bytes.fromhex("d41d8cd98f00b204e9800998ecf8427e")),
            (md5_digest, b"abc", bytes.fromhex("900150983cd24fb0d6963f7d28e17f72")),
            (sha1_digest, b"", bytes.fromhex("da39a3ee5e6b4b0d3255bfef95601890afd80709")),
            (sha1_digest, b"abc", bytes.fromhex("a9993e364706816aba3e25717850c26c9cd0d89d")),
            (
                sha256_digest,
                b"",
                bytes.fromhex(
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                ),
            ),
            (
                sha256_digest,
                b"abc",
                bytes.fromhex(
                    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
                ),
            ),
            (
                sha512_digest,
                b"",
                bytes.fromhex(
                    "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
                    "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
                ),
            ),
            (
                sha512_digest,
                b"abc",
                bytes.fromhex(
                    "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
                    "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
                ),
            ),
        ],
        ids=[
            "md5-empty",
            "md5-abc",
            "sha1-empty",
            "sha1-abc",
            "sha256-empty",
            "sha256-abc",
            "sha512-empty",
            "sha512-abc",
        ],
    )
    def test_digest_primary_source_vectors_verbatim(
        self, tors_fn, data: bytes, expected: bytes
    ) -> None:
        # The same primary-source vectors (RFC 1321 / FIPS 180-4) through
        # the raw-digest spellings: the expected bytes are the vectors'
        # hex, decoded — the digest spelling is the same engine, pinned in
        # its own output shape.
        assert tors_fn(data) == expected

    def test_hmac_digest_rfc4231_case2_verbatim(self) -> None:
        assert hmac_sha256_digest(b"Jefe", b"what do ya want for nothing?") == bytes.fromhex(
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        )


class TestDifferentialParity:
    """The star: exact parity with ``hashlib``/``hmac`` over generated
    corpora. hashlib is OpenSSL-backed and RustCrypto is an independent
    implementation, so this is cross-implementation agreement on every
    generated input, the same differential that pins ``finalize`` and
    ``merkle_root``."""

    @pytest.mark.parametrize("tors_fn", list(_STDLIB_DIGEST), ids=lambda f: f.__name__)
    @given(st.binary(max_size=512))
    @settings(max_examples=1500)
    def test_bytes_parity_with_hashlib(self, tors_fn, raw: bytes) -> None:
        assert tors_fn(raw) == _STDLIB_DIGEST[tors_fn](raw)

    @pytest.mark.parametrize("tors_fn", list(_STDLIB_DIGEST), ids=lambda f: f.__name__)
    @given(st.text(max_size=400))
    @settings(max_examples=1500)
    def test_str_parity_with_hashlib_of_the_utf8_encoding(self, tors_fn, text: str) -> None:
        # The str-in convenience, differentially: hashing a str is hashing
        # its UTF-8 bytes, exactly what the caller would spell by hand.
        assert tors_fn(text) == _STDLIB_DIGEST[tors_fn](text.encode("utf-8"))

    @given(st.binary(max_size=512), st.text(max_size=400))
    @settings(max_examples=1500)
    def test_hex_spelling_is_the_digest_spelling_hex_encoded(
        self, raw: bytes, text: str
    ) -> None:
        # The one-digest-two-spellings invariant, differentially over both
        # input spellings: every _hex name is exactly its _digest twin,
        # hex-encoded — the DRY shape the Rust cores implement (the hex
        # path consumes the digest path), pinned from the Python side.
        for hex_fn, digest_fn in _HEX_OF_DIGEST.items():
            assert hex_fn(raw) == digest_fn(raw).hex()
            assert hex_fn(text) == digest_fn(text).hex()
        assert hmac_sha256_hex(b"k", raw) == hmac_sha256_digest(b"k", raw).hex()
        assert hmac_sha256_hex("k", text) == hmac_sha256_digest("k", text).hex()

    @pytest.mark.parametrize("spelling", list(_HMAC_SPELLINGS), ids=lambda f: f.__name__)
    @given(st.binary(max_size=256), st.binary(max_size=256))
    @settings(max_examples=1200)
    def test_hmac_bytes_parity_with_stdlib_hmac(self, spelling, key: bytes, data: bytes) -> None:
        assert spelling(key, data) == _HMAC_SPELLINGS[spelling](key, data)

    @pytest.mark.parametrize("spelling", list(_HMAC_SPELLINGS), ids=lambda f: f.__name__)
    @given(st.text(max_size=200), st.text(max_size=200))
    @settings(max_examples=1200)
    def test_hmac_str_parity_with_stdlib_hmac_of_the_utf8_encoding(
        self, spelling, key: str, data: str
    ) -> None:
        # Both str arguments are their UTF-8 bytes, the webhook-signature
        # spelling where key and payload are both text.
        assert spelling(key, data) == _HMAC_SPELLINGS[spelling](
            key.encode("utf-8"), data.encode("utf-8")
        )

    @pytest.mark.parametrize("spelling", list(_HMAC_SPELLINGS), ids=lambda f: f.__name__)
    @given(st.text(max_size=200), st.binary(max_size=256))
    @settings(max_examples=600)
    def test_hmac_mixed_str_key_bytes_data_parity(
        self, spelling, key: str, data: bytes
    ) -> None:
        assert spelling(key, data) == _HMAC_SPELLINGS[spelling](key.encode("utf-8"), data)

    @pytest.mark.parametrize("spelling", list(_HMAC_SPELLINGS), ids=lambda f: f.__name__)
    @given(st.binary(max_size=256), st.text(max_size=200))
    @settings(max_examples=600)
    def test_hmac_mixed_bytes_key_str_data_parity(
        self, spelling, key: bytes, data: str
    ) -> None:
        assert spelling(key, data) == _HMAC_SPELLINGS[spelling](key, data.encode("utf-8"))

    @pytest.mark.parametrize("tors_fn", list(_STDLIB_DIGEST), ids=lambda f: f.__name__)
    def test_every_single_byte_value(self, tors_fn) -> None:
        # The adversarial-byte sweep: every possible byte, NUL and high
        # bytes included, each pinned against hashlib.
        for value in range(256):
            raw = bytes([value])
            assert tors_fn(raw) == _STDLIB_DIGEST[tors_fn](raw), f"byte 0x{value:02x}"

    @pytest.mark.parametrize(
        "text",
        [
            "café",
            "e" + "́",
            "🦀🚀",
            "Zͧͦ̓҉͇͈̹̩͕͘g̴̢͓̘͔̲͓͉̲o̡̱̮̗̼ͥ͒̇",
            "田中さんにあげて下さい",
            "Quick brown fox: 🦊 jumps over 13 lazy dogs.",
            "\x00\x1f\x7f\x80\x9f",
            "a" * 1000 + "é" * 500,
        ],
        ids=[
            "multibyte",
            "combining",
            "emoji",
            "zalgo",
            "cjk",
            "mixed",
            "controls",
            "long-mixed",
        ],
    )
    @pytest.mark.parametrize("tors_fn", list(_STDLIB_DIGEST), ids=lambda f: f.__name__)
    def test_multibyte_str_parity(self, tors_fn, text: str) -> None:
        assert tors_fn(text) == _STDLIB_DIGEST[tors_fn](text.encode("utf-8"))

    def test_str_and_bytes_spellings_agree_over_the_corpus(self) -> None:
        # The equivalence the convenience rests on: hash(s) == hash(s.encode()),
        # on real corpus text (prose and the non-ASCII decomposed corpus).
        for kind in ("prose", "decomposed"):
            text = corpus_utf8(kind, 64 * 1024).decode("utf-8")
            for tors_fn in _STDLIB_DIGEST:
                assert tors_fn(text) == tors_fn(text.encode("utf-8"))


class TestBlockBoundaries:
    """Every padding-edge and block-boundary shape, differentially pinned:
    the lengths where a digest's buffering/padding logic changes blocks.
    55/56/63/64/65 for the 64-byte-block family (md5/sha1/sha256: a
    55-byte message plus 0x80 plus the 8-byte length exactly fills one
    block; 56 spills to two), 111/112/119/120/127/128/129 for sha512's
    128-byte blocks (16-byte length field)."""

    @pytest.mark.parametrize(
        "length",
        [0, 1, 54, 55, 56, 57, 63, 64, 65, 111, 112, 119, 120, 127, 128, 129],
        ids=[
            "0", "1", "54", "55", "56", "57", "63", "64", "65",
            "111", "112", "119", "120", "127", "128", "129",
        ],
    )
    @pytest.mark.parametrize("tors_fn", list(_STDLIB_DIGEST), ids=lambda f: f.__name__)
    def test_block_edge_lengths_match_hashlib(self, tors_fn, length: int) -> None:
        raw = bytes((i * 251 + 7) % 256 for i in range(length))
        assert tors_fn(raw) == _STDLIB_DIGEST[tors_fn](raw)

    @pytest.mark.parametrize("tors_fn", list(_STDLIB_DIGEST), ids=lambda f: f.__name__)
    def test_every_length_up_to_131_matches_hashlib(self, tors_fn) -> None:
        # The exhaustive sweep: both digest block sizes covered twice over,
        # every padding shape included, data that varies per length.
        for length in range(132):
            raw = bytes((i * 131 + 11) % 256 for i in range(length))
            assert tors_fn(raw) == _STDLIB_DIGEST[tors_fn](raw), f"length {length}"

    @pytest.mark.parametrize(
        ("key_len", "data_len"),
        [
            (0, 0),
            (0, 64),
            (20, 50),
            (64, 0),
            (64, 64),
            (65, 50),
            (131, 54),
            (131, 152),
        ],
        ids=[
            "empty-key-empty-data",
            "empty-key-one-block",
            "rfc-shape",
            "block-key-empty-data",
            "block-key-block-data",
            "over-block-key",
            "hashed-key-short-data",
            "hashed-key-multi-block-data",
        ],
    )
    @pytest.mark.parametrize("spelling", list(_HMAC_SPELLINGS), ids=lambda f: f.__name__)
    def test_hmac_block_edge_shapes_match_stdlib(
        self, spelling, key_len: int, data_len: int
    ) -> None:
        # The HMAC-specific edges: key exactly at/over the 64-byte inner
        # block size (where the key itself gets hashed), data at the digest
        # block edges, and the empty-key/empty-data corners.
        key = bytes((i * 7 + 1) % 256 for i in range(key_len))
        data = bytes((i * 251 + 3) % 256 for i in range(data_len))
        assert spelling(key, data) == _HMAC_SPELLINGS[spelling](key, data)

    @pytest.mark.parametrize("tors_fn", list(_STDLIB_DIGEST), ids=lambda f: f.__name__)
    def test_twelve_mib_single_shot_matches_hashlib(self, tors_fn) -> None:
        # The size where the GIL story is told (tests/test_gil_release.py):
        # one 12 MiB shot, parity preserved, no chunking under the hood.
        raw = corpus_utf8("prose", 12 * _MIB)
        assert tors_fn(raw) == _STDLIB_DIGEST[tors_fn](raw)

    @pytest.mark.parametrize("spelling", list(_HMAC_SPELLINGS), ids=lambda f: f.__name__)
    def test_twelve_mib_hmac_matches_stdlib(self, spelling) -> None:
        raw = corpus_utf8("prose", 12 * _MIB)
        assert spelling(b"corpus-key", raw) == _HMAC_SPELLINGS[spelling](b"corpus-key", raw)


class TestStrInContract:
    """The crate-wide str-in convention, pinned: a str is its UTF-8 bytes;
    a lone surrogate (which CPython can hold but UTF-8 cannot encode)
    raises UnicodeEncodeError at the argument boundary, pyo3's borrow
    refusing it before any Rust code runs (the same contract
    tests/test_finalize.py::TestSurrogateBehavior pins for normalize)."""

    @pytest.mark.parametrize("tors_fn", list(_STDLIB_DIGEST), ids=lambda f: f.__name__)
    def test_lone_surrogate_raises_unicode_encode_error(self, tors_fn) -> None:
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            tors_fn("x\ud800")

    @pytest.mark.parametrize("spelling", list(_HMAC_SPELLINGS), ids=lambda f: f.__name__)
    def test_hmac_lone_surrogate_in_key_raises_unicode_encode_error(self, spelling) -> None:
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            spelling("k\udfff", b"data")

    @pytest.mark.parametrize("spelling", list(_HMAC_SPELLINGS), ids=lambda f: f.__name__)
    def test_hmac_lone_surrogate_in_data_raises_unicode_encode_error(self, spelling) -> None:
        # The key is checked and borrowed first, so the surrogate in the
        # SECOND argument is what raises: pinned so the argument order of
        # the validation is a visible contract, not an accident.
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            spelling(b"key", "d\ud800ata")


class TestArgumentContract:
    """The exactly-``bytes`` doctrine of the bytes-in family
    (tests/test_b64.py::TestBytesOnlyArgumentContract): the union accepts
    exactly ``str`` or ``bytes`` — a ``bytearray``/``memoryview`` is a
    TypeError rather than a silent copy, because the GIL-released read
    wants an immutable buffer; everything else is a TypeError the same
    way."""

    @pytest.mark.parametrize(
        "not_str_or_bytes",
        [
            bytearray(b"abc"),
            memoryview(b"abc"),
            123,
            None,
            [b"abc"],
            1.5,
            True,
        ],
        ids=["bytearray", "memoryview", "int", "none", "list", "float", "bool"],
    )
    @pytest.mark.parametrize("tors_fn", list(_STDLIB_DIGEST), ids=lambda f: f.__name__)
    def test_digest_non_str_or_bytes_arguments_raise_type_error(
        self, tors_fn, not_str_or_bytes: object
    ) -> None:
        with pytest.raises(TypeError):
            tors_fn(not_str_or_bytes)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "not_str_or_bytes",
        [bytearray(b"k"), memoryview(b"k"), 123, None, [b"k"]],
        ids=["bytearray", "memoryview", "int", "none", "list"],
    )
    @pytest.mark.parametrize("spelling", list(_HMAC_SPELLINGS), ids=lambda f: f.__name__)
    def test_hmac_non_str_or_bytes_key_raises_type_error(
        self, spelling, not_str_or_bytes: object
    ) -> None:
        with pytest.raises(TypeError):
            spelling(not_str_or_bytes, b"data")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "not_str_or_bytes",
        [bytearray(b"d"), memoryview(b"d"), 123, None, [b"d"]],
        ids=["bytearray", "memoryview", "int", "none", "list"],
    )
    @pytest.mark.parametrize("spelling", list(_HMAC_SPELLINGS), ids=lambda f: f.__name__)
    def test_hmac_non_str_or_bytes_data_raises_type_error(
        self, spelling, not_str_or_bytes: object
    ) -> None:
        with pytest.raises(TypeError):
            spelling(b"key", not_str_or_bytes)  # type: ignore[arg-type]

    @pytest.mark.parametrize("spelling", list(_HMAC_SPELLINGS), ids=lambda f: f.__name__)
    def test_hmac_key_is_validated_before_data(self, spelling) -> None:
        # The validation order the wrapper's nested borrows guarantee (the
        # key is borrowed and validated first), pinned explicitly with BOTH
        # arguments bad: the key's error is the one that fires. The order
        # holds for both spellings — they share the same wrapper contract.
        with pytest.raises(TypeError, match="^key must be str or bytes"):
            spelling(bytearray(b"k"), 123)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="^key must be str or bytes"):
            spelling(None, memoryview(b"d"))  # type: ignore[arg-type]
        # A str key holding a lone surrogate (valid type, failed borrow)
        # still raises before the data's TypeError: order is type-check and
        # borrow of the key, then the data, exactly.
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            spelling("k\ud800", 123)  # type: ignore[arg-type]


class TestOutputInvariants:
    """Structural contracts that hold independent of the exact digest:
    exact hex length per algorithm, lowercase hex charset, empty-input
    stability (empty input is legal — no ValueError paths anywhere on
    this surface), and determinism across calls."""

    @pytest.mark.parametrize("tors_fn", list(_HEX_LENGTH), ids=lambda f: f.__name__)
    def test_hex_length_is_exact(self, tors_fn) -> None:
        for raw in (b"", b"x", b"a longer payload with multiple blocks" * 4):
            assert len(tors_fn(raw)) == _HEX_LENGTH[tors_fn]

    @pytest.mark.parametrize("tors_fn", list(_HEX_LENGTH), ids=lambda f: f.__name__)
    def test_output_is_lowercase_hex_charset(self, tors_fn) -> None:
        digest = tors_fn("The quick brown fox jumps over the lazy dog 🦊")
        assert digest == digest.lower()
        assert all(c in "0123456789abcdef" for c in digest)

    @pytest.mark.parametrize("tors_fn", list(_HEX_LENGTH), ids=lambda f: f.__name__)
    def test_empty_input_is_legal_and_stable(self, tors_fn) -> None:
        first = tors_fn(b"")
        assert first == tors_fn("")
        assert first == tors_fn(b"")
        assert first == _STDLIB_DIGEST[tors_fn](b"")

    @pytest.mark.parametrize("tors_fn", list(_HEX_LENGTH), ids=lambda f: f.__name__)
    def test_output_is_deterministic_across_calls(self, tors_fn) -> None:
        raw = b"determinism check payload"
        assert tors_fn(raw) == tors_fn(raw)

    def test_hmac_empty_key_is_legal_and_matches_stdlib(self) -> None:
        # Parity with stdlib hmac: an empty key is a legal key
        # (hmac.new(b"", ...) works); the digest is pinned differentially.
        assert hmac_sha256_hex(b"", b"data") == _stdlib_hmac(b"", b"data")
        assert hmac_sha256_hex("", "data") == _stdlib_hmac(b"", b"data")

    def test_hmac_output_is_lowercase_hex_of_length_64(self) -> None:
        digest = hmac_sha256_hex(b"key", b"data")
        assert len(digest) == 64
        assert digest == digest.lower()
        assert all(c in "0123456789abcdef" for c in digest)


class TestDigestOutputInvariants:
    """The raw-digest spellings' structural contracts, the bytes twins of
    ``TestOutputInvariants``' hex pins: exact byte length per algorithm
    (16/20/32/64, hmac's 32), a true ``bytes`` return, empty-input
    stability, and determinism."""

    @pytest.mark.parametrize("tors_fn", list(_DIGEST_LENGTH), ids=lambda f: f.__name__)
    def test_digest_is_exact_length_bytes(self, tors_fn) -> None:
        for raw in (b"", b"x", b"a longer payload with multiple blocks" * 4):
            digest = tors_fn(raw)
            assert type(digest) is bytes
            assert len(digest) == _DIGEST_LENGTH[tors_fn]

    @pytest.mark.parametrize("tors_fn", list(_DIGEST_LENGTH), ids=lambda f: f.__name__)
    def test_digest_empty_input_is_legal_and_stable(self, tors_fn) -> None:
        # Empty input is legal on the digest spellings too — no ValueError
        # paths anywhere on this surface — and the raw bytes match the
        # stdlib oracle, str and bytes inputs agreeing.
        first = tors_fn(b"")
        assert first == tors_fn("")
        assert first == tors_fn(b"")
        assert first == _STDLIB_DIGEST[tors_fn](b"")

    @pytest.mark.parametrize("tors_fn", list(_DIGEST_LENGTH), ids=lambda f: f.__name__)
    def test_digest_is_deterministic_across_calls(self, tors_fn) -> None:
        raw = b"determinism check payload"
        assert tors_fn(raw) == tors_fn(raw)

    def test_hmac_digest_is_32_bytes_and_matches_stdlib(self) -> None:
        digest = hmac_sha256_digest(b"key", b"data")
        assert type(digest) is bytes
        assert len(digest) == 32
        assert digest == _stdlib_hmac_digest(b"key", b"data")

    def test_hmac_digest_empty_key_is_legal_and_matches_stdlib(self) -> None:
        # The empty-key parity the hex spelling pins, in the raw shape.
        assert hmac_sha256_digest(b"", b"data") == _stdlib_hmac_digest(b"", b"data")
        assert hmac_sha256_digest("", "data") == _stdlib_hmac_digest(b"", b"data")


class TestConsumerShapes:
    """The raw-digest call sites the ``_digest`` spellings exist for,
    pinned in the shapes real consumers use: the webhook scheme that
    base64-encodes the HMAC digest and verifies with
    ``hmac.compare_digest`` (never ``==`` — the comparison hygiene holds
    in the examples too), the advisory-lock int sliced from a digest's
    first 8 bytes, and the labelled HMAC derivation chain (a digest fed
    back in as a key). Each is pinned differentially against the stdlib
    chain it replaces."""

    def test_webhook_style_b64_signature_verify(self) -> None:
        # The Standard-Webhooks shape: the secret is the bytes after the
        # "whsec_" prefix, the signed content is
        # "{msg_id}.{timestamp}.{payload}", and the signature is the RAW
        # digest, urlsafe-base64 — not the hex. compare_digest verifies.
        import base64

        secret = base64.b64decode("whsec_3f9d2a8c".partition("_")[2])
        signed_content = (
            b"msg_5fXn0.1731634200."
            b'{"event":"invoice.paid","id":"evt_88213","amount":4200}'
        )
        signature = base64.urlsafe_b64encode(hmac_sha256_digest(secret, signed_content)).decode(
            "ascii"
        )
        expected = base64.urlsafe_b64encode(_stdlib_hmac_digest(secret, signed_content)).decode(
            "ascii"
        )
        assert signature == expected
        # The verify compare is compare_digest, never ==: the honest
        # signature verifies True, a tampered payload's does not.
        assert hmac_module.compare_digest(signature, expected)
        tampered = base64.urlsafe_b64encode(
            hmac_sha256_digest(secret, signed_content + b"!")
        ).decode("ascii")
        assert not hmac_module.compare_digest(signature, tampered)

    def test_advisory_lock_int_from_digest_prefix(self) -> None:
        # The advisory-lock shape: a stable 64-bit int for an arbitrary
        # resource name, sliced off the digest's first 8 bytes (big-endian)
        # — the spelling a caller uses to map names onto a database's
        # signed-64-bit lock identifier space.
        for name in ("tenant:42:resource:7", "db:migration:2026-09-13", "locks/shard/3"):
            raw = name.encode("utf-8")
            lock_id = int.from_bytes(sha256_digest(raw)[:8], "big")
            assert lock_id == int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
            assert 0 <= lock_id < 2**64
            # a str name hashes to the same lock int as its UTF-8 bytes
            assert lock_id == int.from_bytes(sha256_digest(name)[:8], "big")

    def test_labelled_digest_chaining_derivation(self) -> None:
        # The HKDF-style labelled derivation shape: a subkey is
        # hmac(hmac(root, label), data) — a digest fed back in as the key,
        # the chain key-derivation call sites build. The label is a str,
        # the root key bytes: both spellings of the str|bytes contract in
        # one consumer shape.
        root = b"root-key-material-32-bytes-long!!"
        label = "tors/db-session-key"
        data = "user:42:session:8f3a"
        derived = hmac_sha256_digest(hmac_sha256_digest(root, label), data)
        stdlib = hmac_module.new(
            hmac_module.new(root, label.encode("utf-8"), hashlib.sha256).digest(),
            data.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        assert derived == stdlib
        # domain separation: a different label derives a different subkey
        assert derived != hmac_sha256_digest(hmac_sha256_digest(root, "tors/cache-key"), data)
