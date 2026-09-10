"""Contract gate for ``tors.detect_encoding``: a best-guess codec name for
non-UTF8 legacy/OCR byte content, via ``chardetng`` (the detector Firefox
ships).

Unlike ``decode_utf8``/``utf8_is_valid``, there is no stdlib oracle to be
byte-exact against here: encoding detection is inherently heuristic (the same
bytes can be plausible under more than one codec), so this suite pins the
contract: never raises for well-formedness reasons, always returns some
codec name Python's own ``bytes.decode()`` accepts, and correctly identifies
a battery of real, unambiguous encoding-specific fixtures, rather than
claiming universal correctness on arbitrary input.

The intended pipeline shape, documented on the function itself: call
``utf8_is_valid`` first; only reach for ``detect_encoding`` on the bytes that
already failed that check, then decode with the returned codec name. This
suite does not re-litigate that framing, only the function's own behavior.

The argument contract is the bytes-in surface's: exactly ``bytes``
(``bytearray`` / ``memoryview`` / ``str`` -> ``TypeError``), the same
zero-copy immutable-borrow rationale as ``decode_utf8``/``utf8_is_valid``.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tors import detect_encoding


class TestReturnsADecodableCodecName:
    """Whatever name comes back must be one Python's own ``bytes.decode``
    accepts, the contract that makes the return value useful, not just
    non-empty."""

    @given(raw=st.binary(max_size=4096))
    def test_the_guess_is_always_a_usable_codec_name(self, raw: bytes) -> None:
        name = detect_encoding(raw)
        assert isinstance(name, str)
        assert name
        # Never raises LookupError: whatever chardetng names, Python's own
        # codec registry recognizes it (case/punctuation-insensitively).
        raw.decode(name, errors="replace")

    def test_empty_bytes_guess_utf8(self) -> None:
        assert detect_encoding(b"").lower() == "utf-8"

    def test_plain_ascii_guesses_utf8(self) -> None:
        assert detect_encoding(b"hello, plain ASCII text.").lower() == "utf-8"

    def test_well_formed_multibyte_utf8_guesses_utf8(self) -> None:
        text = "café société 日本語のテキスト".encode()
        assert detect_encoding(text).lower() == "utf-8"


class TestKnownEncodingFixtures:
    """Real, unambiguous byte sequences for a handful of legacy encodings:
    the deterministic floor under the hypothesis property above. Each fixture
    round-trips through its own encoding (the ground truth) and is asserted
    to decode cleanly under tors's guess, which is the property that
    actually matters for the ingestion pipeline this exists for: not "did it
    guess the exact codec name" (multiple names can be correct; cp1252 and
    windows-1252 are the same table) but "does decoding with the guess
    recover the original text.\""""

    @pytest.mark.parametrize(
        ("text", "encoding"),
        [
            ("Über die Straße, naïve café — €5,00", "windows-1252"),
            ("Résumé: crème brûlée, garçon.", "iso-8859-1"),
            ("日本語のテキストです。", "shift_jis"),
            ("한국어 텍스트입니다.", "euc-kr"),
        ],
        ids=["windows-1252", "iso-8859-1", "shift_jis", "euc-kr"],
    )
    def test_the_guess_decodes_the_fixture_back_to_the_original(
        self, text: str, encoding: str
    ) -> None:
        raw = text.encode(encoding)
        guess = detect_encoding(raw)
        assert raw.decode(guess) == text


class TestNeverPanicsOnArbitraryBytes:
    """The whole point of a heuristic detector: some guess for any byte
    string, well-formed UTF-8 or not, ASCII or not, empty or huge."""

    @given(raw=st.binary(min_size=0, max_size=8192))
    def test_arbitrary_bytes_never_raise(self, raw: bytes) -> None:
        detect_encoding(raw)

    def test_all_256_byte_values_at_once(self) -> None:
        detect_encoding(bytes(range(256)))


class TestTldHint:
    """The ``tld`` parameter is threaded through to chardetng's own
    disambiguation logic; this suite pins that it's accepted and never
    breaks the "always returns a usable name" contract, not that it changes
    any specific outcome (that heuristic is chardetng's to evolve)."""

    def test_tld_hint_is_accepted_without_leading_dot(self) -> None:
        raw = "日本語のテキストです。".encode("shift_jis")
        guess = detect_encoding(raw, tld="jp")
        assert raw.decode(guess) == "日本語のテキストです。"

    def test_none_and_omitted_tld_agree(self) -> None:
        raw = b"plain ascii"
        assert detect_encoding(raw, tld=None) == detect_encoding(raw)

    def test_tld_hint_normalizes_leading_dot_and_case(self) -> None:
        """The doc contract (src/encoding_impl.rs `normalize_tld`, and the
        pyfunction docstring at src/py/encoding.rs) promises the hint is
        accepted in "whatever natural spelling a caller has": leading dot
        or not, any case. Every spelling below must agree with the bare
        lowercase form already pinned above."""
        raw = "日本語のテキストです。".encode("shift_jis")
        baseline = detect_encoding(raw, tld="jp")
        assert detect_encoding(raw, tld=".jp") == baseline
        assert detect_encoding(raw, tld="JP") == baseline
        assert detect_encoding(raw, tld=".JP") == baseline

    def test_a_malformed_tld_hint_degrades_to_no_hint(self) -> None:
        """A hint the normalizer can't use (an embedded period, or empty)
        is documented to degrade to "no hint" rather than panic or produce
        a different result than omitting the parameter entirely."""
        raw = "日本語のテキストです。".encode("shift_jis")
        no_hint = detect_encoding(raw)
        assert detect_encoding(raw, tld="example.co.jp") == no_hint
        assert detect_encoding(raw, tld="") == no_hint

    def test_windows_874_translates_to_cp874(self) -> None:
        """chardetng can guess the label "windows-874", which Python's codec
        registry does not recognize under that spelling; src/encoding_impl.rs
        `python_codec_name` documents this as the one translation needed
        (windows-874 -> cp874), pinned crate-side with this exact fixture but
        never before exercised through the pyo3 wrapper."""
        assert detect_encoding(b"A\xa7\xa8") == "cp874"


class TestBytesOnlyArgumentContract:
    """Same zero-copy immutable-borrow rationale as ``decode_utf8``/
    ``utf8_is_valid`` (see those test modules' docstrings). The pin:
    anything else is a ``TypeError``."""

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
            detect_encoding(not_bytes)  # type: ignore[arg-type]

    def test_non_str_tld_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            detect_encoding(b"abc", tld=123)  # type: ignore[arg-type]
