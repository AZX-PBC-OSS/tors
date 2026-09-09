"""Contract gate for ``tors.finalize``: ``(normalized, sha256)`` in one GIL-free pass.

``finalize`` exists so a normalize-then-content-hash tail (two Python-level steps in
the pipeline's original spelling (normalize followed by
``hashlib.sha256(...).hexdigest()``) becomes one
``py.detach`` call: same transform, byte-identical hash,
no second pass over the text and no GIL-held hashing. Every test here pins byte-identity
against the stdlib expression

    (tors.normalize(t), hashlib.sha256(tors.normalize(t).encode("utf-8")).hexdigest())

: exactly the two-step expression it replaces, cross-checked against the pure-Python
finalize oracle (``reference_finalize``) so the pipeline itself stays pinned too.
"""

from __future__ import annotations

import hashlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reference import (  # noqa: I001 -- the shared oracle module (tests/reference.py)
    _COMBINING_ACUTE,
    _E_ACUTE_PRECOMPOSED,
    pathological_text,
    reference_finalize,
)
from tors import finalize, normalize

_SHA256_OF_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_SHA256_OF_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def _stdlib_finalize(text: str) -> tuple[str, str]:
    """The contract expression, spelled out: tors.normalize + stdlib sha256 of its UTF-8."""
    normalized = normalize(text)
    return normalized, hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class TestFinalizeContract:
    def test_empty_string_hashes_to_the_known_sha256_of_empty(self) -> None:
        # FIPS 180-4 test vector for the empty string, pinned literally so a wrong hash
        # wiring cannot hide behind a self-consistent-but-wrong digest compared only
        # against itself.
        assert finalize("") == ("", _SHA256_OF_EMPTY)

    def test_plain_ascii_matches_the_known_sha256_vector(self) -> None:
        # "abc" is unchanged by the pipeline, so finalize must return the FIPS test
        # vector for "abc" verbatim.
        assert finalize("abc") == ("abc", _SHA256_OF_ABC)

    def test_whitespace_only_normalizes_to_empty_and_hashes_accordingly(self) -> None:
        assert finalize("   \t\n\n\n   ") == ("", _SHA256_OF_EMPTY)

    @pytest.mark.parametrize(
        "text",
        [
            "plain text",
            "a\r\nb\rc\nd",
            "a\t\nb\n\n\nc",
            "  \t\n\n\n leading and trailing \t\n\n\n\n  ",
            "cafe" + _COMBINING_ACUTE + " \n\n\n\nwater",
            ("e" + _COMBINING_ACUTE) * 20,
            "a" + ("\n" * 100_000) + "b",
            "line1\nline2\n\n\n\n\nline3\t \t\n",
        ],
    )
    def test_finalize_is_normalize_plus_stdlib_sha256(self, text: str) -> None:
        assert finalize(text) == _stdlib_finalize(text)

    @pytest.mark.parametrize(
        "text",
        [
            "plain text",
            "a\r\nb\rc\nd",
            "cafe" + _COMBINING_ACUTE + " \n\n\n\nwater",
        ],
    )
    def test_finalize_matches_the_pure_python_finalize_oracle(self, text: str) -> None:
        # Pins the full pipeline (not just the hash tail) against the reference.
        assert finalize(text) == reference_finalize(text)

    def test_finalize_hash_is_lowercase_hex(self) -> None:
        digest = finalize("x")[1]
        assert digest == digest.lower()
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


class TestFinalizeProperty:
    @given(st.text(max_size=200))
    @settings(max_examples=300)
    def test_finalize_equals_normalize_plus_stdlib_sha256_over_arbitrary_unicode(
        self, text: str
    ) -> None:
        assert finalize(text) == _stdlib_finalize(text)
        assert finalize(text) == reference_finalize(text)

    @given(pathological_text())
    @settings(max_examples=500)
    def test_finalize_equals_normalize_plus_stdlib_sha256_over_pathological_whitespace(
        self, text: str
    ) -> None:
        assert finalize(text) == _stdlib_finalize(text)
        assert finalize(text) == reference_finalize(text)

    @given(st.text(alphabet="ab" + _E_ACUTE_PRECOMPOSED + _COMBINING_ACUTE, max_size=300))
    @settings(max_examples=300)
    def test_finalize_agrees_with_normalize_on_nfc_equivalent_inputs(self, text: str) -> None:
        # Hash-gated dedupe depends on NFC-equivalent inputs hashing identically: the
        # normalized text (and therefore its hash) must be a function of the composed
        # content, not of the incoming decomposition form.
        nfc_text = normalize(text)
        assert finalize(text) == finalize(nfc_text)


class TestSurrogateBehavior:
    """Pins the OBSERVED behavior at the pyo3 argument boundary for strings CPython can
    hold but UTF-8 cannot encode: lone surrogates (e.g. from ``surrogatepass`` decoders
    or hand-built strings). pyo3's ``&str`` extraction refuses them with
    ``UnicodeEncodeError`` before any Rust code runs, measured, not assumed (see the
    module docstring in ``src/lib.rs``). This is an open contract question,
    not a settled design: the pure-Python oracle handles surrogates fine
    at the normalize stage and only fails at the same ``encode("utf-8")`` step finalize's
    hash would hit anyway. The pin exists so any change in this behavior is a visible
    decision, not drift."""

    def test_normalize_rejects_lone_surrogates_at_the_argument_boundary(self) -> None:
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            normalize("x\ud800")

    def test_finalize_rejects_lone_surrogates_at_the_argument_boundary(self) -> None:
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            finalize("x\ud800")


class TestArgumentContract:
    """``normalize()`` never had a non-str-argument ``TypeError`` test
    anywhere in the repo; every sibling form (nfc/nfd/nfkc/nfkd) gets one
    in ``tests/test_forms.py::TestArgumentContract``, but ``normalize()``
    only had the surrogate-rejection case above pinned."""

    @pytest.mark.parametrize(
        "not_str", [b"abc", bytearray(b"abc"), 123, None], ids=["bytes", "bytearray", "int", "none"]
    )
    def test_normalize_non_str_argument_raises_type_error(self, not_str: object) -> None:
        with pytest.raises(TypeError):
            normalize(not_str)  # type: ignore[arg-type]


class TestIdentityReturnContract:
    """The identity-return contract for ``tors.finalize``: on an input
    the whole pipeline leaves untouched, the STRING element is the ORIGINAL
    object (``finalize(s)[0] is s``) and the hash still computes, from the
    borrowed input buffer, with no output allocation at all (the integrated
    hash and the identity probe live in normalize_impl/finalize_impl; the
    quick-check lane and the post-scan output==input lane are the same two
    lanes ``tors.normalize`` documents)."""

    def test_already_clean_input_returns_the_same_object_and_the_right_hash(self) -> None:
        clean = "plain text\n\nwith paragraphs\n\ncaf\u00e9 na\u00efve"
        text, digest = finalize(clean)
        assert text is clean
        assert digest == hashlib.sha256(clean.encode("utf-8")).hexdigest()

    @pytest.mark.parametrize(
        "dirty",
        ["a\r\nb", "a \nb", "a\n\n\nb", "  leading", "trailing  ", "cafe\u0301"],
        ids=["crlf", "space-before-nl", "blank-run", "leading", "trailing", "decomposed"],
    )
    def test_dirty_inputs_string_element_is_a_new_object(self, dirty: str) -> None:
        text, digest = finalize(dirty)
        assert (text, digest) == _stdlib_finalize(dirty)  # the pair stays pinned
        assert text is not dirty

    @given(pathological_text())
    @settings(max_examples=500)
    def test_value_identity_implies_object_identity_for_the_string_element(self, text: str) -> None:
        result, _ = finalize(text)
        if result == text:
            assert result is text
