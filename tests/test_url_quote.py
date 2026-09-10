"""Contract gate for the percent-encoding pair: ``tors.quote``,
``tors.quote_plus``, ``tors.unquote``, ``tors.unquote_plus``: byte-exact
``urllib.parse`` parity, UTF-8 only, one detached native pass.

The stdlib spelling is pure Python (Lib/urllib/parse.py): a GIL-held
whole-text pass for the most-used encoding operation in web and ingestion
pipelines; tors is the same semantics as one native pass under
``py.detach`` (the GIL bands pinned in tests/test_gil_release.py).

Parity is byte-exact against the running interpreter (the differentials
below, over arbitrary text and safe-sets; they settle everything), with
the stdlib's quirks pinned by name so a naive RFC 3986 implementation
cannot pass this gate:

- The never-quote set is RFC 3986 unreserved (ASCII letters, digits,
  ``_ . - ~``, the stdlib's ``_ALWAYS_SAFE``) plus ``safe``'s ASCII
  members; every other byte of the input's UTF-8 encoding becomes ``%xx``
  uppercase hex.
- ``safe`` is byte-level and ASCII-only: the stdlib normalizes a str
  ``safe`` with ``safe.encode('ascii', 'ignore')``, silently dropping
  non-ASCII members: ``quote("é", "é")`` is ``"%c3%a9"``, not ``"é"``.
- ``%`` in ``safe`` is honored like any other byte: it stays literal
  (``quote("50%", "%")`` → ``"50%"``).
- ``quote_plus`` is not "quote then replace ``%20`` with ``+``": the
  stdlib quotes with ``' '`` appended to ``safe`` (spaces never encode)
  and then swaps every space for ``+``: same output for valid input,
  but a literal ``+`` in the text escapes to ``%2B`` unless the caller
  safed it, and a caller-safed space still becomes ``+``.
- ``unquote`` accepts lowercase hex; a ``%`` not followed by two hex
  digits stays verbatim (``%zz``, a trailing ``%``, ``%%41`` → ``"%A"``);
  escapes decoding to invalid UTF-8 take the replace handler
  (``%e2%28%a1`` → ``"\ufffd(\ufffd"``, maximal-subpart).
- The ASCII-run fragmentation quirk: the stdlib unquotes and UTF-8-decodes
  each maximal ASCII run independently (its ``_asciire`` walk), passing
  non-ASCII segments through verbatim, so an escape split across an
  ASCII/non-ASCII boundary decodes as two fragments with separate replace
  verdicts (``unquote("%c3é%a9")`` → ``"\ufffdé\ufffd"``, not ``"éé"``),
  and a non-ASCII char interrupts an escape (``unquote("%Cé3")`` →
  verbatim).
- ``unquote_plus`` swaps ``+`` → ``' '`` before unquoting, so an escaped
  ``%2B`` survives as a literal ``+`` while a raw ``+`` becomes a space;
  the order is the semantics.

The stdlib's ``encoding``/``errors`` parameters are out of scope,
documented as such: the encode side is always UTF-8-strict, the decode
side always UTF-8-replace.

The identity contracts (each pinned below as object identity):

- ``quote(s, safe) is s`` exactly when ``quote(s, safe) == s``: the
  borrowed lane is "no byte needs encoding", which is precisely output
  equals input. Strictly stronger than the stdlib, which re-decodes to a
  fresh string on every nonempty input (only its empty-input early return
  hands back the original object).
- ``unquote(s) is s`` when ``s`` contains no ``%``, exactly the stdlib's
  own ``'%' not in string`` fast path, which is also where it returns the
  original object. The asymmetric residue, both engines: an input whose
  every ``%`` is invalid hex (``"a%zz"``) decodes to an equal but new
  string. (One measured artifact: an output that is a single Latin-1
  character is CPython's cached 1-char singleton, so ``unquote("%")``
  hands back an object indistinguishable from the input by ``is``; an
  object-model fact of every fresh 1-char string, pinned as such.)
- ``unquote_plus(s) is s`` when ``s`` contains neither ``+`` nor ``%``.

Round-trip properties: ``unquote∘quote == id`` and
``unquote_plus∘quote_plus == id`` over UTF-8 text, for safe-sets
without ``%`` (a literal ``%`` preserved by a caller's safe is then
re-read as an escape by the decoder: ``quote("50%41", "%")`` →
``"50%41"`` → ``unquote`` → ``"50A"``, inherent to the design and the
stdlib's answer too), and additionally without ``+`` for the plus pair
(a caller-safed literal ``+`` is a space to ``unquote_plus``).

Argument-boundary contract:

- ``text`` and ``safe`` must be exactly ``str``: anything else raises
  ``TypeError`` (the str-exactly rule every tors str argument follows).
- A ``text`` holding lone surrogates raises ``UnicodeEncodeError`` on the
  encode side in both engines, message-for-message (the stdlib's
  ``string.encode('utf-8', 'strict')``; pinned against the live oracle).
  The decode side is the one intentional asymmetry: the stdlib's
  ``unquote`` passes lone surrogates through (its no-``%`` fast path
  returns the input without ever encoding, and non-ASCII segments are
  yielded verbatim), while tors refuses them with ``UnicodeEncodeError``
  at the boundary, the standard str-in price every tors function pays
  for the zero-copy UTF-8 borrow (the ``finalize``/``diff_opcodes``
  precedent), recorded here rather than silently diverging.
- A ``safe`` holding lone surrogates: the stdlib's ``'ignore'``
  normalization drops them silently; tors refuses with
  ``UnicodeEncodeError`` (the same boundary); the refusal is pinned, the
  stdlib's silent drop recorded as the contrast.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Callable

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tors import quote, quote_plus, unquote, unquote_plus

_E_MOJI = "\U0001f600"  # 4-byte UTF-8, astral plane
_LONG_ROW = "user name+50%/café?page=1&x=~y#frag" + _E_MOJI
_LONG_ROW_QUOTED = "user%20name%2B50%25/caf%C3%A9?page=1&x=~y#frag%F0%9F%98%80"


# --- The golden batteries --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "safe", "expected"),
    [
        # The never-quote set (RFC 3986 unreserved) passes with safe="".
        ("AZaz09_.-~", "", "AZaz09_.-~"),
        ("", "/", ""),
        ("abc", "/", "abc"),
        # Reserved and unsafe bytes escape; the default safe="/" keeps
        # slashes literal.
        ("a b/c?d#e", "/", "a%20b/c%3Fd%23e"),
        ("a b/c?d#e", "", "a%20b%2Fc%3Fd%23e"),
        ("a+b", "", "a%2Bb"),
        ("\n", "", "%0A"),
        # Non-ASCII encodes as its UTF-8 bytes, uppercase hex: 2-byte é,
        # 3-byte CJK, 4-byte astral emoji.
        ("é", "", "%C3%A9"),
        ("東京", "", "%E6%9D%B1%E4%BA%AC"),
        (_E_MOJI, "", "%F0%9F%98%80"),
        # `%` is just a byte: escaped by default, literal when safed.
        ("50%", "", "50%25"),
        ("50%", "%", "50%"),
        # Any ASCII member of safe is honored, including space.
        ("a b", " ", "a b"),
        ("a+b", "+", "a+b"),
        # Non-ASCII safe members are ignored (the stdlib normalizes safe
        # with encode('ascii', 'ignore'); byte-level, ASCII-only).
        ("é", "é", "%C3%A9"),
        ("aéb", "é/", "a%C3%A9b"),
        ("東京", "東京", "%E6%9D%B1%E4%BA%AC"),
        # The mixed long row: every quirk class at once.
        (_LONG_ROW, "/?&=#", _LONG_ROW_QUOTED),
    ],
    ids=[
        "unreserved-never-quoted",
        "empty",
        "plain-ascii-identity",
        "default-safe-keeps-slashes",
        "empty-safe-escapes-slashes",
        "plus-escapes",
        "control-byte",
        "two-byte-utf8",
        "three-byte-utf8",
        "four-byte-utf8",
        "percent-escapes-by-default",
        "percent-in-safe-honored",
        "space-in-safe-honored",
        "plus-in-safe-honored",
        "non-ascii-safe-ignored",
        "non-ascii-safe-ignored-mixed",
        "non-ascii-safe-ignored-cjk",
        "long-mixed-row",
    ],
)
def test_quote_golden_battery(text: str, safe: str, expected: str) -> None:
    """The fixed anchor of the encode contract: every row asserts the exact
    expected string and the running stdlib's agreement; the differential
    below proves the agreement in general, this battery pins the named
    quirk classes so a regression reads as a row, not a shrug."""
    assert quote(text, safe) == expected
    assert quote(text, safe) == urllib.parse.quote(text, safe)


@pytest.mark.parametrize(
    ("text", "safe", "expected"),
    [
        ("", "", ""),
        ("abc", "", "abc"),
        # Space is swapped for `+`, never %20: the stdlib quotes with
        # space added to safe, then swaps.
        ("a b", "", "a+b"),
        ("a b c", "", "a+b+c"),
        # A literal `+` escapes to %2B unless the caller safed it, the
        # quirk that proves quote_plus is not quote-then-replace-%20.
        ("a+b", "", "a%2Bb"),
        ("a+b", "+", "a+b"),
        # No space in the text: exactly plain quote (safe honored as such).
        ("a/b", "/", "a/b"),
        ("a/b", "", "a%2Fb"),
        # A caller-safed space still becomes `+` (space is always swapped).
        ("a b", " ", "a+b"),
        # Non-ASCII and the safe-ignore quirk carry through unchanged.
        ("é", "", "%C3%A9"),
        ("é x", "", "%C3%A9+x"),
        ("é", "é", "%C3%A9"),
        (_LONG_ROW, "", "user+name%2B50%25%2Fcaf%C3%A9%3Fpage%3D1%26x%3D~y%23frag%F0%9F%98%80"),
    ],
    ids=[
        "empty",
        "plain-ascii-identity",
        "space-becomes-plus",
        "repeated-spaces",
        "literal-plus-escapes",
        "caller-safed-plus-stays",
        "no-space-is-plain-quote",
        "no-space-plain-quote-unsafe-slash",
        "caller-safed-space-still-swaps",
        "non-ascii",
        "non-ascii-with-space",
        "non-ascii-safe-ignored",
        "long-mixed-row",
    ],
)
def test_quote_plus_golden_battery(text: str, safe: str, expected: str) -> None:
    """The plus-shaped anchor: the space/plus interplay the stdlib's own
    spelling produces (space in safe, then swap; not quote-then-replace),
    each row cross-checked against the running stdlib."""
    assert quote_plus(text, safe) == expected
    assert quote_plus(text, safe) == urllib.parse.quote_plus(text, safe)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("abc", "abc"),
        ("plain text no escapes", "plain text no escapes"),
        # Valid escapes, both hex cases.
        ("abc%20def", "abc def"),
        ("%C3%A9", "é"),
        ("%c3%a9", "é"),
        ("%F0%9F%98%80", _E_MOJI),
        ("%2F", "/"),
        # Invalid escapes stay verbatim: non-hex, truncated, trailing,
        # and the %%41 double (first % pairs with nothing, second decodes).
        ("%zz", "%zz"),
        ("%e", "%e"),
        ("abc%", "abc%"),
        ("100%", "100%"),
        ("%g%41", "%gA"),
        ("%%41", "%A"),
        ("a%", "a%"),
        # Invalid UTF-8 from escapes decodes with the replace handler:
        # maximal-subpart substitution, one U+FFFD per invalid subpart.
        ("%ff", "\ufffd"),
        ("%e2%28%a1", "\ufffd(\ufffd"),
        ("x%c3", "x\ufffd"),
        # The fragmentation quirk: each maximal ASCII run decodes
        # independently, non-ASCII segments pass verbatim. An escape split
        # across the boundary is two fragments with separate replace
        # verdicts (%c3é%a9 is not éé), and a non-ASCII char interrupts
        # an escape entirely.
        ("%C3é%A9", "\ufffdé\ufffd"),
        ("%Cé3", "%Cé3"),
        ("é%41", "éA"),
        ("%41é%42", "AéB"),
        # `+` is not special to unquote.
        ("a+b", "a+b"),
        ("%2B", "+"),
    ],
    ids=[
        "empty",
        "no-escapes",
        "no-escapes-longer",
        "valid-escape-space",
        "valid-escape-uppercase-hex",
        "valid-escape-lowercase-hex",
        "valid-escape-astral",
        "valid-escape-reserved",
        "invalid-non-hex-verbatim",
        "invalid-truncated-verbatim",
        "invalid-trailing-verbatim",
        "invalid-trailing-verbatim-word",
        "non-hex-then-valid",
        "double-percent-pair",
        "trailing-percent-word",
        "invalid-utf8-single",
        "invalid-utf8-maximal-subpart",
        "invalid-utf8-truncated-lead",
        "fragmentation-split-escape",
        "fragmentation-interrupted-escape",
        "non-ascii-verbatim-around-escape",
        "non-ascii-between-escapes",
        "plus-not-special",
        "escaped-plus-is-literal-plus",
    ],
)
def test_unquote_golden_battery(text: str, expected: str) -> None:
    """The decode anchor: the stdlib's verbatim-escape, replace-handler,
    and ASCII-run-fragmentation behaviors, every row cross-checked against
    the running stdlib (the fragmentation rows are where a whole-string
    decoder gives the wrong answer)."""
    assert unquote(text) == expected
    assert unquote(text) == urllib.parse.unquote(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("abc", "abc"),
        # `+` becomes a space before unquoting: a raw `+` is a space, an
        # escaped %2B survives as a literal `+`.
        ("a+b", "a b"),
        ("%2B", "+"),
        ("+%2B", " +"),
        ("a+b%41", "a bA"),
        ("%c3%a9+ok", "é ok"),
        ("cake%2B+%C3%A9", "cake+ é"),
        # No `+` and no `%`: exactly unquote's behavior.
        ("plain", "plain"),
        ("%C3%A9", "é"),
        ("%%41", "%A"),
    ],
    ids=[
        "empty",
        "identity",
        "raw-plus-becomes-space",
        "escaped-plus-survives",
        "both-plus-kinds",
        "plus-and-escape",
        "non-ascii-with-plus",
        "mixed-row",
        "no-plus-is-plain-unquote",
        "no-plus-escape",
        "no-plus-double-percent",
    ],
)
def test_unquote_plus_golden_battery(text: str, expected: str) -> None:
    """The plus-decode anchor: the swap-first ordering pinned by the rows
    that distinguish it (``%2B`` survives, a raw ``+`` does not), every
    row cross-checked against the running stdlib."""
    assert unquote_plus(text) == expected
    assert unquote_plus(text) == urllib.parse.unquote_plus(text)


def test_the_default_safe_values_are_the_stdlibs_own() -> None:
    """The pyo3 layer owns the defaults and they are the stdlib's:
    ``quote(s)`` is ``quote(s, "/")`` and ``quote_plus(s)`` is
    ``quote_plus(s, "")``, the most common call shapes match with no
    arguments at all."""
    assert quote("a b/c") == "a%20b/c"
    assert quote("a b/c") == urllib.parse.quote("a b/c")
    assert quote_plus("a b/c") == "a+b%2Fc"
    assert quote_plus("a b/c") == urllib.parse.quote_plus("a b/c")


# --- The hypothesis differentials ------------------------------------------------------


@st.composite
def _text_and_safe(draw: st.DrawFn) -> tuple[str, str]:
    """Arbitrary text (hypothesis's full alphabet: any script, any marks,
    any width; surrogates excluded by the strategy, they get the argument
    contract) with safe-sets from two alphabets: the punctuation-heavy set
    that exercises every quirk class (reserved bytes, ``%``, space, ``+``,
    the always-safe members) and arbitrary Unicode, the non-ASCII-member
    ignore lane."""
    text = draw(st.text(max_size=60))
    safe = draw(
        st.one_of(
            st.text(alphabet="/?#+&=%_.-~ Z9", max_size=8),
            st.text(max_size=6),
        )
    )
    return text, safe


@given(_text_and_safe())
@settings(max_examples=400)
def test_quote_matches_the_running_stdlib_exactly(text_safe: tuple[str, str]) -> None:
    """The encode differential: byte-exact agreement with the running
    ``urllib.parse.quote`` over arbitrary text and safe-sets: the proof
    that settles the never-quote set, the safe-member quirks (ASCII
    honored, non-ASCII ignored), uppercase hex, and multibyte widths all
    at once."""
    text, safe = text_safe
    assert quote(text, safe) == urllib.parse.quote(text, safe)


@given(_text_and_safe())
@settings(max_examples=400)
def test_quote_plus_matches_the_running_stdlib_exactly(text_safe: tuple[str, str]) -> None:
    """The plus differential: same coverage through the space/plus
    machinery (space added to safe then swapped, literal ``+`` escaped
    unless caller-safed)."""
    text, safe = text_safe
    assert quote_plus(text, safe) == urllib.parse.quote_plus(text, safe)


@given(st.text(max_size=60))
@settings(max_examples=400)
def test_unquote_matches_the_running_stdlib_exactly(text: str) -> None:
    """The decode differential over arbitrary text: valid and invalid
    escapes, both hex cases, the replace handler, and the ASCII-run
    fragmentation all settle here (the battery pins the named rows)."""
    assert unquote(text) == urllib.parse.unquote(text)


@given(st.text(max_size=60))
@settings(max_examples=400)
def test_unquote_plus_matches_the_running_stdlib_exactly(text: str) -> None:
    """The plus-decode differential: the swap-first ordering under
    arbitrary mixes of ``+`` and ``%``."""
    assert unquote_plus(text) == urllib.parse.unquote_plus(text)


# --- The round-trip properties ---------------------------------------------------------


@given(_text_and_safe())
@settings(max_examples=300)
def test_unquote_inverts_quote_over_utf8_text(text_safe: tuple[str, str]) -> None:
    """``unquote(quote(s)) == s`` for UTF-8 text. The property is claimed
    over ``%``-free safe-sets: a literal ``%`` a caller safed is re-read
    as an escape by the decoder (``quote("50%41", "%")`` → ``"50%41"`` →
    ``"50A"``), inherent to the design and the stdlib's answer too, so
    excluding it states the inverse's actual scope, not a hole."""
    text, safe = text_safe
    assume("%" not in safe)
    assert unquote(quote(text, safe)) == text


@given(_text_and_safe())
@settings(max_examples=300)
def test_unquote_plus_inverts_quote_plus_over_utf8_text(text_safe: tuple[str, str]) -> None:
    """``unquote_plus(quote_plus(s)) == s``, additionally over ``+``-free
    safe-sets (a caller-safed literal ``+`` survives the encode and is a
    space to the decoder), for the same reason as the plain pair."""
    text, safe = text_safe
    assume("%" not in safe and "+" not in safe)
    assert unquote_plus(quote_plus(text, safe)) == text


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello world",
        "café naïve",
        "\U0001f600 emoji ☃ snowman",
        "path/to/file?query=1&other=2",
        "50% plus + signs",
        "tilde~under_score.dot-dash",
        "latin àéîòû extremes ÿ",
        "cjk 世界 katakana カ",
    ],
    ids=[
        "empty",
        "spaces",
        "accents",
        "astral",
        "reserved",
        "percent-and-plus",
        "unreserved",
        "latin-extremes",
        "cjk",
    ],
)
@pytest.mark.parametrize("safe", ["", "/?"], ids=["empty-safe", "safe-members"])
def test_round_trips_over_the_fixed_sample_rows(text: str, safe: str) -> None:
    """The deterministic round-trip anchor: both pairs close the loop on
    the fixed samples (every UTF-8 width, reserved and unreserved bytes,
    the ``%``/``+`` literals), with safe members riding through under the
    ``%``/``+``-free safe-sets the inverse property is claimed over."""
    assert unquote(quote(text, safe)) == text
    assert unquote_plus(quote_plus(text, safe)) == text


# --- The identity contracts -----------------------------------------------------------


class TestIdentityContracts:
    def test_quote_returns_the_original_object_when_nothing_encodes(self) -> None:
        """The zero-cost lane: text over the never-quote set returns the
        original input object (no allocation, no copy; ``Cow::Borrowed``
        handed back as the same ``PyObject``)."""
        s = "AZaz09_.-~/"
        assert quote(s) is s

    def test_quote_identity_is_the_net_identity_idiom(self) -> None:
        """The complete form: a safe-set that covers every escapable byte
        of the input still returns the original object: ``is s`` exactly
        when ``== s`` (the no-encode lane is precisely output equals
        input, so a caller-safed no-op encode borrows too)."""
        s = "a b"
        assert quote(s, " ") is s
        s2 = "a+b"
        assert quote(s2, "+") is s2
        s3 = "50%"
        assert quote(s3, "%") is s3

    def test_quote_identity_is_stronger_than_the_stdlibs(self) -> None:
        """The measured contrast: the stdlib's nonempty fast path
        (``not bs.rstrip(...)`` → ``bs.decode()``) always allocates a
        fresh string, so ``urllib.parse.quote(s) is s`` is False for every
        nonempty s; tors returns the input object itself. (The stdlib's
        only identity lane is the EMPTY input, where ``quote`` returns its
        argument before any encoding.)"""
        s = "AZaz09"
        assert urllib.parse.quote(s) is not s
        assert quote(s) is s
        empty = ""
        assert urllib.parse.quote(empty) is empty
        assert quote(empty) is empty

    def test_quote_that_changes_anything_returns_a_new_object(self) -> None:
        """The contrapositive: an encode that escapes anything is a fresh
        string (the input object never mutated or handed back unequal)."""
        s = "a b"
        result = quote(s)
        assert result == "a%20b"
        assert result is not s

    def test_quote_plus_identity_lanes(self) -> None:
        """The plus shapes: no space and nothing encodable → the original
        object (the no-space branch is plain quote, so its borrow lane
        holds); a space in the text always rewrites (the swap), so a fresh
        string comes back there no matter what."""
        s = "a+b"
        assert quote_plus(s, "+") is s
        s2 = "abc"
        assert quote_plus(s2) is s2
        empty = ""
        assert quote_plus(empty) is empty
        s3 = "a b"
        assert quote_plus(s3) is not s3

    def test_unquote_returns_the_original_object_exactly_when_no_percent(self) -> None:
        """The decode identity lane is the stdlib's own fast path, and it
        is exactly no-``%``: any input containing ``%`` takes the fragment
        walk and returns a fresh string, including the asymmetric residue
        where nothing valid decodes (``"a%zz"`` comes back equal but new,
        in both engines; the stdlib's ``''.join`` builds it fresh too)."""
        for s in ("abc", "", "a+b c/d", "plain text", "é 東京"):
            assert unquote(s) is s
        for fresh in ("a%zz", "%C3%A9", "a%", "%%41", "%2B"):
            assert unquote(fresh) is not fresh

    def test_unquote_percent_row_is_the_latin1_singleton_artifact(self) -> None:
        """The one measured ``%``-containing input whose output is the
        input object: ``unquote("%")``. The lane is not the borrow (the
        input contains ``%``); the fresh one-character output is CPython's
        cached Latin-1 singleton, the same object as the input literal,
        the object-model fact of every fresh 1-char string. Not the
        identity contract; recorded so the row cannot be misread as a
        borrow-lane widening."""
        singleton = "%"
        sliced = "%%"[:1]  # a fresh 1-char slice: the cached singleton
        assert sliced is singleton
        assert unquote("%") is singleton
        # Three characters, same walk: a fresh string.
        s = "%zz"
        assert unquote(s) == s
        assert unquote(s) is not s

    def test_unquote_plus_identity_lanes(self) -> None:
        """The plus-decode lane is exactly no-``+``-and-no-``%``: a raw
        ``+`` always rewrites (swap first), a ``%`` takes the walk."""
        s = "abc"
        assert unquote_plus(s) is s
        s2 = ""
        assert unquote_plus(s2) is s2
        s3 = "a+b"
        assert unquote_plus(s3) is not s3
        s4 = "a%zz"
        assert unquote_plus(s4) is not s4


# --- The argument-boundary contract ---------------------------------------------------


class TestArgumentContract:
    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), memoryview(b"abc"), 123, None],
        ids=["bytes", "bytearray", "memoryview", "int", "none"],
    )
    @pytest.mark.parametrize("fn", [quote, quote_plus], ids=["quote", "quote_plus"])
    def test_non_str_text_raises_type_error_on_the_encode_side(
        self, fn: Callable[..., str], not_str: object
    ) -> None:
        """``text`` is exactly ``str`` (the str-exactly rule every tors str
        argument follows). The
        stdlib's bytes-accepting spelling (``quote_from_bytes``) stays
        with the stdlib: tors is the str surface, UTF-8 only; the
        bytes-side caller already has the stdlib answer."""
        with pytest.raises(TypeError):
            fn(not_str, "/")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), memoryview(b"abc"), 123, None],
        ids=["bytes", "bytearray", "memoryview", "int", "none"],
    )
    @pytest.mark.parametrize("fn", [unquote, unquote_plus], ids=["unquote", "unquote_plus"])
    def test_non_str_text_raises_type_error_on_the_decode_side(
        self, fn: Callable[..., str], not_str: object
    ) -> None:
        """The decode side's str-only contract (one argument, no safe):
        bytes input raises ``TypeError``, the same scope line as the
        encode side (the stdlib's bytes decode stays with the stdlib)."""
        with pytest.raises(TypeError):
            fn(not_str)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "not_str",
        [b"/", bytearray(b"/"), 123, None],
        ids=["bytes", "bytearray", "int", "none"],
    )
    @pytest.mark.parametrize("fn", [quote, quote_plus], ids=["quote", "quote_plus"])
    def test_non_str_safe_raises_type_error(self, fn: Callable[..., str], not_str: object) -> None:
        """``safe`` is exactly ``str`` too; the stdlib also accepts bytes
        safe-sets (byte-level), out of scope for the same reason as bytes
        text."""
        with pytest.raises(TypeError):
            fn("abc", not_str)  # type: ignore[arg-type]

    def test_lone_surrogate_text_matches_the_stdlib_on_the_encode_side(self) -> None:
        """The encode-side surrogate boundary, message-for-message: both
        engines raise ``UnicodeEncodeError`` from the UTF-8 encode (the
        stdlib's ``string.encode('utf-8', 'strict')``, tors's zero-copy
        ``&str`` borrow), the same exception text, pinned against the
        live oracle, not a recorded literal."""
        for fn, stdlib_fn in (
            (quote, urllib.parse.quote),
            (quote_plus, urllib.parse.quote_plus),
        ):
            with pytest.raises(UnicodeEncodeError) as tors_exc:
                fn("abc\ud800")
            with pytest.raises(UnicodeEncodeError) as stdlib_exc:
                stdlib_fn("abc\ud800")
            assert str(tors_exc.value) == str(stdlib_exc.value)

    def test_lone_surrogate_text_is_refused_on_the_decode_side(self) -> None:
        """The decode side's one intentional asymmetry, recorded rather
        than laundered: the stdlib's ``unquote`` passes lone surrogates
        through (its no-``%`` fast path returns the input without ever
        encoding, and non-ASCII segments are yielded verbatim by the
        ``_asciire`` walk; the surrogate never meets an encode call), so
        ``urllib.parse.unquote("abc\\ud800")`` is ``"abc\\ud800"``. tors
        refuses with ``UnicodeEncodeError`` at the boundary, the standard
        str-in price every tors function pays for the zero-copy UTF-8
        borrow (the ``finalize``/``diff_opcodes`` precedent), the same
        boundary the encode side pays above."""
        with pytest.raises(UnicodeEncodeError):
            unquote("abc\ud800")
        with pytest.raises(UnicodeEncodeError):
            unquote("%C3\ud800%A9")
        with pytest.raises(UnicodeEncodeError):
            unquote_plus("abc\ud800")
        with pytest.raises(UnicodeEncodeError):
            unquote_plus("a+\ud800")

    def test_lone_surrogate_safe_is_refused_where_the_stdlib_drops_it(self) -> None:
        """``safe`` holding a lone surrogate: the stdlib's
        ``encode('ascii', 'ignore')`` normalization drops it silently
        (measured: ``urllib.parse.quote("abc", safe="\\ud800")`` →
        ``"abc"``), while tors refuses with ``UnicodeEncodeError``, the
        same str-in boundary as the text. The refusal is the pinned
        behavior; the stdlib's silent drop is recorded as the contrast,
        the same class as its non-ASCII-member drop this gate pins as a
        quirk above."""
        assert urllib.parse.quote("abc", safe="\ud800") == "abc"
        with pytest.raises(UnicodeEncodeError):
            quote("abc", safe="\ud800")
        with pytest.raises(UnicodeEncodeError):
            quote_plus("abc", safe="\ud800")
