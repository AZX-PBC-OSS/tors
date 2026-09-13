"""Contract gate for ``tors.utf8_byte_len``: the UTF-8 byte length of a
``str`` without building the bytes object — ``len(s.encode("utf-8"))``
with the copy taken out.

The expression this replaces is pure waste whenever only the COUNT is
wanted: ``len(s.encode("utf-8"))`` allocates a full ``bytes`` object,
measures it, and throws it away. The motivating pattern is a size cap in
front of a store (the issue's own framing): TaskQ spells it twice —
``client/_args.py`` checks idempotency-key and scope byte caps on every
enqueue, and ``backend/_terminal.py`` re-encodes a serialized result of
up to 64 KiB (``MAX_RESULT_BYTES``) on every success, a genuine double
pass (the byte count existed inside the serializer's output and was
discarded by the ``.decode()`` that produced the ``str``). The general
shape is wire limits, column limits, batch byte budgets: any gate that
asks "how many UTF-8 bytes is this?" without wanting the bytes.

**Companion, not standalone**: this function ships in the scan family's
binding module (``src/py/scan.rs``, the ``contains_unescaped`` /
``find_unescaped`` surface) because it is the pinned companion of the
escape-parity scan (#50) — same module, same test/bench harness patterns
— and honest sizing says it would not stand alone: a short-string encode
is a few hundred nanoseconds, so the win only exists at large inputs or
hot paths where the copy is the cost (the 64 KiB terminal case is about
a microsecond and a half of pure memcpy per success).

Implementation, and the deliberate deviation from the issue's sketch: the
issue proposed hand-rolled per-range arithmetic over CPython's internal
UCS1/UCS2/UCS4 storage (1/2/3/4 bytes per codepoint by range, surrogate
pairs in UCS2 counting as one 4-byte sequence). This implementation does
NOT do that. It uses the repo's standard str borrow — pyo3's ``to_str``
(``PyUnicode_AsUTF8AndSize``) — and returns the borrowed ``&str``'s
``len()``: a Rust ``&str`` IS its UTF-8 bytes, so ``s.len()`` IS the
UTF-8 byte length, and the "core" is literally one expression. The
mechanism is dried into the language instead of reimplemented; the
maintenance burden of an unsafe FFI walk of ``PyUnicode_KIND``/data with
surrogate-pair arithmetic (to save one cached materialization) is
exactly the hand-rolled complexity the repo refuses.

The cost model that buys (measured, and pinned by the wall cells in
tests/test_performance.py and the heartbeat cell in
tests/test_gil_release.py):

- **ASCII** (the serialized-JSON case, ``ensure_ascii=True``): compact
  ASCII data is its own UTF-8, so the borrow is a zero-copy alias and the
  call is O(1) with no allocation at all — versus the expression's
  alloc+memcpy every call.
- **Non-ASCII, first call on the object**: CPython materializes and
  CACHES the UTF-8 view on the ``str`` object (an internal cache, not a
  Python-visible ``bytes``; shared with every other str-in tors call on
  the same object), so the first call is O(n) — encode-parity in cost
  class, GIL-held like every str-in borrow (the ``finalize`` first-call
  class), with no Python-visible object to collect.
- **Non-ASCII, repeat calls on the same object**: O(1) — the cached view
  is borrowed zero-copy — strictly better than ``len(s.encode())``,
  which re-copies on every call.

Error parity comes free: a ``str`` holding lone surrogates cannot be
UTF-8-encoded, and ``PyUnicode_AsUTF8AndSize`` raises CPython's own
``UnicodeEncodeError`` (pyo3 propagates it from the ``&str`` extraction,
before any tors code runs) — the SAME exception object shape ``encode``
raises, verified attribute-for-attribute below (type, ``.encoding``,
``.reason``, ``.start``/``.end``, ``.object``). No tors error path
exists past that case.

Pins in this file: the oracle equality ``utf8_byte_len(s) ==
len(s.encode("utf-8"))`` over hypothesis-generated text (full Unicode
plus a targeted astral/mixing strategy) and a hand-computed boundary
battery (every UTF-8 sequence length 1-4, the boundary codepoints, and
the 1 MiB scale); the surrogate error parity; the argument contract
(exactly ``str``-accepted, ``bytes``/``bytearray``/``memoryview``/
``int`` -> ``TypeError``); the cache-behavior pin (identical result on
repeat calls — a semantic pin; the TIMING of the cache is CPython's
internal business, measured in the wall cells, not asserted here); and
the motivating TaskQ byte-cap gate spelled with ``utf8_byte_len``.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import utf8_byte_len

_MIB = 1024 * 1024

_E_ACUTE = "\u00e9"  # 2 UTF-8 bytes, 1 char
_COMBINING_ACUTE = "\u0301"  # 2 UTF-8 bytes, a combining mark
_ZWJ = "\u200d"  # 3 UTF-8 bytes, the emoji joiner
_CJK = "\u6771\u4eac"  # 2 chars, 6 UTF-8 bytes
_FAMILY = "\U0001f600"  # 4 UTF-8 bytes, 1 char (astral)
_MATH_BOLD_A = "\U0001d5d8"  # 4 UTF-8 bytes, astral non-emoji


# --- The golden boundary battery ------------------------------------------------------
#
# Every row: (text, expected byte length), hand-computed from the UTF-8
# sequence table (1 byte below U+0080, 2 to U+07FF, 3 to U+FFFF, 4 above),
# cross-checked against the oracle in the test body so a wrong hand
# computation fails loudly instead of laundering itself into the suite.

_BOUNDARY_LADDER: list[tuple[str, int]] = [
    ("", 0),
    ("a", 1),
    ("~", 1),
    ("\x00", 1),  # a real NUL codepoint: 1 byte (its JSON escape text is 6)
    ("\x7f", 1),  # U+007F: the last 1-byte codepoint
    ("\x80", 2),  # U+0080: the first 2-byte codepoint
    ("\u07ff", 2),  # the last 2-byte codepoint
    ("\u0800", 3),  # the first 3-byte codepoint
    ("\uffff", 3),  # the last 3-byte codepoint (a noncharacter, still legal)
    ("\U00010000", 4),  # the first 4-byte codepoint
    ("\U0010ffff", 4),  # the last codepoint
]

_CONTENT_ROWS: list[tuple[str, int]] = [
    ("caf\u00e9", 5),  # ASCII run then one 2-byte char
    ("\u6771\u4eac", 6),  # CJK: two 3-byte chars
    ("\U0001f600", 4),  # astral emoji: one 4-byte char
    ("e\u0301", 3),  # a combining mark: 1 + 2
    ("\U0001f468\u200d\U0001f469\u200d\U0001f467", 18),  # family: 4+3+4+3+4
    ("abc\u00e9\U0001f600def\u6771", 3 + 2 + 4 + 3 + 3),  # mixed scripts
    ("Torque caf\u00e9 \u6771\u4eac \U0001f600", 6 + 1 + 5 + 1 + 6 + 1 + 4),
    ("a" * 100, 100),  # a compact-ASCII run
]

_GOLDEN_CASES = _BOUNDARY_LADDER + _CONTENT_ROWS
_GOLDEN_IDS = [
    f"boundary-{text!r}-{expected}b" for text, expected in _BOUNDARY_LADDER
] + [
    "latin-1-accent",
    "cjk",
    "astral-emoji",
    "combining-mark",
    "zwj-family-sequence",
    "mixed-scripts",
    "sentence-with-mix",
    "ascii-run",
]


@pytest.mark.parametrize(("text", "expected"), _GOLDEN_CASES, ids=_GOLDEN_IDS)
def test_golden_boundary_battery(text: str, expected: int) -> None:
    """The fixed anchor: every row asserts the hand-computed byte length AND
    the oracle's agreement (the hand computation and ``len(s.encode())``
    disagreeing would fail here, not ship)."""
    got = utf8_byte_len(text)
    assert got == expected
    assert got == len(text.encode("utf-8"))


def test_one_mib_of_mixed_script_text_matches_the_oracle() -> None:
    """The 1 MiB scale (the issue's ladder top): a deterministic mixed-script
    corpus (ASCII prose, Latin-1 accents, CJK, astral emoji, a combining
    mark, and a NUL — every sequence length 1-4 plus the scan family's own
    NUL concern), unit-quantized to ~1 MiB the reference-corpus way. The
    oracle comparison is the whole pin at this scale (hand-computing a
    megabyte is silliness); the point is that the answer holds at a size
    where the encode being replaced allocates a full megabyte."""
    unit = (
        "Torque specifications, caf\u00e9 \u6771\u4eac \U0001f600 e\u0301\x00\n\n"
    )
    text = unit * (1 * _MIB // len(unit.encode("utf-8")))
    assert len(text.encode("utf-8")) >= 1 * _MIB - len(unit.encode("utf-8"))
    assert utf8_byte_len(text) == len(text.encode("utf-8"))


# --- The differential oracle over generated inputs -----------------------------------


# The targeted alphabet: every UTF-8 sequence length (1-4), the boundary
# codepoints, combining marks, the emoji joiner, astral non-emoji, and a
# NUL — the classes the broad st.text() draw hits only rarely (astral
# codepoints are a small fraction of the BMP+astral draw space).
_CLASS_CHARS = [
    "a",
    "Z",
    "0",
    " ",
    "\x00",
    "\x7f",
    "\x80",
    "\u07ff",
    "\u0800",
    "\uffff",
    _E_ACUTE,
    _COMBINING_ACUTE,
    _ZWJ,
    "\u6771",
    _FAMILY,
    _MATH_BOLD_A,
    "\U00010000",
    "\U0010ffff",
]

_class_text = st.lists(st.sampled_from(_CLASS_CHARS), min_size=0, max_size=80).map(
    "".join
)

_astral_text = st.text(
    alphabet=st.characters(min_codepoint=0x10000, max_codepoint=0x10FFFF),
    min_size=0,
    max_size=60,
)


class TestDifferentialOracle:
    @given(st.text())
    @settings(max_examples=400)
    def test_matches_len_encode_over_full_unicode(self, text: str) -> None:
        """The broad differential: every hypothesis-drawn surrogate-free
        Unicode string (any plane, any script, any control character) must
        answer exactly ``len(text.encode("utf-8"))``. A byte-counting bug of
        any kind (a codepoint counted instead of a byte, a sequence length
        off by one at a boundary) breaks this property; the boundary battery
        pins the individual rows it finds."""
        assert utf8_byte_len(text) == len(text.encode("utf-8"))

    @given(_class_text)
    @settings(max_examples=400)
    def test_matches_len_encode_over_the_targeted_classes(self, text: str) -> None:
        """The targeted differential: the boundary codepoints, astral
        codepoints, combining marks, the ZWJ, and NUL, densely mixed — the
        classes the broad draw reaches only by chance."""
        assert utf8_byte_len(text) == len(text.encode("utf-8"))

    @given(_astral_text)
    @settings(max_examples=200)
    def test_matches_len_encode_over_pure_astral_text(self, text: str) -> None:
        """The astral-heavy differential: every char 4 UTF-8 bytes, the
        regime where codepoint-count and byte-count diverge by exactly 4x
        and a char-unit implementation would fail maximally."""
        assert utf8_byte_len(text) == len(text.encode("utf-8"))
        if text:
            assert utf8_byte_len(text) == 4 * len(text)


# --- Surrogate error parity -----------------------------------------------------------


_SURROGATE_CASES = [
    "\ud800",  # a lone HIGH surrogate (the first)
    "\udbff",  # a lone HIGH surrogate (the last)
    "\udcff",  # a lone LOW surrogate (the last overall)
    "\udc80",  # a lone LOW surrogate
    "a\ud800b",  # a surrogate embedded in legal text
    "\ud83d\ude00",  # the surrogate-PAIR spelling of an emoji: in a str these
    # are two lone surrogates, NOT one astral char (chr() builds those), and
    # encode refuses them exactly the same way
]

_SURROGATE_IDS = [
    "lone-high-first",
    "lone-high-last",
    "lone-low-last",
    "lone-low",
    "embedded-in-legal-text",
    "surrogate-pair-spelling-is-not-astral",
]


@pytest.mark.parametrize("text", _SURROGATE_CASES, ids=_SURROGATE_IDS)
def test_lone_surrogates_raise_exactly_encodes_error(text: str) -> None:
    """Error parity, pinned to the attribute: a str holding lone surrogates
    cannot be UTF-8-encoded, and the function must refuse it with the SAME
    exception ``encode`` raises — not merely the same type. The error is
    CPython's own (``PyUnicode_AsUTF8AndSize`` raises it; pyo3 propagates it
    from the ``&str`` extraction before any tors code runs), so the parity
    is exact by construction; this pin exists so a future wrapper change
    (catching and re-raising, or a validation pass of our own) cannot
    silently narrow it to a same-type-different-attributes error."""
    with pytest.raises(UnicodeEncodeError) as tors_exc_info:
        utf8_byte_len(text)
    with pytest.raises(UnicodeEncodeError) as encode_exc_info:
        text.encode("utf-8")
    tors_err, encode_err = tors_exc_info.value, encode_exc_info.value
    assert type(tors_err) is type(encode_err) is UnicodeEncodeError
    assert tors_err.encoding == encode_err.encoding == "utf-8"
    assert tors_err.reason == encode_err.reason == "surrogates not allowed"
    assert (tors_err.start, tors_err.end) == (encode_err.start, encode_err.end)
    assert tors_err.object == encode_err.object == text


def test_the_surrogate_error_fires_before_any_len_is_computed() -> None:
    """The error path is the borrow's, not a post-check: it fires on the
    argument extraction (before any tors code, under the GIL), which is why
    there is no tors-side error construction to drift. The observable
    consequence pinned here: the exception's ``.object`` is the ORIGINAL
    str (encode's own behavior — the error names the input it refused), and
    the message positions name the surrogate's codepoint span, not a byte
    span of some intermediate copy."""
    text = "prefix \ud800 suffix"
    with pytest.raises(UnicodeEncodeError) as exc_info:
        utf8_byte_len(text)
    err = exc_info.value
    assert err.object is text
    assert text[err.start : err.end] == "\ud800"


# --- The argument contract -----------------------------------------------------------


class TestArgumentContract:
    @pytest.mark.parametrize(
        "not_a_str",
        [b"bytes", bytearray(b"bytes"), memoryview(b"bytes"), 123, None, ["a", "b"]],
        ids=["bytes", "bytearray", "memoryview", "int", "none", "list"],
    )
    def test_non_str_raises_type_error(self, not_a_str: object) -> None:
        """Exactly the str-in surface: ``bytes`` (the very bytes the function
        counts the length OF, but in the other direction), ``bytearray``,
        ``memoryview``, and non-buffer types all raise ``TypeError`` — the
        standard str-in borrow contract every str-argument tors function
        enforces via ``&str`` extraction."""
        with pytest.raises(TypeError):
            utf8_byte_len(not_a_str)  # type: ignore[arg-type]

    def test_the_return_is_a_genuine_int(self) -> None:
        """The marshalling shape: a bare ``int`` (never a ``bool`` or
        ``None``), the ``grapheme_count``/``word_count`` single-int class."""
        assert type(utf8_byte_len("abc")) is int
        assert type(utf8_byte_len("")) is int
        assert utf8_byte_len("abc") == 3

    def test_a_str_subclass_is_accepted_not_refused(self) -> None:
        """The isinstance contract: the extraction accepts a ``str``
        subclass (CPython's own coercion behavior for str APIs —
        ``"x".join`` and ``str.encode`` accept subclasses too), so the
        function measures the subclass instance's own value."""
        class Shout(str):
            pass

        assert utf8_byte_len(Shout("caf\u00e9")) == 5


# --- The cache-behavior pin -----------------------------------------------------------


def test_repeat_calls_on_the_same_object_answer_identically() -> None:
    """The semantic cache pin: two calls on the same object must answer the
    same int (and the answer must stay oracle-equal). What is deliberately
    NOT pinned: the timing. The UTF-8 view cache is a CPython-internal
    implementation detail — its presence and cost class are MEASURED (the
    wall cells in tests/test_performance.py race the cached and
    first-call lanes against the encode expression; the heartbeat cell in
    tests/test_gil_release.py pins the first-call materialization band),
    because a timing assertion here would pin CPython internals into the
    suite's contract and flake with them, where the semantic contract
    (same object, same answer) is ours and stable."""
    text = "Torque caf\u00e9 \u6771\u4eac \U0001f600" * 100
    first = utf8_byte_len(text)
    second = utf8_byte_len(text)
    third = utf8_byte_len(text)
    assert first == second == third == len(text.encode("utf-8"))


def test_a_fresh_equal_object_answers_the_same_as_a_cached_one() -> None:
    """The cache never changes the ANSWER, only the cost: a separately
    built (uncached) equal string must answer exactly what the cached one
    does — the cache is invisible in the value space by construction, and
    this pin keeps it that way."""
    # The cache never changes the ANSWER, only the cost: a separately
    # built (uncached) equal string must answer exactly what the cached one
    # does — the cache is invisible in the value space by construction, and
    # this pin keeps it that way. Both strings are built from a runtime
    # variable (compile-time literals would be constant-folded, and CPython
    # dedupes equal constants in one code object into a single shared
    # object — the first draft of this test tripped exactly that).
    unit = "caf\u00e9 \u6771\u4eac \U0001f600"
    cached = unit * 50
    fresh = unit * 49 + unit
    assert fresh is not cached
    utf8_byte_len(cached)  # materialize the cache on the first object
    assert utf8_byte_len(fresh) == utf8_byte_len(cached)


# --- The motivating TaskQ pattern, as an invariant ------------------------------------


# The terminal's result cap from the issue (backend/_terminal.py's
# MAX_RESULT_BYTES): the size the 64 KiB double-pass pays on every success.
_MAX_RESULT_BYTES = 64 * 1024


@pytest.mark.parametrize(
    "bytes_over_cap",
    [-1, 0, 1],
    ids=["one-under-the-cap", "exactly-the-cap", "one-over-the-cap"],
)
def test_the_taskq_byte_cap_gate_trips_where_the_encode_gate_trips(
    bytes_over_cap: int,
) -> None:
    """The motivating invariant: a byte-cap gate spelled with
    ``utf8_byte_len`` must accept and reject EXACTLY the payloads the
    ``len(s.encode("utf-8"))`` gate does, at the boundary itself. The
    payloads are mixed-script (ASCII + accents + CJK + astral), so the cap
    does not sit at a codepoint count the char-length would coincidentally
    reproduce — a 64 KiB byte cap over this alphabet rejects strings
    ``len(s)`` would have accepted, which is the whole reason the gate
    counts bytes."""
    unit = "key=caf\u00e9;\u6771\u4eac=\U0001f600;"  # 3+1+2+1... mixed, 26 bytes
    target = _MAX_RESULT_BYTES + bytes_over_cap
    # Trim or extend a repeated unit to hit the exact byte target: the unit
    # is 26 bytes, so tile it and patch the tail with ASCII to land exact.
    tiles, remainder = divmod(target, len(unit.encode("utf-8")))
    payload = unit * tiles + "k" * remainder
    assert len(payload.encode("utf-8")) == target
    assert utf8_byte_len(payload) == target
    assert (utf8_byte_len(payload) <= _MAX_RESULT_BYTES) == (
        len(payload.encode("utf-8")) <= _MAX_RESULT_BYTES
    )
    if bytes_over_cap <= 0:
        assert utf8_byte_len(payload) <= _MAX_RESULT_BYTES  # the gate accepts
    else:
        assert utf8_byte_len(payload) > _MAX_RESULT_BYTES  # the gate rejects


def test_the_scope_and_idempotency_key_pattern_counts_not_copies() -> None:
    """The enqueue-site pattern (client/_args.py: idempotency-key and scope
    byte caps on every enqueue): the same gate spelled over two
    independently-built payloads — the idempotency key (ASCII-dominant, the
    serialize-once case) and a scope string (mixed-script) — answering the
    same numbers the encode expression answers, without either expression
    allocating. A regression that answered codepoints (or chars) instead of
    bytes fails the scope leg's divergence pin, not the ASCII leg's
    accidental equality."""
    idempotency_key = f"taskq:v1:{'k' * 200}"
    scope = f"tenant=caf\u00e9;region=\u6771\u4eac;tag=\U0001f600;{'s' * 200}"
    for payload in (idempotency_key, scope):
        assert utf8_byte_len(payload) == len(payload.encode("utf-8"))
    # The divergence pin: the scope string's byte length exceeds its char
    # length (mixed scripts), so a char-counting regression cannot pass the
    # scope leg while passing the ASCII leg.
    assert utf8_byte_len(scope) > len(scope)
    assert utf8_byte_len(idempotency_key) == len(idempotency_key)
