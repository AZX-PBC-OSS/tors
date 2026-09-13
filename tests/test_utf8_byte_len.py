"""Contract gate for the byte-len family: ``tors.utf8_byte_len`` and
``tors.utf16_byte_len`` — Python ``len()`` converted to bytes without
building the bytes object.

The family is the maintainer's ask (a util to convert a UTF8/UTF16
python len() into bytes for the API — backend devs need byte caps for
storage and interop): two functions, one per encoding, each answering
``len(s.encode(...))`` for its codec without the copy that expression
allocates. ``utf8_byte_len`` is the storage-world twin (the byte caps in
front of UTF-8 stores); ``utf16_byte_len`` is the interop twin below.

The expression both replace is pure waste whenever only the COUNT is
wanted: ``len(s.encode(...))`` allocates a full ``bytes`` object,
measures it, and throws it away. The motivating pattern is a size cap in
front of a store or a wire (the issue's own framing): TaskQ spells it
twice — ``client/_args.py`` checks idempotency-key and scope byte caps
on every enqueue, and ``backend/_terminal.py`` re-encodes a serialized
result of up to 64 KiB (``MAX_RESULT_BYTES``) on every success, a
genuine double pass (the byte count existed inside the serializer's
output and was discarded by the ``.decode()`` that produced the
``str``). The general shape is wire limits, column limits, batch byte
budgets: any gate that asks "how many bytes is this?" without wanting
the bytes.

**Companion, not standalone**: this function ships in the scan family's
binding module (``src/py/scan.rs``, the ``contains_unescaped`` /
``find_unescaped`` surface) because it is the pinned companion of the
escape-parity scan (#50) — same module, same test/bench harness patterns
— and honest sizing says it would not stand alone: a short-string encode
is a few hundred nanoseconds, so the win only exists at large inputs or
hot paths where the copy is the cost (the 64 KiB terminal case is ~0.9 µs
of pure alloc+memcpy per success — the lane table's ASCII 64 KiB
expression cell in tests/test_performance.py).

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
- **Non-ASCII, first call on the object (a cold UTF-8 cache)**: CPython
  materializes and CACHES the UTF-8 view on the ``str`` object (an
  internal cache, not a Python-visible ``bytes``; filled by this borrow
  and by any earlier str-in tors call on the same object, read but never
  filled by ``encode`` — a prior ``len(s.encode())`` does not warm it),
  so the first call is O(n) — encode-parity in cost class, GIL-held like
  every str-in borrow (the ``finalize`` first-call class), with no
  Python-visible object to collect.
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

``tors.utf16_byte_len(s)``: the UTF-16 byte length — 2 bytes per BMP
codepoint, 4 per astral codepoint (a surrogate PAIR in the UTF-16
encoding; the pair spelling ``"\\ud83d\\ude00"`` inside a ``str`` is two
lone surrogates, not one astral codepoint, and is refused below). The
world that caps in these units: UTF-16 is the code-unit world of
JavaScript (``"😀".length === 2``), Java, Windows, and .NET — column
caps (``NVARCHAR``), wire caps, and interop size checks there are UTF-16
bytes, and the Python spelling ``len(s.encode("utf-16-le"))`` allocates
the whole 2n copy to count it.

The implementation is the same borrow-plus-arithmetic route as the
UTF-8 twin, and the arithmetic is derived, not table-driven: UTF-16
bytes = 2 * (#codepoints + #astral codepoints) — every codepoint is one
2-byte unit, astral codepoints are the 2-unit surrogate pairs (4 bytes =
2 + 2 more). Over the borrowed UTF-8 view both counts are byte classes:
#codepoints is the lead-byte count (every UTF-8 sequence has exactly one
lead byte; continuation bytes are exactly ``0x80-0xBF``), #astral is the
count of bytes >= ``0xF0`` (in valid UTF-8 those are exactly the 4-byte
lead bytes ``0xF0-0xF4`` — one per astral codepoint; the boundary
battery pins all four lead values). One pass over the UTF-8 bytes, no
allocation, no per-codepoint decoding; the derivation is proved against
a ``chars()``-based naive count crate-side and against the
``len(s.encode("utf-16-le"))`` oracle here, over the boundary battery,
an exhaustive short-string sweep, every reference corpus, and
hypothesis-generated text. The odd corner the derivation buys: with no
astral codepoints the answer is exactly ``2 * len(s)`` (Python ``len``,
the codepoint count) for ALL BMP text — CJK and combining marks
included, where the UTF-8 byte count diverges — and pure ASCII
additionally answers ``2 * len(s)`` in UTF-8 bytes.

Cache semantics are the UTF-8 twin's exactly (same borrow, same cache):
ASCII is a zero-copy alias; a non-ASCII object's first call — the
cold-cache case — materializes and caches the UTF-8 view (encode-parity
cost, GIL-held; ``encode`` reads that cache and never fills it), and
repeat calls borrow it zero-copy, with only the byte-class scan left to
pay.

The surrogate lane is REFUSAL PARITY with the replaced expression,
measured and pinned (a first-draft claim that the stdlib's utf-16-le
encode ACCEPTS lone surrogates was wrong — the battery caught it before
anything shipped): the STRICT ``encode("utf-16-le")`` — the codec of the
exact expression this function replaces — refuses lone surrogates
exactly like ``encode("utf-8")`` does ("surrogates not allowed"), so
tors refuses the same strings the expression itself refuses. The two
honest observable differences, both pinned below: the tors error is the
str-in borrow's own — ``.encoding == "utf-8"`` (the borrow materializes
the object's UTF-8 view before any arithmetic runs, and that is the
step that fails; the expression's error says ``"utf-16-le"``, and on a
multi-surrogate run it reports only the first unit where the
utf-8-flavored error names the whole run) — and the stdlib's
``errors="surrogatepass"`` mode WOULD encode lone surrogates (one
2-byte unit each in utf-16-le; the 3-byte WTF-8 spelling in utf-8), an
escape hatch tors deliberately does not offer: the crate-wide str-in
contract (the ``finalize``/``fence``/``search`` families' lane — every
str-argument tors function needs the UTF-8 view, so none has a
surrogate mode).

Pins in this file: the oracle equality ``utf8_byte_len(s) ==
len(s.encode("utf-8"))`` over hypothesis-generated text (full Unicode
plus a targeted astral/mixing strategy) and a hand-computed boundary
battery (every UTF-8 sequence length 1-4, the boundary codepoints, and
the 1 MiB scale); the surrogate error parity; the argument contract
(exactly ``str``-accepted, ``bytes``/``bytearray``/``memoryview``/
``int`` -> ``TypeError``); the cache-behavior pin (identical result on
repeat calls — a semantic pin; the TIMING of the cache is CPython's
internal business, measured in the wall cells, not asserted here); and
the motivating TaskQ byte-cap gate spelled with ``utf8_byte_len``. The
``utf16_byte_len`` pins mirror every one: the
``len(s.encode("utf-16-le"))`` oracle over the same strategy shapes plus
an exhaustive boundary-alphabet sweep and every reference corpus; the
hand-computed boundary battery including all four 4-byte lead values;
the DERIVATION identity (``2 * (len(s) + #astral)``) as a direct pin;
the evenness of every UTF-16 byte length; the surrogate lane's REFUSAL
PARITY (the borrow's ``utf-8``-flavored error pinned against the
replaced expression's own refusal — same reason, different codec label
and span spelling — against ``utf8_byte_len``'s identical error, and
against the ``surrogatepass`` mode tors does not offer); the same
argument contract; the same cache pins; the BMP/ASCII/
astral corners of the arithmetic; and the motivating interop cap gate
spelled with ``utf16_byte_len``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reference import compat, crlf, decomposed, entities, prose
from tors import utf8_byte_len, utf16_byte_len

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
_GOLDEN_IDS = [f"boundary-{text!r}-{expected}b" for text, expected in _BOUNDARY_LADDER] + [
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
    unit = "Torque specifications, caf\u00e9 \u6771\u4eac \U0001f600 e\u0301\x00\n\n"
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

_class_text = st.lists(st.sampled_from(_CLASS_CHARS), min_size=0, max_size=80).map("".join)

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


# ============================================================================
# utf16_byte_len: the interop twin (the code-unit world's byte count)
# ============================================================================
#
# Sections mirror the utf8 pins one for one: the golden boundary battery
# (hand-computed rows cross-checked against the oracle so a wrong hand sum
# fails loudly), the differential oracle over generated text plus the
# derivation identity, the exhaustive boundary-alphabet sweep, the reference
# corpora, the surrogate REFUSAL-PARITY pin (against the replaced
# expression's own refusal, against utf8_byte_len's identical error, and
# against the surrogatepass mode tors does not offer), the argument
# contract, the cache pins, the arithmetic corners, and the motivating
# interop cap gate.


_UTF16_BOUNDARY_LADDER: list[tuple[str, int]] = [
    ("", 0),
    ("a", 2),
    ("~", 2),
    ("\x00", 2),  # a real NUL codepoint: one BMP unit
    ("\x7f", 2),
    ("\x80", 2),  # U+0080: still one BMP unit — the UTF-8 2-byte class
    ("\u07ff", 2),
    ("\u0800", 2),  # the UTF-8 3-byte class begins; UTF-16 still one unit
    ("\uffff", 2),  # the last BMP codepoint (a noncharacter, still legal)
    ("\U00010000", 4),  # the first astral codepoint: a surrogate pair
    ("\U00040000", 4),  # 0xF1 lead: pins the class against a 0xF0-only bug
    ("\U00080000", 4),  # 0xF2 lead
    ("\U000c0000", 4),  # 0xF3 lead (a noncharacter, still legal)
    ("\U0010ffff", 4),  # 0xF4 lead: the last codepoint
]

_UTF16_CONTENT_ROWS: list[tuple[str, int]] = [
    ("caf\u00e9", 8),  # four BMP codepoints
    ("\u6771\u4eac", 4),  # CJK: two units — where UTF-8 answers 6
    ("\U0001f600", 4),  # astral emoji: one surrogate pair
    ("e\u0301", 4),  # a combining mark: two BMP units
    ("\U0001f468\u200d\U0001f469\u200d\U0001f467", 16),  # 3 pairs + 2 ZWJ units
    ("abc\u00e9\U0001f600def\u6771", 20),  # 8 BMP units + 1 pair
    ("Torque caf\u00e9 \u6771\u4eac \U0001f600", 34),  # 15 BMP + 1 pair
    ("a" * 100, 200),  # a compact-ASCII run
]

_UTF16_GOLDEN_CASES = _UTF16_BOUNDARY_LADDER + _UTF16_CONTENT_ROWS
_UTF16_GOLDEN_IDS = [
    f"boundary-{text!r}-{expected}b" for text, expected in _UTF16_BOUNDARY_LADDER
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


@pytest.mark.parametrize(("text", "expected"), _UTF16_GOLDEN_CASES, ids=_UTF16_GOLDEN_IDS)
def test_utf16_golden_boundary_battery(text: str, expected: int) -> None:
    """The fixed anchor, utf16 leg: every row asserts the hand-computed
    UTF-16 byte length AND the oracle's agreement (a wrong hand computation
    fails here, not ship). The four 0xF0-0xF4 lead rows exist because the
    implementation counts astral codepoints as the bytes >= 0xF0 over the
    UTF-8 view — a predicate that caught only 0xF0 would answer 2 for
    U+40000..U+10FFFF and fail exactly here."""
    got = utf16_byte_len(text)
    assert got == expected
    assert got == len(text.encode("utf-16-le"))


def test_utf16_one_mib_of_astral_bearing_text_matches_the_oracle() -> None:
    """The 1 MiB scale, utf16 leg: a deterministic mixed corpus carrying
    every UTF-8 sequence length 1-4 including two astral codepoints with
    DIFFERENT lead bytes (0xF0 and 0xF1), unit-quantized to ~1 MiB the
    reference-corpus way. The oracle comparison is the whole pin at this
    scale (hand-computing a megabyte is silliness); the point is that the
    answer holds at a size where the encode being replaced allocates the
    whole UTF-16 copy of it, two bytes per codepoint."""
    unit = "Torque specifications, caf\u00e9 \u6771\u4eac \U0001f600 \U00040000 e\u0301\x00\n\n"
    text = unit * (1 * _MIB // len(unit.encode("utf-8")))
    assert len(text.encode("utf-8")) >= 1 * _MIB - len(unit.encode("utf-8"))
    assert utf16_byte_len(text) == len(text.encode("utf-16-le"))


def _utf16_oracle(text: str) -> int:
    return len(text.encode("utf-16-le"))


def _astral_count(text: str) -> int:
    """The derivation's second count, spelled the naive way: one pass over
    codepoints. This is the Python-readable form of the crate-side
    ``chars()`` oracle the arithmetic is proved against."""
    return sum(1 for ch in text if ord(ch) > 0xFFFF)


# The targeted alphabet: every UTF-8 sequence length (1-4), the boundary
# codepoints, ALL FOUR 4-byte lead values, combining marks, the emoji
# joiner, astral non-emoji, and a NUL.
_UTF16_CLASS_CHARS = [
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
    "\U00040000",
    "\U00080000",
    "\U000c0000",
    "\U0010ffff",
]

_utf16_class_text = st.lists(st.sampled_from(_UTF16_CLASS_CHARS), min_size=0, max_size=80).map(
    "".join
)

_utf16_astral_text = st.text(
    alphabet=st.characters(min_codepoint=0x10000, max_codepoint=0x10FFFF),
    min_size=0,
    max_size=60,
)


class TestUtf16DifferentialOracle:
    @given(st.text())
    @settings(max_examples=400)
    def test_matches_len_encode_over_full_unicode(self, text: str) -> None:
        """The broad differential, utf16 leg: every hypothesis-drawn
        surrogate-free Unicode string must answer exactly
        ``len(text.encode("utf-16-le"))``, and the DERIVATION identity
        (2 bytes per codepoint plus 2 more per astral codepoint) must hold
        alongside it — a char-counting regression, a pair-counting miss, or
        an off-by-one at any plane boundary breaks one of the two. Every
        UTF-16 byte length is even, pinned with the parity corner the cap
        gate below builds on."""
        assert utf16_byte_len(text) == _utf16_oracle(text)
        assert utf16_byte_len(text) == 2 * (len(text) + _astral_count(text))
        assert utf16_byte_len(text) % 2 == 0

    @given(_utf16_class_text)
    @settings(max_examples=400)
    def test_matches_len_encode_over_the_targeted_classes(self, text: str) -> None:
        """The targeted differential: the boundary codepoints, all four
        4-byte lead values densely mixed with every shorter class, the
        combining marks, the ZWJ, and NUL — the classes the broad draw
        reaches only by chance."""
        assert utf16_byte_len(text) == _utf16_oracle(text)
        assert utf16_byte_len(text) == 2 * (len(text) + _astral_count(text))
        assert utf16_byte_len(text) % 2 == 0

    @given(_utf16_astral_text)
    @settings(max_examples=200)
    def test_matches_len_encode_over_pure_astral_text(self, text: str) -> None:
        """The astral-heavy differential: every codepoint a surrogate pair,
        the regime where the UTF-16 answer is exactly 4 bytes per Python
        ``len()`` unit and a BMP-unit implementation would fail
        maximally."""
        assert utf16_byte_len(text) == _utf16_oracle(text)
        assert utf16_byte_len(text) == 2 * (len(text) + _astral_count(text))
        if text:
            assert utf16_byte_len(text) == 4 * len(text)


def test_utf16_exhaustive_sweep_over_the_boundary_alphabet() -> None:
    """The brute-force sweep: EVERY string of length 1-3 over the boundary
    alphabet (21 chars -> 9,723 cases), each answered by the oracle AND the
    derivation identity. Both sides of the identity are additive over
    concatenation (UTF-16 bytes sum per codepoint; the byte-class counts
    sum per codepoint), so a counterexample of any length would already
    show at length <= 3 — the sweep is the exhaustive proof, not a
    sample."""
    for length in (1, 2, 3):
        for combo in itertools.product(_UTF16_CLASS_CHARS, repeat=length):
            text = "".join(combo)
            got = utf16_byte_len(text)
            assert got == _utf16_oracle(text), f"sweep len={length} {text!r}"
            assert got == 2 * (len(text) + _astral_count(text)), f"identity {text!r}"


@pytest.mark.parametrize(
    "builder",
    [prose, decomposed, compat, crlf, entities],
    ids=["prose", "decomposed", "compat", "crlf", "entities"],
)
def test_utf16_reference_corpora_match_the_oracle_and_the_bmp_corner(
    builder: Callable[[int], str],
) -> None:
    """Every reference corpus, utf16 leg: oracle equality at scale, plus
    the BMP corner — the corpora are BMP-only, so each must answer exactly
    ``2 * len(corpus)`` (the no-astral arithmetic), which also guards the
    corpora's own BMP-ness loudly: a future corpus edit that introduces an
    astral codepoint trips this pin and forces the wall-cell corpus story
    to be revisited with it."""
    corpus = builder(256 * 1024)
    assert utf16_byte_len(corpus) == _utf16_oracle(corpus)
    assert utf16_byte_len(corpus) == 2 * len(corpus)


# --- utf16: the surrogate REFUSAL PARITY with the replaced expression -----------


_UTF16_SURROGATE_CASES = [
    "\ud800",  # a lone HIGH surrogate (the first)
    "\udbff",  # a lone HIGH surrogate (the last)
    "\udc80",  # a lone LOW surrogate
    "\udcff",  # a lone LOW surrogate (the last overall)
    "a\ud800b",  # a surrogate embedded in legal text
    "\ud83d\ude00",  # the surrogate-PAIR spelling of an emoji: in a str these
    # are two lone surrogates, NOT one astral char (chr() builds those)
    "\ud800abc",  # at the start
    "abc\ud800",  # at the end
    "ab\ud800cd",  # in the middle
]

_UTF16_SURROGATE_IDS = [
    "lone-high-first",
    "lone-high-last",
    "lone-low",
    "lone-low-last",
    "embedded-in-legal-text",
    "surrogate-pair-spelling-is-not-astral",
    "at-the-start",
    "at-the-end",
    "in-the-middle",
]


@pytest.mark.parametrize("text", _UTF16_SURROGATE_CASES, ids=_UTF16_SURROGATE_IDS)
def test_utf16_lone_surrogates_raise_where_the_replaced_expression_raises(text: str) -> None:
    """The surrogate lane, pinned as REFUSAL PARITY (measured; a
    first-draft version of this pin asserted the stdlib's utf-16-le
    encode ACCEPTS lone surrogates and failed — the strict codec refuses
    them): ``len(s.encode("utf-16-le"))`` — the exact expression this
    function replaces — raises ``UnicodeEncodeError`` on the same
    strings ``utf16_byte_len`` refuses, same reason ("surrogates not
    allowed"), the utf-8 codec's own policy. tors's error is the str-in
    borrow's (every str-argument tors function's lane), so
    ``.encoding`` is ``"utf-8"`` — the flavor of the step that actually
    fails, materializing the UTF-8 view — while the expression's error
    says ``"utf-16-le"``; on a multi-surrogate run the utf-16-flavored
    error reports only the first unit, where the borrow's names the
    whole run. The stdlib's ``surrogatepass`` mode WOULD encode the
    string (one 2-byte unit per lone surrogate) — the escape hatch tors
    deliberately does not offer, pinned for the record."""
    with pytest.raises(UnicodeEncodeError) as exc_info:
        utf16_byte_len(text)
    err = exc_info.value
    assert type(err) is UnicodeEncodeError
    assert err.encoding == "utf-8"  # the str-in borrow's error, the utf-8 flavor
    assert err.reason == "surrogates not allowed"
    assert err.object is text
    span = text[err.start : err.end]
    assert span and all(0xD800 <= ord(ch) <= 0xDFFF for ch in span)  # a surrogate run
    # The utf-8 twin raises the IDENTICAL error on the same string: one
    # shared lane (the borrow), pinned attribute-for-attribute.
    with pytest.raises(UnicodeEncodeError) as utf8_exc_info:
        utf8_byte_len(text)
    twin = utf8_exc_info.value
    assert (twin.encoding, twin.reason, twin.start, twin.end, twin.object) == (
        err.encoding,
        err.reason,
        err.start,
        err.end,
        err.object,
    )
    # The replaced expression's own refusal, same string: the strict
    # utf-16-le codec raises too — parity in the refusal DECISION, the
    # codec label (and, on surrogate runs, the span spelling) the only
    # flavor differences.
    with pytest.raises(UnicodeEncodeError) as stdlib_exc_info:
        text.encode("utf-16-le")
    stdlib_err = stdlib_exc_info.value
    assert stdlib_err.encoding == "utf-16-le"
    assert stdlib_err.reason == err.reason
    assert stdlib_err.object is text
    # The escape hatch tors does not offer, pinned: surrogatepass encodes
    # every codepoint — lone surrogates included — as one unit.
    raw = text.encode("utf-16-le", "surrogatepass")
    assert len(raw) == 2 * len(text)


def test_the_utf16_surrogate_error_is_the_borrow_lane_not_a_utf16_error() -> None:
    """The error-path contract, stated once on its own: the exception is
    the crate-wide str-in borrow's (the ``finalize``/``fence``/``search``
    families' lane — pyo3's ``&str`` extraction raising CPython's
    ``UnicodeEncodeError`` before any tors code runs), NOT a tors-side
    utf-16 validation. The observable consequences pinned: ``.encoding``
    is ``"utf-8"`` and the reported span names the surrogate's CODEPOINT
    position in the original str — exactly the utf-8 encode's own error
    on the same string (the borrow's failure IS a utf-8 materialization
    failure), while the replaced expression's utf-16-le error carries
    its own codec label; a future wrapper change (catching and
    re-raising, or a validation pass of our own with utf-16-flavored
    attributes) cannot silently narrow the lane."""
    text = "prefix \ud800 suffix"
    with pytest.raises(UnicodeEncodeError) as exc_info:
        utf16_byte_len(text)
    err = exc_info.value
    assert err.encoding == "utf-8"
    assert err.object is text
    assert text[err.start : err.end] == "\ud800"
    # The borrow's error is the utf-8 encode's own, attribute-for-
    # attribute (same reason, same span) — the step that fails is the
    # utf-8 materialization, so CPython reports it in utf-8's flavor.
    with pytest.raises(UnicodeEncodeError) as utf8_exc_info:
        text.encode("utf-8")
    utf8_err = utf8_exc_info.value
    assert (utf8_err.encoding, utf8_err.reason, utf8_err.start, utf8_err.end) == (
        err.encoding,
        err.reason,
        err.start,
        err.end,
    )
    # The replaced expression's refusal, same string: same reason and
    # (single surrogate) the same span, its own codec label the one
    # difference; and the surrogatepass mode below is the acceptance
    # path tors does not offer.
    with pytest.raises(UnicodeEncodeError) as stdlib_exc_info:
        text.encode("utf-16-le")
    stdlib_err = stdlib_exc_info.value
    assert stdlib_err.encoding == "utf-16-le"
    assert stdlib_err.reason == err.reason
    assert (stdlib_err.start, stdlib_err.end) == (err.start, err.end)
    assert len(text.encode("utf-16-le", "surrogatepass")) == 2 * len(text)


# --- utf16: the argument contract (the utf8 twin's exactly) --------------------


class TestUtf16ArgumentContract:
    @pytest.mark.parametrize(
        "not_a_str",
        [b"bytes", bytearray(b"bytes"), memoryview(b"bytes"), 123, None, ["a", "b"]],
        ids=["bytes", "bytearray", "memoryview", "int", "none", "list"],
    )
    def test_non_str_raises_type_error(self, not_a_str: object) -> None:
        """``utf8_byte_len``'s exact argument contract: the same ``&str``
        extraction, so the same refusals — ``bytes`` (which IS a UTF-16
        spelling on the wire, but this function counts a ``str``, not
        decodes one — the decode side is ``decode_utf16``), ``bytearray``,
        ``memoryview``, and non-buffer types all raise ``TypeError``."""
        with pytest.raises(TypeError):
            utf16_byte_len(not_a_str)  # type: ignore[arg-type]

    def test_the_return_is_a_genuine_int(self) -> None:
        """The marshalling shape: a bare ``int`` (never a ``bool`` or
        ``None``), the ``grapheme_count``/``word_count`` single-int class
        the utf8 twin returns."""
        assert type(utf16_byte_len("abc")) is int
        assert type(utf16_byte_len("")) is int
        assert utf16_byte_len("abc") == 6

    def test_a_str_subclass_is_accepted_not_refused(self) -> None:
        """The isinstance contract, the utf8 twin's: the extraction accepts
        a ``str`` subclass and measures the subclass instance's own
        value."""

        class Shout(str):
            pass

        assert utf16_byte_len(Shout("caf\u00e9")) == 8


# --- utf16: the cache pins (semantic only, the twin's design) ------------------


def test_utf16_repeat_calls_on_the_same_object_answer_identically() -> None:
    """The semantic cache pin, the utf8 twin's design exactly: same
    object, same int, oracle-equal; the TIMING of the UTF-8 view cache is
    CPython's internal business (measured in the wall cells, never
    asserted here)."""
    text = "Torque caf\u00e9 \u6771\u4eac \U0001f600" * 100
    first = utf16_byte_len(text)
    second = utf16_byte_len(text)
    third = utf16_byte_len(text)
    assert first == second == third == _utf16_oracle(text)


def test_utf16_a_fresh_equal_object_answers_the_same_as_a_cached_one() -> None:
    """The cache never changes the ANSWER, only the cost — the utf8 twin's
    fresh-object pin, runtime-built the same way (compile-time literals
    would be constant-folded, and CPython dedupes equal constants in one
    code object into a single shared object)."""
    unit = "caf\u00e9 \u6771\u4eac \U0001f600"
    cached = unit * 50
    fresh = unit * 49 + unit
    assert fresh is not cached
    utf16_byte_len(cached)  # materialize the UTF-8 cache on the first object
    assert utf16_byte_len(fresh) == utf16_byte_len(cached)


# --- utf16: the arithmetic corners ---------------------------------------------


def test_bmp_only_text_answers_two_bytes_per_codepoint() -> None:
    """The no-astral corner: ``2 * len(s)`` for ALL BMP text — the count
    the code-unit world caps on. CJK and combining marks are the rows
    where the UTF-8 byte count diverges (3 UTF-8 bytes, still 2 UTF-16
    bytes), pinned as the direct contrast: a UTF-8-unit regression fails
    here, not on the ASCII rows where the two coincide."""
    for text in ("caf\u00e9", "\u6771\u4eac", "e\u0301", "\uffff", "Torque caf\u00e9 \u6771\u4eac"):
        assert utf16_byte_len(text) == 2 * len(text)
        assert utf16_byte_len(text) == _utf16_oracle(text)
    # the divergence rows: 3-byte UTF-8 codepoints are still one UTF-16 unit
    assert utf16_byte_len("\u6771\u4eac") == 4
    assert utf8_byte_len("\u6771\u4eac") == 6


def test_pure_ascii_answers_two_bytes_per_byte() -> None:
    """The ASCII corner: one BMP unit per byte, so the UTF-16 answer is
    double the UTF-8 answer — the relation the JS world's ``s.length``
    coincides with for ASCII, and the lane the wall cells race."""
    text = "k" * 1000
    assert utf16_byte_len(text) == 2000
    assert utf16_byte_len(text) == 2 * utf8_byte_len(text)
    assert utf16_byte_len(text) == 2 * len(text)


def test_astral_codepoints_answer_the_surrogate_pair_four_bytes() -> None:
    """The astral corner: 4 bytes per astral codepoint — the surrogate
    pair — for every 4-byte lead value, and the pair arithmetic at
    ZWJ-sequence density."""
    for ch in ("\U0001f600", "\U00010000", "\U00040000", "\U00080000", "\U000c0000", "\U0010ffff"):
        assert utf16_byte_len(ch) == 4
        assert utf16_byte_len(ch) == _utf16_oracle(ch)
    family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
    assert utf16_byte_len(family) == 3 * 4 + 2 * 2
    assert utf16_byte_len(family * 1000) == 16000


# --- utf16: the motivating interop cap gate, as an invariant --------------------


# A UTF-16 wire cap in the code-unit world's own units: 140 units = 280
# bytes (the JS-side column/tweet-class cap; SQL Server NVARCHAR(140)
# counts the same units).
_UTF16_CAP_BYTES = 280


@pytest.mark.parametrize(
    "units_over_cap",
    [-1, 0, 1],
    ids=["one-unit-under-the-cap", "exactly-the-cap", "one-unit-over-the-cap"],
)
def test_the_utf16_cap_gate_trips_where_the_encode_gate_trips(units_over_cap: int) -> None:
    """The motivating invariant, utf16 leg: a code-unit cap gate spelled
    with ``utf16_byte_len`` must accept and reject EXACTLY the payloads
    the ``len(s.encode("utf-16-le"))`` gate does, at the boundary itself.
    The payloads are mixed-script (ASCII + accents + CJK + one astral
    emoji per unit), so the cap does not sit at a codepoint count the
    ``2 * len(s)`` arithmetic would coincidentally reproduce. The unit is
    26 UTF-16 bytes (12 codepoints, one of them astral); the target is
    tiled in units and patched with ASCII to land EXACT, and every
    UTF-16 total is even — hence the one-UNIT steps, not one-byte steps:
    an odd byte target is unreachable by any str, which the parity at
    even targets pins as the evenness corner."""
    unit = "k=caf\u00e9;\u6771\u4eac=\U0001f600;"  # 12 codepoints, 1 astral: 26 bytes
    assert _utf16_oracle(unit) == 26
    target = _UTF16_CAP_BYTES + 2 * units_over_cap
    tiles, remainder = divmod(target, 26)
    assert remainder % 2 == 0  # ASCII patches add 2 UTF-16 bytes per char
    payload = unit * tiles + "k" * (remainder // 2)
    assert _utf16_oracle(payload) == target
    assert utf16_byte_len(payload) == target
    assert (utf16_byte_len(payload) <= _UTF16_CAP_BYTES) == (
        _utf16_oracle(payload) <= _UTF16_CAP_BYTES
    )
    if units_over_cap <= 0:
        assert utf16_byte_len(payload) <= _UTF16_CAP_BYTES  # the gate accepts
    else:
        assert utf16_byte_len(payload) > _UTF16_CAP_BYTES  # the gate rejects


# --- utf16: BOM exclusion, byte-order parity, and the JS-unit corner -----
#
# The BOM exclusion sentence, asserted: `utf16_byte_len` counts UTF-16
# units only — no byte-order mark. `s.encode("utf-16")` (the BOM form)
# answers the le length plus the 2-byte BOM; `s.encode("utf-16-be")`
# answers the same bytes as le in the other order.


def test_utf16_excludes_the_bom_that_encode_utf16_adds() -> None:
    """BOM exclusion, pinned: `encode("utf-16")` writes a 2-byte BOM the
    le/be forms never do, and `utf16_byte_len` counts units only — so it
    equals the le length and trails the BOM form by exactly 2."""
    for text in ("", "a", "café", "東京", "😀", "a😀b"):
        assert utf16_byte_len(text) == len(text.encode("utf-16-le"))
        assert len(text.encode("utf-16")) == len(text.encode("utf-16-le")) + 2
        assert utf16_byte_len(text) == len(text.encode("utf-16")) - 2


def test_utf16_be_parity_matches_the_le_oracle() -> None:
    """Byte-order parity: le and be differ only in byte order, never in
    length — so the le oracle this function pins also equals the be
    oracle on every input, surrogate-free or otherwise countable."""
    for text in ("", "a", "café", "東京", "é", "😀", "a😀b東京"):
        assert len(text.encode("utf-16-be")) == len(text.encode("utf-16-le"))
        assert utf16_byte_len(text) == len(text.encode("utf-16-be"))


def test_js_string_length_is_half_the_utf16_byte_length() -> None:
    """The JS-unit corner: `String.prototype.length` counts UTF-16 units,
    so `"😀".length === 2` in JS is `utf16_byte_len("😀") == 4` here —
    two units, one per surrogate. BMP text is one unit per codepoint
    (`len(s)` units); astral text is two per codepoint."""
    assert utf16_byte_len("😀") == 4
    assert len("😀") == 1  # one codepoint, two UTF-16 units
    assert utf16_byte_len("😀") == 2 * 2
    assert utf16_byte_len("abc") == 2 * len("abc")
    assert utf16_byte_len("東京") == 2 * len("東京")


def test_surrogatepass_decoded_input_is_refused() -> None:
    """The surrogatepass-decode input lane: WTF-8 bytes decoded with
    `errors="surrogatepass"` produce a `str` holding lone surrogates —
    exactly the input the crate-wide str-in contract refuses. Both twins
    raise the borrow's `UnicodeEncodeError` there (refusal parity), and
    the stdlib's surrogatepass ENCODE would accept it (the mode tors
    does not offer)."""
    text = b"\xed\xa0\x80".decode("utf-8", "surrogatepass")
    assert len(text) == 1 and 0xD800 <= ord(text) <= 0xDFFF
    with pytest.raises(UnicodeEncodeError):
        utf8_byte_len(text)
    with pytest.raises(UnicodeEncodeError) as exc_info:
        utf16_byte_len(text)
    assert exc_info.value.encoding == "utf-8"
    # The escape hatch tors does not offer, pinned on this input too:
    assert len(text.encode("utf-16-le", "surrogatepass")) == 2 * len(text)


def test_utf16_exhaustive_sweep_count_and_alphabet_contract() -> None:
    """Sweep-count unification pin: the Python sweep runs 21 alphabet
    entries (extra ASCII representatives `Z`, `0`, ` ` plus the Rust
    sweep's 17) over lengths 1–3 in codepoints — 21 + 21² + 21³ = 9723
    cases — where the Rust sweep's 17-entry single-codepoint alphabet
    gives 17 + 17² + 17³ = 5219. Same theorem (additivity over
    concatenation of valid-UTF-8 single-codepoint entries), denser ASCII
    sampling here; both alphabets are length-in-codepoints and
    valid-UTF-8-only by construction."""
    assert len(_UTF16_CLASS_CHARS) == 21
    assert all(len(ch) == 1 for ch in _UTF16_CLASS_CHARS)
    total = sum(len(_UTF16_CLASS_CHARS) ** n for n in (1, 2, 3))
    assert total == 9723
    # Every entry encodes cleanly (valid UTF-8/UTF-16, no surrogates):
    for ch in _UTF16_CLASS_CHARS:
        ch.encode("utf-8")
        ch.encode("utf-16-le")


def test_overflow_contract_documents_the_32bit_boundary() -> None:
    """The 32-bit overflow contract, stated without a gigabyte: on 64-bit
    (this runner) `2 * (codepoints + astral)` cannot overflow `usize`,
    so large-but-reasonable inputs — including 12 MiB of astral text —
    answer exactly. Past ~1 GiB of astral-dense text on a 32-bit target
    the checked arithmetic raises `OverflowError` instead of wrapping
    (crate-side `utf16_combine_counts` injectable unit plus the
    32-bit-only tier-3 leg in `src/scan_impl.rs`;
    the Python wrapper maps the failure to `OverflowError`)."""
    import struct

    assert struct.calcsize("P") == 8  # documents this leg ran 64-bit
    astral = "😀" * (3 * 1024 * 1024)  # 12 MiB of UTF-8, astral-dense
    assert utf16_byte_len(astral) == 4 * len(astral) == len(astral.encode("utf-16-le"))


def test_interpreter_matrix_note_gil_and_pypy() -> None:
    """Interpreter-matrix note, pinned as behavior: the detach is a
    static property of the wrapper (`py.detach` around the core — see
    the static presence pin below), not a timing claim. On free-threaded
    CPython the GIL-held residue (the borrow's first-call
    materialization) still serializes on the object's cache state, and
    on PyPy the UTF-8-view cache internals differ — so only the semantic
    pins (same object, same answer) are portable; wall cells are
    CPython-3.12-calibration-box numbers, re-measure per leg."""
    import sys

    _ = sys.version_info  # the note travels with the suite, not a branch
    text = "café 東京 😀" * 100
    assert utf16_byte_len(text) == len(text.encode("utf-16-le"))
    assert utf8_byte_len(text) == len(text.encode("utf-8"))


def test_detach_is_statically_present_in_both_wrappers() -> None:
    """Static detach-presence pin: the heartbeat cells admit they cannot
    catch a detach removal (a removed detach still passes the 100 ms
    ceiling at these sizes — the wall just moves under the GIL), so this
    cell pins the mechanism statically instead: both `utf8_byte_len`
    and `utf16_byte_len` bodies in `src/py/scan.rs` must route the core
    through `py.detach`. The 1.0 ms scan-band ceiling in
    `tests/test_performance.py` stays as the dynamic backstop (a
    scalar-loop regression blows it by 3x); this is the static one.

    Airtight spelling (MEDIUM-3): each function body is parsed from its
    `pub fn {name}` to the next `pub fn` (or EOF — the function's full
    extent, not a fixed window), and must contain the exact call string
    `py.detach(|| scan_impl::{name}` — detach presence AND core routing
    together, so neither a removed detach nor a detach around other
    work passes."""
    import pathlib

    src = pathlib.Path(__file__).parent.parent.joinpath("src", "py", "scan.rs").read_text()
    for name in ("utf8_byte_len", "utf16_byte_len"):
        start = src.index(f"pub fn {name}")
        next_fn = src.find("\npub fn ", start + 1)
        body = src[start:] if next_fn == -1 else src[start:next_fn]
        # The function extent must close: a top-level `}` ends the body
        # (guards against the slice running past the function when the
        # next-pub-fn search misses).
        assert "\n}" in body, f"{name} body has no closing brace"
        assert (
            f"py.detach(|| scan_impl::{name}" in body
        ), f"{name} lost its py.detach-routed core call"


def test_cheap_utf16_differential_against_encode_utf16_count() -> None:
    """Cheap utf16 differential fuzz (no hypothesis profile needed): the
    `str.encode_utf16()` iterator count doubled must equal the byte
    length on a fixed pseudo-random corpus — an independent Rust-side
    spelling (`encode_utf16().count()`, per-codepoint UTF-16 units)
    cross-checked against the byte-class arithmetic without building
    the 2n copy. Deterministic (LCG over the boundary codepoints), 2000
    cases, mean length ~20."""
    boundary = [
        0x61,
        0x7F,
        0x80,
        0x7FF,
        0x800,
        0xFFFF,
        0x10000,
        0x40000,
        0x80000,
        0xC0000,
        0x10FFFF,
    ]
    state = 0x12345678
    for _ in range(2000):
        n = state % 41
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        chars = []
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            chars.append(chr(boundary[state % len(boundary)]))
        text = "".join(chars)
        assert utf16_byte_len(text) == len(text.encode("utf-16-le"))
        assert utf16_byte_len(text) % 2 == 0
