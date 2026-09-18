"""Contract gate for ``tors.contains_unescaped`` / ``tors.find_unescaped``:
the escape-parity byte scan, at native speed, GIL-released.

``tors.find_unescaped(haystack, needle)`` finds the first occurrence of
``needle`` in ``haystack`` that is not itself escaped: an occurrence at byte
offset ``i`` counts only when the maximal run of ``b"\\"`` immediately before
``i`` has EVEN length. ``tors.contains_unescaped`` is the boolean spelling of
the same question. Both take ``bytes`` and only ``bytes``.

The motivating case (why parity, and why a public primitive): a JSON
serializer renders a real NUL codepoint (U+0000) as the six-byte escape text
``\u0000`` and the literal six-character text as seven bytes (the backslash
itself escaped: ``\\u0000``), and the second contains the first at offset +1,
so a plain substring test cannot tell them apart. PostgreSQL settles the
question downstream — a real NUL is fatal in a ``jsonb`` column (SQLSTATE
22P05), the literal text is fine — so a pipeline that binds serialized JSON
must answer "is this occurrence real or literal?" before the INSERT, and the
only stdlib spelling is find-then-re-parse-the-whole-value-and-recursively-
walk-it. The parity rule answers it directly from the raw bytes: the escape
text is live exactly when the backslash run before it is even (zero or a
pair-balanced run of escaped backslashes), and literal exactly when it is odd
(the run's first backslash is itself escaped, consuming the parity).

Semantics, pinned precisely (each row below is a pin):

1. **Even run = live, odd run = rejected**: the maximal run of backslashes
   immediately before the occurrence decides; nothing else (an ``a`` before
   the run, a backslash after the occurrence) can matter.
2. **Offset 0 is live**: the run before offset 0 is empty, and 0 is even.
3. **``-1`` when no live occurrence exists**: ``bytes.find``'s own sentinel,
   kept over ``Optional[int]`` deliberately (the stdlib spelling callers
   already branch on).
4. **Rejected hits advance the scan one byte past the hit, not past the
   whole match**: self-overlapping needles stay correct — the two-byte
   needle ``b"00"`` in a haystack of one backslash then ``000`` rejects the
   hit at 1 (one backslash before it) and finds the overlapping live hit at
   2, which a resume-at-match-end scan would skip entirely.
5. **Bytes in, byte offsets out**: the return of ``find_unescaped`` indexes
   the haystack's BYTES, never str codepoints — the same class of confusion
   ``find_patterns`` had to solve with its byte→char mapping, deliberately
   NOT solved here because the input is ``bytes`` and the contract is
   byte-space end to end (``haystack[i:i + len(needle)]`` is the needle).
   The multibyte battery below pins the divergence numerically against the
   decoded ``str``'s ``find``.
6. **No JSON knowledge lives in the function**: parity is the mechanism;
   "the needle is an escape sequence, so parity means live-vs-literal" is
   the caller's reading of it. Any backslash-escaped grammar (printf
   format strings, shell quotes, regex sources) can use the same scan.

Contract decisions at the argument boundary (each pinned below):

- an empty needle raises ``ValueError("empty needle")``: it would match at
  every position, the same rationale as ``find_patterns``' empty pattern;
- exactly ``bytes`` on both arguments (``bytearray`` / ``memoryview`` /
  ``str`` → ``TypeError``): the bytes-in surface's zero-copy
  immutable-borrow-under-detach contract (a writable buffer mutated by
  another thread mid-read is a data race, not a semantic difference);
- an empty haystack is legal and answerable (``-1`` / ``False``), and a
  needle longer than the haystack likewise.

No stdlib oracle exists for the parity question (the gap is the feature), so
the contract is proven the ``find_patterns`` way: (a) structural validity
over arbitrary generated inputs (a returned offset really holds the needle,
behind an even run), (b) a brute-force pure-Python backward parity walk (the
shared oracle, ``reference_find_unescaped`` in tests/reference.py — which is
also the manual loop the wall cells race) over hypothesis-driven
backslash-dense alphabets and arbitrary bytes, and (c) golden rows with exact
expected offsets, hand-computed, oracle-cross-checked so a wrong pin fails
loudly instead of laundering itself into the suite.

The invariant ``contains_unescaped(h, n) == (find_unescaped(h, n) != -1)``
is pinned over the same strategies (the two spellings are one scan by
construction; the pin exists so a future "optimized" contains cannot drift
from find without failing here).

The GIL-release claim (the whole scan under one ``py.detach``; the
bytes-in borrow plus a ``bool``/``int`` return, so no marshalling class at
all — the ``utf8_is_valid`` extreme point, ceiling-only heartbeat budget) is
pinned in ``tests/test_gil_release.py``; the criterion ladder for the Rust
core alone is the ``unescaped_scan`` group in ``benches/search.rs``.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reference import (
    UNESCAPED_NEEDLE,
    corpus_utf8,
    reference_find_unescaped,
    unescaped_false_positive,
)
from tors import contains_unescaped, find_unescaped

_MIB = 1024 * 1024

_NUL_ESCAPE = UNESCAPED_NEEDLE  # the six-byte escape text: \ u 0 0 0 0


def _assert_find_is_valid(haystack: bytes, needle: bytes, got: int) -> None:
    """Contract part (a), the structural validity every answer must satisfy:
    a live offset is in range, holds the needle at BYTE granularity (the
    slice check is the byte-offset pin), and sits behind an even backslash
    run. ``-1`` carries no structural content to check (absence is the
    oracle's to cross-check, run beside this helper in every test that uses
    it)."""
    if got == -1:
        return
    assert 0 <= got <= len(haystack) - len(needle), f"offset {got} out of range"
    assert haystack[got : got + len(needle)] == needle, (
        f"haystack[{got}:{got + len(needle)}] is not the needle: "
        "the offset does not index haystack bytes"
    )
    run = 0
    j = got - 1
    while j >= 0 and haystack[j] == 0x5C:
        run += 1
        j -= 1
    assert run % 2 == 0, f"offset {got} sits behind an odd run of {run} backslashes"


# --- Part (c): the golden parity battery ---------------------------------------------
#
# Every row: (haystack, needle, expected find_unescaped). Rows are built from
# code (backslash runs by repetition, JSON renderings by concatenation) so
# the source stays ASCII and unambiguous. The runs 0-7 ladder is generated,
# the class of shapes the adversarial matrix verified the algorithm
# against (backslash runs 0 through 7, mixed literal/real, adjacency).

_RUN_LADDER: list[tuple[bytes, bytes, int]] = [
    (b"\\" * k + _NUL_ESCAPE + b" tail", _NUL_ESCAPE, k if k % 2 == 0 else -1)
    for k in range(8)
]
_RUN_LADDER_IDS = [
    f"run-{k}-{'live-at-offset-' + str(k) if k % 2 == 0 else 'rejected'}" for k in range(8)
]

_GOLDEN_PARITY_CASES: list[tuple[bytes, bytes, int]] = _RUN_LADDER + [
    # orjson's two renderings, as they appear inside a JSON string value: a
    # real NUL codepoint ("a\u0000b") and the literal six-character text
    # ("a\\u0000b" on the wire). The literal contains the needle at +1
    # behind one backslash: rejected, -1.
    (b'"a\\u0000b"', _NUL_ESCAPE, 2),
    (b'"a\\\\u0000b"', _NUL_ESCAPE, -1),
    # Both orders of literal-then-real inside one serialized document: the
    # scan must reject the literal's occurrence and keep walking to the real
    # one, or (real first) answer at it immediately.
    (b'"a\\\\u0000b"' b'"c\\u0000d"', _NUL_ESCAPE, 13),
    (b'"c\\u0000d"' b'"a\\\\u0000b"', _NUL_ESCAPE, 2),
    # Adjacency: three literal renderings back to back (every occurrence
    # behind an odd run, the miniature false-positive corpus), then the same
    # with a real escape appended — the scan walks past all three rejections
    # to the live one.
    ((b"\\" + _NUL_ESCAPE) * 3, _NUL_ESCAPE, -1),
    ((b"\\" + _NUL_ESCAPE) * 3 + _NUL_ESCAPE, _NUL_ESCAPE, 21),
    # A rejected occurrence with a gap before the live one: the advance is
    # one byte past the hit, then the scan runs to the next occurrence.
    (b"\\" + _NUL_ESCAPE + b" middle " + _NUL_ESCAPE, _NUL_ESCAPE, 15),
    # The degenerate shapes: empty haystack, needle-is-the-haystack (offset
    # 0, run 0), needle longer than the haystack, no occurrence at all.
    (b"", _NUL_ESCAPE, -1),
    (_NUL_ESCAPE, _NUL_ESCAPE, 0),
    (b"\\u00", _NUL_ESCAPE, -1),
    (b"plain text, no escapes here", _NUL_ESCAPE, -1),
    # Buffer-end shapes: the needle live at the very end (run 2, even, hit
    # at 2, no tail past the match) and the rejected-at-end shape (run 1,
    # odd, -1 with no tail to walk past — the scan must not read past the
    # buffer nor invent a tail hit).
    (b"\\" * 2 + _NUL_ESCAPE, _NUL_ESCAPE, 2),
    (b"\\" + _NUL_ESCAPE, _NUL_ESCAPE, -1),
    # Adjacent live-live: two live occurrences back to back answer the
    # FIRST, not the last.
    (_NUL_ESCAPE + _NUL_ESCAPE, _NUL_ESCAPE, 0),
]

_GOLDEN_PARITY_IDS = _RUN_LADDER_IDS + [
    "json-real-nul-escape",
    "json-literal-escape-text",
    "json-literal-then-real",
    "json-real-then-literal",
    "adjacent-literals-all-rejected",
    "adjacent-literals-then-real",
    "rejected-then-live-across-a-gap",
    "empty-haystack",
    "needle-is-the-whole-haystack",
    "needle-longer-than-haystack",
    "no-occurrence",
    "live-at-buffer-end-no-tail",
    "rejected-at-buffer-end-no-tail",
    "adjacent-live-live-answers-first",
]


@pytest.mark.parametrize(
    ("haystack", "needle", "expected"),
    _GOLDEN_PARITY_CASES,
    ids=_GOLDEN_PARITY_IDS,
)
def test_golden_parity_battery(
    haystack: bytes, needle: bytes, expected: int
) -> None:
    """The fixed anchor of the contract: every golden row asserts the exact
    expected offset and the oracle's agreement, so a hand-computed expectation
    that disagreed with the brute-force reference would fail loudly here
    rather than silently laundering a wrong pin into the suite."""
    got = find_unescaped(haystack, needle)
    assert got == expected
    assert got == reference_find_unescaped(haystack, needle)
    _assert_find_is_valid(haystack, needle, got)
    assert contains_unescaped(haystack, needle) == (expected != -1)


# --- The self-overlapping-needle battery (the advance-one-byte pin) -------------------


_SELF_OVERLAP_CASES: list[tuple[bytes, bytes, int]] = [
    (b"\\000", b"00", 2),
    (b"\\0000", b"000", 2),
    (b"\\\\", b"\\", 0),
    (b"\\" * 8, b"\\\\", 0),
    (b"x" + b"\\" * 3 + b"y", b"\\\\", 1),
    (b"\\a\\aa", b"a", 4),
    # The border needles beyond the uniform-alphabet rows: ``abab`` (a proper
    # border of period 2 over a MIXED alphabet — the resume rule's shape when
    # the overlap stride and the alphabet both vary) rejects the hit at 1 and
    # finds the overlapping live hit at 3, inside the rejected match; ``aaa``
    # (period 1) does the same at 2. Both verified against the oracle.
    (b"\\abababab", b"abab", 3),
    (b"\\aaaaa", b"aaa", 2),
    # A chain of four ALL-REJECTED hits (every occurrence of ``\\u`` behind
    # exactly one backslash: runs 3, 1, 1, 1), the consecutive-rejected-hits
    # boundary shape — then the same chain with a live occurrence appended
    # behind a non-backslash, so the scan walks past every rejection to the
    # live hit.
    (b"\\" * 4 + b"u\\\\" * 3 + b"u", b"\\u", -1),
    (b"\\" * 4 + b"u\\\\" * 3 + b"u" + b"x\\u", b"\\u", 15),
]

_SELF_OVERLAP_IDS = [
    "two-byte-needle-overlapping-dead-then-live",
    "three-byte-needle-overlapping-dead-then-live",
    "all-backslash-needle-answers-at-run-start",
    "all-backslash-needle-over-a-long-run",
    "backslash-pair-needle-after-a-non-backslash",
    "single-char-needle-walks-the-overlap-chain",
    "period-two-border-needle-mixed-alphabet",
    "period-one-border-needle-chain",
    "four-rejected-hits-each-behind-one-backslash",
    "rejected-chain-then-live-behind-a-non-backslash",
]


@pytest.mark.parametrize(
    ("haystack", "needle", "expected"),
    _SELF_OVERLAP_CASES,
    ids=_SELF_OVERLAP_IDS,
)
def test_self_overlapping_needles_advance_one_byte_past_rejected_hits(
    haystack: bytes, needle: bytes, expected: int
) -> None:
    """The resume rule, pinned: a rejected hit advances the scan one byte
    past the HIT, not past the whole match, so an occurrence that starts
    inside the rejected match is still found. ``b"00"`` in ``b"\\000"`` is
    the canonical row: the hit at 1 is rejected (one backslash before it),
    the overlapping hit at 2 is live (the byte before it is the previous
    match's own ``0``), and a resume-at-match-end scan — memmem's
    ``find_iter`` default — would skip it and answer ``-1``. The
    all-backslash-needle rows pin the other edge: an all-backslash needle's
    first occurrence always sits at a run start (an even, empty run before
    it), so those scans answer at the first hit and never walk."""
    got = find_unescaped(haystack, needle)
    assert got == expected
    assert got == reference_find_unescaped(haystack, needle)
    _assert_find_is_valid(haystack, needle, got)


# --- The byte-offset battery (bytes in, BYTE offsets out) -----------------------------

_E_ACUTE = "\u00e9"  # 2 UTF-8 bytes, 1 char
_FAMILY = "\U0001f600"  # 4 UTF-8 bytes, 1 char
_CJK = "\u6771\u4eac"  # 2 chars, 6 UTF-8 bytes

# Each row: (haystack, needle, expected byte offset, the decoded str's own
# find for the same needle-as-text). The fourth column is the divergence pin:
# what a caller who decoded first (or assumed str semantics) would compute,
# which is exactly what find_unescaped must NOT report.
_BYTE_OFFSET_CASES: list[tuple[bytes, bytes, int, int]] = [
    (("caf" + _E_ACUTE + " ").encode("utf-8") + _NUL_ESCAPE, _NUL_ESCAPE, 6, 5),
    (_FAMILY.encode("utf-8") + _NUL_ESCAPE, _NUL_ESCAPE, 4, 1),
    (_CJK.encode("utf-8") + _NUL_ESCAPE, _NUL_ESCAPE, 6, 2),
    ((b"\\" + _NUL_ESCAPE) * 2 + b"caf" + _E_ACUTE.encode("utf-8"), _NUL_ESCAPE, -1, 1),
]

_BYTE_OFFSET_IDS = [
    "two-byte-char-before-the-needle",
    "four-byte-emoji-before-the-needle",
    "cjk-before-the-needle",
    "rejected-hits-then-multibyte-then-nothing",
]


@pytest.mark.parametrize(
    ("haystack", "needle", "expected", "str_find"),
    _BYTE_OFFSET_CASES,
    ids=_BYTE_OFFSET_IDS,
)
def test_offsets_index_haystack_bytes_not_str_codepoints(
    haystack: bytes, needle: bytes, expected: int, str_find: int
) -> None:
    """The byte-offset contract, pinned against the decoded ``str``'s own
    ``find`` on the same content: the two disagree on every multibyte row
    (``caf\u00e9`` puts the needle at BYTE 6 but CHAR 5; the emoji at BYTE 4
    but CHAR 1), and ``find_unescaped`` must report the byte offset — the
    value that slices the ``bytes`` argument it was handed. A char-unit
    implementation (the bug class the ``find_patterns`` byte→char mapping
    warns about, inverted here) fails every row. The last row additionally
    walks two rejected hits before running out of haystack past a multibyte
    tail: ``-1`` with the multibyte bytes still traversed."""
    got = find_unescaped(haystack, needle)
    assert got == expected
    assert got == reference_find_unescaped(haystack, needle)
    _assert_find_is_valid(haystack, needle, got)
    # The divergence pin: the decoded text's find is a DIFFERENT number on
    # every row (the column is hand-computed), so this asserts the two
    # coordinate systems really are in play, not accidentally equal.
    assert haystack.decode("utf-8").find(needle.decode("ascii")) == str_find
    assert str_find != expected


# --- The argument-boundary contract --------------------------------------------------


class TestArgumentContract:
    def test_empty_needle_raises_value_error(self) -> None:
        """An empty needle would match at every position and has no
        parity meaning (there is no occurrence to stand before), refused up
        front with exactly ``ValueError("empty needle")`` — the
        ``find_patterns`` empty-pattern rationale — by both spellings, and
        before any scanning runs (the error fires under the GIL, ahead of
        the detach)."""
        with pytest.raises(ValueError, match="^empty needle$"):
            find_unescaped(b"any haystack", b"")
        with pytest.raises(ValueError, match="^empty needle$"):
            contains_unescaped(b"any haystack", b"")

    @pytest.mark.parametrize(
        "not_bytes",
        [bytearray(b"abc"), memoryview(b"abc"), "abc", 123, None, [b"a", b"b"]],
        ids=["bytearray", "memoryview", "str", "int", "none", "list"],
    )
    def test_non_bytes_haystack_raises_type_error(self, not_bytes: object) -> None:
        """Exactly ``bytes`` (the bytes-in surface's contract, the
        ``utf8_is_valid``/``decode_utf8`` pin): a writable buffer borrowed
        zero-copy under a detached scan is a data race, not a semantic
        difference, and a ``str`` haystack is a different coordinate system
        entirely (codepoints, not bytes — the very confusion the offset
        contract forbids)."""
        with pytest.raises(TypeError):
            find_unescaped(not_bytes, b"\\u0000")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            contains_unescaped(not_bytes, b"\\u0000")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "not_bytes",
        [bytearray(b"abc"), memoryview(b"abc"), "abc", 123, None, [b"a", b"b"]],
        ids=["bytearray", "memoryview", "str", "int", "none", "list"],
    )
    def test_non_bytes_needle_raises_type_error(self, not_bytes: object) -> None:
        """The needle pays the same exactly-``bytes`` contract as the
        haystack (the same borrow, the same detach, the same coordinate
        system)."""
        with pytest.raises(TypeError):
            find_unescaped(b"\\u0000", not_bytes)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            contains_unescaped(b"\\u0000", not_bytes)  # type: ignore[arg-type]

    def test_empty_haystack_is_answerable_not_an_error(self) -> None:
        """The degenerate-but-legal input: no occurrence can exist, so the
        answer is ``-1`` / ``False``, never an error (the empty needle is
        the error; the empty haystack is just a haystack with nothing in
        it)."""
        assert find_unescaped(b"", b"\\u0000") == -1
        assert contains_unescaped(b"", b"\\u0000") is False

    def test_the_returns_are_genuine_int_and_bool(self) -> None:
        """The marshalling shapes, pinned as exact types on both paths: a
        bare ``int`` (with ``-1`` an ``int``, never a ``bool`` or ``None``)
        and a genuine ``bool`` on both of its paths — the ``utf8_is_valid``
        no-marshalling-class design."""
        assert type(find_unescaped(b"\\u0000", b"\\u0000")) is int
        assert type(find_unescaped(b"nothing", b"\\u0000")) is int
        assert find_unescaped(b"nothing", b"\\u0000") == -1
        assert contains_unescaped(b"\\u0000", b"\\u0000") is True
        assert contains_unescaped(b"nothing", b"\\u0000") is False


# --- Parts (a)+(b): validity + the differential oracle over generated inputs ---------


# The backslash-dense alphabets: from pure backslash (the run-machinery's
# worst case) through the escape-text alphabet to a mixed composition with a
# real NUL byte in it (arbitrary bytes are legal haystack content; the scan
# must not care).
_DENSE_ALPHABETS = (b"\\", b"\\u", b"\\u0", b"\\u0a", b"\\u0a\x00")


def _bytes_over(alphabet: bytes, min_size: int = 0, max_size: int = 60) -> st.SearchStrategy[bytes]:
    """Byte strings drawn from exactly ``alphabet``'s bytes (the strategy
    this hypothesis version's ``st.binary`` cannot spell: no ``alphabet=``
    parameter there, so lists of one-byte ``bytes`` are drawn and joined)."""
    return st.lists(
        st.sampled_from([bytes((c,)) for c in alphabet]),
        min_size=min_size,
        max_size=max_size,
    ).map(b"".join)


@st.composite
def _parity_haystack_and_needle(draw: st.DrawFn) -> tuple[bytes, bytes]:
    """Backslash-dense (haystack, needle) pairs: the haystack drawn over one
    of the dense alphabets (pure backslash included, so long runs and the
    carried-run-state machinery face their worst case), the needle either a
    substring of the haystack (so occurrences — live and rejected alike —
    actually occur) or drawn over the same alphabet."""
    alphabet = draw(st.sampled_from(_DENSE_ALPHABETS))
    haystack = draw(_bytes_over(alphabet, max_size=60))
    if haystack and draw(st.booleans()):
        i = draw(st.integers(0, len(haystack) - 1))
        j = draw(st.integers(i + 1, len(haystack)))
        needle = haystack[i:j]
    else:
        needle = draw(_bytes_over(alphabet, min_size=1, max_size=5))
    return haystack, needle


@st.composite
def _arbitrary_haystack_and_needle(draw: st.DrawFn) -> tuple[bytes, bytes]:
    """Arbitrary bytes (hypothesis's full ``st.binary``: any byte value, no
    alphabet bias) with substring-biased needles, the coverage class the
    fixed alphabets cannot reach — multibyte UTF-8 fragments, NULs, high
    bytes, all interleaved with whatever backslashes hypothesis draws."""
    haystack = draw(st.binary(max_size=64))
    if haystack and draw(st.booleans()):
        i = draw(st.integers(0, len(haystack) - 1))
        j = draw(st.integers(i + 1, len(haystack)))
        needle = haystack[i:j]
    else:
        needle = draw(st.binary(min_size=1, max_size=6))
    return haystack, needle


class TestDifferentialParity:
    @given(_parity_haystack_and_needle())
    @settings(max_examples=500)
    def test_matches_the_backward_parity_walk_over_backslash_dense_alphabets(
        self, haystack_needle: tuple[bytes, bytes]
    ) -> None:
        """The differential proof over the dense alphabets: tors's answer
        must equal the brute-force oracle's exactly for every generated
        pair. An escape-parity bug of any kind (a run miscounted at a
        boundary, a carried state desynced after a rejected hit, a resume
        advanced past a live overlapping occurrence) breaks this property;
        the golden batteries pin the individual shapes it finds."""
        haystack, needle = haystack_needle
        got = find_unescaped(haystack, needle)
        _assert_find_is_valid(haystack, needle, got)
        assert got == reference_find_unescaped(haystack, needle)

    @given(_arbitrary_haystack_and_needle())
    @settings(max_examples=300)
    def test_matches_the_backward_parity_walk_over_arbitrary_bytes(
        self, haystack_needle: tuple[bytes, bytes]
    ) -> None:
        """The arbitrary-bytes differential: substring-biased needles over
        hypothesis's full byte space, still under exact equality with the
        oracle — the scan's no-JSON-assumptions contract means arbitrary
        bytes are in-contract input, and the oracle is byte-blind by
        construction."""
        haystack, needle = haystack_needle
        got = find_unescaped(haystack, needle)
        _assert_find_is_valid(haystack, needle, got)
        assert got == reference_find_unescaped(haystack, needle)

    @given(st.one_of(_parity_haystack_and_needle(), _arbitrary_haystack_and_needle()))
    @settings(max_examples=400)
    def test_contains_is_exactly_find_having_found(
        self, haystack_needle: tuple[bytes, bytes]
    ) -> None:
        """The invariant, pinned over both strategies:
        ``contains_unescaped(h, n) == (find_unescaped(h, n) != -1)``. The
        two spellings are one scan by construction today; the pin exists so
        that stays true — a future "optimized" contains that takes a
        different code path and drifts (a missed early exit, a different
        resume rule) fails here first, before any caller's guard does."""
        haystack, needle = haystack_needle
        assert contains_unescaped(haystack, needle) == (
            find_unescaped(haystack, needle) != -1
        )


def test_every_needle_over_a_tiny_alphabet_matches_the_reference() -> None:
    """The deterministic sweep (the suite's exhaustive-small-alphabet idiom):
    every needle of 1-2 bytes over ``{b"\\", b"u", b"0"}`` (12 needles,
    self-overlapping ones like ``b"\\\\"`` and ``b"00"`` included) crossed
    with every haystack over the same alphabet up to length 7 (3,280
    haystacks): 39,360 pairs, the complete small space of run/overlap/
    adjacency interactions at that size, no sampling at all."""
    alphabet = (b"\\", b"u", b"0")
    needles: list[bytes] = []
    for size in 1, 2:
        needles.extend(b"".join(combo) for combo in itertools.product(alphabet, repeat=size))
    haystacks: list[bytes] = []
    for length in range(8):
        haystacks.extend(
            b"".join(combo) for combo in itertools.product(alphabet, repeat=length)
        )
    for needle in needles:
        for haystack in haystacks:
            assert find_unescaped(haystack, needle) == reference_find_unescaped(
                haystack, needle
            )


# --- Wall cells ----------------------------------------------------------------------
#
# No stdlib spelling of the parity question exists (the gap is the feature),
# so the racers are the two expressions a caller writes today, both
# load-fair in-process (a shared-runner slowdown inflates both sides
# together): the substring prefilter (``needle in data`` — the cheap first
# pass of every guard that then NEEDS the confirm walk, so it is the floor,
# not the competition) and the manual parity loop
# (``reference_find_unescaped``: find + count-backslashes-per-hit, the exact
# hand-rolled algorithm this surface lifts out of one consumer's private
# module). Ceiling ratios against the floor, win margins against the loop.


def _min_wall_ms(op: Callable[[], object], samples: int = 3, warmup: int = 1) -> float:
    """Min-of-``samples`` wall after ``warmup`` runs (the suite's shared
    methodology)."""
    for _ in range(warmup):
        op()
    best = float("inf")
    for _ in range(samples):
        started = time.perf_counter()
        op()
        best = min(best, time.perf_counter() - started)
    return best * 1000.0


# The no-match cell's ceiling vs the bare prefilter: the measured direction
# is tors FASTER than CPython's `in` by ~20x (memchr's memmem is SIMD-class
# where CPython's substring search is not: 0.26ms vs 5.4ms at 12 MiB), so
# the 8x ceiling is not a "may pay a little more" bound but the class pin
# with room for a box where the two engines' relative speed sits
# differently — the chunk_hierarchical no-match precedent's margin over the
# same `in` racer. The absolute companion (40ms, ~150x above the measured
# wall) catches the per-byte classes — a quadratic scan, a per-byte
# structure walk — on every box; per-HIT classes are invisible to this cell
# by construction (a no-occurrence corpus never bills per-hit work), which
# is why the dense cell below carries the tightened per-hit band; and a
# memchr scalar-fallback swap (~6-12ms here) is a memchr-version event,
# not a tors code regression, recorded as out of this cell's teeth rather
# than pretended caught.
_SCAN_VS_PREFILTER_CEILING = 8.0
_SCAN_12MIB_CEILING_MS = 40.0


@pytest.mark.timing
def test_no_match_scan_stays_within_the_bare_prefilter_band_at_12mib() -> None:
    """The pure-scan shape (the sparse corpus: prose bytes, no backslash, no
    occurrence — the prefilter's own best case, no confirm walk would ever
    fire): one ``find_unescaped`` call vs one ``needle in data`` check over
    the same 12 MiB. Measured on this 16-core macOS box (ambient load
    ~9-11, min-of-3 after warm-up, the cells' own methodology): tors 0.26ms
    against the prefilter's 5.4ms — the native scan is ~20x FASTER than
    CPython's own substring search, recorded, not thresholded away; the
    assertions pin the class relationship (8x) and the absolute band
    (40ms)."""
    data = corpus_utf8("prose", 12 * _MIB)
    needle = UNESCAPED_NEEDLE
    tors_ms = _min_wall_ms(lambda: find_unescaped(data, needle))
    prefilter_ms = _min_wall_ms(lambda: needle in data)
    assert tors_ms < _SCAN_VS_PREFILTER_CEILING * prefilter_ms, (
        f"12 MiB no-match scan: tors {tors_ms:.1f}ms against an "
        f"{prefilter_ms:.1f}ms bare prefilter ({tors_ms / prefilter_ms:.0f}x); "
        "the scan left the substring-search class it shares with `in`"
    )
    assert tors_ms < _SCAN_12MIB_CEILING_MS, (
        f"12 MiB no-match scan: tors took {tors_ms:.1f}ms, over the absolute "
        f"band (ceiling {_SCAN_12MIB_CEILING_MS:.0f}ms); the scan regressed "
        "out of its measured class"
    )


# The dense cell's margin: the stdlib-expression race idiom (tors < 0.9 x
# the expression it replaces), the same _MARGIN as every test_performance.py
# cell; the absolute companion catches the classes the load-fair race is
# blind to (both sides slowing together) AND the per-hit classes the
# no-match cell's corpus can never bill: 4ms is ~5.7x above the measured
# 0.70ms wall, so ~55ns of new per-hit work over this corpus's 72,520
# rejected hits (one small heap allocation, say) already fails the band.
_LOOP_WIN_MARGIN = 0.9
_DENSE_12MIB_CEILING_MS = 4.0


@pytest.mark.timing
def test_false_positive_scan_beats_the_manual_parity_loop_at_12mib() -> None:
    """The hit-dense shape (the false-positive corpus: one literal
    ``\\\\u0000`` per sentence, every occurrence behind an odd run, so the
    scan rejects every hit and runs to the end — the workload the
    confirm-by-reparse walk existed for, and the worst case for both wall
    time and GIL release): one ``find_unescaped`` call vs the manual parity
    loop over the same bytes. Measured on this 16-core macOS box (ambient
    load ~9-11, min-of-3 after warm-up, the cells' own methodology): tors
    0.70ms (72,520 rejected hits) against the loop's 13.2ms, ratio 0.053 —
    the native scan wins by ~19x, and the 0.9 margin absorbs any load that
    moves both sides together. ``contains_unescaped`` measured 0.71ms on
    the same corpus (the same scan by construction, pinned by the
    invariant test), so the race answers for both spellings. The tightened
    4ms absolute band (was 40ms) is what makes the per-hit class catchable
    at all: a no-occurrence corpus never bills per-hit work, and the
    load-fair race stays quiet while both sides pay the same per-hit
    cost."""
    data = unescaped_false_positive(12 * _MIB)
    needle = UNESCAPED_NEEDLE
    tors_ms = _min_wall_ms(lambda: find_unescaped(data, needle))
    loop_ms = _min_wall_ms(lambda: reference_find_unescaped(data, needle))
    assert tors_ms < _LOOP_WIN_MARGIN * loop_ms, (
        f"12 MiB false-positive scan: tors {tors_ms:.1f}ms vs the manual "
        f"parity loop's {loop_ms:.1f}ms (ratio {tors_ms / loop_ms:.2f}): the "
        "native scan lost more than the tolerance margin to the hand-rolled "
        "expression it exists to replace"
    )
    assert tors_ms < _DENSE_12MIB_CEILING_MS, (
        f"12 MiB false-positive scan: tors took {tors_ms:.1f}ms, over the "
        f"absolute band (ceiling {_DENSE_12MIB_CEILING_MS:.0f}ms, ~5.7x "
        "above the measured wall); the scan regressed out of its measured "
        "class"
    )
