"""Contract gate for ``tors.b64_decode``: indistinguishable from
``base64.b64decode(s, validate=...)`` over ASCII strings (results, raised
exceptions, and messages): the decode direction of the content-addressing pair
whose encode half (``b64_encode_bytes``) is pinned in tests/test_b64.py.

Why error-path parity is the product: the decoded bytes are stored and compared
(content-addressing), so a caller swapping ``base64.b64decode`` for tors must
see the same ``binascii.Error`` messages any log scraping, retry logic, or test
suite already matches on, not just the same bytes on the happy path.

The implementation is a line-for-line port of the POST-gh-145264
``binascii_a2b_base64_impl`` (Modules/binascii.c on CPython's 3.13 and 3.14
maintenance branches, identical there by the backport commits 1f9958f9 /
e31c5512), the algorithm ``base64.b64decode`` has delegated to since 3.11.
gh-145264 (March 2026) changed the machine in two ways, both pinned below:

- LENIENT MODE no longer stops at the first completed pad sequence: excess
  pads are ignored per RFC 4648 §3.3 and later data chars resume decoding
  (``'Zg==Zg=='`` → ``b'f\\x06`'`` where the pre-fix machine returned
  ``b'f'``, silently DROPPING the second block). That truncation was a
  parser-differential CPython fixed as a security issue; tors ships the fixed
  machine on every interpreter it supports.
- STRICT MODE classifies a pad at quad position 1 as the end-of-input length
  error (``'Z=g='`` → "number of data characters (1)" where the pre-fix
  machine raised "Discontinuous padding not allowed"), and a third pad beyond
  a quad is "Excess padding" rather than "Excess data after padding"
  (``'Zg==='``). Data-after-padding and discontinuous-padding remain distinct,
  reachable messages (``'Zg==Z'``, ``'Zg=g'``).

ONE behavior, by design: tors's output does NOT depend on the interpreter.
The recorded literals below are the post-fix machine's outputs, measured on
CPython 3.13.14 and byte-identical on 3.15.0b2 (both measured pre-implementation
on this box). The LIVE differential against the running stdlib is gated on the
stdlib actually having the fix: asserted where it does, recorded-not-asserted
where it does not (every divergence printed for the record). The gate is a
BEHAVIORAL probe of the running ``base64.b64decode``, not a version tuple;
measured on this box, 3.14.0 does NOT have the fix (the e31c5512 backport
arrives in later 3.14.x patch releases) while 3.13.14 does, so a
``sys.version_info >= (3, 13, 14) or >= (3, 14)`` gate would misfire on
exactly the patch levels a version gate exists to get right.

Documented divergences, named (the parity discipline):

- PRE-FIX stdlib (CPython ≤3.12.x, 3.13.0–3.13.13, 3.14.0 measured here): the
  lenient truncation and the two strict message changes above: tors keeps the
  fixed machine; the battery prints each divergence instead of asserting.
- CPython 3.10: ``base64.b64decode(validate=True)`` is NOT the binascii
  strict machine at all (3.11 added ``strict_mode``); it is a regex validator
  with its own messages ("Non-base64 digit found") that even ACCEPTS inputs
  the machine rejects (``'Zm9v='`` → ``b'foo'``). tors ships the 3.11+
  machine on 3.10 too; the stdlib comparison is recorded, never asserted,
  there.

Decisions pinned here (from the spec, unchanged by the retarget):
- ``validate=True`` is the DEFAULT (the spec's signature; stdlib's default is
  False; the delta is intentional: decode-side callers want invalid input to
  fail loudly, and every encode-side caller already produced strict-valid
  output).
- The raised class is the REAL ``binascii.Error``, constructed via pyo3's
  ``PyErr::from_value`` on the class imported from the ``binascii`` module at
  raise time (pyo3 0.29 has no ``PyBinasciiError`` (``binascii.Error`` is a
  module-level C exception outside the ``PyExc_*`` table), but any exception
  class is constructible through ``PyErr``; verified in pyo3 0.29.2's
  ``src/err/mod.rs``). So ``except binascii.Error`` and ``except ValueError``
  both catch it, and ``type(exc) is binascii.Error``.
- The argument must be exactly ``str`` (``bytes`` / ``bytearray`` /
  ``memoryview`` → ``TypeError``): the signature's contract. stdlib also
  accepts bytes-likes; callers holding bytes already have the stdlib spelling
  and ``b64_encode_bytes`` is the bytes-side of this pair.
- A non-ASCII ``str`` raises PLAIN ``ValueError("string argument should
  contain only ASCII characters")`` (from ``base64.py``'s
  ``_bytes_from_decode_data``, before any decoding), not ``binascii.Error``.
  The same holds for a str holding LONE SURROGATES: the stdlib's
  ``s.encode("ascii")`` fails inside ``_bytes_from_decode_data`` and is
  converted to that same plain ``ValueError``, so tors pre-checks and raises
  it rather than letting pyo3's UTF-8 borrow surface its own
  ``UnicodeEncodeError`` (a type match with the stdlib by design, not the
  asymmetry a naive pyo3 passthrough would produce).
"""

from __future__ import annotations

import base64
import binascii

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import b64_decode, b64_encode_bytes

# The measured malformed battery, recorded on the POST-gh-145264 machine
# (CPython 3.13.14; byte-identical outputs and messages on 3.15.0b2, both
# measured on this box before the retarget): (input, validate) →
# (expected bytes | (exception qualname, message)). Rows whose literal CHANGED
# from the pre-fix battery carry a comment naming the change; the rows new to
# this battery pin behaviors only the fixed machine has.
_BATTERY: list[tuple[str, bool, bytes | tuple[str, str]]] = [
    # --- valid strict decoding (incl. the non-canonical trailing-bits case) ---
    ("Zm9vYmFy", True, b"foobar"),
    ("Zm9vYmFy", False, b"foobar"),
    ("", True, b""),
    ("", False, b""),
    ("Zg==", True, b"f"),
    ("Zm9=", True, b"fo"),
    ("Zh==", True, b"f"),  # non-canonical trailing bits: accepted, like a2b_base64
    ("ABCDEFGH", True, b"\x00\x10\x83\x10Q\x87"),
    # --- strict-mode errors ---
    (
        "Z",
        True,
        (
            "Error",
            "Invalid base64-encoded string: number of data characters (1) "
            "cannot be 1 more than a multiple of 4",
        ),
    ),
    (
        "Zm9vY",
        True,
        (
            "Error",
            "Invalid base64-encoded string: number of data characters (5) "
            "cannot be 1 more than a multiple of 4",
        ),
    ),
    (
        "Z==",
        True,
        (
            "Error",
            "Invalid base64-encoded string: number of data characters (1) "
            "cannot be 1 more than a multiple of 4",
        ),
    ),
    (
        "A==",
        True,
        (
            "Error",
            "Invalid base64-encoded string: number of data characters (1) "
            "cannot be 1 more than a multiple of 4",
        ),
    ),
    ("Zg=", True, ("Error", "Incorrect padding")),
    ("Zm9vYg", True, ("Error", "Incorrect padding")),
    ("Zm9vYg=", True, ("Error", "Incorrect padding")),
    # A third pad beyond a complete quad: "Excess padding" (was "Excess data
    # after padding" pre-fix; the completed quad no longer ends the parse).
    ("Zg===", True, ("Error", "Excess padding not allowed")),
    ("Zm9vYg===", True, ("Error", "Excess padding not allowed")),
    ("Zm9vYg====", True, ("Error", "Excess padding not allowed")),
    # Input continuing past a completed pad sequence with DATA: unchanged.
    ("Zg==Zg==", True, ("Error", "Excess data after padding")),
    ("Zm9vYg==Zg==", True, ("Error", "Excess data after padding")),
    ("Zg==Z", True, ("Error", "Excess data after padding")),
    ("Zh==X", True, ("Error", "Excess data after padding")),
    ("Zm9vYg==Zm9vYg==", True, ("Error", "Excess data after padding")),
    # A non-alphabet char after padding is "Only base64 data" (was "Excess
    # data after padding" pre-fix: the pad no longer ends the parse, so the
    # non-alphabet check fires first).
    ("Zm9vYg==\n", True, ("Error", "Only base64 data is allowed")),
    ("Zg==\n", True, ("Error", "Only base64 data is allowed")),
    ("Zm9v!", True, ("Error", "Only base64 data is allowed")),
    ("Zm 9v\nYg==", True, ("Error", "Only base64 data is allowed")),
    ("Zm9vYg=\n=", True, ("Error", "Only base64 data is allowed")),
    ("\n====", True, ("Error", "Only base64 data is allowed")),
    # A pad at quad position 1: the end-of-input length error (was
    # "Discontinuous padding" pre-fix: the machine now breaks to the count
    # error instead of classifying the pad itself).
    (
        "Z=g=",
        True,
        (
            "Error",
            "Invalid base64-encoded string: number of data characters (1) "
            "cannot be 1 more than a multiple of 4",
        ),
    ),
    (
        "Z==g=",
        True,
        (
            "Error",
            "Invalid base64-encoded string: number of data characters (1) "
            "cannot be 1 more than a multiple of 4",
        ),
    ),
    # A data char after ONE pad at quad position 2: still discontinuous.
    ("Zg=g", True, ("Error", "Discontinuous padding not allowed")),
    ("Zm8=g", True, ("Error", "Excess data after padding")),
    ("====", True, ("Error", "Leading padding not allowed")),
    ("=====", True, ("Error", "Leading padding not allowed")),
    ("Zm9v=", True, ("Error", "Excess padding not allowed")),
    ("Zm9v=Zg==", True, ("Error", "Excess padding not allowed")),
    # --- lenient-mode: non-alphabet discarded, excess pads ignored, data
    # --- after padding RESUMES decoding (the gh-145264 fix itself) ---
    (
        "Z",
        False,
        (
            "Error",
            "Invalid base64-encoded string: number of data characters (1) "
            "cannot be 1 more than a multiple of 4",
        ),
    ),
    (
        "Zm9vY",
        False,
        (
            "Error",
            "Invalid base64-encoded string: number of data characters (5) "
            "cannot be 1 more than a multiple of 4",
        ),
    ),
    (
        "Z==",
        False,
        (
            "Error",
            "Invalid base64-encoded string: number of data characters (1) "
            "cannot be 1 more than a multiple of 4",
        ),
    ),
    (
        "A=",
        False,
        (
            "Error",
            "Invalid base64-encoded string: number of data characters (1) "
            "cannot be 1 more than a multiple of 4",
        ),
    ),
    ("Zg=", False, ("Error", "Incorrect padding")),
    ("Zm9vYg", False, ("Error", "Incorrect padding")),
    ("Z=g=", False, ("Error", "Incorrect padding")),
    ("Zm9v!", False, b"foo"),
    ("Zm 9v\nYg==", False, b"foob"),
    ("Zm9vYg=\n=", False, b"foob"),
    ("Zg==\n", False, b"f"),
    # The security fix: data after a completed pad sequence is DECODED, not
    # dropped (each of these was truncated at the pad pre-fix).
    ("Zg==!Zg==", False, b"f\x06`"),
    ("Zg==Zg==", False, b"f\x06`"),
    ("Zg==Zg==Zg==", False, b"f\x06`f"),
    ("Zm9vYg==Zg==", False, b"foob\x06`"),
    ("Zm9vYg==Zm9vYg==", False, b"foob\x06f\xf6\xf6 "),
    ("====", False, b""),
    ("=====", False, b""),
    ("Zm9v=", False, b"foo"),
    ("Zm9v=Zg==", False, b"foof"),  # stray mid-string pad starts a fresh quad
    ("Zg===", False, b"f"),
    ("Zm9vYm E=", False, b"fooba"),
]


def _stdlib_outcome(s: str, validate: bool) -> bytes | tuple[str, str]:
    """The stdlib's outcome shape: decoded bytes, or (class qualname, message)."""
    try:
        return base64.b64decode(s, validate=validate)
    except Exception as exc:  # noqa: BLE001 - the battery's own classification
        return type(exc).__qualname__, str(exc)


def _tors_outcome(s: str, validate: bool) -> bytes | tuple[str, str]:
    try:
        return b64_decode(s, validate=validate)
    except Exception as exc:  # noqa: BLE001 - the battery's own classification
        return type(exc).__qualname__, str(exc)


def _stdlib_has_the_security_fix() -> bool:
    """Does the RUNNING ``base64.b64decode`` carry the gh-145264 machine?

    Probed behaviorally, not by version tuple: the pre-fix machine truncates
    lenient decoding at the first completed pad sequence (``b'f'``), the
    fixed machine decodes the trailing block (``b'f\\x06`'``). Measured on
    this box: fixed in 3.13.14 and 3.15.0b2, NOT in 3.14.0 (the e31c5512
    backport arrives in later 3.14.x) nor in any 3.12.x, so a version gate keyed
    on ">= (3, 13, 14) or >= (3, 14)" would misfire on 3.14.0, and patch
    levels are exactly where version gates lie. The probe reads the behavior
    itself, so it is right on every patch release without claiming anything
    about ones not measured.
    """
    return base64.b64decode("Zg==Zg==") == b"f\x06`"


_STDLIB_FIXED = _stdlib_has_the_security_fix()

# The variants where the pre-fix stdlib truncates but tors keeps decoding,
# printed (recorded, not asserted) on pre-fix interpreters so the known
# divergence is visible in every run's log instead of silently passing.
_PRE_FIX_DIVERGENCE_NOTE = (
    "gh-145264 divergence (recorded, not asserted; the running stdlib "
    "predates the fix; tors ships the fixed machine): "
)


class TestBatteryAgainstTheStdlib:
    @pytest.mark.parametrize(
        ("s", "validate", "expected"),
        _BATTERY,
        ids=[f"{s!a}-{'strict' if v else 'lenient'}" for s, v, _ in _BATTERY],
    )
    def test_tors_ships_the_recorded_post_fix_machine_and_matches_the_stdlib_where_it_has_the_fix(
        self, s: str, validate: bool, expected: bytes | tuple[str, str]
    ) -> None:
        """Every battery row must equal the recorded post-fix literal (tors
        ships ONE behavior on every interpreter: output must not depend on
        the Python it runs under), and must equal the RUNNING interpreter's
        own ``base64.b64decode`` outcome WHERE THE STDLIB HAS THE FIX. On
        pre-fix stdlibs the comparison is recorded, not asserted: each
        divergence is printed with the gh-145264 note (the 3.10 leg diverges
        on far more: its regex validator is a different machine entirely)."""
        tors = _tors_outcome(s, validate)
        assert tors == expected, (
            f"{s!r} validate={validate}: tors diverged from the recorded "
            "post-gh-145264 literal; tors must ship the fixed machine on "
            "every interpreter"
        )
        stdlib = _stdlib_outcome(s, validate)
        if _STDLIB_FIXED:
            assert stdlib == expected, (
                f"{s!r} validate={validate}: the running (post-fix) stdlib "
                "disagrees with the recorded literal; re-record the battery"
            )
        elif tors != stdlib:
            print(
                f"{_PRE_FIX_DIVERGENCE_NOTE}{s!r} validate={validate}: "
                f"tors={tors!r} stdlib={stdlib!r}"
            )


class TestErrorPathTypeParity:
    @pytest.mark.parametrize(
        ("s", "validate", "expected"),
        [(s, v, e) for s, v, e in _BATTERY if isinstance(e, tuple)],
        ids=[
            f"{s!a}-{'strict' if v else 'lenient'}" for s, v, e in _BATTERY if isinstance(e, tuple)
        ],
    )
    def test_raises_the_real_binascii_error_class_with_the_recorded_message(
        self, s: str, validate: bool, expected: tuple[str, str]
    ) -> None:
        """The raised exception is ``binascii.Error`` itself, constructed via
        pyo3's PyErr machinery from the class imported from the ``binascii``
        module, so existing ``except binascii.Error`` handlers keep working,
        and ``except ValueError`` catches it too (the subclass relation the
        spec's parity requirement rests on). The MESSAGE is tors's recorded
        literal (the fixed machine) on every interpreter; it equals the
        running stdlib's message only where the stdlib has the fix."""
        with pytest.raises(binascii.Error) as excinfo:
            b64_decode(s, validate=validate)
        exc = excinfo.value
        assert type(exc) is binascii.Error
        assert isinstance(exc, ValueError)
        assert str(exc) == expected[1]
        if _STDLIB_FIXED:
            assert str(exc) == _stdlib_outcome(s, validate)[1]

    def test_the_length_error_interpolates_the_data_char_count(self) -> None:
        """The one parameterized message: the count is derived from emitted
        bytes (complete quads × 4 + 1), so a 5-data-char string reports 5, a
        9-data-char string 9, the count a caller's log scraping keys on.
        The formula is UNCHANGED by the gh-145264 retarget."""
        for data_chars in (1, 5, 9):
            count = data_chars  # a bare data run has no complete quads before it
            s = "A" * data_chars
            with pytest.raises(binascii.Error, match=rf"data characters \({count}\)"):
                b64_decode(s, validate=True)
        # Two complete quads (8 chars) plus one straggler, no padding at all:
        # the count is complete-quads × 4 + 1 = 9, in both modes.
        for validate in (True, False):
            with pytest.raises(binascii.Error, match=r"data characters \(9\)"):
                b64_decode("Zm9vYmFyZ", validate=validate)

    def test_the_pad_at_quad_position_one_is_the_length_error(self) -> None:
        """The gh-145264 strict-mode change, pinned by name: a pad arriving
        at quad position 1 breaks to the end-of-input count error (count 1,
        no complete quad was emitted), NOT the pre-fix machine's
        "Discontinuous padding" classification of the following data char."""
        for s in ("Z=g=", "Z==g="):
            with pytest.raises(binascii.Error, match=r"data characters \(1\)"):
                b64_decode(s, validate=True)
        # The contrast row: a data char after ONE pad at quad position 2 is
        # still "Discontinuous padding"; the two classes stay distinct.
        with pytest.raises(binascii.Error, match="Discontinuous padding"):
            b64_decode("Zg=Zg==", validate=True)


class TestValidateFalseLenientParity:
    @given(st.binary(max_size=48))
    @settings(max_examples=300)
    def test_lenient_mode_equals_the_stdlib_over_arbitrary_shapes(self, raw: bytes) -> None:
        """Differential over arbitrary bytes rendered to b64 strings and then
        corrupted: whitespace and junk chars injected between the alphabet
        characters, plus truncated/padded variants: the lenient discard
        rules must agree with the post-fix ``a2b_base64`` on every shape.
        Runs only where the running stdlib HAS the fix: on a pre-fix stdlib
        the machines diverge by design (tors decodes past padding), so
        there is nothing to assert; the battery's recorded leg carries the
        pre-fix signal."""
        if not _STDLIB_FIXED:
            pytest.skip(
                "running stdlib predates the gh-145264 base64 security fix; "
                "tors ships the fixed machine, so lenient "
                "parity with THIS stdlib does not hold; asserted on "
                "post-fix interpreters (3.13.14+/3.15+ measured), recorded "
                "in the battery leg on pre-fix ones"
            )
        s = base64.b64encode(raw).decode("ascii")
        variants = [
            s,
            "  \t" + s + "\n",
            s[: max(0, len(s) - 1)],
            s + "==",
            s + "!!!",
            "!!!".join(s[i : i + 3] for i in range(0, len(s), 3)) or "!",
            "=" + s,
            s.replace("=", "=\n=", 1) if "=" in s else s + "=",
        ]
        for variant in variants:
            assert _tors_outcome(variant, False) == _stdlib_outcome(variant, False), (
                f"lenient divergence on {variant!r}"
            )

    @given(st.binary(max_size=48))
    @settings(max_examples=300)
    def test_strict_mode_equals_the_stdlib_over_the_same_shapes(self, raw: bytes) -> None:
        """The strict twin of the lenient differential: the same corrupted
        variants, where every non-alphabet byte must raise the stdlib's own
        message. Same gating as the lenient twin (3.10's regex validator and
        pre-fix machines are documented divergences, not parity targets)."""
        if not _STDLIB_FIXED:
            pytest.skip(
                "running stdlib predates the gh-145264 base64 security fix "
                "(or is 3.10's regex validator); tors ships the 3.11+ "
                "strict machine, so parity with THIS stdlib does not hold; "
                "asserted on post-fix interpreters, recorded in the battery "
                "leg on pre-fix ones"
            )
        s = base64.b64encode(raw).decode("ascii")
        variants = [
            s,
            "  \t" + s + "\n",
            s[: max(0, len(s) - 1)],
            s + "==",
            s + "!!!",
            "=" + s,
        ]
        for variant in variants:
            assert _tors_outcome(variant, True) == _stdlib_outcome(variant, True), (
                f"strict divergence on {variant!r}"
            )


class TestRoundTrip:
    @given(st.binary(max_size=96))
    @settings(max_examples=400)
    def test_round_trips_through_tors_b64_encode_bytes(self, raw: bytes) -> None:
        """The pair's own round trip: ``b64_decode(b64_encode_bytes(raw))`` is
        the identity, both strict (the encoder always emits strict-valid
        output) and lenient. Interpreter-independent (no stdlib comparison),
        so it runs ungated."""
        s = b64_encode_bytes(raw)
        assert b64_decode(s, validate=True) == raw
        assert b64_decode(s, validate=False) == raw

    @pytest.mark.parametrize("length", range(65))
    def test_every_length_up_to_64_round_trips(self, length: int) -> None:
        raw = bytes((i * 251 + 7) % 256 for i in range(length))
        s = b64_encode_bytes(raw)
        assert b64_decode(s) == raw
        if _STDLIB_FIXED:
            assert b64_decode(s) == base64.b64decode(s)


class TestArgumentContract:
    @pytest.mark.parametrize(
        "not_str",
        [b"Zm9vYg==", bytearray(b"Zm9vYg=="), memoryview(b"Zm9vYg=="), 123, None],
        ids=["bytes", "bytearray", "memoryview", "int", "none"],
    )
    def test_non_str_arguments_raise_type_error(self, not_str: object) -> None:
        """The signature is ``s: str``, the str-side counterpart of
        ``b64_encode_bytes``'s bytes-only contract. (stdlib also accepts
        bytes-likes; that spelling stays with the stdlib.)"""
        with pytest.raises(TypeError):
            b64_decode(not_str)  # type: ignore[arg-type]

    @pytest.mark.parametrize("validate", [True, False])
    def test_non_ascii_str_raises_value_error_before_any_decoding(self, validate: bool) -> None:
        """``base64.b64decode`` refuses a non-ASCII ``str`` with plain
        ``ValueError`` from ``_bytes_from_decode_data``, before binascii is
        ever reached, so tors must too: same type (NOT binascii.Error), same
        message, in both validate modes."""
        ascii_only = "string argument should contain only ASCII"
        with pytest.raises(ValueError, match=ascii_only) as excinfo:
            b64_decode("\u00f1", validate=validate)
        assert type(excinfo.value) is ValueError

    @pytest.mark.parametrize("surrogate", ["\udcff", "\udcffZg==", "Zg==\udcff"])
    @pytest.mark.parametrize("validate", [True, False])
    def test_lone_surrogates_raise_the_stdlibs_plain_value_error(
        self, surrogate: str, validate: bool
    ) -> None:
        """A ``str`` holding lone surrogates (possible in CPython, impossible
        in UTF-8) meets the stdlib's ``s.encode("ascii")`` inside
        ``_bytes_from_decode_data``, whose handler converts the encode
        failure to the SAME plain ``ValueError`` every other non-ASCII str
        gets, measured on 3.12.7 and 3.13.14, and the same on every
        supported interpreter. tors raises that ValueError at the argument
        boundary instead of letting the UTF-8 borrow surface pyo3's own
        ``UnicodeEncodeError`` (a subclass of ValueError, so naive
        ``pytest.raises(ValueError)`` would pass; the ``type(...) is``
        assertion is the actual pin)."""
        with pytest.raises(
            ValueError, match="string argument should contain only ASCII"
        ) as excinfo:
            b64_decode(surrogate, validate=validate)
        assert type(excinfo.value) is ValueError
        try:
            base64.b64decode(surrogate, validate=validate)
        except ValueError as stdlib_exc:
            assert type(stdlib_exc) is ValueError
            assert str(stdlib_exc) == str(excinfo.value)
        else:  # pragma: no cover - the stdlib raising here is measured
            pytest.fail("the running stdlib accepted a lone-surrogate str")
