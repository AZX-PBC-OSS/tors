"""The PDF surface's two crash classes, pinned failing-first (the red evidence
is in the class docstrings, measured on this tree at the pin's own commit):

- bug 1, the pdf_oxide parse-depth bomb: pdf_oxide 0.3.78's object parser
  (``parse_object`` in its parser.rs) is mutually recursive with its array and
  dictionary parsers and has no syntactic depth cap (its
  ``ParserConfig::max_nesting`` field is dead code, zero uses outside
  parser_config.rs), so a ~60 KB PDF whose trailer (or any object body)
  nests ``[`` arrays ~10,659 deep exhausts the native stack and EVERY entry
  point that opens the document dies with SIGSEGV (exit -11) in ~170 ms:
  uncatchable from Python (``except BaseException`` never runs). A keyed
  dictionary bomb (``<< /K `` per level) crashes the same way; bare ``<<
  << <<`` (no key between the opens) and ``{}`` braces do not recurse (the
  lexer refuses them, catchably; measured). The pin: over-depth is a
  CATCHABLE ``ValueError`` (the taxonomy's malformed-document shape) on
  every public entry point, for arrays and keyed dicts, trailer and
  object-body placement, never signal death.
- bug 2, the ``pages=(start, stop)`` range bomb: ``parse_pages``
  (tors-documents/src/lib.rs) does ``(start..stop).collect()``,
  materializing ``stop - start`` ``usize`` entries, BEFORE anything reads
  the document or checks the page count, so ``pages=(0, 10**17)`` asks the
  allocator for ~800 PB and dies with SIGABRT (handle_alloc_error, exit -6)
  in ~110 ms on bytes that are not even a valid PDF. Measured flip point on
  this box: 10**10 (80 GiB) still survives via overcommit (wasting 7.6 s
  before a catchable parse error), 10**11 (800 GiB) aborts. The pin: an
  absurd range endpoint is a catchable ``ValueError`` naming ``pages``, with
  any bytes (even an invalid PDF), never an abort.

All crash-shaped probes run in disposable subprocesses (this venv's own
interpreter, like the hardening suite's ``_probe``) with explicit timeouts,
so the pre-fix SIGSEGV/SIGABRT can never touch the pytest runner; the parent
asserts on the child's captured outcome and folds stdout/stderr into every
failure message. The positive controls are honest in-process conversions:
the pins must survive the fix without weakening the corpus behavior.

Timing note: deliberately NOT ``timing``-marked: these are crash/catchability
pins, not wall-time bands; the file stays under ~30 s (the depth probes crash
or refuse in ~0.2 s each, the fixture PDFs are ~100 KB). Green, measured on
this tree (quiet box): every probe child refuses in ~10-20 ms end to end,
interpreter startup dominating (the depth pre-scan and the pages check are
sub-millisecond), so the 30 s backstop sits ~1500x above the green wall and
~150x above the pre-fix crash time (~170 ms): the timeouts are hang
backstops, never wall gates.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from documents import ENGINES_CORPUS
from tors.documents import (
    pdf_classify,
    pdf_extract,
    pdf_link_uris,
    pdf_page_count,
    to_markdown,
    to_text,
)

_PDF_TWO_PAGE = ENGINES_CORPUS["pdf_two_page"]
_REPORT_PDF = Path(__file__).parent / "corpus" / "report.pdf"

# Every public entry point that opens a PDF document (the docs/documents.md
# signature table): all six share pdf_impl's open() seam, so the depth bomb
# must refuse on all of them, not just the ones the reporter happened to try.
_ENTRY_POINTS = (
    to_markdown,
    to_text,
    pdf_extract,
    pdf_page_count,
    pdf_classify,
    pdf_link_uris,
)

# The death-by-signal codes a catchable refusal must never produce: SIGSEGV
# (-11), SIGABRT (-6), the SIGKILL/SIGABRT shell spellings (137/134), and
# SIGSEGV-by-way-of-139. Asserted before the rc==0 check so a red run's
# failure message says "killed by signal", not a generic child failure.
_SIGNAL_DEATH = {-6, -9, -11, 134, 137, 139}


def _probe(code: str, *args: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    """Run one probe child (this venv's own interpreter) and capture its
    outcome verbatim. The child is disposable by design: the SIGSEGV/SIGABRT
    shapes under test must never run in the pytest process. ``args`` ride as
    argv (kinds, placements, ints), never interpolated into the source."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _child_outcome(done: subprocess.CompletedProcess[str], context: str) -> str:
    """The probe child's printed one-line outcome, asserted to be a clean
    exit: the crash shapes under test kill the interpreter by signal
    (uncatchable), so a signal-death returncode is THE red/green fact and
    gets the loudest possible message before anything else is checked."""
    assert done.returncode not in _SIGNAL_DEATH, (
        f"the {context} killed the interpreter with signal "
        f"{-done.returncode if done.returncode < 0 else done.returncode} "
        f"(returncode {done.returncode}): a crash a Python caller cannot "
        f"catch. Correct behavior: the input is refused with a catchable "
        f"exception and the child exits 0.\n"
        f"child stdout:\n{done.stdout}\nchild stderr:\n{done.stderr}"
    )
    assert done.returncode == 0, (
        f"the {context} probe child failed before printing an outcome:\n"
        f"{done.stdout}\n{done.stderr}"
    )
    return done.stdout.strip()


# --- bug 1: the pdf_oxide parse-depth bomb ----------------------------------
#
# N=20000 is ~2x above this box's measured crash threshold (a binary search
# over plain trailer arrays flips between 10,658 brackets, which survive, and
# 10,659, which SIGSEGV (main-thread stack 8 MiB, ~170 ms to die), so the
# pin stays red on any stack this side of ~16 MiB while the fixture stays
# tiny (~80 KB, built in microseconds). The post-fix cost is a bounded
# pre-scan, not a deep parse: green must stay well under the probe timeout.

_DEPTH_PROBE = r"""
import sys
import tors.documents as d

kind, placement, fn = sys.argv[1], sys.argv[2], sys.argv[3]
N = 20000
if kind == "array":
    # Plain nested arrays: the reporter's shape. Two parser frames per level
    # (parse_object -> parse_array), no cap anywhere.
    payload = b"[" * N + b"]" * N
else:
    # Nested dictionaries, keyed: each nesting level needs a name between
    # the opens or the lexer never recurses (bare `<< << <<` is refused,
    # catchably; measured). parse_object -> parse_dictionary, same crash.
    payload = b"<< /K " * N + b">>" * N
if placement == "trailer":
    pdf = (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
        b"trailer\n<< /Root 1 0 R /X " + payload + b" >>\n%%EOF\n"
    )
else:
    pdf = (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /X " + payload
        + b" >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"
    )
try:
    out = getattr(d, fn)(data=pdf)
    print(f"CONVERTED {out!r}"[:120])
except BaseException as exc:  # the pin's green shape: caught, named, exit 0
    print(f"{type(exc).__name__}: {str(exc)[:200]}")
"""


class TestTheDepthBombIsACatchableRefusal:
    """Bug 1's pins, over every public entry point x placement x object kind.
    Red, measured on this tree (N=20000, each cell): the probe child died
    with returncode -11 (SIGSEGV) in ~170 ms (all six entry points, both
    placements, arrays and keyed dicts alike); the crash is upstream
    (pdf_oxide 0.3.78's uncapped parse_object/parse_array/parse_dictionary
    recursion; its max_nesting config is dead code) but the refusal belongs
    at this crate's seam (pdf_impl's shared open()): over-depth must come
    back as a CATCHABLE ValueError (the taxonomy's malformed-document shape:
    docs/documents.md, "ValueError for ... a malformed or encrypted
    document"), never signal death."""

    @pytest.mark.parametrize("placement", ["trailer", "object"], ids=str)
    @pytest.mark.parametrize("kind", ["array", "dict"], ids=str)
    @pytest.mark.parametrize("fn", _ENTRY_POINTS, ids=lambda call: call.__name__)
    def test_the_depth_bomb_refuses_catchably(self, kind: str, placement: str, fn: Any) -> None:
        done = _probe(_DEPTH_PROBE, kind, placement, fn.__name__, timeout=30.0)
        report = _child_outcome(done, f"{fn.__name__} depth bomb ({kind}, {placement})")
        assert report.startswith("ValueError"), (
            f"{fn.__name__} did not refuse the {kind} depth bomb placed in "
            f"the {placement} with a ValueError (the malformed-document shape "
            f"the error taxonomy names): {report!r}"
        )

    def test_the_lexically_rejected_shapes_stay_catchable(self) -> None:
        """The guard the kind parametrization implies: bare ``<< << <<``
        (no key between the opens) and ``{}`` braces do not recurse in
        pdf_oxide 0.3.78 (measured: both come back as catchable ValueErrors
        today), and the fix must not turn that into a refusal of VALID
        documents or a crash."""
        for shape in ("bare_dict", "braces"):
            done = _probe(_NONRECURSIVE_PROBE, shape, timeout=30.0)
            report = _child_outcome(done, f"non-recursive nesting shape {shape}")
            assert report.startswith(("ValueError", "CONVERTED")), (
                f"a lexically non-recursive nesting shape ({shape}) became a "
                f"new failure mode after the fix: {report!r}"
            )


# The lexically non-recursive nesting shapes (no key between the dict opens;
# braces are not PDF syntax at all): the guard above's fixture, showing the
# depth defense must key on real recursion depth, not on any bracket byte.
_NONRECURSIVE_PROBE = r"""
import sys
import tors.documents as d

shape = sys.argv[1]
N = 20000
if shape == "bare_dict":
    payload = b"<< " * N + b">>" * N
else:  # braces: not PDF syntax at all
    payload = b"{" * N + b"}" * N
pdf = (
    b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    b"trailer\n<< /Root 1 0 R /X " + payload + b" >>\n%%EOF\n"
)
try:
    d.pdf_page_count(data=pdf)
    print("CONVERTED")
except BaseException as exc:
    print(f"{type(exc).__name__}: {str(exc)[:120]}")
"""


# --- bug 2: the pages=(start, stop) range bomb -------------------------------

_PAGES_PROBE = r"""
import sys
import tors.documents as d

stop = int(sys.argv[1])
try:
    out = d.to_text(data=b"%PDF-1.4\n", pages=(0, stop))
    print(f"CONVERTED {out!r}"[:120])
except BaseException as exc:  # the pin's green shape: caught, named, exit 0
    print(f"{type(exc).__name__}: {str(exc)[:200]}")
"""


class TestAnAbsurdPagesRangeIsACatchableRefusal:
    """Bug 2's pins. Red, measured on this tree, two different reds: for
    stop=10**17 and stop=10**15 the probe child died with returncode -6
    (SIGABRT, handle_alloc_error) in ~110 ms, on bytes
    (``b\"%PDF-1.4\\n\"``) that are not even a valid PDF, because
    parse_pages materializes ``(start..stop).collect()`` before anything
    looks at the document; for stop=10**9 the child survived this box's
    overcommit (an 80 GiB ``Vec<usize>`` at 10**10 still survives, wasting
    7.6 s) and raised a catchable parse ``ValueError`` that does NOT name
    ``pages``, so that cell pins the message contract alone. Green
    contract: the range is validated BEFORE materialization (under the GIL
    at argument-contract time, the house rule: invalid input raises before
    the detach; or against the real page count inside the detached pass),
    and the refusal is a ValueError whose message names ``pages``, with ANY
    bytes, valid or not."""

    @pytest.mark.parametrize("stop", ["100000000000000000", "1000000000000000", "1000000000"])
    def test_the_range_bomb_refuses_catchably_naming_pages(self, stop: str) -> None:
        done = _probe(_PAGES_PROBE, stop, timeout=30.0)
        report = _child_outcome(done, f"pages=(0, 10**{len(str(int(stop))) - 1}) range bomb")
        assert report.startswith("ValueError"), (
            f"pages=(0, {stop}) did not refuse with a ValueError: {report!r}"
        )
        assert "pages" in report, (
            f"the pages= refusal must name the argument (the surface's "
            f"every-message-names-the-argument convention): {report!r}"
        )


# --- the compressed-stream blast radius --------------------------------------
#
# The raw-byte eye cannot see inside compressed streams, and pdf_oxide 0.3.78
# parses a /Type /ObjStm stream's contents with the same uncapped recursive
# parse_object (its objstm.rs:168). Measured on this tree at the pre-scan
# commit: a VALID PDF 1.5 file whose /Root catalog lives inside a
# FlateDecode object stream, bomb in the inflated object text, died with
# returncode -11 (SIGSEGV) at N=12000 and N=20000 (every entry point, same
# as the raw bomb), while the same bomb hidden in a compressed xref
# stream's DATA never crashed at all (pdf_oxide reads that data as packed
# binary rows and never recurses; it converted, pre-scan, at every N). So
# the fix inflates every /Type /ObjStm and /Type /XRef stream it passes
# (hard 256 MiB inflated ceiling, breach = the same catchable refusal) and
# runs the same depth scan over the inflated bytes; these cells pin
# the whole class green, no xfail: both shapes refuse catchably.

_COMPRESSED_PROBE = r"""
import sys
import zlib

import tors.documents as d

kind, fn = sys.argv[1], sys.argv[2]
N = 20000


def row(type_, a, b):  # one xref-stream entry, /W [1 4 2]
    return bytes([type_]) + a.to_bytes(4, "big") + b.to_bytes(2, "big")


bomb = b"[" * N + b"]" * N
out = bytearray(b"%PDF-1.7\n")
offsets = {}
if kind == "objstm":
    # The /Root catalog lives INSIDE the FlateDecode object stream: the
    # bomb rides in the inflated object text, invisible to a raw-byte scan.
    pairs = b"1 0\n"
    flow = zlib.compress(pairs + b"<< /Type /Catalog /X " + bomb + b" >>")
    offsets[2] = len(out)
    out += b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
    offsets[3] = len(out)
    out += (
        b"3 0 obj\n<< /Type /ObjStm /N 1 /First " + str(len(pairs)).encode()
        + b" /Filter /FlateDecode /Length " + str(len(flow)).encode()
        + b" >>\nstream\n" + flow + b"\nendstream\nendobj\n"
    )
    rows = row(0, 0, 65535) + row(2, 3, 0) + row(1, offsets[2], 0) + row(1, offsets[3], 0)
else:
    # A valid compressed xref stream: real entries for a valid document,
    # then the bomb as trailing bytes in the INFLATED data (binary data,
    # never object syntax; pdf_oxide converted this shape pre-fix at every N).
    offsets[1] = len(out)
    out += b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    offsets[2] = len(out)
    out += b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
    offsets[3] = len(out)
    out += b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    rows = (
        row(0, 0, 65535) + row(1, offsets[1], 0) + row(1, offsets[2], 0)
        + row(1, offsets[3], 0) + bomb
    )
# The xref stream itself (object 4) closes the file; its own entry points
# at its start, known before it is written.
xref_at = len(out)
rows += row(1, xref_at, 0)
data = zlib.compress(rows)
out += (
    b"4 0 obj\n<< /Type /XRef /Size 5 /W [1 4 2] /Root 1 0 R"
    b" /Filter /FlateDecode /Length " + str(len(data)).encode()
    + b" >>\nstream\n" + data + b"\nendstream\nendobj\nstartxref\n"
    + str(xref_at).encode() + b"\n%%EOF\n"
)

try:
    result = getattr(d, fn)(data=bytes(out))
    print(f"CONVERTED {result!r}"[:120])
except BaseException as exc:  # the pin's green shape: caught, named, exit 0
    print(f"{type(exc).__name__}: {str(exc)[:200]}")
"""


class TestCompressedStreamBombsRefuseCatchably:
    """The blast-radius probe's verdicts, pinned green (no xfail): the
    ObjStm bomb crashed pre-scan (returncode -11 at N=12000/20000, the
    class docstring's measurement) and must come back as the same catchable
    ValueError as the raw bomb; the xref-stream bomb never crashed (no
    recursion into xref data) but is still contained: a document whose
    compressed xref data hides an object-nesting bomb is refused
    catchably, not waved through."""

    @pytest.mark.parametrize(
        "fn", [pdf_page_count, to_markdown, pdf_classify], ids=lambda call: call.__name__
    )
    @pytest.mark.parametrize("kind", ["objstm", "xref"], ids=str)
    def test_a_bomb_hidden_in_a_compressed_stream_refuses_catchably(
        self, kind: str, fn: Any
    ) -> None:
        done = _probe(_COMPRESSED_PROBE, kind, fn.__name__, timeout=30.0)
        report = _child_outcome(done, f"{fn.__name__} compressed {kind} bomb")
        assert report.startswith("ValueError"), (
            f"{fn.__name__} did not refuse the {kind} bomb hidden in the "
            f"compressed stream with a ValueError: {report!r}"
        )


# --- positive controls: the fix must not cost the corpus anything -----------
# In-process (no crash shape here): the tuple-range and whole-document
# conversions are cheap and their equality is the pins that keep the
# pages= fix honest: a lazy range must convert byte-identically to the
# explicit list and to None, and the depth pre-scan must never touch a
# legitimate document.


class TestPositiveControlsTheFixMustKeep:
    def test_a_corpus_pdf_converts_with_a_tuple_range(self) -> None:
        _fmt, text = to_text(data=_PDF_TWO_PAGE, pages=(0, 1))
        assert "first page line" in text, text
        assert "second page line" not in text, (
            "the half-open range (0, 1) selected page 2: the range is not half-open"
        )

    def test_a_full_tuple_range_equals_the_explicit_list_and_the_whole(self) -> None:
        whole_md = to_markdown(data=_PDF_TWO_PAGE)[1]
        whole_text = to_text(data=_PDF_TWO_PAGE)[1]
        assert to_text(data=_PDF_TWO_PAGE, pages=(0, 2))[1] == whole_text
        assert to_text(data=_PDF_TWO_PAGE, pages=list(range(2)))[1] == whole_text
        assert to_markdown(data=_PDF_TWO_PAGE, pages=(0, 2))[1] == whole_md

    def test_the_report_corpus_pdf_still_converts_untouched(self) -> None:
        pdf = _REPORT_PDF.read_bytes()
        assert pdf_page_count(data=pdf) == 1
        whole = to_markdown(data=pdf)[1]
        assert to_markdown(data=pdf, pages=(0, 1))[1] == whole
        assert whole.strip(), "the corpus PDF converted to nothing"
