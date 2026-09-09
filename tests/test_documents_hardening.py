"""The PR #27 red-team follow-up: the pyo3 binding layer's input-side
hardening, each fix pinned red-green (the probe evidence is in the class
docstrings; the fix numbers match the follow-up's numbering).

- FIX 1, the unbounded ``std::fs::read`` in ``Source::into_input``: a
  ``path=`` naming a FIFO blocks forever inside open(2) with the GIL
  released, ``/dev/zero`` reads unboundedly at memory-bandwidth speed
  (``max_bytes=`` never protects — the ceiling ran only in tors-core,
  only AFTER the read), and an oversized regular file is fully resident
  before the refusal. The fix: non-regular path kinds are typed
  ``ValueError`` refusals before the read, and an EXPLICIT ``max_bytes``
  is binding on every lane, checked before a byte is read or copied.
  That second half is a deliberate CONTRACT CHANGE — today the pdf/HTML
  lanes silently ignore the knob (``max_bytes=1`` on a real PDF
  converts) — and the engines suite's old pin of that behavior
  (``test_max_bytes_does_not_apply_to_the_non_anydoc_lanes``) is expected
  to fail until the wave-2 pass repins it; THIS file carries the
  new-contract pins. Directories keep their pinned ``OSError``
  (``IsADirectoryError``), and ``max_bytes=None`` keeps the
  default-ceiling doctrine exactly (post-read, anydoc/oxide lanes only).
- FIX 2, the ``data=`` TypeError reprs the rejected value: a 1 MiB
  bytearray produced a ~4 MB exception message (document content
  disclosed into logged tracebacks) and a memoryview repr leaked a raw
  heap address. The refusal now names the type only, the ``password=``
  doctrine.
- FIX 3, a NUL inside ``path`` raised a plain ``OSError`` where CPython's
  own ``open("a\\0b")`` raises ``ValueError: embedded null byte`` — now a
  ``ValueError`` naming ``path``.
- FIX 4, ``to_markdown(123)`` surfaced os.fspath's bare ``TypeError`` —
  no argument named, against the surface's every-message-names-the-
  argument convention. The typed wrappers' coercion now refuses naming
  ``path``.
- FIX 5, the ``data=`` copy ran under the GIL: a 400 MB upload was a
  ~78 ms GIL-held memcpy (measured 2026-09-09, the heartbeat probe
  below). The copy now runs inside ``py.detach`` — pinned by the same
  probe (max heartbeat gap collapses), with peak RSS held roughly
  unchanged (the same two copies, different side of the detach: the win
  is GIL residency, not memory).
- FIXES 6/7, docstring honesty: the aio modules must state the
  uncancellable ``asyncio.to_thread`` semantics, and ``sniff``'s
  "never opens a parser" claim must become the true cost (anydoc's
  detect parses the package containers — a 120 KiB zip measured 267 MiB
  peak RSS to answer docx).

Crash/hang probes run in SUBPROCESSES with timeouts and (where the
pre-fix shape reads without bound) an RLIMIT_AS belt, so a regression
dies loudly in a child instead of eating the test runner; the parent
asserts on the child's captured outcome and folds stdout/stderr into
every failure message, so the red evidence rides in the report.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from tors_documents import (
    pdf_classify,
    pdf_extract,
    pdf_link_uris,
    pdf_page_count,
    to_markdown,
    to_text,
)

from documents import ENGINES_CORPUS

_PDF_BYTES = ENGINES_CORPUS["pdf_two_page"]
# The oversized-file probe's fixture: csv rows, so the PRE-fix behavior is
# the core's own post-read ceiling refusal (the anydoc lane's) — the red
# that proves the file was read before the refusal is the child's VmHWM,
# not the absence of a refusal.
_OVERSIZE_ROWS = 16_700_000  # b"unit,status\n" * rows ~= 191 MiB


def _probe(code: str, *args: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    """Run one probe child (this venv's own interpreter) and capture its
    outcome verbatim. The child is disposable by design: the hang/OOM
    shapes under test must never run in the pytest process. ``args`` ride
    as argv (paths and such), never interpolated into the source."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _run_or_fail(code: str, *args: str, timeout: float = 30.0) -> str:
    """_probe plus the timeout translation: a child that blocks past the
    deadline is exactly the pre-fix hang under test, so it fails the test
    SAYING that (with the deadline), instead of erroring opaquely."""
    try:
        done = _probe(code, *args, timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"probe child blocked past the {timeout:g}s deadline — the "
            "forever-shape under test (pre-fix: open(2)/read unbounded); "
            "child output lost with the kill"
        )
    assert done.returncode == 0, f"probe child failed:\n{done.stdout}\n{done.stderr}"
    return done.stdout


def _hwm_kb(report: str) -> int:
    """The child's VmHWM (peak RSS, kB) from its printed /proc status line."""
    found = re.search(r"VmHWM:\s+(\d+) kB", report)
    assert found, f"the probe did not report VmHWM:\n{report}"
    return int(found.group(1))


# --- fix 1a: non-regular path kinds are typed refusals, never open(2)'d ----


# The device-lane probe: /dev/zero under a 2 GiB RLIMIT_AS, so the pre-fix
# unbounded read dies at the limit (MemoryError) instead of eating the box
# at memory-bandwidth speed — and the post-fix refusal needs neither.
_DEV_ZERO_PROBE = r"""
import resource
limit = 2 * 1024 ** 3
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
from tors_documents import to_text
try:
    out = to_text(path="/dev/zero", max_bytes=65536)
    print(f"CONVERTED {out[0]}")
except BaseException as exc:
    print(f"{type(exc).__name__}: {exc}")
"""


class TestNonRegularPathsAreTypedRefusals:
    """Fix 1a's pins: FIFOs and devices are valid PATHS of unusable input
    KIND — a value refusal (ValueError) naming ``path`` and the kind,
    raised before open(2)/read can begin, on every path-taking entry
    point (the shared source spine). Red, measured 2026-09-09 on this
    tree: the /dev/zero child died at its 2 GiB rlimit with
    ``MemoryError: out of memory`` (the read is unbounded, ~6 GB/s), and
    the no-writer FIFO child blocked in open(2) until the harness killed
    it (the ``timeout(1)``-equivalent exit 124)."""

    def test_dev_zero_is_refused_before_the_read_not_after_the_memory(self) -> None:
        report = _run_or_fail(_DEV_ZERO_PROBE, timeout=30)
        assert report.startswith("ValueError"), (
            f"/dev/zero was not a typed refusal: {report!r}"
        )
        assert "path must name a regular file" in report, report
        assert "character device" in report, report

    def test_dev_zero_without_a_budget_is_refused_the_same(self) -> None:
        """The kind refusal is not budget machinery: with no max_bytes at
        all, /dev/zero is the same before-the-read refusal (the pre-fix
        read was unbounded with or without a budget — the ceiling only
        ever ran post-read)."""
        code = _DEV_ZERO_PROBE.replace(
            'to_text(path="/dev/zero", max_bytes=65536)', 'to_text(path="/dev/zero")'
        )
        report = _run_or_fail(code, timeout=30)
        assert report.startswith("ValueError"), f"/dev/zero was not refused: {report!r}"
        assert "character device" in report, report

    def test_dev_null_is_the_in_process_refusal_pin(self) -> None:
        """/dev/null EOFs immediately (no hang either way), so the message
        is pinnable in-process — and the PDF-only lane shares the spine,
        so its refusal is the same message."""
        for call in (
            to_markdown,
            to_text,
            pdf_classify,
            pdf_extract,
            pdf_page_count,
            pdf_link_uris,
        ):
            with pytest.raises(ValueError, match="path must name a regular file") as raised:
                call("/dev/null")
            assert "character device" in str(raised.value)

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO")
    def test_a_fifo_with_no_writer_is_a_refusal_not_a_forever_block(self, tmp_path: Path) -> None:
        """Red: the child blocked in open(2) past the harness deadline
        (exit-124 shape) — open for read blocks until a writer appears,
        with the GIL released and the thread unreclaimable. Green: the
        kind is refused at the stat, before open."""
        fifo = tmp_path / "feed.pdf"
        os.mkfifo(fifo)
        code = r"""
import sys
from tors_documents import to_text
try:
    out = to_text(path=sys.argv[1])
    print(f"CONVERTED {out[0]}")
except BaseException as exc:
    print(f"{type(exc).__name__}: {exc}")
"""
        report = _run_or_fail(code, str(fifo), timeout=20)
        assert report.startswith("ValueError"), f"the FIFO was not a typed refusal: {report!r}"
        assert "path must name a regular file" in report, report
        assert "FIFO" in report, report

    def test_directories_and_missing_paths_keep_their_pinned_oserror(self, tmp_path: Path) -> None:
        """The guard around fix 1a: directories deliberately keep the
        pinned environment-failure shape (OSError — IsADirectoryError on
        Linux, the engines suite's pin), and a missing path stays
        FileNotFoundError; the kind refusal must not eat either."""
        with pytest.raises(OSError):
            to_markdown(str(tmp_path))
        with pytest.raises(OSError):
            to_markdown(str(tmp_path / "nope.pdf"))
        if sys.platform == "linux":
            with pytest.raises(IsADirectoryError):
                to_markdown(str(tmp_path))
            with pytest.raises(FileNotFoundError):
                pdf_page_count(str(tmp_path / "nope.pdf"))

    def test_a_directory_with_a_tiny_explicit_budget_keeps_the_eisdir(self, tmp_path: Path) -> None:
        """The budget gate shares the directory doctrine the kind gate has:
        a directory is a KIND refusal, not a SIZE one, so an explicit tiny
        max_bytes never pre-empts the documented IsADirectoryError with the
        ceiling ValueError. Red, measured 2026-09-09 on this tree: both
        lanes raised "the document is 40 bytes and the input ceiling is 1
        bytes" — the tmpdir's stat size beat the budget, and the read's own
        EISDIR never got to run."""
        for call in (pdf_extract, to_markdown):
            if sys.platform == "linux":
                with pytest.raises(IsADirectoryError):
                    call(str(tmp_path), max_bytes=1)
            else:
                with pytest.raises(OSError):
                    call(str(tmp_path), max_bytes=1)


# --- fix 1b: an explicit max_bytes binds every lane, before the read -------


class TestAnExplicitBudgetBindsEveryLane:
    """Fix 1b's pins — the deliberate contract change. Red, measured
    2026-09-09 on this tree: ``to_markdown(path=<real pdf>, max_bytes=1)``
    and the data-lane twin both CONVERTED (the pdf lane never saw the
    knob). Green: the same calls refuse with the ceiling ValueError,
    before a byte is read (path) or copied (data). The old engines pin
    of the converted shape is the wave-2 agent's to repin."""

    def test_the_pdf_path_lane_refuses_under_an_explicit_budget(self, tmp_path: Path) -> None:
        path = tmp_path / "two.pdf"
        path.write_bytes(_PDF_BYTES)
        with pytest.raises(ValueError, match="ceiling") as raised:
            to_markdown(str(path), max_bytes=1)
        message = str(raised.value)
        assert "max_bytes" in message, message

    def test_the_pdf_data_lane_refuses_under_an_explicit_budget(self) -> None:
        """The engines suite's repinned shape, carried here: the in-memory
        twin of the path-lane pin (an explicit budget is binding wherever
        the bytes come from)."""
        with pytest.raises(ValueError, match="ceiling"):
            to_markdown(data=_PDF_BYTES, max_bytes=1)

    def test_the_html_lane_refuses_under_an_explicit_budget(self) -> None:
        with pytest.raises(ValueError, match="ceiling"):
            to_text(data=b"<html><body><p>budgeted</p></body></html>", max_bytes=4)

    def test_a_budget_the_input_fits_under_still_converts(self, tmp_path: Path) -> None:
        """The budget refuses nothing it should not: a roomy explicit
        max_bytes converts the same PDF and HTML inputs on the lanes that
        the pre-fix shape ignored the knob on."""
        path = tmp_path / "two.pdf"
        path.write_bytes(_PDF_BYTES)
        resolved, markdown = to_markdown(str(path), max_bytes=10_000_000)
        assert resolved == "pdf"
        assert "first page line" in markdown
        resolved, text = to_text(
            data=b"<html><body><p>budgeted</p></body></html>", max_bytes=10_000
        )
        assert resolved == "html"
        assert "budgeted" in text

    def test_no_budget_keeps_the_default_ceiling_doctrine_exactly(self, tmp_path: Path) -> None:
        """max_bytes=None is the untouched default: the pdf/HTML lanes
        stay unmetered (the 32 MiB default is the core's post-read check,
        on the anydoc/oxide lanes only — the lane is unknowable before
        the container sniff, which is exactly why only the EXPLICIT
        budget can be pre-read)."""
        path = tmp_path / "two.pdf"
        path.write_bytes(_PDF_BYTES)
        resolved, markdown = to_markdown(str(path))
        assert resolved == "pdf"
        assert "first page line" in markdown

    def test_an_oversized_regular_file_is_refused_before_it_is_read(self) -> None:
        """The pre-read half of fix 1b, proven two ways: by the MESSAGE
        (red, verbatim: the CORE's own refusal — "the document is
        191.1 MiB and the anydoc engine lane's input ceiling is 65536
        bytes" — a message only READ bytes can produce, arriving 0.077s
        in; green: the binding's pre-read refusal names the knob, not a
        lane) and by the child's VmHWM under a CHUNKED-built fixture (the
        red probe's first cut built the fixture as one 200 MB bytes
        object, which masked the read in the high-water mark; chunked,
        the read is the only 200 MB allocation the call can make, so a
        green child never materializes the file)."""
        code = rf"""
import os, tempfile, time
d = tempfile.mkdtemp()
p = os.path.join(d, "big.csv")
row = b"unit,status\n"
chunk = row * 500_000  # 6 MB: the fixture build never holds the whole file
full, rem = divmod({_OVERSIZE_ROWS}, 500_000)
with open(p, "wb") as f:
    for _ in range(full):
        f.write(chunk)
    f.write(row * rem)
from tors_documents import to_text
t0 = time.perf_counter()
try:
    out = to_text(path=p, max_bytes=65536)
    print(f"CONVERTED {{out[0]}}")
except ValueError as exc:
    elapsed = time.perf_counter() - t0
    hwm = next(l for l in open("/proc/self/status") if l.startswith("VmHWM"))
    print(f"ValueError: {{exc}}")
    print(f"elapsed={{elapsed:.3f}}s {{hwm.strip()}}")
finally:
    os.unlink(p)
    os.rmdir(d)
"""
        report = _run_or_fail(code, timeout=120)
        assert report.startswith("ValueError"), f"the oversized file was not refused: {report!r}"
        assert "ceiling" in report, report
        assert "binding on every engine lane" in report, (
            f"the refusal is the core's POST-read message (the pre-fix shape — "
            f"the file was read first):\n{report}"
        )
        peak = _hwm_kb(report)
        assert peak < 100_000, (
            f"the file was read before the refusal (VmHWM {peak} kB ~ the "
            f"whole file resident — the pre-fix shape):\n{report}"
        )


# --- fix 2: the data= refusal carries the type, never the content ----------


class TestDataRefusalNamesTheTypeOnly:
    """Fix 2's pins. Red, measured 2026-09-09:
    ``to_text(data=bytearray(1 MiB))`` refused with a 4,194,367-char
    message — the bytearray's full repr, i.e. the document content it
    carries, riding into every logged traceback — and the memoryview
    refusal embedded ``<memory at 0x7f0f97b474c0>`` (a raw heap address).
    Green: the message names the type only, the password= doctrine
    (data= is the other content-bearing argument)."""

    def test_a_huge_bytearray_refusal_stays_small(self) -> None:
        with pytest.raises(TypeError, match="data must be bytes") as raised:
            to_text(data=bytearray(1024 * 1024))
        message = str(raised.value)
        assert len(message) < 300, (
            f"the refusal message is {len(message)} chars — the rejected "
            f"value's content rode into the exception: {message[:120]!r}"
        )
        assert "bytearray" in message, message

    def test_a_memoryview_refusal_leaks_no_heap_address(self) -> None:
        with pytest.raises(TypeError, match="data must be bytes") as raised:
            to_text(data=memoryview(b"secret-document-bytes" * 8))
        message = str(raised.value)
        assert "memoryview" in message, message
        assert "0x" not in message, message

    def test_the_pdf_family_shares_the_small_refusal(self) -> None:
        with pytest.raises(TypeError, match="data must be bytes") as raised:
            pdf_extract(data=bytearray(1024 * 1024))
        assert len(str(raised.value)) < 300


# --- fix 3: a NUL in the path is a ValueError, CPython's convention --------


class TestNulInPathIsAValueError:
    """Fix 3's pins. Red, measured 2026-09-09: the path reached the
    filesystem layer and surfaced as a plain
    ``OSError('file name contained an unexpected NUL byte')`` — where
    CPython's own ``open("a\\0b")`` raises ``ValueError: embedded null
    byte``. Green: the binding refuses at parse time, a ValueError
    naming ``path`` (the message convention) and repr'ing the value."""

    def test_the_refusal_matches_cpythons_class_and_names_path(self) -> None:
        for call in (to_markdown, to_text, pdf_page_count, pdf_classify):
            with pytest.raises(ValueError, match="(?i)null") as raised:
                call("feed\x00name.pdf")
            message = str(raised.value)
            assert "path" in message, message
            assert "feed" in message, message  # the repr of the value rides in

    def test_cpython_itself_refuses_the_same_path_with_valueerror(self) -> None:
        """The convention this fix aligns to, pinned from the reference
        implementation's own behavior (not this package's)."""
        with pytest.raises(ValueError, match="null"):
            open("a\x00b")  # the refusal IS the point


# --- fix 4: the path coercion names the argument ----------------------------


class TestPathCoercionNamesTheArgument:
    """Fix 4's pins. Red, measured 2026-09-09: ``to_markdown(123)``
    raised os.fspath's bare ``TypeError: expected str, bytes or
    os.PathLike object, not int`` — no argument named, against the
    surface's own every-message-names-the-argument convention (the
    native parse_path refuses non-str paths naming ``path``; the bare
    error came from the typed wrappers' own os.fspath coercion, before
    the native layer could see the value). Green: the coercion refuses
    naming ``path``, on every entry point, while PathLike inputs keep
    working."""

    @pytest.mark.parametrize(
        "call",
        [to_markdown, to_text, pdf_classify, pdf_extract, pdf_page_count, pdf_link_uris],
        ids=lambda call: call.__name__,
    )
    def test_an_int_path_is_a_type_error_naming_path(self, call: Any) -> None:
        with pytest.raises(TypeError) as raised:
            call(123)
        message = str(raised.value)
        assert "path" in message, message
        assert "123" in message, message

    def test_pathlike_inputs_still_convert_identically(self, tmp_path: Path) -> None:
        """The coercion fix must not narrow the accepted shapes: a
        pathlib.Path converts byte-identically to its str spelling (the
        typed surface's annotated contract)."""
        path = tmp_path / "two.pdf"
        path.write_bytes(_PDF_BYTES)
        assert to_markdown(path) == to_markdown(str(path))


# --- fix 5: the data= copy runs inside the detach ---------------------------


_HEARTBEAT_PROBE = r"""
import threading, time
# 400 MB of HTML whose whole body is a <script> the engine drops by
# construction: the OUTPUT is 5 bytes, so the probe's GIL window is the
# input side alone (the copy), not an O(output) return marshalling.
data = b"<html><body><p>tiny</p><script>" + b"A" * (400 * 1024 * 1024) \
    + b"</script></body></html>"
from tors_documents import to_text

stop = threading.Event()
gaps = []
def monitor():
    last = time.perf_counter()
    while not stop.is_set():
        time.sleep(0.001)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now
# the monitor starts AFTER the fixture is built: b"A" * n and the concat
# are themselves GIL-held memcpys, and the probe measures the CALL.
t = threading.Thread(target=monitor)
t.start()
t0 = time.perf_counter()
fmt, out = to_text(data=data)
wall = time.perf_counter() - t0
stop.set()
t.join()
hwm = next(l for l in open("/proc/self/status") if l.startswith("VmHWM"))
print(f"resolved={fmt} output_len={len(out)} wall={wall:.2f}s max_gap={max(gaps) * 1000:.1f}ms")
print(hwm.strip())
"""


@pytest.mark.skipif(
    not Path("/proc/self/status").exists(), reason="the VmHWM half of the probe is Linux-shaped"
)
def test_a_400mb_data_call_keeps_the_gil_at_heartbeat_granularity() -> None:
    """Fix 5's pin. Red, measured 2026-09-09 on this tree: the probe's max
    heartbeat gap was 77.6 ms (p99 1.1 ms) — the 400 MB ``data=`` copy
    ran under the GIL in parse_source, and a monitor thread pinging every
    1 ms starved for the whole memcpy. Green: the copy runs inside
    py.detach and the gap collapses to the ping floor. Peak RSS is held
    as a PARITY band, not a win: the same two copies exist before and
    after (the caller's bytes plus the engine-bound Vec), just on
    different sides of the detach — red measured VmHWM 838,328 kB, and
    the green run must stay in the same neighborhood (the win this fix
    buys is GIL residency, not memory)."""
    report = _run_or_fail(_HEARTBEAT_PROBE, timeout=180)
    found = re.search(r"max_gap=(\d+(?:\.\d+)?)ms", report)
    assert found, f"the probe did not report its max gap:\n{report}"
    max_gap_ms = float(found.group(1))
    assert "output_len=5" in report, (
        f"the probe's fixture changed shape (the output is no longer tiny, "
        f"so the window is no longer the copy alone):\n{report}"
    )
    assert max_gap_ms < 30.0, (
        f"the GIL was held {max_gap_ms:.1f} ms through a 400 MB data= call — "
        "the copy ran GIL-side (pre-fix measured 77.6 ms); it belongs inside "
        "the detach"
    )
    peak = _hwm_kb(report)
    assert peak < 1_600_000, (
        f"peak RSS grew to {peak} kB — a third copy appeared (the fix moves "
        "the existing one, it must not add one):\n{report}"
    )


# --- fixes 6/7: docstring honesty -------------------------------------------


class TestDocstringHonesty:
    """The documentation fixes, pinned as contract: the aio modules must
    state the cancellation semantics a caller can be hurt by, and
    sniff's cost claim must be the measured truth."""

    def test_the_aio_modules_state_the_uncancellable_thread_hop(self) -> None:
        """Fix 6: asyncio.to_thread cannot cancel the native pass — the
        thread runs to completion holding its memory, and repeated
        wait_for timeouts pin the shared default executor. Red: neither
        aio module's docstring mentioned cancellation at all."""
        from tors_documents import aio as payload_aio

        from tors.documents import aio as shim_aio

        for module in (payload_aio, shim_aio):
            doc = module.__doc__ or ""
            assert "cancel" in doc.lower(), (
                f"{module.__name__} must document the uncancellation semantics"
            )
            assert "to_thread" in doc, f"{module.__name__}: name the mechanism"

    def test_sniffs_docstring_tells_the_true_container_cost(self) -> None:
        """Fix 7: sniff claimed "never opens a parser: the markers are
        container-level", but anydoc's detect opens the ZIP/OLE package
        and parses its metadata (and the main part when the markers need
        it) — a 120 KiB zip measured 267 MiB peak RSS to answer docx.
        The docstring must not carry the false claim, and must name the
        package parse it actually does."""
        from tors_documents._tors_documents import sniff as native_sniff

        doc = native_sniff.__doc__ or ""
        assert "never opens a parser" not in doc, (
            "sniff's docstring still carries the false bounded-scan claim"
        )
        assert "package" in doc.lower(), "the docstring must name the package parse it does"
