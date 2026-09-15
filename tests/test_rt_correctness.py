"""Red-team correctness probes for PR #89 (issue #80's bounded two-phase read).

Each test pins one HYPOTHESIS about the desired behavior of the new read
spine (`read_bounded_into` + `provisional_read_ceiling` + the two-phase
`Source::into_input`). A PASS pins the behavior; a FAIL is a finding.

Hypotheses under test (one test each unless noted):

- H1  Boundary pin: a file of EXACTLY 32 MiB on a metered lane (csv)
  with ``max_bytes=None`` converts; 32 MiB + 1 byte refuses with a
  ValueError naming the 32 MiB ceiling (the ceiling is `>`, not `>=`).
- H2  Splice integrity at the SNIFF_PREFIX seam: files of size
  64*1024-1, 64*1024, 64*1024+1 (plus a 64*1024+4096 variant where
  phase 2 contributes real bytes) convert with sentinels intact: one in
  the first 100 bytes, one at the tail (for 64*1024+1 the tail token's
  last byte IS the byte that crossed the phase-1 prefix ceiling — the
  `_prefix_over`-ignored over-read byte must be retained, not clipped),
  and one straddling the phase-1/phase-2 splice boundary.
- H3  An explicit budget beats the sniff in BOTH directions: a metered
  40 MiB csv with ``max_bytes=64 MiB`` succeeds end-to-end (a 32 MiB
  refusal here would make the docs' override promise FALSE); and
  ``max_bytes=1024`` refuses pre-read naming the user's budget.
- H4  Error kinds: a directory is IsADirectoryError; a missing path is
  FileNotFoundError; a sparse 1 TiB .pdf under None refuses with a
  ValueError (never signal-death); ``max_bytes=0`` is a ValueError
  naming ``max_bytes``.
- H5  Reader robustness on procfs (st_size=0, S_ISREG): a tiny explicit
  budget yields a clean ValueError, never a hang or abort.
- H6  Refusal honesty: the metered-None refusal names the ~32 MiB
  ceiling and neither refusal names the file's absolute tmp path (no
  path leakage into error messages); the explicit-budget refusal names
  the user's budget; the None-branch refusal does not fabricate a
  document size it cannot know.
- H7  (added) The data= lane keeps its own boundary: exactly 32 MiB of
  csv via ``data=`` converts under None; 40 MiB refuses with the core's
  post-read lane-ceiling message (the two-phase change must not have
  perturbed the data lane's doctrine).
- H8  (added) The provisional sniff reads CONTENT, not the name hint: a
  40 MiB csv NAMED .pdf refuses during the read at 32 MiB (message says
  "default read ceiling"), not post-read via the core's lane message —
  the read never buffers past the metered ceiling even when the name
  claims an unmetered lane.

Probe design (the hardening suite's `_probe` pattern): anything that
could hang, abort, or signal-die runs in a disposable subprocess with a
timeout, and the parent asserts on the child's captured outcome. For
verbatim-message assertions (H6, H8) the child CATCHES the exception and
prints a parseable ``RESULT|type|message`` line instead of letting the
traceback render it: a traceback echoes the probe's own source line,
which embeds the tmp path and would false-positive the no-leakage
assertions. Success-path conversions run in-process (no crash shape).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from tors_documents import Format, to_text

from documents import ENGINES_CORPUS

_PDF_BYTES = ENGINES_CORPUS["pdf_two_page"]

MIB = 1024 * 1024
_SNIFF_PREFIX = 64 * 1024  # the binding's phase-1 prefix (SNIFF_PREFIX)
_METERED_CEILING = 32 * MIB  # DEFAULT_ANYDOC_INPUT_LIMIT
_KILL_SIGNALS = (-9, -6, 137, 134)  # SIGKILL/SIGABRT/SIGKILL|SIGABRT exit codes

# Distinct ASCII tokens (lorem filler never contains them).
_EARLY = "HEADTOKEN_a1"  # placed inside the first 100 bytes
_TAIL = "TAILTOKEN_z9"  # ends at the file's final byte
_SEAM = "SEAMTOKEN_m5"  # straddles the 64 KiB splice boundary


def _probe(code: str, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    """Run one probe child (this venv's own interpreter) and capture its
    outcome verbatim. The child is disposable by design: the hang/OOM/
    signal shapes under test must never run in the pytest process."""
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# The catch-print child: the call's outcome arrives as one parseable line,
# so the parent can assert on the VERBATIM exception message (type + text)
# without the traceback noise — which matters because a traceback echoes
# the child's source line and the paths there would defeat the H6/H8
# no-leakage assertions. A signal death skips the print entirely, which is
# exactly the failure shape the rc assertion then catches.
_CATCH_TEMPLATE = """\
import sys
from tors_documents import {fn}
try:
    out = {fn}(path=sys.argv[1]{extra_kwargs})
    print("RESULT|OK|")
except BaseException as exc:
    print(f"RESULT|{type(exc).__name__}|{exc}")
"""


def _catch_probe(fn: str, path: Path, extra_kwargs: str = "", timeout: float = 120.0):
    """_probe with the catch-print child; returns (kind, message, done)."""
    code = _CATCH_TEMPLATE.replace("{fn}", fn).replace("{extra_kwargs}", extra_kwargs)
    done = _probe(code, str(path), timeout=timeout)
    found = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT|")]
    assert found, f"probe child printed no RESULT line (rc={done.returncode}):\n{done.stderr}"
    parts = found[-1].split("|", 2)
    return parts[1], (parts[2] if len(parts) > 2 else ""), done


_CSV_ROW = b"a" * 1024 + b",ok\n"  # 1027 bytes, exactly one comma per row
_CSV_HEADER = b"unit,status\n"  # one comma


def _exact_csv_bytes(total: int) -> bytes:
    """A metered-lane (csv) fixture of EXACTLY `total` bytes: every line
    carries exactly one comma (the heuristic's witness), and the long cells
    keep the anydoc conversion cheap (~10x, not the ~146x short-cell worst
    case), so the success-path pins stay fast and small in RSS."""
    rows, rem = divmod(total - len(_CSV_HEADER), len(_CSV_ROW))
    if 0 < rem < 4:  # the final row needs >= 4 bytes for b",ok\n"
        rows, rem = rows - 1, rem + len(_CSV_ROW)
    last = b"" if rem == 0 else b"a" * (rem - 4) + b",ok\n"
    data = _CSV_HEADER + _CSV_ROW * rows + last
    assert len(data) == total
    return data


def _write_exact_csv(path: Path, total: int) -> Path:
    path.write_bytes(_exact_csv_bytes(total))
    assert path.stat().st_size == total
    return path


_HTML_HEAD = (
    f"<!doctype html><html><head><title>t</title></head><body>\n<p>{_EARLY}</p>\n"
).encode()
assert len(_HTML_HEAD) < 100 and _EARLY.encode() in _HTML_HEAD[:100]
_HTML_FILLER = b"<p>lorem ipsum dolor sit amet</p>\n"  # 33 bytes


def _pad_piece(n: int) -> bytes:
    """One adjustable filler paragraph of exactly n bytes (n >= 8)."""
    assert n >= 8, f"cannot pad {n} bytes"
    return b"<p>" + b"p" * (n - 8) + b"</p>\n"


def _filler_for(room: int) -> bytes:
    """Exactly `room` filler bytes: whole rows plus one adjustable pad row."""
    if room == 0:
        return b""
    rows, rem = divmod(room, len(_HTML_FILLER))
    if 0 < rem < 8 and rows > 0:  # make room for the minimum pad row
        rows, rem = rows - 1, rem + len(_HTML_FILLER)
    return _HTML_FILLER * rows + (b"" if rem == 0 else _pad_piece(rem))


def _write_exact_html(path: Path, total: int, seam: bool) -> Path:
    """An unmetered-lane (html) fixture of EXACTLY `total` bytes with the
    sentinels placed to probe the splice: _EARLY in the first 100 bytes,
    _TAIL ending at the final byte, and (seam=True) _SEAM straddling the
    phase-1/phase-2 boundary between byte offsets 65536 and 65537."""
    tail = f"<p>{_TAIL}".encode()  # unclosed final <p>: the token ends at EOF
    pieces = [_HTML_HEAD]
    filled = len(_HTML_HEAD)
    if seam:
        seam_piece = f"<p>{_SEAM}".encode()  # 15 bytes: token at +3..+15
        seam_start = _SNIFF_PREFIX - 12  # token spans offsets 65527..65538
        room1 = seam_start - filled
        assert room1 > 0
        pieces.append(_filler_for(room1))
        pieces.append(seam_piece)
        filled = seam_start + len(seam_piece)
    room2 = total - filled - len(tail)
    assert room2 > 200, f"tail token would sit inside the first 100 bytes (room {room2})"
    pieces.append(_filler_for(room2))
    pieces.append(tail)
    data = b"".join(pieces)
    assert len(data) == total
    path.write_bytes(data)
    assert path.stat().st_size == total
    return path


class TestH1MeteredCeilingBoundary:
    def test_exact_32mib_csv_converts_under_none(self, tmp_path: Path) -> None:
        # HYPOTHESIS H1 (success half): the 32 MiB metered ceiling is `>`-shaped
        # at EVERY gate (phase-2 read `buf.len() > ceiling`, core post-read
        # `bytes.len() > limit`), so a csv of EXACTLY 32 MiB under max_bytes=None
        # converts end-to-end instead of refusing.
        p = _write_exact_csv(tmp_path / "exact.csv", _METERED_CEILING)
        fmt, text = to_text(path=str(p))
        assert fmt == Format.CSV
        assert len(text) > 0

    def test_32mib_plus_one_refuses_naming_the_32mib_ceiling(self, tmp_path: Path) -> None:
        # HYPOTHESIS H1 (refusal half): one byte past the ceiling flips the
        # phase-2 read to over=True and refuses with a ValueError whose message
        # names the 32 MiB ceiling that was actually enforced.
        p = _write_exact_csv(tmp_path / "over.csv", _METERED_CEILING + 1)
        kind, message, done = _catch_probe("to_text", p)
        assert done.returncode == 0, done.stderr
        assert kind == "ValueError", f"expected a value refusal, got {kind}: {message}"
        assert "32.0 MiB" in message, f"the ceiling was not named verbatim: {message}"


class TestH2SpliceIntegrityAtThePrefixSeam:
    @pytest.mark.parametrize("total", [_SNIFF_PREFIX - 1, _SNIFF_PREFIX, _SNIFF_PREFIX + 1])
    def test_boundary_sizes_carry_both_sentinels(self, tmp_path: Path, total: int) -> None:
        # HYPOTHESIS H2: the two-phase read is a byte-exact splice at the
        # SNIFF_PREFIX seam: no drop, duplication, or reorder across phase 1
        # (the prefix) and phase 2 (the continuation). The +1 size is the
        # sharpest: phase 1 reads exactly one byte PAST SNIFF_PREFIX (the
        # over-read byte is the file's last byte and the tail token's last
        # character); the deliberately-ignored _prefix_over must not cost
        # that byte.
        p = _write_exact_html(tmp_path / f"splice_{total}.html", total, seam=False)
        fmt, text = to_text(path=str(p))
        assert fmt == Format.HTML
        assert _EARLY in text, "the first-100-bytes sentinel was lost"
        assert _TAIL in text, "the tail sentinel was lost across the seam"

    def test_seam_token_straddling_the_phase_boundary_survives(self, tmp_path: Path) -> None:
        # HYPOTHESIS H2 (added variant): with 64 KiB + 4096 bytes, phase 2
        # contributes real bytes; a token spanning the splice boundary
        # (buffer positions 65536|65537) must arrive intact in the output.
        p = _write_exact_html(tmp_path / "splice_seam.html", _SNIFF_PREFIX + 4096, seam=True)
        fmt, text = to_text(path=str(p))
        assert fmt == Format.HTML
        assert _EARLY in text
        assert _SEAM in text, "the token straddling the splice boundary was corrupted"
        assert _TAIL in text


class TestH3ExplicitBudgetOverridesTheSniffBothWays:
    def test_40mib_csv_with_64mib_budget_converts_end_to_end(self, tmp_path: Path) -> None:
        # HYPOTHESIS H3 (raise direction): an explicit max_bytes IS the read
        # ceiling (no sniff, no 32 MiB lane default), so a metered 40 MiB csv
        # with max_bytes=64 MiB converts. If this refuses with a 32 MiB
        # mention, the documented "explicit max_bytes overrides in either
        # direction" promise is FALSE (High finding).
        p = _write_exact_csv(tmp_path / "forty.csv", 40 * MIB)
        fmt, text = to_text(path=str(p), max_bytes=64 * MIB)
        assert fmt == Format.CSV
        assert len(text) > 0

    def test_1024_budget_refuses_pre_read_naming_the_budget(self, tmp_path: Path) -> None:
        # HYPOTHESIS H3 (lower direction): max_bytes=1024 refuses PRE-read (the
        # binding's fstat gate, whose message shape names "the input ceiling is
        # N" — the core's post-read message would name the engine lane instead)
        # with the user's budget verbatim in the message.
        p = _write_exact_csv(tmp_path / "budget.csv", 40 * MIB)
        kind, message, done = _catch_probe("to_text", p, extra_kwargs=", max_bytes=1024")
        assert done.returncode == 0, done.stderr
        assert kind == "ValueError", f"expected a value refusal, got {kind}: {message}"
        assert "1024 bytes" in message, f"the explicit budget was not named: {message}"
        assert "max_bytes" in message, f"the knob was not named: {message}"
        assert "input ceiling" in message, (
            f"the refusal does not look pre-read (fstat-gate message shape): {message}"
        )


class TestH4ErrorKinds:
    def test_directory_path_is_is_a_directory_error(self, tmp_path: Path) -> None:
        # HYPOTHESIS H4a: a directory keeps its pinned EISDIR OSError through
        # the NEW read path: the two-phase reader must surface the directory
        # read's errno as the matched subclass, never a ValueError or EOF.
        with pytest.raises(OSError) as ei:
            to_text(path=str(tmp_path))
        assert isinstance(ei.value, IsADirectoryError), (
            f"wrong OSError subclass: {type(ei.value).__name__}: {ei.value}"
        )

    def test_missing_path_is_file_not_found_error(self, tmp_path: Path) -> None:
        # HYPOTHESIS H4b: a missing path is FileNotFoundError (the matched
        # subclass), not a bare OSError.
        missing = tmp_path / "no-such-file.csv"
        with pytest.raises(OSError) as ei:
            to_text(path=str(missing))
        assert isinstance(ei.value, FileNotFoundError), (
            f"wrong OSError subclass: {type(ei.value).__name__}: {ei.value}"
        )

    def test_sparse_1tib_pdf_refuses_not_signal_death(self, tmp_path: Path) -> None:
        # HYPOTHESIS H4c: a 1 TiB sparse .pdf under max_bytes=None refuses with
        # a ValueError raised after the 512 MiB backstop read — the child must
        # survive to print it (rc never a kill signal), and the refusal must
        # name the backstop it actually enforced.
        p = tmp_path / "huge.pdf"
        with open(p, "wb") as f:
            f.write(b"%PDF-1.4\n")
            f.truncate(1024**4)
        kind, message, done = _catch_probe("pdf_page_count", p)
        assert done.returncode not in _KILL_SIGNALS, (
            f"the child died by signal rc={done.returncode}: the read was not bounded"
        )
        assert kind == "ValueError", f"expected a value refusal, got {kind}: {message}"
        assert "512.0 MiB" in message, f"the enforced backstop was not named: {message}"

    def test_zero_budget_is_a_value_error_naming_max_bytes(self, tmp_path: Path) -> None:
        # HYPOTHESIS H4d: max_bytes=0 is refused as a ValueError naming the
        # max_bytes argument, under the GIL, before any file work.
        p = tmp_path / "x.pdf"
        p.write_bytes(_PDF_BYTES)
        with pytest.raises(ValueError) as ei:
            to_text(path=str(p), max_bytes=0)
        assert "max_bytes" in str(ei.value), f"the knob was not named: {ei.value}"


@pytest.mark.skipif(
    not Path("/proc/self/maps").exists(), reason="procfs is Linux-only (see issue #86)"
)
class TestH5ProcfsStSizeZero:
    def test_proc_self_maps_with_tiny_budget_is_a_clean_value_error(self) -> None:
        # HYPOTHESIS H5: procfs files are S_ISREG with st_size=0 but read
        # unboundedly until EOF; with an explicit tiny budget the read must
        # stop by BYTES READ (not the stat) and refuse with a clean ValueError
        # naming the budget — no hang, no abort, no MemoryError.
        kind, message, done = _catch_probe(
            "to_text", Path("/proc/self/maps"), extra_kwargs=', format="csv", max_bytes=64'
        )
        assert done.returncode == 0, done.stderr
        assert kind == "ValueError", f"expected a clean value refusal, got {kind}: {message}"
        assert "64 bytes" in message, f"the enforced budget was not named: {message}"


class TestH6RefusalHonesty:
    def test_metered_none_refusal_names_32mib_and_leaks_no_path(self, tmp_path: Path) -> None:
        # HYPOTHESIS H6a: the during-read refusal under None names the ~32 MiB
        # ceiling it enforced, does NOT fabricate a document size it cannot
        # know (past the cap the size is unknown), and never leaks the input's
        # absolute tmp path into the message.
        p = _write_exact_csv(tmp_path / "leak_none.csv", 40 * MIB)
        kind, message, done = _catch_probe("to_text", p)
        assert done.returncode == 0, done.stderr
        assert kind == "ValueError"
        assert "32.0 MiB" in message, f"the metered ceiling was not named: {message}"
        assert "the document is" not in message, (
            f"the refusal fabricates a size it cannot know: {message}"
        )
        assert str(tmp_path) not in message, f"path leaked into the refusal: {message}"

    def test_explicit_budget_refusal_names_budget_and_leaks_no_path(self, tmp_path: Path) -> None:
        # HYPOTHESIS H6b: the explicit-budget pre-read refusal names the USER's
        # budget (1024 bytes), the known document size, and never the path.
        p = _write_exact_csv(tmp_path / "leak_budget.csv", 40 * MIB)
        kind, message, done = _catch_probe("to_text", p, extra_kwargs=", max_bytes=1024")
        assert done.returncode == 0, done.stderr
        assert kind == "ValueError"
        assert "1024 bytes" in message, f"the user's budget was not named: {message}"
        assert "40.0 MiB" in message, f"the (stat-known) document size was not named: {message}"
        assert str(tmp_path) not in message, f"path leaked into the refusal: {message}"


class TestH7DataLaneKeepsItsOwnBoundary:
    def test_data_exact_32mib_csv_converts_under_none(self) -> None:
        # HYPOTHESIS H7 (added): the data= lane's gate is `size <= limit` and
        # the core's post-read check is `bytes.len() > limit`, so exactly 32
        # MiB via data= under None converts — the two-phase path work must not
        # have introduced an off-by-one on the lane it did not touch.
        data = _exact_csv_bytes(_METERED_CEILING)
        fmt, text = to_text(data=data)
        assert fmt == Format.CSV
        assert len(text) > 0

    def test_data_40mib_over_default_refuses_with_lane_ceiling(self) -> None:
        # HYPOTHESIS H7 (added): the data= lane's over-default refusal is the
        # core's post-read lane message (the size and the engine lane named),
        # unchanged by this PR. The ceiling check precedes the engine match,
        # so the conversion never runs (the refusal is fast despite the size).
        data = _CSV_HEADER + b"a,ok\n" * (7 * MIB)  # ~35 MiB > 32 MiB
        with pytest.raises(ValueError) as ei:
            to_text(data=data)
        message = str(ei.value)
        assert "32.0 MiB" in message and "anydoc" in message, message


class TestH8SniffReadsContentNotTheNameHint:
    def test_pdf_named_40mib_csv_refuses_during_the_read(self, tmp_path: Path) -> None:
        # HYPOTHESIS H8 (added): the provisional sniff resolves the lane from
        # CONTENT exactly as resolve() does, so a csv NAMED .pdf still gets the
        # metered 32 MiB ceiling DURING the read. The message shape is the
        # discriminator: the phase-2 refusal says "default read ceiling" and
        # neither the size nor the engine lane (the core's post-read message,
        # which would fire if the sniff had trusted the .pdf name and read all
        # 40 MiB under the 512 MiB backstop) appears.
        p = _write_exact_csv(tmp_path / "mislabeled.pdf", 40 * MIB)
        kind, message, done = _catch_probe("to_text", p)
        assert done.returncode == 0, done.stderr
        assert kind == "ValueError", f"expected a value refusal, got {kind}: {message}"
        assert "32.0 MiB" in message, f"the metered ceiling was not named: {message}"
        assert "default read ceiling" in message, (
            "refusal is not the during-read shape "
            f"(the sniff may have trusted the .pdf name): {message}"
        )
        assert "40.0 MiB" not in message and "anydoc" not in message, (
            f"the refusal came from the core's post-read lane check, not the read: {message}"
        )
