"""Resource-exhaustion pins for the ``tors.documents`` surface, red-first
(the probe evidence is in the class docstrings; the "fix 3"/"fix 4" labels
keep the original numbering).

- fix 3, the GFM strip is O(n^2) on unmatched brackets: every unescaped
  ``[`` whose match scan runs to end-of-line costs a full rescan
  (``strip_emphasis_and_links_at_depth`` -> ``parse_link`` ->
  ``find_unescaped_bracket_close`` in src/gfm_strip_impl.rs), so a line
  of N bare ``[`` is N forward scans of ~N chars. Measured on this tree
  (quiet box): 50k -> 0.76 s, 100k -> 3.0 s, 200k -> 11.9 s/12.0 s,
  400k -> 48.1 s, a clean 4x per doubling. The same loop's URL-paren scan
  (the depth counter in ``parse_link`` after the close bracket,
  src/gfm_strip_impl.rs) is a second, independent quadratic:
  ``[a](`` x N measures 1.2 s / 5.2 s / 20.9 s at 50k/100k/200k. The
  contract: ``to_text`` on bracket/paren-heavy input completes within a
  hard child deadline sized for LINEAR cost (5 s at sizes that measure
  12-21 s today), and the strip's bytes never change (the exact-output
  pins ride along so the fixer cannot buy speed with output).
- fix 4, the oxide lane inflates a zip part with no total cap:
  office_oxide 0.1.10 bounds each part at 512 MiB (declared or actual)
  but nothing bounds the total across parts, and nothing in our seam
  knows what a part inflates to. Measured on this tree: a 599 KiB docx
  whose two parts inflate to 300 MiB each (600 MiB total, each part far
  under the 512 MiB per-part cap) CONVERTS on ``backend="oxide"`` at
  ~2.1 GiB peak RSS in ~1.2 s; a 150 KiB single-part bomb (150 MiB part)
  peaks at ~1.05 GiB, ~7x the inflated bytes and ~7,000x the input bytes.
  Worse, office_oxide never validates declared against actual sizes:
  parts DECLARING 1,000 bytes that inflate to 300 MiB each convert
  identically (~2.1 GiB), so a declared-size-sum pre-scan alone cannot
  close the hole. The contract: the bomb is either refused with a
  catchable ``ValueError`` naming the ceiling/``max_bytes`` (the repo's
  documented refusal style) or completes with peak RSS under 400 MiB;
  never a signal death. Anydoc control: the same bomb is refused
  pre-decompression by that lane's 128 MiB per-entry cap (~20 MiB RSS on
  this tree, instant, naming ``max_entry_bytes``): exactly the
  boundedness the oxide lane lacks.

Crash/hang/memory probes run in subprocesses with explicit deadlines
(the same convention as tests/test_documents_hardening.py), so a
regression dies loudly in a child instead of eating the test runner;
every failure message carries the child's captured output. The bombs are
generated at runtime under tmp_path (the encrypted-PDF precedent: no
committed binary artifacts) and stream their inflated parts in 1 MiB
chunks, so the tests never hold an inflated part themselves.
"""

from __future__ import annotations

import re
import struct
import subprocess
import sys
import textwrap
import zipfile
import zlib
from pathlib import Path

import pytest
from tors_documents import to_text

from documents import ENGINES_CORPUS


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


def _probe_report(code: str, *args: str, timeout: float = 30.0) -> str:
    """_probe plus the red-pin translation: a child that outlives its
    deadline is exactly the quadratic shape under test, so the failure
    states the linear-cost contract it broke (with the deadline and the
    measured pre-fix numbers)."""
    try:
        done = _probe(code, *args, timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"probe child exceeded the {timeout:g}s linear-cost deadline: the "
            "quadratic shape under test (fix 3: measured ~12 s at "
            "200k unmatched brackets, ~21 s at 200k unclosed URL parens, 4x "
            "per doubling; linear cost is milliseconds); child output lost "
            "with the kill"
        )
    return f"rc={done.returncode}\n{done.stdout}\n{done.stderr}"


def _hwm_kb(report: str) -> int:
    """The child's peak RSS (kB) from its printed VmHWM line (ru_maxrss on
    procfs-less platforms)."""
    found = re.search(r"hwm=(\d+) kB source=(\w+)", report)
    assert found, f"the probe did not report its peak RSS:\n{report}"
    assert found.group(2) == "VmHWM", (
        f"the probe fell back to ru_maxrss (fork-inherited watermark: "
        f"unreliable; the RSS bound is Linux-procfs-shaped):\n{report}"
    )
    return int(found.group(1))


# --- fix 3: the GFM strip is linear, and its bytes never change -------------


_GFM_PROBE = r"""
import sys, time
from tors_documents import to_text

n = int(sys.argv[1])
kind = sys.argv[2]
if kind == "brackets":
    html = b"<html><body><p>" + b"[" * n + b"</p></body></html>"
    expected = ("[" * n) + "\n"
elif kind == "parens":
    html = b"<html><body><p>" + b"[a](" * n + b"</p></body></html>"
    expected = ("[a](" * n) + "\n"
elif kind == "benign":
    html = b"<html><body><p>" + b"x" * n + b"</p></body></html>"
    expected = ("x" * n) + "\n"
else:
    raise SystemExit(f"unknown kind {kind}")
t0 = time.perf_counter()
_fmt, text = to_text(data=html, format="html")
elapsed = time.perf_counter() - t0
print(f"n={n} kind={kind} elapsed={elapsed:.3f}s outlen={len(text)}")
assert text == expected, (
    f"the strip changed bytes on {kind} input: got {text[:80]!r}"
)
"""


class TestGfmStripIsLinear:
    """Fix 3's pins: bracket- and paren-heavy input costs linear time, and
    the strip's output bytes on it are exactly today's (unbalanced markup
    passes through verbatim, the module's own
    ``unbalanced_markup_passes_through_unchanged`` pin, at scale). Red,
    measured on this tree: the 200k probes need 11.9-12.0 s (unmatched
    brackets) and 20.9 s (unclosed URL parens), 4x per doubling; each
    child below is killed at its 5 s linear-cost deadline. Green,
    measured on this tree (quiet box, three child runs per cell, across
    three builds of the in-flight tree): 2.9-4.4 ms (unmatched
    brackets), 9.3-13.1 ms (unclosed URL parens), and 2.4-3.3 ms (the
    benign control) at 200k, so the 5 s deadline sits ~380-1700x above
    the post-fix cost while the pre-fix 12-21 s reds die at it (2.4-
    4.2x past the gate): the gate is placed for the linear contract
    itself, not tuned to any one box's speed or build."""

    @pytest.mark.parametrize(
        ("kind", "n"),
        [("brackets", 200_000), ("parens", 200_000)],
        ids=["unmatched-brackets", "unclosed-url-parens"],
    )
    def test_bracket_heavy_input_meets_the_linear_cost_deadline(self, kind: str, n: int) -> None:
        report = _probe_report(_GFM_PROBE, str(n), kind, timeout=5.0)
        found = re.search(r"elapsed=(\d+(?:\.\d+)?)s", report)
        assert found, f"the probe did not report its elapsed time:\n{report}"
        assert float(found.group(1)) < 5.0, (
            f"the strip took {found.group(1)}s (wall) on {n} {kind}, over the "
            f"5s linear-cost deadline (pre-fix measured ~12s/~21s at this size, "
            f"4x per doubling: find_unescaped_bracket_close rescans to "
            f"end-of-line per `[`, and parse_link's URL-paren depth scan "
            f"rescans per `]`):\n{report}"
        )

    def test_benign_input_of_the_same_length_stays_cheap(self) -> None:
        """The parity control: same-length input with no brackets converts
        in milliseconds today and must stay there (the post-fix target:
        the strip's linear pass, not a new quadratic elsewhere)."""
        report = _probe_report(_GFM_PROBE, "200000", "benign", timeout=5.0)
        found = re.search(r"elapsed=(\d+(?:\.\d+)?)s", report)
        assert found, f"the probe did not report its elapsed time:\n{report}"
        assert float(found.group(1)) < 5.0, (
            f"benign 200k-char input took {found.group(1)}s: the linear "
            f"baseline regressed:\n{report}"
        )


class TestGfmStripBytesNeverChange:
    """The byte-identity guard the fixer must hold while buying the linear
    pass: small bracket/link shapes whose current output is pinned exactly
    (in-process: milliseconds). The first case distinguishes rescan
    semantics: the outer `[` of ``[[x](y)`` has no matching close (its
    scan's depth never returns to zero) and stays literal, while the
    inner ``[x](y)`` parses; a naive re-anchor on ``]`` would pair the
    outer bracket with a later ``]`` and change the bytes."""

    def test_nested_outer_open_bracket_stays_literal(self) -> None:
        _fmt, text = to_text(data=b"<html><body><p>[[x](y)</p></body></html>", format="html")
        assert text == "[x (y)\n", repr(text)

    def test_trailing_unmatched_bracket_after_a_link(self) -> None:
        _fmt, text = to_text(
            data=b"<html><body><p>a [b](c) [d</p></body></html>", format="html"
        )
        assert text == "a b (c) [d\n", repr(text)

    def test_links_images_and_collapse_shapes(self) -> None:
        for html, expected in [
            (
                b"<html><body><p>see [handbook](https://e.com/x) and "
                b"![logo](i.png)</p></body></html>",
                "see handbook (https://e.com/x) and logo\n",
            ),
            (
                b"<html><body><p>go [https://e.com/x](https://e.com/x) now"
                b"</p></body></html>",
                "go https://e.com/x now\n",
            ),
            (
                b"<html><body><p>mail [u@e.com](mailto:u@e.com) now</p></body></html>",
                "mail u@e.com now\n",
            ),
            (
                b"<html><body><p>deep [[a](b) inner](outer)</p></body></html>",
                "deep a (b) inner (outer)\n",
            ),
            (
                b"<html><body><p>[a](b (c)) tail</p></body></html>",
                "a (b (c)) tail\n",
            ),
            (
                b"<html><body><p>trail ] alone and ]]</p></body></html>",
                "trail ] alone and ]]\n",
            ),
        ]:
            _fmt, text = to_text(data=html, format="html")
            assert text == expected, f"{html!r}: got {text!r}"


# --- fix 4: the oxide lane bounds total decompression -----------------------

_BOMB_WRAPS: dict[str, tuple[bytes, bytes]] = {
    "word/document.xml": (
        b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b"<w:body><w:p><w:r><w:t xml:space=\"preserve\">",
        b"</w:t></w:r></w:p></w:body></w:document>",
    ),
    "word/header1.xml": (
        b'<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b"<w:p><w:r><w:t xml:space=\"preserve\">",
        b"</w:t></w:r></w:p></w:hdr>",
    ),
}

_BOMB_META = (
    (
        b"[Content_Types].xml",
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        b'<Default Extension="rels" ContentType='
        b'"application/vnd.openxmlformats-package.relationships+xml"/>'
        b'<Default Extension="xml" ContentType="application/xml"/>'
        b'<Override PartName="/word/document.xml" ContentType='
        b'"application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        b'.main+xml"/>'
        b'<Override PartName="/word/header1.xml" ContentType='
        b'"application/vnd.openxmlformats-officedocument.wordprocessingml.header'
        b'+xml"/></Types>\n',
    ),
    (
        b"_rels/.rels",
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        b'<Relationship Id="rId1" Type='
        b'"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
        b' Target="word/document.xml"/></Relationships>\n',
    ),
    (
        b"word/_rels/document.xml.rels",
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        b'<Relationship Id="rId1" Type='
        b'"http://schemas.openxmlformats.org/officeDocument/2006/relationships/header"'
        b' Target="header1.xml"/></Relationships>\n',
    ),
)


def _zip_bomb_docx(path: Path, parts: list[tuple[str, int]], declare: int | None = None) -> None:
    """A valid docx container whose XML parts each inflate to their listed
    MiB (streamed in 1 MiB chunks: the test never holds an inflated part).
    ``declare=None`` writes honest zip sizes; an int makes every bomb part
    DECLARE that many uncompressed bytes while actually inflating fully
    (office_oxide 0.1.10 never checks declared against actual; measured)."""
    if declare is None:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, meta in _BOMB_META:
                archive.writestr(name.decode(), meta)
            for name, mib in parts:
                prefix, suffix = _BOMB_WRAPS[name]
                with archive.open(name, "w") as part:
                    part.write(prefix)
                    written = len(prefix)
                    target = mib * 1024 * 1024
                    chunk = b"x" * (1024 * 1024)
                    while written < target:
                        n = min(len(chunk), target - written)
                        part.write(chunk[:n])
                        written += n
                    part.write(suffix)
        return

    entries: list[tuple[bytes, bytes, int, int, int]] = []
    for name, meta in _BOMB_META:
        entries.append(
            (name, meta, zlib.crc32(meta) & 0xFFFFFFFF, len(meta), len(meta))
        )
    for name, mib in parts:
        prefix, suffix = _BOMB_WRAPS[name]
        # Streamed, never materialized: the inflated part is fed through
        # the compressor and the running CRC in 1 MiB chunks (the honest
        # path's own discipline), so the pytest process never holds a
        # 300 MiB payload. Deflate output and CRC32 are chunking-
        # invariant, so the fixture bytes are identical to a
        # whole-payload build (verified byte-for-byte when this was
        # changed).
        body_chars = mib * 1024 * 1024 - len(prefix) - len(suffix)
        co = zlib.compressobj(9, zlib.DEFLATED, -15)
        comp = bytearray()
        crc = zlib.crc32(prefix)
        comp += co.compress(prefix)
        chunk = b"x" * (1024 * 1024)
        written = 0
        while written < body_chars:
            take = min(len(chunk), body_chars - written)
            crc = zlib.crc32(chunk[:take], crc)
            comp += co.compress(chunk[:take])
            written += take
        crc = zlib.crc32(suffix, crc)
        comp += co.compress(suffix)
        comp += co.flush()
        entries.append(
            (name.encode(), bytes(comp), crc & 0xFFFFFFFF, len(comp), declare)
        )
    out = bytearray()
    offsets = []
    for nm, body, crc, csize, usize in entries:
        method = 8 if usize != len(body) else 0
        offsets.append(len(out))
        out += (
            b"PK\x03\x04"
            + struct.pack("<HHHHH", 20, 0, method, 0, 0)
            + struct.pack("<III", crc, csize, usize)
            + struct.pack("<HH", len(nm), 0)
            + nm
            + body
        )
    cd_offset = len(out)
    central = bytearray()
    for (nm, body, crc, csize, usize), off in zip(entries, offsets, strict=True):
        method = 8 if usize != len(body) else 0
        central += (
            b"PK\x01\x02"
            + struct.pack("<HHHHHH", 20, 20, 0, method, 0, 0)
            + struct.pack("<III", crc, csize, usize)
            + struct.pack("<HHHHHII", len(nm), 0, 0, 0, 0, 0, off)
            + nm
        )
    out += central
    out += (
        b"PK\x05\x06"
        + struct.pack(
            "<HHHHIIH", 0, 0, len(entries), len(entries), len(central), cd_offset, 0
        )
    )
    path.write_bytes(bytes(out))


_OXIDE_BOMB_CHILD = r"""
import resource, sys, time
from tors_documents import to_text

data = open(sys.argv[1], "rb").read()
backend = sys.argv[2]
t0 = time.perf_counter()
try:
    fmt, text = to_text(data=data, backend=backend)
    outcome = f"CONVERTED fmt={fmt} outlen={len(text)}"
except BaseException as exc:
    outcome = f"{type(exc).__name__}: {str(exc)[:400]}"
elapsed = time.perf_counter() - t0
# VmHWM, not getrusage: a forked child inherits its parent's ru_maxrss
# watermark across execve (measured: a 12 MB child of a 900 MB parent
# reports the parent's peak), so procfs is the only clean peak-RSS signal
# in a probe child. ru_maxrss is the fallback where procfs does not exist.
try:
    hwm = next(l for l in open("/proc/self/status") if l.startswith("VmHWM"))
    hwm_kb, source = int(hwm.split()[1]), "VmHWM"
except OSError:
    hwm_kb, source = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "ru_maxrss"
print(f"backend={backend} {outcome} elapsed={elapsed:.3f}s hwm={hwm_kb} kB source={source}")
"""

# The RSS ceiling the pins encode. Measured on this tree (VmHWM, read
# inside the child): the anydoc control refuses at ~19.7-20.0 MiB (the
# child's own baseline; the guard fires before any inflation), the
# oxide-lane refusals land at the same ~19.6-20 MiB baseline (the
# declared-size refusal is pre-inflation and instant; the declared-lie
# refusal streams the parts in bounded windows, 0.019 s, never holding
# one), and the corpus control converts at ~21 MiB. The 400 MiB cap
# sits ~20x above all of those and ~5x below the pre-fix red (a
# conversion at ~2.1e6 kB, two 300 MiB parts): far enough above the
# baseline-dominated measurements to be box-portable, far enough below
# the red to keep catching it.
_BOMB_RSS_CAP_KB = 400_000


def _assert_refused_or_bounded(done: subprocess.CompletedProcess[str], detail: str) -> None:
    """The fix-4 contract, shared by both bomb pins: no signal death, and
    either a typed refusal naming the ceiling or a bounded conversion."""
    report = f"rc={done.returncode}\n{done.stdout}\n{done.stderr}"
    assert done.returncode not in (-6, -9, 134, 137), (
        f"the oxide lane died by signal on the zip bomb (rc={done.returncode}) "
        f"instead of refusing or bounding ({detail}):\n{report}"
    )
    if done.returncode == 0 and "CONVERTED" in done.stdout:
        peak = _hwm_kb(done.stdout)
        assert peak < _BOMB_RSS_CAP_KB, (
            f"the oxide lane converted the zip bomb ({detail}) at {peak} kB peak "
            f"RSS (measured pre-fix: ~2.1e6 kB, ~3,600x the input); the total "
            f"decompressed across parts must be capped: refuse with a ValueError "
            f"naming the ceiling/max_bytes, or bound the conversion:\n{report}"
        )
    else:
        assert "ValueError" in done.stdout, (
            f"the oxide lane neither converted bounded nor refused with the "
            f"documented ValueError ({detail}):\n{report}"
        )
        assert re.search(r"ceiling|max_bytes|limit", done.stdout, re.IGNORECASE), (
            f"the refusal does not name the ceiling/max_bytes:\n{report}"
        )


class TestOxideLaneBoundsTotalDecompression:
    """Fix 4's pins: the oxide lane must not turn a sub-MiB zip into
    gigabytes of resident memory. Red, measured on this tree: both bombs
    below CONVERT on backend="oxide" at ~2.1 GiB peak RSS in ~1.2 s (a
    ~3,600x RSS-to-input expansion): office_oxide 0.1.10 caps each part
    at 512 MiB but nothing caps the total across parts, and our seam
    checks only the compressed input against the 32 MiB lane ceiling
    (src/documents_impl.rs), which a 600 KiB zip sails under. The
    anydoc control refuses the same bomb pre-decompression (~20 MiB RSS
    on this tree, instant, naming max_entry_bytes): the boundedness the oxide lane must
    match. Green: the call either raises a catchable ValueError naming
    the ceiling/max_bytes or completes with peak RSS under the cap;
    never a signal death (SIGABRT/SIGKILL are allocator/oom-killer
    shapes, uncatchable and pinned out). Green, measured on this tree
    (quiet box): both bombs below refuse (the honest one pre-inflation
    through the declared-size sum, the declared lie 0.019 s into
    streamed inflation, naming the 536,870,912-byte ceiling across all
    parts) at the ~20 MiB child baseline, so the 400 MiB cap is ~20x
    above every green cell and the 60 s probe timeout ~3000x above the
    slowest green wall."""

    def test_two_part_bomb_is_refused_or_bounded(self, tmp_path: Path) -> None:
        """600 MiB total across two honest 300 MiB parts (each far under
        the 512 MiB per-part cap, so that cap cannot fire): the total is
        what must be refused or bounded."""
        path = tmp_path / "bomb_2x300h.docx"
        _zip_bomb_docx(path, [("word/document.xml", 300), ("word/header1.xml", 300)])
        assert path.stat().st_size < 1024 * 1024, "the fixture lost the bomb shape"
        done = _probe(_OXIDE_BOMB_CHILD, str(path), "oxide", timeout=60.0)
        _assert_refused_or_bounded(
            done, "two honest 300 MiB parts, 600 MiB total, each under the 512 MiB per-part cap"
        )

    def test_declared_size_lies_are_not_trusted(self, tmp_path: Path) -> None:
        """The same bomb with every part DECLARING 1,000 uncompressed bytes
        while inflating to 300 MiB: office_oxide 0.1.10 never validates
        declared against actual (measured: it converts identically at
        ~2.1 GiB), so a fix that sums declared sizes alone is defeated.
        The contract is the same refusal-or-bound, enforced on the bytes
        that actually inflate."""
        path = tmp_path / "bomb_2x300l.docx"
        _zip_bomb_docx(
            path,
            [("word/document.xml", 300), ("word/header1.xml", 300)],
            declare=1000,
        )
        assert path.stat().st_size < 1024 * 1024, "the fixture lost the bomb shape"
        with zipfile.ZipFile(path) as archive:
            for name in ("word/document.xml", "word/header1.xml"):
                assert archive.getinfo(name).file_size == 1000, (
                    "the fixture lost the declared-size lie"
                )
        done = _probe(_OXIDE_BOMB_CHILD, str(path), "oxide", timeout=60.0)
        _assert_refused_or_bounded(
            done,
            "two 300 MiB parts declaring 1,000 bytes each; declared-size "
            "sums cannot see this bomb",
        )

    def test_the_anydoc_lane_stays_the_bounded_control(self, tmp_path: Path) -> None:
        """The control that must not regress: the same bomb on the auto
        lane (docx -> anydoc) is refused pre-decompression by the engine's
        128 MiB per-entry cap (instant, ~20 MiB RSS on this tree, naming
        the cap). If
        this ever converts, the anydoc lane lost its own caps."""
        path = tmp_path / "bomb_control.docx"
        _zip_bomb_docx(path, [("word/document.xml", 300), ("word/header1.xml", 300)])
        done = _probe(_OXIDE_BOMB_CHILD, str(path), "auto", timeout=60.0)
        report = f"rc={done.returncode}\n{done.stdout}\n{done.stderr}"
        assert done.returncode == 0, f"the control probe failed outright:\n{report}"
        assert "ValueError" in done.stdout, (
            f"the anydoc lane converted the bomb the engines suite pins as a "
            f"refusal (tests/test_documents_engines.py::TestZipBombRefusals):\n{report}"
        )
        assert "max_entry_bytes" in done.stdout, (
            f"the refusal does not name the engine-side cap:\n{report}"
        )
        peak = _hwm_kb(done.stdout)
        assert peak < _BOMB_RSS_CAP_KB, (
            f"the anydoc control peaked at {peak} kB refusing the bomb: the "
            f"pre-decompression guard regressed:\n{report}"
        )


class TestPositiveControls:
    """The fixes must not cost the legit path: the corpus docx converts on
    the oxide lane with bounded RSS/time and stable text."""

    def test_corpus_docx_converts_on_oxide_bounded(self, tmp_path: Path) -> None:
        path = tmp_path / "report.docx"
        path.write_bytes(ENGINES_CORPUS["docx"])
        done = _probe(_OXIDE_BOMB_CHILD, str(path), "oxide", timeout=30.0)
        report = f"rc={done.returncode}\n{done.stdout}\n{done.stderr}"
        assert done.returncode == 0, f"the corpus docx failed on the oxide lane:\n{report}"
        assert "CONVERTED" in done.stdout, report
        found = re.search(r"elapsed=(\d+(?:\.\d+)?)s", done.stdout)
        assert found and float(found.group(1)) < 5.0, (
            f"the corpus docx took too long on the oxide lane "
            f"(measured ~1 ms on this tree, so the 5 s gate is ~5000x "
            f"above it):\n{report}"
        )
        peak = _hwm_kb(done.stdout)
        assert peak < 150_000, (
            f"the corpus docx peaked at {peak} kB on the oxide lane "
            f"(measured ~21,000 kB on this tree, only ~1-2 MB over the "
            f"child's import baseline, so the 150,000 kB gate is ~7x "
            f"above it: room for box-to-box baseline variance, still "
            f"catches any fix that buffers at the 512 MiB ceiling's "
            f"scale); the cap machinery is taxing legit input:\n{report}"
        )

    def test_corpus_docx_text_is_stable_through_oxide(self, tmp_path: Path) -> None:
        resolved, markdown = to_text(
            data=ENGINES_CORPUS["docx"], backend="oxide", format="docx"
        )
        assert resolved.value == "docx"
        assert "Quarterly Review Q3 2026" in markdown
