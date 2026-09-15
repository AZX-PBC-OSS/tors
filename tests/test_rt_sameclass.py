"""Red-team follow-up to PR #89: hunt for OTHER members of the same bug
class the PR fixed (untrusted input read into memory without a bound, or
an allocation sized from an untrusted number), reachable from the Python
surface (``import tors_documents`` / ``tors.documents``).

The PR closed the *read* side (``read_bounded_into`` + the two-phase
``into_input``). This file probes the class's other shapes:

1. declared-page-count allocation: ``pdf_impl::link_uris`` sizes
   ``Vec::with_capacity(count)`` from ``doc.page_count()`` — whose
   primary source is the PDF's *declared* ``/Count`` (pdf_oxide
   ``document.rs::get_page_count_standard``). Pin: does the declared
   number survive to the allocation, or do the /Count validations
   (MAX_PAGES + xref-len) bound it first?
2. decompression bombs on the office lanes: office_oxide caps a part at
   512 MiB (declared-size pre-check + ``take()`` during decompression +
   post-check), anydoc at 128 MiB/entry — pin both refusals empirically
   with one small crafted docx.
3. the lying part: a zip entry that *declares* 10 MiB but inflates to
   3 GiB — pin that the oxide lane's during-decompression cap keeps RSS
   bounded and the call refuses cleanly.
4. pdf_oxide's xref stream: the flate decode is capped (256 MB/stream)
   but every decoded byte can become a ``HashMap`` entry in
   ``CrossRefTable`` (``xref.rs::add_entry``) — a ~50x amplification past
   the decode cap, reachable from every PDF entry point. Probe with an
   RLIMIT_AS belt; either a clean bounded refusal (pinned) or a
   belt-limited abort (finding).

Every crash/hang/OOM shape runs in a disposable subprocess with a
timeout and (where the shape allocates) an RLIMIT_AS belt, the
``tests/test_documents_hardening.py`` probe discipline; the parent
asserts on the child's captured outcome and folds stdout/stderr and the
child's ``/proc/self/status`` VmHWM/VmPeak into every failure message.
"""

from __future__ import annotations

import re
import struct
import subprocess
import sys
import textwrap
import zlib
from pathlib import Path

import pytest

# Every probe in this module reads /proc/self/status (VmHWM/VmPeak) inside
# its child, so on a non-Linux host the children would die on the read
# instead of the assertion failing: skip the module there, the same shape
# the hardening suite's procfs tests use.
pytestmark = pytest.mark.skipif(
    not Path("/proc/self/status").exists(),
    reason="the same-class probes read /proc VmHWM: Linux-only (issue #86)",
)

# --- artifact builders -------------------------------------------------------


def _pdf_from_objects(objects: list[bytes]) -> bytes:
    """The byte-deterministic PDF writer pattern from tests/documents.py
    (correct offsets, xref, trailer), self-contained on purpose: this file
    must survive independently of the corpus module it red-teams."""
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def _pdf_with_declared_count(count: int) -> bytes:
    """A 3-object PDF whose page tree declares an attacker-chosen /Count.
    One real page; the declared number is the only lie."""
    return _pdf_from_objects(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [3 0 R] /Count {count} >>".encode("ascii"),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
        ]
    )


def _deflate_zeros(total: int) -> tuple[bytes, int]:
    """A deflate stream that inflates to `total` zero bytes, built without
    ever holding the expansion in memory: (payload, crc32 of the expansion)."""
    compressor = zlib.compressobj(9)
    crc = 0
    chunk = bytes(1 << 20)
    remaining = total
    parts: list[bytes] = []
    while remaining > 0:
        step = min(len(chunk), remaining)
        piece = chunk[:step] if step != len(chunk) else chunk
        parts.append(compressor.compress(piece))
        crc = zlib.crc32(piece, crc)
        remaining -= step
    parts.append(compressor.flush())
    return b"".join(parts), crc & 0xFFFFFFFF


def _zip_bytes(entries: list[tuple[str, bytes, int, int, int]]) -> bytes:
    """A hand-built zip: each entry is
    ``(name, compressed_payload, method, declared_uncompressed, crc)``.
    The declared uncompressed size and the payload are independent knobs —
    exactly the lie a zip bomb needs."""
    local = bytearray()
    central = bytearray()
    offset = 0
    for name, payload, method, declared, crc in entries:
        encoded = name.encode("ascii")
        header = struct.pack(
            "<IHHHHHIIIHH",
            0x04034B50,  # local file header signature
            20,  # version needed
            0,  # flags
            method,
            0,  # time
            0,  # date
            crc,
            len(payload),
            declared,
            len(encoded),
            0,  # extra len
        )
        local += header + encoded + payload
        central += struct.pack(
            "<IHHHHHHIIIHHHHHII",
            0x02014B50,  # central directory signature
            20,  # version made by
            20,  # version needed
            0,  # flags
            method,
            0,  # time
            0,  # date
            crc,
            len(payload),
            declared,
            len(encoded),
            0,  # extra len
            0,  # comment len
            0,  # disk start
            0,  # internal attrs
            0,  # external attrs
            offset,  # local header offset
        ) + encoded
        offset += len(header) + len(encoded) + len(payload)
    eocd = struct.pack(
        "<IHHHHIIH",
        0x06054B50,
        0,
        0,
        len(entries),
        len(entries),
        len(central),
        len(local),
        0,
    )
    return bytes(local) + bytes(central) + eocd


_CONTENT_TYPES = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    b'<Default Extension="xml" ContentType="application/xml"/></Types>'
)

_RELS = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    b'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
    b'officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
    b"</Relationships>"
)


def _docx_declared_bomb(declared: int, payload: bytes, crc: int) -> bytes:
    """A docx whose word/document.xml entry declares `declared` uncompressed
    bytes but carries `payload` as its (much smaller) deflate stream. The
    parts the package reader walks first ([Content_Types].xml and the .rels
    root) are honest and tiny, so the bomb part is actually opened."""
    rels = ("_rels/.rels", _RELS, 0, len(_RELS), zlib.crc32(_RELS) & 0xFFFFFFFF)
    content_types = (
        "[Content_Types].xml",
        _CONTENT_TYPES,
        0,
        len(_CONTENT_TYPES),
        zlib.crc32(_CONTENT_TYPES) & 0xFFFFFFFF,
    )
    return _zip_bytes(
        [content_types, rels, ("word/document.xml", payload, 8, declared, crc)]
    )


def _pdf_with_xref_stream_bomb(decoded_len: int, declared_entries: int) -> bytes:
    """A PDF whose startxref points at an xref STREAM: /W [1 2 1] (4 bytes
    per entry) over a FlateDecode stream inflating to `decoded_len` zero
    bytes, with /Index declaring `declared_entries` entries. The flate cap
    bounds the decode; every decoded byte still becomes xref-table state
    (CrossRefTable::add_entry, a HashMap<u32, XRefEntry>)."""
    payload, _crc = _deflate_zeros(decoded_len)
    stream_dict = (
        f"<< /Type /XRef /Size 1 /W [1 2 1] /Index [0 {declared_entries}]"
        f" /Filter /FlateDecode /Length {len(payload)} /Root 2 0 R >>"
    ).encode("ascii")
    out = bytearray(b"%PDF-1.4\n")
    xref_obj_at = len(out)
    out += b"1 0 obj\n" + stream_dict + b"\nstream\n" + payload + b"\nendstream\nendobj\n"
    out += b"2 0 obj\n<< /Type /Catalog /Pages 3 0 R >>\nendobj\n"
    out += b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
    out += b"startxref\n" + str(xref_obj_at).encode("ascii") + b"\n%%EOF\n"
    return bytes(out)


# --- probe discipline (the hardening file's pattern) -------------------------


def _probe(code: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    """Run one probe child (this venv's own interpreter) and capture its
    outcome verbatim. The child is disposable by design."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _hwm_kb(report: str, field: str = "VmHWM") -> int:
    found = re.search(rf"{field}:\s+(\d+) kB", report)
    assert found, f"the probe did not report {field}:\n{report}"
    return int(found.group(1))


_KILLED = (-9, -6, 137, 134)


# --- 1: the declared /Count and the link walk's with_capacity ----------------


class TestDeclaredPageCountCannotSizeTheLinkWalk:
    """pdf_impl::link_uris (src/pdf_impl.rs:267) allocates
    Vec::with_capacity(doc.page_count()?) — and pdf_oxide's
    page_count trusts the page tree's DECLARED /Count first
    (document.rs::get_page_count_standard). The validations that must
    fire before the allocation: /Count > 8,388,607 (MAX_PAGES) falls
    back to a tree scan, and /Count > xref.len() falls back too. Pin:
    a declared billion pages must not reach the allocation."""

    def test_a_declared_billion_pages_cannot_abort_or_bloat_the_link_walk(self) -> None:
        pdf = _pdf_with_declared_count(2_000_000_000)
        code = f"""
from tors_documents import pdf_link_uris, pdf_page_count
try:
    count = pdf_page_count(data={pdf!r})
    print(f"PAGE_COUNT {{count}}")
except BaseException as exc:
    print(f"page_count raised {{type(exc).__name__}}: {{str(exc)[:120]}}")
try:
    links = pdf_link_uris(data={pdf!r})
    print(f"LINK_URIS ok ({{len(links)}} pages)")
except BaseException as exc:
    print(f"link_uris raised {{type(exc).__name__}}: {{str(exc)[:120]}}")
for line in open('/proc/self/status'):
    if line.startswith(('VmHWM:', 'VmPeak:')):
        import sys; sys.stdout.write(line)
"""
        done = _probe(code, timeout=60)
        assert done.returncode not in _KILLED, (
            f"the declared /Count reached the with_capacity and the process died "
            f"(rc={done.returncode}):\n{done.stdout}\n{done.stderr}"
        )
        assert done.returncode == 0, f"probe child failed:\n{done.stdout}\n{done.stderr}"
        assert "PAGE_COUNT" in done.stdout or "raised" in done.stdout
        assert "LINK_URIS" in done.stdout or "raised" in done.stdout
        # Bounded: a handful of pages' worth of allocation, not billions of
        # Vec elements (each 24 bytes). 300 MB of high-water is already
        # generous for a 3-object PDF.
        assert _hwm_kb(done.stdout) < 300_000, (
            f"the declared /Count inflated the child's peak RSS:\n{done.stdout}"
        )

    def test_count_just_under_max_pages_still_falls_back_to_the_scan(self) -> None:
        # /Count = 8_388_607 passes MAX_PAGES but fails the xref-len check
        # (3 objects): the second validation must catch what the first let
        # through. Same boundedness assertion.
        pdf = _pdf_with_declared_count(8_388_607)
        code = f"""
import sys
from tors_documents import pdf_link_uris
try:
    links = pdf_link_uris(data={pdf!r})
    print(f"LINK_URIS ok ({{len(links)}} pages)")
except BaseException as exc:
    print(f"link_uris raised {{type(exc).__name__}}: {{str(exc)[:120]}}")
for line in open('/proc/self/status'):
    if line.startswith(('VmHWM:', 'VmPeak:')):
        sys.stdout.write(line)
"""
        done = _probe(code, timeout=60)
        assert done.returncode not in _KILLED, (
            f"rc={done.returncode}:\n{done.stdout}\n{done.stderr}"
        )
        assert done.returncode == 0, f"probe child failed:\n{done.stdout}\n{done.stderr}"
        assert _hwm_kb(done.stdout) < 300_000, (
            f"a /Count just under MAX_PAGES inflated the child's peak RSS:\n{done.stdout}"
        )


# --- 2/3: the office lanes' decompression caps -------------------------------


class TestOfficeLaneDecompressionCaps:
    """office_oxide 0.1.10 caps a package part at 512 MiB three ways
    (core/opc.rs: the declared-size pre-check at :509, the
    ``take(MAX_PART_SIZE + 1)`` at :516, the post-read check at :531);
    anydoc 0.2.4 caps an entry at 128 MiB on its declared size
    (package/archive.rs:55) plus a ``take``-bounded read
    (shared/binary.rs:34). Pin both refusals with one small artifact."""

    def test_oxide_lane_refuses_a_declared_600mib_part_pre_decompression(
        self, tmp_path: Path
    ) -> None:
        # The payload is garbage (never decompressed): the declared size
        # alone must trip the refusal, like the docs' "598 KiB zip
        # declaring a 600 MiB part is refused pre-decompression".
        payload, _crc = _deflate_zeros(64 * 1024)
        crc = zlib.crc32(b"\0" * (64 * 1024)) & 0xFFFFFFFF
        docx = _docx_declared_bomb(600 * 1024 * 1024, payload, crc)
        assert len(docx) < 100 * 1024
        p = tmp_path / "bomb.docx"
        p.write_bytes(docx)
        code = f"""
from tors_documents import to_markdown
try:
    fmt, md = to_markdown({str(p)!r}, backend="oxide")
    print(f"CONVERTED {{fmt}} ({{len(md)}} chars)")
except BaseException as exc:
    print(f"{{type(exc).__name__}}: {{exc}}")
for line in open('/proc/self/status'):
    if line.startswith(('VmHWM:', 'VmPeak:')):
        import sys; sys.stdout.write(line)
"""
        done = _probe(code, timeout=60)
        assert done.returncode not in _KILLED, (
            f"the oxide lane died on a declared-size bomb (rc={done.returncode}):\n"
            f"{done.stdout}\n{done.stderr}"
        )
        assert done.returncode == 0, f"probe child failed:\n{done.stdout}\n{done.stderr}"
        # The exact refusal, pinning the MECHANISM (opc.rs:509's declared-size
        # pre-check), not just any value error:
        assert "decompression limit exceeded" in done.stdout, (
            f"expected the 512 MiB per-part refusal (office_oxide opc.rs:509), got:\n{done.stdout}"
        )
        assert "536870912" in done.stdout, (
            f"the refusal must name the 512 MiB cap:\n{done.stdout}"
        )
        assert _hwm_kb(done.stdout) < 500_000, (
            f"the declared-600-MiB part blew past the pre-decompression refusal:\n{done.stdout}"
        )

    def test_anydoc_lane_refuses_the_same_bomb_on_its_128mib_entry_cap(
        self, tmp_path: Path
    ) -> None:
        payload, _crc = _deflate_zeros(64 * 1024)
        crc = zlib.crc32(b"\0" * (64 * 1024)) & 0xFFFFFFFF
        docx = _docx_declared_bomb(600 * 1024 * 1024, payload, crc)
        p = tmp_path / "bomb.docx"
        p.write_bytes(docx)
        code = f"""
from tors_documents import to_markdown
try:
    fmt, md = to_markdown({str(p)!r})
    print(f"CONVERTED {{fmt}} ({{len(md)}} chars)")
except BaseException as exc:
    print(f"{{type(exc).__name__}}: {{exc}}")
for line in open('/proc/self/status'):
    if line.startswith(('VmHWM:', 'VmPeak:')):
        import sys; sys.stdout.write(line)
"""
        done = _probe(code, timeout=60)
        assert done.returncode not in _KILLED, (
            f"the anydoc lane died on a declared-size bomb (rc={done.returncode}):\n"
            f"{done.stdout}\n{done.stderr}"
        )
        assert done.returncode == 0, f"probe child failed:\n{done.stdout}\n{done.stderr}"
        assert "ValueError" in done.stdout, (
            "expected anydoc's 128 MiB max_entry_bytes refusal "
            f"(archive.rs:55), got:\n{done.stdout}"
        )
        assert _hwm_kb(done.stdout) < 500_000, (
            f"the declared-600-MiB part inflated the anydoc lane's peak RSS:\n{done.stdout}"
        )

    def test_oxide_lane_stays_bounded_when_the_part_lies_about_its_size(
        self, tmp_path: Path
    ) -> None:
        # The declared size is UNDER the cap (10 MiB) but the stream really
        # inflates to 3 GiB: only the during-decompression take() cap
        # (opc.rs:516) stands between the bomb and the allocator. Measured
        # mechanism: the zip reader truncates the entry at its DECLARED
        # size, so the 3 GiB stream surfaces as a corrupt-deflate refusal
        # long before the take() cap matters — the part cannot inflate
        # past its declaration, and the declaration is capped (the
        # previous test). Bounded twice over.
        expansion = 3 * 1024 * 1024 * 1024
        payload, crc = _deflate_zeros(expansion)
        assert len(payload) < 8 * 1024 * 1024, "the bomb artifact must stay small"
        docx = _docx_declared_bomb(10 * 1024 * 1024, payload, crc)
        p = tmp_path / "liar.docx"
        p.write_bytes(docx)
        code = f"""
import resource
limit = 2 * 1024 ** 3
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
from tors_documents import to_markdown
try:
    fmt, md = to_markdown({str(p)!r}, backend="oxide")
    print(f"CONVERTED {{fmt}} ({{len(md)}} chars)")
except BaseException as exc:
    print(f"{{type(exc).__name__}}: {{str(exc)[:200]}}")
for line in open('/proc/self/status'):
    if line.startswith(('VmHWM:', 'VmPeak:')):
        import sys; sys.stdout.write(line)
"""
        done = _probe(code, timeout=60)
        assert done.returncode not in _KILLED, (
            f"the lying part got past the take() cap and the process died "
            f"(rc={done.returncode}):\n{done.stdout}\n{done.stderr}"
        )
        assert done.returncode == 0, f"probe child failed:\n{done.stdout}\n{done.stderr}"
        assert "ValueError" in done.stdout, (
            f"expected a clean bounded refusal for the 3 GiB inflation, got:\n{done.stdout}"
        )
        assert _hwm_kb(done.stdout) < 1_500_000, (
            f"the 3 GiB inflation drove RSS past the take() cap's promise:\n{done.stdout}"
        )


# --- 4: the pdf_oxide xref stream's decode-to-HashMap amplification ----------


class TestXrefStreamExpansionIntoTheXrefTable:
    """The sharpest remaining shape: pdf_oxide caps a flate stream at
    256 MB (decoders/flate.rs take at :163) — but the xref-stream parser
    (xref.rs::parse_xref_stream) then turns every `entry_size` decoded
    bytes into a HashMap entry (add_entry at :117), ~40 bytes of table
    per 4 bytes of stream. /W [1 2 1] over a 250 MB decoded stream =
    ~62M entries ~= multi-GB of HashMap from a ~245 KB PDF. Probe under
    an RLIMIT_AS belt: a clean bounded refusal pins it; a belt-limited
    abort (or a multi-GB high-water) is a finding."""

    @pytest.mark.xfail(
        strict=True,
        reason="F-3, upstream pdf_oxide (github.com/yfedoseev/pdf_oxide): "
        "parse_xref_stream grows the CrossRefTable HashMap unboundedly from "
        "the decoded xref stream (flate caps the DECODE at 256 MB; the table "
        "the decoded bytes become is uncapped, ~10,000x past it) — a 245 KB "
        "PDF reaches 5.2 GB RSS or SIGABRTs. Goes XPASS (and this strict "
        "marker fails the suite, forcing this pin to be re-pointed) when an "
        "upstream release caps xref entries with fallible growth.",
    )
    def test_a_245kb_pdf_cannot_inflate_into_a_multigb_xref_table(self, tmp_path: Path) -> None:
        decoded = 250 * 1024 * 1024  # under flate.rs's 256 MB per-stream cap
        # past the decoded data's end: truncation error after the fill
        declared_entries = 100_000_000
        pdf = _pdf_with_xref_stream_bomb(decoded, declared_entries)
        assert len(pdf) < 1024 * 1024, "the bomb artifact must stay small"
        p = tmp_path / "xrefbomb.pdf"
        p.write_bytes(pdf)
        code = f"""
import resource
limit = 2 * 1024 ** 3
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
from tors_documents import pdf_page_count
try:
    count = pdf_page_count(path={str(p)!r})
    print(f"PAGE_COUNT {{count}}")
except BaseException as exc:
    print(f"{{type(exc).__name__}}: {{str(exc)[:200]}}")
for line in open('/proc/self/status'):
    if line.startswith(('VmHWM:', 'VmPeak:')):
        import sys; sys.stdout.write(line)
"""
        done = _probe(code, timeout=60)
        assert done.returncode not in _KILLED, (
            f"FINDING: a {len(pdf)}-byte PDF inflated into xref-table state past a "
            f"2 GiB address-space belt and the process died (rc={done.returncode}); "
            f"the flate cap (256 MB) bounds the decode but not the CrossRefTable "
            f"the decoded bytes become:\n{done.stdout}\n{done.stderr}"
        )
        assert done.returncode == 0, f"probe child failed:\n{done.stdout}\n{done.stderr}"
        assert "ValueError" in done.stdout or "PAGE_COUNT" in done.stdout, (
            f"expected a clean refusal or count, got:\n{done.stdout}"
        )
        assert _hwm_kb(done.stdout) < 1_500_000, (
            f"FINDING: the xref stream's decode-to-HashMap amplification drove the "
            f"child's peak RSS to {_hwm_kb(done.stdout)} kB from a {len(pdf)}-byte artifact:\n"
            f"{done.stdout}"
        )
