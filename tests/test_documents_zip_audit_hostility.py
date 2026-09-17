"""The oxide zip-audit hostility pins: the audit that gates
``backend="oxide"`` decides a container's fate from its central
directory, and every field of those end-of-directory records is
attacker-controlled. This file pins the divergence families the audit
was rewritten to close, each red-first where the pre-fix behavior was
measured on this tree (VmHWM read inside the probe children, quiet box,
2026-09-16), plus the refusal shapes the audit must keep refusing and
the benign shapes it must keep converting.

The three measured pre-fix bypasses (each a ~600 KiB docx that CONVERTED
on ``backend="oxide"`` at ~2.13 GiB peak RSS in ~1.3 s, 600 MiB inflated
across two 300 MiB parts, both far under office_oxide's own 512 MiB
per-part cap; post-fix each refuses in ~0.02-0.05 s at the ~18 MiB child
baseline):

- **the count fields**: the engine's zip reader counts central-directory
  records from the EOCD's *on-this-disk* field (zip 8.6's
  CentralDirectoryInfo reads that field, not the total-entries field),
  while the audit counted from the total. An EOCD declaring 1 total
  entry against 5 on-this-disk passed the audit on the one small record
  while the engine walked all five, two of them bombs. The reverse, a
  total of 0 against 5 on-this-disk, passed the audit as an *empty*
  archive while the engine's early-accept walked the five records.
- **the trailing directory**: a total of 0 makes the engine accept the
  EOCD with no first-record search, its directory start clamped to the
  declared offset (saturating arithmetic), so records planted AFTER the
  end record are walked by the engine and were never walked by an audit
  that returned an empty pass at the sight of a zero count.
- **the zip64 locator fallback**: a parseable EOCD64 locator puts the
  engine's candidate on a HARD zip64 branch: it searches forward from
  the locator's declared record offset and rejects the whole end record
  when no EOCD64 record is found, falling back to an EARLIER end-record
  signature. The audit treated zip64 as optional enrichment (zip32
  values stand when the record does not resolve), so a last end record
  with a sentinel cd-size, a parseable locator pointing at no record,
  and honest small zip32 values passed the audit while the engine fell
  back to the earlier, honest, two-bomb end record.

The rewrite mirrors the reader's resolution field for field (count from
on-this-disk unioned with the total, the empty-archive early accept, the
first-signature search and its prepend delta, the hard zip64 branch and
its arithmetic) and UNIONS the records of every end-record candidate
that resolves, not just the one the reader would pick: a reader that
rejects a candidate for any reason this mirror does not reproduce (a
known extra field whose content parser errors is the residual family)
falls back to an earlier signature whose records are already in the
union, already bounded. Two more functional bugs the rewrite closed are
pinned here as conversions: the prepend search anchored on the
signature's last byte and then matched AT it (a match no real signature
can satisfy), so every prepended container died unauditable, and a
record walk bounded by the end record's position instead of the end of
file rejected the trailing-directory shape the engine accepts.

The remaining pins are the audit's standing refusals (encrypted parts,
unsupported compression methods, unreadable local headers, degenerate
directories) and its standing conversions (a legitimate zip64 archive,
a prepended container, an embedded zip stored as a part, two entries
sharing one local header): the shapes a real writer can produce must
keep converting while the hostile ones stay refused.
"""

from __future__ import annotations

import struct
import subprocess
import sys
import zlib
from functools import lru_cache
from pathlib import Path

import pytest
from tors_documents import to_text

# The RSS ceiling these pins encode, the same number the resource-bounds
# file carries: ~20x above the ~19-20 MiB refusal/conversion baseline of
# a probe child, ~5x below the measured pre-fix red (~2.1e6 kB).
_BOMB_RSS_CAP_KB = 400_000

_META = (
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
_WRAPS = {
    b"word/document.xml": (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b"<w:body><w:p><w:r><w:t>",
        b"</w:t></w:r></w:p></w:body></w:document>",
    ),
    b"word/header1.xml": (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        b'<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b"<w:p><w:r><w:t>",
        b"</w:t></w:r></w:p></w:hdr>",
    ),
}


class _Entry:
    """One hand-rolled zip entry: the bytes and the lies are the test's
    to choose (declared sizes, compression method, flags, extra fields,
    header offsets), because the audit's job is exactly to not trust
    them."""

    def __init__(
        self,
        name: bytes,
        body: bytes,
        *,
        method: int = 8,
        declare: int | None = None,
        cd_extra: bytes = b"",
        header_override: int | None = None,
        flags: int = 0,
        comp_override: bytes | None = None,
    ) -> None:
        if comp_override is not None:
            comp = comp_override
        elif method == 8:
            co = zlib.compressobj(9, zlib.DEFLATED, -15)
            comp = co.compress(body) + co.flush()
        else:
            comp = body
        self.name = name
        self.comp = comp
        self.crc = zlib.crc32(body) & 0xFFFFFFFF
        self.usize = declare if declare is not None else len(body)
        self.csize = len(comp)
        self.method = method
        self.cd_extra = cd_extra
        self.header_override = header_override
        self.flags = flags


def _meta_entries() -> list[_Entry]:
    return [_Entry(n, m, method=0) for n, m in _META]


def _small_document() -> _Entry:
    prefix, suffix = _WRAPS[b"word/document.xml"]
    return _Entry(b"word/document.xml", prefix + b"hello" + suffix, method=0)


@lru_cache(maxsize=1)
def _bomb_entries() -> list[_Entry]:
    """Two 300 MiB parts declaring 1,000 uncompressed bytes each (the
    declared lie the audit's second stage exists for), built once and
    shared by every bomb pin: 600 MiB total, each part far under
    office_oxide's 512 MiB per-part cap."""
    out = []
    for name in (b"word/document.xml", b"word/header1.xml"):
        prefix, suffix = _WRAPS[name]
        payload = prefix + b"x" * (300 * 1024 * 1024 - len(prefix) - len(suffix)) + suffix
        out.append(_Entry(name, payload, declare=1000))
    return out


def _assemble(
    entries: list[_Entry],
    *,
    on_disk: int | None = None,
    total: int | None = None,
    cd_offset_override: int | None = None,
    cd_size_override: int | None = None,
    trailing_central: bool = False,
    comment: bytes = b"",
) -> bytes:
    """Locals, then the central directory, then the end record: every
    count/offset/size the audit consumes is a keyword away from a lie."""
    out = bytearray()
    offsets = []
    for e in entries:
        offsets.append(len(out))
        out += (
            b"PK\x03\x04"
            + struct.pack("<HHHHH", 20, e.flags, e.method, 0, 0)
            + struct.pack("<III", e.crc, e.csize, e.usize)
            + struct.pack("<HH", len(e.name), 0)
            + e.name
            + e.comp
        )
    central = bytearray()
    for e, off in zip(entries, offsets, strict=True):
        header = e.header_override if e.header_override is not None else off
        central += (
            b"PK\x01\x02"
            + struct.pack("<HHHHHH", 20, 20, e.flags, e.method, 0, 0)
            + struct.pack("<III", e.crc, e.csize, e.usize)
            + struct.pack("<HHHHHII", len(e.name), len(e.cd_extra), 0, 0, 0, 0, header)
            + e.name
            + e.cd_extra
        )
    n = len(entries)
    on_disk = n if on_disk is None else on_disk
    total = n if total is None else total
    cd_size = len(central) if cd_size_override is None else cd_size_override
    if trailing_central:
        # the end record FIRST, the directory after it: the engine's
        # total-is-zero early accept clamps its directory start to the
        # declared offset, so the trailing records are the ones it walks
        cd_offset = cd_offset_override if cd_offset_override is not None else len(out) + 22
        return bytes(out + (
            b"PK\x05\x06"
            + struct.pack("<HHHHIIH", 0, 0, on_disk, total, cd_size, cd_offset, 0)
        ) + central)
    cd_offset = len(out) if cd_offset_override is None else cd_offset_override
    return bytes(out + central + (
        b"PK\x05\x06"
        + struct.pack("<HHHHIIH", 0, 0, on_disk, total, cd_size, cd_offset, len(comment))
        + comment
    ))


def _zip64_wrap(entries: list[_Entry], *, lying_cd_offset: int | None = None) -> bytes:
    """A full zip64 footer over honest locals and directory: the EOCD64
    record, its locator, then an all-sentinel zip32 end record. Offsets
    are zip-relative (no prepend), the record's size field is the
    spec-honest 44, and the locator names the record's own position."""
    out = bytearray()
    offsets = []
    for e in entries:
        offsets.append(len(out))
        out += (
            b"PK\x03\x04"
            + struct.pack("<HHHHH", 20, e.flags, e.method, 0, 0)
            + struct.pack("<III", e.crc, e.csize, e.usize)
            + struct.pack("<HH", len(e.name), 0)
            + e.name
            + e.comp
        )
    central = bytearray()
    for e, off in zip(entries, offsets, strict=True):
        header = e.header_override if e.header_override is not None else off
        central += (
            b"PK\x01\x02"
            + struct.pack("<HHHHHH", 20, 20, e.flags, e.method, 0, 0)
            + struct.pack("<III", e.crc, e.csize, e.usize)
            + struct.pack("<HHHHHII", len(e.name), len(e.cd_extra), 0, 0, 0, 0, header)
            + e.name
            + e.cd_extra
        )
    n = len(entries)
    cd_offset = len(out)
    eocd64_pos = cd_offset + len(central)
    record = b"PK\x06\x06" + struct.pack(
        "<QHHIIQQQQ",
        44,
        45,
        45,
        0,
        0,
        n,
        n,
        len(central),
        lying_cd_offset if lying_cd_offset is not None else cd_offset,
    )
    locator = b"PK\x06\x07" + struct.pack("<IQI", 0, eocd64_pos, 1)
    eocd32 = b"PK\x05\x06" + struct.pack(
        "<HHHHIIH", 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0
    )
    return bytes(out + central + record + locator + eocd32)


def _two_eocd_bomb(*, locator_sig: bool = True, sentinel: bool = True) -> bytes:
    """The locator-fallback shape: an honest two-bomb archive's locals and
    directory, its honest end record, then a small directory, a locator,
    and a LAST end record the engine must hard-reject (its locator
    parses, its sentinel cd-size forces the zip64 branch, and no EOCD64
    record is findable) so it falls back to the earlier bomb record."""
    ents = _meta_entries() + _bomb_entries()
    honest = _assemble(ents)
    eocd_pos = honest.rfind(b"PK\x05\x06")
    cd_offset = struct.unpack_from("<I", honest, eocd_pos + 16)[0]
    cd_size = struct.unpack_from("<I", honest, eocd_pos + 12)[0]
    bomb_core = honest[:eocd_pos]
    small = _assemble(_meta_entries())
    seocd = small.rfind(b"PK\x05\x06")
    small_central = small[struct.unpack_from("<I", small, seocd + 16)[0] : seocd]
    eocd1 = b"PK\x05\x06" + struct.pack(
        "<HHHHIIH", 0, 0, len(ents), len(ents), cd_size, cd_offset, 0
    )
    small_cd_offset = len(bomb_core) + 22
    locator = (
        (b"PK\x06\x07" if locator_sig else b"\x00\x00\x00\x00")
        + struct.pack("<IQI", 0, 0, 1)
    )
    cd_size2 = 0xFFFFFFFF if sentinel else len(small_central)
    eocd2 = b"PK\x05\x06" + struct.pack(
        "<HHHHIIH",
        0,
        0,
        len(_META),
        len(_META),
        cd_size2,
        small_cd_offset,
        0,
    )
    return bytes(bomb_core + eocd1 + small_central + locator + eocd2)


_PROBE_CHILD = r"""
import resource, sys, time
from tors_documents import to_text

data = open(sys.argv[1], "rb").read()
t0 = time.perf_counter()
try:
    fmt, text = to_text(data=data, format="docx", backend="oxide")
    outcome = f"CONVERTED outlen={len(text)}"
except BaseException as exc:
    outcome = f"{type(exc).__name__}: {str(exc)[:400]}"
elapsed = time.perf_counter() - t0
# VmHWM, not getrusage: a forked child inherits its parent's ru_maxrss
# watermark across execve, so procfs is the only clean peak-RSS signal in
# a probe child; ru_maxrss is the fallback where procfs does not exist.
try:
    hwm = next(l for l in open("/proc/self/status") if l.startswith("VmHWM"))
    hwm_kb, source = int(hwm.split()[1]), "VmHWM"
except OSError:
    hwm_kb, source = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "ru_maxrss"
print(f"{outcome} elapsed={elapsed:.3f}s hwm={hwm_kb} kB source={source}")
"""


def _probe(path: Path, *, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _PROBE_CHILD, str(path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _assert_refused_or_bounded(done: subprocess.CompletedProcess[str], detail: str) -> None:
    """The audit's contract on a bomb shape: no signal death, and either
    a typed refusal naming the ceiling or a bounded conversion."""
    report = f"rc={done.returncode}\n{done.stdout}\n{done.stderr}"
    assert done.returncode not in (-6, -9, 134, 137), (
        f"the oxide lane died by signal on the hostile zip ({detail}):\n{report}"
    )
    if done.returncode == 0 and "CONVERTED" in done.stdout:
        hwm_line = next(line for line in done.stdout.splitlines() if "hwm=" in line)
        peak = int(hwm_line.split("hwm=")[1].split()[0])
        assert peak < _BOMB_RSS_CAP_KB, (
            f"the hostile zip CONVERTED ({detail}) at {peak} kB peak RSS (pre-fix red: "
            f"~2.13e6 kB); the audit must refuse it or bound the conversion:\n{report}"
        )
    else:
        assert "ValueError" in done.stdout, (
            f"the hostile zip neither converted bounded nor refused with the documented "
            f"ValueError ({detail}):\n{report}"
        )


class TestTheCountFields:
    """The engine counts directory records from the EOCD's on-this-disk
    field; the audit once counted from the total. Both directions of the
    divergence were measured pre-fix as full bypasses (CONVERTED,
    ~2.13 GiB peak RSS, ~1.3 s, from a ~600 KiB input); post-fix both are
    refused at the ~18 MiB child baseline in hundredths of a second, the
    refusal naming the 536,870,912-byte ceiling across all parts."""

    def test_records_beyond_the_total_are_audited(self, tmp_path: Path) -> None:
        """On-this-disk 5, total 1: the engine walked five records (two of
        them 300 MiB bombs) while the audit walked one small one."""
        path = tmp_path / "counts_disagree.docx"
        path.write_bytes(
            _assemble(_meta_entries() + _bomb_entries(), on_disk=5, total=1)
        )
        _assert_refused_or_bounded(
            _probe(path), "on-this-disk 5 against a total of 1, two 300 MiB parts"
        )

    def test_a_zero_total_is_not_an_empty_archive(self, tmp_path: Path) -> None:
        """On-this-disk 5, total 0, the directory planted after the end
        record: the engine's early accept walked the trailing five (the
        audit returned an empty pass at the sight of the zero)."""
        path = tmp_path / "total_zero_trailing.docx"
        path.write_bytes(
            _assemble(
                _meta_entries() + _bomb_entries(),
                on_disk=5,
                total=0,
                trailing_central=True,
            )
        )
        _assert_refused_or_bounded(
            _probe(path), "total 0 against on-this-disk 5, the directory after the end record"
        )


class TestTheLocatorFallback:
    """A parseable EOCD64 locator is a hard branch for the engine: no
    EOCD64 record findable means the whole end record is rejected and an
    EARLIER one wins. The audit once fell back to the zip32 values
    instead, so the last record's small directory passed it while the
    engine fell back to the earlier two-bomb record: measured pre-fix as
    CONVERTED at ~2.13 GiB peak RSS. The union closes the whole family:
    every candidate that resolves is audited, whichever one the reader
    picks."""

    @pytest.mark.parametrize(
        ("locator_sig", "sentinel"),
        [(True, True), (False, True), (True, False)],
        ids=["locator_and_sentinel", "sentinel_only", "locator_only"],
    )
    def test_the_earlier_end_record_is_audited_whichever_wins(
        self, tmp_path: Path, locator_sig: bool, sentinel: bool
    ) -> None:
        """The last end record is engine-rejected in the first cell (the
        locator parses, the sentinel forces zip64, no record is
        findable) and engine-ACCEPTED in the other two (its zip32 values
        walk the small directory): in every cell the audit refuses,
        because the earlier two-bomb record resolves too and its parts
        are in the union."""
        path = tmp_path / "two_end_records.docx"
        path.write_bytes(_two_eocd_bomb(locator_sig=locator_sig, sentinel=sentinel))
        _assert_refused_or_bounded(
            _probe(path), "two end records, the earlier one a two-bomb archive"
        )

    def test_a_zip64_extra_field_size_lie_is_not_trusted(self, tmp_path: Path) -> None:
        """Two 300 MiB parts whose central records carry a full (24-byte)
        zip64 extended-information field declaring 1,000 uncompressed
        bytes, compressed size and header offset honest: the declared lie
        must not bound what a part inflates to. Already refused before
        the rewrite (the extra field's sizes fed the declared pass and
        the re-inflate caught the lie, 2026-09-16, measured ~18.5 MiB):
        pinned here so the zip64 path cannot regress into trusting it."""
        ents = _meta_entries()
        first = _assemble(_meta_entries() + _bomb_entries())
        offsets: dict[bytes, int] = {}
        at = 0
        while True:
            idx = first.find(b"PK\x03\x04", at)
            if idx < 0:
                break
            nlen = struct.unpack_from("<H", first, idx + 26)[0]
            name = bytes(first[idx + 30 : idx + 30 + nlen])
            offsets[name] = idx
            at = idx + 4
        for name in (b"word/document.xml", b"word/header1.xml"):
            prefix, suffix = _WRAPS[name]
            payload = prefix + b"x" * (300 * 1024 * 1024 - len(prefix) - len(suffix)) + suffix
            co = zlib.compressobj(9, zlib.DEFLATED, -15)
            comp = co.compress(payload) + co.flush()
            extra = struct.pack("<HH", 0x0001, 24) + struct.pack(
                "<QQQ", 1000, len(comp), offsets[name]
            )
            ents.append(
                _Entry(
                    name,
                    payload,
                    declare=0xFFFFFFFF,
                    cd_extra=extra,
                    comp_override=comp,
                )
            )
        path = tmp_path / "zip64_extra_lies.docx"
        path.write_bytes(_assemble(ents))
        _assert_refused_or_bounded(_probe(path), "zip64 extra fields declaring 1,000 bytes")


class TestTheStandingRefusals:
    """The shapes the audit must keep refusing, each in-process (they are
    refusals, not bombs: the audit refuses before the engine runs)."""

    @pytest.mark.parametrize("method", [9, 12, 93], ids=["deflate64", "bzip2", "zstd"])
    def test_unsupported_compression_methods_are_refused_naming_the_method(
        self, method: int
    ) -> None:
        """Deflate64, bzip2 and zstd parts: the audit cannot re-inflate
        them, so the container is refused before the engine can."""
        blob = _assemble(
            _meta_entries() + [_Entry(b"word/document.xml", b"junk", method=method)]
        )
        with pytest.raises(ValueError, match=f"compression method {method}"):
            to_text(data=blob, format="docx", backend="oxide")

    def test_an_encrypted_part_is_refused(self) -> None:
        """The encryption flag: the audit audits the bytes it inflates
        and cannot bound encrypted ones."""
        blob = _assemble(
            _meta_entries() + [_Entry(b"word/document.xml", b"junk", method=0, flags=1)]
        )
        with pytest.raises(ValueError, match="is encrypted"):
            to_text(data=blob, format="docx", backend="oxide")

    def test_a_local_header_inside_another_payload_is_refused(self) -> None:
        """A header offset pointing into another entry's data: the audit
        cannot locate the bytes the reader would inflate there."""
        blob = _assemble(
            _meta_entries()
            + [_Entry(b"word/document.xml", b"junk", method=0, header_override=700)]
        )
        with pytest.raises(ValueError, match="unreadable local header"):
            to_text(data=blob, format="docx", backend="oxide")

    def test_a_directory_offset_past_the_end_of_file_refuses(self) -> None:
        """A declared directory start beyond the file: unauditable, a
        catchable refusal, never a panic on the offset arithmetic."""
        blob = _assemble(
            _meta_entries() + [_small_document()], cd_offset_override=0xFFFFFFF0
        )
        with pytest.raises(ValueError, match="did not parse under this lane"):
            to_text(data=blob, format="docx", backend="oxide")

    def test_a_count_the_directory_cannot_hold_refuses(self) -> None:
        """Both count fields claiming 65,535 records over a three-record
        directory: forged, refused typed."""
        blob = _assemble(_meta_entries() + [_small_document()], on_disk=65535, total=65535)
        with pytest.raises(ValueError):
            to_text(data=blob, format="docx", backend="oxide")

    def test_an_empty_directory_is_the_engine_missing_part_error(self) -> None:
        """A zero-entry archive audits to nothing and converts nowhere:
        the engine's own missing-part refusal is the answer."""
        blob = b"PK\x05\x06" + struct.pack("<HHHHIIH", 0, 0, 0, 0, 0, 0, 0)
        with pytest.raises(ValueError, match=r"\[Content_Types\]\.xml"):
            to_text(data=blob, format="docx", backend="oxide")

    def test_a_lying_zip64_directory_offset_refuses(self) -> None:
        """An honest zip64 footer whose EOCD64 record declares a
        directory offset of 5: the rebased start lands inside the first
        local header, and the refusal is catchable on either side of the
        seam (the audit's unauditable voice or the engine's own)."""
        blob = _zip64_wrap(_meta_entries() + [_small_document()], lying_cd_offset=5)
        with pytest.raises(ValueError):
            to_text(data=blob, format="docx", backend="oxide")

    def test_method_eight_claimed_over_stored_bytes_fails_typed(self) -> None:
        """A part claiming deflate over raw stored bytes: the audit's
        re-inflate dies on the first block and the engine's own inflate
        dies the same way; either way a typed refusal, never a hang."""
        prefix, suffix = _WRAPS[b"word/document.xml"]
        body = prefix + b"hello" + suffix
        blob = _assemble(
            _meta_entries()
            + [_Entry(b"word/document.xml", body, method=8, comp_override=body)]
        )
        with pytest.raises(ValueError):
            to_text(data=blob, format="docx", backend="oxide")

    def test_data_descriptor_zeroed_sizes_read_nothing(self) -> None:
        """The descriptor flag with zeroed central sizes: the engine takes
        a zero-length compressed stream, so nothing inflates and the
        conversion fails typed on the empty part (or converts that
        emptiness), never on unbounded bytes."""
        ents = _meta_entries() + [_Entry(b"word/document.xml", b"junk", method=0, flags=8)]
        for e in ents:
            e.csize = 0
            e.usize = 0
        blob = _assemble(ents)
        try:
            fmt, text = to_text(data=blob, format="docx", backend="oxide")
            assert len(text) <= len(blob)
        except ValueError:
            pass

    def test_an_eocd_signature_inside_the_comment_is_refused_or_skipped(
        self, tmp_path: Path
    ) -> None:
        """A decoy end-record signature planted in the trailing comment of
        a real zip: its garbage fields must not win the backward scan (the
        engine's own duplicate-count or directory checks refuse it), and
        the real archive must never inflate because of it."""
        real = _assemble(_meta_entries() + [_small_document()])
        blob = real[:-2] + struct.pack("<H", 44) + (
            b"junkjunk" + b"PK\x05\x06" + b"\x11\x22\x33\x44" + b"z" * 34
        )
        path = tmp_path / "decoy_in_comment.docx"
        path.write_bytes(blob)
        done = _probe(path, timeout=30.0)
        _assert_refused_or_bounded(done, "a decoy end-record signature inside the comment")


class TestTheStandingConversions:
    """The shapes a real writer can produce: each must keep CONVERTING
    (the audit's mirror must never be stricter than the reader on a
    legitimate container)."""

    def test_a_legitimate_zip64_archive_converts(self) -> None:
        """The full zip64 footer, honest everywhere: sentinel zip32 end
        record, EOCD64 record of size 44, locator naming the record."""
        blob = _zip64_wrap(_meta_entries() + [_small_document()])
        fmt, text = to_text(data=blob, format="docx", backend="oxide")
        assert "hello" in text

    def test_a_prepended_container_converts(self) -> None:
        """Junk before the zip (the self-extracting shape): the directory
        is found by the first-signature search and every local offset is
        rebased by the same prepend delta. The search once anchored on
        the signature's last byte and matched AT it, a shape no real
        signature satisfies, so every prepended container died
        unauditable (fail-closed, but a conversion the engine could make
        and the audit's own docstring claimed); pinned as the conversion
        it must be."""
        blob = b"#!/bin/sh\nexec java -jar sfx.jar\n" + _assemble(
            _meta_entries() + [_small_document()]
        )
        fmt, text = to_text(data=blob, format="docx", backend="oxide")
        assert "hello" in text

    def test_an_embedded_zip_stored_as_a_part_converts(self) -> None:
        """A whole inner zip stored as one outer part: the inner's
        end-record signature is an earlier candidate the union audits
        too (its records are real, its parts small), and the outer
        archive converts on the candidate the reader picks."""
        inner = _assemble(_meta_entries() + [_small_document()])
        blob = _assemble(
            _meta_entries()
            + [_small_document()]
            + [_Entry(b"word/embedded.docx", inner, method=0)]
        )
        fmt, text = to_text(data=blob, format="docx", backend="oxide")
        assert "hello" in text

    def test_two_entries_sharing_one_local_header_stay_bounded(self) -> None:
        """Two directory entries pointing at the same local header (the
        overlapping-span shape): both are audited, the shared span is
        counted per entry, and the conversion stays bounded."""
        ents = _meta_entries() + [_small_document()]
        first = _assemble(ents)
        doc_off = None
        at = 0
        while True:
            idx = first.find(b"PK\x03\x04", at)
            if idx < 0:
                break
            nlen = struct.unpack_from("<H", first, idx + 26)[0]
            name = bytes(first[idx + 30 : idx + 30 + nlen])
            if name == b"word/document.xml":
                doc_off = idx
                break
            at = idx + 4
        assert doc_off is not None
        blob = _assemble(
            ents + [_Entry(b"word/header1.xml", b"junk", method=0, header_override=doc_off)]
        )
        fmt, text = to_text(data=blob, format="docx", backend="oxide")
        assert len(text) < len(blob) * 4
