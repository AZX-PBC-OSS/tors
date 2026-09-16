"""The PDF pre-scan's hostility pins: the seam that guards the pdf_oxide
lane runs a nesting walk plus a compressed-object scan before the engine
opens the document, and the engine behind it resolves filter names
through a spec-abbreviation and case-insensitive table and accepts
spec-violating stream keyword spellings. Four engine behaviors the scan
did not mirror were each measured as an uncatchable SIGSEGV (rc=-11,
process death, the crash class the crash file's taxonomy calls the worst
shape) from ~1.1 KB documents on this tree, 2026-09-16:

- the Table 6 abbreviation: pdf_oxide resolves ``/Fl`` to FlateDecode
  (its decoders/mod.rs), the scan's trigger was a literal ``FlateDecode``
  byte containment, so a ``/Type /ObjStm /Filter /Fl`` bomb sailed past
  the scan's inflate into the engine's uncapped recursive object parser;
- the case-insensitive fallback: ``/flatedecode`` (any casing) resolves
  the same way, with the same miss;
- a filter the scan has no decoder for: an ``/ASCIIHexDecode`` object
  stream's payload is hex text to a raw scan (no nesting, nothing to
  see), and the engine decodes it into the parser;
- the stream keyword's end-of-line: pdf_oxide accepts a CR-only or
  MISSING end-of-line after ``stream`` ("SPEC VIOLATION ... Accepting in
  lenient mode", its parser.rs) and starts the payload right after the
  keyword; the scan demanded a spec EOL and never collected those
  streams at all.

The fix mirrors the engine: the Flate vocabulary (any-cased full name,
the ``/Fl`` abbreviation as a whole name), the four end-of-line shapes
(CRLF, LF, CR, none), and a fail-closed refusal for ObjStm/XRef streams
declaring a filter pdf_oxide can decode but the scan cannot (LZW,
RunLength, ASCIIHex/85, the image filters, Brotli): a payload the scan
cannot inflate is a payload whose nesting it cannot bound, and the
engine's uncapped ``parse_object`` behind it is exactly what the layer
exists to guard. The benign shapes (a legal ObjStm spelled ``/Fl``, the
array form, odd ``/Type`` whitespace, ``endstream`` with trailing junk,
an indirect filter reference) must keep converting or keep failing
catchably where they already did.
"""

from __future__ import annotations

import subprocess
import sys
import zlib

import pytest

from tors.documents import pdf_classify, pdf_extract, pdf_link_uris, pdf_page_count


def _row(type_: int, a: int, b: int) -> bytes:
    return bytes([type_]) + a.to_bytes(4, "big") + b.to_bytes(2, "big")


def _objstm_document(
    filter_bytes: bytes | None,
    *,
    n: int,
    payload: bytes | None = None,
    extra_objects: bytes = b"",
    stream_eol: bytes = b"\n",
    endstream_junk: bytes = b"",
    dict_type: bytes = b"/Type /ObjStm",
) -> bytes:
    """The pinned ObjStm-bomb document shape (the crash file's
    construction) with every dial the hostility cells need: the filter
    spelling, the nesting depth, the payload, the keyword's end-of-line,
    and the endstream's spelling."""
    bomb = b"[" * n + b"]" * n
    out = bytearray(b"%PDF-1.7\n")
    pairs = b"1 0\n"
    if payload is None:
        payload = zlib.compress(pairs + b"<< /Type /Catalog /X " + bomb + b" >>")
    offsets = {}
    offsets[2] = len(out)
    out += b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
    offsets[3] = len(out)
    out += extra_objects
    filt = b"" if filter_bytes is None else b" /Filter " + filter_bytes
    out += (
        b"3 0 obj\n<< "
        + dict_type
        + b" /N 1 /First "
        + str(len(pairs)).encode()
        + filt
        + b" /Length "
        + str(len(payload)).encode()
        + b" >>\nstream"
        + stream_eol
        + payload
        + b"\nendstream"
        + endstream_junk
        + b"\nendobj\n"
    )
    rows = (
        _row(0, 0, 65535)
        + _row(2, 3, 0)
        + _row(1, offsets[2], 0)
        + _row(1, offsets[3], 0)
    )
    xref_at = len(out)
    rows += _row(1, xref_at, 0)
    data = zlib.compress(rows)
    out += (
        b"4 0 obj\n<< /Type /XRef /Size 5 /W [1 4 2] /Root 1 0 R"
        b" /Filter /FlateDecode /Length "
        + str(len(data)).encode()
        + b" >>\nstream\n"
        + data
        + b"\nendstream\nendobj\nstartxref\n"
        + str(xref_at).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


_ENTRY_POINTS = [pdf_page_count, pdf_extract, pdf_classify, pdf_link_uris]


def _probe(blob: bytes) -> subprocess.CompletedProcess[str]:
    """Every entry point opens the same document through the same
    pre-scan; one child per cell, the crash file's discipline: a
    regression dies as a child's signal, never in the runner."""
    path = "/tmp/opencode/prescan_cell.pdf"
    with open(path, "wb") as fh:
        fh.write(blob)
    child = r"""
import sys
import tors.documents as d

data = open(sys.argv[1], "rb").read()
try:
    result = d.pdf_page_count(data=data)
    print(f"CONVERTED {result!r}"[:120])
except BaseException as exc:
    print(f"{type(exc).__name__}: {str(exc)[:200]}")
"""
    return subprocess.run(
        [sys.executable, "-c", child, path],
        capture_output=True,
        text=True,
        timeout=60.0,
    )


def _assert_catchable(done: subprocess.CompletedProcess[str], detail: str) -> None:
    report = f"rc={done.returncode}\n{done.stdout}\n{done.stderr}"
    assert done.returncode not in (-6, -9, -11, 134, 137), (
        f"the PDF pre-scan shape died by signal ({detail}, measured pre-fix: "
        f"SIGSEGV on exactly these cells):\n{report}"
    )
    assert done.returncode == 0, f"the probe child itself failed ({detail}):\n{report}"
    assert "CONVERTED" in done.stdout or "ValueError" in done.stdout, (
        f"the shape neither converted nor refused with the documented ValueError "
        f"({detail}):\n{report}"
    )


class TestTheFilterVocabulary:
    """The Flate spellings pdf_oxide resolves: the exact name, the Table 6
    abbreviation ``/Fl``, and every casing of the full name. Each was a
    measured SIGSEGV bypass before the scan mirrored the vocabulary."""

    @pytest.mark.parametrize(
        "spelling",
        [b"/FlateDecode", b"/Fl", b"/flatedecode", b"/FLATEDECODE", b"[/FlateDecode]", b"[/Fl]"],
        ids=["exact", "abbreviation", "lowercase", "uppercase", "array-exact", "array-abbrev"],
    )
    def test_every_flate_spelling_is_scanned(self, spelling: bytes) -> None:
        """A nesting bomb behind any Flate spelling: refused catchably
        (the scan inflates under its ceiling and the depth scan fires),
        never signal death."""
        blob = _objstm_document(spelling, n=400_000)
        _assert_catchable(_probe(blob), f"the {spelling.decode()} Flate spelling")

    @pytest.mark.parametrize(
        "spelling",
        [b"/ASCIIHexDecode", b"/AHx", b"/LZWDecode", b"/LZW", b"/RunLengthDecode", b"/RL"],
        ids=["hex-full", "hex-abbrev", "lzw-full", "lzw-abbrev", "rl-full", "rl-abbrev"],
    )
    def test_a_filter_the_scan_cannot_decode_refuses(self, spelling: bytes) -> None:
        """An ObjStm declaring a filter pdf_oxide decodes but this scan
        cannot: the fail-closed refusal naming the filter, never the
        engine's uncapped parser behind it. The ASCIIHex cell was a
        measured SIGSEGV pre-fix (the payload is hex text to a raw scan)."""
        hex_payload = (b"1 0\n" + b"<< /Type /Catalog /X " + b"[" * 400_000 + b"]" * 400_000
                       + b" >>").hex().encode() + b">"
        payload = hex_payload if spelling in (b"/ASCIIHexDecode", b"/AHx") else None
        blob = _objstm_document(spelling, n=400_000, payload=payload)
        done = _probe(blob)
        _assert_catchable(done, f"the {spelling.decode()} filter")
        if spelling in (b"/ASCIIHexDecode", b"/AHx"):
            assert "ASCIIHexDecode" in done.stdout, (
                f"the refusal must name the filter it cannot decode:\n{done.stdout}"
            )

    def test_a_benign_objstm_spelled_with_the_abbreviation_converts(self) -> None:
        """The mirror must not be stricter than the engine on a legal
        document: a real object stream spelling its filter ``/Fl`` still
        converts."""
        blob = _objstm_document(
            b"/Fl", n=0, payload=zlib.compress(b"1 0\n<< /Type /Catalog /Pages 2 0 R >>")
        )
        pages, text = pdf_extract(data=blob, password=None)
        assert pages == []
        assert text == ""

    def test_an_indirect_filter_reference_stays_the_engine_own_error(self) -> None:
        """``/Filter 5 0 R``: pdf_oxide does not resolve the reference
        (measured: it refuses the object stream as not-a-Stream), so the
        shape stays its catchable error, whichever side answers first."""
        extra = b"5 0 obj\n<< /FlateDecode >>\nendobj\n"
        blob = _objstm_document(b"5 0 R", n=20_000, extra_objects=extra)
        _assert_catchable(_probe(blob), "an indirect /Filter reference")


class TestTheStreamKeyword:
    """The end-of-line shapes pdf_oxide accepts after ``stream``: CRLF
    and LF per the spec, and (its own leniency, "SPEC VIOLATION ...
    Accepting in lenient mode") a lone CR and NO end-of-line at all. The
    last two were measured SIGSEGV bypasses before the mirror."""

    @pytest.mark.parametrize(
        "eol",
        [b"\r\n", b"\n", b"\r", b""],
        ids=["crlf", "lf", "cr-only", "missing"],
    )
    def test_every_eol_shape_is_collected(self, eol: bytes) -> None:
        """The bomb behind each end-of-line spelling: collected, inflated,
        refused catchably."""
        blob = _objstm_document(b"/FlateDecode", n=400_000, stream_eol=eol)
        _assert_catchable(_probe(blob), f"the {eol!r} end-of-line after stream")

    def test_whitespace_before_the_eol_stays_the_engine_own_error(self) -> None:
        """``stream`` + whitespace + EOL: pdf_oxide's own decompressor
        chokes on the space-led payload (measured, catchably), so the
        shape needs no scan collection to stay safe."""
        blob = _objstm_document(b"/FlateDecode", n=400_000, stream_eol=b" \n")
        _assert_catchable(_probe(blob), "whitespace between stream and its EOL")

    def test_endstream_with_trailing_junk_is_found(self) -> None:
        """``endstreamjunk``: the payload ends at the embedded keyword
        either way, catchably."""
        blob = _objstm_document(b"/FlateDecode", n=20_000, endstream_junk=b"junk")
        _assert_catchable(_probe(blob), "endstream with trailing junk glued on")


class TestTheDictShapes:
    """The dictionary-side shapes: the ObjStm marker is matched as bytes
    anywhere in the dict (so no-space and nested-dict spellings cannot
    hide it), and a stream whose dict does not carry the marker at all is
    never layer-2's business."""

    @pytest.mark.parametrize(
        "dict_type",
        [b"/Type /ObjStm", b"/Type/ObjStm", b"/Type\t/ObjStm", b"<< /A << /Type /ObjStm >> >>"],
        ids=["spaced", "glued", "tabbed", "nested-dict"],
    )
    def test_every_objstm_marker_spelling_is_scanned(self, dict_type: bytes) -> None:
        """The marker as bytes: whitespace and nesting cannot move it out
        of the dict the stream's own close delimited."""
        blob = _objstm_document(
            None, n=20_000, dict_type=dict_type, payload=b"1 0\n" + b"[" * 20_000 + b"]" * 20_000
        )
        _assert_catchable(_probe(blob), f"the {dict_type!r} marker spelling")

    def test_an_objstm_inside_an_objstm_is_catchable(self) -> None:
        """A whole inner ObjStm stream object riding in the outer stream's
        inflated text: the outer's scan covers the inner dict's nesting,
        the spec forbids the inner stream (pdf_oxide refuses it
        catchably, measured), and the shape never reaches the uncapped
        parser unguarded."""
        inner_payload = zlib.compress(
            b"1 0\n" + b"<< /Type /Catalog /X " + b"[" * 20_000 + b"]" * 20_000 + b" >>"
        )
        inner = (
            b"<< /Type /ObjStm /N 1 /First 4 /Filter /FlateDecode"
            + b" /Length "
            + str(len(inner_payload)).encode()
            + b" >>\nstream\n"
            + inner_payload
            + b"\nendstream"
        )
        blob = _objstm_document(b"/Fl", n=0, payload=zlib.compress(b"2 0\n" + inner))
        _assert_catchable(_probe(blob), "an ObjStm nested inside an ObjStm payload")

    def test_every_entry_point_agrees_on_the_abbreviation_bomb(self) -> None:
        """The four public entry points all open through the same
        pre-scan: the /Fl bomb is a catchable refusal on each, in-process
        (the scan refuses before the engine, so no child is needed)."""
        blob = _objstm_document(b"/Fl", n=400_000)
        for fn in _ENTRY_POINTS:
            with pytest.raises(ValueError, match="nests arrays/dictionaries"):
                fn(data=blob, password=None)
