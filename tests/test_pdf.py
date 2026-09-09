"""The PDF surface (`tors.documents.pdf_extract`/`tors.documents.pdf_page_count`): correctness over
real PDF bytes, the structure-preservation properties that motivated the
surface (headings, links, column separation), error mapping, concurrent
correctness under REAL parallelism (the GIL is released, so N threads run the
native core simultaneously — the property pdfium does not have), and the
GIL-release band, pinned with the suite's shared heartbeat methodology.

Why this surface exists (the caller's history, 2026-09): the knowledge-indexing
caller routed born-digital PDFs through pypdf (pure Python, GIL-held at
bytecode granularity, no structure output, merges visual lines across column
gaps) and pdfium entered its tree only as a transitive of the heavy OCR
fallback, where it is lock-guarded inside that library and isolated in a child
process — but pdfium itself is not thread-safe and its bindings hold the GIL
per call, so it could never be the fast path. pdf_oxide measured better on
every axis that matters here (structure, links, columns, speed, 100% pass rate
on the veraPDF/pdf.js/SafeDocs corpora) — and called directly from Rust under
``py.detach``, its whole open+extract+convert pass is GIL-free.

Fixtures are hand-built PDF bytes (the same byte-deterministic object-graph
pattern as ``documents.generate_pdf``), so every property under test — a
/Link annotation, a two-column content stream, a 24pt heading line, a
contentless page — is visible and greppable in this file rather than hidden
inside a binary blob. The corpus document (``documents.generate_pdf()``) adds
a FlateDecode + WinAnsi non-ASCII case the hand-built fixtures don't cover.

GIL band (measured on the dev box, 2026-09-08, ambient load noted per cell,
the corrected ``_gap_and_wall_during`` harness, 5 samples):

- ``pdf_extract`` over a ~470KB single-page document (6,000 lines): walls
  130-160ms, worst gaps 10.7-13.9ms (ratio 0.07-0.10) — the 10ms ping floor
  plus the O(output) return marshalling (one str per page, here one, plus the
  markdown string), the same marshalling class as ``finalize``'s two output
  strings. A detach regression holds the whole ~140ms wall (ratio ~1.0) and
  fails both budgets by an order of magnitude; the red side was also measured
  directly against the official pdf_oxide pyo3 wheel (GIL-held per call:
  worst gap 23.6ms on a 9-page/33KB document under a 10ms ping, 2026-09), the
  hazard this wrapper exists to remove.
- Budgets: the suite's shared 0.30 ratio and 100ms ceiling hold ~3x margin on
  the worst measured ratio and ~7x on the worst measured gap.

Concurrency: 8 threads, one ``pdf_extract`` each over the same file — with the
GIL released these are 8 simultaneously-running native extractions (pdfium
cannot do this without an external global lock; its own bindings document the
hazard). The assertion is not just "no crash": every thread's result must be
byte-identical to the single-threaded answer.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from test_gil_release import _assert_loop_stays_responsive

from documents import PDF_TEXT, generate_pdf
from tors.documents import pdf_extract, pdf_page_count


def _objects_pdf(objects: list[bytes]) -> bytes:
    """The shared byte-deterministic writer: correct offsets, xref, trailer."""
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


def _page_objects(content: bytes, *, annots: bytes | None = None) -> list[bytes]:
    """Catalog/pages/page/font/contents — one page, arbitrary content stream."""
    annot_entry = b" /Annots [6 0 R]" if annots is not None else b""
    return [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >>"
        b" /MediaBox [0 0 612 792] /Contents 5 0 R" + annot_entry + b" >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + content + b"\nendstream",
    ]


def _write(tmp_path: Path, name: str, blob: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(blob)
    return str(path)


def _digital_pdf(text: str, *, font_size: int = 12) -> bytes:
    content = f"BT /F1 {font_size} Tf 50 700 Td ({text}) Tj ET".encode("ascii")
    return _objects_pdf(_page_objects(content))


def _blank_pdf() -> bytes:
    return _objects_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
        ]
    )


def _two_page_pdf(first: str, second: str) -> bytes:
    """Two pages, one line each — pins page order and the per-page list shape."""
    c1 = f"BT /F1 12 Tf 50 700 Td ({first}) Tj ET".encode("ascii")
    c2 = f"BT /F1 12 Tf 50 700 Td ({second}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >>"
        b" /MediaBox [0 0 612 792] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(c1)} >>\nstream\n".encode("ascii") + c1 + b"\nendstream",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >>"
        b" /MediaBox [0 0 612 792] /Contents 7 0 R >>",
        f"<< /Length {len(c2)} >>\nstream\n".encode("ascii") + c2 + b"\nendstream",
    ]
    return _objects_pdf(objects)


def _two_column_pdf() -> bytes:
    """A left column (x=50) and a right column (x=350), three lines each — the
    layout where plain-text extraction interleaves columns into single visual
    lines and the markdown reading-order pass must not."""
    content = (
        b"BT /F1 12 Tf 50 700 Td (LEFT-A first line) Tj 0 -14 Td (LEFT-A second) Tj"
        b" 0 -14 Td (LEFT-A third) Tj ET\n"
        b"BT /F1 12 Tf 350 700 Td (RIGHT-B first line) Tj 0 -14 Td (RIGHT-B second)"
        b" Tj 0 -14 Td (RIGHT-B third) Tj ET"
    )
    return _objects_pdf(_page_objects(content))


def _link_pdf() -> bytes:
    """A page whose only text sits under a /Link annotation with a /URI action —
    the fixture that pins [text](uri) in the markdown output."""
    content = b"BT /F1 12 Tf 72 700 Td (Visit the handbook) Tj ET"
    annot = (
        b"<< /Type /Annot /Subtype /Link /Rect [72 695 200 715]"
        b" /A << /Type /Action /S /URI /URI (https://handbook.example.com/guide) >>"
        b" /Border [0 0 1] >>"
    )
    return _objects_pdf(_page_objects(content, annots=annot) + [annot])


def _heading_pdf() -> bytes:
    """A 24pt line over 12pt body — heading detection keys off font size."""
    content = (
        b"BT /F1 24 Tf 50 700 Td (Quarterly Report) Tj ET\n"
        b"BT /F1 12 Tf 50 660 Td (Revenue grew twelve percent.) Tj ET"
    )
    return _objects_pdf(_page_objects(content))


def _big_pdf(lines: int) -> bytes:
    """One page, `lines` text-showing operators — the input-scaling cell for the
    GIL band (a ~80-byte sentence per line, so 6,000 lines is ~470KB of text)."""
    sentence = "The transformer maintenance schedule covers 138kV oil-filled units."
    moves = ["BT", "/F1 12 Tf 50 750 Td"]
    for _ in range(lines):
        moves.append(f"({sentence}) Tj")
        moves.append("0 -12 Td")
    moves.append("ET")
    content = "\n".join(moves).encode("ascii")
    return _objects_pdf(_page_objects(content))


class TestPdfExtract:
    def test_born_digital_page_text_and_markdown(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "digital.pdf",
            _digital_pdf("This PDF has a genuine text layer with plenty of characters."),
        )
        pages, markdown = pdf_extract(path)
        assert pages == ["This PDF has a genuine text layer with plenty of characters."]
        assert "genuine text layer" in markdown

    def test_corpus_document_with_flate_and_winansi(self, tmp_path: Path) -> None:
        """The corpus fixture: FlateDecode content stream, WinAnsi font, non-ASCII
        (e-acute) text — the encoded-stream path the bare fixtures don't cover."""
        path = _write(tmp_path, "corpus.pdf", generate_pdf())
        pages, markdown = pdf_extract(path)
        assert len(pages) == 1
        for line in PDF_TEXT.splitlines():
            assert line in pages[0]
            assert line in markdown

    def test_two_pages_come_back_in_page_order(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "two.pdf", _two_page_pdf("first page line", "second page line"))
        pages, markdown = pdf_extract(path)
        assert pages == ["first page line", "second page line"]
        assert "first page line" in markdown
        assert "second page line" in markdown

    def test_contentless_page_is_empty_not_an_error(self, tmp_path: Path) -> None:
        """The scanned-PDF routing case: an image-only page has no text layer, which
        must read as empty output (the CALLER routes to OCR), never as a raise."""
        path = _write(tmp_path, "blank.pdf", _blank_pdf())
        pages, markdown = pdf_extract(path)
        assert pages == [""]
        assert markdown == ""

    def test_unparseable_bytes_raise_value_error(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "garbage.pdf", b"not a pdf at all")
        with pytest.raises(ValueError):
            pdf_extract(path)

    def test_missing_file_raises_oserror(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            pdf_extract(str(tmp_path / "nope.pdf"))


class TestStructurePreservation:
    def test_markdown_detects_headings(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "heading.pdf", _heading_pdf())
        _, markdown = pdf_extract(path)
        assert "# Quarterly Report" in markdown
        assert "Revenue grew twelve percent." in markdown

    def test_markdown_renders_link_annotations(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "link.pdf", _link_pdf())
        pages, markdown = pdf_extract(path)
        assert "[Visit the handbook](https://handbook.example.com/guide)" in markdown
        # Links are annotation-derived: the per-page PLAIN text carries the
        # label only, never the URI.
        assert "handbook.example.com" not in pages[0]
        assert "Visit the handbook" in pages[0]

    def test_markdown_keeps_columns_as_separate_blocks(self, tmp_path: Path) -> None:
        """The measured failure mode this pins: whole-document PLAIN text joins a
        left and right column into one visual line ("LEFT-A first line RIGHT-B
        first line"); the markdown reading-order pass must keep them apart."""
        path = _write(tmp_path, "twocol.pdf", _two_column_pdf())
        _, markdown = pdf_extract(path)
        left_block = markdown.index("LEFT-A first line")
        right_block = markdown.index("RIGHT-B first line")
        assert left_block < right_block
        for line in markdown.splitlines():
            assert not ("LEFT-A" in line and "RIGHT-B" in line), line


class TestPdfPageCount:
    def test_counts_pages(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "two.pdf", _two_page_pdf("a", "b"))
        assert pdf_page_count(path) == 2

    def test_unparseable_bytes_raise_value_error(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "garbage.pdf", b"not a pdf at all")
        with pytest.raises(ValueError):
            pdf_page_count(path)

    def test_missing_file_raises_oserror(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            pdf_page_count(str(tmp_path / "nope.pdf"))


class TestConcurrency:
    def test_parallel_extracts_are_byte_identical(self, tmp_path: Path) -> None:
        """8 threads, one pdf_extract each, GIL released for every native pass —
        simultaneously-running extractions, not GIL-serialized ones (the pdfium
        hazard this surface exists to be safe for). Every thread's answer must
        equal the single-threaded answer exactly."""
        path = _write(
            tmp_path,
            "twocol.pdf",
            _two_column_pdf(),
        )
        expected = pdf_extract(path)
        results: list[tuple[list[str], str]] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            try:
                barrier.wait()
                results.append(pdf_extract(path))
            except BaseException as exc:  # noqa: BLE001 -- recorded, asserted below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        assert len(results) == 8
        assert all(result == expected for result in results)


@pytest.mark.timing
def test_pdf_extract_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    tmp_path: Path,
) -> None:
    """The GIL-release claim as a test, the suite's shared methodology: the whole
    open+extract+convert pass runs under py.detach, so a caller that off-loads
    to a thread (tors.documents.aio.pdf_extract does exactly this) keeps a loop that ticks
    every ~10ms, where a GIL-held binding (the official pdf_oxide wheel,
    measured 2026-09: worst gap ~= the whole call) blocks it for the full wall.
    The ~470KB single-page document keeps the wall an order of magnitude over
    the 10ms ping floor so the ratio resolves (the module docstring's guidance).
    A TIMING-lane cell: the load-sensitive band measurement CI's matrix legs
    deselect (`-m "not timing and not sweep"`), one 3.12 leg running it — the
    marker-split contract every lane cell carries.
    """
    import asyncio

    path = _write(tmp_path, "big.pdf", _big_pdf(6000))

    def op() -> None:
        pdf_extract(path)

    asyncio.run(
        _assert_loop_stays_responsive(lambda: asyncio.to_thread(op)),
    )


def test_aio_spellings_exist_and_await() -> None:
    """The curated large-input subset must include the PDF pair: input cost
    scales with the file, the exact class the aio module exists for."""
    import asyncio

    from tors.documents import aio

    assert "pdf_extract" in aio.__all__
    assert "pdf_page_count" in aio.__all__
    assert asyncio.iscoroutinefunction(aio.pdf_extract)
    assert asyncio.iscoroutinefunction(aio.pdf_page_count)


def test_aio_pdf_extracts(tmp_path: Path) -> None:
    import asyncio

    from tors.documents import aio

    path = _write(tmp_path, "digital.pdf", _digital_pdf("Awaitable extraction."))
    pages, markdown = asyncio.run(aio.pdf_extract(path))
    assert pages == ["Awaitable extraction."]
    assert "Awaitable extraction." in markdown
    assert asyncio.run(aio.pdf_page_count(path)) == 1
