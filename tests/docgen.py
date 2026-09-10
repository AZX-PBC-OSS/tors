"""The randomized document generator for the cross-format validation suite:
seeded, reproducible documents across every engine format with varying
format, structure, columns, links, and formatting: each carrying its own
ground truth (the content units emitted), so the gates measure parse
quality as alignment: what went in must come out, nothing fabricated,
structure rendered as structure.

Two generation sources, by lane:

- The randomized lane (this module's ``generators``) writes documents with
  the standard maintained writer libraries (fpdf2, python-docx, openpyxl,
  python-pptx (dev-group only, nothing ships)) so the engines are measured
  against real authoring-tool bytes: real Word styles and numbering, real
  pptx layouts and notes parts, real xlsx workbooks, real PDF content
  streams with link annotations. Seeded via ``random.Random(seed)``; every
  failure reproduces from the seed in the test id. These documents are
  never committed (writer output is not byte-stable), so determinism is not
  required: ground truth is recorded in memory at emission time.
- The committed corpus (``documents.ENGINES_CORPUS``) keeps the
  hand-built byte-deterministic fixtures for the pin tests: the minimal
  subsets, pinned by regeneration, owned by the fixed-fixture lane.

Ground truth is generator-vs-engine, never engine-vs-engine: the generator
appends each unit to the ``GroundTruth`` as it writes it into the document,
and the gates normalize engine output (entity unescape, backslash-escape
removal, whitespace collapse) before matching.
"""

from __future__ import annotations

import csv
import io
import random
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field

from documents import (
    _XML_DECL,
    _odt_bytes,
    _xml_escape,
)

# --- the adversarial content vocabulary (pure-ASCII source per the repo
# convention; non-ASCII built from chr()) -------------------------------------

WORDS = (
    "transformer",
    "bushing",
    "dielectric",
    "outage",
    "torque",
    "sample",
    "interval",
    "maintenance",
    "schedule",
    "field",
    "review",
    "quarterly",
    "figures",
    "readiness",
    "checkpoint",
    "handbook",
    "units",
    "program",
)
_ADVERSARIAL = (
    "Torque &amp; bushing figures &copy; readiness &#233;.",
    "asterisks *inside* plain_text _here_",
    "pipes | inside | plain text",
    "brackets [not a link] (not a url)",
    "backticks `inline` shaped text",
    "CJK " + chr(0x6771) + chr(0x4EAC) + " office",
    "emoji lab lead " + chr(0x1F469) + chr(0x200D) + chr(0x1F52C),
    "caf" + chr(0xE9) + " figures for the " + chr(0xE9) + "quipe.",
)
# The WinAnsi-renderable subset (fpdf2 core fonts): CJK and ZWJ emoji cannot
# be drawn with the standard PDF fonts, so the PDF lane draws from this
# filtered pool.
_PDF_ADVERSARIAL = tuple(line for line in _ADVERSARIAL if all(ord(char) < 0x100 for char in line))
_LINK_URLS = (
    "https://handbook.example.com/torque",
    "https://docs.example.com/programs/2026",
    "https://internal.example.invalid/wiki/units",
)


@dataclass
class GroundTruth:
    """What the generator emitted, as the alignment oracle. Units carry the
    content as plain TEXT; the gates normalize the engines' output before
    matching. ``blank_pages`` is 0-based PDF page indexing."""

    kind: str
    headings: list[tuple[int, str]] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    list_items: list[str] = field(default_factory=list)
    table_cells: list[str] = field(default_factory=list)
    link_labels: list[str] = field(default_factory=list)
    link_urls: list[str] = field(default_factory=list)
    bold_spans: list[str] = field(default_factory=list)
    code_lines: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    pages: int = 0
    blank_pages: tuple[int, ...] = ()


def _sentence(
    rng: random.Random,
    *,
    adversarial: bool = True,
    pool: tuple[str, ...] = _ADVERSARIAL,
) -> str:
    if adversarial and rng.random() < 0.35:
        return rng.choice(pool)
    count = rng.randint(4, 9)
    words = [rng.choice(WORDS) for _ in range(count)]
    words[0] = words[0].capitalize()
    text = " ".join(words) + "."
    if rng.random() < 0.15:
        return text + " " + " ".join(rng.choice(WORDS) for _ in range(40)) + "."
    return text


def _cell(rng: random.Random) -> str:
    if rng.random() < 0.2:
        return ""
    return rng.choice(WORDS) + ("" if rng.random() < 0.5 else f" {rng.randint(1, 999)}")


def _rtf_literal(text: str) -> str:
    out: list[str] = []
    for char in text:
        if char in "\\{}":
            out.append("\\" + char)
        elif ord(char) > 127:
            out.append(f"\\'{ord(char):02x}")
        else:
            out.append(char)
    return "".join(out)


# --- PDF (fpdf2: real content streams, real /Link annotations) ---------------


def random_pdf(seed: int) -> tuple[bytes, GroundTruth]:
    """A 1-4 page PDF via fpdf2: a random subset of pages left blank (the
    scanned-page shape, recorded 0-based for the routing alignment check),
    optional oversize-font headings, 1-3 paragraphs each (occasionally
    wrapped long lines), an optional two-column page, an optional /Link
    annotation over drawn text."""
    from fpdf import FPDF

    rng = random.Random(seed)
    truth = GroundTruth(kind="pdf")
    pages = rng.randint(1, 4)
    blank = tuple(sorted(rng.sample(range(pages), k=rng.randint(0, min(1, pages)))))
    truth.blank_pages = blank
    truth.pages = pages

    pdf = FPDF()
    pdf.set_auto_page_break(False)
    for page in range(pages):
        pdf.add_page()
        if page in blank:
            continue
        if rng.random() < 0.7:
            heading = _sentence(rng, adversarial=False).removesuffix(".")
            truth.headings.append((1, heading))
            pdf.set_font("helvetica", size=rng.choice((18, 24, 28)))
            pdf.set_xy(20, 20)
            pdf.cell(0, 12, heading)
            y = 40
        else:
            y = 20
        two_column = rng.random() < 0.3
        for _ in range(rng.randint(1, 3)):
            text = _sentence(rng, pool=_PDF_ADVERSARIAL)
            truth.paragraphs.append(text)
            pdf.set_font("helvetica", size=11)
            if two_column:
                pdf.set_xy(20, y)
                pdf.multi_cell(80, 6, f"LEFT {text}")
                pdf.set_xy(110, y)
                pdf.multi_cell(80, 6, f"RIGHT {text}")
                truth.paragraphs.append(f"LEFT {text}")
                truth.paragraphs.append(f"RIGHT {text}")
                y += 30
            else:
                pdf.set_xy(20, y)
                pdf.multi_cell(170, 6, text)
                y += 24
        if rng.random() < 0.4:
            label = f"Visit the {rng.choice(WORDS)} {rng.choice(WORDS)}"
            url = rng.choice(_LINK_URLS)
            truth.link_labels.append(label)
            truth.link_urls.append(url)
            pdf.set_font("helvetica", size=11)
            pdf.set_xy(20, 270)
            pdf.cell(0, 8, label, link=url)
    return bytes(pdf.output()), truth


# --- docx (python-docx: real Heading styles, list styles, tables, hyperlinks) --


def _docx_hyperlink(paragraph, url: str, label: str) -> None:
    """python-docx has no first-class hyperlink API; the standard low-level
    add (relationship + w:hyperlink run), the same shape Word itself
    writes."""
    import docx.oxml.shared
    from docx.opc.constants import RELATIONSHIP_TYPE

    part = paragraph.part
    relationship = part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    hyperlink = docx.oxml.shared.OxmlElement("w:hyperlink")
    hyperlink.set(docx.oxml.shared.qn("r:id"), relationship)
    run = paragraph.add_run(label)
    run.font.underline = True
    hyperlink.append(run._element)
    paragraph._p.append(hyperlink)


def random_docx(seed: int) -> tuple[bytes, GroundTruth]:
    """A real python-docx document: built-in Heading 1/2 styles (real
    styles.xml), adversarial paragraphs with bold/italic runs, List Bullet /
    List Number styled items (real numbering), a table with empty cells, an
    optional external hyperlink."""
    from docx import Document

    rng = random.Random(seed)
    truth = GroundTruth(kind="docx")
    document = Document()
    for _ in range(rng.randint(1, 2)):
        level = rng.randint(1, 2)
        heading = _sentence(rng, adversarial=False).removesuffix(".")
        truth.headings.append((level, heading))
        document.add_heading(heading, level=level)
    for _ in range(rng.randint(1, 3)):
        paragraph = document.add_paragraph()
        lead = paragraph.add_run(_sentence(rng))
        if rng.random() < 0.4:
            emphasis_word = rng.choice(WORDS)
            paragraph.add_run(f" {emphasis_word} figures").bold = True
            truth.bold_spans.append(f"{emphasis_word} figures")
        if rng.random() < 0.3:
            paragraph.add_run(f" {rng.choice(WORDS)} sample").italic = True
        truth.paragraphs.append(lead.text)
    for style in ("List Bullet", "List Number"):
        if rng.random() < 0.7:
            for _ in range(rng.randint(1, 3)):
                item = _sentence(rng, adversarial=False).removesuffix(".")
                truth.list_items.append(item)
                document.add_paragraph(item, style=style)
    rows, cols = rng.randint(1, 4), rng.randint(1, 4)
    table = document.add_table(rows=rows, cols=cols)
    for row in range(rows):
        for col in range(cols):
            cell = _cell(rng)
            if cell:
                truth.table_cells.append(cell)
            table.cell(row, col).text = cell
    if rng.random() < 0.6:
        label = f"{rng.choice(WORDS)} {rng.choice(WORDS)}"
        url = rng.choice(_LINK_URLS)
        truth.link_labels.append(label)
        truth.link_urls.append(url)
        paragraph = document.add_paragraph("See the ")
        _docx_hyperlink(paragraph, url, label)
        paragraph.add_run(".")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue(), truth


# --- pptx (python-pptx: real layouts, leveled body text, speaker notes) -------


def random_pptx(seed: int) -> tuple[bytes, GroundTruth]:
    """A real python-pptx deck on the Title-and-Content layout: 1-3 slides,
    a title each, 0-3 leveled body paragraphs, speaker notes on a random
    slide."""
    from pptx import Presentation

    rng = random.Random(seed)
    truth = GroundTruth(kind="pptx")
    presentation = Presentation()
    for _ in range(rng.randint(1, 3)):
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        title = _sentence(rng, adversarial=False).removesuffix(".")
        truth.headings.append((2, title))
        slide.shapes.title.text = title
        body = slide.placeholders[1].text_frame
        first = True
        for _ in range(rng.randint(0, 3)):
            text = _sentence(rng, adversarial=False).removesuffix(".")
            truth.paragraphs.append(text)
            paragraph = body.paragraphs[0] if first else body.add_paragraph()
            first = False
            paragraph.text = text
            paragraph.level = rng.randint(0, 1)
        if rng.random() < 0.7:
            note = _sentence(rng, adversarial=False).removesuffix(".")
            truth.notes.append(note)
            slide.notes_slide.notes_text_frame.text = note
    buffer = io.BytesIO()
    presentation.save(buffer)
    return buffer.getvalue(), truth


# --- xlsx (openpyxl: real workbook, shared strings, sheet names) --------------


def random_xlsx(seed: int) -> tuple[bytes, GroundTruth]:
    rng = random.Random(seed)
    truth = GroundTruth(kind="xlsx")
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    for row in range(1, rng.randint(2, 6)):
        for column in range(1, rng.randint(2, 5)):
            cell = _cell(rng)
            if cell:
                truth.table_cells.append(cell)
            sheet.cell(row=row, column=column, value=cell or None)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue(), truth


# --- odt (the minimal deterministic zip: no maintained writer worth its API) ---


def random_odt(seed: int) -> tuple[bytes, GroundTruth]:
    rng = random.Random(seed)
    truth = GroundTruth(kind="odt")
    office = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    text_ns = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    table_ns = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
    parts: list[str] = []
    for _ in range(rng.randint(1, 2)):
        level = rng.randint(1, 2)
        heading = _sentence(rng, adversarial=False).removesuffix(".")
        truth.headings.append((level, heading))
        parts.append(f'<text:h text:outline-level="{level}">{heading}</text:h>')
    for _ in range(rng.randint(1, 3)):
        paragraph = _sentence(rng)
        truth.paragraphs.append(paragraph)
        parts.append(f"<text:p>{paragraph}</text:p>")
    items = [_sentence(rng, adversarial=False).removesuffix(".") for _ in range(rng.randint(1, 3))]
    truth.list_items.extend(items)
    parts.append(
        "<text:list>"
        + "".join(f"<text:list-item><text:p>{item}</text:p></text:list-item>" for item in items)
        + "</text:list>"
    )
    rows, cols = rng.randint(1, 3), rng.randint(1, 3)
    cells: list[str] = []
    table = ['<table:table table:name="generated">']
    for _row in range(rows):
        row_cells: list[str] = []
        for _col in range(cols):
            cell = _cell(rng)
            cells.append(cell)
            row_cells.append(f"<table:table-cell><text:p>{cell}</text:p></table:table-cell>")
        table.append("<table:table-row>" + "".join(row_cells) + "</table:table-row>")
    table.append("</table:table>")
    truth.table_cells.extend(cell for cell in cells if cell)
    parts.append("".join(table))
    content = _XML_DECL + (
        f'<office:document-content xmlns:office="{office}" xmlns:text="{text_ns}"'
        f' xmlns:table="{table_ns}"><office:body><office:text>'
        + "".join(parts)
        + "</office:text></office:body></office:document-content>"
    )
    return _odt_bytes([("content.xml", content.encode("utf-8"))]), truth


# --- csv (stdlib writer: spec-correct quoting edge cases) ---------------------


def random_csv(seed: int) -> tuple[bytes, GroundTruth]:
    rng = random.Random(seed)
    truth = GroundTruth(kind="csv")
    rows, cols = rng.randint(2, 5), rng.randint(2, 4)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for _row in range(rows):
        row: list[str] = []
        for _col in range(cols):
            cell = _cell(rng)
            if cell and rng.random() < 0.2:
                cell += ', with "quotes"'
            row.append(cell)
            if cell:
                truth.table_cells.append(cell)
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8"), truth


# --- rtf (the minimal subset: no maintained writer library exists) ------------


def random_rtf(seed: int) -> tuple[bytes, GroundTruth]:
    """A minimal well-formed RTF: fonttbl header, fs24 body, paragraphs as
    ``text\\par``, cp1252 escapes for non-ASCII (CJK/emoji excluded: the
    minimal subset cannot carry them)."""
    rng = random.Random(seed)
    truth = GroundTruth(kind="rtf")
    pool = _PDF_ADVERSARIAL  # the cp1252-renderable subset
    parts: list[str] = []
    for _ in range(rng.randint(1, 4)):
        paragraph = _sentence(rng, pool=pool)
        truth.paragraphs.append(paragraph)
        # The trailing space is the RTF control-word delimiter: without it
        # "\par" + "Next" parses as one unknown control word "\parNext" and
        # the paragraph's first word is swallowed (measured: anydoc drops
        # the glued text). RTF consumes exactly one space after a control
        # word, so no space reaches the document text.
        parts.append(_rtf_literal(paragraph) + "\\par ")
    body = "".join(parts)
    raw = (
        "{\\rtf1\\ansi\\ansicpg1252\\deff0{\\fonttbl{\\f0\\froman Times New Roman;}}\n"
        "\\f0\\fs24 " + body + "}\n"
    ).encode("cp1252", errors="replace")
    return raw, truth


# --- the OOXML alias containers (docm / xlsm / ppsx) --------------------------
#
# The macro-enabled / slide-show variants of the writer libraries' own
# output: the same package with [Content_Types].xml's main-document
# override respelled (the only difference between the containers), so the
# fuzz lane measures the aliases over real writer bytes. Ground truth is
# the base generator's unchanged: the alias is the base document, only
# the container's content-type name differs.

_CT_DOCX_MAIN = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
)
_CT_DOCM_MAIN = "application/vnd.ms-word.document.macroEnabled.main+xml"
_CT_XLSX_MAIN = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"
)
_CT_XLSM_MAIN = "application/vnd.ms-excel.sheet.macroEnabled.main+xml"
_CT_PPTX_MAIN = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
)
_CT_PPSX_MAIN = (
    "application/vnd.openxmlformats-officedocument.presentationml"
    ".slideshow.macroEnabled.main+xml"
)


def _ooxml_alias(raw: bytes, base_ct: str, alias_ct: str) -> bytes:
    """Rewrite [Content_Types].xml's main-document override inside a
    writer-produced OOXML package: the macro-enabled/show variants are the
    same container with that one attribute respelled."""
    source = zipfile.ZipFile(io.BytesIO(raw))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = data.replace(base_ct.encode(), alias_ct.encode())
            archive.writestr(item, data)
    return out.getvalue()


def random_docm(seed: int) -> tuple[bytes, GroundTruth]:
    """random_docx's document in the macro-enabled container: the alias
    the engines convert as docx on both lanes (anydoc and office_oxide)."""
    raw, truth = random_docx(seed)
    return _ooxml_alias(raw, _CT_DOCX_MAIN, _CT_DOCM_MAIN), truth


def random_xlsm(seed: int) -> tuple[bytes, GroundTruth]:
    """random_xlsx's workbook in the macro-enabled container: the alias
    the engines convert as xlsx on both lanes (anydoc and office_oxide)."""
    raw, truth = random_xlsx(seed)
    return _ooxml_alias(raw, _CT_XLSX_MAIN, _CT_XLSM_MAIN), truth


def random_ppsx(seed: int) -> tuple[bytes, GroundTruth]:
    """random_pptx's deck in the slide-show container: the alias the
    auto/anydoc lane converts as pptx (office_oxide refuses the slideshow
    content type; the fuzz lane's auto conversion is the coverage)."""
    raw, truth = random_pptx(seed)
    return _ooxml_alias(raw, _CT_PPTX_MAIN, _CT_PPSX_MAIN), truth


# --- html (string templates: generation is the format) ------------------------


def random_html(seed: int) -> tuple[bytes, GroundTruth]:
    """The polluted-head shape (title/style/script: must never leak) plus a
    random body: headings, adversarial paragraphs with strong/em/code
    spans, a nested list, a table, an optional link, an optional code
    block."""
    rng = random.Random(seed)
    truth = GroundTruth(kind="html")
    parts: list[str] = [
        "<!DOCTYPE html>",
        "<html><head><title>Junk Title</title>",
        "<style>body { color: red; }</style>",
        '<script>console.log("tracking junk");</script></head>',
        "<body>",
    ]
    for _ in range(rng.randint(1, 2)):
        level = rng.randint(1, 2)
        heading = _sentence(rng, adversarial=False).removesuffix(".")
        truth.headings.append((level, heading))
        parts.append(f"<h{level}>{_xml_escape(heading)}</h{level}>")
    for _ in range(rng.randint(1, 3)):
        paragraph = _sentence(rng)
        truth.paragraphs.append(paragraph)
        rendered = _xml_escape(paragraph)
        if rng.random() < 0.4:
            span = f"{rng.choice(WORDS)} figures"
            truth.bold_spans.append(span)
            rendered += f" <strong>{span}</strong>"
        if rng.random() < 0.3:
            rendered += f" <em>{rng.choice(WORDS)} sample</em>"
        parts.append(f"<p>{rendered}</p>")
    items = [_sentence(rng, adversarial=False).removesuffix(".") for _ in range(rng.randint(1, 3))]
    truth.list_items.extend(items)
    nested = _sentence(rng, adversarial=False).removesuffix(".")
    truth.list_items.append(nested)
    # Every flat item gets its own <li> (the emitted set is the recorded
    # set: the alignment gate exists to catch exactly this drift), with
    # the nested one inside the last.
    outer = "".join(f"<li>{_xml_escape(item)}</li>" for item in items[:-1])
    outer += f"<li>{_xml_escape(items[-1])}<ul><li>{_xml_escape(nested)}</li></ul></li>"
    parts.append(f"<ul>{outer}</ul>")
    rows, cols = rng.randint(1, 3), rng.randint(1, 3)
    cells: list[str] = []
    table = ["<table><thead><tr>"]
    for _col in range(cols):
        cell = _cell(rng)
        cells.append(cell)
        table.append(f"<th>{_xml_escape(cell)}</th>")
    table.append("</tr></thead><tbody>")
    for _row in range(rows):
        table.append("<tr>")
        for _col in range(cols):
            cell = _cell(rng)
            cells.append(cell)
            table.append(f"<td>{_xml_escape(cell)}</td>")
        table.append("</tr>")
    table.append("</tbody></table>")
    truth.table_cells.extend(cell for cell in cells if cell)
    parts.append("".join(table))
    if rng.random() < 0.6:
        label = f"{rng.choice(WORDS)} {rng.choice(WORDS)}"
        url = rng.choice(_LINK_URLS)
        truth.link_labels.append(label)
        truth.link_urls.append(url)
        parts.append(f'<p>See the <a href="{url}">{label}</a>.</p>')
    if rng.random() < 0.5:
        code = "value = " + str(rng.randint(1, 99))
        truth.code_lines.append(code)
        parts.append(f"<pre><code>{code}</code></pre>")
    parts.append("</body></html>")
    return ("\n".join(parts) + "\n").encode("utf-8"), truth


GENERATORS: dict[str, Callable[[int], tuple[bytes, GroundTruth]]] = {
    "pdf": random_pdf,
    "docx": random_docx,
    "docm": random_docm,
    "pptx": random_pptx,
    "ppsx": random_ppsx,
    "xlsx": random_xlsx,
    "xlsm": random_xlsm,
    "odt": random_odt,
    "csv": random_csv,
    "rtf": random_rtf,
    "html": random_html,
}

FILE_EXTENSIONS: dict[str, str] = {
    "pdf": ".pdf",
    "docx": ".docx",
    "docm": ".docm",
    "pptx": ".pptx",
    "ppsx": ".ppsx",
    "xlsx": ".xlsx",
    "xlsm": ".xlsm",
    "odt": ".odt",
    "csv": ".csv",
    "rtf": ".rtf",
    "html": ".html",
}
