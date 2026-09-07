"""The deterministic real-document corpus: five documents in the
formats downstream pipelines actually hold (markdown, RTF, DOCX, XLSX, PDF),
GENERATED here byte-deterministically and extracted here with readers
built from the standard library alone (zipfile + xml.etree for the OOXML
pair, zlib for the PDF's FlateDecode stream, plain text for RTF/MD). No new
dependencies: the point is to prove tors's surface over REAL document bytes
and really-extracted text, not to ship a document parser.

Determinism contract (pinned by tests/test_documents.py): every generator is
byte-stable: fixed zip metadata (``ZipInfo`` with ``date_time`` pinned to
1980-01-01, ``create_system``/``external_attr`` set explicitly so the bytes
do not depend on the building platform), the PDF's xref offsets computed
programmatically from its own fixed objects, so regenerating the corpus is
byte-identical, and the committed files under ``tests/corpus/`` are exactly
``CORPUS`` (a test re-derives and byte-compares them).

DRY spine: each document's TEXT is defined ONCE as module constants (pure
ASCII source; every non-ASCII character built from ``chr()`` (the repo
convention, because visually-ambiguous literals are how pins rot) and both
the generator and ``EXPECTED_TEXT`` (the extraction oracle) derive from the
SAME constants, so a generator edit that changed the text would fail the
extraction pin instead of silently laundering through.

Extractor scope: each reader covers exactly the subset its
generator emits (the RTF control words below; ``w:p``/``w:r``/``w:t``/
``w:tab``/``w:br``; shared-string cells; one ``Tj`` per line with
``\\( \\) \\\\`` and octal escapes in WinAnsi). They are NOT general
parsers for those formats. One intentional content split: CJK and emoji live
in every format EXCEPT the PDF: WinAnsi-encoded Helvetica cannot
represent them (the constraint of the minimal-PDF subset), so the
PDF carries ASCII + latin-1 text only.
"""

from __future__ import annotations

import io
import re
import zipfile
import zlib
from pathlib import Path
from xml.etree import ElementTree

# Non-ASCII content constants (pure-ASCII source, per the repo convention).
_E_ACUTE = chr(0xE9)  # é, precomposed
_TOKYO = chr(0x6771)  # 東
_KYOTO = chr(0x4EAC)  # 京
_WOMAN = chr(0x1F469)
_ZWJ = chr(0x200D)
_MICROSCOPE = chr(0x1F52C)
_NBSP = chr(0xA0)
_EM_DASH = chr(0x2014)

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

# --- the shared content spine (every document tells the same story) ---------------
#
# The sentences carry the messy shapes tors exists to clean:
# entity references, CRLF endings, [ \t] runs before newlines, a 3+ blank-line
# run, leading/trailing whitespace (incl. an NBSP), and a tab-joined figure row.

MD_HEADING = "# Quarterly Review Q3 2026"
MD_INTRO = (
    "The quarterly oil sample interval for field outages was adjusted after "
    "the bushing torque specifications changed."
)
MD_ENTITY_SENTENCE = (
    "Torque &amp; bushing figures &copy; " + _E_ACUTE + "quipe readiness &#233;."
)
MD_CRLF_BLOCK = (
    "Windows-origin excerpt: adjusted after the bushing scan.\r\n"
    "Torque windows close within fourteen days.\r\n"
)
MD_SLOPPY_BLOCK = "Interval drift  \nseen after the outage.\t\n"
MD_CJK_LINE = _TOKYO + _KYOTO + " office: " + _E_ACUTE + "quipe readiness."
MD_EMOJI_LINE = "Lab lead: " + _WOMAN + _ZWJ + _MICROSCOPE
MD_FIGURES = "figures: interval 14\toutage 3\tspec 2"

MARKDOWN_TEXT = "\n".join(
    [
        MD_HEADING,
        "",
        MD_INTRO,
        "",
        MD_ENTITY_SENTENCE,
        "",
        MD_CRLF_BLOCK,
        MD_SLOPPY_BLOCK,
        "\n\n\n\n",
        MD_CJK_LINE,
        "",
        MD_EMOJI_LINE,
        "",
        MD_FIGURES,
        "\n" + _NBSP,
    ]
)

# The pipeline-cleaned spelling of MARKDOWN_TEXT, derived by hand from the
# stages (fold CRLF, drop [ \t] before newlines, collapse 3+ newline runs to
# two, strip the trailing whitespace incl. the NBSP): the pinned normalize
# expectation, kept beside its input so the two cannot drift apart. Note the
# blank line between the CRLF block and the sloppy block: the block's own
# trailing \r\n plus the joining \n is a legitimate TWO-newline run, which
# the collapse stage (3+ only) correctly leaves alone.
MARKDOWN_NORMALIZED = "\n".join(
    [
        MD_HEADING,
        "",
        MD_INTRO,
        "",
        MD_ENTITY_SENTENCE,
        "",
        "Windows-origin excerpt: adjusted after the bushing scan.",
        "Torque windows close within fourteen days.",
        "",
        "Interval drift",
        "seen after the outage.",
        "",
        MD_CJK_LINE,
        "",
        MD_EMOJI_LINE,
        "",
        MD_FIGURES,
    ]
)

# The entity-decoded spelling of the markdown text (html.unescape semantics:
# &amp; &copy; &#233; decode; everything else verbatim).
MARKDOWN_UNESCAPED = MARKDOWN_TEXT.replace("&amp;", "&").replace("&copy;", chr(0xA9)).replace(
    "&#233;", _E_ACUTE
)

# --- RTF ---------------------------------------------------------------------------
#
# The generator's subset: {\rtf1\ansi, \pard, \par line breaks, signed-16-bit
# \uN? escapes (a negative N means codepoint 65536+N), \'hh cp1252 hex
# escapes, literal text, and grouping braces.

RTF_PARAGRAPHS = [
    MD_INTRO,
    "Torque, caf\\u233? and \\'e9quipe figures.",  # both escape spellings of é
    "\\u26481?\\u20140? office: caf\\u233? readiness.",  # 東京 via \uN escapes
    "Windows close within fourteen days.",
]

RTF_DOCUMENT = (
    "{\\rtf1\\ansi\n\\pard\n"
    + "".join(paragraph + "\\par\n" for paragraph in RTF_PARAGRAPHS)
    + "}\n"
)


def _rtf_unescape(text: str) -> str:
    """The RTF subset extractor: control words are consumed WITH their
    optional numeric parameter and their single delimiter whitespace (the
    RTF rule: ``\\rtf1`` and ``\\par `` never leak their parameter or
    delimiter into the text), ``\\par`` breaks lines, ``\\uN?`` becomes
    chr(N) with the signed-16-bit rule, ``\\'hh`` becomes the cp1252 byte,
    braces vanish, and literal text passes through. Scoped to the
    generator's subset, not a general RTF reader."""
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "{}":
            i += 1
        elif ch == "\\":
            i += 1
            if i < len(text) and text[i] == "u":
                match = re.match(r"u(-?\d+)\??", text[i:])
                if match is None:  # pragma: no cover - the generator always matches
                    raise ValueError(f"malformed \\u escape at {i}")
                code = int(match.group(1))
                out.append(chr(code if code >= 0 else 65536 + code))
                i += match.end()
            elif i < len(text) and text[i] == "'":
                out.append(chr(int(text[i + 1 : i + 3], 16)))  # cp1252 == latin-1 subset used
                i += 3
            else:
                control = re.match(r"[a-zA-Z]+(-?\d+)?", text[i:])
                if control is not None:  # \par, \pard, \rtf1, ...: dropped
                    i += control.end()
                    if control.group(0).rstrip("0123456789-") == "par":
                        out.append("\n")
                    if i < len(text) and text[i] in " \n\r\t":
                        i += 1  # the control word's single delimiter whitespace
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def extract_rtf(raw: bytes) -> str:
    text = _rtf_unescape(raw.decode("ascii"))
    return text.strip("\n")


RTF_TEXT = "\n".join(
    [
        MD_INTRO,
        "Torque, caf" + _E_ACUTE + " and " + _E_ACUTE + "quipe figures.",
        _TOKYO + _KYOTO + " office: caf" + _E_ACUTE + " readiness.",
        "Windows close within fourteen days.",
    ]
)

# --- DOCX (minimal OOXML: zip + word/document.xml) ---------------------------------

DOCX_PARAGRAPHS = [
    MD_HEADING.removeprefix("# "),
    MD_INTRO,
    MD_ENTITY_SENTENCE,
    MD_CJK_LINE,
    MD_EMOJI_LINE,
    "figures: interval 14\toutage 3\tspec 2",  # rendered with a real w:tab
    "hard break line one\nhard break line two",  # rendered with a real w:br
]

_XML_DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _docx_paragraph(text: str) -> str:
    """One w:p; the figure row's tab and the hard-break line's newline are
    real w:tab/w:br runs (XML parsers normalize literal \\r/\\n in text, so
    structural breaks must be elements)."""
    _PRESERVE = '<w:t xml:space="preserve">'
    if "\t" in text:
        left, _, right = text.partition("\t")
        runs = (
            f"<w:r>{_PRESERVE}{_xml_escape(left)}</w:t>"
            f"<w:tab/>{_PRESERVE}{_xml_escape(right)}</w:t></w:r>"
        )
    elif "\n" in text:
        left, _, right = text.partition("\n")
        runs = (
            f"<w:r>{_PRESERVE}{_xml_escape(left)}</w:t>"
            f"<w:br/>{_PRESERVE}{_xml_escape(right)}</w:t></w:r>"
        )
    else:
        runs = f"<w:r>{_PRESERVE}{_xml_escape(text)}</w:t></w:r>"
    return f"<w:p>{runs}</w:p>"


DOCX_DOCUMENT_XML = (
    _XML_DECL
    + f'<w:document xmlns:w="{_W_NS}"><w:body>'
    + "".join(_docx_paragraph(paragraph) for paragraph in DOCX_PARAGRAPHS)
    + "</w:body></w:document>\n"
)

_CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_DOC_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_DOCX_MAIN_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"

DOCX_CONTENT_TYPES = (
    _XML_DECL
    + f'<Types xmlns="{_CT_NS}">'
    + '<Default Extension="rels" '
    + 'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    + '<Default Extension="xml" ContentType="application/xml"/>'
    + '<Override PartName="/word/document.xml" '
    + f'ContentType="{_DOCX_MAIN_CT}"/>'
    + "</Types>\n"
)

DOCX_RELS = (
    _XML_DECL
    + f'<Relationships xmlns="{_REL_NS}">'
    + f'<Relationship Id="rId1" Type="{_DOC_NS}/officeDocument" '
    + 'Target="word/document.xml"/>'
    + "</Relationships>\n"
)


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    """A byte-deterministic zip: every entry carries a pinned ZipInfo (the
    1980-01-01 DOS epoch, an explicit create_system and external_attr) so
    the bytes depend on nothing but the entries themselves."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return buffer.getvalue()


def generate_docx() -> bytes:
    return _zip_bytes(
        [
            ("[Content_Types].xml", DOCX_CONTENT_TYPES.encode("utf-8")),
            ("_rels/.rels", DOCX_RELS.encode("utf-8")),
            ("word/document.xml", DOCX_DOCUMENT_XML.encode("utf-8")),
        ]
    )


def extract_docx(raw: bytes) -> str:
    """The minimal-OOXML reader: word/document.xml via xml.etree; each w:p's
    runs concatenate (w:t text, w:tab → \\t, w:br → \\n); paragraphs join
    with \\n. Scoped to the generator's elements."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    ns = f"{{{_W_NS}}}"
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{ns}p"):
        parts: list[str] = []
        for child in paragraph.iter():
            if child.tag == f"{ns}t":
                parts.append(child.text or "")
            elif child.tag == f"{ns}tab":
                parts.append("\t")
            elif child.tag in (f"{ns}br", f"{ns}cr"):
                parts.append("\n")
        paragraphs.append("".join(parts))
    return "\n".join(paragraphs)


DOCX_TEXT = "\n".join(DOCX_PARAGRAPHS)

# --- XLSX (minimal OOXML: workbook + one sheet + shared strings) --------------------

XLSX_SHARED_STRINGS = [
    "Quarterly oil sample interval",
    "Torque and bushing " + _E_ACUTE + " figures",
    _TOKYO + _KYOTO + " office readiness",
    "figures: interval 14\toutage 3\tspec 2",
    MD_ENTITY_SENTENCE,
]
XLSX_ROWS = [[0, 1], [2, 3], [4]]

_PKG_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _sst_xml() -> str:
    items = "".join(f"<si><t>{_xml_escape(text)}</t></si>" for text in XLSX_SHARED_STRINGS)
    return (
        _XML_DECL
        + f'<sst xmlns="{_MAIN_NS}" count="{len(XLSX_SHARED_STRINGS)}" '
        + f'uniqueCount="{len(XLSX_SHARED_STRINGS)}">{items}</sst>\n'
    )


def _sheet_xml() -> str:
    rows = []
    for row_index, row in enumerate(XLSX_ROWS, start=1):
        cells = "".join(
            f'<c r="{chr(ord("A") + column_index)}{row_index}" t="s"><v>{value}</v></c>'
            for column_index, value in enumerate(row)
        )
        rows.append(f'<row r="{row_index}">{cells}</row>')
    sheet_data = "".join(rows)
    return (
        _XML_DECL
        + f'<worksheet xmlns="{_MAIN_NS}"><sheetData>{sheet_data}</sheetData></worksheet>\n'
    )


def generate_xlsx() -> bytes:
    workbook = (
        _XML_DECL
        + f'<workbook xmlns="{_MAIN_NS}" xmlns:r="{_DOC_NS}">'
        + '<sheets><sheet name="Samples" sheetId="1" r:id="rId1"/></sheets></workbook>\n'
    )
    workbook_rels = (
        _XML_DECL
        + f'<Relationships xmlns="{_PKG_NS}">'
        + f'<Relationship Id="rId1" Type="{_DOC_NS}/worksheet" '
        + 'Target="worksheets/sheet1.xml"/>'
        + f'<Relationship Id="rId2" Type="{_DOC_NS}/sharedStrings" '
        + 'Target="sharedStrings.xml"/>'
        + "</Relationships>\n"
    )
    content_types = (
        _XML_DECL
        + f'<Types xmlns="{_CT_NS}">'
        + '<Default Extension="rels" '
        + 'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        + '<Default Extension="xml" ContentType="application/xml"/>'
        + '<Override PartName="/xl/workbook.xml" ContentType='
        + '"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        + '<Override PartName="/xl/worksheets/sheet1.xml" ContentType='
        + '"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        + '<Override PartName="/xl/sharedStrings.xml" ContentType='
        + '"application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        + "</Types>\n"
    )
    rels = (
        _XML_DECL
        + f'<Relationships xmlns="{_PKG_NS}">'
        + f'<Relationship Id="rId1" Type="{_DOC_NS}/officeDocument" '
        + 'Target="xl/workbook.xml"/>'
        + "</Relationships>\n"
    )
    return _zip_bytes(
        [
            ("[Content_Types].xml", content_types.encode("utf-8")),
            ("_rels/.rels", rels.encode("utf-8")),
            ("xl/workbook.xml", workbook.encode("utf-8")),
            ("xl/_rels/workbook.xml.rels", workbook_rels.encode("utf-8")),
            ("xl/worksheets/sheet1.xml", _sheet_xml().encode("utf-8")),
            ("xl/sharedStrings.xml", _sst_xml().encode("utf-8")),
        ]
    )


def extract_xlsx(raw: bytes) -> str:
    """The minimal-xlsx reader: shared strings by index, sheet rows joined
    with \\t and \\n. Scoped to t="s" cells (the generator emits no inline
    or numeric cells)."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        shared = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    ns = f"{{{_MAIN_NS}}}"
    strings = ["".join(t.text or "" for t in si.iter(f"{ns}t")) for si in shared.iter(f"{ns}si")]
    rows: list[str] = []
    for row in sheet.iter(f"{ns}row"):
        cells: list[str] = []
        for cell in row.iter(f"{ns}c"):
            if cell.get("t") == "s":
                value = cell.find(f"{ns}v")
                cells.append(strings[int(value.text or "0")])
        rows.append("\t".join(cells))
    return "\n".join(rows)


XLSX_TEXT = "\n".join(
    "\t".join(XLSX_SHARED_STRINGS[value] for value in row) for row in XLSX_ROWS
)

# --- PDF (minimal: FlateDecode content stream, one Tj per line) ---------------------


def _pdf_literal(text: str) -> str:
    """A PDF literal string for WinAnsi text: escapes ( ) \\, and every
    non-ASCII/ control byte as \\ddd octal (é → \\351 in cp1252)."""
    out: list[str] = []
    for ch in text:
        if ch in "()\\":
            out.append("\\" + ch)
            continue
        for byte in ch.encode("cp1252"):
            if 32 <= byte <= 126:
                out.append(chr(byte))
            else:
                out.append(f"\\{byte:03o}")
    return "".join(out)


PDF_LINES = [
    "Quarterly Review Q3 2026",
    MD_INTRO,
    "Torque and caf" + _E_ACUTE + " figures for the " + _E_ACUTE + "quipe.",
    "Windows close within fourteen days.",
]


def _pdf_content() -> bytes:
    moves = ["BT", "/F1 12 Tf", "72 720 Td"]
    for line in PDF_LINES:
        moves.append(f"({_pdf_literal(line)}) Tj")
        moves.append("0 -14 Td")
    moves.append("ET")
    return zlib.compress(("\n".join(moves) + "\n").encode("ascii"), 9)


def generate_pdf() -> bytes:
    """A minimal but structurally real PDF: catalog/pages/page, a Helvetica
    WinAnsi font, a FlateDecode content stream, and an xref table whose
    offsets are computed from the objects themselves (byte-deterministic)."""
    stream = _pdf_content()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        (
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" /Filter /FlateDecode >>\nstream\n"
            + stream
            + b"\nendstream"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
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


_PDF_STRING = re.compile(rb"\((?:\\.|[^\\()])*\)\s*Tj")
_PDF_ESCAPE = re.compile(rb"\\([()\\nrtbf]|(\d{1,3}))")


def _pdf_unescape(literal: bytes) -> bytes:
    def replace(match: re.Match[bytes]) -> bytes:
        token = match.group(1)
        if token == b"n":
            return b"\n"
        if token == b"r":
            return b"\r"
        if token == b"t":
            return b"\t"
        if token == b"b":
            return b"\b"
        if token == b"f":
            return b"\f"
        if token.isdigit():  # type: ignore[union-attr]
            return bytes([int(token, 8)])  # PDF \ddd literals are OCTAL (\351 = 233)
        return token

    return _PDF_ESCAPE.sub(replace, literal)


def extract_pdf(raw: bytes) -> str:
    """The minimal-PDF reader: every FlateDecode stream is decompressed and
    its Tj literals become lines (one Tj per line is the generator's shape).
    Scoped to that subset: no TJ arrays, no font encodings beyond the
    WinAnsi bytes the generator emits."""
    lines: list[str] = []
    for match in re.finditer(rb"stream\r?\n(.*?)\n?endstream", raw, re.DOTALL):
        content = zlib.decompress(match.group(1))
        for string in _PDF_STRING.finditer(content):
            literal = _pdf_unescape(string.group(0)[1 : string.group(0).rindex(b")")])
            lines.append(literal.decode("cp1252"))
    return "\n".join(lines)


PDF_TEXT = "\n".join(PDF_LINES)

# --- the corpus ---------------------------------------------------------------------


CORPUS: dict[str, bytes] = {
    "md": MARKDOWN_TEXT.encode("utf-8"),
    "rtf": RTF_DOCUMENT.encode("ascii"),
    "docx": generate_docx(),
    "xlsx": generate_xlsx(),
    "pdf": generate_pdf(),
}

EXPECTED_TEXT: dict[str, str] = {
    "md": MARKDOWN_TEXT,
    "rtf": RTF_TEXT,
    "docx": DOCX_TEXT,
    "xlsx": XLSX_TEXT,
    "pdf": PDF_TEXT,
}

_FILE_NAMES = {
    "md": "meeting_notes.md",
    "rtf": "notes.rtf",
    "docx": "report.docx",
    "xlsx": "samples.xlsx",
    "pdf": "report.pdf",
}


def extract(kind: str) -> str:
    """The extraction oracle: kind → the extracted text (the md/rtf kinds
    are plain text; docx/xlsx/pdf go through their readers)."""
    if kind == "md":
        return CORPUS["md"].decode("utf-8")
    if kind == "rtf":
        return extract_rtf(CORPUS["rtf"])
    if kind == "docx":
        return extract_docx(CORPUS["docx"])
    if kind == "xlsx":
        return extract_xlsx(CORPUS["xlsx"])
    if kind == "pdf":
        return extract_pdf(CORPUS["pdf"])
    raise ValueError(f"unknown corpus kind: {kind}")


def write_corpus(directory: Path = CORPUS_DIR) -> None:
    """Materialize the five documents (the committed copies under
    tests/corpus/ are exactly CORPUS; the pin test re-derives and
    byte-compares them)."""
    directory.mkdir(parents=True, exist_ok=True)
    for kind, raw in CORPUS.items():
        (directory / _FILE_NAMES[kind]).write_bytes(raw)
