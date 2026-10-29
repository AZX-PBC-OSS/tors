"""The deterministic real-document corpus: five documents in the
formats downstream pipelines actually hold (markdown, RTF, DOCX, XLSX, PDF),
generated here byte-deterministically and extracted here with readers
built from the standard library alone (zipfile + xml.etree for the OOXML
pair, zlib for the PDF's FlateDecode stream, plain text for RTF/md). No new
dependencies: the point is to prove tors's surface over real document bytes
and really-extracted text, not to ship a document parser.

Determinism contract (pinned by tests/test_documents.py): every generator is
byte-stable: fixed zip metadata (``ZipInfo`` with ``date_time`` pinned to
1980-01-01, ``create_system``/``external_attr`` set explicitly so the bytes
do not depend on the building platform), the PDF's xref offsets computed
programmatically from its own fixed objects, so regenerating the corpus is
byte-identical, and the committed files under ``tests/corpus/`` are exactly
``corpus`` (a test re-derives and byte-compares them).

dry spine: each document's TEXT is defined once as module constants (pure
ASCII source; every non-ASCII character built from ``chr()`` (the repo
convention, because visually-ambiguous literals are how pins rot) and both
the generator and ``EXPECTED_TEXT`` (the extraction oracle) derive from the
same constants, so a generator edit that changed the text would fail the
extraction pin instead of silently laundering through.

Extractor scope: each reader covers exactly the subset its
generator emits (the RTF control words below; ``w:p``/``w:r``/``w:t``/
``w:tab``/``w:br``; shared-string cells; one ``Tj`` per line with
``\\ (\\) \\\\`` and octal escapes in WinAnsi). They are not general
parsers for those formats. One intentional content split: CJK and emoji live
in every format except the PDF: WinAnsi-encoded Helvetica cannot
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
MD_ENTITY_SENTENCE = "Torque &amp; bushing figures &copy; " + _E_ACUTE + "quipe readiness &#233;."
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
# trailing \r\n plus the joining \n is a legitimate two-newline run, which
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
MARKDOWN_UNESCAPED = (
    MARKDOWN_TEXT.replace("&amp;", "&").replace("&copy;", chr(0xA9)).replace("&#233;", _E_ACUTE)
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
    """The RTF subset extractor: control words are consumed with their
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

# Shared OOXML content-type strings (every builder's [Content_Types].xml
# parts, base corpus and engines matrix alike; extracted so no single line
# carries the whole attribute string).
_CT_RELS = (
    '<Default Extension="rels" ContentType='
    '"application/vnd.openxmlformats-package.relationships+xml"/>'
)
_CT_XML = '<Default Extension="xml" ContentType="application/xml"/>'
_CT_DOCX_MAIN = (
    '<Override PartName="/word/document.xml" ContentType='
    '"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
)
_CT_PPTX_MAIN = (
    '<Override PartName="/ppt/presentation.xml" ContentType='
    '"application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
)
_CT_SLIDE = (
    '<Override PartName="/ppt/slides/slide{n}.xml" ContentType='
    '"application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
)
_CT_NOTES = (
    '<Override PartName="/ppt/notesSlides/notesSlide{n}.xml" ContentType='
    '"application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>'
)
_CT_XLSX_MAIN = (
    '<Override PartName="/xl/workbook.xml" ContentType='
    '"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
)
# The macro-enabled / slide-show container variants: the genuine OOXML
# aliases: docm/xlsm/ppsx are the same packages as docx/xlsx/pptx (every
# part identical) with [Content_Types].xml's main-document override naming
# the variant, so the engines convert them as their base kinds. xlsb is
# not one (its sheets are BIFF12 .bin streams, not worksheet XML): the
# name routes the Excel kind, but genuine xlsb content is refused: pinned
# in the engines suite, disclosed in docs/documents.md.
_CT_DOCM_MAIN = (
    '<Override PartName="/word/document.xml" ContentType='
    '"application/vnd.ms-word.document.macroEnabled.main+xml"/>'
)
_CT_XLSM_MAIN = (
    '<Override PartName="/xl/workbook.xml" ContentType='
    '"application/vnd.ms-excel.sheet.macroEnabled.main+xml"/>'
)
_CT_PPSX_MAIN = (
    '<Override PartName="/ppt/presentation.xml" ContentType='
    '"application/vnd.openxmlformats-officedocument.presentationml'
    '.slideshow.macroEnabled.main+xml"/>'
)


def _root_rels(target: str) -> str:
    """The package's root relationships: the one officeDocument part every
    OOXML container anchors at its main document (word/, xl/, ppt/)."""
    return _XML_DECL + (
        f'<Relationships xmlns="{_REL_NS}"><Relationship Id="rId1"'
        f' Type="{_DOC_NS}/officeDocument" Target="{target}"/></Relationships>'
    )


DOCX_CONTENT_TYPES = (
    _XML_DECL + f'<Types xmlns="{_CT_NS}">' + _CT_RELS + _CT_XML + _CT_DOCX_MAIN + "</Types>\n"
)

DOCX_RELS = _root_rels("word/document.xml") + "\n"


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
        + f'<Relationships xmlns="{_REL_NS}">'
        + f'<Relationship Id="rId1" Type="{_DOC_NS}/worksheet" '
        + 'Target="worksheets/sheet1.xml"/>'
        + f'<Relationship Id="rId2" Type="{_DOC_NS}/sharedStrings" '
        + 'Target="sharedStrings.xml"/>'
        + "</Relationships>\n"
    )
    content_types = (
        _XML_DECL
        + f'<Types xmlns="{_CT_NS}">'
        + _CT_RELS
        + _CT_XML
        + _CT_XLSX_MAIN
        + '<Override PartName="/xl/worksheets/sheet1.xml" ContentType='
        + '"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        + '<Override PartName="/xl/sharedStrings.xml" ContentType='
        + '"application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        + "</Types>\n"
    )
    rels = _root_rels("xl/workbook.xml") + "\n"
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


XLSX_TEXT = "\n".join("\t".join(XLSX_SHARED_STRINGS[value] for value in row) for row in XLSX_ROWS)

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
            return bytes([int(token, 8)])  # PDF \ddd literals are octal (\351 = 233)
        return token

    return _PDF_ESCAPE.sub(replace, literal)


def extract_pdf(raw: bytes) -> str:
    """The minimal-PDF reader: every FlateDecode stream is decompressed and
    its Tj literals become lines (one Tj per line is the generator's shape).
    Scoped to that subset: no tj arrays, no font encodings beyond the
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

# The five-document corpus's extraction oracle (each generator's text,
# cross-pinned by extract()). The engines matrix's own kinds join this table
# further down (see the registry section) as content oracles: no stdlib
# reader exists for them, but the same table is where the matrix's
# _oracle_lines looks first.
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
    tests/corpus/ are exactly corpus; the pin test re-derives and
    byte-compares them)."""
    directory.mkdir(parents=True, exist_ok=True)
    for kind, raw in CORPUS.items():
        (directory / _FILE_NAMES[kind]).write_bytes(raw)


# ---------------------------------------------------------------------------
# The engines-suite fixture matrix: the cross-format validation suite's
# corpus (tests/test_documents_engines.py, tools/eval_documents.py). Every
# generatable working-format family × a structural variant rich enough to
# gate on: style-based headings, hyperlinks, numbered/nested lists, tables,
# speaker notes, code, the HTML noise elements that must never leak, the
# ODF spreadsheet/presentation flavors, and the epub container. A separate
# registry from corpus on purpose: the five-document corpus above is pinned
# by tests/test_documents.py; this matrix is pinned by
# tests/test_documents_engines.py, and its committed copies live under
# tests/engines_corpus/. The same determinism contract applies: the
# generators are byte-stable (the pinned-ZipInfo `_zip_bytes`/`_odf_bytes`
# pattern, and the PDF writers below compute xref offsets from their own
# objects), so regeneration is byte-identical.
#
# Shared oracle content (the same lines across every format that can carry
# them, so the to_text gates are content-identical across families). Pure
# ASCII source per the repo convention.

RICH_H1 = "Annual Engineering Report"
RICH_H2 = "Transformer Program"
RICH_LINK_LABEL = "field handbook"
RICH_LINK_URL = "https://handbook.example.com/torque"
RICH_LINK_SENTENCE = f"See the {RICH_LINK_LABEL} for torque tables."
RICH_LIST_FLAT = ("first checkpoint", "second checkpoint")
RICH_LIST_NESTED = "nested detail"
RICH_TABLE_HEADER = ("Unit", "Status")
RICH_TABLE_ROW = ("T-101", "healthy")
RICH_TABLE_ROW2 = ("T-102", "needs review")
RICH_NOTES = "Mention the Q3 outage."
RICH_CODE = "x = 1"
SLIDE1_TITLE = "Program Review"
SLIDE1_BODY = ("budget on track", "contingency held")
SLIDE2_TITLE = "Next Steps"
HTML_SCRIPT_JUNK = 'console.log("tracking junk");'
HTML_STYLE_JUNK = "body { color: red; }"
HTML_TITLE = "Page Title"


# --- rich docx: pStyle headings, external hyperlink, numbered+nested list ---

_DOCX_STYLES = (
    _XML_DECL
    + f"""<w:styles xmlns:w="{_W_NS}">
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>
<w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>
<w:pPr><w:outlineLvl w:val="1"/></w:pPr></w:style>
<w:style w:type="character" w:styleId="Hyperlink"><w:name w:val="Hyperlink"/>
</w:style></w:styles>"""
)

_DOCX_NUMBERING = (
    _XML_DECL
    + f"""<w:numbering xmlns:w="{_W_NS}">
<w:abstractNum w:abstractNumId="0">
<w:lvl w:ilvl="0"><w:numFmt w:val="bullet"/><w:lvlText w:val="%1"/></w:lvl>
<w:lvl w:ilvl="1"><w:numFmt w:val="lowerLetter"/><w:lvlText w:val="%2"/></w:lvl>
</w:abstractNum>
<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num></w:numbering>"""
)


def _docx_rich_paragraph(text: str, *, style: str | None = None, ilvl: int | None = None) -> str:
    """The rich builder's w:p: optional pStyle heading, optional numbering
    level (the base corpus's _docx_paragraph handles tab/br runs instead:
    same name family, different subset, and no shadowing)."""
    props = ""
    if style:
        props += f'<w:pStyle w:val="{style}"/>'
    if ilvl is not None:
        props += f'<w:numPr><w:ilvl w:val="{ilvl}"/><w:numId w:val="1"/></w:numPr>'
    runs = f'<w:r><w:t xml:space="preserve">{_xml_escape(text)}</w:t></w:r>'
    return f"<w:p><w:pPr>{props}</w:pPr>{runs}</w:p>"


def generate_docx_rich() -> bytes:
    """A style-bearing docx: Heading1/Heading2 via pStyle (the way real Word
    documents carry headings), an external hyperlink via document.xml.rels,
    a numbered+nested list via numbering.xml, and a 2x2 table."""
    hyperlink = (
        '<w:hyperlink r:id="rId1"><w:r><w:rPr><w:rStyle w:val="Hyperlink"/></w:rPr>'
        f"<w:t>{RICH_LINK_LABEL}</w:t></w:r></w:hyperlink>"
    )
    link_paragraph = (
        '<w:p><w:r><w:t xml:space="preserve">See the </w:t></w:r>'
        + hyperlink
        + "<w:r><w:t> for torque tables.</w:t></w:r></w:p>"
    )
    table = (
        '<w:tbl><w:tblPr/><w:tblGrid><w:gridCol w:w="2000"/><w:gridCol w:w="2000"/></w:tblGrid>'
        f"<w:tr><w:tc><w:p><w:r><w:t>{RICH_TABLE_HEADER[0]}</w:t></w:r></w:p></w:tc>"
        f"<w:tc><w:p><w:r><w:t>{RICH_TABLE_HEADER[1]}</w:t></w:r></w:p></w:tc></w:tr>"
        f"<w:tr><w:tc><w:p><w:r><w:t>{RICH_TABLE_ROW[0]}</w:t></w:r></w:p></w:tc>"
        f"<w:tc><w:p><w:r><w:t>{RICH_TABLE_ROW[1]}</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
    )
    body = (
        _docx_rich_paragraph(RICH_H1, style="Heading1")
        + _docx_rich_paragraph(RICH_H2, style="Heading2")
        + link_paragraph
        + _docx_rich_paragraph(RICH_LIST_FLAT[0], ilvl=0)
        + _docx_rich_paragraph(RICH_LIST_FLAT[1], ilvl=0)
        + _docx_rich_paragraph(RICH_LIST_NESTED, ilvl=1)
        + table
        + _docx_rich_paragraph("Closing note.")
    )
    document = _XML_DECL + (
        f'<w:document xmlns:w="{_W_NS}" xmlns:r="{_DOC_NS}">'
        f"<w:body>{body}<w:sectPr/></w:body></w:document>"
    )
    rels = _XML_DECL + (
        f'<Relationships xmlns="{_REL_NS}"><Relationship Id="rId1"'
        f' Type="{_DOC_NS}/hyperlink" Target="{RICH_LINK_URL}" TargetMode="External"/>'
        "</Relationships>"
    )
    content_types = _XML_DECL + (
        f'<Types xmlns="{_CT_NS}">' + _CT_RELS + _CT_XML + _CT_DOCX_MAIN + "</Types>"
    )
    root_rels = _root_rels("word/document.xml")
    return _zip_bytes(
        [
            ("[Content_Types].xml", content_types.encode("utf-8")),
            ("_rels/.rels", root_rels.encode("utf-8")),
            ("word/document.xml", document.encode("utf-8")),
            ("word/styles.xml", _DOCX_STYLES.encode("utf-8")),
            ("word/numbering.xml", _DOCX_NUMBERING.encode("utf-8")),
            ("word/_rels/document.xml.rels", rels.encode("utf-8")),
        ]
    )


# --- rich pptx: two slides (title + leveled body) + speaker notes ---------

_P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _pptx_shape(placeholder: str, idx: str, paragraphs: list[tuple[int, str]]) -> str:
    body = "".join(
        f'<a:p><a:pPr lvl="{lvl}"/><a:r><a:t>{text}</a:t></a:r></a:p>' for lvl, text in paragraphs
    )
    return (
        '<p:sp><p:nvSpPr><p:cNvPr id="1" name="placeholder"/>'
        f'<p:cNvSpPr/><p:nvPr><p:ph type="{placeholder}"{idx}/></p:nvPr></p:nvSpPr>'
        f"<p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>{body}</p:txBody></p:sp>"
    )


def _pptx_part(root: str, shapes: str) -> str:
    """One presentationml part: the cSld/spTree skeleton slides and notes
    slides share, around the shapes (the root element name is the only
    difference between a slide and a notes slide)."""
    return _XML_DECL + (
        f'<p:{root} xmlns:p="{_P_NS}" xmlns:a="{_A_NS}"><p:cSld><p:spTree><p:nvGrpSpPr/>'
        f"<p:grpSpPr/>{shapes}</p:spTree></p:cSld></p:{root}>"
    )


def generate_pptx_rich() -> bytes:
    slide1 = _pptx_part(
        "sld",
        _pptx_shape("title", "", [(0, SLIDE1_TITLE)])
        + _pptx_shape("body", ' idx="1"', [(0, SLIDE1_BODY[0]), (1, SLIDE1_BODY[1])]),
    )
    slide2 = _pptx_part("sld", _pptx_shape("title", "", [(0, SLIDE2_TITLE)]))
    notes1 = _pptx_part("notes", _pptx_shape("body", ' idx="1"', [(0, RICH_NOTES)]))
    presentation = _XML_DECL + (
        f'<p:presentation xmlns:p="{_P_NS}" xmlns:r="{_DOC_NS}">'
        '<p:sldIdLst><p:sldId id="256" r:id="rId1"/><p:sldId id="257" r:id="rId2"/></p:sldIdLst>'
        "</p:presentation>"
    )
    pres_rels = _XML_DECL + (
        f'<Relationships xmlns="{_REL_NS}">'
        f'<Relationship Id="rId1" Type="{_DOC_NS}/slide" Target="slides/slide1.xml"/>'
        f'<Relationship Id="rId2" Type="{_DOC_NS}/slide" Target="slides/slide2.xml"/>'
        "</Relationships>"
    )
    slide1_rels = _XML_DECL + (
        f'<Relationships xmlns="{_REL_NS}">'
        f'<Relationship Id="rId3" Type="{_DOC_NS}/notesSlide"'
        ' Target="../notesSlides/notesSlide1.xml"/>'
        "</Relationships>"
    )
    content_types = _XML_DECL + (
        f'<Types xmlns="{_CT_NS}">'
        + _CT_RELS
        + _CT_XML
        + _CT_PPTX_MAIN
        + _CT_SLIDE.format(n=1)
        + _CT_SLIDE.format(n=2)
        + _CT_NOTES.format(n=1)
        + "</Types>"
    )
    root_rels = _root_rels("ppt/presentation.xml")
    return _zip_bytes(
        [
            ("[Content_Types].xml", content_types.encode("utf-8")),
            ("_rels/.rels", root_rels.encode("utf-8")),
            ("ppt/presentation.xml", presentation.encode("utf-8")),
            ("ppt/_rels/presentation.xml.rels", pres_rels.encode("utf-8")),
            ("ppt/slides/slide1.xml", slide1.encode("utf-8")),
            ("ppt/slides/slide2.xml", slide2.encode("utf-8")),
            ("ppt/slides/_rels/slide1.xml.rels", slide1_rels.encode("utf-8")),
            ("ppt/notesSlides/notesSlide1.xml", notes1.encode("utf-8")),
        ]
    )


# --- the OOXML alias containers (docm / xlsm / ppsx) -----------------------------
#
# The genuine aliases as [Content_Types].xml rewrites of the base generators
# above: the macro-enabled (docm/xlsm) and slide-show (ppsx) variants are
# the same packages with the one main-document override respelled: the
# only difference between the containers. The rewrite is byte-stable (a
# no-op rewrite round-trips the base bytes exactly, probed), so the
# committed fixtures hold the same determinism contract as their bases.


def _ooxml_alias_container(raw: bytes, base_ct: str, alias_ct: str) -> bytes:
    """The base generator's package with [Content_Types].xml's main
    override respelled to the variant's content type, rebuilt through
    `_zip_bytes` so the pinned-ZipInfo determinism contract holds."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = [
            (
                item.filename,
                archive.read(item.filename).replace(base_ct.encode(), alias_ct.encode()),
            )
            for item in archive.infolist()
        ]
    return _zip_bytes(entries)


def generate_docm_rich() -> bytes:
    """generate_docx_rich's package as a docm: the macro-enabled alias the
    engines convert as docx on both lanes (anydoc and office_oxide)."""
    return _ooxml_alias_container(generate_docx_rich(), _CT_DOCX_MAIN, _CT_DOCM_MAIN)


def generate_xlsm() -> bytes:
    """generate_xlsx's package as an xlsm: the macro-enabled alias the
    engines convert as xlsx on both lanes (anydoc and office_oxide)."""
    return _ooxml_alias_container(generate_xlsx(), _CT_XLSX_MAIN, _CT_XLSM_MAIN)


def generate_ppsx_rich() -> bytes:
    """generate_pptx_rich's package as a ppsx: the slide-show alias the
    auto/anydoc lane converts as pptx (office_oxide refuses the slideshow
    content type: an honest clean-refusal divergence the engines suite
    pins, never a silent fallback)."""
    return _ooxml_alias_container(generate_pptx_rich(), _CT_PPTX_MAIN, _CT_PPSX_MAIN)


# --- the ODF family (odt / ods / odp) + epub, the OCF sibling ------------------
#
# One container contract (the mimetype entry stored and first: `_odf_bytes`)
# and one content-part skeleton (`_odf_content`); the flavors differ only in
# the body element and the namespaces they need. epub rides the same
# container contract: OCF is ODF's zip sibling, mimetype stored first there
# too.

_ODT_OFFICE = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
_ODT_TEXT = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_ODT_TABLE = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_ODT_DRAW = "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"


def _odf_bytes(mimetype: str, entries: list[tuple[str, bytes]]) -> bytes:
    """A byte-deterministic ODF/OCF container: the mimetype entry stored and
    first (the ODF spec requirement, and OCF's for epub: same rule), every
    entry on a pinned ZipInfo (the `_zip_bytes` determinism contract)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        info = zipfile.ZipInfo("mimetype", date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o644 << 16
        info.compress_type = zipfile.ZIP_STORED
        archive.writestr(info, mimetype)
        for name, data in entries:
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    return buffer.getvalue()


def _odt_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    """The ODT container: the ODF text flavor of `_odf_bytes` (kept as its
    own name: tests/docgen.py's odt generator builds on it)."""
    return _odf_bytes("application/vnd.oasis.opendocument.text", entries)


def _odf_content(namespaces: str, body: str) -> str:
    """The ODF content-part skeleton: one office:document-content carrying
    the format's namespaces, around the office:body payload (text,
    spreadsheet, and presentation flavors differ only in the body element
    and the namespaces they need)."""
    return _XML_DECL + (
        f"<office:document-content {namespaces}>"
        f"<office:body>{body}</office:body></office:document-content>"
    )


def generate_odt_rich() -> bytes:
    """A style-bearing odt: outline-level headings, a hyperlink, a nested
    list, and a table: the ODF text flavor of the shared rich vocabulary."""
    body = (
        "<office:text>"
        f'<text:h text:outline-level="1">{RICH_H1}</text:h>'
        f'<text:h text:outline-level="2">{RICH_H2}</text:h>'
        "<text:p>Draw oil samples quarterly.</text:p>"
        f'<text:p>See the <text:a xlink:href="{RICH_LINK_URL}">{RICH_LINK_LABEL}</text:a>'
        " for torque tables.</text:p>"
        f"<text:list><text:list-item><text:p>{RICH_LIST_FLAT[0]}</text:p></text:list-item>"
        f"<text:list-item><text:p>{RICH_LIST_FLAT[1]}</text:p>"
        f"<text:list><text:list-item><text:p>{RICH_LIST_NESTED}</text:p></text:list-item></text:list>"
        "</text:list-item></text:list>"
        '<table:table table:name="units"><table:table-row>'
        f"<table:table-cell><text:p>{RICH_TABLE_HEADER[0]}</text:p></table:table-cell>"
        f"<table:table-cell><text:p>{RICH_TABLE_HEADER[1]}</text:p></table:table-cell></table:table-row>"
        f"<table:table-row><table:table-cell><text:p>{RICH_TABLE_ROW[0]}</text:p></table:table-cell>"
        f"<table:table-cell><text:p>{RICH_TABLE_ROW[1]}</text:p></table:table-cell></table:table-row>"
        "</table:table></office:text>"
    )
    content = _odf_content(
        f'xmlns:office="{_ODT_OFFICE}" xmlns:text="{_ODT_TEXT}"'
        f' xmlns:table="{_ODT_TABLE}" xmlns:xlink="http://www.w3.org/1999/xlink"',
        body,
    )
    return _odf_bytes(
        "application/vnd.oasis.opendocument.text", [("content.xml", content.encode("utf-8"))]
    )


# --- rich ods: the units table, spreadsheet flavor ----------------------------

ODS_TABLE = (RICH_TABLE_HEADER, RICH_TABLE_ROW, RICH_TABLE_ROW2)


def _ods_cell(text: str) -> str:
    """One spreadsheet cell: its value as text:p, the string-value spelling
    real ODS writers emit (office:value-type="string")."""
    return (
        f'<table:table-cell office:value-type="string"><text:p>{text}</text:p></table:table-cell>'
    )


def generate_ods_rich() -> bytes:
    """The ODF spreadsheet: the units table (header + two data rows): the
    csv fixture's story in spreadsheet flavor, sniffed by the ZIP mimetype
    entry and rendered as a GFM table by the engine."""
    rows = "".join(
        f"<table:table-row>{_ods_cell(row[0])}{_ods_cell(row[1])}</table:table-row>"
        for row in ODS_TABLE
    )
    content = _odf_content(
        f'xmlns:office="{_ODT_OFFICE}" xmlns:text="{_ODT_TEXT}" xmlns:table="{_ODT_TABLE}"',
        "<office:spreadsheet>"
        f'<table:table table:name="units">{rows}</table:table>'
        "</office:spreadsheet>",
    )
    return _odf_bytes(
        "application/vnd.oasis.opendocument.spreadsheet",
        [("content.xml", content.encode("utf-8"))],
    )


# --- rich odp: two pages of title + body text boxes ----------------------------


def _odp_text_box(paragraphs: tuple[str, ...]) -> str:
    """One presentation frame: a text box carrying its paragraphs (the
    title/bullet shapes; ODP carries styling out-of-line, so a text-only
    box is the minimal real shape)."""
    body = "".join(f"<text:p>{text}</text:p>" for text in paragraphs)
    return f"<draw:frame><draw:text-box>{body}</draw:text-box></draw:frame>"


def generate_odp_rich() -> bytes:
    """The ODF presentation: two draw:page elements (a title box plus a
    two-paragraph body box, then a second title) the pptx fixture's story
    in the ODF presentation flavor."""
    content = _odf_content(
        f'xmlns:office="{_ODT_OFFICE}" xmlns:text="{_ODT_TEXT}" xmlns:draw="{_ODT_DRAW}"',
        "<office:presentation>"
        '<draw:page draw:name="slide1">'
        + _odp_text_box((SLIDE1_TITLE,))
        + _odp_text_box(SLIDE1_BODY)
        + '</draw:page><draw:page draw:name="slide2">'
        + _odp_text_box((SLIDE2_TITLE,))
        + "</draw:page></office:presentation>",
    )
    return _odf_bytes(
        "application/vnd.oasis.opendocument.presentation",
        [("content.xml", content.encode("utf-8"))],
    )


# --- rich epub: one-chapter OCF container ---------------------------------------

_EPUB_CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
_EPUB_OPF_NS = "http://www.idpf.org/2007/opf"
_EPUB_DC_NS = "http://purl.org/dc/elements/1.1/"
_EPUB_XHTML_NS = "http://www.w3.org/1999/xhtml"
_EPUB_BOOK_ID = "urn:uuid:00000000-0000-0000-0000-000000000001"


def generate_epub_rich() -> bytes:
    """An OCF container: the mimetype entry stored first (OCF's rule, the
    same one as ODF's), META-INF/container.xml naming the OPF, the OPF
    manifesting one chapter, and the chapter carrying the shared rich
    vocabulary (headings, hyperlink, nested list) as xhtml."""
    container = _XML_DECL + (
        f'<container version="1.0" xmlns="{_EPUB_CONTAINER_NS}">'
        '<rootfiles><rootfile full-path="OEBPS/content.opf"'
        ' media-type="application/oebps-package+xml"/></rootfiles></container>'
    )
    opf = _XML_DECL + (
        f'<package xmlns="{_EPUB_OPF_NS}" version="2.0" unique-identifier="bookid">'
        f'<metadata xmlns:dc="{_EPUB_DC_NS}"><dc:title>{RICH_H1}</dc:title>'
        f'<dc:identifier id="bookid">{_EPUB_BOOK_ID}</dc:identifier>'
        "<dc:language>en</dc:language></metadata>"
        '<manifest><item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
        '</manifest><spine><itemref idref="ch1"/></spine></package>'
    )
    chapter = _XML_DECL + (
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN"'
        ' "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">\n'
        f'<html xmlns="{_EPUB_XHTML_NS}"><head><title>{RICH_H1}</title></head><body>'
        f"<h1>{RICH_H1}</h1><h2>{RICH_H2}</h2>"
        f'<p>See the <a href="{RICH_LINK_URL}">{RICH_LINK_LABEL}</a> for torque tables.</p>'
        f"<ul><li>{RICH_LIST_FLAT[0]}</li><li>{RICH_LIST_FLAT[1]}"
        f"<ul><li>{RICH_LIST_NESTED}</li></ul></li></ul>"
        "</body></html>\n"
    )
    return _odf_bytes(
        "application/epub+zip",
        [
            ("META-INF/container.xml", container.encode("utf-8")),
            ("OEBPS/content.opf", opf.encode("utf-8")),
            ("OEBPS/ch1.xhtml", chapter.encode("utf-8")),
        ],
    )


# --- csv: header + rows ------------------------------------------------------

CSV_ROWS = (
    "unit,status",
    "T-101,healthy",
    "T-102,needs review",
)


def generate_csv_rich() -> bytes:
    return ("\n".join(CSV_ROWS) + "\n").encode("ascii")


# --- html: polluted head + full body structure ------------------------------

HTML_DOCUMENT = (
    "<!DOCTYPE html>\n"
    f"<html><head><title>{HTML_TITLE}</title><style>{HTML_STYLE_JUNK}</style>\n"
    f"<script>{HTML_SCRIPT_JUNK}</script></head>\n"
    "<body>\n"
    f"<h1>{RICH_H1}</h1>\n"
    f"<h2>{RICH_H2}</h2>\n"
    f'<p>See the <a href="{RICH_LINK_URL}">{RICH_LINK_LABEL}</a> for torque tables.</p>\n'
    f"<ul><li>{RICH_LIST_FLAT[0]}</li><li>{RICH_LIST_FLAT[1]}"
    f"<ul><li>{RICH_LIST_NESTED}</li></ul></li></ul>\n"
    f"<table><thead><tr><th>{RICH_TABLE_HEADER[0]}</th>"
    f"<th>{RICH_TABLE_HEADER[1]}</th></tr></thead>\n"
    f"<tbody><tr><td>{RICH_TABLE_ROW[0]}</td><td>{RICH_TABLE_ROW[1]}</td>"
    "</tr></tbody></table>\n"
    f"<blockquote><p>{RICH_NOTES}</p></blockquote>\n"
    f"<pre><code>{RICH_CODE}</code></pre>\n"
    "</body></html>\n"
)


def generate_html_rich() -> bytes:
    return HTML_DOCUMENT.encode("utf-8")


# --- the document-structure fixtures (issues #209/#210/#211) -----------
#
# The structure surface reads the RAW layers (office_oxide's
# DocxDocument/XlsxDocument, the html-to-markdown-rs structure collector),
# so these fixtures carry what the raw readers actually need and the
# conversion lane never looks at: the styles/numbering relationships in
# document.xml.rels (office_oxide resolves the stylesheet through the
# document part's OWN relationships — the conversion lane's fixture above
# omits them and is correct for it), the note and header parts, the media
# part an image rid points at, the chartsheet relationship, and the
# declared <dimension> that lies. Byte-deterministic through `_zip_bytes`
# like every generator here; consumed by the structure suite
# (tests/test_documents_engines.py's TestDocumentStructure) via `data=`,
# never by the pinned engines corpus (whose committed copies these would
# otherwise have to re-derive).

STRUCTURE_H1 = "Structure Report"
STRUCTURE_H2 = "Imaging Program"
STRUCTURE_H3_NAME = "Imaging Detail"
STRUCTURE_LINK_URL = "https://handbook.example.com/structure"
STRUCTURE_ANCHOR = "imaging-anchor"
STRUCTURE_IMAGE_URL = "https://example.com/units-chart.png"
STRUCTURE_IMAGE_ALT = "units chart"


def _docx_structure_drawing(rid: str | None, *, cx: int = 1_905_000, cy: int = 1_270_000,
                            alt: str | None = None, decorative: bool = False,
                            shape: bool = False) -> str:
    """One `<w:drawing>`: a picture (`rid`, the declared EMU extent, alt
    text, the decorative flag) or a vector shape (no rid — the
    shape-filtered case)."""
    wp = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
    a = "http://schemas.openxmlformats.org/drawingml/2006/main"
    pic = "http://schemas.openxmlformats.org/drawingml/2006/picture"
    descr = f' descr="{_xml_escape(alt)}"' if alt else ""
    extent = f'<wp:extent cx="{cx}" cy="{cy}"/>' if cx or cy else ""
    if decorative:
        doc_pr = (
            f'<wp:docPr{descr}><adec:decorative xmlns:adec="{a}" val="1"/></wp:docPr>'
        )
    else:
        doc_pr = f'<wp:docPr{descr}/>'
    if shape:
        graphic = (
            f'<a:graphic xmlns:a="{a}"><a:graphicData>'
            '<wps:wsp xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">'
            '<wps:spPr><a:prstGeom prst="rect"/></wps:spPr></wps:wsp>'
            "</a:graphicData></a:graphic>"
        )
    else:
        embed = f'<a:blip r:embed="{rid}"/>' if rid else "<a:blip/>"
        graphic = (
            f'<a:graphic xmlns:a="{a}"><a:graphicData>'
            f'<pic:pic xmlns:pic="{pic}"><pic:blipFill>{embed}</pic:blipFill></pic:pic>'
            "</a:graphicData></a:graphic>"
        )
    return (
        f'<w:drawing><wp:inline xmlns:wp="{wp}">'
        f"{extent}{doc_pr}{graphic}</wp:inline></w:drawing>"
    )


def generate_docx_structure() -> bytes:
    """The structure fixture's docx: every raw-layer pin in one body —
    a style-outline heading and a style-NAME-convention heading (the
    parity pair), a direct outlineLvl 7 (the clamp case), a
    style-carried numPr list, a numbered heading, the block-text walk
    (tab/br/hidden run), an external and an internal hyperlink, the
    picture pair (declared extent + 0-extent) plus a shape-only drawing
    and a decorative image, a gridSpan + vMerge table with a caption, two
    in-paragraph sectPr (three sections), a footnote carrying a
    hyperlink, and an unreadable header part."""
    w = _W_NS
    r = _DOC_NS
    document = _XML_DECL + f"""<w:document xmlns:w="{w}" xmlns:r="{r}"><w:body>""" + "".join(
        [
            # the agreeing parity case: pStyle -> stylesheet outlineLvl
            f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
            f"<w:r><w:t>{RICH_H1}</w:t></w:r></w:p>",
            # the divergent parity case: only the style NAME says heading
            # (the structure's convention rung; the oxide markdown's
            # two-rung ladder renders plain text)
            f'<w:p><w:pPr><w:pStyle w:val="FancyH3"/></w:pPr>'
            f"<w:r><w:t>{STRUCTURE_H3_NAME}</w:t></w:r></w:p>",
            # the clamp case: direct outlineLvl 7 -> level 6, raw 7
            '<w:p><w:pPr><w:outlineLvl w:val="7"/></w:pPr>'
            "<w:r><w:t>Deep heading</w:t></w:r></w:p>",
            # the style-carried numPr list ("List Paragraph" holds it in
            # the STYLE) plus a direct numPr item
            '<w:p><w:pPr><w:pStyle w:val="ListPx"/></w:pPr>'
            "<w:r><w:t>styled bullet</w:t></w:r></w:p>",
            '<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/>'
            "</w:numPr></w:pPr><w:r><w:t>direct numbered item</w:t></w:r></w:p>",
            # a numbered heading: kind heading, the numPr reported
            '<w:p><w:pPr><w:pStyle w:val="Heading2"/>'
            '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr></w:pPr>'
            f"<w:r><w:t>{STRUCTURE_H2}</w:t></w:r></w:p>",
            # the block-text walk: br, tab, hidden run
            "<w:p><w:r><w:t>line one</w:t></w:r><w:r><w:br/></w:r><w:r><w:tab/></w:r>"
            '<w:r><w:t>after tab</w:t></w:r>'
            '<w:r><w:rPr><w:vanish/></w:rPr><w:t>secretly hidden</w:t></w:r></w:p>',
            # external + internal hyperlinks
            '<w:p><w:hyperlink r:id="rId7"><w:r><w:t>the external link</w:t></w:r>'
            "</w:hyperlink></w:p>",
            f'<w:p><w:hyperlink w:anchor="{STRUCTURE_ANCHOR}">'
            "<w:r><w:t>the internal link</w:t></w:r></w:hyperlink></w:p>",
            # pictures: declared extent + alt; 0-extent; shape-only
            # (filtered); decorative
            f'<w:p><w:r>{_docx_structure_drawing("rId9", alt=STRUCTURE_IMAGE_ALT)}</w:r></w:p>',
            f'<w:p><w:r>{_docx_structure_drawing("rId9", cx=0, cy=0)}</w:r></w:p>',
            f'<w:p><w:r>{_docx_structure_drawing(None, shape=True)}</w:r></w:p>',
            f'<w:p><w:r>{_docx_structure_drawing("rId9", alt="deco", decorative=True)}</w:r></w:p>',
            # the table: gridSpan, a vMerge Restart/Continue pair, a caption
            '<w:tbl><w:tblPr><w:tblCaption w:val="Units table"/></w:tblPr>'
            '<w:tblGrid><w:gridCol w:w="1800"/><w:gridCol w:w="1800"/>'
            '<w:gridCol w:w="1800"/></w:tblGrid>'
            '<w:tr><w:tc><w:tcPr><w:gridSpan w:val="2"/></w:tcPr>'
            f"<w:p><w:r><w:t>{RICH_TABLE_HEADER[0]}</w:t></w:r></w:p></w:tc>"
            f"<w:tc><w:p><w:r><w:t>{RICH_TABLE_HEADER[1]}</w:t></w:r></w:p></w:tc></w:tr>"
            '<w:tr><w:tc><w:tcPr><w:vMerge w:val="restart"/></w:tcPr>'
            "<w:p><w:r><w:t>merged col</w:t></w:r></w:p></w:tc>"
            f"<w:tc><w:p><w:r><w:t>{RICH_TABLE_ROW[0]}</w:t></w:r></w:p></w:tc>"
            f"<w:tc><w:p><w:r><w:t>{RICH_TABLE_ROW[1]}</w:t></w:r></w:p></w:tc></w:tr>"
            '<w:tr><w:tc><w:tcPr><w:vMerge/></w:tcPr><w:p/></w:tc>'
            f"<w:tc><w:p><w:r><w:t>{RICH_TABLE_ROW2[0]}</w:t></w:r></w:p></w:tc>"
            f"<w:tc><w:p><w:r><w:t>{RICH_TABLE_ROW2[1]}</w:t></w:r></w:p>"
            "</w:tc></w:tr></w:tbl>",
            # section one ends (the sectPr paragraph terminates it)
            '<w:p><w:pPr><w:sectPr>'
            '<w:headerReference w:type="default" r:id="rId6"/>'
            '<w:pgSz w:w="12240" w:h="15840"/></w:sectPr></w:pPr>'
            "<w:r><w:t>end of section one</w:t></w:r></w:p>",
            # section two ends; the bookmark the internal link names
            '<w:p><w:pPr><w:sectPr><w:pgSz w:w="12240" w:h="15840"/></w:sectPr></w:pPr>'
            f'<w:bookmarkStart w:id="1" w:name="{STRUCTURE_ANCHOR}"/>'
            '<w:bookmarkEnd w:id="1"/>'
            "<w:r><w:t>end of section two</w:t></w:r></w:p>",
            # the footnote citation
            "<w:p><w:r><w:t>the cited claim</w:t></w:r>"
            '<w:r><w:footnoteReference w:id="1"/></w:r></w:p>',
            "</w:body></w:document>",
        ]
    )
    styles = _XML_DECL + f"""<w:styles xmlns:w="{w}">
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>
<w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>
<w:pPr><w:outlineLvl w:val="1"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="FancyH3"><w:name w:val="heading 3"/></w:style>
<w:style w:type="paragraph" w:styleId="ListPx"><w:name w:val="List Paragraph"/>
<w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr></w:style>
<w:style w:type="character" w:styleId="Hyperlink"><w:name w:val="Hyperlink"/></w:style>
</w:styles>"""
    numbering = _XML_DECL + f"""<w:numbering xmlns:w="{w}">
<w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0">
<w:numFmt w:val="bullet"/><w:start w:val="1"/></w:lvl></w:abstractNum>
<w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0">
<w:numFmt w:val="decimal"/><w:start w:val="4"/></w:lvl></w:abstractNum>
<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
<w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num>
</w:numbering>"""
    rels = _XML_DECL + f"""<Relationships xmlns="{_REL_NS}">
<Relationship Id="rId7" Type="{r}/hyperlink" Target="{STRUCTURE_LINK_URL}"
 TargetMode="External"/>
<Relationship Id="rId9" Type="{r}/image" Target="media/image9.png"/>
<Relationship Id="rId6" Type="{r}/header" Target="header1.xml"/>
<Relationship Id="rId3" Type="{r}/styles" Target="styles.xml"/>
<Relationship Id="rId4" Type="{r}/numbering" Target="numbering.xml"/>
<Relationship Id="rId5" Type="{r}/footnotes" Target="footnotes.xml"/>
</Relationships>"""
    footnotes = _XML_DECL + f"""<w:footnotes xmlns:w="{w}" xmlns:r="{r}">
<w:footnote w:id="1"><w:p><w:r><w:t>the footnote text</w:t></w:r>
<w:hyperlink r:id="rId2"><w:r><w:t>footnote link</w:t></w:r></w:hyperlink></w:p>
</w:footnote></w:footnotes>"""
    footnote_rels = _XML_DECL + f"""<Relationships xmlns="{_REL_NS}">
<Relationship Id="rId2" Type="{r}/hyperlink" Target="{STRUCTURE_LINK_URL}"
 TargetMode="External"/>
</Relationships>"""
    content_types = (
        _XML_DECL
        + f"""<Types xmlns="{_CT_NS}">"""
        + _CT_RELS
        + _CT_XML
        + _CT_DOCX_MAIN
        + '<Override PartName="/word/footnotes.xml" ContentType='
        + '"application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/>'
        + "</Types>"
    )
    return _zip_bytes(
        [
            ("[Content_Types].xml", content_types.encode("utf-8")),
            ("_rels/.rels", _root_rels("word/document.xml").encode("utf-8")),
            ("word/document.xml", document.encode("utf-8")),
            ("word/styles.xml", styles.encode("utf-8")),
            ("word/numbering.xml", numbering.encode("utf-8")),
            ("word/_rels/document.xml.rels", rels.encode("utf-8")),
            ("word/footnotes.xml", footnotes.encode("utf-8")),
            ("word/_rels/footnotes.xml.rels", footnote_rels.encode("utf-8")),
            ("word/media/image9.png", b"png-bytes"),
            ("word/header1.xml", b"<w:hdr"),
        ]
    )


def generate_xlsx_structure() -> bytes:
    """The structure fixture's xlsx: three worksheets (visible, hidden,
    veryHidden) + one chartsheet, typed cells (number, boolean, error,
    t="d" date, styled-empty), a formula with and one without a cached
    value, a lying declared dimension, a merged range, an external and an
    internal sheet hyperlink, and a global + a sheet-scoped defined
    name."""
    main = _MAIN_NS
    r = _DOC_NS
    sheet = _XML_DECL + f"""<worksheet xmlns="{main}" xmlns:r="{r}">
<dimension ref="A1:C4"/><sheetData>
<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>3.5</v></c>
<c r="C1" t="b"><v>1</v></c><c r="D1" s="1"/></row>
<row r="2"><c r="A2"><f>SUM(B1:B1)</f><v>3.5</v></c>
<c r="B2" t="e"><v>#REF!</v></c><c r="C2" t="d"><v>2026-01-02</v></c></row>
<row r="4"><c r="D4" t="s"><v>1</v></c></row>
<row r="5"><c r="A5"><f>1+1</f></c></row>
</sheetData><mergeCells count="1"><mergeCell ref="A3:B3"/></mergeCells>
<hyperlinks><hyperlink ref="B1" r:id="rId9" tooltip="the guide"/>
<hyperlink ref="C1" location="Hidden!A1"/></hyperlinks></worksheet>"""
    empty_sheet = _XML_DECL + f'<worksheet xmlns="{main}"/>'
    shared = _XML_DECL + f"""<sst xmlns="{main}" count="2" uniqueCount="2">
<si><t>unit label</t></si><si><t>second label</t></si></sst>"""
    workbook = _XML_DECL + f"""<workbook xmlns="{main}" xmlns:r="{r}"><sheets>
<sheet name="Visible" sheetId="1" r:id="rId1"/>
<sheet name="Hidden" sheetId="2" state="hidden" r:id="rId2"/>
<sheet name="VeryHidden" sheetId="3" state="veryHidden" r:id="rId3"/>
<sheet name="ChartTab" sheetId="4" r:id="rId4"/>
</sheets><definedNames>
<definedName name="GLOB">Visible!$A$1</definedName>
<definedName name="SCOPED" localSheetId="0">Visible!$B$1</definedName>
</definedNames></workbook>"""
    workbook_rels = _XML_DECL + f"""<Relationships xmlns="{_REL_NS}">
<Relationship Id="rId1" Type="{r}/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="{r}/worksheet" Target="worksheets/sheet2.xml"/>
<Relationship Id="rId3" Type="{r}/worksheet" Target="worksheets/sheet3.xml"/>
<Relationship Id="rId4" Type="{r}/chartsheet" Target="chartsheets/chart1.xml"/>
<Relationship Id="rId5" Type="{r}/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>"""
    sheet_rels = _XML_DECL + f"""<Relationships xmlns="{_REL_NS}">
<Relationship Id="rId9" Type="{r}/hyperlink" Target="{STRUCTURE_LINK_URL}"
 TargetMode="External"/>
</Relationships>"""
    content_types = (
        _XML_DECL
        + f"""<Types xmlns="{_CT_NS}">"""
        + _CT_RELS
        + _CT_XML
        + _CT_XLSX_MAIN
        + "</Types>"
    )
    return _zip_bytes(
        [
            ("[Content_Types].xml", content_types.encode("utf-8")),
            ("_rels/.rels", _root_rels("xl/workbook.xml").encode("utf-8")),
            ("xl/workbook.xml", workbook.encode("utf-8")),
            ("xl/_rels/workbook.xml.rels", workbook_rels.encode("utf-8")),
            ("xl/worksheets/sheet1.xml", sheet.encode("utf-8")),
            ("xl/worksheets/sheet2.xml", empty_sheet.encode("utf-8")),
            ("xl/worksheets/sheet3.xml", empty_sheet.encode("utf-8")),
            ("xl/sharedStrings.xml", shared.encode("utf-8")),
            ("xl/worksheets/_rels/sheet1.xml.rels", sheet_rels.encode("utf-8")),
        ]
    )


HTML_STRUCTURE_DOCUMENT = (
    "<!DOCTYPE html>\n"
    "<html><head><title>Structure Page</title><style>body { color: junk; }</style>"
    '<script>console.log("structure junk");</script></head>\n'
    "<body>\n"
    f'<h1 id="top">{STRUCTURE_H1}</h1>\n'
    f'<h2 id="sub">{STRUCTURE_H2}</h2>\n'
    f'<p>See the <a href="{STRUCTURE_LINK_URL}" title="the guide">field handbook</a>.</p>\n'
    f'<img src="{STRUCTURE_IMAGE_URL}" alt="{STRUCTURE_IMAGE_ALT}">\n'
    '<img src="https://example.com/deco.png">\n'
    "<table><tr><td>plain</td><td>no</td></tr>"
    "<tr><td>header</td><td>cells</td></tr></table>\n"
    "<table><tr><th>H</th></tr>"
    "<tr><td><table><tr><td>inner cell</td></tr></table></td></tr></table>\n"
    "</body></html>\n"
)


def generate_html_structure() -> bytes:
    """The structure fixture's html: anchor-heavy headings (id
    attributes), a titled link, images with and without alt (no
    dimensions — the collector drops img attributes), a `<th>`-less
    table, and a nested table (the collector flattens the inner one into
    the cell's text)."""
    return HTML_STRUCTURE_DOCUMENT.encode("utf-8")


HTML_FRAGMENT_DOCUMENT = "<div><p>just a fragment</p><ul><li>fragment item</li></ul></div>"


def generate_html_fragment() -> bytes:
    """A fragment (no doctype, no `<html>`): deliberately NOT
    content-resolvable (the pinned `looks_like_html` policy) — served
    only through `format="html"` or an .html-family extension."""
    return HTML_FRAGMENT_DOCUMENT.encode("utf-8")


def generate_xlsx_damaged() -> bytes:
    """A workbook whose second sheet's part is binary garbage: the intact
    sheet is served and the loss is on record (``unreadable_sheets``) —
    never a whole-file refusal, never a silent skip."""
    main = _MAIN_NS
    r = _DOC_NS
    workbook = _XML_DECL + (
        f'<workbook xmlns="{main}" xmlns:r="{r}"><sheets>'
        '<sheet name="Intact" sheetId="1" r:id="rId1"/>'
        '<sheet name="Damaged" sheetId="2" r:id="rId2"/></sheets></workbook>'
    )
    workbook_rels = _XML_DECL + (
        f'<Relationships xmlns="{_REL_NS}">'
        f'<Relationship Id="rId1" Type="{r}/worksheet" Target="worksheets/sheet1.xml"/>'
        f'<Relationship Id="rId2" Type="{r}/worksheet" Target="worksheets/sheet2.xml"/>'
        "</Relationships>"
    )
    sheet = _XML_DECL + (
        f'<worksheet xmlns="{main}"><sheetData>'
        '<row r="1"><c r="A1" t="s"><v>0</v></c></row></sheetData></worksheet>'
    )
    shared = _XML_DECL + (
        f'<sst xmlns="{main}" count="1" uniqueCount="1">'
        "<si><t>intact label</t></si></sst>"
    )
    content_types = _XML_DECL + (
        f'<Types xmlns="{_CT_NS}">' + _CT_RELS + _CT_XML + _CT_XLSX_MAIN + "</Types>"
    )
    return _zip_bytes(
        [
            ("[Content_Types].xml", content_types.encode("utf-8")),
            ("_rels/.rels", _root_rels("xl/workbook.xml").encode("utf-8")),
            ("xl/workbook.xml", workbook.encode("utf-8")),
            ("xl/_rels/workbook.xml.rels", workbook_rels.encode("utf-8")),
            ("xl/worksheets/sheet1.xml", sheet.encode("utf-8")),
            ("xl/worksheets/sheet2.xml", b"<worksheet"),  # EOF mid-tag: the fast reader errors
            ("xl/sharedStrings.xml", shared.encode("utf-8")),
        ]
    )


def generate_xlsx_cells(count: int) -> bytes:
    """One worksheet with `count` numeric cells in column A: the entry
    cap's boundary fixture (199_999 / 200_000 / 200_001)."""
    main = _MAIN_NS
    r = _DOC_NS
    rows = "".join(
        f'<row r="{row}"><c r="A{row}"><v>{row}</v></c></row>'
        for row in range(1, count + 1)
    )
    sheet = _XML_DECL + f'<worksheet xmlns="{main}"><sheetData>{rows}</sheetData></worksheet>'
    workbook = _XML_DECL + (
        f'<workbook xmlns="{main}" xmlns:r="{r}"><sheets>'
        '<sheet name="Big" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    workbook_rels = _XML_DECL + (
        f'<Relationships xmlns="{_REL_NS}">'
        f'<Relationship Id="rId1" Type="{r}/worksheet" Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    content_types = _XML_DECL + (
        f'<Types xmlns="{_CT_NS}">' + _CT_RELS + _CT_XML + _CT_XLSX_MAIN + "</Types>"
    )
    return _zip_bytes(
        [
            ("[Content_Types].xml", content_types.encode("utf-8")),
            ("_rels/.rels", _root_rels("xl/workbook.xml").encode("utf-8")),
            ("xl/workbook.xml", workbook.encode("utf-8")),
            ("xl/_rels/workbook.xml.rels", workbook_rels.encode("utf-8")),
            ("xl/worksheets/sheet1.xml", sheet.encode("utf-8")),
        ]
    )


# --- pdf structural variants (hand-built object graphs, the same
# --- byte-deterministic writer pattern as generate_pdf) ----------------------


def _pdf_from_objects(objects: list[bytes]) -> bytes:
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


def _pdf_page_objects(
    content: bytes,
    *,
    annots: bool = False,
    mediabox: str = "[0 0 612 792]",
    rotate: int | None = None,
    cropbox: str | None = None,
) -> list[bytes]:
    """One page's objects, with the page-box fragments the geometry
    fixtures need: ``mediabox``/``rotate``/``cropbox`` spell the page
    dict's /MediaBox, /Rotate, and /CropBox. The defaults are the
    historical bytes exactly (``[0 0 612 792]``, no /Rotate, no /CropBox),
    so every committed fixture the pinned corpora carry builds
    byte-identically through this function."""
    extras = ""
    if rotate is not None:
        extras += f" /Rotate {rotate}"
    if cropbox is not None:
        extras += f" /CropBox {cropbox}"
    annot_entry = b" /Annots [6 0 R]" if annots else b""
    page_dict = (
        "<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >>"
        f" /MediaBox {mediabox}{extras} /Contents 5 0 R"
    ).encode("ascii")
    return [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        page_dict + annot_entry + b" >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + content + b"\nendstream",
    ]


PDF_LINK_URL = "https://handbook.example.com/guide"
PDF_LINK_LABEL = f"Visit the {RICH_LINK_LABEL}"


def generate_pdf_link() -> bytes:
    """A page whose only text sits under a /Link annotation with a /URI
    action: pins [text](uri) in markdown output."""
    content = f"BT /F1 12 Tf 72 700 Td ({PDF_LINK_LABEL}) Tj ET".encode("ascii")
    annot = (
        b"<< /Type /Annot /Subtype /Link /Rect [72 695 200 715]"
        b" /A << /Type /Action /S /URI /URI (https://handbook.example.com/guide) >>"
        b" /Border [0 0 1] >>"
    )
    return _pdf_from_objects(_pdf_page_objects(content, annots=True) + [annot])


def generate_pdf_twocol() -> bytes:
    """A left column (x=50) and a right column (x=350), three lines each:
    the layout where weak reading order interleaves columns into single
    visual lines; the gate demands the columns as separate blocks in
    document order."""
    content = (
        b"BT /F1 12 Tf 50 700 Td (LEFT-A first line) Tj 0 -14 Td (LEFT-A second) Tj"
        b" 0 -14 Td (LEFT-A third) Tj ET\n"
        b"BT /F1 12 Tf 350 700 Td (RIGHT-B first line) Tj 0 -14 Td (RIGHT-B second)"
        b" Tj 0 -14 Td (RIGHT-B third) Tj ET"
    )
    return _pdf_from_objects(_pdf_page_objects(content))


def generate_pdf_heading() -> bytes:
    """A 24pt line over 12pt body: heading detection keys off font size."""
    content = (
        f"BT /F1 24 Tf 50 700 Td ({RICH_H1}) Tj ET\n".encode("ascii")
        + b"BT /F1 12 Tf 50 660 Td (Revenue grew twelve percent.) Tj ET"
    )
    return _pdf_from_objects(_pdf_page_objects(content))


def generate_pdf_blank() -> bytes:
    """A single page with no content stream at all: the contentless shape:
    pdf_oxide's classify says ``"empty"`` (neither extract nor OCR), the
    anydoc backend says NeedsOcr. The gate pins the divergence as a
    documented fact, never a crash, never fabricated content."""
    return _pdf_from_objects(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
        ]
    )


def generate_pdf_scanned() -> bytes:
    """A single image-only page: a 64x64 grayscale FlateDecode image XObject
    with varying pixel values (a deterministic gradient: a uniform raster
    reads as near-empty to the classifier, which is correct behavior, not
    the scanned shape), drawn by a `/Im0 Do` content stream, no text
    operators: the scanned shape every engine must route to OCR
    (pdf_oxide classify: ``"scanned"``, in ``pages_needing_ocr``; the anydoc
    backend: NeedsOcr naming it)."""
    image = zlib.compress(bytes((index * 7) % 256 for index in range(64 * 64)), 9)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Resources << /XObject << /Im0 6 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length 27 >>\nstream\nq 612 0 0 792 0 0 cm /Im0 Do Q\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /XObject /Subtype /Image /Width 64 /Height 64"
        b" /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode"
        + f" /Length {len(image)} >>\nstream\n".encode("ascii")
        + image
        + b"\nendstream",
    ]
    return _pdf_from_objects(objects)


def generate_pdf_mixed() -> bytes:
    """Page 1 born-digital text, page 2 image-only: the mixed document:
    classify must say ``["text", "scanned"]`` with ``pages_needing_ocr ==
    [1]`` (0-based), ``has_text`` true, ``image_only`` false; the text page
    still extracts locally."""
    image = zlib.compress(bytes((index * 7) % 256 for index in range(64 * 64)), 9)
    text = b"BT /F1 12 Tf 50 700 Td (first page line) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 5 0 R >> >>"
        b" /MediaBox [0 0 612 792] /Contents 4 0 R >>",
        f"<< /Length {len(text)} >>\nstream\n".encode("ascii") + text + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Resources << /XObject << /Im0 7 0 R >> >> /Contents 8 0 R >>",
        b"<< /Type /XObject /Subtype /Image /Width 64 /Height 64"
        b" /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode"
        + f" /Length {len(image)} >>\nstream\n".encode("ascii")
        + image
        + b"\nendstream",
        b"<< /Length 27 >>\nstream\nq 612 0 0 792 0 0 cm /Im0 Do Q\nendstream",
    ]
    return _pdf_from_objects(objects)


def generate_pdf_two_page() -> bytes:
    """Two pages, one line each: pins page order, per-page text, and the
    page_count probe."""
    c1 = "BT /F1 12 Tf 50 700 Td (first page line) Tj ET".encode("ascii")
    c2 = "BT /F1 12 Tf 50 700 Td (second page line) Tj ET".encode("ascii")
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
    return _pdf_from_objects(objects)


def generate_pdf_big(lines: int = 6000) -> bytes:
    """One page, `lines` text-showing operators (~80 bytes each): the
    input-scaling cell for GIL bands and the eval runner's timing lane. Not
    committed to the corpus; built on demand."""
    sentence = "The transformer maintenance schedule covers 138kV oil-filled units."
    moves = ["BT", "/F1 12 Tf 50 750 Td"]
    for _ in range(lines):
        moves.append(f"({sentence}) Tj")
        moves.append("0 -12 Td")
    moves.append("ET")
    content = "\n".join(moves).encode("ascii")
    return _pdf_from_objects(_pdf_page_objects(content))


# --- the page-geometry fixtures (standalone, on demand: the
# --- generate_pdf_big precedent — committed-corpus membership would add
# --- them to the conversion matrix's sweeps, which they carry no new
# --- content for; the geometry probes build their own bytes) ---------------


def generate_pdf_crop() -> bytes:
    """A CropBox strictly inside the MediaBox ([100 100 500 700] on
    [0 0 612 792]): the geometry probe's crop-subset golden — the render
    extent 400×600, crop provenance Some((100, 100, 500, 700))."""
    content = b"BT /F1 12 Tf 150 600 Td (cropped page line) Tj ET"
    return _pdf_from_objects(
        _pdf_page_objects(content, cropbox="[100 100 500 700]")
    )


def generate_pdf_rotate(angle: int) -> bytes:
    """A portrait page declared /Rotate `angle`: the swap-law goldens
    (90/270 → 792×612; 180 → 612×792) and, for a non-right angle, the
    engine-normalization pin (rotation 0, unrotated frame)."""
    content = b"BT /F1 12 Tf 72 700 Td (rotated page line) Tj ET"
    return _pdf_from_objects(_pdf_page_objects(content, rotate=angle))


def generate_pdf_media_origin() -> bytes:
    """A nonzero MediaBox origin ([50 50 662 842]): the extent is the
    box's width/height (612×792), never the corners; with a CropBox the
    declared corners survive verbatim."""
    content = b"BT /F1 12 Tf 100 700 Td (offset origin line) Tj ET"
    return _pdf_from_objects(
        _pdf_page_objects(content, mediabox="[50 50 662 842]", cropbox="[100 100 500 700]")
    )


def generate_pdf_zero_pages() -> bytes:
    """A page tree with /Count 0 and no kids: the zero-page shape whose
    geometry answer is the empty list ([] is a count, not an error)."""
    return _pdf_from_objects(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [] /Count 0 >>",
        ]
    )


def generate_pdf_pages(count: int) -> bytes:
    """A `count`-page PDF of bare MediaBox-only page dicts (no content
    streams): the geometry probe's own load fixture — page_geometry parses
    no content, so the page count IS the work, and the per-entry pyclass
    marshalling is what the §9 item-12 heartbeat cell exercises. Not
    committed to the corpus; built on demand (the generate_pdf_big
    precedent)."""
    kids = " ".join(f"{3 + page} 0 R" for page in range(count))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode("ascii"),
    ]
    objects.extend(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>"
        for _ in range(count)
    )
    return _pdf_from_objects(objects)


def generate_pdf_crop_partial() -> bytes:
    """A CropBox that HANGS OFF the MediaBox's edge ([300 300 800 900] on
    [0 0 612 792]): the partial-overlap golden — the entry is the
    INTERSECTION (the engine only exposes the intersection): extent
    312x492, crop (300, 300, 612, 792), never the declared box's own
    500x600. The crop wording ("the declared CropBox reduced to the
    MediaBox — the effective crop") is this fixture's shape."""
    content = b"BT /F1 12 Tf 350 650 Td (partially cropped page line) Tj ET"
    return _pdf_from_objects(
        _pdf_page_objects(content, cropbox="[300 300 800 900]")
    )


def generate_pdf_page_labels() -> bytes:
    """A 6-page PDF whose catalog carries a multi-range /PageLabels
    numbering tree (ISO 32000 §12.4.2): lower-roman from value 3
    (pages 0-2: iii, iv, v), decimal with an "A-" prefix (pages 3-4:
    A-1, A-2), and a styleless prefix-only "cover" range (page 5: the
    constant label). The page-labels probe's load fixture: the tree is
    catalog data, so no page needs content, and the three styles'
    rendered goldens pin the covering rule end to end (never a
    fabricated ordinal)."""
    kids = " ".join(f"{3 + page} 0 R" for page in range(6))
    return _pdf_from_objects(
        [
            b"<< /Type /Catalog /Pages 2 0 R /PageLabels << /Nums "
            b"[0 << /S /r /St 3 >> 3 << /S /D /P (A-) >> 5 << /P (cover) >>] >> >>",
            f"<< /Type /Pages /Kids [{kids}] /Count 6 >>".encode("ascii"),
            *(
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>"
                for _ in range(6)
            ),
        ]
    )


# --- the page-layout fixtures (standalone, on demand: the geometry
# --- fixtures' precedent — none carries conversion content the matrix's
# --- sweeps need; the layout probes build their own bytes) -------------------

LAYOUT_RED_1X1 = (
    b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1"
    b" /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /ASCIIHexDecode"
    b" /Length 7 >>\nstream\nFF0000>\nendstream"
)


def generate_pdf_layout(
    content: bytes,
    *,
    mediabox: str = "[0 0 612 792]",
    rotate: int | None = None,
    cropbox: str | None = None,
    fonts: list[str] | None = None,
    image_ops: bytes | None = None,
    images: list[bytes] | None = None,
) -> bytes:
    """The layout fixtures' one builder: one page with a content stream,
    optional /Rotate and /CropBox page-dict entries, optional extra fonts
    (the /F2+ faces named in ``fonts`` — /F1 is always Helvetica), and
    either an image draw (`image_ops` — the numbers between `cm` and
    `/Im1 Do`; the red 1×1 XObject is added and referenced as /Im1) or a
    content stream drawing ``images[0..]`` itself (the /Im1, /Im2…
    references it spells). Object numbering: catalog 1, pages 2, page 3,
    fonts 4.., content stream next, then the image XObjects."""
    fonts = ["Helvetica", *(fonts or [])]
    if image_ops is not None:
        content = b"q " + image_ops + b" cm /Im1 Do Q"
    content_number = 4 + len(fonts)
    extras = ""
    if rotate is not None:
        extras += f" /Rotate {rotate}"
    if cropbox is not None:
        extras += f" /CropBox {cropbox}"
    xobject_entry = ""
    if image_ops is not None or images:
        xobject_entry = f" /XObject << /Im1 {content_number + 1} 0 R >>"
    resources = (
        "<< /Font << "
        + " ".join(f"/F{1 + i} {4 + i} 0 R" for i in range(len(fonts)))
        + " >>"
        + xobject_entry
        + " >>"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /Resources {resources}"
            f" /MediaBox {mediabox}{extras} /Contents {content_number} 0 R >>"
        ).encode("ascii"),
    ]
    for base_font in fonts:
        objects.append(
            f"<< /Type /Font /Subtype /Type1 /BaseFont /{base_font} >>".encode("ascii")
        )
    objects.append(
        f"<< /Length {len(content)} >>\nstream\n".encode("ascii")
        + content
        + b"\nendstream"
    )
    if image_ops is not None:
        images = [LAYOUT_RED_1X1]
    for image in images or []:
        objects.append(image)
    return _pdf_from_objects(objects)


def generate_pdf_layout_baseline() -> bytes:
    """The D2 pin's fixture: runs at the y-up baselines 700 (24pt "Hello")
    and 100 (12pt "World") — the layout probe's rot-0 identity golden
    (72, 700, ~54.67, 24), no y-flip."""
    content = b"BT /F1 24 Tf 72 700 Td (Hello) Tj ET\nBT /F1 12 Tf 72 100 Td (World) Tj ET"
    return generate_pdf_layout(content)


def generate_pdf_layout_rotated(angle: int) -> bytes:
    """A horizontal run at (72, 700) on a page declared /Rotate `angle`:
    the forward-rule goldens ((700, 485.33, 24, 54.67) at 90;
    (485.33, 68, 54.67, 24) at 180) and the /Rotate 45 normalization pin."""
    content = b"BT /F1 24 Tf 72 700 Td (Hello) Tj ET"
    return generate_pdf_layout(content, rotate=angle)


def generate_pdf_layout_crop() -> bytes:
    """The crop-subset golden: CropBox [100 100 500 700], one run inside
    (150, 600) — lands at (50, 500, 64.03, 24) — and one fully outside
    (30, 30), which the engine drops."""
    content = b"BT /F1 24 Tf 150 600 Td (Inside) Tj ET\nBT /F1 24 Tf 30 30 Td (Outside) Tj ET"
    return generate_pdf_layout(content, cropbox="[100 100 500 700]")


def generate_pdf_layout_crop_partial_span() -> bytes:
    """T7's fixture: a run starting at x=480 that advances past the crop's
    right edge (500) — kept, and its box comes back UNCLIPPED."""
    content = b"BT /F1 24 Tf 480 600 Td (Hang) Tj ET"
    return generate_pdf_layout(content, cropbox="[100 100 500 700]")


def generate_pdf_layout_rotated_run() -> bytes:
    """T3's fixture: a horizontal run (72, 700) plus a rotated-content run
    (`Tm [0 1 -1 0 300 300]`, "Side") on a /Rotate 90 page — the
    inverse-map case: both land in the same frame; the rotated run's box
    is the engine's flattened approximation, carried by
    rotation_degrees=90."""
    content = (
        b"BT /F1 24 Tf 72 700 Td (Hello) Tj ET\n"
        b"BT /F1 24 Tf 0 1 -1 0 300 300 Tm (Side) Tj ET"
    )
    return generate_pdf_layout(content, rotate=90)


def generate_pdf_layout_bold_mid_line() -> bytes:
    """Two runs on ONE baseline, the second /Helvetica-Bold: spans split
    where lines merge (two spans, one line, bold evidence per run)."""
    return generate_pdf_layout(
        b"BT /F1 24 Tf 72 700 Td (Hello) Tj ET\nBT /F2 24 Tf 140 700 Td (World) Tj ET",
        fonts=["Helvetica-Bold"],
    )


def generate_pdf_layout_artifact() -> bytes:
    """A /Artifact-tagged (typed BDC) header run over a body run: the
    is_artifact pin (declared furniture flagged, body not)."""
    content = (
        b"/Artifact << /Type /Pagination /Subtype /Header >> BDC\n"
        b"BT /F1 24 Tf 72 700 Td (Header) Tj ET\nEMC\n"
        b"BT /F1 12 Tf 72 600 Td (Body) Tj ET"
    )
    return generate_pdf_layout(content)


def generate_pdf_layout_artifact_bmc() -> bytes:
    """The T11 limitation's fixture: the BARE `/Artifact BMC` form (no
    properties dictionary) over a body run. The engine's BMC handler sets
    its internal inside-artifact bit but deliberately leaves the artifact
    type unset (extractors/text.rs:6487-6489), and the span carries no
    other artifact bit — so the furniture span is NOT flagged, and
    `is_artifact == false` is NOT evidence of body text. Pinned False so
    the limitation cannot be silently "fixed" without a spec re-read."""
    content = (
        b"/Artifact BMC\n"
        b"BT /F1 24 Tf 72 700 Td (Header) Tj ET\nEMC\n"
        b"BT /F1 12 Tf 72 600 Td (Body) Tj ET"
    )
    return generate_pdf_layout(content)


def generate_pdf_layout_image(rotate: int | None = None) -> bytes:
    """The red 1×1 image drawn at CTM [100 0 0 100 200 200] (user-space
    rect 200,200 100×100), optionally on a /Rotate 90 page: the
    handles-only pins (extract_images would silently return ZERO for the
    1×1 image)."""
    return generate_pdf_layout(b"", rotate=rotate, image_ops=b"100 0 0 100 200 200")


def generate_pdf_layout_two_images() -> bytes:
    """Two images, one page (the content stream draws both): paint_order
    pins content-stream order."""
    return generate_pdf_layout(
        b"q 50 0 0 50 100 100 cm /Im1 Do Q\nq 60 0 0 60 300 400 cm /Im1 Do Q",
        images=[LAYOUT_RED_1X1],
    )


def generate_pdf_layout_pages(count: int, lines: int = 1) -> bytes:
    """`count` pages, each carrying `lines` real text spans (12 lines per
    page makes the native span walk dominate the call's wall — the GIL
    heartbeat cell's op must be several heartbeat ticks long for the
    ratio to resolve, the sibling cells' methodology): the GIL cell's
    fixture (§9 item 12 — page_layout's marshalling is O(pages × boxes)
    nested Python-object construction post-detach, so the fixture needs
    spans, which generate_pdf_pages's MediaBox-only pages lack). Not
    committed to the corpus; built on demand."""
    # Object layout: catalog 1, pages 2, font 3, then per page a page dict
    # (4 + 2i) and its content stream (5 + 2i).
    kids = " ".join(f"{4 + page * 2} 0 R" for page in range(count))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    body = "\n".join(
        f"BT /F1 12 Tf 50 {720 - line * 20} Td (page line {line} of the layout heartbeat"
        " fixture) Tj ET"
        for line in range(lines)
    ).encode("ascii")
    for page in range(count):
        objects.append(

                "<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >>"
                f" /MediaBox [0 0 612 792] /Contents {5 + page * 2} 0 R >>".encode("ascii")

        )
        objects.append(
            f"<< /Length {len(body)} >>\nstream\n".encode("ascii") + body + b"\nendstream"
        )
    return _pdf_from_objects(objects)


def generate_pdf_layout_wmode1() -> bytes:
    """T9's verified fixture: a WMode-1 (tategaki, vertical-writing) run —
    a Type0 font with ``/Encoding /Identity-V`` (the predefined-CMap-name
    vertical signal the engine reads on Tf), a non-embedded CIDFontType2
    descendant (``/DW 1000``), and a ToUnicode CMap mapping CIDs 0x41/0x42
    to 'A'/'B' — drawn at ``300 700 Td``, 24pt. Measured (pdf_oxide
    0.3.78): the span reports bbox ``(300, 700, -48, 24)`` (the run's
    vertical advance, -24 pt/glyph via the spec /DW2 default ``w1y =
    -1000``, flattened onto x as a NEGATIVE width) with
    ``rotation_degrees == 0.0`` (the angle is the text matrix's; the
    writing mode plays no part in it) — the honesty flag does NOT cover
    vertical writing; the line surface re-lays the glyphs over the
    normalized box and reports the REVERSED text ("BA"), box ``(252, 700,
    48, 24)``."""
    tounicode = (
        "/CIDInit /ProcSet findresource begin\n"
        "12 dict begin\n"
        "begincmap\n"
        "/CMapName /Test-V def\n"
        "1 begincodespacerange\n"
        "<0000> <FFFF>\n"
        "endcodespacerange\n"
        "2 beginbfchar\n"
        "<0041> <0041>\n"
        "<0042> <0042>\n"
        "endbfchar\n"
        "endcmap\n"
        "CMapName currentdict /CMap defineresource pop\n"
        "end\n"
        "end"
    ).encode("ascii")
    content = b"BT /F1 24 Tf 300 700 Td <00410042> Tj ET"
    return _pdf_from_objects(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                b"<< /Type /Page /Parent 2 0 R /Resources"
                b" << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>"
            ),
            (
                b"<< /Type /Font /Subtype /Type0 /BaseFont /TestVert /Encoding /Identity-V"
                b" /DescendantFonts [6 0 R] /ToUnicode 7 0 R >>"
            ),
            f"<< /Length {len(content)} >>\nstream\n".encode("ascii")
            + content
            + b"\nendstream",
            (
                b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /TestVert /CIDSystemInfo"
                b" << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /DW 1000 >>"
            ),
            f"<< /Length {len(tounicode)} >>\nstream\n".encode("ascii")
            + tounicode
            + b"\nendstream",
        ]
    )


def generate_pdf_layout_broken_flate() -> bytes:
    """T8's decode-failure fixture: a page whose /Contents declares
    /FlateDecode but carries bytes that are not zlib (the header byte
    alone refuses to inflate). The engine's unwrap_or_default law: the
    page's geometry stays intact and every content list comes back EMPTY,
    never an error."""
    junk = b"this is not zlib data, not even close"
    return _pdf_from_objects(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
            f"<< /Length {len(junk)} /Filter /FlateDecode >>\nstream\n".encode("ascii")
            + junk
            + b"\nendstream",
        ]
    )


def generate_pdf_layout_dangling_contents() -> bytes:
    """T8's dangling-ref fixture: a page dict whose /Contents names an
    object the file does not carry (99 0 R). Same law as the broken-Flate
    shape: geometry intact, empty content lists, never an error."""
    return _pdf_from_objects(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 99 0 R >>",
        ]
    )


# --- the page-tables fixtures (standalone, on demand: the layout
# --- fixtures' precedent; every content stream is the probe crate's
# --- measured fixture verbatim — /tmp/opencode/probe-tables out.txt and
# --- out2-out7.txt are the preserved oracle) --------------------------------


def generate_pdf_table_ruled(rotate: int | None = None, cropbox: str | None = None) -> bytes:
    """The ruled 2×2 intersection grid (the probe's fixture verbatim):
    verticals x ∈ {100, 300, 500} spanning y 600..800, horizontals
    y ∈ {600, 700, 800} spanning x 100..500, one glyph per cell (A/B/C/D
    at (120,750)/(320,750)/(120,650)/(320,650)). The engine's cells are
    the RULING rectangles — A (100,700,200,100) etc. at rotation 0
    (measured out.txt) — so the render-anchored oracle checks the cell
    edges against the rendered ruling positions. The top rule rides 8pt
    above the 792pt page (y=800): the cells' boxes are unclipped, the
    unclipped law's own fixture shape."""
    content = (
        b"0.5 w 100 600 m 100 800 l S\n"
        b"0.5 w 300 600 m 300 800 l S\n"
        b"0.5 w 500 600 m 500 800 l S\n"
        b"0.5 w 100 600 m 500 600 l S\n"
        b"0.5 w 100 700 m 500 700 l S\n"
        b"0.5 w 100 800 m 500 800 l S\n"
        b"BT /F1 12 Tf 120 750 Td (A) Tj ET\n"
        b"BT /F1 12 Tf 320 750 Td (B) Tj ET\n"
        b"BT /F1 12 Tf 120 650 Td (C) Tj ET\n"
        b"BT /F1 12 Tf 320 650 Td (D) Tj ET"
    )
    return generate_pdf_layout(content, rotate=rotate, cropbox=cropbox)


def generate_pdf_table_symmetric(rotate: int | None = None) -> bytes:
    """The 180-symmetric ruled 2×2 (the probe's sym_grid verbatim): the
    grid centered on (306, 396) is invariant under (x, y) → (612−x,
    792−y), so on /Rotate 180 the detector DOES find a grid — with every
    cell's text 180°-swapped (measured out2: r0c0 = "D" where the raw
    top-left cell held "A"). The gate's regression fixture: the refusal
    is a content law, not the vanished-table accident. At rotation 0
    (the control) it serves with the upright cells."""
    content = (
        b"0.5 w 206 346 m 206 446 l S\n"
        b"0.5 w 306 346 m 306 446 l S\n"
        b"0.5 w 406 346 m 406 446 l S\n"
        b"0.5 w 206 346 m 406 346 l S\n"
        b"0.5 w 206 396 m 406 396 l S\n"
        b"0.5 w 206 446 m 406 446 l S\n"
        b"BT /F1 12 Tf 220 420 Td (A) Tj ET\n"
        b"BT /F1 12 Tf 320 420 Td (B) Tj ET\n"
        b"BT /F1 12 Tf 220 370 Td (C) Tj ET\n"
        b"BT /F1 12 Tf 320 370 Td (D) Tj ET"
    )
    return generate_pdf_layout(content, rotate=rotate)


def generate_pdf_table_sideways_ruled(rotate: int) -> bytes:
    """Sideways (rotated-Tm) text inside a ruled 2×2 (the probe's
    sideways_ruled verbatim): the 90/270-with-rulings-and-mapped-spans
    class — the detector's text and ruling inputs share no frame
    (measured out2: tables: 0)."""
    content = (
        b"0.5 w 100 600 m 100 800 l S\n"
        b"0.5 w 300 600 m 300 800 l S\n"
        b"0.5 w 500 600 m 500 800 l S\n"
        b"0.5 w 100 600 m 500 600 l S\n"
        b"0.5 w 100 700 m 500 700 l S\n"
        b"0.5 w 100 800 m 500 800 l S\n"
        b"BT /F1 12 Tf 0 1 -1 0 130 610 Tm (A) Tj ET\n"
        b"BT /F1 12 Tf 0 1 -1 0 330 610 Tm (B) Tj ET\n"
        b"BT /F1 12 Tf 0 1 -1 0 130 710 Tm (C) Tj ET\n"
        b"BT /F1 12 Tf 0 1 -1 0 330 710 Tm (D) Tj ET"
    )
    return generate_pdf_layout(content, rotate=rotate)


def generate_pdf_table_text_only(rotate: int | None = None) -> bytes:
    """The text-only 4×2 (the probe's fixture verbatim): two x-columns
    (100/300), four baselines (700/650/600/550), no rulings anywhere —
    the text-branch detection (span-union cells)."""
    content = (
        b"BT /F1 12 Tf 100 700 Td (Region) Tj ET\n"
        b"BT /F1 12 Tf 300 700 Td (Value) Tj ET\n"
        b"BT /F1 12 Tf 100 650 Td (North) Tj ET\n"
        b"BT /F1 12 Tf 300 650 Td (120) Tj ET\n"
        b"BT /F1 12 Tf 100 600 Td (South) Tj ET\n"
        b"BT /F1 12 Tf 300 600 Td (340) Tj ET\n"
        b"BT /F1 12 Tf 100 550 Td (East) Tj ET\n"
        b"BT /F1 12 Tf 300 550 Td (560) Tj ET"
    )
    return generate_pdf_layout(content, rotate=rotate)


def generate_pdf_table_text_only_rot180() -> bytes:
    """The text-only 4×2 on /Rotate 180 (the same fixture bytes at
    rotate=180 — measured out.txt): the A12 SERVED 180 class — a pathless
    page whose every span is engine-mapped (`page_rotation_applied ==
    180`), so the detector DOES find the grid — one table, every cell's
    bbox the span's own (cell bbox == span bbox, measured), served
    through unmap→forward."""
    return generate_pdf_table_text_only(rotate=180)


def generate_pdf_table_sideways_text_only(rotate: int) -> bytes:
    """The A12 served class: a pathless page whose every span is drawn
    with `Tm [0 1 -1 0 x y]` — engine-mapped on a 90/270 page
    (`page_rotation_applied == rotation`). Measured (out3): at 270 the
    detector finds a 4×4 grid with None-padded cells (the unmap→forward
    goldens); at 90 it finds NOTHING (the detector's sideways limit —
    served empty, refused false)."""
    content = (
        b"BT /F1 12 Tf 0 1 -1 0 100 700 Tm (Region) Tj ET\n"
        b"BT /F1 12 Tf 0 1 -1 0 300 700 Tm (Value) Tj ET\n"
        b"BT /F1 12 Tf 0 1 -1 0 100 650 Tm (North) Tj ET\n"
        b"BT /F1 12 Tf 0 1 -1 0 300 650 Tm (120) Tj ET\n"
        b"BT /F1 12 Tf 0 1 -1 0 100 600 Tm (South) Tj ET\n"
        b"BT /F1 12 Tf 0 1 -1 0 300 600 Tm (340) Tj ET\n"
        b"BT /F1 12 Tf 0 1 -1 0 100 550 Tm (East) Tj ET\n"
        b"BT /F1 12 Tf 0 1 -1 0 300 550 Tm (560) Tj ET"
    )
    return generate_pdf_layout(content, rotate=rotate)


def generate_pdf_table_mixed_spans(rotate: int) -> bytes:
    """The A12 mixed-span refusal class: a pathless page whose spans MIX
    engine-mapped (sideways) and raw (upright) runs — the detector would
    compare two text frames; the fail-closed refusal."""
    content = (
        b"BT /F1 12 Tf 100 700 Td (up) Tj ET\n"
        b"BT /F1 12 Tf 0 1 -1 0 300 650 Tm (side) Tj ET\n"
        b"BT /F1 12 Tf 100 600 Td (raw) Tj ET\n"
        b"BT /F1 12 Tf 0 1 -1 0 300 550 Tm (flip) Tj ET"
    )
    return generate_pdf_layout(content, rotate=rotate)


def generate_pdf_table_stack_disjoint() -> bytes:
    """Two ruled grids stacked with disjoint x-ranges and a 20pt y gap
    (the probe's stacked_disjoint verbatim): the vertical MERGE path's
    fixture — one merged table whose rows are BOTTOM-GROUP-FIRST
    (measured out2), every cell a ruling-rect box."""
    content = (
        b"0.5 w 100 600 m 100 800 l S\n"
        b"0.5 w 200 600 m 200 800 l S\n"
        b"0.5 w 300 600 m 300 800 l S\n"
        b"0.5 w 100 600 m 300 600 l S\n"
        b"0.5 w 100 700 m 300 700 l S\n"
        b"0.5 w 100 800 m 300 800 l S\n"
        b"BT /F1 12 Tf 120 750 Td (T1) Tj ET\n"
        b"BT /F1 12 Tf 220 750 Td (T2) Tj ET\n"
        b"BT /F1 12 Tf 120 650 Td (T3) Tj ET\n"
        b"BT /F1 12 Tf 220 650 Td (T4) Tj ET\n"
        b"0.5 w 350 380 m 350 580 l S\n"
        b"0.5 w 450 380 m 450 580 l S\n"
        b"0.5 w 550 380 m 550 580 l S\n"
        b"0.5 w 350 380 m 550 380 l S\n"
        b"0.5 w 350 480 m 550 480 l S\n"
        b"0.5 w 350 580 m 550 580 l S\n"
        b"BT /F1 12 Tf 370 530 Td (B1) Tj ET\n"
        b"BT /F1 12 Tf 470 530 Td (B2) Tj ET\n"
        b"BT /F1 12 Tf 370 430 Td (B3) Tj ET\n"
        b"BT /F1 12 Tf 470 430 Td (B4) Tj ET"
    )
    return generate_pdf_layout(content)


def generate_pdf_table_cluster_shared_x() -> bytes:
    """Two ruled grids stacked with SHARED x and a 20pt gap, the bottom
    one 3-column (the probe's stacked_grids verbatim): the CLUSTER
    path's fixture — one table, TOP-GROUP-FIRST (measured out2),
    has_header TRUE (the cluster branch's second header source), the
    bbox the cluster's line-extent × row-extent, and the 2-col group
    padded with `"" None` cells."""
    content = (
        b"0.5 w 100 600 m 100 800 l S\n"
        b"0.5 w 300 600 m 300 800 l S\n"
        b"0.5 w 500 600 m 500 800 l S\n"
        b"0.5 w 100 600 m 500 600 l S\n"
        b"0.5 w 100 800 m 500 800 l S\n"
        b"BT /F1 12 Tf 120 700 Td (T1) Tj ET\n"
        b"BT /F1 12 Tf 320 700 Td (T2) Tj ET\n"
        b"0.5 w 100 380 m 100 580 l S\n"
        b"0.5 w 233 380 m 233 580 l S\n"
        b"0.5 w 366 380 m 366 580 l S\n"
        b"0.5 w 500 380 m 500 580 l S\n"
        b"0.5 w 100 380 m 500 380 l S\n"
        b"0.5 w 100 580 m 500 580 l S\n"
        b"BT /F1 12 Tf 120 480 Td (B1) Tj ET\n"
        b"BT /F1 12 Tf 250 480 Td (B2) Tj ET\n"
        b"BT /F1 12 Tf 380 480 Td (B3) Tj ET"
    )
    return generate_pdf_layout(content)


def generate_pdf_table_crop_survivor() -> bytes:
    """The crop survivor (measured out2): the ruled 2×2 under the
    [110 550 550 800] crop — the grid survives with cell boxes that lie
    OUTSIDE the visible box (cell A's x=100 < the crop's left edge 110),
    returned unclipped."""
    return generate_pdf_table_ruled(cropbox="[110 550 550 800]")


def generate_pdf_table_crop_vanished() -> bytes:
    """The crop vanish variant (measured out2): the ruled 2×2 under the
    [150 600 550 800] crop — the TEXT side is crop-filtered (the 2pt
    law), the left column's text is dropped, and the table VANISHES:
    `tables: []`, `tables_refused` false."""
    return generate_pdf_table_ruled(cropbox="[150 600 550 800]")


def generate_pdf_table_prose() -> bytes:
    """Plain prose (measured out3): three full-width lines — no table."""
    content = (
        b"BT /F1 12 Tf 72 700 Td (It was a dark and stormy night; the rain fell) Tj ET\n"
        b"BT /F1 12 Tf 72 682 Td (over the plain, and the waves dashed against the rocks.) Tj ET\n"
        b"BT /F1 12 Tf 72 664 Td (The captain stood at the rail and watched the sea.) Tj ET"
    )
    return generate_pdf_layout(content)


def generate_pdf_table_toc() -> bytes:
    """A TOC with INLINE dot leaders (measured out3): one span per entry
    (single-column) — the detector-limit pin. The measured boundary,
    stated in the test's docstring: a TWO-column TOC shape (title + page
    number spans) IS detected as a table by the same engine (out3) —
    these filters are heuristics; the FIXTURE BYTES are pinned."""
    content = (
        b"BT /F1 18 Tf 72 720 Td (Contents) Tj ET\n"
        b"BT /F1 12 Tf 72 680 Td (Introduction . . . . . . . . . . . . 1) Tj ET\n"
        b"BT /F1 12 Tf 72 656 Td (Methods and materials . . . . . . . . . . . . 9) Tj ET\n"
        b"BT /F1 12 Tf 72 632 Td (Results of the study . . . . . . . . . . . . 17) Tj ET"
    )
    return generate_pdf_layout(content)


def generate_pdf_table_multi() -> bytes:
    """Two ruled grids stacked far apart (disjoint x, a 100pt y gap —
    past the merge's reach; measured out4): TWO tables, in the engine's
    own order (the BOTTOM grid first)."""
    content = (
        b"0.5 w 100 600 m 100 800 l S\n"
        b"0.5 w 200 600 m 200 800 l S\n"
        b"0.5 w 300 600 m 300 800 l S\n"
        b"0.5 w 100 600 m 300 600 l S\n"
        b"0.5 w 100 700 m 300 700 l S\n"
        b"0.5 w 100 800 m 300 800 l S\n"
        b"BT /F1 12 Tf 120 750 Td (T1) Tj ET\n"
        b"BT /F1 12 Tf 220 750 Td (T2) Tj ET\n"
        b"BT /F1 12 Tf 120 650 Td (T3) Tj ET\n"
        b"BT /F1 12 Tf 220 650 Td (T4) Tj ET\n"
        b"0.5 w 350 300 m 350 500 l S\n"
        b"0.5 w 450 300 m 450 500 l S\n"
        b"0.5 w 550 300 m 550 500 l S\n"
        b"0.5 w 350 300 m 550 300 l S\n"
        b"0.5 w 350 400 m 550 400 l S\n"
        b"0.5 w 350 500 m 550 500 l S\n"
        b"BT /F1 12 Tf 370 450 Td (B1) Tj ET\n"
        b"BT /F1 12 Tf 470 450 Td (B2) Tj ET\n"
        b"BT /F1 12 Tf 370 350 Td (B3) Tj ET\n"
        b"BT /F1 12 Tf 470 350 Td (B4) Tj ET"
    )
    return generate_pdf_layout(content)


def generate_pdf_table_cap() -> bytes:
    """The A6 cap fixture: 751 verticals + 751 horizontals = 1502 table
    primitives — one past the markdown lane's MAX_TABLE_EDGES bail-out.
    Detection is skipped: `tables: []`, `tables_refused` true (at
    rotation 0 — the reading rule that distinguishes the cap from the
    frame refusals)."""
    rules = []
    for i in range(751):
        x = 100.0 + i * 0.5
        rules.append(f"0.5 w {x} 600 m {x} 800 l S".encode("ascii"))
    for i in range(751):
        y = 600.0 + i * 0.25
        rules.append(f"0.5 w 100 {y} m 500 {y} l S".encode("ascii"))
    return generate_pdf_layout(b"\n".join(rules))


def generate_pdf_table_pages(count: int) -> bytes:
    """`count` pages, each carrying the ruled 2×2 grid (the rot-0
    intersection fixture's content): the GIL heartbeat cell's fixture —
    the tables marshalling is O(pages) entries + O(pages × tables ×
    cells) nested construction post-detach, so the op must be several
    heartbeat ticks long (the sibling cells' methodology). Not committed
    to the corpus; built on demand."""
    # Object layout: catalog 1, pages 2, font 3, then per page a page
    # dict (4 + 2i) and its content stream (5 + 2i).
    kids = " ".join(f"{4 + page * 2} 0 R" for page in range(count))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    body = b"\n".join(
        [
            b"0.5 w 100 600 m 100 800 l S",
            b"0.5 w 300 600 m 300 800 l S",
            b"0.5 w 500 600 m 500 800 l S",
            b"0.5 w 100 600 m 500 600 l S",
            b"0.5 w 100 700 m 500 700 l S",
            b"0.5 w 100 800 m 500 800 l S",
            b"BT /F1 12 Tf 120 750 Td (A) Tj ET",
            b"BT /F1 12 Tf 320 750 Td (B) Tj ET",
            b"BT /F1 12 Tf 120 650 Td (C) Tj ET",
            b"BT /F1 12 Tf 320 650 Td (D) Tj ET",
        ]
    )
    for page in range(count):
        objects.append(
            "<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >>"
            f" /MediaBox [0 0 612 792] /Contents {5 + page * 2} 0 R >>".encode("ascii")
        )
        objects.append(
            f"<< /Length {len(body)} >>\nstream\n".encode("ascii") + body + b"\nendstream"
        )
    return _pdf_from_objects(objects)


# --- the office geometry fixtures (standalone, on demand: the same
# --- generate_pdf_big precedent — none of them carries conversion content
# --- the matrix's sweeps need, and the committed corpora stay unpinned to
# --- them) --------------------------------------------------------------

_DOCX_PGSZ = '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/></w:sectPr>'
_DOCX_MARGINS_ONLY = (
    '<w:sectPr><w:pgMar w:top="1440" w:bottom="1440" w:left="1800" w:right="1800"/></w:sectPr>'
)


def _docx_with_sectpr(sectpr: str) -> bytes:
    """generate_docx's container with the body's trailing `<w:sectPr>`
    spelled by the fixture (the office geometry lane's raw-layer
    fixtures)."""
    body = "".join(_docx_paragraph(paragraph) for paragraph in DOCX_PARAGRAPHS)
    document = (
        _XML_DECL
        + f'<w:document xmlns:w="{_W_NS}"><w:body>{body}{sectpr}</w:body></w:document>\n'
    )
    return _zip_bytes(
        [
            ("[Content_Types].xml", DOCX_CONTENT_TYPES.encode("utf-8")),
            ("_rels/.rels", DOCX_RELS.encode("utf-8")),
            ("word/document.xml", document.encode("utf-8")),
        ]
    )


def generate_docx_pgsz() -> bytes:
    """A pgSz-bearing docx (Letter, 12240×15840 twips): the geometry
    probe's Some(612.0, 792.0) exact golden."""
    return _docx_with_sectpr(_DOCX_PGSZ)


def generate_docx_margins_only() -> bytes:
    """A margins-only sectPr (no `<w:pgSz>`): the raw-layer amendment's
    own pinning fixture — the geometry probe must answer None, where the
    to_ir() path would fabricate Letter."""
    return _docx_with_sectpr(_DOCX_MARGINS_ONLY)


def generate_pptx_sldsz() -> bytes:
    """generate_pptx_rich's deck with a declared slide size
    (12192000×6858000 EMU): the geometry probe's Some(960.0, 540.0)
    exact golden (EMU ÷ 12700.0 in float)."""
    with zipfile.ZipFile(io.BytesIO(generate_pptx_rich())) as archive:
        entries = [(item.filename, archive.read(item.filename)) for item in archive.infolist()]
    rewritten = []
    for name, data in entries:
        if name == "ppt/presentation.xml":
            text = data.decode("utf-8")
            marker = "</p:sldIdLst>"
            text = text.replace(
                marker,
                marker + '<p:sldSz cx="12192000" cy="6858000"/>',
            )
            data = text.encode("utf-8")
        rewritten.append((name, data))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in rewritten:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return buffer.getvalue()


def generate_pptx_empty() -> bytes:
    """A deck whose slide list is empty and which carries no slide parts:
    the zero-slide shape (geometry answer: the empty list), if the
    engine's parser accepts it."""
    presentation = (
        _XML_DECL
        + f'<p:presentation xmlns:p="{_P_NS}" xmlns:r="{_DOC_NS}">'
        "<p:sldIdLst/></p:presentation>"
    )
    content_types = (
        _XML_DECL
        + f'<Types xmlns="{_CT_NS}">'
        + _CT_RELS
        + _CT_XML
        + _CT_PPTX_MAIN
        + "</Types>"
    )
    return _zip_bytes(
        [
            ("[Content_Types].xml", content_types.encode("utf-8")),
            ("_rels/.rels", _root_rels("ppt/presentation.xml").encode("utf-8")),
            ("ppt/presentation.xml", presentation.encode("utf-8")),
        ]
    )


def generate_xlsx_chart() -> bytes:
    """generate_xlsx's workbook plus one `xl/charts/chart1.xml` part
    carrying chart text: the IR gains a "Charts" section for it (the
    converter's worksheet+Charts count) while the geometry probe's
    length law stays the RAW worksheet count — the fixture that pins the
    two counts apart."""
    chart = (
        _XML_DECL
        + '<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"'
        + ' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        + "<c:chart><c:title><c:tx><c:rich><a:p><a:r><a:t>Units per quarter"
        + "</a:t></a:r></a:p></c:rich></c:tx></c:title>"
        + '<c:plotArea><c:barChart><c:ser><c:tx><c:v>Series name</c:v></c:tx>'
        + '<c:cat><c:strRef><c:strCache><c:pt idx="0"><c:v>Q1</c:v></c:pt>'
        + "</c:strCache></c:strRef></c:cat></c:ser></c:barChart></c:plotArea>"
        + "</c:chart></c:chartSpace>"
    )
    with zipfile.ZipFile(io.BytesIO(generate_xlsx())) as archive:
        entries = [(item.filename, archive.read(item.filename)) for item in archive.infolist()]
    return _zip_bytes(entries + [("xl/charts/chart1.xml", chart.encode("utf-8"))])


# --- the registry ------------------------------------------------------------
#
# The corpus's five documents ride along (same bytes, no duplication) so the
# engine matrix covers every format the suite can build; the rich variants
# add the structural depth the gates need. Markdown is deliberately absent:
# it is already ir, not an engine format. The OOXML alias containers
# (docm_rich/xlsm/ppsx_rich: the [Content_Types].xml rewrites above) are
# fixture-covered: the engines convert them as their base kinds, resolved
# under the base names. xlsb is deliberately not among them: its sheets
# are BIFF12 .bin streams, not worksheet XML, so genuine xlsb content is
# refused (the name routes the Excel kind, vocabulary sugar only; the
# refusal is pinned in the engines suite). The remaining vocabulary gap is
# the legacy MS-binary trio (doc/xls/ppt): OLE compound files are not
# hand-generatable without an Office writer, so no fixture pins them:
# their coverage rests on the crate-side routing-table test
# (src/documents_impl.rs) and the mutation lane's typed-error contract,
# and docs/documents.md discloses them as uncovered.
#
# The engines kinds' content oracle: EXPECTED_TEXT began as the five-document
# corpus's extraction oracle; the matrix's own kinds (no stdlib reader, no
# extract()) join it here as pure content oracles: the lines every
# conversion must carry. tests/test_documents_engines.py's _oracle_lines
# reads EXPECTED_TEXT first, so these entries gate the kinds through the
# existing matrix with no test-file edits. The update sits here, after the
# RICH_* vocabulary the lines cite.

EXPECTED_TEXT.update(
    {
        "ods_rich": "\n".join((*RICH_TABLE_HEADER, *RICH_TABLE_ROW, *RICH_TABLE_ROW2)),
        "odp_rich": "\n".join((SLIDE1_TITLE, *SLIDE1_BODY, SLIDE2_TITLE)),
        "epub_rich": "\n".join(
            (RICH_H1, RICH_H2, RICH_LINK_SENTENCE, *RICH_LIST_FLAT, RICH_LIST_NESTED)
        ),
    }
)

ENGINES_CORPUS: dict[str, bytes] = {
    "rtf": CORPUS["rtf"],
    "docx": CORPUS["docx"],
    "docx_rich": generate_docx_rich(),
    "docm_rich": generate_docm_rich(),
    "xlsx": CORPUS["xlsx"],
    "xlsm": generate_xlsm(),
    "pptx_rich": generate_pptx_rich(),
    "ppsx_rich": generate_ppsx_rich(),
    "odt_rich": generate_odt_rich(),
    "ods_rich": generate_ods_rich(),
    "odp_rich": generate_odp_rich(),
    "epub_rich": generate_epub_rich(),
    "csv_rich": generate_csv_rich(),
    "html_rich": generate_html_rich(),
    "pdf": CORPUS["pdf"],
    "pdf_link": generate_pdf_link(),
    "pdf_twocol": generate_pdf_twocol(),
    "pdf_heading": generate_pdf_heading(),
    "pdf_blank": generate_pdf_blank(),
    "pdf_scanned": generate_pdf_scanned(),
    "pdf_mixed": generate_pdf_mixed(),
    "pdf_two_page": generate_pdf_two_page(),
}

ENGINES_FILENAMES: dict[str, str] = {
    "rtf": "engines_notes.rtf",
    "docx": "engines_report.docx",
    "docx_rich": "engines_annual_report.docx",
    "docm_rich": "engines_annual_report_macro.docm",
    "xlsx": "engines_samples.xlsx",
    "xlsm": "engines_samples_macro.xlsm",
    "pptx_rich": "engines_review.pptx",
    "ppsx_rich": "engines_review_show.ppsx",
    "odt_rich": "engines_manual.odt",
    "ods_rich": "engines_units.ods",
    "odp_rich": "engines_review.odp",
    "epub_rich": "engines_handbook.epub",
    "csv_rich": "engines_units.csv",
    "html_rich": "engines_page.html",
    "pdf": "engines_report.pdf",
    "pdf_link": "engines_link.pdf",
    "pdf_twocol": "engines_twocol.pdf",
    "pdf_heading": "engines_heading.pdf",
    "pdf_blank": "engines_blank.pdf",
    "pdf_scanned": "engines_scanned.pdf",
    "pdf_mixed": "engines_mixed.pdf",
    "pdf_two_page": "engines_two_page.pdf",
}

# The engines matrix's committed home: its own directory (not under
# tests/corpus/, whose pin test asserts that directory holds exactly the
# five-document corpus: a sibling keeps both registries' pins independent).
ENGINES_DIR = CORPUS_DIR.parent / "engines_corpus"


def write_engines_corpus(directory: Path = ENGINES_DIR) -> None:
    """Materialize the engines matrix (committed copies under
    tests/engines_corpus/; the pin test re-derives and byte-compares)."""
    directory.mkdir(parents=True, exist_ok=True)
    for kind, raw in ENGINES_CORPUS.items():
        (directory / ENGINES_FILENAMES[kind]).write_bytes(raw)
