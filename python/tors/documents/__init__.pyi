"""The documents-extraction surface's typed shim stub: the API the
`tors-documents` payload wheel provides, mirrored standalone (see
tors/documents/__init__.py for the shim's runtime re-export and install
hint).

Deliberately not a `from tors_documents import ...` re-export, even
though the runtime shim does exactly that: base-only installs (``pip
install tors`` without the extra) still ship this stub, and a type check
on a base-only environment must resolve `tors.documents.Format` and the
typed signatures: an import of the absent payload wheel would error on
every name instead, making the stub useless exactly where it is the only
documentation of the optional surface. The cost is duplication with
tors_documents/__init__.pyi, which stays the source of truth: keep this
mirror in lockstep with it (drift here is a stub bug, not a behavior
change; the runtime surface is the payload's either way)."""

from __future__ import annotations

import os
from enum import Enum

__version__: str
"""The payload wheel's version, re-exported from ``tors_documents``: the
shim bakes none of its own, and release-please bumps both wheels together
so they cannot drift apart. Runtime nuance: the re-export exists only
where the payload is installed (a base-only install refuses the whole
module with the install hint), so a base-only type check resolves this
declaration without a runtime behind it."""

class Backend(str, Enum):
    """The engine selector: ``AUTO`` routes by the measured table,
    ``OXIDE``/``ANYDOC`` force one engine and raise on a format it cannot
    read, never a silent fallback. Members are their accepted strings, so
    ``backend="auto"`` and ``backend=Backend.AUTO`` are the same call. On
    the geometry surfaces (``page_geometry``/``page_layout``) ``AUTO``
    and ``OXIDE`` are additionally identical for every format (the
    geometry lane is a property of the format, not a choice), and the
    no-geometry formats (html/csv/rtf/epub/odt/ods/odp) answer the same
    single entry on every backend — ``ANYDOC`` included: no engine runs
    there, so nothing is refused and nothing is fallen back (documented
    divergences from this enum's conversion contract)."""

    AUTO = "auto"
    OXIDE = "oxide"
    ANYDOC = "anydoc"

class Format(str, Enum):
    """The format vocabulary: the ``format=`` names, the resolved-format
    values, and ``sniff``'s answers. Members ARE their strings, so plain
    strings the Rust validator accepts (container variants like
    ``docm``/``xlsm``/``ppsx``, and ``tsv`` as an input name) keep working
    without enum churn. ``tsv`` maps onto the csv kind: it is an accepted
    INPUT name, never a resolved or sniffed one (those report ``csv``)."""

    PDF = "pdf"
    HTML = "html"
    DOC = "doc"
    DOCX = "docx"
    XLS = "xls"
    XLSX = "xlsx"
    PPT = "ppt"
    PPTX = "pptx"
    RTF = "rtf"
    ODT = "odt"
    ODS = "ods"
    ODP = "odp"
    EPUB = "epub"
    CSV = "csv"
    TSV = "tsv"

class PageKind(str, Enum):
    """One page's ``pdf_classify`` verdict. ``EMPTY`` is distinct from
    ``SCANNED``: a blank page is neither extractable nor an image to
    recover, and ``pages_needing_ocr`` deliberately excludes it."""

    TEXT = "text"
    SCANNED = "scanned"
    IMAGE_TEXT = "image_text"
    MIXED = "mixed"
    EMPTY = "empty"

class NeedsOcrError(ValueError):
    """A PDF's scanned/image-only pages, which no local engine can read:
    route the document to an OCR stage. Raised by the anydoc backend's PDF
    lane (``backend="anydoc"``); the auto/oxide pdf_oxide lane returns
    empty output for scanned pages instead, leaving the routing decision
    to ``pdf_classify``/``pdf_extract``."""

    pages: list[int]
    """The 0-based page indices needing OCR: the same convention as
    ``pages=`` and ``pages_needing_ocr``."""
    page_count: int

class PdfClassification:
    """The ``pdf_classify`` result: the text-vs-image preflight's answer,
    with the routing rules (``has_text``/``image_only``) derived in the
    native getters, exactly once."""
    @property
    def page_count(self) -> int:
        """Pages in the document."""
    @property
    def page_kinds(self) -> list[PageKind]:
        """Every page's verdict, page order."""
    @property
    def pages_needing_ocr(self) -> list[int]:
        """The 0-based indices of image-only pages (blank pages are
        deliberately not listed; they are ``PageKind.EMPTY``)."""
    @property
    def has_text(self) -> bool:
        """At least one page carries a text layer: extraction will yield
        something."""
    @property
    def image_only(self) -> bool:
        """Every page is image-only: nothing local can read this
        document; route it to an OCR stage."""

class PageGeometry:
    """One ``page_geometry`` entry: a unit's declared render box, in
    points — the native pyclass exposed directly (no Python wrapper view:
    the deliberate deviation from the PdfClassification wrapper
    precedent, whose wrapper exists to re-type verdict strings a
    geometry entry has none of); ``==`` is structural (the four
    fields)."""
    @property
    def width_pt(self) -> float:
        """The unit's render width in points (positive, finite; the axes
        swapped for a 90/270 PDF rotation)."""
    @property
    def height_pt(self) -> float:
        """The unit's render height in points (the same law)."""
    @property
    def rotation(self) -> int:
        """The declared rotation, engine-normalized: {0, 90, 180, 270} on
        the PDF lane (a non-right-angle ``/Rotate`` reports 0), always 0
        on the office lanes."""
    @property
    def crop(self) -> tuple[float, float, float, float] | None:
        """The PDF page's declared CropBox reduced to the MediaBox (the
        effective crop: the intersection the engine's visible box stands
        on), normalized corners, the unrotated source frame, points —
        provenance for the extent, never re-applied, ``None`` unless a
        CropBox is declared AND it actually crops; ``None`` on every
        office lane."""

class PageLayout:
    """One ``page_layout`` entry: the unit's geometry plus its text
    spans, lines, and image boxes in the page's declared render space —
    the native pyclass exposed directly (the ``PageGeometry`` deviation
    note applies to all five layout classes); ``==`` is structural (the
    five fields)."""
    @property
    def page_index(self) -> int:
        """The 0-based page (or section/slide/worksheet index) this entry
        describes — mandatory on every entry: the ``pages=``
        normalization reorders."""
    @property
    def geometry(self) -> PageGeometry | None:
        """The unit's declared render box (the ``page_geometry`` entry):
        ``None`` where the unit's box cannot be read or declares no
        positive extent — then the layout lists are empty too (no frame,
        no boxes)."""
    @property
    def spans(self) -> list[TextSpan]:
        """The unit's text spans, reading order (PDF pages only; office
        units carry empty lists — no office engine exposes text
        boxes)."""
    @property
    def lines(self) -> list[TextLine]:
        """The unit's text lines, reading order, words nested (PDF pages
        only)."""
    @property
    def images(self) -> list[ImageBox]:
        """The unit's image boxes, paint order (PDF pages only)."""

class TextSpan:
    """One ``page_layout`` text span: a run's text, its box, and the
    font/style evidence; ``==`` is structural (the eight fields)."""
    @property
    def text(self) -> str:
        """The run's text."""
    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """The run's box, ``(x, y, w, h)``: points, bottom-left origin of
        the page's render space; y is the box's bottom edge; for text the
        bottom edge is the glyph baseline and the box extends one
        font-size upward (em-box; descenders may extend below). Boxes are
        unclipped; the page box is the consumer's clip."""
    @property
    def font_size(self) -> float:
        """The run's font size in points (the em-box's height)."""
    @property
    def font_name(self) -> str:
        """The run's font name."""
    @property
    def is_bold(self) -> bool:
        """Bold evidence."""
    @property
    def is_italic(self) -> bool:
        """Italic evidence."""
    @property
    def rotation_degrees(self) -> float:
        """The run's own content rotation in degrees — non-zero marks a
        sideways run whose box is the engine's flattened approximation
        (on a rotated page, transposed; the line box degrades to a
        square): carried honestly, not re-derived."""
    @property
    def is_artifact(self) -> bool:
        """Declared ``/Artifact`` marked content (running headers/footers,
        page numbers, watermarks): the consumer tells declared furniture
        from body with this."""

class Word:
    """One word inside a ``TextLine`` (``page_layout``
    ``lines[].words``): the text and its box; ``==`` is structural (both
    fields)."""
    @property
    def text(self) -> str:
        """The word's text."""
    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """The word's box, ``(x, y, w, h)``: points, bottom-left origin
        of the page's render space; y is the box's bottom edge; for text
        the bottom edge is the glyph baseline and the box extends one
        font-size upward (em-box; descenders may extend below). Boxes are
        unclipped; the page box is the consumer's clip."""

class TextLine:
    """One ``page_layout`` text line: the engine's own line (its word
    clustering is the engine's, never re-implemented downstream); ``==``
    is structural (the three fields — the words included). The join law:
    the line's text is its words' texts joined with single spaces, and
    every word's box is inside the line's box."""
    @property
    def text(self) -> str:
        """The line's text (the engine's join of its words)."""
    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """The line's box, ``(x, y, w, h)``: points, bottom-left origin
        of the page's render space; y is the box's bottom edge; for text
        the bottom edge is the glyph baseline and the box extends one
        font-size upward (em-box; descenders may extend below). Boxes are
        unclipped; the page box is the consumer's clip."""
    @property
    def words(self) -> list[Word]:
        """The line's words, in reading order."""

class ImageBox:
    """One ``page_layout`` image box: an image draw's placement, from the
    engine's image HANDLES (no decode, no silent small-image filter, no
    swallowed errors); ``==`` is structural (the five fields)."""
    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """The image's axis-aligned box, ``(x, y, w, h)``: points,
        bottom-left origin of the page's render space; y is the box's
        bottom edge; for text the bottom edge is the glyph baseline and
        the box extends one font-size upward (em-box; descenders may
        extend below). Boxes are unclipped; the page box is the
        consumer's clip."""
    @property
    def width_px(self) -> int:
        """The image's pixel width."""
    @property
    def height_px(self) -> int:
        """The image's pixel height."""
    @property
    def rotation_degrees(self) -> float:
        """The CTM-derived content rotation in degrees (the bbox is
        axis-aligned but the pixels may be turned)."""
    @property
    def paint_order(self) -> int:
        """Page-local identity/z-order: the image's index among the
        page's painted images, in content-stream order."""

def to_markdown(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    format: Format | str | None = ...,
    backend: Backend | str = ...,
    pages: int | list[int] | tuple[int, int] | None = ...,
    password: str | None = None,
    max_bytes: int | None = None,
) -> tuple[Format, str]:
    """Convert any working-format document (pdf, doc/docx, xls/xlsx,
    ppt/pptx, rtf, odt/ods/odp, epub, csv/tsv, html) to GitHub-Flavored
    Markdown on the measured-best engine for its format. ``path=`` must
    name a regular file (FIFOs/devices/sockets are refused before the
    read). ``password=`` unlocks an encrypted PDF (the PDF kinds only);
    an explicit ``max_bytes=`` is binding on every engine lane, checked
    before a byte is read or copied, while ``None`` keeps the 32 MiB
    default (enforced after the read, on the anydoc/oxide lanes only).
    Returns ``(Format, markdown)``: the format the conversion actually
    used. The whole read+sniff+convert pass runs GIL-free. ⚠ A malformed
    PDF can crash the process (uncatchable native stack exhaustion in the
    pdf_oxide engine): sandbox untrusted PDFs (docs/documents.md, SECURITY.md)."""

def to_text(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    format: Format | str | None = ...,
    backend: Backend | str = ...,
    pages: int | list[int] | tuple[int, int] | None = ...,
    password: str | None = None,
    max_bytes: int | None = None,
) -> tuple[Format, str]:
    """The same conversion, routing, source, ``pages=``, ``password=``,
    and ``max_bytes=`` semantics as ``to_markdown``, with the markdown
    normalized to plain text: one text shape for every format and
    engine. Returns ``(Format, text)``. ⚠ A malformed PDF can crash the
    process (uncatchable native stack exhaustion in the pdf_oxide engine):
    sandbox untrusted PDFs (docs/documents.md, SECURITY.md)."""

def sniff(data: bytes) -> Format | None:
    """The content-marker format detector over bytes alone: what
    ``to_markdown`` would resolve ``data`` to with no path and no
    extension. ``None`` = not a document at all, or a guard-refused shape
    such as JSON-lines (record lines the delimiter witness alone would
    claim as csv). Not a bounded marker scan: the package containers are
    parsed to answer (a 120 KiB zip measured 267 MiB peak RSS). Runs
    GIL-free."""

def pdf_classify(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
    backend: Backend | str = ...,
    max_bytes: int | None = None,
) -> PdfClassification:
    """The cheap text-vs-image preflight over an open PDF: no content
    conversion, no OCR, no rasterization. Encrypted documents fail closed
    (``ValueError``). ``backend=`` is ``"auto"``/``"oxide"`` (the pdf_oxide
    lane; ``"anydoc"`` is a capability refusal: its PDF surface is the
    ``to_markdown``/``to_text`` conversion pair); an explicit
    ``max_bytes=`` binds before the read, ``None`` leaves the lane
    unmetered. Runs GIL-free. ⚠ A malformed PDF can crash the process
    (uncatchable native stack exhaustion in the pdf_oxide engine): sandbox
    untrusted PDFs (docs/documents.md, SECURITY.md)."""

def pdf_extract(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
    backend: Backend | str = ...,
    max_bytes: int | None = None,
) -> tuple[list[str], str]:
    """Open a PDF and return ``(per_page_plain_text, markdown)``: one
    GIL-free pass over one open document (the parse is paid once for both
    outputs). An image-only page is an empty string, not an error; routing
    decisions are the caller's. ``backend=`` is ``"auto"``/``"oxide"``
    (the pdf_oxide lane; ``"anydoc"`` is a capability refusal: the
    per-page probe is the OCR-routing signal and anydoc has none; its
    PDF surface is the ``to_markdown``/``to_text`` conversion pair); an
    explicit ``max_bytes=`` binds before the read, ``None`` leaves the
    lane unmetered. ⚠ A malformed PDF can crash the process (uncatchable
    native stack exhaustion in the pdf_oxide engine): sandbox untrusted
    PDFs (docs/documents.md, SECURITY.md)."""

def pdf_page_count(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
    backend: Backend | str = ...,
    max_bytes: int | None = None,
) -> int:
    """The page tree and nothing else: open + count, no content
    extraction, one GIL-free pass. ``backend=`` is ``"auto"``/``"oxide"``
    (the pdf_oxide lane; ``"anydoc"`` is a capability refusal: a count
    its reader never returns on success; its PDF surface is the
    ``to_markdown``/``to_text`` conversion pair); an explicit
    ``max_bytes=`` binds before the read, ``None`` leaves the lane
    unmetered. ⚠ A malformed PDF can crash the process (uncatchable
    native stack exhaustion in the pdf_oxide engine): sandbox untrusted
    PDFs (docs/documents.md, SECURITY.md)."""

def pdf_link_uris(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
    backend: Backend | str = ...,
    max_bytes: int | None = None,
) -> list[list[str]]:
    """The ``/Annots`` link walk: for every page, the URIs of its link
    annotations, in annotation order: the raw navigation surface no text
    rendering carries. Verbatim (never deduped); URI actions only.
    ``backend=`` is ``"auto"``/``"oxide"`` (the pdf_oxide lane;
    ``"anydoc"`` is a capability refusal: it has no annotation surface;
    its PDF surface is the ``to_markdown``/``to_text`` conversion pair);
    an explicit ``max_bytes=`` binds before the read, ``None`` leaves the
    lane unmetered. ⚠ A malformed PDF can crash the process (uncatchable
    native stack exhaustion in the pdf_oxide engine): sandbox untrusted
    PDFs (docs/documents.md, SECURITY.md)."""

def page_geometry(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    format: Format | str | None = ...,
    password: str | None = None,
    backend: Backend | str = ...,
    max_bytes: int | None = None,
) -> list[PageGeometry | None]:
    """The declared-box geometry probe: one entry per unit of the
    resolved format — PDF pages (the visible box's render extent, axes
    swapped for a 90/270 rotation; the engine-normalized ``/Rotate``; the
    declared CropBox reduced to the MediaBox (the effective crop) as
    ``crop`` provenance), docx sections (the RAW
    layer's declared size, twips ÷ 20 — a margins-only section is
    ``None``, never a fabricated Letter), pptx slides (the raw deck-wide
    slide size, EMU ÷ 12700.0, ``None`` everywhere when undeclared),
    xlsx worksheets (all ``None``: an unbounded grid is not a page),
    legacy doc/xls/ppt sections (all ``None``), and exactly one ``None``
    entry for the single-flow formats (html/csv/rtf/epub/odt/ods/odp:
    unit = whole document). ``[]`` only for zero units. ``format=``
    names the format (explicit name wins over sniff wins over
    extension); ``backend=`` is ``"auto"``/``"oxide"`` — identical on
    this surface for every format — while ``"anydoc"`` is refused where
    geometry is obtainable (pdf and the office families) and answered
    ``[None]`` on the no-geometry row; ``password=`` unlocks an encrypted
    PDF (any other format: a clean error; an encrypted OOXML package: a
    typed ``ValueError``); an explicit ``max_bytes=`` binds before the
    read, ``None`` keeps the 32 MiB default ceiling on the office leg.
    Runs GIL-free. ⚠ A malformed PDF can crash the process (uncatchable
    native stack exhaustion in the pdf_oxide engine): sandbox untrusted
    PDFs (docs/documents.md, SECURITY.md)."""

def page_layout(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    format: Format | str | None = ...,
    password: str | None = None,
    backend: Backend | str = ...,
    pages: int | list[int] | tuple[int, int] | None = ...,
    max_bytes: int | None = None,
) -> list[PageLayout]:
    """The layout probe: per unit of the resolved format's geometry lane,
    the unit's geometry plus — PDF pages only — the page's text spans,
    text lines (words nested), and image boxes, all in the page's
    declared render space (points, bottom-left origin, y-up; /Rotate
    applied; one transform owned by the library, the same frame
    ``page_geometry`` declares). Office rows serve their declared
    geometry with empty layout lists (PDF-only for layout); the
    no-geometry formats answer one ``geometry=None`` entry. ``pages=``
    normalizes exactly as the conversion pair's (sort + dedup; the
    byte-pinned empty-selection and out-of-range errors) and the output
    is parallel to the NORMALIZED selection, every entry carrying
    ``page_index``. ``backend=`` is ``"auto"``/``"oxide"`` — identical on
    this surface — while ``"anydoc"`` is refused where geometry is
    obtainable (pdf and the office families); ``password=`` unlocks an
    encrypted PDF (any other format: a clean error; an encrypted OOXML
    package: a typed ``ValueError``); an explicit ``max_bytes=`` binds
    before the read, ``None`` keeps the 32 MiB default ceiling on the
    office leg. Runs GIL-free. ⚠ A malformed PDF can crash the process
    (uncatchable native stack exhaustion in the pdf_oxide engine):
    sandbox untrusted PDFs (docs/documents.md, SECURITY.md)."""
