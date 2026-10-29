"""The tors.documents payload's typed surface (see tors_documents/__init__.py).

Every signature here is the live package's, pinned by probe against
``inspect.signature``; the summaries are one-line condensations of the
native docstrings (tors-documents/src/lib.rs is the full contract).
"""

from __future__ import annotations

import os
from enum import Enum

__version__: str
"""The wheel's version, baked from the crate's Cargo.toml at build time
(release-please bumps Cargo.toml and pyproject.toml together, one
release and two wheels, so the two can never drift apart)."""

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
    six fields). Provenance rides on the entry: ``format`` is the
    resolved format's vocabulary name (the ``to_markdown`` spelling),
    and it — not the entry's shape — tells you why
    ``spans``/``lines``/``images`` may be empty: an office row carries
    declared geometry with no text surfaces, while a blank/scanned PDF
    page looks identical in shape on the pdf lane."""
    @property
    def page_index(self) -> int:
        """The 0-based page (or section/slide/worksheet index) this entry
        describes — mandatory on every entry: the ``pages=``
        normalization reorders."""
    @property
    def format(self) -> str:
        """The resolved format's vocabulary name (the ``to_markdown``
        spelling, the container-aware ``xls``/``xlsx`` case included):
        the entry's provenance marker. It — not the entry's shape —
        tells you why ``spans``/``lines``/``images`` may be empty: an
        office row carries declared geometry with no text surfaces; a
        blank or scanned PDF page looks identical. Every entry of one
        call carries the same name."""
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
        boxes; the entry's ``format`` field is the provenance that tells
        the office case from a blank/scanned PDF page)."""
    @property
    def lines(self) -> list[TextLine]:
        """The unit's text lines, reading order, words nested (PDF pages
        only; office units carry empty lists — the entry's ``format``
        field is the provenance the class docstring points at)."""
    @property
    def images(self) -> list[ImageBox]:
        """The unit's image boxes, paint order (PDF pages only). Never
        visibility-filtered: an image wholly outside the visible box is
        still returned with out-of-range coordinates — the consumer
        clips to the page box."""

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
        unclipped; the page box is the consumer's clip. Coordinates are
        single-precision f32; expect ~1e-4 pt quantization."""
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
        """The run's own content rotation in degrees — non-zero means
        the box is the engine's flattened approximation and MUST NOT be
        used as a highlight box (transposed or mis-anchored at every
        non-zero content rotation, on any page; the line box degrades to
        a square on a rotated page): carried honestly, not re-derived.
        Verified NOT to cover vertical writing (T9, probed against
        pdf_oxide 0.3.78): a WMode-1 run (tategaki — a Type0 font whose
        ``/Encoding`` CMap name ends in ``-V``, e.g. ``/Identity-V``)
        carries ``rotation_degrees == 0`` — the angle is the text
        matrix's, the writing mode plays no part in it — while its box
        is the flattened fiction at its worst: measured, a 24pt 2-glyph
        run at (300, 700) reports ``(300, 700, -48, 24)`` (the vertical
        advance flattened onto x as a NEGATIVE width) where the true ink
        is a column at ``(300, 652, 24, 72)`` with zero overlap, and the
        line surface reports the REVERSED text ("BA" for "AB"). A
        highlight box for text that may be vertical writing cannot be
        validated by this flag alone."""
    @property
    def is_artifact(self) -> bool:
        """Declared ``/Artifact`` marked content — TYPED-FORM-ONLY: true
        only for a typed ``/Artifact <<…>> BDC`` whose ``/Type`` or
        ``/Subtype`` the engine recognizes (Pagination/Header/Footer/
        Watermark/PageNumber, Layout, Page, Background); false for a
        bare ``/Artifact BMC`` (the engine sets no artifact type for it)
        and unrecognized types. So ``is_artifact == false`` is NOT
        evidence of body text: header/footer detection must not rely on
        this flag alone."""

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
        unclipped; the page box is the consumer's clip. Coordinates are
        single-precision f32; expect ~1e-4 pt quantization."""

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
        unclipped; the page box is the consumer's clip. Coordinates are
        single-precision f32; expect ~1e-4 pt quantization."""
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
        consumer's clip. Coordinates are single-precision f32; expect
        ~1e-4 pt quantization. Image boxes are never visibility-filtered (the
        engine's 2 pt drop law is the TEXT walk's): an image wholly
        outside the visible box is still returned with out-of-range
        coordinates — the consumer clips to the page box, and never
        reads "image present" as "image visible"."""
    @property
    def width_px(self) -> int:
        """The image's pixel width."""
    @property
    def height_px(self) -> int:
        """The image's pixel height."""
    @property
    def rotation_degrees(self) -> float:
        """The CTM's rotation angle in degrees, ``atan2``-normalized to
        ``[0, 360)`` — not restricted to right angles (a sheared or
        arbitrarily-rotated CTM reports e.g. ``45.0``; a mirrored CTM
        ``180.0``): the bbox is always the axis-aligned bound of the
        transformed unit square."""
    @property
    def paint_order(self) -> int:
        """Page-local identity/z-order: the image's index among the
        page's painted images, in content-stream order."""

class PageTables:
    """One ``page_tables`` entry: the page's geometry plus its detected
    tables, per-cell boxes in the page's declared render space — the
    native pyclass exposed directly (the ``PageGeometry`` deviation note
    applies to all four tables classes); ``==`` is structural (the five
    fields). Provenance rides on the entry: ``format`` is the resolved
    format's vocabulary name, and it — not the entry's shape — tells you
    why ``tables`` may be empty: on an OFFICE row ``tables: []`` is the
    per-format capability statement (tables are PDF-only in v1), so
    ``format`` marks "unsupported here", not "no tables"; on the pdf
    lane ``tables: []`` with ``tables_refused`` false means "no tables /
    content unreadable" (deliberately indistinguishable)."""
    @property
    def page_index(self) -> int:
        """The 0-based page this entry describes — mandatory on every
        entry: the ``pages=`` normalization reorders."""
    @property
    def format(self) -> str:
        """The resolved format's vocabulary name (the ``to_markdown``
        spelling, the container-aware ``xls``/``xlsx`` case included):
        the entry's provenance marker — on an office row it is what
        marks ``tables: []`` as "unsupported here" rather than "no
        tables". Every entry of one call carries the same name."""
    @property
    def geometry(self) -> PageGeometry | None:
        """The page's declared render box (the ``page_geometry`` entry):
        ``None`` where the page's box cannot be read or declares no
        positive extent — then ``tables`` is empty too (no frame to map
        into). THE READING RULE for refusals: ``tables_refused`` with
        ``geometry.rotation != 0`` is frame incoherence (``/Rotate`` vs
        the drawn content); ``true`` at rotation 0 is the edge cap."""
    @property
    def tables(self) -> list[Table]:
        """The page's detected tables (PDF pages only; office units carry
        empty lists — tables are PDF-only in v1, the entry's ``format``
        field is the marker; a no-tables or content-failed PDF page is
        the same empty shape with ``tables_refused`` false)."""
    @property
    def tables_refused(self) -> bool:
        """TRUE iff this surface deliberately declined to serve a page
        whose inputs indicate possible table content: the
        frame-incoherent ``/Rotate`` classes (the detector's text and
        ruling inputs disagree about the frame there — notably /Rotate
        180 with ruled tables) or the table-primitive edge cap (a page
        with more than the markdown lane's own 1500-primitive bail-out).
        FALSE for no-tables pages, for content-stream failures
        (unknowable — ``tables: []`` stays indistinguishable from a
        no-tables page by design), and on office rows (their marker is
        ``format``). Reading rule: ``true`` + ``geometry.rotation != 0``
        → frame incoherence; ``true`` at rotation 0 → the edge cap."""

class Table:
    """One detected table (``PageTables.tables[]``): the engine's grid;
    ``==`` is structural (the four fields)."""
    @property
    def rows(self) -> list[Row]:
        """The grid's rows, the engine's order (branch-deterministic,
        never re-sorted: a vertical merge emits its merged groups
        bottom-group-first, the cluster path top-first)."""
    @property
    def has_header(self) -> bool:
        """Whether the engine found a header section (boldness/font-size
        heuristics; ALWAYS false on the intersection branch — the pinned
        engine limitation)."""
    @property
    def col_count(self) -> int:
        """The engine's grid column count. Rows may be shorter under
        merges (see ``Row.cells``): a consumer lays cells out by walking
        ``colspan``, never by ``cells[i]``-as-column-i."""
    @property
    def bbox(self) -> tuple[float, float, float, float] | None:
        """The table's box, ``(x, y, w, h)``: points, bottom-left origin
        of the page's render space, y-up — the ENGINE's own union (cell
        boxes on the intersection branch, member spans on the other
        branches, the cluster grid's line-extent × row-extent, the two
        prior boxes after a vertical merge), never re-derived from the
        cells here. Some on every table the engine returns; NOT the
        table's ink extent — it includes empty grid cells. The box is
        unclipped: it can lie outside the page box, and the page box is
        the consumer's clip. Coordinates are single-precision f32;
        expect ~1e-4 pt quantization."""

class Row:
    """One row in a ``Table`` (``Table.rows[]``); ``==`` is structural
    (both fields)."""
    @property
    def cells(self) -> list[Cell]:
        """The row's cells, grid order. ``len`` may be LESS than the
        table's ``col_count`` when a colspan/rowspan merge covers cells
        (the cluster-branch merge): lay cells out by walking
        ``colspan``, never by ``cells[i]``-as-column-i."""
    @property
    def is_header(self) -> bool:
        """Whether the engine judged this row a header row
        (boldness/font-size heuristics; ALWAYS false on the intersection
        branch — the pinned engine limitation)."""

class Cell:
    """One cell in a ``Row`` (``Row.cells[]``): the cell's text, its
    box, and the merge spans; ``==`` is structural (the four fields)."""
    @property
    def text(self) -> str:
        """The cell's text (verbatim; an empty cell is ``""``)."""
    @property
    def bbox(self) -> tuple[float, float, float, float] | None:
        """The cell's box, ``(x, y, w, h)``: points, bottom-left origin
        of the page's render space, y-up. Its regime is the engine's,
        per branch (never re-normalized): intersection-branch cells are
        RULING rectangles (always Some, empty cells included);
        cluster/band/text cells are span unions (None when empty); a
        vertical merge pads merged rows with None cells inside an
        otherwise ruling-rect table. GEOMETRY, not ownership: an empty
        cell's box is declared grid geometry, not a claim on the text
        inside it — do not subtract cell boxes from other surfaces'
        text. Boxes are unclipped — a cell box may lie outside the page
        box (ruling lines are not crop-filtered; the TEXT side is, so a
        grid can lose enough cells to vanish) — the page box is the
        consumer's clip, and "cell box present" is never "cell is
        visible". Coordinates are single-precision f32; expect ~1e-4 pt
        quantization."""
    @property
    def colspan(self) -> int:
        """The column span (the engine's; 1 unless a merge covers this
        cell)."""
    @property
    def rowspan(self) -> int:
        """The row span (the engine's; 1 unless a merge covers this
        cell)."""

class PdfSecurity:
    """The ``pdf_security`` answer: the four security fields the reading
    rule composes (``readable = !encrypted ||
    unlocked_with_supplied_password || owner_password_only``); exposed
    directly (no Python wrapper view: the fields are bools and an
    optional ``PdfPermissions``, nothing to re-type — the
    ``PageGeometry`` deviation note applies); ``==`` is structural (the
    four fields)."""
    @property
    def encrypted(self) -> bool:
        """The document carries an ``/Encrypt`` entry (handler present,
        or the trailer names one) — before and regardless of any
        authentication."""
    @property
    def unlocked_with_supplied_password(self) -> bool:
        """The content was readable under the credentials THIS call
        held: the supplied password's result when one was given, else
        the empty-password open law (``password=None`` MEANS the
        implicit empty attempt, never "no attempt")."""
    @property
    def permissions(self) -> PdfPermissions | None:
        """The declared ``/P`` permission flags: ``None`` when
        unencrypted AND when the ``/Encrypt`` dict fails to initialize
        (malformed shape, unsupported ``/V``/``/R``, a crypto-policy
        denial — three states that collapse into ``encrypted &&
        permissions is None``); ``Some`` otherwise, INCLUDING
        pre-authentication. ``/P`` is advisory (§7.6.3.2): the probe
        reports, callers enforce."""
    @property
    def owner_password_only(self) -> bool:
        """The empty password authenticated — the owner-password-only
        shape, whose content surfaces read fine with no password. An
        inference from the open's own single implicit empty attempt,
        never a credential-kind probe (the engine never exposes
        user-vs-owner): an empty OWNER password with a non-empty USER
        password also reports ``True``. Honest meaning: "the empty
        string opens this document"."""

class PdfPermissions:
    """One decoded ``/P`` permission flag set (``pdf_security``'s
    ``permissions``): the eight named bits of the §7.6.3.2 Table 22
    decode plus the raw integer; exposed directly (the
    ``PageGeometry`` deviation note applies); ``==`` is structural (the
    nine fields). Callers enforcing a policy should prefer the named
    bits and keep ``raw_p`` for provenance."""
    @property
    def print_low_res(self) -> bool:
        """Bit 3: print (low resolution)."""
    @property
    def modify(self) -> bool:
        """Bit 4: modify (other than annotation / form fill)."""
    @property
    def copy(self) -> bool:
        """Bit 5: copy text and graphics."""
    @property
    def annotate(self) -> bool:
        """Bit 6: annotate and form-fill."""
    @property
    def fill_forms(self) -> bool:
        """Bit 9 (rev ≥ 3): fill interactive form fields."""
    @property
    def accessibility(self) -> bool:
        """Bit 10 (rev ≥ 3): extract for accessibility."""
    @property
    def assemble(self) -> bool:
        """Bit 11 (rev ≥ 3): assemble (insert/rotate/delete pages)."""
    @property
    def print_high_res(self) -> bool:
        """Bit 12 (rev ≥ 3): print high resolution."""
    @property
    def raw_p(self) -> int:
        """The raw ``/P`` integer, the engine's ``*i as i32`` view: a
        ``/P`` outside i32 (e.g. 4294967292) reports truncated (−4) —
        lossless ONLY within i32, the two's-complement form the spec
        itself writes."""

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

def pdf_page_labels(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
    backend: Backend | str = ...,
    max_bytes: int | None = None,
) -> list[str | None]:
    """The page label a reader displays, per page (the ISO 32000 §12.4.2
    numbering tree; one GIL-free pass). ``None`` = the page declares no
    label (no tree, or before the tree's first range): the
    fabricate-an-ordinal signal — no label is ever fabricated here. A
    declared label is always ``str``, including the empty string (a
    styleless, prefixless range declares it); a styleless range with a
    prefix yields that same string for every page it covers (constant
    labels); the render arithmetic is u64 (a legal ``/St 4294967294``
    never panics or wraps; an over-u32 ``/St`` carries the engine's own
    parse-time truncation). Rotation/crop irrelevant; a corrupt tree is
    presented as "no labels" (every page ``None`` — the documented
    masking). ``backend=`` is ``"auto"``/``"oxide"`` (the pdf_oxide
    lane; ``"anydoc"`` is a capability refusal: a surface its reader
    never renders; its PDF surface is the ``to_markdown``/``to_text``
    conversion pair); ``password=`` unlocks an encrypted PDF (the door
    governs unchanged: a locked document raises, never ciphertext
    labels); an explicit ``max_bytes=`` binds before the read, ``None``
    leaves the lane unmetered. ⚠ A malformed PDF can crash the process
    (uncatchable native stack exhaustion in the pdf_oxide engine):
    sandbox untrusted PDFs (docs/documents.md, SECURITY.md)."""

def pdf_security(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
    backend: Backend | str = ...,
    max_bytes: int | None = None,
) -> PdfSecurity:
    """The typed encryption/permissions probe: the four security fields
    REPORTED, never a raise on a locked state, never unlocking beyond
    the supplied password (one password per call plus the open's own
    implicit empty attempt; the fail-closed door stays every content
    surface's single encryption authority). Reading rule:
    ``readable = !encrypted || unlocked_with_supplied_password ||
    owner_password_only`` — the trap cell is owner-only + a WRONG
    supplied password (``unlocked`` false there while the document reads
    fine with NO password, and the door raises on the wrong one). Three
    states collapse into ``encrypted && permissions is None`` (malformed
    ``/Encrypt``, unsupported ``/V``/``/R``, a crypto-policy denial):
    "needs a password" is indistinguishable from "this build/policy
    refuses the cipher", and the report is identical with or without a
    supplied password. ``permissions`` is advisory (§7.6.3.2): the probe
    reports, callers enforce. ``backend=`` is ``"auto"``/``"oxide"``
    (the pdf_oxide lane; ``"anydoc"`` is a capability refusal: a report
    its reader never makes; its PDF surface is the
    ``to_markdown``/``to_text`` conversion pair); an explicit
    ``max_bytes=`` binds before the read, ``None`` leaves the lane
    unmetered."""

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
    ``page_geometry`` declares). A non-zero ``rotation_degrees`` means
    the box is the engine's flattened approximation and MUST NOT be used
    as a highlight box (transposed or mis-anchored at every non-zero
    content rotation, on any page). Image boxes are never
    visibility-filtered: an image wholly outside the visible box is
    still returned with out-of-range coordinates — the consumer clips to
    the page box. Office rows serve their declared geometry with empty
    layout lists (PDF-only for layout; the entry's ``format`` field
    names the resolved format — it, not the entry's shape, tells the
    office case from a blank/scanned page); the no-geometry formats answer
    one ``geometry=None`` entry. ``pages=`` is PDF-only (a page index
    cannot name a section): a non-None selection on any other format
    raises the conversion lane's own ``ValueError``; on a PDF it
    normalizes exactly as the conversion pair's (sort + dedup; the
    byte-pinned empty-selection and out-of-range errors) and the output
    is parallel to the NORMALIZED selection, every entry carrying
    ``page_index`` — and every malformed ``pages=``/``max_bytes``
    refusal reports after the capability refusal. ``backend=`` is
    ``"auto"``/``"oxide"`` — identical on
    this surface — while ``"anydoc"`` is refused where geometry is
    obtainable (pdf and the office families); ``password=`` unlocks an
    encrypted PDF (any other format: a clean error; an encrypted OOXML
    package: a typed ``ValueError``); an explicit ``max_bytes=`` binds
    before the read, ``None`` keeps the 32 MiB default ceiling on the
    office leg. Runs GIL-free. ⚠ A malformed PDF can crash the process
    (uncatchable native stack exhaustion in the pdf_oxide engine):
    sandbox untrusted PDFs (docs/documents.md, SECURITY.md)."""

def page_tables(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    format: Format | str | None = ...,
    password: str | None = None,
    backend: Backend | str = ...,
    pages: int | list[int] | tuple[int, int] | None = ...,
    max_bytes: int | None = None,
) -> list[PageTables]:
    """The per-cell table-box probe: per unit of the resolved format's
    geometry lane, the unit's geometry plus — PDF pages only — the
    page's detected tables with per-cell text and boxes, all in the
    page's declared render space (the same frame ``page_layout`` maps:
    points, bottom-left origin, y-up; /Rotate applied; one transform
    owned by the library; boxes are unclipped — a cell box may lie
    outside the page box, the consumer clips). ``Table.bbox`` is the
    engine's own union (Some on every returned table; not the ink
    extent); a cell's box regime is per-branch (intersection: ruling
    rectangles, always Some; cluster/band/text: span unions, None when
    empty; a vertical merge pads with None cells) and is GEOMETRY, not
    ownership — do not subtract cell boxes from other surfaces' text.
    The entry's ``tables_refused`` flag: TRUE iff the surface
    deliberately declined a page whose inputs indicate possible table
    content — the frame-incoherent ``/Rotate`` classes
    (``geometry.rotation != 0`` names them) or the table-primitive edge
    cap (rotation 0); FALSE for no-tables pages, content failures
    (unknowable, indistinguishable from no-tables), and office rows
    (tables are PDF-only in v1: an office row's ``tables: []`` is the
    per-format capability statement, the entry's ``format`` field the
    marker). The table set is the public spatial detector's answer —
    ``to_markdown`` renders a different (private, consolidating)
    pipeline, so the sets can differ on any PDF; and content inside a
    detected table's area may appear in BOTH this surface and
    ``page_layout`` (no de-dup, no identity join, no subtraction by
    cell bbox). The spine, ``pages=`` (PDF-only, byte-pinned errors,
    output parallel to the NORMALIZED selection with ``page_index`` on
    every entry), ``backend=``/``password=``/``max_bytes=`` and the
    refusal order are ``page_layout``'s exactly (the encrypted door
    precedes the ``pages=`` validation). Runs GIL-free. ⚠ A malformed
    PDF can crash the process (uncatchable native stack exhaustion in
    the pdf_oxide engine): sandbox untrusted PDFs (docs/documents.md,
    SECURITY.md)."""

class DocumentStructure:
    """The ``document_structure`` report: the resolved format's vocabulary
    name and exactly one per-format payload — narrow with
    ``isinstance(report.payload, ...)`` (the compile-time one-of is the
    core's enum; a ``format`` string cannot narrow it statically). ``==``
    is structural (both fields)."""
    @property
    def format(self) -> str:
        """The resolved format's vocabulary name (the ``to_markdown``
        spelling, the container-aware ``xls``/``xlsx`` case included):
        provenance, and the same spelling the refusals use."""
    @property
    def payload(self) -> DocxStructure | XlsxStructure | HtmlStructure:
        """The per-format structure: exactly one of the three payload
        classes by construction. A fresh native object per access,
        structurally equal to its siblings."""

class DocxStructure:
    """The docx payload: the body's blocks in document order from the
    office_oxide RAW layer (headings via the ladder, effective-property
    list state, block text, tables, links, real pictures), the section
    partition, the note counts, and the unreadable parts. Entries carry
    container anchors, never boxes. ``==`` is structural (all eight
    fields)."""
    @property
    def blocks(self) -> list[DocxBlock]:
        """The body's blocks, body order, one entry per body element."""
    @property
    def sections(self) -> int:
        """The body partition law: ``section_breaks``' length + 1 — NOT
        the declared SectionProperties count (a document with no
        body-level sectPr is ONE section; declared sizes belong to
        ``page_geometry``)."""
    @property
    def section_breaks(self) -> list[int]:
        """``body.section_breaks`` verbatim: section i covers body
        elements ``[prev_break, section_breaks[i])`` — anchor math for
        ``block.section``."""
    @property
    def footnotes(self) -> int:
        """Footnote body count (v1 keeps note BODIES out; the counts are
        the cheap part)."""
    @property
    def endnotes(self) -> int:
        """Endnote body count."""
    @property
    def comments(self) -> int:
        """Comment body count."""
    @property
    def unreadable_parts(self) -> list[str]:
        """Header/footer/note parts that exist but could not be read or
        parsed, verbatim (the report-the-loss law — content IS partial
        here, the loss on record)."""
    @property
    def entries_truncated(self) -> bool:
        """True when the blocks stopped at ``MAX_STRUCTURE_ENTRIES``
        (the 200,000-entry marshalling bound — never a refusal; the
        metadata above stays complete)."""

class DocxBlock:
    """One docx body block; ``==`` is structural (all fourteen fields)."""
    @property
    def index(self) -> int:
        """Position in ``body.elements`` (the stable container anchor)."""
    @property
    def section(self) -> int:
        """The covering section (the count of ``section_breaks <=
        index``)."""
    @property
    def kind(self) -> str:
        """``"heading" | "list_item" | "paragraph" | "table"`` — heading
        wins over list (a numbered heading is a heading carrying
        numPr)."""
    @property
    def text(self) -> str:
        """The block's visible text: runs joined, hidden runs resolved
        through the style chain, breaks/tabs verbatim, text-box prose
        recursed, form-field display text surfaced, note-ref marks
        skipped; ``""`` for table blocks (the cells carry the text)."""
    @property
    def level(self) -> int | None:
        """Headings only: the markdown depth, ``clamp(outline + 1,
        1..=6)`` (every markdown renderer prints ``######`` at depth
        7-9; the raw depth is ``outline_level``)."""
    @property
    def outline_level(self) -> int | None:
        """Headings only: the ladder's raw 0-based outline level. The
        ladder: direct ``w:outlineLvl`` -> the stylesheet's basedOn
        chain -> the ``Heading1..9`` style-ID convention -> the
        ``heading N`` style-NAME convention — the convention rung is a
        documented SUPERSET of the ``backend="oxide"`` markdown's
        two-rung ladder."""
    @property
    def style_id(self) -> str | None:
        """The raw ``w:pStyle`` string ("Heading1", "Caption", ...) —
        the document's own name."""
    @property
    def num_id(self) -> int | None:
        """The numbering instance, whenever the EFFECTIVE properties
        carry numPr with numId != 0 — style-carried numPr included
        (Word's "List Bullet"/"List Number" styles hold it in the
        style), a numbered heading's included. Inside table cells the
        table-style LAYER is unreachable (crate-private), so cell-level
        list state misses table-style numPr — disclosed."""
    @property
    def ilvl(self) -> int | None:
        """The numbering level (0-based)."""
    @property
    def list_format(self) -> str | None:
        """The resolved number format's own name ("decimal", "bullet",
        "lower_letter", ...): the file's vocabulary, never a rendered
        marker."""
    @property
    def start_number(self) -> int | None:
        """The level's effective start (startOverride-aware) — the
        level's own start, NOT a rendered ordinal (the markdown lane
        renumbers sequentially)."""
    @property
    def table(self) -> DocxTable | None:
        """``kind == "table"`` only: the grid."""
    @property
    def links(self) -> list[DocxLink]:
        """The block's hyperlinks, in order."""
    @property
    def images(self) -> list[DocxImage]:
        """The block's real pictures, in order (a DrawingML shape with
        no relationship id is filtered)."""

class DocxTable:
    """One docx table's grid; ``==`` is structural (all five fields)."""
    @property
    def rows(self) -> int:
        """The grid's row count."""
    @property
    def cols(self) -> int:
        """The grid's column count (``grid.len()``)."""
    @property
    def col_widths_twips(self) -> list[int]:
        """The ``w:tblGrid`` column widths, twips, verbatim."""
    @property
    def cells(self) -> list[DocxCell]:
        """The present cells, row-major (a vMerge-continue cell is
        emitted as its own cell, inside the Restart span it continues)."""
    @property
    def caption(self) -> str | None:
        """The ``w:tblCaption`` accessibility caption (the raw layer's
        only caption field)."""

class DocxCell:
    """One docx table cell; ``==`` is structural (all five fields)."""
    @property
    def row(self) -> int:
        """0-based row."""
    @property
    def col(self) -> int:
        """0-based starting column (gridSpan accumulation)."""
    @property
    def text(self) -> str:
        """The cell's text (the same block walk as body blocks,
        trimmed)."""
    @property
    def row_span(self) -> int:
        """The row span: a vMerge Restart plus the same-column Continue
        cells below it; 1 otherwise."""
    @property
    def col_span(self) -> int:
        """The column span (``w:gridSpan``, clamped 1..=1000)."""

class DocxLink:
    """One docx hyperlink; ``==`` is structural (all five fields)."""
    @property
    def text(self) -> str:
        """The hyperlink's runs' text."""
    @property
    def href(self) -> str | None:
        """The external URL (rels-resolved); ``None`` for an internal
        link."""
    @property
    def anchor(self) -> str | None:
        """The internal bookmark name; ``None`` for an external link."""
    @property
    def fragment(self) -> str | None:
        """The URI fragment when an external link also declares
        ``w:anchor``."""
    @property
    def tooltip(self) -> str | None:
        """The screen-tip tooltip, when declared."""

class DocxImage:
    """One docx picture (a drawing with a relationship id; bytes never
    cross this surface); ``==`` is structural (all seven fields)."""
    @property
    def relationship_id(self) -> str:
        """The relationship id (``a:blip/@r:embed``, or a linked
        picture's ``@r:link``)."""
    @property
    def file_ext(self) -> str | None:
        """The image part's extension from the package's map ("png",
        ...); ``None`` when the part is not in the map."""
    @property
    def description(self) -> str | None:
        """The alt text (``wp:docPr/@descr``)."""
    @property
    def width_pt(self) -> float | None:
        """The declared width, points (EMU / 12700.0); ``None`` when the
        declared extent is not positive (an undeclared extent is 0)."""
    @property
    def height_pt(self) -> float | None:
        """The declared height, points (the same law)."""
    @property
    def inline(self) -> bool:
        """True for an inline drawing, false for a floating anchor."""
    @property
    def decorative(self) -> bool:
        """The ``adec:decorative`` flag (the alt-less placeholder
        pictures assistive technology skips)."""

class XlsxStructure:
    """The xlsx payload: the worksheets (kind-filtered from the raw
    layer's own list), the chart/dialog sheet names, defined names, the
    unreadable sheets. ``==`` is structural (all six fields)."""
    @property
    def sheets(self) -> list[XlsxSheet]:
        """The worksheets, sheet order, hidden and macro sheets served —
        built from the raw layer's OWN worksheet list (never zip'd
        against ``workbook.sheets``, which also names chartsheets and
        partless sheets)."""
    @property
    def chartsheets(self) -> list[str]:
        """The workbook's chart/dialog sheet names (kind-tagged entries
        with no cells in the raw layer's list)."""
    @property
    def defined_names(self) -> list[XlsxDefinedName]:
        """The workbook-level defined names."""
    @property
    def date1904(self) -> bool:
        """The 1904 date system flag."""
    @property
    def unreadable_sheets(self) -> list[str]:
        """Names of sheets whose part could not be parsed (the intact
        sheets are served; the loss is on record)."""
    @property
    def entries_truncated(self) -> bool:
        """True when the cells stopped at ``MAX_STRUCTURE_ENTRIES`` (the
        sheet/range/merge/name metadata stays complete)."""

class XlsxSheet:
    """One xlsx worksheet; ``==`` is structural (all ten fields)."""
    @property
    def name(self) -> str:
        """The sheet's display name."""
    @property
    def state(self) -> str:
        """``"visible" | "hidden" | "veryHidden"``."""
    @property
    def used_range(self) -> str | None:
        """The used range, COMPUTED from the parsed cells — min/max over
        every present cell, styled-but-empty included — rendered
        ``"A1:D10"``. NOT the declared ``<dimension>`` (attacker-
        controlled, frequently absent); ``None`` for a sheet with no
        cells."""
    @property
    def declared_dimension(self) -> str | None:
        """The declared ``<dimension>`` string, verbatim (provenance,
        never the used range's source)."""
    @property
    def max_row(self) -> int | None:
        """The used range's max row index, 0-based; ``None`` for an
        empty sheet."""
    @property
    def max_col(self) -> int | None:
        """The used range's max column index, 0-based (the same law)."""
    @property
    def cells(self) -> list[XlsxCell]:
        """The present cells, row-major, sparse."""
    @property
    def merged_cells(self) -> list[str]:
        """The merged ranges as ref strings ("B2:C3"), verbatim."""
    @property
    def hyperlinks(self) -> list[XlsxHyperlink]:
        """The sheet's hyperlinks, part order."""
    @property
    def comments(self) -> list[XlsxComment]:
        """The sheet's cell comments, document order."""

class XlsxCell:
    """One xlsx cell: the value AS THE FILE STATES IT, discriminated by
    ``type``, never coerced; ``==`` is structural (all seven fields)."""
    @property
    def ref(self) -> str:
        """The cell reference ("B3")."""
    @property
    def row(self) -> int:
        """0-based row."""
    @property
    def col(self) -> int:
        """0-based column."""
    @property
    def type(self) -> str:
        """``"number" | "string" | "boolean" | "error" | "date" |
        "empty"`` — ``date`` is ONLY the ``t="d"`` ISO-8601 branch (a
        serial number with a date number-format stays ``"number"``: the
        seam cannot decode the number-format stylesheet)."""
    @property
    def value(self) -> bool | float | str | None:
        """The value by ``type``: number -> float, boolean -> bool,
        string/error -> str, date -> the ISO string, empty -> None.
        Shared strings are DEREFERENCED to their FULL text (no
        markdown-lane 32,768-char truncation — a disclosed divergence
        from ``to_markdown``'s DoS guard)."""
    @property
    def formula(self) -> str | None:
        """The ``<f>`` formula text, verbatim."""
    @property
    def has_cached_value(self) -> bool:
        """Formula cells: whether a cached value rode the file (a
        formula whose cached ``<v>`` is absent reports False; a
        ``t="e"`` cell with no ``<v>`` parses to the empty error string
        and reports True)."""

class XlsxDefinedName:
    """One defined name; ``==`` is structural (all five fields)."""
    @property
    def name(self) -> str:
        """The name."""
    @property
    def value(self) -> str:
        """The formula/reference value, verbatim."""
    @property
    def local_sheet_id(self) -> int | None:
        """The sheet the name is scoped to, 0-based into the WORKBOOK's
        own sheet list (chartsheets included), verbatim; ``None`` for a
        workbook-level name."""
    @property
    def local_sheet_name(self) -> str | None:
        """That sheet's display name, resolved from the workbook's sheet
        list when in range."""
    @property
    def hidden(self) -> bool:
        """The name's hidden flag."""

class XlsxHyperlink:
    """One xlsx sheet hyperlink; ``==`` is structural (all four
    fields)."""
    @property
    def cell_ref(self) -> str:
        """The cell the link sits on."""
    @property
    def href(self) -> str | None:
        """The external URL; ``None`` for an internal location."""
    @property
    def anchor(self) -> str | None:
        """The internal location (sheet/cell); ``None`` for a URL."""
    @property
    def tooltip(self) -> str | None:
        """The tooltip, when declared."""

class XlsxComment:
    """One xlsx cell comment; ``==`` is structural (all three fields)."""
    @property
    def cell_ref(self) -> str:
        """The cell the comment is attached to."""
    @property
    def author(self) -> str | None:
        """The comment's author, when the part names one."""
    @property
    def text(self) -> str:
        """The comment's text."""

class HtmlStructure:
    """The html payload: the conversion lane's own parse run once with
    the structure collector on — the flat node array, the conversion
    output discarded (no second tables list: the Table nodes carry the
    grids). ``==`` is structural (both fields)."""
    @property
    def nodes(self) -> list[HtmlNode]:
        """The nodes, reading order, parents before children (the
        auto-built Group tree IS the heading tree)."""
    @property
    def entries_truncated(self) -> bool:
        """True when nodes stopped at ``MAX_STRUCTURE_ENTRIES``: the kept
        prefix stays index-sound (parents are always earlier) and every
        kept node's children is filtered to kept indices."""

class HtmlNode:
    """One html node (the NINE reachable kinds — "heading", "paragraph",
    "list", "list_item", "table", "image", "code", "quote", "group"; an
    unreachable variant is a typed refusal, never a silent kind=None);
    ``==`` is structural (all fourteen fields)."""
    @property
    def id(self) -> str:
        """The crate's deterministic node id."""
    @property
    def parent(self) -> int | None:
        """The parent node's index (``None`` for roots) — always
        earlier."""
    @property
    def children(self) -> list[int]:
        """The children's indices, reading order."""
    @property
    def attributes(self) -> dict[str, str] | None:
        """The crate's per-node attribute map — headings carry
        ``{"id": ...}`` when the source heading had one; image nodes
        carry ``None`` (the collector drops img attributes: declared
        image dimensions are NOT served — the disclosed v1 gap, the
        visitor-feature path the v1.x completion)."""
    @property
    def kind(self) -> str:
        """The node kind (the nine reachable ones)."""
    @property
    def text(self) -> str | None:
        """The node's text (heading/paragraph/list_item/code)."""
    @property
    def level(self) -> int | None:
        """Heading level 1-6 (a Group: the level that created it)."""
    @property
    def ordered(self) -> bool | None:
        """List: whether ordered."""
    @property
    def language(self) -> str | None:
        """Code: the language, when declared."""
    @property
    def label(self) -> str | None:
        """Group: the section label — the creating heading's text (the
        crate's ``label`` and ``heading_text`` are the same string)."""
    @property
    def src(self) -> str | None:
        """Image: the src URL."""
    @property
    def alt(self) -> str | None:
        """Image: the alt text."""
    @property
    def grid(self) -> HtmlGrid | None:
        """Table: the grid. Nested tables: the inner grid is ALSO
        collected (its own Table node, the inner table closing first)
        while the outer cell carries the flattened text — the observed
        collector behavior, disclosed."""
    @property
    def annotations(self) -> list[HtmlAnnotation]:
        """Inline annotations, byte offsets into ``text``. A link's
        enclosing section is the nearest GROUP ancestor (walk ``parent``;
        headings have no children — a heading CREATES a Group sibling,
        and a link on a heading lives on the Heading node itself)."""

class HtmlGrid:
    """One html table grid; ``==`` is structural (all three fields)."""
    @property
    def rows(self) -> int:
        """The grid's row count."""
    @property
    def cols(self) -> int:
        """The grid's column count."""
    @property
    def cells(self) -> list[HtmlGridCell]:
        """The origin cells, ordered by (row, col), sparse under spans.
        A table's first bare ``<tr>`` (no thead/tbody wrapper) is the
        collector's own header inference."""

class HtmlGridCell:
    """One html grid cell; ``==`` is structural (all six fields)."""
    @property
    def content(self) -> str:
        """The cell's text."""
    @property
    def row(self) -> int:
        """0-based row."""
    @property
    def col(self) -> int:
        """0-based column."""
    @property
    def row_span(self) -> int:
        """The row span (1 unless a merge covers)."""
    @property
    def col_span(self) -> int:
        """The column span (1 unless a merge covers)."""
    @property
    def is_header(self) -> bool:
        """Whether this is a header cell (a ``<th>``, or a row under a
        ``<thead>``/first-bare-row)."""

class HtmlAnnotation:
    """One inline annotation on an html node's text; ``==`` is
    structural (all five fields)."""
    @property
    def kind(self) -> str:
        """``"link" | "bold" | "italic" | "underline" |
        "strikethrough" | "code" | "subscript" | "superscript" |
        "highlight"``."""
    @property
    def start(self) -> int:
        """Start byte offset into the node's text (inclusive)."""
    @property
    def end(self) -> int:
        """End byte offset (exclusive)."""
    @property
    def url(self) -> str | None:
        """Link: the href, verbatim (no decoding, no resolution)."""
    @property
    def title(self) -> str | None:
        """Link: the ``title`` attribute, when declared."""

def document_structure(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    format: Format | str | None = ...,
    backend: Backend | str = ...,
    max_bytes: int | None = ...,
) -> DocumentStructure:
    """The structure surface (issues #209/#210/#211): one report whose
    ``format`` discriminates and whose ``payload`` is exactly one of
    :class:`DocxStructure` / :class:`XlsxStructure` / :class:`HtmlStructure`
    (isinstance narrowing). docx: the raw layer's blocks (headings via the
    ladder — ``level`` clamped 1..=6 plus the raw ``outline_level`` —
    effective-property list state, block text, table grids with
    gridSpan/vMerge spans, links, real pictures, the section partition,
    note counts, unreadable_parts); xlsx: the worksheets (hidden/macro
    served, chart/dialog tabs named), the used range COMPUTED from the
    parsed cells, values as the file states them (``type``-discriminated,
    shared strings dereferenced to full text, ``t="d"`` the only date
    shape), formulas text+cache both labeled, merges, hyperlinks,
    comments, defined names; html: the conversion lane's own parse with
    the structure collector on (the nine reachable node kinds, heading id
    attributes, link annotations with byte ranges, table grids with
    spans/header flags) — the conversion output discarded. Entries carry
    container anchors, never boxes. Every refused format (pdf,
    doc/xls/ppt, pptx, rtf/odt/ods/odp/epub/csv) refuses on EVERY backend
    with a format-level ValueError fired before any read (pdf's answer is
    already page_layout/page_tables; "no structure surface", never
    "unreadable"); the served office rows refuse ``"anydoc"`` as a
    capability; html's forced backends are conversion's own
    UnsupportedBackend pair; an OLE container under the Excel kind
    refuses with the legacy-xls-or-encrypted-xlsx union message.
    ``format=`` is trusted and never re-sniffed (``sniff()`` is the
    fix-it). There is NO ``password=``: no served format accepts one. An
    explicit ``max_bytes=`` binds before the read; ``None`` keeps the
    32 MiB default ceiling on every served row; the 200,000-entry payload
    cap truncates with ``entries_truncated`` (metadata stays complete),
    never refusing. The whole read+resolve+structure pass runs
    GIL-free."""
