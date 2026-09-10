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
    ``backend="auto"`` and ``backend=Backend.AUTO`` are the same call."""

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
    used. The whole read+sniff+convert pass runs GIL-free."""

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
    engine. Returns ``(Format, text)``."""

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
    unmetered. Runs GIL-free."""

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
    lane unmetered."""

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
    lane unmetered."""
