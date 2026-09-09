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
(release-please bumps Cargo.toml and pyproject.toml together — one
release, two wheels — so the two can never drift apart)."""

class Backend(str, Enum):
    """The engine selector: ``AUTO`` routes by the measured table,
    ``OXIDE``/``ANYDOC`` force one engine and raise on a format it cannot
    read — never a silent fallback. Members ARE their accepted strings, so
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
    """The 0-based page indices needing OCR — the same convention as
    ``pages=`` and ``pages_needing_ocr``."""
    page_count: int

class PdfClassification:
    """The ``pdf_classify`` result: the text-vs-image preflight's answer,
    with the routing rules (``has_text``/``image_only``) derived in the
    native getters — exactly once."""
    @property
    def page_count(self) -> int:
        """Pages in the document."""
    @property
    def page_kinds(self) -> list[PageKind]:
        """Every page's verdict, page order."""
    @property
    def pages_needing_ocr(self) -> list[int]:
        """The 0-based indices of image-only pages (blank pages are
        deliberately not listed — they are ``PageKind.EMPTY``)."""
    @property
    def has_text(self) -> bool:
        """At least one page carries a text layer — extraction will yield
        something."""
    @property
    def image_only(self) -> bool:
        """Every page is image-only — nothing local can read this
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
    Markdown on the measured-best engine for its format. ``password=``
    unlocks an encrypted PDF (the PDF kinds only); ``max_bytes=``
    overrides the anydoc lane's 32 MiB input ceiling. Returns
    ``(Format, markdown)`` — the format the conversion actually used. The
    whole read+sniff+convert pass runs GIL-free."""

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
    normalized to plain text — one text shape for every format and
    engine. Returns ``(Format, text)``."""

def sniff(data: bytes) -> Format | None:
    """The content-marker format detector over bytes alone: what
    ``to_markdown`` would resolve ``data`` to with no path and no
    extension. ``None`` = not a document at all, or a guard-refused shape
    such as JSON-lines (record lines the delimiter witness alone would
    claim as csv). Never opens a parser; runs GIL-free."""

def pdf_classify(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
) -> PdfClassification:
    """The cheap text-vs-image preflight over an open PDF — no content
    conversion, no OCR, no rasterization. Encrypted documents fail closed
    (``ValueError``). Runs GIL-free."""

def pdf_extract(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
) -> tuple[list[str], str]:
    """Open a PDF and return ``(per_page_plain_text, markdown)`` — one
    GIL-free pass over one open document (the parse is paid once for both
    outputs). An image-only page is an empty string, not an error; routing
    decisions are the caller's."""

def pdf_page_count(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
) -> int:
    """The page tree and nothing else: open + count, no content
    extraction, one GIL-free pass."""

def pdf_link_uris(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
) -> list[list[str]]:
    """The ``/Annots`` link walk: for every page, the URIs of its link
    annotations, in annotation order — the raw navigation surface no text
    rendering carries. Verbatim (never deduped); URI actions only."""
