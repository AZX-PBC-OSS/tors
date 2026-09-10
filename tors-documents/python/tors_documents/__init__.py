"""The ``tors.documents`` payload: document-format extraction to GitHub-Flavored
Markdown or plain text, GIL-free.

This package is the second wheel (``tors-documents``, installed by
``pip install tors[documents]``); the base ``tors`` wheel re-exports it as
``tors.documents`` with a helpful install hint when this payload is absent.
Import through ``tors.documents``: this package's name is its distribution
name, not the surface callers should target.

The engines, chosen per family by head-to-head measurement (see
``tors-core``'s ``documents_impl`` docs): pdf_oxide for PDF (two-column
reading order, ``[text](uri)`` link annotations, heading detection), anydoc
for the office/rtf/odt/epub/csv families, and the HTML lane's own engine.
Every function releases the GIL for its whole native pass.

The typed surface: ``Backend``/``Format``/``PageKind`` are ``str`` enums:
each member is its accepted string, so every plain-string call keeps
working and the enums cost nothing at the boundary (the Rust validator
remains the authority; plain strings stay accepted so the vocabulary can
grow without enum churn in caller code). ``to_markdown``/``to_text`` return
the resolved format as a ``Format`` member, ``sniff`` returns
``Format | None``, and ``pdf_classify`` returns a typed view whose
derivation rules (``has_text``/``image_only``) live in the native getter,
exactly once.
"""

from __future__ import annotations

import os
from enum import Enum

from tors_documents._tors_documents import NeedsOcrError
from tors_documents._tors_documents import PdfClassification as _NativePdfClassification

# The wheel's version, baked from the crate's Cargo.toml at build time
# (release-please bumps Cargo.toml and pyproject.toml together, one
# release and two wheels, so the native constant is the wheel version; the
# importlib.metadata alternative costs a site-packages scan on every
# import and needs a static fallback pin that drifts stale in source
# checkouts).
from tors_documents._tors_documents import __version__ as __version__
from tors_documents._tors_documents import pdf_classify as _native_pdf_classify
from tors_documents._tors_documents import pdf_extract as _native_pdf_extract
from tors_documents._tors_documents import pdf_link_uris as _native_pdf_link_uris
from tors_documents._tors_documents import (
    pdf_page_count as _native_pdf_page_count,
)
from tors_documents._tors_documents import sniff as _native_sniff
from tors_documents._tors_documents import to_markdown as _native_to_markdown
from tors_documents._tors_documents import to_text as _native_to_text

__all__ = [
    "__version__",
    "Backend",
    "Format",
    "NeedsOcrError",
    "PageKind",
    "PdfClassification",
    "pdf_classify",
    "pdf_extract",
    "pdf_link_uris",
    "pdf_page_count",
    "sniff",
    "to_markdown",
    "to_text",
]


class Backend(str, Enum):
    """The engine selector. ``AUTO`` routes by the measured table (PDF to
    pdf_oxide, HTML to its own engine, the office/text families to anydoc);
    ``OXIDE`` forces the oxide family (pdf_oxide for PDF, office_oxide for
    the OOXML/legacy office formats); ``ANYDOC`` forces anydoc. A forced
    backend raises on a format it cannot read, never a silent fallback.
    Members are the accepted strings, so ``backend="auto"`` and
    ``backend=Backend.AUTO`` are the same call."""

    AUTO = "auto"
    OXIDE = "oxide"
    ANYDOC = "anydoc"


class Format(str, Enum):
    """The format vocabulary: the ``format=`` names, the resolved-format
    values, and ``sniff``'s answers. Members are their strings, so
    ``format="docx"`` and ``format=Format.DOCX`` are the same call, and
    plain strings the Rust validator accepts (container variants like
    ``docm``/``xlsm``/``ppsx``) keep working without enum churn. The set is
    maintained in lockstep with the engine's vocabulary; a resolved name
    outside it is a bug, surfaced as ``ValueError``."""

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


# `pages=` is annotated as the exact shapes the native validator accepts
# (int, list of ints, 2-tuple range); the previous `Sequence[int]` promised
# shapes the validator refuses (a `range`, a generator): an annotation that
# over-promises is a bug the type checker enforces against the caller.
# Generic-iterable acceptance is a proposed extension, not a current fact.


def _coerce_path(path: str | os.PathLike[str] | None) -> str | None:
    """os.fspath with the surface's own refusal convention: a ``path`` that
    is neither str nor PathLike is a TypeError naming ``path`` (and
    repr'ing the value); os.fspath's own bare ``expected str, bytes or
    os.PathLike object, not int`` names neither the argument nor the
    function, against the every-message-names-the-argument convention the
    native layer already holds (probed once: ``to_markdown(123)`` died
    in the wrapper before the native parse_path could refuse it properly)."""
    if path is None:
        return None
    try:
        return os.fspath(path)
    except TypeError:
        raise TypeError(f"path must be a str or os.PathLike, not {path!r}") from None


def to_markdown(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    format: Format | str | None = None,
    backend: Backend | str = Backend.AUTO,
    pages: int | list[int] | tuple[int, int] | None = None,
    password: str | None = None,
    max_bytes: int | None = None,
) -> tuple[Format, str]:
    """Convert any working-format document to GFM markdown on the
    measured-best engine (the full contract is the native function's
    docstring). Exactly one of ``path`` (a file, the only positional, a
    regular file; FIFOs/devices/sockets are refused before the read)
    or ``data`` (the document's bytes: the in-memory caller's entry, no
    temp-file roundtrip; no name to consult, so resolution rests on
    ``format=`` and the content markers alone). ``password=`` unlocks an
    encrypted PDF (the PDF kinds only; a password on any other format is
    a clean error); an explicit ``max_bytes=`` is binding on every engine
    lane, checked before a byte is read or copied, while ``None`` keeps
    the 32 MiB default (enforced after the read, on the anydoc/oxide
    lanes only). Returns ``(Format, markdown)``: the format the
    conversion actually used. The whole read+sniff+convert pass runs
    GIL-free."""
    resolved, output = _native_to_markdown(
        _coerce_path(path),
        data,
        format,
        backend,
        pages,
        password,
        max_bytes,
    )
    return Format(resolved), output


def to_text(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    format: Format | str | None = None,
    backend: Backend | str = Backend.AUTO,
    pages: int | list[int] | tuple[int, int] | None = None,
    password: str | None = None,
    max_bytes: int | None = None,
) -> tuple[Format, str]:
    """The same conversion, routing, source, ``pages=``, ``password=``,
    and ``max_bytes=`` semantics as ``to_markdown``, with the markdown
    normalized to plain text: one text shape for every format and
    engine. Returns ``(Format, text)``."""
    resolved, text = _native_to_text(
        _coerce_path(path),
        data,
        format,
        backend,
        pages,
        password,
        max_bytes,
    )
    return Format(resolved), text


def sniff(data: bytes) -> Format | None:
    """The content-marker format detector over bytes alone: what
    ``to_markdown`` would resolve ``data`` to with no path and no
    extension. ``None`` = the content names no format. Not a bounded
    marker scan: anydoc's detect parses the package containers; a
    120 KiB zip measured 267 MiB peak RSS to answer docx (the full cost
    is the native docstring's)."""
    name = _native_sniff(data)
    return None if name is None else Format(name)


def pdf_classify(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
    backend: Backend | str = Backend.AUTO,
    max_bytes: int | None = None,
) -> PdfClassification:
    """The cheap text-vs-image preflight (the full contract is the native
    function's docstring), as the typed view. ``path`` or ``data=``;
    ``password=`` for an encrypted PDF. ``backend=`` is ``"auto"``/``"oxide"``
    (the pdf_oxide lane, byte-identical; ``"anydoc"`` is refused: its PDF
    surface is the ``to_markdown``/``to_text`` conversion pair); an explicit
    ``max_bytes=`` binds before the read, ``None`` leaves the pdf lane
    unmetered."""
    return PdfClassification(
        _native_pdf_classify(_coerce_path(path), data, password, backend, max_bytes)
    )


def pdf_extract(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
    backend: Backend | str = Backend.AUTO,
    max_bytes: int | None = None,
) -> tuple[list[str], str]:
    """Read a PDF (``path`` or ``data=``) and return
    ``(per_page_plain_text, markdown)``: one GIL-free pass over one open
    document (the full contract is the native function's docstring);
    ``password=`` for an encrypted PDF. ``backend=`` is ``"auto"``/``"oxide"``
    (the pdf_oxide lane, byte-identical; ``"anydoc"`` is refused: the
    per-page probe is the OCR-routing signal and anydoc has none; its PDF
    surface is the ``to_markdown``/``to_text`` conversion pair); an
    explicit ``max_bytes=`` binds before the read, ``None`` leaves the pdf
    lane unmetered."""
    return _native_pdf_extract(_coerce_path(path), data, password, backend, max_bytes)


def pdf_link_uris(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
    backend: Backend | str = Backend.AUTO,
    max_bytes: int | None = None,
) -> list[list[str]]:
    """The ``/Annots`` link walk (``path`` or ``data=``): for every page,
    the URIs of its link annotations, in annotation order: the raw
    navigation surface no text rendering carries (the full contract is the
    native function's docstring). One GIL-free pass; ``password=`` for an
    encrypted PDF. ``backend=`` is ``"auto"``/``"oxide"`` (the pdf_oxide
    lane, byte-identical; ``"anydoc"`` is refused: it has no annotation
    surface; its PDF surface is the ``to_markdown``/``to_text``
    conversion pair); an explicit ``max_bytes=`` binds before the read,
    ``None`` leaves the pdf lane unmetered."""
    return _native_pdf_link_uris(_coerce_path(path), data, password, backend, max_bytes)


def pdf_page_count(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
    backend: Backend | str = Backend.AUTO,
    max_bytes: int | None = None,
) -> int:
    """The page tree and nothing else (``path`` or ``data=``), one
    GIL-free pass; ``password=`` for an encrypted PDF. ``backend=`` is
    ``"auto"``/``"oxide"`` (the pdf_oxide lane, byte-identical;
    ``"anydoc"`` is refused: its reader reports a count only inside the
    NeedsOcr refusal; its PDF surface is the ``to_markdown``/``to_text``
    conversion pair); an explicit ``max_bytes=`` binds before the read,
    ``None`` leaves the pdf lane unmetered."""
    return _native_pdf_page_count(_coerce_path(path), data, password, backend, max_bytes)


class PdfClassification:
    """The ``pdf_classify`` result, typed: the native preflight's answer
    with ``page_kinds`` as ``PageKind`` members. The derivation rules
    (``has_text``/``image_only``) live in the native getters, exactly
    once; this view adds only typing, never a second copy of a rule."""

    __slots__ = ("_inner",)

    def __init__(self, inner: _NativePdfClassification) -> None:
        self._inner = inner

    @property
    def page_count(self) -> int:
        """Pages in the document."""
        return self._inner.page_count

    @property
    def page_kinds(self) -> list[PageKind]:
        """Every page's verdict, page order."""
        return [PageKind(kind) for kind in self._inner.page_kinds]

    @property
    def pages_needing_ocr(self) -> list[int]:
        """The 0-based indices of image-only pages (blank pages are
        deliberately not listed; they are ``PageKind.EMPTY``)."""
        return list(self._inner.pages_needing_ocr)

    @property
    def has_text(self) -> bool:
        """At least one page carries a text layer."""
        return self._inner.has_text

    @property
    def image_only(self) -> bool:
        """Every page is image-only: route the document to an OCR stage."""
        return self._inner.image_only

    def __repr__(self) -> str:
        return (
            f"PdfClassification(page_count={self.page_count}, "
            f"page_kinds={self.page_kinds}, pages_needing_ocr={self.pages_needing_ocr})"
        )
