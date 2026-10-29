"""The documents-extraction surface's base-wheel shim.

The compiled surface ships in the separate `tors-documents` wheel
(`tors[documents]`, a real PyPI extra): the base `tors` wheel carries this
typed shim, which re-exports the payload's API when it is installed and
raises the install hint when it is not: the base extension module stays
engine-free, which is why the split exists.

Install: ``pip install tors[documents]``, or the payload directly,
``pip install tors-documents`` (kept in version lockstep with ``tors``).
"""

from __future__ import annotations

__all__ = [
    "Backend",
    "Cell",
    "DocumentStructure",
    "DocxBlock",
    "DocxCell",
    "DocxImage",
    "DocxLink",
    "DocxStructure",
    "DocxTable",
    "Format",
    "HtmlAnnotation",
    "HtmlGrid",
    "HtmlGridCell",
    "HtmlNode",
    "HtmlStructure",
    "ImageBox",
    "NeedsOcrError",
    "PageGeometry",
    "PageKind",
    "PageLayout",
    "PageTables",
    "PdfClassification",
    "PdfPermissions",
    "PdfSecurity",
    "Row",
    "Table",
    "TextLine",
    "TextSpan",
    "Word",
    "XlsxCell",
    "XlsxComment",
    "XlsxDefinedName",
    "XlsxHyperlink",
    "XlsxSheet",
    "XlsxStructure",
    "__version__",
    "document_structure",
    "page_geometry",
    "page_layout",
    "page_tables",
    "pdf_classify",
    "pdf_extract",
    "pdf_link_uris",
    "pdf_page_count",
    "pdf_page_labels",
    "pdf_security",
    "sniff",
    "to_markdown",
    "to_text",
]

try:
    from tors_documents import (  # noqa: F401
        Backend,
        Cell,
        DocumentStructure,
        DocxBlock,
        DocxCell,
        DocxImage,
        DocxLink,
        DocxStructure,
        DocxTable,
        Format,
        HtmlAnnotation,
        HtmlGrid,
        HtmlGridCell,
        HtmlNode,
        HtmlStructure,
        ImageBox,
        NeedsOcrError,
        PageGeometry,
        PageKind,
        PageLayout,
        PageTables,
        PdfClassification,
        PdfPermissions,
        PdfSecurity,
        Row,
        Table,
        TextLine,
        TextSpan,
        Word,
        XlsxCell,
        XlsxComment,
        XlsxDefinedName,
        XlsxHyperlink,
        XlsxSheet,
        XlsxStructure,
        __version__,
        document_structure,
        page_geometry,
        page_layout,
        page_tables,
        pdf_classify,
        pdf_extract,
        pdf_link_uris,
        pdf_page_count,
        pdf_page_labels,
        pdf_security,
        sniff,
        to_markdown,
        to_text,
    )
except ImportError as exc:  # noqa: PERF203
    raise ImportError(
        "the documents-extraction surface ships in the tors-documents wheel: "
        "install it with `pip install tors[documents]`"
    ) from exc
