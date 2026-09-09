"""The documents-extraction surface's base-wheel shim.

The compiled surface ships in the separate `tors-documents` wheel
(`tors[documents]`, a real PyPI extra): the base `tors` wheel carries this
typed shim, which re-exports the payload's API when it is installed and
raises the install hint when it is not — the base extension module stays
engine-free, which is the whole point of the split.

Install: ``pip install tors[documents]`` — or the payload directly,
``pip install tors-documents`` (kept in version lockstep with ``tors``).
"""

from __future__ import annotations

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

try:
    from tors_documents import (  # noqa: F401
        Backend,
        Format,
        NeedsOcrError,
        PageKind,
        PdfClassification,
        __version__,
        pdf_classify,
        pdf_extract,
        pdf_link_uris,
        pdf_page_count,
        sniff,
        to_markdown,
        to_text,
    )
except ImportError as exc:  # noqa: PERF203
    raise ImportError(
        "the documents-extraction surface ships in the tors-documents wheel: "
        "install it with `pip install tors[documents]`"
    ) from exc
