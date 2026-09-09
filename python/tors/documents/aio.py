"""The awaitable spellings of the documents surface, re-exported from the
payload wheel (``tors_documents.aio``): the sync/async choice discipline is
the base ``tors.aio``'s — the input-scaling functions (file read + parse +
conversion) get ``asyncio.to_thread`` twins because a large document costs
hundreds of milliseconds and the loop must never feel it; microsecond-scale
calls (``sniff``'s marker scan) stay sync-only, the thread hop costing more
than the call.
"""

from __future__ import annotations

from tors_documents.aio import (
    pdf_classify as pdf_classify,
)
from tors_documents.aio import (
    pdf_extract as pdf_extract,
)
from tors_documents.aio import (
    pdf_link_uris as pdf_link_uris,
)
from tors_documents.aio import (
    pdf_page_count as pdf_page_count,
)
from tors_documents.aio import (
    to_markdown as to_markdown,
)
from tors_documents.aio import (
    to_text as to_text,
)

__all__ = [
    "pdf_classify",
    "pdf_extract",
    "pdf_link_uris",
    "pdf_page_count",
    "to_markdown",
    "to_text",
]
