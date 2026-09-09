"""The awaitable spellings of the documents surface, re-exported from the
payload wheel (``tors_documents.aio``): the sync/async choice discipline is
the base ``tors.aio``'s — the input-scaling functions (file read + parse +
conversion) get ``asyncio.to_thread`` twins because a large document costs
hundreds of milliseconds and the loop must never feel it; ``sniff``
stays sync-only — a single short native pass (a container parse, not a
marker scan), where the thread hop would price the awaitable spelling
above its value.

Cancellation semantics (the payload aio module's, stated here because this
re-export is the surface base-wheel callers hold): ``asyncio.to_thread``
is uncancellable mid-pass — cancelling the awaiting task (or a
``wait_for`` timeout) detaches the await only, while the underlying
thread runs the native pass to completion, holding its memory, because
the engines have no cancellation cooperation. Repeated timeouts against
large inputs therefore pin the shared default executor's threads, one
per abandoned call. A caller that needs abandonable conversion should
run it in a process it controls; see ``tors_documents.aio``'s docstring
for the full statement.
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
