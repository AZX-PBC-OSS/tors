"""The awaitable spellings of the documents surface (the base wheel's
``tors.aio`` pattern, applied to this payload's functions: every one is a
single native pass whose cost scales with the document, the exact class a
thread hop pays for — a small document is still milliseconds, a large one
is hundreds, and the loop must never feel the difference).

The wrapped functions are the TYPED wrappers from ``tors_documents``'s
``__init__`` (resolved via ``getattr`` at import), so the async answers
carry the same types as the sync ones — ``Format`` for the resolved
format, the ``PdfClassification`` view for ``pdf_classify`` — never the
raw native strings — and the ``path``/``data=`` pair flows through
unchanged (``data=`` being the in-memory caller's whole point: no thread
hop should reintroduce a temp file). ``sniff`` is deliberately absent: its marker scan is
microsecond-scale and stays sync-only, the thread hop costing more than
the call.

Unconditionally ``asyncio.to_thread``, no size-based branching, the same
contract as ``tors.aio``: the choice between the sync spelling and this
module is the caller's, made once at the call site, not a runtime guess.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any

import tors_documents

__all__: list[str] = []

_WRAPPED = (
    "pdf_classify",
    "pdf_extract",
    "pdf_link_uris",
    "pdf_page_count",
    "to_markdown",
    "to_text",
)


def _make_async(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(fn, *args, **kwargs)

    wrapper.__qualname__ = f"tors_documents.aio.{name}"
    return wrapper


for _name in _WRAPPED:
    _fn = getattr(tors_documents, _name)
    globals()[_name] = _make_async(_name, _fn)
    __all__.append(_name)

__all__.sort()
del _name, _fn
