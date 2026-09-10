"""The awaitable spellings of the documents surface (the base wheel's
``tors.aio`` pattern, applied to this payload's functions: every one is a
single native pass whose cost scales with the document, the exact class a
thread hop pays for: a small document is still milliseconds, a large one
is hundreds, and the loop must never feel the difference).

The wrapped functions are the typed wrappers from ``tors_documents``'s
``__init__`` (resolved via ``getattr`` at import), so the async answers
carry the same types as the sync ones (``Format`` for the resolved
format, the ``PdfClassification`` view for ``pdf_classify``, never the
raw native strings), and the ``path``/``data=`` pair flows through
unchanged (``data=`` being the in-memory caller's whole point: no thread
hop should reintroduce a temp file). ``sniff`` is deliberately absent:
it stays sync-only, a single short native pass (a container parse, not
a marker scan; see the sync docstring's measured cost), where the
thread hop would price the awaitable spelling above its value.

Unconditionally ``asyncio.to_thread``, no size-based branching, the same
contract as ``tors.aio``: the choice between the sync spelling and this
module is the caller's, made once at the call site, not a runtime guess.

Cancellation semantics, read before relying on ``wait_for``
timeouts: ``asyncio.to_thread`` is uncancellable mid-pass. Cancelling the
awaiting task (or a ``wait_for`` timeout firing) detaches the await only;
the underlying thread keeps running the native pass (read, sniff,
conversion) to completion, holding the memory that pass needs, because
the engines have no cancellation cooperation of their own and the thread
cannot be reclaimed or interrupted. Two consequences worth pricing
before the pattern is load-bearing: a large document keeps its thread
(and its peak RSS) busy for the whole native wall no matter how quickly
the caller gives up, and repeated timeouts against large inputs pin the
shared default executor's threads one per abandoned call, each running
to completion, a slow executor-pool leak under retry loops. True
cancellation needs engine cooperation and is out of scope here; a caller
that needs abandonable conversion should run it in a process the caller
controls (``concurrent.futures.ProcessPoolExecutor`` or a worker whose
lifetime the caller owns) and abandon that.
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
