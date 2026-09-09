"""The awaitable spellings of the documents surface: every wrapped
function is ``asyncio.to_thread`` over the typed wrapper, so the async
answers carry the same types as the sync ones — ``Format`` for the
resolved format, the ``PdfClassification`` view for ``pdf_classify`` —
and the ``path``/``data=`` pair flows through unchanged (``data=`` being
the in-memory caller's whole point: no thread hop should reintroduce a
temp file). ``sniff`` is deliberately absent:
it stays sync-only — a single short native pass (a container parse, not
a marker scan; see the sync docstring's measured cost), where the
thread hop would price the awaitable spelling above its value.

Cancellation semantics: ``asyncio.to_thread`` is uncancellable
mid-pass — cancelling the awaiting task (or a ``wait_for`` timeout)
detaches the await only, while the underlying thread runs the native
pass to completion, holding its memory (the engines have no
cancellation cooperation); repeated timeouts against large inputs pin
the shared default executor's threads, one per abandoned call. A caller
that needs abandonable conversion should run it in a process it
controls. See ``tors_documents.aio``'s docstring for the full statement.
"""

from __future__ import annotations

import os

from tors_documents import Backend, Format, PdfClassification

async def to_markdown(
    path: str | os.PathLike[str] | None = ...,
    data: bytes | None = ...,
    format: Format | str | None = ...,
    backend: Backend | str = ...,
    pages: int | list[int] | tuple[int, int] | None = ...,
    password: str | None = ...,
    max_bytes: int | None = ...,
) -> tuple[Format, str]: ...
async def to_text(
    path: str | os.PathLike[str] | None = ...,
    data: bytes | None = ...,
    format: Format | str | None = ...,
    backend: Backend | str = ...,
    pages: int | list[int] | tuple[int, int] | None = ...,
    password: str | None = ...,
    max_bytes: int | None = ...,
) -> tuple[Format, str]: ...
async def pdf_extract(
    path: str | os.PathLike[str] | None = ...,
    data: bytes | None = ...,
    password: str | None = ...,
) -> tuple[list[str], str]: ...
async def pdf_page_count(
    path: str | os.PathLike[str] | None = ...,
    data: bytes | None = ...,
    password: str | None = ...,
) -> int: ...
async def pdf_link_uris(
    path: str | os.PathLike[str] | None = ...,
    data: bytes | None = ...,
    password: str | None = ...,
) -> list[list[str]]: ...
async def pdf_classify(
    path: str | os.PathLike[str] | None = ...,
    data: bytes | None = ...,
    password: str | None = ...,
) -> PdfClassification: ...
