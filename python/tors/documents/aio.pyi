"""The awaitable spellings of the documents surface, mirrored standalone
from the payload wheel's aio stub (see tors/documents/aio.py for the
runtime re-export; the standalone mirror follows the base stub's rule —
base-only type checks must resolve without the payload wheel installed).
Each is ``asyncio.to_thread`` over the typed wrapper, so the async answers
carry the same types as the sync ones, and the ``path``/``data=`` pair
flows through unchanged (``data=`` being the in-memory caller's whole
point: no thread hop should reintroduce a temp file). ``sniff`` stays
sync-only (the marker scan is microseconds; the thread hop would cost
more than the call)."""

from __future__ import annotations

import os

from tors.documents import Backend, Format, PdfClassification

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
