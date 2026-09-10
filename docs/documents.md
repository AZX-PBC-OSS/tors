# Documents

`tors.documents` converts document formats to GitHub-Flavored Markdown or plain
text: PDF, the office and text formats (doc/docx, xls/xlsx, ppt/pptx, rtf,
odt/ods/odp, epub, csv/tsv), and HTML. The whole read + sniff + convert pass
runs inside one `py.detach`, so the GIL stays free for the rest of your program
while a large document converts. It ships as a second wheel:

```sh
pip install tors[documents]
```

From a checkout: `uv sync --locked --extra documents`.

## Why two wheels

The extraction engines live in `tors-core` behind the cargo feature
`documents`, which is default OFF, so the base `tors` build carries none of the
engine weight; only the payload wheel compiles it. The compiled surface itself
is the payload package `tors_documents`; the base wheel ships only a typed
shim, `tors.documents`, which re-exports the payload's names when it is
installed (import through `tors.documents`: the payload's own name is the
distribution it rides on, not the API) and raises the install hint when it is
not:

```
ImportError: the documents-extraction surface ships in the tors-documents wheel: install it with `pip install tors[documents]`
```

Why the split: the base extension is 7.4 MiB; the same extension with the
documents feature compiled in (engines, extraction cores, bindings) is
30.1 MiB, 4.1x (linux x86-64, release build, symbols stripped). Users who
don't extract documents keep the lean download. One known cost of the
current build: the payload `.so` also carries the base surface's compiled
code a second time (~37.5 MiB of extensions against the ~30 MiB a single
combined wheel would weigh), an engine-lane cleanup candidate, not a cost of
the split itself.

The payload is version-locked to `tors`: the same number in both
`pyproject.toml`s and both `Cargo.toml`s, bumped together by release-please,
with the `publish` workflow asserting the lockstep before uploading.

## Example

```python
import tors.documents

fmt, markdown = tors.documents.to_markdown("report.docx")
# (Format.DOCX, "Quarterly Review Q3 2026\n\n...")

cls = tors.documents.pdf_classify("incoming.pdf")
if cls.image_only:
    route_to_ocr("incoming.pdf")  # every page is a scan: nothing local can read it
elif cls.pages_needing_ocr:  # 0-based indices of the image-only pages
    ocr_pages("incoming.pdf", cls.pages_needing_ocr)

try:
    fmt, text = tors.documents.to_text("incoming.pdf", backend="anydoc")
except tors.documents.NeedsOcrError as exc:
    ocr_pages("incoming.pdf", exc.pages)  # 0-based indices, same as pages_needing_ocr
```

`pages=` selects a PDF page subset on the pdf_oxide lane only (one 0-based
page, a list, or a half-open `(start, stop)` range); the selection is deduped
into document order and a full range is byte-identical to the whole-document
conversion. Any other format, or `backend="anydoc"` on a PDF, refuses `pages=`
with `ValueError` instead of silently converting the whole document.

## The API

```python
to_markdown(path=None, data=None, format=None, backend="auto", pages=None,
            password=None, max_bytes=None) -> tuple[Format, str]
to_text(path=None, data=None, format=None, backend="auto", pages=None,
        password=None, max_bytes=None) -> tuple[Format, str]
sniff(data: bytes) -> Format | None
pdf_extract(path=None, data=None, password=None, backend="auto",
            max_bytes=None) -> tuple[list[str], str]
pdf_page_count(path=None, data=None, password=None, backend="auto",
               max_bytes=None) -> int
pdf_classify(path=None, data=None, password=None, backend="auto",
             max_bytes=None) -> PdfClassification
pdf_link_uris(path=None, data=None, password=None, backend="auto",
              max_bytes=None) -> list[list[str]]
```

The full per-function contract (signatures, the `pages=` and
format-resolution rules, the error taxonomy, the enums, `PdfClassification`,
`NeedsOcrError`, `tors.documents.aio`) is the documents section of the
[API reference](api.md#torsdocuments). The native docstrings live in
`tors-documents/src/lib.rs`, held by `tests/test_documents_engines.py` and
`tests/test_pdf.py`.

The rules in brief:

- The document is a `path` or in-memory `data=` bytes; a `data=` call converts
  byte-identically with no temp-file roundtrip.
- `format=` names the format explicitly (extension spelling,
  case-insensitive). The container variants `docm`/`xlsm`/`ppsx` map onto
  their parents: the same OOXML packages with the content-type override naming
  the macro/show variant, resolved and sniffed as their base kinds. `tsv` is
  accepted as a name and resolves onto the csv kind; the resolved and sniffed
  name is always `"csv"`, never `"tsv"`. `xlsb` is vocabulary only: the name
  routes the Excel kind, but genuine xlsb content is BIFF12 `.bin` sheets, not
  worksheet XML, and is refused (the engines do not read it). `None` sniffs
  the format from content markers (the PDF header, the RTF open group, OLE
  stream names, the ZIP package mimetype, the HTML document marker, with a
  delimiter-agreement heuristic for signature-less CSV), with the path's
  extension as the fallback, so a mislabeled file still converts. The returned
  `Format` is what the conversion actually used, and the Excel family reports
  its container honestly: `"xls"` for OLE bytes, `"xlsx"` for a ZIP package,
  regardless of what the explicit name said.
- `sniff(data)` answers the same content question standalone, over bytes alone
  with no path: the mislabeled-download answer. It opens and parses the
  container (anydoc's detection reads the ZIP/OLE package metadata; a 120 KiB
  zip measured 267 MiB peak RSS to answer docx), so budget it like a parse,
  not a marker scan. `None` is not an error: the content names no format.- `pdf_extract` returns `(per_page_plain_text, whole_document_markdown)` from
  one pass over one open document; the parse is paid once for both outputs.
  An image-only page is an empty string, not an error; routing is the
  caller's decision.
- `pdf_classify` is the cheap text-vs-image preflight: `PdfClassification`
  carries `.page_count`, `.page_kinds` (per page: `"text"`, `"scanned"`,
  `"image_text"`, `"mixed"`, or `"empty"`; a blank page is a different thing
  from a scan), `.pages_needing_ocr` (the 0-based indices of image-only pages,
  excluding blanks), and the derived `.has_text` / `.image_only`. Encrypted
  documents fail closed on every entry (`ValueError` without `password=`).
- `max_bytes=` is the input budget. An explicit value binds every engine lane
  before a byte is read or copied; `None` keeps the 32 MiB default, a
  post-read check on the amplifying lanes (anydoc, office_oxide) only. On the
  PDF-only family `None` is unmetered: those calls never run either metered
  lane.
- Errors: `OSError` for a missing/unreadable file (the matched subclass, e.g.
  `IsADirectoryError` on a directory, `FileNotFoundError` for a missing path);
  `ValueError` for an unknown format name, an undetectable file, an unusable
  backend/format pair, the PDF-only family's `backend="anydoc"` capability
  refusal (raised before any work runs), an invalid `pages=` selection, a
  malformed or encrypted document, an input over `max_bytes=`, a non-regular
  `path=` (FIFO/device/socket: typed, naming `path` and the kind, before the
  read), or a NUL byte inside `path=` (CPython's own `open()` convention).
  `NeedsOcrError` (a `ValueError` subclass carrying `.pages`, the 0-based
  indices needing OCR, and `.page_count`) is raised only on the anydoc PDF
  lane; the default pdf_oxide lane returns empty output for scanned pages and
  leaves the OCR decision to `pdf_classify`/`pdf_extract`.

## The engine matrix

Four engines, routed per format family by head-to-head measurement (the full
comparison lives in the `documents_impl` crate docs; the suite pinning it is
`tests/test_documents_engines.py`). `backend="auto"` is the measured-best
lane; a forced backend raises on a format its engine cannot read, never a
silent fallback:

| format family | `backend="auto"` | `backend="oxide"` | `backend="anydoc"` |
|---|---|---|---|
| pdf | pdf_oxide 0.3.78 | pdf_oxide | anydoc (pdf-inspector) |
| the PDF-only family (`pdf_extract`/`pdf_page_count`/`pdf_classify`/`pdf_link_uris`) | pdf_oxide | pdf_oxide | `ValueError`: a capability refusal, not a format one (see below) |
| html / htm | html-to-markdown-rs 3.12 | `ValueError` | `ValueError` |
| doc / docx (incl. `docm`) | anydoc 0.2.4 | office_oxide 0.1.10 | anydoc |
| xls / xlsx (incl. `xlsm`) | anydoc | office_oxide | anydoc |
| ppt / pptx (incl. `ppsx`) | anydoc | `ValueError` (ppsx) / office_oxide (pptx) | anydoc |
| rtf | anydoc | `ValueError` | anydoc |
| odt / ods / odp | anydoc | `ValueError` | anydoc |
| epub | anydoc | `ValueError` | anydoc |
| csv / tsv | anydoc | `ValueError` | anydoc |

`xlsb` is absent on purpose: the name routes the Excel kind, but genuine xlsb
content (BIFF12 `.bin` sheets, not worksheet XML) is refused by both engines.
`ppsx` converts on the auto/anydoc lane and is refused by office_oxide, which
checks the presentation content type. Both refusals are typed errors pinned in
the suite.

The PDF-only family's row is a capability refusal, not a format one:
PDF+anydoc converts (`to_markdown`/`to_text` with `backend="anydoc"`), but
anydoc's entire PDF surface is whole-document markdown:
`to_markdown(bytes)` is the one function its PDF module exposes
(~anydoc-0.2.4/src/formats/pdf.rs), and its only per-page knowledge is the
NeedsOcr refusal. The four probe calls need exactly the surfaces it lacks: the
per-page text probe (the OCR-routing signal), the page tree, per-page
classification, and the `/Annots` walk. `backend="anydoc"` on any of the four
is a named `ValueError` raised before any work runs, pointing at the
conversion pair.

Why this routing: pdf_oxide reads two-column layouts as separate
reading-order blocks, renders `/Link` annotations as `[text](uri)`, and
detects oversize-font headings, where anydoc's PDF engine (pdf-inspector)
interleaves columns line-by-line and drops links; html-to-markdown-rs is the
only measured HTML engine that drops `<script>`/`<style>` by construction;
anydoc renders style-based docx headings (`pStyle` → `#`) and list markers
that office_oxide drops entirely, and covers rtf/odt/epub/csv office_oxide
cannot read at all. `backend="oxide"` exists so a pipeline can diff the two
engines' output on its own corpus; office_oxide's one measured win is exact
entity text (no `&`-escaping, which anydoc does and the plain-text strip
normalizes back).

Every probe-able cell above was verified against the committed fixtures in
`tests/engines_corpus/`, the OOXML alias containers included (`docm`/`xlsm`/
`ppsx` fixtures are [Content_Types].xml rewrites of the base generators,
converted byte-identically to their bases and pinned as such). The legacy
`doc`/`xls`/`ppt` cells have no Python fixture (OLE compound files are not
hand-generatable without an Office writer) and are pinned by the crate-side
routing-table test (`src/documents_impl.rs`) plus the mutation lane's
typed-error contract; they are uncovered at the conversion level.

## Async and the GIL

`tors.documents.aio` gives the six input-scaling functions (`to_markdown`,
`to_text`, `pdf_extract`, `pdf_page_count`, `pdf_classify`, `pdf_link_uris`)
their `asyncio.to_thread` twins: same signatures, same discipline as
[`tors.aio`](async.md) (an unconditional thread hop, never a size-based
branch). `sniff` stays sync-only: its cost is a container parse, but it
remains a single short native pass a sync caller runs directly.

Every function runs its whole native pass (file read, format sniff, engine
conversion, and `to_text`'s strip) inside one `py.detach`, with only the
argument borrow, validation, and the O(output) return marshalling under the
GIL. The O(n) bytes copy a `data=` call pays rides inside the detach with the
rest of the pass: a 400 MB call's max heartbeat gap is ~1.1 ms. The official
pdf_oxide pyo3 wheel measures as GIL-held per call (worst heartbeat gap
23.6 ms on a 9-page document, growing with document size), so this payload
calls the crate's Rust API directly under `py.detach` instead.

The bands are pinned by the heartbeat-granularity and 8-thread
byte-identical concurrency gates in `tests/test_documents_engines.py` and
`tests/test_pdf.py` (`timing`-lane marked; CI's matrix legs deselect the
lane, one 3.12 leg runs it).

`asyncio.to_thread` is uncancellable mid-pass (the aio docstrings carry this
too): a `wait_for` timeout returns control while the thread runs the
conversion to completion, and repeated timeouts pin the default executor's
threads. See [`tors.documents.aio`](api.md#torsdocumentsaio) in
the API reference.
