# tors-documents

The `tors.documents` payload wheel: document-format extraction to
GitHub-Flavored Markdown or plain text, GIL-free. PDF, the office and text
formats (doc/docx, xls/xlsx, ppt/pptx, rtf, odt/ods/odp, epub, csv/tsv), and
HTML, converted on the engine measured best for each family (pdf_oxide for
PDF, anydoc for the office/text families, html-to-markdown-rs for HTML,
office_oxide as the caller-selectable `backend="oxide"` lane). The whole
read + sniff + convert pass runs under one `py.detach`, so the GIL stays
free for the rest of your program while a large document converts.

## Install

```sh
pip install tors[documents]
```

or this wheel directly:

```sh
pip install tors-documents
```

The base `tors` wheel ships no engines at all; this payload is what
`pip install tors[documents]` pulls in. The split is the lazy-import design
too: `import tors` never touches this wheel, and `import tors.documents`
peaks under 19 MB in a fresh process despite all four engines compiled into
the one payload `.so` (demand paging). The measured import-cost numbers are
in the `tors.documents` section of the full API reference (linked below).

## Import path

```python
import tors.documents
```

The base wheel re-exports this payload as `tors.documents`, with an install
hint pointing at the extra when this wheel is absent, so that spelling is
the surface to target. `tors_documents` is the distribution name, not the
API.

## Version lockstep with `tors`

This wheel carries the same version number as the base `tors` wheel, and the
two release together (one release PR bumps both manifests; see the
repository's `release-please-config.json`). The `tors[documents]` extra
resolves this wheel; the matching versions install side by side.

## Documentation

- [Documents](https://azx-pbc-oss.github.io/tors/documents/): what the
  surface is, the engine table, and why the split into two wheels
- [The full API reference](https://azx-pbc-oss.github.io/tors/api/#torsdocuments):
  `to_markdown`/`to_text`/`sniff`/`pdf_extract`/`pdf_page_count`/`pdf_classify`/`pdf_link_uris`,
  the `Backend`/`Format`/`PageKind` enums, `PdfClassification`,
  `NeedsOcrError`, and `tors.documents.aio`
