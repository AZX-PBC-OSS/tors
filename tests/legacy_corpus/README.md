Committed binary fixtures: the legacy office trio (real [MS-CFB]/OLE
containers, which no in-test builder can fake), generated on the
authoring box with `/usr/bin/soffice --headless --convert-to` from
minimal OOXML sources built by the standard writers (python-docx,
openpyxl, python-pptx):

- `t.doc`     — a one-section Word document.            → geometry: `[None]`
- `book3.xls` — a three-sheet workbook (Alpha/Beta/Gamma). → geometry: `[None, None, None]`
- `deck3.ppt` — a three-slide presentation.             → geometry: `[None, None, None]`

Provenance and determinism: these bytes are COMMITTED AS-IS and are NOT
regenerable byte-identically (soffice stamps creation times and revision
metadata into the OLE header), so they deliberately live outside the
byte-pinned corpus registries (tests/corpus/, tests/engines_corpus/) and
have no generator. The pin is the committed bytes themselves; a refresh
is a deliberate fixture change, re-run through the same pipeline.

Loaded by both ends of the geometry pin: the core
(`src/documents_impl.rs`'s tests, `include_bytes!`) and the payload
suite (`tests/test_documents_engines.py`, the legacy-fixture cells).
`deck3.ppt` is large on disk (~600 KB) because soffice's legacy
PowerPoint export carries a fixed template payload; it is ~6 KB
compressed in git.
