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
conversion. A range endpoint no PDF can carry (past the 8,388,607-object
cross-reference ceiling) is refused with `ValueError` under the GIL, before
anything is built on it: the measured pre-fix failure asked the allocator
for 800 PB at `pages=(0, 10**17)` and died by SIGABRT. Any other format, or
`backend="anydoc"` on a PDF, refuses `pages=`
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
pdf_page_labels(path=None, data=None, password=None, backend="auto",
                max_bytes=None) -> list[str | None]
pdf_security(path=None, data=None, password=None, backend="auto",
             max_bytes=None) -> PdfSecurity
pdf_classify(path=None, data=None, password=None, backend="auto",
             max_bytes=None) -> PdfClassification
pdf_link_uris(path=None, data=None, password=None, backend="auto",
              max_bytes=None) -> list[list[str]]
page_geometry(path=None, data=None, format=None, password=None, backend="auto",
              max_bytes=None) -> list[PageGeometry | None]
page_layout(path=None, data=None, format=None, password=None, backend="auto",
            pages=None, max_bytes=None) -> list[PageLayout]
page_tables(path=None, data=None, format=None, password=None, backend="auto",
            pages=None, max_bytes=None) -> list[PageTables]
page_images(path=None, data=None, format=None, password=None, backend="auto",
            pages=None, max_bytes=None, decode=None,
            max_image_bytes=None) -> list[PageImagesData]
page_text_chars(path=None, data=None, format=None, password=None,
                backend="auto", pages=None, max_bytes=None) -> list[PageCharsData]
document_structure(path=None, data=None, format=None, backend="auto",
                   max_bytes=None) -> DocumentStructure
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
  documents fail closed on every entry (`ValueError` without `password=`;
  the carve-out is `pdf_security`, which reports state instead — below).
- `pdf_page_labels` returns the page label a reader displays, per page
  (`"i"`, `"A-1"`, `"12"`): the ISO 32000 §12.4.2 numbering tree. `None` means
  the page declares no label (no tree, or before the tree's first range) — the
  caller's fabricate-an-ordinal signal, never fabricated here; a declared label
  is always a string, including the empty one (a styleless, prefixless range
  declares it). Labels render on u64 arithmetic under the bounded-label law:
  a label whose render exceeds 4 KiB bounds out to `None` for that page, never
  truncated — the subtractive Roman render is one char per thousand of the
  value, so a legal `/St 4294967294` renders 4,294,967 chars per page, a value
  display-prohibitive by construction (decimal and alpha renders are O(log n)
  and never reach the bound); the arithmetic itself never panics or wraps.
  Rotation and crop are irrelevant (tree data); a corrupt tree is presented as
  "no labels" (the documented masking). The encrypted door governs unchanged:
  a locked document raises, never ciphertext labels.
- `pdf_security` is the typed encryption/permissions probe — the door's one
  documented carve-out: it REPORTS the locked state (`ValueError`-free) instead
  of failing closed. `PdfSecurity` carries `.encrypted`,
  `.unlocked_with_supplied_password` (the credentials THIS call held:
  `password=None` means the implicit empty attempt), `.permissions` (the
  declared `/P` flags, advisory per §7.6.3.2, or `None`), and
  `.owner_password_only` (the empty password authenticated). Reading rule:
  `readable = !encrypted || unlocked || owner_password_only` — never read
  `unlocked == false` as "locked" (the owner-only + wrong-password cell). Three
  states collapse into `encrypted && permissions is None` (malformed `/Encrypt`,
  unsupported `/V`/`/R`, a crypto-policy denial): "needs a password" is
  indistinguishable from "this build/policy refuses the cipher", and the report
  is identical with or without a password. It never unlocks beyond the supplied
  password (one password per call plus the open's own single implicit empty
  attempt); the fail-closed door stays every content surface's single encryption
  authority.
- `max_bytes=` is the input budget. An explicit value binds every engine lane
  before a byte is read or copied, overriding the default in either
  direction; `None` reads under `MAX_INPUT_READ`, a finite 512 MiB read
  ceiling enforced during the read itself (a bounded prefix picks the lane,
  then the rest is read under that lane's ceiling), plus the existing
  32 MiB default, a post-read check on the amplifying lanes (anydoc,
  office_oxide, HTML — the HTML converter holds input and output at once,
  ~23x input, measured) only. On the PDF-only family `None` has no lane-specific
  32 MiB policy: those calls never run a metered lane, but they still
  read bounded by the 512 MiB backstop, a memory-safety floor rather than a
  lane policy, not unbounded. What the ceiling bounds is the bytes handed
  in, never the bytes they inflate to; the inflated side is bounded one
  lane deep. anydoc caps decompression engine-side (128 MiB per entry,
  512 MiB total across entries). The oxide lane runs a decompression audit
  at the seam before its engine parses anything: a container whose parts
  inflate past 512 MiB in total (declared sizes summed, then the parts
  themselves re-inflated under a hard per-part bound, so a lying header
  cannot get past it) is refused with `ValueError` naming the ceiling, and
  office_oxide's own 512 MiB per-part cap stands underneath it. No lane
  has an output-size cap beyond those.
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

**A malformed PDF's crash class is refused at the seam, not passed to the
parser.** The `ValueError` for a malformed document above is the ordinary
path, and the one crash class found on it, pdf_oxide 0.3.78's object parser
recurses with no depth cap (its `ParserConfig::max_nesting` is dead code; a
trailer nesting `[` arrays 10,659 deep SIGSEGV'd every entry point, exit -11
in ~170 ms, uncatchable from Python), is defended at the binding's own seam
in three layers that run before pdf_oxide parses anything
(`src/pdf_impl.rs`'s module docs carry the full measured record): a linear
raw-byte nesting scan (refusal past 100 levels, pdf_oxide's own intended
cap), the same scan re-run over FlateDecode object-stream and xref-stream
payloads after inflating them under a hard 256 MiB ceiling (the compressed
shape the raw scan cannot see, measured still SIGSEGV-ing pre-fix at
N=12,000), and the whole native pass on a 256 MiB-stack thread whose panic
maps to the same catchable error. Over-depth and over-ceiling documents
raise the malformed-document `ValueError` on every entry point; all shapes
pinned red-first in `tests/test_documents_pdf_crashes.py`. The real cap
belongs upstream (issue yfedoseev/pdf_oxide#1474, fix commit 68da2cf8,
0.3.79+); the in-repo layers stay as defense-in-depth after it ships, and
see SECURITY.md for the coordination status.

## The engine matrix

Four engines, routed per format family by head-to-head measurement (the full
comparison lives in the `documents_impl` crate docs; the suite pinning it is
`tests/test_documents_engines.py`). `backend="auto"` is the measured-best
lane; a forced backend raises on a format its engine cannot read, never a
silent fallback:

| format family | `backend="auto"` | `backend="oxide"` | `backend="anydoc"` |
|---|---|---|---|
| pdf | pdf_oxide 0.3.78 | pdf_oxide | anydoc (pdf-inspector) |
| the PDF-only family (`pdf_extract`/`pdf_page_count`/`pdf_page_labels`/`pdf_security`/`pdf_classify`/`pdf_link_uris`) | pdf_oxide | pdf_oxide | `ValueError`: a capability refusal, not a format one (see below) |
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
normalizes back). The opt-in lane carries its own bounds: the 32 MiB input
ceiling (or the caller's `max_bytes=`), office_oxide's 512 MiB per-part
cap, and the seam's decompression audit refusing any container whose parts
inflate past 512 MiB in total (declared sizes summed first, then the parts
re-inflated under a hard per-part bound: a lying header cannot get past
it), the multi-part shape every per-part cap alone slips under.

Every probe-able cell above was verified against the committed fixtures in
`tests/engines_corpus/`, the OOXML alias containers included (`docm`/`xlsm`/
`ppsx` fixtures are [Content_Types].xml rewrites of the base generators,
converted byte-identically to their bases and pinned as such). The legacy
`doc`/`xls`/`ppt` cells have no Python fixture (OLE compound files are not
hand-generatable without an Office writer) and are pinned by the crate-side
routing-table test (`src/documents_impl.rs`) plus the mutation lane's
typed-error contract; they are uncovered at the conversion level.

## Page geometry and layout

`page_geometry` and `page_layout` are the two geometry probes: per unit of
the resolved format's geometry lane, the unit's declared render box — and,
PDF pages only for `page_layout`, the page's text spans, text lines (words
nested), and image boxes, all in the page's declared render space, from one
open. (`page_tables` is the third probe on the same lane: per-cell table
boxes, PDF pages only — see [below](#per-cell-table-boxes); and the two
content probes [`page_images`](#page-image-bytes) and
[`page_text_chars`](#char-level-boxes) ride the same lane: the bytes behind
the image handles, and per-character boxes from the cached span parse.) The
same
source spine, refusal order, and `format=`/`password=/
`max_bytes=` contracts as the probes above; `auto` and `oxide` are identical
on these surfaces for every format (the geometry lane is a property of the
format, not a choice), and `pages=` is PDF-only — a page index cannot name a
section, so a non-`None` selection on any other format raises the conversion
lane's own `ValueError`, never a silent whole-document answer.

Geometry routing is its own table, next to the conversion matrix above —
conversion answers "who converts best"; geometry answers "who can declare
geometry at all", and the two deliberately disagree:

| format | unit | length | `auto` / `oxide` | `anydoc` |
|---|---|---|---|---|
| pdf | page | page count | pdf_oxide | `ValueError`: the capability refusal |
| docx | section | section count | office_oxide (raw layer) | `ValueError` |
| pptx | slide | slide count | office_oxide (raw layer) | `ValueError` |
| xlsx | worksheet | worksheet count | office_oxide (count only) | `ValueError` |
| doc / xls / ppt | section | section count | office_oxide (IR count) | `ValueError` |
| html / csv / tsv / rtf / epub / odt / ods / odp | document | exactly 1 (`[None]`) | no engine runs | the same `[None]` — nothing to refuse |

The worksheet count is the raw layer's kind-filtered WORKSHEETS
(`SheetKind::Worksheet | Macro`): a chartsheet or dialog tab is not a
unit — the same filter the structure surface's sheet list applies, so
`page_geometry`/`page_layout` and `document_structure` agree on the same
workbook (a chartsheet-bearing workbook answers 3 and 3, pre-fix 4 and 3).

Office sizes are honest per unit: docx sections read the RAW declared size
(twips ÷ 20 — a margins-only section is `None`, never a fabricated Letter),
pptx the raw deck-wide slide size (EMU ÷ 12700.0, `None` everywhere when
undeclared), xlsx all-`None` (a worksheet is an unbounded grid, not a page),
legacy doc/xls/ppt all-`None` at the converter's section count. `ppsx` is
disclosed as its typed refusal here too: the alias resolves onto the pptx
kind and office_oxide's content-type check refuses the slideshow package
(the same `ValueError` the conversion lane raises). The no-geometry row runs
no engine and answers the same `[None]` on every backend, `anydoc` included
— nothing is silently fallen back, so nothing is refused; office units
always report `rotation = 0` and no crop.

Verified edge, disclosed beside the flag (T9, probed against pdf_oxide
0.3.78 — was "known-unverified", now measured): the `rotation_degrees`
flag does NOT cover vertical writing. A WMode-1 run (tategaki — a Type0
font whose `/Encoding` CMap name ends in `-V`, e.g. `/Identity-V`, or
whose embedded CMap declares `/WMode 1` — the engine reads the writing
mode from the font on `Tf`) carries `rotation_degrees == 0.0` (the angle
is the text matrix's; the writing mode plays no part in it) while its box
is the flattened fiction at its worst: measured, a 24pt 2-glyph run at
(300, 700) reports span bbox `(300, 700, -48, 24)` — the run's vertical
advance (-24 pt per glyph) flattened onto x as a NEGATIVE width — where
the glyphs' true ink is a column at `(300, 652, 24, 72)` with zero
overlap, and the line surface re-lays the glyphs over the normalized box
and reports the REVERSED text (`"BA"` for `"AB"`, box `(252, 700, 48,
24)`). The engine's span carries `wmode == 1`, but wmode is deliberately
cut from this surface, so a consumer cannot see it here; on a /Rotate
90/270 page the run is not engine-premapped and the fiction rides the
forward map. Treat vertical-writing sources as outside the highlight-box
guarantee entirely: a highlight box for text that may be tategaki cannot
be validated by `rotation_degrees` alone.

Every box `page_layout` returns lives in the page's **declared render
space**: the visible box's (the CropBox ∩ MediaBox region a reader displays)
with the page's `/Rotate` applied — points, bottom-left origin, y-up; for
text the bottom edge is the glyph baseline and the box extends one font-size
upward (em-box; descenders may extend below). Boxes are unclipped — the page
box is the consumer's clip — and one transform owns the mapping, so no
consumer re-implements it. Three honesty laws ride on the boxes:

- **`rotation_degrees != 0` means "not a highlight box."** A run with its
  own content rotation reports the engine's flattened-extent box, and that
  box is wrong at every non-zero content rotation on any page (180°
  same-size/wrong-anchor with zero ink overlap; ±90° transposed; on a
  /Rotate 90/270 page additionally mapped, its line box degraded to a
  square). It is a flag, not an approximation footnote — and the flag has
  one verified hole: vertical writing (WMode 1, tategaki) carries
  `rotation_degrees == 0` with the flattened box at its worst (measured
  above), so tategaki sources are outside the highlight-box guarantee
  entirely.
- **`is_artifact` is typed-BDC-only.** It is true only for a typed
  `/Artifact <<…>> BDC` whose `/Type` or `/Subtype` the engine recognizes;
  it is false for a bare `/Artifact BMC` and for unrecognized types — so
  `is_artifact == false` is NOT evidence of body text.
- **Image boxes are never visibility-filtered.** The engine's 2 pt drop law
  is the text walk's only: an image wholly outside the visible box is still
  returned, with out-of-range coordinates. The consumer clips to the page
  box, and never reads "image present" as "image visible".

The text `page_layout` returns is its own stream, grounded per line: a
line's text is its words' texts joined with exactly one space (the engine
builds the line from its words and drops every whitespace character before
forming a word, so no word's text carries whitespace and the separator is
always one U+0020). It is a reconstruction, not a slice of the page: a wide
geometric gap yields two words and a space that was never in the content
stream, and a line break is not represented at all (lines are separate
objects). A consumer grounding extracted fields by character offsets grounds
into this stream — not into `extract_text`/`to_text` output, where the two
differ; no `concat(spans) == extract_text` invariant exists or is implied.

The office-row caveat, resolved at the entry level: an office unit's
entry is its declared geometry with EMPTY `spans`/`lines`/`images` — the
same shape a blank, scanned, or content-unparseable PDF page reports —
and every entry now carries its provenance: `PageLayout.format` is the
resolved format's vocabulary name (the `to_markdown` spelling — the
resolved format, not the file extension: the sniff outranks the
extension, and the alias containers resolve onto their base, so `xlsm`
bytes report `"xlsx"`). The format — not the entry's shape — tells you
why the lists are empty: a consumer holding only an entry reads
`entry.format` and knows the office case (no text surfaces) from the
blank/scanned PDF case (no text on the page).

### Per-cell table boxes

`page_tables` is the third geometry-lane surface: per unit of the same
lane, the unit's geometry plus — PDF pages only — the page's detected
tables with per-cell text and boxes, in the same declared render space as
`page_layout`, from one open. The spine, refusal order, `pages=`
semantics, and `backend=`/`password=`/`max_bytes=` contracts are
`page_layout`'s exactly; the capability table above applies to it
identically (it IS the same table), with one addition: the office rows'
`tables: []` is the per-format capability statement — the office engines'
tables carry grid and cell text but no coordinates, and "boxes sometimes"
is the worst answer — so there the entry's `format` field marks
"unsupported here", not "no tables", and `tables_refused` stays false.

| format | tables served |
|---|---|
| pdf | the spatial detector's tables, per-cell boxes in render space |
| office families | `[]` with `tables_refused: false` (capability statement; the `format` field is the marker) |
| html / csv / tsv / rtf / epub / odt / ods / odp | `[]` at the one `geometry=None` entry |

Three honesty laws ride on the boxes, and one flag:

- **`tables_refused` is the deliberate-decline marker.** TRUE iff the
  surface declined a page whose inputs indicate possible table content:
  the frame-incoherent `/Rotate` classes or the edge cap. Reading rule:
  `tables_refused: true` + `geometry.rotation != 0` → **frame
  incoherence** — on pages where `/Rotate` and the drawn content disagree
  (notably /Rotate 180 with ruled tables), the detector compares text in
  one frame against rulings in another: a ruled table vanishes there, a
  180-symmetric one comes back with every cell's text 180°-swapped — so
  this surface reports no tables rather than wrong ones. `tables_refused:
  true` at rotation 0 → **the edge cap** (more table primitives than the
  markdown lane's own 1500-edge bail-out; detection skipped). FALSE for
  no-tables pages, for content-stream failures (unknowable — `tables: []`
  stays indistinguishable from a no-tables page by design), and on office
  rows (their marker is `format`).
- **Cell boxes are GEOMETRY, not ownership.** A cell's box is the
  engine's box for that cell; its regime is per-branch and not observable
  from the return value (intersection cells: ruling rectangles, always
  Some, empty cells included; cluster/band/text cells: span unions, None
  when empty; a vertical merge pads merged rows with None cells). An
  empty cell's box is declared grid geometry, not a claim on the text
  inside it — do NOT subtract cell bboxes from other surfaces' text (the
  engine's own assembly needs per-cell token budgets to do that safely).
  Cell boxes may lie outside the page box (ruling lines are not
  crop-filtered; the TEXT side IS, so a grid can lose enough cells to
  vanish entirely) — returned, the consumer clips; never read "cell box
  present" as "cell is visible".
- **The table set is the spatial detector's answer, not the markdown's.**
  `page_tables` builds on the public `extract_tables` (spatial-only,
  unconsolidated); `to_markdown` renders tables through a different,
  private pipeline (structure-tree-first for tagged PDFs, fragment
  consolidation, the RTL guard, the edge cap, the relaxed wide-table
  retry), so the sets can differ on ANY PDF. Replicating the private
  pipeline is an upstream ask, not a v1 goal.
- **Double-reporting is real.** `page_layout` serves `extract_spans`,
  which returns table members (the drop lives in the markdown assembly's
  consumer filter, not the core): content inside a detected table's area
  may appear in BOTH surfaces; neither de-dups the other, there is no
  identity join. Use `page_tables` for the grid and cell text, `page_layout`
  for everything else, and accept the overlap; do not subtract cell bboxes.

## Page image bytes

`page_images` is the content seam behind `page_layout`'s image handles
([#205]): per PDF page, the page's image handles — the **unfiltered** walk
(no 8×8 gate, no max-pixels filter, paint order; never `extract_images`,
which silently drops sub-8×8 images and swallows per-image decode errors) —
with each image's evidence in the page's declared render space (the same
mapped boxes `page_layout`'s `images` carry), and — only under `decode` —
the bytes behind the handles. `pages=`/`backend=`/`password=`/`max_bytes=`
and the refusal order are `page_layout`'s exactly; office rows carry
`images: []` with `format` provenance (a DOCX's images are real and are not
served — the per-format capability statement, not a content judgment).

The decode knob (two modes, R3): `decode=None` (the default) decodes nothing
— metadata only, every `data.kind == "not_requested"`, the cheap law.
`decode="container"` serves the CONTAINER bytes: a DCT filter anywhere →
the JPEG passthrough with **decryption included** (the raw stream bytes are
ciphertext on an unlocked encrypted PDF and are never served); a bare
`/JPXDecode` stream → its container sniffed from the stream's own bytes
(`"jp2"` for the 12-byte signature box, `"j2k_codestream"` for the SOC
marker), served only when the image is inline or the document is unencrypted,
else the typed `unsupported` error naming both causes; anything else stays
`not_requested`. `decode="pixels"` decodes to the engine's own classification
(raw samples, or the typed errors — the shipped wheel's JBIG2/JPX pixel
decode is the engine's `Unsupported` verbatim, feature-gated).

`data` is the **always-present four-way sum**
(`"not_requested" | "container" | "raw" | "error"`); every arm getter is
`None` unless `kind` names it — "no bytes" is never two different things.
A decode failure is a **visible typed entry**, siblings intact. Caps: 256
MiB per image, 512 MiB per call cumulative — the caps bind on **actual
bytes**: the chain-aware pre-decode estimate is the advisory pre-labour
check, and the same caps re-run on each decode's ACTUAL output
immediately after it, before the bytes enter the payload (one chain class
under-estimates by construction: a Flate→DCT chain is applied
innermost-first, so the container is the INFLATED stream, measured
3,898 stored bytes → 4,000,000 container bytes — caught post-decode). The
over-cap image is a typed `size_cap` entry
(`scope`/`limit_bytes`/`needed_bytes`) — refused, never truncated (a
truncated image is a wrong image with no marker); the call budget is a
budget checked at decode time, so an under-estimated chain can push the
call past it by at most one image's inflation (that entry refuses, scope
`call`; entries already served stand); `max_image_bytes=`
overrides the per-image ceiling (raising it past the engine's env-read
`PDF_OXIDE_MAX_DECOMPRESS_MB` cannot make the ENGINE decode more). The
payload describes itself (`components`/`bpc` are the buffer's stride;
`pixel_format` is a display tag that can lie; `samples_are_raw` TRUE is the
common case — "nothing transformed this buffer", never "undecoded"; `ccitt`
carries all seven CCITT params, without which the packed stream cannot be
decoded at all; ICC profiles are named, not carried).

Three residual gaps the engine leaves, stated so a consumer never mistakes
a shape for completeness: an image XObject with a missing or non-positive
`/Width`·`/Height` yields NO handle at all (this surface cannot see it); a
Form-XObject subtree's error is dropped by the engine's own `if let Ok`; a
missing/unparseable `/ColorSpace` reports `DeviceRGB` with
`color_space_declared: false`. A caller who needs certainty about the paint
count must compare `len(images)` with the render's. Identity:
`(page_index, paint_order)` pairs with `page_layout`'s `images` — the two
surfaces are separate opens over the bytes (nothing cached across them), so
a changed file between the calls breaks the pairing; and bytes decode EVERY
selected image of every selected page (no per-image selection in v1).

Two recipes the seam exists for:

- **Content hash** (dedup/evidence): `page_images(data=pdf, decode="container")`,
  then hash each container entry's `data.bytes` with `hashlib.sha256` — the
  bytes are the file's own JPEG (no re-encode), decryption included, so the
  same scan always hashes equal across copies that re-encode nothing.
- **VLM labeling**: `page_images(data=pdf, decode="pixels")` — each raw
  entry names its own stride (`components`/`bpc`/`pixel_format`) and its
  space (`color_space`/`indexed_base`/`samples_are_raw`); a CMYK JPEG stays
  a CMYK JPEG container (the `color_space` names it — convert downstream or
  at decode time is the caller's plan, never the seam's).

## Char-level boxes

`page_text_chars` is the char-level seam behind `page_layout`'s spans
([#206]): per PDF page, the page's per-character boxes from the **cached
span parse** (`extract_spans` → `TextSpan::to_chars()` per span — never
`extract_chars`, which is raw-only and includes content the span path drops
at the crop), a flat list in reading order. The three laws that make it
trustworthy:

- **The join law:** `"".join(c.char for c in chars if c.span_index == s)`
  == `page_layout(p).spans[s].text` **exactly** (both sides iterate the
  same deterministic parse) — `span_index` + `index_in_span` is a complete
  addressing scheme: the chars with `span_index == s` and
  `index_in_span ∈ [i, j)` ground the substring `spans[s].text[i:j]`
  (code points — Python string slicing aligns). The WHOLE-span box is the
  span's own mapped box, never the char union.
- **The anchor law:** the char box is the **advance box from the
  baseline** — `y` is the baseline (the em-box law, the span surface's
  own), `h` is the font size, `w` the char's advance; descenders hang
  BELOW the box's bottom edge.
- **The honesty flags** (span-constant, pinned): `wmode` — a WMode-1
  (vertical-writing) run's char boxes are the engine's flattened fiction,
  displaced one advance LEFT of their origins and stacked along x
  (measured: the span box `(300, 700, -48, 24)` for a column of true ink
  at `(300, 652, 24, 72)`); never pretend `wmode == 1` chars are
  highlight-able. `x_measured` — `True` = the engine's measured origins
  (drift ≈ the 0.5 pt tolerance + sub-point interpolation); `False` = the
  prefix-sum/uniform arm — the TJ-kerning drift class, bounded only by
  measurement (the engine folds Tc/Tw/Tz/TJ into the widths, so measured
  drift on every probe fixture is 0.000; the analytic bound is
  `|Σ TJ adjustments before i| × font_size/1000` pt). A highlight box
  built from `x_measured: false` chars is APPROXIMATE — tighten only via
  the span box (whose own box is engine-built and kerning-correct).

On a page whose rotation the engine applied to a span, the chars are
derived from the span's unmapped box and mapped forward — the mapped
frame's glyph order runs right-to-left while `to_chars` prefix-sums
left-to-right, so the unmapped derivation is the only one that lands the
boxes where the render draws the ink (pre-fix, measured: ~35 pt displaced,
the 'e' box sitting on the 'W').

The substring-highlight recipe: `spans = page_layout(data=pdf)[p].spans`,
`chars = page_text_chars(data=pdf)[p].chars`; find `i = spans[s].text.find(needle)`,
then the highlight box is the union of the char boxes with
`span_index == s` and `index_in_span ∈ [i, i + len(needle))` — exact on the
measured arm, approximate (per the bound above) on `x_measured: false`
chars; the whole-span box is the span's own, never the union.

## Document structure

`document_structure` is the third capability table's surface: one report
whose resolved `format` discriminates and whose `payload` is exactly one of
`DocxStructure` / `XlsxStructure` / `HtmlStructure` (narrow with
`isinstance(report.payload, ...)`). It answers a different question from
everything above — not "how do I render this document" and not "where is
its content on a page", but **who can parse structure at all**: which
formats hand over a parse tree (headings, list state, table grids, cell
values, sheet bounds) as their own engines see it.

| format | structure surface | engine | note |
|---|---|---|---|
| docx (incl. `docm`) | `DocxStructure` — blocks, sections, tables, links, images | office_oxide (raw layer, `as_docx`) | the ladder's convention rung is a markdown superset (below) |
| xlsx (incl. `xlsm`, ZIP) | `XlsxStructure` — sheets, used ranges, typed cells, formulas, merges, defined names | office_oxide (raw layer, `as_xlsx`) | hidden and macro sheets served; chart/dialog tabs named in `chartsheets` |
| html / htm / xhtml | `HtmlStructure` — the node array (the nine kinds), link annotations, table grids | html-to-markdown-rs (the conversion lane's own parse + structure collector) | the conversion output is discarded; the surface returns structure only |
| pdf | refused | — | PDF's structure answer is already two surfaces: `page_layout` + `page_tables` |
| doc / xls / ppt | refused | — | office_oxide's converter-internal IR is not a raw layer (the converter's structure is a rendering choice) |
| pptx (incl. `ppsx`) | refused (v1) | raw layer reachable | "reachable, not built yet" — the message never claims impossibility |
| rtf / odt / ods / odp / epub / csv | refused | — | no parse tree anywhere; the format converts fine ("no structure surface", never "unreadable") |

The refusal families, in order: every refused format refuses on EVERY
backend with a format-level `ValueError` ("document_structure cannot serve
…") fired before the backend check, resolved from `format=`/the 64 KiB
prefix alone — a `path=` + `format="pdf"` call on a 40 MiB file answers
the refusal, never a ceiling error: no body read (the 64 KiB format probe
runs only when the name alone cannot decide). The probe's own byte
budget: an explicit `format=` naming a non-Excel kind consults NO bytes
and touches NO filesystem (a chmod-000 file, a directory, or a missing
path with `format="pdf"` answers the refusal — its `OSError` never
fires, pre-fix the probe opened and read the file it was about to
refuse), and on `data=` the probe sniffs only the bounded 64 KiB prefix,
never the whole buffer (the GIL contract: pre-fix a 128 MiB blob held
the GIL 356.5 ms in the pre-read sniff; the authoritative resolve on the
full bytes stays post-detach). The served office
rows refuse `backend="anydoc"` as a capability (its office lane is
whole-document markdown; the raw layers are office_oxide's). html's forced
backends are conversion's own `UnsupportedBackend` pair, byte-identical. An
OLE container under the Excel kind refuses with the union message — it is a
legacy xls workbook OR an encrypted xlsx package, both CFB, and the seam
cannot disambiguate without a full legacy parse, so it never diagnoses one.
There is NO `password=` parameter: no served format accepts one.

The surface-wide laws:

- **Entries carry container anchors, never boxes.** No coordinates
  anywhere: a docx section index, a cell reference ("B3"), a node id —
  those are the anchors. A caller wanting office or PDF coordinates is on
  `page_geometry`/`page_tables`/`page_layout`.
- **Loss is reported, never skipped.** `DocxStructure.unreadable_parts`
  and `XlsxStructure.unreadable_sheets` carry the raw layers' own
  cannot-parse records verbatim; the intact parts are served.
- **One entry cap, truncate-with-marker.** `MAX_STRUCTURE_ENTRIES`
  (200,000) bounds the payload's entry lists in document order as ONE
  running budget that counts every materialized entry OBJECT: docx
  blocks AND the table cells inside a table block, xlsx cells, html
  nodes AND the grid cells inside a table node — a cell is an entry
  object (pre-fix the cap let a 1M-cell docx table ride inside one
  block and 300k html cells inside one node). The walk stops building
  at zero remaining: the kept prefix stays in document order
  (parents-before-children holds on html; a stopped table keeps its
  grid metadata, its kept cells row-major; cells inside a stopped table
  drop with the table), `entries_truncated: true` marks the affected
  payload, and all sheet/range/merge/name metadata stays complete. It
  bounds PAYLOAD MARSHALLING, not the parse — the engine has already
  materialized everything by the time the cap applies — and it never
  refuses.
- **`format=` is trusted, never re-sniffed** (the conversion lane's own
  precedence): `format="html"` on docx bytes is the html lane's own
  binary-input refusal; `format="docx"` on pdf bytes is the office leg's
  typed parse error; the fix-it is `sniff()`.
- **An explicit `max_bytes=` binds pre-read; `None` keeps the 32 MiB
  default ceiling on every served row.** The conversion lane's ~23×
  measured HTML amplification is a LOWER bound for the structure lane
  (which also holds the node array); the office legs' amplification is
  bounded by the audit + ceiling pair every oxide lane carries.

Per-format disclosures a consumer should read before relying on the fields:

- **docx headings.** The ladder is direct `w:outlineLvl` → the
  stylesheet's `basedOn` chain → the `Heading1..9` style-ID convention →
  the style-NAME convention. Parity with `backend="oxide"` markdown holds
  where heading-ness comes from `w:outlineLvl` and the depth is ≤ 6
  (pinned both ways: one agreeing case, one divergent case); the
  convention rung is a documented SUPERSET (a paragraph only the
  convention rung names a heading renders as plain text on
  `backend="oxide"`), and depths 7–9 exist only in `outline_level` while
  `level` clamps to 6. Headers/footers and text-box prose are not body
  blocks (the markdown lane renders both) — the parity claim excludes
  them. List membership reads the EFFECTIVE properties (style-carried
  `numPr` — the "List Bullet"/"List Number" styles — included); inside
  table cells the table-style LAYER is unreachable (crate-private), so
  cell-level list state misses table-style numPr. A numbered heading
  reports `kind="heading"` AND its `num_id`/`ilvl` (the numPr is
  information; `kind` alone discriminates). Table `vMerge` continue cells
  are emitted as their own cells (inside the Restart span they continue);
  a table block's own `text` is `""` (the cells carry it).
- **xlsx values.** The value is the file's own, discriminated by `type`,
  never coerced: `number`/`string`/`boolean`/`error`/`empty` plus `date` —
  and `date` is ONLY the `t="d"` ISO-8601 branch: an ordinary Excel date
  (a serial number with a date number-format) reports `number`, because
  decoding it needs the stylesheet the raw layer only exposes behind
  `&mut self`. Shared strings are dereferenced to their FULL text (no
  markdown-lane 32,768-char truncation — a disclosed divergence from
  `to_markdown`'s DoS guard). Formulas report text and cached value both,
  labeled (`formula` + `has_cached_value`). The used range is COMPUTED
  from the parsed cells (styled-empty cells included); the declared
  `<dimension>` rides beside it for provenance, never as the source.
- **html.** The node kinds are the NINE the live collector emits
  (`heading`, `paragraph`, `list`, `list_item`, `table`, `image`, `code`,
  `quote`, `group`); an unreachable variant is a typed refusal, never a
  silent `kind=None`. Headings carry their `id` attributes; image nodes
  carry `attributes: None` — **declared image dimensions are NOT served**
  (the collector drops img attributes; the visitor-feature path is the
  v1.x completion). A link's enclosing section is the nearest GROUP
  ancestor (headings have no children — a heading CREATES a Group
  sibling). Nested tables: the inner grid is ALSO collected as its own
  Table node while the outer cell carries the flattened text. Malformed
  HTML and fragments behave exactly as the conversion lane's (the
  structure IS the conversion lane's parse, one parse, no second
  opinion); a fragment without a name stays unresolvable (the pinned
  `looks_like_html` policy), served only through `format="html"` or an
  `.html`-family extension.

## Async and the GIL

`tors.documents.aio` gives the fourteen input-scaling functions (`to_markdown`,
`to_text`, `pdf_extract`, `pdf_page_count`, `pdf_page_labels`, `pdf_security`,
`pdf_classify`, `pdf_link_uris`, `page_geometry`, `page_layout`, `page_tables`,
`page_images`, `page_text_chars`, `document_structure`)
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
