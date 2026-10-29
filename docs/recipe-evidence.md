# Recipe: evidence-grounded extraction with page coordinates

The realistic shape of extracting a field from a document and being able to
show *where* it was: the page it lives on, the text it was grounded in, and
the box to draw when a reviewer asks for a highlight. Every step composes
functions documented individually in the [API reference](api.md); this page
shows the pipeline shape and the rules the coordinates carry with them.

The running example: a born-digital PDF with a heading, body text, and a
table. The same flow covers DOCX, XLSX, PPTX, and HTML through
`tors.documents.to_markdown`; the coordinate stages are PDF-only, and the
surface says so per format rather than inventing boxes.

## 1. Convert: the text is the grounding target

Everything starts with the markdown conversion — it is the text later stages
ground against:

```python
from tors.documents import to_markdown

fmt, md = to_markdown("report.pdf")
```

`to_markdown` returns the resolved format (so you never re-sniff) and the
whole-document markdown: headings, lists, tables, and link annotations.
This is the text your extractor sees and quotes.

## 2. Declare the space: `page_geometry`

Before drawing anything, ask what space you are drawing in:

```python
from tors.documents import page_geometry

geoms = page_geometry("report.pdf")
geoms[0]
# PageGeometry(width_pt=612.0, height_pt=792.0, rotation=0, crop=None)
```

One entry per unit (pages for a PDF; sections/slides/worksheets where the
format declares them). The coordinates are **points, bottom-left origin,
y-up** — the space the page's render shows: the visible box (CropBox reduced
to the MediaBox), with width/height swapped for /Rotate 90 and 270.
`rotation` is the turn *already applied*: a viewer must not rotate again.
`crop` is provenance of the extent, not a rectangle to transform.

The None rules: `None` means the unit exists but declares no box (a DOCX
section that states no size); `[]` means zero units; a format with no page
concept answers all-`None` — an XLSX worksheet is not a page, and the surface
will not dress print setup up as document geometry. Only `Some` values are
safe to freeze into a canonical store; the rest stay "unknown".

Scaling to a preview is one formula per axis:

```python
px = pt * dpi / 72          # and back: pt = px * 72 / dpi
left_px = box_x * scale_x
top_px  = (page_h - box_y - box_h) * scale_y   # bottom-left origin
```

## 3. Locate the content: `page_layout`

`page_layout` opens the document once and returns, per page, the geometry
plus text spans, lines with word boxes, and image boxes — all in the same
declared space:

```python
from tors.documents import page_layout

pages = page_layout("report.pdf", pages=[0])
page = pages[0]

line = next(l for l in page.lines if "Q3 revenue" in l.text)
line.bbox      # (x, y, w, h) — y is the line's baseline, the box extends one font-size up
```

Boxes here need two readings and both are pinned by tests: a text box's
bottom edge is the glyph **baseline** (the em-box; descenders hang below),
and a span with a non-zero `rotation_degrees` has an approximate box —
do not use it as a highlight box. Vertical-writing pages report flattened
boxes the rotation flag does not cover. Image boxes are unclipped: the page
box is your clip. When a page has tables, the cells' boxes live on
`page_tables` (below), not here.

Ground the extracted value by **string search in the line texts**, not by
counting characters against `to_markdown`'s output: lines join their words
with single spaces and there is no invariant tying the span list to the
whole-document markdown. The layout text is its own stream; quote from it
and the char offsets are yours to keep alongside the box.

## 4. Tables: `page_tables`

Cell-level boxes come from their own surface, because the text pipeline
re-renders tables inline and their member spans never carry coordinates:

```python
from tors.documents import page_tables

tables = page_tables("report.pdf", pages=[0])
tbl = tables[0].tables[0]
cell = tbl.rows[0].cells[1]

cell.text      # the cell's text
cell.bbox      # (x, y, w, h) in the same declared space
```

Rows run top-to-bottom; `col_count` names the grid's width; a `None` cell
box means the cell has text-positioning but no resolvable box (the
text-positioned path), while ruling-line grids box every cell including
empty ones. The engine only detects tables it can see — a page whose table
was declined (rotated-ruled pages where detection is unsound, or over the
1,500-primitive cap) comes back with `tables_refused: true`, so "no tables"
and "withheld" never share a value.

## 5. A page's name: `pdf_page_labels`

Citations and UIs usually want the label a reader sees, not the ordinal:

```python
from tors.documents import pdf_page_labels

labels = pdf_page_labels("report.pdf")
labels[:3]     # ["cover", "i", "ii"] — or None where the file declares none
```

`None` means "no declared label": the file has no numbering tree, or the
page sits before the first range. Falling back to the ordinal is your
choice; the surface never fabricates one.

## 6. Before the loop: `pdf_security`

For a batch, `pdf_security` answers "can I read this, and what does it
allow" without touching content:

```python
from tors.documents import pdf_security

sec = pdf_security(path)
if sec.encrypted and not sec.unlocked_with_supplied_password:
    ...  # skip, re-queue with a password, or route to a service that has one
```

The other surfaces raise on a locked document (the fail-closed door);
`pdf_security` reports the state instead, so routing decisions happen before
the loop. `readable = not encrypted or unlocked_with_supplied_password or
owner_password_only`. The `/P` permissions are advisory — the PDF format
never enforces them; enforce them in your product if your threat model
cares.

## 7. Office and HTML: structure without coordinates

DOCX, XLSX, and HTML have no per-block page geometry, and none is invented:

```python
from tors.documents import document_structure

report = document_structure("report.docx")
report.payload.docx.headings     # level, text, section anchor
report.payload.xlsx.sheets       # names, state, used range
report.payload.xlsx.sheets[0].cells["B3"]  # typed value, formula, cache
```

The heading levels follow `outlineLvl` and the style chain (clamped 1-6, raw
outline level alongside); list membership comes from the effective
properties; XLSX cells carry typed values plus formula text and cached value
when both exist; HTML returns the heading/link/image/table structure its
parser collects. Refused formats (PDF, PPTX, RTF, ODF, EPUB, CSV) refuse for
every backend — PDF points back at `page_layout`/`page_tables`, PPTX's raw
layer is reachable but not built yet.

## The rules, in one place

- Coordinates are **points, bottom-left origin, y-up**, in the space the
  render shows (rotation and crop applied; `rotation` names the turn —
  never rotate twice).
- Text boxes are em-boxes from the baseline; `rotation_degrees != 0` means
  approximate — not a highlight box; vertical writing is a known gap.
- `page_layout`'s text is its own stream: ground in it, not in the markdown.
- Image boxes are unclipped; the page box is the clip.
- `None` coordinates mean "exists, declares no box" — never guess a Letter
  or A4, and never drop the unit.
- Values are f32; expect ~1e-4 pt quantization.
