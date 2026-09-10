# tors

Fast, GIL-free text and document operations for Python, backed by Rust: normalization,
Unicode segmentation, diffing, fuzzy and phonetic matching, multi-pattern search and
redaction, chunking, and lightweight retrieval (TF-IDF, BM25, SimHash, Merkle integrity)
— plus, behind an optional extra, cross-format document extraction (PDF, Office,
RTF/ODF/EPUB, CSV, HTML to markdown or plain text) —
in the spirit of `orjson` for JSON or `polars` for dataframes.

## Why

Python's `re` module and `str` methods never release the GIL, no matter how large the
input is: a multi-megabyte text transform runs as one long GIL-held call that stalls
every other thread and the asyncio event loop for its whole duration. `tors` does the
same kind of transform as a single native Rust pass, wrapped in `py.detach` (PyO3's
GIL-release call) for the entire computation, so the GIL is free for the rest of your
program while it runs.

That covers the functions with a stdlib equivalent (`normalize`'s pipeline mirrors
`unicodedata.normalize` + a few `re.sub` calls; `quote`/`unquote` mirror `urllib.parse`;
`b64_encode_bytes`/`b64_decode` mirror `base64`). Where the stdlib has no equivalent
at all (Unicode text segmentation: grapheme clusters, word and sentence boundaries;
leftmost-longest multi-pattern search; edit-distance and phonetic matching; content-
defined chunking; SimHash near-duplicate detection; Merkle tree integrity; and encoding
detection), `tors` supplies it over maintained Rust crates, GIL-released the same way,
so async services and threaded pipelines don't pay a blocking tax for text work either
way.

## Install

```sh
pip install tors
```

Building from source (a Rust toolchain and [maturin](https://www.maturin.rs/)):

```sh
pip install maturin
maturin develop --release
```

The document-extraction surface is a second wheel behind an extra — `pip install
"tors[documents]"` — with its own layout, engine matrix, and current publication
state in [Documents](#documents), below.
From a checkout, `uv sync --locked --extra documents` builds and installs the
payload wheel from this tree.

The underlying Rust crate is also on crates.io, published separately as `tors-core`
(the plain `tors` name belongs to an unrelated, dormant crate). `cargo add tors-core`,
then `use tors::...` in code — `[lib] name` in `Cargo.toml` keeps the importable crate
name `tors` regardless of the published package name.

### Pyodide / WebAssembly

The Rust cores are OS-free, so `tors` also builds for
[Pyodide](https://pyodide.org) (CPython compiled to WebAssembly) as a
PEP 783 `pyemscripten` wheel — the abi3-py310 story carries over unchanged,
one wheel covers every Pyodide Python ≥ 3.10. The `WASM` workflow builds it
on every push (artifact only; publishing to PyPI's Emscripten platform is not
wired up). To build one locally:

```sh
rustup target add wasm32-unknown-emscripten
uvx --python 3.14 --from pyodide-build pyodide build . -o dist
```

The driving interpreter matters: Python 3.14 targets Pyodide 314.x /
`pyemscripten_2026_0` with a stable Rust toolchain; driving from 3.13 pins a
Rust nightly older than this crate's MSRV. SIMD-dependent dependencies
(base64, simdutf8, memchr, aho-corasick) compile their scalar fallbacks for
wasm; the extension has been smoke-tested (import + representative calls
across every core family) in Pyodide under node.

## What's in it

72 functions plus two small helper classes, grouped by what they do (the `documents`
extra, below, adds seven document-extraction functions and its own helper types).
Each entry is a one-line description; full signatures, argument contracts, and edge
cases are in the [API reference](docs/api.md).

**Unicode normalization & forms**: clean up messy extracted text, or apply a single
normalization form directly.
- `normalize`: NFC + CRLF folding + blank-line collapsing + strip, the common
  PDF/OCR-extraction cleanup pipeline
- `finalize`: `normalize` plus a SHA-256 of the result, in one pass
- `nfc` / `nfd` / `nfkc` / `nfkd`: the four Unicode normalization forms standalone
- `html_unescape`: `html.unescape`, full HTML5 entity table
- `strip_controls`: every C0/DEL control run becomes one space (model-output scrub)

**UTF-8 / UTF-16 / base64 codecs**: validate and decode bytes without holding the GIL
for the whole buffer.
- `decode_utf8` / `finalize_utf8`: UTF-8 decode, and decode+normalize+hash fused
- `utf8_is_valid`: SIMD UTF-8 validity check, no `str` materialized, no exception flow
- `decode_utf16` / `utf16_is_valid`: the same pair for UTF-16, with BOM sniffing
- `b64_encode_bytes` / `b64_decode`: RFC 4648 base64, byte-exact `base64` module parity
- `detect_encoding`: heuristic legacy-encoding guesser (`chardetng`, Firefox's detector)

**Text segmentation**: Unicode-correct boundaries the stdlib has no segmenter for at
all.
- `grapheme_count`: extended grapheme cluster count (UAX #29)
- `word_bounds` / `word_bounds_iter` / `word_count`: word boundaries, list, iterator,
  and count forms
- `sentence_bounds` / `sentence_bounds_iter` / `sentence_count`: sentence boundaries,
  same three forms

**Diffing**: `difflib`-compatible opcodes at native speed.
- `diff_opcodes`: character-level `SequenceMatcher.get_opcodes()` shape
- `diff_opcodes_lines`: the line-level spelling for document/version diffs

**Fuzzy string matching & phonetic matching**: edit-distance and sound-alike matching,
none of which the stdlib ships.
- `similarity_ratio` / `get_close_matches`: `difflib`'s ratio and closest-match search
- `levenshtein` / `jaro` / `jaro_winkler`: classic edit-distance and similarity metrics
- `soundex` / `metaphone` / `double_metaphone` / `nysiis` / `daitch_mokotoff` /
  `refined_soundex`: classic English/Latin-script phonetic codes, five distinct
  mapping tables for the same name-matching/dedup lane

**Multi-pattern search & redaction**: leftmost-longest search and simultaneous replace,
a combination no stdlib or maintained GIL-free binding offers.
- `find_patterns` / `find_patterns_iter` / `count_matches`: multi-pattern search, list,
  iterator, and count forms
- `replace_many`: simultaneous multi-pattern replace, one pass, no re-scanning
- `replace_many_masked`: the same, length-preserving, for offset-safe redaction
- `CompiledPatterns`: a build-once handle for a fixed pattern list, reused across calls

**Markdown / code-fence extraction**: pull structured content out of model output.
- `extract_code_blocks`: every fenced code block, per CommonMark's fence grammar
- `strip_code_fences`: unwrap a whole response wrapped in exactly one fence
- `dedent`: `textwrap.dedent`, byte-exact

**JSON repair**: fix malformed JSON from model output, with or without a schema.
- `repair_json` / `repair_json_loads` / `repair_json_diagnostics`: port of
  json_repair (parity pinned to 0.63.4); syntax repair, schema-guided
  alignment and coercion (dict, bool, or pydantic v2 model as the schema;
  `locale=` for separator conventions), and the repair action log as data

**Truncation & lexical grounding**: fit text to a budget, or sanity-check a claim
against its source.
- `truncate_to_bounds`: cut to a character budget at a word/sentence boundary, never
  mid-grapheme
- `truncate_ellipsis`: hard cut to a character budget plus a `…` marker, never
  mid-grapheme (the DB-column shape)
- `is_grounded`: exact or fuzzy substring check of a claim against its source

**URL encoding**: `urllib.parse`'s percent-encoding quartet, GIL-released.
- `quote` / `quote_plus` / `unquote` / `unquote_plus`

**Text chunking**: split text or bytes for embedding, indexing, or context-window
packing.
- `chunk_cdc`: FastCDC content-defined byte chunking (edit-local, for dedup/sync)
- `chunk_text` / `chunk_text_iter`: character-budget chunking, word/sentence-boundary
  aware, optional overlap
- `chunk_by_words` / `chunk_by_words_iter`: fixed word-count chunks
- `chunk_by_sentences` / `chunk_by_sentences_iter`: fixed sentence-count chunks
- `chunk_by_paragraphs` / `chunk_by_paragraphs_iter`: fixed paragraph-count chunks
  (blank-line heuristic)
- `chunk_by_lines` / `chunk_by_lines_iter`: fixed line-count chunks (blank lines ride
  along, never counted — the log/chat-thread shape)
- `chunk_hierarchical`: priority-ordered fallback chunking (`RecursiveCharacterTextSplitter`
  pattern); `None` entries splice the accurate default hierarchy into a custom one —
  `["\n", None]` is line-first with UAX #29 fallback

**Information retrieval**: lexical/statistical primitives for small-corpus search and
integrity, without an embeddings dependency.
- `tf_idf`: stateless TF-IDF term scoring per document
- `bm25_rank`: Okapi BM25 reranking of a corpus against a query
- `simhash64` / `simhash128`: SimHash near-duplicate fingerprints
- `merkle_root` / `merkle_diff`: domain-separated Merkle root and per-index chunk diff

**Text-processing pipelines**: batch preprocessing without a stateful pipeline object.
- `apply_pipeline`: fused NFD/lowercase/accent-fold/stem/lemma/whitespace-collapse pass
  over a whole text list
- `CompiledLemmaDict`: a build-once handle for a large `lemma_dict`, reused across calls

**Document-format extraction** (`tors.documents`, the `documents` extra — see
[Documents](#documents)): bytes on disk to markdown or plain text, GIL-free,
engine-routed per format family.
- `to_markdown` / `to_text`: any working format (pdf, doc/docx, xls/xlsx, ppt/pptx,
  rtf, odt/ods/odp, epub, csv/tsv, html/xhtml) to GFM markdown or plain text,
  `(format, output)` back, with `pages=` PDF subsets, `password=` for encrypted
  PDFs (fail closed without), and `max_bytes=` as the input budget — an explicit
  value binds every engine lane before a byte is read, `None` keeping the 32 MiB
  default (post-read, the amplifying lanes only) — the document is a `path` or
  in-memory `data=` bytes
  (byte-identical answers, no temp-file roundtrip)
- `sniff`: the content-marker format detector over bytes alone
- `pdf_extract`: per-page plain text plus whole-document markdown, one pass
- `pdf_page_count`: the page tree and nothing else
- `pdf_classify` / `PdfClassification`: the cheap text-vs-image preflight (which pages
  are scans), plus `NeedsOcrError`, the typed route-to-OCR signal
- `pdf_link_uris`: the `/Annots` link walk — every page's link URIs, the raw
  navigation surface no text rendering carries

## Documents

`tors.documents` is document-format extraction: PDF, the office and text formats
(doc/docx, xls/xlsx, ppt/pptx, rtf, odt/ods/odp, epub, csv/tsv), and HTML, converted to
GitHub-Flavored Markdown or plain text — GIL-free the same way everything above is, with
the whole read + sniff + convert pass inside one `py.detach`. It ships as a second wheel:

```sh
pip install tors[documents]
```

The two-wheel split is deliberate: the extraction engines live in `tors-core` behind the
cargo feature `documents`, which is default OFF, so the base `tors` build carries none of
the engine weight — only the payload wheel compiles it. The compiled surface itself is
the payload package `tors_documents`; the base wheel ships only a typed shim,
`tors.documents`, which re-exports the payload's names when it is installed (import
through `tors.documents` — the payload's own name is the wire it rides on, not the
surface to target) and raises the install hint when it is not:

```
ImportError: the documents-extraction surface ships in the tors-documents wheel: install it with `pip install tors[documents]`
```

The payload is version-locked to `tors` (the same number in both `pyproject.toml`s and
both `Cargo.toml`s — release-please bumps all four in one release PR, and the
`publish` workflow asserts the lockstep before uploading). The `publish` workflow
builds and publishes BOTH wheels on a release tag: the base `tors` wheel from the root
manifest, the `tors-documents` payload from its own directory, each uploading through
its own Trusted Publishing identity (the payload's PyPI project needs its trusted
publisher configured once — this repo, workflow `publish.yml`, environment `pypi` —
before the first release; a pending publisher can be set up before any release exists).
Until the first release lands on the index, `pip install tors[documents]` works from a
checkout (`uv sync --locked --extra documents`).

The measured reason for the split (release, linux x86-64, symbols stripped,
2026-09-09): the base extension is 7.4 MiB; the same extension with the documents
feature compiled in — engines, extraction cores, bindings — is 30.1 MiB, 4.1×. Users
who don't extract documents keep the lean download. One known cost of the current
build, stated plainly: the payload `.so` also carries the base surface's compiled code
a second time (`tors-core`'s pyo3 module rides along in its rlib), so a
`tors[documents]` install holds ~37.5 MiB of extensions against the ~30 MiB a single
combined wheel would weigh — an engine-lane cleanup candidate (compile the base
module out when `tors-core` is built as a dependency), not a cost of the split itself.

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

`pages=` selects a PDF page subset on the pdf_oxide lane only (one 0-based page, a
list, or a half-open `(start, stop)` range); the selection is deduped into document
order and a full range is byte-identical to the whole-document conversion. Any other
format, or `backend="anydoc"` on a PDF, refuses `pages=` with `ValueError` rather
than silently converting the whole document.

### The API

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

`path` accepts `str | os.PathLike[str]`. `Format`, `Backend`, and `PageKind` are
`str` enums whose members ARE their accepted strings, so every plain-string call
keeps working and the vocabulary can grow without enum churn in caller code. The
full per-function contract (format resolution, error mapping, the GIL model) is the
native docstrings in `tors-documents/src/lib.rs`, held by
`tests/test_documents_engines.py` and `tests/test_pdf.py`; the user-facing
reference — signatures, the `pages=` and format-resolution rules, the error
taxonomy, `tors.documents.aio` — is the documents section of
[`docs/api.md`](docs/api.md).

- `format=` names the format explicitly (extension spelling, case-insensitive;
  the container variants `docm`/`xlsm`/`ppsx` map onto their parents — the same
  OOXML packages with the content-type override naming the macro/show variant,
  resolved and sniffed as their base kinds; `tsv` is
  accepted as a name and resolves onto the csv kind — the resolved and sniffed
  name is always `"csv"`, never `"tsv"`. `xlsb` is vocabulary sugar only: the
  name routes the Excel kind, but genuine xlsb content is BIFF12 `.bin` sheets,
  not worksheet XML, and is refused — the engines do not read it). `None` sniffs
  the format from content markers
  (the PDF header, the RTF open group, OLE stream names, the ZIP package mimetype,
  the HTML document marker, with a delimiter-agreement heuristic for signature-less
  CSV), the path's extension as the fallback — so a mislabeled file still converts.
  The returned `Format` is what the conversion actually used, and the Excel family
  reports its container honestly: `"xls"` for OLE bytes, `"xlsx"` for a ZIP package,
  regardless of what the explicit name said.
- `sniff(data)` answers the same content question standalone, over bytes alone with
  no path — the mislabeled-download answer. It OPENS AND PARSES THE CONTAINER
  (anydoc's detection reads the ZIP/OLE package metadata; a 120 KiB zip measured
  267 MiB peak RSS to answer docx), so budget it like a parse, not a marker scan.
  `None` is not an error: the content names no format.
- `pdf_extract` returns `(per_page_plain_text, whole_document_markdown)` — one pass
  over one open document, the parse paid once for both outputs. An image-only page
  is an empty string, not an error; routing decisions are the caller's, never
  silently made.
- `pdf_classify` is the cheap text-vs-image preflight: `PdfClassification` carries
  `.page_count`, `.page_kinds` (per page: `"text"`, `"scanned"`, `"image_text"`,
  `"mixed"`, or `"empty"` — a blank page is deliberately distinct from a scan),
  `.pages_needing_ocr` (the 0-based indices of image-only pages, excluding blanks),
  and the derived `.has_text` / `.image_only`. Encrypted documents fail closed on
  every entry (`ValueError` without `password=`).
- Errors: `OSError` for a missing/unreadable file (the matched subclass —
  `IsADirectoryError` on a directory, `FileNotFoundError` for a missing path);
  `ValueError` for an unknown format name, an undetectable file, an unusable
  backend/format pair, the PDF-only family's `backend="anydoc"` capability
  refusal (raised before any work runs — anydoc's PDF surface is the
  `to_markdown`/`to_text` conversion pair, not the per-page probes these calls
  are), an invalid `pages=` selection, a malformed/encrypted document, an input
  over `max_bytes=` (an EXPLICIT budget binds every lane — pdf and HTML
  included, the PDF-only family too — checked before a byte is read or copied;
  the 32 MiB default, post-read, covers the anydoc and office_oxide lanes only;
  on the PDF-only family `None` is unmetered outright, those calls never run
  either metered lane), a non-regular `path=` (FIFO/device/socket — typed,
  naming `path` and the kind, before the read), or a NUL byte inside `path=`
  (CPython's own `open()` convention). `NeedsOcrError` (a `ValueError` subclass
  carrying `.pages` — the 0-based indices needing OCR, the same convention as
  `pages=` and `pages_needing_ocr` — and `.page_count`) is raised only on the
  anydoc PDF lane; the default pdf_oxide lane returns empty output for scanned
  pages and leaves the OCR decision to `pdf_classify`/`pdf_extract`.

### The engine matrix

Four engines, routed per format family by head-to-head measurement (the full
comparison lives in the `documents_impl` crate docs; the suite pinning it is
`tests/test_documents_engines.py`). `backend="auto"` is the measured-best lane;
the forced backends raise on a format their engine cannot read, never a silent
fallback:

| format family | `backend="auto"` | `backend="oxide"` | `backend="anydoc"` |
|---|---|---|---|
| pdf | pdf_oxide 0.3.78 | pdf_oxide | anydoc (pdf-inspector) |
| the PDF-only family (`pdf_extract`/`pdf_page_count`/`pdf_classify`/`pdf_link_uris`) | pdf_oxide | pdf_oxide | `ValueError` — a capability refusal, not a format one (see below) |
| html / htm | html-to-markdown-rs 3.12 | `ValueError` | `ValueError` |
| doc / docx (incl. `docm`) | anydoc 0.2.4 | office_oxide 0.1.10 | anydoc |
| xls / xlsx (incl. `xlsm`) | anydoc | office_oxide | anydoc |
| ppt / pptx (incl. `ppsx`) | anydoc | `ValueError` (ppsx) / office_oxide (pptx) | anydoc |
| rtf | anydoc | `ValueError` | anydoc |
| odt / ods / odp | anydoc | `ValueError` | anydoc |
| epub | anydoc | `ValueError` | anydoc |
| csv / tsv | anydoc | `ValueError` | anydoc |

(`xlsb` is absent on purpose: the name routes the Excel kind as vocabulary sugar,
but genuine xlsb content — BIFF12 `.bin` sheets, not worksheet XML — is refused by
both engines; `ppsx` converts on the auto/anydoc lane and is refused by
office_oxide, which checks the presentation content type — clean refusals, never
silent fallbacks, both pinned in the suite.)

The PDF-only family's row is a capability refusal, not a format one: PDF+anydoc
converts (`to_markdown`/`to_text` with `backend="anydoc"`), but anydoc's entire
PDF surface is whole-document markdown — `to_markdown(bytes)` is the one function
its PDF module exposes (~anydoc-0.2.4/src/formats/pdf.rs), its only per-page
knowledge the NeedsOcr refusal — while the four probe calls need exactly the
surfaces it lacks: the per-page text probe (itself the OCR-routing signal), the
page tree, per-page classification, and the `/Annots` walk. `backend="anydoc"`
on any of the four is a named `ValueError` raised before any work runs, pointing
at the conversion pair. Their `max_bytes=` matches the conversion pair's: an
explicit budget binds before a byte is read or copied; `None` keeps the pdf lane
unmetered (the 32 MiB default is the anydoc/office_oxide lanes' post-read check,
and those PDF-only calls never run either lane).

Why this split: pdf_oxide reads two-column layouts as separate reading-order blocks,
renders `/Link` annotations as `[text](uri)`, and detects oversize-font headings,
where anydoc's PDF engine (pdf-inspector) interleaves columns line-by-line and drops
links; html-to-markdown-rs is the only measured HTML engine that drops
`<script>`/`<style>` by construction; anydoc renders style-based docx headings
(`pStyle` → `#`) and list markers that office_oxide drops entirely, and covers
rtf/odt/epub/csv office_oxide cannot read at all. `backend="oxide"` exists so a
pipeline can diff the two engines' output on its own corpus — office_oxide's one
measured win is exact entity text (no `&`-escaping, which anydoc does and the
plain-text strip normalizes back). Every probe-able cell above was verified against
the committed fixtures in `tests/engines_corpus/` — the OOXML alias containers
included (`docm`/`xlsm`/`ppsx` fixtures are [Content_Types].xml rewrites of the
base generators, converted byte-identically to their bases and pinned as such);
the legacy `doc`/`xls`/`ppt` cells have no Python fixture (OLE compound files are
not hand-generatable without an Office writer) and are pinned by the crate-side
routing-table test (`src/documents_impl.rs`) plus the mutation lane's typed-error
contract — honestly uncovered at the conversion level.

### Async and the GIL

`tors.documents.aio` gives the six input-scaling functions (`to_markdown`,
`to_text`, `pdf_extract`, `pdf_page_count`, `pdf_classify`, `pdf_link_uris`) their
`asyncio.to_thread` twins — same signatures, same discipline as `tors.aio` (an
unconditional thread hop,
never a size-based branch; `sniff` stays sync-only — its cost is a container
parse, but it remains a single short native pass a sync caller runs directly).
One caveat the aio docstrings carry too: `asyncio.to_thread` is uncancellable
mid-pass — a `wait_for` timeout returns control while the thread runs the
conversion to completion, and repeated timeouts pin the default executor's
threads. See the `tors.documents.aio` section of the API reference.

The GIL claim is the point of the payload: every function runs its whole native
pass — file read, format sniff, engine conversion, and `to_text`'s strip — inside
one `py.detach`, with only the argument borrow, validation, and the O(output) return
marshalling under the GIL; the O(n) bytes copy a `data=` call pays rides inside
the detach with the rest of the pass (a 400 MB call's max heartbeat gap measured
~1.1 ms, 2026-09-09). The official pdf_oxide pyo3 wheel measures as GIL-held per
call — worst heartbeat gap 23.6ms on a 9-page document under a 10ms ping (2026-09),
growing with document size — so this payload calls the crate's Rust API directly
under `py.detach` instead. The bands are pinned by name:
`tests/test_documents_engines.py::test_to_markdown_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity`
(a 10ms heartbeat stays live through a ~470KB conversion),
`::test_concurrent_conversions_are_identical` (8 threads convert concurrently,
byte-identical results),
`::test_to_text_on_a_big_csv_keeps_the_event_loop_at_heartbeat_granularity` (the
anydoc amplification lane), and
`tests/test_pdf.py::test_pdf_extract_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity`
(worst gaps 10.7–13.9ms on 130–160ms walls — the marshalling floor; a detach
regression holds the whole wall and fails the budget by an order of magnitude).
The heartbeat cells are `timing`-lane marked (CI's matrix legs deselect the lane,
one 3.12 leg runs it).

## Examples

```python
import tors

tors.normalize("line one  \n\n\n\nline two\r\n")
# 'line one\n\nline two'

tors.finalize("line one  \n\n\n\nline two\r\n")
# ('line one\n\nline two', 'e986ba083c7c1a9361143d2d8ccd8477d1d5eeef8b94b67c6ad4693f8f7b942a')
```

```python
text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
chunks = tors.chunk_by_sentences(text, 2)
# [(0, 44), (44, 90)]

[text[s:e] for s, e in chunks]
# ['This is sentence one. This is sentence two. ', 'This is sentence three. This is sentence four.']
```

```python
thread = (
    "Ana: kickoff at nine.\n"
    "Ben: We briefed the U.S. team on the numbers. They asked for a follow-up.\n"
    "Ana: done."
)
chunks = tors.chunk_hierarchical(thread, 40, ["\n", None])
# [(0, 21), (22, 59), (59, 95), (96, 106)]

[thread[s:e] for s, e in chunks]
# ['Ana: kickoff at nine.',
#  'Ben: We briefed the U.S. team on the ',
#  'numbers. They asked for a follow-up.',
#  'Ana: done.']
```

```python
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "a lazy cat sleeps all day",
    "the fox and the dog are friends",
]
tors.bm25_rank("quick fox", corpus)
# [(0, 1.3162195220480066), (2, 0.4798180901812613), (1, 0.0)]
```

All four run against the built extension; the output above is what they actually
return. For every function's full argument contract, error behavior, and more examples,
see:

- [`docs/api.md`](docs/api.md): the full API reference
- [`docs/recipe-ingest.md`](docs/recipe-ingest.md): bytes to clean, chunked text, end
  to end
- [`docs/recipe-retrieval.md`](docs/recipe-retrieval.md): a small-corpus,
  no-embeddings search pipeline

## Design philosophy and non-goals

`tors` is a stateless library: every call does its own work from scratch, with nothing
cached or built up across calls. That keeps every function simple to reason about and
safe to call from anywhere: no handle to manage, no invalidation to think about, no
surprise from a stale cache.

`CompiledLemmaDict` is the one narrow exception, and it's worth being
precise about why. `tf_idf`, `bm25_rank`, and `apply_pipeline` accept an optional
caller-supplied `lemma_dict`: a `word -> lemma` map. A raw `dict[str, str]` is
re-materialized into a Rust `HashMap` on every call, and for a realistic multi-thousand-
entry lemma table that cost is measured, not theoretical: roughly 1.4ms per call on a
20,000-entry map, independent of how much text the call actually processes, which can
make repeated small calls slower than the equivalent pure-Python loop.
`CompiledLemmaDict` builds that `HashMap` once and hands back an immutable, cheaply
cloned handle, the same `re.compile()` shape the stdlib already uses for a comparable
problem. It exists for exactly this one measured cost and does not reopen the case for
a general pipeline object: nothing else in this library gets a persistent handle.

The scope cuts below are decisions, not oversights:

- **General regex.** `find_patterns`/`replace_many` are leftmost-longest multi-pattern
  literal search, not a regex engine; `chunk_hierarchical`'s `separators` are literal
  strings — or `None`, splicing in the default hierarchy — not patterns.
- **Schema-aware JSON/YAML coercion.** The JSON side moved IN with the
  `repair_json` family: syntax repair of malformed JSON and schema-guided
  alignment/coercion against a JSON Schema are algorithms — a repair
  parser's heuristics plus a standard validator over caller-supplied schema
  data, the shape json_repair itself proved — not a model, so they fit this
  crate's posture. Still out, for the original reason: other schema
  languages, YAML/TOML repair, prompt-side constrained generation (the
  BAML-style problem of steering the model while it writes, rather than
  repairing after), and full JSON-Schema-language tooling — tors repairs
  and validates against schemas; it does not generate them.
- **A bundled lemma dictionary.** `apply_pipeline`/`tf_idf`/`bm25_rank` apply a
  caller-supplied lemma map; `tors` ships no lemma data of its own, because full
  lemmatization needs a per-language dataset or a POS-tagging model, not an algorithm;
  that is outside a text-operations library's job.
- **A persistent search index.** `bm25_rank` recomputes corpus statistics from scratch
  on every call: the right shape for reranking a small, already-retrieved candidate
  set, the wrong shape for querying a large corpus repeatedly. Reach for a real search
  engine (`tantivy`, in Rust) for that; `tors` does not build or expose index objects.
- **A bespoke coroutine API.** Every function already releases the GIL for its native
  pass, so the async surface is one thread dispatch per call (`tors.aio`, below)
  rather than a purpose-built event-loop integration.
- **A general persistent pipeline object.** `apply_pipeline` re-describes and re-applies
  its steps on every call rather than compiling a reusable pipeline handle: see
  `CompiledLemmaDict` above for the one measured exception.

Beyond scope, a few limitations are worth stating plainly rather than glossing over:

- **SimHash is not cryptographic.** It's a fast, uniformly-spreading voting hash, not a
  security primitive: two unrelated documents can coincidentally land close together,
  especially on short text, and there's no universal "near-duplicate" distance
  threshold; calibrate per deployment.
- **`soundex`/`metaphone` are English/Latin-script-oriented.** Both pre-filter input to
  ASCII letters; accented and non-Latin characters are dropped, not encoded.
- **Chunking makes no retrieval-quality promise.** Every chunker guarantees a mechanical
  contract (correct boundaries, genuine overlap when requested); none of them promises a
  particular chunk size or strategy helps any downstream model.
- **Segmentation is rule-based UAX #29 only.** No dictionary segmentation for spaceless
  scripts (Thai, Khmer, Burmese, Japanese word breaks are a different, dictionary-based
  problem).
- **`tf_idf`/`bm25_rank` make no relevance claim.** Both are correctly implemented,
  well-specified ranking formulas; neither promises retrieval quality for any
  particular corpus or query.

## Async use

Releasing the GIL is not the same as not blocking: a native pass called directly from a
coroutine still occupies that coroutine's own turn on the event loop for the call's full
wall-clock duration. `tors.aio` is the pre-wired fix for the functions where that matters:
`await tors.aio.tf_idf(corpus)` runs the native pass in a worker thread via
`asyncio.to_thread`, and the event loop stays responsive for its whole duration.

It covers only the large-input-shaped functions (the chunking family, `tf_idf`,
`bm25_rank`, `diff_opcodes`, `diff_opcodes_lines`, `apply_pipeline`, the
`normalize`/`finalize` pipeline pair, the `decode_utf8`/`finalize_utf8`/
`decode_utf16`/`b64_encode_bytes`/`b64_decode` byte codecs, and
`truncate_ellipsis`/`strip_controls`), not all of
`tors`. Thread dispatch costs on the order of tens of microseconds: noise next to a
millisecond-or-slower native pass over a real corpus or document, real overhead next to a
microsecond-scale call over a short string. Wrapping every export would make the small,
common calls slower through this module than through the plain sync spelling, for no
benefit, so the rest of `tors` keeps exactly one spelling: call it directly from a
coroutine when the input is small enough that the whole thing finishes in microseconds.

There is no size-based branch inside any wrapper, and there never will be: a function
that sometimes runs inline and sometimes hops to a thread depending on its input is
unpredictable from the caller's side and can silently block the loop when the heuristic
misjudges. Every function in `tors.aio` always dispatches through `asyncio.to_thread`,
unconditionally; the choice between the sync spelling and `tors.aio` is the caller's,
made once at the call site, not a runtime guess. `tests/test_aio.py` pins this
structurally (no branch in the wrapper body) as well as behaviorally (a heartbeat
coroutine keeps ticking with worst gaps well under the call's own wall during a large
`diff_opcodes` await).

The streaming iterator constructors (`word_bounds_iter` and siblings, including the
chunking family's own `chunk_text_iter`/`chunk_by_words_iter`/`chunk_by_sentences_iter`/
`chunk_by_paragraphs_iter`/`chunk_by_lines_iter`) have no async twin: an iterator is
not an awaitable shape, and
draining one to a list inside a worker thread is exactly what the already-covered
list-returning sibling does.
The eager construction pass is the GIL-released part anyway, so the manual
`await asyncio.to_thread(lambda: list(tors.word_bounds_iter(text)))` covers the
streaming shape when it is genuinely needed. Signatures are identical to the sync
spellings, pinned by `tests/test_aio.py`; the stub `aio.pyi` is generated by
`tools/gen_aio_stub.py`.

## Performance

Full measured tables (GIL heartbeat gaps under `asyncio`, wall-time races against the
stdlib and `difflib`, and criterion throughput) live in the test suite itself
(`tests/test_gil_release.py`, `tests/test_performance.py`) so every number stays
reproducible and re-runnable: the summary here gives the shape, and the test files
are the ledger.

**The GIL release is the headline.** Running `tors.finalize` on 12 MiB of prose in a
background thread holds the asyncio event loop's worst heartbeat gap to 10–14 ms; the
equivalent pure-Python pipeline (`unicodedata.normalize` + `str.replace` + `re.sub` +
`hashlib.sha256`) holds it for 92–108 ms in the same thread placement: roughly 6–12×
worse, and it blows past this project's own CI budget in every sample. At 32 MiB the gap
widens further (17–21 ms vs 250–272 ms).

**Already-normalized input is close to free.** A quick-check fast path means
`normalize`/`finalize`/`nfc`/`nfkd` on already-clean 12 MiB text return the original
object with only a SIMD sentinel scan: 2.4–7.7 ms where a full pass costs 100+ ms
(28–45× faster).

**Chunking at document scale costs the segmentation walks it actually consults, not
per-codepoint bookkeeping or levels it never reaches.** `chunk_hierarchical` builds
each separator level at its first consultation, so over 12 MiB of prose with the
default paragraph→sentence→word hierarchy and a 2000-codepoint budget — where every
window is answered at the paragraph level — only the paragraph walk runs: ~3.4 ms,
against ~350 ms for the eager level builds it replaced, which scanned its word walk
(~130 ms) plus sentence walk (~190 ms) up front whatever the budget asked. Duplicate
separators are deduped at slot construction — `None` and repeated literals both — so
`[None] * 100` costs what `[None]` does (~0.5 ms at a 2000-codepoint budget over
6 MiB; the per-duplicate spelling measured 17.2 s and +3,120 MiB of peak RSS) and
`[" "] * 100` what `[" "]` does (~9 ms; was 790 ms and +1,560 MiB). A custom
hierarchy whose separators never match, under a whole-document budget, is one
codepoint count and nothing else (~0.15 ms): no window consults a level, so not even
the literal's scan runs. The line and paragraph twins are byte-level scans — an
inline density window before each `memchr2` hop so break soup never pays a hop, and
a sliding ASCII certificate that batches purity checks one 4 KiB stride per ~50
segments — so `chunk_by_lines` over 12 MiB of prose at 50 lines/chunk runs in
~0.5 ms (formerly ~8 ms), ~2.3 ms on a one-line-per-~80-bytes log corpus (formerly
~8.3 ms), and `chunk_by_paragraphs` ~1.4 ms on that log corpus (formerly ~7.5 ms);
break soup is at parity with the old decoder, and non-ASCII segments keep the byte
path through a per-segment fallback (outputs differential-pinned identical). These
functions used to build a
`Vec<char>` of the whole text plus a `HashSet` of every grapheme boundary —
unconditionally, before any early exit could matter — which dominated their cost and
grew superlinearly; that machinery is now a lazily-built one-bit-per-codepoint bitmap
(two SIMD scans on pure-ASCII text, no segmentation walk), differential-pinned to the
old behavior, with `chunk_by_words` carrying the same history (12 MiB: ~1.9 s →
~160 ms). The measurement cells live in `tools/bench_chunking.py` and the
`chunk_hierarchical`/`chunk_by_segment` criterion groups; the wall contracts gate in
`tests/test_performance.py`.

**Where no stdlib equivalent exists, the comparison is against the real alternative.**
`diff_opcodes` diffs 256 KiB of mutated prose in ~4 ms against `difflib`'s ~3 seconds
(~750×); `get_close_matches` against 13,900 candidates runs in ~65–72 ms against
`difflib`'s ~3.4 s (~50×); `find_patterns` fills the gap left by `pyahocorasick`, which
holds the GIL for its entire scan by design (no `ALLOW_THREADS` anywhere in its scan
iterator); `utf8_is_valid` has no stdlib boolean primitive to race at all, so its record
is absolute throughput: up to ~95 GiB/s at 12 MiB, memory-bound above L3 cache.

**List-returning functions have a real, disclosed cost at scale.** `word_bounds` on
12 MiB of prose (3.67M segments) holds the GIL for 328–344 ms just marshalling the
returned list (measured worst-gap band; the absolute band moves with box pace — the
dev box the test ledger records measured 428–497 ms, and the load-stable constant is
the ratio, ~0.72 of the call's wall): a genuine cost of the list shape, not a bug.
The `_iter` twins
(`word_bounds_iter`, `chunk_text_iter`, `find_patterns_iter`, and friends) exist for
exactly this: the same sequence, streamed, with each `__next__` holding the GIL for one
tuple instead of the whole list at once, and 2.1× faster in wall time as well, in the
measured case.

Every number above traces to a specific measured cell; see the linked test files for
methodology, corpus construction, and the full per-function tables.

## Dependencies and licensing

The license gate is `cargo deny check licenses advisories bans` (`make deny`; the same
three checks run in CI's lint job). The allowlist in `deny.toml` is permissive-only:
MIT, Apache-2.0, BSD-2/3-Clause, ISC, Zlib, plus four documented additions, all
permissive grants inside the gate's intent:
`Apache-2.0 WITH LLVM-exception` (target-lexicon, a pyo3 build dependency: the
LLVM-exception *removes* attribution obligations from Apache-2.0), `Unicode-3.0`
(unicode-ident, the permissive license the Unicode Consortium publishes the UCD data
under), `0BSD` (enum-iterator/enum-iterator-derive, `soundex`/`metaphone`'s
`rphonetic` dependency's own dependencies: the BSD Zero Clause License is
OSI-approved and public-domain-equivalent, strictly more permissive than plain
MIT), and `MIT-0` (borrow-or-share, a fluent-uri dependency pulled in by the
jsonschema crate's `referencing` $ref machinery, the json_repair port's validation
engine: MIT No Attribution is plain MIT with the attribution obligation removed,
the same family as 0BSD — recorded in `deny.toml`).
No GPL/LGPL/AGPL/MPL, no unlicensed. Dual/tri-licensed crates are consumed via
an allowed branch: notably r-efi (a getrandom dependency, dev tree only) offers
`LGPL-2.1-or-later` as one branch of `MIT OR Apache-2.0 OR LGPL-2.1-or-later`; tors
consumes it under MIT/Apache and the LGPL branch is never elected, which is exactly
what cargo-deny's SPDX expression evaluation verifies.

Direct dependencies (the full transitive closure is machine-checked by the gate; the
dev tree, criterion and friends, is included in the check):

| crate | version | license | role |
|---|---|---|---|
| pyo3 | 0.29.2 | MIT OR Apache-2.0 | the CPython extension layer (abi3-py310) |
| unicode-normalization | 0.1.25 | MIT OR Apache-2.0 | NFC/NFD/NFKC/NFKD tables (Unicode 16.0.0) |
| unicode-segmentation | 1.13.3 | MIT OR Apache-2.0 | UAX #29 grapheme/word tables (Unicode 17.0.0) |
| sha2 | 0.10.9 | MIT OR Apache-2.0 | finalize's SHA-256 |
| const-hex | 1.19.1 | MIT OR Apache-2.0 | digest hex encoding |
| base64 | 0.23.1 | MIT OR Apache-2.0 | RFC 4648 encode/decode core, `simd-unsafe` feature enabled (the crate's own AVX2/NEON kernels, runtime-detected with a scalar fallback: already-shipped, widely-exercised unsafe code upstream, not written in tors) |
| memchr | 2.8.3 | Unlicense OR MIT | SIMD sentinel scans |
| simdutf8 | 0.1.5 | MIT OR Apache-2.0 | SIMD UTF-8 validity scan |
| similar | 3.2.0 | Apache-2.0 | Myers diff engine for diff_opcodes |
| aho-corasick | 1.1.5 | Unlicense OR MIT | leftmost-longest multi-pattern search engine |
| rs_merkle | 1.5.0 | Apache-2.0 OR MIT | merkle_root/merkle_diff's tree structure (domain-separated SHA-256 Hasher supplied by tors, see merkle_impl.rs) |
| fastcdc | 5.0.0 | MIT | chunk_cdc's FastCDC 2020 content-defined chunking |
| chardetng | 1.0.0 | Apache-2.0 OR MIT | detect_encoding's guesser (the engine Firefox ships) |
| encoding_rs | 0.8.35 | (Apache-2.0 OR MIT) AND BSD-3-Clause | the `Encoding` type chardetng's guess returns: already pulled in transitively by chardetng; named directly only to call `.name()` on it, no new package in the tree (the BSD-3-Clause conjunct is the WHATWG Encoding Standard data files' grant, inside the gate's BSD-3 allowance) |
| rust-stemmers | 1.2.0 | MIT OR BSD-3-Clause | tf_idf's/bm25_rank's opt-in Snowball stemmer= knob (18 languages): pulls in serde/serde_derive as a non-optional dependency (an `Algorithm` enum derive, unused by tors's own call sites); recorded here because it is the one real transitive-weight addition in this table |
| rphonetic | 4.0.0 | Apache-2.0 | soundex/metaphone's phonetic-code algorithms (an Apache Commons Codec port): pulls in enum-iterator/enum-iterator-derive (0BSD, the license-gate addition noted above), nom, and thiserror; tors pre-filters every input to ASCII letters before calling into it, working around a real, verified panic in the crate's own Soundex/DoubleMetaphone encoders on ordinary accented input (see the API docs) |
| jsonschema | 0.55 | MIT OR Apache-2.0 | json_repair's schema-guided validation engine: the mature Rust JSON Schema validator (drafts 4-2020-12), maintained by a core maintainer of Python's own jsonschema; local `#/...` refs only (default-features off — its remote-$ref resolvers would pull a TLS stack); pulls the num family via its exact-rational multipleOf arithmetic (fraction) — the tree's one real transitive-weight addition, the rust-stemmers precedent |
| serde_json | 1 | MIT OR Apache-2.0 | the JSON interchange type the validator works over; already in the tree as criterion's transitive, so the direct edge adds no new package (the aho-corasick/encoding_rs precedent) |
| regex | 1 | MIT OR Apache-2.0 | json_repair's single-number extraction grammars (tier-3 prose/currency/percent tokens and the tier-4 separator readings); already in the tree transitively (aho-corasick/memchr elect it via other consumers), so the direct edge adds no new package (the same precedent) |
| jiff | 0.2 | MIT OR Unlicense | json_repair's date/time normalization engine (`format: date`/`date-time`/`time`): calendar + timezone-instant math from the datetime crate the Rust ecosystem's own docs point at (the memchr Unlicense-election precedent); default-features off, std only — no TZDB backend, the accept-list shapes need none |
| pdf_oxide *(optional, `documents`)* | 0.3.78 | MIT OR Apache-2.0 | the documents payload's PDF engine (two-column reading order, link annotations, headings); default features only, and the caret bounds the 0.x line — see Cargo.toml's own comment for the measured rationale |
| anydoc *(optional, `documents`)* | 0.2.4 | MIT | the payload's office/text engine (doc/docx, xls/xlsx, ppt/pptx, rtf, odt/ods/odp, epub, csv) |
| office_oxide *(optional, `documents`)* | 0.1.10 | MIT OR Apache-2.0 | the payload's caller-selectable `backend="oxide"` lane; already compiled in via pdf_oxide's tree, so the direct edge adds no new package |
| html-to-markdown-rs *(optional, `documents`)* | 3.12 | MIT | the payload's HTML engine; default-features off (its optional HTTP/MCP stack stays out) |
| criterion *(dev)* | 0.8.2 | Apache-2.0 OR MIT | the benchmark harness |
| strsim *(dev)* | 0.11.1 | MIT | differential oracle for levenshtein/jaro/jaro_winkler tests |

Maintenance note (the spec's module decision): `simdutf8`'s
release line has been quiet since 2024-09-22 (0.1.5) while the repository
itself stays active (commits into 2026-06): a mature, zero-dependency
implementation of an algorithm that does not churn (UTF-8 validation), picked
with that fact known and disclosed.

Maintenance note on `similar` (the same decision): Apache-2.0 only, actively
maintained by mitsuhiko (the Flask author), and the diff engine behind insta,
a crate with a large existing consumer base; no Python binding for it exists, so
tors binds it directly for `diff_opcodes`.

Maintenance note on `aho-corasick` (the same decision): dual `Unlicense OR
MIT`: the MIT branch is elected and recorded here (cargo-deny's SPDX
expression evaluation verifies exactly that election, the same mechanism that
handles memchr's `Unlicense OR MIT`, the same spelling). By BurntSushi, and
the Aho-Corasick engine inside Rust's own `regex` crate: the most
battle-tested implementation of this exact algorithm in the Rust ecosystem;
it was already in the lock as a transitive dependency (criterion's regex) before
find_patterns made it direct, so the dependency tree grew by zero packages.

Transitive closure at the current lock state (325 `Cargo.lock` entries
including tors-core itself, i.e. 324 dependency packages incl. dev and the
documents engine tree, re-derived with `cargo metadata --all-features` over
the current lock): 176 `MIT OR Apache-2.0`, 60 MIT (fastcdc, strsim, and
anydoc among them), 18 `Apache-2.0 OR MIT` (chardetng, autocfg), 12
`MIT/Apache-2.0` (version_check, winapi, siphasher) plus 2 `Apache-2.0/MIT`
(rs_merkle, bytecount) and 1 `Apache-2.0 / MIT` (fnv) — three more spellings
of the same dual grant, 10 `Unlicense OR MIT` (aho-corasick, memchr, jiff)
and 4 `Unlicense/MIT` (csv, same-file, walkdir), 8 Apache-2.0 (rphonetic,
soundex/metaphone's crate, among them), 3 `Apache-2.0 WITH LLVM-exception OR
Apache-2.0 OR MIT` (wasip2, wit-bindgen), 3 `Zlib OR Apache-2.0 OR MIT`
(tinyvec, bytemuck), 3 `MIT OR Apache-2.0 OR Zlib` (tinyvec_macros, the
zune image crates) and 2 `MIT OR Zlib OR Apache-2.0` (miniz_oxide, both
versions), 3 Zlib (foldhash, slotmap, zlib-rs — engine-tree arrivals), 2
`BSD-2-Clause OR Apache-2.0 OR MIT` (zerocopy), 2 `BSD-3-Clause OR
Apache-2.0` (moxcms, pxfm, the engines' color management), 2 BSD-3-Clause
(the brotli alloc pair), 2 `MIT OR Apache-2.0 OR LGPL-2.1-or-later` (r-efi,
both major versions, the tri-license noted above), 2 `0BSD`
(enum-iterator/enum-iterator-derive, rphonetic's own dependencies: the
license-gate addition above), and 1 each of the singles: `0BSD OR MIT OR
Apache-2.0` (adler2, a miniz_oxide dependency in the engine tree), `MIT-0`
(borrow-or-share), `Apache-2.0 WITH LLVM-exception` (target-lexicon), `(MIT
OR Apache-2.0) AND Unicode-3.0` (unicode-ident), `(Apache-2.0 OR MIT) AND
BSD-3-Clause` (encoding_rs, a direct dependency), `CC0-1.0` (tiny-keccak,
the documents-tree addition), `Apache-2.0 OR BSL-1.0` (ryu, the csv crate's
float formatter in anydoc's tree — the Apache-2.0 branch elected), `BSD-3-Clause AND MIT`
(brotli) and `BSD-3-Clause/MIT` (brotli-decompressor), and
`MIT/BSD-3-Clause` (rust-stemmers, a fourth spelling of the same dual-grant
idea). Every one satisfies the allowlist.

The count above is the current lock state, json_repair and the documents
engine tree both in. The json_repair port's four direct dependencies grew
the runtime package closure from 55 to 102 (jsonschema's draft-4-2020-12
validation tree is the addition; serde_json
and regex were already in the lock as transitives, and jiff brings one small
crate, so the real addition is jsonschema's tree); the engine tree is the
next delta on top, inside the count through the all-features gate
resolution below. The gate re-checks every new entry against the allowlist
on every run — the MIT-0 license it flagged on the way in (borrow-or-share)
is recorded above and in `deny.toml`. The four `documents` engines above are
the same story one feature later: feature-gated (default OFF), so they are
absent from the base wheel's build and present in the payload's. One gate
fact stated exactly (verified 2026-09-09): the wired gate — `make deny` and
CI's cargo-deny step — runs at the repo root and covers the FULL engine
tree, because `deny.toml`'s `[graph] all-features = true` resolves every
cargo feature of the workspace into the checked graph, `documents`
included: the root lock's 325 entries carry pdf_oxide/anydoc/office_oxide/
html-to-markdown-rs and their transitive trees, resolution is
metadata-only (nothing links), and `cargo deny check licenses advisories
bans` at the root passes over all of them. The payload-manifest invocation
— `cargo deny --manifest-path tors-documents/Cargo.toml check licenses
advisories bans` — also passes at the current lock state (verified
2026-09-09), but it is a redundant subset check, not the engine tree's
only gate: every crate the payload's own graph builds is already present
in the root graph (`deny.toml`'s `[graph]` comment records the same). The
two documents-tree entries the engine tree forced into `deny.toml`:
`CC0-1.0` is allowlisted (`tiny-keccak` 2.0.2, public-domain-equivalent,
pulled by html-to-markdown-rs's `compile-time-rng` ahash feature — it is
not in the base graph), and RUSTSEC-2026-0192 (`ttf-parser` unmaintained,
an unconditional transitive of both pdf_oxide's and anydoc's PDF stacks)
is ignored with its reason recorded in `deny.toml`.

## Development

The `fuzz/` crate drives the `*_impl.rs` cores with raw adversarial bytes via
cargo-fuzz (libFuzzer): a bug class the hypothesis-based Python tests cannot
reach, since they shape input around documented contracts rather than raw
bytes. `cargo fuzz run <target> -- -max_total_time=30` runs one target
briefly (nightly toolchain and `cargo install cargo-fuzz --locked`
required); targets assert the same invariants the Python gates pin, at
raw-byte depth; crashes are minimized with `cargo fuzz tmin`. `make
fuzz-quick` runs every target for 30s each (the full 17-target set); CI runs
subsets of that set — the `fuzz-smoke` job 15 targets at 15s each on every
push, the weekly `fuzz` workflow 14 of them at 30s each — so the full set is
a local `make fuzz-quick`. See
[CONTRIBUTING.md](CONTRIBUTING.md#fuzzing) for the full setup.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, the pre-PR checklist (tests, clippy,
fmt, ruff, the license gate), and commit-message conventions.
