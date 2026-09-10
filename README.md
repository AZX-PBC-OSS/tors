# tors

Fast, GIL-free text and document operations for Python, backed by Rust:
normalization, Unicode segmentation, diffing, fuzzy and phonetic matching,
multi-pattern search and redaction, chunking, and lightweight retrieval
(TF-IDF, BM25, SimHash, Merkle integrity). An optional extra adds
cross-format document extraction (PDF, Office, RTF/ODF/EPUB, CSV, HTML to
markdown or plain text). In the spirit of `orjson` for JSON or `polars` for
dataframes.

Full documentation lives at
[azx-pbc-oss.github.io/tors](https://azx-pbc-oss.github.io/tors/) and in
[`docs/`](docs/) in this repo: the [API reference](docs/api.md),
[Documents](docs/documents.md), [Performance](docs/performance.md),
[Async use](docs/async.md), [Design and scope](docs/design.md),
[Dependencies and licensing](docs/dependencies.md), and three worked
[recipes](docs/recipe-ingest.md).

## Why

Python's `re` module and `str` methods never release the GIL, no matter how
large the input is: a multi-megabyte text transform runs as one long GIL-held
call that stalls every other thread and the asyncio event loop for its whole
duration. `tors` does the same kind of transform as a single native Rust pass,
wrapped in `py.detach` (PyO3's GIL-release call) for the entire computation,
so the GIL is free for the rest of your program while it runs.

That covers the functions with a stdlib equivalent (`normalize`'s pipeline
mirrors `unicodedata.normalize` + a few `re.sub` calls; `quote`/`unquote`
mirror `urllib.parse`; `b64_encode_bytes`/`b64_decode` mirror `base64`). Where
the stdlib has no equivalent at all (Unicode text segmentation: grapheme
clusters, word and sentence boundaries; leftmost-longest multi-pattern search;
edit-distance and phonetic matching; content-defined chunking; SimHash
near-duplicate detection; Merkle tree integrity; encoding detection), `tors`
supplies it over maintained Rust crates, GIL-released the same way, so async
services and threaded pipelines don't pay a blocking tax for text work.

## Install

```sh
pip install tors
```

Building from source (a Rust toolchain and [maturin](https://www.maturin.rs/)):

```sh
pip install maturin
maturin develop --release
```

The document-extraction surface is a second wheel behind an extra:
`pip install "tors[documents]"`. From a checkout,
`uv sync --locked --extra documents` builds and installs the payload wheel
from this tree. See [Documents](docs/documents.md).

The underlying Rust crate is also on crates.io, published separately as
`tors-core` (the plain `tors` name belongs to an unrelated, dormant crate).
`cargo add tors-core`, then `use tors::...` in code: `[lib] name` in
`Cargo.toml` keeps the importable crate name `tors` regardless of the
published package name.

### Pyodide / WebAssembly

The Rust cores are OS-free, so `tors` also builds for
[Pyodide](https://pyodide.org) as a PEP 783 `pyemscripten` wheel: the
abi3-py310 story carries over unchanged, so one wheel covers every Pyodide
Python >= 3.10. The `WASM` workflow builds it on every push (artifact only;
publishing to PyPI's Emscripten platform is not wired up). To build one
locally:

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

## Quickstart

```python
import tors

tors.normalize("line one  \n\n\n\nline two\r\n")
# 'line one\n\nline two'

tors.finalize("line one  \n\n\n\nline two\r\n")
# ('line one\n\nline two', 'e986ba083c7c1a9361143d2d8ccd8477d1d5eeef8b94b67c6ad4693f8f7b942a')

tors.diff_opcodes_lines("l1\nl2\nl3\n", "l1\nX\nl3\nl4\n")
# [("equal", 0, 1, 0, 1), ("replace", 1, 2, 1, 2), ("equal", 2, 3, 2, 3),
#  ("insert", 3, 3, 3, 4)]
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

All of these run against the built extension; the output above is what they
actually return.

## What's inside

73 functions plus two small helper classes, grouped by what they do; the
`documents` extra adds seven document-extraction functions and its own helper
types. Full signatures, argument contracts, and edge cases are in the
[API reference](docs/api.md).

| family | functions |
|---|---|
| Unicode normalization & forms | `normalize`, `finalize`, `nfc`/`nfd`/`nfkc`/`nfkd`, `html_unescape`, `strip_controls` |
| UTF-8 / UTF-16 / base64 codecs | `decode_utf8`, `finalize_utf8`, `utf8_is_valid`, `decode_utf16`, `utf16_is_valid`, `b64_encode_bytes`, `b64_decode`, `detect_encoding` |
| Text segmentation (UAX #29) | `grapheme_count`, `word_bounds`(+`_iter`), `word_count`, `sentence_bounds`(+`_iter`), `sentence_count` |
| Diffing (`difflib`-compatible) | `diff_opcodes`, `diff_opcodes_lines` |
| Fuzzy & phonetic matching | `similarity_ratio`, `get_close_matches`, `levenshtein`, `jaro`, `jaro_winkler`, `soundex`, `metaphone`, `double_metaphone`, `nysiis`, `daitch_mokotoff`, `refined_soundex` |
| Multi-pattern search & redaction | `find_patterns`(+`_iter`), `count_matches`, `replace_many`, `replace_many_masked`, `CompiledPatterns` |
| Markdown / code-fence extraction | `extract_code_blocks`, `strip_code_fences`, `dedent` |
| JSON repair (json_repair port) | `repair_json`, `repair_json_loads`, `repair_json_diagnostics` |
| Truncation & lexical grounding | `truncate_to_bounds`, `truncate_ellipsis`, `is_grounded` |
| URL encoding | `quote`, `quote_plus`, `unquote`, `unquote_plus` |
| Text chunking | `chunk_cdc`, `chunk_text`(+`_iter`), `chunk_by_words`/`_sentences`/`_paragraphs`/`_lines`(+`_iter`), `chunk_hierarchical` |
| Information retrieval & integrity | `tf_idf`, `bm25_rank`, `simhash64`, `simhash128`, `merkle_root`, `merkle_diff` |
| Text-processing pipelines | `apply_pipeline`, `CompiledLemmaDict` |
| Document-format extraction (`tors.documents`) | `to_markdown`, `to_text`, `sniff`, `pdf_extract`, `pdf_page_count`, `pdf_classify`, `pdf_link_uris` |

## Highlights

- **GIL release on every call.** 12 MiB of prose through `tors.finalize` in a
  background thread holds the event loop's worst heartbeat gap to 10-14 ms;
  the pure-Python equivalent holds it for 92-108 ms. Full measured tables:
  [Performance](docs/performance.md).
- **Async where it matters.** `tors.aio` wraps exactly the large-input
  functions in `asyncio.to_thread`, unconditionally, with no size-based
  branch; everything else keeps one sync spelling. Details:
  [Async use](docs/async.md).
- **Stateless by design.** No pipeline objects, no caches, no handles to
  manage; the build-once `CompiledLemmaDict` and `CompiledPatterns` handles
  are the measured exceptions, and the scope cuts (no regex engine, no
  bundled lemma data, no search index) are decisions, not oversights.
  Details: [Design and scope](docs/design.md).
- **Permissive-only dependency tree**, machine-checked by `cargo deny` on
  every push. The full table and license accounting:
  [Dependencies and licensing](docs/dependencies.md).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, the pre-PR checklist
(tests, clippy, fmt, ruff, the license gate), fuzzing, release process, and
commit-message conventions. For security vulnerabilities, see
[`SECURITY.md`](SECURITY.md) instead of filing a public issue.
