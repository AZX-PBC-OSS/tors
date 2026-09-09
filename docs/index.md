# tors

Fast, GIL-free text operations for Python, backed by Rust, in the spirit of `orjson` for
JSON or `polars` for dataframes.

## Why

Python's `re` module and `str` methods never release the GIL, no matter how large the
input is: a multi-megabyte text transform runs as one long GIL-held call that stalls
every other thread and the asyncio event loop for its whole duration. `tors` does the same
kind of transform as a single native Rust pass, wrapped in `py.detach` (PyO3's GIL-release
call) for the entire computation, so the GIL is free for the rest of your program while it
runs.

## Install

```sh
pip install tors
```

Building from source (a Rust toolchain and [maturin](https://www.maturin.rs/)):

```sh
pip install maturin
maturin develop --release
```

The underlying Rust crate is also on crates.io as `tors-core` (`cargo add tors-core`,
then `use tors::...` — see the [README's Install section](https://github.com/AZX-PBC-OSS/tors#install)
for why the crate and package names differ).

## Quickstart

```python
import tors

tors.normalize("line one  \n\n\n\nline two\r\n")
# "line one\n\nline two"

tors.chunk_text("a long document...", max_chars=500, overlap=50)
# [(0, 500), (450, 950), ...] — word-boundary-aware (start, end) spans, with overlap

tors.find_patterns(["cat", "catalogue"], "the cat sat in the catalogue")
# [(4, 7, 0), (19, 28, 1)] — leftmost-longest, one native pass over every pattern
```

See the [API reference](api.md) for the full behavior of every function, grouped by
what they do (normalization, segmentation, diffing, fuzzy/phonetic matching,
multi-pattern search, chunking, retrieval, and more).

## Documents

Document-format extraction — PDF, the office and text formats, HTML — to GitHub-Flavored
Markdown or plain text, GIL-free, ships as a second wheel: `pip install tors[documents]`.
The base wheel re-exports it as `tors.documents` (with an install hint when the extra is
absent), and the base build carries none of the engine weight. The engine table, the
measured routing rationale, and a quick example are in the
[README's Documents section](https://github.com/AZX-PBC-OSS/tors#documents); the full
reference — `to_markdown`/`to_text`/`sniff`, the PDF family, the typed enums, and
`tors.documents.aio` — is in the [API reference's documents section](api.md#torsdocuments).

## Sync and async

Every function releases the GIL for its native pass — call it directly and the rest of
your program (other threads, the asyncio event loop) stays free while it runs. That is
not the same as not blocking the *calling* coroutine: a native pass invoked directly
from a coroutine still occupies that coroutine's own turn on the event loop for the
call's full wall-clock duration.

`tors.aio` covers exactly the functions where that duration is large enough to matter
(the chunking family, `tf_idf`, `bm25_rank`, `diff_opcodes`, `diff_opcodes_lines`,
`apply_pipeline`, the `normalize`/`finalize` pipeline pair, the
`decode_utf8`/`finalize_utf8`/`decode_utf16`/`b64_encode_bytes`/`b64_decode` byte
codecs, and `truncate_ellipsis`/`strip_controls`) with a plain `asyncio.to_thread`
dispatch, so the event loop stays responsive across the call:

```python
import tors
import tors.aio

# Sync: fine for a script, or for a short call even inside a coroutine.
scores = tors.tf_idf(corpus)

# Async: use inside a service's request path when corpus/document size means
# the native pass could otherwise hold up the event loop for the call's duration.
scores = await tors.aio.tf_idf(corpus)
```

There is no size-based branching inside any wrapper — the choice between the sync
spelling and `tors.aio` is always the caller's, made once at the call site. The rest of
`tors` (short, µs-to-low-ms-scale calls — `normalize`, `find_patterns`, phonetic codes,
and the rest) keeps exactly one, sync, spelling: wrapping those in a thread dispatch
would add real overhead for no benefit. See the
[README's Async use section](https://github.com/AZX-PBC-OSS/tors#async-use) for the
full rationale, the complete `tors.aio` function list, and measured heartbeat-gap
numbers.

## Recipes

Worked, end-to-end pipelines composing several functions together:

- [Ingesting a real document](recipe-ingest.md): arbitrary bytes to clean,
  chunked text.
- [A lightweight retrieval/reranking pipeline](recipe-retrieval.md): a
  small-corpus, no-embeddings search pipeline.
