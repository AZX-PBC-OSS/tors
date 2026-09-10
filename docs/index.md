# tors

Fast, GIL-free text and document operations for Python, backed by Rust, in
the spirit of `orjson` for JSON or `polars` for dataframes: normalization,
Unicode segmentation, diffing, fuzzy and phonetic matching, multi-pattern
search and redaction, chunking, and lightweight retrieval (TF-IDF, BM25,
SimHash, Merkle integrity). An optional extra adds cross-format document
extraction (PDF, Office, RTF/ODF/EPUB, CSV, HTML to markdown or plain text).

## Why

Python's `re` module and `str` methods never release the GIL, no matter how
large the input is: a multi-megabyte text transform runs as one long GIL-held
call that stalls every other thread and the asyncio event loop for its whole
duration. `tors` does the same kind of transform as a single native Rust pass,
wrapped in `py.detach` (PyO3's GIL-release call) for the entire computation,
so the GIL is free for the rest of your program while it runs.

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
`pip install tors[documents]` (see [Documents](documents.md)). The underlying
Rust crate is also on crates.io, published separately as `tors-core` (the
plain `tors` name belongs to an unrelated, dormant crate): `cargo add
tors-core`, then `use tors::...` in code (`[lib] name` in `Cargo.toml` keeps
the importable crate name `tors` regardless of the published package name).

## Quickstart

```python
import tors

tors.normalize("line one  \n\n\n\nline two\r\n")
# "line one\n\nline two"

tors.chunk_text("cats are cute and cats are fun", 12)
# [(0, 8), (8, 17), (17, 26), (26, 30)]: word-boundary-aware (start, end) spans

tors.find_patterns(["cat", "catalogue"], "the cat sat in the catalogue")
# [(4, 7, 0), (19, 28, 1)]: leftmost-longest, one native pass over every pattern
```

All of these run against the built extension; the output above is what they
actually return.

## Where to go next

- [API reference](api.md): every function, full argument contracts, edge
  cases, and examples.
- [Documents](documents.md): PDF/Office/HTML extraction, the engine matrix,
  and the second wheel.
- [Async use](async.md): `tors.aio` and when to reach for it.
- [Performance](performance.md): the measured GIL, wall-time, and throughput
  tables.
- [Design and scope](design.md): the stateless model, the scope cuts, and the
  limitations.
- [Dependencies and licensing](dependencies.md): the permissive-only tree,
  machine-checked.

## Recipes

Worked, end-to-end pipelines composing several functions together:

- [Ingesting a real document](recipe-ingest.md): arbitrary bytes to clean,
  chunked text.
- [A lightweight retrieval/reranking pipeline](recipe-retrieval.md): a
  small-corpus, no-embeddings search pipeline.
- [Chunking threads and transcripts](recipe-transcripts.md): chat threads and
  subtitle tracks without cutting mid-message.

## Sync and async

Every function releases the GIL for its native pass. That is not the same as
not blocking the *calling* coroutine: a native pass invoked directly from a
coroutine still occupies that coroutine's own turn on the event loop for the
call's full wall-clock duration.

```python
import tors
import tors.aio

# Sync: fine for a script, or for a short call even inside a coroutine.
scores = tors.tf_idf(corpus)

# Async: use inside a service's request path when corpus/document size means
# the native pass could otherwise hold up the event loop for the call's duration.
scores = await tors.aio.tf_idf(corpus)
```

`tors.aio` covers exactly the functions where that duration is large enough
to matter, with a plain `asyncio.to_thread` dispatch and no size-based
branching inside any wrapper. The rest of `tors` (µs-to-low-ms-scale calls)
keeps exactly one, sync, spelling: wrapping those in a thread dispatch would
add real overhead for no benefit. See [Async use](async.md) for the full
rationale and the complete function list.

## Contributing

Development setup, the pre-PR checklist, fuzzing, and the release process are
in the repository's
[CONTRIBUTING.md](https://github.com/AZX-PBC-OSS/tors/blob/main/CONTRIBUTING.md).
Security vulnerabilities go to
[SECURITY.md](https://github.com/AZX-PBC-OSS/tors/blob/main/SECURITY.md),
never a public issue.
