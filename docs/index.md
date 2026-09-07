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

Not yet published to PyPI. For now, build from source with a Rust toolchain and
[maturin](https://www.maturin.rs/):

```sh
pip install maturin
maturin develop --release
```

Once published: `pip install tors`.

## Quickstart

```python
import tors

tors.normalize("line one  \n\n\n\nline two\r\n")
# "line one\n\nline two"
```

See the [API reference](api.md) for the full behavior of `tors.normalize`.

## Recipes

Worked, end-to-end pipelines composing several functions together:

- [Ingesting a real document](recipe-ingest.md): arbitrary bytes to clean,
  chunked text.
- [A lightweight retrieval/reranking pipeline](recipe-retrieval.md): a
  small-corpus, no-embeddings search pipeline.
