# tors

Fast, GIL-free text operations for Python, backed by Rust — in the spirit of `orjson` for
JSON or `polars` for dataframes.

## Why

Python's `re` module and `str` methods never release the GIL, no matter how large the
input is — a multi-megabyte text transform runs as one long GIL-held call that stalls
every other thread and the asyncio event loop for its whole duration. `tors` does the same
kind of transform as a single native Rust pass, wrapped in `py.detach` (PyO3's GIL-release
call) for the entire computation, so the GIL is free for the rest of your program while it
runs.

## Install

Not yet published to PyPI. For now, build from source:

```sh
pip install maturin
maturin develop --release
```

Once published: `pip install tors`.

## API

```python
import tors

tors.normalize("line one  \n\n\n\nline two\r\n")
# "line one\n\nline two"
```

`tors.normalize(text: str) -> str` applies, in order: Unicode NFC normalization, CRLF/CR
to LF folding, trailing whitespace-before-newline trimming, blank-line-run collapsing
(3+ consecutive newlines become 2), and a final strip — a common preprocessing pipeline
for text extracted from PDFs, OCR, and other messy sources.
