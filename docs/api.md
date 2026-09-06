# API reference

## `tors.normalize`

```python
def normalize(text: str) -> str: ...
```

Applies, in order:

1. Unicode NFC normalization.
2. `\r\n` and `\r` folded to `\n`.
3. Trailing whitespace (spaces/tabs) before a newline is dropped.
4. Runs of 3 or more consecutive newlines collapse to exactly 2.
5. A final strip, using Python's exact whitespace set (including the `0x1c`-`0x1f`
   separator control characters, which `str.strip()` treats as whitespace but Rust's
   `char::is_whitespace()` does not).

This is a common preprocessing pipeline for text extracted from PDFs, OCR, and other messy
sources.

### GIL behavior

The entire transform runs inside `py.detach` (PyO3's GIL-release call) — the GIL is free
for the rest of your program for the whole call, not just part of it. This is the reason
`tors` exists: Python's `re` module and `str` methods never release the GIL regardless of
input size, so the equivalent pure-Python pipeline holds the GIL for its whole duration no
matter what thread it runs on.

### Example

```python
import tors

tors.normalize("line one  \n\n\n\nline two\r\n")
# "line one\n\nline two"
```
