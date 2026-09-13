# Cache proof: the UTF-8 view cache's one-directional sharing

The byte-len family's cost model rests on one CPython-internal fact: the
`str` object's cached UTF-8 view is filled by the str-in borrow
(`PyUnicode_AsUTF8AndSize`) and read — never filled — by `encode`.
This file vendors the source links per CPython version so the claim is
checkable rather than asserted.

## What to look for

In every `Objects/unicodeobject.c` from 3.10 through 3.14:

- `unicode_fill_utf8` is the ONLY writer of the `utf8` cache field. It is
  reachable solely from `PyUnicode_AsUTF8AndSize` (the str-in borrow —
  pyo3's `to_str` calls it).
- `unicode_encode_utf8` (the `encode("utf-8")` path) reads the cache when
  present and returns a fresh copy; on a miss it encodes without writing
  anything back.

Reproduce locally with ripgrep against a CPython checkout:

```
rg -n "unicode_fill_utf8" Objects/unicodeobject.c
rg -n "PyUnicode_AsUTF8AndSize|unicode_encode_utf8" Objects/unicodeobject.c
```

The first query's call sites should all sit under `PyUnicode_AsUTF8AndSize`;
the second should show `unicode_encode_utf8` reading `unicode->utf8`
without assigning it.

## Per-version links

(`main` links rot; prefer the version tag matching the interpreter under
test. Observed on CPython 3.12 here; expected from the sources on the rest.)

- 3.10: https://github.com/python/cpython/blob/v3.10.0/Objects/unicodeobject.c
- 3.11: https://github.com/python/cpython/blob/v3.11.0/Objects/unicodeobject.c
- 3.12: https://github.com/python/cpython/blob/v3.12.0/Objects/unicodeobject.c
- 3.13: https://github.com/python/cpython/blob/v3.13.0/Objects/unicodeobject.c
- 3.14: https://github.com/python/cpython/blob/v3.14.0/Objects/unicodeobject.c

Search each for `unicode_fill_utf8` per the ripgrep recipe above.

## Both sharing directions, re-measured in-process

The claim is checked in both directions by the re-measure cells in
`tests/test_performance.py` (the "Both sharing directions, measured"
comment block): fresh 12 MiB objects per lane, min-of-7 — tors-primed
`encode` consults the filled cache (~177 µs, the warm lane), while
encode-primed first `utf8_byte_len` still pays full materialization
(3.85–4.79 ms, the cold class). A prior report of 0.12–0.21 µs "first
call after encode" reproduces only as a cached (repeat-call) timing.

## Contract vs timing

Semantic pins (same object, same answer; fresh-equal-object equality) are
the contract and are asserted in `tests/test_utf8_byte_len.py`. Timing —
which lane is fast on which interpreter — is CPython-internal business:
measured in the wall cells, recorded in the lane tables, never asserted
as a cross-version guarantee.
