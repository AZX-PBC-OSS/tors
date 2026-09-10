# Performance

Every number below is a measured cell; the ledgers are
`tests/test_gil_release.py`, `tests/test_performance.py`, and the bench
scripts in `tools/` (corpus construction and methodology included there).

## GIL release

`tors.finalize` over 12 MiB of prose in a background thread holds the asyncio
event loop's worst heartbeat gap to 10-14 ms; the pure-Python pipeline
(`unicodedata.normalize` + `str.replace` + `re.sub` + `hashlib.sha256`) holds
it for 92-108 ms, ~6-12x worse. At 32 MiB: 16-21 ms vs 250-272 ms. Every
function holds the same discipline: one `py.detach` around the whole native
pass, only the argument borrow and the return marshalling under the GIL.

## Fast paths

Already-normalized 12 MiB text returns the original object after a SIMD
sentinel scan, skipping the full pipeline: 2.4-7.7 ms where a full pass costs
100+ ms (16-43x: normalize 123.0 ms → 2.89 ms, finalize 126.3 ms → 7.72 ms,
min-of-5 build vs build).

## Against the real alternatives

Where no stdlib equivalent exists, the comparison is against what you would
otherwise use:

- `diff_opcodes`, 256 KiB of mutated prose: ~4 ms vs `difflib`'s ~3 s (~750x).
- `diff_opcodes_lines`, 12 MiB near-identical pair: 1.9-2.6 ms and 5
  opcodes, where the char-level spelling emits 235 opcodes over 76-88 ms.
- `get_close_matches`, 13,900 candidates: ~65-72 ms vs `difflib`'s ~3.4 s
  (~50x).
- `utf8_is_valid`: up to ~95 GiB/s at 12 MiB, memory-bound above L3 cache;
  the stdlib's only spelling is decode-and-catch.
- `find_patterns`: `pyahocorasick` holds the GIL for its entire scan (no
  `ALLOW_THREADS` anywhere in its scan iterator); `tors` releases it.
- The diffing and fuzzy functions bound their superlinear worst cases with
  `deadline_ms`: a character-level permutation grows ~n² under Myers (50k
  chars 0.32 s, 1M chars 183.6 s unbounded); the deadline turns that into a
  `TimeoutError`.

## Chunking

Chunking costs the segmentation walks it actually consults, not per-codepoint
bookkeeping or levels it never reaches. Measured on 12 MiB of prose
(`tools/bench_chunking.py`):

- `chunk_hierarchical`, default hierarchy, 2000-codepoint budget: ~3.4 ms.
  Only the paragraph walk runs; the eager level builds this replaced measured
  ~350 ms. 500-codepoint budget: ~192 ms (sentence walk joins).
  100-codepoint budget: ~385 ms (all three walks).
- A custom hierarchy that never matches, under a whole-document budget, is
  one codepoint count and nothing else (~2.5 ms): no window consults a level,
  so not even the literal's scan runs.
- Duplicate separators are deduped at list construction, so `[None] * 100`
  costs what `[None]` does (~1.7 ms at a 2000-codepoint budget over 6 MiB)
  and `[" "] * 100` what `[" "]` does (~9 ms). Before that fix the
  per-duplicate spelling was an unbounded, caller-controlled cost
  (`[None] * 100` measured 17.2 s and +3,120 MiB peak RSS; `[" "] * 100`
  790 ms and +1,560 MiB).
- The unconditional `Vec<char>` + grapheme-boundary `HashSet` the chunkers
  used to build up front is gone, replaced by a lazily-built
  one-bit-per-codepoint bitmap (two SIMD scans on pure-ASCII text):
  `chunk_by_words` over 12 MiB went from ~1.9 s and ~500 MiB transient to
  ~160 ms and ~90 MiB.

The line and paragraph twins gained an ASCII `memchr2` fast path on top:
`chunk_by_lines` over 12 MiB of prose at 50 lines/chunk went ~14 ms →
~0.4 ms, and `chunk_by_paragraphs` at 5 went ~10 ms → ~0.5 ms (non-ASCII
keeps the per-codepoint char machine; outputs are differential-pinned
identical either way). The wall contracts gate in `tests/test_performance.py`.

## List returns have a cost at scale

The list-returning functions marshal one tuple per segment under the GIL:
`word_bounds` on 12 MiB of prose (3.67M segments) holds the GIL for 428-497
ms just marshalling the list, roughly 0.72 of the call's wall (the ratio is
the load-stable number; the absolute band moves with the box). The `_iter`
twins (`word_bounds_iter`, `find_patterns_iter`, `chunk_text_iter`, and
friends) exist for exactly this: the same sequence, streamed, each `__next__`
holding the GIL for one tuple, and the full drain ~2.1x faster in wall time
in the measured case.
