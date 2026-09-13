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
- `find_unescaped`/`contains_unescaped`: the escape-parity scan answers "is
  this `\u0000` a real NUL or the literal text?" straight from raw serialized
  JSON in one detached pass — ≈0.26 ms at 12 MiB with no occurrence
  and ≈0.70 ms over a 72,520-rejected-hit false-positive corpus (box- and
  load-dependent; the wall cells in tests/test_unescaped_scan.py, same
  order as the inline band in tests/test_gil_release.py), where the
  hand-rolled find-and-count-backslashes loop it replaces takes 13 ms (and
  the confirm-by-re-parse guard the algorithm was lifted from pays a full
  parse plus a recursive walk per prefilter hit).
- `utf8_byte_len`: `len(s.encode("utf-8"))` allocates the full `bytes`
  object just to count it; `tors` reads the count off the borrowed UTF-8
  view — flat ~0.1 µs from 1 KiB to 12 MiB on ASCII (compact ASCII is its
  own UTF-8, a zero-copy alias; the expression pays ~0.9 µs at 64 KiB, the
  TaskQ result-cap size, and ~180 µs at 12 MiB) and ~0.1 µs on repeat calls
  over a cached non-ASCII object, where even the warm expression pays a full
  copy out of the same cache (~196 µs at 12 MiB). The honest lanes,
  recorded: a fresh non-ASCII object's first call — the cold-cache case — is
  encode-parity (the cache materialization IS an encode — the encoder pass
  plus a malloc plus a second memcpy, ~4.7 ms at 12 MiB against a cold
  encode's ~4.9 ms), 1 KiB is a dead heat (pure call overhead on both
  sides), and the cache sharing with `encode` is one-directional: the
  str-in borrow fills the cache — `encode` then reads it, a ~184 µs copy at
  12 MiB instead of ~4.9 ms — but a prior `encode` fills nothing, so the
  first `utf8_byte_len` after an encode still pays the full materialization
  (measured ~3.8-4.8 ms at 12 MiB; observed on CPython 3.12 here, expected
  from the sources on 3.10–3.14 — see `docs/cache-proof.md` for the
  per-version `Objects/unicodeobject.c` links. Semantic pins are the
  contract; timing is not). That first-call
  materialization is the function's one GIL-held O(n) pass (~5 ms at
  12 MiB, under the heartbeat interval); the full lane table is in
  `tests/test_performance.py` (both sharing directions re-measured
  in-process), the criterion core-vs-copy group in
  `benches/search.rs` (`core` ~0.5 ns flat against the `memcpy_floor`'s
  ~65 GiB/s bench artifact on the calibration box).
- `utf16_byte_len`: the interop twin — `len(s.encode("utf-16-le"))`
  allocates and encodes the full 2n `bytes` object just to count it;
  `tors` derives the count from the borrowed UTF-8 view (2 bytes per
  codepoint plus 2 more per astral codepoint, both counts byte classes)
  in one chunked pass with no allocation (bench artifact on the
  calibration box, arm64 rustc release: ~30 GB/s; re-measure on your
  target): ~2.2 µs at 64 KiB
  against the expression's ~9-10 µs, ~34 µs at 1 MiB against ~145 µs,
  ~400 µs at 12 MiB against ~1.7 ms — ratios 0.21-0.26 on every warm
  lane, both corpus kinds (the scan is representation-independent).
  The one lane the expression wins, recorded: a fresh non-ASCII
  object's first call pays the borrow's UTF-8-cache materialization
  (the utf8 twin's cold class) before the scan, while the utf-16
  expression never touches UTF-8 — 376 µs against 146 µs at 1 MiB, the
  trade buying every later call at 4-5x and no 2n allocation per call;
  one-shot counts of fresh non-ASCII strings are not the recommended
  lane. The lane table is in `tests/test_performance.py` (including the
  fresh-object end-to-end bench), the criterion group in
  `benches/search.rs` (the chunked scan against the `rust_utf16_shape`
  baseline; the utf8 group races `core` against the `memcpy_floor`).
- The diffing and fuzzy functions bound their superlinear worst cases with
  `deadline_ms`: a character-level permutation grows ~n² under Myers (50k
  chars 0.32 s, 1M chars 183.6 s unbounded); the deadline turns that into a
  `TimeoutError`.

## Chunking

Chunking costs the segmentation walks it actually consults, not per-codepoint
bookkeeping or levels it never reaches. Measured on 12 MiB of prose
(`tools/bench_chunking.py`, `tests/test_performance.py`):

- `chunk_hierarchical`, default hierarchy: a whole-document budget consults
  no level at all and costs ~0.1 ms whatever the hierarchy (formerly ~340 ms:
  the walks were built before any window asked). A 2000-codepoint budget
  whose windows are all served by the paragraph level costs ~1.2 ms
  (formerly ~350 ms: the eager build paid the sentence and word walks no
  window consulted). Budgets that genuinely descend pay the walks they use:
  ~190 ms at a 600-codepoint budget (paragraph plus sentence), ~400 ms at
  100 (all three walks plus a denser chunk loop).
- A custom hierarchy that never matches, under a whole-document budget, is
  one codepoint count and nothing else: no window consults a level, so not
  even the literal's scan runs.
- Duplicate separators are deduped at slot construction, so `[None] * 100`
  costs what `[None]` does (~0.5 ms at a 2000-codepoint budget over 6 MiB)
  and `[" "] * 100` what `[" "]` does (~9 ms). Before that fix the
  per-duplicate spelling was an unbounded, caller-controlled cost
  (`[None] * 100` measured 17.2 s and +3,120 MiB peak RSS; `[" "] * 100`
  790 ms and +1,560 MiB), and between the dedup and the lazy levels a
  single-chunk budget over 6 MiB still spent ~176 ms building levels that
  supplied zero cuts.
- The unconditional `Vec<char>` + grapheme-boundary `HashSet` the chunkers
  used to build up front is gone, replaced by a lazily-built
  one-bit-per-codepoint bitmap (two SIMD scans on pure-ASCII text):
  `chunk_by_words` over 12 MiB went from ~1.9 s and ~500 MiB transient to
  ~160 ms and ~90 MiB.

The line and paragraph twins are byte-level scans: `memchr2` hops between
break bytes behind an 8-byte inline density window (dense break runs never
pay a hop), with a per-codepoint fallback for non-ASCII segments and ASCII
certification batched as one 4 KiB stride per ~50 segments. That structure
trades a small constant against the simpler whole-text `is_ascii` gate on
pure-ASCII densities and wins big on the shapes the gate leaves unguarded
(the gate has no density guard, and one non-ASCII byte forfeits an entire
document to the per-codepoint decoder):

| input (12 MiB) | `is_ascii`-gated scan | windowed scan |
|---|---|---|
| `chunk_by_lines`, prose | ~0.4 ms | ~0.45 ms |
| `chunk_by_lines`, log corpus (~80 B/line) | ~1.2-1.3 ms | ~2.2 ms |
| `chunk_by_paragraphs`, log corpus | ~1.3 ms | ~1.5 ms |
| `chunk_by_lines`, break soup (break unit every ~2.5 B) | ~42 ms | ~10 ms |
| `chunk_by_paragraphs`, break soup | ~54 ms | ~22 ms |
| `chunk_by_lines`, one non-ASCII byte anywhere | ~7.6 ms | ~0.46 ms (~16x) |
| `chunk_by_lines`, CJK-dense (every segment non-ASCII) | ~20 ms | ~21 ms |
| `chunk_by_paragraphs`, CJK-dense | ~9 ms | ~12 ms |

Outputs are differential-pinned identical across every one of these shapes;
the wall contracts gate in `tests/test_performance.py`.

## List returns have a cost at scale

The list-returning functions marshal one tuple per segment under the GIL:
`word_bounds` on 12 MiB of prose (3.67M segments) holds the GIL for 328-344
ms just marshalling the list (worst-gap band; the dev box the test ledger
records measured 428-497 ms, and the load-stable constant is the ratio,
~0.72 of the call's wall). The `_iter` twins (`word_bounds_iter`,
`find_patterns_iter`, `chunk_text_iter`, and friends) exist for exactly this:
the same sequence, streamed, each `__next__` holding the GIL for one tuple,
and the full drain ~2.1x faster in wall time in the measured case.
