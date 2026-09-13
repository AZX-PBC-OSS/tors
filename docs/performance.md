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
`minhash_signature`'s marshalling is bounded by contract (`num_perm` ints,
≤ 1024): 11 ms worst gaps over ~460 ms walls at 12 MiB, the ping floor plus
that bounded list, with no streaming twin warranted.

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
- `minhash_signature`, `num_perm=128`, vs the pure-Python MinHash loop it
  replaces (the same tokens, shingles, XXH64, and permutation arithmetic
  in Python): 0.03 ms vs 3.2 ms at 1 KiB (~104x), 3.6 ms vs 274 ms at
  100 KiB (~77x), 38.2 ms vs 2.83 s at 1 MiB (~74x). The sweep is
  O(shingles × num_perm): ~100 ms at 1 MiB with `num_perm=512`, ~0.3 ms at
  1 KiB with the default.
- `first_invalid_charset`: a 1000-item identifier batch validates in ~14 µs
  against ~97 µs for the per-item anchored-regex loop (~7x), and the
  batch-only shape is the point — a per-item tors call (~0.25 µs) loses to
  one compiled regex match (~0.08 µs), so only the one-detach batch wins.
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
  with no allocation. CROSS-ARCH, stated: pure-ASCII text takes a
  portably-SIMD `is_ascii` fast path (`2 * len`) that wins outright on
  every target; non-ASCII text runs the chunked counting loop, a bench
  artifact that auto-vectorizes on arm64 NEON (~30 GB/s there:
  ~2.2 µs at 64 KiB against the expression's ~9-10 µs, ~34 µs at 1 MiB
  against ~145 µs) but NOT on SSE2-baseline x86-64 (the CI runners
  measured it 2.2-2.5x SLOWER than the expression there — the value on
  that target is the zero-allocation and the GIL release, not the wall
  win; the wall cells assert the cross-arch no-catastrophe bound).
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

## One-shot hashing vs hashlib, honestly measured

The hashing surface (`md5_hex`/`sha1_hex`/`sha256_hex`/`sha512_hex`/
`hmac_sha256_hex`, each with a raw-digest `_digest` twin) is the one tors
family where the stdlib alternative is
C-native and fast, so the honest tables, measured (Apple Silicon, ambient
load ~8-10, min-of-3 after warmup, prose corpus bytes; the ledger cells are
`tests/test_performance.py`'s). `md5_hex` and `sha1_hex` are
checksum/legacy-interop primitives only, never security: both are broken
for security since the 2000s (practical md5 collisions date to 2004,
sha1's first public collision to 2017) — their cells below are the
Content-MD5/ETag/quick-compare jobs, never signatures, certificates, or
passwords. The `_digest` spellings are each `_hex` twin's computation
minus the hex tail (same digest, the O(digest-size) hex formatting and
its marshalling dropped for one fixed-size bytes return), so the tables
below cover them unchanged — no separate cells.

- Raw digest throughput, tors vs `hashlib` (OpenSSL, hardware SHA
  extensions): **hashlib wins or ties every engine-dominated cell** —
  sha256 12 MiB ~4-5 ms vs ~4 ms (ratio ~1.1-1.2x), sha1 4.2-4.4 vs
  3.7-4.2 (~1.06-1.11), md5 ~15.0 vs ~14.5 (~1.03), sha512 ~7.2 vs ~7.4
  (~0.98). Absolute figures move run to run (box, ambient load ~8-10,
  min-of-3 after warmup); the load-stable statement is the band, not
  any single pair. Recorded, asserted
  nowhere: at these sizes the surface's value is the parity digest, the
  str convenience, and the GIL uniformity, not throughput.
- Short-str hashing (the cache-key/ETag/request-ID spelling, where
  `hashlib` makes you encode first): **tors wins ~2x on ASCII str** —
  `sha256_hex(s)` at 0.40-0.54 of `hashlib.sha256(s.encode("utf-8"))
  .hexdigest()` (0.17 µs vs 0.33 µs at 128 B; 0.28 vs 0.52 at 512 B),
  asserted in the wall cells. ASCII is the zero-copy lane (pyo3's
  `to_str` borrow); non-ASCII str pays the one-time O(input) UTF-8
  materialization on the first call, so the win narrows there — the
  wall cells pin the ASCII band and record a non-ASCII micro-cell
  alongside it.
- HMAC at request-signature sizes: **tors wins ~3x against even the
  stdlib's fastest spelling** — 0.29 µs vs `hmac.digest(key, data,
  "sha256").hex()`'s 0.92 µs (ratio ~0.31; the idiomatic `hmac.new(...)
  .hexdigest()` costs 1.12 µs), asserted.
- GIL: tors releases the GIL for the whole digest (hex formatting
  included) at every size; `hashlib` releases it for updates of 2048+
  bytes (the `_hashopenssl` threshold), so at multi-MiB sizes the stdlib
  is loop-friendly too — worst heartbeat gaps at the ~10 ms ping floor
  for both at 12 MiB and 96 MiB (measured in `tests/test_gil_release.py`,
  recorded, not asserted as a stdlib failure). The tors difference is
  uniformity (no 2048-byte threshold, no held hex tail) plus the
  µs-scale short-call wins above; a 12 MiB digest walls at 4-8 ms on the
  calibration hardware, under the ping floor itself.

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

## Object content hashing

`content_hash` vs the full stdlib spelling
(`sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()`)
is a measured dead heat at every scale over the records corpus
(`tests/test_performance.py`, min-of-N):

| size | tors | stdlib | ratio |
|---|---|---|---|
| 64 KiB | 0.20ms | 0.21ms | 0.97 |
| 1 MiB | 3.38ms | 3.37ms | 1.00 |
| 12 MiB | 41.87ms | 41.79ms | 1.00 |

Both sides do equivalent work (CPython's C encoder builds the canonical
string in one GIL-held pass, then pays `str.encode` and a released-GIL
`sha256`; tors pays a GIL-held walk into an owned tree plus a detached
emit-and-hash), so no wall win is asserted anywhere on this surface. The
value is the GIL release — the stdlib holds the loop for `json.dumps` +
`str.encode`, inline ratio 1.00-1.02 in every sample against tors's
0.45-0.60 at 12 MiB (`tests/test_gil_release.py`) — and the byte-exact
parity contract. The detached half (emission + SHA-256) benches at
~870 MiB/s on the records corpus and ~786 MiB/s on an escape-heavy
every-codepoint corpus (`benches/canon.rs`): the full `ensure_ascii`
escape table costs ~10% over raw-run copying.

## List returns have a cost at scale

The list-returning functions marshal one tuple per segment under the GIL:
`word_bounds` on 12 MiB of prose (3.67M segments) holds the GIL for 328-344
ms just marshalling the list (worst-gap band; the dev box the test ledger
records measured 428-497 ms, and the load-stable constant is the ratio,
~0.72 of the call's wall). The `_iter` twins (`word_bounds_iter`,
`find_patterns_iter`, `chunk_text_iter`, and friends) exist for exactly this:
the same sequence, streamed, each `__next__` holding the GIL for one tuple,
and the full drain ~2.1x faster in wall time in the measured case.
