"""The GIL-release claim, as a test: ``tors.finalize``'s whole native pass runs under
``py.detach``, so a thread running it leaves the event loop schedulable at heartbeat
granularity while megabytes of text are transformed and hashed.

Methodology ported from the specification's loop-safety harness: a heartbeat
task appends monotonic ticks every 10ms while the operation runs; the assertion is on the
worst tick gap, and a sample is clean only when that gap is under both budgets: the
ratio budget (worst gap < a fraction of the operation's wall) and a generous absolute
ceiling. The test takes up to 3 samples (6 for the marshalling-heaviest members of
the chunking family, where a 3-sample window can miss the end-of-call marshalling
alignment; see that family's note below) and passes on the first clean one. A GIL-held
whole-text pass blocks the loop in every sample (the loop's thread cannot acquire the GIL
while the C call holds it, so worst gap ~= wall, ratio ~= 1.0, every time), while
transient whole-process CPU starvation on a shared runner reads as a block only in the
sample it hits, so a single starved sample is retried rather than failing the test, and
the multi-sample design is load-robust without weakening the red side (a
regression dirties every sample, so it still fails). This file uses no
async pytest plugin (tors has none and needs none): each test owns its loop via
``asyncio.run``.

Measured on the dev box this test was written on (WSL2, 28 logical cores, CPython 3.12.7;
5 samples per cell, prose and decomposed corpora from ``tests/reference.py``; the prose
cells at ambient load ~5, the decomposed cells (added with the corpus-kind parametrize)
at ambient load 1.6-4.0; walls as the corrected harness measures them, the operation's
own wall, captured at op completion before the heartbeat join, so no drain ping is
absorbed):

- ``tors.finalize`` via ``asyncio.to_thread``: worst gap / wall:
  12 MiB: 10.3-14.0ms of 109-147ms walls (ratio 0.082-0.095);
  32 MiB: 16.3-20.7ms of 353-372ms walls (ratio 0.044-0.058).
  The gap is the ~10ms ping floor plus pyo3's return marshalling of the two output
  strings (O(output), size-coupled: ~0-4ms over the floor at 12 MiB, ~6-11ms at 32 MiB),
  i.e. the marshalling band, not the transform. The 12/32 MiB pair proves the residual stays
  in that band as the corpus grows 2.7x.
- ``tors.finalize`` on the decomposed (non-ASCII) corpus, the first-call residue: the
  first sample pays pyo3's one-time O(input) UTF-8 materialization of the argument under
  the GIL (``&str`` extraction builds and caches the UTF-8 copy); every later sample on
  the same object borrows that copy zero-copy, and the prose cells' first calls sit in
  the same band as their cached samples (the ASCII borrow is a zero-copy alias, so
  there is no materialization to pay):
  12 MiB: first-call gap 13.9-15.6ms of 149-158ms walls (ratio 0.088-0.105) against a
  10.2-14.9ms cached band: the GIL-held materialization itself is ~5-8ms (a fresh-object
  ``str.encode("utf-8")`` of the same corpus, the same conversion, costs 7.7-8.6ms);
  32 MiB: first-call gap 37.6-44.1ms of 403-406ms walls (ratio 0.093-0.110) against a
  18.6-31.7ms cached band: materialization ~16-24ms (encode anchor 21.4-23.5ms).
  Even the 32 MiB first call is clean on both budgets (worst ratio 0.110 vs the 0.30
  budget, worst gap 44.1ms vs the 100ms ceiling), so these cells pass on sample 1; the
  pass-on-first-clean design also tolerates a loaded-box first sample, since samples 2+
  sit in the cached band. The first call's wall is ~20ms slower at 12 MiB (~30ms at
  32 MiB) than cached calls, but that extra is non-GIL first-call warm-up of the
  detached pass, so the loop never sees it; the gap delta and the encode
  anchor, not the wall delta, are the materialization's measure.
- The pure-Python reference finalize (``reference_finalize``) in the same to_thread
  placement, the red side: same work, same thread, but its whole-text GIL-held C calls
  (``unicodedata.normalize``, ``str.replace`` ×2, ``re.sub`` ×2, ``encode``) each hold
  the GIL for their full duration; only ``hashlib.sha256`` releases it (CPython drops
  the GIL above 2047-byte updates), which cannot save the loop because the worst gap is
  set by the largest ``re.sub``:
  12 MiB: 91.6-107.5ms gaps (ratio 0.591-0.613): fails the 0.30 ratio budget in every
  sample; 32 MiB: 249.9-271.5ms gaps (ratio 0.539-0.555), also blowing the 100ms ceiling
  by ~2.5-2.7x. Run inline on the loop instead, it is ratio ~1.0 (151-161ms at 12 MiB,
  460-505ms at 32 MiB).
- Budget margins: 0.30 sits ~3.2x above tors's worst measured ratio and ~1.8x below the
  reference's best; the 100ms ceiling sits ~4.8x above tors's worst measured gap (the
  32 MiB gap, under ambient load ~5; at 12 MiB it is ~7x).
- What this suite pins: both bands of the crate GIL model (src/lib.rs), the ASCII
  marshalling band via the prose cells and the non-ASCII first-call materialization via
  the decomposed cells, not just the ASCII band.

(quick-check fast paths + identity returns + the streaming iterator), and what
they did to these cells, measured on the dev box, ambient load 4.4-5.7 unless noted:

- The identity path's band (the zero-cost lane): on already-normalized 12 MiB prose
  (``reference_normalize(prose(...))``, the pipeline's own output), ``tors.normalize``
  walls collapse to ~3-4ms and ``tors.finalize`` to ~9-10ms (the quick-check scan plus,
  for finalize, SHA-256 over the borrowed input, measured 2.89ms/7.72ms min-of-5 at
  load 4.6; the build measured 123.0ms/126.3ms at load 12.1, min-of-N robust on
  this box). Those walls sit at or under the 10ms ping floor, so their gap/wall ratios
  (~1.2-3.0) are the documented sub-ping artifact: the identity-path cells assert the
  100ms ceiling only (>=9x margin), exactly the b64 12 MiB precedent.
- The QC-Yes-but-scan-dirty prose corpus keeps paying the scan but skips the NFC pass:
  ``finalize`` prose walls shrink ~3x (42-46ms at 12 MiB, 111-123ms at 32 MiB,
  measured) while the GIL-held residue (the O(output) marshalling) is unchanged,
  so the residue/wall fraction structurally rose at 12 MiB: measured ratios 0.27-0.34.
  The shared 0.30 budget is therefore no longer attainable with margin on that one
  cell, and a budget that flakes is worse than a budget derived from the band: the
  12 MiB prose cells (finalize, finalize_utf8) move to a bespoke 0.60 ratio budget,
  ~1.8x above the worst measured ratio (0.34) and ~40% below the ~1.0 a detach
  regression shows (the whole 42-45ms transform held would fail it by far; the 100ms
  ceiling holds ~2.3x margin). The 32 MiB prose cells stay on the shared budget
  (measured 0.09-0.17).
- The D-forms' full pass lost its corpus: plain decomposed prose carries no
  compatibility mappings, so under NFKC/NFKD it now quick-checks Yes and comes back
  as the input object; ``nfkd(decomposed)`` walls collapse from 90-98ms to 10-23ms
  (measured; the residue is the str-in first-call materialization plus the quick-check
  scan). The forms cells therefore moved to the new ``compat`` corpus (decomposed
  accents + U+FB01 ligature + U+FF10 fullwidth digit per unit; the compatibility
  mappings keep the quick check at No), whose measured bands sit inside both shared
  budgets (nfc 12 MiB 16.8-17.4ms of 127-145ms walls, 0.10-0.13; nfkd 12 MiB
  14.7-17.8ms of 87-104ms, 0.14-0.20; nfc 32 MiB 36.3-40.1ms of 346-391ms, 0.10-0.12;
  nfkd 32 MiB 35.7-38.2ms of 240-263ms, 0.15-0.16). A separate ceiling-only cell pins
  the D-form fast path on the decomposed corpus (walls at the ping floor → ratio
  artifact, the b64 12 MiB precedent again).
- The word_bounds streaming answer: ``tors.word_bounds_iter`` full drain at 12 MiB
  (3.67M bounds) measured worst gaps 15.4ms of 327-358ms walls (0.04-0.05), inside
  both shared budgets, against the list shape's structurally-unattainable 428-497ms
  band. The wall surprise, recorded: the full drain measured 347ms min-of-3
  against the list API's 724ms in the same process; the per-``__next__`` path is
  faster per bound than the list conversion (~47ns vs ~150ns per bound by
  subtraction of the shared ~175ms detached core), so the iterator wins on both axes
  at this size; the list API stays for small inputs and one-shot batch work.

Corpus sizes start at 12 MiB because the ratio only means something when the wall exceeds
the 10ms ping floor by a wide margin (at 1 MiB the whole finalize is ~10ms: gap ~= ping
floor ~= wall, ratio ~1.0 by artifact, measured, not evidence of blocking).

bytes-in cells (``finalize_utf8``, ``b64_encode_bytes``; measured on the same box,
ambient load 4.3-5.6, 5 samples per cell, corpora from ``reference.corpus_utf8``, the
str corpora rendered to UTF-8 bytes):

- ``tors.finalize_utf8`` via ``asyncio.to_thread``: the no-materialization claim, on
  both corpus kinds: prose 12 MiB 10.3-13.0ms of 113-143ms walls (0.08-0.10); decomposed
  12 MiB 10.3-13.0ms of 116-131ms (0.09-0.10); prose 32 MiB 13.5-20.7ms of 345-370ms
  (0.04-0.06); decomposed 32 MiB 17.6-26.7ms of 350-405ms (0.04-0.07). The decomposed
  cells' sample-1 gaps sit in the same band as samples 2+: the one-time O(input) UTF-8
  materialization that the str-in ``finalize`` cells pay on their first sample
  (13.9-15.6ms at 12 MiB, 37.6-44.1ms at 32 MiB, above) does not exist for bytes-in:
  pyo3's ``&[u8]`` extraction is a zero-copy borrow of the immutable buffer (verified in
  pyo3 0.29's source; see src/lib.rs's GIL model). The 12/32 MiB pair proves the residue
  stays in the marshalling band as the corpus grows.
- ``tors.b64_encode_bytes`` via ``asyncio.to_thread``: the residue is the marshalling
  of the 4/3x-sized ASCII output: 12 MiB worst gaps 10.1-11.2ms against 5-9ms walls
  (wall under the ping floor: the ratio there is the sub-ping artifact, hence a ceiling-only
  cell); 32 MiB 27.5-29.3ms of 48-50ms (0.57-0.58) on the dev box: ceiling-only
  since the 0.2.1 recalibration, because on a fast quiet box the walls collapse to
  12-19ms where the heartbeat's own wobble decides the ratio (same
  floor-resolution failure the diff cell recalibrated for); 96 MiB is the
  regression-detecting ratio cell at the b64-specific 0.80 budget (0.46-0.50 dev,
  0.23-0.33 fast box: ~1.6x margin both boxes, detach shows ~1.0). 48 MiB
  measured and unasserted: 37.0-48.8ms of 68-80ms (0.54-0.61).
  The b64 budget derivation: 0.80 is ~1.3x above the worst measured ratio (0.61) and
  ~20% below the ~1.0 a GIL-held pass shows in every sample; the shared 0.30 is
  structurally unattainable for b64 because the fast encode makes the marshalling the
  dominant share of the wall (gap/wall -> marshal/(encode+marshal) ~ 0.6-0.75 as size
  grows, measured). The red side: `base64.b64encode` holds the GIL, with 122.3-127.6ms
  held at 96 MiB, reproducing the ~150ms@100MB GIL-held b64encode observation that
  motivated the function; at 12/32 MiB the red expression measures ratio 0.63-0.66
  (window-dependent at 12) and passes those cells' budgets; the red-side cell below
  records the structural reason and runs the 96 MiB expression inline on the
  loop (the to_thread placement's two-call boundary leaves the worst gap at
  the encode alone, under the 0.80 budget, its ceiling margin
  box-speed-dependent), asserting the budgets mechanically.

diff_opcodes cells (the 32 MiB near-identical and 12 MiB shuffled pairs from
``reference.diff_pair_near_identical`` / ``reference.diff_pair_shuffled``;
measured on the dev box, ambient load 5.3-6.6, 3 samples per cell):

- Near-identical (scattered line edits -> dozens of opcodes): at 12 MiB,
  worst gaps 10.5-11.2ms of 76-88ms walls (ratio 0.12-0.14): the ping floor
  plus the two zero-copy ASCII argument borrows and a ~0.05ms opcode-tuple
  marshalling; inside both shared budgets with ~2x ratio margin. The pair
  is 32 MiB, not 12, deliberately: on a fast quiet box the 12 MiB walls
  (~30ms) sit inside the heartbeat floor's own run-to-run wobble (5-11ms
  gaps: ratio 0.18 one run, 0.35 the next, same budget, no code change:
  the ratio stops resolving and the cell flakes). At 32 MiB the walls
  (~100ms on that box) restore ~5x margin (measured 0.06) while a detach
  regression still holds the whole wall at ~1.0 on every box; ceiling-only
  was rejected because a held 30ms wall on a fast box sits under the 100ms
  ceiling, which would blind the cell exactly where it is weakest.
- Shuffled (103,421 opcodes; the O(ops) marshalling band made visible, the
  ``word_bounds`` list-shape class): worst gaps 20.4-26.3ms of 1554-1626ms
  walls (0.013-0.017). The marshalling itself is the delta over the
  near-identical cell's floor band: ~10-15ms for 103,421 5-tuples, ~0.1-0.15µs
  per opcode (one tuple plus up to four fresh PyLongs each; the four tag
  strings are constructed once per call and shared by reference into every
  tuple, difflib's own interned-tag shape; a per-op PyString would
  multiply the band several-fold). The cell pins the band with a bespoke 60ms
  gap ceiling (~2.3x above the measured worst, catching a several-fold
  marshalling blowout that the shared 100ms ceiling would let through at
  this op count) plus the shared 0.30 ratio (a detach regression holds the
  whole ~1.6s wall). The op count is reported, not asserted: it is the
  algorithm's business (similar's bounded search trades minimality for speed
  on hard inputs, and a crate bump may change the count without changing
  validity).

find_patterns cells (the sparse and dense 12 MiB shapes from
``reference.SEARCH_SPARSE_PATTERNS`` / ``SEARCH_DENSE_PATTERNS``; measured on
the dev box, ambient load 6.9-7.3, 3 samples per cell):

- Sparse (the diff near-identical pair's edited corpus, the terminology-scan
  shape: ``"monthly"`` occurs exactly once, the other two terms never):
  worst gaps 10.3-10.7ms of 6-7ms walls: the ping floor plus ~nothing (one
  3-tuple). The pure scan is ~2 GiB/s, so the wall sits under the 10ms ping
  floor and any gap/wall ratio is the suite's documented sub-ping artifact;
  the cell asserts the 100ms ceiling only (the b64 12 MiB / utf8_is_valid
  precedent). A 6ms scan held or released is invisible under the
  floor either way, so this cell cannot discriminate a detach regression;
  the dense cell below carries that (a held dense pass shows ratio ~1.0
  against its ~215ms wall); the sparse cell pins that the no-match scan
  leaves the loop at the floor at all.
- Dense (17 prose words over the plain 12 MiB prose corpus, 1,284,724
  matches; the O(matches) marshalling band made visible, the
  ``word_bounds`` list-shape class): worst gaps 171.0-174.7ms of 214-224ms
  walls (ratio 0.76-0.80). The marshalling itself is the delta over the
  sparse cell's floor band: ~160-165ms for 1.28M 3-tuples, ~0.13µs per match,
  the same per-element band diff_opcodes measures per opcode. The cell
  pins the band with a bespoke 400ms gap ceiling (~2.3x above the measured
  worst; the word_bounds-precedent derivation, catching a several-fold
  marshalling blowout the shared 100ms ceiling is not sized for at this
  match count) plus a 0.90 ratio budget as the detach discriminator. Why
  0.90 and not word_bounds' 0.85: this cell's band sits structurally
  higher than word_bounds' 0.72-0.74, because the search core is ~3.5x
  faster than segmentation, so the marshalling is a larger share of the
  wall; 0.90 sits ~1.125x above the worst measured ratio and ~10% below
  the ~1.0 a detach regression shows (the whole ~215ms wall held). The
  match count is reported, not asserted (it is the corpus's business).
  Guidance, the word_bounds finding's shape: the list-returning API
  holds ~0.13µs per match under the GIL: ~13ms at 100k matches (document
  scale), ~170ms at 1.28M (whole-corpus keyword sweeps); callers producing
  millions of matches are the streaming-API question the word_bounds
  finding already raised, recorded again in docs/async.md.

unescaped-scan cells (``contains_unescaped``/``find_unescaped``, the
escape-parity byte scan; measured on the calibration box, macOS, 16
cores, ambient load ~6-17, 3 samples per cell, corpora from
``reference``: the plain prose bytes for the sparse shape and
``unescaped_false_positive`` for the hit-dense all-rejected shape):

- Both cells ceiling-only (the b64 12 MiB / utf8_is_valid precedent):
  the scan is memchr-class — measured 0.25ms (sparse, no occurrence) and
  0.79ms (dense, 72,520 rejected hits) inline at 12 MiB — so the walls
  sit an order of magnitude under the 10ms ping floor and any gap/wall
  ratio is the suite's documented sub-ping artifact. Measured worst gaps
  10.3-11.1ms (the floor plus ~0.3-1.1ms of to_thread dispatch and the
  two argument borrows) of 0.4-1.3ms walls, both shapes, both spellings
  (``contains`` measured the same band: it is the same scan by
  construction, pinned by the invariant test in
  tests/test_unescaped_scan.py). The 100ms ceiling alone is the
  assertion (~9x margin); the same limitation as every sub-floor cell
  applies: a held ~1ms scan is invisible under the floor either way, so
  these cells pin that the whole memmem loop plus the parity walk leaves
  the loop at the floor at all, and the wall cells in
  tests/test_unescaped_scan.py carry the throughput side.

utf8_byte_len cells (the byte-count companion, #52; measured on the
calibration box, macOS, 16 cores, ambient load ~10-18, 3 samples per
cell, corpora from ``reference``: plain prose for the ASCII lane,
``decomposed`` with a FRESH object per sample for the non-ASCII
first-call lane):

- Both cells ceiling-only, for two different structural reasons. The
  ASCII lane is O(1) end to end (compact ASCII is its own UTF-8, so the
  str-in borrow is a zero-copy alias; everything past the borrow is a
  field read and a single int out): measured worst gaps 10.2-11.3ms of
  0.1-0.4ms walls — the ping floor plus to_thread dispatch, the call
  itself ~0.1µs — the b64 12 MiB / utf8_is_valid sub-floor artifact.
  Nothing O(n) exists to detach on this lane, so it pins that the call
  leaves the loop at the floor at all; the O(1) band is pinned in wall
  time by tests/test_performance.py.
- The non-ASCII first-call lane is the function's one heavy lane, made
  structural by the fresh-object-per-sample design: the borrow's
  materialization of the UTF-8 view is GIL-held O(n) (the cold-cache case:
  the borrow fills the cache, ``encode`` only reads it, so only a str-in
  call — not a prior encode — ends the cold lane; there is no way to fill
  an object's cache without the GIL), measured 4.6-5.6ms walls
  inline at 12 MiB with worst gaps 10.6-10.8ms — the materialization
  (~5ms) sits under the 10ms ping interval itself, so the loop never
  misses a tick beyond the floor at this size; the 100ms ceiling holds
  ~10x, and the linear envelope (~0.4-0.5ms of GIL hold per MiB) puts a
  ~200 MiB non-ASCII string at the ceiling (the recorded scale
  guidance). The gap/wall ratio is ~1.0 by construction (the wall IS
  the GIL-held materialization), which is why this leg is ceiling-only
  like the D-form fast-path cell, not because the wall is sub-floor.

utf16_byte_len cell (the interop twin, #52; same corpus shapes and
fresh-object-per-sample design): both legs ceiling-only for the twin's
two reasons, with the one honest structural difference — this core's
detach carries REAL work (the O(n) byte-class scan, ~30 GB/s, ~0.4ms
at 12 MiB — an order under the ping floor), which is why it keeps its
detach where the utf8 twin's nominal one was removed (#108: a detach
must bracket the real work). The ASCII leg's whole call is the
zero-copy alias
borrow plus the detached scan, sub-floor end to end; the non-ASCII
first-call leg's worst gap is the same GIL-held materialization as the
twin's (~5ms at 12 MiB, under the ping interval), with the scan
detached behind it.

cells (``replace_many`` dense, ``sentence_bounds`` list,
``diff_opcodes_lines`` near-identical, all at 12 MiB; measured on the dev
box, ambient load 2.0, 5 samples per cell, corpora from
``tests/reference.py``):

- ``replace_many`` dense (the 17 SEARCH_DENSE_PATTERNS words each mapped
  to a redaction token over the plain prose corpus, the redaction-map
  shape the function exists for): worst gaps 10.3-10.9ms of 47-60ms walls
  (ratio 0.18-0.23): the 10ms ping floor plus the O(entries) argument
  walk and the O(output) marshalling of one ~10 MiB string, exactly the
  crate GIL model's prediction: the scan+splice is one detached pass and
  the return is a single string, so no list-shape class exists at all.
  The wall clears the ping floor ~5x, so the ratio is not the sub-ping
  artifact, and the cell takes the shared budgets (0.30 sits ~1.3x above the
  worst measured ratio and ~70% below the ~1.0 a detach regression shows
  against these ~50ms walls; the 100ms ceiling holds ~9x).
- ``sentence_bounds`` list (the word_bounds pair's list shape, sentences
  being far sparser: 170,037 segments measured at 12 MiB, ~1/22nd of
  word_bounds' 3.67M, even sparser than the ~1/10th design estimate in
  src/lib.rs): worst gaps 17.6-23.8ms of 185-190ms walls (ratio
  0.09-0.13): the ping floor plus the O(sentences) 2-tuple marshalling.
  Every sample inside both shared budgets (~2.3x ratio margin, ~4.2x
  ceiling margin); a detach regression holds the whole ~190ms wall at
  ratio ~1.0 and fails by far. The list shape meets the shared budgets
  here (unlike word_bounds' 428-497ms band) precisely because the tuple
  count is ~20x smaller; the per-shape guidance, not a reason to
  skip the streaming spelling for whole-corpus sweeps.
- ``diff_opcodes_lines`` on the near-identical pair (5 line-opcodes
  measured, the line-level Myers over ~37.8k mostly-distinct lines
  anchored to near-nothing): worst gaps 10.5-12.0ms against walls of only
  1.9-2.6ms: the wall sits an order of magnitude under the 10ms ping
  floor, so the gap/wall ratio (4.2-6.4) is the suite's documented
  sub-ping artifact and the cell asserts the 100ms ceiling only (the b64
  12 MiB / utf8_is_valid / find_patterns-sparse precedent, ~8x margin).
  The line-level spelling is ~30x cheaper in wall than the char-level
  spelling on the same 12 MiB pair (76-88ms on the dev box, when the
  char-level cell still measured that size before its 32 MiB
  recalibration above) because the diff runs
  over lines, not characters; a limitation in the sparse-search
  cell's shape: a ~2.5ms diff held or released is invisible under the floor
  either way, so this cell pins that the line-split + Myers pass leaves the
  loop at the floor at all.

cells (``count_matches`` dense and ``find_patterns_iter`` dense drain,
both at 12 MiB, the 17-word ``SEARCH_DENSE_PATTERNS`` set over the plain
prose corpus, 1,284,724 matches; measured on the dev box, ambient load 4.4,
5 samples per cell):

- ``count_matches`` dense (the same scan the list shape pays 171.0-174.7ms
  of GIL hold to marshal; here with no match vector, no byte→char
  conversion pass, and a single int return): worst gaps 10.4-11.2ms of
  31.0-31.8ms walls: the ping floor plus ~nothing, the grapheme_count
  precedent's no-marshalling band. The budget consequence: the wall
  clears the ping floor only ~3x, so gap/wall sits at 0.336-0.353 by
  arithmetic and the shared 0.30 ratio budget is unattainable-by-
  construction for this wall (the QC-Yes 12 MiB situation verbatim: a fast
  detached core makes the floor a large fraction of the wall); the cell
  takes a bespoke 0.60 ratio budget: ~1.7x above the worst measured ratio,
  ~40% below the ~1.0 a detach regression shows (a held scan pins the whole
  ~31ms wall), plus the shared 100ms ceiling (~9x). The inverse of the
  find_patterns-sparse limitation: a held ~31ms scan
  passes the 100ms ceiling easily, so the ratio budget is this cell's only
  detach discriminator.
- ``find_patterns_iter`` dense drain (construction plus every ``__next__``,
  the streaming answer to the list shape's 171-175ms band): worst gaps
  15.4ms of 129.0-132.0ms walls (ratio 0.116-0.119), inside both shared
  budgets (~2.5x ratio margin, ~6.5x ceiling margin), the word_bounds_iter
  band exactly (15.4ms there: the ping floor plus the two-thread GIL
  contention of the draining thread's per-``__next__`` bytecode). The wall
  contrast, measured in the same process (min-of-3, ambient load 3.1): full
  drain 117.4ms against the list API's 211.1ms and the count core's
  28.3ms: the drain is the ~28-31ms construction scan plus ~86-90ms of
  1.28M per-``__next__`` 3-tuple handoffs (~70ns per match, against the
  list shape's ~0.13µs per match of GIL-held marshalling), so the iterator
  wins both axes at this size (the word_bounds_iter finding's shape:
  347ms vs 724ms there). A discrimination caveat: the construction pass is
  only ~31ms of the ~130ms wall, so a construction-only detach regression
  shows ~41ms gaps at ratio ~0.32; the 0.30 budget catches it, but
  thinly; the grosser classes (a per-``__next__`` re-scan, per-next chunk
  marshalling) blow through both budgets.

chunking-family cells (the ``chunk_*`` GIL-model claim pinned directly
for the first time, #30 item 5: ``chunk_text``, ``chunk_by_words`` (+
``_iter``), ``chunk_by_sentences``, ``chunk_by_paragraphs`` (+
``_iter``), ``chunk_by_lines`` (+ ``_iter``), and ``chunk_hierarchical``
with both the default hierarchy and the line-first ``["\n", None]``
splice, all at 12 MiB of the new ``chatlog`` corpus; the lazy levels and
the certificate scanners moved several members' walls (the
paragraph/line scans ~3x faster, the default hierarchy ~2x), so the
  bands below are the current tree's: 3 samples per member (the
marshalling-heaviest members take 6: a 3-sample window can miss the
end-of-call marshalling alignment):

- The corpus, built local to this module (the ``_invalid_utf8_corpus``/
  ``_close_matches_corpus`` precedent): the line-heavy chat-thread shape
  the streaming twins' docstrings justify themselves with: five
  rotating speakers' one-line messages (~66-68 bytes each), a blank line
  every 5 lines, so 12 MiB holds ~185k content lines in ~37k paragraphs.
  Pure ASCII, deterministic, no RNG, unit-quantized like every
  ``reference`` corpus.
- Per-member parameters sized for piece-count-heavy outputs where the
  O(chunks) marshalling is real but not word_bounds' 3.67M-segment
  428-497ms list class: 37k-370k pieces (chunk_text 64,278 at
  ``max_chars=200``; chunk_by_words 129,528 at 20 words/chunk;
  chunk_by_sentences 44,410 at 5; chunk_by_paragraphs 37,008 at 1;
  chunk_by_lines 37,008 at 5; chunk_hierarchical 111,024 default at
  ``max_chars=200`` / 370,080 line-first at ``max_chars=40``). The
  line-first leg's budget is 40, not the 200 its siblings use, because
  of what the lazy levels did to the 200-budget leg: every window of
  this corpus was served by the "\n" literal level alone (no spliced
  level ever consulted, none of the spliced walks built), so the wall
  collapsed to ~13ms, where a full GIL-hold regression passes the
  100ms ceiling and the member pins nothing; at 40 every ~66-68-byte
  line is oversized, windows descend past the line literal
  into the spliced paragraph/sentence/word levels, and the leg is the
  family's heaviest again (~405-409ms, 370,080 pieces).
- Ceiling-only everywhere (``ratio_budget=None``, the b64 12 MiB /
  utf8_is_valid precedent): the light members (paragraphs/lines + their
  ``_iter`` drains, single-pass newline scans behind the sliding ASCII
  certificate) wall at ~3.4-5.5ms (under the ping floor), where any
  ratio is the documented sub-ping artifact (measured 2.0-3.2), and the
  heavy members' ~164-410ms walls are dominated by their detached
  cores, so a detach regression holds the whole wall and blows the
  100ms ceiling in every sample: the ceiling alone is the family's
  detach pin, and the red-side cell below now proves it mechanically
  for five heavy members run inline on the loop (ratio ~1.00, over the
  ceiling in every sample). Measured worst gaps:
  chunk_text 10.2-13.1ms of 395-451ms (0.02-0.03); chunk_by_words
  10.4-19.4ms of 170-175ms (0.06-0.11); chunk_by_words_iter
  10.6-11.9ms of 164-168ms (0.06-0.07); chunk_by_sentences 10.3-10.7ms
  of 187-190ms (0.05-0.06); chunk_by_paragraphs 10.1-11.0ms of
  3.4-3.9ms; chunk_by_paragraphs_iter 10.5-11.1ms of 3.4-4.0ms;
  chunk_by_lines 10.6-11.0ms of 4.4-5.5ms; chunk_by_lines_iter
  10.5-10.6ms of 4.4-4.5ms; chunk_hierarchical default 10.3-15.3ms of
  194-204ms (0.05-0.08); line-first@40 30.7-41.8ms of 400-409ms
  (0.08-0.10, the family's most-marshalling member: 370,080 pieces,
  ~20-31ms of 2-tuple construction over the floor, the residue
  gradient's top step). Ceiling margins ~2.4x (line-first) to ~9.4x
  on every member; the light members' can't-discriminate-a-held-
  sub-ceiling-wall limitation and the mid-weight members' fast-box
  caveat are stated in the cell's docstring.

hashing cells (``sha256_hex``/``sha512_hex``/``sha256_digest`` plus
``hmac_sha256_hex``/``hmac_sha256_digest`` with a short key, all at
12 MiB, measured on the
dev box this section was calibrated on, Apple Silicon, ambient load
7.8-9.7, 3 samples per cell, the prose corpus's UTF-8 bytes):

- The walls are the story: 4-5ms (sha256) and 7-8ms (sha512) for one
  12 MiB digest, at or under the 10ms ping floor itself, so every
  worst-gap/wall ratio is the suite's documented sub-ping artifact and
  both cells are ceiling-only (``ratio_budget=None``, the b64 12 MiB /
  find_patterns sparse / chunking light-member precedent): measured
  worst gaps 10.1-10.7ms — the ping floor plus the zero-copy argument
  borrow and the O(64..128) hex-string marshalling, nothing else, the
  no-residue-class claim of src/lib.rs's hashing paragraph measured
  directly.
- The honest hashlib red side, measured and NOT asserted: CPython's
  ``hashlib`` releases the GIL for digest updates of 2048+ bytes (the
  ``_hashopenssl`` threshold), so at digest-scale sizes the stdlib is
  loop-friendly too — measured at 12 MiB (worst gaps 10.4-10.8ms of
  4-8ms walls, the ping floor) and at 96 MiB (10.9-11.1ms of 33-38ms
  walls for sha256, 11.1ms of 117-121ms for md5: the floor against
  walls 3-12x over it). There is therefore no GIL-blocked stdlib red
  row to assert against at any size where the work is visible: below
  the 2048-byte threshold hashlib holds the GIL, but a sub-threshold
  digest is ~11µs (measured), invisible under the floor either way.
  The urllib red-side precedent applies: the red side is measured live
  and recorded in the cell below, and what is asserted is the value
  parity (the 12 MiB differential anchor).
- The discriminating-power limitation, stated (the chunking family's
  light-member note, verbatim precedent): on this hardware a lost
  detach at 12 MiB shows a sub-floor wall (inline tors measured
  10.7ms worst gaps of 5ms walls — the floor, indistinguishable from
  the green band), so no budget the green cells use could discriminate
  it here. What a detach regression would look like on slower
  hardware, measured at 96 MiB (where the sha256 wall clears the floor
  ~4x): to_thread worst gaps 11.1ms of 39-40ms (ratio 0.28) vs inline
  39.0-39.7ms of 39-40ms (ratio 1.00, the whole wall held) — the
  one-call lost-detach shape, recorded so the next reader knows the
  ceiling-only design is a hardware-speed statement, not a no-op.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import difflib
import hashlib
import itertools
import json
import random
import string
import time
import urllib.parse
from collections.abc import Awaitable, Callable

import pytest

import tors
from reference import (
    SEARCH_DENSE_PATTERNS,
    SEARCH_SPARSE_PATTERNS,
    UNESCAPED_NEEDLE,
    compat,
    contacts,
    content_object,
    corpus_b64,
    corpus_utf8,
    decomposed,
    diff_pair_near_identical,
    diff_pair_shuffled,
    entities,
    prose,
    reference_finalize,
    reference_normalize,
    scrub_corpus,
    unescaped_false_positive,
)

# The timing lane: every test in this module is a measurement cell (worst
# heartbeat-gap bands over multi-MiB corpora), slow and load-sensitive, so
# CI's matrix legs deselect it (`-m "not timing"`) and one dedicated step on
# the 3.12 leg runs it (`-m timing`): the GIL contracts still gate every
# push, once, instead of being paid on all five legs. Locally a bare
# ``pytest`` (and ``make test``) run everything, marker or not.
pytestmark = pytest.mark.timing

_PING_S = 0.01
_RATIO_BUDGET = 0.30
_ABS_CEILING_S = 0.100
_SAMPLES = 3

_MIB = 1024 * 1024

# b64_encode_bytes's own ratio budget; see the module docstring's b64
# discussion for why it is not the shared 0.30: the output is 4/3x the input, so the O(output)
# marshalling residue is a structurally larger fraction of b64's wall than of
# finalize's (whose transform dwarfs its residue). 0.80 sits ~1.3x above the worst
# measured ratio (0.61, 48 MiB under ambient load ~5.5) and ~20% below the ~1.0 a
# GIL-held pass shows in every sample.
_B64_RATIO_BUDGET = 0.80

# word_bounds's marshalling-band regression ceiling: not the suite's 100ms
# ceiling, which the list-returning API shape cannot meet at whole-file sizes
# (measured: 428-497ms worst gaps at 12 MiB, 3.67M segments; see the cell's
# docstring and docs/async.md). 1.0s is ~2x above
# the measured band, so it catches marshalling blowouts without pretending
# the 100ms budget is attainable here. The streaming sibling
# (word_bounds_iter, cell below) does meet the shared budgets; this cell
# keeps the list shape's band pinned.
_WORD_BOUNDS_CEILING_S = 1.0

# word_bounds's detach-regression discriminator: the measured marshalling
# band sits at ratio 0.72-0.74 (gap/wall), while
# a detach regression (the segmentation itself GIL-held) shows ratio ~1.0
# with a gap of only ~0.4-0.6s at 12 MiB, which the 1.0s ceiling alone lets
# through (a simulated regression measured 0.37-0.40s gaps at ratio 100%
# passed the old ceiling-only assertion). 0.85 sits ~1.15x above the band's
# worst measured ratio and well below the ~1.0 a detach regression shows.
_WORD_BOUNDS_RATIO_BUDGET = 0.85

# The 12 MiB QC-Yes prose cells' ratio budget (finalize / finalize_utf8), a
# recalibration: the quick-check skip shrank those walls ~3x (42-46ms,
# measured) while the O(output) marshalling residue is unchanged, so the
# shared 0.30 became a coin flip (measured ratios 0.27-0.34). 0.60 sits ~1.8x
# above the worst measured ratio and ~40% below the ~1.0 a detach regression
# shows (the whole transform held); the 100ms ceiling independently holds
# ~2.3x margin. Same derivation shape as _B64_RATIO_BUDGET below.
_QC_YES_12MIB_RATIO_BUDGET = 0.60

# The 12 MiB scrub_pii cell's ratio budget, same derivation: the scrub's
# double scan is fast enough that the 12 MiB contacts corpus completes in
# ~36-40ms while the O(output) marshalling of its ~11.8 MiB result string
# costs ~12ms, so the residue is structurally ~33-34% of the wall (measured,
# every sample) where the shared 0.30 cannot hold. 0.60 sits ~1.8x above
# the worst measured ratio and ~40% below the ~1.0 a detach regression
# shows (a held double scan pins the whole ~36ms wall as one gap); the
# 100ms ceiling independently holds ~8x over the worst gap.
_SCRUB_PII_12MIB_RATIO_BUDGET = 0.60

# content_hash's own ratio budget (the object-walk residue class, a new
# class: the GIL-held walk materializes the whole value tree -- one borrow
# plus copy per str, one i64 read per int, one repr call per float -- and
# the canonical-form emission plus SHA-256 run detached under one
# py.detach, so the residue is structurally ~half the call rather than a
# small marshalling tail). Measured on the dev box over two load windows
# (ambient load ~2 and ~10-16, 3-5 samples, the records corpus
# (reference.content_object) at 12 MiB, ~30.7k records): worst gaps
# 23.3-32.2ms of 41.9-60.4ms walls, ratios 0.45-0.60, the loaded window's
# 0.60 the worst observed. 0.80 sits ~1.3x above that worst ratio and
# ~20% below the ~1.0 the lost-detach shape shows in every sample (the
# red row below, measured 1.00-1.02); the 100ms ceiling holds ~3x margin
# over the worst gap. Same derivation shape as _B64_RATIO_BUDGET.
_CONTENT_HASH_RATIO_BUDGET = 0.80

# The line-heavy corpus for the chunking family's cells (#30 item 5): the
# chat-thread/log shape the streaming twins' own docstrings justify
# themselves with ("a line-oriented corpus (a multi-MiB log or transcript)
# ... chunking into hundreds of thousands of pieces", chunk_by_lines_iter)
# and the ["\n", None] hierarchy splice exists for ("a chat thread, one
# message per line"). Five rotating speakers' short messages, one content
# line per ~66-68 bytes, a blank line (the paragraph gap) every 5 lines,
# unit-quantized to the target bytes by repetition: the deterministic
# no-RNG idiom every reference.py corpus uses, but local to this file (the
# _invalid_utf8_corpus/_close_matches_corpus precedent: only this module's
# cells consume it). Pure ASCII, as a real log/transcript overwhelmingly
# is, so the str-in borrow is zero-copy and the cells' bands isolate the
# chunking passes themselves: 12 MiB holds ~185k content lines in ~37k
# blank-line-separated paragraphs (~4.48M word segments, ~222k sentences).
_CHATLOG_SPEAKERS = ("Ana", "Bo", "Cleo", "Dov", "Eun")


def chatlog(target_bytes: int) -> str:
    """The chat-thread corpus: five speakers' one-line messages with a
    blank-line paragraph gap every 5 lines, repeated to ``target_bytes``
    (UTF-8, like every corpus builder's sizing)."""
    content = [
        f"{speaker}: message {n} acknowledged, window {n * 7} days, torque spec unchanged."
        for n, speaker in enumerate(_CHATLOG_SPEAKERS, start=1)
    ]
    unit = "\n".join(content) + "\n\n"
    return unit * max(1, target_bytes // len(unit.encode("utf-8")))


# The corpus kinds the str-in cells parametrize over: prose (ASCII) pins
# the O(output) marshalling band; decomposed (non-ASCII) pins the one-time
# O(input) first-call UTF-8 materialization (samples 2+ on the same object
# borrow the cached copy zero-copy); chatlog (the line-heavy kind above)
# feeds the chunking family's cells. Keyed lookups only, so the third
# kind changes no existing cell.
_CORPORA: dict[str, Callable[[int], str]] = {
    "prose": prose,
    "decomposed": decomposed,
    "chatlog": chatlog,
}


def _chunk_family_calls(corpus: str) -> dict[str, Callable[[], object]]:
    """The chunking family's member calls over ``corpus``, one shared
    definition so the green family cell and the red-side cell's inline
    rows run the same call at the same params (a red row that drifted
    from its green twin would prove nothing about the twin's budget).
    The line-first hierarchy leg's ``max_chars=40`` is sized with the
    lazy levels in mind: at 200 every chatlog window is served by the
    "\n" literal alone (no spliced walk ever built, wall ~13ms, under
    the ceiling even when held), at 40 the descent into the spliced
    levels makes it the family's heaviest member (~405ms, 370,080
    pieces); see the family cell's docstring for the full story."""
    return {
        "chunk_text": lambda: tors.chunk_text(corpus, 200),
        "chunk_by_words": lambda: tors.chunk_by_words(corpus, 20),
        "chunk_by_words_iter": lambda: sum(1 for _ in tors.chunk_by_words_iter(corpus, 20)),
        "chunk_by_sentences": lambda: tors.chunk_by_sentences(corpus, 5),
        "chunk_by_paragraphs": lambda: tors.chunk_by_paragraphs(corpus, 1),
        "chunk_by_paragraphs_iter": lambda: sum(
            1 for _ in tors.chunk_by_paragraphs_iter(corpus, 1)
        ),
        "chunk_by_lines": lambda: tors.chunk_by_lines(corpus, 5),
        "chunk_by_lines_iter": lambda: sum(1 for _ in tors.chunk_by_lines_iter(corpus, 5)),
        "chunk_hierarchical_default": lambda: tors.chunk_hierarchical(corpus, 200),
        "chunk_hierarchical_line_first": lambda: tors.chunk_hierarchical(corpus, 40, ["\n", None]),
    }


# The family's marshalling-heaviest members (the ones whose end-of-call
# O(pieces) 2-tuple construction is a real residue over the 10ms ping floor
# (the residue gradient's upper steps: chunk_text at 64,278 pieces,
# chunk_by_words at 129,528, both chunk_hierarchical legs at 111,024 /
# 370,080) take a 6-sample window in the family cell below: whether a
# sample's worst gap captures that residue depends on tick alignment
# against the end-of-call marshalling window, so a 3-sample window can
# miss the alignment. The floor-band members (chunk_by_sentences,
# the _iter drains, the light scans) keep the default 3: their residue is
# the floor itself, which every alignment sees.
_FAMILY_MARSHALLING_HEAVY = frozenset(
    {"chunk_text", "chunk_by_words", "chunk_hierarchical_default", "chunk_hierarchical_line_first"}
)


async def _gap_and_wall_during(op: Callable[[], Awaitable[object]]) -> tuple[float, float]:
    """Run ``op`` concurrently with a heartbeat; return ``(worst_tick_gap, op_wall)``.

    The leading ``sleep(0)`` guarantees the heartbeat's first tick lands before the
    operation starts, so a fully blocking operation shows its whole duration as the worst
    gap. The wall ends the moment ``op`` returns, inside the ``try``, before the
    ``finally`` joins the heartbeat, so it is the operation's wall, not the operation
    plus up to one 10ms heartbeat-drain ping (capturing it after the join deflated every
    measured ratio by up to ~9% at 12 MiB)."""
    ticks: list[float] = []
    stop = asyncio.Event()

    async def _heartbeat() -> None:
        while True:
            ticks.append(time.monotonic())
            if stop.is_set():
                return
            await asyncio.sleep(_PING_S)

    heartbeat = asyncio.create_task(_heartbeat())
    await asyncio.sleep(0)
    started = time.monotonic()
    try:
        await op()
        end = time.monotonic()
    finally:
        stop.set()
        await heartbeat
    wall = end - started
    worst_gap = max((b - a for a, b in itertools.pairwise(ticks)), default=0.0)
    return worst_gap, wall


def _budget_misses(gap: float, wall: float, ratio_budget: float | None) -> list[str]:
    """The budgets a sample missed: the single source for what "clean" means. A sample
    is clean exactly when this list is empty (worst gap under both the absolute
    ceiling and the ratio budget), and the failure detail reports the same list, so
    the check and the message can never disagree on the boundary. ``ratio_budget``
    is ``None`` only for cells whose wall sits under the 10ms ping floor, where any
    ratio is an artifact (see the module docstring); there the ceiling alone is the
    assertion."""
    missed: list[str] = []
    if gap >= _ABS_CEILING_S:
        missed.append(f"the {_ABS_CEILING_S * 1000:.0f}ms ceiling")
    if ratio_budget is not None and gap >= ratio_budget * wall:
        missed.append(f"the {ratio_budget:.0%} ratio budget")
    return missed


async def _assert_loop_stays_responsive(
    op: Callable[[], Awaitable[object]],
    ratio_budget: float | None = _RATIO_BUDGET,
    samples: int = _SAMPLES,
) -> None:
    """Assert ``op`` leaves the event loop schedulable, over up to ``samples``
    measurements (default ``_SAMPLES``), passing on the first clean one (a
    sample is clean exactly when ``_budget_misses`` returns an empty list: the
    worst tick gap under both budgets). A GIL-held whole-text pass
    reproduces in every sample; whole-process CPU starvation does not, so a
    single starved sample is retried instead of failing the test outright.
    The absolute ceiling catches pathological regressions independently of
    the ratio; both budgets derived from the measured bands in this module's
    docstring; ``ratio_budget`` is per-cell (see ``_B64_RATIO_BUDGET``), and
    ``samples`` widens the retry window for the marshalling-heaviest cells
    (the chunking family's heavy members take 6): whether a sample's worst
    gap captures the end-of-call marshalling residue depends on tick
    alignment against that marshalling window, so a 3-sample window can miss
    the alignment; a wider window observes more alignments, which is a
    stricter test (more chances to catch a dirty one), never a looser one
    (the pass-on-first-clean semantics are unchanged)."""
    observed: list[tuple[float, float]] = []
    for _ in range(samples):
        worst_gap, wall = await _gap_and_wall_during(op)
        observed.append((worst_gap, wall))
        if not _budget_misses(worst_gap, wall, ratio_budget):
            return
    detail = "; ".join(
        f"blocked {gap * 1000:.0f}ms of a {wall_ * 1000:.0f}ms operation "
        f"({(gap / wall_ if wall_ else 0):.0%}, "
        f"over {' and '.join(_budget_misses(gap, wall_, ratio_budget))})"
        for gap, wall_ in observed
    )
    raise AssertionError(
        f"the event loop was blocked in every one of {samples} samples ({detail}): "
        "tors's py.detach release is not freeing the loop while it runs, or "
        "the return marshalling regressed out of its band (src/lib.rs, "
        "tests/test_gil_release.py)"
    )


@pytest.mark.parametrize(
    ("corpus_kind", "size_bytes", "ratio_budget"),
    [
        ("prose", 12 * _MIB, _QC_YES_12MIB_RATIO_BUDGET),
        ("prose", 32 * _MIB, _RATIO_BUDGET),
        ("decomposed", 12 * _MIB, _RATIO_BUDGET),
        ("decomposed", 32 * _MIB, _RATIO_BUDGET),
    ],
    ids=["prose-12MiB", "prose-32MiB", "decomposed-12MiB", "decomposed-32MiB"],
)
def test_finalize_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    corpus_kind: str, size_bytes: int, ratio_budget: float
) -> None:
    """The claim under test: with the GIL released for the whole native pass, a caller
    that off-loads ``tors.finalize`` to a thread (the wrap extraction-pipeline callers
    keep for the GIL-held boundary residue) gets a loop that ticks every ~10ms through
    the whole transform+hash, where the same work as pure Python starves the loop for
    its largest whole-text C call (91.6-271.5ms at these sizes, measured; see the
    module docstring). Parametrized over both corpus kinds the GIL model names:
    ``prose`` (ASCII, pins the O(output) marshalling band) and ``decomposed``
    (non-ASCII, pins the one-time O(input) first-call UTF-8 materialization; even
    sample 1 stays inside both budgets, and samples 2+ hit the cached borrow).

    note on the prose-12MiB cell's budget: the quick-check skip removed the
    NFC pass from this QC-Yes corpus, shrinking the wall ~3x (42-46ms measured)
    while the marshalling residue is unchanged; the per-cell 0.60 budget is
    derived from the new band (see ``_QC_YES_12MIB_RATIO_BUDGET``); a detach
    regression still fails it at ratio ~1.0."""
    corpus = _CORPORA[corpus_kind](size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.finalize, corpus),
            ratio_budget=ratio_budget,
        )
    )


@pytest.mark.parametrize(
    ("corpus_kind", "size_bytes", "ratio_budget"),
    [
        ("prose", 12 * _MIB, _QC_YES_12MIB_RATIO_BUDGET),
        ("prose", 32 * _MIB, _RATIO_BUDGET),
        ("decomposed", 12 * _MIB, _RATIO_BUDGET),
        ("decomposed", 32 * _MIB, _RATIO_BUDGET),
    ],
    ids=["prose-12MiB", "prose-32MiB", "decomposed-12MiB", "decomposed-32MiB"],
)
def test_finalize_utf8_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    corpus_kind: str, size_bytes: int, ratio_budget: float
) -> None:
    """The bytes-in claim: ``tors.finalize_utf8``'s ``PyBytes`` argument
    extraction is a zero-copy borrow of the immutable buffer, so there is no
    argument-materialization class at all, so even the non-ASCII (decomposed)
    corpus's first call sits in the plain marshalling band, where the str-in
    ``finalize`` twin pays its one-time O(input) materialization (13.9-15.6ms
    first-call gap at 12 MiB, 37.6-44.1ms at 32 MiB, measured; see the
    module docstring). Measured bands (ambient load 4.3-5.6, 5 samples per cell):
    prose 12 MiB: 10.3-13.0ms of 113-143ms walls (ratio 0.08-0.10);
    decomposed 12 MiB: 10.3-13.0ms of 116-131ms (0.09-0.10), sample 1 in the
    same band as samples 2+;
    prose 32 MiB: 13.5-20.7ms of 345-370ms (0.04-0.06);
    decomposed 32 MiB: 17.6-26.7ms of 350-405ms (0.04-0.07), again sample 1
    indistinguishable from the cached band, against the str-in twin's
    37.6-44.1ms first call.

    note on the prose-12MiB cell's budget: same recalibration as the
    str-in ``finalize`` cell (see ``_QC_YES_12MIB_RATIO_BUDGET``): the
    quick-check skip shrank the QC-Yes prose wall ~3x (34-45ms measured)
    while the bytes-in marshalling residue is unchanged; measured ratios
    0.27-0.34 against the new 0.60 budget. A detach regression still fails
    it at ratio ~1.0."""
    corpus = corpus_utf8(corpus_kind, size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.finalize_utf8, corpus),
            ratio_budget=ratio_budget,
        )
    )


@pytest.mark.parametrize(
    "size_bytes, ratio_budget",
    [(12 * _MIB, None), (32 * _MIB, None), (96 * _MIB, _B64_RATIO_BUDGET)],
    ids=["12MiB-ceiling-only", "32MiB-ceiling-only", "96MiB-ratio"],
)
def test_b64_encode_bytes_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int, ratio_budget: float | None
) -> None:
    """The b64 claim, in three cells with different jobs. The GIL-held residue
    is the marshalling of the 4/3x-sized ASCII output, structurally a larger
    fraction of the wall than finalize's residue, because the encode itself is so
    fast. Measured on the dev box (ambient load 4.3-5.6, 5 samples per cell,
    prose bytes):

    - 12 MiB: worst gaps 10.1-11.2ms (the ping floor plus ~1ms of marshalling a
      16 MiB output) against walls of only 5-9ms, and the wall sits under the 10ms
      ping floor, so the gap/wall ratio (1.0-2.3) is the artifact this suite
      already documents for sub-ping walls, not evidence of blocking; the cell
      asserts the 100ms ceiling (>=9x margin) and records the band.
    - 32 MiB: worst gaps 27.5-29.3ms of 48-50ms walls (ratio 0.57-0.58) on the
      dev box: the 43 MiB output's marshalling band at ~2.4GB/s under load.
      Ceiling-only since the 0.2.1 recalibration: on a fast quiet box the
      walls collapse to 12-19ms (barely above the 10ms ping floor) where
      the heartbeat's own 5-11ms run-to-run wobble decides the ratio (whole
      samples at ~1.0 when no tick lands inside the marshal window, ~0.4
      when one does: flaky with no code change either way). The ratio at
      this size no longer resolves, the same floor-resolution failure the
      diff near-identical cell recalibrated for; the cell keeps the 100ms
      ceiling (a several-fold marshalling blowout still trips it) and the
      detach-detection job moves to the 96 MiB cell, whose walls dominate
      the floor on every box.
    - 96 MiB: worst gaps 67.2-78.3ms of 139-161ms walls (ratio 0.46-0.50) on
      the dev box; 9.4-15.6ms of 40-52ms (0.23-0.33) on the fast box. This
      is the regression-detecting cell: the 0.80 b64 ratio budget holds with
      ~1.6x margin on both boxes, while a detach regression (GIL-held
      encode, one C call holding encode+marshalling together with no
      bytecode boundary) shows ratio ~1.0 in every sample and fails it; the
      ceiling holds with margin to spare on both boxes.
    - The b64 budget derivation: 0.80 is ~1.3x above the worst measured ratio
      and ~20% below the ~1.0 a GIL-held pass shows in every sample; the
      shared 0.30 is structurally unattainable for b64 because the fast
      encode makes the marshalling the dominant share of the wall
      (gap/wall -> marshal/(encode+marshal) ~ 0.6-0.75 as size grows,
      measured).
    - The red side, same placement: ``base64.b64encode(...).decode("ascii")``
      holds the GIL for the encode: measured 122.7-127.6ms of 188-199ms walls
      at 96 MiB, reproducing the ~150ms GIL-held b64encode observation at
      a 100MB document cap that motivated the function. The red expression's
      ratio at 12 MiB is window-dependent (1.00 in the quiet window the band
      was first recorded in; 0.63 under load) and at 32 MiB measures
      0.64-0.66, and both pass this cell's budgets, a structural property of the
      two-call expression (the eval loop can tick at the bytecode boundary
      between ``b64encode`` and ``decode("ascii")``, so the worst gap is the
      encode alone); the red-side cell below therefore runs the expression
      inline on the loop -- where the whole wall is the worst gap, ratio
      ~1.0 in every sample -- and asserts both budgets mechanically."""
    corpus = corpus_utf8("prose", size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.b64_encode_bytes, corpus),
            ratio_budget=ratio_budget,
        )
    )


def _stdlib_b64_expression(raw: bytes) -> str:
    """The stdlib expression ``tors.b64_encode_bytes`` replaces (the red side)."""
    return base64.b64encode(raw).decode("ascii")


def _stdlib_content_hash_expression(obj: object) -> str:
    """The stdlib expression ``tors.content_hash`` replaces (the red side):
    the exact canonical-form spelling the contract defines, hashed with
    hashlib."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


async def _call_inline_on_the_loop(fn: Callable[[], object]) -> object:
    """The inline-run red idiom (the module docstring's reference-finalize
    "run inline on the loop instead, it is ratio ~1.0" note): call the
    sync function directly on the event loop's thread instead of in a
    worker. The native pass still releases the GIL, but releasing the
    GIL only lets other threads run; it does not hand control back to
    the loop (tors/aio.py's module docstring is the whole writeup), so
    the call occupies the loop's own turn for its full wall and the
    worst heartbeat gap is the wall: the shape a lost ``py.detach``
    regression shows, which is what these red rows exist to trip."""
    return fn()


@pytest.mark.parametrize(
    ("cell", "size_bytes", "ratio_budget"),
    [
        ("reference-finalize", 12 * _MIB, _RATIO_BUDGET),
        ("reference-finalize", 32 * _MIB, _RATIO_BUDGET),
        ("stdlib-b64-encode", 96 * _MIB, _B64_RATIO_BUDGET),
        ("stdlib-content-hash", 12 * _MIB, _CONTENT_HASH_RATIO_BUDGET),
        ("inline-chunk_text", 12 * _MIB, None),
        ("inline-chunk_by_words", 12 * _MIB, None),
        ("inline-chunk_by_sentences", 12 * _MIB, None),
        ("inline-chunk_hierarchical_default", 12 * _MIB, None),
        ("inline-chunk_hierarchical_line_first", 12 * _MIB, None),
    ],
    ids=[
        "ref-finalize-12MiB",
        "ref-finalize-32MiB",
        "inline-stdlib-b64-96MiB",
        "inline-stdlib-content-hash-12MiB",
        "inline-chunk_text-12MiB",
        "inline-chunk_by_words-12MiB",
        "inline-chunk_by_sentences-12MiB",
        "inline-chunk_hierarchical-default-12MiB",
        "inline-chunk_hierarchical-line-first-12MiB",
    ],
)
def test_the_gil_held_red_sides_fail_their_budgets_in_every_sample(
    cell: str, size_bytes: int, ratio_budget: float | None
) -> None:
    """The red side, asserted mechanically: the same
    budgets tors's cells above pass must be failed by the GIL-held
    expressions those cells replace (or, for the chunking family's
    rows, by tors's own calls with their detach effectively undone, run
    inline on the loop), otherwise the budgets would have no
    discriminating power and a tors regression into GIL-held behavior would
    sail through the same numbers. Every sample of every parametrized red
    cell must miss at least one budget, judged by the same
    ``_budget_misses`` list the green cells use. The first two cell kinds
    measured on the dev box (ambient load ~3.5-5, 3 samples per cell):

    - ``reference_finalize`` (the pure-Python pipeline) in the same
      to_thread placement: at 12 MiB, 95.1-98.0ms gaps of 161-165ms walls
      (ratio 0.58-0.60), missing the 0.30 ratio budget by ~2x in every
      sample; at 32 MiB, 238-259ms of 447-481ms (0.53-0.55), missing both the
      ratio budget and the 100ms ceiling (~2.4x over).
    - The stdlib b64 expression at 96 MiB, run inline on the loop (the
      family rows' idiom, not the finalize reds' to_thread placement):
      the whole two-call wall is the worst gap, ratio ~1.0 in every
      sample, missing both the 0.80 ratio budget and the 100ms ceiling
      (walls 139-199ms measured) -- the one-call lost-detach shape the
      96 MiB tors cell's budgets exist to discriminate. Why not
      to_thread: the expression is two C calls with an eval-loop bytecode
      boundary between them, so a worker thread's loop ticks between
      encode and ``decode("ascii")`` and the worst gap is the encode
      alone (122.3-127.6ms of 188-194ms walls, ratio 0.65-0.66, on the
      dev box the row was first calibrated on; 94-95ms on fast CI
      runners) -- structurally under the 0.80 budget at every size, which
      left the 100ms ceiling as the row's only discriminator with a
      box-speed-dependent margin, and the row measured a CLEAN sample on
      CI twice with no code change: the flake that moved it inline.
    - The stdlib content_hash expression (``sha256(json.dumps(...,
      sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()``
      over the records corpus) at 12 MiB, run inline on the loop: worst
      gaps 44.9-50.0ms of 44.8-49.4ms walls (ratio 1.00-1.02, measured
      under ambient load 10-16), missing the 0.80 ratio budget in every
      sample on any box speed -- ``json.dumps`` + ``str.encode`` are two
      GIL-held C calls covering ~the whole wall, the one-call lost-detach
      shape the green content_hash cell's budget exists to discriminate.
      The 100ms ceiling does NOT discriminate here (walls ~45ms under it
      even on this box), so the ratio budget is the row's only budget,
      which is why the row carries the green cell's own 0.80 rather than
      a ceiling-only ``None``: a tors detach regression pins the whole
      walk+emit wall (the tors inline shape measured ratio 1.00 in every
      sample) and fails it regardless of how fast the box runs the work.
      In the to_thread placement the same expression measures ratio 0.89
      (the worker's loop ticks between the three calls), under the 0.80
      budget by only ~10%: box-dependent, which is why the row runs
      inline (the b64 96 MiB row's rationale).
    - The chunking family's heavy members (``inline-*`` rows), run inline
      on the event loop via ``_call_inline_on_the_loop``: the green
      family cell's own calls at its own params
      (``_chunk_family_calls``, chatlog 12 MiB), just placed on the loop
      instead of in a worker thread: the worst gap is the wall (ratio
      ~1.00 every sample), so the 100ms ceiling (the family cell's
      only budget) is missed in every sample, which is the mechanical
      proof that the ceiling-only design actually discriminates a lost
      detach. The rows carry ``ratio_budget=None`` to mirror the green
      cell's ceiling-only budgets exactly. Measured (3 samples per
      row): ``chunk_text`` 396.6-406.6ms of 396.6-406.5ms walls (~4.0x
      the ceiling); ``chunk_by_words`` 170.0-173.4ms of 170.0-173.3ms
      (~1.7x); ``chunk_by_sentences`` 183.6-187.2ms of 183.6-187.1ms
      (~1.8x); ``chunk_hierarchical`` default 196.9-199.2ms of
      196.8-199.1ms (~2.0x); line-first ``["\n", None]`` at
      ``max_chars=40`` 410.6-414.6ms of 410.6-414.4ms (~4.1x).       The
      light members (paragraphs/lines and their ``_iter`` drains,
      ~3.4-5.5ms walls) are deliberately not rows here: an inline hold
      of a sub-10ms wall passes the ceiling, so no budget the green
      cell uses could discriminate their hold, the limitation the
      green cell's docstring already states.

    Measured and not asserted: the 32 MiB b64 red side, 41.5-42.5ms of 63-65ms
    walls (ratio 0.64-0.66), and 12 MiB 15.3-16.5ms of 24-26ms (0.63);
    both pass the b64 cells' budgets. The structural reason, and why the
    tors cells still discriminate: the stdlib expression is two C calls
    with an eval-loop bytecode boundary between them, so the loop ticks
    between encode and ``decode("ascii")`` and the worst gap is the encode
    alone, under both budgets at those sizes. A tors detach regression is
    one C call (encode and marshalling held together, no bytecode
    boundary), which pins the whole wall: the 96 MiB tors cell's measured
    band is 0.46-0.50 (dev box) and 0.23-0.33 (fast box) against its 0.80
    budget, and a one-call hold of such a wall shows ratio ~1.0, the shape
    the single-C-call reds above demonstrate directly. The b64 budget's
    discriminating power for tors's own shape is real, but it rests on the
    one-call structure, not on the stdlib red side's two-call shape at
    small sizes; the same bytecode boundary is why the asserted 96 MiB
    row runs inline (the to_thread placement's worst gap is the encode
    alone, under the 0.80 budget, its ceiling margin box-speed-dependent),
    recorded here so the next reader does not mistake the 12/32 MiB b64
    red sides for regression-proof."""
    if cell == "stdlib-b64-encode":
        # The b64 red side, inline on the loop (the family rows' idiom),
        # not in the finalize reds' to_thread placement: the stdlib
        # expression is two C calls with an eval-loop bytecode boundary
        # between them, so a worker thread's loop ticks between encode
        # and decode("ascii") and the worst gap is the encode alone --
        # structurally ~0.5 of the wall, under this row's 0.80 ratio
        # budget at every size, which left the 100ms ceiling as the row's
        # only discriminator and its margin box-speed-dependent (122-127ms
        # encodes on the dev box the row was calibrated on; 94-95ms on
        # fast CI runners, where the row measured a CLEAN sample twice
        # with no code change). Inline, the loop thread is inside the
        # expression for its whole wall: the worst gap is the wall, ratio
        # ~1.0 in every sample on every box, missing both the 0.80 budget
        # and the 100ms ceiling (139-199ms walls measured) -- the
        # one-call lost-detach shape the 96 MiB tors cell's budgets exist
        # to discriminate.
        corpus = corpus_utf8("prose", size_bytes)
        observed = [
            asyncio.run(
                _gap_and_wall_during(
                    lambda: _call_inline_on_the_loop(lambda: _stdlib_b64_expression(corpus))
                )
            )
            for _ in range(_SAMPLES)
        ]
    elif cell == "stdlib-content-hash":
        # The stdlib content_hash expression, inline on the loop (the b64
        # row's placement rationale): json.dumps and str.encode are
        # GIL-held C calls covering ~the whole wall, so inline the worst
        # gap IS the wall (ratio ~1.0 in every sample on every box,
        # missing the 0.80 ratio budget), while the to_thread placement
        # only reaches ~0.89 (the loop ticks between the three calls) --
        # box-dependent margin, not a discriminator.
        obj = content_object(size_bytes)
        observed = [
            asyncio.run(
                _gap_and_wall_during(
                    lambda: _call_inline_on_the_loop(lambda: _stdlib_content_hash_expression(obj))
                )
            )
            for _ in range(_SAMPLES)
        ]
    elif cell.startswith("inline-"):
        # The family's inline-hold rows: the green cell's own call, at
        # the same params, run directly on the loop's thread (the lost-
        # detach shape _call_inline_on_the_loop's docs describe).
        red_call = _chunk_family_calls(_CORPORA["chatlog"](size_bytes))[
            cell.removeprefix("inline-")
        ]
        observed = [
            asyncio.run(_gap_and_wall_during(lambda: _call_inline_on_the_loop(red_call)))
            for _ in range(_SAMPLES)
        ]
    else:
        corpus = prose(size_bytes)
        observed = [
            asyncio.run(_gap_and_wall_during(lambda: asyncio.to_thread(reference_finalize, corpus)))
            for _ in range(_SAMPLES)
        ]
    for gap, wall in observed:
        assert _budget_misses(gap, wall, ratio_budget), (
            f"the {cell} red side at {size_bytes // _MIB} MiB measured a CLEAN "
            f"sample ({gap * 1000:.0f}ms of a {wall * 1000:.0f}ms wall): the "
            "budgets tors's cells pass no longer discriminate against the "
            "GIL-held expression they replace, and a detach regression could "
            "pass them too (tests/test_gil_release.py)"
        )


@pytest.mark.parametrize("form_name", ["nfc", "nfkd"])
@pytest.mark.parametrize("size_bytes", [12 * _MIB, 32 * _MIB], ids=["12MiB", "32MiB"])
def test_standalone_forms_full_pass_on_the_compat_corpus_keep_the_loop_at_heartbeat_granularity(
    form_name: str, size_bytes: int
) -> None:
    """The standalone-forms GIL claim, on the corpus that still
    pays it: the whole transform runs under ``py.detach`` (same pattern as
    ``tors.normalize``), so the residue is the str-in call's two known bands:
    the O(output) return marshalling, plus the one-time O(input) UTF-8
    materialization on the first non-ASCII call. Measured on the dev box
    (ambient load 4.4-5.7, 3 samples per cell, compat corpus: decomposed
    accents plus a compatibility ligature and fullwidth digit per unit, the
    input that keeps NFKC/NFKD's quick check at No):

    - ``nfc`` 12 MiB: 16.8-17.4ms of 127-145ms walls (0.10-0.13): the
      sample-1 gap carries the materialization, the same class every str-in
      function pays.
    - ``nfkd`` 12 MiB: 14.7-17.8ms of 87-104ms walls (0.14-0.20).
    - ``nfc`` 32 MiB: 36.3-40.1ms of 346-391ms walls (0.10-0.12).
    - ``nfkd`` 32 MiB: 35.7-38.2ms of 240-263ms walls (0.15-0.16).

    Every measured sample sits under both shared budgets (worst ratio 0.20 vs
    the 0.30 budget; worst gap 40.1ms vs the 100ms ceiling), and the red side
    is the whole reason the functions exist: ``unicodedata.normalize`` is one
    GIL-held C call for the entire text (the reference finalize's 91.6-271.5ms
    gaps in the module docstring are set by exactly that call).

    Why this corpus replaced the decomposed one (a finding, pinned by
    the fast-path cell below): plain decomposed prose carries no compatibility
    mappings, so under NFKC/NFKD it quick-checks Yes and the identity-return
    fast path hands it back untouched, so the D-forms' transform band cannot be
    measured on it anymore."""
    corpus = compat(size_bytes)
    form = {"nfc": tors.nfc, "nfkd": tors.nfkd}[form_name]
    asyncio.run(_assert_loop_stays_responsive(lambda: asyncio.to_thread(form, corpus)))


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_d_form_fast_path_on_the_decomposed_corpus_is_ceiling_only(
    size_bytes: int,
) -> None:
    """The quick-check fast path's own band: under NFKD the decomposed
    corpus (decomposed accents, ASCII otherwise, no compatibility mappings)
    quick-checks Yes, so ``tors.nfkd`` returns the input object; the
    transform is the quick-check scan (2.4ms at 12 MiB, measured) plus the
    str-in first-call O(input) UTF-8 materialization, and nothing else.
    Measured on the dev box (ambient load 4.4-5.7, 3 samples): worst gaps
    10.4-13.7ms of 10-23ms walls: the wall sits at the 10ms ping floor, so
    the gap/wall ratio (0.59-1.13) is the documented sub-ping artifact (the
    b64 12 MiB precedent), and the cell asserts the 100ms ceiling (~9x
    margin) only. ``nfc`` on the same corpus still pays its full pass (the
    combining marks keep NFC's quick check at Maybe); that band is covered
    by the compat cell above, whose nfc leg runs the same pass shape."""
    corpus = decomposed(size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.nfkd, corpus),
            ratio_budget=None,
        )
    )


@pytest.mark.parametrize("fn_name", ["normalize", "finalize"])
@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_identity_paths_on_already_normalized_prose_are_ceiling_only(
    fn_name: str, size_bytes: int
) -> None:
    """The identity-return contract's GIL band, the zero-cost path in
    the crate GIL model: on already-normalized text (this corpus is the
    pipeline's own output: ``reference_normalize(prose(...))``), the complete
    transform is provably a no-op (quick-check Yes + every scan stage's
    fingerprint absent), so the original object comes back (no allocation,
    no copy, no marshalling), and the only GIL-held residue is the argument
    borrow itself (zero-copy for the ASCII corpus). ``finalize`` additionally
    computes SHA-256 over the borrowed input, detached.

    Measured on the dev box (ambient load 4.4-4.6, 3 samples per cell):
    ``normalize`` worst gaps 10.4-10.9ms of 3-4ms walls, ``finalize``
    10.9ms of ~9ms walls, and the wall is at or under the 10ms ping floor (the
    whole call is now cheaper than one heartbeat), so the ratio is the
    documented sub-ping artifact and the cell asserts the 100ms ceiling
    (~9-10x margin) only. The wall collapse this pins, for the record:
    normalize 123.0ms -> 2.89ms, finalize 126.3ms -> 7.72ms (min-of-5,
    build vs build; see the ledger tables for loads)."""
    corpus = reference_normalize(prose(size_bytes))
    fn = {"normalize": tors.normalize, "finalize": tors.finalize}[fn_name]
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(fn, corpus),
            ratio_budget=None,
        )
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB, 32 * _MIB], ids=["12MiB", "32MiB"])
def test_html_unescape_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The html claim: the whole entity scan runs under ``py.detach``, where
    ``html.unescape`` is a GIL-held ``re.sub`` over the whole string
    with a Python callback invoked per entity. Measured on the dev box
    (ambient load 10.0, the heaviest window in this module's tables; 3
    samples per cell, entities corpus):

    - 12 MiB: worst gaps 16.4-17.6ms of 79-88ms walls (ratio 0.19-0.22):
      the ping floor plus the marshalling of the ~12 MiB decoded output;
      every sample inside both shared budgets (~1.4x ratio margin under this
      load, more when quiet).
    - 32 MiB: worst gaps 28.6-40.9ms of 223-225ms walls (ratio 0.13-0.18).

    The red side, same placement, is not ratio ~1.0 like the other stdlib
    red sides, and the reason is the function's own structure:
    ``html.unescape``'s per-entity ``_replace_charref`` callback is Python
    bytecode, and the eval loop checks the GIL drop request between
    bytecodes, so ~800k callbacks per 12 MiB give the interpreter frequent
    (if tiny) yield points; measured worst gaps 35.7-49.4ms of 236-250ms
    walls (ratio 0.15-0.20). tors still roughly halves the worst gap
    (16.4-17.6 vs 35.7-49.4ms) and is ~3x faster in wall time (the
    callbacks are the cost); the accurate statement is "half the block time,
    a third of the wall", not "unblocked vs fully blocked"."""
    corpus = entities(size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(lambda: asyncio.to_thread(tors.html_unescape, corpus))
    )


@pytest.mark.parametrize(
    "size_bytes, ratio_budget",
    [(12 * _MIB, None), (32 * _MIB, _B64_RATIO_BUDGET)],
    ids=["12MiB-ceiling-only", "32MiB-both-budgets"],
)
def test_b64_decode_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int, ratio_budget: float | None
) -> None:
    """The decode claim (input: the b64-encoded corpora from
    ``reference.corpus_b64``, a 4/3x-sized ASCII str; output: the decoded
    bytes). The GIL-held residue is the marshalling of the decoded bytes
    (O(output), 3/4 of the input text), the same structural shape as
    ``b64_encode_bytes``'s output marshalling, so it uses the same b64 budget
    derivation (the shared 0.30 is unattainable for the same reason: the scan
    is fast, so the marshalling is a large wall share). Measured on the dev box
    (ambient load 6.0, 3 samples per cell, prose):

    - 12 MiB decoded (16 MiB b64 text): worst gaps 11.0-11.7ms against
      14.0-23.2ms walls (ratio 0.50-0.78): the wall sits near the 10ms ping
      floor, so the ratio is the documented sub-ping artifact; the cell asserts
      the 100ms ceiling (~9x margin) and records the band.
    - 32 MiB decoded (43 MiB b64 text): worst gaps 16.6-22.7ms of 59.1-65.3ms
      walls (ratio 0.28-0.35), comfortably inside both the 0.80 b64 budget
      (~2.3x margin) and the 100ms ceiling (~4.4x margin). A detach regression
      shows ratio ~1.0 (the red side below) and fails this cell.
    - The red side, same placement: ``base64.b64decode`` holds the GIL for the
      whole C decode: 14.8-27.5ms gaps of 15.0-27.8ms walls (ratio 0.98-0.99)
      in every 12 MiB sample, measured alongside the tors cells above."""
    corpus = corpus_b64("prose", size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.b64_decode, corpus),
            ratio_budget=ratio_budget,
        )
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_repair_json_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The json-repair claim: the whole repair pass (fence pre-pass, strict
    probe, repair parser, serializer) runs under ``py.detach``, and the
    return is one string, so the loop stays at heartbeat granularity while
    megabytes of damaged JSON are repaired. Measured on the dev box
    (ambient load ~3, 3 samples per cell): prose 12 MiB of structurally
    damaged LLM-JSON (unterminated strings, missing commas and closers):
    worst gap 10.2ms of a 122ms wall (0.08): the ping floor plus the
    str-out marshalling band, deep inside both shared budgets (~3x on the
    ratio, ~8x on the ceiling)."""
    chunks, _remainder = divmod(size_bytes, 20_000)
    payload = (
        "{" + "".join(f'"k{i}": "{_CORPORA["prose"](20_000)}, ' for i in range(chunks)) + '"t": 1'
    )
    assert len(payload) >= 11 * _MIB  # scale sanity, not a budget
    asyncio.run(_assert_loop_stays_responsive(lambda: asyncio.to_thread(tors.repair_json, payload)))


@pytest.mark.parametrize("corpus_kind", ["prose", "decomposed"])
@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_grapheme_count_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    corpus_kind: str, size_bytes: int
) -> None:
    """The grapheme claim: the whole cluster scan runs under
    ``py.detach`` and the return is a single int, so no list marshalling at all.
    Measured on the dev box (ambient load 8.3, 3 samples per cell):

    - prose 12 MiB: worst gap 10.4ms of a 129ms wall (0.08): the ping floor
      plus the int return; nothing else to hold.
    - decomposed 12 MiB: first call 12.8ms of 134ms (0.10, the str-in
      one-time O(input) UTF-8 materialization on top of the floor, the same
      class every str-in function pays), cached 10.5ms of 122ms (0.09).

    Every sample sits deep inside both shared budgets (~3x on the ratio,
    ~8x on the ceiling)."""
    corpus = _CORPORA[corpus_kind](size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(lambda: asyncio.to_thread(tors.grapheme_count, corpus))
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_word_bounds_marshalling_band_is_pinned_against_regression(
    size_bytes: int,
) -> None:
    """The word_bounds marshalling finding, as a regression ceiling:
    not the suite's 100ms ceiling, which this API shape cannot
    meet at whole-file sizes, and this cell exists to say so.

    The measurement (dev box, ambient load 8.3, 3 samples, prose 12 MiB =
    12.58M chars = 3,665,242 word segments): worst gaps 428-497ms of
    591-671ms walls (ratio 0.72-0.74): the GIL-held construction of 3.67M
    2-tuples and 7.33M ints dominates the call. The segmentation itself runs
    detached; it is the return marshalling that holds the loop, O(number of
    segments), so the 100ms ceiling is out of reach by ~4.5x for the
    list-returning shape at this size; a real, reported cost, not one to
    threshold away or hide.

    The API question this raises (recorded in docs/async.md and the
    report): a streaming shape for large inputs, either a lazy iterator
    (pyo3 `PyIterator` yielding `(start, end)` tuples in chunks, so the GIL
    is held only per-chunk), or an explicit `word_bounds_into(text, chunk)`
    callback/`count_only` fast path. Until such an API exists, the guidance is
    the measured band: fine at document scale (a 100KB chunk is ~30k
    segments, ~4ms held), a half-second GIL hold at 12 MiB.

    The cell pins the band against regression with both budgets, pass-on-
    first-clean like every other cell in this file (a regression
    dirties every sample; a starved box dirties only the sample it hits):
    a sample is clean when its worst gap is under both the 1.0s
    marshalling-band ceiling (~2x above the measured 0.43-0.50s band; the
    marshalling is allocation-bound and stable under load) and the 0.85 ratio
    budget of its own wall. The ratio budget is the detach-regression
    discriminator: the measured band sits at
    ratio 0.72-0.74, while a detach regression (the segmentation itself
    GIL-held) holds essentially the whole wall (ratio ~1.0) with a gap of
    only ~0.4-0.6s at this size, which the 1.0s ceiling alone would let
    through; a simulated-regression probe measured exactly that
    (0.37-0.40s gaps at ratio 100%, passing the old ceiling-only assertion
    and failing this one)."""
    corpus = prose(size_bytes)
    observed = [
        asyncio.run(_gap_and_wall_during(lambda: asyncio.to_thread(tors.word_bounds, corpus)))
        for _ in range(_SAMPLES)
    ]
    for gap, wall in observed:
        if gap < _WORD_BOUNDS_CEILING_S and gap < _WORD_BOUNDS_RATIO_BUDGET * wall:
            return
    detail = "; ".join(
        f"blocked {gap * 1000:.0f}ms of a {wall * 1000:.0f}ms operation "
        f"({gap / wall:.0%}, over the 1.0s ceiling and/or the "
        f"{_WORD_BOUNDS_RATIO_BUDGET:.0%} ratio budget)"
        if wall
        else "n/a"
        for gap, wall in observed
    )
    raise AssertionError(
        f"the word_bounds band regressed in every one of {_SAMPLES} samples ({detail}): "
        "either the bounds-list marshalling blew past its measured band or the "
        "segmentation lost its detach (ratio ~1.0) (src/segmentation_impl.rs, "
        "tests/test_gil_release.py)"
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_word_bounds_iter_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The streaming answer to the word_bounds marshalling finding: the
    full consumption of ``tors.word_bounds_iter(text)`` over the same 12 MiB
    prose corpus (3.67M segments; construction plus every ``__next__``)
    against the suite's shared budgets (0.30 ratio, 100ms ceiling), which the
    list-returning shape structurally could not meet (428-497ms held, the
    cell above).

    The design (src/lib.rs, ``WordBoundsIter``): the whole segmentation
    (the same detached core pass the list API runs) computes the bounds
    Vec up front under one ``py.detach`` (GIL-free for its full duration;
    the Vec is 16 bytes per segment, ~59 MiB at this size, versus the list
    API's ~hundreds of MiB of Python tuples), and each ``__next__`` then
    holds the GIL only to construct one 2-tuple (µs-scale), so the worst
    heartbeat gap collapses back to the ping-floor band every other
    str-in cell sits in.

    The tradeoff, recorded in docs/performance.md,
    measured the opposite way round from the design's expectation: the
    full drain measured 347ms (min-of-3) against the list API's 724ms in
    the same process; the per-``__next__`` tuple path is faster per bound
    than the list-return conversion, so the iterator wins on both axes at
    this size (gap band above, wall ~2.1x). The list API stays the right
    shape for small inputs and one-shot batch work; the iterator is the
    shape for whole-file segmentation on a live loop.

    Measured on the dev box (ambient load 4.4-4.6, 3 samples): construction
    plus full drain: worst gaps 15.4ms of 327-358ms walls (0.04-0.05),
    inside both shared budgets; the construction's own detached pass is the
    same ~175ms core the list API runs."""
    corpus = prose(size_bytes)

    def consume() -> int:
        return sum(1 for _ in tors.word_bounds_iter(corpus))

    asyncio.run(_assert_loop_stays_responsive(lambda: asyncio.to_thread(consume)))


def _invalid_utf8_corpus(size_bytes: int) -> bytes:
    """The prose corpus as UTF-8 bytes with its final byte replaced by 0xFF,
    an invalid lead byte at the very end, so the validity scan still traverses
    the whole corpus before answering False (an invalid byte earlier in the
    buffer would let the validator stop early and hold proportionally less of
    the loop; end placement makes the cell pin the whole-input scan's detach)."""
    raw = corpus_utf8("prose", size_bytes)
    return raw[:-1] + b"\xff"


@pytest.mark.parametrize(
    ("corpus_kind", "size_bytes", "ratio_budget"),
    [
        ("valid", 12 * _MIB, None),
        ("valid", 32 * _MIB, None),
        ("invalid", 12 * _MIB, None),
        ("invalid", 32 * _MIB, None),
    ],
    ids=[
        "valid-12MiB-ceiling-only",
        "valid-32MiB-ceiling-only",
        "invalid-12MiB-ceiling-only",
        "invalid-32MiB-ceiling-only",
    ],
)
def test_utf8_is_valid_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    corpus_kind: str, size_bytes: int, ratio_budget: float | None
) -> None:
    """The validity claim: the whole SIMD scan runs under ``py.detach``,
    and the call's GIL-held residue is the argument borrow alone: the bytes-in
    family's extreme point (a ``bool`` return, so no marshalling class at all)
    with no exception path either (invalid input answers False; nothing ever
    raises), so the valid and invalid corpora's cells must sit in the same
    band. Both cells are ceiling-only by the b64 12 MiB precedent: validation
    is SIMD-fast, so the wall at these sizes sits far under the 10ms ping
    floor and any gap/wall ratio is the suite's documented sub-ping artifact;
    the 100ms ceiling alone is the assertion.

    Measured on the dev box (ambient load 1.7, 3 samples per cell): valid
    12 MiB worst gaps 10.9-11.5ms of 0.7-1.3ms walls; valid 32 MiB 11.3-11.5ms
    of 1.1-1.4ms; invalid 12 MiB 10.9-11.0ms of 0.7ms; invalid 32 MiB
    11.5-11.7ms of 1.3-1.4ms. Every gap is the 10ms ping floor plus ~1-1.7ms
    of to_thread dispatch and the argument borrow (there is no marshalling
    class to pay at all), and the walls (the SIMD scan plus the dispatch;
    the inline call itself measures 0.09-0.55ms) sit an order of magnitude
    under the ping floor, so the cells assert the 100ms ceiling (~9x margin)
    only, at both sizes and both corpus kinds."""
    corpus = (
        corpus_utf8("prose", size_bytes)
        if corpus_kind == "valid"
        else _invalid_utf8_corpus(size_bytes)
    )
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.utf8_is_valid, corpus),
            ratio_budget=ratio_budget,
        )
    )


@pytest.mark.parametrize(
    ("corpus_kind", "size_bytes", "ratio_budget"),
    [
        ("sparse", 12 * _MIB, None),
        ("false-positive", 12 * _MIB, None),
    ],
    ids=["sparse-12MiB-ceiling-only", "false-positive-12MiB-ceiling-only"],
)
def test_unescaped_scan_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    corpus_kind: str, size_bytes: int, ratio_budget: float | None
) -> None:
    """The escape-parity scan claim: the whole pass — the memmem occurrence
    loop and the per-hit backward run walk — runs under ``py.detach``, and
    the call's GIL-held residue is the two zero-copy
    ``PyBytes`` borrows alone (a ``bool``/``int`` return, so no marshalling
    class at all; the empty-needle ``ValueError`` is the only error path and
    it fires before the detach), the ``utf8_is_valid`` extreme point applied
    to search. Two corpus shapes: ``sparse`` (plain prose bytes, no
    occurrence: the pure scan) and ``false-positive`` (one literal
    ``\\\\u0000`` per sentence, every one of its ~72,520 occurrences behind
    an odd run and rejected: the full scan plus the per-hit parity work,
    no early exit — the worst case for both wall time and GIL release).

    Both cells ceiling-only by the b64 12 MiB / utf8_is_valid precedent:
    the scan is memchr-class (measured inline 0.25ms sparse / 0.79ms dense
    at 12 MiB on the calibration box), so the wall sits an order of
    magnitude under the 10ms ping floor and any gap/wall ratio is the
    suite's documented sub-ping artifact. Measured on the calibration box
    (macOS, 16 cores, ambient load ~6-17, 3 samples per cell): worst gaps
    10.3-11.1ms of 0.4-1.3ms walls, both shapes and both spellings (the
    module docstring's ledger). The 100ms ceiling alone is the assertion
    (~9x margin); the sub-floor limitation is the same as every
    ceiling-only cell's: a held ~1ms scan is invisible under the floor
    either way, so this cell pins that the scan leaves the loop at the
    floor at all, and the wall cells in tests/test_unescaped_scan.py carry
    the throughput side."""
    corpus = (
        corpus_utf8("prose", size_bytes)
        if corpus_kind == "sparse"
        else unescaped_false_positive(size_bytes)
    )
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.find_unescaped, corpus, UNESCAPED_NEEDLE),
            ratio_budget=ratio_budget,
        )
    )


@pytest.mark.parametrize(
    ("corpus_kind", "size_bytes", "ratio_budget"),
    [
        ("ascii", 12 * _MIB, None),
        ("non-ascii-first-call", 12 * _MIB, None),
    ],
    ids=["ascii-12MiB-ceiling-only", "non-ascii-first-call-12MiB-ceiling-only"],
)
def test_utf8_byte_len_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    corpus_kind: str, size_bytes: int, ratio_budget: float | None
) -> None:
    """The byte-count claim, and its honest limit: the call's only O(n)
    work is the str-in borrow itself — CPython materializes the UTF-8 view
    under the GIL on a non-ASCII object's first contact, the cold-cache
    case exactly (a prior ``encode`` does not warm it: ``encode`` reads
    this cache and never fills it, so only a str-in call ends the cold
    lane; there is no way to fill an object's cache without holding the
    GIL; the ``finalize`` cells' first-call class) — while everything past
    the borrow is O(1) (a field read) and the return is a single int, so
    there is no marshalling class and no error path past the borrow's own
    ``UnicodeEncodeError`` on lone surrogates. The call is GIL-HELD end to
    end, deliberately detached from nothing: #108 measured that the
    pre-fix nominal ``py.detach`` around the O(1) core starved a
    co-resident event loop completely — every nanosecond detach/re-attach
    bumped ``switch_number`` inside a waiting thread's ``take_gil`` window,
    so the loop never escalated its switch request (zero heartbeat ticks
    in a 2 s window) — so the detach was removed; see the starvation pin
    below and the wrapper's GIL-model docs.

    Two legs, both ceiling-only:

    - ``ascii`` (prose): the borrow is a zero-copy alias (compact ASCII
      data is its own UTF-8), so the whole call is O(1) end to end and
      the wall sits five orders of magnitude under the 10ms ping floor —
      the sub-ping artifact, the b64 12 MiB / utf8_is_valid precedent.
      The pin's limit, stated: nothing O(n) exists to detach, so this
      leg cannot discriminate a detach regression; it pins that the call
      leaves the loop at the floor at all, and the wall cells in
      tests/test_performance.py carry the O(1) band.
    - ``non-ascii-first-call`` (decomposed, a FRESH object per sample so
      every sample carries the worst case): the materialization itself
      is GIL-held O(n), measured ~4.7-5.7ms inline at 12 MiB on the
      calibration box (the encoder pass plus a malloc plus the second
      memcpy into the permanent cache, within ~10-20% of a cold
      ``encode`` of the same object). The gap/wall ratio here is ~1.0 by
      construction (the wall IS the GIL-held materialization — the
      D-form fast-path cell's situation, not a detach regression), so
      the 100ms ceiling alone is the assertion, holding ~20x margin at
      this size; the linear envelope (~0.4-0.5ms of GIL hold per MiB)
      puts a ~200 MiB non-ASCII string at the ceiling, the recorded
      scale guidance for this function's one heavy lane.
    """
    if corpus_kind == "ascii":
        corpus = _CORPORA["prose"](size_bytes)
        asyncio.run(
            _assert_loop_stays_responsive(
                lambda: asyncio.to_thread(tors.utf8_byte_len, corpus),
                ratio_budget=ratio_budget,
            )
        )
    else:
        copies = iter([_CORPORA["decomposed"](size_bytes) for _ in range(_SAMPLES)])
        asyncio.run(
            _assert_loop_stays_responsive(
                lambda: asyncio.to_thread(tors.utf8_byte_len, next(copies)),
                ratio_budget=ratio_budget,
            )
        )


# The #108 starvation probe's shape, pinned: a FRESH large non-ASCII str per
# op (``s[1:]`` — never the cached object), looped in a worker thread, against
# a 1ms heartbeat on the co-resident loop. Deterministic content (seeded rng,
# no hypothesis): the probe is a timing shape, not a data-shape one — every
# non-ASCII char class materializes the UTF-8 view the same way.
_STARVE_PROBE_TEXT = "".join(
    random.Random(7).choice("abc déf ü 日本語 🙂 ñ") for _ in range(65536)
)
_STARVE_PROBE_SECONDS = 2.0


def test_utf8_byte_len_fresh_object_worker_thread_does_not_starve_the_heartbeat() -> None:
    """#108's signature, pinned with a huge margin for CI noise. PRE-fix,
    this probe reproduced the issue's starvation 2/2 times: the worker's
    GIL-held UTF-8-cache materialization followed by a nanosecond
    ``py.detach``/re-attach bumped ``switch_number`` inside the loop
    thread's ``take_gil`` window on every call, so the loop never escalated
    its switch request and the 1ms heartbeat delivered ONE tick in a 2s
    window (first tick at 2.001s, both repetitions). POST-fix (no detach:
    the whole call was GIL-held anyway, so the detach bracketed no work and
    was pure cost) the heartbeat ticks ~continuously — measured first tick
    1-2ms, p99 gap ~1ms, the pure-GIL-held class exactly.

    The assertion is the FIRST tick only, and generous: the starvation
    signature is a multi-second first tick, so 500ms sits two-plus orders
    of magnitude above the healthy band (~1-12ms, the pure-GIL reference
    rows in the issue's table) while leaving a loaded CI runner every
    advantage. The tick-count floor below is a belt against a partial
    regression (first tick lucky, then starvation), not a throughput
    pin: ~250 ticks is the pure-GIL-held worst shape (the ``encode``
    reference row), so 10 is nowhere near any healthy regime's floor.
    The sibling lanes are NOT pinned here — ``utf16_byte_len``/
    ``utf8_is_valid``/``decode_utf8`` keep their detaches because theirs
    bracket real work (the gap cells above already pin them healthy)."""
    ticks: list[float] = []

    async def probe() -> None:
        loop = asyncio.get_running_loop()
        deadline = time.perf_counter() + _STARVE_PROBE_SECONDS
        start = time.perf_counter()

        def worker() -> None:
            while time.perf_counter() < deadline:
                tors.utf8_byte_len(_STARVE_PROBE_TEXT[1:])

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(0.001)
                ticks.append(time.perf_counter() - start)
                if ticks[-1] >= _STARVE_PROBE_SECONDS:
                    break

        hb = asyncio.create_task(heartbeat())
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            fut = loop.run_in_executor(ex, worker)
            await hb
            await fut

    asyncio.run(probe())
    assert len(ticks) > 10, f"heartbeat starved: {len(ticks)} ticks in {_STARVE_PROBE_SECONDS}s"
    assert ticks[0] < 0.5, (
        f"heartbeat first tick at {ticks[0]:.3f}s (the #108 starvation "
        f"signature is a multi-second first tick; {len(ticks)} ticks delivered)"
    )


@pytest.mark.parametrize(
    ("corpus_kind", "size_bytes", "ratio_budget"),
    [
        ("ascii", 12 * _MIB, None),
        ("non-ascii-first-call", 12 * _MIB, None),
    ],
    ids=["ascii-12MiB-ceiling-only", "non-ascii-first-call-12MiB-ceiling-only"],
)
def test_utf16_byte_len_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    corpus_kind: str, size_bytes: int, ratio_budget: float | None
) -> None:
    """The interop twin's cell, and the one honest difference from the
    utf8 twin's: the detach around this core carries REAL work (the
    O(n) byte-class scan, ~30 GB/s — measured ~0.4ms at 12 MiB, an
    order under the 10ms ping floor), which is why it keeps its
    ``py.detach`` — #108 removed the utf8 twin's only because ITS detach
    bracketed no work (an O(1) field read behind the borrow), the
    starvation mechanism the twin is structurally immune to. The
    GIL-held residue is the same
    borrow class: the cold-cache first call's materialization of the
    UTF-8 view (there is no way to fill an object's cache without
    holding the GIL; a prior ``encode`` does not warm it), with the
    scan detached behind it.

    Two legs, both ceiling-only, the twin's reasons:

    - ``ascii`` (prose): the borrow is a zero-copy alias and the scan
      is detached, so the whole call's GIL-held residue is call
      overhead — the wall (~0.4ms at 12 MiB) sits an order under the
      ping floor, and the pin's limit is the utf8 twin's: a sub-floor
      call cannot discriminate a detach regression by gap alone, it
      pins that the call leaves the loop at the floor at all (the wall
      cells in tests/test_performance.py carry the scan band).
    - ``non-ascii-first-call`` (decomposed, a FRESH object per sample):
      the materialization is GIL-held O(n) — the utf8 twin's measured
      ~4.7-5.7ms inline class at 12 MiB — with the detached scan
      (~0.4ms) behind it, so the worst gap is the materialization
      itself, still under the 10ms ping interval; the 100ms ceiling
      holds ~20x, and the linear envelope is the twin's (~0.4-0.5ms of
      GIL hold per MiB, a ~200 MiB non-ASCII string at the ceiling).
    """
    if corpus_kind == "ascii":
        corpus = _CORPORA["prose"](size_bytes)
        asyncio.run(
            _assert_loop_stays_responsive(
                lambda: asyncio.to_thread(tors.utf16_byte_len, corpus),
                ratio_budget=ratio_budget,
            )
        )
    else:
        copies = iter([_CORPORA["decomposed"](size_bytes) for _ in range(_SAMPLES)])
        asyncio.run(
            _assert_loop_stays_responsive(
                lambda: asyncio.to_thread(tors.utf16_byte_len, next(copies)),
                ratio_budget=ratio_budget,
            )
        )


@pytest.mark.parametrize("size_bytes", [32 * _MIB], ids=["32MiB"])
def test_diff_opcodes_near_identical_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The diff claim on the few-opcode shape: the whole diff (both
    operands' ``Vec<char>`` materialization and the Myers search) runs under
    ``py.detach``, so a near-identical pair (six scattered line edits,
    dozens of opcodes) leaves the loop ticking at heartbeat granularity
    while ~100ms of native diff work runs. Measured on the dev box (ambient
    load 5.3, 3 samples, 12 MiB pair): worst gaps 10.5-11.2ms of 76-88ms
    walls (ratio 0.12-0.14): the ping floor plus the two zero-copy ASCII
    argument borrows and a ~0.05ms opcode-tuple marshalling. Inside both
    shared budgets with ~2x ratio margin; a detach regression (the diff
    itself GIL-held) holds the whole wall at ratio ~1.0 and fails by ~3x.

    Why 32 MiB and not 12: the ratio only resolves when the wall clears
    the 10ms ping floor by a wide margin, and on a fast quiet box the 12
    MiB pair's ~30ms walls sit inside the heartbeat floor's own wobble
    (5-11ms gaps run to run: ratio 0.18 one run, 0.35 the next, against
    the same 0.30 budget: a measurement-resolution failure, not a code
    regression; tors's detach verified working throughout). At 32 MiB the
    walls (~100ms on that box, ~210ms implied for the dev box's pace)
    restore ~5x ratio margin (measured 0.06) on every box while a detach
    regression still holds the whole wall at ~1.0. The ceiling-only
    alternative was considered and rejected: on a fast box a held 30ms
    wall shows ~30ms gaps, under the 100ms ceiling, so ceiling-only would
    go blind exactly where the ratio is weakest: enlargement keeps both
    budgets discriminating everywhere."""
    a, b = diff_pair_near_identical(size_bytes)
    asyncio.run(_assert_loop_stays_responsive(lambda: asyncio.to_thread(tors.diff_opcodes, a, b)))


# The shuffled cell's marshalling-band regression ceiling: not the suite's
# 100ms ceiling, which a several-fold marshalling blowout at this op count
# would still clear (measured band 20.4-26.3ms; ~5x the ~10-15ms marshalling
# share lands at ~60-75ms < 100ms). 60ms sits ~2.3x above the measured worst
# gap and ~5x above the marshalling share itself; the word_bounds-precedent
# derivation (a bespoke ceiling ~2x above the measured band), sized to catch
# the blowout class (per-op PyString construction, per-op tuple churn) while
# tolerating ping-floor wobble under load. The shared 0.30 ratio budget is
# the detach-regression discriminator alongside it (a held diff shows ratio
# ~1.0 against these ~1.6s walls).
_DIFF_SHUFFLED_CEILING_S = 0.060


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_diff_opcodes_shuffled_pair_marshalling_band_is_pinned_against_regression(
    size_bytes: int,
) -> None:
    """The diff claim on the many-opcode shape: the O(ops) return
    marshalling made visible (the ``word_bounds`` list-shape class, applied to
    a function whose output is a list by necessity): the character-level diff
    of the 12 MiB shuffled pair (same content, different line order, the
    corpus that makes the opcode count large). The diff itself is detached
    (walls 1554-1626ms measured); the GIL-held residue is the construction of
    103,421 5-tuples with up to four fresh PyLongs each, the four tag strings
    pre-built once per call and shared by reference into every tuple.

    Measured on the dev box (ambient load 5.3-6.6, 3 samples): worst gaps
    20.4-26.3ms of 1554-1626ms walls (ratio 0.013-0.017). The marshalling
    itself is the delta over the near-identical cell's floor band:
    ~10-15ms for 103,421 opcodes, ~0.1-0.15µs per opcode.

    The cell pins the band pass-on-first-clean over both budgets: the 60ms
    bespoke ceiling (the word_bounds-precedent marshalling-band pin, catching
    a several-fold marshalling blowout the shared 100ms ceiling would let
    through at this op count) and the shared 0.30 ratio (a detach regression
    holds the whole ~1.6s wall). The op count is reported in the failure
    detail, never asserted; it is the algorithm's business: similar's
    bounded search trades minimality for speed on hard inputs, and a crate
    bump may legitimately change the count without changing validity (the
    structural contract lives in tests/test_diff_opcodes.py)."""
    a, b = diff_pair_shuffled(size_bytes)
    op_counts: list[int] = []

    def op() -> list[tuple[str, int, int, int, int]]:
        opcodes = tors.diff_opcodes(a, b)
        op_counts.append(len(opcodes))
        return opcodes

    observed = [
        asyncio.run(_gap_and_wall_during(lambda: asyncio.to_thread(op))) for _ in range(_SAMPLES)
    ]
    for gap, wall in observed:
        if gap < _DIFF_SHUFFLED_CEILING_S and gap < _RATIO_BUDGET * wall:
            return
    detail = "; ".join(
        f"blocked {gap * 1000:.0f}ms of a {wall_ * 1000:.0f}ms operation "
        f"({(gap / wall_ if wall_ else 0):.0%}, "
        f"over the {_DIFF_SHUFFLED_CEILING_S * 1000:.0f}ms marshalling ceiling "
        f"and/or the {_RATIO_BUDGET:.0%} ratio budget)"
        for gap, wall_ in observed
    )
    raise AssertionError(
        f"the diff_opcodes marshalling band regressed in every one of {_SAMPLES} "
        f"samples ({detail}; {op_counts[0]} opcodes in the pair): either the "
        "opcode-list marshalling blew past its measured band (~10-15ms above "
        "the ping floor for ~103k opcodes) or the diff lost its detach "
        "(ratio ~1.0) (src/diff_impl.rs, tests/test_gil_release.py)"
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_find_patterns_sparse_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The sparse-search claim: the whole search (automaton build, scan,
    and the byte→char conversion, idle here since the corpus is ASCII) runs
    under ``py.detach``, over the diff near-identical pair's edited corpus
    with the sparse terminology-scan pattern set: ``"monthly"`` occurs
    exactly once (the inserted line; the diff builder's four replace
    positions land on the corpus's empty separator lines at this size), the
    other two terms never. Measured on the dev box (ambient load 6.9-7.3, 3
    samples): worst gaps 10.3-10.7ms of 6-7ms walls: the ping floor plus
    one 3-tuple.

    Ceiling-only by the b64 12 MiB / utf8_is_valid precedent: the pure scan
    is ~2 GiB/s, so the wall sits under the 10ms ping floor and any
    gap/wall ratio is the suite's documented sub-ping artifact; the 100ms
    ceiling alone is the assertion (~10x margin). A limitation:
    a ~6ms scan held or released is invisible under the floor
    either way, so this cell cannot by itself discriminate a detach
    regression; the dense cell below carries that (a held dense pass shows
    ratio ~1.0 against its ~215ms wall); this cell pins that the no-match
    scan leaves the loop at the floor at all."""
    edited = diff_pair_near_identical(size_bytes)[1]
    patterns = list(SEARCH_SPARSE_PATTERNS)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.find_patterns, patterns, edited),
            ratio_budget=None,
        )
    )


# The dense cell's budgets: the word_bounds-precedent marshalling-band pin,
# with the ratio recalibrated to this cell's structurally higher marshalling
# share: the search core is ~3.5x faster than segmentation, so the O(matches)
# marshalling is a larger fraction of the wall (measured 0.76-0.80, against
# word_bounds' 0.72-0.74). 0.90 sits ~1.125x above the worst measured ratio
# and ~10% below the ~1.0 a detach regression shows; the same derivation
# shape as _WORD_BOUNDS_RATIO_BUDGET, sized for this band. The 400ms ceiling
# is ~2.3x above the measured 171.0-174.7ms worst gaps (the diff-shuffled
# cell's margin), sized to catch a several-fold marshalling blowout the
# shared 100ms ceiling is not sized for at 1.28M matches.
_SEARCH_DENSE_CEILING_S = 0.400
_SEARCH_DENSE_RATIO_BUDGET = 0.90


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_find_patterns_dense_marshalling_band_is_pinned_against_regression(
    size_bytes: int,
) -> None:
    """The dense-search claim: the O(matches) return marshalling made
    visible (the ``word_bounds``/``diff_opcodes`` list-shape class, applied
    to a function whose output is a list by necessity): the 17-word dense
    pattern set over the plain 12 MiB prose corpus, 1,284,724 matches. The
    search itself is detached (walls 214-224ms measured); the GIL-held
    residue is the construction of 1.28M 3-tuples of ints.

    Measured on the dev box (ambient load 6.9-7.3, 3 samples): worst gaps
    171.0-174.7ms of 214-224ms walls (ratio 0.76-0.80). The marshalling
    itself is the delta over the sparse cell's floor band: ~160-165ms for
    1.28M 3-tuples, ~0.13µs per match.

    The cell pins the band pass-on-first-clean over both budgets: the 400ms
    bespoke ceiling (the word_bounds-precedent marshalling-band pin) and the
    0.90 ratio budget (the detach discriminator: a held search shows ratio
    ~1.0 against these ~215ms walls; see the constant's comment for why 0.90
    and not word_bounds' 0.85). The match count is reported in the failure
    detail, never asserted; it is the corpus's business, not a contract.

    Guidance (the word_bounds finding's shape, recorded in
    docs/performance.md): ~0.13µs of GIL hold per match: ~13ms at 100k
    matches (document scale), ~170ms at 1.28M (whole-corpus keyword sweeps);
    callers producing millions of matches are the streaming-API question the
    word_bounds finding already raised."""
    text = prose(size_bytes)
    patterns = list(SEARCH_DENSE_PATTERNS)
    match_counts: list[int] = []

    def op() -> list[tuple[int, int, int]]:
        matches = tors.find_patterns(patterns, text)
        match_counts.append(len(matches))
        return matches

    observed = [
        asyncio.run(_gap_and_wall_during(lambda: asyncio.to_thread(op))) for _ in range(_SAMPLES)
    ]
    for gap, wall in observed:
        if gap < _SEARCH_DENSE_CEILING_S and gap < _SEARCH_DENSE_RATIO_BUDGET * wall:
            return
    detail = "; ".join(
        f"blocked {gap * 1000:.0f}ms of a {wall_ * 1000:.0f}ms operation "
        f"({(gap / wall_ if wall_ else 0):.0%}, "
        f"over the {_SEARCH_DENSE_CEILING_S * 1000:.0f}ms marshalling ceiling "
        f"and/or the {_SEARCH_DENSE_RATIO_BUDGET:.0%} ratio budget)"
        for gap, wall_ in observed
    )
    raise AssertionError(
        f"the find_patterns marshalling band regressed in every one of {_SAMPLES} "
        f"samples ({detail}; {match_counts[0]} matches in the corpus): either the "
        "match-list marshalling blew past its measured band (~160-165ms above "
        "the ping floor for ~1.28M matches) or the search lost its detach "
        "(ratio ~1.0) (src/search_impl.rs, tests/test_gil_release.py)"
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_replace_many_dense_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The replace claim on the redaction-map shape the function exists
    for: the 17 dense prose words each mapped to a redaction token, over the
    plain 12 MiB prose corpus: one automaton build, one scan, one splice,
    all detached, and the return is one ~10 MiB string, so the GIL-held
    residue is the O(entries) argument walk plus that single string's
    marshalling: the no-list-shape-class prediction of the crate GIL
    model, measured to be exactly the ping-floor band.

    Measured on the dev box (ambient load 2.0, 5 samples): worst gaps
    10.3-10.9ms of 47-60ms walls (ratio 0.18-0.23). The wall clears the
    10ms ping floor ~5x, so the ratio is not the sub-ping artifact, and the
    cell takes the shared budgets: the 0.30 ratio sits ~1.3x above the
    worst measured ratio and ~70% below the ~1.0 a detach regression shows
    (a held scan+splice pins the whole ~50ms wall), and the 100ms ceiling
    holds ~9x over the worst gap. The residue band is the structural
    contrast with the find_patterns dense cell above (171.0-174.7ms for
    the same 1.28M matches): reporting the matches as a list costs
    ~0.13µs of GIL hold per match, while splicing them into one output
    string costs ~nothing the loop can see."""
    text = prose(size_bytes)
    replacements = {word: "[REDACTED]" for word in SEARCH_DENSE_PATTERNS}
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.replace_many, text, replacements)
        )
    )


@pytest.mark.parametrize("size_bytes", [96 * _MIB], ids=["96MiB"])
def test_scrub_log_text_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The scrub claim, the replace_many dense cell's shape over the
    surface it exists for: the whole four-pass rule chain (DETAIL line
    scan, escaped-run scan, userinfo scan, query-param scan, plus the
    splice) detached under one ``py.detach``, over the exception-shaped
    scrub corpus where every rule fires once per unit, and the return is
    one string, so the GIL-held residue is the argument borrow plus that
    single string's marshalling — ``detached_transform``'s classes, the
    no-list-shape prediction again.

    The 96 MiB size is the cell's own derivation, not the suite's usual
    12 MiB: the scrub core is memchr/memmem-scanned Rust, so 12 MiB walls
    only 8-10ms — at the 10ms ping floor, where the ratio is the
    documented sub-ping artifact (the b64 12 MiB precedent) and cannot
    carry the shared budget. 96 MiB walls 56-72ms, clearing the floor ~6x.

    Measured on the dev box (ambient load ~4, 3 samples per side): worst
    gaps 11.1-13.6ms of 56-72ms walls (ratio 0.19-0.20) — the ping floor
    plus the one output string's marshalling (the DETAIL deletions leave
    the output ~87% of the input). The cell takes the shared budgets: the
    0.30 ratio sits ~1.5x above the worst measured ratio and ~70% below
    the ~1.0 a detach regression shows, and the 100ms ceiling holds ~7x
    over the worst gap. The red side, measured in the same placement: the
    four-pass ``re.sub`` chain this port replaces holds the loop for
    2487-2489ms of 2759-2762ms walls (ratio 0.90) over the same corpus —
    ``re.sub`` never releases the GIL, the exact GIL-tax the port exists
    to remove (up to four passes per text and ~24 per failed job in the
    consumer's error path)."""
    text = scrub_corpus(size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(lambda: asyncio.to_thread(tors.scrub_log_text, text))
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_scrub_pii_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The scrub claim on the contact-dense shape the function exists
    for: one email and one human-spelled E.164 number per sentence of
    the 12 MiB contacts corpus (~119k matches measured), one email pass
    plus one phone pass plus every token digest, all under the one
    ``py.detach``, and the return is one ~11.8 MiB string, so the
    GIL-held residue is the argument borrow plus that single string's
    marshalling: ``replace_many`` dense's no-list-shape class exactly
    (the structural contrast with ``find_patterns``' per-match tuples).

    Measured on the dev box (ambient load ~3.5, 3 samples): worst gaps
    ~12ms of 36-40ms walls (ratio 0.33-0.34, every sample) — the ping
    floor plus the end-of-call marshalling of the ~11.8 MiB result
    string. The scrub's double scan is fast enough that this residue is
    structurally a third of the wall, so the cell takes the bespoke
    0.60 ratio budget (``_SCRUB_PII_12MIB_RATIO_BUDGET``, the
    QC-Yes/b64 derivation shape: ~1.8x above the worst measured ratio,
    ~40% below the ~1.0 a detach regression shows when a held double
    scan pins the whole wall as one gap); the 100ms ceiling
    independently holds ~8x over the worst gap."""
    corpus = contacts(size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.scrub_pii, corpus),
            ratio_budget=_SCRUB_PII_12MIB_RATIO_BUDGET,
        )
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_sentence_bounds_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The sentence claim, list spelling: the whole UAX #29 sentence
    segmentation runs under ``py.detach`` (the word_bounds pair's GIL
    model exactly), and the GIL-held residue is the O(sentences) 2-tuple
    marshalling: 170,037 segments measured at 12 MiB, ~1/22nd of
    word_bounds' 3.67M on the same corpus (even sparser than the ~1/10th
    design estimate in src/lib.rs), which is why the list shape meets the
    shared budgets here where word_bounds' 428-497ms band could not.

    Measured on the dev box (ambient load 2.0, 5 samples): worst gaps
    17.6-23.8ms of 185-190ms walls (ratio 0.09-0.13): the ping floor
    plus ~8-14ms of 2-tuple construction for 170k segments (~0.05-0.08µs
    per segment, the word_bounds per-element band). Every sample inside
    both shared budgets (~2.3x ratio margin, ~4.2x ceiling margin); a
    detach regression (the segmentation itself GIL-held) shows ratio ~1.0
    against these ~190ms walls and fails by far. The streaming
    ``sentence_bounds_iter`` exists for whole-corpus sweeps that never
    materialize the list; its sequence parity is pinned in
    tests/test_sentence_bounds.py, and the shared-budget band here is the
    per-shape guidance for callers choosing between the two."""
    corpus = prose(size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(lambda: asyncio.to_thread(tors.sentence_bounds, corpus))
    )


@pytest.mark.parametrize(
    "member",
    [
        "chunk_text",
        "chunk_by_words",
        "chunk_by_words_iter",
        "chunk_by_sentences",
        "chunk_by_paragraphs",
        "chunk_by_paragraphs_iter",
        "chunk_by_lines",
        "chunk_by_lines_iter",
        "chunk_hierarchical_default",
        "chunk_hierarchical_line_first",
    ],
    ids=[
        "chunk_text-12MiB-ceiling-only",
        "chunk_by_words-12MiB-ceiling-only",
        "chunk_by_words_iter-12MiB-ceiling-only",
        "chunk_by_sentences-12MiB-ceiling-only",
        "chunk_by_paragraphs-12MiB-ceiling-only",
        "chunk_by_paragraphs_iter-12MiB-ceiling-only",
        "chunk_by_lines-12MiB-ceiling-only",
        "chunk_by_lines_iter-12MiB-ceiling-only",
        "chunk_hierarchical-default-12MiB-ceiling-only",
        "chunk_hierarchical-line-first-12MiB-ceiling-only",
    ],
)
def test_chunk_family_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    member: str,
) -> None:
    """The chunking family's GIL claim, pinned directly for the first time:
    every ``chunk_*`` function's docs say "GIL model: identical to
    ``chunk_by_words``" (the whole segmentation/window pass under one
    ``py.detach``, the return marshalling O(chunks) 2-tuples of ints under
    the GIL), and until this cell that claim was pinned only by
    inheritance from tested siblings (#30 item 5): no ``chunk_*``
    function had a cell of its own, before or after #28. One parametrized
    cell over the family's members (the ``test_utf8_is_valid`` cell's
    corpus-kind parametrize grain, applied to API members), every member
    over the same line-heavy chatlog corpus at 12 MiB: the
    chat-thread/log shape the ``chunk_by_lines_iter``/``chunk_by_words_iter``
    docstrings justify the streaming spellings with and the ``["\n", None]``
    hierarchy splice exists for: ~185k content lines, ~37k blank-line-
    separated paragraphs, ~4.48M word segments, ~222k sentences.

    Per-member parameters (``_chunk_family_calls``), sized for
    piece-count-heavy outputs where the O(chunks) marshalling is real
    but not the word_bounds 428-497ms list-shape class (measured piece
    counts at 12 MiB): ``chunk_text``/``chunk_hierarchical`` default at
    ``max_chars=200`` (64,278 / 111,024 pieces); ``chunk_by_words`` (and
    the ``_iter`` full drain) at 20 words per chunk (129,528 pieces,
    1/28th of word_bounds' 3.67M: the sentence_bounds band class, not
    the list-shape blowout); ``chunk_by_sentences`` at 5 sentences per
    chunk (44,410); ``chunk_by_paragraphs`` (and the ``_iter`` full
    drain) at 1 paragraph per chunk (37,008: the corpus's coarsest unit
    is a 5-line ~340-byte block, so 1-per-chunk is its piece-heaviest
    legal setting); ``chunk_by_lines`` (and the ``_iter`` full drain) at
    5 lines per chunk (37,008); and the line-first hierarchy leg at
    ``max_chars=40`` (370,080 pieces): 40, not the 200 its siblings
    use, because the lazy levels emptied the 200-budget leg: at 200
    every chatlog window is served by the "\n" literal level alone (no
    spliced level ever consulted, none of the spliced
    paragraph/sentence/word walks built), so the wall collapses to
    ~13ms: a full GIL-hold regression passes the 100ms ceiling there,
    and the member pins nothing; at 40 every ~66-68-byte line is
    oversized, windows descend past the line literal into the
    spliced levels, and the leg is the family's heaviest member again
    (~405ms, 370,080 pieces).

    Every member asserts the 100ms ceiling only (``ratio_budget=None``),
    the b64 12 MiB / utf8_is_valid / find_patterns-sparse precedent, for
    two documented reasons:

    - The light members (``chunk_by_paragraphs``/``chunk_by_lines``,
      each with its ``_iter`` full drain: single-pass newline scans
      behind the sliding ASCII certificate) wall at ~3.4-5.5ms (under
      the 10ms ping floor), where any gap/wall ratio is the suite's
      documented sub-ping artifact (measured 2.0-3.2). They carry the
      sparse cell's limitation: a wall this size held
      passes the ceiling too, so these members pin that the scans leave
      the loop at the floor at all, not a detach regression by
      themselves.
    - The heavy members (the UAX #29-backed ones: ``chunk_text``,
      ``chunk_by_words`` (+``_iter``), ``chunk_by_sentences``, and both
      ``chunk_hierarchical`` hierarchies) have walls dominated by their
      detached cores with marshalling residues far under the ceiling
      (bands below), so a detach regression holds the whole wall and
      blows the 100ms ceiling in every sample: the ceiling alone is the
      family's detach pin, which is what #30 item 5 asked this cell to
      pin, and the red-side cell above now asserts that mechanically
      for five of the six heavy spellings (run inline on the loop, they
      hold at ratio ~1.00 and miss the ceiling in every sample; the
      ``_iter`` drain shares ``chunk_by_words``'s core, so the list
      spelling's row carries the pair). A fast-box caveat, the diff
      near-identical cell's shape, stated for the record:
      ``chunk_by_words`` (~170ms), ``chunk_by_sentences`` (~188ms), and
      the default hierarchy (~200ms, the lazy levels halved it; a held
      wall is only ~2.0x the ceiling now, where the eager tree's
      347-416ms held ~3.5-4.2x) sit close enough that a box ~2x faster
      could pull a held wall under 100ms; ``chunk_text`` (~395-451ms)
      and the line-first leg at ``max_chars=40`` (~400-409ms, ~4.1x the
      ceiling when held) stay discriminating on every box this suite
      has measured.

    Measured (3 samples per member; the marshalling-heaviest members
    take 6: a 3-sample window can miss the end-of-call marshalling
    alignment; worst gap of wall, ratio in parentheses):

    - ``chunk_text``: 10.2-13.1ms of 395-451ms walls (0.02-0.03): the
      ping floor plus up to ~2.5ms of 2-tuple marshalling for 64,278
      pieces; ceiling margin ~7.6x.
    - ``chunk_by_words``: 10.4-19.4ms of 170-175ms walls (0.06-0.11):
      the floor plus ~0-9ms of GIL-held 2-tuple construction for
      129,528 pieces (~0.005-0.07µs per piece, the word_bounds
      per-element band); margin ~5.2x at the worst gap.
    - ``chunk_by_words_iter`` full drain: 10.6-11.9ms of 164-168ms
      walls (0.06-0.07): the floor plus ~0-1.5ms of per-``__next__``
      contention for the same 129,528 handoffs, the word_bounds_iter
      band's shape at 1/28th its piece count; margin ~8.4x.
    - ``chunk_by_sentences``: 10.3-10.7ms of 187-190ms walls
      (0.05-0.06); margin ~9.3x.
    - ``chunk_by_paragraphs``: 10.1-11.0ms of 3.4-3.9ms walls
      (2.7-3.2): the wall sits under the ping floor, so the ratio is
      the documented sub-ping artifact: the ceiling-only branch of
      this cell's design, measured; margin ~9.1x on the gap itself.
    - ``chunk_by_paragraphs_iter`` full drain: 10.5-11.1ms of 3.4-4.0ms
      walls (2.8-3.1, the same sub-ping artifact): the drain skips the
      list spelling's O(pieces) marshalling, but at 37,008 pieces that
      marshalling was only ever ~1ms, and the certificate scanners
      collapsed the list twin's own wall from the former 12-13ms to
      ~3.5ms, so the two spellings wall about the same here; the
      word_bounds_iter finding's per-``__next__`` advantage needs the
      3.67M-piece scale to show; margin ~9.0x.
    - ``chunk_by_lines``: 10.6-11.0ms of 4.4-5.5ms walls (2.0-2.4, the
      same artifact); margin ~9.1x.
    - ``chunk_by_lines_iter`` full drain: 10.5-10.6ms of 4.4-4.5ms
      walls (2.3-2.4, the same artifact); margin ~9.4x.
    - ``chunk_hierarchical`` default hierarchy: 10.3-15.3ms of
      194-204ms walls (0.05-0.08): the floor plus up to ~5ms for
      111,024 pieces: the lazy levels halved this wall (windows at
      this budget are served by the paragraph scan or the sentence
      walk; the word walk is never built, where the eager tree ran all
      three, 347-416ms); margin ~6.5x, held margin ~2.0x.
    - ``chunk_hierarchical`` line-first ``["\n", None]`` at
      ``max_chars=40``: 30.7-41.8ms of 400-409ms walls (0.08-0.10):
      the floor plus ~20-31ms of 2-tuple construction for 370,080
      pieces (~0.05-0.08µs per piece): the family's most-marshalling
      member (a title the 200-budget leg's 92,520 pieces never held)
      and the residue gradient's top step; margin ~2.4x on the gap,
      ~4.1x held.

    The residue gradient across members is the O(pieces) marshalling
    band made visible at a sane piece count: the no-list-class floor
    (~10-11ms) where pieces are absent or few, up to ~9ms over the
    floor at 64-130k pieces, ~20-31ms at 370k; every member far under
    the ceiling, none anywhere near the word_bounds list shape's
    428-497ms at 3.67M."""
    corpus = _CORPORA["chatlog"](12 * _MIB)
    calls = _chunk_family_calls(corpus)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(calls[member]),
            ratio_budget=None,
            samples=6 if member in _FAMILY_MARSHALLING_HEAVY else _SAMPLES,
        )
    )


def para_soup(target_bytes: int) -> str:
    """The paragraph-soup corpus for the paragraph scanner's heaviest drain
    cell below: one single-character paragraph per 3-byte unit ("a\\n\\n"),
    repeated to ``target_bytes`` (UTF-8, like every corpus builder's sizing),
    so 12 MiB holds 4,194,304 paragraphs: word_bounds' 3.67M-segment
    piece-count class, ~113x the chatlog corpus's 37k. Pure ASCII and
    deterministic like every corpus in this module; the break-soup shape (a
    blank-line run every 3 bytes) is the paragraph scanner's density guard
    at its densest."""
    unit = "a\n\n"
    return unit * max(1, target_bytes // len(unit.encode("utf-8")))


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_chunk_by_paragraphs_iter_soup_drain_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The paragraph scanner's heaviest drain, the re-sized answer to the
    family cell's stated limitation: the chatlog corpus's by-paragraphs
    members (list and ``_iter`` alike) wall at ~3.4-4.0ms (under the 10ms
    ping floor), where the family cell concedes they "pin that the scans
    leave the loop at the floor at all, not a detach regression by
    themselves". This cell takes the same member to a corpus where the
    drain's wall clears the floor by ~19x (the diff near-identical cell's
    enlargement precedent: when a wall sits under the floor, grow the
    input until the budgets resolve): ``chunk_by_paragraphs_iter`` at one
    paragraph per chunk over 12 MiB of paragraph soup (4,194,304 pieces,
    the word_bounds piece-count class), fully drained.

    Why the ``_iter`` spelling and not the list twin: at this piece count
    the list shape's O(pieces) GIL-held 2-tuple marshalling is the
    word_bounds list-shape class (~0.1µs per piece, a several-hundred-ms
    hold), structurally over the 100ms ceiling in the green state: the
    exact disclosed cost the streaming twins exist to avoid, so only the
    drain can take the shared budgets.

    Measured on the box this cell was calibrated on (Linux, 32 logical
    cores, ambient load ~4.8, 5 samples): the construction scan over the
    soup ~17ms (a one-chunk call over the same corpus, the density
    guard's per-boundary work at 8.4M break bytes); full drain walls
    182-191ms; worst gaps 15.3-20.3ms (ratio 0.08-0.11), inside both
    shared budgets (~2.7x ratio margin, ~5x ceiling margin). The gap band
    is the per-``__next__`` contention band the word_bounds_iter
    (15.4ms at 3.67M) and find_patterns_iter (15.4ms at 1.28M) cells
    already record, now pinned for the paragraph scanner at the family's
    heaviest piece count: a per-``__next__`` regression that re-pays
    O(remaining) work per handoff (a re-materialized bounds buffer, a
    per-next chunk marshalling) holds ~the whole construction per
    ``__next__`` and blows the ceiling on its first dirty sample.

    Limitations (the sparse cell's discipline): the drain's
    ~185-190ms wall is per-``__next__`` Python bytecode, and the eval loop
    drops the GIL between handoffs, so a construction-only detach
    regression shows only ~17ms gaps (the soup scan, held; under the
    ceiling, not discriminated here), and a per-``__next__`` whole-text
    re-scan holds only scan-sized gaps while exploding the wall into
    CI-timeout territory, which the budgets cannot see. What the budgets
    do catch is the O(remaining)-per-``__next__`` class above. The
    construction scan's ~17ms is the by-paragraphs scanner lane's
    business (a scanner retune moves the wall, not this band: the gap is
    the ping floor plus handoff contention, with ~5x/2.7x margins to
    spare)."""
    corpus = para_soup(size_bytes)

    def consume() -> int:
        return sum(1 for _ in tors.chunk_by_paragraphs_iter(corpus, 1))

    asyncio.run(_assert_loop_stays_responsive(lambda: asyncio.to_thread(consume)))


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_diff_opcodes_lines_near_identical_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The line-diff claim on the near-identical pair: the whole pass
    (both operands' line split and the Myers search over the ~37.8k lines)
    runs under ``py.detach``, and the return is a 5-tuple list measured at
    just 5 opcodes for the pair's six scattered line edits (the line-level
    diff collapses the char-level twin's 235 opcodes to the edit blocks'
    line granularity), so the marshalling is µs-scale.

    Measured on the dev box (ambient load 2.0, 5 samples): worst gaps
    10.5-12.0ms against walls of only 1.9-2.6ms: the wall sits an order
    of magnitude under the 10ms ping floor (the line-level spelling is
    ~30x cheaper in wall than its char-level twin's 76-88ms on the same
    pair, the cell, because the Myers search runs over lines, not
    characters), so the gap/wall ratio (4.2-6.4) is the suite's
    documented sub-ping artifact and the cell asserts the 100ms ceiling
    only (~8x margin; the b64 12 MiB / utf8_is_valid /
    find_patterns-sparse precedent). A limitation, the sparse cell's
    shape: a ~2.5ms diff held or released is invisible under the floor
    either way, so this cell cannot discriminate a detach regression by
    itself; it pins that the line-split + Myers pass leaves the loop at
    the floor at all."""
    a, b = diff_pair_near_identical(size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.diff_opcodes_lines, a, b),
            ratio_budget=None,
        )
    )


# count_matches's ratio budget: not the shared 0.30, which this wall cannot
# attain by construction: the count core is a single detached scan with an
# int return (no marshalling class at all), so the GIL-held residue is the
# 10ms ping floor alone, but the wall (~31ms at 12 MiB dense) clears the
# floor only ~3x, making gap/wall ≈ 0.34 by arithmetic. The QC-Yes 12 MiB
# derivation verbatim (_QC_YES_12MIB_RATIO_BUDGET's comment): 0.60 sits
# ~1.7x above the worst measured ratio (0.353) and ~40% below the ~1.0 a
# detach regression shows (a held scan pins the whole ~31ms wall). The 100ms
# ceiling holds ~9x over the measured gaps, and it is not a
# detach discriminator at this size (a held ~31ms wall passes it easily);
# the ratio budget is this cell's only one.
_COUNT_MATCHES_RATIO_BUDGET = 0.60


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_count_matches_dense_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The count claim: the whole count pass (automaton build, scan,
    and nothing else: no match vector, no byte→char conversion pass, a
    single int return) runs under ``py.detach``, so the GIL-held residue
    is the ping floor alone, the grapheme_count precedent's no-marshalling
    band. The corpus is the find_patterns dense cell's own (the 17-word
    dense set over the plain prose corpus, 1,284,724 matches): the count
    answers the same question the list shape marshalled 171.0-174.7ms of
    GIL hold for, at zero measurable hold.

    Measured on the dev box (ambient load 4.4, 5 samples): worst gaps
    10.4-11.2ms of 31.0-31.8ms walls (ratio 0.336-0.353). The budget
    consequence: the wall clears the ping floor only ~3x,
    so the gap/wall ratio sits at 0.34-0.35 by arithmetic; the shared 0.30
    budget is unattainable for any sub-40ms wall with a floor-plus-nothing
    residue (a budget that flakes is worse than a budget derived from the
    band; see ``_QC_YES_12MIB_RATIO_BUDGET`` for the same recalibration
    shape). The cell takes the bespoke 0.60 ratio budget (a detach
    regression holds the whole ~31ms wall at ratio ~1.0 and fails by far)
    plus the shared 100ms ceiling (~9x margin). The inverse of the sparse
    cell's limitation: a held ~31ms scan passes the ceiling easily,
    so the ratio budget is this cell's only detach discriminator."""
    text = prose(size_bytes)
    patterns = list(SEARCH_DENSE_PATTERNS)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.count_matches, patterns, text),
            ratio_budget=_COUNT_MATCHES_RATIO_BUDGET,
        )
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_find_patterns_iter_dense_drain_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The streaming answer to the find_patterns dense marshalling band
    (171.0-174.7ms of GIL hold for 1.28M 3-tuples, the cell above): the
    full consumption of ``tors.find_patterns_iter(patterns, text)`` over the
    same dense corpus (construction plus every ``__next__``) against the
    suite's shared budgets, which the list-returning shape structurally
    could not meet.

    The design (the word_bounds_iter cell's, verbatim): the whole search
    (automaton build, scan, conversion, buffer fill) runs under one
    ``py.detach`` when the iterator is constructed, and each ``__next__``
    then holds the GIL only to construct one 3-tuple (µs-scale), so the
    worst heartbeat gap collapses back to the ping-floor band.

    Measured on the dev box (ambient load 4.4, 5 samples): worst gaps
    15.4ms of 129.0-132.0ms walls (ratio 0.116-0.119): the word_bounds_iter
    band exactly (15.4ms there: the ping floor plus the two-thread GIL
    contention of the draining thread's per-``__next__`` bytecode), inside
    both shared budgets (~2.5x ratio margin, ~6.5x ceiling margin).

    The wall contrast, measured in the same process (min-of-3, ambient load
    3.1): full drain 117.4ms against the list API's 211.1ms and the count
    core's 28.3ms: the drain is the ~28-31ms construction scan plus
    ~86-90ms of 1.28M per-``__next__`` 3-tuple handoffs (~70ns per match,
    against the list shape's ~0.13µs per match of GIL-held marshalling), so
    the iterator wins both axes at this size (the word_bounds_iter
    finding's shape: 347ms vs 724ms there). A discrimination caveat: the
    construction pass is only ~31ms of the ~130ms wall, so a
    construction-only detach regression shows ~41ms gaps at ratio ~0.32;
    the 0.30 budget catches it, but thinly; the grosser classes (a
    per-``__next__`` re-scan, per-next chunk marshalling) blow through both
    budgets."""
    text = prose(size_bytes)
    patterns = list(SEARCH_DENSE_PATTERNS)

    def consume() -> int:
        return sum(1 for _ in tors.find_patterns_iter(patterns, text))

    asyncio.run(_assert_loop_stays_responsive(lambda: asyncio.to_thread(consume)))


# tors.quote's 32 MiB ratio budget: not the shared 0.30, which this wall
# cannot attain: the output is ~5/4 the input (roughly every sixth prose
# char is a space escaping to %20), so the O(output) marshalling residue
# is a structurally large share of a fast-scan wall, the b64_encode_bytes
# shape (see _B64_RATIO_BUDGET's comment). Measured band (ambient load
# ~5): worst gaps 24.3-28.8ms of 55-60ms walls (ratio 0.44-0.48). 0.60
# sits ~1.25x above the worst measured ratio and ~40% below the ~1.0 a
# detach regression shows (the whole wall held); a several-fold
# marshalling blowout crosses 0.60 or the 100ms ceiling (~3.5x
# above the band). If a crate bump ever speeds the detached scan enough
# to collapse the wall toward the marshalling share, recalibrate per the
# QC-Yes 12 MiB precedent rather than letting the budget flake.
_QUOTE_32MIB_RATIO_BUDGET = 0.60


@pytest.mark.parametrize(
    ("size_bytes", "ratio_budget"),
    [(12 * _MIB, None), (32 * _MIB, _QUOTE_32MIB_RATIO_BUDGET)],
    ids=["12MiB-ceiling-only", "32MiB-both-budgets"],
)
def test_quote_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int, ratio_budget: float | None
) -> None:
    """The percent-encode claim: the whole escape scan runs under
    ``py.detach``, and the GIL-held residue is the argument borrow plus
    the marshalling of one output string (~5/4 the input at this corpus),
    the no-list-shape class, ``replace_many``'s band shape.

    Measured on the dev box (ambient load 0.4-5.2, 3 samples per cell,
    prose corpus):

    - 12 MiB: worst gaps 10.1-10.8ms against walls of 10.1-23.5ms: the
      wall straddles the 10ms ping floor (a fast native pass: ~20ms
      inline), so any gap/wall ratio is the suite's documented sub-ping
      artifact and the cell asserts the 100ms ceiling only (the b64
      12 MiB precedent). A limitation: a held ~20ms encode passes
      the ceiling too, so the ratio discriminator for this function lives
      in the 32 MiB cell and the red-side cells below.
    - 32 MiB: worst gaps 24.3-28.8ms of 55.2-59.7ms walls (ratio
      0.44-0.48): the ping floor plus the ~43 MiB output's marshalling
      band, the b64 structural shape (the output is larger than the
      input, so the marshalling is a larger wall share than finalize's).
      The 0.60 cell budget sits ~1.25x above the worst measured ratio;
      a detach regression (the scan itself GIL-held) shows ratio ~1.0
      and fails by far (see ``_QUOTE_32MIB_RATIO_BUDGET``'s comment).

    The red side, same corpus and placement, is the whole reason the
    function exists: ``urllib.parse.quote`` is pure Python; its
    measured bands and the structural surprise (the ≥200KB chunked path
    gh-95865 added) are pinned by the red-side cells below."""
    corpus = prose(size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.quote, corpus),
            ratio_budget=ratio_budget,
        )
    )


@pytest.mark.parametrize(
    ("size_bytes", "ratio_budget"),
    [(12 * _MIB, None), (32 * _MIB, _RATIO_BUDGET)],
    ids=["12MiB-ceiling-only", "32MiB-both-budgets"],
)
def test_unquote_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int, ratio_budget: float | None
) -> None:
    """The percent-decode claim, on the encoder's own output (the corpus
    is ``tors.quote(prose(...))``, the canonical decoder input, its
    round trip pinned in tests/test_url_quote.py): the whole fragment
    walk + replace-decode runs under ``py.detach``, the residue is the
    marshalling of one decoded string (3/4 the encoded size).

    Measured on the dev box (ambient load 0.4-5.2, 3 samples per cell):

    - 12 MiB decoded (16 MiB encoded): worst gaps 10.8-11.1ms against
      walls of 14.3-23.5ms: the wall straddles the ping floor, the
      sub-ping artifact; the cell asserts the 100ms ceiling only (the
      b64 12 MiB precedent).
    - 32 MiB decoded (43 MiB encoded): worst gaps 15.5-16.5ms of
      77.0-77.8ms walls (ratio 0.20-0.21): the floor plus the ~32 MiB
      output's marshalling. Every sample inside both shared budgets
      (~1.4x ratio margin, ~6x ceiling margin); a detach regression
      holds the whole ~77ms wall at ratio ~1.0 and fails by far.

    The red side: ``urllib.parse.unquote`` over the same encoded corpus
    measured worst gaps 50.5-52.0ms of 217.6-221.5ms walls (ratio
    0.23-0.24): it passes both budgets (recorded in the red-side cells
    below, with the structural reason), so the statement is the
    wall (~12x: ~18ms tors inline vs ~223ms stdlib) and the worst gap
    (~5x), not "blocked vs unblocked" (the html.unescape red-side
    finding's shape)."""
    corpus = tors.quote(prose(size_bytes))
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.unquote, corpus),
            ratio_budget=ratio_budget,
        )
    )


def _close_matches_corpus() -> tuple[str, list[str]]:
    """One query × 13,900 candidates for the bulk get_close_matches
    cells: the prose corpus's own four-sentence unit (664 chars) with
    the diff pair's word swap as the query, and every single-character
    substitution of the unit (lowercase letters at the lowercase
    positions) as the candidate pool: the corpus's own text at a
    length where difflib's quadratic char matcher costs seconds
    (measured ~3.4s) and tors's per-candidate Myers passes are real
    work (~65ms). The short-candidate alternative (every mutation of
    the 23-word vocabulary (~7.9k × ~8 chars)) measures both engines at
    ~7.5ms (difflib is fast on short candidates; the wall race's
    seconds live at sentence length), so the bulk cell's pool is this
    one."""
    sentence = prose(64 * 1024).split("\n\n")[0]
    query = sentence.replace("quarterly", "monthly", 1)
    pool = list(
        dict.fromkeys(
            sentence[:i] + c + sentence[i + 1 :]
            for i in range(len(sentence))
            if sentence[i].isalpha() and sentence[i].islower()
            for c in string.ascii_lowercase
            if c != sentence[i]
        )
    )
    return query, pool


def test_get_close_matches_bulk_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity() -> None:
    """The bulk-similarity claim: one query × 13,900 664-char candidates
    (the corpus builder above): the sweep's per-candidate Myers passes
    run under ``py.detach``, and the GIL-held residue is the O(candidates)
    argument walk plus the ping floor: the returned elements are the
    original candidate objects (references, zero marshalling per hit,
    pinned in tests/test_similarity.py), so there is no list-shape class
    at all.

    Measured on the dev box (ambient load 4.1-5.2, 3 samples): worst
    gaps 10.6-11.3ms of 64.2-71.7ms walls (ratio 0.15-0.18): inside
    both shared budgets (~1.7x ratio margin, ~9x ceiling margin); a
    detach regression (the sweep itself GIL-held) pins the whole ~65ms
    wall at ratio ~1.0 and fails by far. The wall race against difflib
    on the same input (~50x, single sample; the difflib precedent) is
    the cell below."""
    query, pool = _close_matches_corpus()
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.get_close_matches, query, pool, 5)
        )
    )


def _quote_unchunked_core(text: str) -> str:
    """``urllib.parse.quote``'s per-byte core statement: the
    ``''.join(map(quoter, bytes))`` that ``quote_from_bytes`` runs for
    inputs under 200 KB, with the quoter table built here rather than
    reaching into ``urllib.parse``'s private ``_byte_quoter_factory``
    (absent on 3.10, measured): the stdlib's ``_Quoter`` is a dict
    subclass whose ``__getitem__`` this fully-populated table's is. The
    join over a C-callable mapper holds the GIL for its whole duration,
    i.e. the GIL-held whole-text pass the stdlib spelling is below the
    chunking threshold; the value-parity assert in the red-side cell
    proves the table is the stdlib's own."""
    always_safe = frozenset(string.ascii_letters + string.digits + "_.-~" + "/")
    table = {byte: chr(byte) if chr(byte) in always_safe else f"%{byte:02X}" for byte in range(256)}
    return "".join(map(table.__getitem__, text.encode("utf-8")))


def test_the_gil_held_url_red_sides_fail_their_budgets_in_every_sample() -> None:
    """The red side, asserted mechanically: the per-byte join that is
    ``urllib.parse.quote``'s core (the ``<200 KB`` statement; see
    ``_quote_unchunked_core``) holds the GIL for its whole duration, so
    the budgets tors's quote cells above pass must be failed by it in
    every sample, judged by the same ``_budget_misses`` list the green
    cells use. Measured on the dev box (ambient load 4.1-11.8, 3
    samples, 12 MiB prose): worst gaps 179.8-187.8ms of 180.0-188.0ms
    walls: ratio 0.999, the whole-text hold, missing both budgets in
    every sample (the single-C-hold shape the b64 96 MiB red side
    demonstrates for ``b64encode``).

    Measured and not asserted: the two stdlib red
    expressions that pass the budgets at this size, with the structural
    reasons (the b64 12/32 MiB red-side precedent: record the shape,
    assert only what discriminates):

    - ``urllib.parse.quote(text)`` itself, same corpus and placement:
      worst gaps 15.6-22.0ms of 163.9-183.0ms walls (ratio 0.09-0.12;
      ambient load 0.4-32 across the measurement windows).
      The surprise vs the whole-text-hold expectation: inputs ≥200 KB
      take the chunked path gh-95865 added (``chunk_size =
      isqrt(len)``), whose per-chunk joins hold only µs-bursts between
      the chunk comprehension's bytecode; the eval loop drops the GIL
      between chunks, so the worst gap is a burst, not the wall. The
      unchunked core above is what the function runs below the
      threshold and ran at every size before the chunking; the race in
      wall is unchanged (~8x: tors ~20ms inline vs stdlib 158-183ms).
    - ``urllib.parse.unquote(encoded)`` (the 12 MiB percent-encoded
      corpus): worst gaps 50.5-65.0ms of 217.6-267.0ms walls (ratio
      0.23-0.25). Its ``_generate_unquoted_parts`` generator yields
      bytecode per ASCII run (eval-loop GIL drops), and the worst gap
      is set by the per-run C work between those yield points: the
      html.unescape red-side class ("a fifth of the block time, an
      eighth of the wall" there; ~5x/~12x here).

    The live measurements of both recorded legs run in the sister cell
    below, in the same to_thread cell style, with their value parity
    asserted as a 12 MiB differential anchor."""
    corpus = prose(12 * _MIB)
    # The table is the stdlib's: the core statement produces quote()'s
    # own output, and tors agrees with both (the 12 MiB parity anchor).
    assert _quote_unchunked_core(corpus) == tors.quote(corpus) == urllib.parse.quote(corpus)
    observed = [
        asyncio.run(_gap_and_wall_during(lambda: asyncio.to_thread(_quote_unchunked_core, corpus)))
        for _ in range(_SAMPLES)
    ]
    for gap, wall in observed:
        assert _budget_misses(gap, wall, _RATIO_BUDGET), (
            f"the unchunked quote core measured a CLEAN sample ({gap * 1000:.0f}ms of a "
            f"{wall * 1000:.0f}ms wall): the budgets tors's quote cells pass no longer "
            "discriminate against the GIL-held whole-text join they replace, and a "
            "detach regression could pass them too (tests/test_gil_release.py)"
        )


def _measure_red(red: Callable[[str], str], corpus: str) -> list[tuple[float, float]]:
    """Measure a stdlib red expression in the green cells' own cell style
    (``asyncio.to_thread`` under the heartbeat), ``_SAMPLES`` samples."""
    return [
        asyncio.run(_gap_and_wall_during(lambda: asyncio.to_thread(red, corpus)))
        for _ in range(_SAMPLES)
    ]


def test_urllib_quote_and_unquote_red_sides_measured_value_parity_and_recorded() -> None:
    """The stdlib red sides, measured live in the same to_thread cell
    style, recorded, not budget-asserted (both pass the shared budgets
    at this size; the structural reasons are in the asserted red-side
    cell's docstring above). What is asserted is the value parity (the
    12 MiB differential anchor for both directions, the same equality
    the hypothesis gates in tests/test_url_quote.py prove at small
    sizes, here at corpus scale), and the bands are printed so every
    run's log carries the recorded divergence shape."""
    text = prose(12 * _MIB)
    encoded = tors.quote(text)
    # The 12 MiB parity anchors (one inline call each side).
    assert urllib.parse.quote(text) == tors.quote(text)
    assert urllib.parse.unquote(encoded) == tors.unquote(encoded)

    note = "urllib red side (recorded, not asserted, passes the budgets; see the red-side cell): "
    for name, red, corpus in (
        ("urllib.parse.quote", urllib.parse.quote, text),
        ("urllib.parse.unquote", urllib.parse.unquote, encoded),
    ):
        observed = _measure_red(red, corpus)
        for gap, wall in observed:
            print(
                f"{note}{name} blocked {gap * 1000:.0f}ms of a {wall * 1000:.0f}ms "
                f"operation ({gap / wall:.0%})"
            )


# The bulk wall race's tolerance margin: tors's measured wall (~65ms)
# against difflib's (~3.4s) is a ratio of ~0.02, so 0.25 leaves >10x
# headroom; the _DIFF_WALL_MARGIN precedent (the assertion pins the
# quadratic-vs-native relationship, not a close race).
_GCM_WALL_MARGIN = 0.25


def test_get_close_matches_beats_difflib_on_the_bulk_corpus() -> None:
    """The wall race at the bulk shape: difflib's per-candidate
    ``SequenceMatcher`` is quadratic in the common region, and 664-char
    candidates with a one-word-different query put nearly the whole
    candidate in that region: ~0.24ms per candidate, ~3.4s for the
    13,900-candidate sweep (measured once, single sample: noise only
    ever adds time, so one sample is conservative for the denominator;
    the difflib precedent test_diff_opcodes.py set at its 256 KiB
    race). tors runs the same sweep's Myers passes natively: measured
    64.2-71.7ms walls (ambient load 4.1-5.2), ratio ~0.02 against the
    0.25 margin.

    difflib's own GIL band, measured in the same to_thread placement:
    worst gaps 15.5-20.5ms of 3,346-3,931ms walls (~0.5%): the sweep
    is pure-Python bytecode, so the eval loop drops the GIL at switch
    intervals and the loop stays schedulable; the wall is the finding
    (~50x), not the blocking; recorded here so the race is not
    misread as a GIL claim. tors's GIL band is the green bulk cell
    above."""
    query, pool = _close_matches_corpus()
    tors_walls = [
        asyncio.run(
            _gap_and_wall_during(lambda: asyncio.to_thread(tors.get_close_matches, query, pool, 5))
        )
        for _ in range(_SAMPLES)
    ]
    tors_wall = min(wall for _, wall in tors_walls)
    difflib_gap, difflib_wall = asyncio.run(
        _gap_and_wall_during(lambda: asyncio.to_thread(difflib.get_close_matches, query, pool, 5))
    )
    print(
        f"difflib.get_close_matches red side (recorded): blocked {difflib_gap * 1000:.0f}ms "
        f"of a {difflib_wall * 1000:.0f}ms wall ({difflib_gap / difflib_wall:.1%}): "
        "pure-Python eval-loop yields, the html.unescape red-side class; the WALL is the finding"
    )
    assert tors_wall < _GCM_WALL_MARGIN * difflib_wall, (
        f"bulk get_close_matches: tors {tors_wall * 1000:.0f}ms vs difflib "
        f"{difflib_wall * 1000:.0f}ms (ratio {tors_wall / difflib_wall:.4f}): the native "
        "sweep lost more than the tolerance margin to the quadratic stdlib matcher"
    )


@pytest.mark.parametrize(
    "length",
    [1024 * 1024, 2 * 1024 * 1024],
    ids=["1MiB-chars", "2MiB-chars"],
)
def test_random_hex_generation_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    length: int,
) -> None:
    """The random-generation claim, ceiling-only cells (the b64 12 MiB
    precedent): the whole pass — the block-buffered OS-entropy fills (one
    getrandom syscall per 1024 bytes, no process or thread RNG state) plus
    the per-character sampling and string build — runs under one
    ``py.detach``, and the return is one string, so the GIL-held residue is
    the O(output) marshalling alone (1 byte per ASCII hex character here).
    The length-first refactor moved hex onto the char-sampling engine, so
    the wall at a given output length is the b62-class wall (measured
    ~31ns/char), not the old byte-fill wall: these cells hold the output
    sizes the old byte-drawn cells produced (1 MiB and 2 MiB of hex
    string), which now costs 8 MiB / 16 MiB of stream drawn through the
    block buffer — thousands of syscalls, all detached.

    Measured on the dev box (Apple Silicon, quiet, 3 samples per cell):

    - 1 MiB of output (2^20 chars): worst gaps ~11.1ms of ~33.5-33.8ms
      walls — the ping floor; the whole generation+marshalling sequence
      stays off the loop.
    - 2 MiB of output (2^21 chars): worst gaps ~11.1ms of ~66.6-67.1ms
      walls, the same band.

    The wall grew ~10x over the old byte-fill spelling at equal output
    (the engine-class change the performance ledger records); the GIL
    claim is unchanged — the gap is the floor, and the cell's ceiling
    (100ms) still has ~9x margin over it. A detach regression (the fills
    held under the GIL) would show the same wall but block the loop for
    it: 33-67ms held is over the ping floor but under the ceiling, so the
    ceiling alone cannot separate that — the cell's real regression teeth
    are the blowout class (a per-char syscall regression would put ~1.2us
    x 2^21 ~ 2.5s of held work behind one call and trip the ceiling by
    ~25x).

    The seeded spelling (ChaCha20 userspace, no syscall at all) is strictly
    cheaper on the detached side; unseeded is the shape worth the cell."""
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.random_hex, length),
            ratio_budget=None,
        )
    )


@pytest.mark.parametrize("corpus_kind", ["prose", "decomposed"])
@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_minhash_signature_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    corpus_kind: str, size_bytes: int
) -> None:
    """The MinHash claim: the whole tokenize + shingle + XXH64 + min-sweep
    (the sweep is the dominant cost, O(shingles x num_perm)) runs under one
    ``py.detach``, and the return marshalling is bounded by contract at
    ``num_perm`` ints (<= 1024), so the GIL-held residue is the argument
    borrow plus at most a thousand fresh ints -- two orders of magnitude
    under the word_bounds 3.67M-tuple band at the same corpus size, the
    structural reason no streaming twin exists for this shape.

    Measured on the dev box (macOS/arm64, ambient load ~2, 3 samples per
    cell, ``num_perm`` 128):

    - prose 12 MiB: worst gap 11.1ms of 456-466ms walls (ratio 0.024):
      the ping floor plus the borrow and the 128-int marshalling.
    - decomposed 12 MiB: the same band plus the str-in one-time O(input)
      UTF-8 materialization on the first sample (the class every str-in
      function pays), still at the floor scale.

    Every sample sits deep inside both shared budgets (~12x on the ratio,
    ~9x on the ceiling)."""
    corpus = _CORPORA[corpus_kind](size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(lambda: asyncio.to_thread(tors.minhash_signature, corpus))
    )


@pytest.mark.parametrize("size_bytes", [12 * _MIB], ids=["12MiB"])
def test_content_hash_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    size_bytes: int,
) -> None:
    """The object-walk residue class, pinned: ``tors.content_hash``'s walk
    materializes the whole value tree under the GIL (the standard arg-walk
    class scaled to an object: one ``to_str`` borrow plus copy per str,
    one i64 storage read per fast-path int, one Python ``repr`` call per
    float or big int, one CPython sort per non-str/int-keyed dict), then
    the canonical-form emission and the SHA-256 run under one
    ``py.detach``. The residue is therefore structurally ~half the call,
    not a small marshalling tail: measured on the dev box over two load
    windows (ambient load ~2 and ~10-16, 3-5 samples, the records corpus
    (``reference.content_object``) at 12 MiB, ~30.7k records of str/int/
    float/bool/list fields): worst gaps 23.3-32.2ms of 41.9-60.4ms walls,
    ratios 0.45-0.60, every sample inside both budgets (the 0.80 ratio
    budget ~1.3x above the worst observed ratio, the 100ms ceiling ~3x
    above the worst gap). The corpus is all-ASCII, so no first-call
    UTF-8-materialization band exists: every sample pays the same walk
    (compact-ASCII ``to_str`` borrows are zero-copy aliases; the copy into
    the owned tree is the cost).

    What the detach buys, and what it cannot: the stdlib spelling
    (``sha256(json.dumps(...).encode()).hexdigest()``) holds the GIL for
    ``json.dumps`` plus ``str.encode``, ~the whole wall -- inline it
    measures ratio 1.00-1.02 in every sample (the red row), and even in
    this to_thread placement 0.89 -- where tors's worst is 0.60 under the
    same load. The walk itself is irreducible without an interpreter-free
    object format: every step is a CPython API call, so the O(tree) walk
    is the documented price of the parity contract, and the emitted-bytes
    half of the call is what the detach removes.

    No 32 MiB twin: the wall cell (tests/test_performance.py) records the
    size curve as a measured dead heat with the stdlib at 64 KiB-12 MiB,
    and the residue/wall ratio is the load-stable constant this cell
    pins, not a size-dependent quantity."""
    obj = content_object(size_bytes)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(tors.content_hash, obj),
            ratio_budget=_CONTENT_HASH_RATIO_BUDGET,
        )
    )


# --- The one-shot hashing surface -------------------------------------------------
#
# sha256_hex/sha512_hex/sha256_digest plus the HMAC spellings at 12 MiB:
# ceiling-only cells (walls at or under the ping floor on the calibration
# hardware), plus the measured-not-asserted hashlib red-side recording
# cell (the urllib red-side precedent: the stdlib releases the GIL for
# 2048+-byte digest updates, so there is no GIL-blocked red row to
# assert; see the module docstring's hashing paragraph for the full
# measured story, including the 96 MiB inline lost-detach discrimination
# measurements). The digest and HMAC rows pin src/lib.rs's hashing
# paragraph directly: the `_digest` spellings marshal a fixed-size
# PyBytes instead of the hex string, and the HMAC spellings borrow two
# arguments, all under the same single detach — measured here, not just
# claimed for the hex spellings.


@pytest.mark.parametrize(
    "fn_name",
    ["sha256_hex", "sha512_hex", "sha256_digest"],
    ids=["sha256", "sha512", "sha256-digest"],
)
def test_hash_digest_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    fn_name: str,
) -> None:
    """The hashing surface's GIL claim, at the size the wall-vs-hashlib
    comparison is told (tests/test_performance.py): one 12 MiB digest in a
    worker thread, the whole computation (hex formatting included on the
    `_hex` spellings) under one ``py.detach``, and the loop ticks at the
    ping floor through it — measured 10.1-10.7ms worst gaps (the floor
    plus the argument borrow and the O(64..128) hex marshalling, or the
    fixed-size PyBytes on the `_digest` spelling; the crate GIL model's
    no-residue-class claim for this surface, measured directly on hex,
    digest, and HMAC rows alike).

    Ceiling-only (``ratio_budget=None``, the b64 12 MiB / find_patterns
    sparse precedent): the 12 MiB digest walls on the calibration hardware
    are 4-5ms (sha256) and 7-8ms (sha512), at or under the 10ms ping
    floor, so any gap/wall ratio is the documented sub-ping artifact. The
    limitation, stated rather than thresholded away (the chunking
    family's light-member note): a lost detach at this size holds a
    sub-floor wall on this hardware and no budget here discriminates it;
    the 96 MiB measurements in the module docstring (to_thread 0.28 vs
    inline 1.00) are what the regression looks like where the wall clears
    the floor, and the 100ms ceiling is the pin that catches it on
    hardware slow enough for a held 12 MiB digest to reach it."""
    corpus = corpus_utf8("prose", 12 * _MIB)
    fn = getattr(tors, fn_name)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(fn, corpus),
            ratio_budget=None,
        )
    )


@pytest.mark.parametrize(
    "fn_name", ["hmac_sha256_hex", "hmac_sha256_digest"], ids=["hmac-hex", "hmac-digest"]
)
def test_hmac_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    fn_name: str,
) -> None:
    """The HMAC spellings' GIL claim: a short key over 12 MiB of data in a
    worker thread, the whole keyed digest (key derivation included, hex
    formatting included on the hex spelling) under one ``py.detach`` —
    both borrows under the GIL, nothing else held. Ceiling-only like the
    digest cells above (the 12 MiB HMAC wall sits at the same
    engine-dominated scale, at or under the ping floor); the 100ms
    ceiling is the pin. The short key is the request-signing shape (a
    webhook secret, bytes, not blocks)."""
    corpus = corpus_utf8("prose", 12 * _MIB)
    fn = getattr(tors, fn_name)
    key = b"corpus-key"
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(fn, key, corpus),
            ratio_budget=None,
        )
    )


def test_hashlib_red_side_is_measured_and_value_parity_is_asserted() -> None:
    """The stdlib red side for the hashing surface, measured live in the
    same to_thread cell style and recorded, not budget-asserted (the
    urllib red-side precedent): CPython's ``hashlib`` releases the GIL for
    digest updates of 2048+ bytes (the ``_hashopenssl`` threshold), so at
    digest-scale sizes the stdlib keeps the loop at the ping floor too and
    asserting a blocked red row would be manufacturing a win the
    measurement does not show. Below the threshold hashlib does hold the
    GIL, but a sub-threshold digest is ~11µs of held GIL (measured),
    invisible under the 10ms floor either way — the honest statement is
    that tors's GIL release is uniform at every size while hashlib's
    starts at 2048 bytes, and that this surface's GIL value over the
    stdlib is the µs-scale uniformity plus the never-held hex tail, not a
    latency win at digest sizes. What IS asserted is the value parity
    (the 12 MiB differential anchor, the same equality the hypothesis
    gates in tests/test_hash.py prove at generated sizes, here at corpus
    scale); the bands are printed so every run's log carries the recorded
    shape."""
    import hashlib

    corpus = corpus_utf8("prose", 12 * _MIB)
    # The 12 MiB parity anchors (one inline call each side).
    assert tors.md5_hex(corpus) == hashlib.md5(corpus).hexdigest()
    assert tors.sha1_hex(corpus) == hashlib.sha1(corpus).hexdigest()
    assert tors.sha256_hex(corpus) == hashlib.sha256(corpus).hexdigest()
    assert tors.sha512_hex(corpus) == hashlib.sha512(corpus).hexdigest()

    note = "hashlib red side (recorded, not asserted; GIL released for 2048+-byte updates): "
    for name, red in (
        ("hashlib.sha256", lambda raw: hashlib.sha256(raw).hexdigest()),
        ("hashlib.sha512", lambda raw: hashlib.sha512(raw).hexdigest()),
    ):
        observed = [
            asyncio.run(
                _gap_and_wall_during(lambda red=red: asyncio.to_thread(red, corpus))
            )
            for _ in range(_SAMPLES)
        ]
        for gap, wall in observed:
            print(
                f"{note}{name} blocked {gap * 1000:.0f}ms of a {wall * 1000:.0f}ms "
                f"operation ({gap / wall:.0%})"
            )


# The UUIDv7 helper trio's fixed v7 (timestamp field 1_750_000_000_000 ms,
# version 7, RFC 4122 variant; the same fixed UUID docs/api.md's example and
# tests/test_uuid.py's doc-example pin use), in both accepted spellings.
# Self-contained literals (no uuid import): the timing lane builds its own
# corpora, the chatlog precedent.
_UUID7_BATCH_BYTES = bytes.fromhex("01977420dc007abc9def98765432100f")
_UUID7_BATCH_TEXT = "01977420-dc00-7abc-9def-98765432100f"

# The trio's batch size and bespoke gap ceiling. One call is ~60-90ns (16
# bytes in, one int or one 16-byte value out; measured 59-76ns inline,
# 63-85ns per loop iteration with the generator bookkeeping), so a single
# call sits four orders of magnitude under the 10ms ping floor and no
# single-call cell can mean anything; the batch loop is the only honest
# shape. The structural gap band of that shape is NOT the quiet-box
# 10-11ms ping floor: a worker thread reacquiring the GIL every ~70ns
# contends with the heartbeat for it, and under sustained ambient load
# that handoff contention stretches worst gaps into the tens-to-low-
# hundreds of ms -- measured 27-75ms at ambient load 8.6-12.3 (N=4M) and
# 27-118ms at 17-19 (N=8M), a band that is roughly wall-independent
# (per-tick reacquisition delay) while a wholesale GIL hold of the loop
# shows gap ~= wall. N is therefore sized by separation, the
# word_bounds-precedent derivation: at 8M calls the walls (~0.9-1.2s
# measured) put the wholesale-hold regression class at ~0.9-1.2s of gap,
# far above the load band, and the 400ms bespoke ceiling sits ~3.4x above
# the worst measured loaded-band gap (118ms) and ~2.2-3x below the hold
# class. The suite's shared 100ms ceiling is structurally at risk for
# this shape under sustained load (the first full-gate run flaked on it:
# all 3 samples dirty at ambient load 8-12, the int-out pair's detach
# churn contending hardest), so this cell carries its own ceiling, the
# word_bounds/list-shape situation; the shared 0.30 ratio budget stays
# (2.7x above the worst measured loaded-band ratio, 0.11; the hold class
# sits at ~1.0 and fails it by >3x). The one regression this cell
# honestly CANNOT catch is a lost py.detach on the int-out pair: the
# extraction itself is a handful of nanoseconds, invisible next to the
# call machinery's own GIL traffic -- the detach on this surface is
# contract uniformity with the rest of the crate, not a measurable
# GIL-release payoff (src/py/uuid.rs's doc comment records the same
# reasoning from the implementation side).
_UUID_BATCH_CALLS = 8_000_000
_UUID_CELL_CEILING_S = 0.400


@pytest.mark.parametrize(
    "helper",
    ["uuid7_timestamp_ms", "uuid_version", "uuid_parse"],
)
def test_uuid_helpers_batch_loop_keeps_the_event_loop_at_heartbeat_granularity(
    helper: str,
) -> None:
    """The UUIDv7 helper trio's GIL claim, at its honest scale: every call's
    GIL-held residue is sub-µs (the int-out pair's argument borrow plus one
    int out, with the bit extraction detached; uuid_parse's whole 36-byte
    parse, which runs GIL-held by design -- there is no int-out tail to
    detach and a detach around a 36-byte scan would be overhead for its own
    sake), so a batch loop of the calls in a worker thread leaves the loop
    ticking at heartbeat granularity over a ~0.9-1.2s wall, even on a
    heavily loaded box.

    The budgets, derived per the constant block above: the 400ms bespoke
    gap ceiling (~3.4x above the measured loaded-band worst of 118ms at
    ambient load 17-19) and the shared 0.30 ratio budget (2.7x above the
    loaded-band worst ratio of 0.11). What they discriminate: a wholesale
    hold of the loop (gap ~= wall ~= 0.9-1.2s, ratio ~1.0: both budgets,
    by >2x) and a per-call GIL-held residue grown to the ~400ms class. A
    lost detach on the extraction is invisible either way (nanoseconds of
    work), the limitation recorded beside the constant; pass-on-first-clean
    over 3 samples tolerates the transient single-sample starvation the
    module's design already retries."""
    calls: dict[str, Callable[[], object]] = {
        "uuid7_timestamp_ms": lambda: tors.uuid7_timestamp_ms(_UUID7_BATCH_BYTES),
        "uuid_version": lambda: tors.uuid_version(_UUID7_BATCH_BYTES),
        "uuid_parse": lambda: tors.uuid_parse(_UUID7_BATCH_TEXT),
    }
    call = calls[helper]

    def consume() -> int:
        return sum(1 for _ in range(_UUID_BATCH_CALLS) if call() is not None)

    observed = [
        asyncio.run(_gap_and_wall_during(lambda: asyncio.to_thread(consume)))
        for _ in range(_SAMPLES)
    ]
    for gap, wall in observed:
        if gap < _UUID_CELL_CEILING_S and gap < _RATIO_BUDGET * wall:
            return
    detail = "; ".join(
        f"blocked {gap * 1000:.0f}ms of a {wall * 1000:.0f}ms batch "
        f"({gap / wall:.0%}, over the {_UUID_CELL_CEILING_S * 1000:.0f}ms ceiling "
        f"and/or the {_RATIO_BUDGET:.0%} ratio budget)"
        for gap, wall in observed
    )
    raise AssertionError(
        f"the uuid helper batch loop regressed in every one of {_SAMPLES} samples "
        f"({detail}): either a per-call GIL-held residue grew into the hundreds "
        "of ms or the loop lost its responsiveness class (src/py/uuid.rs, "
        "tests/test_gil_release.py)"
    )


# The identifier rule's two halves (the letters-and-underscore-at-0,
# digits-join-after shape): the rule this module's
# cell, the wall race in tests/test_performance.py, and the bench group in
# benches/search.rs all drive.
_IDENT_FIRST = string.ascii_letters + "_"
_IDENT_REST = string.ascii_letters + string.digits + "_"


def _ident_items(count: int) -> list[str]:
    """A deterministic all-valid identifier batch (the job/queue/worker/tag
    spellings an enqueue path validates), local to this module (the
    chatlog/_close_matches_corpus precedent: only this module's cells
    consume it)."""
    shapes = ("job_{n}", "queue_eu_{n}", "worker_{n}", "tag_{n}")
    return [shapes[n % 4].format(n=n) for n in range(count)]


@pytest.mark.parametrize("count", [100_000, 1_000_000], ids=["100k-items", "1M-items"])
def test_first_invalid_charset_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    count: int,
) -> None:
    """The batch-validator claim: the whole batch pass (set builds + scan)
    runs under one ``py.detach``, and the call's GIL-held residue is the
    O(items) argument walk (the standard str-in borrow class, the
    ``get_close_matches`` candidate-walk shape over a Sequence) plus a
    single int return: no marshalling class at all.

    Ceiling-only by the ``utf8_is_valid`` precedent, honestly so. The
    function exists for batches of hundreds of items, where the whole
    call measures ~2 µs (the wall race in tests/test_performance.py):
    far under the 10 ms ping floor, no realistic batch can produce a
    measurable gap at all. These cells pin the detach claim at batch
    sizes a thousandfold and ten-thousandfold past realistic, and there
    the measured worst gaps stay in the ping-floor band — the walk's
    contiguous GIL hold never exceeds one ping period (even the 1M-item
    walk's hold sits under the floor, so the worst gap is the floor
    itself, not the walk) — so the 100 ms ceiling (~9x margin) is the
    assertion and any ratio is the suite's documented sub-floor
    artifact. A detach regression at these sub-ceiling walls would hold
    the loop for the whole ~2-20 ms wall and still pass the ceiling:
    the same can't-discriminate-a-held-sub-ceiling-wall limitation the
    ``utf8_is_valid`` cells state for their own sub-floor walls,
    recorded here rather than thresholded away; the wall race and the
    bench carry the performance contract instead.

    Measured on the dev box (ambient load 5.5-8.7, 3 samples per cell):
    100k items worst gaps 11.0-11.1 ms of 1.7-3.0 ms walls; 1M items
    10.5-11.4 ms of 16.4-19.7 ms walls (the floor band: the scan is
    detached, the walk's hold is under the ping period)."""
    items = _ident_items(count)
    asyncio.run(
        _assert_loop_stays_responsive(
            lambda: asyncio.to_thread(
                tors.first_invalid_charset, items, first=_IDENT_FIRST, rest=_IDENT_REST
            ),
            ratio_budget=None,
        )
    )
