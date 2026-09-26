# Missing `**Async**:` lines for the tors.aio twins (audit fragments)

Target file: `docs/api.md`. All 22 twins were import-verified against the
built extension module (proof block at the bottom). Every line below
follows the section-final placement convention (`tors.truncate_ellipsis`,
`tors.scrub_pii`) and the existing voice: the standard opening clause plus
the honest hop-cost sentence. No sync section is missing; every line
belongs under an existing `##` heading. Insert each line at the END of the
named section unless noted.

Shared sections get ONE combined line (the convention
`tors.sentence_bounds / tors.sentence_bounds_iter` uses); both/three names
of a shared section are covered by that one line.

---

## 1. `## tors.repair_json`

**Async**: `await tors.aio.repair_json(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)). The whole repair pass is one detached native pass whose documented measurements reach seconds and minutes on large inputs, so the tens-of-microsecond hop is noise on MB-scale documents; a KiB-scale LLM snippet repairs in microseconds, prefer the sync spelling there.

Evidence: `src/py/json_repair.rs` runs the fence pre-pass, strict probe, repair parser, and serializer under one `py.detach`; `tests/test_gil_release.py::test_repair_json_in_a_thread_...` measures a 12 MiB damaged payload at a 122 ms wall (ratio 0.08), and `docs/async.md` states the "seconds and minutes on large inputs" band.

## 2. `## tors.repair_json_loads`

**Async**: `await tors.aio.repair_json_loads(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)). It is `tors.repair_json`'s exact pass with the decoded object marshalled out, so the same story holds: the hop is noise next to a multi-MB repair, real overhead next to a microsecond-scale KiB snippet; prefer the sync spelling on snippets, `tors.aio` on documents.

Evidence: `src/py/json_repair.rs` (the same `py.detach(|| json_repair::repair(...))` pass as `repair_json`, difference is return marshalling only); the 12 MiB GIL cell above covers the pass both spellings share.

## 3. `## tors.repair_json_diagnostics`

**Async**: `await tors.aio.repair_json_diagnostics(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)). Same detached pass as `tors.repair_json`, with an O(actions) diagnostics list added to the residue, so the guidance is unchanged: prefer the sync spelling on KiB-scale snippets, `tors.aio` on MB-scale documents.

Evidence: `src/py/json_repair.rs` detaches the identical repair pass and only appends the diagnostics-list marshalling after it; the schema-layer action log is bounded by the repair actions actually taken, not by input size.

## 4. `## tors.truncate_to_bounds`

**Async**: `await tors.aio.truncate_to_bounds(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

Evidence: sibling `tors.truncate_ellipsis` (same `detached_transform` GIL shape, `src/py/truncate.rs`) carries exactly this plain line at `docs/api.md:2531`; no test or doc states a size threshold for the truncation pair, so no cost sentence is invented.

## 5. `## tors.is_grounded`

**Async**: `await tors.aio.is_grounded(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)). The default exact path is a memchr-class containment that finishes in microseconds even on MiB passages, pure overhead to hop; `fuzzy=True`'s windowed diff scan is linear in the source and is the shape a long retrieved passage makes millisecond-scale, where the hop is noise.

Evidence: `src/py/grounded.rs` documents `fuzzy=False` as one `py.detach`'d `memmem` check and `fuzzy=True` as the whole windowed scan under one `py.detach`; `docs/api.md` (lines 2549-2551, 2587-2593) puts the exact path at memchr speed and the fuzzy windowing at linear-in-source cost with an O(claim)-sized buffer.

## 6. `## tors.highlight`

**Async**: `await tors.aio.highlight(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)). The capped grounding pass measures ~1 ms on a 2k-token chunk (docs/async.md's own thread-hop-territory threshold) and ~5 ms at document scale, so the hop is noise at chunk scale and up; a sentence-length text is a microsecond pass, prefer the sync spelling there.

Evidence: `src/py/grounding.rs` runs the whole pass under one `py.detach`; `docs/async.md` lines 25-27 state "a 2k-token chunk already measures ~1 ms", and `docs/performance.md` lines 163-165 measure one `highlight` at ~5.2-5.5 ms.

## 7. `## tors.ground_sentences`

**Async**: `await tors.aio.ground_sentences(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)). The batch runs the grounding pass over every sentence of the text (measured ~6 ms on a 1,251-sentence document, ~400 ms walls at 3 MiB of chatlog), so the hop is noise at any batch scale; a short paragraph is microsecond-scale, prefer the sync spelling there.

Evidence: `src/py/grounding.rs` detaches the whole segment/tokenize/score pass; `tests/test_gil_release.py::test_ground_sentences_in_a_thread_...` measures ~40-50 ms gaps of ~400 ms walls at 3 MiB (~55k sentences), and `docs/performance.md` measures ~5.7-6.3 ms on the 1,251-sentence document.

## 8. `## tors.grounding_coverage`

**Async**: `await tors.aio.grounding_coverage(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)). The weighted-LCS DP is O(|S|·|T|) up to the 16384-token operand caps, milliseconds at page scale and seconds at the caps, so the hop is noise on document-scale operands; a sentence-or-paragraph pair is a microsecond pass, prefer the sync spelling there.

Evidence: `src/py/grounded.rs` detaches the whole pass with a single-float residue; `tests/test_gil_release.py::test_grounding_coverage_in_a_thread_...` states the wall at 12 MiB x 12 MiB operands is seconds (caps bounding it at ~10^8 cells) and measures ~10-50 ms walls at 2 MiB + 1 MiB.

## 9. `## tors.similarity_ratio` / `tors.get_close_matches` (one line covers both names)

**Async**: `await tors.aio.similarity_ratio(...)` and `await tors.aio.get_close_matches(...)` run under `asyncio.to_thread` (see [Async use](async.md)). A two-short-string ratio is a microsecond call the hop outruns; the bulk sweep is where the twin pays, measured ~65 ms walls over 13,900 x 664-char candidates with the whole per-candidate Myers sweep detached.

Evidence: `src/py/fuzzy.rs` detaches the Myers scoring (one pair for `similarity_ratio`, the sweep for `get_close_matches`); `tests/test_gil_release.py::test_get_close_matches_bulk_...` measures worst gaps 10.6-11.3 ms of 64.2-71.7 ms walls (ratio 0.15-0.18) on that corpus.

## 10. `## tors.levenshtein` / `tors.jaro` / `tors.jaro_winkler` (one line covers all three names)

**Async**: `await tors.aio.levenshtein(...)`, `await tors.aio.jaro(...)`, and `await tors.aio.jaro_winkler(...)` run under `asyncio.to_thread` (see [Async use](async.md)). Each is a quadratic DP over the operand lengths: the hop costs more than the call on the word/name-scale pairs these metrics usually see, and is noise on the multi-KB and adversarial multi-MB operands the two-row DP and `deadline_ms` exist for.

Evidence: `src/py/fuzzy.rs` detaches the DP/matching pass with a single int/float out (no marshalling class), and `docs/api.md` (lines 2985-2991, 3004-3008) documents the O(n·m) worst case, the linear-space two-row DP, and the adversarial multi-MB sizing; no test pins a KB crossover, so the line stays qualitative.

## 11. `## tors.lsh_candidates`

**Async**: `await tors.aio.lsh_candidates(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)). Honest caveat, measured: the O(n · num_perm) signature walk is interpreter-side int extraction and stays GIL-held inside the worker thread (ratios 0.47-0.61 at 4k x 128 signatures), so the hop buys the detached banding pass and the caller's concurrency shape, not a fully detached call.

Evidence: `src/py/lsh.rs` runs the signature walk under the GIL and only the banding pass under one `py.detach`; `tests/test_gil_release.py::test_lsh_candidates_in_a_thread_...` measures worst gaps 32-53 ms of 68-86 ms walls (ratios 0.47-0.61) against the bespoke `_LSH_RATIO_BUDGET = 0.75`. NOTE: the section's trailing "GIL:" paragraph ends with "`aio` twin: `tors.aio.lsh_candidates`."; fold that sentence into this line (or delete it) so the twin is stated once.

## 12. `## tors.shingle_jaccard` / `tors.shingle_dice` (one line covers both names)

**Async**: `await tors.aio.shingle_jaccard(...)` and `await tors.aio.shingle_dice(...)` run under `asyncio.to_thread` (see [Async use](async.md)). The whole tokenize + shingle + set pass is one detached native pass measuring ~9 ms on 100 KiB pairs, so the hop is noise there and up; a short-string pair is a microsecond call, prefer the sync spelling there.

Evidence: `src/py/near_dup.rs` detaches the whole pass with a single float out; `docs/api.md` lines 5117-5122 measure ~9 ms native on 100 KiB pairs (vs ~15 ms pure Python). NOTE: the section's trailing "GIL:" paragraph ends with "`aio` twins: `tors.aio.shingle_jaccard` / `tors.aio.shingle_dice`."; fold or delete, as above.

## 13. `## tors.dedup_near_dup`

**Async**: `await tors.aio.dedup_near_dup(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)). The fingerprint pass and the documented O(n²) pairwise sweep run detached as one pass (12k documents measure ~0.2-0.4 s walls), so the hop is noise on any corpus-shaped list; a handful of short strings is microsecond-scale, prefer the sync spelling there.

Evidence: `src/py/near_dup.rs` detaches the fingerprint pass AND the pairwise sweep together; `tests/test_gil_release.py::test_dedup_near_dup_in_a_thread_...` measures worst gap 11-14 ms of 230-420 ms walls (ratio 0.03-0.06) at 12k documents with the full pair ladder walked. NOTE: the trailing "GIL:" paragraph ends with "`aio` twin: `tors.aio.dedup_near_dup`."; fold or delete.

## 14. `## tors.rank_fuse` / `tors.ndcg_at_k` / `tors.mrr` / `tors.recall_at_k` / `tors.precision_at_k` (one line covers all five names; REPLACES the existing collective line)

**Async**: each of the five has an `await tors.aio.<name>(...)` twin under `asyncio.to_thread` (see [Async use](async.md)). Honest caveat, measured: the fusion dedup walk and the metrics' membership walks are interpreter-side hashing and stay GIL-held inside the worker thread (ratios 0.56-0.62 at 200k entries for `rank_fuse`, 0.85-0.91 at 100k ids for `ndcg_at_k`), so the hop buys the detached arithmetic and the caller's concurrency shape; past ~10^6 total entries the walk alone holds the GIL for 100ms+ and no placement buys it back.

Evidence: `src/py/rank_fusion.rs` holds the walks under the GIL and detaches only the sweep/sort/metric arithmetic; `tests/test_gil_release.py::test_rank_fuse_in_a_thread_...` (ratios 0.56-0.62 at 200k, bespoke 0.80 budget) and `::test_ndcg_at_k_on_a_large_ranking_...` (0.85-0.91 at 100k, ceiling-only) carry the bands, and the section's own GIL-model paragraph (lines 5644-5659) states the ~10^6-entry guidance and id-shape caveat.

This line REPLACES docs/api.md lines 5553-5554 (the audit missed the section because the existing collective line names no twin explicitly; it also lacks the cost guidance every other wrapped family's line carries). It is the only entry here that is a replacement rather than an insertion.

---

## Verification: all 22 twins import (built extension, this checkout)

```
$ PYTHONPATH=/home/rich/src/tors/python /home/rich/src/tors/.venv/bin/python -c "import tors.aio; print(tors.aio.NAME)"
dedup_near_dup -> <function tors.aio.dedup_near_dup at 0x7f13a49011c0>
get_close_matches -> <function tors.aio.get_close_matches at 0x7f6b48c01620>
ground_sentences -> <function tors.aio.ground_sentences at 0x7f5f187456c0>
grounding_coverage -> <function tors.aio.grounding_coverage at 0x7f197dc4d760>
highlight -> <function tors.aio.highlight at 0x7fd8b4001800>
is_grounded -> <function tors.aio.is_grounded at 0x7ff4da7258a0>
jaro -> <function tors.aio.jaro at 0x7f2ad9445940>
jaro_winkler -> <function tors.aio.jaro_winkler at 0x7f29376319e0>
levenshtein -> <function tors.aio.levenshtein at 0x7f42ea5fda80>
lsh_candidates -> <function tors.aio.lsh_candidates at 0x7f4eb6221b20>
mrr -> <function tors.aio.mrr at 0x7feefa049c60>
ndcg_at_k -> <function tors.aio.ndcg_at_k at 0x7fbb16e49d00>
precision_at_k -> <function tors.aio.precision_at_k at 0x7f7822849e40>
rank_fuse -> <function tors.aio.rank_fuse at 0x7fb77a701ee0>
recall_at_k -> <function tors.aio.recall_at_k at 0x7fb2e0745f80>
repair_json -> <function tors.aio.repair_json at 0x7f00efa26020>
repair_json_diagnostics -> <function tors.aio.repair_json_diagnostics at 0x7f46556460c0>
repair_json_loads -> <function tors.aio.repair_json_loads at 0x7f5bc6146160>
shingle_dice -> <function tors.aio.shingle_dice at 0x7f71dbc4a700>
shingle_jaccard -> <function tors.aio.shingle_jaccard at 0x7f0d389467a0>
similarity_ratio -> <function tors.aio.similarity_ratio at 0x7f1c7db46840>
truncate_to_bounds -> <function tors.aio.truncate_to_bounds at 0x7f9a6f602b60>
```

## Discrepancies found

1. **rank family (5 of the 22) already has a collective Async line**: `docs/api.md:5553-5554` says "each of the five has an `await tors.aio.<name>(...)` twin". The audit's per-name pattern missed it. The entry above is a replacement that adds the measured GIL-hold caveat; alternatively the audit can recount those five as documented.
2. **Three sections state the twin in trailing prose, not an Async line**: `tors.lsh_candidates` ("`aio` twin: `tors.aio.lsh_candidates`."), `tors.shingle_jaccard` / `tors.shingle_dice` ("`aio` twins: ..."), `tors.dedup_near_dup` ("`aio` twin: ..."). Inserting the Async line duplicates that sentence; fold it into the Async line or delete it (noted per entry).
3. **No twin is absent and no sync section is absent**: all 22 exist in `tors.aio._WRAPPED`, import from the built module, and each has an existing `##` section in `docs/api.md`.
4. **No test contradicts any cost claim written above.** One honesty note: `levenshtein`/`jaro`/`jaro_winkler`, `similarity_ratio`, `is_grounded` (both paths), `highlight`, and the `shingle_*` pair have no direct heartbeat cells of their own (only the bulk `get_close_matches`, `ground_sentences`, `grounding_coverage` siblings were measured); their lines rest on the impl GIL comments plus the documented bands cited per entry, which is why none of those lines invents a numeric crossover.
5. **Module-docstring tension, worth a follow-up, not a docs error**: `python/tors/aio.py`'s docstring says microsecond-scale calls stay sync-only, yet wraps the fuzzy pair metrics (whose common calls are microsecond-scale). The per-call guidance above resolves this the only honest way: sync below KB-scale operands, `tors.aio` on the adversarial multi-KB/MB operands the wrap exists for.
