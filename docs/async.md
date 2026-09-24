# Async use

Releasing the GIL is not the same as not blocking. A native pass called
directly from a coroutine still occupies that coroutine's own turn on the
event loop for the call's full wall-clock duration. `tors.aio` is the
pre-wired fix for the functions where that matters:

```python
import tors.aio  # the facade is its own module, imported explicitly

await tors.aio.tf_idf(corpus)
```

The `await tors.aio.tf_idf(corpus)` call runs the native pass in a worker
thread via `asyncio.to_thread`, and the event loop stays responsive for
the whole call.

It covers only the large-input functions: the chunking family, `tf_idf`,
`bm25_rank`, the rank-fusion family (`rank_fuse`, `ndcg_at_k`, `mrr`,
`recall_at_k`, `precision_at_k` — the fusion/membership walks over large
rankings are interpreter-side hashing, so the worker thread buys their
detached arithmetic plus the caller's concurrency shape, and the measured
GIL bands in test_gil_release.py carry the honest residue),
`diff_opcodes`, `diff_opcodes_lines`, `apply_pipeline`,
`minhash_signature`, `highlight` (the grounding pass is linear in the
chunk with the DP capped, but a 2k-token chunk already measures ~1 ms —
thread-hop territory), the
`normalize`/`finalize` pipeline pair, the `decode_utf8`/`finalize_utf8`/
`decode_utf16`/`b64_encode_bytes`/`b64_decode` byte codecs, the
fuzzy-matching and JSON-repair families
(`levenshtein`/`jaro`/`jaro_winkler`, `similarity_ratio`/
`get_close_matches`, `is_grounded`, and the `repair_json*` trio:
quadratic and linear native passes whose documented measurements reach
seconds and minutes on large inputs, exactly the calls that starve a loop
un-wrapped), both truncate spellings,
`strip_controls`/`scrub_log_text`/`scrub_pii`/`scrub_pii_report`, and the
bounds reporters `word_bounds`/`sentence_bounds` (their marshalling band
at whole-file sizes is the list shape's own; see performance.md). Thread dispatch costs on the order of
tens of microseconds: noise next to a millisecond-or-slower native pass over a
real corpus or document, real overhead next to a microsecond-scale call over a
short string. Exception-size guidance: `scrub_log_text`'s error-path inputs
are KiB-scale (microseconds per call — prefer the sync spelling); its `aio`
twin is for MB-scale aggregates only (batched logs, multi-MB corpora).
Wrapping every export would make the small, common calls slower
through this module than through the plain sync spelling, for no benefit, so
the rest of `tors` keeps exactly one spelling: call it directly from a
coroutine when the input is small enough that the whole thing finishes in
microseconds.

There is no size-based branch inside any wrapper, and there never will be: a
function that sometimes runs inline and sometimes hops to a thread depending
on its input is unpredictable from the caller's side and can silently block
the loop when the heuristic misjudges. Every function in `tors.aio` dispatches
through `asyncio.to_thread` unconditionally; the choice between the sync
spelling and `tors.aio` is the caller's, made once at the call site, not a
runtime guess. `tests/test_aio.py` pins this structurally (no branch in the
wrapper body) as well as behaviorally (a heartbeat coroutine keeps ticking
with worst gaps well under the call's own wall during a large `diff_opcodes`
await), and pins the covered set against `tors.aio._WRAPPED`: the curated
list is the contract, and the families it names are the input-scaling set
it covers — the chunking, fuzzy-matching, and JSON-repair families
(minutes-scale calls most need the thread hop), the search/replace family
(`find_patterns`, `count_matches`, `replace_many`,
`replace_many_masked`), the code-block family (`extract_code_blocks`,
`strip_code_fences`), and the bounds reporters (`word_bounds`,
`sentence_bounds`) — the whole surface is wrapped now, so a consumer
never hand-rolls the `asyncio.to_thread` spelling for a batch call.

The streaming iterator constructors (`word_bounds_iter` and siblings,
including the chunking family's own `chunk_text_iter`/`chunk_by_words_iter`/
`chunk_by_sentences_iter`/`chunk_by_paragraphs_iter`/`chunk_by_lines_iter`)
have no async twin: an iterator is not an awaitable shape, and draining one to
a list inside a worker thread is exactly what the already-covered
list-returning sibling does. The eager construction pass is the GIL-released
part anyway, so `await asyncio.to_thread(lambda: list(tors.word_bounds_iter(text)))`
covers the streaming shape when it is needed. Signatures are identical to the
sync spellings, pinned by `tests/test_aio.py`; the stub `aio.pyi` is generated
by `tools/gen_aio_stub.py`.

The random-generation family (`random_string`, `random_hex`, `random_b62`,
`random_b64url`, `uuid4`, `uuid7`) has no async twin either: every generator
is a fast CPU/syscall call — a block-buffered getrandom draw plus
sampling/formatting, microseconds at real token/key sizes — not the
detached-transform input class this module exists for. A thread hop would
cost more than the call at every realistic size; call them directly from a
coroutine.
