# Async use

Releasing the GIL is not the same as not blocking. A native pass called
directly from a coroutine still occupies that coroutine's own turn on the
event loop for the call's full wall-clock duration. `tors.aio` is the
pre-wired fix for the functions where that matters: `await
tors.aio.tf_idf(corpus)` runs the native pass in a worker thread via
`asyncio.to_thread`, and the event loop stays responsive for the whole call.

It covers only the large-input functions: the chunking family, `tf_idf`,
`bm25_rank`, `diff_opcodes`, `diff_opcodes_lines`, `apply_pipeline`, the
`normalize`/`finalize` pipeline pair, the `decode_utf8`/`finalize_utf8`/
`decode_utf16`/`b64_encode_bytes`/`b64_decode` byte codecs, and
`truncate_ellipsis`/`strip_controls`. Thread dispatch costs on the order of
tens of microseconds: noise next to a millisecond-or-slower native pass over a
real corpus or document, real overhead next to a microsecond-scale call over a
short string. Wrapping every export would make the small, common calls slower
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
await).

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
