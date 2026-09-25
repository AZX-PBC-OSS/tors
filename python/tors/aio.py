"""The awaitable spellings of tors's large-input functions.

Every tors function releases the GIL for its whole native pass, but
releasing the GIL is not the same as not blocking: a native pass called
directly from a coroutine still occupies that coroutine's own turn on the
event loop for the call's full wall-clock duration (the GIL release lets
other threads make progress; it does not hand control back to the loop).
The fix is a thread hop, and this module ships it pre-wired:
``await tors.aio.tf_idf(corpus)`` runs the native pass in a worker
thread, and the event loop stays responsive for its whole duration.

Which functions, and why only these: thread dispatch through
``asyncio.to_thread`` costs on the order of tens of microseconds, noise
next to a millisecond-or-slower native pass over a real corpus or
document, real overhead next to a microsecond-scale call over a short
string. So this module covers only the functions whose realistic inputs
are large enough that the thread-hop cost is reliably negligible: the
chunking family, the retrieval/scoring primitives, the diff engine, the
batch pipeline, the fuzzy-matching and JSON-repair families
(``levenshtein``/``jaro``/``jaro_winkler``,
``similarity_ratio``/``get_close_matches``, ``is_grounded``, the
grounding pair ``ground_sentences``/``grounding_coverage``, the
``repair_json*`` trio: quadratic and linear native passes whose
documented measurements reach seconds and minutes on large inputs), and
the input-scaling text/byte pipeline codecs
(``normalize``/``finalize``, ``decode_utf8``/``finalize_utf8``/
``decode_utf16``, ``b64_encode_bytes``/``b64_decode``,
both truncate spellings, ``strip_controls``, ``scrub_log_text``,
``scrub_pii``/``scrub_pii_report``, the bounds reporters
``word_bounds``/``sentence_bounds``, the search/replace family
``find_patterns``/``count_matches``/``replace_many``/
``replace_many_masked``, and the code-block family
``extract_code_blocks``/``strip_code_fences``); each a single native pass
whose cost scales with its input (``scrub_log_text``: four linear scans +
splice under one ``py.detach``), e.g. ``finalize`` over a 12 MiB
document. Exception-size guidance: ``scrub_log_text``'s error-path inputs
are KiB-scale (a single message/traceback scrubs in microseconds, well
under the hop cost — prefer the sync spelling there); its ``aio`` twin is
for MB-scale aggregates only (batched logs, multi-MB exception corpora),
where the hop is noise next to the pass. Every other tors function keeps exactly one spelling
(the sync one); call it directly from a coroutine when the input is
small; a synchronous call that finishes in microseconds does not need
asyncio at all, and wrapping it here would be lying about a cost that
isn't there. See docs/async.md for the size guidance in
full.

There is no size-based branch inside any wrapper here, and there never
will be: a function that sometimes runs inline and sometimes hops to a
thread depending on its input is unpredictable from the caller's side and
can silently block the loop when the heuristic misjudges. Every function
in this module always dispatches through ``asyncio.to_thread``,
unconditionally. The choice between the sync spelling and this module is
the caller's, made once at the call site, not a runtime guess.

The excluded shapes: the streaming iterator constructors
(``word_bounds_iter`` and siblings, including the chunking family's own
``chunk_text_iter``/``chunk_by_words_iter``/``chunk_by_sentences_iter``/
``chunk_by_paragraphs_iter``/``chunk_by_lines_iter``) are not wrapped,
because an iterator is not an awaitable shape and draining one to a list
inside a worker thread is exactly what the already-covered list-returning
sibling (``chunk_text``, ``chunk_by_words``, ``chunk_by_sentences``,
``chunk_by_paragraphs``, ``chunk_by_lines``) already does; a caller who
wants the streaming shape under asyncio writes
``await asyncio.to_thread(lambda: list(tors.word_bounds_iter(text)))``
directly; that one line is the whole pattern, matching the manual
``to_thread`` wrap this module exists to save callers from writing
repeatedly for the functions it pre-wires.

The signatures are identical to the sync spellings (pinned by
tests/test_aio.py against the live functions, and the stub ``aio.pyi``
is checked against ``__init__.pyi`` by the same gate); keyword-only
parameters pass through unchanged, including ``lemma_dict`` accepting
either a plain ``dict`` or a ``tors.CompiledLemmaDict``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any

import tors

__all__: list[str] = []

# The curated large-input subset, named explicitly rather than inferred
# from tors.__all__ by exclusion (see the module docstring for why the
# rest of tors intentionally has no async twin): the chunking family, the
# retrieval/scoring primitives, the diff engine, the batch pipeline, the
# fuzzy-matching, grounding, and JSON-repair families (quadratic/linear native passes
# whose docs measure seconds-to-minutes on large inputs), and the
# input-scaling text/byte pipeline codecs (normalize/finalize,
# decode_utf8/finalize_utf8/decode_utf16, b64_encode_bytes/b64_decode,
# both truncate spellings, strip_controls, scrub_log_text, scrub_pii (and
# its report twin), word_bounds/sentence_bounds, the search/replace family
# (find_patterns, count_matches, replace_many, replace_many_masked), and
# the code-block family (extract_code_blocks, strip_code_fences): each a
# single native pass
# whose cost scales with its input (scrub_log_text: four linear scans + splice under
# one py.detach), the 12 MiB-document shape this module exists
# for). Microsecond-scale calls over short strings (the normalization
# forms, html_unescape, quote/unquote, the utf8/utf16 validity booleans and
# json_is_valid (whose 1 MiB ceiling scans sub-millisecond, the
# utf8_is_valid class), detect_encoding's guess, the random generators — a block-buffered
# syscall plus sampling/formatting at every realistic token/key size) stay
# sync-only: the thread
# hop would cost more than the call itself.
_WRAPPED = (
    "apply_pipeline",
    "b64_decode",
    "b64_encode_bytes",
    "bm25_rank",
    "chunk_by_lines",
    "chunk_by_paragraphs",
    "chunk_by_sentences",
    "chunk_by_words",
    "chunk_cdc",
    "chunk_hierarchical",
    "chunk_text",
    # chunk_to_budget is the one wrapped function that is not a single
    # detached native pass: its token_counter is a Python callable that
    # can only run under the GIL. The hop still helps when the loop can
    # take the GIL between callbacks -- the packing core runs detached
    # and re-attaches the GIL per counter call, so the worker's
    # per-callback handoffs interleave with the event loop's thread
    # (tests/test_gil_release.py pins the gap tracking the callbacks,
    # never the call) -- where the sync spelling run inline would hold
    # the GIL for the whole packing. Honest caveat, measured (finding
    # R1): the interleave needs each callback to exceed
    # sys.getswitchinterval() (5ms default) or substantial native
    # windows between them; a sub-switch-interval callback on a small
    # text can starve the loop for the whole call (the drop and
    # re-acquire outruns the woken loop thread), for this hop exactly as
    # for the sync spelling in a worker thread.
    "chunk_to_budget",
    "chunk_to_offsets",
    "count_matches",
    "decode_utf16",
    "decode_utf8",
    "diff_opcodes",
    "diff_opcodes_lines",
    "extract_code_blocks",
    "finalize",
    "finalize_utf8",
    "find_patterns",
    "get_close_matches",
    "ground_sentences",
    "grounding_coverage",
    "highlight",
    "is_grounded",
    "jaro",
    "jaro_winkler",
    "levenshtein",
    "minhash_signature",
    "mrr",
    "ndcg_at_k",
    "normalize",
    "precision_at_k",
    "rank_fuse",
    "recall_at_k",
    "repair_json",
    "repair_json_diagnostics",
    "repair_json_loads",
    "replace_many",
    "replace_many_masked",
    "scrub_log_text",
    "scrub_pii",
    "scrub_pii_report",
    "sentence_bounds",
    "similarity_ratio",
    "strip_code_fences",
    "strip_controls",
    "tf_idf",
    "truncate_ellipsis",
    "truncate_to_bounds",
    "word_bounds",
)


def _make_async(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    # @wraps carries the sync function's name, docstring, and (through
    # __wrapped__) its signature, so inspect on the wrapper reports the
    # sync spelling's parameters exactly (pinned by tests/test_aio.py).
    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(fn, *args, **kwargs)

    wrapper.__qualname__ = f"tors.aio.{name}"
    return wrapper


for _name in _WRAPPED:
    _fn = getattr(tors, _name)
    globals()[_name] = _make_async(_name, _fn)
    __all__.append(_name)

__all__.sort()
del _name, _fn
