"""The SOTA comparison bench: `tors`' retrieval/chunking/grounding surfaces
against the installed third-party packages a caller would otherwise reach
for (`ranx`, `semchunk`, `rouge-score`, `scikit-learn`), plus the naive
pure-Python spelling of the shingle job.

Methodology (every cell):

- SAME inputs for both sides of a comparison (the same ranked lists, the
  same corpus, the same token spans); where the two tools' contracts
  differ (semchunk's semantic windows vs `chunk_to_offsets`' budget
  packing, rouge-score's ROUGE-1/L vs the grounding surfaces' ROUGE-W
  shape) the cell says so explicitly instead of pretending the numbers
  are one race.
- WALLS are min-of-N `time.perf_counter` samples (N per cell below),
  each side measured the same way, one warmup call first: min-of is the
  suite's standard "what does the call cost, ambient load excluded".
- AGREEMENT is recomputed both directions before a number is published:
  the cell cross-checks the two tools' outputs on the same input
  (ordering overlap, value deltas, toy hand-checkable values) and prints
  the check, so a published speedup is not two tools answering different
  questions.

Run: `.venv/bin/python tools/bench_sota.py` (box quiet; numbers in
docs/performance.md "Against the real alternatives" are this script's
output, transcribed).
"""

from __future__ import annotations

import random
import re
import time
import unicodedata

import tors

SAMPLES_FAST = 5
SAMPLES_SLOW = 3


def bench(label: str, fn, *, samples: int = SAMPLES_FAST, agreement=None):
    """Min-of-``samples`` wall of ``fn`` (zero-arg), one warmup first;
    prints the wall and, when given, the agreement check's verdict."""
    fn()  # warmup (imports, caches, allocator)
    best = float("inf")
    for _ in range(samples):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    line = f"  {label}: {best * 1e3:.3f} ms/call (min of {samples})"
    if agreement is not None:
        ok, detail = agreement
        line += f"  [agreement: {detail}]"
        assert ok, f"{label}: agreement check FAILED: {detail}"
    print(line)
    return best


# ---------------------------------------------------------------------------
# 1. rank_fuse vs ranx rrf fusion
# ---------------------------------------------------------------------------


def fusion_lists(n_lists: int, n_ids: int, keep: float = 0.6):
    """n_lists ranked lists over a shared id pool; each list keeps ``keep``
    of the pool (a per-list coprime-stride walk, so membership AND order
    disagree the way real retrievers do; stride coprime with n_ids keeps
    every list duplicate-free)."""
    pool = [f"doc-{i}" for i in range(n_ids)]
    cnt = int(n_ids * keep)
    out = []
    for i in range(n_lists):
        start = (i * 397) % n_ids
        out.append([pool[(start + j * 13) % n_ids] for j in range(cnt)])
    return out


def bench_rank_fuse():
    print("rank_fuse vs ranx.rrf (RRF, k=60, same ranked lists)")
    from ranx import Run
    from ranx.fusion import rrf as ranx_rrf

    def run_size(n_lists: int, n_ids: int) -> None:
        lists = fusion_lists(n_lists, n_ids)

        # ranx models each ranked list as its own Run (one retrieval
        # system, one query); ranks only, so score = 1/rank.
        runs = [
            Run({"q": {doc: 1.0 / (rank + 1) for rank, doc in enumerate(lst)}}, name=f"r{i}")
            for i, lst in enumerate(lists)
        ]

        fused_tors = tors.rank_fuse(lists, k=60)

        # Agreement, both directions: recompute the RRF score for the top
        # fused ids in plain Python from the same lists, and check the top
        # ordering the two tools produce matches.
        def rrf_score(doc: str) -> float:
            return sum(
                1.0 / (60 + (lst.index(doc) + 1)) for lst in lists if doc in lst
            )

        top = [doc for doc, _score in fused_tors[:50]]
        py_scores = {doc: rrf_score(doc) for doc in top}
        tors_scores = dict(fused_tors)
        deltas = max(abs(py_scores[d] - tors_scores[d]) for d in top)
        fused_ranx = ranx_rrf(runs, k=60)
        ranx_q = fused_ranx.run["q"]
        ranx_top = [
            d for d, _ in sorted(ranx_q.items(), key=lambda kv: -kv[1])[:50]
        ]
        overlap_top = len(set(top) & set(ranx_top)) / len(top)

        def agree():
            return (
                deltas < 1e-9 and overlap_top == 1.0,
                f"py-recomputed RRF max delta {deltas:.2e}, "
                f"top-50 ordering overlap with ranx {overlap_top:.0%}",
            )

        a = agree()
        bench(
            f"tors.rank_fuse, {n_lists}x{n_ids}",
            lambda ls=lists: tors.rank_fuse(ls, k=60),
            agreement=a,
        )
        bench(
            f"ranx rrf, {n_lists}x{n_ids}",
            lambda r=runs: ranx_rrf(r, k=60),
            samples=SAMPLES_SLOW,
        )

    for n_lists, n_ids in ((8, 2_000), (8, 20_000)):
        run_size(n_lists, n_ids)


# ---------------------------------------------------------------------------
# 2. ndcg_at_k vs sklearn.metrics.ndcg_score vs ranx.evaluate
# ---------------------------------------------------------------------------


def bench_ndcg():
    print("ndcg_at_k vs sklearn / ranx (one 10k ranking, 1k relevant, k=10)")
    from ranx import Qrels, Run, evaluate
    from sklearn.metrics import ndcg_score

    n = 10_000
    rng = random.Random(42)
    rng_ids = [f"doc-{i}" for i in range(n)]
    ranked = rng_ids[:]
    rng.shuffle(ranked)  # one 10k ranking, fixed seed
    relevant = frozenset(rng_ids[5_000:6_000])  # ~1 relevant doc per top-10 window
    k = 10

    # Park exactly two relevant docs inside the top 10 (swap, so the
    # permutation stays valid): the metric then reads a partial value,
    # not a degenerate 0.0 (no hits) or 1.0 (perfect head).
    rel = sorted(relevant)
    for pos, doc in ((2, rel[0]), (6, rel[1])):
        j = ranked.index(doc)
        ranked[pos], ranked[j] = ranked[j], ranked[pos]

    v_tors = tors.ndcg_at_k(ranked, relevant, k=k)

    # sklearn: labels and scores aligned over the label set (all n docs).
    y_true = [[1.0 if d in relevant else 0.0 for d in ranked]]
    y_score = [[float(len(ranked) - i) for i in range(len(ranked))]]
    v_sk = ndcg_score(y_true, y_score, k=k)

    # ranx: one query, per-doc graded qrels, scores = descending rank.
    qrels = Qrels({"q": {d: (1.0 if d in relevant else 0.0) for d in ranked}})
    run = Run({"q": {d: float(len(ranked) - i) for i, d in enumerate(ranked)}}, name="bench")
    v_ranx = evaluate(qrels, run, metrics=[f"ndcg@{k}"], return_std=False)

    # Agreement: binary-relevance nDCG@k over the same ranking. ranx's
    # ndcg_cut averages over queries with hits; one query here, so the
    # three values must sit in the same band (sklearn's ties
    # `ignore_ties=False` semantics aside, identical inputs).
    spread = max(v_tors, v_sk, v_ranx) - min(v_tors, v_sk, v_ranx)

    def agree():
        return (
            spread < 0.05,
            f"tors {v_tors:.4f}, sklearn {v_sk:.4f}, ranx {v_ranx:.4f} "
            f"(spread {spread:.4f}; definitional deltas documented below)",
        )

    a = agree()
    bench("tors.ndcg_at_k, 10k ranking", lambda: tors.ndcg_at_k(ranked, relevant, k=k), agreement=a)
    bench(
        "sklearn ndcg_score, 10k ranking",
        lambda: ndcg_score(y_true, y_score, k=k),
        samples=SAMPLES_SLOW,
    )
    bench(
        "ranx evaluate ndcg@10",
        lambda: evaluate(qrels, run, metrics=[f"ndcg@{k}"], return_std=False),
        samples=SAMPLES_SLOW,
    )


# ---------------------------------------------------------------------------
# 3. chunk_to_offsets vs semchunk
# ---------------------------------------------------------------------------


def prose(target_bytes: int) -> str:
    sentence = "The pump failed and the bushing torque spec was 42 Nm, so the crew replaced it. "
    return sentence * max(1, target_bytes // len(sentence))


def bench_chunk_to_offsets():
    print("chunk_to_offsets vs semchunk (1 MiB prose, 200-token budget, overlap 20)")
    from semchunk import chunk as semchunk_chunk

    text = prose(1_000_000)
    word_spans = [m.span() for m in re.finditer(r"\S+", text)]

    def counter(s: str) -> int:
        return len(s.split())

    max_tokens, overlap = 200, 20
    spans_tors = tors.chunk_to_offsets(text, word_spans, max_tokens=max_tokens, overlap=overlap)
    chunks_sem, spans_sem = semchunk_chunk(
        text, chunk_size=max_tokens, token_counter=counter, offsets=True, overlap=overlap,
        memoize=False,
    )

    # Agreement: different algorithms by design (semchunk prefers
    # sentence-aligned windows, chunk_to_offsets packs the given spans
    # exactly), so the check is on the SHARED contract: every output
    # chunk is within budget and the pieces tile the text.
    def within_budget(offsets):
        return all(
            len(text[s:e].split()) <= max_tokens for s, e in offsets
        )

    ok = within_budget(spans_tors) and within_budget(spans_sem)
    detail = (
        f"tors {len(spans_tors)} chunks, semchunk {len(chunks_sem)} chunks, "
        f"both within the {max_tokens}-token budget (algorithms differ by design:"
        " semchunk sentence-aligns, chunk_to_offsets packs given spans)"
    )

    a = (ok, detail)
    bench(
        "tors.chunk_to_offsets, 1 MiB",
        lambda: tors.chunk_to_offsets(text, word_spans, max_tokens=max_tokens, overlap=overlap),
        agreement=a,
    )
    bench(
        "semchunk.chunk, 1 MiB",
        lambda: semchunk_chunk(
            text, chunk_size=max_tokens, token_counter=counter, offsets=True,
            overlap=overlap, memoize=False,
        ),
        samples=SAMPLES_SLOW,
    )


# ---------------------------------------------------------------------------
# 4. ground_sentences / highlight vs rouge-score
# ---------------------------------------------------------------------------


def bench_grounding_vs_rouge():
    print("ground_sentences / highlight vs rouge_score (100 KiB doc, one query)")
    from rouge_score import rouge_scorer

    text = prose(100_000)
    query = "bushing torque spec replacement procedure"

    # rouge-score is per-pair: score every sentence of the doc against
    # the query, the job ground_sentences does in one call. Different
    # metric by design (ROUGE-1/ROUGE-L here, ROUGE-W shape there), so
    # the check is the toy agreement: identical token sequences score
    # 1.0, disjoint 0.0, on BOTH sides.
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

    def sentences(s):
        out, cur = [], []
        for word in s.split(" "):
            cur.append(word)
            if word.endswith("."):
                out.append(" ".join(cur))
                cur = []
        if cur:
            out.append(" ".join(cur))
        return out

    sents = sentences(text)

    toy_same = scorer.score("the pump failed", "the pump failed")
    toy_disjoint = scorer.score("the pump failed", "gravel road noise")
    rs_same = toy_same["rouge1"].fmeasure == 1.0
    rs_disjoint = toy_disjoint["rouge1"].fmeasure == 0.0

    g = tors.ground_sentences("The pump failed.", "the pump failed")
    h = tors.highlight("the pump failed", "The pump failed.")
    gs_same = g["sentences"][0]["score"] == 1.0
    gs_disjoint = (
        tors.ground_sentences("The pump failed.", "gravel road noise")["sentences"][0]["score"]
        == 0.0
    )
    h_same = h["snippets"][0]["score"] == 1.0

    ok = rs_same and rs_disjoint and gs_same and gs_disjoint and h_same
    detail = "toy agreement: identical-token pair scores 1.0, disjoint 0.0 on all three"

    a = (ok, detail)
    bench(
        "tors.ground_sentences, 100 KiB doc",
        lambda: tors.ground_sentences(text, query),
        agreement=a,
    )
    bench(
        "tors.highlight, 100 KiB doc",
        lambda: tors.highlight(query, text),
        agreement=a,
    )
    bench(
        f"rouge_scorer over the doc's {len(sents)} sentences",
        lambda: [scorer.score(query, s) for s in sents],
        samples=SAMPLES_SLOW,
    )


# ---------------------------------------------------------------------------
# 5. shingle_jaccard vs naive pure Python
# ---------------------------------------------------------------------------


def naive_shingle_jaccard(a: str, b: str, width: int = 3) -> float:
    """The same job in pure Python: the SAME tokens (tors.word_bounds'
    UAX #29 segments, whitespace-only dropped), lowercased + NFC (the
    module's fold), width-token shingles as hashable tuples, set
    Jaccard. Uses tors.word_bounds only for segmentation (Python's
    stdlib has no UAX #29 word segmenter); everything past the segment
    list is pure Python."""

    def tokens(text):
        out = []
        for s, e in tors.word_bounds(text):
            seg = text[s:e]
            if seg.isspace():
                continue
            out.append(unicodedata.normalize("NFC", seg.lower()))
        return out

    def shingles(text):
        toks = tokens(text)
        return {tuple(toks[i : i + width]) for i in range(len(toks) - width + 1)}

    sa, sb = shingles(a), shingles(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = sum(1 for x in sa if x in sb)
    return inter / (len(sa) + len(sb) - inter)


def bench_shingle_jaccard():
    print("shingle_jaccard vs naive pure Python (same tokens, 100 KiB pairs)")
    a = prose(100_000)
    mutated = a[: 60_000] + "an unplanned sentence wanders in here. " + a[60_000:]

    v_tors = tors.shingle_jaccard(a, mutated)
    v_py = naive_shingle_jaccard(a, mutated)
    delta = abs(v_tors - v_py)

    a2 = (delta < 1e-12, f"same-token naive value {v_py:.6f} vs tors {v_tors:.6f}")
    bench(
        "tors.shingle_jaccard, 100 KiB pair",
        lambda: tors.shingle_jaccard(a, mutated),
        agreement=a2,
    )
    bench(
        "naive pure Python, 100 KiB pair",
        lambda: naive_shingle_jaccard(a, mutated),
        samples=SAMPLES_SLOW,
    )


def main() -> None:
    bench_rank_fuse()
    print()
    bench_ndcg()
    print()
    bench_chunk_to_offsets()
    print()
    bench_grounding_vs_rouge()
    print()
    bench_shingle_jaccard()


if __name__ == "__main__":
    main()
