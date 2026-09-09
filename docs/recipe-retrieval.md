# Recipe: a lightweight retrieval/reranking pipeline

The realistic shape of small-corpus search without reaching for a vector
database or an embedding model: chunk a corpus, drop near-duplicate chunks,
score or rank by keyword overlap, and lexically sanity-check a downstream
claim against what was actually retrieved.

Say this plainly up front: this is a **lexical, statistical** pipeline —
SimHash, TF-IDF, BM25, substring/fuzzy matching. No embeddings, no neural
ranking, no semantic understanding of either the query or the documents.
It's the "fast, no GPU, no API key, good enough for tens to a few hundred
documents" tier, not a claim that it beats an embedding-based retrieval
system on relevance. Reach for one of those when this tier's quality isn't
enough; reach for this when it's enough and you'd rather not run one.

## 1. Chunk the corpus

Use `tors.chunk_by_words` (or `chunk_hierarchical` for structured sources —
see the [ingestion recipe](recipe-ingest.md)) to split each document into
retrieval-sized pieces:

```python
import tors

doc = " ".join(f"word{i}" for i in range(40))
chunks = tors.chunk_by_words(doc, 10)
# [(0, 59), (60, 129), ...] — 4 chunks of 10 words each
```

Do this per document in your corpus, and keep the resulting chunk strings in
a flat list — that flat list is what every step below operates on.

## 2. Drop near-duplicate chunks

Two chunkers producing overlapping windows, or a corpus with genuinely
repeated content, both leave near-identical chunks behind. `simhash64` gives
each chunk a 64-bit fingerprint; Hamming distance between two fingerprints
grows slowly with edit distance, so near-duplicates cluster at a small
distance while unrelated chunks sit far apart:

```python
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog!",  # one-character edit
    "A lazy cat sleeps all day in the warm sun.",
    "Rust is a systems programming language focused on safety and speed.",
]

fingerprints = [tors.simhash64(c) for c in corpus]


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


hamming(fingerprints[0], fingerprints[1])  # near-duplicate pair
# 1
hamming(fingerprints[0], fingerprints[3])  # unrelated pair
# 22
```

`tors` doesn't ship a distance function — `(a ^ b).bit_count()` at the call
site is the whole thing. There's also no universal "this many bits means
duplicate" cutoff: the gap between a near-duplicate band and the unrelated
floor scales with document length and corpus vocabulary, so calibrate a
threshold against known near-dup and known-far pairs from your own corpus
rather than importing a fixed number. `simhash128` is the same construction
at twice the width, for corpora where the 64-bit bands sit too close
together to separate reliably.

If your chunk boundaries are stable across runs and you want to detect
exactly which chunks changed between two versions of the same document
rather than fuzzy near-duplicates, `merkle_diff` answers a different
question — exact equality per index, not similarity:

```python
a = "The quick brown fox jumps over the lazy dog.".split()
b = "The quick brown fox jumps over the lazy dog!".split()
tors.merkle_diff([w.encode() for w in a], [w.encode() for w in b])
# [8]  — only the last word-chunk differs
```

Use `simhash64`/`simhash128` when you don't control chunk alignment and want
"how similar." Use `merkle_diff` when chunk boundaries line up positionally
(the same document re-chunked the same way) and you want "which indices
changed, exactly."

## 3. Score or rank

Two primitives, different jobs. `tf_idf` scores every term in every
document against the whole corpus — useful for surfacing a document's most
distinctive terms, or comparing documents by their score vectors:

```python
tfidf = tors.tf_idf(corpus)
sorted(tfidf[0], key=lambda term_score: -term_score[1])[:3]
# [('the', 2.4462871026284194), ('brown', 1.5108256237659907), ('dog', 1.5108256237659907)]
```

`bm25_rank` reranks the corpus against one query, which is the shape a RAG
pipeline actually wants after a first-pass retrieval step narrows things
down to a small candidate set:

```python
tors.bm25_rank("quick fox", corpus)
# [(0, 1.4312282719845209), (1, 1.4312282719845209), (2, 0.0), (3, 0.0)]
```

Results are `(index, score)` pairs for every document, sorted descending, no
top-k cutoff applied — slice the result yourself. `bm25_rank` recomputes
corpus statistics from scratch on every call, which is the right shape for
reranking tens to a few hundred already-retrieved candidates and the wrong
shape for querying a corpus of thousands repeatedly (build a real inverted
index for that; `tantivy` is the standard choice in Rust). Neither function
makes any claim about retrieval *quality* — BM25 and TF-IDF are correctly
implemented ranking formulas, not a promise that keyword overlap is what
your downstream task needs.

## 4. Verify a claim is actually grounded

Once a passage comes back from retrieval and a model generates an answer
citing it, `is_grounded` gives a cheap lexical check that the claim is
actually supported by the source text — not a hallucination-detection
model, a substring/fuzzy-match check:

```python
source = corpus[0]  # "The quick brown fox jumps over the lazy dog."

tors.is_grounded("the fox jumps over the dog", source)
# False — not an exact substring

tors.is_grounded("the fox jumps over the dog", source, fuzzy=True, threshold=0.6)
# True — close enough under a difflib-style ratio

tors.is_grounded("the fox can fly to the moon", source, fuzzy=True, threshold=0.6)
# False — genuinely unsupported
```

`fuzzy=False` (the default) is an exact substring check. `fuzzy=True`
compares the claim against overlapping windows of the source and passes if
the best window's similarity ratio clears `threshold`. This catches
paraphrase-shaped near-misses (word order, minor rewording) that an exact
substring check would reject — it does not catch semantic entailment. A
claim can pass this check and still not follow logically from the source,
and a claim can fail it while being a reasonable paraphrase if `threshold`
is set too high. Treat it as a fast pre-filter (does the model's answer even
lexically resemble something in the retrieved text) ahead of a heavier check,
not as a standalone correctness guarantee.

## The whole pipeline, together

```python
import tors


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def search(corpus: list[str], query: str, *, dup_threshold: int = 3) -> list[tuple[int, float]]:
    # drop near-duplicate chunks before scoring
    fingerprints = [tors.simhash64(c) for c in corpus]
    keep = []
    for i, fp in enumerate(fingerprints):
        if not any(hamming(fp, fingerprints[j]) <= dup_threshold for j in keep):
            keep.append(i)
    deduped = [corpus[i] for i in keep]

    ranked = tors.bm25_rank(query, deduped)
    return [(keep[i], score) for i, score in ranked]
```

`dup_threshold` here is a placeholder, not a recommendation — measure it
against your own corpus before trusting it in production, per the
calibration note above.
