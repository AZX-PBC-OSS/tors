# Design and scope

## Stateless by default

`tors` is a stateless library: every call does its own work from scratch, with
nothing cached or built up across calls. Every function stays simple to reason
about and safe to call from anywhere: no handle to manage, no invalidation to
think about, no stale cache.

`CompiledLemmaDict` is one narrow exception; `CompiledPatterns` (a fixed
pattern list compiled once) is the other. `tf_idf`, `bm25_rank`, and
`apply_pipeline` accept an optional caller-supplied `lemma_dict`: a
`word -> lemma` map. A raw `dict[str, str]` is re-materialized into a Rust
`HashMap` on every call, and for a realistic multi-thousand-entry lemma table
that cost is measured, not theoretical: roughly 1.4ms per call on a
20,000-entry map, independent of how much text the call actually processes,
which can make repeated small calls slower than the equivalent pure-Python
loop. `CompiledLemmaDict` builds that `HashMap` once and hands back an
immutable, cheaply cloned handle: the same `re.compile()` shape the stdlib
already uses for a comparable problem. It exists for this one measured cost
and does not reopen the case for a general pipeline object; beyond its
sibling `CompiledPatterns`, nothing else in this library gets a persistent
handle.

## Scope cuts

The cuts below are decisions, not oversights:

- **General regex.** `find_patterns`/`replace_many` are leftmost-longest
  multi-pattern literal search, not a regex engine; `chunk_hierarchical`'s
  `separators` are literal strings (or `None`, splicing in the default
  hierarchy), not patterns.
- **Schema-aware JSON/YAML coercion.** The JSON side moved IN with the
  `repair_json` family: syntax repair of malformed JSON and schema-guided
  alignment/coercion against a JSON Schema are algorithms (a repair parser's
  heuristics plus a standard validator over caller-supplied schema data, the
  shape json_repair itself proved), not a model, so they fit this crate's
  posture. Still out, for the original reason: other schema languages,
  YAML/TOML repair, prompt-side constrained generation (the BAML-style
  problem of steering the model while it writes, rather than repairing
  after), and full JSON-Schema-language tooling. `tors` repairs and validates
  against schemas; it does not generate them.
- **A bundled lemma dictionary.** `apply_pipeline`/`tf_idf`/`bm25_rank` apply
  a caller-supplied lemma map; `tors` ships no lemma data of its own, because
  full lemmatization needs a per-language dataset or a POS-tagging model, not
  an algorithm. That is outside a text-operations library's job.
- **A persistent search index.** `bm25_rank` recomputes corpus statistics
  from scratch on every call: the right shape for reranking a small,
  already-retrieved candidate set, the wrong shape for querying a large
  corpus repeatedly. Reach for a real search engine (`tantivy`, in Rust) for
  that; `tors` does not build or expose index objects.
- **A bespoke coroutine API.** Every function already releases the GIL for
  its native pass, so the async surface is one thread dispatch per call
  ([`tors.aio`](async.md)) rather than a purpose-built event-loop
  integration.
- **A general persistent pipeline object.** `apply_pipeline` re-describes and
  re-applies its steps on every call rather than compiling a reusable
  pipeline handle; see `CompiledLemmaDict` above for that measured
  exception.

## Limitations

- **SimHash is not cryptographic.** It is a fast, uniformly-spreading voting
  hash, not a security primitive: two unrelated documents can coincidentally
  land close together, especially on short text, and there is no universal
  "near-duplicate" distance threshold; calibrate per deployment.
- **`soundex`/`metaphone` are English/Latin-script-oriented.** Both
  pre-filter input to ASCII letters; accented and non-Latin characters are
  dropped, not encoded.
- **Chunking makes no retrieval-quality promise.** Every chunker guarantees a
  mechanical contract (correct boundaries, genuine overlap when requested);
  none of them promises a particular chunk size or strategy helps any
  downstream model.
- **Segmentation is rule-based UAX #29 only.** No dictionary segmentation for
  spaceless scripts (Thai, Khmer, Burmese, Japanese word breaks are a
  different, dictionary-based problem).
- **`tf_idf`/`bm25_rank` make no relevance claim.** Both are correctly
  implemented, well-specified ranking formulas; neither promises retrieval
  quality for any particular corpus or query.
