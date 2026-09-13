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

The random-generation family is the doctrine's strongest case, not an
exception: even its entropy source is stateless. The unseeded spelling draws
fresh bytes from the operating system's CSPRNG on every call (no cached
userspace engine, no thread-local stream), which is exactly why it is
fork-safe — a `fork()` child cannot inherit a parent's stream because there
is no stream state to inherit, `secrets`' own semantics. The seeded spelling
is a pure function of (seed, arguments): nothing carries across calls there
either. A faster thread-cached engine (what `uuid_utils`' counter and the
uuid crate's `fast-rng` feature ship) is a stateful design tors deliberately
declines for this family; see the API reference's security contract.

## Scope cuts

The cuts below are decisions, not oversights:

- **General regex.** `find_patterns`/`replace_many` are leftmost-longest
  multi-pattern literal search, not a regex engine; `chunk_hierarchical`'s
  `separators` are literal strings (or `None`, splicing in the default
  hierarchy), not patterns; `first_invalid_charset`'s `first`/`rest` are the
  same principle on the validation side — plain strings of permitted
  codepoints (data, not patterns: no ranges, escapes, or classes), and the
  Unicode-category classes (`\w`) that would need property tables stay out.
  The pinned `CHARSET_*` constants carry that principle one step further:
  the common alphabets (base62, unpadded base64url, hex lower/upper/mixed)
  published once as module constants — lexical data, not N wrapper
  functions around the single engine — and the bundled-lemma-data cut below
  is about per-language datasets, not about naming a 16-codepoint hex
  alphabet. The same cut is why `scrub_log_text`'s rules
  are a closed set of *names* (`pg_detail_lines`, `uri_userinfo`,
  `uri_query_creds`), not patterns: each rule is a call-site regex the
  scrub exists to port (TaskQ's exception-text chain), hand-rolled in Rust
  and pinned byte-identical to it — a caller-supplied pattern language
  would reopen the regex-semantics question this cut closes. New scrubs
  arrive as new named rules with their own pinned contracts
  (`strip_controls` is the family's first member), never as parameters.
  The scrub family (`strip_controls`,
  `scrub_pii`) follows the same charter from the other direction: each is
  a hand-rolled scanner for one pinned grammar (a ported call-site
  contract), never a general pattern surface. `scrub_pii`'s rule set is
  the closed two-name contact set — email addresses and `+`-led phone
  numbers — because that is the contract the adopted telemetry-safety
  module states; other redaction grammars (credential-shaped material,
  national identifiers) are separate follow-up contracts with their own
  pinned sources, not silent extensions of this one, the same way C1
  controls are a follow-up to `strip_controls` rather than a widening of
  it.
- **A general RNG engine surface.** The random-generation family is a
  closed set of named generators — `random_string`, `random_hex`,
  `random_b62`, `random_b64url`, `uuid4`, `uuid7` — over exactly two
  entropy spellings: fresh OS CSPRNG bytes per call (the default, the
  secrets-safe one), or a seed-keyed ChaCha20 stream for reproducible
  tests and fixtures. Every token spelling is length-first (the output
  length IS the argument: "I want a base62 id 22 characters long"), with
  `random_hex`/`random_b62`/`random_b64url` all one char-sampling engine
  over their fixed alphabets. tors does not expose a PRNG handle, a
  stream/next API, or distribution samplers; `seed=` is a fixture tool,
  never an entropy source. The hard parts are maintained crates under the
  dependency policy (rand's `OsRng`, rand_chacha's stream, uuid's field
  builders); what tors owns is the glue — the block-buffered word sampler
  and Lemire's unbiased index draw.
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
- **An LSH banding index.** `minhash_signature` computes one document's
  MinHash signature from scratch and keeps nothing across calls; the
  banding table and candidate store a corpus-scale near-duplicate pipeline
  builds on those signatures are caller state, the same boundary that
  keeps `bm25_rank` index-free. A banding helper (cutting a signature into
  `r`-element bands and hashing them for table keys) is a future
  companion question, not a hidden commitment inside the signature core.
- **A bespoke coroutine API.** Every function already releases the GIL for
  its native pass, so the async surface is one thread dispatch per call
  ([`tors.aio`](async.md)) rather than a purpose-built event-loop
  integration.
- **A general persistent pipeline object.** `apply_pipeline` re-describes and
  re-applies its steps on every call rather than compiling a reusable
  pipeline handle; see `CompiledLemmaDict` above for that measured
  exception.
- **A second canonical-serialization dialect.** `content_hash` emits
  exactly one canonical form: the `json.dumps`-compatible one (`sort_keys`,
  `ensure_ascii`, compact separators), pinned differentially against the
  stdlib itself — not `serde_json`'s or `orjson`'s raw-UTF-8 dialects, and
  deliberately not configurable: a hash surface must be one fixed byte
  format forever, or every previously-computed digest silently changes
  meaning.
- **Streaming hash objects / an open-ended digest registry.** The hashing
  surface (`md5_hex`/`sha1_hex`/`sha256_hex`/`sha512_hex`/
  `hmac_sha256_hex`, each with a raw-digest `_digest` twin) is a closed
  set of named one-shot functions, not a
  `hashlib`-style object API: tors is stateless by charter, and a
  constructor-plus-`update()` object is exactly the persistent-handle
  shape that charter cuts (`hashlib.sha256()` construction is O(1), so
  unlike `CompiledPatterns`/`CompiledLemmaDict` there is no measured
  re-materialization cost to justify one). Callers feeding a stream hash
  chunk digests and combine them (`merkle_root`, or a running HMAC chain);
  for incremental feeding `hashlib`'s object API already exists and is not
  duplicated. The algorithm set is closed for the same reason every
  surface here is: each addition is a permanent compatibility and
  maintenance commitment, and the five algorithms — two output spellings
  each, hex and raw digest bytes, the latter for the base64-signature,
  derivation-chain, and thumbprint/lock-int call sites — cover the
  request-signing and content-check jobs a text pipeline actually has
  (anything keyed to a
  newer digest is a security-primitive decision, not a text-ops one). The
  engines are the maintained RustCrypto crates (`md-5`, `sha1`, `sha2`,
  `hmac`), nothing hand-rolled — a digest implementation is the worst
  kind of code to hand-roll, every line a maintenance burden and a
  silent-corruption risk the maintained crates already carry
  primary-source test vectors against.

## Limitations

- **`md5_hex`/`md5_digest` and `sha1_hex`/`sha1_digest` are not security
  primitives.** Checksum /
  ETag / legacy-interop only: md5 has had practical collisions since 2004
  and sha1 since 2017. The security side of the hashing surface is
  `sha256`/`sha512`/`hmac_sha256`, either spelling.
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
- **Seeded random output is predictable.** `seed=` on the random-generation
  family is a reproducible-fixture tool: the output is a pure function of
  the seed, fully predictable from it, and never safe for secrets, keys, or
  tokens. The unseeded spelling (fresh OS entropy per call, fork-safe) is
  the secrets-safe one; the distinction is the family's whole security
  contract, stated on every surface.
- **`uuid7` is probabilistically unique, not monotonic.** 48 Unix
  milliseconds + 74 random bits per call: same-millisecond calls order by
  their random bits, and a backwards clock step flows straight into the
  timestamp. `uuid_utils`' strictly-monotonic counter (process-local
  state, no syscall per call) is a different product promise, not a
  missing feature.
