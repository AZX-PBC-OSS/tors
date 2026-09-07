# tors

Fast, GIL-free text and document operations for Python, backed by Rust: normalization,
Unicode segmentation, diffing, fuzzy and phonetic matching, multi-pattern search and
redaction, chunking, and lightweight retrieval (TF-IDF, BM25, SimHash, Merkle integrity),
in the spirit of `orjson` for JSON or `polars` for dataframes.

## Why

Python's `re` module and `str` methods never release the GIL, no matter how large the
input is: a multi-megabyte text transform runs as one long GIL-held call that stalls
every other thread and the asyncio event loop for its whole duration. `tors` does the
same kind of transform as a single native Rust pass, wrapped in `py.detach` (PyO3's
GIL-release call) for the entire computation, so the GIL is free for the rest of your
program while it runs.

That covers the functions with a stdlib equivalent (`normalize`'s pipeline mirrors
`unicodedata.normalize` + a few `re.sub` calls; `quote`/`unquote` mirror `urllib.parse`;
`b64_encode_bytes`/`b64_decode` mirror `base64`). Where the stdlib has no equivalent
at all (Unicode text segmentation: grapheme clusters, word and sentence boundaries;
leftmost-longest multi-pattern search; edit-distance and phonetic matching; content-
defined chunking; SimHash near-duplicate detection; Merkle tree integrity; and encoding
detection), `tors` supplies it over maintained Rust crates, GIL-released the same way,
so async services and threaded pipelines don't pay a blocking tax for text work either
way.

## Install

Not yet published to PyPI. For now, build from source:

```sh
pip install maturin
maturin develop --release
```

Once published: `pip install tors`.

### Pyodide / WebAssembly

The Rust cores are OS-free, so `tors` also builds for
[Pyodide](https://pyodide.org) (CPython compiled to WebAssembly) as a
PEP 783 `pyemscripten` wheel — the abi3-py310 story carries over unchanged,
one wheel covers every Pyodide Python ≥ 3.10. The `WASM` workflow builds it
on every push (artifact only; publishing to PyPI's Emscripten platform is not
wired up). To build one locally:

```sh
rustup target add wasm32-unknown-emscripten
uvx --python 3.14 --from pyodide-build pyodide build . -o dist
```

The driving interpreter matters: Python 3.14 targets Pyodide 314.x /
`pyemscripten_2026_0` with a stable Rust toolchain; driving from 3.13 pins a
Rust nightly older than this crate's MSRV. SIMD-dependent dependencies
(base64, simdutf8, memchr, aho-corasick) compile their scalar fallbacks for
wasm; the extension has been smoke-tested (import + representative calls
across every core family) in Pyodide under node.

## What's in it

61 functions plus one small helper class, grouped by what they do. Each entry is a
one-line description; full signatures, argument contracts, and edge cases are in the
[API reference](docs/api.md).

**Unicode normalization & forms**: clean up messy extracted text, or apply a single
normalization form directly.
- `normalize`: NFC + CRLF folding + blank-line collapsing + strip, the common
  PDF/OCR-extraction cleanup pipeline
- `finalize`: `normalize` plus a SHA-256 of the result, in one pass
- `nfc` / `nfd` / `nfkc` / `nfkd`: the four Unicode normalization forms standalone
- `html_unescape`: `html.unescape`, full HTML5 entity table

**UTF-8 / UTF-16 / base64 codecs**: validate and decode bytes without holding the GIL
for the whole buffer.
- `decode_utf8` / `finalize_utf8`: UTF-8 decode, and decode+normalize+hash fused
- `utf8_is_valid`: SIMD UTF-8 validity check, no `str` materialized, no exception flow
- `decode_utf16` / `utf16_is_valid`: the same pair for UTF-16, with BOM sniffing
- `b64_encode_bytes` / `b64_decode`: RFC 4648 base64, byte-exact `base64` module parity
- `detect_encoding`: heuristic legacy-encoding guesser (`chardetng`, Firefox's detector)

**Text segmentation**: Unicode-correct boundaries the stdlib has no segmenter for at
all.
- `grapheme_count`: extended grapheme cluster count (UAX #29)
- `word_bounds` / `word_bounds_iter` / `word_count`: word boundaries, list, iterator,
  and count forms
- `sentence_bounds` / `sentence_bounds_iter` / `sentence_count`: sentence boundaries,
  same three forms

**Diffing**: `difflib`-compatible opcodes at native speed.
- `diff_opcodes`: character-level `SequenceMatcher.get_opcodes()` shape
- `diff_opcodes_lines`: the line-level spelling for document/version diffs

**Fuzzy string matching & phonetic matching**: edit-distance and sound-alike matching,
none of which the stdlib ships.
- `similarity_ratio` / `get_close_matches`: `difflib`'s ratio and closest-match search
- `levenshtein` / `jaro` / `jaro_winkler`: classic edit-distance and similarity metrics
- `soundex` / `metaphone` / `double_metaphone` / `nysiis` / `daitch_mokotoff` /
  `refined_soundex`: classic English/Latin-script phonetic codes, five distinct
  mapping tables for the same name-matching/dedup lane

**Multi-pattern search & redaction**: leftmost-longest search and simultaneous replace,
a combination no stdlib or maintained GIL-free binding offers.
- `find_patterns` / `find_patterns_iter` / `count_matches`: multi-pattern search, list,
  iterator, and count forms
- `replace_many`: simultaneous multi-pattern replace, one pass, no re-scanning
- `replace_many_masked`: the same, length-preserving, for offset-safe redaction

**Markdown / code-fence extraction**: pull structured content out of model output.
- `extract_code_blocks`: every fenced code block, per CommonMark's fence grammar
- `strip_code_fences`: unwrap a whole response wrapped in exactly one fence
- `dedent`: `textwrap.dedent`, byte-exact

**Truncation & lexical grounding**: fit text to a budget, or sanity-check a claim
against its source.
- `truncate_to_bounds`: cut to a character budget at a word/sentence boundary, never
  mid-grapheme
- `is_grounded`: exact or fuzzy substring check of a claim against its source

**URL encoding**: `urllib.parse`'s percent-encoding quartet, GIL-released.
- `quote` / `quote_plus` / `unquote` / `unquote_plus`

**Text chunking**: split text or bytes for embedding, indexing, or context-window
packing.
- `chunk_cdc`: FastCDC content-defined byte chunking (edit-local, for dedup/sync)
- `chunk_text` / `chunk_text_iter`: character-budget chunking, word/sentence-boundary
  aware, optional overlap
- `chunk_by_words` / `chunk_by_words_iter`: fixed word-count chunks
- `chunk_by_sentences` / `chunk_by_sentences_iter`: fixed sentence-count chunks
- `chunk_by_paragraphs`: fixed paragraph-count chunks (blank-line heuristic)
- `chunk_hierarchical`: priority-ordered fallback chunking (`RecursiveCharacterTextSplitter`
  pattern)

**Information retrieval**: lexical/statistical primitives for small-corpus search and
integrity, without an embeddings dependency.
- `tf_idf`: stateless TF-IDF term scoring per document
- `bm25_rank`: Okapi BM25 reranking of a corpus against a query
- `simhash64` / `simhash128`: SimHash near-duplicate fingerprints
- `merkle_root` / `merkle_diff`: domain-separated Merkle root and per-index chunk diff

**Text-processing pipelines**: batch preprocessing without a stateful pipeline object.
- `apply_pipeline`: fused NFD/lowercase/accent-fold/stem/lemma/whitespace-collapse pass
  over a whole text list
- `CompiledLemmaDict`: a build-once handle for a large `lemma_dict`, reused across calls

## Examples

```python
import tors

tors.normalize("line one  \n\n\n\nline two\r\n")
# 'line one\n\nline two'

tors.finalize("line one  \n\n\n\nline two\r\n")
# ('line one\n\nline two', 'e986ba083c7c1a9361143d2d8ccd8477d1d5eeef8b94b67c6ad4693f8f7b942a')
```

```python
text = (
    "This is sentence one. This is sentence two. "
    "This is sentence three. This is sentence four."
)
chunks = tors.chunk_by_sentences(text, 2)
# [(0, 44), (44, 90)]

[text[s:e] for s, e in chunks]
# ['This is sentence one. This is sentence two. ', 'This is sentence three. This is sentence four.']
```

```python
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "a lazy cat sleeps all day",
    "the fox and the dog are friends",
]
tors.bm25_rank("quick fox", corpus)
# [(0, 1.3162195220480066), (2, 0.4798180901812613), (1, 0.0)]
```

All three run against the built extension; the output above is what they actually
return. For every function's full argument contract, error behavior, and more examples,
see:

- [`docs/api.md`](docs/api.md): the full API reference
- [`docs/recipe-ingest.md`](docs/recipe-ingest.md): bytes to clean, chunked text, end
  to end
- [`docs/recipe-retrieval.md`](docs/recipe-retrieval.md): a small-corpus,
  no-embeddings search pipeline

## Design philosophy and non-goals

`tors` is a stateless library: every call does its own work from scratch, with nothing
cached or built up across calls. That keeps every function simple to reason about and
safe to call from anywhere: no handle to manage, no invalidation to think about, no
surprise from a stale cache.

`CompiledLemmaDict` is the one narrow exception, and it's worth being
precise about why. `tf_idf`, `bm25_rank`, and `apply_pipeline` accept an optional
caller-supplied `lemma_dict`: a `word -> lemma` map. A raw `dict[str, str]` is
re-materialized into a Rust `HashMap` on every call, and for a realistic multi-thousand-
entry lemma table that cost is measured, not theoretical: roughly 1.4ms per call on a
20,000-entry map, independent of how much text the call actually processes, which can
make repeated small calls slower than the equivalent pure-Python loop.
`CompiledLemmaDict` builds that `HashMap` once and hands back an immutable, cheaply
cloned handle, the same `re.compile()` shape the stdlib already uses for a comparable
problem. It exists for exactly this one measured cost and does not reopen the case for
a general pipeline object: nothing else in this library gets a persistent handle.

The scope cuts below are decisions, not oversights:

- **General regex.** `find_patterns`/`replace_many` are leftmost-longest multi-pattern
  literal search, not a regex engine; `chunk_hierarchical`'s `separators` are literal
  strings, not patterns.
- **Schema-aware JSON/YAML coercion.** Out of scope for the same reason lemmatization
  is: it needs a schema or model, not an algorithm.
- **A bundled lemma dictionary.** `apply_pipeline`/`tf_idf`/`bm25_rank` apply a
  caller-supplied lemma map; `tors` ships no lemma data of its own, because full
  lemmatization needs a per-language dataset or a POS-tagging model, not an algorithm;
  that is outside a text-operations library's job.
- **A persistent search index.** `bm25_rank` recomputes corpus statistics from scratch
  on every call: the right shape for reranking a small, already-retrieved candidate
  set, the wrong shape for querying a large corpus repeatedly. Reach for a real search
  engine (`tantivy`, in Rust) for that; `tors` does not build or expose index objects.
- **A bespoke coroutine API.** Every function already releases the GIL for its native
  pass, so the async surface is one thread dispatch per call (`tors.aio`, below)
  rather than a purpose-built event-loop integration.
- **A general persistent pipeline object.** `apply_pipeline` re-describes and re-applies
  its steps on every call rather than compiling a reusable pipeline handle: see
  `CompiledLemmaDict` above for the one measured exception.

Beyond scope, a few limitations are worth stating plainly rather than glossing over:

- **SimHash is not cryptographic.** It's a fast, uniformly-spreading voting hash, not a
  security primitive: two unrelated documents can coincidentally land close together,
  especially on short text, and there's no universal "near-duplicate" distance
  threshold; calibrate per deployment.
- **`soundex`/`metaphone` are English/Latin-script-oriented.** Both pre-filter input to
  ASCII letters; accented and non-Latin characters are dropped, not encoded.
- **Chunking makes no retrieval-quality promise.** Every chunker guarantees a mechanical
  contract (correct boundaries, genuine overlap when requested); none of them promises a
  particular chunk size or strategy helps any downstream model.
- **Segmentation is rule-based UAX #29 only.** No dictionary segmentation for spaceless
  scripts (Thai, Khmer, Burmese, Japanese word breaks are a different, dictionary-based
  problem).
- **`tf_idf`/`bm25_rank` make no relevance claim.** Both are correctly implemented,
  well-specified ranking formulas; neither promises retrieval quality for any
  particular corpus or query.

## Async use

Releasing the GIL is not the same as not blocking: a native pass called directly from a
coroutine still occupies that coroutine's own turn on the event loop for the call's full
wall-clock duration. `tors.aio` is the pre-wired fix for the functions where that matters:
`await tors.aio.tf_idf(corpus)` runs the native pass in a worker thread via
`asyncio.to_thread`, and the event loop stays responsive for its whole duration.

It covers only the large-input-shaped functions (the chunking family, `tf_idf`,
`bm25_rank`, `diff_opcodes`, `diff_opcodes_lines`, and `apply_pipeline`), not all of
`tors`. Thread dispatch costs on the order of tens of microseconds: noise next to a
millisecond-or-slower native pass over a real corpus or document, real overhead next to a
microsecond-scale call over a short string. Wrapping every export would make the small,
common calls slower through this module than through the plain sync spelling, for no
benefit, so the rest of `tors` keeps exactly one spelling: call it directly from a
coroutine when the input is small enough that the whole thing finishes in microseconds.

There is no size-based branch inside any wrapper, and there never will be: a function
that sometimes runs inline and sometimes hops to a thread depending on its input is
unpredictable from the caller's side and can silently block the loop when the heuristic
misjudges. Every function in `tors.aio` always dispatches through `asyncio.to_thread`,
unconditionally; the choice between the sync spelling and `tors.aio` is the caller's,
made once at the call site, not a runtime guess. `tests/test_aio.py` pins this
structurally (no branch in the wrapper body) as well as behaviorally (a heartbeat
coroutine keeps ticking with worst gaps well under the call's own wall during a large
`diff_opcodes` await).

The streaming iterator constructors (`word_bounds_iter` and siblings, including the
chunking family's own `chunk_text_iter`/`chunk_by_words_iter`/`chunk_by_sentences_iter`)
have no async twin: an iterator is not an awaitable shape, and draining one to a list
inside a worker thread is exactly what the already-covered list-returning sibling does.
The eager construction pass is the GIL-released part anyway, so the manual
`await asyncio.to_thread(lambda: list(tors.word_bounds_iter(text)))` covers the
streaming shape when it is genuinely needed. Signatures are identical to the sync
spellings, pinned by `tests/test_aio.py`; the stub `aio.pyi` is generated by
`tools/gen_aio_stub.py`.

## Performance

Full measured tables (GIL heartbeat gaps under `asyncio`, wall-time races against the
stdlib and `difflib`, and criterion throughput) live in the test suite itself
(`tests/test_gil_release.py`, `tests/test_performance.py`) so every number stays
reproducible and re-runnable: the summary here gives the shape, and the test files
are the ledger.

**The GIL release is the headline.** Running `tors.finalize` on 12 MiB of prose in a
background thread holds the asyncio event loop's worst heartbeat gap to 10–14 ms; the
equivalent pure-Python pipeline (`unicodedata.normalize` + `str.replace` + `re.sub` +
`hashlib.sha256`) holds it for 92–108 ms in the same thread placement: roughly 6–12×
worse, and it blows past this project's own CI budget in every sample. At 32 MiB the gap
widens further (17–21 ms vs 250–272 ms).

**Already-normalized input is close to free.** A quick-check fast path means
`normalize`/`finalize`/`nfc`/`nfkd` on already-clean 12 MiB text return the original
object with only a SIMD sentinel scan: 2.4–7.7 ms where a full pass costs 100+ ms
(28–45× faster).

**Where no stdlib equivalent exists, the comparison is against the real alternative.**
`diff_opcodes` diffs 256 KiB of mutated prose in ~4 ms against `difflib`'s ~3 seconds
(~750×); `get_close_matches` against 13,900 candidates runs in ~65–72 ms against
`difflib`'s ~3.4 s (~50×); `find_patterns` fills the gap left by `pyahocorasick`, which
holds the GIL for its entire scan by design (no `ALLOW_THREADS` anywhere in its scan
iterator); `utf8_is_valid` has no stdlib boolean primitive to race at all, so its record
is absolute throughput: up to ~95 GiB/s at 12 MiB, memory-bound above L3 cache.

**List-returning functions have a real, disclosed cost at scale.** `word_bounds` on
12 MiB of prose (3.67M segments) holds the GIL for 428–497 ms just marshalling the
returned list: a genuine cost of the list shape, not a bug. The `_iter` twins
(`word_bounds_iter`, `chunk_text_iter`, `find_patterns_iter`, and friends) exist for
exactly this: the same sequence, streamed, with each `__next__` holding the GIL for one
tuple instead of the whole list at once, and 2.1× faster in wall time as well, in the
measured case.

Every number above traces to a specific measured cell; see the linked test files for
methodology, corpus construction, and the full per-function tables.

## Dependencies and licensing

The license gate is `cargo deny check licenses advisories bans` (`make deny`; the same
three checks run in CI's lint job). The allowlist in `deny.toml` is permissive-only:
MIT, Apache-2.0, BSD-2/3-Clause, ISC, Zlib, plus three documented additions, all
permissive grants inside the gate's intent:
`Apache-2.0 WITH LLVM-exception` (target-lexicon, a pyo3 build dependency: the
LLVM-exception *removes* attribution obligations from Apache-2.0), `Unicode-3.0`
(unicode-ident, the permissive license the Unicode Consortium publishes the UCD data
under), and `0BSD` (enum-iterator/enum-iterator-derive, `soundex`/`metaphone`'s
`rphonetic` dependency's own dependencies: the BSD Zero Clause License is
OSI-approved and public-domain-equivalent, strictly more permissive than plain MIT).
No GPL/LGPL/AGPL/MPL, no unlicensed. Dual/tri-licensed crates are consumed via
an allowed branch: notably r-efi (a getrandom dependency, dev tree only) offers
`LGPL-2.1-or-later` as one branch of `MIT OR Apache-2.0 OR LGPL-2.1-or-later`; tors
consumes it under MIT/Apache and the LGPL branch is never elected, which is exactly
what cargo-deny's SPDX expression evaluation verifies.

Direct dependencies (the full transitive closure is machine-checked by the gate; the
dev tree, criterion and friends, is included in the check):

| crate | version | license | role |
|---|---|---|---|
| pyo3 | 0.29.2 | MIT OR Apache-2.0 | the CPython extension layer (abi3-py310) |
| unicode-normalization | 0.1.25 | MIT OR Apache-2.0 | NFC/NFD/NFKC/NFKD tables (Unicode 16.0.0) |
| unicode-segmentation | 1.13.3 | MIT OR Apache-2.0 | UAX #29 grapheme/word tables (Unicode 17.0.0) |
| sha2 | 0.10.9 | MIT OR Apache-2.0 | finalize's SHA-256 |
| const-hex | 1.19.1 | MIT OR Apache-2.0 | digest hex encoding |
| base64 | 0.23.1 | MIT OR Apache-2.0 | RFC 4648 encode/decode core, `simd-unsafe` feature enabled (the crate's own AVX2/NEON kernels, runtime-detected with a scalar fallback: already-shipped, widely-exercised unsafe code upstream, not written in tors) |
| memchr | 2.8.3 | Unlicense OR MIT | SIMD sentinel scans |
| simdutf8 | 0.1.5 | MIT OR Apache-2.0 | SIMD UTF-8 validity scan |
| similar | 3.2.0 | Apache-2.0 | Myers diff engine for diff_opcodes |
| aho-corasick | 1.1.5 | Unlicense OR MIT | leftmost-longest multi-pattern search engine |
| rs_merkle | 1.5.0 | Apache-2.0 OR MIT | merkle_root/merkle_diff's tree structure (domain-separated SHA-256 Hasher supplied by tors, see merkle_impl.rs) |
| fastcdc | 5.0.0 | MIT | chunk_cdc's FastCDC 2020 content-defined chunking |
| chardetng | 1.0.0 | Apache-2.0 OR MIT | detect_encoding's guesser (the engine Firefox ships) |
| encoding_rs | 0.8.35 | (Apache-2.0 OR MIT) AND BSD-3-Clause | the `Encoding` type chardetng's guess returns: already pulled in transitively by chardetng; named directly only to call `.name()` on it, no new package in the tree (the BSD-3-Clause conjunct is the WHATWG Encoding Standard data files' grant, inside the gate's BSD-3 allowance) |
| rust-stemmers | 1.2.0 | MIT OR BSD-3-Clause | tf_idf's/bm25_rank's opt-in Snowball stemmer= knob (18 languages): pulls in serde/serde_derive as a non-optional dependency (an `Algorithm` enum derive, unused by tors's own call sites); recorded here because it is the one real transitive-weight addition in this table |
| rphonetic | 4.0.0 | Apache-2.0 | soundex/metaphone's phonetic-code algorithms (an Apache Commons Codec port): pulls in enum-iterator/enum-iterator-derive (0BSD, the license-gate addition noted above), nom, and thiserror; tors pre-filters every input to ASCII letters before calling into it, working around a real, verified panic in the crate's own Soundex/DoubleMetaphone encoders on ordinary accented input (see the API docs) |
| criterion *(dev)* | 0.5.1 | Apache-2.0 OR MIT | the benchmark harness |
| strsim *(dev)* | 0.11.1 | MIT | differential oracle for levenshtein/jaro/jaro_winkler tests |

Maintenance note (the spec's module decision): `simdutf8`'s
release line has been quiet since 2024-09-22 (0.1.5) while the repository
itself stays active (commits into 2026-06): a mature, zero-dependency
implementation of an algorithm that does not churn (UTF-8 validation), picked
with that fact known and disclosed.

Maintenance note on `similar` (the same decision): Apache-2.0 only, actively
maintained by mitsuhiko (the Flask author), and the diff engine behind insta,
a crate with a large existing consumer base; no Python binding for it exists, so
tors binds it directly for `diff_opcodes`.

Maintenance note on `aho-corasick` (the same decision): dual `Unlicense OR
MIT`: the MIT branch is elected and recorded here (cargo-deny's SPDX
expression evaluation verifies exactly that election, the same mechanism that
handles memchr's `Unlicense OR MIT`, the same spelling). By BurntSushi, and
the Aho-Corasick engine inside Rust's own `regex` crate: the most
battle-tested implementation of this exact algorithm in the Rust ecosystem;
it was already in the lock as a transitive dependency (criterion's regex) before
find_patterns made it direct, so the dependency tree grew by zero packages.

Transitive closure at the last lock-state count (120 `Cargo.lock` entries
including tors itself, i.e. 119 dependency packages incl. dev, re-derived with
`cargo metadata` over the current lock): 74 `MIT OR Apache-2.0`, 12 MIT
(fastcdc and strsim among them), 6 `Apache-2.0 OR MIT`, 5 Apache-2.0
(rphonetic, soundex/metaphone's crate, among them), 3
`MIT/Apache-2.0` (criterion-plot, itertools, version_check), 3
`Unlicense OR MIT` (aho-corasick, memchr, winapi-util), 2 `0BSD`
(enum-iterator/enum-iterator-derive, rphonetic's own dependencies: the
license-gate addition above), 2
`Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` (wasip2, wit-bindgen),
2 `BSD-2-Clause OR Apache-2.0 OR MIT` (zerocopy), 2 `Unlicense/MIT`
(same-file, walkdir), 1 `(Apache-2.0 OR MIT) AND BSD-3-Clause` (encoding_rs,
a direct dependency), 1 `Zlib OR Apache-2.0 OR MIT` (tinyvec), 1
`MIT OR Apache-2.0 OR Zlib` (tinyvec_macros), 1
`Apache-2.0 WITH LLVM-exception` (target-lexicon), 1
`(MIT OR Apache-2.0) AND Unicode-3.0` (unicode-ident), 1
`MIT OR Apache-2.0 OR LGPL-2.1-or-later` (r-efi, the tri-license noted above),
1 `Apache-2.0/MIT` (rs_merkle: the closure's third spelling of a dual
grant), and 1 `MIT/BSD-3-Clause` (rust-stemmers, a fourth spelling of the
same dual-grant idea). Every one satisfies the allowlist.

## Development

The `fuzz/` crate drives the `*_impl.rs` cores with raw adversarial bytes via
cargo-fuzz (libFuzzer): a bug class the hypothesis-based Python tests cannot
reach, since they shape input around documented contracts rather than raw
bytes. `cargo fuzz run <target> -- -max_total_time=30` runs one target
briefly (nightly toolchain and `cargo install cargo-fuzz --locked`
required); targets assert the same invariants the Python gates pin, at
raw-byte depth; crashes are minimized with `cargo fuzz tmin`. `make
fuzz-quick` runs every target for 30s each; the weekly `fuzz` workflow runs
the same set in CI. See
[CONTRIBUTING.md](CONTRIBUTING.md#fuzzing) for the full setup.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, the pre-PR checklist (tests, clippy,
fmt, ruff, the license gate), and commit-message conventions.
