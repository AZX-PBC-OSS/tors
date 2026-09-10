# Dependencies and licensing

The license gate is `cargo deny check licenses advisories bans` (`make deny`;
the same three checks run in CI's lint job). The allowlist in `deny.toml` is
permissive-only: MIT, Apache-2.0, BSD-2/3-Clause, ISC, Zlib, plus four
documented additions, all permissive grants inside the gate's intent:

- `Apache-2.0 WITH LLVM-exception` (target-lexicon, a pyo3 build dependency:
  the LLVM-exception *removes* attribution obligations from Apache-2.0)
- `Unicode-3.0` (unicode-ident, the permissive license the Unicode Consortium
  publishes the UCD data under)
- `0BSD` (enum-iterator/enum-iterator-derive, `soundex`/`metaphone`'s
  `rphonetic` dependency's own dependencies: the BSD Zero Clause License is
  OSI-approved and public-domain-equivalent, strictly more permissive than
  plain MIT)
- `MIT-0` (borrow-or-share, a fluent-uri dependency pulled in by the
  jsonschema crate's `referencing` $ref machinery, the json_repair port's
  validation engine: MIT No Attribution is plain MIT with the attribution
  obligation removed, the same family as 0BSD; recorded in `deny.toml`)

No GPL/LGPL/AGPL/MPL, no unlicensed. Dual/tri-licensed crates are consumed via
an allowed branch: notably r-efi (a getrandom dependency, dev tree only)
offers `LGPL-2.1-or-later` as one branch of `MIT OR Apache-2.0 OR
LGPL-2.1-or-later`; tors consumes it under MIT/Apache and the LGPL branch is
never elected, which is exactly what cargo-deny's SPDX expression evaluation
verifies.

## Direct dependencies

The full transitive closure is machine-checked by the gate; the dev tree
(criterion and friends) is included in the check.

| crate | version | license | role |
|---|---|---|---|
| pyo3 | 0.29.2 | MIT OR Apache-2.0 | the CPython extension layer (abi3-py310) |
| unicode-normalization | 0.1.25 | MIT OR Apache-2.0 | NFC/NFD/NFKC/NFKD tables (Unicode 16.0.0) |
| unicode-segmentation | 1.13.3 | MIT OR Apache-2.0 | UAX #29 grapheme/word tables (Unicode 17.0.0) |
| sha2 | 0.11.0 | MIT OR Apache-2.0 | finalize's SHA-256 |
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
| jsonschema | 0.55 | MIT | json_repair's schema-guided validation engine: the mature Rust JSON Schema validator (drafts 4-2020-12), maintained by a core maintainer of Python's own jsonschema; local `#/...` refs only (default-features off: its remote-$ref resolvers would pull a TLS stack); pulls the num family via its exact-rational multipleOf arithmetic (fraction), the tree's one real transitive-weight addition, the rust-stemmers precedent |
| serde_json | 1 | MIT OR Apache-2.0 | the JSON interchange type the validator works over; already in the tree as criterion's transitive, so the direct edge adds no new package (the aho-corasick/encoding_rs precedent) |
| regex | 1 | MIT OR Apache-2.0 | json_repair's single-number extraction grammars (tier-3 prose/currency/percent tokens and the tier-4 separator readings); already in the tree transitively (aho-corasick/memchr elect it via other consumers), so the direct edge adds no new package |
| jiff | 0.2 | MIT OR Unlicense | json_repair's date/time normalization engine (`format: date`/`date-time`/`time`): calendar + timezone-instant math from the datetime crate the Rust ecosystem's own docs point at (the memchr Unlicense-election precedent); default-features off, std only, no TZDB backend: the accept-list shapes need none |
| pdf_oxide *(optional, `documents`)* | 0.3.78 | MIT OR Apache-2.0 | the documents payload's PDF engine (two-column reading order, link annotations, headings); default features only, and the caret bounds the 0.x line (see Cargo.toml's own comment for the measured rationale) |
| anydoc *(optional, `documents`)* | 0.2.4 | MIT | the payload's office/text engine (doc/docx, xls/xlsx, ppt/pptx, rtf, odt/ods/odp, epub, csv) |
| office_oxide *(optional, `documents`)* | 0.1.10 | MIT OR Apache-2.0 | the payload's caller-selectable `backend="oxide"` lane; already compiled in via pdf_oxide's tree, so the direct edge adds no new package |
| html-to-markdown-rs *(optional, `documents`)* | 3.12 | MIT | the payload's HTML engine; default-features off (its optional HTTP/MCP stack stays out) |
| criterion *(dev)* | 0.8.2 | Apache-2.0 OR MIT | the benchmark harness |
| strsim *(dev)* | 0.11.1 | MIT | differential oracle for levenshtein/jaro/jaro_winkler tests |

## Maintenance notes

- `simdutf8`'s release line has been quiet since 2024 (0.1.5) while the
  repository itself stays active: a mature, zero-dependency implementation of
  an algorithm that does not churn (UTF-8 validation), picked with that fact
  known and disclosed.
- `similar` (the same decision): Apache-2.0 only, actively maintained by
  mitsuhiko (the Flask author), and the diff engine behind insta, a crate
  with a large existing consumer base; no Python binding for it exists, so
  tors binds it directly for `diff_opcodes`.
- `aho-corasick` (the same decision): dual `Unlicense OR MIT`; the MIT branch
  is elected and recorded here (cargo-deny's SPDX expression evaluation
  verifies exactly that election, the same mechanism that handles memchr's
  `Unlicense OR MIT`). By BurntSushi, and the Aho-Corasick engine inside
  Rust's own `regex` crate: the most exercised implementation of this
  exact algorithm in the Rust ecosystem. It was already in the lock as a
  transitive dependency (criterion's regex) before find_patterns made it
  direct, so the dependency tree grew by zero packages.

## Transitive closure

At the current lock state (325 `Cargo.lock` entries including tors-core
itself, i.e. 324 dependency packages incl. dev and the documents engine
tree, re-derived with `cargo metadata --all-features` over the current lock):
176 `MIT OR Apache-2.0`, 60 MIT (fastcdc, strsim, and anydoc among them), 18
`Apache-2.0 OR MIT` (chardetng, autocfg), 12 `MIT/Apache-2.0` (version_check,
winapi, siphasher) plus 2 `Apache-2.0/MIT` (rs_merkle, bytecount) and 1
`Apache-2.0 / MIT` (fnv), three more spellings of the same dual grant, 10
`Unlicense OR MIT` (aho-corasick, memchr, jiff) and 4 `Unlicense/MIT` (csv,
same-file, walkdir), 8 Apache-2.0 (rphonetic, soundex/metaphone's crate,
among them), 3 `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` (wasip2,
wit-bindgen), 3 `Zlib OR Apache-2.0 OR MIT` (tinyvec, bytemuck), 3
`MIT OR Apache-2.0 OR Zlib` (tinyvec_macros, the zune image crates) and 2
`MIT OR Zlib OR Apache-2.0` (miniz_oxide, both versions), 3 Zlib (foldhash,
slotmap, zlib-rs: engine-tree arrivals), 2 `BSD-2-Clause OR Apache-2.0 OR
MIT` (zerocopy), 2 `BSD-3-Clause OR Apache-2.0` (moxcms, pxfm, the engines'
color management), 2 BSD-3-Clause (the brotli alloc pair), 2
`MIT OR Apache-2.0 OR LGPL-2.1-or-later` (r-efi, both major versions, the
tri-license noted above), 2 `0BSD` (enum-iterator/enum-iterator-derive,
rphonetic's own dependencies: the license-gate addition above), and 1 each of
the singles: `0BSD OR MIT OR Apache-2.0` (adler2, a miniz_oxide dependency in
the engine tree), `MIT-0` (borrow-or-share), `Apache-2.0 WITH
LLVM-exception` (target-lexicon), `(MIT OR Apache-2.0) AND Unicode-3.0`
(unicode-ident), `(Apache-2.0 OR MIT) AND BSD-3-Clause` (encoding_rs, a
direct dependency), `CC0-1.0` (tiny-keccak, the documents-tree addition),
`Apache-2.0 OR BSL-1.0` (ryu, the csv crate's float formatter in anydoc's
tree: the Apache-2.0 branch elected), `BSD-3-Clause AND MIT` (brotli) and
`BSD-3-Clause/MIT` (brotli-decompressor), and `MIT/BSD-3-Clause`
(rust-stemmers, a fourth spelling of the same dual-grant idea). Every one
satisfies the allowlist.

The count above is the current lock state, json_repair and the documents
engine tree both in. The json_repair port's four direct dependencies grew the
runtime package closure from 55 to 102 (jsonschema's draft-4-2020-12
validation tree is the addition; serde_json and regex were already in the
lock as transitives, and jiff brings one small crate, so the real addition is
jsonschema's tree). The engine tree is the next delta on top, inside the
count through the all-features gate resolution below. The gate re-checks
every new entry against the allowlist on every run: the MIT-0 license it
flagged on the way in (borrow-or-share) is recorded above and in `deny.toml`.

The four `documents` engines are the same story one feature later:
feature-gated (default OFF), so they are absent from the base wheel's build
and present in the payload's.

## Gate coverage

`make deny` and CI's cargo-deny step run at the repo root and cover the full
engine tree: `deny.toml`'s `[graph] all-features = true` resolves every cargo
feature of the workspace into the checked graph, `documents` included. The
root lock's 325 entries carry
pdf_oxide/anydoc/office_oxide/html-to-markdown-rs and their transitive trees,
resolution is metadata-only (nothing links), and the check passes over all of
them.

The payload-manifest invocation (`cargo deny --manifest-path
tors-documents/Cargo.toml check licenses advisories bans`) also passes at the
current lock state, and it is not a redundant subset check: the payload
resolves its own independent `tors-documents/Cargo.lock` whose package set
carries crates and versions the root lock never resolves (measured:
payload-only names, plus version-skewed pairs), so both lockfiles are gated
(`deny.toml`'s `[graph]` comment records the same).

Two documents-tree entries the engine tree forced into `deny.toml`:
`CC0-1.0` is allowlisted (`tiny-keccak` 2.0.2, public-domain-equivalent,
pulled by html-to-markdown-rs's `compile-time-rng` ahash feature; it is not
in the base graph), and RUSTSEC-2026-0192 (`ttf-parser` unmaintained, an
unconditional transitive of both pdf_oxide's and anydoc's PDF stacks) is
ignored with its reason recorded in `deny.toml`.
