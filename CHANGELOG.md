# Changelog

## [0.6.1](https://github.com/AZX-PBC-OSS/tors/compare/v0.6.0...v0.6.1) (2026-09-10)


### Bug Fixes

* clear the lookahead memo at the empty-object splice too ([#39](https://github.com/AZX-PBC-OSS/tors/issues/39)) ([#42](https://github.com/AZX-PBC-OSS/tors/issues/42)) ([f735874](https://github.com/AZX-PBC-OSS/tors/commit/f735874b3ce00d026072086faba9f5576c83e861))
* publish tors-documents wheels from tors-documents/dist, not the empty root dist ([#46](https://github.com/AZX-PBC-OSS/tors/issues/46)) ([e697686](https://github.com/AZX-PBC-OSS/tors/commit/e697686b34adf9fd108ed7349d94d2b8ebe830cb))
* redact internal project codename from comment and docs ([#41](https://github.com/AZX-PBC-OSS/tors/issues/41)) ([18d2036](https://github.com/AZX-PBC-OSS/tors/commit/18d2036b3da9af73e4e1b71dd451053bba7080f8))
* score is_grounded's truncated tail window against the claim-length denominator ([#44](https://github.com/AZX-PBC-OSS/tors/issues/44)) ([27e584c](https://github.com/AZX-PBC-OSS/tors/commit/27e584cdd8d693d71b5bbf97a0717439830837bd))
* **tests:** run the stdlib-b64 red side inline on the loop, not in a thread ([#45](https://github.com/AZX-PBC-OSS/tors/issues/45)) ([cf3cd72](https://github.com/AZX-PBC-OSS/tors/commit/cf3cd726d5283ce0ce5da96c8b630e43dc1b37e4))

## [0.6.0](https://github.com/AZX-PBC-OSS/tors/compare/v0.5.0...v0.6.0) (2026-09-10)


### ⚠ BREAKING CHANGES

* chunk_hierarchical's Rust signature separators is now Option<&[Option<&str>]> (landed in #28; restated here because #28's squash carried no footer and release-please would not surface it). Python callers are unaffected: all-literal lists behave identically.

### Features

* add chunk_by_lines and None-entry hierarchy splicing to chunk_hierarchical ([#28](https://github.com/AZX-PBC-OSS/tors/issues/28)) ([b5a7d66](https://github.com/AZX-PBC-OSS/tors/commit/b5a7d6678fb0bc9d14d1a8d349a3659a5202570d))
* document-format extraction as tors.documents (second wheel, GIL-free) ([#27](https://github.com/AZX-PBC-OSS/tors/issues/27)) ([eade003](https://github.com/AZX-PBC-OSS/tors/commit/eade003156bc705a3aa29b04b59c7cf7d44f1146))
* memchr line/paragraph scans, lazy hierarchy levels, chunk_by_paragraphs_iter, iter error precedence ([#30](https://github.com/AZX-PBC-OSS/tors/issues/30)) ([#34](https://github.com/AZX-PBC-OSS/tors/issues/34)) ([48e91bc](https://github.com/AZX-PBC-OSS/tors/commit/48e91bcd94967efcb625de1d17cf35ed9b3656bf))


### Bug Fixes

* dedup None splices, single-pass the line/paragraph scans, pin [#28](https://github.com/AZX-PBC-OSS/tors/issues/28)'s contracts ([#31](https://github.com/AZX-PBC-OSS/tors/issues/31)) ([3fd4291](https://github.com/AZX-PBC-OSS/tors/commit/3fd42915e152ed46ccd8b334013f93e1c2e2f923))


### Performance Improvements

* build levels on first consultation, memchr2 the ASCII line/paragraph scans ([#32](https://github.com/AZX-PBC-OSS/tors/issues/32)) ([e955e9b](https://github.com/AZX-PBC-OSS/tors/commit/e955e9b37fe254b37fe819983e900b11a2915a9a))


### Documentation

* improve documentation clarity ([#35](https://github.com/AZX-PBC-OSS/tors/issues/35)) ([aa4ebe8](https://github.com/AZX-PBC-OSS/tors/commit/aa4ebe8dd4ae5a305272597389a016f1662d490e))

## [0.5.0](https://github.com/AZX-PBC-OSS/tors/compare/v0.4.1...v0.5.0) (2026-09-09)


### Features

* **repair:** add deadline_ms to bound pathological O(n²) inputs ([#21](https://github.com/AZX-PBC-OSS/tors/issues/21)) ([09a97b3](https://github.com/AZX-PBC-OSS/tors/commit/09a97b341d03cffd1a6ad6edfb5fd0aeb2b8ceb0))


### Bug Fixes

* is_grounded(fuzzy=True) rejects a verbatim substring ([#17](https://github.com/AZX-PBC-OSS/tors/issues/17)) ([1cce6d2](https://github.com/AZX-PBC-OSS/tors/commit/1cce6d28c520a0a60158e00c3532e3fc8ee1c2bc))
* repair_json O(n²) on array-close runs (part A of [#13](https://github.com/AZX-PBC-OSS/tors/issues/13)) ([#18](https://github.com/AZX-PBC-OSS/tors/issues/18)) ([c15b493](https://github.com/AZX-PBC-OSS/tors/commit/c15b4936f9a1590838945ae60bea9f3a9b5beaa5))
* repair_json SIGSEGV on comma-joined object fragments ([#19](https://github.com/AZX-PBC-OSS/tors/issues/19)) ([d9e1845](https://github.com/AZX-PBC-OSS/tors/commit/d9e184581d4e9a76361db8bf9e04d9ef846adec8))
* restore [#21](https://github.com/AZX-PBC-OSS/tors/issues/21)'s deadline_ms, silently reverted by [#24](https://github.com/AZX-PBC-OSS/tors/issues/24)'s stale-base squash ([#25](https://github.com/AZX-PBC-OSS/tors/issues/25)) ([ec494c5](https://github.com/AZX-PBC-OSS/tors/commit/ec494c5a30930c4b723c2f89ed7fd9fe25b6c8db))


### Performance Improvements

* replace the chunkers' unconditional per-char grapheme structures with a lazy bitmap ([#24](https://github.com/AZX-PBC-OSS/tors/issues/24)) ([36d4016](https://github.com/AZX-PBC-OSS/tors/commit/36d4016bbea24f9f53f688140faa42ac4eb28688))

## [0.4.1](https://github.com/AZX-PBC-OSS/tors/compare/v0.4.0...v0.4.1) (2026-09-08)


### Documentation

* under schema= the ""-nothing-recoverable sentinel is validated, not escaped ([69946f1](https://github.com/AZX-PBC-OSS/tors/commit/69946f19538fa198e597f382d20ec3025178df68))

## [0.4.0](https://github.com/AZX-PBC-OSS/tors/compare/v0.3.1...v0.4.0) (2026-09-08)


### Features

* port json_repair 0.63.4 — schema-guided LLM JSON repair, GIL-free ([626c589](https://github.com/AZX-PBC-OSS/tors/commit/626c589c15f1263e604a35902e07b3052d3f4971))
* port json_repair 0.63.4 — schema-guided LLM JSON repair, GIL-free ([6d2ddcd](https://github.com/AZX-PBC-OSS/tors/commit/6d2ddcdb83ad76cd434d3b110bc1aa342a62e3f3))

## [0.3.1](https://github.com/AZX-PBC-OSS/tors/compare/v0.3.0...v0.3.1) (2026-09-08)


### Bug Fixes

* keep release tags shippable despite lockfile drift ([b0bc294](https://github.com/AZX-PBC-OSS/tors/commit/b0bc294e5f663a81ad1bb817ab04a700dd9e9c3e))

## [0.3.0](https://github.com/AZX-PBC-OSS/tors/compare/v0.2.0...v0.3.0) (2026-09-08)


### Features

* add truncate_ellipsis and strip_controls, extend tors.aio to input-scaling codecs ([9d4d824](https://github.com/AZX-PBC-OSS/tors/commit/9d4d824f03c761f86d7782f6ceeea7d7942e5a35))


### Documentation

* update install instructions now that tors/tors-core are published ([9a32133](https://github.com/AZX-PBC-OSS/tors/commit/9a32133a7d50c5099070648b1347ae11e452fbfe))

## [0.2.0](https://github.com/AZX-PBC-OSS/tors/compare/v0.1.0...v0.2.0) (2026-09-07)


### Features

* abi3 wheels, PyPI trusted publishing, release-please, dependabot ([3b6af74](https://github.com/AZX-PBC-OSS/tors/commit/3b6af744152a16d3a0c360594ff544087b016758))
* more phonetic algorithms, WASM feasibility CI, weekly deep-fuzz job, close_matches prefilter ([d159155](https://github.com/AZX-PBC-OSS/tors/commit/d1591556065c7234363f9c0ec8868c1c152197af))
* publish tors-core to crates.io alongside PyPI, harden CI permissions ([394b067](https://github.com/AZX-PBC-OSS/tors/commit/394b0679d13d2e496b3968bd3ad62f9160159afc))
* v0 - single-pass GIL-free text normalization ([124ef5d](https://github.com/AZX-PBC-OSS/tors/commit/124ef5d391fa614941dda19a38c21ae349090cd6))
* v0.1.0 feature set — text normalization, segmentation, diffing, fuzzy/phonetic matching, retrieval, chunking, and integrity primitives ([7fb43db](https://github.com/AZX-PBC-OSS/tors/commit/7fb43db1e8b6e8f563b62a65ec9c877df3b3fabd))


### Bug Fixes

* **ci:** astral-sh/setup-uv has no floating major tag, pin full v10.0.1 ([ca2069c](https://github.com/AZX-PBC-OSS/tors/commit/ca2069c3a8ecd00ef4fe8a91b015b98bf30cecc4))
* close deadline-bypass DoS gap, cut chunk/truncate to one pass, DRY the py layer ([134c07c](https://github.com/AZX-PBC-OSS/tors/commit/134c07c869114e5e0224c516315ce363484c6e7b))
* **dedent,tokenize:** adopt CPython 3.14 dedent rule, drop empty terms ([a9030dd](https://github.com/AZX-PBC-OSS/tors/commit/a9030dd874c38799bacaa14cc06b58a09b5c63e0))
* **release-please:** drop package-name so tags stay plain vX.Y.Z ([212739b](https://github.com/AZX-PBC-OSS/tors/commit/212739b3af36726a6c5f07ca2b760ac9a9cf327b))


### Documentation

* cover sync/async on every tors.aio-wrapped function, fix stale fuzz-target list ([a676062](https://github.com/AZX-PBC-OSS/tors/commit/a6760628ebde436d1701f8e8daef00da60a629af))
* mkdocs site + GitHub Pages deploy, uv.lock discipline in CI ([4cf8ad0](https://github.com/AZX-PBC-OSS/tors/commit/4cf8ad0f4420002dfc3b5e58bfed06a101cfc23c))
