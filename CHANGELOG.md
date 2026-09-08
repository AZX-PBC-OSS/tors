# Changelog

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
