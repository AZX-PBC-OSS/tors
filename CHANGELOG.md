# Changelog

## [0.8.0](https://github.com/AZX-PBC-OSS/tors/compare/v0.7.0...v0.8.0) (2026-09-19)


### Features

* **aio:** word_bounds/sentence_bounds async twins, and the docs' truth corrections ([cc26514](https://github.com/AZX-PBC-OSS/tors/commit/cc265148ad73058d79b4f07a138fbd55b5688d9f))


### Performance Improvements

* **pii, search, repair:** bucketed key families, id-indexed masked splice, integer address probes ([#109](https://github.com/AZX-PBC-OSS/tors/issues/109)) ([431eda8](https://github.com/AZX-PBC-OSS/tors/commit/431eda8b1633abaead121b1c257baa17262c49e7))

## [0.7.0](https://github.com/AZX-PBC-OSS/tors/compare/v0.6.1...v0.7.0) (2026-09-17)


### Features

* add the uuid7 helper trio (uuid7_timestamp_ms, uuid_version, uuid_parse) ([e99f10a](https://github.com/AZX-PBC-OSS/tors/commit/e99f10a3f720897f803164c5f88367dcf4f5e491))
* **canon:** criterion bench at an object-size ladder ([e8f52fc](https://github.com/AZX-PBC-OSS/tors/commit/e8f52fc2c4f344eb4ffc8b047bff4fe1be844252))
* **canon:** fuzz target over adversarial value trees, wired into the run lists ([1e52d65](https://github.com/AZX-PBC-OSS/tors/commit/1e52d65caa866c0aa71afac30c1edf1e5ed3471f))
* **canon:** GIL heartbeat and wall cells with measured budgets ([b11fcf4](https://github.com/AZX-PBC-OSS/tors/commit/b11fcf489c52e83489ba8bcb6f54c7a65d635b2c))
* **canon:** tors.content_hash — the json.dumps-compatible object content hash ([94b4f04](https://github.com/AZX-PBC-OSS/tors/commit/94b4f04b2503da98d03eca447d5ed0b4c7e75b67))
* **charset:** first_invalid_offender, the offender detail for rejection UX ([b44c1af](https://github.com/AZX-PBC-OSS/tors/commit/b44c1af4ea6be5ceab35a6a483caf5b46d0492e9))
* **charset:** pinned common-alphabet constants (B62/B64URL/HEX) answering the validator-scope question ([6fb1100](https://github.com/AZX-PBC-OSS/tors/commit/6fb1100ce71b3fa4f7cbd48d8defebc5e304ba70))
* first_invalid_charset — batch codepoint-set validator, one detach per batch ([#53](https://github.com/AZX-PBC-OSS/tors/issues/53)) ([77a90a4](https://github.com/AZX-PBC-OSS/tors/commit/77a90a41758f56737f09a5de1e3149451fada31e))
* **hash:** digest-bytes spellings for the raw-digest call sites ([67ac84d](https://github.com/AZX-PBC-OSS/tors/commit/67ac84d632c2014fcfad289b432944fc4db907e0))
* **minhash:** docs -- api reference, family row and count, scope cut, perf rows, license table ([25dcca3](https://github.com/AZX-PBC-OSS/tors/commit/25dcca33081d1fb55d3cd839e128524406acde68))
* **minhash:** fuzz target, wired into all three run lists ([7a08ae3](https://github.com/AZX-PBC-OSS/tors/commit/7a08ae3de70400230cd91fe25687967570b972c5))
* **minhash:** GIL heartbeat and wall-time cells, plus the criterion bench ([7ef1bba](https://github.com/AZX-PBC-OSS/tors/commit/7ef1bba57528eaf28aba296d88b05ac9e50441f1))
* **minhash:** MinHash signatures for near-dup recall, oracle-pinned (issue [#62](https://github.com/AZX-PBC-OSS/tors/issues/62)) ([d456109](https://github.com/AZX-PBC-OSS/tors/commit/d45610927affa05de89edb7a39c7df01ab2177b2))
* one-shot hashing surface: md5/sha1/sha256/sha512 digets + hmac-sha256 request signing ([131273c](https://github.com/AZX-PBC-OSS/tors/commit/131273c1a0544cb26b789a6860946cf9be797806))
* **pii:** add the pii fuzz target and criterion bench ([8166c9e](https://github.com/AZX-PBC-OSS/tors/commit/8166c9eec4fa4c80aeb429ee94439176c80bbfe1))
* **pii:** domestic NANP shapes for contact_phone ([282d7e3](https://github.com/AZX-PBC-OSS/tors/commit/282d7e3c1f0ab6602d057e7a9e037785d5b21410))
* **pii:** per-family key selection, five new families, scrub_pii_report ([c8ed006](https://github.com/AZX-PBC-OSS/tors/commit/c8ed006cdfef629480d17027103f1bd006f54d2b))
* **pii:** pin the GIL and wall cells, document the scrub surface ([e8891b7](https://github.com/AZX-PBC-OSS/tors/commit/e8891b77686aa7e97e1b512f9ba4d7d701a5bd57))
* **pii:** port the contact-material scrub as tors.scrub_pii ([d618fa2](https://github.com/AZX-PBC-OSS/tors/commit/d618fa27f2196d8f219df3ae0e87cff77239e821))
* **pii:** the api_keys rule — scanner, per-rule salt split, binding ([bd92b1e](https://github.com/AZX-PBC-OSS/tors/commit/bd92b1e73ade314d2fb65d5458b7146e27cbac51))
* **pii:** wire the aio twin ([59ad804](https://github.com/AZX-PBC-OSS/tors/commit/59ad80469b347eb17e7cc92634b0e527aca4c493))
* **random:** add the random-generation family (strings, hex, b62, b64url, uuid4, uuid7) ([0f6ee65](https://github.com/AZX-PBC-OSS/tors/commit/0f6ee6555f3dd6146603afe227325e1ec96048e4))
* **random:** uuid4_bytes and uuid7_bytes ([11c677d](https://github.com/AZX-PBC-OSS/tors/commit/11c677dc066fe8c76600e71ce462ab639db851b8))
* **scan:** escape-parity byte scan core + contains_unescaped/find_unescaped binding (issue [#50](https://github.com/AZX-PBC-OSS/tors/issues/50)) ([fae7ee8](https://github.com/AZX-PBC-OSS/tors/commit/fae7ee8f5e98e7101ed91f9b1ed1334efd1733b8))
* **scan:** utf16_byte_len core + binding + export (issue [#52](https://github.com/AZX-PBC-OSS/tors/issues/52)) ([b969455](https://github.com/AZX-PBC-OSS/tors/commit/b9694559ffaae303c173c5f95b8604f259574cf5))
* **scan:** utf8_byte_len core + binding + export (issue [#52](https://github.com/AZX-PBC-OSS/tors/issues/52)) ([2f01930](https://github.com/AZX-PBC-OSS/tors/commit/2f01930ae90eff92a398c2006b4858088ad1a026))
* scrub_log_text, the named-rule log-scrub grammar ([6257031](https://github.com/AZX-PBC-OSS/tors/commit/62570310f80576fe2d519e4281f504a37062fbae))
* the api_keys rule for scrub_pii, format-anchored credential scrubbing ([e44c994](https://github.com/AZX-PBC-OSS/tors/commit/e44c994b92b37b48a2a132fa16df2e43f3914993))


### Bug Fixes

* address red-team findings on minhash_signature - injective framing, dedup sweep, index-based seed, streamed tokens ([3dca85e](https://github.com/AZX-PBC-OSS/tors/commit/3dca85e1d738d43a37ea68eec625fb642b8ed9dc))
* address red-team on content_hash - depth caps, iter bounds, mixed nesting docs, drop-in-detach ([c918249](https://github.com/AZX-PBC-OSS/tors/commit/c9182491eb12b6b3e537cd2ca54e516b06e11b4f))
* address red-team on first_invalid_charset - delink uuid_parse, normalize docs, bands ([9505c10](https://github.com/AZX-PBC-OSS/tors/commit/9505c1054f1409914bf6acf38090e5a1242ab746))
* address red-team on hashing - digest fuzz/bench/GIL twins, doc bands ([71c885f](https://github.com/AZX-PBC-OSS/tors/commit/71c885f0dda46993ce3f666795fbf2055f17ab87))
* address red-team on random-generation - fork probe, bias battery, RFC anchor, docs ([981bcd0](https://github.com/AZX-PBC-OSS/tors/commit/981bcd0c0693486a2fd0a88720ce39b6959d0e7d))
* address red-team on scrub_log_text - linearize userinfo, vendored regen, exhaustive tables ([446c079](https://github.com/AZX-PBC-OSS/tors/commit/446c079e5ed5d97b8f55629a6c218125311de847))
* address red-team on scrub_pii - threat-model docs, independent Nd table, domestic guard, clean-boundary ([9e951b4](https://github.com/AZX-PBC-OSS/tors/commit/9e951b4cc5fc11972e36c4a869908ed061fb4c78))
* address red-team on unescaped scan - unbounded fuzz oracle, sweep counts, asserts ([3d721f2](https://github.com/AZX-PBC-OSS/tors/commit/3d721f2bb23d9a20947de7a276b9cff016d7027c))
* address red-team on utf8_byte_len stacked - fresh-vs-repeat, cache proof, checked arith ([b8ddefb](https://github.com/AZX-PBC-OSS/tors/commit/b8ddefbd65065f646d81f362dd40410c40c467d6))
* address red-team on uuid7 helpers - bytes footnote, future-proof found, caret pin, message pins ([0c93201](https://github.com/AZX-PBC-OSS/tors/commit/0c93201e0d315ada7f369b202788de62211eedad))
* batch of tracker bugs, bounded and pinned ([#95](https://github.com/AZX-PBC-OSS/tors/issues/95)) ([9a7793c](https://github.com/AZX-PBC-OSS/tors/commit/9a7793ca5936621b9515895e0cb93de8bc1e7693))
* **canon:** honor overridden rich comparison on subclass dict keys ([4777e5d](https://github.com/AZX-PBC-OSS/tors/commit/4777e5dd3785ddb37226f8f45de9acf752af90b9))
* **canon:** protocol iteration for subclass containers, matching json.dumps ([422e1fa](https://github.com/AZX-PBC-OSS/tors/commit/422e1fa212d0cdbdce0ac77740bdd20d052966d7))
* **canon:** widen the int/bool key-sort fast path to bool keys ([ce642af](https://github.com/AZX-PBC-OSS/tors/commit/ce642af8ffec1a0b4093878aceb6b35d2ab1edc3))
* CI - 3.10 tomllib fallback, ASan-safe fuzz huge arm ([9be9ee3](https://github.com/AZX-PBC-OSS/tors/commit/9be9ee3e1c31815ecc2f1b19f67253c7bd87173a))
* CI - box-independent huge-window memory gate ([7e38e9f](https://github.com/AZX-PBC-OSS/tors/commit/7e38e9f527bf42b32c19107b70b648fc368e0ec2))
* CI - eliminate huge-window retention ([383b86b](https://github.com/AZX-PBC-OSS/tors/commit/383b86b7eec23e48862e8eb82b7cb2e2071c289f))
* CI - eliminate huge-window retention at the source ([7302e78](https://github.com/AZX-PBC-OSS/tors/commit/7302e78dcbedb1df357bff5ed11bb4be946de01d))
* CI - fuzz crash + UCD freshness pins ([87adef8](https://github.com/AZX-PBC-OSS/tors/commit/87adef843c7fecb046480a5dcfa2154d7eb96d07))
* CI - recursion-safe boundary test construction ([369a9e5](https://github.com/AZX-PBC-OSS/tors/commit/369a9e5bac813bf0bd9bc764eb59488d98782033))
* CI - version-gate UCD toolchain pin ([b13ff21](https://github.com/AZX-PBC-OSS/tors/commit/b13ff2107a638986a1a11d021ba508f7e63fbff0))
* close NO-GO - token-breaker digest isolation, BOTH-lane guard, fuzz assert, docs ([c9631db](https://github.com/AZX-PBC-OSS/tors/commit/c9631db622ff552578fb331d8425697a64d09dd6))
* **documents:** bound the document read ([#80](https://github.com/AZX-PBC-OSS/tors/issues/80)) ([c8604c9](https://github.com/AZX-PBC-OSS/tors/commit/c8604c951a6d18b5998fc5e2192f4a524b9a2564))
* lint ([376e026](https://github.com/AZX-PBC-OSS/tors/commit/376e02643ab1f235c77ff34229cdbd3e3bba98d7))
* lint ([b1b2621](https://github.com/AZX-PBC-OSS/tors/commit/b1b262123d88462fdb6a5b24697003dc9b3d3004))
* lint - fmt/clippy/ruff ([338b4c1](https://github.com/AZX-PBC-OSS/tors/commit/338b4c1e8493c2146f051273517c03bb0ef3e993))
* lint - fmt/ruff/clippy for hashing CI ([15216ec](https://github.com/AZX-PBC-OSS/tors/commit/15216ec1fba6565448054b963e32d19ea3b7f7b8))
* lint - imports/fmt ([03ecd55](https://github.com/AZX-PBC-OSS/tors/commit/03ecd55d3b367ffa8061d38ebe13e57c48e98e44))
* **pii:** linearize the PEM mismatched-END flood ([34ad678](https://github.com/AZX-PBC-OSS/tors/commit/34ad678a73c26c11f2ba5570fc6cd82bb1b2c30d))
* **pii:** map span ends at the verbatim head boundary affinely ([6b3eb21](https://github.com/AZX-PBC-OSS/tors/commit/6b3eb21e8737a4e2d6e0e682c1dbe145b327f984))
* post-rebase lint conformance - rustfmt wrap, clippy doc-lazy-continuation in scan_impl docs, one E501 wrap (same three fixes the integration tree needed) ([5a9186b](https://github.com/AZX-PBC-OSS/tors/commit/5a9186b44ce95077abfb4dff46cdd54c0eda46c3))
* **random:** align seed to the house int-like convention ([016b7aa](https://github.com/AZX-PBC-OSS/tors/commit/016b7aa5c4ea16c3f072a60af5249e0416689615))
* **random:** catchable MemoryError on oversized lengths ([6a277eb](https://github.com/AZX-PBC-OSS/tors/commit/6a277eb15acd016f318385cdb5d05b41e17615b0))
* residual charset - grapheme/confusable/hoist docs, pins ([ba3b05b](https://github.com/AZX-PBC-OSS/tors/commit/ba3b05b6aa2b8c345e2338c668124495518817fe))
* residual content_hash - threat-fitted caps, delegated-sort bounds, protocol budget honesty ([54d0329](https://github.com/AZX-PBC-OSS/tors/commit/54d03293645ce4de70548f5c1917d1b389da1be5))
* residual hashing - checksum notes, HMAC docs, pins ([a978014](https://github.com/AZX-PBC-OSS/tors/commit/a978014b5f0489d391624c8cd488935784e5fb5c))
* residual minhash - huge-window memory, cost honesty, distinct worst-case, binding hardening ([c66731c](https://github.com/AZX-PBC-OSS/tors/commit/c66731c29baab511d6dbd51740f7b7a0ba2bdb0e))
* residual random - cargo guard, ceiling matrix, docs ([d1c536f](https://github.com/AZX-PBC-OSS/tors/commit/d1c536f5b0ceac1dc379f6203b0af09b1eb0171d))
* residual scrub_log - warning, tripwire, lemmas, policy ([9df9370](https://github.com/AZX-PBC-OSS/tors/commit/9df9370a8d8eed381a915c2dfe0c9674c21d6b06))
* residual scrub_pii - leak docs, oracle freshness, guard pins ([4048f3f](https://github.com/AZX-PBC-OSS/tors/commit/4048f3f0669649192c296035def3bf519c343f57))
* residual unescaped - budget shape, CJK row, header honesty ([d3fa063](https://github.com/AZX-PBC-OSS/tors/commit/d3fa06380bdb222d1e68a074dc9aef64db8786c0))
* residual utf8_len - fallible core, airtight detach pin, docs ([1da2d98](https://github.com/AZX-PBC-OSS/tors/commit/1da2d989f3e7c3fe16f950d3ee3a69fe6bbe94e4))
* residual uuid7 - message honesty, doc wording ([fbd1dff](https://github.com/AZX-PBC-OSS/tors/commit/fbd1dff023505bccd6d8f3b9321373988a949612))
* **scan:** satisfy clippy byte_char_slices in the crate-side sweep ([924baba](https://github.com/AZX-PBC-OSS/tors/commit/924baba28649f285bbeac6d7ef03e6767385d248))
* **scan:** the utf16 ASCII fast path and cross-arch-honest wall cells; restore the scan family-table rows ([075ed7b](https://github.com/AZX-PBC-OSS/tors/commit/075ed7b690af895fd8f1bfb2026765e659ab119a))
* **scrub:** import the oracle for the corner pin ([8d838a4](https://github.com/AZX-PBC-OSS/tors/commit/8d838a4bae945560bc507d7397214d9308dda4d7))
* **scrub:** memoize the escaped-DETAIL pass's line state, not per needle ([06eb03b](https://github.com/AZX-PBC-OSS/tors/commit/06eb03b37a519e3067f79565ffecede04698a135))
* **scrub:** the honest convergence contract — a param mask can unblock a userinfo match on pass two ([36798fe](https://github.com/AZX-PBC-OSS/tors/commit/36798fe52bae5e73d2cc584281eb3dd2ebccd3ae))
* the ten open tracker bugs, bounded, pinned, and red-teamed ([#106](https://github.com/AZX-PBC-OSS/tors/issues/106)) ([32a9c7d](https://github.com/AZX-PBC-OSS/tors/commit/32a9c7d55658986fe45cfb23bb2fa7b87bec08e1))
* **uuid:** the all-versions datetime.timezone.utc spelling (3.10 has no datetime.UTC) ([92dbed4](https://github.com/AZX-PBC-OSS/tors/commit/92dbed4fc5ae6b7332557bb90e409bbaf6e8a1b4))


### Performance Improvements

* bench the first_invalid_charset core over 1/10/100-item batches ([#53](https://github.com/AZX-PBC-OSS/tors/issues/53)) ([654a6ca](https://github.com/AZX-PBC-OSS/tors/commit/654a6ca6657068c764c044b895663a28de9d83d1))
* **scan:** unescaped_scan criterion group + bench-corpus parity pins ([fc3adc5](https://github.com/AZX-PBC-OSS/tors/commit/fc3adc5ddb9fa6a6ccd484f6a986b5ed22c9a112))
* **scan:** utf16_byte_len wall cells, heartbeat cell, bench group (issue [#52](https://github.com/AZX-PBC-OSS/tors/issues/52)) ([d80912e](https://github.com/AZX-PBC-OSS/tors/commit/d80912eaa20e986412a7309db5fbd56f1d833da4))
* **scan:** utf8_byte_len criterion group ([1e1b171](https://github.com/AZX-PBC-OSS/tors/commit/1e1b171c1f32321ccebd3d892bbcd63e95c2aa3b))


### Documentation

* **canon:** the content_hash contract spec and the family/doc rows ([9ea7bd9](https://github.com/AZX-PBC-OSS/tors/commit/9ea7bd9a4810da40ee89e8dc474d56593038be87))
* **charset:** the ==-1 port caveat for first_invalid_offender ([87ce229](https://github.com/AZX-PBC-OSS/tors/commit/87ce2299997591c7038a7f1399b1333dd7dacb52))
* **charset:** the hoist claim corrected - the borrow warms, the per-call build does not ([f0e1365](https://github.com/AZX-PBC-OSS/tors/commit/f0e1365a5750083faa5f31783f46759f2d0df649))
* count the five pinned charset constants in the README census ([4881d28](https://github.com/AZX-PBC-OSS/tors/commit/4881d28eee261a459af9717e56603f3f71dee674))
* document the uuid7 helper trio in api.md and the README family table ([296a0ec](https://github.com/AZX-PBC-OSS/tors/commit/296a0ec60ea8b36cb81b0093115b4da516a8e604))
* first_invalid_charset section, charter clause, measured bands ([#53](https://github.com/AZX-PBC-OSS/tors/issues/53)) ([410e37e](https://github.com/AZX-PBC-OSS/tors/commit/410e37e81cbd2f746b8a7b93eba5085bb5214abe))
* **hash:** the checksum-only note where the family is named ([4da9798](https://github.com/AZX-PBC-OSS/tors/commit/4da9798039cbb67568f4b0e4c92a31e706b1ce93))
* **minhash:** correct the token_count_up_to provenance - the 570-615MB reading was a getrusage fiction, not a glibc RSS peak ([93921ce](https://github.com/AZX-PBC-OSS/tors/commit/93921ce55ba49616757d87df3661a8f7a8b4409a))
* **minhash:** true injectivity invariant, version-stability boundary, bounds error classes ([9c06efb](https://github.com/AZX-PBC-OSS/tors/commit/9c06efb8fb4255f48ac10da6664efd5315adb940))
* **pii:** families table, exclusions, report section, red-team risk bullets ([c66d777](https://github.com/AZX-PBC-OSS/tors/commit/c66d7775e776d1ae13fbd46f4e229b3ac4c318c1))
* **pii:** the api_keys rule — family table, salt split, pyi ([f8b37fb](https://github.com/AZX-PBC-OSS/tors/commit/f8b37fb9f0df3b44bdee46822b85e06aa5884245))
* **pii:** the head-truncation exception, README census, order wording ([7c70ced](https://github.com/AZX-PBC-OSS/tors/commit/7c70ced4adc7672b1a7f7dbd74a6fb290512ccd6))
* **random:** correct the b64url final-char arithmetic ([89979ba](https://github.com/AZX-PBC-OSS/tors/commit/89979ba18f5888fe204007362e0bd2b9d1260e75))
* **random:** state uuid7's true timestamp horizon ([cc39311](https://github.com/AZX-PBC-OSS/tors/commit/cc39311216bd25103a292d0e1069d2c3f628bf77))
* **random:** the length-first ledger -- re-measured engine tables and the slimmed dependency story ([6495548](https://github.com/AZX-PBC-OSS/tors/commit/6495548dc53c782adbe890d09edebfa9498e6df6))
* **random:** the security contract, family sections, and measured tables ([a7e0fc8](https://github.com/AZX-PBC-OSS/tors/commit/a7e0fc87348c5f634d6acc811597ad4faac24a20))
* README family row + function count for first_invalid_charset ([c411835](https://github.com/AZX-PBC-OSS/tors/commit/c411835621c2f7f40287d26a6446cb66f9c5d849))
* **readme:** 75 functions and a family-table row for the escape-parity scan ([89d266d](https://github.com/AZX-PBC-OSS/tors/commit/89d266dcc188c23215380c5408e200da35ece365))
* **readme:** fix the stale function count for the scan family (73 -&gt; 76) ([31097bb](https://github.com/AZX-PBC-OSS/tors/commit/31097bb270ed834d6bf433ac9ef9e617d7509298))
* **scan:** api.md section + example pins + performance.md entry ([dc05b2f](https://github.com/AZX-PBC-OSS/tors/commit/dc05b2f3d9c3b601db6a7d3dd9881cf9b7236ab3))
* **scan:** correct the utf16 overflow story - unreachable for real inputs on every width ([b75108d](https://github.com/AZX-PBC-OSS/tors/commit/b75108d35d6994805f3b2fc4dfade0b635051f68))
* **scan:** polish two core doc comments ([c791c40](https://github.com/AZX-PBC-OSS/tors/commit/c791c407861ebbb857a9696b3833190f35d25e91))
* **scan:** the true cost story for the parity walk ([206b952](https://github.com/AZX-PBC-OSS/tors/commit/206b952a8543cb0f74be29275cee0517d628444e))
* **scan:** utf16_byte_len api.md section + example pins + performance.md row + README count ([922c137](https://github.com/AZX-PBC-OSS/tors/commit/922c137d9ae6279e335925e442906b93cfe71b93))
* **scan:** utf8_byte_len api.md section + example pins + performance.md entry ([3fb848f](https://github.com/AZX-PBC-OSS/tors/commit/3fb848f5f6903a764751602438ace9e809f0e6af))
* the exact head-truncation mechanism, census 105, crash artifact corrected ([f1d5241](https://github.com/AZX-PBC-OSS/tors/commit/f1d52410cd7e55f78bc1dc8b4dc0a34d86bfd6c0))
* the scrub_log_text section, scope entry, family row, example pins ([70b1a60](https://github.com/AZX-PBC-OSS/tors/commit/70b1a60d223e30182d3a1d825585dfa330618e27))
* **uuid:** record the uuid-crate adoption in api.md and the dependency records ([3598acd](https://github.com/AZX-PBC-OSS/tors/commit/3598acded982cd6cecf8c7c8ce8f43cd9990a132))

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
