# Documentation accuracy report — mechanical verification of runnable examples

Date: 2026-09-25 · Scope: every ```python fence in `docs/api.md`, `docs/async.md`,
`docs/performance.md`, and `README.md` · Environment: repo checkout, built extension,
`PYTHONPATH=python`, CPython 3.12 venv.

## Method

Each markdown file was split on `## ` headings; each section's python blocks were run
**together in one fresh subprocess, in document order, in a shared namespace** (so
`original`/`edited`-style context variables shared across a section's blocks work as the
doc intends). For every statement executed, the value of top-level expression statements
(and single-name assignments) was captured and compared against the claim comment that
follows or trails it (`# ['foo']`, `# 'the cat sat'`, `# 5`, …), including multi-line
wrapped claims and claim+prose hybrids (`# ([0], 1): page index 0 …`). Comparison is
against the real `repr`, whitespace-collapsed for line-wrapping tolerance; a secondary
value-equality check absorbs quote-style differences. Blocks raising an exception whose
type is named in the following comment are verified as expected raises, with the claimed
message text compared against the actual one. Sections failing with `NameError` were
retried with the accumulated namespace of all prior sections (doc-internal shared
context). Blocks that fail only because the doc conventionally omits `import tors` (or
uses bare `tors`-member names) were re-run with the module's public names injected and
are marked **[scaffold]** below. Signature stubs (`def f(...) -> T: ...` from the
`.pyi`) and one placeholder pseudo-code block were skipped. Claims inside compound
statements (`try`/`except` bodies) and claims requiring reader-side inputs or prose
variables were verified with targeted manual runs, listed explicitly below.

Harness: `/tmp/opencode/docverify/run.py` + `section_runner.py`; raw per-block results:
`/tmp/opencode/docverify/results.json`.

## Summary

| file | python blocks | run OK | expected-raise OK | skipped (signature stubs) | claims checked | mismatches |
|---|---|---|---|---|---|---|
| docs/api.md | 218 | 101 (68 of them [scaffold]) | 0¹ | 105 + 1 pseudo-code | 186 mechanical + 24 manual | 4 raw, all resolved (below); 1 DOC WRONG (spelling) |
| docs/async.md | 1 | 0² | 0 | 0 | 0 mechanical + 1 manual | 0 |
| docs/performance.md | 0 | — | — | — | — | — (file contains no code fences at all) |
| README.md | 3 | 3 | 0 | 0 | 6 | 0 |

¹ Blocks whose only execution failure is the doc's own intentional `ValueError` demo end
up classified OK once the raise is verified against its claimed message.
² The single block uses `await` at top level and a prose-defined `corpus`; verified by
wrapping in `asyncio.run` with a scaffold corpus (below).

**192 mechanically checked output claims: 188 exact matches; 4 raw mismatches, all
resolved below (elision/nondeterminism, no factual error). One DOC WRONG finding (missing
`tors.` prefix, outputs themselves all correct).**

## Mismatches (every raw strict-mismatch recorded)

1. **docs/api.md:4152** (`tors.merkle_root`) — claimed `'36642e73...e6c021ec1'`, actual
   `'36642e73c2540ab121e3a6bf9545b0a24982cd830eb13d3cd19de3ce6c021ec1'`.
   **Not wrong**: intentional `...` elision; claimed prefix and suffix both byte-exact.
2. **docs/api.md:6151** (`tors.uuid7`) — claimed
   `'01a09927-e261-7153-8d74-283f6d488162'`, actual `'01a0db7b-…'` (fresh run).
   **Not wrong (doc nit)**: `uuid7` is unseeded and clock-based, so no literal is
   reproducible; the doc shows a real run's output. Minor inconsistency: the
   `uuid7_timestamp_ms` section (api.md:4500) explicitly pins a fixed UUID "so the
   outputs below are literals", while this example presents an unreproducible literal
   with no "output varies" note. Claimed pair is internally consistent (the claimed
   uuid's timestamp field decodes to the claimed `1789275923041`).
3. **docs/api.md:6153** (`tors.uuid7`) — claimed `1789275923041`, actual `1790388725374`.
   **Not wrong**: same nondeterminism; actual value equals today's Unix epoch ms.
4. **docs/api.md:6178** (`tors.uuid7_bytes`) — claimed `'01a09927e261'`, actual
   `'01a0db7be67e'`. **Not wrong**: same nondeterminism.

## DOC WRONG (1)

- **docs/api.md:5670** (`tors.rank_fuse` / `ndcg_at_k` / … section): the example calls
  bare `rank_fuse([...])`, `ndcg_at_k(ranked, relevant)`, `mrr`, `recall_at_k`,
  `precision_at_k` — no `tors.` prefix and no import — and raises
  `NameError: name 'rank_fuse' is not defined` exactly as written. Every other runnable
  example in api.md uses the `tors.` prefix (or shows `import tors`), so this block's
  spelling contradicts the doc's own convention. **The claimed outputs are all correct**:
  re-run with the module names in scope, all seven claims match exactly
  (`0.7039180890341347`, `1.0`, `0.3333333333333333`, `0.6666666666666666`, `0.5`,
  `0.6131471927654584`, and the wrapped `rank_fuse` tie-order tuple). Fix: prefix each
  call with `tors.`.

## Skipped blocks (all accounted)

- **docs/api.md** — 105 signature-stub blocks (`def f(...) -> T: ...` / class stubs from
  `python/tors/__init__.pyi`), by design not runnable.
- **docs/api.md:515** (`tors.scrub_pii` report shape) — pseudo-code with placeholders
  (`{"text": <scrubbed str>, …}`); illustrative shape, not runnable (SyntaxError by
  design). The surrounding prose defines it as "the shape below is the accounting".

## Claims verified manually (mechanical checker cannot reach them)

- **docs/api.md:48** — `tors.normalize(s) is s` (`s` from prose): verified `True` on a
  clean sample string; the identity contract holds.
- **docs/api.md:87** — `text, digest = tors.finalize(...)` claim tuple (elided digest):
  text element and full digest `e986ba083c7c…f7b942a` verified via the README quickstart
  block (which matched exactly).
- **docs/api.md:1753/1754** — claims `5` and `4` for `utf8_byte_len`: matched (the
  block's trailing `if` statement uses the prose variable `serialized_result` and is not
  runnable; that statement is illustrative).
- **docs/api.md:1844–1846** — claims `8`, `4`, `4` for `utf16_byte_len`: matched (same
  trailing-illustrative-`if` situation with prose `s`).
- **docs/api.md:1932–1936** — `CompiledPatterns` example: `'the dog sat'` claim matched;
  the `for doc in corpus:` loop uses a prose-defined `corpus`, verified to run with a
  scaffold corpus.
- **docs/api.md:2089–2097** — both `repair_json` claims matched; the third fragment's
  `response` is prose-defined; verified with a scaffold two-fence response
  (`['{"a": 1}', '[2]']`).
- **docs/api.md:3923 / 3930** (`content_hash` oracle) — definitional snippets referencing
  prose `obj`/`json`/`hashlib`; verified as the exact contract: the stdlib oracle equals
  `tors.content_hash(obj)` on scaffold input.
- **docs/api.md:4685** — MinHash agreement formula with prose `sig_a`/`sig_b`: runs and
  returns 1.0 for identical signatures, as the prose states.
- **docs/api.md:5394–5396** — `lemma_dict`/`many_batches` are prose-defined; verified the
  `CompiledLemmaDict` + `apply_pipeline` shape runs with a scaffold mapping/batches.
- **docs/api.md:3355–3358** (`chunk_by_paragraphs_iter`) — claims
  `[(0, 69), (32, 111), (71, 141)]` and the chunk-text explanation for `(32, 111)`:
  both exact.
- **docs/api.md:3717–3721** (`chunk_hierarchical`) — claims `[(0, 39), (41, 100),
  (102, 125)]` and `"## Usage"` chunk-start explanation: exact.
- **docs/api.md:6703–6724** (`documents.to_markdown`/`to_text`) — `Format.HTML/PDF/CSV`
  enum tuples all match (actual repr spells `<Format.HTML: 'html'>`; the doc uses
  constructor spelling, values identical), including the `pages=1` and full-range
  byte-identity `True` claims.
- **docs/api.md:6729–6738** — references reader-side files (`upload.bin`,
  `engines_report.docx` in the CWD); with the repo's
  `tests/engines_corpus/engines_report.docx` fixture both claims verify: sniff returns
  `Format.DOCX` for the extensionless copy, and the `data=`/path byte-identity returns
  `True`.
- **docs/api.md:6766–6780** (`documents.sniff`) — all six claims exact: `Format.PDF`,
  `Format.RTF`, `Format.HTML`, `Format.CSV`, and both `None` refusals (prose, JSON-lines).
- **docs/api.md:6849–6866** (`pdf_extract`) — the `(["first page line", "second page
  line"], …)` tuple matches (quote style only), the backend-parity `True` claim matched
  mechanically, and the caught `ValueError`'s repr — printed by the `except` body —
  matches the doc's claimed message byte-for-byte.
- **docs/api.md:6991–7001** (`pdf_classify`) — both `PdfClassification(page_count=…,
  page_kinds=[…], pages_needing_ocr=…)` reprs and the `(True, False, [1])` claim: exact.
- **docs/api.md:7034–7040** (`NeedsOcrError`) — the `except`-body claim `([0], 1)`:
  verified, `exc.pages == [0]`, `exc.page_count == 1`.
- **docs/api.md:7126** (`documents.aio`) — `report.docx` is reader-side; verified with
  the fixture via `asyncio.run`: returns `(Format.DOCX, 'Quarterly Review Q3 2026…')`.
- **docs/async.md:8–11** — `await tors.aio.tf_idf(corpus)` with prose `corpus` and
  top-level `await`: verified via `asyncio.run(tors.aio.tf_idf([...]))` — returns the
  expected list.

## Observations (not defects)

- **`import tors` convention**: 68 api.md blocks omit `import tors` (or use bare member
  names, in the one DOC WRONG case above), relying on the module being imported
  earlier in the document. All claims in them verify once the import is in scope.
  A one-line "examples assume `import tors`" note in the api.md preamble would make this
  explicit.
- **README prose count**: "107 functions plus two small helper classes and six pinned
  constants" — the core surface actually exposes 127 callables in `__all__`
  (125 functions + `CompiledPatterns` + `CompiledLemmaDict`) plus 6 pinned constants.
  107 is reachable only by excluding twin/alias spellings (`_digest` ×5, `_iter` ×8,
  `nfc`/`nfd`/`nfkc`/`nfkd` ×4, …); the README table itself lists those twins, so the
  count as written does not decompose cleanly. Worth re-counting when the surface next
  changes. (Outside the python-block scope; no verdict issued.)
- **docs/performance.md** contains no ```python fences (its numbers are prose + tables);
  nothing to execute.

## Bottom line

Every claimed output in the four files that is mechanically checkable — 192 claims —
matches the runtime, the only as-written failure being the missing `tors.` prefix in the
rank-fusion example (api.md:5670), whose outputs are themselves correct. No runtime
behavior contradicts the documented contracts; no RUNTIME BUG found.
