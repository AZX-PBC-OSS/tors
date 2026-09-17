# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in `tors`, please report it
responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please use [GitHub Security Advisories](https://github.com/AZX-PBC-OSS/tors/security/advisories/new)
to report vulnerabilities privately.

## Response Timeline

- **Acknowledgment:** Within 48 hours
- **Initial Assessment:** Within 5 business days
- **Fix or Mitigation:** Depends on severity, typically within 30 days for high-severity issues

## Scope

This policy covers `tors` itself (the Rust extension and its Python bindings), the
second wheel `tors-documents` (the `tors.documents` payload and its engines), and
their CI/CD pipeline. Vulnerabilities in third-party dependencies (PyO3,
`unicode-normalization`, the document engines (`pdf_oxide`, `anydoc`, `office_oxide`,
`html-to-markdown-rs`), or others in `Cargo.lock`/`uv.lock`) should be reported to
their respective maintainers; if you're unsure who owns a dependency, report it to us
and we'll help route it.

When reporting, include:

- **Memory safety** in the Rust extension, especially anything reachable from
  untrusted Python-side input (`tors`'s functions take arbitrary `str`/`bytes`).
- **Native parsing of untrusted documents**: `tors.documents` hands four native
  engines (`pdf_oxide`, `anydoc`, `office_oxide`, `html-to-markdown-rs`) document
  bytes a caller cannot trust, so parser-level memory safety, hangs, and crashes
  are in scope on every entry point (`to_markdown`/`to_text` and the PDF-only
  calls, by `path=` or `data=`). The resource-ceiling threat model is part of this
  surface: the explicit `max_bytes=` pre-read bound (every lane), the 32 MiB
  default post-read ceiling on the amplifying lanes (anydoc, office_oxide), and
  the decompression caps: anydoc's per-entry and total package limits;
  office_oxide's 512 MiB per-part cap plus the tors-documents seam's own
  512 MiB total-across-parts audit on the oxide lane (`backend="oxide"`),
  which refuses a container whose parts inflate past the ceiling in sum
  (declared sizes summed, then the parts re-inflated under a hard per-part
  bound so a lying header cannot get past it). A bypass, degradation, or
  regression of those caps is a security finding here, not just a bug.
   One crash class was found here and is mitigated in-repo while its
   upstream fix ships: pdf_oxide 0.3.78's `parse_object`/`parse_array`
   recursion has no depth cap (its `max_nesting` config is dead code; a
   trailer nesting `[` arrays 10,659 deep SIGSEGV'd every PDF entry point,
   exit -11, uncatchable from Python). `pdf_impl::open()` refuses
   over-nested documents before pdf_oxide parses them: a linear raw-byte
   nesting scan (100-level cap, pdf_oxide's own intended value), the same
   scan re-run over FlateDecode object-stream and xref-stream payloads
   after inflating them under a hard 256 MiB inflated-bytes ceiling (the
   compressed shape the raw scan cannot see, measured still SIGSEGV-ing
   pre-fix), and the pass itself on a 256 MiB-stack thread with panic
   containment, so every probed shape surfaces as the catchable
   malformed-document `ValueError`, pinned red-first in
   `tests/test_documents_pdf_crashes.py`. Upstream: issue
   yfedoseev/pdf_oxide#1474, fix commit 68da2cf8 on the release/v0.3.79
   branch, unpublished on crates.io at this writing; the in-repo layers
   stay as defense-in-depth once it publishes. The `documents_markdown`
   fuzz target is built and committed as the repro harness but deliberately
   kept out of every run list (the Makefile's `FUZZ_TARGETS`, ci.yml's
   smoke loop, fuzz.yml's weekly matrix): a target aimed at a
   known-uncapped parser is a time bomb until 0.3.79 ships, and the
   measured bombs are covered by the pin suite instead.
- **Supply chain**: the PyPI release pipeline (`publish.yml`, which builds and
  publishes both wheels, each through its own Trusted Publishing identity) uses
  PyPI Trusted Publishing (OIDC): there is no long-lived API token to leak, but a
  compromised GitHub Actions dependency in that workflow's chain is in scope.

## Disclosure

We follow coordinated disclosure. Once a fix is released, we will publish a GitHub Security
Advisory with credit to the reporter (unless they prefer to remain anonymous).
