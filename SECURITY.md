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
  the engine-side decompression caps (anydoc's per-entry and total package
   limits; office_oxide 0.1.10's 512 MiB per-part cap). A bypass, degradation,
   or regression of those caps is a security finding here, not just a bug. One
   known limit is already tracked for an upstream fix and is not itself a
   reportable finding: pdf_oxide 0.3.78's `parse_object`/`parse_array` recursion
   has no depth cap (its `max_nesting` config is dead code), so a ~60 KB PDF with
   a 30k-deep trailer array crashes every entry point. It is noted in
   `src/pdf_impl.rs`'s docs and exercised by the `documents_markdown` fuzz target;
   the real fix belongs upstream.
- **Supply chain**: the PyPI release pipeline (`publish.yml`, which builds and
  publishes both wheels, each through its own Trusted Publishing identity) uses
  PyPI Trusted Publishing (OIDC): there is no long-lived API token to leak, but a
  compromised GitHub Actions dependency in that workflow's chain is in scope.

## Disclosure

We follow coordinated disclosure. Once a fix is released, we will publish a GitHub Security
Advisory with credit to the reporter (unless they prefer to remain anonymous).
