# Security Policy

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in
`tors`, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please use [GitHub Security Advisories](https://github.com/AZX-PBC-OSS/tors/security/advisories/new)
to report vulnerabilities privately.

## Response Timeline

- **Acknowledgment:** Within 48 hours
- **Initial Assessment:** Within 5 business days
- **Fix or Mitigation:** Depends on severity, typically within 30 days for high-severity issues

## Scope

This policy covers `tors` itself (the Rust extension and its Python bindings) and its CI/CD
pipeline. Vulnerabilities in third-party dependencies — PyO3, `unicode-normalization`, or
others in `Cargo.lock`/`uv.lock` — should be reported to their respective maintainers; if
you're unsure who owns a dependency, report it to us and we'll help route it.

Areas worth being precise about when reporting:

- **Memory safety** in the Rust extension, especially anything reachable from untrusted
  Python-side input (`tors`'s functions take arbitrary `str`/`bytes`).
- **Supply chain**: the PyPI release pipeline (`publish.yml`) uses PyPI Trusted Publishing
  (OIDC) — there is no long-lived API token to leak, but a compromised GitHub Actions
  dependency in that workflow's chain is in scope.

## Disclosure

We follow coordinated disclosure. Once a fix is released, we will publish a GitHub Security
Advisory with credit to the reporter (unless they prefer to remain anonymous).
