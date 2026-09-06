# Contributing

Thanks for your interest in contributing to `tors`!

## Setup

```sh
uv sync --locked
```

Every command after that takes `--no-sync` (a bare `uv run` re-resolves the environment and
can silently drop the Rust extension build — see `README.md`'s Development section).

After changing Rust code: `uv sync --locked --reinstall-package tors` or `maturin develop`.

## Before you open a PR

```sh
uv run --no-sync pytest -q
cargo test --no-default-features
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Also update `README.md` and `docs/` if your change affects behavior or the public API — a
stale doc is treated as a defect, the same severity as a stale test.

## Commit style

PR titles must follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, etc.) — enforced by CI and parsed by release-please to decide the
next version and CHANGELOG entry. Do not hand-edit `CHANGELOG.md` or bump the version in
`pyproject.toml` yourself; release-please owns both.

## Reporting bugs and requesting features

Use [GitHub Issues](https://github.com/AZX-PBC-OSS/tors/issues). For security
vulnerabilities, see [`SECURITY.md`](SECURITY.md) instead — please don't file those as
public issues.
