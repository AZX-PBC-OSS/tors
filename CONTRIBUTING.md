# Contributing

Thanks for your interest in contributing to `tors`!

## Setup

```sh
uv sync --locked
```

Every command after that takes `--no-sync` (a bare `uv run` re-resolves the environment and
can silently drop the Rust extension build).

After changing Rust code: `uv sync --locked --reinstall-package tors` or `maturin develop`.

## Before you open a PR

```sh
uv run --no-sync pytest -q
cargo test --no-default-features
cargo clippy --all-targets -- -D warnings
cargo fmt --check
uv run --no-sync ruff check .
cargo deny check licenses advisories bans
```

`make lint` wraps the fmt/clippy/ruff lines (clippy in both feature configs);
`make test` wraps the pytest and cargo-test lines plus the extension rebuild. The
pytest suite has a `timing` marker for the measurement cells
(`tests/test_gil_release.py`, `tests/test_performance.py`; slow,
load-sensitive): local runs execute everything, marker or not, while CI's
matrix legs run `-m "not timing"` and one dedicated 3.12-leg step runs
`-m timing`, so those contracts gate every push without being paid five
times. The
last one is the licensing gate: the permissive-only
dependency allowlist and the advisory/ban policy live in `deny.toml` (`make
deny` wraps it). It requires `cargo-deny` on PATH: `cargo install cargo-deny
--locked` (~1 minute).

If a change must regenerate `src/html_table.rs` (a Python release changed the
HTML entity tables): `make gen-html-table`, and update the pinned counts in
`src/html_impl.rs`'s length-pin test, as part of the same change.

Also update `README.md` and `docs/` if your change affects behavior or the public API: a
stale doc is treated as a defect, the same severity as a stale test.

## Commit style

PR titles must follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, etc.): enforced by CI and parsed by release-please to decide the
next version and CHANGELOG entry. Do not hand-edit `CHANGELOG.md` or bump the version in
`pyproject.toml` yourself; release-please owns both.

## Reporting bugs and requesting features

Use [GitHub Issues](https://github.com/AZX-PBC-OSS/tors/issues). For security
vulnerabilities, see [`SECURITY.md`](SECURITY.md) instead: please don't file those as
public issues.
