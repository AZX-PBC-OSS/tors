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

## Fuzzing

`fuzz/` holds `cargo-fuzz` (libFuzzer) targets for the parsing/decoding/scoring
cores most exposed to raw adversarial input: `decode_utf8`, `decode_utf16`,
`b64_decode`, `html_unescape`, `fence`, `chunk_hierarchical`, `normalize`,
`search`, `segmentation`, `diff`, `phonetic`, `bm25`, `tfidf`,
`truncate_ellipsis`, and `strip_controls` (the current
list is `fuzz/Cargo.toml`'s `[[bin]]` entries — treat that file, not this one,
as the source of truth if the two ever disagree). These call the `*_impl.rs`
cores directly (no pyo3 boundary, no Python interpreter needed) and check
crash-freedom plus a few cross-function invariants (`utf8_is_valid` must never
disagree with `decode_utf8`'s own success/failure; a zero-cost identity return
must never be a false positive). This is a different bug class than the
hypothesis-based Python tests, which shape input around documented contracts
rather than raw adversarial bytes.

Requires nightly and `cargo-fuzz`:

```sh
rustup install nightly
cargo install cargo-fuzz --locked
```

`make fuzz-quick` runs every target for 30 seconds each (a smoke pass, worth
running before a PR that touches any of the covered functions); `make fuzz`
runs them without a time limit (stop with Ctrl-C). The `fuzz/` crate is
deliberately not a Cargo workspace member, so it never affects the main
`cargo build`/`cargo test` at the repo root. A crash writes a repro case to
`fuzz/artifacts/<target>/` (gitignored, local-only) — reproduce it directly
with `cargo +nightly fuzz run <target> fuzz/artifacts/<target>/<crash-file>`.

## Commit style

PR titles must follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, etc.): enforced by CI and parsed by release-please to decide the
next version and CHANGELOG entry. Do not hand-edit `CHANGELOG.md` or bump the version in
`pyproject.toml`/`Cargo.toml` yourself; release-please owns all three (see
`release-please-config.json`'s `extra-files` entry, which keeps Cargo.toml's version in
step with pyproject.toml's on every release).

## Reporting bugs and requesting features

Use [GitHub Issues](https://github.com/AZX-PBC-OSS/tors/issues). For security
vulnerabilities, see [`SECURITY.md`](SECURITY.md) instead: please don't file those as
public issues.
