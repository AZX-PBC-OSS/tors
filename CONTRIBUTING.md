# Contributing

Thanks for your interest in contributing to `tors`!

## Setup

```sh
uv sync --locked --extra documents
```

Every command after that takes `--no-sync` (a bare `uv run` re-resolves the environment and
can silently drop the Rust extension build). The `--extra documents` is required: it
keeps the tors-documents payload wheel installed. A plain `uv sync --locked` would
uninstall the payload, and the pytest suite's documents gates would silently skip at
module level (the same hazard `make install`/`make dev` and CI's sync step avoid by
always carrying the extra).

After changing Rust code: `uv sync --locked --extra documents --reinstall-package tors`
or `maturin develop`.

## Before you open a PR

```sh
uv run --no-sync pytest -q
cargo test --no-default-features
cargo clippy --all-targets -- -D warnings
cargo fmt --check
uv run --no-sync ruff check .
cargo deny check licenses advisories bans
```

The pytest line presupposes Setup's `--extra documents`: the payload wheel must still
be installed (`--no-sync` never reinstalls it), or the documents gates skip at module
level: a green run that validated nothing on the document side.

`make lint` wraps the fmt/clippy/ruff lines (clippy in both feature configs);
`make test` wraps the pytest and cargo-test lines plus the extension rebuild. The
pytest suite has a `timing` marker for the measurement cells
(`tests/test_gil_release.py`, `tests/test_performance.py`; slow,
load-sensitive): local runs execute everything, marker or not, while CI's
matrix legs run `-m "not timing"` and one dedicated 3.12-leg step runs
`-m timing`, so those contracts gate every push without running on every matrix
leg. The final command is the licensing gate: the permissive-only
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
`search`, `segmentation`, `diff`, `grounded`, `phonetic`, `bm25`, `tfidf`,
`truncate_ellipsis`, `controls`, `json_repair`, and `gfm_strip` (the current
list is `fuzz/Cargo.toml`'s `[[bin]]` entries; treat that file, not this one,
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

`make fuzz-quick` runs every target for 30 seconds each (a smoke pass; run it
before a PR that touches any of the covered functions); `make fuzz`
runs them without a time limit (stop with Ctrl-C). The `fuzz/` crate is
intentionally outside the Cargo workspace, so it never affects the main
`cargo build`/`cargo test` at the repo root. A crash writes a repro case to
`fuzz/artifacts/<target>/` (gitignored, local-only). Reproduce it directly
with `cargo +nightly fuzz run <target> fuzz/artifacts/<target>/<crash-file>`.

## Commit style

PR titles must follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, etc.): enforced by CI and parsed by release-please to decide the
next version and CHANGELOG entry. Do not hand-edit `CHANGELOG.md` or bump the version in
`pyproject.toml`/`Cargo.toml` yourself; release-please owns all three (see
`release-please-config.json`'s `extra-files` entry, which keeps Cargo.toml's version in
step with pyproject.toml's on every release).

## Releasing

One release ships three artifacts: the `tors` wheels + sdist and the
`tors-documents` wheels to PyPI, and the `tors-core` crate to crates.io. The
`tors-documents` wheel is version-locked to `tors` (same number, one release
PR, two wheels), and `publish.yml`'s `publish-documents` job asserts the
lockstep before uploading, because the `tors[documents]` extra pins the payload
by name, not version: a mismatched pair would install side by side silently.
Drift is therefore a build failure, not a user-visible bug.

Release-please owns every version: the release PR it opens from Conventional
Commits bumps `pyproject.toml`, `Cargo.toml`, `tors-documents/Cargo.toml`, and
`tors-documents/pyproject.toml` together (`release-please-config.json`'s
`extra-files`), and `release-lock-sync.yml` refreshes `uv.lock` and both
`Cargo.lock`s on the PR itself. Never hand-bump any of the four files.

Merging the release PR cuts the tag but does not publish: a tag cut with
`GITHUB_TOKEN` cannot start a workflow run (see `release-please.yml`'s header,
which prints this reminder with the tag name on every release). Unless the
optional GitHub App token described there is configured, trigger the publish
run by hand: Actions → publish → Run workflow → select the new tag `vX.Y.Z`.
That one run publishes everything: the base jobs, the documents jobs, and the
crates.io job run from the same workflow file.

### One-time publishing setup

The base `tors` PyPI project, the `tors-core` crates.io crate, and the `pypi`
and `crates-io` GitHub environments are already configured (crates.io has had
a Trusted Publisher on this repo since `tors-core` 0.3.1). One setup remains, required once, before the first `tors-documents` release
tag:

- **PyPI pending publisher** (no API token exists or is needed anywhere). On
  pypi.org, logged in as the account that
  will own the project: account menu → *Publishing* → *Add a pending
  publisher*, with exactly:
  - Project name: `tors-documents`
  - Owner: `AZX-PBC-OSS`
  - Repository: `tors`
  - Workflow filename: `publish.yml`
  - Environment name: `pypi`

  A pending publisher is required before the first release because PyPI
  refuses a Trusted Publishing upload for a project that does not exist yet;
  with this in place, the first `publish-documents` run creates the project
  and every later release authenticates the same way. The `environment: pypi`
  the workflow's jobs declare (publish.yml) must match the environment name
  entered here exactly; it is part of the OIDC claim PyPI verifies.

- **GitHub environment `pypi`**: Settings → Environments → `pypi`, already in
  use by the base `publish` job; `publish-documents` reuses it, and a Trusted
  Publishing identity is per PyPI project, not per environment, so nothing new
  is needed here. (Only create protection rules if you want a human approval
  gate in front of registry uploads; they then apply to both PyPI jobs.)

Nothing is needed on the crates.io side for the documents work: the four
engines (`pdf_oxide`, `anydoc`, `office_oxide`, `html-to-markdown-rs`) are
optional, registry-sourced dependencies of `tors-core`, so they ride the next
`cargo publish` normally and stay opt-in for Rust consumers (the feature is
off by default, and publish.yml's crates job publishes with `--no-default-features`,
so its verification build never even compiles them). The payload crate itself
(`tors-documents`) is `publish = false` in its Cargo.toml: its `tors-core`
dependency is a path dependency, which cargo cannot publish, and PyPI is its
only distribution channel. That line is intentional.

The docs site needs nothing from a release: `docs.yml` builds the static
`docs/*.md` with only the docs dependency group installed (`--no-install-project`),
so no wheel, base or payload, is ever needed for the site to build or deploy.

## Reporting bugs and requesting features

Use [GitHub Issues](https://github.com/AZX-PBC-OSS/tors/issues). For security
vulnerabilities, see [`SECURITY.md`](SECURITY.md) instead: please don't file those as
public issues.
