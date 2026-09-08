# tors: development convenience targets.
#
# Each target wraps the exact command CONTRIBUTING.md / README.md document (the
# one exception: gen-html-table's command is documented in the header of the
# file it regenerates, src/html_table.rs), so the Makefile cannot drift from
# the docs. `make lint` and `make test` are the ci.yml `lint` / `test` gates;
# `make check` is the full pre-PR gate.

# Route rustc through sccache when it is on PATH: the rustc-equivalent of ccache
# (ccache cannot cache rustc invocations). Measured on this crate: a post-
# `cargo clean` rebuild is fully served from cache (27/27 hits, 2.4s vs 8.3s cold),
# and dependency artifacts are shared across worktrees and target dirs: sccache
# keys on file contents, not paths, so only each worktree's own crate recompiles.
# This is the env form of cargo's `build.rustc-wrapper` setting, which a sibling
# repo documents in its .cargo/config.toml and this machine already sets globally
# in ~/.cargo/config.toml; carrying it here makes the repo behave the same for
# contributors without a global config. CI does NOT use sccache; see the Cache
# cargo step in .github/workflows/ci.yml for why rust-cache wins there.
ifneq ($(shell command -v sccache 2>/dev/null),)
export RUSTC_WRAPPER := sccache
endif

.DEFAULT_GOAL := check

.PHONY: check install dev lint test bench fmt deny gen-html-table gen-documents fuzz fuzz-quick

# The full CONTRIBUTING.md "Before you open a PR" gate (the standard
# aggregate-target pattern: one `check` target wrapping the whole list).
check: lint test

# Setup: build the extension + dev deps (CONTRIBUTING.md "Setup").
install:
	uv sync --locked

# Rebuild the extension after Rust edits. uv's wheel cache does not key on Rust
# sources, so a bare `uv sync` would leave pytest importing the stale .so.
# The find FIRST removes any non-abi3 extension shadows from python/tors/ (an
# early non-abi3 `maturin develop` leaves _tors.cpython-*.so files there, and
# CPython's extension-suffix order ranks a version-specific .so AHEAD of the
# abi3 one, the dev-loop landmine tests/conftest.py fails loudly on): after
# `make dev` exactly one fresh _tors.abi3.so remains, and plain `pytest`
# (without the preload runner) binds it.
dev:
	-find python/tors -maxdepth 1 -name '_tors*.so' ! -name '_tors.abi3.so' -delete
	uv sync --locked --reinstall-package tors

# The ci.yml lint job, verbatim: fmt gate, clippy in both feature configs
# (pyo3's cfg flags differ between them, so each pass surfaces lints in code the
# other never compiles), and ruff over the Python side (tests/, tools/,
# python/; config in pyproject.toml, the dev-group ruff runs it).
lint:
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings
	cargo clippy --all-targets --no-default-features -- -D warnings
	uv run --no-sync ruff check .

# Rust unit tests (extension-module off: it doesn't link libpython) and the pytest
# suite. Depends on `dev` so pytest always imports the extension built from the
# current tree, never a stale wheel. NOTE: unlike ci.yml's matrix legs (which
# deselect the timing lane and run it in ONE dedicated 3.12 step), this target
# runs EVERYTHING, the timing measurement cells included, because a local
# `make test` is the full pre-PR gate; the lane split exists to stop CI paying
# the slow, load-sensitive cells five times, not to thin the local run.
test: dev
	cargo test --no-default-features
	uv run --no-sync pytest -q

# Run the criterion suite (CI only compiles it, with --no-run): both benches.
bench:
	cargo bench --no-default-features

# Fix formatting (`lint` checks it).
fmt:
	cargo fmt

# The licensing gate (the ci.yml lint job's Cargo deny step): the
# permissive-only license allowlist, the advisory policy, and the ban
# policy live in deny.toml. Requires cargo-deny on PATH
# (`cargo install cargo-deny --locked`, ~1min).
deny:
	cargo deny check licenses advisories bans

# Regenerate src/html_table.rs from the RUNNING interpreter's html module
# (tools/gen_html_table.py; the provenance line lives in the generated header).
# cargo fmt after: rustfmt's array reflow is the file's canonical formatting
# (the generator emits a stable unpacked form that fmt then packs). If the
# regenerated counts differ from the header's pinned numbers, the crate-side
# length-pin test (src/html_impl.rs) fails; update it in the same change as
# this regeneration.
gen-html-table:
	uv run --no-sync python tools/gen_html_table.py
	cargo fmt

# cargo-fuzz (libFuzzer): raw adversarial byte/string input straight into the
# *_impl.rs cores, no pyo3 boundary — a different bug class than the
# hypothesis-based Python tests (panics, cross-function invariant breaks) over
# an input space they don't shape. Requires nightly (`rustup install nightly`)
# and `cargo install cargo-fuzz --locked`; the fuzz/ crate is intentionally
# NOT a workspace member (see fuzz/Cargo.toml's own header), so it never
# affects `cargo build`/`cargo test` at the repo root. `fuzz-quick` is the
# 30s-per-target smoke run suitable before a PR; a real fuzzing campaign
# (hours, one target, targeted at a specific area of suspicion) is
# `cargo +nightly fuzz run <target>` run directly, not through this target.
FUZZ_TARGETS := decode_utf8 decode_utf16 b64_decode html_unescape fence chunk_hierarchical normalize search segmentation diff phonetic bm25 tfidf truncate_ellipsis controls

fuzz-quick:
	@for t in $(FUZZ_TARGETS); do \
		echo "=== $$t (30s) ==="; \
		cargo +nightly fuzz run $$t -- -max_total_time=30 -close_fd_mask=3 || exit 1; \
	done

fuzz:
	@for t in $(FUZZ_TARGETS); do \
		echo "=== $$t ==="; \
		cargo +nightly fuzz run $$t || exit 1; \
	done

# Regenerate the deterministic document corpus (tests/corpus/): the five
# real-format files (md/rtf/docx/xlsx/pdf) are byte-deterministic, so this
# is idempotent: tests/test_documents.py pins the committed files to the
# generator's output and fails loudly if the two ever drift apart.
gen-documents:
	uv run --no-sync python -c "import sys; sys.path.insert(0, 'tests'); import documents; documents.write_corpus()"
