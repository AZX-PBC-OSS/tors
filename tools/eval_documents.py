"""The cross-format eval runner: the engines matrix as a REPORT — every
fixture (the committed corpus and, with --fuzz, seeded randomized
documents) through every offered backend and both output modes, timed, and
held to the same gates as the pytest suite.

Usage (repo root):
    uv run --no-sync python tools/eval_documents.py            # fixed corpus
    uv run --no-sync python tools/eval_documents.py --fuzz 10  # + 10 seeds/format
    uv run --no-sync python tools/eval_documents.py --json     # machine-readable
    uv run --no-sync python tools/eval_documents.py --kind csv --backend auto

Exit status: non-zero when any MUST-WORK cell (the ``auto`` routing lane)
fails its gates — the runner is a gate, not just a report. Cells a forced
backend does not offer record ``not offered`` (that is routing policy, not
a failure); cells that answer are held to the full contract.

The gates are the suite's own (tests/documents_gates.py) so the report and
the pytest lane can never disagree on what "good" means.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))

from docgen import FILE_EXTENSIONS, GENERATORS  # noqa: E402
from documents_gates import (  # noqa: E402
    check_alignment,
    check_markdown_structure,
    check_no_leak,
    check_text_clean,
)

from documents import ENGINES_CORPUS, ENGINES_FILENAMES  # noqa: E402

try:
    from tors_documents import NeedsOcrError, to_markdown, to_text
except ImportError as exc:
    raise SystemExit(
        "documents surface not built — build the payload first "
        "(uv sync --reinstall-package tors-documents)"
    ) from exc

BACKENDS = ("auto", "oxide", "anydoc")


def _gate_cell(kind: str, backend: str, mode: str, path: str, truth=None) -> dict:
    convert = to_markdown if mode == "markdown" else to_text
    started = time.perf_counter()
    try:
        if backend == "auto":
            resolved, output = convert(path)
        else:
            resolved, output = convert(path, backend=backend)
    except (ValueError, OSError, NeedsOcrError) as exc:
        # Every row carries the full shape the printer and --json readers
        # expect — a partial dict here crashes the report on the first
        # not-offered cell (measured: csv_rich x oxide x markdown, sorted
        # order, before any gated cell printed).
        return {
            "kind": kind,
            "backend": backend,
            "mode": mode,
            "status": "not offered",
            "detail": type(exc).__name__,
            "ms": 0.0,
            "chars": 0,
            "resolved": None,
            "alignment": None,
        }
    ms = (time.perf_counter() - started) * 1000
    failures: list[str] = []
    if kind in {"html", "html_rich"}:
        failures += check_no_leak(output)
    if truth is not None:
        result = check_alignment(truth, output)
        failures += result.failures
        alignment = result.alignment
        if mode == "markdown":
            gate_links = kind in {"docx", "html"}
            failures += check_markdown_structure(truth, output, gate_links=gate_links)
    else:
        alignment = None
    if mode == "text":
        failures += check_text_clean(output)
    return {
        "kind": kind,
        "backend": backend,
        "mode": mode,
        "status": "pass" if not failures else "FAIL",
        "detail": "; ".join(failures[:3]),
        "ms": round(ms, 2),
        "chars": len(output),
        "resolved": resolved,
        "alignment": alignment,
    }


def _oracle_truth(kind: str):
    """A GroundTruth-shaped oracle for the fixed fixtures, from the suite's
    oracle lines (alignment score for the report)."""
    from test_documents_engines import _oracle_lines

    class _Truth:
        def __init__(self, lines: list[str]) -> None:
            self.kind = kind
            self.headings = []
            self.paragraphs = [line for line in lines if line.strip()]
            self.list_items, self.table_cells = [], []
            self.link_labels, self.link_urls, self.bold_spans = [], [], []
            self.code_lines, self.notes = [], []
            self.pages, self.blank_pages = 0, ()

    return _Truth(_oracle_lines(kind))


def run(
    fuzz_seeds: int,
    as_json: bool,
    *,
    kind_filter: str | None = None,
    backend_filter: str | None = None,
    mode_filter: str | None = None,
) -> int:
    import tempfile

    def _selected(kind: str, backend: str, mode: str) -> bool:
        return (
            (kind_filter is None or kind == kind_filter)
            and (backend_filter is None or backend == backend_filter)
            and (mode_filter is None or mode == mode_filter)
        )

    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="tors-eval-") as tmp:
        root = Path(tmp)
        for kind in sorted(ENGINES_CORPUS):
            if kind_filter is not None and kind != kind_filter:
                continue
            path = root / ENGINES_FILENAMES[kind]
            path.write_bytes(ENGINES_CORPUS[kind])
            truth = _oracle_truth(kind)
            for backend in BACKENDS:
                for mode in ("markdown", "text"):
                    if _selected(kind, backend, mode):
                        rows.append(_gate_cell(kind, backend, mode, str(path), truth))
        for kind, generator in sorted(GENERATORS.items()):
            if kind_filter is not None and kind != kind_filter:
                continue
            for seed in range(fuzz_seeds):
                raw, truth = generator(seed)
                # The fuzz file's extension must be the KIND's (sniff wins on
                # content, but the name should not lie about what ran).
                path = root / f"fuzz_{kind}_{seed}{FILE_EXTENSIONS.get(kind, '.bin')}"
                path.write_bytes(raw)
                for mode in ("markdown", "text"):
                    if _selected(kind, "auto", mode):
                        rows.append(_gate_cell(kind, "auto", mode, str(path), truth))

    must_work = [row for row in rows if row["backend"] == "auto" and row["status"] == "FAIL"]
    if as_json:
        print(json.dumps({"rows": rows, "must_work_failures": len(must_work)}, indent=2))
    else:
        print(
            f"{'kind':<13} {'backend':<8} {'mode':<9} {'status':<12} {'ms':>8} {'chars':>8}  detail"
        )
        for row in rows:
            alignment = f" align={row['alignment']:.0%}" if row.get("alignment") is not None else ""
            print(
                f"{row['kind']:<13} {row['backend']:<8} {row['mode']:<9} "
                f"{row['status'] + alignment:<12}"
                f" {row['ms']:>8.1f} {row['chars']:>8}  {row['detail']}"
            )
        print(f"\n{len(rows)} cells; must-work failures: {len(must_work)}")
        if not rows and kind_filter in GENERATORS and fuzz_seeds == 0:
            print(f"note: {kind_filter!r} is a fuzz-only kind — pass --fuzz N to run its seeds")
    return 1 if must_work else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fuzz", type=int, default=0, help="randomized seeds per format")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--kind",
        help="restrict the matrix to one kind (a corpus kind or a fuzz generator kind)",
    )
    parser.add_argument("--backend", choices=BACKENDS, help="restrict to one backend")
    parser.add_argument("--mode", choices=("markdown", "text"), help="restrict to one mode")
    args = parser.parse_args()
    if args.kind is not None and args.kind not in {*ENGINES_CORPUS, *GENERATORS}:
        known = ", ".join(sorted({*ENGINES_CORPUS, *GENERATORS}))
        parser.error(f"unknown kind {args.kind!r}; known kinds: {known}")
    return run(
        args.fuzz,
        args.json,
        kind_filter=args.kind,
        backend_filter=args.backend,
        mode_filter=args.mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
