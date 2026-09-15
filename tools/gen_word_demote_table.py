#!/usr/bin/env python3
"""Regenerate WORD_DEMOTE_RANGES in src/scrub_impl.rs (step 2 of 2).

Step 1: build and run tools/enum.rs against the pinned rustc:
    rustc -O tools/enum.rs -o /tmp/enum && /tmp/enum > /tmp/rust_alnum.txt

Step 2: run this script:
    python3 tools/gen_word_demote_table.py /tmp/rust_alnum.txt

What it does: intersects the Rust alphanumeric set with the running
interpreter's `re` \\w set (Unicode 16.0.0 on CPython 3.14), computes the
demote set (Rust-alnum MINUS re-\\w, plus `_` handling per the doc comment),
folds it to sorted non-adjacent (lo, hi) ranges, and diffs against the
committed WORD_DEMOTE_RANGES table. Exits nonzero on any drift, printing
the Rust replacement block.

Pinned inputs the committed table was generated against (2026-09-12):
    PINNED_RUSTC = 1.98.1, PINNED_UNIDATA = 16.0.0, CPython 3.14.
On an older interpreter the only possible divergence is on codepoints the
older UCD leaves unassigned; a rustc that adopts a newer UCD can classify
newly-assigned Other_Alphabetic codepoints the table does not list. Either
case is a deliberate re-sync (update the pins in this file AND the version
test in tests/test_scrub_log_text_parity.py together), never a silent edit.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from pathlib import Path

PINNED_RUSTC = "1.98.1"
PINNED_UNIDATA = "16.0.0"

WORD = re.compile(r"\w")


def fold_to_ranges(points: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for cp in points:
        if ranges and cp == ranges[-1][1] + 1:
            lo, _ = ranges[-1]
            ranges[-1] = (lo, cp)
        else:
            ranges.append((cp, cp))
    return ranges


def load_committed_table(src: Path) -> list[tuple[int, int]]:
    text = src.read_text()
    start = text.index("const WORD_DEMOTE_RANGES")
    end = text.index("];", start)
    block = text[start:end]
    out: list[tuple[int, int]] = []
    for line in block.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("(0x"):
            lo, hi = line.strip("()").split(",")
            out.append((int(lo.strip(), 16), int(hi.strip(), 16)))
    return out


def main() -> int:
    rustc = subprocess.run(
        ["rustc", "--version"], capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()
    print(f"rustc: {rustc}")
    print(f"unidata: {unicodedata.unidata_version} (python {sys.version.split()[0]})")
    if PINNED_RUSTC not in rustc:
        print(
            f"NOTE: running rustc ({rustc}) differs from the pinned "
            f"{PINNED_RUSTC}: the table below is still computed, but a "
            "version drift means newly-assigned codepoints may differ; "
            "treat any diff as a re-sync candidate."
        )
    if unicodedata.unidata_version != PINNED_UNIDATA:
        print(
            f"NOTE: running UCD ({unicodedata.unidata_version}) differs from "
            f"the pinned {PINNED_UNIDATA}: divergences confined to newly-"
            "assigned codepoints are expected."
        )
    if len(sys.argv) != 2:
        print("usage: gen_word_demote_table.py /tmp/rust_alnum.txt", file=sys.stderr)
        return 2
    rust = frozenset(int(line) for line in Path(sys.argv[1]).read_text().split())
    demote = sorted((rust | {0x5F}) - {cp for cp in range(0x110000) if WORD.match(chr(cp))})
    ranges = fold_to_ranges(demote)
    print(f"demote codepoints: {len(demote)} in {len(ranges)} ranges")
    try:
        committed = load_committed_table(Path("src/scrub_impl.rs"))
    except (ValueError, FileNotFoundError) as exc:
        print(f"cannot load committed table: {exc}", file=sys.stderr)
        return 2
    if ranges != committed:
        print("DRIFT: recomputed ranges differ from the committed table.")
        print("Replacement block:")
        for lo, hi in ranges:
            print(f"    (0x{lo:04X}, 0x{hi:04X}),")
        return 1
    print("OK: recomputed ranges match the committed table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
