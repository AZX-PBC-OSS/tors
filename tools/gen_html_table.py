"""Regenerate ``src/html_table.rs`` from the RUNNING interpreter's ``html`` module.

Reads ``html.entities.html5``, ``html._invalid_charrefs`` and
``html._invalid_codepoints`` (private but stable since CPython 3.8; the exact
data ``html.unescape`` itself consults) and emits the three statics plus a
header recording the emitted counts. Deterministic: same interpreter tables →
byte-identical file (sorted input, stable formatting, ``\\u{...}``-escaped
values so the output is pure ASCII). Run via ``make gen-html-table``.

The generated counts are pinned crate-side by the length-pin test in
``src/html_impl.rs``, and every entry of all three tables is re-verified per CI
leg against the running interpreter by ``tests/test_html_unescape.py``
(``TestFullHtml5Table`` + ``TestNumericSets``). If a future Python grows or
changes a table, THOSE fail loudly; re-run this generator (against that
interpreter) to bring the file up to date, and update the pin test's numbers
as part of that change.
"""

from __future__ import annotations

import html
import html.entities
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "src" / "html_table.rs"


def rust_escape(s: str) -> str:
    out: list[str] = []
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif 0x20 <= o < 0x7F:
            out.append(ch)
        else:
            out.append(f"\\u{{{o:x}}}")
    return '"' + "".join(out) + '"'


def main() -> None:
    entities = sorted(html.entities.html5.items())
    with_semi = sum(1 for k, _ in entities if k.endswith(";"))
    without = len(entities) - with_semi
    invalid_charrefs = sorted(html._invalid_charrefs.items())
    invalid_codepoints = sorted(html._invalid_codepoints)

    lines: list[str] = []
    lines.append(
        f"""//! The HTML5 named-entity table and CPython's numeric-reference
//! classification data: generated, not hand-written.
//!
//! Provenance: generated from ``html.entities.html5``,
//! ``html._invalid_charrefs`` and ``html._invalid_codepoints`` of the running
//! interpreter (all three themselves generated from the WHATWG
//! ``entities.json`` and the HTML5 spec's numeric tables, and identical
//! across CPython 3.8+) by ``tools/gen_html_table.py``; ``make
//! gen-html-table`` regenerates this file against whatever interpreter runs
//! it. The counts below are pinned crate-side by the length-pin test in
//! ``src/html_impl.rs`` (tied to these header numbers), so a regeneration
//! against a changed table is visible even without a Python interpreter; and
//! the Python-side contract gate (tests/test_html_unescape.py) re-verifies,
//! per CI leg against the RUNNING interpreter's own tables, EVERY entry of
//! all three: the {len(entities)} named entities (``TestFullHtml5Table``) and
//! both numeric sets in both ``&#N;`` and ``&#xN;`` spellings
//! (``TestNumericSets``). Those are the pins that actually matter: if a
//! future Python grows or changes a table, that gate fails loudly and this
//! file must be regenerated (``make gen-html-table``), with the pin test's
//! numbers updated as part of the same change.
//!
//! Layout: ``HTML5_ENTITIES`` is sorted by key for binary search ({len(entities)}
//! entries: {with_semi} with-semicolon + {without} legacy without-semicolon);
//! ``INVALID_CHARREFS`` is the {len(invalid_charrefs)}-entry Windows-1252/special remap,
//! consulted BEFORE the surrogate/range guard exactly as
//! ``html._replace_charref`` does; ``INVALID_CODEPOINTS`` is the {len(invalid_codepoints)}-member
//! set mapping to the EMPTY string. Pure data, no code.

    /// Every HTML5 named character reference: ``(name, replacement)``; `name`
    /// INCLUDES the semicolon where the with-semicolon spelling exists. Sorted
    /// by name for binary search.
pub static HTML5_ENTITIES: &[(&str, &str)] = &["""
    )
    for k, v in entities:
        lines.append(f"    ({rust_escape(k)}, {rust_escape(v)}),")
    lines.append("];")
    lines.append(
        """
/// CPython's ``html._invalid_charrefs``: numeric references remapped to these
/// replacements before any range check (the Windows-1252 remap, NUL/CR
/// special cases). Sorted by codepoint.
pub static INVALID_CHARREFS: &[(u32, &str)] = &["""
    )
    for cp, v in invalid_charrefs:
        lines.append(f"    (0x{cp:x}, {rust_escape(v)}),")
    lines.append("];")
    lines.append(
        """
    /// CPython's ``html._invalid_codepoints``: numeric references in this set map
    /// to the EMPTY string (the HTML5 "not allowed" list: C1 controls,
    /// noncharacters). Sorted.
pub static INVALID_CODEPOINTS: &[u32] = &["""
    )
    for i in range(0, len(invalid_codepoints), 8):
        chunk = invalid_codepoints[i : i + 8]
        lines.append("    " + " ".join(f"0x{cp:x}," for cp in chunk))
    lines.append("];")
    lines.append("")

    TARGET.write_text("\n".join(lines), encoding="utf-8")
    multi_char = sum(1 for _, v in entities if len(v) > 1)
    print(
        f"written {TARGET}: {len(entities)} entities ({with_semi}+{without}, "
        f"{multi_char} multi-char values), {len(invalid_charrefs)} charrefs, "
        f"{len(invalid_codepoints)} codepoints"
    )


if __name__ == "__main__":
    main()
