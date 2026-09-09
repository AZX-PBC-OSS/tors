"""The contract gates for the cross-format validation suite: what every
engine/backend must hold to, checked identically wherever they run (the
pytest suite in tests/test_documents_engines.py and the report runner in
tools/eval_documents.py).

The central idea is ALIGNMENT: the generator emitted known content units
into a real document (tests/docgen.py's ``GroundTruth``, or the fixed
fixtures' oracle constants); an engine's output — markdown OR plain text —
is normalized by :func:`align` (entity unescape, backslash-escape removal,
marker stripping, whitespace collapse — stdlib-only, deliberately NOT the
code under test) and every unit must appear. Parse quality is therefore
measured, not eyeballed: a dropped heading, a swallowed table cell, a
mangled link label is a number and a name.

Beyond alignment, the gates hold the structural contract (headings render
as heading LINES, links carry their URLs, text output carries no markdown
syntax) and the no-leak contract (HTML head noise never reaches the body).
Every checker returns a list of failures — empty is pass — so the runner
can print them and pytest can assert them wholesale.
"""

from __future__ import annotations

import html as html_module
import re
from dataclasses import dataclass

# --- the alignment normalizer -------------------------------------------------

_BACKSLASH_ESCAPE = re.compile(r"\\([!\"#$%&'()*+,./:;<=>?@\[\\\]^_`{|}~-])")
_EMPHASIS_PAIRS = re.compile(r"(\*\*|__|~~|`)")
# Link syntax unwrapped to its LABEL only: the url is markdown machinery,
# not sentence content — a unit cannot know which url the engine will
# inline mid-sentence, and the url's PRESENCE is gated separately
# (check_markdown_structure). Same for the strip's plain-text spelling
# `label (url)` (_PAREN_URL, applied after this).
_LINK_SYNTAX = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_PAREN_URL = re.compile(r"\s*\((?:https?://|mailto:)[^)]*\)")
_TABLE_DELIMITER = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_WHITESPACE_RUN = re.compile(r"\s+")
# An unescaped <uri-or-email> is autolink SYNTAX (its text is the destination
# itself); a literal angle-bracket text arrives entity-escaped (&lt;…&gt;),
# which is why this runs BEFORE the entity fixpoint — the distinction is
# exactly what the ordering preserves.
_AUTO_LINK = re.compile(r"<(https?://[^\s<>]+|mailto:[^\s<>]+|[^\s<>@]+@[^\s<>@]+\.[^\s<>]+)>")


def align(text: str) -> str:
    """Normalize engine output (either mode) to comparable plain text:
    autolink brackets unwrapped, HTML entities unescaped to a fixpoint
    (anydoc HTML-escapes its markdown, so a literal ``&amp;`` in the
    document arrives as ``&amp;amp;`` — render-correct, but one unescape
    level deeper than the unit's own spelling; both sides reach the same
    fixpoint), markdown backslash escapes removed, link syntax unwrapped,
    emphasis markers and heading/list/quote/table syntax stripped,
    whitespace runs collapsed. Independent of the code under test on
    purpose — the oracle must not share a normalizer with the
    implementation."""
    unescaped = _AUTO_LINK.sub(r"\1", text)
    for _ in range(3):
        step = html_module.unescape(unescaped)
        if step == unescaped:
            break
        unescaped = step
    unescaped = _BACKSLASH_ESCAPE.sub(r"\1", unescaped)
    unescaped = _LINK_SYNTAX.sub(r"\1", unescaped)
    unescaped = _PAREN_URL.sub("", unescaped)
    unescaped = _EMPHASIS_PAIRS.sub(" ", unescaped)
    lines: list[str] = []
    for line in unescaped.splitlines():
        stripped = line.strip()
        stripped = re.sub(r"^#{1,6}\s+", "", stripped)
        stripped = re.sub(r"^>\s*", "", stripped)
        stripped = re.sub(r"^[-*+]\s+", "", stripped)
        stripped = re.sub(r"^\d+[.)]\s+", "", stripped)
        stripped = stripped.replace("|", " ")
        lines.append(stripped)
    return _WHITESPACE_RUN.sub(" ", "\n".join(lines)).strip()


def _contains(normalized_output: str, unit: str) -> bool:
    """Both sides through the SAME normalizer: the unit's own entity/markdown
    spelling (``&amp;``, ``\\*``, pipes-in-prose) aligns with the output's.
    Comparing a raw unit against normalized output could never match for
    units that CARRY those shapes as content — the exact defect the pdf/rtf
    seed-0 misses exposed (the engines emit such units verbatim; alignment
    must confirm that, not forbid it)."""
    return align(unit) in normalized_output


# Characters whose spelling legitimately varies between an emitted unit and
# the engine's output (backslash- or entity-escaped, marker-wrapped).
_ESCAPEABLE = re.compile(r"[&<>`\\*_\[\]()|~]")


def _raw_tier_failure(unit: str, collapsed_output: str) -> str | None:
    """The honesty tier on top of the aligned comparison: a word with no
    escapable characters has no legitimate alternative spelling, so it must
    appear in the output VERBATIM (whitespace-collapsed). Catches what the
    spelling-tolerant aligned comparison masks — an engine that entity-
    encodes plain text (``it&#39;s``), smart-quotes it, or otherwise mangles
    characters it had no reason to touch. Word-level on purpose: formatting
    the engine inserts AROUND words (autolink angle brackets, emphasis
    markers wrapping the word) must not fail the tier, only per-word text
    alteration can. Words carrying escapables are exempt (their spelling
    legitimately varies with the engine's escaping)."""
    for word in _WHITESPACE_RUN.sub(" ", unit).split(" "):
        if word and not _ESCAPEABLE.search(word) and word not in collapsed_output:
            return f"plain word mangled or missing verbatim: {word!r} (unit {unit!r})"
    return None


# --- gate checkers ------------------------------------------------------------


@dataclass
class GateResult:
    """One conversion's verdict: the failure list (empty is pass) plus the
    alignment score (units found / units total) for the report runner."""

    failures: list[str]
    found: int
    total: int

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def alignment(self) -> float:
        return self.found / self.total if self.total else 1.0


def check_alignment(truth, output: str) -> GateResult:
    """Every emitted content unit must appear in the normalized output:
    headings' text, paragraphs, list items, non-empty table cells, link
    labels, speaker notes, code lines. Bold spans are content too (the
    words must survive even if the marker does not). Both sides normalize
    through :func:`align` — the unit's spelling of entity/markdown shapes
    and the output's are the same text."""
    units: list[tuple[str, str]] = []
    for _level, heading in truth.headings:
        units.append(("heading", heading))
    units.extend(("paragraph", p) for p in truth.paragraphs)
    units.extend(("list item", item) for item in truth.list_items)
    units.extend(("table cell", cell) for cell in truth.table_cells)
    units.extend(("link label", label) for label in truth.link_labels)
    units.extend(("note", note) for note in truth.notes)
    units.extend(("code line", line) for line in truth.code_lines)
    units.extend(("bold span", span) for span in truth.bold_spans)
    normalized = align(output)
    collapsed_output = _WHITESPACE_RUN.sub(" ", output)
    failures: list[str] = []
    found = 0
    for kind, unit in units:
        # BOTH sides through align(), as the docstring has always claimed:
        # a unit carrying entities (``&amp;``), markdown markers
        # (backticks), or pipes must be normalized exactly like the output
        # it is matched against, or it can never align by construction
        # (measured 2026-09: the engines emitted the units verbatim while
        # the raw-unit comparison failed).
        if _contains(normalized, align(unit)):
            found += 1
            # Aligned-passed units face the raw tier: plain words must be
            # verbatim — the aligned comparison alone is spelling-tolerant
            # enough to mask plain-text mangling (see _raw_tier_failure).
            raw_failure = _raw_tier_failure(unit, collapsed_output)
            if raw_failure is not None:
                failures.append(f"{kind}: {raw_failure}")
        else:
            failures.append(f"{kind} missing from output: {unit!r}")
    return GateResult(failures=failures, found=found, total=len(units))


def check_markdown_structure(truth, markdown: str, *, gate_links: bool) -> list[str]:
    """The structural contract for markdown output: each heading renders as
    a heading LINE (a ``#``-prefixed line containing the text), and — where
    the lane gates links — every emitted link appears as
    ``[label](url)``. Heading-line matching applies to the structured
    formats (docx styles, odt outline levels, html h1-h6, pptx titles);
    the PDF's font-size heuristic is a fixed-fixture gate, not a fuzz gate,
    because detection thresholds are engine policy, not contract."""
    failures: list[str] = []
    if truth.kind in {"docx", "odt", "html", "pptx"}:
        for _level, heading in truth.headings:
            wanted = _WHITESPACE_RUN.sub(" ", html_module.unescape(heading)).strip()
            if not any(
                re.match(r"^#{1,6}\s", line)
                and wanted in _WHITESPACE_RUN.sub(" ", html_module.unescape(line))
                for line in markdown.splitlines()
            ):
                failures.append(f"heading not rendered as a heading line: {heading!r}")
    if gate_links:
        for _label, url in zip(truth.link_labels, truth.link_urls, strict=False):
            if f"]({url})" not in markdown and f"<{url}>" not in markdown:
                failures.append(f"link URL missing from markdown: {url!r}")
    return failures


def check_text_clean(text: str) -> list[str]:
    """The plain-text contract: no markdown SYNTAX SURVIVES into to_text —
    heading lines, table delimiter rows, code fences. Deliberately NOT
    flagged: stray link/emphasis character shapes (``](``, ``**``) — a
    document whose literal source text contains those shapes produces them
    in honest plain text (the engines escape markdown-significant source
    characters, and the strip un-escapes them back), so the shape alone
    cannot distinguish a strip bug from real content. The line-level syntax
    shapes below have no such literal-content false-positive mode: a source
    line genuinely starting with ``#`` or ``|---|`` or a fence is exotic,
    and the strip removes exactly these."""
    failures: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if re.match(r"^#{1,6}\s", stripped):
            failures.append(f"heading marker survived into text: {line!r}")
        if _TABLE_DELIMITER.match(stripped):
            failures.append(f"table delimiter row survived into text: {line!r}")
        if stripped.startswith("```") or stripped.startswith("~~~"):
            failures.append(f"code fence survived into text: {line!r}")
    return failures


def check_no_leak(output: str) -> list[str]:
    """The no-leak contract (the html lane): head noise must never reach the
    body output, in either mode, on any backend. Two layers: the fixtures'
    own noise markers (script bodies, style rules, and the title texts of
    both lanes' fixtures — the fuzz documents' and the fixed matrix's), and
    the generic head/script/style tag shapes — no engine may pass those
    through as markup, whatever the fixture says."""
    failures: list[str] = []
    for junk in ("tracking junk", "color: red", "Junk Title", "Page Title"):
        if junk in output:
            failures.append(f"head noise leaked into output: {junk!r}")
    for tag in ("<script", "</script", "<style", "</style", "<title", "</title"):
        if tag in output:
            failures.append(f"head/script/style markup leaked into output: {tag!r}")
    return failures
