"""Unit pins for the oracle library itself (tests/documents_gates.py):
every normalization the aligner performs, every checker's verdict, and the
raw-tier honesty rule: a unit with no escapable characters must appear in
the output verbatim (whitespace-collapsed), not only through the aligned
comparison. The gates are the suite's definition of "good"; these tests are
the gates' own contract. Pure Python, no payload needed.

The raw tier exists because the aligned comparison is deliberately
spelling-tolerant (both sides run through :func:`align`, whose fixpoint
entity-unescape accepts any escape depth). That tolerance is right for
units that carry escapable shapes as content, but for plain units it can
mask an engine that HTML-escapes or mangles text it had no reason to
touch: a plain unit has no legitimate alternative spelling."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from documents_gates import (  # noqa: E402
    GateResult,
    align,
    check_alignment,
    check_markdown_structure,
    check_no_leak,
    check_text_clean,
)


class _Truth:
    """The GroundTruth shape the checkers read (attributes only)."""

    def __init__(
        self,
        kind: str = "docx",
        headings: list[tuple[int, str]] | None = None,
        paragraphs: list[str] | None = None,
        list_items: list[str] | None = None,
        table_cells: list[str] | None = None,
        link_labels: list[str] | None = None,
        link_urls: list[str] | None = None,
        bold_spans: list[str] | None = None,
        code_lines: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> None:
        self.kind = kind
        self.headings = headings or []
        self.paragraphs = paragraphs or []
        self.list_items = list_items or []
        self.table_cells = table_cells or []
        self.link_labels = link_labels or []
        self.link_urls = link_urls or []
        self.bold_spans = bold_spans or []
        self.code_lines = code_lines or []
        self.notes = notes or []


# --- align: every normalization it performs ------------------------------------


def test_entities_unescape_to_a_fixpoint_at_any_depth() -> None:
    assert align("&amp;") == "&"
    assert align("&amp;amp;") == "&"
    # named and numeric, same fixpoint
    assert align("&#38;") == "&"


def test_backslash_escapes_resolve_to_the_escaped_character() -> None:
    assert align(r"literal \* star") == "literal * star"
    assert align(r"pipes in a table \| cell") == "pipes in a table cell"


def test_link_syntax_keeps_the_label_and_drops_the_url() -> None:
    assert align("See the [handbook](https://x.example.com/t) now") == "See the handbook now"
    # a bare paren-URL (no label) disappears entirely
    assert align("go (https://x.example.com) now") == "go now"


def test_line_markers_strip_and_whitespace_collapses() -> None:
    markdown = "# Title\n\n> quoted\n\n- item one\n\n2) item two\n\n| a | b |\n"
    assert align(markdown) == "Title quoted item one item two a b"


def test_align_leaves_plain_text_verbatim() -> None:
    assert align("Draw oil samples quarterly.") == "Draw oil samples quarterly."


# --- check_alignment: verdicts and the raw tier --------------------------------

_PLAIN = re.compile(r"[&<>`\\*_\[\]()|~]")


def test_every_unit_kind_is_checked_and_counted() -> None:
    truth = _Truth(
        headings=[(1, "Title")],
        paragraphs=["Draw oil samples quarterly."],
        list_items=["first checkpoint"],
        table_cells=["T-101", "healthy"],
        link_labels=["handbook"],
        bold_spans=["torque tables"],
        code_lines=["let x = 1;"],
        notes=["speaker note"],
    )
    output = (
        "# Title\n\nDraw oil samples quarterly.\n\n- first checkpoint\n\n"
        "| Unit | Status |\n| --- | --- |\n| T-101 | healthy |\n\n"
        "See the [handbook](https://x.example.com/t) for **torque tables**.\n\n"
        "`let x = 1;`\n\nspeaker note\n"
    )
    result = check_alignment(truth, output)
    assert result == GateResult(failures=[], found=9, total=9)
    assert result.passed is True
    assert result.alignment == 1.0


def test_a_missing_unit_is_a_failure_that_names_it() -> None:
    truth = _Truth(paragraphs=["Draw oil samples quarterly."])
    result = check_alignment(truth, "unrelated text")
    assert not result.passed
    assert result.failures == ["paragraph missing from output: 'Draw oil samples quarterly.'"]
    assert result.alignment == 0.0


def test_escapable_carrying_units_align_through_their_own_spelling() -> None:
    # The unit's literal backtick/entity content is normalized on both
    # sides: the engine emitting it verbatim (or escaped) still aligns.
    truth = _Truth(paragraphs=["rate & yield", "literal `tick` mark"])
    output = "rate &amp; yield\n\nliteral `tick` mark\n"
    assert check_alignment(truth, output).passed


def test_a_plain_unit_must_appear_verbatim_not_only_aligned() -> None:
    # The masking case the raw tier exists for: the aligned comparison
    # unescapes entities on both sides, so an engine that entity-encoded
    # plain text would still "align". A plain word has no legitimate
    # alternative spelling: it must be present verbatim.
    truth = _Truth(paragraphs=["it's due for review"])
    encoded_output = "it&#39;s due for review"  # aligns, but is not the text
    result = check_alignment(truth, encoded_output)
    assert not result.passed, "plain-text encoding must not hide behind align()"
    assert any("verbatim" in f for f in result.failures)
    # the same text un-encoded passes both tiers
    assert check_alignment(truth, "it's due for review").passed
    # soft-wrapped output still satisfies the raw tier (ws-collapsed)
    assert check_alignment(truth, "it's due for\nreview.").passed


def test_the_raw_tier_tolerates_formatting_inserted_around_words() -> None:
    # Word-level on purpose: the engine may wrap plain words in autolink
    # angle brackets or emphasis markers: the words survive verbatim.
    truth = _Truth(paragraphs=["see https://example.com/x there", "urgent torque check"])
    output = "see <https://example.com/x> there\n\n**urgent** torque check\n"
    assert check_alignment(truth, output).passed


def test_alignment_score_counts_partial_finds() -> None:
    truth = _Truth(paragraphs=["one", "two", "three"])
    result = check_alignment(truth, "one and three only")
    assert result.found == 2 and result.total == 3
    assert result.alignment == 2 / 3


def test_a_truth_with_no_units_passes_trivially() -> None:
    result = check_alignment(_Truth(), "anything at all")
    assert result.passed and result.alignment == 1.0


# --- check_markdown_structure --------------------------------------------------


def test_structured_kinds_require_heading_lines() -> None:
    truth = _Truth(kind="docx", headings=[(1, "Annual Report")])
    assert check_markdown_structure(truth, "# Annual Report\n", gate_links=False) == []
    failures = check_markdown_structure(truth, "Annual Report\n", gate_links=False)
    assert len(failures) == 1 and "heading line" in failures[0]


def test_the_link_gate_holds_only_where_asked() -> None:
    truth = _Truth(kind="docx", headings=[], link_labels=["handbook"], link_urls=["https://x"])
    missing = check_markdown_structure(truth, "See the handbook.\n", gate_links=True)
    assert missing and "https://x" in missing[0]
    # either markdown spelling satisfies the gate: [label](url) or <url>
    linked = check_markdown_structure(truth, "See the [handbook](https://x).\n", gate_links=True)
    assert linked == []
    bare = check_markdown_structure(truth, "See <https://x>.\n", gate_links=True)
    assert bare == []
    # gate_links off: the plain label is enough
    assert check_markdown_structure(truth, "See the handbook.\n", gate_links=False) == []


def test_autolink_brackets_unwrap_but_literal_angle_text_stays() -> None:
    assert align("see <https://example.com/x> now") == "see https://example.com/x now"
    assert align("mail <user@example.com> now") == "mail user@example.com now"
    # entity-escaped angle brackets are literal text, not autolink syntax
    assert align("literal &lt;https://example.com/x&gt; text") == (
        "literal <https://example.com/x> text"
    )


# --- check_text_clean ----------------------------------------------------------


def test_text_mode_rejects_line_level_markdown_syntax() -> None:
    failures = check_text_clean("# Title\n\n- bullet\n\n| a | b |\n| --- | --- |\n")
    assert failures, "heading/bullet/table syntax must be flagged in text mode"


def test_text_mode_accepts_plain_prose() -> None:
    assert check_text_clean("Just prose, with a comma and a stop.\n") == []


# --- check_no_leak -------------------------------------------------------------


def test_html_output_must_not_leak_head_noise_or_script_markup() -> None:
    fixture_noise = "Intro\n\ncolor: red\nJunk Title\ntracking junk\n"
    assert check_no_leak(fixture_noise), "the fixtures' own noise markers must be flagged"
    generic = "Intro\n\n<script>alert(1)</script>\n<style>body{}</style>\n<title>t</title>\n"
    assert check_no_leak(generic), "generic script/style/title markup must be flagged"
    assert check_no_leak("Clean content only.\n") == []
