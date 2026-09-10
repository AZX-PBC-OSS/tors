"""The real-document corpus gate: tors's whole surface proven over
text extracted from real document formats: markdown, RTF, DOCX, XLSX, PDF
(``tests/documents.py`` generates them byte-deterministically and extracts
them with stdlib-only readers), plus the raw-bytes surface over the
document bytes themselves.

What this gate buys, per format family:

- extract → transform: the extraction feeds every str-in surface
  (``normalize``: the messy-whitespace stages fire on real document
  shapes; ``html_unescape`` on entities pasted into documents;
  ``finalize``'s hash; segmentation; ``find_patterns``; ``replace_many``
  redaction; both diff spellings between two document versions).
- raw bytes → the bytes surface: real document bytes are arbitrary binary
  (zips, compressed streams) or UTF-8 text depending on format, so the
  gate exercises ``utf8_is_valid``/``decode_utf8`` exactly where each
  format lands (differentially against the stdlib, which settles
  validity without a hand-pin), the ``b64`` content-addressing pair over
  the real bytes, and ``finalize_utf8``'s replace flavor over corrupted
  bytes.

No exact-difflib assertion appears for the document-version diffs: the
documented boundary-class divergences (difflib's non-minimal anchored
splits on repeated-flank contexts) apply to line and char diffs alike, so
the gate pins structural validity plus one hand-derived golden.

One cell is marked ``timing`` (the marker lane registered in
pyproject.toml): a wall band for normalize+finalize over ~2 MiB built from
the extracted markdown text, the sanity band that document-scale text
stays in tors's measured class, not a race.
"""

from __future__ import annotations

import base64
import hashlib
import html
import time
from pathlib import Path

import pytest

import tors
from documents import (
    _FILE_NAMES,
    CORPUS,
    CORPUS_DIR,
    EXPECTED_TEXT,
    MARKDOWN_NORMALIZED,
    MARKDOWN_TEXT,
    MARKDOWN_UNESCAPED,
    extract,
    write_corpus,
)
from reference import (
    assert_opcodes_are_valid,
    reference_find_patterns,
    reference_replace_many,
)

_KINDS = sorted(CORPUS)


# --- the corpus itself: determinism + extraction -------------------------------------


def test_the_committed_corpus_is_exactly_the_generator_output() -> None:
    """The five files under tests/corpus/ are ``corpus``; regenerating is
    byte-identical (fixed zip metadata, programmatically computed xref), so
    the committed artifacts can never silently drift from the generator."""
    assert sorted(path.name for path in CORPUS_DIR.iterdir()) == sorted(_FILE_NAMES.values()), (
        "tests/corpus/ contents drifted; run documents.write_corpus()"
    )
    files = {path.name: path.read_bytes() for path in CORPUS_DIR.iterdir()}
    for kind, name in _FILE_NAMES.items():
        assert files[name] == CORPUS[kind], f"{name} drifted from the generator"


def test_regeneration_into_a_fresh_directory_is_byte_identical(tmp_path: Path) -> None:
    """The determinism proof: writing the corpus anywhere else produces the
    same bytes (no timestamps, no platform-dependent zip fields)."""
    write_corpus(tmp_path)
    for kind, name in _FILE_NAMES.items():
        assert (tmp_path / name).read_bytes() == CORPUS[kind]


@pytest.mark.parametrize("kind", _KINDS)
def test_extraction_reproduces_the_expected_text(kind: str) -> None:
    """Every stdlib reader extracts its document to the text constant both
    the generator and this expectation derive from, the cross-pin that
    keeps generator and extractor aligned with each other."""
    assert extract(kind) == EXPECTED_TEXT[kind]


# --- the text surface over extracted text -------------------------------------------


def test_normalize_cleans_the_real_markdown_shapes() -> None:
    """The headline pin: the markdown text carries every messy shape the
    pipeline exists for (CRLF endings, [ \\t] before newlines, a 4-newline
    blank run, trailing NBSP) and ``normalize`` returns exactly the
    hand-derived cleaned spelling."""
    assert tors.normalize(MARKDOWN_TEXT) == MARKDOWN_NORMALIZED


@pytest.mark.parametrize("kind", _KINDS)
def test_normalize_is_idempotent_on_every_extracted_text(kind: str) -> None:
    """The pipeline's fixed-point property on real document text."""
    once = tors.normalize(EXPECTED_TEXT[kind])
    assert tors.normalize(once) == once


def test_html_unescape_decodes_entities_in_document_text() -> None:
    """Entities pasted into markdown decode exactly, and identically to the
    stdlib (the differential)."""
    assert tors.html_unescape(MARKDOWN_TEXT) == MARKDOWN_UNESCAPED
    assert tors.html_unescape(MARKDOWN_TEXT) == html.unescape(MARKDOWN_TEXT)


@pytest.mark.parametrize("kind", _KINDS)
def test_finalize_matches_normalize_then_hashlib(kind: str) -> None:
    """The one-call hash contract over real document text: byte-identical to
    the two-step expression it replaces."""
    text = EXPECTED_TEXT[kind]
    normalized = tors.normalize(text)
    assert tors.finalize(text) == (
        normalized,
        hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    )


@pytest.mark.parametrize("kind", _KINDS)
def test_segmentation_is_structurally_sound_on_document_text(kind: str) -> None:
    """Grapheme/word/sentence segmentation over extracted text: bounds cover
    the text, slices join back to it, segments are non-empty, and the
    streaming iterator yields exactly the list API's sequence."""
    text = EXPECTED_TEXT[kind]
    words = tors.word_bounds(text)
    assert words[0][0] == 0 and words[-1][1] == len(text)
    assert all(a < b for a, b in words)
    assert "".join(text[a:b] for a, b in words) == text
    assert list(tors.word_bounds_iter(text)) == words
    sentences = tors.sentence_bounds(text)
    assert sentences[0][0] == 0 and sentences[-1][1] == len(text)
    assert all(a < b for a, b in sentences)
    assert "".join(text[a:b] for a, b in sentences) == text
    assert list(tors.sentence_bounds_iter(text)) == sentences
    assert tors.grapheme_count(text) >= len(words)


def test_sentence_bounds_pin_real_document_rows() -> None:
    """The hand-derived sentence rows for the markdown text's head, each
    cited to its rule: the heading plus its line separator is one sentence
    (sb4 breaks after the separator, so the trailing \\n rides the sentence,
    the same attachment quirk word spaces show), the blank line that
    follows is a standalone segment (sb4 again; nothing joins two
    separators), and the intro sentence after it is one unit ending at its
    own separator. Derived from the constants: the heading is 26 chars +
    '\\n' = (0, 27); the blank line's '\\n' = (27, 28); the intro (113
    chars) + '\\n' = (28, 142)."""
    assert tors.sentence_bounds(MARKDOWN_TEXT)[:3] == [(0, 27), (27, 28), (28, 142)]


def test_find_patterns_over_document_text() -> None:
    """A document-vocabulary scan over the extracted markdown: exact
    brute-oracle agreement, plus the hand-derived content pins: "quarterly"
    (lowercase) matches exactly once, in the intro (the heading's
    "Quarterly" is capital-Q and case sensitivity is the contract, derived:
    28 chars of heading+blank line + "The " = offset 32); "Tokyo" never
    matches (the corpus's CJK is 東京, not the ASCII spelling); "figures"
    occurs exactly twice (the entity sentence and the figure row)."""
    text = MARKDOWN_TEXT
    patterns = ["quarterly", "bushing", "Tokyo", "figures"]
    matches = tors.find_patterns(patterns, text)
    assert matches == reference_find_patterns(patterns, text)
    assert [m for m in matches if m[2] == 0] == [(32, 41, 0)]
    assert all(m[2] != 2 for m in matches)
    assert len([m for m in matches if m[2] == 3]) == 2


def test_replace_many_redacts_document_text() -> None:
    """The redaction scenario over real document text: a pii/terminology map
    applied in one native call: exact expected output and brute-oracle
    agreement, plus the identity lane (a map whose keys never occur returns
    the same object)."""
    text = MARKDOWN_TEXT
    redactions = {"quarterly": "monthly", "bushing": "insulator", "outage": "incident"}
    assert tors.replace_many(text, redactions) == reference_replace_many(text, redactions)
    assert tors.replace_many(text, redactions) == text.replace("quarterly", "monthly").replace(
        "bushing", "insulator"
    ).replace("outage", "incident")
    assert tors.replace_many(text, {"nonexistent-key": "x"}) is text


def test_diff_opcodes_between_document_versions() -> None:
    """Two derived document versions (scattered word swaps, an inserted and a
    deleted line) diffed at both granularities: char-level validity over the
    strings, line-level validity over the splitlines lists, plus one pinned
    golden line-level case. Exact difflib equality is not
    asserted (the documented boundary-class divergences)."""
    v1 = MARKDOWN_TEXT
    v2 = (
        MARKDOWN_TEXT.replace("quarterly", "monthly")
        .replace("bushing torque", "insulator torque")
        .replace("Interval drift", "Interval variance")
        + "\nEscalation note: adjusted after the outage."
    )
    assert_opcodes_are_valid(v1, v2, tors.diff_opcodes(v1, v2))
    v1_lines = v1.splitlines(keepends=True)
    v2_lines = v2.splitlines(keepends=True)
    line_ops = tors.diff_opcodes_lines(v1, v2)
    assert_opcodes_are_valid(v1_lines, v2_lines, line_ops)
    # The golden: a one-line replacement in a two-line document (both
    # operands are two lines, so the op list ends at the replace).
    a = "alpha\nbeta\n"
    b = "alpha\ngamma\n"
    assert tors.diff_opcodes_lines(a, b) == [("equal", 0, 1, 0, 1), ("replace", 1, 2, 1, 2)]


def test_diff_opcodes_lines_reconstructs_both_sides() -> None:
    """The line-level reconstruction property: applying the ops as line
    slices rebuilds each operand exactly (splitlines keepends shape)."""
    v1 = MARKDOWN_TEXT
    v2 = MARKDOWN_TEXT.replace("quarterly", "monthly") + "extra line\n"
    a_lines = v1.splitlines(keepends=True)
    b_lines = v2.splitlines(keepends=True)
    rebuilt_a: list[str] = []
    rebuilt_b: list[str] = []
    for tag, i1, i2, j1, j2 in tors.diff_opcodes_lines(v1, v2):
        if tag in ("equal", "replace", "delete"):
            rebuilt_a.extend(a_lines[i1:i2])
        if tag in ("equal", "replace", "insert"):
            rebuilt_b.extend(b_lines[j1:j2])
    assert rebuilt_a == a_lines
    assert rebuilt_b == b_lines


# --- the raw-bytes surface over document bytes ---------------------------------------


@pytest.mark.parametrize("kind", _KINDS)
def test_utf8_is_valid_agrees_with_the_stdlib_decoder(kind: str) -> None:
    """Real document bytes land on both sides of the validity line (the
    md/rtf are valid UTF-8/ASCII text; the zips and the PDF's compressed
    stream are arbitrary bytes), settled differentially, not hand-pinned."""
    raw = CORPUS[kind]
    try:
        raw.decode("utf-8")
        expected = True
    except UnicodeDecodeError:
        expected = False
    assert tors.utf8_is_valid(raw) is expected


@pytest.mark.parametrize("kind", _KINDS)
def test_b64_content_addressing_round_trips_real_document_bytes(kind: str) -> None:
    """The content-addressing pair over real bytes: encode parity with the
    stdlib and a decode round trip (the OCR/dedup gate shape)."""
    raw = CORPUS[kind]
    encoded = tors.b64_encode_bytes(raw)
    assert encoded == base64.b64encode(raw).decode("ascii")
    assert tors.b64_decode(encoded) == raw


def test_decode_utf8_over_valid_and_corrupted_document_bytes() -> None:
    """The markdown bytes decode byte-exactly; a corrupted copy (invalid
    bytes injected at a known offset) raises the very UnicodeDecodeError the
    stdlib raises (type + message) on strict and matches replace exactly."""
    raw = CORPUS["md"]
    assert tors.decode_utf8(raw) == raw.decode("utf-8")
    corrupted = raw[:200] + b"\xff\xed\xa0\x80" + raw[200:]
    with pytest.raises(UnicodeDecodeError) as tors_exc:
        tors.decode_utf8(corrupted)
    with pytest.raises(UnicodeDecodeError) as stdlib_exc:
        corrupted.decode("utf-8")
    assert type(tors_exc.value) is type(stdlib_exc.value)
    assert str(tors_exc.value) == str(stdlib_exc.value)
    assert tors.decode_utf8(corrupted, errors="replace") == corrupted.decode(
        "utf-8", errors="replace"
    )


def test_finalize_utf8_replace_flows_corrupted_document_bytes() -> None:
    """The one-call decode+normalize+hash over corrupted document bytes: the
    replace flavor's U+FFFD substitutions flow through the pipeline and the
    hash exactly as the two-step expression computes them."""
    raw = CORPUS["md"]
    corrupted = raw[:200] + b"\xff\xed\xa0\x80" + raw[200:]
    decoded = corrupted.decode("utf-8", errors="replace")
    normalized = tors.normalize(decoded)
    assert tors.finalize_utf8(corrupted, errors="replace") == (
        normalized,
        hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    )


# --- one timing-lane wall cell --------------------------------------------------------


def _min_wall_ms(op, samples: int = 3, warmup: int = 1) -> float:
    for _ in range(warmup):
        op()
    best = float("inf")
    for _ in range(samples):
        started = time.perf_counter()
        op()
        best = min(best, time.perf_counter() - started)
    return best * 1000.0


@pytest.mark.timing
def test_document_scale_normalize_finalize_wall_band() -> None:
    """The sanity band: normalize+finalize over ~2 MiB built by repeating
    the extracted markdown text stays comfortably inside the corpus-class
    wall tors measures on this shape (measured ~20-30ms each on the dev box
    at ambient load ~2; the 400ms ceiling is the ~15x sanity margin that
    only an accidental quadratic or per-char regression would blow through).
    The timing lane's slow, load-sensitive cells live behind the marker,
    CI runs this once on the 3.12 leg."""
    corpus_text = extract("md")
    text = corpus_text * (2 * 1024 * 1024 // len(corpus_text.encode("utf-8")) + 1)
    assert len(text.encode("utf-8")) >= 2 * 1024 * 1024
    assert _min_wall_ms(lambda: tors.normalize(text)) < 400.0
    assert _min_wall_ms(lambda: tors.finalize(text)) < 400.0
