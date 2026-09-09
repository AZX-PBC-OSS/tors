"""The cross-format validation suite: every documents engine, held to one
contract over generated documents of every working format.

Three generation lanes, one set of gates (tests/documents_gates.py):

- FIXED FIXTURES (tests/documents.py's ``ENGINES_CORPUS``, committed under
  tests/engines_corpus/ and byte-pinned): one structurally rich document
  per format family, plus PDF structural variants (link annotation,
  two-column, oversize heading, blank page, two pages). These gate the
  MEASURED structural facts — style-based docx headings, link rendering,
  GFM table delimiter rows, column separation, NeedsOcr routing — the
  properties the engine choice was made on.
- FUZZ (tests/docgen.py, seeded): randomized real-writer documents — fpdf2,
  python-docx, openpyxl, python-pptx — with varying structure, columns,
  links, and formatting, each carrying its ground truth. These gate
  ALIGNMENT: every emitted content unit must survive conversion (markdown
  and plain-text modes), nothing fabricated, deterministic across calls.
- MUTATION: corrupted bytes (truncation, flips, splices) must end in a
  typed error or a usable result — never a crash, never a hang.

The API contract is gated where it exists today and SKIPPED WITH A LOUD
REASON where a surface has not landed yet: the skips disappear the moment the
implementation arrives, and the tests then hold it. The suite imports the payload surface
from ``tors_documents`` (the tors-documents wheel) with a fallback to
in-base wrappers — whichever wiring ships, the gates are the same.
"""

from __future__ import annotations

import asyncio
import random
import threading
import zipfile
from pathlib import Path

import pytest
from docgen import GENERATORS
from documents_gates import (
    align,
    check_alignment,
    check_markdown_structure,
    check_no_leak,
    check_text_clean,
)

from documents import (
    ENGINES_CORPUS,
    ENGINES_DIR,
    ENGINES_FILENAMES,
    PDF_LINK_LABEL,
    PDF_LINK_URL,
    RICH_H1,
    RICH_H2,
    RICH_LINK_LABEL,
    RICH_LINK_SENTENCE,
    RICH_LINK_URL,
    RICH_LIST_FLAT,
    RICH_LIST_NESTED,
    RICH_NOTES,
    RICH_TABLE_HEADER,
    RICH_TABLE_ROW,
    SLIDE1_BODY,
    SLIDE1_TITLE,
    SLIDE2_TITLE,
    generate_pdf_big,
)

try:  # the payload wheel (tors[documents])
    from tors_documents import (
        NeedsOcrError,
        pdf_classify,
        pdf_extract,
        pdf_page_count,
        sniff,
        to_markdown,
        to_text,
    )
except ImportError:  # pragma: no cover - the in-base wiring alternative
    try:
        from tors import (  # type: ignore[attr-defined]  # noqa: F401
            NeedsOcrError,
            pdf_classify,
            pdf_extract,
            pdf_page_count,
            sniff,
            to_markdown,
            to_text,
        )
    except ImportError:
        pytest.skip(
            "documents surface not built: install the payload wheel "
            "(uv sync --reinstall-package tors-documents, or maturin develop "
            "in tors-documents/) — the suite has nothing to validate",
            allow_module_level=True,
        )


SEEDS = (0, 1, 2, 3)
CORRUPTIONS_PER_FIXTURE = 4


def _materialize(tmp_path: Path, kind: str) -> str:
    path = tmp_path / ENGINES_FILENAMES[kind]
    path.write_bytes(ENGINES_CORPUS[kind])
    return str(path)


def _format_name(kind: str) -> str:
    """The format= spelling for a corpus kind: its fixture filename's
    extension (every kind's extension IS a name in the format vocabulary)."""
    return Path(ENGINES_FILENAMES[kind]).suffix.lstrip(".")


# --- the committed corpus pin -------------------------------------------------


def test_engines_corpus_files_are_pinned() -> None:
    """The committed copies under tests/engines_corpus/ are exactly
    ENGINES_CORPUS (the documents.py determinism contract extended to this
    matrix); regenerate with write_engines_corpus() after any generator
    edit."""
    missing = [
        kind for kind in ENGINES_CORPUS if not (ENGINES_DIR / ENGINES_FILENAMES[kind]).exists()
    ]
    assert not missing, f"uncommitted engines fixtures (run write_engines_corpus): {missing}"
    for kind, raw in ENGINES_CORPUS.items():
        committed = (ENGINES_DIR / ENGINES_FILENAMES[kind]).read_bytes()
        assert committed == raw, f"{kind}: committed fixture drifted from the generator"


# --- the fixed-fixture matrix -------------------------------------------------


@pytest.mark.parametrize("kind", sorted(ENGINES_CORPUS), ids=lambda kind: kind)
def test_auto_backend_converts_every_format_and_aligns(tmp_path: Path, kind: str) -> None:
    """The routing default must answer every format in the matrix, and its
    markdown must carry every oracle line (alignment) with no head-noise
    leak (the html fixture's title/style/script)."""
    path = _materialize(tmp_path, kind)
    resolved, markdown = to_markdown(path)
    assert resolved, "the resolved format name must be non-empty"
    failures = check_no_leak(markdown)
    assert not failures, failures
    for line in _oracle_lines(kind):
        assert align(line) in align(markdown), f"{kind}: oracle line missing: {line!r}"


def _oracle_lines(kind: str) -> list[str]:
    """The content every conversion of the fixed fixtures must carry: the
    corpus five carry EXPECTED_TEXT's lines (their own extraction oracle);
    the shared-vocabulary fixtures (docx_rich/odt_rich/html_rich — built
    from the RICH_* constants) carry that vocabulary plus their per-fixture
    extras; the remaining variants carry only their own lines (csv is
    tabular data, the pptx/pdf variants have no shared vocabulary)."""
    from documents import EXPECTED_TEXT

    if kind in EXPECTED_TEXT:
        return EXPECTED_TEXT[kind].splitlines()
    shared = [
        RICH_H1,
        RICH_H2,
        RICH_LINK_SENTENCE,
        *RICH_LIST_FLAT,
        RICH_LIST_NESTED,
        *RICH_TABLE_HEADER,
        *RICH_TABLE_ROW,
    ]
    per_kind = {
        "pdf_blank": [],
        "pdf_scanned": [],
        "pdf_mixed": ["first page line"],
        "pdf_twocol": ["LEFT-A first line", "RIGHT-B first line"],
        "pdf_heading": ["Revenue grew twelve percent."],
        "pdf_link": [PDF_LINK_LABEL],
        "pdf_two_page": ["first page line", "second page line"],
        "docx_rich": ["Closing note."],
        "pptx_rich": [SLIDE1_TITLE, *SLIDE1_BODY, SLIDE2_TITLE, RICH_NOTES],
        "odt_rich": ["Draw oil samples quarterly."],
        "csv_rich": ["T-101", "healthy", "T-102", "needs review"],
        "html_rich": [RICH_NOTES],
    }
    lines = per_kind[kind]
    if kind in {"docx_rich", "odt_rich", "html_rich"}:
        lines = shared + lines
    return lines


@pytest.mark.parametrize("kind", ["docx_rich", "xlsx", "pptx_rich"], ids=lambda kind: kind)
def test_the_oxide_forced_backend_converts_the_ooxml_family_and_aligns(
    tmp_path: Path, kind: str
) -> None:
    """The oxide family's WORKING overlap, content-gated: backend="oxide"
    forces office_oxide over the OOXML containers, and the auto lane's
    choice for the office family is anydoc — so without this pin the
    oxide lane's only pytest coverage is its refusals. The same alignment
    gate as the auto lane's matrix (every oracle line survives) plus the
    resolved name; the structural gates stay on the auto lane (lean by
    design — the overlap is content parity, not a second structure
    contract)."""
    path = _materialize(tmp_path, kind)
    resolved, markdown = to_markdown(path, backend="oxide")
    assert resolved == _format_name(kind)
    for line in _oracle_lines(kind):
        assert align(line) in align(markdown), f"{kind}: oracle line missing: {line!r}"


class TestStructureGates:
    """The measured structural facts per format family — the properties the
    engine stack was chosen on (see tors-core's documents_impl docs). The
    asserts are SEMANTIC (a heading LINE containing the text, the URL
    present, nesting indented), not exact-shape: any engine spelling that
    carries the semantics passes."""

    def test_docx_styles_render_as_heading_lines(self, tmp_path: Path) -> None:
        _, markdown = to_markdown(_materialize(tmp_path, "docx_rich"))
        lines = markdown.splitlines()
        assert any(line.lstrip().startswith("#") and RICH_H1 in line for line in lines), markdown
        assert any(line.lstrip().startswith("##") and RICH_H2 in line for line in lines), markdown

    def test_docx_hyperlink_carries_the_url(self, tmp_path: Path) -> None:
        _, markdown = to_markdown(_materialize(tmp_path, "docx_rich"))
        assert RICH_LINK_URL in markdown and RICH_LINK_LABEL in markdown

    def test_docx_nested_list_is_indented(self, tmp_path: Path) -> None:
        _, markdown = to_markdown(_materialize(tmp_path, "docx_rich"))
        nested = [line for line in markdown.splitlines() if RICH_LIST_NESTED in line]
        assert nested and nested[0][:1] in (" ", "\t"), markdown

    def test_tables_carry_the_delimiter_row(self, tmp_path: Path) -> None:
        """GFM tables need the |---| delimiter row to be tables at all."""
        for kind in ("docx_rich", "odt_rich", "csv_rich", "html_rich"):
            _, markdown = to_markdown(_materialize(tmp_path, kind))
            assert any(
                set(line.replace("|", "").replace(" ", "").replace(":", "")) == {"-"}
                for line in markdown.splitlines()
            ), f"{kind}: no GFM table delimiter row"

    def test_pptx_titles_and_speaker_notes_survive(self, tmp_path: Path) -> None:
        _, markdown = to_markdown(_materialize(tmp_path, "pptx_rich"))
        assert SLIDE1_TITLE in markdown and SLIDE2_TITLE in markdown
        assert RICH_NOTES in markdown

    def test_pdf_link_annotation_carries_the_url(self, tmp_path: Path) -> None:
        _, markdown = to_markdown(_materialize(tmp_path, "pdf_link"))
        assert PDF_LINK_URL in markdown and PDF_LINK_LABEL in markdown

    def test_pdf_two_column_layout_stays_separate_blocks(self, tmp_path: Path) -> None:
        """The measured pdf_oxide property (and the anydoc-pdf failure mode
        this pins against): columns must not interleave line-by-line."""
        _, markdown = to_markdown(_materialize(tmp_path, "pdf_twocol"))
        left = markdown.index("LEFT-A first line")
        right = markdown.index("RIGHT-B first line")
        assert left < right
        for line in markdown.splitlines():
            assert not ("LEFT-A" in line and "RIGHT-B" in line), line

    def test_pdf_oversize_font_detected_as_heading(self, tmp_path: Path) -> None:
        _, markdown = to_markdown(_materialize(tmp_path, "pdf_heading"))
        assert any(
            line.lstrip().startswith("#") and RICH_H1 in line for line in markdown.splitlines()
        )

    def test_classify_separates_empty_blank_from_scanned_image_pages(self, tmp_path: Path) -> None:
        """The classify seam's whole point: a contentless page is ``empty``
        (neither extract nor OCR — NOT in the OCR list); an image-only page
        is ``scanned`` (in the OCR list, 0-based); a mixed document names
        both kinds and lists only its image page."""
        blank = pdf_classify(_materialize(tmp_path, "pdf_blank"))
        assert blank.page_kinds == ["empty"]
        assert blank.pages_needing_ocr == []
        assert not blank.has_text

        scanned = pdf_classify(_materialize(tmp_path, "pdf_scanned"))
        assert scanned.page_kinds == ["scanned"]
        assert scanned.pages_needing_ocr == [0]
        assert scanned.image_only

        mixed = pdf_classify(_materialize(tmp_path, "pdf_mixed"))
        assert mixed.page_kinds == ["text", "scanned"]
        assert mixed.pages_needing_ocr == [1]
        assert mixed.has_text and not mixed.image_only

    def test_blank_and_scanned_routing_aligns_with_anydoc(self, tmp_path: Path) -> None:
        """Cross-engine routing alignment: both engines name the SAME pages
        as needing OCR, in the SAME 0-based convention (anydoc's 1-based
        numbers are re-based at the core's seam — one convention across the
        whole surface). (A blank/contentless page is the documented
        divergence: pdf_oxide says empty/not-OCR, anydoc says NeedsOcr;
        that difference is pinned in the blank test, not here.)"""
        scanned_path = _materialize(tmp_path, "pdf_scanned")
        classification = pdf_classify(scanned_path)
        with pytest.raises(NeedsOcrError) as raised:
            to_markdown(scanned_path, backend="anydoc")
        assert raised.value.pages == classification.pages_needing_ocr

        mixed_path = _materialize(tmp_path, "pdf_mixed")
        classification = pdf_classify(mixed_path)
        with pytest.raises(NeedsOcrError) as raised:
            to_markdown(mixed_path, backend="anydoc")
        assert raised.value.pages == classification.pages_needing_ocr

    def test_blank_pdf_reads_empty_and_anydoc_flags_ocr(self, tmp_path: Path) -> None:
        """The contentless-page routing case: the pdf_oxide path returns
        empty output (the CALLER routes — classify says ``empty``, neither
        extract nor OCR); the anydoc backend raises the typed NeedsOcr
        signal (the documented cross-engine divergence, pinned as a fact)."""
        path = _materialize(tmp_path, "pdf_blank")
        pages, markdown = pdf_extract(path)
        assert pages == [""]
        assert markdown.strip() == ""
        with pytest.raises(NeedsOcrError) as raised:
            to_markdown(path, backend="anydoc")
        assert raised.value.pages == [0]
        assert raised.value.page_count == 1

    def test_pdf_two_pages_come_back_in_order(self, tmp_path: Path) -> None:
        path = _materialize(tmp_path, "pdf_two_page")
        assert pdf_page_count(path) == 2
        pages, markdown = pdf_extract(path)
        assert pages == ["first page line", "second page line"]

    def test_pdf_backend_choice_is_real(self, tmp_path: Path) -> None:
        """Both PDF engines answer; the forced-anydoc spelling must convert
        (the overlap is exactly PDF), and a forced oxide backend on a format
        its lane cannot read must refuse cleanly, never silently fall
        back."""
        path = _materialize(tmp_path, "pdf")
        resolved, markdown = to_markdown(path, backend="anydoc")
        assert resolved
        assert markdown.strip()
        with pytest.raises(ValueError):
            to_markdown(_materialize(tmp_path, "odt_rich"), backend="oxide")

    def test_tsv_is_supported_when_the_vocabulary_names_it(self, tmp_path: Path) -> None:
        """TSV: the delimiter-separated family's other half. anydoc's csv
        parser is delimiter-separated, so the engine lane's one-line
        vocabulary mapping (``tsv`` → the csv kind) is all that stands
        between here and support. This gate SKIPS WITH THIS REASON until
        the name resolves, then holds it to the same table contract."""
        path = tmp_path / "engines_units.tsv"
        path.write_bytes(b"unit\tstatus\nT-101\thealthy\nT-102\tneeds review\n")
        try:
            resolved, markdown = to_markdown(str(path), format="tsv")
        except ValueError as exc:
            pytest.skip(
                f"format name 'tsv' not yet in the vocabulary ({exc}); "
                "the engine lane's mapping lands it"
            )
        assert resolved in {"tsv", "csv"}
        assert "T-101" in markdown and "healthy" in markdown
        assert any(
            set(line.replace("|", "").replace(" ", "").replace(":", "")) == {"-"}
            for line in markdown.splitlines()
        )


class TestTextMode:
    """to_text: the same content, no markdown syntax."""

    @pytest.mark.parametrize("kind", sorted(ENGINES_CORPUS), ids=lambda kind: kind)
    def test_text_carries_content_without_markdown_syntax(self, tmp_path: Path, kind: str) -> None:
        path = _materialize(tmp_path, kind)
        resolved, text = to_text(path)
        assert resolved
        for line in _oracle_lines(kind):
            assert align(line) in align(text), f"{kind}: oracle line missing from text: {line!r}"
        assert not check_no_leak(text)
        failures = check_text_clean(text)
        assert not failures, failures


# --- the fuzz lane --------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"seed{seed}")
@pytest.mark.parametrize("kind", sorted(GENERATORS), ids=lambda kind: kind)
def test_random_documents_align_in_both_modes(tmp_path: Path, kind: str, seed: int) -> None:
    """The alignment gate over library-written documents: every emitted
    content unit survives both conversions; markdown renders structured
    formats' headings as heading lines and (docx/html) carries link URLs;
    the html lane never leaks head noise; conversion is deterministic."""
    raw, truth = GENERATORS[kind](seed)
    from docgen import FILE_EXTENSIONS

    path = tmp_path / f"generated_{kind}_{seed}{FILE_EXTENSIONS[kind]}"
    path.write_bytes(raw)

    resolved, markdown = to_markdown(str(path))
    assert resolved
    result = check_alignment(truth, markdown)
    assert not result.failures, result.failures
    structure = check_markdown_structure(truth, markdown, gate_links=kind in {"docx", "html"})
    assert not structure, structure
    if kind == "html":
        assert not check_no_leak(markdown)

    _, text = to_text(str(path))
    text_result = check_alignment(truth, text)
    assert not text_result.failures, text_result.failures
    assert not check_text_clean(text)

    _, again = to_markdown(str(path))
    assert again == markdown, "conversion must be deterministic"


# --- the mutation lane -----------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(ENGINES_CORPUS), ids=lambda kind: kind)
def test_corrupted_documents_fail_typed_or_succeed(tmp_path: Path, kind: str) -> None:
    """Robustness: truncation, byte flips, and splices must end in a typed
    error (ValueError/OSError/NeedsOcrError) or a usable result — never an
    untyped crash. Content is NOT gated on corrupted input (partial output
    from a damaged file is legitimate); only the failure TYPE is."""
    rng = random.Random(20260909)
    raw = ENGINES_CORPUS[kind]
    for corruption in range(CORRUPTIONS_PER_FIXTURE):
        damaged = _corrupt(rng, raw, corruption)
        path = tmp_path / f"corrupt_{kind}_{corruption}{Path(ENGINES_FILENAMES[kind]).suffix}"
        path.write_bytes(damaged)
        for convert in (to_markdown, to_text):
            try:
                convert(str(path))
            except (ValueError, OSError, NeedsOcrError):
                pass  # the typed refusal is the contract
            except Exception as exc:  # noqa: BLE001 -- the point of the gate
                pytest.fail(f"{kind} corruption {corruption}: untyped {type(exc).__name__}: {exc}")


def _corrupt(rng: random.Random, raw: bytes, corruption: int) -> bytes:
    if corruption == 0:  # truncate at a random interior point
        return raw[: max(1, rng.randrange(1, len(raw)))]
    if corruption == 1:  # flip one random byte
        data = bytearray(raw)
        data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
        return bytes(data)
    if corruption == 2:  # splice: head of this document, tail of another
        other = ENGINES_CORPUS["pdf_two_page"]
        return raw[: len(raw) // 2] + other[len(other) // 2 :]
    return raw + b"\x00garbage-tail\x00"  # trailing junk


# --- the API contract ------------------------------------------------------------


class TestApiContract:
    def test_missing_file_raises_oserror(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            to_markdown(str(tmp_path / "nope.pdf"))

    def test_garbage_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "garbage.pdf"
        path.write_bytes(b"not a pdf at all")
        with pytest.raises(ValueError):
            to_markdown(str(path))

    def test_explicit_unknown_format_name_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "engines_report.pdf"
        path.write_bytes(ENGINES_CORPUS["pdf"])
        with pytest.raises(ValueError, match="(?i)format"):
            to_markdown(str(path), format="no-such-format")

    def test_bad_backend_name_refused_before_any_work(self, tmp_path: Path) -> None:
        path = tmp_path / "engines_report.pdf"
        path.write_bytes(ENGINES_CORPUS["pdf"])
        with pytest.raises(ValueError, match="(?i)backend"):
            to_markdown(str(path), backend="turbo")

    def test_io_failures_raise_a_typed_oserror_naming_the_cause(self, tmp_path: Path) -> None:
        """A directory standing where a file belongs, and a path that
        resolves to nothing, are IO failures: OSError (never ValueError,
        never a crash), with the cause named in the message, in both
        modes. The landed shape (the io::Error → errno mapping the
        engine-lane handoff delivered): the errno-matched OSError SUBCLASS
        — IsADirectoryError, FileNotFoundError — so a caller's ``except
        FileNotFoundError`` catches, with the errno text carried in the
        message; ``.errno`` itself is None on the mapped subclass (the
        mapping constructs the subclass without the errno attribute — the
        message, not ``.errno``, is where the cause lives)."""
        cases = [
            (str(tmp_path), "Is a directory"),
            (str(tmp_path / "nope.pdf"), "No such file"),
        ]
        for path, cause in cases:
            for convert in (to_markdown, to_text):
                with pytest.raises(OSError, match=cause) as raised:
                    convert(path)
                assert raised.value.errno is None, cause


def test_sniff_exposes_magic_byte_detection(tmp_path: Path) -> None:
    """API contract: sniff(data) names the format from content markers alone
    (the extension-independent signal), None for the signature-less and the
    unrecognized. SKIPPED WITH THIS REASON until the surface lands."""
    import tors_documents as payload

    sniff = getattr(payload, "sniff", None) or getattr(payload, "sniff_format", None)
    if sniff is None:
        pytest.skip("sniff not yet exposed by the payload (spec'd API; lands with the wiring)")
    expectations = {
        "pdf": "pdf",
        "docx_rich": "docx",
        "xlsx": "xlsx",
        "pptx_rich": "pptx",
        "odt_rich": "odt",
        "rtf": "rtf",
        "html_rich": "html",
    }
    for kind, expected in expectations.items():
        assert sniff(ENGINES_CORPUS[kind]) == expected, kind
    assert sniff(b"plain text, no markers") is None
    assert sniff(b"plain prose, no markers and no delimiter structure") is None
    # CSV is signature-less by anydoc's own docs, but the engine's sniffer
    # may heuristically name delimiter-separated bytes as csv — either
    # answer is its documented behavior; the CONTRACT is that sniff never
    # names a format the bytes are not (pinned by every other row above).


def test_truncated_magic_bytes_sniff_to_none() -> None:
    """A truncated prefix of a magic is not the format: the empty bytes,
    a two-byte ZIP head, a three-byte PDF head, and a JSON fragment all
    sniff to None — never a guess from a partial signature. The whole
    markers those prefixes truncate are the contrast rows: sniff answers
    them, so the None answers are about truncation, not about sniff never
    answering."""
    assert sniff(b"") is None
    for prefix in (b"PK", b"%PD", b'{"he'):
        assert sniff(prefix) is None, prefix
    assert sniff(b"%PDF-") == "pdf"
    assert sniff(b"{\\rtf1") == "rtf"


def test_a_generic_zip_is_not_an_office_document(tmp_path: Path) -> None:
    """The OOXML formats are ZIP packages, but a ZIP package is not
    therefore an OOXML document: a generic zip (one .txt member, no
    [Content_Types].xml) must not be claimed as docx/xlsx/pptx — sniff
    says None — and converting it is a typed ValueError under its own
    extension AND under a lying .docx extension (the extension fallback
    engages, then the docx parse refuses cleanly: the fallback path, the
    one no corpus kind exercises because every corpus kind's content is
    marker-named)."""
    path = tmp_path / "plain.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("notes.txt", "just a text file inside a zip\n")
    raw = path.read_bytes()
    assert sniff(raw) is None
    with pytest.raises(ValueError):
        to_markdown(str(path))
    mislabeled = tmp_path / "plain.docx"
    mislabeled.write_bytes(raw)
    with pytest.raises(ValueError):
        to_markdown(str(mislabeled))


# The formats whose EMPTY bytes are themselves a valid empty document, so
# an empty file converts (to empty output) instead of refusing: the two
# text formats with no required structure (measured 2026-09-09; every
# other kind refuses — see the empty-file gate below).
_EMPTY_READS_AS_A_DOCUMENT = {"csv", "html"}


@pytest.mark.parametrize("kind", sorted(ENGINES_CORPUS), ids=lambda kind: kind)
def test_an_empty_file_is_a_typed_refusal_or_an_empty_document(tmp_path: Path, kind: str) -> None:
    """A 0-byte file never crashes and never raises OSError: the
    structured formats (pdf/docx/xlsx/pptx/odt/rtf) refuse it with a
    typed ValueError naming the malformed container; csv and html — whose
    empty bytes ARE a valid empty document — convert to exactly empty
    output (the typed current behavior, pinned as a fact; if the contract
    wants a refusal there too, that is an engine-lane decision and this
    gate is where it lands). Both modes, every corpus kind."""
    path = tmp_path / f"empty_{ENGINES_FILENAMES[kind]}"
    path.write_bytes(b"")
    name = _format_name(kind)
    if name in _EMPTY_READS_AS_A_DOCUMENT:
        for convert in (to_markdown, to_text):
            resolved, output = convert(str(path), format=name)
            assert resolved == name
            assert output == ""
    else:
        for convert in (to_markdown, to_text):
            with pytest.raises(ValueError):
                convert(str(path), format=name)


@pytest.mark.parametrize(
    ("backend", "kind"),
    [
        ("oxide", "rtf"),
        ("oxide", "odt_rich"),
        ("oxide", "csv_rich"),
        ("oxide", "html_rich"),
        ("anydoc", "html_rich"),
    ],
    ids=["oxide-rtf", "oxide-odt", "oxide-csv", "oxide-html", "anydoc-html"],
)
def test_a_forced_backend_refuses_a_format_it_cannot_read(
    tmp_path: Path, backend: str, kind: str
) -> None:
    """A forced backend never silently falls back: every (backend, format)
    pair outside the engine's lane refuses with a ValueError naming the
    incompatibility — oxide on the anydoc-lane formats (rtf/odt/csv) and
    on html (the html engine's own lane), anydoc on html. (The odt pair
    is also pinned by test_pdf_backend_choice_is_real; the WORKING
    overlaps — anydoc on pdf, oxide on the OOXML family — by the same
    test, the structure gates, and the forced-oxide alignment matrix.)"""
    path = _materialize(tmp_path, kind)
    with pytest.raises(ValueError, match="cannot read"):
        to_markdown(path, backend=backend)


@pytest.mark.parametrize("kind", sorted(ENGINES_CORPUS), ids=lambda kind: kind)
def test_an_explicitly_named_format_matches_the_sniffed_conversion(
    tmp_path: Path, kind: str
) -> None:
    """format= names the format instead of sniffing it — a routing
    instruction, never a different conversion: for every corpus kind the
    resolved name is the one asked for, and the output is byte-identical
    to the sniffed default's (which itself resolved the same name — the
    sniffed and explicit paths agree on every kind in the matrix)."""
    path = _materialize(tmp_path, kind)
    name = _format_name(kind)
    sniffed = to_markdown(path)
    assert sniffed[0] == name
    resolved, markdown = to_markdown(path, format=name)
    assert resolved == name
    assert markdown == sniffed[1]


def test_pdf_blank_page_routing_aligns_on_the_oxide_probe(tmp_path: Path) -> None:
    """Routing alignment for GENERATED PDFs (fpdf2's blank pages carry an
    empty content stream, unlike the fixed fixtures' no-stream pages — a
    shape anydoc treats differently, pinned by the fixed-fixture lane): the
    pdf_oxide probe's empty-page list must be exactly the generated blank
    pages. The anydoc cross-check stays on the canonical fixtures, where
    the shapes are controlled."""
    from docgen import random_pdf

    checked = 0
    for seed in range(24):
        raw, truth = random_pdf(seed)
        if not truth.blank_pages:
            continue
        path = tmp_path / f"blank_{seed}.pdf"
        path.write_bytes(raw)
        pages, _ = pdf_extract(str(path))
        empty = tuple(index for index, page in enumerate(pages) if not page.strip())
        assert empty == truth.blank_pages, (
            f"seed {seed}: probe empty pages {empty} != {truth.blank_pages}"
        )
        checked += 1
        if checked >= 3:
            return
    pytest.skip("no generated PDF carried a blank page in 24 seeds (raise the seed budget)")


def test_page_range_extraction(tmp_path: Path) -> None:
    """API contract: pages= selects 0-based PDF page indices, deduplicated,
    in document order, byte-identical to the whole document when the range
    is every page; non-PDF formats and the anydoc backend refuse it. SKIPPED
    WITH THIS REASON until the parameter lands."""
    import inspect

    if "pages" not in inspect.signature(to_markdown).parameters:
        pytest.skip("pages= not yet exposed by the payload (spec'd API; lands with the wiring)")
    path = _materialize(tmp_path, "pdf_two_page")
    _, whole = to_markdown(path)
    _, second_only = to_markdown(path, pages=[1])
    assert "second page line" in second_only
    assert "first page line" not in second_only
    _, both = to_markdown(path, pages=[1, 0])
    assert both == whole, "a full range must be byte-identical to pages=None"
    with pytest.raises(ValueError):
        to_markdown(_materialize(tmp_path, "docx_rich"), pages=[0])
    with pytest.raises(ValueError):
        to_markdown(path, backend="anydoc", pages=[0])
    with pytest.raises(ValueError):
        to_markdown(path, pages=[17])


def test_page_selection_is_deduped_into_document_order(tmp_path: Path) -> None:
    """pages= is a SET of page positions, not a sequence of instructions:
    duplicates collapse ([0, 0] is page 0 once) and the caller's ordering
    is ignored — [1, 0] and [0, 1] both come back as the document's own
    page order, byte-identical to the whole-document conversion, with
    page 0's content first."""
    path = _materialize(tmp_path, "pdf_two_page")
    _, whole = to_markdown(path)
    for selection in ([1, 0], [0, 1]):
        _, selected = to_markdown(path, pages=selection)
        assert selected == whole, selection
    _, reversed_call = to_markdown(path, pages=[1, 0])
    assert reversed_call.index("first page line") < reversed_call.index("second page line")
    assert to_markdown(path, pages=[0, 0])[1] == to_markdown(path, pages=[0])[1]


def test_an_empty_or_negative_page_selection_is_a_typed_refusal(tmp_path: Path) -> None:
    """pages=[] names no page (never an empty-output success: pass None
    for the whole document) and a negative index names no page either (the
    0-based contract has no Python-style negative indexing) — both are
    typed ValueError refusals. (Out-of-range indices are pinned by
    test_page_range_extraction's [17].)"""
    path = _materialize(tmp_path, "pdf_two_page")
    with pytest.raises(ValueError):
        to_markdown(path, pages=[])
    with pytest.raises(ValueError):
        to_markdown(path, pages=[-1])


def test_the_int_and_range_page_spellings_select_the_same_pages(tmp_path: Path) -> None:
    """The three accepted spellings — one int, the explicit list, the
    half-open (start, stop) tuple — select the same pages: 1 and (1, 2)
    both yield exactly page 1 (the tuple's stop is exclusive, Python
    convention), and the full-range tuple (0, 2) is byte-identical to the
    whole document."""
    path = _materialize(tmp_path, "pdf_two_page")
    _, whole = to_markdown(path)
    _, page_one = to_markdown(path, pages=[1])
    assert "second page line" in page_one
    assert "first page line" not in page_one
    assert to_markdown(path, pages=1)[1] == page_one
    assert to_markdown(path, pages=(1, 2))[1] == page_one
    assert to_markdown(path, pages=(0, 2))[1] == whole


def test_text_mode_holds_the_same_page_selection_contract(tmp_path: Path) -> None:
    """to_text's pages= carries the same selection semantics as
    to_markdown's, not a markdown-only parameter: one page selects that
    page's text alone, a full range is byte-identical to the
    whole-document text, duplicates dedup away, and the non-PDF and
    anydoc-lane refusals match to_markdown's exactly."""
    path = _materialize(tmp_path, "pdf_two_page")
    _, whole = to_text(path)
    _, second_only = to_text(path, pages=[1])
    assert "second page line" in second_only
    assert "first page line" not in second_only
    assert to_text(path, pages=[1, 0])[1] == whole
    assert to_text(path, pages=[0, 0])[1] == to_text(path, pages=[0])[1]
    with pytest.raises(ValueError):
        to_text(_materialize(tmp_path, "docx_rich"), pages=[0])
    with pytest.raises(ValueError):
        to_text(path, backend="anydoc", pages=[0])


@pytest.mark.parametrize("kind", sorted(ENGINES_CORPUS), ids=lambda kind: kind)
def test_content_markers_win_over_the_files_extension(tmp_path: Path, kind: str) -> None:
    """A mislabeled file still converts correctly (the native contract's
    promise): with format=None the content markers decide — every corpus
    kind's bytes under a deliberately WRONG extension resolve to their
    TRUE format (the extension never wins) and carry the same oracle
    content as the correctly-named file. The extension is only the
    fallback for content no marker names — that path is pinned by the
    generic-zip gate, not by any corpus kind (every kind's content is
    marker-named, csv included: the delimiter heuristic names it)."""
    name = _format_name(kind)
    wrong_extension = ".docx" if name == "pdf" else ".pdf"
    path = tmp_path / f"mislabeled_{kind}{wrong_extension}"
    path.write_bytes(ENGINES_CORPUS[kind])
    resolved, markdown = to_markdown(str(path))
    assert resolved == name, "the extension must not win over the content markers"
    for line in _oracle_lines(kind):
        assert align(line) in align(markdown), f"{kind}: oracle line missing: {line!r}"


# --- GIL band and concurrency (the payload-level claims) -------------------------


def test_to_markdown_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    tmp_path: Path,
) -> None:
    """The GIL-release claim as a test, the suite's shared methodology: the
    whole read+sniff+convert pass runs under py.detach, so a to_thread wrap
    keeps a 10ms heartbeat alive through a ~470KB document conversion (the
    official pdf_oxide wheel measures GIL-held per call instead — the
    hazard this payload exists to remove)."""
    from test_gil_release import _assert_loop_stays_responsive

    path = tmp_path / "big.pdf"
    path.write_bytes(generate_pdf_big(6000))

    def op() -> None:
        to_markdown(str(path))

    asyncio.run(_assert_loop_stays_responsive(lambda: asyncio.to_thread(op)))


def test_concurrent_conversions_are_identical(tmp_path: Path) -> None:
    """8 threads, one conversion each over three formats — with the GIL
    released these run natively in parallel (the pdfium hazard this surface
    exists to be safe for); every thread's answer must equal the
    single-threaded answer byte-for-byte."""
    paths = [
        _materialize(tmp_path, kind) for kind in ("pdf_two_page", "docx_rich", "html_rich")
    ] * 2  # 6 workers over 3 documents; plus 2 more below for 8
    paths += [_materialize(tmp_path, "xlsx"), _materialize(tmp_path, "odt_rich")]
    expected = [to_markdown(path) for path in paths]
    results: list[tuple[str, str]] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(len(paths))

    def worker(path: str) -> None:
        try:
            barrier.wait()
            results.append(to_markdown(path))
        except BaseException as exc:  # noqa: BLE001 -- recorded, asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(path,)) for path in paths]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert sorted(results) == sorted(expected)


def test_to_text_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    tmp_path: Path,
) -> None:
    """The GIL-release claim holds in the TEXT lane too, the same shared
    methodology: the convert+strip pass runs under the same py.detach band
    as to_markdown's (measured 2026-09-09: ~77ms wall over the ~470KB
    document, against to_markdown's ~140ms — the strip rides inside the
    released band), so a to_thread wrap keeps the 10ms heartbeat alive in
    both modes."""
    from test_gil_release import _assert_loop_stays_responsive

    path = tmp_path / "big.pdf"
    path.write_bytes(generate_pdf_big(6000))

    def op() -> None:
        to_text(str(path))

    asyncio.run(_assert_loop_stays_responsive(lambda: asyncio.to_thread(op)))


@pytest.mark.parametrize("kind", ("pdf_two_page", "docx_rich", "html_rich"), ids=lambda kind: kind)
def test_concurrent_text_conversions_are_identical(tmp_path: Path, kind: str) -> None:
    """8 threads, one to_text each over the SAME file — with the GIL
    released these run natively in parallel in the text lane exactly as
    in the markdown lane, and every thread's answer must equal the
    single-threaded answer byte-for-byte."""
    path = _materialize(tmp_path, kind)
    expected = to_text(path)
    results: list[tuple[str, str]] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        try:
            barrier.wait()
            results.append(to_text(path))
        except BaseException as exc:  # noqa: BLE001 -- recorded, asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(results) == 8
    assert all(result == expected for result in results)


# --- payload adversarial surface (red-team lane) ----------------------------------
#
# One clearly-marked section, appended by the payload red-team lane
# (2026-09): every adversarial input probed against the installed wheel,
# pinned as a gate — the binding layer's refusal surface (wrong types ->
# TypeError, wrong values/shapes -> ValueError, environment failures ->
# OSError, each naming the argument it refused), the typed-enum identity
# contract, and the aio parity. Findings that measured as BUGS on the
# pre-fix wheel are named on the tests that pin the fix: pages=True
# silently selected page 1 (bools launder through pyo3's i64 extraction),
# float indices surfaced as a bare "'float' object cannot be interpreted
# as an integer" naming no argument (and 1.0 as a scalar raised
# ValueError), str/dict pages= raised ValueError instead of TypeError,
# non-str format=/backend= raised a pyo3 TypeError naming no argument,
# and a lone-surrogate path raised a bare UnicodeEncodeError that read
# like an engine bug.


class TestPayloadAdversarialSurface:
    """The binding layer's refusal surface and the typed surface's
    identity, held to Python's exception convention: type problems are
    TypeErrors, value/shape problems are ValueErrors, environment
    failures are OSErrors — and every message names the argument."""

    def test_pages_booleans_are_rejected_not_laundered_to_indices(self, tmp_path: Path) -> None:
        """pages=True on the pre-fix wheel converted page 1 silently
        (pyo3's i64 extraction accepts bool because isinstance(True, int)
        holds): booleans must be a TypeError naming them, in every
        position they can appear — the scalar, a list item, a tuple item —
        and in both conversion modes (one shared parse_pages spine)."""
        path = _materialize(tmp_path, "pdf_two_page")
        for convert in (to_markdown, to_text):
            for bad in (True, False, [True, 0], [1, False], (True, 2), (0, False)):
                with pytest.raises(TypeError, match="(?i)booleans are not page indices"):
                    convert(path, pages=bad)

    def test_pages_floats_are_type_errors_naming_pages(self, tmp_path: Path) -> None:
        """A float index is a type problem (Python's own convention:
        range(1.0) and seq[1.0] raise TypeError), and the refusal must
        name pages= and repr the value — the pre-fix wheel's bare
        "'float' object cannot be interpreted as an integer" (list/tuple
        items) left a four-argument call site guessing which argument
        failed, and 1.0 as a scalar raised ValueError instead."""
        path = _materialize(tmp_path, "pdf_two_page")
        for bad in ([1.5], 1.0, (0.5, 2)):
            with pytest.raises(TypeError, match="(?i)pages"):
                to_markdown(path, pages=bad)
        with pytest.raises(TypeError, match=r"pages= indices must be ints, not 1\.5"):
            to_markdown(path, pages=[1.5])

    def test_pages_wrong_container_types_are_type_errors(self, tmp_path: Path) -> None:
        """A str, dict, or set where the int/list/tuple selection belongs
        is a type problem: TypeError naming pages= and repr'ing the value
        (the pre-fix wheel raised ValueError here — the wrong class for a
        type problem, per the convention this class pins)."""
        path = _materialize(tmp_path, "pdf_two_page")
        for bad in ("1", {0: 1}, {0, 1}):
            with pytest.raises(TypeError, match="(?i)pages must be an int"):
                to_markdown(path, pages=bad)

    def test_pages_tuple_arity_is_a_value_error(self, tmp_path: Path) -> None:
        """A 1-tuple or 3-tuple is the right TYPE in the wrong SHAPE: the
        tuple lane is exactly the (start, stop) range, and the refusal
        points at the list spelling for an explicit page set."""
        path = _materialize(tmp_path, "pdf_two_page")
        for bad in ((1,), (0, 1, 2)):
            with pytest.raises(ValueError, match=r"\(start, stop\)"):
                to_markdown(path, pages=bad)

    def test_pages_backwards_range_selects_no_pages(self, tmp_path: Path) -> None:
        """(2, 1) and (0, 0) are valid types with a value problem: a
        half-open range that selects nothing, refused with the half-open
        convention spelled out in the message."""
        path = _materialize(tmp_path, "pdf_two_page")
        for bad in ((2, 1), (0, 0)):
            with pytest.raises(ValueError, match="selects no pages"):
                to_markdown(path, pages=bad)

    def test_pages_too_large_int_is_a_value_error_not_a_type_error(self, tmp_path: Path) -> None:
        """An int too large for the i64 the binding extracts (10**30) is
        type-correct and value-absurd: ValueError, not the TypeError a
        non-int would get — the class distinction the convention draws
        (pinned on both the scalar and the list-item paths)."""
        path = _materialize(tmp_path, "pdf_two_page")
        for bad in (10**30, [0, 10**30]):
            with pytest.raises(ValueError, match="(?i)too large"):
                to_markdown(path, pages=bad)

    def test_pages_out_of_bounds_names_the_page_count(self, tmp_path: Path) -> None:
        """Bounds are the core's to referee (only it knows the real page
        count), so the selection reaches the native pass and comes back
        as a ValueError whose message names the count — the routing
        caller's decision input, not just a bare 'bad index'."""
        path = _materialize(tmp_path, "pdf_two_page")
        with pytest.raises(ValueError, match=r"out of range for a 2-page document"):
            to_markdown(path, pages=[0, 999999])

    def test_non_string_format_and_backend_are_type_errors_naming_the_argument(
        self,
        tmp_path: Path,
    ) -> None:
        """format=123 / backend=123 on the pre-fix wheel raised pyo3's
        bare "'int' object is not an instance of 'str'" — naming neither
        the argument nor the vocabulary; the refusal must name both."""
        path = _materialize(tmp_path, "pdf_two_page")
        with pytest.raises(TypeError, match="(?i)format must be a format-name string"):
            to_markdown(path, format=123)
        with pytest.raises(TypeError, match="(?i)backend must be one of"):
            to_markdown(path, backend=123)

    def test_empty_format_and_backend_strings_are_value_errors(self, tmp_path: Path) -> None:
        """The empty string is the right type with a wrong value: an
        unknown format name (the core's vocabulary refusal, passed
        through by the binding) and a backend outside the vocabulary,
        both naming the argument."""
        path = _materialize(tmp_path, "pdf_two_page")
        with pytest.raises(ValueError, match="(?i)format"):
            to_markdown(path, format="")
        with pytest.raises(ValueError, match="(?i)backend must be one of"):
            to_markdown(path, backend="")

    def test_sniff_refuses_str_and_answers_none_for_unmarked_bytes(self) -> None:
        """sniff's input contract is bytes: a str is a TypeError (the
        pyo3 extraction's own refusal — acceptable because there is only
        one bytes-ish argument to blame), and empty or BOM-only bytes are
        not errors but None (the content names no format)."""
        with pytest.raises(TypeError, match="bytes"):
            sniff("string")
        assert sniff(b"") is None
        assert sniff(b"\xef\xbb\xbf") is None

    def test_directory_and_missing_paths_raise_oserror(self, tmp_path: Path) -> None:
        """Environment failures are OSErrors on every path-taking entry
        point (the shared parse_path spine): a directory path and a
        missing one, in the convert lane and the PDF-only lane alike. On
        Linux the matched subclass is pinned too — pyo3's io::Error
        mapping raises IsADirectoryError/FileNotFoundError where the
        pre-fix wheel raised a plain OSError with the errno buried in
        the message."""
        from tors_documents import pdf_link_uris

        for convert in (
            to_markdown,
            to_text,
            pdf_extract,
            pdf_classify,
            pdf_page_count,
            pdf_link_uris,
        ):
            with pytest.raises(OSError):
                convert(str(tmp_path))
            with pytest.raises(OSError):
                convert(str(tmp_path / "nope.pdf"))
        import sys

        with pytest.raises(OSError) as excinfo:
            to_markdown(str(tmp_path))
        if sys.platform == "linux":  # EISDIR: the matched OSError subclass (probed 2026-09)
            assert isinstance(excinfo.value, IsADirectoryError)

    def test_lone_surrogate_path_is_a_value_error_naming_path(self) -> None:
        """A path with a lone surrogate (a filename that escaped a
        POSIX-only tool) cannot become a Rust String: the pre-fix wheel
        surfaced pyo3's bare UnicodeEncodeError — 'utf-8' codec, no
        argument named, reading like an engine conversion bug. The
        refusal must be a ValueError naming path; the parse_path spine
        raises before any file is touched, so no fixture is needed."""
        with pytest.raises(ValueError, match="(?i)path must be valid unicode"):
            to_markdown("no-\udcff-such.pdf")
        with pytest.raises(ValueError, match="(?i)path must be valid unicode"):
            pdf_page_count("no-\udcff-such.pdf")

    def test_needs_ocr_error_is_a_value_error_with_the_engine_attributes(
        self, tmp_path: Path
    ) -> None:
        """NeedsOcrError's whole value is being catchable as a ValueError
        (route-to-OCR without a special except arm) while carrying the
        routing facts: .pages (0-based indices, the same convention as
        pages= and pages_needing_ocr) and .page_count, set on every
        engine-raised instance, with a message that names the pages and
        says what to do with them."""
        assert issubclass(NeedsOcrError, ValueError)
        path = _materialize(tmp_path, "pdf_mixed")
        with pytest.raises(NeedsOcrError, match="(?i)need ocr") as raised:
            to_markdown(path, backend="anydoc")
        exc = raised.value
        assert isinstance(exc, ValueError)
        assert exc.pages == [1]  # 0-based: page 0 was the text page
        assert exc.page_count == 2
        assert "0-based" in str(exc)
        assert "1" in str(exc)

    def test_typed_enums_are_their_strings_and_resolved_formats_are_members(
        self,
        tmp_path: Path,
    ) -> None:
        """The str-enum contract: every member IS its accepted string, so
        plain-string call sites and enum call sites are the same call; a
        resolved format from to_markdown is a Format member by IDENTITY
        (not just equality) while still being a str; and sniff answers
        Format members or None — never a bare string."""
        import tors_documents as payload

        assert payload.Format("docx") is payload.Format.DOCX
        assert payload.Format("docx") == "docx"
        assert payload.Backend.AUTO == "auto"
        assert isinstance(payload.Backend.AUTO, str)
        assert payload.PageKind.TEXT == "text"
        resolved, _ = to_markdown(_materialize(tmp_path, "pdf_two_page"))
        assert resolved is payload.Format.PDF
        assert isinstance(resolved, str)
        assert resolved == "pdf"
        assert payload.sniff(ENGINES_CORPUS["pdf"]) is payload.Format.PDF
        assert payload.sniff(b"plain text, no markers") is None

    def test_aio_returns_the_typed_surface_and_sniff_stays_sync_only(self, tmp_path: Path) -> None:
        """aio parity: the async spellings wrap the TYPED wrappers, so
        asyncio.run(aio.to_markdown(...)) resolves a Format member too;
        the kwargs forward through the thread hop unchanged — pages=
        selects the same page subset the sync call selects,
        byte-identical (the forwarding pin beyond the path/data/password
        trio the other aio gates cover); and sniff (a microsecond marker
        scan) has no async twin in either the payload's aio or the base
        shim's re-export, per the house rule that the thread hop would
        cost more than the call."""
        from tors_documents import Format
        from tors_documents import aio as payload_aio

        from tors.documents import aio as shim_aio

        path = _materialize(tmp_path, "pdf_two_page")
        expected_page = to_markdown(path, pages=[1])
        for aio_module in (payload_aio, shim_aio):
            resolved, markdown = asyncio.run(aio_module.to_markdown(path))
            assert resolved is Format.PDF
            assert "first page line" in markdown
            assert asyncio.run(aio_module.to_markdown(path, pages=[1])) == expected_page
            assert "sniff" not in aio_module.__all__
            assert not hasattr(aio_module, "sniff")

    def test_payload_version_matches_the_installed_wheel(self) -> None:
        """tors_documents.__version__ is baked from the crate's
        Cargo.toml at build time (release-please bumps it together with
        pyproject.toml — one release, two wheels): the pin is that the
        baked constant equals the wheel metadata's version, which is
        exactly the lockstep the release process promises."""
        from importlib.metadata import version

        import tors_documents as payload

        assert payload.__version__ == version("tors-documents")


# --- in-memory bytes input + the annotation link walk (red-team lane) ------


class TestBytesInputAndLinkWalk:
    """The ``data=`` entry and ``pdf_link_uris`` — the two surfaces the
    downstream audit demanded (the bytes-in pipeline caller holds uploads
    in memory and pays a temp-file roundtrip per document against a
    path-only API; the link walk is the raw URI surface no text rendering
    carries). ``data=`` IS the same conversion: byte-identical output, the
    same typed answers, the same error taxonomy — only the source differs.
    """

    def test_data_conversions_are_byte_identical_to_path_conversions(self, tmp_path: Path) -> None:
        """The in-memory entry is not a second implementation: for a
        marker-bearing format (docx) and a PDF alike, the ``data=`` call
        returns exactly the ``path=`` call's pair — same resolved Format
        member, byte-identical output, in both output modes."""
        from tors_documents import Format  # noqa: F401 - assertion clarity

        for kind in ("docx_rich", "pdf_two_page"):
            path = _materialize(tmp_path, kind)
            data = Path(path).read_bytes()
            for convert in (to_markdown, to_text):
                assert convert(path) == convert(data=data), f"{kind}: data= diverged"

    def test_data_has_no_name_so_resolution_rests_on_markers_and_format(
        self, tmp_path: Path
    ) -> None:
        """A ``data=`` call has no file name to consult: the content
        markers carry it alone, the markerless formats name themselves via
        ``format=``, and an undetectable byte set's error names the fix
        (``format=``) instead of a path — the bytes-in half of the
        resolution doctrine, pinned end to end."""
        from tors_documents import Format

        rows = Path(_materialize(tmp_path, "csv_rich")).read_text().encode()
        # consistent delimitation resolves from content alone
        resolved, _ = to_markdown(data=rows)
        assert resolved == Format.CSV
        # a single-column csv carries no witness: the error names format=
        with pytest.raises(ValueError, match="format="):
            to_markdown(data=b"single\ncolumn\n")
        # ...and the explicit name is the escape hatch
        assert to_markdown(data=b"single\ncolumn\n", format="csv")[0] == Format.CSV
        # an HTML fragment (no full-document marker) same doctrine
        fragment = b"<div><p>fragment body</p></div>"
        with pytest.raises(ValueError, match="format="):
            to_markdown(data=fragment)
        assert to_markdown(data=fragment, format="html")[0] == Format.HTML

    def test_the_pdf_family_takes_data_everywhere(self, tmp_path: Path) -> None:
        """All four PDF-only entry points take ``data=`` with the same
        answers as ``path=`` — the probe, the page tree, the preflight,
        and the link walk — one source-resolution spine, four functions."""
        from tors_documents import pdf_link_uris

        path = _materialize(tmp_path, "pdf_link")
        data = Path(path).read_bytes()
        assert pdf_page_count(path) == pdf_page_count(data=data)
        _pages_a, markdown_a = pdf_extract(path)
        _pages_b, markdown_b = pdf_extract(data=data)
        assert markdown_a == markdown_b
        by_path = pdf_classify(path)
        by_data = pdf_classify(data=data)
        assert by_path.page_count == by_data.page_count
        assert by_path.page_kinds == by_data.page_kinds
        assert pdf_link_uris(path) == pdf_link_uris(data=data)

    def test_exactly_one_of_path_or_data(self, tmp_path: Path) -> None:
        """Both is a ValueError, neither a TypeError (the
        missing-required-argument convention), a non-bytes ``data=`` a
        TypeError naming the argument — on the convert lane and the PDF
        lane alike, raised under the GIL before any work runs."""
        path = _materialize(tmp_path, "pdf_two_page")
        data = Path(path).read_bytes()
        for convert in (to_markdown, to_text):
            with pytest.raises(ValueError, match="either path or data"):
                convert(path, data)
            with pytest.raises(TypeError, match="missing required argument"):
                convert()
        from tors_documents import pdf_link_uris

        for pdf_call in (pdf_extract, pdf_classify, pdf_page_count, pdf_link_uris):
            with pytest.raises(ValueError, match="either path or data"):
                pdf_call(path, data)
            with pytest.raises(TypeError, match="missing required argument"):
                pdf_call()
        with pytest.raises(TypeError, match="data must be bytes"):
            to_markdown(data="not bytes")

    def test_pdf_link_uris_walk_the_annotations_per_page(self, tmp_path: Path) -> None:
        """The raw URI surface: the link-annotation fixture's page carries
        its URI in annotation order (the generator wrote exactly one), a
        document without link annotations yields empty lists per page —
        never fabricated navigation, never deduped (the verbatim walk is
        the honest output; canonicalization is the caller's)."""
        from tors_documents import pdf_link_uris

        path = _materialize(tmp_path, "pdf_link")
        uris = pdf_link_uris(path)
        assert uris == [[PDF_LINK_URL]]
        plain = _materialize(tmp_path, "pdf_two_page")
        plain_uris = pdf_link_uris(plain)
        assert plain_uris == [[], []]

    def test_sniff_then_data_is_the_bytes_pipeline_shape(self, tmp_path: Path) -> None:
        """The upload route's two-step: sniff the leading bytes (no
        parser, microseconds), then convert the full body in memory —
        no temp file anywhere in the flow, and both halves agree on what
        the bytes are."""
        from tors_documents import Format

        data = Path(_materialize(tmp_path, "pdf_link")).read_bytes()
        assert sniff(data[:1024]) == Format.PDF
        resolved, markdown = to_markdown(data=data)
        assert resolved == Format.PDF
        assert f"[{PDF_LINK_LABEL}]({PDF_LINK_URL})" in markdown

    def test_aio_carries_data_and_the_link_walk(self, tmp_path: Path) -> None:
        """The async twins pass the source pair through unchanged (a
        thread hop must not reintroduce a temp file) and carry the new
        link walk: ``data=`` in, typed answers out — both the payload's
        aio and the base shim's re-export."""
        from tors_documents import Format, pdf_link_uris
        from tors_documents import aio as payload_aio

        from tors.documents import aio as shim_aio

        data = Path(_materialize(tmp_path, "pdf_link")).read_bytes()
        expected = pdf_link_uris(data=data)
        for aio_module in (payload_aio, shim_aio):
            assert asyncio.run(aio_module.pdf_link_uris(data=data)) == expected
            resolved, _text = asyncio.run(aio_module.to_text(data=data))
            assert resolved is Format.PDF


# --- encrypted PDFs: the password door (fail-closed lane) ------------------------


def _encrypted_pdf(password: str = "torque") -> bytes:
    """A one-page RC4-128-encrypted PDF carrying exactly one line of text:
    fpdf2 writes the page, pypdf rewrites it encrypted — RC4-128 because
    it is pure-python in pypdf (AES would drag in the cryptography
    package, which the dev group does not carry). pypdf salts the
    encryption randomly per call, so no test may assert on these BYTES —
    only on behavior."""
    import io

    from fpdf import FPDF
    from pypdf import PdfWriter

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, text="Confidential torque figures", new_x="LMARGIN", new_y="NEXT")
    writer = PdfWriter()
    writer.append(io.BytesIO(bytes(pdf.output())))
    writer.encrypt(user_password=password, algorithm="RC4-128")
    locked = io.BytesIO()
    writer.write(locked)
    return locked.getvalue()


class TestEncryptedPdfs:
    """The password door, held to the fail-closed doctrine: a locked PDF
    is a typed refusal on every entry point until the right password is
    named — never empty output (the pre-fix masking: to_markdown and
    pdf_extract silently returned "" on a locked PDF while classify
    raised, so a routing caller read "no text" where the truth was "no
    access"; the door check now sits in front of every engine lane)."""

    def test_every_entry_fails_closed_without_a_password(self) -> None:
        """All six entries — both convert modes and all four PDF-only
        calls — refuse a locked PDF with pdf_oxide's ValueError naming the
        encryption, before any engine work. The defect this gate makes
        impossible to regress: the pre-fix wheel's to_markdown/pdf_extract
        returned EMPTY output on these same bytes (the empty-string
        masking — indistinguishable from a blank document); today no entry
        returns at all."""
        from tors_documents import pdf_link_uris

        locked = _encrypted_pdf()
        entries = (to_markdown, to_text, pdf_classify, pdf_extract, pdf_page_count, pdf_link_uris)
        for entry in entries:
            with pytest.raises(ValueError, match="encrypted and requires a password"):
                entry(data=locked)

    def test_a_wrong_password_is_refused_not_treated_as_none(self) -> None:
        """A wrong password must be distinguishable from no password:
        pdf_oxide's authenticate() refusal — "did not unlock" — on the
        convert lane and the probe lanes alike (a wrong password laundered
        into the generic no-access error would send the caller hunting
        the wrong knob)."""
        locked = _encrypted_pdf()
        for entry in (to_markdown, pdf_extract, pdf_classify):
            with pytest.raises(ValueError, match="did not unlock"):
                entry(data=locked, password="not-the-password")

    def test_the_right_password_converts_every_lane(self) -> None:
        """The unlock: with the right password the document is exactly the
        document — markdown carries the cell text, classify says one TEXT
        page, the count is 1, the link walk says no links, and extract's
        pages carry the text: no lane answers empty-on-success once
        unlocked."""
        from tors_documents import Format, PageKind, pdf_link_uris

        locked = _encrypted_pdf()
        resolved, markdown = to_markdown(data=locked, password="torque")
        assert resolved is Format.PDF
        assert "Confidential torque figures" in markdown
        classification = pdf_classify(data=locked, password="torque")
        assert classification.page_kinds == [PageKind.TEXT]
        assert pdf_page_count(data=locked, password="torque") == 1
        assert pdf_link_uris(data=locked, password="torque") == [[]]
        pages, extracted = pdf_extract(data=locked, password="torque")
        assert len(pages) == 1
        assert "Confidential torque figures" in pages[0]
        assert "Confidential torque figures" in extracted

    def test_password_on_a_non_pdf_format_is_refused_not_ignored(self, tmp_path: Path) -> None:
        """A password on a docx is not a no-op: the caller believes the
        document is protected, and the docx lane is not the lane that
        would know — the refusal names the parameter and the format the
        password does not apply to, in both convert modes."""
        path = _materialize(tmp_path, "docx_rich")
        for convert in (to_markdown, to_text):
            with pytest.raises(
                ValueError, match=r'password= applies to PDF documents only, not "docx"'
            ):
                convert(path, password="torque")

    def test_password_must_be_a_str_not_an_int(self) -> None:
        """password=123 is a type problem (the exception convention this
        suite pins everywhere): TypeError naming password=, raised in the
        binding before any bytes are sniffed or read — so arbitrary bytes
        carry the pin, no fixture needed."""
        with pytest.raises(TypeError, match="password must be a str"):
            to_markdown(data=b"junk bytes", password=123)
        with pytest.raises(TypeError, match="password must be a str"):
            pdf_page_count(data=b"junk bytes", password=123)

    def test_password_on_the_anydoc_pdf_lane_is_refused_not_ignored(self, tmp_path: Path) -> None:
        """anydoc's PDF reader takes no password, so a CORRECT password on
        backend="anydoc" would otherwise fail as a bare "document is
        encrypted" — indistinguishable from having passed nothing, the
        silently-ignored-argument shape. The refusal names the lane the
        password needs instead (the guard fires before any parsing, so an
        ordinary PDF carries the pin)."""
        path = _materialize(tmp_path, "pdf_two_page")
        from tors_documents import Format  # noqa: F401 - resolved-member assert

        for convert in (to_markdown, to_text):
            with pytest.raises(ValueError, match="password= requires the pdf_oxide lane"):
                convert(path, backend="anydoc", password="anything")
        # and the pdf_oxide lanes stay the password-bearing ones
        assert to_markdown(path, password="anything")[0] == Format.PDF

    def test_aio_carries_the_password_door_unchanged(self) -> None:
        """The async twins pass password= through the thread hop with the
        door intact: locked bytes still fail closed without the password,
        and unlock with the right one — in the payload's aio AND the base
        shim's re-export, the same loop pattern the data= aio gate uses."""
        from tors_documents import Format
        from tors_documents import aio as payload_aio

        from tors.documents import aio as shim_aio

        locked = _encrypted_pdf()
        for aio_module in (payload_aio, shim_aio):
            with pytest.raises(ValueError, match="encrypted and requires a password"):
                asyncio.run(aio_module.to_markdown(data=locked))
            resolved, markdown = asyncio.run(aio_module.to_markdown(data=locked, password="torque"))
            assert resolved is Format.PDF
            assert "Confidential torque figures" in markdown


# --- the anydoc input ceiling (max_bytes=) ---------------------------------------


# The csv the ceiling gates feed on: ~118 bytes of real delimiter-separated
# content, crafted inline (no fixture file) so the over/under arithmetic is
# visible in the test itself.
_CEILING_CSV = (
    b"unit,status,region,inspector\n"
    b"T-101,healthy,north,Okoye\n"
    b"T-102,needs review,south,Divsalar\n"
    b"T-103,healthy,east,Marchetti\n"
)


class TestAnydocInputCeiling:
    """The anydoc lane's memory posture as a caller-facing contract: that
    engine measures ~36x RSS amplification on delimiter formats, so the
    lane carries a DEFAULT 32 MiB input ceiling (a crate-side constant,
    pinned in the Rust tests — no 33 MB fixture is built here) and
    max_bytes= is the per-call override — a typed refusal naming both
    sizes, never a silent OOM."""

    def test_a_document_over_the_ceiling_is_refused_naming_both_sizes(self) -> None:
        """Over the ceiling is a ValueError naming the engine lane, the
        knob (max_bytes), and BOTH sizes — the document's and the
        ceiling's — at fixture scale (the ~118-byte csv against 64, raw
        byte counts: sub-0.1-MiB figures render as bytes, never the
        rounded-away "0.0 MiB vs 0.0 MiB") in both convert modes, and at
        a scale where the two figures differ (a 1.1 MiB csv against a
        1.0 MiB ceiling), so the message is proved to carry the two real
        numbers, not two constants."""
        for convert in (to_markdown, to_text):
            with pytest.raises(ValueError, match="ceiling") as raised:
                convert(data=_CEILING_CSV, max_bytes=64)
            message = str(raised.value)
            assert "max_bytes" in message
            assert "the document is" in message and "input ceiling is" in message
            assert "118 bytes" in message and "64 bytes" in message
        big = b"unit,status\n" * 100_000  # 1.2 MB: "1.1 MiB" against a "1.0 MiB" ceiling
        with pytest.raises(ValueError, match="ceiling") as raised:
            to_markdown(data=big, max_bytes=1_048_576)
        message = str(raised.value)
        assert "1.1 MiB" in message and "1.0 MiB" in message

    def test_max_bytes_overrides_the_default_ceiling_per_call(self) -> None:
        """The override semantics, pinned WITHOUT a 33 MB fixture: the
        same 1.2 MB csv sits far under the lane's DEFAULT ceiling (it
        converts, resolved csv), and max_bytes= pulls the ceiling under it
        (the same call now refuses) — a per-call knob, not a global
        reset."""
        from tors_documents import Format

        big = b"unit,status\n" * 100_000
        resolved, markdown = to_markdown(data=big)
        assert resolved is Format.CSV
        assert markdown.strip()
        with pytest.raises(ValueError, match="ceiling"):
            to_markdown(data=big, max_bytes=1_048_576)

    def test_a_document_under_the_ceiling_converts(self) -> None:
        """The ceiling refuses nothing it should not: the same csv with a
        roomy max_bytes converts — resolved csv, content intact."""
        from tors_documents import Format

        resolved, markdown = to_markdown(data=_CEILING_CSV, max_bytes=1_000_000)
        assert resolved is Format.CSV
        assert "T-101" in markdown and "needs review" in markdown

    def test_max_bytes_does_not_apply_to_the_non_anydoc_lanes(self) -> None:
        """The ceiling is the anydoc lane's memory posture, not a global
        input meter: the pdf_oxide lane is unmetered, so max_bytes=1 over
        real PDF bytes converts unmolested — the knob simply does not
        apply to that lane. A regression to GLOBAL enforcement (the PDF
        refusing under max_bytes=1) or to a per-lane refusal (a ValueError
        naming a ceiling the lane does not carry) fails this pin."""
        from tors_documents import Format

        resolved, markdown = to_markdown(data=ENGINES_CORPUS["pdf_two_page"], max_bytes=1)
        assert resolved is Format.PDF
        assert "first page line" in markdown and "second page line" in markdown

    def test_max_bytes_must_be_a_positive_int(self) -> None:
        """0 and -5 are value problems ("must be positive"); True, "64KB",
        and 1.5 are type problems naming max_bytes= — and True is checked
        FIRST, as a TYPE (a bool launders through an int extraction as 1,
        the same hazard the pages= lane's bool gate pins), so it gets the
        TypeError, never the value refusal. Both convert modes."""
        for convert in (to_markdown, to_text):
            for bad in (0, -5):
                with pytest.raises(ValueError, match="positive"):
                    convert(data=_CEILING_CSV, max_bytes=bad)
            for bad in (True, "64KB", 1.5):
                with pytest.raises(TypeError, match="max_bytes must be an int"):
                    convert(data=_CEILING_CSV, max_bytes=bad)


# --- JSON-lines is not csv (the sniffer's record-shape guard) ---------------------


class TestJsonLinesIsNotCsv:
    """The delimiter witness alone would claim JSON-lines records as csv
    (their comma counts agree line over line), so the sniffer carries a
    record-shape guard: a line opening with { or [ is a record, not a
    field row. The guard is the sniffer's honesty — sniff never names a
    format the bytes are not — and the convert lane leans on it: unnamed
    JSON-lines is a refusal naming format=, and the explicit name is the
    caller's escape hatch."""

    def test_record_lines_sniff_to_none_while_real_csv_sniffs_csv(self) -> None:
        """{ and [ records sniff to None even though the delimiter witness
        alone would claim them (the comma counts agree); the contrast row
        — real delimiter-separated bytes — still sniffs csv, so the None
        answers are the record-shape guard at work, not sniff refusing
        csv ever."""
        from tors_documents import Format

        assert sniff(b'{"a":1,"b":2}\n{"a":3,"b":4}\n') is None
        assert sniff(b"[1,2]\n[3,4]\n") is None
        assert sniff(b"unit,status\nT-101,healthy\n") is Format.CSV

    def test_unnamed_jsonl_is_refused_naming_format_and_csv_is_the_escape_hatch(self) -> None:
        """The convert lane inherits the guard: JSON-lines bytes with no
        name are a ValueError naming format= (the no-name fix — the call
        must not guess csv from the same witness the guard overruled), in
        both record shapes and both modes; with format="csv" the caller
        owns the decision and it converts — resolved csv is pinned, the
        output shape deliberately is not (the escape hatch is a
        resolution contract, not a rendering promise)."""
        from tors_documents import Format

        for records in (b'{"a":1,"b":2}\n{"a":3,"b":4}\n', b"[1,2]\n[3,4]\n"):
            for convert in (to_markdown, to_text):
                with pytest.raises(ValueError, match="format="):
                    convert(data=records)
            resolved, _markdown = to_markdown(data=records, format="csv")
            assert resolved is Format.CSV


# --- the split-wheel import laziness (the base wheel) -----------------------------


class TestImportLaziness:
    """The split-wheel design's whole point, pinned as the structural
    property: the BASE wheel (tors) carries only the lazy re-export shim,
    so importing it never loads the engines (tors.documents /
    tors_documents — the anydoc + pdf_oxide stack); they load on first
    touch of the documents attribute. Measured in a fresh interpreter —
    which modules the import touched — and deliberately NOT in megabytes
    (load-size numbers vary across platforms and flake; the module-touched
    assert is the design invariant)."""

    def test_importing_the_base_wheel_never_touches_the_engines(self) -> None:
        """A fresh interpreter (this venv's own python) imports tors and
        must not carry tors.documents or tors_documents in sys.modules. If
        this regresses — an eager import creeps into the base package —
        every base-wheel consumer pays the engines' load cost on import,
        and the failure reports the subprocess's own stderr (the
        assertion that fired)."""
        import subprocess
        import sys

        code = (
            "import sys; import tors; "
            "assert 'tors.documents' not in sys.modules; "
            "assert 'tors_documents' not in sys.modules; "
            "print('lazy')"
        )
        try:
            done = subprocess.run(
                [sys.executable, "-c", code], check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as exc:
            pytest.fail(f"import tors loaded the engines: {exc.stderr.strip()}")
        assert done.returncode == 0
        assert "lazy" in done.stdout

    def test_the_documents_attribute_is_the_lazy_door_to_the_payload(self) -> None:
        """The ergonomic half of the split-wheel design (what the README
        examples' ``tors.documents`` spelling rides on): the base package
        carries a PEP 562 ``__getattr__``, so ``tors.documents`` on an
        imported-but-untouched base package imports the shim — and through
        it the payload — on FIRST touch, and the returned object re-exports
        the payload's surface by identity
        (``tors.documents.to_markdown is tors_documents.to_markdown``).
        The laziness gate still holds BEFORE the access (both engine
        modules untouched by the plain import — the same fresh-interpreter
        measurement as the gate above), and any other missing name keeps
        the standard module AttributeError."""
        import subprocess
        import sys

        code = (
            "import sys; import tors; "
            "assert 'tors.documents' not in sys.modules; "
            "assert 'tors_documents' not in sys.modules; "
            "documents = tors.documents; "
            "assert 'tors.documents' in sys.modules; "
            "assert 'tors_documents' in sys.modules; "
            "import tors_documents; "
            "assert documents.to_markdown is tors_documents.to_markdown; "
            "print('lazy-then-loaded')"
        )
        try:
            done = subprocess.run(
                [sys.executable, "-c", code], check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as exc:
            pytest.fail(f"tors.documents lazy access failed: {exc.stderr.strip()}")
        assert done.returncode == 0
        assert "lazy-then-loaded" in done.stdout

        import tors

        with pytest.raises(AttributeError):
            tors.definitely_not_an_attribute  # noqa: B018 -- the access IS the point

    def test_the_sync_shim_re_exports_the_whole_payload_surface(self, tmp_path: Path) -> None:
        """The base wheel's tors.documents shim IS the payload's public
        surface, never a subset or a re-wrap: ``__all__`` equal name for
        name (``__version__`` included), every name the very SAME object
        tors_documents exports (an identity re-export — a divergence would
        mean a second implementation hiding behind the door), and one full
        conversion through the shim path end to end — the spelling the
        README examples use, exercised the way callers spell it."""
        import tors_documents as payload

        import tors.documents as shim

        assert shim.__all__ == payload.__all__
        for name in shim.__all__:
            assert getattr(shim, name) is getattr(payload, name), name
        resolved, markdown = shim.to_markdown(_materialize(tmp_path, "pdf_two_page"))
        assert resolved is payload.Format.PDF
        assert "first page line" in markdown and "second page line" in markdown
