"""The cross-format validation suite: every documents engine, held to one
contract over generated documents of every working format.

Three generation lanes, one set of gates (tests/documents_gates.py):

- fixed fixtures (tests/documents.py's ``ENGINES_CORPUS``, committed under
  tests/engines_corpus/ and byte-pinned): one structurally rich document
  per format family, plus PDF structural variants (link annotation,
  two-column, oversize heading, blank page, two pages). These gate the
  measured structural facts (style-based docx headings, link rendering,
  GFM table delimiter rows, column separation, NeedsOcr routing) the
  properties the engine choice was made on.
- fuzz (tests/docgen.py, seeded): randomized real-writer documents (fpdf2,
  python-docx, openpyxl, python-pptx) with varying structure, columns,
  links, and formatting, each carrying its ground truth. These gate
  alignment: every emitted content unit must survive conversion (markdown
  and plain-text modes), nothing fabricated, deterministic across calls.
- mutation: corrupted bytes (truncation, flips, splices) must end in a
  typed error or a usable result: never a crash, never a hang.

The API contract is gated where it exists today and skipped with A loud
reason where a surface has not landed yet: the skips disappear the moment the
implementation arrives, and the tests then hold it. The suite imports the payload surface
from ``tors_documents`` (the tors-documents wheel) with a fallback to
in-base wrappers: whichever wiring ships, the gates are the same.
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
    generate_docx_margins_only,
    generate_docx_pgsz,
    generate_pdf_big,
    generate_pdf_crop,
    generate_pdf_crop_partial,
    generate_pdf_layout,
    generate_pdf_layout_artifact,
    generate_pdf_layout_artifact_bmc,
    generate_pdf_layout_baseline,
    generate_pdf_layout_bold_mid_line,
    generate_pdf_layout_crop,
    generate_pdf_layout_crop_partial_span,
    generate_pdf_layout_image,
    generate_pdf_layout_pages,
    generate_pdf_layout_rotated,
    generate_pdf_layout_rotated_run,
    generate_pdf_layout_two_images,
    generate_pdf_media_origin,
    generate_pdf_pages,
    generate_pdf_rotate,
    generate_pdf_zero_pages,
    generate_pptx_empty,
    generate_pptx_sldsz,
    generate_xlsx_chart,
)

try:  # the payload wheel (tors[documents])
    from tors_documents import (
        ImageBox,
        NeedsOcrError,
        PageGeometry,
        PageLayout,
        TextLine,
        TextSpan,
        Word,
        page_geometry,
        page_layout,
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
            ImageBox,
            NeedsOcrError,
            PageGeometry,
            PageLayout,
            TextLine,
            TextSpan,
            Word,
            page_geometry,
            page_layout,
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
    extension (every kind's extension is a name in the format vocabulary)."""
    return Path(ENGINES_FILENAMES[kind]).suffix.lstrip(".")


def _resolved_name(kind: str) -> str:
    """The format name a conversion of this corpus kind resolves to. The
    OOXML alias containers (docm_rich/xlsm/ppsx_rich) resolve onto their
    base kinds (one OOXML package each, the [Content_Types].xml override
    names the macro/show variant and the engines read the same parts the
    base carries) so the resolved (and sniffed) name is the container's
    own (docx/xlsx/pptx), never the alias; every other kind's is its
    extension."""
    return {"docm_rich": "docx", "xlsm": "xlsx", "ppsx_rich": "pptx"}.get(
        kind, _format_name(kind)
    )


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
    the shared-vocabulary fixtures (docx_rich/odt_rich/html_rich: built
    from the RICH_* constants) carry that vocabulary plus their per-fixture
    extras; the OOXML aliases (docm_rich/xlsm/ppsx_rich) carry their base
    fixtures' lines (the container is the base's, only the content-type
    override differs); the remaining variants carry only their own lines
    (csv is tabular data, the pptx/pdf variants have no shared vocabulary)."""
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
        "docm_rich": ["Closing note."],  # generate_docx_rich's container as a docm
        "pptx_rich": [SLIDE1_TITLE, *SLIDE1_BODY, SLIDE2_TITLE, RICH_NOTES],
        "ppsx_rich": [SLIDE1_TITLE, *SLIDE1_BODY, SLIDE2_TITLE, RICH_NOTES],
        "odt_rich": ["Draw oil samples quarterly."],
        "csv_rich": ["T-101", "healthy", "T-102", "needs review"],
        "html_rich": [RICH_NOTES],
        "xlsm": EXPECTED_TEXT["xlsx"].splitlines(),  # generate_xlsx's container as an xlsm
    }
    lines = per_kind[kind]
    if kind in {"docx_rich", "docm_rich", "odt_rich", "html_rich"}:
        lines = shared + lines
    return lines


@pytest.mark.parametrize(
    "kind",
    ["docx_rich", "docm_rich", "xlsx", "xlsm", "pptx_rich"],
    ids=["docx-rich", "docm-rich", "xlsx", "xlsm", "pptx-rich"],
)
def test_the_oxide_forced_backend_converts_the_ooxml_family_and_aligns(
    tmp_path: Path, kind: str
) -> None:
    """The oxide family's working overlap, content-gated: backend="oxide"
    forces office_oxide over the OOXML containers, and the auto lane's
    choice for the office family is anydoc: so without this pin the
    oxide lane's only pytest coverage is its refusals. The same alignment
    gate as the auto lane's matrix (every oracle line survives) plus the
    resolved name; the structural gates stay on the auto lane (lean by
    design: the overlap is content parity, not a second structure
    contract). The macro-enabled aliases (docm_rich/xlsm) are part of the
    working overlap (office_oxide reads the macroEnabled main+xml content
    type); ppsx_rich is deliberately absent: office_oxide refuses the
    slideshow content type, a divergence the alias class below pins."""
    path = _materialize(tmp_path, kind)
    resolved, markdown = to_markdown(path, backend="oxide")
    assert resolved == _resolved_name(kind)
    for line in _oracle_lines(kind):
        assert align(line) in align(markdown), f"{kind}: oracle line missing: {line!r}"


class TestStructureGates:
    """The measured structural facts per format family: the properties the
    engine stack was chosen on (see tors-core's documents_impl docs). The
    asserts are semantic (a heading line containing the text, the URL
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
        (neither extract nor OCR: not in the OCR list); an image-only page
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
        """Cross-engine routing alignment: both engines name the same pages
        as needing OCR, in the same 0-based convention (anydoc's 1-based
        numbers are re-based at the core's seam: one convention across the
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
        empty output (the caller routes: classify says ``empty``, neither
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
        between here and support. This gate skips with this reason until
        the name resolves, then holds it to the same table contract. The
        resolved name is pinned exactly as csv, the docs/documents.md "always csv,
        never tsv" contract (probed before pinning: the mapping
        lands, the resolved answer is Format.CSV)."""
        path = tmp_path / "engines_units.tsv"
        path.write_bytes(b"unit\tstatus\nT-101\thealthy\nT-102\tneeds review\n")
        try:
            resolved, markdown = to_markdown(str(path), format="tsv")
        except ValueError as exc:
            pytest.skip(
                f"format name 'tsv' not yet in the vocabulary ({exc}); "
                "the engine lane's mapping lands it"
            )
        assert resolved == "csv"
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
    structure = check_markdown_structure(
        truth, markdown, gate_links=kind in {"docx", "docm", "html"}
    )
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
    error (ValueError/OSError/NeedsOcrError) or a usable result: never an
    untyped crash. Content is not gated on corrupted input (partial output
    from a damaged file is legitimate); only the failure type is."""
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
        engine-lane handoff delivered): the errno-matched OSError subclass
        (IsADirectoryError, FileNotFoundError) so a caller's ``except
        FileNotFoundError`` catches, with the errno text carried in the
        message; ``.errno`` itself is None on the mapped subclass (the
        mapping constructs the subclass without the errno attribute: the
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
    unrecognized. Skipped with this reason until the surface lands."""
    import tors_documents as payload

    sniff = getattr(payload, "sniff", None) or getattr(payload, "sniff_format", None)
    if sniff is None:
        pytest.skip("sniff not yet exposed by the payload (spec'd API; lands with the wiring)")
    expectations = {
        "pdf": "pdf",
        "docx_rich": "docx",
        "docm_rich": "docx",  # the alias sniffs as its container (docx), never its extension
        "xlsx": "xlsx",
        "xlsm": "xlsx",  # same container honesty: the macro-enabled workbook sniffs xlsx
        "pptx_rich": "pptx",
        "ppsx_rich": "pptx",  # the slide-show alias sniffs as its container (pptx)
        "odt_rich": "odt",
        "rtf": "rtf",
        "html_rich": "html",
    }
    for kind, expected in expectations.items():
        assert sniff(ENGINES_CORPUS[kind]) == expected, kind
    assert sniff(b"plain text, no markers") is None
    assert sniff(b"plain prose, no markers and no delimiter structure") is None
    # CSV is signature-less by anydoc's own docs, but the engine's sniffer
    # may heuristically name delimiter-separated bytes as csv: either
    # answer is its documented behavior; the contract is that sniff never
    # names a format the bytes are not (pinned by every other row above).


def test_truncated_magic_bytes_sniff_to_none() -> None:
    """A truncated prefix of a magic is not the format: the empty bytes,
    a two-byte ZIP head, a three-byte PDF head, and a JSON fragment all
    sniff to None: never a guess from a partial signature. The whole
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
    [Content_Types].xml) must not be claimed as docx/xlsx/pptx (sniff
    says None) and converting it is a typed ValueError under its own
    extension and under a lying .docx extension (the extension fallback
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
# text formats with no required structure (measured; every
# other kind refuses: see the empty-file gate below).
_EMPTY_READS_AS_A_DOCUMENT = {"csv", "html"}


@pytest.mark.parametrize("kind", sorted(ENGINES_CORPUS), ids=lambda kind: kind)
def test_an_empty_file_is_a_typed_refusal_or_an_empty_document(tmp_path: Path, kind: str) -> None:
    """A 0-byte file never crashes and never raises OSError: the
    structured formats (pdf/docx/xlsx/pptx/odt/rtf) refuse it with a
    typed ValueError naming the malformed container; csv and html (whose
    empty bytes are a valid empty document) convert to exactly empty
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
    incompatibility: oxide on the anydoc-lane formats (rtf/odt/csv) and
    on html (the html engine's own lane), anydoc on html. (The odt pair
    is also pinned by test_pdf_backend_choice_is_real; the working
    overlaps (anydoc on pdf, oxide on the OOXML family) by the same
    test, the structure gates, and the forced-oxide alignment matrix.)"""
    path = _materialize(tmp_path, kind)
    with pytest.raises(ValueError, match="cannot read"):
        to_markdown(path, backend=backend)


# --- the OOXML alias containers (docm / xlsm / ppsx) ------------------------------
#
# The macro-enabled (docm/xlsm) and slide-show (ppsx) variants are genuine
# aliases: the same OOXML packages as docx/xlsx/pptx with
# [Content_Types].xml's main-document override naming the variant (the
# only difference between the containers: tests/documents.py's
# `_ooxml_alias_container` builds the committed fixtures by exactly that
# rewrite, and docgen.py's `_ooxml_alias` does the same over the writer
# libraries' own output for the fuzz lane). xlsb is not an alias and is
# pinned as refused below; the legacy OLE trio (doc/xls/ppt) stays
# uncovered (no fixture can hand-generate OLE compound files;
# docs/documents.md discloses them).


class TestOoxmlAliasContainers:
    """The alias contract: the conversion is the base format's:
    byte-identical output, the base kind's resolved name, reachable by the
    alias extension (the routing caller's file naming) and the alias name
    (the format= vocabulary): plus the honest lane divergence (oxide
    refuses ppsx's slideshow content type) and the xlsb disclosure (the
    name routes the Excel kind; genuine BIFF12 content is refused)."""

    @pytest.mark.parametrize(
        ("kind", "base"),
        [("docm_rich", "docx_rich"), ("xlsm", "xlsx"), ("ppsx_rich", "pptx_rich")],
        ids=["docm", "xlsm", "ppsx"],
    )
    def test_the_alias_container_converts_as_its_base_byte_identically(
        self, tmp_path: Path, kind: str, base: str
    ) -> None:
        """The alias's whole contract: same parts, same conversion: both
        modes byte-identical to the base fixture's, and the resolved name
        the container's own (docx/xlsx/pptx), never the alias."""
        alias_path = _materialize(tmp_path, kind)
        base_path = _materialize(tmp_path, base)
        assert to_markdown(alias_path) == to_markdown(base_path), f"{kind}: output diverged"
        assert to_text(alias_path) == to_text(base_path), f"{kind}: text diverged"
        assert to_markdown(alias_path)[0] == _resolved_name(kind)

    def test_the_alias_names_route_their_base_kinds(self, tmp_path: Path) -> None:
        """format="docm"/"xlsm"/"ppsx" are vocabulary spellings of the base
        kinds: the explicit name resolves the base kind and converts
        byte-identically to the sniffed call: a routing instruction, never
        a different conversion (the doctrine the explicit-name matrix holds
        for every kind, held here for the alias spellings specifically)."""
        for kind in ("docm_rich", "xlsm", "ppsx_rich"):
            path = _materialize(tmp_path, kind)
            sniffed = to_markdown(path)
            resolved, markdown = to_markdown(path, format=_format_name(kind))
            assert resolved == _resolved_name(kind)
            assert markdown == sniffed[1], f"{kind}: the alias NAME changed the conversion"

    def test_the_oxide_lane_refuses_ppsx_cleanly(self, tmp_path: Path) -> None:
        """The honest lane divergence: office_oxide checks the presentation
        content type and refuses the slide-show variant: a typed
        ValueError naming what the package holds and what was expected,
        never a silent fallback and never a crash. (docm/xlsm are in the
        oxide lane's working overlap: the forced-oxide matrix above.)"""
        path = _materialize(tmp_path, "ppsx_rich")
        with pytest.raises(ValueError, match="format mismatch") as raised:
            to_markdown(path, backend="oxide")
        message = str(raised.value)
        assert "slideshow.macroEnabled.main+xml" in message, message
        assert "expected a PresentationML presentation" in message, message

    def test_genuine_xlsb_content_is_refused_not_mangled(self, tmp_path: Path) -> None:
        """The xlsb disclosure, pinned: "xlsb" is a name in the format
        vocabulary (it routes the Excel kind: OOXML Excel bytes under
        format="xlsb" convert, vocabulary sugar), but the format itself is
        a different binary container (BIFF12 .bin sheet streams, not
        worksheet XML): a zip carrying the genuine shape is refused with a
        typed ValueError, never silently empty output pretending to be a
        read. The engines do not read xlsb, and this pin holds them to
        saying so."""
        import io

        xlsx = ENGINES_CORPUS["xlsx"]
        source = zipfile.ZipFile(io.BytesIO(xlsx))
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == "[Content_Types].xml":
                    data = data.replace(
                        b"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        b"application/vnd.ms-excel.sheet.binary.macroEnabled.12",
                    )
                if item.filename == "xl/worksheets/sheet1.xml":
                    # the real xlsb carries BIFF12 binary streams here, not XML
                    item.filename = "xl/worksheets/sheet1.bin"
                    data = b"\x86\x00\x00\x01\x00\x00\x00\x01" + b"\x00" * 40
                archive.writestr(item, data)
        path = tmp_path / "engines_binary.xlsb"
        path.write_bytes(out.getvalue())
        with pytest.raises(ValueError, match="(?i)sheet"):
            to_markdown(str(path), format="xlsb")
        # the name itself still routes OOXML Excel bytes (the vocabulary sugar)
        resolved, markdown = to_markdown(data=xlsx, format="xlsb")
        assert resolved == "xlsx"
        assert markdown == to_markdown(data=xlsx)[1]


@pytest.mark.parametrize("kind", sorted(ENGINES_CORPUS), ids=lambda kind: kind)
def test_an_explicitly_named_format_matches_the_sniffed_conversion(
    tmp_path: Path, kind: str
) -> None:
    """format= names the format instead of sniffing it: a routing
    instruction, never a different conversion: for every corpus kind the
    output is byte-identical to the sniffed default's. The resolved name
    is the container's own (_resolved_name): the alias kinds' extensions
    (docm/xlsm/ppsx) spell names that route their base kinds, and the
    resolved answer names the container (docx/xlsx/pptx), never the alias:
    the same container-honesty the Excel family's xls/xlsx split carries."""
    path = _materialize(tmp_path, kind)
    name = _format_name(kind)
    sniffed = to_markdown(path)
    assert sniffed[0] == _resolved_name(kind)
    resolved, markdown = to_markdown(path, format=name)
    assert resolved == _resolved_name(kind)
    assert markdown == sniffed[1]


def test_pdf_blank_page_routing_aligns_on_the_oxide_probe(tmp_path: Path) -> None:
    """Routing alignment for generated PDFs (fpdf2's blank pages carry an
    empty content stream, unlike the fixed fixtures' no-stream pages: a
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
    is every page; non-PDF formats and the anydoc backend refuse it. Skipped
    with this reason until the parameter lands."""
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
    """pages= is a set of page positions, not a sequence of instructions:
    duplicates collapse ([0, 0] is page 0 once) and the caller's ordering
    is ignored: [1, 0] and [0, 1] both come back as the document's own
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
    0-based contract has no Python-style negative indexing): both are
    typed ValueError refusals. (Out-of-range indices are pinned by
    test_page_range_extraction's [17].)"""
    path = _materialize(tmp_path, "pdf_two_page")
    with pytest.raises(ValueError):
        to_markdown(path, pages=[])
    with pytest.raises(ValueError):
        to_markdown(path, pages=[-1])


def test_the_int_and_range_page_spellings_select_the_same_pages(tmp_path: Path) -> None:
    """The three accepted spellings (one int, the explicit list, the
    half-open (start, stop) tuple) select the same pages: 1 and (1, 2)
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
    promise): with format=None the content markers decide: every corpus
    kind's bytes under a deliberately wrong extension resolve to their
    true format (the extension never wins) and carry the same oracle
    content as the correctly-named file. The extension is only the
    fallback for content no marker names: that path is pinned by the
    generic-zip gate, not by any corpus kind (every kind's content is
    marker-named, csv included: the delimiter heuristic names it). The
    alias kinds resolve to their base names under the wrong extension
    too: the markers name the container (docx/xlsx/pptx), which is the
    true format of the bytes."""
    name = _format_name(kind)
    wrong_extension = ".docx" if name == "pdf" else ".pdf"
    path = tmp_path / f"mislabeled_{kind}{wrong_extension}"
    path.write_bytes(ENGINES_CORPUS[kind])
    resolved, markdown = to_markdown(str(path))
    assert resolved == _resolved_name(kind), "the extension must not win over the content markers"
    for line in _oracle_lines(kind):
        assert align(line) in align(markdown), f"{kind}: oracle line missing: {line!r}"


# --- GIL band and concurrency (the payload-level claims) -------------------------
#
# The heartbeat cells in this section carry ``@pytest.mark.timing`` (the
# measurement lane): they are load-sensitive band measurements over
# generated multi-hundred-KB documents, the same class of cell as
# tests/test_gil_release.py's, and CI's matrix legs deselect the lane
# (`-m "not timing and not sweep"`) with one 3.12 leg running it: the
# marker-split contract ci.yml's comments claim (every lane cell carries
# its marker). The concurrency cells stay in the fast lane: they assert
# byte-identical answers under threads, not gap bands, and are
# load-insensitive.


@pytest.mark.timing
def test_to_markdown_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    tmp_path: Path,
) -> None:
    """The GIL-release claim as a test, the suite's shared methodology: the
    whole read+sniff+convert pass runs under py.detach, so a to_thread wrap
    keeps a 10ms heartbeat alive through a ~470KB document conversion (the
    official pdf_oxide wheel measures GIL-held per call instead: the
    hazard this payload exists to remove)."""
    from test_gil_release import _assert_loop_stays_responsive

    path = tmp_path / "big.pdf"
    path.write_bytes(generate_pdf_big(6000))

    def op() -> None:
        to_markdown(str(path))

    asyncio.run(_assert_loop_stays_responsive(lambda: asyncio.to_thread(op)))


def test_concurrent_conversions_are_identical(tmp_path: Path) -> None:
    """8 threads, one conversion each over three formats: with the GIL
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


@pytest.mark.timing
def test_to_text_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    tmp_path: Path,
) -> None:
    """The GIL-release claim holds in the TEXT lane too, the same shared
    methodology: the convert+strip pass runs under the same py.detach band
    as to_markdown's (measured: ~77ms wall over the ~470KB
    document, against to_markdown's ~140ms: the strip rides inside the
    released band), so a to_thread wrap keeps the 10ms heartbeat alive in
    both modes."""
    from test_gil_release import _assert_loop_stays_responsive

    path = tmp_path / "big.pdf"
    path.write_bytes(generate_pdf_big(6000))

    def op() -> None:
        to_text(str(path))

    asyncio.run(_assert_loop_stays_responsive(lambda: asyncio.to_thread(op)))


@pytest.mark.timing
def test_page_geometry_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
    tmp_path: Path,
) -> None:
    """The GIL-release claim on the geometry probe (§9 item 12's cell):
    the whole read+resolve+geometry pass runs under py.detach and the
    O(pages) pyclass marshalling lands after the GIL is reacquired, so a
    to_thread wrap keeps the 10ms heartbeat alive through a 500-page
    document — the multi-hundred-page fixture the item asks for, since
    per-entry marshalling is the residue class this surface adds (one
    native pyobject per Some entry)."""
    from test_gil_release import _assert_loop_stays_responsive

    path = tmp_path / "geometry_500.pdf"
    path.write_bytes(generate_pdf_pages(500))
    entries = page_geometry(str(path))
    assert len(entries) == 500
    assert all(entry is not None for entry in entries)

    def op() -> None:
        page_geometry(str(path))

    asyncio.run(_assert_loop_stays_responsive(lambda: asyncio.to_thread(op)))


@pytest.mark.timing
def test_to_text_on_a_big_csv_keeps_the_event_loop_at_heartbeat_granularity(
    tmp_path: Path,
) -> None:
    """The anydoc lane's heartbeat cell: the amplification lane, the one
    the 32 MiB default input ceiling exists for (a many-short-cells csv
    holds a measured ~146x its input in resident memory at the worst case),
    and so the heaviest native pass per input byte. The same shared
    methodology over a generated 2 MiB csv (measured on this
    box: ~0.3s wall, ~190 MiB peak: an order of magnitude over the 10ms
    ping floor so the ratio resolves, while staying memory-sane for a
    measurement cell). The pdf-lane cells above hold the read+convert
    shape; this one holds the lane whose engine materializes row
    structures per cell."""
    from test_gil_release import _assert_loop_stays_responsive

    header = ",".join(f"h{column}" for column in range(16)) + "\n"
    row = ",".join(f"c{column}" for column in range(16)) + "\n"
    unit = (header + row).encode()
    path = tmp_path / "engines_units_big.csv"
    path.write_bytes(unit * (2 * 1024 * 1024 // len(unit)))

    def op() -> None:
        to_text(str(path))

    asyncio.run(_assert_loop_stays_responsive(lambda: asyncio.to_thread(op)))


@pytest.mark.parametrize("kind", ("pdf_two_page", "docx_rich", "html_rich"), ids=lambda kind: kind)
def test_concurrent_text_conversions_are_identical(tmp_path: Path, kind: str) -> None:
    """8 threads, one to_text each over the same file: with the GIL
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
# every adversarial input probed against the installed wheel,
# pinned as a gate: the binding layer's refusal surface (wrong types ->
# TypeError, wrong values/shapes -> ValueError, environment failures ->
# OSError, each naming the argument it refused), the typed-enum identity
# contract, and the aio parity. Findings that measured as bugs on the
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
    failures are OSErrors; and every message names the argument."""

    def test_pages_booleans_are_rejected_not_laundered_to_indices(self, tmp_path: Path) -> None:
        """pages=True on the pre-fix wheel converted page 1 silently
        (pyo3's i64 extraction accepts bool because isinstance(True, int)
        holds): booleans must be a TypeError naming them, in every
        position they can appear (the scalar, a list item, a tuple item)
        and in both conversion modes (one shared parse_pages spine)."""
        path = _materialize(tmp_path, "pdf_two_page")
        for convert in (to_markdown, to_text):
            for bad in (True, False, [True, 0], [1, False], (True, 2), (0, False)):
                with pytest.raises(TypeError, match="(?i)booleans are not page indices"):
                    convert(path, pages=bad)

    def test_pages_floats_are_type_errors_naming_pages(self, tmp_path: Path) -> None:
        """A float index is a type problem (Python's own convention:
        range(1.0) and seq[1.0] raise TypeError), and the refusal must
        name pages= and repr the value: the pre-fix wheel's bare
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
        (the pre-fix wheel raised ValueError here: the wrong class for a
        type problem, per the convention this class pins)."""
        path = _materialize(tmp_path, "pdf_two_page")
        for bad in ("1", {0: 1}, {0, 1}):
            with pytest.raises(TypeError, match="(?i)pages must be an int"):
                to_markdown(path, pages=bad)

    def test_pages_tuple_arity_is_a_value_error(self, tmp_path: Path) -> None:
        """A 1-tuple or 3-tuple is the right type in the wrong shape: the
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
        non-int would get: the class distinction the convention draws
        (pinned on both the scalar and the list-item paths)."""
        path = _materialize(tmp_path, "pdf_two_page")
        for bad in (10**30, [0, 10**30]):
            with pytest.raises(ValueError, match="(?i)too large"):
                to_markdown(path, pages=bad)

    def test_pages_out_of_bounds_names_the_page_count(self, tmp_path: Path) -> None:
        """Bounds are the core's to referee (only it knows the real page
        count), so the selection reaches the native pass and comes back
        as a ValueError whose message names the count: the routing
        caller's decision input, not just a bare 'bad index'."""
        path = _materialize(tmp_path, "pdf_two_page")
        with pytest.raises(ValueError, match=r"out of range for a 2-page document"):
            to_markdown(path, pages=[0, 999999])

    def test_non_string_format_and_backend_are_type_errors_naming_the_argument(
        self,
        tmp_path: Path,
    ) -> None:
        """format=123 / backend=123 on the pre-fix wheel raised pyo3's
        bare "'int' object is not an instance of 'str'": naming neither
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
        pyo3 extraction's own refusal: acceptable because there is only
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
        Linux the matched subclass is pinned too: pyo3's io::Error
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
        if sys.platform == "linux":  # EISDIR: the matched OSError subclass (probed)
            assert isinstance(excinfo.value, IsADirectoryError)

    def test_lone_surrogate_path_is_a_value_error_naming_path(self) -> None:
        """A path with a lone surrogate (a filename that escaped a
        POSIX-only tool) cannot become a Rust String: the pre-fix wheel
        surfaced pyo3's bare UnicodeEncodeError: 'utf-8' codec, no
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
        """The str-enum contract: every member is its accepted string, so
        plain-string call sites and enum call sites are the same call; a
        resolved format from to_markdown is a Format member by identity
        (not just equality) while still being a str; and sniff answers
        Format members or None: never a bare string."""
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
        """aio parity: the async spellings wrap the typed wrappers, so
        asyncio.run(aio.to_markdown(...)) resolves a Format member too;
        the kwargs forward through the thread hop unchanged: pages=
        selects the same page subset the sync call selects,
        byte-identical (the forwarding pin beyond the path/data/password
        trio the other aio gates cover); and sniff has no async twin in
        either the payload's aio or the base shim's re-export, per the
        house rule that the thread hop must pay for itself (sniff's cost
        is a container parse, not a marker scan: measured 267 MiB peak
        RSS on a 120 KiB zip; but it stays a sync caller's call: no
        awaitable spelling ships)."""
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
        pyproject.toml: one release, two wheels): the pin is that the
        baked constant equals the wheel metadata's version, which is
        exactly the lockstep the release process promises."""
        from importlib.metadata import version

        import tors_documents as payload

        assert payload.__version__ == version("tors-documents")


# --- the PDF family's backend lanes and input budget ------------------------------


class TestPdfFamilyBackendAndBudget:
    """The four PDF-only entry points join the conversion pair's parameter
    vocabulary (``backend=`` and ``max_bytes=``) with honest engine
    routing: ``"auto"``/``"oxide"`` are the pdf_oxide lane (the mapping
    ``engine_for`` makes for PDF, byte-identical either way), and
    ``"anydoc"`` is a capability refusal raised under the GIL before any
    work runs. Red, captured verbatim on this tree before the params
    existed: ``pdf_extract(path, backend="anydoc")`` died as
    ``TypeError: pdf_extract() got an unexpected keyword argument
    'backend'`` (all four functions, ``max_bytes=`` the same shape) the
    vocabulary was simply absent. The refusal is deliberately not the
    format-level UnsupportedBackend (PDF+anydoc is a usable conversion
    pair, pinned elsewhere in this suite): anydoc's entire PDF surface
    is whole-document markdown (~anydoc-0.2.4/src/formats/pdf.rs: no
    per-page text, no page-count success return, no per-page
    classification, no annotation walk), and each of these calls needs
    exactly one of those surfaces."""

    def test_anydoc_is_a_capability_refusal_naming_the_lane_and_the_alternative(
        self, tmp_path: Path
    ) -> None:
        """The named ValueError, message pinned: it names ``backend`` and
        the value, names the capability the call needs, points at the
        conversion pair as the anydoc PDF surface (with NeedsOcrError as
        that lane's scanned-page signal), and names the lanes that serve
        this call. On both source spellings (``path=`` and ``data=``) and
        through the enum member: the doctrine text is one helper in the
        core (next to the routing table), so one full pin plus the
        capability names for the other three calls."""
        from tors_documents import Backend, pdf_link_uris

        path = _materialize(tmp_path, "pdf_two_page")
        data = ENGINES_CORPUS["pdf_two_page"]
        capabilities = {
            pdf_extract: "the per-page text probe",
            pdf_page_count: "the page tree",
            pdf_classify: "per-page classification",
            pdf_link_uris: "the /Annots link walk",
        }
        for call, capability in capabilities.items():
            for source in ({"path": path}, {"data": data}):
                with pytest.raises(ValueError, match="backend") as raised:
                    call(**source, backend="anydoc")
                message = str(raised.value)
                assert 'backend "anydoc" cannot serve' in message, message
                assert capability in message, message
                assert "to_markdown/to_text" in message, message
                assert "NeedsOcrError" in message, message
                assert "backend='auto' or 'oxide'" in message, message
        # the enum member is its string: the same refusal
        with pytest.raises(ValueError, match="cannot serve the per-page text probe"):
            pdf_extract(path=path, backend=Backend.ANYDOC)

    def test_the_refusal_precedes_the_encrypted_door_check(self) -> None:
        """A contract failure precedes work: on a locked PDF the
        ``backend="anydoc"`` call surfaces the capability refusal, never
        the door-check error: the refusal is raised under the GIL, ahead
        of the detach where the door check lives, so the caller learns
        the argument is wrong, not that a document they never asked to
        convert is encrypted."""
        from tors_documents import page_layout, pdf_link_uris

        locked = _encrypted_pdf()
        for call in (pdf_extract, pdf_page_count, pdf_classify, pdf_link_uris, page_layout):
            with pytest.raises(ValueError, match="cannot serve") as raised:
                call(data=locked, backend="anydoc")
            assert "encrypted" not in str(raised.value)

    def test_the_refusal_precedes_the_missing_path_check(self, tmp_path: Path) -> None:
        """Refusal-before-IO, the ``path=`` spelling (the ``data=`` rows
        above): on a path that does not exist the ``backend="anydoc"``
        call surfaces the capability ValueError, never FileNotFoundError:
        the argument contract is checked under the GIL, ahead of the read
        the missing path would fail, so the caller learns the argument is
        wrong, not that a file they cannot use is also absent."""
        missing = str(tmp_path / "not-there.pdf")
        with pytest.raises(ValueError, match="cannot serve the per-page text probe") as raised:
            pdf_extract(path=missing, backend="anydoc")
        assert not isinstance(raised.value, OSError)

    def test_the_same_bytes_convert_on_the_pair_and_refuse_on_the_probe(
        self, tmp_path: Path
    ) -> None:
        """The surface's shape in one test, on one byte set (the corpus's
        engines_report.pdf): the conversion pair's anydoc lane converts it
        (``Format.PDF``, real markdown) while the probe's anydoc call
        refuses with the capability message pointing at that pair. The two
        halves are pinned separately elsewhere (the backend matrix's
        overlap cell, this class's refusal pin); held together they are
        the doctrine itself: the refusal is about the call's surface, not
        the format's: PDF+anydoc is a usable pair, and the probe is not
        the pair."""
        from tors_documents import Format

        data = ENGINES_CORPUS["pdf"]
        resolved, markdown = to_markdown(data=data, backend="anydoc")
        assert resolved is Format.PDF
        assert "Quarterly Review Q3 2026" in markdown
        with pytest.raises(ValueError, match="cannot serve the per-page text probe") as raised:
            pdf_extract(data=data, backend="anydoc")
        assert "to_markdown/to_text" in str(raised.value)

    def test_oxide_is_the_default_lane_byte_identically(self, tmp_path: Path) -> None:
        """"oxide" is the oxide-family engine for PDF (the same mapping
        ``engine_for`` makes) so forcing it changes nothing: byte-identical
        outputs on all four calls, both source spellings where cheap (the
        default's preference for pdf_oxide is a routing choice, never a
        behavior difference between the two spellings). The link walk's
        identity pin rides the link-bearing fixture: the real URI list,
        not the link-less empty shape the bytes-input suite already
        holds."""
        from tors_documents import pdf_link_uris

        path = _materialize(tmp_path, "pdf_two_page")
        data = ENGINES_CORPUS["pdf_two_page"]
        assert pdf_extract(path, backend="oxide") == pdf_extract(path)
        assert pdf_extract(data=data, backend="oxide") == pdf_extract(data=data)
        assert pdf_page_count(path, backend="oxide") == pdf_page_count(path) == 2
        link_path = _materialize(tmp_path, "pdf_link")
        assert pdf_link_uris(link_path, backend="oxide") == pdf_link_uris(link_path) == [
            [PDF_LINK_URL]
        ]
        by_default = pdf_classify(path)
        by_oxide = pdf_classify(path, backend="oxide")
        assert (by_oxide.page_count, by_oxide.page_kinds, by_oxide.pages_needing_ocr) == (
            by_default.page_count,
            by_default.page_kinds,
            by_default.pages_needing_ocr,
        )

    def test_auto_and_the_enum_members_are_the_default_lane(self, tmp_path: Path) -> None:
        """"auto" spelled explicitly, and the ``Backend.AUTO``/
        ``Backend.OXIDE`` members (each member is its accepted string), are
        the default lane on the whole family: the enum vocabulary costs
        nothing at the boundary and never introduces a second routing
        rule."""
        from tors_documents import Backend, pdf_link_uris

        path = _materialize(tmp_path, "pdf_two_page")
        for call in (pdf_extract, pdf_page_count, pdf_link_uris, page_layout):
            for lane in ("auto", Backend.AUTO, Backend.OXIDE):
                assert call(path, backend=lane) == call(path)
        by_default = pdf_classify(path)
        for lane in ("auto", Backend.AUTO, Backend.OXIDE):
            forced = pdf_classify(path, backend=lane)
            assert (forced.page_count, forced.page_kinds, forced.pages_needing_ocr) == (
                by_default.page_count,
                by_default.page_kinds,
                by_default.pages_needing_ocr,
            )

    def test_backend_argument_convention_on_the_pdf_family(self) -> None:
        """The type-vs-value convention, shared with the conversion pair's
        ``parse_backend``: a non-str ``backend=`` (bool, int, float, bytes;
        ``b"anydoc"`` is a str-shaped value of the wrong type, not a
        lane name) is a TypeError naming the argument and repr'ing the
        value; a str outside the vocabulary (the empty string included) is
        a ValueError naming it: raised under the GIL before any work runs,
        so arbitrary bytes carry the pin, no fixture needed."""
        from tors_documents import pdf_link_uris

        for call in (pdf_extract, pdf_page_count, pdf_classify, pdf_link_uris, page_layout):
            for bad in (True, 1, 1.5, b"anydoc"):
                with pytest.raises(TypeError, match="backend must be one of"):
                    call(data=b"junk bytes", backend=bad)
            for bad in ("", "nope"):
                with pytest.raises(ValueError, match="backend must be one of"):
                    call(data=b"junk bytes", backend=bad)

    def test_max_bytes_argument_convention_on_the_pdf_family(self) -> None:
        """The ceiling knob's convention, the conversion pair's pins
        mirrored (``TestAnydocInputCeiling``'s): 0 and -1 are value
        problems ("must be positive"); True, "64KB", and 1.5 are type
        problems naming ``max_bytes=`` (True first, as a type: a bool
        launders through an int extraction as 1); a 10**30 int is
        type-correct, value-absurd: the too-large ValueError."""
        from tors_documents import pdf_link_uris

        for call in (pdf_extract, pdf_page_count, pdf_classify, pdf_link_uris, page_layout):
            for bad in (-1,):
                with pytest.raises(ValueError, match="positive"):
                    call(data=b"junk bytes", max_bytes=bad)
            for bad in (True, "64KB", 1.5):
                with pytest.raises(TypeError, match="max_bytes must be an int"):
                    call(data=b"junk bytes", max_bytes=bad)
            with pytest.raises(ValueError, match="too large"):
                call(data=b"junk bytes", max_bytes=10**30)
        # the ZERO value: the PDF-only probes refuse it at the binding
        # (their family's parse), while page_layout threads it through to
        # the CORE's sequenced check (§6: after resolve/capability/
        # password-kind) — on resolvable bytes it is the zero refusal, on
        # junk the resolve error (resolve first)
        for call in (pdf_extract, pdf_page_count, pdf_classify, pdf_link_uris):
            with pytest.raises(ValueError, match="positive"):
                call(data=b"junk bytes", max_bytes=0)
        with pytest.raises(ValueError, match="max_bytes must be positive"):
            page_layout(data=generate_docx_pgsz(), max_bytes=0)
        with pytest.raises(ValueError, match="(?i)unrecognized"):
            page_layout(data=b"junk bytes", max_bytes=0)

    def test_an_explicit_budget_binds_the_pdf_family_pre_read(self, tmp_path: Path) -> None:
        """The shared source-spine gate, extended to the pdf family: the
        two-page fixture (859 bytes) under ``max_bytes=1`` refuses on
        every call, both source spellings, with the binding's own message
        naming both sizes: the same pre-read refusal the hardening suite
        pins on the conversion pair (the pdf lane's read path,
        ``into_bytes``, now carries the same over-ceiling gate
        ``into_input`` always had). ``max_bytes=None`` converts the same
        fixture unmolested (the unmetered pin: the 32 MiB default is
        the core's post-read check on the anydoc/office_oxide lanes only,
        lanes these PDF-only calls never run) and a roomy budget is
        byte-identical to None."""
        from tors_documents import pdf_link_uris

        path = _materialize(tmp_path, "pdf_two_page")
        data = ENGINES_CORPUS["pdf_two_page"]
        for call in (pdf_extract, pdf_page_count, pdf_classify, pdf_link_uris, page_layout):
            for source in ({"path": path}, {"data": data}):
                with pytest.raises(ValueError, match="ceiling") as raised:
                    call(**source, max_bytes=1)
                message = str(raised.value)
                assert "the document is" in message and "input ceiling is" in message, message
                assert "859 bytes" in message and "1 bytes" in message, message
                assert "max_bytes" in message, message
        assert pdf_extract(path) == pdf_extract(path, max_bytes=10_000_000)
        assert pdf_page_count(data=data) == 2

    def test_the_ceiling_is_inclusive_at_the_boundary(self) -> None:
        """``max_bytes == len(data)`` converts: the off-by-one guard on
        the pre-read gate: a budget exactly the document's size is not
        over it (the comparison is ``size > ceiling``, never ``>=``), one
        byte under is the refusal, both sizes named in the message."""
        data = ENGINES_CORPUS["pdf_two_page"]
        assert pdf_extract(data=data, max_bytes=len(data))[0] == [
            "first page line",
            "second page line",
        ]
        with pytest.raises(ValueError, match="ceiling") as raised:
            pdf_extract(data=data, max_bytes=len(data) - 1)
        assert "859 bytes" in str(raised.value) and "858 bytes" in str(raised.value)

    def test_none_leaves_the_pdf_lane_unmetered_past_the_default_ceiling(
        self, tmp_path: Path
    ) -> None:
        """The budget doctrine's headline pin: ``max_bytes=None`` (the
        default) leaves the pdf lane unmetered: a PDF just over the 32
        MiB default ceiling (the GIL heartbeat cells' own generator, sized
        at 410_000 lines ≈ 33.6 MB to stay fast) page-counts fine with
        both params at rest, while the same file under
        ``max_bytes=<size-1>`` refuses with the ceiling ValueError. The
        32 MiB default is the anydoc/office_oxide lanes' post-read check
        (lanes these PDF-only calls never run) so a regression that wires
        it into the pdf lane (pre-read or post) fails here."""
        path = tmp_path / "big.pdf"
        path.write_bytes(generate_pdf_big(410_000))
        size = path.stat().st_size
        assert size > 32 * 1024 * 1024
        assert pdf_page_count(str(path)) == 1
        assert pdf_page_count(str(path), backend=None, max_bytes=None) == 1
        with pytest.raises(ValueError, match="ceiling") as raised:
            pdf_page_count(str(path), max_bytes=size - 1)
        assert "max_bytes" in str(raised.value)

    def test_aio_carries_the_new_params_through_the_thread_hop(self, tmp_path: Path) -> None:
        """aio parity for the new params (the existing aio gates' loop
        pattern, both the payload's aio and the base shim's re-export),
        all four functions: ``backend="oxide"`` through the hop is the
        default lane's byte-identical answer, ``backend="anydoc"`` the
        same capability refusal (each call's own capability named), and
        an explicit budget the same pre-read ceiling: the awaitable
        spellings wrap the typed wrappers, so the contract rides along
        unchanged."""
        from tors_documents import aio as payload_aio
        from tors_documents import pdf_extract as sync_extract

        from tors.documents import aio as shim_aio

        path = _materialize(tmp_path, "pdf_two_page")
        capabilities = {
            "pdf_extract": "the per-page text probe",
            "pdf_page_count": "the page tree",
            "pdf_classify": "per-page classification",
            "pdf_link_uris": "the /Annots link walk",
        }
        for aio_module in (payload_aio, shim_aio):
            assert asyncio.run(aio_module.pdf_extract(path, backend="oxide")) == sync_extract(path)
            for name, capability in capabilities.items():
                call = getattr(aio_module, name)
                with pytest.raises(ValueError, match=f"cannot serve {capability}"):
                    asyncio.run(call(path, backend="anydoc"))
                with pytest.raises(ValueError, match="ceiling"):
                    asyncio.run(call(path, max_bytes=1))


# --- in-memory bytes input + the annotation link walk (red-team lane) ------


class TestBytesInputAndLinkWalk:
    """The ``data=`` entry and ``pdf_link_uris``: the two surfaces the
    downstream audit demanded (the bytes-in pipeline caller holds uploads
    in memory and pays a temp-file roundtrip per document against a
    path-only API; the link walk is the raw URI surface no text rendering
    carries). ``data=`` is the same conversion: byte-identical output, the
    same typed answers, the same error taxonomy: only the source differs.
    The O(n) copy a ``data=`` call pays rides inside the detach with the
    rest of the pass (a 400 MB call's max heartbeat gap measured ~1.1 ms,
    pinned red-green by the hardening suite's probe).    """

    def test_data_conversions_are_byte_identical_to_path_conversions(self, tmp_path: Path) -> None:
        """The in-memory entry is not a second implementation: for a
        marker-bearing format (docx) and a PDF alike, the ``data=`` call
        returns exactly the ``path=`` call's pair: same resolved Format
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
        (``format=``) instead of a path: the bytes-in half of the
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
        answers as ``path=`` (the probe, the page tree, the preflight,
        and the link walk) one source-resolution spine, four functions."""
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
        TypeError naming the argument: on the convert lane and the PDF
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
        document without link annotations yields empty lists per page:
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
        """The upload route's two-step: sniff the leading bytes (the true
        cost is a container parse, not a marker scan: anydoc's detect
        opens the ZIP/OLE package and reads its metadata; a 120 KiB zip
        measured 267 MiB peak RSS to answer docx), then convert the full
        body in memory: no temp file anywhere in the flow, and both
        halves agree on what the bytes are."""
        from tors_documents import Format

        data = Path(_materialize(tmp_path, "pdf_link")).read_bytes()
        assert sniff(data[:1024]) == Format.PDF
        resolved, markdown = to_markdown(data=data)
        assert resolved == Format.PDF
        assert f"[{PDF_LINK_LABEL}]({PDF_LINK_URL})" in markdown

    def test_aio_carries_data_and_the_link_walk(self, tmp_path: Path) -> None:
        """The async twins pass the source pair through unchanged (a
        thread hop must not reintroduce a temp file) and carry the new
        link walk: ``data=`` in, typed answers out: both the payload's
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
    fpdf2 writes the page, pypdf rewrites it encrypted: RC4-128 because
    it is pure-python in pypdf (AES would drag in the cryptography
    package, which the dev group does not carry). pypdf salts the
    encryption randomly per call, so no test may assert on these bytes:
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
    named: never empty output (the pre-fix masking: to_markdown and
    pdf_extract silently returned "" on a locked PDF while classify
    raised, so a routing caller read "no text" where the truth was "no
    access"; the door check now sits in front of every engine lane)."""

    def test_every_entry_fails_closed_without_a_password(self) -> None:
        """All eight entries (both convert modes, all four PDF-only
        calls, and the two geometry-lane probes) refuse a locked PDF with
        pdf_oxide's ValueError naming the encryption, before any engine
        work. The defect this gate makes
        impossible to regress: the pre-fix wheel's to_markdown/pdf_extract
        returned EMPTY output on these same bytes (the empty-string
        masking: indistinguishable from a blank document); today no entry
        returns at all."""
        from tors_documents import page_geometry, page_layout, pdf_link_uris

        locked = _encrypted_pdf()
        entries = (
            to_markdown,
            to_text,
            page_geometry,
            page_layout,
            pdf_classify,
            pdf_extract,
            pdf_page_count,
            pdf_link_uris,
        )
        for entry in entries:
            with pytest.raises(ValueError, match="encrypted and requires a password"):
                entry(data=locked)

    def test_a_wrong_password_is_refused_not_treated_as_none(self) -> None:
        """A wrong password must be distinguishable from no password:
        pdf_oxide's authenticate() refusal ("did not unlock") on the
        convert lane and the probe lanes alike (a wrong password laundered
        into the generic no-access error would send the caller hunting
        the wrong knob)."""
        locked = _encrypted_pdf()
        for entry in (to_markdown, pdf_extract, pdf_classify):
            with pytest.raises(ValueError, match="did not unlock"):
                entry(data=locked, password="not-the-password")

    def test_the_right_password_converts_every_lane(self) -> None:
        """The unlock: with the right password the document is exactly the
        document: markdown carries the cell text, classify says one TEXT
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
        would know: the refusal names the parameter and the format the
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
        binding before any bytes are sniffed or read: so arbitrary bytes
        carry the pin, no fixture needed."""
        with pytest.raises(TypeError, match="password must be a str"):
            to_markdown(data=b"junk bytes", password=123)
        with pytest.raises(TypeError, match="password must be a str"):
            pdf_page_count(data=b"junk bytes", password=123)

    def test_password_on_the_anydoc_pdf_lane_is_refused_not_ignored(self, tmp_path: Path) -> None:
        """anydoc's PDF reader takes no password, so a correct password on
        backend="anydoc" would otherwise fail as a bare "document is
        encrypted": indistinguishable from having passed nothing, the
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
        and unlock with the right one: in the payload's aio and the base
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


# --- the page-geometry probe (the geometry lane's contract) -----------------------


# The §6 length rule per engines-corpus kind: PDF kinds align with
# pdf_page_count (asserted, not tabled); docx/docm are one converter
# section; xlsx/xlsm one worksheet; the pptx pair two slides; every
# single-flow kind exactly one. No committed-corpus fixture declares a
# page size, so every entry is None — the Some pins live on the dedicated
# fixtures below.
_SINGLE_FLOW_KINDS = (
    "rtf",
    "odt_rich",
    "ods_rich",
    "odp_rich",
    "epub_rich",
    "csv_rich",
    "html_rich",
)
_UNIT_COUNTS = {
    "docx": 1,
    "docx_rich": 1,
    "docm_rich": 1,
    "xlsx": 1,
    "xlsm": 1,
    "pptx_rich": 2,
    "ppsx_rich": 2,
    **{kind: 1 for kind in _SINGLE_FLOW_KINDS},
}
# The one alias the geometry lane cannot read: ppsx_rich resolves as the
# pptx kind, whose §6 lane is office_oxide — and office_oxide refuses the
# slideshow content type (the conversion lane's auto route answers with
# anydoc, a lane geometry has no answer on; never a silent fallback). The
# refusal is the typed ValueError, pinned below.
_REFUSES_ON_THE_GEOMETRY_LANE = {"ppsx_rich"}


class TestPageGeometry:
    """The geometry probe's contract, held over the fixture corpus and
    the dedicated geometry fixtures: the length law (§9 item 10's
    property sweep — PDF lengths == pdf_page_count, office lengths the
    §6 matrix's counts, single-flow exactly one), the entry values (the
    raw-layer and PDF goldens), the backend matrix (auto == oxide;
    anydoc refused where geometry is obtainable, answered [None]
    elsewhere), and the shared source/refusal spine."""

    @pytest.mark.parametrize("kind", sorted(ENGINES_CORPUS), ids=lambda kind: kind)
    def test_lengths_follow_the_matrix_and_the_entry_kinds(
        self, tmp_path: Path, kind: str
    ) -> None:
        """§9 item 10's property test, over the whole fixture corpus: the
        list length is the §6 rule (pdf: the page count; docx: the
        converter's section count; pptx: the slide count; xlsx: the raw
        worksheet count; single-flow: exactly one). PDF pages declare
        their MediaBox, so their entries are Some; no committed office or
        single-flow fixture declares a page size, so theirs are None."""
        path = _materialize(tmp_path, kind)
        if kind in _REFUSES_ON_THE_GEOMETRY_LANE:
            # The alias's length law IS the refusal: the ppsx container
            # resolves as the pptx kind, office_oxide refuses the
            # slideshow content type, and the typed ValueError is the
            # answer — never a silent skip (the full refusal is pinned by
            # the dedicated alias test below).
            with pytest.raises(ValueError, match="(?i)format mismatch") as raised:
                page_geometry(path)
            assert "expected a PresentationML presentation" in str(raised.value)
            return
        entries = page_geometry(path)
        assert len(entries) >= 1, "only a zero-unit document may answer []"
        if kind.startswith("pdf"):
            assert len(entries) == pdf_page_count(path), kind
            for entry in entries:
                assert entry is not None, kind
                assert entry.width_pt > 0 and entry.height_pt > 0, kind
                assert entry.rotation == 0, kind
                assert entry.crop is None, kind
        else:
            assert len(entries) == _UNIT_COUNTS[kind], kind
            assert all(entry is None for entry in entries), kind

    def test_the_ppsx_alias_refuses_cleanly_on_the_geometry_lane(
        self, tmp_path: Path
    ) -> None:
        """The known alias divergence, pinned (the conversion lane's own
        forced-oxide pin, mirrored here): the ppsx container resolves as
        the pptx kind, the geometry lane is office_oxide, and
        office_oxide refuses the slideshow content type — a typed
        ValueError, never a silent fallback and never a crash. Under
        backend='anydoc' the capability refusal fires first (the pinned
        order)."""
        path = _materialize(tmp_path, "ppsx_rich")
        with pytest.raises(ValueError, match="(?i)format mismatch") as raised:
            page_geometry(path)
        assert "expected a PresentationML presentation" in str(raised.value)
        with pytest.raises(ValueError, match="cannot serve page geometry for pptx"):
            page_geometry(path, backend="anydoc")

    def test_pdf_pages_align_with_the_page_count_probe(self, tmp_path: Path) -> None:
        """The length law's PDF half, explicit: one entry per page of the
        page-count probe, index == page number (the content-id contract
        the full-length surface keeps)."""
        path = _materialize(tmp_path, "pdf_mixed")
        entries = page_geometry(path)
        assert len(entries) == pdf_page_count(path) == 2
        for entry in entries:
            assert entry is not None
            assert entry.width_pt > 0 and entry.height_pt > 0

    def test_the_pdf_geometry_goldens(self, tmp_path: Path) -> None:
        """The probe goldens, through the Python surface: rotation 0 is
        the identity (612×792 verbatim — and the crop corners come back
        y-up, never mirrored); the crop-subset fixture reads 400×600 with
        crop Some((100, 100, 500, 700)); a /Rotate 90 page swaps the
        render extent (792×612); a non-right-angle /Rotate reports 0 in
        the unrotated frame; the zero-page document is the one [] answer."""
        data = generate_pdf_crop()
        (entry,) = page_geometry(data=data)
        assert (entry.width_pt, entry.height_pt) == (400.0, 600.0)
        assert entry.rotation == 0
        assert entry.crop == (100.0, 100.0, 500.0, 700.0)  # y-up, no flip

        (rotated,) = page_geometry(data=generate_pdf_rotate(90))
        assert (rotated.width_pt, rotated.height_pt) == (792.0, 612.0)
        assert rotated.rotation == 90
        (upside_down,) = page_geometry(data=generate_pdf_rotate(180))
        assert (upside_down.width_pt, upside_down.height_pt) == (612.0, 792.0)

        (askew,) = page_geometry(data=generate_pdf_rotate(45))
        assert (askew.width_pt, askew.height_pt) == (612.0, 792.0)
        assert askew.rotation == 0  # a non-right-angle /Rotate reports 0

        (origin,) = page_geometry(data=generate_pdf_media_origin())
        assert (origin.width_pt, origin.height_pt) == (400.0, 600.0)
        assert origin.crop == (100.0, 100.0, 500.0, 700.0)  # corners, not extents

        assert page_geometry(data=generate_pdf_zero_pages()) == []

    def test_the_partial_overlap_crop_is_the_intersection(self) -> None:
        """The partial-overlap golden (the red-team probe, Python side):
        a CropBox hanging off the MediaBox's edge ([300 300 800 900] on
        [0 0 612 792]) crops to the INTERSECTION — the engine only
        exposes the intersection, so the entry is its extent (312×492)
        and its normalized corners as the crop provenance, never the
        declared box's own 500×600. The crop wording ("the declared
        CropBox reduced to the MediaBox — the effective crop") is this
        fixture's shape."""
        (entry,) = page_geometry(data=generate_pdf_crop_partial())
        assert (entry.width_pt, entry.height_pt) == (312.0, 492.0)
        assert entry.rotation == 0
        assert entry.crop == (300.0, 300.0, 612.0, 792.0)

    def test_docx_geometry_reads_the_raw_layer(self) -> None:
        """§9 item 8: a pgSz-bearing section is Some(612.0, 792.0) exact
        (twips ÷ 20); a margins-only section is None — the raw-layer
        amendment's own pinning test, which FAILS under the to_ir() path
        that fabricates Letter for it; a `<w:pgSz>` without attributes
        reports the engine's Letter default (the named engine
        limitation); office entries always carry rotation 0 and no crop."""
        (sized,) = page_geometry(data=generate_docx_pgsz())
        assert isinstance(sized, PageGeometry)
        assert (sized.width_pt, sized.height_pt) == (612.0, 792.0)
        assert sized.rotation == 0 and sized.crop is None
        assert page_geometry(data=generate_docx_margins_only()) == [None]
        # the attribute-less pgSz: the engine's Letter default, documented
        import io
        import zipfile

        with zipfile.ZipFile(io.BytesIO(generate_docx_pgsz())) as src:
            entries = [(i.filename, src.read(i.filename)) for i in src.infolist()]
        stripped = [
            (
                name,
                data.replace(b' w:w="12240" w:h="15840"', b"")
                if name == "word/document.xml"
                else data,
            )
            for name, data in entries
        ]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in stripped:
                archive.writestr(name, data)
        (bare,) = page_geometry(data=buffer.getvalue())
        assert (bare.width_pt, bare.height_pt) == (612.0, 792.0)

    def test_pptx_geometry_is_the_raw_emu_layer(self) -> None:
        """§9 item 8's PPTX pins: a declared sldSz (12192000×6858000 EMU)
        is Some(960.0, 540.0) exactly (EMU ÷ 12700.0 in float), one entry
        per slide; an undeclared deck is None everywhere (the corpus's
        rich pptx); the 0-slide deck is the other [] answer."""
        entries = page_geometry(data=generate_pptx_sldsz())
        assert len(entries) == 2
        for entry in entries:
            assert entry is not None
            assert (entry.width_pt, entry.height_pt) == (960.0, 540.0)
            assert entry.rotation == 0 and entry.crop is None
        assert page_geometry(data=generate_pptx_empty()) == []

    def test_xlsx_geometry_is_none_at_the_raw_worksheet_count(self) -> None:
        """§9 item 8/10's XLSX doctrine pin: a workbook whose chart text
        gives the IR a "Charts" section (so the converter's section count
        is 2) still answers ONE entry — the raw worksheet count — and
        None (a pageSetup-bearing sheet would be None anyway: the
        worksheet is an unbounded grid, not a page)."""
        entries = page_geometry(data=generate_xlsx_chart())
        assert entries == [None]
        # the witness that the IR count really did run to two: the chart
        # text surfaces on the oxide conversion lane
        _, markdown = to_markdown(data=generate_xlsx_chart(), backend="oxide")
        assert "Units per quarter" in markdown

    def test_the_legacy_trio_fixtures_answer_all_none_at_the_ir_count(
        self,
    ) -> None:
        """§4's legacy row, end to end on real OLE bytes (the committed
        soffice-generated fixtures under tests/legacy_corpus/ — no
        in-test builder can fake a [MS-CFB] container; see that
        directory's README for provenance): t.doc → [None] (the one
        document section), book3.xls → [None, None, None] (one IR section
        per worksheet), deck3.ppt → [None, None, None] (one per slide).
        The xls fixture is ALSO the red-team blocker's regression pin:
        the pre-fix leg demanded the xlsx raw layer of every Excel-kind
        parse and raised on exactly these genuine bytes ('the xlsx parse
        lost its raw layer') instead of answering."""
        fixtures = Path(__file__).resolve().parent / "legacy_corpus"
        expected = {
            "t.doc": [None],
            "book3.xls": [None, None, None],
            "deck3.ppt": [None, None, None],
        }
        for name, want in expected.items():
            entries = page_geometry(str(fixtures / name))
            assert entries == want, name
        # and the container decision rides the bytes, not the name: the
        # same workbook named .xlsx still routes the legacy parser
        # (all-None at three), with the container's own name reported
        assert page_geometry(str(fixtures / "book3.xls"), format="xlsx") == [
            None,
            None,
            None,
        ]
        assert sniff((fixtures / "book3.xls").read_bytes()) == "xls"

    def test_single_flow_formats_answer_one_none_on_every_backend(
        self, tmp_path: Path
    ) -> None:
        """§9 item 9: the no-geometry row is the backend-independent
        [None] — one entry, unit = whole document, for auto, oxide, AND
        anydoc (no refusal: nothing runs, nothing is fallen back) — the
        same by content-marker sniff and by explicit format= name."""
        for kind in _SINGLE_FLOW_KINDS:
            path = _materialize(tmp_path, kind)
            name = _format_name(kind)
            for backend in ("auto", "oxide", "anydoc"):
                entries = page_geometry(path, backend=backend)
                assert entries == [None], f"{kind} x {backend}"
            assert page_geometry(path, format=name) == [None], kind
            assert page_geometry(data=ENGINES_CORPUS[kind], backend="oxide") == [None], kind

    def test_auto_and_oxide_are_the_same_lane_compared_as_pyclasses(
        self, tmp_path: Path
    ) -> None:
        """auto == oxide on this surface for every format — a documented
        divergence from Backend.AUTO's conversion contract — and the
        equality is the native pyclass's structural __eq__ (list ==
        compares entries with ==)."""
        for kind in ("pdf_two_page", "docx_rich", "pptx_rich", "xlsx"):
            path = _materialize(tmp_path, kind)
            assert page_geometry(path) == page_geometry(path, backend="oxide"), kind
            assert page_geometry(path, backend="auto") == page_geometry(
                path, backend="auto"
            ), kind
        # __eq__ is structural: a different document's geometry differs
        crop, plain = (
            page_geometry(data=generate_pdf_crop())[0],
            page_geometry(data=generate_pdf_rotate(0))[0],
        )
        assert crop is not None and plain is not None
        assert crop != plain
        assert crop == page_geometry(data=generate_pdf_crop())[0]

    def test_anydoc_refusals_are_format_aware_and_pinned(self, tmp_path: Path) -> None:
        """The capability refusal: on a PDF it is the historical PDF
        doctrine text with this probe's capability named; on an office
        format it is the second message, byte-pinned per the spec. The
        refusal precedes the encrypted door (a locked PDF under anydoc
        surfaces the refusal), precedes the password-kind refusal, and
        never fires on the no-geometry row (the test above)."""
        path = _materialize(tmp_path, "pdf_two_page")
        with pytest.raises(ValueError) as raised:
            page_geometry(path, backend="anydoc")
        pdf_message = str(raised.value)
        assert 'backend "anydoc" cannot serve the declared page-box geometry' in pdf_message
        assert 'to_markdown/to_text with backend="anydoc"' in pdf_message
        assert "backend='auto' or 'oxide'" in pdf_message

        office_path = _materialize(tmp_path, "docx_rich")
        with pytest.raises(ValueError) as raised:
            page_geometry(office_path, backend="anydoc")
        assert str(raised.value) == (
            'backend "anydoc" cannot serve page geometry for docx: anydoc has no '
            "geometry surface; use backend='auto' or 'oxide' here"
        )
        # the refusal names the CONTAINER the bytes are (resolved_name's
        # one container-aware case): a legacy OLE workbook under anydoc
        # says xls, never the Excel kind's default xlsx spelling
        ole_workbook = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 4096
        with pytest.raises(
            ValueError,
            match=r'cannot serve page geometry for xls',
        ):
            page_geometry(data=ole_workbook, format="xlsx", backend="anydoc")
        # the refusal precedes the door: an encrypted PDF under anydoc
        locked = _encrypted_pdf()
        with pytest.raises(ValueError, match="cannot serve the declared page-box geometry"):
            page_geometry(data=locked, backend="anydoc")
        # ...and precedes the password-kind refusal
        with pytest.raises(ValueError, match="cannot serve page geometry for docx"):
            page_geometry(office_path, backend="anydoc", password="torque")

    def test_password_on_a_non_pdf_format_is_refused(self, tmp_path: Path) -> None:
        """A password on a non-PDF is refused on this surface too (never a
        silently-ignored argument), naming the parameter and the format —
        on the office row and the no-geometry row alike."""
        office_path = _materialize(tmp_path, "docx_rich")
        with pytest.raises(
            ValueError, match=r'password= applies to PDF documents only, not "docx"'
        ):
            page_geometry(office_path, password="torque")
        csv_path = _materialize(tmp_path, "csv_rich")
        with pytest.raises(
            ValueError, match=r'password= applies to PDF documents only, not "csv"'
        ):
            page_geometry(csv_path, password="torque")

    def test_encrypted_pdfs_fail_closed_and_owner_passwords_read(
        self,
    ) -> None:
        """§9 item 6 on the geometry probe: a locked PDF is a fail-closed
        ValueError (never empty output, never a partial list); the right
        password unlocks (the page's geometry comes through); a wrong one
        is its own error."""
        locked = _encrypted_pdf()
        with pytest.raises(ValueError, match="encrypted and requires a password"):
            page_geometry(data=locked)
        with pytest.raises(ValueError, match="did not unlock"):
            page_geometry(data=locked, password="not-the-password")
        unlocked = page_geometry(data=locked, password="torque")
        assert len(unlocked) == 1
        assert unlocked[0] is not None
        assert unlocked[0].width_pt > 0 and unlocked[0].height_pt > 0

    def test_an_encrypted_ooxml_container_is_a_typed_refusal_never_none(
        self,
    ) -> None:
        """§9 item 6's office half: an encrypted OOXML package is a CFB
        container — through the geometry lane it is a typed ValueError
        naming the encryption, never None, never [], never a password
        prompt; and a mis-sniff as the legacy xls (the OLE magic's own
        resolution) fails the same typed way."""
        ole = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 128
        with pytest.raises(ValueError, match="(?i)encrypted"):
            page_geometry(data=ole, format="docx")
        with pytest.raises(ValueError):
            page_geometry(data=ole, format="xlsx")  # the mis-sniff: legacy parse, typed failure

    def test_the_argument_contracts_match_the_family(self, tmp_path: Path) -> None:
        """The shared spine's refusals: a missing file is an OSError, a
        garbage document a ValueError, a bad backend name a ValueError
        naming backend — on the geometry probe as on the conversion pair."""
        with pytest.raises(OSError):
            page_geometry(str(tmp_path / "nope.pdf"))
        with pytest.raises(ValueError):
            page_geometry(data=b"not a document at all")
        with pytest.raises(ValueError, match="(?i)backend"):
            page_geometry(
                _materialize(tmp_path, "pdf_two_page"), backend="turbo"
            )
        with pytest.raises(TypeError, match="(?i)backend"):
            page_geometry(data=b"junk", backend=123)

    def test_format_and_data_parity_follow_the_matrix(self, tmp_path: Path) -> None:
        """The format=/data= parity: for every matrix row, the data= call
        (resolved by sniff, or by format= where the bytes are
        signature-less) answers byte-identically to the path= call, and
        an explicit format= never changes the answer. The ppsx alias's
        rows are the refusal's parity: both lanes, the same typed
        ValueError."""
        for kind in sorted(ENGINES_CORPUS):
            path = _materialize(tmp_path, kind)
            data = Path(path).read_bytes()
            name = _format_name(kind)
            if kind in _REFUSES_ON_THE_GEOMETRY_LANE:
                with pytest.raises(ValueError):
                    page_geometry(path)
                with pytest.raises(ValueError):
                    page_geometry(data=data)
                with pytest.raises(ValueError):
                    page_geometry(data=data, format=name)
                continue
            by_path = page_geometry(path)
            assert page_geometry(data=data) == by_path, kind
            assert page_geometry(data=data, format=name) == by_path, kind

    def test_max_bytes_binds_on_the_geometry_lane_too(self, tmp_path: Path) -> None:
        """§9 item 11: zero is refused as nonsense; an explicit budget
        binds pre-read (path and data, the same ceiling ValueError); and
        None keeps the 32 MiB default as the OFFICE leg's post-read check
        — a 33 MiB docx refuses by default, converts with a raised
        budget — while the pdf lane stays unmolested by it."""
        path = _materialize(tmp_path, "pdf_two_page")
        data = Path(path).read_bytes()
        with pytest.raises(ValueError, match="(?i)positive"):
            page_geometry(data=data, max_bytes=0)
        for source in ({"path": path}, {"data": data}):
            with pytest.raises(ValueError, match="ceiling") as raised:
                page_geometry(**source, max_bytes=1)
            assert "max_bytes" in str(raised.value)
        assert page_geometry(path, max_bytes=len(data)) == page_geometry(path)

        # the office leg's default ceiling: a >32 MiB docx refuses under
        # None (post-read, the office_oxide lane named) and converts with
        # a raised budget. The fat part is STORED (no compression): the
        # ceiling keys on the input bytes, and a deflated filler would
        # shrink the fixture far under it.
        import io
        import zipfile

        with zipfile.ZipFile(io.BytesIO(ENGINES_CORPUS["docx_rich"])) as src:
            entries = [(i.filename, src.read(i.filename)) for i in src.infolist()]
        filler = b"<w:p><w:r><w:t>geometry filler paragraph</w:t></w:r></w:p>" * 700_000
        fattened = [
            (
                name,
                data.replace(b"</w:body>", filler + b"</w:body>")
                if name == "word/document.xml"
                else data,
            )
            for name, data in entries
        ]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
            for name, part in fattened:
                archive.writestr(name, part)
        fat = buffer.getvalue()
        assert len(fat) > 32 * 1024 * 1024
        fat_path = tmp_path / "fat_geometry.docx"
        fat_path.write_bytes(fat)
        # path= under None: the shared source spine meters the office lane
        # during the read (the docx conversion lane's 32 MiB), so the
        # refusal is the read-phase one
        with pytest.raises(ValueError, match="default read ceiling") as raised:
            page_geometry(str(fat_path))
        # data= under None: the resident bytes cannot grow, so the read
        # passes and the CORE's post-read office-leg default fires — the
        # office_oxide lane named by name
        with pytest.raises(ValueError, match="input ceiling") as raised:
            page_geometry(data=fat)
        assert "office_oxide" in str(raised.value)

    def test_the_read_ceiling_keys_on_the_geometry_table_not_the_conversion_table(
        self, tmp_path: Path
    ) -> None:
        """§6's ceiling law on this surface: only the office row meters
        at the 32 MiB default. A 40 MiB HTML is answered [None] on BOTH
        lanes (the probed violation: the conversion table's ceiling
        refused the path= call during the read while data= answered —
        parity broken, and against a row on which no engine runs), and a
        34 MiB PDF under backend='anydoc' surfaces the capability
        refusal, not the ceiling refusal (the probed violation: the
        conversion routing's anydoc lane metered a read this surface
        never runs) — refused before the read, under the GIL, exactly
        where the sibling PDF probes refuse."""
        fat_html = (
            b"<!DOCTYPE html><html><body><p>x</p>"
            + b" " * (40 * 1024 * 1024)
            + b"</body></html>"
        )
        html_path = tmp_path / "big.html"
        html_path.write_bytes(fat_html)
        # path/data parity on the no-geometry row, with and without format=
        assert page_geometry(str(html_path)) == [None]
        assert page_geometry(data=fat_html) == [None]
        assert page_geometry(str(html_path), format="html") == [None]
        assert page_geometry(data=fat_html, format="html") == [None]
        # and the row answers [None] for anydoc too — refused nowhere
        assert page_geometry(str(html_path), backend="anydoc") == [None]

        fat_pdf = generate_pdf_big(410_000)  # ≈ 33.6 MB > the 32 MiB default
        assert len(fat_pdf) > 32 * 1024 * 1024
        pdf_path = tmp_path / "big_geometry.pdf"
        pdf_path.write_bytes(fat_pdf)
        for source in ({"path": str(pdf_path)}, {"data": fat_pdf}):
            with pytest.raises(ValueError) as raised:
                page_geometry(**source, backend="anydoc")
            assert "cannot serve the declared page-box geometry" in str(raised.value)
            assert "read ceiling" not in str(raised.value)
        # the pdf row's own read is unmetered on this surface: the same
        # file answers under the default budget
        assert len(page_geometry(str(pdf_path))) == 1

    def test_the_refusal_order_matches_the_locked_precedence(self, tmp_path: Path) -> None:
        """§6's amended order, pinned on the Python surface (the core
        test only sees the core): resolve/format → the capability refusal
        (under the GIL, before the detach) → the password-kind refusal →
        max_bytes → the encrypted door. The red-team probes: a zero
        max_bytes must not mask the capability refusal (nor the
        password-kind refusal behind it), and a resolve failure on
        garbage precedes everything (a capability refusal cannot name a
        format it has not resolved)."""
        docx = generate_docx_pgsz()
        # the capability refusal precedes the zero-max_bytes refusal...
        with pytest.raises(ValueError, match="cannot serve page geometry for docx"):
            page_geometry(data=docx, backend="anydoc", max_bytes=0)
        # ...and precedes the password-kind refusal that would otherwise
        # sit between them (the probed: max_bytes answered first)
        with pytest.raises(ValueError, match="cannot serve page geometry for docx"):
            page_geometry(data=docx, backend="anydoc", max_bytes=0, password="torque")
        # auto backend: capability passes, the password-kind refusal
        # precedes the zero refusal
        with pytest.raises(
            ValueError, match=r'password= applies to PDF documents only, not "docx"'
        ):
            page_geometry(data=docx, max_bytes=0, password="torque")
        # the zero refusal itself, sequenced last of the three — the
        # core's own message, on the data lane and the path lane alike
        # (the binding's parse threads zero through on this surface)
        with pytest.raises(ValueError, match="max_bytes must be positive"):
            page_geometry(data=docx, max_bytes=0)
        zero_path = tmp_path / "zero_budget.docx"
        zero_path.write_bytes(docx)
        with pytest.raises(ValueError, match="max_bytes must be positive"):
            page_geometry(str(zero_path), max_bytes=0)
        # and under anydoc the same path call surfaces the capability
        # refusal, never the zero
        with pytest.raises(ValueError, match="cannot serve page geometry for docx"):
            page_geometry(str(zero_path), backend="anydoc", max_bytes=0)
        # resolve/format precedes the capability refusal: garbage has no
        # format to name, so the resolve error is the answer
        with pytest.raises(ValueError, match="(?i)unrecognized"):
            page_geometry(data=b"garbage", backend="anydoc")
        # ...even when max_bytes is nonsense too: resolve first (§6)
        with pytest.raises(ValueError, match="(?i)unrecognized"):
            page_geometry(data=b"garbage", backend="anydoc", max_bytes=0)
        # the door stays last: an encrypted PDF under a zero budget
        # surfaces the zero refusal, never the encrypted door
        with pytest.raises(ValueError, match="max_bytes must be positive"):
            page_geometry(data=_encrypted_pdf(), max_bytes=0)

    def test_aio_carries_the_geometry_lane(self, tmp_path: Path) -> None:
        """The awaitable spellings (both aio modules) wrap the typed
        wrapper: the same entries, the same refusals, the same budget
        behavior through the thread hop."""
        from tors_documents import aio as payload_aio

        from tors.documents import aio as shim_aio

        path = _materialize(tmp_path, "pdf_two_page")
        expected = page_geometry(path)
        for aio_module in (payload_aio, shim_aio):
            assert asyncio.run(aio_module.page_geometry(path)) == expected
            assert asyncio.run(aio_module.page_geometry(path, backend="oxide")) == expected
            with pytest.raises(ValueError, match="cannot serve the declared page-box geometry"):
                asyncio.run(aio_module.page_geometry(path, backend="anydoc"))
            with pytest.raises(ValueError, match="ceiling"):
                asyncio.run(aio_module.page_geometry(path, max_bytes=1))

    def test_concurrent_geometry_calls_are_identical(self, tmp_path: Path) -> None:
        """The concurrency identity on the new surface (the §9 item-12
        property's load-insensitive half): 8 threads, one geometry call
        each over three formats — every thread's answer equals the
        single-threaded one."""
        paths = [
            _materialize(tmp_path, kind)
            for kind in ("pdf_two_page", "docx_rich", "pptx_rich")
        ] * 2
        paths += [_materialize(tmp_path, "xlsx"), _materialize(tmp_path, "csv_rich")]
        expected = [page_geometry(path) for path in paths]
        results: list[object] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(len(paths))

        def worker(path: str) -> None:
            try:
                barrier.wait()
                results.append(page_geometry(path))
            except BaseException as exc:  # noqa: BLE001 -- recorded, asserted below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(path,)) for path in paths]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        assert sorted(map(repr, results)) == sorted(map(repr, expected))


# tolerance for the f32-borne box goldens on the Python surface (the
# engine's advance widths are f32; the goldens are the probe's, ±0.01)
_LAYOUT_TOL = 0.01


def _box_near(
    box: tuple[float, float, float, float],
    expected: tuple[float, float, float, float],
) -> None:
    assert all(
        abs(a - b) <= _LAYOUT_TOL for a, b in zip(box, expected, strict=True)
    ), f"expected {expected}, got {box}"


class TestPageLayout:
    """The layout probe's contract (spec §5, ruling D1–D4/D6): every box
    in the page's declared render space (bottom-left origin, y-up,
    /Rotate applied, unclipped), the probe goldens at box level, the
    join/union laws, the handles-only image lane, pages='s byte-pinned
    refusals and reorder-with-page_index law, auto==oxide as plain `==`,
    and the shared refusal spine."""

    def test_the_rot0_identity_golden_no_y_flip(self) -> None:
        """The D2 resolution made permanent on the Python surface: the run
        at Td(72,700), 24pt lands at (72, 700, ~54.67, 24) — span, line,
        and word alike — with NO y-flip anywhere."""
        (page,) = page_layout(data=generate_pdf_layout_baseline())
        assert page.page_index == 0
        assert (page.geometry.width_pt, page.geometry.height_pt) == (612.0, 792.0)
        assert page.geometry.rotation == 0 and page.geometry.crop is None
        (span, world) = (page.spans[0], page.spans[1])
        assert span.text == "Hello" and world.text == "World"
        _box_near(span.bbox, (72.0, 700.0, 54.67, 24.0))
        _box_near(world.bbox, (72.0, 100.0, 31.33, 12.0))
        assert (span.font_size, span.font_name) == (24.0, "Helvetica")
        assert not (span.is_bold or span.is_italic or span.is_artifact)
        assert span.rotation_degrees == 0.0
        (line,) = (line for line in page.lines if line.text == "Hello")
        _box_near(line.bbox, (72.0, 700.0, 54.67, 24.0))
        assert line.words[0].text == "Hello"
        _box_near(line.words[0].bbox, (72.0, 700.0, 54.67, 24.0))

    def test_the_quarter_turn_goldens(self) -> None:
        """§9 item 3's forward-rule goldens through Python: at 90 the
        horizontal run maps to (700, 612−72−54.67, 24, 54.67) with the
        page swapped to 792×612; at 180 to (485.33, 68, 54.67, 24); a
        non-right-angle /Rotate 45 reports rotation 0 in the unrotated
        frame (T10)."""
        (rotated,) = page_layout(data=generate_pdf_layout_rotated(90))
        assert (rotated.geometry.width_pt, rotated.geometry.height_pt) == (792.0, 612.0)
        assert rotated.geometry.rotation == 90
        _box_near(rotated.spans[0].bbox, (700.0, 485.33, 24.0, 54.67))
        _box_near(rotated.lines[0].bbox, (700.0, 485.33, 24.0, 54.67))
        _box_near(rotated.lines[0].words[0].bbox, (700.0, 485.33, 24.0, 54.67))
        (upside_down,) = page_layout(data=generate_pdf_layout_rotated(180))
        assert upside_down.geometry.rotation == 180
        _box_near(upside_down.spans[0].bbox, (485.33, 68.0, 54.67, 24.0))
        (askew,) = page_layout(data=generate_pdf_layout_rotated(45))
        assert askew.geometry.rotation == 0
        assert (askew.geometry.width_pt, askew.geometry.height_pt) == (612.0, 792.0)
        _box_near(askew.spans[0].bbox, (72.0, 700.0, 54.67, 24.0))

    def test_the_crop_goldens(self) -> None:
        """§9 item 1 + item 2's layout pins: on the [100 100 500 700] crop
        the Inside run lands at (50, 500, 64.03, 24), the Outside run is
        ABSENT (the engine's drop law), the page is 400×600 with
        crop=Some((100, 100, 500, 700)); a span hanging past the crop edge
        is kept and UNCLIPPED (T7); the nonzero MediaBox origin is
        translated away (a run at the box's own origin lands at (0, 0) —
        the origin law)."""
        (cropped,) = page_layout(data=generate_pdf_layout_crop())
        assert (cropped.geometry.width_pt, cropped.geometry.height_pt) == (400.0, 600.0)
        assert cropped.geometry.crop == (100.0, 100.0, 500.0, 700.0)
        assert len(cropped.spans) == 1, "the fully-outside span is dropped"
        assert cropped.spans[0].text == "Inside"
        _box_near(cropped.spans[0].bbox, (50.0, 500.0, 64.03, 24.0))

        (hanging,) = page_layout(data=generate_pdf_layout_crop_partial_span())
        assert len(hanging.spans) == 1, "the partial overlap is kept"
        box = hanging.spans[0].bbox
        _box_near(box, (380.0, 500.0, 57.36, 24.0))
        assert box[0] + box[2] > 400.0, "the box extends past the 400pt page, unclipped"

        # the origin law: a run at the MediaBox's own origin → (0, 0)
        (origin,) = page_layout(
            data=generate_pdf_layout(
                b"BT /F1 12 Tf 50 50 Td (Origin) Tj ET", mediabox="[50 50 662 842]"
            )
        )
        box = origin.spans[0].bbox
        assert abs(box[0]) <= _LAYOUT_TOL and abs(box[1]) <= _LAYOUT_TOL, (
            f"the source origin is translated away, never re-added: {box}"
        )

    def test_the_rotated_run_inverse_map(self) -> None:
        """The T3 pins (§9 item 3's inverse-map case): on a /Rotate 90
        page the engine's pre-mapped rotated run comes back into the SAME
        frame as the horizontal runs — the engine-mapped (300, 263.98,
        24, 48.02) round-trips exactly (no crop: the frames coincide),
        with the honest rotation_degrees=90 flag; on a CROPPED rotated
        page the inverse is load-bearing: the engine mapped about the
        MediaBox, tors maps about the visible box, so the rotated run at
        raw (250, 350) — reported by the engine at (350, 313.98, 24,
        48.02) — lands at (250, 201.98, 24, 48.02). The line box is the
        engine's degraded SQUARE (to_chars re-lays the run over the
        mapped box), pinned as measured and carried by rotation_degrees,
        never re-derived."""
        (mixed,) = page_layout(data=generate_pdf_layout_rotated_run())
        hello = next(span for span in mixed.spans if span.text == "Hello")
        side = next(span for span in mixed.spans if span.text == "Side")
        _box_near(hello.bbox, (700.0, 485.33, 24.0, 54.67))
        _box_near(side.bbox, (300.0, 263.98, 24.0, 48.02))
        assert side.rotation_degrees == 90.0 and hello.rotation_degrees == 0.0
        side_line = next(line for line in mixed.lines if line.text == "Side")
        _box_near(side_line.bbox, (300.0, 263.98, 48.02, 48.02))

        # the decisive case: the same content, cropped — the engine's
        # verbatim number would be (350, 313.98, ...); tors's frame
        # correction lands it at (250, 201.98, ...).
        cropped = generate_pdf_layout(
            b"BT /F1 24 Tf 200 400 Td (Hello) Tj ET\n"
            b"BT /F1 24 Tf 0 1 -1 0 250 350 Tm (Side) Tj ET",
            rotate=90,
            cropbox="[100 100 500 700]",
        )
        (mixed,) = page_layout(data=cropped)
        hello = next(span for span in mixed.spans if span.text == "Hello")
        side = next(span for span in mixed.spans if span.text == "Side")
        _box_near(hello.bbox, (300.0, 245.33, 24.0, 54.67))
        _box_near(side.bbox, (250.0, 201.98, 24.0, 48.02))
        assert side.rotation_degrees == 90.0

    def test_images_come_from_handles(self) -> None:
        """The D4c pins (probe P5/P6/P7): the 1×1 image surfaces via the
        HANDLES path (extract_images' silent <8×8 filter would return
        zero); its box is the forward rule over the always-raw handle
        bbox — on a /Rotate 90 page forward-normalized to (200, 312,
        100, 100); a rotated CTM reports rotation_degrees 90; a
        two-image page pins paint_order to content-stream order."""
        (plain,) = page_layout(data=generate_pdf_layout_image())
        assert len(plain.images) == 1
        image = plain.images[0]
        _box_near(image.bbox, (200.0, 200.0, 100.0, 100.0))
        assert (image.width_px, image.height_px) == (1, 1)
        assert (image.rotation_degrees, image.paint_order) == (0.0, 0)

        (turned,) = page_layout(data=generate_pdf_layout_image(rotate=90))
        _box_near(turned.images[0].bbox, (200.0, 312.0, 100.0, 100.0))
        assert turned.images[0].rotation_degrees == 0.0, (
            "the handle's CTM rotation, never the page's"
        )

        (two,) = page_layout(data=generate_pdf_layout_two_images())
        assert [image.paint_order for image in two.images] == [0, 1]
        _box_near(two.images[0].bbox, (100.0, 100.0, 50.0, 50.0))
        _box_near(two.images[1].bbox, (300.0, 400.0, 60.0, 60.0))

    def test_the_join_and_union_laws(self) -> None:
        """The grounding law (§5's join contract), pinned per line: the
        line's text is its words' texts joined with single spaces (the
        engine's own TextLine::new rule — a bare concat is the two-word
        case's PRE-space text and is not the law), and every word's box
        is inside the line's box (the union law)."""
        (page,) = page_layout(data=generate_pdf_layout_baseline())
        for line in page.lines:
            assert " ".join(word.text for word in line.words) == line.text
            for word in line.words:
                # the invariant that keeps the single-space law true across
                # engine bumps: a word's text can never carry whitespace
                # (the line builder drops every whitespace character before
                # forming a word) — so the join separator is always exactly
                # one U+0020 and the offset arithmetic above stays sound
                assert not any(ch.isspace() for ch in word.text), word.text
                assert word.bbox[0] >= line.bbox[0] - _LAYOUT_TOL
                assert word.bbox[1] >= line.bbox[1] - _LAYOUT_TOL
                assert word.bbox[0] + word.bbox[2] <= line.bbox[0] + line.bbox[2] + _LAYOUT_TOL
                assert word.bbox[1] + word.bbox[3] <= line.bbox[1] + line.bbox[3] + _LAYOUT_TOL

    def test_bold_mid_line_spans_split_where_lines_merge(self) -> None:
        """D3's bold-mid-line pin: two runs on one baseline — one bold —
        are TWO spans (the font evidence) in ONE line (lines merge where
        spans split)."""
        (page,) = page_layout(data=generate_pdf_layout_bold_mid_line())
        assert len(page.spans) == 2
        hello = next(span for span in page.spans if span.text == "Hello")
        world = next(span for span in page.spans if span.text == "World")
        assert not hello.is_bold and world.is_bold
        assert len(page.lines) == 1
        assert len(page.lines[0].words) == 2
        assert " ".join(word.text for word in page.lines[0].words) == page.lines[0].text

    def test_declared_artifacts_are_flagged(self) -> None:
        """D6's is_artifact pin: the typed-/Artifact-tagged header run
        reports is_artifact=True; the body run False — declared furniture
        distinguishable from body."""
        (page,) = page_layout(data=generate_pdf_layout_artifact())
        header = next(span for span in page.spans if span.text == "Header")
        body = next(span for span in page.spans if span.text == "Body")
        assert header.is_artifact and not body.is_artifact

    def test_the_bmc_artifact_form_is_not_flagged(self) -> None:
        """T11's limitation pin (deliberately, not as a bug to fix
        silently): a bare ``/Artifact BMC`` — no properties dictionary —
        reports ``is_artifact=False`` on this surface (the engine's BMC
        handler leaves the artifact type unset, and the span carries no
        other artifact bit), exactly like the body run beside it. The
        consumer law that follows: ``is_artifact == false`` is NOT
        evidence of body text. An engine bump that starts classifying BMC
        form fails here and must re-open the spec caveat."""
        (page,) = page_layout(data=generate_pdf_layout_artifact_bmc())
        assert len(page.spans) == 2
        header = next(span for span in page.spans if span.text == "Header")
        body = next(span for span in page.spans if span.text == "Body")
        assert not header.is_artifact, (
            "the BMC form's limitation pin: not flagged on this surface"
        )
        assert not body.is_artifact

    def test_pages_is_pdf_only_refused_on_every_non_pdf_row(self) -> None:
        """F1's law (pages= is PDF-only — a page index cannot name a
        section/slide/worksheet): a non-None selection on every non-PDF
        row raises the conversion lane's own typed refusal
        (byte-identical wording), never the silent whole-document answer
        the phase-2 diff shipped (pages=[0] on a two-section docx
        returned BOTH sections). The capability refusal still wins first;
        on the PDF lane pages=[] keeps its empty-selection error."""
        for kind in ("docx_rich", "pptx_rich", "xlsx", "html_rich"):
            with pytest.raises(ValueError) as raised:
                page_layout(data=ENGINES_CORPUS[kind], pages=[0])
            # the message is the core's own `{:?}` spelling: double quotes
            assert str(raised.value) == (
                f'pages= applies to PDF documents only, not "{kind.split("_")[0]}"'
            ), kind
        # the capability refusal still wins first (anydoc on a docx names
        # the capability, not pages=)
        with pytest.raises(ValueError) as raised:
            page_layout(data=ENGINES_CORPUS["docx_rich"], backend="anydoc", pages=[0])
        assert "cannot serve page geometry" in str(raised.value)
        # and the PDF lane's pages=[] keeps the empty-selection error
        data = generate_pdf_layout_pages(2)
        with pytest.raises(ValueError) as raised:
            page_layout(data=data, pages=[])
        assert str(raised.value) == "pages= selected no pages: pass None for the whole document"

    def test_the_270_inverse_map_and_180_crop_goldens(self) -> None:
        """§9 item 3's missing goldens, Python side: the 270 arm of the
        inverse map (a distinct branch, shipped untested by phase 2,
        probed correct by the red-team pass) — the rotated-content run
        whose raw box is (300, 300, 48.02, 24) lands at (468, 300, 24,
        48.02) on a /Rotate 270 page, (376, 200, 24, 48.02) with the
        [100 100 500 700] crop; and the 180+crop golden — the
        crop-subset fixture's run on a /Rotate 180 page lands at
        (285.97, 76, 64.03, 24) (inverse about the MediaBox, forward
        about the visible box)."""
        side = b"BT /F1 24 Tf 0 1 -1 0 300 300 Tm (Side) Tj ET"
        (plain,) = page_layout(data=generate_pdf_layout(side, rotate=270))
        _box_near(plain.spans[0].bbox, (468.0, 300.0, 24.0, 48.02))
        assert plain.spans[0].rotation_degrees == 90.0
        (cropped,) = page_layout(
            data=generate_pdf_layout(side, rotate=270, cropbox="[100 100 500 700]")
        )
        _box_near(cropped.spans[0].bbox, (376.0, 200.0, 24.0, 48.02))
        (flipped,) = page_layout(
            data=generate_pdf_layout(
                b"BT /F1 24 Tf 150 600 Td (Inside) Tj ET",
                rotate=180,
                cropbox="[100 100 500 700]",
            )
        )
        _box_near(flipped.spans[0].bbox, (285.97, 76.0, 64.03, 24.0))

    def test_image_boxes_are_never_visibility_filtered(self) -> None:
        """T12's no-drop law, Python side: the engine's 2 pt
        visibility-drop law is the TEXT walk's only — an image wholly
        outside the visible box is still RETURNED, with out-of-range
        coordinates (raw (30,30,100,100) on the [100 100 500 700] crop
        → (-70, -70, 100, 100)), while a text run at the same place is
        dropped. The consumer clips to the page box, and never reads
        "image present" as "image visible"."""
        (page,) = page_layout(
            data=generate_pdf_layout(
                b"BT /F1 24 Tf 30 30 Td (Outside) Tj ET",
                cropbox="[100 100 500 700]",
                image_ops=b"100 0 0 100 30 30",
            )
        )
        assert page.spans == [], "a text run at (30, 30) is dropped"
        assert len(page.images) == 1, "the same-position image is returned"
        _box_near(page.images[0].bbox, (-70.0, -70.0, 100.0, 100.0))
        assert page.images[0].bbox[0] < 0 and page.images[0].bbox[1] < 0, (
            "out-of-range coordinates: never clipped, never dropped"
        )

    def test_scanned_and_junk_pages_yield_empty_lists_not_errors(self) -> None:
        """§7's empty-lists-not-errors engine law, Python side: a scanned
        page (image-only) and a junk content stream both answer entries
        with the geometry intact and EMPTY spans/lines/images — never an
        error, never fabricated content."""
        (scan,) = page_layout(data=ENGINES_CORPUS["pdf_scanned"])
        assert scan.geometry is not None and scan.geometry.width_pt == 612.0
        assert scan.spans == [] and scan.lines == []
        junk = generate_pdf_layout(b"this is not a content stream, just junk tokens")
        (page,) = page_layout(data=junk)
        assert page.geometry is not None
        assert page.spans == [] and page.lines == [] and page.images == []

    def test_pages_selection_normalizes_and_carries_page_index(self) -> None:
        """§9 item 13's pages= law: pages=[1, 0] == pages=[0, 1] (the
        output is parallel to the NORMALIZED selection and every entry
        carries its page_index); duplicates normalize away; and the empty
        and out-of-range selections raise the byte-pinned markdown_pages
        error strings."""
        data = generate_pdf_layout_pages(3)
        sorted_pages = page_layout(data=data, pages=[0, 1, 2])
        assert page_layout(data=data, pages=[2, 1, 0]) == sorted_pages
        assert page_layout(data=data, pages=[1, 0]) == page_layout(data=data, pages=[0, 1])
        by_index = {entry.page_index: entry for entry in page_layout(data=data, pages=[2, 0])}
        assert sorted(by_index) == [0, 2]
        assert by_index[2].spans[0].text == by_index[0].spans[0].text
        assert by_index[2].page_index == 2 and by_index[0].page_index == 0
        # the byte-pinned refusals
        with pytest.raises(ValueError) as raised:
            page_layout(data=data, pages=[])
        assert str(raised.value) == "pages= selected no pages: pass None for the whole document"
        with pytest.raises(ValueError) as raised:
            page_layout(data=data, pages=[3])
        assert (
            str(raised.value)
            == "page index 3 is out of range for a 3-page document (pages= is 0-based)"
        )

    def test_auto_equals_oxide_with_plain_equality(self) -> None:
        """§9 item 13: auto == oxide compared with `==` DIRECTLY — the
        native pyclasses' Rust-side structural __eq__, lists elementwise:
        the geometry lane is a property of the format, not a choice, on
        the layout surface too."""
        data = generate_pdf_layout_baseline()
        assert page_layout(data=data) == page_layout(data=data, backend="oxide")
        assert page_layout(data=data, backend="auto") == page_layout(data=data, backend="auto")
        docx = generate_docx_pgsz()
        assert page_layout(data=docx) == page_layout(data=docx, backend="oxide")
        # and the equality is field-structural: a different call's entries
        # are not equal
        assert page_layout(data=data) != page_layout(data=generate_pdf_layout_rotated(90))

    def test_the_office_and_no_geometry_rows(self) -> None:
        """The capability table applies to page_layout identically, with
        layout content PDF-only: the office rows serve their declared
        geometry with EMPTY spans/lines/images; the no-geometry row
        answers exactly one geometry-None entry on every backend."""
        (sized,) = page_layout(data=generate_docx_pgsz())
        assert (sized.geometry.width_pt, sized.geometry.height_pt) == (612.0, 792.0)
        assert sized.geometry.rotation == 0 and sized.geometry.crop is None
        assert sized.spans == [] and sized.lines == [] and sized.images == []
        (flow,) = page_layout(data=ENGINES_CORPUS["html_rich"])
        assert flow.geometry is None
        assert flow.spans == [] and flow.lines == [] and flow.images == []
        assert page_layout(data=ENGINES_CORPUS["html_rich"], backend="anydoc") == [flow]
        assert page_layout(data=ENGINES_CORPUS["csv_rich"], backend="anydoc") == (
            page_layout(data=ENGINES_CORPUS["csv_rich"])
        )

    @pytest.mark.parametrize("kind", sorted(ENGINES_CORPUS), ids=lambda kind: kind)
    def test_lengths_follow_the_matrix_on_the_layout_surface(
        self, tmp_path: Path, kind: str
    ) -> None:
        """§6's length rule, on page_layout: pdf → the page count (or the
        normalized pages= selection's length); office → the §6 matrix's
        unit counts with EMPTY layout lists; single-flow → exactly one
        entry with geometry None. The shared corpus sweeps it."""
        path = _materialize(tmp_path, kind)
        if kind in _REFUSES_ON_THE_GEOMETRY_LANE:
            with pytest.raises(ValueError, match="(?i)format mismatch"):
                page_layout(path)
            return
        entries = page_layout(path)
        if kind.startswith("pdf"):
            assert len(entries) == pdf_page_count(path), kind
            for entry in entries:
                assert entry.geometry is not None, kind
        else:
            assert len(entries) == _UNIT_COUNTS[kind], kind
        # and pages= parallels the normalized selection (the pdf half)
        if kind.startswith("pdf"):
            subset = page_layout(path, pages=[0])
            assert len(subset) == 1
            assert subset[0].page_index == 0

    def test_the_refusal_order_matches_the_locked_precedence(self, tmp_path: Path) -> None:
        """§6's amended order, on the layout surface: the capability
        refusal (the table's own, byte-identical to page_geometry's)
        precedes the password-kind refusal, which precedes the zero
        max_bytes refusal, which precedes the engine's doors; resolve
        comes first (garbage has no format to name)."""
        docx = generate_docx_pgsz()
        with pytest.raises(ValueError) as raised:
            page_layout(data=docx, backend="anydoc")
        assert (
            str(raised.value)
            == 'backend "anydoc" cannot serve page geometry for docx: anydoc has no '
            "geometry surface; use backend='auto' or 'oxide' here"
        )
        # the capability refusal precedes the password-kind refusal
        with pytest.raises(ValueError) as raised:
            page_layout(data=docx, backend="anydoc", password="torque")
        assert "cannot serve page geometry" in str(raised.value)
        # the password-kind refusal precedes the zero-max_bytes refusal
        with pytest.raises(ValueError) as raised:
            page_layout(data=docx, password="torque", max_bytes=0)
        assert "password= applies to PDF documents only" in str(raised.value)
        # zero max_bytes refused core-side (never pre-empting)
        with pytest.raises(ValueError, match="max_bytes must be positive"):
            page_layout(data=docx, max_bytes=0)
        # resolve/format first: garbage under anydoc has no format to name
        with pytest.raises(ValueError, match="(?i)unrecognized"):
            page_layout(data=b"garbage", backend="anydoc")
        # and the door stays last: an encrypted PDF under a zero budget
        # surfaces the zero refusal
        with pytest.raises(ValueError, match="max_bytes must be positive"):
            page_layout(data=_encrypted_pdf(), max_bytes=0)

    def test_aio_carries_the_layout_lane(self, tmp_path: Path) -> None:
        """The awaitable spellings (both aio modules) wrap the typed
        wrapper: the same entries through the thread hop, the same
        refusals."""
        from tors_documents import aio as payload_aio

        from tors.documents import aio as shim_aio

        path = tmp_path / "layout_aio.pdf"
        path.write_bytes(generate_pdf_layout_baseline())
        expected = page_layout(str(path))
        for aio_module in (payload_aio, shim_aio):
            assert asyncio.run(aio_module.page_layout(str(path))) == expected
            assert asyncio.run(aio_module.page_layout(str(path), backend="oxide")) == expected
            with pytest.raises(ValueError, match="cannot serve the declared page-box geometry"):
                asyncio.run(aio_module.page_layout(str(path), backend="anydoc"))
            with pytest.raises(ValueError, match="ceiling"):
                asyncio.run(aio_module.page_layout(str(path), max_bytes=1))

    def test_concurrent_layout_calls_are_identical(self, tmp_path: Path) -> None:
        """The concurrency identity on the layout surface: 8 threads over
        the same bytes, every answer equal to the single-threaded one."""
        data = generate_pdf_layout_baseline()
        expected = page_layout(data=data)
        results: list[object] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            try:
                barrier.wait()
                results.append(page_layout(data=data))
            except BaseException as exc:  # noqa: BLE001 -- recorded, asserted below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        assert all(result == expected for result in results)

    @pytest.mark.timing
    def test_page_layout_in_a_thread_keeps_the_event_loop_at_heartbeat_granularity(
        self, tmp_path: Path
    ) -> None:
        """§9 item 12's GIL cell, on the surface where it matters most:
        page_layout's post-detach marshalling is O(pages × boxes) NESTED
        Python-object construction (spans, lines, words, images — read
        through every getter, the realistic consumer). The fixture: 500
        pages × 12 lines (the native span walk makes the call's wall
        ~120ms — several heartbeat ticks, the sibling cells' regime, so
        the marshalling's GIL-held fraction is small against it) — and
        the whole read+parse pass must keep the 10ms heartbeat alive in a
        to_thread wrap."""
        from test_gil_release import _assert_loop_stays_responsive

        data = generate_pdf_layout_pages(500, lines=12)
        path = tmp_path / "layout_500x12.pdf"
        path.write_bytes(data)
        entries = page_layout(data=data)
        assert len(entries) == 500
        assert all(len(entry.spans) == 12 for entry in entries)

        def op() -> None:
            for entry in page_layout(str(path)):
                # the nested marshalling: every box-bearing field, every
                # entry — the O(pages × boxes) construction post-detach
                _ = entry.geometry.width_pt
                for span in entry.spans:
                    _ = (span.text, span.bbox, span.font_size, span.is_bold)
                for line in entry.lines:
                    _ = (line.text, line.bbox, [(w.text, w.bbox) for w in line.words])
                for image in entry.images:
                    _ = (image.bbox, image.width_px, image.paint_order)

        asyncio.run(_assert_loop_stays_responsive(lambda: asyncio.to_thread(op)))

    def test_the_docstring_carries_the_space_law_and_the_sandbox_warning(self) -> None:
        """The per-box docstring law (D1's verbatim wording) and the ⚠
        sandbox warning: the native page_layout docstring carries both the
        coordinate-space law and the warning; every box-bearing getter's
        docstring carries the law; the data classes' own docstrings stay
        law-only (the warning is the FUNCTION's per-call contract)."""
        law = (
            "points, bottom-left origin of the page's render space; y is the "
            "box's bottom edge; for text the bottom edge is the glyph "
            "baseline and the box extends one font-size upward (em-box; "
            "descenders may extend below). Boxes are unclipped; the page "
            "box is the consumer's clip."
        )

        def states_law(doc: str | None) -> bool:
            import re

            return re.sub(r"\s+", " ", doc or "").find(re.sub(r"\s+", " ", law)) >= 0

        assert states_law(page_layout.__doc__), (
            "the native page_layout docstring must state the D1 law verbatim"
        )
        assert "sandbox untrusted PDFs" in (page_layout.__doc__ or "")
        # every box-bearing getter states the law (the D1 docstring law)
        for getter in (TextSpan.bbox, TextLine.bbox, Word.bbox, ImageBox.bbox):
            assert states_law(getter.__doc__), getter
        # the data classes' own docstrings carry no warning of their own
        for klass in (PageLayout, TextSpan, TextLine, Word, ImageBox):
            assert "sandbox untrusted PDFs" not in (klass.__doc__ or ""), (
                "the ⚠ warning is the FUNCTION's (per-call), not the data classes'"
            )


class TestLayoutClassContracts:
    """The five layout classes' surface pins (§9 item 13 + D6's field
    bar): each class's getter set is EXACTLY the §5/D6 field list
    (documented in the stub), read-only, and structurally comparable."""

    @pytest.mark.parametrize(
        ("klass", "fields"),
        [
            (PageLayout, ["page_index", "geometry", "spans", "lines", "images"]),
            (TextSpan, [
                "text",
                "bbox",
                "font_size",
                "font_name",
                "is_bold",
                "is_italic",
                "rotation_degrees",
                "is_artifact",
            ]),
            (TextLine, ["text", "bbox", "words"]),
            (Word, ["text", "bbox"]),
            (ImageBox, ["bbox", "width_px", "height_px", "rotation_degrees", "paint_order"]),
        ],
    )
    def test_the_getter_set_is_exactly_the_field_list_and_read_only(
        self, klass: type, fields: list[str]
    ) -> None:
        """The D6 field bar, pinned: the class exposes exactly the §5/D6
        fields (nothing cut earlier, nothing added later without a
        breaking-shape decision), and every field is read-only (a setter
        attempt raises AttributeError)."""
        assert sorted(name for name in vars(klass) if not name.startswith("_")) == sorted(fields), (
            (klass.__name__, sorted(vars(klass)))
        )
        # read-only, on a live instance (the ImageBox from an image-bearing
        # fixture — the baseline page carries none). pyo3's getters are the
        # native type's own descriptors (not `property` objects); the
        # read-only half is the setattr refusal.
        page = page_layout(data=generate_pdf_layout_baseline())[0]
        image_page = page_layout(data=generate_pdf_layout_image())[0]
        samples: dict[type, object] = {
            PageLayout: page,
            TextSpan: page.spans[0],
            TextLine: page.lines[0],
            Word: page.lines[0].words[0],
            ImageBox: image_page.images[0],
        }
        live = samples[klass]
        for field in fields:
            with pytest.raises(AttributeError):
                setattr(live, field, getattr(live, field))

    def test_equality_is_structural_and_repr_round_trips_the_name(self) -> None:
        """The native __eq__ (all fields) and __repr__ (the class name and
        every field): equal entries compare equal, a changed field
        compares unequal, and the repr names the class."""
        page = page_layout(data=generate_pdf_layout_baseline())[0]
        same = page_layout(data=generate_pdf_layout_baseline())[0]
        assert page == same
        assert page.spans[0] == same.spans[0]
        assert page.lines[0] == same.lines[0]
        assert page.lines[0].words[0] == same.lines[0].words[0]
        assert page.geometry == same.geometry
        assert page is not same
        # repr: the class name and no crash
        for value in (page, page.spans[0], page.lines[0], page.lines[0].words[0], page.geometry):
            assert type(value).__name__ in repr(value)


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
    engine amplifies input into resident memory at ~36x on benign
    delimiter shapes and ~146x at the measured worst case (this box: a
    24 MiB many-short-cells csv peaked 3.4 GiB, stable across
    sizes (the old "~36x" figure was a benign long-cell shape), so the
    lane carries a default 32 MiB input ceiling (a crate-side constant,
    pinned in the Rust tests) no 33 MB fixture is built here; at the
    worst-case multiple that default budgets ~4.6 GiB) and max_bytes= is
    the per-call override: a typed refusal naming both sizes, never a
    silent OOM."""

    def test_a_document_over_the_ceiling_is_refused_naming_both_sizes(self) -> None:
        """Over the ceiling is a ValueError naming the engine lane, the
        knob (max_bytes), and both sizes (the document's and the
        ceiling's) at fixture scale (the ~118-byte csv against 64, raw
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
        """The override semantics, pinned without a 33 MB fixture: the
        same 1.2 MB csv sits far under the lane's default ceiling (it
        converts, resolved csv), and max_bytes= pulls the ceiling under it
        (the same call now refuses): a per-call knob, not a global
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
        roomy max_bytes converts: resolved csv, content intact."""
        from tors_documents import Format

        resolved, markdown = to_markdown(data=_CEILING_CSV, max_bytes=1_000_000)
        assert resolved is Format.CSV
        assert "T-101" in markdown and "needs review" in markdown

    def test_an_explicit_max_bytes_binds_every_lane_not_just_the_anydoc_one(
        self, tmp_path: Path
    ) -> None:
        """The repinned contract (the input-side hardening's deliberate
        change; the new-contract pins also live in the hardening suite:
        this is the engines suite's own hold on it). Red, captured verbatim
        on this tree before the repin: this test's old pin (max_bytes=1
        over real PDF bytes converts, "the knob simply does not apply to
        that lane") failed as
        ``ValueError: the document is 859 bytes and the input ceiling is
        1 bytes (an explicit max_bytes is binding on every engine lane;
        pdf and HTML included ...)``. Green now pins the new doctrine: an
        explicit max_bytes refuses the PDF on both source lanes (data= and
        path=, the pre-read/pre-copy check) with the input-ceiling
        ValueError naming both sizes, while max_bytes=None keeps the
        default doctrine exactly: the 32 MiB default is the post-read
        check on the anydoc/oxide lanes only, so the same PDF converts
        unmolested without a budget (the lane is unknowable before the
        container sniff, which is exactly why only the explicit budget
        can be pre-read)."""
        from tors_documents import Format

        pdf = ENGINES_CORPUS["pdf_two_page"]
        path = tmp_path / "two.pdf"
        path.write_bytes(pdf)
        for refused in (
            lambda: to_markdown(data=pdf, max_bytes=1),
            lambda: to_markdown(str(path), max_bytes=1),
        ):
            with pytest.raises(ValueError, match="ceiling") as raised:
                refused()
            message = str(raised.value)
            assert "max_bytes" in message, message
            assert "the document is" in message and "input ceiling is" in message, message
            assert "859 bytes" in message and "1 bytes" in message, message
        resolved, markdown = to_markdown(str(path))
        assert resolved is Format.PDF
        assert "first page line" in markdown and "second page line" in markdown

    def test_max_bytes_must_be_a_positive_int(self) -> None:
        """0 and -5 are value problems ("must be positive"); True, "64KB",
        and 1.5 are type problems naming max_bytes=; and True is checked
        first, as a type (a bool launders through an int extraction as 1,
        the same hazard the pages= lane's bool gate pins), so it gets the
        TypeError, never the value refusal. Both convert modes."""
        for convert in (to_markdown, to_text):
            for bad in (0, -5):
                with pytest.raises(ValueError, match="positive"):
                    convert(data=_CEILING_CSV, max_bytes=bad)
            for bad in (True, "64KB", 1.5):
                with pytest.raises(TypeError, match="max_bytes must be an int"):
                    convert(data=_CEILING_CSV, max_bytes=bad)


# --- the inline-recursion bound (the emphasis-strip depth cap) -------------------
#
# The gfm strip's SIGSEGV fix, pinned at the Python layer: the inline
# machinery (link labels, image labels, emphasis) recurses per nesting
# level, so a sufficiently deep `[…](…)` nest overflowed the stack and
# killed the process: the crash shape a subprocess probe exists to catch
# (the pre-fix behavior must never run in the pytest runner). The bound is
# MAX_INLINE_DEPTH=256, Rust-pinned in src/gfm_strip_impl.rs (the exact
# degradation: past 256 the remaining machinery degrades to literal text
# instead of recursing); this is the loose lane pin: the conversion
# survives the input and answers non-empty, in a child, on a bounded
# budget.

_RECURSION_DEPTH = 30_000  # ~120 KB of paragraph text: the nest, fast to build


class TestInlineRecursionBound:
    """The 30,000-deep `[[[…x…]]()…]()` construct through the oxide text
    lane: red (measured on the pre-fix tree) the process SIGSEGVed; green
    it converts, exit 0, non-empty output. The exact degradation shape is
    the Rust unit pins' to hold (don't double-pin brittle exactness here);
    this pin holds the survival contract at the entry-point layer."""

    def test_a_30k_deep_link_label_nest_converts_in_a_subprocess(self, tmp_path: Path) -> None:
        """A docx (python-docx, the dev group's writer: docgen.py's lane)
        whose one paragraph is the nest, converted by `to_text(
        backend="oxide")` in a subprocess: the pre-fix shape is a SIGSEGV
        (a dead child, nonzero exit), so the probe must never run
        in-process. The child prints a bounded summary (resolved name,
        output length: never the ~120 KB output itself); the parent
        asserts exit 0 and the non-empty answer the child reports."""
        import subprocess
        import sys
        import textwrap

        from docx import Document

        document = Document()
        document.add_paragraph("[" * _RECURSION_DEPTH + "x" + "]()" * _RECURSION_DEPTH)
        path = tmp_path / "deep_nest.docx"
        document.save(path)

        code = textwrap.dedent(
            """
            import sys
            from tors_documents import to_text
            resolved, out = to_text(path=sys.argv[1], backend="oxide")
            assert out, "the conversion answered empty"
            print(f"resolved={resolved.value} out_len={len(out)}")
            """
        )
        done = subprocess.run(
            [sys.executable, "-c", code, str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert done.returncode == 0, (
            "the deep-nest probe child died — the pre-fix shape is a "
            f"SIGSEGV in the strip:\n{done.stdout}\n{done.stderr}"
        )
        assert "resolved=docx out_len=" in done.stdout, done.stdout


# --- zip-bomb refusal: the engine-side decompression caps ------------------------
#
# The regression net the PR description claimed but never had: the caps
# fire. Both fixtures are runtime-generated (never committed binaries;
# the encrypted-PDF lane's precedent) under tmp_path, and both stream
# their part content in bounded chunks so the test never holds the
# inflated part in one bytes object. The refusals are asserted, never RSS
# bounds, so the cells cannot flake on memory accounting.

_MIB = 1024 * 1024


def _zip_bomb_docx(path: Path, inflated_mib: int) -> None:
    """A valid docx container whose one word/document.xml declares and
    inflates to `inflated_mib` MiB: the part is a single compressible text
    run (a real zip bomb's shape: hundreds of MiB inside a few hundred
    KiB of stored zip) streamed in 1 MiB chunks, wrapped in enough XML to
    be the part it claims."""
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType='
        '"application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType='
        '"application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        '.main+xml"/></Types>\n'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type='
        '"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
        ' Target="word/document.xml"/></Relationships>\n'
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        with archive.open("word/document.xml", "w") as part:
            prefix = (
                b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                b"<w:body><w:p><w:r><w:t xml:space=\"preserve\">"
            )
            suffix = b"</w:t></w:r></w:p></w:body></w:document>"
            part.write(prefix)
            written = len(prefix)
            target = inflated_mib * _MIB - len(suffix)
            chunk = b"x" * _MIB
            while written < target:
                part.write(chunk[: min(len(chunk), target - written)])
                written += min(len(chunk), target - written)
            part.write(suffix)


class TestZipBombRefusals:
    """The two decompression caps a zip-bombed OOXML container must hit,
    each pinned as the typed refusal it is (the caps' existence is the
    pin: an RSS bound would flake on memory accounting, so none is
    asserted). Measured on this tree: the anydoc lane refuses
    a 400 MiB part on its declared size at the zip directory, before an
    inflate; the oxide lane refuses a 600 MiB declared part
    pre-decompression in ~0.03s at ~20 MiB RSS (asserted here only as a
    typed refusal plus a sub-second wall: the speed is the
    pre-decompression proof: an inflated refusal would take seconds and
    gigabytes)."""

    def test_the_anydoc_lane_refuses_a_400_mib_part_bomb(self, tmp_path: Path) -> None:
        """anydoc's engine-side package caps (128 MiB per entry, 512 MiB
        total across entries): a ~400 KiB zip whose word/document.xml
        declares ~400 MiB is refused by `to_markdown` (the docx auto lane
        is anydoc) with the typed ValueError naming the cap and the entry:
        not the lane's input-ceiling message (the zip itself is far
        under the 32 MiB default; the engine-side cap is what fires), the
        discriminator this pin exists to hold."""
        path = tmp_path / "bomb_anydoc.docx"
        _zip_bomb_docx(path, inflated_mib=400)
        assert path.stat().st_size < _MIB, "the fixture lost the bomb shape (stored > 1 MiB)"
        with pytest.raises(ValueError, match="resource limit exceeded") as raised:
            to_markdown(str(path))
        message = str(raised.value)
        assert "max_entry_bytes" in message, message
        assert "word/document.xml" in message, message
        assert "input ceiling" not in message, (
            f"the input ceiling fired where the engine-side cap should have:\n{message}"
        )

    def test_the_oxide_lane_refuses_a_600_mib_declared_part_pre_decompression(
        self, tmp_path: Path
    ) -> None:
        """office_oxide 0.1.10's per-part cap (512 MiB, declared and
        actual): the refusal fires on the declared size in the zip
        directory, before a single byte is inflated: pinned as the typed
        refusal naming the part and the cap's own number, plus a
        sub-second wall (measured 0.03s on this box; inflating 600 MiB
        and parsing it would cost seconds and ~1.6 GiB, so the wall band
        is the pre-decompression proof that cannot flake)."""
        import time

        path = tmp_path / "bomb_oxide.docx"
        _zip_bomb_docx(path, inflated_mib=600)
        assert path.stat().st_size < _MIB, "the fixture lost the bomb shape (stored > 1 MiB)"
        started = time.perf_counter()
        with pytest.raises(ValueError, match="decompression limit exceeded") as raised:
            to_markdown(str(path), backend="oxide")
        elapsed = time.perf_counter() - started
        message = str(raised.value)
        assert "word/document.xml" in message, message
        assert "536870912" in message, message  # 512 MiB: the cap's own number, verbatim
        assert elapsed < 2.0, (
            f"the refusal took {elapsed:.2f}s — the pre-decompression guard "
            "regressed (measured 0.03s; an inflated refusal costs seconds)"
        )


# --- JSON-lines is not csv (the sniffer's record-shape guard) ---------------------


class TestJsonLinesIsNotCsv:
    """The delimiter witness alone would claim JSON-lines records as csv
    (their comma counts agree line over line), so the sniffer carries a
    record-shape guard: a line opening with { or [ is a record, not a
    field row. The guard is the sniffer's honesty (sniff never names a
    format the bytes are not) and the convert lane leans on it: unnamed
    JSON-lines is a refusal naming format=, and the explicit name is the
    caller's escape hatch."""

    def test_record_lines_sniff_to_none_while_real_csv_sniffs_csv(self) -> None:
        """{ and [ records sniff to None even though the delimiter witness
        alone would claim them (the comma counts agree); the contrast row
        (real delimiter-separated bytes) still sniffs csv, so the None
        answers are the record-shape guard at work, not sniff refusing
        csv ever."""
        from tors_documents import Format

        assert sniff(b'{"a":1,"b":2}\n{"a":3,"b":4}\n') is None
        assert sniff(b"[1,2]\n[3,4]\n") is None
        assert sniff(b"unit,status\nT-101,healthy\n") is Format.CSV

    def test_unnamed_jsonl_is_refused_naming_format_and_csv_is_the_escape_hatch(self) -> None:
        """The convert lane inherits the guard: JSON-lines bytes with no
        name are a ValueError naming format= (the no-name fix: the call
        must not guess csv from the same witness the guard overruled), in
        both record shapes and both modes; with format="csv" the caller
        owns the decision and it converts: resolved csv is pinned, the
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
    property: the base wheel (tors) carries only the lazy re-export shim,
    so importing it never loads the engines (tors.documents /
    tors_documents: the anydoc + pdf_oxide stack); they load on first
    touch of the documents attribute. Measured in a fresh interpreter
    (which modules the import touched) and deliberately not in megabytes
    (load-size numbers vary across platforms and flake; the module-touched
    assert is the design invariant)."""

    def test_importing_the_base_wheel_never_touches_the_engines(self) -> None:
        """A fresh interpreter (this venv's own python) imports tors and
        must not carry tors.documents or tors_documents in sys.modules. If
        this regresses (an eager import creeps into the base package)
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
        """The ergonomic half of the split-wheel design (what docs/documents.md
        examples' ``tors.documents`` spelling rides on): the base package
        carries a PEP 562 ``__getattr__``, so ``tors.documents`` on an
        imported-but-untouched base package imports the shim (and through
        it the payload) on first touch, and the returned object re-exports
        the payload's surface by identity
        (``tors.documents.to_markdown is tors_documents.to_markdown``).
        The laziness gate still holds before the access (both engine
        modules untouched by the plain import: the same fresh-interpreter
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
            tors.definitely_not_an_attribute  # noqa: B018 -- the access is the point

    def test_the_sync_shim_re_exports_the_whole_payload_surface(self, tmp_path: Path) -> None:
        """The base wheel's tors.documents shim is the payload's public
        surface, never a subset or a re-wrap: ``__all__`` equal name for
        name (``__version__`` included), every name the very same object
        tors_documents exports (an identity re-export: a divergence would
        mean a second implementation hiding behind the door), and one full
        conversion through the shim path end to end, the spelling the
        docs examples use, exercised the way callers spell it."""
        import tors_documents as payload

        import tors.documents as shim

        assert shim.__all__ == payload.__all__
        for name in shim.__all__:
            assert getattr(shim, name) is getattr(payload, name), name
        resolved, markdown = shim.to_markdown(_materialize(tmp_path, "pdf_two_page"))
        assert resolved is payload.Format.PDF
        assert "first page line" in markdown and "second page line" in markdown
