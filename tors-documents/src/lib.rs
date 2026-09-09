//! The `tors.documents` payload bindings: the pyo3 surface for the
//! document-format extraction cores in `tors-core` (`pdf_impl`,
//! `documents_impl`, `gfm_strip_impl`), enabled by that crate's `documents`
//! feature. Shipped as the `tors-documents` PyPI wheel, re-exported by the
//! base wheel's `tors.documents` shim — this is the only place these
//! functions exist, so the base `tors` build carries none of the engine
//! weight.
//!
//! The surface, one call per question a document pipeline asks:
//!
//! - [`to_markdown`]/[`to_text`] — any working format to GFM markdown /
//!   plain text (path in, `(format, output)` out, engine routed per the
//!   measured table in `tors-core`'s `documents_impl`, `pages=` for PDF
//!   subsets)
//! - [`sniff`] — the standalone content-marker format detector over bytes
//! - [`pdf_extract`] — the per-page text probe + whole-document markdown in
//!   one pass over one open PDF
//! - [`pdf_page_count`] — the page tree and nothing else
//! - [`pdf_classify`] — the cheap text-vs-image preflight (which pages are
//!   image-only)
//!
//! GIL model (the crate-wide discipline, applied per function): the
//! argument borrow of the path string plus argument validation happen
//! under the GIL, the WHOLE native pass — file read, format sniff, engine
//! conversion, and for `to_text` the markdown strip — runs under one
//! `py.detach`, and the exception mapping and return marshalling happen
//! after the GIL is reacquired. Nothing raises from inside the detached
//! region. The residue classes are `pdf_extract`'s (documented on it) for
//! every function here: the O(output) string marshalling, plus
//! `to_markdown`/`to_text`'s two-string tuple.

use std::path::PathBuf;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyInt, PyList, PyString, PyTuple};
use tors::documents_impl::{self, Backend, Converted, DocumentError};

pyo3::create_exception!(
    _tors_documents,
    NeedsOcrError,
    PyValueError,
    "A PDF's scanned/image-only pages, which no local engine can read: route the document to an OCR stage. Carries .pages (the 0-based page indices — the same convention as pages= and pages_needing_ocr) and .page_count, set on every instance the engine raises (a plain user construction of the class carries neither)."
);

/// `tors.documents.pdf_classify(path) -> PdfClassification` (or
/// `pdf_classify(data=...)`, `password=` for an encrypted one): the cheap
/// text-vs-image preflight over a PDF
/// — no content conversion, no OCR, no rasterization. The answer to "does
/// this PDF have a text layer, or is it an image we can do nothing with
/// locally":
///
/// - `.page_count` — the page tree.
/// - `.page_kinds` — every page's verdict, page order: `"text"` (native
///   text layer), `"scanned"` (image-dominated — OCR the page),
///   `"image_text"` (hybrid), `"mixed"`, or `"empty"` (blank/near-empty —
///   neither extract nor OCR, not an error, and DISTINCT from a scan).
/// - `.pages_needing_ocr` — the 0-based indices of the image-only pages
///   (the `"scanned"` ones; `"empty"` pages are deliberately NOT listed —
///   a blank page is not an image to recover).
/// - `.has_text` — at least one page carries a text layer (`"text"`,
///   `"image_text"`, or `"mixed"`); extraction will yield something.
/// - `.image_only` — every page is image-only; nothing local can read
///   this document, route it to an OCR stage.
///
/// Encrypted documents fail closed (`ValueError`, pdf_oxide's security
/// rule: a security state is never masked as "all pages empty") —
/// `password=` unlocks one (a wrong password is its own clean
/// `ValueError`, and the unlock applies to every PDF entry, not just
/// this one).
///
/// GIL model: the source marshalling under the GIL (`parse_source`: the
/// path type-checked and Unicode-validated, a `data=` call paying its one
/// bytes copy, every refusal naming its argument), the read+classify under
/// `py.detach`, then the return object construction (three attribute
/// values, O(pages)) — `pdf_extract`'s residue class with the smallest
/// native pass of the PDF family.
#[pyfunction(signature = (path = None, data = None, password = None))]
pub fn pdf_classify(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
) -> PyResult<PdfClassification> {
    let source = parse_source(py, path, data, "pdf_classify")?;
    let password = parse_password(password)?;
    let classification = py
        .detach(move || {
            let bytes = source.into_bytes().map_err(tors::pdf_impl::PdfError::Io)?;
            tors::pdf_impl::classify(bytes, password.as_deref())
        })
        .map_err(pdf_error)?;
    Ok(PdfClassification {
        page_count: classification.page_count,
        page_kinds: classification.page_kinds,
        pages_needing_ocr: classification.pages_needing_ocr,
    })
}

/// The `pdf_classify` result: the text-vs-image preflight's answer, with
/// the routing rules derived for the caller. Constructed only by
/// `pdf_classify`; the attribute reads are plain `#[getter]`s. The kinds
/// ride as the core's `PageKind` — the rules-once doctrine across the
/// seam: `has_text` and the scanned match call the core's own rule, never
/// a re-derived string match here — stringified only at the `page_kinds`
/// getter.
#[pyclass]
pub struct PdfClassification {
    page_count: usize,
    page_kinds: Vec<tors::pdf_impl::PageKind>,
    pages_needing_ocr: Vec<usize>,
}

#[pymethods]
impl PdfClassification {
    /// Pages in the document.
    #[getter]
    fn page_count(&self) -> usize {
        self.page_count
    }

    /// Every page's verdict, page order: `"text"`, `"scanned"`,
    /// `"image_text"`, `"mixed"`, or `"empty"`.
    #[getter]
    fn page_kinds(&self) -> Vec<&'static str> {
        self.page_kinds.iter().map(|kind| kind.name()).collect()
    }

    /// The 0-based indices of image-only pages: empty for a born-digital
    /// document, every page for a scan, the difference for a mixed one.
    /// Blank pages are not listed (they are `"empty"`, not `"scanned"`).
    #[getter]
    fn pages_needing_ocr(&self) -> Vec<usize> {
        self.pages_needing_ocr.clone()
    }

    /// At least one page carries a text layer — extraction will yield
    /// something. `False` for a zero-page, all-blank, or fully-scanned
    /// document.
    #[getter]
    fn has_text(&self) -> bool {
        self.page_kinds.iter().any(|kind| kind.has_text())
    }

    /// Every page is image-only — nothing local can read this document;
    /// route it to an OCR stage. `False` for a zero-page document (that is
    /// a different failure, already surfaced at open).
    #[getter]
    fn image_only(&self) -> bool {
        self.page_count > 0
            && self
                .page_kinds
                .iter()
                .all(|kind| *kind == tors::pdf_impl::PageKind::Scanned)
    }
}

/// `tors.documents.pdf_extract(path) -> (pages, markdown)` (or
/// `pdf_extract(data=...)`, `password=` for an encrypted one): read a PDF
/// — from `path`, or from the caller's `data=` bytes — and return
/// `(per_page_plain_text, whole_document_markdown)` — one native pass over
/// one open document (the parse is paid once for both outputs).
///
/// `pages` is a `list[str]`, one entry per page in page order, from
/// pdf_oxide's per-page plain-text surface (`extract_text`): the
/// text-layer-probe view. An image-only/scanned page is an empty string,
/// not an error, and a zero-page or textless document yields empty output —
/// routing decisions ("this PDF needs OCR") are the CALLER's, made on these
/// values, never silently made here. `markdown` is pdf_oxide's
/// `to_markdown_all` under default conversion options: heading detection
/// on, images off (no base64 bloat), Tagged-PDF structure-tree reading
/// order falling back to XY-Cut on untagged documents, and `/Link`
/// annotations rendered as `[text](uri)`.
///
/// Always the pdf_oxide engine: this is the probe-rich call, and only
/// pdf_oxide's per-page surface exists. The generic entry points
/// (`to_markdown`/`to_text`) take the backend choice.
///
/// Errors: a missing/unreadable file raises `OSError`; anything that fails
/// to parse as a PDF raises `ValueError` with pdf_oxide's reason. Both are
/// constructed AFTER the GIL is reacquired.
///
/// GIL model: the whole pass — file read, PDF parse, per-page text
/// extraction, markdown conversion — runs under one `py.detach`. The
/// GIL-held residue is the path str marshalling (`parse_path`, a handful
/// of bytes), plus the return marshalling: one `str` per page and the
/// markdown string, O(output). The official pdf_oxide pyo3 wheel measures
/// as GIL-held per call instead (worst heartbeat gap 23.6ms on a 9-page
/// document under a 10ms ping, 2026-09) — the hazard this binding exists
/// to remove; the band is pinned by the shared documents suite.
#[pyfunction(signature = (path = None, data = None, password = None))]
pub fn pdf_extract(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Vec<String>, String)> {
    let source = parse_source(py, path, data, "pdf_extract")?;
    let password = parse_password(password)?;
    let extracted = py
        .detach(move || {
            let bytes = source.into_bytes().map_err(tors::pdf_impl::PdfError::Io)?;
            tors::pdf_impl::extract(bytes, password.as_deref())
        })
        .map_err(pdf_error)?;
    Ok((extracted.pages, extracted.markdown))
}

/// `tors.documents.pdf_page_count(path) -> int` (or
/// `pdf_page_count(data=...)`, `password=` for an encrypted one): read a
/// PDF and return its page count —
/// the page tree and nothing else, no content extraction. For gating
/// expensive downstream work (an OCR/conversion pass that scales with page
/// count) without paying for any of it.
///
/// Errors: `OSError` for a missing/unreadable file, `ValueError` for bytes
/// that do not parse as a PDF, both raised after the GIL is reacquired.
///
/// GIL model: `pdf_extract`'s shape with the smallest possible return —
/// the path str marshalling (`parse_path`), the open+page-tree walk under
/// `py.detach`, and a single `int` back, no marshalling class at all.
#[pyfunction(signature = (path = None, data = None, password = None))]
pub fn pdf_page_count(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
) -> PyResult<usize> {
    let source = parse_source(py, path, data, "pdf_page_count")?;
    let password = parse_password(password)?;
    py.detach(move || {
        let bytes = source.into_bytes().map_err(tors::pdf_impl::PdfError::Io)?;
        tors::pdf_impl::page_count(bytes, password.as_deref())
    })
    .map_err(pdf_error)
}

/// `tors.documents.pdf_link_uris(path) -> list[list[str]]` (or
/// `pdf_link_uris(data=...)`, `password=` for an encrypted one): the
/// `/Annots` link walk — for every page,
/// the URIs of its `/Subtype /Link` annotations whose action is a URI
/// (`/A << /S /URI /URI (...) >>`), in annotation order: one `list[str]`
/// per page, page order, empty lists for pages without link annotations.
///
/// Why this exists alongside the markdown's inline `[text](uri)` links:
/// the two answer different questions. The markdown carries links whose
/// VISIBLE TEXT belongs in prose; this walk carries the raw URI list —
/// including URIs behind link rectangles whose text is not itself a link
/// (a "click here" button, an image, a bare rectangle) which no text
/// rendering surfaces at all. The caller that motivated it measured the
/// difference on real resumes: 60 documents, 16 links from the text layer,
/// 34 from the annotations — a quarter of candidates gained a
/// LinkedIn/GitHub URL no text shape would show. Deduplication and
/// canonicalization are deliberately NOT done here: the verbatim
/// engine-read list is the honest output, and callers canonicalize
/// differently (per-page review panels vs whole-document projections).
///
/// Scope, honestly narrow: URI actions only — `GoTo` (in-document
/// navigation) and `GoToR` (a remote FILE) contribute nothing and are not
/// fabricated into `file://` shapes. Non-link annotations are skipped by
/// the subtype filter; malformed annotation dictionaries are skipped by
/// pdf_oxide (its parse tolerates them) rather than poisoning the page.
///
/// Always the pdf_oxide engine (the annotation walk is its reader), the
/// same lane `pdf_extract` runs.
///
/// Errors: `OSError` (matched subclass) for a missing/unreadable `path=`;
/// `ValueError` for bytes that do not parse as a PDF. Same GIL model as
/// `pdf_classify` — the source marshalling under the GIL, the
/// read+page-tree+annotation walk under one `py.detach`, the nested list
/// marshalling (O(annotations)) after the reacquire.
#[pyfunction(signature = (path = None, data = None, password = None))]
pub fn pdf_link_uris(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<Vec<String>>> {
    let source = parse_source(py, path, data, "pdf_link_uris")?;
    let password = parse_password(password)?;
    py.detach(move || {
        let bytes = source.into_bytes().map_err(tors::pdf_impl::PdfError::Io)?;
        tors::pdf_impl::link_uris(bytes, password.as_deref())
    })
    .map_err(pdf_error)
}

/// `tors.documents.to_markdown(path, data=None, format=None,
/// backend="auto", pages=None, password=None, max_bytes=None)
/// -> (format, markdown)` — or `to_markdown(data=...)`: convert
/// any working-format document — pdf, doc/docx, xls/xlsx, ppt/pptx, rtf,
/// odt/ods/odp, epub, csv, html — to GitHub-Flavored Markdown, on the
/// measured-best engine for its format. Exactly one of `path` (a file, the
/// only positional) or `data` (the document's bytes — the in-memory
/// caller's entry, no temp-file roundtrip; a `data=` call has no name to
/// consult, so format resolution rests on `format=` and the content
/// markers alone, `sniff`'s doctrine) — passing both or neither raises
/// before any work runs.
///
/// `format` names the format explicitly (extension spelling, no dot,
/// case-insensitive: `"pdf"`, `"html"`, `"docx"`, `"xlsx"`, `"pptx"`,
/// `"doc"`, `"xls"`, `"ppt"`, `"rtf"`, `"odt"`, `"ods"`, `"odp"`,
/// `"epub"`, `"csv"`, plus container variants like `"docm"`/`"xlsm"`).
/// `None` (the default) sniffs it from the file's content markers — the
/// PDF header, the RTF open group, OLE stream names, the ZIP package
/// mimetype, the HTML document marker — with the path's extension as the
/// fallback signature-less formats (CSV) need, so a mislabeled file still
/// converts correctly. The returned `format` is the name the conversion
/// actually used.
///
/// `password` unlocks an encrypted PDF (the PDF kinds only — a password
/// on any other format is a `ValueError`, never a silently-ignored
/// argument). Without one, every PDF entry FAILS CLOSED: the door check
/// at open raises, never empty output masquerading as "no content" (the
/// pre-fix shape, measured 2026-09 on an RC4-128 fixture: the markdown
/// and text surfaces returned `""` on a locked document while classify
/// raised). A wrong password is its own clean `ValueError`.
///
/// `max_bytes` overrides the engine-lane input ceiling (default 32 MiB,
/// covering the anydoc AND office_oxide lanes — anydoc measures ~36x RSS
/// amplification on delimiter formats, 2026-09: a 100 MiB csv peaked at
/// 3.7 GiB; a zip-bombed office_oxide container measured far worse, and
/// that engine has no decompression caps of its own, so the input ceiling
/// is the only guard on the opt-in `backend="oxide"` lane. The pdf_oxide
/// and HTML lanes are unmetered). A positive `int`.
///
/// `backend` picks the engine where they overlap: `"auto"` (the default;
/// the native layer takes `None` as the same choice) routes by the
/// measured table in `tors-core`'s `documents_impl` — PDF to pdf_oxide (two-column reading order, link annotations, line structure),
/// HTML to html-to-markdown-rs (script/style dropped by construction), the
/// office and text formats to anydoc (style-based docx headings, list
/// markers, rtf/odt/epub/csv coverage); `"oxide"` forces the oxide family
/// (pdf_oxide for PDF, office_oxide for docx/xlsx/pptx and legacy
/// doc/xls/ppt) and `"anydoc"` forces anydoc — both raise `ValueError` on
/// a format the forced engine cannot read, never a silent fallback.
///
/// `pages` selects a PDF page subset (PDF on the pdf_oxide lane only, else
/// `ValueError`): a single `int` (one 0-based page), a `list` of ints (the
/// explicit set), or a 2-tuple `(start, stop)` (a half-open range, Python
/// convention: `(1, 3)` is the second and third pages — indices are
/// 0-based). The selection is deduped into
/// document order and the per-page conversions are joined with pdf_oxide's
/// own inter-page separator, so a full range is byte-identical to the
/// whole-document conversion.
///
/// Errors: argument refusals raise under the GIL, before any work runs,
/// each naming the argument and repr'ing the offending value —
/// `TypeError` for a wrong-typed `format=`/`backend=`/`path`/`pages=`
/// (a non-string format or backend; a bool, float, or str where a page
/// index belongs), `ValueError` for a wrong value or shape (an unknown
/// or empty `backend=` name; a negative, empty, or backwards `pages=`
/// range; a non-Unicode path). The document failures are constructed
/// after the GIL is reacquired: `OSError` for a missing/unreadable file
/// (the matched subclass — `FileNotFoundError`, `IsADirectoryError`);
/// `NeedsOcrError` (a `ValueError` subclass carrying `.pages` and
/// `.page_count`) when the anydoc backend hits a PDF with scanned pages —
/// route the document to OCR; `ValueError` for an unknown format name,
/// an undetectable file, an unusable backend/format pair, an
/// out-of-bounds `pages=` selection (the core's message names the page
/// count), or a malformed/encrypted/over-limit document.
///
/// GIL model: the whole argument marshalling (path, format, backend,
/// pages — validation and normalization) under the GIL, the whole
/// read+sniff+convert pass under one `py.detach`, then the two-string
/// tuple marshalling, O(output).
// pyo3's #[pyfunction] needs the Python signature FLAT — one Rust
// parameter per Python argument, no struct grouping possible — so the
// Rust-side argument count is the Python surface's count by construction
// (7 here: a caller-visible API shape, not a design smell).
#[allow(clippy::too_many_arguments)]
#[pyfunction(signature = (
    path = None,
    data = None,
    format = None,
    backend = None,
    pages = None,
    password = None,
    max_bytes = None,
))]
pub fn to_markdown(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    format: Option<&Bound<'_, PyAny>>,
    backend: Option<&Bound<'_, PyAny>>,
    pages: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
    max_bytes: Option<&Bound<'_, PyAny>>,
) -> PyResult<(&'static str, String)> {
    convert(
        py,
        path,
        data,
        format,
        backend,
        pages,
        password,
        max_bytes,
        "to_markdown",
        documents_impl::to_markdown_with,
    )
}

/// `tors.documents.to_text(path, data=None, format=None, backend="auto",
/// pages=None, password=None, max_bytes=None)
/// -> (format, text)` — or `to_text(data=...)`: the same conversion,
/// routing, sniffing, source, `pages=`, `password=`, and `max_bytes=`
/// semantics as `to_markdown`,
/// with the markdown normalized to
/// plain text — one text shape for every format and engine (headings keep
/// their text, list items keep indentation and numbering, table rows keep
/// cell boundaries as `" | "` joins, code blocks keep their content without
/// fences, links become `label (url)`), instead of each engine's own
/// plain-text surface (pdf_oxide's, measured, merges two-column layouts
/// line-by-line; the normalization keeps the markdown converter's
/// reading-order blocks).
///
/// Same arguments, same errors, same GIL model as `to_markdown` (the strip
/// runs inside the same detached pass).
// to_markdown's flat-signature rationale, verbatim.
#[allow(clippy::too_many_arguments)]
#[pyfunction(signature = (
    path = None,
    data = None,
    format = None,
    backend = None,
    pages = None,
    password = None,
    max_bytes = None,
))]
pub fn to_text(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    format: Option<&Bound<'_, PyAny>>,
    backend: Option<&Bound<'_, PyAny>>,
    pages: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
    max_bytes: Option<&Bound<'_, PyAny>>,
) -> PyResult<(&'static str, String)> {
    convert(
        py,
        path,
        data,
        format,
        backend,
        pages,
        password,
        max_bytes,
        "to_text",
        documents_impl::to_text_with,
    )
}

/// The core conversion entry points `convert` routes between — the exact
/// signature `tors-core`'s `to_markdown` and `to_text` share (named for
/// the clippy `type_complexity` lint; one alias, both functions). The core
/// is bytes-in: the name hint is the `path=` the payload read, the `data=`
/// entry passing none (see [`Source`]).
type CoreConvert = for<'a> fn(
    Vec<u8>,
    Option<&str>,
    Option<&str>,
    Backend,
    Option<Vec<usize>>,
    documents_impl::ConvertOptions<'a>,
) -> Result<Converted, DocumentError>;

/// The marshalling spine `to_markdown` and `to_text` share — they differ
/// in nothing but which core function the detached pass calls (`to_text`'s
/// strip runs inside the core, so the binding sees one shape for both):
/// marshal and validate every argument under the GIL (`parse_path`,
/// `parse_format`, `parse_backend`, `parse_pages`, every refusal naming
/// its argument and repr'ing the value), run the WHOLE native pass — file
/// read, format sniff, engine conversion — under one `py.detach`, then
/// the error mapping and the two-string return after the GIL is
/// reacquired. `what` is the CALLER's own name (`"to_markdown"`/
/// `"to_text"`), riding into `parse_source`'s both/neither refusals so
/// each function's messages are attributed to itself — the shared spine
/// must not hardcode one twin's name into the other's errors. The
/// contract docstrings stay per-function: they are the surface callers
/// read. (The argument count mirrors the two pyfunction signatures it
/// marshals — see to_markdown's allow rationale.)
#[allow(clippy::too_many_arguments)]
fn convert(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    format: Option<&Bound<'_, PyAny>>,
    backend: Option<&Bound<'_, PyAny>>,
    pages: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
    max_bytes: Option<&Bound<'_, PyAny>>,
    what: &str,
    core: CoreConvert,
) -> PyResult<(&'static str, String)> {
    let source = parse_source(py, path, data, what)?;
    let format = parse_format(format)?;
    let backend = parse_backend(backend)?;
    let pages = parse_pages(pages)?;
    let password = parse_password(password)?;
    let max_bytes = parse_max_bytes(max_bytes)?;
    let converted = match py.detach(move || {
        let (bytes, hint) = source.into_input().map_err(DocumentError::Io)?;
        core(
            bytes,
            hint.as_deref(),
            format,
            backend,
            pages,
            documents_impl::ConvertOptions {
                password: password.as_deref(),
                max_bytes,
            },
        )
    }) {
        Ok(converted) => converted,
        Err(err) => return Err(document_error(py, err)?),
    };
    // Converted.format is &'static (the vocabulary's own name spelling),
    // so the resolved-format half of the pair marshals straight to a
    // Python str with no intermediate Rust String.
    Ok((converted.format, converted.output))
}

/// `tors.documents.sniff(data) -> str | None`: the standalone
/// content-marker format detector — what `to_markdown`/`to_text` would
/// resolve these BYTES to from content alone, with no path and no
/// extension: the PDF header, the RTF open group, OLE stream names, the
/// ZIP package mimetype, the HTML document marker. The routing caller's
/// mislabeled-download answer: the bytes' verdict overrides any label the
/// download carried.
///
/// `None` is not an error: the content names no format (a signature-less
/// text format such as CSV — name it via `format=` — or not a document at
/// all). `sniff` never opens a parser: the markers are container-level.
///
/// `data=` is strict `bytes` — anything else is a `TypeError` naming the
/// argument, the house convention (the raw pyo3 cast error the manual
/// marshal replaces, `'bytearray' object is not an instance of 'bytes'`,
/// probed on the 0.5.0 wheel, named neither); `bytearray`/`memoryview`
/// callers pass `bytes(data)`, the same strict contract as the convert
/// lane's `data=`.
///
/// GIL model: the bytes-in zero-copy borrow under the GIL, the marker scan
/// under `py.detach`, and the `str | None` return (the name is `&'static`
/// — the vocabulary's own spelling — so no owned String is built just to
/// be copied into the Python str) — the bytes-in family's
/// `utf8_is_valid` residue class, no marshalling beyond the answer itself.
#[pyfunction]
pub fn sniff(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Option<&'static str>> {
    // The manual marshal keeps the refusal in the house message style; the
    // zero-copy borrow of the immutable buffer then crosses the detach for
    // the scan (sound by construction — nothing can mutate it GIL-free).
    let bytes = data.cast::<PyBytes>().map_err(|_| {
        PyTypeError::new_err(format!(
            "data must be bytes (the content to sniff), not {}",
            type_name(data)
        ))
    })?;
    let bytes = bytes.as_bytes();
    Ok(py.detach(|| documents_impl::sniff(bytes)))
}

/// Marshal the `path` argument under the GIL: a `str` (the typed wrappers
/// have already `os.fspath`'d PathLike inputs) whose content must be valid
/// Unicode. Both refusal shapes name `path` and repr the value — the raw
/// pyo3 `&str`-conversion failures they replace do not: a non-str path
/// surfaces as a bare `'int' object is not an instance of 'str'` that
/// leaves a four-argument call site guessing which argument failed, and a
/// lone-surrogate path (a filename that escaped a POSIX-only tool) as a
/// `UnicodeEncodeError` about 'utf-8' that reads like a conversion bug in
/// the document engine, not a caller-input problem (both probed on the
/// 0.5.0 wheel, 2026-09).
/// The document's input: a `path=` the detached pass reads, or the
/// caller's `data=` bytes (copied once under the GIL — a borrow cannot
/// cross `py.detach`, and a memcpy of even a 25MB upload is single-digit
/// milliseconds against the conversion pass it precedes). `into_bytes`
/// also yields the NAME hint the extension fallback of format resolution
/// consults: the path, when there was one — a `data=` call has no name and
/// rests on `format=`/content markers alone, the doctrine `sniff`
/// encodes (the core's `resolve` says the same, in its own words).
enum Source {
    Path(PathBuf),
    Data(Vec<u8>),
}

impl Source {
    /// The bytes alone — the PDF family's entry (no name to consult:
    /// those functions are PDF-only by construction).
    fn into_bytes(self) -> Result<Vec<u8>, std::io::Error> {
        Ok(self.into_input()?.0)
    }

    /// The bytes AND the name hint — the convert spine's entry (the
    /// extension fallback's last resort; `None` for a `data=` call).
    fn into_input(self) -> Result<(Vec<u8>, Option<String>), std::io::Error> {
        match self {
            Source::Path(path) => {
                // parse_path already validated the Unicode, so the hint is
                // always representable; `.to_str()` cannot fail here.
                let hint = path.to_str().map(str::to_string);
                Ok((std::fs::read(&path)?, hint))
            }
            Source::Data(bytes) => Ok((bytes, None)),
        }
    }
}

/// The `path`/`data` pair to one [`Source`], under the GIL: exactly one of
/// the two — both is a `ValueError`, neither a `TypeError` (the
/// missing-required-argument convention; the function name rides in
/// `what` so the message reads like Python's own). A `data=` that is not
/// `bytes` is a `TypeError` naming the argument (strict `bytes`, matching
/// the typed surface's signature — `bytearray`/`memoryview` callers pass
/// `bytes(data)`, the copy they are paying anyway).
fn parse_source(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    what: &str,
) -> PyResult<Source> {
    match (path, data) {
        (Some(_), Some(_)) => Err(PyValueError::new_err(format!(
            "{what}: pass either path or data, not both"
        ))),
        (None, None) => Err(PyTypeError::new_err(format!(
            "{what}: missing required argument — pass path (a file path) or data (the document bytes)"
        ))),
        (Some(path), None) => Ok(Source::Path(parse_path(path)?)),
        (None, Some(data)) => {
            let bytes = data.cast::<PyBytes>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "data must be bytes (the document's content), not {}",
                    py_repr(data)
                ))
            })?;
            let _ = py;
            Ok(Source::Data(bytes.as_bytes().to_vec()))
        }
    }
}

fn parse_path(path: &Bound<'_, PyAny>) -> PyResult<PathBuf> {
    let string = path.cast::<PyString>().map_err(|_| {
        PyTypeError::new_err(format!(
            "path must be a str or os.PathLike, not {}",
            py_repr(path)
        ))
    })?;
    match string.to_str() {
        Ok(text) => Ok(PathBuf::from(text)),
        Err(_) => Err(PyValueError::new_err(format!(
            "path must be valid Unicode (no lone surrogates), not {}",
            py_repr(path)
        ))),
    }
}

/// Marshal the `format=` argument under the GIL: `None` (sniff from
/// content, the default) or a format-name `str`, borrowed straight from
/// the argument. A non-str is a `TypeError` naming the argument — the raw
/// pyo3 conversion failure (`'int' object is not an instance of 'str'`)
/// names neither. The NAME itself is the core's vocabulary to referee: an
/// empty or unknown name passes through and fails in the detached pass as
/// `ValueError: unknown format name ...` (the core's message, measured
/// clean), so this layer does not duplicate the vocabulary.
fn parse_format<'a>(format: Option<&'a Bound<'_, PyAny>>) -> PyResult<Option<&'a str>> {
    let Some(format) = format else {
        return Ok(None);
    };
    let string = format.cast::<PyString>().map_err(|_| {
        PyTypeError::new_err(format!(
            "format must be a format-name string like \"pdf\" (or None to sniff it from the content), not {}",
            py_repr(format)
        ))
    })?;
    match string.to_str() {
        Ok(name) => Ok(Some(name)),
        Err(_) => Err(PyValueError::new_err(format!(
            "format must be valid Unicode (no lone surrogates), not {}",
            py_repr(format)
        ))),
    }
}

/// Marshal the `password=` argument under the GIL: `None` (the default —
/// pdf_oxide's fail-closed doctrine, the empty password tried at open) or
/// a `str`, owned so it can ride into the detached pass. A non-str is a
/// `TypeError` naming the argument, the same convention as `format=` —
/// with the one doctrine `format=` does not carry: `password` is the only
/// SECRET-bearing argument, so its refusals never repr the value (a
/// secret echoed into a message rides into the logged tracebacks those
/// messages land in — probed on the 0.5.0 wheel: `password=b"s3cret-pw"`
/// refused as "..., not b's3cret-pw'", the secret in the record). The
/// non-str refusal names the type only ([`type_name`]); the surrogate
/// refusal names the shape, never the text.
fn parse_password(password: Option<&Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    let Some(password) = password else {
        return Ok(None);
    };
    let string = password.cast::<PyString>().map_err(|_| {
        PyTypeError::new_err(format!(
            "password must be a str (the PDF's password), not {}",
            type_name(password)
        ))
    })?;
    match string.to_str() {
        Ok(text) => Ok(Some(text.to_string())),
        Err(_) => Err(PyValueError::new_err(
            "password must be valid Unicode (no lone surrogates), not a str carrying them",
        )),
    }
}

/// Marshal the `max_bytes=` argument under the GIL: `None` (the default —
/// the measured 32 MiB engine-lane ceiling, see the core's
/// `DEFAULT_ANYDOC_INPUT_LIMIT` — the name is historical, the ceiling
/// covers the anydoc and office_oxide lanes) or a positive `int` the
/// caller budgets instead. Wrong TYPES raise `TypeError` (bool first — it
/// launders as an int), wrong VALUES `ValueError` (negative, zero,
/// i64-overflow).
fn parse_max_bytes(max_bytes: Option<&Bound<'_, PyAny>>) -> PyResult<Option<usize>> {
    let Some(max_bytes) = max_bytes else {
        return Ok(None);
    };
    if max_bytes.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "max_bytes must be an int (the engine lane's input ceiling in bytes), not a bool",
        ));
    }
    let Ok(limit) = max_bytes.extract::<i64>() else {
        if max_bytes.cast::<PyInt>().is_ok() {
            return Err(PyValueError::new_err(format!(
                "max_bytes {} is too large",
                py_repr(max_bytes)
            )));
        }
        return Err(PyTypeError::new_err(format!(
            "max_bytes must be an int (the engine lane's input ceiling in bytes), not {}",
            py_repr(max_bytes)
        )));
    };
    if limit <= 0 {
        return Err(PyValueError::new_err(format!(
            "max_bytes must be positive (the engine lane's input ceiling in bytes), not {limit}"
        )));
    }
    // Belt-and-braces (wheels are 64-bit-only): a 32-bit `as` would silently lower the caller's budget.
    let limit = usize::try_from(limit)
        .map_err(|_| PyValueError::new_err(format!("max_bytes {limit} is too large")))?;
    Ok(Some(limit))
}

/// Parse and validate the `backend=` argument under the GIL, before any
/// work runs — the house rule (invalid input raises before the detach, so
/// nothing native ever runs for a call that will fail anyway). `None` (the
/// native layer's default; the typed wrappers pass `Backend.AUTO`, the
/// same choice) routes by the measured table. A non-str is a `TypeError`
/// and a non-vocabulary str (the empty string included) a `ValueError`,
/// both naming the argument and repr'ing the value.
fn parse_backend(backend: Option<&Bound<'_, PyAny>>) -> PyResult<Backend> {
    let Some(backend) = backend else {
        return Ok(Backend::Auto);
    };
    let name = backend
        .cast::<PyString>()
        .map_err(|_| {
            PyTypeError::new_err(format!(
                "backend must be one of ('auto', 'oxide', 'anydoc'), not {}",
                py_repr(backend)
            ))
        })?
        .to_str()
        .map_err(|_| {
            PyValueError::new_err(format!(
                "backend must be valid Unicode (no lone surrogates), not {}",
                py_repr(backend)
            ))
        })?;
    match name {
        "auto" => Ok(Backend::Auto),
        "oxide" => Ok(Backend::Oxide),
        "anydoc" => Ok(Backend::Anydoc),
        other => Err(PyValueError::new_err(format!(
            "backend must be one of ('auto', 'oxide', 'anydoc'), not {other:?}"
        ))),
    }
}

/// Parse and normalize the `pages=` argument under the GIL, before any
/// work runs: a single `int` (one 0-based page), a `list` of ints (the
/// explicit set), or a 2-tuple `(start, stop)` (a half-open range, Python
/// convention). Wrong TYPES raise `TypeError` — Python's own convention
/// (`range(1.0)` and `seq[1.0]` both raise TypeError), with the bool
/// checked FIRST because pyo3's `extract::<i64>` launders `True`/`False`
/// to 1/0 (`isinstance(True, int)` holds): `pages=True` on the pre-fix
/// wheel silently selected page 1, probed 2026-09. Wrong VALUES or SHAPES
/// raise `ValueError`: a negative, an empty list, an empty or backwards
/// range, a tuple that is not a 2-tuple, an index too large for i64. The
/// selection is then deduped into document order (the contract the core
/// assumes), and bounds are validated against the real page count inside
/// the detached pass.
fn parse_pages(pages: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<usize>>> {
    let Some(pages) = pages else {
        return Ok(None);
    };
    let selected: Vec<usize> = if pages.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "booleans are not page indices: pages= takes 0-based ints",
        ));
    } else if let Ok(one) = pages.extract::<i64>() {
        if one < 0 {
            return Err(PyValueError::new_err(format!(
                "page indices are 0-based and cannot be negative, not {one}"
            )));
        }
        // Belt-and-braces (wheels are 64-bit-only): a 32-bit `as` would silently select the wrong page.
        vec![
            usize::try_from(one)
                .map_err(|_| PyValueError::new_err(format!("page index {one} is too large")))?,
        ]
    } else if let Ok(tuple) = pages.cast::<PyTuple>() {
        if tuple.len() != 2 {
            return Err(PyValueError::new_err(
                "a pages= tuple is a (start, stop) range: pass a list for an explicit page set",
            ));
        }
        let start = index_of(&tuple.get_item(0)?)?;
        let stop = index_of(&tuple.get_item(1)?)?;
        if start >= stop {
            return Err(PyValueError::new_err(format!(
                "the pages= range ({start}, {stop}) selects no pages (ranges are half-open, indices 0-based: (1, 3) is the second and third pages)"
            )));
        }
        (start..stop).collect()
    } else if let Ok(list) = pages.cast::<PyList>() {
        let mut selected = Vec::with_capacity(list.len());
        for item in list.iter() {
            selected.push(index_of(&item)?);
        }
        selected
    } else if pages.cast::<PyInt>().is_ok() {
        // The only int that fails extract::<i64> is one that overflows it
        // (10**30 and friends): type-correct, value-absurd — a value
        // refusal, per the convention above.
        return Err(PyValueError::new_err(format!(
            "page index {} is too large",
            py_repr(pages)
        )));
    } else {
        return Err(PyTypeError::new_err(format!(
            "pages must be an int (one 0-based page), a list of ints, or a (start, stop) tuple, not {}",
            py_repr(pages)
        )));
    };
    if selected.is_empty() {
        return Err(PyValueError::new_err(
            "pages= selected no pages: pass None for the whole document",
        ));
    }
    // Dedup into document order: the conversion joins in page order, so
    // caller-supplied order and repeats are normalized away here, under
    // the GIL, before anything native runs.
    let mut deduped = selected;
    deduped.sort_unstable();
    deduped.dedup();
    Ok(Some(deduped))
}

/// One `pages=` index — the scalar value, or one item of a list or range
/// tuple — to a 0-based `usize`. `TypeError` for a non-index type (bool
/// first, then floats, strs, and anything else without `__index__`),
/// `ValueError` for a negative or an i64-overflowing int.
fn index_of(item: &Bound<'_, PyAny>) -> PyResult<usize> {
    if item.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "booleans are not page indices: pages= takes 0-based ints",
        ));
    }
    let Ok(index) = item.extract::<i64>() else {
        if item.cast::<PyInt>().is_ok() {
            return Err(PyValueError::new_err(format!(
                "page index {} is too large",
                py_repr(item)
            )));
        }
        return Err(PyTypeError::new_err(format!(
            "pages= indices must be ints, not {}",
            py_repr(item)
        )));
    };
    if index < 0 {
        return Err(PyValueError::new_err(format!(
            "page indices are 0-based and cannot be negative, not {index}"
        )));
    }
    // Belt-and-braces (wheels are 64-bit-only): a 32-bit `as` would silently select the wrong page.
    usize::try_from(index)
        .map_err(|_| PyValueError::new_err(format!("page index {index} is too large")))
}

/// The value's type name (`"bytes"`, `"int"`, …) for the refusals that
/// must not carry the value: `password=`'s (the one secret-bearing
/// argument — a secret never rides into a message) and `sniff`'s bytes
/// refusal. The complement of [`py_repr`]: names the TYPE, never the
/// value; cannot fail on a real object (every type has a `__name__`), the
/// placeholder the same belt as `py_repr`'s.
fn type_name(value: &Bound<'_, PyAny>) -> String {
    value
        .get_type()
        .name()
        .and_then(|name| name.to_str().map(str::to_string))
        .unwrap_or_else(|_| "<unknown type>".to_string())
}

/// repr() for the argument-validation errors: the repr names the TYPE too
/// (`123` vs `'123'` vs `1.5`), which a bare class name does not, and
/// cannot fail on well-formed values (repr escapes lone surrogates); a
/// hostile `__repr__` that raises or returns non-printable text falls
/// back to a placeholder — the surrounding message already names the
/// argument, which is the part that cannot be lost.
fn py_repr(value: &Bound<'_, PyAny>) -> String {
    value
        .repr()
        .and_then(|repr| repr.to_str().map(str::to_string))
        .unwrap_or_else(|_| "<unrepresentable value>".to_string())
}

/// pdf_oxide's error to the Python side, `pdf_extract`/`pdf_page_count`/
/// `pdf_classify`'s mapping: IO failures are environment failures
/// (`OSError` — pyo3's `From<io::Error>` picks the matched subclass,
/// `FileNotFoundError`/`IsADirectoryError`/`PermissionError`, where the
/// previous `PyOSError::new_err(io.to_string())` always raised a plain
/// `OSError` with the errno buried in the message); every other variant
/// means "these bytes are not a readable PDF" (`ValueError`).
fn pdf_error(err: tors::pdf_impl::PdfError) -> PyErr {
    match err {
        tors::pdf_impl::PdfError::Io(io) => PyErr::from(io),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// `documents_impl`'s error to the Python side: `Io` → `OSError` (pyo3's
/// `From<io::Error>`, the matched subclass — see `pdf_error`);
/// `NeedsOcr` → `NeedsOcrError` carrying `.pages`/`.page_count`;
/// everything else → `ValueError`. All mapping happens after the detached
/// region, and the attributes are set on the materialized exception
/// instance under the GIL. `.pages` holds 0-based page indices — the
/// core re-bases anydoc's 1-based numbers once, at its seam, so every
/// page number this payload names (`pages=`, `pages_needing_ocr`,
/// `pdf_extract`'s list positions, `.pages`) carries ONE convention.
fn document_error(py: Python<'_>, err: DocumentError) -> PyResult<PyErr> {
    match err {
        DocumentError::Io(io) => Ok(PyErr::from(io)),
        DocumentError::NeedsOcr { pages, page_count } => {
            let exc = NeedsOcrError::new_err(format!(
                "pages {} of {page_count} need OCR (indices are 0-based, like pages=): \
                 route this document to an OCR stage",
                pages
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            let value = exc.value(py);
            // Propagated (`?`), not `let _`-swallowed: an instance that
            // silently lacked .pages/.page_count would break the contract
            // far from its cause — the caller's attribute read would
            // AttributeError with no pointer back to this raise. It cannot
            // fail in practice (NeedsOcrError instances are plain
            // BaseException subclasses with a __dict__, and the values are
            // a list and an int), but if it ever could, failing loudly at
            // the raise site beats a silently-missing attribute.
            value.setattr("pages", pages)?;
            value.setattr("page_count", page_count)?;
            Ok(exc)
        }
        DocumentError::UnknownFormat(what) => Ok(PyValueError::new_err(what)),
        DocumentError::UnsupportedBackend { format, backend } => Ok(PyValueError::new_err(
            format!("backend {backend:?} cannot read format {format:?}: use backend='auto'"),
        )),
        DocumentError::Pages(what) => Ok(PyValueError::new_err(what)),
        DocumentError::Convert(what) => Ok(PyValueError::new_err(what)),
    }
}

#[pymodule]
fn _tors_documents(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // The wheel's version, baked from THIS crate's Cargo.toml at build
    // time: release-please bumps Cargo.toml, pyproject.toml, and the root
    // crate together (extra-files — one release, two wheels), so this is
    // the wheel's version by construction. Chosen over reading
    // importlib.metadata at import time (a site-packages scan on every
    // import, plus a static fallback pin that drifts stale in source
    // checkouts); the payload's Python layer re-exports it as
    // tors_documents.__version__.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("NeedsOcrError", m.py().get_type::<NeedsOcrError>())?;
    m.add_class::<PdfClassification>()?;
    m.add_function(wrap_pyfunction!(pdf_classify, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_extract, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_link_uris, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_page_count, m)?)?;
    m.add_function(wrap_pyfunction!(to_markdown, m)?)?;
    m.add_function(wrap_pyfunction!(to_text, m)?)?;
    m.add_function(wrap_pyfunction!(sniff, m)?)?;
    Ok(())
}
