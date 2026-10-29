//! The `tors.documents` payload bindings: the pyo3 surface for the
//! document-format extraction cores in `tors-core` (`pdf_impl`,
//! `documents_impl`, `gfm_strip_impl`), enabled by that crate's `documents`
//! feature. Shipped as the `tors-documents` PyPI wheel, re-exported by the
//! base wheel's `tors.documents` shim: this is the only place these
//! functions exist, so the base `tors` build carries none of the engine
//! weight.
//!
//! The surface, one call per question a document pipeline asks:
//!
//! - [`to_markdown`]/[`to_text`]: any working format to GFM markdown /
//!   plain text (path in, `(format, output)` out, engine routed per the
//!   measured table in `tors-core`'s `documents_impl`, `pages=` for PDF
//!   subsets)
//! - [`sniff`]: the standalone content-marker format detector over bytes
//! - [`pdf_extract`]: the per-page text probe + whole-document markdown in
//!   one pass over one open PDF
//! - [`pdf_page_count`]: the page tree and nothing else
//! - [`pdf_page_labels`]: the page label a reader displays, per page
//!   (the ISO 32000 §12.4.2 numbering tree; `None` = the page declares
//!   no label — never a fabricated ordinal)
//! - [`pdf_security`]: the typed encryption/permissions probe — the
//!   four security fields reported (never a raise on a locked state),
//!   the fail-closed door's carve-out
//! - [`pdf_classify`]: the cheap text-vs-image preflight (which pages are
//!   image-only)
//! - [`page_geometry`]: every unit's declared render box — PDF pages, the
//!   office families' sections/slides/worksheets, the single-flow
//!   formats' one `[None]` — the cheap geometry probe the payload shares
//!   with [`page_layout`]'s per-page boxes (the native
//!   [`PageGeometry`] pyclass is that surface's entry type too)
//! - [`page_layout`]: the combined per-unit probe — geometry plus, for
//!   PDF pages, text spans/lines (words nested) and image boxes, all in
//!   the page's declared render space, one transform owned here, one
//!   open per call (`pages=` subsets like the conversion pair)
//! - [`page_tables`]: the per-cell table-box probe — geometry plus, for
//!   PDF pages, the spatial detector's tables with per-cell text and
//!   boxes in the same render space (`pages=` subsets; office rows serve
//!   geometry only — tables are PDF-only in v1), with the frame-coherence
//!   gate's refused marker (`tables_refused`) on every entry
//! - [`document_structure`]: the multi-format structure probe — one
//!   report whose resolved `format` discriminates and whose payload is
//!   exactly one of the three per-format structures (the docx
//!   heading/list/table/link surface, the xlsx sheets/cells/types
//!   surface, the html node surface; the refused rows answer the
//!   format-level table's own ValueError)
//!
//! GIL model (the crate-wide discipline, applied per function): the
//! argument borrow of the path string plus argument validation happen
//! under the GIL, the whole native pass (file read, format sniff, engine
//! conversion, and for `to_text` the markdown strip) runs under one
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

/// `tors.documents.pdf_classify(path, data=None, password=None,
/// backend="auto", max_bytes=None) -> PdfClassification`: or
/// `pdf_classify(data=...)`: the cheap
/// text-vs-image preflight over a PDF:
/// no content conversion, no OCR, no rasterization. The answer to "does
/// this PDF have a text layer, or is it an image we can do nothing with
/// locally":
///
/// - `.page_count`: the page tree.
/// - `.page_kinds`: every page's verdict, page order: `"text"` (native
///   text layer), `"scanned"` (image-dominated: OCR the page),
///   `"image_text"` (hybrid), `"mixed"`, or `"empty"` (blank/near-empty:
///   neither extract nor OCR, not an error, and distinct from a scan).
/// - `.pages_needing_ocr`: the 0-based indices of the image-only pages
///   (the `"scanned"` ones; `"empty"` pages are deliberately not listed:
///   a blank page is not an image to recover).
/// - `.has_text`: at least one page carries a text layer (`"text"`,
///   `"image_text"`, or `"mixed"`); extraction will yield something.
/// - `.image_only`: every page is image-only; nothing local can read
///   this document, route it to an OCR stage.
///
/// `backend` names the engine lane, the same vocabulary the conversion
/// pair takes: `"auto"` (the default; the native layer takes `None` as
/// the same choice) and `"oxide"` both run pdf_oxide:
/// the mapping `engine_for` makes for PDF ("oxide" is the oxide-family
/// engine for this format), byte-identical output either way.
/// `"anydoc"` is refused before any work runs, a named `ValueError`
/// raised under the GIL ahead of the detach and the encrypted door
/// check: a capability refusal, not the format-level UnsupportedBackend
/// (PDF+anydoc is a usable conversion pair): anydoc's PDF surface is
/// whole-document markdown only (~anydoc-0.2.4/src/formats/pdf.rs), and
/// this call needs per-page classification, of which that engine's only
/// per-page knowledge is the binary needs-OCR refusal. Whole-document
/// conversion is the conversion pair's call: `to_markdown`/`to_text`
/// with `backend="anydoc"` (`NeedsOcrError` is that lane's scanned-page
/// signal).
///
/// `max_bytes` is the caller's input budget in bytes (a positive `int`).
/// An explicit budget binds this family pre-read exactly as it binds the
/// conversion pair, the shared source-spine gate: the `path=`'s size at
/// the stat and again on the open handle, the `data=` length before the
/// copy; never a byte over budget is read or copied, and it overrides the
/// default in either direction. `None` (the default) still runs no
/// lane-specific policy on the pdf lane, the unchanged doctrine: the
/// 32 MiB default ceiling is the core's post-read check on the
/// amplified lanes (anydoc/office_oxide/html), lanes these PDF-only calls never run
/// (pdf_oxide's own resource limits govern here); the pdf lane instead
/// reads under [`MAX_INPUT_READ`], the finite 512 MiB backstop, a
/// memory-safety floor rather than a lane policy.
///
/// Encrypted documents fail closed (`ValueError`, pdf_oxide's security
/// rule: a security state is never masked as "all pages empty"):
/// `password=` unlocks one (a wrong password is its own clean
/// `ValueError`, and the unlock applies to every PDF entry, not just
/// this one; a contract failure precedes the work, so an encrypted
/// document under `backend="anydoc"` surfaces the capability refusal,
/// never the door-check error). A `path=` naming a FIFO, device, or
/// socket is a `ValueError` naming `path` and the kind (the shared
/// source spine reads regular files only: open(2) on a no-writer FIFO
/// blocks forever with the GIL released, and `/dev/zero` reads
/// unboundedly); an input over an explicit `max_bytes` is the ceiling
/// `ValueError`, both sizes named, refused before the read or copy.
///
/// GIL model: the source marshalling under the GIL (`parse_source`: the
/// path type-checked and Unicode-validated, a `data=` call borrowing
/// the immutable buffer: the one bytes copy runs inside the detach,
/// see [`Source`]), the read+classify under
/// `py.detach`, then the return object construction (three attribute
/// values, O(pages)): `pdf_extract`'s residue class with the smallest
/// native pass of the PDF family.
#[pyfunction(signature = (
    path = None,
    data = None,
    password = None,
    backend = None,
    max_bytes = None,
))]
pub fn pdf_classify(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
    backend: Option<&Bound<'_, PyAny>>,
    max_bytes: Option<&Bound<'_, PyAny>>,
) -> PyResult<PdfClassification> {
    let source = parse_source(path, data, "pdf_classify")?;
    let password = parse_password(password)?;
    let backend = parse_backend(backend)?;
    if backend == Backend::Anydoc {
        return Err(refuse_anydoc(
            "per-page classification (its only per-page knowledge is the binary \
             needs-OCR refusal)",
        ));
    }
    let max_bytes = parse_max_bytes(max_bytes)?;
    let classification = py
        .detach(move || {
            let bytes = source.into_bytes(max_bytes)?;
            tors::pdf_impl::classify(bytes, password.as_deref()).map_err(PdfLaneError::Pdf)
        })
        .map_err(pdf_lane_error)?;
    Ok(PdfClassification {
        page_count: classification.page_count,
        page_kinds: classification.page_kinds,
        pages_needing_ocr: classification.pages_needing_ocr,
    })
}

/// The `pdf_classify` result: the text-vs-image preflight's answer, with
/// the routing rules derived for the caller. Constructed only by
/// `pdf_classify`; the attribute reads are plain `#[getter]`s. The kinds
/// ride as the core's `PageKind`: the rules-once doctrine across the
/// seam: `has_text` and the scanned match call the core's own rule, never
/// a re-derived string match here: stringified only at the `page_kinds`
/// getter.
#[pyclass(module = "_tors_documents")]
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

    /// At least one page carries a text layer: extraction will yield
    /// something. `False` for a zero-page, all-blank, or fully-scanned
    /// document.
    #[getter]
    fn has_text(&self) -> bool {
        self.page_kinds.iter().any(|kind| kind.has_text())
    }

    /// Every page is image-only: nothing local can read this document;
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

/// `tors.documents.pdf_extract(path, data=None, password=None,
/// backend="auto", max_bytes=None) -> (pages, markdown)`: or
/// `pdf_extract(data=...)`: read a PDF
/// (from `path`, or from the caller's `data=` bytes) and return
/// `(per_page_plain_text, whole_document_markdown)`: one native pass over
/// one open document (the parse is paid once for both outputs).
///
/// `pages` is a `list[str]`, one entry per page in page order, from
/// pdf_oxide's per-page plain-text surface (`extract_text`): the
/// text-layer-probe view. An image-only/scanned page is an empty string,
/// not an error, and a zero-page or textless document yields empty output:
/// routing decisions ("this PDF needs OCR") are the caller's, made on these
/// values, never silently made here. `markdown` is pdf_oxide's
/// `to_markdown_all` under default conversion options: heading detection
/// on, images off (no base64 bloat), Tagged-PDF structure-tree reading
/// order falling back to XY-Cut on untagged documents, and `/Link`
/// annotations rendered as `[text](uri)`.
///
/// `backend` names the engine lane, the same vocabulary the conversion
/// pair takes: `"auto"` (the default; the native layer takes `None` as
/// the same choice) and `"oxide"` both run pdf_oxide:
/// the mapping `engine_for` makes for PDF ("oxide" is the oxide-family
/// engine for this format), byte-identical output either way. This is
/// the probe-rich call, and `"anydoc"` is refused before any work runs,
/// a named `ValueError` raised under the GIL ahead of the detach and the
/// encrypted door check: a capability refusal, not the format-level
/// UnsupportedBackend (PDF+anydoc is a usable conversion pair):
/// anydoc's PDF surface is whole-document markdown only
/// (~anydoc-0.2.4/src/formats/pdf.rs), and this call needs the per-page
/// text probe, which is the OCR-routing signal and has no anydoc
/// counterpart. Whole-document conversion is the conversion pair's
/// call: `to_markdown`/`to_text` with `backend="anydoc"`
/// (`NeedsOcrError` is that lane's scanned-page signal).
///
/// `max_bytes` is the caller's input budget in bytes (a positive `int`).
/// An explicit budget binds this family pre-read exactly as it binds the
/// conversion pair, the shared source-spine gate: the `path=`'s size at
/// the stat and again on the open handle, the `data=` length before the
/// copy; never a byte over budget is read or copied, and it overrides the
/// default in either direction. `None` (the default) still runs no
/// lane-specific policy on the pdf lane, the unchanged doctrine: the
/// 32 MiB default ceiling is the core's post-read check on the
/// amplified lanes (anydoc/office_oxide/html), lanes these PDF-only calls never run
/// (pdf_oxide's own resource limits govern here); the pdf lane instead
/// reads under [`MAX_INPUT_READ`], the finite 512 MiB backstop, a
/// memory-safety floor rather than a lane policy.
///
/// Errors: a missing/unreadable file raises `OSError`; anything that fails
/// to parse as a PDF raises `ValueError` with pdf_oxide's reason; a `path=`
/// naming a FIFO, device, or socket is a `ValueError` naming `path` and
/// the kind (the shared source spine reads regular files only);
/// `backend="anydoc"` is the capability `ValueError` above (a contract
/// failure precedes the work, so an encrypted document under it surfaces
/// the backend refusal, never the door-check error); an input over an
/// explicit `max_bytes` is the ceiling `ValueError`, both sizes named,
/// refused before the read or copy. The input-side refusals are
/// constructed after the GIL is reacquired; the backend one is raised
/// under the GIL, before any work runs.
///
/// GIL model: the whole pass (file read, PDF parse, per-page text
/// extraction, markdown conversion) runs under one `py.detach`. The
/// GIL-held residue is the path str marshalling (`parse_path`, a handful
/// of bytes), plus the return marshalling: one `str` per page and the
/// markdown string, O(output). The official pdf_oxide pyo3 wheel measures
/// as GIL-held per call instead (worst heartbeat gap 23.6ms on a 9-page
/// document under a 10ms ping): the hazard this binding exists
/// to remove; the band is pinned by the shared documents suite.
#[pyfunction(signature = (
    path = None,
    data = None,
    password = None,
    backend = None,
    max_bytes = None,
))]
pub fn pdf_extract(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
    backend: Option<&Bound<'_, PyAny>>,
    max_bytes: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Vec<String>, String)> {
    let source = parse_source(path, data, "pdf_extract")?;
    let password = parse_password(password)?;
    let backend = parse_backend(backend)?;
    if backend == Backend::Anydoc {
        return Err(refuse_anydoc(
            "the per-page text probe (the OCR-routing signal)",
        ));
    }
    let max_bytes = parse_max_bytes(max_bytes)?;
    let extracted = py
        .detach(move || {
            let bytes = source.into_bytes(max_bytes)?;
            tors::pdf_impl::extract(bytes, password.as_deref()).map_err(PdfLaneError::Pdf)
        })
        .map_err(pdf_lane_error)?;
    Ok((extracted.pages, extracted.markdown))
}

/// `tors.documents.pdf_page_count(path, data=None, password=None,
/// backend="auto", max_bytes=None) -> int` (or `pdf_page_count(data=...)`):
/// read a PDF and return its page count:
/// the page tree and nothing else, no content extraction. For gating
/// expensive downstream work (an OCR/conversion pass that scales with page
/// count) without paying for any of it.
///
/// `backend` names the engine lane, the same vocabulary the conversion
/// pair takes: `"auto"` (the default; the native layer takes `None` as
/// the same choice) and `"oxide"` both run pdf_oxide:
/// the mapping `engine_for` makes for PDF ("oxide" is the oxide-family
/// engine for this format), byte-identical output either way.
/// `"anydoc"` is refused before any work runs, a named `ValueError`
/// raised under the GIL ahead of the detach and the encrypted door
/// check: a capability refusal, not the format-level UnsupportedBackend
/// (PDF+anydoc is a usable conversion pair): anydoc's PDF surface is
/// whole-document markdown only (~anydoc-0.2.4/src/formats/pdf.rs), and
/// this call needs the page tree (a count its reader never returns on
/// success). Whole-document conversion is the
/// conversion pair's call: `to_markdown`/`to_text` with
/// `backend="anydoc"` (`NeedsOcrError` is that lane's scanned-page
/// signal).
///
/// `max_bytes` is the caller's input budget in bytes (a positive `int`).
/// An explicit budget binds this family pre-read exactly as it binds the
/// conversion pair, the shared source-spine gate: the `path=`'s size at
/// the stat and again on the open handle, the `data=` length before the
/// copy; never a byte over budget is read or copied, and it overrides the
/// default in either direction. `None` (the default) still runs no
/// lane-specific policy on the pdf lane, the unchanged doctrine: the
/// 32 MiB default ceiling is the core's post-read check on the
/// amplified lanes (anydoc/office_oxide/html), lanes these PDF-only calls never run
/// (pdf_oxide's own resource limits govern here); the pdf lane instead
/// reads under [`MAX_INPUT_READ`], the finite 512 MiB backstop, a
/// memory-safety floor rather than a lane policy.
///
/// Errors: `OSError` for a missing/unreadable file, `ValueError` for bytes
/// that do not parse as a PDF, `ValueError` naming `path` for a
/// FIFO/device/socket path (regular files only, the shared source spine),
/// the capability `ValueError` for `backend="anydoc"` (under the GIL,
/// before any work runs, so an encrypted document under it surfaces the
/// backend refusal, never the door-check error), and the ceiling
/// `ValueError` for an input over an explicit `max_bytes` (both sizes
/// named, refused before the read or copy): the input-side refusals
/// constructed after the GIL is reacquired.
///
/// GIL model: `pdf_extract`'s shape with the smallest possible return:
/// the path str marshalling (`parse_path`), the open+page-tree walk under
/// `py.detach`, and a single `int` back, no marshalling class at all.
#[pyfunction(signature = (
    path = None,
    data = None,
    password = None,
    backend = None,
    max_bytes = None,
))]
pub fn pdf_page_count(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
    backend: Option<&Bound<'_, PyAny>>,
    max_bytes: Option<&Bound<'_, PyAny>>,
) -> PyResult<usize> {
    let source = parse_source(path, data, "pdf_page_count")?;
    let password = parse_password(password)?;
    let backend = parse_backend(backend)?;
    if backend == Backend::Anydoc {
        return Err(refuse_anydoc(
            "the page tree (a count its reader never returns on success)",
        ));
    }
    let max_bytes = parse_max_bytes(max_bytes)?;
    py.detach(move || {
        let bytes = source.into_bytes(max_bytes)?;
        tors::pdf_impl::page_count(bytes, password.as_deref()).map_err(PdfLaneError::Pdf)
    })
    .map_err(pdf_lane_error)
}

/// `tors.documents.pdf_page_labels(path, data=None, password=None,
/// backend="auto", max_bytes=None) -> list[str | None]` (or
/// `pdf_page_labels(data=...)`): read a PDF and return the page label a
/// reader displays for every page ("i", "A-1", "12") — the ISO 32000
/// §12.4.2 numbering tree and nothing else, no content extraction.
/// Citations and UIs name pages the way the document does, not by
/// 0-based ordinal.
///
/// `None` means the page DECLARES no label — no `/PageLabels` tree at
/// all, or the page sits before the tree's first range (gaps exist only
/// there; the list is always the page count long, page order) — and is
/// the caller's fabricate-an-ordinal signal: no label is ever
/// fabricated here (the #207 law). A declared label is always `Some`,
/// including the empty string: a styleless range (`/S` absent) with no
/// prefix declares the empty label, and a styleless range WITH a prefix
/// yields that same string for every page it covers (constant labels, a
/// per-page counter only the numbered styles carry). A prefix that
/// fails to decode (unmapped PDFDocEncoding bytes dropped; invalid
/// UTF-16) is silently lost, so such a page can also answer `Some("")`
/// (styleless) or the bare number (a style present).
///
/// The covering rule is the engine's minus its ordinal fallback: the
/// last range in the stable-sorted tree with `start_page ≤ i` covers
/// page `i`; two ranges sharing a start page resolve to the last in
/// traversal order (root `/Nums` before `/Kids`, kids in array order).
/// Labels render on u64 arithmetic under the bounded-label law: a label
/// the render builds past 4 KiB (`pdf_impl::MAX_LABEL_CHARS`) bounds OUT
/// to `None` for that page, never a truncated numeral — the subtractive
/// Roman render is one char per thousand of the value, so a legal
/// `/St 4294967294` renders 4,294,967 chars per page (measured 858,994,700
/// total chars, 1.67 GiB RSS at 200 pages before the bound), a value
/// display-prohibitive by construction. Decimal and alpha renders are
/// O(log n) and can never reach the bound. The arithmetic itself never
/// panics or wraps (never the engine formatter's `u32 + u32`; an
/// over-u32 `/St` carries the engine's own parse-time truncation, e.g.
/// 5000000000 reads back as 705032704). Rotation and crop are irrelevant
/// (tree data). A corrupt tree (a non-dict `/PageLabels`, tree node,
/// `/Kids` member, or label entry) is presented as "no labels" — every
/// page `None`, the content-read-failure law, the same list a tree-less
/// document answers; that is a deliberate masking, stated here the way
/// `tables_refused: false` is.
///
/// `backend` names the engine lane, the same vocabulary the conversion
/// pair takes: `"auto"` (the default; the native layer takes `None` as
/// the same choice) and `"oxide"` both run pdf_oxide:
/// the mapping `engine_for` makes for PDF ("oxide" is the oxide-family
/// engine for this format), byte-identical output either way.
/// `"anydoc"` is refused before any work runs, a named `ValueError`
/// raised under the GIL ahead of the detach and the encrypted door
/// check: a capability refusal, not the format-level UnsupportedBackend
/// (PDF+anydoc is a usable conversion pair): anydoc's PDF surface is
/// whole-document markdown only (~anydoc-0.2.4/src/formats/pdf.rs), and
/// this call needs the /PageLabels numbering tree (labels its reader
/// never renders). Whole-document conversion is the
/// conversion pair's call: `to_markdown`/`to_text` with
/// `backend="anydoc"` (`NeedsOcrError` is that lane's scanned-page
/// signal).
///
/// `max_bytes` is the caller's input budget in bytes (a positive `int`).
/// An explicit budget binds this family pre-read exactly as it binds the
/// conversion pair, the shared source-spine gate: the `path=`'s size at
/// the stat and again on the open handle, the `data=` length before the
/// copy; never a byte over budget is read or copied, and it overrides the
/// default in either direction. `None` (the default) still runs no
/// lane-specific policy on the pdf lane, the unchanged doctrine: the
/// 32 MiB default ceiling is the core's post-read check on the
/// amplified lanes (anydoc/office_oxide/html), lanes these PDF-only calls never run
/// (pdf_oxide's own resource limits govern here); the pdf lane instead
/// reads under the finite 512 MiB backstop, a
/// memory-safety floor rather than a lane policy.
///
/// `password` unlocks an encrypted PDF. The door governs unchanged (the
/// single encryption authority): a locked document raises the family's
/// `ValueError` — never ciphertext labels (the labels are read from
/// encrypted string objects, and the engine's extractor would return
/// ciphertext prefixes on a locked file; `open()` is the only gate) —
/// and owner-password-only files authenticate with the empty password
/// and read their decrypted labels.
///
/// Errors: `OSError` for a missing/unreadable file, `ValueError` for bytes
/// that do not parse as a PDF, `ValueError` naming `path` for a
/// FIFO/device/socket path (regular files only, the shared source spine),
/// the capability `ValueError` for `backend="anydoc"` (under the GIL,
/// before any work runs, so an encrypted document under it surfaces the
/// backend refusal, never the door-check error), the encrypted door
/// `ValueError` (with `password=`: its "did not unlock" error), and the ceiling
/// `ValueError` for an input over an explicit `max_bytes` (both sizes
/// named, refused before the read or copy): the input-side refusals
/// constructed after the GIL is reacquired.
///
/// GIL model: `pdf_page_count`'s shape — the path str marshalling
/// (`parse_path`), the open+tree walk under `py.detach`, and one
/// `list[str | None]` back (O(pages) of Python strings, no
/// marshalling class).
#[pyfunction(signature = (
    path = None,
    data = None,
    password = None,
    backend = None,
    max_bytes = None,
))]
pub fn pdf_page_labels(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
    backend: Option<&Bound<'_, PyAny>>,
    max_bytes: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<Option<String>>> {
    let source = parse_source(path, data, "pdf_page_labels")?;
    let password = parse_password(password)?;
    let backend = parse_backend(backend)?;
    if backend == Backend::Anydoc {
        return Err(refuse_anydoc(
            "the /PageLabels numbering tree (labels its reader never renders)",
        ));
    }
    let max_bytes = parse_max_bytes(max_bytes)?;
    py.detach(move || {
        let bytes = source.into_bytes(max_bytes)?;
        tors::pdf_impl::page_labels(bytes, password.as_deref()).map_err(PdfLaneError::Pdf)
    })
    .map_err(pdf_lane_error)
}

/// `tors.documents.pdf_security(path, data=None, password=None,
/// backend="auto", max_bytes=None) -> PdfSecurity` (or
/// `pdf_security(data=...)`): the typed encryption/permissions probe —
/// report the document's security state; never unlock beyond the
/// supplied password, never raise for a state the caller asked about.
/// Right now "can I read this document and what does it allow" is
/// answered by poking (open, `authenticate("")`, `permissions()`, then
/// see which call refuses); this answers it in one call, and the
/// fail-closed door stays every content surface's single encryption
/// authority — this surface REPORTS state instead of raising on it.
///
/// The four fields (the reading rule first — they are not one boolean):
/// `readable = !encrypted || unlocked_with_supplied_password ||
/// owner_password_only`. The DOOR is deliberately stricter: a wrong
/// supplied password raises on every content surface even when
/// `owner_password_only` is true (owner-only + wrong password reports
/// `{encrypted: true, unlocked: false, permissions: Some, owner_only:
/// true}` while `pdf_page_count(…, "wrong")` raises) — never read
/// `unlocked == false` as "locked"; read the rule.
///
/// - `encrypted`: the document carries an `/Encrypt` entry (handler
///   present, or the trailer names one), before and regardless of any
///   authentication.
/// - `unlocked_with_supplied_password`: the content was readable under
///   the credentials THIS call held — the supplied password's result
///   when one was given, else the empty-password open law (`password=
///   None` MEANS the implicit empty attempt, never "no attempt"):
///   owner-only + no password → true (the reads-fine case), user-locked
///   + no password → false, wrong password → false. A supplied password
///   on an unencrypted document is a no-op (nothing to authenticate),
///   never an error.
/// - `permissions`: the declared `/P` permission flags (a
///   [`PdfPermissions`], below) — `None` when unencrypted AND when the
///   `/Encrypt` dict fails to initialize; `Some` otherwise, INCLUDING
///   pre-authentication (the raw `/P` off the parsed dict). `/P` is
///   ADVISORY (§7.6.3.2): this probe reports, callers enforce.
/// - `owner_password_only`: the empty password authenticated — the
///   owner-password-only shape, whose content surfaces read fine with
///   no password. An inference from the open's own single implicit
///   empty attempt, never a credential-kind probe (the engine never
///   exposes user-vs-owner): an empty OWNER password with a non-empty
///   USER password also reports true. Honest meaning: "the empty string
///   opens this document".
///
/// Three distinct states collapse into `encrypted == true &&
/// permissions is None` and are NOT distinguishable in v1: a malformed
/// `/Encrypt` dict; an unsupported `/V`/`/R`; and a crypto-policy
/// denial (`fips_strict`'s R≤4/MD5 refusal). A caller cannot tell
/// "needs a password" from "this build/policy refuses the cipher" —
/// deliberate (no typed flag), and the report is IDENTICAL with or
/// without a supplied password (a state report must not depend on the
/// caller's password choice).
///
/// `Some(permissions)` vs `None` reads "the /P dict parsed and its
/// flags are these" vs "no password answer exists in this build" — the
/// `needs-a-password`/`no-answer` split the conflation above names.
///
/// One password is tried per call (the supplied one) plus the open's
/// own single implicit empty attempt: no amplification, no dictionary
/// walk, no /Encrypt version or cipher enumeration.
///
/// `backend` names the engine lane, the same vocabulary the conversion
/// pair takes: `"auto"` (the default; the native layer takes `None` as
/// the same choice) and `"oxide"` both run pdf_oxide:
/// the mapping `engine_for` makes for PDF ("oxide" is the oxide-family
/// engine for this format), byte-identical output either way.
/// `"anydoc"` is refused before any work runs, a named `ValueError`
/// raised under the GIL ahead of the detach: a capability refusal, not
/// the format-level UnsupportedBackend (PDF+anydoc is a usable
/// conversion pair): anydoc's PDF surface is whole-document markdown
/// only (~anydoc-0.2.4/src/formats/pdf.rs), and
/// this call needs the /Encrypt security state (an encryption report
/// its reader never makes). Whole-document conversion is the
/// conversion pair's call: `to_markdown`/`to_text` with
/// `backend="anydoc"` (`NeedsOcrError` is that lane's scanned-page
/// signal).
///
/// `max_bytes` is the caller's input budget in bytes (a positive `int`).
/// An explicit budget binds this family pre-read exactly as it binds the
/// conversion pair, the shared source-spine gate: the `path=`'s size at
/// the stat and again on the open handle, the `data=` length before the
/// copy; never a byte over budget is read or copied, and it overrides the
/// default in either direction. `None` (the default) still runs no
/// lane-specific policy on the pdf lane, the unchanged doctrine: the
/// 32 MiB default ceiling is the core's post-read check on the
/// amplified lanes (anydoc/office_oxide/html), lanes these PDF-only calls never run
/// (pdf_oxide's own resource limits govern here); the pdf lane instead
/// reads under the finite 512 MiB backstop, a
/// memory-safety floor rather than a lane policy.
///
/// Errors: `OSError` for a missing/unreadable file, `ValueError` for bytes
/// that do not parse as a PDF, `ValueError` naming `path` for a
/// FIFO/device/socket path (regular files only, the shared source spine),
/// the capability `ValueError` for `backend="anydoc"` (under the GIL,
/// before any work runs), and the ceiling
/// `ValueError` for an input over an explicit `max_bytes` (both sizes
/// named, refused before the read or copy). NOT raised: the encrypted
/// door itself — this surface reports the locked state instead (the
/// carve-out the fail-closed doctrine documents); an encrypted document
/// under `backend="anydoc"` still surfaces the backend refusal (under
/// the GIL, before any work runs).
///
/// GIL model: `pdf_page_count`'s shape — the path str marshalling
/// (`parse_path`), the open+state read under `py.detach`, then one
/// pyclass back (O(1); the `permissions` getter materializes its own
/// pyclass on access, post-detach).
#[pyfunction(signature = (
    path = None,
    data = None,
    password = None,
    backend = None,
    max_bytes = None,
))]
pub fn pdf_security(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
    backend: Option<&Bound<'_, PyAny>>,
    max_bytes: Option<&Bound<'_, PyAny>>,
) -> PyResult<PdfSecurity> {
    let source = parse_source(path, data, "pdf_security")?;
    let password = parse_password(password)?;
    let backend = parse_backend(backend)?;
    if backend == Backend::Anydoc {
        return Err(refuse_anydoc(
            "the /Encrypt security state (an encryption report its reader never makes)",
        ));
    }
    let max_bytes = parse_max_bytes(max_bytes)?;
    py.detach(move || {
        let bytes = source.into_bytes(max_bytes)?;
        tors::pdf_impl::security(bytes, password.as_deref()).map_err(PdfLaneError::Pdf)
    })
    .map_err(pdf_lane_error)
    .map(pdf_security_inner)
}

/// The core's `PdfSecurity` into the binding's pyclass (the
/// `PageTables` entry precedent: one constructor, post-detach).
fn pdf_security_inner(security: tors::pdf_impl::PdfSecurity) -> PdfSecurity {
    PdfSecurity { inner: security }
}

/// The `pdf_security` answer as the native pyclass: the four fields the
/// reading rule composes (`readable = !encrypted ||
/// unlocked_with_supplied_password || owner_password_only` — the trap
/// cell is owner-only + a WRONG supplied password: `unlocked` false
/// there while the document reads fine with NO password, and the door
/// raises on the wrong one; never read `unlocked == false` as
/// "locked").
///
/// Exposed DIRECTLY (no Python wrapper view) — the `PageGeometry`
/// deviation note applies: the fields are bools and an optional
/// [`PdfPermissions`], exactly as returned, nothing to re-type; `==` is
/// structural (the four fields).
#[pyclass(module = "_tors_documents")]
pub struct PdfSecurity {
    inner: tors::pdf_impl::PdfSecurity,
}

#[pymethods]
impl PdfSecurity {
    /// The document carries an `/Encrypt` entry (handler present, or the
    /// trailer names one) — before and regardless of any authentication.
    #[getter]
    fn encrypted(&self) -> bool {
        self.inner.encrypted
    }

    /// The content was readable under the credentials THIS call held:
    /// the supplied password's result when one was given, else the
    /// empty-password open law (`password=None` MEANS the implicit
    /// empty attempt, never "no attempt").
    #[getter]
    fn unlocked_with_supplied_password(&self) -> bool {
        self.inner.unlocked_with_supplied_password
    }

    /// The declared `/P` permission flags (a `PdfPermissions`, or None):
    /// None when unencrypted AND when the /Encrypt dict fails to
    /// initialize; Some otherwise, INCLUDING pre-authentication. /P is
    /// advisory (§7.6.3.2): the probe reports, callers enforce.
    #[getter]
    fn permissions(&self) -> Option<PdfPermissions> {
        self.inner.permissions.map(|p| PdfPermissions { inner: p })
    }

    /// The empty password authenticated (the owner-password-only
    /// shape): an inference from the open's own single implicit empty
    /// attempt, never a credential-kind probe — an empty OWNER password
    /// with a non-empty USER password also reports true. Honest
    /// meaning: "the empty string opens this document".
    #[getter]
    fn owner_password_only(&self) -> bool {
        self.inner.owner_password_only
    }

    /// Structural equality (all four fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        security_repr(&self.inner)
    }
}

/// The `PdfSecurity` repr, the Python shape (the repr law: never the
/// Rust Debug).
fn security_repr(security: &tors::pdf_impl::PdfSecurity) -> String {
    let py_bool = |value: bool| if value { "True" } else { "False" };
    let permissions = match &security.permissions {
        Some(_) => "PdfPermissions(...)".to_string(),
        None => "None".to_string(),
    };
    format!(
        "PdfSecurity(encrypted={}, unlocked_with_supplied_password={}, permissions={}, owner_password_only={})",
        py_bool(security.encrypted),
        py_bool(security.unlocked_with_supplied_password),
        permissions,
        py_bool(security.owner_password_only),
    )
}

/// One decoded `/P` permission flag set (`pdf_security`'s
/// `permissions`): the eight named bits of the §7.6.3.2 Table 22 decode,
/// plus `raw_p`, the raw `/P` integer. The named bits are what callers
/// enforcing a policy should read; `raw_p` is provenance.
///
/// Exposed DIRECTLY (no Python wrapper view) — the `PageGeometry`
/// deviation note applies; `==` is structural (the nine fields).
#[pyclass(module = "_tors_documents")]
pub struct PdfPermissions {
    inner: tors::pdf_impl::PdfPermissions,
}

#[pymethods]
impl PdfPermissions {
    /// Bit 3: print (low resolution).
    #[getter]
    fn print_low_res(&self) -> bool {
        self.inner.print_low_res
    }

    /// Bit 4: modify (other than annotation / form fill).
    #[getter]
    fn modify(&self) -> bool {
        self.inner.modify
    }

    /// Bit 5: copy text and graphics.
    #[getter]
    fn copy(&self) -> bool {
        self.inner.copy
    }

    /// Bit 6: annotate and form-fill.
    #[getter]
    fn annotate(&self) -> bool {
        self.inner.annotate
    }

    /// Bit 9 (rev ≥ 3): fill interactive form fields.
    #[getter]
    fn fill_forms(&self) -> bool {
        self.inner.fill_forms
    }

    /// Bit 10 (rev ≥ 3): extract for accessibility.
    #[getter]
    fn accessibility(&self) -> bool {
        self.inner.accessibility
    }

    /// Bit 11 (rev ≥ 3): assemble (insert/rotate/delete pages).
    #[getter]
    fn assemble(&self) -> bool {
        self.inner.assemble
    }

    /// Bit 12 (rev ≥ 3): print high resolution.
    #[getter]
    fn print_high_res(&self) -> bool {
        self.inner.print_high_res
    }

    /// The raw `/P` integer, the engine's `*i as i32` view: a `/P`
    /// outside i32 (e.g. 4294967292) reports truncated (−4) — lossless
    /// ONLY within i32, the two's-complement form the spec itself
    /// writes. Prefer the named bits; keep this for provenance.
    #[getter]
    fn raw_p(&self) -> i32 {
        self.inner.raw_p
    }

    /// Structural equality (all nine fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        permissions_repr(&self.inner)
    }
}

/// The `PdfPermissions` repr, the Python shape.
fn permissions_repr(permissions: &tors::pdf_impl::PdfPermissions) -> String {
    let py_bool = |value: bool| if value { "True" } else { "False" };
    format!(
        "PdfPermissions(print_low_res={}, modify={}, copy={}, annotate={}, \
         fill_forms={}, accessibility={}, assemble={}, print_high_res={}, raw_p={})",
        py_bool(permissions.print_low_res),
        py_bool(permissions.modify),
        py_bool(permissions.copy),
        py_bool(permissions.annotate),
        py_bool(permissions.fill_forms),
        py_bool(permissions.accessibility),
        py_bool(permissions.assemble),
        py_bool(permissions.print_high_res),
        permissions.raw_p,
    )
}

/// The `backend="anydoc"` capability probe's GIL-side half
/// (`page_geometry`'s only pre-detach work beyond argument parsing):
/// resolve the format as far as it resolves WITHOUT the authoritative
/// read — an explicit `format=` decides alone, the `data=` lane sniffs
/// its own resident bytes, the `path=` lane reads the same bounded 64
/// KiB prefix the two-phase read's first phase reads — and return the
/// geometry capability refusal when the probed row is one anydoc cannot
/// serve (the pdf row, the office row; the refusal names the container:
/// an OLE workbook says `xls`).
///
/// `None` = the format did not resolve (defer to the core's
/// authoritative resolve, post-read) or the row serves the backend. The
/// path lane's IO failures defer the same way — the detached read
/// re-runs every gate and produces the real `OSError`/refusal, so a
/// missing file stays an `OSError` and never a swallowed decision. The
/// FIFO/device gate runs BEFORE the open: open(2) on a no-writer FIFO
/// blocks, and this probe runs under the GIL, where that block is
/// unreclaimable.
fn probe_capability_refusal(
    source: &Source<'_>,
    format: Option<&str>,
    backend: Backend,
) -> Option<DocumentError> {
    let probed: Result<Option<tors::documents_impl::Kind>, DocumentError> = match source {
        Source::Data(bytes) => {
            documents_impl::probe_geometry_capability(probe_sniff_bytes(bytes, format), None, format, backend)
        }
        Source::Path(path) => {
            // The name-decided answer touches nothing (the
            // probe_sniff_bytes law, the path branch): an explicit
            // format= that resolves to a non-Excel kind decides from the
            // name alone — open(2) and the 64 KiB read have no business
            // running, and their failures (a chmod-000 file, a
            // directory, a missing path) must not bury the refusal.
            if name_decides_probe(format) {
                return documents_impl::probe_geometry_capability(&[], None, format, backend).err();
            }
            let gated = std::fs::metadata(path)
                .map_err(|_| ())
                .and_then(|meta| refuse_non_regular(path, meta.file_type()).map_err(|_| ()));
            if gated.is_err() {
                return None;
            }
            let mut file = std::fs::File::open(path).ok()?;
            let (prefix, _over) =
                documents_impl::read_bounded_into(&mut file, Vec::new(), SNIFF_PREFIX).ok()?;
            documents_impl::probe_geometry_capability(&prefix, path.to_str(), format, backend)
        }
    };
    probed.err()
}

/// The probe's own byte budget for a `data=` call: an explicit `format=`
/// that resolves to a non-Excel kind consults NO bytes (the core's
/// `resolve` short-circuits on the name; the probe's tables consult
/// bytes for the Excel kind's OLE-container case only) — the full
/// buffer is skipped entirely. Every other case hands the probe the
/// bounded 64 KiB prefix the path lane reads — NEVER the full buffer:
/// pre-fix the probe sniffed all of `data=` under the GIL
/// (`looks_like_html` is a lossy-UTF-8 walk of the whole slice), and a
/// 128 MiB blob held the GIL 356.5 ms max (measured), against the
/// crate's own GIL contract — the whole native pass runs under
/// `py.detach`.
///
/// The Excel carve-out keeps the prefix so the OLE-container decision
/// (a legacy xls workbook or an encrypted xlsx package — the union
/// refusal) can fire pre-read exactly as before. A name that resolves
/// to NOTHING (an unknown `format=`) falls through to the prefix path
/// and defers, exactly as an unresolvable prefix does — the core's
/// authoritative resolve post-read raises the unknown-name error.
fn probe_sniff_bytes<'a>(bytes: &'a [u8], format: Option<&str>) -> &'a [u8] {
    if name_decides_probe(format) {
        return &[];
    }
    &bytes[..bytes.len().min(SNIFF_PREFIX)]
}

/// Whether an explicit `format=` alone decides this probe: the name
/// resolves to a kind, and that kind is not Excel (the one kind whose
/// probe consults bytes — the OLE-container decision).
fn name_decides_probe(format: Option<&str>) -> bool {
    matches!(
        format.map(documents_impl::Kind::from_name),
        Some(Some(kind)) if kind != documents_impl::Kind::Excel
    )
}

/// `tors.documents.page_geometry(path, data=None, format=None,
/// password=None, backend=None, max_bytes=None)
/// -> list[PageGeometry | None]` (or `page_geometry(data=...)`): the
/// declared-box geometry probe — one entry per unit of the resolved
/// format, `None` where that unit declares no geometry.
///
/// The entry, per unit, in points: `width_pt`/`height_pt` (the render
/// extent — for a PDF page the visible box's extent with the axes
/// swapped for a 90/270 rotation, the turn already applied), `rotation`
/// (the PDF engine-normalized `/Rotate` ∈ {0, 90, 180, 270}, a
/// non-right-angle declaration reporting 0; office units always 0), and
/// `crop` (a PDF page's declared CropBox REDUCED TO THE MEDIABOX — the
/// effective crop, the intersection the engine's visible box stands on —
/// normalized corners, the unrotated source frame; provenance only,
/// never re-applied; `None` unless a CropBox is declared AND it actually
/// crops). Office entries always carry `rotation = 0`, `crop = None`.
///
/// The unit and length are a documented function of the resolved format:
/// pdf → one entry per page (the page count); docx → one per section
/// (the converter's section count, sizes read from the RAW document
/// layer — a section declaring only margins is `None`, never a
/// fabricated Letter); pptx → one per slide (the raw deck-wide slide
/// size, EMU ÷ 12700.0 in float, `None` everywhere when undeclared);
/// xlsx → one per WORKSHEET, all `None` (a worksheet is an unbounded
/// grid, not a page; a chartsheet or dialog tab is NOT a unit — the raw
/// layer's kind filter, the structure surface's sheet list's own rule:
/// the surfaces agree on the same workbook, pre-fix they counted apart);
/// legacy doc/xls/ppt → all `None` at the converter's
/// section count; html/csv/rtf/epub/odt/ods/odp → exactly one entry,
/// `None` (unit = whole document, single-flow — never `[]`). `[]` is
/// reachable exactly when the resolved unit count is zero (a zero-page
/// PDF, a zero-slide deck). A per-unit box read that fails on an index
/// the length rule reported is a `None` entry, its siblings intact.
///
/// `backend` names the engine lane: `"auto"` (the default) and `"oxide"`
/// are IDENTICAL on this surface for every format — the geometry lane is
/// a property of the format, not a choice (a documented divergence from
/// `Backend.AUTO`'s conversion contract). `"anydoc"` is refused on the
/// formats whose geometry anydoc structurally lacks (pdf and the office
/// families) — a format-aware capability `ValueError` raised ahead of
/// every other refusal; the no-geometry row runs no engine at all, so
/// every backend (anydoc included) gets the same `[None]` (another
/// documented divergence from conversion routing's unsupported-pair
/// errors: there is nothing to refuse and nothing is fallen back).
///
/// `format` names the format explicitly (extension spelling, no dot,
/// case-insensitive; `None` sniffs it from the content markers, with the
/// path's extension as the last resort) — the explicit name wins over
/// the sniff wins over the extension, `convert()`'s precedence, so a
/// `data=` call can name its format with no file to consult.
///
/// `password` unlocks an encrypted PDF (the PDF kinds only: a password
/// on any other format is a `ValueError`, never a silently-ignored
/// argument). Without one, an encrypted PDF fails closed (`ValueError`);
/// owner-password-only files authenticate with the empty password and
/// read normally. An encrypted OOXML package is a typed `ValueError`
/// from the office parse — never `None`/`[]`, never a password prompt.
///
/// `max_bytes` is the caller's input budget in bytes (a positive `int`):
/// an explicit budget binds pre-read (the `path=`'s stat, the `data=`
/// length) exactly as on every other entry point, and `None` keeps the
/// 32 MiB default ceiling as the OFFICE leg's post-read check (the
/// office parse amplifies its container into resident memory) — the pdf
/// row and the no-geometry row carry no default ceiling on this surface
/// (the pdf lane's own limits govern; no engine runs on the no-geometry
/// row), the read bounded only by the finite 512 MiB backstop. Zero is
/// refused as the nonsense it is — but CORE-side, sequenced (§6) after
/// the capability and password-kind refusals, never pre-empting them the
/// way the other surfaces' binding-side parse does.
///
/// With an explicit `format=` naming a no-geometry format
/// (html/csv/rtf/epub/odt/ods/odp), the answer is format-determined —
/// `[None]` — and the file is never read: the regular-file gates still
/// run (a missing path is the same `OSError`, a FIFO the same refusal),
/// but the bytes are consulted by nobody.
///
/// Errors: `OSError` for a missing/unreadable file; `ValueError` for
/// bytes that resolve to no format, an unknown `format=` name, a bad
/// backend name, a password on a non-PDF, an over-budget input, a
/// malformed/encrypted document, or an office container that will not
/// parse — the pinned §6 order: the format resolves first (the
/// capability refusal names it, so it cannot precede resolve), then the
/// capability refusal (under the GIL, before the detach), then the
/// password-kind refusal, then the max_bytes refusals (zero's included),
/// then the engine's own encrypted doors.
///
/// GIL model: the argument marshalling under the GIL, plus — on
/// `backend="anydoc"` only — a bounded 64 KiB prefix read that resolves
/// the format so the capability refusal can fire before the detach (the
/// sibling PDF probes' ordering, which this surface's format-aware
/// refusal needs a resolved format to keep); the whole
/// read+resolve+geometry pass under one `py.detach`, then the entry-list
/// marshalling (O(units): one small native object per `Some` entry).
/// ⚠ A malformed PDF can crash the process (uncatchable native stack
/// exhaustion in the pdf_oxide engine): sandbox untrusted PDFs
/// (docs/documents.md, SECURITY.md).
#[pyfunction(signature = (
    path = None,
    data = None,
    format = None,
    password = None,
    backend = None,
    max_bytes = None,
))]
pub fn page_geometry(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    format: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
    backend: Option<&Bound<'_, PyAny>>,
    max_bytes: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<Option<PageGeometry>>> {
    let source = parse_source(path, data, "page_geometry")?;
    let format = parse_format(format)?;
    let password = parse_password(password)?;
    let backend = parse_backend(backend)?;
    // The capability refusal fires HERE — under the GIL, before the
    // detach and before any byte of the document is read — exactly where
    // the four PDF-only siblings fire theirs. They can name their format
    // without resolving (PDF-only by construction); this surface's
    // refusal is format-aware, so the format resolves first (the helper
    // below). When even the prefix cannot resolve the format, the
    // refusal defers to the core's authoritative resolve — §6's
    // resolve-first clause: a capability refusal cannot name a format it
    // has not resolved.
    if backend == Backend::Anydoc
        && let Some(refusal) = probe_capability_refusal(&source, format, backend)
    {
        return Err(document_error(py, refusal)?);
    }
    // The zero value threads through (see `parse_max_bytes_on`): the
    // core's sequenced check is this surface's zero refusal.
    let max_bytes = parse_max_bytes_on(max_bytes, ZeroRefusal::CoreSide)?;
    let entries = match py.detach(move || {
        let (bytes, hint) = source
            .into_geometry_input(max_bytes, format, backend)
            .map_err(input_document_error)?;
        documents_impl::page_geometry(
            bytes,
            hint.as_deref(),
            format,
            backend,
            &documents_impl::ConvertOptions {
                password: password.as_deref(),
                max_bytes,
            },
        )
    }) {
        Ok(entries) => entries,
        Err(err) => return Err(document_error(py, err)?),
    };
    // O(units) of native-object construction, post-detach (the
    // PdfClassification precedent's residue class, one pyclass per entry).
    Ok(entries
        .into_iter()
        .map(|entry| entry.map(|geometry| PageGeometry { inner: geometry }))
        .collect())
}

/// One `page_geometry` entry: a unit's declared render box, in points —
/// the native pyclass the probe returns per page/section/slide/worksheet.
///
/// Deliberate deviation from the `PdfClassification` wrapper precedent:
/// this class is exposed DIRECTLY (no Python wrapper view around it).
/// The wrapper pattern exists to add typing the native layer lacks —
/// `PdfClassification` re-types verdict strings as `PageKind` members —
/// while a geometry entry carries nothing to re-type: the fields are
/// floats, an int, and a tuple-or-None, exactly as returned. §12 item 2
/// resolved for native + Rust-side `__eq__`: comparisons are structural
/// (`==` on two entries compares the four fields; lists of them compare
/// elementwise), so `page_geometry(path) == page_geometry(path,
/// backend="oxide")` is a plain list equality — the property the tests
/// pin.
#[pyclass(module = "_tors_documents")]
pub struct PageGeometry {
    inner: documents_impl::PageGeometry,
}

/// The layout classes' shared repr shapes: each class's own Python shape,
/// never the Rust Debug of the inner structs (`LayoutSpan { … }` is not a
/// Python shape; a repr is surface). `PageLayout`'s repr nests these, so
/// the shapes live once and every class prints the way its own `__repr__`
/// prints standalone.
fn page_geometry_repr(geometry: &documents_impl::PageGeometry) -> String {
    let crop = match geometry.crop {
        Some((x0, y0, x1, y1)) => format!("({x0:?}, {y0:?}, {x1:?}, {y1:?})"),
        None => "None".to_string(),
    };
    format!(
        "PageGeometry(width_pt={:?}, height_pt={:?}, rotation={}, crop={})",
        geometry.width_pt, geometry.height_pt, geometry.rotation, crop
    )
}

fn text_span_repr(span: &tors::pdf_impl::LayoutSpan) -> String {
    // the bools spell Python's `True`/`False`, not Rust's `false`
    let py_bool = |value: bool| if value { "True" } else { "False" };
    format!(
        "TextSpan(text={:?}, bbox={:?}, font_size={:?}, font_name={:?}, \
         is_bold={}, is_italic={}, rotation_degrees={:?}, is_artifact={})",
        span.text,
        span.bbox,
        span.font_size,
        span.font_name,
        py_bool(span.is_bold),
        py_bool(span.is_italic),
        span.rotation_degrees,
        py_bool(span.is_artifact)
    )
}

fn text_line_repr(line: &tors::pdf_impl::LayoutLine) -> String {
    // the words as (text, bbox) pairs — the Word getter's own element shape
    let words = line
        .words
        .iter()
        .map(|w| format!("({:?}, {:?})", w.text, w.bbox))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "TextLine(text={:?}, bbox={:?}, words=[{}])",
        line.text, line.bbox, words
    )
}

fn image_box_repr(image: &tors::pdf_impl::LayoutImage) -> String {
    format!(
        "ImageBox(bbox={:?}, width_px={}, height_px={}, rotation_degrees={:?}, paint_order={})",
        image.bbox, image.width_px, image.height_px, image.rotation_degrees, image.paint_order
    )
}

#[pymethods]
impl PageGeometry {
    /// The unit's render width in points (positive, finite; the axes
    /// swapped for a 90/270 PDF rotation).
    #[getter]
    fn width_pt(&self) -> f32 {
        self.inner.width_pt
    }

    /// The unit's render height in points (the same law).
    #[getter]
    fn height_pt(&self) -> f32 {
        self.inner.height_pt
    }

    /// The declared rotation, engine-normalized: {0, 90, 180, 270} on the
    /// PDF lane (a non-right-angle `/Rotate` reports 0), always 0 on the
    /// office lanes.
    #[getter]
    fn rotation(&self) -> i32 {
        self.inner.rotation
    }

    /// The PDF page's declared CropBox reduced to the MediaBox (the
    /// effective crop: the intersection the engine's visible box stands
    /// on), normalized corners, the unrotated source frame, points —
    /// `Some` only when a CropBox is declared AND it actually crops;
    /// provenance for the extent, never re-applied. `None` on every
    /// office lane.
    #[getter]
    fn crop(&self) -> Option<(f32, f32, f32, f32)> {
        self.inner.crop
    }

    /// Structural equality: two entries are equal exactly when their four
    /// fields are (an entry against any other type is `NotImplemented`,
    /// Python's own protocol).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        page_geometry_repr(&self.inner)
    }
}

/// `tors.documents.page_layout(path, data=None, format=None,
/// password=None, backend=None, pages=None, max_bytes=None)
/// -> list[PageLayout]` (or `page_layout(data=...)`): the layout probe —
/// per unit of the resolved format's geometry lane, the unit's geometry
/// plus, for PDF pages, the page's text spans, text lines, and image
/// boxes, all in the page's declared render space, from ONE open.
///
/// The coordinate space (every box-bearing class here states it):
/// "points, bottom-left origin of the page's render space; y is the
/// box's bottom edge; for text the bottom edge is the glyph baseline and
/// the box extends one font-size upward (em-box; descenders may extend
/// below). Boxes are unclipped; the page box is the consumer's clip."
/// The page's render space is the visible box's (the CropBox ∩ MediaBox
/// region a reader displays) with the page's /Rotate applied — the same
/// frame `page_geometry`'s width/height declare. One transform owns the
/// mapping (the engine's raw y-up boxes translated off the visible-box
/// origin, then rotated 0=identity/90/180/270; the engine's own
/// pre-mapped spans are mapped back to raw first and carried through the
/// same rule; image boxes are always raw and take the rule directly), so
/// no consumer ever re-implements it.
///
/// The entries: `page_index` (0-based, mandatory — the `pages=`
/// normalization reorders and this is what maps an entry back to its
/// page), `format` (the resolved format's vocabulary name — the
/// `to_markdown` spelling: the provenance marker that tells you why the
/// layout lists may be empty, the office row vs a blank/scanned PDF
/// page), `geometry` (the `page_geometry` entry for that unit — `None`
/// where the unit's box cannot be read or declares no positive extent,
/// in which case the layout lists are empty too: no frame, no boxes),
/// `spans` (runs: text, bbox, font_size, font_name, is_bold, is_italic,
/// rotation_degrees, is_artifact — the typed-BDC-only /Artifact flag, see
/// the caveat below), `lines` (text, bbox, words — each word
/// text + bbox; the per-line join law holds: the line's text is its
/// words' texts joined with single spaces, and every word's box is
/// inside the line's box), `images` (bbox, width_px, height_px,
/// rotation_degrees, paint_order — from the engine's image HANDLES: no
/// decode, no silent small-image filter, never a swallowed error; the
/// image never carries a pixel decode here).
///
/// Approximation caveat: a non-zero `rotation_degrees` means the box is
/// the engine's FLATTENED approximation and MUST NOT be used as a
/// highlight box — transposed or mis-anchored at every non-zero content
/// rotation, on ANY page (180° same-size/wrong-anchor with zero ink
/// overlap; ±90° transposed; on a /Rotate 90/270 page additionally mapped,
/// its line box degraded to a square). Carried honestly by
/// rotation_degrees, documented, not re-derived. Verified edge (T9,
/// probed): the flag does NOT cover vertical writing — a WMode-1 run
/// (tategaki; a Type0 font whose /Encoding CMap name ends in `-V`, e.g.
/// /Identity-V, or whose embedded CMap declares /WMode 1) carries
/// `rotation_degrees == 0.0` while its box is the flattened fiction at
/// its worst: measured (pdf_oxide 0.3.78, 24pt `<00410042>` under
/// /Identity-V): span bbox (300, 700, -48, 24) — the vertical advance
/// (-24 pt/glyph) flattened onto x as a NEGATIVE width, the glyphs' true
/// ink a column at (300, 652, 24, 72), zero overlap — and the line
/// surface re-lays the glyphs over the normalized box and reports the
/// REVERSED text ("BA"). The engine's span carries wmode == 1 but wmode
/// is cut from this surface, so a consumer cannot see it here; on a
/// /Rotate 90/270 page the run is not engine-premapped and the fiction
/// rides the forward map. A highlight box for text that may be vertical
/// writing cannot be validated by rotation_degrees alone.
///
/// Image-box caveat: image boxes are NEVER visibility-filtered (the
/// engine's 2 pt drop law is the TEXT walk's): an image wholly outside
/// the visible box is still returned, with out-of-range coordinates —
/// the consumer clips to the page box, and never reads "image present"
/// as "image visible".
///
/// The is_artifact law (typed-BDC-only): true only for a typed
/// `/Artifact <<…>> BDC` whose `/Type` or `/Subtype` the engine
/// recognizes (Pagination/Header/Footer/Watermark/PageNumber, Layout,
/// Page, Background); false for a bare `/Artifact BMC` (the engine's BMC
/// handler deliberately leaves the artifact type unset, and the span
/// carries no other artifact bit) and for unrecognized types. So
/// `is_artifact == false` is NOT evidence of body text: header/footer
/// detection must not rely on this flag alone.
///
/// The unit and length are a documented function of the resolved format —
/// the GEOMETRY capability table applies to this surface IDENTICALLY:
/// pdf → one entry per page (or the normalized pages= selection's), docx
/// → one per section, pptx → one per slide, xlsx → one per WORKSHEET
/// (a chartsheet is not a unit — the structure surface's own filter),
/// legacy doc/xls/ppt → one per converter section — every office row
/// carrying its declared geometry with EMPTY spans/lines/images (no
/// office engine exposes text/image boxes: PDF-only for layout), and
/// html/csv/rtf/epub/odt/ods/odp → exactly one entry, geometry None
/// (unit = whole document, single-flow). `pages=` is PDF-ONLY (a page
/// index cannot name a section): a non-None selection on any non-PDF row
/// is the conversion lane's own typed refusal, never a silent
/// whole-document answer; on the PDF lane it normalizes exactly as the
/// conversion pair's (sort + dedup into document order; an empty
/// selection and an out-of-range index raise the byte-pinned errors);
/// the output is parallel to the NORMALIZED selection.
///
/// `backend` names the engine lane: `"auto"` (the default) and `"oxide"`
/// are IDENTICAL on this surface for every format (a documented divergence
/// from `Backend.AUTO`'s conversion contract). `"anydoc"` is refused on
/// the formats whose geometry anydoc structurally lacks (pdf and the
/// office families) — the capability table's own refusal, byte-identical
/// to `page_geometry`'s, raised ahead of every other refusal; the
/// no-geometry row runs no engine at all, so every backend gets the same
/// single entry.
///
/// `format` names the format explicitly (extension spelling, no dot,
/// case-insensitive; `None` sniffs it from the content markers, with the
/// path's extension as the last resort) — the explicit name wins over
/// the sniff wins over the extension, `convert()`'s precedence.
///
/// `password` unlocks an encrypted PDF (the PDF kinds only: a password
/// on any other format is a `ValueError`, never a silently-ignored
/// argument). Without one, an encrypted PDF fails closed (`ValueError`);
/// owner-password-only files authenticate with the empty password and
/// read normally. An encrypted OOXML package is a typed `ValueError`
/// from the office parse — never `None`/`[]`, never a password prompt.
///
/// `max_bytes` is the caller's input budget in bytes (a positive `int`):
/// an explicit budget binds pre-read exactly as on every other entry
/// point, and `None` keeps the 32 MiB default ceiling as the OFFICE
/// leg's post-read check — the pdf row and the no-geometry row carry no
/// default ceiling on this surface (the read bounded only by the finite
/// 512 MiB backstop). Zero is refused as the nonsense it is — but
/// CORE-side, sequenced after the capability and password-kind refusals,
/// never pre-empting them. With an explicit `format=` naming a
/// no-geometry format, the answer is format-determined and the file is
/// never read (the regular-file gates still run).
///
/// Errors: `OSError` for a missing/unreadable file; `ValueError` for
/// bytes that resolve to no format, an unknown `format=` name, a bad
/// backend name, a `pages=` selection on a non-PDF (PDF-only, above), a
/// password on a non-PDF, an over-budget input, a
/// malformed/encrypted document, an office container that will not
/// parse, or a `pages=` selection that names no page or an out-of-range
/// index — the pinned §6 order: the format resolves first, then the
/// capability refusal (under the GIL, before the detach), then the
/// password-kind refusal, then the max_bytes refusals, then the
/// engine's own encrypted doors, then the `pages=` range errors against
/// the real page count (the range check needs `page_count`, which needs
/// the open — `pages=[999]` on a locked PDF answers the door error).
/// Every malformed `pages=`/`max_bytes` refusal
/// reports AFTER the capability refusal, never before it.
///
/// Deliberate deviation from the `PdfClassification` wrapper precedent,
/// the same one `page_geometry` states: these five classes are exposed
/// DIRECTLY (no Python wrapper view) — the fields are floats, ints,
/// bools, strs, and tuples, exactly as returned, nothing to re-type —
/// and comparison is structural (`==` compares all fields; lists of
/// them compare elementwise), so `page_layout(path) ==
/// page_layout(path, backend="oxide")` is a plain list equality.
///
/// GIL model: the argument marshalling (including `pages=`' parse and
/// normalization) under the GIL; on `backend="anydoc"` only, a bounded
/// 64 KiB prefix read that resolves the format so the capability refusal
/// can fire before the detach; the whole read+resolve+layout pass under
/// one `py.detach`; then the O(pages × boxes) nested pyobject
/// construction (the residue class §12 item 2 names: every span, line,
/// word, and image the consumer reads materializes its native object
/// post-detach, on access). ⚠ A malformed PDF can crash the process
/// (uncatchable native stack exhaustion in the pdf_oxide engine):
/// sandbox untrusted PDFs (docs/documents.md, SECURITY.md).
#[pyfunction(signature = (
    path = None,
    data = None,
    format = None,
    password = None,
    backend = None,
    pages = None,
    max_bytes = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn page_layout(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    format: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
    backend: Option<&Bound<'_, PyAny>>,
    pages: Option<&Bound<'_, PyAny>>,
    max_bytes: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<PageLayout>> {
    let source = parse_source(path, data, "page_layout")?;
    let format = parse_format(format)?;
    let password = parse_password(password)?;
    let backend = parse_backend(backend)?;
    // The capability refusal fires HERE — the geometry capability table's
    // own refusal, byte-identical to page_geometry's (one table, one
    // message), under the GIL, before the detach and before any byte of
    // the document is read.
    if backend == Backend::Anydoc
        && let Some(refusal) = probe_capability_refusal(&source, format, backend)
    {
        return Err(document_error(py, refusal)?);
    }
    // The zero value threads through (see `parse_max_bytes_on`): the
    // core's sequenced check is this surface's zero refusal.
    let max_bytes = parse_max_bytes_on(max_bytes, ZeroRefusal::CoreSide)?;
    // The pages= shape refusals (empty selection, negative, non-index
    // types) are the BINDING's, parsed under the GIL here — BEFORE the
    // detach and so before the encrypted door: A10's order is two-tier
    // (measured: pages=[] on a locked document answers the shape
    // refusal, not the encryption refusal). The core's range
    // validation (an index past the page count) runs post-open, after
    // the door. The pdf_impl docstring states the PDF leg alone.
    let pages = parse_pages(pages)?;
    let entries = match py.detach(move || {
        let (bytes, hint) = source
            .into_geometry_input(max_bytes, format, backend)
            .map_err(input_document_error)?;
        documents_impl::page_layout(
            bytes,
            hint.as_deref(),
            format,
            backend,
            pages,
            &documents_impl::ConvertOptions {
                password: password.as_deref(),
                max_bytes,
            },
        )
    }) {
        Ok(entries) => entries,
        Err(err) => return Err(document_error(py, err)?),
    };
    // O(pages) of pyclass construction here (one PageLayout per entry);
    // the O(pages × boxes) nested construction is the getters', on
    // access, post-detach — the residue class the GIL heartbeat cell
    // exercises through every field.
    Ok(entries
        .into_iter()
        .map(|inner| PageLayout { inner })
        .collect())
}

/// The pyclass for one unit's text run ([`PageLayout.spans`]): the
/// engine's span evidence, in the page's render space.
///
/// Exposed DIRECTLY (no Python wrapper view) — the deliberate deviation
/// from the `PdfClassification` wrapper precedent, `PageGeometry`'s
/// stated rationale: the fields are exactly as returned, nothing to
/// re-type; `==` is structural.
#[pyclass(module = "_tors_documents")]
pub struct TextSpan {
    inner: tors::pdf_impl::LayoutSpan,
}

#[pymethods]
impl TextSpan {
    /// The run's text.
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    /// The run's box, `(x, y, w, h)`: points, bottom-left origin of the
    /// page's render space; y is the box's bottom edge; for text the
    /// bottom edge is the glyph baseline and the box extends one
    /// font-size upward (em-box; descenders may extend below). Boxes are
    /// unclipped; the page box is the consumer's clip.
    #[getter]
    fn bbox(&self) -> (f32, f32, f32, f32) {
        self.inner.bbox
    }

    /// The run's font size in points (the em-box's height).
    #[getter]
    fn font_size(&self) -> f32 {
        self.inner.font_size
    }

    /// The run's font name.
    #[getter]
    fn font_name(&self) -> &str {
        &self.inner.font_name
    }

    /// Bold evidence.
    #[getter]
    fn is_bold(&self) -> bool {
        self.inner.is_bold
    }

    /// Italic evidence.
    #[getter]
    fn is_italic(&self) -> bool {
        self.inner.is_italic
    }

    /// The run's own content rotation in degrees — non-zero means the
    /// box is the engine's flattened approximation and MUST NOT be used
    /// as a highlight box (transposed or mis-anchored at every non-zero
    /// content rotation, on any page). Verified NOT to cover vertical
    /// writing (T9, probed): a WMode-1 run (tategaki) carries
    /// `rotation_degrees == 0` — the angle is the text matrix's, the
    /// writing mode plays no part in it — while its box is the flattened
    /// fiction at its worst (the vertical advance flattened onto x as a
    /// NEGATIVE width; measured: (300, 700, -48, 24) for a 24pt 2-glyph
    /// /Identity-V run whose ink is a column at (300, 652, 24, 72)). A
    /// highlight box for text that may be vertical writing cannot be
    /// validated by this flag alone.
    #[getter]
    fn rotation_degrees(&self) -> f32 {
        self.inner.rotation_degrees
    }

    /// Declared /Artifact marked content — TYPED-FORM-ONLY: true only
    /// for a typed `/Artifact <<…>> BDC` whose /Type or /Subtype the
    /// engine recognizes; false for a bare `/Artifact BMC` (the engine
    /// sets no artifact type for it) and unrecognized types. So
    /// `is_artifact == false` is NOT evidence of body text.
    #[getter]
    fn is_artifact(&self) -> bool {
        self.inner.is_artifact
    }

    /// Structural equality (all eight fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        text_span_repr(&self.inner)
    }
}

/// One word inside a [`TextLine`] (`PageLayout.lines[].words`): the
/// text and its box — the engine's own gap-clustered word box, never
/// re-derived downstream.
#[pyclass(module = "_tors_documents")]
pub struct Word {
    inner: tors::pdf_impl::LayoutWord,
}

#[pymethods]
impl Word {
    /// The word's text.
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    /// The word's box, `(x, y, w, h)`: points, bottom-left origin of the
    /// page's render space; y is the box's bottom edge; for text the
    /// bottom edge is the glyph baseline and the box extends one
    /// font-size upward (em-box; descenders may extend below). Boxes are
    /// unclipped; the page box is the consumer's clip.
    #[getter]
    fn bbox(&self) -> (f32, f32, f32, f32) {
        self.inner.bbox
    }

    /// Structural equality (both fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Word(text={:?}, bbox={:?})",
            self.inner.text, self.inner.bbox
        )
    }
}

/// One unit's text line ([`PageLayout.lines`]): the engine's own line —
/// its word clustering is the engine's (gap thresholds are never
/// re-implemented downstream). The join law: this line's text is its
/// words' texts joined with single spaces, and every word's box is
/// inside this line's box (both pinned in the tests).
#[pyclass(module = "_tors_documents")]
pub struct TextLine {
    inner: tors::pdf_impl::LayoutLine,
}

#[pymethods]
impl TextLine {
    /// The line's text (the engine's join of its words).
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    /// The line's box, `(x, y, w, h)`: points, bottom-left origin of the
    /// page's render space; y is the box's bottom edge; for text the
    /// bottom edge is the glyph baseline and the box extends one
    /// font-size upward (em-box; descenders may extend below). Boxes are
    /// unclipped; the page box is the consumer's clip.
    #[getter]
    fn bbox(&self) -> (f32, f32, f32, f32) {
        self.inner.bbox
    }

    /// The line's words, in reading order.
    #[getter]
    fn words(&self) -> Vec<Word> {
        self.inner
            .words
            .iter()
            .map(|w| Word { inner: w.clone() })
            .collect()
    }

    /// Structural equality (all three fields — the words included).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        text_line_repr(&self.inner)
    }
}

/// One unit's image box ([`PageLayout.images`]): an image draw's
/// placement — from the engine's image HANDLES (no decode, no silent
/// small-image filter, no swallowed errors; `extract_images` is
/// disqualified for an evidence surface).
#[pyclass(module = "_tors_documents")]
pub struct ImageBox {
    inner: tors::pdf_impl::LayoutImage,
}

#[pymethods]
impl ImageBox {
    /// The image's axis-aligned box, `(x, y, w, h)`: points, bottom-left
    /// origin of the page's render space; y is the box's bottom edge;
    /// for text the bottom edge is the glyph baseline and the box
    /// extends one font-size upward (em-box; descenders may extend
    /// below). Boxes are unclipped; the page box is the consumer's clip.
    /// Image boxes are never visibility-filtered (the engine's 2 pt drop
    /// law is the TEXT walk's): an image wholly outside the visible box
    /// is still returned, with out-of-range coordinates — the consumer
    /// clips to the page box, and never reads "image present" as "image
    /// visible".
    #[getter]
    fn bbox(&self) -> (f32, f32, f32, f32) {
        self.inner.bbox
    }

    /// The image's pixel width.
    #[getter]
    fn width_px(&self) -> u32 {
        self.inner.width_px
    }

    /// The image's pixel height.
    #[getter]
    fn height_px(&self) -> u32 {
        self.inner.height_px
    }

    /// The CTM's rotation angle in degrees, `atan2`-normalized to
    /// [0, 360) — not restricted to right angles (a sheared or
    /// arbitrarily-rotated CTM reports e.g. 45.0; a mirrored CTM 180.0):
    /// the bbox is always the axis-aligned bound of the transformed unit
    /// square.
    #[getter]
    fn rotation_degrees(&self) -> f32 {
        self.inner.rotation_degrees
    }

    /// Page-local identity/z-order: the image's index among the page's
    /// painted images, in content-stream order.
    #[getter]
    fn paint_order(&self) -> u32 {
        self.inner.paint_order
    }

    /// Structural equality (all five fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        image_box_repr(&self.inner)
    }
}

/// One `page_layout` entry: the unit's geometry plus its text spans,
/// lines, and image boxes in the page's render space.
///
/// Provenance rides ON the entry: `format` is the resolved format's
/// vocabulary name (the `to_markdown` spelling — `resolved_name`'s, the
/// Excel kind's container-aware `xls`/`xlsx` case included), and it —
/// not the entry's shape — tells you why `spans`/`lines`/`images` may
/// be empty: an office row carries declared geometry with no text
/// surfaces (no office engine exposes text/image boxes), while a blank,
/// scanned, or content-unparseable PDF page looks identical in shape on
/// the pdf lane. Every entry of one call carries the same name (the
/// format resolves once, before any entry exists).
///
/// Exposed DIRECTLY (no Python wrapper view) — the same deliberate
/// deviation from the `PdfClassification` wrapper precedent `PageGeometry`
/// states; `==` is structural.
#[pyclass(module = "_tors_documents")]
pub struct PageLayout {
    inner: tors::pdf_impl::PageLayoutData,
}

#[pymethods]
impl PageLayout {
    /// The 0-based page (or section/slide/worksheet index) this entry
    /// describes — mandatory on every entry, because the `pages=`
    /// normalization reorders.
    #[getter]
    fn page_index(&self) -> usize {
        self.inner.page_index
    }

    /// The resolved format's vocabulary name (``to_markdown``'s
    /// spelling): the entry's provenance marker. It — not the entry's
    /// shape — tells you why ``spans``/``lines``/``images`` may be
    /// empty: an office row carries declared geometry with no text
    /// surfaces; a blank or scanned PDF page looks identical.
    #[getter]
    fn format(&self) -> &str {
        &self.inner.format
    }

    /// The unit's declared render box (the `page_geometry` entry):
    /// `None` where the unit's box cannot be read or declares no
    /// positive extent — then the layout lists are empty too (no frame,
    /// no boxes).
    #[getter]
    fn geometry(&self) -> Option<PageGeometry> {
        self.inner.geometry.map(|box_| PageGeometry {
            inner: documents_impl::PageGeometry::from_page_box(box_),
        })
    }

    /// The unit's text spans, reading order (PDF pages only; office
    /// units carry empty lists — no office engine exposes text boxes;
    /// the entry's `format` field is the provenance that tells the
    /// office case from a blank/scanned PDF page).
    #[getter]
    fn spans(&self) -> Vec<TextSpan> {
        self.inner
            .spans
            .iter()
            .map(|s| TextSpan { inner: s.clone() })
            .collect()
    }

    /// The unit's text lines, reading order, words nested (PDF pages
    /// only; office units carry empty lists — the entry's `format`
    /// field is the provenance the class docstring points at).
    #[getter]
    fn lines(&self) -> Vec<TextLine> {
        self.inner
            .lines
            .iter()
            .map(|l| TextLine { inner: l.clone() })
            .collect()
    }

    /// The unit's image boxes, paint order (PDF pages only). Never
    /// visibility-filtered: an image wholly outside the visible box is
    /// still returned with out-of-range coordinates — the consumer clips
    /// to the page box, and never reads "image present" as "image
    /// visible".
    #[getter]
    fn images(&self) -> Vec<ImageBox> {
        self.inner
            .images
            .iter()
            .map(|i| ImageBox { inner: i.clone() })
            .collect()
    }

    /// Structural equality (all six fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        // each list formatted via its class's own Python shape — never the
        // Rust Debug of the inner structs (`LayoutSpan { … }` is not a
        // shape a Python consumer recognizes) — through the ONE capped
        // list renderer (finding 14: an uncapped repr printed every
        // element; a 4,288-char page's repr was 511,468 chars — the repr
        // of a container is a shape preview, not a serialization).
        format!(
            "PageLayout(page_index={}, format={:?}, geometry={}, spans={}, lines={}, images={})",
            self.inner.page_index,
            self.inner.format,
            match &self.inner.geometry {
                Some(box_) =>
                    page_geometry_repr(&documents_impl::PageGeometry::from_page_box(*box_,)),
                None => "None".to_string(),
            },
            structure_list_repr(&self.inner.spans, text_span_repr),
            structure_list_repr(&self.inner.lines, text_line_repr),
            structure_list_repr(&self.inner.images, image_box_repr),
        )
    }
}

/// `tors.documents.page_tables(path, data=None, format=None, password=None,
/// backend=None, pages=None, max_bytes=None) -> list[PageTables]` (or
/// `page_tables(data=...)`): the per-cell table-box probe — one entry per
/// selected page, the page's detected tables with per-cell text and boxes,
/// all in the page's declared render space. Builds on #203 (`page_layout`):
/// the SAME render space, the SAME transform, the SAME spine.
///
/// The boxes: every `Some` bbox rides the one forward rule
/// (`map_box_forward` — translate to the visible box, rotate, swap), the
/// same frame `page_geometry` declares and `page_layout` maps into:
/// points, the bottom-left origin of the page's declared render space,
/// y-up. Engine-premapped boxes (the span-union branches on a mapped
/// page) are inverse-mapped first, exactly as `page_layout` treats its
/// spans. Boxes are unclipped — cell boxes may lie outside the page box
/// (ruling lines are not crop-filtered; the TEXT side is, so a grid can
/// also lose enough cells to vanish) — the page box is the consumer's
/// clip. `Table.bbox` is the engine's own union (Some on every returned
/// table; not the table's ink extent — it includes empty grid cells); a
/// cell's box is the engine's box for that cell and its regime is
/// per-branch (intersection: ruling rectangles, always Some, empty cells
/// included; cluster/band/text: span unions, None when empty; a vertical
/// merge pads merged rows with None cells) — and it is GEOMETRY, not
/// ownership: an empty cell's box is declared grid geometry, not a claim
/// on the text inside it.
///
/// `tables_refused` (A9): TRUE iff the surface deliberately declined to
/// serve a page whose inputs indicate possible table content — the
/// frame-incoherent `/Rotate` classes or the table-primitive edge cap.
/// Reading rule: `tables_refused: true` with `geometry.rotation != 0` →
/// frame incoherence (`/Rotate` and the drawn content disagree — notably
/// /Rotate 180 with ruled tables, where the detector compares text in one
/// frame against rulings in another); `true` at rotation 0 → the edge
/// cap (a page with more table primitives than the markdown lane's own
/// 1500-edge bail-out; detection skipped). FALSE for no-tables pages,
/// for content-stream failures (unknowable — `tables: []` stays
/// indistinguishable from a no-tables page, like `page_layout`'s empty
/// lists), and on office rows (there `tables: []` is the per-format
/// capability statement — tables are PDF-only in v1 — and the entry's
/// `format` field is the marker, `tables_refused` false: not a per-page
/// content judgment).
///
/// The frame-coherence gate (A1): the detector's inputs live in two
/// frames — text spans inherit the engine's page-rotation mapping, ruling
/// paths never do — and three page classes are coherent (served exactly):
/// rotation 0; 90/270 with NO engine-mapped span (upright text); a
/// pathless page whose every span is engine-mapped (unmap → forward).
/// Every other combination — /Rotate 180 with rulings (a ruled table
/// vanishes upstream; a 180-symmetric one comes back 180°-SWAPPED, the
/// measured qualitative failure), 90/270 with sideways text under rulings,
/// mixed mapped/raw spans — refuses: `tables: []`, `tables_refused: true`,
/// geometry intact (fail-closed: the unlisted combinations refuse too).
///
/// Two documented divergences a consumer must read before joining
/// surfaces: the table set is the PUBLIC spatial detector's answer
/// (spatial-only, unconsolidated) — `to_markdown` renders a different,
/// private pipeline (structure-tree-first for tagged PDFs, fragment
/// consolidation, the RTL guard, the edge cap, the relaxed wide-table
/// retry), so the sets can differ on ANY PDF; and content inside a
/// detected table's area may appear in BOTH this surface and
/// `page_layout` (the latter serves `extract_spans`, which returns table
/// members) — neither de-dups the other, there is no identity join, and
/// cell bboxes must NOT be used to subtract from `page_layout`'s output.
///
/// The spine, capability table, refusal order, `backend=`/`format=`/
/// `password=`/`max_bytes=` contracts, the unit table, and the `pages=`
/// semantics are `page_layout`'s exactly (A8): same capability refusal
/// (byte-identical), `pages=` PDF-only and normalized with the same
/// byte-pinned error strings, output parallel to the NORMALIZED selection
/// with `page_index` on every entry, the office leg's 32 MiB ceiling
/// binding even though `tables` is always `[]` there, and the encrypted
/// door preceding the `pages=` range validation on the PDF lane (A10: an
/// encrypted document with a malformed selection answers the encryption).
/// A10's full order is two-tier (the binding parses the pages= SHAPE
/// under the GIL before the detach: an empty/negative/non-index selection
/// refuses BEFORE the door; the door precedes the core's RANGE
/// validation) — and its door-first order also means `max_bytes` is
/// inert on the PDF lane
/// except the zero-value refusal (pdf_oxide's own limits govern there).
///
/// GIL model: page_layout's shape — argument marshalling under the GIL,
/// the capability refusal (on `backend="anydoc"`) under the GIL before
/// the detach, the whole read+resolve+tables pass under one `py.detach`,
/// then the O(pages) entry construction (the nested tables are the
/// getters', on access, post-detach). ⚠ A malformed PDF can crash the
/// process (uncatchable native stack exhaustion in the pdf_oxide engine):
/// sandbox untrusted PDFs (docs/documents.md, SECURITY.md).
#[pyfunction(signature = (
    path = None,
    data = None,
    format = None,
    password = None,
    backend = None,
    pages = None,
    max_bytes = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn page_tables(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    format: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
    backend: Option<&Bound<'_, PyAny>>,
    pages: Option<&Bound<'_, PyAny>>,
    max_bytes: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<PageTables>> {
    let source = parse_source(path, data, "page_tables")?;
    let format = parse_format(format)?;
    let password = parse_password(password)?;
    let backend = parse_backend(backend)?;
    // The capability refusal fires HERE — the geometry capability table's
    // own refusal, byte-identical to page_geometry's/page_layout's (one
    // table, one message), under the GIL, before the detach and before
    // any byte of the document is read.
    if backend == Backend::Anydoc
        && let Some(refusal) = probe_capability_refusal(&source, format, backend)
    {
        return Err(document_error(py, refusal)?);
    }
    // The zero value threads through (see `parse_max_bytes_on`): the
    // core's sequenced check is this surface's zero refusal.
    let max_bytes = parse_max_bytes_on(max_bytes, ZeroRefusal::CoreSide)?;
    // The pages= shape refusals (empty selection, negative, non-index
    // types) are the BINDING's, parsed under the GIL here — BEFORE the
    // detach and so before the encrypted door: A10's order is two-tier
    // (measured: pages=[] on a locked document answers the shape
    // refusal, not the encryption refusal). The core's range
    // validation (an index past the page count) runs post-open, after
    // the door. The pdf_impl docstring states the PDF leg alone.
    let pages = parse_pages(pages)?;
    let entries = match py.detach(move || {
        let (bytes, hint) = source
            .into_geometry_input(max_bytes, format, backend)
            .map_err(input_document_error)?;
        documents_impl::page_tables(
            bytes,
            hint.as_deref(),
            format,
            backend,
            pages,
            &documents_impl::ConvertOptions {
                password: password.as_deref(),
                max_bytes,
            },
        )
    }) {
        Ok(entries) => entries,
        Err(err) => return Err(document_error(py, err)?),
    };
    // O(pages) of pyclass construction here (one PageTables per entry);
    // the O(pages × tables × cells) nested construction is the getters',
    // on access, post-detach — the residue class the GIL heartbeat cell
    // exercises through every field.
    Ok(entries
        .into_iter()
        .map(|inner| PageTables { inner })
        .collect())
}

/// The tables classes' shared repr shapes: each class's own Python shape,
/// never the Rust Debug of the inner structs (page_layout's repr law).
fn tables_box_repr(bbox: Option<(f32, f32, f32, f32)>) -> String {
    match bbox {
        Some(box_) => format!("{box_:?}"),
        None => "None".to_string(),
    }
}

fn tables_cell_repr(cell: &tors::pdf_impl::Cell) -> String {
    format!(
        "Cell(text={:?}, bbox={}, colspan={}, rowspan={})",
        cell.text,
        tables_box_repr(cell.bbox),
        cell.colspan,
        cell.rowspan
    )
}

fn tables_row_repr(row: &tors::pdf_impl::Row) -> String {
    // the cells as their own class's repr — the Row getter's element shape
    let cells = row
        .cells
        .iter()
        .map(tables_cell_repr)
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Row(cells=[{}], is_header={})",
        cells,
        if row.is_header { "True" } else { "False" }
    )
}

fn tables_table_repr(table: &tors::pdf_impl::TableEntry) -> String {
    let rows = table
        .rows
        .iter()
        .map(tables_row_repr)
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Table(rows=[{}], has_header={}, col_count={}, bbox={})",
        rows,
        if table.has_header { "True" } else { "False" },
        table.col_count,
        tables_box_repr(table.bbox)
    )
}

/// One cell in a [`Table`]
/// (`PageTables.tables[].rows[].cells[]`): the cell's text, its box, and
/// the merge spans.
///
/// Exposed DIRECTLY (no Python wrapper view) — the `PageGeometry`
/// deviation note applies to all four tables classes; `==` is structural.
#[pyclass(module = "_tors_documents")]
pub struct Cell {
    inner: tors::pdf_impl::Cell,
}

#[pymethods]
impl Cell {
    /// The cell's text (verbatim; an empty cell is `""`).
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    /// The cell's box, `(x, y, w, h)`: points, bottom-left origin of the
    /// page's render space, y-up. Its regime is the engine's, per branch
    /// (never re-normalized here): intersection-branch cells are RULING
    /// rectangles (always Some, empty cells included); cluster/band/text
    /// cells are span unions (None when empty); a vertical merge pads
    /// merged rows with None cells inside an otherwise ruling-rect table.
    /// GEOMETRY, not ownership: an empty cell's box is declared grid
    /// geometry, not a claim on the text inside it — do not subtract cell
    /// boxes from other surfaces' text. Boxes are unclipped — a cell box
    /// may lie outside the page box (ruling lines are not crop-filtered;
    /// the TEXT side is, so a grid can lose enough cells to vanish) —
    /// the page box is the consumer's clip, and "cell box present" is
    /// never "cell is visible". Coordinates are single-precision f32;
    /// expect ~1e-4 pt quantization.
    #[getter]
    fn bbox(&self) -> Option<(f32, f32, f32, f32)> {
        self.inner.bbox
    }

    /// The column span (the engine's; 1 unless a merge covers this cell).
    #[getter]
    fn colspan(&self) -> u32 {
        self.inner.colspan
    }

    /// The row span (the engine's; 1 unless a merge covers this cell).
    #[getter]
    fn rowspan(&self) -> u32 {
        self.inner.rowspan
    }

    /// Structural equality (all four fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        tables_cell_repr(&self.inner)
    }
}

/// One row in a [`Table`]: the row's cells and the engine's own header
/// verdict.
#[pyclass(module = "_tors_documents")]
pub struct Row {
    inner: tors::pdf_impl::Row,
}

#[pymethods]
impl Row {
    /// The row's cells, grid order. `len` may be LESS than the table's
    /// `col_count` when a colspan/rowspan merge covers cells (the
    /// cluster-branch merge): lay cells out by walking `colspan`, never
    /// by `cells[i]`-as-column-i.
    #[getter]
    fn cells(&self) -> Vec<Cell> {
        self.inner
            .cells
            .iter()
            .map(|c| Cell { inner: c.clone() })
            .collect()
    }

    /// Whether the engine judged this row a header row (boldness/
    /// font-size heuristics; ALWAYS false on the intersection branch —
    /// the pinned engine limitation).
    #[getter]
    fn is_header(&self) -> bool {
        self.inner.is_header
    }

    /// Structural equality (both fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        tables_row_repr(&self.inner)
    }
}

/// One detected table in a [`PageTables`] entry: the engine's grid.
#[pyclass(module = "_tors_documents")]
pub struct Table {
    inner: tors::pdf_impl::TableEntry,
}

#[pymethods]
impl Table {
    /// The grid's rows, the engine's order (branch-deterministic, never
    /// re-sorted here: a vertical merge emits its merged groups
    /// bottom-group-first, the cluster path top-first).
    #[getter]
    fn rows(&self) -> Vec<Row> {
        self.inner
            .rows
            .iter()
            .map(|r| Row { inner: r.clone() })
            .collect()
    }

    /// Whether the engine found a header section (boldness/font-size
    /// heuristics; ALWAYS false on the intersection branch — the pinned
    /// engine limitation).
    #[getter]
    fn has_header(&self) -> bool {
        self.inner.has_header
    }

    /// The engine's grid column count. Rows may be shorter under merges
    /// (see `Row.cells`): a consumer lays cells out by walking
    /// `colspan`, never by `cells[i]`-as-column-i.
    #[getter]
    fn col_count(&self) -> usize {
        self.inner.col_count
    }

    /// The table's box, `(x, y, w, h)`: points, bottom-left origin of the
    /// page's render space, y-up — the ENGINE's own union (cell boxes on
    /// the intersection branch, member spans on the other branches, the
    /// cluster grid's line-extent × row-extent, the two prior boxes after
    /// a vertical merge), never re-derived from the cells here. Some on
    /// every table the engine returns; NOT the table's ink extent — it
    /// includes empty grid cells. The box is unclipped: it can lie
    /// outside the page box, and the page box is the consumer's clip.
    /// Coordinates are single-precision f32; expect ~1e-4 pt quantization.
    #[getter]
    fn bbox(&self) -> Option<(f32, f32, f32, f32)> {
        self.inner.bbox
    }

    /// Structural equality (all four fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        tables_table_repr(&self.inner)
    }
}

/// One `page_tables` entry: the page's geometry plus its detected tables,
/// per-cell boxes in the page's render space.
///
/// Provenance rides ON the entry (page_layout's law): `format` is the
/// resolved format's vocabulary name, and it — not the entry's shape —
/// tells you why `tables` may be empty. On an OFFICE row `tables: []` is
/// the per-format capability statement (tables are PDF-only in v1: the
/// office engines' tables carry no coordinates), so `format` marks
/// "unsupported here", not "no tables", and `tables_refused` is FALSE
/// there (not a per-page content judgment). On the pdf lane
/// `tables: []` with `tables_refused: false` means "no tables / content
/// unreadable" (deliberately indistinguishable); `tables_refused: true`
/// means THIS SURFACE declined the page — see the getter.
///
/// Exposed DIRECTLY (the `PageGeometry` deviation note applies); `==` is
/// structural (the five fields).
#[pyclass(module = "_tors_documents")]
pub struct PageTables {
    inner: tors::pdf_impl::PageTablesData,
}

#[pymethods]
impl PageTables {
    /// The 0-based page this entry describes — mandatory on every entry,
    /// because the `pages=` normalization reorders.
    #[getter]
    fn page_index(&self) -> usize {
        self.inner.page_index
    }

    /// The resolved format's vocabulary name (`to_markdown`'s spelling):
    /// the entry's provenance marker — on an office row it is what marks
    /// `tables: []` as "unsupported here" rather than "no tables".
    #[getter]
    fn format(&self) -> &str {
        &self.inner.format
    }

    /// The page's declared render box (the `page_geometry` entry): `None`
    /// where the page's box cannot be read or declares no positive extent
    /// — then `tables` is empty too (no frame to map into). THE READING
    /// RULE for refusals: `tables_refused` with `geometry.rotation != 0`
    /// is frame incoherence (`/Rotate` vs the drawn content); `true` at
    /// rotation 0 is the edge cap.
    #[getter]
    fn geometry(&self) -> Option<PageGeometry> {
        self.inner.geometry.map(|box_| PageGeometry {
            inner: documents_impl::PageGeometry::from_page_box(box_),
        })
    }

    /// The page's detected tables (PDF pages only; office units carry
    /// empty lists — tables are PDF-only in v1, the entry's `format`
    /// field is the marker; a no-tables or content-failed PDF page is
    /// the same empty shape with `tables_refused` false).
    #[getter]
    fn tables(&self) -> Vec<Table> {
        self.inner
            .tables
            .iter()
            .map(|t| Table { inner: t.clone() })
            .collect()
    }

    /// TRUE iff this surface deliberately declined to serve a page whose
    /// inputs indicate possible table content: the frame-incoherent
    /// `/Rotate` classes (the detector's text and ruling inputs disagree
    /// about the frame there — notably /Rotate 180 with ruled tables) or
    /// the table-primitive edge cap (a page with more than the markdown
    /// lane's own 1500-primitive bail-out). FALSE for no-tables pages,
    /// for content-stream failures (unknowable — `tables: []` stays
    /// indistinguishable from a no-tables page by design), and on office
    /// rows (their marker is `format`). Reading rule: `true` +
    /// `geometry.rotation != 0` → frame incoherence; `true` at rotation
    /// 0 → the edge cap.
    #[getter]
    fn tables_refused(&self) -> bool {
        self.inner.tables_refused
    }

    /// Structural equality (all five fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        // each list formatted via its class's own Python shape — never
        // the Rust Debug of the inner structs (page_layout's repr law)
        let tables = self
            .inner
            .tables
            .iter()
            .map(tables_table_repr)
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "PageTables(page_index={}, format={:?}, geometry={}, tables=[{}], tables_refused={})",
            self.inner.page_index,
            self.inner.format,
            match &self.inner.geometry {
                Some(box_) =>
                    page_geometry_repr(&documents_impl::PageGeometry::from_page_box(*box_,)),
                None => "None".to_string(),
            },
            tables,
            if self.inner.tables_refused {
                "True"
            } else {
                "False"
            }
        )
    }
}

/// `tors.documents.page_images(path=None, data=None, format=None,
/// password=None, backend=None, pages=None, max_bytes=None, decode=None,
/// max_image_bytes=None) -> list[PageImagesData]`: the image-bytes probe
/// (issue #205) — per unit of the resolved format's geometry lane, the
/// unit's geometry plus — PDF pages only — the page's image handles
/// (the unfiltered Phase-1 walk: no 8×8 gate, no max-pixels filter,
/// paint order; NEVER `extract_images`, which silently drops sub-8×8
/// images and swallows per-image decode errors) with each image's
/// evidence, and — only under `decode` — the bytes behind the handles.
///
/// The entry (per image): `bbox` (the mapped box, the same
/// `map_box_forward` call `page_layout` gives its image boxes — image
/// handles are always raw user space; `None` when the page's box cannot
/// be read: [R11], the bytes must not drop for a frame reason),
/// `width_px`/`height_px` (the DICTIONARY's), `rotation_degrees` (the
/// CTM angle, evidence not an instruction), `paint_order` (the
/// page-local identity — the same order `page_layout`'s `images` list),
/// `color_space` + `color_space_declared` + `indexed_base`,
/// `bits_per_component` (the DICTIONARY's — R22), `byte_size_compressed`,
/// `filters` (spelled, outermost first, `Other(k)` keeping its raw name
/// — R20), `is_inline`, and `data` — the ALWAYS-PRESENT four-arm sum
/// (R6/F9): `kind == "not_requested"` (no decode asked, or a
/// containers-only selection left this image out — `filters` say which)
/// | `"container"` (`bytes` + `container`: `"jpeg" | "jp2" |
/// "j2k_codestream"`) | `"raw"` (decoded samples that describe
/// themselves: `pixels`, the payload's own `width_px`/`height_px`/
/// `components`/`bpc`, the display tag `pixel_format`,
/// `samples_are_raw`, `decode_folded_in`, `icc`, `ccitt`) | `"error"`
/// (`error_kind`: `"decode" | "unsupported" | "size_cap"`, `message`,
/// and `scope`/`limit_bytes`/`needed_bytes` for a size cap). Every arm
/// getter is None unless `kind` names it (F9) — "no bytes" is never two
/// different things.
///
/// `decode` (R3 as amended): `None` = metadata only, decodes nothing;
/// `"container"` = decode, containers only — a DCT filter anywhere in
/// the chain serves its container bytes via the engine's decode()
/// passthrough (DECRYPTION INCLUDED — F1/R1: raw stream bytes are
/// ciphertext on an unlocked encrypted PDF, and `raw_compressed_bytes()`
/// never feeds this surface), a bare `/JPXDecode` stream serves its
/// container sniffed from the stream's own bytes (the 12-byte JP2
/// signature box → `"jp2"`, the SOC marker → `"j2k_codestream"`; the
/// shipped build cannot decode JPX at all (R2, feature-gated) so the
/// container is the only JPX content this surface can ever serve) —
/// served ONLY when the image is inline or the document is unencrypted,
/// else the typed `"unsupported"` error naming the cause (F1); anything
/// else stays `not_requested`. `"pixels"` = decode every image to the
/// engine's own classification: Raw — or the typed errors; on the
/// shipped build a JBIG2/JPX decode is the engine's `Unsupported`
/// verbatim (R2, feature-gated), a `rendering`/`jpeg2000` build decodes
/// them (build-dependent, documented). Anything else is the pinned
/// `ValueError`.
///
/// Caps (R7): 256 MiB per image (the chain-aware pre-decode estimate —
/// the container's own size for a DCT chain, the packed bits for CCITT,
/// the unpacked one-byte-per-component arithmetic otherwise, the base's
/// components for Indexed), 512 MiB per call cumulative (estimates
/// checked before each decode, actuals accumulated after), the over-cap
/// image a typed `size_cap` entry (`scope: "image" | "call"`,
/// `limit_bytes`, `needed_bytes`) — refused, NEVER truncated (a
/// truncated image is a wrong image with no marker). `max_image_bytes=`
/// overrides the per-image ceiling; raising it past the ENGINE's
/// env-read `PDF_OXIDE_MAX_DECOMPRESS_MB` (the flate/unpack/Indexed
/// ceilings it governs) cannot make the engine decode more — the
/// over-ceiling refusal just moves into the engine's own error (F8).
///
/// Honesty laws (R8), stated where the engine leaves them: an image
/// XObject with a missing or non-positive /Width·/Height yields NO
/// handle at all (measured: 0 handles — this surface cannot see it); a
/// Form-XObject subtree's error is dropped by the engine's own
/// `if let Ok`; a missing or unparseable /ColorSpace reports
/// `DeviceRGB` with `color_space_declared: false`. A caller who needs
/// certainty about the paint count must compare `len(images)` with the
/// render's — no completeness is provable beyond the three gaps above.
/// A decode failure is a VISIBLE TYPED ENTRY (`data.kind == "error"`),
/// siblings intact; a content-stream failure is `images: []` with the
/// geometry intact (the engine's empty-on-error law — a scanned page is
/// the same shape); a DOCX's images are real and are NOT served (the
/// office row: `images: []` with `format` provenance — R24); the
/// no-geometry formats answer one `geometry=None` entry.
///
/// The identity key (page_index, paint_order) pairs with `page_layout`'s
/// `images` list — the two surfaces are separate opens over the bytes
/// (nothing cached across them, R12): a changed file between the calls
/// breaks the pairing. Bytes decode EVERY selected image of every
/// selected page — there is no per-image selection in v1 (R12).
///
/// `format`/`password`/`backend`/`max_bytes`/`pages=` and the refusal
/// order are `page_layout`'s exactly (the capability refusal under the
/// GIL first, then the password-kind and max_bytes refusals, then the
/// engine doors, then the `pages=` range against the real count).
///
/// GIL model: page_tables' shape — argument marshalling under the GIL,
/// the capability refusal (on `backend="anydoc"`) under the GIL before
/// the detach, the whole read+resolve+images pass under one
/// `py.detach`, then the O(pages) entry construction (the nested images
/// are the getters', on access, post-detach). ⚠ A malformed PDF can
/// crash the process (uncatchable native stack exhaustion in the
/// pdf_oxide engine): sandbox untrusted PDFs (docs/documents.md,
/// SECURITY.md).
#[pyfunction(signature = (
    path = None,
    data = None,
    format = None,
    password = None,
    backend = None,
    pages = None,
    max_bytes = None,
    decode = None,
    max_image_bytes = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn page_images(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    format: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
    backend: Option<&Bound<'_, PyAny>>,
    pages: Option<&Bound<'_, PyAny>>,
    max_bytes: Option<&Bound<'_, PyAny>>,
    decode: Option<&Bound<'_, PyAny>>,
    max_image_bytes: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<PageImagesData>> {
    let source = parse_source(path, data, "page_images")?;
    let format = parse_format(format)?;
    let password = parse_password(password)?;
    let backend = parse_backend(backend)?;
    // The capability refusal fires HERE — the geometry capability table's
    // own refusal, byte-identical to page_geometry's/page_layout's (one
    // table, one message), under the GIL, before the detach and before
    // any byte of the document is read.
    if backend == Backend::Anydoc
        && let Some(refusal) = probe_capability_refusal(&source, format, backend)
    {
        return Err(document_error(py, refusal)?);
    }
    let max_bytes = parse_max_bytes_on(max_bytes, ZeroRefusal::CoreSide)?;
    let decode = parse_image_decode(decode)?;
    let max_image_bytes = parse_max_image_bytes(max_image_bytes)?;
    // The pages= shape refusals (empty selection, negative, non-index
    // types) are the BINDING's, parsed under the GIL here — BEFORE the
    // detach and so before the encrypted door: A10's order is two-tier
    // (measured: pages=[] on a locked document answers the shape
    // refusal, not the encryption refusal). The core's range
    // validation (an index past the page count) runs post-open, after
    // the door. The pdf_impl docstring states the PDF leg alone.
    let pages = parse_pages(pages)?;
    let entries = match py.detach(move || {
        let (bytes, hint) = source
            .into_geometry_input(max_bytes, format, backend)
            .map_err(input_document_error)?;
        documents_impl::page_images(
            bytes,
            hint.as_deref(),
            format,
            backend,
            pages,
            &documents_impl::ConvertOptions {
                password: password.as_deref(),
                max_bytes,
            },
            decode,
            max_image_bytes,
        )
    }) {
        Ok(entries) => entries,
        Err(err) => return Err(document_error(py, err)?),
    };
    // O(pages) of pyclass construction here; the O(pages × images) nested
    // construction is the getters', on access, post-detach.
    Ok(entries
        .into_iter()
        .map(|inner| PageImagesData { inner })
        .collect())
}

/// The `decode=` argument's parse (R3): `None` → the handles-only mode;
/// `"container"` / `"pixels"` the two decode modes. A non-str is a
/// `TypeError` and a non-vocabulary str a `ValueError`, both naming the
/// argument and repr'ing the value (parse_backend's convention).
fn parse_image_decode(decode: Option<&Bound<'_, PyAny>>) -> PyResult<tors::pdf_impl::ImageDecode> {
    use tors::pdf_impl::ImageDecode;
    let Some(decode) = decode else {
        return Ok(ImageDecode::None);
    };
    let name = decode
        .cast::<PyString>()
        .map_err(|_| {
            PyTypeError::new_err(format!(
                "decode must be one of ('container', 'pixels'), not {}",
                py_repr(decode)
            ))
        })?
        .to_str()
        .map_err(|_| {
            PyValueError::new_err(format!(
                "decode must be valid Unicode (no lone surrogates), not {}",
                py_repr(decode)
            ))
        })?;
    match name {
        "container" => Ok(ImageDecode::Container),
        "pixels" => Ok(ImageDecode::Pixels),
        other => Err(PyValueError::new_err(format!(
            "decode must be one of ('container', 'pixels'), not {other:?}"
        ))),
    }
}

/// The `max_image_bytes=` argument's parse (the per-image decode cap's
/// override, R7): a positive int, `parse_max_bytes_on`'s type-vs-value
/// convention, `ZeroRefusal::Binding` (a zero cap could never serve an
/// image; refused at the parse, naming the argument).
fn parse_max_image_bytes(max_image_bytes: Option<&Bound<'_, PyAny>>) -> PyResult<Option<u64>> {
    let Some(max_image_bytes) = max_image_bytes else {
        return Ok(None);
    };
    if max_image_bytes.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "max_image_bytes must be an int (the per-image decode ceiling in bytes), not a bool",
        ));
    }
    let Ok(limit) = max_image_bytes.extract::<i64>() else {
        if max_image_bytes.cast::<PyInt>().is_ok() {
            return Err(PyValueError::new_err(format!(
                "max_image_bytes {} is too large",
                py_repr(max_image_bytes)
            )));
        }
        return Err(PyTypeError::new_err(format!(
            "max_image_bytes must be an int (the per-image decode ceiling in bytes), not {}",
            py_repr(max_image_bytes)
        )));
    };
    if limit <= 0 {
        return Err(PyValueError::new_err(format!(
            "max_image_bytes must be positive (the per-image decode ceiling in bytes), not {limit}"
        )));
    }
    Ok(Some(limit as u64))
}

/// `tors.documents.page_text_chars(path=None, data=None, format=None,
/// password=None, backend=None, pages=None, max_bytes=None) ->
/// list[PageCharsData]`: the char-boxes probe (issue #206) — per unit of
/// the resolved format's geometry lane, the unit's geometry plus — PDF
/// pages only — the page's per-character boxes from the CACHED SPAN PARSE
/// ONLY (`extract_spans` → `TextSpan::to_chars()` per span; never
/// `extract_chars` as the source — it is raw-only and includes content
/// the span path drops at the crop; R13: the span parse already runs the
/// char extractor under the stamp, so this is the consistency choice, not
/// a parse-independence one).
///
/// The char (per entry, a flat list in reading order — span order ×
/// in-span order): `char` (one Unicode scalar — Python string indices,
/// code points), `bbox` — `(x, y, w, h)`, points, bottom-left origin of
/// the page's render space, **y is the BASELINE** (the em-box law: the
/// box extends one font-size upward; descenders hang BELOW the box's
/// bottom edge — the same convention as the span surface), `span_index`
/// (into `page_layout`'s `spans` for the same page — the join law:
/// `"".join(c.char for c in chars if c.span_index == s) ==
/// spans[s].text` EXACTLY, pinned across two opens), `index_in_span`
/// (code-point indexing: chars `[i, j)` of span s ground the substring
/// `spans[s].text[i:j]`; the whole-span box is the SPAN's own mapped box,
/// never the char union), `wmode` (0 horizontal, 1 vertical — the
/// WMode-1 honesty field: a vertical run's char boxes are the engine's
/// flattened fiction, displaced one advance LEFT of their origins and
/// stacked along x, measured — a consumer may not pretend they are
/// highlight-able), and `x_measured` (`true` = the engine's measured
/// origins — drift ≈ the 0.5 pt tolerance + sub-point interpolation;
/// `false` = the prefix-sum/uniform arm — the TJ-kerning drift class,
/// bounded only by measurement: the engine folds Tc/Tw/Tz/TJ into the
/// widths, so measured drift on every probe fixture is 0.000, and the
/// analytic bound is `|Σ TJ adjustments before i| × font_size/1000` pt;
/// a highlight box built from `x_measured: false` chars is APPROXIMATE).
/// The flags are SPAN-CONSTANT (pinned); `x_measured` mirrors to_chars'
/// OWN gate (R9), never `!offsets.is_empty()`.
///
/// The R4 mirror fix: on a page whose rotation the engine applied to a
/// span, the pass derives the chars from the span's UNMAPPED box and maps
/// each char box forward — the mapped frame's glyph order runs
/// right-to-left while to_chars prefix-sums left-to-right, so the
/// unmapped derivation is the only one that lands the boxes where the
/// render draws the ink (measured pre-fix: ~35 pt displaced, chars
/// overlapping).
///
/// Failure semantics: a page whose box read fails is `geometry: None` +
/// `chars: []` (no frame to map into — page_layout's law); a content
/// parse failure is `chars: []`, geometry intact (the engine's
/// empty-on-error law); office rows carry `chars: []` with `format`
/// provenance (R24); the `pages=`/`backend=`/`password=`/`max_bytes=`
/// contracts and the refusal order are `page_layout`'s exactly.
///
/// GIL model: page_tables' shape (the O(pages × chars) nested
/// construction is the getters', on access, post-detach). ⚠ A malformed
/// PDF can crash the process (uncatchable native stack exhaustion in the
/// pdf_oxide engine): sandbox untrusted PDFs (docs/documents.md,
/// SECURITY.md).
#[pyfunction(signature = (
    path = None,
    data = None,
    format = None,
    password = None,
    backend = None,
    pages = None,
    max_bytes = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn page_text_chars(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    format: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
    backend: Option<&Bound<'_, PyAny>>,
    pages: Option<&Bound<'_, PyAny>>,
    max_bytes: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<PageCharsData>> {
    let source = parse_source(path, data, "page_text_chars")?;
    let format = parse_format(format)?;
    let password = parse_password(password)?;
    let backend = parse_backend(backend)?;
    // The capability refusal fires HERE — byte-identical to the other
    // geometry surfaces' (one table, one message).
    if backend == Backend::Anydoc
        && let Some(refusal) = probe_capability_refusal(&source, format, backend)
    {
        return Err(document_error(py, refusal)?);
    }
    let max_bytes = parse_max_bytes_on(max_bytes, ZeroRefusal::CoreSide)?;
    // The pages= shape refusals (empty selection, negative, non-index
    // types) are the BINDING's, parsed under the GIL here — BEFORE the
    // detach and so before the encrypted door: A10's order is two-tier
    // (measured: pages=[] on a locked document answers the shape
    // refusal, not the encryption refusal). The core's range
    // validation (an index past the page count) runs post-open, after
    // the door. The pdf_impl docstring states the PDF leg alone.
    let pages = parse_pages(pages)?;
    let entries = match py.detach(move || {
        let (bytes, hint) = source
            .into_geometry_input(max_bytes, format, backend)
            .map_err(input_document_error)?;
        documents_impl::page_text_chars(
            bytes,
            hint.as_deref(),
            format,
            backend,
            pages,
            &documents_impl::ConvertOptions {
                password: password.as_deref(),
                max_bytes,
            },
        )
    }) {
        Ok(entries) => entries,
        Err(err) => return Err(document_error(py, err)?),
    };
    Ok(entries
        .into_iter()
        .map(|inner| PageCharsData { inner })
        .collect())
}

/// One `page_images` entry: the page's image metadata, and — only where
/// `decode` asked and the decode succeeded — the bytes.
///
/// Exposed DIRECTLY (the `PageGeometry` deviation note applies); `==` is
/// structural (the four fields).
#[pyclass(module = "_tors_documents")]
pub struct PageImagesData {
    inner: tors::pdf_impl::PageImagesData,
}

#[pymethods]
impl PageImagesData {
    /// The 0-based page this entry describes — mandatory on every entry,
    /// because the `pages=` normalization reorders.
    #[getter]
    fn page_index(&self) -> usize {
        self.inner.page_index
    }

    /// The resolved format's vocabulary name (`to_markdown`'s spelling):
    /// the entry's provenance marker — on an office row it is what marks
    /// `images: []` as "not served here" (a DOCX's images are real)
    /// rather than "no images".
    #[getter]
    fn format(&self) -> &str {
        &self.inner.format
    }

    /// The page's declared render box (the `page_geometry` entry): `None`
    /// where the page's box cannot be read or declares no positive
    /// extent — the images are STILL enumerated ([R11]: the bytes must
    /// not drop for a frame reason), their `bbox` None.
    #[getter]
    fn geometry(&self) -> Option<PageGeometry> {
        self.inner.geometry.map(|box_| PageGeometry {
            inner: documents_impl::PageGeometry::from_page_box(box_),
        })
    }

    /// The page's images, paint order — the same order `page_layout`'s
    /// `images` lists (same enumeration, same transform). Each access
    /// materializes a fresh list of fresh entries (a clone per element —
    /// the `geometry` getter's fresh-object law, the deliberate shape of
    /// these pyclasses: thin getters over plain core structs, no shared
    /// mutable state, `==` stays structural). The bytes the entries
    /// carry are COPIED, not aliased: on `decode=` payloads that is
    /// O(decode-output) transient per access, by construction — the
    /// caller holding the entry list owns its bytes outright.
    #[getter]
    fn images(&self) -> Vec<PageImageEntry> {
        self.inner
            .images
            .iter()
            .map(|i| PageImageEntry { inner: i.clone() })
            .collect()
    }

    /// Structural equality (all four fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        let images = self
            .inner
            .images
            .iter()
            .map(image_entry_repr)
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "PageImagesData(page_index={}, format={:?}, geometry={}, images=[{}])",
            self.inner.page_index,
            self.inner.format,
            match &self.inner.geometry {
                Some(box_) =>
                    page_geometry_repr(&documents_impl::PageGeometry::from_page_box(*box_,)),
                None => "None".to_string(),
            },
            images,
        )
    }
}

/// The `PageImagesData.images` element's repr: the entry's own Python
/// shape (the repr law — never the Rust Debug).
fn image_entry_repr(entry: &tors::pdf_impl::PageImageEntry) -> String {
    format!(
        "PageImageEntry(bbox={}, width_px={}, height_px={}, rotation_degrees={}, \
         paint_order={}, color_space={:?}, color_space_declared={}, indexed_base={}, \
         bits_per_component={}, byte_size_compressed={}, filters={:?}, is_inline={}, data={})",
        match entry.bbox {
            Some(bbox) => format!("{bbox:?}"),
            None => "None".to_string(),
        },
        entry.width_px,
        entry.height_px,
        entry.rotation_degrees,
        entry.paint_order,
        entry.color_space,
        entry.color_space_declared,
        match &entry.indexed_base {
            Some(base) => format!("Some({base:?})"),
            None => "None".to_string(),
        },
        entry.bits_per_component,
        entry.byte_size_compressed,
        entry.filters,
        entry.is_inline,
        image_bytes_repr(&entry.data),
    )
}

/// The `PageImageEntry.data` repr: the ImageBytes' own Python shape (the
/// four-arm sum, each arm's fields as the getters return them).
fn image_bytes_repr(data: &tors::pdf_impl::ImageBytes) -> String {
    use tors::pdf_impl::{ContainerFormat, ImageBytes as IB};
    match data {
        IB::NotRequested => "ImageBytes(kind='not_requested')".to_string(),
        IB::Container { bytes, container } => format!(
            "ImageBytes(kind='container', bytes=<{} bytes>, container={:?})",
            bytes.len(),
            match container {
                ContainerFormat::Jpeg => "jpeg",
                ContainerFormat::Jp2 => "jp2",
                ContainerFormat::J2kCodestream => "j2k_codestream",
            }
        ),
        IB::Raw {
            pixels,
            width_px,
            height_px,
            components,
            bits_per_component,
            pixel_format,
            samples_are_raw,
            decode_folded_in,
            icc,
            ccitt,
        } => format!(
            "ImageBytes(kind='raw', pixels=<{} bytes>, width_px={}, height_px={}, \
             components={}, bpc={}, pixel_format={:?}, samples_are_raw={}, \
             decode_folded_in={}, icc={}, ccitt={})",
            pixels.len(),
            width_px,
            height_px,
            components,
            bits_per_component,
            match pixel_format {
                tors::pdf_impl::PixelFormat::RGB => "rgb",
                tors::pdf_impl::PixelFormat::Grayscale => "grayscale",
                tors::pdf_impl::PixelFormat::CMYK => "cmyk",
            },
            samples_are_raw,
            decode_folded_in,
            icc,
            match ccitt {
                Some(params) => format!(
                    "(k={}, columns={}, rows={}, black_is_1={}, end_of_line={}, \
                     encoded_byte_align={}, end_of_block={})",
                    params.k,
                    params.columns,
                    params.rows.map_or("None".to_string(), |r| r.to_string()),
                    params.black_is_1,
                    params.end_of_line,
                    params.encoded_byte_align,
                    params.end_of_block
                ),
                None => "None".to_string(),
            }
        ),
        IB::Error(error) => match error {
            tors::pdf_impl::ImageDataError::Decode(message) => {
                format!("ImageBytes(kind='error', error_kind='decode', message={message:?})")
            }
            tors::pdf_impl::ImageDataError::Unsupported(message) => {
                format!("ImageBytes(kind='error', error_kind='unsupported', message={message:?})")
            }
            tors::pdf_impl::ImageDataError::SizeCap {
                scope,
                limit_bytes,
                needed_bytes,
            } => format!(
                "ImageBytes(kind='error', error_kind='size_cap', scope={:?}, \
                 limit_bytes={}, needed_bytes={})",
                match scope {
                    tors::pdf_impl::CapScope::Image => "image",
                    tors::pdf_impl::CapScope::Call => "call",
                },
                limit_bytes,
                needed_bytes
            ),
        },
    }
}

/// One image ([`PageImagesData.images`]): the handle's evidence always,
/// the bytes only under opt-in decode.
///
/// Exposed DIRECTLY; `==` is structural (all thirteen fields).
#[pyclass(module = "_tors_documents")]
pub struct PageImageEntry {
    inner: tors::pdf_impl::PageImageEntry,
}

#[pymethods]
impl PageImageEntry {
    /// The image's axis-aligned box, `(x, y, w, h)`: points, bottom-left
    /// origin of the page's render space — the same mapped box
    /// `page_layout`'s `ImageBox` carries. `None` when the page has no
    /// USABLE frame to map into — the box cannot be read, or it declares
    /// a degenerate (zero/non-finite) extent ([R11]: the bytes are still
    /// served either way; only the BOX is frame-gated).
    #[getter]
    fn bbox(&self) -> Option<(f32, f32, f32, f32)> {
        self.inner.bbox
    }

    /// The image's pixel width (the DICTIONARY's /Width — the decoded
    /// buffer's own dims ride `data`).
    #[getter]
    fn width_px(&self) -> u32 {
        self.inner.width_px
    }

    /// The image's pixel height (the dictionary's).
    #[getter]
    fn height_px(&self) -> u32 {
        self.inner.height_px
    }

    /// The CTM's rotation angle in degrees, `atan2`-normalized to
    /// [0, 360) — evidence, not an instruction.
    #[getter]
    fn rotation_degrees(&self) -> f32 {
        self.inner.rotation_degrees
    }

    /// Page-local identity/z-order: the index among the page's painted
    /// images, in content-stream order — pairs with `page_layout`'s
    /// `images[paint_order]` (separate opens: a changed file between the
    /// calls breaks the pairing).
    #[getter]
    fn paint_order(&self) -> u32 {
        self.inner.paint_order
    }

    /// The resolved color space, spelled: `"DeviceGray" | "DeviceRGB" |
    /// "DeviceCMYK" | "Indexed" | "CalGray" | "CalRGB" | "Lab" |
    /// "ICCBased(n)" | "Separation" | "DeviceN" | "Pattern"`.
    #[getter]
    fn color_space(&self) -> &str {
        &self.inner.color_space
    }

    /// Whether the resolved color space is engine-verified (R8): the
    /// engine falls back to `DeviceRGB` when a dictionary's /ColorSpace
    /// is absent or unparseable (and decode() then fails on that same
    /// dictionary), so a `DeviceRGB` entry whose decode was not attempted
    /// and successful reads `false` (the guess, marked); true for any
    /// space the fallback cannot produce, and for a decode that
    /// succeeded.
    #[getter]
    fn color_space_declared(&self) -> bool {
        self.inner.color_space_declared
    }

    /// For an Indexed image, the de-indexed base space's name; None
    /// otherwise.
    #[getter]
    fn indexed_base(&self) -> Option<String> {
        self.inner.indexed_base.clone()
    }

    /// /BitsPerComponent — the DICTIONARY's value (R22); the decoded
    /// buffer's own depth rides `data` (a consumer computing a stride
    /// from this field's 16 would read an 8-byte buffer).
    #[getter]
    fn bits_per_component(&self) -> u8 {
        self.inner.bits_per_component
    }

    /// /Length — the compressed stream's size in bytes (an inline
    /// image's: the stored bytes between ID/EI).
    #[getter]
    fn byte_size_compressed(&self) -> u64 {
        self.inner.byte_size_compressed
    }

    /// The filter chain, outermost first, spelled (`"FlateDecode"`, …);
    /// an unlisted filter keeps its raw PDF name (R20).
    #[getter]
    fn filters(&self) -> Vec<String> {
        self.inner.filters.clone()
    }

    /// Inline (BI/ID/EI) vs an XObject Do.
    #[getter]
    fn is_inline(&self) -> bool {
        self.inner.is_inline
    }

    /// The decode result — the ALWAYS-PRESENT four-arm sum (R6/F9):
    /// `kind` is `"not_requested" | "container" | "raw" | "error"` and
    /// every other getter is None unless `kind` names it. A fresh native
    /// object per access — a clone of the entry's own arm (the
    /// `geometry` getter's fresh-object law): the byte getters below
    /// copy, they do not alias.
    #[getter]
    fn data(&self) -> ImageBytes {
        ImageBytes {
            inner: self.inner.data.clone(),
        }
    }

    /// Structural equality (all thirteen fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        image_entry_repr(&self.inner)
    }
}

/// The opt-in decode result ([`PageImageEntry.data`]): the always-present
/// four-arm sum (R6). `kind` discriminates (`"not_requested" |
/// "container" | "raw" | "error"`); every arm getter is None unless
/// `kind` names it (F9) — "no bytes" is never two different things.
///
/// Exposed DIRECTLY; `==` is structural (the whole arm).
#[pyclass(module = "_tors_documents")]
pub struct ImageBytes {
    inner: tors::pdf_impl::ImageBytes,
}

#[pymethods]
impl ImageBytes {
    /// The arm: `"not_requested"` (no decode asked, or a containers-only
    /// selection left this image out — the entry's `filters` say which),
    /// `"container"` (the container bytes: the engine's decode()
    /// passthrough — decryption included; `"jp2"`/`"j2k_codestream"` for
    /// a bare JPX stream, F1's sniff), `"raw"` (decoded samples that
    /// describe themselves), or `"error"` (a typed refusal: decode,
    /// unsupported, or size cap).
    #[getter]
    fn kind(&self) -> &str {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::NotRequested => "not_requested",
            IB::Container { .. } => "container",
            IB::Raw { .. } => "raw",
            IB::Error(_) => "error",
        }
    }

    /// The container bytes — None unless `kind == "container"` (F9).
    /// Each access COPIES the bytes into a fresh Python object (no
    /// aliasing, no caching: the getters are pure functions of the arm,
    /// and a caller mutating the returned `bytes` can never reach the
    /// entry). The copy is O(container) transient per access, by
    /// construction.
    #[getter]
    fn bytes(&self) -> Option<Vec<u8>> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Container { bytes, .. } => Some(bytes.clone()),
            _ => None,
        }
    }

    /// What the container bytes are: `"jpeg" | "jp2" |
    /// "j2k_codestream"` — None unless `kind == "container"` (F9).
    #[getter]
    fn container(&self) -> Option<&'static str> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Container {
                container: tors::pdf_impl::ContainerFormat::Jpeg,
                ..
            } => Some("jpeg"),
            IB::Container {
                container: tors::pdf_impl::ContainerFormat::Jp2,
                ..
            } => Some("jp2"),
            IB::Container {
                container: tors::pdf_impl::ContainerFormat::J2kCodestream,
                ..
            } => Some("j2k_codestream"),
            _ => None,
        }
    }

    /// The sample bytes — None unless `kind == "raw"` (F9). Each access
    /// COPIES the pixels into a fresh Python object (no aliasing, no
    /// caching: the `bytes` getter's copy law).
    #[getter]
    fn pixels(&self) -> Option<Vec<u8>> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Raw { pixels, .. } => Some(pixels.clone()),
            _ => None,
        }
    }

    /// The decoded buffer's width — None unless `kind == "raw"`.
    #[getter]
    fn width_px(&self) -> Option<u32> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Raw { width_px, .. } => Some(*width_px),
            _ => None,
        }
    }

    /// The decoded buffer's height — None unless `kind == "raw"`.
    #[getter]
    fn height_px(&self) -> Option<u32> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Raw { height_px, .. } => Some(*height_px),
            _ => None,
        }
    }

    /// The decoded buffer's component count (the Indexed case names the
    /// base's — the expansion's stride; the engine's `DeviceN` answer is
    /// its parser's flat 4 even when the real ink count differs — the
    /// buffer LENGTH is the truth there) — None unless `kind == "raw"`.
    #[getter]
    fn components(&self) -> Option<u8> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Raw { components, .. } => Some(*components),
            _ => None,
        }
    }

    /// The decoded buffer's bits per component (8 whenever a transform
    /// rewrote the samples; the dictionary's own value when the buffer
    /// stayed packed — CCITT, Lab, a non-1/3/4-component space) — None
    /// unless `kind == "raw"`.
    #[getter]
    fn bpc(&self) -> Option<u8> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Raw {
                bits_per_component, ..
            } => Some(*bits_per_component),
            _ => None,
        }
    }

    /// The engine's display tag for the buffer: `"rgb" | "grayscale" |
    /// "cmyk"` — a 1/3/4-byte tag that can LIE about the stride (R16:
    /// Lab → `"rgb"` over Lab samples; an `ICCBased(2)` → `"rgb"` over a
    /// packed 2-component buffer): `components`/`bpc` are the honest
    /// stride. None unless `kind == "raw"`.
    #[getter]
    fn pixel_format(&self) -> Option<&'static str> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Raw {
                pixel_format: tors::pdf_impl::PixelFormat::RGB,
                ..
            } => Some("rgb"),
            IB::Raw {
                pixel_format: tors::pdf_impl::PixelFormat::Grayscale,
                ..
            } => Some("grayscale"),
            IB::Raw {
                pixel_format: tors::pdf_impl::PixelFormat::CMYK,
                ..
            } => Some("cmyk"),
            _ => None,
        }
    }

    /// Whether the samples are still the values the image dictionary's
    /// own entries are expressed in — TRUE is the COMMON case (R25: an
    /// 8-bpc image with no non-default /Decode in a 1/3/4-component
    /// space keeps it; it means "nothing transformed this buffer", never
    /// "undecoded"). None unless `kind == "raw"`.
    #[getter]
    fn samples_are_raw(&self) -> Option<bool> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Raw {
                samples_are_raw, ..
            } => Some(*samples_are_raw),
            _ => None,
        }
    }

    /// Whether a non-default `/Decode` array is already mapped into the
    /// samples — strictly narrower than `!samples_are_raw` (R6's
    /// un-collapsed pair): a consumer that applies `/Decode` itself
    /// (separation-plate routing) must read this one. None unless
    /// `kind == "raw"`.
    #[getter]
    fn decode_folded_in(&self) -> Option<bool> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Raw {
                decode_folded_in, ..
            } => Some(*decode_folded_in),
            _ => None,
        }
    }

    /// Whether an ICC profile rode the decode — NAME-ONLY (R23): the
    /// profile's bytes are deliberately not carried (they can be hundreds
    /// of KiB and are shared per document); the space is named by the
    /// entry's `color_space`/`indexed_base`. None unless `kind == "raw"`.
    #[getter]
    fn icc(&self) -> Option<bool> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Raw { icc, .. } => Some(*icc),
            _ => None,
        }
    }

    /// The CCITT parameters the packed codestream cannot be decoded
    /// without (R10/F6: the packed stream is served VERBATIM — the
    /// consumer has no PDF to read /DecodeParms from): the tuple
    /// `(k, columns, rows, black_is_1, end_of_line, encoded_byte_align,
    /// end_of_block)` — all seven fields the engine carries; `rows` is
    /// None when undeclared. None for every non-CCITT image and unless
    /// `kind == "raw"`.
    #[getter]
    #[allow(clippy::type_complexity)] // the seven params ride one tuple
    fn ccitt(&self) -> Option<(i64, u32, Option<u32>, bool, bool, bool, bool)> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Raw {
                ccitt: Some(params),
                ..
            } => Some((
                params.k,
                params.columns,
                params.rows,
                params.black_is_1,
                params.end_of_line,
                params.encoded_byte_align,
                params.end_of_block,
            )),
            _ => None,
        }
    }

    /// Why the decode failed: `"decode"` (the engine's decode() failed —
    /// the message is its Display), `"unsupported"` (a filter this build
    /// cannot decode — retrying cannot fix it; the message is the
    /// engine's feature-naming string verbatim on the shipped build), or
    /// `"size_cap"` (this seam's ceiling refused BEFORE decoding — see
    /// `scope`). None unless `kind == "error"` (F9).
    #[getter]
    fn error_kind(&self) -> Option<&'static str> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Error(tors::pdf_impl::ImageDataError::Decode(_)) => Some("decode"),
            IB::Error(tors::pdf_impl::ImageDataError::Unsupported(_)) => Some("unsupported"),
            IB::Error(tors::pdf_impl::ImageDataError::SizeCap { .. }) => Some("size_cap"),
            _ => None,
        }
    }

    /// The failure's message (the engine's own text, or the composed F1
    /// refusal for an encrypted image whose container cannot be served) —
    /// None unless `kind == "error"`. The size-cap message names the
    /// scope (finding 11: a call-scope refusal is not an image-size
    /// story — the two ceilings are different sentences).
    #[getter]
    fn message(&self) -> Option<String> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Error(tors::pdf_impl::ImageDataError::Decode(message))
            | IB::Error(tors::pdf_impl::ImageDataError::Unsupported(message)) => {
                Some(message.clone())
            }
            IB::Error(tors::pdf_impl::ImageDataError::SizeCap {
                scope: tors::pdf_impl::CapScope::Image,
                ..
            }) => Some(
                "the image's decoded size exceeded the per-image ceiling \
                 (the caps bind on actual bytes)"
                    .to_string(),
            ),
            IB::Error(tors::pdf_impl::ImageDataError::SizeCap {
                scope: tors::pdf_impl::CapScope::Call,
                ..
            }) => Some(
                "serving this image would push the call past the cumulative \
                 decode budget (the caps bind on actual bytes)"
                    .to_string(),
            ),
            _ => None,
        }
    }

    /// Which ceiling fired: `"image"` (the image's own estimate over the
    /// per-image cap) or `"call"` (the cumulative budget the image would
    /// push past) — None unless `kind == "error"` AND `error_kind ==
    /// "size_cap"`.
    #[getter]
    fn scope(&self) -> Option<&'static str> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Error(tors::pdf_impl::ImageDataError::SizeCap {
                scope: tors::pdf_impl::CapScope::Image,
                ..
            }) => Some("image"),
            IB::Error(tors::pdf_impl::ImageDataError::SizeCap {
                scope: tors::pdf_impl::CapScope::Call,
                ..
            }) => Some("call"),
            _ => None,
        }
    }

    /// The ceiling that fired, in bytes: the per-image cap (the default
    /// 256 MiB or the `max_image_bytes=` override) for `scope ==
    /// "image"`, the 512 MiB per-call budget for `scope == "call"` —
    /// None unless `error_kind == "size_cap"`.
    #[getter]
    fn limit_bytes(&self) -> Option<u64> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Error(tors::pdf_impl::ImageDataError::SizeCap { limit_bytes, .. }) => {
                Some(*limit_bytes)
            }
            _ => None,
        }
    }

    /// The bytes the work needed: the image's own estimate OR ACTUAL
    /// size (whichever check fired) for `scope == "image"`, the call's
    /// cumulative total had the image been served for `scope == "call"`
    /// — None unless `error_kind == "size_cap"`.
    #[getter]
    fn needed_bytes(&self) -> Option<u64> {
        use tors::pdf_impl::ImageBytes as IB;
        match &self.inner {
            IB::Error(tors::pdf_impl::ImageDataError::SizeCap { needed_bytes, .. }) => {
                Some(*needed_bytes)
            }
            _ => None,
        }
    }

    /// Structural equality (the whole arm).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        image_bytes_repr(&self.inner)
    }
}

/// One `page_text_chars` entry: the page's chars from the cached span
/// parse, reading order.
///
/// Exposed DIRECTLY; `==` is structural (the four fields).
#[pyclass(module = "_tors_documents")]
pub struct PageCharsData {
    inner: tors::pdf_impl::PageCharsData,
}

#[pymethods]
impl PageCharsData {
    /// The 0-based page this entry describes — mandatory on every entry,
    /// because the `pages=` normalization reorders.
    #[getter]
    fn page_index(&self) -> usize {
        self.inner.page_index
    }

    /// The resolved format's vocabulary name (`to_markdown`'s spelling):
    /// the entry's provenance marker — on an office row `chars: []` is
    /// the per-format capability statement (chars are PDF-only in v1),
    /// not "no text".
    #[getter]
    fn format(&self) -> &str {
        &self.inner.format
    }

    /// The page's declared render box (the `page_geometry` entry): `None`
    /// where the page's box cannot be read or declares no positive
    /// extent — then `chars` is empty too (no frame to map into).
    #[getter]
    fn geometry(&self) -> Option<PageGeometry> {
        self.inner.geometry.map(|box_| PageGeometry {
            inner: documents_impl::PageGeometry::from_page_box(box_),
        })
    }

    /// The page's chars, reading order (span order × in-span order).
    #[getter]
    fn chars(&self) -> Vec<TextChar> {
        self.inner
            .chars
            .iter()
            .map(|c| TextChar { inner: c.clone() })
            .collect()
    }

    /// Structural equality (all four fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        // the capped list renderer (finding 14: an uncapped repr printed
        // every char of the page — a 4,288-char page repr'd 511,468 chars)
        format!(
            "PageCharsData(page_index={}, format={:?}, geometry={}, chars={})",
            self.inner.page_index,
            self.inner.format,
            match &self.inner.geometry {
                Some(box_) =>
                    page_geometry_repr(&documents_impl::PageGeometry::from_page_box(*box_,)),
                None => "None".to_string(),
            },
            structure_list_repr(&self.inner.chars, text_char_repr),
        )
    }
}

/// The `PageCharsData.chars` element's repr (the repr law).
fn text_char_repr(ch: &tors::pdf_impl::TextCharBox) -> String {
    format!(
        "TextChar(char={:?}, bbox={:?}, span_index={}, index_in_span={}, wmode={}, x_measured={})",
        ch.ch, ch.bbox, ch.span_index, ch.index_in_span, ch.wmode, ch.x_measured,
    )
}

/// One character ([`PageCharsData.chars`]): the advance box from the
/// baseline, in the page's render space — the per-char extension of the
/// span surface's em-box law.
///
/// Exposed DIRECTLY; `==` is structural (all six fields).
#[pyclass(module = "_tors_documents")]
pub struct TextChar {
    inner: tors::pdf_impl::TextCharBox,
}

#[pymethods]
impl TextChar {
    /// The character (one Unicode scalar — Python string indices are
    /// code points, so `spans[s].text[i:j]` slicing aligns with
    /// `index_in_span`).
    #[getter]
    fn char(&self) -> char {
        self.inner.ch
    }

    /// `(x, y, w, h)`: points, bottom-left origin of the page's render
    /// space; **y is the BASELINE** — the box extends one font-size
    /// upward (the em-box law; descenders hang BELOW the box's bottom
    /// edge), h is the span's font size, w the char's advance. Boxes are
    /// unclipped; the page box is the consumer's clip.
    #[getter]
    fn bbox(&self) -> (f32, f32, f32, f32) {
        self.inner.bbox
    }

    /// The owning span's index in `page_layout`'s `spans` list for this
    /// page — the join law makes `"".join(chars' char)` ==
    /// `spans[span_index].text` EXACTLY.
    #[getter]
    fn span_index(&self) -> usize {
        self.inner.span_index
    }

    /// The char's index WITHIN its span, in code points — chars
    /// `[i, j)` of span s ground the substring `spans[s].text[i:j]`
    /// (their union box). The WHOLE-span box is the span's own mapped
    /// box, never the char union.
    #[getter]
    fn index_in_span(&self) -> usize {
        self.inner.index_in_span
    }

    /// The span's writing mode: 0 horizontal, 1 vertical. The WMode-1
    /// honesty field — a vertical run's char boxes are the engine's
    /// flattened fiction (displaced one advance LEFT of their origins,
    /// stacked along x, measured): a consumer may not pretend
    /// `wmode == 1` chars are highlight-able.
    #[getter]
    fn wmode(&self) -> u8 {
        self.inner.wmode
    }

    /// How the char's x was derived — SPAN-CONSTANT (pinned): `true` =
    /// the engine's measured origins (drift ≈ the 0.5 pt tolerance +
    /// sub-point interpolation); `false` = the prefix-sum/uniform arm
    /// (the TJ-kerning drift class — the engine folds Tc/Tw/Tz/TJ into
    /// the widths, so measured drift on every probe fixture is 0.000;
    /// the analytic bound is `|Σ TJ adjustments before i| ×
    /// font_size/1000` pt). A highlight box built from `x_measured:
    /// false` chars is APPROXIMATE.
    #[getter]
    fn x_measured(&self) -> bool {
        self.inner.x_measured
    }

    /// Structural equality (all six fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        text_char_repr(&self.inner)
    }
}

/// `tors.documents.document_structure(path, data=None, format=None,
/// backend=None, max_bytes=None) -> DocumentStructure`: or
/// `document_structure(data=...)`: the structure surface (issues
/// #209/#210/#211) — one report whose resolved `format` discriminates and
/// whose `payload` is exactly one of the three per-format structures:
///
/// - **docx** (`DocxStructure`): the body's blocks in document order from
///   the office_oxide RAW layer — headings (`level` clamped 1..=6 plus
///   the raw 0-based `outline_level`; the ladder is direct
///   `w:outlineLvl` → the stylesheet's basedOn chain → the Heading1..9
///   style-ID convention → the style-NAME convention), list state from
///   the EFFECTIVE properties (style-carried `numPr` included), the
///   block's visible text (hidden runs resolved, breaks/tabs, text-box
///   prose recursed, form fields), tables (grid verbatim, gridSpan- and
///   vMerge-derived spans, `tblCaption`), hyperlinks (href/anchor split),
///   real pictures (rid + extension + declared extents ÷ 12700), the
///   section partition (`section_breaks` + the count law), note counts,
///   and `unreadable_parts`.
/// - **xlsx** (`XlsxStructure`): the worksheets (hidden and macro sheets
///   served, each carrying its own name and state), the workbook's
///   chart/dialog sheet names, the used range COMPUTED from the parsed
///   cells (styled-empty cells included; the declared `<dimension>` rides
///   beside it for provenance), the cells with values AS THE FILE STATES
///   THEM (`type`-discriminated bool | float | str | None, shared strings
///   dereferenced to their full text, `t="d"` the only date shape),
///   formula text + cached value both labeled, merged ranges, sheet
///   hyperlinks (href/anchor split), comments, and defined names.
/// - **html** (`HtmlStructure`): the conversion lane's own parse run once
///   with the structure collector on (no second parser, no second
///   opinion): the flat node array (parents precede children), the NINE
///   reachable kinds, heading `id` attributes, link annotations with byte
///   ranges and url/title, table grids with row/col/spans/header cells —
///   the conversion output is discarded; the surface returns structure
///   only.
///
/// Entries carry container anchors, never boxes — no coordinates anywhere
/// (a docx section index, a cell reference, a node id: those are the
/// anchors; a caller wanting office/PDF coordinates is on
/// page_geometry/page_tables/page_layout).
///
/// `backend` names the engine lane: `"auto"` is the default and `"oxide"`
/// is identical on the served office rows (the structure lane is a
/// property of the format). The refused formats refuse on EVERY backend —
/// a format-level `ValueError` ("document_structure cannot serve …")
/// fired before the backend check, so `(pdf, anydoc)` answers the pdf
/// refusal, never an anydoc-template message whose "use backend='auto'"
/// tail would be a lie: pdf (its answer is already geometry + tables),
/// doc/xls/ppt (office_oxide's converter-internal IR is not a raw layer),
/// pptx (the raw layer is reachable; the surface is not built yet), and
/// rtf/odt/ods/odp/epub/csv (no parse tree anywhere — the format converts
/// fine; "no structure surface", never "unreadable"). The served office
/// rows refuse `backend="anydoc"` as a capability (`ValueError`); html's
/// forced backends are conversion's own `UnsupportedBackend` pair
/// byte-identical.
///
/// `format` names the format explicitly (extension spelling, no dot,
/// case-insensitive; the container aliases `docm`/`xlsm` resolve onto
/// their served bases). The explicit name is trusted and NEVER re-sniffed
/// — the conversion lane's own precedence, so `format="html"` on docx
/// bytes is the html lane's own binary-input refusal, and the fix-it is
/// `sniff()`. The one container-dependent name is the Excel kind's OLE
/// case: an OLE container under the Excel kind is a legacy xls workbook
/// OR an encrypted xlsx package (both CFB, indistinguishable without a
/// full legacy parse) and refuses with the union message naming both.
///
/// There is NO `password=` parameter: no served format accepts one (the
/// engines are office_oxide's raw readers and the html collector; an
/// encrypted OOXML package is a typed engine `ValueError`, a PDF refuses
/// first, and a Rust caller passing one into the core gets the explicit
/// refusal — never a silently-ignored argument).
///
/// `max_bytes` is the caller's input budget in bytes (a positive `int`):
/// an explicit budget binds pre-read exactly as on every other entry
/// point, and `None` keeps the 32 MiB default ceiling on every served row
/// (the office legs' audit-and-ceiling pairing and the html lane's
/// — the conversion lane's ~23× measured amplification is a LOWER bound
/// for the structure lane, which also holds the node array). Zero is
/// refused core-side, sequenced after the capability refusal. The entry
/// cap is separate and honest: [`MAX_STRUCTURE_ENTRIES`] — 200,000 —
/// bounds the PAYLOAD's entry lists in document order as ONE RUNNING
/// BUDGET that counts every materialized entry object: docx blocks AND
/// their table cells, xlsx cells, html nodes AND their grid cells (a
/// cell is an entry object — pre-fix the cap let 1M docx cells and 300k
/// html cells ride inside one entry), with `entries_truncated` on the
/// affected payload and all sheet/range/merge/name
/// metadata complete — a marshalling bound, never a refusal.
///
/// Errors: `OSError` for a missing/unreadable file — but NEVER for a
/// call the name already refused: an explicit `format=` naming a
/// non-Excel kind decides the refusal from the name alone and touches
/// no filesystem (a chmod-000/directory/missing path answers the format
/// refusal, its OSError never fires); `ValueError` for
/// bytes that resolve to no format (`sniff()` is the fix-it), an unknown
/// `format=` name, a bad backend name, every refused format row (GIL-side,
/// resolved from `format=`/the 64 KiB prefix alone, no body read — the 64
/// KiB format probe runs only when the name alone cannot decide; a
/// positive content-resolve to a refused kind stops at the prefix and
/// refuses), the Excel-OLE union, `backend="anydoc"` on a served office
/// row, an over-budget input, `max_bytes=0`, a malformed/encrypted
/// document, an office container that will not parse or pass the
/// decompression audit, and an over-ceiling input.
///
/// GIL model: the argument marshalling plus the capability probe — a
/// bounded 64 KiB prefix on the data= lane (NEVER the full buffer; an
/// explicit non-Excel name consults no bytes at all) or a bounded
/// prefix read on the path lane — under the GIL; the whole
/// read+resolve+structure pass under one `py.detach`; then O(payload) of
/// native-object construction post-detach (the payload pyclasses are
/// thin getters over plain core structs — structural `==` throughout).
/// The html lane parses UNTRUSTED markup with the same engine the
/// conversion pair uses; see docs/documents.md, SECURITY.md.
#[pyfunction(signature = (
    path = None,
    data = None,
    format = None,
    backend = None,
    max_bytes = None,
))]
pub fn document_structure(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    format: Option<&Bound<'_, PyAny>>,
    backend: Option<&Bound<'_, PyAny>>,
    max_bytes: Option<&Bound<'_, PyAny>>,
) -> PyResult<DocumentStructure> {
    let source = parse_source(path, data, "document_structure")?;
    let format = parse_format(format)?;
    let backend = parse_backend(backend)?;
    // The capability refusal fires HERE — under the GIL, before the
    // detach and before any byte of the document is read — for EVERY
    // backend (the format-level family precedes the backend check; the
    // geometry surfaces' probe is anydoc-only because their refusals
    // are, and this surface's are not). The probe resolves from the
    // format name (an explicit name that resolves to a non-Excel kind
    // consults NO bytes and touches NO filesystem — the path branch's
    // open(2) has no business running when the name decides), the
    // content markers in the 64 KiB prefix (data= hands the probe the
    // BOUNDED prefix, never the full buffer), or the
    // name hint's extension — the same precedence the core's resolve
    // runs — and defers when even the prefix cannot resolve (the
    // core's authoritative resolve is the only truth).
    if let Some(refusal) = probe_structure_refusal(&source, format, backend) {
        return Err(document_error(py, refusal)?);
    }
    // There is no password= parameter to parse (F25): no served format
    // accepts one, and the core's own check keeps the Rust seam honest.
    // The zero value threads through (`parse_max_bytes_on`): the core's
    // sequenced check is this surface's zero refusal.
    let max_bytes = parse_max_bytes_on(max_bytes, ZeroRefusal::CoreSide)?;
    let report = match py.detach(move || {
        let (bytes, hint) = source
            .into_structure_input(max_bytes, format, backend)
            .map_err(input_document_error)?;
        documents_impl::document_structure(
            bytes,
            hint.as_deref(),
            format,
            backend,
            &documents_impl::ConvertOptions {
                password: None,
                max_bytes,
            },
        )
    }) {
        Ok(report) => report,
        Err(err) => return Err(document_error(py, err)?),
    };
    // O(1) of native construction here — the payload pyclasses are thin
    // getters, materializing their entries on access (the PageTables
    // residue class).
    Ok(DocumentStructure { inner: report })
}

/// The structure surface's pre-read capability probe (the
/// `probe_capability_refusal` twin, keyed on the structure table and
/// fired for EVERY backend): the format-level refusal the call will
/// raise, resolved as far as the format resolves without the
/// authoritative read. `None` defers to the core (unresolvable prefix /
/// a served row).
fn probe_structure_refusal(
    source: &Source<'_>,
    format: Option<&str>,
    backend: Backend,
) -> Option<DocumentError> {
    let probed: Result<Option<tors::documents_impl::Kind>, DocumentError> = match source {
        Source::Data(bytes) => {
            documents_impl::probe_structure_capability(probe_sniff_bytes(bytes, format), None, format, backend)
        }
        Source::Path(path) => {
            // The name-decided refusal opens NOTHING (the claim at the
            // docstring above — "resolved from format=… no body read" —
            // is now true of the path lane too): an explicit format=
            // that resolves to a non-Excel kind decides from the name
            // alone, so a chmod-000 file with format=pdf answers the
            // "cannot serve pdf" refusal (measured pre-fix:
            // PermissionError INSTEAD of the refusal — the probe
            // opened and read the file it was about to refuse), a
            // directory answers it (pre-fix IsADirectoryError), a
            // missing path answers it (pre-fix FileNotFoundError).
            // Excel keeps the prefix read: the OLE-container union
            // decision is bytes'.
            if name_decides_probe(format) {
                return documents_impl::probe_structure_capability(&[], None, format, backend)
                    .err();
            }
            let gated = std::fs::metadata(path)
                .map_err(|_| ())
                .and_then(|meta| refuse_non_regular(path, meta.file_type()).map_err(|_| ()));
            if gated.is_err() {
                return None;
            }
            let mut file = std::fs::File::open(path).ok()?;
            let (prefix, _over) =
                documents_impl::read_bounded_into(&mut file, Vec::new(), SNIFF_PREFIX).ok()?;
            documents_impl::probe_structure_capability(&prefix, path.to_str(), format, backend)
        }
    };
    probed.err()
}

/// The `document_structure` report: the resolved format's vocabulary name
/// and exactly one per-format payload. The payload getter is the ONE
/// discriminating accessor (`isinstance(report.payload, DocxStructure)`
/// narrows); `==` is structural (the whole report).
#[pyclass(module = "_tors_documents")]
pub struct DocumentStructure {
    inner: documents_impl::StructureReport,
}

#[pymethods]
impl DocumentStructure {
    /// The resolved format's vocabulary name (the `to_markdown` spelling,
    /// the container-aware `xls`/`xlsx` case included): provenance, and
    /// the same spelling the refusals use.
    #[getter]
    fn format(&self) -> &str {
        self.inner.format
    }

    /// The per-format structure: exactly one of `DocxStructure`,
    /// `XlsxStructure`, or `HtmlStructure` — narrow with `isinstance`.
    /// A fresh native object per access (the `PageLayout.geometry`
    /// precedent), structurally equal to its siblings.
    #[getter]
    fn payload(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.payload {
            documents_impl::StructurePayload::Docx(inner) => Ok(Py::new(
                py,
                DocxStructure {
                    inner: inner.clone(),
                },
            )?
            .into_any()),
            documents_impl::StructurePayload::Xlsx(inner) => Ok(Py::new(
                py,
                XlsxStructure {
                    inner: inner.clone(),
                },
            )?
            .into_any()),
            documents_impl::StructurePayload::Html(inner) => Ok(Py::new(
                py,
                HtmlStructure {
                    inner: inner.clone(),
                },
            )?
            .into_any()),
        }
    }

    /// Structural equality: two reports are equal exactly when their
    /// formats and payloads are (the core owns plain `PartialEq`
    /// structs — office_oxide's own types carry no `PartialEq`, so the
    /// seam's types are the comparison's home).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "DocumentStructure(format={:?}, payload={})",
            self.inner.format,
            structure_payload_repr(&self.inner.payload)
        )
    }
}

/// The payload pyclasses' shared repr shape: each payload's own class
/// name and fields, the way its own `__repr__` prints standalone.
fn structure_payload_repr(payload: &documents_impl::StructurePayload) -> String {
    match payload {
        documents_impl::StructurePayload::Docx(inner) => docx_structure_repr(inner),
        documents_impl::StructurePayload::Xlsx(inner) => xlsx_structure_repr(inner),
        documents_impl::StructurePayload::Html(inner) => html_structure_repr(inner),
    }
}

/// The payload lists' repr: the entries as their own classes' Python
/// shapes, capped so a repr of a capped payload stays usable (the
/// truncation marker is part of the shape).
fn structure_list_repr<T>(entries: &[T], repr: impl Fn(&T) -> String) -> String {
    if entries.is_empty() {
        return "[]".to_string();
    }
    let shown = entries.len().min(3);
    let items = entries[..shown]
        .iter()
        .map(repr)
        .collect::<Vec<_>>()
        .join(", ");
    let rest = if entries.len() > shown {
        format!(", … {} more", entries.len() - shown)
    } else {
        String::new()
    };
    format!("[{items}{rest}]")
}

fn docx_link_repr(link: &documents_impl::DocxLink) -> String {
    format!(
        "DocxLink(text={:?}, href={}, anchor={})",
        link.text,
        option_str_repr(link.href.as_deref()),
        option_str_repr(link.anchor.as_deref()),
    )
}

fn docx_image_repr(image: &documents_impl::DocxImage) -> String {
    format!(
        "DocxImage(relationship_id={:?}, file_ext={}, description={}, width_pt={}, height_pt={}, inline={}, decorative={})",
        image.relationship_id,
        option_str_repr(image.file_ext.as_deref()),
        option_str_repr(image.description.as_deref()),
        option_float_repr(image.width_pt),
        option_float_repr(image.height_pt),
        image.inline,
        image.decorative,
    )
}

fn option_str_repr(value: Option<&str>) -> String {
    match value {
        Some(text) => format!("{text:?}"),
        None => "None".to_string(),
    }
}

fn option_float_repr(value: Option<f32>) -> String {
    match value {
        Some(number) => format!("{number:?}"),
        None => "None".to_string(),
    }
}

fn docx_cell_repr(cell: &documents_impl::DocxCell) -> String {
    format!(
        "DocxCell(row={}, col={}, text={:?}, row_span={}, col_span={})",
        cell.row, cell.col, cell.text, cell.row_span, cell.col_span,
    )
}

fn docx_table_repr(table: &documents_impl::DocxTable) -> String {
    format!(
        "DocxTable(rows={}, cols={}, cells={}, caption={})",
        table.rows,
        table.cols,
        structure_list_repr(&table.cells, docx_cell_repr),
        option_str_repr(table.caption.as_deref()),
    )
}

fn docx_block_repr(block: &documents_impl::DocxBlock) -> String {
    format!(
        "DocxBlock(index={}, section={}, kind={:?}, text={:?}, level={}, outline_level={}, table={}, links={}, images={})",
        block.index,
        block.section,
        block.kind,
        block.text,
        option_match_repr(block.level),
        option_match_repr(block.outline_level),
        block
            .table
            .as_ref()
            .map(docx_table_repr)
            .unwrap_or_else(|| "None".to_string()),
        structure_list_repr(&block.links, docx_link_repr),
        structure_list_repr(&block.images, docx_image_repr),
    )
}

fn option_match_repr(value: Option<impl std::fmt::Display>) -> String {
    match value {
        Some(number) => format!("{number}"),
        None => "None".to_string(),
    }
}

fn docx_structure_repr(docx: &documents_impl::DocxStructure) -> String {
    format!(
        "DocxStructure(blocks={}, sections={}, footnotes={}, endnotes={}, comments={}, entries_truncated={})",
        structure_list_repr(&docx.blocks, docx_block_repr),
        docx.sections,
        docx.footnotes,
        docx.endnotes,
        docx.comments,
        docx.entries_truncated,
    )
}

fn xlsx_cell_repr(cell: &documents_impl::XlsxCell) -> String {
    format!(
        "XlsxCell(ref={:?}, row={}, col={}, type={:?}, value={}, formula={}, has_cached_value={})",
        cell.reference,
        cell.row,
        cell.col,
        cell.kind,
        xlsx_scalar_repr(cell.value.as_ref()),
        option_str_repr(cell.formula.as_deref()),
        cell.has_cached_value,
    )
}

fn xlsx_scalar_repr(value: Option<&documents_impl::XlsxScalar>) -> String {
    match value {
        None => "None".to_string(),
        Some(documents_impl::XlsxScalar::Bool(true)) => "True".to_string(),
        Some(documents_impl::XlsxScalar::Bool(false)) => "False".to_string(),
        Some(documents_impl::XlsxScalar::Number(number)) => format!("{number:?}"),
        Some(documents_impl::XlsxScalar::Text(text)) => format!("{text:?}"),
    }
}

fn xlsx_hyperlink_repr(link: &documents_impl::XlsxHyperlink) -> String {
    format!(
        "XlsxHyperlink(cell_ref={:?}, href={}, anchor={})",
        link.cell_ref,
        option_str_repr(link.href.as_deref()),
        option_str_repr(link.anchor.as_deref()),
    )
}

fn xlsx_comment_repr(comment: &documents_impl::XlsxComment) -> String {
    format!(
        "XlsxComment(cell_ref={:?}, author={}, text={:?})",
        comment.cell_ref,
        option_str_repr(comment.author.as_deref()),
        comment.text,
    )
}

fn xlsx_defined_name_repr(name: &documents_impl::XlsxDefinedName) -> String {
    format!(
        "XlsxDefinedName(name={:?}, value={:?}, local_sheet_id={}, local_sheet_name={}, hidden={})",
        name.name,
        name.value,
        option_match_repr(name.local_sheet_id),
        option_str_repr(name.local_sheet_name.as_deref()),
        name.hidden,
    )
}

fn xlsx_sheet_repr(sheet: &documents_impl::XlsxSheet) -> String {
    format!(
        "XlsxSheet(name={:?}, state={:?}, used_range={}, declared_dimension={}, max_row={}, max_col={}, cells={}, merged_cells={:?}, hyperlinks={}, comments={})",
        sheet.name,
        sheet.state,
        option_str_repr(sheet.used_range.as_deref()),
        option_str_repr(sheet.declared_dimension.as_deref()),
        option_match_repr(sheet.max_row),
        option_match_repr(sheet.max_col),
        structure_list_repr(&sheet.cells, xlsx_cell_repr),
        sheet.merged_cells,
        structure_list_repr(&sheet.hyperlinks, xlsx_hyperlink_repr),
        structure_list_repr(&sheet.comments, xlsx_comment_repr),
    )
}

fn xlsx_structure_repr(xlsx: &documents_impl::XlsxStructure) -> String {
    format!(
        "XlsxStructure(sheets={}, chartsheets={:?}, defined_names={}, date1904={}, unreadable_sheets={:?}, entries_truncated={})",
        structure_list_repr(&xlsx.sheets, xlsx_sheet_repr),
        xlsx.chartsheets,
        structure_list_repr(&xlsx.defined_names, xlsx_defined_name_repr),
        xlsx.date1904,
        xlsx.unreadable_sheets,
        xlsx.entries_truncated,
    )
}

fn html_annotation_repr(annotation: &documents_impl::HtmlAnnotation) -> String {
    format!(
        "HtmlAnnotation(kind={:?}, start={}, end={}, url={}, title={})",
        annotation.kind,
        annotation.start,
        annotation.end,
        option_str_repr(annotation.url.as_deref()),
        option_str_repr(annotation.title.as_deref()),
    )
}

fn html_grid_cell_repr(cell: &documents_impl::HtmlGridCell) -> String {
    format!(
        "HtmlGridCell(row={}, col={}, row_span={}, col_span={}, is_header={}, content={:?})",
        cell.row, cell.col, cell.row_span, cell.col_span, cell.is_header, cell.content,
    )
}

fn html_grid_repr(grid: &documents_impl::HtmlGrid) -> String {
    format!(
        "HtmlGrid(rows={}, cols={}, cells={})",
        grid.rows,
        grid.cols,
        structure_list_repr(&grid.cells, html_grid_cell_repr),
    )
}

fn html_attributes_repr(attributes: &Option<std::collections::HashMap<String, String>>) -> String {
    match attributes {
        None => "None".to_string(),
        Some(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let pairs = keys
                .iter()
                .map(|key| format!("{key:?}: {:?}", map[*key]))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{pairs}}}")
        }
    }
}

fn html_node_repr(node: &documents_impl::HtmlNode) -> String {
    format!(
        "HtmlNode(id={:?}, kind={:?}, text={}, level={}, parent={}, attributes={}, annotations={})",
        node.id,
        node.kind,
        option_str_repr(node.text.as_deref()),
        option_match_repr(node.level),
        option_match_repr(node.parent),
        html_attributes_repr(&node.attributes),
        structure_list_repr(&node.annotations, html_annotation_repr),
    )
}

fn html_structure_repr(html: &documents_impl::HtmlStructure) -> String {
    format!(
        "HtmlStructure(nodes={}, entries_truncated={})",
        structure_list_repr(&html.nodes, html_node_repr),
        html.entries_truncated,
    )
}

/// The DOCX payload (the `document_structure` report's `payload` for a
/// docx): thin getters over the core's plain struct; `==` is structural.
#[pyclass(module = "_tors_documents")]
pub struct DocxStructure {
    inner: documents_impl::DocxStructure,
}

#[pymethods]
impl DocxStructure {
    /// The body's blocks, body order.
    #[getter]
    fn blocks(&self) -> Vec<DocxBlock> {
        self.inner
            .blocks
            .iter()
            .map(|block| DocxBlock {
                inner: block.clone(),
            })
            .collect()
    }

    /// The body partition law: `section_breaks.len() + 1`.
    #[getter]
    fn sections(&self) -> usize {
        self.inner.sections
    }

    /// `body.section_breaks` verbatim (anchor math for `block.section`).
    #[getter]
    fn section_breaks(&self) -> Vec<usize> {
        self.inner.section_breaks.clone()
    }

    /// Footnote body count.
    #[getter]
    fn footnotes(&self) -> usize {
        self.inner.footnotes
    }

    /// Endnote body count.
    #[getter]
    fn endnotes(&self) -> usize {
        self.inner.endnotes
    }

    /// Comment body count.
    #[getter]
    fn comments(&self) -> usize {
        self.inner.comments
    }

    /// The raw layer's unreadable parts, verbatim (the report-the-loss
    /// law).
    #[getter]
    fn unreadable_parts(&self) -> Vec<String> {
        self.inner.unreadable_parts.clone()
    }

    /// True when the blocks stopped at `MAX_STRUCTURE_ENTRIES` (the
    /// marshalling bound; the metadata above stays complete).
    #[getter]
    fn entries_truncated(&self) -> bool {
        self.inner.entries_truncated
    }

    /// Structural equality (all eight fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        docx_structure_repr(&self.inner)
    }
}

/// One DOCX body block.
#[pyclass(module = "_tors_documents")]
pub struct DocxBlock {
    inner: documents_impl::DocxBlock,
}

#[pymethods]
impl DocxBlock {
    /// Position in `body.elements` (the stable container anchor).
    #[getter]
    fn index(&self) -> usize {
        self.inner.index
    }

    /// The covering section (the count of `section_breaks <= index`).
    #[getter]
    fn section(&self) -> usize {
        self.inner.section
    }

    /// `"heading" | "list_item" | "paragraph" | "table"` — the heading
    /// wins over the list (a numbered heading is a heading carrying
    /// numPr).
    #[getter]
    fn kind(&self) -> &str {
        self.inner.kind
    }

    /// The block's visible text (the walk's own rules; `""` for table
    /// blocks — the cells carry the text).
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    /// Headings only: the markdown depth, clamped 1..=6.
    #[getter]
    fn level(&self) -> Option<u8> {
        self.inner.level
    }

    /// Headings only: the ladder's raw 0-based outline level.
    #[getter]
    fn outline_level(&self) -> Option<u8> {
        self.inner.outline_level
    }

    /// The raw `w:pStyle` string.
    #[getter]
    fn style_id(&self) -> Option<String> {
        self.inner.style_id.clone()
    }

    /// The numbering instance, whenever the EFFECTIVE properties carry
    /// `numPr` with `numId != 0` (a numbered heading's included).
    #[getter]
    fn num_id(&self) -> Option<u32> {
        self.inner.num_id
    }

    /// The numbering level.
    #[getter]
    fn ilvl(&self) -> Option<u8> {
        self.inner.ilvl
    }

    /// The resolved number format's own name ("decimal", "bullet", …).
    #[getter]
    fn list_format(&self) -> Option<String> {
        self.inner.list_format.clone()
    }

    /// The level's effective start (startOverride-aware).
    #[getter]
    fn start_number(&self) -> Option<u32> {
        self.inner.start_number
    }

    /// `kind == "table"` only: the grid.
    #[getter]
    fn table(&self) -> Option<DocxTable> {
        self.inner
            .table
            .clone()
            .map(|table| DocxTable { inner: table })
    }

    /// The block's hyperlinks, in order.
    #[getter]
    fn links(&self) -> Vec<DocxLink> {
        self.inner
            .links
            .iter()
            .map(|link| DocxLink {
                inner: link.clone(),
            })
            .collect()
    }

    /// The block's real pictures, in order.
    #[getter]
    fn images(&self) -> Vec<DocxImage> {
        self.inner
            .images
            .iter()
            .map(|image| DocxImage {
                inner: image.clone(),
            })
            .collect()
    }

    /// Structural equality.
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        docx_block_repr(&self.inner)
    }
}

/// One DOCX table's grid.
#[pyclass(module = "_tors_documents")]
pub struct DocxTable {
    inner: documents_impl::DocxTable,
}

#[pymethods]
impl DocxTable {
    /// The grid's row count.
    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows
    }

    /// The grid's column count (`grid.len()`).
    #[getter]
    fn cols(&self) -> usize {
        self.inner.cols
    }

    /// The `w:tblGrid` column widths, twips, verbatim.
    #[getter]
    fn col_widths_twips(&self) -> Vec<i32> {
        self.inner.col_widths_twips.clone()
    }

    /// The present cells, row-major (vMerge-continue cells emitted as
    /// their own cells, inside the Restart span).
    #[getter]
    fn cells(&self) -> Vec<DocxCell> {
        self.inner
            .cells
            .iter()
            .map(|cell| DocxCell {
                inner: cell.clone(),
            })
            .collect()
    }

    /// The `w:tblCaption` accessibility caption.
    #[getter]
    fn caption(&self) -> Option<String> {
        self.inner.caption.clone()
    }

    /// Structural equality (all five fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        docx_table_repr(&self.inner)
    }
}

/// One DOCX table cell.
#[pyclass(module = "_tors_documents")]
pub struct DocxCell {
    inner: documents_impl::DocxCell,
}

#[pymethods]
impl DocxCell {
    /// 0-based row.
    #[getter]
    fn row(&self) -> usize {
        self.inner.row
    }

    /// 0-based starting column (gridSpan accumulation).
    #[getter]
    fn col(&self) -> usize {
        self.inner.col
    }

    /// The cell's text.
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    /// The row span (the vMerge walk; 1 otherwise).
    #[getter]
    fn row_span(&self) -> usize {
        self.inner.row_span
    }

    /// The column span (`w:gridSpan`, clamped 1..=1000).
    #[getter]
    fn col_span(&self) -> usize {
        self.inner.col_span
    }

    /// Structural equality (all five fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        docx_cell_repr(&self.inner)
    }
}

/// One DOCX hyperlink.
#[pyclass(module = "_tors_documents")]
pub struct DocxLink {
    inner: documents_impl::DocxLink,
}

#[pymethods]
impl DocxLink {
    /// The hyperlink's runs' text.
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    /// The external URL (rels-resolved); `None` for an internal link.
    #[getter]
    fn href(&self) -> Option<String> {
        self.inner.href.clone()
    }

    /// The internal bookmark name; `None` for an external link.
    #[getter]
    fn anchor(&self) -> Option<String> {
        self.inner.anchor.clone()
    }

    /// The URI fragment when an external link also declares `w:anchor`.
    #[getter]
    fn fragment(&self) -> Option<String> {
        self.inner.fragment.clone()
    }

    /// The screen-tip tooltip, when declared.
    #[getter]
    fn tooltip(&self) -> Option<String> {
        self.inner.tooltip.clone()
    }

    /// Structural equality (all five fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        docx_link_repr(&self.inner)
    }
}

/// One DOCX picture.
#[pyclass(module = "_tors_documents")]
pub struct DocxImage {
    inner: documents_impl::DocxImage,
}

#[pymethods]
impl DocxImage {
    /// The relationship id (joins to the package's image parts; bytes
    /// never cross this surface).
    #[getter]
    fn relationship_id(&self) -> &str {
        &self.inner.relationship_id
    }

    /// The image part's extension from the package's map ("png", …).
    #[getter]
    fn file_ext(&self) -> Option<String> {
        self.inner.file_ext.clone()
    }

    /// The alt text (`wp:docPr/@descr`).
    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }

    /// The declared width, points (EMU ÷ 12700.0); `None` when the
    /// declared extent is not positive.
    #[getter]
    fn width_pt(&self) -> Option<f32> {
        self.inner.width_pt
    }

    /// The declared height, points (the same law).
    #[getter]
    fn height_pt(&self) -> Option<f32> {
        self.inner.height_pt
    }

    /// True for an inline drawing, false for a floating anchor.
    #[getter]
    fn inline(&self) -> bool {
        self.inner.inline
    }

    /// The `adec:decorative` flag (the alt-less placeholder).
    #[getter]
    fn decorative(&self) -> bool {
        self.inner.decorative
    }

    /// Structural equality (all seven fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        docx_image_repr(&self.inner)
    }
}

/// The XLSX payload (the `document_structure` report's `payload` for an
/// xlsx): thin getters over the core's plain struct; `==` is structural.
#[pyclass(module = "_tors_documents")]
pub struct XlsxStructure {
    inner: documents_impl::XlsxStructure,
}

#[pymethods]
impl XlsxStructure {
    /// The worksheets, sheet order, hidden and macro sheets served.
    #[getter]
    fn sheets(&self) -> Vec<XlsxSheet> {
        self.inner
            .sheets
            .iter()
            .map(|sheet| XlsxSheet {
                inner: sheet.clone(),
            })
            .collect()
    }

    /// The workbook's chart/dialog sheet names.
    #[getter]
    fn chartsheets(&self) -> Vec<String> {
        self.inner.chartsheets.clone()
    }

    /// The workbook-level defined names.
    #[getter]
    fn defined_names(&self) -> Vec<XlsxDefinedName> {
        self.inner
            .defined_names
            .iter()
            .map(|name| XlsxDefinedName {
                inner: name.clone(),
            })
            .collect()
    }

    /// The 1904 date system flag.
    #[getter]
    fn date1904(&self) -> bool {
        self.inner.date1904
    }

    /// Names of sheets whose part could not be parsed (the intact sheets
    /// are served; the loss is on record).
    #[getter]
    fn unreadable_sheets(&self) -> Vec<String> {
        self.inner.unreadable_sheets.clone()
    }

    /// True when cells stopped at `MAX_STRUCTURE_ENTRIES`.
    #[getter]
    fn entries_truncated(&self) -> bool {
        self.inner.entries_truncated
    }

    /// Structural equality (all six fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        xlsx_structure_repr(&self.inner)
    }
}

/// One XLSX worksheet.
#[pyclass(module = "_tors_documents")]
pub struct XlsxSheet {
    inner: documents_impl::XlsxSheet,
}

#[pymethods]
impl XlsxSheet {
    /// The sheet's display name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// `"visible" | "hidden" | "veryHidden"`.
    #[getter]
    fn state(&self) -> &str {
        self.inner.state
    }

    /// The used range, computed from the parsed cells (styled-empty
    /// cells included); `None` for a sheet with no cells.
    #[getter]
    fn used_range(&self) -> Option<String> {
        self.inner.used_range.clone()
    }

    /// The declared `<dimension>` string, verbatim (provenance, never
    /// the used range's source).
    #[getter]
    fn declared_dimension(&self) -> Option<String> {
        self.inner.declared_dimension.clone()
    }

    /// The used range's max row index, 0-based; `None` for an empty
    /// sheet.
    #[getter]
    fn max_row(&self) -> Option<usize> {
        self.inner.max_row
    }

    /// The used range's max column index, 0-based; `None` for an empty
    /// sheet.
    #[getter]
    fn max_col(&self) -> Option<usize> {
        self.inner.max_col
    }

    /// The present cells, row-major, sparse.
    #[getter]
    fn cells(&self) -> Vec<XlsxCell> {
        self.inner
            .cells
            .iter()
            .map(|cell| XlsxCell {
                inner: cell.clone(),
            })
            .collect()
    }

    /// The merged ranges as ref strings.
    #[getter]
    fn merged_cells(&self) -> Vec<String> {
        self.inner.merged_cells.clone()
    }

    /// The sheet's hyperlinks.
    #[getter]
    fn hyperlinks(&self) -> Vec<XlsxHyperlink> {
        self.inner
            .hyperlinks
            .iter()
            .map(|link| XlsxHyperlink {
                inner: link.clone(),
            })
            .collect()
    }

    /// The sheet's cell comments.
    #[getter]
    fn comments(&self) -> Vec<XlsxComment> {
        self.inner
            .comments
            .iter()
            .map(|comment| XlsxComment {
                inner: comment.clone(),
            })
            .collect()
    }

    /// Structural equality (all ten fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        xlsx_sheet_repr(&self.inner)
    }
}

/// One XLSX cell: the value as the file states it, discriminated by
/// `type`, never coerced.
#[pyclass(module = "_tors_documents")]
pub struct XlsxCell {
    inner: documents_impl::XlsxCell,
}

#[pymethods]
impl XlsxCell {
    /// The cell reference ("B3").
    #[getter]
    #[pyo3(name = "ref")]
    fn cell_reference(&self) -> &str {
        &self.inner.reference
    }

    /// 0-based row.
    #[getter]
    fn row(&self) -> usize {
        self.inner.row
    }

    /// 0-based column.
    #[getter]
    fn col(&self) -> usize {
        self.inner.col
    }

    /// `"number" | "string" | "boolean" | "error" | "date" | "empty"` —
    /// `date` is ONLY the `t="d"` ISO-8601 branch (a serial number with
    /// a date format stays `"number"`: the seam cannot decode the
    /// number-format stylesheet).
    #[getter]
    #[pyo3(name = "type")]
    fn cell_type(&self) -> &str {
        self.inner.kind
    }

    /// The value by `type`: bool | float | str | None (shared strings
    /// dereferenced to their FULL text — no markdown-lane truncation).
    #[getter]
    fn value(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.inner.value {
            None => py.None().into_any(),
            Some(documents_impl::XlsxScalar::Bool(value)) => value
                .into_pyobject(py)
                .unwrap()
                .as_any()
                .clone()
                .unbind()
                .into_any(),
            Some(documents_impl::XlsxScalar::Number(value)) => value
                .into_pyobject(py)
                .unwrap()
                .as_any()
                .clone()
                .unbind()
                .into_any(),
            Some(documents_impl::XlsxScalar::Text(value)) => value
                .into_pyobject(py)
                .unwrap()
                .as_any()
                .clone()
                .unbind()
                .into_any(),
        }
    }

    /// The `<f>` formula text, verbatim.
    #[getter]
    fn formula(&self) -> Option<String> {
        self.inner.formula.clone()
    }

    /// Formula cells: whether a cached value rode the file.
    #[getter]
    fn has_cached_value(&self) -> bool {
        self.inner.has_cached_value
    }

    /// Structural equality (all seven fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        xlsx_cell_repr(&self.inner)
    }
}

/// One defined name.
#[pyclass(module = "_tors_documents")]
pub struct XlsxDefinedName {
    inner: documents_impl::XlsxDefinedName,
}

#[pymethods]
impl XlsxDefinedName {
    /// The name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// The formula/reference value, verbatim.
    #[getter]
    fn value(&self) -> &str {
        &self.inner.value
    }

    /// The sheet the name is scoped to, 0-based into the WORKBOOK's own
    /// sheet list (chartsheets included), verbatim; `None` for a
    /// workbook-level name.
    #[getter]
    fn local_sheet_id(&self) -> Option<u32> {
        self.inner.local_sheet_id
    }

    /// That sheet's display name, resolved from the workbook's sheet
    /// list when in range.
    #[getter]
    fn local_sheet_name(&self) -> Option<String> {
        self.inner.local_sheet_name.clone()
    }

    /// The name's hidden flag.
    #[getter]
    fn hidden(&self) -> bool {
        self.inner.hidden
    }

    /// Structural equality (all five fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        xlsx_defined_name_repr(&self.inner)
    }
}

/// One XLSX sheet hyperlink.
#[pyclass(module = "_tors_documents")]
pub struct XlsxHyperlink {
    inner: documents_impl::XlsxHyperlink,
}

#[pymethods]
impl XlsxHyperlink {
    /// The cell the link sits on.
    #[getter]
    fn cell_ref(&self) -> &str {
        &self.inner.cell_ref
    }

    /// The external URL; `None` for an internal location.
    #[getter]
    fn href(&self) -> Option<String> {
        self.inner.href.clone()
    }

    /// The internal location; `None` for a URL.
    #[getter]
    fn anchor(&self) -> Option<String> {
        self.inner.anchor.clone()
    }

    /// The tooltip, when declared.
    #[getter]
    fn tooltip(&self) -> Option<String> {
        self.inner.tooltip.clone()
    }

    /// Structural equality (all four fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        xlsx_hyperlink_repr(&self.inner)
    }
}

/// One XLSX cell comment.
#[pyclass(module = "_tors_documents")]
pub struct XlsxComment {
    inner: documents_impl::XlsxComment,
}

#[pymethods]
impl XlsxComment {
    /// The cell the comment is attached to.
    #[getter]
    fn cell_ref(&self) -> &str {
        &self.inner.cell_ref
    }

    /// The comment's author, when the part names one.
    #[getter]
    fn author(&self) -> Option<String> {
        self.inner.author.clone()
    }

    /// The comment's text.
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    /// Structural equality (all three fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        xlsx_comment_repr(&self.inner)
    }
}

/// The HTML payload (the `document_structure` report's `payload` for
/// html): thin getters over the core's plain struct; `==` is structural.
#[pyclass(module = "_tors_documents")]
pub struct HtmlStructure {
    inner: documents_impl::HtmlStructure,
}

#[pymethods]
impl HtmlStructure {
    /// The nodes, reading order, parents before children.
    #[getter]
    fn nodes(&self) -> Vec<HtmlNode> {
        self.inner
            .nodes
            .iter()
            .map(|node| HtmlNode {
                inner: node.clone(),
            })
            .collect()
    }

    /// True when nodes stopped at `MAX_STRUCTURE_ENTRIES` (the kept
    /// prefix stays index-sound; children filtered to kept indices).
    #[getter]
    fn entries_truncated(&self) -> bool {
        self.inner.entries_truncated
    }

    /// Structural equality (both fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        html_structure_repr(&self.inner)
    }
}

/// One HTML node (the nine reachable kinds).
#[pyclass(module = "_tors_documents")]
pub struct HtmlNode {
    inner: documents_impl::HtmlNode,
}

#[pymethods]
impl HtmlNode {
    /// The crate's deterministic node id.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// The parent node's index (`None` for roots) — always earlier.
    #[getter]
    fn parent(&self) -> Option<u32> {
        self.inner.parent
    }

    /// The children's indices, reading order.
    #[getter]
    fn children(&self) -> Vec<u32> {
        self.inner.children.clone()
    }

    /// The crate's per-node attribute map — headings carry `{"id": …}`
    /// when the source heading had one; image nodes carry `None` (the
    /// collector drops img attributes: declared dimensions are NOT
    /// served — the disclosed v1 gap).
    #[getter]
    fn attributes(&self) -> Option<std::collections::HashMap<String, String>> {
        self.inner.attributes.clone()
    }

    /// The node kind: `"heading" | "paragraph" | "list" | "list_item" |
    /// "table" | "image" | "code" | "quote" | "group"` — the nine the
    /// live collector emits (an unreachable variant is a typed refusal,
    /// never a silent kind=None).
    #[getter]
    fn kind(&self) -> &str {
        self.inner.kind
    }

    /// The node's text (heading/paragraph/list_item/code).
    #[getter]
    fn text(&self) -> Option<String> {
        self.inner.text.clone()
    }

    /// Heading level 1-6 (a Group: the level that created it).
    #[getter]
    fn level(&self) -> Option<u8> {
        self.inner.level
    }

    /// List: whether ordered.
    #[getter]
    fn ordered(&self) -> Option<bool> {
        self.inner.ordered
    }

    /// Code: the language, when declared.
    #[getter]
    fn language(&self) -> Option<String> {
        self.inner.language.clone()
    }

    /// Group: the section label — the creating heading's text (the
    /// crate's `label` and `heading_text` are the same string).
    #[getter]
    fn label(&self) -> Option<String> {
        self.inner.label.clone()
    }

    /// Image: the src URL.
    #[getter]
    fn src(&self) -> Option<String> {
        self.inner.src.clone()
    }

    /// Image: the alt text.
    #[getter]
    fn alt(&self) -> Option<String> {
        self.inner.alt.clone()
    }

    /// Table: the grid.
    #[getter]
    fn grid(&self) -> Option<HtmlGrid> {
        self.inner.grid.clone().map(|grid| HtmlGrid { inner: grid })
    }

    /// Inline annotations, byte offsets into `text`.
    #[getter]
    fn annotations(&self) -> Vec<HtmlAnnotation> {
        self.inner
            .annotations
            .iter()
            .map(|annotation| HtmlAnnotation {
                inner: annotation.clone(),
            })
            .collect()
    }

    /// Structural equality (all fourteen fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        html_node_repr(&self.inner)
    }
}

/// One HTML table grid.
#[pyclass(module = "_tors_documents")]
pub struct HtmlGrid {
    inner: documents_impl::HtmlGrid,
}

#[pymethods]
impl HtmlGrid {
    /// The grid's row count.
    #[getter]
    fn rows(&self) -> u32 {
        self.inner.rows
    }

    /// The grid's column count.
    #[getter]
    fn cols(&self) -> u32 {
        self.inner.cols
    }

    /// The origin cells, ordered by (row, col), sparse under spans.
    #[getter]
    fn cells(&self) -> Vec<HtmlGridCell> {
        self.inner
            .cells
            .iter()
            .map(|cell| HtmlGridCell {
                inner: cell.clone(),
            })
            .collect()
    }

    /// Structural equality (all three fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        html_grid_repr(&self.inner)
    }
}

/// One HTML grid cell.
#[pyclass(module = "_tors_documents")]
pub struct HtmlGridCell {
    inner: documents_impl::HtmlGridCell,
}

#[pymethods]
impl HtmlGridCell {
    /// The cell's text.
    #[getter]
    fn content(&self) -> &str {
        &self.inner.content
    }

    /// 0-based row.
    #[getter]
    fn row(&self) -> u32 {
        self.inner.row
    }

    /// 0-based column.
    #[getter]
    fn col(&self) -> u32 {
        self.inner.col
    }

    /// The row span (1 unless a merge covers).
    #[getter]
    fn row_span(&self) -> u32 {
        self.inner.row_span
    }

    /// The column span (1 unless a merge covers).
    #[getter]
    fn col_span(&self) -> u32 {
        self.inner.col_span
    }

    /// Whether this is a header cell (`<th>`).
    #[getter]
    fn is_header(&self) -> bool {
        self.inner.is_header
    }

    /// Structural equality (all six fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        html_grid_cell_repr(&self.inner)
    }
}

/// One inline annotation on an HTML node's text.
#[pyclass(module = "_tors_documents")]
pub struct HtmlAnnotation {
    inner: documents_impl::HtmlAnnotation,
}

#[pymethods]
impl HtmlAnnotation {
    /// The annotation kind ("link", "bold", …).
    #[getter]
    fn kind(&self) -> &str {
        self.inner.kind
    }

    /// Start byte offset into the node's text (inclusive).
    #[getter]
    fn start(&self) -> u32 {
        self.inner.start
    }

    /// End byte offset (exclusive).
    #[getter]
    fn end(&self) -> u32 {
        self.inner.end
    }

    /// Link: the href, verbatim (no decoding, no resolution).
    #[getter]
    fn url(&self) -> Option<String> {
        self.inner.url.clone()
    }

    /// Link: the `title` attribute, when declared.
    #[getter]
    fn title(&self) -> Option<String> {
        self.inner.title.clone()
    }

    /// Structural equality (all five fields).
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        html_annotation_repr(&self.inner)
    }
}

/// `tors.documents.pdf_link_uris(path, data=None, password=None,
/// backend="auto", max_bytes=None) -> list[list[str]]`: or
/// `pdf_link_uris(data=...)`: the
/// `/Annots` link walk: for every page,
/// the URIs of its `/Subtype /Link` annotations whose action is a URI
/// (`/A << /S /URI /URI (...) >>`), in annotation order: one `list[str]`
/// per page, page order, empty lists for pages without link annotations.
///
/// Why this exists alongside the markdown's inline `[text](uri)` links:
/// the two answer different questions. The markdown carries links whose
/// visible text belongs in prose; this walk carries the raw URI list:
/// including URIs behind link rectangles whose text is not itself a link
/// (a "click here" button, an image, a bare rectangle) which no text
/// rendering surfaces at all. The caller that motivated it measured the
/// difference on real resumes: 60 documents, 16 links from the text layer,
/// 34 from the annotations: a quarter of candidates gained a
/// LinkedIn/GitHub URL no text shape would show. Deduplication and
/// canonicalization are deliberately not done here: the verbatim
/// engine-read list is the honest output, and callers canonicalize
/// differently (per-page review panels vs whole-document projections).
///
/// Scope, narrow: URI actions only. `GoTo` (in-document
/// navigation) and `GoToR` (a remote file) contribute nothing and are not
/// fabricated into `file://` shapes. Non-link annotations are skipped by
/// the subtype filter; malformed annotation dictionaries are skipped by
/// pdf_oxide (its parse tolerates them) rather than poisoning the page.
///
/// `backend` names the engine lane, the same vocabulary the conversion
/// pair takes: `"auto"` (the default; the native layer takes `None` as
/// the same choice) and `"oxide"` both run pdf_oxide:
/// the mapping `engine_for` makes for PDF ("oxide" is the oxide-family
/// engine for this format), byte-identical output either way (the
/// annotation walk is pdf_oxide's reader). `"anydoc"` is refused before
/// any work runs, a named `ValueError` raised under the GIL ahead of the
/// detach and the encrypted door check: a capability refusal, not the
/// format-level UnsupportedBackend (PDF+anydoc is a usable conversion
/// pair): anydoc's PDF surface is whole-document markdown only
/// (~anydoc-0.2.4/src/formats/pdf.rs), and it has no annotation surface
/// at all to walk. Whole-document conversion is the conversion pair's
/// call: `to_markdown`/`to_text` with `backend="anydoc"`
/// (`NeedsOcrError` is that lane's scanned-page signal).
///
/// `max_bytes` is the caller's input budget in bytes (a positive `int`).
/// An explicit budget binds this family pre-read exactly as it binds the
/// conversion pair, the shared source-spine gate: the `path=`'s size at
/// the stat and again on the open handle, the `data=` length before the
/// copy; never a byte over budget is read or copied, and it overrides the
/// default in either direction. `None` (the default) still runs no
/// lane-specific policy on the pdf lane, the unchanged doctrine: the
/// 32 MiB default ceiling is the core's post-read check on the
/// amplified lanes (anydoc/office_oxide/html), lanes these PDF-only calls never run
/// (pdf_oxide's own resource limits govern here); the pdf lane instead
/// reads under [`MAX_INPUT_READ`], the finite 512 MiB backstop, a
/// memory-safety floor rather than a lane policy.
///
/// Errors: `OSError` (matched subclass) for a missing/unreadable `path=`;
/// `ValueError` for bytes that do not parse as a PDF, and for a
/// FIFO/device/socket `path=` (regular files only, the shared source
/// spine); the capability `ValueError` for `backend="anydoc"` (under the
/// GIL, before any work runs, so an encrypted document under it
/// surfaces the backend refusal, never the door-check error); the
/// ceiling `ValueError` for an input over an explicit `max_bytes` (both
/// sizes named, refused before the read or copy). Same GIL model as
/// `pdf_classify`: the source marshalling under the GIL, the
/// read+page-tree+annotation walk under one `py.detach`, the nested list
/// marshalling (O(annotations)) after the reacquire.
#[pyfunction(signature = (
    path = None,
    data = None,
    password = None,
    backend = None,
    max_bytes = None,
))]
pub fn pdf_link_uris(
    py: Python<'_>,
    path: Option<&Bound<'_, PyAny>>,
    data: Option<&Bound<'_, PyAny>>,
    password: Option<&Bound<'_, PyAny>>,
    backend: Option<&Bound<'_, PyAny>>,
    max_bytes: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<Vec<String>>> {
    let source = parse_source(path, data, "pdf_link_uris")?;
    let password = parse_password(password)?;
    let backend = parse_backend(backend)?;
    if backend == Backend::Anydoc {
        return Err(refuse_anydoc(
            "the /Annots link walk (it has no annotation surface)",
        ));
    }
    let max_bytes = parse_max_bytes(max_bytes)?;
    py.detach(move || {
        let bytes = source.into_bytes(max_bytes)?;
        tors::pdf_impl::link_uris(bytes, password.as_deref()).map_err(PdfLaneError::Pdf)
    })
    .map_err(pdf_lane_error)
}

/// `tors.documents.to_markdown(path, data=None, format=None,
/// backend="auto", pages=None, password=None, max_bytes=None)
/// -> (format, markdown)` (or `to_markdown(data=...)`): convert
/// any working-format document: pdf, doc/docx, xls/xlsx, ppt/pptx, rtf,
/// odt/ods/odp, epub, csv, html, to GitHub-Flavored Markdown, on the
/// measured-best engine for its format. Exactly one of `path` (a file, the
/// only positional) or `data` (the document's bytes: the in-memory
/// caller's entry, no temp-file roundtrip; a `data=` call has no name to
/// consult, so format resolution rests on `format=` and the content
/// markers alone, `sniff`'s doctrine): passing both or neither raises
/// before any work runs.
///
/// `format` names the format explicitly (extension spelling, no dot,
/// case-insensitive: `"pdf"`, `"html"`, `"docx"`, `"xlsx"`, `"pptx"`,
/// `"doc"`, `"xls"`, `"ppt"`, `"rtf"`, `"odt"`, `"ods"`, `"odp"`,
/// `"epub"`, `"csv"`, plus container variants like `"docm"`/`"xlsm"`).
/// `None` (the default) sniffs it from the file's content markers: the
/// PDF header, the RTF open group, OLE stream names, the ZIP package
/// mimetype, the HTML document marker: with the path's extension as the
/// fallback signature-less formats (CSV) need, so a mislabeled file still
/// converts correctly. The returned `format` is the name the conversion
/// actually used.
///
/// `password` unlocks an encrypted PDF (the PDF kinds only: a password
/// on any other format is a `ValueError`, never a silently-ignored
/// argument). Without one, every PDF entry fails closed: the door check
/// at open raises, never empty output masquerading as "no content" (the
/// pre-fix shape, measured on an RC4-128 fixture: the markdown
/// and text surfaces returned `""` on a locked document while classify
/// raised). A wrong password is its own clean `ValueError`.
///
/// `max_bytes` is the caller's input budget in bytes (a positive `int`).
/// An explicit budget is binding on every engine lane: it is enforced
/// before a byte is read (a `path=`'s size at open, a `data=` call's
/// length): the engine lane is unknowable before the container sniff
/// that these very bytes feed, so a budget that waited for the lane to
/// be known would be a budget the pdf/HTML lanes never see (the pre-fix
/// shape, probed: `max_bytes=1` on a real PDF converted
/// unmolested; and worse, `to_text(path="/dev/zero", max_bytes=65536)`
/// read forever: the ceiling ran only in the core, only after the
/// read). `None` (the default) keeps the default-ceiling doctrine
/// exactly as before: the measured 32 MiB, checked after the read, on
/// the lanes that amplify input into resident memory: anydoc
/// (~36x on benign csv shapes but ~146x worst case on adversarial ones (a
/// 24 MiB many-short-cells csv peaked at 3.4 GiB, stable across sizes,
/// so the 32 MiB default ceiling budgets ~4.6 GiB worst case: size
/// workers for that or pass a lower `max_bytes`), office_oxide
/// 0.1.10 (per-part decompression caps of 512 MiB per part, declared
/// and actual, XML depth 256 on a 16 MiB parse stack) but still has no
/// total-across-parts cap and no output cap (a 399 KiB zip carrying a
/// 400 MiB part converts at ~1.6 GiB peak; a 600 MiB part is refused
/// pre-decompression), and HTML (~23x input: the converter holds the
/// whole input and output at once; a 48 MiB doctype HTML peaked at
/// 1118 MiB, measured). The pdf lane runs no lane-specific 32 MiB
/// policy under the default (pdf_oxide's own resource limits govern
/// there), but it no longer reads unbounded either: it reads under
/// [`MAX_INPUT_READ`], the finite 512 MiB backstop, a memory-safety
/// floor rather than a lane policy.
///
/// `backend` picks the engine where they overlap: `"auto"` (the default;
/// the native layer takes `None` as the same choice) routes by the
/// measured table in `tors-core`'s `documents_impl`: PDF to pdf_oxide (two-column reading order, link annotations, line structure),
/// HTML to html-to-markdown-rs (script/style dropped by construction), the
/// office and text formats to anydoc (style-based docx headings, list
/// markers, rtf/odt/epub/csv coverage); `"oxide"` forces the oxide family
/// (pdf_oxide for PDF, office_oxide for docx/xlsx/pptx and legacy
/// doc/xls/ppt) and `"anydoc"` forces anydoc: both raise `ValueError` on
/// a format the forced engine cannot read, never a silent fallback.
///
/// `pages` selects a PDF page subset (PDF on the pdf_oxide lane only, else
/// `ValueError`): a single `int` (one 0-based page), a `list` of ints (the
/// explicit set), or a 2-tuple `(start, stop)` (a half-open range, Python
/// convention: `(1, 3)` is the second and third pages; indices are
/// 0-based). The selection is deduped into
/// document order and the per-page conversions are joined with pdf_oxide's
/// own inter-page separator, so a full range is byte-identical to the
/// whole-document conversion.
///
/// Errors: argument refusals raise under the GIL, before any work runs,
/// each naming the argument and repr'ing the offending value:
/// `TypeError` for a wrong-typed `format=`/`backend=`/`path`/`pages=`
/// (a non-string format or backend; a bool, float, or str where a page
/// index belongs), `ValueError` for a wrong value or shape (an unknown
/// or empty `backend=` name; a negative, empty, or backwards `pages=`
/// range, a range endpoint no PDF can carry (past the 8,388,607-object
/// cross-reference ceiling, refused before the range is built); a
/// non-Unicode path; a path with an embedded NUL byte:
/// CPython's own `open()` refusal). The input-side and document
/// failures are constructed after the GIL is reacquired: `OSError` for
/// a missing/unreadable file (the matched subclass:
/// `FileNotFoundError`, `IsADirectoryError`: directories keep their
/// pinned shape); `ValueError` naming `path` and the kind for a
/// FIFO/device/socket `path=` (regular files only, so the unbounded
/// `/dev/zero` read and the no-writer FIFO block can never begin), and
/// the ceiling `ValueError`: both sizes named, the core's own message
/// on the metered lanes: for any input over an explicit `max_bytes`
/// (refused before the read or copy); `NeedsOcrError` (a `ValueError`
/// subclass carrying `.pages` and `.page_count`) when the anydoc
/// backend hits a PDF with scanned pages: route the document to OCR;
/// `ValueError` for an unknown format name, an undetectable file, an
/// unusable backend/format pair, an out-of-bounds `pages=` selection
/// (the core's message names the page count), or a
/// malformed/encrypted/over-limit document.
///
/// GIL model: the whole argument marshalling (path, format, backend,
/// pages: validation and normalization) under the GIL, the whole
/// read+sniff+convert pass under one `py.detach`, then the two-string
/// tuple marshalling, O(output).
// pyo3's #[pyfunction] needs the Python signature flat: one Rust
// parameter per Python argument, no struct grouping possible, so the
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
/// -> (format, text)` (or `to_text(data=...)`): the same conversion,
/// routing, sniffing, source, `pages=`, `password=`, and `max_bytes=`
/// semantics as `to_markdown`,
/// with the markdown normalized to
/// plain text: one text shape for every format and engine (headings keep
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

/// The core conversion entry points `convert` routes between: the exact
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

/// The marshalling spine `to_markdown` and `to_text` share: they differ
/// in nothing but which core function the detached pass calls (`to_text`'s
/// strip runs inside the core, so the binding sees one shape for both):
/// marshal and validate every argument under the GIL (`parse_path`,
/// `parse_format`, `parse_backend`, `parse_pages`, every refusal naming
/// its argument: repr'ing the value, except the content-bearing
/// `data=`, which names the type only), run the whole native pass: file
/// read (regular files only, under an explicit budget), format sniff,
/// engine conversion: under one `py.detach`, then
/// the error mapping and the two-string return after the GIL is
/// reacquired. `what` is the caller's own name (`"to_markdown"`/
/// `"to_text"`), riding into `parse_source`'s both/neither refusals so
/// each function's messages are attributed to itself: the shared spine
/// must not hardcode one twin's name into the other's errors. The
/// contract docstrings stay per-function: they are the surface callers
/// read. (The argument count mirrors the two pyfunction signatures it
/// marshals: see to_markdown's allow rationale.)
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
    let source = parse_source(path, data, what)?;
    let format = parse_format(format)?;
    let backend = parse_backend(backend)?;
    // The pages= shape refusals (empty selection, negative, non-index
    // types) are the BINDING's, parsed under the GIL here — BEFORE the
    // detach and so before the encrypted door: A10's order is two-tier
    // (measured: pages=[] on a locked document answers the shape
    // refusal, not the encryption refusal). The core's range
    // validation (an index past the page count) runs post-open, after
    // the door. The pdf_impl docstring states the PDF leg alone.
    let pages = parse_pages(pages)?;
    let password = parse_password(password)?;
    let max_bytes = parse_max_bytes(max_bytes)?;
    let converted = match py.detach(move || {
        let (bytes, hint) = source
            .into_input(max_bytes, format, backend)
            .map_err(input_document_error)?;
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
/// content-marker format detector: what `to_markdown`/`to_text` would
/// resolve these bytes to from content alone, with no path and no
/// extension: the PDF header, the RTF open group, OLE stream names, the
/// ZIP package mimetype, the HTML document marker. The routing caller's
/// mislabeled-download answer: the bytes' verdict overrides any label the
/// download carried.
///
/// `None` is not an error: the content names no format (a signature-less
/// text format such as CSV (name it via `format=`) or not a document at
/// all).
///
/// Cost: `sniff` is not a bounded marker scan. anydoc's detect
/// opens the ZIP/OLE package and parses its metadata (rels and
/// content-types, with a fallback into the main part when the markers
/// need it), so a classification-only pipeline pays a container-level
/// parse of the package, measured: a 120 KiB zip peaked at
/// 267 MiB RSS to answer "docx". Budget a `sniff` call like a small
/// open, not like a header peek. (A `max_bytes` knob for `sniff` alone
/// is deliberately out of scope: the convert lane's ceiling is the
/// convert lane's.)
///
/// `data=` is strict `bytes`: anything else is a `TypeError` naming the
/// argument, the house convention (the raw pyo3 cast error the manual
/// marshal replaces, `'bytearray' object is not an instance of 'bytes'`,
/// probed on the 0.5.0 wheel, named neither); `bytearray`/`memoryview`
/// callers pass `bytes(data)`, the same strict contract as the convert
/// lane's `data=`.
///
/// GIL model: the bytes-in zero-copy borrow under the GIL, the
/// detection pass (the package parse the cost paragraph above names)
/// under `py.detach`, and the `str | None` return (the name is `&'static`
/// (the vocabulary's own spelling) so no owned String is built just to
/// be copied into the Python str): the bytes-in family's
/// `utf8_is_valid` residue class, no marshalling beyond the answer itself.
#[pyfunction]
pub fn sniff(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Option<&'static str>> {
    // The manual marshal keeps the refusal in the house message style; the
    // zero-copy borrow of the immutable buffer then crosses the detach for
    // the scan (sound by construction: nothing can mutate it GIL-free).
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
/// Unicode and carry no NUL. All three refusal shapes name `path` and
/// repr the value: the raw pyo3 `&str`-conversion failures they replace
/// do not: a non-str path surfaces as a bare `'int' object is not an
/// instance of 'str'` that leaves a four-argument call site guessing which
/// argument failed, a lone-surrogate path (a filename that escaped a
/// POSIX-only tool) as a `UnicodeEncodeError` about 'utf-8' that reads
/// like a conversion bug in the document engine, and a NUL-bearing path
/// as a plain `OSError` from the filesystem layer where CPython's own
/// `open("a\0b")` raises `ValueError: embedded null byte` (all probed on
/// the 0.5.0 wheel / the PR #27 red-team pass).
fn parse_path(path: &Bound<'_, PyAny>) -> PyResult<PathBuf> {
    let string = path.cast::<PyString>().map_err(|_| {
        PyTypeError::new_err(format!(
            "path must be a str or os.PathLike, not {}",
            py_repr(path)
        ))
    })?;
    match string.to_str() {
        Ok(text) if text.contains('\0') => Err(PyValueError::new_err(format!(
            "path must be a valid filesystem path (no embedded null byte), not {}",
            py_repr(path)
        ))),
        Ok(text) => Ok(PathBuf::from(text)),
        Err(_) => Err(PyValueError::new_err(format!(
            "path must be valid Unicode (no lone surrogates), not {}",
            py_repr(path)
        ))),
    }
}

/// The document's input: a `path=` the detached pass reads (regular
/// files only: [`Source::into_input`] refuses the other kinds before
/// open(2)/read can begin), or the caller's `data=` bytes, borrowed
/// zero-copy from the caller's `PyBytes`.
///
/// The borrow crossing `py.detach` is sound by construction: `sniff`'s
/// precedent, the same argument restated: the buffer is immutable
/// (nothing can mutate a `bytes` object GIL-free), and CPython holds the
/// caller's argument reference for the whole call, so the slice outlives
/// the detached pass that reads it. The one copy (`to_vec`) runs inside
/// the detach, where it belongs: measured, a 400 MB `data=`
/// call held the GIL ~78 ms when the copy ran parse-side: a 1 GiB
/// upload is a 1 GiB GIL-held interpreter stall that way, and pure
/// memory bandwidth inside the detach. Do not extend this pattern to a
/// mutable buffer (bytearray/memoryview): the immutability is the whole
/// safety argument, which is why `parse_source` refuses those on the
/// spot (strict `bytes`) and must keep refusing them.
///
/// `into_input` also yields the name hint the extension fallback of
/// format resolution consults: the path, when there was one; a `data=`
/// call has no name and rests on `format=`/content markers alone, the
/// doctrine `sniff` encodes (the core's `resolve` says the same, in its
/// own words).
enum Source<'a> {
    Path(PathBuf),
    Data(&'a [u8]),
}

/// The binding-layer input refusals: the shapes an `io::Error` cannot
/// carry, because they are typed value refusals, not environment
/// failures: `Io` maps to `OSError` (pyo3's matched-subclass
/// `From<io::Error>`, the existing doctrine), `Refused` to `ValueError`
/// with the message pre-built where the refusal fired (it names the
/// argument and the reason, the house convention). Returned as a value
/// out of the detached pass (nothing raises from inside the detach)
/// and mapped to a `PyErr` after the GIL is reacquired.
enum InputError {
    Io(std::io::Error),
    Refused(String),
}

/// The finite backstop the unmetered pdf lane reads under when the
/// caller passed no `max_bytes`: the two-phase read's `None` ceiling once the
/// prefix sniff positively resolves to pdf (the only unmetered lane; every
/// other lane meters at the 32 MiB default). Large enough to be no practical
/// limit on a real document, finite enough that a lying stat or a sparse hole
/// can no longer make the read unbounded.
pub(crate) const MAX_INPUT_READ: usize = 512 * 1024 * 1024;
/// The prefix read before the content-only lane guess: 64 KiB is enough for
/// every content marker `resolve` consults, and is far under any ceiling the
/// guess can pick (>= 32 MiB), so the prefix read is never itself "over".
const SNIFF_PREFIX: usize = 64 * 1024;

impl Source<'_> {
    /// The bytes alone: the PDF family's entry (no name to consult:
    /// those functions are PDF-only by construction). `max_bytes` is the
    /// caller's explicit budget, binding pre-read on this family exactly
    /// as on the conversion pair: the same [`Source::into_input`] gate
    /// runs: the path stat, the open-fd fstat, and the `data=` slice,
    /// all before a byte is read or copied, because a budget that
    /// skipped this family would be a budget the pdf lane never saw (the
    /// `max_bytes=` the four functions now take had nowhere to land
    /// before). `None` runs no lane-specific policy on the pdf lane, the
    /// unchanged doctrine: the 32 MiB default is the core's post-read
    /// check on the amplified lanes (anydoc/office_oxide/html): lanes
    /// these PDF-only calls never run; the pdf lane instead reads under
    /// [`MAX_INPUT_READ`], the finite 512 MiB backstop.
    fn into_bytes(self, max_bytes: Option<usize>) -> Result<Vec<u8>, InputError> {
        // The PDF-only calls have no format/backend to sniff by; a real PDF
        // resolves to the unmetered lane and gets the MAX_INPUT_READ fallback,
        // which is the correct ceiling for them.
        self.into_input(max_bytes, None, Backend::Auto)
            .map(|(b, _)| b)
    }

    /// The bytes and the name hint: the convert spine's entry (the
    /// extension fallback's last resort; `None` for a `data=` call).
    ///
    /// `max_bytes` is the caller's explicit budget (the absurd values
    /// were already refused under the GIL by `parse_max_bytes`):
    /// `Some(limit)` is binding here, before a byte is read or copied,
    /// on every lane: the engine lane is unknowable before the
    /// container sniff that these very bytes feed, so an explicit budget
    /// cannot wait for the lane to be known (the red-team findings:
    /// `max_bytes=65536` on `/dev/zero` read forever because the ceiling
    /// ran only post-read in the core, and a caller budgeting on a PDF
    /// got the full parse: that lane never saw the
    /// knob). `None` keeps the core's post-read 32 MiB check on the
    /// amplified lanes (anydoc, office_oxide, and now HTML, which
    /// measures ~23x input) exactly as before; the pdf lane runs no
    /// lane-specific policy, but reads under
    /// [`MAX_INPUT_READ`], the finite 512 MiB backstop this function
    /// enforces via the two-phase read below, rather than unbounded.
    fn into_input(
        self,
        max_bytes: Option<usize>,
        format: Option<&str>,
        backend: Backend,
    ) -> Result<(Vec<u8>, Option<String>), InputError> {
        self.read_input(max_bytes, format, backend, Spine::Conversion)
    }

    /// The bytes and the name hint: the GEOMETRY probe's entry. The
    /// divergence from [`Source::into_input`] is two halves, both keyed
    /// on the geometry capability table (the locked §6, which says the
    /// read metering keys on the table the surface actually routes by):
    ///
    /// - the provisional read ceiling (the `None` budget's two-phase
    ///   guess) keys on [`documents_impl::provisional_geometry_read_ceiling`]
    ///   — only the office row meters at the 32 MiB default; the pdf row
    ///   and the no-geometry row read under [`MAX_INPUT_READ`], so a
    ///   40 MiB HTML answers `[None]` on `path=` exactly as on `data=`
    ///   (the parity the conversion table's ceiling broke), and a 34 MiB
    ///   PDF is read (its own limits govern) instead of refused by a
    ///   lane it never runs.
    /// - the no-read door: an explicit `format=` naming a no-geometry
    ///   format settles the answer before any byte — the core's resolve
    ///   short-circuits on the name, the geometry table runs no engine
    ///   on the row, and the entry is the format-determined `[None]` —
    ///   so the regular-file gates still run (a missing path is the same
    ///   `OSError`, a FIFO the same refusal) but nothing is read. A
    ///   directory keeps its pinned EISDIR read refusal, so only regular
    ///   files pass.
    fn into_geometry_input(
        self,
        max_bytes: Option<usize>,
        format: Option<&str>,
        backend: Backend,
    ) -> Result<(Vec<u8>, Option<String>), InputError> {
        self.read_input(max_bytes, format, backend, Spine::Geometry)
    }

    /// The bytes and the name hint: the STRUCTURE surface's entry. No
    /// no-read door: every served row consumes the bytes (the office
    /// legs parse the container, the html lane parses the markup), and
    /// every refused row is refused earlier — the binding-level
    /// capability probe (fired for every backend, from `format=`/the 64
    /// KiB prefix/extension alone) rejects them before this read runs.
    /// Reusing either sibling spine's read is the trap the review
    /// closed: `into_geometry_input`'s no-read door would answer
    /// `format="html"` with ZERO bytes (html is a no-geometry format —
    /// an empty structure where the real one lives), and the conversion
    /// spine's ceiling treats pdf as unmetered and meters refused rows,
    /// so a 40 MiB odt would get a ceiling refusal instead of "no
    /// structure surface for odt". Every served row meters at the 32
    /// MiB default; an explicit budget binds pre-read as everywhere;
    /// the zero value threads through (CoreSide).
    fn into_structure_input(
        self,
        max_bytes: Option<usize>,
        format: Option<&str>,
        backend: Backend,
    ) -> Result<(Vec<u8>, Option<String>), InputError> {
        self.read_input(max_bytes, format, backend, Spine::Structure)
    }

    /// [`Source::into_input`]/[`Source::into_geometry_input`]'s shared
    /// body, keyed by which routing table meters the read.
    fn read_input(
        self,
        max_bytes: Option<usize>,
        format: Option<&str>,
        backend: Backend,
        spine: Spine,
    ) -> Result<(Vec<u8>, Option<String>), InputError> {
        match self {
            Source::Path(path) => {
                // parse_path already validated the Unicode, so the hint is
                // always representable; `.to_str()` cannot fail here.
                let hint_name = path.to_str().map(str::to_string);
                // The fast refusal, before open(2) is ever entered: a FIFO
                // with no writer blocks inside open (the GIL released, the
                // thread unreclaimable), and /dev/zero reads unboundedly
                // once opened, so the exotic kinds must be turned away at
                // the stat, not at the open. This is the cheap gate; the
                // fstat below is the authority.
                let pre = std::fs::metadata(&path).map_err(InputError::Io)?;
                refuse_non_regular(&path, pre.file_type())?;
                refuse_over_ceiling(max_bytes, Some(pre.file_type()), pre.len())?;
                // The authority: fstat on the open handle; whatever a
                // stat/open race did to the path in between, this is the
                // file the read below actually reads, and it must be a
                // regular file under the same budget. (Directories
                // deliberately flow past both checks: their pinned
                // refusal is the read's own EISDIR OSError, unchanged.)
                let mut file = std::fs::File::open(&path).map_err(InputError::Io)?;
                let meta = file.metadata().map_err(InputError::Io)?;
                refuse_non_regular(&path, meta.file_type())?;
                refuse_over_ceiling(max_bytes, Some(meta.file_type()), meta.len())?;

                // The geometry spine's no-read door (the method's doc):
                // an explicit format name that settles the row as
                // no-geometry answers without the file. Directories fall
                // through to the read — the EISDIR refusal is the read's
                // own, and the door must not swallow it.
                if spine == Spine::Geometry
                    && !meta.is_dir()
                    && format.is_some_and(documents_impl::is_no_geometry_format)
                {
                    return Ok((Vec::new(), hint_name));
                }

                // The stat is no longer the authority on how much to read: a
                // lying stat (procfs' st_size=0) or a sparse hole made the
                // old read_to_end unbounded. The bounded read is the ceiling.
                let (bytes, over, used_ceiling) = match max_bytes {
                    // Explicit budget: it is the ceiling, no sniff needed.
                    Some(limit) => {
                        if limit == 0 {
                            // A zero budget binds maximally — nothing of
                            // the document is read — but the value itself
                            // is the CORE's sequenced refusal on this
                            // surface (§6: after the capability and
                            // password-kind checks), so the sniff prefix
                            // (the metering read's own first phase) still
                            // feeds the core's resolve and nothing more.
                            let (prefix, _) = documents_impl::read_bounded_into(
                                &mut file,
                                Vec::new(),
                                SNIFF_PREFIX,
                            )
                            .map_err(map_read_err)?;
                            return Ok((prefix, hint_name));
                        }
                        let (b, o) = documents_impl::read_bounded_into(file, Vec::new(), limit)
                            .map_err(map_read_err)?;
                        (b, o, limit)
                    }
                    None => {
                        // Phase 1: a bounded prefix, then a content-only lane guess.
                        let (prefix, _prefix_over) =
                            documents_impl::read_bounded_into(&mut file, Vec::new(), SNIFF_PREFIX)
                                .map_err(map_read_err)?;
                        // We ignore whether the prefix alone exceeded SNIFF_PREFIX
                        // (_prefix_over is true for any file over 64 KiB): phase 2
                        // re-reads from the current file position and the real
                        // ceiling (>= 32 MiB) governs the refusal below.
                        let ceiling = match spine {
                            Spine::Conversion => documents_impl::provisional_read_ceiling(
                                &prefix,
                                hint_name.as_deref(),
                                format,
                                backend,
                                MAX_INPUT_READ,
                            ),
                            Spine::Geometry => documents_impl::provisional_geometry_read_ceiling(
                                &prefix,
                                hint_name.as_deref(),
                                format,
                                backend,
                                MAX_INPUT_READ,
                            ),
                            // The structure surface's every served row
                            // meters at the 32 MiB default (the html
                            // lane included: the conversion lane's ~23x
                            // measured amplification is a LOWER bound
                            // for the structure lane, which also holds
                            // the node array); refused rows never reach
                            // this read (the capability probe refuses
                            // them under the GIL), and an unresolvable
                            // prefix meters — the safe direction, the
                            // core's authoritative resolve refusing it
                            // post-read anyway.
                            Spine::Structure => documents_impl::DEFAULT_ANYDOC_INPUT_LIMIT,
                        };
                        // Phase 2: continue from the SAME file (position is already past
                        // the prefix), prepending the prefix we already read.
                        let (b, o) = documents_impl::read_bounded_into(file, prefix, ceiling)
                            .map_err(map_read_err)?;
                        (b, o, ceiling)
                    }
                };
                if over {
                    // The honest number: the ceiling phase 2 actually enforced
                    // (the explicit budget, or the provisional lane ceiling under
                    // None), never the 512 MiB backstop when a metered lane's
                    // 32 MiB was what was hit.
                    return Err(over_ceiling_refusal(max_bytes, used_ceiling));
                }
                Ok((bytes, hint_name))
            }
            Source::Data(bytes) => {
                // The data lane's twin gate: the budget refuses before the
                // copy (the bytes are already resident: the caller's own
                // memory, but the parse it would feed is not free). No new
                // backstop: resident bytes cannot grow past themselves.
                refuse_over_ceiling(max_bytes, None, bytes.len() as u64)?;
                // The one copy, inside the detach (the enum's docs). The
                // caller's bytes stay WHOLE even under a zero budget: the
                // core's resolve reads them (the zero VALUE refusal is
                // sequenced after resolve/capability/password-kind in the
                // core, and resolve is the step that needs the truth).
                Ok((bytes.to_vec(), None))
            }
        }
    }
}

/// Which routing table keys the two-phase read's provisional ceiling (and
/// whether the no-read door exists): the conversion spine's own
/// ([`documents_impl::provisional_read_ceiling`] — the anydoc,
/// office_oxide, and HTML lanes meter at the 32 MiB default) or the
/// geometry surface's ([`documents_impl::provisional_geometry_read_ceiling`]
/// — the office row alone meters; the pdf row and the no-geometry row
/// carry no default ceiling). The structure spine
/// ([`Source::into_structure_input`]) is the third: no no-read door, and
/// every served row meters at the 32 MiB default.
#[derive(Clone, Copy, PartialEq)]
enum Spine {
    Conversion,
    Geometry,
    Structure,
}

/// io::OutOfMemory from the bounded reader means the input was too large to
/// buffer within the ceiling: a value refusal (ValueError), not an
/// environment failure. Every other io error stays Io (OSError).
fn map_read_err(e: std::io::Error) -> InputError {
    if e.kind() == std::io::ErrorKind::OutOfMemory {
        InputError::Refused(
            "the document is too large to buffer within the read ceiling: \
             split the file or pass a smaller max_bytes"
                .into(),
        )
    } else {
        InputError::Io(e)
    }
}

/// Refuse a `path` that names anything but a regular file (or a
/// directory: those keep their pinned EISDIR `OSError`): FIFOs,
/// character/block devices, and sockets are valid paths of unusable
/// input kind: a value refusal (`ValueError`), per the module's
/// type-vs-value convention: naming `path` and the kind, so the caller
/// can act on what the path actually is.
fn refuse_non_regular(
    path: &std::path::Path,
    file_type: std::fs::FileType,
) -> Result<(), InputError> {
    if file_type.is_file() || file_type.is_dir() {
        return Ok(());
    }
    Err(InputError::Refused(format!(
        "path must name a regular file (the document's bytes), not {}: {}",
        input_kind(&file_type),
        path.display()
    )))
}

/// The explicit-budget pre-gate: `Some(limit)` with the input over it is
/// the same ceiling `ValueError` the engine lanes raise, fired before
/// the read (path) or the copy (data): same class, same two rendered
/// sizes. Where the core's message names the engine lane (knowable only
/// after the sniff), this one names the knob and states the doctrine:
/// an explicit budget binds every lane; the 32 MiB default is the
/// core's post-read check on the anydoc/oxide lanes only. `None` never
/// refuses here. Directories flow past untouched, the same deliberate
/// pass `refuse_non_regular` gives them: a directory is a kind refusal
/// (the read's own EISDIR `OSError`, the pinned doctrine), never a size
/// one, so the budget never pre-empts it. `file_type` is `None` on the
/// data lane, which cannot name a directory.
fn refuse_over_ceiling(
    max_bytes: Option<usize>,
    file_type: Option<std::fs::FileType>,
    size: u64,
) -> Result<(), InputError> {
    let Some(limit) = max_bytes else {
        return Ok(());
    };
    // The directory doctrine, refuse_non_regular's own pass: the kind
    // refusal (the read's EISDIR) is never pre-empted by a size one.
    if file_type.is_some_and(|ft| ft.is_dir()) {
        return Ok(());
    }
    // usize -> u64 is lossless on every platform; the stat side is u64
    // natively, so the gate compares in the wider type.
    if size <= limit as u64 {
        return Ok(());
    }
    // A zero LIMIT is never a size refusal: on the geometry surface the
    // zero value threads through to the core's sequenced value refusal
    // (§6: after the capability and password-kind checks — the red-team
    // probe had the binding's pre-emption masking the anydoc refusal on a
    // docx), and on every other surface `parse_max_bytes` refuses zero
    // binding-side before a budget exists. Skipping keeps the zero case
    // out of the ceiling vocabulary entirely ("must be positive" names
    // it, never a 0-byte ceiling).
    if limit == 0 {
        return Ok(());
    }
    Err(InputError::Refused(format!(
        "the document is {} and the input ceiling is {} (an explicit max_bytes is binding \
         on every engine lane — pdf and HTML included — and is checked before any work \
         runs; under the default max_bytes=None the read is bounded in two phases: the \
         metered anydoc, office_oxide, and HTML lanes refuse during the read at the 32 MiB \
         ceiling, while the pdf lane reads under the 512 MiB backstop): split \
         the file, or pass a larger max_bytes",
        render_size(size as usize),
        render_size(limit),
    )))
}

/// The over-the-read-ceiling refusal. Renders the CEILING, not the bytes
/// read: past the cap the real size is unknown, so "the document is N" would
/// be a false size. When `max_bytes` was explicit it names the knob; under
/// None it names the lane ceiling that was hit (the 32 MiB default on the
/// metered lanes, the MAX_INPUT_READ backstop on pdf) and how to raise it.
fn over_ceiling_refusal(max_bytes: Option<usize>, ceiling: usize) -> InputError {
    let detail = if max_bytes.is_some() {
        format!(
            "exceeds the max_bytes ceiling of {}: split the file or pass a larger max_bytes",
            render_size(ceiling)
        )
    } else {
        format!(
            "exceeds the {} default read ceiling: pass an explicit max_bytes to raise it",
            render_size(ceiling)
        )
    };
    InputError::Refused(format!("the document {detail}"))
}

/// One size for a ceiling refusal, legible at every scale: the core's
/// own `render_size` spelling (src/documents_impl.rs), mirrored here so
/// the binding's pre-read refusal and the core's post-read refusal
/// render identically (MiB where that rounding stays honest, raw bytes
/// below it; a 52 KiB refusal must not round to "0.0 MiB vs 0.0 MiB",
/// two sizes neither readable).
fn render_size(size: usize) -> String {
    let mib = size as f64 / (1024.0 * 1024.0);
    if mib < 0.1 {
        format!("{size} bytes")
    } else {
        format!("{mib:.1} MiB")
    }
}

/// The stat kind of an irregular input, for the refusal message: the
/// caller can act on the kind (a FIFO means a stuck producer; a
/// character device means /dev/zero's unbounded read), not on a bare
/// "not a regular file". Unix names them precisely; elsewhere the
/// honest generic is all stat gives.
#[cfg(unix)]
fn input_kind(file_type: &std::fs::FileType) -> &'static str {
    use std::os::unix::fs::FileTypeExt as _;
    if file_type.is_fifo() {
        "a FIFO"
    } else if file_type.is_char_device() {
        "a character device"
    } else if file_type.is_block_device() {
        "a block device"
    } else if file_type.is_socket() {
        "a socket"
    } else {
        "not a regular file"
    }
}

#[cfg(not(unix))]
fn input_kind(_: &std::fs::FileType) -> &'static str {
    "not a regular file"
}

/// `InputError` to the convert lane's core error: `Io` rides as `Io`
/// (the existing `OSError` mapping), a typed refusal as `Convert` (the
/// `ValueError` arm of `document_error`): the refusal crosses out of
/// the detach as a value and becomes a Python `ValueError` after the
/// GIL is reacquired, like every other document refusal.
fn input_document_error(err: InputError) -> DocumentError {
    match err {
        InputError::Io(io) => DocumentError::Io(io),
        InputError::Refused(what) => DocumentError::Convert(what),
    }
}

/// The PDF family's detach-pass error: the engine's own `PdfError`, or
/// the input refusal kept distinct from it. The distinction is the
/// point: `PdfError`'s only string-carrying variant renders as
/// "Invalid PDF: {message}", and an input refusal must never wear that
/// prefix: a FIFO path or an oversized file was never read, let alone
/// parsed, so "invalid PDF" about it would be a lie in the message. Both
/// arms map to their own `PyErr` after the GIL is reacquired.
enum PdfLaneError {
    Pdf(tors::pdf_impl::PdfError),
    Input(InputError),
}

impl From<InputError> for PdfLaneError {
    fn from(err: InputError) -> Self {
        PdfLaneError::Input(err)
    }
}

/// [`PdfLaneError`] to the Python side: the engine arm keeps
/// `pdf_error`'s mapping (`Io` → the matched `OSError` subclass,
/// everything else "not a readable PDF" → `ValueError`), the input arm
/// maps its own way (`Io` → `OSError`, a typed refusal → `ValueError`
/// with the message verbatim, no engine prefix).
fn pdf_lane_error(err: PdfLaneError) -> PyErr {
    match err {
        PdfLaneError::Pdf(err) => pdf_error(err),
        PdfLaneError::Input(InputError::Io(io)) => PyErr::from(io),
        PdfLaneError::Input(InputError::Refused(what)) => PyValueError::new_err(what),
    }
}

/// The `path`/`data` pair to one [`Source`], under the GIL: exactly one of
/// the two: both is a `ValueError`, neither a `TypeError` (the
/// missing-required-argument convention; the function name rides in
/// `what` so the message reads like Python's own). A `data=` that is not
/// `bytes` is a `TypeError` naming the argument and the type only (the
/// `password=` doctrine, extended to the one other content-bearing
/// argument: the pre-fix repr rode a rejected 1 MiB bytearray's whole
/// ~4 MB content into the exception message (straight into logged
/// tracebacks) and a `memoryview` repr leaked a raw heap address; both
/// probed). Strict `bytes`, matching the typed surface's
/// signature: `bytearray`/`memoryview` callers pass `bytes(data)`, the
/// copy they are paying anyway. A `data=` call borrows the buffer here
/// (see [`Source`]'s docs for the soundness argument); the copy runs
/// inside the detach.
fn parse_source<'a>(
    path: Option<&'a Bound<'_, PyAny>>,
    data: Option<&'a Bound<'_, PyAny>>,
    what: &str,
) -> PyResult<Source<'a>> {
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
                    type_name(data)
                ))
            })?;
            Ok(Source::Data(bytes.as_bytes()))
        }
    }
}

/// Marshal the `format=` argument under the GIL: `None` (sniff from
/// content, the default) or a format-name `str`, borrowed straight from
/// the argument. A non-str is a `TypeError` naming the argument: the raw
/// pyo3 conversion failure (`'int' object is not an instance of 'str'`)
/// names neither. The name itself is the core's vocabulary to referee: an
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

/// Marshal the `password=` argument under the GIL: `None` (the default:
/// pdf_oxide's fail-closed doctrine, the empty password tried at open) or
/// a `str`, owned so it can ride into the detached pass. A non-str is a
/// `TypeError` naming the argument, the same convention as `format=`:
/// with the one doctrine `format=` does not carry: `password` is the only
/// secret-bearing argument, so its refusals never repr the value (a
/// secret echoed into a message rides into the logged tracebacks those
/// messages land in, probed on the 0.5.0 wheel: `password=b"s3cret-pw"`
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

/// Marshal the `max_bytes=` argument under the GIL: `None` (the default:
/// the measured 32 MiB engine-lane ceiling, see the core's
/// `DEFAULT_ANYDOC_INPUT_LIMIT`: the name is historical, the ceiling
/// covers the anydoc and office_oxide lanes) or a positive `int` the
/// caller budgets instead. An explicit budget is binding on every lane
/// and is enforced pre-read by the binding itself (see
/// [`Source::into_input`]); `None` keeps the core's post-read default
/// doctrine. Wrong types raise `TypeError` (bool first: it launders as
/// an int), wrong values `ValueError` (negative, zero, i64-overflow).
fn parse_max_bytes(max_bytes: Option<&Bound<'_, PyAny>>) -> PyResult<Option<usize>> {
    parse_max_bytes_on(max_bytes, ZeroRefusal::Binding)
}

/// Whether a zero budget is refused at the parse or threaded through.
/// Every surface but one refuses it here (the house rule: a nonsense
/// argument never reaches the engines). The exception is the geometry
/// surfaces (`page_geometry`, `page_layout`): their zero VALUE refusal
/// is a max_bytes-class refusal
/// whose locked §6 position is core-side, AFTER the capability and
/// password-kind checks — the binding's pre-emption would fire it first
/// (the red-team probe: `page_geometry(data=docx, backend="anydoc",
/// max_bytes=0)` answered the max_bytes error where the capability
/// refusal is the truth), so the zero threads through as `Some(0)` and
/// the core's own zero check (already sequenced) names it.
#[derive(Clone, Copy)]
enum ZeroRefusal {
    /// Refuse zero at the parse (every surface but the geometry
    /// surfaces).
    Binding,
    /// Thread zero through as `Some(0)`; the core's sequenced check
    /// refuses it (the geometry surfaces: `page_geometry`,
    /// `page_layout`).
    CoreSide,
}

fn parse_max_bytes_on(
    max_bytes: Option<&Bound<'_, PyAny>>,
    zero: ZeroRefusal,
) -> PyResult<Option<usize>> {
    let Some(max_bytes) = max_bytes else {
        return Ok(None);
    };
    if max_bytes.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "max_bytes must be an int (the input ceiling in bytes, binding on every lane \
             when explicit), not a bool",
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
            "max_bytes must be an int (the input ceiling in bytes, binding on every lane \
             when explicit), not {}",
            py_repr(max_bytes)
        )));
    };
    // Negative budgets are refused at the parse on EVERY surface,
    // `page_geometry` included: they are unrepresentable in the core's
    // `usize` budget, so they are argument-contract refusals (the same
    // class as a non-int), not sequenced value refusals — zero is the
    // only representable value refusal, and only it can sequence.
    if limit < 0 || (limit == 0 && matches!(zero, ZeroRefusal::Binding)) {
        return Err(PyValueError::new_err(format!(
            "max_bytes must be positive (the input ceiling in bytes, binding on every lane \
             when explicit), not {limit}"
        )));
    }
    // Belt-and-braces (wheels are 64-bit-only): a 32-bit `as` would silently lower the caller's budget.
    let limit = usize::try_from(limit)
        .map_err(|_| PyValueError::new_err(format!("max_bytes {limit} is too large")))?;
    Ok(Some(limit))
}

/// Parse and validate the `backend=` argument under the GIL, before any
/// work runs: the house rule (invalid input raises before the detach, so
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

/// The PDF family's `backend="anydoc"` gate: `documents_impl`'s
/// capability refusal (the doctrine text lives with the routing table,
/// so the four functions' messages cannot drift), raised here: under
/// the GIL, at argument-contract time, ahead of the detach and the
/// encrypted door check. The ordering is the point: a contract failure
/// precedes work, so an encrypted PDF under `backend="anydoc"` surfaces
/// this refusal, not the door-check error, and no byte is read for a
/// call that could never run. A plain `ValueError`: the `Convert` arm's
/// own mapping, a value refusal per the module's type-vs-value
/// convention (this is not the format-level UnsupportedBackend, which
/// only the conversion pair can raise). The family is PDF-only by
/// construction, so the refusal's format is always `"pdf"` (the
/// byte-pinned historical text); `page_geometry`'s format-aware refusal
/// is raised in the core, where the resolved format lives.
fn refuse_anydoc(capability: &str) -> PyErr {
    PyValueError::new_err(documents_impl::anydoc_capability_refusal(capability, "pdf").to_string())
}

/// The largest page index any PDF can carry, and therefore the ceiling a
/// `pages=` range endpoint is refused above, under the GIL, before the
/// range materializes: a page is an object with a cross-reference entry,
/// and the PDF implementation limits (PDF Reference 1.7, Appendix C) cap
/// both an array and a cross-reference section at 8,388,607 (2^23-1)
/// elements/entries, so no document any reader accepts can index a page
/// above it. Chosen over the other two candidates with the pin's own
/// measurements (tests/test_documents_pdf_crashes.py's docstring):
/// `usize::MAX / 8` is the theoretical `Vec` bound and `u32::MAX` the PDF
/// integer bound, but both license allocations no valid PDF justifies:
/// the measured bug was SIGABRT (`handle_alloc_error`, exit -6) at
/// stop=10**11 (800 GiB requested), with 10**10 surviving overcommit only
/// to waste 7.6 s. At 8,388,607 the materialized half-open range costs at
/// worst 8 × 8,388,607 = 64 MiB of `usize`, once, bounded by construction.
const PDF_MAX_PAGE_INDEX: usize = 8_388_607;

/// Parse and normalize the `pages=` argument under the GIL, before any
/// work runs: a single `int` (one 0-based page), a `list` of ints (the
/// explicit set), or a 2-tuple `(start, stop)` (a half-open range, Python
/// convention). Wrong types raise `TypeError`: Python's own convention
/// (`range(1.0)` and `seq[1.0]` both raise TypeError), with the bool
/// checked first because pyo3's `extract::<i64>` launders `True`/`False`
/// to 1/0 (`isinstance(True, int)` holds): `pages=True` on the pre-fix
/// wheel silently selected page 1. Wrong values or shapes
/// raise `ValueError`: a negative, an empty list, an empty or backwards
/// range, a tuple that is not a 2-tuple, an index too large for i64, or a
/// range endpoint past the 8,388,607-object cross-reference ceiling
/// ([`PDF_MAX_PAGE_INDEX`]: no PDF can carry it; refused before the range
/// materializes, so the request can never allocate past 64 MiB of
/// indices). The
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
        // The endpoint no PDF can carry, refused before the materialization
        // below allocates it: [`PDF_MAX_PAGE_INDEX`] for the bound's own
        // derivation (the measured bug: (start..stop).collect() under the
        // GIL asked the allocator for 800 PB at stop=10**17 and died by
        // SIGABRT, before anything read the document, which here is not
        // even valid PDF bytes).
        for endpoint in [start, stop] {
            if endpoint > PDF_MAX_PAGE_INDEX {
                return Err(PyValueError::new_err(format!(
                    "the pages= range ({start}, {stop}) cannot name a page: page index \
                     {endpoint} is beyond what any PDF can carry (every page is an object \
                     with a cross-reference entry, and the implementation limit is \
                     {PDF_MAX_PAGE_INDEX} entries), so the range is refused before it \
                     can be built"
                )));
            }
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
        // (10**30 and friends): type-correct, value-absurd; a value
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

/// One `pages=` index: the scalar value, or one item of a list or range
/// tuple: to a 0-based `usize`. `TypeError` for a non-index type (bool
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
/// must not carry the value: `password=`'s (a secret never rides into a
/// message), `sniff`'s bytes refusal, and `parse_source`'s `data=`
/// refusal (document content never rides into one either: a rejected
/// 1 MiB bytearray's repr was a ~4 MB message, probed). The
/// complement of [`py_repr`]: names the type, never the
/// value; cannot fail on a real object (every type has a `__name__`), the
/// placeholder the same belt as `py_repr`'s.
fn type_name(value: &Bound<'_, PyAny>) -> String {
    value
        .get_type()
        .name()
        .and_then(|name| name.to_str().map(str::to_string))
        .unwrap_or_else(|_| "<unknown type>".to_string())
}

/// repr() for the argument-validation errors: the repr names the type too
/// (`123` vs `'123'` vs `1.5`), which a bare class name does not, and
/// cannot fail on well-formed values (repr escapes lone surrogates); a
/// hostile `__repr__` that raises or returns non-printable text falls
/// back to a placeholder: the surrounding message already names the
/// argument, which is the part that cannot be lost.
fn py_repr(value: &Bound<'_, PyAny>) -> String {
    value
        .repr()
        .and_then(|repr| repr.to_str().map(str::to_string))
        .unwrap_or_else(|_| "<unrepresentable value>".to_string())
}

/// pdf_oxide's error to the Python side, `pdf_extract`/`pdf_page_count`/
/// `pdf_classify`'s mapping: IO failures are environment failures
/// (`OSError`: pyo3's `From<io::Error>` picks the matched subclass,
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
/// `From<io::Error>`, the matched subclass: see `pdf_error`);
/// `NeedsOcr` → `NeedsOcrError` carrying `.pages`/`.page_count`;
/// everything else → `ValueError`. All mapping happens after the detached
/// region, and the attributes are set on the materialized exception
/// instance under the GIL. `.pages` holds 0-based page indices: the
/// core re-bases anydoc's 1-based numbers once, at its seam, so every
/// page number this payload names (`pages=`, `pages_needing_ocr`,
/// `pdf_extract`'s list positions, `.pages`) carries one convention.
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
            // far from its cause: the caller's attribute read would
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
    // The wheel's version, baked from this crate's Cargo.toml at build
    // time: release-please bumps Cargo.toml, pyproject.toml, and the root
    // crate together (extra-files: one release, two wheels), so this is
    // the wheel's version by construction. Chosen over reading
    // importlib.metadata at import time (a site-packages scan on every
    // import, plus a static fallback pin that drifts stale in source
    // checkouts); the payload's Python layer re-exports it as
    // tors_documents.__version__.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("NeedsOcrError", m.py().get_type::<NeedsOcrError>())?;
    m.add_class::<PdfClassification>()?;
    m.add_class::<PageGeometry>()?;
    m.add_class::<PageLayout>()?;
    m.add_class::<TextSpan>()?;
    m.add_class::<TextLine>()?;
    m.add_class::<Word>()?;
    m.add_class::<ImageBox>()?;
    m.add_class::<PageTables>()?;
    m.add_class::<Table>()?;
    m.add_class::<Row>()?;
    m.add_class::<Cell>()?;
    m.add_class::<PageImagesData>()?;
    m.add_class::<PageImageEntry>()?;
    m.add_class::<ImageBytes>()?;
    m.add_class::<PageCharsData>()?;
    m.add_class::<TextChar>()?;
    m.add_class::<PdfSecurity>()?;
    m.add_class::<PdfPermissions>()?;
    m.add_class::<DocumentStructure>()?;
    m.add_class::<DocxStructure>()?;
    m.add_class::<DocxBlock>()?;
    m.add_class::<DocxTable>()?;
    m.add_class::<DocxCell>()?;
    m.add_class::<DocxLink>()?;
    m.add_class::<DocxImage>()?;
    m.add_class::<XlsxStructure>()?;
    m.add_class::<XlsxSheet>()?;
    m.add_class::<XlsxCell>()?;
    m.add_class::<XlsxDefinedName>()?;
    m.add_class::<XlsxHyperlink>()?;
    m.add_class::<XlsxComment>()?;
    m.add_class::<HtmlStructure>()?;
    m.add_class::<HtmlNode>()?;
    m.add_class::<HtmlGrid>()?;
    m.add_class::<HtmlGridCell>()?;
    m.add_class::<HtmlAnnotation>()?;
    m.add_function(wrap_pyfunction!(page_geometry, m)?)?;
    m.add_function(wrap_pyfunction!(page_layout, m)?)?;
    m.add_function(wrap_pyfunction!(page_tables, m)?)?;
    m.add_function(wrap_pyfunction!(page_images, m)?)?;
    m.add_function(wrap_pyfunction!(page_text_chars, m)?)?;
    m.add_function(wrap_pyfunction!(document_structure, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_classify, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_extract, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_link_uris, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_page_count, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_page_labels, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_security, m)?)?;
    m.add_function(wrap_pyfunction!(to_markdown, m)?)?;
    m.add_function(wrap_pyfunction!(to_text, m)?)?;
    m.add_function(wrap_pyfunction!(sniff, m)?)?;
    Ok(())
}
