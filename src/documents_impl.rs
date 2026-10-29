//! The multi-format document extraction core (the `tors-documents`
//! payload crate's bindings): resolve a format, route it to the measured
//! best engine, convert to GitHub-Flavored Markdown, and optionally
//! normalize that markdown to plain text. Pure Rust, no pyo3: the binding
//! layer lives in `tors-documents/src/lib.rs`, which releases the GIL
//! around this whole pass.
//!
//! # Engine routing, and why exactly this split
//!
//! Four engines, chosen per family by head-to-head measurement (over
//! hand-built deterministic fixtures: two-column and link-annotation PDFs,
//! style-bearing docx with numbering/hyperlink/table, pptx with speaker
//! notes, odt, csv, a polluted-head HTML page, plus the `tests/documents.py`
//! corpus):
//!
//! - **PDF → pdf_oxide.** Measured wins that pdf-inspector (anydoc's PDF
//!   engine) lacks: two-column layouts come back as separate reading-order
//!   blocks instead of an interleaved table; `/Link` annotations render as
//!   `[text](uri)` instead of being dropped; line structure is preserved.
//! - **HTML → html-to-markdown-rs** (`extract_metadata: false`). The only
//!   measured engine that drops `<script>`/`<style>` by construction:
//!   htmd routes them through its generic block handler, leaking CSS/JS
//!   text into the body markdown, which is disqualifying for web-page
//!   ingestion; fast_html2md omits the GFM table delimiter row (tables
//!   render as prose). Also measured on the pick: padded GFM tables with
//!   delimiter rows, indented nested lists, fully decoded entities, clean
//!   code fences. The `<title>` would surface as YAML frontmatter;
//!   suppressed by the option (verified), the output is exactly the body.
//! - **Everything else → anydoc** (doc/docx, xls/xlsx, ppt/pptx, rtf,
//!   odt/ods/odp, epub, csv). Measured wins over office_oxide (which rides
//!   along compiled-in via pdf_oxide's tree): anydoc renders style-based
//!   docx headings (`pStyle` → `#`) and list markers that office_oxide
//!   drops entirely, and covers rtf/odt/epub/csv office_oxide cannot read
//!   at all. anydoc's one measured loss is cosmetic: `&` is HTML-escaped
//!   in its markdown (`&amp;`), which [`gfm_strip_impl`] un-escapes for
//!   plain-text output.
//! - **office_oxide is the caller-selectable `backend="oxide"` lane** for
//!   the OOXML + legacy office formats (docx/xlsx/pptx, doc/xls/ppt): it
//!   is already compiled in via pdf_oxide's tree, so exposing it costs no
//!   weight, and it measured one real win (exact entity text, no
//!   `&`-escaping) against the heading/list losses above: a documented
//!   alternative, never the default. `backend="oxide"` on any other format
//!   is an error, never a silent fallback.
//!
//! The backend is caller-selectable where engines overlap:
//! `Backend::Auto` routes by the table above; `Backend::Oxide` and
//! `Backend::Anydoc` force one engine and error on a format that engine
//! cannot read.
//!
//! # Format resolution
//!
//! An explicit format name (extension spelling, no dot, case-insensitive:
//! anydoc's `Format::from_extension` vocabulary plus `html`/`htm`/`xhtml`
//! and `tsv`, HTML's and the delimiter-separated family's name-only
//! spellings) beats everything. Without one, the format is sniffed from
//! the bytes' content markers (PDF header, RTF open group, OLE stream
//! names, ZIP mimetype, HTML document marker) with the input name's
//! extension as the last resort for signature-less formats (CSV): the
//! `path=` the payload read, when there was one; the `data=` entry has no
//! name and rests on `format=`/markers alone, the doctrine [`sniff`]
//! encodes, and the extension resolves through the same [`Kind::from_name`]
//! vocabulary the
//! explicit name uses, so one table owns both spellings. anydoc's
//! doctrine, reused here so a mislabeled file still converts correctly.
//! [`sniff`] exposes the content-marker half standalone, over bytes
//! alone, reporting the same container-true name a conversion would.
//!
//! # Plain-text output
//!
//! `to_text` converts to markdown first and then normalizes via
//! [`gfm_strip_impl::strip`]: one text shape for every format and engine,
//! instead of each engine's ad-hoc plain-text surface (pdf_oxide's native
//! plain text, measured, merges two-column layouts line-by-line; the strip
//! keeps the markdown converter's reading-order blocks). The entity policy
//! is per-engine: anydoc HTML-escapes `&` in its markdown, so its output is
//! un-escaped; pdf_oxide, office_oxide, and html-to-markdown-rs emit text
//! literally, so theirs is passed through untouched.
//!
//! # Page subsets
//!
//! `pages=` (0-based PDF page indices, deduped, in document order, as the
//! binding layer delivers them) converts each selected page and joins with
//! pdf_oxide's own inter-page separator: measured, a full range is
//! byte-identical to the whole-document conversion. PDF on the pdf_oxide
//! lane only: any other format, or `backend="anydoc"` on a PDF, is a
//! `Pages` error, never a silent whole-document fallback.

use std::io::Cursor;

use anydoc::ConvertError as AnydocError;
use anydoc::Format;

use crate::documents_impl::DocumentError::{Convert, Io};
use crate::gfm_strip_impl;
use crate::pdf_impl;

/// Which engine a conversion runs on. `Auto` follows the routing table in
/// the module docs; the explicit variants force one engine and error on
/// formats it cannot read (parsed from the Python `backend=` string in the
/// binding layer, before any work runs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Auto,
    Oxide,
    Anydoc,
}

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Auto => "auto",
        Backend::Oxide => "oxide",
        Backend::Anydoc => "anydoc",
    }
}

/// The engine lane's own name for the ceiling refusal (and only there:
/// routing answers speak in backend terms, resource answers in engine
/// ones: the ceiling is an engine-lane property).
fn engine_name(engine: Engine) -> &'static str {
    match engine {
        Engine::PdfOxide => "pdf_oxide",
        Engine::OfficeOxide => "office_oxide",
        Engine::Anydoc => "anydoc",
        Engine::Html2Md => "html-to-markdown-rs",
    }
}

/// One size for the ceiling refusal, legible at every scale: MiB where
/// that rounding stays honest, raw bytes below it (a 52 KiB refusal used
/// to round to "0.0 MiB … 0.0 MiB": two sizes, neither readable).
fn render_size(size: usize) -> String {
    let mib = size as f64 / (1024.0 * 1024.0);
    if mib < 0.1 {
        format!("{size} bytes")
    } else {
        format!("{mib:.1} MiB")
    }
}

/// The working-format vocabulary: anydoc's `Format` (which also names the
/// OOXML container variants: `docm`/`xlsm`/`xlsb`/`ppsx` map onto these)
/// plus HTML: XHTML included, it being HTML's XML serialization on the
/// same engine, which no anydoc parser reads and the lane's own engine
/// serves. The names are the `format=`/`Format::from_extension` vocabulary
/// and the spelling the resolved format is reported back as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Pdf,
    Html,
    Doc,
    Docx,
    Excel,
    Ppt,
    Pptx,
    Rtf,
    Odt,
    Ods,
    Odp,
    Epub,
    Csv,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Pdf => "pdf",
            Kind::Html => "html",
            Kind::Doc => "doc",
            Kind::Docx => "docx",
            Kind::Excel => "xlsx",
            Kind::Ppt => "ppt",
            Kind::Pptx => "pptx",
            Kind::Rtf => "rtf",
            Kind::Odt => "odt",
            Kind::Ods => "ods",
            Kind::Odp => "odp",
            Kind::Epub => "epub",
            Kind::Csv => "csv",
        }
    }

    /// Resolve an explicit format name: HTML's, XHTML's, and TSV's names
    /// route here (HTML/XHTML have no anydoc parser and ride this lane's
    /// own engine: XHTML is HTML's XML serialization, and its
    /// `<?xml …?>` prologue is the one prefix variant the content marker
    /// knows how to skip; TSV is the delimiter-separated family's other
    /// half: anydoc's csv parser is delimiter-separated, and the content
    /// heuristic already resolves TSV bytes, so the name is vocabulary
    /// sugar mapping onto the csv kind), everything else through anydoc's
    /// `from_extension` (which also maps the container variants:
    /// `docm` → Docx, `xlsm`/`xlsb` → Excel, `ppsx` → Pptx).
    pub fn from_name(named: &str) -> Option<Kind> {
        let spelled = named.trim_start_matches('.').trim();
        if spelled.eq_ignore_ascii_case("html")
            || spelled.eq_ignore_ascii_case("htm")
            || spelled.eq_ignore_ascii_case("xhtml")
        {
            return Some(Kind::Html);
        }
        if spelled.eq_ignore_ascii_case("tsv") {
            return Some(Kind::Csv);
        }
        Format::from_extension(spelled).map(Kind::from_anydoc)
    }

    fn from_anydoc(format: Format) -> Kind {
        match format {
            Format::Pdf => Kind::Pdf,
            Format::Docx => Kind::Docx,
            Format::Doc => Kind::Doc,
            Format::Excel => Kind::Excel,
            Format::Ppt => Kind::Ppt,
            Format::Pptx => Kind::Pptx,
            Format::Rtf => Kind::Rtf,
            Format::Odt => Kind::Odt,
            Format::Ods => Kind::Ods,
            Format::Odp => Kind::Odp,
            Format::Epub => Kind::Epub,
            Format::Csv => Kind::Csv,
        }
    }

    fn to_anydoc(self) -> Format {
        match self {
            Kind::Pdf => Format::Pdf,
            Kind::Docx => Format::Docx,
            Kind::Doc => Format::Doc,
            Kind::Excel => Format::Excel,
            Kind::Ppt => Format::Ppt,
            Kind::Pptx => Format::Pptx,
            Kind::Rtf => Format::Rtf,
            Kind::Odt => Format::Odt,
            Kind::Ods => Format::Ods,
            Kind::Odp => Format::Odp,
            Kind::Epub => Format::Epub,
            Kind::Csv => Format::Csv,
            // unreachable by construction: engine_for only routes Html to
            // the HTML engine, which never touches anydoc.
            Kind::Html => Format::Csv,
        }
    }
}

/// One completed conversion: the format it actually was (the sniffed or
/// explicit format name: the caller may have passed `format=None` and wants
/// to know what came back) and the converted output (markdown for
/// `to_markdown`, plain text for `to_text`).
#[derive(Debug)]
pub struct Converted {
    pub format: &'static str,
    pub output: String,
}

/// Why a conversion could not run. The binding layer maps `Io` to `OSError`,
/// `NeedsOcr` to the payload's `NeedsOcrError` (carrying the page list), and
/// everything else to `ValueError`.
#[derive(Debug)]
pub enum DocumentError {
    /// The file could not be read.
    Io(std::io::Error),
    /// `format=None` and neither the content markers nor the extension named
    /// a known format, or an explicit format name is not in the vocabulary.
    UnknownFormat(String),
    /// The forced backend cannot read this format (`Backend::Oxide` on
    /// anything but PDF and the OOXML/legacy office family;
    /// `Backend::Anydoc` on HTML).
    UnsupportedBackend {
        format: &'static str,
        backend: &'static str,
    },
    /// The `pages=` selection is invalid for this conversion: a non-PDF
    /// format, the `anydoc` PDF lane, or (from `pdf_impl`) an out-of-range
    /// or empty selection.
    Pages(String),
    /// anydoc's page-precise signal that these pages are image-only and need
    /// OCR: routing information, not a generic failure: the caller's chain
    /// should send this document to its OCR stage. `pages` holds 0-based
    /// page indices (the surface-wide convention; anydoc's 1-based numbers
    /// are re-based once, in `anydoc_error`, where the engine's answer
    /// crosses the seam).
    NeedsOcr { pages: Vec<u32>, page_count: u32 },
    /// The document is not convertible: malformed, encrypted, over a
    /// resource limit, a required part missing, or an engine parse failure.
    Convert(String),
}

/// The error text: byte-equal to the strings the binding layer raises on
/// the Python side, so a Rust caller `?`-ing into `Box<dyn Error>`/anyhow
/// sees exactly what the payload's user sees (the payload's mapping arms
/// carry the same strings; its `UnknownFormat`/`Pages`/`Convert` arms
/// could now collapse onto `to_string()`: noted for that crate, not
/// changed here).
impl std::fmt::Display for DocumentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Io(io) => write!(f, "{io}"),
            DocumentError::UnknownFormat(what) => write!(f, "{what}"),
            DocumentError::UnsupportedBackend { format, backend } => write!(
                f,
                "backend {backend:?} cannot read format {format:?}: use backend='auto'"
            ),
            DocumentError::Pages(what) => write!(f, "{what}"),
            DocumentError::NeedsOcr { pages, page_count } => write!(
                f,
                "pages {} of {page_count} need OCR (indices are 0-based, like pages=): \
                 route this document to an OCR stage",
                pages
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Convert(what) => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for DocumentError {}

/// The engine a (format, backend) pair runs on. Derived, never stored: the
/// routing table is one function, the single place the decision exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Engine {
    PdfOxide,
    OfficeOxide,
    Anydoc,
    Html2Md,
}

fn engine_for(kind: Kind, backend: Backend) -> Result<Engine, DocumentError> {
    match (kind, backend) {
        (Kind::Pdf, Backend::Auto) | (Kind::Pdf, Backend::Oxide) => Ok(Engine::PdfOxide),
        (Kind::Pdf, Backend::Anydoc) => Ok(Engine::Anydoc),
        (Kind::Html, Backend::Auto) => Ok(Engine::Html2Md),
        (Kind::Html, Backend::Oxide) | (Kind::Html, Backend::Anydoc) => {
            Err(DocumentError::UnsupportedBackend {
                format: "html",
                backend: backend_name(backend),
            })
        }
        (Kind::Doc | Kind::Docx | Kind::Excel | Kind::Ppt | Kind::Pptx, Backend::Oxide) => {
            Ok(Engine::OfficeOxide)
        }
        (
            Kind::Odt | Kind::Ods | Kind::Odp | Kind::Epub | Kind::Rtf | Kind::Csv,
            Backend::Oxide,
        ) => Err(DocumentError::UnsupportedBackend {
            format: kind.name(),
            backend: "oxide",
        }),
        (_, Backend::Auto) | (_, Backend::Anydoc) => Ok(Engine::Anydoc),
    }
}

/// The PDF-family payload calls' `backend="anydoc"` refusal: a
/// capability-level refusal, deliberately not the format-level
/// [`DocumentError::UnsupportedBackend`] (that one answers "can this
/// engine read this format", and PDF+anydoc can: [`engine_for`] routes
/// the pair; this one answers "can this engine serve this call"). The
/// probe-rich calls need per-engine surfaces anydoc's PDF reader
/// structurally lacks: its entire PDF surface is `to_markdown(bytes)`
/// whole-document markdown (~anydoc-0.2.4/src/formats/pdf.rs: no
/// per-page text, no page-count success return, no per-page
/// classification, no annotation walk, no page-box read), while
/// `pdf_extract`'s per-page probe is the OCR-routing signal,
/// `pdf_page_count` walks the page tree, `pdf_classify` classifies per
/// page, `pdf_link_uris` walks `/Annots`, and `page_geometry` reads the
/// declared page boxes. The doctrine text lives here, next to the routing
/// table, so the messages cannot drift apart; the binding layer raises
/// them under the GIL at argument-contract time where the lane is known
/// without reading bytes (the PDF-only family), and out of the detached
/// pass otherwise (`page_geometry`, whose refusal is format-aware and so
/// needs the resolved format first). Message shape follows
/// [`DocumentError::UnsupportedBackend`]'s Display: name the backend and
/// the value, name the lack, point at the fix.
///
/// `format` selects the doctrine text: `"pdf"` is the byte-pinned
/// historical message (the four PDF-only calls' text, unchanged); any
/// other format name is the geometry surface's office-family refusal —
/// same shape, the format named, the lack restated for the lane anydoc
/// has no geometry surface on at all. Both strings are pinned in the
/// core tests.
pub fn anydoc_capability_refusal(capability: &str, format: &str) -> DocumentError {
    DocumentError::Convert(if format == "pdf" {
        format!(
            "backend \"anydoc\" cannot serve {capability}: anydoc's PDF surface is \
             whole-document conversion only — to_markdown/to_text with backend=\"anydoc\" \
             (NeedsOcrError is that lane's scanned-page signal); use backend='auto' or \
             'oxide' here"
        )
    } else {
        format!(
            "backend \"anydoc\" cannot serve {capability} for {format}: anydoc has no \
             geometry surface; use backend='auto' or 'oxide' here"
        )
    })
}

/// One `page_geometry` entry: a unit's declared render box, in points —
/// the shared core type. `pdf_impl::PageBox` (the PDF probe's own unit)
/// maps into it, and the payload crate exposes it as the native
/// `PageGeometry` pyclass; the fields are the same law on every lane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PageGeometry {
    /// The unit's render width in points. PDF: the visible box's extent,
    /// the axes swapped for a 90/270 rotation. Office: the declared page
    /// size (twips ÷ 20 for DOCX, EMU ÷ 12700.0 for PPTX — the raw
    /// layers' honest conversions), never rotated (`w:orient` is never
    /// used to swap declared dims — dims only).
    pub width_pt: f32,
    /// The unit's render height in points (the same law).
    pub height_pt: f32,
    /// The declared rotation, engine-normalized: the PDF engine's
    /// `/Rotate` ∈ {0, 90, 180, 270} (a non-right-angle declaration
    /// reports 0); an office unit has no rotation concept and reports 0.
    pub rotation: i32,
    /// The PDF page's declared CropBox reduced to the MediaBox (the
    /// effective crop: the intersection the engine's visible box stands
    /// on), normalized corners, the unrotated source frame, points —
    /// `Some` only when a CropBox is declared AND it actually crops.
    /// `None` on every office lane (no office format declares a crop) and
    /// on PDF pages without an effective crop. (Phase 2's note, the same
    /// on `pdf_impl::PageBox`: the visible-box ORIGIN is not carried —
    /// provenance for the extent, never a rectangle to transform — and
    /// the phase-2 layout transform must fetch the corners per page
    /// itself.)
    pub crop: Option<(f32, f32, f32, f32)>,
}

impl PageGeometry {
    /// `pdf_impl::PageBox` (the PDF probe's unit) into the shared type:
    /// the fields are already the same law, so this is one of the two
    /// places the two vocabularies meet — the other is `page_layout`'s
    /// office arm, which maps the shared type back into `pdf_impl::PageBox`
    /// to serve the layout probe's uniform entry shape. Public for the
    /// same reason: the payload binding's `PageLayout.geometry` getter
    /// materializes the shared pyclass from the layout probe's `PageBox`
    /// per page.
    pub fn from_page_box(page: pdf_impl::PageBox) -> PageGeometry {
        PageGeometry {
            width_pt: page.width_pt,
            height_pt: page.height_pt,
            rotation: page.rotation,
            crop: page.crop,
        }
    }
}

/// The geometry capability table: the second routing table, documented
/// next to (never as a comment on) [`engine_for`]. Conversion routing
/// answers "who converts best"; this table answers "who can declare
/// geometry at all" — and the two deliberately disagree in places:
///
/// - `auto` and `oxide` are identical on this surface for every format
///   (the geometry lane is a property of the format, not a choice): PDF
///   pages read through pdf_oxide's page-tree walk, the office families
///   through office_oxide's raw/IR layers, and nothing else runs an
///   engine at all.
/// - `anydoc` is refused exactly where geometry is obtainable on a lane
///   the caller excluded — the pdf row and the office row — with the
///   format-aware capability refusal ([`anydoc_capability_refusal`]).
/// - The no-geometry row (html/csv/rtf/epub/odt/ods/odp) runs NO engine
///   and refuses NO backend: the answer is the backend-independent
///   single `[None]` entry (unit = whole document, single-flow). This is
///   a deliberate divergence from conversion routing's `UnsupportedBackend`
///   pairs (html+oxide/anydoc, the ODF family+oxide): geometry runs no
///   engine on that row, so there is nothing to refuse and nothing is
///   silently fallen back — the same statement both backends' docs
///   carry.
///
/// The table is total: every `Kind` is one `Ok` arm (or the refusal) —
/// no format falls out of a default arm unasked.
///
/// `Ok(None)` = the no-geometry row; `Ok(Some(Engine))` = the engine the
/// geometry read runs on; `Err` = the anydoc capability refusal.
///
/// `bytes` is the (prefix of the) document the format is being resolved
/// for, so the refusal names the CONTAINER the bytes are
/// ([`resolved_name`]: the Excel kind's one container-aware case — a
/// legacy OLE workbook is `xls`, a ZIP package `xlsx`), never the kind's
/// default spelling.
fn geometry_engine(
    kind: Kind,
    backend: Backend,
    bytes: &[u8],
) -> Result<Option<Engine>, DocumentError> {
    let format_name = resolved_name(kind, bytes);
    match (kind, backend) {
        (Kind::Pdf, Backend::Anydoc) => Err(anydoc_capability_refusal(
            PDF_GEOMETRY_CAPABILITY,
            Kind::Pdf.name(),
        )),
        (Kind::Pdf, _) => Ok(Some(Engine::PdfOxide)),
        (Kind::Docx | Kind::Pptx | Kind::Excel | Kind::Doc | Kind::Ppt, Backend::Anydoc) => Err(
            anydoc_capability_refusal(OFFICE_GEOMETRY_CAPABILITY, format_name),
        ),
        (Kind::Docx | Kind::Pptx | Kind::Excel | Kind::Doc | Kind::Ppt, _) => {
            Ok(Some(Engine::OfficeOxide))
        }
        (
            Kind::Html | Kind::Rtf | Kind::Odt | Kind::Ods | Kind::Odp | Kind::Epub | Kind::Csv,
            _,
        ) => Ok(None),
    }
}

/// The PDF lane's capability name in the anydoc refusal (why this call
/// cannot run there): the declared page-box read anydoc's whole-document
/// PDF surface has no counterpart for.
const PDF_GEOMETRY_CAPABILITY: &str = "the declared page-box geometry (no per-page box surface)";

/// The office lane's capability name in the anydoc refusal: the spec's
/// own "page geometry" — the refusal template spells the format beside
/// it (`cannot serve page geometry for {format}`).
const OFFICE_GEOMETRY_CAPABILITY: &str = "page geometry";

/// The document-engine lanes' default input ceiling: 32 MiB, applied to
/// the two lanes that amplify their input into resident memory: anydoc
/// and office_oxide. anydoc amplifies at a measured ~146× worst case on
/// adversarial delimiter formats (this box: a 24 MiB csv of
/// 1,258,291 rows × 10 one-char cells peaked at 3.42 GiB RSS through
/// to_text in 6.4 s, and a 12 MiB one at 1.73 GiB in 3.1 s: the
/// multiple is stable across sizes, a per-byte property: anydoc
/// materializes row structures per cell. The earlier ~36× figure (a
/// 100 MiB csv → 3.7 GiB) was a benign-shape measurement; cells per
/// byte, not file size, drives the multiple) and additionally caps
/// decompression engine-side (its own package limits: 128 MiB per entry,
/// 512 MiB total, a 4M× expansion bound). office_oxide 0.1.10 enforces
/// MAX_PART_SIZE = 512 MiB per part: declared and actual bytes; XML
/// nesting depth 256 on its own 16 MB parse stack. Its remaining gap, no
/// total-across-parts cap (measured: a 399 KiB zip with a 400 MiB
/// word/document.xml converted successfully at 1.58 GiB peak RSS in under
/// a second, the ~400 MiB markdown handed over whole), is closed at the
/// seam: the oxide lane audits every container before the engine parses
/// it and refuses when the parts' inflated bytes pass 512 MiB in total
/// (see [`OXIDE_MAX_TOTAL_DECOMPRESSED`] and [`audit_oxide_container`]),
/// so this input-side bound now composes with a decompression-side one
/// instead of standing alone against it. At the measured ~146×, the 32
/// MiB default ceiling's honest worst case on the anydoc lane is ~4.6
/// GiB: not a survivable spike on a small ingestion worker; callers
/// with tight budgets must lower [`ConvertOptions::max_bytes`] or split
/// the file. The name stays anydoc-branded (history: the knob was born
/// anydoc-only and the payload crate's docs reference it by this name);
/// the semantics are all amplified lanes. The pdf_oxide lane is unmetered
/// here: pdf_oxide's own resource limits govern there. The HTML lane is
/// metered like the others: its converter holds the whole input and output
/// in memory at once, measured at ~23x input peak RSS (a 48 MiB doctype
/// HTML peaked at 1118 MiB), so it gets the same input-side bound as the
/// lanes that amplify by delimiter or container.
pub const DEFAULT_ANYDOC_INPUT_LIMIT: usize = 32 * 1024 * 1024;

/// The oxide lane's total-decompression ceiling: 512 MiB, the most one
/// container's parts may inflate to IN SUM on `backend="oxide"`, the same
/// number anydoc's package limit carries and office_oxide's per-part cap
/// names. office_oxide 0.1.10 bounds each part separately (its
/// `MAX_PART_SIZE`, declared and actual) and nothing bounds their sum,
/// and it never checks a declared size against the inflated bytes:
/// measured on this tree, a 599 KiB docx whose two parts inflate to
/// 300 MiB each (each far under the per-part cap) converted on
/// `backend="oxide"` at ~2.1 GiB peak RSS in ~1.2 s, and with both parts
/// declaring 1,000 bytes it converted identically. The ceiling is ours,
/// enforced at the seam before the engine parses anything: see
/// [`audit_oxide_container`].
pub const OXIDE_MAX_TOTAL_DECOMPRESSED: u64 = 512 * 1024 * 1024;

/// The zip record signatures the audit walks: local file header, central
/// directory record, the two end-of-directory records (zip32 and the
/// zip64 locator/record pair).
const ZIP_LOCAL_HEADER_SIG: [u8; 4] = [0x50, 0x4B, 0x03, 0x04];
const ZIP_CENTRAL_SIG: [u8; 4] = [0x50, 0x4B, 0x01, 0x02];
const ZIP_EOCD_SIG: [u8; 4] = [0x50, 0x4B, 0x05, 0x06];
const ZIP64_LOCATOR_SIG: [u8; 4] = [0x50, 0x4B, 0x06, 0x07];
const ZIP64_EOCD_SIG: [u8; 4] = [0x50, 0x4B, 0x06, 0x06];

/// Little-endian field reads, bounds-checked: a truncated record is
/// `None`, never a panic (the audit runs on attacker-shaped bytes).
fn le16(bytes: &[u8], at: usize) -> Option<u16> {
    let end = at.checked_add(2)?;
    let window = bytes.get(at..end)?;
    Some(u16::from_le_bytes([window[0], window[1]]))
}

fn le32(bytes: &[u8], at: usize) -> Option<u32> {
    let end = at.checked_add(4)?;
    let window = bytes.get(at..end)?;
    Some(u32::from_le_bytes([
        window[0], window[1], window[2], window[3],
    ]))
}

fn le64(bytes: &[u8], at: usize) -> Option<u64> {
    let end = at.checked_add(8)?;
    let window = bytes.get(at..end)?;
    Some(u64::from_le_bytes(window.try_into().ok()?))
}

/// The [MS-CFB] OLE compound-file signature: the legacy office container
/// (doc/ppt, and Excel's legacy `xls` inside the Excel kind). One
/// constant, shared by the reported-name choice in [`resolved_name`] and
/// the oxide lane's engine choice in `office_markdown`, so the two can
/// never disagree about which container the bytes are.
const OLE_MAGIC: [u8; 8] = [0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1];

fn is_ole_container(bytes: &[u8]) -> bool {
    bytes.starts_with(&OLE_MAGIC)
}

/// The name a resolved kind reports back as: the Excel kind's one
/// container-aware case: the OLE signature is legacy `xls`, anything else
/// (a ZIP local-file header) is the `xlsx` family, so the reported name
/// matches the container the bytes actually are (the same container check
/// `office_markdown` makes for the engine choice, applied to the report:
/// and to [`sniff`]'s answer, which routes through here for the same
/// reason).
fn resolved_name(kind: Kind, bytes: &[u8]) -> &'static str {
    if kind == Kind::Excel && is_ole_container(bytes) {
        return "xls";
    }
    kind.name()
}

/// The caller-tuned conversion knobs beyond the routing arguments: one
/// struct so later knobs (an HTML title mode, positioned-output shape,
/// …) extend the surface without re-breaking every signature. `Default`
/// is the measured-safe posture: no password, the default input ceiling.
#[derive(Default)]
pub struct ConvertOptions<'a> {
    /// The PDF's password, when it is encrypted. `None` keeps pdf_oxide's
    /// fail-closed doctrine (the empty password is tried at open; a
    /// document that stays locked fails its first content operation,
    /// never masked as empty output). Applies to the pdf_oxide lane only
    /// (`backend='auto'`/`'oxide'` on a PDF): a password on any other
    /// format is a `Convert` error, and so is one on the anydoc PDF lane:
    /// its reader takes no password, so a correct one would fail exactly
    /// like a wrong one, indistinguishable from ignored. Never a silently
    /// ignored argument.
    pub password: Option<&'a str>,
    /// The engine-lane input ceiling, in bytes: it bounds input size on
    /// the lanes that amplify input into resident memory: anydoc
    /// (~146× RSS worst case on adversarial delimiter formats: a 24 MiB
    /// csv → 3.42 GiB peak, a 12 MiB one → 1.73 GiB),
    /// office_oxide (a 399 KiB zip-bombed docx with a 400 MiB part →
    /// 1.58 GiB peak on that lane, same date), and HTML (~23× input:
    /// a 48 MiB doctype HTML → 1118 MiB peak, measured 2026-09-14).
    /// `None` is the default:
    /// the 32 MiB [`DEFAULT_ANYDOC_INPUT_LIMIT`]; `Some(n)` raises or
    /// lowers it for callers with a bigger (or tighter) memory budget:
    /// at the measured multiple the default's anydoc-lane worst case is
    /// ~4.6 GiB, so tighter is often right. The honest limits of what it
    /// bounds: the bytes handed in, never the bytes they inflate to. The
    /// inflated side is bounded one lane deep: anydoc caps decompression
    /// engine-side (its package limits), and the oxide lane caps what a
    /// container's parts inflate to in total at the seam
    /// ([`OXIDE_MAX_TOTAL_DECOMPRESSED`], 512 MiB across all parts, on
    /// top of office_oxide's 512 MiB per-part cap). No lane has an
    /// output-size cap beyond those. The pdf_oxide lane is unmetered by
    /// this knob: its own limits govern.
    pub max_bytes: Option<usize>,
}

/// Convert document bytes to GitHub-Flavored Markdown on the routed (or
/// forced) engine. `name_hint` is the input's name: the `path=` the
/// payload read, when there was one: used only by the extension fallback
/// of format resolution (the `data=` entry passes `None`, and resolution
/// then rests on the explicit `format=` or the content markers alone, the
/// same doctrine [`sniff`] encodes). `pages` (a deduped, document-order
/// selection of 0-based PDF indices, delivered by the binding layer) is
/// PDF-on-pdf_oxide only. [`ConvertOptions::default`] for the knobs.
pub fn to_markdown(
    bytes: Vec<u8>,
    name_hint: Option<&str>,
    format: Option<&str>,
    backend: Backend,
    pages: Option<Vec<usize>>,
) -> Result<Converted, DocumentError> {
    to_markdown_with(
        bytes,
        name_hint,
        format,
        backend,
        pages,
        ConvertOptions::default(),
    )
}

/// [`to_markdown`] with the caller-tuned knobs (password, input ceiling).
pub fn to_markdown_with<'a>(
    bytes: Vec<u8>,
    name_hint: Option<&str>,
    format: Option<&str>,
    backend: Backend,
    pages: Option<Vec<usize>>,
    options: ConvertOptions<'a>,
) -> Result<Converted, DocumentError> {
    let (format, _engine, markdown) = convert(bytes, name_hint, format, backend, pages, &options)?;
    Ok(Converted {
        format,
        output: markdown,
    })
}

/// Convert document bytes to plain text: the markdown conversion, then
/// [`gfm_strip_impl::strip`] with the engine's entity policy. See the
/// module docs for why text is normalized from the markdown rather than
/// taken from each engine's own plain-text surface. Same `name_hint`/
/// `format=`/`backend=`/`pages=` contract as [`to_markdown`].
pub fn to_text(
    bytes: Vec<u8>,
    name_hint: Option<&str>,
    format: Option<&str>,
    backend: Backend,
    pages: Option<Vec<usize>>,
) -> Result<Converted, DocumentError> {
    to_text_with(
        bytes,
        name_hint,
        format,
        backend,
        pages,
        ConvertOptions::default(),
    )
}

/// [`to_text`] with the caller-tuned knobs (password, input ceiling).
pub fn to_text_with<'a>(
    bytes: Vec<u8>,
    name_hint: Option<&str>,
    format: Option<&str>,
    backend: Backend,
    pages: Option<Vec<usize>>,
    options: ConvertOptions<'a>,
) -> Result<Converted, DocumentError> {
    let (format, engine, markdown) = convert(bytes, name_hint, format, backend, pages, &options)?;
    // anydoc HTML-escapes `&` (and friends) in its markdown; the other three
    // engines emit text literally: un-escaping theirs would corrupt text
    // that contains `&amp;`.
    let unescape = engine == Engine::Anydoc;
    let text = gfm_strip_impl::strip(&markdown, unescape);
    Ok(Converted {
        format,
        output: text,
    })
}

/// The standalone content sniffer: what the conversion would resolve these
/// bytes to from content alone: no path, no extension. The PDF header, the
/// RTF open group, OLE stream names, the ZIP package mimetype, and the HTML
/// document marker. `None` = the content names no format (a signature-less
/// text format such as CSV, or not a document at all). The reported name is
/// the conversion's own [`resolved_name`]: Excel's one container-aware
/// case included: an OLE workbook sniffs as `xls`, a ZIP one as `xlsx`:
/// so `sniff` and `to_markdown` can never disagree about what the bytes
/// are.
pub fn sniff(bytes: &[u8]) -> Option<&'static str> {
    sniff_kind(bytes).map(|kind| resolved_name(kind, bytes))
}

/// The shared spine of both public functions: resolve, route, convert. The
/// bytes are already in hand: the payload read the `path=` (or took the
/// caller's `data=`) before calling here, so no IO happens in this seam.
fn convert(
    bytes: Vec<u8>,
    name_hint: Option<&str>,
    format: Option<&str>,
    backend: Backend,
    pages: Option<Vec<usize>>,
    options: &ConvertOptions,
) -> Result<(&'static str, Engine, String), DocumentError> {
    // A zero ceiling is caller nonsense, not a policy: it would either
    // refuse every input ("0.0 vs 0.0") or, on empty input, silently mean
    // "no document may exceed nothing" while converting nothing at all.
    // The binding layer already refuses it under the GIL; this makes the
    // core self-sufficient (the same self-sufficiency doctrine as
    // markdown_pages' selection normalization).
    if options.max_bytes.is_some_and(|limit| limit == 0) {
        return Err(DocumentError::Convert("max_bytes must be positive".into()));
    }
    let kind = resolve(&bytes, name_hint, format)?;
    let engine = engine_for(kind, backend)?;
    if pages.is_some() {
        if kind != Kind::Pdf {
            return Err(DocumentError::Pages(format!(
                "pages= applies to PDF documents only, not {:?}",
                kind.name()
            )));
        }
        if engine != Engine::PdfOxide {
            return Err(DocumentError::Pages(
                "pages= requires the pdf_oxide lane (backend='auto' or 'oxide')".into(),
            ));
        }
    }
    if options.password.is_some() && kind != Kind::Pdf {
        // a silently-ignored password is worse than an error: the caller
        // believes the document is protected when it is not the lane that
        // would know
        return Err(DocumentError::Convert(format!(
            "password= applies to PDF documents only, not {:?}",
            kind.name()
        )));
    }
    if options.password.is_some() && engine == Engine::Anydoc {
        // the anydoc PDF lane's reader takes no password at all: a correct
        // password would fail Convert("document is encrypted") exactly
        // like a wrong one: indistinguishable from having been ignored.
        // Refuse up front, mirroring the pages= lane guard.
        return Err(DocumentError::Convert(
            "password= requires the pdf_oxide lane (backend='auto' or 'oxide'); \
             anydoc's PDF reader takes no password"
                .into(),
        ));
    }
    // The input ceiling: the lanes that amplify their input into resident
    // memory: anydoc (~146x worst case on adversarial delimiter formats, and
    // engine-side decompression caps on top; see [DEFAULT_ANYDOC_INPUT_LIMIT]),
    // office_oxide (512 MiB per part since 0.1.10, plus the seam's
    // total-across-parts audit below: [audit_oxide_container]; the 399 KiB
    // zip-bomb docx with a 400 MiB part that peaked at 1.58 GiB RSS was
    // measured through that per-part cap before the total audit existed),
    // and HTML (~23x input: the converter holds the whole input and output
    // at once; a 48 MiB doctype HTML peaked at 1118 MiB, measured), so the
    // last gets the same input-side bound as the first two.
    // What no ceiling here does is bound OUTPUT: the inflated bytes are
    // capped on the office lanes (engine-side, or at the seam on the oxide
    // lane) and nowhere else.
    if engine == Engine::Anydoc || engine == Engine::OfficeOxide || engine == Engine::Html2Md {
        let limit = options.max_bytes.unwrap_or(DEFAULT_ANYDOC_INPUT_LIMIT);
        if bytes.len() > limit {
            return Err(DocumentError::Convert(format!(
                "the document is {} and the {} engine lane's input ceiling is {} \
                 (that lane amplifies input into resident memory at a measured multiple \
                 — 2026-09-09: ~146x worst case on anydoc's delimiter formats, \
                 ~4100x on a zip-bombed office_oxide container): split the file, or pass a larger \
                 max_bytes",
                render_size(bytes.len()),
                engine_name(engine),
                render_size(limit),
            )));
        }
    }
    // resolved before the engine match: the pdf lane moves the bytes
    let resolved = resolved_name(kind, &bytes);
    let markdown = match engine {
        Engine::PdfOxide => pdf_impl::markdown_pages(bytes, pages.as_deref(), options.password)
            .map_err(pages_error)?,
        Engine::OfficeOxide => office_markdown(bytes, kind)?,
        Engine::Anydoc => {
            anydoc::to_markdown_bytes(&bytes, Some(kind.to_anydoc())).map_err(anydoc_error)?
        }
        Engine::Html2Md => html_markdown(&bytes)?,
    };
    Ok((resolved, engine, markdown))
}

/// Read from `reader` into `buf` (which may already hold a prefix) until EOF
/// or one byte past `ceiling`. Returns the bytes and whether the input
/// exceeded `ceiling`. Never allocates more than `ceiling + 1 + CHUNK` and
/// never grows through infallible `reserve`, so an oversized input surfaces
/// as an `OutOfMemory` io error rather than an allocation abort. This is the
/// piece that closes issue #80 symptoms 2 and 3: the read is self-limiting
/// regardless of what the file's stat claimed. Callers guarantee
/// `buf.len() <= ceiling` on entry (the prefix is always <= SNIFF_PREFIX + 1,
/// which is below any ceiling used here).
pub fn read_bounded_into<R: std::io::Read>(
    mut reader: R,
    mut buf: Vec<u8>,
    ceiling: usize,
) -> std::io::Result<(Vec<u8>, bool)> {
    const CHUNK: usize = 64 * 1024;
    loop {
        if buf.len() > ceiling {
            return Ok((buf, true));
        }
        // The `buf.len() > ceiling` guard above guarantees
        // `ceiling.saturating_add(1) - buf.len() >= 1`, so `want >= 1`; the
        // saturating add keeps the reader sound even at `ceiling == usize::MAX`
        // (a plain `ceiling + 1` would overflow there).
        let want = std::cmp::min(CHUNK, ceiling.saturating_add(1) - buf.len());
        let start = buf.len();
        buf.try_reserve_exact(want).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::OutOfMemory, "input too large to buffer")
        })?;
        buf.resize(start + want, 0);
        let n = match reader.read(&mut buf[start..]) {
            Ok(n) => n,
            // Retry a signal that landed mid-read, for parity with the
            // `read_to_end` this loop replaced (CPython installs handlers
            // without SA_RESTART, so EINTR reaches us). Real io errors return.
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                buf.truncate(start);
                continue;
            }
            Err(e) => return Err(e),
        };
        buf.truncate(start + n);
        if n == 0 {
            return Ok((buf, false));
        }
    }
}

/// The most bytes worth reading before the authoritative `resolve()` runs,
/// judged from a prefix. Content markers decide, exactly as `resolve()` does,
/// so a `.pdf`-named CSV is treated as the metered lane it really is. A
/// metered lane (anydoc/office_oxide/html) returns `DEFAULT_ANYDOC_INPUT_LIMIT`;
/// only a prefix that POSITIVELY resolves to the unmetered pdf lane
/// returns `fallback`. A prefix whose lane cannot be told at all (a truncated
/// ZIP central directory, a content-blind leader) also returns the metered
/// ceiling, erring safe: content that never resolves is refused by `resolve()`
/// anyway, and content that resolves only past the prefix is either a metered
/// lane (the same 32 MiB ceiling, post-read — enforced here during the read)
/// or a marker-buried pdf (an explicit `max_bytes=` unlocks it). Without
/// this, a 64 KiB content-blind leader on an extensionless metered file would
/// hand the attacker the fallback and move the refusal post-read. Only
/// complete lines of the prefix are considered, so a prefix cut mid-line does
/// not skew the CSV heuristic — but the untrimmed prefix is checked too, so a
/// long partial final line (a CSV row wider than the prefix, no trailing
/// newline) whose completeness the trim would erase is still metered.
pub fn provisional_read_ceiling(
    prefix: &[u8],
    name_hint: Option<&str>,
    format: Option<&str>,
    backend: Backend,
    fallback: usize,
) -> usize {
    // Some(true) = a metered lane (anydoc/office_oxide/html); Some(false) =
    // positively unmetered (pdf_oxide); None = the prefix does not
    // resolve at all. None is deliberately NOT the fallback: see below.
    let lane = |bytes: &[u8]| {
        resolve(bytes, name_hint, format)
            .ok()
            .and_then(|kind| engine_for(kind, backend).ok())
            .map(|engine| engine != Engine::PdfOxide)
    };
    // Drop a trailing partial line so the CSV witness sees only whole lines.
    let end = match prefix.iter().rposition(|&b| b == b'\n') {
        Some(nl) => nl + 1,
        None => prefix.len(),
    };
    let head = &prefix[..end];
    // Meter unless a view POSITIVELY resolves to the unmetered pdf lane. The
    // trim protects the mid-line-cut delimiter case; the untrimmed check
    // closes the long-row CSV gap where the trim would drop the only witness
    // lines. An all-None prefix (unresolvable content, no name to consult)
    // meters — the safe direction, matching the OR's own bias toward metering.
    let unmetered = lane(head) == Some(false) || lane(prefix) == Some(false);
    if unmetered {
        fallback
    } else {
        DEFAULT_ANYDOC_INPUT_LIMIT
    }
}

/// The geometry surface's pre-read capability probe: the
/// [`anydoc_capability_refusal`] the `backend="anydoc"` call will raise,
/// resolved as far as the format resolves WITHOUT the authoritative
/// (whole-document) read — an explicit `format=` decides alone, then the
/// content markers in `bytes` (the `data=` lane's own bytes; the `path=`
/// lane's bounded 64 KiB sniff prefix), then the name hint's extension.
///
/// - `Err(refusal)`: the probed format is on a row anydoc cannot serve
///   (the pdf row, the office row — the refusal names the container, so
///   an OLE workbook says `xls`). The binding raises it under the GIL,
///   before the detach and before the read: the four PDF-only siblings'
///   exact ordering, which this surface could not keep while its refusal
///   needed a resolved format. (§6's resolve-first clause is why the
///   probe cannot fire on an unresolvable prefix — a refusal cannot name
///   a format it has not resolved.)
/// - `Ok(None)`: the probe could not resolve the format. The core's
///   authoritative `resolve` — on the real bytes, post-read — stays the
///   only truth; whatever refusal fires there fires then.
/// - `Ok(Some(kind))`: resolved, and the row serves this backend.
pub fn probe_geometry_capability(
    bytes: &[u8],
    name_hint: Option<&str>,
    format: Option<&str>,
    backend: Backend,
) -> Result<Option<Kind>, DocumentError> {
    match resolve(bytes, name_hint, format) {
        Err(_) => Ok(None),
        Ok(kind) => geometry_engine(kind, backend, bytes).map(|_| Some(kind)),
    }
}

/// The geometry surface's provisional read ceiling —
/// [`provisional_read_ceiling`] keyed on the GEOMETRY capability table
/// ([`geometry_engine`]), not the conversion routing table, because the
/// two meter different lanes: the only amplifying lane on this surface
/// is the office leg (its parse materializes the container into resident
/// memory — the same amplification class the conversion lane meters), so
/// the 32 MiB default applies to the office row ALONE. The pdf row
/// (pdf_oxide's own limits govern) and the no-geometry row (no engine
/// runs at all — spec §6 says so) carry NO default ceiling: an
/// unbounded-prefix 40 MiB HTML is answered `[None]`, not refused, on
/// this surface, `path=` and `data=` alike (the parity the conversion
/// table's ceiling broke). An explicit `max_bytes` still binds
/// everywhere, pre-read — this function is only consulted under `None`.
/// An unresolvable prefix meters — the safe direction (the office leg's
/// post-read default still binds if the full bytes resolve there), the
/// conversion twin's own bias.
pub fn provisional_geometry_read_ceiling(
    prefix: &[u8],
    name_hint: Option<&str>,
    format: Option<&str>,
    backend: Backend,
    fallback: usize,
) -> usize {
    // Some(Some(office)) = the amplifying row; Some(Some(pdf)) /
    // Some(None) = rows with no default ceiling; None = the prefix does
    // not resolve at all (meters, the safe direction).
    let lane = |bytes: &[u8]| {
        resolve(bytes, name_hint, format)
            .ok()
            .and_then(|kind| geometry_engine(kind, backend, bytes).ok())
    };
    let no_default_ceiling = |resolved: Option<Option<Engine>>| {
        matches!(resolved, Some(Some(Engine::PdfOxide)) | Some(None))
    };
    // Drop a trailing partial line so the CSV witness sees only whole
    // lines — the conversion twin's trim (the untrimmed view below closes
    // the long-row gap the trim opens).
    let end = match prefix.iter().rposition(|&b| b == b'\n') {
        Some(nl) => nl + 1,
        None => prefix.len(),
    };
    let head = &prefix[..end];
    let unmetered = no_default_ceiling(lane(head)) || no_default_ceiling(lane(prefix));
    if unmetered {
        fallback
    } else {
        DEFAULT_ANYDOC_INPUT_LIMIT
    }
}

/// Whether an explicit format NAME lands on the geometry table's
/// no-geometry row (the format-determined single `[None]` answer): the
/// binding layer's no-read door's predicate. When the caller named the
/// format, the core's `resolve` short-circuits on the name and the
/// geometry table runs no engine on this row — so the bytes can be
/// consulted by nobody, and the file need not be read at all. Keyed on
/// the table itself, never a second spelling of its row.
pub fn is_no_geometry_format(name: &str) -> bool {
    Kind::from_name(name)
        .is_some_and(|kind| geometry_engine(kind, Backend::Auto, &[]).ok() == Some(None))
}

/// Resolve the format: the explicit name (any leading dot tolerated) beats
/// the content markers, which beat the name hint's extension, and the
/// extension resolves through the same [`Kind::from_name`] vocabulary the
/// explicit name uses, so `file.xhtml`/`file.tsv`/`file.html` work by
/// extension exactly as `format="xhtml"`/`"tsv"`/`"html"` do by name, with
/// no second spelling table to keep in step (anydoc's own extension
/// vocabulary rides inside `from_name`; its `from_path` adds nothing to
/// it). `UnknownFormat` carries what was tried: the name hint when there
/// was one, the no-hint doctrine's fix (an explicit `format=`) when the
/// bytes came in bare.
fn resolve(
    bytes: &[u8],
    name_hint: Option<&str>,
    format: Option<&str>,
) -> Result<Kind, DocumentError> {
    if let Some(named) = format {
        return Kind::from_name(named)
            .ok_or_else(|| DocumentError::UnknownFormat(format!("unknown format name {named:?}")));
    }
    if let Some(kind) = sniff_kind(bytes) {
        return Ok(kind);
    }
    if let Some(kind) = name_hint
        .and_then(|hint| hint.rsplit('.').next())
        .and_then(Kind::from_name)
    {
        return Ok(kind);
    }
    let what = match name_hint {
        Some(hint) => format!("unrecognized content and extension: {hint}"),
        // the data= entry: no name to consult: the fix is the explicit
        // name (or sniff() on the same bytes to see what the markers said)
        None => "unrecognized content and no name to consult: pass format= \
                 (or sniff(data) first)"
            .into(),
    };
    Err(DocumentError::UnknownFormat(what))
}

/// The content-marker sniffer: anydoc's `from_bytes` first (its markers are
/// unambiguous binary signatures), then the HTML document marker: an HTML
/// page is text, so it can only be decided after the binary formats had
/// their chance.
fn sniff_kind(bytes: &[u8]) -> Option<Kind> {
    if let Some(format) = Format::from_bytes(bytes) {
        return Some(Kind::from_anydoc(format));
    }
    if looks_like_html(bytes) {
        return Some(Kind::Html);
    }
    looks_like_csv(bytes).then_some(Kind::Csv)
}

/// The HTML document marker: after a UTF-8 BOM, whitespace, and one
/// optional XML declaration (XHTML's `<?xml version="1.0"…?>` prologue,
/// which precedes the doctype and would hide it from a prefix-anchored
/// check), the first markup is a document type declaration or the
/// `<html>` element open (case-insensitive: both `<!DOCTYPE html…` and
/// `<HTML…` are real-world spellings). A leading comment (`<!--`) is not
/// taken as a marker: it is not specific to HTML, and comment-prefixed
/// non-HTML text is rare enough that the extension fallback should decide
/// it. Nor is any leading `<tag`: an HTML fragment (no doctype, no
/// `<html>`, e.g. starting `<div>`/`<p>`) is deliberately not
/// content-resolvable: every XML vocabulary opens with a tag (`<svg`,
/// `<book`, …), so a fragment marker would claim all of them as HTML. A
/// fragment needs `format="html"` or an `.html`-family extension, both of
/// which resolve it through the shared name vocabulary.
fn looks_like_html(bytes: &[u8]) -> bool {
    let head = bytes
        .strip_prefix([0xEF, 0xBB, 0xBF].as_slice())
        .unwrap_or(bytes);
    let text = String::from_utf8_lossy(head);
    let mut probe = text.trim_start();
    if probe.starts_with("<?xml") {
        // `?>` is ASCII, so the byte-level search cannot land mid-codepoint;
        // the 1 KiB bound keeps the skip constant-time against adversarial
        // input (a real declaration is a few dozen bytes by spec).
        let Some(end) = probe.as_bytes()[..probe.len().min(1024)]
            .windows(2)
            .position(|w| w == b"?>")
        else {
            // Unterminated declaration: malformed markup, not a marker:
            // the extension fallback decides.
            return false;
        };
        probe = probe[end + 2..].trim_start();
    }
    let window: String = probe
        .chars()
        .take(14)
        .flat_map(char::to_lowercase)
        .collect();
    window.starts_with("<!doctype html") || window.starts_with("<html")
}

/// The CSV heuristic: the last resort of content resolution, after every
/// binary marker and the HTML check declined: text, not markup, where the
/// first up-to-64 non-empty lines each contain the same count (≥1) of one
/// delimiter candidate (`,` / `;` / tab, tried in that order). Two lines
/// minimum: a single line of comma-separated words is prose, not a table.
/// The edge behavior, probed and pinned in the tests: a UTF-8 BOM rides
/// the first field harmlessly (it is not a delimiter); `lines()` strips a
/// trailing `\r`, so CRLF files count cleanly (a lone `\r` is not a line
/// ending: classic-Mac files are not heuristically CSV); quoted commas
/// that agree across lines route here (anydoc's quote-aware parser does
/// the real reading: the heuristic's job is routing, not parsing), while
/// quoted commas that differ are prose-shaped and rejected; a
/// single-column file carries 0 delimiters on every line and is rejected
/// (name it with `format=` if it is one); UTF-16 text fails the UTF-8
/// gate, but the `format="csv"` escape hatch reaches anydoc's BOM-aware
/// decoder. A false positive requires prose whose every line carries an
/// identical comma count: the price of resolving signature-less CSV from
/// content alone, which the no-extension routing doctrine (temp files with
/// no suffix, so content, never the name, picks the extractor) demands;
/// an explicit `format=` always overrides the heuristic, and every
/// marker-bearing format was already ruled out before it runs.
fn looks_like_csv(bytes: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(64)
        .collect();
    if lines.len() < 2 {
        return false;
    }
    // JSON-Lines is the one structured format whose comma counts agree
    // across lines (every record serializes the same keys), so the
    // delimiter witness alone would claim it, and anydoc's csv parser
    // would then mangle records that are not cells. A record line opens
    // with `{` (or `[`): not this heuristic's format. The cost is the
    // rare csv whose first field opens with a brace: name it with
    // format=, the same escape hatch every other rejected shape has.
    if lines
        .first()
        .is_some_and(|first| first.starts_with('{') || first.starts_with('['))
    {
        return false;
    }
    const CANDIDATES: [char; 3] = [',', ';', '\t'];
    CANDIDATES.iter().any(|delimiter| {
        let counts: Vec<usize> = lines
            .iter()
            .map(|line| line.chars().filter(|c| c == delimiter).count())
            .collect();
        counts.iter().all(|count| *count >= 1) && counts.iter().all(|count| *count == counts[0])
    })
}

/// The oxide lane's decompression audit, run on every office container
/// before [`office_oxide::Document::from_reader`] sees it: refuse a zip
/// whose parts inflate past [`OXIDE_MAX_TOTAL_DECOMPRESSED`] in total.
/// Two stages, both over the container's own zip central directory
/// (parsed here, not through the `zip` crate: it is in the lock only as
/// anydoc's and office_oxide's transitive, and the audit needs exactly
/// four record layouts, not an archive API):
///
/// 1. the **declared pass** sums every entry's declared uncompressed size
///    and refuses past the ceiling without inflating a byte: the honest
///    multi-part bomb (600 MiB declared across two 300 MiB parts, each
///    far under office_oxide's 512 MiB per-part cap, so that cap cannot
///    fire) dies here sub-ms at ~40 MiB RSS;
/// 2. the **actual pass** does not trust those sizes (they are
///    attacker-controlled and office_oxide validates none of them:
///    measured, parts declaring 1,000 bytes inflate to 300 MiB each and
///    convert identically): it re-inflates every part itself, each under
///    a hard `take(ceiling + 1)`, counts the bytes through a 64 KiB
///    scratch (nothing is ever materialized at part size), and refuses
///    when the running sum passes the ceiling. office_oxide reads a
///    subset of the entries audited here, under its own per-part cap, so
///    a container this audit passes cannot hand it more than the ceiling
///    of inflated bytes in total.
///
/// The audit reads the exact spans the reader will: data located through
/// each entry's LOCAL header (whose name/extra lengths can differ from
/// the central record's), sized by the central directory's compressed
/// size, rebased by the same prepend delta the reader computes (its
/// search for the first central-directory signature). The directory
/// resolution mirrors the reader's own ladder field for field,
/// including the two places where the zip 8.6 reader's field choice is
/// not what a spec reading would suggest: it counts records from the
/// EOCD's on-this-disk field (not the total-entries field, whose only
/// reader-visible roles are the empty-archive early accept and the
/// zip64 sentinel trigger), and a zip64 locator that parses puts the
/// candidate on a hard zip64 branch with no zip32 fallback. Both were
/// measured as live bypass shapes before this mirror existed: an EOCD
/// declaring 1 total entry against 5 on-this-disk, and one declaring 0
/// total against 2 with the directory planted after the EOCD, each
/// passed the audit while the engine inflated 600 MiB at ~2.1 GiB peak
/// RSS from a ~600 KiB input (2026-09-16). Where the audit's model and
/// the reader's acceptance can still part ways (a known extra field
/// whose content parser rejects a record is the residual family), the
/// union closes it: every end-of-directory candidate that resolves has
/// its records audited, not just the one the reader would pick, so a
/// reader that falls back to an earlier signature lands on records the
/// audit has already bounded. Anything the directory parse cannot
/// resolve is refused, never skipped: a container the reader can open
/// while the audit cannot is precisely the shape that must fail closed.
/// Legacy OLE containers (doc/xls/ppt) have no zip directory:
/// office_oxide reads them through its own CFB reader, whose sector
/// addressing is bounded by the input length.
fn audit_oxide_container(bytes: &[u8]) -> Result<(), DocumentError> {
    if is_ole_container(bytes) {
        return Ok(());
    }
    let Some(parts) = zip_parts(bytes)? else {
        return Ok(());
    };
    // Stage 1: declared sizes. One number, attacker-supplied, so the pass
    // is a cheap pre-filter (sub-ms), not the bound.
    let mut declared_total: u64 = 0;
    for part in &parts {
        declared_total = declared_total.saturating_add(part.uncompressed_size);
        if declared_total > OXIDE_MAX_TOTAL_DECOMPRESSED {
            return Err(Convert(format!(
                "office_oxide: decompression limit exceeded: part '{}' declares {} uncompressed \
                 bytes ({} across all parts), over this lane's {}-byte ceiling across all parts \
                 of one container (office_oxide's own cap bounds each part separately and cannot \
                 see the sum)",
                part.name, part.uncompressed_size, declared_total, OXIDE_MAX_TOTAL_DECOMPRESSED,
            )));
        }
    }
    // Stage 2: the inflated bytes themselves, the bound the declared sizes
    // cannot give.
    let mut inflated_total: u64 = 0;
    for part in &parts {
        inflated_total = inflated_total.saturating_add(part_inflated_bytes(bytes, part)?);
        if inflated_total > OXIDE_MAX_TOTAL_DECOMPRESSED {
            return Err(Convert(format!(
                "office_oxide: decompression limit exceeded: the container's parts inflate past \
                 this lane's {}-byte ceiling across all parts while their declared sizes sum to \
                 only {} (a declared size does not bound what a part inflates to)",
                OXIDE_MAX_TOTAL_DECOMPRESSED, declared_total,
            )));
        }
    }
    Ok(())
}

/// One central-directory entry, the fields the audit consumes. The name is
/// message-furniture only (lossy-decoded); `header_start` is the LOCAL
/// header's position after the prepend rebase.
struct ZipPart {
    name: String,
    method: u16,
    encrypted: bool,
    compressed_size: u64,
    uncompressed_size: u64,
    header_start: u64,
}

/// Every central-directory entry the reader could resolve, or `None` when
/// the bytes carry no end-of-central-directory signature at all (not a
/// zip: office_oxide's own error paths stand, and they cannot inflate
/// anything). Zip signatures that never resolve to a valid directory are
/// an `Err`: the reader's directory search (its retry ladder, prepend
/// handling, zip64 fallbacks) accepts strictly more shapes than this
/// parse, and a container it can open while the audit cannot is the
/// evasion shape. Candidates are tried last-first, the reader's own
/// backward order, but not first-wins: the records of EVERY candidate
/// that resolves (under [`resolve_eocd`], the reader's own acceptance
/// rules) are unioned. The reader itself stops at the first candidate it
/// accepts and only walks backwards again when one rejects, so the union
/// is a superset of the records it can ever materialize, which is the
/// point: a reader that rejects a candidate for a reason this mirror
/// does not reproduce (a known extra field whose content parser errors
/// is the residual family) falls back to an earlier signature whose
/// records are already in the union, already bounded. A record walk is
/// bounded by the end of file, not by the candidate's own position: the
/// reader's records are read sequentially off the whole-file cursor and
/// only the file's end stops them, so a directory planted after the end
/// record (the total-entries-is-zero shape) is walked here exactly as
/// the reader walks it.
fn zip_parts(bytes: &[u8]) -> Result<Option<Vec<ZipPart>>, DocumentError> {
    let unauditable = |why: &str| {
        Convert(format!(
            "office_oxide: the container's zip directory did not parse under this lane's \
             decompression audit: {why}"
        ))
    };
    let mut saw_signature = false;
    let mut resolved_any = false;
    let mut union: Vec<ZipPart> = Vec::new();
    // Candidates by backward scan on the signature's last byte (the rare
    // one: memchr's search over the reader's own whole-file range);
    // `probe` shrinks past each hit so earlier signatures surface next.
    let mut probe = bytes.len();
    while let Some(hit) = memchr::memrchr(0x06, &bytes[..probe]) {
        probe = hit;
        let Some(pos) = hit.checked_sub(3) else { break };
        if !bytes[pos..=hit].starts_with(&ZIP_EOCD_SIG) {
            continue;
        }
        saw_signature = true;
        let Some(found) = resolve_eocd(bytes, pos) else {
            // The reader rejects this candidate (its disks, its comment,
            // its zip64 follow-ups) and tries an earlier signature: so
            // does this scan. Rejection is never silent acceptance of the
            // remaining bytes; the union below still covers every earlier
            // candidate that resolves.
            continue;
        };
        let ResolvedEocd {
            records,
            base,
            delta,
        } = found;
        if records == 0 {
            // An empty directory contributes nothing; the reader's own
            // missing-part error stands for the candidate it picks.
            resolved_any = true;
            continue;
        }
        let Some(base) = usize::try_from(base).ok() else {
            continue;
        };
        // A record needs at least its 46 fixed bytes: an entry count the
        // span between the directory's first record and the end of file
        // cannot physically hold is forged, and walking it is waste (the
        // reader's own sequential walk would die the same way, rejecting
        // the candidate).
        if records.saturating_mul(46) > bytes.len().saturating_sub(base) as u64 {
            continue;
        }
        let mut parts = Vec::new();
        let mut at = base;
        let mut parsed = true;
        for _ in 0..records {
            match parse_cd_record(bytes, at, bytes.len(), delta) {
                Some((next, part)) => {
                    parts.push(part);
                    at = next;
                }
                None => {
                    parsed = false;
                    break;
                }
            }
        }
        if parsed {
            // Only a candidate whose records fully parse counts as
            // resolved: one whose walk dies mid-record is a candidate the
            // reader's own walk rejects the same way, so neither side
            // ever reads a partial take, and the audit says so in its
            // own refusal voice instead of handing the engine a silently
            // emptier directory than any candidate declared.
            resolved_any = true;
            union.extend(parts);
        }
    }
    if !saw_signature {
        return Ok(None);
    }
    if resolved_any {
        Ok(Some(union))
    } else {
        Err(unauditable(
            "no end-of-directory candidate resolved to a central directory",
        ))
    }
}

/// One end-of-directory candidate, resolved the way the reader resolves
/// it: `None` (from [`resolve_eocd`]) is a candidate the reader itself
/// rejects (and so retries an earlier signature; this scan does the
/// same), and the struct carries the three numbers the reader's
/// directory walk is fully determined by: how many records it reads,
/// where the first one sits, and the prepend delta it adds to every
/// record's local-header offset.
struct ResolvedEocd {
    records: u64,
    base: u64,
    delta: u64,
}

/// Resolve one end-of-directory candidate at `pos` under the reader's own
/// acceptance ladder: the fixed fields must read (a truncated record is
/// rejected), the comment must fit to the end of file (the reader's
/// relaxed rule: it may end early against garbage-after-comment writers,
/// but never run past the end), the two disk fields must agree (the
/// reader's multi-disk refusal), and a zip64-capped record (any of the
/// three sentinel fields) with a parseable locator takes the reader's
/// HARD zip64 branch ([`resolve_zip64`]) with no zip32 fallback. An
/// absent locator leaves the zip32 values, exactly the reader's own
/// fallback, and the zip32 resolution ([`resolve_zip32`]) models the
/// two field choices that are not what the record's names suggest: the
/// record count comes from the on-this-disk field, and a total-entries
/// of zero takes the reader's early accept whose directory starts at the
/// declared offset itself.
fn resolve_eocd(bytes: &[u8], pos: usize) -> Option<ResolvedEocd> {
    let comment_len = le16(bytes, pos + 20)? as usize;
    let end = pos.checked_add(22)?.checked_add(comment_len)?;
    if end > bytes.len() {
        return None;
    }
    if le16(bytes, pos + 4)? != le16(bytes, pos + 6)? {
        // multi-disk: the reader's UnsupportedArchive, a candidate rejection
        return None;
    }
    let on_this_disk = u64::from(le16(bytes, pos + 8)?);
    let total = u64::from(le16(bytes, pos + 10)?);
    let cd_size = le32(bytes, pos + 12)?;
    let cd_offset = u64::from(le32(bytes, pos + 16)?);
    // The reader's may_be_zip64: any one of the three sentinel fields puts
    // the candidate on the zip64 ladder (a locator that fails to parse
    // falls back to these zip32 values, the reader's own behavior).
    let zip64_capped =
        total == u64::from(u16::MAX) || cd_size == u32::MAX || cd_offset == u64::from(u32::MAX);
    if zip64_capped && let Some(locator) = read_zip64_locator(bytes, pos) {
        return resolve_zip64(bytes, pos, locator);
    }
    resolve_zip32(bytes, pos, on_this_disk, total, cd_offset)
}

/// The zip32 resolution, the reader's two paths. A nonzero total takes the
/// search path: the directory cannot start at or after this record, and
/// its first record is the first central signature from the declared
/// offset up to this record (prepended junk moved the real directory
/// forward; the delta that search computes rebases every local offset).
/// The record count is the on-this-disk field unioned with the total: the
/// reader counts records from the on-this-disk field alone (zip 8.6's
/// CentralDirectoryInfo reads that field, not the total, the divergence
/// this union is written for), and the total is unioned in so the audited
/// set covers either field a future reader might count from. A total of
/// zero takes the reader's early accept instead: no search, the directory
/// is the declared offset itself when that lies past this record (a
/// trailing directory, delta zero, the saturating archive offset the
/// reader computes clamps to zero), and when it does not, the directory
/// would start on this record's own signature, a magic mismatch the
/// reader's record parse rejects the whole candidate for.
fn resolve_zip32(
    bytes: &[u8],
    pos: usize,
    on_this_disk: u64,
    total: u64,
    cd_offset: u64,
) -> Option<ResolvedEocd> {
    if total == 0 {
        if on_this_disk == 0 {
            // The reader's empty archive: zero records walk, whatever the
            // declared offset was, and its missing-part error stands.
            return Some(ResolvedEocd {
                records: 0,
                base: pos as u64,
                delta: 0,
            });
        }
        let base = cd_offset.max(pos as u64);
        if base == pos as u64 {
            return None;
        }
        return Some(ResolvedEocd {
            records: on_this_disk,
            base,
            delta: 0,
        });
    }
    if cd_offset >= pos as u64 {
        // the reader's "Invalid CDFH offset in EOCD": the directory cannot
        // start at or after the end record on this path
        return None;
    }
    let base = locate_cd_base(bytes, cd_offset, pos)?;
    let delta = base as u64 - cd_offset;
    Some(ResolvedEocd {
        records: on_this_disk.max(total),
        base: base as u64,
        delta,
    })
}

/// The zip64 end-of-directory locator, 20 bytes before the zip32 end
/// record: its signature and the three fields the reader's hard zip64
/// branch consumes (the disk the directory sits on, the offset of the
/// zip64 end record itself, and the disk count). `None` when the record
/// is missing or mistyped, which is NOT a candidate rejection: the
/// reader's own fallback for an unparseable locator is the zip32 values,
/// and [`resolve_eocd`] does the same.
fn read_zip64_locator(bytes: &[u8], eocd_pos: usize) -> Option<(u32, u64, u32)> {
    let locator = eocd_pos.checked_sub(20)?;
    if !bytes
        .get(locator..locator.checked_add(4)?)?
        .starts_with(&ZIP64_LOCATOR_SIG)
    {
        return None;
    }
    Some((
        le32(bytes, locator + 4)?,
        le64(bytes, locator + 8)?,
        le32(bytes, locator + 16)?,
    ))
}

/// The hard zip64 branch a parseable locator puts the candidate on: the
/// reader searches FORWARD from the locator's declared record offset for
/// the first zip64 end-record signature before the locator (it does not
/// trust the declared offset as the record's position: the delta it
/// computes for the whole archive comes from where the record is FOUND),
/// and every check along the way is a candidate rejection, never a zip32
/// fallback: the record's own size field must equal the distance from
/// the record to the locator, the record's extensible tail must fit
/// before the end of file, the record's disk must be the locator's disk
/// and its two disk fields must agree, the record must sit past
/// `count x 46 + declared directory offset` (the directory has to fit
/// before it), the on-this-disk count cannot exceed the total, and the
/// rebased directory start (`declared offset + delta`) must not wrap.
/// On success the directory is EXACTLY the rebased offset (the zip64
/// path does no first-signature search, unlike the zip32 one), and
/// `delta` rebases every local offset.
fn resolve_zip64(
    bytes: &[u8],
    eocd_pos: usize,
    (locator_disk, declared_record, disks): (u32, u64, u32),
) -> Option<ResolvedEocd> {
    let locator = eocd_pos.checked_sub(20)?;
    // The reader's two locator-level candidate rejections.
    if declared_record >= locator as u64 || disks > 1 {
        return None;
    }
    let start = usize::try_from(declared_record).ok()?;
    let window = bytes.get(start..locator)?;
    // Forward search for the record, first hit wins (the reader's
    // OptimisticMagicFinder over the same window).
    let mut probe = 0;
    while let Some(hit) = memchr::memchr(0x06, &window[probe..]) {
        let at = probe + hit;
        probe = at + 1;
        let Some(record) = at.checked_sub(3) else {
            continue;
        };
        if !window[record..=at].starts_with(&ZIP64_EOCD_SIG) {
            continue;
        }
        let record = start + record;
        let delta = record as u64 - declared_record;
        // try_read_eocd64's checks, in the reader's order; a failed check
        // rejects this POSITION (the search continues), not the candidate.
        let record_size = le64(bytes, record + 4)?;
        if record_size < 40 {
            continue;
        }
        if record_size.checked_add(12) != Some((locator - record) as u64) {
            continue;
        }
        // The record's fixed 56 bytes plus any extensible tail past the
        // 44-byte minimum must fit before the end of file: the reader
        // reads them off the whole-file cursor.
        let tail = record_size.saturating_sub(44) as usize;
        let record_end = record.checked_add(56)?.checked_add(tail)?;
        if record_end > bytes.len() {
            continue;
        }
        let disk = le32(bytes, record + 16)?;
        let disk_with = le32(bytes, record + 20)?;
        if disk_with != locator_disk {
            continue;
        }
        let on_this_disk = le64(bytes, record + 24)?;
        let count = le64(bytes, record + 32)?;
        let cd_offset = le64(bytes, record + 48)?;
        if (record as u64) < count.saturating_mul(46).saturating_add(cd_offset) {
            continue;
        }
        // CentralDirectoryInfo's and read_central_header's checks: these
        // reject the whole candidate, like the reader's retry ladder.
        if disk != disk_with || on_this_disk > count {
            return None;
        }
        let base = cd_offset.checked_add(delta)?;
        return Some(ResolvedEocd {
            records: count,
            base,
            delta,
        });
    }
    // The locator parsed but no record resolved: the reader rejects the
    // candidate outright: there is no zip32 fallback once the locator is
    // parseable, the divergence the old optional-zip64 enrichment missed.
    None
}

/// Where the central directory's records actually start. The common case
/// is the declared offset; a prepended container (the reader's prepend
/// handling) moved them together, so the fallback is the reader's own
/// move: the first central-directory signature between the declared
/// offset and the end record, and every local offset shifts by the same
/// delta. The search anchors on the signature's LAST byte (0x02) and
/// matches the four bytes ENDING there; anchoring on the first byte and
/// matching forward, or matching AT the anchor, finds nothing (the
/// signature never starts with its own last byte), which is exactly how
/// the prepended shape used to die unauditable before this was pinned.
fn locate_cd_base(bytes: &[u8], cd_offset: u64, limit: usize) -> Option<usize> {
    let stated = usize::try_from(cd_offset).ok()?;
    let stated_end = stated.checked_add(4)?;
    if bytes.get(stated..stated_end)?.starts_with(&ZIP_CENTRAL_SIG) {
        return Some(stated);
    }
    let window = bytes.get(stated..limit)?;
    let mut probe = 0;
    while let Some(hit) = memchr::memchr(0x02, &window[probe..]) {
        let at = probe + hit;
        probe = at + 1;
        let Some(sig) = at.checked_sub(3) else {
            continue;
        };
        if window[sig..=at].starts_with(&ZIP_CENTRAL_SIG) {
            return Some(stated + sig);
        }
    }
    None
}

/// One central-directory record: the fields the audit consumes plus the
/// next record's offset (records end before `limit`, the end of the file:
/// the reader reads records sequentially off the whole-file cursor and
/// only the file's end stops them; a record run may cross the end
/// record's own position, and a trailing directory past it is exactly the
/// total-entries-is-zero shape). `delta` is the prepend rebase the reader
/// applies to every local offset. The extra field is walked with the
/// reader's own well-formedness rules ([`zip64_extra`]): a chunk the
/// reader's parse would reject rejects the RECORD, which rejects the
/// whole candidate, never a silent fall-back to the fixed values over
/// bytes the reader would have refused.
fn parse_cd_record(bytes: &[u8], at: usize, limit: usize, delta: u64) -> Option<(usize, ZipPart)> {
    let fixed_end = at.checked_add(46)?;
    if fixed_end > limit || !bytes[at..at + 4].starts_with(&ZIP_CENTRAL_SIG) {
        return None;
    }
    let flags = le16(bytes, at + 8)?;
    let method = le16(bytes, at + 10)?;
    let mut compressed_size = le32(bytes, at + 20)? as u64;
    let mut uncompressed_size = le32(bytes, at + 24)? as u64;
    let name_len = le16(bytes, at + 28)? as usize;
    let extra_len = le16(bytes, at + 30)? as usize;
    let comment_len = le16(bytes, at + 32)? as usize;
    let mut header_start = le32(bytes, at + 42)? as u64;
    let name_end = fixed_end.checked_add(name_len)?;
    let extra_end = name_end.checked_add(extra_len)?;
    let record_end = extra_end.checked_add(comment_len)?;
    if record_end > limit {
        return None;
    }
    if let Some(extra) = bytes.get(name_end..extra_end) {
        match zip64_extra(extra, uncompressed_size, compressed_size, header_start) {
            Zip64Extra::Absent => {}
            Zip64Extra::Parsed {
                size,
                compressed,
                header,
            } => {
                uncompressed_size = size;
                compressed_size = compressed;
                header_start = header;
            }
            // A malformed extra field is a record the reader's own parse
            // rejects the whole candidate for: this walk does the same,
            // and the candidate scan moves to an earlier signature.
            Zip64Extra::Malformed => return None,
        }
    }
    let name = String::from_utf8_lossy(bytes.get(fixed_end..name_end)?).into_owned();
    Some((
        record_end,
        ZipPart {
            name,
            method,
            encrypted: flags & 1 == 1,
            compressed_size,
            uncompressed_size,
            header_start: header_start.checked_add(delta)?,
        },
    ))
}

/// The extra-field ids the reader's own parse knows by number: for these,
/// a chunk whose header or content cannot be read is a parse ERROR (the
/// candidate is rejected), where an unknown id's same bytes are padding
/// the reader skips past.
const KNOWN_EXTRA_IDS: [u16; 7] = [0x0001, 0x000A, 0x5455, 0x6375, 0x7075, 0x9901, 0xA11E];

/// The walk over one record's extra field, mirroring the reader's chunk
/// loop field for field: every chunk's declared length must fit inside
/// the extra field (the reader reads the chunk's content off its cursor
/// and an overrun is a truncation error, not padding); a trailing run of
/// one byte is padding (the reader's field-id read consumes it and stops
/// the loop), while two or three trailing bytes that name a KNOWN id are
/// a header the reader cannot finish (an error), and any other trailing
/// bytes are padding. The zip64 extended-information field (id 0x0001)
/// fills any fixed size field capped at 0xFFFFFFFF, read in the order the
/// reader's own parse fills them (uncompressed, compressed, local
/// offset; a field of 24+ bytes carries all three), and the walk
/// continues past it; the reader parses every remaining chunk too, and
/// any of them can still reject the record. `Malformed` is the reader's
/// parse error; `Absent` leaves the fixed values standing, exactly the
/// reader's behavior when no zip64 field is present; later zip64 fields
/// override earlier ones, as they do in the reader's parse.
enum Zip64Extra {
    Absent,
    Malformed,
    Parsed {
        size: u64,
        compressed: u64,
        header: u64,
    },
}

fn zip64_extra(extra: &[u8], uncompressed: u64, compressed: u64, header_start: u64) -> Zip64Extra {
    let mut probe = 0;
    let mut parsed: Option<(u64, u64, u64)> = None;
    while probe + 4 <= extra.len() {
        let Some(id) = le16(extra, probe) else {
            return Zip64Extra::Malformed;
        };
        let Some(len) = le16(extra, probe + 2) else {
            return Zip64Extra::Malformed;
        };
        let Some(end) = probe
            .checked_add(4)
            .and_then(|at| at.checked_add(len as usize))
        else {
            return Zip64Extra::Malformed;
        };
        if end > extra.len() {
            // the reader reads the chunk's content off its extra-field
            // cursor: an overrun is a truncation error, never padding
            return Zip64Extra::Malformed;
        }
        if id == 0x0001 {
            let body = &extra[probe + 4..end];
            let full = body.len() >= 24;
            let mut at = 0;
            // One field read under the reader's presence rule: `Some(None)`
            // = the rule says stand pat on the fixed value, `None` = the
            // rule says read and the chunk is too short for it (the
            // reader's "ZIP64 extra field truncated" / "wrong length").
            let mut read = |present: bool| -> Option<Option<u64>> {
                if !full && !present {
                    return Some(None);
                }
                let value = le64(body, at)?;
                at += 8;
                Some(Some(value))
            };
            let size = match read(uncompressed == u64::from(u32::MAX)) {
                Some(Some(value)) => value,
                Some(None) => uncompressed,
                None => return Zip64Extra::Malformed,
            };
            let compressed = match read(compressed == u64::from(u32::MAX)) {
                Some(Some(value)) => value,
                Some(None) => compressed,
                None => return Zip64Extra::Malformed,
            };
            let header = match read(header_start == u64::from(u32::MAX)) {
                Some(Some(value)) => value,
                Some(None) => header_start,
                None => return Zip64Extra::Malformed,
            };
            parsed = Some((size, compressed, header));
        }
        probe = end;
    }
    // Two or three trailing bytes: a known id makes the header the reader
    // cannot finish (an error); anything else is padding.
    if extra.len() - probe >= 2
        && let Some(id) = le16(extra, probe)
        && KNOWN_EXTRA_IDS.contains(&id)
    {
        return Zip64Extra::Malformed;
    }
    match parsed {
        Some((size, compressed, header)) => Zip64Extra::Parsed {
            size,
            compressed,
            header,
        },
        None => Zip64Extra::Absent,
    }
}

/// The bytes one part inflates to, counted (never materialized: the only
/// buffer is a 64 KiB scratch) under a hard per-part
/// `take(ceiling + 1)`. The span is the reader's own: data located
/// through the LOCAL header (its name/extra lengths are the ones that
/// matter and can differ from the central record's), sized by the
/// central directory's compressed size, which is what the reader's
/// `find_content` takes its compressed stream to. An entry the audit
/// cannot bound (encrypted, an unsupported compression method: both are
/// reader errors today) is refused here, before it can reach the engine.
fn part_inflated_bytes(bytes: &[u8], part: &ZipPart) -> Result<u64, DocumentError> {
    if part.encrypted {
        return Err(Convert(format!(
            "office_oxide: part '{}' is encrypted: this lane audits the bytes it inflates and \
             cannot bound encrypted ones",
            part.name,
        )));
    }
    let header = usize::try_from(part.header_start).map_err(|_| {
        Convert(format!(
            "office_oxide: part '{}' names a local header this audit cannot address",
            part.name,
        ))
    })?;
    let data_start = (|| {
        let fixed_end = header.checked_add(30)?;
        if !bytes
            .get(header..fixed_end)?
            .starts_with(&ZIP_LOCAL_HEADER_SIG)
        {
            return None;
        }
        let name_len = le16(bytes, header + 26)? as usize;
        let extra_len = le16(bytes, header + 28)? as usize;
        fixed_end.checked_add(name_len)?.checked_add(extra_len)
    })()
    .ok_or_else(|| {
        Convert(format!(
            "office_oxide: part '{}' has an unreadable local header: the audit cannot locate \
             the bytes the reader will inflate",
            part.name,
        ))
    })?;
    let compressed = usize::try_from(part.compressed_size).unwrap_or(usize::MAX);
    let span_end = data_start.saturating_add(compressed).min(bytes.len());
    let Some(span) = (data_start <= span_end).then(|| &bytes[data_start..span_end]) else {
        return Ok(0);
    };
    match part.method {
        // Stored: the inflated bytes are the span.
        0 => Ok(span.len() as u64),
        // Deflated: count the inflate under the per-part ceiling.
        8 => {
            use std::io::Read as _;
            let decoder = flate2::read::DeflateDecoder::new(span);
            let mut capped = decoder.take(OXIDE_MAX_TOTAL_DECOMPRESSED + 1);
            let mut scratch = [0u8; 64 * 1024];
            let mut count: u64 = 0;
            loop {
                match capped.read(&mut scratch) {
                    Ok(0) => break,
                    Ok(n) => count += n as u64,
                    // a stream that dies mid-part: the reader's own inflate
                    // dies the same way on the same bytes, so the count so
                    // far is what it could have produced
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
            Ok(count)
        }
        _ => Err(Convert(format!(
            "office_oxide: part '{}' names compression method {}: this lane audits stored and \
             deflated parts only, so the container is refused before it can inflate",
            part.name, part.method,
        ))),
    }
}

/// The `backend="oxide"` office conversion: office_oxide's unified reader.
/// Its `from_reader` takes the format explicitly, and anydoc's Excel kind
/// covers both the xlsx family and legacy xls, so the shared container
/// check ([`is_ole_container`]: the same one the reported name and sniff
/// use) decides that one: the OLE signature is legacy, anything else (a
/// ZIP local-file header) is OOXML. Takes the bytes owned: the container
/// check borrows them, then they move into the `Cursor` whole: zero
/// copies (convert() already owns the Vec; `from_reader`'s `Read + Seek +
/// 'static` bound demands ownership of the reader, not a second copy of
/// the bytes). Before any of that, the decompression audit
/// ([`audit_oxide_container`]): office_oxide bounds each part at 512 MiB
/// and nothing else, so the total across parts is refused here.
fn office_markdown(bytes: Vec<u8>, kind: Kind) -> Result<String, DocumentError> {
    audit_oxide_container(&bytes)?;
    let format = match kind {
        Kind::Docx => office_oxide::DocumentFormat::Docx,
        Kind::Excel => {
            if is_ole_container(&bytes) {
                office_oxide::DocumentFormat::Xls
            } else {
                office_oxide::DocumentFormat::Xlsx
            }
        }
        Kind::Pptx => office_oxide::DocumentFormat::Pptx,
        Kind::Doc => office_oxide::DocumentFormat::Doc,
        Kind::Ppt => office_oxide::DocumentFormat::Ppt,
        // unreachable by construction: engine_for only routes the OOXML and
        // legacy office kinds here.
        _ => {
            return Err(DocumentError::Convert(
                "office_oxide: unreachable kind".into(),
            ));
        }
    };
    office_oxide::Document::from_reader(Cursor::new(bytes), format)
        .map_err(|e| Convert(format!("office_oxide: {e}")))
        .map(|doc| doc.to_markdown())
}

/// The multi-format `page_geometry` probe: one entry per unit of the
/// resolved format's geometry lane (the §6 matrix), `None` where a unit
/// declares no geometry. The spine mirrors [`convert`]'s shape — resolve,
/// route through the geometry table, refuse, run the one engine leg —
/// with the surface's own refusal order (§6, pinned): the format resolves
/// first (the capability refusal names it), then the capability refusal,
/// then the password-kind refusal, then `max_bytes` (zero refused HERE —
/// the binding layer no longer pre-empts it on this surface, so the
/// value refusal lands after the capability and password-kind checks
/// where §6 sequences it; an explicit budget is the binding layer's
/// pre-read gate and the core's post-read ceiling on the office leg;
/// `None` keeps the 32 MiB default there), then the engine legs' own
/// encrypted doors.
///
/// Per-entry `None` is the single carve-out to no-partial-success: a unit
/// whose own box declaration cannot be read (or whose declared extent is
/// non-positive/non-finite) is a `None` entry, siblings intact — see
/// `pdf_impl::page_geometry` for the PDF rule and [`office_geometry`]'s
/// per-section alignment for the office one. The no-geometry row is the
/// backend-independent `[None]`: no engine runs, no refusal fires, and a
/// `password=` is still refused (a silently-ignored password would claim
/// a check this surface never ran).
pub fn page_geometry(
    bytes: Vec<u8>,
    name_hint: Option<&str>,
    format: Option<&str>,
    backend: Backend,
    options: &ConvertOptions,
) -> Result<Vec<Option<PageGeometry>>, DocumentError> {
    let kind = resolve(&bytes, name_hint, format)?;
    let lane = geometry_engine(kind, backend, &bytes)?;
    if options.password.is_some() && kind != Kind::Pdf {
        return Err(DocumentError::Convert(format!(
            "password= applies to PDF documents only, not {:?}",
            kind.name()
        )));
    }
    if options.max_bytes.is_some_and(|limit| limit == 0) {
        return Err(DocumentError::Convert("max_bytes must be positive".into()));
    }
    match lane {
        // The no-geometry row: unit = whole document (single-flow), so
        // exactly one entry, `None` (never `[]`), for every backend.
        None => Ok(vec![None]),
        Some(Engine::PdfOxide) => {
            let entries = pdf_impl::page_geometry(bytes, options.password).map_err(pdf_error)?;
            Ok(entries
                .into_iter()
                .map(|entry| entry.map(PageGeometry::from_page_box))
                .collect())
        }
        Some(Engine::OfficeOxide) => {
            // The office leg parses the container into resident memory (the
            // same amplification class the conversion lane meters): the
            // same input ceiling, explicit budget over the 32 MiB default.
            let limit = options.max_bytes.unwrap_or(DEFAULT_ANYDOC_INPUT_LIMIT);
            if bytes.len() > limit {
                return Err(DocumentError::Convert(format!(
                    "the document is {} and the {} engine lane's input ceiling is {} \
                     (that lane amplifies input into resident memory at a measured multiple \
                     — 2026-09-09: ~146x worst case on anydoc's delimiter formats, \
                     ~4100x on a zip-bombed office_oxide container): split the file, or pass a larger \
                     max_bytes",
                    render_size(bytes.len()),
                    engine_name(Engine::OfficeOxide),
                    render_size(limit),
                )));
            }
            office_geometry(bytes, kind)
        }
        // geometry_engine's table is total: the Anydoc and Html2Md lanes
        // never come back from it.
        Some(_) => Err(DocumentError::Convert(
            "geometry: unreachable engine lane".into(),
        )),
    }
}

/// The office leg of [`page_geometry`]: the OOXML families read their RAW
/// office_oxide layers for the declared sizes (the honest `Option`s —
/// `to_ir()`'s page-setup fabrication is the hole this surface exists to
/// avoid: a margins-only DOCX sectPr or an attribute-less `<w:pgSz>`
/// fabricates Letter through the IR, and XLSX's converter substitutes
/// Letter for sheets that declare only margins), with the counts from the
/// layer the §6 matrix names. The legacy trio (and the Excel kind's OLE
/// container, which IS a legacy workbook) reads the IR for the section
/// count alone. The container audit runs first, exactly as on the
/// conversion lane, and the parse's typed errors (an encrypted-OOXML CFB
/// container named as such, a missing part) are the whole call's
/// `ValueError`s, never `None`/`[]`.
fn office_geometry(bytes: Vec<u8>, kind: Kind) -> Result<Vec<Option<PageGeometry>>, DocumentError> {
    audit_oxide_container(&bytes)?;
    // The Excel kind's container decision, taken while the bytes are
    // still owned (the parse below moves them): the OLE signature is
    // legacy xls, a ZIP package is xlsx — the same check the reported
    // name and the conversion lane's engine choice make.
    let legacy_xls = kind == Kind::Excel && is_ole_container(&bytes);
    let format = match kind {
        // The Excel kind's one container-aware case (office_markdown's own
        // check): the OLE signature is legacy xls, a ZIP package is xlsx —
        // the hoisted `legacy_xls`, not a second container read.
        Kind::Excel if legacy_xls => office_oxide::DocumentFormat::Xls,
        Kind::Excel => office_oxide::DocumentFormat::Xlsx,
        Kind::Docx => office_oxide::DocumentFormat::Docx,
        Kind::Pptx => office_oxide::DocumentFormat::Pptx,
        Kind::Doc => office_oxide::DocumentFormat::Doc,
        Kind::Ppt => office_oxide::DocumentFormat::Ppt,
        // unreachable by construction: geometry_engine routes only the
        // office families here.
        _ => {
            return Err(Convert("office_oxide: unreachable kind".into()));
        }
    };
    let doc = office_oxide::Document::from_reader(Cursor::new(bytes), format)
        .map_err(|e| Convert(format!("office_oxide: {e}")))?;
    match kind {
        // DOCX: the declared bit AND the size from the RAW sections
        // (twips ÷ 20.0, positive dims required); the section count and
        // order from `to_ir()` (the converter-internal body-window logic
        // is not replicated here), aligned by the converter's own
        // `doc.sections.get(idx)` rule: an IR section whose index reaches
        // past the raw sections is a None entry, mirroring the converter.
        Kind::Docx => {
            let count = doc.to_ir().sections.len();
            let raw = doc
                .as_docx()
                .ok_or_else(|| Convert("office_oxide: the docx parse lost its raw layer".into()))?;
            Ok((0..count)
                .map(|idx| docx_section_geometry(raw, idx))
                .collect())
        }
        // PPTX: the raw layer's deck-wide slide size, EMU ÷ 12700.0 in
        // float (NOT the IR's ÷635→twips-integer path, which truncates
        // before this surface would see it), one entry per raw slide.
        Kind::Pptx => {
            let raw = doc
                .as_pptx()
                .ok_or_else(|| Convert("office_oxide: the pptx parse lost its raw layer".into()))?;
            let count = raw.slides.len();
            let base = raw.presentation.slide_size.as_ref().and_then(|size| {
                let (width_pt, height_pt) = (size.cx as f32 / 12700.0, size.cy as f32 / 12700.0);
                let finite_positive = |value: f32| value.is_finite() && value > 0.0;
                (finite_positive(width_pt) && finite_positive(height_pt)).then_some(PageGeometry {
                    width_pt,
                    height_pt,
                    rotation: 0,
                    crop: None,
                })
            });
            Ok(vec![base; count])
        }
        // The legacy trio — the Excel kind's OLE container included: a
        // genuine .xls parsed through the legacy converter declares zero
        // geometry anywhere, and its IR carries one Section per worksheet
        // (convert_xls's sheets loop; convert_doc one for the document,
        // convert_ppt one per slide), so the answer is all-None at the IR
        // section count. The raw xlsx layer this branch must NOT reach
        // for is None by construction here (a legacy parse has no xlsx
        // raw layer to lose) — the arm the all-None answer protects.
        Kind::Doc | Kind::Ppt | Kind::Excel if legacy_xls => {
            let count = doc.to_ir().sections.len();
            Ok(vec![None; count])
        }
        // XLSX (the ZIP container): all-None — a worksheet is an
        // unbounded grid, not a page, and the IR's default substitution
        // makes Some untrustworthy anyway. The count is the RAW worksheet
        // layer's (all sheets, hidden included), not the IR's
        // worksheet+Charts count.
        Kind::Excel => {
            let raw = doc
                .as_xlsx()
                .ok_or_else(|| Convert("office_oxide: the xlsx parse lost its raw layer".into()))?;
            Ok(vec![None; raw.worksheets.len()])
        }
        // unreachable by construction: every office kind is covered above
        // (the guarded legacy arm took the OLE-container Excel case).
        _ => {
            let count = doc.to_ir().sections.len();
            Ok(vec![None; count])
        }
    }
}

/// One DOCX section's geometry: the RAW `SectionProperties` at `idx`, its
/// `page_size` if declared with positive dims (twips ÷ 20.0). `None` when
/// the index is beyond the raw sections, the section declares only
/// margins (the exact shape `to_ir()` fabricates Letter for — the hole
/// this raw-layer read exists to close), or the declared dims are
/// non-positive. Office entries always report rotation 0 and no crop.
fn docx_section_geometry(
    raw: &office_oxide::docx::DocxDocument,
    idx: usize,
) -> Option<PageGeometry> {
    let size = &raw.sections.get(idx)?.page_size.as_ref()?;
    let (width, height) = (size.width.0, size.height.0);
    (width > 0 && height > 0).then_some(PageGeometry {
        width_pt: width as f32 / 20.0,
        height_pt: height as f32 / 20.0,
        rotation: 0,
        crop: None,
    })
}

/// The multi-format `page_layout` probe: per unit of the resolved format's
/// geometry lane, the unit's geometry PLUS — PDF only — the page's text
/// spans, lines, and image boxes in the page's declared render space
/// (`pdf_impl::page_layout`, spec §5: the D1 space, one transform, one
/// open). The spine is [`page_geometry`]'s exactly (resolve → capability →
/// password-kind → max_bytes → door), with the `pages=` selection the two
/// conversion probes take: delivered to the PDF leg, which normalizes it
/// with `markdown_pages`' exact rule and error strings, and returns the
/// output parallel to the NORMALIZED selection, every entry carrying its
/// `page_index`. `pages=` is PDF-ONLY (a page index cannot name a
/// section/slide/worksheet): a non-`None` selection on any non-PDF row is
/// convert()'s own typed refusal, never a silent whole-document answer.
///
/// Layout content is PDF-only ("the capability table applies to
/// page_layout identically; PDF-only for layout"): no office engine
/// exposes text/image boxes, so the office rows serve their geometry with
/// EMPTY spans/lines/images — the honest empty (the empty-lists-not-errors
/// law's shape: no engine surface, never a fabricated one), the no-geometry
/// row its single `geometry: None` entry. Every entry carries the resolved
/// format's vocabulary name (`resolved_name` spelling) as its `format`
/// field: the format — not the entry's shape — tells a consumer holding
/// only an entry why the layout lists are empty (an office row: no text
/// surfaces; a blank, scanned, or content-unparseable PDF page: the same
/// shape on the pdf lane). PDF failure semantics: per
/// `pdf_impl::page_layout` (§7): a page whose box read fails is
/// `geometry: None` + empty lists, siblings intact; an unparseable content
/// stream is empty lists, geometry intact; encrypted doors fail closed.
/// Image boxes are never visibility-filtered (the engine's 2 pt drop law
/// is the TEXT walk's): a wholly-outside-the-crop image is returned with
/// out-of-range coordinates — the consumer clips to the page box.
/// A non-zero `rotation_degrees` on a span means its box is the engine's
/// flattened approximation and MUST NOT be used as a highlight box —
/// and the flag does NOT cover vertical writing (the verified T9 edge,
/// probed: a WMode-1 tategaki run carries `rotation_degrees == 0` with a
/// negative-width flattened box; see `pdf_impl::page_layout` for the
/// measured evidence).
///
/// Refusal order (the pinned §6 precedence, unchanged by the `pages=`
/// refusal): a malformed `pages=` selection or a zero `max_bytes` is
/// reported AFTER the capability refusal, never before it — the sibling
/// surfaces' behavior note, restated here because both arguments parse at
/// this seam.
pub fn page_layout(
    bytes: Vec<u8>,
    name_hint: Option<&str>,
    format: Option<&str>,
    backend: Backend,
    pages: Option<Vec<usize>>,
    options: &ConvertOptions,
) -> Result<Vec<pdf_impl::PageLayoutData>, DocumentError> {
    let kind = resolve(&bytes, name_hint, format)?;
    let lane = geometry_engine(kind, backend, &bytes)?;
    // The entry's provenance marker: the resolved format's vocabulary
    // name (`resolved_name` spelling — the same string to_markdown
    // reports, the Excel kind's container-aware xls/xlsx case
    // included), resolved ONCE here and stamped on every entry of every
    // lane below (pdf_impl threads it through its constructors). The
    // format — not the entry's shape — tells a consumer holding only an
    // entry why its spans/lines/images are empty (an office row carries
    // declared geometry with no text surfaces; a blank or scanned PDF
    // page looks identical).
    let format_name = resolved_name(kind, &bytes);
    // `pages=` is a PDF-lane parameter: the selection indexes PAGES, and a
    // section/slide/worksheet is not a page — a non-None selection on any
    // non-PDF row is convert()'s own typed refusal (byte-identical wording,
    // the same `Pages` error), never a silent whole-document answer. The
    // capability refusal above still wins first (convert()'s ordering: the
    // engine route precedes the pages check). On the PDF row the selection
    // normalizes in `pdf_impl::page_layout` (the empty-selection and
    // out-of-range refusals, markdown_pages' byte-pinned strings).
    if pages.is_some() && kind != Kind::Pdf {
        return Err(DocumentError::Pages(format!(
            "pages= applies to PDF documents only, not {:?}",
            kind.name()
        )));
    }
    if options.password.is_some() && kind != Kind::Pdf {
        return Err(DocumentError::Convert(format!(
            "password= applies to PDF documents only, not {:?}",
            kind.name()
        )));
    }
    if options.max_bytes.is_some_and(|limit| limit == 0) {
        return Err(DocumentError::Convert("max_bytes must be positive".into()));
    }
    match lane {
        // The no-geometry row: unit = whole document (single-flow), so
        // exactly one entry with no geometry and no layout content, for
        // every backend.
        None => Ok(vec![pdf_impl::PageLayoutData {
            page_index: 0,
            format: format_name.to_string(),
            geometry: None,
            spans: Vec::new(),
            lines: Vec::new(),
            images: Vec::new(),
        }]),
        Some(Engine::PdfOxide) => {
            pdf_impl::page_layout(bytes, pages.as_deref(), options.password, format_name)
                .map_err(pages_error)
        }
        Some(Engine::OfficeOxide) => {
            // The office leg parses the container into resident memory (the
            // same amplification class the conversion lane meters): the
            // same input ceiling, explicit budget over the 32 MiB default.
            let limit = options.max_bytes.unwrap_or(DEFAULT_ANYDOC_INPUT_LIMIT);
            if bytes.len() > limit {
                return Err(DocumentError::Convert(format!(
                    "the document is {} and the {} engine lane's input ceiling is {} \
                     (that lane amplifies input into resident memory at a measured multiple \
                     — 2026-09-09: ~146x worst case on anydoc's delimiter formats, \
                     ~4100x on a zip-bombed office_oxide container): split the file, or pass a larger \
                     max_bytes",
                    render_size(bytes.len()),
                    engine_name(Engine::OfficeOxide),
                    render_size(limit),
                )));
            }
            Ok(office_geometry(bytes, kind)?
                .into_iter()
                .enumerate()
                .map(|(page_index, geometry)| pdf_impl::PageLayoutData {
                    page_index,
                    format: format_name.to_string(),
                    geometry: geometry.map(|g| pdf_impl::PageBox {
                        width_pt: g.width_pt,
                        height_pt: g.height_pt,
                        rotation: g.rotation,
                        crop: g.crop,
                    }),
                    spans: Vec::new(),
                    lines: Vec::new(),
                    images: Vec::new(),
                })
                .collect())
        }
        // geometry_engine's table is total: the Anydoc and Html2Md lanes
        // never come back from it.
        Some(_) => Err(DocumentError::Convert(
            "layout: unreachable engine lane".into(),
        )),
    }
}

/// The multi-format `page_tables` probe: per unit of the resolved format's
/// geometry lane, the unit's geometry PLUS — PDF only — the page's
/// detected tables with per-cell boxes in the page's declared render
/// space (`pdf_impl::page_tables`, issue #204: the frame-coherence gate
/// A1, the primitive edge cap A6, the refused marker A9). The spine is
/// [`page_layout`]'s exactly (A8 — resolve → capability (`geometry_engine`)
/// → `format_name` → `pages=` non-PDF refusal → password non-PDF refusal →
/// `max_bytes == 0` → engine leg), with the same `pages=` selection
/// semantics (PDF-only, normalized with `markdown_pages`' exact rule and
/// error strings, output parallel to the NORMALIZED selection,
/// `page_index` on every entry).
///
/// Tables are PDF-only in v1: the office engines' IR tables carry grid and
/// cell text but NO coordinates (office_oxide ir.rs `Element::Table`), and
/// "grid without boxes" is ruled out — a row that has boxes sometimes is
/// the worst answer — so the office rows serve their declared geometry
/// with `tables: []` and `tables_refused: false` (A9: `[]` there is the
/// per-format capability statement, NOT a per-page content judgment; the
/// entry's `format` field is the marker that says so). DOCX table→section
/// attribution needs converter-internal body-window logic the seam does
/// not replicate (see the `office_geometry` note); the grid-without-boxes
/// spelling is the v2 candidate (clean for PPTX/XLSX via `to_ir()`).
///
/// The no-geometry row: exactly one entry with `geometry: None` and no
/// tables, for every backend. Every entry carries the resolved format's
/// vocabulary name (`resolved_name` spelling) as its `format` field.
/// PDF failure semantics: per `pdf_impl::page_tables` — a page whose box
/// read fails is `geometry: None` + no tables, siblings intact; a
/// content-stream failure is no tables, geometry intact, and (unlike the
/// frame refusals) `tables_refused: false` — unknowable, documented;
/// encrypted doors fail closed. `max_bytes` is inert on the PDF lane
/// except the zero-value refusal (pdf_oxide's own limits govern there —
/// the PDF lane is unmetered); the office leg's input ceiling applies
/// even though `tables` is always `[]` there (the container still parses
/// for geometry), and it must raise identically to `page_layout`'s.
pub fn page_tables(
    bytes: Vec<u8>,
    name_hint: Option<&str>,
    format: Option<&str>,
    backend: Backend,
    pages: Option<Vec<usize>>,
    options: &ConvertOptions,
) -> Result<Vec<pdf_impl::PageTablesData>, DocumentError> {
    let kind = resolve(&bytes, name_hint, format)?;
    let lane = geometry_engine(kind, backend, &bytes)?;
    // The entry's provenance marker: the resolved format's vocabulary
    // name, resolved ONCE here and stamped on every entry (page_layout's
    // exact rule — the format, not the entry's shape, tells a consumer
    // holding only an entry why `tables` is empty: an office row's
    // per-format capability statement, a PDF page's no-tables/failure
    // shape, a refusal's skip).
    let format_name = resolved_name(kind, &bytes);
    // `pages=` is a PDF-lane parameter (page_layout's own refusal,
    // byte-identical wording): a non-None selection on any non-PDF row is
    // convert()'s typed refusal. The capability refusal above still wins
    // first; the password-kind refusal and the zero refusal follow in the
    // same order (A8).
    if pages.is_some() && kind != Kind::Pdf {
        return Err(DocumentError::Pages(format!(
            "pages= applies to PDF documents only, not {:?}",
            kind.name()
        )));
    }
    if options.password.is_some() && kind != Kind::Pdf {
        return Err(DocumentError::Convert(format!(
            "password= applies to PDF documents only, not {:?}",
            kind.name()
        )));
    }
    if options.max_bytes.is_some_and(|limit| limit == 0) {
        return Err(DocumentError::Convert("max_bytes must be positive".into()));
    }
    match lane {
        // The no-geometry row: unit = whole document (single-flow), so
        // exactly one entry with no geometry and no tables, for every
        // backend.
        None => Ok(vec![pdf_impl::PageTablesData {
            page_index: 0,
            format: format_name.to_string(),
            geometry: None,
            tables: Vec::new(),
            tables_refused: false,
        }]),
        Some(Engine::PdfOxide) => {
            pdf_impl::page_tables(bytes, pages.as_deref(), options.password, format_name)
                .map_err(pages_error)
        }
        Some(Engine::OfficeOxide) => {
            // The office leg parses the container into resident memory (the
            // same amplification class the conversion lane meters): the
            // same input ceiling, explicit budget over the 32 MiB default.
            let limit = options.max_bytes.unwrap_or(DEFAULT_ANYDOC_INPUT_LIMIT);
            if bytes.len() > limit {
                return Err(DocumentError::Convert(format!(
                    "the document is {} and the {} engine lane's input ceiling is {} \
                     (that lane amplifies input into resident memory at a measured multiple \
                     — 2026-09-09: ~146x worst case on anydoc's delimiter formats, \
                     ~4100x on a zip-bombed office_oxide container): split the file, or pass a larger \
                     max_bytes",
                    render_size(bytes.len()),
                    engine_name(Engine::OfficeOxide),
                    render_size(limit),
                )));
            }
            Ok(office_geometry(bytes, kind)?
                .into_iter()
                .enumerate()
                .map(|(page_index, geometry)| pdf_impl::PageTablesData {
                    page_index,
                    format: format_name.to_string(),
                    geometry: geometry.map(|g| pdf_impl::PageBox {
                        width_pt: g.width_pt,
                        height_pt: g.height_pt,
                        rotation: g.rotation,
                        crop: g.crop,
                    }),
                    tables: Vec::new(),
                    tables_refused: false,
                })
                .collect())
        }
        // geometry_engine's table is total: the Anydoc and Html2Md lanes
        // never come back from it.
        Some(_) => Err(DocumentError::Convert(
            "tables: unreachable engine lane".into(),
        )),
    }
}

/// The multi-format `page_images` probe (issue #205): per unit of the
/// resolved format's geometry lane, the unit's geometry PLUS — PDF pages
/// only — the page's image handles with the bytes behind them under the
/// opt-in `decode` (`pdf_impl::page_images`: the F1 byte-source ruling,
/// the R7 caps, the R8 honesty laws). The spine is [`page_tables`]'s
/// exactly (R17/R21 — this body is its verbatim copy): resolve →
/// capability (`geometry_engine`) → `format_name` → `pages=` non-PDF
/// refusal → password non-PDF refusal → `max_bytes == 0` → engine leg,
/// with the same `pages=` selection semantics (PDF-only, normalized with
/// `markdown_pages`' exact rule and error strings, output parallel to
/// the NORMALIZED selection, `page_index` on every entry).
///
/// Images are PDF-only in v1: an office row carries its declared
/// geometry with `images: []` (R24: the per-format capability statement,
/// NOT a per-page content judgment — a DOCX's images are real and are
/// not served; the entry's `format` field is the marker), with no
/// refused-marker (the engine exposes no drop counts). The no-geometry
/// row: exactly one entry with `geometry: None` and no images. PDF
/// failure semantics: per `pdf_impl::page_images` — the frame and
/// content-failure laws, the typed decode entries, the caps.
/// `decode=` is the core's [`pdf_impl::ImageDecode`] (None →
/// Container → Pixels, R3); `max_image_bytes` the per-image cap
/// override (`None` = the 256 MiB default; the 512 MiB per-call budget
/// is not overridable). `max_bytes` is inert on the PDF lane except the
/// zero-value refusal, and the office leg's input ceiling applies even
/// though `images` is always `[]` there (the container still parses for
/// geometry), raising identically to `page_layout`'s.
#[allow(clippy::too_many_arguments)]
pub fn page_images(
    bytes: Vec<u8>,
    name_hint: Option<&str>,
    format: Option<&str>,
    backend: Backend,
    pages: Option<Vec<usize>>,
    options: &ConvertOptions,
    decode: pdf_impl::ImageDecode,
    max_image_bytes: Option<u64>,
) -> Result<Vec<pdf_impl::PageImagesData>, DocumentError> {
    let kind = resolve(&bytes, name_hint, format)?;
    let lane = geometry_engine(kind, backend, &bytes)?;
    // The entry's provenance marker: the resolved format's vocabulary
    // name, resolved ONCE here and stamped on every entry (page_tables'
    // exact rule).
    let format_name = resolved_name(kind, &bytes);
    if pages.is_some() && kind != Kind::Pdf {
        return Err(DocumentError::Pages(format!(
            "pages= applies to PDF documents only, not {:?}",
            kind.name()
        )));
    }
    if options.password.is_some() && kind != Kind::Pdf {
        return Err(DocumentError::Convert(format!(
            "password= applies to PDF documents only, not {:?}",
            kind.name()
        )));
    }
    if options.max_bytes.is_some_and(|limit| limit == 0) {
        return Err(DocumentError::Convert("max_bytes must be positive".into()));
    }
    match lane {
        // The no-geometry row: unit = whole document (single-flow), so
        // exactly one entry with no geometry and no images, for every
        // backend.
        None => Ok(vec![pdf_impl::PageImagesData {
            page_index: 0,
            format: format_name.to_string(),
            geometry: None,
            images: Vec::new(),
        }]),
        Some(Engine::PdfOxide) => pdf_impl::page_images(
            bytes,
            pages.as_deref(),
            options.password,
            format_name,
            decode,
            max_image_bytes,
        )
        .map_err(pages_error),
        Some(Engine::OfficeOxide) => {
            // The office leg parses the container into resident memory (the
            // same amplification class the conversion lane meters): the
            // same input ceiling, explicit budget over the 32 MiB default.
            let limit = options.max_bytes.unwrap_or(DEFAULT_ANYDOC_INPUT_LIMIT);
            if bytes.len() > limit {
                return Err(DocumentError::Convert(format!(
                    "the document is {} and the {} engine lane's input ceiling is {} \
                     (that lane amplifies input into resident memory at a measured multiple \
                     — 2026-09-09: ~146x worst case on anydoc's delimiter formats, \
                     ~4100x on a zip-bombed office_oxide container): split the file, or pass a larger \
                     max_bytes",
                    render_size(bytes.len()),
                    engine_name(Engine::OfficeOxide),
                    render_size(limit),
                )));
            }
            Ok(office_geometry(bytes, kind)?
                .into_iter()
                .enumerate()
                .map(|(page_index, geometry)| pdf_impl::PageImagesData {
                    page_index,
                    format: format_name.to_string(),
                    geometry: geometry.map(|g| pdf_impl::PageBox {
                        width_pt: g.width_pt,
                        height_pt: g.height_pt,
                        rotation: g.rotation,
                        crop: g.crop,
                    }),
                    images: Vec::new(),
                })
                .collect())
        }
        // geometry_engine's table is total: the Anydoc and Html2Md lanes
        // never come back from it.
        Some(_) => Err(DocumentError::Convert(
            "images: unreachable engine lane".into(),
        )),
    }
}

/// The multi-format `page_text_chars` probe (issue #206): per unit of the
/// resolved format's geometry lane, the unit's geometry PLUS — PDF pages
/// only — the page's per-character boxes from the cached span parse
/// (`pdf_impl::page_text_chars`: the R4 mirror fix, the R9 mirrored
/// predicate, the join law). The spine is [`page_tables`]'s exactly —
/// this body is its verbatim copy, `page_images`' too: resolve →
/// capability → `format_name` → `pages=` non-PDF refusal → password
/// non-PDF refusal → `max_bytes == 0` → engine leg, the same `pages=`
/// selection semantics. Chars are PDF-only in v1: an office row carries
/// its declared geometry with `chars: []` (R24: the per-format
/// capability statement; the `format` field is the marker), no
/// refused-marker. The no-geometry row: exactly one entry with
/// `geometry: None` and no chars. There is no decode knob (R13: the cost
/// is ≈ page_layout's span leg) and no image-cap equivalent ([R7's #206
/// cell]: the ceiling beyond the input budget is none).
pub fn page_text_chars(
    bytes: Vec<u8>,
    name_hint: Option<&str>,
    format: Option<&str>,
    backend: Backend,
    pages: Option<Vec<usize>>,
    options: &ConvertOptions,
) -> Result<Vec<pdf_impl::PageCharsData>, DocumentError> {
    let kind = resolve(&bytes, name_hint, format)?;
    let lane = geometry_engine(kind, backend, &bytes)?;
    let format_name = resolved_name(kind, &bytes);
    if pages.is_some() && kind != Kind::Pdf {
        return Err(DocumentError::Pages(format!(
            "pages= applies to PDF documents only, not {:?}",
            kind.name()
        )));
    }
    if options.password.is_some() && kind != Kind::Pdf {
        return Err(DocumentError::Convert(format!(
            "password= applies to PDF documents only, not {:?}",
            kind.name()
        )));
    }
    if options.max_bytes.is_some_and(|limit| limit == 0) {
        return Err(DocumentError::Convert("max_bytes must be positive".into()));
    }
    match lane {
        None => Ok(vec![pdf_impl::PageCharsData {
            page_index: 0,
            format: format_name.to_string(),
            geometry: None,
            chars: Vec::new(),
        }]),
        Some(Engine::PdfOxide) => {
            pdf_impl::page_text_chars(bytes, pages.as_deref(), options.password, format_name)
                .map_err(pages_error)
        }
        Some(Engine::OfficeOxide) => {
            let limit = options.max_bytes.unwrap_or(DEFAULT_ANYDOC_INPUT_LIMIT);
            if bytes.len() > limit {
                return Err(DocumentError::Convert(format!(
                    "the document is {} and the {} engine lane's input ceiling is {} \
                     (that lane amplifies input into resident memory at a measured multiple \
                     — 2026-09-09: ~146x worst case on anydoc's delimiter formats, \
                     ~4100x on a zip-bombed office_oxide container): split the file, or pass a larger \
                     max_bytes",
                    render_size(bytes.len()),
                    engine_name(Engine::OfficeOxide),
                    render_size(limit),
                )));
            }
            Ok(office_geometry(bytes, kind)?
                .into_iter()
                .enumerate()
                .map(|(page_index, geometry)| pdf_impl::PageCharsData {
                    page_index,
                    format: format_name.to_string(),
                    geometry: geometry.map(|g| pdf_impl::PageBox {
                        width_pt: g.width_pt,
                        height_pt: g.height_pt,
                        rotation: g.rotation,
                        crop: g.crop,
                    }),
                    chars: Vec::new(),
                })
                .collect())
        }
        Some(_) => Err(DocumentError::Convert(
            "text-chars: unreachable engine lane".into(),
        )),
    }
}

/// The most entries one structure payload materializes: 200,000, applied
/// in document order to ALL THREE entry lists (xlsx cells, docx blocks,
/// html nodes — the same O(N)-Python-objects class). The honest statement
/// of what this bounds: PAYLOAD CONSTRUCTION, not the parse — office_oxide
/// has already materialized every Cell and every docx block by the time
/// the seam sees them, so nothing here can un-spend the parse, and the
/// html collector has already built the node array. A marshalling bound
/// with a marker (`entries_truncated`), never a refusal: sheet/range/
/// merge/name metadata stays complete past the cap, and per-object Python
/// overhead (>> string bytes) is what a big workbook costs past this
/// point. Truncate-with-marker, never refuse — and never
/// `page_tables`' `tables_refused` shape, which is an engine-cost
/// bail-out with per-page semantics, not a truncation precedent.
pub const MAX_STRUCTURE_ENTRIES: usize = 200_000;

/// Which engine lane a structure surface runs on. Derived, never stored:
/// the third routing table's one served answer per row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StructureLane {
    Docx,
    Xlsx,
    Html,
}

/// The structure surface's format-level refusal template (F8's first
/// family): "document_structure cannot serve {format}: {why}" — fired for
/// every kind refused on ALL backends, before the backend check, so a
/// `(pdf, anydoc)` call answers the pdf refusal, never an anydoc-template
/// message whose "use backend='auto' or 'oxide'" tail would be a lie here
/// (every backend refuses these rows). A capability refusal carried by
/// [`DocumentError::Convert`] (the geometry refusals' own vehicle), never
/// a new error kind.
fn structure_refusal(format: &str, why: &str) -> DocumentError {
    DocumentError::Convert(format!("document_structure cannot serve {format}: {why}"))
}

/// The pdf row's doctrine text: PDF's structure answer is already two
/// surfaces (page_layout's spans/lines/images, page_tables' per-cell
/// boxes), and this one serves the office/html families' containers.
const PDF_STRUCTURE_WHY: &str = "pdf has no structure surface here — page_layout and \
     page_tables serve PDF pages (spans, lines, images, per-cell table boxes); \
     document_structure serves the office/html families' containers";

/// The legacy trio's row: office_oxide parses doc/xls/ppt into its
/// converter-internal IR only (no `as_doc()`/`as_xls()`/`as_ppt()` raw
/// layer), and the IR's structure is the converter's rendering choice —
/// the thing the issues explicitly reject.
const LEGACY_STRUCTURE_WHY: &str = "office_oxide reads {format} only through its \
     converter-internal IR (no raw layer) — the converter's structure is a rendering \
     choice, not the document's";

/// The pptx row: the raw layer IS reachable (`as_pptx()`), so the message
/// must never claim impossibility — the surface is simply not built yet
/// (the review's ruling on open question 1: refuse in v1 with an honest
/// message, ship in v1.5; the single `payload` getter makes a fourth
/// variant additive later).
const PPTX_STRUCTURE_WHY: &str = "pptx's raw layer is reachable but its structure \
     surface is not built yet (slides/shapes/notes)";

/// The ODF/rtf/epub/csv row: these formats convert fine (anydoc's
/// whole-document markdown), but nothing exposes a parse tree: no raw
/// layer anywhere, and "the format converts fine" is stated so the
/// refusal is never misread as "unreadable" (the review's ruling 7).
fn no_structure_surface_why(format: &str) -> String {
    format!(
        "no structure surface exists for {format}: the conversion lane's anydoc engine is \
         whole-document markdown with no parse-tree exposure; office_oxide (the raw-layer \
         engine) reads the six MS formats only. The format converts fine (to_markdown/to_text); \
         it has no parse tree to serve"
    )
}

/// The Excel kind's OLE-container refusal (F9/R11): an OLE compound file
/// under the Excel kind is EITHER a legacy xls workbook OR an encrypted
/// xlsx package — both are CFB, and anydoc's detector answers `None` for
/// `EncryptedPackage`, so the seam cannot tell them apart without a full
/// legacy parse (bespoke CFB stream-name sniffing for one message: not
/// worth it). One union message naming both possibilities, container-aware
/// (`resolved_name`'s "xls"), never the wrong single diagnosis a
/// "legacy xls, no raw layer" message would give a password-protected
/// workbook.
fn structure_ole_refusal() -> DocumentError {
    structure_refusal(
        "xls",
        "the OLE container is a legacy xls workbook or an encrypted xlsx package (both are \
         CFB, and telling them apart needs a full legacy parse this surface does not run); \
         the xlsx raw layer is unreachable through it",
    )
}

/// The anydoc capability refusal on the SERVED office rows (F8's second
/// family): the same shape [`anydoc_capability_refusal`] carries, with
/// this surface's own lack named. Deliberately NOT [`DocumentError::UnsupportedBackend`]:
/// anydoc READS docx/xlsx (it is the conversion lane's engine for them),
/// so the format-level answer would be false; this refusal answers "can
/// this backend serve THIS call" — it cannot, its office lane is
/// whole-document markdown with no raw layer to walk. html's forced
/// backends stay [`DocumentError::UnsupportedBackend`] (byte-identical to
/// [`engine_for`]'s own html row): there the pair cannot convert at all.
fn structure_anydoc_refusal(format: &str) -> DocumentError {
    DocumentError::Convert(format!(
        "backend \"anydoc\" cannot serve document_structure for {format}: no structure \
         surface for that backend (its office lane is whole-document markdown; the raw \
         layers are office_oxide's); use backend='auto' or 'oxide' here"
    ))
}

/// The structure capability table: the THIRD routing table, documented
/// next to (never as a comment on) [`geometry_engine`]. Conversion routing
/// answers "who converts best"; geometry "who can declare geometry at
/// all"; this one answers "who can serve structure at all" — who has a
/// parse tree to report. The served rows are exactly the formats whose
/// engine hands over a raw layer (or, for html, whose collector IS the
/// parse path):
///
/// - docx, xlsx (ZIP): office_oxide's raw layers (`as_docx`/`as_xlsx`).
/// - html: html-to-markdown-rs's structure collector (a runtime option on
///   the conversion lane's own parse — no second parser, no second
///   opinion).
///
/// Every other kind is a format-level refusal ([`structure_refusal`]),
/// fired for EVERY backend (anydoc included, before the backend check —
/// the `(pdf, anydoc)` precedence pin): pdf (its answer is geometry +
/// tables, already two surfaces), the legacy trio (no raw layer; the IR
/// is converter-internal), pptx (raw layer reachable, surface not built),
/// and the anydoc-only families (no structure surface; the format
/// converts fine). html's forced backends keep conversion's own
/// [`DocumentError::UnsupportedBackend`] row byte-identical. The served
/// office rows refuse `backend="anydoc"` with the capability refusal
/// ([`structure_anydoc_refusal`]).
///
/// `bytes` is the (prefix of the) document, so the Excel kind's one
/// container-aware case can decide: an OLE signature under the Excel kind
/// is a legacy xls workbook OR an encrypted xlsx package — refused with
/// the union message ([`structure_ole_refusal`]) for every backend,
/// before the backend match (both possibilities refuse, so the container
/// decision precedes it like the other format-level rows).
///
/// The table is total: every `Kind` is one arm — no format falls out of a
/// default arm unasked.
fn structure_engine(
    kind: Kind,
    backend: Backend,
    bytes: &[u8],
) -> Result<StructureLane, DocumentError> {
    if kind == Kind::Excel && is_ole_container(bytes) {
        return Err(structure_ole_refusal());
    }
    let format_name = resolved_name(kind, bytes);
    match (kind, backend) {
        (Kind::Pdf, _) => Err(structure_refusal(Kind::Pdf.name(), PDF_STRUCTURE_WHY)),
        (Kind::Doc | Kind::Ppt, _) => {
            let why = LEGACY_STRUCTURE_WHY.replace("{format}", format_name);
            Err(structure_refusal(format_name, &why))
        }
        (Kind::Pptx, _) => Err(structure_refusal(format_name, PPTX_STRUCTURE_WHY)),
        (Kind::Rtf | Kind::Odt | Kind::Ods | Kind::Odp | Kind::Epub | Kind::Csv, _) => {
            let why = no_structure_surface_why(format_name);
            Err(structure_refusal(format_name, &why))
        }
        // conversion's own html row, byte-identical: the pair cannot
        // convert, so the structure lane — the same parse — cannot run.
        (Kind::Html, Backend::Auto) => Ok(StructureLane::Html),
        (Kind::Html, _) => Err(DocumentError::UnsupportedBackend {
            format: "html",
            backend: backend_name(backend),
        }),
        (Kind::Docx | Kind::Excel, Backend::Anydoc) => Err(structure_anydoc_refusal(format_name)),
        (Kind::Docx, _) => Ok(StructureLane::Docx),
        (Kind::Excel, _) => Ok(StructureLane::Xlsx),
    }
}

/// The structure surface's pre-read capability probe: the
/// [`structure_engine`] refusal the call will raise, resolved as far as
/// the format resolves WITHOUT the authoritative (whole-document) read —
/// an explicit `format=` decides alone, then the content markers in
/// `bytes` (the `data=` lane's own bytes; the `path=` lane's bounded
/// 64 KiB sniff prefix), then the name hint's extension. Same contract as
/// [`probe_geometry_capability`], keyed on the structure table, and — the
/// divergence the sibling surfaces don't have — fired for EVERY backend:
/// the format-level refusal family precedes the backend check, so a
/// `format="pdf"` refusal needs no bytes at all and fires under the GIL
/// from the name alone.
///
/// - `Err(refusal)`: the probed format is on a refused row (naming the
///   container: an OLE workbook says `xls`). The binding raises it under
///   the GIL, before the detach and before the read: a
///   positive content-resolve to a refused kind stops at the 64 KiB
///   prefix and refuses, never reading the 32 MiB default first.
/// - `Ok(None)`: the probe could not resolve the format. The core's
///   authoritative `resolve` — on the real bytes, post-read — stays the
///   only truth; whatever refusal fires there fires then.
/// - `Ok(Some(kind))`: resolved, and the row serves this backend.
pub fn probe_structure_capability(
    bytes: &[u8],
    name_hint: Option<&str>,
    format: Option<&str>,
    backend: Backend,
) -> Result<Option<Kind>, DocumentError> {
    match resolve(bytes, name_hint, format) {
        Err(_) => Ok(None),
        Ok(kind) => structure_engine(kind, backend, bytes).map(|_| Some(kind)),
    }
}

/// One `document_structure` report: the resolved format's vocabulary name
/// (the [`resolved_name`] spelling — the Excel kind's container-aware
/// `xls`/`xlsx` case included, `PageLayout.format`'s provenance precedent)
/// and exactly one per-format payload ([`StructurePayload`], the
/// compile-time one-of).
#[derive(Clone, Debug, PartialEq)]
pub struct StructureReport {
    /// The resolved format's vocabulary name. Provenance only (constant
    /// per served payload type), but the same spelling the refusals use.
    pub format: &'static str,
    /// The payload: exactly one variant by construction.
    pub payload: StructurePayload,
}

/// The one-of payload ([R9]): a core enum, compile-time total — three
/// optional getters would make the invariant untypable and `format`
/// un-narrowable for every static checker. The core owns plain
/// `#[derive(Clone, Debug, PartialEq)]` structs (office_oxide's own types
/// are `Debug, Clone` only — `CellValue` has no `PartialEq` — so
/// structural `==` is only cheap on the seam's own types), and the
/// binding layer wraps them in thin getter pyclasses the
/// `PageGeometry` way. `isinstance(report.payload, DocxStructure)` is the
/// Python narrowing.
#[derive(Clone, Debug, PartialEq)]
pub enum StructurePayload {
    Docx(DocxStructure),
    Xlsx(XlsxStructure),
    Html(HtmlStructure),
}

/// The DOCX structure payload (issue #209): the body's blocks in document
/// order, the section partition they hang from, and the note counts. The
/// tree is derivable by consumers (sections by the
/// [`DocxStructure::section_breaks`] counts; tables are nested payloads);
/// the list is the anchor surface. Entries carry container anchors, never
/// boxes — no coordinates anywhere.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DocxStructure {
    /// The body's blocks, body order, one entry per
    /// `BlockElement` (paragraph or table). Headers/footers and note
    /// BODIES are not body blocks and are not here (the parity claim's
    /// documented exclusions); their counts ride alongside.
    pub blocks: Vec<DocxBlock>,
    /// PINNED: `body.section_breaks.len() + 1` — the body partition law
    /// `DocxBlock.section` derives from. NOT `doc.sections.len()`: a
    /// document with no body-level sectPr declares zero
    /// `SectionProperties` yet is one section. Declared sizes belong to
    /// `page_geometry`, not here.
    pub sections: usize,
    /// `body.section_breaks` verbatim: section i covers body elements
    /// `[prev_break, section_breaks[i])`; the final section runs to the
    /// end. Anchor math for `DocxBlock.section`.
    pub section_breaks: Vec<usize>,
    /// Footnote/endnote/comment BODY counts (v1 keeps note bodies out;
    /// the counts are the cheap part of the answer).
    pub footnotes: usize,
    /// Endnote body count.
    pub endnotes: usize,
    /// Comment body count.
    pub comments: usize,
    /// The raw layer's `unreadable_parts` verbatim (header/footer/note
    /// parts that exist but could not be read or parsed): the
    /// report-the-loss law, `XlsxStructure.unreadable_sheets`' twin (the
    /// asymmetry the review flagged was a bug, not a choice).
    pub unreadable_parts: Vec<String>,
    /// True when `blocks` stopped at [`MAX_STRUCTURE_ENTRIES`] (the
    /// marshalling bound; sheet/section metadata above stays complete —
    /// never a refusal).
    pub entries_truncated: bool,
}

/// One DOCX body block. `kind` discriminates; `level`/`outline_level`
/// exist for headings, `num_*`/`list_*` for list state (a numbered
/// heading carries BOTH — kind alone is the discriminator), `table` for
/// tables.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DocxBlock {
    /// Position in `body.elements` — the stable container anchor.
    pub index: usize,
    /// Which section covers this block: the count of
    /// `section_breaks <= index` (section i owns elements
    /// `[prev_break, section_breaks[i])`).
    pub section: usize,
    /// `"heading" | "list_item" | "paragraph" | "table"`. Heading wins
    /// over list (Word's multilevel-list "Heading" gallery attaches numPr
    /// to heading styles: a numbered heading is the normal shape of
    /// headings), the converter's own rule.
    pub kind: &'static str,
    /// The block's visible text: runs joined with hidden-run resolution
    /// through the style chain, `w:br` → newline, `w:tab` → tab,
    /// text-box prose recursed, form-field display text surfaced,
    /// note-ref marks skipped, hyperlink runs included inline — the
    /// markdown lane's own block-text rules, re-derived. Pinned `""` for
    /// table blocks (the cells carry the text; a table block's own text
    /// would double it).
    pub text: String,
    /// Headings only: the markdown depth, `clamp(outline + 1, 1..=6)` —
    /// a depth-7..9 outline renders `######` on every markdown lane, so
    /// `level` clamps and `outline_level` carries the true depth.
    /// `None` for everything else.
    pub level: Option<u8>,
    /// Headings only: the heading ladder's raw 0-based outline level
    /// (0..8), the value `level` clamps away. The ladder: direct
    /// `w:outlineLvl` → the stylesheet's `basedOn` chain
    /// (`resolve_outline_level`) → the `Heading1..9` style-ID convention →
    /// the `heading N` style-NAME convention — the converter's ladder,
    /// re-derived in the seam (upstream keeps it crate-private).
    /// `None` for everything else.
    pub outline_level: Option<u8>,
    /// The raw `w:pStyle` string ("Heading1", "Caption", …) — the
    /// document's own name. `None` when the paragraph declares none.
    pub style_id: Option<String>,
    /// Numbering instance id, whenever the EFFECTIVE paragraph
    /// properties carry `numPr` with `numId != 0` — including a numbered
    /// heading's (that numPr is information, never suppressed) and
    /// including style-carried numPr (Word's "List Bullet"/"List Number"
    /// styles hold it in the style, and the markdown renders those
    /// lists). `numId=0` is ECMA-376's "no numbering" and reports `None`.
    pub num_id: Option<u32>,
    /// The numbering level (0-based), alongside `num_id`.
    pub ilvl: Option<u8>,
    /// The resolved number format's own name ("decimal", "bullet",
    /// "lower_letter", …) from `numbering.resolve_level` — the file's
    /// own vocabulary, never a rendered marker.
    pub list_format: Option<String>,
    /// The level's effective start (`startOverride`-aware, from
    /// `numbering.resolve_start`) — the level's own start, NOT a
    /// rendered ordinal (the markdown lane renumbers sequentially).
    pub start_number: Option<u32>,
    /// `kind == "table"` only: the grid.
    pub table: Option<DocxTable>,
    /// The block's hyperlinks, in order (hyperlink runs included).
    pub links: Vec<DocxLink>,
    /// The block's real pictures (drawings with a relationship id), in
    /// order. A DrawingML `shape` (no rid) is not a picture and is
    /// filtered; text-box drawings are not collected (the text walk
    /// recurses, the image list does not — v1 keeps the list anchored to
    /// the block's own runs).
    pub images: Vec<DocxImage>,
}

/// One DOCX table's grid. `cells` are present cells in row-major order;
/// every grid position is either a cell's origin or covered by a span.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DocxTable {
    /// The grid's row count (`rows.len()`).
    pub rows: usize,
    /// The grid's column count (`grid.len()`).
    pub cols: usize,
    /// The `w:tblGrid` column widths, twips, verbatim (the raw layer's
    /// own numbers).
    pub col_widths_twips: Vec<i32>,
    /// The cells, row-major, origin cells AND `vMerge`-continue cells
    /// (a continue cell is emitted as its own cell at its own position,
    /// disclosed as inside the Restart span — the raw layer parses it as
    /// a real `w:tc`, and hiding it would misalign the row-major order).
    pub cells: Vec<DocxCell>,
    /// The `w:tblCaption` accessibility caption — the raw layer's only
    /// caption field (there is no "style convention's caption" in the
    /// raw layer to fall back to).
    pub caption: Option<String>,
}

/// One DOCX table cell. `row`/`col` are 0-based grid positions (col
/// accumulated through the row's `gridSpan`s, the raw layer's own
/// `grid_starts` walk re-derived); spans are 1 unless a merge covers.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DocxCell {
    /// 0-based row.
    pub row: usize,
    /// 0-based starting column (gridSpan accumulation).
    pub col: usize,
    /// The cell's text (the same block walk as body blocks, trimmed).
    pub text: String,
    /// The row span: a `vMerge Restart` plus the count of same-column
    /// `Continue` cells below it; 1 otherwise (a hostile/absent value
    /// stays 1).
    pub row_span: usize,
    /// The column span (`w:gridSpan`, clamped to 1..=1000 — the raw
    /// layer's own `MAX_GRID_SPAN` clamp, re-derived).
    pub col_span: usize,
}

/// One DOCX hyperlink. `href`/`anchor` split the two target kinds
/// (rels-resolved External URL vs `w:anchor` internal bookmark name); a
/// target that carries BOTH a URL and a fragment keeps the fragment
/// alongside (`externalURL#fragment` is one target's two halves, never
/// an internal link).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DocxLink {
    /// The hyperlink's runs' text.
    pub text: String,
    /// The external URL (already resolved through the rels). `None` for
    /// an internal link.
    pub href: Option<String>,
    /// The internal bookmark name. `None` for an external link.
    pub anchor: Option<String>,
    /// The URI fragment when an external link also declares `w:anchor`.
    pub fragment: Option<String>,
    /// The screen-tip tooltip, when declared.
    pub tooltip: Option<String>,
}

/// One DOCX picture (a drawing with a relationship id). Bytes never cross
/// this surface; the rid joins to the package's image parts (the raw
/// layer keys its image map by rid and does not retain the target path,
/// so an image reports rid + extension, never `media/image5.png`).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DocxImage {
    /// The relationship id (`a:blip/@r:embed`, or a linked picture's
    /// `a:blip/@r:link`).
    pub relationship_id: String,
    /// The image part's file extension from the package's image map
    /// ("png", "jpeg", …), `None` when the part is not in the map (a
    /// linked image the package did not embed).
    pub file_ext: Option<String>,
    /// The alt text (`wp:docPr/@descr`).
    pub description: Option<String>,
    /// The declared width, points (EMU ÷ 12700.0, the pptx leg's own
    /// rule). `None` when the declared extent is 0 or negative — an
    /// undeclared extent is 0, and a non-positive one is no extent.
    pub width_pt: Option<f32>,
    /// The declared height, points (the same law).
    pub height_pt: Option<f32>,
    /// True for an inline drawing, false for a floating anchor.
    pub inline: bool,
    /// The `adec:decorative` flag: the picture carries no meaning
    /// (alt-less placeholder), assistive technology skips it. Carried,
    /// never dropped: it is the honest "why is there no alt text".
    pub decorative: bool,
}

/// The XLSX structure payload (issue #210).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct XlsxStructure {
    /// The worksheets, sheet order, hidden included — built from the raw
    /// layer's `worksheets` filtered to worksheet/macro kinds, each
    /// carrying its own name and state (never zip'd against
    /// `workbook.sheets`: that list also names chartsheets and partless
    /// sheets, and a positional zip mislabels on any workbook with one).
    /// Macro sheets are served (their cells are the worksheet cell
    /// model).
    pub sheets: Vec<XlsxSheet>,
    /// The workbook's chart/dialog sheet names — kind-tagged worksheet
    /// entries with no cells in the raw layer's own list, named here so
    /// the tab list stays complete.
    pub chartsheets: Vec<String>,
    /// The workbook-level defined names, verbatim plus the sheet name
    /// a sheet-scoped `local_sheet_id` resolves to (it indexes
    /// `workbook.sheets`, not `sheets`).
    pub defined_names: Vec<XlsxDefinedName>,
    /// The 1904 date system flag, verbatim.
    pub date1904: bool,
    /// Names of sheets whose part could not be parsed — the raw layer's
    /// own (name, error) pairs' names. Content IS partial here; the law
    /// is report-the-loss, the intact sheets served.
    pub unreadable_sheets: Vec<String>,
    /// True when cells stopped at [`MAX_STRUCTURE_ENTRIES`] (the
    /// marshalling bound; sheet/range/merge/name metadata above stays
    /// complete — never a refusal).
    pub entries_truncated: bool,
}

/// One XLSX worksheet.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct XlsxSheet {
    /// The sheet's display name.
    pub name: String,
    /// The visibility state spelled out: `"visible" | "hidden" |
    /// "veryHidden"`.
    pub state: &'static str,
    /// PINNED DEFINITION: the used range computed from the PARSED cells
    /// — min/max over every present cell (styled-but-empty cells count
    /// as present: Excel's own used-range notion) — rendered
    /// `"A1:D10"`. NOT the declared `<dimension>` (attacker-controlled,
    /// frequently absent; a sparse sheet whose dimension lies keeps its
    /// computed range and the declared string rides beside it).
    /// `None` for a sheet with no cells at all.
    pub used_range: Option<String>,
    /// The declared `<dimension>` string, verbatim, for provenance —
    /// never the used range's source.
    pub declared_dimension: Option<String>,
    /// The used range's max row index, 0-based. `None` for an empty
    /// sheet (no cells → no bounds — `used_range` is `Option` and these
    /// are too).
    pub max_row: Option<usize>,
    /// The used range's max column index, 0-based (the same law).
    pub max_col: Option<usize>,
    /// The sheet's present cells, row-major, sparse.
    pub cells: Vec<XlsxCell>,
    /// The merged ranges as ref strings ("B2:C3"), verbatim.
    pub merged_cells: Vec<String>,
    /// The sheet's hyperlinks, part order.
    pub hyperlinks: Vec<XlsxHyperlink>,
    /// The sheet's cell comments, document order.
    pub comments: Vec<XlsxComment>,
}

/// One XLSX cell: the reference, the value as the file states it (no
/// coercion), and the formula when present.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct XlsxCell {
    /// The cell reference ("B3" — the raw `CellRef`'s own Display).
    pub reference: String,
    /// 0-based row.
    pub row: usize,
    /// 0-based column.
    pub col: usize,
    /// The value's own type, spelled out: `"number" | "string" |
    /// "boolean" | "error" | "date" | "empty"`. `date` is ONLY the
    /// `t="d"` ISO-8601 branch; the ordinary Excel date (a serial number
    /// with a date number-format) is `number` — decoding it needs the
    /// stylesheet, which the raw layer only exposes behind `&mut self`
    /// (`ensure_styles`), so the seam reports the value as the file
    /// stores it and the format stays uninterpreted.
    pub kind: &'static str,
    /// The value by `kind`: number → float, boolean → bool,
    /// string/error → the string, date → the ISO-8601 string, empty →
    /// `None`. Shared strings are DEREFERENCED (the full text, no
    /// 32,768-char markdown-lane truncation — a disclosed divergence
    /// from `to_markdown`'s DoS guard: the structure surface reports the
    /// file's own value). A missing shared-string index reports `""`
    /// (the renderer's own `unwrap_or("")`).
    pub value: Option<XlsxScalar>,
    /// The `<f>` formula text, verbatim, when present.
    pub formula: Option<String>,
    /// Formula cells: whether a cached value rode the file (`formula` is
    /// Some AND the value is not Empty). A formula whose cached `<v>` is
    /// absent reports False; a `t="e"` cell with no `<v>` parses to the
    /// empty error string and reports True (the file stated an error).
    pub has_cached_value: bool,
}

/// The Python-side value a cell's `value` getter returns: bool | float |
/// str | None, discriminated by [`XlsxCell::kind`]. The core's own enum
/// so the payload struct stays a plain `PartialEq` struct.
#[derive(Clone, Debug, PartialEq)]
pub enum XlsxScalar {
    /// A boolean cell.
    Bool(bool),
    /// A numeric cell (dates-as-serials included — see `kind`).
    Number(f64),
    /// A string, error, date, or dereferenced shared string.
    Text(String),
}

/// One sheet-scoped or workbook-level defined name.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct XlsxDefinedName {
    /// The name.
    pub name: String,
    /// The formula/reference value, verbatim.
    pub value: String,
    /// The sheet the name is scoped to, 0-based into `workbook.sheets`
    /// (the workbook's OWN sheet list — chartsheets included — never
    /// this payload's `sheets`), verbatim; `None` for a workbook-level
    /// name.
    pub local_sheet_id: Option<u32>,
    /// That sheet's display name resolved from `workbook.sheets`, when
    /// `local_sheet_id` is in range.
    pub local_sheet_name: Option<String>,
    /// The name's hidden flag, verbatim.
    pub hidden: bool,
}

/// One XLSX sheet hyperlink: `href`/`anchor` split exactly like
/// [`DocxLink`] (a sheet-internal location is not a URL string).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct XlsxHyperlink {
    /// The cell the link sits on ("A1").
    pub cell_ref: String,
    /// The external URL. `None` for an internal location.
    pub href: Option<String>,
    /// The internal location (sheet/cell). `None` for a URL.
    pub anchor: Option<String>,
    /// The tooltip, when declared.
    pub tooltip: Option<String>,
}

/// One XLSX cell comment.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct XlsxComment {
    /// The cell the comment is attached to.
    pub cell_ref: String,
    /// The comment's author, when the part names one.
    pub author: Option<String>,
    /// The comment's text.
    pub text: String,
}

/// The HTML structure payload (issue #211): the collector's flat node
/// array. No second tables list — the Table nodes carry the grids (the
/// collector clones each grid into the node AND into
/// `ConversionResult.tables`, whose `TableData` also carries the markdown
/// rendering this surface discards; exposing both would double the table
/// payload for zero information).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct HtmlStructure {
    /// The nodes, reading order, parents before children (a parent index
    /// is always earlier: the collector pushes the container before its
    /// child). The auto-built `Group` tree IS the heading tree; the flat
    /// `Heading` nodes are the entries.
    pub nodes: Vec<HtmlNode>,
    /// True when nodes stopped at [`MAX_STRUCTURE_ENTRIES`]: the kept
    /// prefix stays index-sound (parents are always earlier) and every
    /// kept node's `children` is filtered to kept indices.
    pub entries_truncated: bool,
}

/// One HTML node. `kind` is one of the NINE reachable kinds
/// ("heading", "paragraph", "list", "list_item", "table", "image",
/// "code", "quote", "group"); the crate enum's other variants
/// (definition lists, raw blocks, metadata blocks) are produced only by
/// a builder nothing in the convert path calls, and a future/unreachable
/// variant mapping here is a typed `Convert` error, never a silent
/// `kind=None`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct HtmlNode {
    /// The crate's deterministic node id.
    pub id: String,
    /// The parent node's index (`None` for roots). Always earlier than
    /// this node.
    pub parent: Option<u32>,
    /// The children's indices, reading order.
    pub children: Vec<u32>,
    /// The crate's per-node attribute map — headings carry `{"id": …}`
    /// when the source heading had one (the anchor work's handle);
    /// image nodes carry `None` (the collector drops img attributes:
    /// declared dimensions are NOT served — the spec's own disclosure,
    /// with the visitor-feature path as the v1.x completion).
    pub attributes: Option<std::collections::HashMap<String, String>>,
    /// The node kind (the nine reachable ones, above).
    pub kind: &'static str,
    /// The node's text (heading/paragraph/list_item/code).
    pub text: Option<String>,
    /// Heading level 1-6 (a Group: the level that created it).
    pub level: Option<u8>,
    /// List: whether ordered.
    pub ordered: Option<bool>,
    /// Code: the language, when declared.
    pub language: Option<String>,
    /// Group: the section label — the creating heading's text (the
    /// crate's `label` and `heading_text` are the same string; one
    /// field is carried).
    pub label: Option<String>,
    /// Image: the src URL.
    pub src: Option<String>,
    /// Image: the alt text.
    pub alt: Option<String>,
    /// Table: the grid (rows/cols + sparse origin cells with
    /// row/col/row_span/col_span/is_header). Nested tables: the inner
    /// grid is ALSO collected (its own Table node, the inner table
    /// closing first) while the outer cell carries the flattened text —
    /// the observed collector behavior, disclosed.
    pub grid: Option<HtmlGrid>,
    /// Inline annotations, byte offsets into `text` — links carry
    /// url+title. The link-context rule: a link's enclosing section is
    /// the nearest GROUP ancestor (walk `parent`); headings have no
    /// children (a heading CREATEs a Group sibling), and a link on a
    /// heading lives on the Heading node itself.
    pub annotations: Vec<HtmlAnnotation>,
}

/// One HTML table grid (the crate's `TableGrid`, verbatim shape).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct HtmlGrid {
    /// The grid's row count.
    pub rows: u32,
    /// The grid's column count.
    pub cols: u32,
    /// The origin cells, ordered by (row, col), sparse: positions
    /// covered by a span have no entry.
    pub cells: Vec<HtmlGridCell>,
}

/// One HTML grid cell (the crate's `GridCell`, verbatim shape).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct HtmlGridCell {
    /// The cell's text.
    pub content: String,
    /// 0-based row.
    pub row: u32,
    /// 0-based column.
    pub col: u32,
    /// The row span (1 unless a merge covers).
    pub row_span: u32,
    /// The column span (1 unless a merge covers).
    pub col_span: u32,
    /// Whether this is a header cell (`<th>`).
    pub is_header: bool,
}

/// One inline annotation on an HTML node's text: a byte range plus its
/// kind — url+title for links.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct HtmlAnnotation {
    /// The annotation kind ("link", "bold", "italic", "underline",
    /// "strikethrough", "code", "subscript", "superscript",
    /// "highlight").
    pub kind: &'static str,
    /// Start byte offset into the node's text (inclusive).
    pub start: u32,
    /// End byte offset (exclusive).
    pub end: u32,
    /// Link: the href, verbatim (no decoding, no resolution).
    pub url: Option<String>,
    /// Link: the `title` attribute, when declared.
    pub title: Option<String>,
}

/// The multi-format `document_structure` probe (issues #209/#210/#211):
/// ONE report whose resolved `format` discriminates and whose
/// `payload` is exactly one of the three per-format structures — the
/// docx heading/list/table/link surface from the office_oxide RAW layer,
/// the xlsx sheets/cells/types surface from the same, and the html
/// heading/link/image/table surface from the conversion lane's own
/// structure collector (one parse, no second opinion).
///
/// The spine mirrors [`page_geometry`]'s exactly (resolve → capability
/// table → password → max_bytes → engine doors), with the surface's own
/// refusal order (the review's R7/R8, pinned):
///
/// 1. `resolve` — the explicit `format=` is trusted and NEVER re-sniffed
///    (the conversion lane's own precedence): `format="html"` on docx
///    bytes is the html lane's own binary-input refusal, `format="docx"`
///    on pdf bytes the office leg's typed parse error; the fix-it is
///    `sniff()`. The one container-dependent name is the Excel OLE case
///    (`resolved_name`).
/// 2. `structure_engine` — the format-level refusals (every refused kind
///    on every backend: pdf, the legacy trio, pptx, the anydoc-only
///    families, the Excel-OLE union), then html's forced-backend
///    `UnsupportedBackend`, then the served office rows' anydoc
///    capability refusal.
/// 3. the password refusal — fired for EVERY kind: no served format
///    accepts one, so even a pdf call reaches this refusal with an
///    unexamined password (the surface takes no `password=` parameter;
///    the core check keeps the Rust seam honest).
/// 4. `max_bytes == 0` → `"max_bytes must be positive"`.
/// 5. the engine doors — the office legs' audit + input ceiling (the
///    same 32 MiB default, the same message shape, as
///    [`page_geometry`]'s office leg) and the html leg's input ceiling
///    (the conversion lane's ~23× figure is a LOWER bound for the
///    structure lane, which also holds the node array; 32 MiB kept, so
///    labeled).
///
/// Per-format payload laws live on the payload structs
/// ([`DocxStructure`]/[`XlsxStructure`]/[`HtmlStructure`]) and their
/// entries; the surface-wide laws: entries carry container anchors,
/// never boxes (no coordinates anywhere); loss is reported
/// (`unreadable_parts`/`unreadable_sheets`), never skipped silently; and
/// the one entry cap ([`MAX_STRUCTURE_ENTRIES`]) truncates with a marker
/// rather than refusing.
pub fn document_structure(
    bytes: Vec<u8>,
    name_hint: Option<&str>,
    format: Option<&str>,
    backend: Backend,
    options: &ConvertOptions,
) -> Result<StructureReport, DocumentError> {
    let kind = resolve(&bytes, name_hint, format)?;
    let lane = structure_engine(kind, backend, &bytes)?;
    // The password refusal fires for EVERY kind (the F25 ruling): the
    // geometry surfaces' `kind != Kind::Pdf` guard would let a pdf call
    // reach the engines with an unexamined password, and no format this
    // surface serves accepts one.
    if options.password.is_some() {
        return Err(DocumentError::Convert(
            "document_structure takes no password: no served format accepts one".into(),
        ));
    }
    if options.max_bytes.is_some_and(|limit| limit == 0) {
        return Err(DocumentError::Convert("max_bytes must be positive".into()));
    }
    // The provenance name, resolved while the bytes are still borrowed
    // (the parse below moves them): the Excel kind's container-aware
    // xls/xlsx spelling included.
    let format_name = resolved_name(kind, &bytes);
    let payload = match lane {
        StructureLane::Docx | StructureLane::Xlsx => {
            // The office leg parses the container into resident memory
            // (the same amplification class the conversion lane meters):
            // the same input ceiling, explicit budget over the 32 MiB
            // default, the same message shape as page_geometry's office
            // leg.
            let limit = options.max_bytes.unwrap_or(DEFAULT_ANYDOC_INPUT_LIMIT);
            if bytes.len() > limit {
                return Err(DocumentError::Convert(format!(
                    "the document is {} and the {} engine lane's input ceiling is {} \
                     (that lane amplifies input into resident memory at a measured multiple \
                     — 2026-09-09: ~146x worst case on anydoc's delimiter formats, \
                     ~4100x on a zip-bombed office_oxide container): split the file, or pass a larger \
                     max_bytes",
                    render_size(bytes.len()),
                    engine_name(Engine::OfficeOxide),
                    render_size(limit),
                )));
            }
            audit_oxide_container(&bytes)?;
            let format = match kind {
                Kind::Docx => office_oxide::DocumentFormat::Docx,
                // The Excel kind's OLE container was refused in
                // structure_engine; a ZIP package is xlsx.
                Kind::Excel => office_oxide::DocumentFormat::Xlsx,
                _ => {
                    return Err(DocumentError::Convert(
                        "office_oxide: unreachable kind".into(),
                    ));
                }
            };
            let doc = office_oxide::Document::from_reader(Cursor::new(bytes), format)
                .map_err(|e| Convert(format!("office_oxide: {e}")))?;
            match kind {
                Kind::Docx => {
                    let raw = doc.as_docx().ok_or_else(|| {
                        Convert("office_oxide: the docx parse lost its raw layer".into())
                    })?;
                    StructurePayload::Docx(structure_docx(raw))
                }
                Kind::Excel => {
                    let raw = doc.as_xlsx().ok_or_else(|| {
                        Convert("office_oxide: the xlsx parse lost its raw layer".into())
                    })?;
                    StructurePayload::Xlsx(structure_xlsx(raw))
                }
                // structure_engine's table is total: nothing else routes here.
                _ => {
                    return Err(DocumentError::Convert(
                        "office_oxide: unreachable kind".into(),
                    ));
                }
            }
        }
        StructureLane::Html => {
            // The html leg's input ceiling: the conversion lane's ~23x
            // measured amplification is the figure for the lane WITHOUT
            // the structure collector — this lane also holds the node
            // array (node text duplicates body text, group labels
            // duplicate heading text, annotations and ids), so 23x is a
            // LOWER bound here. The data is already resident in the
            // collector's tree, so the marginal factor is bounded; the
            // 32 MiB default stands, labeled as the lower bound it is.
            let limit = options.max_bytes.unwrap_or(DEFAULT_ANYDOC_INPUT_LIMIT);
            if bytes.len() > limit {
                return Err(DocumentError::Convert(format!(
                    "the document is {} and the {} engine lane's input ceiling is {} \
                     (that lane amplifies input into resident memory at a measured multiple \
                     — 2026-09-09: ~146x worst case on anydoc's delimiter formats, \
                     ~4100x on a zip-bombed office_oxide container): split the file, or pass a larger \
                     max_bytes",
                    render_size(bytes.len()),
                    engine_name(Engine::Html2Md),
                    render_size(limit),
                )));
            }
            StructurePayload::Html(structure_html(&bytes)?)
        }
    };
    Ok(StructureReport {
        format: format_name,
        payload,
    })
}

/// The HTML structure leg: the conversion lane's own parse, run once with
/// the structure collector on (`include_document_structure: true` routes
/// it to tier-2; the conversion output is IGNORED — the surface returns
/// structure only). `result.document` being `None` is a refusal, never an
/// empty structure; `result.content` is deliberately NOT required (the
/// html_markdown lane's `.content.ok_or("no content")` is the wrong copy
/// here: an all-style document can have structure and no markdown).
fn structure_html(bytes: &[u8]) -> Result<HtmlStructure, DocumentError> {
    let html = String::from_utf8_lossy(
        bytes
            .strip_prefix([0xEF, 0xBB, 0xBF].as_slice())
            .unwrap_or(bytes),
    );
    let options = html_to_markdown_rs::ConversionOptions {
        extract_metadata: false,
        include_document_structure: true,
        ..Default::default()
    };
    let result = html_to_markdown_rs::convert(&html, options)
        .map_err(|e| Convert(format!("html-to-markdown-rs: {e}")))?;
    let document = result.document.ok_or_else(|| {
        Convert("html-to-markdown-rs: the structure collector returned no document".into())
    })?;
    let mut nodes = Vec::with_capacity(document.nodes.len());
    for node in &document.nodes {
        nodes.push(html_node(node)?);
    }
    // The entry cap: keep the reading-order prefix. Parents are always
    // earlier (the collector pushes the container before its child), so
    // every kept node's parent is kept; children are filtered to kept
    // indices so no kept node points past the array.
    let entries_truncated = nodes.len() > MAX_STRUCTURE_ENTRIES;
    if entries_truncated {
        nodes.truncate(MAX_STRUCTURE_ENTRIES);
        let kept = MAX_STRUCTURE_ENTRIES as u32;
        for node in &mut nodes {
            node.children.retain(|&child| child < kept);
        }
    }
    Ok(HtmlStructure {
        nodes,
        entries_truncated,
    })
}

/// One collector node into the payload's own type. An unreachable node
/// kind (the `tl`-based builder's variants: definition lists, raw
/// blocks, metadata blocks — nothing in the convert path pushes them)
/// is a typed `Convert` error, never a silent `kind=None`.
fn html_node(node: &html_to_markdown_rs::DocumentNode) -> Result<HtmlNode, DocumentError> {
    let (kind, text, level, ordered, language, label, src, alt, grid) = match &node.content {
        html_to_markdown_rs::NodeContent::Heading { level, text } => (
            "heading",
            Some(text.clone()),
            Some(*level),
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        html_to_markdown_rs::NodeContent::Paragraph { text } => (
            "paragraph",
            Some(text.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        html_to_markdown_rs::NodeContent::List { ordered } => {
            ("list", None, None, Some(*ordered), None, None, None, None, None)
        }
        html_to_markdown_rs::NodeContent::ListItem { text } => (
            "list_item",
            Some(text.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        html_to_markdown_rs::NodeContent::Table { grid } => (
            "table",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(html_grid(grid)),
        ),
        html_to_markdown_rs::NodeContent::Image {
            description,
            src,
            ..
            // image_index is always None here: the inline-images feature
            // is off (default-features = false), so no index is ever
            // assigned — carried as absent, disclosed.
        } => (
            "image",
            None,
            None,
            None,
            None,
            None,
            src.clone(),
            description.clone(),
            None,
        ),
        html_to_markdown_rs::NodeContent::Code { text, language } => (
            "code",
            Some(text.clone()),
            None,
            None,
            language.clone(),
            None,
            None,
            None,
            None,
        ),
        html_to_markdown_rs::NodeContent::Quote => {
            ("quote", None, None, None, None, None, None, None, None)
        }
        html_to_markdown_rs::NodeContent::Group {
            label,
            heading_level,
            ..
            // heading_text duplicates `label` (the collector sets both
            // to the creating heading's text): one field carried.
        } => (
            "group",
            None,
            *heading_level,
            None,
            None,
            label.clone(),
            None,
            None,
            None,
        ),
        // Nothing in the convert path pushes these four (the `tl`-based
        // builder that does is dead); a variant that ever appears is a
        // typed refusal naming it, never a silent kind=None.
        other @ (html_to_markdown_rs::NodeContent::DefinitionList
        | html_to_markdown_rs::NodeContent::DefinitionItem { .. }
        | html_to_markdown_rs::NodeContent::RawBlock { .. }
        | html_to_markdown_rs::NodeContent::MetadataBlock { .. }) => {
            return Err(Convert(format!(
                "html-to-markdown-rs: the structure collector reported the node kind {other:?}, which this surface does not model"
            )));
        }
    };
    let annotations = node
        .annotations
        .iter()
        .map(|a| {
            let (kind, url, title) = match &a.kind {
                html_to_markdown_rs::AnnotationKind::Bold => ("bold", None, None),
                html_to_markdown_rs::AnnotationKind::Italic => ("italic", None, None),
                html_to_markdown_rs::AnnotationKind::Underline => ("underline", None, None),
                html_to_markdown_rs::AnnotationKind::Strikethrough => ("strikethrough", None, None),
                html_to_markdown_rs::AnnotationKind::Code => ("code", None, None),
                html_to_markdown_rs::AnnotationKind::Subscript => ("subscript", None, None),
                html_to_markdown_rs::AnnotationKind::Superscript => ("superscript", None, None),
                html_to_markdown_rs::AnnotationKind::Highlight => ("highlight", None, None),
                html_to_markdown_rs::AnnotationKind::Link { url, title } => {
                    ("link", Some(url.clone()), title.clone())
                }
            };
            HtmlAnnotation {
                kind,
                start: a.start,
                end: a.end,
                url,
                title,
            }
        })
        .collect();
    Ok(HtmlNode {
        id: node.id.clone(),
        parent: node.parent,
        children: node.children.clone(),
        attributes: node.attributes.clone(),
        kind,
        text,
        level,
        ordered,
        language,
        label,
        src,
        alt,
        grid,
        annotations,
    })
}

/// The crate's grid, verbatim shape.
fn html_grid(grid: &html_to_markdown_rs::TableGrid) -> HtmlGrid {
    HtmlGrid {
        rows: grid.rows,
        cols: grid.cols,
        cells: grid
            .cells
            .iter()
            .map(|cell| HtmlGridCell {
                content: cell.content.clone(),
                row: cell.row,
                col: cell.col,
                row_span: cell.row_span,
                col_span: cell.col_span,
                is_header: cell.is_header,
            })
            .collect(),
    }
}

/// The DOCX structure builder: the raw layer's body walked once — the
/// heading ladder re-derived (F1's two-rung markdown lane + the
/// converter's convention rung), list membership on the EFFECTIVE
/// properties, the block-text walk, and the table grids.
fn structure_docx(raw: &office_oxide::docx::DocxDocument) -> DocxStructure {
    let styles = raw.styles.as_ref();
    let mut blocks = Vec::with_capacity(raw.body.elements.len());
    for (index, element) in raw.body.elements.iter().enumerate() {
        let section = raw
            .body
            .section_breaks
            .iter()
            .filter(|&&count| count <= index)
            .count();
        let block = match element {
            office_oxide::docx::BlockElement::Paragraph(p) => {
                let outline = docx_prose::outline_level(p, styles);
                // List membership is on the EFFECTIVE properties (the
                // markdown lane's own rule): Word's "List Bullet"/"List
                // Number" styles carry `w:numPr` in the STYLE, and a
                // direct-`w:pPr` read reports None for exactly the lists
                // the markdown renders.
                let effective = match styles {
                    Some(sheet) => sheet.effective_paragraph_properties(p.properties.as_ref()),
                    None => p.properties.clone().unwrap_or_default(),
                };
                let numbering = effective
                    .numbering_ref
                    .as_ref()
                    .filter(|nr| nr.num_id != 0)
                    .map(|nr| (nr.num_id, nr.ilvl));
                let (num_id, ilvl, list_format, start_number) = match numbering {
                    Some((num_id, ilvl)) => {
                        let format = raw
                            .numbering
                            .as_ref()
                            .and_then(|n| n.resolve_level(num_id, ilvl))
                            .map(|level| number_format_name(&level.format));
                        let start = raw
                            .numbering
                            .as_ref()
                            .and_then(|n| n.resolve_start(num_id, ilvl));
                        (Some(num_id), Some(ilvl), format, start)
                    }
                    None => (None, None, None, None),
                };
                let kind = if outline.is_some() {
                    "heading"
                } else if numbering.is_some() {
                    "list_item"
                } else {
                    "paragraph"
                };
                DocxBlock {
                    index,
                    section,
                    kind,
                    text: docx_prose::paragraph_text(p, styles),
                    level: outline.map(|o| (o as usize + 1).clamp(1, 6) as u8),
                    outline_level: outline,
                    style_id: p.properties.as_ref().and_then(|pp| pp.style_id.clone()),
                    num_id,
                    ilvl,
                    list_format,
                    start_number,
                    table: None,
                    links: docx_prose::paragraph_links(p),
                    images: docx_prose::paragraph_images(p, &raw.images),
                }
            }
            office_oxide::docx::BlockElement::Table(table) => DocxBlock {
                index,
                section,
                kind: "table",
                // PINNED: a table block's own text is "" — the cells
                // carry the text (a table block's text would double it).
                text: String::new(),
                level: None,
                outline_level: None,
                style_id: None,
                num_id: None,
                ilvl: None,
                list_format: None,
                start_number: None,
                table: Some(docx_prose::table_structure(table, styles)),
                links: Vec::new(),
                images: Vec::new(),
            },
        };
        blocks.push(block);
    }
    let entries_truncated = blocks.len() > MAX_STRUCTURE_ENTRIES;
    if entries_truncated {
        blocks.truncate(MAX_STRUCTURE_ENTRIES);
    }
    DocxStructure {
        blocks,
        // The partition law `block.section` derives from: a document with
        // no body-level sectPr is ONE section.
        sections: raw.body.section_breaks.len() + 1,
        section_breaks: raw.body.section_breaks.clone(),
        footnotes: raw.footnotes.len(),
        endnotes: raw.endnotes.len(),
        comments: raw.comments.len(),
        unreadable_parts: raw.unreadable_parts.clone(),
        entries_truncated,
    }
}

/// The numbering format's own name (the file's vocabulary, snake_case:
/// "decimal", "bullet", "lower_letter", …; an unrecognized `w:numFmt`
/// value rides verbatim).
fn number_format_name(format: &office_oxide::docx::NumberFormat) -> String {
    use office_oxide::docx::NumberFormat;
    match format {
        NumberFormat::Decimal => "decimal".to_string(),
        NumberFormat::Bullet => "bullet".to_string(),
        NumberFormat::LowerLetter => "lower_letter".to_string(),
        NumberFormat::UpperLetter => "upper_letter".to_string(),
        NumberFormat::LowerRoman => "lower_roman".to_string(),
        NumberFormat::UpperRoman => "upper_roman".to_string(),
        NumberFormat::None => "none".to_string(),
        NumberFormat::Other(other) => other.clone(),
    }
}

/// The XLSX structure builder: sheets from the raw layer's OWN worksheet
/// list filtered by kind (chartsheets/dialogs are kind-tagged entries
/// INSIDE `worksheets` in 0.1.15 — `Worksheet::without_cells` — so the
/// filter is on `Worksheet.kind`, never a zip against `workbook.sheets`),
/// the used range computed from the parsed cells, and the cells' values
/// as the file states them.
fn structure_xlsx(raw: &office_oxide::xlsx::XlsxDocument) -> XlsxStructure {
    use office_oxide::ir::SheetKind;
    let mut sheets = Vec::new();
    let mut chartsheets = Vec::new();
    let mut entries_truncated = false;
    let mut budget = MAX_STRUCTURE_ENTRIES;
    for ws in &raw.worksheets {
        match ws.kind {
            SheetKind::Chart | SheetKind::Dialog => chartsheets.push(ws.name.clone()),
            SheetKind::Worksheet | SheetKind::Macro => {
                sheets.push(structure_sheet(
                    ws,
                    raw,
                    &mut budget,
                    &mut entries_truncated,
                ));
            }
        }
    }
    XlsxStructure {
        sheets,
        chartsheets,
        defined_names: raw
            .workbook
            .defined_names
            .iter()
            .map(|name| XlsxDefinedName {
                name: name.name.clone(),
                value: name.value.clone(),
                local_sheet_id: name.local_sheet_id,
                local_sheet_name: name
                    .local_sheet_id
                    .and_then(|id| raw.workbook.sheets.get(id as usize))
                    .map(|sheet| sheet.name.clone()),
                hidden: name.hidden,
            })
            .collect(),
        date1904: raw.workbook.date1904,
        unreadable_sheets: raw
            .unreadable_sheets
            .iter()
            .map(|(name, _)| name.clone())
            .collect(),
        entries_truncated,
    }
}

/// One worksheet's payload. `budget` is the shared entry budget across
/// every sheet (the cap is per-PAYLOAD, applied in document order);
/// `truncated` flips once cells are dropped and never flips back.
fn structure_sheet(
    ws: &office_oxide::xlsx::Worksheet,
    raw: &office_oxide::xlsx::XlsxDocument,
    budget: &mut usize,
    truncated: &mut bool,
) -> XlsxSheet {
    let state = match ws.state {
        office_oxide::xlsx::SheetState::Visible => "visible",
        office_oxide::xlsx::SheetState::Hidden => "hidden",
        office_oxide::xlsx::SheetState::VeryHidden => "veryHidden",
    };
    // The used range: computed from the parsed cells — min/max over
    // every present cell, styled-but-empty included (Excel's own
    // used-range notion). The declared <dimension> rides beside it for
    // provenance, never as the source.
    let mut min_row = u32::MAX;
    let mut min_col = u32::MAX;
    let mut max_row = 0u32;
    let mut max_col = 0u32;
    let mut any = false;
    for row in &ws.rows {
        for cell in &row.cells {
            any = true;
            min_row = min_row.min(cell.reference.row);
            min_col = min_col.min(cell.reference.col);
            max_row = max_row.max(cell.reference.row);
            max_col = max_col.max(cell.reference.col);
        }
    }
    let used_range = any.then(|| {
        format!(
            "{mincol}{minr}:{maxcol}{maxr}",
            mincol = office_oxide::xlsx::CellRef::col_name(min_col),
            minr = min_row + 1,
            maxcol = office_oxide::xlsx::CellRef::col_name(max_col),
            maxr = max_row + 1,
        )
    });
    let mut cells = Vec::new();
    'outer: for row in &ws.rows {
        for cell in &row.cells {
            if *budget == 0 {
                *truncated = true;
                break 'outer;
            }
            *budget -= 1;
            let value = match &cell.value {
                office_oxide::xlsx::CellValue::Empty => None,
                office_oxide::xlsx::CellValue::Number(n) => Some(XlsxScalar::Number(*n)),
                office_oxide::xlsx::CellValue::String(s) => Some(XlsxScalar::Text(s.clone())),
                // The shared-string dereference is the structure leg's
                // own job (the cell keeps the index at rest): the FULL
                // text, no 32,768-char markdown-lane truncation — a
                // disclosed divergence from the renderer's DoS guard.
                office_oxide::xlsx::CellValue::SharedString(idx) => Some(XlsxScalar::Text(
                    raw.shared_strings.get(*idx).unwrap_or("").to_string(),
                )),
                office_oxide::xlsx::CellValue::Boolean(b) => Some(XlsxScalar::Bool(*b)),
                office_oxide::xlsx::CellValue::Error(e) => Some(XlsxScalar::Text(e.clone())),
                office_oxide::xlsx::CellValue::Date(date) => {
                    Some(XlsxScalar::Text(date.to_iso_string()))
                }
            };
            let kind = match &cell.value {
                office_oxide::xlsx::CellValue::Empty => "empty",
                office_oxide::xlsx::CellValue::Number(_) => "number",
                office_oxide::xlsx::CellValue::String(_)
                | office_oxide::xlsx::CellValue::SharedString(_) => "string",
                office_oxide::xlsx::CellValue::Boolean(_) => "boolean",
                office_oxide::xlsx::CellValue::Error(_) => "error",
                // `date` is ONLY the t="d" ISO-8601 branch: a serial
                // number with a date number-format stays "number" (the
                // seam cannot decode the format; see `XlsxCell::kind`).
                office_oxide::xlsx::CellValue::Date(_) => "date",
            };
            cells.push(XlsxCell {
                reference: cell.reference.to_string(),
                row: cell.reference.row as usize,
                col: cell.reference.col as usize,
                kind,
                value,
                formula: cell.formula.clone(),
                // A formula whose cached <v> is absent parses to Empty;
                // a t="e" cell with no <v> parses to Error("") — both
                // are the file's own statement.
                has_cached_value: cell.formula.is_some()
                    && !matches!(cell.value, office_oxide::xlsx::CellValue::Empty),
            });
        }
    }
    XlsxSheet {
        name: ws.name.clone(),
        state,
        used_range,
        declared_dimension: ws.dimension.clone(),
        max_row: any.then_some(max_row as usize),
        max_col: any.then_some(max_col as usize),
        cells,
        merged_cells: ws.merged_cells.clone(),
        hyperlinks: ws
            .hyperlinks
            .iter()
            .map(|link| {
                let (href, anchor) = match &link.target {
                    office_oxide::xlsx::HyperlinkTarget::External(url) => (Some(url.clone()), None),
                    office_oxide::xlsx::HyperlinkTarget::Internal(anchor) => {
                        (None, Some(anchor.clone()))
                    }
                };
                XlsxHyperlink {
                    cell_ref: link.cell_ref.clone(),
                    href,
                    anchor,
                    tooltip: link.tooltip.clone(),
                }
            })
            .collect(),
        comments: ws
            .comments
            .iter()
            .map(|comment| XlsxComment {
                cell_ref: comment.cell_ref.clone(),
                author: comment.author.clone(),
                text: comment.text.clone(),
            })
            .collect(),
    }
}

/// The seam's own re-derivation of the office_oxide block-text/heading
/// helpers it needs for the structure payload (the crate keeps them
/// private; the upstream ask is the bundle — `resolve_heading_level` +
/// per-paragraph effective props + per-block text + the grid walks — not
/// one hook). Each helper is pinned against the markdown lane's own
/// behavior by fixture.
mod docx_prose {
    use office_oxide::docx::document::BlockElement;
    use office_oxide::docx::hyperlink::HyperlinkTarget;
    use office_oxide::docx::image::DrawingInfo;
    use office_oxide::docx::paragraph::{Paragraph, ParagraphContent, Run, RunContent};
    use office_oxide::docx::styles::StyleSheet;
    use office_oxide::docx::table::{MergeType, Table};

    /// The single gridSpan clamp, mirroring the raw layer's own
    /// `MAX_GRID_SPAN` (crate-private): Word's table limit is 63
    /// columns; 1000 leaves headroom and keeps a hostile `w:gridSpan`
    /// from sizing anything unboundedly.
    const MAX_GRID_SPAN: u32 = 1_000;

    /// The heading ladder's 0-based outline level for one paragraph:
    /// direct `w:outlineLvl` → the stylesheet's `basedOn` chain
    /// (`resolve_outline_level`) → the `Heading1..9` style-ID convention
    /// → the `heading 1..9` style-NAME convention. The first two rungs
    /// are the markdown lane's own two-rung ladder (docx/text.rs); the
    /// convention rung is the converter's (convert_docx.rs), a disclosed
    /// SUPERSET: a paragraph only the convention rung names a heading is
    /// reported here but renders as plain text on `backend="oxide"`.
    pub fn outline_level(p: &Paragraph, styles: Option<&StyleSheet>) -> Option<u8> {
        let props = p.properties.as_ref()?;
        if let Some(level) = props.outline_level {
            return Some(level);
        }
        let style_id = props.style_id.as_ref()?;
        let sheet = styles?;
        if let Some(level) = sheet.resolve_outline_level(style_id) {
            return Some(level);
        }
        if let Some(level) = heading_level_from_style_name(style_id) {
            return Some(level);
        }
        sheet
            .styles
            .get(style_id)
            .and_then(|style| style.name.as_deref())
            .and_then(heading_level_from_style_name)
    }

    /// The converter's naming-convention rung: `Heading1..9` (style id or
    /// style name, trimmed, case-insensitive) → 0-based level.
    fn heading_level_from_style_name(name: &str) -> Option<u8> {
        let lower = name.trim().to_ascii_lowercase();
        let digits: u8 = lower.strip_prefix("heading")?.trim().parse().ok()?;
        ((1..=9).contains(&digits)).then(|| digits - 1)
    }

    /// One block's visible text: paragraphs join their runs (hidden runs
    /// resolved through the style chain), tables walk their cells.
    /// Whitespace-trimmed at the BLOCK edges (block boundaries carry the
    /// separation; interior breaks and tabs ride verbatim).
    pub fn block_text(elements: &[BlockElement], styles: Option<&StyleSheet>) -> String {
        let mut out = String::new();
        blocks_text(elements, styles, &mut out);
        out.trim().to_string()
    }

    /// One paragraph's text.
    pub fn paragraph_text(p: &Paragraph, styles: Option<&StyleSheet>) -> String {
        let style_id = p.properties.as_ref().and_then(|pp| pp.style_id.as_deref());
        let mut out = String::new();
        for content in &p.content {
            match content {
                ParagraphContent::Run(run) => run_text(run, style_id, styles, &mut out),
                ParagraphContent::Hyperlink(hyperlink) => {
                    for run in &hyperlink.runs {
                        run_text(run, style_id, styles, &mut out);
                    }
                }
            }
        }
        out.trim().to_string()
    }

    fn blocks_text(elements: &[BlockElement], styles: Option<&StyleSheet>, out: &mut String) {
        for element in elements {
            match element {
                BlockElement::Paragraph(p) => {
                    let text = paragraph_text_inner(p, styles);
                    if !text.is_empty() {
                        if !out.is_empty() && !out.ends_with(['\n', ' ', '\t']) {
                            out.push('\n');
                        }
                        out.push_str(&text);
                    }
                    out.push('\n');
                }
                // A cell can hold a table; the walk recurses (the same
                // shape the markdown lane's own walker has).
                BlockElement::Table(table) => {
                    for row in &table.rows {
                        for cell in &row.cells {
                            blocks_text(&cell.content, styles, out);
                        }
                    }
                }
            }
        }
    }

    /// The paragraph walk WITHOUT the trim (the text-box recursion needs
    /// the raw shape the markdown lane's own walker produces).
    fn paragraph_text_inner(p: &Paragraph, styles: Option<&StyleSheet>) -> String {
        let style_id = p.properties.as_ref().and_then(|pp| pp.style_id.as_deref());
        let mut out = String::new();
        for content in &p.content {
            match content {
                ParagraphContent::Run(run) => run_text(run, style_id, styles, &mut out),
                ParagraphContent::Hyperlink(hyperlink) => {
                    for run in &hyperlink.runs {
                        run_text(run, style_id, styles, &mut out);
                    }
                }
            }
        }
        out
    }

    /// One run's text, the markdown lane's own `plain_text_run` rules:
    /// hidden runs (`w:vanish`, resolved through the style chain —
    /// `effective_hidden` is public; the table-style layer it also folds
    /// is `pub(crate)`, so a cell's table-style-hidden run is NOT caught:
    /// disclosed), text, breaks, tabs, chart/diagram text, text-box
    /// prose (recursed, separated), form-field display text, note-ref
    /// marks skipped, unresolvable deferred parts dropped.
    fn run_text(
        run: &Run,
        paragraph_style_id: Option<&str>,
        styles: Option<&StyleSheet>,
        out: &mut String,
    ) {
        if styles.is_some_and(|sheet| {
            sheet.effective_hidden(paragraph_style_id, run.properties.as_ref())
        }) {
            return;
        }
        for content in &run.content {
            match content {
                RunContent::Text(text) => out.push_str(text),
                // Every break kind separates: a line break is a newline,
                // and page/column breaks separate text the same way on
                // the plain-text lane.
                RunContent::Break(_) => out.push('\n'),
                RunContent::Tab => out.push('\t'),
                // A drawing has no text of its own — except a native
                // chart or SmartArt diagram, whose text was resolved
                // from the referenced part at open time.
                RunContent::Drawing(drawing) => {
                    let lines: Vec<&str> = drawing
                        .chart_text
                        .iter()
                        .chain(drawing.dgm_text.iter())
                        .map(String::as_str)
                        .collect();
                    if !lines.is_empty() {
                        if !out.is_empty() && !out.ends_with(['\n', ' ', '\t']) {
                            out.push('\n');
                        }
                        out.push_str(&lines.join("\n"));
                        out.push('\n');
                    }
                }
                // Text-box prose is block content: separated from the
                // surrounding run, recursed (a text box is not formatted
                // by the table it sits in — the markdown lane's own
                // rule).
                RunContent::TextBox(blocks) => {
                    let mut inner = String::new();
                    blocks_text(blocks, styles, &mut inner);
                    let inner = inner.trim();
                    if !inner.is_empty() {
                        if !out.is_empty() && !out.ends_with(['\n', ' ', '\t']) {
                            out.push('\n');
                        }
                        out.push_str(inner);
                        out.push('\n');
                    }
                }
                // The citation point carries no text; the note body is
                // not part of the block.
                RunContent::FootnoteRef(..)
                | RunContent::EndnoteRef(..)
                | RunContent::CommentRef(_)
                | RunContent::CommentRangeStart(_) => {}
                // The field's display text is surfaced when the field
                // carries no cached result runs of its own (the plain
                // text lane's exact rule).
                RunContent::FormField(field) => {
                    if let Some(text) = &field.display_text {
                        out.push_str(text);
                    }
                }
                // Resolved into a TextBox during open when the reference
                // could be followed; an unresolvable one is dropped.
                RunContent::DeferredPart(_) => {}
            }
        }
    }

    /// One paragraph's hyperlinks, in order: target kind split into
    /// href/anchor, the runs' text joined.
    pub fn paragraph_links(p: &Paragraph) -> Vec<super::DocxLink> {
        let mut out = Vec::new();
        for content in &p.content {
            let ParagraphContent::Hyperlink(hyperlink) = content else {
                continue;
            };
            let mut text = String::new();
            for run in &hyperlink.runs {
                for content in &run.content {
                    if let RunContent::Text(text_piece) = content {
                        text.push_str(text_piece);
                    }
                }
            }
            let (href, anchor) = match &hyperlink.target {
                HyperlinkTarget::External(url) => (Some(url.clone()), None),
                HyperlinkTarget::Internal(anchor) => (None, Some(anchor.clone())),
            };
            out.push(super::DocxLink {
                text: text.trim().to_string(),
                href,
                anchor,
                fragment: hyperlink.fragment.clone(),
                tooltip: hyperlink.tooltip.clone(),
            });
        }
        out
    }

    /// One paragraph's real pictures, in order: drawings with a
    /// relationship id (embedded `r:embed` or linked `r:link`) — a
    /// DrawingML shape (empty rid) is not a picture and is filtered.
    /// Extents are EMU ÷ 12700.0, `None` when the declared extent is
    /// not positive (an undeclared extent is 0). Text-box drawings are
    /// not collected here (the text walk recurses; the image list stays
    /// anchored to the block's own runs).
    pub fn paragraph_images(
        p: &Paragraph,
        images: &std::collections::HashMap<String, (Vec<u8>, Option<String>)>,
    ) -> Vec<super::DocxImage> {
        let mut out = Vec::new();
        for content in &p.content {
            let runs: &[Run] = match content {
                ParagraphContent::Run(run) => std::slice::from_ref(run),
                ParagraphContent::Hyperlink(hyperlink) => &hyperlink.runs,
            };
            for run in runs {
                for content in &run.content {
                    let RunContent::Drawing(drawing) = content else {
                        continue;
                    };
                    push_image(drawing, images, &mut out);
                }
            }
        }
        out
    }

    fn push_image(
        drawing: &DrawingInfo,
        images: &std::collections::HashMap<String, (Vec<u8>, Option<String>)>,
        out: &mut Vec<super::DocxImage>,
    ) {
        let rid = if drawing.relationship_id.is_empty() {
            drawing.linked_image_rid.as_deref()
        } else {
            Some(drawing.relationship_id.as_str())
        };
        let Some(rid) = rid else {
            return;
        };
        let width_pt = (drawing.width.0 > 0).then(|| drawing.width.0 as f32 / 12700.0);
        let height_pt = (drawing.height.0 > 0).then(|| drawing.height.0 as f32 / 12700.0);
        out.push(super::DocxImage {
            relationship_id: rid.to_string(),
            file_ext: images.get(rid).and_then(|(_, ext)| ext.clone()),
            description: drawing.description.clone(),
            width_pt,
            height_pt,
            inline: drawing.inline,
            decorative: drawing.decorative,
        });
    }

    /// One table's grid: rows/columns from the raw layer's own numbers,
    /// cells in row-major order with gridSpan-accumulated columns and
    /// vMerge-walked row spans.
    pub fn table_structure(table: &Table, styles: Option<&StyleSheet>) -> super::DocxTable {
        // Each row's cells' starting columns: the gridSpan accumulation
        // (the raw layer's `grid_starts` walk, re-derived with the same
        // clamp).
        let starts: Vec<Vec<usize>> = table
            .rows
            .iter()
            .map(|row| {
                let mut col = 0usize;
                row.cells
                    .iter()
                    .map(|cell| {
                        let start = col;
                        col = col.saturating_add(grid_span(cell) as usize);
                        start
                    })
                    .collect()
            })
            .collect();
        let mut cells = Vec::new();
        for (row_index, row) in table.rows.iter().enumerate() {
            for (cell_index, cell) in row.cells.iter().enumerate() {
                let col = starts[row_index][cell_index];
                // A vMerge Restart spans the same-column Continue cells
                // below it; a Continue cell is emitted as its own cell
                // (disclosed as inside the Restart span), row span 1.
                let row_span = match cell
                    .properties
                    .as_ref()
                    .and_then(|props| props.vertical_merge)
                {
                    Some(MergeType::Restart) => {
                        let mut span = 1usize;
                        let mut below = row_index + 1;
                        while below < table.rows.len() {
                            let Some(index) = starts[below].iter().position(|&start| start == col)
                            else {
                                break;
                            };
                            let merged = &table.rows[below].cells[index];
                            match merged
                                .properties
                                .as_ref()
                                .and_then(|props| props.vertical_merge)
                            {
                                Some(MergeType::Continue) => {
                                    span += 1;
                                    below += 1;
                                }
                                _ => break,
                            }
                        }
                        span
                    }
                    _ => 1,
                };
                cells.push(super::DocxCell {
                    row: row_index,
                    col,
                    text: block_text(&cell.content, styles),
                    row_span,
                    col_span: grid_span(cell) as usize,
                });
            }
        }
        super::DocxTable {
            rows: table.rows.len(),
            cols: table.grid.len(),
            col_widths_twips: table.grid.iter().map(|width| width.0).collect(),
            cells,
            caption: table
                .properties
                .as_ref()
                .and_then(|properties| properties.caption.clone()),
        }
    }

    /// The cell's `w:gridSpan`, clamped to `1..=MAX_GRID_SPAN` (the raw
    /// layer's own clamp; the value is untrusted).
    fn grid_span(cell: &office_oxide::docx::table::TableCell) -> u32 {
        cell.properties
            .as_ref()
            .and_then(|props| props.grid_span)
            .unwrap_or(1)
            .clamp(1, MAX_GRID_SPAN)
    }
}

/// The HTML conversion: html-to-markdown-rs under the options the lane
/// measured with. `extract_metadata: false` suppresses the YAML frontmatter
/// its default configuration emits for `<title>`/`<meta>`: the body is the
/// IR, and the title is not body content (a document's own `<h1>` carries
/// the identity). The UTF-8 BOM is stripped at the door: it is an encoding
/// signature, not content, and left in place it rides into the markdown as
/// an invisible U+FEFF (measured on a BOM-prefixed page: the output opened
/// with one, and a BOM-only file converted to its BOM and nothing else).
fn html_markdown(bytes: &[u8]) -> Result<String, DocumentError> {
    let html = String::from_utf8_lossy(
        bytes
            .strip_prefix([0xEF, 0xBB, 0xBF].as_slice())
            .unwrap_or(bytes),
    );
    let options = html_to_markdown_rs::ConversionOptions {
        extract_metadata: false,
        ..Default::default()
    };
    html_to_markdown_rs::convert(&html, options)
        .map_err(|e| Convert(format!("html-to-markdown-rs: {e}")))
        .and_then(|result| {
            result
                .content
                .ok_or_else(|| Convert("html-to-markdown-rs: no content".into()))
        })
}

fn pages_error(err: pdf_impl::PagesError) -> DocumentError {
    match err {
        pdf_impl::PagesError::Pages(what) => DocumentError::Pages(what),
        pdf_impl::PagesError::Pdf(err) => pdf_error(err),
    }
}

fn pdf_error(err: pdf_impl::PdfError) -> DocumentError {
    match err {
        pdf_impl::PdfError::Io(io) => Io(io),
        other => Convert(other.to_string()),
    }
}

fn anydoc_error(err: AnydocError) -> DocumentError {
    match err {
        AnydocError::Io(io) => Io(io),
        AnydocError::NeedsOcr { pages, page_count } => {
            // anydoc reports 1-based page numbers; this surface names pages
            // as 0-based indices everywhere, so the conversion lives here:
            // the one place the engine's answer crosses the seam.
            // `saturating_sub`: a nonsense 0 from the engine maps to the
            // first page instead of wrapping.
            let pages = pages
                .into_iter()
                .map(|page| page.saturating_sub(1))
                .collect();
            DocumentError::NeedsOcr { pages, page_count }
        }
        other => Convert(other.to_string()),
    }
}

#[cfg(all(test, feature = "documents"))]
mod tests {

    use super::*;

    /// The polluted-head HTML fixture: title, style, and script all noise a
    /// web-page converter must not let into the body markdown, around a
    /// structure-bearing body.
    const POLLUTED_HTML: &str = "<!DOCTYPE html>\
<html><head><title>Page Title</title>\
<style>body { color: red; }</style>\
<script>console.log(\"tracking junk\")</script></head>\
<body><h1>Real Heading</h1>\
<p>See the <a href=\"https://handbook.example.com/torque\">field handbook</a> for caf&#233; tables.</p>\
<ul><li>first checkpoint</li><li><ul><li>nested detail</li></ul></li></ul>\
<table><tr><th>Unit</th><th>Status</th></tr><tr><td>T-101</td><td>healthy</td></tr></table>\
</body></html>";

    /// A hand-built [MS-CFB] (OLE) compound file with one named stream:
    /// the minimal container: 512-byte header, one FAT sector, one
    /// directory sector (Root Entry + the stream), then 8 sectors of junk
    /// stream data (exactly the mini-stream cutoff, so the stream lives in
    /// regular sectors and no mini FAT is needed). The byte layout was
    /// verified parseable by cfb (anydoc's OLE reader) through the
    /// installed wheel before being pinned here; the stream's content is
    /// junk on purpose: these pins are about container identity, not about
    /// any parser's success on fake BIFF bytes.
    fn ole_with_stream(name: &str) -> Vec<u8> {
        const SECTOR: usize = 512;
        const FREE: u32 = 0xFFFF_FFFF;
        const END_OF_CHAIN: u32 = 0xFFFF_FFFE;
        const FAT_SECT: u32 = 0xFFFF_FFFD;
        const STREAM_SECTORS: usize = 8;

        fn dirent(name: &str, kind: u8, child: u32, start: u32, size: u64) -> [u8; 128] {
            let mut entry = [0u8; 128];
            let mut encoded: Vec<u16> = name.encode_utf16().collect();
            encoded.push(0);
            for (i, unit) in encoded.iter().enumerate() {
                entry[2 * i..2 * i + 2].copy_from_slice(&unit.to_le_bytes());
            }
            entry[64..66].copy_from_slice(&((encoded.len() * 2) as u16).to_le_bytes());
            entry[66] = kind;
            entry[67] = 1; // black
            entry[68..72].copy_from_slice(&FREE.to_le_bytes()); // left sibling
            entry[72..76].copy_from_slice(&FREE.to_le_bytes()); // right sibling
            entry[76..80].copy_from_slice(&child.to_le_bytes());
            entry[116..120].copy_from_slice(&start.to_le_bytes());
            entry[120..128].copy_from_slice(&size.to_le_bytes());
            entry
        }

        let data = [b'J'; STREAM_SECTORS * SECTOR];
        let mut fat = [FREE; 128];
        fat[0] = FAT_SECT;
        fat[1] = END_OF_CHAIN; // the directory sector's chain
        for i in 0..STREAM_SECTORS {
            fat[2 + i] = if i == STREAM_SECTORS - 1 {
                END_OF_CHAIN
            } else {
                (3 + i) as u32
            };
        }
        let mut header = vec![0u8; SECTOR];
        header[0..8].copy_from_slice(&OLE_MAGIC);
        header[24..26].copy_from_slice(&0x003Eu16.to_le_bytes()); // minor version
        header[26..28].copy_from_slice(&3u16.to_le_bytes()); // version 3
        header[28..30].copy_from_slice(&0xFFFEu16.to_le_bytes()); // little-endian
        header[30..32].copy_from_slice(&9u16.to_le_bytes()); // 512-byte sectors
        header[32..34].copy_from_slice(&6u16.to_le_bytes()); // 64-byte mini sectors
        header[44..48].copy_from_slice(&1u32.to_le_bytes()); // one FAT sector
        header[48..52].copy_from_slice(&1u32.to_le_bytes()); // directory at sector 1
        header[56..60].copy_from_slice(&0x1000u32.to_le_bytes()); // mini-stream cutoff
        header[60..64].copy_from_slice(&END_OF_CHAIN.to_le_bytes()); // no mini FAT
        header[68..72].copy_from_slice(&END_OF_CHAIN.to_le_bytes()); // no DIFAT chain
        header[76..80].copy_from_slice(&0u32.to_le_bytes()); // DIFAT[0]: the FAT is sector 0
        for slot in 1..109 {
            header[76 + 4 * slot..80 + 4 * slot].copy_from_slice(&FREE.to_le_bytes());
        }
        let mut out = header;
        for entry in fat {
            out.extend(entry.to_le_bytes());
        }
        out.extend(dirent("Root Entry", 5, 1, END_OF_CHAIN, 0));
        out.extend(dirent(name, 2, FREE, 2, data.len() as u64));
        out.extend([0u8; 128]); // the sector's two unallocated entries
        out.extend([0u8; 128]);
        out.extend(data);
        out
    }

    /// A stored (method 0, no compression) zip of the given parts: the
    /// container identity without any deflate dependency. CRC-32 is
    /// computed bitwise (no crc crate in the tree); correctness matters
    /// only so the archive reads as well-formed to anydoc's package
    /// reader, which keys on part names, never on file content.
    fn zip_of(parts: &[(&str, &[u8])]) -> Vec<u8> {
        fn crc32(data: &[u8]) -> u32 {
            let mut crc = 0xFFFF_FFFFu32;
            for &byte in data {
                crc ^= byte as u32;
                for _ in 0..8 {
                    crc = if crc & 1 != 0 {
                        (crc >> 1) ^ 0xEDB8_8320
                    } else {
                        crc >> 1
                    };
                }
            }
            !crc
        }
        let mut out = Vec::new();
        let mut central = Vec::new();
        for (name, data) in parts {
            let offset = out.len() as u32;
            let crc = crc32(data);
            let name = name.as_bytes();
            // local file header: stored, no flags, no timestamps
            out.extend(b"PK\x03\x04");
            out.extend(20u16.to_le_bytes()); // version needed
            out.extend(0u16.to_le_bytes()); // flags
            out.extend(0u16.to_le_bytes()); // method: stored
            out.extend(0u16.to_le_bytes()); // time
            out.extend(0u16.to_le_bytes()); // date
            out.extend(crc.to_le_bytes());
            out.extend((data.len() as u32).to_le_bytes()); // compressed
            out.extend((data.len() as u32).to_le_bytes()); // uncompressed
            out.extend((name.len() as u16).to_le_bytes());
            out.extend(0u16.to_le_bytes()); // extra length
            out.extend(name);
            out.extend(*data);
            central.extend(b"PK\x01\x02");
            central.extend(20u16.to_le_bytes()); // version made by
            central.extend(20u16.to_le_bytes()); // version needed
            central.extend(0u16.to_le_bytes()); // flags
            central.extend(0u16.to_le_bytes()); // method: stored
            central.extend(0u16.to_le_bytes()); // time
            central.extend(0u16.to_le_bytes()); // date
            central.extend(crc.to_le_bytes());
            central.extend((data.len() as u32).to_le_bytes()); // compressed
            central.extend((data.len() as u32).to_le_bytes()); // uncompressed
            central.extend((name.len() as u16).to_le_bytes());
            central.extend(0u16.to_le_bytes()); // extra length
            central.extend(0u16.to_le_bytes()); // comment length
            central.extend(0u16.to_le_bytes()); // disk start
            central.extend(0u16.to_le_bytes()); // internal attributes
            central.extend(0u32.to_le_bytes()); // external attributes
            central.extend(offset.to_le_bytes()); // local header offset
            central.extend(name);
        }
        let cd_offset = out.len() as u32;
        let cd_size = central.len() as u32;
        out.extend(&central);
        out.extend(b"PK\x05\x06");
        out.extend(0u16.to_le_bytes()); // this disk
        out.extend(0u16.to_le_bytes()); // central-directory disk
        out.extend((parts.len() as u16).to_le_bytes()); // entries on this disk
        out.extend((parts.len() as u16).to_le_bytes()); // total entries
        out.extend(cd_size.to_le_bytes());
        out.extend(cd_offset.to_le_bytes());
        out.extend(0u16.to_le_bytes()); // comment length
        out
    }

    /// A minimal readable docx (the committed-corpus generator's shape):
    /// [Content_Types].xml + word/document.xml with one paragraph.
    fn write_docx_bytes() -> Vec<u8> {
        let document = b"<?xml version=\"1.0\"?>\n<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body><w:p><w:r><w:t>Torque figures</w:t></w:r></w:p></w:body></w:document>";
        let content_types = b"<?xml version=\"1.0\"?>\n<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"xml\" ContentType=\"application/xml\"/><Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/></Types>";
        zip_of(&[
            ("[Content_Types].xml", content_types),
            ("word/document.xml", document),
        ])
    }

    /// A minimal readable two-page PDF: the same hand-built object-graph
    /// byte layout `pdf_impl`'s test module writes (correct offsets, xref,
    /// trailer; that module's pins defend the layout: this one only needs
    /// a document the pdf lane can actually convert, for the lane guards
    /// and the entity-policy wiring).
    fn two_page_pdf(first: &str, second: &str) -> Vec<u8> {
        let stream = |text: &str| -> Vec<u8> {
            let content = format!("BT /F1 12 Tf 50 700 Td ({text}) Tj ET").into_bytes();
            let mut v = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
            v.extend(&content);
            v.extend(b"\nendstream");
            v
        };
        let s1 = stream(first);
        let s2 = stream(second);
        let objects: Vec<&[u8]> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            &s1,
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 7 0 R >>",
            &s2,
        ];
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = vec![0usize];
        for (number, body) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n", number + 1).as_bytes());
            out.extend_from_slice(body);
            out.extend_from_slice(b"\nendobj\n");
        }
        let xref_at = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n", objects.len() + 1).as_bytes());
        out.extend_from_slice(b"0000000000 65535 f \n");
        for offset in &offsets[1..] {
            out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF",
                objects.len() + 1
            )
            .as_bytes(),
        );
        out
    }

    /// The must-fail probe: a conversion that succeeds is a test bug, not
    /// an engine property. The name rides along so the extension fallback
    /// and the error's named input both behave as in the payload.
    fn convert_err(name: &str, bytes: &[u8]) -> DocumentError {
        match to_markdown(bytes.to_vec(), Some(name), None, Backend::Auto, None) {
            Err(err) => err,
            Ok(_) => panic!("{name} converted, expected an error"),
        }
    }

    #[test]
    fn html_conversion_is_the_clean_body_only() {
        let converted = to_markdown(
            POLLUTED_HTML.as_bytes().to_vec(),
            Some("no-leak.html"),
            None,
            Backend::Auto,
            None,
        )
        .unwrap();
        assert_eq!(converted.format, "html");
        for noise in [
            "Page Title",
            "color: red",
            "console.log",
            "tracking junk",
            "title:",
        ] {
            assert!(
                !converted.output.contains(noise),
                "HTML noise leaked into the body markdown: {noise:?}"
            );
        }
        assert!(converted.output.contains("# Real Heading"));
        assert!(
            converted
                .output
                .contains("[field handbook](https://handbook.example.com/torque)")
        );
        // entities fully decoded: &#233; -> é
        assert!(converted.output.contains("café"));
        // GFM table with a delimiter row: the bar the chunker keys on.
        // Cell padding is the engine's choice, so the header row is matched
        // structurally (a pipe row carrying the cell texts) and the
        // delimiter row by its character class.
        let header_row = converted
            .output
            .lines()
            .any(|l| l.starts_with('|') && l.contains("Unit") && l.contains("Status"));
        let delimiter_row = converted.output.lines().any(|l| {
            l.starts_with('|')
                && l.contains("--")
                && l.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
        });
        assert!(
            header_row,
            "no GFM table header row in: {:?}",
            converted.output
        );
        assert!(
            delimiter_row,
            "no GFM table delimiter row in: {:?}",
            converted.output
        );
    }

    #[test]
    fn html_sniffing_takes_the_markers_and_rejects_lookalikes() {
        assert_eq!(sniff(POLLUTED_HTML.as_bytes()), Some("html"));
        // case-insensitive and whitespace-tolerant: the docstring's
        // claims, each pinned: uppercase element, uppercase doctype, a
        // BOM plus blank lines before the marker, mixed case
        assert_eq!(sniff(b"\n  <HTML lang=\"en\">"), Some("html"));
        assert_eq!(
            sniff(b"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01//EN\">"),
            Some("html")
        );
        assert_eq!(
            sniff(b"\xef\xbb\xbf\n\n<html><body></body></html>"),
            Some("html")
        );
        assert_eq!(sniff(b"<HtMl lang='en'><body></body></html>"), Some("html"));
        // text that merely mentions markup is not HTML
        assert_eq!(sniff(b"we discuss <html> tags in chapter 3"), None);
    }

    #[test]
    fn signature_less_csv_resolves_from_content_alone() {
        // The no-extension routing doctrine: consistent delimitation across
        // lines is the only witness a signature-less format has.
        assert_eq!(sniff(b"unit,status\nT-101,healthy\n"), Some("csv"));
        assert_eq!(sniff(b"unit;status\nT-101;healthy\n"), Some("csv"));
        assert_eq!(sniff(b"unit\tstatus\nT-101\thealthy\n"), Some("csv"));
        // two lines minimum: one line of comma-separated words is prose
        assert_eq!(sniff(b"one, line only"), None);
        // and the counts must agree and be non-zero on every line: prose
        // whose comma counts vary is not a table. (Prose whose counts never
        // vary is the documented false positive (the accepted price of
        // content-only resolution) and is deliberately not asserted
        // against: the contract is the delimiter agreement, not prose
        // immunity.)
        assert_eq!(sniff(b"one, line\nand another here\n"), None);
        assert_eq!(sniff(b"one, line\ntwo commas, here, too\n"), None);
        // non-UTF-8 is not CSV text
        assert_eq!(sniff(b"\xff\xfe,\x00\n\x01,\x02\n"), None);
    }

    #[test]
    fn json_lines_is_not_csv_no_matter_how_the_commas_agree() {
        // The one structured format whose delimiter counts agree across
        // lines (every record serializes the same keys): the witness
        // alone would claim it, and anydoc's csv parser would mangle the
        // records. The record-open guard ({ or [) declines it; the fix is
        // the caller's: json-lines is deliberately not a documents
        // format (name it and parse it as what it is).
        assert_eq!(sniff(b"{\"a\":1,\"b\":2}\n{\"a\":3,\"b\":4}\n"), None);
        assert_eq!(sniff(b"[{\"a\":1},{\"a\":2}]\n[{\"a\":3}]\n"), None);
        // ...and the escape hatch stands: a real csv named as one converts
        let named = to_markdown(
            b"{\"a\":1,\"b\":2}\n".to_vec(),
            None,
            Some("csv"),
            Backend::Auto,
            None,
        )
        .unwrap();
        assert_eq!(named.format, "csv");
    }

    #[test]
    fn the_anydoc_lane_ceiling_is_a_clean_error_and_an_override() {
        // A tiny explicit ceiling (the same guard the 32 MiB default
        // runs, at fixture scale): over -> Convert naming both sizes and
        // the override; under -> converts. The default's value is pinned
        // separately (the constant is the policy). At this tiny scale the
        // sizes print as raw bytes (the sub-0.1-MiB regime, where the old
        // MiB rounding read "0.0 MiB … 0.0 MiB": two sizes, neither
        // legible) and must be two distinct legible numbers.
        let rows = b"unit,status\nT-101,healthy\n".repeat(20);
        let over = to_markdown_with(
            rows.clone(),
            Some("rows.csv"),
            None,
            Backend::Auto,
            None,
            ConvertOptions {
                max_bytes: Some(64),
                ..Default::default()
            },
        );
        let Err(DocumentError::Convert(what)) = over else {
            panic!("the ceiling must be a Convert error");
        };
        assert!(
            what.contains("ceiling") && what.contains("max_bytes"),
            "must name the ceiling and the override: {what:?}"
        );
        assert!(
            what.contains("520 bytes") && what.contains("64 bytes"),
            "tiny-scale sizes must be raw, distinct, legible: {what:?}"
        );
        assert!(
            !what.contains("max_bytes="),
            "no dangling '=' on the override: {what:?}"
        );
        let under = to_markdown_with(
            rows,
            Some("rows.csv"),
            None,
            Backend::Auto,
            None,
            ConvertOptions {
                max_bytes: Some(1024),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(under.format, "csv");
        assert_eq!(DEFAULT_ANYDOC_INPUT_LIMIT, 32 * 1024 * 1024);
    }

    #[test]
    fn the_oxide_lane_shares_the_input_ceiling() {
        // office_oxide had no input bound of its own (measured,
        // against 0.1.9, which had no decompression caps at all: a 333
        // KiB zip-bombed docx peaked at 1.7 GiB RSS on that lane, before
        // the ceiling was extended to it), so the opt-in lane gets the
        // same input-side guard as the default one: named for its
        // engine lane. What the guard does not do is bound decompression
        // beyond office_oxide 0.1.10's own 512 MiB-per-part cap: there is
        // no total-across-parts cap and no output cap (see
        // ConvertOptions::max_bytes's honest statement of that).
        let docx = write_docx_bytes();
        let over = to_markdown_with(
            docx.clone(),
            Some("tiny.docx"),
            None,
            Backend::Oxide,
            None,
            ConvertOptions {
                max_bytes: Some(64),
                ..Default::default()
            },
        );
        let Err(DocumentError::Convert(what)) = over else {
            panic!("the oxide lane must refuse an over-limit input");
        };
        assert!(
            what.contains("office_oxide engine lane's input ceiling"),
            "the refusal must name the engine lane generically: {what:?}"
        );
        // under the same ceiling, the lane converts as before
        let under = to_markdown_with(
            docx,
            Some("tiny.docx"),
            None,
            Backend::Oxide,
            None,
            ConvertOptions {
                max_bytes: Some(1024 * 1024),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(under.format, "docx");
    }

    #[test]
    fn a_zero_max_bytes_is_refused_as_nonsense() {
        // Some(0) is not a policy ("nothing may exceed nothing"): it would
        // either refuse every input with "0.0 vs 0.0" arithmetic or, on
        // empty input, convert nothing while pretending to be a limit. The
        // binding layer already refuses it under the GIL; the core is
        // self-sufficient about it too.
        let rows = b"unit,status\nT-101,healthy\n".to_vec();
        let err = to_markdown_with(
            rows,
            Some("rows.csv"),
            None,
            Backend::Auto,
            None,
            ConvertOptions {
                max_bytes: Some(0),
                ..Default::default()
            },
        );
        let Err(DocumentError::Convert(what)) = err else {
            panic!("a zero max_bytes must be a Convert error");
        };
        assert_eq!(what, "max_bytes must be positive");
    }

    #[test]
    fn a_password_on_a_non_pdf_kind_is_an_error_not_a_silent_ignore() {
        // The caller believes the document is protected; silently
        // ignoring their password would leave them sure of a property
        // the lane never checked.
        let docx = write_docx_bytes();
        let err = to_markdown_with(
            docx,
            Some("locked.docx"),
            None,
            Backend::Auto,
            None,
            ConvertOptions {
                password: Some("torque"),
                ..Default::default()
            },
        );
        assert!(matches!(err, Err(DocumentError::Convert(_))));
    }

    #[test]
    fn a_password_on_the_anydoc_pdf_lane_is_refused_up_front() {
        // The anydoc PDF lane's reader takes no password at all: a correct
        // password would fail Convert("document is encrypted") exactly
        // like a wrong one: indistinguishable from having been ignored.
        // The guard fires before any parsing semantics (an unencrypted
        // fixture is enough to pin it: the refusal, not the parse, is the
        // behavior), mirroring the pages= lane guard.
        let pdf = two_page_pdf("Alpha page text", "Beta page text");
        let err = to_markdown_with(
            pdf,
            None,
            Some("pdf"),
            Backend::Anydoc,
            None,
            ConvertOptions {
                password: Some("whatever-it-says"),
                ..Default::default()
            },
        );
        let Err(DocumentError::Convert(what)) = err else {
            panic!("the anydoc PDF lane must refuse a password up front");
        };
        assert_eq!(
            what,
            "password= requires the pdf_oxide lane (backend='auto' or 'oxide'); \
             anydoc's PDF reader takes no password"
        );
        // and the pdf_oxide lanes take it as before: same bytes, same
        // password, auto routing: an unencrypted document ignores the
        // password and converts
        let pdf = two_page_pdf("Alpha page text", "Beta page text");
        let converted = to_markdown_with(
            pdf,
            None,
            Some("pdf"),
            Backend::Auto,
            None,
            ConvertOptions {
                password: Some("whatever-it-says"),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(converted.format, "pdf");
        assert!(converted.output.contains("Alpha page text"));
    }

    #[test]
    fn the_excel_kind_reports_its_container_honestly() {
        const OLE: &[u8] = &[0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1];
        assert_eq!(resolved_name(Kind::Excel, OLE), "xls");
        assert_eq!(resolved_name(Kind::Excel, b"PK\x03\x04anything"), "xlsx");
        // every other kind is its own name
        assert_eq!(resolved_name(Kind::Pdf, OLE), "pdf");
    }

    #[test]
    fn explicit_names_cover_the_vocabulary() {
        for (name, kind) in [
            ("pdf", Kind::Pdf),
            ("html", Kind::Html),
            ("htm", Kind::Html),
            ("xhtml", Kind::Html),
            ("XHTML", Kind::Html),
            (".docx", Kind::Docx),
            ("DOCX", Kind::Docx),
            ("xlsx", Kind::Excel),
            ("xls", Kind::Excel),
            ("xlsm", Kind::Excel),
            ("ppt", Kind::Ppt),
            ("pptx", Kind::Pptx),
            ("rtf", Kind::Rtf),
            ("odt", Kind::Odt),
            ("epub", Kind::Epub),
            ("csv", Kind::Csv),
            ("tsv", Kind::Csv),
            ("TSV", Kind::Csv),
        ] {
            assert_eq!(Kind::from_name(name), Some(kind), "name {name:?}");
        }
        assert_eq!(Kind::from_name("bogus"), None);
    }

    #[test]
    fn the_routing_table_is_exactly_the_measured_one() {
        // `.ok()`: DocumentError carries io::Error and cannot derive
        // PartialEq; the Ok arms are what this table pins, and the Err arms
        // assert through matches! below.
        assert_eq!(
            engine_for(Kind::Pdf, Backend::Auto).ok(),
            Some(Engine::PdfOxide)
        );
        assert_eq!(
            engine_for(Kind::Pdf, Backend::Oxide).ok(),
            Some(Engine::PdfOxide)
        );
        assert_eq!(
            engine_for(Kind::Pdf, Backend::Anydoc).ok(),
            Some(Engine::Anydoc)
        );
        assert_eq!(
            engine_for(Kind::Html, Backend::Auto).ok(),
            Some(Engine::Html2Md)
        );
        assert!(matches!(
            engine_for(Kind::Html, Backend::Oxide),
            Err(DocumentError::UnsupportedBackend { format: "html", .. })
        ));
        assert!(matches!(
            engine_for(Kind::Html, Backend::Anydoc),
            Err(DocumentError::UnsupportedBackend { format: "html", .. })
        ));
        for kind in [Kind::Docx, Kind::Excel, Kind::Ppt, Kind::Pptx, Kind::Doc] {
            assert_eq!(
                engine_for(kind, Backend::Oxide).ok(),
                Some(Engine::OfficeOxide)
            );
        }
        for kind in [
            Kind::Odt,
            Kind::Ods,
            Kind::Odp,
            Kind::Epub,
            Kind::Rtf,
            Kind::Csv,
        ] {
            assert!(matches!(
                engine_for(kind, Backend::Oxide),
                Err(DocumentError::UnsupportedBackend { .. })
            ));
        }
        for kind in [Kind::Docx, Kind::Excel, Kind::Rtf, Kind::Odt, Kind::Csv] {
            assert_eq!(engine_for(kind, Backend::Auto).ok(), Some(Engine::Anydoc));
            assert_eq!(engine_for(kind, Backend::Anydoc).ok(), Some(Engine::Anydoc));
        }
    }

    #[test]
    fn pages_are_a_pdf_oxide_lane_parameter() {
        // non-PDF format: a Pages error, not a silent whole-document run
        assert!(matches!(
            to_markdown(
                POLLUTED_HTML.as_bytes().to_vec(),
                Some("pages.html"),
                None,
                Backend::Auto,
                Some(vec![0])
            ),
            Err(DocumentError::Pages(_))
        ));
    }

    #[test]
    fn unknown_format_names_and_undetectable_files_fail_loudly() {
        const MYSTERY: &[u8] = b"just some bytes, no markers at all";
        // and the errors name their input: the bogus format string, the
        // name, so the caller can tell which one to fix; the data= shape
        // (no name at all) names its fix instead
        let Err(DocumentError::UnknownFormat(what)) = to_markdown(
            MYSTERY.to_vec(),
            Some("mystery.bin"),
            Some("bogus"),
            Backend::Auto,
            None,
        ) else {
            panic!("an unknown format name must be UnknownFormat");
        };
        assert!(
            what.contains("bogus"),
            "must name the format string: {what:?}"
        );
        let Err(DocumentError::UnknownFormat(what)) = to_markdown(
            MYSTERY.to_vec(),
            Some("mystery.bin"),
            None,
            Backend::Auto,
            None,
        ) else {
            panic!("an undetectable file must be UnknownFormat");
        };
        assert!(what.contains("mystery.bin"), "must name the file: {what:?}");
        let Err(DocumentError::UnknownFormat(what)) =
            to_markdown(MYSTERY.to_vec(), None, None, Backend::Auto, None)
        else {
            panic!("nameless undetectable bytes must be UnknownFormat");
        };
        assert!(
            what.contains("format="),
            "the no-name error must name its fix: {what:?}"
        );
    }

    #[test]
    fn xhtml_prologues_resolve_as_html() {
        // The XML declaration precedes the doctype, so a prefix-anchored
        // marker check has to skip exactly one to see XHTML content at all
        // (anydoc's Format vocabulary knows neither the name nor the
        // bytes: this lane owns both).
        const WITH_DOCTYPE: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">\
<html xmlns=\"http://www.w3.org/1999/xhtml\"><body><h1>Prologue Heading</h1></body></html>";
        const BARE_MARKER: &str =
            "<?xml version=\"1.0\"?>\n<html><body><p>marker only</p></body></html>";
        assert_eq!(sniff(WITH_DOCTYPE.as_bytes()), Some("html"));
        assert_eq!(sniff(BARE_MARKER.as_bytes()), Some("html"));
        // every tolerance at once: BOM, blank lines, declaration, doctype
        let bommed = format!("\u{feff}\n  {WITH_DOCTYPE}");
        assert_eq!(sniff(bommed.as_bytes()), Some("html"));
        // the name spells it too, and only the real extension: "xhtm" is
        // not a thing
        for name in ["xhtml", "XHTML", ".xhtml"] {
            assert_eq!(Kind::from_name(name), Some(Kind::Html), "name {name:?}");
        }
        assert_eq!(Kind::from_name("xhtm"), None);
        // end-to-end on the no-extension doctrine: content markers alone
        // convert the extensionless file, and the .xhtml extension carries
        // one too
        let converted = to_markdown(
            WITH_DOCTYPE.as_bytes().to_vec(),
            Some("xhtmlless"),
            None,
            Backend::Auto,
            None,
        )
        .unwrap();
        assert_eq!(converted.format, "html");
        assert!(converted.output.contains("# Prologue Heading"));
        let converted = to_markdown(
            WITH_DOCTYPE.as_bytes().to_vec(),
            Some("page.xhtml"),
            None,
            Backend::Auto,
            None,
        )
        .unwrap();
        assert_eq!(converted.format, "html");
        assert!(converted.output.contains("# Prologue Heading"));
    }

    #[test]
    fn one_xml_declaration_is_skipped_but_the_marker_stays_strict() {
        // The skip is one optional declaration, not "any leading XML":
        // every XML vocabulary opens with a declaration-then-tag, and the
        // full-document markers must not claim the tag ones.
        for not_html in [
            "<?xml version=\"1.0\"?><svg xmlns=\"http://www.w3.org/2000/svg\"/>",
            "<?xml version=\"1.0\"?><book xmlns=\"http://docbook.org/ns/docbook\"/>",
            // a second declaration is not skipped: malformed by any spec
            "<?xml?><?xml?><html><body></body></html>",
            // an unterminated declaration is malformed markup, not a marker
            "<?xml version=\"1.0\"<html>",
        ] {
            assert_eq!(sniff(not_html.as_bytes()), None, "sniffed {not_html:?}");
        }
    }

    #[test]
    fn html_fragments_need_a_name_or_an_extension() {
        // The fragment doctrine (see looks_like_html): markers are
        // full-document only, because a leading `<tag` would claim every
        // XML vocabulary. An extensionless fragment is UnknownFormat; the
        // escape hatches (format="html" and the .html-family extension)
        // both convert it.
        const FRAGMENT: &str = "<div><p>fragment body</p></div>";
        assert_eq!(sniff(FRAGMENT.as_bytes()), None);
        let err = convert_err("fragment", FRAGMENT.as_bytes());
        assert!(
            matches!(err, DocumentError::UnknownFormat(_)),
            "got {err:?}"
        );
        let converted = to_markdown(
            FRAGMENT.as_bytes().to_vec(),
            Some("fragment.html"),
            None,
            Backend::Auto,
            None,
        )
        .unwrap();
        assert_eq!(converted.format, "html");
        assert!(converted.output.contains("fragment body"));
        let explicit = to_markdown(
            FRAGMENT.as_bytes().to_vec(),
            None,
            Some("html"),
            Backend::Auto,
            None,
        )
        .unwrap();
        assert_eq!(explicit.format, "html");
        assert!(explicit.output.contains("fragment body"));
    }

    #[test]
    fn degenerate_extensionless_files_fail_loudly_not_weirdly() {
        // Zero bytes, whitespace only, BOM only: no marker, no extension:
        // a clean UnknownFormat (the error, never a panic or a hang, and
        // each degenerate byte set runs its own full conversion).
        for (label, bytes) in [
            ("empty", b"".as_slice()),
            ("whitespace-only", b"  \n\t \n".as_slice()),
            ("bom-only", b"\xef\xbb\xbf".as_slice()),
        ] {
            let err = convert_err("degenerate", bytes);
            match err {
                DocumentError::UnknownFormat(what) => assert!(
                    what.contains("degenerate"),
                    "must name the input, got {what:?} ({label})"
                ),
                other => panic!("{label}: degenerate file surfaced {other:?}"),
            }
        }
    }

    #[test]
    fn empty_files_with_extensions_pin_each_engine_honestly() {
        // Empty-by-extension reaches the named engine, and each engine's
        // own empty-input surface is the pinned behavior: anydoc's csv
        // parser yields an empty document (empty output: a zero-record
        // table is a valid empty file, not an error), the HTML engine
        // converts "" to "", and pdf_oxide names its header failure.
        let converted =
            to_markdown(Vec::new(), Some("empty.csv"), None, Backend::Auto, None).unwrap();
        assert_eq!((converted.format, converted.output.as_str()), ("csv", ""));
        let converted =
            to_markdown(Vec::new(), Some("empty.html"), None, Backend::Auto, None).unwrap();
        assert_eq!((converted.format, converted.output.as_str()), ("html", ""));
        let err = convert_err("empty.pdf", b"");
        match err {
            DocumentError::Convert(what) => assert!(
                what.contains("%PDF-"),
                "the empty-PDF error should name the missing header, got {what:?}"
            ),
            other => panic!("empty .pdf surfaced {other:?}"),
        }
    }

    #[test]
    fn a_utf8_bom_never_leaks_into_html_output() {
        // The BOM is an encoding signature, not content: before the strip
        // it rode into the markdown as an invisible U+FEFF (measured on a
        // BOM-prefixed page: the output opened with one). Stripped at the
        // engine door, also for a BOM-only file whose whole output is "".
        let bommed = [b"\xef\xbb\xbf".as_slice(), POLLUTED_HTML.as_bytes()].concat();
        let converted = to_markdown(
            bommed.clone(),
            Some("bommed.html"),
            None,
            Backend::Auto,
            None,
        )
        .unwrap();
        assert!(
            !converted.output.contains('\u{feff}'),
            "the BOM leaked into: {:?}",
            converted.output
        );
        assert!(converted.output.contains("# Real Heading"));
        let converted = to_markdown(
            b"\xef\xbb\xbf".to_vec(),
            Some("bom-only.html"),
            None,
            Backend::Auto,
            None,
        )
        .unwrap();
        assert_eq!((converted.format, converted.output.as_str()), ("html", ""));
    }

    #[test]
    fn csv_heuristic_edges_are_pinned() {
        // BOM rides the first field (not a delimiter); lines() strips the
        // \r of a \r\n ending so CRLF counts stay clean; quoted commas that
        // agree route to the csv kind (anydoc's quote-aware parser does the
        // real reading: routing, not parsing, is the heuristic's job)
        // while differing counts are prose-shaped and rejected.
        assert_eq!(
            sniff(b"\xef\xbb\xbfunit,status\nT-101,healthy\n"),
            Some("csv")
        );
        assert_eq!(sniff(b"unit,status\r\nT-101,healthy\r\n"), Some("csv"));
        assert_eq!(sniff(b"\"x,y\",z\n\"1,2\",3\n"), Some("csv"));
        assert_eq!(sniff(b"name,note\n\"a,b\",c\n"), None);
        // single-column text carries 0 delimiters on every line: not a
        // table without format=
        assert_eq!(sniff(b"unit\nT-101\n"), None);
        // a lone \r is not a line ending for lines(): classic-Mac files
        // are not heuristically CSV
        assert_eq!(sniff(b"unit,status\rT-101,healthy\r"), None);
        // UTF-16 text fails the UTF-8 gate, but the format= escape hatch
        // reaches anydoc's BOM-aware csv decoder (probed through the wheel
        // before being pinned here)
        let mut utf16 = vec![0xFF, 0xFE];
        utf16.extend(
            "unit,status\nT-101,ok\n"
                .encode_utf16()
                .flat_map(u16::to_le_bytes),
        );
        assert_eq!(sniff(&utf16), None);
        let converted = to_markdown(utf16.clone(), None, Some("csv"), Backend::Auto, None).unwrap();
        assert_eq!(converted.format, "csv");
        assert!(converted.output.contains("T-101"));
    }

    #[test]
    fn the_extension_fallback_shares_the_name_vocabulary() {
        // One vocabulary table: the extension resolves through the same
        // from_name the explicit format= name uses, so the name-only
        // spellings work from the file name too. A one-line .tsv (the
        // content heuristic demands two lines) and an .xhtml fragment (no
        // full-document marker) both convert by extension, where a second
        // spelling table would have dropped them on the floor.
        let converted = to_markdown(
            b"unit\tstatus".to_vec(),
            Some("one-line.tsv"),
            None,
            Backend::Auto,
            None,
        )
        .unwrap();
        assert_eq!(converted.format, "csv");
        assert!(converted.output.contains("unit"));
        let converted = to_markdown(
            b"<div><p>xhtml fragment</p></div>".to_vec(),
            Some("fragment.xhtml"),
            None,
            Backend::Auto,
            None,
        )
        .unwrap();
        assert_eq!(converted.format, "html");
        assert!(converted.output.contains("xhtml fragment"));
    }

    #[test]
    fn the_name_hint_extension_fallback_handles_the_edge_spellings() {
        // Probed (all four through the wheel before being
        // pinned here): the extension is the text after the last dot of
        // the whole hint, which handles the hidden file (`.html`'s last
        // dot is its first), the uppercase spelling (case-insensitive
        // vocabulary), and declines the pseudo-extensions: a dot inside a
        // directory component (`dir.d/file`: "d/file" names nothing) and
        // a real-but-unknown one (`archive.tar.gz`: "gz" names nothing),
        // where resolution falls to the content markers, never a bogus
        // claim.
        let fragment = b"<div><p>fragment body</p></div>".to_vec();
        for hint in [".html", "FILE.HTML"] {
            let converted =
                to_markdown(fragment.clone(), Some(hint), None, Backend::Auto, None).unwrap();
            assert_eq!(converted.format, "html", "hint {hint:?}");
            assert!(converted.output.contains("fragment body"));
        }
        // the pseudo-extension shapes: no marker in the bytes -> the
        // honest UnknownFormat naming the input; an HTML marker -> the
        // marker decides (the extension never overclaims)
        const PLAIN: &[u8] = b"just text, no markers at all";
        for hint in ["dir.d/file", "archive.tar.gz"] {
            let err = convert_err(hint, PLAIN);
            assert!(
                matches!(err, DocumentError::UnknownFormat(ref what) if what.contains(hint)),
                "hint {hint:?} surfaced {err:?}"
            );
            let converted = to_markdown(
                b"<html><body><p>marker wins</p></body></html>".to_vec(),
                Some(hint),
                None,
                Backend::Auto,
                None,
            )
            .unwrap();
            assert_eq!(converted.format, "html", "hint {hint:?}");
        }
    }

    #[test]
    fn to_text_entity_policy_follows_the_engine_lane() {
        // The wiring pin for the per-engine entity flag (probed
        // through the wheel): anydoc escapes entity-shaped `&` in its
        // markdown (`a &amp; b` cell -> `a &amp;amp; b`), so the anydoc
        // lane's strip un-escapes once: the document's own text comes
        // back. pdf_oxide emits text literally, so its lane never
        // un-escapes: a PDF whose text layer contains `&amp;`
        // keeps it (un-escaping would corrupt text that is not an escape).
        let csv = b"unit,note\nT-101,a &amp; b\n".to_vec();
        let markdown = to_markdown(csv.clone(), None, Some("csv"), Backend::Auto, None).unwrap();
        assert!(
            markdown.output.contains("&amp;amp;"),
            "anydoc must have escaped the entity-shaped &: {:?}",
            markdown.output
        );
        let text = to_text(csv, None, Some("csv"), Backend::Auto, None).unwrap();
        assert!(
            text.output.contains("a &amp; b"),
            "the anydoc lane un-escapes exactly once: {:?}",
            text.output
        );
        // the pdf lane: same characters, emitted literally, left alone
        let pdf = two_page_pdf("a &amp; b", "second page");
        let markdown = to_markdown(pdf.clone(), None, Some("pdf"), Backend::Auto, None).unwrap();
        assert!(
            markdown.output.contains("&amp;") && !markdown.output.contains("&amp;amp;"),
            "pdf_oxide emits the text literally: {:?}",
            markdown.output
        );
        let text = to_text(pdf, None, Some("pdf"), Backend::Auto, None).unwrap();
        assert!(
            text.output.contains("a &amp; b"),
            "the pdf lane must NOT un-escape: {:?}",
            text.output
        );
    }

    #[test]
    fn the_error_types_display_the_strings_the_payload_raises() {
        // Display + std::error::Error make the cores `?`-able into
        // Box<dyn Error>/anyhow; the strings are pinned byte-equal to the
        // binding layer's, so a Rust caller reads exactly what the Python
        // side raises (the payload's mapping arms carry these same
        // literals).
        let err = DocumentError::UnsupportedBackend {
            format: "html",
            backend: "oxide",
        };
        assert_eq!(
            err.to_string(),
            "backend \"oxide\" cannot read format \"html\": use backend='auto'"
        );
        let err = DocumentError::NeedsOcr {
            pages: vec![2, 5, 1],
            page_count: 9,
        };
        assert_eq!(
            err.to_string(),
            "pages 2, 5, 1 of 9 need OCR (indices are 0-based, like pages=): \
             route this document to an OCR stage"
        );
        // the `?`-into-Box<dyn Error> seam itself: both error types
        let boxed: Box<dyn std::error::Error> = DocumentError::Convert("nope".into()).into();
        assert_eq!(boxed.to_string(), "nope");
        let boxed: Box<dyn std::error::Error> =
            pdf_impl::PagesError::Pages("out of range".into()).into();
        assert_eq!(boxed.to_string(), "out of range");
    }

    #[test]
    fn the_anydoc_capability_refusal_names_the_lack_and_the_alternative() {
        // The PDF-family payload calls' shared doctrine, pinned byte-equal
        // (UnsupportedBackend's message shape (backend and value, the
        // lack, the fix) with the capability named per call and the
        // anydoc PDF surface pointed at): the binding layer raises this
        // exact text, under the GIL, for all four functions, so the
        // string is the contract and lives with the routing table.
        let err = anydoc_capability_refusal(
            "the per-page text probe (the OCR-routing signal)",
            Kind::Pdf.name(),
        );
        assert_eq!(
            err.to_string(),
            "backend \"anydoc\" cannot serve the per-page text probe (the OCR-routing signal): \
             anydoc's PDF surface is whole-document conversion only — to_markdown/to_text \
             with backend=\"anydoc\" (NeedsOcrError is that lane's scanned-page signal); \
             use backend='auto' or 'oxide' here"
        );
        // the other three capabilities ride the same doctrine text
        for capability in [
            "the page tree (a count its reader never returns on success)",
            "per-page classification (its only per-page knowledge is the binary needs-OCR refusal)",
            "the /Annots link walk (it has no annotation surface)",
        ] {
            let text = anydoc_capability_refusal(capability, Kind::Pdf.name()).to_string();
            assert!(
                text.contains(capability),
                "must name the capability: {text:?}"
            );
            assert!(
                text.contains("to_markdown/to_text with backend=\"anydoc\""),
                "must point at the anydoc PDF surface: {text:?}"
            );
        }
        // the office-family refusal (the geometry surface's second
        // message, the spec's own string): same shape, the format named,
        // the lack restated — byte-pinned per format.
        for format in ["docx", "pptx", "xlsx", "doc", "xls", "ppt"] {
            let err = anydoc_capability_refusal("page geometry", format);
            assert_eq!(
                err.to_string(),
                format!(
                    "backend \"anydoc\" cannot serve page geometry for {format}: anydoc has \
                     no geometry surface; use backend='auto' or 'oxide' here"
                )
            );
        }
        // and the geometry probe's own PDF capability rides the historical
        // message unchanged (the format parameter selects the doctrine
        // text, never a second PDF wording)
        let geometry_pdf = anydoc_capability_refusal(PDF_GEOMETRY_CAPABILITY, "pdf").to_string();
        assert!(geometry_pdf.contains("cannot serve the declared page-box geometry"));
        assert!(geometry_pdf.contains("to_markdown/to_text with backend=\"anydoc\""));
    }

    #[test]
    fn the_geometry_capability_table_is_total() {
        // Every kind × every backend: the §6 matrix, asserted as the
        // function's own totality. `Ok(None)` is the no-geometry row (no
        // engine runs, the backend-independent [None]); the office and
        // pdf rows route to their engines for auto/oxide and refuse
        // anydoc; and — the deliberate divergence from conversion
        // routing — NO no-geometry pair raises UnsupportedBackend here
        // (html+oxide/anydoc and the ODF family+oxide convert-refuse, but
        // geometry runs no engine on that row, so there is nothing to
        // refuse).
        for backend in [Backend::Auto, Backend::Oxide, Backend::Anydoc] {
            match geometry_engine(Kind::Pdf, backend, b"") {
                Ok(Some(Engine::PdfOxide)) => assert!(backend != Backend::Anydoc),
                Err(DocumentError::Convert(what)) => {
                    assert_eq!(backend, Backend::Anydoc, "pdf");
                    assert!(
                        what.contains("cannot serve the declared page-box geometry"),
                        "{what}"
                    );
                }
                other => panic!("pdf x {backend:?}: {other:?}"),
            }
            for kind in [Kind::Docx, Kind::Pptx, Kind::Excel, Kind::Doc, Kind::Ppt] {
                match geometry_engine(kind, backend, b"") {
                    Ok(Some(Engine::OfficeOxide)) => assert!(backend != Backend::Anydoc),
                    Err(DocumentError::Convert(what)) => {
                        assert_eq!(backend, Backend::Anydoc, "{kind:?}");
                        assert!(what.contains("no geometry surface"), "{what}");
                    }
                    other => panic!("{kind:?} x {backend:?}: {other:?}"),
                }
            }
            for kind in [
                Kind::Html,
                Kind::Rtf,
                Kind::Odt,
                Kind::Ods,
                Kind::Odp,
                Kind::Epub,
                Kind::Csv,
            ] {
                assert_eq!(
                    geometry_engine(kind, backend, b"").ok(),
                    Some(None),
                    "the no-geometry row must be backend-independent: {kind:?} x {backend:?}"
                );
            }
        }
    }

    // --- the page-geometry spine and its office legs (the §9 pins) ---------

    /// A minimal readable docx whose body XML is the fixture's to write
    /// (the sectPr/pgSz shapes the DOCX leg's pins need). The container is
    /// `write_docx_bytes`'s shape: content types + document part.
    fn docx_with_body(body: &str) -> Vec<u8> {
        let document = format!(
            "<?xml version=\"1.0\"?>\n<w:document xmlns:w=\"http://schemas.openxmlformats.org/\
             wordprocessingml/2006/main\"><w:body>{body}</w:body></w:document>"
        );
        let content_types = b"<?xml version=\"1.0\"?>\n<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"xml\" ContentType=\"application/xml\"/><Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/></Types>";
        let rels = b"<?xml version=\"1.0\"?>\n<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/></Relationships>";
        zip_of(&[
            ("[Content_Types].xml", content_types),
            ("_rels/.rels", rels),
            ("word/document.xml", document.as_bytes()),
        ])
    }

    /// A minimal readable pptx: `slide_size` (the raw sldSz fragment, or
    /// `None` to omit the element) and the slide parts (an empty deck for
    /// `&[]`).
    fn pptx_with(slide_size: Option<&str>, slides: &[&str]) -> Vec<u8> {
        let sld_id_list = slides
            .iter()
            .enumerate()
            .map(|(index, _)| {
                format!(
                    "<p:sldId id=\"{}\" r:id=\"rId{}\"/>",
                    256 + index,
                    index + 1
                )
            })
            .collect::<String>();
        let sld_sz = slide_size.unwrap_or("");
        let presentation = format!(
            "<?xml version=\"1.0\"?>\n<p:presentation xmlns:p=\"http://schemas.openxmlformats.org/\
             presentationml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/\
             officeDocument/2006/relationships\"><p:sldIdLst>{sld_id_list}</p:sldIdLst>\
             {sld_sz}</p:presentation>"
        );
        let mut entries: Vec<(&str, Vec<u8>)> = vec![
            (
                "[Content_Types].xml",
                br#"<?xml version="1.0"?><?xml version="1.0"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/></Types>"#
                    .to_vec(),
            ),
            (
                "_rels/.rels",
                br#"<?xml version="1.0"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>"#
                    .to_vec(),
            ),
            ("ppt/presentation.xml", presentation.into_bytes()),
        ];
        let mut slide_rels = String::from(
            "<?xml version=\"1.0\"?>\n<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">",
        );
        for (index, body) in slides.iter().enumerate() {
            let name = format!("ppt/slides/slide{}.xml", index + 1);
            let slide = format!(
                "<?xml version=\"1.0\"?>\n<p:sld xmlns:p=\"http://schemas.openxmlformats.org/\
                 presentationml/2006/main\" xmlns:a=\"http://schemas.openxmlformats.org/\
                 drawingml/2006/main\"><p:cSld><p:spTree><p:nvGrpSpPr/><p:grpSpPr/>{body}\
                 </p:spTree></p:cSld></p:sld>"
            );
            entries.push((Box::leak(name.into_boxed_str()), slide.into_bytes()));
            slide_rels.push_str(&format!(
                "<Relationship Id=\"rId{}\" Type=\"http://schemas.openxmlformats.org/\
                 officeDocument/2006/relationships/slide\" Target=\"slides/slide{}.xml\"/>",
                index + 1,
                index + 1
            ));
        }
        slide_rels.push_str("</Relationships>");
        entries.push(("ppt/_rels/presentation.xml.rels", slide_rels.into_bytes()));
        let owned: Vec<(&str, &[u8])> = entries
            .iter()
            .map(|(name, data)| (*name, data.as_slice()))
            .collect();
        zip_of(&owned)
    }

    /// A minimal readable xlsx: `sheet_states` carries one sheet name per
    /// entry (`Some("hidden")` for a hidden one); `sheet_extras` rides
    /// into every sheet's XML (the pageSetup fragment the doctrine pin
    /// needs).
    fn xlsx_with(sheets: &[(&str, Option<&str>)], sheet_extras: &str) -> Vec<u8> {
        let main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
        let rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
        let sheet_list = sheets
            .iter()
            .enumerate()
            .map(|(index, (name, state))| {
                let hidden = match state {
                    Some("hidden") => " state=\"hidden\"",
                    _ => "",
                };
                format!(
                    "<sheet name=\"{name}\" sheetId=\"{}\" r:id=\"rId{}\"{hidden}/>",
                    index + 1,
                    index + 1
                )
            })
            .collect::<String>();
        let workbook = format!(
            "<?xml version=\"1.0\"?>\n<workbook xmlns=\"{main_ns}\" xmlns:r=\"{rel_ns}\">\
             <sheets>{sheet_list}</sheets></workbook>"
        );
        let mut rels = String::from(
            "<?xml version=\"1.0\"?>\n<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">",
        );
        for index in 0..sheets.len() {
            rels.push_str(&format!(
                "<Relationship Id=\"rId{}\" Type=\"{rel_ns}/worksheet\" \
                 Target=\"worksheets/sheet{}.xml\"/>",
                index + 1,
                index + 1
            ));
        }
        rels.push_str("</Relationships>");
        let mut content_types = String::from(
            "<?xml version=\"1.0\"?>\n<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/><Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>",
        );
        for index in 0..sheets.len() {
            content_types.push_str(&format!(
                "<Override PartName=\"/xl/worksheets/sheet{}.xml\" ContentType=\
                 \"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>",
                index + 1
            ));
        }
        content_types.push_str("</Types>");
        let mut entries: Vec<(&str, Vec<u8>)> = vec![
            ("[Content_Types].xml", content_types.into_bytes()),
            (
                "_rels/.rels",
                format!(
                    "<?xml version=\"1.0\"?>\n<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"{rel_ns}/officeDocument\" Target=\"xl/workbook.xml\"/></Relationships>"
                )
                .into_bytes(),
            ),
            ("xl/workbook.xml", workbook.into_bytes()),
            ("xl/_rels/workbook.xml.rels", rels.into_bytes()),
        ];
        for index in 0..sheets.len() {
            let name = format!("xl/worksheets/sheet{}.xml", index + 1);
            let sheet = format!(
                "<?xml version=\"1.0\"?>\n<worksheet xmlns=\"{main_ns}\">\
                 <sheetData><row r=\"1\"><c r=\"A1\" t=\"inlineStr\"><is><t>cell</t></is></c>\
                 </row></sheetData>{sheet_extras}</worksheet>"
            );
            entries.push((Box::leak(name.into_boxed_str()), sheet.into_bytes()));
        }
        let owned: Vec<(&str, &[u8])> = entries
            .iter()
            .map(|(name, data)| (*name, data.as_slice()))
            .collect();
        zip_of(&owned)
    }

    /// The geometry answers for bytes named by extension: the spine's
    /// entry as the binding calls it (no explicit format; the extension
    /// resolves).
    fn geometry_named(name: &str, bytes: &[u8]) -> Vec<Option<PageGeometry>> {
        page_geometry(
            bytes.to_vec(),
            Some(name),
            None,
            Backend::Auto,
            &ConvertOptions::default(),
        )
        .unwrap()
    }

    fn geometry_err(name: &str, bytes: &[u8], backend: Backend) -> DocumentError {
        page_geometry(
            bytes.to_vec(),
            Some(name),
            None,
            backend,
            &ConvertOptions::default(),
        )
        .expect_err("the call must refuse")
    }

    /// §9 item 8, the DOCX leg: pgSz-bearing → Some(612, 792) exact;
    /// margins-only → None (the raw-layer amendment's own pinning test —
    /// this FAILS under the to_ir()-only path, which fabricates Letter);
    /// no sectPr → the single None; `<w:pgSz>` without attributes → the
    /// engine's Letter default (the named engine limitation); a section
    /// index reaching past the raw sections → None, the converter's own
    /// `sections.get(idx)` rule; office entries always rotation 0, no
    /// crop.
    #[test]
    fn the_docx_geometry_is_the_raw_layers_declared_size() {
        let pgsz = geometry_named(
            "a.docx",
            &docx_with_body(
                "<w:p><w:r><w:t>body</w:t></w:r></w:p>\
                 <w:sectPr><w:headerReference w:type=\"default\" r:id=\"rId6\"/><w:pgSz w:w=\"12240\" w:h=\"15840\"/></w:sectPr>",
            ),
        );
        assert_eq!(pgsz.len(), 1);
        assert_eq!(
            pgsz[0],
            Some(PageGeometry {
                width_pt: 612.0,
                height_pt: 792.0,
                rotation: 0,
                crop: None,
            })
        );
        // margins-only: page_size is None in the RAW layer → None entry.
        // Under to_ir() this same sectPr yields a fabricated Letter — the
        // hole the raw read closes.
        let margins_only = geometry_named(
            "m.docx",
            &docx_with_body(
                "<w:p><w:r><w:t>body</w:t></w:r></w:p>\
                 <w:sectPr><w:pgMar w:top=\"1440\" w:bottom=\"1440\" \
                 w:left=\"1800\" w:right=\"1800\"/></w:sectPr>",
            ),
        );
        assert_eq!(margins_only, vec![None]);
        // no sectPr at all: one section by construction, no declared size
        let no_sectpr = geometry_named("n.docx", &write_docx_bytes());
        assert_eq!(no_sectpr, vec![None]);
        // `<w:pgSz>` without attributes: the RAW layer's Letter default —
        // the named engine limitation, pinned as the documented behavior.
        let bare = geometry_named(
            "b.docx",
            &docx_with_body("<w:p><w:r><w:t>body</w:t></w:r></w:p><w:sectPr><w:pgSz/></w:sectPr>"),
        );
        assert_eq!(
            bare[0],
            Some(PageGeometry {
                width_pt: 612.0,
                height_pt: 792.0,
                rotation: 0,
                crop: None,
            })
        );
        // a mid-document section break + a body-level sectPr: two IR
        // sections, each aligned to its raw section IN ORDER (the
        // converter's own `sections.get(idx)` rule) — different declared
        // sizes per section, both Some.
        let two_sections = geometry_named(
            "two.docx",
            &docx_with_body(
                "<w:p><w:pPr><w:sectPr><w:pgSz w:w=\"11906\" w:h=\"16838\"/></w:sectPr>\
                 </w:pPr><w:r><w:t>first</w:t></w:r></w:p>\
                 <w:p><w:r><w:t>second</w:t></w:r></w:p>\
                 <w:sectPr><w:headerReference w:type=\"default\" r:id=\"rId6\"/><w:pgSz w:w=\"12240\" w:h=\"15840\"/></w:sectPr>",
            ),
        );
        assert_eq!(two_sections.len(), 2);
        assert_eq!(two_sections[0].unwrap().width_pt, 595.3); // 11906 twips ÷ 20
        assert_eq!(two_sections[1].unwrap().width_pt, 612.0); // 12240 twips ÷ 20
        let beyond = geometry_named(
            "beyond.docx",
            &docx_with_body(
                "<w:p><w:pPr><w:sectPr><w:pgSz w:w=\"11906\" w:h=\"16838\"/></w:sectPr>\
                 </w:pPr><w:r><w:t>first</w:t></w:r></w:p>\
                 <w:p><w:r><w:t>second</w:t></w:r></w:p>\
                 <w:p><w:r><w:t>third</w:t></w:r></w:p>",
            ),
        );
        assert_eq!(beyond.len(), 2);
        assert!(beyond[0].is_some());
        assert_eq!(
            beyond[1], None,
            "the trailing window beyond the raw sections"
        );
    }

    /// §9 item 8/9: the PPTX leg reads the RAW slide size (EMU ÷ 12700.0
    /// in float — 12192000×6858000 EMU is exactly 960×540 points), one
    /// entry per slide; undeclared → None everywhere; a 0-slide deck →
    /// `[]` (zero units); and the XLSX leg is all-None at the RAW
    /// worksheet count — a pageSetup-bearing sheet included, hidden
    /// sheets included.
    #[test]
    fn the_pptx_geometry_is_the_raw_emu_layer_and_xlsx_is_all_none() {
        let sized = geometry_named(
            "deck.pptx",
            &pptx_with(
                Some("<p:sldSz cx=\"12192000\" cy=\"6858000\"/>"),
                &["<p:sp/>", "<p:sp/>"],
            ),
        );
        assert_eq!(sized.len(), 2);
        for entry in &sized {
            assert_eq!(
                entry,
                &Some(PageGeometry {
                    width_pt: 960.0,
                    height_pt: 540.0,
                    rotation: 0,
                    crop: None,
                })
            );
        }
        // undeclared slide size: None everywhere, slides still counted
        let undeclared = geometry_named(
            "plain.pptx",
            &pptx_with(None, &["<p:sp/>", "<p:sp/>", "<p:sp/>"]),
        );
        assert_eq!(undeclared, vec![None, None, None]);
        // the 0-slide deck: [] is a count (zero units), reachable here
        let empty_deck = geometry_named("none.pptx", &pptx_with(None, &[]));
        assert_eq!(empty_deck, Vec::<Option<PageGeometry>>::new());
        // XLSX: a pageSetup-bearing sheet is None anyway (the doctrine
        // pin: the IR would fabricate Letter for it); the count is the
        // raw worksheet count, hidden sheets included.
        let paged = geometry_named(
            "book.xlsx",
            &xlsx_with(
                &[("Samples", None)],
                "<pageMargins left=\"0.7\" right=\"0.7\"/><pageSetup paperSize=\"1\"/>",
            ),
        );
        assert_eq!(paged, vec![None]);
        let hidden = geometry_named(
            "two.xlsx",
            &xlsx_with(&[("Samples", None), ("Secret", Some("hidden"))], ""),
        );
        assert_eq!(hidden, vec![None, None]);
    }

    /// §9 item 9/10: the single-flow row is exactly `[None]` — by
    /// content-marker sniff and by explicit name (the same answer), on
    /// every backend; and the length law over a mini-corpus of every
    /// family: PDF pages, docx sections, pptx slides, xlsx worksheets,
    /// legacy IR sections, single-flow one.
    #[test]
    fn single_flow_formats_are_one_none_entry_and_lengths_follow_the_matrix() {
        for (name, bytes) in [
            ("a.html", POLLUTED_HTML.as_bytes().to_vec()),
            ("b.csv", b"unit,status\nT-101,healthy\n".to_vec()),
            ("c.rtf", b"{\\rtf1\\ansi\\pard x\\par\n}\n".to_vec()),
        ] {
            for backend in [Backend::Auto, Backend::Oxide, Backend::Anydoc] {
                let entries = page_geometry(
                    bytes.clone(),
                    Some(name),
                    None,
                    backend,
                    &ConvertOptions::default(),
                )
                .unwrap();
                assert_eq!(entries, vec![None], "{name} x {backend:?}");
            }
            // the explicit name is the same answer as the sniff
            let named = page_geometry(
                bytes.clone(),
                None,
                Some(name.split('.').next_back().unwrap()),
                Backend::Auto,
                &ConvertOptions::default(),
            )
            .unwrap();
            assert_eq!(named, vec![None], "{name} by format=");
        }
        // the length law, every family (the mini-corpus):
        let pdf = geometry_named("a.pdf", &two_page_pdf("one", "two"));
        assert_eq!(pdf.len(), 2);
        assert!(pdf.iter().all(|entry| entry.is_some()));
        let docx = geometry_named(
            "a.docx",
            &docx_with_body("<w:p><w:r><w:t>x</w:t></w:r></w:p>"),
        );
        assert_eq!(docx.len(), 1);
        let pptx = geometry_named("a.pptx", &pptx_with(None, &["<p:sp/>"]));
        assert_eq!(pptx.len(), 1);
        let xlsx = geometry_named("a.xlsx", &xlsx_with(&[("S", None), ("T", None)], ""));
        assert_eq!(xlsx.len(), 2);
        // the legacy trio, end to end (the committed soffice-generated
        // fixtures under tests/legacy_corpus/ — OLE bytes no in-test
        // builder can fake): all-None at the IR section count — doc: the
        // one document section; xls: one Section per worksheet (three);
        // ppt: one per slide (three). §4's legacy row, pinned on real
        // bytes.
        let doc_fixture = page_geometry(
            include_bytes!("../tests/legacy_corpus/t.doc").to_vec(),
            Some("t.doc"),
            None,
            Backend::Auto,
            &ConvertOptions::default(),
        )
        .unwrap();
        assert_eq!(doc_fixture, vec![None]);
        let xls_fixture = page_geometry(
            include_bytes!("../tests/legacy_corpus/book3.xls").to_vec(),
            Some("book3.xls"),
            None,
            Backend::Auto,
            &ConvertOptions::default(),
        )
        .unwrap();
        assert_eq!(xls_fixture, vec![None, None, None]);
        let ppt_fixture = page_geometry(
            include_bytes!("../tests/legacy_corpus/deck3.ppt").to_vec(),
            Some("deck3.ppt"),
            None,
            Backend::Auto,
            &ConvertOptions::default(),
        )
        .unwrap();
        assert_eq!(ppt_fixture, vec![None, None, None]);
        // the container decision rides the BYTES, not the name: the same
        // workbook named .xlsx still routes the legacy parser (all-None,
        // three) — and sniff reports the container's own name.
        let by_xlsx_name = page_geometry(
            include_bytes!("../tests/legacy_corpus/book3.xls").to_vec(),
            Some("book3.xlsx"),
            None,
            Backend::Auto,
            &ConvertOptions::default(),
        )
        .unwrap();
        assert_eq!(by_xlsx_name, vec![None, None, None]);
        assert_eq!(
            sniff(include_bytes!("../tests/legacy_corpus/book3.xls")),
            Some("xls")
        );
    }

    /// The refusal order on the geometry spine (§6, pinned): the
    /// capability refusal (format-aware) precedes the password-kind
    /// refusal, which precedes the zero-max_bytes refusal, which precedes
    /// the encrypted door; the pdf lane's own refusals keep their texts.
    #[test]
    fn the_geometry_refusals_precede_each_other_in_the_pinned_order() {
        let docx = docx_with_body("<w:p><w:r><w:t>x</w:t></w:r></w:p>");
        // anydoc on an office format: the office message, byte-pinned
        let err = geometry_err("a.docx", &docx, Backend::Anydoc);
        assert_eq!(
            err.to_string(),
            "backend \"anydoc\" cannot serve page geometry for docx: anydoc has no \
             geometry surface; use backend='auto' or 'oxide' here"
        );
        // the refusal names the CONTAINER the bytes are (resolved_name's
        // one container-aware case): a legacy OLE workbook under anydoc
        // says xls, never the Excel kind's default xlsx spelling.
        let err = geometry_err("a.xls", &ole_with_stream("Workbook"), Backend::Anydoc);
        assert_eq!(
            err.to_string(),
            "backend \"anydoc\" cannot serve page geometry for xls: anydoc has no \
             geometry surface; use backend='auto' or 'oxide' here"
        );
        // capability refusal BEFORE the password-kind refusal: a password
        // on a docx under anydoc surfaces the capability refusal
        let err = page_geometry(
            docx.clone(),
            Some("a.docx"),
            None,
            Backend::Anydoc,
            &ConvertOptions {
                password: Some("torque"),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("cannot serve page geometry"),
            "{err:?}"
        );
        // the password-kind refusal precedes the zero-max_bytes refusal
        let err = page_geometry(
            docx.clone(),
            Some("a.docx"),
            None,
            Backend::Auto,
            &ConvertOptions {
                password: Some("torque"),
                max_bytes: Some(0),
            },
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("password= applies to PDF documents only"),
            "{err:?}"
        );
        // zero max_bytes refused (the binding refuses it under the GIL;
        // the core is self-sufficient about it too)
        let err = page_geometry(
            docx,
            Some("a.docx"),
            None,
            Backend::Auto,
            &ConvertOptions {
                max_bytes: Some(0),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "max_bytes must be positive");
        // anydoc on a pdf: the historical PDF message shape, with this
        // probe's capability named
        let err = geometry_err("a.pdf", &two_page_pdf("one", "two"), Backend::Anydoc);
        assert!(
            err.to_string().starts_with(
                "backend \"anydoc\" cannot serve the declared page-box geometry \
                 (no per-page box surface): anydoc's PDF surface is whole-document \
                 conversion only"
            ),
            "{err:?}"
        );
    }

    /// §9 item 11, the office leg's input ceiling: an explicit budget
    /// binds (the same refusal the conversion lane raises, naming the
    /// office_oxide lane), `None` keeps the 32 MiB default, and the
    /// garbage container is a typed parse error — never None/[].
    #[test]
    fn the_office_geometry_leg_carries_the_input_ceiling_and_typed_parse_errors() {
        let docx = docx_with_body("<w:p><w:r><w:t>x</w:t></w:r></w:p>");
        let err = page_geometry(
            docx.clone(),
            Some("a.docx"),
            None,
            Backend::Auto,
            &ConvertOptions {
                max_bytes: Some(64),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("office_oxide engine lane's input ceiling"),
            "{err:?}"
        );
        // under the explicit ceiling: the geometry comes through
        let entries = page_geometry(
            docx.clone(),
            Some("a.docx"),
            None,
            Backend::Auto,
            &ConvertOptions {
                max_bytes: Some(1024 * 1024),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(entries, vec![None]);
        // garbage named docx: the container parse's typed Convert error
        let err = geometry_err(
            "junk.docx",
            b"PK\x03\x04 not a package at all",
            Backend::Auto,
        );
        assert!(matches!(err, DocumentError::Convert(_)), "{err:?}");
        // the encrypted-OOXML door (§7): an OLE/CFB container standing
        // where a docx belongs — the encrypted-package shape's container —
        // is the typed "encrypted" refusal, never None/[], never a
        // password prompt.
        let err = geometry_err(
            "locked.docx",
            &ole_with_stream("EncryptedPackage"),
            Backend::Auto,
        );
        assert!(err.to_string().contains("encrypted"), "{err:?}");
        // an OLE container standing where an XLSX is named: the container
        // decision routes it to the LEGACY parser (the same check the
        // conversion lane's engine choice makes), whose zero-worksheet
        // answer for these junk-stream bytes is the zero-unit [] — never
        // the xlsx raw-layer error the pre-fix leg raised for every OLE
        // workbook (the red-team blocker: a genuine .xls lost its parse
        // to a raw layer it never had). The real workbook is pinned
        // all-None at three by the committed fixture, below.
        let entries = page_geometry(
            ole_with_stream("Workbook"),
            Some("locked.xlsx"),
            None,
            Backend::Auto,
            &ConvertOptions::default(),
        )
        .unwrap();
        assert_eq!(entries, Vec::<Option<PageGeometry>>::new());
        // garbage named pdf: the pdf lane's own typed refusal
        let err = geometry_err("junk.pdf", b"garbage", Backend::Auto);
        assert!(matches!(err, DocumentError::Convert(_)), "{err:?}");
        // the no-geometry row never parses: undetectable bytes named csv
        // still answer [None] (validation = format resolution)
        let entries = geometry_named("x.csv", b"single line only");
        assert_eq!(entries, vec![None]);
    }

    // --- the page_layout spine (spec §8 commit 4's routing half) -----------

    /// The spine's routes on the layout surface: the pdf leg carries the
    /// real content (spans/lines/images in render space, pages= honored);
    /// the office rows serve their geometry with EMPTY layout lists (no
    /// office engine exposes text/image boxes — PDF-only for layout); the
    /// no-geometry row is the single geometry-None entry; and the refusal
    /// order is page_geometry's (the capability refusal is format-aware
    /// and names the same messages).
    #[test]
    fn the_layout_spine_routes_and_refuses_like_the_geometry_spine() {
        let layout = |name: &str, bytes: &[u8], pages: Option<Vec<usize>>| {
            page_layout(
                bytes.to_vec(),
                Some(name),
                None,
                Backend::Auto,
                pages,
                &ConvertOptions::default(),
            )
        };
        // pdf: one entry per page with real content; page_index carried.
        let entries = layout("a.pdf", &two_page_pdf("alpha", "beta"), None).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries.iter().map(|e| e.page_index).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(entries[0].geometry.is_some());
        assert_eq!(entries[0].spans[0].text, "alpha");
        assert!(!entries[0].spans[0].is_bold);
        assert!(!entries[0].lines.is_empty() && entries[0].images.is_empty());
        // the format marker: every entry of one call carries the resolved
        // format's vocabulary name — on the pdf lane always "pdf"
        assert!(
            entries.iter().all(|e| e.format == "pdf"),
            "the pdf lane's entries carry format=pdf"
        );
        // pages= subset: parallel to the normalized selection
        let entries = layout("a.pdf", &two_page_pdf("alpha", "beta"), Some(vec![1])).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].page_index, 1);
        assert_eq!(entries[0].spans[0].text, "beta");
        // the byte-pinned pages= refusals (markdown_pages' strings)
        let err = layout("a.pdf", &two_page_pdf("alpha", "beta"), Some(vec![])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "pages= selected no pages: pass None for the whole document"
        );
        let err = layout("a.pdf", &two_page_pdf("alpha", "beta"), Some(vec![2])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "page index 2 is out of range for a 2-page document (pages= is 0-based)"
        );
        // pages= is PDF-only (F1): a non-None selection on every non-PDF
        // row — office and no-geometry alike — is convert()'s own typed
        // refusal, byte-identical wording, never the silent
        // whole-document answer the phase-2 diff shipped (probed:
        // pages=[0] and pages=[7] on a two-section docx both returned
        // BOTH sections).
        let html = b"<!DOCTYPE html>\n<html><body>hi</body></html>".to_vec();
        for (name, bytes, resolved) in [
            (
                "a.docx",
                docx_with_body("<w:p><w:r><w:t>x</w:t></w:r></w:p>"),
                "docx",
            ),
            ("a.pptx", pptx_with(Some(""), &[""]), "pptx"),
            ("a.xlsx", xlsx_with(&[("Sheet1", None)], ""), "xlsx"),
            ("x.html", html, "html"),
        ] {
            let err = layout(name, &bytes, Some(vec![0])).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!("pages= applies to PDF documents only, not {resolved:?}"),
                "{name}: pages= must be refused off the PDF lane"
            );
        }
        // office: geometry + empty layout lists (PDF-only for layout), the
        // format marker naming the resolved kind
        let docx = docx_with_body(
            "<w:p><w:r><w:t>body</w:t></w:r></w:p>\
             <w:sectPr><w:headerReference w:type=\"default\" r:id=\"rId6\"/><w:pgSz w:w=\"12240\" w:h=\"15840\"/></w:sectPr>",
        );
        let entries = layout("a.docx", &docx, None).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].page_index, 0);
        assert_eq!(entries[0].format, "docx", "the office arm's format marker");
        assert_eq!(
            entries[0]
                .geometry
                .map(|g| (g.width_pt, g.height_pt, g.rotation, g.crop)),
            Some((612.0, 792.0, 0, None))
        );
        assert!(
            entries[0].spans.is_empty()
                && entries[0].lines.is_empty()
                && entries[0].images.is_empty()
        );
        // the no-geometry row: the single geometry-None entry, format named
        let entries = layout("x.csv", b"single line only", None).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].geometry.is_none());
        assert!(entries[0].spans.is_empty());
        assert_eq!(entries[0].format, "csv");
        // the legacy trio on the committed OLE fixtures: all-None geometry,
        // IR section counts, and the format marker naming the RESOLVED
        // kind — the Excel kind's one container-aware case included (the
        // OLE workbook reports "xls", never the kind's default "xlsx"
        // spelling — the same name to_markdown reports).
        let doc_fixture = page_layout(
            include_bytes!("../tests/legacy_corpus/t.doc").to_vec(),
            Some("t.doc"),
            None,
            Backend::Auto,
            None,
            &ConvertOptions::default(),
        )
        .unwrap();
        assert_eq!(doc_fixture.len(), 1);
        assert_eq!(doc_fixture[0].format, "doc");
        assert!(doc_fixture[0].geometry.is_none());
        let xls_fixture = page_layout(
            include_bytes!("../tests/legacy_corpus/book3.xls").to_vec(),
            Some("book3.xls"),
            None,
            Backend::Auto,
            None,
            &ConvertOptions::default(),
        )
        .unwrap();
        assert_eq!(xls_fixture.len(), 3);
        assert!(
            xls_fixture.iter().all(|e| e.format == "xls"),
            "{xls_fixture:?}"
        );
        let ppt_fixture = page_layout(
            include_bytes!("../tests/legacy_corpus/deck3.ppt").to_vec(),
            Some("deck3.ppt"),
            None,
            Backend::Auto,
            None,
            &ConvertOptions::default(),
        )
        .unwrap();
        assert_eq!(ppt_fixture.len(), 3);
        assert!(ppt_fixture.iter().all(|e| e.format == "ppt"));
        // refusal order, page_geometry's: anydoc capability refusal on the
        // office row — THE TABLE'S ONE REFUSAL, byte-identical on both
        // surfaces (the geometry capability table applies to page_layout
        // identically, refusal text included)
        let err = page_layout(
            docx.clone(),
            Some("a.docx"),
            None,
            Backend::Anydoc,
            None,
            &ConvertOptions::default(),
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "backend \"anydoc\" cannot serve page geometry for docx: anydoc has no \
             geometry surface; use backend='auto' or 'oxide' here"
        );
        // password-kind refusal precedes zero-max_bytes
        let err = page_layout(
            docx,
            Some("a.docx"),
            None,
            Backend::Auto,
            None,
            &ConvertOptions {
                password: Some("torque"),
                max_bytes: Some(0),
            },
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("password= applies to PDF documents only"),
            "{err:?}"
        );
        // the pdf lane's malformed document is the typed whole-call
        // refusal (the encrypted DOOR is pdf_impl::open's, shared by every
        // probe, pinned on the geometry probe in pdf_impl's tests and on
        // page_layout at the binding layer's encrypted-entries tuple)
        let err = page_layout(
            b"%PDF-1.4 garbage".to_vec(),
            Some("a.pdf"),
            None,
            Backend::Auto,
            None,
            &ConvertOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("cross-reference"), "{err:?}");
    }

    #[test]
    fn adversarial_containers_sniff_and_fail_honestly() {
        // A truncated header is still a PDF header: sniff says pdf (the
        // marker is honest about what the bytes open with), and the
        // conversion then fails on the real reason downstream (pdf_oxide's
        // parse), not on a misdetected format.
        assert_eq!(sniff(b"%PDF-1.7"), Some("pdf"));
        let err = convert_err("truncated.pdf", b"%PDF-1.7");
        assert!(matches!(err, DocumentError::Convert(_)), "got {err:?}");
        // a zip that is no document at all: no marker says anything...
        let plain = zip_of(&[("readme.txt", b"hello\n"), ("data/file.bin", b"\x00\x01")]);
        assert_eq!(sniff(&plain), None);
        // ...and named .docx it fails naming the missing part, not a lie
        match convert_err("plain.docx", &plain) {
            DocumentError::Convert(what) => assert!(
                what.contains("word/document.xml"),
                "should name the missing part, got {what:?}"
            ),
            other => panic!("plain zip named .docx surfaced {other:?}"),
        }
        // an OLE container that is no office document (the .msg shape:
        // streams no office spec mandates) names nothing either
        let msg_shaped = ole_with_stream("PropertiesOfStorage");
        assert_eq!(sniff(&msg_shaped), None);
        let err = convert_err("message", &msg_shaped);
        assert!(
            matches!(err, DocumentError::UnknownFormat(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn an_ole_workbook_sniffs_as_its_container_true_name() {
        // sniff reports the same name a conversion would (resolved_name's
        // container-aware Excel case): a Workbook stream inside an OLE
        // container is legacy xls, a ZIP-based one is the xlsx family:
        // sniff and to_markdown cannot disagree about what the bytes are.
        let workbook = ole_with_stream("Workbook");
        assert_eq!(sniff(&workbook), Some("xls"));
        let zip_workbook = zip_of(&[("xl/workbook.xml", b"<workbook/>")]);
        assert_eq!(sniff(&zip_workbook), Some("xlsx"));
        // resolution works even when the parse of junk BIFF bytes cannot:
        // named .xlsx, the OLE container still picks the Excel kind, and
        // the failure is anydoc's honest "not a readable workbook": never
        // a misdetected format (the mislabeled-extension doctrine).
        let err = convert_err("workbook.xlsx", &workbook);
        assert!(matches!(err, DocumentError::Convert(_)), "got {err:?}");
    }

    #[test]
    fn read_bounded_exactly_at_ceiling_is_accepted() {
        use std::io::Cursor;
        let (bytes, over) =
            super::read_bounded_into(Cursor::new(vec![7u8; 200]), Vec::new(), 200).unwrap();
        assert_eq!(bytes.len(), 200);
        assert!(!over);
    }

    #[test]
    fn read_bounded_one_past_ceiling_is_flagged() {
        use std::io::Cursor;
        let (_, over) =
            super::read_bounded_into(Cursor::new(vec![7u8; 201]), Vec::new(), 200).unwrap();
        assert!(over);
    }

    #[test]
    fn read_bounded_infinite_stream_is_bounded_without_doubling() {
        use std::io::Read as _;
        // io::repeat is infinite; the read must stop and must not over-allocate.
        let r: std::io::Repeat = std::io::repeat(0u8);
        let (bytes, over) =
            super::read_bounded_into(r.take(u64::MAX), Vec::new(), 1_000_000).unwrap();
        assert!(over);
        // capacity stays near the ceiling, proving no geometric doubling.
        assert!(
            bytes.capacity() <= 1_000_000 + 1 + 64 * 1024,
            "capacity {} overshot",
            bytes.capacity()
        );
    }

    #[test]
    fn read_bounded_empty_reader_returns_empty() {
        use std::io::Cursor;
        let (bytes, over) =
            super::read_bounded_into(Cursor::new(Vec::<u8>::new()), Vec::new(), 200).unwrap();
        assert!(bytes.is_empty());
        assert!(!over);
    }

    #[test]
    fn read_bounded_continues_from_a_prefix() {
        use std::io::Cursor;
        // buf already holds 3 bytes (a "prefix"); reader adds 2 more; ceiling 10.
        let (bytes, over) =
            super::read_bounded_into(Cursor::new(vec![9u8; 2]), vec![1u8, 2, 3], 10).unwrap();
        assert_eq!(bytes, vec![1, 2, 3, 9, 9]);
        assert!(!over);
    }

    const PROVISIONAL_FALLBACK: usize = 512 * 1024 * 1024;

    #[test]
    fn provisional_csv_prefix_picks_the_metered_ceiling() {
        // Two comma lines: the CSV heuristic routes to the anydoc lane.
        let prefix = b"unit,status\na,ok\nb,ok\n";
        let c = provisional_read_ceiling(
            prefix,
            Some("data.csv"),
            None,
            Backend::Auto,
            PROVISIONAL_FALLBACK,
        );
        assert_eq!(c, DEFAULT_ANYDOC_INPUT_LIMIT);
    }

    #[test]
    fn provisional_pdf_prefix_stays_unmetered() {
        let c = provisional_read_ceiling(
            b"%PDF-1.4\n...",
            Some("x.pdf"),
            None,
            Backend::Auto,
            PROVISIONAL_FALLBACK,
        );
        assert_eq!(c, PROVISIONAL_FALLBACK);
    }

    #[test]
    fn provisional_html_prefix_meters() {
        // The HTML lane amplifies at a measured ~23x input (a 48 MiB doctype
        // HTML peaked at 1118 MiB): it meters like anydoc/office_oxide, and
        // the fallback is the pdf lane's alone.
        let c = provisional_read_ceiling(
            b"<!doctype html><html><body>\n<h1>t</h1>\n",
            Some("x.html"),
            None,
            Backend::Auto,
            PROVISIONAL_FALLBACK,
        );
        assert_eq!(c, DEFAULT_ANYDOC_INPUT_LIMIT);
    }

    // --- the geometry surface's own read metering and pre-read probe
    // --- (the conversion table's ceiling metered lanes this surface
    // --- never runs; the §6 law keys the geometry table) ----------------

    #[test]
    fn the_geometry_ceiling_meters_the_office_row_only() {
        // office row: the amplifying lane — the 32 MiB default. Both
        // resolution paths: an explicit format= (the bytes never
        // consulted) and the name-hint fallback (the bytes carry no
        // positive marker).
        assert_eq!(
            provisional_geometry_read_ceiling(
                b"\x00\x01",
                None,
                Some("docx"),
                Backend::Auto,
                PROVISIONAL_FALLBACK,
            ),
            DEFAULT_ANYDOC_INPUT_LIMIT
        );
        assert_eq!(
            provisional_geometry_read_ceiling(
                b"\x00\x01",
                Some("a.docx"),
                None,
                Backend::Auto,
                PROVISIONAL_FALLBACK,
            ),
            DEFAULT_ANYDOC_INPUT_LIMIT
        );
        // the pdf row: no default ceiling (pdf_oxide's own limits govern)
        assert_eq!(
            provisional_geometry_read_ceiling(
                b"%PDF-1.4\n...",
                Some("x.pdf"),
                None,
                Backend::Auto,
                PROVISIONAL_FALLBACK,
            ),
            PROVISIONAL_FALLBACK
        );
        // the no-geometry row: no engine runs, no default ceiling — the
        // parity case the conversion table's ceiling broke (a 40 MiB html
        // answered [None] on data= and refused on path=)
        for (bytes, hint) in [
            (&b"<!doctype html><html><body>\n<h1>t</h1>\n"[..], "x.html"),
            (b"unit,status\na,ok\n", "x.csv"),
            (b"{\\rtf1\\ansi x}", "x.rtf"),
        ] {
            assert_eq!(
                provisional_geometry_read_ceiling(
                    bytes,
                    Some(hint),
                    None,
                    Backend::Auto,
                    PROVISIONAL_FALLBACK,
                ),
                PROVISIONAL_FALLBACK,
                "{hint}"
            );
        }
        // unresolvable: meters — the safe direction (the office leg's
        // post-read default still binds if the full bytes resolve there)
        assert_eq!(
            provisional_geometry_read_ceiling(
                b"\x00\x01\x02",
                Some("blob.bin"),
                None,
                Backend::Auto,
                PROVISIONAL_FALLBACK,
            ),
            DEFAULT_ANYDOC_INPUT_LIMIT
        );
        // and the office row meters under anydoc too (the refusal would
        // fire pre-read at the binding; the ceiling stays consistent)
        assert_eq!(
            provisional_geometry_read_ceiling(
                b"PK\x03\x04",
                Some("a.xlsx"),
                None,
                Backend::Anydoc,
                PROVISIONAL_FALLBACK,
            ),
            DEFAULT_ANYDOC_INPUT_LIMIT
        );
    }

    #[test]
    fn the_geometry_probe_refuses_the_served_rows_and_defers_the_unresolvable() {
        // resolved to a served row under anydoc: the refusal, naming the
        // container the bytes are
        let err = probe_geometry_capability(b"%PDF-1.4\n", Some("x.pdf"), None, Backend::Anydoc)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot serve the declared page-box geometry"),
            "{err:?}"
        );
        let err = probe_geometry_capability(b"PK\x03\x04", Some("a.docx"), None, Backend::Anydoc)
            .unwrap_err();
        assert!(err.to_string().contains("for docx"), "{err:?}");
        // resolved to the no-geometry row under anydoc: no refusal
        assert_eq!(
            probe_geometry_capability(
                b"<!doctype html><html><body>",
                Some("x.html"),
                None,
                Backend::Anydoc,
            )
            .unwrap(),
            Some(Kind::Html)
        );
        // auto/oxide refuse nothing on any row
        assert_eq!(
            probe_geometry_capability(b"%PDF-1.4\n", Some("x.pdf"), None, Backend::Auto).unwrap(),
            Some(Kind::Pdf)
        );
        // unresolvable: Ok(None) — the refusal defers to the core's
        // authoritative resolve (§6's resolve-first clause)
        assert_eq!(
            probe_geometry_capability(b"garbage", None, None, Backend::Anydoc).unwrap(),
            None
        );
        // an explicit format= decides without the bytes at all
        let err =
            probe_geometry_capability(b"garbage", None, Some("pdf"), Backend::Anydoc).unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot serve the declared page-box geometry"),
            "{err:?}"
        );
    }

    #[test]
    fn the_no_geometry_door_predicate_is_the_tables_row_not_a_second_spelling() {
        for name in ["html", "csv", "tsv", "rtf", "epub", "odt", "ods", "odp"] {
            assert!(is_no_geometry_format(name), "{name}");
        }
        for name in ["pdf", "docx", "xlsx", "pptx", "doc", "xls", "ppt"] {
            assert!(!is_no_geometry_format(name), "{name}");
        }
        // an unknown name is simply not on the row (resolve rejects it
        // later, its own refusal)
        assert!(!is_no_geometry_format("flib"));
    }

    #[test]
    fn provisional_html_prefix_meters_by_content_not_name() {
        // Content beats name: an html-named csv is metered; a csv-named html
        // is metered too — the lane guess reads the prefix's markers.
        let c = provisional_read_ceiling(
            b"<!doctype html><html><body>\n<h1>t</h1>\n",
            Some("x.csv"),
            None,
            Backend::Auto,
            PROVISIONAL_FALLBACK,
        );
        assert_eq!(c, DEFAULT_ANYDOC_INPUT_LIMIT);
    }

    #[test]
    fn provisional_pdf_named_csv_is_metered_by_content() {
        // Content beats name: a .pdf-named CSV must still get the metered ceiling.
        let prefix = b"unit,status\na,ok\nb,ok\n";
        let c = provisional_read_ceiling(
            prefix,
            Some("x.pdf"),
            None,
            Backend::Auto,
            PROVISIONAL_FALLBACK,
        );
        assert_eq!(c, DEFAULT_ANYDOC_INPUT_LIMIT);
    }

    #[test]
    fn provisional_unresolvable_prefix_errs_toward_the_metered_ceiling() {
        // A ZIP local-file header with no central directory cannot be resolved
        // from the prefix, and neither a metered nor an unmetered lane can be
        // told: the read errs SAFE at the metered ceiling, never the fallback.
        // Content that stays unresolvable is refused by resolve() anyway; an
        // extensionless metered file behind a content-blind leader must not
        // buy the fallback (a 64 KiB leader would else defeat read metering).
        let c = provisional_read_ceiling(
            b"PK\x03\x04\x14\x00",
            None,
            None,
            Backend::Auto,
            PROVISIONAL_FALLBACK,
        );
        assert_eq!(c, DEFAULT_ANYDOC_INPUT_LIMIT);
    }

    #[test]
    fn provisional_positively_unmetered_prefix_keeps_the_fallback() {
        // The fallback is reserved for prefixes that POSITIVELY resolve to an
        // unmetered lane (pdf) — here by name hint over pdf content.
        let c = provisional_read_ceiling(
            b"%PDF-1.4\n",
            Some("x.pdf"),
            None,
            Backend::Auto,
            PROVISIONAL_FALLBACK,
        );
        assert_eq!(c, PROVISIONAL_FALLBACK);
    }

    #[test]
    fn provisional_partial_last_line_does_not_skew_csv() {
        // A prefix cut mid-line must not miscount; only complete lines are fed.
        let prefix = b"a,b\nc,d\ne,"; // last line incomplete
        let c = provisional_read_ceiling(
            prefix,
            Some("f.csv"),
            None,
            Backend::Auto,
            PROVISIONAL_FALLBACK,
        );
        assert_eq!(c, DEFAULT_ANYDOC_INPUT_LIMIT); // still CSV from the two complete lines
    }

    #[test]
    fn read_bounded_usize_max_ceiling_does_not_overflow() {
        use std::io::Cursor;
        // The pub reader must be sound at the extreme ceiling: `ceiling + 1`
        // would overflow, but `saturating_add(1)` keeps `want` well-formed.
        let (bytes, over) =
            super::read_bounded_into(Cursor::new(vec![0u8; 10]), Vec::new(), usize::MAX).unwrap();
        assert_eq!(bytes.len(), 10);
        assert!(!over);
    }

    #[test]
    fn provisional_long_row_csv_is_metered_from_untrimmed_prefix() {
        // A CSV whose second row is wider than the prefix, with no trailing
        // newline: trimming to complete lines leaves one line (not CSV), so the
        // metering must come from the UNTRIMMED prefix, which sees both rows.
        let mut prefix = b"unit,status\na,".to_vec();
        prefix.extend(std::iter::repeat_n(b'x', 80 * 1024));
        let c = provisional_read_ceiling(&prefix, None, None, Backend::Auto, PROVISIONAL_FALLBACK);
        assert_eq!(c, DEFAULT_ANYDOC_INPUT_LIMIT);
    }

    // --- the oxide lane's decompression audit: the container shapes it
    // must pass and refuse, at fixture scale. The gigabyte bombs are
    // pinned end-to-end through the wheel in
    // tests/test_documents_resource_bounds.py; these pin the parse and
    // refusal logic itself on hand-built containers.

    /// A minimal OPC-shaped zip: local headers, data, central directory,
    /// EOCD, every offset and size honest unless `declare` overrides a
    /// part's declared uncompressed size in the directory (the size-lie
    /// shape). A method-8 part carries its content deflate-compressed (the
    /// reader's method 8), anything else stored.
    fn zip_container(parts: &[(&str, u16, &[u8], Option<u32>)]) -> Vec<u8> {
        use std::io::Write as _;

        let mut out = Vec::new();
        let mut records: Vec<(u32, String, u16, u32, u32)> = Vec::new();
        for (name, method, content, _declare) in parts {
            let (data, csize, actual) = match *method {
                8 => {
                    let mut encoder = flate2::write::DeflateEncoder::new(
                        Vec::new(),
                        flate2::Compression::default(),
                    );
                    encoder.write_all(content).unwrap();
                    let data = encoder.finish().unwrap();
                    let len = data.len() as u32;
                    (data, len, len)
                }
                _ => {
                    let len = content.len() as u32;
                    (content.to_vec(), len, len)
                }
            };
            records.push((out.len() as u32, name.to_string(), *method, csize, actual));
            out.extend_from_slice(&ZIP_LOCAL_HEADER_SIG);
            out.extend_from_slice(&20u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&method.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&csize.to_le_bytes());
            out.extend_from_slice(&actual.to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&data);
        }
        let cd_offset = out.len() as u32;
        for (offset, name, method, csize, actual) in &records {
            let declared = declare_of(parts, name).unwrap_or(*actual);
            out.extend_from_slice(&ZIP_CENTRAL_SIG);
            out.extend_from_slice(&20u16.to_le_bytes());
            out.extend_from_slice(&20u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&method.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&csize.to_le_bytes());
            out.extend_from_slice(&declared.to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
        }
        let cd_size = out.len() as u32 - cd_offset;
        out.extend_from_slice(&ZIP_EOCD_SIG);
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(records.len() as u16).to_le_bytes());
        out.extend_from_slice(&(records.len() as u16).to_le_bytes());
        out.extend_from_slice(&cd_size.to_le_bytes());
        out.extend_from_slice(&cd_offset.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    /// The `declare` override of the named part (the builder's own table,
    /// keyed by name so the central-directory loop above can stay a flat
    /// walk).
    fn declare_of(parts: &[(&str, u16, &[u8], Option<u32>)], name: &str) -> Option<u32> {
        parts
            .iter()
            .find(|(part, ..)| *part == name)
            .and_then(|(.., declare)| *declare)
    }

    #[test]
    fn oxide_audit_passes_a_legit_container() {
        // A docx-shaped container at fixture scale: stored meta parts and
        // a deflated document part, every size honest. Both audit stages
        // must clear it (the Python corpus pins the end-to-end convert on
        // the engines' own bytes).
        let mut document = b"<w:document><w:body><w:p><w:r><w:t>Quarterly Review</w:t>".to_vec();
        document.extend(std::iter::repeat_n(b'x', 64 * 1024));
        document.extend_from_slice(b"</w:t></w:r></w:p></w:body></w:document>");
        let container = zip_container(&[
            ("[Content_Types].xml", 0, b"<Types/>".as_slice(), None),
            ("_rels/.rels", 0, b"<Relationships/>".as_slice(), None),
            ("word/document.xml", 8, &document, None),
        ]);
        assert!(audit_oxide_container(&container).is_ok());
    }

    #[test]
    fn oxide_audit_refuses_a_multi_part_bomb_on_declared_sizes() {
        // Two parts declaring 300 MiB each: every one far under
        // office_oxide's 512 MiB per-part cap (that cap cannot fire), so
        // only the total across parts catches the bomb. The declared pass
        // refuses without inflating a byte, naming the part that breaches,
        // the sum, and the ceiling's own number.
        let tiny = b"<w:document/>";
        let container = zip_container(&[
            ("word/document.xml", 0, tiny, Some(300 * 1024 * 1024)),
            ("word/header1.xml", 0, tiny, Some(300 * 1024 * 1024)),
        ]);
        let Err(DocumentError::Convert(what)) = audit_oxide_container(&container) else {
            panic!("the multi-part bomb must be refused");
        };
        assert!(what.contains("decompression limit exceeded"), "{what}");
        assert!(what.contains("536870912"), "{what}");
        assert!(what.contains("word/header1.xml"), "{what}");
    }

    #[test]
    fn oxide_audit_counts_the_inflated_bytes_exactly() {
        // The actual pass's number is the inflated count (the reader's own
        // per-part take discipline, mirrored): a deflated part counts its
        // content length, a stored one its span.
        let mut content = Vec::new();
        for _ in 0..6_000 {
            content.extend_from_slice(b"<w:t>quarterly</w:t>");
        }
        let container = zip_container(&[("word/document.xml", 8, &content, None)]);
        let parts = zip_parts(&container).unwrap().unwrap();
        assert_eq!(parts.len(), 1);
        let count = part_inflated_bytes(&container, &parts[0]).unwrap();
        assert_eq!(count, content.len() as u64);
        let stored = zip_container(&[("word/document.xml", 0, &content, None)]);
        let parts = zip_parts(&stored).unwrap().unwrap();
        assert_eq!(
            part_inflated_bytes(&stored, &parts[0]).unwrap(),
            content.len() as u64
        );
    }

    #[test]
    fn oxide_audit_refuses_a_method_it_cannot_bound() {
        // An entry whose compression method the audit cannot inflate is a
        // reader error today; the audit refuses it up front with the same
        // outcome and the method named, rather than passing an unbounded
        // part on to the engine.
        let container = zip_container(&[("word/document.xml", 99, b"x", None)]);
        let Err(DocumentError::Convert(what)) = audit_oxide_container(&container) else {
            panic!("an unboundable method must be refused");
        };
        assert!(what.contains("compression method 99"), "{what}");
    }

    #[test]
    fn oxide_audit_skips_non_zip_bytes_and_refuses_unresolvable_ones() {
        // No end-of-directory signature anywhere: not a zip, the reader's
        // own error path stands (OLE bytes take the same skip through the
        // container check). An end-of-directory signature that never
        // resolves to a directory is refused, never skipped: the reader's
        // directory search is more tolerant than the audit's parse, and a
        // container the engine can open while the audit cannot is the
        // evasion shape.
        assert!(audit_oxide_container(b"not a zip at all").is_ok());
        let mut empty_directory = b"PK\x05\x06".to_vec();
        empty_directory.extend_from_slice(&0u16.to_le_bytes()); // disk
        empty_directory.extend_from_slice(&0u16.to_le_bytes()); // cd disk
        empty_directory.extend_from_slice(&1u16.to_le_bytes()); // one entry
        empty_directory.extend_from_slice(&1u16.to_le_bytes());
        empty_directory.extend_from_slice(&0u32.to_le_bytes()); // size
        empty_directory.extend_from_slice(&0u32.to_le_bytes()); // offset
        empty_directory.extend_from_slice(&0u16.to_le_bytes()); // comment
        let Err(DocumentError::Convert(what)) = audit_oxide_container(&empty_directory) else {
            panic!("zip signatures that resolve to nothing must be refused");
        };
        assert!(what.contains("zip directory did not parse"), "{what}");
    }

    // --- the document-structure surface (issues #209/#210/#211) ----------

    /// The structure surface's docx fixture: a body carrying every pin the
    /// payload needs — a style-outline heading, a direct `outlineLvl="7"`
    /// (the clamp case), a style-carried `numPr` list paragraph, a
    /// numbered heading (numPr + heading: heading wins, the numPr rides),
    /// a tab/br/hidden-run walk paragraph, an external hyperlink, a
    /// 3×3 table with a gridSpan cell and a vMerge Restart/Continue pair,
    /// an inline picture with a declared extent, two sections, and an
    /// unreadable header part.
    fn structure_docx_fixture() -> Vec<u8> {
        structure_docx_fixture_with_header(b"<w:hdr")
    }

    fn structure_docx_fixture_with_header(header: &[u8]) -> Vec<u8> {
        const W: &str = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";
        const R: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
        const WP: &str = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing";
        const A: &str = "http://schemas.openxmlformats.org/drawingml/2006/main";
        const PIC: &str = "http://schemas.openxmlformats.org/drawingml/2006/picture";
        let document = format!(
            "<?xml version=\"1.0\"?>\
<w:document xmlns:w=\"{W}\" xmlns:r=\"{R}\"><w:body>\
<w:p><w:pPr><w:pStyle w:val=\"Heading1\"/></w:pPr><w:r><w:t>Annual Report</w:t></w:r></w:p>\
<w:p><w:pPr><w:outlineLvl w:val=\"7\"/></w:pPr><w:r><w:t>Deep</w:t></w:r></w:p>\
<w:p><w:pPr><w:pStyle w:val=\"ListPx\"/></w:pPr><w:r><w:t>bulleted through the style</w:t></w:r></w:p>\
<w:p><w:pPr><w:pStyle w:val=\"Heading2\"/><w:numPr><w:ilvl w:val=\"0\"/><w:numId w:val=\"1\"/></w:numPr></w:pPr><w:r><w:t>1. Numbered heading</w:t></w:r></w:p>\
<w:p><w:r><w:t>before</w:t></w:r><w:r><w:br/></w:r><w:r><w:tab/></w:r><w:r><w:t>after</w:t></w:r><w:r><w:rPr><w:vanish/></w:rPr><w:t>hidden text</w:t></w:r></w:p>\
<w:p><w:hyperlink r:id=\"rId1\"><w:r><w:t>field handbook</w:t></w:r></w:hyperlink></w:p>\
<w:p><w:r><w:drawing><wp:inline xmlns:wp=\"{WP}\"><wp:extent cx=\"1905000\" cy=\"1270000\"/><wp:docPr descr=\"torque chart\"/><a:graphic xmlns:a=\"{A}\"><a:graphicData><pic:pic xmlns:pic=\"{PIC}\"><pic:blipFill><a:blip r:embed=\"rId5\"/></pic:blipFill></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>\
<w:tbl><w:tblPr><w:tblCaption w:val=\"Units\"/></w:tblPr><w:tblGrid><w:gridCol w:w=\"2000\"/><w:gridCol w:w=\"2000\"/><w:gridCol w:w=\"2000\"/></w:tblGrid>\
<w:tr><w:tc><w:tcPr><w:gridSpan w:val=\"2\"/></w:tcPr><w:p><w:r><w:t>wide</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>third</w:t></w:r></w:p></w:tc></w:tr>\
<w:tr><w:tc><w:tcPr><w:vMerge w:val=\"restart\"/></w:tcPr><w:p><w:r><w:t>span</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>b1</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>c1</w:t></w:r></w:p></w:tc></w:tr>\
<w:tr><w:tc><w:tcPr><w:vMerge/></w:tcPr><w:p/></w:tc><w:tc><w:p><w:r><w:t>b2</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>c2</w:t></w:r></w:p></w:tc></w:tr></w:tbl>\
<w:p><w:pPr><w:sectPr><w:headerReference w:type=\"default\" r:id=\"rId6\"/><w:pgSz w:w=\"12240\" w:h=\"15840\"/></w:sectPr></w:pPr><w:r><w:t>section one ends</w:t></w:r></w:p>\
<w:p><w:r><w:t>second section body</w:t></w:r></w:p>\
</w:body></w:document>"
        );
        let styles = format!(
            "<?xml version=\"1.0\"?>\
<w:styles xmlns:w=\"{W}\">\
<w:style w:type=\"paragraph\" w:styleId=\"Heading1\"><w:name w:val=\"heading 1\"/><w:pPr><w:outlineLvl w:val=\"0\"/></w:pPr></w:style>\
<w:style w:type=\"paragraph\" w:styleId=\"Heading2\"><w:name w:val=\"heading 2\"/><w:pPr><w:outlineLvl w:val=\"1\"/></w:pPr></w:style>\
<w:style w:type=\"paragraph\" w:styleId=\"ListPx\"><w:name w:val=\"List Paragraph\"/><w:pPr><w:numPr><w:ilvl w:val=\"0\"/><w:numId w:val=\"1\"/></w:numPr></w:pPr></w:style>\
</w:styles>"
        );
        let numbering = format!(
            "<?xml version=\"1.0\"?>\
<w:numbering xmlns:w=\"{W}\">\
<w:abstractNum w:abstractNumId=\"0\"><w:lvl w:ilvl=\"0\"><w:numFmt w:val=\"bullet\"/><w:start w:val=\"1\"/></w:lvl></w:abstractNum>\
<w:num w:numId=\"1\"><w:abstractNumId w:val=\"0\"/></w:num></w:numbering>"
        );
        let rels = format!(
            "<?xml version=\"1.0\"?>\
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\
<Relationship Id=\"rId1\" Type=\"{R}/hyperlink\" Target=\"https://handbook.example.com/torque\" TargetMode=\"External\"/>\
<Relationship Id=\"rId5\" Type=\"{R}/image\" Target=\"media/image5.png\"/>\
<Relationship Id=\"rId6\" Type=\"{R}/header\" Target=\"header1.xml\"/>\
<Relationship Id=\"rId3\" Type=\"{R}/styles\" Target=\"styles.xml\"/>\
<Relationship Id=\"rId4\" Type=\"{R}/numbering\" Target=\"numbering.xml\"/>\
</Relationships>"
        );
        let content_types = "<?xml version=\"1.0\"?>\
<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">\
<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\
<Default Extension=\"xml\" ContentType=\"application/xml\"/>\
<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>\
</Types>"
            .to_string();
        let root_rels = format!(
            "<?xml version=\"1.0\"?>\
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\
<Relationship Id=\"rId1\" Type=\"{R}/officeDocument\" Target=\"word/document.xml\"/>\
</Relationships>"
        );
        zip_of(&[
            ("[Content_Types].xml", content_types.as_bytes()),
            ("_rels/.rels", root_rels.as_bytes()),
            ("word/document.xml", document.as_bytes()),
            ("word/styles.xml", styles.as_bytes()),
            ("word/numbering.xml", numbering.as_bytes()),
            ("word/_rels/document.xml.rels", rels.as_bytes()),
            // the image part the rid points at (the ext rides the target
            // path's extension)
            ("word/media/image5.png", b"png-bytes"),
            ("word/header1.xml", header),
        ])
    }

    /// The structure surface's xlsx fixture: three worksheets (visible,
    /// hidden, veryHidden) + one chartsheet, typed cells (number,
    /// string/shared, boolean, error, t="d" date, styled-empty), a
    /// formula with and one without a cached value, a lying declared
    /// dimension, a merged range, an external + an internal hyperlink,
    /// and a global + a sheet-scoped defined name.
    fn structure_xlsx_fixture() -> Vec<u8> {
        const M: &str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
        const R: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
        let workbook = format!(
            "<?xml version=\"1.0\"?>\
<workbook xmlns=\"{M}\" xmlns:r=\"{R}\"><sheets>\
<sheet name=\"Visible\" sheetId=\"1\" r:id=\"rId1\"/>\
<sheet name=\"Hidden\" sheetId=\"2\" state=\"hidden\" r:id=\"rId2\"/>\
<sheet name=\"VeryHidden\" sheetId=\"3\" state=\"veryHidden\" r:id=\"rId3\"/>\
<sheet name=\"ChartTab\" sheetId=\"4\" r:id=\"rId4\"/>\
</sheets><definedNames>\
<definedName name=\"GLOB\">Visible!$A$1</definedName>\
<definedName name=\"SCOPED\" localSheetId=\"0\">Visible!$B$1</definedName>\
</definedNames></workbook>"
        );
        let workbook_rels = format!(
            "<?xml version=\"1.0\"?>\
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\
<Relationship Id=\"rId1\" Type=\"{R}/worksheet\" Target=\"worksheets/sheet1.xml\"/>\
<Relationship Id=\"rId2\" Type=\"{R}/worksheet\" Target=\"worksheets/sheet2.xml\"/>\
<Relationship Id=\"rId3\" Type=\"{R}/worksheet\" Target=\"worksheets/sheet3.xml\"/>\
<Relationship Id=\"rId4\" Type=\"{R}/chartsheet\" Target=\"chartsheets/chart1.xml\"/>\
<Relationship Id=\"rId5\" Type=\"{R}/sharedStrings\" Target=\"sharedStrings.xml\"/>\
</Relationships>"
        );
        let sheet = format!(
            "<?xml version=\"1.0\"?>\
<worksheet xmlns=\"{M}\" xmlns:r=\"{R}\"><dimension ref=\"A1:C4\"/><sheetData>\
<row r=\"1\"><c r=\"A1\" t=\"s\"><v>0</v></c><c r=\"B1\"><v>3.5</v></c><c r=\"C1\" t=\"b\"><v>1</v></c><c r=\"D1\" s=\"1\"/></row>\
<row r=\"2\"><c r=\"A2\"><f>SUM(B1:B1)</f><v>3.5</v></c><c r=\"B2\" t=\"e\"><v>#REF!</v></c><c r=\"C2\" t=\"d\"><v>2026-01-02</v></c></row>\
<row r=\"4\"><c r=\"D4\" t=\"s\"><v>1</v></c></row>\
<row r=\"5\"><c r=\"A5\"><f>1+1</f></c></row>\
</sheetData><mergeCells count=\"1\"><mergeCell ref=\"A3:B3\"/></mergeCells>\
<hyperlinks><hyperlink ref=\"B1\" r:id=\"rId9\" tooltip=\"the guide\"/><hyperlink ref=\"C1\" location=\"Hidden!A1\"/></hyperlinks></worksheet>"
        );
        let hidden = "<?xml version=\"1.0\"?>\
<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"/>"
            .to_string();
        let shared = "<?xml version=\"1.0\"?>\
<sst xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><si><t>unit label</t></si><si><t>second label</t></si></sst>"
            .to_string();
        let sheet_rels = format!(
            "<?xml version=\"1.0\"?>\
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\
<Relationship Id=\"rId9\" Type=\"{R}/hyperlink\" Target=\"https://handbook.example.com/torque\" TargetMode=\"External\"/>\
</Relationships>"
        );
        let content_types = "<?xml version=\"1.0\"?>\
<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">\
<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\
<Default Extension=\"xml\" ContentType=\"application/xml\"/>\
<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>\
</Types>"
            .to_string();
        let root_rels = format!(
            "<?xml version=\"1.0\"?>\
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\
<Relationship Id=\"rId1\" Type=\"{R}/officeDocument\" Target=\"xl/workbook.xml\"/>\
</Relationships>"
        );
        zip_of(&[
            ("[Content_Types].xml", content_types.as_bytes()),
            ("_rels/.rels", root_rels.as_bytes()),
            ("xl/workbook.xml", workbook.as_bytes()),
            ("xl/_rels/workbook.xml.rels", workbook_rels.as_bytes()),
            ("xl/worksheets/sheet1.xml", sheet.as_bytes()),
            ("xl/worksheets/sheet2.xml", hidden.as_bytes()),
            ("xl/worksheets/sheet3.xml", hidden.as_bytes()),
            ("xl/sharedStrings.xml", shared.as_bytes()),
            ("xl/worksheets/_rels/sheet1.xml.rels", sheet_rels.as_bytes()),
        ])
    }

    fn structure_of(name: &str, bytes: &[u8]) -> StructureReport {
        document_structure(
            bytes.to_vec(),
            Some(name),
            None,
            Backend::Auto,
            &ConvertOptions::default(),
        )
        .unwrap()
    }

    #[test]
    fn the_structure_capability_table_is_total() {
        // Every kind × every backend: the third table's totality, asserted
        // as the function's own shape. Served rows: docx/xlsx (office legs)
        // and html (auto only); everything else refuses on every backend.
        for backend in [Backend::Auto, Backend::Oxide, Backend::Anydoc] {
            match structure_engine(Kind::Docx, backend, b"PK") {
                Ok(StructureLane::Docx) => assert!(backend != Backend::Anydoc),
                Err(DocumentError::Convert(what)) => {
                    assert_eq!(backend, Backend::Anydoc, "docx");
                    assert!(
                        what.contains("cannot serve document_structure for docx"),
                        "{what}"
                    );
                    assert!(what.contains("use backend='auto' or 'oxide'"), "{what}");
                }
                other => panic!("docx x {backend:?}: {other:?}"),
            }
            match structure_engine(Kind::Excel, backend, b"PK") {
                Ok(StructureLane::Xlsx) => assert!(backend != Backend::Anydoc),
                Err(DocumentError::Convert(what)) => {
                    assert_eq!(backend, Backend::Anydoc, "xlsx");
                    assert!(
                        what.contains("cannot serve document_structure for xlsx"),
                        "{what}"
                    );
                }
                other => panic!("xlsx(ZIP) x {backend:?}: {other:?}"),
            }
            // The Excel kind's OLE container: the union refusal on EVERY
            // backend (legacy xls or encrypted xlsx — the seam cannot
            // disambiguate without a full legacy parse).
            match structure_engine(Kind::Excel, backend, &ole_with_stream("Workbook")) {
                Err(DocumentError::Convert(what)) => {
                    assert!(what.contains("cannot serve xls"), "{what}");
                    assert!(what.contains("legacy xls workbook"), "{what}");
                    assert!(what.contains("encrypted xlsx package"), "{what}");
                }
                other => panic!("xlsx(OLE) x {backend:?}: {other:?}"),
            }
            match structure_engine(Kind::Html, backend, b"") {
                Ok(StructureLane::Html) => assert_eq!(backend, Backend::Auto),
                Err(DocumentError::UnsupportedBackend {
                    format,
                    backend: refused,
                }) => {
                    assert_eq!((format, refused), ("html", backend_name(backend)));
                }
                other => panic!("html x {backend:?}: {other:?}"),
            }
            for kind in [Kind::Doc, Kind::Ppt] {
                match structure_engine(kind, backend, b"") {
                    Err(DocumentError::Convert(what)) => {
                        assert!(
                            what.contains("converter-internal IR (no raw layer)"),
                            "{what}"
                        );
                    }
                    other => panic!("{kind:?} x {backend:?}: {other:?}"),
                }
            }
            match structure_engine(Kind::Pptx, backend, b"") {
                Err(DocumentError::Convert(what)) => {
                    assert!(what.contains("raw layer is reachable"), "{what}");
                    assert!(what.contains("not built yet"), "{what}");
                }
                other => panic!("pptx x {backend:?}: {other:?}"),
            }
            for kind in [
                Kind::Rtf,
                Kind::Odt,
                Kind::Ods,
                Kind::Odp,
                Kind::Epub,
                Kind::Csv,
            ] {
                match structure_engine(kind, backend, b"") {
                    Err(DocumentError::Convert(what)) => {
                        assert!(
                            what.contains(&format!(
                                "no structure surface exists for {}",
                                kind.name()
                            )),
                            "{what}"
                        );
                        // never "unreadable": the format converts fine
                        assert!(what.contains("converts fine"), "{what}");
                    }
                    other => panic!("{kind:?} x {backend:?}: {other:?}"),
                }
            }
            match structure_engine(Kind::Pdf, backend, b"") {
                Err(DocumentError::Convert(what)) => {
                    assert_eq!(
                        what,
                        "document_structure cannot serve pdf: pdf has no structure surface \
                         here — page_layout and page_tables serve PDF pages (spans, lines, \
                         images, per-cell table boxes); document_structure serves the \
                         office/html families' containers"
                    );
                }
                other => panic!("pdf x {backend:?}: {other:?}"),
            }
        }
        // html's forced backends are conversion's own row byte-identical
        // (the pair cannot convert; the structure lane is the same parse).
        for backend in [Backend::Oxide, Backend::Anydoc] {
            let mine = structure_engine(Kind::Html, backend, b"").unwrap_err();
            let conversions = engine_for(Kind::Html, backend).unwrap_err();
            assert_eq!(mine.to_string(), conversions.to_string());
        }
    }

    #[test]
    fn the_structure_refusal_order_is_pinned() {
        // resolve → capability → password (EVERY kind) → max_bytes=0 →
        // engine doors. A pdf call reaches the pdf refusal with an
        // unexamined password (the F25 trap: the geometry surfaces'
        // kind!=Pdf guard would let it through); max_bytes=0 never
        // pre-empts the capability refusal.
        let options = ConvertOptions {
            password: Some("pw"),
            ..Default::default()
        };
        let err = document_structure(b"junk".to_vec(), None, Some("pdf"), Backend::Auto, &options)
            .unwrap_err();
        assert!(err.to_string().contains("cannot serve pdf"), "{err}");
        // html: served row, so the password refusal fires (the same for
        // every kind)
        let err = document_structure(
            POLLUTED_HTML.as_bytes().to_vec(),
            None,
            Some("html"),
            Backend::Auto,
            &options,
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "document_structure takes no password: no served format accepts one"
        );
        // max_bytes=0 after the capability refusal...
        let options = ConvertOptions {
            max_bytes: Some(0),
            ..Default::default()
        };
        let err = document_structure(b"junk".to_vec(), None, Some("pdf"), Backend::Auto, &options)
            .unwrap_err();
        assert!(err.to_string().contains("cannot serve pdf"), "{err}");
        // ...and after the password refusal (F25 fires for every kind)
        let options = ConvertOptions {
            password: Some("pw"),
            ..Default::default()
        };
        let err = document_structure(
            POLLUTED_HTML.as_bytes().to_vec(),
            None,
            Some("html"),
            Backend::Auto,
            &options,
        )
        .unwrap_err();
        assert!(err.to_string().contains("takes no password"), "{err}");
        // max_bytes=0 alone: the probes' exact string
        let options = ConvertOptions {
            max_bytes: Some(0),
            ..Default::default()
        };
        let err = document_structure(
            POLLUTED_HTML.as_bytes().to_vec(),
            None,
            Some("html"),
            Backend::Auto,
            &options,
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "max_bytes must be positive");
        // (pdf, anydoc): the format-level pdf refusal, never an
        // anydoc-template message (F8's precedence pin)
        let err = document_structure(
            b"junk".to_vec(),
            None,
            Some("pdf"),
            Backend::Anydoc,
            &ConvertOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("cannot serve pdf"), "{err}");
        assert!(!err.to_string().contains("backend \"anydoc\""), "{err}");
        // an OLE workbook under the Excel kind refuses naming xls — from
        // the probe's own 64 KiB prefix too (the container decision needs
        // only the 8 magic bytes)
        let ole = ole_with_stream("Workbook");
        for backend in [Backend::Auto, Backend::Anydoc] {
            let err = document_structure(
                ole.clone(),
                Some("book.xlsx"),
                None,
                backend,
                &ConvertOptions::default(),
            )
            .unwrap_err();
            assert!(err.to_string().contains("cannot serve xls"), "{err}");
        }
        assert!(matches!(
            probe_structure_capability(&ole[..8], Some("book.xlsx"), None, Backend::Auto),
            Err(DocumentError::Convert(_))
        ));
    }

    #[test]
    fn the_structure_probe_refuses_from_the_name_alone() {
        // format= is trusted and never re-sniffed: the pdf refusal fires
        // under the GIL from the NAME alone, before any read (the R7
        // observable's core half; the binding half is the Python no-read
        // pin). The refused anydoc-only families answer from the name too.
        for (format, needle) in [
            ("pdf", "cannot serve pdf"),
            ("odt", "no structure surface exists for odt"),
            ("pptx", "not built yet"),
            ("doc", "converter-internal IR"),
        ] {
            let err = probe_structure_capability(b"", None, Some(format), Backend::Auto)
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle), "{format}: {err}");
        }
        // an unresolvable prefix defers (the core's authoritative resolve
        // is the only truth)
        assert_eq!(
            probe_structure_capability(b"", None, None, Backend::Auto).unwrap(),
            None
        );
        // a served row resolves
        assert_eq!(
            probe_structure_capability(
                POLLUTED_HTML.as_bytes(),
                Some("p.html"),
                None,
                Backend::Auto
            )
            .unwrap(),
            Some(Kind::Html)
        );
    }

    #[test]
    fn the_structure_lanes_serve_their_formats() {
        // html: the collector's nodes, the nine reachable kinds, no
        // script/style junk node, the link annotation's byte range.
        let report = structure_of("page.html", POLLUTED_HTML.as_bytes());
        assert_eq!(report.format, "html");
        let StructurePayload::Html(html) = report.payload else {
            panic!("html payload expected");
        };
        assert!(!html.entries_truncated);
        let kinds: std::collections::HashSet<&str> = html.nodes.iter().map(|n| n.kind).collect();
        // the reachable kinds this fixture reaches (the closed set of
        // nine is pinned by the payload's own vocabulary and the docs;
        // the junk variants cannot be emitted at all)
        for kind in [
            "heading",
            "paragraph",
            "list",
            "list_item",
            "table",
            "group",
        ] {
            assert!(kinds.contains(&kind), "missing {kind}: {kinds:?}");
        }
        // the Group tree: parents precede children, headings are Group
        // children carrying the label (the heading tree IS the Group
        // hierarchy)
        for (index, node) in html.nodes.iter().enumerate() {
            if let Some(parent) = node.parent {
                assert!((parent as usize) < index, "parent {parent} after {index}");
            }
        }
        let heading = html
            .nodes
            .iter()
            .find(|n| n.kind == "heading" && n.text.as_deref() == Some("Real Heading"))
            .expect("the h1");
        assert_eq!(heading.level, Some(1));
        let group_index = html
            .nodes
            .iter()
            .position(|n| n.kind == "group" && n.label.as_deref() == Some("Real Heading"))
            .expect("the h1's Group");
        assert_eq!(heading.parent, Some(group_index as u32));
        let paragraph = html
            .nodes
            .iter()
            .find(|n| n.kind == "paragraph" && n.annotations.iter().any(|a| a.kind == "link"))
            .expect("the link paragraph");
        assert_eq!(paragraph.kind, "paragraph");
        let link = paragraph
            .annotations
            .iter()
            .find(|a| a.kind == "link")
            .expect("the link annotation");
        assert_eq!(
            link.url.as_deref(),
            Some("https://handbook.example.com/torque")
        );
        assert_eq!(
            paragraph.text.as_deref().unwrap_or_default().as_bytes()
                [link.start as usize..link.end as usize],
            *b"field handbook"
        );
        // the table grid, verbatim cells
        let table = html
            .nodes
            .iter()
            .find(|n| n.kind == "table")
            .expect("the table node");
        let grid = table.grid.as_ref().unwrap();
        assert_eq!((grid.rows, grid.cols), (2, 2));
        assert_eq!(grid.cells[0].content, "Unit");
        assert!(grid.cells[0].is_header);
        assert_eq!(grid.cells[2].content, "T-101");
        assert!(!grid.cells[2].is_header);
        // NO node whose text carries the script/style junk (the F13
        // replacement for the tautological MetadataBlock assertion —
        // RawBlock/MetadataBlock nodes cannot be pushed at all; the junk
        // must not leak either way)
        for node in &html.nodes {
            for junk in ["console.log", "tracking junk", "color: red", "Page Title"] {
                assert!(
                    !node.text.as_deref().unwrap_or("").contains(junk),
                    "junk leaked into node {}",
                    node.id
                );
            }
        }
    }

    #[test]
    fn the_html_lane_is_lenient_and_fragment_resolving() {
        // Malformed html parses with the conversion lane's own recovery —
        // the structure IS the conversion lane's parse (F28: one parse,
        // one collector, by construction).
        let malformed = b"<html><body><h1>unclosed heading<p>text</body>";
        let report = structure_of("mess.html", malformed);
        let StructurePayload::Html(html) = report.payload else {
            panic!("html payload expected");
        };
        assert!(
            html.nodes.iter().any(|n| n.kind == "heading"
                && n.text.as_deref().unwrap_or("").contains("unclosed heading"))
        );
        // A fragment without a name is NOT content-resolvable (the pinned
        // looks_like_html policy); with format="html" it is served — and
        // binary bytes named html are html5ever-recovered text (F26's
        // honest consequence).
        let fragment = b"<div><p>just a fragment</p></div>";
        let err = document_structure(
            fragment.to_vec(),
            None,
            None,
            Backend::Auto,
            &ConvertOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, DocumentError::UnknownFormat(_)), "{err}");
        let report = document_structure(
            fragment.to_vec(),
            None,
            Some("html"),
            Backend::Auto,
            &ConvertOptions::default(),
        )
        .unwrap();
        let StructurePayload::Html(html) = report.payload else {
            panic!("html payload expected");
        };
        assert!(
            html.nodes
                .iter()
                .any(|n| n.text.as_deref() == Some("just a fragment"))
        );
        // Binary bytes named html: the lane's own binary-input refusal
        // (the structure IS the conversion lane's parse, and the
        // conversion lane detects binary data and refuses — the honest
        // consequence of trusted format=, never a fabricated structure
        // of garbage; the review's F26 premise about html5ever-recovered
        // garbage did not survive contact with the lane's binary gate).
        let err = document_structure(
            write_docx_bytes(),
            None,
            Some("html"),
            Backend::Auto,
            &ConvertOptions::default(),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("html-to-markdown-rs: Invalid input: binary data"),
            "{err}"
        );
        // Non-markup text named html is the lane's own leniency: bare
        // text parses as text (a no-node structure for tagless text is
        // the collector's own answer — the document is refused there, the
        // one shape where `document: None` is honest)
        let report = document_structure(
            b"<p>just a paragraph, no document marker</p>".to_vec(),
            None,
            Some("html"),
            Backend::Auto,
            &ConvertOptions::default(),
        )
        .unwrap();
        assert_eq!(report.format, "html");
        assert!(matches!(report.payload, StructurePayload::Html(_)));
        // format="docx" on pdf bytes: the office leg's own typed parse
        // error (trusted format=, never a re-sniff)
        let err = document_structure(
            two_page_pdf("a", "b"),
            None,
            Some("docx"),
            Backend::Auto,
            &ConvertOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("office_oxide"), "{err}");
    }

    #[test]
    fn the_docx_structure_payload_is_pinned() {
        let report = structure_of("report.docx", &structure_docx_fixture());
        assert_eq!(report.format, "docx");
        let StructurePayload::Docx(docx) = report.payload else {
            panic!("docx payload expected");
        };
        assert_eq!(docx.sections, 2);
        assert_eq!(docx.section_breaks, [9]);
        assert_eq!(docx.footnotes, 0);
        assert_eq!(docx.comments, 0);
        // the unreadable header part is reported, never skipped silently
        assert_eq!(docx.unreadable_parts.len(), 1);
        assert!(docx.unreadable_parts[0].contains("header1.xml"));
        assert!(!docx.entries_truncated);
        let blocks = &docx.blocks;
        assert_eq!(blocks.len(), 10);
        assert_eq!(blocks[0].section, 0);
        assert_eq!(blocks[9].section, 1);

        // the heading ladder: style outlineLvl → level 1; direct
        // outlineLvl 7 → level clamps to 6 with the raw 7 carried
        assert_eq!(
            (blocks[0].kind, blocks[0].level, blocks[0].outline_level),
            ("heading", Some(1), Some(0))
        );
        assert_eq!(blocks[0].style_id.as_deref(), Some("Heading1"));
        assert_eq!(
            (blocks[1].kind, blocks[1].level, blocks[1].outline_level),
            ("heading", Some(6), Some(7))
        );
        assert_eq!(blocks[1].style_id, None);

        // list membership on the EFFECTIVE properties: the style carries
        // the numPr ("List Paragraph" style)
        assert_eq!(blocks[2].kind, "list_item");
        assert_eq!(blocks[2].num_id, Some(1));
        assert_eq!(blocks[2].ilvl, Some(0));
        assert_eq!(blocks[2].list_format.as_deref(), Some("bullet"));
        assert_eq!(blocks[2].start_number, Some(1));
        assert_eq!(blocks[2].text, "bulleted through the style");

        // a numbered heading: heading wins, the numPr is information
        assert_eq!(blocks[3].kind, "heading");
        assert_eq!(blocks[3].level, Some(2));
        assert_eq!(blocks[3].num_id, Some(1));
        assert_eq!(blocks[3].list_format.as_deref(), Some("bullet"));

        // the block-text walk: br → newline, tab → tab, the hidden run
        // resolved through the style chain and dropped
        assert_eq!(blocks[4].kind, "paragraph");
        assert_eq!(blocks[4].text, "before\n\tafter");
        assert!(!blocks[4].text.contains("hidden text"));

        // the external hyperlink: runs' text, rels-resolved href
        assert_eq!(blocks[5].links.len(), 1);
        assert_eq!(blocks[5].links[0].text, "field handbook");
        assert_eq!(
            blocks[5].links[0].href.as_deref(),
            Some("https://handbook.example.com/torque")
        );
        assert_eq!(blocks[5].links[0].anchor, None);
        assert_eq!(blocks[5].text, "field handbook");

        // the picture: rid, ext from the package map, declared extent
        // ÷ 12700 (1905000 EMU = 150 pt), alt text, inline
        assert_eq!(blocks[6].images.len(), 1);
        let image = &blocks[6].images[0];
        assert_eq!(image.relationship_id, "rId5");
        assert_eq!(image.file_ext.as_deref(), Some("png"));
        assert_eq!(image.description.as_deref(), Some("torque chart"));
        assert_eq!(image.width_pt, Some(150.0));
        assert_eq!(image.height_pt, Some(100.0));
        assert!(image.inline);
        assert!(!image.decorative);

        // the table: grid verbatim, caption = tblCaption, block text ""
        let table = blocks[7].table.as_ref().unwrap();
        assert_eq!(blocks[7].kind, "table");
        assert_eq!(blocks[7].text, "");
        assert_eq!((table.rows, table.cols), (3, 3));
        assert_eq!(table.col_widths_twips, [2000, 2000, 2000]);
        assert_eq!(table.caption.as_deref(), Some("Units"));
        let cells = &table.cells;
        assert_eq!(cells.len(), 8);
        // the gridSpan cell: col 0, col_span 2; the third cell sits at
        // col 2 (the accumulation, the raw layer's grid_starts walk)
        assert_eq!((cells[0].row, cells[0].col, cells[0].col_span), (0, 0, 2));
        assert_eq!(cells[0].text, "wide");
        assert_eq!((cells[1].row, cells[1].col), (0, 2));
        // the vMerge pair: the Restart carries the walked row span; the
        // Continue cell is emitted as its own cell, inside the span
        assert_eq!((cells[2].row, cells[2].col), (1, 0));
        assert_eq!(cells[2].row_span, 2);
        assert_eq!(cells[2].text, "span");
        assert_eq!((cells[5].row, cells[5].col), (2, 0));
        assert_eq!(cells[5].row_span, 1);
        assert_eq!(cells[5].text, "");
        assert_eq!((cells[6].row, cells[6].col), (2, 1));
        assert_eq!(cells[6].text, "b2");
        assert_eq!((cells[7].row, cells[7].col), (2, 2));
        assert_eq!(cells[7].text, "c2");

        // stable anchors
        for (index, block) in blocks.iter().enumerate() {
            assert_eq!(block.index, index);
        }
    }

    #[test]
    fn the_xlsx_structure_payload_is_pinned() {
        let report = structure_of("book.xlsx", &structure_xlsx_fixture());
        assert_eq!(report.format, "xlsx");
        let StructurePayload::Xlsx(xlsx) = report.payload else {
            panic!("xlsx payload expected");
        };
        assert!(!xlsx.date1904);
        // sheets from the raw layer's own list, kind-filtered: the
        // chartsheet is a name here, never a mislabeled sheet
        assert_eq!(
            xlsx.sheets
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>(),
            ["Visible", "Hidden", "VeryHidden"]
        );
        assert_eq!(xlsx.chartsheets, ["ChartTab"]);
        assert_eq!(
            xlsx.sheets.iter().map(|s| s.state).collect::<Vec<_>>(),
            ["visible", "hidden", "veryHidden"]
        );
        // defined names: verbatim value, the scoped one's sheet resolved
        assert_eq!(xlsx.defined_names.len(), 2);
        assert_eq!(xlsx.defined_names[0].name, "GLOB");
        assert_eq!(xlsx.defined_names[0].value, "Visible!$A$1");
        assert_eq!(xlsx.defined_names[0].local_sheet_id, None);
        assert_eq!(xlsx.defined_names[0].local_sheet_name, None);
        assert_eq!(xlsx.defined_names[1].local_sheet_id, Some(0));
        assert_eq!(
            xlsx.defined_names[1].local_sheet_name.as_deref(),
            Some("Visible")
        );

        let sheet = &xlsx.sheets[0];
        // the used range is COMPUTED from the parsed cells (styled-empty
        // cells included), never the declared dimension: content sits at
        // D4/D1 past the declared A1:C4
        assert_eq!(sheet.used_range.as_deref(), Some("A1:D5"));
        assert_eq!(sheet.declared_dimension.as_deref(), Some("A1:C4"));
        assert_eq!((sheet.max_row, sheet.max_col), (Some(4), Some(3)));
        let cells = &sheet.cells;
        assert_eq!(cells.len(), 9);

        let by_ref = |reference: &str| {
            cells
                .iter()
                .find(|c| c.reference == reference)
                .unwrap_or_else(|| panic!("{reference}"))
        };
        // shared string: dereferenced, full text
        let a1 = by_ref("A1");
        assert_eq!(a1.kind, "string");
        assert_eq!(a1.value, Some(XlsxScalar::Text("unit label".into())));
        assert_eq!(a1.row, 0);
        assert_eq!(a1.col, 0);
        // typed cells, as the file states them — no coercion
        assert_eq!(by_ref("B1").kind, "number");
        assert_eq!(by_ref("B1").value, Some(XlsxScalar::Number(3.5)));
        assert_eq!(by_ref("C1").kind, "boolean");
        assert_eq!(by_ref("C1").value, Some(XlsxScalar::Bool(true)));
        assert_eq!(by_ref("B2").kind, "error");
        assert_eq!(by_ref("B2").value, Some(XlsxScalar::Text("#REF!".into())));
        // t="d" is the ONLY date shape
        assert_eq!(by_ref("C2").kind, "date");
        assert_eq!(
            by_ref("C2").value,
            Some(XlsxScalar::Text("2026-01-02".into()))
        );
        // a styled-but-empty cell is present (type empty, value None)
        let d1 = by_ref("D1");
        assert_eq!(d1.kind, "empty");
        assert_eq!(d1.value, None);
        // formulas: text and cached value, both labeled
        let a2 = by_ref("A2");
        assert_eq!(a2.formula.as_deref(), Some("SUM(B1:B1)"));
        assert!(a2.has_cached_value);
        assert_eq!(a2.value, Some(XlsxScalar::Number(3.5)));
        // a formula with no cached value
        let a5 = by_ref("A5");
        assert_eq!(a5.formula.as_deref(), Some("1+1"));
        assert!(!a5.has_cached_value);
        assert_eq!(a5.value, None);

        assert_eq!(sheet.merged_cells, ["A3:B3"]);
        assert_eq!(sheet.hyperlinks.len(), 2);
        assert_eq!(sheet.hyperlinks[0].cell_ref, "B1");
        assert_eq!(
            sheet.hyperlinks[0].href.as_deref(),
            Some("https://handbook.example.com/torque")
        );
        assert_eq!(sheet.hyperlinks[0].anchor, None);
        assert_eq!(sheet.hyperlinks[0].tooltip.as_deref(), Some("the guide"));
        assert_eq!(sheet.hyperlinks[1].cell_ref, "C1");
        assert_eq!(sheet.hyperlinks[1].href, None);
        assert_eq!(sheet.hyperlinks[1].anchor.as_deref(), Some("Hidden!A1"));

        // the hidden sheets are present with their states
        assert_eq!(xlsx.sheets[1].used_range, None);
        assert_eq!(
            (xlsx.sheets[1].max_row, xlsx.sheets[1].max_col),
            (None, None)
        );
    }

    #[test]
    fn the_entry_cap_truncates_with_a_marker() {
        // The cap boundary, measured through the real reader: 199,999
        // cells fit; exactly 200,000 fit; 200,001 truncates to 200,000
        // with the marker, the sheets/ranges/metadata intact. A
        // marshalling bound, never a refusal.
        let cells = |count: usize| {
            let rows: Vec<String> = (1..=count)
                .map(|row| format!("<row r=\"{row}\"><c r=\"A{row}\"><v>{row}</v></c></row>"))
                .collect();
            let sheet = format!(
                "<?xml version=\"1.0\"?><worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><sheetData>{}</sheetData></worksheet>",
                rows.concat()
            );
            const M: &str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
            const R: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
            let workbook = format!(
                "<?xml version=\"1.0\"?><workbook xmlns=\"{M}\" xmlns:r=\"{R}\"><sheets>\
                 <sheet name=\"Big\" sheetId=\"1\" r:id=\"rId1\"/></sheets></workbook>"
            );
            let workbook_rels = format!(
                "<?xml version=\"1.0\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\
                 <Relationship Id=\"rId1\" Type=\"{R}/worksheet\" Target=\"worksheets/sheet1.xml\"/>\
                 <Relationship Id=\"rId2\" Type=\"{R}/sharedStrings\" Target=\"sharedStrings.xml\"/></Relationships>"
            );
            let content_types = "<?xml version=\"1.0\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/><Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/></Types>".to_string();
            let root_rels = format!(
                "<?xml version=\"1.0\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\
                 <Relationship Id=\"rId1\" Type=\"{R}/officeDocument\" Target=\"xl/workbook.xml\"/></Relationships>"
            );
            zip_of(&[
                ("[Content_Types].xml", content_types.as_bytes()),
                ("_rels/.rels", root_rels.as_bytes()),
                ("xl/workbook.xml", workbook.as_bytes()),
                ("xl/_rels/workbook.xml.rels", workbook_rels.as_bytes()),
                ("xl/worksheets/sheet1.xml", sheet.as_bytes()),
            ])
        };
        for (count, expected, truncated) in [
            (199_999, 199_999, false),
            (200_000, 200_000, false),
            (200_001, 200_000, true),
        ] {
            let report = structure_of("big.xlsx", &cells(count));
            let StructurePayload::Xlsx(xlsx) = report.payload else {
                panic!("xlsx payload expected");
            };
            assert_eq!(xlsx.sheets.len(), 1, "{count}");
            assert_eq!(xlsx.sheets[0].cells.len(), expected, "{count}");
            assert_eq!(xlsx.entries_truncated, truncated, "{count}");
            // the metadata stays complete past the cap: the used range is
            // the FULL computed range even when cells were dropped
            let sheet = &xlsx.sheets[0];
            assert_eq!(
                sheet.used_range.as_deref(),
                Some(format!("A1:A{count}").as_str()),
                "{count}"
            );
            assert_eq!(
                (sheet.max_row, sheet.max_col),
                (Some(count - 1), Some(0)),
                "{count}"
            );
            assert_eq!(sheet.name, "Big");
        }
    }

    #[test]
    fn the_structure_ceiling_and_audit_bind_the_office_leg() {
        // the office leg's input ceiling: the same message shape as
        // page_geometry's office leg (the 32 MiB default under None)
        let big = vec![b'x'; 33 * 1024 * 1024];
        let err = document_structure(
            big,
            Some("big.docx"),
            None,
            Backend::Auto,
            &ConvertOptions::default(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("input ceiling is 32.0 MiB"),
            "{err}"
        );
        // a zip bomb dies in the audit before the engine parses anything
        // (the office leg runs the same audit every office lane runs)
        let bomb = zip_container(&[("word/document.xml", 8, b"x", Some(600 * 1024 * 1024))]);
        let err = document_structure(
            bomb,
            Some("bomb.docx"),
            None,
            Backend::Auto,
            &ConvertOptions::default(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("decompression limit exceeded"),
            "{err}"
        );
    }
}
