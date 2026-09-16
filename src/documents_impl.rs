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
/// four probe-rich calls need per-engine surfaces anydoc's PDF reader
/// structurally lacks: its entire PDF surface is `to_markdown(bytes)`
/// whole-document markdown (~anydoc-0.2.4/src/formats/pdf.rs: no
/// per-page text, no page-count success return, no per-page
/// classification, no annotation walk), while `pdf_extract`'s per-page
/// probe is the OCR-routing signal, `pdf_page_count` walks the page
/// tree, `pdf_classify` classifies per page, and `pdf_link_uris` walks
/// `/Annots`. The doctrine text lives here, next to the routing table,
/// so the four functions' messages cannot drift apart; the binding
/// layer raises it under the GIL at argument-contract time, before any
/// work runs. Message shape follows [`DocumentError::UnsupportedBackend`]'s
/// Display: name the backend and the value, name the lack, point at the
/// fix.
pub fn anydoc_capability_refusal(capability: &str) -> DocumentError {
    DocumentError::Convert(format!(
        "backend \"anydoc\" cannot serve {capability}: anydoc's PDF surface is \
         whole-document conversion only — to_markdown/to_text with backend=\"anydoc\" \
         (NeedsOcrError is that lane's scanned-page signal); use backend='auto' or \
         'oxide' here"
    ))
}

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
        let err = anydoc_capability_refusal("the per-page text probe (the OCR-routing signal)");
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
            let text = anydoc_capability_refusal(capability).to_string();
            assert!(
                text.contains(capability),
                "must name the capability: {text:?}"
            );
            assert!(
                text.contains("to_markdown/to_text with backend=\"anydoc\""),
                "must point at the anydoc PDF surface: {text:?}"
            );
        }
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
}
