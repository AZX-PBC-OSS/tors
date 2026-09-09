//! The multi-format document extraction core (the `tors-documents`
//! payload crate's bindings): resolve a format, route it to the measured
//! best engine, convert to GitHub-Flavored Markdown, and optionally
//! normalize that markdown to plain text. Pure Rust, no pyo3 — the binding
//! layer lives in `tors-documents/src/lib.rs`, which releases the GIL
//! around this whole pass.
//!
//! # Engine routing, and why exactly this split
//!
//! Four engines, chosen per family by head-to-head measurement (2026-09, over
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
//!   measured engine that drops `<script>`/`<style>` BY CONSTRUCTION —
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
//!   `&`-escaping) against the heading/list losses above — a documented
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
//! An explicit format name (extension spelling, no dot, case-insensitive —
//! anydoc's `Format::from_extension` vocabulary plus `html`/`htm`/`xhtml`
//! and `tsv`, HTML's and the delimiter-separated family's name-only
//! spellings) beats everything. Without one, the format is sniffed from
//! the bytes' content markers (PDF header, RTF open group, OLE stream
//! names, ZIP mimetype, HTML document marker) with the input NAME's
//! extension as the last resort for signature-less formats (CSV) — the
//! `path=` the payload read, when there was one; the `data=` entry has no
//! name and rests on `format=`/markers alone, the doctrine [`sniff`]
//! encodes — and the extension resolves through the SAME [`Kind::from_name`]
//! vocabulary the
//! explicit name uses, so one table owns both spellings. anydoc's
//! doctrine, reused here so a mislabeled file still converts correctly.
//! [`sniff`] exposes the content-marker half standalone, over bytes
//! alone, reporting the same container-true name a conversion would.
//!
//! # Plain-text output
//!
//! `to_text` converts to markdown first and then normalizes via
//! [`gfm_strip_impl::strip`] — one text shape for every format and engine,
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
//! pdf_oxide's own inter-page separator — measured, a full range is
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
/// routing answers speak in BACKEND terms, resource answers in ENGINE
/// ones — the ceiling is an engine-lane property).
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
/// to round to "0.0 MiB … 0.0 MiB" — two sizes, neither readable).
fn render_size(size: usize) -> String {
    let mib = size as f64 / (1024.0 * 1024.0);
    if mib < 0.1 {
        format!("{size} bytes")
    } else {
        format!("{mib:.1} MiB")
    }
}

/// The working-format vocabulary: anydoc's `Format` (which also names the
/// OOXML container variants — `docm`/`xlsm`/`xlsb`/`ppsx` map onto these)
/// plus HTML — XHTML included, it being HTML's XML serialization on the
/// same engine — which no anydoc parser reads and the lane's own engine
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
    /// own engine — XHTML is HTML's XML serialization, and its
    /// `<?xml …?>` prologue is the one prefix variant the content marker
    /// knows how to skip; TSV is the delimiter-separated family's other
    /// half — anydoc's csv parser is delimiter-separated, and the content
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
/// explicit format name — the caller may have passed `format=None` and wants
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
    /// OCR — routing information, not a generic failure: the caller's chain
    /// should send this document to its OCR stage. `pages` holds 0-based
    /// page INDICES (the surface-wide convention; anydoc's 1-based numbers
    /// are re-based once, in `anydoc_error`, where the engine's answer
    /// crosses the seam).
    NeedsOcr { pages: Vec<u32>, page_count: u32 },
    /// The document is not convertible: malformed, encrypted, over a
    /// resource limit, a required part missing, or an engine parse failure.
    Convert(String),
}

/// The error text — byte-equal to the strings the binding layer raises on
/// the Python side, so a Rust caller `?`-ing into `Box<dyn Error>`/anyhow
/// sees exactly what the payload's user sees (the payload's mapping arms
/// carry the same strings; its `UnknownFormat`/`Pages`/`Convert` arms
/// could now collapse onto `to_string()` — noted for that crate, not
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

/// The document-engine lanes' default input ceiling: 32 MiB, applied to
/// the two lanes that AMPLIFY their input into resident memory — anydoc
/// and office_oxide. anydoc AMPLIFIES at a measured ~146× worst case on
/// adversarial delimiter formats (2026-09-09, this box: a 24 MiB csv of
/// 1,258,291 rows × 10 one-char cells peaked at 3.42 GiB RSS through
/// to_text in 6.4 s, and a 12 MiB one at 1.73 GiB in 3.1 s — the
/// multiple is stable across sizes, a per-byte property: anydoc
/// materializes row structures per cell. The earlier ~36× figure — a
/// 100 MiB csv → 3.7 GiB — was a benign-shape measurement; cells per
/// byte, not file size, drives the multiple) and additionally caps
/// decompression engine-side (its own package limits: 128 MiB per entry,
/// 512 MiB total, a 4M× expansion bound). office_oxide 0.1.10 (released
/// 2026-09-09, its changelog #144/#151) now enforces MAX_PART_SIZE =
/// 512 MiB per part — declared and actual bytes; XML nesting depth 256
/// on its own 16 MB parse stack — but still has NO total-across-parts
/// cap and NO output cap: measured the same day, a 399 KiB zip with a
/// 400 MiB word/document.xml converts successfully at 1.58 GiB peak RSS
/// in under a second (the ~400 MiB markdown handed over whole), while a
/// 598 KiB zip declaring a 600 MiB part is refused pre-decompression
/// ("decompression limit exceeded … more than 536870912 bytes", 0.03 s,
/// ~20 MiB RSS). That is why the oxide lane gets the same input-side
/// bound even though the bound does not — cannot — cap what a compressed
/// container inflates to; selecting `backend="oxide"` accepts that risk
/// (an opt-in lane, never the default). At the measured ~146×, the 32
/// MiB default ceiling's honest worst case on the anydoc lane is ~4.6
/// GiB — not a survivable spike on a small ingestion worker; callers
/// with tight budgets must lower [`ConvertOptions::max_bytes`] or split
/// the file. The NAME stays anydoc-branded (history: the knob was born
/// anydoc-only and the payload crate's docs reference it by this name);
/// the SEMANTICS are both amplified lanes. The pdf_oxide and HTML lanes
/// are unmetered here: pdf_oxide's own resource limits govern there, and
/// the HTML lane converts text it can size directly.
pub const DEFAULT_ANYDOC_INPUT_LIMIT: usize = 32 * 1024 * 1024;

/// The [MS-CFB] OLE compound-file signature: the legacy office container
/// (doc/ppt, and Excel's legacy `xls` inside the Excel kind). One
/// constant, shared by the reported-name choice in [`resolved_name`] and
/// the oxide lane's engine choice in `office_markdown`, so the two can
/// never disagree about which container the bytes are.
const OLE_MAGIC: [u8; 8] = [0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1];

fn is_ole_container(bytes: &[u8]) -> bool {
    bytes.starts_with(&OLE_MAGIC)
}

/// The name a resolved kind reports back as — the Excel kind's one
/// container-aware case: the OLE signature is legacy `xls`, anything else
/// (a ZIP local-file header) is the `xlsx` family, so the reported name
/// matches the container the bytes actually are (the same container check
/// `office_markdown` makes for the engine choice, applied to the report —
/// and to [`sniff`]'s answer, which routes through here for the same
/// reason).
fn resolved_name(kind: Kind, bytes: &[u8]) -> &'static str {
    if kind == Kind::Excel && is_ole_container(bytes) {
        return "xls";
    }
    kind.name()
}

/// The caller-tuned conversion knobs beyond the routing arguments — one
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
    /// FORMAT is a `Convert` error, and so is one on the anydoc PDF lane —
    /// its reader takes no password, so a correct one would fail exactly
    /// like a wrong one, indistinguishable from ignored. Never a silently
    /// ignored argument.
    pub password: Option<&'a str>,
    /// The engine-lane input ceiling, in bytes: it bounds INPUT size on
    /// the two lanes that amplify input into resident memory — anydoc
    /// (~146× RSS worst case on adversarial delimiter formats: a 24 MiB
    /// csv → 3.42 GiB peak, a 12 MiB one → 1.73 GiB, 2026-09-09) and
    /// office_oxide (a 399 KiB zip-bombed docx with a 400 MiB part →
    /// 1.58 GiB peak on that lane, same date). `None` is the default:
    /// the 32 MiB [`DEFAULT_ANYDOC_INPUT_LIMIT`]; `Some(n)` raises or
    /// lowers it for callers with a bigger (or tighter) memory budget —
    /// at the measured multiple the default's anydoc-lane worst case is
    /// ~4.6 GiB, so tighter is often right. The honest limits of what it
    /// bounds: anydoc additionally caps decompression ENGINE-SIDE (its
    /// package limits), while office_oxide 0.1.10 caps 512 MiB per part
    /// but has NO total-across-parts cap and NO output cap — an opt-in
    /// lane whose caller accepts unbounded multi-part decompression risk
    /// by selecting it; the ceiling bounds the bytes handed in, never
    /// the bytes they inflate to. The pdf_oxide and HTML lanes are
    /// unmetered by this knob: their own limits govern.
    pub max_bytes: Option<usize>,
}

/// Convert document bytes to GitHub-Flavored Markdown on the routed (or
/// forced) engine. `name_hint` is the input's name — the `path=` the
/// payload read, when there was one — used ONLY by the extension fallback
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
    // engines emit text literally — un-escaping theirs would corrupt text
    // that genuinely contains `&amp;`.
    let unescape = engine == Engine::Anydoc;
    let text = gfm_strip_impl::strip(&markdown, unescape);
    Ok(Converted {
        format,
        output: text,
    })
}

/// The standalone content sniffer: what the conversion would resolve these
/// bytes to from CONTENT alone — no path, no extension. The PDF header, the
/// RTF open group, OLE stream names, the ZIP package mimetype, and the HTML
/// document marker. `None` = the content names no format (a signature-less
/// text format such as CSV, or not a document at all). The reported name is
/// the conversion's own [`resolved_name`] — Excel's one container-aware
/// case included: an OLE workbook sniffs as `xls`, a ZIP one as `xlsx` —
/// so `sniff` and `to_markdown` can never disagree about what the bytes
/// are.
pub fn sniff(bytes: &[u8]) -> Option<&'static str> {
    sniff_kind(bytes).map(|kind| resolved_name(kind, bytes))
}

/// The shared spine of both public functions: resolve, route, convert. The
/// bytes are already in hand — the payload read the `path=` (or took the
/// caller's `data=`) before calling here — so no IO happens in this seam.
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
        // like a wrong one — indistinguishable from having been ignored.
        // Refuse up front, mirroring the pages= lane guard.
        return Err(DocumentError::Convert(
            "password= requires the pdf_oxide lane (backend='auto' or 'oxide'); \
             anydoc's PDF reader takes no password"
                .into(),
        ));
    }
    // The input ceiling: BOTH document-holding lanes that AMPLIFY their
    // input into resident memory — anydoc (~146x worst case on
    // adversarial delimiter formats, and engine-side decompression caps
    // on top; see [DEFAULT_ANYDOC_INPUT_LIMIT]) and office_oxide (512
    // MiB per part since 0.1.10, but no total-across-parts cap and no
    // output cap: a 399 KiB zip-bomb docx with a 400 MiB part peaked at
    // 1.58 GiB RSS on the oxide lane, measured 2026-09-09) — so the
    // opt-in lane gets the same input-side bound as the default one.
    // What the ceiling does NOT do is bound office_oxide's DEcompression
    // beyond that per-part cap: that lane's caller accepts unbounded
    // multi-part decompression risk by selecting it (the honest state,
    // stated in [`ConvertOptions::max_bytes`]'s docs).
    if engine == Engine::Anydoc || engine == Engine::OfficeOxide {
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
    // resolved BEFORE the engine match: the pdf lane moves the bytes
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

/// Resolve the format: the explicit name (any leading dot tolerated) beats
/// the content markers, which beat the name hint's extension — and the
/// extension resolves through the SAME [`Kind::from_name`] vocabulary the
/// explicit name uses, so `file.xhtml`/`file.tsv`/`file.html` work by
/// extension exactly as `format="xhtml"`/`"tsv"`/`"html"` do by name, with
/// no second spelling table to keep in step (anydoc's own extension
/// vocabulary rides inside `from_name`; its `from_path` adds nothing to
/// it). `UnknownFormat` carries what was tried — the name hint when there
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
        // the data= entry: no name to consult — the fix is the explicit
        // name (or sniff() on the same bytes to see what the markers said)
        None => "unrecognized content and no name to consult: pass format= \
                 (or sniff(data) first)"
            .into(),
    };
    Err(DocumentError::UnknownFormat(what))
}

/// The content-marker sniffer: anydoc's `from_bytes` first (its markers are
/// unambiguous binary signatures), then the HTML document marker — an HTML
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

/// The HTML document marker: after a UTF-8 BOM, whitespace, and ONE
/// optional XML declaration (XHTML's `<?xml version="1.0"…?>` prologue,
/// which precedes the doctype and would hide it from a prefix-anchored
/// check), the first markup is a document type declaration or the
/// `<html>` element open (case-insensitive — both `<!DOCTYPE html…` and
/// `<HTML…` are real-world spellings). A leading comment (`<!--`) is NOT
/// taken as a marker: it is not specific to HTML, and comment-prefixed
/// non-HTML text is rare enough that the extension fallback should decide
/// it. Nor is any leading `<tag`: an HTML FRAGMENT (no doctype, no
/// `<html>`, e.g. starting `<div>`/`<p>`) is deliberately not
/// content-resolvable — every XML vocabulary opens with a tag (`<svg`,
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
            // Unterminated declaration: malformed markup, not a marker —
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

/// The CSV heuristic — the LAST resort of content resolution, after every
/// binary marker and the HTML check declined: text, not markup, where the
/// first up-to-64 non-empty lines each contain the SAME count (≥1) of one
/// delimiter candidate (`,` / `;` / TAB, tried in that order). Two lines
/// minimum: a single line of comma-separated words is prose, not a table.
/// The edge behavior, probed and pinned in the tests: a UTF-8 BOM rides
/// the first field harmlessly (it is not a delimiter); `lines()` strips a
/// trailing `\r`, so CRLF files count cleanly (a lone `\r` is NOT a line
/// ending — classic-Mac files are not heuristically CSV); quoted commas
/// that AGREE across lines route here (anydoc's quote-aware parser does
/// the real reading — the heuristic's job is routing, not parsing), while
/// quoted commas that differ are prose-shaped and rejected; a
/// single-column file carries 0 delimiters on every line and is rejected
/// (name it with `format=` if it is one); UTF-16 text fails the UTF-8
/// gate, but the `format="csv"` escape hatch reaches anydoc's BOM-aware
/// decoder. A false positive requires prose whose every line carries an
/// identical comma count — the price of resolving signature-less CSV from
/// content alone, which the no-extension routing doctrine (temp files with
/// no suffix, so content — never the name — picks the extractor) demands;
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
    // JSON-LINES is the one structured format whose comma counts AGREE
    // across lines (every record serializes the same keys), so the
    // delimiter witness alone would claim it — and anydoc's csv parser
    // would then mangle records that are not cells. A record line opens
    // with `{` (or `[`): not this heuristic's format. The cost is the
    // rare csv whose FIRST field opens with a brace — name it with
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

/// The `backend="oxide"` office conversion: office_oxide's unified reader.
/// Its `from_reader` takes the format explicitly, and anydoc's Excel kind
/// covers both the xlsx family and legacy xls, so the shared container
/// check ([`is_ole_container`] — the same one the reported name and sniff
/// use) decides that one: the OLE signature is legacy, anything else (a
/// ZIP local-file header) is OOXML. Takes the bytes OWNED: the container
/// check borrows them, then they move into the `Cursor` whole — zero
/// copies (convert() already owns the Vec; `from_reader`'s `Read + Seek +
/// 'static` bound demands ownership of the READER, not a second copy of
/// the bytes).
fn office_markdown(bytes: Vec<u8>, kind: Kind) -> Result<String, DocumentError> {
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
/// its default configuration emits for `<title>`/`<meta>` — the body is the
/// IR, and the title is not body content (a document's own `<h1>` carries
/// the identity). The UTF-8 BOM is stripped at the door: it is an encoding
/// signature, not content, and left in place it rides into the markdown as
/// an invisible U+FEFF (measured on a BOM-prefixed page: the output opened
/// with one — and a BOM-only file converted to its BOM and nothing else).
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
            // as 0-based indices everywhere, so the conversion lives here —
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
    /// web-page converter must NOT let into the body markdown, around a
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

    /// A hand-built [MS-CFB] (OLE) compound file with one named stream —
    /// the minimal container: 512-byte header, one FAT sector, one
    /// directory sector (Root Entry + the stream), then 8 sectors of junk
    /// stream data (exactly the mini-stream cutoff, so the stream lives in
    /// regular sectors and no mini FAT is needed). The byte layout was
    /// verified parseable by cfb — anydoc's OLE reader — through the
    /// installed wheel before being pinned here; the stream's CONTENT is
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

    /// A STORED (method 0, no compression) zip of the given parts — the
    /// container identity without any deflate dependency. CRC-32 is
    /// computed bitwise (no crc crate in the tree); correctness matters
    /// only so the archive reads as well-formed to anydoc's package
    /// reader, which keys on part NAMES, never on file content.
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

    /// A minimal readable two-page PDF — the same hand-built object-graph
    /// byte layout `pdf_impl`'s test module writes (correct offsets, xref,
    /// trailer; THAT module's pins defend the layout — this one only needs
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
        // GFM table with a delimiter row — the bar the chunker keys on.
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
        // case-insensitive and whitespace-tolerant — the docstring's
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
        // and the counts must AGREE and be non-zero on every line: prose
        // whose comma counts vary is not a table. (Prose whose counts never
        // vary IS the documented false positive — the accepted price of
        // content-only resolution — and is deliberately NOT asserted
        // against: the contract is the delimiter agreement, not prose
        // immunity.)
        assert_eq!(sniff(b"one, line\nand another here\n"), None);
        assert_eq!(sniff(b"one, line\ntwo commas, here, too\n"), None);
        // non-UTF-8 is not CSV text
        assert_eq!(sniff(b"\xff\xfe,\x00\n\x01,\x02\n"), None);
    }

    #[test]
    fn json_lines_is_not_csv_no_matter_how_the_commas_agree() {
        // The one structured format whose delimiter counts AGREE across
        // lines (every record serializes the same keys): the witness
        // alone would claim it, and anydoc's csv parser would mangle the
        // records. The record-open guard ({ or [) declines it; the fix is
        // the caller's — json-lines is deliberately NOT a documents
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
        // the override; under -> converts. The DEFAULT's value is pinned
        // separately (the constant IS the policy). At this tiny scale the
        // sizes print as RAW BYTES (the sub-0.1-MiB regime, where the old
        // MiB rounding read "0.0 MiB … 0.0 MiB" — two sizes, neither
        // legible) and must be TWO DISTINCT legible numbers.
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
        // office_oxide had NO input bound of its own (measured 2026-09,
        // against 0.1.9 — which had no decompression caps at all: a 333
        // KiB zip-bombed docx peaked at 1.7 GiB RSS on that lane, before
        // the ceiling was extended to it), so the opt-in lane gets the
        // same input-side guard as the default one — named for ITS
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
        // The anydoc PDF lane's reader takes no password at all: a CORRECT
        // password would fail Convert("document is encrypted") exactly
        // like a wrong one — indistinguishable from having been ignored.
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
        // password, auto routing — an unencrypted document ignores the
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
        // and the errors NAME their input — the bogus format string, the
        // name — so the caller can tell which one to fix; the data= shape
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
        // the name spells it too — and only the real extension: "xhtm" is
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
        // The skip is ONE optional declaration, not "any leading XML":
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
        // FULL-document only, because a leading `<tag` would claim every
        // XML vocabulary. An extensionless fragment is UnknownFormat; the
        // escape hatches — format="html" and the .html-family extension —
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
        // Zero bytes, whitespace only, BOM only: no marker, no extension —
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
        // parser yields an empty document (empty output — a zero-record
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
        // BOM-prefixed page — the output OPENED with one). Stripped at the
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
        // AGREE route to the csv kind (anydoc's quote-aware parser does the
        // real reading — routing, not parsing, is the heuristic's job)
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
        // UTF-16 text fails the UTF-8 gate — but the format= escape hatch
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
        // full-document marker) both convert by extension — where a second
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
        // Probed 2026-09-09 (all four through the wheel before being
        // pinned here): the extension is the text AFTER THE LAST DOT of
        // the whole hint — which handles the hidden file (`.html`'s last
        // dot is its first), the uppercase spelling (case-insensitive
        // vocabulary), and declines the pseudo-extensions: a dot inside a
        // DIRECTORY component (`dir.d/file` — "d/file" names nothing) and
        // a real-but-unknown one (`archive.tar.gz` — "gz" names nothing),
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
        // The wiring pin for the per-engine entity flag (probed 2026-09-09
        // through the wheel): anydoc escapes entity-shaped `&` in its
        // markdown (`a &amp; b` cell -> `a &amp;amp; b`), so the anydoc
        // lane's strip un-escapes ONCE — the document's own text comes
        // back. pdf_oxide emits text literally, so its lane never
        // un-escapes — a PDF whose text layer genuinely contains `&amp;`
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
        // Box<dyn Error>/anyhow; the STRINGS are pinned byte-equal to the
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
    fn adversarial_containers_sniff_and_fail_honestly() {
        // A truncated header is still a PDF header: sniff says pdf (the
        // marker is honest about what the bytes open with), and the
        // conversion then fails on the REAL reason downstream — pdf_oxide's
        // parse — not on a misdetected format.
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
        // container is legacy xls, a ZIP-based one is the xlsx family —
        // sniff and to_markdown cannot disagree about what the bytes are.
        let workbook = ole_with_stream("Workbook");
        assert_eq!(sniff(&workbook), Some("xls"));
        let zip_workbook = zip_of(&[("xl/workbook.xml", b"<workbook/>")]);
        assert_eq!(sniff(&zip_workbook), Some("xlsx"));
        // resolution works even when the parse of junk BIFF bytes cannot:
        // named .xlsx, the OLE container still picks the Excel kind, and
        // the failure is anydoc's honest "not a readable workbook" — never
        // a misdetected format (the mislabeled-extension doctrine).
        let err = convert_err("workbook.xlsx", &workbook);
        assert!(matches!(err, DocumentError::Convert(_)), "got {err:?}");
    }
}
