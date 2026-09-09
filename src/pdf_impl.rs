//! PDF text/markdown extraction over `pdf_oxide`, the core behind the
//! `pdf_extract`/`pdf_page_count`/`pdf_classify`/`pdf_link_uris`/`pages=`
//! surfaces of the `tors-documents` payload crate.
//!
//! What this layer is FOR: one place where the PDF library lives. The pyo3
//! bindings resolve the input (a `path=` they read, or the caller's `data=`
//! bytes) and release the GIL; the *choice* of extractor — pdf_oxide, its
//! conversion options, the per-page + whole-document composition, the
//! page-subset join, the text-vs-image preflight, the link-annotation walk —
//! is decided here and nowhere else, so a future swap of the underlying
//! library touches this file alone (the same seam role every other
//! `*_impl` module plays for its own algorithm). Bytes-in, answers-out: the
//! payload owns the file read; this seam never touches the filesystem.
//!
//! Why pdf_oxide and not the alternatives that were measured against it
//! (2026-09, on this box, over hand-built born-digital/two-column/link-annot
//! fixtures plus a 9-page synthetic document):
//!
//! - **pypdf** (the incumbent in the caller that motivated this surface):
//!   pure-Python, GIL-held at bytecode granularity, no structure output, no
//!   link preservation, merges visual lines across column gaps. Measured
//!   13.1ms against pdf_oxide's 3.5ms on the same 9-page document.
//! - **pypdfium2/pdfium**: native but NOT thread-safe (its own bindings
//!   document this; downstream users wrap every call in a global lock), and
//!   the shipped wheel holds the GIL for each whole-document call — measured
//!   worst heartbeat gap 23.6ms on the 9-pager under a 10ms ping, growing
//!   with document size, because the binding never calls `Python::detach`.
//! - **pdf_oxide**: pure-Rust, `Send + Sync` document type, 100% pass rate on
//!   the veraPDF/pdf.js/SafeDocs corpora (3,830 PDFs) per its benchmark
//!   methodology, MIT OR Apache-2.0, and — decisive for tors — a plain Rust
//!   API this crate calls directly under `py.detach`, so the whole
//!   open+extract+convert pass is GIL-free rather than GIL-held.
//!
//! The composition, and why exactly these outputs: the caller (a
//! knowledge-indexing extractor) needs (a) a per-page plain-text PROBE —
//! "does this page carry a text layer at all" — to route scanned PDFs to OCR
//! instead of indexing them as empty, and (b) the document's IR for
//! chunking/embedding. For (a), `extract_text(page)` is pdf_oxide's per-page
//! plain-text surface (glyph-walk assembly), and `classify_document()` is
//! the cheap whole-document preflight (no OCR, no rasterization) that names
//! the image-only pages outright. For (b), `to_markdown_all` with default
//! `ConversionOptions` — measured to be the structure-preserving choice:
//! heading detection on, images off (no base64 bloat in IR),
//! StructureTreeFirst reading order falling back to XY-Cut on untagged PDFs,
//! and `/Link` annotations rendered as `[text](uri)`. The plain-text
//! whole-document surface was measured and REJECTED for (b): it joins text
//! across column gaps line-by-line, interleaving two columns into single
//! visual lines, where the markdown converter's reading-order pass keeps the
//! columns as separate blocks.
//!
//! The page-subset conversion (`markdown_pages`) joins per-page markdown
//! with pdf_oxide's own inter-page separator — measured: `to_markdown_all`
//! IS that join — so a full document range selected through `pages=`
//! reproduces the whole-document conversion byte-for-byte.
//!
//! # One upstream robustness fact, measured and on record
//!
//! pdf_oxide 0.3.78's object parser (`parse_object` in its parser.rs) is
//! mutually recursive with its array/dictionary parsers and has no depth
//! cap — its `max_nesting: 100` parser-config field is dead code (zero
//! uses outside parser_config.rs) — so a 60,336-byte PDF carrying a
//! 30,000-deep nested array in its trailer SIGSEGVs every entry point
//! that opens the document (pdf_page_count, pdf_extract, pdf_classify,
//! pdf_link_uris, to_markdown, to_text — all share this seam's `open()`;
//! reproduced 2026-09-09 on pdf_page_count, to_markdown, and
//! pdf_classify: exit -11, uncatchable). The crash is upstream, not
//! fixable at this seam — a tors-side parser pre-scan would mean
//! duplicating pdf_oxide's parser to defend against one defect — and is
//! tracked by the new fuzz target, with the real fix (a depth cap in
//! pdf_oxide's parser) to be reported upstream.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::document::PdfDocument;
// Public: the payload crate's error mapping and `documents_impl`'s
// `PagesError` both name this type through `pdf_impl`.
pub use pdf_oxide::Error as PdfError;
use pdf_oxide::{AnnotationSubtype, LinkAction};

/// One `pdf_extract` result: the per-page plain text and the whole-document
/// markdown, as the binding marshals them.
pub struct PdfExtract {
    /// `extract_text(page)` per page, page order — the text-layer probe
    /// surface. Empty-string pages (image-only/scanned) are kept so the
    /// per-page average is honest, and a zero-page document yields an empty
    /// vec, which the caller reads as "no text layer at all".
    pub pages: Vec<String>,
    /// `to_markdown_all` over the whole document: headings, lists, tables,
    /// and `[text](uri)` links, default conversion options.
    pub markdown: String,
}

/// One page's preflight verdict, pdf_oxide's classification mapped onto
/// this seam's own vocabulary (the engine's type never crosses the seam).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageKind {
    /// Usable native text dominates — extract the text layer.
    TextLayer,
    /// Image-dominated, no/garbled text — OCR the page.
    Scanned,
    /// Native text AND image regions containing text — hybrid.
    ImageText,
    /// Heterogeneous within the page (text + image-table/figure).
    Mixed,
    /// Blank/near-empty — neither extract nor OCR; not an error. DISTINCT
    /// from [`PageKind::Scanned`]: a blank page is not an image to recover,
    /// and pdf_oxide's `pages_needing_ocr` deliberately excludes it.
    Empty,
}

impl PageKind {
    /// The stable wire name (the payload exposes the kinds as strings).
    pub fn name(self) -> &'static str {
        match self {
            PageKind::TextLayer => "text",
            PageKind::Scanned => "scanned",
            PageKind::ImageText => "image_text",
            PageKind::Mixed => "mixed",
            PageKind::Empty => "empty",
        }
    }

    fn from_engine(kind: pdf_oxide::extractors::auto::PageKind) -> PageKind {
        match kind {
            pdf_oxide::extractors::auto::PageKind::TextLayer => PageKind::TextLayer,
            pdf_oxide::extractors::auto::PageKind::Scanned => PageKind::Scanned,
            pdf_oxide::extractors::auto::PageKind::ImageText => PageKind::ImageText,
            pdf_oxide::extractors::auto::PageKind::Mixed => PageKind::Mixed,
            pdf_oxide::extractors::auto::PageKind::Empty => PageKind::Empty,
            // pdf_oxide's PageKind is #[non_exhaustive]: a future kind maps
            // conservatively — no text claimed (has_text stays false) and
            // no OCR fabricated (the OCR list comes from pdf_oxide's own
            // classify_document, never derived here), which is exactly the
            // Empty verdict's contract: neither extract nor OCR.
            _ => PageKind::Empty,
        }
    }

    /// Whether the page carries any native text (the `has_text` rule).
    pub fn has_text(self) -> bool {
        matches!(
            self,
            PageKind::TextLayer | PageKind::ImageText | PageKind::Mixed
        )
    }
}

/// One `pdf_classify` result: the page tree's text-vs-image preflight.
pub struct Classification {
    /// Pages in the document.
    pub page_count: usize,
    /// Every page's verdict, page order — the full answer, because the
    /// OCR list alone cannot derive it (an `Empty` page is neither text
    /// nor an image to recover).
    pub page_kinds: Vec<PageKind>,
    /// The 0-based indices of image-only pages (pdf_oxide's
    /// `classify_document`, the cheap preflight — no OCR, no rasterization):
    /// empty for a born-digital document, every page for a scan, the
    /// difference for a mixed document. Blank pages are NOT listed (they
    /// are `Empty`, not `Scanned` — see [`PageKind::Empty`]). The routing
    /// rules are the CALLER's: all pages listed = "we cannot do anything
    /// with this locally", an empty list = extract, a partial list =
    /// per-page or whole-doc OCR.
    pub pages_needing_ocr: Vec<usize>,
}

/// Open the bytes, authenticating first when a password was supplied.
/// `None` keeps pdf_oxide's own doctrine: the empty password is tried at
/// open (the common "owner-encrypted, user-readable" shape), and a
/// document that stays locked fails its first content operation with
/// `EncryptedPdf` — mapped to `ValueError` by the payload, fail closed: a
/// security state is never masked as empty output. A supplied password
/// that does not unlock fails HERE, before any content work, with a
/// message that names what happened (pdf_oxide's `authenticate` returns
/// `Ok(false)` for a wrong one — not an error, so the seam makes it one).
fn open(bytes: Vec<u8>, password: Option<&str>) -> Result<PdfDocument, PdfError> {
    let doc = PdfDocument::from_bytes(bytes)?;
    if let Some(password) = password {
        if !doc.authenticate(password.as_bytes())? {
            return Err(PdfError::InvalidPdf(
                "the password did not unlock this PDF (authenticate() returned false)".into(),
            ));
        }
    } else if doc.is_encrypted() && !doc.is_authenticated() {
        // pdf_oxide's markdown and text surfaces TOLERATE a locked
        // document: a structured warning, then empty output — measured
        // 2026-09 on an RC4-128 fixture (pypdf-written, user password
        // "torque"): `to_markdown_all` and `extract_text` both returned
        // empty strings with no error, while `classify_document` raised
        // `EncryptedPdf`. An empty-string answer would mask a security
        // state as "this document has no content" — the exact shape this
        // seam's fail-closed doctrine forbids — so the door check raises
        // HERE, before any content pass, and every entry shares it.
        return Err(PdfError::EncryptedPdf);
    }
    Ok(doc)
}

/// Parse the document's bytes and extract per-page plain text plus the
/// whole-document markdown — one pass over one open document, so the parse
/// cost is paid once for both outputs. The bytes are the payload's to
/// provide: a `path=` it read, or the caller's `data=` verbatim.
pub fn extract(bytes: Vec<u8>, password: Option<&str>) -> Result<PdfExtract, PdfError> {
    let doc = open(bytes, password)?;
    let pages = doc
        .page_indices()
        .map(|page| doc.extract_text(page))
        .collect::<Result<Vec<String>, PdfError>>()?;
    let markdown = doc.to_markdown_all(&ConversionOptions::default())?;
    Ok(PdfExtract { pages, markdown })
}

/// The cheap page-count probe: parse the page tree, nothing else. For
/// callers that gate expensive downstream work on page count (the
/// docling-conversion caller's pre-flight) without paying for any content
/// extraction.
pub fn page_count(bytes: Vec<u8>, password: Option<&str>) -> Result<usize, PdfError> {
    open(bytes, password)?.page_count()
}

/// The cheap text-vs-image preflight: page count plus the image-only page
/// list, no content conversion. Encrypted documents fail closed
/// (pdf_oxide's security rule, propagated as-is): a security state is never
/// masked as "all pages empty".
pub fn classify(bytes: Vec<u8>, password: Option<&str>) -> Result<Classification, PdfError> {
    let doc = open(bytes, password)?;
    let page_count = doc.page_count()?;
    let classification = doc.classify_document()?;
    Ok(Classification {
        page_count,
        page_kinds: classification
            .pages
            .iter()
            .map(|k| PageKind::from_engine(*k))
            .collect(),
        pages_needing_ocr: classification.pages_needing_ocr,
    })
}

/// The `/Annots` link-annotation walk: for every page, the URIs of its
/// `/Subtype /Link` annotations whose action is a URI (`/A << /S /URI /URI
/// (...) >>`), in annotation order — one `Vec<String>` per page, page
/// order, empty vecs for pages without link annotations.
///
/// Why this surface exists alongside the markdown's inline
/// `[text](uri)` links: the two answer different questions. The markdown
/// carries links whose VISIBLE TEXT is worth keeping in prose; this walk
/// carries the raw URI list — including URIs behind link rectangles whose
/// text is not itself a link (a "click here" button, an image, a bare
/// rectangle) which no text-oriented rendering surfaces at all. The caller
/// that motivated it measured the difference on real resumes (their
/// `annotations.py`: 60 documents — text-layer extraction surfaced 16
/// links, the annotation walk 34; a quarter of candidates gained a
/// LinkedIn/GitHub URL no text shape would show). Deduplication and
/// canonicalization are deliberately NOT done here: the verbatim
/// engine-read list is the honest output, and every caller canonicalizes
/// differently (per-page review panels vs whole-document projections).
///
/// Scope, honestly narrow: URI actions only. `GoTo` (an in-document
/// destination) and `GoToR` (a remote FILE) are navigation, not web
/// URIs — they contribute nothing here and are not fabricated into
/// `file://` shapes. Non-link annotations (widgets, stamps, notes) are
/// skipped by the subtype filter; pdf_oxide tolerates malformed annotation
/// dictionaries (its parse skips them) rather than poisoning the page.
pub fn link_uris(bytes: Vec<u8>, password: Option<&str>) -> Result<Vec<Vec<String>>, PdfError> {
    let doc = open(bytes, password)?;
    let count = doc.page_count()?;
    let mut out = Vec::with_capacity(count);
    for page in doc.page_indices() {
        let mut uris = Vec::new();
        for annotation in doc.get_annotations(page)? {
            if annotation.subtype_enum != AnnotationSubtype::Link {
                continue;
            }
            if let Some(LinkAction::Uri(uri)) = annotation.action {
                uris.push(uri);
            }
        }
        out.push(uris);
    }
    Ok(out)
}

/// `markdown_pages`'s error: pdf_oxide's own parse/conversion failures, or
/// the `pages=` selection validation (an out-of-range index, an empty
/// selection). The binding layer maps both to the Python side's taxonomy
/// (`Pages` is a `ValueError`, `Pdf` follows `pdf_extract`'s mapping).
#[derive(Debug)]
pub enum PagesError {
    Pdf(PdfError),
    Pages(String),
}

/// The error text: the payload of whichever arm failed, verbatim — so a
/// Rust caller `?`-ing into `Box<dyn Error>`/anyhow reads the same
/// message the Python side raises (the binding maps `Pages` to
/// `ValueError(what)` and `Pdf` through pdf_oxide's own `to_string`).
impl std::fmt::Display for PagesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PagesError::Pdf(err) => write!(f, "{err}"),
            PagesError::Pages(what) => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for PagesError {}

/// pdf_oxide's own inter-page separator in `to_markdown_all` (measured:
/// joining the per-page conversions with exactly this string reproduces the
/// whole-document output byte-for-byte). Public so the payload's `pages=`
/// contract can cite it, and so the crate-side separator pin test has one
/// constant to defend.
pub const PAGE_SEPARATOR: &str = "\n---\n\n";

/// Convert in-memory PDF bytes to markdown over a page subset. `None` is
/// the whole document (`to_markdown_all`); `Some(pages)` — normalized
/// HERE (sorted, deduped) into document order, so a Rust caller's
/// arbitrary `&[1, 0]` joins as document order rather than a silently
/// out-of-order concatenation; the binding layer already delivers
/// normalized selections, this makes the core self-sufficient — converts
/// each selected page with `to_markdown(page)` and joins with
/// [`PAGE_SEPARATOR`], so `Some(&(0..n).collect::<Vec<_>>())` is
/// byte-identical to `None` (the crate-side test pins this).
pub fn markdown_pages(
    bytes: Vec<u8>,
    pages: Option<&[usize]>,
    password: Option<&str>,
) -> Result<String, PagesError> {
    let doc = open(bytes, password).map_err(PagesError::Pdf)?;
    let options = ConversionOptions::default();
    let Some(selected) = pages else {
        return doc.to_markdown_all(&options).map_err(PagesError::Pdf);
    };
    // Sort + dedup the caller's selection: duplicates would repeat a
    // page, an unsorted one would join out of document order.
    let mut selected: Vec<usize> = selected.to_vec();
    selected.sort_unstable();
    selected.dedup();
    if selected.is_empty() {
        return Err(PagesError::Pages(
            "pages= selected no pages: pass None for the whole document".into(),
        ));
    }
    let count = doc.page_count().map_err(PagesError::Pdf)?;
    if let Some(out_of_range) = selected.iter().find(|page| **page >= count) {
        return Err(PagesError::Pages(format!(
            "page index {out_of_range} is out of range for a {count}-page document (pages= is 0-based)"
        )));
    }
    let mut out = String::new();
    for (position, page) in selected.iter().enumerate() {
        if position > 0 {
            out.push_str(PAGE_SEPARATOR);
        }
        out.push_str(&doc.to_markdown(*page, &options).map_err(PagesError::Pdf)?);
    }
    Ok(out)
}

#[cfg(all(test, feature = "documents"))]
mod tests {
    use super::*;

    /// The hand-built byte-deterministic PDF writer (the same object-graph
    /// pattern as the Python suite's fixtures): correct offsets, xref,
    /// trailer.
    fn objects_pdf(objects: &[&[u8]]) -> Vec<u8> {
        let mut out = bytearray_header();
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

    fn bytearray_header() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"%PDF-1.4\n");
        out
    }

    fn content_stream(text: &str) -> Vec<u8> {
        format!("BT /F1 12 Tf 50 700 Td ({text}) Tj ET").into_bytes()
    }

    /// Two single-line pages — the `pages=` semantics' minimal honest case.
    fn two_page_pdf(first: &str, second: &str) -> Vec<u8> {
        let c1 = content_stream(first);
        let c2 = content_stream(second);
        let stream = |c: &[u8]| -> Vec<u8> {
            let mut v = format!("<< /Length {} >>\nstream\n", c.len()).into_bytes();
            v.extend_from_slice(c);
            v.extend_from_slice(b"\nendstream");
            v
        };
        let s1 = stream(&c1);
        let s2 = stream(&c2);
        objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            &s1,
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 7 0 R >>",
            &s2,
        ])
    }

    #[test]
    fn a_full_page_range_reproduces_the_whole_document_byte_for_byte() {
        let bytes = two_page_pdf("Alpha page text", "Beta page text");
        let whole = markdown_pages(bytes.clone(), None, None).unwrap();
        let full_range = markdown_pages(bytes.clone(), Some(&[0, 1]), None).unwrap();
        assert_eq!(whole, full_range, "the separator contract is broken");
        // And a subset really is a subset: page 1 alone is the tail of the
        // whole document's join.
        let only_second = markdown_pages(bytes, Some(&[1]), None).unwrap();
        assert_eq!(
            only_second,
            full_range.rsplit_once(PAGE_SEPARATOR).unwrap().1
        );
    }

    #[test]
    fn out_of_range_and_empty_selections_fail_loudly() {
        let bytes = two_page_pdf("Alpha", "Beta");
        assert!(matches!(
            markdown_pages(bytes.clone(), Some(&[2]), None),
            Err(PagesError::Pages(_))
        ));
        assert!(matches!(
            markdown_pages(bytes, Some(&[]), None),
            Err(PagesError::Pages(_))
        ));
    }

    #[test]
    fn the_selection_is_normalized_into_document_order() {
        // The core does not trust the caller's deduped/order contract: a
        // Rust-side `&[1, 0]` joins in DOCUMENT order (byte-identical to
        // the sorted selection and to the whole-document tail), and a
        // duplicated index converts its page once — never a repeated or
        // out-of-order concatenation. The binding layer already delivers
        // normalized selections; this makes the core self-sufficient.
        let bytes = two_page_pdf("Alpha page text", "Beta page text");
        let sorted = markdown_pages(bytes.clone(), Some(&[0, 1]), None).unwrap();
        assert_eq!(
            markdown_pages(bytes.clone(), Some(&[1, 0]), None).unwrap(),
            sorted
        );
        assert_eq!(
            markdown_pages(bytes.clone(), Some(&[0, 0, 1, 1]), None).unwrap(),
            sorted
        );
        // a normalized subset still behaves: page 1 alone, however spelled
        let only_second = markdown_pages(bytes.clone(), Some(&[1]), None).unwrap();
        assert_eq!(
            markdown_pages(bytes, Some(&[1, 1, 1]), None).unwrap(),
            only_second
        );
    }

    #[test]
    fn classify_names_the_image_only_pages() {
        // A contentless page (no content stream) is image-only by the
        // preflight's definition; the text page is not.
        let text_stream = {
            let c = content_stream("text page");
            let mut v = format!("<< /Length {} >>\nstream\n", c.len()).into_bytes();
            v.extend_from_slice(&c);
            v.extend_from_slice(b"\nendstream");
            v
        };
        let objects: Vec<&[u8]> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            &text_stream,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
        ];
        let classification = classify(objects_pdf(&objects), None).unwrap();
        assert_eq!(classification.page_count, 2);
        // A text page and a BLANK page: the blank page is Empty — which is
        // DISTINCT from Scanned, so the OCR list is EMPTY (a blank page is
        // not an image to recover; pdf_oxide's rule, mapped faithfully).
        assert_eq!(
            classification.page_kinds,
            vec![PageKind::TextLayer, PageKind::Empty]
        );
        assert_eq!(classification.pages_needing_ocr, Vec::<usize>::new());
    }

    /// A page whose only content is an image XObject (an uncompressed
    /// 64x64 grayscale raster, no /Filter — valid PDF, no compressor
    /// needed): the SCANNED shape, which must land in pages_needing_ocr
    /// (unlike blank, and unlike the 1x1 near-empty case the classifier
    /// legitimately reads as blank).
    #[test]
    fn classify_names_an_image_only_page_scanned_and_lists_it_for_ocr() {
        let mut image_object: Vec<u8> = b"<< /Type /XObject /Subtype /Image /Width 64 /Height 64 /ColorSpace /DeviceGray /BitsPerComponent 8 /Length 4096 >>\nstream\n".to_vec();
        image_object.extend_from_slice(&vec![0u8; 4096]);
        image_object.extend_from_slice(b"\nendstream");
        let objects: Vec<&[u8]> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length 27 >>\nstream\nq 612 0 0 792 0 0 cm /Im0 Do Q\nendstream",
            &image_object,
        ];
        let bytes = objects_pdf(&objects);
        let classification = classify(bytes.clone(), None).unwrap();
        assert_eq!(classification.page_kinds, vec![PageKind::Scanned]);
        assert_eq!(classification.pages_needing_ocr, vec![0]);
        // Nothing local can read a scan: the extract probe agrees (empty
        // text) without fabricating content.
        let extracted = extract(bytes, None).unwrap();
        assert_eq!(extracted.pages, vec![String::new()]);
    }

    /// The per-page probe + whole-document markdown, on a two-page
    /// born-digital document: page order, per-page text, and the markdown
    /// carrying both lines.
    #[test]
    fn extract_returns_per_page_text_and_whole_document_markdown() {
        let bytes = two_page_pdf("Alpha page text", "Beta page text");
        let extracted = extract(bytes, None).unwrap();
        assert_eq!(extracted.pages, vec!["Alpha page text", "Beta page text"]);
        assert!(extracted.markdown.contains("Alpha page text"));
        assert!(extracted.markdown.contains("Beta page text"));
    }

    /// The /Annots link walk: URI actions surface per page in annotation
    /// order; GoTo destinations (navigation, not web URIs) and pages
    /// without annotations contribute nothing — and the whole-document
    /// projection is just the per-page lists flattened.
    #[test]
    fn link_uris_walk_the_annotations_per_page() {
        let c1 = content_stream("Alpha page text");
        let c2 = content_stream("Beta page text");
        let stream = |c: &[u8]| -> Vec<u8> {
            let mut v = format!("<< /Length {} >>\nstream\n", c.len()).into_bytes();
            v.extend_from_slice(c);
            v.extend_from_slice(b"\nendstream");
            v
        };
        let s1 = stream(&c1);
        let s2 = stream(&c2);
        let bytes = objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R 7 0 R] /Count 2 >>",
            // page 1 carries TWO link annotations: a URI action, then a
            // GoTo destination (no /A at all)
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R /Annots [8 0 R 9 0 R] >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            &s1,
            &s2,
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 6 0 R >>",
            b"<< /Type /Annot /Subtype /Link /Rect [50 700 200 720] /A << /S /URI /URI (https://example.com/handbook) >> >>",
            b"<< /Type /Annot /Subtype /Link /Rect [50 650 200 670] /Dest [3 0 R /Fit] >>",
        ]);
        let uris = link_uris(bytes, None).unwrap();
        assert_eq!(
            uris,
            vec![
                vec!["https://example.com/handbook".to_string()],
                Vec::<String>::new(),
            ]
        );
        // and a document with no annotations at all: empty lists per page
        let plain = two_page_pdf("Alpha", "Beta");
        assert_eq!(
            link_uris(plain, None).unwrap(),
            vec![Vec::<String>::new(), Vec::<String>::new()]
        );
    }

    /// The pdfium-hazard property, as a Rust test: pdfium is not thread-safe
    /// (downstream users serialize it behind a global lock); pdf_oxide's
    /// documents are `Send + Sync`, so N threads opening their OWN document
    /// over the same bytes must all succeed with byte-identical results —
    /// real parallelism, not GIL-serialized turns (the payload's py.detach
    /// gives Python threads this same parallelism; the core proves it
    /// without any binding).
    #[test]
    fn concurrent_extractions_are_byte_identical() {
        let bytes = std::sync::Arc::new(two_page_pdf("Alpha page text", "Beta page text"));
        let expected = extract((*bytes).clone(), None)
            .map(|out| (out.pages, out.markdown))
            .unwrap();
        let mut handles = Vec::new();
        for _ in 0..8 {
            let bytes = std::sync::Arc::clone(&bytes);
            handles.push(std::thread::spawn(move || {
                extract((*bytes).clone(), None).map(|out| (out.pages, out.markdown))
            }));
        }
        for handle in handles {
            assert_eq!(handle.join().unwrap().unwrap(), expected);
        }
    }
}
