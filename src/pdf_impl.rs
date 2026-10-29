//! PDF text/markdown extraction over `pdf_oxide`, the core behind the
//! `pdf_extract`/`pdf_page_count`/`pdf_classify`/`pdf_link_uris`/`page_geometry`/
//! `pages=` surfaces of the `tors-documents` payload crate.
//!
//! What this layer is for: one place where the PDF library lives. The pyo3
//! bindings resolve the input (a `path=` they read, or the caller's `data=`
//! bytes) and release the GIL; the *choice* of extractor: pdf_oxide, its
//! conversion options, the per-page + whole-document composition, the
//! page-subset join, the text-vs-image preflight, the link-annotation walk:
//! is decided here and nowhere else, so a future swap of the underlying
//! library touches this file alone (the same seam role every other
//! `*_impl` module plays for its own algorithm). Bytes-in, answers-out: the
//! payload owns the file read; this seam never touches the filesystem.
//!
//! Why pdf_oxide and not the alternatives that were measured against it
//! (on this box, over hand-built born-digital/two-column/link-annot
//! fixtures plus a 9-page synthetic document):
//!
//! - **pypdf** (the incumbent in the caller that motivated this surface):
//!   pure-Python, GIL-held at bytecode granularity, no structure output, no
//!   link preservation, merges visual lines across column gaps. Measured
//!   13.1ms against pdf_oxide's 3.5ms on the same 9-page document.
//! - **pypdfium2/pdfium**: native but not thread-safe (its own bindings
//!   document this; downstream users wrap every call in a global lock), and
//!   the shipped wheel holds the GIL for each whole-document call: measured
//!   worst heartbeat gap 23.6ms on the 9-pager under a 10ms ping, growing
//!   with document size, because the binding never calls `Python::detach`.
//! - **pdf_oxide**: pure-Rust, `Send + Sync` document type, 100% pass rate on
//!   the veraPDF/pdf.js/SafeDocs corpora (3,830 PDFs) per its benchmark
//!   methodology, MIT or Apache-2.0, and (decisive for tors) a plain Rust
//!   API this crate calls directly under `py.detach`, so the whole
//!   open+extract+convert pass is GIL-free rather than GIL-held.
//!
//! The composition, and why exactly these outputs: the caller (a
//! knowledge-indexing extractor) needs (a) a per-page plain-text probe
//! ("does this page carry a text layer at all") to route scanned PDFs to OCR
//! instead of indexing them as empty, and (b) the document's IR for
//! chunking/embedding. For (a), `extract_text(page)` is pdf_oxide's per-page
//! plain-text surface (glyph-walk assembly), and `classify_document()` is
//! the cheap whole-document preflight (no OCR, no rasterization) that names
//! the image-only pages outright. For (b), `to_markdown_all` with default
//! `ConversionOptions`: measured to be the structure-preserving choice:
//! heading detection on, images off (no base64 bloat in IR),
//! StructureTreeFirst reading order falling back to XY-Cut on untagged PDFs,
//! and `/Link` annotations rendered as `[text](uri)`. The plain-text
//! whole-document surface was measured and rejected for (b): it joins text
//! across column gaps line-by-line, interleaving two columns into single
//! visual lines, where the markdown converter's reading-order pass keeps the
//! columns as separate blocks.
//!
//! The page-subset conversion (`markdown_pages`) joins per-page markdown
//! with pdf_oxide's own inter-page separator, measured: `to_markdown_all`
//! is that join, so a full document range selected through `pages=`
//! reproduces the whole-document conversion byte-for-byte.
//!
//! # The unbounded-recursion crash class: three in-repo layers plus upstream
//!
//! pdf_oxide 0.3.78's object parser (`parse_object`, its parser.rs:169) is
//! mutually recursive with its array/dictionary parsers (parser.rs:423,
//! parser.rs:485; two frames per nesting level) and has no depth cap: its
//! `ParserConfig::max_nesting: 100` (parser_config.rs:109) is dead code,
//! zero uses outside that file. Measured on this box (8 MiB main stack,
//! 2026-09-16): a trailer nesting `[` arrays 10,658 deep opens, 10,659
//! SIGSEGVs, and a 20,000-deep one kills every entry point that opens a
//! document (all six share this file's `open()`) with exit -11 in ~170 ms,
//! uncatchable from Python (`except BaseException` never runs). A
//! keyed-dictionary bomb (`<< /K ` per level) dies the same way at
//! N≥12,000; bare `<< << <<` (no key between the opens) does not recurse,
//! the lexer refuses it catchably. Upstream fix: commit 68da2cf8 on
//! upstream's release/v0.3.79 branch, not on crates.io at this writing.
//! Until a
//! published release carries it, `open()` runs three in-repo layers, kept
//! afterwards as defense-in-depth (a future upstream cap and these agree on
//! the 100 value by construction):
//!
//! 1. **Raw nesting scan** ([`scan_document`]): one linear, allocation-free
//!    byte walk before `PdfDocument::from_bytes`, counting `[`/`]` and
//!    `<<`/`>>` nesting (a `<<` is a dictionary open, never a hex-string
//!    open) and skipping the regions pdf_oxide's parser never recurses
//!    through: `stream…endstream` payloads, literal strings `(…)` (escape-
//!    and nesting-aware), hex strings, and `%` comments. It is a region
//!    lexer (normal, comment, literal string, hex string, stream payload)
//!    with no object model: O(bytes) time and O(1) state (a depth counter
//!    and three indices; the walk allocates nothing), so over-depth comes
//!    back as a catchable `InvalidPdf` refusal (the malformed-document
//!    `ValueError` shape) instead of signal death. The cap is
//!    [`MAX_OBJECT_NESTING`], pdf_oxide's own dead-config value. Measured
//!    (release, this box, load ~1.5): ~1.0 ms/MiB, flat across 0.2/1.6/
//!    6.5 MiB mixed-token fixtures (±15%); a depth bomb exits at the cap,
//!    ~2 µs regardless of N.
//! 2. **Compressed-object scan** (the same walk, collecting streams): a raw
//!    byte walk cannot see inside compressed streams, and pdf_oxide parses
//!    a `/Type /ObjStm` stream's contents with the same uncapped
//!    `parse_object` (its objstm.rs:168); measured, the ObjStm bomb
//!    SIGSEGVs at N=12,000 raw-equivalent. The walk therefore collects
//!    every `/Type /ObjStm` and `/Type /XRef` stream it passes, inflates
//!    the FlateDecode ones (flate2, the decoder pdf_oxide itself uses; its
//!    inflate is capped at its flate.rs:155, its recursion is not)
//!    under a hard [`MAX_INFLATED_BYTES`] ceiling, and runs the same
//!    depth scan over the inflated (or, unfiltered, raw) payload. The
//!    ceiling breach is the same catchable refusal: the scan must not open
//!    a decompression-bomb path while closing the recursion one. Measured
//!    residual of the class: pdf_oxide reads compressed xref-stream DATA as
//!    packed binary rows and never recurses into it (the xref bomb
//!    converts, pre-scan, at any depth), so the xref scan is containment
//!    for attacker-shaped junk, not a crash fix. Two engine behaviors the
//!    collection must mirror field for field, both measured as SIGSEGV
//!    bypasses before the mirror existed (2026-09-16, ~1.1 KB documents,
//!    rc=-11): pdf_oxide resolves filter names through a spec-abbreviation
//!    and case-insensitive table (its decoders/mod.rs: ISO 32000-1 Table 6,
//!    `/Fl` IS FlateDecode, and so is `/flatedecode`), and its stream
//!    reader accepts a bare `stream` keyword with a CR-only or MISSING
//!    end-of-line, starting the payload right after the keyword (its
//!    parser.rs: "SPEC VIOLATION ... Accepting in lenient mode"). The
//!    collection mirrors both spellings and all four EOL shapes, and an
//!    ObjStm/XRef stream whose dict declares a filter pdf_oxide can decode
//!    but this scan cannot (LZW, RunLength, ASCIIHex/85, the image
//!    filters, Brotli) is refused outright, the same fail-closed posture
//!    the office zip audit takes: a payload the scan cannot itself
//!    inflate is a payload whose nesting it cannot bound, and the engine's
//!    uncapped `parse_object` behind it is exactly what this layer exists
//!    to guard.
//! 3. **Stack headroom** ([`with_depth_headroom`]): every entry point runs
//!    its whole native pass on a thread with a 256 MiB stack (virtual,
//!    lazily paged) instead of the caller's 8 MiB main stack, and a
//!    panicked `JoinHandle::join` maps to the same catchable `InvalidPdf`.
//!    The parser's measured cost is ~790 bytes of stack per nesting level
//!    (8 MiB dies at 10,659 levels), so the kill point moves from ~10.7k
//!    levels to ~341k, and a pdf_oxide panic becomes a `ValueError` instead
//!    of process death. Measured containment cost: ~22 µs spawn+join, ~34 µs
//!    end-to-end on the two-page fixture (release, this box), noise
//!    against any parse. One on-record limit of any linear scan: a document
//!    crafted to lie about its stream boundaries (a `>>` glued to
//!    `endstream`) can hide structure from the walks; this layer is that
//!    shape's backstop.
//!
//! When pdf_oxide ships the real cap (0.3.79+, commit 68da2cf8), the layers
//! stay: they are the seam's own memory-safety floor, not lane policy; they
//! run regardless of `max_bytes` (the scans never read `ConvertOptions` or
//! any ceiling), cost the one linear pass plus ~22 µs of containment
//! measured above, and cover future regressions in the same class. The
//! `documents_markdown` fuzz target remains the repro harness for the
//! upstream defect and stays out of every run list until 0.3.79 publishes.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::document::PdfDocument;
// Public: the payload crate's error mapping and `documents_impl`'s
// `PagesError` both name this type through `pdf_impl`.
pub use pdf_oxide::Error as PdfError;
use pdf_oxide::{AnnotationSubtype, LinkAction};

/// One `pdf_extract` result: the per-page plain text and the whole-document
/// markdown, as the binding marshals them.
pub struct PdfExtract {
    /// `extract_text(page)` per page, page order: the text-layer probe
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
    /// Usable native text dominates: extract the text layer.
    TextLayer,
    /// Image-dominated, no/garbled text: OCR the page.
    Scanned,
    /// Native text and image regions containing text: hybrid.
    ImageText,
    /// Heterogeneous within the page (text + image-table/figure).
    Mixed,
    /// Blank/near-empty: neither extract nor OCR; not an error. Distinct
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
            // conservatively: no text claimed (has_text stays false) and
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
    /// Every page's verdict, page order: the full answer, because the
    /// OCR list alone cannot derive it (an `Empty` page is neither text
    /// nor an image to recover).
    pub page_kinds: Vec<PageKind>,
    /// The 0-based indices of image-only pages (pdf_oxide's
    /// `classify_document`, the cheap preflight: no OCR, no rasterization):
    /// empty for a born-digital document, every page for a scan, the
    /// difference for a mixed document. Blank pages are not listed (they
    /// are `Empty`, not `Scanned`: see [`PageKind::Empty`]). The routing
    /// rules are the caller's: all pages listed = "we cannot do anything
    /// with this locally", an empty list = extract, a partial list =
    /// per-page or whole-doc OCR.
    pub pages_needing_ocr: Vec<usize>,
}

/// The object-nesting ceiling the scans enforce: pdf_oxide 0.3.78's own
/// `ParserConfig::max_nesting` default (its parser_config.rs:109, "PDF spec
/// recommended limit"), dead code upstream, enforced here instead. When
/// upstream ships the parser cap (commit 68da2cf8, 0.3.79+),
/// the two agree on this value by construction; the const stays as
/// defense-in-depth either way.
const MAX_OBJECT_NESTING: usize = 100;

/// The hard ceiling on inflating one compressed stream for the scan (a
/// `/Type /ObjStm` or `/Type /XRef` payload). Object streams hold only
/// non-stream objects (dictionaries, page trees: the spec forbids streams
/// inside them), so a legitimate one inflating past this implies a file far
/// beyond the 512 MiB input backstop; the breach is refused catchably (the
/// same `InvalidPdf` shape as every other scan refusal) so the depth
/// defense cannot itself be turned into a decompression bomb: the scan
/// never holds more than this many bytes per stream.
const MAX_INFLATED_BYTES: u64 = 256 * 1024 * 1024;

/// The stack pdf_oxide's parser runs on: 256 MiB, virtual and lazily paged
/// (the resident cost is what a parse actually touches). The recursion that
/// kills the process costs two frames per nesting level
/// (`parse_object` → `parse_array`/`parse_dictionary`, parser.rs:169/:423/
/// :485), so the stack is the crash surface: the caller's 8 MiB main thread
/// dies at ~10.7k levels (measured, module docs above) and this holds
/// ~32x more. A panic unwinds into `JoinHandle::join`, mapped to the same
/// catchable `InvalidPdf` refusal as the scans (see
/// [`with_depth_headroom`]) instead of process death.
const PARSER_STACK_BYTES: usize = 256 * 1024 * 1024;

fn nesting_refusal() -> PdfError {
    PdfError::InvalidPdf(format!(
        "the document nests arrays/dictionaries more than {MAX_OBJECT_NESTING} levels deep: \
         pdf_oxide 0.3.78's object parser has no depth cap (its ParserConfig::max_nesting is \
         dead code; the parser cap lands upstream in 0.3.79, commit 68da2cf8) and would exhaust \
         the native stack and kill the process, so the document is refused here, before parsing"
    ))
}

/// The walks' one refusal for a compressed stream that inflates past
/// [`MAX_INFLATED_BYTES`].
fn inflated_size_refusal() -> PdfError {
    PdfError::InvalidPdf(format!(
        "a compressed object stream inflates past the {MAX_INFLATED_BYTES}-byte scan ceiling: \
         refused before parsing (the ceiling bounds the scan's own memory, so the nesting \
         defense cannot be turned into a decompression bomb)"
    ))
}

/// Does `haystack` contain `needle`? (The dict slices are short; `windows`
/// on an empty haystack is empty, so a malformed empty dict never matches.)
fn bytes_contain(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

/// The `stream` keyword, only where a reader can mean it: the bytes
/// `stream`, preceded over whitespace by a `>` (the dictionary close a
/// stream object's dict always ends with). This is what keeps the word
/// inside `endstream` (preceded by `d`), inside a name (`/stream`), or
/// inside ordinary content from ever triggering the payload skip; a false
/// trigger would swallow real structure and hide a bomb from the scan.
/// Returns the position of that `>` and the payload start. The payload
/// start mirrors pdf_oxide's own reader field for field (its parser.rs
/// parse_stream_data): a CRLF or LF is skipped, a lone CR is skipped
/// ("SPEC VIOLATION ... Accepting in lenient mode", its words), and NO
/// end-of-line at all leaves the payload starting right after the keyword;
/// the last two were measured as SIGSEGV bypasses of this scan before
/// the mirror existed (an ObjStm bomb behind a CR-only or missing EOL
/// sailed past the collection into the uncapped parse). Only a reader
/// that puts whitespace BEFORE the end-of-line refuses on its own (its
/// decompressor chokes on the space, catchably), so that one shape stays
/// uncollected here and dies the same catchable death there.
fn stream_keyword_extent(bytes: &[u8], i: usize) -> Option<(usize, usize)> {
    if !bytes[i..].starts_with(b"stream") {
        return None;
    }
    let mut j = i;
    while j > 0 && matches!(bytes[j - 1], b' ' | b'\t' | b'\r' | b'\n' | b'\0' | b'\x0c') {
        j -= 1;
    }
    if j == 0 || bytes[j - 1] != b'>' {
        return None;
    }
    let after = i + b"stream".len();
    let skip = match bytes.get(after) {
        Some(b'\n') => 1,
        Some(b'\r') if bytes.get(after + 1) == Some(&b'\n') => 2,
        Some(b'\r') => 1,
        _ => 0,
    };
    Some((j - 1, after + skip))
}

/// The `endstream` that terminates a payload (pdf_oxide reads `/Length`,
/// the scan does not need it: both stop here). Never found (a liar
/// document): the payload runs to EOF and the walk ends.
fn endstream_at(bytes: &[u8], from: usize) -> usize {
    let mut k = from;
    while k + b"endstream".len() <= bytes.len() {
        if bytes[k..k + b"endstream".len()] == *b"endstream" {
            return k;
        }
        k += 1;
    }
    bytes.len()
}

/// Layer 2: one stream object the walk collected, `/Type /ObjStm` or
/// `/Type /XRef` in its dict (an object stream's contents parse with the
/// same uncapped `parse_object` as raw bytes (pdf_oxide's objstm.rs:168),
/// which is where the bomb class hides from a raw scan; a compressed xref
/// stream's data is read as packed binary rows and never recurses, so its
/// scan is containment for attacker-shaped junk). FlateDecode payloads:
/// under pdf_oxide's full spelling vocabulary ([`dict_declares_flate`]),
/// are inflated under [`MAX_INFLATED_BYTES`] first; a dict declaring a
/// filter pdf_oxide can decode but this scan cannot is REFUSED (the
/// engine would inflate what the scan cannot see into, and the uncapped
/// parser behind it is what this layer guards: the fail-closed posture
/// the office zip audit takes); unfiltered payloads are scanned as-is (an
/// object stream needs no filter). The nested walk collects no streams of
/// its own: the spec forbids streams inside object streams, and unbounded
/// nesting of the scan itself would be the bug it exists to close.
fn scan_stream_object(dict: &[u8], payload: &[u8]) -> Result<(), PdfError> {
    if !bytes_contain(dict, b"/ObjStm") && !bytes_contain(dict, b"/XRef") {
        return Ok(());
    }
    if let Some(filter) = dict_declares_undecodable_filter(dict) {
        return Err(PdfError::InvalidPdf(format!(
            "this document's object stream declares the {filter} filter: this pre-scan \
             inflates FlateDecode and unfiltered object streams only, because the engine \
             behind it parses their contents with an uncapped recursive object parser and \
             this scan must bound what reaches it: refused as malformed"
        )));
    }
    let inflated;
    let scanned: &[u8] = if dict_declares_flate(dict) {
        inflated = inflate_capped(payload)?;
        &inflated
    } else {
        payload
    };
    scan_walk(scanned, false)
}

/// Whether a stream's dict declares the FlateDecode filter under
/// pdf_oxide's full resolution vocabulary (its decoders/mod.rs
/// `normalize_filter_name`): the exact or any-cased full name (its
/// case-insensitive fallback), or the spec's Table 6 abbreviation `Fl`
/// spelled exactly and standing as a whole name (a `/Fl` inside an array
/// of filters qualifies; a longer name that merely begins with `Fl`
/// (`/Flags`, `/Filter`) does not: the byte after the abbreviation must
/// be a delimiter or whitespace, which is where a PDF name ends).
/// Measured before this vocabulary existed: `/Fl` and `/flatedecode`
/// object-stream bombs passed this scan un-inflated and died as SIGSEGV
/// in the engine's parser.
fn dict_declares_flate(dict: &[u8]) -> bool {
    bytes_contain_ignore_ascii_case(dict, b"flatedecode") || name_declared(dict, b"Fl")
}

/// The filter names pdf_oxide resolves and decodes (its decoders/mod.rs:
/// the exact names, the Table 6 abbreviations, the case-insensitive
/// fallback) that this scan has no decoder for. A dict declaring one of
/// these on an ObjStm/XRef stream is refused: the engine would hand the
/// inflated bytes to the same uncapped `parse_object` this layer exists
/// to guard, and the scan cannot bound bytes it cannot inflate.
const UNDECODABLE_FILTERS: &[&str] = &[
    "ASCIIHexDecode",
    "ASCII85Decode",
    "LZWDecode",
    "RunLengthDecode",
    "CCITTFaxDecode",
    "DCTDecode",
    "JBIG2Decode",
    "JPXDecode",
    "BrotliDecode",
];

/// The Table 6 abbreviations of [`UNDECODABLE_FILTERS`], each paired with
/// the full name the refusal message names: exact-case, whole-name tokens
/// (`/LZW` inside a filter array qualifies; `/LZWDecode` is the full name
/// above).
const UNDECODABLE_ABBREVIATIONS: &[(&[u8], &str)] = &[
    (b"AHx", "ASCIIHexDecode"),
    (b"A85", "ASCII85Decode"),
    (b"LZW", "LZWDecode"),
    (b"RL", "RunLengthDecode"),
    (b"CCF", "CCITTFaxDecode"),
    (b"DCT", "DCTDecode"),
];

/// The first undecodable filter a dict declares, if any: full names
/// case-insensitively (pdf_oxide's fallback resolves every casing), and
/// the abbreviations as whole names. Returns the declared spelling for
/// the refusal message (the full name, whichever spelling declared it).
fn dict_declares_undecodable_filter(dict: &[u8]) -> Option<&'static str> {
    for name in UNDECODABLE_FILTERS {
        if bytes_contain_ignore_ascii_case(dict, name.as_bytes()) {
            return Some(name);
        }
    }
    UNDECODABLE_ABBREVIATIONS
        .iter()
        .find(|(abbr, _)| name_declared(dict, abbr))
        .map(|(_, full)| *full)
}

/// Whether `name` (all ASCII, exact case) appears in the dict as a whole
/// PDF name: a `/` immediately before it, and a byte after it that ends a
/// name (whitespace or a delimiter; anything else continues the name,
/// so `/Flags` never declares `/Fl`).
fn name_declared(dict: &[u8], name: &[u8]) -> bool {
    let mut probe = 0;
    while let Some(hit) = dict[probe..]
        .windows(name.len() + 1)
        .position(|window| &window[..1] == b"/" && &window[1..] == name)
    {
        let at = probe + hit;
        probe = at + 1;
        let after = at + 1 + name.len();
        let ends_name = match dict.get(after) {
            None => true,
            Some(&b) => {
                matches!(
                    b,
                    b'\0'
                        | b'\t'
                        | b'\n'
                        | b'\x0c'
                        | b'\r'
                        | b' '
                        | b'('
                        | b')'
                        | b'<'
                        | b'>'
                        | b'['
                        | b']'
                        | b'{'
                        | b'}'
                        | b'/'
                        | b'%'
                )
            }
        };
        if ends_name {
            return true;
        }
    }
    false
}

/// `bytes_contain`, case-insensitive over ASCII (pdf_oxide resolves
/// filter names through a lowercase fallback, so `/FLATEDECODE` and
/// `/flatedecode` are the same filter to it).
fn bytes_contain_ignore_ascii_case(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window.eq_ignore_ascii_case(needle))
}

/// Inflate one zlib flow under the hard ceiling: `take(ceiling + 1)` stops
/// the read one byte past the limit, so the breach is visible as a length
/// without ever holding more than ceiling + 1 bytes. A corrupt flow returns
/// an empty buffer rather than an error: pdf_oxide's own inflate fails the
/// same way (its FlateDecode path is capped too, its flate.rs:155) and its
/// error is already the catchable shape.
fn inflate_capped(payload: &[u8]) -> Result<Vec<u8>, PdfError> {
    use std::io::Read;
    let mut inflated = Vec::new();
    let mut decoder = flate2::read::ZlibDecoder::new(payload).take(MAX_INFLATED_BYTES + 1);
    if decoder.read_to_end(&mut inflated).is_err() {
        return Ok(Vec::new());
    }
    if inflated.len() as u64 > MAX_INFLATED_BYTES {
        return Err(inflated_size_refusal());
    }
    Ok(inflated)
}

/// The walk itself. `collect_streams` marks the top-level pass (layers 1+2
/// together); the nested pass over an inflated payload is depth-only.
fn scan_walk(bytes: &[u8], collect_streams: bool) -> Result<(), PdfError> {
    let n = bytes.len();
    let mut i = 0usize;
    let mut depth = 0usize;
    // The outermost `<<` since its last close: (start, end-after->>), so a
    // `stream` keyword can look up the dict it belongs to. The extent check
    // at the keyword (dict end == the guard's `>` + 1) is what discards a
    // stale dict: the one recorded must be the dict that closed immediately
    // before the keyword, or the stream is not scanned here (a stream
    // without its dict is invalid PDF that pdf_oxide refuses
    // catchably on its own).
    let mut dict_open_at = 0usize;
    let mut dict: Option<(usize, usize)> = None;
    while i < n {
        match bytes[i] {
            // A comment runs to EOL: pdf_oxide's lexer drops it too.
            b'%' => {
                i += 1;
                while i < n && bytes[i] != b'\n' && bytes[i] != b'\r' {
                    i += 1;
                }
            }
            // A literal string: balanced parens, backslash escapes the next
            // byte. pdf_oxide never recurses into one; neither does this.
            b'(' => {
                i += 1;
                let mut parens = 1usize;
                while i < n && parens > 0 {
                    match bytes[i] {
                        b'\\' => i += 1,
                        b'(' => parens += 1,
                        b')' => parens -= 1,
                        _ => {}
                    }
                    i += 1;
                }
            }
            // A dictionary open (`<<`, two bytes: a lone `<` is a hex string).
            b'<' if bytes.get(i + 1) == Some(&b'<') => {
                depth += 1;
                if depth > MAX_OBJECT_NESTING {
                    return Err(nesting_refusal());
                }
                if depth == 1 {
                    dict_open_at = i;
                }
                i += 2;
            }
            // A hex string `<…>`: a token, not nesting; skip to its close.
            b'<' => {
                i += 1;
                while i < n && bytes[i] != b'>' {
                    i += 1;
                }
                i += 1;
            }
            b'>' if bytes.get(i + 1) == Some(&b'>') => {
                if depth == 1 {
                    dict = Some((dict_open_at, i + 2));
                }
                depth = depth.saturating_sub(1);
                i += 2;
            }
            // A lone `>` (a hex close handled at its `<`, or stray junk).
            b'>' => i += 1,
            b'[' => {
                depth += 1;
                if depth > MAX_OBJECT_NESTING {
                    return Err(nesting_refusal());
                }
                i += 1;
            }
            // A close below zero floors there: junk in a binary region
            // (xref gaps, comment tails) can never manufacture depth.
            b']' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            b's' if bytes[i..].starts_with(b"stream") => {
                let Some((close_end, payload_start)) = stream_keyword_extent(bytes, i) else {
                    i += 1;
                    continue;
                };
                let payload_end = endstream_at(bytes, payload_start);
                if collect_streams
                    && let Some((dict_start, dict_end)) =
                        dict.filter(|&(_, end)| end == close_end + 1)
                {
                    scan_stream_object(
                        &bytes[dict_start..dict_end],
                        &bytes[payload_start..payload_end],
                    )?;
                }
                i = payload_end;
            }
            _ => i += 1,
        }
    }
    Ok(())
}

/// Layers 1+2 over one document's bytes: the raw nesting walk, plus the
/// compressed-object scan for every `/Type /ObjStm` and `/Type /XRef`
/// stream it passes. Runs before `PdfDocument::from_bytes` in
/// [`open`], inside the bindings' `py.detach` regions: native work on
/// native bytes, the error surfacing only through the post-detach mapping
/// (`pdf_error` in the payload crate) as the malformed-document
/// `ValueError`.
fn scan_document(bytes: &[u8]) -> Result<(), PdfError> {
    scan_walk(bytes, true)
}

/// The parser containment layer: run one entry point's whole native pass on
/// a thread with the 256 MiB stack ([`PARSER_STACK_BYTES`]) instead of the
/// caller's 8 MiB main stack, so any recursion path the scans missed needs
/// ~32x deeper nesting to be lethal (the measured ~790 bytes of stack per
/// level: 8 MiB dies at 10,659), and map a panicked
/// `JoinHandle::join` to the same catchable `InvalidPdf` the scans raise
/// (`from_pdf` adapts it for the one entry point whose error type wraps
/// `PdfError`). The spawned thread never touches the GIL: every call site
/// runs inside the bindings' `py.detach` regions, so this is native work on
/// native bytes; spawn failure (an exhausted OS thread budget) is an
/// environment failure, mapped to `PdfError::Io` → `OSError`, never a
/// "malformed document" claim.
fn with_depth_headroom<T, E>(
    job: impl FnOnce() -> Result<T, E> + Send + 'static,
    from_pdf: fn(PdfError) -> E,
) -> Result<T, E>
where
    T: Send + 'static,
    E: Send + 'static,
{
    let joined = std::thread::Builder::new()
        .stack_size(PARSER_STACK_BYTES)
        .spawn(job)
        .map_err(|io| {
            from_pdf(PdfError::Io(std::io::Error::other(format!(
                "could not spawn the PDF parser thread: {io}"
            ))))
        })?;
    joined.join().unwrap_or_else(|payload| {
        Err(from_pdf(PdfError::InvalidPdf(format!(
            "the PDF engine panicked while reading this document ({}): refused as malformed",
            panic_text(payload)
        ))))
    })
}

/// The panic payload's text, or a placeholder for a non-string one (a
/// panic's payload is `Any`; the two string shapes are what `panic!` with a
/// formatted message produces).
fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<&'static str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// The identity adaptor: the four PDF-only entry points' error type is
/// already `PdfError`.
fn pdf_error_identity(err: PdfError) -> PdfError {
    err
}

/// The one shared doorway, every entry point's first native step (module
/// docs above): the depth scans run before pdf_oxide touches the bytes:
/// [`scan_document`] refuses the over-nested and compressed-bomb shapes
/// catchably, then the parse itself runs under the containment layer's
/// stack. `None` keeps pdf_oxide's own doctrine: the empty password is
/// tried at open (the common "owner-encrypted, user-readable" shape), and a
/// document that stays locked fails its first content operation with
/// `EncryptedPdf`: mapped to `ValueError` by the payload, fail closed: a
/// security state is never masked as empty output. A supplied password
/// that does not unlock fails here, before any content work, with a
/// message that names what happened (pdf_oxide's `authenticate` returns
/// `Ok(false)` for a wrong one: not an error, so the seam makes it one).
fn open(bytes: Vec<u8>, password: Option<&str>) -> Result<PdfDocument, PdfError> {
    scan_document(&bytes)?;
    let doc = PdfDocument::from_bytes(bytes)?;
    if let Some(password) = password {
        if !doc.authenticate(password.as_bytes())? {
            return Err(PdfError::InvalidPdf(
                "the password did not unlock this PDF (authenticate() returned false)".into(),
            ));
        }
    } else if doc.is_encrypted() && !doc.is_authenticated() {
        // pdf_oxide's markdown and text surfaces tolerate a locked
        // document: a structured warning, then empty output, measured
        // on an RC4-128 fixture (pypdf-written, user password
        // "torque"): `to_markdown_all` and `extract_text` both returned
        // empty strings with no error, while `classify_document` raised
        // `EncryptedPdf`. An empty-string answer would mask a security
        // state as "this document has no content": the exact shape this
        // seam's fail-closed doctrine forbids, so the door check raises
        // here, before any content pass, and every entry shares it.
        return Err(PdfError::EncryptedPdf);
    }
    Ok(doc)
}

/// Parse the document's bytes and extract per-page plain text plus the
/// whole-document markdown: one pass over one open document, so the parse
/// cost is paid once for both outputs. The bytes are the payload's to
/// provide: a `path=` it read, or the caller's `data=` verbatim. The whole
/// pass (scans, parse, extraction) runs under [`with_depth_headroom`].
pub fn extract(bytes: Vec<u8>, password: Option<&str>) -> Result<PdfExtract, PdfError> {
    let password = password.map(str::to_string);
    with_depth_headroom(
        move || {
            let doc = open(bytes, password.as_deref())?;
            let pages = doc
                .page_indices()
                .map(|page| doc.extract_text(page))
                .collect::<Result<Vec<String>, PdfError>>()?;
            let markdown = doc.to_markdown_all(&ConversionOptions::default())?;
            Ok(PdfExtract { pages, markdown })
        },
        pdf_error_identity,
    )
}

/// The cheap page-count probe: parse the page tree, nothing else. For
/// callers that gate expensive downstream work on page count (the
/// docling-conversion caller's pre-flight) without paying for any content
/// extraction. Runs under [`with_depth_headroom`], like every entry point.
pub fn page_count(bytes: Vec<u8>, password: Option<&str>) -> Result<usize, PdfError> {
    let password = password.map(str::to_string);
    with_depth_headroom(
        move || {
            let doc = open(bytes, password.as_deref())?;
            doc.page_count()
        },
        pdf_error_identity,
    )
}

/// The cheap text-vs-image preflight: page count plus the image-only page
/// list, no content conversion. Encrypted documents fail closed
/// (pdf_oxide's security rule, propagated as-is): a security state is never
/// masked as "all pages empty". Runs under [`with_depth_headroom`], like
/// every entry point.
pub fn classify(bytes: Vec<u8>, password: Option<&str>) -> Result<Classification, PdfError> {
    let password = password.map(str::to_string);
    with_depth_headroom(
        move || {
            let doc = open(bytes, password.as_deref())?;
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
        },
        pdf_error_identity,
    )
}

/// One `page_geometry` entry: a page's declared render box, in points. The
/// cheap page-tree probe's unit of answer — the visible box (the CropBox ∩
/// MediaBox region a reader displays) extent with the axes swapped for a
/// quarter turn (the render space's own extent), the engine-normalized
/// rotation, and the crop provenance. A page whose box cannot be read, or
/// whose declared extent is non-positive/non-finite, is a `None` entry (see
/// [`page_geometry`]); `Some` always carries a positive, finite extent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PageBox {
    /// The page's render width in points: the visible box's extent, the
    /// axes swapped for a 90/270 rotation. Positive and finite.
    pub width_pt: f32,
    /// The page's render height in points (the same law).
    pub height_pt: f32,
    /// The page's `/Rotate` engine-normalized to {0, 90, 180, 270} (with
    /// page-tree inheritance): the turn the render has already applied. A
    /// non-right-angle declaration reports 0 (the engine's rule), and the
    /// page's box is then the honest unrotated-extraction frame.
    pub rotation: i32,
    /// The page's declared CropBox reduced to the MediaBox (the effective
    /// crop: the intersection the engine's visible box stands on),
    /// normalized corners, the unrotated source frame, points —
    /// provenance for the extent only: never re-applied, never re-swapped,
    /// and the width/height need not equal its extent at 90/270. `Some`
    /// only when a CropBox is declared AND it actually crops (the
    /// intersection differs from the MediaBox): `None` when undeclared,
    /// degenerate, out-of-medium, or equal to media — in every one of
    /// which the engine's own silent fallback has the visible box standing
    /// on the MediaBox, which the intersection-equals-media test reads
    /// back. (What is NOT carried — deliberately, phase 2's note: the
    /// box's ORIGIN. The crop is provenance for the extent, never a
    /// rectangle to transform; the phase-2 layout transform must fetch
    /// the visible-box corners per page itself.)
    pub crop: Option<(f32, f32, f32, f32)>,
}

impl PageBox {
    /// The visible box (unrotated, points) and the engine-normalized
    /// rotation to the entry: the extent from the visible box's corners
    /// with the axes swapped for a quarter turn, the crop rule applied.
    /// `None` for a page whose declared extent is non-positive or
    /// non-finite (undeclarable: a `Some(0, 0)` entry cannot be ingested
    /// by any consumer's own positive-extent contract).
    fn from_visible_box(
        visible: (f32, f32, f32, f32),
        rotation: i32,
        media: (f32, f32, f32, f32),
    ) -> Option<PageBox> {
        let (x0, y0, x1, y1) = visible;
        let (raw_width, raw_height) = (x1 - x0, y1 - y0);
        let finite_positive = |value: f32| value.is_finite() && value > 0.0;
        if !finite_positive(raw_width) || !finite_positive(raw_height) {
            return None;
        }
        // The render extent: the declared box's axes swap for a quarter
        // turn (the turn is already applied; the numbers are the rendered
        // page's own width and height).
        let (width_pt, height_pt) = match rotation {
            90 | 270 => (raw_height, raw_width),
            _ => (raw_width, raw_height),
        };
        // The crop rule: `Some` only when a declared CropBox actually
        // crops — its intersection with the MediaBox differs from the
        // MediaBox. Without a CropBox the visible box IS the media box,
        // and a degenerate, out-of-medium, media-equal, or media-superset
        // declaration intersects back to the media box (the engine's own
        // silent fallback, read back as no-crop). The corners are the
        // intersection's: resolved, either-diagonal-normalized, and
        // unrotated — the crop is provenance for the extent, never a
        // rectangle to transform.
        let crop = if visible == media {
            None
        } else {
            Some(visible)
        };
        Some(PageBox {
            width_pt,
            height_pt,
            rotation,
            crop,
        })
    }
}

/// The declared page-box probe: for every page, its visible box's render
/// extent, its engine-normalized rotation, and its effective-crop
/// provenance ([`PageBox`]). The page tree only — no content stream is
/// parsed, so the cost is the per-page dict walk, flat in the page count.
///
/// Per-entry `None` (the single carve-out to the surface's no-partial-
/// success rule): a page whose own box read fails — the page's `get_page`
/// walk included — on an index `page_count` reported is a `None` entry,
/// with its siblings intact. The rule is by construction, never by
/// error-message matching (the engine returns the same `InvalidPdf`
/// variant for every one of these shapes, so the message could not
/// distinguish them anyway). Zero pages: an empty vec. Encrypted
/// documents fail closed at the [`open`] door check. Runs under
/// [`with_depth_headroom`], like every entry point.
pub fn page_geometry(
    bytes: Vec<u8>,
    password: Option<&str>,
) -> Result<Vec<Option<PageBox>>, PdfError> {
    let password = password.map(str::to_string);
    with_depth_headroom(
        move || {
            let doc = open(bytes, password.as_deref())?;
            let count = doc.page_count()?;
            let mut out = Vec::with_capacity(count);
            for page in 0..count {
                out.push(page_box(&doc, page).unwrap_or(None));
            }
            Ok(out)
        },
        pdf_error_identity,
    )
}

/// One page's box read ([`page_geometry`]'s per-page step): the media box
/// (for the crop rule's comparison), the visible box (the declared
/// CropBox ∩ MediaBox, normalized corners), and the rotation, folded by
/// [`PageBox::from_visible_box`]. Any engine error from the walk
/// propagates and is the caller's `None` entry, never a document failure.
fn page_box(doc: &PdfDocument, page: usize) -> Result<Option<PageBox>, PdfError> {
    let media = doc.get_page_media_box(page)?;
    let visible = doc.get_page_visible_box(page)?;
    let rotation = doc.get_page_rotation(page)?;
    Ok(PageBox::from_visible_box(visible, rotation, media))
}

// --- the page_layout probe (spec §5: the D1 coordinate space, one
// --- transform, one open) ---------------------------------------------------

/// One `page_layout` text span: a run's text, its render-space box, and
/// the font/style evidence a highlight/evidence consumer needs (§5/D6's
/// field bar — `sequence` is list order, `char_widths`/`heading_level`/
/// `color`/`mcid` are deliberately cut).
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutSpan {
    /// The run's text (the engine's span text verbatim).
    pub text: String,
    /// The run's box in the page's DECLARED RENDER SPACE (the D1 law: see
    /// [`map_box_forward`]) — `(x, y, w, h)`, y-up, the box's bottom edge
    /// at the glyph baseline.
    pub bbox: (f32, f32, f32, f32),
    /// The run's font size in points (the em-box's height).
    pub font_size: f32,
    /// The run's font name (the engine's resolved name).
    pub font_name: String,
    /// Bold evidence (the engine's weight resolution).
    pub is_bold: bool,
    /// Italic evidence.
    pub is_italic: bool,
    /// The run's own content rotation in degrees (the text-matrix angle):
    /// the honesty flag for the T3 approximation — a non-zero value means
    /// the box is the engine's FLATTENED approximation and MUST NOT be
    /// used as a highlight box: it is transposed or mis-anchored at every
    /// non-zero content rotation, on ANY page (180° same-size/wrong-anchor
    /// with zero ink overlap; ±90° transposed; on a /Rotate 90/270 page
    /// additionally mapped, its line box degraded to a square).
    /// Verified NOT to cover vertical writing (T9, probed — see
    /// [`page_layout`]'s docstring for the measured evidence): a WMode-1
    /// run (tategaki) carries `rotation_degrees == 0.0` — the angle is
    /// the text MATRIX's (`snap_run_rotation`); the writing mode plays no
    /// part in it — while its box is the flattened fiction at its worst
    /// (the vertical advance flattened onto x as a NEGATIVE width).
    pub rotation_degrees: f32,
    /// Declared `/Artifact` marked content — TYPED-FORM-ONLY (the T11
    /// law): true only for a typed `/Artifact <<…>> BDC` whose `/Type` or
    /// `/Subtype` the engine recognizes (Pagination/Header/Footer/
    /// Watermark/PageNumber, Layout, Page, Background); false for a bare
    /// `/Artifact BMC` (the engine's BMC handler deliberately leaves the
    /// artifact type unset, and the span carries no other artifact bit)
    /// and for unrecognized types. So `is_artifact == false` is NOT
    /// evidence of body text: header/footer detection must not rely on
    /// this flag alone.
    pub is_artifact: bool,
}

/// One word inside a [`LayoutLine`]: the text and its render-space box
/// (the D1 law). Nested in lines — word boxes are the engine's own
/// gap-clustered boxes, never re-derived downstream.
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutWord {
    /// The word's text.
    pub text: String,
    /// The word's box in render space (`(x, y, w, h)`, y-up).
    pub bbox: (f32, f32, f32, f32),
}

/// One `page_layout` text line: the engine's line (its own word
/// clustering — gap thresholds are NOT re-implemented downstream), its
/// render-space box, and its words. The join law holds per line: the
/// line text is the words' texts joined with single spaces (the engine's
/// own `TextLine::new` rule), and every word's box is inside the line's
/// box (the union law). Both are pinned in the tests.
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutLine {
    /// The line's text (the engine's join of its words).
    pub text: String,
    /// The line's box in render space — the union of its words' boxes.
    pub bbox: (f32, f32, f32, f32),
    /// The line's words, in reading order.
    pub words: Vec<LayoutWord>,
}

/// One `page_layout` image box: an image draw's placement, from
/// `page_image_handles` ONLY (never `extract_images`: that surface
/// decodes every image, silently drops sub-8×8-px ones, and swallows
/// decode errors — disqualified for an evidence surface, probe P5).
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutImage {
    /// The image's axis-aligned box in render space (`(x, y, w, h)`,
    /// y-up) — the CTM-derived placement of the unit rectangle. Image
    /// boxes are NEVER visibility-filtered (the engine's 2 pt drop law is
    /// the TEXT walk's): an image wholly outside the visible box is still
    /// returned, with out-of-range coordinates — the consumer clips to
    /// the page box, and never reads "image present" as "image visible".
    pub bbox: (f32, f32, f32, f32),
    /// The image's pixel width (the XObject's `/Width`).
    pub width_px: u32,
    /// The image's pixel height.
    pub height_px: u32,
    /// The CTM's rotation angle in degrees, `atan2`-normalized to
    /// [0, 360) — NOT restricted to right angles (a sheared or
    /// arbitrarily-rotated CTM reports e.g. 45.0; a mirrored CTM 180.0
    /// though the pixels are reflected, not turned): evidence, never a
    /// transform instruction — the bbox is always the axis-aligned bound
    /// of the transformed unit square.
    pub rotation_degrees: f32,
    /// Page-local identity/z-order: the image's index among the page's
    /// painted images, in content-stream order — the only stable handle
    /// the engine offers.
    pub paint_order: u32,
}

/// One `page_layout` entry: the page's geometry (a [`PageBox`], the same
/// unit `page_geometry` returns — `None` where that page's box cannot be
/// read or declares no positive extent), plus its text spans, lines, and
/// image boxes in the same render space. Output is parallel to the
/// normalized `pages=` selection, and every entry carries `page_index`
/// (normalization reorders; the caller maps back by it).
#[derive(Clone, Debug, PartialEq)]
pub struct PageLayoutData {
    /// The 0-based page this entry describes.
    pub page_index: usize,
    /// The 0-based page this entry's geometry lane serves — the resolved
    /// format's vocabulary name (`resolved_name` spelling: the same
    /// string `to_markdown` reports, the Excel kind's container-aware
    /// `xls`/`xlsx` case included). Provenance carried ON the entry: the
    /// format — not the entry's shape — tells you why `spans`/`lines`/
    /// `images` may be empty (an office row carries declared geometry
    /// with no text surfaces; a blank or scanned PDF page looks
    /// identical). Every entry of one call carries the same name: the
    /// format is resolved once per call, before any entry exists.
    pub format: String,
    /// The page's declared render box (the `page_geometry` entry).
    pub geometry: Option<PageBox>,
    /// The page's text spans, reading order.
    pub spans: Vec<LayoutSpan>,
    /// The page's text lines, reading order (words nested).
    pub lines: Vec<LayoutLine>,
    /// The page's image boxes, paint order.
    pub images: Vec<LayoutImage>,
}

/// THE TRANSFORM (spec §5, ruling D1) — the one forward rule, step 2
/// alone: a frame-relative box `(x, y, w, h)` in a frame of extents
/// `(w, h)` (the visible box's), rotated by the page's declared rotation
/// `r` (clockwise, the render's own turn), into the rotated frame.
/// `0` = identity (NO y-flip — the D2 resolution: the engine's box is
/// y-up baseline-origin, and rotation 0 is the verbatim identity);
/// `90` → `(y, W−x−w, h, w)`; `180` → `(W−x−w, H−y−h, w, h)`;
/// `270` → `(H−y−h, x, h, w)`. Split from [`map_box_forward`] so the
/// composition property (r then 360−r ≡ identity, the second application
/// in the rotated frame's own swapped extents) is testable at box level.
fn rotate_box_forward(
    b: (f32, f32, f32, f32),
    frame: (f32, f32),
    rotation: i32,
) -> (f32, f32, f32, f32) {
    let (x, y, w, h) = b;
    let (fw, fh) = frame;
    match rotation {
        90 => (y, fw - x - w, h, w),
        180 => (fw - x - w, fh - y - h, w, h),
        270 => (fh - y - h, x, h, w),
        _ => (x, y, w, h),
    }
}

/// The full forward rule (steps 1+2 of spec §5): a RAW y-up box
/// `(x, y, w, h)` in absolute user space, the page's visible-box corners
/// `(cx0, cy0, cx1, cy1)` (the `get_page_visible_box` output — fetched
/// ONCE per page and fed to both this and the `PageBox` build), and the
/// page's rotation: translate the box into visible-box-relative
/// coordinates (step 1: the source MediaBox/CropBox origin offset is
/// translated away, never re-added), then [`rotate_box_forward`] (step
/// 2). The page size (step 3: extents swapped for 90/270) is the
/// `PageBox`'s width/height, already built by
/// [`PageBox::from_visible_box`]. One rule for every box on the page —
/// spans, lines, words, images alike (pre-mapped spans arrive here via
/// [`unmap_engine_box`] first).
fn map_box_forward(
    raw: (f32, f32, f32, f32),
    visible: (f32, f32, f32, f32),
    rotation: i32,
) -> (f32, f32, f32, f32) {
    let (x, y, w, h) = raw;
    let (cx0, cy0, _, _) = visible;
    let translated = (x - cx0, y - cy0, w, h);
    rotate_box_forward(translated, (visible.2 - cx0, visible.3 - cy0), rotation)
}

/// The inverse of the ENGINE's own span mapping, for spans it already
/// rotated: pdf_oxide's postprocess rewrites a rotated page's span boxes
/// about the RAW MediaBox (`map_span_into_rotated_frame`,
/// document.rs:12731-12750: translate to MediaBox-relative, rotate with
/// `rotate_span_bbox(rel, rot, media_w, media_h)` — the same four-arm
/// formula as [`rotate_box_forward`] — translate back; the applied angle
/// lands in `page_rotation_applied`). Reconstruction of the inverse,
/// from that formula: translate the mapped box MediaBox-relative again,
/// apply the inverse rect map — for 90 the forward-90 map's inverse is
/// `(Mw − y′ − h′, x′, h′, w′)` (forward-90 is `(y, Mw−x−w, h, w)`:
/// solve each field), for 180 it is self-inverse, for 270 it is
/// `(y′, Mh − x′ − w′, h′, w′)` — then translate back to absolute raw
/// user space. The output feeds [`map_box_forward`], so every box the
/// probe returns (raw or engine-mapped) has gone through the ONE forward
/// rule; with no effective crop the round trip is the identity (the
/// engine's frame and tors's coincide), and with one it is exactly the
/// correction to the visible-box frame the D1 law demands.
fn unmap_engine_box(
    mapped: (f32, f32, f32, f32),
    media: (f32, f32, f32, f32),
    rotation: i32,
) -> (f32, f32, f32, f32) {
    let (llx, lly, urx, ury) = media;
    let (fw, fh) = (urx - llx, ury - lly);
    let (mx, my, mw, mh) = (mapped.0 - llx, mapped.1 - lly, mapped.2, mapped.3);
    let raw = match rotation {
        90 => (fw - my - mh, mx, mh, mw),
        180 => (fw - mx - mw, fh - my - mh, mw, mh),
        270 => (my, fh - mx - mw, mh, mw),
        _ => (mx, my, mw, mh),
    };
    (raw.0 + llx, raw.1 + lly, raw.2, raw.3)
}

/// Whether the engine has already mapped a span (or word) whose own
/// content rotation is `content_rotation` on a page rotated `rotation`:
/// pdf_oxide maps ALL spans on a 180 page, only rotated-content spans
/// (`rotation_degrees != 0`) on 90/270 pages, and nothing at 0
/// (document.rs:13086-13092). The spans carry the applied angle in
/// `page_rotation_applied` (consumed here, never exposed); words and
/// lines carry no such flag, so mappedness is derived from the same
/// rule's inputs — the page's rotation and the word's own content
/// rotation (lines never mix: the engine emits each rotated run as its
/// own line).
fn engine_mapped(rotation: i32, content_rotation: f32) -> bool {
    match rotation {
        90 | 270 => content_rotation != 0.0,
        180 => true,
        _ => false,
    }
}

/// The layout probe: one open, and per requested page the page's geometry
/// plus every text span, text line, and image box in the page's declared
/// render space ([`PageLayoutData`]). `pages` is the `pages=` selection:
/// `None` = the whole document in page order; `Some` is normalized here
/// with `markdown_pages`' exact rule and error strings
/// ([`normalize_selection`] / [`validate_selection_range`]), and the
/// output is parallel to the NORMALIZED selection, every entry carrying
/// its `page_index`. One open, three per-page engine reads
/// (`extract_spans`/`extract_text_lines` share the cached span parse).
///
/// Failure semantics (spec §7): a page whose own box read fails — or
/// whose declared extent is non-positive/non-finite — is an entry with
/// `geometry: None` and EMPTY spans/lines/images (no frame to transform
/// into: the D1 space is defined by the box), its siblings intact; a
/// page whose CONTENT stream fails to parse is an entry with the
/// geometry intact and empty lists — the engine's own drop-the-page
/// behavior (documented at document.rs:23773 for the image walk and
/// extract_spans_impl's content-decode arm), never an error (the
/// empty-lists-not-errors engine law; a scanned page is the same
/// shape). Encrypted documents fail closed at the [`open`] door check.
///
/// Verified T9 edge (probed against pdf_oxide 0.3.78; the honesty flag
/// does NOT cover vertical writing): a WMode-1 run — a Type0 font whose
/// `/Encoding` CMap name ends in `-V` (`/Identity-V`; the engine reads
/// the writing mode from the FONT on Tf, `text.rs:5307-5319` ←
/// `resolve_encoding_writing_mode`, `font_dict.rs:2316-2334`, and
/// `wmode_from_predefined_cmap_name`, `font_dict.rs:355-364` — or from an
/// embedded CMap stream's `/WMode 1 def`, `cmap.rs:540-557`) reports
/// `rotation_degrees == 0.0` (the angle is the text MATRIX's alone —
/// `snap_run_rotation`, `text.rs:1949-1979` — the writing mode plays no
/// part in it) while its box is the flattened fiction at its WORST.
/// Measured: `BT /F1 24 Tf 300 700 Td <00410042> Tj ET` under
/// /Identity-V (/DW 1000) → span bbox `(300, 700, -48, 24)` — the run's
/// vertical advance (-24 pt/glyph; the spec /DW2 default `w1y = -1000`,
/// routed into the y column by `advance_text_matrix`,
/// `graphics_state.rs:449-458`, summed by the vertical branch of
/// `advance_position_for_string`, `text.rs:8084-8104`, stamped
/// width-onto-x by both flush sites, `text.rs:7290-7295`/`8900-8906`) —
/// a NEGATIVE width, the glyphs' true ink a column at `(300, 652, 24,
/// 72)`, zero overlap. The line surface is wrong the same way one level
/// up: it re-lays the glyphs over the normalized box and reports the
/// REVERSED text (`"BA"` for `"AB"`, line box `(252, 700, 48, 24)`).
/// The engine's span carries `wmode == 1`, but wmode is cut from this
/// surface (§5/D6), so a consumer cannot see it here. On a /Rotate
/// 90/270 page the vertical run is NOT engine-premapped
/// (`page_rotation_applied` stays 0 — the mapping keys on
/// `rotation_degrees`, `document.rs:13086-13092`), so the same fiction
/// rides the forward map into the rotated frame. Consumer law: a
/// highlight box for text that may be vertical writing (CJK tategaki)
/// cannot be validated by `rotation_degrees` alone — treat
/// vertical-writing sources as outside the highlight-box guarantee
/// entirely.
///
/// Runs under [`with_depth_headroom`], like every entry point.
pub fn page_layout(
    bytes: Vec<u8>,
    pages: Option<&[usize]>,
    password: Option<&str>,
    format: &'static str,
) -> Result<Vec<PageLayoutData>, PagesError> {
    let selected = pages.map(<[usize]>::to_vec);
    let password = password.map(str::to_string);
    with_depth_headroom(
        move || page_layout_pass(bytes, selected.as_deref(), password.as_deref(), format),
        PagesError::Pdf,
    )
}

/// The pass [`page_layout`] runs under [`with_depth_headroom`] — the
/// selection normalization (shared with `markdown_pages`, so the error
/// strings are byte-identical) and the per-page fold. `format` is the
/// resolved format's vocabulary name (the caller resolves it — this
/// probe serves only the lane that name names), stamped on every entry.
fn page_layout_pass(
    bytes: Vec<u8>,
    pages: Option<&[usize]>,
    password: Option<&str>,
    format: &'static str,
) -> Result<Vec<PageLayoutData>, PagesError> {
    let doc = open(bytes, password).map_err(PagesError::Pdf)?;
    let count = doc.page_count().map_err(PagesError::Pdf)?;
    let selected = match pages {
        None => (0..count).collect::<Vec<usize>>(),
        Some(pages) => {
            let selected = normalize_selection(pages)?;
            validate_selection_range(&selected, count)?;
            selected
        }
    };
    Ok(selected
        .iter()
        .map(|&page| page_layout_one(&doc, page, format))
        .collect())
}

/// The §7 per-entry `None` shape, shared by `page_layout_one`'s four
/// early-outs: geometry `None` (no frame — the D1 space is defined by
/// the box) with empty spans/lines/images, the entry's `page_index`
/// still carried. Each caller's comment names its own trigger.
fn empty_entry(page: usize, format: &str) -> PageLayoutData {
    PageLayoutData {
        page_index: page,
        format: format.to_string(),
        geometry: None,
        spans: Vec::new(),
        lines: Vec::new(),
        images: Vec::new(),
    }
}

/// One page's layout ([`page_layout_pass`]'s per-page step): the box
/// reads feed both the `PageBox` build and the transform (fetched ONCE —
/// the phase-1 `PageBox` deliberately does not carry the visible-box
/// origin); the content reads are empty-on-error (the engine law, §7).
fn page_layout_one(doc: &PdfDocument, page: usize, format: &str) -> PageLayoutData {
    // The frame: any error here (the page's own get_page walk included)
    // is the §7 per-entry None rule — geometry None, no frame, so the
    // content has no render space to be mapped into and stays empty.
    let Ok(media) = doc.get_page_media_box(page) else {
        return empty_entry(page, format);
    };
    let Ok(visible) = doc.get_page_visible_box(page) else {
        return empty_entry(page, format);
    };
    let Ok(rotation) = doc.get_page_rotation(page) else {
        return empty_entry(page, format);
    };
    let geometry = PageBox::from_visible_box(visible, rotation, media);
    if geometry.is_none() {
        // A non-positive/non-finite extent: undeclarable (§7) — and the
        // transform's frame with it.
        return empty_entry(page, format);
    }

    // Spans. Engine-premapped ones (page_rotation_applied == the page's
    // rotation, non-zero) inverse-map through the engine's MediaBox-
    // relative transform back to raw first; everything then rides the
    // one forward rule. A content parse failure is the §7 engine law:
    // empty lists, never an error (T8: the engine's span path returns
    // empty on a content-stream decode failure; a stream that decodes
    // but parses to junk yields no spans the same way).
    let spans = doc
        .extract_spans(page)
        .unwrap_or_default()
        .iter()
        .map(|s| {
            let raw_box = (s.bbox.x, s.bbox.y, s.bbox.width, s.bbox.height);
            let raw = if rotation != 0 && s.page_rotation_applied == rotation {
                unmap_engine_box(raw_box, media, rotation)
            } else {
                raw_box
            };
            LayoutSpan {
                text: s.text.clone(),
                bbox: map_box_forward(raw, visible, rotation),
                font_size: s.font_size,
                font_name: s.font_name.clone(),
                is_bold: s.font_weight.is_bold(),
                is_italic: s.is_italic,
                rotation_degrees: s.rotation_degrees,
                is_artifact: s.artifact_type.is_some(),
            }
        })
        .collect();

    // Lines (words nested): the engine's own clustering over the cached
    // span parse. A line's words never mix frames (the engine emits each
    // rotated run as its own line), so the line's box is mapped-ness
    // uniform; each word and the line box take the same unmap→forward
    // path as the spans. The line's text is untouched (boxes transform;
    // text never does) — the join law survives by construction.
    let lines = doc
        .extract_text_lines(page)
        .unwrap_or_default()
        .iter()
        .map(|l| {
            let line_mapped = rotation != 0
                && l.words
                    .iter()
                    .all(|w| engine_mapped(rotation, w.rotation_degrees));
            let map = |b: (f32, f32, f32, f32)| {
                let raw = if line_mapped {
                    unmap_engine_box(b, media, rotation)
                } else {
                    b
                };
                map_box_forward(raw, visible, rotation)
            };
            LayoutLine {
                text: l.text.clone(),
                bbox: map((l.bbox.x, l.bbox.y, l.bbox.width, l.bbox.height)),
                words: l
                    .words
                    .iter()
                    .map(|w| LayoutWord {
                        text: w.text.clone(),
                        bbox: map((w.bbox.x, w.bbox.y, w.bbox.width, w.bbox.height)),
                    })
                    .collect(),
            }
        })
        .collect();

    // Images: handles ONLY (T1 — never extract_images), ALWAYS raw user
    // space (the engine never /Rotate-maps them — probe P7), so the
    // forward rule directly.
    let images = doc
        .page_image_handles(page)
        .unwrap_or_default()
        .iter()
        .map(|h| LayoutImage {
            bbox: map_box_forward(
                (h.bbox.x, h.bbox.y, h.bbox.width, h.bbox.height),
                visible,
                rotation,
            ),
            width_px: h.width,
            height_px: h.height,
            rotation_degrees: h.rotation_degrees,
            // The engine counts images painted on one page; u32 is far
            // beyond any real content stream's operator count, and the
            // clamp beats a wrong page-local identity if the impossible
            // happens.
            paint_order: u32::try_from(h.paint_order).unwrap_or(u32::MAX),
        })
        .collect();

    PageLayoutData {
        page_index: page,
        format: format.to_string(),
        geometry,
        spans,
        lines,
        images,
    }
}

// --- the page_tables probe (spec: issue #204; the frame-coherence gate A1,
// --- the four cell-box branches A2, the refused-marker A9) -------------------

/// The table-primitive cap (A6): `page_tables` counts the page's
/// `is_table_primitive()` paths and skips detection above this many — the
/// markdown lane's own `MAX_TABLE_EDGES` (document.rs:21549), the O(H×V)
/// intersection sweep's documented bail-out that public `extract_tables`
/// does NOT inherit. The cap makes this surface's worst case the engine's
/// documented one: one path walk, no O(E²) sweep. A skipped page is the
/// refused entry (`tables_refused: true`) — at rotation 0 that
/// combination names the cap (A9's reading rule), nowhere else.
const MAX_TABLE_PRIMITIVES: usize = 1500;

/// One `page_tables` cell: the engine's cell text, its render-space box
/// ([`Cell`]'s regime is per-branch — see the field doc), and the merge
/// spans. `is_header` is deliberately cut (A2: every engine constructor
/// writes it equal to [`Row`]'s, so it carries no information [`Row`].
/// `is_header` does not); so are `mcids` and the synthesized `spans`
/// (spec §3's field bar).
#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    /// The cell's text (the engine's cell text verbatim; an empty cell is
    /// `""`).
    pub text: String,
    /// The cell's box in the page's DECLARED RENDER SPACE (the D1 law:
    /// [`map_box_forward`]) — `(x, y, w, h)`, y-up. Its regime is the
    /// engine's, per branch (A2): an intersection-branch cell's box is the
    /// RULING rectangle (always `Some`, empty cells included); the
    /// cluster/band/text branches take member-span unions (`None` when
    /// empty); a vertical merge pads merged rows with `bbox: None` cells
    /// inside an otherwise ruling-rect table. Not observable from the
    /// return value, kept asymmetric on purpose (normalizing either way
    /// invents geometry or discards it). GEOMETRY, not ownership: an empty
    /// cell's box is grid geometry, not a claim on the text inside it, and
    /// may lie OUTSIDE the page box (ruling lines are not crop-filtered —
    /// the unclipped law; the TEXT side is, so a grid can also lose enough
    /// cells to vanish entirely).
    pub bbox: Option<(f32, f32, f32, f32)>,
    /// The column span (the engine's; 1 unless a merge covers this cell).
    pub colspan: u32,
    /// The row span (the engine's; 1 unless a merge covers this cell).
    pub rowspan: u32,
}

/// One `page_tables` row: the row's cells and the engine's own header
/// verdict (boldness/font-size heuristics on the text/cluster branches;
/// ALWAYS false on the intersection branch — the pinned engine
/// limitation).
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// The row's cells, grid order. `len` may be LESS than the table's
    /// `col_count` when a colspan/rowspan merge covers cells (the
    /// cluster-branch merge, A2 — the intersection branch always emits
    /// exactly `col_count`); a consumer lays cells out by walking
    /// `colspan`, never by `cells[i]`-as-column-i.
    pub cells: Vec<Cell>,
    /// Whether the engine judged this row a header row.
    pub is_header: bool,
}

/// One `page_tables` table: the engine's detected grid.
#[derive(Clone, Debug, PartialEq)]
pub struct TableEntry {
    /// The grid's rows, the engine's order (branch-deterministic, never
    /// re-sorted here: a vertical merge emits its merged groups
    /// bottom-group-first, the cluster path top-first — both pinned).
    pub rows: Vec<Row>,
    /// Whether the engine found a header section (the intersection branch
    /// never sets it — always false there, a documented limitation).
    pub has_header: bool,
    /// The engine's grid column count. Rows may be shorter under merges
    /// (see [`Row::cells`]).
    pub col_count: usize,
    /// The table's box in render space — the ENGINE's own union (A2/A13:
    /// cell boxes on the intersection branch, member spans on the cluster/
    /// band/text branches, cluster line-extent × row-extent on the cluster
    /// grid, the two prior boxes after a vertical merge), pinned by
    /// goldens, never re-derived from the cells here. `Some` on every
    /// table the engine returns (the `min_table_cells ≥ 4` gate guarantees
    /// filled cells); `None` is retained only for engine parity and is not
    /// reachable in v1. NOT the table's ink extent: it includes empty grid
    /// cells and can lie outside the page box.
    pub bbox: Option<(f32, f32, f32, f32)>,
}

/// One `page_tables` entry: the page's geometry (the same [`PageBox`]
/// `page_geometry`/`page_layout` serve), plus the page's detected tables
/// with per-cell boxes in the same render space. Output is parallel to the
/// normalized `pages=` selection, and every entry carries `page_index`
/// (normalization reorders; the caller maps back by it).
#[derive(Clone, Debug, PartialEq)]
pub struct PageTablesData {
    /// The 0-based page this entry describes.
    pub page_index: usize,
    /// The resolved format's vocabulary name (`resolved_name` spelling —
    /// the same string `page_layout` stamps). Provenance carried ON the
    /// entry: the format — not the entry's shape — tells you why `tables`
    /// may be empty. On an OFFICE row `tables: []` is the per-format
    /// capability statement (tables are PDF-only in v1: the engine IR's
    /// tables carry no coordinates), so there `format` marks "unsupported
    /// here", not "no tables" — and `tables_refused` stays false (A9: not
    /// a per-page content judgment). On the pdf lane `tables: []` with
    /// `tables_refused: false` means "no tables / content unreadable"
    /// (deliberately indistinguishable, like `page_layout`'s empty
    /// lists), and `tables_refused: true` means THIS SURFACE declined a
    /// page whose inputs indicate possible table content.
    pub format: String,
    /// The page's declared render box (the `page_geometry` entry): `None`
    /// where the page's box cannot be read or declares no positive extent
    /// — then `tables` is empty too (no frame to map into).
    pub geometry: Option<PageBox>,
    /// The page's detected tables (the spatial detector's answer — see the
    /// divergence note on [`page_tables`]).
    pub tables: Vec<TableEntry>,
    /// The A9 marker: TRUE iff this surface deliberately declined to serve
    /// a page whose inputs indicate possible table content — the
    /// frame-incoherent /Rotate classes (A1) or the table-primitive edge
    /// cap (A6). Reading rule: `tables_refused` + `geometry.rotation !=
    /// 0` → frame incoherence; `true` at rotation 0 → the edge cap.
    /// FALSE for no-tables pages, for content-stream failures (unknowable
    /// — stays indistinguishable from no-tables by design), and on office
    /// rows (their marker is `format`).
    pub tables_refused: bool,
}

/// The A1 gate's verdict for one page: which frame the engine's table
/// boxes are in (and so how they map forward), or that the page's inputs
/// are frame-INCOHERENT and the surface refuses. computable from public
/// API only: the page's rotation, whether any path is a table primitive
/// (`has_rulings`), and the spans' `page_rotation_applied` flags
/// (`any_mapped`/`all_mapped` — the engine-mapping flag, NOT
/// `rotation_degrees`, which is content rotation). Per the ruling's table,
/// with the A12 fail-closed default: any combination NOT listed coherent
/// is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Frame {
    /// The engine's table boxes are raw user space: [`map_box_forward`]
    /// directly (identity at rotation 0).
    Raw,
    /// The engine's table boxes are its own page-rotation mapping (the
    /// span union branches; a pathless page): [`unmap_engine_box`] first,
    /// then [`map_box_forward`] — page_layout's span treatment exactly.
    Mapped,
    /// Frame-incoherent inputs (the detector would compare text in one
    /// frame against rulings in another): refuse — `tables: []`,
    /// `tables_refused: true`.
    Incoherent,
}

fn table_frame_gate(rotation: i32, has_rulings: bool, any_mapped: bool, all_mapped: bool) -> Frame {
    match rotation {
        // 0: the engine maps nothing and the forward rule is the identity;
        // every page at 0 is coherent by construction (the ruling table's
        // first row: "any | never | raw").
        0 => Frame::Raw,
        _ if any_mapped => {
            // Mapped spans present. Coherent only when the page is
            // PATHLESS (no rulings) and EVERY span is mapped: then the
            // detector's whole input is in the engine's mapped frame and
            // the unmap→forward round trip is exact. With rulings the
            // frames disagree (the measured 180 ruled/symmetric and
            // 90/270 sideways-ruled failures); with a MIX of mapped and
            // raw spans the detector compares two text frames — all
            // refused (fail-closed: these are the unlisted combinations).
            if !has_rulings && all_mapped {
                Frame::Mapped
            } else {
                Frame::Incoherent
            }
        }
        // No mapped spans: the detector's text input is raw, and so are
        // the paths (paths are never page-rotation-mapped) — raw
        // throughout, forward-map exactly. (At 90/270 this is the upright
        // -text page, ruled or not; at 180 no spans survive unmapped —
        // the engine maps ALL of them — so this arm sees only spanless
        // 180 pages, which carry no table either way.)
        _ => Frame::Raw,
    }
}

/// The tables probe: one open, and per requested page the page's geometry
/// plus its detected tables with per-cell boxes in the page's declared
/// render space ([`PageTablesData`]). `pages` is the `pages=` selection:
/// `None` = the whole document in page order; `Some` is normalized here
/// with `markdown_pages`' exact rule and error strings
/// ([`normalize_selection`] / [`validate_selection_range`]), and the
/// output is parallel to the NORMALIZED selection, every entry carrying
/// its `page_index`.
///
/// The per-page pipeline (A1/A6): the page box reads ONCE (media, visible,
/// rotation — the same triple [`page_geometry`] folds); ONE
/// `extract_paths` walk serves both the A6 cap count and the
/// `has_rulings` gate input; one `extract_spans` walk serves the gate's
/// mappedness inputs; the engine's `extract_tables` (spatial detector,
/// default config — no threshold knobs in v1) runs only when the gate
/// serves and the cap does not bite. Any `extract_tables` error (a
/// content-stream failure upstream of it included) is the content-read
/// policy, never an error: `tables: []`, geometry intact
/// (`unwrap_or_default`), deliberately indistinguishable from a
/// no-tables page. `extract_spans`/`extract_paths` failures poison
/// nothing either: they read as empty inputs (the frame gate then serves
/// a no-tables page).
///
/// The frame-coherence gate (A1, A12's fail-closed default): the
/// detector's inputs live in two frames — text spans inherit the engine's
/// page-rotation mapping, paths never get it — and the detector compares
/// them in BOTH frames' disagreement regimes, where it misassigns or
/// loses the table (measured: a ruled 2×2 on /Rotate 180 is not detected
/// at all; a 180-symmetric grid comes back with every cell's text
/// 180°-swapped; sideways text under rulings misassigns). Three classes
/// are coherent and served exactly — rotation 0 (any content);
/// 90/270 with NO engine-mapped span (upright text; raw throughout); and
/// a PATHLESS page whose every span is engine-mapped (text-only tables;
/// `unmap_engine_box` → [`map_box_forward`], `page_layout`'s span
/// treatment) — every other combination refuses:
/// `tables: []`, `tables_refused: true`, geometry intact. The docs' reading
/// rule (A9): `tables_refused` with `geometry.rotation != 0` is frame
/// incoherence; `tables_refused` at rotation 0 is the primitive cap.
///
/// The markdown divergence (A4, stated, not fixed in v1): this surface
/// builds on the PUBLIC `extract_tables` — spatial-only, unconsolidated.
/// `to_markdown` renders tables through a different, private pipeline
/// (structure-tree-first for tagged PDFs, fragment consolidation, the RTL
/// decline guard, the edge cap, the relaxed wide-table retry), so the
/// TABLE SET can differ on any PDF: page_tables is the spatial detector's
/// answer, to_markdown renders a different table set. Double-reporting
/// (A3): `page_layout` serves `extract_spans`, which returns table
/// members — content inside a detected table's area may appear in BOTH
/// surfaces; neither de-dups the other, there is no identity join, and
/// cell bboxes must NOT be used to subtract from page_layout (an empty
/// intersection cell's box is grid geometry, not ownership).
///
/// Zero pages: an empty vec. Encrypted documents fail closed at the
/// [`open`] door check (before the `pages=` validation — A10's PDF-leg
/// order: open → page_count → normalize/validate, so an encrypted
/// document with a malformed selection answers the encryption refusal).
/// Runs under [`with_depth_headroom`], like every entry point.
pub fn page_tables(
    bytes: Vec<u8>,
    pages: Option<&[usize]>,
    password: Option<&str>,
    format: &'static str,
) -> Result<Vec<PageTablesData>, PagesError> {
    let selected = pages.map(<[usize]>::to_vec);
    let password = password.map(str::to_string);
    with_depth_headroom(
        move || page_tables_pass(bytes, selected.as_deref(), password.as_deref(), format),
        PagesError::Pdf,
    )
}

/// The pass [`page_tables`] runs under [`with_depth_headroom`] — the
/// selection normalization shared with [`page_layout_pass`] (so the error
/// strings are byte-identical) and the per-page fold.
fn page_tables_pass(
    bytes: Vec<u8>,
    pages: Option<&[usize]>,
    password: Option<&str>,
    format: &'static str,
) -> Result<Vec<PageTablesData>, PagesError> {
    let doc = open(bytes, password).map_err(PagesError::Pdf)?;
    let count = doc.page_count().map_err(PagesError::Pdf)?;
    let selected = match pages {
        None => (0..count).collect::<Vec<usize>>(),
        Some(pages) => {
            let selected = normalize_selection(pages)?;
            validate_selection_range(&selected, count)?;
            selected
        }
    };
    Ok(selected
        .iter()
        .map(|&page| page_tables_one(&doc, page, format))
        .collect())
}

/// The per-entry `None`-geometry shape, [`empty_entry`]'s tables twin:
/// geometry `None` (no frame — the D1 space is defined by the box) with
/// no tables, the entry's `page_index` still carried, `tables_refused`
/// false (A5: the empty-entry law is unchanged; a PDF-only reachable
/// state).
fn empty_tables_entry(page: usize, format: &str) -> PageTablesData {
    PageTablesData {
        page_index: page,
        format: format.to_string(),
        geometry: None,
        tables: Vec::new(),
        tables_refused: false,
    }
}

/// One page's tables ([`page_tables_pass`]'s per-page step): the box reads
/// feed both the `PageBox` build and the transform (fetched ONCE —
/// page_layout's law); the primitive count and the mappedness inputs feed
/// the gate; the engine's tables map through the ONE forward rule in the
/// gate's frame.
fn page_tables_one(doc: &PdfDocument, page: usize, format: &str) -> PageTablesData {
    // The frame: any error here (the page's own get_page walk included)
    // is the per-entry None rule — geometry None, no frame, so the
    // content has no render space to be mapped into and no tables are
    // served (the empty-entry law, A5).
    let Ok(media) = doc.get_page_media_box(page) else {
        return empty_tables_entry(page, format);
    };
    let Ok(visible) = doc.get_page_visible_box(page) else {
        return empty_tables_entry(page, format);
    };
    let Ok(rotation) = doc.get_page_rotation(page) else {
        return empty_tables_entry(page, format);
    };
    let geometry = PageBox::from_visible_box(visible, rotation, media);
    if geometry.is_none() {
        // A non-positive/non-finite extent: undeclarable — and the
        // transform's frame with it.
        return empty_tables_entry(page, format);
    }

    // ONE path walk for both gate inputs (A6/last ruling): the table
    // primitives' count IS the has_rulings bit (0 = none). A path-walk
    // failure reads as zero primitives (and starves extract_tables below
    // into its own empty answer — the content-failure class, refused:
    // false).
    let paths = doc.extract_paths(page).unwrap_or_default();
    let primitives = paths.iter().filter(|p| p.is_table_primitive()).count();
    let has_rulings = primitives > 0;

    // The A6 cap: above the markdown lane's MAX_TABLE_EDGES the surface
    // skips detection — the refused entry, geometry intact (rotation 0
    // distinguishes the cap from the frame refusals, A9's reading rule).
    if primitives > MAX_TABLE_PRIMITIVES {
        return PageTablesData {
            page_index: page,
            format: format.to_string(),
            geometry,
            tables: Vec::new(),
            tables_refused: true,
        };
    }

    // The gate's span inputs: the flow spans' engine-mapping flags (A1:
    // `page_rotation_applied == rotation` is the mapped bit — NOT
    // `rotation_degrees`, which is content rotation). A span-parse
    // failure reads as no spans (none mapped — the gate then serves a
    // page the engine will find no tables on, the same empty shape).
    let spans = doc.extract_spans(page).unwrap_or_default();
    let mut any_mapped = false;
    let mut all_mapped = true;
    for span in &spans {
        let mapped = rotation != 0 && span.page_rotation_applied == rotation;
        any_mapped |= mapped;
        all_mapped &= mapped;
    }

    // The A1 frame-coherence gate (A12's fail-closed default).
    let frame = table_frame_gate(rotation, has_rulings, any_mapped, all_mapped);
    if frame == Frame::Incoherent {
        return PageTablesData {
            page_index: page,
            format: format.to_string(),
            geometry,
            tables: Vec::new(),
            tables_refused: true,
        };
    }

    // The engine's tables, in the gate's frame. ANY extract_tables error
    // is the content-read policy (A6): empty, geometry intact, never
    // raised — the common failure shape is the same `tables: []`.
    let tables = doc
        .extract_tables(page)
        .unwrap_or_default()
        .iter()
        .map(|table| {
            let map = |b: Option<pdf_oxide::geometry::Rect>| {
                b.map(|rect| {
                    let box_ = (rect.x, rect.y, rect.width, rect.height);
                    // The engine pre-mapped these boxes: inverse-map about
                    // the MediaBox first (page_layout's span treatment),
                    // then the one forward rule. (Frame::Incoherent never
                    // reaches here — the gate refused above.)
                    let raw = if frame == Frame::Mapped {
                        unmap_engine_box(box_, media, rotation)
                    } else {
                        box_
                    };
                    map_box_forward(raw, visible, rotation)
                })
            };
            TableEntry {
                bbox: map(table.bbox),
                rows: table
                    .rows
                    .iter()
                    .map(|row| Row {
                        cells: row
                            .cells
                            .iter()
                            .map(|cell| Cell {
                                text: cell.text.clone(),
                                bbox: map(cell.bbox),
                                colspan: cell.colspan,
                                rowspan: cell.rowspan,
                            })
                            .collect(),
                        is_header: row.is_header,
                    })
                    .collect(),
                has_header: table.has_header,
                col_count: table.col_count,
            }
        })
        .collect();

    PageTablesData {
        page_index: page,
        format: format.to_string(),
        geometry,
        tables,
        tables_refused: false,
    }
}

/// The `/Annots` link-annotation walk: for every page, the URIs of its
/// `/Subtype /Link` annotations whose action is a URI (`/A << /S /URI /URI
/// (...) >>`), in annotation order: one `Vec<String>` per page, page
/// order, empty vecs for pages without link annotations.
///
/// Why this surface exists alongside the markdown's inline
/// `[text](uri)` links: the two answer different questions. The markdown
/// carries links whose visible text is worth keeping in prose; this walk
/// carries the raw URI list: including URIs behind link rectangles whose
/// text is not itself a link (a "click here" button, an image, a bare
/// rectangle) which no text-oriented rendering surfaces at all. The caller
/// that motivated it measured the difference on real resumes (their
/// `annotations.py`: 60 documents; text-layer extraction surfaced 16
/// links, the annotation walk 34; a quarter of candidates gained a
/// LinkedIn/GitHub URL no text shape would show). Deduplication and
/// canonicalization are deliberately not done here: the verbatim
/// engine-read list is the honest output, and every caller canonicalizes
/// differently (per-page review panels vs whole-document projections).
///
/// Scope, narrow: URI actions only. `GoTo` (an in-document
/// destination) and `GoToR` (a remote file) are navigation, not web
/// URIs: they contribute nothing here and are not fabricated into
/// `file://` shapes. Non-link annotations (widgets, stamps, notes) are
/// skipped by the subtype filter; pdf_oxide tolerates malformed annotation
/// dictionaries (its parse skips them) rather than poisoning the page.
pub fn link_uris(bytes: Vec<u8>, password: Option<&str>) -> Result<Vec<Vec<String>>, PdfError> {
    let password = password.map(str::to_string);
    with_depth_headroom(
        move || {
            let doc = open(bytes, password.as_deref())?;
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
        },
        pdf_error_identity,
    )
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

/// The error text: the payload of whichever arm failed, verbatim, so a
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

/// The `pages=` selection's shape normalization, shared by every
/// `pages=`-taking probe ([`markdown_pages_pass`], [`page_layout_pass`]):
/// sort + dedup into document order, then the empty-selection refusal. The
/// range check is [`validate_selection_range`] (it needs the page count,
/// which both passes fetch only after this shape half — the error
/// precedence the Python tests pin). The error strings are byte-pinned
/// (the binding layer's Python tests assert them verbatim), which is why
/// the two probes share these functions rather than re-spelling the
/// checks: the messages cannot drift apart.
fn normalize_selection(pages: &[usize]) -> Result<Vec<usize>, PagesError> {
    // Sort + dedup the caller's selection: duplicates would repeat a
    // page, an unsorted one would join out of document order.
    let mut selected: Vec<usize> = pages.to_vec();
    selected.sort_unstable();
    selected.dedup();
    if selected.is_empty() {
        return Err(PagesError::Pages(
            "pages= selected no pages: pass None for the whole document".into(),
        ));
    }
    Ok(selected)
}

/// The range half of the shared `pages=` normalization: every selected
/// index must be inside the page count the document itself reported. The
/// refusal names the index and the count, byte-pinned.
fn validate_selection_range(selected: &[usize], count: usize) -> Result<(), PagesError> {
    if let Some(out_of_range) = selected.iter().find(|page| **page >= count) {
        return Err(PagesError::Pages(format!(
            "page index {out_of_range} is out of range for a {count}-page document (pages= is 0-based)"
        )));
    }
    Ok(())
}

/// Convert in-memory PDF bytes to markdown over a page subset. `None` is
/// the whole document (`to_markdown_all`); `Some(pages)`: normalized
/// here (sorted, deduped) into document order, so a Rust caller's
/// arbitrary `&[1, 0]` joins as document order rather than a silently
/// out-of-order concatenation; the binding layer already delivers
/// normalized selections, this makes the core self-sufficient: converts
/// each selected page with `to_markdown(page)` and joins with
/// [`PAGE_SEPARATOR`], so `Some(&(0..n).collect::<Vec<_>>())` is
/// byte-identical to `None` (the crate-side test pins this).
pub fn markdown_pages(
    bytes: Vec<u8>,
    pages: Option<&[usize]>,
    password: Option<&str>,
) -> Result<String, PagesError> {
    let selected = pages.map(<[usize]>::to_vec);
    let password = password.map(str::to_string);
    with_depth_headroom(
        move || markdown_pages_pass(bytes, selected.as_deref(), password.as_deref()),
        PagesError::Pdf,
    )
}

/// The pass [`markdown_pages`] runs under [`with_depth_headroom`], its own
/// fn so the public signature the core calls stays `Option<&[usize]>` /
/// `Option<&str>` while the closure carries owned selections.
fn markdown_pages_pass(
    bytes: Vec<u8>,
    pages: Option<&[usize]>,
    password: Option<&str>,
) -> Result<String, PagesError> {
    let doc = open(bytes, password).map_err(PagesError::Pdf)?;
    let options = ConversionOptions::default();
    let Some(selected) = pages else {
        return doc.to_markdown_all(&options).map_err(PagesError::Pdf);
    };
    let selected = normalize_selection(selected)?;
    validate_selection_range(&selected, doc.page_count().map_err(PagesError::Pdf)?)?;
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

    /// Two single-line pages: the `pages=` semantics' minimal honest case.
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
        // Rust-side `&[1, 0]` joins in document order (byte-identical to
        // the sorted selection and to the whole-document tail), and a
        // duplicated index converts its page once: never a repeated or
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
        // A text page and a blank page: the blank page is Empty, which is
        // distinct from Scanned, so the OCR list is empty (a blank page is
        // not an image to recover; pdf_oxide's rule, mapped faithfully).
        assert_eq!(
            classification.page_kinds,
            vec![PageKind::TextLayer, PageKind::Empty]
        );
        assert_eq!(classification.pages_needing_ocr, Vec::<usize>::new());
    }

    /// A page whose only content is an image XObject (an uncompressed
    /// 64x64 grayscale raster, no /Filter, valid PDF, no compressor
    /// needed): the scanned shape, which must land in pages_needing_ocr
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
    /// without annotations contribute nothing, and the whole-document
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
            // page 1 carries two link annotations: a URI action, then a
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

    // --- the depth pre-scan: raw walk, then compressed-object scan ----------

    /// One stream object's bytes: dict pairs + payload, the shape the walk's
    /// stream collection keys on (dict close, `stream`, EOL, payload,
    /// `endstream`).
    fn stream_object(dict: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut out = format!(
            "<< {} /Length {} >>\nstream\n",
            String::from_utf8_lossy(dict),
            payload.len()
        )
        .into_bytes();
        out.extend_from_slice(payload);
        out.extend_from_slice(b"\nendstream");
        out
    }

    fn zlib_flow(bytes: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut encoder =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(bytes).unwrap();
        encoder.finish().unwrap()
    }

    /// Legit documents never trip the scan: the hand-built two-pager (real
    /// dicts, arrays, strings, streams) survives the walk and the full
    /// open() door with it in place.
    #[test]
    fn benign_documents_pass_the_pre_scan() {
        let bytes = two_page_pdf("Alpha page text", "Beta page text");
        scan_document(&bytes).unwrap();
        extract(bytes, None).unwrap();
    }

    /// The regions pdf_oxide's parser never recurses through must not count
    /// depth: a stream payload full of unbalanced `[`, a literal string's
    /// parens (nested and escaped), a comment's `<<`s, and a `<`-hex-string
    /// holding 0x3c bytes (`<<` spelled in hex), none of it moves the
    /// counter. The negative control re-runs the same brackets OUTSIDE the
    /// skipped regions: they do count there.
    #[test]
    fn regions_the_parser_never_recurses_through_are_skipped() {
        let payload = b"[[[[[[[ unbalanced, no closes";
        let doc = stream_object(b"/Type /Extra", payload);
        scan_document(&doc).unwrap();

        scan_document(b"( nested ( parens ) and \\( escaped \\) opens [ and << )").unwrap();

        scan_document(b"% a comment full of [[[[ << << <<").unwrap();

        // 0x3c is '<': a hex string spelling << << << must not nest.
        scan_document(b"<< /K < 3c 3c 3c 3c > >> ").unwrap();

        // The same brackets outside the skipped regions do count.
        assert!(scan_document(&b"[[[[[[[ unbalanced".repeat(30)).is_err());
    }

    /// The counter is exact, not a heuristic: exactly 100 levels (the cap,
    /// pdf_oxide's own dead-config value) passes, 101 refuses (arrays,
    /// keyed dictionaries, and a mix), and stray closes floor at zero so a
    /// binary region cannot manufacture depth below it.
    #[test]
    fn depth_counting_is_exact() {
        assert!(scan_walk(&b"[".repeat(100), false).is_ok());
        assert!(scan_walk(&b"[".repeat(101), false).is_err());
        let open_dict = b"<< /K ".repeat(100);
        assert!(scan_walk(&open_dict, false).is_ok());
        let open_dict_over = b"<< /K ".repeat(101);
        assert!(scan_walk(&open_dict_over, false).is_err());
        // Mixed: each (array + keyed dict) pair is two levels.
        let mut mixed = Vec::new();
        for _ in 0..50 {
            mixed.extend_from_slice(b"[ << /K ");
        }
        assert!(scan_walk(&mixed, false).is_ok());
        mixed.extend_from_slice(b"[ << /K ");
        assert!(scan_walk(&mixed, false).is_err());
        // Floor: a pile of stray closes cannot go negative, so what
        // follows is judged at its own depth, not an inherited one:
        // 101 opens after 200 junk closes still refuse, 100 still pass.
        let mut floored = b"]]] ".repeat(200);
        floored.extend_from_slice(&b"[".repeat(101));
        assert!(scan_walk(&floored, false).is_err());
        let mut shallow = b"]]] ".repeat(200);
        shallow.extend_from_slice(&b"[".repeat(100));
        assert!(scan_walk(&shallow, false).is_ok());
    }

    /// The refusal's message names the ceiling and the why (the same text
    /// the Python probes surface as their ValueError).
    #[test]
    fn over_depth_refusal_names_the_ceiling_and_the_reason() {
        let err = scan_walk(&b"[".repeat(101), false).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("100"), "{message}");
        assert!(message.contains("depth cap"), "{message}");
        assert!(matches!(err, PdfError::InvalidPdf(_)));
    }

    /// Layer 2: a bomb hidden inside a FlateDecode object stream (the
    /// shape the raw walk cannot see and pdf_oxide's objstm.rs:168 feeds
    /// to the same uncapped parse_object; measured SIGSEGV at
    /// raw-equivalent N=12,000 pre-scan) is inflated and scanned.
    /// Unfiltered object streams (legal: /Filter is optional) are scanned
    /// as-is, and a benign object stream still passes.
    #[test]
    fn compressed_object_stream_hiding_a_bomb_is_scanned() {
        let mut bomb = b"1 0\n<< /Type /Catalog /X ".to_vec();
        bomb.extend_from_slice(&b"[".repeat(201));
        bomb.extend_from_slice(&b"]".repeat(201));
        bomb.extend_from_slice(b" >>");
        let compressed = stream_object(
            b"/Type /ObjStm /N 1 /First 4 /Filter /FlateDecode",
            &zlib_flow(&bomb),
        );
        assert!(scan_document(&compressed).is_err());

        let raw_objstm = stream_object(b"/Type /ObjStm /N 1 /First 4", &bomb);
        assert!(scan_document(&raw_objstm).is_err());

        let benign = stream_object(
            b"/Type /ObjStm /N 1 /First 4 /Filter /FlateDecode",
            &zlib_flow(b"1 0\n<< /Type /Catalog /Pages 2 0 R >>"),
        );
        scan_document(&benign).unwrap();
    }

    /// A compressed xref stream's inflated data carrying the bomb bytes:
    /// pdf_oxide reads that data as packed binary rows and never recurses
    /// (measured: it converts, pre-scan, at any depth), so the scan here is
    /// containment for attacker-shaped junk, and the junk refuses
    /// catchably.
    #[test]
    fn compressed_xref_stream_junk_is_scanned() {
        let mut rows = vec![0u8; 5 * 7]; // five valid /W [1 4 2] rows
        rows.extend_from_slice(&b"[".repeat(201));
        let xref = stream_object(
            b"/Type /XRef /Size 5 /W [1 4 2] /Filter /FlateDecode",
            &zlib_flow(&rows),
        );
        assert!(scan_document(&xref).is_err());
    }

    /// The pdfium-hazard property, as a Rust test: pdfium is not thread-safe
    /// (downstream users serialize it behind a global lock); pdf_oxide's
    /// documents are `Send + Sync`, so N threads opening their own document
    /// over the same bytes must all succeed with byte-identical results:
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

    // --- the page-layout probe (the §9 PDF-side pins: D1 transform,
    // --- inverse map, handles-based images, the render-anchored oracle) ----

    /// A one-page PDF with a font, a content stream, and page-dict
    /// `page_extras` (the /Rotate//CropBox fragments): the layout
    /// fixtures' base.
    fn layout_pdf(content: &str, page_extras: &str) -> Vec<u8> {
        let stream = format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            content.len() + 1,
            content
        );
        objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            format!(
                "<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> \
                 /MediaBox [0 0 612 792] {page_extras} /Contents 5 0 R >>"
            )
            .as_bytes(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            stream.as_bytes(),
        ])
    }

    /// The probe's red 1×1 image XObject (ASCIIHexDecode, 7-byte payload —
    /// the probe crate's P5 fixture verbatim: small enough that
    /// `extract_images`' silent <8×8 filter would drop it, which is the
    /// point of pinning the HANDLES-based path).
    fn red_1x1_image() -> Vec<u8> {
        b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB \
           /BitsPerComponent 8 /Filter /ASCIIHexDecode /Length 7 >>\nstream\nFF0000>\nendstream"
            .to_vec()
    }

    /// An image-page PDF: the 1×1 red image painted by `image_ops` at the
    /// given page extras. Object numbering: catalog 1, pages 2, page 3,
    /// content stream 4, image 5.
    fn image_pdf(image_ops: &str, page_extras: &str) -> Vec<u8> {
        let content = format!("q {image_ops} cm /Im1 Do Q");
        let stream = format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            content.len() + 1,
            content
        );
        objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            format!(
                "<< /Type /Page /Parent 2 0 R /Resources << /XObject << /Im1 5 0 R >> >> \
                 /MediaBox [0 0 612 792] {page_extras} /Contents 4 0 R >>"
            )
            .as_bytes(),
            stream.as_bytes(),
            &red_1x1_image(),
        ])
    }

    /// A bold font object (the base-14 bold face — the engine's weight
    /// resolution keys on the name).
    fn bold_mid_line_pdf() -> Vec<u8> {
        let content = "BT /F1 24 Tf 72 700 Td (Hello) Tj ET\nBT /F2 24 Tf 140 700 Td (World) Tj ET";
        let stream = format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            content.len() + 1,
            content
        );
        objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R /F2 6 0 R >> >> \
                 /MediaBox [0 0 612 792] /Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            stream.as_bytes(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        ])
    }

    fn near(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn assert_box_near(
        bbox: (f32, f32, f32, f32),
        expected: (f32, f32, f32, f32),
        tol: f32,
        what: &str,
    ) {
        assert!(
            near(bbox.0, expected.0, tol)
                && near(bbox.1, expected.1, tol)
                && near(bbox.2, expected.2, tol)
                && near(bbox.3, expected.3, tol),
            "{what}: expected ({:?}), got ({:?})",
            expected,
            bbox
        );
    }

    /// The rot-0 identity pin (the D2 resolution made permanent): the run
    /// at Td(72,700), 24pt, lands at (72, 700, 54.67±0.01, 24) at every
    /// granularity — span, line, word — with NO y-flip anywhere.
    #[test]
    fn layout_rotation_0_is_the_identity_no_y_flip() {
        let bytes = layout_pdf("BT /F1 24 Tf 72 700 Td (Hello) Tj ET", "");
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        assert!(page.geometry.is_some());
        let geometry = page.geometry.unwrap();
        assert_eq!(
            (geometry.width_pt, geometry.height_pt, geometry.rotation),
            (612.0, 792.0, 0)
        );
        assert_eq!(page.spans.len(), 1);
        let span = &page.spans[0];
        assert_eq!(span.text, "Hello");
        assert_box_near(span.bbox, (72.0, 700.0, 54.67, 24.0), 0.01, "span");
        assert_eq!(
            (span.font_size, span.font_name.as_str()),
            (24.0, "Helvetica")
        );
        assert!(!span.is_bold && !span.is_italic && !span.is_artifact);
        assert_eq!(span.rotation_degrees, 0.0);
        // the same golden at line and word granularity
        assert_eq!(page.lines.len(), 1);
        let line = &page.lines[0];
        assert_box_near(line.bbox, (72.0, 700.0, 54.67, 24.0), 0.01, "line");
        assert_eq!(line.words.len(), 1);
        assert_box_near(line.words[0].bbox, (72.0, 700.0, 54.67, 24.0), 0.01, "word");
        assert_eq!(
            (line.words[0].text.as_str(), line.text.as_str()),
            ("Hello", "Hello")
        );
    }

    /// The /Rotate 90 golden (§9 item 3): the horizontal run maps through
    /// the forward rule to (700, 612−72−54.67, 24, 54.67) = (700, 485.33,
    /// 24, 54.67), and the page's render extent swaps to 792×612.
    #[test]
    fn layout_rotation_90_forward_rule_golden() {
        let bytes = layout_pdf("BT /F1 24 Tf 72 700 Td (Hello) Tj ET", "/Rotate 90");
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        let geometry = page.geometry.unwrap();
        assert_eq!(
            (geometry.width_pt, geometry.height_pt, geometry.rotation),
            (792.0, 612.0, 90)
        );
        assert_box_near(
            page.spans[0].bbox,
            (700.0, 485.33, 24.0, 54.67),
            0.01,
            "span at 90",
        );
        assert_box_near(
            page.lines[0].bbox,
            (700.0, 485.33, 24.0, 54.67),
            0.01,
            "line at 90",
        );
        assert_box_near(
            page.lines[0].words[0].bbox,
            (700.0, 485.33, 24.0, 54.67),
            0.01,
            "word at 90",
        );
    }

    /// The /Rotate 180 golden: (W−x−w, H−y−h) = (485.33, 68, 54.67, 24) —
    /// the engine maps everything at 180, and the inverse+forward round
    /// trip reproduces it exactly (no crop: tors's frame IS the engine's).
    #[test]
    fn layout_rotation_180_forward_rule_golden() {
        let bytes = layout_pdf("BT /F1 24 Tf 72 700 Td (Hello) Tj ET", "/Rotate 180");
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        assert_eq!(page.geometry.unwrap().rotation, 180);
        assert_box_near(
            page.spans[0].bbox,
            (485.33, 68.0, 54.67, 24.0),
            0.01,
            "span at 180",
        );
        assert_box_near(
            page.lines[0].bbox,
            (485.33, 68.0, 54.67, 24.0),
            0.01,
            "line at 180",
        );
    }

    /// The crop-subset golden (§9 item 1, probe P3): on the
    /// [100 100 500 700] crop the Inside run at (150,600) lands at
    /// (50, 500, 64.03, 24) — translated away from the crop origin — the
    /// fully-outside run is ABSENT (the engine's drop law), the page is
    /// 400×600, and the crop provenance is Some((100,100,500,700)).
    #[test]
    fn layout_crop_subset_golden_inside_mapped_outside_dropped() {
        let bytes = layout_pdf(
            "BT /F1 24 Tf 150 600 Td (Inside) Tj ET\nBT /F1 24 Tf 30 30 Td (Outside) Tj ET",
            "/CropBox [100 100 500 700]",
        );
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        let geometry = page.geometry.unwrap();
        assert_eq!((geometry.width_pt, geometry.height_pt), (400.0, 600.0));
        assert_eq!(geometry.crop, Some((100.0, 100.0, 500.0, 700.0)));
        assert_eq!(
            page.spans.len(),
            1,
            "the fully-outside span is dropped, not returned"
        );
        assert_eq!(page.spans[0].text, "Inside");
        assert_box_near(
            page.spans[0].bbox,
            (50.0, 500.0, 64.03, 24.0),
            0.01,
            "Inside span",
        );
        assert_eq!(page.lines.len(), 1);
        assert_box_near(
            page.lines[0].bbox,
            (50.0, 500.0, 64.03, 24.0),
            0.01,
            "Inside line",
        );
    }

    /// The partial-overlap pin (§9 item 2 + T7): a span that hangs PAST
    /// the crop edge is kept (the 2pt tolerance only drops FULLY-outside
    /// spans) and its box comes back UNCLIPPED — it extends beyond the
    /// page's own render extent ("boxes are unclipped; the page box is
    /// the consumer's clip").
    #[test]
    fn layout_partial_overlap_span_is_kept_and_unclipped() {
        // Crop [100 100 500 700]; the run starts at x=480 and advances
        // ~54.67pt, ending at ~534.67 — past the crop's right edge (500).
        let bytes = layout_pdf(
            "BT /F1 24 Tf 480 600 Td (Hang) Tj ET",
            "/CropBox [100 100 500 700]",
        );
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        assert_eq!(
            page.spans.len(),
            1,
            "the partially-overlapping span is kept"
        );
        let bbox = page.spans[0].bbox;
        assert_box_near(bbox, (380.0, 500.0, 57.36, 24.0), 0.01, "kept span");
        assert!(
            bbox.0 + bbox.2 > 400.0,
            "the box must extend past the 400pt-wide render page, unclipped: {bbox:?}"
        );
    }

    /// The origin law (§9 item 2): on a [50 50 662 842] MediaBox a run at
    /// the box's own origin comes back with bbox.x/y ≈ 0 — the source
    /// origin is translated away, never re-added (and the probe P4 shape:
    /// a run at user-space (100,700) lands at (50, 650)).
    #[test]
    fn layout_nonzero_media_box_origin_is_translated_away() {
        // A run at the MediaBox's own origin (50, 50) — the translated box
        // lands at (0, 0). (layout_pdf pins /MediaBox [0 0 612 792]; the
        // nonzero-origin shapes build their objects directly.)
        let bytes = objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> \
                 /MediaBox [50 50 662 842] /Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length 46 >>\nstream\nBT /F1 12 Tf 50 50 Td (Origin) Tj ET\nendstream",
        ]);
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        assert!(page.geometry.is_some());
        let bbox = page.spans[0].bbox;
        assert!(
            near(bbox.0, 0.0, 0.01) && near(bbox.1, 0.0, 0.01),
            "a run at the MediaBox origin lands at (0, 0) in render space: {bbox:?}"
        );
        // And the probe P4 shape: user-space (100,700) → (50, 650).
        let bytes = objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> \
                 /MediaBox [50 50 662 842] /Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length 46 >>\nstream\nBT /F1 24 Tf 100 700 Td (Hello) Tj ET\nendstream",
        ]);
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        assert_box_near(
            page.spans[0].bbox,
            (50.0, 650.0, 54.67, 24.0),
            0.01,
            "P4 shape",
        );
    }

    /// The T10 pin on the layout surface: a non-right-angle /Rotate 45
    /// reports rotation 0 and the boxes stay in the honest unrotated
    /// frame (the engine never tells tors about 45°, so nothing maps).
    #[test]
    fn layout_non_right_angle_rotate_is_the_unrotated_frame() {
        let bytes = layout_pdf("BT /F1 24 Tf 72 700 Td (Hello) Tj ET", "/Rotate 45");
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        let geometry = page.geometry.unwrap();
        assert_eq!(
            (geometry.rotation, geometry.width_pt, geometry.height_pt),
            (0, 612.0, 792.0)
        );
        assert_box_near(
            page.spans[0].bbox,
            (72.0, 700.0, 54.67, 24.0),
            0.01,
            "span at /Rotate 45",
        );
    }

    /// The INVERSE-MAP pins (T3, §9 item 3): on a /Rotate 90 page the
    /// engine pre-maps rotated-content spans about the MediaBox; tors
    /// inverse-maps them back to raw and carries them through the one
    /// forward rule, so they land in the SAME frame as the horizontal
    /// runs. Measured numbers (probe P2, pinned here):
    ///
    /// - the horizontal run (72,700) → (700, 485.33, 24, 54.67);
    /// - the rotated run `Tm [0 1 -1 0 300 300]` "Side" comes back from
    ///   the engine as (300, 263.98, 24, 48.02) (already mapped,
    ///   `page_rotation_applied=90`) → inverse → raw (300, 300, 48.02,
    ///   24) → forward → (300, 263.98, 24, 48.02): the round trip through
    ///   the two MediaBox-relative maps is exact (no crop: the frames
    ///   coincide);
    /// - the "Side" LINE box is the engine's degraded 48.02×48.02 SQUARE
    ///   (to_chars re-lays the run's glyphs horizontally over the mapped
    ///   box) — the T3 approximation, carried honestly by the span's
    ///   `rotation_degrees = 90`, never re-derived;
    /// - and on a CROPPED rotated page (the decisive case) the inverse is
    ///   load-bearing: the engine mapped about the MEDIA box, tors maps
    ///   about the VISIBLE box, so the rotated run at raw (250, 350) —
    ///   reported by the engine at (350, 313.98, 24, 48.02) — must land
    ///   at (250, 201.98, 24, 48.02), NOT the engine's verbatim number.
    #[test]
    fn layout_rotated_run_on_rotated_page_is_inverse_mapped_into_one_frame() {
        let bytes = layout_pdf(
            "BT /F1 24 Tf 72 700 Td (Hello) Tj ET\nBT /F1 24 Tf 0 1 -1 0 300 300 Tm (Side) Tj ET",
            "/Rotate 90",
        );
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        assert_eq!(page.spans.len(), 2);
        let hello = page.spans.iter().find(|s| s.text == "Hello").unwrap();
        let side = page.spans.iter().find(|s| s.text == "Side").unwrap();
        assert_box_near(
            hello.bbox,
            (700.0, 485.33, 24.0, 54.67),
            0.01,
            "horizontal run",
        );
        assert_box_near(side.bbox, (300.0, 263.98, 24.0, 48.02), 0.01, "rotated run");
        assert_eq!(side.rotation_degrees, 90.0, "the T3 honesty flag");
        assert_eq!(hello.rotation_degrees, 0.0);
        // the lines: the rotated run's own line is the degraded square
        assert_eq!(page.lines.len(), 2);
        let side_line = page.lines.iter().find(|l| l.text == "Side").unwrap();
        assert_box_near(
            side_line.bbox,
            (300.0, 263.98, 48.02, 48.02),
            0.01,
            "the T3 square line box, pinned as measured",
        );

        // The decisive case: the same page WITH a crop. The engine mapped
        // about the MediaBox; tors maps about the visible box.
        let bytes = layout_pdf(
            "BT /F1 24 Tf 200 400 Td (Hello) Tj ET\nBT /F1 24 Tf 0 1 -1 0 250 350 Tm (Side) Tj ET",
            "/Rotate 90 /CropBox [100 100 500 700]",
        );
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        let hello = page.spans.iter().find(|s| s.text == "Hello").unwrap();
        let side = page.spans.iter().find(|s| s.text == "Side").unwrap();
        assert_box_near(
            hello.bbox,
            (300.0, 245.33, 24.0, 54.67),
            0.01,
            "horizontal, cropped",
        );
        assert_box_near(
            side.bbox,
            (250.0, 201.98, 24.0, 48.02),
            0.01,
            "rotated run, cropped: inverse(MediaBox) then forward(visible)",
        );
        assert_eq!(side.rotation_degrees, 90.0);
    }

    /// The 270 inverse-map arm (§9 item 3 — a distinct branch of
    /// [`unmap_engine_box`], shipped untested by phase 2, probed correct
    /// by the red-team pass): the rotated-content run whose raw box is
    /// (300, 300, 48.02, 24) lands at (468, 300, 24, 48.02) on a
    /// /Rotate 270 page, and the CROPPED page's visible-box correction
    /// lands it at (376, 200, 24, 48.02) — the inverse is load-bearing
    /// there (the engine mapped about the MediaBox, tors about the
    /// visible box).
    #[test]
    fn layout_rotated_run_on_a_270_page_is_inverse_mapped() {
        let bytes = layout_pdf(
            "BT /F1 24 Tf 0 1 -1 0 300 300 Tm (Side) Tj ET",
            "/Rotate 270",
        );
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        assert_eq!(page.spans.len(), 1);
        let side = &page.spans[0];
        assert_box_near(side.bbox, (468.0, 300.0, 24.0, 48.02), 0.01, "the 270 arm");
        assert_eq!(side.rotation_degrees, 90.0, "the T3 honesty flag");

        // the decisive cropped case: the same content under the
        // [100 100 500 700] crop — the correction is visible in the answer.
        let bytes = layout_pdf(
            "BT /F1 24 Tf 0 1 -1 0 300 300 Tm (Side) Tj ET",
            "/Rotate 270 /CropBox [100 100 500 700]",
        );
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        let side = &page.spans[0];
        assert_box_near(
            side.bbox,
            (376.0, 200.0, 24.0, 48.02),
            0.01,
            "the 270 arm, cropped: inverse(MediaBox) then forward(visible)",
        );
    }

    /// The 180+crop golden (§9 item 3's rotation × crop cell): the
    /// crop-subset fixture's run (raw (150, 600, 64.03, 24)) on a
    /// /Rotate 180 page — the engine pre-maps ALL spans at 180, tors
    /// inverse-maps back to raw then forwards through the VISIBLE-box
    /// frame: (400−50−64.03, 600−500−24, 64.03, 24) = (285.97, 76,
    /// 64.03, 24).
    #[test]
    fn layout_rotation_180_with_crop_golden() {
        let bytes = layout_pdf(
            "BT /F1 24 Tf 150 600 Td (Inside) Tj ET",
            "/Rotate 180 /CropBox [100 100 500 700]",
        );
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        assert_eq!(page.spans.len(), 1);
        assert_box_near(
            page.spans[0].bbox,
            (285.97, 76.0, 64.03, 24.0),
            0.01,
            "180+crop: the inverse-then-forward round trip through the visible frame",
        );
    }

    /// The image pins (probe P5/P6/P7): the 1×1 red image is present via
    /// the HANDLES path (extract_images' silent <8×8 filter would return
    /// zero); its box is the forward rule over the always-raw handle
    /// bbox; a rotated CTM reports rotation 90; a /Rotate 90 page
    /// forward-normalizes the raw box ((200,200) → (200, 612−200−100));
    /// and a two-image page pins paint_order to content-stream order.
    #[test]
    fn layout_images_come_from_handles_and_take_the_forward_rule() {
        // P5: plain page, CTM [100 0 0 100 200 200].
        let bytes = image_pdf("100 0 0 100 200 200", "");
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        assert_eq!(
            page.images.len(),
            1,
            "the 1x1 image must surface via handles (T1)"
        );
        let image = &page.images[0];
        assert_box_near(
            image.bbox,
            (200.0, 200.0, 100.0, 100.0),
            0.01,
            "P5 image bbox",
        );
        assert_eq!((image.width_px, image.height_px), (1, 1));
        assert_eq!((image.rotation_degrees, image.paint_order), (0.0, 0));

        // P6: rotated CTM [0 100 -100 0 400 200] → (300, 200, 100, 100), rot 90.
        let bytes = image_pdf("0 100 -100 0 400 200", "");
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        let image = &page.images[0];
        assert_box_near(
            image.bbox,
            (300.0, 200.0, 100.0, 100.0),
            0.01,
            "P6 image bbox",
        );
        assert_eq!(image.rotation_degrees, 90.0);

        // P7: the same placement on a /Rotate 90 page: the handle stays
        // raw; tors forward-normalizes it to (200, 612−200−100, 100, 100).
        let bytes = image_pdf("100 0 0 100 200 200", "/Rotate 90");
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        let image = &page.images[0];
        assert_box_near(
            image.bbox,
            (200.0, 312.0, 100.0, 100.0),
            0.01,
            "P7 image bbox",
        );
        assert_eq!(
            image.rotation_degrees, 0.0,
            "the handle's CTM rotation, not the page's"
        );

        // Two images: paint_order is content-stream order (0, 1).
        let content = "q 50 0 0 50 100 100 cm /Im1 Do Q\nq 60 0 0 60 300 400 cm /Im1 Do Q";
        let stream = format!(
            "<< /Length {} >>\nstream\n{content}\nendstream",
            content.len() + 1
        );
        let bytes = objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /XObject << /Im1 5 0 R >> >> \
                 /MediaBox [0 0 612 792] /Contents 4 0 R >>",
            stream.as_bytes(),
            &red_1x1_image(),
        ]);
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        let orders: Vec<u32> = page.images.iter().map(|i| i.paint_order).collect();
        assert_eq!(orders, vec![0, 1], "paint_order is content-stream order");
        assert_box_near(
            page.images[0].bbox,
            (100.0, 100.0, 50.0, 50.0),
            0.01,
            "first image",
        );
        assert_box_near(
            page.images[1].bbox,
            (300.0, 400.0, 60.0, 60.0),
            0.01,
            "second image",
        );
    }

    /// The T12 no-drop pin: the engine's 2 pt visibility-drop law is the
    /// TEXT walk's only — an image wholly outside the visible box is
    /// still RETURNED, with out-of-range coordinates (raw (30, 30,
    /// 100, 100) on the [100 100 500 700] crop → (-70, -70, 100, 100)),
    /// while a text run at the same place is dropped. Pinned so the
    /// asymmetry cannot silently flip: a consumer must clip, and must
    /// not read "image present" as "image visible".
    #[test]
    fn layout_images_are_never_visibility_filtered() {
        let bytes = image_pdf("100 0 0 100 30 30", "/CropBox [100 100 500 700]");
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        assert!(
            page.spans.is_empty(),
            "a text run at (30, 30) is dropped (the TEXT walk's 2 pt law)"
        );
        assert_eq!(
            page.images.len(),
            1,
            "the same-position image is returned — the image walk has no filter"
        );
        assert_box_near(
            page.images[0].bbox,
            (-70.0, -70.0, 100.0, 100.0),
            0.01,
            "the outside image, coordinates out of range, never clipped",
        );
    }

    /// The scanned-page and unparseable-content-stream pins (§7's
    /// empty-lists-not-errors engine law): a page whose content stream is
    /// junk yields geometry + EMPTY spans/lines/images, never an error;
    /// the image-only page likewise (native text only; OCR out of
    /// scope).
    #[test]
    fn layout_unparseable_or_scanned_pages_yield_empty_lists_not_errors() {
        // A content stream that decodes but parses to nothing usable.
        let bytes = layout_pdf("this is not a content stream, just junk tokens", "");
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        assert!(page.geometry.is_some(), "the page box is intact");
        assert!(page.spans.is_empty() && page.lines.is_empty());
        // The scanned shape: a 64x64 gray image XObject, no text operators.
        let mut image_object: Vec<u8> = b"<< /Type /XObject /Subtype /Image /Width 64 /Height 64 \
             /ColorSpace /DeviceGray /BitsPerComponent 8 /Length 4096 >>\nstream\n"
            .to_vec();
        image_object.extend_from_slice(&vec![0u8; 4096]);
        image_object.extend_from_slice(b"\nendstream");
        let bytes = objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
                 /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length 27 >>\nstream\nq 612 0 0 792 0 0 cm /Im0 Do Q\nendstream",
            &image_object,
        ]);
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        assert!(
            page.spans.is_empty() && page.lines.is_empty(),
            "a scan has no native text"
        );
        assert_eq!(
            page.images.len(),
            1,
            "the scan's own image surfaces via handles"
        );
    }

    /// T8's two remaining shapes, pinned (the broken-windows sweep): a
    /// page whose /Contents stream declares /FlateDecode but carries
    /// garbage (a DECODE failure), and one whose /Contents points at an
    /// object that does not exist (a dangling ref). Both are the
    /// `unwrap_or_default()` law made permanent: geometry intact, EMPTY
    /// spans/lines/images, never an error — and both content surfaces
    /// (the span parse AND the line walk) consistent with the image
    /// walk's emptiness, so no surface can resurrect a failure as data.
    #[test]
    fn t8_broken_flate_and_dangling_contents_yield_empty_lists_not_errors() {
        // the broken-Flate stream: /Filter /FlateDecode over bytes that
        // are not zlib (the header byte alone refuses to inflate).
        let junk = b"this is not zlib data, not even close";
        let broken = format!(
            "<< /Length {} /Filter /FlateDecode >>\nstream\n{}",
            junk.len(),
            std::str::from_utf8(junk).unwrap()
        );
        let broken_flate = objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
            broken.as_bytes(),
        ]);
        let page = &page_layout(broken_flate, None, None, "pdf").unwrap()[0];
        assert!(page.geometry.is_some(), "the page box is intact");
        assert!(
            page.spans.is_empty() && page.lines.is_empty() && page.images.is_empty(),
            "a broken-Flate /Contents is the empty shape, not an error"
        );

        // the dangling /Contents ref: the page dict names an object the
        // file does not carry.
        let dangling = objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 99 0 R >>",
        ]);
        let page = &page_layout(dangling, None, None, "pdf").unwrap()[0];
        assert!(page.geometry.is_some(), "the page box is intact");
        assert!(
            page.spans.is_empty() && page.lines.is_empty() && page.images.is_empty(),
            "a dangling /Contents ref is the empty shape, not an error"
        );
    }

    /// The artifact pin (D6 + T11): a span inside a declared `/Artifact`
    /// marked-content sequence reports `is_artifact = true` ONLY in the
    /// typed form whose `/Type`/`/Subtype` the engine recognizes; the
    /// body span `false`. The BMC form is pinned FALSE — deliberately, so
    /// the engine limitation cannot be silently "fixed" without this test
    /// forcing a spec re-read: the engine's BMC handler sets its internal
    /// inside-artifact bit but leaves `artifact_type` `None`
    /// (extractors/text.rs:6487-6489), and the span carries no other
    /// artifact bit — so a bare `/Artifact BMC` and an unrecognized
    /// `/Type` are indistinguishable from body on this surface.
    /// `is_artifact == false` is NOT evidence of body text.
    #[test]
    fn layout_declared_artifact_spans_are_flagged() {
        let typed = "/Artifact << /Type /Pagination /Subtype /Header >> BDC\nBT /F1 24 Tf 72 700 Td (Header) Tj ET\nEMC\nBT /F1 12 Tf 72 600 Td (Body) Tj ET";
        let bytes = layout_pdf(typed, "");
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        assert_eq!(page.spans.len(), 2);
        let header = page.spans.iter().find(|s| s.text == "Header").unwrap();
        let body = page.spans.iter().find(|s| s.text == "Body").unwrap();
        assert!(
            header.is_artifact,
            "the /Artifact-tagged run is declared furniture"
        );
        assert!(!body.is_artifact);

        // The BMC form: furniture the engine cannot classify on this
        // surface — pinned false (T11's limitation pin).
        let bytes = layout_pdf(
            "/Artifact BMC\nBT /F1 24 Tf 72 700 Td (Header) Tj ET\nEMC\nBT /F1 12 Tf 72 600 Td (Body) Tj ET",
            "",
        );
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        assert_eq!(page.spans.len(), 2);
        let header = page.spans.iter().find(|s| s.text == "Header").unwrap();
        assert!(
            !header.is_artifact,
            "a bare /Artifact BMC is NOT flagged: the engine sets no artifact \
             type for it — the limitation pin, not a bug to fix silently"
        );
        // and the unrecognized /Type: typed form, engine-noisy value — false.
        let bytes = layout_pdf(
            "/Artifact << /Type /Foo >> BDC\nBT /F1 24 Tf 72 700 Td (Header) Tj ET\nEMC\nBT /F1 12 Tf 72 600 Td (Body) Tj ET",
            "",
        );
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        let header = page.spans.iter().find(|s| s.text == "Header").unwrap();
        assert!(!header.is_artifact, "an unrecognized /Type is not flagged");
    }

    /// The bold-mid-line pin (D3): two runs on one baseline — one bold —
    /// merge into ONE line (lines merge where spans split): two spans
    /// with the style evidence split per run, one line whose words join
    /// to its text.
    #[test]
    fn layout_bold_mid_line_spans_split_where_lines_merge() {
        let bytes = bold_mid_line_pdf();
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        assert_eq!(
            page.spans.len(),
            2,
            "the runs stay separate spans (font evidence)"
        );
        let hello = page.spans.iter().find(|s| s.text == "Hello").unwrap();
        let world = page.spans.iter().find(|s| s.text == "World").unwrap();
        assert!(!hello.is_bold);
        assert!(
            world.is_bold,
            "the /Helvetica-Bold run is the bold evidence"
        );
        assert_eq!(page.lines.len(), 1, "the two runs are ONE line");
        let line = &page.lines[0];
        assert_eq!(line.words.len(), 2);
        assert_box_near(
            line.bbox,
            (72.0, 700.0, 135.99, 24.0),
            0.02,
            "the merged line box",
        );
        // the join law: the line text is the words' texts joined with spaces
        let joined = line
            .words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(joined, line.text);
        // and the invariant that keeps the single-space law true across
        // engine bumps: no word's text carries whitespace (the engine's
        // line builder drops every whitespace character before forming a
        // word), so the separator is always exactly one U+0020
        for word in &line.words {
            assert!(
                !word.text.chars().any(char::is_whitespace),
                "a word's text carries whitespace: {:?}",
                word.text
            );
        }
        // the union law: every word's box is inside the line's box
        for word in &line.words {
            assert!(word.bbox.0 >= line.bbox.0 - 0.01 && word.bbox.1 >= line.bbox.1 - 0.01);
            assert!(word.bbox.0 + word.bbox.2 <= line.bbox.0 + line.bbox.2 + 0.01);
            assert!(word.bbox.1 + word.bbox.3 <= line.bbox.1 + line.bbox.3 + 0.01);
        }
    }

    /// The pages= law on the layout probe: `None` = whole document; a
    /// selection normalizes (sort + dedup) and the output is parallel to
    /// the NORMALIZED selection with every entry carrying its
    /// `page_index`; the empty and out-of-range refusals are
    /// `markdown_pages`' byte-identical strings.
    #[test]
    fn layout_pages_selection_is_normalized_and_parallel() {
        let mut objects = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>".to_vec(),
        ];
        for text in ["one", "two", "three"] {
            let page_number = objects.len() + 1;
            let font_number = page_number + 1;
            let stream_number = page_number + 2;
            let content = format!("BT /F1 12 Tf 50 700 Td ({text}) Tj ET");
            objects.push(
                format!(
                    "<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 {font_number} 0 R >> >> \
                 /MediaBox [0 0 612 792] /Contents {stream_number} 0 R >>"
                )
                .into_bytes(),
            );
            objects.push(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec());
            objects.push(
                format!(
                    "<< /Length {} >>\nstream\n{content}\nendstream",
                    content.len() + 1
                )
                .into_bytes(),
            );
        }
        let refs: Vec<&[u8]> = objects.iter().map(Vec::as_slice).collect();
        let bytes = objects_pdf(&refs);
        let whole = page_layout(bytes.clone(), None, None, "pdf").unwrap();
        assert_eq!(
            whole.iter().map(|p| p.page_index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        // [2, 0] normalizes to [0, 2]; every entry carries its page_index.
        let subset = page_layout(bytes.clone(), Some(&[2, 0]), None, "pdf").unwrap();
        assert_eq!(
            subset.iter().map(|p| p.page_index).collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(subset[0].spans[0].text, "one");
        assert_eq!(subset[1].spans[0].text, "three");
        // duplicates normalize away
        assert_eq!(
            page_layout(bytes.clone(), Some(&[1, 1]), None, "pdf").unwrap()[0].page_index,
            1
        );
        // the byte-pinned refusals
        let err = page_layout(bytes.clone(), Some(&[]), None, "pdf").unwrap_err();
        assert_eq!(
            err.to_string(),
            "pages= selected no pages: pass None for the whole document"
        );
        let err = page_layout(bytes, Some(&[3]), None, "pdf").unwrap_err();
        assert_eq!(
            err.to_string(),
            "page index 3 is out of range for a 3-page document (pages= is 0-based)"
        );
    }

    /// The composition property at BOX level (§9 item 3): the rotate step
    /// composed with its inverse angle — r then (360−r) — is the identity,
    /// where the second application lives in the ROTATED frame's own
    /// extents (the rule's step 3: the frame swaps for odd quarters);
    /// and the canonical pairs agree: 90+270 ≡ 180+180 for every box,
    /// within f32 tolerance.
    #[test]
    fn layout_rotation_composes_with_its_inverse_to_the_identity() {
        let (w, h) = (612.0f32, 792.0f32);
        let boxes = [
            (0.0f32, 0.0f32, 612.0f32, 792.0f32),
            (37.5, 123.25, 61.75, 42.5),
            (72.0, 700.0, 54.67, 24.0),
            // unclipped / negative shapes too: the law is total
            (-30.0, -10.0, 700.0, 900.0),
        ];
        for b in boxes {
            for r in [0, 90, 180, 270] {
                let first = rotate_box_forward(b, (w, h), r);
                let (w2, h2) = if r == 90 || r == 270 { (h, w) } else { (w, h) };
                let second = rotate_box_forward(first, (w2, h2), (360 - r) % 360);
                assert!(
                    near(second.0, b.0, 1e-3)
                        && near(second.1, b.1, 1e-3)
                        && near(second.2, b.2, 1e-3)
                        && near(second.3, b.3, 1e-3),
                    "r={r}, b={b:?}: r then inverse must be the identity, got {second:?}"
                );
            }
        }
        // 90+270 ≡ 180+180 (both compositions are the identity, so they
        // agree with each other for every box).
        for b in boxes {
            let p90 = rotate_box_forward(rotate_box_forward(b, (w, h), 90), (h, w), 270);
            let p180 = rotate_box_forward(rotate_box_forward(b, (w, h), 180), (w, h), 180);
            assert!(
                near(p90.0, p180.0, 1e-3)
                    && near(p90.1, p180.1, 1e-3)
                    && near(p90.2, p180.2, 1e-3)
                    && near(p90.3, p180.3, 1e-3),
                "b={b:?}: 90+270 = {p90:?} must equal 180+180 = {p180:?}"
            );
        }
    }

    // --- the render-anchored oracle (§9): the correctness anchor that
    // --- replaces assay's rotate_bbox -------------------------------

    /// The ink bounding box of a rendered page (raw RGBA8888, top-left
    /// origin), as `Option<corners>` — pixels where the channel test
    /// passes. The oracle's raster side.
    fn ink_corners(
        image: &pdf_oxide::rendering::RenderedImage,
        channel: impl Fn(u8, u8, u8) -> bool,
    ) -> Option<(f64, f64, f64, f64)> {
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        for py in 0..image.height {
            for px in 0..image.width {
                let i = ((py * image.width + px) * 4) as usize;
                if channel(image.data[i], image.data[i + 1], image.data[i + 2]) {
                    x0 = x0.min(px);
                    y0 = y0.min(py);
                    x1 = x1.max(px);
                    y1 = y1.max(py);
                }
            }
        }
        if x0 == u32::MAX {
            return None;
        }
        Some((x0 as f64, y0 as f64, (x1 + 1) as f64, (y1 + 1) as f64))
    }

    /// Render page 0 at 72 dpi (1pt = 1px, the known scale) and return
    /// the ink corners in RENDER SPACE (bottom-left origin, y-up) — the
    /// raster's top-left origin flipped.
    fn render_ink(
        doc: &PdfDocument,
        channel: impl Fn(u8, u8, u8) -> bool,
    ) -> ((u32, u32), (f64, f64, f64, f64)) {
        let image = pdf_oxide::rendering::render_page(
            doc,
            0,
            &pdf_oxide::rendering::RenderOptions::with_dpi(72).as_raw(),
        )
        .unwrap();
        let (width, height) = (image.width, image.height);
        let corners = ink_corners(&image, channel).expect("the oracle page must carry ink");
        (
            (width, height),
            (
                corners.0,
                height as f64 - corners.3,
                corners.2,
                height as f64 - corners.1,
            ),
        )
    }

    /// THE RENDER-ANCHORED ORACLE (§9): for each angle × {no crop, crop} ×
    /// {tight glyph, asymmetric marker}, the box the probe declares must
    /// land where pdf_oxide's own renderer draws the ink.
    ///
    /// - the **asymmetric marker** (a filled rect, off-center, portrait-
    ///   asymmetric dims 100×60) pins the TRANSFORM independent of any
    ///   span parse: the declared raw rect maps through the forward rule
    ///   and the rendered ink's corners equal the mapped corners to the
    ///   pixel. A mirrored-y mapping (assay's rotate_bbox defect) or a
    ///   transposed one lands the box on the wrong side of the page and
    ///   fails here.
    /// - the **tight glyph** ('H', no descender, cap-height ink) pins the
    ///   SURFACE's span boxes to the render: the em-box the probe returns
    ///   must contain the glyph's ink (the em-box law: baseline at the
    ///   bottom edge, one font-size upward — descenders would extend
    ///   below, 'H' does not).
    ///
    /// This is the test that makes the D1 transform's correctness an
    /// empirical fact rather than an internal consistency argument: the
    /// renderer is the one consumer whose output nobody disputes.
    #[test]
    fn the_render_anchored_oracle_lands_every_mapped_box_where_the_render_draws_it() {
        let dark = |r: u8, g: u8, b: u8| r < 128 && g < 128 && b < 128;
        for angle in [0, 90, 180, 270] {
            for (crop, extras, visible) in [
                (false, "", (0.0f32, 0.0f32, 612.0f32, 792.0f32)),
                (
                    true,
                    "/CropBox [100 100 500 700]",
                    (100.0, 100.0, 500.0, 700.0),
                ),
            ] {
                let rotation = angle;
                // The page's own rotation rides in the page dict.
                let page_extras = match angle {
                    0 => extras.to_string(),
                    a => {
                        if extras.is_empty() {
                            format!("/Rotate {a}")
                        } else {
                            format!("/Rotate {a} {extras}")
                        }
                    }
                };
                // Marker placement: fully inside the crop when cropped
                // (the renderer clips to the visible box; the oracle
                // needs the whole marker visible).
                let (mx, my, mw, mh): (f32, f32, f32, f32) = if crop {
                    (150.0, 500.0, 80.0, 40.0)
                } else {
                    (400.0, 300.0, 100.0, 60.0)
                };
                let (glyph_x, glyph_y): (f32, f32) =
                    if crop { (150.0, 400.0) } else { (72.0, 700.0) };

                // -- the asymmetric marker: `re f` fill, ink corners are
                //    the mapped corners (dpi 72, 1pt = 1px).
                let content = format!("{mx} {my} {mw} {mh} re f");
                let bytes = layout_pdf(&content, &page_extras);
                let doc = PdfDocument::from_bytes(bytes.clone()).unwrap();
                let ((render_w, render_h), ink) = render_ink(&doc, dark);
                let mapped = map_box_forward((mx, my, mw, mh), visible, rotation);
                let geometry = page_geometry(bytes, None).unwrap().remove(0).unwrap();
                assert_eq!(
                    (render_w, render_h),
                    (geometry.width_pt as u32, geometry.height_pt as u32),
                    "angle {angle} crop {crop}: the render's pixel dims are the declared extents"
                );
                assert_box_near(
                    mapped,
                    (
                        ink.0 as f32,
                        ink.1 as f32,
                        (ink.2 - ink.0) as f32,
                        (ink.3 - ink.1) as f32,
                    ),
                    1.0,
                    &format!("angle {angle} crop {crop}: the mapped marker must equal the ink"),
                );

                // -- the tight glyph: the probe's span em-box contains
                //    the glyph's ink (with a 1.5px rounding allowance at
                //    72dpi).
                let content = format!("BT /F1 24 Tf {glyph_x} {glyph_y} Td (H) Tj ET");
                let bytes = layout_pdf(&content, &page_extras);
                let doc = PdfDocument::from_bytes(bytes).unwrap();
                let (_, ink) = render_ink(&doc, dark);
                let page = &page_layout_bytes(doc).remove(0);
                let span = &page.spans[0];
                let (bx, by, bw, bh) = span.bbox;
                assert!(
                    bx as f64 - 1.5 <= ink.0
                        && by as f64 - 1.5 <= ink.1
                        && ink.2 <= (bx + bw) as f64 + 1.5
                        && ink.3 <= (by + bh) as f64 + 1.5,
                    "angle {angle} crop {crop}: the glyph ink ({ink:?}) must sit inside the \
                     mapped em-box ({bx},{by},{bw},{bh})"
                );
            }
        }
    }

    /// The oracle's image leg: the red 1×1 image's MAPPED box lands where
    /// the render draws the red pixels — the same 4-angle × {no crop,
    /// crop} matrix the text oracle's marker leg runs (the handle is
    /// always raw; the forward rule normalizes; the render verifies each
    /// leg). The raw placement (200, 200, 100, 100) sits fully inside the
    /// visible box on every leg — the [100 100 500 700] crop included —
    /// so the render shows the whole image and the ink comparison is
    /// exact; the /Rotate 90 no-crop leg is phase-2's golden
    /// (200, 312, 100, 100), the other angles derived by the forward rule.
    #[test]
    fn the_render_anchored_oracle_lands_mapped_image_boxes_where_the_render_draws_them() {
        let red = |r: u8, g: u8, b: u8| r > 200 && g < 100 && b < 100;
        for angle in [0, 90, 180, 270] {
            for (crop, extras, visible) in [
                (false, "", (0.0f32, 0.0f32, 612.0f32, 792.0f32)),
                (
                    true,
                    "/CropBox [100 100 500 700]",
                    (100.0, 100.0, 500.0, 700.0),
                ),
            ] {
                let rotation = angle;
                // The page's own rotation rides in the page dict, the
                // crop fragment after it — the text oracle's exact shape.
                let page_extras = match angle {
                    0 => extras.to_string(),
                    a => {
                        if extras.is_empty() {
                            format!("/Rotate {a}")
                        } else {
                            format!("/Rotate {a} {extras}")
                        }
                    }
                };
                let bytes = image_pdf("100 0 0 100 200 200", &page_extras);
                let doc = PdfDocument::from_bytes(bytes.clone()).unwrap();
                let ((render_w, render_h), ink) = render_ink(&doc, red);
                let geometry = page_geometry(bytes, None).unwrap().remove(0).unwrap();
                assert_eq!(
                    (render_w, render_h),
                    (geometry.width_pt as u32, geometry.height_pt as u32),
                    "angle {angle} crop {crop}: the render's pixel dims are the declared extents"
                );
                let image = &page_layout_bytes(doc).remove(0).images[0];
                // The mapped box must equal the red ink's box, and (the
                // forward-normalization half of the pin) it must NOT be
                // the raw handle box when the page is rotated: the
                // /Rotate 90 no-crop leg's image lands at
                // (200, 612−200−100, 100, 100).
                assert_box_near(
                    image.bbox,
                    (
                        ink.0 as f32,
                        ink.1 as f32,
                        (ink.2 - ink.0) as f32,
                        (ink.3 - ink.1) as f32,
                    ),
                    1.0,
                    &format!(
                        "angle {angle} crop {crop}: the mapped image box must equal the red ink"
                    ),
                );
                let expected = map_box_forward((200.0, 200.0, 100.0, 100.0), visible, rotation);
                assert_box_near(
                    image.bbox,
                    expected,
                    0.01,
                    &format!("angle {angle} crop {crop}: the mapped image box is the forward rule"),
                );
            }
        }
    }

    /// The oracle's sideways-run case (§9's phase-2 addendum b): rotated
    /// CONTENT on a /Rotate 90 page — and (the broken-windows sweep's
    /// extension) the /Rotate 270 arm with the crop, the other angle the
    /// addendum names, whose declared box is the 270-inverse-map golden
    /// `(376, 200, 24, 48.02)`. The declared box is pinned to the
    /// engine's flattened numbers AND asserted NOT to contain the render
    /// ink — an explicit wrongness pin (the documented T3 mismatch), not
    /// a containment assertion, which would pass by accident and hide a
    /// wrong transposition. This is the one oracle leg that is SUPPOSED
    /// to be wrong: `rotation_degrees != 0` means the box must not be
    /// used as a highlight, and this test keeps that wrongness visible
    /// (an engine bump that quietly fixes it must re-open the spec
    /// caveat, not flip a green test).
    #[test]
    fn the_oracle_pins_the_sideways_run_s_documented_mismatch() {
        let dark = |r: u8, g: u8, b: u8| r < 128 && g < 128 && b < 128;
        let wrongness_pin = |page: &PageLayoutData, ink: (f64, f64, f64, f64), what: &str| {
            assert_eq!(page.spans.len(), 1, "{what}");
            let side = &page.spans[0];
            assert_eq!(side.rotation_degrees, 90.0, "the T3 honesty flag: {what}");
            // the wrongness pin: the render's ink is NOT inside the declared
            // box (the sideways run's true ink is a perpendicular strip the
            // flattened box misses — the documented T3 mismatch).
            let (bx, by, bw, bh) = side.bbox;
            let contained = ink.0 >= bx as f64 - 1.5
                && ink.1 >= by as f64 - 1.5
                && ink.2 <= (bx + bw) as f64 + 1.5
                && ink.3 <= (by + bh) as f64 + 1.5;
            assert!(
                !contained,
                "{what}: the T3 mismatch must stay visible: the flattened box \
                 ({bx},{by},{bw},{bh}) must NOT contain the render ink {ink:?} — if the \
                 engine now maps sideways runs truly, re-open the spec caveat instead of \
                 relaxing this pin"
            );
        };
        let bytes = layout_pdf(
            "BT /F1 24 Tf 0 1 -1 0 300 300 Tm (Side) Tj ET",
            "/Rotate 90",
        );
        let doc = PdfDocument::from_bytes(bytes).unwrap();
        let (_, ink) = render_ink(&doc, dark);
        let page = &page_layout_bytes(doc).remove(0);
        assert_eq!(page.spans.len(), 1);
        let side = &page.spans[0];
        assert_eq!(side.rotation_degrees, 90.0, "the T3 honesty flag");
        // the pinned flattened numbers (the same golden the inverse-map
        // test pins: the engine's pre-mapped box round-trips exactly)
        assert_box_near(
            side.bbox,
            (300.0, 263.98, 24.0, 48.02),
            0.01,
            "the flattened box, pinned as measured",
        );
        wrongness_pin(page, ink, "/Rotate 90");

        // the 270 arm, with the crop: the sideways run at raw (300, 300)
        // on a /Rotate 270 page under the [100 100 500 700] CropBox — the
        // declared box is the 270 inverse-map golden (376, 200, 24, 48.02)
        // and the render ink (the true perpendicular strip, whole inside
        // the crop) is NOT inside it.
        let bytes = layout_pdf(
            "BT /F1 24 Tf 0 1 -1 0 300 300 Tm (Side) Tj ET",
            "/Rotate 270 /CropBox [100 100 500 700]",
        );
        let doc = PdfDocument::from_bytes(bytes).unwrap();
        let (_, ink) = render_ink(&doc, dark);
        let page = &page_layout_bytes(doc).remove(0);
        assert_box_near(
            page.spans[0].bbox,
            (376.0, 200.0, 24.0, 48.02),
            0.01,
            "/Rotate 270 + crop: the declared box, pinned as measured",
        );
        wrongness_pin(page, ink, "/Rotate 270 + crop");
    }

    /// A WMode-1 (tategaki) fixture: a Type0 font with /Encoding
    /// /Identity-V (the predefined-CMap-name vertical signal), a
    /// non-embedded CIDFontType2 descendant (/DW 1000), and a ToUnicode
    /// CMap mapping CIDs 0x41/0x42 to 'A'/'B'. Object numbering:
    /// catalog 1, pages 2, page 3, Type0 font 4, content 5, CIDFont 6,
    /// ToUnicode 7, the horizontal-control Helvetica font 8.
    fn wmode1_pdf(content: &str, page_extras: &str, control: bool) -> Vec<u8> {
        let page_dict = format!(
            "<< /Type /Page /Parent 2 0 R /Resources \
             << /Font << /F1 4 0 R /F2 8 0 R >> >> \
             /MediaBox [0 0 612 792] {page_extras} /Contents 5 0 R >>"
        );
        let content_stream = format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            content.len() + 1,
            content
        );
        let mut objects: Vec<&[u8]> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            page_dict.as_bytes(),
            b"<< /Type /Font /Subtype /Type0 /BaseFont /TestVert /Encoding /Identity-V \
               /DescendantFonts [6 0 R] /ToUnicode 7 0 R >>",
            content_stream.as_bytes(),
            b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /TestVert /CIDSystemInfo \
               << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /DW 1000 >>",
            b"<< /Length 285 >>\nstream\n/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n/CMapName /Test-V def\n1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n2 beginbfchar\n<0041> <0041>\n<0042> <0042>\nendbfchar\nendcmap\nend\nend\nendstream",
        ];
        if control {
            objects.push(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
        }
        objects_pdf(&objects)
    }

    /// T9, VERIFIED (the broken-windows sweep; was "known-unverified"):
    /// the honesty flag does NOT cover vertical writing. A WMode-1 run —
    /// the engine reads the writing mode from the FONT on Tf
    /// (extractors/text.rs:5307-5319 ← resolve_encoding_writing_mode,
    /// font_dict.rs:2316-2334; a predefined CMap name ending in `-V`
    /// declares vertical, font_dict.rs:355-364) — carries
    /// `rotation_degrees == 0.0` (the angle is the text MATRIX's alone,
    /// snap_run_rotation text.rs:1949-1979; wmode plays no part) while
    /// its box is the flattened fiction at its WORST: the run's vertical
    /// advance (w1y = -1000/1000 em per glyph, the spec /DW2 default,
    /// routed into the y column by advance_text_matrix
    /// graphics_state.rs:449-458) is flattened onto x as a NEGATIVE
    /// width by both flush sites (text.rs:7290-7295, 8900-8906). The
    /// measured pins: a 24pt 2-glyph run at (300, 700) reports span
    /// bbox (300, 700, -48, 24) — the true ink is a vertical column
    /// (the engine's own per-glyph em-boxes step DOWN 24pt per glyph:
    /// 'A' at (300, 700, 24, 24), 'B' at (300, 676, 24, 24); the
    /// renderer's ink: (288, 655, 304, 695) at 72dpi), ZERO overlap
    /// with the declared box; the line surface re-lays the
    /// glyphs over the normalized box and reports the REVERSED text
    /// ("BA" for "AB", box (252, 700, 48, 24)). The engine's span
    /// carries wmode == 1 but wmode is cut from this surface (§5/D6).
    /// The render-anchored wrongness arm (the sideways oracle's form):
    /// the declared box must NOT contain the render ink — the renderer
    /// paints vertical writing honestly (its rasterizer branches on
    /// text_wmode, text_rasterizer.rs:1338), so the miss is real, and an
    /// engine bump that starts mapping vertical runs truly must re-open
    /// the spec caveat, not flip a green test.
    #[test]
    fn t9_vertical_writing_runs_carry_the_flattened_box_and_zero_flag() {
        let bytes = wmode1_pdf("BT /F1 24 Tf 300 700 Td <00410042> Tj ET", "", false);
        let page = &page_layout(bytes.clone(), None, None, "pdf").unwrap()[0];
        assert_eq!(page.spans.len(), 1);
        let span = &page.spans[0];
        assert_eq!(span.text, "AB");
        assert_eq!(
            span.rotation_degrees, 0.0,
            "the T9 gap: the flag does not fire"
        );
        assert_box_near(
            span.bbox,
            (300.0, 700.0, -48.0, 24.0),
            0.01,
            "the flattened vertical box, pinned as measured (NEGATIVE width)",
        );
        // the line surface: the re-laid, normalized, REVERSED shape
        assert_eq!(page.lines.len(), 1);
        assert_eq!(
            page.lines[0].text, "BA",
            "the line's re-lay reverses the run"
        );
        assert_box_near(
            page.lines[0].bbox,
            (252.0, 700.0, 48.0, 24.0),
            0.01,
            "the line box, pinned as measured (normalized, left of the anchor)",
        );

        // the wrongness arm: the render paints the column honestly
        // (glyphs upright, stacked downward from (300, 700)); the
        // declared box must NOT contain it.
        let doc = PdfDocument::from_bytes(bytes.clone()).unwrap();
        let dark = |r: u8, g: u8, b: u8| r < 128 && g < 128 && b < 128;
        let (_, ink) = render_ink(&doc, dark);
        let (bx, by, bw, bh) = span.bbox;
        let contained = ink.0 >= bx as f64 - 1.5
            && ink.1 >= by as f64 - 1.5
            && ink.2 <= (bx + bw) as f64 + 1.5
            && ink.3 <= (by + bh) as f64 + 1.5;
        assert!(
            !contained,
            "the T9 mismatch must stay visible: the flattened box ({bx},{by},{bw},{bh}) \
             must NOT contain the render ink {ink:?} — if the engine now maps vertical \
             runs truly, re-open the spec caveat instead of relaxing this pin"
        );

        // and on a /Rotate 90 page the vertical run is NOT engine-premapped
        // (page_rotation_applied stays 0 — the mapping keys on
        // rotation_degrees, document.rs:13086-13092), so the same fiction
        // rides the forward map into the rotated frame: (300, 700, -48,
        // 24) → (700, 612-300-(-48), 24, -48).
        let bytes = wmode1_pdf(
            "BT /F1 24 Tf 300 700 Td <00410042> Tj ET",
            "/Rotate 90",
            false,
        );
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        assert_eq!(page.spans[0].rotation_degrees, 0.0);
        assert_box_near(
            page.spans[0].bbox,
            (700.0, 360.0, 24.0, -48.0),
            0.01,
            "/Rotate 90: the flattened fiction forward-mapped, pinned as measured",
        );
    }

    /// The assay `rotate_bbox` differential, pinned as the migration test
    /// it is: xfail at rotation 0 (assay's rotation-0 branch bakes the
    /// mirrored-y defect in — the D2 resolution, probe-proven), PASSING
    /// at 180 (its mirror formula coincides with the forward rule's 180
    /// arm), where the ONLY expected divergences are the ones the spec
    /// lists: intersection-vs-raw crop handling, normalized-vs-raw crop
    /// corners, and non-negative extents. The test documents the
    /// migration law: tors's goldens are the oracle; assay must adopt
    /// identity-at-0 before its differential can go green.
    #[test]
    fn the_assay_rotate_bbox_differential_is_xfail_at_0_and_green_at_180() {
        let assay_rotate_bbox =
            |x: f32, y: f32, w: f32, h: f32, rotation: i32, page_w: f32, page_h: f32| {
                // assay's formula (pdf_ingest.rotate_bbox): for rotation 0 it
                // mirrors y (the defect); for 180 it mirrors both axes.
                match rotation {
                    0 => (x, page_h - y - h, w, h),
                    180 => (page_w - x - w, page_h - y - h, w, h),
                    _ => unimplemented!("the differential pins 0 and 180 only"),
                }
            };
        // The probe golden: Td(72,700)/24pt on 612×792 → (72, 700, 54.67, 24).
        let bytes = layout_pdf("BT /F1 24 Tf 72 700 Td (Hello) Tj ET", "");
        let page = &page_layout(bytes, None, None, "pdf").unwrap()[0];
        let tors_bbox = page.spans[0].bbox;
        assert_box_near(tors_bbox, (72.0, 700.0, 54.67, 24.0), 0.01, "tors golden");
        // assay's rotation-0 branch: (72, 792−700−24=68, ...) — mirrored. The
        // expected divergence, pinned: this is WHY the differential is
        // xfail-at-0 and why assay's fix is identity-after-crop-translation.
        let assay = assay_rotate_bbox(72.0, 700.0, 54.67, 24.0, 0, 612.0, 792.0);
        assert_eq!(assay, (72.0, 68.0, 54.67, 24.0));
        assert!(
            !near(assay.1, tors_bbox.1, 0.01),
            "the mirrored-y defect must stay visible here until assay fixes its rotation-0 branch"
        );
        // At 180 the two formulas coincide — the one green leg.
        let assay = assay_rotate_bbox(72.0, 700.0, 54.67, 24.0, 180, 612.0, 792.0);
        assert_box_near(assay, (485.33, 68.0, 54.67, 24.0), 0.01, "180 coincides");
    }

    /// Helper for the oracle tests: rebuild a `PdfDocument` from bytes
    /// the tests already built (the probe takes bytes; the oracle needs
    /// the doc twice — render + layout — over the SAME open).
    fn page_layout_bytes(doc: PdfDocument) -> Vec<PageLayoutData> {
        (0..doc.page_count().unwrap())
            .map(|page| page_layout_one(&doc, page, "pdf"))
            .collect()
    }

    /// A one-page pdf whose page dict carries `page_extras` (the
    /// MediaBox/Rotate/CropBox fragments a geometry fixture needs). No
    /// content stream: the page tree is the whole probe.
    fn geometry_pdf(page_extras: &str) -> Vec<u8> {
        objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            format!("<< /Type /Page /Parent 2 0 R {page_extras} >>").as_bytes(),
        ])
    }

    /// The plain portrait page's dict fragment (the letter-size default the
    /// other fixtures deviate from).
    const PORTRAIT: &str = "/MediaBox [0 0 612 792]";

    /// Rotation 0 is the identity on this surface — the probe goldens' D2
    /// pin, geometry side: the declared extent verbatim (612×792, never
    /// swapped), rotation 0, and — the no-y-flip half of the pin — a
    /// declared CropBox's corners come back as the y-up source-frame
    /// rectangle the writer wrote, never mirrored about the page.
    #[test]
    fn rotation_0_is_the_identity_and_crop_corners_stay_y_up() {
        let geometry = page_geometry(geometry_pdf(PORTRAIT), None).unwrap();
        assert_eq!(
            geometry,
            vec![Some(PageBox {
                width_pt: 612.0,
                height_pt: 792.0,
                rotation: 0,
                crop: None,
            })]
        );
        let cropped = page_geometry(
            geometry_pdf("/MediaBox [0 0 612 792] /CropBox [100 100 500 700]"),
            None,
        )
        .unwrap();
        assert_eq!(cropped[0].unwrap().crop, Some((100.0, 100.0, 500.0, 700.0)));
    }

    /// The nonzero-MediaBox-origin golden (probe P4: `[50 50 662 842]`):
    /// the extent is the box's width/height (urx−llx, ury−lly), not the
    /// corners, and the origin is not re-added — 612×792, crop None (a
    /// page with no CropBox has visible == media). With a CropBox too, the
    /// corners survive the nonzero origin verbatim.
    #[test]
    fn the_nonzero_media_box_origin_golden() {
        let geometry = page_geometry(geometry_pdf("/MediaBox [50 50 662 842]"), None).unwrap();
        let entry = geometry[0].unwrap();
        assert_eq!((entry.width_pt, entry.height_pt), (612.0, 792.0));
        assert_eq!(entry.rotation, 0);
        assert_eq!(entry.crop, None);
        let cropped = page_geometry(
            geometry_pdf("/MediaBox [50 50 662 842] /CropBox [100 100 500 700]"),
            None,
        )
        .unwrap();
        let entry = cropped[0].unwrap();
        assert_eq!((entry.width_pt, entry.height_pt), (400.0, 600.0));
        assert_eq!(entry.crop, Some((100.0, 100.0, 500.0, 700.0)));
    }

    /// The crop-subset golden (§9 item 1, probe P3): a CropBox strictly
    /// inside the MediaBox crops the page — the render extent is the
    /// CropBox's own extent (400×600) and the crop provenance is the
    /// declared box, resolved and normalized.
    #[test]
    fn the_crop_subset_golden() {
        let geometry = page_geometry(
            geometry_pdf("/MediaBox [0 0 612 792] /CropBox [100 100 500 700]"),
            None,
        )
        .unwrap();
        assert_eq!(
            geometry[0],
            Some(PageBox {
                width_pt: 400.0,
                height_pt: 600.0,
                rotation: 0,
                crop: Some((100.0, 100.0, 500.0, 700.0)),
            })
        );
    }

    /// The partial-overlap golden (the red-team probe): a CropBox that
    /// HANGS OFF the MediaBox's edge ([300 300 800 900] on [0 0 612 792])
    /// crops to the INTERSECTION — the engine only exposes the
    /// intersection, so the entry is its extent (312×492) and its
    /// normalized corners as the crop provenance, never the declared
    /// box's own 500×600. The crop wording says "the declared CropBox
    /// reduced to the MediaBox (the effective crop)" for exactly this
    /// shape.
    #[test]
    fn a_partial_overlap_crop_is_the_intersection() {
        let geometry = page_geometry(
            geometry_pdf("/MediaBox [0 0 612 792] /CropBox [300 300 800 900]"),
            None,
        )
        .unwrap();
        assert_eq!(
            geometry[0],
            Some(PageBox {
                width_pt: 312.0,
                height_pt: 492.0,
                rotation: 0,
                crop: Some((300.0, 300.0, 612.0, 792.0)),
            })
        );
    }

    /// A non-right-angle /Rotate reports rotation 0 (the engine's
    /// normalization rule, T10) and the page's box stays the honest
    /// unrotated-extraction frame: nothing swaps, nothing maps.
    #[test]
    fn a_non_right_angle_rotate_reports_rotation_0_unrotated() {
        for declared in ["45", "-45", "135", "91"] {
            let geometry = page_geometry(
                geometry_pdf(&format!("{PORTRAIT} /Rotate {declared}")),
                None,
            )
            .unwrap();
            assert_eq!(
                geometry[0],
                Some(PageBox {
                    width_pt: 612.0,
                    height_pt: 792.0,
                    rotation: 0,
                    crop: None,
                }),
                "a /Rotate {declared} page must report rotation 0, unrotated frame"
            );
        }
    }

    /// The quarter-turn swap goldens: 90 and 270 swap the render extent
    /// (the turn is already applied), 180 does not — including the
    /// landscape-source golden (§9 item 3: 792×612 at /Rotate 90 →
    /// 612×792).
    #[test]
    fn quarter_turns_swap_the_render_extent() {
        let entry = |extras: &str| page_geometry(geometry_pdf(extras), None).unwrap()[0].unwrap();
        let rotated = entry(&format!("{PORTRAIT} /Rotate 90"));
        assert_eq!((rotated.width_pt, rotated.height_pt), (792.0, 612.0));
        assert_eq!(rotated.rotation, 90);
        let rotated = entry(&format!("{PORTRAIT} /Rotate 270"));
        assert_eq!((rotated.width_pt, rotated.height_pt), (792.0, 612.0));
        assert_eq!(rotated.rotation, 270);
        let rotated = entry(&format!("{PORTRAIT} /Rotate 180"));
        assert_eq!((rotated.width_pt, rotated.height_pt), (612.0, 792.0));
        assert_eq!(rotated.rotation, 180);
        // the landscape source: unrotated 792×612, /Rotate 90 renders 612×792
        let landscape = entry("/MediaBox [0 0 792 612] /Rotate 90");
        assert_eq!((landscape.width_pt, landscape.height_pt), (612.0, 792.0));
        assert_eq!(landscape.rotation, 90);
    }

    /// The render-anchored oracle, geometry side: the declared rotation's
    /// render extent is the extent the renderer draws at — pinned through
    /// pdf_oxide's own render box (page_render_box), not just through the
    /// box-reading surfaces that share its inputs. A 90° page renders
    /// 612×792-shaped output for a 612×792 source; the geometry says the
    /// same.
    #[test]
    fn the_render_extent_agrees_with_the_renderer_at_each_quarter_turn() {
        for rotation in [0, 90, 180, 270] {
            let geometry = page_geometry(
                geometry_pdf(&format!("{PORTRAIT} /Rotate {rotation}")),
                None,
            )
            .unwrap();
            let entry = geometry[0].unwrap();
            let (w, h) = (entry.width_pt, entry.height_pt);
            assert_eq!(
                (w, h),
                if rotation == 90 || rotation == 270 {
                    (792.0, 612.0)
                } else {
                    (612.0, 792.0)
                },
                "rotation {rotation}"
            );
        }
    }

    /// The composition law (§9 item 3, geometry side): the declared
    /// rotation composes with its inverse angle — r then (360 − r) ≡
    /// identity (the canonical inverse pairs 90+270 and 180+180 both sum
    /// to 360, which the engine normalizes to rotation 0) — so a page
    /// declared /Rotate 360 (90 then 270) and /Rotate 720 (180 twice)
    /// carry the identity geometry, extents equal within f32 tolerance.
    /// (The box-level forward-rule form of this property is the layout
    /// surface's pin, where the transform exists.)
    #[test]
    fn the_rotation_composes_with_its_inverse_to_the_identity() {
        let entry = |extras: &str| page_geometry(geometry_pdf(extras), None).unwrap()[0].unwrap();
        let identity = entry(PORTRAIT);
        // 90 + 270 and 180 + 180, each composed to the declared angle the
        // engine normalizes back to 0: the identity geometry.
        for composed in [360, 720] {
            let rotated = entry(&format!("{PORTRAIT} /Rotate {composed}"));
            assert_eq!(rotated.rotation, 0, "/Rotate {composed}");
            assert!((rotated.width_pt - identity.width_pt).abs() < f32::EPSILON * 8.0);
            assert!((rotated.height_pt - identity.height_pt).abs() < f32::EPSILON * 8.0);
        }
        // and per angle: the extent at r equals the extent at r+180 (the
        // half-turn is extent-neutral), so the swap composes: applying the
        // declared swap for r and then for (360 − r) returns the unswapped
        // extent — the identity, for every angle.
        for rotation in [0, 90, 180, 270] {
            let direct = entry(&format!("{PORTRAIT} /Rotate {rotation}"));
            let half_turn = entry(&format!("{PORTRAIT} /Rotate {}", (rotation + 180) % 360));
            assert_eq!(
                (direct.width_pt, direct.height_pt),
                (half_turn.width_pt, half_turn.height_pt)
            );
            let inverse = (360 - rotation) % 360;
            let swapped_back = entry(&format!("{PORTRAIT} /Rotate {inverse}"));
            let composed_swapped = match inverse {
                90 | 270 => (swapped_back.height_pt, swapped_back.width_pt),
                _ => (swapped_back.width_pt, swapped_back.height_pt),
            };
            assert!(
                (composed_swapped.0 - identity.width_pt).abs() < f32::EPSILON * 8.0
                    && (composed_swapped.1 - identity.height_pt).abs() < f32::EPSILON * 8.0,
                "r={rotation}: composing the swap with its inverse must return the identity extent"
            );
        }
    }

    /// The degenerate-crop pins (T6): a CropBox that misses the medium
    /// entirely, carries no area, equals the media box, or contains it
    /// does not actually crop — the visible box falls back to the MediaBox
    /// silently (the engine's rule), so the entry is `Some` with the FULL
    /// media extent and `crop = None`. A degenerate CropBox is NOT a None
    /// entry.
    #[test]
    fn a_degenerate_crop_is_not_a_none_entry_and_crops_nothing() {
        let entry = |extras: &str| page_geometry(geometry_pdf(extras), None).unwrap()[0].unwrap();
        // out-of-medium: the intersection is empty
        let out_of_medium = entry("/MediaBox [0 0 612 792] /CropBox [700 900 800 1000]");
        assert_eq!(
            (
                out_of_medium.width_pt,
                out_of_medium.height_pt,
                out_of_medium.rotation
            ),
            (612.0, 792.0, 0)
        );
        assert_eq!(out_of_medium.crop, None);
        // zero-area after normalization
        let zero_area = entry("/MediaBox [0 0 612 792] /CropBox [100 100 100 700]");
        assert_eq!((zero_area.width_pt, zero_area.height_pt), (612.0, 792.0));
        assert_eq!(zero_area.crop, None);
        // equal to the media box: declared, but nothing cropped
        let equal = entry("/MediaBox [0 0 612 792] /CropBox [0 0 612 792]");
        assert_eq!(equal.crop, None);
        // containing the media box: the intersection IS the media box
        let superset = entry("/MediaBox [0 0 612 792] /CropBox [-50 -50 662 842]");
        assert_eq!((superset.width_pt, superset.height_pt), (612.0, 792.0));
        assert_eq!(superset.crop, None);
        // the either-diagonal rule: [500 700 100 100] normalizes to the
        // subset crop, not to a degenerate one
        let other_diagonal = entry("/MediaBox [0 0 612 792] /CropBox [500 700 100 100]");
        assert_eq!(other_diagonal.crop, Some((100.0, 100.0, 500.0, 700.0)));
        assert_eq!(
            (other_diagonal.width_pt, other_diagonal.height_pt),
            (400.0, 600.0)
        );
    }

    /// The None-entry rule (§9 item 4): a page whose MediaBox is missing,
    /// empty, degenerate ([0 0 0 0]), or non-numeric cannot declare an
    /// extent — a None entry on an index page_count reported, with its
    /// siblings unaffected. The rule is by construction (the loop's
    /// per-page Result fold), never by error-message matching.
    #[test]
    fn undeclarable_pages_are_none_entries_and_siblings_are_unaffected() {
        // page 0: no MediaBox at all (the engine's read errors); page 1: valid.
        let bytes = objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
            b"<< /Type /Page /Parent 2 0 R >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
        ]);
        let geometry = page_geometry(bytes, None).unwrap();
        assert_eq!(geometry.len(), 2);
        assert_eq!(geometry[0], None, "the MediaBox-less page is a None entry");
        assert_eq!(
            geometry[1].unwrap().width_pt,
            612.0,
            "the sibling is intact"
        );
        // MediaBox [0 0 0 0]: zero extent → None entry (never Some(0, 0))
        let degenerate = page_geometry(geometry_pdf("/MediaBox [0 0 0 0]"), None).unwrap();
        assert_eq!(degenerate, vec![None]);
        // non-numeric corners read as the engine's 0.0 → zero extent → None
        let junk = page_geometry(geometry_pdf("/MediaBox [0 0 612 (junk)]"), None).unwrap();
        assert_eq!(junk, vec![None]);
    }

    /// Zero pages → `[]` (§9 item 5): the list length is the page count,
    /// and zero is a count.
    #[test]
    fn a_zero_page_document_yields_an_empty_list() {
        let bytes = objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [] /Count 0 >>",
        ]);
        assert_eq!(
            page_geometry(bytes, None).unwrap(),
            Vec::<Option<PageBox>>::new()
        );
    }

    /// The mid-iteration damage pins (§9 item 7's per-entry None rule,
    /// deepseek's probed-and-correct behaviors, now pinned): a page whose
    /// own box read fails on an index `page_count` reported is a None
    /// entry with its siblings intact — an indirect-ref MediaBox naming a
    /// missing object, a /Count over a shorter /Kids tree, and a
    /// mid-document truncation the open survives — while a truncation the
    /// OPEN cannot survive is the whole-call ValueError, never a list.
    #[test]
    fn mid_iteration_page_damage_is_none_entries_not_a_document_failure() {
        // Page 2 of 5 declares its MediaBox as an indirect ref to an
        // object that does not exist (9 0 R; the document ends at 7):
        // the box read fails for that page alone — [Some, None, Some, Some, Some].
        let objects: Vec<Vec<u8>> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R 6 0 R 7 0 R] /Count 5 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox 9 0 R >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>".to_vec(),
        ];
        let refs: Vec<&[u8]> = objects.iter().map(Vec::as_slice).collect();
        let geometry = page_geometry(objects_pdf(&refs), None).unwrap();
        assert_eq!(geometry.len(), 5);
        assert_eq!(geometry[0].unwrap().width_pt, 612.0);
        assert_eq!(
            geometry[1], None,
            "the missing-object MediaBox is a None entry"
        );
        for entry in geometry.iter().skip(2) {
            assert_eq!(entry.unwrap().width_pt, 612.0, "the siblings are intact");
        }

        // /Count 5 over three kids (and no further objects): the count
        // probe reports 5 (the declared count), pages 3-5 have no page
        // objects to walk — [Some, Some, Some, None, None].
        let geometry = page_geometry(
            objects_pdf(&[
                b"<< /Type /Catalog /Pages 2 0 R >>",
                b"<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 5 >>",
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
            ]),
            None,
        )
        .unwrap();
        assert_eq!(
            geometry.len(),
            5,
            "the length law reports the declared count"
        );
        assert_eq!(
            geometry,
            vec![
                Some(PageBox {
                    width_pt: 612.0,
                    height_pt: 792.0,
                    rotation: 0,
                    crop: None
                }),
                Some(PageBox {
                    width_pt: 612.0,
                    height_pt: 792.0,
                    rotation: 0,
                    crop: None
                }),
                Some(PageBox {
                    width_pt: 612.0,
                    height_pt: 792.0,
                    rotation: 0,
                    crop: None
                }),
                None,
                None,
            ]
        );

        // A 45% truncation: the open survives (the header and page 1's
        // objects are inside the kept prefix and the engine reconstructs
        // the rest), page 2's are gone — [Some, None].
        let full = two_page_pdf("one", "two");
        let truncated = full[..full.len() * 45 / 100].to_vec();
        let geometry = page_geometry(truncated, None).unwrap();
        assert_eq!(
            geometry,
            vec![
                Some(PageBox {
                    width_pt: 612.0,
                    height_pt: 792.0,
                    rotation: 0,
                    crop: None
                }),
                None,
            ]
        );

        // A 5% truncation: nothing left to reconstruct the document from
        // — the open fails and the WHOLE call raises (the typed
        // InvalidPdf shape), never a partial list.
        let truncated = full[..full.len() * 5 / 100].to_vec();
        let err = page_geometry(truncated, None).unwrap_err();
        assert!(matches!(err, PdfError::InvalidPdf(_)), "{err}");
    }

    // --- the encrypted door on the geometry probe (§9 item 6) --------------
    //
    // A real R2/V1 (RC4-40) Standard-security document, built here: the
    // fixture bytes are the pin (pypdf salts per call, so no external
    // encrypted fixture could be one), and the crypto is the spec's own
    // Algorithms 2-4 at 40 bits: MD5 key derivation (md-5, already in the
    // lock as pdf_oxide's legacy-crypto dep) plus a hand-rolled RC4.

    /// The Standard security handler's 32-byte padding string.
    const ENC_PAD: [u8; 32] = [
        0x28, 0xBF, 0x4E, 0x5E, 0x4E, 0x75, 0x8A, 0x41, 0x64, 0x00, 0x4E, 0x56, 0xFF, 0xFA, 0x01,
        0x08, 0x2E, 0x2E, 0x00, 0xB6, 0xD0, 0x68, 0x3E, 0x80, 0x2F, 0x0C, 0xA9, 0xFE, 0x64, 0x53,
        0x69, 0x7A,
    ];

    fn rc4(key: &[u8], data: &[u8]) -> Vec<u8> {
        let mut state: [u8; 256] = core::array::from_fn(|i| i as u8);
        let mut j = 0usize;
        for i in 0..256 {
            j = (j + state[i] as usize + key[i % key.len()] as usize) & 0xFF;
            state.swap(i, j);
        }
        let (mut i, mut j) = (0usize, 0usize);
        data.iter()
            .map(|&byte| {
                i = (i + 1) & 0xFF;
                j = (j + state[i] as usize) & 0xFF;
                state.swap(i, j);
                byte ^ state[(state[i] as usize + state[j] as usize) & 0xFF]
            })
            .collect()
    }

    /// Pad-or-truncate a password to the handler's 32 bytes.
    fn padded_password(password: &[u8]) -> Vec<u8> {
        let mut padded = password[..password.len().min(32)].to_vec();
        padded.extend_from_slice(&ENC_PAD[..32 - padded.len()]);
        padded
    }

    /// Algorithm 3 (R=2, V=1): the /O value — the padding RC4'd under
    /// MD5(padded owner password) truncated to the 5-byte key.
    fn r2_owner_value(owner_pw: &[u8]) -> Vec<u8> {
        use md5::{Digest, Md5};
        let mut hasher = Md5::new();
        hasher.update(padded_password(owner_pw));
        rc4(&hasher.finalize()[..5], &ENC_PAD)
    }

    /// Algorithm 2 (R=2, V=1): the file key — MD5(padded user pw || O ||
    /// P le32 || ID[0]) truncated to 5 bytes.
    fn r2_file_key(user_pw: &[u8], owner_value: &[u8], permissions: i32, id0: &[u8]) -> Vec<u8> {
        use md5::{Digest, Md5};
        let mut hasher = Md5::new();
        hasher.update(padded_password(user_pw));
        hasher.update(owner_value);
        hasher.update(permissions.to_le_bytes());
        hasher.update(id0);
        hasher.finalize()[..5].to_vec()
    }

    /// Algorithm 4 (R=2): the /U value — the padding RC4'd under the file
    /// key (what `authenticate` recomputes and compares).
    fn r2_user_value(user_pw: &[u8], owner_value: &[u8], permissions: i32, id0: &[u8]) -> Vec<u8> {
        rc4(
            &r2_file_key(user_pw, owner_value, permissions, id0),
            &ENC_PAD,
        )
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02X}")).collect()
    }

    /// The hand-rolled RC4 against the standard known-answer vectors (the
    /// two the RC4 literature pins): the encrypted fixtures above are only
    /// as trustworthy as this primitive, so it verifies itself before any
    /// of them leans on it.
    #[test]
    fn rc4_matches_the_standard_known_answer_vectors() {
        assert_eq!(hex(&rc4(b"Key", b"Plaintext")), "BBF316E8D940AF0AD3");
        assert_eq!(hex(&rc4(b"Wiki", b"pedia")), "1021BF0420");
    }

    /// A one-page, R2/V1-encrypted PDF with the given user/owner
    /// passwords (no content stream: the geometry probe needs the page
    /// tree only, and nothing in it is a string or stream that would
    /// itself need encrypting). Fixed /ID, computed /O and /U.
    fn encrypted_pdf(user_pw: &str, owner_pw: &str) -> Vec<u8> {
        const ID0: [u8; 16] = [0x71; 16];
        let owner_value = r2_owner_value(owner_pw.as_bytes());
        let user_value = r2_user_value(user_pw.as_bytes(), &owner_value, -4, &ID0);
        let encrypt_dict = format!(
            "<< /Filter /Standard /V 1 /R 2 /O <{}> /U <{}> /P -4 >>",
            hex(&owner_value),
            hex(&user_value)
        );
        let objects: Vec<Vec<u8>> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>".to_vec(),
            encrypt_dict.into_bytes(),
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
                "trailer\n<< /Size {} /Root 1 0 R /Encrypt 4 0 R /ID [<{}> <{}>] >>\n\
                 startxref\n{xref_at}\n%%EOF",
                objects.len() + 1,
                hex(&ID0),
                hex(&ID0)
            )
            .as_bytes(),
        );
        out
    }

    /// An encrypted PDF fails closed on the geometry probe (§9 item 6):
    /// the door check raises before any content walk, the same
    /// ValueError-shaped refusal the other probes raise. A wrong password
    /// is its own error, and the right one unlocks.
    #[test]
    fn encrypted_documents_fail_closed_on_the_geometry_probe() {
        let locked = encrypted_pdf("torque", "owner-pw");
        let err = page_geometry(locked.clone(), None).unwrap_err();
        assert!(matches!(err, PdfError::EncryptedPdf), "{err}");
        let err = page_geometry(locked.clone(), Some("not-the-password")).unwrap_err();
        assert!(err.to_string().contains("did not unlock"), "{err}");
        // the right password unlocks: the page's geometry comes through
        let geometry = page_geometry(locked, Some("torque")).unwrap();
        assert_eq!(geometry.len(), 1);
        assert_eq!(
            geometry[0].unwrap(),
            PageBox {
                width_pt: 612.0,
                height_pt: 792.0,
                rotation: 0,
                crop: None,
            }
        );
    }

    /// The layout probe's same door, pinned as the geometry probe's twin
    /// (§9 item 6): the door check raises before any content walk, a
    /// wrong password is its own error, and the right one unlocks into
    /// the entry (the geometry through, no content — the fixture carries
    /// the page tree only).
    #[test]
    fn encrypted_documents_fail_closed_on_the_layout_probe() {
        let locked = encrypted_pdf("torque", "owner-pw");
        let err = page_layout(locked.clone(), None, None, "pdf").unwrap_err();
        assert!(
            matches!(err, PagesError::Pdf(PdfError::EncryptedPdf)),
            "{err}"
        );
        let err = page_layout(locked.clone(), None, Some("not-the-password"), "pdf").unwrap_err();
        assert!(err.to_string().contains("did not unlock"), "{err}");
        // the right password unlocks: the page's geometry comes through
        let pages = page_layout(locked, None, Some("torque"), "pdf").unwrap();
        assert_eq!(pages.len(), 1);
        assert_eq!(
            pages[0].geometry,
            Some(PageBox {
                width_pt: 612.0,
                height_pt: 792.0,
                rotation: 0,
                crop: None,
            })
        );
        assert!(
            pages[0].spans.is_empty() && pages[0].lines.is_empty() && pages[0].images.is_empty()
        );
    }

    /// The owner-password-only shape reads normally with the empty
    /// password (the engine's open law, pinned on this probe): U is
    /// computed for the EMPTY user password (the file is owner-locked
    /// only), so the empty password authenticates at open and the
    /// geometry comes through — never masked as empty output.
    #[test]
    fn the_owner_password_only_document_reads_with_the_empty_password() {
        let locked = encrypted_pdf("", "owner-pw");
        let geometry = page_geometry(locked.clone(), None).unwrap();
        assert_eq!(geometry.len(), 1);
        assert_eq!(geometry[0].unwrap().height_pt, 792.0);
        // and the explicit empty password is the same call
        assert_eq!(page_geometry(locked, Some("")).unwrap(), geometry);
    }

    // --- the page_tables probe (issue #204; the A1 gate, the A6 cap, the
    // --- A9 refused marker; evidence: /tmp/opencode/probe-tables out.txt
    // --- and out2-out7.txt, the preserved measurement oracle) --------------

    /// The ruled 2×2's content stream (the probe's fixture verbatim):
    /// three verticals x ∈ {100, 300, 500} spanning y 600..800, three
    /// horizontals y ∈ {600, 700, 800} spanning x 100..500, and one glyph
    /// per cell (A/B/C/D at (120,750)/(320,750)/(120,650)/(320,650)).
    const RULED_2X2: &str = "0.5 w 100 600 m 100 800 l S\n\
         0.5 w 300 600 m 300 800 l S\n\
         0.5 w 500 600 m 500 800 l S\n\
         0.5 w 100 600 m 500 600 l S\n\
         0.5 w 100 700 m 500 700 l S\n\
         0.5 w 100 800 m 500 800 l S\n\
         BT /F1 12 Tf 120 750 Td (A) Tj ET\n\
         BT /F1 12 Tf 320 750 Td (B) Tj ET\n\
         BT /F1 12 Tf 120 650 Td (C) Tj ET\n\
         BT /F1 12 Tf 320 650 Td (D) Tj ET";

    /// The text-only 4×2's content stream (the probe's fixture verbatim):
    /// two x-columns (100/300), four baselines (700/650/600/550), no
    /// rulings anywhere.
    const TEXT_ONLY_4X2: &str = "BT /F1 12 Tf 100 700 Td (Region) Tj ET\n\
         BT /F1 12 Tf 300 700 Td (Value) Tj ET\n\
         BT /F1 12 Tf 100 650 Td (North) Tj ET\n\
         BT /F1 12 Tf 300 650 Td (120) Tj ET\n\
         BT /F1 12 Tf 100 600 Td (South) Tj ET\n\
         BT /F1 12 Tf 300 600 Td (340) Tj ET\n\
         BT /F1 12 Tf 100 550 Td (East) Tj ET\n\
         BT /F1 12 Tf 300 550 Td (560) Tj ET";

    /// The 180-symmetric ruled 2×2 (the probe's sym_grid verbatim): the
    /// grid centered on (306, 396) is invariant under (x, y) → (612−x,
    /// 792−y), so the /Rotate 180 detector DOES find a grid — with every
    /// cell's text 180°-swapped (measured out2). The gate's regression
    /// fixture: the refusal is not the vanished-table accident.
    const SYM_GRID_2X2: &str = "0.5 w 206 346 m 206 446 l S\n\
         0.5 w 306 346 m 306 446 l S\n\
         0.5 w 406 346 m 406 446 l S\n\
         0.5 w 206 346 m 406 346 l S\n\
         0.5 w 206 396 m 406 396 l S\n\
         0.5 w 206 446 m 406 446 l S\n\
         BT /F1 12 Tf 220 420 Td (A) Tj ET\n\
         BT /F1 12 Tf 320 420 Td (B) Tj ET\n\
         BT /F1 12 Tf 220 370 Td (C) Tj ET\n\
         BT /F1 12 Tf 320 370 Td (D) Tj ET";

    /// Sideways (rotated-Tm) text inside a ruled 2×2 (the probe's
    /// sideways_ruled verbatim): the 90/270-with-rulings-and-mapped-spans
    /// class — text and rules share no frame.
    const SIDEWAYS_RULED: &str = "0.5 w 100 600 m 100 800 l S\n\
         0.5 w 300 600 m 300 800 l S\n\
         0.5 w 500 600 m 500 800 l S\n\
         0.5 w 100 600 m 500 600 l S\n\
         0.5 w 100 700 m 500 700 l S\n\
         0.5 w 100 800 m 500 800 l S\n\
         BT /F1 12 Tf 0 1 -1 0 130 610 Tm (A) Tj ET\n\
         BT /F1 12 Tf 0 1 -1 0 330 610 Tm (B) Tj ET\n\
         BT /F1 12 Tf 0 1 -1 0 130 710 Tm (C) Tj ET\n\
         BT /F1 12 Tf 0 1 -1 0 330 710 Tm (D) Tj ET";

    /// Sideways text-only 4×2 (the A12 served class at 90/270): every
    /// span drawn with `Tm [0 1 -1 0 x y]` — engine-mapped on a
    /// 90/270 page (`page_rotation_applied == rotation`), no rulings.
    const SIDEWAYS_TEXT_ONLY: &str = "BT /F1 12 Tf 0 1 -1 0 100 700 Tm (Region) Tj ET\n\
         BT /F1 12 Tf 0 1 -1 0 300 700 Tm (Value) Tj ET\n\
         BT /F1 12 Tf 0 1 -1 0 100 650 Tm (North) Tj ET\n\
         BT /F1 12 Tf 0 1 -1 0 300 650 Tm (120) Tj ET\n\
         BT /F1 12 Tf 0 1 -1 0 100 600 Tm (South) Tj ET\n\
         BT /F1 12 Tf 0 1 -1 0 300 600 Tm (340) Tj ET\n\
         BT /F1 12 Tf 0 1 -1 0 100 550 Tm (East) Tj ET\n\
         BT /F1 12 Tf 0 1 -1 0 300 550 Tm (560) Tj ET";

    /// Two ruled grids stacked with disjoint x-ranges and a 20 pt gap (the
    /// probe's stacked_disjoint verbatim): the vertical MERGE path's
    /// fixture (A11 — the disjoint-x spelling).
    const STACKED_DISJOINT: &str = "0.5 w 100 600 m 100 800 l S\n\
         0.5 w 200 600 m 200 800 l S\n\
         0.5 w 300 600 m 300 800 l S\n\
         0.5 w 100 600 m 300 600 l S\n\
         0.5 w 100 700 m 300 700 l S\n\
         0.5 w 100 800 m 300 800 l S\n\
         BT /F1 12 Tf 120 750 Td (T1) Tj ET\n\
         BT /F1 12 Tf 220 750 Td (T2) Tj ET\n\
         BT /F1 12 Tf 120 650 Td (T3) Tj ET\n\
         BT /F1 12 Tf 220 650 Td (T4) Tj ET\n\
         0.5 w 350 380 m 350 580 l S\n\
         0.5 w 450 380 m 450 580 l S\n\
         0.5 w 550 380 m 550 580 l S\n\
         0.5 w 350 380 m 550 380 l S\n\
         0.5 w 350 480 m 550 480 l S\n\
         0.5 w 350 580 m 550 580 l S\n\
         BT /F1 12 Tf 370 530 Td (B1) Tj ET\n\
         BT /F1 12 Tf 470 530 Td (B2) Tj ET\n\
         BT /F1 12 Tf 370 430 Td (B3) Tj ET\n\
         BT /F1 12 Tf 470 430 Td (B4) Tj ET";

    /// Two ruled grids stacked with SHARED x and a 20 pt gap, the bottom
    /// one 3-column (the probe's stacked_grids verbatim): the CLUSTER
    /// path's fixture (A11 — shared-x routes to the cluster; A13's
    /// line-extent bbox and second has_header source; A2's None-pad row).
    const STACKED_SHARED_X: &str = "0.5 w 100 600 m 100 800 l S\n\
         0.5 w 300 600 m 300 800 l S\n\
         0.5 w 500 600 m 500 800 l S\n\
         0.5 w 100 600 m 500 600 l S\n\
         0.5 w 100 800 m 500 800 l S\n\
         BT /F1 12 Tf 120 700 Td (T1) Tj ET\n\
         BT /F1 12 Tf 320 700 Td (T2) Tj ET\n\
         0.5 w 100 380 m 100 580 l S\n\
         0.5 w 233 380 m 233 580 l S\n\
         0.5 w 366 380 m 366 580 l S\n\
         0.5 w 500 380 m 500 580 l S\n\
         0.5 w 100 380 m 500 380 l S\n\
         0.5 w 100 580 m 500 580 l S\n\
         BT /F1 12 Tf 120 480 Td (B1) Tj ET\n\
         BT /F1 12 Tf 250 480 Td (B2) Tj ET\n\
         BT /F1 12 Tf 380 480 Td (B3) Tj ET";

    /// The page_tables fixture builder: one page, the given content
    /// stream, `page_extras` (the /Rotate//CropBox fragments) — the
    /// layout fixtures' exact base (`layout_pdf`).
    fn tables_pdf(content: &str, page_extras: &str) -> Vec<u8> {
        layout_pdf(content, page_extras)
    }

    /// The single-page tables entry (most fixtures are one page).
    fn one_table_page(bytes: Vec<u8>) -> PageTablesData {
        page_tables(bytes, None, None, "pdf").unwrap().remove(0)
    }

    /// The first table's first-row cell texts (the swapped-text pins).
    fn row_texts(page: &PageTablesData, row: usize) -> Vec<&str> {
        page.tables[0].rows[row]
            .cells
            .iter()
            .map(|c| c.text.as_str())
            .collect()
    }

    /// Whether the rendered page has ink within ±1 px of the render-space
    /// point (the ruling-position oracle's pixel probe; 72 dpi, 1pt = 1px,
    /// the raster's top-left origin flipped to render space first). The
    /// channel test takes the stroke's anti-aliasing into account: a 0.5pt
    /// ruling renders ~25%-coverage gray (~191 on white — measured), so
    /// `ruling` ink is anything clearly off the white paper, while the
    /// solid text/fill ink the sibling oracles use stays `< 128`.
    fn ink_near(image: &pdf_oxide::rendering::RenderedImage, x: f64, y: f64) -> bool {
        let ruling = |r: u8, _g: u8, _b: u8| r < 220;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let px = x as i64 + dx as i64;
                let py = image.height as i64 - 1 - (y as i64) + dy as i64;
                if px < 0 || py < 0 || px >= image.width as i64 || py >= image.height as i64 {
                    continue;
                }
                let i = ((py * image.width as i64 + px) * 4) as usize;
                if ruling(image.data[i], image.data[i + 1], image.data[i + 2]) {
                    return true;
                }
            }
        }
        false
    }

    fn rendered(doc: &PdfDocument) -> pdf_oxide::rendering::RenderedImage {
        pdf_oxide::rendering::render_page(
            doc,
            0,
            &pdf_oxide::rendering::RenderOptions::with_dpi(72).as_raw(),
        )
        .unwrap()
    }

    /// The rot-0 intersection-grid golden, RENDER-ANCHORED (Q4's oracle:
    /// the cell edges are checked against the RENDERED RULING positions,
    /// not the fixture's own coordinates): the engine's cell boxes are the
    /// ruling rectangles — A(100,700,200,100) B(300,700,200,100)
    /// C(100,600,200,100) D(300,600,200,100) measured (out.txt) — and
    /// every cell edge lands on ink the renderer drew. `has_header` is
    /// FALSE (the intersection branch never sets it — the pinned engine
    /// limitation), `col_count == 2`, the table bbox the ruled rectangle
    /// (100,600,400,200), and `tables_refused` false.
    #[test]
    fn tables_ruled_2x2_rot0_cells_match_the_rendered_rulings() {
        let bytes = tables_pdf(RULED_2X2, "");
        let doc = PdfDocument::from_bytes(bytes.clone()).unwrap();
        let image = rendered(&doc);
        // the rendered ruling positions: the ON-PAGE points of the grid
        // carry ink (the fixture's y=800 top rule rides above the 792pt
        // render — off-raster, the unclipped law's own evidence below);
        // a point just off the grid does not.
        for x in [100.0, 300.0, 500.0] {
            for y in [600.0, 700.0, 750.0] {
                assert!(ink_near(&image, x, y), "no rendered ruling at ({x}, {y})");
            }
        }
        for x in [150.0, 400.0] {
            assert!(
                ink_near(&image, x, 600.0),
                "no rendered horizontal at ({x}, 600)"
            );
            assert!(
                ink_near(&image, x, 700.0),
                "no rendered horizontal at ({x}, 700)"
            );
        }
        assert!(
            !ink_near(&image, 90.0, 590.0),
            "the oracle's negative control: nothing at (90, 590)"
        );

        let page = one_table_page(bytes);
        assert!(!page.tables_refused);
        assert_eq!(page.page_index, 0);
        assert_eq!(page.format, "pdf");
        let geometry = page.geometry.unwrap();
        assert_eq!(
            (geometry.width_pt, geometry.height_pt, geometry.rotation),
            (612.0, 792.0, 0)
        );
        assert_eq!(page.tables.len(), 1);
        let table = &page.tables[0];
        assert_eq!(
            (table.col_count, table.has_header),
            (2, false),
            "the pinned engine limitation"
        );
        assert_eq!(table.bbox, Some((100.0, 600.0, 400.0, 200.0)));
        assert_eq!(table.rows.len(), 2);
        assert!(!table.rows[0].is_header && !table.rows[1].is_header);
        let expected = [
            ("A", (100.0, 700.0, 200.0, 100.0)),
            ("B", (300.0, 700.0, 200.0, 100.0)),
            ("C", (100.0, 600.0, 200.0, 100.0)),
            ("D", (300.0, 600.0, 200.0, 100.0)),
        ];
        /// A cell's text and its measured box (the golden pair's type).
        type CellGolden<'a> = (&'a str, (f32, f32, f32, f32));
        let got: Vec<CellGolden> = table
            .rows
            .iter()
            .flat_map(|r| &r.cells)
            .map(|c| (c.text.as_str(), c.bbox.unwrap()))
            .collect();
        assert_eq!(got.len(), 4);
        for ((text, bbox), (want_text, want_bbox)) in got.iter().zip(expected.iter()) {
            assert_eq!(text, want_text);
            assert_box_near(*bbox, *want_bbox, 0.01, &format!("cell {want_text}"));
            // the Q4 oracle: every ON-PAGE cell edge is a rendered ruling
            // (the intersection branch's boxes ARE ruling geometry) —
            // sample each edge's midpoint. The top row's top edge (y=800)
            // is off the 792pt page: the UNCLIPPED law's own evidence
            // (ruling geometry is not crop-clipped, and here not even
            // page-clipped in the answer), so it is pinned off-render.
            let (x, y, w, h) = *bbox;
            let mid_x = f64::from(x + w / 2.0);
            let mid_y = f64::from(y + h / 2.0);
            let edge = |px: f64, py: f64, name: &str| {
                if (0.0..=612.0).contains(&px) && (0.0..=792.0).contains(&py) {
                    assert!(
                        ink_near(&image, px, py),
                        "cell {text}: {name} edge off ruling"
                    );
                } else {
                    assert!(
                        px < 0.0 || px > 612.0 || py < 0.0 || py > 792.0,
                        "cell {text}: {name} edge inside the render but unrendered"
                    );
                }
            };
            edge(f64::from(x), mid_y, "left");
            edge(f64::from(x + w), mid_y, "right");
            edge(mid_x, f64::from(y), "bottom");
            edge(mid_x, f64::from(y + h), "top");
        }
        // the top row's boxes ride 100pt past the render page's top edge
        // (measured fixture geometry: rules at y=800 on a 792pt page) —
        // returned unclipped, the cell-boxes-are-geometry law
        assert_eq!(table.rows[0].cells[0].bbox.unwrap().3 + 700.0, 800.0);
    }

    /// The A1 refusal classes, pinned (Q4's oracle change: /Rotate 180
    /// fixtures pin `tables: []`, NOT mapped boxes):
    ///
    /// - a ruled 2×2 on /Rotate 180 — the engine detects NOTHING (measured
    ///   out.txt: text mapped to y≈30/130, rules at y=600..800) — and the
    ///   surface refuses it as frame-incoherent (`tables_refused: true`,
    ///   geometry intact with rotation 180, A9);
    /// - the 180-SYMMETRIC ruled grid on /Rotate 180 — the engine DOES
    ///   detect it and every cell's text comes back 180°-swapped (measured
    ///   out2: r0c0 = "D" where the raw top-left cell held "A") — the
    ///   qualitative-failure regression that proves the gate is a content
    ///   law, not the vanished-table accident; refused the same way;
    /// - the same symmetric grid at rotation 0 SERVES with the upright
    ///   cells (the control: the fixture is not the problem);
    /// - sideways (rotated-Tm) text under rulings at 90 and 270 — the
    ///   measured misassignment class (out2: tables: 0) — refused.
    #[test]
    fn tables_the_frame_incoherent_classes_are_refused() {
        // ruled 2x2 at 180: the vanished table
        let page = one_table_page(tables_pdf(RULED_2X2, "/Rotate 180"));
        assert!(page.tables.is_empty());
        assert!(
            page.tables_refused,
            "180 + rulings: the frame-incoherent refusal"
        );
        assert_eq!(page.geometry.unwrap().rotation, 180);

        // the 180-symmetric grid at 180: detected-but-swapped upstream,
        // refused here — the regression that proves the gate
        let page = one_table_page(tables_pdf(SYM_GRID_2X2, "/Rotate 180"));
        assert!(page.tables.is_empty());
        assert!(page.tables_refused);
        // ...and the rot-0 control: the same grid serves upright
        let page = one_table_page(tables_pdf(SYM_GRID_2X2, ""));
        assert!(!page.tables_refused);
        assert_eq!(page.tables.len(), 1);
        assert_eq!(row_texts(&page, 0), vec!["A", "B"]);
        assert_eq!(row_texts(&page, 1), vec!["C", "D"]);

        // sideways text under rulings at 90 and 270
        for rotation in ["/Rotate 90", "/Rotate 270"] {
            let page = one_table_page(tables_pdf(SIDEWAYS_RULED, rotation));
            assert!(page.tables.is_empty(), "{rotation}");
            assert!(page.tables_refused, "{rotation}");
        }
    }

    /// The 90/270 coherent class with rulings (A1's row 5): upright text,
    /// raw frame throughout — the engine's ruling-rect cells forward-map
    /// exactly. Goldens (derived from the measured raw boxes, out.txt's
    /// rot-0 cells through the forward rule): cell A raw (100,700,200,100)
    /// at 90 → (700, 612−100−200, 100, 200) = (700, 312, 100, 200).
    #[test]
    fn tables_the_90_270_upright_ruled_class_forward_maps_exactly() {
        let page = one_table_page(tables_pdf(RULED_2X2, "/Rotate 90"));
        assert!(!page.tables_refused);
        assert_eq!(page.tables.len(), 1);
        assert_eq!(page.geometry.unwrap().rotation, 90);
        let table = &page.tables[0];
        assert_eq!(table.bbox, Some((600.0, 112.0, 200.0, 400.0)));
        let a = &table.rows[0].cells[0];
        assert_eq!(a.text, "A");
        assert_box_near(
            a.bbox.unwrap(),
            (700.0, 312.0, 100.0, 200.0),
            0.01,
            "cell A at 90",
        );

        // 270: the other swap arm — A → (792−700−100, 100, 100, 200)
        let page = one_table_page(tables_pdf(RULED_2X2, "/Rotate 270"));
        assert!(!page.tables_refused);
        let table = &page.tables[0];
        assert_box_near(
            table.rows[0].cells[0].bbox.unwrap(),
            (-8.0, 100.0, 100.0, 200.0),
            0.01,
            "cell A at 270",
        );
    }

    /// The 90/270 coherent class without rulings (A1's row 2): upright
    /// text-only tables serve with the forward rule over the span-union
    /// cells. Measured raw boxes (out.txt rot 0) forward-mapped: at 90,
    /// "Region" (100,700,38.02,12) → (700, 473.98, 12, 38.02); the table
    /// bbox (100,550,230.684,162) → (550, 281.32, 162, 230.684).
    #[test]
    fn tables_the_90_270_upright_text_only_class_forward_maps_exactly() {
        for rotation in [90, 270] {
            let page = one_table_page(tables_pdf(TEXT_ONLY_4X2, &format!("/Rotate {rotation}")));
            assert!(!page.tables_refused, "{rotation}");
            assert_eq!(page.tables.len(), 1, "{rotation}: the measured detection");
            assert_eq!(page.geometry.unwrap().rotation, rotation);
            let table = &page.tables[0];
            assert_eq!((table.col_count, table.has_header), (2, false));
            let region = &table.rows[0].cells[0];
            assert_eq!(region.text, "Region");
            if rotation == 90 {
                assert_box_near(
                    region.bbox.unwrap(),
                    (700.0, 612.0 - 100.0 - 38.016, 12.0, 38.016),
                    0.01,
                    "Region at 90",
                );
                assert_box_near(
                    table.bbox.unwrap(),
                    (550.0, 612.0 - 100.0 - 230.684, 162.0, 230.684),
                    0.01,
                    "table bbox at 90",
                );
            } else {
                assert_box_near(
                    region.bbox.unwrap(),
                    (792.0 - 700.0 - 12.0, 100.0, 12.0, 38.016),
                    0.01,
                    "Region at 270",
                );
                assert_box_near(
                    table.bbox.unwrap(),
                    (792.0 - 550.0 - 162.0, 100.0, 162.0, 230.684),
                    0.01,
                    "table bbox at 270",
                );
            }
        }
    }

    /// The A12 served class: a PATHLESS page whose every span is
    /// engine-mapped — the sideways text-only table at 270 (measured
    /// out3: detected, 4×4 with None-padded cells) serves through
    /// unmap→forward, which is the identity here (no crop: the frames
    /// coincide), so the boxes are the engine's mapped numbers verbatim.
    /// The 90 arm of the same fixture is coherent too (all mapped) and
    /// the detector finds NOTHING there (measured out3) — served empty,
    /// `tables_refused` false (the detector's sideways limit at 90, not a
    /// refusal; the asymmetry is the engine's).
    #[test]
    fn tables_the_sideways_text_only_class_serves_via_unmap_then_forward() {
        // 270: served, the mapped numbers round-trip exactly
        let page = one_table_page(tables_pdf(SIDEWAYS_TEXT_ONLY, "/Rotate 270"));
        assert!(!page.tables_refused);
        assert_eq!(page.geometry.unwrap().rotation, 270);
        assert_eq!(page.tables.len(), 1);
        let table = &page.tables[0];
        assert_eq!((table.col_count, table.has_header), (4, false));
        assert_box_near(
            table.bbox.unwrap(),
            (80.0, 100.0, 174.012, 230.684),
            0.01,
            "T0 bbox",
        );
        // the measured grid (out3): r0 [Value, pad, pad, pad]; the pads
        // are the cluster branch's None cells, INSIDE the table
        let r0 = &table.rows[0];
        assert_eq!(r0.cells[0].text, "Value");
        assert_box_near(
            r0.cells[0].bbox.unwrap(),
            (80.0, 300.0, 30.684, 30.684),
            0.01,
            "Value",
        );
        for pad in &r0.cells[1..] {
            assert_eq!(pad.text, "");
            assert_eq!(
                pad.bbox, None,
                "a pad cell's None bbox is the engine's truth"
            );
        }
        // r1's filled cells (the measured boxes, mapped verbatim)
        let r1 = &table.rows[1];
        assert_eq!(r1.cells[0].text, "");
        assert_eq!(r1.cells[0].bbox, None);
        assert_eq!(r1.cells[1].text, "120");
        assert_box_near(
            r1.cells[1].bbox.unwrap(),
            (130.0, 300.0, 20.016, 20.016),
            0.01,
            "120",
        );
        assert_eq!(r1.cells[3].text, "560");
        assert_box_near(
            r1.cells[3].bbox.unwrap(),
            (230.0, 300.0, 20.016, 20.016),
            0.01,
            "560",
        );

        // 90: coherent, served, the detector declines — empty, not refused
        let page = one_table_page(tables_pdf(SIDEWAYS_TEXT_ONLY, "/Rotate 90"));
        assert!(!page.tables_refused);
        assert!(
            page.tables.is_empty(),
            "measured out3: the 90 detector limit"
        );
    }

    /// The A12 mixed-span refusal classes: a pathless page whose spans MIX
    /// engine-mapped (sideways) and raw (upright) runs — the detector
    /// would compare two text frames; refused (fail-closed: the unlisted
    /// combination), geometry intact, at both 90 and 270.
    #[test]
    fn tables_the_mixed_span_classes_are_refused() {
        let mixed = "BT /F1 12 Tf 100 700 Td (up) Tj ET\n\
             BT /F1 12 Tf 0 1 -1 0 300 650 Tm (side) Tj ET\n\
             BT /F1 12 Tf 100 600 Td (raw) Tj ET\n\
             BT /F1 12 Tf 0 1 -1 0 300 550 Tm (flip) Tj ET";
        for rotation in [90, 270] {
            let page = one_table_page(tables_pdf(mixed, &format!("/Rotate {rotation}")));
            assert!(page.tables.is_empty(), "{rotation}");
            assert!(page.tables_refused, "{rotation}: the mixed-span refusal");
            assert_eq!(page.geometry.unwrap().rotation, rotation);
        }
    }

    /// The crop laws on this surface (the unclipped law + the text-side
    /// asymmetry, both measured out2): under the [110 550 550 800] crop
    /// the ruled grid survives with cells whose boxes lie OUTSIDE the
    /// visible box — cell A raw (100,700,200,100) → (100−110, 700−550,
    /// 200, 100) = (−10, 150, 200, 100), a negative x on a 440-wide page —
    /// returned unclipped, the consumer clips. And under [150 600 550
    /// 800] the crop drops the LEFT column's text (the TEXT side IS
    /// crop-filtered — the 2pt law) and the table VANISHES: `tables: []`
    /// with `tables_refused` FALSE (no tables found ≠ a refusal).
    #[test]
    fn tables_the_crop_laws_out_of_box_cells_and_the_text_side_filter() {
        // the survivor: an out-of-box cell, returned unclipped
        let page = one_table_page(tables_pdf(RULED_2X2, "/CropBox [110 550 550 800]"));
        assert!(!page.tables_refused);
        assert_eq!(
            page.tables.len(),
            1,
            "measured out2: the grid survives this crop"
        );
        let geometry = page.geometry.unwrap();
        // the crop's intersection with the MediaBox: x 110..550, y 550..792
        assert_eq!((geometry.width_pt, geometry.height_pt), (440.0, 242.0));
        let a = &page.tables[0].rows[0].cells[0];
        assert_eq!(a.text, "A");
        assert_box_near(
            a.bbox.unwrap(),
            (-10.0, 150.0, 200.0, 100.0),
            0.01,
            "the unclipped cell",
        );
        assert!(
            a.bbox.unwrap().0 < 0.0,
            "the box lies outside the page box: the consumer clips"
        );

        // the vanish variant: the text side is crop-filtered, the table dies
        let page = one_table_page(tables_pdf(RULED_2X2, "/CropBox [150 600 550 800]"));
        assert!(
            page.tables.is_empty(),
            "measured out2: the crop drops the left column's text"
        );
        assert!(
            !page.tables_refused,
            "a vanished table is the engine's answer, not a refusal"
        );
        assert!(page.geometry.is_some(), "geometry intact");
    }

    /// No rules and no table-shaped text → `tables: []`, refused FALSE —
    /// the no-tables page, never an error (the detector-limit pins: these
    /// two filters are engine heuristics, so the FIXTURE BYTES are
    /// pinned, not the rule — measured out3): plain prose lines decline;
    /// a TOC with INLINE dot leaders (one span per entry: single-column)
    /// declines. (The measured boundary, stated in the docstring not
    /// pinned as a rule: a TWO-column TOC shape — title + page number
    /// spans — IS detected as a table by the same engine, out3; the
    /// heuristic's answer is the fixture's, and these bytes' answer is
    /// [].)
    #[test]
    fn tables_prose_and_toc_pages_serve_empty_not_refused() {
        // the two-page prose fixture: no tables on either page
        let bytes = two_page_pdf("Alpha page text", "Beta page text");
        for (page_index, entry) in page_tables(bytes, None, None, "pdf")
            .unwrap()
            .iter()
            .enumerate()
        {
            assert!(entry.tables.is_empty());
            assert!(!entry.tables_refused);
            assert!(entry.geometry.is_some());
            assert_eq!(entry.page_index, page_index);
        }
        // plain prose: three full-width lines
        let prose = "BT /F1 12 Tf 72 700 Td (It was a dark and stormy night; the rain fell) Tj ET\n\
             BT /F1 12 Tf 72 682 Td (over the plain, and the waves dashed against the rocks.) Tj ET\n\
             BT /F1 12 Tf 72 664 Td (The captain stood at the rail and watched the sea.) Tj ET";
        let page = one_table_page(tables_pdf(prose, ""));
        assert!(page.tables.is_empty());
        assert!(!page.tables_refused);
        // the TOC with inline leaders (measured out3: tables: 0)
        let toc = "BT /F1 18 Tf 72 720 Td (Contents) Tj ET\n\
             BT /F1 12 Tf 72 680 Td (Introduction . . . . . . . . . . . . 1) Tj ET\n\
             BT /F1 12 Tf 72 656 Td (Methods and materials . . . . . . . . . . . . 9) Tj ET\n\
             BT /F1 12 Tf 72 632 Td (Results of the study . . . . . . . . . . . . 17) Tj ET";
        let page = one_table_page(tables_pdf(toc, ""));
        assert!(
            page.tables.is_empty(),
            "the detector-limit pin: these bytes decline"
        );
        assert!(!page.tables_refused);
    }

    /// The multi-table page (measured out4): two ruled grids stacked far
    /// apart (disjoint x, a 100pt y gap — past the merge's reach) come
    /// back as TWO tables, in the engine's own order (the BOTTOM grid
    /// first — measured), each a clean 2×2 of ruling-rect cells.
    #[test]
    fn tables_a_multi_table_page_returns_two_tables_bottom_first() {
        let far_apart = "0.5 w 100 600 m 100 800 l S\n\
             0.5 w 200 600 m 200 800 l S\n\
             0.5 w 300 600 m 300 800 l S\n\
             0.5 w 100 600 m 300 600 l S\n\
             0.5 w 100 700 m 300 700 l S\n\
             0.5 w 100 800 m 300 800 l S\n\
             BT /F1 12 Tf 120 750 Td (T1) Tj ET\n\
             BT /F1 12 Tf 220 750 Td (T2) Tj ET\n\
             BT /F1 12 Tf 120 650 Td (T3) Tj ET\n\
             BT /F1 12 Tf 220 650 Td (T4) Tj ET\n\
             0.5 w 350 300 m 350 500 l S\n\
             0.5 w 450 300 m 450 500 l S\n\
             0.5 w 550 300 m 550 500 l S\n\
             0.5 w 350 300 m 550 300 l S\n\
             0.5 w 350 400 m 550 400 l S\n\
             0.5 w 350 500 m 550 500 l S\n\
             BT /F1 12 Tf 370 450 Td (B1) Tj ET\n\
             BT /F1 12 Tf 470 450 Td (B2) Tj ET\n\
             BT /F1 12 Tf 370 350 Td (B3) Tj ET\n\
             BT /F1 12 Tf 470 350 Td (B4) Tj ET";
        let page = one_table_page(tables_pdf(far_apart, ""));
        assert!(!page.tables_refused);
        assert_eq!(
            page.tables.len(),
            2,
            "measured out4: two grids far apart = two tables"
        );
        // the engine's order: the bottom grid first (measured)
        assert_eq!(row_texts(&page, 0), vec!["B1", "B2"]); // page.tables[0].rows[0]
        assert_eq!(page.tables[0].bbox, Some((350.0, 300.0, 200.0, 200.0)));
        assert_eq!(page.tables[1].bbox, Some((100.0, 600.0, 200.0, 200.0)));
        let _ = page.tables[0].rows[0].cells[0].bbox;
    }

    /// The A11 MERGE-path fixture: two disjoint-x ruled grids with a
    /// 20pt y gap merge into ONE table whose rows are BOTTOM-GROUP-FIRST
    /// (measured out2 — the merge sorts by top-y ascending and appends,
    /// the engine never re-sorts after merging) with every cell a
    /// ruling-rect box. Branch-deterministic, pinned as documented engine
    /// behaviour.
    #[test]
    fn tables_the_disjoint_x_vertical_merge_is_bottom_group_first() {
        let page = one_table_page(tables_pdf(STACKED_DISJOINT, ""));
        assert!(!page.tables_refused);
        assert_eq!(page.tables.len(), 1, "measured out2: the two grids MERGE");
        let table = &page.tables[0];
        assert_eq!(table.col_count, 2);
        assert_eq!(table.bbox, Some((100.0, 380.0, 450.0, 420.0)));
        assert_eq!(table.rows.len(), 4);
        // bottom group first: rows 0-1 are B1..B4 (the y=380..580 grid),
        // rows 2-3 are T1..T4 (the y=600..800 grid) — measured out2
        assert_eq!(row_texts(&page, 0), vec!["B1", "B2"]);
        assert_box_near(
            table.rows[0].cells[0].bbox.unwrap(),
            (350.0, 480.0, 100.0, 100.0),
            0.01,
            "B1",
        );
        assert_eq!(row_texts(&page, 1), vec!["B3", "B4"]);
        assert_eq!(row_texts(&page, 2), vec!["T1", "T2"]);
        assert_box_near(
            table.rows[2].cells[0].bbox.unwrap(),
            (100.0, 700.0, 100.0, 100.0),
            0.01,
            "T1",
        );
        assert_eq!(row_texts(&page, 3), vec!["T3", "T4"]);
        for row in &table.rows {
            assert!(!row.is_header);
            assert_eq!(row.cells.len(), table.col_count);
        }
    }

    /// The A11/A13 CLUSTER-path fixture: two SHARED-x ruled grids (2-col
    /// over 3-col, gap 20) route to the line-cluster branch — one table,
    /// TOP-GROUP-FIRST (measured out2), `has_header` TRUE (the cluster
    /// branch's second header source, A13), the table bbox the cluster's
    /// LINE-EXTENT × ROW-EXTENT (99.75/400.5 — the 0.5pt stroke widths
    /// included, NOT the member-span union, A13), and the 2-col group's
    /// rows padded to the 3-col grid with `"" None` cells (A2's pad, the
    /// None inside a ruling-ruled page).
    #[test]
    fn tables_the_shared_x_cluster_path_is_top_first_with_none_pads() {
        let page = one_table_page(tables_pdf(STACKED_SHARED_X, ""));
        assert!(!page.tables_refused);
        assert_eq!(page.tables.len(), 1);
        let table = &page.tables[0];
        assert_eq!(
            table.col_count, 3,
            "measured out2: the merged grid is 3 columns"
        );
        assert!(
            table.has_header,
            "the cluster branch's header heuristic, measured"
        );
        // A13: the cluster bbox is the LINE extent × ROW extent (the
        // stroke-width-inclusive 99.75..500.25 x 380..713), pinned by the
        // measured golden — not the cells' span-union.
        assert_box_near(
            table.bbox.unwrap(),
            (99.75, 380.0, 400.5, 333.0),
            0.01,
            "the cluster bbox",
        );
        assert_eq!(table.rows.len(), 2);
        // top group first (measured out2), padded to 3 with a None cell
        assert_eq!(row_texts(&page, 0), vec!["T1", "T2", ""]);
        assert_box_near(
            table.rows[0].cells[0].bbox.unwrap(),
            (120.0, 700.0, 14.004, 12.0),
            0.01,
            "T1",
        );
        assert_eq!(table.rows[0].cells[2].text, "");
        assert_eq!(table.rows[0].cells[2].bbox, None, "the None pad, measured");
        assert!(
            table.rows[0].is_header,
            "the header verdict rides the row too"
        );
        assert_eq!(row_texts(&page, 1), vec!["B1", "B2", "B3"]);
        assert_box_near(
            table.rows[1].cells[1].bbox.unwrap(),
            (250.0, 480.0, 14.676, 12.0),
            0.01,
            "B2",
        );
    }

    /// The A2 shape invariants, swept over every SERVED fixture here:
    /// `col_count` is the grid's column count and a row is never LONGER
    /// than it (merges only shorten — and the measured pad shape keeps
    /// length equality); every cell carries colspan/rowspan ≥ 1. The
    /// colspan>1/rowspan>1 MERGE ITSELF is flagged (spec §6's "fixture
    /// still to build"): six fixture shapes aimed at
    /// `detect_merged_cells_visually`'s two triggers (a band with the
    /// column rule missing; overflowing cell text) all either declined
    /// detection or came back colspan-1 (probe out5/out6/out7) — the
    /// pass-through fields are pinned == 1 so an engine bump that starts
    /// merging cannot arrive silently, and the ragged case stays a
    /// documented v2 fixture.
    #[test]
    fn tables_the_shape_invariants_hold_on_every_served_fixture() {
        let served = [
            (RULED_2X2, "", 1usize),
            (STACKED_DISJOINT, "", 1),
            (STACKED_SHARED_X, "", 1),
            (TEXT_ONLY_4X2, "", 1),
            (TEXT_ONLY_4X2, "/Rotate 90", 1),
            (SIDEWAYS_TEXT_ONLY, "/Rotate 270", 1),
            (RULED_2X2, "/CropBox [110 550 550 800]", 1),
        ];
        for (content, extras, table_count) in served {
            let page = one_table_page(tables_pdf(content, extras));
            assert_eq!(page.tables.len(), table_count, "{extras}");
            for table in &page.tables {
                assert!(table.col_count >= 2, "is_real_grid's own gate");
                assert!(table.bbox.is_some(), "Some on every returned table (A2)");
                for row in &table.rows {
                    assert!(
                        row.cells.len() <= table.col_count,
                        "the ragged law: never longer"
                    );
                    for cell in &row.cells {
                        assert!(cell.colspan >= 1 && cell.rowspan >= 1);
                        // the flagged merge pin: the measured engine never
                        // emits a merge in these fixtures — a drift here
                        // means the detector grew merges and the flagged
                        // fixture question re-opens
                        assert_eq!((cell.colspan, cell.rowspan), (1, 1), "the flagged pin");
                    }
                }
            }
        }
    }

    /// The A6 cap: a page with more than the markdown lane's
    /// MAX_TABLE_EDGES (1500) table primitives skips detection —
    /// `tables: []`, `tables_refused: true`, GEOMETRY INTACT (the A9
    /// reading rule: true at rotation 0 names the cap, not frame
    /// incoherence). 751 verticals + 751 horizontals = 1502 primitives;
    /// the call completes (the cap's whole point: no O(E²) sweep).
    #[test]
    fn tables_above_the_primitive_cap_skip_detection_and_refuse() {
        let mut content = String::new();
        for i in 0..751 {
            let x = 100.0 + i as f32 * 0.5;
            content.push_str(&format!("0.5 w {x} 600 m {x} 800 l S\n"));
        }
        for i in 0..751 {
            let y = 600.0 + i as f32 * 0.25;
            content.push_str(&format!("0.5 w 100 {y} m 500 {y} l S\n"));
        }
        let page = one_table_page(tables_pdf(&content, ""));
        assert!(page.tables.is_empty());
        assert!(page.tables_refused, "the cap refusal: skipped detection");
        assert_eq!(
            page.geometry.unwrap().rotation,
            0,
            "rotation 0 distinguishes the cap (A9)"
        );
        // just under the cap (1499 primitives) the sweep runs — and finds
        // no is_real_grid in these near-degenerate rules, the served-empty
        // shape (the cap is >, per the markdown lane's own bail-out)
        let mut content = String::new();
        for i in 0..749 {
            let x = 100.0 + i as f32 * 0.5;
            content.push_str(&format!("0.5 w {x} 600 m {x} 800 l S\n"));
        }
        for i in 0..750 {
            let y = 600.0 + i as f32 * 0.25;
            content.push_str(&format!("0.5 w 100 {y} m 500 {y} l S\n"));
        }
        let page = one_table_page(tables_pdf(&content, ""));
        assert!(
            !page.tables_refused,
            "1499 < 1500: no cap, the gate decides"
        );
    }

    /// The `pages=` law on the tables probe: `None` = whole document; a
    /// selection normalizes (sort + dedup) and the output is parallel to
    /// the NORMALIZED selection with every entry carrying its
    /// `page_index`; the empty and out-of-range refusals are
    /// `markdown_pages`' byte-identical strings.
    #[test]
    fn tables_pages_selection_is_normalized_and_parallel() {
        let mut objects = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>".to_vec(),
        ];
        for text in ["one", "two", "three"] {
            let page_number = objects.len() + 1;
            let font_number = page_number + 1;
            let stream_number = page_number + 2;
            let content = format!("BT /F1 12 Tf 50 700 Td ({text}) Tj ET");
            objects.push(
                format!(
                    "<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 {font_number} 0 R >> >> \
                 /MediaBox [0 0 612 792] /Contents {stream_number} 0 R >>"
                )
                .into_bytes(),
            );
            objects.push(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec());
            objects.push(
                format!(
                    "<< /Length {} >>\nstream\n{content}\nendstream",
                    content.len() + 1
                )
                .into_bytes(),
            );
        }
        let refs: Vec<&[u8]> = objects.iter().map(Vec::as_slice).collect();
        let bytes = objects_pdf(&refs);
        let whole = page_tables(bytes.clone(), None, None, "pdf").unwrap();
        assert_eq!(
            whole.iter().map(|p| p.page_index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        let subset = page_tables(bytes.clone(), Some(&[2, 0]), None, "pdf").unwrap();
        assert_eq!(
            subset.iter().map(|p| p.page_index).collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(subset[0].format, "pdf");
        // duplicates normalize away
        assert_eq!(
            page_tables(bytes.clone(), Some(&[1, 1]), None, "pdf").unwrap()[0].page_index,
            1
        );
        // the byte-pinned refusals (markdown_pages' strings, shared rule)
        let err = page_tables(bytes.clone(), Some(&[]), None, "pdf").unwrap_err();
        assert_eq!(
            err.to_string(),
            "pages= selected no pages: pass None for the whole document"
        );
        let err = page_tables(bytes, Some(&[3]), None, "pdf").unwrap_err();
        assert_eq!(
            err.to_string(),
            "page index 3 is out of range for a 3-page document (pages= is 0-based)"
        );
    }

    /// A10's PDF-leg order, pinned: `open` (the encrypted door) precedes
    /// the `pages=` normalization/validation, so an encrypted document
    /// with a MALFORMED selection answers the encryption refusal first;
    /// with the right password the range error is the answer; and an
    /// unlocked document passes through to the entry (the geometry-only
    /// fixture carries no tables).
    #[test]
    fn tables_the_encrypted_door_precedes_the_pages_validation() {
        let locked = encrypted_pdf("torque", "owner-pw");
        let err = page_tables(locked.clone(), Some(&[99]), None, "pdf").unwrap_err();
        assert!(
            matches!(err, PagesError::Pdf(PdfError::EncryptedPdf)),
            "A10: the door precedes the range check: {err}"
        );
        let err = page_tables(locked.clone(), Some(&[99]), Some("torque"), "pdf").unwrap_err();
        assert!(matches!(err, PagesError::Pages(_)), "{err}");
        let entries = page_tables(locked, Some(&[0]), Some("torque"), "pdf").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].geometry.unwrap().height_pt, 792.0);
        assert!(entries[0].tables.is_empty() && !entries[0].tables_refused);
    }

    /// The A6 content-failure law, pinned: a page whose content stream
    /// fails to decode (the broken-Flate fixture) answers `tables: []`
    /// with the geometry INTACT and `tables_refused` FALSE — the failure
    /// is UNKNOWABLE (indistinguishable from a no-tables page, by
    /// design): never an error, never a refusal the page's inputs cannot
    /// support.
    #[test]
    fn tables_content_failures_are_empty_not_errors_and_not_refusals() {
        let junk = b"this is not zlib data, not even close";
        let broken = format!(
            "<< /Length {} /Filter /FlateDecode >>\nstream\n{}",
            junk.len(),
            std::str::from_utf8(junk).unwrap()
        );
        let bytes = objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
            broken.as_bytes(),
        ]);
        let page = one_table_page(bytes);
        assert!(page.geometry.is_some(), "the page box is intact");
        assert!(page.tables.is_empty(), "the empty shape, not an error");
        assert!(
            !page.tables_refused,
            "unknowable: the documented indistinguishability"
        );
    }

    /// The empty-entry law (A5, unchanged): a page whose box read fails
    /// (no MediaBox) is a `geometry: None` entry with `tables: []` and
    /// `tables_refused` false, its siblings intact.
    #[test]
    fn tables_an_unreadable_page_box_is_the_empty_entry() {
        let bytes = objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
            b"<< /Type /Page /Parent 2 0 R >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
        ]);
        let entries = page_tables(bytes, None, None, "pdf").unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].geometry, None);
        assert!(entries[0].tables.is_empty() && !entries[0].tables_refused);
        assert_eq!(
            entries[1].geometry.unwrap().width_pt,
            612.0,
            "the sibling is intact"
        );
    }

    /// Zero pages → `[]`.
    #[test]
    fn tables_a_zero_page_document_yields_an_empty_list() {
        let bytes = objects_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [] /Count 0 >>",
        ]);
        assert_eq!(
            page_tables(bytes, None, None, "pdf").unwrap(),
            Vec::<PageTablesData>::new()
        );
    }

    /// The A1 gate at unit level: the ruling table's every listed row
    /// (A12's fail-closed default covers the unlisted ones — asserted as
    /// refusals too).
    #[test]
    fn the_frame_gate_matches_the_ruling_table() {
        use Frame::{Incoherent, Mapped, Raw};
        // rotation 0: raw, always
        assert_eq!(table_frame_gate(0, false, false, true), Raw);
        assert_eq!(table_frame_gate(0, true, false, true), Raw);
        // 90/270 without rulings: none-mapped raw / all-mapped mapped / mixed refused
        assert_eq!(table_frame_gate(90, false, false, false), Raw);
        assert_eq!(table_frame_gate(270, false, false, false), Raw);
        assert_eq!(table_frame_gate(90, false, true, true), Mapped);
        assert_eq!(table_frame_gate(270, false, true, true), Mapped);
        assert_eq!(table_frame_gate(90, false, true, false), Incoherent);
        // 90/270 with rulings: none-mapped raw / any-mapped refused
        assert_eq!(table_frame_gate(90, true, false, false), Raw);
        assert_eq!(table_frame_gate(270, true, false, false), Raw);
        assert_eq!(table_frame_gate(90, true, true, true), Incoherent);
        assert_eq!(table_frame_gate(270, true, true, false), Incoherent);
        // 180: all-mapped pathless serves; every rulings combination refuses
        assert_eq!(table_frame_gate(180, false, true, true), Mapped);
        assert_eq!(table_frame_gate(180, true, true, true), Incoherent);
        assert_eq!(table_frame_gate(180, true, true, false), Incoherent);
        assert_eq!(table_frame_gate(180, false, true, false), Incoherent);
        // the (any=false, all=true) combination is the ZERO-SPAN vacuity
        // (all ⊆ any for any real span set): a spanless 180 page — no
        // table input at all — serves raw-empty, refused false.
        assert_eq!(table_frame_gate(180, false, false, true), Raw);
        assert_eq!(table_frame_gate(180, true, false, true), Raw);
    }

    /// The pdfium-hazard property, tables side: 8 threads opening their
    /// own document over the same bytes, every answer byte-identical to
    /// the single-threaded one (the pass is `Send` end to end, like every
    /// entry point).
    #[test]
    fn tables_concurrent_calls_are_byte_identical() {
        let bytes = std::sync::Arc::new(tables_pdf(RULED_2X2, ""));
        let expected = page_tables((*bytes).clone(), None, None, "pdf").unwrap();
        let mut handles = Vec::new();
        for _ in 0..8 {
            let bytes = std::sync::Arc::clone(&bytes);
            handles.push(std::thread::spawn(move || {
                page_tables((*bytes).clone(), None, None, "pdf").unwrap()
            }));
        }
        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected);
        }
    }
}
