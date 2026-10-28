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

    // --- the page-geometry probe (the §9 PDF-side pins) --------------------

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
}
