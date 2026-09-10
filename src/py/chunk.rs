use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::EagerIter;
use crate::chunk_by_segment_impl;
use crate::chunk_hierarchical_impl;
use crate::chunk_impl;
use crate::parse_boundary;
use crate::py::eager_iter_class;

/// `tors.chunk_cdc(data: bytes, *, min_size=4096, avg_size=16384,
/// max_size=65534) -> list[tuple[int, int]]`: FastCDC 2020 content-defined
/// chunking. Produces `(start, end)` byte-offset spans (not codepoints; this
/// is a byte-level primitive, unlike `word_bounds`/`sentence_bounds`)
/// partitioning `data` exactly, in one GIL-released native pass.
/// Content-defined chunking's advantage over fixed-size chunking: a small
/// edit near the start of `data` only perturbs the 1-2 chunks nearest the
/// edit, not every boundary after it, which makes it the natural upstream of
/// `merkle_root`/`merkle_diff`'s `list[bytes]` for a byte-level
/// dedup/incremental-sync pipeline. Empty input returns `[]`; input shorter
/// than `min_size` returns one chunk covering the whole input. Deterministic:
/// the same bytes at the same parameters always cut at the same offsets.
///
/// `min_size`/`avg_size`/`max_size` must satisfy `fastcdc`'s own documented
/// bounds (each even; `min_size` in `[64, 1_048_576]`, `avg_size` in
/// `[256, 4_194_304]`, `max_size` in `[1024, 16_777_216]`) and
/// `min_size <= avg_size <= max_size`. Violating any of this raises
/// `ValueError` before any chunking runs (the crate itself only
/// `debug_assert!`s these, a no-op in a release build, so tors validates
/// them itself rather than risk silent misbehavior on an out-of-range call).
/// A negative `min_size`/`avg_size`/`max_size` raises `ValueError` too, the
/// same error type and message style as every other size/count argument in
/// this family: not `OverflowError` (a raw `usize` parameter would let
/// pyo3's own negative-to-unsigned conversion raise that instead, breaking
/// the family's shared error-contract). Defaults are the crate's own
/// documented example values.
///
/// GIL model: the whole scan runs under one `py.detach`; the return
/// marshalling is O(chunks) 2-tuples of ints.
#[pyfunction(signature = (data, *, min_size = 4096, avg_size = 16384, max_size = 65534))]
pub fn chunk_cdc(
    py: Python<'_>,
    data: &[u8],
    min_size: i64,
    avg_size: i64,
    max_size: i64,
) -> PyResult<Vec<(usize, usize)>> {
    if min_size < 0 {
        return Err(PyValueError::new_err(format!(
            "min_size must be >= 0, got {min_size}"
        )));
    }
    if avg_size < 0 {
        return Err(PyValueError::new_err(format!(
            "avg_size must be >= 0, got {avg_size}"
        )));
    }
    if max_size < 0 {
        return Err(PyValueError::new_err(format!(
            "max_size must be >= 0, got {max_size}"
        )));
    }
    let min_size = min_size as usize;
    let avg_size = avg_size as usize;
    let max_size = max_size as usize;
    py.detach(|| chunk_impl::chunk_cdc(data, min_size, avg_size, max_size))
        .map_err(|err| PyValueError::new_err(err.0))
}

/// `tors.chunk_text(text, max_chars, *, overlap=0, boundary="word")`:
/// boundary-aware chunking of text into `(start, end)` pairs in Python
/// str index (codepoint) units, each chunk at most `max_chars`
/// codepoints, cut at word or sentence boundaries wherever the budget
/// allows. This is the context-window/RAG packing primitive: it cuts a
/// document at a safe boundary under a token-proxy budget instead of the
/// naive mid-word slice. The cut rule is `truncate_to_bounds`'s own,
/// applied repeatedly: the largest boundary end within the budget, with a
/// grapheme-safe hard cut when a single word/sentence exceeds it. Every
/// cut is also grapheme-cluster-safe, so `max_chars` can be exceeded only
/// in the pathological case of a single cluster (e.g. an oversized ZWJ
/// emoji chain) wider than the remaining budget; ordinary text never hits
/// this.
///
/// `overlap=0` (the default): the original lossless-partition contract,
/// byte-for-byte unchanged. Chunks are non-empty, contiguous, strictly
/// increasing, cover `[0, len(text))`, and joining the slices reproduces
/// the input exactly (interior-cut trailing whitespace rides the next
/// chunk's head rather than being dropped; the final chunk runs to the
/// end untrimmed). See `src/chunk_impl.rs` for the full contract.
///
/// `overlap > 0`: each chunk after the first starts `overlap` codepoints
/// before the previous chunk's end, snapped to the nearest `boundary`,
/// never mid-word/mid-sentence: the RAG-retrieval shape where a fact
/// split across a cut is still whole in at least one chunk. This trades
/// the lossless-join guarantee for genuine overlap; every chunk's own
/// `<= max_chars` and boundary-safety invariants still hold. `overlap`
/// must be `< max_chars` (no forward progress otherwise), which raises
/// `ValueError`; a chunk shorter than the requested overlap silently
/// degrades to zero overlap for just that one transition rather than
/// stall or violate the budget (documented on
/// `chunk_impl::chunk_text_overlapping`).
///
/// `max_chars < 1` or `overlap < 0` raise `ValueError`; an unrecognized
/// `boundary` raises `ValueError` (the `truncate_to_bounds` spelling).
///
/// GIL model: the same shape as `word_bounds`/`chunk_by_words`: the
/// whole-text bounds computed once under one `py.detach`, the return
/// marshalling O(chunks) 2-tuples of ints under the GIL.
#[pyfunction(signature = (text, max_chars, *, overlap = 0, boundary = "word"))]
pub fn chunk_text(
    py: Python<'_>,
    text: &str,
    max_chars: i64,
    overlap: i64,
    boundary: &str,
) -> PyResult<Vec<(usize, usize)>> {
    if max_chars < 1 {
        return Err(PyValueError::new_err(format!(
            "max_chars must be >= 1, got {max_chars}"
        )));
    }
    if overlap < 0 {
        return Err(PyValueError::new_err(format!(
            "overlap must be >= 0, got {overlap}"
        )));
    }
    if overlap >= max_chars {
        return Err(PyValueError::new_err(format!(
            "overlap must be < max_chars, got overlap={overlap}, max_chars={max_chars}"
        )));
    }
    let boundary = parse_boundary(boundary)?;
    let max_chars = max_chars as usize;
    let overlap = overlap as usize;
    let chunks =
        py.detach(|| chunk_impl::chunk_text_overlapping(text, max_chars, overlap, boundary));
    Ok(chunks)
}

eager_iter_class! {
    /// The streaming twin of [`chunk_text`]: same `(start, end)` sequence, same
    /// argument contract, the [`EagerIter`] shape (whole scan under one
    /// `py.detach` at construction, one 2-tuple per `__next__`) `word_bounds_iter`
    /// already established. Avoids materializing a `list` for documents that
    /// chunk into the hundreds of thousands of pieces, where the GIL-held
    /// tuple-marshalling cost `word_bounds_iter`'s docs measure would otherwise
    /// dominate.
    ChunkTextIter, (usize, usize);
}

/// `tors.chunk_text_iter(text, max_chars, *, overlap=0, boundary="word")`:
/// [`chunk_text`]'s streaming spelling; see [`ChunkTextIter`].
#[pyfunction(signature = (text, max_chars, *, overlap = 0, boundary = "word"))]
pub fn chunk_text_iter(
    py: Python<'_>,
    text: Bound<'_, PyString>,
    max_chars: i64,
    overlap: i64,
    boundary: &str,
) -> PyResult<Py<ChunkTextIter>> {
    if max_chars < 1 {
        return Err(PyValueError::new_err(format!(
            "max_chars must be >= 1, got {max_chars}"
        )));
    }
    if overlap < 0 {
        return Err(PyValueError::new_err(format!(
            "overlap must be >= 0, got {overlap}"
        )));
    }
    if overlap >= max_chars {
        return Err(PyValueError::new_err(format!(
            "overlap must be < max_chars, got overlap={overlap}, max_chars={max_chars}"
        )));
    }
    let boundary_kind = parse_boundary(boundary)?;
    let max_chars = max_chars as usize;
    let overlap = overlap as usize;
    let s = text.to_str()?;
    let chunks =
        py.detach(|| chunk_impl::chunk_text_overlapping(s, max_chars, overlap, boundary_kind));
    Py::new(py, ChunkTextIter(EagerIter::new(py, text, chunks)))
}

/// `tors.chunk_by_words(text, words_per_chunk, *, overlap=0)`:
/// word-count-windowed chunking. Each chunk spans `words_per_chunk`
/// consecutive real word tokens, not `word_bounds`' raw segment count:
/// `word_bounds` gives an inter-word space run its own segment, so
/// grouping raw segments would silently mean "words_per_chunk roughly
/// halved" on ordinary prose (see `chunk_by_segment_impl::chunk_by_words`'s doc for
/// the full reasoning). `(start, end)` codepoint offsets span the first
/// included token's start through the last's end, not through any
/// trailing whitespace after it, so unlike `chunk_text` this is not a
/// covering partition (chunks are not necessarily contiguous). `overlap`
/// words repeat at the start of the next chunk: the semantic-chunking
/// RAG shape, measured in units rather than a character budget. The
/// final chunk may hold fewer than `words_per_chunk` tokens when the
/// total doesn't divide evenly. Empty text, or text with no word tokens
/// at all, returns `[]`.
///
/// `words_per_chunk < 1` or `overlap < 0` raise `ValueError`; `overlap >=
/// words_per_chunk` raises `ValueError` (no forward progress: each
/// chunk's stride is `words_per_chunk - overlap` tokens, and unlike
/// `chunk_text`'s character-granularity overlap this stride is always
/// `>= 1` by construction once validated, so no runtime snap-fallback
/// is needed).
///
/// GIL model: the whole segment + window pass runs under one
/// `py.detach`; the return marshalling is O(chunks) 2-tuples of ints.
#[pyfunction(signature = (text, words_per_chunk, *, overlap = 0))]
pub fn chunk_by_words(
    py: Python<'_>,
    text: &str,
    words_per_chunk: i64,
    overlap: i64,
) -> PyResult<Vec<(usize, usize)>> {
    if words_per_chunk < 1 {
        return Err(PyValueError::new_err(format!(
            "words_per_chunk must be >= 1, got {words_per_chunk}"
        )));
    }
    if overlap < 0 {
        return Err(PyValueError::new_err(format!(
            "overlap must be >= 0, got {overlap}"
        )));
    }
    if overlap >= words_per_chunk {
        return Err(PyValueError::new_err(format!(
            "overlap must be < words_per_chunk, got overlap={overlap}, words_per_chunk={words_per_chunk}"
        )));
    }
    let words_per_chunk = words_per_chunk as usize;
    let overlap = overlap as usize;
    Ok(py.detach(|| chunk_by_segment_impl::chunk_by_words(text, words_per_chunk, overlap)))
}

eager_iter_class! {
    /// The streaming twin of [`chunk_by_words`]: same shape as [`ChunkTextIter`].
    ChunkByWordsIter, (usize, usize);
}

/// `tors.chunk_by_words_iter(text, words_per_chunk, *, overlap=0)`:
/// [`chunk_by_words`]'s streaming spelling; see [`ChunkByWordsIter`].
#[pyfunction(signature = (text, words_per_chunk, *, overlap = 0))]
pub fn chunk_by_words_iter(
    py: Python<'_>,
    text: Bound<'_, PyString>,
    words_per_chunk: i64,
    overlap: i64,
) -> PyResult<Py<ChunkByWordsIter>> {
    if words_per_chunk < 1 {
        return Err(PyValueError::new_err(format!(
            "words_per_chunk must be >= 1, got {words_per_chunk}"
        )));
    }
    if overlap < 0 {
        return Err(PyValueError::new_err(format!(
            "overlap must be >= 0, got {overlap}"
        )));
    }
    if overlap >= words_per_chunk {
        return Err(PyValueError::new_err(format!(
            "overlap must be < words_per_chunk, got overlap={overlap}, words_per_chunk={words_per_chunk}"
        )));
    }
    let words_per_chunk = words_per_chunk as usize;
    let overlap = overlap as usize;
    let s = text.to_str()?;
    let chunks = py.detach(|| chunk_by_segment_impl::chunk_by_words(s, words_per_chunk, overlap));
    Py::new(py, ChunkByWordsIter(EagerIter::new(py, text, chunks)))
}

/// `tors.chunk_by_sentences(text, sentences_per_chunk, *, overlap=0)`:
/// [`chunk_by_words`]'s sentence-count twin. Each chunk spans
/// `sentences_per_chunk` consecutive UAX #29 sentence segments
/// (`sentence_bounds`), `overlap` sentences repeated. Same argument
/// contract, same empty-input answer, same forward-progress-by-
/// construction guarantee.
///
/// GIL model: identical to [`chunk_by_words`].
#[pyfunction(signature = (text, sentences_per_chunk, *, overlap = 0))]
pub fn chunk_by_sentences(
    py: Python<'_>,
    text: &str,
    sentences_per_chunk: i64,
    overlap: i64,
) -> PyResult<Vec<(usize, usize)>> {
    if sentences_per_chunk < 1 {
        return Err(PyValueError::new_err(format!(
            "sentences_per_chunk must be >= 1, got {sentences_per_chunk}"
        )));
    }
    if overlap < 0 {
        return Err(PyValueError::new_err(format!(
            "overlap must be >= 0, got {overlap}"
        )));
    }
    if overlap >= sentences_per_chunk {
        return Err(PyValueError::new_err(format!(
            "overlap must be < sentences_per_chunk, got overlap={overlap}, sentences_per_chunk={sentences_per_chunk}"
        )));
    }
    let sentences_per_chunk = sentences_per_chunk as usize;
    let overlap = overlap as usize;
    Ok(py.detach(|| chunk_by_segment_impl::chunk_by_sentences(text, sentences_per_chunk, overlap)))
}

eager_iter_class! {
    /// The streaming twin of [`chunk_by_sentences`]: same shape as [`ChunkTextIter`].
    ChunkBySentencesIter, (usize, usize);
}

/// `tors.chunk_by_sentences_iter(text, sentences_per_chunk, *, overlap=0)`:
/// [`chunk_by_sentences`]'s streaming spelling; see [`ChunkBySentencesIter`].
#[pyfunction(signature = (text, sentences_per_chunk, *, overlap = 0))]
pub fn chunk_by_sentences_iter(
    py: Python<'_>,
    text: Bound<'_, PyString>,
    sentences_per_chunk: i64,
    overlap: i64,
) -> PyResult<Py<ChunkBySentencesIter>> {
    if sentences_per_chunk < 1 {
        return Err(PyValueError::new_err(format!(
            "sentences_per_chunk must be >= 1, got {sentences_per_chunk}"
        )));
    }
    if overlap < 0 {
        return Err(PyValueError::new_err(format!(
            "overlap must be >= 0, got {overlap}"
        )));
    }
    if overlap >= sentences_per_chunk {
        return Err(PyValueError::new_err(format!(
            "overlap must be < sentences_per_chunk, got overlap={overlap}, sentences_per_chunk={sentences_per_chunk}"
        )));
    }
    let sentences_per_chunk = sentences_per_chunk as usize;
    let overlap = overlap as usize;
    let s = text.to_str()?;
    let chunks =
        py.detach(|| chunk_by_segment_impl::chunk_by_sentences(s, sentences_per_chunk, overlap));
    Py::new(py, ChunkBySentencesIter(EagerIter::new(py, text, chunks)))
}

/// `tors.chunk_by_paragraphs(text, paragraphs_per_chunk, *, overlap=0)`:
/// [`chunk_by_words`]/[`chunk_by_sentences`]'s paragraph-count twin. Each
/// chunk spans `paragraphs_per_chunk` consecutive paragraphs, `overlap`
/// paragraphs repeated. A paragraph boundary here is a run of 2+
/// consecutive newline characters (`\r\n` counts as one unit, matching
/// `tors.normalize`'s own CR/CRLF folding), the same "2+ newlines is the
/// surviving paragraph gap" convention `normalize` already establishes.
/// This is a heuristic, not a Unicode Standard segmentation (there is no
/// UAX for paragraphs, unlike UAX #29 for words/sentences): a single `\n`
/// is ordinary content, not a break. Same argument contract, same
/// empty-input answer, same forward-progress-by-construction guarantee
/// as its siblings.
///
/// GIL model: identical to [`chunk_by_words`].
#[pyfunction(signature = (text, paragraphs_per_chunk, *, overlap = 0))]
pub fn chunk_by_paragraphs(
    py: Python<'_>,
    text: &str,
    paragraphs_per_chunk: i64,
    overlap: i64,
) -> PyResult<Vec<(usize, usize)>> {
    if paragraphs_per_chunk < 1 {
        return Err(PyValueError::new_err(format!(
            "paragraphs_per_chunk must be >= 1, got {paragraphs_per_chunk}"
        )));
    }
    if overlap < 0 {
        return Err(PyValueError::new_err(format!(
            "overlap must be >= 0, got {overlap}"
        )));
    }
    if overlap >= paragraphs_per_chunk {
        return Err(PyValueError::new_err(format!(
            "overlap must be < paragraphs_per_chunk, got overlap={overlap}, paragraphs_per_chunk={paragraphs_per_chunk}"
        )));
    }
    let paragraphs_per_chunk = paragraphs_per_chunk as usize;
    let overlap = overlap as usize;
    Ok(py
        .detach(|| chunk_by_segment_impl::chunk_by_paragraphs(text, paragraphs_per_chunk, overlap)))
}

/// `tors.chunk_hierarchical(text, max_chars, separators=None, *,
/// overlap=0)`: priority-ordered fallback chunking. Produces `(start, end)`
/// codepoint pairs, each chunk at most `max_chars` codepoints, cut at the
/// coarsest separator level that fits within budget, falling back to
/// progressively finer levels only when a coarser one has no in-budget
/// cut over the current window. This is the same pattern LangChain's
/// `RecursiveCharacterTextSplitter` popularized (default separators
/// `["\n\n", "\n", " ", ""]`), except tors's default hierarchy uses its
/// own accurate UAX #29 segmenters instead of literal guesses.
///
/// `separators=None` (the default): paragraph → sentence → word → a
/// grapheme-safe raw cut, always the final, unconditional fallback that
/// never fails to produce a chunk. `separators=[...]`: a caller-supplied
/// list of literal strings (not regex; a documented scope line, see
/// `src/chunk_hierarchical_impl.rs`), coarsest first, e.g.
/// `["\n## ", "\n\n", ". ", " "]` for markdown-header-aware chunking.
/// This replaces the default hierarchy for the levels it specifies, but the
/// grapheme-safe raw cut is still always appended as the final fallback
/// regardless (unlike LangChain, no trailing `""` sentinel is required).
/// A `None` entry in an otherwise-literal list splices the default
/// hierarchy's three accurate levels in at that position:
/// `["\n", None]` is line → paragraph → sentence → word → raw cut: the
/// line-oriented-text shape (a chat thread, one message per line, never
/// split mid-line) whose oversized-line fallback is the real UAX #29
/// segmenter rather than the `". "`/`" "` literal guesses an all-literal
/// list would pin it to. `[None]` is identical to `separators=None`.
///
/// Unlike `chunk_text`, this is not a lossless covering partition: at
/// every level except the raw cut, the separator itself is dropped
/// between chunks (the same convention `chunk_by_paragraphs` already
/// applies to blank-line runs), since a caller splitting on a marker wants
/// it gone, not duplicated. `overlap` snaps the next chunk's start backward
/// to the nearest grapheme boundary (not necessarily a semantic
/// paragraph/sentence/word boundary; a documented simplification of the
/// single-hierarchy overlap snap `chunk_text_overlapping` uses), with the
/// same snap-collapse-to-zero-overlap degradation when the grapheme-snapped
/// overlap target would reach the chunk's own start (a too-short chunk, or
/// one whose overlap window is consumed by a multi-codepoint cluster such
/// as `\r\n`).
///
/// `max_chars < 1` or `overlap < 0` raise `ValueError`. Empty `text`
/// returns `[]`. An empty `separators` list is legal and skips straight to
/// the raw-cut fallback for every chunk.
///
/// Cost at document scale is the levels your budget actually consults:
/// each level's scan runs at most once per call, at its first
/// consultation, so a budget that never falls past the paragraph level
/// never pays the sentence or word walks at all, and duplicate entries
/// (`None` or a repeated literal) are deduped: `[None] * 100` and
/// `[" "] * 100` cost what the single entry does.
///
/// GIL model: identical to `chunk_by_words`. The whole multi-level scan
/// runs under one `py.detach`; the return marshalling is O(chunks)
/// 2-tuples of ints.
#[pyfunction(signature = (text, max_chars, separators = None, *, overlap = 0))]
pub fn chunk_hierarchical(
    py: Python<'_>,
    text: &str,
    max_chars: i64,
    separators: Option<Vec<Option<String>>>,
    overlap: i64,
) -> PyResult<Vec<(usize, usize)>> {
    if max_chars < 1 {
        return Err(PyValueError::new_err(format!(
            "max_chars must be >= 1, got {max_chars}"
        )));
    }
    if overlap < 0 {
        return Err(PyValueError::new_err(format!(
            "overlap must be >= 0, got {overlap}"
        )));
    }
    if overlap >= max_chars {
        return Err(PyValueError::new_err(format!(
            "overlap must be < max_chars, got overlap={overlap}, max_chars={max_chars}"
        )));
    }
    let max_chars = max_chars as usize;
    let overlap = overlap as usize;
    // The one intermediate materialization pyo3's borrowed-Vec
    // limitation forces (`Option<Vec<Option<&str>>>` cannot be extracted
    // directly: FromPyObject is not general enough over the borrowed
    // element lifetime): O(list) transient, owned Strings freed with the
    // call. The slot list the core builds from it is O(distinct entries)
    // after the dedup, so the pathological `[None] * N` spellings pay
    // this pass and nothing beyond it.
    let seps: Option<Vec<Option<&str>>> = separators
        .as_ref()
        .map(|v| v.iter().map(|entry| entry.as_deref()).collect());
    Ok(py.detach(|| {
        chunk_hierarchical_impl::chunk_hierarchical(text, max_chars, seps.as_deref(), overlap)
    }))
}

/// `tors.chunk_by_lines(text, lines_per_chunk, *, overlap=0)`:
/// [`chunk_by_words`]/[`chunk_by_sentences`]/[`chunk_by_paragraphs`]'s
/// line-count twin. Each chunk spans `lines_per_chunk` consecutive lines,
/// `overlap` lines repeated at the start of the next chunk. A line break
/// is a `\n`, a lone `\r`, or a `\r\n` pair counted as one unit (the same
/// CR/CRLF folding convention `chunk_by_paragraphs` and `normalize`'s own
/// pipeline use; `str.splitlines`' exotic separators (`\v`, `\f`, NEL,
/// LS, PS) are not breaks here). A line counts as a line only when it
/// carries at least one non-whitespace codepoint, the same real-token
/// discipline `chunk_by_words` applies to word segments: blank lines
/// neither count toward `lines_per_chunk` nor split a chunk's interior
/// (they ride along inside a chunk's span exactly as inter-word
/// whitespace rides along in `chunk_by_words`), so a caller reaching for
/// `lines_per_chunk=200` gets 200 content lines. "Non-whitespace" is
/// definitional here: the Unicode `White_Space` property
/// (`char::is_whitespace`), under which an NBSP-only line is blank and
/// U+001C–U+001F (FS/GS/RS/US) count as line content, diverging from
/// Python's `str.isspace()` (which treats those four as whitespace) and
/// from `str.splitlines` (which even breaks on them; tors does not).
/// `(start, end)` offsets span the first included line's start through
/// the last included line's end (not through the trailing break after
/// it: non-overlapping chunks are not necessarily contiguous). The final
/// chunk may hold fewer lines when the total doesn't divide evenly.
/// Empty text, or text with no content lines at all, returns `[]`. A
/// trailing break at end of text yields no trailing empty line.
///
/// `lines_per_chunk < 1` or `overlap < 0` raise `ValueError`; `overlap >=
/// lines_per_chunk` raises `ValueError` (no forward progress: each
/// chunk's stride is `lines_per_chunk - overlap` lines, always `>= 1` by
/// construction once validated, so no runtime snap-fallback is needed).
///
/// GIL model: identical to [`chunk_by_words`].
#[pyfunction(signature = (text, lines_per_chunk, *, overlap = 0))]
pub fn chunk_by_lines(
    py: Python<'_>,
    text: &str,
    lines_per_chunk: i64,
    overlap: i64,
) -> PyResult<Vec<(usize, usize)>> {
    if lines_per_chunk < 1 {
        return Err(PyValueError::new_err(format!(
            "lines_per_chunk must be >= 1, got {lines_per_chunk}"
        )));
    }
    if overlap < 0 {
        return Err(PyValueError::new_err(format!(
            "overlap must be >= 0, got {overlap}"
        )));
    }
    if overlap >= lines_per_chunk {
        return Err(PyValueError::new_err(format!(
            "overlap must be < lines_per_chunk, got overlap={overlap}, lines_per_chunk={lines_per_chunk}"
        )));
    }
    let lines_per_chunk = lines_per_chunk as usize;
    let overlap = overlap as usize;
    Ok(py.detach(|| chunk_by_segment_impl::chunk_by_lines(text, lines_per_chunk, overlap)))
}

eager_iter_class! {
    /// The streaming twin of [`chunk_by_lines`]: same shape as
    /// [`ChunkTextIter`], and the same rationale as every other `_iter`
    /// spelling: the list shape's GIL-held marshalling cost is measured
    /// for segment-count-heavy outputs (`word_bounds` on 12 MiB of
    /// prose, 3.67M segments, holds the GIL for 428-497ms just
    /// marshalling the list), and a line-oriented corpus (a multi-MiB
    /// log or transcript) is in that piece-count class, chunking into
    /// hundreds of thousands of pieces.
    ChunkByLinesIter, (usize, usize);
}

/// `tors.chunk_by_lines_iter(text, lines_per_chunk, *, overlap=0)`:
/// [`chunk_by_lines`]'s streaming spelling; see [`ChunkByLinesIter`].
#[pyfunction(signature = (text, lines_per_chunk, *, overlap = 0))]
pub fn chunk_by_lines_iter(
    py: Python<'_>,
    text: Bound<'_, PyString>,
    lines_per_chunk: i64,
    overlap: i64,
) -> PyResult<Py<ChunkByLinesIter>> {
    if lines_per_chunk < 1 {
        return Err(PyValueError::new_err(format!(
            "lines_per_chunk must be >= 1, got {lines_per_chunk}"
        )));
    }
    if overlap < 0 {
        return Err(PyValueError::new_err(format!(
            "overlap must be >= 0, got {overlap}"
        )));
    }
    if overlap >= lines_per_chunk {
        return Err(PyValueError::new_err(format!(
            "overlap must be < lines_per_chunk, got overlap={overlap}, lines_per_chunk={lines_per_chunk}"
        )));
    }
    let lines_per_chunk = lines_per_chunk as usize;
    let overlap = overlap as usize;
    let s = text.to_str()?;
    let chunks = py.detach(|| chunk_by_segment_impl::chunk_by_lines(s, lines_per_chunk, overlap));
    Py::new(py, ChunkByLinesIter(EagerIter::new(py, text, chunks)))
}
