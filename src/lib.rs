//! Fast, GIL-free text normalization for Python, backed by Rust.
//!
//! The pure-Rust paths live in [`normalize_impl`] (the v0.1 transform),
//! [`finalize_impl`] (the transform plus its SHA-256 tail, and its bytes-in
//! flavors), [`decode_impl`] (CPython-parity UTF-8 decoding of raw bytes),
//! [`b64_impl`] (RFC 4648 standard base64, both directions), [`forms_impl`]
//! (the v0.3 standalone normalization forms), [`html_impl`] + [`html_table`]
//! (CPython-parity HTML entity unescaping over generated tables),
//! [`segmentation_impl`] (UAX #29 grapheme/word segmentation),
//! [`utf8_impl`] (SIMD UTF-8 validity), [`diff_impl`] (character-level
//! opcode diffs in difflib's shape), [`search_impl`] (leftmost-longest
//! multi-pattern search), [`truncate_impl`] (boundary-safe and
//! ellipsis-marked truncation), and [`controls_impl`] (C0/DEL control-run
//! scrub); they are
//! public so the criterion benches (benches/normalize.rs, benches/bytes.rs,
//! benches/text.rs, benches/utf8.rs, benches/diff.rs, benches/search.rs)
//! drive them directly:
//! the pyo3 wrappers in this file only add the argument borrow and return
//! marshalling, which the Python-side tests measure separately.
//!
//! # GIL model
//!
//! The wrappers release the GIL for the whole native pass via `py.detach` (pyo3 0.26's
//! rename of `allow_threads`), the point of `tors`: Python's `re`/`str` methods never
//! release the GIL regardless of input size, so this is one native pass instead of a
//! GIL-hold-bounding chunked walk. The GIL-held residue of a call is only the argument
//! extraction plus the return marshalling of the output `String`s (O(output)), and the
//! argument borrow itself is a zero-copy alias for ASCII or already-cached inputs; on
//! the first non-ASCII call it is instead a one-time O(input) UTF-8 materialization
//! under the GIL (`&str` extraction builds and caches the UTF-8 copy of the input);
//! measured ~5-8ms at 12 MiB and ~16-24ms at 32 MiB (decomposed corpus; a
//! fresh-object `str.encode("utf-8")` of the same corpus, the same conversion, costs
//! 7.7-8.6 / 21.4-23.5ms). Callers that care keep a `to_thread` wrap.
//! `tests/test_gil_release.py` pins both bands, the marshalling band on ASCII
//! corpora and the first-call materialization on decomposed ones; and the README
//! records their measured sizes.
//!
//! The v0.2 bytes-in functions (`decode_utf8`, `finalize_utf8`, `b64_encode_bytes`)
//! have NO argument-materialization class at all: pyo3's `&[u8]` extraction is
//! `cast::<PyBytes>()?.as_bytes()`, a zero-copy borrow of the immutable bytes
//! buffer, read verbatim in pyo3 0.29's source (src/conversions/std/slice.rs),
//! for every input, ASCII or not. The only GIL-held residue of a bytes-in call is the
//! return marshalling (O(output)); the one exception is the strict-decode ERROR
//! path, which additionally constructs the `UnicodeDecodeError` (its `.object` is
//! a copy of the input, the same value CPython's own decoder sets there). The
//! bytes-in cells of `tests/test_gil_release.py` pin this band, including the
//! decomposed corpus whose str-in twin pays the first-call materialization.
//!
//! `utf8_is_valid` (the v0.5 bytes-in addition) is that family's extreme
//! point: the return is a single `bool`, so there is no marshalling class at
//! all, and no exception path either (the boolean is the answer; nothing
//! raises, valid input and invalid input alike), which makes the argument
//! borrow alone the call's entire GIL-held residue, on both paths.
//!
//! The v0.3 surface adds residue classes of its own, all measured and pinned
//! by the same suite. The complete inventory of O(input) GIL-held scans
//! OUTSIDE `py.detach` on the success path, reconciled to the v0.4 tree (the
//! v0.2/v0.3 list's M6; v0.4 moved or converted some): (1) the argument
//! borrow above, paid by every str-in function, and (2) `b64_decode`'s
//! `s.is_ascii()`, a whole-string byte scan before the detach, measured
//! 1.3µs at 16 MiB and 6.2µs at 43 MiB of b64 text (the 32 MiB-decoded
//! cell's input), three orders of magnitude under the 10ms ping floor;
//! negligible, but named here because it is the only such scan besides the
//! borrow. (The html `&` sentinel scan does not belong on this list: the
//! `detached_transform` restructure moved the memchr bail INSIDE the
//! detach.) The str-in functions (`nfc`/`nfd`/`nfkc`/`nfkd`,
//! `html_unescape`) pay the v0.1 classes (zero-copy ASCII borrow or the
//! one-time O(input) UTF-8 materialization; O(output) return marshalling);
//! `html_unescape`'s zero-`&` fast path returns the INPUT `PyString` itself
//! (CPython's own `return s`), not a marshalled copy. `b64_decode` is str-in
//! (ASCII-only, so the borrow of ACCEPTED input is always zero-copy; a
//! non-ASCII input pays the extraction's one-time materialization BEFORE
//! `is_ascii` rejects it, an error-path cost, paid once per object) with an
//! O(decoded-bytes) bytes return, plus an ERROR path that imports `binascii`
//! and constructs the real `binascii.Error` under the GIL: raise-time only.
//! `grapheme_count` returns a single int (no marshalling class at all), but
//! `word_bounds` returns the FULL bounds list: its GIL-held residue is
//! O(number-of-segments) tuple construction, measured at 428-497ms for 3.67M
//! segments (12 MiB prose), a real cost of the list-returning shape, pinned
//! as a regression band (the 1.0s ceiling plus the 0.85 detach-regression
//! ratio) by `tests/test_gil_release.py`.
//!
//! The v0.4 surface adds the ZERO-COST identity path and the streaming answer to
//! the word_bounds marshalling cost, both measured and pinned. Every
//! normalization entry (`normalize`, `finalize`, `finalize_utf8`, the four
//! forms, `html_unescape`) consults the quick-check property data (the same
//! data CPython's `unicodedata.normalize` fast path uses) plus, for the
//! pipeline, SIMD sentinel scans of each scan stage's fingerprint; when the
//! whole transform is provably a no-op the ORIGINAL input object is returned:
//! zero allocation, zero copy, zero marshalling (CPython's
//! `unicode_result_unchanged` idiom), and `finalize` computes its digest
//! straight from the borrowed input buffer. A post-pass output==input
//! comparison completes the contract (`f(s) is s` exactly when `f(s) == s`).
//! On such inputs the GIL-held residue reduces to the argument borrow alone:
//! measured 2.89ms walls for `normalize` at 12 MiB of already-normalized
//! prose (123.0ms on the v0.3 build), 7.72ms for `finalize` (126.3ms), the
//! whole call cheaper than one 10ms heartbeat. `word_bounds_iter` fills its
//! bounds buffer under ONE detach at construction (the same detached core
//! pass as the list API) and each `__next__` holds the GIL for a single
//! 2-tuple: worst gap 15.4ms over a 327-358ms full drain at 12 MiB (0.04-0.05),
//! inside the suite's shared budgets, where the list shape measured
//! 428-567ms. The QC-Yes-but-scan-dirty corpora keep their marshalling band
//! but lose the NFC pass from the wall (prose `finalize` 12 MiB:
//! ~110-147ms -> 42-46ms), which is why the 12 MiB prose GIL cells carry a
//! recalibrated 0.60 ratio budget derived from the new band (the residue did
//! not grow; the denominator shrank).
//!
//! The v0.6 surface (`diff_opcodes`) adds its own marshalling class, the
//! `word_bounds` list-shape family applied to a function whose output is a
//! list by necessity: the return is ONE 5-TUPLE PER OPCODE, constructed under
//! the GIL, O(ops), with up to four fresh `PyLong`s each and the four tag
//! strings (`"equal"`/`"replace"`/`"delete"`/`"insert"`) built once per call
//! and shared by reference into every tuple (difflib's own interned-tag
//! shape; constructing a `PyString` per op would multiply the band
//! several-fold). Measured at 12 MiB (ambient load 5.3-6.6, pinned by
//! `tests/test_gil_release.py`): the near-identical pair (235 opcodes) sits
//! at the ping floor, worst gaps 10.5-11.2ms of 76-88ms walls, while the
//! shuffled pair (103,421 opcodes) shows the band itself: worst gaps
//! 20.4-26.3ms of 1554-1626ms walls, the ~10-15ms delta over the floor being
//! the tuple construction, ~0.1-0.15µs per opcode. The argument side pays the
//! standard str-in classes TWICE (one borrow per operand: zero-copy for
//! ASCII/cached inputs, the one-time O(input) UTF-8 materialization on the
//! first non-ASCII call), and the whole diff (both operands' `Vec<char>`
//! materialization and the Myers search) runs under `py.detach`. The
//! `deadline_ms` parameter (v0.6.1) changes none of that: the budget is
//! checked in plain Rust inside the detached pass, and the `TimeoutError`
//! on expiry is constructed AFTER the GIL is reacquired; nothing raises
//! from inside the detached region.
//!
//! The v0.7 surface (`find_patterns`) adds the same list-shape marshalling
//! class with a GIL-held ARGUMENT walk of its own: the pattern list is
//! walked once under the GIL, borrowing each entry's UTF-8 zero-copy (the
//! standard str-in borrow class, O(patterns) handles; the one-time O(input)
//! materialization applies per non-ASCII pattern object on first call), then
//! the WHOLE search (automaton build, scan, and the byte→char offset
//! conversion; an `is_ascii` fast path skips the conversion for ASCII text)
//! runs under one `py.detach`, and the return marshalling constructs one
//! 3-tuple of ints per match, O(matches), measured ~0.13µs per match. The
//! sparse 12 MiB cell (a terminology scan with one occurrence) sits at the
//! ping floor over a ~6ms scan, walls UNDER the 10ms ping floor, so that
//! cell is ceiling-only (the b64/utf8_is_valid precedent); while the dense
//! 17-word cell (1,284,724 matches over 12 MiB prose) shows the band itself:
//! worst gaps 171.0-174.7ms of 214-224ms walls (ratio 0.76-0.80: the
//! marshalling is structurally the dominant share of a search call at
//! whole-corpus match counts, because the search core is ~3.5x faster than
//! segmentation), pinned by a bespoke 400ms ceiling and a 0.90 ratio budget
//! in `tests/test_gil_release.py`; the README's Performance section records
//! the caller guidance (~13ms held at 100k matches, ~170ms at 1.28M).
//!
//! The v0.8 surface (`replace_many`, `sentence_bounds`/`sentence_bounds_iter`,
//! `diff_opcodes_lines`) adds NO new residue class. `replace_many` is the
//! `find_patterns` argument shape over a dict: one GIL-held walk borrowing
//! each key and value (the standard str-in borrow class, O(entries) handles),
//! then automaton build + scan + splice under one `py.detach`, then either
//! the identity return (no key matched, or the net effect is the identity:
//! the ORIGINAL object, zero marshalling) or the O(output) string
//! marshalling. `sentence_bounds` is the `word_bounds` pair exactly: the
//! detached core, then either the list API's O(sentences) tuple marshalling
//! (sentences are far sparser than words: measured 170,037 sentences
//! against 3.67M word segments over the same 12 MiB prose, ~1/22nd the
//! tuple count, so the list shape MEETS the shared GIL budgets where
//! `word_bounds` structurally cannot) or the streaming iterator's one-detach
//! buffer with a single 2-tuple per `__next__`. `diff_opcodes_lines` is the
//! `diff_opcodes` marshalling applied to LINE indices, O(line-opcodes),
//! a count far below its char-level twin's on the same corpus, with the
//! same `deadline_ms` machinery (the budget check in plain Rust inside the
//! detached pass; the `TimeoutError` constructed after the GIL is
//! reacquired; an enormous-but-finite budget saturates to unbounded instead
//! of panicking, per `diff_impl::budget_from_ms`).
//!
//! The v0.9 surface (`word_count`, `sentence_count`, `count_matches`,
//! `find_patterns_iter`) adds NO residue class at all: the three counts are
//! `grapheme_count`'s extreme point (a single int return, no marshalling
//! class, and the cores allocate nothing where the list APIs materialize
//! the full vector; counting is the common question, and O(1) memory is the
//! answer), and `find_patterns_iter` is the `word_bounds_iter` design over
//! matches: the whole search (walk of the pattern list, automaton build,
//! scan) under one `py.detach` at construction filling an internal buffer,
//! then one 3-tuple of ints per `__next__` (µs-scale), the streaming answer
//! to the O(matches) list-marshalling caveat the v0.7 section records.
//!
//! The percent-encoding, similarity, fuzzy-metric, and masked-replace
//! surfaces (`quote`/`quote_plus`/`unquote`/
//! `unquote_plus`, `similarity_ratio`/`get_close_matches`,
//! `levenshtein`/`jaro`/`jaro_winkler`, `replace_many_masked`) adds two
//! residue shapes, both reusing shapes established earlier in this file.
//! The percent-encoding quartet is
//! the `detached_transform` classes over `url_impl`'s `Cow` cores (whole
//! scan under `py.detach`; the identity lane returns the ORIGINAL object
//! when nothing encodes/decodes; O(output) marshalling otherwise); the
//! stdlib's own spellings are pure Python and GIL-held whole-text.
//! `similarity_ratio` is the diff classes (two str-in borrows, the Myers
//! search under `py.detach`, a single float out); `get_close_matches`
//! walks the candidate list under the GIL (the standard borrow class) and
//! returns REFERENCES to the original candidate str objects: zero
//! marshalling, difflib's own shape. The three fuzzy metrics are O(n·m)
//! DP/Jaro passes under `py.detach` with a per-row/phase deadline check
//! (the `deadline_ms` DoS discipline: the uninterruptible-`strsim` route
//! could not meet it, which is why the formulas live in this crate with
//! `strsim` as the dev-side differential oracle) and single int/float
//! returns. `replace_many_masked` is `replace_many`'s classes exactly (one
//! GIL-held dict walk, the scan+splice under one detach, one string out).
//!
//! The JSON repair surface (`repair_json`/`repair_json_loads`/
//! `repair_json_diagnostics`) runs the WHOLE repair detached — the strict
//! fast path, the repair parser, the schema alignment, and the validator
//! compile+check — so a malformed multi-megabyte model dump never holds
//! the GIL. The GIL-held residue is the `schema=` argument walk (O(schema)
//! handles; each dict/list entry pays the standard str-in borrow class)
//! and the return marshalling: the O(output) string for `repair_json`, the
//! O(result) object-tree construction for the loads/diagnostics spellings
//! (the `word_bounds` list-marshalling class), plus O(diagnostics) small
//! dicts for the diagnostics flavor.

pub mod b64_impl;
pub mod bm25_impl;
pub mod chunk_by_segment_impl;
pub mod chunk_hierarchical_impl;
pub mod chunk_impl;
pub mod controls_impl;
pub mod decode_impl;
pub mod diff_impl;
pub mod encoding_impl;
pub mod fence_impl;
pub mod finalize_impl;
pub mod forms_impl;
pub mod fuzzy_impl;
pub mod grounded_impl;
pub mod html_impl;
pub mod html_table;
pub mod json_repair;
pub mod json_schema_impl;
pub mod merkle_impl;
pub mod normalize_impl;
pub mod phonetic_impl;
pub mod pipeline_impl;
pub mod search_impl;
pub mod segmentation_impl;
pub mod simhash_impl;
pub mod tfidf_impl;
pub mod tokenize_impl;
pub mod truncate_impl;
pub mod url_impl;
pub mod utf16_impl;
pub mod utf8_impl;

use std::borrow::Cow;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::{Py, PyAny};

/// The str-in identity-return wrapper shape shared by `tors.normalize`, the
/// four standalone forms, and `tors.html_unescape` (v0.4): borrow the
/// argument's UTF-8 under the GIL (`to_str`: a zero-copy alias for
/// ASCII/cached inputs, the one-time O(input) materialization on the first
/// non-ASCII call), run the transform detached, and on the identity path
/// return the ORIGINAL `PyString` object, CPython's own
/// `unicode_result_unchanged` idiom (`unicodedata.normalize`'s
/// `is_normalized` fast path returns the input, `html.unescape`'s
/// no-`&` path returns the input), zero allocation, zero copy, zero
/// marshalling. Otherwise marshal the transformed output (O(output)).
pub(crate) fn detached_transform(
    py: Python<'_>,
    text: Bound<'_, PyString>,
    transform: impl for<'a> Fn(&'a str) -> Cow<'a, str> + Send + Sync,
) -> PyResult<Py<PyAny>> {
    let s = text.to_str()?;
    let (identity, out) = py.detach(|| match transform(s) {
        Cow::Owned(out) => (false, out),
        Cow::Borrowed(_) => (true, String::new()),
    });
    if identity {
        Ok(text.into_any().unbind())
    } else {
        Ok(out.into_pyobject(py)?.into_any().unbind())
    }
}

pub mod py;

use py::bm25::*;
use py::chunk::*;
use py::codec::*;
use py::compiled_patterns::CompiledPatterns;
use py::diff::*;
use py::encoding::*;
use py::fence::*;
use py::forms::*;
use py::fuzzy::*;
use py::grounded::*;
use py::html::*;
use py::json_repair::*;
use py::lemma_dict::CompiledLemmaDict;
use py::merkle::*;
use py::normalize::*;
use py::phonetic::*;
use py::pipeline::*;
use py::search::*;
use py::segmentation::*;
use py::simhash::*;
use py::tfidf::*;
use py::truncate::*;
use py::url::*;

/// The shared core of the three streaming iterators (`word_bounds_iter`,
/// `sentence_bounds_iter`, `find_patterns_iter`): the input kept alive for
/// the iterator's life, the full item sequence computed under ONE detach at
/// construction, and a cursor. Generic over the item (`Copy`: a 2-tuple of
/// bounds or a 3-tuple of a match) so the three `#[pyclass`] wrappers are
/// thin and a future `*_iter` surface is one pyclass away (the class NAME
/// is per-struct in pyo3; the logic is this one struct, DRY).
pub(crate) struct EagerIter<T> {
    /// The input, kept alive for the iterator's life (its cached UTF-8 view
    /// with it), held for the drop side effect, never read after
    /// construction; the items are computed once, up front.
    _text: Py<PyString>,
    /// The full item sequence, computed under one detach at construction.
    items: Vec<T>,
    /// The index of the next item to yield.
    cursor: usize,
}

impl<T: Copy> EagerIter<T> {
    pub(crate) fn new(py: Python<'_>, text: Bound<'_, PyString>, items: Vec<T>) -> Self {
        EagerIter {
            _text: text.as_unbound().clone_ref(py),
            items,
            cursor: 0,
        }
    }

    pub(crate) fn next(&mut self) -> Option<T> {
        let next = self.items.get(self.cursor).copied();
        if next.is_some() {
            self.cursor += 1;
        }
        next
    }

    pub(crate) fn remaining(&self) -> usize {
        self.items.len() - self.cursor
    }
}

/// The `deadline_ms` validation shared by both diff spellings: `None` (the
/// unbounded spelling) or a positive finite number of milliseconds. An
/// enormous-but-finite value is LEGAL: the core saturates any `Duration`
/// overflow to unbounded (`diff_impl::budget_from_ms`) instead of panicking
/// in `Duration::from_secs_f64`.
pub(crate) fn validate_deadline_ms(deadline_ms: Option<f64>) -> PyResult<()> {
    if let Some(ms) = deadline_ms
        && !(ms.is_finite() && ms > 0.0)
    {
        return Err(PyValueError::new_err(
            "deadline_ms must be a positive finite number of milliseconds",
        ));
    }
    Ok(())
}

/// The `boundary=` parameter's two accepted spellings for
/// `truncate_to_bounds`: the same closed-set-of-strings convention as
/// `errors=`/`unicodedata.normalize`'s form argument.
pub(crate) fn parse_boundary(boundary: &str) -> PyResult<truncate_impl::Boundary> {
    match boundary {
        "word" => Ok(truncate_impl::Boundary::Word),
        "sentence" => Ok(truncate_impl::Boundary::Sentence),
        _ => Err(PyValueError::new_err(format!(
            "boundary must be one of ('word', 'sentence'), not {boundary:?}"
        ))),
    }
}

#[pymodule]
fn _tors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(finalize, m)?)?;
    m.add_function(wrap_pyfunction!(nfc, m)?)?;
    m.add_function(wrap_pyfunction!(nfd, m)?)?;
    m.add_function(wrap_pyfunction!(nfkc, m)?)?;
    m.add_function(wrap_pyfunction!(nfkd, m)?)?;
    m.add_function(wrap_pyfunction!(html_unescape, m)?)?;
    m.add_function(wrap_pyfunction!(grapheme_count, m)?)?;
    m.add_function(wrap_pyfunction!(word_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(word_bounds_iter, m)?)?;
    m.add_function(wrap_pyfunction!(decode_utf8, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_utf8, m)?)?;
    m.add_function(wrap_pyfunction!(b64_encode_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(b64_decode, m)?)?;
    m.add_function(wrap_pyfunction!(utf8_is_valid, m)?)?;
    m.add_function(wrap_pyfunction!(decode_utf16, m)?)?;
    m.add_function(wrap_pyfunction!(utf16_is_valid, m)?)?;
    m.add_function(wrap_pyfunction!(detect_encoding, m)?)?;
    m.add_function(wrap_pyfunction!(diff_opcodes, m)?)?;
    m.add_function(wrap_pyfunction!(diff_opcodes_lines, m)?)?;
    m.add_function(wrap_pyfunction!(find_patterns, m)?)?;
    m.add_function(wrap_pyfunction!(find_patterns_iter, m)?)?;
    m.add_function(wrap_pyfunction!(count_matches, m)?)?;
    m.add_function(wrap_pyfunction!(replace_many, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_bounds_iter, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_count, m)?)?;
    m.add_function(wrap_pyfunction!(word_count, m)?)?;
    m.add_function(wrap_pyfunction!(extract_code_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(strip_code_fences, m)?)?;
    m.add_function(wrap_pyfunction!(dedent, m)?)?;
    m.add_function(wrap_pyfunction!(repair_json, m)?)?;
    m.add_function(wrap_pyfunction!(repair_json_loads, m)?)?;
    m.add_function(wrap_pyfunction!(repair_json_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(truncate_to_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(truncate_ellipsis, m)?)?;
    m.add_function(wrap_pyfunction!(strip_controls, m)?)?;
    m.add_function(wrap_pyfunction!(is_grounded, m)?)?;
    m.add_function(wrap_pyfunction!(merkle_root, m)?)?;
    m.add_function(wrap_pyfunction!(merkle_diff, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_cdc, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_text, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_text_iter, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_by_words, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_by_words_iter, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_by_sentences, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_by_sentences_iter, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_by_paragraphs, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_by_lines, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_by_lines_iter, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_hierarchical, m)?)?;
    m.add_function(wrap_pyfunction!(simhash64, m)?)?;
    m.add_function(wrap_pyfunction!(simhash128, m)?)?;
    m.add_function(wrap_pyfunction!(quote, m)?)?;
    m.add_function(wrap_pyfunction!(quote_plus, m)?)?;
    m.add_function(wrap_pyfunction!(unquote, m)?)?;
    m.add_function(wrap_pyfunction!(unquote_plus, m)?)?;
    m.add_function(wrap_pyfunction!(similarity_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(get_close_matches, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(jaro, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler, m)?)?;
    m.add_function(wrap_pyfunction!(replace_many_masked, m)?)?;
    m.add_function(wrap_pyfunction!(bm25_rank, m)?)?;
    m.add_function(wrap_pyfunction!(tf_idf, m)?)?;
    m.add_function(wrap_pyfunction!(apply_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(soundex, m)?)?;
    m.add_function(wrap_pyfunction!(metaphone, m)?)?;
    m.add_function(wrap_pyfunction!(double_metaphone, m)?)?;
    m.add_function(wrap_pyfunction!(nysiis, m)?)?;
    m.add_function(wrap_pyfunction!(daitch_mokotoff, m)?)?;
    m.add_function(wrap_pyfunction!(refined_soundex, m)?)?;
    m.add_class::<CompiledLemmaDict>()?;
    m.add_class::<CompiledPatterns>()?;
    Ok(())
}
