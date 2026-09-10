use pyo3::Py;
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::EagerIter;
use crate::py::eager_iter_class;
use crate::segmentation_impl;

/// `tors.grapheme_count`: the number of UAX #29 extended grapheme clusters
/// (the stdlib has no segmenter at all, which is the gap this fills). Backed
/// by `unicode-segmentation` (Unicode 17.0.0 tables) and pinned by the
/// hand-derived table plus structural properties in
/// tests/test_segmentation.py; see `src/segmentation_impl.rs` for the rules
/// cited per case.
///
/// GIL model: whole scan under `py.detach`; the argument borrow is the
/// standard str-in zero-copy alias (or the one-time O(input) UTF-8
/// materialization for the first non-ASCII call); the return is a single
/// int: no list marshalling at all.
#[pyfunction]
pub fn grapheme_count(py: Python<'_>, text: &str) -> usize {
    py.detach(|| segmentation_impl::grapheme_count(text))
}

/// `tors.word_bounds`: the UAX #29 word-boundary segments as
/// `(start, end)` pairs in Python str index (codepoint) units, so
/// `text[start:end]` is the segment; bounds cover `[0, len(text))` and
/// joining the slices reproduces the input (pinned in
/// tests/test_segmentation.py). Offsets, never string lists: marshalling
/// thousands of small PyStrings under the GIL would eat the win.
///
/// GIL model: the segmentation runs under `py.detach`, but the return
/// marshalling constructs one 2-tuple of ints per segment under the GIL:
/// O(number-of-segments), a measured cost that dominates at whole-file
/// sizes; docs/performance.md records the measured band and the streaming/iterator
/// API shape proposed as a v0.4 question.
#[pyfunction]
pub fn word_bounds(py: Python<'_>, text: &str) -> Vec<(usize, usize)> {
    py.detach(|| segmentation_impl::word_bounds(text))
}

eager_iter_class! {
    /// The streaming spelling of `tors.word_bounds` (v0.4): a lazy iterator
    /// yielding the same `(start, end)` pairs, in the same order, as the list
    /// API, pinned to sequence-parity by tests/test_segmentation.py over every
    /// UAX #29 tricky row and hypothesis text.
    ///
    /// GIL model: this is the design that answers the v0.3 marshalling cost
    /// (428-497ms of GIL-held 3.67M-tuple construction at 12 MiB, see
    /// `word_bounds`' doc and docs/performance.md). The whole segmentation, the same
    /// detached core pass the list API runs, fills an internal bounds buffer
    /// under one `py.detach` when the iterator is constructed and stays GIL-free
    /// for its full duration (the buffer is 16 bytes per segment against the
    /// list shape's Python tuples). Each `__next__` then holds the GIL only to
    /// construct one 2-tuple of ints, µs-scale, so the worst heartbeat gap stays
    /// in the ping-floor band every str-in cell sits in (measured 15.4ms worst
    /// gap over a 327-358ms full-drain wall at 12 MiB, against the list shape's
    /// structurally-unattainable 428-567ms band).
    ///
    /// The full drain is ~2.1x faster than the list API at 12 MiB (347ms vs
    /// 724ms, min-of-3, same process: the per-`__next__` tuple path is cheaper
    /// per bound than the list-return conversion), so the iterator wins on both
    /// axes at whole-file sizes; the list API stays the right shape for small
    /// inputs and one-shot batch work.
    ///
    /// Eager-at-construction is a design requirement: UAX #29 word boundaries are not
    /// safely resumable mid-text (word-class runs span arbitrary distances, so
    /// a chunk-restart would mis-boundary at every cut that lands mid-word, and
    /// the crate exposes no safe-restart points), so the buffer is filled by one
    /// whole-text pass up front rather than per-batch re-segmentation.
    ///
    /// The iterator machinery itself is the shared [`EagerIter`] core (below).
    /// `sentence_bounds_iter` (v0.8) and `find_patterns_iter` (v0.9) use the
    /// same design over UAX #29 sentence boundaries and search matches; the
    /// eager-at-construction argument holds identically for both (SB6/SB7/SB8
    /// lookahead classes and leftmost-longest resume-at-match-end state span
    /// arbitrary distances, so per-batch restarts would mis-segment).
    WordBoundsIter, (usize, usize);
}

eager_iter_class! {
    /// The streaming sentence segmentation (v0.8): the same iterator design and
    /// GIL model as [`WordBoundsIter`] over `sentence_bounds`, with the same
    /// sequence as the list API (pinned to sequence-parity), one detached
    /// whole-text pass at construction, one 2-tuple of ints per `__next__`.
    /// Sentence counts are far below word counts on the same text (measured
    /// ~1/22nd on 12 MiB prose), so the streaming shape matters less here than
    /// for `word_bounds`; it exists for symmetry and for whole-corpus sweeps
    /// that never materialize the list.
    SentenceBoundsIter, (usize, usize);
}

/// `tors.word_bounds_iter(text)`: the streaming segmentation surface; see
/// [`WordBoundsIter`]'s docs for the GIL model and the measured tradeoff
/// against the list API.
#[pyfunction]
pub fn word_bounds_iter(py: Python<'_>, text: Bound<'_, PyString>) -> PyResult<Py<WordBoundsIter>> {
    let s = text.to_str()?;
    let bounds = py.detach(|| segmentation_impl::word_bounds(s));
    Py::new(py, WordBoundsIter(EagerIter::new(py, text, bounds)))
}

/// `tors.sentence_bounds(text)`: the UAX #29 sentence-boundary segments as
/// `(start, end)` pairs in Python str index (codepoint) units, so
/// `text[start:end]` is the sentence; bounds cover `[0, len(text))` and
/// joining the slices reproduces the input. The stdlib has no sentence
/// segmenter either, the same gap `word_bounds` fills, over the same
/// `unicode-segmentation` tables (rules SB1-SB999), pinned the same way: a
/// hand-derived table of the tricky rows (each derived from the cited rule)
/// plus structural properties, in tests/test_sentence_bounds.py; see
/// `src/segmentation_impl.rs` for the rows cited per case. One presentation
/// quirk to know (UAX #29 itself, pinned): trailing spaces after a sentence
/// terminator belong to the preceding sentence, so segments may carry
/// trailing whitespace. Rule-based UAX #29 only, with no dictionary
/// segmentation for spaceless scripts (Thai/Khmer/Burmese/Japanese word
/// breaks are a different, dictionary-based problem; ICU4X is the
/// heavyweight alternative); a limitations note shared with `word_bounds`.
///
/// GIL model: the `word_bounds` pair exactly, whole segmentation under
/// `py.detach`. The list API's GIL-held residue is the O(sentences) tuple
/// marshalling (far sparser than words), and `sentence_bounds_iter` is the
/// streaming answer.
#[pyfunction]
pub fn sentence_bounds(py: Python<'_>, text: &str) -> Vec<(usize, usize)> {
    py.detach(|| segmentation_impl::sentence_bounds(text))
}

/// `tors.word_count(text)`: the number of UAX #29 word segments, as a
/// single int: the `grapheme_count` precedent for the second segmenter
/// (v0.9). Why a separate function: `len(word_bounds(text))` materializes
/// the full bounds list (millions of tuples at whole-file scale) to answer
/// a number: the count core runs the same single segmentation pass and
/// allocates nothing. The invariant `word_count(t) == len(word_bounds(t))`
/// is pinned in tests/test_segmentation.py.
///
/// GIL model: `grapheme_count`'s extreme point: whole scan under
/// `py.detach`, the standard str-in argument borrow, a single int return
/// (no marshalling class at all).
#[pyfunction]
pub fn word_count(py: Python<'_>, text: &str) -> usize {
    py.detach(|| segmentation_impl::word_count(text))
}

/// `tors.sentence_count(text)`: the number of UAX #29 sentences, as a
/// single int: the `grapheme_count` precedent for the third segmenter
/// (v0.9), the same O(1)-memory argument as `word_count`
/// (`len(sentence_bounds(text))` builds the tuple list to answer a number).
/// The invariant `sentence_count(t) == len(sentence_bounds(t))` is pinned
/// in tests/test_sentence_bounds.py.
///
/// GIL model: `grapheme_count`'s extreme point (single int, no marshalling
/// class, whole scan under `py.detach`).
#[pyfunction]
pub fn sentence_count(py: Python<'_>, text: &str) -> usize {
    py.detach(|| segmentation_impl::sentence_count(text))
}

/// `tors.sentence_bounds_iter(text)`: the streaming spelling of
/// `sentence_bounds` (v0.8); see [`SentenceBoundsIter`]'s docs for the
/// design and the GIL model (the `word_bounds_iter` precedent, shared
/// [`BoundsSeq`] core).
#[pyfunction]
pub fn sentence_bounds_iter(
    py: Python<'_>,
    text: Bound<'_, PyString>,
) -> PyResult<Py<SentenceBoundsIter>> {
    let s = text.to_str()?;
    let bounds = py.detach(|| segmentation_impl::sentence_bounds(s));
    Py::new(py, SentenceBoundsIter(EagerIter::new(py, text, bounds)))
}
