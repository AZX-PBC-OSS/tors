//! `tors.first_invalid_charset`: the batch codepoint-set validator binding,
//! the argument-walk + one-detach shape over a str sequence.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PySequence, PyString};

use crate::charset_impl;

/// The GIL-held items walk: [`crate::py::_borrow::borrow_str_list`]'s
/// collect-handles-then-borrow shape, taken over any `Sequence` (a list, a
/// tuple, any `collections.abc.Sequence` subclass — the `separators=`
/// argument's widened spelling) rather than exactly a `PyList`, because the
/// API's annotation is `Sequence[str]`. A bare `str` is refused up front
/// with its own `TypeError`: `str` satisfies the Sequence ABC, so the cast
/// alone would let one through, and it would silently validate the
/// caller's own characters one by one — a question nobody asked (the same
/// refusal pyo3's `Vec` extraction makes for `separators=`).
///
/// Soundness, `borrow_str_list`'s story exactly (each call site keeps a
/// one-line pointer here): the borrows `run` receives point into the str
/// objects' immutable UTF-8 buffers (for a non-ASCII object, the UTF-8
/// copy pyo3 caches on the object at first extraction), valid for as long
/// as the objects are referenced. The caller's sequence argument holds the
/// objects and this walk's handle vector re-pins that for the compiler,
/// so the borrows are readable everywhere inside `run`, including across a
/// `py.detach` it performs.
///
/// The walk validates the WHOLE sequence before `run` runs (every entry is
/// borrowed, so a non-`str` entry or a lone surrogate raises at the
/// boundary wherever it sits), the `find_patterns` pattern-walk discipline:
/// the scan's first-offender short-circuit is a scan property, never an
/// argument-validation one. No empty-entry refusal exists here (the
/// `EmptyPolicy::Allow` side of the shared walk's contract): an empty item
/// is simply an offender the scan reports.
fn borrow_str_sequence<R>(
    items: &Bound<'_, PyAny>,
    run: impl FnOnce(&[Bound<'_, PyAny>], &[&str]) -> PyResult<R>,
) -> PyResult<R> {
    if items.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(
            "items must be a sequence of str, not a bare str (pass [s] to validate one string)",
        ));
    }
    let seq = items.cast::<PySequence>()?;
    let handles: Vec<_> = seq.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    let mut borrowed: Vec<&str> = Vec::with_capacity(handles.len());
    for handle in &handles {
        borrowed.push(handle.extract::<&str>()?);
    }
    run(&handles, &borrowed)
}

/// `tors.first_invalid_charset(items, *, first=None, rest)`: the index into
/// `items` of the first item not built entirely from the caller's two
/// codepoint sets — `first` the set allowed at position 0, `rest` the set
/// allowed at every position after it (and at position 0 too when `first`
/// is `None`, the uniform spelling) — or `-1` when every item passes. The
/// identifier-style rules a caller like TaskQ spells with anchored
/// regexes (`\A[A-Za-z_][A-Za-z0-9_]*\Z` and kin), expressed as plain
/// caller-supplied data and checked for a whole batch in one pass. The
/// sets are data, not patterns: plain strings of permitted codepoints,
/// membership per codepoint (duplicates in a spelling harmless, order
/// irrelevant); ranges, escapes, and Unicode-category classes (`\w`,
/// which would need property tables) are out of scope by charter
/// (docs/design.md). An empty item is an offender, wherever it sits.
///
/// Batch-only by design, and honestly so: each regex call this replaces
/// costs 84-950 ns and a whole enqueue's validation cluster sits under a
/// single `py.detach` round trip, so per-item calls through tors would be
/// SLOWER than the regexes they replace — the only winnable shape is the
/// batch (one detach, one pass over all items, the first offender
/// short-circuits), at the hundreds-of-items sizes a bulk pre-flight or
/// tag batch reaches. The scan makes no promise about work done past the
/// first offender, though the argument walk below does traverse the whole
/// list (a bad entry anywhere raises at the boundary, past a first
/// offender or not).
///
/// Argument contract, each decision pinned in
/// tests/test_first_invalid_charset.py: `items` is a sequence of `str` —
/// a `list`, a `tuple`, or any `Sequence` (a bare `str` raises
/// `TypeError`, as do non-sequences: dict, set, generators, scalars; a
/// non-`str` entry raises `TypeError`); `first`/`rest` must be exactly
/// `str`, both keyword-only, `rest` required, `first` defaulting to
/// `None`; lone surrogates raise `UnicodeEncodeError` at the standard
/// str-in boundary, paid by every item and both set arguments.
///
/// GIL model: one GIL-held sequence walk borrowing each entry's UTF-8
/// (the standard str-in class, O(items) handles; the one-time O(input)
/// materialization applies per non-ASCII item object on first call, and
/// to `first`/`rest` as usual), then set build + the whole batch scan
/// under one `py.detach`, then a single int return (the
/// `grapheme_count`/`count_matches` no-marshalling shape). At realistic
/// batch sizes the whole call sits far under the 10 ms ping floor, so the
/// GIL cell is ceiling-only (the `utf8_is_valid` class), pinned in
/// tests/test_gil_release.py; the criterion ladder for this core is the
/// `first_invalid_charset` group in benches/search.rs.
#[pyfunction(signature = (items, *, first = None, rest))]
pub fn first_invalid_charset(
    py: Python<'_>,
    items: &Bound<'_, PyAny>,
    first: Option<&str>,
    rest: &str,
) -> PyResult<isize> {
    // The shared-sequence walk (the soundness story lives on
    // `borrow_str_sequence`), then one detached pass over the whole batch.
    borrow_str_sequence(items, |_handles, borrowed| {
        Ok(py.detach(|| charset_impl::first_invalid_charset(borrowed, first, rest)))
    })
}
