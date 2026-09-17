//! The pyo3 binding for `tors.json_is_valid` (#61): the RFC 8259
//! validity gate over `crate::json_valid_impl`'s iterative scanner. See
//! that module for the orjson-matched acceptance set, the depth cap, and
//! the float-overflow gate; see docs/api.md for the consumer-facing
//! contract.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyString};

use crate::json_valid_impl;

/// `tors.json_is_valid(data)`: would a JSON parser accept `data` as one
/// complete document — one `bool`, no object tree, no exceptions on
/// invalid input. Built for the validate-and-discard gate: bytes that
/// are parsed once and thrown away, where ~95% of a full parse
/// (`orjson.loads` measured) is constructing objects nobody reads, so a
/// validity-only scan over the same bytes is a 4-5x cheaper pass at
/// these sizes (64 KiB ~40 µs, 1 MiB ~0.7 ms; issue #61's prototype
/// table). The acceptance set is orjson 3.x's, differentially probed and
/// documented (float-overflow literals reject; lone `\uD800`-class
/// surrogate escapes reject; depth cap 1024; NaN/Infinity/BOM reject;
/// duplicate keys accept) — `True` from this call means the consumer's
/// `orjson.loads` will succeed, `False` means it would raise. The one
/// class this cannot promise byte-for-byte is knife-edge float literals
/// within a rounding step of ±1.8e308, where orjson's own float parser
/// is the reference and correctly-rounded parsing is the implementation
/// (zero disagreements over the differential corpus; see
/// docs/api.md and the scanner module's docs). A validity gate is a
/// boolean question: nothing raises for invalid input — `DepthExceeded`
/// is an answer (`False`), not an error — and the only error path is the
/// wrong-type refusal below.
///
/// The name is the maintainer's proposal in #61 (the issue title's
/// `is_valid_json` reading poorly as English); it mirrors the existing
/// `utf8_is_valid`/`utf16_is_valid` validity predicates.
///
/// Argument contract: `bytes` (zero-copy borrowed) or `str` (borrowed
/// through its UTF-8 view — see the GIL model below). `bytearray`,
/// `memoryview`, and everything else raise `TypeError` naming the two
/// accepted types: the bytes-in surface's immutable-borrow contract
/// (a `bytearray` could be mutated mid-scan from another thread under
/// the released GIL), now stated as a union because the motivating
/// caller holds one or the other and nothing in between.
///
/// GIL model: `utf8_is_valid`'s class, with one honest addition. The
/// whole scan runs under one `py.detach` and the `bool` return has no
/// marshalling class, so for a `bytes` argument the argument borrow
/// alone is the call's GIL-held residue — the family's extreme point,
/// valid and invalid input alike. A `str` argument pays the standard
/// str-in borrow first, under the GIL: a zero-copy alias when the
/// string is pure ASCII or its UTF-8 view is already cached (repeat
/// calls on the same object: O(1) borrow, then the detached scan), and
/// a one-time O(input) materialization+cache-fill on the first
/// non-ASCII call (encode-parity; CPython caches the view on the
/// object, `encode` reads it and never fills it). Encoding is NOT done
/// eagerly into a fresh `bytes` object — the borrow is the cached-view
/// path every str-in tors function uses — so a str caller scanning the
/// same object repeatedly pays the materialization once per object,
/// never once per call. The heartbeat pin for both lanes is
/// tests/test_json_is_valid.py (a 1 MiB scan must leave a co-resident
/// asyncio heartbeat ticking); no aio twin, matching
/// `utf8_is_valid`/`utf16_is_valid` (a sub-millisecond-to-few-ms call
/// needs no thread hop; see docs/async.md).
#[pyfunction]
pub fn json_is_valid(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<bool> {
    // The bytes/str union, borrowed zero-copy either way: `bytes`'
    // immutable buffer, or `str`'s (lazily materialized, CPython-cached)
    // UTF-8 view via the standard str-in borrow. Both borrows happen
    // under the GIL, ahead of the detach — the str path's first-call
    // materialization is the one O(input) GIL-held segment, the same
    // class every str-argument function pays.
    if let Ok(bytes) = data.cast::<PyBytes>() {
        let raw = bytes.as_bytes();
        return Ok(py.detach(|| json_valid_impl::is_valid(raw)));
    }
    if let Ok(text) = data.cast::<PyString>() {
        let view = text.to_str()?;
        return Ok(py.detach(|| json_valid_impl::is_valid(view.as_bytes())));
    }
    Err(PyTypeError::new_err(format!(
        "json_is_valid argument must be bytes or str, not {}",
        data.get_type().name()?
    )))
}
