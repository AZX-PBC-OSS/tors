use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::scan_impl;

/// The empty-needle refusal shared by both spellings: an empty needle would
/// match at every position and has no parity meaning (there is no occurrence
/// to stand before), the same rationale as `find_patterns`' empty pattern.
/// Raised under the GIL, ahead of the detach — the call's only error path.
fn refuse_empty_needle(needle: &[u8]) -> PyResult<()> {
    if needle.is_empty() {
        return Err(PyValueError::new_err("empty needle"));
    }
    Ok(())
}

/// `tors.contains_unescaped(haystack, needle)`: is there an occurrence of
/// `needle` in `haystack` that is not itself escaped — the boolean spelling
/// of the escape-parity scan. An occurrence at byte offset `i` counts only
/// when the maximal run of `b"\\"` immediately before `i` has even length
/// (an empty run is even, so an occurrence at offset 0 counts); an odd run
/// means the run's backslash pairs escape each other and the leftover one
/// escapes the occurrence's first byte, so the occurrence is literal text.
/// See `src/scan_impl.rs` for the full contract, the JSON `\u0000`-vs-
/// `\\u0000` motivation (a real NUL is fatal in a PostgreSQL jsonb column,
/// the literal text is fine, and the two are byte-ambiguous), and the
/// amortized run-state machinery; the differential oracle and the golden
/// battery are tests/test_unescaped_scan.py.
///
/// No JSON knowledge lives in the function: parity is the mechanism, and
/// "the needle is an escape sequence" is the caller's reading of it — any
/// backslash-escaped grammar can drive the same scan.
///
/// Argument contract: both arguments exactly `bytes` (`bytearray` /
/// `memoryview` / `str` raise `TypeError`, the bytes-in surface's
/// zero-copy immutable-borrow contract — and a `str` haystack would be a
/// different coordinate system entirely); an empty needle raises
/// `ValueError("empty needle")` before any scanning runs.
///
/// GIL model: `utf8_is_valid`'s extreme point exactly. The two `&[u8]`
/// extractions are zero-copy `PyBytes` borrows, the whole scan (the memmem
/// occurrence loop and the parity walk) runs under one `py.detach`, and
/// the `bool` return has no marshalling class at all, so the argument
/// borrows alone are the call's GIL-held residue — ceiling-only heartbeat
/// budget, pinned by tests/test_gil_release.py. The empty-needle
/// `ValueError` above is the only error path, and it fires under the GIL
/// before the detach: nothing raises from inside the detached region.
#[pyfunction]
pub fn contains_unescaped(py: Python<'_>, haystack: &[u8], needle: &[u8]) -> PyResult<bool> {
    refuse_empty_needle(needle)?;
    Ok(py.detach(|| scan_impl::find_unescaped(haystack, needle).is_some()))
}

/// `tors.find_unescaped(haystack, needle)`: the byte offset of the first
/// unescaped (even-run) occurrence of `needle` in `haystack`, or `-1` when
/// none exists — `bytes.find`'s own sentinel, kept over `Optional[int]`
/// deliberately: it is the spelling stdlib callers already branch on, and
/// it keeps the return a bare `int`. Everything else is
/// `contains_unescaped`'s contract exactly: the parity rule, the bytes-only
/// argument boundary, the `ValueError("empty needle")` refusal, the
/// no-JSON-knowledge scope, and the `utf8_is_valid` GIL class (one detach
/// around the whole scan; a single-int return, no marshalling class).
///
/// **The offsets are byte offsets, not `str` indices**: the return indexes
/// the `bytes` argument it was handed, so
/// `haystack[i:i + len(needle)] == needle` for every answer that is not
/// `-1` — over multibyte UTF-8 content the byte offset and the decoded
/// text's character offset are different numbers (the byte-offset battery
/// in tests/test_unescaped_scan.py pins the divergence numerically), the
/// same class of confusion `find_patterns` solved with its byte→char
/// mapping, deliberately not solved here because the input is bytes and
/// the contract is byte-space end to end.
///
/// Rejected (odd-run) hits advance the scan one byte past the hit, not
/// past the whole match, so self-overlapping needles stay correct: the
/// two-byte needle `b"00"` in a haystack of one backslash then `000`
/// rejects the hit at 1 and finds the overlapping live hit at 2.
#[pyfunction]
pub fn find_unescaped(py: Python<'_>, haystack: &[u8], needle: &[u8]) -> PyResult<isize> {
    refuse_empty_needle(needle)?;
    Ok(py
        .detach(|| scan_impl::find_unescaped(haystack, needle))
        .map_or(-1, |hit| hit as isize))
}
