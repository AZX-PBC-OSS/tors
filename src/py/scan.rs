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
/// parity walk's cost story; the differential oracle and the golden
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
        .map_or(-1, |hit| {
            isize::try_from(hit).expect("haystack offset fits in isize")
        }))
}

/// `tors.utf8_byte_len(s)`: the UTF-8 byte length of `s` — the answer
/// `len(s.encode("utf-8"))` computes by allocating and copying the whole
/// `bytes` object first, taken here without the copy. The count a caller
/// wants when a size cap sits in front of a store: TaskQ's
/// idempotency-key and scope byte caps on every enqueue, and the
/// terminal's re-encode of a serialized result of up to 64 KiB
/// (`MAX_RESULT_BYTES`) on every success — a genuine double pass, the
/// byte count having existed inside the serializer's output and been
/// discarded by the `.decode()` that produced the `str`.
///
/// **Companion, not standalone**: this ships in the scan family's binding
/// module as the pinned companion of `contains_unescaped`/
/// `find_unescaped` (#50) — same module, same harness patterns — and
/// honest sizing says it would not stand alone: a short-string encode is
/// a few hundred nanoseconds, so the win is large inputs and hot paths,
/// where the copy is the cost (the 64 KiB terminal case is ~1.5 µs of
/// pure memcpy per success).
///
/// Cost model (the deliberate deviation from the issue's sketch: no
/// hand-rolled UCS1/UCS2/UCS4 arithmetic — the module docs in
/// `src/scan_impl.rs` carry the full rationale — the standard str-in
/// borrow instead, and the core is the borrowed `&str`'s `len()`, one
/// field read):
///
/// * ASCII (serialized JSON with `ensure_ascii=True`): compact ASCII data
///   is its own UTF-8, so the borrow is a zero-copy alias and the call is
///   O(1), no allocation at all.
/// * Non-ASCII, first call on the object: CPython materializes and CACHES
///   the UTF-8 view on the `str` object (an internal cache, not a
///   Python-visible `bytes`; shared with every other str-in tors call on
///   the same object), so the first call is O(n) — encode-parity in cost
///   class, with no Python-visible object to allocate and collect.
/// * Non-ASCII, repeat calls on the same object: O(1) — strictly better
///   than the expression, which re-copies on every call.
///
/// Error parity: a `str` holding lone surrogates cannot be UTF-8-encoded,
/// and the borrow raises CPython's own `UnicodeEncodeError` (pyo3
/// propagates it from the `&str` extraction, before any tors code runs) —
/// the same exception `encode` raises, attributes included; there is no
/// tors-side error path at all. Pinned attribute-for-attribute in
/// tests/test_utf8_byte_len.py.
///
/// GIL model: `grapheme_count`'s marshalling class (a single `int`
/// return, nothing else held past the borrow), with one honest
/// difference: the call's only O(n) work IS the borrow — the first
/// non-ASCII call's materialization runs under the GIL (the standard
/// str-in first-call class every str-argument tors function pays; there
/// is no way to borrow the view without it), and the `py.detach` around
/// the core is nominal (an O(1) field read), kept for the module's family
/// shape. The heartbeat cell in tests/test_gil_release.py pins the band:
/// the 12 MiB non-ASCII first call's materialization sits under the 10 ms
/// ping floor, so the cell is ceiling-only like every sub-floor member.
/// No aio twin: an O(1)-to-O(n)-borrow call needs no thread hop.
#[pyfunction]
pub fn utf8_byte_len(py: Python<'_>, s: &str) -> usize {
    py.detach(|| scan_impl::utf8_byte_len(s))
}
