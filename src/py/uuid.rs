//! `tors.uuid7_timestamp_ms`, `tors.uuid_version`, `tors.uuid_parse`:
//! the pyo3 bindings of the UUIDv7 helper trio (issue #54), the pure-Rust
//! cores in `crate::uuid_impl`.
//!
//! # GIL shape (the honest sizing, pinned)
//!
//! The issue's stated shape is followed exactly: the str spellings
//! validate and transcode under the GIL *before* the detach, and the bit
//! extraction runs under `py.detach`. At 16 bytes of input that detach is
//! overhead for its own sake -- the extraction it wraps is a handful of
//! nanoseconds, far under the call machinery's own GIL-held time (the
//! argument borrow and the int marshalling dwarf it) -- so a
//! measurement-motivated design would hold everything. The detach is kept
//! anyway, for contract uniformity: every tors int-out primitive runs its
//! core work under `py.detach`, and the GIL-heartbeat suite
//! (tests/test_gil_release.py's uuid cell) pins the trio's per-call
//! GIL-held residue at sub-microsecond scale either way. What that cell
//! honestly cannot catch is a lost detach here (holding ~ns of extraction
//! is invisible); what it can catch is a per-call residue regression into
//! the tens-of-ms class.
//!
//! `uuid_parse` is the trio's zero-detach member, by the same reasoning
//! taken to its conclusion: its whole work IS the str spelling's
//! validate-and-transcode (there is no int-out tail to detach), and the
//! return marshalling (the `PyBytes` construction) needs the GIL anyway,
//! so a detach around the 36-byte parse would wrap nothing measurable.
//! Its entire GIL-held cost is that parse, ~70ns a call (re-measured
//! after the `uuid`-crate adoption: the crate's const-fn parser +
//! canonical-encode-compare is faster than the hand-rolled scan it
//! replaced; pinned by the same heartbeat cell, which batches it like
//! the others).
//!
//! # Argument contract
//!
//! The int-out pair takes exactly `bytes` or `str` (the `resolve_lemma_dict`
//! dispatch shape: `cast` each, a hand-rolled `TypeError` naming the
//! received type otherwise). `bytes` follows the bytes-in surface's
//! exactly-`bytes` convention (test_b64.py's TestBytesOnlyArgumentContract
//! doctrine): `bytearray`/`memoryview` are `TypeError`, not silently
//! accepted -- the borrow is a zero-copy alias of the immutable buffer,
//! and writable views would race the detached read. A wrong-length `bytes`
//! is `ValueError` naming the count (the fixed-width analogue of
//! `mask must be exactly one character`). A `str` holding lone surrogates
//! fails the UTF-8 borrow itself (`UnicodeEncodeError`, the chunk_text_iter
//! precedent) before any grammar check runs.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

use crate::uuid_impl;

/// Resolve the int-out pair's `bytes | str` union into the 16 raw bytes,
/// under the GIL: the str spelling validates and transcodes here (the
/// issue's shape: the whole input is 36 bytes, smaller than the call's own
/// marshalling residue, so a detached parse would be overhead for its own
/// sake), the bytes spelling is the zero-copy borrow plus the fixed-width
/// length check. Anything else is `TypeError` naming what was received.
fn uuid16_from_value(value: &Bound<'_, PyAny>) -> PyResult<[u8; 16]> {
    if let Ok(text) = value.cast::<PyString>() {
        let text = text.to_str()?;
        return uuid_impl::parse_canonical(text)
            .map_err(|err| PyValueError::new_err(err.message()));
    }
    if let Ok(bytes) = value.cast::<PyBytes>() {
        let raw = bytes.as_bytes();
        return raw.try_into().map_err(|_| {
            PyValueError::new_err(format!(
                "value must be exactly 16 bytes for a UUID, got {}",
                raw.len()
            ))
        });
    }
    Err(PyTypeError::new_err(format!(
        "value must be bytes or str, not {}",
        value.get_type().name()?
    )))
}

/// `tors.uuid_version(value: bytes | str) -> int`: the version nibble
/// (byte 6's high half) of any UUID, 0-15, one detached pass over the 16
/// bytes. The version field itself, answered regardless of variant: the
/// variant (byte 8's top two bits) is a different field and out of scope
/// -- `uuid_version` is not an RFC-4122-ness check. `bytes` must be
/// exactly 16 (ValueError otherwise); `str` must be canonical
/// 8-4-4-4-12 lowercase-hyphenated text (see `uuid_parse` for the strict
/// grammar and its deliberate divergences from the stdlib `uuid` module).
///
/// GIL model: the argument dispatch (borrow or validate-and-transcode, the
/// module docs' pinned shape) is GIL-held; the nibble read runs under
/// `py.detach`; a single small int out.
#[pyfunction]
pub fn uuid_version(py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<u8> {
    let bytes = uuid16_from_value(&value)?;
    Ok(py.detach(|| uuid_impl::version_nibble(&bytes)))
}

/// `tors.uuid7_timestamp_ms(value: bytes | str) -> int`: the 48-bit
/// big-endian unix-millisecond field (the leading six bytes) of a UUIDv7,
/// the keyset-pagination / time-bucketed-query primitive over
/// time-ordered IDs: `datetime.datetime.fromtimestamp(ms / 1000, UTC)` is
/// the ID's creation instant. `ValueError` naming the version actually
/// found when the version nibble is not 7 (the field is only defined for
/// v7; the nil UUID's all-zero field must not silently answer 0).
/// Argument contract matches `uuid_version` exactly.
///
/// GIL model: `uuid_version`'s (dispatch GIL-held per the module docs'
/// pinned shape, the field read under `py.detach`, one int out; the
/// version-guard `ValueError` is constructed after the GIL is reacquired).
#[pyfunction]
pub fn uuid7_timestamp_ms(py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<u64> {
    let bytes = uuid16_from_value(&value)?;
    py.detach(|| uuid_impl::v7_timestamp_ms(&bytes))
        .map_err(|found| {
            PyValueError::new_err(format!(
                "value must be a version 7 UUID, got version {found}"
            ))
        })
}

/// `tors.uuid_parse(value: str) -> bytes`: canonical UUID text to the 16
/// raw bytes, strict: exactly 36 characters, hyphens at positions
/// 8/13/18/23 (the 8-4-4-4-12 groups), lowercase hexadecimal everywhere
/// else, and `ValueError` naming the problem and the accepted form on
/// anything else (positions in the messages are 0-based). The stdlib
/// `uuid.UUID` also accepts braced text, the `urn:uuid:` prefix,
/// hyphen-less hex, and uppercase; tors deliberately does not -- the
/// validation-primitive contract (a gate wants one grammar, the canonical
/// one every producer emits), the same closed-set strictness as every
/// other tors argument. Those divergences are pinned as pins-not-parity in
/// tests/test_uuid.py (each loose form proven stdlib-accepted and
/// tors-rejected in the same test). Exactly `str`: bytes in is a
/// `TypeError` (the identity is the caller's to spell), a lone-surrogate
/// str fails the borrow itself.
///
/// GIL model: the trio's zero-detach member (see the module docs): the
/// whole 36-byte validate-and-transcode runs GIL-held by design -- there
/// is no int-out tail to detach, and the `PyBytes` return marshalling
/// needs the GIL regardless.
#[pyfunction]
pub fn uuid_parse(py: Python<'_>, value: &str) -> PyResult<Py<PyBytes>> {
    let bytes =
        uuid_impl::parse_canonical(value).map_err(|err| PyValueError::new_err(err.message()))?;
    Ok(PyBytes::new(py, &bytes).unbind())
}
