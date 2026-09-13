use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

use crate::hash_impl;

/// The str|bytes argument contract shared by all five hashing wrappers,
/// in the `_borrow.rs` run-closure shape (the borrows pyo3's extraction
/// hands back are tied to the handle they came from, so a helper that
/// returned them would be returning a value referencing data it also
/// owns — running `run` inside the scope that owns the handles solves
/// that and keeps them alive across any `py.detach` it performs, by
/// construction; the same soundness story every walk in `_borrow.rs`
/// carries).
///
/// The contract itself: a `str` is taken as its UTF-8 bytes (the
/// crate-wide str-in convention: `tors.sha256_hex(s) ==
/// hashlib.sha256(s.encode("utf-8")).hexdigest()`, the convenience
/// `hashlib` deliberately refuses — it raises TypeError on str —
/// provided here on purpose and documented loudly), and the bytes side is
/// the exactly-`bytes` doctrine of the bytes-in family
/// (tests/test_b64.py::TestBytesOnlyArgumentContract): a `bytearray` or
/// `memoryview` is a TypeError rather than a silent copy, because the
/// GIL-released digest reads the buffer without the GIL held.
///
/// The str borrow is pyo3's `to_str` (the standard str-in class: a
/// zero-copy alias for ASCII/cached inputs, the one-time O(input) UTF-8
/// materialization on the first non-ASCII call), which refuses a str
/// holding lone surrogates with `UnicodeEncodeError` ("surrogates not
/// allowed"), the same contract tests/test_finalize.py::
/// TestSurrogateBehavior pins crate-wide; any other borrow failure
/// propagates unchanged. The bytes borrow is the zero-copy immutable
/// `PyBytes` slice (pyo3's `&[u8]` extraction is `cast::<PyBytes>()?
/// .as_bytes()`, exactly this cast).
fn with_str_or_bytes<R>(
    name: &str,
    arg: &Bound<'_, PyAny>,
    run: impl FnOnce(&[u8]) -> PyResult<R>,
) -> PyResult<R> {
    if let Ok(text) = arg.cast::<PyString>() {
        let s = text.to_str()?;
        run(s.as_bytes())
    } else if let Ok(bytes) = arg.cast::<PyBytes>() {
        run(bytes.as_bytes())
    } else {
        Err(PyTypeError::new_err(format!(
            "{name} must be str or bytes, not {}",
            arg.get_type()
        )))
    }
}

/// `tors.md5_hex(data: str | bytes) -> str`: MD5 as 32 lowercase hex
/// chars, the whole digest under one `py.detach`. **Checksum/ETag/
/// legacy-interop only, never security**: md5 has had practical collisions
/// since 2004 and must not be used for signatures, certificates, or
/// passwords (Content-MD5, S3 ETags, cache-busting, quick compares are
/// its remaining jobs). str input is its UTF-8 bytes; bytes input is
/// exactly `bytes` (`bytearray`/`memoryview` raise TypeError, the
/// bytes-in family's immutable-buffer doctrine); a lone surrogate raises
/// UnicodeEncodeError at the borrow.
///
/// GIL model: borrow under the GIL; digest + hex formatting under one
/// `py.detach`; the O(32) hex string marshalled after.
#[pyfunction]
pub fn md5_hex(py: Python<'_>, data: Bound<'_, PyAny>) -> PyResult<String> {
    with_str_or_bytes("data", &data, |bytes| {
        Ok(py.detach(|| hash_impl::md5_hex(bytes)))
    })
}

/// `tors.sha1_hex(data: str | bytes) -> str`: SHA-1 as 40 lowercase hex
/// chars, the whole digest under one `py.detach`. **Checksum/
/// legacy-interop only, never security**: sha1's first practical collision
/// was published in 2017 (Google's SHAttered); like md5 it must not be
/// used for signatures, certificates, or passwords. Argument contract
/// identical to `md5_hex`'s.
///
/// GIL model: identical to `md5_hex`'s.
#[pyfunction]
pub fn sha1_hex(py: Python<'_>, data: Bound<'_, PyAny>) -> PyResult<String> {
    with_str_or_bytes("data", &data, |bytes| {
        Ok(py.detach(|| hash_impl::sha1_hex(bytes)))
    })
}

/// `tors.sha256_hex(data: str | bytes) -> str`: SHA-256 as 64 lowercase
/// hex chars, the whole digest under one `py.detach` — the same engine
/// `finalize`'s hash tail and `merkle_root`'s leaves use, byte-identical
/// to `hashlib.sha256(...).hexdigest()`, without the normalize stage.
/// The dedup-cache-key / content-addressing primitive. Argument contract
/// identical to `md5_hex`'s.
///
/// GIL model: identical to `md5_hex`'s (O(64) marshalling).
#[pyfunction]
pub fn sha256_hex(py: Python<'_>, data: Bound<'_, PyAny>) -> PyResult<String> {
    with_str_or_bytes("data", &data, |bytes| {
        Ok(py.detach(|| hash_impl::sha256_hex(bytes)))
    })
}

/// `tors.sha512_hex(data: str | bytes) -> str`: SHA-512 as 128 lowercase
/// hex chars, the whole digest under one `py.detach`. Argument contract
/// identical to `md5_hex`'s.
///
/// GIL model: identical to `md5_hex`'s (O(128) marshalling).
#[pyfunction]
pub fn sha512_hex(py: Python<'_>, data: Bound<'_, PyAny>) -> PyResult<String> {
    with_str_or_bytes("data", &data, |bytes| {
        Ok(py.detach(|| hash_impl::sha512_hex(bytes)))
    })
}

/// `tors.hmac_sha256_hex(key: str | bytes, data: str | bytes) -> str`:
/// HMAC-SHA-256 as 64 lowercase hex chars, the request-signing primitive
/// (webhook signature verification, AWS SigV4-style HMAC chains, API
/// auth), byte-identical to
/// `hmac.new(key, data, hashlib.sha256).hexdigest()`. Each argument gets
/// `md5_hex`'s str|bytes contract independently (a str key is its UTF-8
/// bytes, the spelling a webhook secret arrives in); any key length is
/// legal, empty included (parity with the stdlib spelling: HMAC pads
/// short keys and hashes long ones). The key is borrowed and validated
/// before the data.
///
/// GIL model: both borrows under the GIL; the whole keyed digest (key
/// derivation included) plus hex formatting under one `py.detach`.
#[pyfunction]
pub fn hmac_sha256_hex(
    py: Python<'_>,
    key: Bound<'_, PyAny>,
    data: Bound<'_, PyAny>,
) -> PyResult<String> {
    with_str_or_bytes("key", &key, |key_bytes| {
        with_str_or_bytes("data", &data, |data_bytes| {
            Ok(py.detach(|| hash_impl::hmac_sha256_hex(key_bytes, data_bytes)))
        })
    })
}
