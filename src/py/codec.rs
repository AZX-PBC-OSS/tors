use std::borrow::Cow;

use pyo3::exceptions::{PyUnicodeDecodeError, PyUnicodeEncodeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};
use pyo3::{Py, PyAny};

use crate::{b64_impl, decode_impl, finalize_impl, utf8_impl, utf16_impl};

/// The `errors=` parameter's two accepted spellings, shared by `decode_utf8` and
/// `finalize_utf8`. Anything else is a `ValueError`, following the
/// closed-set-of-strings convention of `unicodedata.normalize` ("invalid
/// normalization form"), pinned in tests/test_decode_utf8.py::TestErrorsParameterContract.
enum ErrorsMode {
    Strict,
    Replace,
}

fn parse_errors_mode(errors: &str) -> PyResult<ErrorsMode> {
    match errors {
        "strict" => Ok(ErrorsMode::Strict),
        "replace" => Ok(ErrorsMode::Replace),
        _ => Err(PyValueError::new_err(format!(
            "errors must be one of ('strict', 'replace'), not {errors:?}"
        ))),
    }
}

/// Render a strict-decode failure as the same `UnicodeDecodeError` CPython's own
/// decoder raises: true type, `.encoding` "utf-8", `.object` the input bytes,
/// `.start`/`.end`/`.reason` the classified span (see `decode_impl::classify`'s
/// measured parity), so `str(exc)` matches too, since CPython's `__str__` formats
/// those fields. pyo3's `PyUnicodeDecodeError::new_err_from_utf8` is not used
/// because its fixed reason "invalid utf-8" diverges from CPython's per-shape
/// reasons on every error.
fn decode_error_into_pyerr(py: Python<'_>, raw: &[u8], err: decode_impl::DecodeError) -> PyErr {
    let reason = match err {
        decode_impl::DecodeError::InvalidStart { .. } => c"invalid start byte",
        decode_impl::DecodeError::InvalidContinuation { .. } => c"invalid continuation byte",
        decode_impl::DecodeError::UnexpectedEnd { .. } => c"unexpected end of data",
    };
    match PyUnicodeDecodeError::new(py, c"utf-8", raw, err.start()..err.end(), reason) {
        Ok(exception) => PyErr::from_value(exception.into_any()),
        Err(construction_error) => construction_error,
    }
}

/// `tors.decode_utf8`: `raw.decode("utf-8", errors=...)` byte-exact over arbitrary
/// bytes, as one GIL-released pass: strict raises the very `UnicodeDecodeError`
/// CPython's decoder raises (type, spans, reason, message), replace emits
/// CPython's exact U+FFFD placements. The zero-copy `PyBytes` borrow and the
/// return marshalling are the call's only GIL-held residue; see the crate GIL
/// model above. Valid input marshals straight from the decode result's borrow of
/// the argument (no intermediate owned copy); the return type is spelled
/// `Py<PyAny>` rather than `Bound<'py, PyString>` because pyo3 0.29's
/// pyfunction-return machinery is ambiguous for `PyResult<Bound<'py, T>>` with
/// an explicit `'py` parameter.
#[pyfunction(signature = (raw, *, errors = "strict"))]
pub fn decode_utf8(py: Python<'_>, raw: &[u8], errors: &str) -> PyResult<Py<PyAny>> {
    let decoded = match parse_errors_mode(errors)? {
        ErrorsMode::Strict => py
            .detach(|| decode_impl::decode_strict(raw))
            .map_err(|err| decode_error_into_pyerr(py, raw, err))?,
        ErrorsMode::Replace => py.detach(|| decode_impl::decode_replace(raw)),
    };
    Ok(decoded.into_pyobject(py)?.into_any().unbind())
}

/// The marshalling tail shared by both `finalize_utf8` flavors: run
/// `finalize_checked` over the decoded text detached; on the identity path
/// the output string marshals straight from the borrowed decode of the input
/// bytes (no owned intermediate, no output allocation in the Rust layer).
fn finalize_utf8_tail(py: Python<'_>, decoded: Cow<'_, str>) -> PyResult<(Py<PyAny>, String)> {
    let s = decoded.as_ref();
    let (identity, out, digest) = py.detach(|| match finalize_impl::finalize_checked(s) {
        (Cow::Borrowed(_), digest) => (true, String::new(), digest),
        (Cow::Owned(out), digest) => (false, out, digest),
    });
    let normalized = if identity {
        PyString::new(py, s).into_any().unbind()
    } else {
        out.into_pyobject(py)?.into_any().unbind()
    };
    Ok((normalized, digest))
}

/// `tors.finalize_utf8`: decode + normalize + hash in one GIL-released call:
/// `(normalize(raw.decode("utf-8", errors=...)), sha256-hex-of-the-result)`, the
/// one-call shape an extraction pipeline wants for its text reads. Strict
/// (default) raises the stdlib's own `UnicodeDecodeError` on invalid bytes;
/// `errors="replace"` flows the U+FFFD substitutions through the pipeline. Same
/// GIL model as `decode_utf8` (bytes in: no argument materialization at all).
///
/// v0.4: valid input decodes to a borrowed view of the argument bytes, and an
/// identity pipeline (quick-check Yes + clean scan) returns that borrow with
/// no output allocation. The string element marshals straight from it and
/// the digest is computed from it in the same detached pass.
#[pyfunction(signature = (raw, *, errors = "strict"))]
pub fn finalize_utf8(py: Python<'_>, raw: &[u8], errors: &str) -> PyResult<(Py<PyAny>, String)> {
    match parse_errors_mode(errors)? {
        ErrorsMode::Strict => {
            let decoded = py
                .detach(|| decode_impl::decode_strict(raw))
                .map_err(|err| decode_error_into_pyerr(py, raw, err))?;
            finalize_utf8_tail(py, decoded)
        }
        ErrorsMode::Replace => {
            let decoded = py.detach(|| decode_impl::decode_replace(raw));
            finalize_utf8_tail(py, decoded)
        }
    }
}

/// `tors.b64_encode_bytes`: `base64.b64encode(raw).decode("ascii")` (RFC 4648
/// standard alphabet, padded) in one GIL-released pass; the argument borrow is
/// the same zero-copy `PyBytes` slice and the only GIL-held residue is the
/// marshalling of the (4/3-sized, ASCII) output string. See the crate GIL model
/// above.
#[pyfunction]
pub fn b64_encode_bytes(py: Python<'_>, raw: &[u8]) -> String {
    py.detach(|| b64_impl::encode(raw))
}

/// Raise the real `binascii.Error`, the exact class `base64.b64decode` raises,
/// not a stand-in. pyo3 0.29 has no typed exception for it (`binascii.Error`
/// is a module-level C exception outside the `PyExc_*` table pyo3's
/// `exceptions.rs` is generated from: verified in pyo3 0.29.2's source), but
/// `PyErr::from_value` accepts any exception *instance*: import `binascii`,
/// call `Error(message)`. The result satisfies `except binascii.Error` and
/// `except ValueError` alike (the subclass relation the parity contract rests
/// on), and `type(exc) is binascii.Error`. The import runs at raise time only,
/// under the GIL, against a stdlib builtin: negligible on the error path.
fn binascii_error(py: Python<'_>, message: String) -> PyErr {
    let instance = PyModule::import(py, "binascii")
        .and_then(|binascii| binascii.getattr("Error"))
        .and_then(|error| error.call1((message,)));
    match instance {
        Ok(exception) => PyErr::from_value(exception),
        // binascii is a stdlib builtin; failing to import it means the
        // interpreter itself is broken: surface that, don't mask it.
        Err(import_error) => import_error,
    }
}

/// `tors.b64_decode`: `base64.b64decode(s, validate=...)` over ASCII strings
/// in one GIL-released pass: same decoded bytes, same raised
/// `binascii.Error` (the real class), same messages. This is the decode direction of
/// the content-addressing pair whose encode half is `b64_encode_bytes`. The
/// core is a line-for-line port of the post-gh-145264
/// `binascii_a2b_base64_impl` (CPython 3.13/3.14 branches; `src/b64_impl.rs`,
/// where the parity battery is pinned crate-side), the machine where lenient
/// mode decodes past padding instead of truncating, the parser-differential
/// fix CPython shipped as a security issue. tors runs that machine on every
/// interpreter it supports; pre-fix stdlibs (and 3.10's regex validator) are
/// documented divergences, recorded in tests/test_b64_decode.py's gate.
///
/// GIL model: same as the str-in functions. The `to_str` argument borrow is a
/// zero-copy alias for ASCII input (the only kind accepted), and the whole
/// scan runs under `py.detach`; the GIL-held residue is the marshalling of the
/// decoded bytes (O(output), ~3/4 of the input size).
///
/// Two argument-boundary behaviors are CPython's: a non-ASCII
/// `str` raises plain `ValueError("string argument should contain only ASCII
/// characters")` (base64.py's `_bytes_from_decode_data`, before any decoding),
/// and non-`str` input raises `TypeError` (the signature is `s: str`; the
/// bytes-likes the stdlib also accepts stay with the stdlib spelling). A str
/// holding lone surrogates hits that same `ValueError`: the stdlib's
/// `s.encode("ascii")` fails inside `_bytes_from_decode_data` and its handler
/// converts the encode error to the plain `ValueError`, so the argument is
/// taken as `Bound<PyString>` and the borrow's `UnicodeEncodeError` is mapped
/// to exactly that (any other borrow failure, e.g. a memory error
/// mid-conversion, propagates unchanged). `validate=True` is the default,
/// the v0.3 spec's signature rather than the stdlib's `False`, since decode-side callers
/// want invalid input to fail loudly.
#[pyfunction(signature = (s, *, validate = true))]
pub fn b64_decode(py: Python<'_>, s: Bound<'_, PyString>, validate: bool) -> PyResult<Py<PyAny>> {
    let s = match s.to_str() {
        Ok(s) => s,
        // The only str a UTF-8 borrow rejects is one holding lone surrogates
        // ("surrogates not allowed"); the stdlib's _bytes_from_decode_data
        // converts that same encode failure to plain ValueError, so match it.
        // Anything else (a memory error mid-conversion) propagates as-is.
        Err(err) => {
            if err.is_instance_of::<PyUnicodeEncodeError>(py) {
                return Err(PyValueError::new_err(
                    "string argument should contain only ASCII characters",
                ));
            }
            return Err(err);
        }
    };
    if !s.is_ascii() {
        return Err(PyValueError::new_err(
            "string argument should contain only ASCII characters",
        ));
    }
    match py.detach(|| b64_impl::decode(s, validate)) {
        Ok(bytes) => Ok(PyBytes::new(py, &bytes).into_any().unbind()),
        Err(err) => Err(binascii_error(py, err.message())),
    }
}

/// `tors.utf8_is_valid`: is `raw` well-formed UTF-8, as one boolean: the
/// stdlib has no primitive for the question (its only spelling is
/// decode-and-catch, materializing the whole `str` on the yes path and constructing
/// the `UnicodeDecodeError` on the no path), while this answers directly
/// with a SIMD validity scan (`simdutf8`): no `str` materialized, no
/// exception flow on either path, valid and invalid input alike.
///
/// GIL model: the bytes-in family's extreme point. The same zero-copy
/// immutable `PyBytes` borrow as `decode_utf8`, and a `bool` return, so
/// there is no marshalling class at all and no error-path exception
/// construction; the argument borrow alone is the call's GIL-held residue.
/// See the crate GIL model above.
#[pyfunction]
pub fn utf8_is_valid(py: Python<'_>, raw: &[u8]) -> bool {
    py.detach(|| utf8_impl::is_valid(raw))
}

/// The `byteorder=` parameter's three accepted spellings, shared by
/// `decode_utf16` and `utf16_is_valid`. Anything else is a `ValueError`,
/// the same closed-set convention `errors=` already uses above.
fn parse_byteorder(byteorder: &str) -> PyResult<utf16_impl::ByteOrder> {
    match byteorder {
        "native" => Ok(utf16_impl::ByteOrder::Native),
        "little" => Ok(utf16_impl::ByteOrder::Little),
        "big" => Ok(utf16_impl::ByteOrder::Big),
        _ => Err(PyValueError::new_err(format!(
            "byteorder must be one of ('native', 'little', 'big'), not {byteorder:?}"
        ))),
    }
}

/// Render a strict-decode failure as the same `UnicodeDecodeError` CPython's
/// own UTF-16 decoder raises. `encoding` is the resolved label
/// (`decode_impl::resolve`'s third element): always `"utf-16-le"` or
/// `"utf-16-be"`, matching CPython's `.encoding` field exactly (it never
/// reports the bare `"utf-16"` name, even when `byteorder="native"` picked
/// that order from a BOM or the host's own endianness).
fn decode_utf16_error_into_pyerr(
    py: Python<'_>,
    raw: &[u8],
    err: utf16_impl::DecodeError,
    encoding: &str,
) -> PyErr {
    let reason = match err {
        utf16_impl::DecodeError::TruncatedData { .. } => c"truncated data",
        utf16_impl::DecodeError::UnexpectedEnd { .. } => c"unexpected end of data",
        utf16_impl::DecodeError::IllegalSurrogate { .. } => c"illegal UTF-16 surrogate",
        utf16_impl::DecodeError::IllegalEncoding { .. } => c"illegal encoding",
    };
    let encoding_c = match encoding {
        "utf-16-le" => c"utf-16-le",
        _ => c"utf-16-be",
    };
    match PyUnicodeDecodeError::new(py, encoding_c, raw, err.start()..err.end(), reason) {
        Ok(exception) => PyErr::from_value(exception.into_any()),
        Err(construction_error) => construction_error,
    }
}

/// `tors.decode_utf16`: `raw.decode("utf-16", errors=...)` (or the
/// `-le`/`-be` spellings, via `byteorder=`) byte-exact over arbitrary bytes,
/// as one GIL-released pass: strict raises the very `UnicodeDecodeError`
/// CPython's decoder raises (type, spans, reason, message: see
/// `utf16_impl`'s module docs for the four distinct reason strings and the
/// BOM/byteorder semantics, all verified against a running interpreter),
/// replace emits CPython's exact one-U+FFFD-per-error-unit placements.
/// `byteorder="native"` (the default) sniffs and strips a leading BOM,
/// falling back to the host's own endianness when none is present, matching
/// the plain `"utf-16"` codec name; `"little"`/`"big"` never sniff or strip
/// a BOM, matching `"utf-16-le"`/`"utf-16-be"`. No `encode_utf16` or
/// `finalize_utf16` twin: see the module docs for why.
#[pyfunction(signature = (raw, *, errors = "strict", byteorder = "native"))]
pub fn decode_utf16(py: Python<'_>, raw: &[u8], errors: &str, byteorder: &str) -> PyResult<String> {
    let mode = parse_errors_mode(errors)?;
    let order = parse_byteorder(byteorder)?;
    match mode {
        ErrorsMode::Strict => py
            .detach(|| utf16_impl::decode_strict(raw, order))
            .map(|(text, _encoding)| text.into_owned())
            .map_err(|(err, encoding)| decode_utf16_error_into_pyerr(py, raw, err, encoding)),
        ErrorsMode::Replace => Ok(py.detach(|| utf16_impl::decode_replace(raw, order).0)),
    }
}

/// `tors.utf16_is_valid`: is `raw` well-formed UTF-16 under `byteorder`, as
/// one boolean: `utf8_is_valid`'s shape for the UTF-16 side. `True` exactly
/// when `decode_utf16(raw, byteorder=byteorder)` (strict) would succeed. No
/// `str` is materialized on either path.
#[pyfunction(signature = (raw, *, byteorder = "native"))]
pub fn utf16_is_valid(py: Python<'_>, raw: &[u8], byteorder: &str) -> PyResult<bool> {
    let order = parse_byteorder(byteorder)?;
    Ok(py.detach(|| utf16_impl::is_valid(raw, order)))
}
