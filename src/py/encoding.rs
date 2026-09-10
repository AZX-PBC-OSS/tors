use pyo3::prelude::*;

use crate::encoding_impl;

/// `tors.detect_encoding(raw, *, tld=None) -> str`: a best-guess codec name
/// for non-UTF8 legacy/OCR byte content, via `chardetng` (the detector
/// Firefox ships). The intended pipeline shape: `utf8_is_valid` first, and
/// only reach for this on the bytes that already failed that check. This
/// is a heuristic guesser, not a validator, and it always returns some
/// encoding, confidence unexposed by the underlying crate beyond the single
/// best answer. `tld` is an optional top-level domain without the leading
/// dot (`"jp"`, not `".jp"`) that disambiguates language-family-ambiguous
/// input; an empty string or `None` both mean "no hint". Same bytes-only
/// argument contract as `decode_utf8`/`utf8_is_valid` (`bytearray`/
/// `memoryview`/`str` raise `TypeError`).
///
/// GIL model: the whole `feed`+`guess` pass runs under `py.detach`; the
/// return is a `&'static str` naming a static table entry (`Encoding::name`),
/// with no allocation, no marshalling class beyond the one `PyString` build.
#[pyfunction(signature = (raw, *, tld = None))]
pub fn detect_encoding(py: Python<'_>, raw: &[u8], tld: Option<&str>) -> &'static str {
    py.detach(|| encoding_impl::detect(raw, tld))
}
