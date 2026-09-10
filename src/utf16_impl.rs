//! CPython-parity UTF-16 decoding of raw bytes: the pure-Rust core of
//! `tors.decode_utf16` / `tors.utf16_is_valid`.
//!
//! Hand-rolled directly over byte positions rather than routed through
//! `std::char::decode_utf16`'s iterator: that API reports an unpaired
//! surrogate's raw `u16` value, not the byte-level distinction CPython's own
//! decoder makes between four different error shapes (see [`DecodeError`]),
//! so matching CPython exactly needs the same close-to-the-bytes control
//! `decode_impl.rs` already uses for UTF-8.
//!
//! Semantics were verified against a running CPython 3.12 interpreter, not
//! assumed from the codec's documentation:
//! - The plain `"utf-16"` codec name sniffs a leading BOM (`FF FE` little,
//!   `FE FF` big) and consumes it from the output; with no BOM it falls back
//!   to the host's native byte order. `"utf-16-le"`/`"utf-16-be"` never sniff
//!   or strip a BOM: a leading `FF FE`/`FE FF` under an explicit byte order
//!   decodes as a literal U+FEFF character.
//! - A lone trailing byte (not following an unpaired high surrogate) is
//!   `"truncated data"`, spanning just that byte.
//! - A high surrogate with fewer than two bytes following it is
//!   `"unexpected end of data"`, spanning from the surrogate through the end
//!   of the input (there is nothing left to do with the partial tail, so it
//!   is folded into the one error rather than reported separately).
//! - A high surrogate followed by a full code unit that is not a valid low
//!   surrogate is `"illegal UTF-16 surrogate"`, spanning only the high
//!   surrogate's two bytes; the following unit is reprocessed from scratch.
//! - A lone low surrogate (reached other than as the second half of a valid
//!   pair) is `"illegal encoding"`, spanning only its own two bytes.
//!
//! `.start`/`.end` are always offsets into the original input bytes
//! (including any stripped BOM), matching CPython's own `UnicodeDecodeError`
//! fields exactly: verified directly, not assumed.
//!
//! No `encode_utf16`/`finalize_utf16`: encoding a trusted `str` to bytes
//! carries none of the untrusted-input-parsing risk that motivates the
//! decode side, and `str.encode("utf-16-le")` already covers it without a
//! GIL-hold problem worth solving; a fused decode+normalize+hash twin has no
//! demonstrated caller need yet, unlike `finalize_utf8`'s.

use std::borrow::Cow;

/// Which byte order to decode with. `Native` sniffs a leading BOM and falls
/// back to the host's own order when none is present, matching Python's
/// plain `"utf-16"` codec name; `Little`/`Big` never sniff or strip a BOM,
/// matching `"utf-16-le"`/`"utf-16-be"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    Native,
    Little,
    Big,
}

/// The four `UnicodeDecodeError` shapes CPython's strict UTF-16 decoder
/// produces, with CPython's exact byte span (verified against a running
/// interpreter; see the module docs for each case).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// A lone trailing byte, one short of a full code unit, with no pending
    /// high surrogate. Span: that one byte.
    TruncatedData { start: usize, end: usize },
    /// A high surrogate with fewer than two bytes following it. Span: the
    /// surrogate through the end of the input.
    UnexpectedEnd { start: usize, end: usize },
    /// A high surrogate followed by a full code unit that is not a valid
    /// low surrogate. Span: the high surrogate's two bytes alone.
    IllegalSurrogate { start: usize, end: usize },
    /// A low surrogate reached other than as the second half of a valid
    /// pair. Span: its own two bytes alone.
    IllegalEncoding { start: usize, end: usize },
}

impl DecodeError {
    pub fn start(&self) -> usize {
        match self {
            DecodeError::TruncatedData { start, .. }
            | DecodeError::UnexpectedEnd { start, .. }
            | DecodeError::IllegalSurrogate { start, .. }
            | DecodeError::IllegalEncoding { start, .. } => *start,
        }
    }

    pub fn end(&self) -> usize {
        match self {
            DecodeError::TruncatedData { end, .. }
            | DecodeError::UnexpectedEnd { end, .. }
            | DecodeError::IllegalSurrogate { end, .. }
            | DecodeError::IllegalEncoding { end, .. } => *end,
        }
    }

    /// CPython's reason string, verbatim.
    pub fn reason(&self) -> &'static str {
        match self {
            DecodeError::TruncatedData { .. } => "truncated data",
            DecodeError::UnexpectedEnd { .. } => "unexpected end of data",
            DecodeError::IllegalSurrogate { .. } => "illegal UTF-16 surrogate",
            DecodeError::IllegalEncoding { .. } => "illegal encoding",
        }
    }
}

const HIGH_SURROGATE: std::ops::RangeInclusive<u16> = 0xD800..=0xDBFF;
const LOW_SURROGATE: std::ops::RangeInclusive<u16> = 0xDC00..=0xDFFF;

/// Resolve `byteorder` against a leading BOM in `raw`: `(little_endian,
/// bom_len, encoding_label)`. `encoding_label` is the resolved name CPython
/// reports as `UnicodeDecodeError.encoding`: always `"utf-16-le"` or
/// `"utf-16-be"`, never the bare `"utf-16"` name, even for `Native`.
fn resolve(raw: &[u8], byteorder: ByteOrder) -> (bool, usize, &'static str) {
    match byteorder {
        ByteOrder::Little => (true, 0, "utf-16-le"),
        ByteOrder::Big => (false, 0, "utf-16-be"),
        ByteOrder::Native => {
            if raw.starts_with(&[0xFF, 0xFE]) {
                (true, 2, "utf-16-le")
            } else if raw.starts_with(&[0xFE, 0xFF]) {
                (false, 2, "utf-16-be")
            } else {
                let little = cfg!(target_endian = "little");
                (little, 0, if little { "utf-16-le" } else { "utf-16-be" })
            }
        }
    }
}

fn read_unit(bytes: &[u8], little_endian: bool) -> u16 {
    let pair = [bytes[0], bytes[1]];
    if little_endian {
        u16::from_le_bytes(pair)
    } else {
        u16::from_be_bytes(pair)
    }
}

/// One step of the decode walk: either a decoded scalar value or an error
/// span, handed to a single callback so strict and replace modes (and the
/// validity check) can share one traversal without fighting the borrow
/// checker over two closures that both want the same output buffer.
enum Event {
    Char(char),
    Error(DecodeError),
}

/// The decode core shared by strict, replace, and validity modes: walks
/// `body` (the input after any BOM has been sliced off) two bytes at a
/// time, calling `on_event` for each decoded scalar value or error span.
/// `offset` is `body`'s start position in the original input, so reported
/// spans land in the caller's coordinate system.
///
/// `on_event` returns `true` to continue the walk (replace mode, or a
/// validity check that just wants to note failure and keep scanning if it
/// cared to) or `false` to stop immediately (strict mode: the first error
/// wins, matching CPython's own single-pass strict-mode behavior).
fn walk(body: &[u8], little_endian: bool, offset: usize, mut on_event: impl FnMut(Event) -> bool) {
    let mut pos = 0usize;
    while pos < body.len() {
        if body.len() - pos < 2 {
            if !on_event(Event::Error(DecodeError::TruncatedData {
                start: offset + pos,
                end: offset + pos + 1,
            })) {
                return;
            }
            pos += 1;
            continue;
        }
        let unit = read_unit(&body[pos..], little_endian);
        if HIGH_SURROGATE.contains(&unit) {
            let rest = &body[pos + 2..];
            if rest.len() < 2 {
                // Nothing usable remains either way; the whole tail folds
                // into this one error regardless of what the callback
                // wants to do next, since there is nothing left to walk.
                on_event(Event::Error(DecodeError::UnexpectedEnd {
                    start: offset + pos,
                    end: offset + body.len(),
                }));
                return;
            }
            let next = read_unit(rest, little_endian);
            if LOW_SURROGATE.contains(&next) {
                let c = 0x10000 + ((u32::from(unit) - 0xD800) << 10) + (u32::from(next) - 0xDC00);
                // A valid surrogate pair always yields a valid scalar value
                // in 0x10000..=0x10FFFF.
                on_event(Event::Char(
                    char::from_u32(c).expect("valid surrogate pair"),
                ));
                pos += 4;
                continue;
            }
            if !on_event(Event::Error(DecodeError::IllegalSurrogate {
                start: offset + pos,
                end: offset + pos + 2,
            })) {
                return;
            }
            pos += 2;
            continue;
        }
        if LOW_SURROGATE.contains(&unit) {
            if !on_event(Event::Error(DecodeError::IllegalEncoding {
                start: offset + pos,
                end: offset + pos + 2,
            })) {
                return;
            }
            pos += 2;
            continue;
        }
        // Any u16 outside both surrogate ranges is a valid scalar value on
        // its own.
        on_event(Event::Char(
            char::from_u32(u32::from(unit)).expect("non-surrogate code unit"),
        ));
        pos += 2;
    }
}

/// Strict decoding: the first error encountered (leftmost byte position)
/// stops the walk and is returned, matching CPython's own single-pass
/// strict-mode behavior.
pub fn decode_strict(
    raw: &[u8],
    byteorder: ByteOrder,
) -> Result<(Cow<'_, str>, &'static str), (DecodeError, &'static str)> {
    let (little_endian, bom_len, encoding) = resolve(raw, byteorder);
    let body = &raw[bom_len..];
    let mut out = String::with_capacity(body.len() / 2);
    let mut error = None;
    walk(body, little_endian, bom_len, |event| match event {
        Event::Char(c) => {
            out.push(c);
            true
        }
        Event::Error(err) => {
            error = Some(err);
            false
        }
    });
    match error {
        Some(err) => Err((err, encoding)),
        None => Ok((Cow::Owned(out), encoding)),
    }
}

/// `errors="replace"` decoding: every error emits one U+FFFD and decoding
/// continues, matching CPython's exact per-error-unit granularity (verified:
/// an unpaired surrogate followed by valid text yields exactly one U+FFFD,
/// not one per byte).
pub fn decode_replace(raw: &[u8], byteorder: ByteOrder) -> (String, &'static str) {
    let (little_endian, bom_len, encoding) = resolve(raw, byteorder);
    let body = &raw[bom_len..];
    let mut out = String::with_capacity(body.len() / 2);
    walk(body, little_endian, bom_len, |event| {
        match event {
            Event::Char(c) => out.push(c),
            Event::Error(_) => out.push('\u{FFFD}'),
        }
        true
    });
    (out, encoding)
}

/// Is `raw` well-formed UTF-16 under `byteorder`: `true` exactly when
/// [`decode_strict`] would succeed. No `String` is built on either path.
pub fn is_valid(raw: &[u8], byteorder: ByteOrder) -> bool {
    let (little_endian, bom_len, _encoding) = resolve(raw, byteorder);
    let body = &raw[bom_len..];
    let mut ok = true;
    walk(body, little_endian, bom_len, |event| match event {
        Event::Char(_) => true,
        Event::Error(_) => {
            ok = false;
            false
        }
    });
    ok
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strict(raw: &[u8]) -> Result<String, DecodeError> {
        decode_strict(raw, ByteOrder::Native)
            .map(|(s, _)| s.into_owned())
            .map_err(|(err, _)| err)
    }

    fn strict_le(raw: &[u8]) -> Result<String, DecodeError> {
        decode_strict(raw, ByteOrder::Little)
            .map(|(s, _)| s.into_owned())
            .map_err(|(err, _)| err)
    }

    fn strict_be(raw: &[u8]) -> Result<String, DecodeError> {
        decode_strict(raw, ByteOrder::Big)
            .map(|(s, _)| s.into_owned())
            .map_err(|(err, _)| err)
    }

    #[test]
    fn empty_input_decodes_to_empty_string() {
        assert_eq!(strict(b"").unwrap(), "");
    }

    #[test]
    fn bom_sets_endianness_and_is_stripped_from_output() {
        assert_eq!(strict(b"\xff\xfeh\x00i\x00").unwrap(), "hi");
        assert_eq!(strict(b"\xfe\xff\x00h\x00i").unwrap(), "hi");
    }

    #[test]
    fn no_bom_falls_back_to_native_endianness() {
        // On this (little-endian) box, no-BOM input is read little-endian.
        assert_eq!(strict(b"h\x00i\x00").unwrap(), "hi");
    }

    #[test]
    fn explicit_byteorder_never_strips_a_bom_like_prefix() {
        // FF FE under an explicit little-endian order decodes as the
        // literal BOM character U+FEFF, not a sniffed/stripped marker.
        assert_eq!(strict_le(b"\xff\xfeh\x00i\x00").unwrap(), "\u{feff}hi");
    }

    #[test]
    fn lone_trailing_byte_is_truncated_data() {
        assert!(matches!(
            strict(b"h\x00i"),
            Err(DecodeError::TruncatedData { start: 2, end: 3 })
        ));
    }

    #[test]
    fn high_surrogate_with_nothing_following_is_unexpected_end() {
        assert!(matches!(
            strict_le(b"\x00\xd8"),
            Err(DecodeError::UnexpectedEnd { start: 0, end: 2 })
        ));
    }

    #[test]
    fn high_surrogate_with_one_dangling_byte_folds_into_one_unexpected_end() {
        assert!(matches!(
            strict_le(b"\x00\xd8Z"),
            Err(DecodeError::UnexpectedEnd { start: 0, end: 3 })
        ));
    }

    #[test]
    fn high_surrogate_followed_by_non_low_surrogate_is_illegal_surrogate() {
        assert!(matches!(
            strict_le(b"\x00\xd8A\x00"),
            Err(DecodeError::IllegalSurrogate { start: 0, end: 2 })
        ));
    }

    #[test]
    fn lone_low_surrogate_is_illegal_encoding() {
        assert!(matches!(
            strict_le(b"\x00\xdc"),
            Err(DecodeError::IllegalEncoding { start: 0, end: 2 })
        ));
        assert!(matches!(
            strict_le(b"\x00\xdcA\x00"),
            Err(DecodeError::IllegalEncoding { start: 0, end: 2 })
        ));
    }

    #[test]
    fn valid_surrogate_pair_decodes_to_the_astral_scalar() {
        // U+1F600 GRINNING FACE, little-endian pair.
        assert_eq!(strict_le(b"\x3d\xd8\x00\xde").unwrap(), "\u{1f600}");
    }

    #[test]
    fn replace_emits_one_fffd_per_error_unit_and_continues() {
        let (out, _) = decode_replace(b"\x00\xd8A\x00", ByteOrder::Little);
        assert_eq!(out, "\u{fffd}A");
        let (out, _) = decode_replace(b"\x00\xdcA\x00", ByteOrder::Little);
        assert_eq!(out, "\u{fffd}A");
        let (out, _) = decode_replace(b"h\x00i", ByteOrder::Little);
        assert_eq!(out, "h\u{fffd}");
    }

    #[test]
    fn valid_input_is_valid_and_invalid_input_is_not() {
        assert!(is_valid(b"h\x00i\x00", ByteOrder::Little));
        assert!(!is_valid(b"h\x00i", ByteOrder::Little));
        assert!(!is_valid(b"\x00\xd8", ByteOrder::Little));
    }

    #[test]
    fn big_endian_variants_of_surrogate_errors() {
        assert!(matches!(
            strict_be(b"\xd8\x00"),
            Err(DecodeError::UnexpectedEnd { start: 0, end: 2 })
        ));
        assert!(matches!(
            strict_be(b"\xdc\x00"),
            Err(DecodeError::IllegalEncoding { start: 0, end: 2 })
        ));
    }
}
