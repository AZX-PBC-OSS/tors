//! RFC 4648 standard-alphabet base64 of raw bytes: the pure-Rust core of
//! `tors.b64_encode_bytes` (an OCR content-addressing path: ~150ms of GIL-held
//! `base64.b64encode` at a 100MB document cap, the measured motivating case) and
//! `tors.b64_decode`
//! (the decode direction of the same content-addressing pair).
//!
//! Encode parity contract: `tors.b64_encode_bytes(raw)` equals
//! `base64.b64encode(raw).decode("ascii")` (standard alphabet, padded)
//! over arbitrary bytes, pinned over hypothesis-binary and an exhaustive
//! every-byte-value sweep on the Python side (tests/test_b64.py); the RFC 4648
//! §10 vectors below pin the same semantics crate-side without the stdlib.
//!
//! Decode parity contract: `tors.b64_decode(s, validate=...)` is
//! `base64.b64decode(s, validate=...)` over ASCII strings (results, raised
//! exception type, and messages; the exception is the real `binascii.Error`,
//! constructed by the pyo3 layer), pinned by the battery + differentials in
//! tests/test_b64_decode.py. `decode` is a line-for-line port of the
//! POST-gh-145264 `binascii_a2b_base64_impl` (Modules/binascii.c on CPython's
//! 3.13 and 3.14 maintenance branches: identical there by the backports
//! 1f9958f9 / e31c5512; gh-145264, March 2026). The retarget is
//! security-motivated: the PRE-fix machine (which CPython 3.12.x,
//! 3.13.0–3.13.13 and 3.14.0 run; measured on this box) stopped lenient
//! decoding at the first completed pad sequence and silently DROPPED the rest
//! of the input, a parser differential CPython fixed as a security issue; a
//! decoder that returns `b'f'` for `"Zg==Zg=="` on one interpreter and
//! `b'f\x06`'` on another is a version-dependent parser, which is exactly
//! what a content-addressing pair must never be. tors ships the FIXED machine
//! on every interpreter it supports (output does not depend on the Python it
//! runs under), with the pre-fix stdlibs (and 3.10's entirely different regex
//! validator, which pre-dates the `strict_mode` machine) recorded as
//! documented divergences in the Python-side gate.
//!
//! What the fixed machine changes, both pinned crate-side and in the Python
//! battery (literals recorded on CPython 3.13.14, byte-identical on 3.15.0b2):
//!
//! * LENIENT mode ignores excess pads (RFC 4648 §3.3 "MAY ignore") and
//!   DECODES data chars that follow a completed pad sequence, instead of
//!   truncating: `"Zg==Zg=="` → `b"f\x06\x60"`, `"Zm9vYg==Zg=="`
//!   → `b"foob\x06\x60"`. A stray mid-string pad at a quad boundary still
//!   starts a fresh quad (`Zm9v=Zg==` → `foof`).
//! * STRICT mode: a pad arriving at quad position 1 breaks to the end-of-input
//!   length error (`Z=g=` → count 1) instead of raising "Discontinuous
//!   padding" at the following data char; a third pad beyond a quad is
//!   "Excess padding" (`Zg===`) instead of "Excess data after padding"; a
//!   non-alphabet char after a completed pad sequence is "Only base64 data"
//!   (`Zm9vYg==\n`): the pad no longer ends the parse, so the non-alphabet
//!   check fires first. "Discontinuous padding" remains reachable (a data
//!   char after ONE pad at quad position 2, `Zg=g`), as does "Excess data
//!   after padding" (a data char after a completed quad of pads, `Zg==Z`),
//!   and "Leading padding" is now classified in-loop (`=` as the very first
//!   input char) rather than by a pre-loop check.
//! * Unchanged: the five distinct strict messages' exact text, the
//!   acceptance of non-canonical trailing bits (`Zh==` → `f`), the
//!   quad-position-1 count error's formula (emitted bytes / 3 × 4 + 1, both
//!   modes), and "Incorrect padding" for quad positions 2/3 whose pads do
//!   not complete the quad.
//!
//! The base64 crate's own `decode` could NOT be used here (unchanged from the
//! pre-fix decision): its default engine rejects non-canonical trailing bits
//! (`InvalidLastSymbol`), which CPython accepts, and its error taxonomy does
//! not map onto CPython's strict messages.
//!
//! Pure Rust, no pyo3 types: the criterion bench (benches/bytes.rs) drives this
//! path directly; the pyo3 wrapper in `lib.rs` adds only the argument borrow,
//! the ASCII/surrogate argument-boundary checks, and the return marshalling
//! (see the crate GIL model there).

use base64::Engine;

/// RFC 4648 standard-alphabet, padded base64 of `raw`, exactly
/// `base64.b64encode(raw).decode("ascii")`. The base64 crate's preconfigured
/// `STANDARD` engine is precisely this combination (standard alphabet + `PAD`),
/// verified in its source for both 0.22 and 0.23 (the 0.23 release notes list no
/// encode-side changes), so there is nothing of our own to get wrong here: the
/// parity pins in tests/test_b64.py are the contract's proof.
pub fn encode(raw: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(raw)
}

/// The error shapes of CPython's post-gh-145264 `binascii.a2b_base64`: one
/// variant per distinct message, `message()` rendering each exactly as the
/// stdlib raises it (the pyo3 layer attaches these to the real
/// `binascii.Error` class, so `str(exc)` matches byte-for-byte).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Strict mode, `=` as the very first input char (`====`):
    /// "Leading padding not allowed" (classified in-loop by position in the
    /// post-fix machine; the pre-fix machine pre-checked position 0).
    LeadingPadding,
    /// Strict mode, any other pad the machine rejects (a pad at a quad
    /// boundary mid-string, `Zm9v=`, or a third pad beyond a quad, `Zg===`):
    /// "Excess padding not allowed".
    ExcessPadding,
    /// Strict mode, any non-alphabet char (whitespace included):
    /// "Only base64 data is allowed".
    OnlyBase64Data,
    /// Strict mode, a data char after ONE pad at quad position 2 (`Zg=g`):
    /// "Discontinuous padding not allowed". (A pad at quad position 1 no
    /// longer lands here post-fix; it breaks to the count error below.)
    DiscontinuousPadding,
    /// Strict mode, a data char after a completed quad of pads (`Zg==Z`):
    /// "Excess data after padding".
    ExcessDataAfterPadding,
    /// End of input with one unmatched data char (both modes): the count is
    /// the emitted bytes' complete quads × 4 + 1. Post-fix this is also where
    /// a strict pad at quad position 1 lands (`Z=g=`, `Z==g=`).
    InvalidCharCount(usize),
    /// End of input with 2 or 3 data chars in the final quad whose pads do
    /// not complete it (both modes): "Incorrect padding".
    IncorrectPadding,
}

impl DecodeError {
    /// The exact `str()` of the `binascii.Error` CPython raises for this
    /// shape, recorded on 3.13.14, re-verified per CI leg by the
    /// Python-side battery's live differential against the running stdlib
    /// (where that stdlib has the fix).
    pub fn message(&self) -> String {
        match self {
            DecodeError::LeadingPadding => "Leading padding not allowed".to_string(),
            DecodeError::ExcessPadding => "Excess padding not allowed".to_string(),
            DecodeError::OnlyBase64Data => "Only base64 data is allowed".to_string(),
            DecodeError::DiscontinuousPadding => "Discontinuous padding not allowed".to_string(),
            DecodeError::ExcessDataAfterPadding => "Excess data after padding".to_string(),
            DecodeError::InvalidCharCount(n) => format!(
                "Invalid base64-encoded string: number of data characters ({n}) \
                 cannot be 1 more than a multiple of 4"
            ),
            DecodeError::IncorrectPadding => "Incorrect padding".to_string(),
        }
    }
}

/// The a2b decode table (value 0..=63 for the standard alphabet, 0xFF for
/// everything else), built as a const fn so the alphabet appears literally
/// (a hand-written 256-entry table is the classic place to mistype one entry).
const fn a2b_table() -> [u8; 256] {
    let alphabet: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut table = [0xFFu8; 256];
    let mut i = 0;
    while i < 64 {
        table[alphabet[i] as usize] = i as u8;
        i += 1;
    }
    table
}

static A2B_TABLE: [u8; 256] = a2b_table();

/// A line-for-line port of the POST-gh-145264 `binascii_a2b_base64_impl`
/// (Modules/binascii.c, CPython 3.13/3.14 branches); see the module docs for
/// the parity contract and the retarget's security rationale. The caller
/// guarantees `s` is ASCII (the pyo3 wrapper enforces `base64.b64decode`'s
/// str contract first), so bytes == chars here.
pub fn decode(s: &str, strict: bool) -> Result<Vec<u8>, DecodeError> {
    let ascii = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(ascii.len() / 4 * 3 + 3);
    let mut quad_pos: usize = 0;
    let mut leftchar: u8 = 0;
    // usize, not u8/int: a lenient-mode input can run hundreds of '=' through
    // quads (each ignored per RFC 4648 §3.3) without any gate firing, and the
    // count must not wrap where CPython's `int pads` does not.
    let mut pads: usize = 0;

    for (i, &ch) in ascii.iter().enumerate() {
        if ch == b'=' {
            pads += 1;
            if quad_pos >= 2 && quad_pos + pads <= 4 {
                // A pad completing the current quad (positions 2/3).
                continue;
            }
            // RFC 4648 §3.3: a pad before the end of the encoded data and
            // excess pads MAY be ignored. Lenient mode does exactly that, and
            // (the gh-145264 change) decoding CONTINUES with later data
            // chars instead of truncating at the first pad sequence.
            if !strict {
                continue;
            }
            if quad_pos == 1 {
                // CPython breaks to "set an error below": the end-of-input
                // count error, not a padding classification.
                break;
            }
            return Err(if quad_pos == 0 && i == 0 {
                DecodeError::LeadingPadding
            } else {
                DecodeError::ExcessPadding
            });
        }

        let val = A2B_TABLE[ch as usize];
        if val >= 64 {
            if strict {
                return Err(DecodeError::OnlyBase64Data);
            }
            continue; // lenient: discard any non-alphabet char, whitespace included
        }

        // A data char while pads are pending (strict only; lenient resets
        // below and decodes on, the resumption rule): whether the pads had
        // completed the quad decides the message.
        if pads > 0 && strict {
            return Err(if quad_pos + pads == 4 {
                DecodeError::ExcessDataAfterPadding
            } else {
                DecodeError::DiscontinuousPadding
            });
        }
        pads = 0;

        match quad_pos {
            0 => {
                quad_pos = 1;
                leftchar = val;
            }
            1 => {
                quad_pos = 2;
                out.push((leftchar << 2) | (val >> 4));
                leftchar = val & 0x0f;
            }
            2 => {
                quad_pos = 3;
                out.push((leftchar << 4) | (val >> 2));
                leftchar = val & 0x03;
            }
            _ => {
                quad_pos = 0;
                out.push((leftchar << 6) | val);
                leftchar = 0;
            }
        }
    }

    if quad_pos == 1 {
        // Mode-independent, and post-fix also the destination of a strict pad
        // at quad position 1: the count comes from the emitted bytes
        // (complete quads × 4 + 1), exactly as CPython's PyErr_Format
        // argument does.
        return Err(DecodeError::InvalidCharCount(out.len() / 3 * 4 + 1));
    }
    if quad_pos != 0 && quad_pos + pads < 4 {
        // Quad positions 2/3 whose pads do not complete the quad.
        return Err(DecodeError::IncorrectPadding);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_the_rfc_4648_section_10_vectors_verbatim() {
        // The RFC's own test vectors, pinned literally so a wrong alphabet or
        // padding wiring cannot hide behind parity with the stdlib alone.
        let vectors: &[(&[u8], &str)] = &[
            (b"", ""),
            (b"f", "Zg=="),
            (b"fo", "Zm8="),
            (b"foo", "Zm9v"),
            (b"foob", "Zm9vYg=="),
            (b"fooba", "Zm9vYmE="),
            (b"foobar", "Zm9vYmFy"),
        ];
        for (raw, expected) in vectors {
            assert_eq!(&encode(raw), expected, "mismatch for {expected}");
        }
    }

    #[test]
    fn encodes_the_padding_shape_for_each_input_length_mod_three() {
        // 1 byte -> "==", 2 bytes -> "=", 0 mod 3 -> none: the padding shapes are
        // the parity cases the Python sweep tests stress, pinned directly against
        // the stdlib's own output for these exact inputs.
        assert_eq!(&encode(&[0xff]), "/w==");
        assert_eq!(&encode(&[0xff, 0xff]), "//8=");
        assert_eq!(&encode(&[0xff, 0xff, 0xff]), "////");
        assert_eq!(&encode(&[0xff, 0xff, 0xff, 0xff]), "/////w==");
    }

    #[test]
    fn encodes_a_four_byte_utf8_emoji_to_its_known_encoding() {
        // U+1F600 as UTF-8 bytes: a known-answer vector outside pure ASCII.
        assert_eq!(&encode("\u{1f600}".as_bytes()), "8J+YgA==");
    }

    #[test]
    fn output_is_always_standard_alphabet_plus_padding() {
        for byte in 0..=255u8 {
            let encoded = encode(&[byte]);
            assert_eq!(encoded.len(), 4);
            assert!(
                encoded[..2]
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'+' || c == b'/')
            );
            assert_eq!(&encoded[2..], "==");
        }
    }

    // --- decode: the measured post-gh-145264 battery (see the module docs) ---
    //
    // (input, strict, expected): expected is Ok(bytes) or Err(variant), with
    // InvalidCharCount carrying the stdlib's interpolated count. Every row was
    // recorded from a real post-fix `base64.b64decode` (CPython 3.13.14,
    // cross-verified byte-identical on 3.15.0b2) before this retarget existed
    // and is re-verified against the running interpreter by
    // tests/test_b64_decode.py on every CI leg whose stdlib has the fix.

    #[test]
    fn decode_decodes_the_valid_battery_in_both_modes() {
        let cases: &[(&str, Result<&[u8], DecodeError>)] = &[
            ("", Ok(b"")),
            ("Zm9vYmFy", Ok(b"foobar")),
            ("Zg==", Ok(b"f")),
            ("Zm9=", Ok(b"fo")),
            // Non-canonical trailing bits: ACCEPTED (the base64 crate's default
            // engine would reject `Zh==`; this is the reason the port exists).
            ("Zh==", Ok(b"f")),
            ("ABCDEFGH", Ok(b"\x00\x10\x83\x10Q\x87")),
        ];
        for (s, expected) in cases {
            for &strict in &[true, false] {
                assert_eq!(
                    decode(s, strict),
                    expected.clone().map(<[u8]>::to_vec),
                    "mismatch for {s:?} strict={strict}"
                );
            }
        }
    }

    #[test]
    fn decode_strict_mode_raises_each_of_the_five_distinct_messages() {
        let cases: &[(&str, DecodeError)] = &[
            ("====", DecodeError::LeadingPadding),
            ("=====", DecodeError::LeadingPadding),
            ("Zm9v=", DecodeError::ExcessPadding),
            ("Zm9v=Zg==", DecodeError::ExcessPadding),
            // A third pad beyond a complete quad: post-fix this is "Excess
            // padding" (the completed quad no longer ends the parse).
            ("Zg===", DecodeError::ExcessPadding),
            ("Zm9vYg===", DecodeError::ExcessPadding),
            ("Zm9vYg====", DecodeError::ExcessPadding),
            ("Zm9v!", DecodeError::OnlyBase64Data),
            ("Zm 9v\nYg==", DecodeError::OnlyBase64Data),
            // The non-alphabet check fires BEFORE the pending-pads check, so a
            // newline after a pad is "Only base64 data is allowed", and
            // post-fix a newline after a COMPLETED pad sequence is too (the
            // pad no longer ends the parse).
            ("Zm9vYg=\n=", DecodeError::OnlyBase64Data),
            ("Zm9vYg==\n", DecodeError::OnlyBase64Data),
            ("Zg==\n", DecodeError::OnlyBase64Data),
            // A pad at quad position 1: post-fix the end-of-input COUNT error,
            // not "Discontinuous padding" (which now needs a data char after
            // one pad at quad position 2).
            ("Z=g=", DecodeError::InvalidCharCount(1)),
            ("Z==g=", DecodeError::InvalidCharCount(1)),
            ("Zg=g", DecodeError::DiscontinuousPadding),
            // A data char after a completed quad of pads.
            ("Zg==Z", DecodeError::ExcessDataAfterPadding),
            ("Zg==Zg==", DecodeError::ExcessDataAfterPadding),
            ("Zm9vYg==Zg==", DecodeError::ExcessDataAfterPadding),
            ("Zm9vYg==Zm9vYg==", DecodeError::ExcessDataAfterPadding),
        ];
        for &(s, ref expected) in cases {
            assert_eq!(decode(s, true).as_ref().unwrap_err(), expected, "for {s:?}");
        }
    }

    #[test]
    fn decode_strict_messages_render_byte_identically_to_the_stdlib() {
        // The exact str() a caller's handlers and logs match on, pinned
        // literally, not just as variants.
        assert_eq!(
            decode("Zm9v!", true).unwrap_err().message(),
            "Only base64 data is allowed"
        );
        assert_eq!(
            decode("Zm9v=", true).unwrap_err().message(),
            "Excess padding not allowed"
        );
        assert_eq!(
            decode("Zg===", true).unwrap_err().message(),
            "Excess padding not allowed"
        );
        assert_eq!(
            decode("Zg=g", true).unwrap_err().message(),
            "Discontinuous padding not allowed"
        );
        assert_eq!(
            decode("====", true).unwrap_err().message(),
            "Leading padding not allowed"
        );
        assert_eq!(
            decode("Zg==Zg==", true).unwrap_err().message(),
            "Excess data after padding"
        );
        assert_eq!(
            decode("Zg=", true).unwrap_err().message(),
            "Incorrect padding"
        );
    }

    #[test]
    fn decode_reports_the_interpolated_data_char_count_from_emitted_bytes() {
        // quad_pos == 1 at END OF INPUT: count = complete quads × 4 + 1, in
        // BOTH modes (padding does not rescue a 1-data-char quad; a lenient
        // run of pads after one data char never resumes, so qp is still 1).
        for &(s, count) in &[
            ("Z", 1),
            ("Zm9vY", 5),
            ("Z==", 1),
            ("A=", 1),
            ("Zm9vYmFyZ", 9),
        ] {
            for &strict in &[true, false] {
                assert_eq!(
                    decode(s, strict),
                    Err(DecodeError::InvalidCharCount(count)),
                    "for {s:?} strict={strict}"
                );
            }
        }
        // STRICT-only: a pad arriving AT quad position 1 breaks to this same
        // count error (the post-fix change); the lenient spelling of the
        // same inputs ignores the pads and the following data char RESUMES
        // the quad, so it fails with "Incorrect padding" instead (pinned in
        // the lenient battery below).
        for &(s, count) in &[("Z=g=", 1), ("Z==g=", 1)] {
            assert_eq!(
                decode(s, true),
                Err(DecodeError::InvalidCharCount(count)),
                "for {s:?}"
            );
            assert_eq!(
                decode(s, false),
                Err(DecodeError::IncorrectPadding),
                "for {s:?}"
            );
        }
        assert_eq!(
            decode("Z", true).unwrap_err().message(),
            "Invalid base64-encoded string: number of data characters (1) cannot be \
             1 more than a multiple of 4"
        );
    }

    #[test]
    fn decode_lenient_mode_discards_non_alphabet_and_decodes_past_padding() {
        // The gh-145264 fix itself: excess pads are ignored and later data
        // chars DECODE, where the pre-fix machine silently dropped everything
        // after the first completed pad sequence.
        let cases: &[(&str, Result<&[u8], DecodeError>)] = &[
            ("Zm9v!", Ok(b"foo")),
            ("Zm 9v\nYg==", Ok(b"foob")),
            ("Zm9vYg=\n=", Ok(b"foob")),
            ("Zg==\n", Ok(b"f")),
            // Data after a completed pad sequence is DECODED, not dropped.
            ("Zg==Zg==", Ok(b"f\x06`")),
            ("Zg==!Zg==", Ok(b"f\x06`")),
            ("Zg==Zg==Zg==", Ok(b"f\x06`f")),
            ("Zm9vYg==Zg==", Ok(b"foob\x06`")),
            ("Zm9vYg==Zm9vYg==", Ok(b"foob\x06f\xf6\xf6 ")),
            ("====", Ok(b"")),
            ("=====", Ok(b"")),
            // A stray mid-string pad at a quad boundary starts a fresh quad.
            ("Zm9v=", Ok(b"foo")),
            ("Zm9v=Zg==", Ok(b"foof")),
            ("Zg===", Ok(b"f")),
            ("Zm9vYm E=", Ok(b"fooba")),
            // quad_pos 2/3 at end of input is still "Incorrect padding",
            // including the lenient spelling of a pad at quad position 1
            // whose data char resumes the quad (`Z=g=`, `Z==g=`).
            ("Zg=", Err(DecodeError::IncorrectPadding)),
            ("Zm9vYg", Err(DecodeError::IncorrectPadding)),
            ("Z=g=", Err(DecodeError::IncorrectPadding)),
            ("Z==g=", Err(DecodeError::IncorrectPadding)),
            // A 1-data-char quad is the count error even in lenient mode
            // (the pads after it are ignored, never resumed).
            ("Z==", Err(DecodeError::InvalidCharCount(1))),
            ("A=", Err(DecodeError::InvalidCharCount(1))),
        ];
        for (s, expected) in cases {
            assert_eq!(
                decode(s, false),
                expected.clone().map(<[u8]>::to_vec),
                "mismatch for {s:?} lenient"
            );
        }
    }

    #[test]
    fn decode_round_trips_every_encode_and_handles_very_long_pad_runs() {
        // The pair's own round trip at every residue mod 3 up to 64 bytes.
        for length in 0..=64u32 {
            let raw: Vec<u8> = (0..length).map(|i| ((i * 251 + 7) % 256) as u8).collect();
            let s = encode(&raw);
            assert_eq!(decode(&s, true), Ok(raw));
        }
        // 300 '=' through lenient quads must not wrap any counter (the port
        // uses usize where CPython's C uses int): every pad is ignored and
        // the count error quad_pos-1 fires on the byte 'Z', never here.
        assert_eq!(decode(&"=".repeat(300), false), Ok(Vec::new()));
        // ...and strict still classifies the very first one as leading padding.
        assert_eq!(
            decode(&"=".repeat(300), true),
            Err(DecodeError::LeadingPadding)
        );
    }
}
