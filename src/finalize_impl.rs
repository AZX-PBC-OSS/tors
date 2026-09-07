use std::borrow::Cow;

use crate::decode_impl::{self, DecodeError};
use crate::normalize_impl;
use sha2::{Digest, Sha256};

/// One native pass: the shared ``normalize``, then SHA-256 over the result's UTF-8 bytes
/// as lowercase hex: byte-identical to Python's
/// ``hashlib.sha256(normalize(t).encode("utf-8")).hexdigest()``, the expression a
/// normalize-then-hash dedupe gate computes, so tors's hash drops into such pipelines
/// without changing any stored digest.
///
/// v0.4: the hash is integrated into the transform's output pass. The scan
/// feeds the hasher the confirmed output bytes as it writes them (chunked,
/// one pass rather than a scan pass plus a digest pass). Identity inputs
/// short-circuit the scan entirely, so the digest comes from the borrowed
/// input buffer and the string element is the input itself (no output allocation).
/// The allocating spelling the criterion bench drives; the pyo3 wrapper adds
/// only the pyo3 boundary around [`finalize_checked`]; see the crate GIL
/// model there for that boundary's GIL-held residue.
pub fn finalize(text: &str) -> (String, String) {
    let (cow, digest) = finalize_checked(text);
    (cow.into_owned(), digest)
}

/// The identity-aware core the pyo3 wrapper drives:
/// ``(Cow::Borrowed(input), digest)`` when the whole pipeline changes nothing
/// (the identity probe's fast lane, digest computed straight from the
/// borrowed input, or the post-scan output==input comparison), and
/// ``(Cow::Owned(output), digest)`` otherwise, with the digest produced by
/// the integrated in-scan hasher.
pub fn finalize_checked(text: &str) -> (Cow<'_, str>, String) {
    let (cow, digest) = normalize_impl::pipeline(text, Some(Sha256::new()));
    let digest = const_hex::encode(digest.expect("a hasher was supplied"));
    (cow, digest)
}

/// The bytes-in flavor: strict UTF-8 decode + normalize + hash in one native
/// pass, `(normalize(raw.decode("utf-8")), sha256-hex-of-the-result)` for valid
/// input, or the CPython-parity `DecodeError` (no pipeline work happens) for
/// invalid input. The extraction-pipeline shape: one `py.detach` call instead of a
/// GIL-held decode followed by `finalize`. The decoded text is BORROWED from
/// the input bytes on the valid path, so an identity pipeline returns it
/// borrowed: the wrapper marshals the output string straight from that
/// borrow, with no owned intermediate.
pub fn finalize_utf8_strict(raw: &[u8]) -> Result<(Cow<'_, str>, String), DecodeError> {
    match decode_impl::decode_strict(raw)? {
        Cow::Borrowed(text) => Ok(finalize_checked(text)),
        // The lossy-strict arm cannot exist (strict errors instead), but the
        // Owned shape still must not borrow the local: force ownership.
        Cow::Owned(text) => {
            let (cow, digest) = finalize_checked(text.as_str());
            Ok((Cow::Owned(cow.into_owned()), digest))
        }
    }
}

/// The `errors="replace"` flavor of [`finalize_utf8_strict`]: the lossy decode's
/// U+FFFD substitutions flow through normalize and the hash like any other
/// character.
pub fn finalize_utf8_replace(raw: &[u8]) -> (Cow<'_, str>, String) {
    match decode_impl::decode_replace(raw) {
        Cow::Borrowed(text) => finalize_checked(text),
        // Lossy decoding allocates; on an identity pipeline the decoded text
        // still comes back owned (the borrow cannot outlive the local).
        Cow::Owned(text) => {
            let (cow, digest) = finalize_checked(text.as_str());
            (Cow::Owned(cow.into_owned()), digest)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_impl::DecodeError;

    #[test]
    fn empty_string_finalizes_to_the_sha256_of_empty() {
        assert_eq!(
            finalize(""),
            (
                String::new(),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string()
            )
        );
    }

    #[test]
    fn ascii_finalizes_to_the_known_sha256_vector() {
        // FIPS 180-4 vector for "abc": the pipeline leaves "abc" untouched, so the hash
        // must be the standard digest, not merely self-consistent.
        assert_eq!(
            finalize("abc"),
            (
                "abc".to_string(),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string()
            )
        );
    }

    #[test]
    fn whitespace_only_finalizes_to_empty_and_its_hash() {
        assert_eq!(
            finalize("   \t\n\n\n   "),
            (
                String::new(),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string()
            )
        );
    }

    #[test]
    fn hash_covers_the_pipeline_output_not_the_input() {
        // "a \t\n" normalizes to "a" (the trailing-ws drop yields "a\n", then the final
        // strip removes the newline): the digest must be of "a", proving the hash runs
        // over the normalize result rather than the raw input.
        let (text, digest) = finalize("a \t\n");
        assert_eq!(text, "a");
        assert_eq!(digest, const_hex::encode(Sha256::digest(b"a")));
    }

    #[test]
    fn finalize_utf8_strict_decodes_then_finalizes_in_one_pass() {
        // The one-call shape an extraction pipeline wants: bytes in, the SAME
        // (normalized, sha256) pair finalize would produce on the decoded text.
        // "abc" survives the pipeline, so the FIPS vector must come back verbatim.
        assert_eq!(
            as_owned(finalize_utf8_strict(b"abc").unwrap()),
            (
                "abc".to_string(),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string()
            )
        );
        assert_eq!(as_owned(finalize_utf8_strict(b"").unwrap()), finalize(""));
        let decomposed = "cafe\u{0301} \n\n\n\nwater";
        assert_eq!(
            as_owned(finalize_utf8_strict(decomposed.as_bytes()).unwrap()),
            finalize(decomposed)
        );
        // Whitespace-only bytes: same empty-and-its-hash pair as finalize.
        assert_eq!(
            as_owned(finalize_utf8_strict(b"   \t\n\n\n   ").unwrap()),
            finalize("   \t\n\n\n   ")
        );
    }

    /// The bytes-in flavors' `(Cow, digest)` in the allocating spelling, so
    /// the one-call results compare directly against `finalize`'s pairs.
    fn as_owned(result: (Cow<'_, str>, String)) -> (String, String) {
        (result.0.into_owned(), result.1)
    }

    #[test]
    fn finalize_utf8_strict_reports_decode_errors_before_any_pipeline_work() {
        // Invalid bytes surface the exact CPython-parity decode error (same span
        // and reason classification as decode_strict) rather than a mangled
        // pipeline result.
        assert_eq!(
            finalize_utf8_strict(b"fo\xd8o"),
            Err(DecodeError::InvalidContinuation { start: 2, end: 3 })
        );
        assert_eq!(
            finalize_utf8_strict(b"\xed\xa0\x80"),
            Err(DecodeError::InvalidContinuation { start: 0, end: 1 })
        );
        assert_eq!(
            finalize_utf8_strict(b"a\xc3"),
            Err(DecodeError::UnexpectedEnd { start: 1, end: 2 })
        );
        assert_eq!(
            finalize_utf8_strict(b"\x80"),
            Err(DecodeError::InvalidStart { start: 0, end: 1 })
        );
    }

    #[test]
    fn finalize_utf8_replace_finalizes_the_lossy_decode() {
        // The errors="replace" flavor: finalize of from_utf8_lossy's output.
        // U+FFFD is not Python whitespace, so replacement chars survive the
        // pipeline and land in the hash.
        assert_eq!(
            as_owned(finalize_utf8_replace(b"fo\xd8o")),
            finalize("fo\u{fffd}o")
        );
        assert_eq!(
            as_owned(finalize_utf8_replace(b"\xe0\x80\x80")),
            finalize("\u{fffd}\u{fffd}\u{fffd}")
        );
        // Valid input: identical to the strict flavor (and to finalize itself).
        assert_eq!(as_owned(finalize_utf8_replace(b"abc")), finalize("abc"));
        assert_eq!(
            as_owned(finalize_utf8_replace(
                "cafe\u{0301} \n\n\n\nwater".as_bytes()
            )),
            finalize("cafe\u{0301} \n\n\n\nwater")
        );
    }

    // --- v0.4: integrated hashing + the identity lane ----------------------------

    #[test]
    fn finalize_checked_identity_input_hashes_the_borrowed_buffer() {
        // The complete pipeline is a no-op on this input: the string element
        // comes back borrowed (the wrapper hands the ORIGINAL PyObject on)
        // and the digest is of the input bytes; no output allocation, no
        // scan.
        let clean = "plain text\n\nwith paragraphs\n\ncaf\u{e9}";
        let (cow, digest) = finalize_checked(clean);
        assert!(matches!(cow, Cow::Borrowed(s) if s == clean));
        assert_eq!(digest, const_hex::encode(Sha256::digest(clean.as_bytes())));
    }

    #[test]
    fn finalize_checked_value_identity_but_qc_maybe_input_borrows_via_the_post_compare() {
        let maybe_clean = "q\u{0328}\u{0301} text";
        let (cow, digest) = finalize_checked(maybe_clean);
        assert!(matches!(cow, Cow::Borrowed(s) if s == maybe_clean));
        assert_eq!(
            digest,
            const_hex::encode(Sha256::digest(maybe_clean.as_bytes()))
        );
    }

    #[test]
    fn finalize_checked_transformed_input_carries_the_integrated_hash() {
        let (cow, digest) = finalize_checked("a \nb");
        assert!(matches!(cow, Cow::Owned(s) if s == "a\nb"));
        assert_eq!(digest, const_hex::encode(Sha256::digest(b"a\nb")));
    }

    #[test]
    fn the_integrated_hash_agrees_with_hashing_the_plain_scans_output_everywhere() {
        // The two scan spellings cross-checked over the exhaustive trigger
        // alphabet: finalize_checked (the hasher fed inside the scan) must
        // equal sha256(normalize(text)) (the non-hashing scan, hashed after)
        // in both string and digest, non-circular since the two digests are
        // produced by different code paths.
        let alphabet = [
            " ", "\t", "\n", "\r", "a", "e", "\u{0301}", "\u{e9}", "\u{a0}",
        ];
        let mut texts = vec![String::new()];
        for _ in 0..4 {
            let mut longer = Vec::with_capacity(texts.len() * alphabet.len());
            for prefix in &texts {
                for piece in alphabet {
                    longer.push(format!("{prefix}{piece}"));
                }
            }
            texts.append(&mut longer);
        }
        for text in &texts {
            let expected = finalize(text);
            let (cow, digest) = finalize_checked(text);
            assert_eq!(cow.as_ref(), expected.0.as_str(), "string for {text:?}");
            assert_eq!(digest, expected.1, "digest for {text:?}");
        }
    }

    #[test]
    fn finalize_utf8_strict_on_clean_bytes_borrows_the_decode() {
        // The bytes-in identity lane: valid, already-clean input decodes to a
        // borrow of the argument and the pipeline is the identity, so the
        // string element is that borrow (the wrapper marshals straight from
        // it: no owned intermediate).
        let raw = b"plain text\n\nwith paragraphs";
        let (cow, digest) = finalize_utf8_strict(raw).unwrap();
        assert!(matches!(cow, Cow::Borrowed(s) if s == "plain text\n\nwith paragraphs"));
        assert_eq!(digest, const_hex::encode(Sha256::digest(raw)));
    }
}
