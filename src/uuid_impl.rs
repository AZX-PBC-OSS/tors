//! UUIDv7 helper cores: the pure-Rust hearts of `tors.uuid7_timestamp_ms`,
//! `tors.uuid_version`, and `tors.uuid_parse` (issue #54).
//!
//! The consumer shape (TaskQ is the measured instance): a store keyed by
//! UUIDv7 IDs reimplements the same three bit operations at every site --
//! the 48-bit unix-millisecond timestamp for keyset pagination and
//! time-bucketed queries, the version nibble before trusting that
//! timestamp, and a strict canonical-text parse at the ID-validation
//! boundary. 16 bytes in, integer out: trivial bit work, which is exactly
//! why it keeps getting reimplemented slightly wrong. This module is
//! tors's one spelling of it.
//!
//! # The `uuid` crate underneath, the strictness layer on top
//!
//! The hex grammar work is delegated to the `uuid` crate (the uuid-rs
//! org's, Apache-2.0 OR MIT, `default-features = false`: parse/encode
//! need no features, and the config pulls ZERO transitive dependencies,
//! `cargo tree`-verified). `Uuid::parse_str` is the battle-tested
//! structural parse, and an accepted input's bytes are the crate's
//! `as_bytes()` -- the hex-to-nibble transcode is upstream's, not tors's;
//! no hand-rolled code path in this module produces output bytes. What
//! the crate deliberately does NOT provide is the strict-canonical
//! contract: parse_str accepts the loose forms tors exists to reject
//! (braced text, the URN prefix, hyphen-less hex, any-case hex), so
//! strictness remains tors's own thin layer on top -- canonical-encoding
//! equality: a 36-character input is accepted only if it is byte-equal
//! to its parsed value's re-encoded canonical form
//! (`hyphenated().encode_lower`). That one comparison is the whole
//! layer, and the layer is the point: the strict contract is not the
//! crate's default, it is tors's. The error taxonomy (tors's own
//! messages, the first-divergent positions) is irreducible -- the crate
//! exposes no parse-failure positions on its error type -- so the
//! position scan stays too, but it runs only on the rejection path and
//! names a character for a message; it never produces a value. The two
//! field reads stay direct one-liners over the buffer: the crate's
//! `get_version_num` is the identical shift behind a newtype wrap plus a
//! usize cast (nothing complex is delegated by wrapping it), and it has
//! no v7 timestamp accessor (`get_timestamp` is the v1/v6/v7-agnostic
//! seconds+nanos shape, and the unix-millisecond decode underneath it is
//! `pub(crate)`), so the direct big-endian read is the simpler spelling
//! either way. Generation features (v4/v7/rng) are off: this surface
//! never generates an ID, it reads one back out.
//!
//! # Layout (RFC 9562 section 5.7)
//!
//! A UUIDv7 is 16 big-endian bytes: `unix_ts_ms` (48 bits, bytes 0-5, the
//! unix epoch in milliseconds), `ver` (4 bits, byte 6's high half, 7 for
//! v7), `rand_a` (12 bits), `var` (2 bits, byte 8's top half, 0b10 for the
//! RFC 4122 variant), `rand_b` (62 bits). The two field reads here are
//! `unix_ts_ms` and `ver`; `var` is a different field and deliberately out
//! of scope (`uuid_version` answers the version question for any UUID of
//! any variant, the field itself, not an RFC-4122-ness check).
//!
//! # The strict canonical grammar
//!
//! `parse_canonical` accepts exactly one text shape: 36 characters,
//! hyphens at positions 8, 13, 18, 23 (the 8-4-4-4-12 groups), lowercase
//! hexadecimal everywhere else. The stdlib `uuid.UUID` also accepts
//! braced text (`{...}`), the URN prefix (`urn:uuid:...`), hyphen-less
//! hex, and uppercase; tors deliberately does not, the
//! validation-primitive contract: a caller using `uuid_parse` as a gate
//! wants one grammar -- the canonical one, what `str(uuid.UUID(...))` and
//! every RFC 9562 producer emits -- not the stdlib's permissive union
//! (the same closed-set strictness as every other tors argument: the
//! `errors=`/`boundary=` convention, `b64_decode`'s validate-by-default).
//! The mechanics: tors's own length gate first (the exact-count message),
//! the crate's parse second (structure and transcode), the
//! canonical-encoding comparison third (strictness), and
//! `classify_divergence` last, naming the first divergence for the
//! message taxonomy on whatever the crate's parser itself rejected. The
//! rejections are pins, not parity: tests/test_uuid.py's divergence
//! battery proves the stdlib accepts each loose form in the same test
//! that pins tors rejecting it.
//!
//! Pure Rust, no pyo3 types: the pyo3 wrappers in `src/py/uuid.rs` add
//! only the argument dispatch (the bytes-or-str union), the GIL model,
//! and the return marshalling.

use uuid::Uuid;

/// The rejection shapes of `parse_canonical`: one variant per distinct
/// message, `message()` rendering each (the pyo3 layer attaches them to
/// `ValueError`, so `str(exc)` is exactly what `message()` returns; the
/// positions are 0-based, the natural reading for a caller slicing the
/// rejected text).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextError {
    /// Not 36 characters (braces +2, the URN prefix +9, no hyphens -4 all
    /// land here first): "UUID text must be exactly 36 characters (8-4-4-4-12
    /// hyphenated lowercase hex), got N".
    Length(usize),
    /// A hyphen position (8, 13, 18, 23) holding anything but `-`:
    /// "expected '-' at position P, found C".
    MissingHyphen { position: usize, found: char },
    /// A hex position holding `-` (a misplaced or extra hyphen that kept the
    /// length at 36): "found '-' at position P, where a hex digit belongs".
    StrayHyphen { position: usize },
    /// A hex position holding `A`-`F`: the loudest divergence (the stdlib
    /// accepts uppercase, and so does the crate's parser -- only the
    /// canonical-encoding comparison catches it), so the message says so:
    /// "found uppercase C at position P" plus the stdlib note.
    Uppercase { position: usize, found: char },
    /// A hex position holding anything else (including whitespace and
    /// non-ASCII): "found C at position P".
    NotHex { position: usize, found: char },
    /// The drift fallback: a 36-character input the crate's parser rejected
    /// whose divergence the scan could not name. Unreachable under every
    /// `uuid` 1.x grammar measured -- the scan's accepted set (hyphens at
    /// 8/13/18/23, lowercase hex) is a subset of the crate's 36-character
    /// grammar (any-case hex), so a crate rejection always contains a
    /// divergence the scan finds -- kept so a future grammar narrowing
    /// degrades to a ValueError naming the accepted form instead of a
    /// panic (and the round-trip battery fails loudly on any such
    /// narrowing long before this could ship): the accepted form, no
    /// position (none was diagnosable).
    NotCanonical,
}

impl TextError {
    /// The exact `str()` of the `ValueError` the pyo3 layer raises for this
    /// shape. Each names the problem and the accepted form; the uppercase
    /// case additionally names the deliberate stdlib divergence, because a
    /// caller whose text worked with `uuid.UUID` yesterday deserves the
    /// reason, not just the refusal.
    pub fn message(&self) -> String {
        match self {
            TextError::Length(got) => format!(
                "UUID text must be exactly 36 characters \
                 (8-4-4-4-12 hyphenated lowercase hex), got {got}"
            ),
            TextError::MissingHyphen { position, found } => format!(
                "UUID text must be 8-4-4-4-12 hyphenated: \
                 expected '-' at position {position}, found {found:?}"
            ),
            TextError::StrayHyphen { position } => format!(
                "UUID text must be 8-4-4-4-12 hyphenated: \
                 found '-' at position {position}, where a hex digit belongs"
            ),
            TextError::Uppercase { position, found } => format!(
                "UUID text must be lowercase hex: found uppercase {found:?} \
                 at position {position} (the stdlib uuid module accepts \
                 uppercase; tors's canonical form deliberately does not)"
            ),
            TextError::NotHex { position, found } => {
                format!("UUID text must be lowercase hex: found {found:?} at position {position}")
            }
            TextError::NotCanonical => {
                String::from("UUID text must be 8-4-4-4-12 hyphenated lowercase hex")
            }
        }
    }
}

/// The four hyphen positions of the canonical 8-4-4-4-12 form.
const HYPHEN_POSITIONS: [usize; 4] = [8, 13, 18, 23];

/// Canonical UUID text to the 16 raw bytes, strict: exactly 36 characters,
/// hyphens at 8/13/18/23, lowercase hex elsewhere, anything else
/// `TextError` (first offending position wins, scanned left to right).
///
/// The length check is over bytes; a 36-byte input holding any multi-byte
/// UTF-8 character necessarily has fewer than 36 characters and the
/// character itself is rejected at its position before that could matter
/// (a multi-byte char is never `-` or a hex digit, and its start position
/// is always scanned), so accepted input is pure ASCII by construction.
///
/// The acceptance work is split: `Uuid::parse_str` owns the structural
/// parse (hyphen placement, hex digits -- and the transcode: the accepted
/// input's bytes are `*parsed.as_bytes()`, the crate's own decode), and
/// canonical-encoding equality owns the strictness -- the parsed value is
/// re-encoded through `hyphenated().encode_lower` and must be byte-equal
/// to the input. Under the crate's 36-character grammar (hyphens exactly
/// at 8/13/18/23, ASCII hex of either case elsewhere) that comparison
/// rejects exactly one loose form: uppercase hex, which parses but
/// re-encodes lowercase; every other loose form (braces, urn, hyphen-less)
/// is a different length and never reaches it. Rejections route to
/// `classify_divergence` when the crate's parser itself refused, or to the
/// first-mismatch position when the encoding comparison did.
pub fn parse_canonical(text: &str) -> Result<[u8; 16], TextError> {
    if text.len() != 36 {
        return Err(TextError::Length(text.len()));
    }
    let parsed = match Uuid::parse_str(text) {
        Ok(parsed) => parsed,
        Err(_) => return Err(classify_divergence(text)),
    };
    let mut canonical = [0u8; 36];
    let encoded = parsed.hyphenated().encode_lower(&mut canonical);
    let input = text.as_bytes();
    if encoded.as_bytes() != input {
        // The first mismatching index: under the crate's grammar the
        // mismatching byte is an uppercase hex digit (lowercase hex and
        // the hyphens re-encode identically), so this is the Uppercase
        // arm. The catch-all is drift insurance -- if a future uuid 1.x
        // ever let some other byte survive parse_str at 36 characters,
        // the input still gets a ValueError naming the position and
        // character, never a panic.
        let position = encoded
            .as_bytes()
            .iter()
            .zip(input)
            .position(|(canonical, given)| canonical != given)
            .expect("unequal byte slices always diverge");
        // `position` is a byte index into ASCII-only input under the
        // current crate grammar (parse_str accepted 36 bytes of any-case
        // hex + hyphens), so `input[position] as char` would be exact
        // today; decode the character anyway so a future grammar widening
        // cannot turn this into mojibake (a raw byte cast of a multibyte
        // lead byte).
        let found_char = text[position..]
            .chars()
            .next()
            .unwrap_or(char::REPLACEMENT_CHARACTER);
        return Err(match input[position] {
            b'A'..=b'F' => TextError::Uppercase {
                position,
                found: found_char,
            },
            _ => TextError::NotHex {
                position,
                found: found_char,
            },
        });
    }
    Ok(*parsed.as_bytes())
}

/// The first divergence of a 36-character input the crate's parser itself
/// rejected, as the left-to-right single pass the message taxonomy pins:
/// hyphen slots first (`MissingHyphen`), then hex slots (`StrayHyphen`,
/// `Uppercase`, `NotHex`). This scan names a character and position for a
/// message; it produces no bytes (the transcode lives in the crate, and
/// this path only runs on rejection). Its accepted set -- hyphens at
/// 8/13/18/23, lowercase hex elsewhere -- is a subset of the crate's
/// 36-character grammar (the same shape with any-case hex), so a crate
/// rejection always contains a divergence this scan finds; the
/// fall-through arm is unreachable today and is the `NotCanonical` drift
/// fallback, not a panic.
fn classify_divergence(text: &str) -> TextError {
    for (position, c) in text.char_indices() {
        if HYPHEN_POSITIONS.contains(&position) {
            if c != '-' {
                return TextError::MissingHyphen { position, found: c };
            }
        } else {
            match c {
                '-' => return TextError::StrayHyphen { position },
                '0'..='9' | 'a'..='f' => {}
                'A'..='F' => return TextError::Uppercase { position, found: c },
                _ => return TextError::NotHex { position, found: c },
            }
        }
    }
    TextError::NotCanonical
}

/// The version nibble (byte 6's high half), 0-15, for any UUID of any
/// variant: the field itself. The variant (byte 8's top two bits) is a
/// different field and deliberately not consulted. Kept as the direct
/// shift over the buffer rather than the crate's `get_version_num`: that
/// method is this identical shift behind a newtype wrap plus a usize cast
/// (see the module docs -- no complexity is delegated by the wrap), and
/// the u48 timestamp read below stays direct anyway (the crate has no
/// public v7 accessor), so one spelling style covers both reads.
pub fn version_nibble(bytes: &[u8; 16]) -> u8 {
    bytes[6] >> 4
}

/// The 48-bit big-endian unix-millisecond field (bytes 0-5) of a UUIDv7,
/// `Err(found_version)` when the version nibble is not 7: the field is only
/// defined for v7, and the error carries the version actually found so the
/// pyo3 layer's ValueError can name it.
pub fn v7_timestamp_ms(bytes: &[u8; 16]) -> Result<u64, u8> {
    let version = version_nibble(bytes);
    if version != 7 {
        return Err(version);
    }
    Ok(timestamp_field(bytes))
}

/// The leading six bytes as a big-endian u48, the field read itself
/// (shared by the v7 path and the crate-side tests' independent spelling).
fn timestamp_field(bytes: &[u8; 16]) -> u64 {
    u64::from_be_bytes([
        0, 0, bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixed UUIDv7 of the Python-side doc-example pin (timestamp field
    /// 1_750_000_000_000 ms, version 7, RFC 4122 variant), one shared
    /// known-answer vector on both sides of the FFI.
    const DOC_V7_TEXT: &str = "01977420-dc00-7abc-9def-98765432100f";
    const DOC_V7_BYTES: [u8; 16] = [
        0x01, 0x97, 0x74, 0x20, 0xdc, 0x00, 0x7a, 0xbc, 0x9d, 0xef, 0x98, 0x76, 0x54, 0x32, 0x10,
        0x0f,
    ];

    /// An RFC 9562 section 5.7 UUIDv7 from its parts, the same layout the
    /// Python-side oracle builders use (the independent-agreement argument:
    /// two spellings of the layout, one on each side of the FFI).
    fn v7(ms: u64, rand_a: u16, rand_b: u64) -> [u8; 16] {
        let value: u128 = ((ms as u128) << 80)
            | (7u128 << 76)
            | ((rand_a as u128) << 64)
            | (0b10u128 << 62)
            | (rand_b as u128);
        value.to_be_bytes()
    }

    #[test]
    fn parse_of_the_doc_vector_is_its_bytes() {
        assert_eq!(parse_canonical(DOC_V7_TEXT).unwrap(), DOC_V7_BYTES);
    }

    #[test]
    fn version_and_timestamp_of_the_doc_vector() {
        assert_eq!(version_nibble(&DOC_V7_BYTES), 7);
        assert_eq!(v7_timestamp_ms(&DOC_V7_BYTES).unwrap(), 1_750_000_000_000);
    }

    #[test]
    fn v7_round_trip_over_the_field_range() {
        // Boundaries and a middle: the 48-bit field's whole range.
        for &ms in &[0u64, 1, 1_750_000_000_000, (1 << 48) - 1] {
            let bytes = v7(ms, 0xABC, 0x1DEF_9876_5432_100F);
            assert_eq!(v7_timestamp_ms(&bytes).unwrap(), ms);
        }
    }

    #[test]
    fn timestamp_field_agrees_with_byte_at_a_time_shifts() {
        // The independent spelling of the read: u64 shifts, no from_be_bytes.
        let bytes = v7(0x0123_4567_89AB, 0, 0);
        let expected = (bytes[0] as u64) << 40
            | (bytes[1] as u64) << 32
            | (bytes[2] as u64) << 24
            | (bytes[3] as u64) << 16
            | (bytes[4] as u64) << 8
            | bytes[5] as u64;
        assert_eq!(timestamp_field(&bytes), expected);
        assert_eq!(timestamp_field(&bytes), 0x0123_4567_89AB);
    }

    #[test]
    fn every_version_nibble_round_trips() {
        for version in 0u8..=15 {
            let mut bytes = [0u8; 16];
            bytes[6] = version << 4;
            assert_eq!(version_nibble(&bytes), version);
        }
    }

    #[test]
    fn non_v7_versions_error_with_the_found_version() {
        for version in 0u8..=15 {
            if version == 7 {
                continue;
            }
            let mut bytes = [0u8; 16];
            bytes[6] = version << 4;
            assert_eq!(v7_timestamp_ms(&bytes), Err(version));
        }
    }

    #[test]
    fn nil_and_max_uuids() {
        // Nil: version 0, so the guard fires rather than answering a
        // meaningless 0 timestamp. Max: version 15.
        assert_eq!(version_nibble(&[0u8; 16]), 0);
        assert_eq!(v7_timestamp_ms(&[0u8; 16]), Err(0));
        let max = [0xFFu8; 16];
        assert_eq!(version_nibble(&max), 15);
        assert_eq!(v7_timestamp_ms(&max), Err(15));
    }

    #[test]
    fn parse_accepts_every_canonical_rendering_of_arbitrary_bytes() {
        // str(uuid.UUID(bytes=b)) is canonical for any 16 bytes (pinned on
        // the Python side); the formatter over any bytes is
        // const_hex::encode + the four hyphens, so every buffer's rendering
        // must parse back.
        let canonical = |bytes: &[u8; 16]| {
            let hex = const_hex::encode(bytes);
            format!(
                "{}-{}-{}-{}-{}",
                &hex[0..8],
                &hex[8..12],
                &hex[12..16],
                &hex[16..20],
                &hex[20..32]
            )
        };
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..200 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut bytes = [0u8; 16];
            for (i, slot) in bytes.iter_mut().enumerate() {
                *slot = (state >> ((i % 8) * 8)) as u8;
            }
            let text = canonical(&bytes);
            assert_eq!(parse_canonical(&text).unwrap(), bytes, "{text}");
        }
    }

    #[test]
    fn every_rejection_class_pin() {
        // One pin per TextError variant, message-exact (the pyo3 layer
        // attaches message() to ValueError verbatim; these are the strings
        // the Python-side match= pins grep for).
        let cases: &[(&str, TextError)] = &[
            ("", TextError::Length(0)),
            ("01977420-dc00-7abc-9def-98765432100", TextError::Length(35)),
            (
                "01977420-dc00-7abc-9def-98765432100f0",
                TextError::Length(37),
            ),
            (
                "01977420-dc0007abc-9def-98765432100f",
                TextError::MissingHyphen {
                    position: 13,
                    found: '0',
                },
            ),
            (
                "-1977420-dc00-7abc-9def-98765432100f",
                TextError::StrayHyphen { position: 0 },
            ),
            (
                "01977420-dc00-7abc-9def-98765432100F",
                TextError::Uppercase {
                    position: 35,
                    found: 'F',
                },
            ),
            (
                "01977420-dc0g-7abc-9def-98765432100f",
                TextError::NotHex {
                    position: 12,
                    found: 'g',
                },
            ),
            (
                "0197742 -dc00-7abc-9def-98765432100f",
                TextError::NotHex {
                    position: 7,
                    found: ' ',
                },
            ),
            (
                "01977\x0020-dc00-7abc-9def-98765432100f",
                TextError::NotHex {
                    position: 5,
                    found: '\0',
                },
            ),
            (
                "01977\r20-dc00-7abc-9def-98765432100f",
                TextError::NotHex {
                    position: 5,
                    found: '\r',
                },
            ),
            (
                "01977\t20-dc00-7abc-9def-98765432100f",
                TextError::NotHex {
                    position: 5,
                    found: '\t',
                },
            ),
            (
                "g1977420-dc00-7abc-9def-98765432100f",
                TextError::NotHex {
                    position: 0,
                    found: 'g',
                },
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(parse_canonical(text).unwrap_err(), *expected, "{text:?}");
        }
    }

    #[test]
    fn confusable_non_ascii_is_length_gated() {
        // Fullwidth "ａ" (U+FF41, 3 bytes) and Cyrillic "а" (U+0430,
        // 2 bytes) look like "a" but are not ASCII: a 36-character text
        // holding one is 38/37 bytes, so the byte-counting length gate
        // fires before any character classification.
        let fullwidth = "01977ａ20-dc00-7abc-9def-98765432100f";
        assert_eq!(fullwidth.chars().count(), 36);
        assert_eq!(fullwidth.len(), 38);
        assert_eq!(
            parse_canonical(fullwidth).unwrap_err(),
            TextError::Length(38)
        );
        let cyrillic = "01977а20-dc00-7abc-9def-98765432100f";
        assert_eq!(cyrillic.chars().count(), 36);
        assert_eq!(cyrillic.len(), 37);
        assert_eq!(
            parse_canonical(cyrillic).unwrap_err(),
            TextError::Length(37)
        );
    }

    #[test]
    fn multibyte_at_each_hyphen_slot_is_a_missing_hyphen() {
        // "é" (2 bytes) at each hyphen slot in a 36-BYTE text
        // (35 characters): the hyphen-slot check names the byte position
        // and the found character.
        for &position in &[8usize, 13, 18, 23] {
            let prefix = &DOC_V7_TEXT[..position];
            let suffix = &DOC_V7_TEXT[position + 1..];
            // Drop the last byte to hold 36 bytes total.
            let text = format!("{}é{}", prefix, &suffix[..suffix.len() - 1]);
            assert_eq!(text.len(), 36, "{text:?}");
            assert_eq!(
                parse_canonical(&text).unwrap_err(),
                TextError::MissingHyphen {
                    position,
                    found: 'é'
                },
                "{text:?}"
            );
        }
    }

    #[test]
    fn uppercase_at_index_zero_reports_its_own_position() {
        // The encoding-comparison route (crate accepts uppercase) decodes
        // the found character rather than casting the raw byte, so the
        // position-0 uppercase vector is exact even under a future
        // grammar widening.
        let text = format!("D{}", &DOC_V7_TEXT[1..]);
        assert_eq!(
            parse_canonical(&text).unwrap_err(),
            TextError::Uppercase {
                position: 0,
                found: 'D'
            }
        );
    }

    #[test]
    fn rejection_messages_name_the_problem_and_the_accepted_form() {
        assert_eq!(
            parse_canonical("01977420-dc00-7abc-9def-98765432100")
                .unwrap_err()
                .message(),
            "UUID text must be exactly 36 characters (8-4-4-4-12 hyphenated lowercase \
             hex), got 35"
        );
        assert_eq!(
            parse_canonical("01977420-dc0007abc-9def-98765432100f")
                .unwrap_err()
                .message(),
            "UUID text must be 8-4-4-4-12 hyphenated: expected '-' at position 13, found '0'"
        );
        assert_eq!(
            parse_canonical("-1977420-dc00-7abc-9def-98765432100f")
                .unwrap_err()
                .message(),
            "UUID text must be 8-4-4-4-12 hyphenated: found '-' at position 0, \
             where a hex digit belongs"
        );
        assert_eq!(
            parse_canonical("01977420-dc00-7abc-9def-98765432100F")
                .unwrap_err()
                .message(),
            "UUID text must be lowercase hex: found uppercase 'F' at position 35 \
             (the stdlib uuid module accepts uppercase; tors's canonical form \
             deliberately does not)"
        );
        assert_eq!(
            parse_canonical("01977420-dc0g-7abc-9def-98765432100f")
                .unwrap_err()
                .message(),
            "UUID text must be lowercase hex: found 'g' at position 12"
        );
    }

    #[test]
    fn multibyte_characters_are_rejected_at_their_start_position() {
        // A 36-byte string holding a two-byte character: the character is
        // not hex, and its start position (a hex slot here: group 3's
        // second digit) is what the error names. Braces/urn never reach
        // this class (wrong length fires first), but a 36-byte crafted
        // input can.
        let text = "01977420-dc00-7ébc-9def-9876543210f";
        assert_eq!(text.len(), 36, "the fixture itself");
        assert_eq!(
            parse_canonical(text).unwrap_err(),
            TextError::NotHex {
                position: 15,
                found: 'é'
            }
        );
    }

    #[test]
    fn parse_is_total_no_panic_shapes() {
        // A fixed-corpus sweep over adversarial shapes that must all be
        // rejections, never panics: every prefix of the canonical text,
        // every single-position mutation class, and the stdlib's loose
        // forms.
        let mut shapes: Vec<String> = Vec::new();
        for end in 0..=DOC_V7_TEXT.len() {
            shapes.push(DOC_V7_TEXT[..end].to_string());
        }
        for position in 0..DOC_V7_TEXT.len() {
            let mut chars: Vec<char> = DOC_V7_TEXT.chars().collect();
            chars[position] = if chars[position] == '-' { '0' } else { '-' };
            shapes.push(chars.into_iter().collect());
        }
        shapes.push(format!("{{{DOC_V7_TEXT}}}"));
        shapes.push(format!("urn:uuid:{DOC_V7_TEXT}"));
        shapes.push(DOC_V7_TEXT.replace('-', ""));
        shapes.push(DOC_V7_TEXT.to_uppercase());
        for shape in &shapes {
            // Every shape here is either a clean rejection or (the
            // hyphen-flip cases can land back on canonical only by flipping
            // a hyphen to a hyphen, which this loop never does) a parse;
            // either way, no input may panic.
            let _ = parse_canonical(shape);
        }
    }

    // --- the adoption boundary pins ---------------------------------------
    //
    // The crate adoption's own regression net: these call the `uuid` crate
    // DIRECTLY to prove which side of the parse_str/strictness-layer
    // boundary each input class lands on, so a future uuid 1.x that moves
    // the boundary (a grammar change in either direction) fails HERE
    // first, with a message naming the boundary, instead of silently
    // moving rejections between tors's routes.

    #[test]
    fn the_crate_parses_what_the_strictness_layer_rejects_uppercase() {
        // The strictness layer's whole job in one pin: uppercase hex is
        // INSIDE the crate's grammar (parse_str accepts it -- proven by
        // calling the crate) and OUTSIDE tors's (the canonical-encoding
        // comparison rejects it). Uppercase is the only loose form that is
        // 36 characters, so it is the only one that reaches the comparison
        // at all; braces/urn/simple are length-gated before it.
        let upper = DOC_V7_TEXT.to_uppercase();
        assert!(Uuid::parse_str(&upper).is_ok());
        assert_eq!(
            parse_canonical(&upper).unwrap_err(),
            TextError::Uppercase {
                position: 9,
                found: 'D'
            }
        );
    }

    #[test]
    fn the_crate_parses_the_wrapper_forms_tors_length_gates() {
        // The other three stdlib loose forms, each proven crate-accepted
        // and tors-rejected through the length gate (38/45/32): the route
        // split of the strictness layer, pinned so a uuid 1.x grammar
        // change that somehow made one of these 36 characters (or stopped
        // parsing one) is caught as a boundary move, not a silent pass.
        let braced = format!("{{{DOC_V7_TEXT}}}");
        let urn = format!("urn:uuid:{DOC_V7_TEXT}");
        let simple = DOC_V7_TEXT.replace('-', "");
        assert!(Uuid::parse_str(&braced).is_ok());
        assert!(Uuid::parse_str(&urn).is_ok());
        assert!(Uuid::parse_str(&simple).is_ok());
        assert_eq!(parse_canonical(&braced).unwrap_err(), TextError::Length(38));
        assert_eq!(parse_canonical(&urn).unwrap_err(), TextError::Length(45));
        assert_eq!(parse_canonical(&simple).unwrap_err(), TextError::Length(32));
    }

    #[test]
    fn every_uppercase_position_is_reported_at_its_own_index() {
        // One mutated letter position per group (skipping digits, whose
        // uppercase is themselves, and the hyphen slots): each reports its
        // own index, the first-mismatch semantics of the encoding
        // comparison.
        for &position in &[9, 10, 15, 16, 17, 20, 21, 22, 35] {
            let mut chars: Vec<char> = DOC_V7_TEXT.chars().collect();
            chars[position] = chars[position].to_ascii_uppercase();
            let found = chars[position];
            let text: String = chars.into_iter().collect();
            assert_eq!(
                parse_canonical(&text).unwrap_err(),
                TextError::Uppercase { position, found },
                "{text}"
            );
        }
    }

    #[test]
    fn everything_the_strict_scan_accepts_the_crate_accepts() {
        // The subset invariant classify_divergence's soundness rests on:
        // canonical renderings (what the strict scan accepts) are all
        // inside the crate's permissive grammar. If a future uuid 1.x
        // ever NARROWED its grammar below tors's, this fails first -- and
        // the round-trip test above fails with it -- long before the
        // NotCanonical drift fallback could be reached in production.
        let canonical = |bytes: &[u8; 16]| {
            let hex = const_hex::encode(bytes);
            format!(
                "{}-{}-{}-{}-{}",
                &hex[0..8],
                &hex[8..12],
                &hex[12..16],
                &hex[16..20],
                &hex[20..32]
            )
        };
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..200 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut bytes = [0u8; 16];
            for (i, slot) in bytes.iter_mut().enumerate() {
                *slot = (state >> ((i % 8) * 8)) as u8;
            }
            let text = canonical(&bytes);
            assert!(Uuid::parse_str(&text).is_ok(), "{text}");
        }
    }

    #[test]
    fn the_drift_fallback_message_names_the_accepted_form() {
        // NotCanonical is unreachable through parse_canonical under every
        // measured grammar (the subset pin above); pin its message anyway
        // so the fallback, if a future grammar narrowing ever reaches it,
        // degrades to a precise ValueError rather than an unpinned string.
        assert_eq!(
            TextError::NotCanonical.message(),
            "UUID text must be 8-4-4-4-12 hyphenated lowercase hex"
        );
    }
}
