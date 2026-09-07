//! Phonetic-code primitives: the pure-Rust cores of `tors.soundex`,
//! `tors.metaphone`, `tors.double_metaphone`, `tors.nysiis`, and
//! `tors.daitch_mokotoff`, wrapping the `rphonetic` crate (an Apache
//! Commons Codec port). All are classic, ENGLISH/Latin-script-specific
//! heuristics from the phonetic-matching literature (Soundex: 1918
//! patent; NYSIIS: New York State Identification and Intelligence
//! System, 1970; Double Metaphone: Lawrence Philips, 2000;
//! Daitch-Mokotoff: 1985, designed for Central/Eastern European
//! surnames); not general Unicode phonetic analysis. They exist
//! alongside `levenshtein`/`jaro_winkler` for the same lane: fast,
//! deterministic, no-ML-model fuzzy name/word matching: phonetic codes
//! group words that SOUND alike before or alongside an edit-distance
//! score, the standard combination in name-matching/dedup pipelines.
//!
//! All codes are computed over the input's ASCII LETTERS ONLY: none of
//! the algorithms is Unicode-aware beyond that (per Apache Commons
//! Codec, which `rphonetic` faithfully ports, itself
//! ASCII/English-oriented). This is enforced explicitly by
//! [`ascii_alphabetic`], not left to the crate: a **real, verified
//! upstream defect** in `rphonetic` 4.0.0 makes both `Soundex::encode`
//! and `DoubleMetaphone::encode` PANIC on ordinary accented input:
//! `Soundex`'s "clean" step filters by Unicode `char::is_alphabetic`
//! (too broad: Cyrillic, CJK, Greek, and accented Latin like `'é'` all
//! pass it) and then unconditionally indexes a 26-element mapping table
//! with `ch as usize - 65`, out of bounds for anything outside plain
//! ASCII `A`-`Z`; `DoubleMetaphone` separately panics on multi-byte
//! characters via a byte-index slice that assumes one byte per
//! character. Confirmed empirically (not merely read from source):
//! `Soundex::default().encode("José")`,
//! `DoubleMetaphone::default().encode("café")`, and
//! `DoubleMetaphone::default().encode("Björk")` all panic: i.e. this
//! breaks on exactly the realistic accented-name inputs a
//! name-matching consumer would actually pass. tors NEVER lets a Rust
//! panic reach Python, so every call here pre-filters to ASCII letters
//! FIRST: the same filter the crate's own "clean" step should have
//! applied, and the already-documented scope of these algorithms
//! regardless (non-ASCII input carries no phonetic meaning to a
//! classic English-letters algorithm even when it doesn't crash). The
//! filter is load-bearing for the newer three in different ways than
//! panic-avoidance: `Nysiis`'s own clean step uses the same too-broad
//! `is_alphabetic` test, so accented letters would otherwise leak
//! verbatim INTO the returned code (a non-ASCII NYSIIS key, useless to
//! a grouping consumer); `DaitchMokotoffSoundex` silently skips any
//! character with no rule (harmless, but inconsistent) while its
//! in-crate ASCII folding only covers the characters the rule table
//! lists. Callers working with non-English/accented names should not
//! expect these to behave like general Unicode phonetic algorithms,
//! because they are not: they degrade (dropping accents/non-Latin
//! characters) rather than crashing.

use rphonetic::{DaitchMokotoffSoundex, DoubleMetaphone, Encoder, Nysiis, RefinedSoundex, Soundex};

/// The panic-avoidance filter both functions apply before calling into
/// `rphonetic`: see the module docs for why this is load-bearing, not
/// cosmetic. `char::is_ascii_alphabetic` (not `is_alphabetic`, which is
/// what the crate's own internal cleaning uses and which is exactly the
/// bug) guarantees every character `rphonetic` receives maps cleanly
/// into its ASCII `A`-`Z` mapping tables, and guarantees single-byte
/// characters throughout the string (avoiding `DoubleMetaphone`'s
/// separate byte-index-slicing panic on multi-byte UTF-8).
fn ascii_alphabetic(text: &str) -> String {
    text.chars().filter(char::is_ascii_alphabetic).collect()
}

/// `rphonetic::Soundex::default()`: the classic 1-letter-plus-3-digit
/// Soundex code (`"Robert"` and `"Rupert"` both encode to `"R163"`).
/// Deterministic, stateless per call (the encoder itself carries no
/// state beyond its fixed mapping table, so constructing it per call is
/// O(1) and adds no measurable cost). Input is pre-filtered to ASCII
/// letters: see the module docs.
pub fn soundex(text: &str) -> String {
    Soundex::default().encode(&ascii_alphabetic(text))
}

/// `rphonetic::DoubleMetaphone::default()`'s PRIMARY code: Lawrence
/// Philips' 2000 successor to classic Metaphone, chosen over the
/// original Metaphone as the single exposed algorithm here because it is
/// the more accurate, more widely-used modern default (and `rphonetic`
/// exposes both through the identical `Encoder::encode(&str) -> String`
/// shape, so there is no cost to picking the better one). Double
/// Metaphone can also produce an ALTERNATE code for words with two
/// plausible pronunciations: this function returns only the primary
/// code, matching the scope of a single deterministic string-in
/// string-out primitive; `rphonetic`'s `encode_alternate` is not
/// exposed in this first pass (no consumer has asked for it, and adding
/// it later is a compatible, additive change, not a breaking one). Input
/// is pre-filtered to ASCII letters: see the module docs.
pub fn metaphone(text: &str) -> String {
    DoubleMetaphone::default().encode(&ascii_alphabetic(text))
}

/// `rphonetic::DoubleMetaphone::default()`'s PRIMARY and ALTERNATE codes
/// as a pair. Lawrence Philips' 2000 successor to classic Metaphone; the
/// alternate code is the algorithm's distinguishing feature: for words
/// with two plausible pronunciations (typically a name that could be
/// read in a Germanic/Slavic way or an Anglicized way) it emits a second
/// key, so a match on EITHER key of two names counts as a phonetic
/// match. `tors.metaphone` exposes only the primary; this function
/// exposes both because that dual-key matching is the point of the
/// algorithm and the obvious Python shape for it is a 2-tuple
/// `(primary, alternate)` (for words with one pronunciation the two
/// elements are equal). Input is pre-filtered to ASCII letters: see the
/// module docs.
pub fn double_metaphone(text: &str) -> (String, String) {
    let result = DoubleMetaphone::default().double_metaphone(&ascii_alphabetic(text));
    (result.primary(), result.alternate())
}

/// `rphonetic::Nysiis::default()` (the strict commons-codec variant:
/// codes capped at 6 characters). NYSIIS, the New York State
/// Identification and Intelligence System 1970 algorithm, a Soundex
/// successor with better first-letter and vowel handling (e.g.
/// `"Washington"` encodes to `"WASANG"`). Input is pre-filtered to ASCII
/// letters: see the module docs.
pub fn nysiis(text: &str) -> String {
    Nysiis::default().encode(&ascii_alphabetic(text))
}

/// `rphonetic::DaitchMokotoffSoundex::default()` (commons-codec rules,
/// ASCII folding on) with BRANCHING: the 1985 Daitch-Mokotoff Soundex,
/// the standard code of Jewish-genealogy surname matching, designed for
/// the Central/Eastern European surnames classic Soundex handles poorly
/// (its digit table distinguishes sounds Soundex conflates, e.g.
/// guttural vs sibilant). Its rule table BRANCHES on ambiguous
/// transliterations (a Cyrillic-derived spelling can transliterate
/// multiple ways), so one name can legitimately encode to SEVERAL
/// 6-digit codes: two names match if ANY of their codes intersect.
/// That multiple-candidate reality is why the return is a list, not a
/// string: `inner_soundex(_, true)` gives the honest per-candidate
/// shape without a `'|'`-joined round-trip through string parsing.
/// Each code is padded to 6 digits (a name with no encodable letters
/// yields the all-padding code, pinned by the tests). Input is
/// pre-filtered to ASCII letters: see the module docs.
pub fn daitch_mokotoff(text: &str) -> Vec<String> {
    DaitchMokotoffSoundex::default().inner_soundex(&ascii_alphabetic(text), true)
}

/// `rphonetic::RefinedSoundex::default()`: a Soundex variant with a
/// different (finer-grained) letter-to-digit mapping table than classic
/// Soundex, giving better discrimination between names Soundex conflates
/// (its digit alphabet distinguishes more consonant classes, at the cost
/// of longer, non-length-capped codes rather than Soundex's fixed
/// 1-letter-plus-3-digit shape). A distinct algorithm, not a formatting
/// variant of `soundex`: confirmed empirically (`ENGLISH_MAPPING` differs
/// from Soundex's own `DEFAULT_US_ENGLISH_MAPPING_SOUNDEX` table). Same
/// upstream panic class as classic `Soundex` (verified directly against
/// the raw crate: `RefinedSoundex::default().encode("José")` panics with
/// an out-of-bounds table index, the identical unguarded `ch as usize -
/// 65` bug) — input is pre-filtered to ASCII letters for the same reason:
/// see the module docs.
pub fn refined_soundex(text: &str) -> String {
    RefinedSoundex::default().encode(&ascii_alphabetic(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soundex_groups_classic_homophone_pairs() {
        assert_eq!(soundex("Robert"), soundex("Rupert"));
        assert_eq!(soundex("Robert"), "R163");
    }

    #[test]
    fn soundex_textbook_vector() {
        assert_eq!(soundex("jumped"), "J513");
    }

    #[test]
    fn metaphone_textbook_vector() {
        assert_eq!(metaphone("jumped"), "JMPT");
    }

    #[test]
    fn metaphone_groups_smith_and_smyth() {
        assert_eq!(metaphone("Smith"), metaphone("Smyth"));
    }

    #[test]
    fn empty_input_is_empty_output() {
        assert_eq!(soundex(""), "");
        assert_eq!(metaphone(""), "");
    }

    #[test]
    fn non_alphabetic_and_non_ascii_input_does_not_panic() {
        // Confirms the ascii_alphabetic pre-filter is doing its job: none
        // of these carry a meaningful phonetic code, but none may panic
        // either. "12345"/"!!!" have zero letters -> "" (rphonetic's own
        // empty-input answer). "日本語" has zero ASCII letters -> "" too.
        assert_eq!(soundex("12345"), "");
        assert_eq!(metaphone("12345"), "");
        assert_eq!(soundex("日本語"), "");
        assert_eq!(metaphone("日本語"), "");
        assert_eq!(soundex("!!!"), "");
        assert_eq!(metaphone("!!!"), "");
    }

    #[test]
    fn accented_names_do_not_panic_upstream_bug_regression() {
        // THE regression this module's pre-filter exists for: rphonetic
        // 4.0.0's Soundex::encode and DoubleMetaphone::encode both panic
        // on these exact inputs when called directly (verified against
        // the raw crate, not assumed): ordinary accented names, not
        // adversarial input. Accents are dropped by the ASCII filter
        // (documented degradation), not crashed on.
        assert_eq!(soundex("José"), "J200");
        assert_eq!(soundex("café"), "C100");
        assert_eq!(soundex("Björk"), "B262");
        assert_eq!(metaphone("José"), "JS");
        assert_eq!(metaphone("café"), "KF");
        assert_eq!(metaphone("Björk"), "PJRK");
    }

    #[test]
    fn deterministic_across_repeated_calls() {
        assert_eq!(soundex("Ashcraft"), soundex("Ashcraft"));
        assert_eq!(metaphone("Ashcraft"), metaphone("Ashcraft"));
    }

    // Vectors below were DERIVED by probing the built extension (and
    // cross-checked against the vector tables in rphonetic's own test
    // suite, which ports Apache Commons Codec's test data verbatim):
    // no value below is pinned from memory or from the algorithm papers
    // alone.

    /// `assert_eq!` on a `(String, String)` against `(&str, &str)`
    /// literals: tuple `PartialEq` needs both sides same-typed.
    fn assert_dm(text: &str, expected: (&str, &str)) {
        let result = double_metaphone(text);
        assert_eq!((result.0.as_str(), result.1.as_str()), expected, "{text}");
    }

    #[test]
    fn double_metaphone_textbook_vector() {
        assert_dm("jumped", ("JMPT", "AMPT"));
    }

    #[test]
    fn double_metaphone_alternate_key_matches_schmidt_primary() {
        // The algorithm's headline demonstration, only expressible in
        // the dual-key form: Smith and Schmidt match CROSS-KEY (Smith's
        // alternate equals Schmidt's primary), not primary-to-primary.
        assert_dm("Smith", ("SM0", "XMT"));
        assert_dm("Schmidt", ("XMT", "SMT"));
    }

    #[test]
    fn double_metaphone_groups_smith_and_smythe_on_both_keys() {
        assert_eq!(double_metaphone("Smith"), double_metaphone("Smythe"));
    }

    #[test]
    fn double_metaphone_single_pronunciation_words_have_equal_keys() {
        // The other half of the pair contract: no second pronunciation
        // means the alternate degenerates to the primary.
        assert_dm("Alexander", ("ALKS", "ALKS"));
        assert_dm("cabrillo", ("KPRL", "KPR"));
    }

    #[test]
    fn double_metaphone_empty_input_is_empty_pair() {
        assert_dm("", ("", ""));
    }

    #[test]
    fn nysiis_literature_grouping_rows() {
        // The classic commons-codec test-table grouping rows: several
        // spellings, one NYSIIS code.
        for word in ["Brian", "Brown", "Brun"] {
            assert_eq!(nysiis(word), "BRAN", "{word}");
        }
        for word in ["Capp", "Cope", "Kipp"] {
            assert_eq!(nysiis(word), "CAP", "{word}");
        }
        for word in ["Dane", "Dean", "Dionne"] {
            assert_eq!(nysiis(word), "DAN", "{word}");
        }
        assert_eq!(nysiis("Dent"), "DAD");
        assert_eq!(nysiis("Phil"), "FAL");
    }

    #[test]
    fn nysiis_strict_westerlund_and_washington() {
        // "Westerlund" -> "WASTAR" is the crate's own strict-mode
        // doctest vector. "Washington" -> "WASANG" is hand-traced from
        // the published procedure (the H-after-S step duplicates the S,
        // and strict mode caps at 6 characters: "WASANGT" without the
        // cap) and probe-confirmed; the oft-quoted "WASAN" does not
        // come out of the published procedure as reproduced by
        // commons-codec/rphonetic.
        assert_eq!(nysiis("Westerlund"), "WASTAR");
        assert_eq!(nysiis("Washington"), "WASANG");
    }

    #[test]
    fn nysiis_schmidt_and_smith_differ() {
        assert_eq!(nysiis("Schmidt"), "SNAD");
        assert_eq!(nysiis("Smith"), "SNAT");
    }

    #[test]
    fn nysiis_empty_input_is_empty_output() {
        assert_eq!(nysiis(""), "");
    }

    #[test]
    fn daitch_mokotoff_documented_examples() {
        // The commons-codec DaitchMokotoffSoundex test table (ported
        // verbatim by rphonetic, probe-confirmed here): the famous
        // same-surname pairs collapse to equal or intersecting code
        // lists.
        assert_eq!(daitch_mokotoff("AUERBACH"), vec!["097400", "097500"]);
        assert_eq!(daitch_mokotoff("OHRBACH"), vec!["097400", "097500"]);
        assert_eq!(daitch_mokotoff("LIPSHITZ"), vec!["874400"]);
        assert_eq!(daitch_mokotoff("LIPPSZYC"), vec!["874400", "874500"]);
        assert_eq!(daitch_mokotoff("Moskowitz"), vec!["645740"]);
        assert_eq!(daitch_mokotoff("Moskovitz"), vec!["645740"]);
        assert_eq!(
            daitch_mokotoff("Jackson"),
            vec!["154600", "145460", "454600", "445460"]
        );
    }

    #[test]
    fn daitch_mokotoff_heavy_branching_example() {
        // The crate's own showcase for rule-table branching: one Polish
        // spelling, eight candidate codes.
        assert_eq!(
            daitch_mokotoff("Rosochowaciec"),
            vec![
                "944744", "944745", "944754", "944755", "945744", "945745", "945754", "945755",
            ]
        );
    }

    #[test]
    fn daitch_mokotoff_branch_lists_intersect_across_transliterations() {
        // The matching rule the list shape exists for: the Anglicized
        // spelling's single code is one of the original spelling's
        // candidates.
        let original = daitch_mokotoff("Rosochowaciec");
        let anglicized = daitch_mokotoff("Rosokhovatsets");
        assert_eq!(anglicized, vec!["945744"]);
        assert!(original.iter().any(|code| anglicized.contains(code)));
    }

    #[test]
    fn daitch_mokotoff_no_encodable_letters_is_all_padding() {
        // Unlike the string-returning algorithms, DM pads every code to
        // 6 digits, so letterless input is the all-padding code, not
        // "". Pinned crate behavior.
        assert_eq!(daitch_mokotoff(""), vec!["000000"]);
        assert_eq!(daitch_mokotoff("12345"), vec!["000000"]);
        assert_eq!(daitch_mokotoff("日本語"), vec!["000000"]);
    }

    #[test]
    fn new_algorithms_degrade_accented_input_like_the_existing_pair() {
        // The same partial-degradation contract as soundex/metaphone's
        // "Müller" -> "Mller": accents are dropped, the ASCII letters
        // keep their code. For DM this deliberately BYPASSES the
        // crate's own in-crate ASCII folding (e.g. raw
        // DaitchMokotoffSoundex maps "ţamas" to "364000|464000"; tors
        // drops the ţ first, so "ţamas" encodes like "amas"): the
        // uniform ASCII-letters-only scope documented at the top of
        // this module wins over per-algorithm Unicode behavior.
        assert_eq!(double_metaphone("José"), double_metaphone("Jos"));
        assert_eq!(nysiis("Müller"), nysiis("Mller"));
        assert_eq!(daitch_mokotoff("Müller"), daitch_mokotoff("Mller"));
        assert_eq!(daitch_mokotoff("ţamas"), daitch_mokotoff("amas"));
    }

    #[test]
    fn new_algorithms_deterministic_across_repeated_calls() {
        assert_eq!(double_metaphone("Ashcraft"), double_metaphone("Ashcraft"));
        assert_eq!(nysiis("Ashcraft"), nysiis("Ashcraft"));
        assert_eq!(daitch_mokotoff("Ashcraft"), daitch_mokotoff("Ashcraft"));
    }
}
