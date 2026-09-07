//! Phonetic-code primitives: the pure-Rust cores of `tors.soundex` and
//! `tors.metaphone`, wrapping the `rphonetic` crate (an Apache Commons
//! Codec port). Both algorithms are classic, ENGLISH/Latin-script-specific
//! heuristics from the phonetic-matching literature (Soundex: 1918 patent;
//! Double Metaphone: Lawrence Philips, 2000); not general Unicode phonetic
//! analysis. They exist alongside `levenshtein`/`jaro_winkler` for the same
//! lane: fast, deterministic, no-ML-model fuzzy name/word matching:
//! phonetic codes group words that SOUND alike before or alongside an
//! edit-distance score, the standard combination in name-matching/dedup
//! pipelines.
//!
//! Both codes are computed over the input's ASCII LETTERS ONLY: neither
//! algorithm is Unicode-aware beyond that (per Apache Commons Codec, which
//! `rphonetic` faithfully ports, itself ASCII/English-oriented). This is
//! enforced explicitly by [`ascii_alphabetic`], not left to the crate: a
//! **real, verified upstream defect** in `rphonetic` 4.0.0 makes both
//! `Soundex::encode` and `DoubleMetaphone::encode` PANIC on ordinary
//! accented input: `Soundex`'s "clean" step filters by Unicode
//! `char::is_alphabetic` (too broad: Cyrillic, CJK, Greek, and accented
//! Latin like `'é'` all pass it) and then unconditionally indexes a
//! 26-element mapping table with `ch as usize - 65`, out of bounds for
//! anything outside plain ASCII `A`-`Z`; `DoubleMetaphone` separately
//! panics on multi-byte characters via a byte-index slice that assumes
//! one byte per character. Confirmed empirically (not merely read from
//! source): `Soundex::default().encode("José")`,
//! `DoubleMetaphone::default().encode("café")`, and
//! `DoubleMetaphone::default().encode("Björk")` all panic: i.e. this
//! breaks on exactly the realistic accented-name inputs a name-matching
//! consumer would actually pass. tors NEVER lets a Rust panic reach
//! Python, so every call here pre-filters to ASCII letters FIRST: the
//! same filter the crate's own "clean" step should have applied, and the
//! already-documented scope of these algorithms regardless
//! (non-ASCII input carries no phonetic meaning to a classic
//! English-letters algorithm even when it doesn't crash). Callers working
//! with non-English/accented names should not expect this to behave like
//! a general Unicode phonetic algorithm, because it is not one: it
//! degrades (dropping accents/non-Latin characters) rather than crashing.

use rphonetic::{DoubleMetaphone, Encoder, Soundex};

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
}
