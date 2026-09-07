//! Opt-in term-normalization knobs for `tf_idf`/`bm25_rank`: accent
//! folding and Snowball stemming, shared by both rather than duplicated
//! (the same DRY discipline that already put their base tokenization in
//! `segmentation_impl::lowercased_word_tokens`/`real_word_segments`).
//!
//! # What this is NOT: lemmatization
//!
//! Full lemmatization needs a per-language dictionary or a POS-tagging
//! model. It is not an algorithm you hand-roll or a small crate you
//! vendor, and pulling one in would break tors's no-external-model
//! philosophy (the same boundary that kept schema-aware JSON/YAML
//! coercion and a heavy multi-language detection table out of this
//! crate). Snowball stemming, a fast, deterministic, model-free suffix-
//! stripping algorithm, is the offered alternative: cruder than a real
//! lemmatizer (it can't tell "better" the comparative from "better" the
//! verb), but correct, dependency-light, and in scope.
//!
//! # Accent folding
//!
//! `strip_accents`: NFD-decompose the token (`forms_impl::nfd`, already
//! shipped as `tors.nfd`, no new normalization logic), then drop every
//! codepoint `unicode_normalization::char::is_combining_mark` reports
//! (General_Category=Mark: Mn+Mc+Me) from the decomposed sequence. This is
//! the standard technique (`scikit-learn`'s `strip_accents='unicode'` does
//! the same decompose-then-drop). Order: lowercase FIRST, THEN decompose
//! and strip, matching `pipeline_impl`'s documented `lowercase` ->
//! `strip_accents` step order exactly, so the two normalization surfaces
//! stay in lockstep. Case mapping and accent-stripping commute for every
//! script this crate has been checked against (case affects only a
//! codepoint's identity, not whether it is a combining mark), so this
//! ordering choice is about keeping one documented order across the crate,
//! not about avoiding a correctness hazard the other order would have.
//!
//! **Correctness note, NOT replicated here**: scikit-learn's
//! `strip_accents_unicode` (per its own issue tracker, gh-15087) has a
//! real bug: it short-circuits with `if NFKD(s) == s: return s` BEFORE
//! stripping, so a token that arrives ALREADY decomposed (e.g. `"e"` +
//! U+0301 rather than precomposed `"é"`, plausible input from another
//! normalization stage) silently keeps its combining mark: decomposing an
//! already-decomposed string is a no-op, so the equality check is true and
//! the whole strip is skipped. The fix here is structural, not a special
//! case: stripping always runs over `nfd()`'s result regardless of whether
//! `nfd()` itself detected a no-op and returned the input borrowed.
//! Decomposition being idempotent must never imply stripping is skippable
//! too. Pinned by `already_decomposed_input_still_strips` below.
//!
//! NFD, not NFKD: NFKD's extra COMPATIBILITY decomposition also affects
//! things accent-folding shouldn't touch (ligatures, width/font variants,
//! `"ﬁ"` -> `"fi"`). NFD's purely CANONICAL decomposition is the
//! conservative, correct choice for "strip accents, change nothing else".
//!
//! # Stemming
//!
//! `stemmer`: an optional Snowball algorithm name (`rust-stemmers`, MIT/
//! BSD-3-Clause, both allowed by `deny.toml`), applied AFTER lowercasing
//! and any accent-folding (the crate's own contract: `Stemmer::stem`
//! expects already-lowercased input). `None` (the default) applies no
//! stemming.

use std::borrow::Cow;
use std::collections::HashMap;
use unicode_normalization::char::is_combining_mark;

use crate::forms_impl::nfd;
use crate::segmentation_impl::real_word_segments;
use rust_stemmers::{Algorithm, Stemmer};

/// The full, sorted list of Snowball language names this crate accepts,
/// the exact spelling of `rust-stemmers`' own `Algorithm` variants,
/// lowercased (the caller-facing spelling; `parse_stemmer_algorithm` is
/// case-insensitive). Used both to build the `Literal[...]` set in the
/// `.pyi` stub and to name the valid choices in a bad-name `ValueError`.
pub const STEMMER_LANGUAGES: &[&str] = &[
    "arabic",
    "danish",
    "dutch",
    "english",
    "finnish",
    "french",
    "german",
    "greek",
    "hungarian",
    "italian",
    "norwegian",
    "portuguese",
    "romanian",
    "russian",
    "spanish",
    "swedish",
    "tamil",
    "turkish",
];

/// `name` (case-insensitive) -> the matching `Algorithm`, or `Err` naming
/// every valid choice (never a silent no-op on a typo'd language name).
pub fn parse_stemmer_algorithm(name: &str) -> Result<Algorithm, String> {
    let lower = name.to_lowercase();
    let algorithm = match lower.as_str() {
        "arabic" => Algorithm::Arabic,
        "danish" => Algorithm::Danish,
        "dutch" => Algorithm::Dutch,
        "english" => Algorithm::English,
        "finnish" => Algorithm::Finnish,
        "french" => Algorithm::French,
        "german" => Algorithm::German,
        "greek" => Algorithm::Greek,
        "hungarian" => Algorithm::Hungarian,
        "italian" => Algorithm::Italian,
        "norwegian" => Algorithm::Norwegian,
        "portuguese" => Algorithm::Portuguese,
        "romanian" => Algorithm::Romanian,
        "russian" => Algorithm::Russian,
        "spanish" => Algorithm::Spanish,
        "swedish" => Algorithm::Swedish,
        "tamil" => Algorithm::Tamil,
        "turkish" => Algorithm::Turkish,
        _ => {
            return Err(format!(
                "unrecognized stemmer language {name:?}; valid choices are: {}",
                STEMMER_LANGUAGES.join(", ")
            ));
        }
    };
    Ok(algorithm)
}

/// NFD-decompose `token`, then drop every combining-mark codepoint.
/// Structurally unconditional (runs over `nfd()`'s result regardless of
/// whether it borrowed or owned), the fix for the sklearn short-circuit
/// bug documented in the module doc.
pub(crate) fn strip_accents_from(token: &str) -> Cow<'_, str> {
    let decomposed = nfd(token);
    if !decomposed.chars().any(is_combining_mark) {
        // Nothing to strip; still correct to return `decomposed` as-is
        // (which may itself be `token` borrowed, if `nfd` was a no-op),
        // not a special case: this is just the ordinary "no marks found"
        // answer of the same filter below, short-circuited only to avoid
        // an always-un-needed allocation on the (common) accent-free path.
        return decomposed;
    }
    Cow::Owned(
        decomposed
            .chars()
            .filter(|c| !is_combining_mark(*c))
            .collect(),
    )
}

/// The richer, opt-in-normalizing sibling of
/// `segmentation_impl::lowercased_word_tokens`: the same real-word-segment
/// walk, lowercased, with `strip_accents`/`stemmer`/`lemma_dict` applied in
/// the SAME single pass (no second full-corpus scan). `strip_accents =
/// false`, `stemmer = None`, `lemma_dict = None` (the defaults every
/// caller gets when not opting in) reproduce `lowercased_word_tokens`'
/// exact output, pinned as a byte-identical regression test, not just
/// assumed.
///
/// `lemma_dict`, a caller-supplied `word -> lemma` mapping, is the LAST
/// step, looked up on the fully-folded token (after lowercase/
/// strip_accents/stem, whichever ran): if the folded token is a key,
/// its mapped value REPLACES it; otherwise the folded token is kept
/// as-is. This is mechanism, not data: tors does not bundle
/// a lemma dictionary (full lemmatization needs a per-language dictionary
/// or a POS-tagging model, the same "no external model" boundary that
/// kept schema-aware JSON/YAML coercion and a heavy language-detection
/// table out of this crate); it only APPLIES one, the same shape
/// `replace_many` already takes a caller-supplied replacement map rather
/// than bundling one. Combining `stemmer` and `lemma_dict` together is
/// unusual in practice but well-defined, not an error: the dict is
/// consulted on the ALREADY-stemmed form (a caller wanting lemma-only
/// normalization simply omits `stemmer`).
pub(crate) fn normalized_word_tokens(
    text: &str,
    strip_accents: bool,
    stemmer: Option<&Stemmer>,
    lemma_dict: Option<&HashMap<String, String>>,
) -> Vec<String> {
    real_word_segments(text)
        .map(|segment| {
            let lowered = segment.to_lowercase();
            let folded: String = if strip_accents {
                strip_accents_from(&lowered).into_owned()
            } else {
                lowered
            };
            let stemmed = match stemmer {
                Some(stemmer) => stemmer.stem(&folded).into_owned(),
                None => folded,
            };
            match lemma_dict.and_then(|dict| dict.get(&stemmed)) {
                Some(lemma) => lemma.clone(),
                None => stemmed,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segmentation_impl::lowercased_word_tokens;

    #[test]
    fn defaults_reproduce_lowercased_word_tokens_exactly() {
        let samples = [
            "Cat sat on the MAT.",
            "",
            "   ",
            "café société naïve",
            "日本語のテキスト",
            "MiXeD Ünïcödé Tëxt",
        ];
        for text in samples {
            assert_eq!(
                normalized_word_tokens(text, false, None, None),
                lowercased_word_tokens(text),
                "mismatch for {text:?}"
            );
        }
    }

    #[test]
    fn strip_accents_folds_latin_diacritics() {
        assert_eq!(
            normalized_word_tokens("café", true, None, None),
            vec!["cafe".to_string()]
        );
        assert_eq!(
            normalized_word_tokens("MÜNCHEN", true, None, None),
            vec!["munchen".to_string()]
        );
    }

    #[test]
    fn already_decomposed_input_still_strips() {
        // "e" + COMBINING ACUTE ACCENT (U+0301), NOT precomposed "é": the
        // exact shape sklearn's strip_accents_unicode gets wrong (its
        // NFKD(s)==s short-circuit sees this as already-decomposed and
        // skips stripping entirely). tors must strip it regardless.
        let already_decomposed = "e\u{0301}clair";
        assert_eq!(
            normalized_word_tokens(already_decomposed, true, None, None),
            vec!["eclair".to_string()]
        );
    }

    #[test]
    fn strip_accents_leaves_non_latin_scripts_untouched() {
        // CJK/Cyrillic have no combining marks to strip here; must not be
        // corrupted by the fold.
        assert_eq!(
            normalized_word_tokens("日本語", true, None, None),
            normalized_word_tokens("日本語", false, None, None),
        );
        assert_eq!(
            normalized_word_tokens("Москва", true, None, None),
            vec!["москва".to_string()],
        );
    }

    #[test]
    fn stemmer_none_is_a_true_no_op() {
        let stemmer = Stemmer::create(Algorithm::English);
        assert_eq!(
            normalized_word_tokens("running runs runner", false, Some(&stemmer), None),
            vec!["run".to_string(), "run".to_string(), "runner".to_string()],
        );
    }

    #[test]
    fn stemming_and_accent_folding_compose() {
        let stemmer = Stemmer::create(Algorithm::French);
        // "décider" -> fold -> "decider" -> French-stem -> "decid".
        let tokens = normalized_word_tokens("décider", true, Some(&stemmer), None);
        assert_eq!(tokens, vec!["decid".to_string()]);
    }

    #[test]
    fn unrecognized_language_names_every_valid_choice() {
        let err = parse_stemmer_algorithm("klingon").unwrap_err();
        assert!(err.contains("klingon"));
        for lang in STEMMER_LANGUAGES {
            assert!(err.contains(lang), "{err} missing {lang}");
        }
    }

    #[test]
    fn language_names_are_case_insensitive() {
        assert_eq!(
            parse_stemmer_algorithm("English").unwrap(),
            parse_stemmer_algorithm("ENGLISH").unwrap()
        );
    }

    #[test]
    fn lemma_dict_none_or_empty_is_a_true_no_op() {
        let empty: HashMap<String, String> = HashMap::new();
        let baseline = normalized_word_tokens("better geese cats", false, None, None);
        assert_eq!(
            normalized_word_tokens("better geese cats", false, None, Some(&empty)),
            baseline
        );
    }

    #[test]
    fn lemma_dict_substitutes_a_mapped_token_and_leaves_others_alone() {
        let mut dict = HashMap::new();
        dict.insert("better".to_string(), "good".to_string());
        dict.insert("geese".to_string(), "goose".to_string());
        assert_eq!(
            normalized_word_tokens("the better geese ran", false, None, Some(&dict)),
            vec![
                "the".to_string(),
                "good".to_string(),
                "goose".to_string(),
                "ran".to_string(),
            ]
        );
    }

    #[test]
    fn lemma_dict_is_looked_up_on_the_stemmed_form() {
        // English-stem("running") == "run"; the dict only has "run", not
        // "running", so the lookup must happen AFTER stemming, not before.
        let stemmer = Stemmer::create(Algorithm::English);
        let mut dict = HashMap::new();
        dict.insert("run".to_string(), "MOVE".to_string());
        assert_eq!(
            normalized_word_tokens("running", false, Some(&stemmer), Some(&dict)),
            vec!["MOVE".to_string()]
        );
    }

    #[test]
    fn lemma_dict_lookup_is_case_and_accent_sensitive_to_the_folded_form() {
        // Lookup key must match the ALREADY lowercased+accent-folded
        // token, not the original spelling.
        let mut dict = HashMap::new();
        dict.insert("cafe".to_string(), "COFFEE_SHOP".to_string());
        assert_eq!(
            normalized_word_tokens("CAFÉ", true, None, Some(&dict)),
            vec!["COFFEE_SHOP".to_string()]
        );
        // Without strip_accents, the folded token is "café" (only
        // lowercased), which the dict (keyed on unaccented "cafe") does
        // NOT match; kept as-is.
        assert_eq!(
            normalized_word_tokens("CAFÉ", false, None, Some(&dict)),
            vec!["café".to_string()]
        );
    }
}
