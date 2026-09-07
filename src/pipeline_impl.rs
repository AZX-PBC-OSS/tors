//! Stateless batch text-preprocessing: the pure-Rust core of
//! `tors.apply_pipeline`.
//!
//! # No pipeline OBJECT: pure function composition
//!
//! This fuses NFD-normalize / lowercase / accent-fold / stem / lemma-
//! substitute / whitespace-collapse into ONE GIL-released pass over a
//! WHOLE list of texts. There is no `re.compile()`-style
//! compiled-pipeline `#[pyclass]` handle (build once, `.apply()` many
//! times); pure function composition only, no persistent Rust-side state.
//! Every call re-describes and re-applies its steps fresh: `tf_idf`/
//! `bm25_rank`'s own `lemma_dict` knob already amortizes a caller-supplied
//! dict's Python->Rust marshalling once per WHOLE-corpus call, the same
//! shape `apply_pipeline` uses over a whole list; PyO3's dict `extract` is
//! linear and realistically low-single-digit milliseconds even for tens of
//! thousands of entries: not a bottleneck worth new object-state
//! complexity.
//!
//! # Order of operations
//!
//! `nfd` -> `lowercase` -> `strip_accents` -> (`stemmer` / `lemma_dict`)
//! -> `collapse_whitespace`, each step skipped entirely when its flag is
//! off/`None`. `nfd`/`lowercase`/`strip_accents` are CODEPOINT-level
//! transforms: they don't care about word boundaries, so they run over
//! the WHOLE text directly, reusing `forms_impl::nfd` and
//! `tokenize_impl::strip_accents_from` verbatim (no reimplementation).
//! `stemmer`/`lemma_dict` are WORD-level: only individual tokens can be
//! meaningfully stemmed or looked up, so when either is requested the
//! (already codepoint-transformed) text is walked segment-by-segment via
//! `split_word_bounds`: the SAME UAX #29 walk `tokenize_impl`'s tokenizer
//! is built on, but UNFILTERED: every non-word segment (punctuation,
//! whitespace) is preserved VERBATIM between the transformed word
//! segments, so the output is still readable prose with stemmed/
//! lemma-substituted words, not a bare token list. A segment counts as a
//! "real word" by the same predicate `segmentation_impl::real_word_segments`
//! uses (not entirely whitespace): stemming a punctuation segment (`"."`,
//! `","`) is a harmless no-op (Snowball only touches alphabetic suffixes),
//! kept for consistency with `tf_idf`/`bm25_rank`'s own tokenization
//! rather than special-cased away.
//!
//! `collapse_whitespace` runs LAST, over the fully-transformed text: every
//! run of Python-whitespace-equivalent codepoints
//! (`normalize_impl::is_py_whitespace`: the same whitespace definition
//! the rest of this crate uses, not Rust's narrower `char::is_whitespace`)
//! becomes exactly one ASCII space. This is NOT
//! `tors.normalize`'s full pipeline (no CRLF folding, no blank-line-run
//! collapsing to two newlines, no leading/trailing strip); just
//! whitespace-RUN collapsing, the one thing this flag promises.
//!
//! **Note on `stemmer` and case**: `rust-stemmers`' `Stemmer::stem`
//! expects already-lowercased input (its own documented contract). A
//! caller passing `stemmer` without `lowercase = true` gets whatever the
//! Snowball algorithm does on mixed-case input: usually a degraded but
//! not incorrect stem, not a panic or an error. This is not silently
//! overridden (the caller's explicit `lowercase = false` is respected,
//! matching this crate's "the caller composes" philosophy); documented
//! here and at the pyo3 boundary instead.
//!
//! # Relationship to `tf_idf`/`bm25_rank`
//!
//! Those two already fuse the SAME normalization knobs (`strip_accents`/
//! `stemmer`/`lemma_dict`) directly into their own tokenization: calling
//! `apply_pipeline` first and THEN `tf_idf`/`bm25_rank` on the result
//! would tokenize twice (once here, once inside them) for no benefit.
//! `apply_pipeline` is the general-purpose preprocessing utility for
//! everything ELSE (`chunk_text`, `find_patterns`, or a caller's own
//! logic); reach for `tf_idf`/`bm25_rank`'s own knobs when those are the
//! only consumer.

use std::borrow::Cow;
use std::collections::HashMap;

use rust_stemmers::Stemmer;
use unicode_segmentation::UnicodeSegmentation;

use crate::forms_impl::nfd;
use crate::normalize_impl::is_py_whitespace;
use crate::tokenize_impl::strip_accents_from;

/// Is `segment` a "real word" (not entirely whitespace): the same
/// predicate `segmentation_impl::real_word_segments` filters on, applied
/// here without discarding the non-word segments (they're preserved
/// verbatim in the reassembled output instead).
fn is_real_word(segment: &str) -> bool {
    !segment.chars().all(char::is_whitespace)
}

/// The word-level pass: walk `text`'s UAX #29 word-boundary segments
/// (`split_word_bounds`, unfiltered), stem and/or lemma-substitute every
/// real-word segment, and pass every other segment (punctuation,
/// whitespace) through unchanged: reassembling into one `String` that
/// stays readable prose, not a token list.
fn apply_word_level(
    text: &str,
    stemmer: Option<&Stemmer>,
    lemma_dict: Option<&HashMap<String, String>>,
) -> String {
    let mut out = String::with_capacity(text.len());
    for segment in text.split_word_bounds() {
        if !is_real_word(segment) {
            out.push_str(segment);
            continue;
        }
        let stemmed: Cow<'_, str> = match stemmer {
            Some(stemmer) => stemmer.stem(segment),
            None => Cow::Borrowed(segment),
        };
        match lemma_dict.and_then(|dict| dict.get(stemmed.as_ref())) {
            Some(lemma) => out.push_str(lemma),
            None => out.push_str(&stemmed),
        }
    }
    out
}

/// Collapse every run of `is_py_whitespace` codepoints in `text` to
/// exactly one ASCII space. Leading/trailing whitespace becomes a single
/// leading/trailing space too (this flag does not strip: a caller
/// wanting that composes `.strip()` themselves, matching this crate's
/// "flat primitive, caller composes" convention).
fn collapse_whitespace_runs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_run = false;
    for c in text.chars() {
        if is_py_whitespace(c) {
            if !in_run {
                out.push(' ');
                in_run = true;
            }
        } else {
            out.push(c);
            in_run = false;
        }
    }
    out
}

/// One text's transform, per the module's documented step order. Returns
/// `Cow::Borrowed` only when every requested step was a genuine no-op on
/// THIS text (used by the batch entry point to decide per-text identity;
/// see [`apply_pipeline`]).
fn transform_one<'a>(
    text: &'a str,
    do_nfd: bool,
    lowercase: bool,
    strip_accents: bool,
    stemmer: Option<&Stemmer>,
    lemma_dict: Option<&HashMap<String, String>>,
    collapse_whitespace: bool,
) -> Cow<'a, str> {
    let mut current: Cow<'a, str> = Cow::Borrowed(text);
    if do_nfd {
        current = match nfd(&current) {
            Cow::Borrowed(_) => current,
            Cow::Owned(s) => Cow::Owned(s),
        };
    }
    if lowercase {
        let lowered = current.to_lowercase();
        if lowered != current.as_ref() {
            current = Cow::Owned(lowered);
        }
    }
    if strip_accents {
        current = match strip_accents_from(&current) {
            Cow::Borrowed(_) => current,
            Cow::Owned(s) => Cow::Owned(s),
        };
    }
    if stemmer.is_some() || lemma_dict.is_some() {
        let transformed = apply_word_level(&current, stemmer, lemma_dict);
        if transformed != current.as_ref() {
            current = Cow::Owned(transformed);
        }
    }
    if collapse_whitespace {
        let collapsed = collapse_whitespace_runs(&current);
        if collapsed != current.as_ref() {
            current = Cow::Owned(collapsed);
        }
    }
    current
}

/// `apply_pipeline(texts, ...)`: fuse every requested step into one pass
/// over the WHOLE list, returning one transformed string per input text
/// (position-matched, `texts.len()` entries out). When every step is
/// off/`None` (the all-defaults call), every returned string is
/// `Cow::Borrowed` of its input: the pyo3 layer uses this to return the
/// caller's ORIGINAL list object unchanged (true identity), not just
/// content-equal output; see `src/py/pipeline.rs`. Empty `texts` -> `[]`.
pub fn apply_pipeline<'a>(
    texts: &[&'a str],
    do_nfd: bool,
    lowercase: bool,
    strip_accents: bool,
    stemmer: Option<&Stemmer>,
    lemma_dict: Option<&HashMap<String, String>>,
    collapse_whitespace: bool,
) -> Vec<Cow<'a, str>> {
    texts
        .iter()
        .map(|text| {
            transform_one(
                text,
                do_nfd,
                lowercase,
                strip_accents,
                stemmer,
                lemma_dict,
                collapse_whitespace,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_stemmers::Algorithm;

    fn run(
        texts: &[&str],
        do_nfd: bool,
        lowercase: bool,
        strip_accents: bool,
        stemmer: Option<&Stemmer>,
        lemma_dict: Option<&HashMap<String, String>>,
        collapse_whitespace: bool,
    ) -> Vec<String> {
        apply_pipeline(
            texts,
            do_nfd,
            lowercase,
            strip_accents,
            stemmer,
            lemma_dict,
            collapse_whitespace,
        )
        .into_iter()
        .map(Cow::into_owned)
        .collect()
    }

    #[test]
    fn all_steps_off_is_identity() {
        let texts = ["Hello  World", "café", ""];
        let out = apply_pipeline(&texts, false, false, false, None, None, false);
        for (input, output) in texts.iter().zip(out.iter()) {
            assert!(matches!(output, Cow::Borrowed(s) if *s == *input));
        }
    }

    #[test]
    fn empty_list_is_empty() {
        assert_eq!(
            apply_pipeline(&[], false, false, false, None, None, false).len(),
            0
        );
    }

    #[test]
    fn lowercase_alone() {
        assert_eq!(
            run(&["HELLO World"], false, true, false, None, None, false),
            vec!["hello world".to_string()]
        );
    }

    #[test]
    fn nfd_alone_decomposes_without_stripping() {
        let out = run(&["café"], true, false, false, None, None, false);
        // NFD decomposes é into e + COMBINING ACUTE ACCENT: the combining
        // mark is still present (nfd, unlike strip_accents, never drops
        // it), only the codepoint SEQUENCE changes.
        assert_eq!(out[0].chars().count(), 5); // c a f e U+0301
        assert!(out[0].chars().any(|c| c == '\u{0301}'));
    }

    #[test]
    fn strip_accents_alone_folds_diacritics() {
        assert_eq!(
            run(&["café"], false, false, true, None, None, false),
            vec!["cafe".to_string()]
        );
    }

    #[test]
    fn stemmer_preserves_non_word_content_verbatim() {
        let stemmer = Stemmer::create(Algorithm::English);
        // Punctuation and whitespace ride along unchanged; only the real
        // word tokens get stemmed.
        assert_eq!(
            run(
                &["The cats, running fast!"],
                false,
                true,
                false,
                Some(&stemmer),
                None,
                false
            ),
            vec!["the cat, run fast!".to_string()]
        );
    }

    #[test]
    fn lemma_dict_alone_substitutes_matched_tokens_and_preserves_the_rest() {
        let mut dict = HashMap::new();
        dict.insert("better".to_string(), "good".to_string());
        assert_eq!(
            run(
                &["This is better, right?"],
                false,
                true,
                false,
                None,
                Some(&dict),
                false
            ),
            vec!["this is good, right?".to_string()]
        );
    }

    #[test]
    fn collapse_whitespace_reduces_any_run_to_one_space() {
        assert_eq!(
            run(&["a   b\t\tc\n\nd"], false, false, false, None, None, true),
            vec!["a b c d".to_string()]
        );
    }

    #[test]
    fn order_of_operations_lowercase_before_strip_accents_before_stem() {
        // "DÉCIDER" -> lowercase -> "décider" -> strip_accents -> "decider"
        // -> French-stem -> "decid". A WRONG order (e.g. stemming before
        // lowercasing) would fail to match the Snowball algorithm's
        // lowercase-only rule tables and produce a different, wrong stem.
        let stemmer = Stemmer::create(Algorithm::French);
        assert_eq!(
            run(&["DÉCIDER"], false, true, true, Some(&stemmer), None, false),
            vec!["decid".to_string()]
        );
    }

    #[test]
    fn stemming_and_lemma_dict_compose_dict_wins_on_the_stemmed_form() {
        let stemmer = Stemmer::create(Algorithm::English);
        let mut dict = HashMap::new();
        dict.insert("run".to_string(), "MOVE".to_string());
        assert_eq!(
            run(
                &["running"],
                false,
                true,
                false,
                Some(&stemmer),
                Some(&dict),
                false
            ),
            vec!["MOVE".to_string()]
        );
    }

    #[test]
    fn all_steps_combined_on_a_realistic_sentence() {
        let stemmer = Stemmer::create(Algorithm::English);
        let out = run(
            &["  Café  RUNNERS   are   RUNNING!  "],
            true,
            true,
            true,
            Some(&stemmer),
            None,
            true,
        );
        assert_eq!(out, vec![" cafe runner are run! ".to_string()]);
    }

    #[test]
    fn position_matches_input_order() {
        let texts = ["ONE", "TWO", "THREE"];
        assert_eq!(
            run(&texts, false, true, false, None, None, false),
            vec!["one".to_string(), "two".to_string(), "three".to_string()]
        );
    }

    #[test]
    fn empty_string_input_stays_empty() {
        assert_eq!(
            run(&[""], false, true, true, None, None, true),
            vec!["".to_string()]
        );
    }

    #[test]
    fn non_latin_scripts_untouched_by_accent_folding_or_stemming() {
        let stemmer = Stemmer::create(Algorithm::English);
        assert_eq!(
            run(
                &["日本語のテキスト"],
                false,
                true,
                true,
                Some(&stemmer),
                None,
                false
            ),
            vec!["日本語のテキスト".to_string()]
        );
    }
}
