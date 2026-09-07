"""Contract gate for ``tors.tf_idf``: a STATELESS TF-IDF primitive; no
vocabulary/vectorizer object persists between calls. Tokenization is UAX #29
word segments (non-whitespace only, lowercased; the near-universal
case-folding convention). TF is the raw per-document term count. IDF is the
scikit-learn-style SMOOTHED formula ``ln((1 + N) / (1 + df)) + 1``,
intentionally not the textbook ``ln(N / df)`` (which gives a term in every
document an IDF of exactly 0). See ``src/tfidf_impl.rs`` for the exact
formulas.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import CompiledLemmaDict, tf_idf

_WORDS = st.text(
    alphabet=st.characters(whitelist_categories=("L",), max_codepoint=0x2FFF),
    min_size=1,
    max_size=8,
)
_DOCUMENT = st.lists(_WORDS, max_size=12).map(lambda words: " ".join(words))
_CORPUS = st.lists(_DOCUMENT, max_size=8)


def _idf(n: int, df: int) -> float:
    return math.log((1 + n) / (1 + df)) + 1.0


class TestKnownVectors:
    def test_three_document_toy_corpus(self) -> None:
        corpus = [
            "the cat sat on the mat",
            "the dog sat on the log",
            "birds fly in the sky",
        ]
        result = tf_idf(corpus)
        assert len(result) == 3

        # "the": appears in all 3 documents -> df=3, N=3.
        the_idf = _idf(3, 3)
        # "cat": appears in 1 of 3 documents -> df=1.
        cat_idf = _idf(3, 1)

        doc0 = dict(result[0])
        assert doc0["cat"] == pytest.approx(1 * cat_idf)
        assert doc0["the"] == pytest.approx(2 * the_idf)  # "the" occurs twice
        assert doc0["sat"] == pytest.approx(1 * _idf(3, 2))  # in docs 0 and 1

        doc2 = dict(result[2])
        assert doc2["the"] == pytest.approx(1 * the_idf)  # occurs once here
        assert doc2["birds"] == pytest.approx(1 * _idf(3, 1))

    def test_empty_corpus(self) -> None:
        assert tf_idf([]) == []

    def test_single_document_corpus(self) -> None:
        # N=1: every term has df=1, idf = ln((1+1)/(1+1)) + 1 = 1.0.
        result = tf_idf(["a b a"])
        expected_idf = _idf(1, 1)
        assert dict(result[0]) == {
            "a": pytest.approx(2 * expected_idf),
            "b": pytest.approx(1 * expected_idf),
        }

    def test_empty_string_document_keeps_its_position(self) -> None:
        result = tf_idf(["cat sat", "", "dog ran"])
        assert len(result) == 3
        assert result[1] == []

    def test_all_empty_documents(self) -> None:
        assert tf_idf(["", "", ""]) == [[], [], []]

    def test_case_folding_merges_terms(self) -> None:
        result = tf_idf(["Cat cat CAT cAt"])
        assert len(result[0]) == 1
        term, score = result[0][0]
        assert term == "cat"
        assert score == pytest.approx(4 * _idf(1, 1))

    def test_terms_sorted_alphabetically(self) -> None:
        result = tf_idf(["zebra apple mango apple"])
        terms = [t for t, _ in result[0]]
        assert terms == sorted(terms)

    def test_rare_term_outscores_universal_term_at_equal_raw_frequency(self) -> None:
        corpus = ["cat dog", "cat bird", "cat fish"]
        result = tf_idf(corpus)
        doc0 = dict(result[0])
        # "cat" is universal (df=3), "dog" appears only here (df=1); both
        # occur once in doc 0, so only the IDF difference can separate them.
        assert doc0["dog"] > doc0["cat"]

    def test_universal_term_scores_strictly_positive(self) -> None:
        result = tf_idf(["cat", "cat", "cat", "cat"])
        assert all(score > 0.0 for _, score in result[0])


class TestArgumentContract:
    def test_non_list_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            tf_idf("not a list")  # type: ignore[arg-type]

    def test_non_str_element_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            tf_idf(["fine", 123, "also fine"])  # type: ignore[list-item]

    def test_lone_surrogate_raises(self) -> None:
        with pytest.raises(UnicodeEncodeError):
            tf_idf(["a\ud800b"])


class TestNormalizationKnobs:
    def test_defaults_are_byte_identical_to_no_knobs(self) -> None:
        corpus = ["café société", "naïve MÜNCHEN"]
        assert tf_idf(corpus) == tf_idf(corpus, strip_accents=False, stemmer=None)

    def test_strip_accents_folds_latin_diacritics(self) -> None:
        result = tf_idf(["café société"], strip_accents=True)
        terms = {t for t, _ in result[0]}
        assert terms == {"cafe", "societe"}

    def test_strip_accents_off_by_default_keeps_diacritics(self) -> None:
        result = tf_idf(["café"])
        terms = {t for t, _ in result[0]}
        assert terms == {"café"}

    def test_strip_accents_already_decomposed_input_still_strips(self) -> None:
        # "e" + COMBINING ACUTE ACCENT (U+0301), not precomposed "é": the
        # exact shape scikit-learn's own strip_accents_unicode gets wrong.
        already_decomposed = "éclair"
        result = tf_idf([already_decomposed], strip_accents=True)
        terms = {t for t, _ in result[0]}
        assert terms == {"eclair"}

    def test_stemmer_merges_inflected_forms(self) -> None:
        result = tf_idf(["running runs runner"], stemmer="english")
        terms = {t for t, _ in result[0]}
        # "running"/"runs" both stem to "run"; "runner" stems separately.
        assert terms == {"run", "runner"}

    def test_stemmer_none_is_a_true_no_op(self) -> None:
        corpus = ["running runs runner"]
        assert tf_idf(corpus, stemmer=None) == tf_idf(corpus)

    def test_unrecognized_stemmer_language_raises_value_error_naming_choices(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="klingon"):
            tf_idf(["x"], stemmer="klingon")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="english"):
            tf_idf(["x"], stemmer="klingon")  # type: ignore[arg-type]

    def test_accent_folding_and_stemming_compose(self) -> None:
        # "décider" -> fold -> "decider" -> French-stem -> "decid".
        result = tf_idf(["décider"], strip_accents=True, stemmer="french")
        terms = {t for t, _ in result[0]}
        assert terms == {"decid"}

    def test_strip_accents_leaves_cjk_and_cyrillic_untouched(self) -> None:
        assert tf_idf(["日本語"], strip_accents=True) == tf_idf(["日本語"])
        result = tf_idf(["Москва"], strip_accents=True)
        terms = {t for t, _ in result[0]}
        assert terms == {"москва"}


class TestLemmaDict:
    """``lemma_dict``: a caller-supplied word -> lemma map, applied LAST
    (after any stemming). tors does not bundle a lemma dictionary
    (out-of-scope, needs a per-language dataset or POS model, the same
    boundary that kept schema-aware JSON/YAML coercion out of this crate)
    It only APPLIES one supplied, the same shape ``replace_many`` takes
    a caller-supplied replacement map rather than a bundled one."""

    def test_none_or_empty_dict_is_a_true_no_op(self) -> None:
        corpus = ["better geese cats"]
        baseline = tf_idf(corpus)
        assert tf_idf(corpus, lemma_dict=None) == baseline
        assert tf_idf(corpus, lemma_dict={}) == baseline

    def test_substitutes_a_mapped_term(self) -> None:
        result = tf_idf(["better geese"], lemma_dict={"better": "good", "geese": "goose"})
        terms = {t for t, _ in result[0]}
        assert terms == {"good", "goose"}

    def test_lookup_is_on_the_stemmed_form(self) -> None:
        # english-stem("running") == "run"; the dict only has "run", not
        # "running"; the lookup must happen AFTER stemming.
        result = tf_idf(["running"], stemmer="english", lemma_dict={"run": "MOVE"})
        terms = {t for t, _ in result[0]}
        assert terms == {"MOVE"}

    def test_non_dict_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            tf_idf(["x"], lemma_dict="not a dict")  # type: ignore[arg-type]

    def test_accepts_a_compiled_lemma_dict_with_identical_output(self) -> None:
        mapping = {"better": "good", "geese": "goose"}
        compiled = CompiledLemmaDict(mapping)
        corpus = ["better geese"]
        assert tf_idf(corpus, lemma_dict=mapping) == tf_idf(corpus, lemma_dict=compiled)

    def test_non_str_key_or_value_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            tf_idf(["x"], lemma_dict={1: "a"})  # type: ignore[dict-item]
        with pytest.raises(TypeError):
            tf_idf(["x"], lemma_dict={"a": 1})  # type: ignore[dict-item]


class TestProperties:
    @given(corpus=_CORPUS)
    @settings(max_examples=200)
    def test_output_length_matches_input_length(self, corpus: list[str]) -> None:
        assert len(tf_idf(corpus)) == len(corpus)

    @given(corpus=_CORPUS)
    @settings(max_examples=200)
    def test_all_scores_are_non_negative(self, corpus: list[str]) -> None:
        for doc in tf_idf(corpus):
            for _, score in doc:
                assert score >= 0.0

    @given(corpus=_CORPUS)
    @settings(max_examples=200)
    def test_terms_within_a_document_are_alphabetically_sorted(
        self, corpus: list[str]
    ) -> None:
        for doc in tf_idf(corpus):
            terms = [t for t, _ in doc]
            assert terms == sorted(terms)

    @given(corpus=_CORPUS)
    @settings(max_examples=200)
    def test_terms_within_a_document_are_unique(self, corpus: list[str]) -> None:
        for doc in tf_idf(corpus):
            terms = [t for t, _ in doc]
            assert len(terms) == len(set(terms))

    @given(corpus=_CORPUS)
    @settings(max_examples=100)
    def test_deterministic(self, corpus: list[str]) -> None:
        assert tf_idf(corpus) == tf_idf(corpus)

    @given(corpus=_CORPUS)
    @settings(max_examples=100)
    def test_every_term_is_lowercase(self, corpus: list[str]) -> None:
        for doc in tf_idf(corpus):
            for term, _ in doc:
                assert term == term.lower()
