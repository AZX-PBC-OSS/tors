"""Contract gate for ``tors.bm25_rank``: a reranking primitive over a
caller-supplied ``corpus`` against one ``query``, recomputed from scratch
every call, not a persistent search index (see ``src/bm25_impl.rs`` for the
explicit scope decline: a large, repeatedly-queried corpus wants a real
search engine, e.g. ``tantivy``; this is for reranking a small,
already-retrieved candidate set).

The formula is Okapi bm25 with the always-non-negative "+1" IDF variant
(``ln((N - df + 0.5) / (df + 0.5) + 1)``), not the classic form (which can
go negative for a term in over half the corpus). Tokenization is the same
"real word token, lowercased" convention ``tf_idf`` uses, with the same
opt-in ``strip_accents``/``stemmer`` knobs, applied identically to `query`
and every corpus document.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import CompiledLemmaDict, bm25_rank

_WORDS = st.text(
    alphabet=st.characters(whitelist_categories=("L",), max_codepoint=0x2FFF),
    min_size=1,
    max_size=8,
)
_DOCUMENT = st.lists(_WORDS, min_size=1, max_size=12).map(lambda words: " ".join(words))
_CORPUS = st.lists(_DOCUMENT, min_size=1, max_size=8)

# ASCII-restricted variants for the reference-formula differential test;
# see that test's docstring for why the general Unicode strategy above
# isn't sound for a naive-`.split()`-tokenizer oracle.
_ASCII_WORDS = st.text(
    alphabet=st.characters(whitelist_categories=("L",), max_codepoint=0x7A), min_size=1, max_size=8
)
_ASCII_DOCUMENT = st.lists(_ASCII_WORDS, min_size=1, max_size=12).map(lambda words: " ".join(words))
_ASCII_CORPUS = st.lists(_ASCII_DOCUMENT, min_size=1, max_size=8)


def _idf(n: int, df: int) -> float:
    return math.log((n - df + 0.5) / (df + 0.5) + 1.0)


def _reference_score(query: str, corpus: list[str], k1: float, b: float) -> list[float]:
    """A from-scratch Python re-implementation of the exact same formula,
    the independent oracle for the known-vector/hypothesis tests below."""
    docs = [doc.lower().split() for doc in corpus]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    doc_freq: dict[str, int] = {}
    for d in docs:
        for term in set(d):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    query_terms = list(dict.fromkeys(query.lower().split()))
    scores = []
    for d in docs:
        dl = len(d)
        len_norm = 1 - b + b * (dl / avgdl)
        score = 0.0
        for term in query_terms:
            n_t = doc_freq.get(term, 0)
            idf = _idf(n, n_t)
            f_td = d.count(term)
            if f_td == 0:
                continue
            score += idf * (f_td * (k1 + 1)) / (f_td + k1 * len_norm)
        scores.append(score)
    return scores


class TestKnownVectors:
    def test_three_document_toy_corpus_matches_the_reference_formula(self) -> None:
        corpus = [
            "the quick brown fox jumps over the lazy dog",
            "a lazy cat sleeps all day",
            "the fox and the dog are friends",
        ]
        expected = _reference_score("quick fox", corpus, 1.5, 0.75)
        ranked = dict(bm25_rank("quick fox", corpus))
        for i, score in enumerate(expected):
            assert ranked[i] == pytest.approx(score)

    def test_empty_corpus_returns_empty_list(self) -> None:
        assert bm25_rank("anything", []) == []

    def test_empty_query_scores_every_document_zero(self) -> None:
        corpus = ["cat sat", "dog ran", "bird flew"]
        assert bm25_rank("", corpus) == [(0, 0.0), (1, 0.0), (2, 0.0)]

    def test_query_term_absent_from_every_document_scores_zero(self) -> None:
        corpus = ["cat sat", "dog ran"]
        assert bm25_rank("nonexistentword", corpus) == [(0, 0.0), (1, 0.0)]

    def test_single_document_corpus_does_not_divide_by_zero(self) -> None:
        # avgdl == that document's own length; b-normalization is a no-op.
        result = bm25_rank("cat", ["the cat sat on the mat"])
        assert len(result) == 1
        assert result[0][1] > 0.0

    def test_results_sorted_by_score_descending(self) -> None:
        corpus = ["cat cat cat", "cat", "dog dog dog"]
        ranked = bm25_rank("cat", corpus)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_ties_broken_by_ascending_original_index(self) -> None:
        # Two documents with no query-term matches tie at 0.0.
        corpus = ["x y", "a b", "cat sat"]
        ranked = bm25_rank("cat", corpus)
        zero_score_indices = [i for i, s in ranked if s == 0.0]
        assert zero_score_indices == sorted(zero_score_indices)

    def test_repeated_query_word_does_not_multiply_its_contribution(self) -> None:
        corpus = ["cat sat", "dog ran"]
        once = dict(bm25_rank("cat", corpus))
        repeated = dict(bm25_rank("cat cat cat cat cat", corpus))
        assert once[0] == pytest.approx(repeated[0])

    def test_case_insensitive_matching(self) -> None:
        corpus = ["The Cat Sat"]
        assert bm25_rank("cat", corpus)[0][1] == bm25_rank("CAT", corpus)[0][1]

    def test_scores_are_never_negative(self) -> None:
        # The "+1" IDF variant's whole point: never negative, even for a
        # term appearing in every document.
        corpus = ["cat cat", "cat", "cat cat cat"]
        for _, score in bm25_rank("cat", corpus):
            assert score >= 0.0


class TestArgumentContract:
    def test_non_list_corpus_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            bm25_rank("q", "not a list")  # type: ignore[arg-type]

    def test_non_str_element_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            bm25_rank("q", ["fine", 123, "also fine"])  # type: ignore[list-item]

    def test_lone_surrogate_raises(self) -> None:
        with pytest.raises(UnicodeEncodeError):
            bm25_rank("q", ["a\ud800b"])

    @pytest.mark.parametrize("k1", [-1.0, -0.001, float("nan"), float("inf")])
    def test_invalid_k1_raises_value_error(self, k1: float) -> None:
        with pytest.raises(ValueError):
            bm25_rank("q", ["a"], k1=k1)

    @pytest.mark.parametrize("b", [-0.1, 1.1, -1.0, 2.0])
    def test_invalid_b_raises_value_error(self, b: float) -> None:
        with pytest.raises(ValueError):
            bm25_rank("q", ["a"], b=b)

    def test_k1_zero_and_b_zero_and_b_one_are_legal_boundaries(self) -> None:
        for k1, b in ((0.0, 0.0), (0.0, 1.0), (1.5, 0.0), (1.5, 1.0)):
            bm25_rank("cat", ["cat sat"], k1=k1, b=b)  # must not raise


class TestNormalizationKnobs:
    def test_defaults_are_byte_identical_to_no_knobs(self) -> None:
        corpus = ["café société", "naïve MÜNCHEN"]
        assert bm25_rank("café", corpus) == bm25_rank(
            "café", corpus, strip_accents=False, stemmer=None
        )

    def test_strip_accents_matches_query_against_accented_corpus(self) -> None:
        corpus = ["café société", "totally unrelated text"]
        # Without folding, an unaccented query misses the accented term.
        assert bm25_rank("cafe", corpus)[0][1] == 0.0
        # With folding applied to both query and corpus, it matches.
        folded = dict(bm25_rank("cafe", corpus, strip_accents=True))
        assert folded[0] > 0.0

    def test_stemmer_matches_query_against_inflected_corpus(self) -> None:
        corpus = ["the runner runs fast", "a cat sat"]
        # Without stemming, "running" (query) misses "runner"/"runs".
        assert bm25_rank("running", corpus)[0][1] == 0.0
        # With English stemming applied to both, they collapse to "run".
        stemmed = dict(bm25_rank("running", corpus, stemmer="english"))
        assert stemmed[0] > 0.0

    def test_unrecognized_stemmer_language_raises_value_error_naming_choices(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="klingon"):
            bm25_rank("q", ["a"], stemmer="klingon")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="english"):
            bm25_rank("q", ["a"], stemmer="klingon")  # type: ignore[arg-type]


class TestLemmaDict:
    """``lemma_dict``: tf_idf's exact same caller-supplied word -> lemma
    map, applied identically to query and corpus (required for scores to
    mean anything, not a style choice)."""

    def test_none_or_empty_dict_is_a_true_no_op(self) -> None:
        corpus = ["the better runner", "unrelated text"]
        baseline = bm25_rank("better", corpus)
        assert bm25_rank("better", corpus, lemma_dict=None) == baseline
        assert bm25_rank("better", corpus, lemma_dict={}) == baseline

    def test_matches_query_against_a_lemma_substituted_corpus_term(self) -> None:
        corpus = ["the runner runs fast", "a cat sat"]
        # Without a lemma map, "move" (query) misses "runner"/"runs" entirely.
        assert bm25_rank("move", corpus)[0][1] == 0.0
        # With stemming (-> "run") + a lemma map applied to both query and
        # corpus, "move" and the stemmed corpus terms collapse to the same
        # substituted term.
        mapped = dict(bm25_rank("move", corpus, stemmer="english", lemma_dict={"run": "move"}))
        assert mapped[0] > 0.0

    def test_non_dict_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            bm25_rank("q", ["a"], lemma_dict="not a dict")  # type: ignore[arg-type]

    def test_non_str_key_or_value_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            bm25_rank("q", ["a"], lemma_dict={1: "a"})  # type: ignore[dict-item]

    def test_accepts_a_compiled_lemma_dict_with_identical_output(self) -> None:
        mapping = {"run": "move"}
        compiled = CompiledLemmaDict(mapping)
        corpus = ["the runner runs fast", "a cat sat"]
        assert bm25_rank("move", corpus, stemmer="english", lemma_dict=mapping) == bm25_rank(
            "move", corpus, stemmer="english", lemma_dict=compiled
        )
        with pytest.raises(TypeError):
            bm25_rank("q", ["a"], lemma_dict={"a": 1})  # type: ignore[dict-item]


class TestProperties:
    @given(query=_WORDS, corpus=_CORPUS)
    @settings(max_examples=200)
    def test_output_length_matches_corpus_length(self, query: str, corpus: list[str]) -> None:
        assert len(bm25_rank(query, corpus)) == len(corpus)

    @given(query=_WORDS, corpus=_CORPUS)
    @settings(max_examples=200)
    def test_every_original_index_appears_exactly_once(self, query: str, corpus: list[str]) -> None:
        indices = [i for i, _ in bm25_rank(query, corpus)]
        assert sorted(indices) == list(range(len(corpus)))

    @given(query=_WORDS, corpus=_CORPUS)
    @settings(max_examples=200)
    def test_all_scores_are_non_negative(self, query: str, corpus: list[str]) -> None:
        for _, score in bm25_rank(query, corpus):
            assert score >= 0.0

    @given(query=_WORDS, corpus=_CORPUS)
    @settings(max_examples=200)
    def test_results_are_sorted_by_score_descending(self, query: str, corpus: list[str]) -> None:
        scores = [s for _, s in bm25_rank(query, corpus)]
        assert scores == sorted(scores, reverse=True)

    @given(query=_WORDS, corpus=_CORPUS)
    @settings(max_examples=100)
    def test_deterministic(self, query: str, corpus: list[str]) -> None:
        assert bm25_rank(query, corpus) == bm25_rank(query, corpus)

    @given(query=_ASCII_WORDS, corpus=_ASCII_CORPUS)
    @settings(max_examples=100)
    def test_matches_the_independent_reference_formula(self, query: str, corpus: list[str]) -> None:
        # ASCII-only here by design: `_reference_score`'s naive
        # `.split()` tokenizer and tors's real UAX #29 segmenter agree on
        # ASCII whitespace-delimited words, but can legitimately diverge
        # on a hypothesis-generated multi-script "word" with no space
        # between scripts (e.g. "Aก"): UAX #29 splits that into two real
        # tokens, `.split()` doesn't, which is a naive-oracle limitation,
        # not a tors bug (segmentation correctness itself is pinned
        # exhaustively in tests/test_segmentation.py, not re-litigated
        # here; this test's job is the bm25 arithmetic).
        expected = _reference_score(query, corpus, 1.5, 0.75)
        ranked = dict(bm25_rank(query, corpus))
        for i, score in enumerate(expected):
            assert ranked[i] == pytest.approx(score, abs=1e-9)
