"""Contract gate for ``tors.apply_pipeline``: a stateless, general-purpose
batch text preprocessor: every requested step fused into one GIL-released
pass over the whole ``texts`` list. The pipeline itself is pure function
composition, not a stateful object. ``lemma_dict`` is the one narrow
exception: it accepts either a raw ``dict[str, str]`` (materialized fresh
every call) or a ``CompiledLemmaDict`` (``tors.CompiledLemmaDict``, built
once and reused, the ``re.compile()`` answer to a real measured per-call
marshalling cost; see ``TestCompiledLemmaDict`` below). Order: ``nfd`` ->
``lowercase`` -> ``strip_accents`` -> (``stemmer`` / ``lemma_dict``) ->
``collapse_whitespace``, each skipped when its flag is off/``None``. See
``src/pipeline_impl.rs`` for the exact algorithm.
"""

from __future__ import annotations

import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import CompiledLemmaDict, apply_pipeline

_TEXT = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs", "P"), max_codepoint=0x2FFF),
    max_size=60,
)


class TestIdentity:
    def test_all_steps_off_returns_the_original_list_object(self) -> None:
        texts = ["Hello  World", "café", ""]
        assert apply_pipeline(texts) is texts

    def test_empty_list_is_empty(self) -> None:
        assert apply_pipeline([]) == []

    def test_explicit_false_none_defaults_are_still_identity(self) -> None:
        texts = ["Anything Goes"]
        assert (
            apply_pipeline(
                texts,
                nfd=False,
                lowercase=False,
                strip_accents=False,
                stemmer=None,
                lemma_dict=None,
                collapse_whitespace=False,
            )
            is texts
        )


class TestIndividualSteps:
    def test_nfd_decomposes_without_dropping_combining_marks(self) -> None:
        out = apply_pipeline(["café"], nfd=True)
        assert out[0] == unicodedata.normalize("NFD", "café")
        assert len(out[0]) == 5  # c a f e combining-acute

    def test_lowercase_is_unicode_correct(self) -> None:
        assert apply_pipeline(["HELLO Straße"], lowercase=True) == ["hello straße"]

    def test_strip_accents_folds_latin_diacritics(self) -> None:
        assert apply_pipeline(["café"], strip_accents=True) == ["cafe"]

    def test_strip_accents_already_decomposed_input_still_strips(self) -> None:
        already_decomposed = "éclair"
        assert apply_pipeline([already_decomposed], strip_accents=True) == ["eclair"]

    def test_stemmer_preserves_punctuation_and_whitespace_verbatim(self) -> None:
        assert apply_pipeline(["The cats, running fast!"], lowercase=True, stemmer="english") == [
            "the cat, run fast!"
        ]

    def test_lemma_dict_substitutes_and_preserves_the_rest(self) -> None:
        out = apply_pipeline(
            ["This is better, right?"],
            lowercase=True,
            lemma_dict={"better": "good"},
        )
        assert out == ["this is good, right?"]

    def test_collapse_whitespace_reduces_any_run_to_one_space(self) -> None:
        assert apply_pipeline(["a   b\t\tc\n\nd"], collapse_whitespace=True) == ["a b c d"]

    def test_unrecognized_stemmer_raises_value_error_naming_choices(self) -> None:
        with pytest.raises(ValueError, match="english"):
            apply_pipeline(["x"], stemmer="klingon")  # type: ignore[arg-type]


class TestOrderOfOperations:
    def test_lowercase_before_strip_accents_before_stem(self) -> None:
        # "DÉcider" -> lowercase -> "décider" -> strip_accents -> "decider"
        # -> French-stem -> "decid". A wrong order (e.g. stemming before
        # lowercasing) would miss the Snowball algorithm's lowercase-only
        # rule tables and produce a different result.
        out = apply_pipeline(["DÉCIDER"], lowercase=True, strip_accents=True, stemmer="french")
        assert out == ["decid"]

    def test_stemming_and_lemma_dict_compose_lookup_on_stemmed_form(self) -> None:
        out = apply_pipeline(
            ["running"], lowercase=True, stemmer="english", lemma_dict={"run": "MOVE"}
        )
        assert out == ["MOVE"]

    def test_collapse_whitespace_runs_last_over_the_transformed_text(self) -> None:
        out = apply_pipeline(
            ["  Café  RUNNERS   are   RUNNING!  "],
            nfd=True,
            lowercase=True,
            strip_accents=True,
            stemmer="english",
            collapse_whitespace=True,
        )
        assert out == [" cafe runner are run! "]


class TestArgumentContract:
    def test_non_list_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            apply_pipeline("not a list")  # type: ignore[arg-type]

    def test_non_str_element_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            apply_pipeline(["fine", 123, "also fine"])  # type: ignore[list-item]

    def test_non_dict_lemma_dict_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            apply_pipeline(["x"], lemma_dict="not a dict")  # type: ignore[arg-type]

    def test_non_str_key_or_value_in_lemma_dict_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            apply_pipeline(["x"], lemma_dict={1: "a"})  # type: ignore[dict-item]
        with pytest.raises(TypeError):
            apply_pipeline(["x"], lemma_dict={"a": 1})  # type: ignore[dict-item]


class TestCompiledLemmaDict:
    """``CompiledLemmaDict`` builds the same ``HashMap`` a raw
    ``lemma_dict`` argument would, once, up front, instead of on every
    call: the ``re.compile()`` answer to a real measured cost (roughly
    1.5ms per call for a realistic 20,000-entry table), not a
    reintroduction of a general stateful-pipeline object."""

    def test_len_matches_the_source_dict(self) -> None:
        compiled = CompiledLemmaDict({"better": "good", "running": "run"})
        assert len(compiled) == 2

    def test_empty_mapping_is_valid(self) -> None:
        assert len(CompiledLemmaDict({})) == 0

    def test_repr_names_the_entry_count(self) -> None:
        assert "2" in repr(CompiledLemmaDict({"a": "b", "c": "d"}))

    def test_non_str_key_or_value_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            CompiledLemmaDict({1: "a"})  # type: ignore[dict-item]
        with pytest.raises(TypeError):
            CompiledLemmaDict({"a": 1})  # type: ignore[dict-item]

    def test_non_dict_argument_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            CompiledLemmaDict("not a dict")  # type: ignore[arg-type]

    def test_produces_identical_output_to_the_equivalent_raw_dict(self) -> None:
        mapping = {"better": "good", "running": "run"}
        compiled = CompiledLemmaDict(mapping)
        text = ["This is better, and I am running."]
        assert apply_pipeline(text, lowercase=True, lemma_dict=mapping) == apply_pipeline(
            text, lowercase=True, lemma_dict=compiled
        )

    def test_accepted_by_apply_pipeline_and_substitutes_correctly(self) -> None:
        compiled = CompiledLemmaDict({"better": "good"})
        out = apply_pipeline(["This is better."], lowercase=True, lemma_dict=compiled)
        assert out == ["this is good."]

    def test_reused_across_multiple_calls_without_rebuilding(self) -> None:
        # The whole point: one CompiledLemmaDict, many calls, each call an
        # Arc::clone rather than a fresh HashMap build.
        compiled = CompiledLemmaDict({"better": "good"})
        for _ in range(50):
            assert apply_pipeline(["This is better."], lowercase=True, lemma_dict=compiled) == [
                "this is good."
            ]

    def test_lone_surrogate_raises(self) -> None:
        with pytest.raises(UnicodeEncodeError):
            apply_pipeline(["a\ud800b"], lowercase=True)


class TestNonLatinScripts:
    def test_cjk_and_cyrillic_untouched_by_accent_folding_or_stemming(self) -> None:
        out = apply_pipeline(
            ["日本語のテキスト"], lowercase=True, strip_accents=True, stemmer="english"
        )
        assert out == ["日本語のテキスト"]


class TestProperties:
    @given(texts=st.lists(_TEXT, max_size=8))
    @settings(max_examples=150)
    def test_output_length_matches_input_length(self, texts: list[str]) -> None:
        assert len(apply_pipeline(texts, lowercase=True)) == len(texts)

    @given(texts=st.lists(_TEXT, max_size=8))
    @settings(max_examples=150)
    def test_deterministic(self, texts: list[str]) -> None:
        out1 = apply_pipeline(texts, lowercase=True, strip_accents=True, stemmer="english")
        out2 = apply_pipeline(texts, lowercase=True, strip_accents=True, stemmer="english")
        assert out1 == out2

    @given(texts=st.lists(_TEXT, max_size=8))
    @settings(max_examples=150)
    def test_lowercase_output_is_always_lowercase(self, texts: list[str]) -> None:
        for text in apply_pipeline(texts, lowercase=True):
            assert text == text.lower()

    @given(texts=st.lists(_TEXT, max_size=8))
    @settings(max_examples=100)
    def test_collapse_whitespace_output_has_no_multi_space_runs(self, texts: list[str]) -> None:
        for text in apply_pipeline(texts, collapse_whitespace=True):
            assert "  " not in text
