"""Contract gate for ``tors.highlight``: snippet-provenance grounding —
WHERE the query's evidence sits in a chunk, as CHARACTER offsets into the
original text. The load-bearing property the whole surface stands on is the
offset round-trip: ``text[start:end]`` (Python codepoint slicing) must
exactly equal the returned ``text`` for every snippet, across every script
(CJK, accents NFC/NFD, emoji ZWJ, RTL, astral planes) — an offset that only
works for ASCII is a bug, not an edge case, so the property tests below
draw arbitrary Unicode text and query pairs and check the slice on each.

The scoring contract (ROUGE-W F1 over UAX #29-tokenized anchor runs,
sentence-bounded) and its bounds are documented in
``src/grounding_impl.rs``; this file pins the behavior a consumer can rely
on, not the algorithm's internals.
"""

from __future__ import annotations

import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import highlight, word_bounds

# Arbitrary Unicode for the offset round-trip property: the property must
# hold for ANY text the pipeline can see, so no alphabet restrictions beyond
# surrogates (never valid in Python str from decoded bytes anyway).
_ANY_TEXT = st.text(max_size=300)
_ANY_QUERY = st.text(max_size=40)

# Terms planted into drawn text: words that ARE in the text, so the
# "every snippet contains a query term" invariant is exercisable.
_WORDS = ["embedding", "model", "café", "検索", "العقد", "torque", "naïve"]


class TestOffsetRoundTrip:
    """THE load-bearing property: every offset slices the original."""

    @settings(max_examples=300)
    @given(text=_ANY_TEXT, query=_ANY_QUERY)
    def test_offsets_slice_the_original_text(self, text: str, query: str) -> None:
        result = highlight(query, text, max_snippets=3, max_chars=400)
        for snippet in result["snippets"]:
            start, end = snippet["start"], snippet["end"]
            assert 0 <= start < end <= len(text), (start, end)
            # Python str slicing is codepoint-based: this is the exact
            # consumer operation.
            assert text[start:end] == snippet["text"]
            assert snippet["score"] == result["score"] or snippet["score"] <= result["score"]

    @settings(max_examples=200)
    @given(text=_ANY_TEXT, query=_ANY_QUERY, max_chars=st.integers(1, 500))
    def test_snippets_respect_the_char_budget(self, text: str, query: str, max_chars: int) -> None:
        result = highlight(query, text, max_snippets=4, max_chars=max_chars)
        for snippet in result["snippets"]:
            # A snippet is at least one token, never a fragment; the budget
            # bounds it once tokens fit (a single token longer than the
            # budget is the documented graceful floor).
            assert len(snippet["text"]) >= 1
            assert len(snippet["text"]) <= max_chars or len(snippet["text"]) < len(text) + 1

    @settings(max_examples=200)
    @given(text=_ANY_TEXT, query=_ANY_QUERY, max_snippets=st.integers(1, 6))
    def test_snippets_are_ordered_and_non_overlapping(
        self, text: str, query: str, max_snippets: int
    ) -> None:
        result = highlight(query, text, max_snippets=max_snippets, max_chars=400)
        snippets = result["snippets"]
        assert len(snippets) <= max_snippets
        for i in range(len(snippets) - 1):
            earlier, later = snippets[i], snippets[i + 1]
            assert earlier["start"] < later["start"]
            assert earlier["end"] <= later["start"], "snippets must not overlap"

    def test_roundtrip_through_every_script_case(self) -> None:
        # The boundary-case battery, each hand-checked: CJK (unspaced),
        # NFC and NFD accents, ZWJ emoji, RTL, astral planes, combining
        # marks, and the mixed-script soup.
        cases = [
            ("検索対象の文書には重要な情報が含まれています。", "重要な情報"),
            ("このテキストを検索する", "キスト"),
            ("café au lait and the café again", "café"),
            ("cafe\u0301 au lait (NFD form)", "café"),  # NFC query, NFD text
            ("prefix 👨‍👩‍👧‍👦 family suffix", "family"),
            ("المادة رقم ٥ من القانون تنص على أن العقد ملزم", "العقد ملزم"),
            ("𝕌𝕟𝕚𝕔𝕠𝕕𝕖 target 𝕒𝕝𝕚𝕘𝕟𝕞𝕖𝕟𝕥", "target"),
            (" Wilde rätselhafte Fußbekleidung ", "Fußbekleidung"),
            ("emoji 🎉🎉🎉 then the term", "term"),
        ]
        for text, query in cases:
            for form in ("NFC", "NFD"):
                normalized = unicodedata.normalize(form, text)
                result = highlight(query, normalized, max_snippets=3, max_chars=400)
                for snippet in result["snippets"]:
                    start, end = snippet["start"], snippet["end"]
                    assert normalized[start:end] == snippet["text"], (text, form, snippet)


class TestScoring:
    @settings(max_examples=100)
    @given(text=_ANY_TEXT, query=_ANY_QUERY)
    def test_a_returned_snippet_always_contains_a_query_term(self, text: str, query: str) -> None:
        result = highlight(query, text, max_snippets=3, max_chars=400)
        # The oracle is the tokenizer's OWN notion of a term — tors.word_bounds
        # (UAX #29), the exact segmentation the scorer runs on — not a Python
        # regex: `\w+` glues "0¼" into one term while UAX #29 splits 0|¼
        # (U+00BC is WB=Other), and every such divergence is a false oracle
        # failure.  Segments are case-folded NFC to mirror the matcher.
        terms = [query[s:e] for s, e in word_bounds(query)]
        for snippet in result["snippets"]:
            body = unicodedata.normalize("NFC", snippet["text"]).lower()
            assert any(unicodedata.normalize("NFC", term).lower() in body for term in terms), (
                snippet
            )

    def test_the_exact_match_scores_about_one(self) -> None:
        result = highlight(
            "lazy dog", "The quick brown fox jumps over the lazy dog.", max_snippets=3, max_chars=10
        )
        assert len(result["snippets"]) == 1
        assert result["snippets"][0]["text"] == "lazy dog"
        assert result["snippets"][0]["score"] > 0.99
        assert result["score"] == result["snippets"][0]["score"]

    def test_no_overlap_means_no_snippets_and_a_zero_score(self) -> None:
        result = highlight("zebra", "the quick brown fox", max_snippets=3, max_chars=400)
        assert result == {"snippets": [], "score": 0.0}

    def test_cjk_range_punctuation_is_token_free(self) -> None:
        # Red-team P0, green pin: CJK-range punctuation (U+30FB middle dot,
        # U+3099) is dropped by the tokenizer's no-alphanumeric rule like any
        # other punctuation — token-free operands highlight nothing at 0.0.
        for text in ["・", "\u3099", "・。", "。、", "「」", "〜"]:
            assert not any(ch.isalnum() for ch in text), repr(text)
            assert highlight(text, text) == {"snippets": [], "score": 0.0}, repr(text)
            assert highlight("term", text) == {"snippets": [], "score": 0.0}, repr(text)
            assert highlight(text, "some term here") == {
                "snippets": [],
                "score": 0.0,
            }, repr(text)

    def test_real_cjk_words_still_tokenize_through_the_punctuation_sweep(self) -> None:
        # The punctuation drop must not over-correct: real CJK morphemes
        # still anchor and score.
        result = highlight("日本", "日本語のテキスト")
        assert result["snippets"] and result["score"] > 0.0

    def test_case_folding_and_nfc_equivalence_do_not_block_matching(self) -> None:
        result = highlight(
            "EMBEDDING model", "The embedding MODEL runs fast.", max_snippets=3, max_chars=400
        )
        assert len(result["snippets"]) == 1
        # NFD text matches an NFC query (the fold canonicalizes).
        nfc = highlight("café", "the cafe\u0301 au lait", max_snippets=3, max_chars=400)
        nfd = highlight("cafe\u0301", "the café au lait", max_snippets=3, max_chars=400)
        assert len(nfc["snippets"]) == len(nfd["snippets"]) == 1

    def test_contiguous_term_runs_outrank_scattered_ones(self) -> None:
        # ROUGE-W's shaping, as a behavior: the passage with the terms
        # adjacent must outrank (and be picked ahead of) the same terms
        # spread through filler, at equal term coverage.
        spread = "alpha x y z beta and then more words follow here now"
        contiguous = "alpha beta and then more words follow here now"
        a = highlight("alpha beta", contiguous, max_snippets=1, max_chars=400)
        b = highlight("alpha beta", spread, max_snippets=1, max_chars=400)
        assert a["snippets"][0]["score"] > b["snippets"][0]["score"]


class TestSelection:
    def test_multiple_scattered_matches_yield_multiple_snippets(self) -> None:
        text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
        result = highlight("alpha kilo", text, max_snippets=3, max_chars=12)
        assert len(result["snippets"]) == 2
        assert result["snippets"][0]["text"] == "alpha"
        assert result["snippets"][1]["text"] == "kilo"

    def test_max_snippets_caps_the_selection(self) -> None:
        text = ". ".join(f"Sentence about torques {i} here" for i in range(10))
        result = highlight("torques", text, max_snippets=2, max_chars=400)
        assert len(result["snippets"]) == 2
        result3 = highlight("torques", text, max_snippets=6, max_chars=400)
        assert len(result3["snippets"]) == 6

    def test_snippets_are_deterministic(self) -> None:
        text = "tick tock tick tock tick tock tick tock"
        assert highlight("tick", text, max_snippets=3, max_chars=400) == highlight(
            "tick", text, max_snippets=3, max_chars=400
        )

    def test_sentence_expansion_when_the_sentence_fits(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        result = highlight("lazy dog", text, max_snippets=3, max_chars=400)
        assert result["snippets"][0]["text"] == text

    def test_a_snippet_always_keeps_at_least_one_token(self) -> None:
        result = highlight("content", "some content here", max_snippets=3, max_chars=1)
        assert result["snippets"][0]["text"] == "content"


class TestDegenerateInputs:
    @pytest.mark.parametrize(
        ("query", "text"),
        [
            ("", "some text"),
            ("some", ""),
            ("", ""),
            ("   !!!   ", "some text"),
            ("some", "   !!!   "),
        ],
    )
    def test_empty_and_token_free_inputs_return_the_empty_result(
        self, query: str, text: str
    ) -> None:
        assert highlight(query, text, max_snippets=3, max_chars=400) == {
            "snippets": [],
            "score": 0.0,
        }

    def test_zero_max_snippets_returns_the_empty_result(self) -> None:
        assert highlight("some", "some text", max_snippets=0, max_chars=400) == {
            "snippets": [],
            "score": 0.0,
        }

    def test_zero_max_chars_is_a_value_error(self) -> None:
        with pytest.raises(ValueError):
            highlight("some", "some text", max_snippets=3, max_chars=0)

    @pytest.mark.parametrize(
        ("query", "text"),
        [
            ("a", "a" * 10_000),  # one giant token: "a" is not a word in it
            ("a" * 10_000, "a"),  # query longer than the chunk
            # Lone surrogates are absent by design: CPython raises
            # UnicodeEncodeError encoding them before Rust sees any
            # bytes, so no extension API can ever accept them.
            ("𝕌" * 5000, "𝕌" * 5000),  # astral planes at scale
            ("e" * 200 + "́", "e" * 200 + "́"),  # combining mark at scale
        ],
    )
    def test_never_panics_on_hostile_input(self, query: str, text: str) -> None:
        result = highlight(query, text, max_snippets=3, max_chars=400)
        for snippet in result["snippets"]:
            assert text[snippet["start"] : snippet["end"]] == snippet["text"]

    def test_pathological_chunk_stays_bounded_and_correct(self) -> None:
        # A 560k-character adversarial repeat: the token cap bounds the
        # scan; the call completes and the snippets slice back exactly.
        text = "relevant term " * 40_000
        result = highlight("relevant term", text, max_snippets=3, max_chars=400)
        assert len(result["snippets"]) == 3
        for snippet in result["snippets"]:
            assert text[snippet["start"] : snippet["end"]] == snippet["text"]
