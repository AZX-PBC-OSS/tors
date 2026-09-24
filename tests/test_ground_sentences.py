"""Contract gate for ``tors.ground_sentences``: the sentence-level grounding
batch: EVERY UAX #29 sentence of the text, scored against the query with
the same ROUGE-W F1 the snippet surface ranks spans with, in position
order, plus the aggregate (the best sentence's score, the documented max
policy).

The load-bearing property is the same one ``highlight`` stands on (the
style of its pinned round-trip test is copied here): every sentence's
offsets slice the ORIGINAL text (``text[start:end] == sentence["text"]``)
for every script (CJK, accents NFC/NFD, emoji ZWJ, RTL, astral planes),
because the offsets are Python codepoint indices into the original string.

The scoring contract and its bounds are documented in
``src/grounding_impl.rs``; this file pins the behavior a consumer can rely
on, not the algorithm's internals.
"""

from __future__ import annotations

import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import ground_sentences, sentence_bounds

# Arbitrary Unicode for the offset round-trip property: the property must
# hold for ANY text the pipeline can see, so no alphabet restrictions beyond
# surrogates (never valid in Python str from decoded bytes anyway).
_ANY_TEXT = st.text(max_size=300)
_ANY_QUERY = st.text(max_size=40)


class TestOffsetRoundTrip:
    """THE load-bearing property: every sentence's offsets slice the
    original: text[s.start:s.end] is exactly s["text"]."""

    @settings(max_examples=300)
    @given(text=_ANY_TEXT, query=_ANY_QUERY)
    def test_offsets_slice_the_original_text(self, text: str, query: str) -> None:
        result = ground_sentences(text, query)
        for sentence in result["sentences"]:
            start, end = sentence["start"], sentence["end"]
            assert 0 <= start < end <= len(text), (start, end)
            # Python str slicing is codepoint-based: this is the exact
            # consumer operation.
            assert text[start:end] == sentence["text"]

    @settings(max_examples=200)
    @given(text=_ANY_TEXT, query=_ANY_QUERY, max_chars=st.integers(1, 500))
    def test_max_chars_never_changes_the_reported_spans(
        self, text: str, query: str, max_chars: int
    ) -> None:
        # The budget bounds the SCORED window, never the report: the same
        # text under any budget yields the same spans.
        clamped = ground_sentences(text, query, max_chars=max_chars)
        whole = ground_sentences(text, query)
        assert clamped["sentences"] == whole["sentences"] or all(
            c["start"] == w["start"] and c["end"] == w["end"] and c["text"] == w["text"]
            for c, w in zip(clamped["sentences"], whole["sentences"], strict=True)
        )

    def test_roundtrip_through_every_script_case(self) -> None:
        # The boundary-case battery, each hand-checked: CJK (unspaced),
        # NFC and NFD accents, ZWJ emoji, RTL, astral planes, the same
        # cases test_grounding.py pins for the snippet offsets.
        cases = [
            "検索対象の文書には重要な情報が含まれています。次の文もある。",
            "このテキストを検索する。キストで終わる文。",
            "café au lait and the café again. Second sentence.",
            "cafe\u0301 au lait (NFD form). Second sentence.",
            "prefix 👨‍👩‍👧‍👦 family sentence. Suffix sentence.",
            "المادة رقم ٥ من القانون تنص على أن العقد ملزم. وتابع.",
            "𝕌𝕟𝕚𝕔𝕠𝕕𝕖 target 𝕒𝕝𝕚𝕘𝕟𝕞𝕖𝕟𝕥. Second sentence.",
            " Wilde rätselhafte Fußbekleidung. Und noch eins. ",
            "emoji 🎉🎉🎉 then the term. Trailing sentence.",
        ]
        for text in cases:
            for form in ("NFC", "NFD"):
                normalized = unicodedata.normalize(form, text)
                result = ground_sentences(normalized, "café 検索 family العقد term")
                for sentence in result["sentences"]:
                    start, end = sentence["start"], sentence["end"]
                    assert normalized[start:end] == sentence["text"], (
                        text,
                        form,
                        sentence,
                    )


class TestStructure:
    """One entry per sentence, position order, offsets = sentence_bounds."""

    def test_one_entry_per_sentence_in_position_order(self) -> None:
        text = "The quick brown fox jumps. Second sentence mentions the fox. Done."
        result = ground_sentences(text, "fox")
        assert len(result["sentences"]) == 3
        for earlier, later in zip(result["sentences"], result["sentences"][1:], strict=False):
            assert earlier["start"] < later["start"]
            assert earlier["end"] <= later["start"]
        assert len(result["sentences"][1:]) == len(result["sentences"]) - 1

    def test_offsets_are_exactly_sentence_bounds_tuples(self) -> None:
        # The documented by-construction claim: the batch's per-sentence
        # offsets ARE tors.sentence_bounds(text)'s tuples.
        text = "One. Two!! Three\r\nfour. 3.4 percent stays one."
        result = ground_sentences(text, "three")
        assert [(s["start"], s["end"]) for s in result["sentences"]] == sentence_bounds(text)
        assert [s["text"] for s in result["sentences"]] == [
            text[s:e] for s, e in sentence_bounds(text)
        ]

    def test_scores_are_in_the_unit_interval(self) -> None:
        result = ground_sentences("alpha beta gamma delta. x. y.", "alpha beta gamma")
        for sentence in result["sentences"]:
            assert 0.0 <= sentence["score"] <= 1.0
        assert 0.0 <= result["score"] <= 1.0


class TestScoring:
    def test_identical_text_and_query_scores_one_point_zero(self) -> None:
        text = "The bushing torque specifications changed."
        result = ground_sentences(text, text)
        assert len(result["sentences"]) == 1
        assert result["sentences"][0]["score"] == pytest.approx(1.0)
        assert result["score"] == pytest.approx(1.0)

    def test_a_whole_query_sentence_outranks_a_partial_overlap(self) -> None:
        # Monotonicity, pinned: the sentence containing EVERY query token
        # outscores a partial-overlap sentence and a zero-overlap one.
        text = "Partial torque only. The full bushing torque spec here. Nothing at all."
        result = ground_sentences(text, "full bushing torque spec")
        assert result["sentences"][1]["score"] > result["sentences"][0]["score"]
        assert result["sentences"][1]["score"] > result["sentences"][2]["score"]

    def test_the_aggregate_is_the_max_sentence_score(self) -> None:
        text = "Unrelated opener. The bushing torque spec changed. Quiet closer."
        result = ground_sentences(text, "torque spec")
        assert result["score"] == pytest.approx(
            max(s["score"] for s in result["sentences"])
        )

    def test_contiguous_term_runs_outrank_scattered_ones(self) -> None:
        # ROUGE-W's shaping, as a behavior (the same pin the snippet
        # surface makes: adjacent query terms outrank spread ones.
        contiguous = "alpha beta and then more words follow here now. Tail."
        spread = "alpha x y z beta and then more words follow here now. Tail."
        a = ground_sentences(contiguous, "alpha beta")
        b = ground_sentences(spread, "alpha beta")
        assert a["sentences"][0]["score"] > b["sentences"][0]["score"]

    def test_case_folding_and_nfc_equivalence_do_not_block_matching(self) -> None:
        result = ground_sentences("The embedding MODEL runs fast.", "EMBEDDING model")
        assert result["sentences"][0]["score"] > 0.0
        nfc = ground_sentences("the cafe\u0301 au lait", "café")
        nfd = ground_sentences("the café au lait", "cafe\u0301")
        assert nfc["sentences"][0]["score"] == pytest.approx(
            nfd["sentences"][0]["score"]
        )


class TestMaxChars:
    def test_the_budget_bounds_the_scored_window_not_the_report(self) -> None:
        text = "A very long sentence holding the torque term deep inside its middle."
        clamped = ground_sentences(text, "torque", max_chars=10)
        whole = ground_sentences(text, "torque")
        # The report still covers the whole sentence either way.
        assert clamped["sentences"][0]["text"] == whole["sentences"][0]["text"] == text
        assert clamped["sentences"][0]["start"] == whole["sentences"][0]["start"]
        # The term sits past the leading window: clamped scores 0, whole
        # scores positive.
        assert clamped["sentences"][0]["score"] == 0.0
        assert whole["sentences"][0]["score"] > 0.0

    def test_a_budget_smaller_than_one_token_still_scores_one_token(self) -> None:
        # The documented graceful floor: the scored window keeps at least
        # one token: the sentence's FIRST token (a budget smaller than
        # any token cannot reach a later one; that is the floor's shape).
        # Capitalized sentences: UAX #29's SB7 joins lowercase-after-
        # lowercase ("here. tail." is ONE sentence; see sentence_bounds).
        result = ground_sentences("Some content here. Tail.", "some", max_chars=1)
        assert result["sentences"][0]["score"] > 0.0
        assert result["sentences"][0]["text"] == "Some content here. "

    def test_zero_max_chars_is_a_value_error(self) -> None:
        with pytest.raises(ValueError):
            ground_sentences("some text here.", "text", max_chars=0)

    def test_negative_max_chars_is_rejected(self) -> None:
        # A negative budget fails the usize conversion, the same behavior
        # highlight's max_chars int parameter shows (the family shape).
        with pytest.raises((ValueError, OverflowError)):
            ground_sentences("some text here.", "text", max_chars=-1)


class TestDegenerateInputs:
    @pytest.mark.parametrize(
        ("text", "query"),
        [
            ("", "some query"),
            ("", ""),
            ("   !!!   ", "some query"),
            ("some query", ""),  # empty text: the empty result
        ],
    )
    def test_degenerate_inputs_are_valid_answers(self, text: str, query: str) -> None:
        result = ground_sentences(text, query)
        assert result == {"sentences": [], "score": 0.0} or all(
            s["score"] == 0.0 for s in result["sentences"]
        )

    def test_an_empty_text_returns_the_empty_result(self) -> None:
        assert ground_sentences("", "query") == {"sentences": [], "score": 0.0}

    def test_an_empty_or_token_free_query_scores_all_zero(self) -> None:
        # The segmentation is the answer's shape; the query only drives
        # scores; every sentence is still reported.
        for query in ("", "   !!!   "):
            result = ground_sentences("Two sentences here. Another one.", query)
            assert len(result["sentences"]) == 2
            assert all(s["score"] == 0.0 for s in result["sentences"])
            assert result["score"] == 0.0

    @pytest.mark.parametrize(
        ("query", "text"),
        [
            ("a", "a" * 10_000 + ". tail."),  # one giant token sentence
            ("a" * 10_000, "a" * 10_000 + ". tail."),
            ("𝕌" * 5000, "𝕌" * 5000 + ". tail."),
            ("e" * 200 + "́", "e" * 200 + "́" + ". tail."),
        ],
    )
    def test_never_panics_on_hostile_input(self, query: str, text: str) -> None:
        result = ground_sentences(text, query)
        for sentence in result["sentences"]:
            assert text[sentence["start"] : sentence["end"]] == sentence["text"]

    def test_pathological_chunk_stays_bounded_and_correct(self) -> None:
        # A 560k-character adversarial repeat: the token cap bounds the
        # scan; the call completes and every sentence slices back exactly.
        text = "relevant term " * 40_000
        result = ground_sentences(text, "relevant term")
        assert len(result["sentences"]) == 1  # one giant sentence segment
        assert text[result["sentences"][0]["start"] : result["sentences"][0]["end"]] == (
            result["sentences"][0]["text"]
        )

    def test_sentences_past_the_token_cap_still_report_exactly(self) -> None:
        # Tokens past the 16384-token cap are not scanned (the documented
        # bounded-scan discipline): a tail sentence whose tokens fall past
        # the cap scores 0.0 while its offsets and text stay exact.
        # Capitalized units: UAX #29's SB7 would join lowercase-after-
        # lowercase units into one giant sentence (see sentence_bounds).
        head = "Word. " * 16_400  # one terminated sentence per unit
        tail = "Relevant term sentence."
        text = head + tail
        result = ground_sentences(text, "Relevant term")
        assert result["sentences"][-1]["text"] == tail
        assert text[result["sentences"][-1]["start"] : result["sentences"][-1]["end"]] == tail
        assert result["sentences"][-1]["score"] == 0.0


class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        text = "tick tock one. tick tock two. tick tock three."
        assert ground_sentences(text, "tick") == ground_sentences(text, "tick")
        assert ground_sentences(text, "tick", max_chars=20) == ground_sentences(
            text, "tick", max_chars=20
        )

    @settings(max_examples=100)
    @given(text=_ANY_TEXT, query=_ANY_QUERY)
    def test_determinism_under_arbitrary_input(self, text: str, query: str) -> None:
        assert ground_sentences(text, query) == ground_sentences(text, query)
