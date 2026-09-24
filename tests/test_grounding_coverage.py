"""Contract gate for ``tors.grounding_coverage``: the recall twin of
``is_grounded``: what fraction of the SOURCE's tokens the text actually
utilizes, one float in [0.0, 1.0], the model-free operationalization of
TRACe's uTilization (Friel, Belyi & Sanyal 2024, RAGBench §3.2).

The pinned behaviors: the empty-operand conventions (0.0, exactly), the
identical-operands identity (1.0, exactly), the unit interval under
arbitrary input, monotonicity under text supersets, the contiguity bias
(a contiguous quote outranks a scatter at equal token counts), the
fold/NFC equivalence, and determinism. The scoring contract is documented
in ``src/grounded_impl.rs``; this file pins what a consumer can rely on.
"""

from __future__ import annotations

import pytest
from grounding_reference import (
    _cov,
    _score_sentence,
    _word_seqs,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import grounding_coverage

_ANY_TEXT = st.text(max_size=200)


class TestConventions:
    def test_empty_and_token_free_operands_are_exactly_zero(self) -> None:
        # The pinned degeneracies: no source tokens, no text tokens, or
        # either operand empty: no coverage to measure, 0.0 (TRACe's
        # ratio is 0/0 there; 0.0 is the conservative reading).
        for source, text in [
            ("", ""),
            ("", "text"),
            ("source", ""),
            ("!!!", "text"),
            ("source", "!!!"),
            ("   ", "   "),
        ]:
            assert grounding_coverage(source, text) == 0.0, (source, text)

    def test_identical_operands_are_one(self) -> None:
        # 1.0 up to f64 rounding in the DP's accumulation (within 1e-9,
        # the documented band); the empty conventions above are exact
        # because no arithmetic runs on those paths.
        for s in ["a", "the quick brown fox", "café über 速い茶色の狐", "x " * 500]:
            assert grounding_coverage(s, s) == pytest.approx(1.0, abs=1e-9), s

    def test_disjoint_operands_are_exactly_zero(self) -> None:
        assert grounding_coverage("alpha bravo charlie", "xray yankee zulu") == 0.0

    def test_cjk_range_punctuation_is_token_free(self) -> None:
        # CJK-range punctuation is NOT a token: UAX #29 segments it, but the
        # grounding tokenizer drops every run with no alphanumeric character,
        # the same rule non-CJK punctuation answers to.  The pin: the CJK
        # sub-split branch applies the drop rule too, so "・" scores 0.0.
        # The sweep covers both branches: U+30FB/U+3099 sit in the CJK
        # sub-split's range; 。、「」〜 are 3000-block, the non-CJK branch.
        for text in ["・", "\u3099", "・。", "。", "、", "「」", "〜", "〽", "！？", "・ ・."]:
            assert not any(ch.isalnum() for ch in text), repr(text)
            assert grounding_coverage(text, text) == 0.0, repr(text)
            assert grounding_coverage(text, "real words") == 0.0, repr(text)
            assert grounding_coverage("real words", text) == 0.0, repr(text)

    def test_real_cjk_words_still_tokenize(self) -> None:
        # The punctuation sweep must not over-correct: real CJK morphemes
        # still tokenize per character and score (identical 1.0; a partial
        # per-character quote partially covers; unrelated CJK is 0.0).
        assert grounding_coverage("重要な情報", "重要な情報") == pytest.approx(1.0)
        assert grounding_coverage("日本語のテキスト", "日本") > 0.0
        assert grounding_coverage("重要な情報", "全く無関係") == 0.0


class TestScoring:
    def test_partial_coverage_is_between_zero_and_one(self) -> None:
        got = grounding_coverage("alpha bravo charlie delta", "alpha bravo")
        assert 0.0 < got < 1.0

    def test_a_contiguous_quote_outranks_a_scatter(self) -> None:
        # The weighted-LCS shaping, as a behavior: the same source tokens
        # quoted contiguously utilize the source more than the same tokens
        # scattered through filler.
        source = "alpha bravo charlie delta echo foxtrot golf hotel"
        contiguous = "alpha bravo charlie delta"
        scattered = "alpha x bravo y charlie z delta"
        assert grounding_coverage(source, contiguous) > grounding_coverage(
            source, scattered
        )

    def test_monotonic_under_text_supersets(self) -> None:
        # Adding source material to the text never lowers coverage.
        source = "alpha bravo charlie delta echo foxtrot golf hotel"
        less = "alpha bravo charlie"
        more = "alpha bravo charlie delta echo foxtrot"
        assert grounding_coverage(source, more) >= grounding_coverage(source, less)

    def test_case_folding_and_nfc_equivalence(self) -> None:
        assert grounding_coverage("The café", "the CAFE\u0301") == pytest.approx(1.0)
        assert grounding_coverage("CAFÉ story", "café story") == pytest.approx(1.0)

    def test_cjk_at_the_tokenizer_granularity(self) -> None:
        # Unspaced CJK tokenizes per character (the grounding family's
        # refinement), so partial quotes partially cover and unrelated CJK
        # scores zero.
        assert grounding_coverage("重要な情報です。", "重要な情報です。") == pytest.approx(1.0)
        partial = grounding_coverage("重要な情報です。", "重要な")
        assert 0.0 < partial < 1.0
        assert grounding_coverage("重要な情報です。", "全く無関係") == 0.0

    def test_the_text_does_not_need_the_source_up_front(self) -> None:
        # Utilization, not containment: the source material can appear in
        # any order and with filler between; this is the recall twin, not
        # is_grounded.
        source = "alpha bravo charlie delta"
        text = "delta then alpha and bravo with charlie last"
        assert grounding_coverage(source, text) > 0.0
        assert source not in text  # containment is the OTHER surface

    def test_asymmetric_arguments(self) -> None:
        # Argument order is the metric's direction: a 2-of-4 coverage is
        # not the same number as a 2-of-2 coverage.
        short = grounding_coverage("alpha bravo charlie delta", "alpha bravo")
        full = grounding_coverage("alpha bravo", "alpha bravo")
        assert short < full == pytest.approx(1.0)


class TestProperties:
    @settings(max_examples=200)
    @given(source=_ANY_TEXT, text=_ANY_TEXT)
    def test_always_in_the_unit_interval_and_deterministic(
        self, source: str, text: str
    ) -> None:
        a = grounding_coverage(source, text)
        b = grounding_coverage(source, text)
        assert a == b
        assert 0.0 <= a <= 1.0

    @settings(max_examples=100)
    @given(text=_ANY_TEXT)
    def test_identical_operands_are_one_under_arbitrary_unicode(
        self, text: str
    ) -> None:
        # Identical operands are FULL coverage unless they are empty (the
        # pinned 0.0 empty convention) or token-free (no tokens to cover).
        got = grounding_coverage(text, text)
        assert got == pytest.approx(1.0) or got == 0.0


class TestHostileInput:
    def test_repetitive_and_giant_token_shapes_stay_bounded(self) -> None:
        shapes = [
            ("word " * 2_000, "word " * 1_000),
            ("a" * 1_000, "a" * 1_000),
            ("word " * 2_000, "a" * 1_000),
        ]
        for source, text in shapes:
            got = grounding_coverage(source, text)
            assert 0.0 <= got <= 1.0

    def test_a_token_flood_operand_stops_at_the_cap(self) -> None:
        # The bounded-scan discipline: at most the first 16384 tokens of
        # each operand are scanned, and the denominator is the SOURCE
        # tokens actually scanned; a source past the cap paired with its
        # own leading window still covers fully.
        source = "word " * 30_000
        assert grounding_coverage(source, source) == pytest.approx(1.0)

    def test_never_panics_on_arbitrary_text(self) -> None:
        # Everything CPython can hand over must produce a number, never an
        # exception. Lone surrogates are absent by design: CPython raises
        # UnicodeEncodeError encoding them before Rust sees any bytes, so
        # no extension API can ever accept them (the same boundary every
        # str parameter of every tors function sits behind).
        for weird in ["\x00\x1b", "\u2029", "\r\n\r\n", "🤷‍♀️" * 50, "\u0301" * 20]:
            assert 0.0 <= grounding_coverage(weird, weird) <= 1.0
            assert 0.0 <= grounding_coverage("abc def", weird) <= 1.0
            assert 0.0 <= grounding_coverage(weird, "abc def") <= 1.0


class TestScoreInvariantTorture:
    """identical -> 1.0 (within 1e-9), token-free -> EXACTLY 0.0, unit
    interval, monotone sane self-minus-one-token values."""

    @pytest.mark.parametrize(
        "text",
        [
            "word " * 16_383 + "word",  # at the 16384-token cap, all one token
            ("ab " * 8_192 + "cd ") * 2,  # alternating at the cap
            "café " * 4_000,  # case/accents
            "日本語 の テキスト " * 2_000,  # CJK
        ],
    )
    def test_identical_operands_score_within_1e9_of_one(self, text):
        text = text[: text.rfind("word") + 4] if "word" in text else text.rstrip()
        text = text.strip()
        got = _cov(text, text)
        assert abs(got - 1.0) < 1e-9, got

    @pytest.mark.parametrize(
        "text",
        ["", "   ", "\t\n\r", "。、！", "。！？", "  ...  ", "🎉 🎉🎉", "👨‍👩‍👧‍👦", " ​﻿"],
    )
    def test_token_free_operands_score_exactly_zero(self, text):
        assert _cov(text, text) == 0.0, repr(text)
        assert _cov(text, "real words here") == 0.0
        assert _cov("real words here", text) == 0.0

    @pytest.mark.parametrize("text", ["・", "・。", "\u3099", "・ ・."])
    def test_cjk_range_punctuation_is_token_free(self, text):
        # Green pin: the non-CJK branch's no-alphanumeric drop rule applies
        # to the CJK sub-split branch too, so CJK-range punctuation is
        # token-free on both branches.
        assert not any(ch.isalnum() for ch in text), "precondition: token-free"
        assert _cov(text, text) == 0.0, repr(text)

    @settings(max_examples=300, deadline=None)
    @given(q=_word_seqs(), c=_word_seqs())
    def test_scores_stay_in_the_unit_interval(self, q, c):
        a = _cov(" ".join(q), " ".join(c))
        b = _score_sentence(" ".join(c) + ".", " ".join(q))
        for got in (a, b):
            assert 0.0 <= got <= 1.0, (q, c, got)

    def test_source_minus_one_token_is_monotone_sane(self):
        words = [f"w{i}" for i in range(10)]  # all distinct: one contiguous run
        source = " ".join(words)
        text = " ".join(words[:-1])
        got = _cov(source, text)
        # Contiguous k-of-n coverage is exactly k/n through Equation 15.
        want = 9 / 10
        assert got == pytest.approx(want, abs=1e-9), (got, want)

    def test_contiguous_prefix_coverage_is_exactly_k_over_n(self):
        words = [f"w{i}" for i in range(50)]
        source = " ".join(words)
        for k in (1, 7, 25, 49):
            got = _cov(source, " ".join(words[:k]))
            assert got == pytest.approx(k / len(words), abs=1e-9), k

    def test_query_longer_than_text_and_vice_versa(self):
        long_source = " ".join(f"s{i}" for i in range(500))
        short_text = "s1 s2"
        got = _cov(long_source, short_text)
        assert 0.0 < got <= 1.0
        got2 = _cov(short_text, long_source)
        assert 0.0 < got2 <= 1.0

    def test_determinism_across_calls_and_argument_reuse(self):
        a = _cov("alpha bravo charlie", "bravo charlie delta")
        b = _cov("alpha bravo charlie", "bravo charlie delta")
        c = _cov("bravo charlie delta", "alpha bravo charlie")
        assert a == b
        assert c == pytest.approx(a, abs=1e-9)  # WLCS symmetric, denominator differs


