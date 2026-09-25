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

import asyncio
import itertools
from time import monotonic

import pytest
from grounding_reference import (
    _cov,
    _f,
    _finv,
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


class TestGroundingCoverageWave2:
    def test_asymmetry_pinned_numerically(self):
        """coverage(source, text) normalizes by the SOURCE; swapped operands
        give a different number (recall of a different denominator).  The
        docs carry the f(|source|) formula but no numeric example; pinned
        here: a 3-of-5 contiguous overlap reads 1.0 one way, exactly 3/5 the
        other (Equation 15: finv(f(3)/f(5)) = 3/5)."""
        five = "ba ce di fo gu"
        three = "ba ce di"
        assert _cov(three, five) == pytest.approx(1.0, abs=1e-9)
        assert _cov(five, three) == pytest.approx(3.0 / 5.0, abs=1e-9)
        assert _cov(three, five) != pytest.approx(_cov(five, three), abs=1e-9)

    def test_verbatim_containment_inside_a_larger_body_is_exactly_one(self):
        """Text that CONTAINS the source verbatim inside a larger body covers
        the source fully: recall normalizes by the source alone, so this is
        exactly 1.0 -- a naive containment heuristic agrees here.  The two
        DISAGREE on the scattered case below, which is the point of the
        contiguity shaping."""
        assert _cov("ba ce di", "x ba ce di y") == pytest.approx(1.0, abs=1e-9)

    def test_scattered_coverage_diverges_from_a_naive_containment_heuristic(self):
        """A naive set-containment heuristic ('every source token appears in
        the text') reads 1.0 here; ROUGE-W recall reads finv(3/f(3)) = 3^(-1/6)
        ~ 0.83 -- contiguity is the signal a set overlap cannot see."""
        source = "ba ce di"
        text = "ba x ce y di"
        assert all(tok in text.split() for tok in source.split()), "precondition"
        naive = 1.0
        got = _cov(source, text)
        assert got == pytest.approx(_finv(3.0 / _f(3)), abs=1e-9)
        assert got < naive, "the shaping must price the scattered quoting below naive containment"

    def test_cap_boundary_signal_on_the_last_scanned_token_and_past_it(self):
        """16384-token cap: 'alpha' is the LAST scanned source token (the
        denominator is the scanned 16384), 'beta' sits past the cap and is
        silently ignored -- the documented bounded-scan discipline.  Hand
        values: finv(f(1)/f(16384)) = 1/16384 exactly; two contiguous head
        tokens read finv(f(2)/f(16384)) = 2/16384 exactly (Equation 15)."""
        noise = " ".join(f"n{i}" for i in range(16_383))
        source = f"{noise} alpha beta"
        scanned = 16_384
        whole = _cov(source, "alpha beta")
        assert whole == pytest.approx(_finv(_f(1.0) / _f(scanned)), abs=1e-12)
        assert whole == pytest.approx(1.0 / scanned, rel=1e-9)
        # Past the cap: silently ignored (documented), exactly 0.0.
        assert _cov(source, "beta") == 0.0
        # The last scanned token alone scores the same full-recall shape.
        assert _cov(source, "alpha") == pytest.approx(_finv(_f(1.0) / _f(scanned)), abs=1e-12)
        # Head tokens: two contiguous matches over the scanned denominator.
        assert _cov(source, "n0 n1") == pytest.approx(_finv(_f(2.0) / _f(scanned)), abs=1e-12)

    def test_text_past_the_cap_ignored_even_when_it_is_all_the_signal(self):
        # Inverse shape: the SOURCE fits under the cap; the TEXT's signal
        # sits past ITS cap.  The text's tail is not measured, so the score
        # reads only the text's leading 16384 tokens.
        source = "alpha beta"
        text = f"{'filler '.join(['x'] * 16_384)}alpha beta"
        got = _cov(source, text)
        assert 0.0 <= got < 1.0
        assert got < _cov(source, "alpha beta"), (
            "the same signal, pushed past the text cap, must not score full"
        )


# ---------------------------------------------------------------------------
# 5. GIL wave 2: four CONCURRENT 10MB calls -- held time per call must stay
# flat (no allocator-contention leak into the GIL window).
# ---------------------------------------------------------------------------


async def _gap_and_wall(calls):
    """Run `calls` concurrently under a 10ms heartbeat; return (worst gap,
    total wall)."""
    ticks: list[float] = []
    stop = asyncio.Event()

    async def heartbeat():
        while True:
            ticks.append(monotonic())
            if stop.is_set():
                return
            await asyncio.sleep(0.01)

    task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    started = monotonic()
    try:
        await asyncio.gather(*calls)
        end = monotonic()
    finally:
        stop.set()
        await task
    worst = max((b - a for a, b in itertools.pairwise(ticks)), default=0.0)
    return worst, end - started


def _unit(n: int) -> str:
    return "The pump failed with torque spec drift. " * n


