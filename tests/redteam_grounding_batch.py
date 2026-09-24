"""Red-team attack suite for the grounding batch feature (`ground_sentences`,
`grounding_coverage`) and the SHARED ROUGE-W fill change (max-on-match
replacing Lin's forced diagonal) that `highlight` and the coverage core also
run on.  Written by an adversarial reviewer.  Every green test is a failed
attack; the three red attacks the first pass demonstrated (one P0, two P1s,
then xfail(strict)) were fixed in the fix pass and are green pins now — the
findings list below records which, and how.

Reference implementations live IN THIS FILE, in pure Python, independently
derived (three of them, deliberately disagreeing with each other where the
literature does):

- ``wlcs_lin``    — Lin 2004's published WLCS fill verbatim (the paper's
  own dynamic program: a match cell ALWAYS extends the diagonal run,
  ``c(i,j) = c(i-1,j-1) + f(k+1) - f(k)``; run length resets on non-match).
- ``wlcs_max_on_match`` — the recurrence the Rust core actually implements
  (re-derived from the module docs: a match cell may be skipped when
  ``max(up, left)`` beats the run extension).
- ``wlcs_bruteforce`` — the literal "maximum over all monotone matchings"
  definition (exhaustive, tiny inputs only), which tests what the docs
  CLAIM the max-on-match spelling computes.

Findings pinned here (see the redteam report for the full write-up):

- P0 (was xfail, GREEN since the fix pass): CJK-range PUNCTUATION (U+30FB
  katakana middle dot, U+3099) was tokenized by the CJK sub-split branch
  despite the documented "segments with no alphanumeric character are
  dropped" rule, so token-free operands scored 1.0 instead of exactly 0.0.
  Caught by the project's own fuzz target (fuzz_targets/grounding_coverage.rs,
  artifact crash-094c9dae62d4a55cddba683a1444a2be910dc0f3); 120s on
  ground_sentences found nothing.  FIXED: the no-alphanumeric drop rule now
  applies to the CJK sub-split's flushed runs too; the cell is a green pin.
- P1 (was xfail, GREEN since the fix pass): the core's scores are NOT Lin
  2004 ROUGE-W: the published forced-diagonal fill disagrees with the
  max-on-match spelling on ~12% of random token pairs, so tors scores are
  not comparable with the official ROUGE package / rouge-score.  The
  deviation is DELIBERATE (the forced diagonal is not candidate-monotone)
  and now DISCLOSED on every user-facing surface; the cell pins the
  documented max-on-match recurrence as a green test.
- P1 (was xfail, GREEN since the fix pass): the module-doc claim that the
  max-on-match spelling "computes the weighted-LCS optimum over alignments"
  was false: the brute-force oracle finds lower scores on ~0.6% of random
  small pairs (Pareto-stranded runs the two-row DP cannot represent).  The
  docs now say "max-on-match recurrence (a greedy-run-weighted alignment
  score), not the literal weighted-LCS optimum"; the cell pins that
  documented semantics as a green test.
- GREEN: candidate monotonicity (extending the text never lowers
  grounding_coverage) — the property the change was made to buy — holds
  under Hypothesis attack; the core matches an independent re-derivation
  of ITS OWN documented recurrence to 1e-9 on hundreds of shapes; the
  degenerate conventions hold for every token-free shape that does not
  trip the P0; sentence spans are EXACTLY ``sentence_bounds`` tuples on
  hostile text and round-trip; memory stays two-rows-flat at the
  amplification shapes; ``grounding_coverage``'s GIL release does not
  scale with input size.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import re
import subprocess
import sys
import time
from time import monotonic

import pytest
import tors
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Pure-Python reference implementations (independent of the Rust code)
# ---------------------------------------------------------------------------


def _f(k: float) -> float:
    """Lin 2004's shaping function, f(k) = k^1.2 (the paper's polynomial)."""
    return k**1.2 if k > 0 else 0.0


def _finv(x: float) -> float:
    """The shaping function's inverse (Lin's Equation 13/14 normalization)."""
    return x ** (1.0 / 1.2)


def wlcs_lin(q: list[str], c: list[str]) -> float:
    """Lin 2004's published WLCS fill, verbatim: a match cell ALWAYS
    extends the diagonal run (forced diagonal); w resets on non-match."""

    class _Runner:
        def __init__(self) -> None:
            self.value = 0.0

    runner = _Runner()
    n, m = len(q), len(c)
    s = [[0.0] * (m + 1) for _ in range(n + 1)]
    g = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if q[i - 1] == c[j - 1]:
                k = g[i - 1][j - 1]
                s[i][j] = s[i - 1][j - 1] + _f(k + 1) - _f(k)
                g[i][j] = k + 1
            else:
                if s[i - 1][j] > s[i][j - 1]:
                    s[i][j] = s[i - 1][j]
                else:
                    s[i][j] = s[i][j - 1]
                g[i][j] = 0
    runner.value = s[n][m]
    return runner.value


def wlcs_max_on_match(q: list[str], c: list[str]) -> float:
    """The recurrence the core documents (independent full-matrix
    re-derivation): a match cell takes the diagonal extension only when it
    beats both skips, resetting the run when a skip wins."""

    class _Runner:
        def __init__(self) -> None:
            self.value = 0.0

    runner = _Runner()
    n, m = len(q), len(c)
    s = [[0.0] * (m + 1) for _ in range(n + 1)]
    g = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            up, left = s[i - 1][j], s[i][j - 1]
            if q[i - 1] == c[j - 1]:
                run = g[i - 1][j - 1] + 1
                ext = s[i - 1][j - 1] + _f(run) - _f(run - 1)
                if up > ext or left > ext:
                    s[i][j] = max(up, left)
                    g[i][j] = 0
                else:
                    s[i][j] = ext
                    g[i][j] = run
            else:
                s[i][j] = max(up, left)
                g[i][j] = 0
    runner.value = s[n][m]
    return runner.value


def wlcs_bruteforce(q: list[str], c: list[str]) -> float:
    """The literal maximum over all monotone matchings: sum f(run) over
    maximal consecutive runs.  Exponential; tiny inputs only."""

    class _Runner:
        def __init__(self) -> None:
            self.best = 0.0

    runner = _Runner()
    n, m = len(q), len(c)

    def rec(i: int, j: int, pairs: list[tuple[int, int]]) -> None:
        value = 0.0
        run = 0
        prev: tuple[int, int] | None = None
        for a, b in pairs:
            if prev is not None and a == prev[0] + 1 and b == prev[1] + 1:
                run += 1
            else:
                run = 1
            value += _f(run) - _f(run - 1)
            prev = (a, b)
        runner.best = max(runner.best, value)
        if i >= n or j >= m:
            return
        rec(i + 1, j, pairs)
        rec(i, j + 1, pairs)
        if q[i] == c[j]:
            rec(i + 1, j + 1, pairs + [(i, j)])

    rec(0, 0, [])
    return runner.best


def rouge_w_f1_ref(q: list[str], c: list[str], wlcs=wlcs_max_on_match) -> float:
    """Equation 15's F1 (beta = 1) over a WLCS fill, Lin's Equation 13/14
    normalization through f^-1, defensively clamped to [0, 1]."""
    if not q or not c:
        return 0.0
    w = wlcs(q, c)
    if w <= 0.0:
        return 0.0
    r = _finv(w / _f(len(q)))
    p = _finv(w / _f(len(c)))
    return min(1.0, max(0.0, 2.0 * r * p / (r + p)))


def coverage_ref(source: list[str], text: list[str]) -> float:
    """The coverage core's documented score: Equation 15's R factor
    (f^-1(WLCS / f(|source|))) over the max-on-match fill, clamped."""
    if not source or not text:
        return 0.0
    w = wlcs_max_on_match(source, text)
    if w <= 0.0:
        return 0.0
    return min(1.0, max(0.0, _finv(w / _f(len(source)))))


# ---------------------------------------------------------------------------
# Tokenization-locked lanes: single-sentence ASCII/letter texts where the
# tokenizer's output is exactly the space-separated words, so the reference
# and the core score the same token streams.
# ---------------------------------------------------------------------------

_WORDS = ["ax", "by", "cz", "dd", "ee", "ff"]


def _score_sentence(text: str, query: str) -> float:
    """The core's per-sentence F1 for a one-sentence text."""
    res = tors.ground_sentences(text, query)
    assert len(res["sentences"]) == 1, f"{text!r}: {len(res['sentences'])} sentences"
    return res["sentences"][0]["score"]


def _cov(source: str, text: str) -> float:
    return tors.grounding_coverage(source, text)


@st.composite
def _word_seqs(draw, max_size=12):
    n = draw(st.integers(min_value=1, max_value=max_size))
    return draw(st.lists(st.sampled_from(_WORDS), min_size=n, max_size=n))


class TestDifferentialAgainstReferences:
    """The shared fill, differential-tested against the in-file references."""

    @settings(max_examples=250, deadline=None)
    @given(q=_word_seqs(), c=_word_seqs())
    def test_core_f1_matches_its_own_documented_recurrence(self, q, c):
        """The core vs an independent re-derivation of the recurrence the
        docs SAY it implements (max-on-match).  Any disagreement = P0."""
        want = rouge_w_f1_ref(q, c)
        got = _score_sentence(" ".join(c) + ".", " ".join(q))
        assert abs(got - want) <= 1e-9, (q, c, got, want)

    @settings(max_examples=250, deadline=None)
    @given(q=_word_seqs(), c=_word_seqs())
    def test_core_coverage_matches_its_own_documented_recurrence(self, q, c):
        want = coverage_ref(q, c)
        got = _cov(" ".join(q), " ".join(c))
        assert abs(got - want) <= 1e-9, (q, c, got, want)

    def test_core_uses_the_monotone_max_on_match_recurrence(self):
        # DOCUMENTED DEVIATION, not a bug: the core deliberately does NOT
        # implement Lin 2004's published forced-diagonal fill — Lin's Figure 3
        # spelling is not monotone in the candidate (extending the text can
        # LOWER a score), and candidate monotonicity is the property the
        # grounding family stands on.  Pinned with the red-team repro vector:
        # the core reads the max-on-match value, which DIFFERS from Lin's
        # (and from the official ROUGE package / rouge-score on ~12% of
        # random pairs — disclosed on every user-facing surface, docs/api.md
        # included).
        q = ["a", "a", "c", "a", "b", "b"]
        c = ["b", "b", "a", "a", "b", "b"]
        assert wlcs_max_on_match(q, c) != pytest.approx(wlcs_lin(q, c), abs=1e-9)
        got = _score_sentence(" ".join(c) + ".", " ".join(q))
        want = rouge_w_f1_ref(q, c, wlcs=wlcs_max_on_match)
        assert got == pytest.approx(want, abs=1e-9), (got, want)
        want_lin = rouge_w_f1_ref(q, c, wlcs=wlcs_lin)
        assert got != pytest.approx(want_lin, abs=1e-9), (got, want_lin)

    def test_two_row_dp_is_not_the_literal_wlcs_optimum_documented(self):
        # DOCUMENTED DEVIATION, not a bug: the two-row DP cannot represent
        # Pareto (value, trailing-run) states, so the literal max over all
        # monotone matchings (the brute-force oracle) reads HIGHER on rare
        # pairs.  What the core computes is the max-on-match recurrence — a
        # greedy-run-weighted alignment score — which is exactly what the
        # module docs now call it; pinned with the red-team repro vector.
        q = ["b", "c", "a", "a", "b", "a", "c"]
        c = ["c", "a", "b", "a"]
        assert wlcs_bruteforce(q, c) > wlcs_max_on_match(q, c) + 1e-9
        got = _score_sentence(" ".join(c) + ".", " ".join(q))
        want = rouge_w_f1_ref(q, c, wlcs=wlcs_max_on_match)
        assert got == pytest.approx(want, abs=1e-9), (got, want)


class TestMonotonicity:
    """The candidate-monotonicity property the max-on-match change exists
    to guarantee, attacked through the real API."""

    @settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        q=_word_seqs(max_size=8),
        c=_word_seqs(max_size=8),
        extra=_word_seqs(max_size=6),
    )
    def test_extending_the_text_never_lowers_coverage(self, q, c, extra):
        source = " ".join(q)
        base = _cov(source, " ".join(c))
        extended = _cov(source, " ".join(c + extra))
        assert extended >= base - 1e-9, (q, c, extra, base, extended)

    @settings(max_examples=100, deadline=None)
    @given(q=_word_seqs(max_size=8), c=_word_seqs(max_size=8), extra=_word_seqs(max_size=4))
    def test_extending_the_text_never_lowers_the_wlcs_itself(self, q, c, extra):
        """The doc's premise (WLCS monotone in the candidate), verified on
        the core's own recurrence re-derivation."""
        base = wlcs_max_on_match(q, c)
        ext = wlcs_max_on_match(q, c + extra)
        assert ext >= base - 1e-9, (q, c, extra, base, ext)

    def test_f1_is_NOT_monotone_precision_dilution_is_documented(self):
        """The module doc's 'a scoring function offered MORE evidence cannot
        report LESS' is FALSE for the F1 surfaces (highlight /
        ground_sentences): precision dilution.  Pinned so nobody reads the
        sentence as an F1 guarantee — adding a MATCHING token lowers the F1."""
        got1 = _score_sentence("ax.", "ax")
        got2 = _score_sentence("ax ax.", "ax")
        assert got1 == pytest.approx(1.0, abs=1e-12)
        assert got2 < got1, "precision dilution must lower the F1 (2RP/(R+P))"


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
        # Was the P0 red cell (xfail strict); the fix pass applied the
        # non-CJK branch's no-alphanumeric drop rule to the CJK sub-split
        # branch, so this is now a green pin.
        assert not any(ch.isalnum() for ch in text), "precondition: token-free"
        assert _cov(text, text) == 0.0, repr(text)

    @settings(max_examples=300, deadline=None)
    @given(q=_word_seqs(), c=_word_seqs())
    def test_scores_stay_in_the_unit_interval(self, q, c):
        for got in (_cov(" ".join(q), " ".join(c)), _score_sentence(" ".join(c) + ".", " ".join(q))):
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


class TestSentenceAlignment:
    """ground_sentences' spans must be EXACTLY sentence_bounds' tuples and
    round-trip through slicing, on hostile text."""

    @settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(text=st.text(
        alphabet=st.sampled_from(
            list("abAB09.!? \n日plur下マナellaالسوف🎉café—")
            + ["\u0301", "\u200d", "珠", "\U0001F3E0", "\U0001F468", "\U0001F466", "\r"]
        ),
        min_size=0,
        max_size=300,
    ))
    def test_spans_are_exactly_sentence_bounds_and_round_trip(self, text):
        bounds = tors.sentence_bounds(text)
        res = tors.ground_sentences(text, "test query")
        spans = [(s["start"], s["end"]) for s in res["sentences"]]
        assert spans == list(bounds), (text, spans, bounds)
        for s in res["sentences"]:
            assert text[s["start"]: s["end"]] == s["text"], (text, s)
        if res["sentences"]:
            assert res["score"] == max(s["score"] for s in res["sentences"])
        else:
            assert res["score"] == 0.0

    def test_targeted_hostile_texts(self):
        cases = [
            "第一句。第二句！第三句？",
            "الجملة الأولى. الجملة الثانية؟ والثالثة.",
            "👨‍👩‍👧‍👦 family. cafe\u0301 NFD. café NFC.",
            "no terminal punctuation at all",
            "\n\n\nonly newlines\n\n",
            "word" * 2_500 + ".",  # long single sentence, no breaks
        ]
        for text in cases:
            bounds = tors.sentence_bounds(text)
            res = tors.ground_sentences(text, "test")
            assert [(s["start"], s["end"]) for s in res["sentences"]] == list(bounds), text
            assert all(text[s["start"]: s["end"]] == s["text"] for s in res["sentences"]), text

    def test_10k_sentences_stay_exact(self):
        text = "Sentence number one! " * 10_000
        res = tors.ground_sentences(text, "number one")
        assert len(res["sentences"]) == 10_000
        for s in res["sentences"]:
            assert text[s["start"]: s["end"]] == s["text"]
        assert res["score"] == max(s["score"] for s in res["sentences"])

    def test_max_chars_window_reports_full_span_scores_leading_window(self):
        text = "word " * 50 + "end."
        res = tors.ground_sentences(text, "word end", max_chars=12)
        s = res["sentences"][0]
        assert text[s["start"]: s["end"]] == s["text"], "reported span must round-trip"
        assert len(s["text"]) > 12, "the reported span covers the WHOLE sentence"
        # The score must correspond to the leading token-boundary window:
        # here the window is the first two tokens ("word word", 9 chars fit,
        # 14 do not); recompute the F1 with the in-file reference.
        want = rouge_w_f1_ref(["word", "end"], ["word", "word"])
        assert s["score"] == pytest.approx(want, abs=1e-9), (s["score"], want)

    def test_max_chars_single_giant_token_floor(self):
        text = "a" * 100 + ". tail"
        res = tors.ground_sentences(text, "a", max_chars=5)
        s = res["sentences"][0]
        assert text[s["start"]: s["end"]] == s["text"]
        assert 0.0 <= s["score"] <= 1.0

    def test_tokens_past_the_16384_cap_score_zero_with_exact_offsets(self):
        head = "Word! " * 20_000  # 20k tokenized words, cap 16384
        tail = "target sentence here."
        text = head + tail
        res = tors.ground_sentences(text, "target sentence")
        last = res["sentences"][-1]
        assert last["text"] == tail
        assert text[last["start"]: last["end"]] == tail
        assert last["score"] == 0.0, "the tail's tokens sit past the cap"


class TestApiAbuse:
    """Wrong types, hostile parameter values, no panics."""

    def test_hostile_max_chars_values(self):
        text = "Word. Sentence."
        assert pytest.raises(ValueError, tors.ground_sentences, text, "word", max_chars=0)
        with pytest.raises((TypeError, OverflowError, ValueError)):
            tors.ground_sentences(text, "word", max_chars=-1)
        with pytest.raises(TypeError):
            tors.ground_sentences(text, "word", max_chars=1.5)
        with pytest.raises(TypeError):
            tors.ground_sentences(text, "word", max_chars=float("nan"))
        with pytest.raises(TypeError):
            tors.ground_sentences(text, "word", max_chars=float("inf"))
        # Huge-but-legal values must not overflow window arithmetic.
        for huge in (2**31, 2**63 - 1, 2**64 - 1):
            res = tors.ground_sentences(text, "word", max_chars=huge)
            assert len(res["sentences"]) == 2
        for wrong in (b"word", None, ["word"], 3.5, object()):
            with pytest.raises(Exception):
                tors.ground_sentences(wrong, "word")  # type: ignore[arg-type]
            with pytest.raises(Exception):
                tors.ground_sentences(text, wrong)  # type: ignore[arg-type]
            with pytest.raises(Exception):
                tors.grounding_coverage(wrong, text)  # type: ignore[arg-type]

    def test_keyword_only_arguments_are_enforced(self):
        # max_chars / max_snippets are keyword-only on both surfaces.
        with pytest.raises(TypeError):
            tors.ground_sentences("Word. x.", "word", None)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            tors.highlight("word", "text", 2, 40)  # type: ignore[arg-type]
        # The two surfaces take OPPOSITE positional orders: highlight(query, text)
        # vs ground_sentences(text, query) — pinned so the trap is documented.
        res = tors.ground_sentences("Word. X.", "word")
        assert len(res["sentences"]) == 2
        res2 = tors.highlight("word", "Word. x.")
        assert res2["snippets"]

    def test_adversarial_text_shapes_never_panic(self):
        shapes = [
            "ab" * 10_000,
            "aabab " * 5_000,
            ("ő" * 100 + " ") * 100,
            "🤷‍♂️ x " * 500,
            "ا" * 5_000,
            "с" * 1_000 + "о" * 1_000,
            "x" * 10_000,
            " " * 1_000,
        ]
        for text in shapes:
            for query in ("x", text[:50], ""):
                res = tors.ground_sentences(text, query)
                assert all(
                    text[s["start"]: s["end"]] == s["text"] and 0.0 <= s["score"] <= 1.0
                    for s in res["sentences"]
                )
                assert 0.0 <= tors.grounding_coverage(text, query) <= 1.0

    def test_aio_twin_semantics_match_sync(self):
        import tors.aio

        text = "The pump failed. The bushing torque spec was 42 Nm. Replaced."
        sync = tors.ground_sentences(text, "torque spec")
        async_out = asyncio.run(tors.aio.ground_sentences(text, "torque spec"))
        assert sync == async_out
        assert asyncio.run(tors.aio.grounding_coverage("a b c", "b c")) == _cov("a b c", "b c")


# ---------------------------------------------------------------------------
# GIL-claim audit: heartbeat harness, hostile sizes; held time must not
# scale with input size for the detached pass.
# ---------------------------------------------------------------------------


async def _gap_and_wall(op):
    ticks = []
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
        await (op() if callable(op) else op)
        end = monotonic()
    finally:
        stop.set()
        await task
    wall = end - started
    worst = max((b - a for a, b in itertools.pairwise(ticks)), default=0.0)
    return worst, wall


def _unit(n: int) -> str:
    return "The pump failed with torque spec drift. " * n


class TestGilClaimAudit:
    """Both new surfaces claim py.detach end-to-end with no Python callback
    inside the detached pass.  Heartbeat at hostile sizes."""

    def test_the_harness_detects_a_gil_held_pass(self):
        """Red side: a pure-Python CPU loop in a thread holds the GIL; the
        heartbeat must see it (proves the harness can fail)."""

        def gil_held_c_call():
            # re.sub over a large string: one whole-text GIL-held C pass
            # (measured ~247ms gap of a ~247ms wall), the suite's documented
            # red side for every detach claim.
            return re.sub("x", "y", "x" * 30_000_000)

        worst, wall = asyncio.run(_gap_and_wall(lambda: asyncio.to_thread(gil_held_c_call)))
        assert wall > 0.1 and worst > 0.1, (worst, wall)

    @pytest.mark.timing
    @pytest.mark.parametrize("units", [24_000, 96_000, 212_000])  # ~1 / 4 / 9.5 MiB
    def test_ground_sentences_heartbeat_at_hostile_sizes(self, units):
        big = _unit(units)
        worst, wall = asyncio.run(
            _gap_and_wall(lambda: asyncio.to_thread(tors.ground_sentences, big, "torque spec"))
        )
        assert wall > 0.02, wall
        # Detached pass + documented O(sentences) marshalling residue: the
        # gap must stay far under the wall (a held whole pass reads ~1.0).
        assert worst < 0.35 * wall and worst < 0.25, (worst, wall)

    @pytest.mark.timing
    @pytest.mark.parametrize("units", [24_000, 212_000])  # ~1 / 9.5 MiB
    def test_grounding_coverage_held_time_does_not_scale(self, units):
        big = _unit(units)
        small_gap, _ = asyncio.run(
            _gap_and_wall(
                lambda: asyncio.to_thread(tors.grounding_coverage, _unit(24_000), _unit(24_000))
            )
        )
        worst, wall = asyncio.run(
            _gap_and_wall(lambda: asyncio.to_thread(tors.grounding_coverage, big, big))
        )
        # The point of detach: held time must not scale with input.  The
        # DP is 100x the cells at 10x the side; the GIL-held gap must stay
        # flat (measured 10.8ms -> 15.3ms for 1MB -> 10MB).
        assert worst < max(2.0 * small_gap + 0.02, 0.05), (small_gap, worst, wall)

    @pytest.mark.timing
    def test_tiny_text_huge_query_heartbeat(self):
        huge_query = "torque spec " * 900_000  # ~10 MiB query, tiny text
        worst, wall = asyncio.run(
            _gap_and_wall(lambda: asyncio.to_thread(tors.ground_sentences, "tiny text here.", huge_query))
        )
        assert worst < 0.25 and worst < 0.5 * wall, (worst, wall)


# ---------------------------------------------------------------------------
# Performance cliffs: 100k single-token sentences vs one 100k-token
# sentence; VmHWM guards pin the two-row memory class.
# ---------------------------------------------------------------------------


def _child_hwm(call: str, setup: str = "", timeout: float = 120.0):
    code = f"""\
import tors
{setup}
{call}
hwm = 0
with open("/proc/self/status") as status:
    for line in status:
        if line.startswith("VmHWM:"):
            hwm = int(line.split()[1])
print(f"RESULT|{{hwm}}")
"""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
    assert done.returncode == 0, done.stderr[-300:]
    return int(done.stdout.strip().split("|")[1])


class TestPerformanceCliffs:
    def test_100k_single_token_sentences_complete_quickly(self):
        soup = "Word. " * 100_000
        started = monotonic()
        res = tors.ground_sentences(soup, "word")
        wall = monotonic() - started
        assert len(res["sentences"]) == 100_000
        assert res["score"] > 0.9
        assert wall < 5.0, wall

    def test_one_100k_token_sentence_memory_stays_two_rows(self):
        """The claimed two-row DP must keep peak RSS flat on the
        one-giant-sentence shape (a materialized matrix would be ~GBs)."""
        hwm = _child_hwm(
            "r = tors.ground_sentences('word ' * 100_000, 'word')",
            timeout=60.0,
        )
        assert hwm < 200_000, f"peak {hwm / 1024:.0f} MiB: not the two-row class"

    def test_coverage_at_both_caps_memory_is_two_rows(self):
        hwm = _child_hwm(
            "r = tors.grounding_coverage('word ' * 16_384, 'word ' * 16_384)",
            timeout=120.0,
        )
        assert hwm < 200_000, f"peak {hwm / 1024:.0f} MiB: not the two-row class"

    @pytest.mark.timing
    def test_ground_sentences_wall_scaling_is_linear(self):
        def shape(tokens: int):
            tors.ground_sentences(("Word. " * (tokens // 2))[:-1], "word")

        def min_wall(fn, samples=3):
            fn()
            return min(min((lambda t0=monotonic(): (fn(), monotonic() - t0)[1])() for _ in range(samples)), 1e9)

        small = min_wall(lambda: shape(4_000))
        large = min_wall(lambda: shape(16_000))
        assert large < small * (3.0 ** 2) * 1.2, (small, large)  # 4x input, 3x/doubling


# ---------------------------------------------------------------------------
# Doc examples: every number in docs/api.md must reproduce exactly.
# ---------------------------------------------------------------------------


class TestDocExamplesAreHonest:
    def test_api_md_ground_sentences_example(self):
        got = tors.ground_sentences(
            "The pump failed. The bushing torque spec was 42 Nm. Replaced.", "torque spec"
        )
        assert got == {
            "sentences": [
                {"text": "The pump failed. ", "start": 0, "end": 17, "score": 0.0},
                {
                    "text": "The bushing torque spec was 42 Nm. ",
                    "start": 17,
                    "end": 52,
                    "score": 0.44444444444444436,
                },
                {"text": "Replaced.", "start": 52, "end": 61, "score": 0.0},
            ],
            "score": 0.44444444444444436,
        }

    def test_api_md_grounding_coverage_examples(self):
        assert _cov("the quick brown fox jumps over the lazy dog", "the lazy dog jumps") == (
            0.33333333333333337
        )
        assert _cov("same words both sides", "same words both sides") == 1.0
        assert _cov("alpha bravo charlie", "xray yankee zulu") == 0.0
        assert _cov("", "text") == 0.0
