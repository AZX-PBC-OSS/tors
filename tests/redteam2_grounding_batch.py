"""Wave-2 red-team attack suite for the grounding batch feature.

Scope: the COLLATERAL of the two wave-1 fixes and the attack classes wave 1
skipped (see ``tests/redteam_grounding_batch.py``, studied to avoid
duplication, not re-run):

1. The CJK punctuation filter INVERTED: wave 1 proved CJK punctuation
   (U+30FB, U+3099) is token-free.  Does the per-flushed-run
   ``is_alphanumeric`` drop rule now swallow LEGITIMATE CJK-family tokens?
   Script-class truth table: Yi syllables, Bopomofo, halfwidth Katakana
   (U+FF66-FF9F), halfwidth Hangul (U+FFA0-FFDC), Hangul jamo streams
   (U+1100-11FF), U+3007 (Nl), kana + U+3099 clusters, CJK Extension B,
   fullwidth digits, the long-vowel mark.  The docs' claim "segments with no
   alphanumeric character are dropped" must hold in EVERY is_cjk branch
   without dropping any alphanumeric run.
2. The max-on-match DP's second-order properties: the f(k) = k^1.2 shaping
   spot-checked against hand computations through the public API, the F1
   harmonic bounds (min(P, R) <= F1 <= 1 on random shapes), substitution
   sanity (a synonym never increases the score), and QUERY monotonicity
   (wave 1 pinned candidate/text monotonicity only).
3. ``ground_sentences`` wave 2: embedded UAX #29 separators (VT, FF, NEL,
   LS, PS -- the classes wave 1's generated alphabet missed), texts that are
   ONLY separators, a query longer than every sentence, tie ORDER stability
   (position order, not score order), and the oversized-token max_chars path
   on BOTH surfaces (a 1-giant-token sentence vs max_chars=1).
4. ``grounding_coverage`` wave 2: source/text swapped (the asymmetry pinned
   numerically), verbatim containment vs a naive set-containment heuristic,
   and the 16384-token cap boundary (signal on the last scanned token and
   past it).
5. GIL wave 2: four CONCURRENT 10MB ``grounding_coverage`` calls -- held time
   must stay flat per call (no allocator-contention leak into the GIL
   window); wave 1 measured single detached calls only.
6. docs/api.md's ``highlight`` example recomputed (wave 1 recomputed only the
   ``ground_sentences`` / ``grounding_coverage`` examples) plus the docs'
   "Equation 15's F1 collapses to 2k/(n+m)" identity on fresh shapes.

FINDINGS (report-only, nothing fixed):

- P1, doc drift: api.md and the module docs claim every CJK character
  "(Han, Hiragana, Katakana, Hangul)" inside a word segment becomes its own
  token, with the rationale "without it a Katakana query term could never
  partially match inside a longer Katakana run".  FALSE for the Katakana and
  Hangul blocks is_cjk does not cover: halfwidth Katakana (U+FF66-FF9F),
  halfwidth Hangul (U+FFA0-FFDC) and Hangul jamo streams (U+1100-11FF) stay
  ONE token per UAX #29 run, so a halfwidth-Katakana term cannot partially
  match inside a longer halfwidth run (pinned green below).  The impl doc's
  normative pointer ("see is_cjk") is accurate; the user-facing script list
  and its rationale are not.
- P2, cross-encoding asymmetry: NFC-folded Hangul jamo text ("한국어" spelled
  with U+1100-11FF jamo) and its precomposed spelling produce DIFFERENT
  token streams (the jamo run folds to one composed token, the precomposed
  run splits per char), so neither spelling term-matches the other even
  though NFC is the documented fold.  The docs' NFC claim is scoped to
  accents ("NFD accents match NFC queries"); Hangul falls outside it.
  Pinned green as observed semantics.
- P2, query non-monotonicity: extending the query CAN lower a sentence's
  score (1.0 -> 2/3 below), the F1 precision/recall balance working as
  designed.  No doc claims query monotonicity (the "MORE evidence cannot
  report LESS" sentence is scoped to the candidate/text side), but nothing
  warns the reader either; pinned green so the behavior is at least
  discoverable.
- P2, asymmetry example: the coverage asymmetry is documented only through
  the f^-1(WLCS / f(|source|)) formula -- no numeric example shows that
  swapping the arguments changes the number; the example values are now
  pinned here.
- NO-ARCHAEOLOGY audit of the fix commit (864027c) and its doc pass
  (725d0b0): clean overall -- present tense, no before/after narration.
  Two borderline spots, flagged not fixed: ``grounding_impl.rs``'s fill docs
  say "the fuzz target caught a score DROPPING" and rouge_w_f1's "the fuzz
  target caught exactly that" -- past-tense provenance notes without the
  crash-artifact reference the wave-1 suite carries, so a reader cannot
  reproduce the cited finds.
"""

from __future__ import annotations

import asyncio
import itertools
from time import monotonic

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import tors

# ---------------------------------------------------------------------------
# Hand-computation helpers (independent of the Rust code; the f/finv pair and
# Lin's Equation 15 F1 are re-derived here at the scalar level -- the full
# differential DP references live in the wave-1 suite).
# ---------------------------------------------------------------------------


def _f(k: float) -> float:
    """Lin 2004's shaping function, f(k) = k^1.2."""
    return k**1.2 if k > 0 else 0.0


def _finv(x: float) -> float:
    """The shaping function's inverse (Equation 15's normalization)."""
    return x ** (1.0 / 1.2)


def _wlcs_max_on_match(q: list[str], c: list[str]) -> float:
    """The recurrence the core documents, full-matrix re-derivation: a match
    cell extends the diagonal run only when that beats both skips."""
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
                    s[i][j], g[i][j] = max(up, left), 0
                else:
                    s[i][j], g[i][j] = ext, run
            else:
                s[i][j], g[i][j] = max(up, left), 0
    return s[n][m]


def _f1_parts(q: list[str], c: list[str]) -> tuple[float, float, float]:
    """(F1, min(P, R)) from the documented recurrence, Equation 15."""
    if not q or not c:
        return 0.0, 0.0
    w = _wlcs_max_on_match(q, c)
    if w <= 0.0:
        return 0.0, 0.0
    r = _finv(w / _f(len(q)))
    p = _finv(w / _f(len(c)))
    return min(1.0, max(0.0, 2.0 * r * p / (r + p))), min(r, p)


def _score_sentence(text: str, query: str) -> float:
    """The core's per-sentence F1 for a one-sentence letter-word text."""
    res = tors.ground_sentences(text, query)
    assert len(res["sentences"]) == 1, f"{text!r}: {len(res['sentences'])} sentences"
    return res["sentences"][0]["score"]


def _cov(source: str, text: str) -> float:
    return tors.grounding_coverage(source, text)


_WORDS = ["ba", "ce", "di", "fo", "gu", "ha"]


@st.composite
def _word_seqs(draw, max_size: int = 10) -> list[str]:
    n = draw(st.integers(min_value=1, max_value=max_size))
    return draw(st.lists(st.sampled_from(_WORDS), min_size=n, max_size=n))


# ---------------------------------------------------------------------------
# 1. The CJK filter fix, INVERTED: a script-class truth table.  The invariant
# under attack: a run SURVIVES tokenization iff it contains an alphanumeric
# character, in every is_cjk branch; no legitimate CJK-family token drops.
# ---------------------------------------------------------------------------


class TestCjkFilterCollateral:
    """The alphanumeric drop rule must drop ONLY non-alphanumeric runs."""

    # (label, run, token_expected).  Probe: a single unique run as BOTH
    # operands scores 1.0 iff it produced >= 1 token, 0.0 iff token-free.
    @pytest.mark.parametrize(
        ("label", "run", "token_expected"),
        [
            # CJK proper (is_cjk branch): all alphanumeric, must survive.
            ("Han", "一", True),
            ("Hiragana", "か", True),
            ("Katakana", "ア", True),
            ("Hangul syllable", "한", True),
            # Edge scripts OUTSIDE is_cjk but in the CJK family (non-CJK
            # branch): if any of these dropped, the fix over-reached.
            ("Yi syllable", "ꆈ", True),
            ("Bopomofo", "ㄅ", True),
            ("halfwidth Katakana", "ｱ", True),
            ("halfwidth Hangul", "ﾡ", True),
            ("Hangul jamo", "ᄀ", True),
            ("CJK Extension B", "𠀀", True),
            ("fullwidth digit", "１", True),
            # U+3007, category Nl (alphanumeric): NOT in is_cjk, but UAX #29
            # gives it its own word segment, so it behaves per-character
            # anyway.  Pinned: token survives (alphanumeric).
            ("U+3007 Nl", "〇", True),
            # The long-vowel mark U+30FC (Lm, alphanumeric, inside is_cjk):
            # survives, and splits off its own token inside a Katakana run.
            ("long vowel mark", "ー", True),
            # Kana with an overlapping mark: the U+3099 cluster attaches to
            # the base kana and the RUN is alphanumeric via the base.
            ("kana + U+3099", "か\u3099", True),
            ("Latin + U+3099", "c\u3099", True),
            # Genuinely non-alphanumeric: token-free (the fix's own target).
            ("U+30FB middle dot", "・", False),
            ("U+3099 alone", "\u3099", False),
            ("ideographic full stop", "。", False),
            ("halfwidth ideographic period", "｡", False),
        ],
    )
    def test_truth_table_run_survives_iff_alphanumeric(self, label, run, token_expected):
        assert run == run.strip(), "probe must be a single run"
        got = _cov(run, run)
        want = 1.0 if token_expected else 0.0
        assert got == want, f"{label} {run!r}: coverage {got}, want {want}"

    def test_kana_with_voiced_mark_is_one_token_matching_the_composed_form(self):
        # U+3099 is Mn: an overlapping mark ON a kana.  It must ride its
        # base's grapheme cluster (one token, not a stranded mark) and the
        # NFC fold composes か + U+3099 to が, so the composed query matches.
        decomposed = "か\u3099"
        assert _cov(decomposed, decomposed) == 1.0
        assert _score_sentence(f"{decomposed}き。", "が") > 0.0
        assert _score_sentence("がき。", "が") > 0.0

    def test_cjk_ext_b_and_yi_are_single_tokens_not_dropped(self):
        # Astral-plane Han and Yi: the alphanumeric filter must see the code
        # POINT, not the UTF-8 bytes (a byte-wise check would drop or split).
        assert _cov("𠀀𠀁", "𠀀𠀁") == 1.0
        assert _cov("ꆈꌠ", "ꆈꌠ") == 1.0

    # The docs' per-character claim and its collateral, pinned as OBSERVED
    # semantics (see the P1/P2 findings in the module docstring):
    @pytest.mark.parametrize(
        ("query", "text"),
        [
            # Fullwidth Katakana: the middle of a run IS reachable (docs' claim).
            ("ウ", "アイウエオ。"),
            # Halfwidth Katakana: the middle of a run is NOT (claim falsified).
            ("ｳ", "ｱｲｳｴｵ。"),
            # Halfwidth Hangul likewise.
            ("ﾲ", "ﾱﾲﾳ。"),
        ],
    )
    def test_halfwidth_runs_do_not_subsplit_the_docs_claim_over_reaches(self, query, text):
        score = _score_sentence(text, query)
        if query == "ウ":
            assert score > 0.0, "fullwidth Katakana runs must sub-split"
        else:
            assert score == 0.0, (
                "halfwidth Katakana/Hangul runs stay ONE token (is_cjk does "
                "not cover U+FF66-FF9F/U+FFA0-FFDC); the docs' 'Katakana "
                "query term could never partially match inside a longer "
                "Katakana run' rationale does not hold for halfwidth"
            )

    def test_hangul_jamo_stream_and_precomposed_spelling_never_cross_match(self):
        # Two encodings of the SAME visible word: the jamo stream folds (NFC,
        # per run) to ONE composed token while the precomposed spelling
        # splits per character, so the token streams differ and exact-term
        # matching bridges neither direction.
        jamo = "".join(
            chr(cp)
            for cp in (0x1112, 0x1161, 0x11AB, 0x1100, 0x1165, 0x11A8, 0x110B, 0x1169)
        )
        precomposed = "한국어"
        assert jamo != precomposed, "precondition: different codepoint streams"
        assert _score_sentence(f"{jamo}。", "한국어") == 0.0
        assert _score_sentence(f"{precomposed}。", jamo) == 0.0
        # Each spelling matches ITSELF (query side tokenizes identically).
        assert _score_sentence(f"{jamo}。", jamo) == 1.0
        assert _score_sentence(f"{precomposed}。", precomposed) == 1.0

    def test_u3007_behaves_per_character_through_uax29_not_is_cjk(self):
        # 〇〇〇 splits into three word segments (UAX #29 itself), so the
        # middle 〇 matches even though U+3007 is not in is_cjk; and U+3007
        # breaks the segments around it inside a Latin word too.
        assert _score_sentence("〇〇〇。", "〇") > 0.0
        assert _score_sentence("xe〇fy。", "e〇f") > 0.0


# ---------------------------------------------------------------------------
# 2. The max-on-match DP's second-order properties.
# ---------------------------------------------------------------------------


class TestDpSecondOrder:
    def test_f_shaping_spot_checks_against_hand_computation(self):
        # f(1) = 1.0, f(2) = 2^1.2 ~= 2.2974, f(3) = 3^1.2 ~= 3.7372 all enter
        # the hand-derived value below (q = [ba, ce], c = [ba, x, ce]: two
        # single-match runs, wlcs = f(1) + f(1) = 2.0; R = finv(2/f(2)),
        # P = finv(2/f(3))).  The two skip cells in between force the run
        # resets, so this is the max-on-match recurrence's own value.
        got = _score_sentence("ba x ce.", "ba ce")
        r = _finv(2.0 / _f(2))
        p = _finv(2.0 / _f(3))
        want = 2.0 * r * p / (r + p)
        assert got == pytest.approx(want, abs=1e-9), (got, want)

    def test_contiguous_run_collapses_to_2k_over_n_plus_m(self):
        # The docs' identity (api.md, highlight's example): a candidate whose
        # ONLY match is one contiguous run of k tokens collapses Equation 15
        # to F1 = 2k / (n + m).  Any exponent but 1.2 in f breaks the
        # collapse, so these cells pin the shaping constant end to end.
        for q_words, c_words, k, m, n in [
            (["torque", "spec"], ["The", "bushing", "torque", "spec", "was", "42", "Nm"], 2, 2, 7),
            (["ba", "ce"], ["ba", "ce", "x", "y", "z"], 2, 2, 5),
            (["di"], ["gu", "di", "ha"], 1, 1, 3),
        ]:
            got = _score_sentence(" ".join(c_words) + ".", " ".join(q_words))
            want = 2.0 * k / (n + m)
            assert got == pytest.approx(want, abs=1e-9), (q_words, c_words, got, want)

    def test_scattered_runs_pin_the_exponent_where_the_collapse_does_not(self):
        # The k/n collapse holds for ANY shaping exponent; the SCATTERED
        # value does not: three single-match runs read finv(3.0 / f(3)) =
        # 3^(-1/6), which pins f(3) = 3^1.2 exactly.
        got = _cov("ba ce di", "ba x ce y di")
        want = _finv(3.0 / _f(3))
        assert got == pytest.approx(want, abs=1e-9), (got, want)

    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(q=_word_seqs(), c=_word_seqs())
    def test_f1_is_the_harmonic_mean_between_min_pr_and_1(self, q, c):
        """On random shapes: min(P, R) <= F1 <= 1.  (The harmonic mean of two
        factors in [0, 1] lies between the smaller one and 1; it can exceed
        min(P, R) but never 1, and the core's clamp keeps the ceiling
        airtight.)"""
        f1, min_pr = _f1_parts(q, c)
        got = _score_sentence(" ".join(c) + ".", " ".join(q))
        assert got == pytest.approx(f1, abs=1e-9), (q, c, got, f1)
        assert 0.0 <= got <= 1.0, (q, c, got)
        assert got >= min_pr - 1e-9, (q, c, got, min_pr)

    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(q=_word_seqs(), c=_word_seqs(), idx=st.integers(0, 9))
    def test_substituting_a_synonym_never_increases_the_score(self, q, c, idx):
        """Substitution sanity: replacing a matched TEXT token with a synonym
        (equally absent from every other position) cannot raise the score --
        the lost match weakly lowers both R and P at unchanged lengths."""
        match_positions = [i for i, tok in enumerate(c) if tok in q]
        if not match_positions:
            return  # no match to substitute; other examples still run
        pos = match_positions[idx % len(match_positions)]
        synonym = "zz"  # outside the vocabulary: matches nothing
        base = _score_sentence(" ".join(c) + ".", " ".join(q))
        substituted = _score_sentence(
            " ".join(c[:pos] + [synonym] + c[pos + 1 :]) + ".", " ".join(q)
        )
        assert substituted <= base + 1e-9, (q, c, pos, base, substituted)

    def test_query_monotonicity_does_NOT_hold_pinned(self):
        """Extending the query CAN lower a score (the F1 balance working as
        designed): the added term dilutes recall when absent from the text.
        Pinned -- the docs claim candidate/text monotonicity only, never
        query monotonicity, but nothing warns the reader either (P2)."""
        base = _score_sentence("ba.", "ba")
        extended = _score_sentence("ba.", "ba ce")
        assert base == pytest.approx(1.0, abs=1e-12)
        assert extended == pytest.approx(2.0 / 3.0, abs=1e-9)
        assert extended < base


# ---------------------------------------------------------------------------
# 3. ground_sentences wave 2: separators, degenerate shapes, tie order,
# the oversized-token max_chars path.
# ---------------------------------------------------------------------------


class TestGroundSentencesWave2:
    @settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(text=st.text(
        alphabet=st.sampled_from(list("abAB09.!? word")
                                 + ["\n", "\r", "\x0b", "\x0c", "\x85",
                                    "\u2028", "\u2029", "\u00e9", "\u65e5"]),
        min_size=0,
        max_size=200,
    ))
    def test_embedded_separators_spans_equal_sentence_bounds_exactly(self, text):
        """VT / FF / NEL / LS / PS and friends: UAX #29's separator classes
        must leave the batch's spans EXACTLY the published bounds, slicing
        the original for every sentence."""
        bounds = tors.sentence_bounds(text)
        res = tors.ground_sentences(text, "ba")
        spans = [(s["start"], s["end"]) for s in res["sentences"]]
        assert spans == list(bounds), (text, spans, bounds)
        for s in res["sentences"]:
            assert text[s["start"]: s["end"]] == s["text"], (text, s)
            assert 0.0 <= s["score"] <= 1.0

    @pytest.mark.parametrize(
        "text",
        [
            "ba\r\nce\ndi.",
            "ba\x0bce\x0cdi.",
            "ba\x85ce.   end.",
            "no-break\x0bba.",
            "ba ce di.",
        ],
    )
    def test_targeted_separator_sentences_round_trip(self, text):
        bounds = tors.sentence_bounds(text)
        res = tors.ground_sentences(text, "ba")
        assert [(s["start"], s["end"]) for s in res["sentences"]] == list(bounds)
        assert all(text[s["start"]: s["end"]] == s["text"] for s in res["sentences"])

    @pytest.mark.parametrize(
        "text",
        ["\n\n\n", "\r\n\r\n", "\x0b\x0c\x85", "  ", " \r\n \r\n "],
    )
    def test_only_separator_text_is_a_valid_answer(self, text):
        res = tors.ground_sentences(text, "ba")
        assert res["score"] == 0.0
        for s in res["sentences"]:
            assert text[s["start"]: s["end"]] == s["text"]
            assert s["score"] == 0.0
        got_spans = [(s["start"], s["end"]) for s in res["sentences"]]
        assert got_spans == list(tors.sentence_bounds(text))

    def test_query_longer_than_every_sentence_scores_sanely(self):
        text = "Ba. Ce di."
        res = tors.ground_sentences(text, "a very long query " * 10 + "ba")
        assert len(res["sentences"]) == 2
        assert res["sentences"][0]["score"] > 0.0, "the shared term must anchor"
        assert res["sentences"][1]["score"] == 0.0

    def test_tied_scores_keep_position_order(self):
        text = "Same words here. Same words here. Same words here."
        res = tors.ground_sentences(text, "same words")
        scores = {round(s["score"], 9) for s in res["sentences"]}
        assert len(scores) == 1, scores
        starts = [s["start"] for s in res["sentences"]]
        assert starts == sorted(starts), "ties must keep position order"
        assert starts == [0, 17, 34]
        assert res["score"] == pytest.approx(scores.pop(), abs=1e-9)

    def test_oversized_single_token_vs_max_chars_one(self):
        # One 1000-char token, budget 1: the window floor keeps one token,
        # the report still covers the whole sentence, nothing panics.
        text = "a" * 1000 + ". tail"
        res = tors.ground_sentences(text, "a", max_chars=1)
        s = res["sentences"][0]
        assert text[s["start"]: s["end"]] == s["text"]
        assert len(s["text"]) == 1006, "the report covers the WHOLE sentence"
        assert s["score"] == 0.0, "exact-term matching: the giant token is not 'a'"

    def test_window_cut_with_few_tokens_but_many_chars(self):
        # Two short tokens, budget 3 chars: the window fits ONE token even
        # though the token count is small -- the budget is CHARS.  Scored
        # over the leading token only: hand-derived F1 = 2/3 (one match,
        # P = finv(f(1)/f(1)) = 1, R = finv(f(1)/f(2)) = 1/2 exactly).
        res = tors.ground_sentences("ba ce di.", "ba ce", max_chars=3)
        assert res["sentences"][0]["score"] == pytest.approx(2.0 / 3.0, abs=1e-9)
        # Without the budget the pair scores over all three tokens: one
        # contiguous 2-token run in a 3-token candidate collapses to
        # 2k/(n+m) = 4/5 (the docs' own identity).
        whole = tors.ground_sentences("ba ce di.", "ba ce")
        assert whole["sentences"][0]["score"] == pytest.approx(4.0 / 5.0, abs=1e-9)

    def test_highlight_oversized_token_exceeds_the_budget_by_the_documented_floor(self):
        # The oversized-token path on the snippet surface: a snippet is at
        # least one token even when the token itself busts max_chars.
        text = "a" * 1000 + " tail"
        g = tors.highlight("a" * 1000, text, max_snippets=3, max_chars=1)
        assert len(g["snippets"]) == 1
        s = g["snippets"][0]
        assert len(s["text"]) == 1000 > 1, "the one-token floor exceeds the budget"
        assert text[s["start"]: s["end"]] == s["text"]


# ---------------------------------------------------------------------------
# 4. grounding_coverage wave 2: asymmetry, containment vs naive heuristics,
# the 16384-token cap boundary.
# ---------------------------------------------------------------------------


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


class TestGilConcurrent:
    @pytest.mark.timing
    def test_four_concurrent_10mb_coverages_hold_time_stays_flat(self):
        """Wave 1 audited ONE detached call; here four 10MB
        grounding_coverage calls run concurrently (a real thread pool under
        the gather).  The GIL window is the borrow + the float return, none
        of it scaling with the DP: the worst heartbeat gap must stay flat
        against a single call of the same size, not against the wall (the
        DP work is ~5s)."""
        big = _unit(212_000)  # ~8.5MB, token-capped at 16384 per operand
        single_gap, _ = asyncio.run(
            _gap_and_wall([asyncio.to_thread(tors.grounding_coverage, big, big)])
        )
        worst, wall = asyncio.run(
            _gap_and_wall(
                [asyncio.to_thread(tors.grounding_coverage, big, big) for _ in range(4)]
            )
        )
        assert wall > 1.0, wall  # the DP really ran
        # Flat per call: concurrent held time must not inflate 4x+ with the
        # thread count (allocator contention would show up exactly there).
        assert worst < max(4.0 * single_gap + 0.02, 0.1), (single_gap, worst, wall)
        assert worst < 0.25, "a multi-second call must never hold the GIL this long"


# ---------------------------------------------------------------------------
# 6. docs/api.md's highlight example recomputed (wave 1 covered the other two).
# ---------------------------------------------------------------------------


class TestDocsExamplesWave2:
    def test_api_md_highlight_example(self):
        got = tors.highlight(
            "torque spec",
            "The pump failed. The bushing torque spec was 42 Nm. Replaced.",
        )
        assert got == {
            "snippets": [
                {
                    "text": "The bushing torque spec was 42 Nm. ",
                    "start": 17,
                    "end": 52,
                    "score": 0.44444444444444436,
                }
            ],
            "score": 0.44444444444444436,
        }
        # The docs' parenthetical: the score is exactly 4/9 = 2k/(n+m).
        assert got["score"] == pytest.approx(4.0 / 9.0, abs=1e-12)
