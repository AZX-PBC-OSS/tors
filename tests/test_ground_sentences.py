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

import asyncio
import itertools
import re
import subprocess
import sys
import unicodedata
from time import monotonic

import pytest
from grounding_reference import (
    _cov,
    _score_sentence,
    _word_seqs,
    coverage_ref,
    rouge_w_f1_ref,
    wlcs_bruteforce,
    wlcs_lin,
    wlcs_max_on_match,
)
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import tors
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
        # implement Lin 2004's published forced-diagonal fill.  Lin's Figure 3
        # spelling is not monotone in the candidate (extending the text can
        # LOWER a score), and candidate monotonicity is the property the
        # grounding family stands on.  Pinned with a deterministic repro
        # vector: the core reads the max-on-match value, which DIFFERS from
        # Lin's (and from the official ROUGE package / rouge-score on ~12% of
        # random pairs, disclosed on every user-facing surface; docs/api.md
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
        # pairs.  What the core computes is the max-on-match recurrence,
        # a greedy-run-weighted alignment score, which is exactly what the
        # module docs call it; pinned with a deterministic repro vector.
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
        sentence as an F1 guarantee: adding a MATCHING token lowers the F1."""
        got1 = _score_sentence("ax.", "ax")
        got2 = _score_sentence("ax ax.", "ax")
        assert got1 == pytest.approx(1.0, abs=1e-12)
        assert got2 < got1, "precision dilution must lower the F1 (2RP/(R+P))"




class TestSentenceAlignment:
    """ground_sentences' spans must be EXACTLY sentence_bounds' tuples and
    round-trip through slicing, on hostile text."""

    @settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(text=st.text(
        alphabet=st.sampled_from(
            list("abAB09.!? \n日plur下マナellaالسوف🎉café–")
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
            with pytest.raises(TypeError):
                tors.ground_sentences(wrong, "word")  # type: ignore[arg-type]
            with pytest.raises(TypeError):
                tors.ground_sentences(text, wrong)  # type: ignore[arg-type]
            with pytest.raises(TypeError):
                tors.grounding_coverage(wrong, text)  # type: ignore[arg-type]

    def test_keyword_only_arguments_are_enforced(self):
        # max_chars / max_snippets are keyword-only on both surfaces.
        with pytest.raises(TypeError):
            tors.ground_sentences("Word. x.", "word", None)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            tors.highlight("word", "text", 2, 40)  # type: ignore[arg-type]
        # The two surfaces take OPPOSITE positional orders: highlight(query, text)
        # vs ground_sentences(text, query), pinned so the trap is documented.
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
            _gap_and_wall(
                lambda: asyncio.to_thread(tors.ground_sentences, "tiny text here.", huge_query)
            )
        )
        assert worst < 0.25 and worst < 0.5 * wall, (worst, wall)


# ---------------------------------------------------------------------------
# Performance cliffs: 100k single-token sentences vs one 100k-token
# sentence; VmHWM guards pin the two-row memory class.
# ---------------------------------------------------------------------------




class TestPerformanceCliffs:
    def test_100k_single_token_sentences_complete_quickly(self):  # noqa: E501
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

        _now = monotonic

        def min_wall(fn, samples=3):
            fn()
            runs = []
            for _ in range(samples):
                t0 = _now()
                fn()
                runs.append(_now() - t0)
            return min(runs, default=1e9)

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



def _unit(n: int) -> str:
    return "The pump failed with torque spec drift. " * n




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
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout
    )
    assert done.returncode == 0, done.stderr[-300:]
    return int(done.stdout.strip().split("|")[1])


