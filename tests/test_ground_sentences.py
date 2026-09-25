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
from loop_harness import assert_heartbeat_clean, heartbeat_gap_and_wall

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
# GIL-claim audit: heartbeat harness (tests/loop_harness.py, the shared
# measurement), hostile sizes; held time must not scale with input size
# for the detached pass.
# ---------------------------------------------------------------------------


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

        worst, wall = asyncio.run(
            heartbeat_gap_and_wall(lambda: asyncio.to_thread(gil_held_c_call))
        )
        assert wall > 0.1 and worst > 0.1, (worst, wall)

    @pytest.mark.timing
    def test_the_shared_harness_budgets_fail_a_gil_held_pass(self):
        """Red side for the shared budget shape: the same GIL-held C call
        must FAIL ``assert_heartbeat_clean`` (a held pass reads a ~1.0
        ratio in every sample), not merely report a large gap."""
        def gil_held() -> object:
            return asyncio.to_thread(re.sub, "x", "y", "x" * 30_000_000)

        with pytest.raises(AssertionError, match="missed its budgets"):
            asyncio.run(assert_heartbeat_clean(gil_held))

    @pytest.mark.timing
    @pytest.mark.parametrize("units", [24_000, 96_000, 212_000])  # ~1 / 4 / 9.5 MiB
    def test_ground_sentences_heartbeat_at_hostile_sizes(self, units):
        big = _unit(units)
        # Detached pass + documented O(sentences) marshalling residue: the
        # gap must stay far under the wall (a held whole pass reads ~1.0).
        asyncio.run(
            assert_heartbeat_clean(
                lambda: asyncio.to_thread(tors.ground_sentences, big, "torque spec")
            )
        )

    @pytest.mark.timing
    @pytest.mark.parametrize("units", [24_000, 212_000])  # ~1 / 9.5 MiB
    def test_grounding_coverage_held_time_does_not_scale(self, units):
        big = _unit(units)
        small_gap, _ = asyncio.run(
            heartbeat_gap_and_wall(
                lambda: asyncio.to_thread(tors.grounding_coverage, _unit(24_000), _unit(24_000))
            )
        )
        worst, wall = asyncio.run(
            heartbeat_gap_and_wall(lambda: asyncio.to_thread(tors.grounding_coverage, big, big))
        )
        # The point of detach: held time must not scale with input.  The
        # DP is 100x the cells at 10x the side; the GIL-held gap must stay
        # flat (measured 10.8ms -> 15.3ms for 1MB -> 10MB).
        assert worst < max(2.0 * small_gap + 0.02, 0.05), (small_gap, worst, wall)

    @pytest.mark.timing
    def test_tiny_text_huge_query_heartbeat(self):
        huge_query = "torque spec " * 900_000  # ~10 MiB query, tiny text
        worst, wall = asyncio.run(
            heartbeat_gap_and_wall(
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
    def test_substituting_a_synonym_never_increases_the_optimum(self, q, c, idx):
        """Substitution sanity at the optimum: replacing a matched TEXT token
        with a synonym (equally absent from every other position) cannot
        raise the true weighted-LCS optimum -- the synonym matches nothing,
        so every alignment of the substituted pair is an alignment of the
        original and the optimum can only shrink.  The core's greedy
        max-on-match fill is a different animal: it is not the optimum (its
        own pinned deviation), and unblocking a stranded run can raise its
        score -- the counterexample is pinned in its own cell below."""
        match_positions = [i for i, tok in enumerate(c) if tok in q]
        if not match_positions:
            return  # no match to substitute; other examples still run
        pos = match_positions[idx % len(match_positions)]
        synonym = "zz"  # outside the vocabulary: matches nothing
        c_sub = c[:pos] + [synonym] + c[pos + 1 :]
        base_opt = rouge_w_f1_ref(q, c, wlcs=wlcs_bruteforce)
        sub_opt = rouge_w_f1_ref(q, c_sub, wlcs=wlcs_bruteforce)
        assert sub_opt <= base_opt + 1e-9, (q, c, pos, base_opt, sub_opt)

    def test_a_synonym_substitution_can_raise_the_greedy_fill_pinned(self):
        """The greedy max-on-match fill is not the optimum, and the gap is
        reachable by substitution: here the fill strands the leading 'ce'
        below the optimum, and replacing it with a synonym unblocks the
        optimal alignment, raising the score to exactly the base pair's
        optimum.  Observed semantics, not a defect: the score stays inside
        the unit interval and never exceeds the optimum."""
        q = ["ba", "ce", "ba", "ce", "ba"]
        c = ["ce", "ba", "ba", "ce", "ba"]
        base = _score_sentence(" ".join(c) + ".", " ".join(q))
        substituted = _score_sentence("zz ba ba ce ba.", " ".join(q))
        assert base == pytest.approx(
            rouge_w_f1_ref(q, c, wlcs=wlcs_max_on_match), abs=1e-9
        ), base
        assert substituted > base, (base, substituted)
        assert substituted == pytest.approx(
            rouge_w_f1_ref(q, c, wlcs=wlcs_bruteforce), abs=1e-9
        ), (substituted, base)

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
            heartbeat_gap_and_wall(lambda: asyncio.to_thread(tors.grounding_coverage, big, big))
        )
        worst, wall = asyncio.run(
            heartbeat_gap_and_wall(
                lambda: asyncio.gather(
                    *(asyncio.to_thread(tors.grounding_coverage, big, big) for _ in range(4))
                )
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

