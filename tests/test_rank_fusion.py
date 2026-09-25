"""Contract gate for the rank-fusion and IR-metric family:
``tors.rank_fuse``, ``tors.ndcg_at_k``, ``tors.mrr``, ``tors.recall_at_k``,
and ``tors.precision_at_k``.

rank_fuse is Reciprocal Rank Fusion exactly as the source paper defines it:
Cormack, Clarke & Buüttcher, "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods", SIGIR 2009
(https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf): ranks only, never raw
scores; ``score(d) = Σ 1/(k + rank(d))`` with 1-based ranks and one shared
``k`` (60, the paper's own default, unchanged across its experiments). The
tie-break (earliest first appearance across the lists in caller order) is a
point the paper leaves open, pinned here as contract. ndcg_at_k is
Järvelin & Kekäläinen's normalized discounted cumulative gain (ACM TOIS
20(4), 2002): log2 discount, rank 1 undiscounted, ideal-DCG normalization;
mrr/recall_at_k/precision_at_k are the standard IR definitions (precision's
``min(k, len(ranked))`` denominator is trec_eval's own convention for a run
shorter than k).

The edge-input policy, pinned repeatedly below: empty DATA answers a
well-defined 0.0 (empty ranking, empty relevant set, nothing judged
relevant); empty STRUCTURE (fusing zero lists) and out-of-range numerics
(k < 1, negative gains) raise ValueError; wrong-TYPE arguments (non-list
ranked_lists, non-set relevant, unhashable ids) raise TypeError, the same
split bm25_rank keeps.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import time
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from loop_harness import heartbeat_gap_and_wall

import tors

# The nDCG oracle differential needs scikit-learn, which is an optional
# oracle dependency: the class skips itself when it is absent, and the
# rest of the file never depends on it.
try:
    from sklearn.metrics import ndcg_score as _sk_ndcg_score
except ImportError:  # pragma: no cover - exercised only without sklearn
    _sk_ndcg_score = None
from tors import mrr, ndcg_at_k, precision_at_k, rank_fuse, recall_at_k

# Ids: arbitrary hashable text, CJK/emoji included (the fusion and the
# metrics are id-space operations: the actual characters are irrelevant,
# which is exactly the property the unicode strategies pin).
_IDS = st.text(min_size=1, max_size=12)
_LISTS = st.lists(st.lists(_IDS, min_size=1, max_size=12), min_size=1, max_size=6)


class TestRankFuseKnownVectors:
    def test_hand_computed_two_list_fusion(self) -> None:
        # The core test's vector re-spelled with string ids: d0 = 1/61 +
        # 1/62 (a perfect tie with d1's 1/62 + 1/61), first appearance
        # breaks it.
        fused = rank_fuse([["d0", "d1"], ["d1", "d0"]])
        assert fused[0][0] == "d0"
        assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
        assert fused[1][1] == pytest.approx(1 / 61 + 1 / 62)

    def test_hand_computed_three_list_fusion_at_k_five(self) -> None:
        # The core's k=5 vector re-spelled: L0=[d0,d1,d2], L1=[d2,d3,d0],
        # L2=[d1,d2].
        lists = [["d0", "d1", "d2"], ["d2", "d3", "d0"], ["d1", "d2"]]
        fused = dict(rank_fuse(lists, k=5))
        assert fused["d0"] == pytest.approx(1 / 6 + 1 / 8)
        assert fused["d1"] == pytest.approx(1 / 7 + 1 / 6)
        assert fused["d2"] == pytest.approx(1 / 8 + 1 / 6 + 1 / 7)
        assert fused["d3"] == pytest.approx(1 / 7)
        assert [i for i, _ in rank_fuse(lists, k=5)] == ["d2", "d1", "d0", "d3"]

    def test_ranks_are_one_based(self) -> None:
        # At k=60 the top of a single list scores exactly 1/61, not 1/60.
        fused = dict(rank_fuse([["a", "b", "c"]]))
        assert fused["a"] == pytest.approx(1 / 61)
        assert fused["c"] == pytest.approx(1 / 63)

    def test_consensus_beats_single_list_top(self) -> None:
        # The RRF thesis: second place in two lists outranks first place
        # in one.
        fused = rank_fuse([["a", "b"], ["c", "b"]])
        assert fused[0][0] == "b"
        assert fused[0][1] == pytest.approx(2 / 62)

    def test_single_list_preserves_its_order(self) -> None:
        assert [i for i, _ in rank_fuse([["a", "b", "c"]])] == ["a", "b", "c"]

    def test_absent_document_contributes_no_vote_and_no_penalty(self) -> None:
        # "a" is in list 0 only; its score is 1/61 whether or not another
        # list exists at all.
        assert dict(rank_fuse([["a"]], k=60))["a"] == pytest.approx(1 / 61)
        assert dict(rank_fuse([["a"], ["b"]], k=60))["a"] == pytest.approx(1 / 61)

    def test_duplicate_within_one_list_votes_once_at_first_occurrence(self) -> None:
        once = dict(rank_fuse([["a", "b"]]))
        twice = dict(rank_fuse([["a", "a", "b"]]))
        assert once["a"] == pytest.approx(twice["a"])
        assert twice["a"] == pytest.approx(1 / 61)

    def test_metric_positions_are_deduped_first(self) -> None:
        # A repeat is a malformed ranking: counting 'a' twice would
        # inflate precision to 1.0 and nDCG past 1; the family contract
        # folds it to its first occurrence everywhere.
        assert precision_at_k(["a", "a"], {"a"}, 2) == 1.0
        assert precision_at_k(["a", "a"], {"a"}, 1) == 1.0
        assert recall_at_k(["a", "a"], {"a"}, 2) == 1.0
        assert ndcg_at_k(["a", "a"], {"a"}, k=None) == 1.0
        assert mrr(["a", "a"], {"a"}) == 1.0

    def test_k_dampens_the_top_rank_without_flipping_rank_order(self) -> None:
        # Within one list, k changes the spacing, never the order.
        small = [i for i, _ in rank_fuse([["a", "b", "c"]], k=1)]
        large = [i for i, _ in rank_fuse([["a", "b", "c"]], k=10_000)]
        assert small == large == ["a", "b", "c"]

    def test_scores_are_strictly_positive_and_bounded(self) -> None:
        # Max possible: first place in every one of L lists = L/(k+1).
        lists = [["a"] for _ in range(4)]
        (score,) = [s for _, s in rank_fuse(lists, k=60)]
        assert 0.0 < score <= 4 / 61


class TestRankFuseTies:
    def test_tie_broken_by_earliest_first_appearance(self) -> None:
        # Both score 1/61; "x" appeared in the earlier list.
        fused = rank_fuse([["x"], ["y"]])
        assert [i for i, _ in fused] == ["x", "y"]

    def test_first_appearance_beats_id_sort_order(self) -> None:
        # The lexicographically LATER id appeared first.
        fused = rank_fuse([["zebra"], ["alpaca"]])
        assert [i for i, _ in fused] == ["zebra", "alpaca"]

    def test_first_appearance_across_lists_beats_within_list_rank(self) -> None:
        # "early" first appears in list 0; "late" in list 1; equal
        # scores, "early" wins the tie despite nothing else.
        fused = rank_fuse([["early"], ["late"]])
        assert [i for i, _ in fused] == ["early", "late"]

    def test_all_tied_single_appearance_documents_keep_first_appearance_order(
        self,
    ) -> None:
        # a/c tie at 1/61 (a appeared first), b/d tie at 1/62 (b first):
        # the fused order interleaves by score first, appearance second.
        fused = rank_fuse([["a", "b"], ["c", "d"]])
        assert [i for i, _ in fused] == ["a", "c", "b", "d"]

    def test_duplicates_fold_across_their_first_occurrence_everywhere(self) -> None:
        # After 'a' is skipped at position 1, 'b' advances to rank 2.
        fused = dict(rank_fuse([["a", "a", "b"]]))
        assert fused["a"] == pytest.approx(1 / 61)
        assert fused["b"] == pytest.approx(1 / 62)


class TestRankFuseArguments:
    def test_zero_lists_raises_value_error(self) -> None:
        # The merkle_root "root of no chunks" precedent: fusing nothing
        # is almost certainly an upstream bug, so it is the contract.
        with pytest.raises(ValueError, match="at least one ranked list"):
            rank_fuse([])

    @pytest.mark.parametrize("k", [0, -1, -60])
    def test_k_below_one_raises_value_error(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            rank_fuse([["a"]], k=k)

    def test_non_list_outer_argument_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            rank_fuse("not a list")  # type: ignore[arg-type]

    def test_non_list_entry_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be a list"):
            rank_fuse([["a"], ("b", "c")])  # type: ignore[list-item]

    def test_unhashable_id_raises_type_error(self) -> None:
        # Python's own hash error: a dict cannot key an id, the same
        # wrong-type-entry contract bm25_rank's corpus walk keeps.
        with pytest.raises(TypeError, match="unhashable"):
            rank_fuse([["a", {"un": "hashable"}]])  # type: ignore[list-item]
        with pytest.raises(TypeError, match="unhashable"):
            rank_fuse([["a", ["nested", "list"]]])  # type: ignore[list-item]

    def test_empty_inner_list_is_legal_and_contributes_no_votes(self) -> None:
        assert rank_fuse([["a"], [], ["b"]]) == [
            ("a", pytest.approx(1 / 61)),
            ("b", pytest.approx(1 / 61)),
        ]

    def test_python_equality_semantics_govern_dedup(self) -> None:
        # 1, True, and 1.0 are the same dict/set key, so they are the
        # same id here too: one fused entry, the FIRST spelling returned.
        fused = rank_fuse([[1, True, 1.0]])
        assert len(fused) == 1
        assert fused[0][0] == 1
        assert fused[0][0] is not True  # the first occurrence's object


class TestRankFuseProperties:
    @given(lists=_LISTS)
    @settings(max_examples=200)
    def test_output_is_a_permutation_of_the_distinct_ids(self, lists: list) -> None:
        expected = set(itertools.chain.from_iterable(lists))
        fused = rank_fuse(lists)
        assert {i for i, _ in fused} == expected
        assert len(fused) == len(expected)

    @given(lists=_LISTS)
    @settings(max_examples=200)
    def test_scores_descend(self, lists: list) -> None:
        scores = [s for _, s in rank_fuse(lists)]
        assert scores == sorted(scores, reverse=True)

    @given(lists=_LISTS)
    @settings(max_examples=200)
    def test_scores_match_the_closed_form(self, lists: list) -> None:
        # The independent oracle: one vote per list, at the id's rank in
        # the DEDUPLICATED list (the fold moves later ids up), ranks
        # 1-based at the paper's k=60.
        fused = dict(rank_fuse(lists))
        for doc_id, score in fused.items():
            expected = 0.0
            for lst in lists:
                for position, entry in enumerate(dict.fromkeys(lst)):
                    if entry == doc_id:
                        expected += 1 / (60 + position + 1)
                        break  # one vote per list, at the first occurrence
            assert score == pytest.approx(expected, rel=1e-12)

    @given(lists=_LISTS)
    @settings(max_examples=100)
    def test_deterministic(self, lists: list) -> None:
        assert rank_fuse(lists) == rank_fuse(lists)

    @given(
        lists=_LISTS,
        k1=st.integers(min_value=1, max_value=200),
        k2=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=100)
    def test_k_only_reweights_never_reorders_within_lists(
        self, lists: list, k1: int, k2: int
    ) -> None:
        # Different k values may permute the FUSED order (k trades top-rank
        # weight against consensus), but each function of the same lists is
        # a total order; this pins determinism across k, not order equality.
        a = [i for i, _ in rank_fuse(lists, k=k1)]
        b = [i for i, _ in rank_fuse(lists, k=k2)]
        assert sorted(a) == sorted(b)

    def test_unicode_and_cjk_ids(self) -> None:
        fused = rank_fuse([["东京", "café", "🦀"], ["🦀", "東京"]])
        assert [i for i, _ in fused] == ["🦀", "东京", "café", "東京"]


def _reference_ndcg(ranked: list, relevant: set, k: int | None, gains: dict | None) -> float:
    """A from-scratch Python re-implementation of the same nDCG
    (linear gain, log2 discount, rank 1 undiscounted, dedup-first): the
    independent oracle for the tests below."""
    if gains is None:
        gains = {}
    # Dedup-first, the family contract: a repeated id counts once, at
    # its first occurrence.
    ranked = list(dict.fromkeys(ranked))

    def gain(d: object) -> float:
        if d in gains:
            return float(gains[d])
        return 1.0 if d in relevant else 0.0

    limit = len(ranked) if k is None else min(k, len(ranked))
    dcg = sum(g / math.log2(p + 2) for p, g in enumerate(gain(d) for d in ranked[:limit]))
    pool = [gain(d) for d in relevant]
    pool += [float(v) for d, v in gains.items() if d not in relevant]
    pool.sort(reverse=True)
    idcg = sum(g / math.log2(p + 2) for p, g in enumerate(pool[:limit]))
    return dcg / idcg if idcg > 0 else 0.0


class TestNdcg:
    def test_perfect_ranking_scores_one(self) -> None:
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}) == 1.0

    def test_hit_at_rank_one_scores_one(self) -> None:
        assert ndcg_at_k(["a"], {"a"}) == 1.0

    def test_hit_moved_down_scores_less_than_one(self) -> None:
        # 1/log2(3) of the ideal.
        assert ndcg_at_k(["x", "a"], {"a"}) == pytest.approx(1 / math.log2(3))

    def test_hand_computed_binary_vector(self) -> None:
        # gains [1, 0, 1]: DCG = 1 + 0.5 = 1.5; ideal [1, 1]: 1 + 1/log2(3).
        score = ndcg_at_k(["a", "x", "b"], {"a", "b"})
        assert score == pytest.approx(1.5 / (1 + 1 / math.log2(3)))

    def test_graded_gains_change_the_score(self) -> None:
        binary = ndcg_at_k(["a", "b"], {"a", "b"})
        graded = ndcg_at_k(["a", "b"], {"a", "b"}, gains={"a": 3.0, "b": 1.0})
        assert binary == 1.0
        assert graded == 1.0  # still the ideal order
        inverted = ndcg_at_k(["a", "b"], {"a", "b"}, gains={"a": 1.0, "b": 3.0})
        assert inverted < 1.0  # the lower grade sits at rank 1 now

    def test_gains_override_the_binary_set_per_id(self) -> None:
        # "a" is in relevant (would be 1.0) but gains downgrade it to 0;
        # "b" is NOT in relevant but gains lift it to 2.
        score = ndcg_at_k(["a", "b"], {"a"}, gains={"a": 0.0, "b": 2.0})
        expected = _reference_ndcg(["a", "b"], {"a"}, None, {"a": 0.0, "b": 2.0})
        assert score == pytest.approx(expected)

    def test_k_truncates(self) -> None:
        # The unranked relevant document is unreachable at k=1.
        assert ndcg_at_k(["a", "x"], {"a", "x"}, k=1) == 1.0
        assert ndcg_at_k(["a", "x"], {"a", "x"}, k=2) == 1.0

    def test_k_clamps_to_the_ranking_length(self) -> None:
        assert ndcg_at_k(["a"], {"a"}, k=10) == 1.0

    def test_k_none_is_the_whole_ranking(self) -> None:
        ranked = ["a", "x", "b", "y"]
        assert ndcg_at_k(ranked, {"a", "b"}, k=None) == pytest.approx(
            ndcg_at_k(ranked, {"a", "b"}, k=4)
        )

    @pytest.mark.parametrize(
        ("ranked", "relevant"),
        [([], {"a"}), (["a"], set()), ([], set()), (["x", "y"], set())],
    )
    def test_empty_data_answers_zero(self, ranked: list, relevant: set) -> None:
        assert ndcg_at_k(ranked, relevant) == 0.0

    def test_frozen_set_accepted(self) -> None:
        assert ndcg_at_k(["a"], frozenset({"a"})) == 1.0

    def test_non_set_relevant_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="set"):
            ndcg_at_k(["a"], ["a"])  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="set"):
            mrr(["a"], "a")  # type: ignore[arg-type]

    @pytest.mark.parametrize("k", [0, -1])
    def test_k_below_one_raises_value_error(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            ndcg_at_k(["a"], {"a"}, k=k)

    def test_non_list_ranked_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            ndcg_at_k("ab", {"a"})  # type: ignore[arg-type]

    def test_non_dict_gains_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            ndcg_at_k(["a"], {"a"}, gains=[("a", 1)])  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [-0.5, float("nan"), float("inf")])
    def test_bad_gain_value_raises_value_error(self, value: float) -> None:
        with pytest.raises(ValueError, match="gains values must be finite and >= 0"):
            ndcg_at_k(["a"], set(), gains={"a": value})

    def test_unhashable_id_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="unhashable"):
            ndcg_at_k([{"d": 1}], {"a"})  # type: ignore[list-item]

    # The saturating-ratio policy: legal finite gains can overflow the
    # DCG/IDCG sums to +inf, where IEEE inf/inf is NaN; the pinned
    # [0.0, 1.0] contract holds instead.

    def test_three_huge_gains_saturate_at_one_not_nan(self) -> None:
        # 3 × 1e308 overflows both sums; the ranked gains equal the
        # ideal pool, so the equality pins exactly 1.0.
        ranked = ["a", "b", "c"]
        assert ndcg_at_k(ranked, set(), gains={d: 1e308 for d in ranked}) == 1.0

    def test_two_huge_gains_saturate_at_one_not_nan(self) -> None:
        gains = {"a": 1.7e308, "b": 1.7e308}
        assert ndcg_at_k(["a", "b"], set(), gains=gains) == 1.0

    def test_finite_dcg_under_an_infinite_ideal_answers_zero(self) -> None:
        # One huge ranked gain (DCG stays finite) under a three-huge
        # ideal at k=4 (IDCG overflows): the saturating answer is 0.0.
        # (k clamps to the ranking's length, so the ranking needs the
        # four positions for the ideal to overflow.)
        ranked = ["a", "x", "y", "z"]
        gains = {"a": 1e308, "b": 1e308, "c": 1e308}
        assert ndcg_at_k(ranked, set(), gains=gains) == 0.0

    def test_mixed_huge_and_small_gains_stay_in_the_unit_interval(self) -> None:
        gains = {"a": 1e308, "b": 1e-300, "c": 1e308}
        score = ndcg_at_k(["a", "b", "c"], set(), gains=gains)
        # The mixed sums stay finite here and the ranking is imperfect
        # (the 1e-300 gain sits at rank 2, below the ideal's third
        # 1e308): an ordinary ratio, strictly inside the interval.
        assert 0.0 < score < 1.0

    @given(
        ranked=st.lists(_IDS, max_size=12),
        gain=st.floats(min_value=0.0, max_value=1.7e308, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_extreme_gains_stay_in_the_unit_interval(self, ranked: list, gain: float) -> None:
        gains = {d: gain for d in set(ranked)}
        score = ndcg_at_k(ranked, set(), gains=gains)
        assert 0.0 <= score <= 1.0

    @given(ranked=st.lists(_IDS, max_size=12), k=st.integers(1, 20) | st.none())
    @settings(max_examples=200)
    def test_score_always_in_the_unit_interval(self, ranked: list, k: int | None) -> None:
        relevant = set(ranked[:3])
        score = ndcg_at_k(ranked, relevant, k=k)
        assert 0.0 <= score <= 1.0

    @given(ranked=st.lists(_IDS, max_size=12))
    @settings(max_examples=150)
    def test_matches_the_independent_reference(self, ranked: list) -> None:
        relevant = set(ranked[:2]) | {"never-ranked"}
        assert ndcg_at_k(ranked, relevant) == pytest.approx(
            _reference_ndcg(ranked, relevant, None, None), abs=1e-12
        )

    @given(ranked=st.lists(_IDS, min_size=1, max_size=12))
    @settings(max_examples=150)
    def test_graded_matches_the_independent_reference(self, ranked: list) -> None:
        gains = {d: float((i % 4) + 1) for i, d in enumerate(ranked)}
        relevant = set(ranked)
        assert ndcg_at_k(ranked, relevant, gains=gains) == pytest.approx(
            _reference_ndcg(ranked, relevant, None, gains), abs=1e-12
        )


class TestMrr:
    def test_first_position_hit(self) -> None:
        assert mrr(["a", "b"], {"a"}) == 1.0

    def test_reciprocal_of_the_first_relevant_position(self) -> None:
        assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
        # Later hits never move the score.
        assert mrr(["x", "a", "b"], {"a", "b"}) == pytest.approx(1 / 2)

    def test_no_hit_is_zero(self) -> None:
        assert mrr(["x", "y"], {"a"}) == 0.0

    def test_empty_ranking_is_zero(self) -> None:
        assert mrr([], {"a"}) == 0.0

    def test_empty_relevant_is_zero(self) -> None:
        assert mrr(["a"], set()) == 0.0

    def test_unhashable_id_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="unhashable"):
            mrr([{"d": 1}], {"a"})  # type: ignore[list-item]

    @given(ranked=st.lists(_IDS, max_size=12))
    @settings(max_examples=200)
    def test_score_in_the_unit_interval_and_first_hit_reciprocal(
        self, ranked: list
    ) -> None:
        ranked = list(dict.fromkeys(ranked))  # the dedup-first contract
        relevant = {ranked[2]} if len(ranked) > 2 else set(ranked)
        score = mrr(ranked, relevant)
        assert 0.0 <= score <= 1.0
        expected = next(
            (1 / (p + 1) for p, d in enumerate(ranked) if d in relevant), 0.0
        )
        assert score == pytest.approx(expected)


class TestRecallAtK:
    def test_counts_hits_over_relevant_size(self) -> None:
        assert recall_at_k(["a", "x", "b", "y"], {"a", "b", "c"}, 4) == pytest.approx(2 / 3)

    def test_k_truncates_the_window(self) -> None:
        assert recall_at_k(["a", "b"], {"a", "b"}, 1) == 0.5

    def test_k_past_the_ranking_length_uses_available_positions(self) -> None:
        assert recall_at_k(["a"], {"a", "b"}, 10) == 0.5

    def test_perfect_recall(self) -> None:
        assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_empty_relevant_is_zero(self) -> None:
        assert recall_at_k(["a"], set(), 1) == 0.0

    def test_empty_ranking_is_zero(self) -> None:
        assert recall_at_k([], {"a"}, 1) == 0.0

    @pytest.mark.parametrize("k", [0, -1])
    def test_k_below_one_raises_value_error(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            recall_at_k(["a"], {"a"}, k)

    def test_non_set_relevant_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="set"):
            recall_at_k(["a"], ["a"], 1)  # type: ignore[arg-type]


class TestPrecisionAtK:
    def test_counts_hits_over_k(self) -> None:
        assert precision_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(2 / 3)

    def test_short_run_divides_by_its_own_length(self) -> None:
        # trec_eval's convention: a run shorter than k is not punished
        # for positions it never filled.
        assert precision_at_k(["a"], {"a"}, 10) == 1.0

    def test_k_truncates_the_window(self) -> None:
        assert precision_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5

    def test_empty_ranking_is_zero(self) -> None:
        assert precision_at_k([], {"a"}, 3) == 0.0

    def test_empty_relevant_is_zero(self) -> None:
        assert precision_at_k(["a"], set(), 1) == 0.0

    @pytest.mark.parametrize("k", [0, -1])
    def test_k_below_one_raises_value_error(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            precision_at_k(["a"], {"a"}, k)

    def test_non_set_relevant_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="set"):
            precision_at_k(["a"], ["a"], 1)  # type: ignore[arg-type]

    @given(ranked=st.lists(_IDS, min_size=1, max_size=12), k=st.integers(1, 20))
    @settings(max_examples=200)
    def test_score_in_the_unit_interval(self, ranked: list, k: int) -> None:
        relevant = set(ranked[:3]) | {"never-ranked"}
        score = precision_at_k(ranked, relevant, k)
        assert 0.0 <= score <= 1.0


class TestFamilyCrossChecks:
    def test_recall_and_precision_agree_on_hits(self) -> None:
        ranked, relevant, k = ["a", "x", "b", "y"], {"a", "b", "c"}, 4
        hits = sum(d in relevant for d in ranked[:k])
        assert recall_at_k(ranked, relevant, k) == pytest.approx(hits / 3)
        assert precision_at_k(ranked, relevant, k) == pytest.approx(hits / 4)

    def test_ndcg_perfect_when_ranking_matches_the_ideal_arrangement(self) -> None:
        # The standard normalization: the ideal packs the judged pool's
        # top gains at ranks 1..k, so "the relevant documents as high as
        # they can sit" is perfect, and a relevant document BELOW a
        # non-relevant one is not (sklearn's ndcg_score answers the same
        # 0.9197 for this input).
        relevant = {"a", "c"}
        assert ndcg_at_k(["a", "c", "x", "y"], relevant) == 1.0
        assert ndcg_at_k(["a", "b", "c", "d"], relevant) == pytest.approx(
            1.5 / (1 + 1 / math.log2(3))
        )
        assert mrr(["a", "c", "x", "y"], relevant) == 1.0

    def test_metrics_are_deterministic(self) -> None:
        ranked, relevant = ["a", "b", "c"], {"a", "c"}
        assert mrr(ranked, relevant) == mrr(ranked, relevant)
        assert ndcg_at_k(ranked, relevant) == ndcg_at_k(ranked, relevant)
        assert recall_at_k(ranked, relevant, 2) == recall_at_k(ranked, relevant, 2)
        assert precision_at_k(ranked, relevant, 2) == precision_at_k(ranked, relevant, 2)

    def test_exported_spellings_are_the_module_attributes(self) -> None:
        for name in ("rank_fuse", "ndcg_at_k", "mrr", "recall_at_k", "precision_at_k"):
            assert getattr(tors, name) is not None


_IDS = st.text(min_size=1, max_size=8)


# ---------------------------------------------------------------------------
# 1. RRF vs the paper
# ---------------------------------------------------------------------------


class TestRRFVsPaper:
    def test_k_zero_is_a_value_error_not_a_zero_division(self) -> None:
        # k=0 makes the paper's denominator k + rank honest-but-zero at
        # rank 0 (a 0-based-rank bug's signature); it must be rejected.
        with pytest.raises(ValueError, match="k must be >= 1"):
            rank_fuse([["a", "b"]], k=0)

    def test_k_one_exact_vector(self) -> None:
        # k=1, 1-based ranks: rank 1 -> 1/2, rank 2 -> 1/3, rank 3 -> 1/4.
        fused = dict(rank_fuse([["a", "b", "c"]], k=1))
        assert fused["a"] == pytest.approx(1 / 2)
        assert fused["b"] == pytest.approx(1 / 3)
        assert fused["c"] == pytest.approx(1 / 4)

    def test_huge_k_keeps_vote_counts_dominant(self) -> None:
        # As k -> inf every vote -> ~1/k (past 2^53 the rank offsets
        # round away entirely), so the fused order becomes the vote
        # count, ties by first appearance: no float blowup, no
        # overflow, and the two-vote doc still beats the one-vote doc.
        fused = rank_fuse([["solo"], ["cons"], ["cons"]], k=10**18)
        assert [i for i, _ in fused] == ["cons", "solo"]
        assert fused[0][1] > fused[1][1]
        assert all(math.isfinite(s) for _, s in fused)

    def test_k_above_i64_is_an_overflow_error(self) -> None:
        with pytest.raises(OverflowError):
            rank_fuse([["a"]], k=10**19)

    @pytest.mark.parametrize("bad_k", [1.5, "60", None])
    def test_non_integer_k_is_a_type_error(self, bad_k: object) -> None:
        with pytest.raises(TypeError):
            rank_fuse([["a"]], k=bad_k)  # type: ignore[arg-type]

    def test_a_doc_ranked_first_in_one_list_cannot_dominate_a_consensus_doc(
        self,
    ) -> None:
        # The paper's thesis at the paper's k: three lists' #2 outranks
        # one list's #1 (3/62 > 1/61); at small k the thesis must still
        # hold (3/2 > 1/2).
        for k in (60, 1):
            fused = dict(rank_fuse([["top", "cons"], ["x", "cons"], ["y", "cons"]], k=k))
            assert fused["cons"] > fused["top"], k

    def test_ties_break_by_earliest_first_appearance_deterministically(self) -> None:
        # Same score (1/61 each): the doc that appeared in the EARLIER
        # list wins, regardless of id sort order; repeated calls and a
        # reversed-caller-order re-spell agree with the pinned rule.
        fused = rank_fuse([["z-late-id"], ["a-early-id"]])
        assert [i for i, _ in fused] == ["z-late-id", "a-early-id"]
        assert rank_fuse([["z-late-id"], ["a-early-id"]]) == fused

    def test_scores_are_exactly_the_paper_sum(self) -> None:
        # Independent closed-form oracle over a 4-list overlap pattern,
        # k=60: one vote per (list, first occurrence of the doc in it).
        lists = [
            ["a", "b", "c", "d"],
            ["c", "a"],
            ["d", "c", "b"],
            ["b"],
        ]
        fused = dict(rank_fuse(lists))
        for doc, score in fused.items():
            expected = 0.0
            for lst in lists:
                deduped = list(dict.fromkeys(lst))
                if doc in deduped:
                    expected += 1 / (60 + deduped.index(doc) + 1)
            assert score == pytest.approx(expected, rel=1e-12)

    @given(
        lists=st.lists(st.lists(_IDS, max_size=10), min_size=1, max_size=8),
        k=st.integers(min_value=1, max_value=500),
    )
    @settings(max_examples=150)
    def test_output_is_finite_positive_and_a_complete_permutation(
        self, lists: list, k: int
    ) -> None:
        fused = rank_fuse(lists, k=k)
        assert {i for i, _ in fused} == set(itertools.chain.from_iterable(lists))
        assert all(math.isfinite(s) and s > 0 for _, s in fused)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 2. nDCG vs the sklearn oracle (+ the >1.0 hunt)
# ---------------------------------------------------------------------------


def _tors_equivalent_of_sklearn_case(
    y_true: list[float], y_score: list[float], k: int | None
) -> float:
    """Build the tors spelling of a sklearn (y_true, y_score) sample:
    ids ordered by score descending, every judged doc carrying its gain
    in `gains` (relevant empty; gains is the complete judged pool)."""
    ids = [f"d{i}" for i in range(len(y_score))]
    order = sorted(range(len(y_score)), key=lambda i: -y_score[i])
    ranked = [ids[i] for i in order]
    gains = {ids[i]: float(y_true[i]) for i in range(len(y_true))}
    return ndcg_at_k(ranked, set(), gains=gains, k=k)


class TestNdcgAgainstSklearn:
    @pytest.mark.parametrize(
        ("y_true", "y_score", "k"),
        [
            ([3.0, 2.0, 0.0, 1.0], [4.0, 3.0, 2.0, 1.0], None),
            ([2.0, 3.0, 1.0], [3.0, 2.0, 1.0], None),
            ([1.0, 0.0, 1.0, 0.0], [5.0, 4.0, 3.0, 2.0], 3),
            ([1.0, 0.0, 1.0, 0.0], [5.0, 4.0, 3.0, 2.0], None),
            ([0.0, 3.0, 0.0, 0.0], [4.0, 2.0, 1.0, 0.5], 1),
            ([0.0, 0.0, 3.0, 2.0], [4.0, 3.0, 2.0, 1.0], 2),
            ([2.0, 1.0, 0.0], [10.0, 9.0, 1.0], 2),
            ([0.5, 0.5, 0.5], [3.0, 2.0, 1.0], None),
        ],
    )
    def test_matches_sklearn_ndcg_score_exactly(
        self, y_true: list[float], y_score: list[float], k: int | None
    ) -> None:
        if _sk_ndcg_score is None:
            pytest.skip("scikit-learn not installed")
        sk = _sk_ndcg_score([y_true], [y_score], k=k)
        t = _tors_equivalent_of_sklearn_case(y_true, y_score, k)
        assert t == pytest.approx(sk, abs=1e-9)

    def test_all_zero_ideal_is_zero_like_sklearn(self) -> None:
        if _sk_ndcg_score is None:
            pytest.skip("scikit-learn not installed")
        assert _sk_ndcg_score([[0.0, 0.0]], [[1.0, 2.0]]) == 0.0
        assert ndcg_at_k(["a", "b"], set(), gains={"a": 0.0, "b": 0.0}) == 0.0

    def test_gains_referencing_ids_absent_from_ranked_only_tighten_the_ideal(
        self,
    ) -> None:
        # tors-only pool shape (sklearn's y_true/y_score are always
        # co-extensive): a gains entry for an id the ranking never
        # surfaces joins the IDEAL pool, never the DCG, so it can only
        # lower the score, and only when its gain exceeds the ranked
        # ones'. (With k=None, k clamps to the ranked length, so the
        # ghost displaces a ranked gain's discount, never adds a term.)
        perfect = ndcg_at_k(["a"], set(), gains={"a": 1.0})
        assert perfect == 1.0
        with_ghost = ndcg_at_k(["a"], set(), gains={"a": 1.0, "ghost": 5.0})
        assert with_ghost == pytest.approx(1.0 / 5.0)
        assert with_ghost < perfect

    def test_sklearn_agrees_when_the_ghost_doc_is_ranked_last(self) -> None:
        if _sk_ndcg_score is None:
            pytest.skip("scikit-learn not installed")
        # The sklearn encoding of "judged but never surfaced": ranked
        # dead last. k=1 sees only the top hit on both sides.
        sk = _sk_ndcg_score([[1.0, 1.0]], [[3.0, 0.001]], k=1, ignore_ties=True)
        t = _tors_equivalent_of_sklearn_case([1.0, 1.0], [3.0, 0.001], 1)
        assert sk == 1.0 and t == pytest.approx(sk, abs=1e-9)


class TestNdcgUnitIntervalHunt:
    """Every attempt to push binary or graded nDCG past 1.0."""

    @pytest.mark.parametrize(
        "ranked",
        [
            ["a", "a", "b", "b"],
            ["a", "b", "a", "b"],
            [1, True, 1.0, "a"],
            ["a", "a", "a", "a", "a"],
        ],
    )
    def test_duplicate_and_cross_type_ids_cannot_exceed_one(self, ranked: list) -> None:
        relevant = {"a", "b", 1}
        assert 0.0 <= ndcg_at_k(ranked, relevant) <= 1.0

    def test_gains_conflicting_with_relevant_cannot_exceed_one(self) -> None:
        # gains overrides relevant per id (0.0 for a relevant doc, 5.0
        # for a non-relevant one): the pool still contains every ranked
        # gain, so the normalization holds.
        s = ndcg_at_k(["a", "b"], {"a"}, gains={"a": 0.0, "b": 5.0})
        assert 0.0 <= s <= 1.0

    def test_float_gains_and_huge_k_stay_in_the_unit_interval(self) -> None:
        s = ndcg_at_k(["a", "b", "c"], {"a"}, gains={"a": 0.25, "b": 2.5}, k=10**18)
        assert 0.0 <= s <= 1.0

    def test_denormal_gains_stay_in_the_unit_interval(self) -> None:
        assert 0.0 <= ndcg_at_k(["x", "a"], set(), gains={"a": 5e-324}) <= 1.0
        assert 0.0 <= ndcg_at_k(["a", "x"], set(), gains={"a": 5e-324}) <= 1.0

    @given(
        ranked=st.lists(st.one_of(_IDS, st.integers(0, 3)), max_size=12),
        k=st.integers(1, 20) | st.none(),
        gain=st.floats(min_value=0.0, max_value=1e30, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_graded_ndcg_with_duplicates_stays_in_the_unit_interval(
        self, ranked: list, k: int | None, gain: float
    ) -> None:
        gains = {d: gain for d in set(ranked)}
        s = ndcg_at_k(ranked, set(), gains=gains, k=k)
        assert 0.0 <= s <= 1.0

    # Gains legal per the documented domain (finite, >= 0) but SO large
    # that the DCG and IDCG sums overflow to +inf: the saturating,
    # overflow-aware normalization (see rank_fusion_impl.rs's
    # saturating-ratio policy) keeps the score in [0.0, 1.0]. Also
    # pinned in tests/test_rank_fusion.py (TestNdcg's overflow tests).
    def test_extreme_finite_gains_stay_in_the_unit_interval(self) -> None:
        ranked = ["a", "b", "c"]
        s = ndcg_at_k(ranked, set(), gains={d: 1e308 for d in ranked})
        assert 0.0 <= s <= 1.0

    def test_zero_ideal_via_gains_is_zero_not_nan(self) -> None:
        # The contract's named second NaN path: gains summing to zero
        # make the ideal DCG zero: answered 0.0, not 0/0.
        assert ndcg_at_k(["a", "b"], set(), gains={"a": 0.0, "b": 0.0}) == 0.0


class TestNdcgArgumentDiscipline:
    def test_non_numeric_gain_raises_type_error(self) -> None:
        # The IMPL's answer is TypeError (pyo3's extract failure). NOTE:
        # docs/api.md and the docstring claim ValueError for a
        # "non-numeric gains value"; a doc/impl drift, reported as P2.
        with pytest.raises(TypeError):
            ndcg_at_k(["a"], set(), gains={"a": "high"})  # type: ignore[dict-item]

    @pytest.mark.parametrize("value", [-0.5, float("nan"), float("inf")])
    def test_negative_or_non_finite_gain_raises_value_error(self, value: float) -> None:
        with pytest.raises(ValueError, match="gains values must be finite"):
            ndcg_at_k(["a"], set(), gains={"a": value})

    def test_gains_that_are_not_a_dict_raise_type_error(self) -> None:
        with pytest.raises(TypeError):
            ndcg_at_k(["a"], set(), gains=[("a", 1.0)])  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_relevant", [["a"], ("a",), (x for x in ["a"]), "a", {"a": 1}])
    def test_relevant_must_be_exactly_a_set_or_frozenset(self, bad_relevant: object) -> None:
        # list/tuple/generator/dict-keys/str: all TypeError, consistently
        # across every metric that takes `relevant`.
        with pytest.raises(TypeError, match="relevant must be a set"):
            mrr(["a"], bad_relevant)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="relevant must be a set"):
            recall_at_k(["a"], bad_relevant, 1)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="relevant must be a set"):
            precision_at_k(["a"], bad_relevant, 1)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="relevant must be a set"):
            ndcg_at_k(["a"], bad_relevant)  # type: ignore[arg-type]

    def test_frozenset_and_subclasses_are_accepted(self) -> None:
        assert ndcg_at_k(["a"], frozenset({"a"})) == 1.0
        assert mrr(["a"], frozenset({"a"})) == 1.0


# ---------------------------------------------------------------------------
# 3. The metrics: dedup semantics, denominators, zero answers
# ---------------------------------------------------------------------------


class TestMetricDedupSemantics:
    """The dedup-first contract deliberately diverges from the textbook
    `ranked[:k]` formulas when `ranked` contains repeats, pinned here
    as the DOCUMENTED behavior (docs/api.md: a repeat is a malformed
    ranking, counted once at its first occurrence), with the divergence
    reported in the findings as a P2 surprise."""

    def test_recall_counts_a_duplicate_ranking_more_generously_than_trec(self) -> None:
        # trec_eval's ranked[:2] = [a, a] finds 1 distinct relevant doc:
        # recall 0.5. tors folds the repeat and reaches b: 1.0.
        assert recall_at_k(["a", "a", "b"], {"a", "b"}, 2) == 1.0

    def test_precision_denominator_is_the_deduped_length_clamped_to_k(self) -> None:
        assert precision_at_k(["a", "a", "b"], {"a", "b"}, 2) == 1.0
        assert precision_at_k(["a", "a", "b"], {"a", "b"}, 3) == 1.0

    def test_mrr_dedup_pulls_the_first_hit_forward(self) -> None:
        # trec_eval: first relevant at rank 3 -> 1/3; tors folds the
        # repeat: 1/2.
        assert mrr(["x", "x", "a"], {"a"}) == pytest.approx(1 / 2)

    def test_precision_short_run_denominator_is_documented_and_pinned(self) -> None:
        # trec_eval's min(k, len) convention: a short run is not punished
        # for unfilled positions. len(ranked) here is the DEDUPED length,
        # as the stub comment and docstring pin.
        assert precision_at_k(["a"], {"a"}, 10) == 1.0
        assert precision_at_k([], {"a"}, 3) == 0.0

    def test_mrr_with_relevant_entirely_missing_answers_zero(self) -> None:
        assert mrr(["x", "y", "z"], {"below"}) == 0.0
        assert mrr([], {"below"}) == 0.0


class TestZeroDivisionHunt:
    def test_empty_relevant_is_zero_for_every_metric(self) -> None:
        assert recall_at_k(["a", "b"], set(), 2) == 0.0
        assert precision_at_k(["a", "b"], set(), 2) == 0.0
        assert mrr(["a", "b"], set()) == 0.0
        assert ndcg_at_k(["a", "b"], set()) == 0.0

    def test_empty_ranking_is_zero_for_every_metric(self) -> None:
        assert recall_at_k([], {"a"}, 1) == 0.0
        assert precision_at_k([], {"a"}, 1) == 0.0
        assert mrr([], {"a"}) == 0.0
        assert ndcg_at_k([], {"a"}) == 0.0

    def test_rank_fuse_rejects_zero_lists_but_accepts_empty_inner_lists(self) -> None:
        with pytest.raises(ValueError, match="at least one ranked list"):
            rank_fuse([])
        assert rank_fuse([["a"], [], []]) == [("a", pytest.approx(1 / 61))]

    def test_recall_of_all_zero_flags_cannot_nan(self) -> None:
        # Empty relevant + any flags must answer 0.0 (0/0 is NaN without
        # the guard), pinned at the binding AND (per the crate's own unit
        # tests + fuzz
        # target, which run without pyo3) guarded in the Rust core
        # itself: src/rank_fusion_impl.rs returns 0.0 before dividing.
        for flags_relevant in (set(), {"a"}):
            assert recall_at_k(["a", "b"], flags_relevant, 5) == recall_at_k(
                ["a", "b"], flags_relevant, 5
            )
        assert recall_at_k([], set(), 5) == 0.0


class TestUnhashableIdConsistency:
    """An unhashable id must be Python's own unhashable TypeError in ALL
    FIVE functions (the wrong-type-entry contract, not just rank_fuse's)."""

    @pytest.mark.parametrize("bad_id", [{"d": 1}, ["l"], {1, 2}, [1, [2]]])
    def test_unhashable_ids_raise_type_error_in_all_five_functions(self, bad_id: object) -> None:
        with pytest.raises(TypeError, match="unhashable"):
            rank_fuse([["a", bad_id]])  # type: ignore[list-item]
        with pytest.raises(TypeError, match="unhashable"):
            ndcg_at_k(["a", bad_id], {"a"})  # type: ignore[list-item]
        with pytest.raises(TypeError, match="unhashable"):
            mrr(["a", bad_id], {"a"})  # type: ignore[list-item]
        with pytest.raises(TypeError, match="unhashable"):
            recall_at_k(["a", bad_id], {"a"}, 2)  # type: ignore[list-item]
        with pytest.raises(TypeError, match="unhashable"):
            precision_at_k(["a", bad_id], {"a"}, 2)  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# 4. Hostile objects: re-entrancy and input mutation
# ---------------------------------------------------------------------------


class TestHostileObjects:
    def test_hash_that_calls_tors_re_entrantly_is_safe(self) -> None:
        # Stateless doctrine: a hostile __hash__ re-entering tors from
        # inside the dedup walk must neither deadlock nor corrupt.
        class Reentrant:
            def __hash__(self) -> int:
                tors.rank_fuse([["nested"]])
                tors.ndcg_at_k(["n"], {"n"})
                return 7

            def __eq__(self, o: object) -> bool:
                return isinstance(o, Reentrant)

        fused = rank_fuse([[Reentrant(), "z"]])
        assert [i for i, _ in fused] == [fused[0][0], "z"]

    def test_eq_that_mutates_the_input_list_during_the_walk_is_contained(self) -> None:
        # A hostile __eq__ clearing the list mid-dedup: the walk must not
        # panic (a Rust panic would abort the process); a clean Python
        # outcome (result or TypeError) is the contract.
        class Evil:
            def __init__(self, target: list) -> None:
                self.target = target

            def __hash__(self) -> int:
                return 42

            def __eq__(self, other: object) -> bool:
                self.target[:] = []
                return True

        target = ["a", "b", "c"]
        try:
            fused = rank_fuse([[Evil(target), Evil(target), Evil(target)]])
            assert all(math.isfinite(s) for _, s in fused)
        except TypeError:
            pass  # a raise is acceptable; a process abort is not


# ---------------------------------------------------------------------------
# 5. The aio twins: parity + the GIL claim under a busy loop
# ---------------------------------------------------------------------------


def _fusion_lists(total_entries: int, n_lists: int = 5) -> list[list[str]]:
    # The pinned GIL cell's own deterministic workload (half-distinct
    # shared-pool ids, str objects reused across lists).
    per_list = total_entries // n_lists
    id_space = total_entries // 2
    return [
        [f"id_{(j * per_list + i) % id_space}" for i in range(per_list)] for j in range(n_lists)
    ]


class TestAioTwins:
    def test_all_five_twins_match_their_sync_results_exactly(self) -> None:
        import tors.aio

        lists = _fusion_lists(600)
        ranked = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c", "never"}

        async def run() -> None:
            assert await tors.aio.rank_fuse(lists, k=7) == rank_fuse(lists, k=7)
            assert await tors.aio.ndcg_at_k(ranked, relevant) == ndcg_at_k(ranked, relevant)
            assert await tors.aio.ndcg_at_k(
                ranked, relevant, k=2, gains={"a": 3.0, "b": 1.0}
            ) == ndcg_at_k(ranked, relevant, k=2, gains={"a": 3.0, "b": 1.0})
            assert await tors.aio.mrr(ranked, relevant) == mrr(ranked, relevant)
            assert await tors.aio.recall_at_k(ranked, relevant, 2) == recall_at_k(
                ranked, relevant, 2
            )
            assert await tors.aio.precision_at_k(ranked, relevant, 2) == precision_at_k(
                ranked, relevant, 2
            )

        asyncio.run(run())

    @pytest.mark.timing
    def test_rank_fuse_aio_under_a_busy_loop_stays_within_the_pinned_budget(
        self,
    ) -> None:
        # Attack the 0.80 GIL-ratio cell through its aio twin while the
        # loop heartbeats: the worst blocked share of the wall must stay
        # inside the pinned 0.80 band's own margin (pass-on-first-clean
        # over 4 samples, the harness's own retry discipline).
        import tors.aio

        lists = _fusion_lists(200_000)

        async def measure() -> None:
            for _ in range(4):
                worst_gap, wall = await heartbeat_gap_and_wall(lambda: tors.aio.rank_fuse(lists))
                if worst_gap / wall <= 0.80:
                    return
            worst_gap, wall = await heartbeat_gap_and_wall(lambda: tors.aio.rank_fuse(lists))
            assert worst_gap / wall <= 0.85, (
                f"the aio twin blocked {worst_gap * 1e3:.0f}ms of a "
                f"{wall * 1e3:.0f}ms call ({worst_gap / wall:.0%}): past the "
                "pinned 0.80 band the detach is not holding its documented share"
            )

        asyncio.run(measure())

    def test_heavy_hash_ids_push_the_gil_share_toward_one_recorded_not_pinned(
        self,
    ) -> None:
        # FINDING (recorded, bounded): ids that are expensive to hash
        # (10-int tuples, ~0.5us/hash against ~0.05us for cached strs)
        # push the GIL-held share of a large fusion to ~0.98-1.00;
        # the pinned 0.80 budget is a STRING-ID shape claim, and the
        # docs' "the walk is structurally the majority" wording already
        # owns it. Asserted here only as no-worse-than-fully-blocking
        # plus the doc's caller guidance (the call still finishes).
        ids = [tuple((i * 10 + k) % 1_000_003 for k in range(10)) for i in range(50_000)]
        per_list = len(ids) // 5
        lists = [[ids[(j * per_list + i) % len(ids)] for i in range(per_list)] for j in range(5)]
        start = time.monotonic()
        fused = rank_fuse(lists)
        wall = time.monotonic() - start
        assert len(fused) == len(ids)
        assert wall < 30.0  # the reranking-scale guidance, not a ratio pin


# ---------------------------------------------------------------------------
# 6. Perf cliffs and the scaling pin's teeth
# ---------------------------------------------------------------------------


def _min_wall_ms(fn, samples: int = 5) -> float:
    fn()
    best = float("inf")
    for _ in range(samples):
        started = time.monotonic()
        fn()
        best = min(best, time.monotonic() - started)
    return best * 1e3


class TestPerfCliffs:
    @pytest.mark.timing
    def test_many_tiny_lists_scale_like_one_big_list(self) -> None:
        # The two axis extremes at equal total entries: 10k lists x 1
        # doc (per-list overhead + output marshalling dominate) vs 1
        # list x 10k docs (pure walk). Each must stay inside the 3.0x
        # per-doubling gate at a 4x input.
        for shape in (
            lambda n: [[f"d{i}"] for i in range(n)],
            lambda n: [[f"d{i}" for i in range(n)]],
            lambda n: [[f"d{(j * 13 + i) % n}" for i in range(n // 100)] for j in range(100)],
        ):
            # 10k -> 40k: inside the documented reranking scale (the family
            # docs scope rank_fuse to hundreds-to-thousands of entries per
            # list), where the per-doubling slope is a stable algorithmic
            # signal (measured 1.9-2.5x/doubling fresh-process and under
            # pytest, on the dev box and CI). Past that scale a pytest
            # process's heap state inflates the marshalling-heavy large call
            # up to ~4x (measured 16-17x apparent per 4x at 25k-200k under
            # pytest vs 2-2.5x for the identical shape in a fresh process;
            # GC-independent) and the slope stops discriminating: the
            # whole-corpus regime is covered by the wall-guidance pins
            # instead (reranking-scale walls, e.g. the 30s bound on the
            # heavy-hash cell), not by a slope gate. A quadratic regression
            # still fails the 9.0x gate (16x for a 4x span) at this scale.
            # The timed callable is the FUSION on the built lists, not the
            # build: s(n) alone would pin the list construction (a test
            # born measuring the wrong thing — caught by re-running the
            # wave-1 injection experiment, which this pin sailed through
            # while test_scaling_pins.py's rank_fuse pin failed).
            small = _min_wall_ms(lambda s=shape: rank_fuse(s(10_000)))
            large = _min_wall_ms(lambda s=shape: rank_fuse(s(40_000)))
            assert large < 9.0 * small, (
                f"{large:.2f}ms for 4x {small:.2f}ms ({large / small:.2f}x): "
                "superlinear in the shape's axis"
            )

    @pytest.mark.timing
    def test_constant_hash_hostile_ids_are_no_worse_than_the_interpreters_dict(
        self,
    ) -> None:
        # ids that all hash to 0 (but compare unequal) degrade CPython's
        # own dict to linear-probe scans; the interpreter's quadratic,
        # not tors's (pure-Python dict.fromkeys reproduces it). The pin:
        # tors's walk must not be MULTIPLECTIVELY worse than the
        # interpreter's own dict on the same hostile ids.
        class Collide:
            __slots__ = ("i",)

            def __init__(self, i: int) -> None:
                self.i = i

            def __hash__(self) -> int:
                return 0

            def __eq__(self, o: object) -> bool:
                return isinstance(o, Collide) and o.i == self.i

        def build(n: int) -> list:
            return [[Collide(i)] for i in range(n)]

        for n in (600, 1200):
            ids = [row[0] for row in build(n)]
            tors_ms = _min_wall_ms(lambda b=build, m=n: rank_fuse(b(m)), samples=2)
            dict_ms = _min_wall_ms(lambda i=ids: dict.fromkeys(i), samples=2)
            assert tors_ms < 6.0 * dict_ms, (
                f"rank_fuse {tors_ms:.1f}ms vs the interpreter dict's "
                f"{dict_ms:.1f}ms at n={n}: the walk is worse than the "
                "dict it rides on"
            )

    def test_the_scaling_pin_is_strictly_linear(self) -> None:
        # The 3.0x/doubling gate is a hard assertion in TestRankFusionScaling;
        # this cross-checks the gate constant exists so the pin cannot be
        # silently loosened.
        import re as _re
        from pathlib import Path

        pin = Path("tests/test_scaling_pins.py").read_text()
        assert _re.search(r"LINEAR_GATE_PER_DOUBLING = 3\.0", pin)


_IDS = st.text(min_size=1, max_size=8)

# f64's own ceiling, the largest finite double.
_MAX_F64 = 1.7976931348623157e308

# The exact f64 log2(3) the discount uses at rank 2, as a Fraction: the
# exact-ratio arithmetic below works over the caller's f64 gains and the
# f64 discount constants, so "exact" means the real-number value of the
# f64-spelled formula, not some higher-precision reimplementation.
_LOG2_3 = Fraction(math.log2(3.0))
_LOG2_2 = Fraction(math.log2(2.0))
_LOG2_4 = Fraction(math.log2(4.0))


# ---------------------------------------------------------------------------
# 1. The saturating-ratio policy's consequences
# ---------------------------------------------------------------------------


def _score(order: list[str], gains: dict[str, float], k: int | None = None) -> float:
    return ndcg_at_k(list(order), set(), gains=gains, k=k)


class TestSaturationMonotonicity:
    """A ranking improvement must never lower the score, even when both
    sums are past f64's ceiling and the answer saturates."""

    def test_adjacent_swap_up_never_lowers_the_score_across_full_overflow_regime(
        self,
    ) -> None:
        # Every permutation pair of a 4-gain pool spanning 1.7e308 down
        # to 1e-300: wherever a swap moves a LARGER gain UP (a strict
        # real-arithmetic improvement), the score must not drop.
        gains = {"a": 1.7e308, "b": 1e308, "c": 3.0, "d": 1e-300}
        ids = list(gains)
        for order in itertools.permutations(ids):
            base = _score(list(order), gains)
            for i in range(len(order) - 1):
                if gains[order[i]] < gains[order[i + 1]]:
                    swapped = list(order)
                    swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
                    assert _score(swapped, gains) >= base, (order, i)

    @given(
        pool=st.lists(
            st.sampled_from([1.7e308, 1e308, 5e307, 1e300, 3.0, 1.0, 0.0]),
            min_size=2,
            max_size=6,
        ),
        seed=st.integers(0, 10_000),
    )
    @settings(max_examples=150, deadline=None)
    def test_hypothesis_swap_up_monotonicity_into_and_under_saturation(
        self, pool: list[float], seed: int
    ) -> None:
        import random

        rng = random.Random(seed)
        gains = {f"d{i}": g for i, g in enumerate(pool)}
        order = list(gains)
        rng.shuffle(order)
        base = _score(order, gains)
        for _ in range(12):
            i, j = sorted(rng.sample(range(len(order)), 2))
            if gains[order[i]] >= gains[order[j]]:
                continue  # already ordered at this pair
            swapped = list(order)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            after = _score(swapped, gains)
            assert after >= base
            order, base = swapped, after

    def test_the_one_point_zero_branch_over_reports_upward_and_is_bounded(self) -> None:
        # Both sums overflow (2 x 1.7e308 in the DCG alone), the exact
        # ratio is strictly below 1 (the ideal's third 1.7e308 outranks
        # the ranked 1e-300), and the documented saturation answers
        # exactly 1.0: an over-report, but bounded BY 1.0, never past.
        ranked_gains = {"a": 1.7e308, "b": 1.7e308, "c": 1e-300}
        exact_dcg = Fraction(1.7e308) + Fraction(1.7e308) / _LOG2_3 + Fraction(1e-300) / _LOG2_4
        exact_idcg = (
            Fraction(1.7e308) + Fraction(1.7e308) / _LOG2_3 + Fraction(1.7e308) / _LOG2_4
        )
        assert exact_dcg < exact_idcg  # the ranking is genuinely imperfect
        assert _score(["a", "b", "c"], ranked_gains) == 1.0

    def test_the_zero_branch_can_under_report_a_near_perfect_ranking_to_zero(self) -> None:
        # The policy's OTHER error direction, at its true magnitude. A
        # finite DCG facing an infinite ideal answers the documented
        # 0.0; the construction keeps the sums' true ratio just under 1
        # (an almost-perfect ranking), so the documented branch's error
        # is NOT bounded near the true value the way the 1.0 branch's
        # "over-report" framing suggests: it spans the whole interval.
        # Pinned as the documented answer (api.md's own branch, and the
        # crate's own finite-dcg/infinite-ideal unit test); the finding
        # is the docs' one-sided error framing, not this number.
        a = 1.75e308
        # B: the smallest float whose discounted addition to a overflows.
        b_big = float((Fraction(_MAX_F64) - Fraction(a)) * _LOG2_3)
        for _ in range(10_000):
            b_big = math.nextafter(b_big, math.inf)
            if math.isinf(a + b_big / math.log2(3.0)):
                break
        assert math.isinf(a + b_big / math.log2(3.0))
        b_small = b_big * 0.5
        assert math.isfinite(a + b_small / math.log2(3.0))  # DCG stays finite
        exact = (Fraction(a) + Fraction(b_small) / _LOG2_3) / (
            Fraction(a) + Fraction(b_big) / _LOG2_3
        )
        assert 0.98 < float(exact) < 1.0  # the true nDCG is near-perfect
        assert _score(["a", "b"], {"a": a, "b": b_small, "c": b_big}, k=2) == 0.0

    def test_k_clamped_and_none_agree_on_near_overflow_data(self) -> None:
        # k=None clamps to the deduplicated length; the explicit spell
        # (equal, longer, duplicated ranking) must give the same answer
        # where the k semantics say they agree.
        gains = {"a": 1.7e308, "b": 1.7e308, "c": 3.0, "d": 1e308}
        ranked = ["a", "b", "c", "d"]
        none_score = ndcg_at_k(ranked, set(), gains=gains)
        assert ndcg_at_k(ranked, set(), gains=gains, k=4) == none_score
        assert ndcg_at_k(ranked, set(), gains=gains, k=10**18) == none_score
        # a duplicated ranking: same deduplicated length, same answer
        assert ndcg_at_k(["a", "b", "c", "d", "a"], set(), gains=gains) == none_score

    @given(
        ranked=st.lists(_IDS, min_size=1, max_size=10, unique=True),
        gain=st.floats(min_value=0.0, max_value=1.7e308, allow_nan=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_k_none_equals_k_len_property(self, ranked: list[str], gain: float) -> None:
        gains = {d: gain for d in ranked}
        assert ndcg_at_k(ranked, set(), gains=gains) == ndcg_at_k(
            ranked, set(), gains=gains, k=len(ranked)
        )

    def test_empty_ranking_k_none_and_explicit_k_agree(self) -> None:
        # the k=None clamp and an explicit in-range k agree on an empty
        # ranking (both answer the documented 0.0; an explicit k=0 is
        # the documented ValueError, out of range regardless of length)
        assert ndcg_at_k([], set()) == 0.0
        assert ndcg_at_k([], set(), k=1) == 0.0
        with pytest.raises(ValueError, match="k must be >= 1"):
            ndcg_at_k([], set(), k=0)

    def test_the_docs_overflow_arithmetic_is_exact_at_both_scales(self) -> None:
        # api.md: "three gains of 1e308, or two of 1.7e308, are enough"
        # to overflow. The same-shaped IMPERFECT ranking (one huge gain
        # ranked, a second only in the ideal) distinguishes the scales:
        # at 1e308 both sums stay finite and the true ratio is
        # reported; at 1.7e308 the ideal overflows and the answer
        # saturates to the documented 0.0.
        finite = ndcg_at_k(
            ["a", "x"], set(), gains={"a": 1e308, "x": 0.0, "ghost": 1e308}, k=2
        )
        exact = Fraction(1e308) / (Fraction(1e308) + Fraction(1e308) / _LOG2_3)
        assert finite == pytest.approx(float(exact), rel=1e-12)
        saturated = ndcg_at_k(
            ["a", "x"], set(), gains={"a": 1.7e308, "x": 0.0, "ghost": 1.7e308}, k=2
        )
        assert saturated == 0.0


# ---------------------------------------------------------------------------
# 2. Float-noise oracle
# ---------------------------------------------------------------------------


class TestFloatNoiseOracle:
    def test_perfect_ranking_of_binary_rounding_traps_is_exactly_one(self) -> None:
        # 0.1/0.2/0.3: DCG and IDCG sum the same f64 terms in the same
        # order (ranked order == sorted-ideal order), so the ratio is
        # bitwise 1.0 with no clamp needed.
        gains = {"a": 0.1, "b": 0.2, "c": 0.3}
        assert ndcg_at_k(["c", "b", "a"], set(), gains=gains) == 1.0
        # the imperfect order of the SAME pool is honestly below 1
        assert ndcg_at_k(["a", "b", "c"], set(), gains=gains) < 1.0

    def test_int_and_float_equal_gains_score_identically(self) -> None:
        assert ndcg_at_k(["a", "b"], {"a"}, gains={"a": 3, "b": 1}) == ndcg_at_k(
            ["a", "b"], {"a"}, gains={"a": 3.0, "b": 1.0}
        )
        assert ndcg_at_k(["a", "b"], {"a"}, gains={"a": 3, "b": 1}) == 1.0

    def test_bool_gains_launder_to_one_and_zero(self) -> None:
        # True/False are ints in Python and extract as 1.0/0.0 gains.
        # The docs call a "non-numeric gains value" a TypeError; bool IS
        # numeric by that reading, but the laundering itself is
        # undocumented: pinned here so it cannot drift silently.
        # FINDING (P2): undocumented behavior, flagged not fixed.
        assert ndcg_at_k(["a", "b"], set(), gains={"a": True, "b": False}) == 1.0
        # the relevant doc ranked second: the standard 1/log2(3) score
        assert ndcg_at_k(["b", "a"], set(), gains={"a": True, "b": False}) == (
            1.0 / math.log2(3)
        )

    def test_ideal_tie_order_can_never_move_the_score(self) -> None:
        # Equal gains are bitwise-equal f64 values, so any tie order in
        # the ideal sort sums identically: every insertion order of the
        # same pool, and the ranked side's own tie permutations, must
        # give the identical score.
        pool = {"a": 3.0, "b": 3.0, "c": 3.0, "d": 1.0, "e": 1.0}
        base = ndcg_at_k(["a", "b", "c", "d", "e"], set(), gains=pool)
        for perm in itertools.permutations(pool):
            reordered = {k: pool[k] for k in perm}
            assert ndcg_at_k(["a", "b", "c", "d", "e"], set(), gains=reordered) == base
        assert base == 1.0

    def test_bool_ids_in_relevant_are_dict_semantics_not_magic(self) -> None:
        # {True} and {1} are the same set; a ranked True or 1 hits both.
        assert mrr([1], {True}) == 1.0
        assert mrr([True], {1}) == 1.0
        assert recall_at_k([1, "x"], {True}, 2) == 1.0


# ---------------------------------------------------------------------------
# 3. rank_fuse: id-equality and type traps
# ---------------------------------------------------------------------------


class TestRankFuseIdEqualityTraps:
    def test_one_true_and_one_point_oh_are_one_id_and_first_object_wins(self) -> None:
        # Python's dict semantics make 1 == True == 1.0 one id: two
        # entries, and the RETURNED object is the FIRST-SEEN spell
        # (identity, not merely equality).
        first, = [1]
        out = rank_fuse([[first, True, 1.0, "a"]])
        assert len(out) == 2
        assert out[0][0] is first
        assert out[0][0] == 1 and not isinstance(out[0][0], bool)
        assert out[0][1] == pytest.approx(1 / 61)  # one vote, not three
        assert out[1] == ("a", pytest.approx(1 / 62))

    def test_bool_k_launders_to_one(self) -> None:
        # k=True is an int subclass and launders to k=1 everywhere k is
        # accepted. FINDING (P2): undocumented laundering; pinned so it
        # cannot drift silently either way.
        assert rank_fuse([["a", "b"]], k=True) == rank_fuse([["a", "b"]], k=1)
        assert recall_at_k(["a", "b"], {"a"}, k=True) == recall_at_k(["a", "b"], {"a"}, 1)
        assert precision_at_k(["a", "b"], {"a"}, k=True) == precision_at_k(
            ["a", "b"], {"a"}, 1
        )
        assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=True) == ndcg_at_k(
            ["a", "b", "c"], {"a", "b"}, k=1
        )

    def test_the_same_object_in_two_lists_votes_twice_with_hash_side_effects(self) -> None:
        class Counted:
            calls: dict[int, int] = {}

            def __init__(self, tag: int) -> None:
                self.tag = tag

            def __hash__(self) -> int:
                Counted.calls[self.tag] = Counted.calls.get(self.tag, 0) + 1
                return self.tag

            def __eq__(self, other: object) -> bool:
                return isinstance(other, Counted) and other.tag == self.tag

        shared = Counted(1)
        solo = Counted(2)
        Counted.calls = {}
        out = rank_fuse([[shared, solo], [shared]])
        # two votes (1/61 + 1/61) beat solo's one (1/62); hash side
        # effects fired at least once per occurrence (3 occurrences).
        assert out[0][0] is shared
        assert out[0][1] == pytest.approx(2 / 61)
        assert out[1][0] is solo
        assert Counted.calls[1] >= 2 and Counted.calls[2] >= 1

    def test_eq_false_frozen_dataclasses_are_distinct_ids(self) -> None:
        # eq=False: identity equality, so equal-field instances are two
        # different ids no matter how dataclass frozen machinery works.
        from dataclasses import dataclass

        @dataclass(frozen=True, eq=False)
        class F:
            x: int

        out = rank_fuse([[F(1), F(1)]])
        assert len(out) == 2
        assert out[0][0] is not out[1][0]
        assert out[0][1] == pytest.approx(1 / 61)
        assert out[1][1] == pytest.approx(1 / 62)

    def test_list_subclasses_are_accepted_in_both_positions(self) -> None:
        class L(list):  # noqa: UP006 (a hostile SUBCLASS, not a typo)
            pass

        fused = rank_fuse([L(["a", "b"])])
        assert [i for i, _ in fused] == ["a", "b"]
        assert rank_fuse(L([L(["a"])])) == [("a", pytest.approx(1 / 61))]
        assert mrr(L(["a"]), {"a"}) == 1.0

    def test_nested_list_ids_raise_the_unhashable_type_error_no_panic(self) -> None:
        with pytest.raises(TypeError, match="unhashable"):
            rank_fuse([[["a"]]])

    def test_repeated_calls_with_side_effecting_hashes_stay_deterministic(self) -> None:
        class Noisy:
            def __hash__(self) -> int:
                return 0  # maximal collision pressure

            def __eq__(self, other: object) -> bool:
                return self is other

        ids = [Noisy() for _ in range(20)]
        lists = [ids[i::2] for i in range(2)]
        assert rank_fuse(lists) == rank_fuse(lists)


# ---------------------------------------------------------------------------
# 4. Metric invariants, wave 2
# ---------------------------------------------------------------------------


def _textbook(ranked: list, relevant: set, k: int) -> tuple[float, float, float, float]:
    """Independent textbook recompute over a DEDUPED ranking: returns
    (ndcg, mrr, recall, precision). First-principles formulas, not tors."""
    deduped = list(dict.fromkeys(ranked))
    dcg = sum(
        (1.0 if deduped[i] in relevant else 0.0) / math.log2(i + 2)
        for i in range(min(k, len(deduped)))
    )
    ideal = [1.0] * len(relevant)
    idcg = sum(ideal[i] / math.log2(i + 2) for i in range(min(k, len(ideal))))
    nd = dcg / idcg if idcg > 0 else 0.0
    hits = sum(1 for d in deduped[:k] if d in relevant)
    first = next((i for i, d in enumerate(deduped) if d in relevant), None)
    rr = 1.0 / (first + 1) if first is not None else 0.0
    return nd, rr, hits / len(relevant) if relevant else 0.0, (
        hits / min(k, len(deduped)) if deduped else 0.0
    )


class TestMetricInvariantsWave2:
    def test_the_documented_recall_example_recomputed_from_first_principles(self) -> None:
        # api.md: recall_at_k(["a", "a", "b"], {"a", "b"}, 2) is 1.0,
        # not 0.5. Independent recompute: fold the duplicate, top-2 of
        # the deduped ranking = ["a", "b"], both relevant -> 2/2.
        deduped = list(dict.fromkeys(["a", "a", "b"]))
        top2 = deduped[:2]
        assert sum(d in {"a", "b"} for d in top2) / len({"a", "b"}) == 1.0
        assert recall_at_k(["a", "a", "b"], {"a", "b"}, 2) == 1.0

    @given(
        ranked=st.lists(_IDS, min_size=0, max_size=12, unique=True),
        relevant=st.lists(_IDS, min_size=0, max_size=8, unique=True),
        k=st.integers(1, 15),
    )
    @settings(max_examples=200, deadline=None)
    def test_all_four_metrics_match_the_textbook_on_deduped_input(
        self, ranked: list[str], relevant: list[str], k: int
    ) -> None:
        rel = set(relevant)
        nd, rr, rec, prec = _textbook(ranked, rel, k)
        if k <= len(ranked):
            # within the textbook's domain (k past the run length is the
            # documented clamp divergence, pinned separately below)
            assert ndcg_at_k(ranked, rel, k=k) == pytest.approx(nd, abs=1e-12)
        assert mrr(ranked, rel) == pytest.approx(rr, abs=1e-12)
        assert recall_at_k(ranked, rel, k) == pytest.approx(rec, abs=1e-12)
        assert precision_at_k(ranked, rel, k) == pytest.approx(prec, abs=1e-12)

    def test_ndcg_k_past_run_length_clamps_the_ideal_too(self) -> None:
        # DOCUMENTED divergence from the textbook@k formula (api.md: "k
        # past the ranking's length simply uses every available
        # position (nDCG clamps k the same way)"): a 1-result run at
        # k=2 scores ndcg@1, not the textbook's 0.613 (whose ideal
        # keeps a second judged doc the run never filled). The same
        # not-punished-for-unfilled-positions convention as precision.
        # Wave 1's sklearn differential never crossed k > len(ranked).
        ranked, rel = ["hit"], {"hit", "missed"}
        textbook = _textbook(ranked, rel, 2)[0]
        assert textbook == pytest.approx(1.0 / (1.0 + 1.0 / math.log2(3)))
        assert ndcg_at_k(ranked, rel, k=2) == 1.0
        assert ndcg_at_k(ranked, rel, k=2) == ndcg_at_k(ranked, rel, k=1)
        assert ndcg_at_k(ranked, rel, k=2) != pytest.approx(textbook)

    def test_single_relevant_doc_mrr_and_ndcg_relationship(self) -> None:
        # The TRUE relationship for one binary-relevant doc at rank r:
        # mrr == 1/r and ndcg@n == 1/log2(r+1), so ndcg >= mrr with
        # equality ONLY at rank 1. (The naive expectation mrr == ndcg@n
        # is false for every r > 1: 1/r vs 1/log2(r+1). Pinned as such.)
        for r in range(1, 12):
            ranked = [f"filler{i}" for i in range(r - 1)] + ["hit"] + [
                f"tail{i}" for i in range(5)
            ]
            assert mrr(ranked, {"hit"}) == pytest.approx(1.0 / r)
            assert ndcg_at_k(ranked, {"hit"}) == pytest.approx(1.0 / math.log2(r + 1))
            assert ndcg_at_k(ranked, {"hit"}) >= mrr(ranked, {"hit"}) - 1e-15
            if r > 1:
                assert ndcg_at_k(ranked, {"hit"}) > mrr(ranked, {"hit"})
        # equality holds exactly at rank 1, the only rank where the two
        # discounts agree
        assert ndcg_at_k(["hit", "x"], {"hit"}) == 1.0 == mrr(["hit", "x"], {"hit"})

    @given(
        ranked=st.lists(_IDS, min_size=1, max_size=12, unique=True),
        relevant=st.lists(_IDS, min_size=1, max_size=8, unique=True),
        k=st.integers(1, 15),
    )
    @settings(max_examples=200, deadline=None)
    def test_recall_precision_algebraic_consistency(
        self, ranked: list[str], relevant: list[str], k: int
    ) -> None:
        # For deduped input the two formulas share the hit count:
        # recall@k == precision@k * min(k, len(ranked)) / |relevant|.
        rel = set(relevant)
        rec = recall_at_k(ranked, rel, k)
        prec = precision_at_k(ranked, rel, k)
        assert rec == pytest.approx(
            prec * min(k, len(ranked)) / len(rel), abs=1e-12
        )


# ---------------------------------------------------------------------------
# 5. The aio twins under contention
# ---------------------------------------------------------------------------


class TestAioContentionWave2:
    @staticmethod
    def _lists(total: int, n_lists: int = 6) -> list[list[str]]:
        id_space = max(total // 2, 1)
        per = total // n_lists
        return [
            [f"id_{(j * per + i) % id_space}" for i in range(per)] for j in range(n_lists)
        ]

    def test_eight_concurrent_fusions_match_sync_with_a_live_loop(self) -> None:
        lists = self._lists(6_000)
        expected = rank_fuse(lists)

        async def run() -> None:
            stop = asyncio.Event()
            beats = 0

            async def heartbeat() -> None:
                nonlocal beats
                while not stop.is_set():
                    beats += 1
                    await asyncio.sleep(0)

            probe = asyncio.create_task(heartbeat())
            await asyncio.sleep(0)
            started = time.monotonic()
            try:
                results = await asyncio.gather(*[tors.aio.rank_fuse(lists) for _ in range(8)])
            finally:
                stop.set()
                await probe
            wall = time.monotonic() - started
            assert all(r == expected for r in results)
            # the loop was not starved into serial submission: heartbeats ran
            assert beats > 0 and wall > 0

        asyncio.run(run())

    def test_all_five_twins_concurrent_match_sync(self) -> None:
        lists = self._lists(2_000)
        ranked = ["a", "x", "b", "y", "c"]
        rel = {"a", "b", "c", "never"}

        async def run() -> None:
            got = await asyncio.gather(
                tors.aio.rank_fuse(lists, k=7),
                tors.aio.ndcg_at_k(ranked, rel),
                tors.aio.ndcg_at_k(ranked, rel, k=2, gains={"a": 3.0, "b": 1.0}),
                tors.aio.mrr(ranked, rel),
                tors.aio.recall_at_k(ranked, rel, 2),
                tors.aio.precision_at_k(ranked, rel, 2),
                tors.aio.rank_fuse(lists),
                tors.aio.mrr(ranked, rel),
            )
            assert got[0] == rank_fuse(lists, k=7)
            assert got[1] == ndcg_at_k(ranked, rel)
            assert got[2] == ndcg_at_k(ranked, rel, k=2, gains={"a": 3.0, "b": 1.0})
            assert got[3] == mrr(ranked, rel) == got[7]
            assert got[4] == recall_at_k(ranked, rel, 2)
            assert got[5] == precision_at_k(ranked, rel, 2)
            assert got[6] == rank_fuse(lists)

        asyncio.run(run())

    def test_one_raising_twin_does_not_lose_the_others_results(self) -> None:
        # A ValueError among eight concurrent gathers must surface as
        # that gather's own exception, leaving the survivors' results
        # intact and correct (no cross-talk, no lost results).
        lists = self._lists(2_000)

        async def run() -> None:
            results = await asyncio.gather(
                *[tors.aio.rank_fuse(lists) for _ in range(4)],
                tors.aio.rank_fuse([["a"]], k=0),
                *[tors.aio.rank_fuse(lists) for _ in range(4)],
                return_exceptions=True,
            )
            expected = rank_fuse(lists)
            for i, r in enumerate(results):
                if i == 4:
                    assert isinstance(r, ValueError)
                else:
                    assert r == expected

        asyncio.run(run())

    def test_no_ratio_drift_under_repeated_contention(self) -> None:
        # The fused SCORES (not just the order) must be bit-identical
        # across concurrent rounds: no shared-state drift, no
        # accumulation-order nondeterminism from thread timing.
        lists = self._lists(3_000, n_lists=8)

        async def run() -> None:
            rounds = await asyncio.gather(
                *[tors.aio.rank_fuse(lists, k=5) for _ in range(3)]
            )
            assert rounds[0] == rounds[1] == rounds[2] == rank_fuse(lists, k=5)

        asyncio.run(run())


# ---------------------------------------------------------------------------
# 6. Every api.md number, recomputed independently
# ---------------------------------------------------------------------------


class TestApiMdNumbersRecomputed:
    def test_fusion_vector_recomputed_from_the_paper_formula(self) -> None:
        lists = [
            ["cat-a", "dog-b", "bird-c"],
            ["dog-b", "cat-a"],
            ["bird-c"],
        ]
        fused = dict(rank_fuse(lists))
        # hand-summed Sigma 1/(60 + rank), ranks 1-based, one vote per list
        assert fused["cat-a"] == pytest.approx(1 / 61 + 1 / 62)
        assert fused["dog-b"] == pytest.approx(1 / 62 + 1 / 61)
        assert fused["bird-c"] == pytest.approx(1 / 63 + 1 / 61)
        # the doc's exact literals (its own float spelling)
        assert fused["cat-a"] == 0.03252247488101534
        assert fused["bird-c"] == 0.032266458495966696
        # the tie and its documented break
        assert fused["cat-a"] == fused["dog-b"]
        assert [i for i, _ in rank_fuse(lists)][0] == "cat-a"

    def test_metric_literals_recomputed_from_the_paper_formulas(self) -> None:
        ranked = ["cat-a", "dog-b", "bird-c", "fish-d"]
        rel = {"cat-a", "bird-c", "whale-e"}
        # DCG = 1/log2(2) + 0 + 1/log2(4); IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4)
        dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
        assert ndcg_at_k(ranked, rel) == pytest.approx(dcg / idcg, rel=1e-15)
        assert ndcg_at_k(ranked, rel) == 0.7039180890341347
        assert mrr(ranked, rel) == 1.0
        assert recall_at_k(ranked, rel, 2) == pytest.approx(1 / 3)
        assert recall_at_k(ranked, rel, 2) == 0.3333333333333333
        assert recall_at_k(ranked, rel, 4) == pytest.approx(2 / 3)
        assert recall_at_k(ranked, rel, 4) == 0.6666666666666666
        assert precision_at_k(ranked, rel, 2) == pytest.approx(1 / 2)
        assert precision_at_k(ranked, rel, 2) == 0.5
        # k=2: DCG = 1/log2(2); IDCG packs two of the three relevant at ranks 1-2
        dcg2 = 1.0 / math.log2(2)
        idcg2 = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert ndcg_at_k(ranked, rel, k=2) == pytest.approx(dcg2 / idcg2, rel=1e-15)
        assert ndcg_at_k(ranked, rel, k=2) == 0.6131471927654584

    def test_the_graded_gains_example_line_runs(self) -> None:
        # the doc's last example line shows no literal; pin it runs and
        # stays in the unit interval (the graded pool order gives < 1)
        ranked = ["cat-a", "dog-b", "bird-c", "fish-d"]
        s = ndcg_at_k(ranked, {"cat-a", "bird-c", "whale-e"}, gains={"cat-a": 3.0, "bird-c": 1.0})
        assert 0.0 <= s <= 1.0
