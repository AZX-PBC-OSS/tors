"""Contract gate for the rank-fusion and IR-metric family:
``tors.rank_fuse``, ``tors.ndcg_at_k``, ``tors.mrr``, ``tors.recall_at_k``,
and ``tors.precision_at_k``.

rank_fuse is Reciprocal Rank Fusion exactly as the source paper defines it —
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

import itertools
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors
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
        # "early" first appears in list 0; "late" in list 1 — equal
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
