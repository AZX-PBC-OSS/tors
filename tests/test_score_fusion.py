"""Contract gate for the score-based fusion sibling of the rank-fusion
family: ``tors.score_fuse``, the score-space counterpart of rank_fuse
(same id table, same first-appearance tie-break, same emission
contract; raw similarity scores consumed instead of ranks).

The three methods (the score-based siblings Cormack, Clarke &
Buüttcher's SIGIR 2009 RRF paper compares against, plus Elasticsearch's
linear-retriever pattern):

- ``combmnz`` (the default), Fox & Shaw, "Combination of Multiple
  Searches", TREC-2 1994: ``score(d) = lists(d) x Σ w_i × norm_i(d)`` --
  CombSUM × the containing-list count, ``norm`` min-max per list over
  the list's OWN scores. The best of the score-based family in the
  Cormack 2009 comparison, hence the default.
- ``borda``: the rank-based count -- ``Σ w_i × (n - rank)/n``. RANK
  votes deliberately, the literature's Borda count (each list elects
  its top with n-1 points, its last with 0), rescaled to ``[0, 1)`` so
  a weight means the same thing over lists of any length; normalizing
  the SCORES there would smuggle magnitudes into the one method whose
  whole point is that only the ordering votes.
- ``linear``: Elasticsearch's linear retriever -- ``Σ w_i × norm_i(d)``,
  no MNZ multiplier.

The pinned conventions: a zero-range list (every score equal, a
single-entry list included) normalizes to the neutral midpoint ``0.5``;
negative scores are LEGAL (min-max maps any finite range onto ``[0,
1]``; the rejected domain is only non-finite scores -- NaN and both
infinites are ``ValueError``, an infinite min or max makes the range
ill-defined); a range that itself overflows (``-1.7e308`` to
``1.7e308``) saturates instead of dividing ``inf/inf`` (NaN): the
overflowed numerator answers ``1.0``, the monotone NaN-free policy
ndcg_at_k's saturating ratio carries. Emission is vote-existence, one
``(id, score)`` pair per distinct id (a fused ``0.0`` -- Borda's last
place, an underflowed denormal weight -- still appears, ordered last);
order is fused score descending, ties by earliest first appearance
across the lists in caller order (rank_fuse's contract, extended).

The differential oracle is ``reference.reference_score_fuse`` (a naive
pure-Python walk over dicts and min/max, sharing no machinery with the
Rust core), pinned byte-exact on every hypothesis corpus below.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors
from reference import reference_score_fuse

# Ids: arbitrary hashable text (the fusion is an id-space operation:
# the actual characters are irrelevant, which is exactly the property
# the unicode strategies pin).
_IDS = st.text(min_size=1, max_size=12)

# Scores: the full finite f64 domain (subnormals, ±1.7e308, negatives:
# all legal), NaN/inf excluded (the binding's ValueError domain).
_SCORES = st.floats(allow_nan=False, allow_infinity=False, width=64)

# The boundary magnitudes, sampled alongside the continuous strategy:
# the extreme-magnitude corpora the differential must always reach.
_EXTREME_SCORES = st.sampled_from(
    [0.0, -0.0, 1e-300, -1e-300, 1e300, -1e300, 1e-3, 5.0, 1e308, -1e308]
)

_SCORED_LISTS = st.lists(
    st.lists(st.tuples(_IDS, _SCORES), min_size=1, max_size=12), min_size=1, max_size=6
)


def _assert_matches_oracle(
    scored_lists: list, method: str, weights: list | None = None, k: int | None = None
) -> None:
    """The byte-exact differential: tors against the pure-Python oracle,
    tuples and all (float equality in a tuple compare is bit equality
    modulo -0.0, which the two implementations cannot disagree about:
    neither ever negates a zero it did not receive)."""
    expected = reference_score_fuse(scored_lists, method=method, weights=weights, k=k)
    got = tors.score_fuse(scored_lists, method=method, weights=weights, k=k)
    assert got == expected, (method, got, expected)


class TestScoreFuseKnownVectors:
    def test_hand_computed_combmnz(self) -> None:
        # L0's range [0.5, 1.0]: cat-a norms 1.0, dog-b 0.0. L1's range
        # [0.2, 0.8]: dog-b 1.0, cat-a 0.0. CombSUM: cat-a 1.0, dog-b
        # 1.0; both in both lists, count 2 -> 2.0 each; bird-c rides a
        # single-entry (zero-range) list at the neutral 0.5, count 1.
        lists = [
            [("cat-a", 1.0), ("dog-b", 0.5)],
            [("dog-b", 0.8), ("cat-a", 0.2)],
            [("bird-c", 3.0)],
        ]
        fused = tors.score_fuse(lists)
        assert fused[0] == ("cat-a", 2.0)
        assert fused[1] == ("dog-b", 2.0)
        assert fused[2] == ("bird-c", 0.5)

    def test_hand_computed_borda(self) -> None:
        # Rank votes over the deduplicated lists: L0 (n=2): cat-a 1/2,
        # dog-b 0. L1 (n=2): dog-b 1/2, cat-a 0. bird-c's solo list
        # (n=1) votes (1-1)/1 = 0 -- last place of a one-document list.
        lists = [
            [("cat-a", 10.0), ("dog-b", 0.001)],
            [("dog-b", 0.7), ("cat-a", 0.6)],
            [("bird-c", 99.0)],
        ]
        fused = tors.score_fuse(lists, method="borda")
        assert [i for i, _ in fused] == ["cat-a", "dog-b", "bird-c"]
        assert fused[0][1] == pytest.approx(1 / 2)
        assert fused[1][1] == pytest.approx(1 / 2)
        assert fused[2] == ("bird-c", 0.0)  # present, at its zero vote

    def test_hand_computed_linear(self) -> None:
        # The MNZ multiplier removed: the same norms, no consensus
        # boost -- cat-a 1.0, dog-b 1.0, bird-c 0.5.
        lists = [
            [("cat-a", 1.0), ("dog-b", 0.5)],
            [("dog-b", 0.8), ("cat-a", 0.2)],
            [("bird-c", 3.0)],
        ]
        fused = tors.score_fuse(lists, method="linear")
        assert fused == [
            ("cat-a", 1.0),
            ("dog-b", 1.0),
            ("bird-c", 0.5),
        ]

    def test_min_max_norms_hit_exactly_one_and_zero(self) -> None:
        # Each list's top scores exactly 1.0, bottom exactly 0.0 --
        # whatever the raw magnitudes (10.0 and 1e-300 alike).
        lists = [
            [("a", 10.0), ("b", 5.0), ("c", 1e-300)],
            [("d", 1e300), ("e", -1e300)],
        ]
        fused = dict(tors.score_fuse(lists, method="linear"))
        assert fused["a"] == 1.0
        assert fused["c"] == 0.0
        assert fused["d"] == 1.0
        assert fused["e"] == 0.0

    def test_norms_are_min_max_per_list_over_that_lists_own_scores(self) -> None:
        # L0's range is [10, 20], L1's [0, 0.5]: a list's scores
        # normalize against the list, never the global extremes (the
        # cross-list magnitude comparison the whole method exists to
        # erase).
        lists = [
            [("a", 20.0), ("b", 10.0)],
            [("c", 0.5), ("d", 0.0)],
        ]
        fused = dict(tors.score_fuse(lists, method="linear"))
        assert fused["a"] == 1.0
        assert fused["b"] == 0.0
        assert fused["c"] == 1.0
        assert fused["d"] == 0.0

    def test_single_list_orders_by_the_methods_own_vote(self) -> None:
        # One list in, one order out -- but WHICH order is the method's
        # own vote: Borda preserves the list's POSITIONS (d at rank 4
        # votes 1/5, e at rank 5 votes 0; the raw scores are invisible),
        # the min-max methods re-order by the SCORES (e's 0.0 beats d's
        # -2.0's 0.0-floor... e norms 2/5, d norms 0.0). Both are the
        # documented contract; ties keep first appearance.
        pairs = [("a", 3.0), ("b", 3.0), ("c", 1.0), ("d", -2.0), ("e", 0.0)]
        assert [i for i, _ in tors.score_fuse([pairs], method="borda")] == [
            "a",
            "b",
            "c",
            "d",
            "e",
        ]
        for method in ("combmnz", "linear"):
            assert [i for i, _ in tors.score_fuse([pairs], method=method)] == [
                "a",
                "b",
                "c",
                "e",
                "d",
            ], method


class TestScoreFuseMethodsDiffer:
    """The method knob is real: one corpus, three different fused
    orders, each pinned -- both the consensus boost (combmnz vs linear)
    and the rank-vs-score vote (borda vs both) move the winner."""

    _LISTS = [
        [("p", 1.0), ("f0", 0.0)],  # p norms 1.0, f0 0.0
        [("hi1", 1.0), ("q", 0.3), ("lo1", 0.0)],  # hi1 1.0, q 0.3, lo1 0.0
        [("hi2", 1.0), ("q", 0.3), ("lo2", 0.0)],  # hi2 1.0, q 0.3, lo2 0.0
    ]

    def test_linear_ranks_the_single_norm_one_vote_first(self) -> None:
        # Bare CombSUM: p's 1.0 beats q's 0.3 + 0.3 = 0.6; the hi docs'
        # 1.0s tie p and first appearance orders them.
        fused = tors.score_fuse(self._LISTS, method="linear")
        assert [i for i, _ in fused] == ["p", "hi1", "hi2", "q", "f0", "lo1", "lo2"]
        assert fused[0][1] == 1.0
        assert fused[3][1] == pytest.approx(0.3 + 0.3)

    def test_combmnz_ranks_the_two_list_consensus_first(self) -> None:
        # The MNZ multiplier: q = (0.3 + 0.3) × 2 = 1.2 beats p's
        # 1.0 × 1 = 1.0 -- the exact flip linear answers.
        fused = tors.score_fuse(self._LISTS, method="combmnz")
        assert [i for i, _ in fused] == ["q", "p", "hi1", "hi2", "f0", "lo1", "lo2"]
        assert fused[0][1] == pytest.approx((0.3 + 0.3) * 2)
        assert fused[1][1] == 1.0

    def test_borda_ranks_by_position_and_moves_q_up_again(self) -> None:
        # Rank votes: hi1/hi2 top their lists (2/3 each), q is second
        # of three in TWO lists (1/3 + 1/3 = 2/3), p is first of TWO
        # (1/2). Three-way tie at 2/3, first appearance (hi1 walked
        # before q before hi2).
        fused = tors.score_fuse(self._LISTS, method="borda")
        assert [i for i, _ in fused] == ["hi1", "q", "hi2", "p", "f0", "lo1", "lo2"]
        assert fused[0][1] == pytest.approx(2 / 3)
        assert fused[1][1] == pytest.approx(1 / 3 + 1 / 3)
        assert fused[3][1] == pytest.approx(1 / 2)

    def test_the_three_orders_are_three_different_orders(self) -> None:
        # The pin's own teeth, stated outright: any implementation that
        # confuses the methods (drops the MNZ multiplier, normalizes
        # scores where ranks belong) breaks one of the three pins above.
        orders = {
            method: [i for i, _ in tors.score_fuse(self._LISTS, method=method)]
            for method in ("combmnz", "borda", "linear")
        }
        assert orders["combmnz"] != orders["linear"]
        assert orders["combmnz"] != orders["borda"]
        assert orders["linear"] != orders["borda"]

    def test_method_difference_survives_weights(self) -> None:
        # The same flip under weights (the weighted-RRF philosophy:
        # weights re-scale contributions, they do not change what a
        # method votes with). With q's norm at 0.5 in each of its two
        # unweighted lists and p's list weighted 1.5: linear answers
        # p 1.5 > q (0.5 + 0.5) = 1.0, combmnz answers q (0.5 + 0.5)
        # x 2 lists = 2.0 > p 1.5 -- the two methods still disagree,
        # now under a weighted spelling.
        lists = [
            [("p", 1.0), ("f0", 0.0)],
            [("hi1", 1.0), ("q", 0.5), ("lo1", 0.0)],
            [("hi2", 1.0), ("q", 0.5), ("lo2", 0.0)],
        ]
        weights = [1.5, 1.0, 1.0]
        linear = [i for i, _ in tors.score_fuse(lists, method="linear", weights=weights)]
        combmnz = [i for i, _ in tors.score_fuse(lists, method="combmnz", weights=weights)]
        assert linear != combmnz
        assert linear[0] == "p"
        assert combmnz[0] == "q"


class TestScoreFuseNormConventions:
    def test_zero_range_list_normalizes_to_one_half(self) -> None:
        # The pinned convention: a list whose scores are all equal
        # carries order information only; every entry normalizes to the
        # neutral midpoint 0.5 (never 0.0, which would read as a
        # bottom-of-range vote, nor 1.0).
        lists = [[("a", 7.0), ("b", 7.0), ("c", 7.0)]]
        fused = tors.score_fuse(lists, method="linear")
        assert [i for i, _ in fused] == ["a", "b", "c"]
        assert all(s == 0.5 for _, s in fused)
        # One list: the count multiplier is 1 (each doc is in exactly
        # one list) -> 0.5 each, same as linear.
        assert all(s == pytest.approx(0.5) for _, s in tors.score_fuse(lists, method="combmnz"))
        # Three identical zero-range lists: norms 0.5 from each (sum
        # 1.5), count 3 -> 4.5 each.
        triple = [lists[0], lists[0], lists[0]]
        fused = tors.score_fuse(triple, method="combmnz")
        assert all(s == pytest.approx(0.5 * 3 * 3) for _, s in fused)

    def test_single_entry_list_is_the_zero_range_shape(self) -> None:
        fused = dict(tors.score_fuse([[("solo", 1e300)]], method="linear"))
        assert fused["solo"] == 0.5

    def test_negative_scores_are_legal(self) -> None:
        # Min-max maps ANY finite range onto [0, 1]: a cosine-
        # similarity list (-1..1) and a BM25 list (0..40) normalize to
        # the same interval, the signs of the raw scores wash out. The
        # rejected domain is only non-finite scores.
        lists = [
            [("cos-a", 1.0), ("cos-b", 0.0), ("cos-c", -1.0)],
            [("bm25-a", 40.0), ("bm25-b", 20.0), ("bm25-c", 0.0)],
        ]
        fused = dict(tors.score_fuse(lists, method="linear"))
        assert fused["cos-a"] == 1.0
        assert fused["cos-b"] == 0.5
        assert fused["cos-c"] == 0.0
        assert fused["bm25-a"] == 1.0
        assert fused["bm25-b"] == 0.5
        assert fused["bm25-c"] == 0.0

    def test_extreme_magnitudes_stay_in_the_unit_interval(self) -> None:
        # 1e-300 and 1e300 in one list: the range is finite (2e300),
        # the norms are the ordinary ratios, the magnitudes never leak
        # past the normalization.
        lists = [[("big", 1e300), ("mid", 5e299), ("tiny", 1e-300)]]
        fused = dict(tors.score_fuse(lists, method="linear"))
        assert fused["big"] == 1.0
        assert fused["mid"] == pytest.approx(5e299 / (1e300 - 1e-300))
        assert 0.0 < fused["mid"] < 1.0
        assert fused["tiny"] == 0.0

    def test_overflowed_range_saturates_at_one_not_nan(self) -> None:
        # -1.7e308 to +1.7e308: the range itself overflows to +inf and
        # the top entry's numerator overflows with it -- inf/inf is
        # NaN in IEEE; the saturating-norm policy answers exactly 1.0
        # at the top of the scale (ndcg_at_k's saturating-ratio
        # precedent), finite numerators divide to 0.0-scale values.
        lists = [[("top", 1.7e308), ("zero", 0.0), ("bottom", -1.7e308)]]
        fused = dict(tors.score_fuse(lists, method="linear"))
        assert fused["top"] == 1.0
        assert fused["bottom"] == 0.0
        assert fused["zero"] == 0.0
        # No NaN anywhere (the sort's total-order guarantee), the
        # order still score-descending.
        assert not any(math.isnan(s) for s in fused.values())
        assert fused["top"] >= fused["zero"] >= fused["bottom"]

    def test_overflowed_range_across_lists_stays_nan_free(self) -> None:
        # Two lists, each internally spanning the overflow regime: the
        # fused sums stay finite-or-+inf (all-nonneg terms), the sort
        # stays total.
        lists = [
            [("a", 1.7e308), ("b", -1.7e308)],
            [("c", 1e10), ("d", 1e-10)],
        ]
        for method in ("combmnz", "linear"):
            fused = tors.score_fuse(lists, method=method)
            assert not any(math.isnan(s) for _, s in fused), method
            scores = [s for _, s in fused]
            assert scores == sorted(scores, reverse=True)

    def test_mixed_extreme_magnitudes_across_lists(self) -> None:
        # A 1e300-scale list and a 1e-300-scale list fuse cleanly:
        # each normalizes against its own range.
        lists = [
            [("huge", 1e300), ("huge2", 1e299)],
            [("tiny", 1e-300), ("tiny2", 1e-301)],
        ]
        fused = dict(tors.score_fuse(lists, method="linear"))
        assert fused["huge"] == 1.0
        assert fused["tiny"] == 1.0
        assert fused["huge2"] == 0.0  # its own list's min
        assert fused["tiny2"] == 0.0  # likewise


class TestScoreFuseTies:
    def test_tie_broken_by_earliest_first_appearance(self) -> None:
        # Both norm 1.0 in their own (zero-range, 0.5) lists -- every
        # method ties them; the earlier list wins.
        for method in ("combmnz", "borda", "linear"):
            fused = tors.score_fuse([[("x", 5.0)], [("y", 5.0)]], method=method)
            assert [i for i, _ in fused] == ["x", "y"], method

    def test_first_appearance_beats_id_sort_order(self) -> None:
        fused = tors.score_fuse([[("zebra", 1.0)], [("alpaca", 1.0)]], method="linear")
        assert [i for i, _ in fused] == ["zebra", "alpaca"]

    def test_exact_tie_under_weights_breaks_by_first_appearance(self) -> None:
        # 2.0 × 1.0 vs 1.0 × (1.0 + 1.0): an exact tie under weights;
        # the earlier first appearance wins, deterministic.
        lists = [[("a", 10.0), ("f", 0.0)], [("b", 1.0), ("g", 0.0)], [("b", 1.0), ("h", 0.0)]]
        fused = tors.score_fuse(lists, method="linear", weights=[2.0, 1.0, 1.0])
        assert fused[0][0] == "a" and fused[1][0] == "b"
        assert fused[0][1] == fused[1][1]

    def test_all_tied_zero_range_documents_keep_first_appearance_order(self) -> None:
        fused = tors.score_fuse([[("a", 5.0)], [("b", 5.0)], [("c", 5.0)]], method="linear")
        assert [i for i, _ in fused] == ["a", "b", "c"]


class TestScoreFuseArguments:
    def test_zero_lists_raises_value_error(self) -> None:
        # The merkle_root "root of no chunks" precedent, rank_fuse's
        # own error shape: fusing nothing is almost certainly an
        # upstream bug.
        with pytest.raises(ValueError, match="at least one scored list"):
            tors.score_fuse([])

    def test_non_list_outer_argument_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            tors.score_fuse("not a list")  # type: ignore[arg-type]

    def test_non_list_entry_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be a list"):
            tors.score_fuse([[("a", 1.0)], ("b", 1.0)])  # type: ignore[list-item]

    @pytest.mark.parametrize("bad_pair", [("a",), ("a", 1.0, 2.0), [], 42, {"a": 1.0}])
    def test_malformed_pairs_raise_type_error(self, bad_pair: object) -> None:
        with pytest.raises(TypeError, match="pair"):
            tors.score_fuse([[bad_pair]])  # type: ignore[list-item]

    def test_a_two_char_string_is_a_pair_of_one_char_strings(self) -> None:
        # The sequence protocol is the plain one (the weights=b"12"
        # convention): "ab" IS a 2-element sequence, launders to
        # ("a", "b"), and dies at the score extraction -- TypeError,
        # not the pair-shape error.
        with pytest.raises(TypeError, match="score must be a number"):
            tors.score_fuse([["ab"]])  # type: ignore[list-item]

    def test_tuple_and_list_pairs_are_accepted(self) -> None:
        assert tors.score_fuse([[("a", 1.0)]]) == tors.score_fuse([[["a", 1.0]]])

    def test_non_numeric_score_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="score must be a number"):
            tors.score_fuse([[("a", "high")]])  # type: ignore[list-item]

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    def test_non_finite_score_raises_value_error(self, bad: float) -> None:
        with pytest.raises(ValueError, match="scores values must be finite"):
            tors.score_fuse([[("a", 1.0), ("b", bad)]])  # type: ignore[list-item]

    def test_int_and_bool_scores_launder_to_floats(self) -> None:
        # The int-extraction convention: 2 and True extract as 2.0 and
        # 1.0 (the weights=/gains= convention, same walk).
        assert tors.score_fuse([[("a", 2), ("b", True)]]) == tors.score_fuse(
            [[("a", 2.0), ("b", 1.0)]]
        )

    def test_negative_scores_are_not_rejected(self) -> None:
        # The domain decision, pinned: negatives are legal (min-max
        # maps any finite range onto [0, 1]); only non-finites raise.
        fused = tors.score_fuse([[("a", -5.0), ("b", -1.0)]], method="linear")
        assert dict(fused)["b"] == 1.0
        assert dict(fused)["a"] == 0.0

    @pytest.mark.parametrize("k", [0, -1, -7])
    def test_k_below_one_raises_value_error(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            tors.score_fuse([[("a", 1.0)]], k=k)

    @pytest.mark.parametrize("bad_k", [1.5, "3"])
    def test_non_integer_k_is_a_type_error(self, bad_k: object) -> None:
        with pytest.raises(TypeError):
            tors.score_fuse([[("a", 1.0)]], k=bad_k)  # type: ignore[arg-type]

    def test_k_above_i64_is_an_overflow_error(self) -> None:
        with pytest.raises(OverflowError):
            tors.score_fuse([[("a", 1.0)]], k=10**19)

    def test_k_none_is_all_and_k_n_is_the_top_n(self) -> None:
        lists = [[("a", 3.0), ("b", 2.0), ("c", 1.0)]]
        assert len(tors.score_fuse(lists, k=None)) == 3
        assert [i for i, _ in tors.score_fuse(lists, k=2)] == ["a", "b"]
        assert tors.score_fuse(lists, k=10) == tors.score_fuse(lists, k=None)  # clamps
        assert tors.score_fuse(lists, k=None) == tors.score_fuse(lists)  # the default

    def test_unknown_method_names_the_accepted_set(self) -> None:
        with pytest.raises(ValueError, match="combmnz, borda, linear"):
            tors.score_fuse([[("a", 1.0)]], method="bogus")
        with pytest.raises(ValueError, match="combmnz, borda, linear"):
            tors.score_fuse([[("a", 1.0)]], method="CombMNZ")  # case-sensitive

    def test_unhashable_id_raises_type_error(self) -> None:
        # Python's own hash error: a dict cannot key an id, the same
        # wrong-type-entry contract bm25_rank's corpus walk keeps.
        with pytest.raises(TypeError, match="unhashable"):
            tors.score_fuse([[({"un": "hashable"}, 1.0)]])  # type: ignore[list-item]
        with pytest.raises(TypeError, match="unhashable"):
            tors.score_fuse([[("a", 1.0), (["nested"], 1.0)]])  # type: ignore[list-item]

    def test_empty_inner_list_is_legal_and_contributes_no_votes(self) -> None:
        lists = [[("a", 1.0), ("b", 0.5)], [], [("c", 2.0)]]
        for method in ("combmnz", "borda", "linear"):
            fused = tors.score_fuse(lists, method=method)
            assert {i for i, _ in fused} == {"a", "b", "c"}, method

    def test_python_equality_semantics_govern_dedup(self) -> None:
        # 1, True, and 1.0 are the same dict/set key, so they are the
        # same id here too: one fused entry, the FIRST spelling's score.
        fused = tors.score_fuse([[(1, 5.0), (True, 1.0), (1.0, 0.5)]])
        assert len(fused) == 1
        assert fused[0][0] == 1
        assert fused[0][0] is not True  # the first occurrence's object
        # The FIRST occurrence's score (5.0) is the vote that stands,
        # and one distinct doc in one list is the zero-range shape:
        # norm 0.5, count 1.
        assert fused[0][1] == 0.5

    def test_duplicate_id_folds_to_its_first_occurrence_per_list(self) -> None:
        # Within one list: the first (id, score) stands, later ones are
        # skipped entirely (they do not re-vote, they do not shape the
        # min-max range). The same id in a DIFFERENT list votes again
        # with its own score.
        once = tors.score_fuse([[("a", 5.0), ("b", 1.0)]], method="linear")
        twice = tors.score_fuse([[("a", 5.0), ("a", 100.0), ("b", 1.0)]], method="linear")
        assert once == twice
        # ...and the folded-away 100.0 did not stretch the range: the
        # norms are still 1.0 / 0.0 (folded), not (5-1)/(100-1).
        assert dict(twice)["a"] == 1.0
        assert dict(twice)["b"] == 0.0

    def test_the_same_id_in_two_lists_votes_twice_with_two_scores(self) -> None:
        lists = [[("a", 10.0), ("f", 0.0)], [("a", 2.0), ("g", 0.0)]]
        fused = dict(tors.score_fuse(lists, method="linear"))
        # L0 range [0, 10]: a norms 1.0. L1 range [0, 2]: a norms 1.0.
        assert fused["a"] == pytest.approx(1.0 + 1.0)


class TestScoreFuseWeights:
    """The weighted extension (the weighted-RRF philosophy carried to
    score space): each list's contribution is multiplied by its weight,
    the vote's SHAPE unchanged (norms for combmnz/linear, rank votes
    for borda). The default (``weights=None``) is the unweighted fusion
    EXACTLY, pinned byte-identical; the domain is strictly positive
    finite floats, `rank_fuse`'s own."""

    _LISTS = [
        [("cat-a", 1.0), ("dog-b", 0.5)],
        [("dog-b", 0.8), ("cat-a", 0.2)],
        [("bird-c", 3.0)],
    ]

    def test_weights_none_is_byte_identical_to_all_ones_and_to_the_unweighted_spelling(
        self,
    ) -> None:
        # The default contract, pinned BIT for BIT: None == all-1.0 ==
        # the unweighted spelling. Not approx: identical. Ints launder.
        plain = tors.score_fuse(self._LISTS)
        for method in ("combmnz", "borda", "linear"):
            plain = tors.score_fuse(self._LISTS, method=method)
            assert tors.score_fuse(self._LISTS, method=method, weights=None) == plain
            assert (
                tors.score_fuse(self._LISTS, method=method, weights=[1.0, 1.0, 1.0]) == plain
            )
            assert tors.score_fuse(self._LISTS, method=method, weights=[1, 1, 1]) == plain

    def test_hand_computed_weighted_combmnz(self) -> None:
        # weights [2.0, 1.0, 1.0]: cat-a's L0 vote doubles (2×1.0), its
        # L1 vote is 1.0 × 0.0 = 0.0; CombSUM 2.0, count 2 -> 4.0.
        # dog-b: 2×0.0 + 1.0×1.0 = 1.0, count 2 -> 2.0.
        fused = tors.score_fuse(self._LISTS, weights=[2.0, 1.0, 1.0])
        assert fused == [("cat-a", 4.0), ("dog-b", 2.0), ("bird-c", 0.5)]

    def test_doubling_a_weight_doubles_that_lists_contribution_exactly(self) -> None:
        # A factor-of-two rescaling is exact in IEEE (no rounding to
        # hide behind), pinned bit for bit on a single-list shape where
        # the whole score IS that list's contribution.
        single = [[("a", 3.0), ("b", 1.0), ("c", 0.0)]]
        for method in ("combmnz", "linear"):
            base = dict(tors.score_fuse(single, method=method, weights=[1.0]))
            doubled = dict(tors.score_fuse(single, method=method, weights=[2.0]))
            for doc in base:
                assert doubled[doc] == base[doc] + base[doc], method

    def test_weights_flip_the_score_trade(self) -> None:
        # The extension's point: cons's two 0.6-norm votes beat top's
        # single norm-1.0 vote at equal weights (1.2 > 1.0), and a
        # 2.0-weighted list flips it back (2.0 > 1.2).
        shape = [
            [("top", 1.0), ("f0", 0.0)],
            [("hi1", 1.0), ("cons", 0.6), ("lo1", 0.0)],
            [("hi2", 1.0), ("cons", 0.6), ("lo2", 0.0)],
        ]
        equal = tors.score_fuse(shape, method="linear", weights=[1.0, 1.0, 1.0])
        assert [i for i, _ in equal][0] == "cons"
        assert equal[0][1] == pytest.approx(0.6 + 0.6)
        boosted = tors.score_fuse(shape, method="linear", weights=[2.0, 1.0, 1.0])
        assert [i for i, _ in boosted][0] == "top"
        assert boosted[0][1] == pytest.approx(2.0)

    @pytest.mark.parametrize("bad", [0.0, -0.5, float("nan"), float("inf")])
    def test_non_positive_or_non_finite_weight_raises_value_error(self, bad: float) -> None:
        with pytest.raises(ValueError, match="weights values must be finite and > 0"):
            tors.score_fuse([[("a", 1.0)], [("b", 1.0)]], weights=[1.0, bad])

    def test_length_mismatch_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="one weight per scored list: got 1 for 2"):
            tors.score_fuse([[("a", 1.0)], [("b", 1.0)]], weights=[1.0])
        with pytest.raises(ValueError, match="one weight per scored list: got 0 for 1"):
            tors.score_fuse([[("a", 1.0)]], weights=[])

    @pytest.mark.parametrize("bad", ["1.0", 1.0, {"w": 1.0}, {1.0, 2.0}])
    def test_non_sequence_weights_raises_type_error(self, bad: object) -> None:
        with pytest.raises(TypeError):
            tors.score_fuse([[("a", 1.0)]], weights=bad)  # type: ignore[arg-type]

    def test_tuple_weights_are_accepted(self) -> None:
        assert tors.score_fuse([[("a", 1.0)], [("b", 1.0)]], weights=(2.0, 1.0)) == (
            tors.score_fuse([[("a", 1.0)], [("b", 1.0)]], weights=[2.0, 1.0])
        )

    def test_weights_validation_precedes_the_pair_walk(self) -> None:
        # A bad weight is refused before any id is hashed: an
        # unhashable id downstream never gets to raise first.
        with pytest.raises(ValueError, match="weights values"):
            tors.score_fuse([[({"un": "hashable"}, 1.0)]], weights=[0.0])  # type: ignore[list-item]


class TestScoreFuseEmission:
    def test_one_pair_per_distinct_id(self) -> None:
        lists = [
            [("a", 1.0), ("b", 0.5), ("a", 9.0)],
            [("b", 0.7), ("c", 0.1)],
        ]
        fused = tors.score_fuse(lists)
        assert len(fused) == 3
        assert {i for i, _ in fused} == {"a", "b", "c"}

    def test_a_borda_last_place_zero_vote_still_emits(self) -> None:
        # Emission is vote existence, not score positivity: the bottom
        # of every list votes exactly 0.0 and still appears, ordered
        # last (score descending), ties by first appearance.
        lists = [
            [("a", 5.0), ("b", 1.0)],
            [("c", 5.0), ("d", 1.0)],
        ]
        fused = tors.score_fuse(lists, method="borda")
        assert [i for i, _ in fused] == ["a", "c", "b", "d"]
        assert fused[2] == ("b", 0.0)
        assert fused[3] == ("d", 0.0)

    def test_a_denormal_weight_underflow_still_emits_every_voted_id(self) -> None:
        # 5e-324 (the smallest subnormal) × any norm ≤ 1 underflows to
        # exactly 0.0 for every entry (0.5 × 5e-324 rounds to even,
        # 0.0): the ids still emit, 0.0-score pairs in first-appearance
        # order -- the rank family's underflow-emission policy.
        lists = [[("a", 1.0), ("b", 1.0), ("c", 1.0)]]
        for method in ("combmnz", "linear"):
            fused = tors.score_fuse(lists, method=method, weights=[5e-324])
            assert [i for i, _ in fused] == ["a", "b", "c"], method
            assert all(s == 0.0 for _, s in fused), method
        # The mixed shape: one subnormal survivor (norm 1.0 × 5e-324
        # rounds back to the subnormal itself), one true 0.0 -- both
        # appear, the survivor first.
        mixed = [[("a", 10.0), ("b", 0.0)]]
        for method in ("combmnz", "linear"):
            fused = tors.score_fuse(mixed, method=method, weights=[5e-324])
            assert fused == [("a", 5e-324), ("b", 0.0)], method

    def test_scores_are_never_nan(self) -> None:
        # The all-nonneg accumulation + saturating norm: finite or
        # +inf (weights up to f64::MAX), NaN unreachable.
        lists = [
            [("a", 1e308), ("b", -1e308)],
            [("a", 1e308), ("c", 1e-308)],
        ]
        fused = tors.score_fuse(lists, weights=[1e308, 1e308])
        assert not any(math.isnan(s) for _, s in fused)


class TestScoreFuseDifferential:
    """The byte-exact differential against the pure-Python oracle
    (tests/reference.py), over every corpus class the contract names:
    duplicate ids, single lists, equal-score lists (the zero-range
    edge), extreme magnitudes, single-element lists."""

    def test_the_docs_example_matches_the_oracle(self) -> None:
        lists = [
            [("cat-a", 1.0), ("dog-b", 0.5)],
            [("dog-b", 0.8), ("cat-a", 0.2)],
            [("bird-c", 3.0)],
        ]
        for method in ("combmnz", "borda", "linear"):
            _assert_matches_oracle(lists, method)
        _assert_matches_oracle(lists, "combmnz", weights=[2.0, 1.0, 1.0])

    @given(scored_lists=_SCORED_LISTS)
    @settings(max_examples=200)
    def test_all_three_methods_match_the_oracle_byte_exact(self, scored_lists: list) -> None:
        for method in ("combmnz", "borda", "linear"):
            _assert_matches_oracle(scored_lists, method)

    @given(scored_lists=_SCORED_LISTS)
    @settings(max_examples=150)
    def test_weights_match_the_oracle_byte_exact(self, scored_lists: list) -> None:
        weights = [float(i + 1) for i in range(len(scored_lists))]
        for method in ("combmnz", "borda", "linear"):
            _assert_matches_oracle(scored_lists, method, weights=weights)

    @given(scored_lists=_SCORED_LISTS, k=st.integers(min_value=1, max_value=30))
    @settings(max_examples=100)
    def test_k_truncation_matches_the_oracle_byte_exact(
        self, scored_lists: list, k: int
    ) -> None:
        for method in ("combmnz", "borda", "linear"):
            _assert_matches_oracle(scored_lists, method, k=k)

    @given(
        ids=st.lists(_IDS, min_size=1, max_size=12),
        score=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_duplicate_id_heavy_corpora_match_the_oracle(self, ids: list, score: float) -> None:
        # Duplicate ids within AND across lists (the fold-first shape),
        # at one shared score so the zero-range edge fires too, plus a
        # short third list at a shifted score.
        lists = [
            [(d, score) for d in ids],
            [(d, score) for d in reversed(ids)],
            [(d, score + 1.0) for d in ids[:2]],
        ]
        for method in ("combmnz", "borda", "linear"):
            _assert_matches_oracle(lists, method)

    @given(
        ids=st.lists(_IDS, min_size=1, max_size=8, unique=True),
        score=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        method=st.sampled_from(("combmnz", "borda", "linear")),
    )
    @settings(max_examples=100)
    def test_single_list_and_single_element_corpora_match_the_oracle(
        self, ids: list, score: float, method: str
    ) -> None:
        # A single list (the pure-normalization shape) and a
        # single-element list (the zero-range shape, borda's zero
        # vote): both oracle-identical.
        _assert_matches_oracle([[ (d, score) for d in ids ]], method)
        _assert_matches_oracle([[(ids[0], score)]], method)

    @given(
        ids=st.lists(_IDS, min_size=2, max_size=8, unique=True),
        method=st.sampled_from(("combmnz", "borda", "linear")),
        extreme=st.sampled_from([1e-300, 1e300, 1e308, -1e308, 0.0, -0.0]),
    )
    @settings(max_examples=150)
    def test_extreme_magnitude_corpora_match_the_oracle(
        self, ids: list, method: str, extreme: float
    ) -> None:
        # Extreme magnitudes (the documented overflow-adjacent class):
        # every entry AT the extreme, plus ordinary scores -- the
        # zero-range, saturating-range, and ordinary-range paths all
        # oracle-identical, no NaN anywhere.
        lists = [
            [(ids[0], extreme), (ids[1], -extreme), (ids[0], 1.0)],
            [(ids[1], extreme), (ids[0], extreme)],
        ]
        fused = tors.score_fuse(lists, method=method)
        assert not any(math.isnan(s) for _, s in fused), method
        _assert_matches_oracle(lists, method)

    @given(scored_lists=_SCORED_LISTS)
    @settings(max_examples=100)
    def test_output_is_a_score_descending_permutation_of_the_distinct_ids(
        self, scored_lists: list
    ) -> None:
        expected = {i for i, _ in itertools.chain.from_iterable(scored_lists)}
        for method in ("combmnz", "borda", "linear"):
            fused = tors.score_fuse(scored_lists, method=method)
            assert {i for i, _ in fused} == expected, method
            assert len(fused) == len(expected), method
            scores = [s for _, s in fused]
            assert scores == sorted(scores, reverse=True), method

    @given(scored_lists=_SCORED_LISTS)
    @settings(max_examples=100)
    def test_deterministic(self, scored_lists: list) -> None:
        for method in ("combmnz", "borda", "linear"):
            assert tors.score_fuse(scored_lists, method=method) == tors.score_fuse(
                scored_lists, method=method
            ), method

    @given(scored_lists=_SCORED_LISTS)
    @settings(max_examples=100)
    def test_weights_none_is_byte_identical_property(self, scored_lists: list) -> None:
        n = len(scored_lists)
        for method in ("combmnz", "borda", "linear"):
            plain = tors.score_fuse(scored_lists, method=method)
            assert tors.score_fuse(scored_lists, method=method, weights=None) == plain
            assert (
                tors.score_fuse(scored_lists, method=method, weights=[1.0] * n) == plain
            ), method

    @given(
        scored_lists=st.lists(
            st.lists(st.tuples(_IDS, _EXTREME_SCORES), min_size=1, max_size=8),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=150)
    def test_extreme_boundary_scores_stay_nan_free_and_ordered(self, scored_lists: list) -> None:
        # The boundary-magnitude corpus (1e±300, ±1e308, ±0.0): the
        # saturating norm and the all-nonneg sums keep every method's
        # output NaN-free and score-descending whatever the mix.
        for method in ("combmnz", "borda", "linear"):
            fused = tors.score_fuse(scored_lists, method=method)
            scores = [s for _, s in fused]
            assert not any(math.isnan(s) for s in scores), method
            assert scores == sorted(scores, reverse=True), method
            assert all(s >= 0.0 for s in scores), method


class TestScoreFuseHostileObjects:
    def test_hash_that_calls_tors_re_entrantly_is_safe(self) -> None:
        # Stateless doctrine: a hostile __hash__ re-entering tors from
        # inside the dedup walk must neither deadlock nor corrupt.
        class Reentrant:
            def __hash__(self) -> int:
                tors.score_fuse([[("nested", 1.0)]])
                tors.rank_fuse([["nested"]])
                return 7

            def __eq__(self, o: object) -> bool:
                return isinstance(o, Reentrant)

        fused = tors.score_fuse([[(Reentrant(), 1.0), ("z", 1.0)]])
        assert [i for i, _ in fused] == [fused[0][0], "z"]

    def test_eq_that_mutates_the_input_during_the_walk_is_contained(self) -> None:
        # A hostile __eq__ clearing the list mid-dedup: the walk must
        # not panic (a Rust panic would abort the process); a clean
        # Python outcome (result or TypeError) is the contract.
        class Evil:
            def __init__(self, target: list) -> None:
                self.target = target

            def __hash__(self) -> int:
                return 42

            def __eq__(self, other: object) -> bool:
                self.target[:] = []
                return True

        target = [("a", 1.0), ("b", 1.0), ("c", 1.0)]
        try:
            fused = tors.score_fuse([[Evil(target), Evil(target), Evil(target)]])
            assert all(math.isfinite(s) for _, s in fused)
        except TypeError:
            pass  # a raise is acceptable; a process abort is not


# ---------------------------------------------------------------------------
# The aio twin: parity + the honest GIL-held caveat
# ---------------------------------------------------------------------------


def _scored_lists_workload(total_entries: int, n_lists: int = 5) -> list:
    # The rank family's deterministic GIL-cell workload, extended to
    # (id, score) pairs (half-distinct shared-pool ids, str objects
    # reused across lists, scores a pure function of the position).
    per_list = total_entries // n_lists
    id_space = total_entries // 2
    return [
        [
            (f"id_{(j * per_list + i) % id_space}", ((i * 37) % 100) / 100.0)
            for i in range(per_list)
        ]
        for j in range(n_lists)
    ]


class TestScoreFuseAioTwins:
    def test_all_three_method_twins_match_their_sync_results_exactly(self) -> None:
        import tors.aio

        lists = _scored_lists_workload(600)

        async def run() -> None:
            for method in ("combmnz", "borda", "linear"):
                sync = tors.score_fuse(lists, method=method)
                assert await tors.aio.score_fuse(lists, method=method) == sync
            assert await tors.aio.score_fuse(
                lists, weights=[2.0, 1.0, 1.0, 1.0, 1.0]
            ) == tors.score_fuse(lists, weights=[2.0, 1.0, 1.0, 1.0, 1.0])
            assert await tors.aio.score_fuse(lists, k=7) == tors.score_fuse(lists, k=7)

        asyncio.run(run())

    @pytest.mark.timing
    def test_score_fuse_aio_under_a_busy_loop_stays_within_the_pinned_budget(
        self,
    ) -> None:
        # The GIL claim through the aio twin while the loop heartbeats:
        # the interpreter-side pair walk (one dict op + one score
        # extraction per entry) is GIL-held BY DESIGN -- the rank
        # family's own caveat, structurally the majority of the wall.
        # The pinned 0.80 budget is the rank_fuse cell's, at the 100k
        # shape (measured worst gaps 20-64ms of 30-87ms walls, ratios
        # 0.52-0.74 on the dev box; the walk + the heavier per-entry
        # pair extraction sit under it with ~1.1x margin at the worst
        # observed window). Pass-on-first-clean over 4 samples.
        lists = _scored_lists_workload(100_000)

        async def measure() -> tuple[float, float]:
            ticks: list[float] = []
            stop = asyncio.Event()

            async def heartbeat() -> None:
                while True:
                    ticks.append(time.monotonic())
                    if stop.is_set():
                        return
                    await asyncio.sleep(0.010)

            task = asyncio.create_task(heartbeat())
            await asyncio.sleep(0)
            started = time.monotonic()
            try:
                await tors.aio.score_fuse(lists)
            finally:
                stop.set()
                await task
            wall = time.monotonic() - started
            worst = max((b - a for a, b in itertools.pairwise(ticks)), default=0.0)
            return worst, wall

        async def run() -> None:
            for _ in range(4):
                worst_gap, wall = await measure()
                if worst_gap / wall <= 0.80:
                    return
            worst_gap, wall = await measure()
            assert worst_gap / wall <= 0.85, (
                f"the aio twin blocked {worst_gap * 1e3:.0f}ms of a "
                f"{wall * 1e3:.0f}ms call ({worst_gap / wall:.0%}): past the "
                "pinned 0.80 band the detach is not holding its documented share"
            )

        asyncio.run(run())

    def test_heavy_hash_ids_recorded_not_pinned(self) -> None:
        # FINDING (recorded, bounded): the rank family's own id-shape
        # caveat applies identically -- hash-expensive ids (10-int
        # tuples) push the GIL-held share toward 1.0; the 0.80 budget
        # is a STRING-ID shape claim. Asserted only as
        # no-worse-than-fully-blocking plus the reranking-scale
        # guidance (the call still finishes).
        ids = [tuple((i * 10 + k) % 1_000_003 for k in range(10)) for i in range(20_000)]
        per_list = len(ids) // 5
        lists = [
            [(ids[(j * per_list + i) % len(ids)], float(i % 100)) for i in range(per_list)]
            for j in range(5)
        ]
        start = time.monotonic()
        fused = tors.score_fuse(lists)
        wall = time.monotonic() - start
        assert len(fused) == len(ids)
        assert wall < 30.0  # the reranking-scale guidance, not a ratio pin


# ---------------------------------------------------------------------------
# The scaling pin lives in tests/test_scaling_pins.py (the dedicated
# file, the rank family's own arrangement); the aio twin's busy-loop
# budget above is the GIL claim's runtime check.
# ---------------------------------------------------------------------------


class TestScoreFuseAioContention:
    def test_repeated_concurrent_fusions_match_sync_bit_for_bit(self) -> None:
        # The fused SCORES (not just the order) must be bit-identical
        # across concurrent rounds: no shared-state drift, no
        # accumulation-order nondeterminism from thread timing.
        import tors.aio

        lists = _scored_lists_workload(3_000, n_lists=8)

        async def run() -> None:
            rounds = await asyncio.gather(
                *[tors.aio.score_fuse(lists, method=m) for m in ("combmnz", "borda", "linear")]
            )
            for method, got in zip(("combmnz", "borda", "linear"), rounds, strict=True):
                assert got == tors.score_fuse(lists, method=method), method

        asyncio.run(run())


def _min_wall_ms(fn, samples: int = 5) -> float:
    fn()
    best = float("inf")
    for _ in range(samples):
        started = time.monotonic()
        fn()
        best = min(best, time.monotonic() - started)
    return best * 1e3
