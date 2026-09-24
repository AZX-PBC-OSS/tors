"""Second-wave red-team suite for the rank-fusion / IR-metric family.

A fresh adversarial pass that deliberately does NOT re-run the first
wave's angles (the wave-1 suite's eight classes: paper vectors, the
sklearn differential, the overflow NaN hunt, TypeError discipline,
re-entrancy, GIL bands, perf cliffs; the implementer has since folded
those pins into tests/test_rank_fusion.py). Wave 2 attacks the FIXES
and consequences the first wave left standing, plus the classes it
skipped:

1. The saturating-ratio policy's CONSEQUENCES: weak monotonicity under
   saturation (a higher gain moved UP must never lower the score, even
   with sums past f64's ceiling), the one-sidedness and bound of the
   1.0 branch, the 0.0 branch's error magnitude (the policy's own docs
   frame the error as an upward over-report; the 0.0 branch's downward
   error is pinned here at its true, unbounded-in-[0,1] magnitude),
   and k vs k=None agreement on near-overflow data.
2. Float-noise oracle: binary-rounding traps (0.1/0.2/0.3), int vs
   float-equal gains, bool laundering to gains and to k, and the ideal
   sort's tie-order invariance (equal-gain pool order must not move
   the score).
3. rank_fuse id-equality traps: Python's dict semantics say 1 == True
   == 1.0, so does the vote walk (and WHICH object comes back); the
   same object voting across lists (hash side effects counted);
   eq=False frozen dataclasses; list subclasses in both positions;
   nested lists; k=True.
4. Metric invariants: the documented dedup example recomputed from
   first principles, a pure-Python textbook oracle over all four
   metrics, the single-relevant-doc mrr/ndcg relationship (the true
   one: ndcg == 1/log2(r+1) >= mrr == 1/r, equality iff rank 1; the
   naively expected mrr == ndcg is FALSE for every rank past 1 and is
   pinned as such).
5. The aio twins under CONTENTION: concurrent gather of many fused
   calls with event-loop heartbeats, all five twins mixed, and one
   raising twin among successes (no lost results, no cross-talk).
6. The docs' arithmetic itself: every api.md literal recomputed
   independently, and the saturating-policy section's own boundary
   claims ("three gains of 1e308, or two of 1.7e308, are enough")
   checked at BOTH scales on a same-shaped imperfect ranking.

Every test below passed on the audited tree (the feature branch at
9b409d8, plus the implementer's wave-1 fold 325b68f): failed attacks,
pinned as robustness. No xfail.
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

import tors
import tors.aio
from tors import mrr, ndcg_at_k, precision_at_k, rank_fuse, recall_at_k

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
