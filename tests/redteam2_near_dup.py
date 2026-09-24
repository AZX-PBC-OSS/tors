"""Second-wave adversarial review of the near-duplicate family — the attacks
the first wave (``tests/redteam_near_dup.py``) did NOT run, run here:

1. **The magnitude-classification contract** (``simhash_distance``): a full
   truth table over the 2**63 / 2**64 / 2**128 boundary rows, the empty
   fingerprint edge (``simhash64("") == simhash128("") == 0``), the
   legal-128-bit-below-2**64 straddle, and the classification-line identity
   (the line is 2**64, NOT 2**63).
2. **Width-agnostic dedup mixing**: fingerprints never mix widths inside one
   sweep, and the token-free convention is asymmetric across methods —
   ``simhash`` has NO exactly-one-empty case (its token-free fingerprint is
   0 and merges by popcount budget), ``shingle``/``minhash`` do.
3. **Shingle-width extremes**: width 1 vs token counts 0/1/2, width exactly
   the token count (one shingle) vs one past it (empty set), float/bool/str
   ``width`` (the int-slot discipline), and the i64 extraction boundary
   (2**63-1 accepted, 2**63 OverflowError).
4. **The ``>=`` boundary under f64**: a Fraction-based exact-rational oracle
   over the threshold ladder. FINDING W2-1: when a pair's exact Jaccard
   ``p/q`` rounds DOWN to its nearest double and the caller's threshold is
   exactly that double, the sweep MERGES although the exact Jaccard is
   strictly below the threshold — a 1-ulp over-merge, never the under-merge
   (data-loss) direction. Characterized and pinned; the exact-rational
   reading itself is ``xfail(strict)``. The simhash bit-budget map
   ``floor((1-t)*64)`` and the MinHash map ``ceil(t*128)`` are proven
   dust-free (Sterbenz + power-of-two scaling / dyadic denominators).
5. **Groups structure abuse**: transitive-closure attacks on chains of
   length 4+ — groups must stay STAR-shaped (member↔head edges only), a
   text matching only a DROPPED member must never join that member's group,
   plus Hypothesis partition invariants (non-overlap, coverage, head edges,
   size consistency) across methods and thresholds.
6. **Memory/GIL wave 2**: 10k ALL-IDENTICAL texts (one exploding group) and
   concurrent ``aio`` gathers under heartbeat contention.
7. Every number in the api.md near-dup section recomputed.

All tests GREEN by construction except the one ``xfail(strict)`` bug
reference (W2-1). FIX NOTHING here — this module only reports.
"""

from __future__ import annotations

import asyncio
import math
import random
import subprocess
import sys
import unicodedata
from fractions import Fraction
from math import nextafter
from time import monotonic

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors
import tors.aio

_METHODS = ["simhash", "shingle", "minhash"]


def fold(text: str) -> str:
    """The grounding fold (per-char lowercase, then NFC) — same spelling as
    the first wave's oracle, reused so both waves observe one policy."""
    return unicodedata.normalize("NFC", "".join(ch.lower() for ch in text))


def popcount(x: int) -> int:
    return bin(x).count("1")


def triples(text: str) -> set[tuple[str, ...]]:
    """Exact folded token-triple sets (the shingle method's sets, no hash)."""
    toks = [fold(t) for t in text.split()]
    return {tuple(toks[i : i + 3]) for i in range(len(toks) - 2)}


def window_pair(n: int, off: int) -> tuple[str, str]:
    """Two n-token windows offset by ``off`` over distinct words w0..w19:
    their trigram overlap is exactly (n - off - 2) of (n - 2) each."""
    a = " ".join(f"w{i}" for i in range(n))
    b = " ".join(f"w{i}" for i in range(off, off + n))
    return a, b


# ---------------------------------------------------------------------------
# 1. THE MAGNITUDE-CLASSIFICATION TRUTH TABLE (simhash_distance)
# ---------------------------------------------------------------------------


class TestMagnitudeTruthTable:
    """docs/api.md: each argument is classified by magnitude (only a
    128-bit fingerprint can be >= 2**64) and a pair SPLIT ACROSS THE LINE is
    refused; a pair both of whose values fit in 64 bits compares correctly
    either way. The full boundary-row truth table, pinned."""

    # Rows bracketing every classification boundary: the 2**63 line is NOT
    # a classification boundary (both sides are 64-bit class); 2**64 is;
    # 2**128 is the OverflowError edge, 0 the empty-fingerprint edge.
    ROWS = [0, 1, 2**63 - 1, 2**63, 2**63 + 1, 2**64 - 1, 2**64, 2**64 + 1,
            2**65 - 1, 2**127 - 1, 2**127, 2**128 - 1]

    @pytest.mark.parametrize("a", ROWS)
    @pytest.mark.parametrize("b", ROWS)
    def test_truth_table_matches_api_md_exactly(self, a: int, b: int) -> None:
        classify = lambda x: x >= 2**64  # noqa: E731 (the doc's own rule)
        expected: int | type[Exception]
        if classify(a) != classify(b):
            expected = ValueError
        else:
            expected = popcount(a ^ b)
        if expected is ValueError:
            with pytest.raises(ValueError, match="same width"):
                tors.simhash_distance(a, b)
        else:
            assert tors.simhash_distance(a, b) == expected, (a, b)

    def test_the_empty_fingerprint_edge_zero_zero_is_distance_zero(self) -> None:
        # simhash64("") == 0 and simhash128("") == 0: distance(0, 0) must be
        # 0 and must NOT raise (both classify 64-bit — the classification
        # never sees an empty class). Pinned because a "same width" gate
        # implemented as an equality check would refuse this row.
        assert tors.simhash64("") == 0
        assert tors.simhash128("") == 0
        assert tors.simhash_distance(0, 0) == 0

    def test_the_same_texts_two_spellings_compare_through_the_empty_edge(self) -> None:
        # simhash64("") vs simhash128(""): both are 0, both classify 64-bit,
        # so the SAME text's two spellings compare (to 0) — the documented
        # width-blind edge taken to its extreme.
        assert tors.simhash_distance(tors.simhash64(""), tors.simhash128("")) == 0

    def test_legal_128_bit_below_two_pow_64_vs_64_bit_compares(self) -> None:
        # The honest edge: a legal 128-bit fingerprint that happens to fit
        # in 64 bits ([2**63, 2**64)) against a 64-bit one in the same range
        # — both classify 64-bit, the comparison goes through width-blind.
        a, b = 2**63 + 1, 2**64 - 1
        assert tors.simhash_distance(a, b) == popcount(a ^ b)

    def test_straddle_refusal_is_about_the_line_not_the_width_origin(self) -> None:
        # A 128-bit value in [2**64, 2**65) against a real 64-bit value is
        # refused even though the 64-bit side may itself have come from
        # simhash128; and two values straddling 2**63 (NOT a classification
        # line) compare fine. Together: the gate tracks 2**64 only.
        with pytest.raises(ValueError, match="same width"):
            tors.simhash_distance(2**64 + 1, 2**64 - 1)
        assert tors.simhash_distance(2**63, 2**63 - 1) == popcount(2**63 ^ (2**63 - 1))

    def test_self_distance_is_never_refused_at_any_magnitude(self) -> None:
        for row in self.ROWS:
            assert tors.simhash_distance(row, row) == 0

    def test_refusal_condition_is_exactly_the_classification_split(self) -> None:
        # api.md's "catches the real caller bug with probability 1 - 2**-64":
        # the mechanics are that the refusal fires IFF the two magnitudes
        # land on opposite sides of 2**64 — verified over a random scan
        # (the probability claim itself assumes fingerprint uniformity, a
        # uniformity simhash's own docs caveat — report-only note).
        rng = random.Random(20260924)
        for _ in range(300):
            x, y = rng.getrandbits(128), rng.getrandbits(128)
            refused = (x >= 2**64) != (y >= 2**64)
            if refused:
                with pytest.raises(ValueError):
                    tors.simhash_distance(x, y)
            else:
                assert tors.simhash_distance(x, y) == popcount(x ^ y)


# ---------------------------------------------------------------------------
# 2. WIDTH-AGNOSTIC DEDUP MIXING + THE TOKEN-FREE CONVENTION'S ASYMMETRY
# ---------------------------------------------------------------------------


class TestDedupWidthMixingAndTokenFree:
    def test_no_width_classification_exists_inside_the_dedup_sweep(self) -> None:
        # dedup fingerprints internally with the u64 simhash only — the
        # 2**64 magnitude line (and any width mixing) cannot arise: texts
        # whose fingerprints cover BOTH halves of the u64 range sweep
        # together without the ValueError the pair surface would raise.
        rng = random.Random(5)
        vocab = "alpha beta gamma delta epsilon zeta eta".split()
        # Short-text simhash fingerprints are bit-biased (each bit is a
        # majority vote over few token hashes), so a BOTH-HALVES corpus
        # needs a few hundred draws over this 7-word vocabulary.
        corpus = [" ".join(rng.choice(vocab) for _ in range(6)) for _ in range(400)]
        assert any(tors.simhash64(t) < 2**63 for t in corpus)
        assert any(tors.simhash64(t) >= 2**63 for t in corpus)
        for method in _METHODS:
            out = tors.dedup_near_dup(corpus, threshold=0.9, method=method)
            assert sorted(out["kept"] + out["dropped"]) == list(range(400))

    @pytest.mark.parametrize("method", _METHODS)
    def test_token_free_family_groups_together_and_not_with_real_text(self, method: str) -> None:
        texts = ["", "   ", "\t\n", "real text here"]
        out = tors.dedup_near_dup(texts, threshold=0.9, method=method)
        assert out["groups"] == [[0, 1, 2], [3]], (method, out)

    def test_simhash_method_has_no_exactly_one_empty_case(self) -> None:
        # FINDING W2-2 (P2, pinned as actual): the empty-set convention's
        # "exactly one empty side scores 0.0" is a SHINGLE/MINHASH special
        # case only. The simhash method fingerprints "" as 0 and merges it
        # with a real text whenever popcount(fp(text)) fits the bit budget —
        # demonstrable at a lowered threshold, unreachable in practice at
        # 0.9 (the measured margin below).
        text = "the quick brown fox jumps over the lazy dog"
        p = popcount(tors.simhash64(text))
        threshold = 1.0 - (p + 0.5) / 64  # bit budget exactly p
        for method, expect in [("simhash", [0]), ("shingle", [0, 1]), ("minhash", [0, 1])]:
            out = tors.dedup_near_dup(["", text], threshold=threshold, method=method)
            assert out["kept"] == expect, (method, out)

    def test_measured_margin_no_small_text_fingerprints_within_the_09_budget(self) -> None:
        # The practical safety of the row above: over a large sample of
        # short texts, no simhash64 fingerprint is within the threshold
        # 0.9 budget (6 bits) of the empty fingerprint — sampled evidence,
        # not a proof (the convention asymmetry is the documented point).
        rng = random.Random(11)
        vocab = [f"w{i}" for i in range(60)]
        for _ in range(20_000):
            text = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 6)))
            assert popcount(tors.simhash64(fold(text))) > 6, text

    def test_fewer_tokens_than_the_width_is_token_free_only_for_the_shingle_method(self) -> None:
        # A 2-token text has an EMPTY trigram set (the empty-set convention)
        # but a REAL simhash fingerprint — so it joins the token-free family
        # under the shingle method and stands alone under simhash.
        two_token, real = "alpha beta", "gamma delta epsilon zeta eta theta"
        sh = tors.dedup_near_dup(["", "  ", two_token, real], threshold=0.9, method="shingle")
        assert sh["groups"][0] == [0, 1, 2], sh
        si = tors.dedup_near_dup(["", "  ", two_token, real], threshold=0.9, method="simhash")
        assert si["groups"][0] == [0, 1], si


# ---------------------------------------------------------------------------
# 3. SHINGLE-WIDTH EXTREMES, WAVE 2
# ---------------------------------------------------------------------------


class TestShingleWidthExtremesWave2:
    def test_width_1_against_token_counts_0_1_and_2(self) -> None:
        # width=1: singleton-token sets. 0-token text -> empty set (the
        # convention), 1-token texts compare as singletons.
        assert tors.shingle_jaccard("", "tok", width=1) == 0.0
        assert tors.shingle_jaccard("tok", "tok", width=1) == 1.0
        assert tors.shingle_jaccard("tok", "other", width=1) == 0.0
        assert tors.shingle_jaccard("tok tok", "tok other", width=1) == 0.5
        assert tors.shingle_dice("tok tok", "tok other", width=1) == 2 / 3

    def test_width_equal_to_token_count_is_one_shingle_not_empty(self) -> None:
        # 3 tokens at width 3: exactly ONE shingle. One past: EMPTY set.
        # The pair rows flip between 1.0/0.0 and the both-empty convention.
        assert tors.shingle_jaccard("a b c", "a b c", width=3) == 1.0
        assert tors.shingle_jaccard("a b c", "a b x", width=3) == 0.0
        assert tors.shingle_jaccard("a b c", "a b c", width=4) == 1.0  # both empty
        assert tors.shingle_jaccard("a b c", "a b x", width=4) == 1.0  # both empty
        assert tors.shingle_jaccard("a b c", "a b c d", width=3) == 0.5
    def test_float_and_str_width_are_type_errors_the_int_slot_discipline(self) -> None:
        # The __index__ discipline: a float has no __index__ -> TypeError in
        # both functions; a str likewise; bool is rejected even though it
        # HAS __index__.
        for bad in (3.0, 3.5, float("nan"), float("inf"), "3", None, True, False):
            for fn in (tors.shingle_jaccard, tors.shingle_dice):
                with pytest.raises(TypeError):
                    fn("a b c", "a b c", width=bad)  # type: ignore[arg-type]

    def test_i64_extraction_boundary_2_pow_63_minus_1_vs_2_pow_63(self) -> None:
        # 2**63-1 is inside i64: accepted, and on short texts every side is
        # empty (the 1.0 convention). 2**63 is one past i64: OverflowError
        # from the extraction (NOT a ValueError — the boundary is the int
        # width, not the >= 1 rule).
        assert tors.shingle_jaccard("a b c", "x y z", width=2**63 - 1) == 1.0
        assert tors.shingle_dice("a b c", "x y z", width=2**63 - 1) == 1.0
        with pytest.raises(OverflowError):
            tors.shingle_jaccard("a b c", "a b c", width=2**63)
        with pytest.raises(OverflowError):
            tors.shingle_dice("a b c", "a b c", width=2**63)

    def test_width_1_with_a_huge_single_token_completes(self) -> None:
        token = "x" * 100_000
        assert tors.shingle_jaccard(token, token, width=1) == 1.0
        assert tors.shingle_dice(token, token, width=1) == 1.0


# ---------------------------------------------------------------------------
# 4. THE >= BOUNDARY UNDER F64 vs THE FRACTION ORACLE (FINDING W2-1)
# ---------------------------------------------------------------------------


class TestGeqBoundaryUnderF64:
    """docs/api.md: the shingle method merges when 'the EXACT Jaccard index
    ... is at least threshold'. The score is an f64 division and the
    comparison is f64 — FINDING W2-1: for a pair whose exact Jaccard p/q
    rounds DOWN to fl(p/q) < p/q ... no: the divergence fires when the
    rounding is UP (fl(J) > J) and the threshold IS fl(J): the core merges
    though the exact Jaccard is strictly below the threshold. Over-merge by
    at most 1 ulp; the under-merge (data-loss) direction NEVER fires, so
    this is a doc-limitation / fix-policy question, not a correctness bug.
    Characterized green below; the exact-rational reading is xfail(strict)."""

    # Window pairs with EXACT trigram Jaccards 3/13, 5/13, 5/7 — every one
    # rounds UP to its nearest double (verified in the characterization).
    DUST_PAIRS = [
        (3, 13, window_pair(10, 5)),
        (5, 13, window_pair(11, 4)),
        (5, 7, window_pair(8, 1)),
    ]

    def test_dust_pairs_have_the_exact_fraction_and_round_up(self) -> None:
        for p, q, (a, b) in self.DUST_PAIRS:
            sa, sb = triples(a), triples(b)
            inter, union = len(sa & sb), len(sa | sb)
            assert (inter, union) == (p, q), (p, q, inter, union)
            assert Fraction(p / q) > Fraction(p, q), (p, q)

    @pytest.mark.parametrize("p,q,pair", [(3, 13, 0), (5, 13, 1), (5, 7, 2)])
    def test_w2_1_over_merge_cell_characterized(self, p: int, q: int, pair: int) -> None:
        # The actual contract at the dust cell: threshold == fl(J) == the
        # score itself, so the f64 >= merges — while the exact Jaccard is
        # strictly below the threshold's rational value. PINNED AS ACTUAL
        # (with the shingle core, the dedup sweep, and the direct score all
        # agreeing with each other — the gap is to the DOC's exact reading).
        _, _, (a, b) = self.DUST_PAIRS[pair]
        tau = p / q
        assert Fraction(p, q) < Fraction(tau)
        assert tors.shingle_jaccard(a, b, width=3) == tau
        assert tors.dedup_near_dup([a, b], threshold=tau, method="shingle")["kept"] == [0]

    @pytest.mark.xfail(strict=True, reason="W2-1: 1-ulp over-merge at tau == fl(J) "
                                           "(round-up); exact-rational semantics refused")
    @pytest.mark.parametrize("p,q,pair", [(3, 13, 0), (5, 13, 1), (5, 7, 2)])
    def test_exact_rational_boundary_reading(self, p: int, q: int, pair: int) -> None:
        _, _, (a, b) = self.DUST_PAIRS[pair]
        # Under the doc's exact reading, J = p/q < float(p/q) must NOT merge.
        assert tors.dedup_near_dup([a, b], threshold=p / q, method="shingle")["kept"] == [0, 1]

    @given(n=st.integers(6, 16), off=st.integers(1, 13), variant=st.integers(0, 2))
    @settings(max_examples=120, deadline=None)
    def test_no_under_merge_and_all_divergences_are_1_ulp_over_merges(
        self, n: int, off: int, variant: int
    ) -> None:
        off = min(off, n - 3)
        a, b = window_pair(n, off)
        if variant == 1:
            b = b + " w21"  # widen the union, same shared head
        elif variant == 2:
            a, b = b, a
        sa, sb = triples(a), triples(b)
        inter, union = len(sa & sb), len(sa | sb)
        if union == 0:
            return
        j_exact = Fraction(inter, union)
        for tau in (inter / union, nextafter(inter / union, 0.0), nextafter(inter / union, 2.0)):
            exact = j_exact >= Fraction(tau)
            merged = tors.dedup_near_dup([a, b], threshold=tau, method="shingle")["kept"] == [0]
            if exact:
                assert merged, f"UNDER-merge at J={inter}/{union}, tau={tau!r}"
            elif merged:
                # The only admissible divergence: tau is exactly fl(J) and
                # fl(J) rounds UP — a 1-ulp over-merge, never more.
                assert tau == inter / union, (inter, union, tau)
                assert Fraction(tau) > j_exact

    def test_simhash_bit_budget_map_is_dust_free(self) -> None:
        # floor((1-t)*64) must equal the exact rational floor. For t >= 0.5
        # this is PROVABLE (Sterbenz: 1-t is exact; *64 is exact scaling) —
        # pinned over a dense ladder; below 0.5, sampled.
        for i in range(100_000, 200_001):
            t = i / 200_000
            assert math.floor((1.0 - t) * 64) == math.floor((1 - Fraction(t)) * 64), t
        rng = random.Random(3)
        for _ in range(20_000):
            t = rng.random()
            assert math.floor((1.0 - t) * 64) == math.floor((1 - Fraction(t)) * 64), t

    def test_minhash_needed_map_is_dust_free(self) -> None:
        # ceil(t*128) equals the exact ceil: t = m/128 is dyadic (exact in
        # f64, product exact), and non-dyadic t keeps 128*t away from
        # integers by more than the rounding error.
        for i in range(0, 200_001):
            t = i / 200_000
            assert math.ceil(t * 128) == math.ceil(Fraction(t) * 128), t


# ---------------------------------------------------------------------------
# 5. GROUPS STRUCTURE ABUSE — STAR SHAPE, NOT TRANSITIVE CLOSURE
# ---------------------------------------------------------------------------


class TestGroupsStarShape:
    def test_transitive_closure_attack_chain_of_four(self) -> None:
        # A chain of 4: adjacent windows share ~41% of trigrams, skip-one
        # ~9%. Transitive closure would put all four in ONE group; the
        # greedy sweep must keep two heads and stay star-shaped.
        texts = [" ".join(f"w{i}" for i in range(o, o + 14)) for o in (0, 5, 10, 15)]
        assert tors.shingle_jaccard(texts[0], texts[1], width=3) >= 0.4
        assert tors.shingle_jaccard(texts[1], texts[2], width=3) >= 0.4
        assert tors.shingle_jaccard(texts[2], texts[3], width=3) >= 0.4
        assert tors.shingle_jaccard(texts[0], texts[2], width=3) < 0.4
        assert tors.shingle_jaccard(texts[0], texts[3], width=3) < 0.4
        # NOT [[0, 1, 2, 3]]: a~b, b~c, c~d does not transit.
        assert tors.dedup_near_dup(texts, threshold=0.4, method="shingle") == {
            "kept": [0, 2],
            "dropped": [1, 3],
            "groups": [[0, 1], [2, 3]],
        }

    def test_duplicate_of_a_dropped_member_never_joins_that_members_group(self) -> None:
        # Geometry: t0 = a-family (12 tokens), t1 = a-family + b-family
        # (matches t0 at 10/22), w = the b-family alone (matches t1 at
        # 10/22 but NOT t0). w must become its OWN head, not join group 0:
        # membership is member<->HEAD only, never member<->dropped-member.
        a = " ".join(f"a{i}" for i in range(12))
        b = " ".join(f"b{i}" for i in range(12))
        t0, t1, w = a, f"{a} {b}", b
        assert tors.shingle_jaccard(t0, t1, width=3) >= 0.4
        assert tors.shingle_jaccard(t1, w, width=3) >= 0.4
        assert tors.shingle_jaccard(t0, w, width=3) < 0.4
        assert tors.dedup_near_dup([t0, t1, w], threshold=0.4, method="shingle") == {
            "kept": [0, 2],
            "dropped": [1],
            "groups": [[0, 1], [2]],
        }

    @given(texts=st.lists(st.text(min_size=0, max_size=24), min_size=1, max_size=9))
    @settings(max_examples=40, deadline=None)
    def test_partition_is_star_shaped_across_methods_and_thresholds(self, texts: list[str]) -> None:
        for method in _METHODS:
            for threshold in (0.0, 0.6, 1.0):
                out = tors.dedup_near_dup(texts, threshold=threshold, method=method)
                n = len(texts)
                kept, dropped, groups = out["kept"], out["dropped"], out["groups"]
                assert sorted(kept + dropped) == list(range(n))
                assert sorted(i for g in groups for i in g) == list(range(n))  # disjoint cover
                assert kept == [g[0] for g in groups]  # heads are the representatives
                assert kept == sorted(kept)  # representative order
                assert len(groups) == len(kept)
                assert sum(len(g) - 1 for g in groups) == len(dropped)  # size consistency
                head_of = {i: g[0] for g in groups for i in g}
                for i in dropped:
                    assert head_of[i] < i  # claimed by an EARLIER text only
                    # STAR SHAPE: the member matches its OWN head ...
                    assert self._pair_dup(texts[i], texts[head_of[i]], threshold, method)
                # ... and no group is a clique requirement: members may be
                # far apart (checked on the chains above), but a head must
                # never claim a text it does not match (the loop above is
                # that check, run through the independent pair oracle).

    @staticmethod
    def _pair_dup(a: str, b: str, threshold: float, method: str) -> bool:
        if method == "simhash":
            return popcount(tors.simhash64(fold(a)) ^ tors.simhash64(fold(b))) <= math.floor(
                (1.0 - threshold) * 64
            )
        if method == "shingle":
            sa, sb = triples(a), triples(b)
            if not sa and not sb:
                return True
            if not sa or not sb:
                return threshold == 0.0
            return len(sa & sb) / len(sa | sb) >= threshold
        sigs = [
            tors.minhash_signature(fold(x), num_perm=128, shingle_size=3, seed=0) for x in (a, b)
        ]
        return sum(x == y for x, y in zip(*sigs, strict=True)) / 128 >= threshold


# ---------------------------------------------------------------------------
# 6. MEMORY / AIO WAVE 2
# ---------------------------------------------------------------------------


class TestMemoryAndAioWave2:
    def test_ten_k_all_identical_one_exploding_group_oom_shape(self) -> None:
        # The other n=10k extreme (wave 1 pinned the all-distinct corpus):
        # EVERY text joins ONE group — the group list is one 10k-element
        # index list, the sweep early-exits per text — and the peak stays
        # tied to the input, nowhere near an n² allocation.
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "def vmhwm():\n"
                    "    for line in open('/proc/self/status'):\n"
                    "        if line.startswith('VmHWM:'):\n"
                    "            return int(line.split()[1])\n"
                    "    raise SystemExit('no VmHWM')\n"
                    "import tors\n"
                    "n = int(sys.argv[1])\n"
                    "corpus = ['alpha beta gamma delta epsilon'] * n\n"
                    "for method in ('simhash', 'shingle', 'minhash'):\n"
                    "    out = tors.dedup_near_dup(corpus, threshold=0.9, method=method)\n"
                    "    assert out['groups'] == [list(range(n))], method\n"
                    "print(vmhwm())\n"
                ),
                "10000",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert child.returncode == 0, child.stderr
        assert int(child.stdout.strip()) < 200 * 1024, child.stdout

    def test_aio_gather_contention_agrees_with_sync_and_stays_responsive(self) -> None:
        identical = ["alpha beta gamma delta epsilon"] * 3_000
        distinct = [" ".join(f"tok{i}_{j}" for j in range(20)) for i in range(1_000)]

        async def run() -> None:
            gaps: list[float] = []
            last = monotonic()

            async def heartbeat() -> None:
                nonlocal last
                while True:
                    await asyncio.sleep(0.01)
                    now = monotonic()
                    gaps.append(now - last)
                    last = now

            hb = asyncio.create_task(heartbeat())
            try:
                results = await asyncio.gather(
                    tors.aio.dedup_near_dup(identical, threshold=0.9, method="simhash"),
                    tors.aio.dedup_near_dup(distinct, threshold=0.9, method="minhash"),
                    tors.aio.dedup_near_dup(identical, threshold=0.9, method="shingle"),
                    tors.aio.dedup_near_dup(distinct, threshold=0.9, method="simhash"),
                )
            finally:
                hb.cancel()
                try:
                    await hb
                except asyncio.CancelledError:
                    pass
            assert results[0] == tors.dedup_near_dup(identical, threshold=0.9, method="simhash")
            assert results[1] == tors.dedup_near_dup(distinct, threshold=0.9, method="minhash")
            assert results[2] == tors.dedup_near_dup(identical, threshold=0.9, method="shingle")
            assert results[3] == tors.dedup_near_dup(distinct, threshold=0.9, method="simhash")
            assert max(gaps) < 1.0, f"loop stalled {max(gaps) * 1e3:.0f}ms under gather contention"

        asyncio.run(run())


# ---------------------------------------------------------------------------
# 7. DOC NUMBERS — every api.md figure in the near-dup sections recomputed
# ---------------------------------------------------------------------------


class TestDocNumbersWave2:
    def test_simhash_distance_examples(self) -> None:
        a = tors.simhash64("the quick brown fox jumps over the lazy dog")
        b = tors.simhash64("the quick brown fox jumps over the lazy cat")
        assert tors.simhash_distance(a, b) == 9
        assert tors.simhash_distance(a, a) == 0

    def test_shingle_examples(self) -> None:
        assert (
            tors.shingle_jaccard(
                "The quarterly oil sample interval for field outages was adjusted",
                "the QUARTERLY oil sample interval for field outages was adjusted",
            )
            == 1.0
        )
        left, right = "alpha beta gamma delta epsilon zeta", "alpha beta gamma eta theta iota"
        assert tors.shingle_jaccard(left, right) == 0.14285714285714285
        assert tors.shingle_dice(left, right) == 0.25
        assert tors.shingle_jaccard("", "alpha beta gamma") == 0.0
        assert tors.shingle_jaccard("", "") == 1.0

    def test_dedup_examples(self) -> None:
        corpus = [
            "The quarterly oil sample interval was adjusted after the audit.",
            "the quarterly oil sample interval was adjusted after the audit",
            "The quarterly oil sample interval was extended after the audit.",
            "A completely different memo about the parking garage resurfacing.",
        ]
        assert tors.dedup_near_dup(corpus, threshold=0.8, method="simhash") == {
            "kept": [0, 3],
            "dropped": [1, 2],
            "groups": [[0, 1, 2], [3]],
        }
        assert tors.dedup_near_dup(corpus, threshold=0.8, method="shingle") == {
            "kept": [0, 2, 3],
            "dropped": [1],
            "groups": [[0, 1], [2], [3]],
        }
        assert tors.dedup_near_dup([]) == {"kept": [], "dropped": [], "groups": []}

    def test_doc_probability_claim_is_uniformity_conditional(self) -> None:
        # api.md: the magnitude gate "catches the real caller bug ... with
        # probability 1 - 2**-64". The MISS event is a genuine simhash128
        # value below 2**64 — probability 2**-64 only under fingerprint
        # uniformity, which simhash's own limitations section caveats
        # (short text). The empty fingerprint is a DETERMINISTIC miss:
        # simhash128("") == 0 passes as a 64-bit value. Report-only
        # (W2-3): the claim could name the caveat; pinned here so the
        # deterministic-miss row is at least on the record.
        assert tors.simhash128("") < 2**64
        assert tors.simhash_distance(tors.simhash128(""), 0) == 0  # not refused
