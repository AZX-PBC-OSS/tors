"""Contract gate for the MinHash banding family: ``tors.lsh_candidates``,
``tors.lsh_probability``, and ``tors.lsh_threshold`` — the near-duplicate
candidate generator that turns `minhash_signature` output into candidate
pairs in one stateless pass (no persistent table, docs/design.md's scope
cut). See ``src/lsh_impl.rs`` for the algorithm writeup (Broder,
Glassman, Manasse, and Zweig, "Syntactic Clustering of the Web", WWW
1997; the banding S-curve in Leskovec, Rajaraman, and Ullman, "Mining of
Massive Datasets", ch. 3) and the false-positive contract (band-hash
collisions are false-positive candidates by design, recall-biased); this
module is the Python-visible half of the same contract.

Pins unique to the banding contract: identical signatures are ALWAYS
candidates (P = 1); maximally different signatures are candidates only
at the documented false-positive rate (a statistical pin at a fixed
seed, with a generous margin, since the band hash is deterministic and
the rate is ~k²/2^65); the S-curve against a hand-computed value AND
against exact ``fractions.Fraction`` arithmetic at sample points;
monotonicity in ``s``; pair determinism under input permutation (the
same pair SET, relabeled).

The scaling pins (linear in n·b) live in ``tests/test_scaling_pins.py``,
the GIL-release cell in ``tests/test_gil_release.py``, the peak-memory
guard here.
"""

from __future__ import annotations

import random
import struct
import subprocess
import sys
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors
import tors.aio

# The independent oracle's band-key hash: the INDEPENDENT xxhash package
# over the length-prefixed little-endian frame (LE64 row count, then LE64
# per row), the frame src/lsh_impl.rs documents — computed here without
# touching the extension's own hashing path.


def band_key(rows: list[int] | tuple[int, ...]) -> int:
    import xxhash

    frame = struct.pack("<Q", len(rows))
    for row in rows:
        frame += struct.pack("<Q", row)
    return xxhash.xxh64_intdigest(frame)


def oracle_candidates(
    signatures: list[list[int]], bands: int, rows: int
) -> list[tuple[int, int]]:
    """Naive banding in pure Python: bucket each band's keys, emit all
    within-bucket pairs, sort-dedup. Independent of the core except for
    the frame format itself (which the unit tests pin separately)."""
    pairs: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[int, list[int]] = {}
        for idx, sig in enumerate(signatures):
            key = band_key(sig[band * rows : (band + 1) * rows])
            buckets.setdefault(key, []).append(idx)
        for members in buckets.values():
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    i, j = members[a], members[b]
                    pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def random_signatures(rng: random.Random, n: int, num_perm: int) -> list[list[int]]:
    """Full-range u64 signatures (the shape minhash_signature returns,
    with values the affine range cannot distinguish from arbitrary)."""
    return [[rng.getrandbits(64) for _ in range(num_perm)] for _ in range(n)]


class TestLshCandidates:
    def test_identical_signatures_are_always_candidates(self) -> None:
        # Identical signatures collide in EVERY band, so every pair is a
        # candidate, at every admissible shape (P = 1, the S-curve's
        # s = 1 end, pinned as behavior).
        sigs = [[7] * 128 for _ in range(5)]
        for bands, rows in [(16, 8), (1, 128), (128, 1), (4, 32)]:
            out = tors.lsh_candidates(sigs, bands=bands, rows=rows)
            assert out["pairs"] == [
                (i, j) for i in range(5) for j in range(i + 1, 5)
            ], (bands, rows)

    def test_one_shared_band_makes_a_candidate(self) -> None:
        # Two signatures agreeing on exactly ONE band's rows: that band's
        # frames are identical, so the pair is a candidate regardless of
        # the other bands (the recall side of the contract, exact).
        a = [0] * 32
        b = [1] * 32
        for row in range(8):
            a[row] = 42
            b[row] = 42
        assert tors.lsh_candidates([a, b], bands=4, rows=8)["pairs"] == [(0, 1)]

    def test_disjoint_signatures_are_candidates_only_at_the_false_positive_rate(
        self,
    ) -> None:
        # The false-positive contract's statistical pin: signatures that
        # share NOTHING (fully disjoint u64 rows, distinct in every band)
        # can be candidates only through a 64-bit band-key collision, at
        # ~k^2/2^65 over k distinct keys. Fixed seed, so the run is
        # deterministic; the margin is generous (a tenth of a percent of
        # all pairs plus an absolute slack) so a hash-implementation
        # change cannot make it flaky, while any real pairing bug blows
        # through it by orders of magnitude.
        rng = random.Random(20260924)
        sigs = random_signatures(rng, 2_000, 128)
        out = tors.lsh_candidates(sigs, bands=32, rows=4)
        observed = len(out["pairs"])
        total = 2_000 * 1_999 // 2
        # The documented rate: ~k^2/2^65 ~ 4e-9 expected candidates here.
        margin = total // 1000 + 100
        assert observed <= margin, f"{observed} candidates over {total} pairs"
        # And for THIS pinned seed the honest value is exactly zero: a
        # collision would be an event worth knowing about, not smoothing
        # over.
        assert observed == 0

    def test_pairs_are_sorted_deduplicated_and_acyclic(self) -> None:
        # Three families plus singles: within-family pairs must appear
        # exactly once (deduplicated across bands), ascending (i, j).
        sigs = [
            [10, 11, 12, 13],
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ]
        out = tors.lsh_candidates(sigs, bands=2, rows=2)
        assert out["pairs"] == [(0, 1), (0, 3), (1, 3), (2, 4)]
        for i, j in out["pairs"]:
            assert i < j

    def test_candidate_relation_survives_input_permutation(self) -> None:
        # The actual contract: permuting the input permutes the SAME pair
        # set (the relation is a function of signature content, not
        # position). The naive spelling re-derived over the permuted
        # input must agree with the permuted core output, and the core
        # must agree with the naive spelling on the original order too.
        rng = random.Random(42)
        sigs = random_signatures(rng, 20, 16)
        # Force families so candidates exist: duplicate some signatures.
        for dup in (3, 7, 11):
            sigs[dup] = list(sigs[0])
        perm = list(range(20))
        rng.shuffle(perm)
        permuted = [sigs[q] for q in perm]  # permuted[p] = sigs[perm[p]]
        relabel = [perm.index(q) for q in range(20)]  # original q -> new p
        direct = tors.lsh_candidates(sigs, bands=4, rows=4)
        moved = tors.lsh_candidates(permuted, bands=4, rows=4)
        expected = sorted(
            {tuple(sorted((relabel[i], relabel[j]))) for i, j in direct["pairs"]}
        )
        assert moved["pairs"] == expected
        assert direct["pairs"] == oracle_candidates(sigs, 4, 4)

    def test_determinism_across_calls_and_fresh_objects(self) -> None:
        rng = random.Random(99)
        sigs = random_signatures(rng, 30, 32)
        fresh = [list(sig) for sig in sigs]
        assert tors.lsh_candidates(sigs, bands=8, rows=4) == tors.lsh_candidates(
            fresh, bands=8, rows=4
        )
        assert tors.lsh_candidates(sigs, bands=8, rows=4) == tors.lsh_candidates(
            sigs, bands=8, rows=4
        )

    def test_empty_and_singleton_inputs(self) -> None:
        assert tors.lsh_candidates([], bands=16, rows=8) == {"pairs": []}
        sig = [[5] * 128]
        assert tors.lsh_candidates(sig, bands=16, rows=8) == {"pairs": []}

    def test_result_is_a_plain_dict_of_int_tuples(self) -> None:
        sigs = [[1, 2], [1, 2]]
        out = tors.lsh_candidates(sigs, bands=1, rows=2)
        assert isinstance(out, dict)
        assert set(out) == {"pairs"}
        assert all(isinstance(p, tuple) and len(p) == 2 for p in out["pairs"])

    def test_minhash_signature_integration_recalls_the_near_pair(self) -> None:
        # The end-to-end shape the API exists for: signatures from
        # minhash_signature at num_perm=bands*rows, the near-duplicate
        # pair (agreement 0.6015625) recalled, the unrelated text not a
        # candidate (beyond the documented false-positive channels).
        original = (
            "The quarterly oil sample interval for field outages was adjusted after the "
            "bushing torque specifications changed. Maintenance windows now close within "
            "fourteen days. "
        )
        edited = (
            "The monthly oil sample interval for field outages was adjusted after the "
            "insulator torque specifications changed. Maintenance windows now close within "
            "fourteen days. "
        )
        unrelated = "Pack my box with five dozen liquor jugs."
        sigs = [tors.minhash_signature(t) for t in (original, edited, unrelated)]
        # P(s = 0.6015625 | b=32, r=4) = 0.988: the near pair is recalled
        # with the S-curve's own probability; the unrelated pair sits at
        # the false-positive floor.
        assert tors.lsh_candidates(sigs, bands=32, rows=4) == {"pairs": [(0, 1)]}

    def test_validation_errors(self) -> None:
        sigs = [[0] * 8]
        for bad_bands in [0, -1, -100]:
            with pytest.raises(ValueError, match="bands must be at least 1"):
                tors.lsh_candidates(sigs, bands=bad_bands, rows=8)
        for bad_rows in [0, -1, -100]:
            with pytest.raises(ValueError, match="rows must be at least 1"):
                tors.lsh_candidates(sigs, bands=8, rows=bad_rows)
        # Length contract: every signature must have exactly bands*rows.
        with pytest.raises(ValueError, match="bands \\* rows"):
            tors.lsh_candidates([[0] * 8, [0] * 7], bands=1, rows=8)
        with pytest.raises(ValueError, match="bands \\* rows"):
            tors.lsh_candidates([[0] * 8], bands=2, rows=8)
        # The signatures argument itself: non-sequence and str refusals.
        for bad in [None, 5, 1.5, object()]:
            with pytest.raises(TypeError):
                tors.lsh_candidates(bad, bands=1, rows=8)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            tors.lsh_candidates("not a list", bands=1, rows=8)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            tors.lsh_candidates([[0, 1], "ab"], bands=1, rows=2)  # type: ignore[list-item]
        # Element typing: exact non-negative ints within 2**64 - 1.
        with pytest.raises(ValueError, match="non-negative"):
            tors.lsh_candidates([[-1, 1]], bands=1, rows=2)
        with pytest.raises(OverflowError):
            tors.lsh_candidates([[2**64, 1]], bands=1, rows=2)
        for bad_element in [True, 1.5, None, "1", b"1"]:
            with pytest.raises(TypeError):
                tors.lsh_candidates([[bad_element, 1]])  # type: ignore[list-item]
        # bool in the bands/rows positions is a TypeError too.
        with pytest.raises(TypeError, match="not bool"):
            tors.lsh_candidates(sigs, bands=True, rows=8)
        with pytest.raises(TypeError, match="not bool"):
            tors.lsh_candidates(sigs, bands=8, rows=False)
        with pytest.raises(TypeError, match="bands must be an int"):
            tors.lsh_candidates(sigs, bands="16", rows=8)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="rows must be an int"):
            tors.lsh_candidates(sigs, bands=16, rows=8.0)  # type: ignore[arg-type]

    def test_bands_and_rows_ride_the_index_protocol(self) -> None:
        class Indexable:
            def __init__(self, value: int) -> None:
                self.value = value

            def __index__(self) -> int:
                return self.value

        sigs = [[1, 2], [1, 2]]
        out = tors.lsh_candidates(sigs, bands=Indexable(1), rows=Indexable(2))
        assert out["pairs"] == [(0, 1)]

    def test_huge_shape_product_is_refused_not_garbled(self) -> None:
        with pytest.raises(ValueError):
            tors.lsh_candidates([], bands=2**62, rows=8)

    @given(
        sigs=st.lists(
            st.lists(st.integers(min_value=0, max_value=2**64 - 1), min_size=4, max_size=4),
            min_size=0,
            max_size=10,
        )
    )
    @settings(max_examples=100)
    def test_property_agrees_with_the_naive_oracle(self, sigs: list[list[int]]) -> None:
        # The differential pin over arbitrary inputs: the core's pairs
        # equal the naive pure-Python banding, exactly, at two shapes.
        assert tors.lsh_candidates(sigs, bands=2, rows=2)["pairs"] == oracle_candidates(
            sigs, 2, 2
        )
        assert tors.lsh_candidates(sigs, bands=1, rows=4)["pairs"] == oracle_candidates(
            sigs, 1, 4
        )

    @given(
        sigs=st.lists(
            st.lists(st.integers(min_value=0, max_value=2**64 - 1), min_size=4, max_size=4),
            min_size=2,
            max_size=8,
        )
    )
    @settings(max_examples=100)
    def test_property_structure(self, sigs: list[list[int]]) -> None:
        out = tors.lsh_candidates(sigs, bands=2, rows=2)
        pairs = out["pairs"]
        # Sorted ascending, i < j, no duplicates.
        assert pairs == sorted(set(pairs))
        for i, j in pairs:
            assert i < j
        # Determinism.
        assert out == tors.lsh_candidates(sigs, bands=2, rows=2)
        # Identical signatures are always paired (P = 1).
        for a in range(len(sigs)):
            for b in range(a + 1, len(sigs)):
                if sigs[a] == sigs[b]:
                    assert (a, b) in pairs

    @given(
        data=st.lists(st.integers(min_value=0, max_value=2**64 - 1), min_size=6, max_size=6)
    )
    @settings(max_examples=50)
    def test_property_permutation_invariance(self, data: list[int]) -> None:
        # Six signatures at three distinct content values (0, 1, 2 index
        # into the data): whatever pairs the core finds on the original
        # order must reappear relabeled under any permutation.
        sigs = [[data[0]] * 4, [data[1]] * 4, [data[2]] * 4, [data[0]] * 4, [data[1]] * 4]
        direct = set(tors.lsh_candidates(sigs, bands=2, rows=2)["pairs"])
        order = [4, 1, 3, 0, 2]
        permuted = [sigs[q] for q in order]
        relabel = [order.index(q) for q in range(5)]
        moved = set(tors.lsh_candidates(permuted, bands=2, rows=2)["pairs"])
        expected = {tuple(sorted((relabel[i], relabel[j]))) for i, j in direct}
        assert moved == expected


class TestLshProbability:
    def test_hand_computed_value(self) -> None:
        # 1 - (1 - (1/2)^4)^16 = 1 - (15/16)^16, computed by hand.
        assert tors.lsh_probability(0.5, bands=16, rows=4) == pytest.approx(
            0.6439258695482072, rel=1e-12
        )
        # The ends are exact.
        assert tors.lsh_probability(0.0, bands=16, rows=8) == 0.0
        assert tors.lsh_probability(1.0, bands=16, rows=8) == 1.0
        # b = 1 collapses to the single-band vote s^r.
        assert tors.lsh_probability(0.5, bands=1, rows=2) == 0.25

    @pytest.mark.parametrize(
        ("s", "bands", "rows"),
        [
            (Fraction(1, 2), 16, 4),
            (Fraction(1, 4), 8, 2),
            (Fraction(3, 4), 32, 4),
            (Fraction(1, 10), 128, 1),
        ],
    )
    def test_matches_exact_fraction_arithmetic(
        self, s: Fraction, bands: int, rows: int
    ) -> None:
        # The S-curve against EXACT rational arithmetic: compute
        # 1 - (1 - s^r)^b over Fractions (no float error at all), then
        # compare the f64 implementation's answer to the exactly-rounded
        # result.
        exact = 1 - (1 - s**rows) ** bands
        observed = tors.lsh_probability(float(s), bands=bands, rows=rows)
        assert observed == pytest.approx(float(exact), rel=1e-12)

    def test_matches_the_formula_expression_over_a_grid(self) -> None:
        # The formula the doc spells, over a grid of s values at several
        # shapes (rel tolerance: libm pow is not required correctly
        # rounded, so a 1-ulp cross-libm difference is not a defect).
        for bands, rows in [(16, 8), (32, 4), (1, 1), (4, 4), (128, 1)]:
            for step in range(21):
                s = step / 20
                expected = 1 - (1 - s**rows) ** bands
                assert tors.lsh_probability(s, bands=bands, rows=rows) == pytest.approx(
                    expected, rel=1e-12, abs=1e-15
                ), (s, bands, rows)

    def test_monotone_in_s(self) -> None:
        for bands, rows in [(16, 8), (32, 4), (1, 1), (4, 4), (128, 1)]:
            prev = -1.0
            for step in range(21):
                s = step / 20
                p = tors.lsh_probability(s, bands=bands, rows=rows)
                assert p >= prev, (s, bands, rows)
                assert 0.0 <= p <= 1.0
                prev = p

    def test_validation_errors(self) -> None:
        for bad_s in [1.5, -0.1, float("nan"), float("inf"), float("-inf"), 2.0]:
            with pytest.raises(ValueError, match="s must be in"):
                tors.lsh_probability(bad_s, bands=16, rows=8)
        # s is an f64 parameter: bool coerces through the float
        # extraction exactly as dedup_near_dup's threshold does (True is
        # 1.0), so only non-numeric types are TypeErrors here.
        for bad_s in ["0.5", None, b"x", [0.5]]:
            with pytest.raises(TypeError):
                tors.lsh_probability(bad_s, bands=16, rows=8)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="bands must be at least 1"):
            tors.lsh_probability(0.5, bands=0, rows=8)
        with pytest.raises(ValueError, match="rows must be at least 1"):
            tors.lsh_probability(0.5, bands=16, rows=0)
        with pytest.raises(TypeError, match="not bool"):
            tors.lsh_probability(0.5, bands=True, rows=8)


class TestLshThreshold:
    def test_one_liner_pins(self) -> None:
        # (1/16)^(1/8) = 2^(-1/2), computed by hand.
        assert tors.lsh_threshold(bands=16, rows=8) == pytest.approx(
            0.7071067811865476, rel=1e-12
        )
        assert tors.lsh_threshold(bands=1, rows=8) == 1.0
        assert tors.lsh_threshold(bands=16, rows=1) == pytest.approx(1 / 16, rel=1e-15)

    def test_more_bands_lower_it_more_rows_raise_it(self) -> None:
        assert tors.lsh_threshold(bands=32, rows=8) < tors.lsh_threshold(bands=16, rows=8)
        assert tors.lsh_threshold(bands=16, rows=16) > tors.lsh_threshold(bands=16, rows=8)

    def test_threshold_tracks_the_curve(self) -> None:
        # The honest claim ("NEAR the curve's midpoint", not exactly 0.5):
        # at the approximate threshold the S-curve sits inside a band
        # around 0.5 for a spread of shapes.
        for bands, rows in [(16, 8), (32, 4), (4, 4), (128, 1)]:
            t = tors.lsh_threshold(bands=bands, rows=rows)
            p = tors.lsh_probability(t, bands=bands, rows=rows)
            assert 0.3 < p < 0.8, (bands, rows, p)

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="bands must be at least 1"):
            tors.lsh_threshold(bands=0, rows=8)
        with pytest.raises(ValueError, match="rows must be at least 1"):
            tors.lsh_threshold(bands=16, rows=0)
        with pytest.raises(TypeError, match="not bool"):
            tors.lsh_threshold(bands=True, rows=8)
        with pytest.raises(TypeError):
            tors.lsh_threshold(bands="16", rows=8)  # type: ignore[arg-type]


class TestMemoryGuard:
    """The peak-memory class the API doc promises: the banding pass
    retains one band's bucket table (O(n), freed per band) plus the pair
    set (O(output)) — never an n² structure. The disposable-child /proc
    VmHWM discipline (tests/test_near_dup.py's harness shape): the guard
    runs in a child so a regression balloons the child, never pytest."""

    def test_band_peak_is_tied_to_input_and_output_not_the_pair_space(self) -> None:
        child = subprocess.run(
            [sys.executable, "-c", _CHILD, "10000"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert child.returncode == 0, child.stderr
        peak_kib = int(child.stdout.strip())
        # The input is ~10k x 128 Python ints plus lists (~60-80 MiB live
        # in the child). The peak must be a small multiple of that;
        # an n²-sized pair structure over 10k signatures (~20M pairs)
        # would be gigabytes.
        assert peak_kib < 400 * 1024, f"peak {peak_kib} KiB"


_CHILD = """\\
import sys

def vmhwm_kib():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    raise AssertionError("VmHWM not found")

import random

import tors

n = int(sys.argv[1])
rng = random.Random(20260924)
sigs = [[rng.getrandbits(64) for _ in range(128)] for _ in range(n)]
out = tors.lsh_candidates(sigs, bands=32, rows=4)
assert out == tors.lsh_candidates(sigs, bands=32, rows=4)
print(vmhwm_kib())
"""
