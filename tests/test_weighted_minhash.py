"""Contract gate for the weighted MinHash family:
``tors.weighted_minhash_signature`` and
``tors.minhash_weighted_jaccard`` -- the Consistent Weighted Sampling
engine (Ioffe, "Improved Consistent Sampling, Weighted Minhash and L1
Sketching", ICDM 2010; Shrivastava's NeurIPS 2016 restatement, Algorithm
1; the TKDE-era survey Wu et al., arXiv 1811.04633). The frequency-aware
complement to the binary MinHash family: the binary engine reads a
document as its token SET ('aaa' repeated thrice is one element), the
weighted engine reads the COUNTS, and the fraction of permutations whose
``(token_hash, active-index)`` pairs agree estimates the GENERALIZED
Jaccard similarity ``sum_k min(w_a, w_b) / sum_k max(w_a, w_b)``.

The pinned arithmetic (``src/minhash_impl.rs``'s weighted-engine docs):
token identity is the crate's one XXH64 contract over the single-token
frame (the shingle_size-1 shingle hash); per permutation j and token
hash h with weight w, five SplitMix64 draws from the XXH64 frame
``[seed, j, h]`` give ``r, c ~ Gamma(2, 1)`` (two open-interval exponentials
each) and ``beta ~ U(0, 1)``; ``t = floor(ln(w)/r + beta)`` (the
consistency-bearing integer active index), ``y = exp(r*(t - beta))``,
``z = y*exp(r)``, ``a = c/z``; the permutation's sample is the token
minimizing ``a``, ties breaking to the lowest token hash (tokens sweep in
ascending-(hash, weight) order -- the pinned deterministic order). Rows
are the winner pair ``(hash, t)``, t as its f64 bit pattern; an empty
multiset is the all-sentinel signature. No random entropy anywhere: the
same input at the same parameters is the same signature across
processes, machines, and versions (the fork-safety cell pins the absence
of per-process state the way test_random.py's TestForkSafety does).
"""

from __future__ import annotations

import asyncio
import os
import random
import warnings

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loop_harness import assert_bounded, assert_heartbeat_clean
from reference import (
    reference_minhash_tokens,
    reference_minhash_weighted_jaccard,
    reference_weighted_minhash_signature,
)
from tors import (
    minhash_signature,
    minhash_weighted_jaccard,
    weighted_minhash_signature,
)

_NUM_PERM = st.integers(min_value=1, max_value=32)
_SEED = st.one_of(
    st.integers(min_value=-(2**64), max_value=2**64),
    st.sampled_from([0, -1, 2**63, 2**64 - 1]),
)
_WEIGHTS = st.dictionaries(
    st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=8),
    st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    max_size=8,
)


class TestOracleDifferential:
    """tors vs the transcribed pure-Python ICWS oracle: the count walk,
    the per-(j, h) stream derivation, the Gamma(2, 1) + uniform draw
    order, the floor, the argmin sweep, and the (hash, t) pair emission
    are each pinned by agreement with an independent spelling."""

    @given(weights=_WEIGHTS, num_perm=_NUM_PERM, seed=_SEED)
    @settings(max_examples=100)
    def test_matches_the_oracle_over_explicit_weights(
        self, weights: dict[str, float], num_perm: int, seed: int
    ) -> None:
        assert weighted_minhash_signature(
            weights, num_perm=num_perm, seed=seed
        ) == reference_weighted_minhash_signature(weights, num_perm=num_perm, seed=seed)

    @given(
        tokens=st.lists(
            st.text(alphabet="ab ", max_size=12), min_size=0, max_size=12
        ),
        num_perm=_NUM_PERM,
        seed=_SEED,
    )
    @settings(max_examples=100)
    def test_matches_the_oracle_over_token_lists(
        self, tokens: list[str], num_perm: int, seed: int
    ) -> None:
        assert weighted_minhash_signature(
            tokens, num_perm=num_perm, seed=seed
        ) == reference_weighted_minhash_signature(tokens, num_perm=num_perm, seed=seed)

    @given(text=st.text(max_size=120), num_perm=_NUM_PERM, seed=_SEED)
    @settings(max_examples=100)
    def test_matches_the_oracle_over_text(self, text: str, num_perm: int, seed: int) -> None:
        assert weighted_minhash_signature(
            text, num_perm=num_perm, seed=seed
        ) == reference_weighted_minhash_signature(text, num_perm=num_perm, seed=seed)

    def test_tricky_rows_match(self) -> None:
        # The rows the random alphabets never draw: CJK scriptio continua,
        # ZWJ emoji, the U+001F separator token, case folding.
        rows: list[str | list[str] | dict[str, float]] = [
            "我爱北京天安门天安门上太阳升",
            "\U0001f469‍\U0001f52c test",
            "a\x1fb a\x1fb ccc",
            "Hello, WORLD! hello world",
            ["caf\u00e9", "cafe", "caf\u00e9"],
            {"东京": 2.5, "tokyo": 1.0},
        ]
        for row in rows:
            assert weighted_minhash_signature(
                row, num_perm=8
            ) == reference_weighted_minhash_signature(row, num_perm=8), f"{row!r}"

    def test_text_spelling_equals_the_count_mapping(self) -> None:
        # One multiset, three spellings: the counts a text tokenizes to
        # are the weights a mapping spells, and a token list the literal
        # occurrence stream. All three signature identically.
        text = "the quick brown fox the quick fox fox"
        from_text = weighted_minhash_signature(text, num_perm=16)
        from_list = weighted_minhash_signature(
            reference_minhash_tokens(text), num_perm=16
        )
        from_dict = weighted_minhash_signature(
            {"the": 2, "quick": 2, "brown": 1, "fox": 3}, num_perm=16
        )
        assert from_text == from_list == from_dict


class TestWhatItAdds:
    """The frequency awareness the binary engine cannot have, pinned."""

    def test_binary_engine_cannot_see_frequency(self) -> None:
        # 'aaa' counts thrice: the token SETS of "aaa bbb" and
        # "aaa aaa bbb" are identical, so the binary engine (shingle_size
        # 1: the shingle set IS the token set) emits IDENTICAL signatures
        # and estimates J = 1. The weighted engine reads (1, 1) vs (2, 1)
        # and estimates the generalized Jaccard 2/3.
        assert minhash_signature("aaa bbb", shingle_size=1) == minhash_signature(
            "aaa aaa bbb", shingle_size=1
        )
        a = weighted_minhash_signature("aaa bbb", num_perm=128)
        b = weighted_minhash_signature("aaa aaa bbb", num_perm=128)
        est = minhash_weighted_jaccard(a, b)
        assert abs(est - 2.0 / 3.0) < 0.08, f"{est} vs exact 0.6667"

    def test_scaling_weights_move_the_estimate(self) -> None:
        # The generalized Jaccard is weight-aware in both directions:
        # (2, 1) vs (3, 1) over the same vocabulary is
        # (2 + 1) / (3 + 1) = 0.75, closer than the (1, 1) pair's 2/3.
        a = weighted_minhash_signature({"aaa": 2.0, "bbb": 1.0}, num_perm=128)
        b = weighted_minhash_signature({"aaa": 3.0, "bbb": 1.0}, num_perm=128)
        c = weighted_minhash_signature({"aaa": 1.0, "bbb": 1.0}, num_perm=128)
        # The estimates sit at their true generalized Jaccards (0.75 and
        # 0.667): no strict ordering is pinned -- the two truths are only
        # ~1.3 sigma apart at k=128, inside single-sample noise.
        near = minhash_weighted_jaccard(a, b)
        far = minhash_weighted_jaccard(a, c)
        assert abs(near - 0.75) < 0.08, near
        assert abs(far - 2.0 / 3.0) < 0.08, far

    def test_weighted_jaccard_recovers_the_count_vector_truth(self) -> None:
        # The estimator averages to the exact count-vector generalized
        # Jaccard over a deterministic battery (the empirical anchor the
        # RMSE grid scales up).
        rng = random.Random(20261005)
        vocab = [f"t{i:03d}" for i in range(40)]
        estimates = []
        exacts = []
        for trial in range(120):
            ca = {w: rng.randint(1, 6) for w in vocab if rng.random() < 0.6}
            cb = {w: rng.randint(1, 6) for w in vocab if rng.random() < 0.6}
            if not ca or not cb:
                continue
            union = set(ca) | set(cb)
            num = sum(min(ca.get(w, 0), cb.get(w, 0)) for w in union)
            den = sum(max(ca.get(w, 0), cb.get(w, 0)) for w in union)
            exacts.append(num / den)
            a = weighted_minhash_signature(ca, num_perm=128, seed=trial)
            b = weighted_minhash_signature(cb, num_perm=128, seed=trial)
            estimates.append(minhash_weighted_jaccard(a, b))
        bias = sum(e - t for e, t in zip(estimates, exacts, strict=True)) / len(exacts)
        # Ioffe's consistency: E[estimate] = J. The mean over 100+ trials
        # sits well inside the estimator noise (sigma <= 0.044 per trial,
        # ~0.004 for the mean).
        assert abs(bias) < 0.02, f"bias {bias:+.4f}"


class TestDeterminismAndForkSafety:
    def test_deterministic_across_calls_and_objects(self) -> None:
        text = "the quarterly oil sample interval for field outages"
        a = weighted_minhash_signature(text, num_perm=64)
        assert a == weighted_minhash_signature(text, num_perm=64)
        assert a == weighted_minhash_signature("".join(text), num_perm=64)
        assert a != weighted_minhash_signature(text, num_perm=64, seed=1)
        assert weighted_minhash_signature(
            text, num_perm=64, seed=-1
        ) == weighted_minhash_signature(text, num_perm=64, seed=2**64 - 1)

    def test_seed_none_is_the_fixed_default_seed_zero(self) -> None:
        # seed=None is seed=0, a documented equality: there is no random
        # entropy anywhere in the engine to substitute for it.
        text = "pack my box with five dozen liquor jugs"
        assert weighted_minhash_signature(text, num_perm=32, seed=None) == (
            weighted_minhash_signature(text, num_perm=32, seed=0)
        )

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="no os.fork on this platform")
    def test_fork_child_signature_is_identical(self) -> None:
        # The fork-safety positive control (test_random.py's TestForkSafety
        # pattern): the engine holds no per-process state -- a cached RNG
        # or a lazy global would replay the parent's stream in the child.
        text = "the quick brown fox jumps over the lazy dog"
        parent = weighted_minhash_signature(text, num_perm=64)
        read_fd, write_fd = os.pipe()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            pid = os.fork()
        if pid == 0:
            try:
                os.close(read_fd)
                os.write(write_fd, repr(weighted_minhash_signature(text, num_perm=64)).encode())
            finally:
                os._exit(0)
        os.close(write_fd)
        chunks = []
        while True:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert eval(b"".join(chunks).decode()) == parent  # noqa: S307

    def test_empty_multiset_is_the_all_sentinel(self) -> None:
        # Empty text, whitespace-only text (no tokens), an empty list, an
        # empty mapping, and an all-zero mapping are the empty multiset:
        # every row the u64 MAX sentinel, seed-invariant.
        sentinel = [2**64 - 1] * 16
        for empty in ("", "   ", "\t\n ", [], {}, {"a": 0.0, "b": 0}):
            sig = weighted_minhash_signature(empty, num_perm=8)  # type: ignore[arg-type]
            assert sig == sentinel, f"{empty!r}"
        assert weighted_minhash_signature("", num_perm=8, seed=12345) == sentinel
        # Two empty multisets agree at every permutation (J = 1 by the
        # empty-set convention); empty vs nonempty at none.
        a = weighted_minhash_signature("", num_perm=8)
        assert minhash_weighted_jaccard(a, a) == 1.0
        b = weighted_minhash_signature("a b c", num_perm=8)
        assert minhash_weighted_jaccard(a, b) == 0.0

    def test_zero_weight_tokens_are_excluded(self) -> None:
        # A zero-weight token contributes nothing to the generalized
        # Jaccard: {"a": 0, "b": 1} IS {"b": 1}.
        assert weighted_minhash_signature(
            {"a": 0.0, "b": 1.0}, num_perm=16
        ) == weighted_minhash_signature({"b": 1.0}, num_perm=16)

    def test_single_token_pins_the_exact_pair(self) -> None:
        # One token at weight 1: EVERY permutation samples it (the argmin
        # over one candidate), and t = floor(ln(1)/r + beta) = floor(beta)
        # = 0: the signature is (hash, 0) repeated.
        token = "hello"
        import struct

        import xxhash

        frame = struct.pack("<Q", 1) + struct.pack("<Q", len(token.encode())) + token.encode()
        h = xxhash.xxh64_intdigest(frame)
        assert weighted_minhash_signature(token, num_perm=8) == [h, 0] * 8


class TestArgumentContract:
    def test_num_perm_is_required(self) -> None:
        with pytest.raises(TypeError):
            weighted_minhash_signature("a b c")  # type: ignore[call-arg]

    @pytest.mark.parametrize("num_perm", [0, -1, 1025, 10**9])
    def test_num_perm_out_of_bounds_raises_value_error(self, num_perm: int) -> None:
        with pytest.raises(ValueError, match="1 and 1024"):
            weighted_minhash_signature("a b c", num_perm=num_perm)

    def test_num_perm_bounds_are_inclusive(self) -> None:
        assert len(weighted_minhash_signature("a b c", num_perm=1)) == 2
        assert len(weighted_minhash_signature("a b c", num_perm=1024)) == 2048

    def test_length_is_always_two_times_num_perm(self) -> None:
        for num_perm in (1, 7, 128):
            assert len(weighted_minhash_signature("a b c", num_perm=num_perm)) == 2 * num_perm

    def test_input_spellings(self) -> None:
        # str, list/tuple of str, and the weight mapping all work; bytes,
        # ints, and a sequence of non-strs are TypeErrors.
        assert len(weighted_minhash_signature("a b c", num_perm=4)) == 8
        assert len(weighted_minhash_signature(["a", "b", "c"], num_perm=4)) == 8
        assert len(weighted_minhash_signature(("a", "b", "c"), num_perm=4)) == 8
        assert len(weighted_minhash_signature({"a": 1, "b": 2}, num_perm=4)) == 8
        for bad in (42, b"a b c", None, ["a", 3, "c"]):
            with pytest.raises(TypeError):
                weighted_minhash_signature(bad, num_perm=4)  # type: ignore[arg-type]

    def test_mapping_key_and_value_contract(self) -> None:
        with pytest.raises(TypeError, match="weight keys"):
            weighted_minhash_signature({1: 1.0}, num_perm=4)  # type: ignore[dict-item]
        with pytest.raises(TypeError, match="weight values"):
            weighted_minhash_signature({"a": "1"}, num_perm=4)  # type: ignore[dict-item]
        with pytest.raises(TypeError, match="bool"):
            weighted_minhash_signature({"a": True}, num_perm=4)  # type: ignore[dict-item]
        for bad_value in (-1.0, -1e-9, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="finite and non-negative"):
                weighted_minhash_signature({"a": bad_value}, num_perm=4)

    def test_seed_contract(self) -> None:
        text = "pack my box with five dozen liquor jugs"
        for bad in ("0", 0.5, b"0"):
            with pytest.raises(TypeError):
                weighted_minhash_signature(text, num_perm=4, seed=bad)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            weighted_minhash_signature(text, num_perm=4, seed=True)  # type: ignore[arg-type]
        assert weighted_minhash_signature(
            text, num_perm=4, seed=2**100
        ) == weighted_minhash_signature(text, num_perm=4, seed=0)

    def test_lone_surrogate_text_raises_unicode_encode_error(self) -> None:
        # The crate-wide str-borrow contract.
        with pytest.raises(UnicodeEncodeError):
            weighted_minhash_signature("a\ud800b", num_perm=4)

    @given(weights=_WEIGHTS, num_perm=_NUM_PERM)
    @settings(max_examples=50)
    def test_signature_shape_holds(self, weights: dict[str, float], num_perm: int) -> None:
        sig = weighted_minhash_signature(weights, num_perm=num_perm)
        assert len(sig) == 2 * num_perm
        assert all(isinstance(v, int) for v in sig)


class TestEstimatorContract:
    def test_shape_and_agreement_of_the_estimator(self) -> None:
        a = weighted_minhash_signature("a b c d e", num_perm=32)
        b = weighted_minhash_signature("a b c d f", num_perm=32)
        est = minhash_weighted_jaccard(a, b)
        assert isinstance(est, float)
        assert 0.0 <= est <= 1.0
        # The estimator agrees with the pure-Python oracle.
        assert est == reference_minhash_weighted_jaccard(a, b)
        # Identical multisets: exactly 1.0. Disjoint vocabularies: 0.0.
        assert minhash_weighted_jaccard(a, a) == 1.0
        c = weighted_minhash_signature("x y z w v", num_perm=32)
        assert minhash_weighted_jaccard(a, c) == 0.0

    def test_estimator_shape_contract(self) -> None:
        sig = weighted_minhash_signature("a b c", num_perm=8)
        with pytest.raises(ValueError, match="lengths differ"):
            minhash_weighted_jaccard(sig, sig[:4])
        # Odd EQUAL lengths (15 vs 15) pass the length check and fail the
        # shape one; the empty pair fails it too.
        with pytest.raises(ValueError, match="ICWS signature shape"):
            minhash_weighted_jaccard(sig[:-1], sig[:-1])
        with pytest.raises(ValueError, match="ICWS signature shape"):
            minhash_weighted_jaccard([], [])
        with pytest.raises((TypeError, ValueError)):
            minhash_weighted_jaccard(["x"], ["x"])  # type: ignore[list-item]

    def test_not_cross_compatible_with_the_binary_engines(self) -> None:
        # The weighted rows are (hash, t-bit-pattern) pairs; the binary
        # engines' rows are min-hashes. A mixed call with mismatched
        # lengths raises; a LENGTH-COINCIDENT mixed call is the documented
        # UNDETECTABLE caller error -- rows are opaque ints, nothing can
        # identify their engine, so the contract is documentation (the
        # call returns a meaningless float rather than lying by raising).
        binary = minhash_signature("a b c d e", num_perm=64)
        weighted = weighted_minhash_signature("a b c d e", num_perm=32)
        with pytest.raises(ValueError, match="lengths differ"):
            minhash_weighted_jaccard(weighted, binary + binary)  # type: ignore[arg-type]
        # The length-coincident mix (64 binary rows vs 64 weighted rows:
        # 32 permutations) is not detectable; the hazard is documented.
        mixed = minhash_weighted_jaccard(weighted, binary)  # type: ignore[arg-type]
        assert isinstance(mixed, float)

    @pytest.mark.timing
    def test_generalized_jaccard_rmse_grid(self) -> None:
        # The accuracy cell over a count-vector grid: RMSE of the
        # weighted estimator vs the EXACT generalized Jaccard at k = 128
        # over random count vectors with J spread across the range
        # (measured 0.030-0.041; the theoretical per-trial sigma at the
        # worst J = 0.5 is sqrt(0.25/128) ~ 0.044 -- the RMSE sits at the
        # binomial band, Ioffe's consistency made empirical).
        rng = random.Random(424242)
        vocab = [f"t{i:03d}" for i in range(40)]
        estimates = []
        exacts = []
        for trial in range(300):
            ca = {w: rng.randint(1, 6) for w in vocab if rng.random() < 0.6}
            cb = {w: rng.randint(1, 6) for w in vocab if rng.random() < 0.6}
            if not ca or not cb:
                continue
            union = set(ca) | set(cb)
            num = sum(min(ca.get(w, 0), cb.get(w, 0)) for w in union)
            den = sum(max(ca.get(w, 0), cb.get(w, 0)) for w in union)
            exacts.append(num / den)
            a = weighted_minhash_signature(ca, num_perm=128, seed=trial)
            b = weighted_minhash_signature(cb, num_perm=128, seed=trial)
            estimates.append(minhash_weighted_jaccard(a, b))
        exact_mean = sum(exacts) / len(exacts)
        rmse = sum((e - t) ** 2 for e, t in zip(estimates, exacts, strict=True)) ** 0.5 / len(
            estimates
        ) ** 0.5
        assert rmse < 0.055, f"{rmse:.4f} vs band 0.055 (exact mean J {exact_mean:.3f})"

    @pytest.mark.timing
    def test_weighted_pass_keeps_the_event_loop_responsive(self) -> None:
        # The GIL cell, the shared harness's heartbeat discipline: the
        # count walk plus the ICWS sweep over ~12 MiB of prose run under
        # one py.detach (the corpus is the classic cell's 12 MiB shape: a
        # pass this fast at 1 MiB would sit on the heartbeat cadence's
        # ratio knife-edge).
        from reference import prose

        corpus = prose(12 * 1024 * 1024)
        assert_heartbeat_clean(
            lambda: asyncio.to_thread(weighted_minhash_signature, corpus, num_perm=128),
            subject="the weighted ICWS pass",
        )

    @pytest.mark.timing
    def test_weighted_timing_row(self) -> None:
        # The wall tripwire over the worst-case shape (distinct-rich
        # tokens: every token unique, the count map and the sweep both
        # full-size; ~100k tokens x 128 permutations). A generous ceiling
        # ~10x the dev-box nominal, the assert_bounded discipline.
        corpus = " ".join(f"tok{i:06d}" for i in range(100_000))
        sig = assert_bounded(
            lambda: weighted_minhash_signature(corpus, num_perm=128),
            60.0,
            samples=3,
            label="the 100k-token ICWS pass",
        )
        assert len(sig) == 256
