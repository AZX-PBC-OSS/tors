"""Contract gate for the near-duplicate comparison family:
``tors.simhash_distance``, ``tors.shingle_jaccard``, ``tors.shingle_dice``,
and ``tors.dedup_near_dup`` — the layer that sits on top of the similarity
primitives (`simhash64`, `minhash_signature`) and turns them into
comparisons and a dedup sweep. See ``src/near_dup_impl.rs`` for the
algorithm writeup (Broder 1997's shingling + resemblance; Lee et al. 2021,
<https://arxiv.org/abs/2107.06499>, for the dedup value at LLM-pretraining
scale) and the normalization policy (the grounding layer's lowercase +
NFC fold); this module is the Python-visible half of the same contract.

Scope honesty (docs/design.md's scope cuts, mirrored in the API docs):
``dedup_near_dup`` is O(n²) pair checks BY DESIGN — no LSH banding index,
no persistent state, the small-candidate-set shape `bm25_rank` shares.
The scaling pin (quadratic, with an explicit wall budget) lives in
``tests/test_scaling_pins.py``; the GIL-release cells in
``tests/test_gil_release.py``; the peak-memory guard here.
"""

from __future__ import annotations

import asyncio
import math
import random
import subprocess
import sys
import unicodedata
from fractions import Fraction
from itertools import product
from math import nextafter
from time import monotonic

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors
import tors.aio
from loop_harness import assert_bounded

_TEXT = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs", "P"), max_codepoint=0x2FFF),
    max_size=200,
)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _perturb(words: list[str], *, kind: str) -> str:
    """A small, meaning-preserving perturbation: an adjacent-word
    transposition, or a casing flip on every third word. Both leave the
    document recognizably the same document."""
    out = list(words)
    if kind == "transpose":
        out[3], out[4] = out[4], out[3]
    elif kind == "case":
        out = [w.upper() if i % 3 == 0 else w for i, w in enumerate(out)]
    else:  # pragma: no cover
        raise AssertionError(kind)
    return " ".join(out)


class TestSimHashDistance:
    def test_zero_on_identical_values(self) -> None:
        for text in ["", "hello world", "the quick brown fox jumps over the lazy dog", "café"]:
            fp = tors.simhash64(text)
            assert tors.simhash_distance(fp, fp) == 0

    def test_symmetry(self) -> None:
        a = tors.simhash64("the quick brown fox jumps over the lazy dog")
        b = tors.simhash64("pack my box with five dozen liquor jugs")
        assert tors.simhash_distance(a, b) == tors.simhash_distance(b, a)
        assert tors.simhash_distance(a, b) == _hamming(a, b)

    def test_agrees_with_the_call_site_one_liner(self) -> None:
        # The function IS `(a ^ b).bit_count()`; pin the equality over
        # real fingerprints, both spellings.
        pairs = [
            ("the quick brown fox", "the slow brown fox jumps over the lazy dog"),
            ("hello", "goodbye world"),
            ("café", "café société"),
        ]
        for x, y in pairs:
            a64, b64 = tors.simhash64(x), tors.simhash64(y)
            assert tors.simhash_distance(a64, b64) == _hamming(a64, b64)
            a128, b128 = tors.simhash128(x), tors.simhash128(y)
            assert tors.simhash_distance(a128, b128) == _hamming(a128, b128)

    def test_determinism(self) -> None:
        a = tors.simhash64("the quarterly oil sample interval")
        b = tors.simhash64("pack my box with five dozen liquor jugs")
        assert tors.simhash_distance(a, b) == tors.simhash_distance(a, b)

    def test_width_mismatch_is_a_value_error(self) -> None:
        # One simhash64 value, one simhash128 value: the real caller bug
        # the magnitude check exists to catch.
        a = tors.simhash64("hello world")
        b = tors.simhash128("hello world")
        with pytest.raises(ValueError, match="same width"):
            tors.simhash_distance(a, b)
        with pytest.raises(ValueError, match="same width"):
            tors.simhash_distance(b, a)

    def test_mixed_width_checks_are_by_magnitude_and_documented(self) -> None:
        # The honest edge of the magnitude classification: two values
        # that both fit in 64 bits compare correctly either way (the
        # arithmetic is width-blind), whatever spelling produced them.
        assert tors.simhash_distance(0, 0) == 0
        assert tors.simhash_distance(0b1010, 0b0110) == 2

    def test_non_int_arguments_are_type_errors(self) -> None:
        fp = tors.simhash64("hello")
        for bad in ["hello", None, 1.5, b"x", True, False]:
            with pytest.raises(TypeError):
                tors.simhash_distance(bad, fp)
            with pytest.raises(TypeError):
                tors.simhash_distance(fp, bad)

    def test_negative_is_a_value_error(self) -> None:
        fp = tors.simhash64("hello")
        with pytest.raises(ValueError, match="unsigned"):
            tors.simhash_distance(-1, fp)
        with pytest.raises(ValueError, match="unsigned"):
            tors.simhash_distance(fp, -(2**70))

    def test_past_u128_overflows(self) -> None:
        fp = tors.simhash64("hello")
        with pytest.raises(OverflowError):
            tors.simhash_distance(2**128, fp)

    def test_int_likes_ride_the_index_protocol(self) -> None:
        class Indexable:
            def __index__(self) -> int:
                return 0

        assert tors.simhash_distance(Indexable(), Indexable()) == 0

    @given(text=_TEXT)
    @settings(max_examples=100)
    def test_distance_to_self_is_zero_property(self, text: str) -> None:
        fp = tors.simhash64(text)
        assert tors.simhash_distance(fp, fp) == 0
        fp128 = tors.simhash128(text)
        assert tors.simhash_distance(fp128, fp128) == 0


class TestShingleSimilarities:
    def test_identical_texts_score_one(self) -> None:
        for text in [
            "the quick brown fox jumps over the lazy dog",
            "café société",
            "",
            "   ",
        ]:
            assert tors.shingle_jaccard(text, text) == 1.0
            assert tors.shingle_dice(text, text) == 1.0

    def test_disjoint_texts_score_zero(self) -> None:
        assert tors.shingle_jaccard("alpha beta gamma", "delta epsilon zeta") == 0.0
        assert tors.shingle_dice("alpha beta gamma", "delta epsilon zeta") == 0.0
        # Exactly one token-free side: 0.0.
        assert tors.shingle_jaccard("", "alpha beta gamma") == 0.0
        assert tors.shingle_dice("", "alpha beta gamma") == 0.0

    def test_two_token_free_texts_are_duplicates(self) -> None:
        # The empty-set convention: ∅ ⊆ ∅, so two token-free texts
        # (empty or whitespace-only) are identical — the answer
        # dedup_near_dup(method="shingle") must conclude too.
        assert tors.shingle_jaccard("", "") == 1.0
        assert tors.shingle_jaccard("   ", "\t\n") == 1.0
        assert tors.shingle_dice("", "") == 1.0

    def test_scores_live_in_the_unit_interval_and_are_symmetric(self) -> None:
        pairs = [
            ("the quick brown fox", "the quick brown fox!"),
            ("one two three", "one two three four five"),
            ("a b c d e", "a b c d e f g"),
            ("café société naïve", "cafe societe naive"),
        ]
        for a, b in pairs:
            j, jba = tors.shingle_jaccard(a, b), tors.shingle_jaccard(b, a)
            d, dba = tors.shingle_dice(a, b), tors.shingle_dice(b, a)
            assert j == jba
            assert d == dba
            assert 0.0 <= j <= 1.0
            assert 0.0 <= d <= 1.0
            assert d >= j  # 2i/(x+y) >= i/(x+y-i) whenever i <= x+y-i

    def test_case_and_whitespace_are_invisible(self) -> None:
        # The normalization policy's case half: tokens are lowercased
        # before hashing, so respelled case is the same shingle set.
        assert tors.shingle_jaccard("Hello World", "hello world") == 1.0
        assert tors.shingle_jaccard("A B C D E", "a\nb\tc  d e") == 1.0
        assert tors.shingle_dice("Hello World", "hello world") == 1.0
        # Punctuation is NOT invisible: the tokenizer's rule (simhash's
        # and minhash's own) keeps punctuation-only segments as tokens,
        # so a comma is part of the shingle stream — a documented token
        # model, not a normalization gap.
        assert tors.shingle_jaccard("Hello, World", "hello world") < 1.0

    def test_nfc_equivalent_inputs_behave_identically(self) -> None:
        # The normalization policy's NFC half (the grounding layer's
        # fold, mirrored): NFD "cafe\u0301" and NFC "café" are the same
        # tokens, so they score 1.0 against each other AND identically
        # against any third text.
        nfc = "café société naïve"
        nfd = "cafe\u0301 socie\u0301te\u0301 na\u00efve"
        assert nfc != nfd
        assert tors.shingle_jaccard(nfc, nfd) == 1.0
        assert tors.shingle_dice(nfc, nfd) == 1.0
        third = "the quarterly oil sample interval for field outages"
        assert tors.shingle_jaccard(nfc, third) == tors.shingle_jaccard(nfd, third)

    def test_cjk_and_emoji_safety(self) -> None:
        # UAX #29 word segmentation finds word boundaries inside CJK
        # runs (no whitespace needed), and emoji segments are tokens
        # (real_word_segments keeps every non-whitespace segment).
        cjk = "東京タワーは東京のランドマークである"
        assert tors.shingle_jaccard(cjk, cjk) == 1.0
        assert 0.0 < tors.shingle_jaccard(cjk, cjk + "です") < 1.0
        assert tors.shingle_jaccard(cjk, "全く別の文章です") == 0.0
        emoji = "\U0001f469\u200d\U0001f52c test \U0001f1fa\U0001f1f8"
        assert tors.shingle_jaccard(emoji, emoji) == 1.0

    def test_width_changes_the_shingle_unit(self) -> None:
        # width=1 is the unigram bag; width=3 the trigram window; a
        # shared unigram bag does not imply shared trigrams.
        a, b = "one two three four five", "five two three four one"
        assert tors.shingle_jaccard(a, b, width=1) == 1.0
        assert tors.shingle_jaccard(a, b, width=3) < 1.0

    def test_width_short_of_the_token_stream_is_the_empty_set(self) -> None:
        # A width wider than the token stream: no shingles at all —
        # both sides empty score 1.0, one side 0.0.
        assert tors.shingle_jaccard("one two", "three four", width=3) == 1.0
        assert tors.shingle_jaccard("one two", "one two three", width=3) == 0.0

    def test_width_bounds(self) -> None:
        for bad in [0, -1, -100]:
            with pytest.raises(ValueError, match="width must be at least 1"):
                tors.shingle_jaccard("a b c", "a b c", width=bad)
            with pytest.raises(ValueError, match="width must be at least 1"):
                tors.shingle_dice("a b c", "a b c", width=bad)

    def test_width_rides_the_index_protocol(self) -> None:
        class Width:
            def __index__(self) -> int:
                return 2

        assert tors.shingle_jaccard("a b c", "a b c", width=Width()) == 1.0
        with pytest.raises(TypeError, match="not bool"):
            tors.shingle_jaccard("a b c", "a b c", width=True)

    def test_sweep_budget_gate(self) -> None:
        # The same 2^26 token-hash budget minhash_signature enforces: a
        # width past 1024 over a stream that fills the window re-hashes
        # the whole live window per token, and the pair functions reject
        # the shape before any work runs (measured anchor shared with
        # tests/test_minhash.py's exactly-at/one-past pins).
        text = "w " * 20_000
        with pytest.raises(ValueError, match="token-hash budget"):
            tors.shingle_jaccard(text, text, width=5_000)
        with pytest.raises(ValueError, match="token-hash budget"):
            tors.shingle_dice(text, text, width=5_000)

    def test_non_str_arguments_are_type_errors(self) -> None:
        for bad in [1, None, b"x", ["a"]]:
            with pytest.raises(TypeError):
                tors.shingle_jaccard(bad, "a b c")  # type: ignore[arg-type]
            with pytest.raises(TypeError):
                tors.shingle_dice("a b c", bad)  # type: ignore[arg-type]

    @given(text=_TEXT)
    @settings(max_examples=100)
    def test_self_similarity_property(self, text: str) -> None:
        assert tors.shingle_jaccard(text, text) == 1.0
        assert tors.shingle_dice(text, text) == 1.0


class TestDedupNearDup:
    def test_empty_list_is_the_empty_result(self) -> None:
        assert tors.dedup_near_dup([]) == {"kept": [], "dropped": [], "groups": []}

    def test_single_element(self) -> None:
        assert tors.dedup_near_dup(["one text only"]) == {
            "kept": [0],
            "dropped": [],
            "groups": [[0]],
        }

    @pytest.mark.parametrize("method", ["simhash", "shingle", "minhash"])
    def test_all_identical_keeps_the_first(self, method: str) -> None:
        texts = ["the same words exactly"] * 5
        result = tors.dedup_near_dup(texts, method=method)
        assert result["kept"] == [0]
        assert result["dropped"] == [1, 2, 3, 4]
        assert result["groups"] == [[0, 1, 2, 3, 4]]

    @pytest.mark.parametrize("method", ["simhash", "shingle", "minhash"])
    def test_all_distinct_keeps_everything(self, method: str) -> None:
        # The rows share NO word token (unique vocabulary per row), so
        # no shingle overlaps and no method can merge them.
        texts = [f"tok{i}_a tok{i}_b tok{i}_c tok{i}_d tok{i}_e" for i in range(6)]
        result = tors.dedup_near_dup(texts, method=method)
        assert result["kept"] == list(range(6))
        assert result["dropped"] == []
        assert result["groups"] == [[i] for i in range(6)]

    def test_two_token_free_texts_dedup_together(self) -> None:
        # The empty-set convention through the sweep: two empty (and two
        # whitespace-only) texts are duplicates of each other, for the
        # shingle method exactly and for simhash trivially (both
        # fingerprints 0).
        assert tors.dedup_near_dup(["", ""])["kept"] == [0]
        result = tors.dedup_near_dup(["   ", "\t\n", "real text here"], method="shingle")
        assert result["groups"][:2] == [[0, 1], [2]]

    def test_determinism_and_order_preservation(self) -> None:
        texts = [
            "the lighthouse keeper walked the stone steps every morning",
            "The Lighthouse keeper walked the stone steps every morning",
            "completely unrelated vocabulary about quantum chemistry bonds",
            "the lighthouse keeper walked the stone steps every night",
            "another unrelated row about compiler backends and graphs",
        ]
        first = tors.dedup_near_dup(texts, threshold=0.8, method="simhash")
        again = tors.dedup_near_dup([t for t in texts], threshold=0.8, method="simhash")
        assert first == again
        # Input order is preserved: kept and dropped are ascending, and
        # a fresh list object with equal content gives the same result.
        assert first["kept"] == sorted(first["kept"])
        assert first["dropped"] == sorted(first["dropped"])

    def test_input_order_is_the_tie_break(self) -> None:
        # Three pairwise-identical texts: whichever comes first is the
        # representative — reversing the input keeps the first slot's
        # text as the representative, never a later one.
        texts = ["same shared words here", "same shared words here", "same shared words here"]
        assert tors.dedup_near_dup(texts)["kept"] == [0]
        assert tors.dedup_near_dup(list(reversed(texts)))["kept"] == [0]

    def test_kept_first_uses_the_first_kept_text_as_representative(self) -> None:
        # A text similar to a kept text is claimed by the FIRST one it
        # matches; the case-folded variant belongs to base's group.
        base = "the quarterly oil sample interval for field outages was adjusted"
        variant = "THE QUARTERLY OIL SAMPLE INTERVAL FOR FIELD OUTAGES WAS ADJUSTED"
        third = "a wholly different document about volcanic rock formations and lava"
        result = tors.dedup_near_dup([base, third, variant], method="simhash")
        assert result["kept"] == [0, 1]
        assert result["dropped"] == [2]
        assert result["groups"] == [[0, 2], [1]]

    @pytest.mark.parametrize("method", ["simhash", "shingle", "minhash"])
    def test_groups_partition_the_indices(self, method: str) -> None:
        texts = (
            [f"family alpha row {i} shared core content" for i in range(4)]
            + [f"solo document {i} entirely alone {i}" for i in range(4)]
            + ["shared family text alpha", "shared family text alpha"]
        )
        result = tors.dedup_near_dup(texts, threshold=0.8, method=method)
        assert result["kept"] == [g[0] for g in result["groups"]]
        assert sorted(result["kept"] + result["dropped"]) == list(range(len(texts)))
        flattened = sorted(i for g in result["groups"] for i in g)
        assert flattened == list(range(len(texts)))
        for group in result["groups"]:
            assert group[0] == min(group)

    @pytest.mark.parametrize("method", ["simhash", "shingle", "minhash"])
    def test_threshold_monotonicity_fewer_drops_as_threshold_rises(
        self, method: str
    ) -> None:
        # A higher threshold is stricter: the drop COUNT never grows as
        # the threshold rises. (The dropped SETS are not pinned nested —
        # greedy keep-first re-associates chains when a representative
        # changes; the pair-level exact monotonicity is pinned in
        # src/near_dup_impl.rs's own tests.)
        texts = (
            [f"duplicate family alpha member {i} shared core text" for i in range(6)]
            + [f"duplicate family beta member {i} other core text" for i in range(6)]
            + [f"solo document {i} stands entirely alone here" for i in range(6)]
        )
        prev = None
        for threshold in [0.0, 0.3, 0.6, 0.85, 0.95, 1.0]:
            result = tors.dedup_near_dup(texts, threshold=threshold, method=method)
            if prev is not None:
                assert len(result["dropped"]) <= len(prev), method
            prev = result["dropped"]

    def test_exact_duplicate_families_are_threshold_set_monotone(self) -> None:
        # The clean-corpus subset pin, on the method with exact set
        # semantics: exact-duplicate families (copies) merge at every
        # threshold, so the dropped set only ever shrinks as the
        # threshold rises — set nesting where no re-association can
        # occur. (The simhash method's low thresholds are a different
        # regime: at 0.3 the 64-bit fingerprint budget is 44 bits, wider
        # than the typical unrelated-fingerprint distance, so nearly
        # everything merges — the calibration caveat, honestly, not a
        # defect; the family corpus below is disjoint-vocabulary for it.)
        texts = (
            ["alpha beta gamma delta"] * 3
            + ["epsilon zeta eta theta"] * 3
            + ["a solo document stands alone here"]
        )
        prev: set[int] | None = None
        for threshold in [0.3, 0.6, 0.9, 1.0]:
            dropped = set(
                tors.dedup_near_dup(texts, threshold=threshold, method="shingle")["dropped"]
            )
            assert dropped == {1, 2, 4, 5}  # copies merge, heads stay kept
            if prev is not None:
                assert dropped <= prev
            prev = dropped

    def test_threshold_extremes(self) -> None:
        # threshold=0: everything matches everything -> keep the first.
        for method in ["simhash", "shingle", "minhash"]:
            result = tors.dedup_near_dup(
                ["alpha beta gamma", "delta epsilon zeta"], threshold=0.0, method=method
            )
            assert result["kept"] == [0], method
            assert result["dropped"] == [1], method
        # threshold=1.0: only exact duplicates merge.
        texts = ["alpha beta gamma", "alpha beta gamma", "alpha beta delta"]
        result = tors.dedup_near_dup(texts, threshold=1.0, method="shingle")
        assert result["kept"] == [0, 2]
        assert result["dropped"] == [1]

    def test_method_semantics_differ_where_they_should(self) -> None:
        # A permutation that destroys most trigram windows while leaving
        # the word bag untouched: the shingle method (exact Jaccard)
        # separates the pair, the simhash method (a bag-of-words vote)
        # cannot.
        a = "one two three four five six seven eight nine ten"
        b = "ten two three four five six seven eight nine one"
        shingle = tors.dedup_near_dup([a, b], threshold=0.9, method="shingle")
        assert shingle["kept"] == [0, 1]  # the trigrams are badly damaged
        simhash = tors.dedup_near_dup([a, b], threshold=0.9, method="simhash")
        assert simhash["kept"] == [0]  # the word bag is identical

    def test_minhash_method_merges_near_duplicates(self) -> None:
        # A long-enough base that one word change (sunrise -> sunset)
        # damages only its surrounding windows: exact J ~ 35/41 ~ 0.85,
        # inside the k=128 estimator's reach at threshold 0.75.
        sentence = "the lighthouse keeper walked the stone steps every morning before"
        base = " ".join([sentence] * 4 + ["sunrise"])
        near = " ".join([sentence] * 4 + ["sunset"])
        far = "compiler backends schedule instructions over directed acyclic graphs"
        result = tors.dedup_near_dup([base, near, far], threshold=0.75, method="minhash")
        assert result["groups"][0][:2] == [0, 1]  # the near pair merged
        assert result["kept"] == [0, 2]

    def test_validation_errors(self) -> None:
        texts = ["a b c"]
        for bad_threshold in [1.5, -0.1, float("nan"), float("inf"), float("-inf")]:
            with pytest.raises(ValueError, match="threshold must be in"):
                tors.dedup_near_dup(texts, threshold=bad_threshold)
        with pytest.raises(ValueError, match="valid choices"):
            tors.dedup_near_dup(texts, method="dice")
        with pytest.raises(ValueError, match="valid choices"):
            tors.dedup_near_dup(texts, method="SimHash")  # case-sensitive
        for bad_element in [1, None, True, b"x", 1.5, ["a"]]:
            with pytest.raises(TypeError):
                tors.dedup_near_dup(["a b c", bad_element])  # type: ignore[list-item]
        with pytest.raises(TypeError):
            tors.dedup_near_dup("not a list")  # type: ignore[arg-type]

    @given(
        words=st.lists(
            st.text(alphabet="abcdefghij", min_size=3, max_size=8), min_size=30, max_size=50
        )
    )
    @settings(max_examples=50)
    def test_property_perturbed_copies_are_detected(self, words: list[str]) -> None:
        """The Hypothesis lane: small perturbations (an adjacent-word
        transposition, casing) of arbitrary word text are DETECTED —
        the original and its perturbation land in one group, at high
        thresholds, for the methods whose similarity the perturbation
        provably cannot destroy."""
        text = " ".join(words)
        for kind in ["transpose", "case"]:
            perturbed = _perturb(words, kind=kind)
            # simhash: the dedup fingerprint rides the fold, so BOTH
            # perturbations are distance 0 — grouped at ANY threshold,
            # including 1.0. (The raw simhash64 surface deliberately
            # leaves case folding to the caller: the transposition — a
            # bag-of-words no-op — is distance 0 there too, the casing
            # flip is not, which is exactly why the dedup core folds
            # first.)
            if kind == "transpose":
                assert tors.simhash_distance(tors.simhash64(text), tors.simhash64(perturbed)) == 0
            result = tors.dedup_near_dup([text, perturbed], threshold=1.0, method="simhash")
            assert result["kept"] == [0], (kind, text, perturbed)
        # shingle: a transposition damages only the windows around it —
        # at most 4 distinct trigrams leave A and at most 4 enter B, so
        # with S = |distinct trigrams of text| the exact Jaccard is
        # bounded below by (S-4)/(S+8), an oracle bound derived from the
        # token model (the words are single UAX #29 segments), not a
        # magic constant.
        transposed = _perturb(words, kind="transpose")
        tokens = text.split()
        S = len({tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)})
        assert tors.shingle_jaccard(text, transposed) >= (S - 4) / (S + 8)
        # With enough distinct trigrams the bound clears 0.5 and the
        # dedup sweep groups the pair at that threshold.
        if S >= 28:
            result = tors.dedup_near_dup([text, transposed], threshold=0.5, method="shingle")
            assert result["kept"] == [0]

    @given(texts=st.lists(st.text(min_size=1, max_size=40), max_size=12))
    @settings(max_examples=50)
    def test_property_structure_and_determinism(self, texts: list[str]) -> None:
        """Over arbitrary corpora: the result is deterministic, the
        groups partition the indices, and kept heads are representatives."""
        result = tors.dedup_near_dup(texts, threshold=0.7)
        again = tors.dedup_near_dup(texts, threshold=0.7)
        assert result == again
        assert sorted(result["kept"] + result["dropped"]) == list(range(len(texts)))
        assert result["kept"] == [g[0] for g in result["groups"]]

    def test_result_type_is_a_plain_dict(self) -> None:
        result = tors.dedup_near_dup(["a b c"])
        assert isinstance(result, dict)
        assert set(result) == {"kept", "dropped", "groups"}


class TestMemoryGuard:
    """The peak-memory class the API doc promises: the dedup sweep
    retains O(total input) of fingerprint state — one fingerprint per
    text — and allocates nothing per pair, so the process peak over a
    large corpus is tied to the corpus bytes, not to n². The
    disposable-child /proc VmHWM discipline
    (tests/test_memory_spike_guards.py's harness shape): the guard runs
    in a child so a regression balloons the child, never pytest."""

    def test_dedup_peak_is_tied_to_the_corpus_not_the_pair_count(self) -> None:
        child = subprocess.run(
            [sys.executable, "-c", _CHILD, "10000"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert child.returncode == 0, child.stderr
        peak_kib = int(child.stdout.strip())
        # The corpus is ~0.7 MiB of text. The peak must be a small
        # multiple of the input (interpreter baseline + corpus + one
        # fingerprint per text), nowhere near the ~800 MB an n²-sized
        # allocation over 10k texts would need.
        assert peak_kib < 200 * 1024, f"peak {peak_kib} KiB"


_CHILD = """\\
import sys

def vmhwm_kib():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    raise AssertionError("VmHWM not found")

import tors

n = int(sys.argv[1])
corpus = [
    f"document {i} token stream with distinct vocabulary {i} and a tail"
    for i in range(n)
]
for method in ("simhash", "shingle", "minhash"):
    out = tors.dedup_near_dup(corpus, threshold=0.9, method=method)
    assert len(out["kept"]) + len(out["dropped"]) == n, method
print(vmhwm_kib())
"""


# The pure-Python oracles (independent of the new core by construction)
# ---------------------------------------------------------------------------

# The Unicode White_Space property (PropList.txt) — what Rust's
# `char::is_whitespace` matches. Python's `str.isspace()` is deliberately
# NOT used: it also reports C0 U+001C..U+001F as whitespace, which the
# White_Space property does not, so an `isspace` oracle would diverge from
# the tokenizer on those (a divergence this battery pins by including
# U+001C in the corpus).
_WHITE_SPACE = frozenset(
    [*range(0x09, 0x0E), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B),
     0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
)


def fold(text: str) -> str:
    """The grounding layer's matching form: per-character lowercase
    (NEVER ``str.lower()`` on the whole string — that applies the Greek
    final-sigma context rule the core does not), then NFC."""
    return unicodedata.normalize("NFC", "".join(ch.lower() for ch in text))


def real_tokens(text: str) -> list[str]:
    """The crate's one real-word tokenizer, mirrored: UAX #29 word
    segments (``tors.word_bounds``), whitespace-only segments dropped."""
    return [
        text[s:e]
        for s, e in tors.word_bounds(text)
        if not all(ord(ch) in _WHITE_SPACE for ch in text[s:e])
    ]


def shingle_tuples(text: str, width: int) -> set[tuple[str, ...]]:
    """Exact token-tuple shingle set: no hashing, no 64-bit truncation."""
    if width <= 0:
        return set()
    toks = [fold(t) for t in real_tokens(text)]
    if len(toks) < width:
        return set()
    return {tuple(toks[i : i + width]) for i in range(len(toks) - width + 1)}


def oracle_jaccard(a: str, b: str, width: int) -> float:
    sa, sb = shingle_tuples(a, width), shingle_tuples(b, width)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / (len(sa) + len(sb) - inter)


def oracle_dice(a: str, b: str, width: int) -> float:
    sa, sb = shingle_tuples(a, width), shingle_tuples(b, width)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return 2 * inter / (len(sa) + len(sb))


def oracle_hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def oracle_pair_dup(a: str, b: str, threshold: float, method: str) -> bool:
    """One pair's duplicate decision, per method, from public primitives
    and exact set arithmetic only, the same predicate the sweep's
    docstring names. The shingle branch computes the Jaccard from the
    raw shingle-tuple sets (never the pair surface it oracle-checks),
    reproducing the empty-set conventions explicitly."""
    if method == "simhash":
        max_bits = int((1.0 - threshold) * 64)  # floor for t in [0, 1]
        return oracle_hamming(tors.simhash64(fold(a)), tors.simhash64(fold(b))) <= max_bits
    if method == "shingle":
        sa, sb = shingle_tuples(a, 3), shingle_tuples(b, 3)
        if not sa and not sb:
            return True
        if not sa or not sb:
            return threshold == 0.0
        return len(sa & sb) / len(sa | sb) >= threshold
    if method == "minhash":
        sa = tors.minhash_signature(fold(a), num_perm=128, shingle_size=3, seed=0)
        sb = tors.minhash_signature(fold(b), num_perm=128, shingle_size=3, seed=0)
        return sum(x == y for x, y in zip(sa, sb, strict=True)) / 128 >= threshold
    raise AssertionError(method)


def oracle_dedup(texts: list[str], threshold: float, method: str) -> dict:
    """Naive O(n²) greedy keep-first re-dedup: the partition reference."""
    kept: list[int] = []
    dropped: list[int] = []
    groups: list[list[int]] = []
    for i, text in enumerate(texts):
        claim = next(
            (
                ki
                for ki, rep in enumerate(kept)
                if oracle_pair_dup(text, texts[rep], threshold, method)
            ),
            None,
        )
        if claim is None:
            kept.append(i)
            groups.append([i])
        else:
            dropped.append(i)
            groups[claim].append(i)
    return {"kept": kept, "dropped": dropped, "groups": groups}


def window_pair(n: int, off: int) -> tuple[str, str]:
    """Two n-token windows offset by ``off`` over distinct words w0, w1,
    ...: their trigram overlap is exactly (n - off - 2) of (n - 2) each."""
    a = " ".join(f"w{i}" for i in range(n))
    b = " ".join(f"w{i}" for i in range(off, off + n))
    return a, b


# The adversarial corpus: NFC/NFD spellings, the fi ligature (NFC keeps
# it composed — the policy is NFC, NOT NFKC), Turkish dotless-i and
# dotted-capital-I, Greek final sigma (case context), the sharp-s family,
# digraph spellings, CJK/Hangul runs, ZWJ emoji, regional indicators,
# Unicode-White_Space-only separators AND the C0 U+001C Python-says-space
# trap, zero-width characters, pathologically repetitive tokens, and a
# 100k-token row for the scaling shape.
BATTERY = [
    "",
    " ",
    "  \t\n ",
    "a",
    "a b",
    "a b c",
    "a b c d e",
    "a b c d e f g h",
    "café",
    "cafe\u0301",
    "CAFÉ SOCIÉTÉ",
    "café société naïve",
    "ﬁle ﬂow",
    "file flow",
    "Ş",
    "S\u0327",
    "ışık IŞIK",
    "İstanbul istanbulerİ",
    "Ǆ",
    "DŽ",
    "dž ǆ",
    "ΣΊΣΣΥΣΟΣ ΩΣ",
    "ΑΣ ΩΣ",
    "straße",
    "STRASSE",
    "Strasse",
    "東京タワーはランドマーク",
    "東京 タワー",
    "한국어 한국",
    "\U0001f469\u200d\U0001f52c lab",
    "\U0001f469\U0001f52c lab",
    "\U0001f1fa\U0001f1f8usa",
    "a\x1cb",
    "a\u00a0b",
    "a\u200bb",
    "a\u200db",
    "e\u0301℮㎡",
    "ff ﬀ",
    "ﬅ ﬆ",
    "ǅ ǲ Ǳ",
    "ή ΐ",
    "abab abab " * 4,
    "aabab abab aabab",
    "e" * 50,
    " interdisciplinary",
    "The quarterly oil sample interval for field outages",
    "the QUARTERLY oil sample interval for field outages",
    "   \t  ",
    "\u1680\u3000",  # Ogham space + ideographic space: whitespace-only
]

_METHODS = ["simhash", "shingle", "minhash"]

_TEXT = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs", "P"), max_codepoint=0x2FFF),
    max_size=120,
)


# ---------------------------------------------------------------------------
# FINGERPRINT ABUSE — simhash_distance
# ---------------------------------------------------------------------------


class TestSimhashDistanceAbuse:
    def test_zero_zero(self) -> None:
        assert tors.simhash_distance(0, 0) == 0

    def test_max_int_vs_zero_is_refused_by_the_documented_magnitude_rule(self) -> None:
        # 2**128-1 is a legal u128 fingerprint, but 0 classifies as
        # 64-bit and 2**128-1 as 128-bit: the magnitude classification
        # (documented in docs/api.md — "a pair split across that line is
        # refused") raises ValueError rather than returning 128. Pinned
        # as the ACTUAL contract, both orders.
        with pytest.raises(ValueError, match="same width"):
            tors.simhash_distance(2**128 - 1, 0)
        with pytest.raises(ValueError, match="same width"):
            tors.simhash_distance(0, 2**128 - 1)

    def test_width_boundary_at_two_to_the_64(self) -> None:
        # 2**64 exactly is 128-bit territory; 2**64-1 is 64-bit territory.
        assert tors.simhash_distance(2**64, 2**64) == 0
        assert tors.simhash_distance(2**64, 2**64 + 1) == 1
        with pytest.raises(ValueError, match="same width"):
            tors.simhash_distance(2**64 - 1, 2**64)
        # Two 64-bit values compare fine even at the top of the range.
        assert tors.simhash_distance(2**64 - 1, 2**64 - 1) == 0
        assert tors.simhash_distance(2**64 - 1, 0) == 64

    def test_bool_is_a_type_error_in_both_slots(self) -> None:
        for bad in (True, False):
            with pytest.raises(TypeError):
                tors.simhash_distance(bad, 0)
            with pytest.raises(TypeError):
                tors.simhash_distance(0, bad)

    def test_negative_is_a_value_error(self) -> None:
        with pytest.raises(ValueError):
            tors.simhash_distance(-1, 0)
        with pytest.raises(ValueError):
            tors.simhash_distance(0, -(2**70))

    def test_past_u128_is_an_overflow_error(self) -> None:
        for huge in (2**128, 2**128 + 1, 2**200):
            with pytest.raises(OverflowError):
                tors.simhash_distance(huge, 0)
            with pytest.raises(OverflowError):
                tors.simhash_distance(0, huge)

    def test_non_int_arguments_are_type_errors(self) -> None:
        for bad in ("1", 1.0, None, b"\x01", [], [1], 1j):
            with pytest.raises(TypeError):
                tors.simhash_distance(bad, 0)  # type: ignore[arg-type]
            with pytest.raises(TypeError):
                tors.simhash_distance(0, bad)  # type: ignore[arg-type]

    def test_index_protocol_int_likes_and_side_effect_exactly_once(self) -> None:
        class IndexOnly:
            def __init__(self, value: int) -> None:
                self.value = value
                self.calls = 0

            def __index__(self) -> int:
                self.calls += 1
                return self.value

        a, b = IndexOnly(0b1010), IndexOnly(0b0110)
        assert tors.simhash_distance(a, b) == 2
        assert a.calls == 1 and b.calls == 1

    def test_huge_random_ints_agree_with_the_xor_popcount_one_liner(self) -> None:
        rng = random.Random(20260924)
        for _ in range(200):
            widths = (rng.randint(0, 64), rng.randint(0, 128))
            x, y = (rng.getrandbits(w if w == 0 else max(w, 1)) for w in widths)
            if (x >= 2**64) != (y >= 2**64):
                # Only same-magnitude pairs are admissible (documented).
                with pytest.raises(ValueError):
                    tors.simhash_distance(x, y)
            else:
                assert tors.simhash_distance(x, y) == oracle_hamming(x, y)

    def test_distance_bounds_symmetry_and_self(self) -> None:
        rng = random.Random(42)
        for _ in range(50):
            x = rng.getrandbits(64)
            y = rng.getrandbits(64)
            d = tors.simhash_distance(x, y)
            assert 0 <= d <= 128
            assert d == tors.simhash_distance(y, x)
            assert tors.simhash_distance(x, x) == 0

    def test_real_fingerprint_pairs_match_the_one_liner_both_spellings(self) -> None:
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy cat",
            "pack my box with five dozen liquor jugs",
            "",
            "café société",
        ]
        for a, b in product(texts, repeat=2):
            a64, b64 = tors.simhash64(a), tors.simhash64(b)
            assert tors.simhash_distance(a64, b64) == oracle_hamming(a64, b64)
            a128, b128 = tors.simhash128(a), tors.simhash128(b)
            if (a128 >= 2**64) != (b128 >= 2**64):
                # A real 128-bit fingerprint can legitimately fit in 64
                # bits ("" fingerprints to 0): the documented magnitude
                # classification refuses the pair — the honest edge,
                # pinned.
                with pytest.raises(ValueError, match="same width"):
                    tors.simhash_distance(a128, b128)
            else:
                assert tors.simhash_distance(a128, b128) == oracle_hamming(a128, b128)
            # a64 vs b128: refused only when the magnitudes straddle the
            # 2**64 line; two values that both fit 64 bits compare
            # correctly either way (the documented width-blind edge).
            if b128 >= 2**64:
                with pytest.raises(ValueError, match="same width"):
                    tors.simhash_distance(a64, b128)
            else:
                assert tors.simhash_distance(a64, b128) == oracle_hamming(a64, b128)


# ---------------------------------------------------------------------------
# THE MAGNITUDE-CLASSIFICATION TRUTH TABLE (simhash_distance)
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
            expected = _hamming(a, b)
        if expected is ValueError:
            with pytest.raises(ValueError, match="same width"):
                tors.simhash_distance(a, b)
        else:
            assert tors.simhash_distance(a, b) == expected, (a, b)

    def test_the_empty_fingerprint_edge_zero_zero_is_distance_zero(self) -> None:
        # simhash64("") == 0 and simhash128("") == 0: distance(0, 0) must be
        # 0 and must NOT raise (both classify 64-bit; the classification
        # never sees an empty class). Pinned because a "same width" gate
        # implemented as an equality check would refuse this row.
        assert tors.simhash64("") == 0
        assert tors.simhash128("") == 0
        assert tors.simhash_distance(0, 0) == 0

    def test_the_same_texts_two_spellings_compare_through_the_empty_edge(self) -> None:
        # simhash64("") vs simhash128(""): both are 0, both classify 64-bit,
        # so the SAME text's two spellings compare (to 0): the documented
        # width-blind edge taken to its extreme.
        assert tors.simhash_distance(tors.simhash64(""), tors.simhash128("")) == 0

    def test_legal_128_bit_below_two_pow_64_vs_64_bit_compares(self) -> None:
        # The honest edge: a legal 128-bit fingerprint that happens to fit
        # in 64 bits ([2**63, 2**64)) against a 64-bit one in the same
        # range: both classify 64-bit, the comparison goes through
        # width-blind.
        a, b = 2**63 + 1, 2**64 - 1
        assert tors.simhash_distance(a, b) == _hamming(a, b)

    def test_straddle_refusal_is_about_the_line_not_the_width_origin(self) -> None:
        # A 128-bit value in [2**64, 2**65) against a real 64-bit value is
        # refused even though the 64-bit side may itself have come from
        # simhash128; and two values straddling 2**63 (NOT a classification
        # line) compare fine. Together: the gate tracks 2**64 only.
        with pytest.raises(ValueError, match="same width"):
            tors.simhash_distance(2**64 + 1, 2**64 - 1)
        assert tors.simhash_distance(2**63, 2**63 - 1) == _hamming(2**63, 2**63 - 1)

    def test_self_distance_is_never_refused_at_any_magnitude(self) -> None:
        for row in self.ROWS:
            assert tors.simhash_distance(row, row) == 0

    def test_refusal_condition_is_exactly_the_classification_split(self) -> None:
        # The refusal fires IFF the two magnitudes land on opposite sides
        # of 2**64, over a random scan. The "probability 1 - 2**-64"
        # figure for catching the real width mixup assumes fingerprint
        # uniformity (which simhash64's limitations note caveats); the
        # empty fingerprint is the deterministic miss, pinned separately
        # below.
        rng = random.Random(20260924)
        for _ in range(300):
            x, y = rng.getrandbits(128), rng.getrandbits(128)
            refused = (x >= 2**64) != (y >= 2**64)
            if refused:
                with pytest.raises(ValueError):
                    tors.simhash_distance(x, y)
            else:
                assert tors.simhash_distance(x, y) == _hamming(x, y)

    def test_empty_fingerprint_is_the_deterministic_magnitude_miss(self) -> None:
        # The magnitude gate's miss event is a genuine simhash128 value
        # below 2**64: probability 2**-64 only under fingerprint
        # uniformity, which simhash's own limitations section caveats
        # (short text). The empty fingerprint is the DETERMINISTIC miss:
        # simhash128("") == 0 passes as a 64-bit value, so the same
        # text's two spellings compare instead of refusing.
        assert tors.simhash128("") < 2**64
        assert tors.simhash_distance(tors.simhash128(""), 0) == 0  # not refused


# ---------------------------------------------------------------------------
# ORACLE DIFFERENTIALS — shingle_jaccard / shingle_dice vs exact token sets
# ---------------------------------------------------------------------------


class TestShingleOracleDifferential:
    @pytest.mark.parametrize("width", [1, 2, 3, 5])
    def test_battery_cross_product_exact_float_agreement(self, width: int) -> None:
        for a, b in product(BATTERY, repeat=2):
            assert tors.shingle_jaccard(a, b, width=width) == oracle_jaccard(a, b, width), (a, b)
            assert tors.shingle_dice(a, b, width=width) == oracle_dice(a, b, width), (a, b)

    def test_wide_widths_on_random_pairs_agree_exactly(self) -> None:
        rng = random.Random(7)
        pool = [
            " ".join(rng.choice(BATTERY[10:30]) for _ in range(rng.randint(1, 4)))
            for _ in range(30)
        ]
        for width in [1, 4, 7, 12]:
            for _ in range(150):
                a, b = rng.choice(pool), rng.choice(pool)
                assert tors.shingle_jaccard(a, b, width=width) == oracle_jaccard(a, b, width)
                assert tors.shingle_dice(a, b, width=width) == oracle_dice(a, b, width)

    def test_nfc_equivalent_spellings_score_identically_against_every_text(self) -> None:
        nfc = "café société naïve"
        nfd = "cafe\u0301 socie\u0301te\u0301 nai\u0308ve"
        assert nfc != nfd
        for other in BATTERY:
            assert tors.shingle_jaccard(nfc, other) == tors.shingle_jaccard(nfd, other)
            assert tors.shingle_dice(nfc, other) == tors.shingle_dice(nfd, other)

    def test_ligature_policy_is_nfc_not_nfkc(self) -> None:
        # U+FB01 fi is COMPATIBILITY-decomposable only: NFC leaves it, so
        # "ﬁle" and "file" are DIFFERENT tokens — the doc's NFC-only
        # policy, pinned against the oracle (which does the same).
        assert shingle_tuples("ﬁle", 1) == {("ﬁle",)}
        assert shingle_tuples("file", 1) == {("file",)}
        assert (
            tors.shingle_jaccard("ﬁle ﬁn", "file fin", width=1)
            == oracle_jaccard("ﬁle ﬁn", "file fin", width=1)
            == 0.0
        )

    def test_turkish_and_cedilla_rows_match_the_fold_oracle(self) -> None:
        # The core's fold observed through the merge probe (threshold
        # 1.0, shingle method: merge <=> equal folded shingle sets).
        # Rows have >= 3 tokens so the width-3 shingle sets are
        # non-empty (two token-free-for-the-width texts merge trivially
        # under the empty-set convention). The fold rows: S + combining
        # cedilla NFC-composes to U+015E; the dotted capital I
        # lowercases to i + combining dot; dotless i is a distinct
        # letter; the digraph spellings are distinct tokens.
        rows = [
            ("S\u0327 ışık a", "Ş ışık a", True),
            ("I ışık a", "ı ışık a", False),
            ("İstanbul ışık a", "i̇stanbul ışık a", True),
            ("ΣΣ ΣΣ ΣΣ", "ΣΣ ΣΣ ΣΣ", True),
            ("Ǆ ǆ a", "DŽ dž a", False),
        ]
        for x, y, expect_merge in rows:
            merged = tors.dedup_near_dup([x, y], threshold=1.0, method="shingle")["kept"] == [0]
            assert merged == expect_merge, (x, y, merged)
            assert merged == (shingle_tuples(x, 3) == shingle_tuples(y, 3)), (x, y)

    def test_merge_probes_agree_with_the_oracle_over_the_whole_battery(self) -> None:
        # The core's fold observed through its merge decisions at
        # threshold 1.0 must EXACTLY equal the oracle's shingle-set
        # equality, for every method: any fold/tokenizer/hash divergence
        # between the core and the grounding policy fires here.
        for a, b in product(BATTERY, repeat=2):
            expected = shingle_tuples(a, 3) == shingle_tuples(b, 3)
            assert (
                tors.dedup_near_dup([a, b], threshold=1.0, method="shingle")["kept"] == [0]
            ) == expected, (a, b)
            sim_expected = tors.simhash64(fold(a)) == tors.simhash64(fold(b))
            assert (
                tors.dedup_near_dup([a, b], threshold=1.0, method="simhash")["kept"] == [0]
            ) == sim_expected, (a, b)
            sig_a = tors.minhash_signature(fold(a), num_perm=128, shingle_size=3, seed=0)
            sig_b = tors.minhash_signature(fold(b), num_perm=128, shingle_size=3, seed=0)
            assert (
                tors.dedup_near_dup([a, b], threshold=1.0, method="minhash")["kept"] == [0]
            ) == (sig_a == sig_b), (a, b)

    def test_width_edges(self) -> None:
        # 0 / negative: ValueError; bool: TypeError; huge-but-legal on a
        # short stream: the empty-set convention (both sides empty = 1.0).
        for bad in (0, -1, -10**18):
            with pytest.raises(ValueError, match="width must be at least 1"):
                tors.shingle_jaccard("a b c", "a b c", width=bad)
            with pytest.raises(ValueError, match="width must be at least 1"):
                tors.shingle_dice("a b c", "a b c", width=bad)
        for bad in (True, False):
            with pytest.raises(TypeError):
                tors.shingle_jaccard("a b c", "a b c", width=bad)
        assert tors.shingle_jaccard("a b c", "a b c", width=10**9) == 1.0
        assert tors.shingle_jaccard("a b c", "x y z", width=10**9) == 1.0
        with pytest.raises(OverflowError):
            tors.shingle_jaccard("a b c", "a b c", width=2**63)

    def test_width_past_the_budget_gate_is_refused_before_any_work(self) -> None:
        text = "w " * 20_000
        with pytest.raises(ValueError, match="token-hash budget"):
            tors.shingle_jaccard(text, text, width=5_000)
        with pytest.raises(ValueError, match="token-hash budget"):
            tors.shingle_dice(text, text, width=5_000)

    def test_lone_surrogates_are_unicode_encode_errors(self) -> None:
        with pytest.raises(UnicodeEncodeError):
            tors.shingle_jaccard("\ud800", "a b c")
        with pytest.raises(UnicodeEncodeError):
            tors.shingle_dice("a b c", "\udfff")
        with pytest.raises(UnicodeEncodeError):
            tors.dedup_near_dup(["a b c", "\ud800"])

    def test_100k_token_input_completes_and_scores_plausibly(self) -> None:
        words = [f"w{i % 500}" for i in range(100_000)]
        a = " ".join(words)
        b = " ".join(words[:-1]) + " zzz"
        j = tors.shingle_jaccard(a, b)
        assert 0.9 < j < 1.0
        assert j == oracle_jaccard(a, b, 3)


# ---------------------------------------------------------------------------
# SHINGLE-WIDTH EXTREMES
# ---------------------------------------------------------------------------


class TestShingleWidthExtremes:
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
        # from the extraction (NOT a ValueError; the boundary is the int
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
# DEDUP ORACLE DIFFERENTIAL + PARTITION INVARIANTS
# ---------------------------------------------------------------------------


class TestDedupOracleDifferential:
    @pytest.mark.parametrize("method", _METHODS)
    def test_adversarial_corpora_match_the_naive_reference_exactly(self, method: str) -> None:
        corpora = [
            ["a b c d e", "x b c d e", "a b c d e"],  # exact dup + boundary pair
            ["", "   ", "\t\n", "a b c", ""],  # token-free mixes
            ["w1 w2 w3 w4 w5 w6", "w4 w5 w6 w7 w8 w9", "w7 w8 w9 w10 w11 w12"],  # chain
            ["tok_a tok_b tok_c", "tok_a tok_b tok_c tok_d", "tok_x tok_y tok_z"],
            ["café société", "CAFE\u0301 SOCI\u0301T\u0301E", "ﬁn"],
        ]
        for corpus in corpora:
            for threshold in (0.0, 0.3, 0.5, 0.8, 0.9, 1.0):
                core = tors.dedup_near_dup(corpus, threshold=threshold, method=method)
                assert core == oracle_dedup(corpus, threshold, method), (corpus, threshold)

    @given(texts=st.lists(st.text(min_size=0, max_size=30), max_size=9))
    @settings(max_examples=40, deadline=None)
    def test_random_corpora_match_the_naive_reference_exactly(self, texts: list[str]) -> None:
        for method in _METHODS:
            for threshold in (0.0, 0.4, 0.85, 1.0):
                core = tors.dedup_near_dup(texts, threshold=threshold, method=method)
                assert core == oracle_dedup(texts, threshold, method), (texts, threshold, method)

    @given(texts=st.lists(st.text(min_size=0, max_size=25), max_size=8))
    @settings(max_examples=30, deadline=None)
    def test_partition_invariants_and_member_head_pairs(self, texts: list[str]) -> None:
        for method in _METHODS:
            result = tors.dedup_near_dup(texts, threshold=0.7, method=method)
            n = len(texts)
            # kept ∪ dropped == every index exactly once.
            assert sorted(result["kept"] + result["dropped"]) == list(range(n))
            assert sorted(i for g in result["groups"] for i in g) == list(range(n))
            # kept == group heads == group minima, ascending.
            assert result["kept"] == [g[0] for g in result["groups"]]
            assert result["kept"] == sorted(result["kept"])
            assert result["dropped"] == sorted(result["dropped"])
            # Greedy keep-first: every dropped text IS within threshold
            # of ITS OWN group's representative (the first kept text it
            # matched), verified with the independent pair oracle.
            head_of = {i: g[0] for g in result["groups"] for i in g}
            for i in result["dropped"]:
                head = head_of[i]
                assert oracle_pair_dup(texts[i], texts[head], 0.7, method), (method, i, head, texts)

    @given(texts=st.lists(st.text(min_size=1, max_size=25), min_size=1, max_size=7))
    @settings(max_examples=30, deadline=None)
    def test_drop_count_is_monotone_under_a_rising_threshold(self, texts: list[str]) -> None:
        # The count-level monotonicity the implementer's curated battery
        # pins, re-verified over random corpora at a fine threshold
        # ladder (the dropped SETS are intentionally NOT asserted —
        # greedy keep-first re-associates chains, the docs' position).
        ladder = [0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.95, 1.0]
        for method in _METHODS:
            counts = [
                len(tors.dedup_near_dup(texts, threshold=t, method=method)["dropped"])
                for t in ladder
            ]
            assert all(
                b <= a for a, b in zip(counts, counts[1:], strict=False)
            ), (method, texts, counts)


class TestThresholdExtremesAndBoundaries:
    @pytest.mark.parametrize("method", _METHODS)
    @pytest.mark.parametrize("threshold", [0.0, 1.0])
    def test_extremes_are_defined(self, method: str, threshold: float) -> None:
        corpus = ["alpha beta gamma", "delta epsilon zeta", "alpha beta gamma"]
        result = tors.dedup_near_dup(corpus, threshold=threshold, method=method)
        assert sorted(result["kept"] + result["dropped"]) == list(range(3))
        if threshold == 0.0:
            # Everything matches everything: keep the first only.
            assert result["kept"] == [0]
            assert result["dropped"] == [1, 2]
        else:
            # 1.0: only fingerprint-identical texts merge.
            assert result["kept"] == [0, 1]
            assert result["dropped"] == [2]

    def test_scores_at_exactly_the_threshold_are_inclusive(self) -> None:
        # A pair whose trigram Jaccard is EXACTLY 0.5: "a b c d e" and
        # "x b c d e" share {bcd, cde} of {abc, bcd, cde} -> 2/4.
        x, y = "a b c d e", "x b c d e"
        assert tors.shingle_jaccard(x, y, width=3) == 0.5
        assert tors.shingle_jaccard(x, y, width=3) >= 0.5
        assert tors.dedup_near_dup([x, y], threshold=0.5, method="shingle")["kept"] == [0]
        assert tors.dedup_near_dup([x, y], threshold=0.5000001, method="shingle")["kept"] == [0, 1]

    def test_simhash_threshold_maps_to_the_documented_bit_budget(self) -> None:
        # threshold 1.0 -> 0 bits: fingerprint-identical texts merge at
        # 1.0 and folding-independent respellings merge with them (the
        # dedup fingerprint folds first).
        base = "the quarterly oil sample interval was adjusted"
        upper = "THE QUARTERLY OIL SAMPLE INTERVAL WAS ADJUSTED"
        assert tors.dedup_near_dup([base, upper], threshold=1.0, method="simhash")["kept"] == [0]
        # And a differing fingerprint does NOT merge at 1.0.
        other = "a completely different document about volcanic rock"
        d = tors.simhash_distance(tors.simhash64(fold(base)), tors.simhash64(fold(other)))
        if d > 0:
            kept = tors.dedup_near_dup([base, other], threshold=1.0, method="simhash")["kept"]
            assert kept == [0, 1]

    def test_minhash_boundary_inclusivity(self) -> None:
        # Identical signatures agree on all 128 positions: dup at any
        # threshold incl. 1.0; a one-position difference is 127/128 and
        # must be refused at threshold just above 127/128, accepted at
        # exactly 127/128.
        a = "the lighthouse keeper walked the stone steps"
        assert tors.dedup_near_dup([a, a], threshold=1.0, method="minhash")["kept"] == [0]
        sig_a = tors.minhash_signature(fold(a), num_perm=128, shingle_size=3, seed=0)
        assert len(set(sig_a)) > 64  # sanity: the corpus fixture is not degenerate


class TestOrderSensitivityPinned:
    """Greedy keep-first's documented semantics: input order is the
    tie-break, so for a CHAIN a~b, b~c, a≁c the kept-text SET depends
    on the input order. Pinned here so the behavior is a contract, not
    an accident (the docs promise "the first of a duplicate family
    survives" — which is exactly what this shows)."""

    A, B, C = (" ".join(f"w{i}" for i in range(0, 10)),
               " ".join(f"w{i}" for i in range(5, 15)),
               " ".join(f"w{i}" for i in range(10, 20)))
    T = 3 / 13  # exact J(A, B) == J(B, C) == 3/13, J(A, C) == 0

    def test_chain_geometry(self) -> None:
        w = 3
        assert tors.shingle_jaccard(self.A, self.B, width=w) == pytest.approx(3 / 13, rel=1e-15)
        assert tors.shingle_jaccard(self.B, self.C, width=w) == pytest.approx(3 / 13, rel=1e-15)
        assert tors.shingle_jaccard(self.A, self.C, width=w) == 0.0

    def test_order_changes_which_texts_survive(self) -> None:
        assert tors.dedup_near_dup(
            [self.A, self.B, self.C], threshold=self.T, method="shingle"
        ) == {
            "kept": [0, 2],
            "dropped": [1],
            "groups": [[0, 1], [2]],
        }
        assert tors.dedup_near_dup(
            [self.B, self.A, self.C], threshold=self.T, method="shingle"
        ) == {
            "kept": [0],
            "dropped": [1, 2],
            "groups": [[0, 1, 2]],
        }

    def test_exact_duplicate_families_keep_the_same_texts_under_permutation(self) -> None:
        # For mutually-matching families the kept TEXT SET is invariant.
        families = [
            ["alpha beta gamma delta"] * 3,
            ["epsilon zeta eta theta"] * 2,
            ["a solo document stands alone"],
        ]
        corpus = [t for fam in families for t in fam]
        rng = random.Random(11)
        dedup = tors.dedup_near_dup(corpus, threshold=0.9, method="shingle")
        expected_texts = {corpus[i] for i in dedup["kept"]}
        for _ in range(5):
            permuted = corpus[:]
            rng.shuffle(permuted)
            dedup = tors.dedup_near_dup(permuted, threshold=0.9, method="shingle")
            kept_texts = {permuted[i] for i in dedup["kept"]}
            assert kept_texts == expected_texts

    def test_determinism_same_order_same_result(self) -> None:
        corpus = BATTERY * 2
        first = tors.dedup_near_dup(corpus, threshold=0.6, method="simhash")
        assert first == tors.dedup_near_dup(list(corpus), threshold=0.6, method="simhash")
        assert first == tors.dedup_near_dup(corpus[:], threshold=0.6, method="simhash")


# ---------------------------------------------------------------------------
# GROUPS STRUCTURE ABUSE: STAR SHAPE, NOT TRANSITIVE CLOSURE
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
                    assert oracle_pair_dup(texts[i], texts[head_of[i]], threshold, method)
                # ... and no group is a clique requirement: members may be
                # far apart (checked on the chains above), but a head must
                # never claim a text it does not match (the loop above is
                # that check, run through the independent pair oracle).


# ---------------------------------------------------------------------------
# THE >= BOUNDARY UNDER F64: THE DOCUMENTED FLOAT READING
# ---------------------------------------------------------------------------


class TestGeqBoundaryUnderF64:
    """The shingle method's threshold contract at the f64 dust cells: the
    score is an f64 division and the threshold comparison is float. A pair
    whose exact Jaccard rounds up to exactly the threshold merges; the
    error is at most one ulp in the merge direction, never the data-loss
    direction (a pair whose exact Jaccard is at or above the threshold is
    never refused). The dust pairs below all round UP, so threshold ==
    fl(J) merges though the exact rational is strictly below it, and the
    Fraction oracle over the threshold ladder pins that no under-merge
    ever fires and every divergence is exactly that 1-ulp over-merge. The
    simhash bit-budget map floor((1-t)*64) and the MinHash map ceil(t*128)
    are dust-free (Sterbenz + power-of-two scaling / dyadic denominators),
    so the float reading has no other edge."""

    # Window pairs with EXACT trigram Jaccards 3/13, 5/13, 5/7; every one
    # rounds UP to its nearest double (verified below).
    DUST_PAIRS = [
        (3, 13, window_pair(10, 5)),
        (5, 13, window_pair(11, 4)),
        (5, 7, window_pair(8, 1)),
    ]

    def test_dust_pairs_have_the_exact_fraction_and_round_up(self) -> None:
        for p, q, (a, b) in self.DUST_PAIRS:
            sa, sb = shingle_tuples(a, 3), shingle_tuples(b, 3)
            inter, union = len(sa & sb), len(sa | sb)
            assert (inter, union) == (p, q), (p, q, inter, union)
            assert Fraction(p / q) > Fraction(p, q), (p, q)

    @pytest.mark.parametrize("p,q,pair", [(3, 13, 0), (5, 13, 1), (5, 7, 2)])
    def test_threshold_equal_to_the_rounded_up_double_merges(
        self, p: int, q: int, pair: int
    ) -> None:
        # The documented float reading at the dust cell: threshold ==
        # fl(J) == the score itself, so the f64 >= merges, though the
        # exact Jaccard p/q is strictly below the threshold's rational
        # value. The shingle core, the dedup sweep, and the direct score
        # all agree with each other and with the docs.
        _, _, (a, b) = self.DUST_PAIRS[pair]
        tau = p / q
        assert Fraction(p, q) < Fraction(tau)
        assert tors.shingle_jaccard(a, b, width=3) == tau
        assert tors.dedup_near_dup([a, b], threshold=tau, method="shingle")["kept"] == [0]

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
        sa, sb = shingle_tuples(a, 3), shingle_tuples(b, 3)
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
                # fl(J) rounds UP, a 1-ulp over-merge, never more.
                assert tau == inter / union, (inter, union, tau)
                assert Fraction(tau) > j_exact

    def test_simhash_bit_budget_map_is_dust_free(self) -> None:
        # floor((1-t)*64) must equal the exact rational floor. For t >= 0.5
        # this is PROVABLE (Sterbenz: 1-t is exact; *64 is exact scaling),
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
# METHOD CONSISTENCY
# ---------------------------------------------------------------------------


class TestMethodConsistency:
    def test_unknown_method_names_every_choice(self) -> None:
        for bad in ("bogus", "", "SIMHASH", "dice", "simhash\x00"):
            with pytest.raises(ValueError, match=r"valid choices are: simhash, shingle, minhash"):
                tors.dedup_near_dup(["a b c"], method=bad)

    def test_no_method_is_a_silent_no_op_or_an_always_drop(self) -> None:
        # One near-dup family, one far document: EVERY method must drop
        # the family's variants at a calibrated threshold and keep the
        # far document — a method that never (or always) dropped would
        # have a broken score path.
        family = [
            "the quarterly oil sample interval was adjusted after the audit",
            "the quarterly oil sample interval was adjusted after the Audit.",
            "the quarterly oil sample interval was adjusted after the audits",
        ]
        far = ["a completely different memo about parking garage resurfacing fees"]
        for method, threshold in [("simhash", 0.8), ("shingle", 0.6), ("minhash", 0.7)]:
            result = tors.dedup_near_dup(family + far, threshold=threshold, method=method)
            assert result["kept"] == [0, 3], (method, result)
            assert result["dropped"] == [1, 2]
            assert result["groups"] == [[0, 1, 2], [3]]

    def test_documented_disagreement_is_bounded_not_pathological(self) -> None:
        # The word-permutation pair: simhash (bag of words) merges,
        # shingle (trigram order) does not — a DOCUMENTED divergence,
        # pinned; minhash tracks shingle's reading here.
        a = "one two three four five six seven eight nine ten"
        b = "ten two three four five six seven eight nine one"
        assert tors.dedup_near_dup([a, b], threshold=0.9, method="simhash")["kept"] == [0]
        assert tors.dedup_near_dup([a, b], threshold=0.9, method="shingle")["kept"] == [0, 1]

    def test_threshold_type_liberality_is_pinned(self) -> None:
        # pyo3's f64 slot accepts Python ints (and, unlike the __index__
        # slots, launders bool): pinned as ACTUAL behavior — the crate's
        # "bool rejected in every position" discipline applies to the
        # int slots only.
        assert tors.dedup_near_dup(["a b c", "a b c"], threshold=1)["kept"] == [0]
        assert tors.dedup_near_dup(["a b c", "a b c"], threshold=True)["kept"] == [0]
        for bad in (1.5, -0.1, float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="threshold must be in"):
                tors.dedup_near_dup(["a b c"], threshold=bad)


# ---------------------------------------------------------------------------
# DEGENERATE INPUTS
# ---------------------------------------------------------------------------


class TestDegenerateInputs:
    def test_empty_corpus(self) -> None:
        assert tors.dedup_near_dup([]) == {"kept": [], "dropped": [], "groups": []}

    @pytest.mark.parametrize("method", _METHODS)
    def test_token_free_texts_dedup_together(self, method: str) -> None:
        # [""] and ["", ""] and whitespace-only rows: two token-free
        # texts are duplicates under EVERY method (the empty-set
        # convention: fingerprints all 0 / sets empty / all-sentinel
        # signatures).
        for texts in ([""], ["", ""], ["   ", "\t\n"], ["", "   ", " "]):
            result = tors.dedup_near_dup(texts, threshold=0.9, method=method)
            assert result["kept"] == [0], (method, texts, result)
            assert result["groups"] == [list(range(len(texts)))]

    @pytest.mark.parametrize("method", _METHODS)
    def test_single_token_free_vs_single_real_text(self, method: str) -> None:
        # One token-free + one real text: NOT duplicates at 0.9 (the
        # exactly-one-empty 0.0 row), for every method.
        result = tors.dedup_near_dup(["", "real text here"], threshold=0.9, method=method)
        assert result["kept"] == [0, 1]
        assert result["dropped"] == []

    @pytest.mark.parametrize("method", _METHODS)
    def test_single_text_and_identical_texts(self, method: str) -> None:
        assert tors.dedup_near_dup(["only one"], method=method)["kept"] == [0]
        result = tors.dedup_near_dup(["same words here"] * 7, threshold=0.5, method=method)
        assert result["kept"] == [0]
        assert result["dropped"] == list(range(1, 7))
        assert result["groups"] == [list(range(7))]

    def test_tuple_is_accepted_and_generator_rejected(self) -> None:
        # pyo3's sequence extraction accepts any Sequence: a tuple is
        # liberal acceptance (stub says list[str]) — pinned actual.
        assert tors.dedup_near_dup(("a b c", "a b c")) == {
            "kept": [0],
            "dropped": [1],
            "groups": [[0, 1]],
        }
        with pytest.raises(TypeError):
            tors.dedup_near_dup(x for x in ["a b c"])


# ---------------------------------------------------------------------------
# WIDTH-AGNOSTIC DEDUP MIXING + THE TOKEN-FREE CONVENTION'S ASYMMETRY
# ---------------------------------------------------------------------------


class TestDedupWidthMixingAndTokenFree:
    def test_no_width_classification_exists_inside_the_dedup_sweep(self) -> None:
        # dedup fingerprints internally with the u64 simhash only, so the
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
        # The empty-set convention's "exactly one empty side scores 0.0"
        # is a SHINGLE/MINHASH special case only (the docs own the
        # asymmetry). The simhash method fingerprints "" as 0 and merges
        # it with a real text whenever popcount(fp(text)) fits the bit
        # budget, demonstrable at a lowered threshold, unreachable in
        # practice at 0.9 (the measured margin below).
        text = "the quick brown fox jumps over the lazy dog"
        p = tors.simhash64(text).bit_count()
        threshold = 1.0 - (p + 0.5) / 64  # bit budget exactly p
        for method, expect in [("simhash", [0]), ("shingle", [0, 1]), ("minhash", [0, 1])]:
            out = tors.dedup_near_dup(["", text], threshold=threshold, method=method)
            assert out["kept"] == expect, (method, out)

    def test_measured_margin_no_small_text_fingerprints_within_the_09_budget(self) -> None:
        # The practical safety of the row above: over a large sample of
        # short texts, no simhash64 fingerprint is within the threshold
        # 0.9 budget (6 bits) of the empty fingerprint (sampled evidence,
        # not a proof (the convention asymmetry is the documented point).
        rng = random.Random(11)
        vocab = [f"w{i}" for i in range(60)]
        for _ in range(20_000):
            text = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 6)))
            assert tors.simhash64(fold(text)).bit_count() > 6, text

    def test_fewer_tokens_than_the_width_is_token_free_only_for_the_shingle_method(self) -> None:
        # A 2-token text has an EMPTY trigram set (the empty-set convention)
        # but a REAL simhash fingerprint, so it joins the token-free family
        # under the shingle method and stands alone under simhash.
        two_token, real = "alpha beta", "gamma delta epsilon zeta eta theta"
        sh = tors.dedup_near_dup(["", "  ", two_token, real], threshold=0.9, method="shingle")
        assert sh["groups"][0] == [0, 1, 2], sh
        si = tors.dedup_near_dup(["", "  ", two_token, real], threshold=0.9, method="simhash")
        assert si["groups"][0] == [0, 1], si


# ---------------------------------------------------------------------------
# GIL / AIO TWINS
# ---------------------------------------------------------------------------


class TestGilHeartbeat:
    @staticmethod
    async def _worst_gap(op) -> tuple[float, float]:
        """Run ``op`` as a task while the loop pings every 10ms; return
        (worst tick gap, total wall)."""
        gaps: list[float] = []
        started = monotonic()
        last = started

        async def heartbeat() -> None:
            nonlocal last
            while True:
                await asyncio.sleep(0.01)
                now = monotonic()
                gaps.append(now - last)
                last = now

        hb = asyncio.create_task(heartbeat())
        try:
            await op()
        finally:
            hb.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass
        wall = monotonic() - started
        return (max(gaps) if gaps else 0.0), wall

    def test_aio_dedup_near_dup_keeps_the_loop_responsive(self) -> None:
        corpus = [" ".join(f"tok{i}_{j}" for j in range(40)) for i in range(8_000)]

        async def run() -> None:
            gap, wall = await self._worst_gap(
                lambda: tors.aio.dedup_near_dup(corpus, threshold=0.9, method="simhash")
            )
            assert gap < max(0.05, 0.3 * wall), (
                f"loop blocked {gap * 1e3:.0f}ms of {wall * 1e3:.0f}ms"
            )

        asyncio.run(run())

    def test_aio_shingle_jaccard_keeps_the_loop_responsive(self) -> None:
        text_a = ("the quarterly oil sample interval for field outages " * 8_000).strip()
        text_b = text_a.replace("quarterly", "QUARTERLY", 1)

        async def run() -> None:
            gap, wall = await self._worst_gap(
                lambda: tors.aio.shingle_jaccard(text_a, text_b)
            )
            assert gap < max(0.05, 0.3 * wall), (
                f"loop blocked {gap * 1e3:.0f}ms of {wall * 1e3:.0f}ms"
            )

        asyncio.run(run())

    def test_aio_twins_agree_with_the_sync_surface_and_propagate_errors(self) -> None:
        corpus = ["alpha beta gamma delta", "ALPHA BETA GAMMA delta", "wholly unrelated words here"]
        sync = tors.dedup_near_dup(corpus, threshold=0.9, method="simhash")
        twin = asyncio.run(tors.aio.dedup_near_dup(corpus, threshold=0.9, method="simhash"))
        assert sync == twin
        j = tors.shingle_jaccard("a b c d", "a b c e", width=2)
        assert asyncio.run(tors.aio.shingle_jaccard("a b c d", "a b c e", width=2)) == j
        with pytest.raises(ValueError, match="valid choices"):
            asyncio.run(tors.aio.dedup_near_dup(["a b c"], method="bogus"))
        with pytest.raises(ValueError, match="threshold must be in"):
            asyncio.run(tors.aio.dedup_near_dup(["a b c"], threshold=1.5))

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
# MEMORY / SCALE GUARDS
# ---------------------------------------------------------------------------


class TestMemoryAndScale:
    def test_ten_k_corpus_peak_memory_stays_tied_to_the_input(self) -> None:
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
                    "corpus = [' '.join(f'tok{i}_{j}' for j in range(40)) for i in range(n)]\n"
                    "for method in ('simhash', 'shingle', 'minhash'):\n"
                    "    out = tors.dedup_near_dup(corpus, threshold=0.9, method=method)\n"
                    "    assert len(out['kept']) + len(out['dropped']) == n\n"
                    "print(vmhwm())\n"
                ),
                "10000",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert child.returncode == 0, child.stderr
        # ~0.7 MiB of corpus text: the peak must be interpreter baseline
        # plus O(input), nowhere near an n²-sized allocation (~1.5 GiB of
        # pair state at 10k would blow this by 10x).
        assert int(child.stdout.strip()) < 200 * 1024, child.stdout

    def test_ten_k_all_identical_one_exploding_group_oom_shape(self) -> None:
        # The other n=10k extreme (the all-distinct corpus is pinned
        # above): EVERY text joins ONE group, the group list is one
        # 10k-element index list, and the sweep early-exits per text, so
        # the peak stays tied to the input, nowhere near an n²
        # allocation.
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

    def test_documented_candidate_set_completes_well_under_the_budget(self) -> None:
        corpus = [" ".join(f"tok{i}_{j}" for j in range(40)) for i in range(1_000)]
        for method in _METHODS:
            # Load-robust spelling (tests/loop_harness.py): min-of-3
            # pass-on-first-clean per method; a regression past the 2s
            # budget inflates every sample.
            assert_bounded(
                lambda method=method: tors.dedup_near_dup(corpus, threshold=0.9, method=method),
                2.0,
                samples=3,
                label=f"the {method} candidate-set sweep",
            )


class TestScalingGateBites:
    """The pin-bite proof: the growth-gate arithmetic
    tests/test_scaling_pins.py uses MUST trip a regression past the
    documented class. The dedup sweep's documented cost class IS
    quadratic (the pin's gate admits it), so the injection here is a
    CUBIC add-on — worse than quadratic is the defect the pin exists to
    catch, and the gate must blow through on it. The injection is a
    pure-Python wrapper around the real call (fixing nothing in Rust —
    this validates the GATE, the same formula the shipped pin relies
    on)."""

    @staticmethod
    def _gate(small_ms: float, large_ms: float, factor: int, gate: float) -> float:
        import math

        return gate ** math.log2(factor) * small_ms

    def test_cubic_addon_blows_through_the_quadratic_class_gate(self) -> None:
        def corpus(n: int) -> list[str]:
            return [" ".join(f"tok{i}_{j}" for j in range(40)) for i in range(n)]

        def wall(fn, samples: int = 3) -> float:
            fn()
            best = float("inf")
            for _ in range(samples):
                started = monotonic()
                fn()
                best = min(best, monotonic() - started)
            return best * 1e3

        def honest(texts):
            tors.dedup_near_dup(texts)

        def regressed(texts):
            # An injected O(n³) add-on (full triple loop): the defect
            # shape the quadratic-class gate exists to catch.
            n = len(texts)
            acc = 0
            for i in range(n):
                for j in range(n):
                    for _k in range(n):
                        acc += i & j
            honest(texts)
            assert acc >= 0

        # 2x input (one doubling): a cubic's 8x growth clears the 4.5x
        # gate; a quadratic's 4x does not (the documented class).
        small, large = corpus(200), corpus(400)
        honest_small, honest_large = wall(lambda: honest(small)), wall(lambda: honest(large))
        assert honest_large < self._gate(honest_small, honest_large, 2, 4.5), (
            "the honest call must pass its own gate"
        )
        reg_small, reg_large = wall(lambda: regressed(small), samples=1), wall(
            lambda: regressed(large), samples=1
        )
        assert reg_large > self._gate(reg_small, reg_large, 2, 4.5), (
            "the gate MUST trip a worse-than-quadratic add-on: "
            f"{reg_small:.1f}ms -> {reg_large:.1f}ms"
        )
