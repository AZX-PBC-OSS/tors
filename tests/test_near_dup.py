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

import subprocess
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors

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
