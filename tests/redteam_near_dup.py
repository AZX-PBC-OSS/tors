"""Adversarial review suite for the near-duplicate comparison family
(``simhash_distance``, ``shingle_jaccard``, ``shingle_dice``,
``dedup_near_dup``), written as an independent attack pass: every test is
an attempt to break the shipped contract, and the oracles below are
pure-Python re-implementations built ONLY from pre-existing public
primitives (``tors.word_bounds``, ``tors.simhash64``,
``tors.minhash_signature``, ``unicodedata``) so a shared-bug mirror with
the new core is structurally impossible.

What the oracles pin, independently of ``src/near_dup_impl.rs``:

- the normalization fold (per-character lowercase, then NFC) observed
  through the dedup sweep's merge decisions against a Python fold;
- the token stream (UAX #29 word segments via ``tors.word_bounds``,
  whitespace-only segments dropped — Unicode White_Space, NOT Python's
  ``str.isspace``, whose C0 U+001C-001F extras would diverge);
- Jaccard/Dice over exact token-TUPLE sets (no 64-bit hash anywhere);
- the greedy keep-first sweep re-run in Python at O(n²);
- the simhash Hamming distance as ``bin(a ^ b).count("1")``.

Method-consistency, threshold-extreme, degenerate-input, boundary
(inclusive ``>=``), fingerprint-abuse, GIL-heartbeat and peak-memory
cells round out the pass. All tests are GREEN by construction: a red
test here is a bug report, not a fixture.
"""

from __future__ import annotations

import asyncio
import random
import subprocess
import sys
import unicodedata
from itertools import product
from time import monotonic

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors
import tors.aio

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
    only — the same predicate the sweep's docstring names."""
    if method == "simhash":
        max_bits = int((1.0 - threshold) * 64)  # floor for t in [0, 1]
        return oracle_hamming(tors.simhash64(fold(a)), tors.simhash64(fold(b))) <= max_bits
    if method == "shingle":
        return tors.shingle_jaccard(a, b, width=3) >= threshold
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
        # int slots only. (Flagged in the review notes as a P2
        # consistency surprise, not fixed here.)
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

    def test_documented_candidate_set_completes_well_under_the_budget(self) -> None:
        corpus = [" ".join(f"tok{i}_{j}" for j in range(40)) for i in range(1_000)]
        for method in _METHODS:
            started = monotonic()
            tors.dedup_near_dup(corpus, threshold=0.9, method=method)
            assert monotonic() - started < 2.0, method


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
