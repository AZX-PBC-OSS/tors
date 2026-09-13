"""Contract gate for ``tors.minhash_signature``: the recall-side complement
to the SimHash family. SimHash (``simhash64``/``simhash128``) is the
precision-side near-dup gate -- one fingerprint, small Hamming distance for
near-dupes; MinHash is the recall side at corpus scale -- a signature of
``num_perm`` min-hashes whose agreement fraction estimates the Jaccard
similarity of the two documents' shingle sets, the quantity an LSH-banding
index (caller state; tors stays stateless) buckets candidates on.

The pinned arithmetic (``src/minhash_impl.rs``'s module doc is the full
writeup): tokens are the tokenizer's UAX #29 word segments (whitespace-only
segments skipped, lowercased); shingles are consecutive ``shingle_size``-
token windows joined with U+001F; each shingle is hashed with XXH64 (seed 0,
the frozen-spec algorithm via twox-hash; the differential oracle hashes the
same shingles through the pinned ``xxhash`` package wrapping the C reference
implementation, so agreement is evidence about two independent
implementations of one frozen spec); ``signature[i] = min over shingles of
(a_i * x + b_i) mod (2^61 - 1)`` with the ``(a_i, b_i)`` pairs derived from
``seed`` by a SplitMix64 stream (fixture-grade determinism, not crypto).
Empty text or fewer tokens than ``shingle_size`` is the empty-shingle-set
convention: every element the u64 MAX sentinel (2^64 - 1, outside the
affine maps' [0, 2^61 - 1) output range, so it can never collide with a
real minimum) -- a stable digest for empty documents.
"""

from __future__ import annotations

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reference import (
    _MINHASH_EMPTY,
    reference_minhash_signature,
    reference_minhash_tokens,
)
from tors import minhash_signature

# The mixed text alphabet: letters, numbers, punctuation, spaces, plus the
# hard-break whitespace (excludes surrogates by construction and controls
# below 0x2000 other than the explicitly whitelisted \n\r\t; the U+001F
# separator row and the surrogate row are pinned as explicit fixtures).
_TEXT = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Zs", "P"),
        whitelist_characters="\n\r\t",
        max_codepoint=0x2FFF,
    ),
    max_size=200,
)
_NUM_PERM = st.integers(min_value=1, max_value=32)
_SHINGLE_SIZE = st.integers(min_value=1, max_value=4)
_SEED = st.one_of(
    st.integers(min_value=-(2**64), max_value=2**64),
    st.sampled_from([0, -1, 1, 2**63, 2**64 - 1, -(2**63) - 1]),
)

_FOX = "the quick brown fox jumps over the lazy dog"


def _agreement(a: list[int], b: list[int]) -> int:
    return sum(x == y for x, y in zip(a, b, strict=True))


class TestOracleDifferential:
    """tors vs the transcribed pure-Python oracle (``reference.py``): every
    element of the pipeline but the UAX #29 walk itself is pinned here --
    the whitespace skip, the lowercase fold, the shingle join, the XXH64
    shingle hash, the SplitMix64 coefficient derivation, and the
    Mersenne-affine min-sweep."""

    @given(text=_TEXT, num_perm=_NUM_PERM, shingle_size=_SHINGLE_SIZE, seed=_SEED)
    @settings(max_examples=200)
    def test_matches_the_oracle(
        self, text: str, num_perm: int, shingle_size: int, seed: int
    ) -> None:
        assert minhash_signature(
            text, num_perm=num_perm, shingle_size=shingle_size, seed=seed
        ) == reference_minhash_signature(
            text, num_perm=num_perm, shingle_size=shingle_size, seed=seed
        )

    @given(text=_TEXT, seed=_SEED)
    @settings(max_examples=50)
    def test_matches_the_oracle_at_the_default_params(self, text: str, seed: int) -> None:
        assert minhash_signature(text, seed=seed) == reference_minhash_signature(text, seed=seed)

    def test_tricky_unicode_rows_match(self) -> None:
        # The rows the random alphabet never draws: CRLF, ZWJ emoji,
        # regional indicators, Hangul jamo, the SARA AM spacing mark, CJK
        # scriptio continua, and the U+001F separator itself as a token.
        rows = [
            "a\r\nb",
            "\U0001f469‍\U0001f52c \u30c6\u30b9\u30c8",
            "\U0001f1fa\U0001f1f8\U0001f1fa",
            "각 서울",
            "café naïve",
            "我爱北京天安门天安门上太阳升",
            "a\x1fb c",
            "éclair café",  # decomposed then precomposed accents
            "İstanbul ẞussra",  # full case mapping, not ASCII folding
        ]
        for text in rows:
            assert minhash_signature(text, num_perm=8) == reference_minhash_signature(
                text, num_perm=8
            ), f"oracle disagreement for {text!r}"

    def test_oracle_token_stream_is_word_bounds_derived(self) -> None:
        # The tokenizer-consistency pin: the oracle's token stream is the
        # documented derivation from the public segmentation surface
        # (word_bounds segments, whitespace-only skipped, lowercased), and
        # tors's signature over a text equals the oracle's over a text with
        # the SAME token stream but different raw spelling -- case and
        # whitespace shape are invisible downstream of the tokenizer.
        assert reference_minhash_tokens("Hello, WORLD!") == ["hello", ",", "world", "!"]
        assert reference_minhash_tokens("a\tb\nc\r\nd") == ["a", "b", "c", "d"]
        assert reference_minhash_tokens("") == []
        assert minhash_signature("Hello, WORLD!") == minhash_signature("hello, world!")
        assert minhash_signature("a\tb\nc") == minhash_signature("a b\nc")


class TestGoldenPins:
    """The derivation and hash are platform-independent pinned arithmetic;
    these literals (oracle-derived, byte-exact) make any drift -- a
    coefficient-derivation change, a hash swap, a Mersenne-reduction bug --
    fail loudly instead of silently re-fingerprinting every corpus."""

    def test_full_signature_at_num_perm_8(self) -> None:
        assert minhash_signature(_FOX, num_perm=8) == [
            207829741228551648,
            886224216710130644,
            5239036883153575,
            410872505281952979,
            206408683582913069,
            264955052027332870,
            596676407842964365,
            70385239711227410,
        ]

    def test_first_elements_across_params(self) -> None:
        # (text, kwargs) -> the first four elements plus the last,
        # oracle-derived.
        pins: list[tuple[str, dict[str, int], list[int], int]] = [
            (
                _FOX,
                {},
                [207829741228551648, 886224216710130644, 5239036883153575, 410872505281952979],
                221145959159655086,
            ),
            (
                _FOX,
                {"seed": 42},
                [270372989529981796, 105077448321344730, 235377393589391342, 121953126375922264],
                551058709496425660,
            ),
            (
                _FOX,
                {"seed": -1},
                [254597756377403703, 48405316552563713, 178401967871503489, 313500567055445292],
                383375621771831909,
            ),
            (
                _FOX,
                {"shingle_size": 1},
                [43535712731860995, 400583443249740695, 1101800490404374691, 298222042623614620],
                94148238014863239,
            ),
            (
                _FOX,
                {"shingle_size": 2, "seed": 9},
                [31169703906224804, 227597385112405833, 163487546933959719, 361898046382711100],
                26421055431877821,
            ),
            (
                "café société naïve 東京は日本の首都です",
                {"seed": 7, "num_perm": 16, "shingle_size": 2},
                [118418957341389701, 425684181524778837, 123408742880440953, 147121324028316642],
                64230409300765984,
            ),
        ]
        for text, kwargs, first4, last in pins:
            sig = minhash_signature(text, **kwargs)
            assert sig[:4] == first4, f"{text!r} {kwargs}: head drifted"
            assert sig[-1] == last, f"{text!r} {kwargs}: tail drifted"

    def test_num_perm_prefix_property(self) -> None:
        # The SplitMix64 stream draws (a_i, b_i) in permutation order, so a
        # smaller signature is a prefix of a larger one at the same seed:
        # k=8's signature is exactly k=128's first 8 elements. Pinned as an
        # equality because it is a structural consequence of the pinned
        # derivation, not an accident of one text.
        full = minhash_signature(_FOX)
        for num_perm in (1, 8, 64, 127):
            assert minhash_signature(_FOX, num_perm=num_perm) == full[:num_perm]

    def test_deterministic_across_calls_and_objects(self) -> None:
        text = _FOX
        fresh = "".join(text)  # a distinct str object, same content
        assert text is not fresh
        assert minhash_signature(text) == minhash_signature(text)
        assert minhash_signature(text) == minhash_signature(fresh)


class TestEmptyConvention:
    def test_empty_text_is_all_sentinel(self) -> None:
        sig = minhash_signature("")
        assert sig == [_MINHASH_EMPTY] * 128
        assert sig[0] == 2**64 - 1

    def test_whitespace_only_is_all_sentinel(self) -> None:
        # No word tokens (the WSegSpace artifact segment is skipped), so no
        # shingles: the same empty-set convention as empty text, across
        # whitespace shapes (spaces, tabs, CRLF, NBSP, ideographic space).
        for text in ("   ", "\t\n \r\n ", "    　"):
            assert minhash_signature(text) == [_MINHASH_EMPTY] * 128, f"{text!r}"

    def test_fewer_tokens_than_shingle_size_is_all_sentinel(self) -> None:
        assert minhash_signature("one two") == [_MINHASH_EMPTY] * 128
        assert minhash_signature("one two three", shingle_size=4) == [_MINHASH_EMPTY] * 128
        # Exactly shingle_size tokens is ONE shingle, not the empty set.
        sig = minhash_signature("one two three")
        assert len(sig) == 128
        assert all(v != _MINHASH_EMPTY for v in sig)

    def test_empty_convention_is_seed_invariant(self) -> None:
        # No shingles means no coefficients are ever applied: every seed
        # gives the same all-sentinel answer.
        for seed in (0, 1, -1, 2**100):
            assert minhash_signature("", seed=seed) == [_MINHASH_EMPTY] * 128


class TestBoundsContract:
    def test_num_perm_bounds_are_inclusive(self) -> None:
        assert len(minhash_signature(_FOX, num_perm=1)) == 1
        assert len(minhash_signature(_FOX, num_perm=1024)) == 1024

    @pytest.mark.parametrize("num_perm", [0, -1, 1025, 10**9])
    def test_num_perm_out_of_bounds_raises_value_error(self, num_perm: int) -> None:
        with pytest.raises(ValueError, match="1 and 1024"):
            minhash_signature(_FOX, num_perm=num_perm)

    @pytest.mark.parametrize("shingle_size", [0, -1, -5])
    def test_shingle_size_below_one_raises_value_error(self, shingle_size: int) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            minhash_signature(_FOX, shingle_size=shingle_size)

    def test_length_always_equals_num_perm(self) -> None:
        for num_perm in (1, 7, 128, 1024):
            assert len(minhash_signature(_FOX, num_perm=num_perm)) == num_perm

    def test_values_are_in_the_affine_range_or_sentinel(self) -> None:
        # Every real minimum is an affine output in [0, 2^61 - 1); the only
        # value outside that range the API can ever return is the sentinel.
        sig = minhash_signature(_FOX)
        assert all(0 <= v < 2**61 for v in sig)


class TestSeedContract:
    def test_negative_seed_is_twos_complement_mod_2_64(self) -> None:
        # seed=-1 reduces to 0xFFFF_FFFF_FFFF_FFFF: the house convention,
        # so seed=-1 and seed=2**64-1 are the same permutation draw.
        assert minhash_signature(_FOX, seed=-1) == minhash_signature(_FOX, seed=2**64 - 1)
        assert minhash_signature(_FOX, seed=-(2**63)) == minhash_signature(_FOX, seed=2**63)

    def test_arbitrary_magnitude_seed_reduces_mod_2_64(self) -> None:
        assert minhash_signature(_FOX, seed=2**100) == minhash_signature(_FOX, seed=0)
        assert minhash_signature(_FOX, seed=2**100 + 5) == minhash_signature(_FOX, seed=5)

    def test_default_seed_is_zero(self) -> None:
        assert minhash_signature(_FOX) == minhash_signature(_FOX, seed=0)

    def test_different_seeds_draw_different_permutations(self) -> None:
        # A deterministic row, not a probability claim: seeds 0 and 1
        # differ on this text (pinned; the derivation makes collision
        # astronomically unlikely, and the pin documents the intent).
        assert minhash_signature(_FOX, seed=0) != minhash_signature(_FOX, seed=1)

    @pytest.mark.parametrize("seed", ["0", 0.5, 1.0, None, b"0"])
    def test_non_int_seed_raises_type_error(self, seed: object) -> None:
        with pytest.raises(TypeError):
            minhash_signature(_FOX, seed=seed)  # type: ignore[arg-type]


class TestJaccardProperty:
    """The MinHash guarantee, pinned without statistical flakiness: the
    signature agreement fraction estimates the shingle-set Jaccard
    similarity, E[agreement] = J with standard deviation sqrt(J(1-J)/k).
    The three fixture pairs below are deterministic documents, so their
    agreement counts are exact integers, oracle-derived and pinned; the
    estimation rows assert |estimate - exact| stays inside the k=128
    2-sigma band (0.088 = 1/sqrt(128), twice the max 1-sigma standard
    error sqrt(0.25/128) ~ 0.044 at the worst J=0.5; measured 0.028-0.043
    on these rows)."""

    _NEAR_A = (
        "The quarterly oil sample interval for field outages was adjusted after the bushing "
        "torque specifications changed. Maintenance windows now close within fourteen days. "
    ) * 3
    _NEAR_B = _NEAR_A.replace("quarterly", "monthly").replace("bushing", "insulator")
    _MODERATE_A = (
        "The quarterly oil sample interval for field outages was adjusted after the bushing "
        "torque specifications changed. Maintenance windows now close within fourteen days. "
        "Review panels approved the revised schedule and the field team confirmed the plan. "
    )
    _MODERATE_B = (
        "Review panels approved the revised schedule and the field team confirmed the plan. "
        "Spare parts arrived on site before the storm and the crew replaced the seal. "
        "Documentation updates followed the same revision cycle. "
    )
    _DISJOINT_A = "the quick brown fox jumps over the lazy dog " * 6
    _DISJOINT_B = "compiler backends schedule instructions over directed acyclic graphs " * 6

    def test_near_identical_pair_agreement_is_pinned(self) -> None:
        a = minhash_signature(self._NEAR_A)
        b = minhash_signature(self._NEAR_B)
        assert _agreement(a, b) == 82  # exact J 0.6129, estimate 82/128 = 0.6406

    def test_moderately_similar_pair_agreement_is_pinned(self) -> None:
        a = minhash_signature(self._MODERATE_A)
        b = minhash_signature(self._MODERATE_B)
        assert _agreement(a, b) == 30  # exact J 0.2000, estimate 30/128 = 0.2344

    def test_disjoint_pair_agreement_is_pinned(self) -> None:
        a = minhash_signature(self._DISJOINT_A)
        b = minhash_signature(self._DISJOINT_B)
        assert _agreement(a, b) == 0  # exact J 0.0

    def test_near_beats_moderate_beats_disjoint(self) -> None:
        # The ordering the recall gate exists for, with wide margins.
        near = _agreement(
            minhash_signature(self._NEAR_A), minhash_signature(self._NEAR_B)
        )
        moderate = _agreement(
            minhash_signature(self._MODERATE_A), minhash_signature(self._MODERATE_B)
        )
        disjoint = _agreement(
            minhash_signature(self._DISJOINT_A), minhash_signature(self._DISJOINT_B)
        )
        assert near > moderate > disjoint
        assert near - moderate >= 40  # 82 vs 30 measured: the margins are structural

    def test_estimates_stay_within_the_k_128_error_band(self) -> None:
        # |estimate - exact| over the three fixtures plus a one-word edit of
        # the prose corpus: all deterministic, all measured 0.028-0.043.
        # The max 1-sigma standard error is sqrt(J(1-J)/k) <= sqrt(0.25/128)
        # ~ 0.044 at the worst J=0.5; the asserted band, 0.088 = 1/sqrt(128),
        # is the 2-sigma band at that worst case, with margin.
        from reference import prose

        rows: list[tuple[str, str, float]] = [
            (self._NEAR_A, self._NEAR_B, 0.6129),
            (self._MODERATE_A, self._MODERATE_B, 0.2000),
            (self._DISJOINT_A, self._DISJOINT_B, 0.0),
        ]
        for a_text, b_text, exact in rows:
            est = _agreement(minhash_signature(a_text), minhash_signature(b_text)) / 128
            assert abs(est - exact) < 0.088, f"{est:.4f} vs exact {exact:.4f}"
        # The prose corpus, one word swapped once: exact J 0.9259, estimate
        # 124/128 = 0.9688 (the repeated-sentence corpus has only 25
        # distinct shingles at 64 KiB -- the estimate is coarse there by
        # construction, still inside the band).
        a_text = prose(64 * 1024)
        b_text = a_text.replace("quarterly", "monthly", 1)
        est = _agreement(minhash_signature(a_text), minhash_signature(b_text)) / 128
        assert abs(est - 0.9259) < 0.088


class TestUnicodeAndCrashFreedom:
    def test_cjk_scriptio_continua_signatures_are_non_empty(self) -> None:
        # No whitespace: UAX #29 still segments CJK into word units, so the
        # shingle set is non-empty (unlike a whitespace split).
        zh = "我爱北京天安门天安门上太阳升"
        sig = minhash_signature(zh)
        assert all(v != _MINHASH_EMPTY for v in sig)
        assert minhash_signature(zh) == sig

    def test_emoji_zwj_and_regional_indicators_do_not_crash(self) -> None:
        family = "\U0001f469‍\U0001f468‍\U0001f467‍\U0001f466"
        flags = "\U0001f1fa\U0001f1f8\U0001f1e8\U0001f1e6"
        text = f"I love {family} families and {flags} flags"
        sig = minhash_signature(text)
        assert isinstance(sig, list)
        assert minhash_signature(text) == sig

    def test_lone_surrogate_raises_unicode_encode_error(self) -> None:
        # The crate-wide str-borrow contract: the &str extraction's UTF-8
        # materialization rejects lone surrogates.
        with pytest.raises(UnicodeEncodeError):
            minhash_signature("a\ud800b")


class TestArgumentContract:
    def test_non_str_text_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            minhash_signature(123)  # type: ignore[arg-type]

    def test_missing_text_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            minhash_signature()  # type: ignore[call-arg]

    def test_parameters_are_keyword_only(self) -> None:
        # The API is (text, *, num_perm, shingle_size, seed): positional
        # spellings are TypeError, the pyi-drift-gate-pinned shape.
        with pytest.raises(TypeError):
            minhash_signature(_FOX, 64)  # type: ignore[misc]
        with pytest.raises(TypeError):
            minhash_signature(_FOX, 64, 3)  # type: ignore[misc]

    def test_non_int_num_perm_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            minhash_signature(_FOX, num_perm="128")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            minhash_signature(_FOX, num_perm=128.0)  # type: ignore[arg-type]

    def test_non_int_shingle_size_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            minhash_signature(_FOX, shingle_size=3.5)  # type: ignore[arg-type]

    def test_return_type_is_list_of_ints(self) -> None:
        sig = minhash_signature(_FOX)
        assert isinstance(sig, list)
        assert all(isinstance(v, int) for v in sig)


class TestPerformanceSanity:
    def test_large_input_completes_quickly(self) -> None:
        # ~13.5 MB, the simhash sanity cell's corpus shape: the detached
        # tokenize+shingle+hash+sweep pass is measured in the hundreds of
        # milliseconds at k=128 (see docs/performance.md); 5s is the
        # crash/regression ceiling, not the target.
        big = "the quick brown fox jumps over the lazy dog. " * 300_000
        started = time.perf_counter()
        sig = minhash_signature(big)
        elapsed = time.perf_counter() - started
        assert len(sig) == 128
        assert elapsed < 5.0, f"minhash_signature on ~13.5MB took {elapsed:.2f}s"

    def test_large_input_is_deterministic(self) -> None:
        big = "lorem ipsum dolor sit amet " * 100_000
        assert minhash_signature(big) == minhash_signature(big)
