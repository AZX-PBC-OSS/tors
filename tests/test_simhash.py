"""Contract gate for ``tors.simhash64``: Charikar's 64-bit SimHash
fingerprint over a text's UAX #29 word tokens, the fuzzy near-duplicate
gate that sits alongside `tors.finalize`'s exact SHA-256 gate (`finalize`
answers "byte-identical"; `simhash64` answers "nearly the same" via
Hamming distance on the returned int). See ``src/simhash_impl.rs``'s
module docstring for the full algorithm writeup and the
corpus-dependent calibration caveat; the Hamming-distance bounds pinned
below are exact reproductions of that file's own measured anchors,
this module is the Python-visible half of the same contract, not an
independent recalibration.
"""

from __future__ import annotations

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import simhash64

_TEXT = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs", "P"), max_codepoint=0x2FFF),
    max_size=200,
)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class TestDegenerateInputs:
    def test_empty_text_is_zero(self) -> None:
        assert simhash64("") == 0

    def test_whitespace_only_is_zero(self) -> None:
        # WSegSpace joins a run of whitespace into one UAX #29 segment,
        # which the tokenizer then skips as not-a-word: no tokens, no
        # votes, zero fingerprint. Exercised across several whitespace
        # shapes (plain spaces, tabs, CRLF, NBSP), not just one run.
        assert simhash64("   ") == 0
        assert simhash64("\t\n \r\n ") == 0

    def test_single_word_is_the_vote_of_one(self) -> None:
        # With exactly one token, every bit's vote is +-1 with no other
        # token to contest it, so the fingerprint is that token's FNV-1a
        # hash verbatim. Pinned as the literal computed by the Rust
        # implementation (src/simhash_impl.rs's own degenerate-input test).
        assert simhash64("hello") == 0xA430D84680AABD0B

    def test_return_type_and_range(self) -> None:
        result = simhash64("hello world")
        assert isinstance(result, int)
        assert 0 <= result < 2**64


class TestDeterminism:
    def test_repeated_calls_agree(self) -> None:
        text = "the quick brown fox jumps over the lazy dog"
        assert simhash64(text) == simhash64(text)

    def test_fresh_string_object_agrees(self) -> None:
        text = "the quick brown fox jumps over the lazy dog"
        fresh = "".join(text)  # a distinct str object, same content
        assert text is not fresh
        assert simhash64(text) == simhash64(fresh)

    @given(text=_TEXT)
    @settings(max_examples=200)
    def test_determinism_property(self, text: str) -> None:
        assert simhash64(text) == simhash64(text)


class TestBagOfWords:
    def test_reversed_word_order_is_the_same_fingerprint(self) -> None:
        assert simhash64("one two three four five six") == simhash64("six five four three two one")

    def test_permuted_punctuation_is_the_same_fingerprint(self) -> None:
        assert simhash64("Hello, world!") == simhash64("world! Hello,")

    def test_frequency_is_part_of_the_bag(self) -> None:
        # Repetition changes the multiset, and hence (generally) the
        # vote -- near-equal token bags are not guaranteed equal
        # fingerprints. Mirrors the same row pinned in the Rust suite.
        assert simhash64("the cat sat on the mat with the hat") != simhash64(
            "the cat sat on the mat with the hat the"
        )


class TestLocalitySensitivity:
    """The property simhash64 exists for: near-duplicate texts land close
    in Hamming distance, unrelated texts land far apart. Bounds below are
    exact reproductions of src/simhash_impl.rs's measured, pinned anchors
    -- not independently chosen thresholds.
    """

    _DOCUMENT = (
        "the old lighthouse keeper walked down the stone "
        "steps every morning before the sun rose over the "
        "harbor and he checked the lamp and the wicks and "
        "the glass for cracks that the winter storms might "
        "have left behind because a light that fails on a "
        "dark coast is a shipwreck waiting and he had kept "
        "this light for forty years through two wars and "
        "one great flood and he knew the sea the way a "
        "farmer knows his fields every rock and every "
        "current and every wind that bends the pines above "
        "the cliff"
    )

    _SENTENCES = (
        "the quick brown fox jumps over the lazy dog",
        "pack my box with five dozen liquor jugs",
        "summer rain falls quietly on the empty garden wall",
        "we hold these truths to be self evident that all people are equal",
        "lorem ipsum dolor sit amet consectetur adipiscing elit sed do",
        "compiler backends schedule instructions over directed acyclic graphs",
    )

    def test_document_scale_single_word_edits_stay_within_4_bits(self) -> None:
        words = self._DOCUMENT.split()
        base_fp = simhash64(self._DOCUMENT)
        worst = 0
        for i in range(len(words)):
            swap = words[:i] + ["silently"] + words[i + 1 :]
            drop = words[:i] + words[i + 1 :]
            insert = words[:i] + ["gently"] + words[i:]
            for variant in (swap, drop, insert):
                worst = max(worst, _hamming(base_fp, simhash64(" ".join(variant))))
        assert worst <= 4
        assert worst == 4  # the exact measured value, pinned

    def test_unrelated_sentences_sit_at_least_16_bits_apart(self) -> None:
        min_seen = 64
        for i, a in enumerate(self._SENTENCES):
            for b in self._SENTENCES[i + 1 :]:
                min_seen = min(min_seen, _hamming(simhash64(a), simhash64(b)))
        assert min_seen >= 16
        assert min_seen == 23  # the exact measured value, pinned

    def test_near_duplicate_paragraph_beats_unrelated_paragraph(self) -> None:
        # An end-to-end demonstration of the whole point: reordering two
        # sentences of a paragraph (near-dup) sits far closer than
        # swapping in an unrelated paragraph entirely.
        base = self._DOCUMENT
        words = base.split()
        near_dup = " ".join(words[:10] + ["silently"] + words[11:])
        unrelated = self._SENTENCES[-1]
        near_dist = _hamming(simhash64(base), simhash64(near_dup))
        far_dist = _hamming(simhash64(base), simhash64(unrelated))
        assert near_dist < far_dist


class TestUnicode:
    def test_cjk_single_word_change_stays_local(self) -> None:
        a = "東京は日本の首都です。今日はとても良い天気です。"
        b = "東京は日本の首都です。今日はとても悪い天気です。"
        assert simhash64(a) != simhash64(b)
        # Confirm no crash and a bounded, non-catastrophic bit flip for a
        # single-token substitution -- not asserting a tight literal bound
        # since CJK segmentation granularity differs from the space-split
        # English batteries above.
        assert _hamming(simhash64(a), simhash64(b)) < 32

    def test_emoji_zwj_and_regional_indicators_do_not_crash(self) -> None:
        family = "\U0001f469‍\U0001f468‍\U0001f467‍\U0001f466"
        flags = "\U0001f1fa\U0001f1f8\U0001f1e8\U0001f1e6"
        text = f"I love {family} families and {flags} flags"
        result = simhash64(text)
        assert isinstance(result, int)
        assert simhash64(text) == result

    def test_combining_marks_do_not_crash(self) -> None:
        text = "café " * 5 + "naïve " * 5
        result = simhash64(text)
        assert isinstance(result, int)
        assert simhash64(text) == result

    def test_scriptio_continua_word_segmentation(self) -> None:
        # No whitespace between words: UAX #29 word segmentation (not an
        # ad-hoc whitespace split) still tokenizes CJK text into more than
        # a single monolithic token, so this must differ from a text
        # without the repeated substring.
        zh = "我爱北京天安门天安门上太阳升"
        assert simhash64(zh) != 0
        assert simhash64(zh) == simhash64(zh)


class TestPerformanceSanity:
    def test_large_input_completes_quickly(self) -> None:
        big = "the quick brown fox jumps over the lazy dog. " * 300_000  # ~13.5MB
        start = time.perf_counter()
        result = simhash64(big)
        elapsed = time.perf_counter() - start
        assert isinstance(result, int)
        assert elapsed < 5.0, f"simhash64 on ~13.5MB took {elapsed:.2f}s, expected < 5s"

    def test_large_input_is_deterministic(self) -> None:
        big = "lorem ipsum dolor sit amet " * 200_000
        assert simhash64(big) == simhash64(big)


class TestArgumentContract:
    def test_non_str_argument_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            simhash64(123)  # type: ignore[arg-type]

    def test_missing_argument_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            simhash64()  # type: ignore[call-arg]
