"""Contract gate for ``tors.simhash128``: the 128-bit spelling of
``tors.simhash64`` (see ``tests/test_simhash.py`` and
``src/simhash_impl.rs``'s module docstring for the shared algorithm
writeup). Same tokens (UAX #29 word segments), same bag-of-words vote,
same FNV-1a-is-the-hash-function choice, twice the bit positions.

This suite is NOT simhash64's suite copy-pasted unadapted: the pinned
literals below (the single-word vote-of-one hash, the document/sentence
near-dup bounds, the unrelated floor) are independently measured at 128
bits, not assumed equal to or exactly double the 64-bit anchors; see
``TestLocalitySensitivity`` for the measured relationship between the two
widths, which is the entire reason the wide spelling exists.
"""

from __future__ import annotations

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import simhash64, simhash128

_TEXT = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs", "P"), max_codepoint=0x2FFF),
    max_size=200,
)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class TestDegenerateInputs:
    def test_empty_text_is_zero(self) -> None:
        assert simhash128("") == 0

    def test_whitespace_only_is_zero(self) -> None:
        assert simhash128("   ") == 0
        assert simhash128("\t\n \r\n ") == 0

    def test_single_word_is_the_vote_of_one(self) -> None:
        # Same vote-of-one shape as simhash64, at 128 bits: with exactly one
        # token, the fingerprint IS that token's FNV-1a-128 hash verbatim.
        # Pinned as the literal computed by the Rust implementation
        # (src/simhash_impl.rs's own degenerate-input test).
        assert simhash128("hello") == 0xE3E1EFD54283D94F7081314B599D31B3

    def test_return_type_and_range(self) -> None:
        result = simhash128("hello world")
        assert isinstance(result, int)
        assert 0 <= result < 2**128


class TestDeterminism:
    def test_repeated_calls_agree(self) -> None:
        text = "the quick brown fox jumps over the lazy dog"
        assert simhash128(text) == simhash128(text)

    def test_fresh_string_object_agrees(self) -> None:
        text = "the quick brown fox jumps over the lazy dog"
        fresh = "".join(text)
        assert text is not fresh
        assert simhash128(text) == simhash128(fresh)

    @given(text=_TEXT)
    @settings(max_examples=200)
    def test_determinism_property(self, text: str) -> None:
        assert simhash128(text) == simhash128(text)


class TestBagOfWords:
    def test_reversed_word_order_is_the_same_fingerprint(self) -> None:
        assert simhash128("one two three four five six") == simhash128(
            "six five four three two one"
        )

    def test_permuted_punctuation_is_the_same_fingerprint(self) -> None:
        assert simhash128("Hello, world!") == simhash128("world! Hello,")

    def test_frequency_is_part_of_the_bag(self) -> None:
        # Frequency is part of the vote, but -- unlike simhash64's own pin
        # of this exact row (test_simhash.py) -- at 128 bits the wider
        # margin can absorb a single extra occurrence for a SHORT text
        # without flipping any bit. Confirmed empirically, not assumed
        # equal to the 64-bit outcome: this row and the one below it are
        # a genuine 64-vs-128 divergence, not a copy-paste bug.
        short_unchanged = simhash128("the cat sat on the mat with the hat")
        short_repeated = simhash128("the cat sat on the mat with the hat the")
        assert short_unchanged == short_repeated

        # At document scale (enough tokens for votes to sit near a
        # majority boundary) an extra occurrence of an already-frequent
        # word still moves the fingerprint.
        base = self._document_for_frequency_test()
        assert simhash128(base) != simhash128(base + " the")

    @staticmethod
    def _document_for_frequency_test() -> str:
        return (
            "the old lighthouse keeper walked down the stone steps every "
            "morning before the sun rose over the harbor and he checked "
            "the lamp and the wicks and the glass for cracks that the "
            "winter storms might have left behind"
        )


class TestLocalitySensitivity:
    """The property the WIDE spelling exists for: better separation between
    the near-dup band and the unrelated floor than the 64-bit width gives.
    Bounds below are exact reproductions of src/simhash_impl.rs's measured,
    pinned anchors for the 128-bit width specifically -- not the 64-bit
    numbers reused, and not derived by assuming "twice as wide" doubles
    anything (it doesn't, uniformly: see the doubling assertion below).
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

    def test_document_scale_single_word_edits_stay_within_3_bits(self) -> None:
        # Measured 3 bits worst at 128 bits, vs 4 bits at 64 bits -- the
        # near-dup band shrinks slightly, it does not grow with the width.
        words = self._DOCUMENT.split()
        base_fp = simhash128(self._DOCUMENT)
        worst = 0
        for i in range(len(words)):
            swap = words[:i] + ["silently"] + words[i + 1 :]
            drop = words[:i] + words[i + 1 :]
            insert = words[:i] + ["gently"] + words[i:]
            for variant in (swap, drop, insert):
                worst = max(worst, _hamming(base_fp, simhash128(" ".join(variant))))
        assert worst <= 4
        assert worst == 3  # the exact measured value, pinned

    def test_sentence_scale_single_word_edits_stay_within_20_bits(self) -> None:
        # The thin-margin face of the same property at 128 bits: 20 bits
        # worst here, vs 14 at 64 bits -- short texts still have thin vote
        # margins, and the wider hash does not rescue that.
        worst = 0
        for base in self._SENTENCES[:4]:
            words = base.split()
            base_fp = simhash128(base)
            for i in range(len(words)):
                swap = words[:i] + ["silently"] + words[i + 1 :]
                drop = words[:i] + words[i + 1 :]
                insert = words[:i] + ["gently"] + words[i:]
                for variant in (swap, drop, insert):
                    worst = max(worst, _hamming(base_fp, simhash128(" ".join(variant))))
        assert worst == 20

    def test_unrelated_sentences_sit_farther_apart_than_at_64_bits(self) -> None:
        # The whole point of the wide spelling: the unrelated floor grows
        # (roughly doubles per the module doc) relative to 64 bits, while
        # the near-dup band does not grow -- better separation, not a
        # uniform rescale of every number.
        min_seen_128 = 128
        min_seen_64 = 64
        for i, a in enumerate(self._SENTENCES):
            for b in self._SENTENCES[i + 1 :]:
                min_seen_128 = min(min_seen_128, _hamming(simhash128(a), simhash128(b)))
                min_seen_64 = min(min_seen_64, _hamming(simhash64(a), simhash64(b)))
        assert min_seen_128 == 40  # the exact measured value, pinned
        assert min_seen_64 == 23  # simhash64's own pinned anchor, for context
        assert min_seen_128 > min_seen_64

    def test_near_duplicate_paragraph_beats_unrelated_paragraph(self) -> None:
        base = self._DOCUMENT
        words = base.split()
        near_dup = " ".join(words[:10] + ["silently"] + words[11:])
        unrelated = self._SENTENCES[-1]
        near_dist = _hamming(simhash128(base), simhash128(near_dup))
        far_dist = _hamming(simhash128(base), simhash128(unrelated))
        assert near_dist < far_dist


class TestUnicode:
    def test_cjk_single_word_change_stays_local(self) -> None:
        a = "東京は日本の首都です。今日はとても良い天気です。"
        b = "東京は日本の首都です。今日はとても悪い天気です。"
        assert simhash128(a) != simhash128(b)
        assert _hamming(simhash128(a), simhash128(b)) < 64

    def test_emoji_zwj_and_regional_indicators_do_not_crash(self) -> None:
        family = "\U0001f469‍\U0001f468‍\U0001f467‍\U0001f466"
        flags = "\U0001f1fa\U0001f1f8\U0001f1e8\U0001f1e6"
        text = f"I love {family} families and {flags} flags"
        result = simhash128(text)
        assert isinstance(result, int)
        assert simhash128(text) == result

    def test_combining_marks_do_not_crash(self) -> None:
        text = "café " * 5 + "naïve " * 5
        result = simhash128(text)
        assert isinstance(result, int)
        assert simhash128(text) == result

    def test_scriptio_continua_word_segmentation(self) -> None:
        zh = "我爱北京天安门天安门上太阳升"
        assert simhash128(zh) != 0
        assert simhash128(zh) == simhash128(zh)


class TestIndependenceFromSimhash64:
    """simhash128 is a genuinely separate fingerprint, not simhash64
    zero-extended or otherwise derived from it at the API boundary."""

    @given(text=_TEXT)
    @settings(max_examples=100)
    def test_128_bit_value_is_not_merely_the_64_bit_value_widened(
        self, text: str
    ) -> None:
        wide = simhash128(text)
        narrow = simhash64(text)
        if narrow != 0:
            assert wide != narrow

    def test_upper_64_bits_are_routinely_nonzero(self) -> None:
        # A zero-extended 64-bit value would always have its upper half
        # clear; a genuine 128-bit FNV-1a vote does not.
        text = "the quick brown fox jumps over the lazy dog and then some more words follow"
        assert (simhash128(text) >> 64) != 0


class TestPerformanceSanity:
    def test_large_input_completes_quickly(self) -> None:
        big = "the quick brown fox jumps over the lazy dog. " * 300_000  # ~13.5MB
        start = time.perf_counter()
        result = simhash128(big)
        elapsed = time.perf_counter() - start
        assert isinstance(result, int)
        assert elapsed < 5.0, f"simhash128 on ~13.5MB took {elapsed:.2f}s, expected < 5s"

    def test_large_input_is_deterministic(self) -> None:
        big = "lorem ipsum dolor sit amet " * 200_000
        assert simhash128(big) == simhash128(big)


class TestArgumentContract:
    def test_non_str_argument_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            simhash128(123)  # type: ignore[arg-type]

    def test_missing_argument_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            simhash128()  # type: ignore[call-arg]
