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
token windows hashed under the injective length-prefixed framing
(LE64(n) || (LE64(len) || bytes)*); each shingle is hashed with XXH64 (seed 0,
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

import asyncio
import os
import random
import warnings
from itertools import product

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors
from loop_harness import assert_bounded, assert_heartbeat_clean
from reference import (
    _MINHASH_EMPTY,
    reference_minhash_jaccard,
    reference_minhash_signature,
    reference_minhash_tokens,
    reference_superminhash_signature,
)
from tors import minhash_jaccard, minhash_signature

# The mixed text alphabet: letters, numbers, punctuation, spaces, the
# hard-break whitespace, the C0/C1 controls (U+001F separator included),
# combining marks (Mn: the WB4 separator-attach), format controls (Cf:
# ZWJ, SOFT HYPHEN), and line separators (Zl), out through the astral
# planes (the max_codepoint covers emoji and other supplementary-plane
# text; surrogates are excluded structurally -- the Cs category is absent
# from the whitelist, so no draw can be a surrogate, which would raise
# UnicodeEncodeError at the boundary instead) -- the differential oracle
# must agree with the core on all of it, not just letters. Surrogates are
# excluded explicitly (a lone surrogate is a UnicodeEncodeError at the
# boundary, pinned separately below); the separator row stays an explicit
# fixture too (the Cs surrogate category is absent from the list, so no
# draw can be a surrogate).
_TEXT = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Zs", "P", "Mn", "Cf", "Cc", "Zl"),
        whitelist_characters="\n\r\t",
        max_codepoint=0x10FFFF,
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
        assert reference_minhash_tokens("我爱北京 天安门") == [
            "我",
            "爱",
            "北",
            "京",
            "天",
            "安",
            "门",
        ]
        assert reference_minhash_tokens("a b　c") == ["a", "b", "c"]
        assert reference_minhash_tokens("\x1f\u0301 b") == ["\x1f\u0301", "b"]
        # No NFC in this path (unlike normalize): a precomposed accent and
        # its decomposed spelling stay byte-distinct tokens.
        assert reference_minhash_tokens("é É") == ["é", "é"]
        assert minhash_signature("Hello, WORLD!") == minhash_signature("hello, world!")
        assert minhash_signature("a\tb\nc") == minhash_signature("a b\nc")

    def test_whitespace_skip_matches_the_documented_25_codepoints(self) -> None:
        # The _WHITESPACE transcription drift gate: the oracle's skip set
        # is transcribed from Rust's char::is_whitespace (NOT Python's
        # str.isspace, which also counts U+001C..U+001F). Every documented
        # member must skip, and the U+001C..U+001F controls Python counts
        # but Rust keeps must survive as real tokens -- a transcription
        # edit in either direction fails here instead of silently
        # re-fingerprinting every corpus through the oracle. The probe is
        # solo-character: the skip fires on segments made ENTIRELY of
        # whitespace, so each documented member alone must vanish -- while
        # mid-word glue behavior (U+202F/U+00A0 keep "a b" one segment)
        # is the segmenter's own break contract, not the skip's, and is
        # pinned by the segmentation gate instead.
        from reference import _WHITESPACE

        assert sorted(map(ord, _WHITESPACE)) == [
            0x0009,
            0x000A,
            0x000B,
            0x000C,
            0x000D,
            0x0020,
            0x0085,
            0x00A0,
            0x1680,
            0x2000,
            0x2001,
            0x2002,
            0x2003,
            0x2004,
            0x2005,
            0x2006,
            0x2007,
            0x2008,
            0x2009,
            0x200A,
            0x2028,
            0x2029,
            0x202F,
            0x205F,
            0x3000,
        ]
        for ch in sorted(_WHITESPACE):
            assert reference_minhash_tokens(ch) == [], f"U+{ord(ch):04X} not skipped"
            assert reference_minhash_tokens(ch * 3) == [], f"U+{ord(ch):04X}x3 not skipped"
        for cp in range(0x001C, 0x0020):
            assert reference_minhash_tokens(chr(cp)) == [chr(cp)], f"U+{cp:04X} wrongly skipped"

    def test_dedup_sweep_agrees_with_the_oracle_on_repeated_tokens(self) -> None:
        # The dedup-first sweep's load-bearing equality: minima over
        # occurrences equal minima over the distinct set, so the core
        # (which updates each distinct shingle once) must agree with the
        # per-occurrence oracle exactly where the sets differ most -- the
        # repeated-token pathology ("ab ".repeat(200): 400 occurrences,
        # ONE distinct shingle) and a 3-token-period text (150
        # occurrences, 3 distinct shingles).
        for text in ("ab " * 200, "ab cd ef " * 50):
            for kwargs in (
                {},
                {"num_perm": 8, "seed": 42},
                {"num_perm": 16, "shingle_size": 2, "seed": -1},
            ):
                assert minhash_signature(text, **kwargs) == reference_minhash_signature(
                    text, **kwargs
                ), f"{text[:12]!r}... {kwargs}"


class TestGoldenPins:
    """The derivation and hash are platform-independent pinned arithmetic;
    these literals (oracle-derived, byte-exact) make any drift -- a
    coefficient-derivation change, a hash swap, a Mersenne-reduction bug --
    fail loudly instead of silently re-fingerprinting every corpus."""

    def test_full_signature_at_num_perm_8(self) -> None:
        assert minhash_signature(_FOX, num_perm=8) == [
            169259083321921084,
            172457087984037558,
            128202111860233391,
            493909799184687459,
            178272364556672123,
            724714824782277816,
            541813272509765753,
            270160054342519274,
        ]

    def test_first_elements_across_params(self) -> None:
        # (text, kwargs) -> the first four elements plus the last,
        # oracle-derived.
        pins: list[tuple[str, dict[str, int], list[int], int]] = [
            (
                _FOX,
                {},
                [169259083321921084, 172457087984037558, 128202111860233391, 493909799184687459],
                233113311292222340,
            ),
            (
                _FOX,
                {"seed": 42},
                [131105769293970662, 48546077666774954, 385205238630604648, 418317608302171271],
                224966429528101112,
            ),
            (
                _FOX,
                {"seed": -1},
                [47248574907900023, 265721080939863404, 453343482114475226, 165744942491687530],
                81491349990655831,
            ),
            (
                _FOX,
                {"shingle_size": 1},
                [191606523678226712, 202562416858770285, 182802331010015698, 794464171970867],
                39137848709602292,
            ),
            (
                _FOX,
                {"shingle_size": 2, "seed": 9},
                [827085101323958520, 193580844359367194, 988130364243929771, 811785053874202799],
                412014949155909011,
            ),
            (
                "café société naïve 東京は日本の首都です",
                {"seed": 7, "num_perm": 16, "shingle_size": 2},
                [561701975191372594, 60786007352341992, 308166649822990343, 104059567270939064],
                90746508296396512,
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


class TestShingleSeparatorInjectivity:
    """The separator-behavior pins behind the framing's documented
    injectivity. The first row pins what still holds: over plain
    letter/separator/whitespace text U+001F never MIXES INTO a longer UAX
    #29 segment -- it reaches the token stream only as an entire
    single-character token (``word_bounds("a\\x1fb")`` is the three tokens
    ``["a", "\\x1f", "b"]``: a C0 control is its own word segment, and
    U+001F is not whitespace, so the segmenter keeps it, alone). The
    second row pins where that stops: UAX #29 WB4 glues a following
    combining mark, ZWJ, or SOFT HYPHEN onto the separator, so the stream
    holds tokens like ``"\\x1f\\u0301"`` -- the old "no token contains
    U+001F" premise is false, and the shingle hash (the length-prefixed
    framing pinned in ``TestShingleFraming``) deliberately rests on
    nothing the segmenter can take away. Together the rows guard both
    directions: a segmentation change in either direction -- U+001F
    starting to mix into plain-letter runs, or the WB4 attach going
    away -- is a gate failure instead of a silent fingerprint change."""

    def test_no_token_mixes_the_separator_in_over_a_x1f_bearing_corpus(self) -> None:
        # Every string over the alphabet {a, b, \x1f, space} up to length
        # 6 (5460 strings: every placement of separator runs among
        # letters and whitespace, every edge shape included), plus the
        # explicit edge rows (multi-separator runs, tab-flanked, the
        # uppercase fold, a separator run at each end): any token
        # containing \x1f is exactly the one-character "\x1f" -- never a
        # longer segment.
        alphabet = ["a", "b", "\x1f", " "]
        corpus = {
            "a\x1fb",
            "a\x1f\x1f\x1fb",
            "\x1f",
            "\x1f" * 7,
            " \x1f ",
            "a\t\x1f\tb",
            "A\x1fB",
            "ab\x1fcd",
            "\x1fa b\x1f",
        }
        for length in range(1, 7):
            corpus.update("".join(chars) for chars in product(alphabet, repeat=length))
        for text in sorted(corpus):
            for token in reference_minhash_tokens(text):
                assert token == "\x1f" or "\x1f" not in token, (
                    f"{text!r}: token {token!r} mixes the separator into a longer segment"
                )

    def test_separator_adjacent_marks_attach_to_the_separator(self) -> None:
        # UAX #29 WB4 (ignore Extend/Format/ZWJ): U+001F followed by a
        # combining mark (Mn), ZWJ (Cf), or SOFT HYPHEN (Cf, Format) stays
        # ONE word segment -- so the token stream DOES hold tokens that
        # contain U+001F mixed with more characters. This is the root-cause
        # pin behind the framing fix: the old "U+001F never mixes into a
        # longer segment" premise is false, and the shingle hash no longer
        # relies on it (see TestShingleFraming below).
        assert reference_minhash_tokens("\x1f\u0301") == ["\x1f\u0301"]
        assert reference_minhash_tokens("\x1f\u200d") == ["\x1f\u200d"]
        assert reference_minhash_tokens("\x1f\u00ad") == ["\x1f\u00ad"]
        assert reference_minhash_tokens("a \x1f\u0301") == ["a", "\x1f\u0301"]
        # Every combining mark in U+0300-U+036F attaches the same way.
        for cp in range(0x0300, 0x0370):
            mark = chr(cp)
            assert reference_minhash_tokens("\x1f" + mark) == ["\x1f" + mark.lower()], (
                f"U+{cp:04X} did not attach to the separator"
            )

    def test_signatures_over_separator_bearing_text_match_the_oracle(self) -> None:
        # The framing fix must hold exactly where the old invariant broke:
        # texts whose token stream carries U+001F-bearing tokens.
        rows = [
            "\x1f\u0301 a b c",
            "a \x1f\u200d b \x1f\u00ad c",
            "A\x1f\u0301b c d",
            "\x1f\u0300 \x1f\u036f x y",
        ]
        for text in rows:
            assert minhash_signature(text, num_perm=8) == reference_minhash_signature(
                text, num_perm=8
            ), f"oracle disagreement for {text!r}"

    def test_the_framing_is_injective_over_the_exhaustive_small_domain(self) -> None:
        # Every window (ordered, with repetition) over the token domain
        # {a, b, ab, "\x1f", "\x1f\u0301"} at shingle_size 1..4 frames
        # distinctly under the length-prefixed framing (the domain chosen
        # so the framing must disambiguate a token that is another token's
        # concatenation ("ab" vs [a, b]), the separator's own single token,
        # AND a token carrying the separator itself). The exhaustive form
        # of "two distinct token windows never frame to the same bytes" --
        # the property the old U+001F join could not prove once WB4 lets
        # "\x1f\u0301" into the stream.
        domain = ["a", "b", "ab", "\x1f", "\x1f\u0301"]
        framed: set[bytes] = set()
        windows = 0
        for shingle_size in range(1, 5):
            for window in product(domain, repeat=shingle_size):
                framed.add(TestShingleFraming._frame(list(window)))
                windows += 1
        assert windows == 5 + 25 + 125 + 625
        assert len(framed) == windows, "the shingle framing is not injective"


class TestShingleFraming:
    """The shingle-hash framing pin: each window hashes the length-prefixed
    frame ``LE64(n) || (LE64(len) || bytes)*`` with XXH64 seed 0 -- never
    the ``token + U+001F + token`` join, which is ambiguous once tokens
    themselves can carry U+001F (the WB4 attach pinned above)."""

    @staticmethod
    def _frame(window: list[str]) -> bytes:
        import struct

        out = struct.pack("<Q", len(window))
        for token in window:
            raw = token.encode("utf-8")
            out += struct.pack("<Q", len(raw)) + raw
        return out

    def test_single_shingle_signature_matches_the_framing(self) -> None:
        # "a \\x1f\\u0301" tokenizes to exactly ["a", "\\x1f\\u0301"], so at
        # shingle_size=2 the signature is one shingle's affine image --
        # computed here through the INDEPENDENT xxhash package over the
        # length-prefixed frame, never through the oracle's join.
        import xxhash

        from reference import reference_minhash_coefficients

        text = "a \x1f\u0301"
        assert reference_minhash_tokens(text) == ["a", "\x1f\u0301"]
        x = xxhash.xxh64_intdigest(self._frame(["a", "\x1f\u0301"]))
        expected = [
            (a * x + b) % (2**61 - 1)
            for (a, b) in reference_minhash_coefficients(8, 0)
        ]
        assert minhash_signature(text, num_perm=8, shingle_size=2, seed=0) == expected

    def test_frames_are_distinct_over_a_separator_bearing_domain(self) -> None:
        # The framing construction itself is injective: every window over a
        # domain holding U+001F-bearing tokens frames distinctly (the
        # property the old join lacked a proof of). Pure-Python pin on the
        # construction; the tors half is pinned by the golden above plus
        # the oracle differential.
        from itertools import product

        domain = ["a", "b", "ab", "\x1f", "\x1f\u0301", "\u0301"]
        framed: set[bytes] = set()
        windows = 0
        for shingle_size in range(1, 4):
            for window in product(domain, repeat=shingle_size):
                framed.add(self._frame(list(window)))
                windows += 1
        assert windows == 6 + 36 + 216
        assert len(framed) == windows, "the shingle framing is not injective"


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

    def test_num_perm_1024_matches_the_oracle(self) -> None:
        # The top of the range rides the full coefficient stream: pin it
        # against the oracle, not just for length.
        assert minhash_signature(_FOX, num_perm=1024, seed=7) == reference_minhash_signature(
            _FOX, num_perm=1024, seed=7
        )

    @pytest.mark.timing
    def test_huge_shingle_size_is_sentinel_without_work(self) -> None:
        # A shingle wider than the token stream is the empty set, however
        # huge the width: the sentinel answer with no window blowup.
        # The BEHAVIOR is the sentinel equality; the wall assert rides
        # along as a tripwire (huge measured margin), so the cell takes
        # the timing lane's discipline rather than the fast lane's.
        # Load-robust spelling (tests/loop_harness.py): min-of-3
        # pass-on-first-clean over the tripwire.
        sig = assert_bounded(
            lambda: minhash_signature(_FOX, shingle_size=10**9),
            5.0,
            samples=3,
            label="the huge-shingle sentinel",
        )
        assert sig == [_MINHASH_EMPTY] * 128

    def test_huge_shingle_size_over_large_text_is_sentinel_without_retention(self) -> None:
        # HIGH1: a shingle wider than the token stream must answer the
        # sentinel WITHOUT retaining the stream. The pre-fix sweep grew the
        # window deque to min(tokens, shingle_size), so ~13.5MB of prose at
        # shingle_size=10**9 retained ~2.7M token Strings (~116-145MB child
        # self peak on the dev box, macOS/arm64) for an answer that is
        # always the sentinel. The fix counts tokens retention-free when the
        # width exceeds the stream, so the child peak stays near the
        # tokenizer transient (~29MB dev-box self peak, vs ~29MB for the
        # same text at the default width).
        #
        # Ground truth (CI #74): each probe reports its OWN peak, read from
        # the kernel, not the libc. The previous revision read the child's
        # resource.getrusage(RUSAGE_SELF) -- on the ubuntu CI runners that
        # call returns a workload-invariant fiction (~570-615MB, byte-
        # identical for import-only, huge-window, and default-width probes
        # within one leg) while /proc/self/status VmHWM, the smaps rollup
        # ([heap] under 1MB, ~28MB total), and independent glibc repros
        # (x86_64 + aarch64, glibc 2.35 + 2.39, Rust-level and full Python
        # stack) all agree the true peaks are ~15-38MB. No code change
        # could ever move that reading -- the deque, transient-churn, and
        # zero-alloc counter spellings all measured the same fictional
        # constant -- because no retaining allocation exists: the huge-
        # window short-circuit answers the sentinel with only the
        # interpreter + input string resident. The libc path is therefore
        # unusable as a gate instrument in that environment; VmHWM is the
        # kernel's own high-water mark. Darwin has no /proc and keeps
        # getrusage, truthful there (~29MB dev-box). The gate asserts:
        #   (1) ratio huge-window / small-window < 2.0 -- the regression
        #       pin (retain-all path ~116-177MB / ~29MB ~= 4-6x fails;
        #       fixed ~28MB / ~28MB ~= 1.0 passes; the interpreter
        #       baseline cancels, so box scaling cancels too), and
        #   (2) an absolute ceiling (100MB) as a backstop against a joint
        #       blowup that preserves the ratio -- calibrated ~2.6x above
        #       the worst measured fixed-code peak (38MB) and below every
        #       measured retain-all peak (116MB+).
        import subprocess
        import sys

        def _child_self_peak_mb(prog: str) -> float:
            if sys.platform == "darwin":
                reader = (
                    "import resource; "
                    "peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; "
                    "print(peak / (1024 * 1024))"
                )
            else:
                reader = (
                    "lines = open('/proc/self/status').read().splitlines(); "
                    "hwm = [line for line in lines if line.startswith('VmHWM')]; "
                    "print(int(hwm[0].split()[1]) / 1024)"
                )
            wrapper = prog + "; " + reader
            done = subprocess.run(
                [sys.executable, "-c", wrapper], capture_output=True, text=True
            )
            assert done.returncode == 0, done.stderr[-2000:]
            return float(done.stdout.strip().split()[-1])

        big_expr = "'the quick brown fox jumps over the lazy dog. ' * 300_000"
        huge_prog = (
            f"import tors; big = {big_expr}; "
            "sig = tors.minhash_signature(big, shingle_size=10**9); "
            "assert sig == [2**64 - 1] * 128, 'not sentinel'"
        )
        small_prog = (
            f"import tors; big = {big_expr}; "
            "sig = tors.minhash_signature(big); "
            "assert len(sig) == 128"
        )

        def probe() -> tuple[float, float]:
            huge_mb = _child_self_peak_mb(huge_prog)
            small_mb = _child_self_peak_mb(small_prog)
            return huge_mb, small_mb

        # The wall bound rides along as a tripwire over the two child
        # probes; min-of-3 pass-on-first-clean (tests/loop_harness.py).
        huge_mb, small_mb = assert_bounded(probe, 60.0, samples=3, label="the huge-window probes")
        assert huge_mb < 100.0, f"huge window peak too high: {huge_mb:.1f}MB"
        ratio = huge_mb / small_mb if small_mb > 0 else float("inf")
        assert ratio < 2.0, (
            f"huge window retained the stream: huge {huge_mb:.1f}MB vs "
            f"default-width {small_mb:.1f}MB (ratio {ratio:.2f})"
        )

    def test_values_are_in_the_affine_range_or_sentinel(self) -> None:
        # Every real minimum is an affine output in [0, 2^61 - 1); the only
        # value outside that range the API can ever return is the sentinel.
        sig = minhash_signature(_FOX)
        assert all(0 <= v < 2**61 for v in sig)

    @pytest.mark.parametrize(
        ("tokens", "shingle_size"),
        [(20_000, 5_000), (40_000, 5_000), (100_000, 50_000)],
    )
    def test_fillable_wide_shingle_past_the_sweep_budget_raises_value_error(
        self, tokens: int, shingle_size: int
    ) -> None:
        # The issue-#91 middle range: 1024 < shingle_size <= tokens fills
        # the window, so the sentinel short-circuit cannot decide and every
        # step re-hashes the whole live window -- (tokens - shingle_size +
        # 1) * shingle_size token-hashes, unbounded. The binding counts
        # retention-free and rejects shapes past the 2^26 token-hash budget
        # with a ValueError naming the bound, before the detached pass
        # (these shapes measured 0.6-15s pre-fix).
        text = " ".join(f"w{i}" for i in range(tokens))
        with pytest.raises(ValueError, match=r"token-hash budget"):
            minhash_signature(text, num_perm=8, shingle_size=shingle_size)

    def test_wide_shingle_just_under_the_budget_still_executes(self) -> None:
        # The gate's allowed boundary: (20000 - 4000 + 1) * 4000 = 6.4e7
        # token-hashes, just under the 2^26 budget -- the full sweep runs,
        # and the signature is byte-identical to the pre-fix value (pinned
        # in full; the oracle differential covers the narrow widths).
        text = " ".join(f"w{i}" for i in range(20_000))
        assert minhash_signature(text, num_perm=8, shingle_size=4_000) == [
            280661199516102,
            21719150915306,
            33367424658904,
            119062457005955,
            137317581519668,
            266273482503263,
            39635225421427,
            217535390845298,
        ]

    def test_sweep_budget_boundary_is_exact_on_both_sides(self) -> None:
        # H1: the budget boundary itself. s = 4096 divides 2^26, so
        # 20479 tokens give work exactly (20479 - 4096 + 1) * 4096 = 2^26
        # -- the ceiling is inclusive and the full sweep RUNS, byte-for-
        # byte the pinned signature (release-side execution: the debug
        # cargo suite pins this boundary at the gate, where it is
        # instant). One more token tips (20480 - 4096 + 1) * 4096 =
        # 67,112,960 > 2^26 into the reject -- reported at the gate's
        # walk cap, which here IS the stream length, so the message's
        # "at least 67112960" is also the exact spend.
        at_budget = " ".join(f"w{i}" for i in range(20_479))
        sig = assert_bounded(
            lambda: minhash_signature(at_budget, num_perm=8, shingle_size=4_096),
            5.0,
            samples=3,
            label="the exactly-at-budget sweep",
        )
        assert sig == [
            19447778664110,
            340100930561109,
            60003529736728,
            44457144746778,
            225489404642757,
            21454188447635,
            95003234104397,
            18799151482463,
        ]
        one_past = " ".join(f"w{i}" for i in range(20_480))
        with pytest.raises(ValueError, match=r"at least 67112960 token-hashes"):
            minhash_signature(one_past, num_perm=8, shingle_size=4_096)

    def test_huge_width_sentinel_does_not_trip_the_budget_gate(self) -> None:
        # H5: the sentinel short-circuit runs BEFORE the budget gate -- an
        # unfillable window (however huge the width) answers the sentinel
        # and never raises the sweep-budget ValueError. 10**9 is the
        # pre-existing cell's width; 10**18 is four orders past the budget
        # itself, where a gate-before-short-circuit ordering would reject
        # every fillable-stream probe outright.
        for width in (10**9, 10**18):
            sig = assert_bounded(
                lambda width=width: minhash_signature(_FOX, shingle_size=width),
                1.0,
                samples=3,
                label=f"the width-{width} sentinel",
            )
            assert sig == [_MINHASH_EMPTY] * 128
        # The 1024/1025 boundary of the count-first path: one token short
        # of a 1025-wide window is the sentinel (unfillable -- no gate,
        # no sweep); one token at the width is fillable and executes (the
        # narrowest shape the budget gate even walks the count for).
        short = " ".join(f"w{i}" for i in range(1_024))
        assert minhash_signature(short, num_perm=8, shingle_size=1_025) == [
            _MINHASH_EMPTY
        ] * 8
        full = " ".join(f"w{i}" for i in range(1_025))
        sig = minhash_signature(full, num_perm=8, shingle_size=1_025)
        assert len(sig) == 8
        assert all(v != _MINHASH_EMPTY for v in sig)


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

    def test_index_protocol_seed_is_accepted(self) -> None:
        # Any __index__-only int-like (numpy integers, enums.IntEnum)
        # reduces mod 2**64 like a plain int: the binding must go through
        # the __index__ protocol, never the instance's own __and__.
        class IndexOnly:
            def __init__(self, value: int) -> None:
                self._value = value

            def __index__(self) -> int:
                return self._value

        assert minhash_signature(_FOX, seed=IndexOnly(5)) == minhash_signature(  # type: ignore[arg-type]
            _FOX, seed=5
        )
        assert minhash_signature(_FOX, seed=IndexOnly(-1)) == minhash_signature(  # type: ignore[arg-type]
            _FOX, seed=2**64 - 1
        )
        assert minhash_signature(_FOX, seed=IndexOnly(2**100)) == minhash_signature(  # type: ignore[arg-type]
            _FOX, seed=0
        )

    def test_numpy_int64_seed_is_accepted(self) -> None:
        np = pytest.importorskip("numpy")
        assert minhash_signature(_FOX, seed=np.int64(5)) == minhash_signature(_FOX, seed=5)

    def test_bool_seed_is_rejected(self) -> None:
        # bool IS int, but a seed of True/False is a caller bug magnet, not
        # a seed: rejected explicitly with TypeError.
        with pytest.raises(TypeError):
            minhash_signature(_FOX, seed=True)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            minhash_signature(_FOX, seed=False)  # type: ignore[arg-type]

    def test_index_returning_bool_is_rejected(self) -> None:
        # The __index__ result is launder-checked the same way as the seed
        # itself: an __index__ returning True/False is the same caller bug
        # one dispatch removed (the pre-fix binding reduced it to 1/0).
        class BoolIndex:
            def __index__(self) -> int:
                return True  # type: ignore[return-value]

        with pytest.raises(TypeError):
            minhash_signature(_FOX, seed=BoolIndex())  # type: ignore[arg-type]

    def test_index_that_raises_propagates_its_own_error(self) -> None:
        # __index__ is caller code: its own failure propagates unchanged,
        # never masked as a seed TypeError (the pre-fix binding mapped
        # every __index__ failure, including the caller's own raise, to
        # "seed must be an int").
        class Raising:
            def __index__(self) -> int:
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            minhash_signature(_FOX, seed=Raising())  # type: ignore[arg-type]

    def test_index_without_and_is_accepted_exactly_once(self) -> None:
        # The reduction dispatches the __index__ slot, never the
        # instance's own __and__: an int-like with a hostile __and__
        # reduces identically, and __index__ runs exactly once.
        calls = 0

        class IndexOnly:
            def __init__(self, value: int) -> None:
                self._value = value

            def __index__(self) -> int:
                nonlocal calls
                calls += 1
                return self._value

            def __and__(self, other: object) -> int:
                raise AssertionError("__and__ must never be dispatched")

        assert minhash_signature(_FOX, seed=IndexOnly(5)) == minhash_signature(  # type: ignore[arg-type]
            _FOX, seed=5
        )
        assert calls == 1, f"__index__ dispatched {calls}x"


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
        assert _agreement(a, b) == 74  # exact J 0.6129, estimate 74/128 = 0.5781

    def test_moderately_similar_pair_agreement_is_pinned(self) -> None:
        a = minhash_signature(self._MODERATE_A)
        b = minhash_signature(self._MODERATE_B)
        assert _agreement(a, b) == 22  # exact J 0.2000, estimate 22/128 = 0.1719

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
        assert near - moderate >= 40  # 74 vs 22 measured: the margins are structural

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
        # 118/128 = 0.921875 (the repeated-sentence corpus has only 25
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

    def test_bool_sizes_are_rejected(self) -> None:
        # bool IS int, so True would launder to 1 through a plain i64
        # extraction -- the same caller-bug class as a bool seed, and the
        # binding rejects it the same way (the pre-fix binding accepted
        # num_perm=True as 1 and shingle_size=True as 1).
        with pytest.raises(TypeError):
            minhash_signature(_FOX, num_perm=True)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            minhash_signature(_FOX, num_perm=False)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            minhash_signature(_FOX, shingle_size=True)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            minhash_signature(_FOX, shingle_size=False)  # type: ignore[arg-type]

    def test_huge_int_sizes_raise_overflow_error(self) -> None:
        # An int outside the i64 range the binding extracts raises pyo3's
        # own OverflowError at extraction, never a ValueError -- the
        # truncate_to_bounds-identical pattern api.md documents.
        with pytest.raises(OverflowError):
            minhash_signature(_FOX, num_perm=10**30)  # type: ignore[arg-type]
        with pytest.raises(OverflowError):
            minhash_signature(_FOX, shingle_size=10**30)  # type: ignore[arg-type]

    def test_raising_index_sizes_propagate_their_own_error(self) -> None:
        # Sizes ride the same __index__ protocol as the seed: a raising
        # __index__ propagates unchanged.
        class Raising:
            def __index__(self) -> int:
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            minhash_signature(_FOX, num_perm=Raising())  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="boom"):
            minhash_signature(_FOX, shingle_size=Raising())  # type: ignore[arg-type]

    def test_non_int_shingle_size_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            minhash_signature(_FOX, shingle_size=3.5)  # type: ignore[arg-type]

    def test_return_type_is_list_of_ints(self) -> None:
        sig = minhash_signature(_FOX)
        assert isinstance(sig, list)
        assert all(isinstance(v, int) for v in sig)


class TestPerformanceSanity:
    @staticmethod
    def _distinct_rich(target_bytes: int) -> str:
        # Deterministic distinct-rich corpus: zero-padded hex-counter
        # words, every token unique, so (unlike the repeated-sentence
        # prose) the distinct-shingle set is ~the token count and the
        # O(distinct x num_perm) min-sweep actually runs. The worst-case
        # shape the api.md caller-size bound is calibrated on.
        unit = 9  # "w{:07x} " per word
        n = max(1, target_bytes // unit)
        return "".join(f"w{i:07x} " for i in range(n))

    def test_wide_shingle_timing_row(self) -> None:
        # HIGH2 honesty pin: each step re-hashes the whole window, so the
        # hashing pass is O(tokens x shingle_size), not O(tokens). Wide
        # windows on 100 KiB must still complete -- a tripwire with a wide
        # ceiling, not a target -- and the row documents the scaling the
        # cost formula discloses (dev-box numbers, macOS/arm64: shingle 3
        # ~1.6 ms, 64 and 256 rising ~linearly in the width; see
        # docs/performance.md).
        from reference import prose

        text = prose(100 * 1024)
        for shingle_size in (3, 64, 256):
            sig = assert_bounded(
                lambda s=shingle_size: minhash_signature(text, shingle_size=s),
                30.0,
                samples=3,
                label=f"the shingle-{shingle_size} sweep",
            )
            assert len(sig) == 128

    def test_over_budget_wide_shingle_rejects_fast(self) -> None:
        # The issue-#91 tripwire: pre-fix this shape swept ~2.5e9
        # token-hashes (~15s, GIL released but uninterruptible until
        # completion); post-fix the retention-free count walk rejects it in
        # ~3ms. The ceiling leaves oceans of CI-jitter room yet sits ~7x
        # under the pre-fix wall, so a regression back to the sweep fails
        # it.
        text = " ".join(f"w{i}" for i in range(100_000))

        def reject() -> None:
            with pytest.raises(ValueError, match=r"token-hash budget"):
                minhash_signature(text, num_perm=8, shingle_size=50_000)

        assert_bounded(reject, 2.0, samples=3, label="the over-budget wide-shingle rejection")

    def test_over_budget_reject_walk_is_capped_not_o_tokens(self) -> None:
        # H2: the reject path's count walk stops at the gate's cap --
        # floor(2^26/50000) + 50000 = 51342 tokens -- so the reject does
        # not scale with the stream. The message is the deterministic
        # proof: the reported minimum work at the cap is 1343 * 50000 =
        # 67,150,000 token-hashes, whereas a full-stream walk (pre-fix)
        # reported the exact 2,500,050,000 -- and held the GIL for the
        # whole walk (measured 5.9ms per 100k tokens: ~1.5s GIL-held at
        # 100M tokens, longer than the 0.4s sweep budget the gate exists
        # to enforce). The 300k-token stream rides the same ~3ms reject;
        # the ceiling is a tripwire, the message match is the teeth.
        text = " ".join(f"w{i}" for i in range(300_000))

        def reject() -> None:
            with pytest.raises(ValueError, match=r"at least 67150000 token-hashes"):
                minhash_signature(text, num_perm=8, shingle_size=50_000)

        assert_bounded(reject, 2.0, samples=3, label="the capped reject walk")

    def test_widest_allowed_shingle_completes_quickly(self) -> None:
        # The widest shape the budget still allows on this corpus shape
        # (6.4e7 of the 6.7e7 token-hash budget): the O(tokens x
        # shingle_size) hashing pass runs in full and stays under a
        # generous wall ceiling, ~12x the measured ~0.4s (the wide-shingle
        # row's ceiling style).
        text = " ".join(f"w{i}" for i in range(20_000))
        sig = assert_bounded(
            lambda: minhash_signature(text, num_perm=8, shingle_size=4_000),
            5.0,
            samples=3,
            label="the widest-allowed-shingle sweep",
        )
        assert len(sig) == 8

    @pytest.mark.timing
    # Single-sample absolute ceilings (15s/60s) on a load-sensitive
    # 1 MiB x 1024-perm pass: the timing lane's discipline, the fast
    # lane never reds on a slow runner.
    def test_distinct_rich_worst_case_completes_within_budget(self) -> None:
        # HIGH3 worst-case pin: 1 MiB of distinct-rich text at k=1024 is
        # ~100M affine ops with no deadline_ms on this call, so the lever
        # is the caller's own input size (api.md's bound). The ceiling is
        # ~10x the dev-box nominal (macOS/arm64); the published numbers
        # live in docs/performance.md.
        for num_perm, ceiling in ((128, 15.0), (1024, 60.0)):
            text = self._distinct_rich(1024 * 1024)
            sig = assert_bounded(
                lambda text=text, num_perm=num_perm: minhash_signature(text, num_perm=num_perm),
                ceiling,
                samples=3,
                label=f"the k={num_perm} distinct-rich worst case",
            )
            assert len(sig) == num_perm
            assert all(0 <= v < 2**61 for v in sig)

    def test_large_input_completes_quickly(self) -> None:
        # ~13.5 MB, the simhash sanity cell's corpus shape: the detached
        # tokenize+shingle+hash+sweep pass measures ~0.25 s at k=128 (the
        # dedup-first sweep rides the corpus's handful of distinct
        # shingles; see docs/performance.md), so the 2.5 s ceiling is ~10x
        # nominal: a crash/regression tripwire, not the target.
        big = "the quick brown fox jumps over the lazy dog. " * 300_000
        sig = assert_bounded(
            lambda: minhash_signature(big), 2.5, samples=3, label="the ~13.5MB sweep"
        )
        assert len(sig) == 128

    def test_large_input_is_deterministic(self) -> None:
        big = "lorem ipsum dolor sit amet " * 100_000
        assert minhash_signature(big) == minhash_signature(big)

# --- The literature additions -------------------------------------------------------
#
# The RMSE grid cells below share one corpus builder: sets A, B over a token
# vocabulary with a pinned union cardinality u and Jaccard J, spelled as
# shingle_size=1 texts so the shingle set IS the token set (the estimator's
# true J is then exact by construction). Everything is deterministic (no
# rng beyond the fixed-seed vocabulary shuffle), so the RMSE numbers are
# constants of the build, re-measured here and pinned with margin.


def _jaccard_grid_pair(u: int, j_target: float, seed: int) -> tuple[str, str, float]:
    """Two shingle_size=1 texts over a union of ``u`` distinct tokens with
    Jaccard as close to ``j_target`` as the integer split allows, plus the
    EXACT Jaccard the split realized."""
    rng = random.Random(seed)
    vocab = [f"w{i:05d}" for i in range(u)]
    rng.shuffle(vocab)
    inter = int(round(j_target * u))
    intersection, rest = vocab[:inter], vocab[inter:]
    only_a = rest[: len(rest) // 2]
    only_b = rest[len(rest) // 2 :]
    a_text = " ".join(intersection + only_a)
    b_text = " ".join(intersection + only_b)
    union = len(intersection) + len(only_a) + len(only_b)
    exact = len(intersection) / union
    return a_text, b_text, exact


def _rmse(estimates: list[float], exact: float) -> float:
    return sum((e - exact) ** 2 for e in estimates) ** 0.5 / len(estimates) ** 0.5


class TestSuperMinHashMethod:
    """`method="superminhash"`: Ertl's SuperMinHash (arXiv 1706.05698) as
    the second engine behind `minhash_signature`'s `method` knob. The
    classic default is byte-identical to every pre-method signature ever
    produced (the existing pins above stay green untouched); the new
    engine shares the shingle set, the determinism contract, and the
    empty-set sentinel, and differs in the rows themselves (different
    estimator structure, no prefix property)."""

    def test_deterministic_across_calls_and_seeds(self) -> None:
        text = "the quarterly oil sample interval for field outages"
        a = minhash_signature(text, method="superminhash", num_perm=128)
        assert a == minhash_signature(text, method="superminhash", num_perm=128)
        assert a == minhash_signature("".join(text), method="superminhash", num_perm=128)
        assert a != minhash_signature(text, method="superminhash", num_perm=128, seed=1)
        # The negative-seed reduction rides: seed=-1 is seed=2**64-1.
        assert minhash_signature(text, method="superminhash", seed=-1) == minhash_signature(
            text, method="superminhash", seed=2**64 - 1
        )

    def test_empty_convention_matches_the_classic_sentinel(self) -> None:
        for text in ("", "   ", "\t\n ", "one two"):
            assert minhash_signature(text, method="superminhash") == [_MINHASH_EMPTY] * 128, (
                f"{text!r}"
            )
        for seed in (0, 1, -1, 2**100):
            assert minhash_signature("", method="superminhash", seed=seed) == [_MINHASH_EMPTY] * 128
        # A real document never emits the sentinel pattern: the rows are
        # finite f64 bit patterns (u64 MAX is a NaN pattern), values in
        # [0, m).
        sig = minhash_signature(_FOX, method="superminhash")
        for v in sig:
            h = _row_as_f64(v)
            assert 0.0 <= h < 128.0, f"row {v} decodes to {h}"

    def test_no_prefix_property_across_num_perm(self) -> None:
        # The structural difference the docs state: the SuperMinHash
        # permutation is generated per element over the signature size m,
        # so k=8 is NOT k=128's first 8 rows (the classic engine's prefix
        # property does not transfer).
        full = minhash_signature(_FOX, method="superminhash")
        assert minhash_signature(_FOX, method="superminhash", num_perm=8) != full[:8]

    def test_methods_are_not_cross_compatible(self) -> None:
        # The documented caller error, made visible: the same text at the
        # same parameters through the two engines disagrees on essentially
        # every row (different estimator structures -- the agreement
        # fraction of a mixed pair estimates nothing). Deterministic row:
        # the disagreement is pinned exactly.
        a = minhash_signature(_FOX, method="xxh", num_perm=64)
        b = minhash_signature(_FOX, method="superminhash", num_perm=64)
        assert sum(x == y for x, y in zip(a, b, strict=True)) == 0

    def test_method_validation(self) -> None:
        with pytest.raises(ValueError, match=r'"xxh" or "superminhash"'):
            minhash_signature(_FOX, method="supermin")
        with pytest.raises(ValueError, match=r'"xxh" or "superminhash"'):
            minhash_signature(_FOX, method="")
        for bad in (0, True, None, b"xxh", 1.0):
            with pytest.raises(TypeError):
                minhash_signature(_FOX, method=bad)  # type: ignore[arg-type]

    def test_bits_with_superminhash_is_a_value_error(self) -> None:
        # The b-bit estimator's math needs uniform integer rows; the
        # SuperMinHash rows are f64 bit patterns. The combination is
        # refused, named, before any work.
        with pytest.raises(ValueError, match="superminhash"):
            minhash_signature(_FOX, method="superminhash", bits=8)

    def test_matches_the_pure_python_oracle(self) -> None:
        # The differential pin over the same battery shape the classic
        # oracle cell rides: the algorithm transcription (Algorithm 4's
        # lazy-init permutation, the histogram early exit, the pinned draw
        # order) agrees with an independent pure-Python spelling.
        rows = [
            _FOX,
            "a\r\nb",
            "café société naïve 東京は日本の首都です",
            "ab cd ef " * 50,
            "Hello, world! One. Two.",
            "   ",
        ]
        for text in rows:
            for kwargs in (
                {"num_perm": 8, "shingle_size": 3, "seed": 0},
                {"num_perm": 16, "shingle_size": 2, "seed": 42},
                {"num_perm": 4, "shingle_size": 1, "seed": -1},
            ):
                assert minhash_signature(
                    text, method="superminhash", **kwargs
                ) == reference_superminhash_signature(text, **kwargs), (
                    f"oracle disagreement for {text!r} {kwargs}"
                )

    @given(text=_TEXT, num_perm=_NUM_PERM, shingle_size=_SHINGLE_SIZE, seed=_SEED)
    @settings(max_examples=100)
    def test_matches_the_oracle_hypothesis_lane(self, text, num_perm, shingle_size, seed):
        assert minhash_signature(
            text, method="superminhash", num_perm=num_perm, shingle_size=shingle_size, seed=seed
        ) == reference_superminhash_signature(
            text, num_perm=num_perm, shingle_size=shingle_size, seed=seed
        )

    def test_identical_texts_are_identical_signatures_and_lsh_candidates(self) -> None:
        # The banding consumer's signature contract, on the new engine:
        # identical content -> identical signatures, and identical
        # signatures are always LSH candidates (lsh_candidates bands
        # whatever rows it is given; the near-dup pair is recalled).
        sig = minhash_signature(_FOX, method="superminhash")
        sigs = [sig, sig, minhash_signature(_FOX.replace("fox", "cat"), method="superminhash")]
        out = tors.lsh_candidates(sigs, bands=32, rows=4)
        assert (0, 1) in out["pairs"]

    @pytest.mark.timing
    def test_jaccard_estimation_rmse_grid_both_engines(self) -> None:
        # The estimator-quality cell, the paper's own evaluation shape
        # (section 2.2's simulation grid): sets over union cardinalities
        # u in {30, 100, 1000} at J in {0.1, 0.5, 0.9}, 200 trials per
        # cell, RMSE of the agreement estimator per engine. The pinned
        # claims: both engines are unbiased (every RMSE well inside the
        # classic binomial band sqrt(J(1-J)/128), worst case 0.044), and
        # SuperMinHash's variance advantage is real where the paper says
        # it lives -- u < m (30, 100) with RMSE ~0.7x the classic
        # engine's, converging to parity by u >> m (1000).
        measured: list[tuple[int, float, str, float]] = []
        for method in ("xxh", "superminhash"):
            for u in (30, 100, 1000):
                for j_target in (0.1, 0.5, 0.9):
                    estimates = []
                    exacts = []
                    for trial in range(200):
                        a_text, b_text, exact = _jaccard_grid_pair(u, j_target, seed=trial * 31 + u)
                        sig_a = minhash_signature(
                            a_text, num_perm=128, shingle_size=1, method=method, seed=7
                        )
                        sig_b = minhash_signature(
                            b_text, num_perm=128, shingle_size=1, method=method, seed=7
                        )
                        estimates.append(minhash_jaccard(sig_a, sig_b))
                        exacts.append(exact)
                    exact = sum(exacts) / len(exacts)
                    rmse = _rmse(estimates, exact)
                    measured.append((u, j_target, method, rmse))
        classic = {(u, j): r for (u, j, m, r) in measured if m == "xxh"}
        super_ = {(u, j): r for (u, j, m, r) in measured if m == "superminhash"}
        # Unbiased: every cell inside the classic estimator's 1.5-sigma
        # worst-case band times a safety factor (the measured range spans
        # 0.017-0.050; the ceiling sits above it, below any real bug).
        for rmse in list(classic.values()) + list(super_.values()):
            assert rmse < 0.06, rmse
        # The variance advantage at u < m (the paper's alpha(m, u) < 1):
        # deterministic corpora, so the ordering is a constant.
        for u in (30, 100):
            for j in (0.1, 0.5, 0.9):
                assert super_[(u, j)] < classic[(u, j)], (u, j)
        # ...and convergence at u >> m: the same J cell's ratio nears 1.
        for j in (0.1, 0.5, 0.9):
            ratio = super_[(1000, j)] / classic[(1000, j)]
            assert 0.7 < ratio < 1.15, (j, ratio)

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="no os.fork on this platform")
    def test_fork_child_signatures_are_identical(self) -> None:
        # The fork-safety positive control (test_random.py's TestForkSafety
        # pattern): the engines hold NO per-process state -- a cached RNG
        # or a lazy global would replay the parent's stream in the child
        # or diverge outright. Parent and child compute both engines'
        # signatures of the same fixed text; byte equality IS the
        # argument.
        text = "the quick brown fox jumps over the lazy dog"
        parent = (
            minhash_signature(text, method="xxh", num_perm=64),
            minhash_signature(text, method="superminhash", num_perm=64),
        )
        read_fd, write_fd = os.pipe()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            pid = os.fork()
        if pid == 0:  # child: compute, report over the pipe, exit cleanly.
            try:
                os.close(read_fd)
                child = (
                    minhash_signature(text, method="xxh", num_perm=64),
                    minhash_signature(text, method="superminhash", num_perm=64),
                )
                os.write(write_fd, repr(child).encode("ascii"))
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
        child_xh, child_super = eval(  # noqa: S307 - the child's own repr, fixed shape
            b"".join(chunks).decode("ascii")
        )
        assert child_xh == parent[0]
        assert child_super == parent[1]

    @pytest.mark.timing
    def test_superminhash_pass_keeps_the_event_loop_responsive(self) -> None:
        # The GIL cell, the test_gil_release.py pattern through the shared
        # harness: the whole tokenize + shingle + SuperMinHash sweep runs
        # under one py.detach, so a thread-hopped pass over ~12 MiB of
        # prose leaves the loop at heartbeat granularity (the corpus is
        # the classic cell's 12 MiB shape: a pass this fast at 1 MiB
        # would sit on the 10ms heartbeat cadence's ratio knife-edge).
        from reference import prose

        corpus = prose(12 * 1024 * 1024)
        assert_heartbeat_clean(
            lambda: asyncio.to_thread(minhash_signature, corpus, method="superminhash"),
            subject="the superminhash pass",
        )


def _row_as_f64(v: int) -> float:
    """The f64 a SuperMinHash row's bit pattern carries (the paper's real
    value r + j in [0, m); the row-range decode the cell above reads)."""
    import struct

    return struct.unpack("<d", struct.pack("<Q", v))[0]


class TestBBitCompression:
    """`bits=`: Li and König's b-bit MinHash (WDE 2010). `None` (the
    default) is the full u64 rows, byte-identical to the pre-bits output;
    an int in [1, 63] keeps the lowest b bits of every row, and
    `minhash_jaccard` applies the 2^-b chance correction the masked
    agreement's unbiasedness needs."""

    def test_bits_none_is_byte_identical_to_the_unmasked_output(self) -> None:
        for kwargs in ({}, {"seed": 42}, {"num_perm": 8, "shingle_size": 2, "seed": -1}):
            plain = minhash_signature(_FOX, **kwargs)
            assert minhash_signature(_FOX, bits=None, **kwargs) == plain
            assert minhash_signature(_FOX, method="xxh", bits=None, **kwargs) == plain

    def test_masked_rows_are_the_low_bits_of_the_full_rows(self) -> None:
        full = minhash_signature(_FOX, num_perm=64)
        for bits in (1, 2, 4, 8, 16, 63):
            mask = (1 << bits) - 1
            assert minhash_signature(_FOX, num_perm=64, bits=bits) == [v & mask for v in full], (
                f"bits={bits}"
            )

    @pytest.mark.parametrize("bits", [0, -1, 64, 65, 10**9])
    def test_bits_out_of_bounds_raise_value_error(self, bits: int) -> None:
        with pytest.raises(ValueError, match="between 1 and 63"):
            minhash_signature(_FOX, bits=bits)

    def test_bits_argument_contract(self) -> None:
        # The house argument discipline: bool rejected (a bit-width of
        # True is a caller bug, not 1), non-ints TypeError, huge ints
        # OverflowError at extraction, __index__ int-likes accepted
        # (None is the VALID full-row default and raises nothing).
        for bad in (True, False, "4", 4.0, b"4"):
            with pytest.raises(TypeError):
                minhash_signature(_FOX, bits=bad)  # type: ignore[arg-type]
        with pytest.raises(OverflowError):
            minhash_signature(_FOX, bits=10**30)

        class IndexOnly:
            def __init__(self, value: int) -> None:
                self._value = value

            def __index__(self) -> int:
                return self._value

        assert minhash_signature(_FOX, bits=IndexOnly(4)) == minhash_signature(_FOX, bits=4)

    def test_estimator_matches_the_manual_correction(self) -> None:
        sig_a = minhash_signature(_FOX)
        sig_b = minhash_signature(_FOX.replace("fox", "cat"))
        manual_agree = sum(x == y for x, y in zip(sig_a, sig_b, strict=True))
        assert minhash_jaccard(sig_a, sig_b) == manual_agree / len(sig_a)
        for bits in (1, 2, 4, 8, 16):
            mask = (1 << bits) - 1
            agree = sum(1 for x, y in zip(sig_a, sig_b, strict=True) if x & mask == y & mask)
            p_hat = agree / len(sig_a)
            expected = (p_hat - 2.0**-bits) / (1.0 - 2.0**-bits)
            assert minhash_jaccard(sig_a, sig_b, bits=bits) == pytest.approx(expected, abs=1e-12)
        # Identical signatures estimate 1.0 at every width; the estimator
        # agrees with the pure-Python oracle over both spellings.
        assert minhash_jaccard(sig_a, sig_a, bits=4) == 1.0
        assert minhash_jaccard(sig_a, sig_b, bits=4) == reference_minhash_jaccard(
            sig_a, sig_b, bits=4
        )

    def test_estimator_length_and_shape_contract(self) -> None:
        sig = minhash_signature(_FOX, num_perm=8)
        with pytest.raises(ValueError, match="lengths differ"):
            minhash_jaccard(sig, sig[:4])
        with pytest.raises(ValueError, match="estimate nothing"):
            minhash_jaccard([], [])
        with pytest.raises(ValueError, match="between 1 and 63"):
            minhash_jaccard(sig, sig, bits=0)
        for bad in ([1, True], ["1"], [1, 1.0]):
            with pytest.raises((TypeError, ValueError)):
                minhash_jaccard(bad, bad)  # type: ignore[arg-type]

    @pytest.mark.timing
    def test_bbit_accuracy_vs_bits_ladder(self) -> None:
        # The paper's shape (Li and König, section "accuracy"): RMSE of
        # the b-bit estimator at b = 1, 2, 4, 8, 16 over 300 trials at
        # u = 200, J = 0.5, against the full-row RMSE. The pinned ladder:
        # small b pays the chance term (b=1 measurably worse than full),
        # the ladder converges by b ~ log2(1/J) + 3, and b = 16 is
        # statistically indistinguishable from full rows.
        ladders: dict[object, float] = {}
        for bits in (1, 2, 4, 8, 16, None):
            estimates = []
            exacts = []
            for trial in range(300):
                a_text, b_text, exact = _jaccard_grid_pair(200, 0.5, seed=trial * 17 + 3)
                sig_a = minhash_signature(a_text, num_perm=128, shingle_size=1, seed=7)
                sig_b = minhash_signature(b_text, num_perm=128, shingle_size=1, seed=7)
                estimates.append(minhash_jaccard(sig_a, sig_b, bits=bits))
                exacts.append(exact)
            ladders[bits] = _rmse(estimates, sum(exacts) / len(exacts))
        # The chance term's cost at b=1 is real and bounded; convergence
        # by b=8 (measured: 0.070 at b=1 down to 0.041 by b=8, full rows
        # 0.043); b=16 within 15% of full.
        assert ladders[1] > ladders[4] > ladders[8]
        assert ladders[1] > 0.05
        assert ladders[8] < 0.055
        assert ladders[16] < 0.05
        assert abs(ladders[16] - ladders[None]) < 0.015

    def test_bbit_signatures_band_with_the_corrected_collision_math(self) -> None:
        # lsh_candidates accepts b-bit rows (bands/rows over the masked
        # values -- identical content still produces identical signatures,
        # still always candidates). The collision math the caller needs:
        # per band of r rows the collision probability is
        # (J + (1-J) * 2^-b)^r, not J^r -- the chance term per row lifts
        # the false-positive floor to (2^-b)^r per band, which is why the
        # accuracy contract b >= log2(1/J) doubles as the banding
        # contract. Pinned: the near-dup pair is recalled through the
        # masked rows at b=8, r=2 (chance floor (1/256)^2 ~ 0, and the
        # pair's J ~ 0.85 puts the per-band collision at ~0.73, so the
        # 8-band recall is ~1 - 0.27^8 -- deterministic here because the
        # signatures are), and two DISJOINT documents stay below the floor
        # at that width.
        near = _FOX + " one two three four five six"
        near_a = near * 3
        near_b = (near.replace("quick", "fast")) * 3
        disjoint = "pack my box with five dozen liquor jugs and mixed flour"
        sigs = [
            minhash_signature(near_a, num_perm=16, bits=8),
            minhash_signature(near_b, num_perm=16, bits=8),
            minhash_signature(disjoint, num_perm=16, bits=8),
        ]
        out = tors.lsh_candidates(sigs, bands=8, rows=2)
        assert (0, 1) in out["pairs"]
        assert (0, 2) not in out["pairs"]
        assert (1, 2) not in out["pairs"]
