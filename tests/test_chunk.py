"""Contract gate for ``tors.chunk_cdc``: FastCDC 2020 content-defined
chunking over raw bytes. Unlike every other segmentation primitive in this
crate, offsets here are BYTE spans, not codepoints; there is no text to
respect. The properties below pin: exact partitioning (no gaps, no overlaps,
no chunk shorter than 0 or longer than ``max_size``), determinism, the
empty/shorter-than-``min_size`` special cases, parameter validation, and the
whole reason to reach for content-defined over fixed-size chunking: a small
edit only perturbs the chunks nearest it, not every boundary after it.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import chunk_cdc

_SMALL_PARAMS = {"min_size": 64, "avg_size": 256, "max_size": 1024}


def _xorshift_bytes(n: int, seed: int = 1) -> bytes:
    state = seed | 1
    out = bytearray()
    mask = (1 << 64) - 1
    for _ in range(n):
        state ^= (state << 13) & mask
        state ^= state >> 7
        state ^= (state << 17) & mask
        out.append(state & 0xFF)
    return bytes(out)


class TestEmptyAndShortInput:
    def test_empty_input_is_no_chunks(self) -> None:
        assert chunk_cdc(b"") == []

    def test_input_shorter_than_min_size_is_one_chunk(self) -> None:
        data = b"tiny"
        assert chunk_cdc(data, **_SMALL_PARAMS) == [(0, len(data))]


class TestPartitioning:
    @given(data=st.binary(min_size=0, max_size=20_000))
    @settings(max_examples=200)
    def test_chunks_partition_the_input_exactly(self, data: bytes) -> None:
        chunks = chunk_cdc(data, **_SMALL_PARAMS)
        prev_end = 0
        for start, end in chunks:
            assert start == prev_end
            assert end > start
            assert end - start <= _SMALL_PARAMS["max_size"]
            prev_end = end
        assert prev_end == len(data)

    def test_large_input_partitions_exactly_with_default_params(self) -> None:
        data = _xorshift_bytes(300_000)
        chunks = chunk_cdc(data)
        prev_end = 0
        for start, end in chunks:
            assert start == prev_end
            assert end > start
            prev_end = end
        assert prev_end == len(data)
        assert len(chunks) > 1


class TestDeterminism:
    @given(data=st.binary(max_size=5_000))
    @settings(max_examples=100)
    def test_same_input_same_chunks(self, data: bytes) -> None:
        a = chunk_cdc(data, **_SMALL_PARAMS)
        b = chunk_cdc(data, **_SMALL_PARAMS)
        assert a == b


class TestSmallEditLocality:
    def test_a_small_edit_near_the_start_mostly_preserves_distant_chunks(self) -> None:
        data = bytearray(_xorshift_bytes(400_000, seed=7))
        before = chunk_cdc(bytes(data), **_SMALL_PARAMS)
        data[1000:1000] = bytes([0xAA] * 37)
        after = chunk_cdc(bytes(data), **_SMALL_PARAMS)
        after_set = set(after)

        distant = [(s, e) for s, e in before if s >= 50_000]
        assert len(distant) > 5, "test needs more chunks to be meaningful"
        unchanged = sum(1 for s, e in distant if (s + 37, e + 37) in after_set)
        assert unchanged / len(distant) > 0.8


class TestParameterValidation:
    def test_min_size_below_range_raises(self) -> None:
        with pytest.raises(ValueError, match="min_size"):
            chunk_cdc(b"x" * 100, min_size=63, avg_size=256, max_size=1024)

    def test_min_greater_than_avg_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk_cdc(b"x" * 100, min_size=1024, avg_size=256, max_size=2048)

    def test_avg_greater_than_max_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk_cdc(b"x" * 100, min_size=64, avg_size=2048, max_size=1024)

    def test_odd_size_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk_cdc(b"x" * 100, min_size=65, avg_size=256, max_size=1024)

    def test_default_parameters_are_valid(self) -> None:
        # Would raise if the defaults themselves violated fastcdc's bounds.
        chunk_cdc(b"x" * 100)

    def test_negative_sizes_raise_value_error_not_overflow_error(self) -> None:
        # A raw `usize`-typed pyo3 argument would let a negative Python int
        # raise `OverflowError` instead; the same ValueError contract every
        # other size/count argument across the chunking family (chunk_text's
        # max_chars, chunk_by_words' words_per_chunk, etc.) already gives.
        with pytest.raises(ValueError, match="min_size"):
            chunk_cdc(b"x" * 100, min_size=-1, avg_size=256, max_size=1024)
        with pytest.raises(ValueError, match="avg_size"):
            chunk_cdc(b"x" * 100, min_size=64, avg_size=-1, max_size=1024)
        with pytest.raises(ValueError, match="max_size"):
            chunk_cdc(b"x" * 100, min_size=64, avg_size=256, max_size=-1)


class TestArgumentContract:
    def test_non_bytes_raises(self) -> None:
        with pytest.raises(TypeError):
            chunk_cdc("not bytes")  # type: ignore[arg-type]

    def test_keyword_only_size_params(self) -> None:
        with pytest.raises(TypeError):
            chunk_cdc(b"x" * 100, 64, 256, 1024)  # type: ignore[misc]
