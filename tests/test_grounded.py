"""Contract gate for ``tors.is_grounded``: a LEXICAL claim-grounding check,
not a semantic/NLI one. ``fuzzy=False`` is exact substring containment
(``source.contains(claim)``, Rust's own search; no new dependency);
``fuzzy=True`` is a windowed difflib-ratio scan bounded by ``deadline_ms``,
the same DoS discipline ``diff_opcodes`` already has. See
``src/grounded_impl.rs`` for exactly what the fuzzy score measures.
"""

from __future__ import annotations

import difflib
import re
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import is_grounded

_TEXT = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs"), max_codepoint=0x2FFF),
    max_size=200,
)


class TestExactContainment:
    def test_true_when_claim_is_a_literal_substring(self) -> None:
        assert is_grounded("cat", "the cat sat") is True

    def test_false_when_claim_is_absent(self) -> None:
        assert is_grounded("dog", "the cat sat") is False

    def test_empty_claim_is_vacuously_true(self) -> None:
        assert is_grounded("", "anything") is True
        assert is_grounded("", "") is True

    def test_claim_longer_than_source_is_false(self) -> None:
        assert is_grounded("hello world", "hello") is False

    def test_claim_equals_source_is_true(self) -> None:
        assert is_grounded("exact", "exact") is True

    @given(claim=_TEXT, source=_TEXT)
    @settings(max_examples=300)
    def test_exact_containment_agrees_with_python_in_operator(
        self, claim: str, source: str
    ) -> None:
        assert is_grounded(claim, source) == (claim in source)

    @given(source=_TEXT, claim=_TEXT)
    @settings(max_examples=200)
    def test_a_claim_actually_taken_from_source_is_always_grounded(
        self, source: str, claim: str
    ) -> None:
        # Splice `claim` into the middle of `source`; a claim genuinely
        # drawn from the resulting text must always be reported grounded.
        combined = source[: len(source) // 2] + claim + source[len(source) // 2 :]
        assert is_grounded(claim, combined) is True


class TestFuzzy:
    def test_identical_strings_score_at_the_maximum_threshold(self) -> None:
        assert is_grounded("the cat sat", "the cat sat", fuzzy=True, threshold=1.0) is True

    def test_near_match_inside_a_much_longer_source_clears_a_moderate_threshold(self) -> None:
        source = "Lorem ipsum dolor sit amet. The cats sit on mats today. Consectetur."
        assert is_grounded("the cat sat", source, fuzzy=True, threshold=0.6) is True

    def test_unrelated_text_does_not_clear_a_moderate_threshold(self) -> None:
        assert (
            is_grounded(
                "quantum entanglement", "a recipe for banana bread", fuzzy=True, threshold=0.5
            )
            is False
        )

    def test_threshold_zero_is_always_grounded(self) -> None:
        assert is_grounded("anything", "completely different", fuzzy=True, threshold=0.0) is True

    def test_empty_claim_is_vacuously_true(self) -> None:
        assert is_grounded("", "anything", fuzzy=True, threshold=1.0) is True

    def test_default_threshold_is_0_85(self) -> None:
        assert is_grounded("the cat sat", "the cat sat", fuzzy=True) is True
        assert is_grounded("the cat sat", "a totally different sentence", fuzzy=True) is False

    @given(claim=_TEXT, source=_TEXT)
    @settings(max_examples=300)
    def test_unwindowed_case_is_difflib_shaped_where_difflibs_own_alignment_is_forced(
        self, claim: str, source: str
    ) -> None:
        """The doc contract (src/grounded_impl.rs): when ``source`` is no
        longer than ``claim`` there is nothing to window over, so the score
        is ONE direct ``2*M/T`` ratio. That ``M`` comes from `similar`'s
        Myers engine (the SAME maximal-LCS alignment `similarity_ratio`
        uses, not difflib's own anchored recursion), so "difflib-style" in
        the doc comment describes the FORMULA, not bit-exact parity with
        difflib's alignment (`tests/test_similarity.py`'s validity-first
        discipline applies here for the identical reason: on
        repeated-character inputs the two engines can pick a different,
        equally valid maximal-vs-anchored ``M`` (e.g. ``"010"`` vs
        ``"120"``, difflib's anchored ``M=1`` vs the maximal ``M=2``), a
        real divergence a prior draft of this test wrongly treated as a
        bug). Exact parity IS required wherever difflib's own alignment is
        FORCED: its opcode list carries at most one non-equal op, the same
        gate `test_similarity.py` uses; where difflib itself has no other
        valid answer, any two correct algorithms must agree."""
        if len(source) > len(claim) or not claim:
            return
        ops = difflib.SequenceMatcher(None, claim, source).get_opcodes()
        non_equal = [op for op in ops if op[0] != "equal"]
        if len(non_equal) > 1:
            return
        expected_ratio = difflib.SequenceMatcher(None, claim, source).ratio()
        for threshold in (0.0, expected_ratio, min(expected_ratio + 0.01, 1.0), 1.0):
            expected = expected_ratio >= threshold or threshold == 0.0
            assert (
                is_grounded(claim, source, fuzzy=True, threshold=threshold) == expected
            ), (claim, source, threshold, expected_ratio)

    def test_unwindowed_divergence_row_both_engines_valid_but_different(self) -> None:
        """The pinned real divergence found while fixing the test above:
        difflib's anchored alignment gives ``"010"``/``"120"`` a smaller
        (still valid) ``M=1`` (ratio ``1/3``), while tors's maximal ``M=2``
        (ratio ``2/3``) is realizable too (``"10"`` is a genuine common
        subsequence of both). A threshold between the two must clear on
        tors's side and would NOT clear difflib's; pinned so this class
        of input stays a known, intentional design point, not a silent
        regression risk."""
        assert difflib.SequenceMatcher(None, "010", "120").ratio() == pytest.approx(1 / 3)
        assert is_grounded("010", "120", fuzzy=True, threshold=0.5) is True
        assert is_grounded("010", "120", fuzzy=True, threshold=0.7) is False

    @given(claim=_TEXT, source=_TEXT)
    @settings(max_examples=200)
    def test_threshold_is_monotonic(self, claim: str, source: str) -> None:
        """If a (claim, source) pair clears a higher threshold, it must
        clear every lower one too; the fuzzy verdict is a single scalar
        score compared against ``threshold``, so this must hold regardless
        of the windowing internals."""
        hi, lo = 0.9, 0.3
        cleared_hi = is_grounded(claim, source, fuzzy=True, threshold=hi)
        cleared_lo = is_grounded(claim, source, fuzzy=True, threshold=lo)
        if cleared_hi:
            assert cleared_lo

    def test_a_verbatim_substring_is_grounded_at_every_offset(self) -> None:
        """fuzzy=True is a superset of exact containment: a claim present
        verbatim in source must be grounded wherever it sits. Before the
        exact-containment floor, stride-L/2 windowing straddled the claim at
        unaligned offsets and scored ~0.75 < the 0.85 default, wrongly
        rejecting a literal substring."""
        # ASCII plus a multi-byte claim: str::contains is UTF-8-boundary-safe
        # and the floor runs before any char-window code, so a verbatim claim
        # is grounded regardless of encoding or offset.
        for claim in (
            "the quick brown fox jumps over lazy dog",
            "café über naïve résumé — 速い茶色の狐",
        ):
            for lead in range(41):
                source = ("x" * lead) + claim + ("x" * 30)
                assert claim in source  # verbatim-present
                assert is_grounded(claim, source, fuzzy=True)
                assert is_grounded(claim, source, fuzzy=True, threshold=1.0)


class TestArgumentContract:
    @pytest.mark.parametrize("bad", [-0.1, 1.1], ids=["below-zero", "above-one"])
    def test_out_of_range_threshold_raises_value_error(self, bad: float) -> None:
        with pytest.raises(ValueError, match="threshold"):
            is_grounded("a", "b", fuzzy=True, threshold=bad)

    def test_deadline_ms_without_fuzzy_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="deadline_ms"):
            is_grounded("a", "b", deadline_ms=100.0)

    @pytest.mark.parametrize("bad", [0.0, -50.0], ids=["zero", "negative"])
    def test_nonpositive_deadline_raises_value_error(self, bad: float) -> None:
        with pytest.raises(ValueError, match="deadline_ms"):
            is_grounded("a", "b", fuzzy=True, deadline_ms=bad)

    @pytest.mark.parametrize(
        "not_str", [b"abc", bytearray(b"abc"), 123, None], ids=["bytes", "bytearray", "int", "none"]
    )
    @pytest.mark.parametrize("which", ["claim", "source"])
    def test_non_str_arguments_raise_type_error(self, not_str: object, which: str) -> None:
        """``claim``/``source`` are ordinary ``&str`` pyo3 arguments (same
        boundary as every other str-in function in this crate), never
        pinned for this function before."""
        good = "abc"
        args = (not_str, good) if which == "claim" else (good, not_str)
        with pytest.raises(TypeError):
            is_grounded(*args)  # type: ignore[arg-type]

    def test_lone_surrogates_are_refused_at_the_argument_boundary(self) -> None:
        with pytest.raises(UnicodeEncodeError):
            is_grounded("cat\ud800", "the cat sat")
        with pytest.raises(UnicodeEncodeError):
            is_grounded("cat", "the cat\ud800 sat")


# --- deadline_ms (bounding the windowed fuzzy scan's worst case) --------------------

_DEADLINE_MS = 0.001


class TestDeadline:
    def test_an_effectively_zero_budget_raises_timeout_error(self) -> None:
        claim = "x" * 2_000
        source = "y" * 200_000
        started = time.perf_counter()
        with pytest.raises(TimeoutError, match="deadline") as excinfo:
            is_grounded(claim, source, fuzzy=True, threshold=1.0, deadline_ms=_DEADLINE_MS)
        wall = time.perf_counter() - started
        message = str(excinfo.value)
        assert type(excinfo.value) is TimeoutError
        assert re.search(r"elapsed \d+(\.\d+)?\s*ms", message), message
        assert wall < 2.0, f"deadline-bounded call took {wall:.2f}s"

    def test_a_generous_deadline_never_expires(self) -> None:
        # A NEAR-match, deliberately not a verbatim substring, so the scan
        # still traverses the windowed path (the exact-containment floor would
        # otherwise short-circuit before the deadline machinery is exercised).
        assert (
            is_grounded(
                "the cet sat",
                "the cat sat on the mat",
                fuzzy=True,
                threshold=0.5,
                deadline_ms=60_000.0,
            )
            is True
        )

    def test_a_verbatim_claim_is_grounded_before_the_deadline_applies(self) -> None:
        # Complement of the zero-budget timeout test above (which uses a
        # NON-substring claim): the exact-containment floor short-circuits
        # before deadline setup, so a verbatim substring is grounded even under
        # an effectively-zero budget — never TimeoutError.
        claim = "x" * 2_000
        source = ("y" * 200_000) + claim + ("y" * 200_000)
        assert (
            is_grounded(claim, source, fuzzy=True, threshold=1.0, deadline_ms=_DEADLINE_MS)
            is True
        )
