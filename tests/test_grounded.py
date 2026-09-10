"""Contract gate for ``tors.is_grounded``: a lexical claim-grounding check,
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

from reference import (  # noqa: I001 -- the shared oracle module (tests/reference.py)
    reference_is_grounded_fuzzy,
)
from tors import is_grounded

_TEXT = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs"), max_codepoint=0x2FFF),
    max_size=200,
)


@st.composite
def _spliced(draw):
    """A claim embedded verbatim in a source at an arbitrary offset (the
    `lead`/`tail` padding from a single-alphabet filler): the shape the
    exact-containment floor exists for. Drawing `lead` freely is the point:
    window alignment mod the stride-``L/2`` grid is exactly the dimension
    hand-rolled offset sweeps cannot cover densely."""
    claim = draw(_TEXT)
    lead = draw(st.integers(0, 120))
    tail = draw(st.integers(0, 60))
    return claim, ("x" * lead) + claim + ("x" * tail)


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
        # Splice `claim` into the middle of `source`; a claim
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
        is one direct ``2*M/T`` ratio. That ``M`` comes from `similar`'s
        Myers engine (the same maximal-LCS alignment `similarity_ratio`
        uses, not difflib's own anchored recursion), so "difflib-style" in
        the doc comment describes the formula, not bit-exact parity with
        difflib's alignment (`tests/test_similarity.py`'s validity-first
        discipline applies here for the identical reason: on
        repeated-character inputs the two engines can pick a different,
        equally valid maximal-vs-anchored ``M`` (e.g. ``"010"`` vs
        ``"120"``, difflib's anchored ``M=1`` vs the maximal ``M=2``), a
        real divergence a prior draft of this test wrongly treated as a
        bug). Exact parity is required wherever difflib's own alignment is
        forced: its opcode list carries at most one non-equal op, the same
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
            assert is_grounded(claim, source, fuzzy=True, threshold=threshold) == expected, (
                claim,
                source,
                threshold,
                expected_ratio,
            )

    def test_unwindowed_divergence_row_both_engines_valid_but_different(self) -> None:
        """The pinned real divergence found while fixing the test above:
        difflib's anchored alignment gives ``"010"``/``"120"`` a smaller
        (still valid) ``M=1`` (ratio ``1/3``), while tors's maximal ``M=2``
        (ratio ``2/3``) is realizable too (``"10"`` is a genuine common
        subsequence of both). A threshold between the two must clear on
        tors's side and would not clear difflib's; pinned so this class
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

    @given(claim_source=_spliced())
    @settings(max_examples=300)
    def test_a_spliced_claim_is_fuzzy_grounded_at_every_threshold(self, claim_source) -> None:
        """The superset contract as a property, not just the hand-rolled
        offset sweep above: a claim present verbatim at an arbitrary offset
        is grounded at every threshold (threshold cannot demote a literal
        substring), and before the deadline applies (the floor runs ahead of
        deadline setup, so even an effectively-zero budget must ground a
        verbatim claim, never TimeoutError). This is the red/green row for
        the original bug: it fails on the pre-floor windowed scan wherever
        the offset lands ~L/4 into the stride grid."""
        claim, source = claim_source
        assert claim in source  # the splice is verbatim by construction
        for threshold in (1.0, 0.85, 0.5):
            assert is_grounded(claim, source, fuzzy=True, threshold=threshold)
        assert is_grounded(claim, source, fuzzy=True, threshold=1.0, deadline_ms=_DEADLINE_MS)

    @given(claim=_TEXT, source=_TEXT)
    @settings(max_examples=300)
    def test_threshold_one_fuzzy_is_exactly_exact_containment(
        self, claim: str, source: str
    ) -> None:
        """Both directions of the equivalence the .pyi states ("threshold=1.0
        fuzzy subsumes exact containment"): a verbatim substring must clear
        1.0 (the floor), and a non-substring must never clear it (a window
        only scores 1.0 when it equals the claim, which is containment by
        another name; shorter truncated tail windows and the
        source-shorter-than-claim direct comparison are both strictly < 1).
        Pins the pair of invariants the floor's own tests leave one-sided."""
        assert is_grounded(claim, source, fuzzy=True, threshold=1.0) == (claim in source), (
            claim,
            source,
        )

    def test_a_near_exact_non_substring_clears_085_but_never_a_threshold_of_one(self) -> None:
        """One substitution, so the exact-containment floor cannot
        short-circuit and the windowed scorer must produce the verdict.
        Restores the scorer coverage the floor removed from the
        identical-strings test above (a verbatim claim now returns at the
        floor before any window is diffed) and pins the no-false-positive
        direction at threshold=1.0 alongside it."""
        source = "the cat sat on the mat"
        assert "the cet sat" not in source
        assert is_grounded("the cet sat", source, fuzzy=True, threshold=0.85) is True
        assert is_grounded("the cet sat", source, fuzzy=True, threshold=1.0) is False

    def test_a_one_typo_near_match_at_any_offset_clears_the_default(self) -> None:
        """The refinement pass's headline guarantee: a same-length source
        region matching the claim with ratio r is detected at any offset
        whenever r >= max(0.75, threshold + 1/32). One substitution at L=41
        gives r = 40/41 = 0.976, comfortably above 0.85 + 1/32, so every
        offset must clear the default. Before refinement, stride-L/2
        windowing straddled the region at unaligned offsets (nearest window
        overlapping only ~3L/4, score ~0.73) and wrongly rejected it: the
        same defect class the exact-containment floor fixed for verbatim
        claims, here fixed for near matches."""
        claim = "the bushing torque specifications changed"  # 41 chars
        near = "the bushing torqxe specifications changed"  # one substitution
        assert claim not in near
        for lead in range(41):
            source = ("q" * lead) + near + ("q" * 60)
            assert is_grounded(claim, source, fuzzy=True) is True, f"lead {lead}"

    def test_three_typos_at_the_worst_alignment_still_clear_the_default(self) -> None:
        # r = 38/41 = 0.927 >= 0.85 + 1/32: still inside the guaranteed band,
        # even at the worst stride alignment (lead 10 = L/4 into the grid).
        claim = "the bushing torque specifications changed"
        near = "".join("Z" if i in (5, 18, 33) else c for i, c in enumerate(claim))
        source = ("q" * 10) + near + ("q" * 60)
        assert is_grounded(claim, source, fuzzy=True) is True

    def test_the_detection_margin_is_pinned(self) -> None:
        """The guarantee is sufficient, not necessary: r >= 0.85 + 1/32 is
        the worst-case fine-grid misalignment bound, and at L=41 the fine
        stride is 2, so a region whose start lands on the grid is scored
        exactly aligned. Pinned at the measured edge: six substitutions
        (r = 0.854, below the worst-case line) still clear the default
        here; seven (r = 0.829, below the threshold outright) do not, and
        clear 0.8 where the guarantee covers them; with the oracle's
        agreement asserted on every row."""
        claim = "the bushing torque specifications changed"
        for d, at_default, at_08 in ((6, True, True), (7, False, True)):
            step = 41 // d
            chars = list(claim)
            for k in range(d):
                chars[k * step] = "Z"
            near = "".join(chars)
            source = ("q" * 10) + near + ("q" * 60)
            for threshold, expected in ((0.85, at_default), (0.8, at_08)):
                got = is_grounded(claim, source, fuzzy=True, threshold=threshold)
                assert got is expected, (d, threshold)
                assert got == reference_is_grounded_fuzzy(claim, source, threshold), (d, threshold)

    def test_a_near_tail_region_is_detected_through_the_tail_window_candidate(self) -> None:
        """The tail row: a five-typo region (r = 0.878) whose aligned start
        (76) sits past the last full grid window (60) and 4 chars before
        the truncated tail window (80). No full window reaches it: the
        nearest, [60, 101), overlaps only 25 of the region's 41 chars and
        scores ~0.49 (below the 0.5 candidate entry) so the full-grid
        candidates never cover the region. The truncated tail window's own
        score (~0.79: it holds 37 of the region's chars against the
        denominator 41 + 40) puts it in the candidate band like any other,
        and its refinement range clamps at the last possible region start
        n - L = 79: covering the aligned start 76, which the fine grid
        scans. n = 120: grid 0/20/40/60 full, truncated tail [80, 120);
        region [76, 117). The tail competes like an ordinary candidate:
        score-gated, evictable, no forced mechanism (one existed briefly
        and was removed; the brute-force geometry search found no
        guarantee-band region needing it). The flush variant (region
        ending exactly at the source's end, start = n - L) is pinned
        alongside so the two tail geometries stay distinguishable."""
        claim = "the bushing torque specifications changed"
        near = "".join("Z" if i in (3, 11, 19, 27, 35) else c for i, c in enumerate(claim))
        near_tail = ("q" * 76) + near + ("q" * 3)
        flush = ("q" * 79) + near
        assert len(near_tail) == 120 and claim not in near_tail
        for source in (near_tail, flush):
            assert is_grounded(claim, source, fuzzy=True) is True
            assert reference_is_grounded_fuzzy(claim, source, 0.85) is True

    def test_candidate_eviction_at_the_64th_band_window_is_the_pinned_flood_limit(self) -> None:
        """The 64-candidate cap is the documented adversarial limit: the
        top (score, start) band windows are all the refinement will ever
        see, so a real near-match whose straddled coarse score (~0.73)
        ranks below 64 decoys scoring above it (seven-typo variants,
        r = 0.829, at grid-aligned offsets) is evicted and the verdict is
        false despite r = 40/41 >= the guarantee: the regime deadline_ms
        exists for. The boundary is exact (63 decoys still find it), and
        the oracle mirrors the eviction identically on both sides."""
        claim = "the bushing torque specifications changed"
        real = "the bushing torqxe specifications changed"
        decoy = "".join("Z" if i in (2, 8, 14, 20, 26, 32, 38) else c for i, c in enumerate(claim))

        def build(n_decoys: int) -> str:
            parts = [("q" * 10) + real]
            at = 60  # every decoy starts at a multiple of the stride, 20
            for _ in range(n_decoys):
                have = sum(len(p) for p in parts)
                parts.append(("q" * (at - have)) + decoy)
                at += 60
            parts.append("q" * 60)
            return "".join(parts)

        for n, expected in ((63, True), (64, False)):
            source = build(n)
            got = is_grounded(claim, source, fuzzy=True)
            assert got is expected, n
            assert got == reference_is_grounded_fuzzy(claim, source, 0.85), n

    @given(claim=_TEXT, index=st.data())
    @settings(max_examples=200)
    def test_a_one_substitution_near_match_at_any_offset_is_grounded(self, claim, index) -> None:
        """The guarantee as a property: splice a one-substitution near-match
        of the claim into padding at an arbitrary offset; it must clear the
        default threshold wherever it sits. Claim length is drawn >= 12 so
        r = (L-1)/L >= 0.917 stays above the 0.85 + 1/32 guarantee line.
        This is the red/green row for the refinement pass: it fails on the
        pre-refinement scan wherever the offset lands misaligned."""
        if len(claim) < 12:
            return
        i = index.draw(st.integers(0, len(claim) - 1))
        sub = index.draw(st.sampled_from("qz9"))
        near = claim[:i] + sub + claim[i + 1 :]
        lead = index.draw(st.integers(0, 120))
        tail = index.draw(st.integers(0, 60))
        source = ("q" * lead) + near + ("q" * tail)
        assert is_grounded(claim, source, fuzzy=True, threshold=0.85) is True, (claim, near, lead)

    def test_empty_source_is_ungrounded_except_at_threshold_zero(self) -> None:
        # The empty-source convention (no window to score, the verdict is
        # 0.0 against the threshold): ungrounded at any positive threshold,
        # vacuously grounded at exactly 0.0.
        assert is_grounded("x", "", fuzzy=True, threshold=0.85) is False
        assert is_grounded("claim", "", fuzzy=True, threshold=0.5) is False
        assert is_grounded("x", "", fuzzy=True, threshold=0.0) is True

    @given(claim=_TEXT, source=_TEXT)
    @settings(max_examples=100)
    def test_fuzzy_verdicts_match_the_lcs_window_model_on_realistic_text(
        self, claim: str, source: str
    ) -> None:
        """The full-contract differential oracle (tests/reference.py's
        reference_is_grounded_fuzzy): floor, windowing, stride, truncated
        tail window, the bounded refinement pass (candidate top-K, fine
        grid, capped ranges), and the threshold comparison all modeled
        independently in pure Python, with M from an LCS DP rather than any
        difflib or Myers machinery. Agreement here proves the whole
        algorithm's verdict, not just the pieces the pinned rows cover."""
        for threshold in (1.0, 0.85):
            assert is_grounded(claim, source, fuzzy=True, threshold=threshold) == (
                reference_is_grounded_fuzzy(claim, source, threshold)
            ), (claim, source, threshold)

    @given(
        claim=st.text(alphabet="ab", max_size=20),
        source=st.text(alphabet="ab", max_size=80),
    )
    @settings(max_examples=200)
    def test_fuzzy_verdicts_match_the_lcs_window_model_on_repeated_characters(
        self, claim: str, source: str
    ) -> None:
        """The same oracle over a two-letter alphabet: maximally repeated
        characters are exactly where difflib's anchored alignment and the
        maximal-LCS alignment diverge (the pinned "010"/"120" class), so
        this lane proves tors's M == LCS claim holds on the hard class
        while the realistic-text lane above covers the ordinary shapes."""
        for threshold in (1.0, 0.85, 0.6, 0.3):
            assert is_grounded(claim, source, fuzzy=True, threshold=threshold) == (
                reference_is_grounded_fuzzy(claim, source, threshold)
            ), (claim, source, threshold)


class TestArgumentContract:
    @pytest.mark.parametrize("bad", [-0.1, 1.1], ids=["below-zero", "above-one"])
    def test_out_of_range_threshold_raises_value_error(self, bad: float) -> None:
        with pytest.raises(ValueError, match="threshold"):
            is_grounded("a", "b", fuzzy=True, threshold=bad)

    def test_validation_precedes_the_exact_containment_floor(self) -> None:
        """The floor returns before deadline setup, but argument validation
        is still ahead of it (the pyo3 layer validates threshold and
        deadline_ms before any work runs): a verbatim claim (which the
        floor grounds in microseconds) cannot bypass a bad threshold or a
        nonpositive deadline_ms. Pins the ordering the "a verbatim claim
        never times out" doc leans on: never-timeout is a property of valid
        budgets, not of skipped validation."""
        assert is_grounded("cat", "the cat sat", fuzzy=True) is True  # the floor's lane
        with pytest.raises(ValueError, match="threshold"):
            is_grounded("cat", "the cat sat", fuzzy=True, threshold=1.1)
        with pytest.raises(ValueError, match="deadline_ms"):
            is_grounded("cat", "the cat sat", fuzzy=True, deadline_ms=0.0)

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
        # A near-match, deliberately not a verbatim substring, so the scan
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
        # non-substring claim): the exact-containment floor short-circuits
        # before deadline setup, so a verbatim substring is grounded even under
        # an effectively-zero budget: never TimeoutError.
        claim = "x" * 2_000
        source = ("y" * 200_000) + claim + ("y" * 200_000)
        assert (
            is_grounded(claim, source, fuzzy=True, threshold=1.0, deadline_ms=_DEADLINE_MS) is True
        )

    def test_the_refinement_pass_respects_a_generous_deadline(self) -> None:
        # The straddled one-typo geometry (lead 10: coarse best ~0.73 < 0.85,
        # only the refinement finds it) under a generous budget: the
        # refinement's per-window deadline checks must all pass and the
        # verdict arrive: the deadline plumbing on the refinement path,
        # which the zero-budget test above never reaches (it expires at the
        # coarse stage). 60s against microsecond work is machine-immune.
        claim = "the bushing torque specifications changed"
        near = "the bushing torqxe specifications changed"
        source = ("q" * 10) + near + ("q" * 60)
        assert is_grounded(claim, source, fuzzy=True, threshold=0.85, deadline_ms=60_000.0) is True
