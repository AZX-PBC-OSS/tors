"""Contract gate for the similarity pair: ``tors.similarity_ratio`` and
``tors.get_close_matches``: difflib's shape over the same Myers engine as
``diff_opcodes``, at native speed with the GIL released.

``tors.similarity_ratio(a, b)`` is ``2.0 * M / T`` (``T = len(a) +
len(b)``) with ``M`` the matched-character total over the Myers equal-ops
(``difflib.SequenceMatcher(None, a, b).ratio()``'s formula) over a
DIFFERENT alignment engine, so the parity contract is VALIDITY-FIRST, in
three parts (the ``diff_opcodes`` discipline, no false parity: the
algorithms are different and both are right):

1. **Both engines' values are valid**: every ``M`` is realizable as a
   common subsequence of the two operands (``M <= LCS(a, b)``), and
   tors's is MAXIMAL: ``M == LCS(a, b)`` exactly, pinned as a
   hypothesis differential against a pure-Python LCS oracle below (the
   minimal-edit-script consequence of the Myers engine: minimal edits
   ⟺ maximal matches).
2. **Exact agreement where the alignment is FORCED**: identical operands,
   empty pairs, disjoint alphabets, and the pure insert/delete classes
   with differing flanks (verified, not assumed, by gating on difflib's
   OWN canonical presentation (its opcode list carries exactly one
   non-equal op of the expected tag: where difflib itself is forced
   there, any two correct algorithms must agree). The
   ``tests/test_diff_opcodes.py`` assume-gating idiom, applied to the
   scalar.
3. **Pinned divergence rows, both engines' values asserted with the
   mechanism note**: difflib's longest-match recursion ANCHORS a match
   and splits the change around it, giving a smaller ``M`` than the
   maximal one on repeated-pattern contexts: ``"ppp"`` vs ``"pwpp"``:
   difflib ``4/7`` (its anchored ``"pp"`` splits the insert, ``M = 2``)
   vs tors ``6/7`` (``M = 3 = LCS``); ``"qpqpq"`` vs ``"qpwqpq"``:
   difflib ``6/11`` (the anchored ROTATED equal block ``"qpq"``,
   non-minimal insert+delete split, ``M = 3``) vs tors ``10/11``
   (``M = 5 = LCS``). Both values are valid similarity ratios; tors's
   is the minimal-edit one. Difflib's ratio is direction-symmetric on
   these rows (measured), so both orders are pinned, but NOT in
   general: its anchored ``M`` is direction-DEPENDENT
   (``SequenceMatcher(None, "baab", "abab").ratio()`` is 0.75, the
   swapped order 0.5, measured and pinned below), and
   ``get_close_matches`` scores candidates as ``a=candidate, b=word``,
   the direction the list differential below gates on.

Degenerates are difflib's own: ``("", "")`` → ``1.0`` (``T = 0``, the
convention both engines share), ``("", "x")`` → ``0.0``, identical →
``1.0``, disjoint alphabets → ``0.0``. The scalar is bounded ``[0, 1]``,
symmetric, and ``1.0`` iff ``a == b``, all pinned over hypothesis pairs.

``tors.get_close_matches(word, possibilities, n=3, cutoff=0.6)`` is
difflib's shape over tors's own ratio: every candidate scoring ``>=
cutoff`` is kept, and the top ``n`` come back sorted by
``heapq.nlargest``'s TUPLE order (score descending, then candidate
STRING descending), the stdlib quirk that puts ``"ca"`` before ``"ac"``
for ``get_close_matches("ab", ["ac", "ca"], 2, 0.5)``. The returned
elements are the ORIGINAL candidate objects (references, not copies).
Because the underlying ratio diverges from difflib's on the ambiguous
classes (part 3), full-list parity with difflib is claimed only where
the two score functions agree per candidate (assumed, then asserted);
the nlargest SHAPE is pinned structurally against ``heapq.nlargest``
over tors's own scores.

Argument-boundary pins (measured, difflib's messages in mind):

- ``n <= 0`` (zero OR negative) and an out-of-range ``cutoff`` raise
  ``ValueError`` with difflib's exact message, value interpolated
  (``"n must be > 0: 0"``, ``"n must be > 0: -1"``, ``"cutoff must be
  in [0.0, 1.0]"``, measured on 3.10-3.15). ``n`` is taken SIGNED at
  the pyo3 boundary specifically so this validation runs before any
  unsigned-extraction failure could raise ``OverflowError`` instead,
  so a caller's ``except ValueError`` guard catches every ``n <= 0`` case
  identically to difflib's own.
- ``deadline_ms`` (both functions): ``TimeoutError`` on expiry naming
  the elapsed time and the deadline; ``ValueError`` for zero, negative,
  NaN or infinity; a huge-but-finite budget is legal and saturates to
  unbounded (identical results to the default); non-numeric values
  raise ``TypeError``.
"""

from __future__ import annotations

import difflib
import heapq
import keyword
import re
import time

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from reference import diff_pair_char_shuffled
from tors import get_close_matches, similarity_ratio

# The slow pair for the deadline cells: ~120k chars of char-shuffled
# prose, whose unbounded diff costs ~1.5-2 s (the diff_opcodes deadline
# cell's own bracket), ~30x over the 50 ms budget the cells set.
_DEADLINE_PAIR_BYTES = 120_000
_DEADLINE_MS = 50.0


def _difflib_ratio(a: str, b: str) -> float:
    """The oracle: exactly the stdlib expression ``tors.similarity_ratio``
    replaces."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def _difflib_opcodes(a: str, b: str) -> list[tuple[str, int, int, int, int]]:
    return difflib.SequenceMatcher(None, a, b).get_opcodes()


def _sole_non_equal_op(
    ops: list[tuple[str, int, int, int, int]], tag: str
) -> tuple[str, int, int, int, int] | None:
    """The canonical single-op shape predicate (the diff_opcodes gate's
    idiom): ``ops`` carries EXACTLY ONE non-equal op, its tag is ``tag``,
    every other op an equal, the canonical presentation of one
    contiguous change, where the minimal edit script is forced and any
    two correct algorithms must agree. ``None`` for any other shape,
    including difflib's anchored non-minimal splits."""
    non_equal = [op for op in ops if op[0] != "equal"]
    if len(non_equal) != 1 or non_equal[0][0] != tag:
        return None
    return non_equal[0]


def _lcs_length(a: str, b: str) -> int:
    """A longest-common-subsequence length by the classic DP, the
    independent oracle for the maximality claim: the matched total of a
    minimal edit script is exactly the LCS length, and no valid
    alignment's can exceed it."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for char_a in a:
        current = [0] * (len(b) + 1)
        for j, char_b in enumerate(b):
            if char_a == char_b:
                current[j + 1] = previous[j] + 1
            else:
                current[j + 1] = max(previous[j + 1], current[j])
        previous = current
    return previous[len(b)]


# The pinned divergence rows: where difflib's anchored alignment gives a
# smaller (valid, non-maximal) M than the Myers minimal one (the contract's
# part 3, and the diff_opcodes gate's own boundary pairs: the same
# mechanism, seen through the scalar).
_DIVERGENT_PAIRS = {("ppp", "pwpp"), ("qpqpq", "qpwqpq")}


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("", "", 1.0),
        ("", "x", 0.0),
        ("x", "", 0.0),
        ("abc", "abc", 1.0),
        ("abc", "xyz", 0.0),
        ("kitten", "sitting", 8.0 / 13.0),
        ("qabxcd", "abYcd", 0.7272727272727273),
        ("café", "cafe", 0.75),
        ("東京", "京都", 0.5),
        ("hello world", "hello there world", 0.7857142857142857),
        # The divergence rows: tors's (maximal-M) side of the pair.
        ("ppp", "pwpp", 6.0 / 7.0),
        ("qpqpq", "qpwqpq", 10.0 / 11.0),
    ],
    ids=[
        "empty-pair",
        "empty-vs-nonempty",
        "nonempty-vs-empty",
        "identical",
        "disjoint-alphabets",
        "kitten-sitting",
        "delete-replace-mix",
        "accent-delete",
        "cjk-shared-char",
        "word-insert",
        "divergent-equal-run-slide",
        "divergent-repeated-flank-rotate",
    ],
)
def test_golden_battery(a: str, b: str, expected: float) -> None:
    """The fixed anchor: every row asserts tors's exact value, and the
    UNAMBIGUOUS rows (everything except the pinned divergence pairs)
    must also equal the running difflib's ratio, the scalar twin of the
    diff_opcodes battery discipline."""
    assert similarity_ratio(a, b) == expected
    if (a, b) not in _DIVERGENT_PAIRS:
        assert similarity_ratio(a, b) == _difflib_ratio(a, b)


def test_the_documented_divergence_rows_both_engines_values_pinned() -> None:
    """The divergence rows, both engines' exact values asserted so the
    differences can never become silent drift; the same anchored-split
    mechanism the diff_opcodes gate pins, seen through the scalar, with
    each side's validity verified against the LCS bound (both M's are
    realizable; tors's is maximal, difflib's is not).

    ``"ppp"`` vs ``"pwpp"``: difflib's find_longest_match anchors the
    ``"pp"`` and splits the insertion around it (an insert and a delete,
    ``M = 2`` → ``4/7``); Myers slides the equal run and isolates the
    single insertion (``M = 3 = LCS`` → ``6/7``).

    ``"qpqpq"`` vs ``"qpwqpq"``: the repeated-flank rotation: the suffix
    repeats the prefix's head, difflib anchors the ROTATED longer equal
    block (``"qpq"``, non-minimal insert+delete split, ``M = 3`` →
    ``6/11``); tors emits the minimal contiguous insert (``M = 5 = LCS``
    → ``10/11``)."""
    for a, b in _DIVERGENT_PAIRS:
        tors_ratio = similarity_ratio(a, b)
        difflib_ratio = _difflib_ratio(a, b)
        total = len(a) + len(b)
        lcs = _lcs_length(a, b)
        assert tors_ratio == 2.0 * lcs / total
        # difflib's M: realizable (<= LCS) but strictly smaller here.
        difflib_m = difflib_ratio * total / 2.0
        assert difflib_m <= lcs
        assert difflib_m < lcs
        # Direction symmetry, measured on both engines for these rows.
        assert similarity_ratio(b, a) == tors_ratio
        assert _difflib_ratio(b, a) == difflib_ratio
    assert similarity_ratio("ppp", "pwpp") == 6.0 / 7.0
    assert _difflib_ratio("ppp", "pwpp") == 4.0 / 7.0
    assert similarity_ratio("qpqpq", "qpwqpq") == 10.0 / 11.0
    assert _difflib_ratio("qpqpq", "qpwqpq") == 6.0 / 11.0


def test_difflibs_ratio_is_direction_dependent_and_tors_is_not() -> None:
    """The measured stdlib fact the get_close_matches differential below
    had to respect: difflib's anchored ``M`` is NOT symmetric in its
    operands: ``SequenceMatcher(None, "baab", "abab").ratio()`` is 0.75
    while the swapped order gives 0.5 (the anchored recursion finds a
    different, equally valid, smaller-``M`` alignment from the other
    direction), and ``difflib.get_close_matches`` scores its candidates
    with ``a=candidate, b=word`` (``set_seq2(word)`` once, ``set_seq1(x)``
    per candidate), the direction a naive ``SequenceMatcher(None, word,
    x)`` oracle gets WRONG. tors's LCS-based scalar is symmetric by
    construction (pinned over hypothesis pairs below), which is also why
    full-list parity with difflib can only be claimed where the two
    engines' per-candidate scores agree in the direction
    ``get_close_matches`` actually uses."""
    assert _difflib_ratio("baab", "abab") == 0.75
    assert _difflib_ratio("abab", "baab") == 0.5
    assert similarity_ratio("baab", "abab") == similarity_ratio("abab", "baab") == 0.75


# --- The forced-class parity properties (the assume-gating idiom) ----------------------
#
# Generator alphabets are DISJOINT by class (context over one alphabet,
# the changed content over another), so the changed block's content is
# forced into the non-equal ops, but difflib's PRESENTATION is not
# (repeated flanks let it anchor a rotated block and split the change,
# the divergence rows' mechanism). The unambiguous class is
# difflib's own answer, so exact parity is asserted only where difflib
# emits the canonical single-op shape.
_CONTEXT_ALPHABET = "pqrs"
_CHANGED_ALPHABET = "xyzw"


@st.composite
def _pure_insert_pair(draw: st.DrawFn) -> tuple[str, str]:
    prefix = draw(st.text(alphabet=_CONTEXT_ALPHABET, max_size=8))
    suffix = draw(st.text(alphabet=_CONTEXT_ALPHABET, max_size=8))
    inserted = draw(st.text(alphabet=_CHANGED_ALPHABET, min_size=1, max_size=8))
    # The one-character flank guard excludes the adjacent-slide shape.
    assume(not prefix or not suffix or prefix[-1] != suffix[0])
    return prefix + suffix, prefix + inserted + suffix


@st.composite
def _pure_delete_pair(draw: st.DrawFn) -> tuple[str, str]:
    prefix = draw(st.text(alphabet=_CONTEXT_ALPHABET, max_size=8))
    suffix = draw(st.text(alphabet=_CONTEXT_ALPHABET, max_size=8))
    deleted = draw(st.text(alphabet=_CHANGED_ALPHABET, min_size=1, max_size=8))
    assume(not prefix or not suffix or prefix[-1] != suffix[0])
    return prefix + deleted + suffix, prefix + suffix


@given(_pure_insert_pair())
@settings(max_examples=400)
def test_pure_inserts_match_difflib_where_the_alignment_is_forced(pair: tuple[str, str]) -> None:
    """Generated pure-insert pairs, gated to the unambiguous
    subset: exact agreement is asserted ONLY where difflib's own answer
    carries the canonical single-insert shape: there ``M`` is forced
    (the whole inserted block is unmatched, everything else matches) and
    any two correct algorithms must produce the same ratio. The gate
    excludes the repeated-flank class where difflib's anchored split
    drops its ``M`` below the LCS (the pinned rows); tors's answer there
    is carried by the LCS-maximality property below, not by this
    claim."""
    a, b = pair
    assume(_sole_non_equal_op(_difflib_opcodes(a, b), "insert") is not None)
    assert similarity_ratio(a, b) == _difflib_ratio(a, b)


@given(_pure_delete_pair())
@settings(max_examples=400)
def test_pure_deletes_match_difflib_where_the_alignment_is_forced(pair: tuple[str, str]) -> None:
    """The mirror of the insert class, same gating: exact agreement only
    where difflib itself presents the canonical single-delete shape."""
    a, b = pair
    assume(_sole_non_equal_op(_difflib_opcodes(a, b), "delete") is not None)
    assert similarity_ratio(a, b) == _difflib_ratio(a, b)


@given(st.text(max_size=24))
@settings(max_examples=300)
def test_identical_pairs_agree_with_difflib(text: str) -> None:
    """Identical operands (arbitrary small text, non-ASCII and the empty
    string included): both engines give ``1.0`` (or the empty pair's
    ``1.0`` convention, the one degenerate where ``T = 0``)."""
    assert similarity_ratio(text, text) == _difflib_ratio(text, text) == 1.0


# --- The validity-first oracle and structural properties -------------------------------


@given(st.text(max_size=24), st.text(max_size=24))
@settings(max_examples=300)
def test_ratio_is_two_lcs_over_total_over_arbitrary_pairs(a: str, b: str) -> None:
    """The maximality differential, the strongest form of validity-first:
    over ARBITRARY pairs (any alphabets, shared or disjoint), tors's
    matched total is exactly the LCS length (the minimal-edit-script
    consequence: minimal edits ⟺ maximal matches), so the scalar is
    ``2.0 * LCS(a, b) / (len(a) + len(b))`` with the empty pair's
    ``1.0`` convention. An anchoring-dependent ``M`` (difflib's) breaks
    this on the first repeated-flank pair it draws; a validity bug of
    any kind breaks it on the first draw at all."""
    if not a and not b:
        assert similarity_ratio(a, b) == 1.0
        return
    expected = 2.0 * _lcs_length(a, b) / (len(a) + len(b))
    assert similarity_ratio(a, b) == expected


@given(st.text(max_size=24), st.text(max_size=24))
@settings(max_examples=300)
def test_bounds_symmetry_and_identity_over_arbitrary_pairs(a: str, b: str) -> None:
    """The structural properties: bounded in ``[0.0, 1.0]``,
    direction-symmetric, ``1.0`` exactly when the operands are equal
    (and the empty pair), never above ``1.0`` even for pathological
    repeated content."""
    ratio = similarity_ratio(a, b)
    assert 0.0 <= ratio <= 1.0
    assert similarity_ratio(b, a) == ratio
    assert (ratio == 1.0) == (a == b)


# --- The deadline contract ------------------------------------------------------------


class TestSimilarityRatioDeadline:
    def test_a_slow_pair_exceeding_the_deadline_raises_timeout_error(self) -> None:
        """``deadline_ms`` bounds the whole call: the char-shuffled pair's
        unbounded ratio costs ~1.5-2 s (measured, the diff_opcodes
        deadline cell's bracket), ~30x over the 50 ms budget, and the
        deadline fires with ``TimeoutError`` naming the elapsed cost and
        the deadline, the incomplete result discarded."""
        a, b = diff_pair_char_shuffled(_DEADLINE_PAIR_BYTES)
        started = time.perf_counter()
        with pytest.raises(TimeoutError, match="deadline") as excinfo:
            similarity_ratio(a, b, deadline_ms=_DEADLINE_MS)
        wall = time.perf_counter() - started
        assert type(excinfo.value) is TimeoutError
        message = str(excinfo.value)
        assert re.search(r"elapsed \d+(\.\d+)?\s*ms", message), message
        # The deadline bounded the call (the unbounded ratio is ~30x the
        # budget; generous upper bound for a loaded runner's dispatch).
        assert wall < 1.0, f"deadline-bounded call took {wall:.2f}s"

    def test_deadline_none_is_the_default_and_unchanged(self) -> None:
        """``deadline_ms=None`` is exactly the unbounded behavior; the
        parameter is purely additive."""
        assert similarity_ratio("kitten", "sitting") == similarity_ratio(
            "kitten", "sitting", deadline_ms=None
        )

    def test_a_generous_deadline_yields_the_identical_ratio(self) -> None:
        """A deadline the search never reaches changes NOTHING: a
        far-future budget saturates to unbounded and the scalar is
        byte-identical to the default call."""
        assert similarity_ratio(
            "qpqpq", "qpwqpq", deadline_ms=60_000.0
        ) == similarity_ratio("qpqpq", "qpwqpq")

    @pytest.mark.parametrize("bad", [0.0, -50.0, float("nan"), float("inf")])
    def test_nonpositive_or_nonfinite_deadlines_raise_value_error(self, bad: float) -> None:
        """A budget must be positive and finite: zero, negative, NaN and
        infinity are caller bugs, not instant timeouts or an unbounded
        alias; refused with the exact message before any work runs."""
        with pytest.raises(
            ValueError, match="^deadline_ms must be a positive finite number of milliseconds$"
        ):
            similarity_ratio("abc", "abd", deadline_ms=bad)

    def test_a_non_numeric_deadline_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            similarity_ratio("abc", "abd", deadline_ms="50")  # type: ignore[arg-type]


# --- get_close_matches ----------------------------------------------------------------


class TestGetCloseMatches:
    @pytest.mark.parametrize(
        ("word", "possibilities", "kwargs", "expected"),
        [
            # difflib's own docstring battery, verbatim (the keyword list
            # is the stdlib docstring's own corpus).
            ("appel", ["ape", "apple", "peach", "puppy"], {}, ["apple", "ape"]),
            ("wheel", keyword.kwlist, {}, ["while"]),
            ("Apple", keyword.kwlist, {}, []),
            ("accept", keyword.kwlist, {}, ["except"]),
            # The tie-order quirk: equal scores come back candidate-string
            # DESCENDING (heapq.nlargest's tuple order): "ca" before "ac".
            ("ab", ["ac", "ca"], {"n": 2, "cutoff": 0.5}, ["ca", "ac"]),
            # n larger than the matching set returns every match.
            ("ab", ["ac", "ca"], {"n": 10, "cutoff": 0.5}, ["ca", "ac"]),
            # The cutoff boundary: 0.0 keeps everything, 1.0 keeps only
            # exact matches.
            ("ab", ["ab", "ac"], {"cutoff": 0.0}, ["ab", "ac"]),
            ("ab", ["ab", "ac"], {"cutoff": 1.0}, ["ab"]),
            # Empty possibilities: empty answer, no error.
            ("ab", [], {}, []),
        ],
        ids=[
            "difflib-docstring-appel",
            "difflib-docstring-wheel-kwlist",
            "difflib-docstring-apple-case-sensitive-kwlist",
            "difflib-docstring-accept-kwlist",
            "tie-order-nlargest-quirk",
            "n-exceeds-matches",
            "cutoff-zero-keeps-all",
            "cutoff-one-exact-only",
            "empty-possibilities",
        ],
    )
    def test_docstring_battery(
        self,
        word: str,
        possibilities: list[str],
        kwargs: dict[str, int | float],
        expected: list[str],
    ) -> None:
        """The difflib docstring battery (its own kwlist corpus) plus the
        shape edges (tie order, ``n`` over the match count, the cutoff
        boundary, empty input): every row equals difflib's own answer;
        these pools sit in the forced-alignment classes, so the two
        engines' per-candidate scores agree where it counts (at or above
        the cutoff) and the selection, order, and length must agree with
        them."""
        tors_result = get_close_matches(word, possibilities, **kwargs)
        stdlib_result = difflib.get_close_matches(word, possibilities, **kwargs)
        assert tors_result == expected
        assert tors_result == stdlib_result

    def test_returns_the_original_candidate_objects(self) -> None:
        """The zero-marshalling claim as an object-level fact: the
        returned elements ARE the caller's candidate objects
        (references, not copies or fresh equals), so identity-sensitive
        callers (interning assumptions, ``id`` keys) see their own
        objects come back."""
        possibilities = ["ape", "apple", "peach", "puppy"]
        result = get_close_matches("appel", possibilities, 3, 0.4)
        assert result[0] == "apple"
        assert result[0] is possibilities[1]
        assert all(any(candidate is original for original in possibilities) for candidate in result)

    @given(
        st.text(alphabet="ab", min_size=0, max_size=6),
        st.lists(
            st.text(alphabet="ab", min_size=0, max_size=6),
            min_size=0,
            max_size=8,
            unique=True,
        ),
        st.integers(min_value=1, max_value=5),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=300)
    def test_matches_difflib_wherever_the_two_score_functions_agree(
        self, word: str, possibilities: list[str], n: int, cutoff: float
    ) -> None:
        """The parity differential for the LIST, gated in the
        direction that matters: difflib's ``get_close_matches`` scores
        each candidate as ``SequenceMatcher(None, candidate, word)``
        (``set_seq2(word)`` once, ``set_seq1(x)`` per candidate), and its
        anchored ratio is direction-DEPENDENT (pinned above), so the gate
        compares against that exact spelling. Where the two score
        functions agree on every candidate (the forced-alignment
        classes), the selection, the order (ties included), and the
        length must all equal difflib's; where they do not (the
        anchored-split classes, the pinned divergence mechanism), the
        lists legitimately differ; that divergence is recorded, not
        asserted away."""
        assume(
            all(
                similarity_ratio(word, candidate) == _difflib_ratio(candidate, word)
                for candidate in possibilities
            )
        )
        assert get_close_matches(word, possibilities, n, cutoff) == difflib.get_close_matches(
            word, possibilities, n, cutoff
        )

    @given(
        st.text(alphabet="abc", min_size=0, max_size=6),
        st.lists(
            st.text(alphabet="abc", min_size=0, max_size=6),
            min_size=0,
            max_size=8,
            unique=True,
        ),
        st.integers(min_value=1, max_value=5),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=300)
    def test_is_nlargest_over_its_own_scores_with_the_difflib_filter(
        self, word: str, possibilities: list[str], n: int, cutoff: float
    ) -> None:
        """The structural pin of the SELECTION SHAPE (no difflib needed):
        the answer is exactly ``heapq.nlargest(n, [(score, candidate) for
        candidate in possibilities if score >= cutoff])`` over tors's OWN
        similarity_ratio, which is difflib's own construction
        (``result.append((s.ratio(), x))`` then ``_nlargest``), tie order
        and all: nlargest compares the tuples, so equal scores order by
        candidate string DESCENDING. Where the score functions diverge
        the two engines' lists diverge with them; the SHAPE never
        does."""
        scored = [
            (similarity_ratio(word, candidate), candidate)
            for candidate in possibilities
            if similarity_ratio(word, candidate) >= cutoff
        ]
        expected = [candidate for _, candidate in heapq.nlargest(n, scored)]
        assert get_close_matches(word, possibilities, n, cutoff) == expected

    def test_n_zero_matches_difflibs_value_error_exactly(self) -> None:
        """``n = 0`` is refused with difflib's exact message, value
        interpolated: ``"n must be > 0: 0"`` (measured on 3.10-3.15
        here). ``n`` is taken SIGNED specifically so this validation runs
        before any unsigned-extraction failure could pre-empt it (see the
        negative-``n`` test below)."""
        with pytest.raises(ValueError, match=r"^n must be > 0: 0$"):
            get_close_matches("ab", ["abc"], n=0)

    def test_negative_n_matches_difflib_exactly(self) -> None:
        """The negative-``n`` boundary now matches difflib exactly:
        both raise ``ValueError("n must be > 0: -1")``, so a caller's
        ``except ValueError`` guard catches it identically. Previously
        tors's UNSIGNED argument extraction refused the negative int
        before the ``n > 0`` check ran, surfacing ``OverflowError``
        instead; ``n`` is now taken signed at the pyo3 boundary so the
        stdlib-shaped validation runs first."""
        with pytest.raises(ValueError, match=r"^n must be > 0: -1$"):
            difflib.get_close_matches("ab", ["ac"], n=-1)
        with pytest.raises(ValueError, match=r"^n must be > 0: -1$"):
            get_close_matches("ab", ["ac"], n=-1)

    @pytest.mark.parametrize("cutoff", [-0.1, 1.1, -1.0, 2.0])
    def test_out_of_range_cutoffs_raise_value_error(self, cutoff: float) -> None:
        """``cutoff`` outside ``[0.0, 1.0]`` is refused with difflib's own
        message INCLUDING the interpolated value (verified against the
        running stdlib; the interpolation was added with the ``n``-validation
        fix, and both messages are now difflib-exact). The boundary values
        0.0 and 1.0 are LEGAL, pinned by the battery's cutoff rows
        above."""
        with pytest.raises(ValueError, match="^cutoff must be in \\[0.0, 1.0\\]: -?\\d"):
            get_close_matches("ab", ["abc"], cutoff=cutoff)

    def test_a_slow_pair_exceeding_the_deadline_raises_timeout_error(self) -> None:
        """The deadline bounds the whole call: the first candidate whose
        ratio search cannot finish inside the budget aborts the sweep
        with ``TimeoutError`` (the word here is the char-shuffled hard
        pair's a-side, its partner the first candidate)."""
        a, b = diff_pair_char_shuffled(_DEADLINE_PAIR_BYTES)
        started = time.perf_counter()
        with pytest.raises(TimeoutError, match="deadline"):
            get_close_matches(a, [b, "zzz"], deadline_ms=_DEADLINE_MS)
        assert time.perf_counter() - started < 1.0

    def test_a_generous_deadline_yields_the_identical_list(self) -> None:
        """A far-future budget saturates to unbounded: the answer is
        byte-identical to the default call, and ``None`` is the default
        (purely additive parameter)."""
        possibilities = ["ape", "apple", "peach", "puppy"]
        assert get_close_matches(
            "appel", possibilities, deadline_ms=60_000.0
        ) == get_close_matches("appel", possibilities)
        assert get_close_matches("appel", possibilities, deadline_ms=None) == get_close_matches(
            "appel", possibilities
        )

    @pytest.mark.parametrize("bad", [0.0, -50.0, float("nan"), float("inf")])
    def test_nonpositive_or_nonfinite_deadlines_raise_value_error(self, bad: float) -> None:
        with pytest.raises(
            ValueError, match="^deadline_ms must be a positive finite number of milliseconds$"
        ):
            get_close_matches("ab", ["abc"], deadline_ms=bad)

    def test_non_str_word_or_possibilities_raise_type_error(self) -> None:
        """The argument shapes: ``word`` is exactly ``str`` and
        ``possibilities`` exactly ``list`` of ``str`` (difflib accepts any
        sequence and non-str sequences for word; the fuzzy-match
        signature keeps the typed surface; a tuple of the right strings
        still raises)."""
        bad_calls = [
            lambda: get_close_matches(123, ["abc"]),
            lambda: get_close_matches(None, ["abc"]),
            lambda: get_close_matches("ab", ("a", "b")),
            lambda: get_close_matches("ab", "abc"),
            lambda: get_close_matches("ab", None),
            lambda: get_close_matches("ab", [1, 2]),
        ]
        for bad_call in bad_calls:
            with pytest.raises(TypeError):
                bad_call()

    def test_non_numeric_deadline_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            get_close_matches("ab", ["abc"], deadline_ms="50")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_n", ["3", 3.0, None])
    def test_non_int_n_raises_type_error(self, bad_n: object) -> None:
        """``n`` and ``cutoff`` are the only two parameters of this function
        without a type-boundary test (word, possibilities, and deadline_ms
each have one above); ``n: isize`` is
        extracted the same pyo3 way as everything else that IS tested."""
        with pytest.raises(TypeError):
            get_close_matches("ab", ["abc"], n=bad_n)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_cutoff", ["0.6", None])
    def test_non_float_cutoff_raises_type_error(self, bad_cutoff: object) -> None:
        with pytest.raises(TypeError):
            get_close_matches("ab", ["abc"], cutoff=bad_cutoff)  # type: ignore[arg-type]

    def test_duplicate_candidates_keep_input_order_on_ties(self) -> None:
        """Documented tie-breaking (src/py/fuzzy.rs, src/diff_impl.rs): equal
        ``(score, string)`` pairs keep input order, pinned by a Rust unit
        test but never exercised at the Python boundary, since every
        hypothesis strategy below uses ``unique=True`` candidates and no
        hand-written row repeats a candidate."""
        assert get_close_matches("ab", ["ac", "ac"], n=2, cutoff=0.5) == ["ac", "ac"]

    def test_duplicate_candidates_are_distinct_returned_objects(self) -> None:
        """The zero-marshalling identity claim holds per-occurrence too:
        two equal-valued but distinct candidate objects both come back as
        themselves, not collapsed to one."""
        a, b = "ac", "ac"[:]  # distinct str objects with equal value
        possibilities = [a, b]
        result = get_close_matches("ab", possibilities, n=2, cutoff=0.5)
        assert result == ["ac", "ac"]

    def test_unicode_candidates_are_scored_by_codepoint_not_byte(self) -> None:
        """Every sibling fuzzy function (jaro/jaro_winkler/levenshtein/
        similarity_ratio) carries an explicit multi-byte case (café, 東京,
        emoji); this function scores candidates through the exact same
        char-level matched_chars engine (src/diff_impl.rs) but had zero
        Unicode coverage, so a char-vs-byte boundary regression reachable
        only through the candidate-list path would go undetected."""
        possibilities = ["café", "cafe", "东京", "🦀crab", "unrelated"]
        result = get_close_matches("café", possibilities, n=5, cutoff=0.0)
        assert result[0] == "café"
        stdlib_result = difflib.get_close_matches("café", possibilities, n=5, cutoff=0.0)
        assert result == stdlib_result
