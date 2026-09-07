"""Contract gate for the fuzzy metrics: ``tors.levenshtein``,
``tors.jaro``, ``tors.jaro_winkler``: the edit-distance and
string-similarity scalars CPython has no stdlib spelling of, at native
speed with the GIL released.

All three are CHARACTER-LEVEL over ``str`` operands (a ``str`` IS a
character sequence here (the ``diff_opcodes``/``similarity_ratio``
convention, and what makes the surrogate boundary the standard str-in
one); SYMMETRIC in their operands, and pinned three ways:

1. **Known vectors** (the literature's and the strim crate's own): the
   classic Jaro/Jaro-Winkler pairs (MARTHA/MARHTA → jaro ≈ 0.944,
   jw ≈ 0.961; DIXON/DICKSONX; dwayne/duane; saturday/sunday) and the
   Levenshtein anchors (kitten/sitting = 3, flaw/lawn = 2,
   gumbo/gambol = 2), plus the Winkler mechanics rows: the prefix bonus
   applies only above the 0.7 boost threshold and only over the first
   FOUR shared characters (``café``/``cafe``: jw > jaro from the
   3-character prefix; ``flaw``/``lawn``: no shared prefix, jw == jaro).
2. **Independent pure-Python oracles over hypothesis pairs** (any
   script, any width, non-ASCII included): a two-row DP for
   Levenshtein (unit costs, substitution = 1) and the textbook
   match-window Jaro (plus the Winkler boost on top): exact integer
   agreement for the distance, ``1e-12`` for the floats' summation
   order.
3. **Structural properties**: symmetry for all three; bounds
   ``|len(a) - len(b)| <= levenshtein <= max(len(a), len(b))`` and
   ``0 <= jaro <= jaro_winkler <= 1``; the identity characterizations
   (``levenshtein == 0`` iff equal, ``jaro == 1.0`` iff equal; the
   empty pair included by convention); degenerates
   (empty-vs-nonempty → distance ``len(x)``, similarity ``0.0``;
   disjoint alphabets → ``0.0``).

``deadline_ms`` (all three) bounds the whole DP pass with per-phase
checks: a genuinely hard pair (~120k chars of char-shuffled prose,
whose O(n*m) table is ~1.4e10 cells and minutes of work) raises
``TimeoutError`` at the 50 ms budget naming the elapsed time and the
deadline; zero, negative, NaN and infinity raise ``ValueError`` (a
budget must be positive and finite); a huge-but-finite budget is legal
and saturates to unbounded (identical results to the default; the
diff deadline contract's own saturation pin); non-numeric values raise
``TypeError``. The default (``None``) is the unbounded behavior,
purely additive.

The GIL claim (the whole DP under ``py.detach``, a single int/float
out, so no marshalling class at all) is the crate GIL model's; the
DoS-shaped budget is why the parameter exists: two 1 MiB strings are
minutes of DP, and the metrics gate measures the hard pair's timeout
directly.
"""

from __future__ import annotations

import re
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reference import diff_pair_char_shuffled
from tors import jaro, jaro_winkler, levenshtein

_DEADLINE_PAIR_BYTES = 120_000
_DEADLINE_MS = 50.0

# The known vectors: (a, b, expected_levenshtein, expected_jaro,
# expected_jaro_winkler). Float expectations are the mathematical values
# (fraction-exact), asserted at rel=1e-12 to absorb summation order.
_VECTORS: list[tuple[str, str, int, float, float]] = [
    ("", "", 0, 1.0, 1.0),
    ("", "abc", 3, 0.0, 0.0),
    ("abc", "", 3, 0.0, 0.0),
    ("abc", "abc", 0, 1.0, 1.0),
    # The classic trio, every Jaro/Jaro-Winkler paper's rows.
    ("MARTHA", "MARHTA", 2, 17.0 / 18.0, 173.0 / 180.0),
    ("DIXON", "DICKSONX", 4, 23.0 / 30.0, 0.8133333333333332),
    ("dwayne", "duane", 2, 0.8222222222222222, 0.84),
    # The Winkler mechanics rows.
    ("saturday", "sunday", 3, 0.7527777777777777, 0.7775),
    ("café", "cafe", 1, 0.8333333333333333, 0.8833333333333333),
    ("flaw", "lawn", 2, 0.8333333333333333, 0.8333333333333333),
    # Levenshtein anchors.
    ("kitten", "sitting", 3, 0.746031746031746, 0.746031746031746),
    ("gumbo", "gambol", 2, 0.8222222222222222, 0.84),
    # Non-ASCII: character-level, every UTF-8 width.
    ("東京", "京都", 2, 0.0, 0.0),
    ("ééé", "éé", 1, 0.8888888888888888, 0.9111111111111111),
]


@pytest.mark.parametrize(
    ("a", "b", "expected_lev", "expected_jaro", "expected_jw"),
    _VECTORS,
    ids=[f"{a!a}-{b!a}" for a, b, _, _, _ in _VECTORS],
)
def test_known_vectors(
    a: str, b: str, expected_lev: int, expected_jaro: float, expected_jw: float
) -> None:
    """The fixed anchor: the literature vectors and the Winkler mechanics
    rows (prefix bonus above the 0.7 threshold, capped at four shared
    characters; no shared prefix → jw == jaro; disjoint alphabets and the
    empty pair → the degenerate conventions). The distance is exact; the
    similarities at ``rel=1e-12`` (summation order, not value)."""
    assert levenshtein(a, b) == expected_lev
    assert jaro(a, b) == pytest.approx(expected_jaro, rel=1e-12)
    assert jaro_winkler(a, b) == pytest.approx(expected_jw, rel=1e-12)


# --- The independent oracles ----------------------------------------------------------


def _reference_levenshtein(a: str, b: str) -> int:
    """The textbook two-row DP, unit costs (insert/delete/substitute each
    1), the independent oracle for the distance."""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a):
        current = [i + 1] + [0] * len(b)
        for j, char_b in enumerate(b):
            current[j + 1] = min(
                previous[j + 1] + 1,
                current[j] + 1,
                previous[j] + (char_a != char_b),
            )
        previous = current
    return previous[len(b)]


def _reference_jaro(a: str, b: str) -> float:
    """The textbook Jaro: matches within the ``max(la, lb)//2 - 1``
    window, half the out-of-order matched pairs as transpositions, the
    three-term mean, the independent oracle for the similarity."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    # The window is never negative (one-character operands).
    window = max(0, max(la, lb) // 2 - 1)
    a_matched = [False] * la
    b_matched = [False] * lb
    matches = 0
    for i, char_a in enumerate(a):
        for j in range(max(0, i - window), min(i + window + 1, lb)):
            if not b_matched[j] and char_a == b[j]:
                a_matched[i] = b_matched[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    transpositions = 0
    k = 0
    for i in range(la):
        if a_matched[i]:
            while not b_matched[k]:
                k += 1
            if a[i] != b[k]:
                transpositions += 1
            k += 1
    # INTEGER division, matching strsim's own `transpositions /= 2` (verified
    # against strsim 0.11.1's vendored source); the count is NOT always
    # even (a matched-character cycle longer than a 2-cycle, e.g. a 3-cycle
    # rotation, yields an odd mismatch count: "102" vs "021000" is the
    # pinned adversarial case below), so this oracle must floor exactly like
    # the Rust side to stay a real differential rather than a false failure.
    transpositions //= 2
    return (matches / la + matches / lb + (matches - transpositions) / matches) / 3


def _reference_jaro_winkler(a: str, b: str) -> float:
    """The Winkler boost on the oracle Jaro: above the 0.7 boost
    threshold, ``prefix * 0.1 * (1 - jaro)`` with the prefix capped at
    four shared leading characters."""
    jaro_score = _reference_jaro(a, b)
    if jaro_score <= 0.7:
        return jaro_score
    prefix = 0
    for char_a, char_b in zip(a[:4], b[:4], strict=False):
        if char_a != char_b:
            break
        prefix += 1
    return jaro_score + prefix * 0.1 * (1.0 - jaro_score)


# --- The hypothesis differentials -----------------------------------------------------


@given(st.text(max_size=32), st.text(max_size=32))
@settings(max_examples=400)
def test_levenshtein_matches_the_dp_oracle(a: str, b: str) -> None:
    """The distance differential: exact integer agreement with the
    independent DP over arbitrary pairs (any script, mixed widths,
    combining marks): a sub/ins/del cost slip or a char-vs-byte
    miscount breaks the first draw."""
    assert levenshtein(a, b) == _reference_levenshtein(a, b)


@given(st.text(max_size=24), st.text(max_size=24))
@settings(max_examples=400)
def test_jaro_matches_the_pure_python_oracle(a: str, b: str) -> None:
    """The Jaro differential: the match window, the transposition
    halving, and the three-term mean must agree with the textbook
    implementation to ``1e-12`` over arbitrary pairs, including the
    empty-pair conventions both spellings share."""
    assert jaro(a, b) == pytest.approx(_reference_jaro(a, b), rel=1e-12, abs=1e-12)


def test_jaro_odd_mismatch_count_floors_not_rounds() -> None:
    """``"102"`` vs ``"021000"``: the matched-character alignment is a
    3-cycle rotation (``'1','0','2'`` against ``'0','2','1'`` in b's
    index order), giving an ODD transposition-mismatch count of 3, a
    case hypothesis found that falsifies the tempting "always even"
    assumption. Both tors and strsim floor via integer division
    (``3 // 2 == 1``), landing on ``0.7222...``, not the ``0.6667...``
    a float-division (``3 / 2 == 1.5``) reading would produce; the
    exact bug this test was added to catch after a prior reference-oracle
    regression used float division here."""
    assert jaro("102", "021000") == pytest.approx(0.7222222222222222)
    assert _reference_jaro("102", "021000") == pytest.approx(0.7222222222222222)


@given(st.text(max_size=24), st.text(max_size=24))
@settings(max_examples=400)
def test_jaro_winkler_matches_the_pure_python_oracle(a: str, b: str) -> None:
    """The Winkler differential: the boost threshold and the four-char
    prefix cap on top of the Jaro oracle; the rows where a naive
    always-boost or uncapped-prefix implementation diverge are drawn
    here, not just pinned in the battery."""
    assert jaro_winkler(a, b) == pytest.approx(
        _reference_jaro_winkler(a, b), rel=1e-12, abs=1e-12
    )


@given(st.text(max_size=24), st.text(max_size=24))
@settings(max_examples=300)
def test_symmetry_bounds_and_identity_over_arbitrary_pairs(a: str, b: str) -> None:
    """The structural properties: all three metrics symmetric;
    ``|len(a) - len(b)| <= levenshtein(a, b) <= max(len(a), len(b))``;
    ``0 <= jaro <= jaro_winkler <= 1`` (the boost never subtracts); and
    the identity characterizations: distance zero iff equal, Jaro one
    iff equal (the empty pair included by convention)."""
    assert levenshtein(a, b) == levenshtein(b, a)
    assert jaro(a, b) == pytest.approx(jaro(b, a), rel=1e-12, abs=1e-12)
    assert jaro_winkler(a, b) == pytest.approx(jaro_winkler(b, a), rel=1e-12, abs=1e-12)
    assert abs(len(a) - len(b)) <= levenshtein(a, b) <= max(len(a), len(b))
    jaro_score = jaro(a, b)
    assert 0.0 <= jaro_score <= jaro_winkler(a, b) <= 1.0
    assert (levenshtein(a, b) == 0) == (a == b)
    assert (jaro_score == 1.0) == (a == b)


@given(st.text(max_size=16), st.text(max_size=16), st.text(max_size=16))
@settings(max_examples=300)
def test_levenshtein_satisfies_the_triangle_inequality(a: str, b: str, c: str) -> None:
    """Levenshtein distance is a genuine metric (unlike jaro/jaro_winkler/
    similarity_ratio, none of which need this), the one algebraic property
    specific to it that symmetry/bounds/identity don't cover. A broken DP
    recurrence could satisfy all of those and still violate this."""
    assert levenshtein(a, c) <= levenshtein(a, b) + levenshtein(b, c)


# --- The deadline contract ------------------------------------------------------------


class TestDeadline:
    def test_a_genuinely_hard_pair_exceeding_the_deadline_raises_timeout_error(self) -> None:
        """``deadline_ms`` bounds the whole DP pass: the char-shuffled
        pair's table is ~1.4e10 cells (minutes of work, the DoS shape
        the parameter exists for), and every one of the three metrics
        aborts at the 50 ms budget with ``TimeoutError`` naming the
        elapsed time and the deadline, the incomplete result discarded,
        the call bounded well under a second."""
        a, b = diff_pair_char_shuffled(_DEADLINE_PAIR_BYTES)
        for metric in (levenshtein, jaro, jaro_winkler):
            started = time.perf_counter()
            with pytest.raises(TimeoutError, match="deadline") as excinfo:
                metric(a, b, deadline_ms=_DEADLINE_MS)
            wall = time.perf_counter() - started
            assert type(excinfo.value) is TimeoutError
            message = str(excinfo.value)
            assert re.search(r"elapsed \d+(\.\d+)?\s*ms", message), message
            assert wall < 1.0, f"deadline-bounded {metric.__name__} took {wall:.2f}s"

    def test_deadline_none_is_the_default_and_unchanged(self) -> None:
        """``deadline_ms=None`` is exactly the unbounded behavior; the
        parameter is purely additive."""
        assert levenshtein("kitten", "sitting") == levenshtein(
            "kitten", "sitting", deadline_ms=None
        )
        assert jaro("MARTHA", "MARHTA") == jaro("MARTHA", "MARHTA", deadline_ms=None)
        assert jaro_winkler("MARTHA", "MARHTA") == jaro_winkler(
            "MARTHA", "MARHTA", deadline_ms=None
        )

    def test_an_enormous_but_finite_deadline_saturates_to_unbounded(self) -> None:
        """A huge-but-finite budget is LEGAL (not a ValueError class with
        infinity) and behaves as no deadline at all: identical results to
        the default call for every metric, the saturation pin the diff
        deadline contract set, applied to the DP family."""
        for metric, pair in (
            (levenshtein, ("kitten", "sitting")),
            (jaro, ("MARTHA", "MARHTA")),
            (jaro_winkler, ("MARTHA", "MARHTA")),
        ):
            a, b = pair
            assert metric(a, b, deadline_ms=1e300) == metric(a, b)

    @pytest.mark.parametrize("bad", [0.0, -50.0, float("nan"), float("inf")])
    @pytest.mark.parametrize(
        "metric",
        [levenshtein, jaro, jaro_winkler],
        ids=["levenshtein", "jaro", "jaro_winkler"],
    )
    def test_nonpositive_or_nonfinite_deadlines_raise_value_error(
        self, metric: object, bad: float
    ) -> None:
        """A budget must be positive and finite: zero, negative, NaN and
        infinity are caller bugs; refused with the exact message before
        any work runs (the closed-set-of-strings convention of the
        suite's keyword parameters)."""
        with pytest.raises(
            ValueError, match="^deadline_ms must be a positive finite number of milliseconds$"
        ):
            metric("abc", "abd", deadline_ms=bad)  # type: ignore[operator]

    def test_a_non_numeric_deadline_raises_type_error(self) -> None:
        """The parameter is ``float | None``: anything non-numeric is
        refused with ``TypeError`` by the argument boundary."""
        with pytest.raises(TypeError):
            levenshtein("abc", "abd", deadline_ms="50")  # type: ignore[arg-type]


# --- The argument-boundary contract ---------------------------------------------------


class TestArgumentContract:
    @pytest.mark.parametrize(
        "not_str",
        [b"abc", bytearray(b"abc"), memoryview(b"abc"), 123, None],
        ids=["bytes", "bytearray", "memoryview", "int", "none"],
    )
    @pytest.mark.parametrize("which", ["a", "b"], ids=["first-arg", "second-arg"])
    @pytest.mark.parametrize(
        "metric",
        [levenshtein, jaro, jaro_winkler],
        ids=["levenshtein", "jaro", "jaro_winkler"],
    )
    def test_non_str_arguments_raise_type_error(
        self, metric: object, which: str, not_str: object
    ) -> None:
        """The str-in argument contract, both operands of all three
        metrics (the str-exactly rule every tors str argument follows):
        exactly ``str``; the extraction rejects everything else with
        ``TypeError`` before any Rust code runs."""
        good = "abc"
        args: tuple[object, object] = (not_str, good) if which == "a" else (good, not_str)
        with pytest.raises(TypeError):
            metric(*args)  # type: ignore[operator]

    @pytest.mark.parametrize(
        "metric",
        [levenshtein, jaro, jaro_winkler],
        ids=["levenshtein", "jaro", "jaro_winkler"],
    )
    def test_lone_surrogates_are_refused_at_the_argument_boundary(
        self, metric: object
    ) -> None:
        """Lone surrogates (a ``str`` CPython can hold but UTF-8 cannot
        encode) are refused with ``UnicodeEncodeError`` before any Rust
        code runs, the standard str-in boundary every tors function
        pays; both operands of every metric."""
        with pytest.raises(UnicodeEncodeError):
            metric("abc\ud800", "abc")  # type: ignore[operator]
        with pytest.raises(UnicodeEncodeError):
            metric("abc", "abc\ud800")  # type: ignore[operator]
