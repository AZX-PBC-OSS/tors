"""Contract gate for ``tors.diff_opcodes_lines``: the LINE-level spelling of
``tors.diff_opcodes``: same engine (``similar``'s Myers), same opcode
shape and validity contract, same ``deadline_ms`` machinery, but the operands
are tokenized as LINES and the returned indices address lines, not characters.

This function had NO dedicated Python-level test file before this one (a
coverage gap: only incidental coverage via the document corpus gate and the
GIL-timing suite); this fills that gap directly, rather than trusting the
Rust-internal battery alone to gate the PUBLIC contract.

**The one contract detail worth stating loudly**: the tokenization is
``'\\n'``-only. ``a_lines``/``b_lines`` for reconstructing ``a[i1:i2]``-style
slices from the returned indices must be built as ``a.split("\\n")`` with
each piece's terminator reattached, NOT ``a.splitlines(keepends=True)``,
which additionally breaks on lone ``\\r``, ``\\v``, ``\\f``, and the Unicode
line/paragraph separators. This is documented in ``docs/api.md``, the
``lib.rs`` docstring, and the README; it is pinned here so the divergence is
a visible, tested fact rather than a claim nobody checks.
"""

from __future__ import annotations

import re
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors
from reference import assert_opcodes_are_valid, diff_pair_near_identical


def _tors_lines(text: str) -> list[str]:
    """The ``a_lines`` a caller must build to reconstruct ``diff_opcodes_lines``
    indices: ``'\\n'``-terminated pieces, NOT ``str.splitlines()``.

    ``re.split(r"(?<=\\n)", text)`` is *almost* the recipe, but leaves a
    trailing empty string whenever ``text`` ends with ``'\\n'`` (the common
    case, most files end with a trailing newline); caught by this test
    file's own reconstruction properties below, which is exactly why this
    helper exists rather than inlining the naive regex everywhere."""
    if text == "":
        return []
    parts = re.split(r"(?<=\n)", text)
    if parts[-1] == "":
        parts.pop()
    return parts


class TestReconstruction:
    def test_identical_inputs_return_one_equal_opcode_over_line_count(self) -> None:
        a = "l1\nl2\nl3\n"
        assert tors.diff_opcodes_lines(a, a) == [("equal", 0, 3, 0, 3)]

    def test_the_empty_pair_returns_the_empty_list(self) -> None:
        assert tors.diff_opcodes_lines("", "") == []

    def test_empty_versus_nonempty_covers_the_whole_operand_in_one_op(self) -> None:
        assert tors.diff_opcodes_lines("", "a\nb\n") == [("insert", 0, 0, 0, 2)]
        assert tors.diff_opcodes_lines("a\nb\n", "") == [("delete", 0, 2, 0, 0)]

    def test_readme_worked_example(self) -> None:
        assert tors.diff_opcodes_lines("l1\nl2\nl3\n", "l1\nX\nl3\nl4\n") == [
            ("equal", 0, 1, 0, 1),
            ("replace", 1, 2, 1, 2),
            ("equal", 2, 3, 2, 3),
            ("insert", 3, 3, 3, 4),
        ]

    @given(st.text(alphabet="ab\n", max_size=24), st.text(alphabet="ab\n", max_size=24))
    @settings(max_examples=300)
    def test_opcodes_reconstruct_both_sides_over_arbitrary_line_pairs(self, a: str, b: str) -> None:
        """The structural contract (contiguity, coverage, alternation,
        per-tag nonemptiness, equal-content equals, full reconstruction)
        holds when ``a_lines``/``b_lines`` are built the WAY THE DOCS SAY
        TO (``'\\n'``-split, not ``str.splitlines()``) over an alphabet
        that generates every line-boundary shape (empty lines, no trailing
        terminator, lines that are themselves just ``'\\n'``)."""
        ops = tors.diff_opcodes_lines(a, b)
        assert_opcodes_are_valid(_tors_lines(a), _tors_lines(b), ops)

    def test_reconstruction_via_python_split_matches_the_documented_recipe(self) -> None:
        a, b = diff_pair_near_identical(64 * 1024)
        a_lines = _tors_lines(a)
        b_lines = _tors_lines(b)
        ops = tors.diff_opcodes_lines(a, b)
        assert_opcodes_are_valid(a_lines, b_lines, ops)
        # Every op's a-side slice, concatenated in order, reconstructs a.
        rebuilt = "".join(
            "".join(a_lines[i1:i2]) for tag, i1, i2, _j1, _j2 in ops if tag != "insert"
        )
        assert rebuilt == a


class TestSplitlinesDivergenceIsReal:
    """The documented gap between tors's ``'\\n'``-only tokenization and
    Python's ``str.splitlines(keepends=True)``, demonstrated end-to-end: a
    caller who (incorrectly) reconstructs with ``str.splitlines()`` instead
    of the documented recipe gets WRONG content back for non-LF line
    terminators. This is not a bug; it is the pinned, intentional contract,
    but it must stay a *visible, tested* fact, not a silent trap."""

    def test_lone_cr_terminated_text_is_one_line_to_tors_but_several_to_splitlines(
        self,
    ) -> None:
        a = "line1\rline2\rline3"
        b = "line1\rlineX\rline3"
        # tors sees ONE line each (no '\n' anywhere) -> a single replace op
        # spanning the whole (one-element) line vector.
        assert tors.diff_opcodes_lines(a, b) == [("replace", 0, 1, 0, 1)]
        # Python's splitlines() sees THREE lines each.
        assert a.splitlines(keepends=True) == ["line1\r", "line2\r", "line3"]
        assert b.splitlines(keepends=True) == ["line1\r", "lineX\r", "line3"]
        # A caller who reconstructs with splitlines() using tors's indices
        # gets the WRONG slice: index (0, 1) over the 3-element splitlines()
        # list is just "line1\r", not the whole text tors's op actually
        # describes. This is exactly why the docs say not to do that.
        wrong_reconstruction = "".join(a.splitlines(keepends=True)[0:1])
        assert wrong_reconstruction != a

    def test_the_documented_recipe_reconstructs_correctly_for_the_same_input(self) -> None:
        a = "line1\rline2\rline3"
        b = "line1\rlineX\rline3"
        a_lines = _tors_lines(a)
        b_lines = _tors_lines(b)
        assert a_lines == [a]  # no '\n' at all -> one "line"
        ops = tors.diff_opcodes_lines(a, b)
        assert_opcodes_are_valid(a_lines, b_lines, ops)

    def test_unicode_line_and_paragraph_separators_do_not_split(self) -> None:
        a = "a b c"
        assert tors.diff_opcodes_lines(a, a) == [("equal", 0, 1, 0, 1)]
        assert a.splitlines(keepends=True) == ["a ", "b ", "c"]

    def test_crlf_splits_correctly_because_of_the_lf_not_special_casing(self) -> None:
        # CRLF *does* split at the expected place, but only incidentally --
        # the '\n' half is a real terminator, and '\r' rides along as
        # ordinary trailing content on the line it terminates.
        a = "a\r\nb\r\nc"
        assert tors.diff_opcodes_lines(a, a) == [("equal", 0, 3, 0, 3)]
        assert _tors_lines(a) == ["a\r\n", "b\r\n", "c"]


class TestDeterminism:
    def test_the_same_pair_diffed_twice_returns_byte_identical_opcodes(self) -> None:
        a, b = diff_pair_near_identical(128 * 1024)
        first = tors.diff_opcodes_lines(a, b)
        second = tors.diff_opcodes_lines(a, b)
        assert first == second

    def test_determinism_holds_under_the_deadline_path_too(self) -> None:
        a, b = diff_pair_near_identical(128 * 1024)
        first = tors.diff_opcodes_lines(a, b, deadline_ms=60_000.0)
        second = tors.diff_opcodes_lines(a, b, deadline_ms=60_000.0)
        assert first == second


# --- deadline_ms, at line grain -------------------------------------------------------
#
# Same machinery as tors.diff_opcodes's deadline (see test_diff_opcodes.py's
# TestDeadline docstring for the full mechanism writeup): one clock, started
# before any work, checked against similar's own deadline-aware search plus
# an expiry verdict afterward -- never a GIL reacquire in a polling loop,
# since the whole call runs under one py.detach.
#
# The hard shape here is NOT diff_pair_char_shuffled's prose (that fixture
# shuffles CHARACTERS of prose text -- a hard case for the CHAR-level
# engine, but character-shuffling also redistributes '\n' positions in a way
# that tends to leave few, largely-distinct lines: easy at LINE grain, not
# hard). The genuinely hard line-level shape -- mirrored from
# src/diff_impl.rs's own line-level deadline test -- is many repeated SHORT
# lines over a tiny alphabet, permuted at the CHARACTER level so the '\n'
# separators move too: almost no anchorable unique line records, the same
# superlinear wall the char-level ladder measures, reached at a much smaller
# corpus.


def _line_shuffled_pair(repeats: int) -> tuple[str, str]:
    """``(a, b)``: ``"a\\nb\\nc\\n"`` repeated ``repeats`` times, and the same
    text with every CHARACTER (including the ``'\\n'`` separators) permuted
    by a deterministic LCG Fisher-Yates -- both sides tokenize to
    ``3 * repeats`` lines over a 3-character alphabet with almost no
    anchorable unique records, superlinear for the bounded Myers search."""
    base = "a\nb\nc\n" * repeats
    chars = list(base)
    state = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    for i in range(len(chars) - 1, 0, -1):
        state = (state ^ (state << 13)) & mask
        state = (state ^ (state >> 7)) & mask
        state = (state ^ (state << 17)) & mask
        j = state % (i + 1)
        chars[i], chars[j] = chars[j], chars[i]
    return base, "".join(chars)


@pytest.mark.parametrize(
    "not_str",
    [b"abc", bytearray(b"abc"), 123, None],
    ids=["bytes", "bytearray", "int", "none"],
)
@pytest.mark.parametrize("which", ["a", "b"], ids=["first-arg", "second-arg"])
def test_non_str_arguments_raise_type_error(not_str: object, which: str) -> None:
    """Same str-in argument contract as ``diff_opcodes``
    (tests/test_diff_opcodes.py::test_non_str_arguments_raise_type_error);
    both spellings share the same pyo3 ``&str`` extraction, but this
    function had no dedicated Python-level test file until now, and this
    boundary was never pinned for the line-level entrypoint specifically."""
    good = "a\nb\n"
    args: tuple[object, object] = (not_str, good) if which == "a" else (good, not_str)
    with pytest.raises(TypeError):
        tors.diff_opcodes_lines(*args)  # type: ignore[arg-type]


def test_lone_surrogates_are_refused_at_the_argument_boundary() -> None:
    """Same lone-surrogate ``UnicodeEncodeError`` boundary as ``diff_opcodes``
    (tests/test_diff_opcodes.py::test_lone_surrogates_are_refused_at_the_argument_boundary),
    pinned for the line-level entrypoint, which shares the identical pyo3
    ``&str`` extraction for both operands."""
    with pytest.raises(UnicodeEncodeError):
        tors.diff_opcodes_lines("abc\ud800\n", "abc\n")
    with pytest.raises(UnicodeEncodeError):
        tors.diff_opcodes_lines("abc\n", "abc\ud800\n")


_DEADLINE_MS = 50.0


class TestDeadline:
    def test_a_slow_pair_exceeding_the_deadline_raises_timeout_error(self) -> None:
        a, b = _line_shuffled_pair(10_000)
        started = time.perf_counter()
        with pytest.raises(TimeoutError, match="deadline") as excinfo:
            tors.diff_opcodes_lines(a, b, deadline_ms=_DEADLINE_MS)
        wall = time.perf_counter() - started
        message = str(excinfo.value)
        assert type(excinfo.value) is TimeoutError
        assert re.search(rf"\b{_DEADLINE_MS:g}(\.0)?\s*ms\b", message), message
        assert re.search(r"elapsed \d+(\.\d+)?\s*ms", message), message
        assert wall < 1.0, f"deadline-bounded call took {wall:.2f}s"

    def test_deadline_none_is_the_default_and_unchanged(self) -> None:
        a, b = diff_pair_near_identical(256 * 1024)
        assert tors.diff_opcodes_lines(a, b) == tors.diff_opcodes_lines(a, b, deadline_ms=None)

    def test_a_generous_deadline_yields_the_identical_opcodes(self) -> None:
        a, b = diff_pair_near_identical(256 * 1024)
        assert tors.diff_opcodes_lines(a, b, deadline_ms=60_000.0) == tors.diff_opcodes_lines(a, b)

    def test_identical_inputs_under_a_deadline_still_short_circuit(self) -> None:
        a = diff_pair_near_identical(64 * 1024)[0]
        n_lines = len(_tors_lines(a))
        assert tors.diff_opcodes_lines(a, a, deadline_ms=5_000.0) == [
            ("equal", 0, n_lines, 0, n_lines)
        ]
        assert tors.diff_opcodes_lines("", "", deadline_ms=5_000.0) == []

    @pytest.mark.parametrize("bad", [0.0, -50.0], ids=["zero", "negative"])
    def test_nonpositive_deadlines_raise_value_error(self, bad: float) -> None:
        with pytest.raises(ValueError, match="deadline_ms"):
            tors.diff_opcodes_lines("a\n", "b\n", deadline_ms=bad)

    def test_an_out_of_range_finite_deadline_does_not_panic(self) -> None:
        """The same saturation fix ``diff_opcodes`` carries
        (an enormous-but-finite ``deadline_ms`` must not overflow
        ``Duration::from_secs_f64`` and panic), pinned here too, since both
        spellings share ``budget_from_ms`` but nothing previously exercised
        it through the line-level entrypoint specifically."""
        a, b = diff_pair_near_identical(64 * 1024)
        assert tors.diff_opcodes_lines(a, b, deadline_ms=1e300) == tors.diff_opcodes_lines(a, b)
