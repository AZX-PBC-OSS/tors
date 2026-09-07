"""Contract gate for ``tors.diff_opcodes``: difflib-shaped opcode diffs at
native speed, GIL-released.

``tors.diff_opcodes(a, b)`` returns ``difflib.SequenceMatcher(None, a, b)
.get_opcodes()``'s SHAPE: ``(tag, i1, i2, j1, j2)`` tuples with ``tag`` in
``{"equal", "replace", "delete", "insert"}``, ranges monotone, contiguous and
covering both sides, adjacent delete+insert merged into ``replace`` exactly as
difflib presents it, over a Myers diff (the ``similar`` crate) that runs as
one native pass with the GIL released. CHARACTER-level, like difflib on
``str`` operands (a ``str`` IS a character sequence to ``SequenceMatcher``):
that is what makes difflib the parity oracle, so character-level is the
choice. Callers wanting line-level diffs split the operands themselves and diff
the pieces; the opcode shape composes.

The parity contract, in three parts (no false parity: the algorithms are
different and both are right):

1. **Structural validity, over arbitrary pairs** (hypothesis): applying the
   opcodes reconstructs BOTH ``a`` and ``b`` exactly; ranges are monotone,
   contiguous, covering; tags alternate equal/non-equal (difflib's opcode
   streams never put two non-equal ops adjacent (a delete next to an insert
   is a replace); equal ops carry equal-length, equal-content ranges.
2. **Exact agreement with difflib on the classes whose canonical opcode
   list is FORCED**: pure insert, pure delete, all-equal, single-run
   replace, and empty operands in both directions. "Forced" is verified,
   not assumed: the hypothesis properties below assert exact agreement
   only where difflib's OWN answer carries the canonical single-op shape
   (exactly one non-equal op of the expected tag, plus an optional leading
   and trailing equal): where difflib itself presents the change that
   way, the minimal edit script is forced and any two correct algorithms
   must emit the same list. Where difflib does NOT (the repeated-flank
   class in part 3), the pair is excluded from the exact-parity pin and
   tors's own canonical shape is asserted instead, with no assume.
3. **Documented boundary cases where they legitimately differ**, pinned so
   the differences are visible and intentional, never silent. The shared
   mechanism: difflib's longest-match recursion ANCHORS a match and splits
   the change around it, where Myers + run-maximization emits contiguous
   runs slid to one side. Three pinned shapes of it: on ``"a"`` vs
   ``"baa"`` the anchored middle ``'a'`` splits the insertion (``insert
   'b', equal 'a', insert 'a'`` vs one contiguous ``insert 'ba'``); on
   ``"ppp"`` vs ``"pwpp"`` the anchored ``"pp"`` slides the equal-run
   boundary across a repeated-character insertion point (``insert 'pw',
   equal 'pp', delete 'p'`` vs ``equal 'p', insert 'w', equal 'pp'``); and
   on ``"qpqpq"`` vs ``"qpwqpq"`` the same anchored-split mechanism shows
   up at a REPEATED-FLANK CONTEXT: the suffix repeats the prefix's head,
   so the change is not confined by the generators' one-character flank
   guard, where difflib anchors a ROTATED longer equal block and emits a
   NON-MINIMAL insert+delete split while tors emits the minimal
   contiguous insert. Every side reconstructs both operands; the tors
   sides are minimal edit scripts. This class (a change surrounded by
   repeated patterns, where the equal-block alignment can legitimately
   slide or rotate) is the ordinary shape of the difference, not an edge
   case, and it is why the exact-parity properties gate on difflib's own
   canonical presentation rather than trusting the generator's
   construction alone.

Identical inputs return ``[("equal", 0, len(a), 0, len(b))]``; the
empty-vs-empty pair returns ``[]``, difflib's own answer (its sentinel-only
matching blocks produce no opcodes), pinned below against the oracle rather
than assumed.

Measured wall on the dev box this suite runs on (WSL2, 28 logical cores,
CPython 3.12; loads disclosed per number; corpora from
``reference.diff_pair_near_identical``, the six-edit mutated-line shape;
min-of-3 after one warm-up unless noted):

- 256 KiB: tors 3.9 ms vs difflib 3,059 ms (ambient load 6.0), ratio
  0.0013; difflib is quadratic at character level and this is the size where
  it starts taking seconds.
- 1 MiB: tors 2.0 ms (same load window); difflib measured ONCE at 59,001 ms
  (ambient load 5.6): a minute per call, which is why the 1 MiB cell asserts
  tors's absolute band and records difflib's number instead of racing it
  in-suite: running difflib per suite run at 1 MiB would cost a minute
  locally and several minutes on a 2-vCPU CI runner. The asserted race lives
  at 256 KiB, where difflib's seconds are suite-affordable.

The GIL-release claims (whole diff under ``py.detach``; the O(ops)
tuple-marshalling residue measured on the 12 MiB shuffled pair, the
``word_bounds`` list-shape precedent) are pinned in ``tests/test_gil_release.py``;
the criterion ladder for the Rust core alone is ``benches/diff.rs``.
"""

from __future__ import annotations

import difflib
import re
import time
from collections.abc import Callable

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import tors
from reference import (
    assert_opcodes_are_valid,
    diff_pair_char_shuffled,
    diff_pair_near_identical,
)

_MIB = 1024 * 1024

# The wall race's tolerance margin: tors's measured ratio at 256 KiB is ~0.001,
# so 0.25 leaves ~250x headroom; the assertion pins the quadratic-vs-native
# relationship, not a close race. difflib is sampled ONCE (at ~2.9 s per call,
# extra samples are pure suite time; noise only ever adds time, so a single
# sample is conservative for the ratio's denominator).
_DIFF_WALL_MARGIN = 0.25

# The 1 MiB absolute band: measured 2.0 ms, ceiling 40 ms (~20x margin; the
# same derivation shape as the grapheme_count band: a linear-ish native pass
# with generous room for a loaded 2-vCPU CI runner, blown straight through by
# any accidental quadratic or per-opcode allocation regression).
_DIFF_1MIB_CEILING_MS = 40.0


def _difflib_opcodes(a: str, b: str) -> list[tuple[str, int, int, int, int]]:
    """The oracle: exactly the stdlib expression ``tors.diff_opcodes`` replaces."""
    return difflib.SequenceMatcher(None, a, b).get_opcodes()


def _sole_non_equal_op(
    ops: list[tuple[str, int, int, int, int]], tag: str
) -> tuple[str, int, int, int, int] | None:
    """The canonical single-op shape predicate: ``ops`` carries EXACTLY ONE
    non-equal op, its tag is ``tag``, and every other op is an equal (an
    optional leading and/or trailing equal), the canonical presentation of
    one contiguous change. Returns the op, or ``None`` for any other shape,
    including difflib's anchored non-minimal insert+delete splits, which the
    exact-parity gates below must exclude."""
    non_equal = [op for op in ops if op[0] != "equal"]
    if len(non_equal) != 1 or non_equal[0][0] != tag:
        return None
    return non_equal[0]


# --- Part 2: exact agreement on the unambiguous classes ----------------------------
#
# Generator alphabets are DISJOINT by class (context over one alphabet, the
# changed content over others), so the changed block's CONTENT is forced into
# the non-equal ops: a context character can never pair with a changed
# character inside an equal op. That alone does NOT force difflib's
# PRESENTATION: when the context around the change repeats itself (the
# suffix repeating the prefix's head, as in the pinned "qpqpq" vs "qpwqpq"
# boundary pair), difflib's find_longest_match can anchor a ROTATED longer
# equal block and emit a NON-MINIMAL insert+delete split, while tors (Myers
# + run-maximization) emits the minimal contiguous change. The
# one-character flank guard (prefix's last != suffix's first) excludes the
# adjacent-slide shape but not this repeated-flank one. So the
# definition of the unambiguous class is difflib's own answer: the
# exact-parity properties below gate on difflib emitting the canonical
# single-op shape (exactly one non-equal op of the expected tag; where
# difflib itself is forced there, any two correct algorithms must agree),
# and separate no-assume properties pin that tors ALWAYS emits that
# canonical shape with exactly the generated changed block.
#
# Each generator returns ``(a, b, changed_block)``; the changed block is
# what the tors-side canonical properties assert the single non-equal op
# carries.
_CONTEXT_ALPHABET = "pqrs"
_REPLACE_ALPHABET = "xyzw"
_REPLACEMENT_ALPHABET = "cdef"


@st.composite
def _pure_insert_pair(draw: st.DrawFn) -> tuple[str, str, str]:
    prefix = draw(st.text(alphabet=_CONTEXT_ALPHABET, min_size=0, max_size=8))
    suffix = draw(st.text(alphabet=_CONTEXT_ALPHABET, min_size=0, max_size=8))
    inserted = draw(st.text(alphabet=_REPLACE_ALPHABET, min_size=1, max_size=8))
    assume(not prefix or not suffix or prefix[-1] != suffix[0])
    return prefix + suffix, prefix + inserted + suffix, inserted


@st.composite
def _pure_delete_pair(draw: st.DrawFn) -> tuple[str, str, str]:
    prefix = draw(st.text(alphabet=_CONTEXT_ALPHABET, min_size=0, max_size=8))
    suffix = draw(st.text(alphabet=_CONTEXT_ALPHABET, min_size=0, max_size=8))
    deleted = draw(st.text(alphabet=_REPLACE_ALPHABET, min_size=1, max_size=8))
    assume(not prefix or not suffix or prefix[-1] != suffix[0])
    return prefix + deleted + suffix, prefix + suffix, deleted


@st.composite
def _single_run_replace_pair(draw: st.DrawFn) -> tuple[str, str, str, str]:
    prefix = draw(st.text(alphabet=_CONTEXT_ALPHABET, min_size=0, max_size=8))
    suffix = draw(st.text(alphabet=_CONTEXT_ALPHABET, min_size=0, max_size=8))
    old_run = draw(st.text(alphabet=_REPLACE_ALPHABET, min_size=1, max_size=8))
    new_run = draw(st.text(alphabet=_REPLACEMENT_ALPHABET, min_size=1, max_size=8))
    return prefix + old_run + suffix, prefix + new_run + suffix, old_run, new_run


# The pinned boundary pairs: where difflib and tors legitimately differ (see
# the contract's part 3 and the boundary test below). Every other battery
# member is unambiguous and must match the oracle exactly.
_BOUNDARY_PAIRS = {("a", "baa"), ("ppp", "pwpp"), ("pXpp", "ppp"), ("qpqpq", "qpwqpq")}


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("", ""),
        ("", "abcdefg"),
        ("abcdefg", ""),
        ("a", "baa"),
        ("ppp", "pwpp"),
        ("pXpp", "ppp"),
        ("qpqpq", "qpwqpq"),
        ("ab", "ba"),
        ("qabxcd", "abYcd"),
        ("abcdef", "abXcdY"),
        ("abcd", "acd"),
        ("abcd", "abced"),
        ("hello world", "hello there world"),
        ("the quick brown fox", "the quick brown dog"),
        ("café", "cafe"),
    ],
    ids=[
        "empty-empty",
        "empty-nonempty",
        "nonempty-empty",
        "boundary-insert-split",
        "boundary-equal-run-slide",
        "boundary-delete-split",
        "boundary-repeated-flank-rotate",
        "swap",
        "delete-replace-mix",
        "insert-replace-mix",
        "single-delete",
        "single-insert",
        "word-insert",
        "word-replace",
        "non-ascii-delete",
    ],
)
def test_structural_validity_and_oracle_agreement_on_the_fixed_battery(a: str, b: str) -> None:
    """The fixed battery: every pair is checked for full structural validity,
    and the UNAMBIGUOUS members (everything except the pinned boundary pairs)
    must match difflib's opcode list exactly; the fixed-case anchor of the
    hypothesis properties below. The boundary pairs are the documented
    legitimate differences, pinned by their own test."""
    ops = tors.diff_opcodes(a, b)
    assert_opcodes_are_valid(a, b, ops)
    if (a, b) not in _BOUNDARY_PAIRS:
        assert ops == _difflib_opcodes(a, b), (
            f"tors and difflib disagree on the unambiguous pair {a!r} vs {b!r}"
        )


def test_the_documented_boundary_cases_differ_from_difflib_visibly() -> None:
    """Part 3 of the parity contract: the pinned, intentional differences:
    one block per mechanism shape, both sides' exact lists asserted so the
    differences can never become silent drift, and each side's own validity
    asserted independently.

    ``"a"`` vs ``"baa"``: the anchored-match split: difflib's
    find_longest_match recursion anchors the MIDDLE ``'a'`` (the first
    maximal match it sees) and emits the insertion split around that anchor;
    Myers plus similar's run-maximization emits the same edit volume as ONE
    contiguous insertion slid to the left.

    ``"ppp"`` vs ``"pwpp"``: the equal-run slide (and its delete-class twin
    ``"pXpp"`` vs ``"ppp"``): the insertion sits adjacent to a
    repeated-character boundary, so the equal-run alignment can legitimately
    slide; difflib anchors the ``"pp"`` at ``a[0:2]``/``b[2:4]`` and splits
    the change around it (an insert and a delete), Myers slides the equal run
    right and isolates the single insertion. This is the shape the
    unambiguous-class generators exclude by pinning their flanks; pinned
    here so the exclusion is visible, not silent.

    ``"qpqpq"`` vs ``"qpwqpq"``: the REPEATED-FLANK ROTATION, the same
    anchored-split mechanism at a new class of inputs: a pure insert
    (``"w"`` between prefix ``"qp"`` and suffix ``"qpq"``) whose suffix
    repeats the prefix's head, so the one-character flank guard does not
    exclude it. difflib's find_longest_match anchors the ROTATED longer
    equal block ``a[0:3] == b[3:6]`` (``"qpq"``) and emits a NON-MINIMAL
    insert+delete split around it; tors emits the minimal contiguous insert
    between the unrotated equals. Exhaustive deterministic sweeps of the
    generator class found difflib non-canonical on ~0.2-2.4% of drawn pairs
    (1,008 of 606,900 insert pairs over 4-char contexts up to length 3)
    with tors canonical and minimal on every one; this pair is the smallest
    hand-readable instance, and it is the counterexample that proved the
    pre-gate exact-parity claim wrong."""
    assert tors.diff_opcodes("a", "baa") == [("insert", 0, 0, 0, 2), ("equal", 0, 1, 2, 3)]
    assert _difflib_opcodes("a", "baa") == [
        ("insert", 0, 0, 0, 1),
        ("equal", 0, 1, 1, 2),
        ("insert", 1, 1, 2, 3),
    ]
    assert tors.diff_opcodes("ppp", "pwpp") == [
        ("equal", 0, 1, 0, 1),
        ("insert", 1, 1, 1, 2),
        ("equal", 1, 3, 2, 4),
    ]
    assert _difflib_opcodes("ppp", "pwpp") == [
        ("insert", 0, 0, 0, 2),
        ("equal", 0, 2, 2, 4),
        ("delete", 2, 3, 4, 4),
    ]
    assert tors.diff_opcodes("pXpp", "ppp") == [
        ("equal", 0, 1, 0, 1),
        ("delete", 1, 2, 1, 1),
        ("equal", 2, 4, 1, 3),
    ]
    assert _difflib_opcodes("pXpp", "ppp") == [
        ("delete", 0, 2, 0, 0),
        ("equal", 2, 4, 0, 2),
        ("insert", 4, 4, 2, 3),
    ]
    assert tors.diff_opcodes("qpqpq", "qpwqpq") == [
        ("equal", 0, 2, 0, 2),
        ("insert", 2, 2, 2, 3),
        ("equal", 2, 5, 3, 6),
    ]
    assert _difflib_opcodes("qpqpq", "qpwqpq") == [
        ("insert", 0, 0, 0, 3),
        ("equal", 0, 3, 3, 6),
        ("delete", 3, 5, 6, 6),
    ]
    for a, b in _BOUNDARY_PAIRS:
        assert_opcodes_are_valid(a, b, tors.diff_opcodes(a, b))
        assert_opcodes_are_valid(a, b, _difflib_opcodes(a, b))


@pytest.mark.parametrize(
    "text",
    ["abc", "The quarterly oil sample interval.", "café 東京 🦀", "a" * 500, "  \n\t\r\n "],
    ids=["ascii", "prose", "non-ascii", "repeated", "whitespace"],
)
def test_identical_inputs_return_one_equal_opcode(text: str) -> None:
    """The identical-input contract: a single equal op covering both sides,
    difflib's own answer for ``a == b`` (and the cheap path: one native
    equality scan, no diff search at all)."""
    assert tors.diff_opcodes(text, text) == [("equal", 0, len(text), 0, len(text))]
    assert tors.diff_opcodes(text, text) == _difflib_opcodes(text, text)


def test_the_empty_pair_returns_difflibs_empty_list() -> None:
    """``("", "")`` is the one degenerate spelling where the single-equal
    contract does NOT apply: difflib's matching blocks for two empty operands
    are the zero-size sentinel alone, so ``get_opcodes()`` emits nothing.
    Pinned against the live oracle; the empty list IS the compatible answer."""
    assert tors.diff_opcodes("", "") == []
    assert tors.diff_opcodes("", "") == _difflib_opcodes("", "")


@pytest.mark.parametrize(
    "not_str",
    [b"abc", bytearray(b"abc"), 123, None],
    ids=["bytes", "bytearray", "int", "none"],
)
@pytest.mark.parametrize("which", ["a", "b"], ids=["first-arg", "second-arg"])
def test_non_str_arguments_raise_type_error(not_str: object, which: str) -> None:
    """The str-in argument contract, both operands (the str-exactly rule every
    tors str argument follows): exactly ``str``; pyo3's ``&str`` extraction rejects
    everything else with ``TypeError`` before any Rust code runs."""
    good = "abc"
    args: tuple[object, object] = (not_str, good) if which == "a" else (good, not_str)
    with pytest.raises(TypeError):
        tors.diff_opcodes(*args)  # type: ignore[arg-type]


def test_lone_surrogates_are_refused_at_the_argument_boundary() -> None:
    """Lone surrogates (a ``str`` CPython can hold but UTF-8 cannot encode,
    e.g. from a ``surrogatepass`` decoder) are refused at the argument
    boundary with ``UnicodeEncodeError`` before any Rust code runs, the same
    pyo3 ``&str`` boundary as ``tors.finalize`` (pinned in
    tests/test_finalize.py) and the forms (tests/test_forms.py); both operands
    of this function pay it."""
    with pytest.raises(UnicodeEncodeError):
        tors.diff_opcodes("abc\ud800", "abc")
    with pytest.raises(UnicodeEncodeError):
        tors.diff_opcodes("abc", "abc\ud800")


@given(_pure_insert_pair())
@settings(max_examples=400)
def test_pure_inserts_match_difflib_exactly(case: tuple[str, str, str]) -> None:
    """Generated pure-insert pairs, gated to the unambiguous subset:
    exact agreement with difflib is asserted ONLY where difflib's own answer
    carries the canonical single-insert shape (exactly one insert op, plus
    optional leading/trailing equals); there the minimal edit script is
    forced and any two correct algorithms must emit the same list. The gate
    excludes the repeated-flank class: context that repeats itself around
    the insert (the suffix repeating the prefix's head, the pinned
    ``"qpqpq"`` vs ``"qpwqpq"`` boundary pair, the same anchored-split
    mechanism as ``"a"`` vs ``"baa"`` at a new class of inputs), where
    difflib's find_longest_match anchors a rotated longer equal block and
    emits a non-minimal insert+delete split. tors's answer on that excluded
    class is pinned by the no-assume canonical property below, not by this
    exact-list claim."""
    a, b, _inserted = case
    assume(_sole_non_equal_op(_difflib_opcodes(a, b), "insert") is not None)
    assert tors.diff_opcodes(a, b) == _difflib_opcodes(a, b)


@given(_pure_insert_pair())
@settings(max_examples=400)
def test_canonical_single_insert_shape_holds_for_tors_on_every_pure_insert(
    case: tuple[str, str, str],
) -> None:
    """tors's own canonical property, over EVERY generated pure-insert pair
    (no assume, the gate-free claim): exactly one insert op (plus optional
    equals), whose j-span content is exactly the inserted block: the
    minimal contiguous presentation. Verified before pinning over
    exhaustive deterministic sweeps of the generator class (606,900 pairs
    over 4-char contexts up to length 3, with and without the flank guard):
    tors never splits a pure insert, where difflib splits on the
    repeated-flank draws."""
    a, b, inserted = case
    ops = tors.diff_opcodes(a, b)
    assert_opcodes_are_valid(a, b, ops)
    op = _sole_non_equal_op(ops, "insert")
    assert op is not None, f"tors did not emit a single insert op: {ops!r}"
    _, i1, i2, j1, j2 = op
    assert i1 == i2, f"insert op carries a-side content: {op!r}"
    assert b[j1:j2] == inserted, f"insert span is not the inserted block: {op!r}"


@given(_pure_delete_pair())
@settings(max_examples=400)
def test_pure_deletes_match_difflib_exactly(case: tuple[str, str, str]) -> None:
    """Generated pure-delete pairs, gated to the unambiguous subset,
    the mirror of the insert class: exact agreement with difflib is
    asserted ONLY where difflib's own answer carries the canonical
    single-delete shape, and the gate excludes the repeated-flank class
    where difflib anchors a rotated equal block and splits the deletion
    non-minimally. tors's answer there is pinned by the no-assume canonical
    property below."""
    a, b, _deleted = case
    assume(_sole_non_equal_op(_difflib_opcodes(a, b), "delete") is not None)
    assert tors.diff_opcodes(a, b) == _difflib_opcodes(a, b)


@given(_pure_delete_pair())
@settings(max_examples=400)
def test_canonical_single_delete_shape_holds_for_tors_on_every_pure_delete(
    case: tuple[str, str, str],
) -> None:
    """tors's own canonical property, over EVERY generated pure-delete pair
    (no assume): exactly one delete op (plus optional equals), whose i-span
    content is exactly the deleted block. Verified before pinning over the
    same exhaustive deterministic sweeps as the insert property; tors
    never splits a pure delete."""
    a, b, deleted = case
    ops = tors.diff_opcodes(a, b)
    assert_opcodes_are_valid(a, b, ops)
    op = _sole_non_equal_op(ops, "delete")
    assert op is not None, f"tors did not emit a single delete op: {ops!r}"
    _, i1, i2, j1, j2 = op
    assert j1 == j2, f"delete op carries b-side content: {op!r}"
    assert a[i1:i2] == deleted, f"delete span is not the deleted block: {op!r}"


@given(_single_run_replace_pair())
@settings(max_examples=400)
def test_single_run_replacements_match_difflib_exactly(
    case: tuple[str, str, str, str],
) -> None:
    """Generated single-run-replace pairs, gated to the unambiguous
    subset: one contiguous run of ``a`` replaced by a same-position run of
    disjoint-alphabet content; exact agreement is asserted ONLY where
    difflib's own answer carries the canonical single-replace shape (the
    merged delete+insert presentation both implementations emit). The
    disjoint old/new alphabets keep difflib canonical on every swept pair
    of this class, but the gate is kept for the same reason:
    agreement is claimed exactly where difflib is forced, never by
    construction alone."""
    a, b, _old_run, _new_run = case
    assume(_sole_non_equal_op(_difflib_opcodes(a, b), "replace") is not None)
    assert tors.diff_opcodes(a, b) == _difflib_opcodes(a, b)


@given(_single_run_replace_pair())
@settings(max_examples=400)
def test_canonical_single_replace_shape_holds_for_tors_on_every_single_run_replace(
    case: tuple[str, str, str, str],
) -> None:
    """tors's own canonical property, over EVERY generated single-run-replace
    pair (no assume): exactly one replace op (plus optional equals), whose
    i-span content is exactly the old run and whose j-span content is
    exactly the new run. Verified before pinning over exhaustive
    deterministic sweeps (2.4M pairs over 3-char contexts up to length 3,
    old/new runs up to length 3): tors never splits a single-run replace."""
    a, b, old_run, new_run = case
    ops = tors.diff_opcodes(a, b)
    assert_opcodes_are_valid(a, b, ops)
    op = _sole_non_equal_op(ops, "replace")
    assert op is not None, f"tors did not emit a single replace op: {ops!r}"
    _, i1, i2, j1, j2 = op
    assert a[i1:i2] == old_run, f"replace a-span is not the old run: {op!r}"
    assert b[j1:j2] == new_run, f"replace b-span is not the new run: {op!r}"


@given(st.text(max_size=32))
@settings(max_examples=300)
def test_all_equal_pairs_match_difflib_exactly(text: str) -> None:
    """Unambiguous class: identical operands (arbitrary small text, including
    non-ASCII and the empty string), so both sides must emit the single-equal
    list (or the empty list for the empty pair, the pinned degenerate)."""
    assert tors.diff_opcodes(text, text) == _difflib_opcodes(text, text)


@given(st.text(max_size=24), st.text(max_size=24))
@settings(max_examples=500)
def test_opcodes_reconstruct_both_sides_over_arbitrary_pairs(a: str, b: str) -> None:
    """Parity-contract part 1, over ARBITRARY pairs (any alphabets, shared or
    disjoint, empty or not; the generator does not know about the
    unambiguous classes): the opcodes are a valid, difflib-shaped cover of both
    sides. Where the pair falls in an ambiguous region the two algorithms may
    pick different valid scripts; validity, not byte-agreement, is the
    guarantee here, and the class-specific properties above carry the
    agreement side."""
    assert_opcodes_are_valid(a, b, tors.diff_opcodes(a, b))


# --- Wall cells --------------------------------------------------------------------


def _min_wall_ms(op: Callable[[], object], samples: int = 3, warmup: int = 1) -> float:
    """Min-of-``samples`` wall after ``warmup`` runs (the suite's shared
    methodology; ``samples``/``warmup`` are per-call here because the difflib
    side of a race costs seconds per sample)."""
    for _ in range(warmup):
        op()
    best = float("inf")
    for _ in range(samples):
        started = time.perf_counter()
        op()
        best = min(best, time.perf_counter() - started)
    return best * 1000.0


def test_diff_opcodes_beats_difflib_on_the_mutated_prose_corpus_at_256kib() -> None:
    """The wall race, at the size where difflib takes seconds: the six-edit
    mutated-line corpus at 256 KiB. difflib's character-level matcher is
    quadratic in the region between the first and last edit (the scattered
    edits make that region nearly the whole corpus), while tors runs the Myers
    pass natively. Measured on the dev box (ambient load 6.0): tors 3.9 ms
    (min-of-3 after warm-up) vs difflib 3,059 ms (single sample, ~3 s per call;
    see ``_DIFF_WALL_MARGIN`` for why one sample), ratio 0.0013 against
    the 0.25 margin."""
    a, b = diff_pair_near_identical(256 * 1024)
    tors_ms = _min_wall_ms(lambda: tors.diff_opcodes(a, b))
    difflib_ms = _min_wall_ms(
        lambda: difflib.SequenceMatcher(None, a, b).get_opcodes(), samples=1, warmup=0
    )
    assert tors_ms < _DIFF_WALL_MARGIN * difflib_ms, (
        f"256 KiB mutated prose: tors {tors_ms:.1f}ms vs difflib {difflib_ms:.0f}ms "
        f"(ratio {tors_ms / difflib_ms:.4f}): the native pass lost more than the "
        "tolerance margin to the quadratic stdlib matcher"
    )


def test_diff_opcodes_absolute_wall_band_holds_at_1mib() -> None:
    """The 1 MiB cell: an absolute band, not a race: difflib measured 59,001 ms
    (one call, ambient load 5.6) on this corpus shape, and re-running a
    minute-long stdlib call per suite run (minutes on a 2-vCPU CI runner) is
    exactly the large-size difflib discipline this suite keeps; the asserted
    race lives at 256 KiB above, and this cell pins that tors's own wall stays
    in its native band as the corpus grows 4x. Measured 2.0 ms (ambient load
    5.6-6.0) against the 40 ms ceiling (~20x margin)."""
    a, b = diff_pair_near_identical(1 * _MIB)
    tors_ms = _min_wall_ms(lambda: tors.diff_opcodes(a, b))
    assert tors_ms < _DIFF_1MIB_CEILING_MS, (
        f"1 MiB mutated prose: tors took {tors_ms:.1f}ms, outside the absolute "
        f"band (measured ~3.4ms, ceiling {_DIFF_1MIB_CEILING_MS:.0f}ms with ~12x "
        "margin); the native diff pass regressed"
    )


# --- The deadline_ms parameter (bounding the superlinear worst case) -----------------
#
# The bounded Myers search's work on hard inputs grows superlinearly with size
# (the char-shuffled corpus is the worst case; the README's diff
# section records the measured ladder: 50k chars 0.32 s, 200k 3.67 s, 400k
# 13.81 s, 1M 183.6 s on the dev box, ambient load 2.6-3.7, roughly ~n^2).
# ``deadline_ms`` bounds the WHOLE call: on expiry the incomplete result is
# discarded and ``TimeoutError`` is raised naming the elapsed time and the
# deadline. The default (``None``) preserves the current unbounded behavior
# exactly.

# The slow pair for the timeout cell: ~120k chars of char-shuffled prose,
# bracketed by the measured ladder (100k chars 1.06 s, 200k 3.67 s at load
# ~2.7), so its unbounded diff costs ~1.5-2 s, ~30x over the 50 ms deadline
# the cell sets: the cell cannot flake into "finished before the deadline"
# even on a loaded runner, while the deadline itself keeps the cell's wall in
# the tens of milliseconds.
_DEADLINE_PAIR_BYTES = 120_000
_DEADLINE_MS = 50.0


class TestDeadline:
    def test_a_slow_pair_exceeding_the_deadline_raises_timeout_error(self) -> None:
        """The contract: a diff that cannot finish inside ``deadline_ms``
        raises ``TimeoutError`` (the builtins type, pyo3's PyTimeoutError),
        the message naming BOTH the elapsed cost and the deadline, and the
        deadline actually bounds the call: the unbounded diff of this pair
        costs ~1.5 s (measured) while the call returns in well under a
        second."""
        a, b = diff_pair_char_shuffled(_DEADLINE_PAIR_BYTES)
        started = time.perf_counter()
        with pytest.raises(TimeoutError, match="deadline") as excinfo:
            tors.diff_opcodes(a, b, deadline_ms=_DEADLINE_MS)
        wall = time.perf_counter() - started
        message = str(excinfo.value)
        assert type(excinfo.value) is TimeoutError
        # The message names the deadline and the elapsed cost, both in ms.
        assert re.search(rf"\b{_DEADLINE_MS:g}(\.0)?\s*ms\b", message), message
        assert re.search(r"elapsed \d+(\.\d+)?\s*ms", message), message
        # The deadline bounded the call (the unbounded diff is ~30x the
        # budget; generous upper bound for a loaded runner's dispatch).
        assert wall < 1.0, f"deadline-bounded call took {wall:.2f}s"

    def test_deadline_none_is_the_default_and_unchanged(self) -> None:
        """``deadline_ms=None`` (the default) is exactly the previous
        behavior: the unbounded diff, same opcode list; the parameter is
        purely additive."""
        a, b = diff_pair_near_identical(256 * 1024)
        default_ops = tors.diff_opcodes(a, b)
        explicit_none_ops = tors.diff_opcodes(a, b, deadline_ms=None)
        assert default_ops == explicit_none_ops

    def test_a_generous_deadline_yields_the_identical_opcodes(self) -> None:
        """A deadline the search never reaches changes NOTHING: the opcodes
        are byte-identical to the unbounded call (similar's preflight only
        *skips* work when a deadline has already expired (verified in its
        3.2.0 source), so a far-future deadline is the same code path)."""
        a, b = diff_pair_near_identical(256 * 1024)
        assert tors.diff_opcodes(a, b, deadline_ms=60_000.0) == tors.diff_opcodes(a, b)

    def test_identical_inputs_under_a_deadline_still_short_circuit(self) -> None:
        """The equality short-circuit survives the parameter: identical
        operands inside any ordinary budget return the single-equal list (no
        timeout, no search)."""
        a = diff_pair_near_identical(64 * 1024)[0]
        assert tors.diff_opcodes(a, a, deadline_ms=5_000.0) == [
            ("equal", 0, len(a), 0, len(a))
        ]
        assert tors.diff_opcodes("", "", deadline_ms=5_000.0) == []

    @pytest.mark.parametrize("bad", [0.0, -50.0], ids=["zero", "negative"])
    def test_nonpositive_deadlines_raise_value_error(self, bad: float) -> None:
        """A budget must be positive: zero or negative ``deadline_ms`` is a
        caller bug, not an instant timeout; refused with ``ValueError``
        before any work runs (the closed-set-of-strings convention of the
        suite's other keyword parameters)."""
        with pytest.raises(ValueError, match="deadline_ms"):
            tors.diff_opcodes("abc", "abd", deadline_ms=bad)

    def test_a_non_numeric_deadline_raises_type_error(self) -> None:
        """The parameter is ``float | None``: anything non-numeric is refused
        with ``TypeError`` by the argument boundary."""
        with pytest.raises(TypeError):
            tors.diff_opcodes("abc", "abd", deadline_ms="50")  # type: ignore[arg-type]
