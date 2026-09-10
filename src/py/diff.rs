use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::types::PyString;
use pyo3::{Py, PyAny};

use crate::diff_impl;
use crate::py::_borrow::timeout_err;
use crate::validate_deadline_ms;

/// The shared marshalling tail of `tors.diff_opcodes` and
/// `tors.diff_opcodes_lines`: one 5-tuple per opcode with up to four fresh
/// `PyLong`s, constructed under the GIL: O(ops), the `word_bounds`
/// list-shape class. The four tag strings are interned once per call and
/// shared by reference into every tuple: `op[0] is "equal"` holds exactly as
/// it does for difflib's own tuples (which all carry the same interned
/// literals), and constructing a fresh `PyString` per op would multiply the
/// band several-fold.
fn opcodes_into_pytuples(py: Python<'_>, opcodes: Vec<diff_impl::Opcode>) -> PyResult<Py<PyAny>> {
    let tags = [
        PyString::intern(py, "equal"),
        PyString::intern(py, "replace"),
        PyString::intern(py, "delete"),
        PyString::intern(py, "insert"),
    ];
    let tuples = opcodes
        .into_iter()
        .map(|op| {
            let tag = match op.tag {
                diff_impl::OpcodeTag::Equal => &tags[0],
                diff_impl::OpcodeTag::Replace => &tags[1],
                diff_impl::OpcodeTag::Delete => &tags[2],
                diff_impl::OpcodeTag::Insert => &tags[3],
            };
            (tag.clone(), op.i1, op.i2, op.j1, op.j2)
        })
        .collect::<Vec<_>>();
    Ok(PyList::new(py, tuples)?.into_any().unbind())
}

/// `tors.diff_opcodes(a, b, *, deadline_ms=None)`: `difflib.SequenceMatcher(None, a, b)
/// .get_opcodes()`'s shape at native speed: `(tag, i1, i2, j1, j2)` tuples,
/// `tag` in {`"equal"`, `"replace"`, `"delete"`, `"insert"`}, ranges
/// monotone/contiguous/covering both sides, adjacent delete+insert merged
/// into `"replace"` exactly as difflib presents it. Character-level: a
/// `str` is a character sequence to `SequenceMatcher`, which is what makes
/// difflib the parity oracle; callers wanting line-level diffs split the
/// operands themselves. The engine is `similar`'s Myers; see
/// `src/diff_impl.rs` for the validity-first parity contract: exact
/// agreement with difflib is pinned only on the unambiguous classes (pure
/// insert/delete with differing flanks, all-equal, single-run replace,
/// empty operands), with documented+tested boundary cases where the two
/// algorithms pick different valid alignments (an insertion split around
/// difflib's anchored match vs one contiguous insertion; an equal-run
/// boundary sliding across a repeated flank).
///
/// `deadline_ms` (default `None`, the previous behavior exactly) bounds the
/// whole call: the Myers search's work on hard inputs (few anchorable unique
/// records, where a character-level permutation is the measured shape)
/// grows superlinearly with size, and an unbounded 1M-char pair of that
/// shape measured ~145 s on the dev box (docs/performance.md records
/// the ladder). On expiry the incomplete result is discarded and
/// `TimeoutError` is raised naming the elapsed cost and the deadline.
/// similar's own deadline mechanism makes the search bail at the budget (it
/// has no error surface of its own, so the expiry verdict is ours), and the
/// exception is constructed after the GIL is reacquired, so nothing raises
/// from inside the detached region. A non-positive or non-finite
/// `deadline_ms` is refused with `ValueError` before any work runs.
///
/// GIL model: the whole diff, meaning the `Vec<char>` materialization of both
/// operands and the Myers search over them, runs under `py.detach`; the
/// GIL-held residue is the two str-in argument borrows (each the standard
/// zero-copy alias for ASCII/cached inputs, or the one-time O(input) UTF-8
/// materialization on the first non-ASCII call) plus the return
/// marshalling: one 5-tuple with up to four fresh `PyLong`s per opcode,
/// O(ops), the `word_bounds` list-shape class. The four tag strings are
/// interned once per call and shared by reference into every tuple, so
/// `op[0] is "equal"` holds exactly as it does for difflib's own tuples,
/// and constructing a fresh `PyString` per op would multiply the marshalling
/// cost several-fold. The measured band lives in the crate GIL model above
/// and is pinned by `tests/test_gil_release.py` (the 12 MiB shuffled pair,
/// the many-opcode shape that makes the O(ops) cost visible).
#[pyfunction(signature = (a, b, *, deadline_ms = None))]
pub fn diff_opcodes(
    py: Python<'_>,
    a: &str,
    b: &str,
    deadline_ms: Option<f64>,
) -> PyResult<Py<PyAny>> {
    validate_deadline_ms(deadline_ms)?;
    let opcodes = py
        .detach(|| diff_impl::diff_opcodes_deadline(a, b, deadline_ms))
        .map_err(|err| timeout_err(err.message()))?;
    opcodes_into_pytuples(py, opcodes)
}

/// `tors.diff_opcodes_lines(a, b, *, deadline_ms=None)`: the line-level
/// spelling of `tors.diff_opcodes` (v0.8): the same `(tag, i1, i2, j1, j2)`
/// opcode shape, but the operands are tokenized as lines (each line keeps
/// its `\n`, the last may lack one; `split_keepend_lines` in
/// `src/diff_impl.rs`) and the indices address lines, so `a_lines[i1:i2]`
/// slicing reconstructs, where `a_lines` is `a` split on `'\n'` with each
/// piece's terminator reattached (`re.split(r"(?<=\n)", a)`, dropping a
/// trailing empty string if `a` ends in `'\n'`). This is not
/// `a.splitlines(keepends=True)`, which also breaks on `\r`, `\v`, `\f`,
/// `\x1c`-`\x1e`, `\x85`, and the Unicode line/paragraph separators: text
/// using one of those as its only line terminator tokenizes as one line
/// here where `str.splitlines()` would see several. This is a
/// narrower-than-splitlines contract, not an oversight. This is the
/// document/version-diff shape: doing the split in Rust removes the Python round-trip, and a
/// line diff has far fewer tokens than its char-level twin on the same
/// corpus, so its walls sit proportionally lower. Same engine (`similar`'s
/// Myers), same validity contract, same boundary-class divergences from
/// difflib as the char-level spelling, same `deadline_ms` machinery (bounds
/// the whole call; `TimeoutError` on expiry, constructed after the GIL is
/// reacquired).
///
/// GIL model: the `word_bounds`/`diff_opcodes` classes, meaning the two str-in
/// argument borrows, the whole line split + Myers search under `py.detach`,
/// and the O(line-opcodes) return marshalling (one interned-tag 5-tuple per
/// opcode).
#[pyfunction(signature = (a, b, *, deadline_ms = None))]
pub fn diff_opcodes_lines(
    py: Python<'_>,
    a: &str,
    b: &str,
    deadline_ms: Option<f64>,
) -> PyResult<Py<PyAny>> {
    validate_deadline_ms(deadline_ms)?;
    let opcodes = py
        .detach(|| diff_impl::diff_opcodes_lines_deadline(a, b, deadline_ms))
        .map_err(|err| timeout_err(err.message()))?;
    opcodes_into_pytuples(py, opcodes)
}
